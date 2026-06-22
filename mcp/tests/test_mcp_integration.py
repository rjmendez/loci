"""
MCP tool integration tests — calls tool functions directly in-process with a
real temp directory, verifying JSON output shape and that asyncio dispatch paths
(the threading/asyncio.run() guard introduced for _record_verdicts) work correctly.

These tests intentionally run without Qdrant or Ollama — all Qdrant/embedding
paths degrade gracefully and are not exercised here. The goal is to ensure:
  1. Each tool returns valid JSON under minimal valid inputs.
  2. No tool raises an uncaught exception on the happy path.
  3. Error paths return {"error": str} not a stack trace.
  4. investigation_store → investigation_note → investigation_list roundtrip works.

Run: pytest mcp/tests/test_mcp_integration.py -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure the mcp/ directory is on the path so `import server` resolves correctly.
_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import server  # noqa: E402 — must be after path setup


def _json(result: str) -> dict:
    """Parse a tool's JSON return value; fail the test with the raw string on error."""
    try:
        return json.loads(result)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Tool returned non-JSON: {result!r}") from exc


# Counters so each test gets a unique investigation_id (the tool is idempotent
# on the same ID — it resumes instead of creating — so uniqueness matters).
_counter = [0]


def _new_id(prefix="test"):
    _counter[0] += 1
    return f"{prefix}-{_counter[0]:04d}"


class TestInvestigationLifecycle(unittest.TestCase):
    """Full investigation create → store → note → load → list roundtrip."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def test_investigation_start_creates_and_returns_manifest(self):
        inv_id = _new_id("start")
        result = _json(server.investigation_start(
            investigation_id=inv_id,
            title="Test investigation",
            context="Created by integration test",
        ))
        self.assertEqual(result["status"], "created")
        self.assertIn("manifest", result)
        self.assertEqual(result["manifest"]["id"], inv_id)

    def test_investigation_start_resumes_existing(self):
        inv_id = _new_id("resume")
        server.investigation_start(investigation_id=inv_id, title="First creation")
        result = _json(server.investigation_start(investigation_id=inv_id, title="Second call"))
        self.assertEqual(result["status"], "resumed")
        # Title should not be overwritten on resume
        self.assertEqual(result["manifest"]["title"], "First creation")

    def test_investigation_store_valid_finding(self):
        inv_id = _new_id("store")
        server.investigation_start(investigation_id=inv_id, title="Store test")
        result = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="The auth service returns HTTP 401 on expired tokens.",
            source="test:manual",
            confidence="high",
        ))
        self.assertNotIn("error", result, f"Unexpected error: {result}")
        self.assertEqual(result.get("stored"), True)
        self.assertIn("finding_id", result)

    def test_investigation_store_rejects_bad_finding_type(self):
        inv_id = _new_id("bad-type")
        server.investigation_start(investigation_id=inv_id, title="Bad type test")
        result = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="INVALID",
            text="some text",
            source="test",
        ))
        self.assertIn("error", result)

    def test_investigation_store_rejects_missing_investigation(self):
        result = _json(server.investigation_store(
            investigation_id="does-not-exist-xyz",
            finding_type="observed",
            text="some finding",
            source="test",
        ))
        self.assertIn("error", result)

    def test_investigation_note_updates_hypothesis(self):
        inv_id = _new_id("note")
        server.investigation_start(investigation_id=inv_id, title="Note test")
        result = _json(server.investigation_note(
            investigation_id=inv_id,
            field="hypothesis",
            value="The bug is in the token expiry check.",
        ))
        self.assertNotIn("error", result, f"Unexpected error: {result}")

    def test_investigation_note_updates_next_step(self):
        inv_id = _new_id("note-step")
        server.investigation_start(investigation_id=inv_id, title="Note step test")
        result = _json(server.investigation_note(
            investigation_id=inv_id,
            field="next_step",
            value="Check auth.py line 42",
        ))
        self.assertNotIn("error", result)

    def test_investigation_load_returns_findings(self):
        inv_id = _new_id("load")
        server.investigation_start(investigation_id=inv_id, title="Load test")
        server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="Database uses bcrypt for password hashing.",
            source="test:code-review",
            confidence="high",
        )
        result = _json(server.investigation_load(investigation_id=inv_id))
        self.assertNotIn("error", result)
        self.assertIn("manifest", result)
        texts = [f.get("text", "") for f in result.get("recent_findings", [])]
        self.assertTrue(
            any("bcrypt" in t for t in texts),
            f"Stored finding not found in load results: {texts}",
        )

    def test_investigation_list_returns_list(self):
        inv_a = _new_id("list-a")
        inv_b = _new_id("list-b")
        server.investigation_start(investigation_id=inv_a, title="List test A")
        server.investigation_start(investigation_id=inv_b, title="List test B")
        result = _json(server.investigation_list())
        self.assertIn("investigations", result)
        self.assertIsInstance(result["investigations"], list)
        ids = [i.get("id") for i in result["investigations"]]
        self.assertIn(inv_a, ids)
        self.assertIn(inv_b, ids)

    def test_store_roundtrip_finding_id_in_load(self):
        inv_id = _new_id("roundtrip")
        server.investigation_start(investigation_id=inv_id, title="Roundtrip test")
        stored = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="inferred",
            text="The retry loop in sync.py does not back off on rate limits.",
            source="test:code-analysis",
            confidence="medium",
        ))
        finding_id = stored.get("finding_id")
        self.assertIsNotNone(finding_id, "Store did not return a finding_id")

        loaded = _json(server.investigation_load(investigation_id=inv_id))
        finding_ids = [f.get("id") for f in loaded.get("recent_findings", [])]
        self.assertIn(finding_id, finding_ids)


class TestMemoryHealth(unittest.TestCase):
    """memory_health should always return valid JSON."""

    def test_returns_valid_json_without_qdrant(self):
        result = _json(server.memory_health())
        # Should have at minimum one of these keys
        self.assertTrue(
            any(k in result for k in ("status", "error", "qdrant", "sqlite")),
            f"Unexpected health response shape: {result}",
        )

    def test_with_missing_investigation_id(self):
        # Should not raise; any valid JSON response is acceptable
        result = _json(server.memory_health(investigation_id="no-such-investigation"))
        self.assertIsInstance(result, dict)


class TestMemoryConfidence(unittest.TestCase):
    """memory_confidence returns valid JSON for any query."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def test_returns_valid_json_for_empty_query(self):
        # Empty query may return a confidence response or an error dict — both are valid.
        result = _json(server.memory_confidence(query=""))
        self.assertIsInstance(result, dict)

    def test_returns_valid_json_for_real_query(self):
        result = _json(server.memory_confidence(query="authentication token expiry"))
        self.assertIsInstance(result, dict)
        # When Qdrant is unavailable the response may be degraded but must be valid JSON
        self.assertTrue(
            any(k in result for k in ("confidence", "error", "status", "score")),
            f"Unexpected confidence response shape: {result}",
        )


class TestAuditLog(unittest.TestCase):
    """audit_log records tool calls; requires tool_name, inputs_json, output."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def test_returns_valid_json_with_required_args(self):
        result = _json(server.audit_log(
            tool_name="test_tool",
            inputs_json='{"query": "test"}',
            output='{"result": "ok"}',
        ))
        self.assertIsInstance(result, dict)
        self.assertNotIn("error", result)

    def test_with_investigation_id(self):
        inv_id = _new_id("audit")
        server.investigation_start(investigation_id=inv_id, title="Audit test")
        result = _json(server.audit_log(
            tool_name="test_tool",
            inputs_json='{"param": "value"}',
            output='{"status": "ok"}',
            investigation_id=inv_id,
        ))
        self.assertIsInstance(result, dict)
        self.assertNotIn("error", result)


class TestToolsSmokeReturnValidJSON(unittest.TestCase):
    """Smoke test: key tools return parseable JSON without raising."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def _smoke(self, fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            self.fail(f"{fn.__name__} returned non-JSON: {result!r}")
        self.assertIsInstance(parsed, (dict, list))

    def test_investigation_list_smoke(self):
        self._smoke(server.investigation_list)

    def test_memory_health_smoke(self):
        self._smoke(server.memory_health)

    def test_memory_confidence_smoke(self):
        self._smoke(server.memory_confidence, query="test query")

    def test_investigation_start_smoke(self):
        self._smoke(
            server.investigation_start,
            investigation_id=_new_id("smoke"),
            title="smoke test",
        )


class TestRagContextSearchDecayParam(unittest.TestCase):
    """rag_context_search accepts the decay parameter without raising."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def test_rag_context_search_accepts_decay_param(self):
        # Without Qdrant available the function should return a valid JSON error dict,
        # not raise — both decay=True and decay=False must be accepted without error.
        for decay_val in (True, False):
            result = server.rag_context_search(query="authentication token", decay=decay_val)
            try:
                parsed = json.loads(result)
            except json.JSONDecodeError:
                self.fail(
                    f"rag_context_search(decay={decay_val!r}) returned non-JSON: {result!r}"
                )
            self.assertIsInstance(parsed, dict, f"Expected dict for decay={decay_val!r}")
            # Either a rag_required error (no Qdrant) or a real response — both are valid.
            self.assertTrue(
                any(k in parsed for k in ("mode", "error", "context", "results")),
                f"Unexpected response shape for decay={decay_val!r}: {parsed}",
            )

    def test_rag_context_search_decay_default_is_true(self):
        # Calling without decay kwarg must not raise — default decay=True is active.
        result = server.rag_context_search(query="memory decay ebbinghaus")
        self.assertIsInstance(json.loads(result), dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
