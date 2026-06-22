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
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


class TestInvestigationLifecycle(unittest.TestCase):
    """Full investigation create → store → note → list roundtrip."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # Point the server at a temp MEMORY_DIR so tests don't touch real data.
        self._orig_memory_dir = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig_memory_dir
        self._tmp.cleanup()

    def test_investigation_start_returns_id(self):
        result = _json(server.investigation_start(
            task="Test task",
            context="Test context",
            hypothesis="Test hypothesis",
        ))
        self.assertIn("investigation_id", result)
        self.assertIsInstance(result["investigation_id"], str)
        self.assertGreater(len(result["investigation_id"]), 0)

    def test_investigation_store_valid_finding(self):
        start = _json(server.investigation_start(task="Store test"))
        inv_id = start["investigation_id"]
        result = _json(server.investigation_store(
            investigation_id=inv_id,
            text="Found that the authentication service returns 401 on expired tokens.",
            record_type="observation",
            confidence="high",
            source="test",
        ))
        self.assertNotIn("error", result, f"Unexpected error: {result}")
        self.assertIn("id", result)
        self.assertEqual(result.get("stored"), True)

    def test_investigation_note_updates_manifest(self):
        start = _json(server.investigation_start(task="Note test"))
        inv_id = start["investigation_id"]
        result = _json(server.investigation_note(
            investigation_id=inv_id,
            next_step="Check the token expiry logic in auth.py",
        ))
        self.assertNotIn("error", result, f"Unexpected error: {result}")

    def test_investigation_list_returns_list(self):
        # Create two investigations
        server.investigation_start(task="Inv A")
        server.investigation_start(task="Inv B")
        result = _json(server.investigation_list())
        self.assertIn("investigations", result)
        self.assertIsInstance(result["investigations"], list)
        self.assertGreaterEqual(len(result["investigations"]), 2)

    def test_investigation_store_missing_id_returns_error(self):
        result = _json(server.investigation_store(
            investigation_id="nonexistent-id-xyz",
            text="some finding",
        ))
        self.assertIn("error", result)

    def test_investigation_store_roundtrip_then_load(self):
        start = _json(server.investigation_start(task="Roundtrip test"))
        inv_id = start["investigation_id"]
        store_result = _json(server.investigation_store(
            investigation_id=inv_id,
            text="The database uses bcrypt for password hashing.",
            record_type="finding",
            confidence="high",
        ))
        finding_id = store_result.get("id")
        self.assertIsNotNone(finding_id, "Store did not return an id")

        loaded = _json(server.investigation_load(investigation_id=inv_id))
        self.assertNotIn("error", loaded)
        texts = [f.get("text", "") for f in loaded.get("findings", [])]
        self.assertTrue(
            any("bcrypt" in t for t in texts),
            f"Stored finding not found in load results: {texts}",
        )


class TestMemoryHealth(unittest.TestCase):
    """memory_health should always return valid JSON."""

    def test_returns_valid_json_without_qdrant(self):
        result = _json(server.memory_health())
        # Should have at minimum a status or error key
        self.assertTrue(
            "status" in result or "error" in result or "qdrant" in result,
            f"Unexpected health response shape: {result}",
        )

    def test_with_investigation_id_not_found(self):
        result = _json(server.memory_health(investigation_id="no-such-investigation"))
        # Should not raise; should return gracefully
        self.assertIsInstance(result, dict)


class TestMemoryConfidence(unittest.TestCase):
    """memory_confidence should return valid JSON and handle empty query gracefully."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def test_empty_query_returns_error(self):
        result = _json(server.memory_confidence(query=""))
        self.assertIn("error", result)

    def test_valid_query_returns_json(self):
        result = _json(server.memory_confidence(query="authentication"))
        self.assertIsInstance(result, dict)


class TestAuditLog(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def test_returns_valid_json(self):
        result = _json(server.audit_log())
        self.assertIsInstance(result, dict)

    def test_with_investigation_id(self):
        # Start an investigation then read its audit log
        start = _json(server.investigation_start(task="Audit test"))
        inv_id = start["investigation_id"]
        result = _json(server.audit_log(investigation_id=inv_id))
        self.assertIsInstance(result, dict)
        self.assertNotIn("error", result)


class TestToolsReturnValidJSON(unittest.TestCase):
    """Smoke test: every listed tool returns parseable JSON without crashing."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def _smoke(self, fn, *args, **kwargs):
        try:
            result = fn(*args, **kwargs)
            parsed = json.loads(result)
            self.assertIsInstance(parsed, (dict, list))
        except json.JSONDecodeError:
            self.fail(f"{fn.__name__} returned non-JSON: {result!r}")

    def test_investigation_list_smoke(self):
        self._smoke(server.investigation_list)

    def test_memory_health_smoke(self):
        self._smoke(server.memory_health)

    def test_memory_confidence_empty_query(self):
        # Known error path — still must be valid JSON
        self._smoke(server.memory_confidence, query="")

    def test_investigation_start_smoke(self):
        self._smoke(server.investigation_start, task="smoke test")


if __name__ == "__main__":
    unittest.main(verbosity=2)
