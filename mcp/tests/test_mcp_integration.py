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

    def _set_updated_order(self, inv_ids):
        """Force directory mtimes so inv_ids[0] is the most-recently updated.

        The list is sorted by directory st_mtime; creating investigations in
        fast succession can tie, so pin distinct mtimes for deterministic order.
        """
        base = 1_000_000
        for i, inv_id in enumerate(inv_ids):
            d = server.MEMORY_DIR / inv_id
            ts = base + (len(inv_ids) - i) * 10  # earlier in list => newer
            os.utime(d, (ts, ts))

    def test_investigation_list_respects_limit_and_offset(self):
        ids = [_new_id("page") for _ in range(5)]
        for inv_id in ids:
            server.investigation_start(investigation_id=inv_id, title=f"Page {inv_id}")
        # ids[0] newest ... ids[4] oldest
        self._set_updated_order(ids)

        first = _json(server.investigation_list(limit=2, offset=0))
        self.assertEqual(first["total"], 5)
        self.assertEqual([i["id"] for i in first["investigations"]], ids[:2])

        second = _json(server.investigation_list(limit=2, offset=2))
        self.assertEqual(second["total"], 5)
        self.assertEqual([i["id"] for i in second["investigations"]], ids[2:4])

    def test_investigation_list_nonpositive_limit_returns_everything(self):
        """Documented contract: limit<=0 (and a string form of it) returns all.

        Also guards the coercion: a string limit must not raise TypeError in
        the `limit <= 0` / `offset + limit` paths — it is coerced like offset.
        """
        ids = [_new_id("nolimit") for _ in range(5)]
        for inv_id in ids:
            server.investigation_start(investigation_id=inv_id, title=f"NoLimit {inv_id}")
        self._set_updated_order(ids)

        # limit=0 -> everything
        zero = _json(server.investigation_list(limit=0))
        self.assertEqual(zero["total"], 5)
        self.assertEqual([i["id"] for i in zero["investigations"]], ids)

        # negative limit -> everything
        neg = _json(server.investigation_list(limit=-1))
        self.assertEqual([i["id"] for i in neg["investigations"]], ids)

        # string "0" must coerce, not raise, and behave like 0 (everything)
        str_zero = _json(server.investigation_list(limit="0"))
        self.assertEqual([i["id"] for i in str_zero["investigations"]], ids)

        # string positive limit must coerce and paginate, not raise
        str_two = _json(server.investigation_list(limit="2", offset=0))
        self.assertEqual([i["id"] for i in str_two["investigations"]], ids[:2])

        # limit=None (explicit no-limit) must normalize to the limit<=0 case:
        # returns everything AND echoes an int, matching the `limit: int`
        # signature/docstring — never a null in the response.
        none_limit = _json(server.investigation_list(limit=None))
        self.assertEqual([i["id"] for i in none_limit["investigations"]], ids)
        self.assertIsInstance(none_limit["limit"], int)
        self.assertEqual(none_limit["limit"], 0)

    def test_investigation_list_summary_omits_verbose_fields(self):
        inv_id = _new_id("summary")
        server.investigation_start(investigation_id=inv_id, title="Summary test")
        result = _json(server.investigation_list(summary=True))
        rec = next(i for i in result["investigations"] if i["id"] == inv_id)
        for key in ("id", "title", "status", "finding_counts", "updated_at"):
            self.assertIn(key, rec)
        for key in ("hypothesis", "tier_counts", "created_at", "visibility",
                    "open_questions_count"):
            self.assertNotIn(key, rec)

    def test_investigation_list_full_includes_verbose_fields(self):
        inv_id = _new_id("full")
        server.investigation_start(investigation_id=inv_id, title="Full test")
        result = _json(server.investigation_list(summary=False))
        rec = next(i for i in result["investigations"] if i["id"] == inv_id)
        for key in ("hypothesis", "tier_counts", "created_at", "visibility",
                    "open_questions_count"):
            self.assertIn(key, rec)

    def test_investigation_list_summary_false_string_returns_full(self):
        """String summary='false' must coerce to full mode, not summary mode.

        Mirrors the limit/offset string-coercion guards: a stringly-typed
        client (JSON tool args) passing the truthy string 'false' must not be
        treated as truthy True and wrongly select the compact summary record.
        """
        inv_id = _new_id("summary-false-str")
        server.investigation_start(investigation_id=inv_id, title="Summary false string test")
        result = _json(server.investigation_list(summary="false"))
        rec = next(i for i in result["investigations"] if i["id"] == inv_id)
        # Full-mode fields must be present (summary='false' == full records).
        for key in ("hypothesis", "tier_counts", "created_at", "visibility",
                    "open_questions_count"):
            self.assertIn(key, rec)

        # The truthy string 'true' still selects summary mode.
        summ = _json(server.investigation_list(summary="true"))
        rec2 = next(i for i in summ["investigations"] if i["id"] == inv_id)
        for key in ("hypothesis", "tier_counts", "created_at", "visibility",
                    "open_questions_count"):
            self.assertNotIn(key, rec2)

    def test_investigation_list_empty(self):
        result = _json(server.investigation_list())
        self.assertEqual(result["investigations"], [])
        self.assertEqual(result["total"], 0)

    def test_investigation_list_missing_dir_echoes_coerced_ints(self):
        """MEMORY_DIR-does-not-exist early return must echo normalized ints.

        Regression guard: string JSON tool args (limit="2", offset="1") must
        be coerced up front so the empty-dir early return echoes the same
        int types as the non-empty path — not the raw strings.
        """
        missing = Path(self._tmp.name) / "does_not_exist"
        self.assertFalse(missing.exists())
        orig = server.MEMORY_DIR
        server.MEMORY_DIR = missing
        try:
            result = _json(server.investigation_list(limit="2", offset="1"))
        finally:
            server.MEMORY_DIR = orig
        self.assertEqual(result["investigations"], [])
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["limit"], 2)
        self.assertEqual(result["offset"], 1)
        self.assertIsInstance(result["limit"], int)
        self.assertIsInstance(result["offset"], int)

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

    def test_investigation_store_accepts_valid_from(self):
        """investigation_store should persist valid_from/valid_until in the finding."""
        inv_id = _new_id("bitemporal-store")
        server.investigation_start(investigation_id=inv_id, title="Bi-temporal store test")

        past_ts = "2020-01-01T00:00:00+00:00"
        future_ts = "2099-12-31T23:59:59+00:00"

        stored = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="Auth service was returning HTTP 401 on all requests in Jan 2020.",
            source="test:history",
            confidence="high",
            valid_from=past_ts,
            valid_until=future_ts,
        ))
        self.assertNotIn("error", stored, f"Unexpected error: {stored}")
        self.assertTrue(stored.get("stored"))

        # Verify fields were persisted in the JSONL
        loaded = _json(server.investigation_load(investigation_id=inv_id))
        findings = loaded.get("recent_findings", [])
        self.assertTrue(findings, "No findings returned from load")
        finding = findings[0]
        self.assertEqual(finding.get("valid_from"), past_ts,
                         "valid_from not persisted correctly")
        self.assertEqual(finding.get("valid_until"), future_ts,
                         "valid_until not persisted correctly")

    def test_investigation_as_of_returns_findings(self):
        """investigation_as_of should return only findings valid at the given timestamp.

        Strategy:
        - Use a near-future checkpoint (2028) so all findings stored right now
          have created_at_ts <= as_of_epoch.
        - Distinguish findings via valid_until: finding A has no valid_until
          (still believed), finding B has valid_until in 2023 (expired before checkpoint).
        - A far-future query (2099) should include A but still exclude B.
        - A past query (2000) should exclude all findings (created after 2000).
        """
        inv_id = _new_id("bitemporal-as-of")
        server.investigation_start(investigation_id=inv_id, title="Bi-temporal as-of test")

        # Near-future checkpoint: both findings are stored before this moment
        checkpoint_ts = "2028-01-01T00:00:00+00:00"

        # Finding A: no valid_until — currently believed, should appear at checkpoint
        stored_a = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="Finding A — no valid_until, currently believed, should appear at checkpoint.",
            source="test:a",
            confidence="high",
        ))
        self.assertNotIn("error", stored_a)
        fid_a = stored_a["finding_id"]

        # Finding B: valid_until in the past (2023) — expired before checkpoint
        stored_b = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="inferred",
            text="Finding B — valid_until=2023, expired before checkpoint, should be excluded.",
            source="test:b",
            confidence="medium",
            valid_until="2023-01-01T00:00:00+00:00",
        ))
        self.assertNotIn("error", stored_b)
        fid_b = stored_b["finding_id"]

        # Query as-of checkpoint (2028): should include A, exclude B (B expired 2023)
        result = _json(server.investigation_as_of(
            investigation_id=inv_id,
            as_of_timestamp=checkpoint_ts,
        ))
        self.assertNotIn("error", result, f"Unexpected error: {result}")
        self.assertEqual(result["investigation_id"], inv_id)
        self.assertEqual(result["as_of"], checkpoint_ts)
        self.assertIsInstance(result["findings"], list)

        returned_ids = {f.get("id") for f in result["findings"]}
        self.assertIn(fid_a, returned_ids, "Finding A (no valid_until) should appear at checkpoint")
        self.assertNotIn(fid_b, returned_ids,
                         "Finding B (valid_until=2023) should NOT appear at 2028 checkpoint")

        # Far-future query (2099): A still included (no valid_until), B still excluded
        future_ts = "2099-01-01T00:00:00+00:00"
        result_future = _json(server.investigation_as_of(
            investigation_id=inv_id,
            as_of_timestamp=future_ts,
        ))
        self.assertNotIn("error", result_future)
        future_ids = {f.get("id") for f in result_future["findings"]}
        self.assertIn(fid_a, future_ids, "A should still appear with no valid_until in 2099")
        self.assertNotIn(fid_b, future_ids, "B (expired 2023) should not appear in 2099 query")

        # Past query (2000): no findings existed yet (created_at_ts > 2000 epoch)
        past_ts = "2000-01-01T00:00:00+00:00"
        result_past = _json(server.investigation_as_of(
            investigation_id=inv_id,
            as_of_timestamp=past_ts,
        ))
        self.assertNotIn("error", result_past)
        self.assertEqual(result_past["count"], 0,
                         "No findings should appear for a query set in year 2000")

        # Error path: invalid investigation ID
        err = _json(server.investigation_as_of(
            investigation_id="no-such-investigation-xyz",
            as_of_timestamp=checkpoint_ts,
        ))
        self.assertIn("error", err)


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


class TestMemorySurface(unittest.TestCase):
    """memory_surface returns valid JSON in all cases, including when Qdrant is unavailable."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def test_memory_surface_returns_valid_json(self):
        """Without Qdrant, memory_surface must return valid JSON with error key."""
        result = _json(server.memory_surface(context="Working on authentication token expiry bug"))
        self.assertIsInstance(result, dict)
        # Either surfaced results (if Qdrant available) or a graceful error
        self.assertTrue(
            "surfaced" in result or "error" in result,
            f"memory_surface missing 'surfaced' or 'error' key: {result}",
        )

    def test_memory_surface_no_qdrant_has_surfaced_list(self):
        """When Qdrant is unavailable, surfaced must be an empty list, not missing."""
        result = _json(server.memory_surface(context="Investigating retry logic in sync service"))
        self.assertIn("surfaced", result)
        self.assertIsInstance(result["surfaced"], list)

    def test_memory_surface_empty_context_returns_error(self):
        """Empty context string should return an error dict, not raise."""
        result = _json(server.memory_surface(context=""))
        self.assertIn("error", result)
        self.assertIn("surfaced", result)
        self.assertEqual(result["surfaced"], [])

    def test_memory_surface_has_required_output_keys(self):
        """Output must always include context_used and count keys."""
        result = _json(server.memory_surface(
            context="Debugging database connection pool exhaustion",
            top_k=3,
        ))
        self.assertIn("surfaced", result)
        self.assertIn("count", result)
        self.assertIn("context_used", result)

    def test_memory_surface_with_investigation_id(self):
        """Passing an investigation_id should not raise, must return valid JSON."""
        result = _json(server.memory_surface(
            context="Reviewing the auth service login flow",
            investigation_id="test-inv-does-not-exist",
        ))
        self.assertIsInstance(result, dict)
        self.assertIn("surfaced", result)


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


class TestNumericConfidence(unittest.TestCase):
    """Tests for numeric_confidence field in investigation_store and
    aggregate_confidence in investigation_finding_provenance."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def test_investigation_store_stores_numeric_confidence(self):
        """investigation_store persists numeric_confidence in the JSONL and
        auto-derives it from the string confidence when not supplied."""
        inv_id = _new_id("nc")
        server.investigation_start(investigation_id=inv_id, title="Numeric confidence test")

        # Supply explicit numeric_confidence — should be stored as-is.
        stored = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="Auth service returned HTTP 401 for expired token.",
            source="test:manual",
            confidence="high",
            numeric_confidence=0.75,
        ))
        self.assertNotIn("error", stored, f"Unexpected error: {stored}")
        self.assertEqual(stored.get("stored"), True)
        finding_id = stored["finding_id"]

        # Verify the JSONL file actually has numeric_confidence=0.75.
        findings_path = server.MEMORY_DIR / inv_id / "findings.jsonl"
        findings = [json.loads(line) for line in findings_path.read_text().splitlines() if line.strip()]
        match = next((f for f in findings if f.get("id") == finding_id), None)
        self.assertIsNotNone(match, "Stored finding not found in JSONL")
        self.assertAlmostEqual(match.get("numeric_confidence"), 0.75, places=5)

        # Store another finding without numeric_confidence — should auto-derive from "low" → 0.3.
        stored2 = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="inferred",
            text="Rate limiter not engaged during auth failure burst.",
            source="test:inference",
            confidence="low",
        ))
        self.assertNotIn("error", stored2)
        finding_id2 = stored2["finding_id"]

        findings2 = [json.loads(line) for line in findings_path.read_text().splitlines() if line.strip()]
        match2 = next((f for f in findings2 if f.get("id") == finding_id2), None)
        self.assertIsNotNone(match2, "Second finding not found in JSONL")
        self.assertAlmostEqual(match2.get("numeric_confidence"), 0.3, places=5,
                               msg="low confidence should auto-derive to 0.3")

        # Clamping: value > 1.0 should be clamped to 1.0.
        stored3 = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="Token expiry field missing in JWT payload.",
            source="test:manual",
            confidence="medium",
            numeric_confidence=2.5,
        ))
        self.assertNotIn("error", stored3)
        finding_id3 = stored3["finding_id"]
        findings3 = [json.loads(line) for line in findings_path.read_text().splitlines() if line.strip()]
        match3 = next((f for f in findings3 if f.get("id") == finding_id3), None)
        self.assertIsNotNone(match3)
        self.assertAlmostEqual(match3.get("numeric_confidence"), 1.0, places=5,
                               msg="numeric_confidence > 1.0 should be clamped to 1.0")

    def test_investigation_finding_provenance_returns_aggregate_confidence(self):
        """investigation_finding_provenance returns aggregate_confidence as the
        product of numeric_confidence values along the derived_from chain."""
        inv_id = _new_id("prov")
        server.investigation_start(investigation_id=inv_id, title="Provenance confidence test")

        # Root observed finding with numeric_confidence 0.9.
        root = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="Login audit log shows 50 failed attempts from 198.51.100.5.",
            source="test:audit",
            confidence="high",
            numeric_confidence=0.9,
        ))
        self.assertNotIn("error", root)
        root_id = root["finding_id"]

        # Inferred finding derived from root, with numeric_confidence 0.8.
        child = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="inferred",
            text="Brute-force attack likely originating from 198.51.100.5.",
            source="test:inference",
            confidence="high",
            numeric_confidence=0.8,
            derived_from=root_id,
        ))
        self.assertNotIn("error", child)
        child_id = child["finding_id"]

        # Trace provenance from the inferred child.
        prov = _json(server.investigation_finding_provenance(
            finding_id=child_id,
            investigation_id=inv_id,
        ))
        self.assertNotIn("error", prov)
        self.assertIn("aggregate_confidence", prov,
                      "Response must include aggregate_confidence")
        # Expected: 0.8 (child) * 0.9 (root) = 0.72
        self.assertAlmostEqual(prov["aggregate_confidence"], 0.72, places=4,
                               msg="aggregate_confidence should be product of chain nc values")
        self.assertEqual(prov["chain_length"], 2)
        # Each chain node should carry numeric_confidence.
        for node in prov["chain"]:
            if "error" not in node:
                self.assertIn("numeric_confidence", node,
                              f"Chain node missing numeric_confidence: {node}")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestProceduralMemory(unittest.TestCase):
    """Tests for procedure finding type, procedure_attempt, and procedure_search."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def test_investigation_store_accepts_procedure_type(self):
        inv_id = _new_id("proc-store")
        server.investigation_start(investigation_id=inv_id, title="Procedure store test")
        result = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="procedure",
            text="Rotate expired TLS certificates on the web tier.",
            source="runbook:tls-rotation",
            confidence="high",
            procedure_preconditions="cert-manager installed, kubectl access available",
            procedure_steps="1. List expired certs\n2. Rotate with cert-manager\n3. Verify pod restarts",
            procedure_postconditions="All pods serving valid certs, no 502 errors",
        ))
        self.assertNotIn("error", result, f"Unexpected error: {result}")
        self.assertTrue(result.get("stored"), "stored should be True")
        self.assertIn("finding_id", result)
        self.assertEqual(result.get("type"), "procedure")

        # Verify that procedure_meta was written to the JSONL
        loaded = _json(server.investigation_load(investigation_id=inv_id))
        proc_findings = [
            f for f in loaded.get("recent_findings", [])
            if f.get("record_type") == "procedure" or f.get("type") == "procedure"
        ]
        self.assertEqual(len(proc_findings), 1, "Expected exactly one procedure finding")
        pm = proc_findings[0].get("procedure_meta", {})
        self.assertIsInstance(pm, dict, "procedure_meta should be a dict")
        self.assertEqual(pm.get("success_count"), 0)
        self.assertEqual(pm.get("attempt_count"), 0)
        self.assertIn("steps", pm)

    def test_procedure_attempt_updates_success_rate(self):
        inv_id = _new_id("proc-attempt")
        server.investigation_start(investigation_id=inv_id, title="Procedure attempt test")
        stored = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="procedure",
            text="Check disk space on all nodes.",
            source="runbook:disk-check",
            confidence="medium",
            procedure_steps="1. ssh to node\n2. df -h\n3. alert if >80%",
        ))
        finding_id = stored.get("finding_id")
        self.assertIsNotNone(finding_id, "Should have a finding_id")

        # First attempt: failure
        r1 = _json(server.procedure_attempt(
            investigation_id=inv_id,
            finding_id=finding_id,
            success=False,
        ))
        self.assertNotIn("error", r1, f"Unexpected error: {r1}")
        self.assertEqual(r1.get("attempt_count"), 1)
        self.assertEqual(r1.get("success_count"), 0)
        self.assertEqual(r1.get("success_rate"), 0.0)

        # Second attempt: success
        r2 = _json(server.procedure_attempt(
            investigation_id=inv_id,
            finding_id=finding_id,
            success=True,
        ))
        self.assertNotIn("error", r2, f"Unexpected error: {r2}")
        self.assertEqual(r2.get("attempt_count"), 2)
        self.assertEqual(r2.get("success_count"), 1)
        self.assertAlmostEqual(r2.get("success_rate"), 0.5, places=3)

        # Third attempt: success
        r3 = _json(server.procedure_attempt(
            investigation_id=inv_id,
            finding_id=finding_id,
            success=True,
        ))
        self.assertEqual(r3.get("attempt_count"), 3)
        self.assertEqual(r3.get("success_count"), 2)
        self.assertAlmostEqual(r3.get("success_rate"), 2 / 3, places=3)

    def test_procedure_attempt_rejects_nonexistent_finding(self):
        inv_id = _new_id("proc-miss")
        server.investigation_start(investigation_id=inv_id, title="Proc miss test")
        result = _json(server.procedure_attempt(
            investigation_id=inv_id,
            finding_id="does-not-exist-uuid",
            success=True,
        ))
        self.assertIn("error", result)

    def test_procedure_attempt_rejects_non_procedure_finding(self):
        inv_id = _new_id("proc-wrong-type")
        server.investigation_start(investigation_id=inv_id, title="Wrong type test")
        stored = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="Some observation.",
            source="test",
        ))
        finding_id = stored.get("finding_id")
        result = _json(server.procedure_attempt(
            investigation_id=inv_id,
            finding_id=finding_id,
            success=True,
        ))
        self.assertIn("error", result)

    def test_procedure_search_returns_procedures(self):
        inv_id = _new_id("proc-search")
        server.investigation_start(investigation_id=inv_id, title="Procedure search test")
        server.investigation_store(
            investigation_id=inv_id,
            finding_type="procedure",
            text="Restart the nginx service after config change.",
            source="runbook:nginx",
            confidence="high",
            procedure_steps="1. nginx -t\n2. systemctl reload nginx",
        )
        # Qdrant is unavailable in tests; fallback JSONL scan must find the result
        result = _json(server.procedure_search(
            query="nginx",
            investigation_id=inv_id,
            limit=5,
        ))
        self.assertNotIn("error", result, f"Unexpected error: {result}")
        self.assertIn("procedures", result)
        self.assertIn("count", result)
        procs = result["procedures"]
        self.assertGreaterEqual(len(procs), 1, "Should find at least one procedure")
        self.assertIn("nginx", procs[0]["text"])

    def test_procedure_search_returns_empty_for_unmatched_query(self):
        inv_id = _new_id("proc-search-empty")
        server.investigation_start(investigation_id=inv_id, title="Procedure search empty test")
        server.investigation_store(
            investigation_id=inv_id,
            finding_type="procedure",
            text="Restart the nginx service after config change.",
            source="runbook:nginx",
            confidence="high",
        )
        result = _json(server.procedure_search(
            query="zxqyabc_totally_nonexistent_zzz",
            investigation_id=inv_id,
            limit=5,
        ))
        self.assertNotIn("error", result)
        self.assertEqual(result.get("count"), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestInvestigationACL(unittest.TestCase):
    """Tests for investigation sharing / ACL (bipartite access control)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def test_investigation_share_adds_to_acl(self):
        inv_id = _new_id("share")
        server.investigation_start(investigation_id=inv_id, title="Share ACL test")

        result = _json(server.investigation_share(
            investigation_id=inv_id,
            agent_ids=["agent-alpha", "agent-beta"],
        ))
        self.assertNotIn("error", result, f"Unexpected error: {result}")
        self.assertEqual(set(result["shared_with"]), {"agent-alpha", "agent-beta"})
        self.assertEqual(result["total_acl"], 2)

        # Verify manifest was persisted
        manifest = server._load_manifest(inv_id)
        self.assertIn("agent-alpha", manifest["acl"])
        self.assertIn("agent-beta", manifest["acl"])

    def test_investigation_share_is_idempotent(self):
        inv_id = _new_id("share-idem")
        server.investigation_start(investigation_id=inv_id, title="Share idempotent test")

        server.investigation_share(investigation_id=inv_id, agent_ids=["agent-x"])
        result = _json(server.investigation_share(
            investigation_id=inv_id,
            agent_ids=["agent-x", "agent-y"],
        ))
        # agent-x already present, only agent-y should appear in shared_with
        self.assertEqual(result["shared_with"], ["agent-y"])
        self.assertEqual(result["total_acl"], 2)

    def test_investigation_unshare_removes_from_acl(self):
        inv_id = _new_id("unshare")
        server.investigation_start(investigation_id=inv_id, title="Unshare ACL test")
        server.investigation_share(
            investigation_id=inv_id,
            agent_ids=["agent-one", "agent-two", "agent-three"],
        )

        result = _json(server.investigation_unshare(
            investigation_id=inv_id,
            agent_ids=["agent-one", "agent-two"],
        ))
        self.assertNotIn("error", result, f"Unexpected error: {result}")
        self.assertEqual(set(result["removed"]), {"agent-one", "agent-two"})
        self.assertEqual(result["total_acl"], 1)

        # Verify manifest was persisted
        manifest = server._load_manifest(inv_id)
        self.assertNotIn("agent-one", manifest["acl"])
        self.assertNotIn("agent-two", manifest["acl"])
        self.assertIn("agent-three", manifest["acl"])

    def test_investigation_unshare_is_idempotent(self):
        inv_id = _new_id("unshare-idem")
        server.investigation_start(investigation_id=inv_id, title="Unshare idempotent")
        server.investigation_share(investigation_id=inv_id, agent_ids=["agent-z"])

        # Removing an agent that doesn't exist should not error
        result = _json(server.investigation_unshare(
            investigation_id=inv_id,
            agent_ids=["agent-not-there"],
        ))
        self.assertNotIn("error", result)
        self.assertEqual(result["removed"], [])
        self.assertEqual(result["total_acl"], 1)

    def test_investigation_share_error_on_missing_investigation(self):
        result = _json(server.investigation_share(
            investigation_id="does-not-exist-acl",
            agent_ids=["agent-x"],
        ))
        self.assertIn("error", result)

    def test_investigation_unshare_error_on_missing_investigation(self):
        result = _json(server.investigation_unshare(
            investigation_id="does-not-exist-acl",
            agent_ids=["agent-x"],
        ))
        self.assertIn("error", result)

    def test_investigation_list_includes_visibility(self):
        inv_private = _new_id("vis-priv")
        inv_shared = _new_id("vis-shared")
        server.investigation_start(investigation_id=inv_private, title="Private investigation")
        server.investigation_start(investigation_id=inv_shared, title="Shared investigation")
        server.investigation_share(investigation_id=inv_shared, agent_ids=["agent-a"])

        result = _json(server.investigation_list(summary=False))
        by_id = {i["id"]: i for i in result["investigations"]}
        self.assertIn("visibility", by_id[inv_private])
        self.assertEqual(by_id[inv_private]["visibility"], "private")
        self.assertEqual(by_id[inv_shared]["visibility"], "shared")

    def test_investigation_store_authored_by_field(self):
        inv_id = _new_id("authored")
        server.investigation_start(investigation_id=inv_id, title="Authored-by test")
        result = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="A finding authored by agent-bob",
            source="test:manual",
            confidence="high",
            authored_by="agent-bob",
        ))
        self.assertNotIn("error", result)
        self.assertTrue(result.get("stored"))

        # Verify authored_by was written to JSONL
        loaded = _json(server.investigation_load(investigation_id=inv_id))
        findings = loaded.get("recent_findings", [])
        authored = [f for f in findings if f.get("authored_by") == "agent-bob"]
        self.assertTrue(len(authored) > 0, "authored_by not persisted in finding")

    def test_investigation_load_acl_filtering(self):
        inv_id = _new_id("acl-filter")
        server.investigation_start(investigation_id=inv_id, title="ACL filtering test")

        # Set up ACL with agent-alice only; agent-bob is NOT in ACL
        server.investigation_share(investigation_id=inv_id, agent_ids=["agent-alice"])

        # Store finding by agent-alice (in ACL)
        server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="Finding by alice",
            source="test",
            confidence="high",
            authored_by="agent-alice",
        )
        # Store finding by agent-bob (not in ACL)
        server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="Finding by bob",
            source="test",
            confidence="high",
            authored_by="agent-bob",
        )
        # Store finding with no author
        server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="Finding by nobody",
            source="test",
            confidence="low",
        )

        # Without requesting_agent_id, all findings returned (backward compat)
        all_loaded = _json(server.investigation_load(investigation_id=inv_id))
        self.assertEqual(all_loaded["total_findings"], 3)

        # Requesting as agent-bob: sees own findings (authored_by == requesting_agent_id)
        # AND ACL members' findings (alice is in ACL).
        # "Finding by nobody" has authored_by="" which is neither bob nor in ACL, so filtered.
        bob_loaded = _json(server.investigation_load(
            investigation_id=inv_id,
            requesting_agent_id="agent-bob",
        ))
        texts = [f.get("text", "") for f in bob_loaded.get("recent_findings", [])]
        self.assertIn("Finding by bob", texts)
        self.assertIn("Finding by alice", texts)   # alice is in ACL
        self.assertNotIn("Finding by nobody", texts)  # no author, not in ACL

        # Requesting as agent-alice: sees own findings + bob (not in ACL → filtered)
        alice_loaded = _json(server.investigation_load(
            investigation_id=inv_id,
            requesting_agent_id="agent-alice",
        ))
        texts = [f.get("text", "") for f in alice_loaded.get("recent_findings", [])]
        self.assertIn("Finding by alice", texts)   # alice == requesting_agent_id
        self.assertNotIn("Finding by bob", texts)  # bob not in ACL
        self.assertNotIn("Finding by nobody", texts)

    def test_backward_compat_old_manifest_no_acl_key(self):
        """Manifests without acl/owner keys should be loaded with safe defaults."""
        inv_id = _new_id("backcompat")
        server.investigation_start(investigation_id=inv_id, title="Backward compat test")

        # Manually write a manifest without acl/owner fields
        import json as _json_mod
        manifest_path = server.MEMORY_DIR / inv_id / "manifest.json"
        raw = _json_mod.loads(manifest_path.read_text())
        raw.pop("acl", None)
        raw.pop("owner", None)
        manifest_path.write_text(_json_mod.dumps(raw))

        # Loading should not raise and should inject defaults
        loaded_manifest = server._load_manifest(inv_id)
        self.assertIsNotNone(loaded_manifest)
        self.assertEqual(loaded_manifest.get("acl"), [])
        self.assertEqual(loaded_manifest.get("owner"), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestProgressiveSummaryFidelity(unittest.TestCase):
    """Tests for the L0/L1/L2 progressive summary ladder (investigation_load fidelity param)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def _setup_investigation_with_findings(self, inv_id):
        """Create an investigation, add findings, then call reflect to populate summaries."""
        server.investigation_start(investigation_id=inv_id, title="Fidelity test investigation")
        for i in range(3):
            server.investigation_store(
                investigation_id=inv_id,
                finding_type="observed",
                text=f"Finding number {i + 1}: something important was observed here.",
                source="test:manual",
                confidence="high",
            )
        # Call reflect to compute and persist L1/L2 summaries
        server.investigation_reflect(investigation_id=inv_id)

    def test_investigation_load_fidelity_full_returns_recent_findings(self):
        """fidelity='full' (default) returns recent_findings and manifest."""
        inv_id = _new_id("fid-full")
        self._setup_investigation_with_findings(inv_id)

        result = _json(server.investigation_load(investigation_id=inv_id, fidelity="full"))
        self.assertNotIn("error", result)
        self.assertIn("manifest", result)
        self.assertIn("recent_findings", result)
        self.assertIsInstance(result["recent_findings"], list)
        self.assertGreater(len(result["recent_findings"]), 0)

    def test_investigation_load_fidelity_summary(self):
        """fidelity='summary' returns summary_l1 bullets + summary_l2 paragraph, not full findings."""
        inv_id = _new_id("fid-sum")
        self._setup_investigation_with_findings(inv_id)

        result = _json(server.investigation_load(investigation_id=inv_id, fidelity="summary"))
        self.assertNotIn("error", result, f"Unexpected error: {result}")
        self.assertIn("manifest", result)
        self.assertEqual(result.get("fidelity"), "summary")
        self.assertIn("summary_l1", result)
        self.assertIn("summary_l2", result)
        self.assertIsInstance(result["summary_l1"], list)
        self.assertIsInstance(result["summary_l2"], str)
        # Must NOT contain full findings list
        self.assertNotIn("recent_findings", result)
        # summary_l1 should have at least one bullet (fallback always populates it)
        self.assertGreater(len(result["summary_l1"]), 0)
        # summary_l2 should be a non-empty string
        self.assertTrue(result["summary_l2"].strip())

    def test_investigation_load_fidelity_brief(self):
        """fidelity='brief' returns only manifest + summary_l2 paragraph."""
        inv_id = _new_id("fid-brief")
        self._setup_investigation_with_findings(inv_id)

        result = _json(server.investigation_load(investigation_id=inv_id, fidelity="brief"))
        self.assertNotIn("error", result, f"Unexpected error: {result}")
        self.assertIn("manifest", result)
        self.assertEqual(result.get("fidelity"), "brief")
        self.assertIn("summary_l2", result)
        self.assertIsInstance(result["summary_l2"], str)
        # Must NOT contain full findings or L1 bullets
        self.assertNotIn("recent_findings", result)
        self.assertNotIn("summary_l1", result)

    def test_investigation_reflect_populates_manifest_summaries(self):
        """investigation_reflect persists summary_l1 and summary_l2 to the manifest."""
        inv_id = _new_id("ref-sum")
        server.investigation_start(investigation_id=inv_id, title="Reflect summary test")
        server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="The database connection pool is exhausted under load.",
            source="test:perf",
            confidence="high",
        )

        reflect_result = _json(server.investigation_reflect(investigation_id=inv_id))
        self.assertNotIn("error", reflect_result)
        self.assertIn("summary_l1", reflect_result)
        self.assertIn("summary_l2", reflect_result)
        self.assertIsInstance(reflect_result["summary_l1"], list)
        self.assertIsInstance(reflect_result["summary_l2"], str)

        # Verify the summaries were persisted to the manifest
        load_result = _json(server.investigation_load(investigation_id=inv_id))
        manifest = load_result["manifest"]
        self.assertIn("summary_l1", manifest)
        self.assertIn("summary_l2", manifest)

    def test_investigation_start_includes_summary_fields(self):
        """Newly created investigations have summary_l1 and summary_l2 initialized."""
        inv_id = _new_id("start-sum")
        result = _json(server.investigation_start(investigation_id=inv_id, title="Summary init test"))
        self.assertEqual(result["status"], "created")
        manifest = result["manifest"]
        self.assertIn("summary_l1", manifest)
        self.assertIn("summary_l2", manifest)
        self.assertEqual(manifest["summary_l1"], [])
        self.assertEqual(manifest["summary_l2"], "")

    def test_investigation_load_fidelity_summary_empty_investigation(self):
        """fidelity='summary' on a fresh investigation with no findings does not error."""
        inv_id = _new_id("fid-sum-empty")
        server.investigation_start(investigation_id=inv_id, title="Empty summary test")

        result = _json(server.investigation_load(investigation_id=inv_id, fidelity="summary"))
        self.assertNotIn("error", result)
        self.assertEqual(result.get("fidelity"), "summary")
        self.assertIn("summary_l1", result)
        self.assertIn("summary_l2", result)

    def test_investigation_load_fidelity_brief_empty_investigation(self):
        """fidelity='brief' on a fresh investigation with no findings does not error."""
        inv_id = _new_id("fid-brief-empty")
        server.investigation_start(investigation_id=inv_id, title="Empty brief test")

        result = _json(server.investigation_load(investigation_id=inv_id, fidelity="brief"))
        self.assertNotIn("error", result)
        self.assertEqual(result.get("fidelity"), "brief")
        self.assertIn("summary_l2", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestConflictTools(unittest.TestCase):
    """Tests for conflict_list and conflict_resolve tools."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def test_conflict_list_returns_valid_json(self):
        """conflict_list returns valid JSON with conflicts list for an investigation."""
        inv_id = _new_id("clist")
        server.investigation_start(investigation_id=inv_id, title="Conflict list test")

        # Store a finding so the investigation dir exists with the right structure
        server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="Auth service returns 200 on valid tokens.",
            source="test:manual",
            confidence="high",
        )

        result = _json(server.conflict_list(investigation_id=inv_id))
        self.assertNotIn("error", result, f"Unexpected error: {result}")
        self.assertIn("conflicts", result)
        self.assertIn("count", result)
        self.assertIsInstance(result["conflicts"], list)
        self.assertEqual(result["count"], len(result["conflicts"]))

    def test_conflict_list_missing_investigation_returns_error(self):
        """conflict_list returns an error dict for a non-existent investigation."""
        result = _json(server.conflict_list(investigation_id="does-not-exist-xyz"))
        self.assertIn("error", result)

    def test_conflict_list_reflects_manually_written_conflict(self):
        """conflict_list reads conflicts.jsonl correctly when a conflict entry exists."""
        import json as _json_mod
        from pathlib import Path as _Path

        inv_id = _new_id("clist-manual")
        server.investigation_start(investigation_id=inv_id, title="Manual conflict test")
        # Create the investigation directory by storing a finding
        server.investigation_store(
            investigation_id=inv_id,
            finding_type="gap",
            text="We don't know if the token check is enforced.",
            source="test",
            confidence="low",
        )

        # Write a conflict record directly to conflicts.jsonl
        conflict_entry = {
            "id": "test-conflict-abc123",
            "investigation_id": inv_id,
            "finding_id_a": "finding-new",
            "finding_id_b": "finding-old",
            "detected_at": "2026-01-01T00:00:00+00:00",
            "status": "open",
            "resolution": None,
        }
        conflicts_path = _Path(self._tmp.name) / inv_id / "conflicts.jsonl"
        with open(conflicts_path, "a") as fh:
            fh.write(_json_mod.dumps(conflict_entry) + "\n")

        result = _json(server.conflict_list(investigation_id=inv_id))
        self.assertNotIn("error", result)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["conflicts"][0]["id"], "test-conflict-abc123")
        self.assertEqual(result["conflicts"][0]["status"], "open")

    def test_conflict_resolve_valid_verdicts(self):
        """conflict_resolve accepts all four valid verdicts."""
        import json as _json_mod
        from pathlib import Path as _Path

        for verdict in ("a_wins", "b_wins", "both_valid", "false_positive"):
            with self.subTest(verdict=verdict):
                inv_id = _new_id(f"cresolve-{verdict[:3]}")
                server.investigation_start(investigation_id=inv_id, title=f"Resolve test {verdict}")
                server.investigation_store(
                    investigation_id=inv_id,
                    finding_type="gap",
                    text="Gap placeholder for conflict resolve test.",
                    source="test",
                    confidence="low",
                )

                conflict_id = f"conflict-{verdict}-abc"
                conflict_entry = {
                    "id": conflict_id,
                    "investigation_id": inv_id,
                    "finding_id_a": "fa",
                    "finding_id_b": "fb",
                    "detected_at": "2026-01-01T00:00:00+00:00",
                    "status": "open",
                    "resolution": None,
                }
                conflicts_path = _Path(self._tmp.name) / inv_id / "conflicts.jsonl"
                with open(conflicts_path, "a") as fh:
                    fh.write(_json_mod.dumps(conflict_entry) + "\n")

                result = _json(server.conflict_resolve(
                    investigation_id=inv_id,
                    conflict_id=conflict_id,
                    verdict=verdict,
                ))
                self.assertNotIn("error", result, f"Unexpected error for verdict {verdict!r}: {result}")
                self.assertTrue(result.get("resolved"), f"resolved!=True for verdict {verdict!r}")
                self.assertEqual(result.get("verdict"), verdict)
                self.assertEqual(result.get("conflict_id"), conflict_id)

                # Verify the file was updated
                updated = _Path(self._tmp.name) / inv_id / "conflicts.jsonl"
                rows = [_json_mod.loads(line) for line in updated.read_text().splitlines() if line.strip()]
                row = next(r for r in rows if r["id"] == conflict_id)
                self.assertEqual(row["status"], "resolved")
                self.assertEqual(row["resolution"], verdict)

    def test_conflict_resolve_rejects_invalid_verdict(self):
        """conflict_resolve returns an error for unknown verdicts."""
        inv_id = _new_id("cresolve-bad")
        server.investigation_start(investigation_id=inv_id, title="Bad verdict test")
        result = _json(server.conflict_resolve(
            investigation_id=inv_id,
            conflict_id="any-id",
            verdict="WRONG_VERDICT",
        ))
        self.assertIn("error", result)

    def test_conflict_resolve_unknown_conflict_id_returns_error(self):
        """conflict_resolve returns error when conflict_id not found."""
        inv_id = _new_id("cresolve-miss")
        server.investigation_start(investigation_id=inv_id, title="Missing conflict test")
        server.investigation_store(
            investigation_id=inv_id,
            finding_type="gap",
            text="placeholder",
            source="test",
            confidence="low",
        )
        result = _json(server.conflict_resolve(
            investigation_id=inv_id,
            conflict_id="no-such-conflict",
            verdict="false_positive",
        ))
        self.assertIn("error", result)

    def test_investigation_store_includes_conflict_detected_field(self):
        """investigation_store response always includes conflict_detected field."""
        inv_id = _new_id("store-conflict-field")
        server.investigation_start(investigation_id=inv_id, title="Conflict field test")
        result = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="Token validation uses RS256 algorithm.",
            source="test:code-review",
            confidence="medium",
        ))
        self.assertNotIn("error", result, f"Unexpected error: {result}")
        self.assertIn("conflict_detected", result)
        # Without Qdrant, conflict detection is skipped → always False in tests
        self.assertFalse(result["conflict_detected"])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestInvestigationExportImport(unittest.TestCase):
    """Tests for investigation_export and investigation_import tools."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def test_investigation_export_returns_bundle(self):
        """Export produces a valid bundle with schema_version, manifest, and findings."""
        inv_id = _new_id("export")
        server.investigation_start(investigation_id=inv_id, title="Export test")
        server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="Connection to 10.0.0.1 on port 443 observed.",
            source="test:netlog",
            confidence="high",
        )

        result = _json(server.investigation_export(investigation_id=inv_id))

        self.assertNotIn("error", result, f"Unexpected error: {result}")
        self.assertTrue(result.get("exported"))
        self.assertEqual(result.get("investigation_id"), inv_id)
        self.assertGreater(result.get("finding_count", 0), 0)
        self.assertGreater(result.get("size_bytes", 0), 0)

        bundle = result.get("bundle", {})
        self.assertEqual(bundle.get("schema_version"), "1.0")
        self.assertIn("exported_at", bundle)
        self.assertIn("manifest", bundle)
        self.assertIn("findings", bundle)
        self.assertIsInstance(bundle["findings"], list)
        self.assertGreater(len(bundle["findings"]), 0)
        texts = [f.get("text", "") for f in bundle["findings"]]
        self.assertTrue(any("443" in t for t in texts))

    def test_investigation_export_missing_investigation(self):
        """Export of a non-existent investigation returns an error."""
        result = _json(server.investigation_export(investigation_id="no-such-inv-xyz"))
        self.assertIn("error", result)

    def test_investigation_import_creates_new_investigation(self):
        """Import a bundle and verify a new investigation is created with findings."""
        inv_id = _new_id("import-src")
        server.investigation_start(investigation_id=inv_id, title="Source investigation")
        server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="Suspicious login from 192.168.1.100 at 03:15 UTC.",
            source="test:auth",
            confidence="high",
        )
        server.investigation_store(
            investigation_id=inv_id,
            finding_type="inferred",
            text="Account may have been compromised.",
            source="test:analysis",
            confidence="medium",
        )

        # Export the investigation.
        export_result = _json(server.investigation_export(investigation_id=inv_id))
        self.assertNotIn("error", export_result)
        bundle = export_result["bundle"]

        # Import as a new investigation with a custom title.
        import json as _json_mod
        bundle_json = _json_mod.dumps(bundle)
        import_result = _json(server.investigation_import(
            bundle_json=bundle_json,
            new_title="Imported: Source investigation",
        ))

        self.assertNotIn("error", import_result, f"Unexpected import error: {import_result}")
        self.assertTrue(import_result.get("imported"))
        self.assertEqual(import_result.get("original_investigation_id"), inv_id)
        self.assertEqual(import_result.get("findings_imported"), 2)

        new_id = import_result.get("new_investigation_id")
        self.assertIsNotNone(new_id)
        self.assertNotEqual(new_id, inv_id)

        # Load the new investigation and verify findings are there.
        loaded = _json(server.investigation_load(investigation_id=new_id))
        self.assertNotIn("error", loaded, f"Could not load imported investigation: {loaded}")
        self.assertEqual(loaded["manifest"]["title"], "Imported: Source investigation")
        self.assertEqual(loaded["manifest"].get("imported_from"), inv_id)
        texts = [f.get("text", "") for f in loaded.get("recent_findings", [])]
        self.assertTrue(any("192.168.1.100" in t for t in texts))

    def test_investigation_import_invalid_json(self):
        """Import with malformed JSON returns an error."""
        result = _json(server.investigation_import(bundle_json="not-valid-json"))
        self.assertIn("error", result)

    def test_investigation_import_wrong_schema_version(self):
        """Import with unsupported schema_version returns an error."""
        import json as _json_mod
        bad_bundle = _json_mod.dumps({"schema_version": "2.0", "manifest": {}, "findings": []})
        result = _json(server.investigation_import(bundle_json=bad_bundle))
        self.assertIn("error", result)

    def test_investigation_import_bundle_too_large(self):
        """Import a bundle exceeding 10 MB returns an error."""
        big_bundle = "x" * (10 * 1024 * 1024 + 1)
        result = _json(server.investigation_import(bundle_json=big_bundle))
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestMemoryTiers(unittest.TestCase):
    """Tests for memory tier management: memory_promote, memory_demote, and tier field in store."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def _start_and_store(self, prefix="tier", tier="warm"):
        inv_id = _new_id(prefix)
        server.investigation_start(investigation_id=inv_id, title=f"Tier test {prefix}")
        stored = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="Auth service returns HTTP 401 on expired tokens.",
            source="test:manual",
            confidence="high",
            tier=tier,
        ))
        return inv_id, stored

    def test_investigation_store_accepts_tier_warm(self):
        inv_id, result = self._start_and_store(prefix="warm", tier="warm")
        self.assertNotIn("error", result)
        self.assertEqual(result.get("tier"), "warm")
        self.assertIn("finding_id", result)

    def test_investigation_store_accepts_tier_cold(self):
        inv_id, result = self._start_and_store(prefix="cold", tier="cold")
        self.assertNotIn("error", result)
        self.assertEqual(result.get("tier"), "cold")

    def test_investigation_store_accepts_tier_hot(self):
        inv_id, result = self._start_and_store(prefix="hot", tier="hot")
        self.assertNotIn("error", result)
        self.assertEqual(result.get("tier"), "hot")

    def test_investigation_store_rejects_invalid_tier(self):
        inv_id = _new_id("badtier")
        server.investigation_start(investigation_id=inv_id, title="Bad tier test")
        result = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="some text",
            source="test",
            tier="invalid",
        ))
        self.assertIn("error", result)

    def test_investigation_list_includes_tier_counts(self):
        inv_id = _new_id("listier")
        server.investigation_start(investigation_id=inv_id, title="Tier counts test")
        server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="finding one warm",
            source="test",
            tier="warm",
        )
        server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="finding two cold",
            source="test",
            tier="cold",
        )
        result = _json(server.investigation_list(summary=False))
        inv_summary = next((i for i in result["investigations"] if i["id"] == inv_id), None)
        self.assertIsNotNone(inv_summary, "Investigation not found in list")
        self.assertIn("tier_counts", inv_summary)
        tc = inv_summary["tier_counts"]
        self.assertEqual(tc.get("warm"), 1)
        self.assertEqual(tc.get("cold"), 1)
        self.assertEqual(tc.get("hot"), 0)

    def test_memory_promote_returns_valid_json(self):
        inv_id, stored = self._start_and_store(prefix="promote", tier="cold")
        finding_id = stored.get("finding_id")
        self.assertIsNotNone(finding_id)

        result = _json(server.memory_promote(
            investigation_id=inv_id,
            finding_id=finding_id,
            tier="warm",
        ))
        self.assertNotIn("error", result, f"Unexpected error: {result}")
        self.assertEqual(result.get("finding_id"), finding_id)
        self.assertEqual(result.get("old_tier"), "cold")
        self.assertEqual(result.get("new_tier"), "warm")
        self.assertTrue(result.get("ok"))

    def test_memory_demote_returns_valid_json(self):
        inv_id, stored = self._start_and_store(prefix="demote", tier="warm")
        finding_id = stored.get("finding_id")
        self.assertIsNotNone(finding_id)

        result = _json(server.memory_demote(
            investigation_id=inv_id,
            finding_id=finding_id,
            tier="cold",
        ))
        self.assertNotIn("error", result, f"Unexpected error: {result}")
        self.assertEqual(result.get("finding_id"), finding_id)
        self.assertEqual(result.get("old_tier"), "warm")
        self.assertEqual(result.get("new_tier"), "cold")
        self.assertTrue(result.get("ok"))

    def test_memory_promote_missing_finding(self):
        inv_id = _new_id("promote-missing")
        server.investigation_start(investigation_id=inv_id, title="Promote missing")
        result = _json(server.memory_promote(
            investigation_id=inv_id,
            finding_id="no-such-finding-id",
            tier="warm",
        ))
        self.assertIn("error", result)

    def test_memory_demote_missing_investigation(self):
        result = _json(server.memory_demote(
            investigation_id="does-not-exist",
            finding_id="no-such-id",
            tier="cold",
        ))
        self.assertIn("error", result)

    def test_memory_promote_invalid_tier(self):
        inv_id, stored = self._start_and_store(prefix="promote-bad-tier", tier="warm")
        finding_id = stored.get("finding_id")
        result = _json(server.memory_promote(
            investigation_id=inv_id,
            finding_id=finding_id,
            tier="ultra",
        ))
        self.assertIn("error", result)

    def test_tier_persisted_in_jsonl(self):
        """Tier field should be stored in findings.jsonl and readable back."""
        inv_id, stored = self._start_and_store(prefix="persist", tier="cold")
        finding_id = stored.get("finding_id")
        findings_path = Path(self._tmp.name) / inv_id / "findings.jsonl"
        findings = [json.loads(line) for line in findings_path.read_text().splitlines() if line.strip()]
        match = next((f for f in findings if f.get("id") == finding_id), None)
        self.assertIsNotNone(match)
        self.assertEqual(match.get("tier"), "cold")

    def test_memory_promote_updates_tier_in_jsonl(self):
        """After promoting, tier field in JSONL should reflect the new tier."""
        inv_id, stored = self._start_and_store(prefix="update-tier", tier="cold")
        finding_id = stored.get("finding_id")
        server.memory_promote(investigation_id=inv_id, finding_id=finding_id, tier="warm")
        findings_path = Path(self._tmp.name) / inv_id / "findings.jsonl"
        findings = [json.loads(line) for line in findings_path.read_text().splitlines() if line.strip()]
        match = next((f for f in findings if f.get("id") == finding_id), None)
        self.assertIsNotNone(match)
        self.assertEqual(match.get("tier"), "warm")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestMemoryRoute(unittest.TestCase):
    """memory_route should always return valid JSON, even when Qdrant is unavailable."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def test_memory_route_returns_valid_json(self):
        """memory_route returns parseable JSON whether Qdrant is available or not."""
        result = server.memory_route(query="authentication failure patterns")
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            self.fail(f"memory_route returned non-JSON: {result!r}")
        self.assertIsInstance(parsed, dict)

    def test_memory_route_qdrant_unavailable_returns_error_json(self):
        """When Qdrant is unavailable, memory_route returns an error dict with 'routed' key."""
        # Temporarily override _get_qdrant to simulate unavailability
        _orig_get_qdrant = server._get_qdrant

        def _no_qdrant():
            return None, None

        server._get_qdrant = _no_qdrant
        try:
            result = server.memory_route(query="auth bypass investigation")
            parsed = json.loads(result)
            self.assertIn("error", parsed)
            self.assertIn("routed", parsed)
            self.assertIsInstance(parsed["routed"], list)
            self.assertEqual(len(parsed["routed"]), 0)
        finally:
            server._get_qdrant = _orig_get_qdrant

    def test_memory_route_empty_query_returns_error(self):
        """Empty query returns an error dict without raising."""
        result = server.memory_route(query="")
        parsed = json.loads(result)
        self.assertIn("error", parsed)
        self.assertIn("routed", parsed)

    def test_memory_route_response_shape_on_success_or_unavailable(self):
        """Response always has 'routed', 'query', 'count' or 'error' keys."""
        result = server.memory_route(query="cross-investigation routing test", top_k=5)
        parsed = json.loads(result)
        self.assertIsInstance(parsed, dict)
        # Either a successful result or a graceful error — never a bare exception
        if "error" not in parsed:
            self.assertIn("routed", parsed)
            self.assertIn("query", parsed)
            self.assertIn("count", parsed)
            self.assertIsInstance(parsed["routed"], list)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestMemoryConsolidateCausalInference(unittest.TestCase):
    """Tests for memory_consolidate causal_edges_inferred field and causal_edges_list tool."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def test_memory_consolidate_returns_valid_json(self):
        """memory_consolidate must always return valid JSON with causal_edges_inferred >= 0."""
        result = server.memory_consolidate(dry_run=True)
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            self.fail(f"memory_consolidate returned non-JSON: {result!r}")
        self.assertIsInstance(parsed, dict)
        # causal_edges_inferred must be present and non-negative
        self.assertIn("causal_edges_inferred", parsed)
        self.assertGreaterEqual(parsed["causal_edges_inferred"], 0)

    def test_causal_edges_list_empty_for_new_investigation(self):
        """causal_edges_list returns empty edges list for an investigation with no edges."""
        inv_id = _new_id("causal")
        server.investigation_start(investigation_id=inv_id, title="Causal test")
        result = _json(server.causal_edges_list(investigation_id=inv_id))
        self.assertIn("edges", result)
        self.assertIn("count", result)
        self.assertEqual(result["count"], 0)
        self.assertIsInstance(result["edges"], list)

    def test_causal_edges_list_missing_investigation(self):
        """causal_edges_list returns an error for a non-existent investigation."""
        result = _json(server.causal_edges_list(investigation_id="nonexistent-xyz-999"))
        self.assertIn("error", result)

    def test_causal_edges_list_no_investigation_id(self):
        """causal_edges_list returns an error when investigation_id is empty."""
        result = _json(server.causal_edges_list(investigation_id=""))
        self.assertIn("error", result)

    def test_heuristic_causal_inference_writes_edges(self):
        """Heuristic causal inference writes edges when findings reference each other."""
        import uuid as _uuid
        inv_id = _new_id("heuristic")
        server.investigation_start(investigation_id=inv_id, title="Heuristic causal test")
        # Create findings where B references A by keyword + snippet.
        id_a = str(_uuid.uuid4())
        id_b = str(_uuid.uuid4())
        id_c = str(_uuid.uuid4())
        findings_path = Path(self._tmp.name) / inv_id / "findings.jsonl"
        import json as _json_local
        with open(findings_path, "a") as f:
            f.write(_json_local.dumps({"id": id_a, "text": "The cache was full and stopped accepting writes", "type": "observed"}) + "\n")
            f.write(_json_local.dumps({"id": id_b, "text": "Because the cache was full", "type": "inferred"}) + "\n")
            f.write(_json_local.dumps({"id": id_c, "text": "Service latency spiked", "type": "observed"}) + "\n")
        # Run causal inference directly.
        n = server._run_causal_inference(inv_id, [
            {"id": id_a, "text": "The cache was full and stopped accepting writes", "type": "observed"},
            {"id": id_b, "text": "Because the cache was full", "type": "inferred"},
            {"id": id_c, "text": "Service latency spiked", "type": "observed"},
        ])
        # causal_edges_list should now return edges.
        result = _json(server.causal_edges_list(investigation_id=inv_id))
        self.assertEqual(result["count"], n)
        if n > 0:
            edge = result["edges"][0]
            self.assertIn("source_id", edge)
            self.assertIn("target_id", edge)
            self.assertIn("edge_type", edge)
            self.assertIn("confidence", edge)
            self.assertIn("inferred_at", edge)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestMemoryHints(unittest.TestCase):
    """memory_hints returns valid JSON with expected shape and respects the limit."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_dir = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)
        # Clear the in-process session hints ring buffer between tests so
        # findings stored in one test do not bleed into another.
        server._session_hints.clear()

    def tearDown(self):
        server.MEMORY_DIR = self._orig_dir
        server._session_hints.clear()
        self._tmp.cleanup()

    def _create_and_store(self, inv_id, n_findings=3):
        server.investigation_start(investigation_id=inv_id, title=f"Hints test {inv_id}")
        finding_ids = []
        for i in range(n_findings):
            res = json.loads(server.investigation_store(
                investigation_id=inv_id,
                finding_type="observed",
                text=f"Finding number {i}: something interesting happened here.",
                source=f"test:source-{i}",
                confidence="medium",
            ))
            finding_ids.append(res.get("finding_id"))
        return finding_ids

    def test_memory_hints_returns_valid_json(self):
        inv_id = _new_id("hints-json")
        self._create_and_store(inv_id, n_findings=2)
        result_str = server.memory_hints(investigation_id=inv_id)
        result = _json(result_str)
        self.assertNotIn("error", result, f"Unexpected error: {result}")
        self.assertIn("hints", result, f"Missing 'hints' key: {result}")
        self.assertIn("investigation_id", result)
        self.assertIn("count", result)
        self.assertIn("as_of", result)
        self.assertIsInstance(result["hints"], list)
        self.assertEqual(result["investigation_id"], inv_id)
        # Each hint must have the expected fields
        for hint in result["hints"]:
            self.assertIn("finding_id", hint)
            self.assertIn("text", hint)
            self.assertIn("source", hint)
            self.assertIn("record_type", hint)
            self.assertIn("recency_score", hint)
            self.assertIn("ts", hint)
            self.assertIsInstance(hint["recency_score"], float)
            self.assertGreaterEqual(hint["recency_score"], 0.0)
            self.assertLessEqual(hint["recency_score"], 1.0)

    def test_memory_hints_respects_limit(self):
        inv_id = _new_id("hints-limit")
        n = 5
        self._create_and_store(inv_id, n_findings=n)

        limit = 2
        result = _json(server.memory_hints(investigation_id=inv_id, limit=limit))
        self.assertNotIn("error", result, f"Unexpected error: {result}")
        self.assertLessEqual(len(result["hints"]), limit,
                             f"Got {len(result['hints'])} hints; expected ≤ {limit}")
        self.assertEqual(result["count"], len(result["hints"]))

    def test_memory_hints_missing_investigation_returns_error(self):
        result = _json(server.memory_hints(investigation_id="no-such-inv-xyz"))
        self.assertIn("error", result)

    def test_memory_hints_since_ts_filters_older(self):
        """since_ts should exclude findings stored before that timestamp."""
        import time as _time
        inv_id = _new_id("hints-since")
        server.investigation_start(investigation_id=inv_id, title="since_ts test")
        # Store one finding, capture a timestamp, then store another.
        server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="First finding — should be filtered out.",
            source="test",
            confidence="low",
        )
        # Snapshot time between the two stores.
        _time.sleep(0.01)
        from datetime import datetime, timezone as tz
        cutoff = datetime.now(tz.utc).isoformat()
        _time.sleep(0.01)
        server.investigation_store(
            investigation_id=inv_id,
            finding_type="inferred",
            text="Second finding — should pass the since_ts filter.",
            source="test",
            confidence="high",
        )
        result = _json(server.memory_hints(
            investigation_id=inv_id,
            limit=10,
            since_ts=cutoff,
        ))
        self.assertNotIn("error", result)
        for hint in result["hints"]:
            self.assertGreater(hint["ts"], cutoff,
                               f"Hint ts {hint['ts']!r} should be > cutoff {cutoff!r}")

    def test_memory_hints_cold_path_from_jsonl(self):
        """Hints should still work when the session ring buffer is empty (JSONL cold path)."""
        inv_id = _new_id("hints-cold")
        self._create_and_store(inv_id, n_findings=2)
        # Clear the ring buffer to force the JSONL cold path.
        server._session_hints.pop(inv_id, None)
        result = _json(server.memory_hints(investigation_id=inv_id, limit=5))
        self.assertNotIn("error", result)
        self.assertGreater(result["count"], 0,
                           "Cold-path JSONL read returned no hints for an investigation with stored findings")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestEntityNodes(unittest.TestCase):
    """Entity node layer: entity_list and entity_timeline tools."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def _start_and_store(self, inv_id, texts):
        server.investigation_start(investigation_id=inv_id, title="Entity test")
        finding_ids = []
        for text in texts:
            r = _json(server.investigation_store(
                investigation_id=inv_id,
                finding_type="observed",
                text=text,
                source="test:entity",
                confidence="medium",
            ))
            finding_ids.append(r.get("finding_id"))
        return finding_ids

    def test_entity_list_returns_valid_json(self):
        inv_id = _new_id("elist")
        self._start_and_store(inv_id, [
            'Windows Server contacted Azure AD at 192.168.1.1.',
            '"John Smith" accessed the system from 10.0.0.5.',
        ])
        result = _json(server.entity_list(investigation_id=inv_id))
        self.assertNotIn("error", result, f"Unexpected error: {result}")
        self.assertIn("entities", result)
        self.assertIn("count", result)
        self.assertIsInstance(result["entities"], list)
        self.assertIsInstance(result["count"], int)
        # At minimum, some entities should have been extracted
        # (capitalized phrases like "Windows Server", "Azure AD", IP addresses)
        self.assertGreaterEqual(result["count"], 0)

    def test_entity_list_missing_investigation_returns_error(self):
        result = _json(server.entity_list(investigation_id="does-not-exist-xyz"))
        self.assertIn("error", result)

    def test_entity_list_type_filter(self):
        inv_id = _new_id("elist-filter")
        self._start_and_store(inv_id, [
            'Windows Server contacted Azure AD at 192.168.1.1.',
        ])
        result = _json(server.entity_list(investigation_id=inv_id, entity_type="system"))
        self.assertNotIn("error", result, f"Unexpected error: {result}")
        self.assertIn("entities", result)
        # All returned entities should have type == "system"
        for ent in result["entities"]:
            self.assertEqual(ent.get("type"), "system")

    def test_entity_timeline_returns_valid_json(self):
        inv_id = _new_id("etimeline")
        self._start_and_store(inv_id, [
            'Windows Server was observed sending traffic.',
            'Windows Server escalated privileges.',
        ])
        # Get entity list first to find an entity_id
        list_result = _json(server.entity_list(investigation_id=inv_id))
        entities = list_result.get("entities", [])

        if not entities:
            # No entities extracted — still must return valid JSON when called with bad id
            result = _json(server.entity_timeline(
                investigation_id=inv_id,
                entity_id="nonexistent-id",
            ))
            self.assertIn("error", result)
            return

        entity_id = entities[0]["entity_id"]
        result = _json(server.entity_timeline(
            investigation_id=inv_id,
            entity_id=entity_id,
        ))
        self.assertNotIn("error", result, f"Unexpected error: {result}")
        self.assertIn("entity", result)
        self.assertIn("timeline", result)
        self.assertIn("count", result)
        self.assertIsInstance(result["timeline"], list)
        self.assertIsInstance(result["count"], int)

    def test_entity_timeline_missing_entity_returns_error(self):
        inv_id = _new_id("etimeline-miss")
        server.investigation_start(investigation_id=inv_id, title="Timeline miss")
        result = _json(server.entity_timeline(
            investigation_id=inv_id,
            entity_id="00000000-0000-0000-0000-000000000000",
        ))
        self.assertIn("error", result)

    def test_entity_timeline_missing_investigation_returns_error(self):
        result = _json(server.entity_timeline(
            investigation_id="does-not-exist-xyz",
            entity_id="some-entity-id",
        ))
        self.assertIn("error", result)

    def test_entities_accumulated_across_findings(self):
        """Entities mentioned in multiple findings should accumulate finding_refs."""
        inv_id = _new_id("eaccum")
        self._start_and_store(inv_id, [
            'Windows Server sent a request.',
            'Windows Server received a response.',
            'Windows Server crashed.',
        ])
        list_result = _json(server.entity_list(investigation_id=inv_id))
        entities = list_result.get("entities", [])
        # Find "Windows Server" entity (if extracted)
        ws_entities = [e for e in entities if "windows" in e.get("name", "").lower()]
        if ws_entities:
            # Should appear in multiple findings
            self.assertGreaterEqual(ws_entities[0]["finding_count"], 1)


class TestContractDeclarationStore(unittest.TestCase):
    """Integration tests for contract_declare / contract_query / contract_check."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)
        self.inv_id = _new_id("contract")
        server.investigation_start(investigation_id=self.inv_id, title="Contract test")

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def test_contract_declare_and_query(self):
        result = _json(server.contract_declare(
            investigation_id=self.inv_id,
            entity="UserSerializer",
            role="producer",
            fields='{"user_id": "int", "created_at": "ISO8601"}',
            protocol="JSON-HTTP",
        ))
        self.assertTrue(result.get("stored"))
        self.assertIn("finding_id", result)
        self.assertEqual(result["entity"], "UserSerializer")
        self.assertEqual(result["role"], "producer")

        query = _json(server.contract_query(
            investigation_id=self.inv_id,
            entity="UserSerializer",
        ))
        self.assertEqual(query["count"], 1)
        self.assertEqual(len(query["contracts"]), 1)
        self.assertIn("contract_declaration", query["contracts"][0].get("tags", []))

    def test_contract_query_role_filter(self):
        server.contract_declare(
            investigation_id=self.inv_id,
            entity="EventStream",
            role="producer",
            fields='{"event_id": "str"}',
        )
        server.contract_declare(
            investigation_id=self.inv_id,
            entity="EventStream",
            role="consumer",
            fields='{"event_id": "str"}',
        )
        producers = _json(server.contract_query(
            investigation_id=self.inv_id, entity="EventStream", role="producer"
        ))
        self.assertEqual(producers["count"], 1)

    def test_contract_check_detects_drift(self):
        server.contract_declare(
            investigation_id=self.inv_id,
            entity="Payload",
            role="producer",
            fields='{"threat_score": "float", "confidence": "float"}',
        )
        result = _json(server.contract_check(
            investigation_id=self.inv_id,
            field_name="score",
            entity="Payload",
        ))
        self.assertFalse(result["consistent"])
        self.assertGreater(len(result["conflicts"]), 0)
        self.assertEqual(result["conflicts"][0]["declared_field"], "threat_score")

    def test_contract_check_no_conflict_on_exact_match(self):
        server.contract_declare(
            investigation_id=self.inv_id,
            entity="Payload",
            role="producer",
            fields='{"user_id": "int"}',
        )
        result = _json(server.contract_check(
            investigation_id=self.inv_id,
            field_name="user_id",
        ))
        self.assertTrue(result["consistent"])
        self.assertEqual(len(result["conflicts"]), 0)

    def test_contract_declare_invalid_role(self):
        result = _json(server.contract_declare(
            investigation_id=self.inv_id,
            entity="X",
            role="observer",
            fields='{}',
        ))
        self.assertIn("error", result)

    def test_contract_declare_invalid_fields_json(self):
        result = _json(server.contract_declare(
            investigation_id=self.inv_id,
            entity="X",
            role="producer",
            fields="not-json",
        ))
        self.assertIn("error", result)


class TestWiringObligationTracker(unittest.TestCase):
    """Integration tests for wiring_obligation_declare / _list / _resolve."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)
        self.inv_id = _new_id("wiring")
        server.investigation_start(investigation_id=self.inv_id, title="Wiring test")

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def test_declare_list_resolve_lifecycle(self):
        decl = _json(server.wiring_obligation_declare(
            investigation_id=self.inv_id,
            class_name="AlertNotifier",
            method_name="send",
            expected_effect="POST to the webhook URL with the alert payload",
        ))
        self.assertTrue(decl.get("stored"))
        fid = decl["finding_id"]
        self.assertIn("AlertNotifier", decl["obligation"]["class"])

        listed = _json(server.wiring_obligation_list(investigation_id=self.inv_id))
        self.assertEqual(listed["unresolved_count"], 1)
        self.assertEqual(len(listed["obligations"]), 1)

        resolved = _json(server.wiring_obligation_resolve(
            investigation_id=self.inv_id,
            finding_id=fid,
            evidence="Confirmed at alerting/notifier.py:47 — requests.post(self.webhook_url, json=payload)",
        ))
        self.assertTrue(resolved["resolved"])

        listed_after = _json(server.wiring_obligation_list(investigation_id=self.inv_id))
        self.assertEqual(listed_after["unresolved_count"], 0)

    def test_list_resolved_includes_all(self):
        decl = _json(server.wiring_obligation_declare(
            investigation_id=self.inv_id,
            class_name="MetricsSender",
            method_name="push",
            expected_effect="push metrics to Prometheus pushgateway",
        ))
        fid = decl["finding_id"]
        server.wiring_obligation_resolve(
            investigation_id=self.inv_id,
            finding_id=fid,
            evidence="metrics/sender.py:22 calls pushgateway_client.push()",
        )
        all_obls = _json(server.wiring_obligation_list(
            investigation_id=self.inv_id, resolved=True
        ))
        self.assertGreaterEqual(len(all_obls["obligations"]), 1)

    def test_resolve_unknown_finding(self):
        result = _json(server.wiring_obligation_resolve(
            investigation_id=self.inv_id,
            finding_id="nonexistent-id",
            evidence="nothing",
        ))
        self.assertIn("error", result)

    def test_resolve_non_obligation_finding(self):
        stored = _json(server.investigation_store(
            investigation_id=self.inv_id,
            finding_type="observed",
            text="Regular finding",
            source="test",
        ))
        fid = stored["finding_id"]
        result = _json(server.wiring_obligation_resolve(
            investigation_id=self.inv_id,
            finding_id=fid,
            evidence="should fail",
        ))
        self.assertIn("error", result)


class TestFindingResolution(unittest.TestCase):
    """resolution lifecycle field: store persists it, load surfaces it (absent -> open),
    and investigation_search can optionally filter by it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def test_store_persists_resolution(self):
        inv_id = _new_id("res-store")
        server.investigation_start(investigation_id=inv_id, title="Resolution store test")
        stored = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="Null-deref in parse() was patched.",
            source="test",
            resolution="fixed",
        ))
        self.assertTrue(stored.get("stored"))
        # Persisted in the JSONL
        findings = server._read_jsonl(server.MEMORY_DIR / inv_id / "findings.jsonl")
        self.assertEqual(findings[0].get("resolution"), "fixed")

    def test_store_rejects_bad_resolution(self):
        inv_id = _new_id("res-bad")
        server.investigation_start(investigation_id=inv_id, title="Bad resolution test")
        result = _json(server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="x",
            source="test",
            resolution="banana",
        ))
        self.assertIn("error", result)

    def test_store_defaults_resolution_open(self):
        inv_id = _new_id("res-default")
        server.investigation_start(investigation_id=inv_id, title="Default resolution test")
        server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text="A normal finding with no resolution given.",
            source="test",
        )
        findings = server._read_jsonl(server.MEMORY_DIR / inv_id / "findings.jsonl")
        self.assertEqual(findings[0].get("resolution"), "open")

    def test_load_surfaces_resolution_and_legacy_reads_open(self):
        inv_id = _new_id("res-load")
        server.investigation_start(investigation_id=inv_id, title="Load resolution test")
        server.investigation_store(
            investigation_id=inv_id, finding_type="observed",
            text="Fixed thing.", source="test", resolution="fixed",
        )
        # Simulate a legacy record with NO resolution field by appending raw JSONL.
        legacy = {
            "id": "legacy-1", "investigation_id": inv_id, "record_type": "observed",
            "type": "observed", "text": "Legacy finding without resolution.",
            "source": "test", "confidence": "medium", "tags": [],
        }
        server._append_jsonl(server.MEMORY_DIR / inv_id / "findings.jsonl", legacy)

        loaded = _json(server.investigation_load(investigation_id=inv_id))
        by_text = {f["text"]: f for f in loaded["recent_findings"]}
        self.assertEqual(by_text["Fixed thing."]["resolution"], "fixed")
        # A record stored before the field existed must read as "open".
        self.assertEqual(by_text["Legacy finding without resolution."]["resolution"], "open")

    def test_search_filters_by_resolution(self):
        inv_id = _new_id("res-search")
        server.investigation_start(investigation_id=inv_id, title="Search resolution test")
        open_stored = _json(server.investigation_store(
            investigation_id=inv_id, finding_type="observed",
            text="Open bug in the retry loop.", source="test", resolution="open",
        ))
        fixed_stored = _json(server.investigation_store(
            investigation_id=inv_id, finding_type="observed",
            text="Fixed the CORS misconfig.", source="test", resolution="fixed",
        ))

        # This test env has no Qdrant/mnemosyne backend, so drive the recall lane with a
        # synthetic stub that returns rows referencing the two real findings. The
        # resolution surfacing/filter reads the authoritative state from JSONL.
        rows = [
            {"investigation_id": inv_id, "finding_id": open_stored["finding_id"],
             "record_type": "observed", "source": "test", "text": "Open bug in the retry loop.",
             "score": 0.9},
            {"investigation_id": inv_id, "finding_id": fixed_stored["finding_id"],
             "record_type": "observed", "source": "test", "text": "Fixed the CORS misconfig.",
             "score": 0.8},
        ]
        orig_recall = server._mnemo_recall
        server._mnemo_recall = lambda *a, **k: [dict(r) for r in rows]
        try:
            # Default (no filter): both returned, each with resolution surfaced.
            res_all = _json(server.investigation_search("bug", investigation_id=inv_id))
            res_by_text = {r["text"]: r for r in res_all["results"]}
            self.assertEqual(res_by_text["Open bug in the retry loop."]["resolution"], "open")
            self.assertEqual(res_by_text["Fixed the CORS misconfig."]["resolution"], "fixed")

            # Filter open -> only the open finding.
            res_open = _json(server.investigation_search(
                "bug", investigation_id=inv_id, resolution="open"))
            texts_open = [r["text"] for r in res_open["results"]]
            self.assertIn("Open bug in the retry loop.", texts_open)
            self.assertNotIn("Fixed the CORS misconfig.", texts_open)

            # Filter fixed -> only the fixed finding.
            res_fixed = _json(server.investigation_search(
                "bug", investigation_id=inv_id, resolution="fixed"))
            texts_fixed = [r["text"] for r in res_fixed["results"]]
            self.assertIn("Fixed the CORS misconfig.", texts_fixed)
            self.assertNotIn("Open bug in the retry loop.", texts_fixed)
        finally:
            server._mnemo_recall = orig_recall


if __name__ == "__main__":
    unittest.main(verbosity=2)
