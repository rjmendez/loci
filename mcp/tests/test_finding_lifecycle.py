"""Finding-lifecycle tests: in-place resolve (finding_resolve), code-ref staleness,
and batch adversarial verify (investigation_verify_all).

Runs fully in-process against a temp MEMORY_DIR, with no Qdrant/Ollama (all those
paths are fail-open). gen is stubbed via server._verify_gen_fn.

Run: pytest mcp/tests/test_finding_lifecycle.py -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import server  # noqa: E402


def _json(result: str) -> dict:
    try:
        return json.loads(result)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Tool returned non-JSON: {result!r}") from exc


_counter = [0]


def _new_id(prefix="lc"):
    _counter[0] += 1
    return f"{prefix}-{_counter[0]:04d}"


class FindingLifecycleTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)
        self._orig_gen = server._verify_gen_fn

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        server._verify_gen_fn = self._orig_gen
        self._tmp.cleanup()

    def _start(self, prefix="lc"):
        inv_id = _new_id(prefix)
        server.investigation_start(investigation_id=inv_id, title="lifecycle test")
        return inv_id

    def _store(self, inv_id, text="a finding", ftype="observed", **kw):
        res = _json(server.investigation_store(inv_id, ftype, text, "unit-test", **kw))
        self.assertTrue(res.get("stored"), res)
        return res["finding_id"]

    def _load_findings(self, inv_id):
        loaded = _json(server.investigation_load(inv_id))
        return {f["id"]: f for f in loaded["recent_findings"]}

    # -- #1 finding_resolve --------------------------------------------------

    def test_resolve_appends_and_load_reflects_last_write_wins(self):
        inv_id = self._start()
        fid = self._store(inv_id, "the bug is here")

        # Default state is open.
        self.assertEqual(self._load_findings(inv_id)[fid]["resolution"], "open")

        r1 = _json(server.finding_resolve(inv_id, fid, "fixed", note="patched"))
        self.assertTrue(r1["resolved"])
        self.assertEqual(self._load_findings(inv_id)[fid]["resolution"], "fixed")

        # Second resolve wins (last-write-wins over the append-only log).
        _json(server.finding_resolve(inv_id, fid, "wontfix"))
        self.assertEqual(self._load_findings(inv_id)[fid]["resolution"], "wontfix")

        # findings.jsonl was NOT rewritten — the update lives in a sibling log.
        raw = (server.MEMORY_DIR / inv_id / "findings.jsonl").read_text()
        self.assertNotIn('"record_type": "resolution"', raw)
        self.assertTrue((server.MEMORY_DIR / inv_id / "finding_updates.jsonl").exists())

    def test_resolve_rejects_invalid_resolution(self):
        inv_id = self._start()
        fid = self._store(inv_id)
        res = _json(server.finding_resolve(inv_id, fid, "bogus"))
        self.assertIn("error", res)
        # Unchanged: still open.
        self.assertEqual(self._load_findings(inv_id)[fid]["resolution"], "open")

    def test_resolve_unknown_finding_errors(self):
        inv_id = self._start()
        res = _json(server.finding_resolve(inv_id, "no-such-id", "fixed"))
        self.assertIn("error", res)

    # -- #3 staleness --------------------------------------------------------

    def test_staleness_flags_changed_file_only(self):
        inv_id = self._start()
        # Two referenced files; only one will change.
        f_change = Path(self._tmp.name) / "changing.py"
        f_stable = Path(self._tmp.name) / "stable.py"
        f_change.write_text("original = 1\n")
        f_stable.write_text("constant = 1\n")

        fid_change = self._store(inv_id, "bug in changing code", code_refs=str(f_change))
        fid_stable = self._store(inv_id, "bug in stable code", code_refs=str(f_stable))

        # Nothing changed yet -> both present-with-refs read as not stale.
        loaded = self._load_findings(inv_id)
        self.assertEqual(loaded[fid_change].get("stale"), False)
        self.assertEqual(loaded[fid_stable].get("stale"), False)

        # Mutate one referenced file.
        f_change.write_text("original = 2  # edited\n")

        loaded = self._load_findings(inv_id)
        self.assertTrue(loaded[fid_change].get("stale"))
        self.assertFalse(loaded[fid_stable].get("stale"))

    def test_staleness_absent_when_no_refs(self):
        inv_id = self._start()
        fid = self._store(inv_id, "a plain finding with no file references")
        # No usable code_refs -> the stale key is omitted entirely.
        self.assertNotIn("stale", self._load_findings(inv_id)[fid])

    # -- #5 investigation_verify_all -----------------------------------------

    def test_verify_all_returns_per_finding_verdicts_with_stub(self):
        inv_id = self._start()
        fid1 = self._store(inv_id, "claim one")
        fid2 = self._store(inv_id, "claim two")

        calls = []

        def _stub_gen(prompt, *, fmt=None, max_tokens=256):
            calls.append(prompt)
            return {"ok": True, "text": json.dumps(
                {"verdict": "confirmed", "refutation": "", "confidence": 0.8})}

        server._verify_gen_fn = _stub_gen

        out = _json(server.investigation_verify_all(inv_id))
        self.assertEqual(out["verified"], 2)
        by_id = {r["finding_id"]: r for r in out["results"]}
        self.assertEqual(by_id[fid1]["verdict"], "confirmed")
        self.assertEqual(by_id[fid2]["verdict"], "confirmed")
        self.assertEqual(by_id[fid1]["confidence"], 0.8)
        self.assertEqual(len(calls), 2)

        # A verification note is recorded but MUST NOT change the resolution.
        self.assertEqual(self._load_findings(inv_id)[fid1]["resolution"], "open")

    def test_verify_all_skips_resolved_findings(self):
        inv_id = self._start()
        fid_open = self._store(inv_id, "still open")
        fid_done = self._store(inv_id, "already fixed")
        _json(server.finding_resolve(inv_id, fid_done, "fixed"))

        def _stub_gen(prompt, *, fmt=None, max_tokens=256):
            return {"ok": True, "text": json.dumps(
                {"verdict": "uncertain", "refutation": "", "confidence": 0.0})}

        server._verify_gen_fn = _stub_gen
        out = _json(server.investigation_verify_all(inv_id))
        ids = {r["finding_id"] for r in out["results"]}
        self.assertIn(fid_open, ids)
        self.assertNotIn(fid_done, ids)

    def test_verify_all_respects_limit(self):
        inv_id = self._start()
        for i in range(3):
            self._store(inv_id, f"claim {i}")

        server._verify_gen_fn = lambda *a, **k: {"ok": True, "text": json.dumps(
            {"verdict": "uncertain", "refutation": "", "confidence": 0.0})}
        out = _json(server.investigation_verify_all(inv_id, limit=2))
        self.assertEqual(out["verified"], 2)


if __name__ == "__main__":
    unittest.main()
