"""Finding-lifecycle tests: in-place resolve (finding_resolve), code-ref staleness,
and batch adversarial verify (investigation_verify_all).

Runs fully in-process against a temp MEMORY_DIR, with no Qdrant/Ollama (all those
paths are fail-open). gen is stubbed via server._verify_gen_fn.

Run: pytest mcp/tests/test_finding_lifecycle.py -v
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock
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
        # Code refs are scoped UNDER the code root; point it at a temp repo so
        # relative refs resolve there (and absolute / traversal refs are rejected).
        self._code_root = tempfile.TemporaryDirectory()
        self._orig_code_root = os.environ.get("LOCI_CODE_ROOT")
        os.environ["LOCI_CODE_ROOT"] = self._code_root.name

    def tearDown(self):
        server.MEMORY_DIR = self._orig
        server._verify_gen_fn = self._orig_gen
        if self._orig_code_root is None:
            os.environ.pop("LOCI_CODE_ROOT", None)
        else:
            os.environ["LOCI_CODE_ROOT"] = self._orig_code_root
        self._code_root.cleanup()
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
        # Two referenced files under the code root; only one will change.
        root = Path(self._code_root.name)
        f_change = root / "changing.py"
        f_stable = root / "stable.py"
        f_change.write_text("original = 1\n")
        f_stable.write_text("constant = 1\n")

        # Refs are relative to the code root (absolute refs are rejected by design).
        fid_change = self._store(inv_id, "bug in changing code", code_refs="changing.py")
        fid_stable = self._store(inv_id, "bug in stable code", code_refs="stable.py")

        # Nothing changed yet -> both present-with-refs read as not stale.
        loaded = self._load_findings(inv_id)
        self.assertEqual(loaded[fid_change].get("stale"), False)
        self.assertEqual(loaded[fid_stable].get("stale"), False)

        # Mutate one referenced file.
        f_change.write_text("original = 2  # edited\n")

        loaded = self._load_findings(inv_id)
        self.assertTrue(loaded[fid_change].get("stale"))
        self.assertFalse(loaded[fid_stable].get("stale"))

    def test_search_surfaces_staleness_after_file_change(self):
        # Staleness is surfaced on investigation_search() too (via the JSONL
        # code_refs map), not just investigation_load(). Stub _mnemo_recall so the
        # search returns our finding row without needing Mnemosyne/Qdrant.
        inv_id = self._start()
        root = Path(self._code_root.name)
        (root / "searched.py").write_text("value = 1\n")
        fid = self._store(inv_id, "bug in searched code", code_refs="searched.py")

        orig_recall = server._mnemo_recall
        try:
            server._mnemo_recall = lambda *a, **k: [{
                "investigation_id": inv_id,
                "finding_id": fid,
                "text": "bug in searched code",
                "score": 1.0,
            }]

            def _search_row():
                out = _json(server.investigation_search("bug", investigation_id=inv_id))
                rows = out if isinstance(out, list) else out.get("results", out.get("findings", []))
                match = [r for r in rows if str(r.get("finding_id") or r.get("id")) == fid]
                self.assertTrue(match, f"finding row not in search results: {out}")
                return match[0]

            # Unchanged file -> not stale.
            self.assertEqual(_search_row().get("stale"), False)

            # Mutate the referenced file -> the same search row now reads stale.
            (root / "searched.py").write_text("value = 2  # edited\n")
            self.assertTrue(_search_row().get("stale"))
        finally:
            server._mnemo_recall = orig_recall

    def test_staleness_absent_when_no_refs(self):
        inv_id = self._start()
        fid = self._store(inv_id, "a plain finding with no file references")
        # No usable code_refs -> the stale key is omitted entirely.
        self.assertNotIn("stale", self._load_findings(inv_id)[fid])

    def test_code_refs_accepts_list_form(self):
        # The type hint widened to str | list — a list of refs must still resolve.
        inv_id = self._start()
        (Path(self._code_root.name) / "a.py").write_text("x = 1\n")
        (Path(self._code_root.name) / "b.py").write_text("y = 1\n")
        fid = self._store(inv_id, "spans two files", code_refs=["a.py", "b.py"])
        # Both files present + unchanged -> not stale (proves both hashed).
        self.assertEqual(self._load_findings(inv_id)[fid].get("stale"), False)

    def test_code_refs_empty_is_authoritative_no_refs(self):
        # PROVIDED-BUT-EMPTY ('' or []) is authoritative "no refs": text-parsing
        # must be SKIPPED even though the text contains a parseable ref token.
        # Contrast: omitting code_refs entirely (None) falls back to text parsing.
        (Path(self._code_root.name) / "parsed.py").write_text("q = 1\n")
        text = "the bug lives in parsed.py:12"

        # None (not provided) -> text-parsed ref is stamped.
        self.assertEqual(server._compute_code_refs(text, None), [{"path": "parsed.py", "hash": mock.ANY}])
        # [] (provided-but-empty) -> authoritative no refs.
        self.assertEqual(server._compute_code_refs(text, []), [])
        # '' (provided-but-empty) -> authoritative no refs.
        self.assertEqual(server._compute_code_refs(text, ""), [])

        # End-to-end through investigation_store: code_refs=[] leaves no code_refs
        # on the stored finding despite the parseable token in the text.
        inv_id = self._start()
        fid = self._store(inv_id, text, code_refs=[])
        self.assertNotIn("code_refs", self._load_findings(inv_id)[fid])

    def test_code_ref_path_scoping_rejects_escapes(self):
        # SECURITY: absolute paths and '..' traversal must be refused outright,
        # and a rogue ref must never let the server hash a file outside the root.
        outside = Path(self._tmp.name) / "secret.txt"  # under MEMORY_DIR, NOT code root
        outside.write_text("top secret\n")

        self.assertIsNone(server._safe_repo_path(str(outside)))         # absolute
        self.assertIsNone(server._safe_repo_path("/etc/passwd"))        # absolute
        self.assertIsNone(server._safe_repo_path("../secret.txt"))      # traversal
        self.assertIsNone(server._safe_repo_path("a/../../secret.txt")) # nested traversal
        self.assertIsNone(server._hash_file_bytes(str(outside)))        # not hashed

        # A legitimate in-root ref still resolves + hashes.
        (Path(self._code_root.name) / "ok.py").write_text("z = 1\n")
        self.assertIsNotNone(server._safe_repo_path("ok.py"))
        self.assertIsNotNone(server._hash_file_bytes("ok.py"))

    def test_hash_file_size_cap_skips_oversized(self):
        # SECURITY: files over the cap are skipped (returns None), never read whole.
        big = Path(self._code_root.name) / "big.bin"
        big.write_bytes(b"\0" * 64)
        orig = server._HASH_FILE_MAX_BYTES
        try:
            server._HASH_FILE_MAX_BYTES = 16
            self.assertIsNone(server._hash_file_bytes("big.bin"))
            server._HASH_FILE_MAX_BYTES = 1024
            self.assertIsNotNone(server._hash_file_bytes("big.bin"))
        finally:
            server._HASH_FILE_MAX_BYTES = orig

    def test_hash_file_max_bytes_env_parse_fails_open(self):
        # A bad LOCI_HASH_FILE_MAX_BYTES must fall back to the default cap and
        # NEVER raise (an import-time ValueError would block server startup).
        default = server._HASH_FILE_MAX_BYTES_DEFAULT
        orig = os.environ.get("LOCI_HASH_FILE_MAX_BYTES")
        try:
            for bad in ("not-a-number", "", "8MB", "  ", "-1", "0"):
                os.environ["LOCI_HASH_FILE_MAX_BYTES"] = bad
                self.assertEqual(server._parse_hash_file_max_bytes(), default)
            # A valid positive override is honored.
            os.environ["LOCI_HASH_FILE_MAX_BYTES"] = "4096"
            self.assertEqual(server._parse_hash_file_max_bytes(), 4096)
            # Unset falls back to the default.
            del os.environ["LOCI_HASH_FILE_MAX_BYTES"]
            self.assertEqual(server._parse_hash_file_max_bytes(), default)
        finally:
            if orig is None:
                os.environ.pop("LOCI_HASH_FILE_MAX_BYTES", None)
            else:
                os.environ["LOCI_HASH_FILE_MAX_BYTES"] = orig

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
