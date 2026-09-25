"""Access bookkeeping must not shadow findings, and rewrites must not drop lines.

Older rag_context_search builds appended an ``{id: <finding id>, record_type:
"access"}`` row to findings.jsonl for every hit. Readers where the last row wins
then read that text-less row as the finding. memory_retract lost the seed's text
and entities, so the contaminated lineage went unretracted. The search
resolution map reported a stored 'fixed' as 'open'. Export and memory_health
counted the rows as findings. The live store held 7,190 such rows against about
5,784 real findings.

The rows now go to access.jsonl. Legacy rows already in findings.jsonl are
ignored by every reader, and the tier and procedure rewrites keep every line
they do not change, torn lines included, byte-for-byte.

Runs in-process against a temp MEMORY_DIR with no Qdrant, Mnemosyne or Ollama.
"""

import json
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import server  # noqa: E402
import inv_store  # noqa: E402


def _j(s):
    return json.loads(s)


class AccessRowsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)
        self._stack = ExitStack()
        for name in ("_get_qdrant",):
            self._stack.enter_context(mock.patch.object(server, name, lambda *a, **k: (None, None)))
        for name in ("_qdrant_upsert", "_event_log_append", "_mirror_finding_to_ladybug",
                     "_autolink_finding_to_ladybug", "_update_entities_jsonl"):
            if hasattr(server, name):
                self._stack.enter_context(mock.patch.object(server, name, lambda *a, **k: None))
        for name in ("_mnemo_remember", "_retract_quarantine_verdict"):
            if hasattr(server, name):
                self._stack.enter_context(mock.patch.object(server, name, lambda *a, **k: False))
        AccessRowsTest._n = getattr(AccessRowsTest, "_n", 0) + 1
        self.inv = f"acc-inv-{AccessRowsTest._n:03d}"
        server.investigation_start(investigation_id=self.inv, title="access rows")
        self.fpath = server.MEMORY_DIR / self.inv / "findings.jsonl"

    def tearDown(self):
        self._stack.close()
        server.MEMORY_DIR = self._orig
        self._tmp.cleanup()

    def _store(self, text, **kw):
        r = _j(server.investigation_store(self.inv, "observed", text, "unit-test", **kw))
        self.assertTrue(r.get("stored"), r)
        return r["finding_id"]

    def _legacy_access_row(self, fid):
        """Append the exact row older rag_context_search builds wrote."""
        with open(self.fpath, "a") as fh:
            fh.write(json.dumps({"id": fid, "investigation_id": self.inv, "record_type": "access",
                                 "last_accessed": int(time.time()), "query": "q"}) + "\n")

    def _dry_retract(self, fid):
        d = _j(server.memory_retract(self.inv, fid, dry_run=True, scope_semantic=False))
        return [(i["finding_id"], i["text_excerpt"]) for i in d["would_retract"]]

    # -- access-records-shadow-findings / access-row-shadows-retraction-seed --

    def test_rag_access_goes_to_access_log_not_findings(self):
        a = self._store("C2 server at 10.66.66.66 beacons to evil-c2.example.net")
        before = self.fpath.read_text()
        server._rag_record_access(
            [{"origin": server.QDRANT_COLLECTION_PREFIX, "id": a, "investigation_id": self.inv}], "c2")
        self.assertEqual(self.fpath.read_text(), before, "access marker written into findings.jsonl")
        rows = inv_store._read_jsonl(server.MEMORY_DIR / self.inv / inv_store.ACCESS_LOG_NAME)
        self.assertEqual([(r["id"], r["record_type"]) for r in rows], [(a, "access")])

    def test_rag_access_then_retract_keeps_lineage(self):
        a = self._store("C2 server at 10.66.66.66 beacons to evil-c2.example.net")
        b = self._store("Firewall logs confirm outbound traffic to 10.66.66.66 on 443")
        before = self._dry_retract(a)
        self.assertEqual(len(before), 2, before)
        server._rag_record_access(
            [{"origin": server.QDRANT_COLLECTION_PREFIX, "id": a, "investigation_id": self.inv}], "c2")
        after = self._dry_retract(a)
        self.assertEqual(after, before)
        self.assertIn(b, [fid for fid, _ in after])

    def test_legacy_access_row_does_not_shadow_retraction_seed(self):
        a = self._store("C2 server at 10.66.66.66 beacons to evil-c2.example.net")
        b = self._store("Firewall logs confirm outbound traffic to 10.66.66.66 on 443")
        before = self._dry_retract(a)
        self._legacy_access_row(a)
        after = self._dry_retract(a)
        self.assertEqual(len(after), 2, after)
        self.assertIn(b, [fid for fid, _ in after])
        excerpt = dict(after)[a]
        self.assertTrue(excerpt.startswith("C2 server"), excerpt)
        self.assertEqual(after, before)

    def test_legacy_access_row_does_not_mask_stored_resolution(self):
        c = self._store("Delta cache is fixed already", resolution="fixed")
        self._legacy_access_row(c)
        rm, _ = server._search_resolution_maps([{"investigation_id": self.inv}])
        self.assertEqual(rm.get(c), "fixed")

    def test_export_does_not_count_legacy_access_rows(self):
        ids = [self._store(f"Real finding number {i} about host{i}.example") for i in range(3)]
        self._legacy_access_row(ids[0])
        self._legacy_access_row(ids[1])
        ex = _j(server.investigation_export(self.inv))
        self.assertEqual(ex["finding_count"], 3)

    # -- store-counts-inflated-by-access-rows --

    def test_store_counts_count_real_findings_only(self):
        ids = [self._store(f"Real finding number {i} about host{i}.example") for i in range(2)]
        for fid in ids:
            self._legacy_access_row(fid)
            self._legacy_access_row(fid)
        server._rag_record_access(
            [{"origin": server.QDRANT_COLLECTION_PREFIX, "id": ids[0], "investigation_id": self.inv}], "q")
        status, detail, _ = server._health_probe_store_counts([self.inv], None)
        self.assertEqual(status, "ok")
        self.assertEqual(detail["totals"]["findings"], 2, detail)
        self.assertEqual(detail["per_investigation"][0]["findings"], 2)

    # -- tier-rewrite-drops-unparseable-lines --

    def test_promote_keeps_torn_line(self):
        a = self._store("Alpha finding about 10.2.2.2")
        with open(self.fpath, "a") as fh:
            fh.write('{"id": "torn-record", "text": "half written\n')
        self._store("Beta finding about 10.3.3.3")
        n_before = len(self.fpath.read_text().splitlines())
        # promote reports ok only once the point is indexed; stub a landed upsert.
        with mock.patch.object(server, "_qdrant_upsert", lambda *a, **k: True):
            r = _j(server.memory_promote(self.inv, a, "hot"))
        self.assertTrue(r.get("ok"), r)
        text = self.fpath.read_text()
        self.assertEqual(len(text.splitlines()), n_before)
        self.assertIn('{"id": "torn-record", "text": "half written', text)
        rows = inv_store._read_jsonl(self.fpath)
        self.assertEqual([f.get("tier") for f in rows if f["id"] == a], ["hot"])

    def test_demote_keeps_legacy_access_rows_verbatim(self):
        a = self._store("Alpha finding about 10.2.2.2")
        self._legacy_access_row(a)
        access_line = self.fpath.read_text().splitlines()[-1]
        r = _j(server.memory_demote(self.inv, a, "cold"))
        self.assertTrue(r.get("ok"), r)
        self.assertIn(access_line, self.fpath.read_text().splitlines())

    def test_procedure_attempt_keeps_torn_line(self):
        p = _j(server.investigation_store(self.inv, "procedure", "Run make check before pushing",
                                          "unit-test"))
        self.assertTrue(p.get("stored"), p)
        with open(self.fpath, "a") as fh:
            fh.write('{"id": "torn-proc", "text": "half\n')
        r = _j(server.procedure_attempt(self.inv, p["finding_id"], True))
        self.assertEqual(r.get("attempt_count"), 1, r)
        self.assertIn('{"id": "torn-proc", "text": "half', self.fpath.read_text())


class RewritePreservingTest(unittest.TestCase):
    def test_only_replaced_rows_change(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "findings.jsonl"
            lines = ['{"id": "a",  "tier": "warm"}', "not json at all",
                     '{"id": "a", "record_type": "access"}', '{"id": "b"}']
            p.write_text("\n".join(lines) + "\n")
            n = inv_store._rewrite_jsonl_preserving(
                p, lambda f: {**f, "tier": "hot"} if f.get("id") == "a" else None)
            self.assertEqual(n, 1)
            out = p.read_text().splitlines()
            self.assertEqual(json.loads(out[0]), {"id": "a", "tier": "hot"})
            self.assertEqual(out[1:], lines[1:])


if __name__ == "__main__":
    unittest.main()
