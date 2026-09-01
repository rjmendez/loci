"""Two age checks that could not answer the question they were asked.

Both readers of a finding's age default a missing/unreadable timestamp to
"fresh": _rag_apply_decay keeps the raw score, and the retention purge cannot
match a point with no created_at_ts at all. So a row that loses its age does not
degrade — it wins, and keeps winning as the corpus gets older.

1. investigation_import re-indexed findings with a hand-built payload that
   dropped created_at_ts, so every imported finding was un-ageable and
   un-purgeable.
2. _rag_apply_decay's ISO fallback ran float() on an ISO string, which always
   raised, so any point carrying only `ts` escaped decay through a swallowed
   exception.
"""
import json
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

import investigation_tools
import server


def _bundle(finding):
    return json.dumps({
        "schema_version": "1.0",
        "manifest": {"id": "src-inv", "title": "Source"},
        "findings": [finding],
    })


class TestImportKeepsFindingAge(unittest.TestCase):
    """investigation_import must index the age it was given."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_dir = server.MEMORY_DIR
        server.MEMORY_DIR = Path(self._tmp.name)
        self._orig_upsert = investigation_tools._qdrant_upsert
        self.upserts = []
        investigation_tools._qdrant_upsert = lambda pid, text, payload: self.upserts.append(
            (pid, text, dict(payload))
        )

    def tearDown(self):
        investigation_tools._qdrant_upsert = self._orig_upsert
        server.MEMORY_DIR = self._orig_dir
        self._tmp.cleanup()

    def test_imported_finding_carries_its_own_created_at_ts(self):
        old_ts = int(time.time()) - 90 * 86400
        result = json.loads(server.investigation_import(bundle_json=_bundle({
            "id": "f-old",
            "text": "A finding exported from another machine three months ago.",
            "type": "observed",
            "source": "test:bundle",
            "confidence": "high",
            "created_at_ts": old_ts,
            "ts": datetime.fromtimestamp(old_ts, timezone.utc).isoformat(),
        })))
        self.assertNotIn("error", result)
        self.assertEqual(result["qdrant_indexed"], 1)

        payload = self.upserts[0][2]
        self.assertEqual(payload["created_at_ts"], old_ts)
        self.assertNotIn("age_source", payload)
        # The import still owns the row: investigation_id is rewritten.
        self.assertEqual(payload["investigation_id"], result["new_investigation_id"])

    def test_ageless_finding_gets_import_time_and_says_so(self):
        before = int(time.time())
        result = json.loads(server.investigation_import(bundle_json=_bundle({
            "id": "f-ageless",
            "text": "A finding from a bundle that carried no timestamp at all.",
            "type": "observed",
        })))
        self.assertNotIn("error", result)

        payload = self.upserts[0][2]
        self.assertGreaterEqual(payload["created_at_ts"], before)
        self.assertEqual(payload["age_source"], "imported")


class TestRagDecayReadsIsoTimestamps(unittest.TestCase):
    """A point carrying only `ts` is not exempt from decay."""

    def _row(self, **kw):
        row = {"origin": server.QDRANT_COLLECTION_PREFIX, "score": 1.0}
        row.update(kw)
        return row

    def test_iso_ts_fallback_is_decayed(self):
        old = datetime.fromtimestamp(time.time() - 100 * 86400, timezone.utc)
        rows = [self._row(ts=old.isoformat())]
        server._rag_apply_decay(rows)
        self.assertTrue(rows[0].get("decay_applied"))
        self.assertLess(rows[0]["score"], 0.6)

    def test_iso_and_epoch_rows_of_the_same_age_score_the_same(self):
        old_epoch = int(time.time()) - 100 * 86400
        old_iso = datetime.fromtimestamp(old_epoch, timezone.utc).isoformat()
        rows = [self._row(created_at_ts=old_epoch), self._row(ts=old_iso)]
        server._rag_apply_decay(rows)
        self.assertEqual(rows[0]["score"], rows[1]["score"])

    def test_unreadable_timestamp_still_keeps_the_raw_score(self):
        rows = [self._row(ts="not a timestamp"), self._row()]
        server._rag_apply_decay(rows)
        for row in rows:
            self.assertEqual(row["score"], 1.0)
            self.assertNotIn("decay_applied", row)


if __name__ == "__main__":
    unittest.main(verbosity=2)
