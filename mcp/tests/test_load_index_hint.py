"""investigation_load: id missing from the store but present in the shared index."""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inv_store  # noqa: E402
import investigation_tools as it  # noqa: E402

_ERR = "Investigation 'ghost-inv' not found. Call investigation_start first."


class _Client:
    def __init__(self, n=0, exc=None, delay=0.0):
        self.n, self.exc, self.delay = n, exc, delay
        self.calls = 0

    def scroll(self, col, **kw):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.exc:
            raise self.exc
        return [object()] * self.n, None


class LoadIndexHintTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        for mod in (it, inv_store):
            p = mock.patch.object(mod, "_get_memory_dir", lambda: Path(self._tmp.name))
            p.start()
            self.addCleanup(p.stop)

    def _load(self, client):
        qc = (client, "col") if client else (None, None)
        with mock.patch("qdrant_ops._get_qdrant", return_value=qc):
            return json.loads(it.investigation_load("ghost-inv"))

    def test_found_in_index_adds_fields(self):
        c = _Client(n=3)
        got = self._load(c)
        self.assertEqual(got["error"], _ERR)
        self.assertEqual(got["indexed_findings"], 3)
        self.assertEqual(got["indexed_investigation_id"], "ghost-inv")
        self.assertIn("shared", got["hint"])
        self.assertNotIn("manifest", got)
        self.assertEqual(c.calls, 1)
        self.assertFalse((Path(self._tmp.name) / "ghost-inv").exists())

    def test_count_is_capped(self):
        got = self._load(_Client(n=it._INDEX_HINT_CAP + 1))
        self.assertEqual(got["indexed_findings"], it._INDEX_HINT_CAP)
        self.assertTrue(got["indexed_findings_capped"])

    def test_not_in_index_keeps_old_error(self):
        self.assertEqual(self._load(_Client(n=0)), {"error": _ERR})

    def test_qdrant_unavailable_keeps_old_error(self):
        self.assertEqual(self._load(None), {"error": _ERR})

    def test_qdrant_raises_keeps_old_error(self):
        self.assertEqual(self._load(_Client(exc=RuntimeError("down"))), {"error": _ERR})

    def test_timeout_keeps_old_error(self):
        with mock.patch.object(it, "_INDEX_HINT_TIMEOUT_S", 0.05):
            got = self._load(_Client(n=3, delay=0.5))
        self.assertEqual(got, {"error": _ERR})


if __name__ == "__main__":
    unittest.main()
