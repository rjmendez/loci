"""Tests for the collection-creation and source-table behaviour of the Qdrant sync scripts.

Background: neither script ever created its collection, and a2a_server's _qdrant_search turns
a 404 into [], so a missing collection reads as "0 results" at every call site rather than as
an error. mnemosyne_qdrant_sync additionally read only `working_memory` -- the small staging
tier -- so it reported success while leaving the semantic index effectively empty.

These exercise the real functions with the network helper stubbed; no Qdrant required.
"""

import importlib.util
import os
import pathlib
import sqlite3
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("LOCI_ENV_FILE", "/nonexistent-env-file-for-tests")
os.environ.setdefault("QDRANT_URL", "http://qdrant.invalid:6333")

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent


def _load(modname, filename):
    spec = importlib.util.spec_from_file_location(modname, _SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mq = _load("mnemosyne_sync_impl", "mnemosyne_qdrant_sync.py")
sd = _load("state_db_sync_impl", "state_db_qdrant_sync.py")


class _Recorder:
    """Stands in for curl()/curl_json(); records calls and replays canned responses."""

    def __init__(self, present_after_create=True, present_initially=False):
        self.calls = []
        self.present = present_initially
        self.present_after_create = present_after_create

    def __call__(self, method, url, data=None, *a, **kw):
        self.calls.append((method, url, data))
        if method == "GET":
            return {"result": {"points_count": 0}, "status": "ok"} if self.present else {}
        if method == "PUT" and "/points" not in url:
            self.present = self.present_after_create
            return {"result": True, "status": "ok"}
        return {"result": {}, "status": "ok"}

    def puts(self):
        return [c for c in self.calls if c[0] == "PUT" and "/points" not in c[1]]


class TestMnemosyneEnsureCollection(unittest.TestCase):
    def test_creates_when_absent(self):
        rec = _Recorder()
        with mock.patch.object(mq, "curl", rec):
            self.assertTrue(mq.ensure_collection("mnemosyne"))
        self.assertEqual(len(rec.puts()), 1)

    def test_schema_is_named_dense_cosine(self):
        rec = _Recorder()
        with mock.patch.object(mq, "curl", rec):
            mq.ensure_collection("mnemosyne")
        body = rec.puts()[0][2]
        self.assertEqual(body["vectors"]["dense"]["distance"], "Cosine")
        self.assertEqual(body["vectors"]["dense"]["size"], 768)

    def test_noop_when_already_present(self):
        rec = _Recorder(present_initially=True)
        with mock.patch.object(mq, "curl", rec):
            self.assertFalse(mq.ensure_collection("mnemosyne"))
        self.assertEqual(rec.puts(), [])

    def test_reports_failure_when_create_does_not_stick(self):
        # curl() swallows errors and returns {}, so a failed create must be caught by re-reading.
        rec = _Recorder(present_after_create=False)
        with mock.patch.object(mq, "curl", rec):
            self.assertFalse(mq.ensure_collection("mnemosyne"))

    def test_dim_follows_embedding_env(self):
        rec = _Recorder()
        with mock.patch.dict(os.environ, {"MNEMOSYNE_EMBEDDING_DIM": "1024"}), \
             mock.patch.object(mq, "curl", rec):
            mq.ensure_collection("mnemosyne")
        self.assertEqual(rec.puts()[0][2]["vectors"]["dense"]["size"], 1024)


class TestStateDbEnsureCollection(unittest.TestCase):
    def test_creates_when_absent(self):
        rec = _Recorder()
        with mock.patch.object(sd, "curl_json", rec):
            self.assertTrue(sd.ensure_collection("k", "hermes_sessions"))
        self.assertEqual(len(rec.puts()), 1)

    def test_noop_when_already_present(self):
        rec = _Recorder(present_initially=True)
        with mock.patch.object(sd, "curl_json", rec):
            self.assertFalse(sd.ensure_collection("k", "hermes_sessions"))
        self.assertEqual(rec.puts(), [])


class TestMemorySourceTables(unittest.TestCase):
    """The regression that mattered: syncing only working_memory looked like success."""

    def _db(self, memories, working):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        con = sqlite3.connect(path)
        for t in ("memories", "working_memory"):
            con.execute(
                f"CREATE TABLE {t} (id TEXT, content TEXT, source TEXT, "
                "importance REAL, session_id TEXT, created_at TEXT)"
            )
        for t, rows in (("memories", memories), ("working_memory", working)):
            con.executemany(
                f"INSERT INTO {t} VALUES (?,?,?,?,?,?)",
                [(i, c, "s", 0.5, "", "2026-01-01") for i, c in rows],
            )
        con.commit()
        con.close()
        self.addCleanup(os.unlink, path)
        return path

    def _load(self, memories, working):
        con = sqlite3.connect(self._db(memories, working))
        con.row_factory = sqlite3.Row
        try:
            return mq.load_memories(con)
        finally:
            con.close()

    def test_includes_the_durable_memories_table(self):
        # The regression: the old query read working_memory ONLY, so m1/m2 were never indexed.
        got = self._load([("m1", "durable one"), ("m2", "durable two")], [("w1", "staged")])
        self.assertEqual({m["id"] for m in got}, {"m1", "m2", "w1"})

    def test_tier_is_recorded_per_row(self):
        got = {m["id"]: m["tier"] for m in self._load([("m1", "a")], [("w1", "b")])}
        self.assertEqual(got, {"m1": "memory", "w1": "working"})

    def test_id_in_both_tiers_yields_one_row_preferring_durable(self):
        got = self._load([("dup", "durable")], [("dup", "staged")])
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["tier"], "memory")
        self.assertEqual(got[0]["content"], "durable")

    def test_empty_and_whitespace_content_is_dropped(self):
        got = self._load([("m1", "real"), ("m2", ""), ("m3", "   ")], [])
        self.assertEqual({m["id"] for m in got}, {"m1"})

    def test_missing_table_is_skipped_not_fatal(self):
        con = sqlite3.connect(self._db([("m1", "a")], []))
        con.row_factory = sqlite3.Row
        try:
            got = mq.load_memories(
                con, tables=(("memories", "memory"), ("no_such_table", "ghost")),
            )
        finally:
            con.close()
        self.assertEqual([m["id"] for m in got], ["m1"])


if __name__ == "__main__":
    unittest.main()
