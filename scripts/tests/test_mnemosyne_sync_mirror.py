"""mnemosyne_qdrant_sync must mirror SQLite and fail loudly.

Regressions covered (audit finding mnemo-qdrant-sync-add-only-exit0):
- an unreachable Qdrant, a failed scroll or a failed batch used to print "Sync complete" and
  exit 0, so cron recorded success;
- memories deleted from SQLite (forget/retract) stayed in the `mnemosyne` collection forever;
- an edited memory was skipped by memory_id and kept its stale vector.

Everything network-facing is stubbed; no Qdrant, Ollama or live Mnemosyne DB is touched.
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


def _load():
    spec = importlib.util.spec_from_file_location(
        "mnemosyne_sync_mirror_impl", _SCRIPTS / "mnemosyne_qdrant_sync.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mq = _load()


class FakeQdrant:
    """Stands in for curl(): an in-memory collection, with switchable failures."""

    def __init__(self, points=None, *, down=False, fail_upsert=False, fail_delete=False):
        self.points = {p["id"]: p for p in (points or [])}
        self.down = down
        self.fail_upsert = fail_upsert
        self.fail_delete = fail_delete
        self.calls = []

    def __call__(self, method, url, data=None, extra_headers=None):
        self.calls.append((method, url, data))
        if self.down:
            return {}  # what curl() returns for every HTTP / connection error
        if method == "GET":
            return {"result": {"points_count": len(self.points)}, "status": "ok"}
        if url.endswith("/points/scroll"):
            pts = [{"id": pid, "payload": dict(p["payload"])} for pid, p in self.points.items()]
            return {"result": {"points": pts, "next_page_offset": None}, "status": "ok"}
        if url.endswith("/points/delete"):
            if self.fail_delete:
                return {}
            for pid in data["points"]:
                self.points.pop(pid, None)
            return {"result": {}, "status": "ok"}
        if method == "PUT" and url.endswith("/points"):
            if self.fail_upsert:
                return {"status": {"error": "boom"}}
            for p in data["points"]:
                self.points[p["id"]] = p
            return {"result": {}, "status": "ok"}
        return {"result": True, "status": "ok"}

    def deletes(self):
        return [c for c in self.calls if c[1].endswith("/points/delete")]


def _point(mid, content, agent_id="", profile=""):
    return {"id": mq.stable_num_id(mid), "payload": {
        "memory_id": mid, "content": content, "agent_id": agent_id, "profile": profile}}


class MirrorSyncTest(unittest.TestCase):
    def _db(self, rows, *, tables=True):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        con = sqlite3.connect(path)
        if tables:
            for t in ("memories", "working_memory"):
                con.execute(f"CREATE TABLE {t} (id TEXT, content TEXT, source TEXT, "
                            "importance REAL, session_id TEXT, created_at TEXT)")
            con.executemany("INSERT INTO memories VALUES (?,?,?,?,?,?)",
                            [(i, c, "s", 0.5, "", "2026-01-01") for i, c in rows])
        con.commit()
        con.close()
        return path

    def _run(self, db_path, fake, embed_ok=True, argv=(), agent_id="", profile=""):
        def embed(chunks):
            return {c["id"]: [0.1, 0.2] for c in chunks} if embed_ok else {}
        with mock.patch.object(mq, "curl", fake), \
             mock.patch.object(mq, "MNEMOSYNE_DB", db_path), \
             mock.patch.object(mq, "AGENT_ID", agent_id), \
             mock.patch.object(mq, "PROFILE", profile), \
             mock.patch.object(mq, "embed_and_get_vector", embed), \
             mock.patch.object(mq.time, "sleep", lambda *_: None), \
             mock.patch.object(mq.sys, "argv", ["mnemosyne_qdrant_sync.py", *argv]):
            return mq.main()

    def test_unreachable_qdrant_exits_nonzero(self):
        rc = self._run(self._db([("m1", "hello")]), FakeQdrant(down=True))
        self.assertEqual(rc, 1)

    def test_failed_batch_exits_nonzero(self):
        rc = self._run(self._db([("m1", "hello")]), FakeQdrant(fail_upsert=True))
        self.assertEqual(rc, 1)

    def test_failed_embedding_exits_nonzero(self):
        rc = self._run(self._db([("m1", "hello")]), FakeQdrant(), embed_ok=False)
        self.assertEqual(rc, 1)

    def test_clean_run_exits_zero(self):
        fake = FakeQdrant()
        rc = self._run(self._db([("m1", "hello")]), fake)
        self.assertEqual(rc, 0)
        self.assertEqual({p["payload"]["memory_id"] for p in fake.points.values()}, {"m1"})

    HOST = {"agent_id": "host-a", "profile": "default"}

    def _hp(self, mid, content):
        return _point(mid, content, **self.HOST)

    def test_deleted_memory_is_removed_from_qdrant(self):
        other_host = _point("x9", "someone else's memory", agent_id="other-host")
        foreign = {"id": 77, "payload": {"content": "agentHER synthetic", "source": "agentHER"}}
        fake = FakeQdrant([self._hp("m1", "kept"), self._hp("gone", "forgotten memory"),
                           other_host, foreign])
        rc = self._run(self._db([("m1", "kept")]), fake, argv=["--prune"], **self.HOST)
        self.assertEqual(rc, 0)
        self.assertNotIn(mq.stable_num_id("gone"), fake.points)
        # Only this host's mirrored points are pruned.
        self.assertIn(other_host["id"], fake.points)
        self.assertIn(77, fake.points)
        self.assertIn(mq.stable_num_id("m1"), fake.points)

    def test_no_prune_flag_keeps_orphans(self):
        fake = FakeQdrant([self._hp("m1", "kept"), self._hp("gone", "forgotten memory")])
        rc = self._run(self._db([("m1", "kept")]), fake, argv=["--no-prune"], **self.HOST)
        self.assertEqual(rc, 0)
        self.assertEqual(fake.deletes(), [])
        self.assertIn(mq.stable_num_id("gone"), fake.points)

    def test_pruning_is_off_by_default(self):
        # Review blocker: the 30-minute cron must not delete anything unless asked to.
        fake = FakeQdrant([self._hp("m1", "kept"), self._hp("gone", "forgotten memory")])
        rc = self._run(self._db([("m1", "kept")]), fake, **self.HOST)
        self.assertEqual(rc, 0)
        self.assertEqual(fake.deletes(), [])
        self.assertIn(mq.stable_num_id("gone"), fake.points)

    def test_failed_delete_exits_nonzero(self):
        fake = FakeQdrant([self._hp("m1", "kept"), self._hp("gone", "forgotten memory")],
                          fail_delete=True)
        self.assertEqual(
            self._run(self._db([("m1", "kept")]), fake, argv=["--prune"], **self.HOST), 1)

    def test_schema_drift_is_fatal_and_deletes_nothing(self):
        # Review blocker: a `memories` table missing a column used to be skipped, read as "no
        # memories", and every mirrored point (this host's and, on default env, other hosts')
        # was pruned.
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE memories (id TEXT, content TEXT, source TEXT, "
                    "session_id TEXT, created_at TEXT)")
        con.execute("INSERT INTO memories VALUES ('m1','hello','s','b','t')")
        con.commit()
        con.close()
        fake = FakeQdrant([self._hp("m1", "hello"), self._hp("m2", "x"),
                           _point("other-host-mem", "x")])
        for argv in (["--prune"], ["--prune", "--force-prune"]):
            rc = self._run(path, fake, argv=argv, **self.HOST)
            self.assertEqual(rc, 1)
            self.assertEqual(fake.deletes(), [])
            self.assertEqual(len(fake.points), 3)

    def test_prune_refused_without_host_identity(self):
        # Empty AGENT_ID/PROFILE match every other default-env host's points.
        other = _point("other-host-mem", "x")
        fake = FakeQdrant([_point("m1", "kept"), _point("m2", "kept2"), other])
        rc = self._run(self._db([("m1", "kept"), ("m2", "kept2")]), fake, argv=["--prune"])
        self.assertEqual(rc, 1)
        self.assertEqual(fake.deletes(), [])
        self.assertIn(other["id"], fake.points)

    def test_prune_cap_refuses_mass_delete_without_force(self):
        pts = [self._hp(f"g{i}", "gone") for i in range(3)] + [self._hp("m1", "kept")]
        fake = FakeQdrant(pts)
        db = self._db([("m1", "kept")])
        self.assertEqual(self._run(db, fake, argv=["--prune"], **self.HOST), 1)
        self.assertEqual(fake.deletes(), [])
        self.assertEqual(self._run(db, fake, argv=["--prune", "--force-prune"], **self.HOST), 0)
        self.assertEqual(set(fake.points), {mq.stable_num_id("m1")})

    def test_edited_memory_is_reembedded(self):
        fake = FakeQdrant([_point("m1", "old text")])
        rc = self._run(self._db([("m1", "new text")]), fake)
        self.assertEqual(rc, 0)
        self.assertEqual(fake.points[mq.stable_num_id("m1")]["payload"]["content"], "new text")

    def test_missing_db_fails_without_touching_qdrant(self):
        fake = FakeQdrant([_point("m1", "kept")])
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        missing = os.path.join(tmpdir.name, "mnemosyne.db")
        rc = self._run(missing, fake)
        self.assertEqual(rc, 1)
        self.assertFalse(os.path.exists(missing), "sync must not create an empty DB")
        self.assertEqual(fake.deletes(), [])
        self.assertIn(mq.stable_num_id("m1"), fake.points)

    def test_db_without_memory_tables_does_not_prune(self):
        fake = FakeQdrant([_point("m1", "kept")])
        rc = self._run(self._db([], tables=False), fake)
        self.assertEqual(rc, 1)
        self.assertEqual(fake.deletes(), [])


if __name__ == "__main__":
    unittest.main()
