"""The bridge's watermark must attest to delivery, not to an HTTP code.

Two ways the bridge used to record work it had not done, both of which are permanent:

1. `_broadcast_memory` returned status 'ok' on any 200 from the LOCAL server. The local
   server answers 200 with per-peer status 'skipped' when PEER_A2A_URLS is unset or a
   peer has no token / a bad TOTP seed, and 'http_401' when a peer rejects the code.
   store_local is False, so in every one of those cases the memory reached nobody — yet
   the id went into sent_ids and the watermark advanced past it, and no later tick
   retries a memory that is behind both gates.

2. `_fetch_recent_memories` returned [] for "could not read the DB" exactly as it does
   for "the window was quiet", and run() advances the watermark on []. One tick with a
   missing/unreadable MNEMOSYNE_DB drops every memory written during the outage.

These pin the failure direction (no delivery => no stamp) and the success direction
(a genuinely quiet window still advances, or the bridge would re-scan forever).
"""

import asyncio
import importlib.util
import os
import pathlib
import sqlite3
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("LOCI_ENV_FILE", "/nonexistent-env-file-for-tests")
os.environ.setdefault("LOCI_A2A_TOKEN", "t")

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "bridge_delivery_impl", _SCRIPTS / "a2a_context_bridge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bridge = _load()

MEM = {"id": "m1", "content": "hello", "importance": 0.9}


class _FakeResp:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def json(self):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Stands in for aiohttp.ClientSession; .post() is an async context manager."""

    def __init__(self, status, payload):
        self._status = status
        self._payload = payload

    def post(self, *a, **kw):
        return _FakeResp(self._status, self._payload)


def _reply(broadcast, peers_count):
    return {"result": {"output": {"broadcast": broadcast,
                                  "peers_count": peers_count,
                                  "stored_locally": False}}}


def _send(broadcast, peers_count, status=200):
    session = _FakeSession(status, _reply(broadcast, peers_count))
    return asyncio.run(bridge._broadcast_memory(session, MEM, dry_run=False))


class TestBroadcastOutcome(unittest.TestCase):
    def test_a_peer_that_accepted_is_ok(self):
        r = _send([{"peer": "p1", "status": "ok"}], 1)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["peers_ok"], 1)

    def test_one_ok_among_failures_is_still_ok(self):
        r = _send([{"peer": "p1", "status": "ok"},
                   {"peer": "p2", "status": "http_401"}], 2)
        self.assertEqual(r["status"], "ok")

    def test_no_peers_configured_is_not_ok(self):
        # The server's own shape when PEER_A2A_URLS is unset.
        r = _send([{"peer": "none", "status": "skipped",
                    "error": "PEER_A2A_URLS not set"}], 0)
        self.assertNotEqual(r["status"], "ok")
        self.assertEqual(r["status"], "no_peers")

    def test_every_peer_skipped_is_not_ok(self):
        r = _send([{"peer": "p1", "status": "skipped", "error": "no token configured"},
                   {"peer": "p2", "status": "skipped", "error": "invalid TOTP seed"}], 2)
        self.assertNotEqual(r["status"], "ok")
        self.assertEqual(r["peers_ok"], 0)

    def test_every_peer_401_is_not_ok(self):
        # The recorded mesh failure: a drifted TOTP seed 401s every hop.
        r = _send([{"peer": "p1", "status": "http_401"}], 1)
        self.assertNotEqual(r["status"], "ok")

    def test_the_skip_reason_is_reported(self):
        r = _send([{"peer": "p1", "status": "http_401"}], 1)
        self.assertIn("http_401", r.get("error", ""))

    def test_non_200_is_still_a_failure(self):
        r = _send([], 0, status=503)
        self.assertEqual(r["status"], "http_503")


class TestWatermarkHeldWhenNothingLanded(unittest.TestCase):
    def _run_with(self, result):
        saved = {}
        state = {"last_run": "2026-07-01T00:00:00", "sent_ids": []}

        async def fake_broadcast(session, mem, dry_run):
            return dict(result, id=mem["id"])

        with mock.patch.object(bridge, "_load_state", return_value=state), \
                mock.patch.object(bridge, "_save_state", side_effect=saved.update), \
                mock.patch.object(bridge, "_fetch_recent_memories", return_value=[dict(MEM)]), \
                mock.patch.object(bridge, "_broadcast_memory", side_effect=fake_broadcast):
            asyncio.run(bridge.run(dry_run=False, verbose=False))
        return saved

    def test_undelivered_memory_does_not_advance_the_watermark(self):
        saved = self._run_with({"status": "no_peers", "peers_ok": 0})
        self.assertEqual(saved["last_run"], "2026-07-01T00:00:00")

    def test_undelivered_memory_is_not_marked_sent(self):
        saved = self._run_with({"status": "no_peers", "peers_ok": 0})
        self.assertEqual(saved["sent_ids"], [])
        self.assertEqual(saved["last_fail"], 1)

    def test_delivered_memory_does_advance_the_watermark(self):
        saved = self._run_with({"status": "ok", "peers_ok": 1})
        self.assertNotEqual(saved["last_run"], "2026-07-01T00:00:00")
        self.assertEqual(saved["sent_ids"], ["m1"])


SCHEMA = ("CREATE TABLE {t} (id TEXT, content TEXT, importance REAL, "
          "created_at TEXT, source TEXT)")


class TestUnreadableDbIsNotAnEmptyWindow(unittest.TestCase):
    def _db(self, tables=("memories", "working_memory")):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        con = sqlite3.connect(path)
        for t in tables:
            con.execute(SCHEMA.format(t=t))
        con.commit()
        con.close()
        return path

    def _fetch(self, path):
        with mock.patch.object(bridge, "MNEMOSYNE_DB", path):
            return bridge._fetch_recent_memories("2026-07-01T00:00:00", 0.5, 20)

    def test_quiet_window_is_an_empty_list(self):
        self.assertEqual(self._fetch(self._db()), [])

    def test_missing_db_is_none(self):
        self.assertIsNone(self._fetch("/nonexistent/mnemosyne.db"))

    def test_db_with_no_memory_table_is_none(self):
        # Both per-table queries raise OperationalError and are skipped; the old code
        # returned [] from here, which reads as "nothing new" and moves the watermark.
        self.assertIsNone(self._fetch(self._db(tables=("something_else",))))

    def _run_with_fetch(self, value):
        saved = {}
        state = {"last_run": "2026-07-01T00:00:00", "sent_ids": []}
        with mock.patch.object(bridge, "_load_state", return_value=state), \
                mock.patch.object(bridge, "_save_state", side_effect=saved.update), \
                mock.patch.object(bridge, "_fetch_recent_memories", return_value=value):
            asyncio.run(bridge.run(dry_run=False, verbose=False))
        return saved

    def test_run_writes_no_state_when_the_db_could_not_be_read(self):
        self.assertEqual(self._run_with_fetch(None), {})

    def test_run_still_advances_on_a_genuinely_quiet_window(self):
        saved = self._run_with_fetch([])
        self.assertNotEqual(saved.get("last_run"), "2026-07-01T00:00:00")


if __name__ == "__main__":
    unittest.main()
