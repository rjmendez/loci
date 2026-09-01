"""Bounds on the accumulating structures from #107.

The audit listed seven items. Three of the target artifacts do not exist on this
host yet (`.emb_cache.npz`, the mlops loop state, and the *default* sync-cache
dir), so those bounds are preventive. Two are live: the session-sync cache the
Stop hook actually writes (`LOCI_SYNC_CACHE`, 7 files), and the daemon's
unbounded recv().
"""
import asyncio
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hooks"))

from pathlib import Path  # noqa: E402

from memcheck import cli  # noqa: E402
from memcheck.backend import InMemoryBackend  # noqa: E402
from memcheck.verdict import new_verdict  # noqa: E402


def _emb(i, n=64):
    """One-hot: distinct verdicts must be mutually DISSIMILAR or the backend
    coalesces them by cosine similarity and nothing accumulates to evict.
    [i,i,i,i]-style vectors are all parallel — cosine 1.0 — which is how the
    first draft of this test measured coalescing instead of eviction."""
    v = [0.0] * n
    v[i % n] = 1.0
    return v


def _v(i):
    return new_verdict(subject_kind="memory", subject_signature=f"s{i}",
                       subject_excerpt="", verdict_type="unsupported_observed",
                       decision="warn", confidence=0.5, rationale="", source="rule")


class InMemoryBackendBoundTest(unittest.TestCase):
    """The reference backend documents the semantics the persistent ones mirror.
    Those evict; a list that only grows is not the same semantics."""

    def test_verdicts_are_capped(self):
        b = InMemoryBackend(max_verdicts=10)
        async def run():
            for i in range(25):
                await b.record_with_embedding(_v(i), _emb(i))
        asyncio.run(run())
        self.assertEqual(len(b._verdicts), 10)

    def test_eviction_drops_the_oldest(self):
        b = InMemoryBackend(max_verdicts=5)
        async def run():
            for i in range(12):
                await b.record_with_embedding(_v(i), _emb(i))
        asyncio.run(run())
        kept = [v.subject_signature for v in b._verdicts]
        self.assertEqual(kept, [f"s{i}" for i in range(7, 12)])

    def test_embeddings_are_evicted_with_their_verdicts(self):
        """Capping the list alone would swap one unbounded structure for another."""
        b = InMemoryBackend(max_verdicts=5)
        async def run():
            for i in range(20):
                await b.record_with_embedding(_v(i), _emb(i))
        asyncio.run(run())
        self.assertLessEqual(len(b._embeddings), 5)
        self.assertEqual(set(b._embeddings), {v.id for v in b._verdicts})

    def test_zero_disables_the_bound(self):
        b = InMemoryBackend(max_verdicts=0)
        async def run():
            for i in range(30):
                await b.record_with_embedding(_v(i), _emb(i))
        asyncio.run(run())
        self.assertEqual(len(b._verdicts), 30)


class DaemonReadTimeoutTest(unittest.TestCase):
    """A client that connects and never speaks must not own a worker forever."""

    def test_a_silent_client_does_not_hang_the_worker(self):
        from memcheck import daemon
        sock_dir = tempfile.mkdtemp()
        path = os.path.join(sock_dir, "s.sock")
        ready = threading.Event()
        srv = {}

        def run():
            s = daemon.MemcheckDaemon(  # type: ignore[call-arg]
                __import__("pathlib").Path(path),
                engine_factory=lambda: None,
            )
            srv["s"] = s
            ready.set()
            s.serve_forever(poll_interval=0.05)

        t = threading.Thread(target=run, daemon=True)
        t.start()
        self.assertTrue(ready.wait(5), "daemon did not start")
        try:
            with unittest.mock.patch.object(daemon, "_REQUEST_TIMEOUT_S", 0.3):
                c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                c.settimeout(5)
                c.connect(path)
                # connect, then say nothing and never half-close
                t0 = time.monotonic()
                try:
                    c.recv(4096)          # server should close after its timeout
                except (socket.timeout, OSError):
                    pass
                elapsed = time.monotonic() - t0
                c.close()
            self.assertLess(elapsed, 4.0,
                            "worker stayed blocked on a silent client")
        finally:
            srv["s"].shutdown()
            srv["s"].server_close()


class MemcheckAuditLogBoundTests(unittest.TestCase):
    """The PreToolUse audit log must rotate like its sibling hook's does.

    `_append_audit_line` runs once per tool call and had no size check, so the
    log grew forever while scripts/hooks/pre_tool_grounding.py — the other hook,
    same host, same kind of log — stayed flat under MAX_AUDIT_BYTES.
    """

    def _append_into(self, tmpdir: str, records: int, max_bytes: int, keep_bytes: int) -> Path:
        path = Path(tmpdir) / "memcheck-audit.jsonl"
        with unittest.mock.patch.dict(os.environ, {"MEMCHECK_AUDIT_LOG": str(path)}), \
                unittest.mock.patch.object(cli, "MAX_AUDIT_BYTES", max_bytes), \
                unittest.mock.patch.object(cli, "KEEP_AUDIT_BYTES", keep_bytes):
            for i in range(records):
                cli._append_audit_line({"i": i, "pad": "x" * 200})
        return path

    def test_append_rotates_once_past_the_cap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._append_into(tmpdir, records=200, max_bytes=4096, keep_bytes=2048)
            size = path.stat().st_size
            lines = path.read_text(encoding="utf-8").splitlines()

        # Without a rotation guard 200 records is ~45 KB and only ever grows.
        self.assertLessEqual(size, 4096 + 512)
        # Rotation slices bytes, so the retained head must not be half a record.
        for line in lines:
            json.loads(line)
        # The newest record is always kept.
        self.assertEqual(json.loads(lines[-1])["i"], 199)

    def test_append_below_the_cap_keeps_everything(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._append_into(tmpdir, records=5, max_bytes=4096, keep_bytes=2048)
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual([json.loads(line)["i"] for line in lines], [0, 1, 2, 3, 4])


if __name__ == "__main__":
    import unittest.mock  # noqa: F401
    unittest.main()
