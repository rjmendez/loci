"""Tests for the per-operation cross-process lease in graph.kuzu_store.KuzuStore.

The old store opened Kuzu read-write at construction and held it for the whole
process life, so N Loci servers contended on Kuzu's single-writer lock and all but
one degraded. These tests verify the new contract: nothing is held between ops, so
two store instances (standing in for two server processes) coexist, and a contended
lease fails open within a bounded wait instead of hanging.
"""

from __future__ import annotations

import fcntl
import os
import threading

import pytest

kuzu = pytest.importorskip("kuzu")

from graph import kuzu_store as kz
from graph.kuzu_store import KuzuStore


def test_no_lock_held_between_ops_two_instances(tmp_path):
    """Two KuzuStore instances on the same DB both read+write — proving neither holds
    the writer lock between operations (the whole point of the fix)."""
    path = str(tmp_path / "graphdb")
    a = KuzuStore(path)
    b = KuzuStore(path)
    assert a.available() and b.available()

    assert a.upsert_investigation("inv1", "from A") is True
    # B writes immediately after A's op returned — would fail if A still held RW.
    assert b.upsert_investigation("inv2", "from B") is True
    assert a.upsert_finding({"id": "f1", "investigation": "inv1", "text": "x"}) is True

    # Each instance sees the other's committed writes (fresh leased read sessions).
    ids = {r[0] for r in b.related_investigations("inv1")} if False else None
    rows = a._rows("MATCH (i:Investigation) RETURN i.id ORDER BY i.id")
    assert [r[0] for r in rows] == ["inv1", "inv2"]


def test_write_visible_to_a_fresh_instance(tmp_path):
    """A write committed by one instance is visible to a brand-new instance opening a
    read-only session — proves the write session flushes on close (durability)."""
    path = str(tmp_path / "graphdb")
    KuzuStore(path).upsert_finding({"id": "f9", "investigation": "invZ", "text": "hello"})
    fresh = KuzuStore(path)
    rows = fresh._rows("MATCH (f:Finding {id:'f9'}) RETURN f.text")
    assert rows and rows[0][0] == "hello"


def test_concurrent_readers_share(tmp_path):
    """Many reader sessions run concurrently (shared lease) without error."""
    path = str(tmp_path / "graphdb")
    s = KuzuStore(path)
    s.upsert_investigation("invR", "reader test")
    errors: list = []

    def reader():
        try:
            for _ in range(3):
                assert s._rows("MATCH (i:Investigation) RETURN count(i)")[0][0] >= 1
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_failopen_when_lease_is_held(tmp_path, monkeypatch):
    """When another holder owns the lease exclusively, a write fails open (False)
    within the bounded timeout — never hangs — and recovers once the lease frees."""
    monkeypatch.setattr(kz, "_LEASE_TIMEOUT_S", 0.4)  # keep the test fast
    monkeypatch.setattr(kz, "_LEASE_POLL_S", 0.02)
    path = str(tmp_path / "graphdb")
    s = KuzuStore(path)
    s.upsert_investigation("seed", "seed")           # create the lease file + schema

    # Hold the lease EXCLUSIVELY from an independent fd (stands in for another process).
    held = os.open(s._lease_path, os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(held, fcntl.LOCK_EX)
    try:
        assert s.writable_probe() is False           # contended -> not writable
        assert s.upsert_investigation("blocked", "nope") is False   # fails open, bounded
        assert s._rows("MATCH (i) RETURN i") == []   # read also fails open under EX hold
    finally:
        fcntl.flock(held, fcntl.LOCK_UN)
        os.close(held)

    # Lease freed -> the very next op succeeds (self-heals, no restart).
    assert s.writable_probe() is True
    assert s.upsert_investigation("recovered", "ok") is True


def test_holder_pid_is_stamped(tmp_path):
    """A completed write stamps this process's PID in the lease file (diagnostics)."""
    path = str(tmp_path / "graphdb")
    s = KuzuStore(path)
    s.upsert_investigation("p", "pid test")
    assert s.lock_holder_pid() == os.getpid()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
