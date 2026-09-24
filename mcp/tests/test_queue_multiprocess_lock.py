"""The coordination queue's read-modify-write paths hold the investigation lock.

The queue lives in manifest.json. Before the lock, concurrent enqueues from
separate processes each read the manifest, appended their item and wrote it
back, so every write but the last was lost; two sessions claiming the same
item could both be told they own it. These tests race real OS processes
(fork), because the in-process threading lock never covered that case.
"""
import json
import multiprocessing
import sys

import pytest

import server

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs fork + flock")

_N_PROCS = 6
_PER_PROC = 8


def _enqueue_worker(inv_id, proc_idx, barrier, out):
    barrier.wait()
    results = []
    for i in range(_PER_PROC):
        results.append(json.loads(server.investigation_queue_enqueue(
            investigation_id=inv_id, item_id=f"p{proc_idx}-i{i}",
        )))
    out.put(results)


def _claim_worker(inv_id, proc_idx, barrier, out):
    barrier.wait()
    out.put(json.loads(server.investigation_queue_claim(
        investigation_id=inv_id, item_id="contested", owner_session=f"session-{proc_idx}",
    )))


def _run(target, inv_id):
    ctx = multiprocessing.get_context("fork")
    barrier = ctx.Barrier(_N_PROCS)
    out = ctx.Queue()
    procs = [ctx.Process(target=target, args=(inv_id, n, barrier, out)) for n in range(_N_PROCS)]
    for p in procs:
        p.start()
    results = [out.get(timeout=60) for _ in procs]
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    return results


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "MEMORY_DIR", tmp_path)
    return tmp_path


def _queue_ids(inv_id):
    status = json.loads(server.investigation_queue_status(investigation_id=inv_id))
    return {item["id"] for item in status["queue"]}


def test_concurrent_enqueues_from_many_processes_lose_nothing(store):
    inv_id = "q-mp-enqueue"
    server.investigation_start(investigation_id=inv_id, title="multiprocess enqueue")
    batches = _run(_enqueue_worker, inv_id)
    accepted = {r["item"]["id"] for batch in batches for r in batch if r.get("queued")}
    busy = [r for batch in batches for r in batch if r.get("error") == "busy"]
    # Every enqueue that reported success must be in the queue afterwards.
    server.inv_store._manifest_cache.clear()
    on_disk = _queue_ids(inv_id)
    assert accepted <= on_disk, f"lost {len(accepted - on_disk)} acknowledged enqueues"
    # And with a bounded lock, a contended call is either accepted or told to retry.
    assert len(accepted) + len(busy) == _N_PROCS * _PER_PROC


def test_contested_claim_from_many_processes_has_one_winner(store):
    inv_id = "q-mp-claim"
    server.investigation_start(investigation_id=inv_id, title="multiprocess claim")
    server.investigation_queue_enqueue(investigation_id=inv_id, item_id="contested")
    results = _run(_claim_worker, inv_id)
    winners = [r for r in results if r.get("claimed")]
    assert len(winners) == 1, f"{len(winners)} sessions were told they own the item"
    server.inv_store._manifest_cache.clear()
    status = json.loads(server.investigation_queue_status(investigation_id=inv_id, item_id="contested"))
    item = status.get("item") or status["queue"][0]
    assert item["owner_session"] == winners[0]["item"]["owner_session"]


def test_queue_writes_take_the_investigation_lock(store, monkeypatch):
    """Each mutating queue call enters _locked_file on <inv>/.lock."""
    inv_id = "q-lock-seen"
    server.investigation_start(investigation_id=inv_id, title="lock observed")
    queue_globals = server.investigation_queue_claim.__globals__
    real = queue_globals["_locked_file"]
    seen = []

    def spy(path, *a, **k):
        seen.append(str(path))
        return real(path, *a, **k)

    monkeypatch.setitem(queue_globals, "_locked_file", spy)
    server.investigation_queue_enqueue(investigation_id=inv_id, item_id="a")
    server.investigation_queue_claim(investigation_id=inv_id, item_id="a", owner_session="s1")
    server.investigation_queue_release(investigation_id=inv_id, item_id="a", owner_session="s1")
    server.investigation_queue_claim(investigation_id=inv_id, item_id="a", owner_session="s2")
    server.investigation_queue_complete(investigation_id=inv_id, item_id="a", owner_session="s2")
    assert len(seen) == 5 and all(p.endswith(f"{inv_id}/.lock") for p in seen), seen


def test_queue_lock_busy_is_retryable_not_a_silent_write(store, monkeypatch):
    inv_id = "q-busy"
    server.investigation_start(investigation_id=inv_id, title="busy")
    queue_globals = server.investigation_queue_claim.__globals__
    StoreBusyError = queue_globals["StoreBusyError"]

    class _Busy:
        def __init__(self, path, *a, **k):
            self.path = path

        def __enter__(self):
            raise StoreBusyError(self.path, timeout_s=0.01, operation="queue")

        def __exit__(self, *exc):
            return False

    monkeypatch.setitem(queue_globals, "_locked_file", _Busy)
    r = json.loads(server.investigation_queue_enqueue(investigation_id=inv_id, item_id="x"))
    assert r["error"] == "busy" and r["retryable"] is True
    monkeypatch.undo()
    monkeypatch.setattr(server, "MEMORY_DIR", store)
    assert "x" not in _queue_ids(inv_id)
