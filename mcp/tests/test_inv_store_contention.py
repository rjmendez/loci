import json
import multiprocessing as mp
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import fcntl
import pytest

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import inv_store  # noqa: E402
import server  # noqa: E402


def _json(result: str) -> dict:
    try:
        return json.loads(result)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"Tool returned non-JSON: {result!r}") from exc


@contextmanager
def _held_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextmanager
def _short_lock_timeout(timeout_s: float = 0.05):
    orig_timeout = inv_store._STORE_LOCK_TIMEOUT_S
    orig_initial = inv_store._STORE_LOCK_INITIAL_BACKOFF_S
    orig_max = inv_store._STORE_LOCK_MAX_BACKOFF_S
    inv_store._STORE_LOCK_TIMEOUT_S = timeout_s
    inv_store._STORE_LOCK_INITIAL_BACKOFF_S = 0.01
    inv_store._STORE_LOCK_MAX_BACKOFF_S = 0.01
    try:
        yield
    finally:
        inv_store._STORE_LOCK_TIMEOUT_S = orig_timeout
        inv_store._STORE_LOCK_INITIAL_BACKOFF_S = orig_initial
        inv_store._STORE_LOCK_MAX_BACKOFF_S = orig_max


@pytest.fixture
def isolated_store(monkeypatch):
    mem_tmp = tempfile.TemporaryDirectory()
    code_tmp = tempfile.TemporaryDirectory()
    old_memory_dir = server.MEMORY_DIR
    old_code_root = os.environ.get("LOCI_CODE_ROOT")
    monkeypatch.setattr(server, "MEMORY_DIR", Path(mem_tmp.name))
    os.environ["LOCI_CODE_ROOT"] = code_tmp.name
    try:
        yield Path(mem_tmp.name)
    finally:
        server.MEMORY_DIR = old_memory_dir
        if old_code_root is None:
            os.environ.pop("LOCI_CODE_ROOT", None)
        else:
            os.environ["LOCI_CODE_ROOT"] = old_code_root
        code_tmp.cleanup()
        mem_tmp.cleanup()


def _start_investigation(inv_id: str) -> str:
    res = _json(server.investigation_start(investigation_id=inv_id, title="lock audit"))
    assert res["status"] == "created"
    return inv_id


def _store_finding(inv_id: str, text: str = "seed finding", **kwargs) -> str:
    res = _json(server.investigation_store(inv_id, "observed", text, "unit-test", **kwargs))
    assert res.get("stored"), res
    return str(res["finding_id"])


def _assert_retryable_busy(result: dict, **expected):
    assert result.get("error") == "busy", result
    assert result.get("retryable") is True, result
    for key, value in expected.items():
        assert result.get(key) == value, result


def test_contract_declare_contention_returns_retryable_busy(isolated_store):
    inv_id = _start_investigation("contract-busy")
    with _short_lock_timeout(), _held_lock(server._inv_dir(inv_id) / "findings.jsonl"):
        result = _json(server.contract_declare(inv_id, "Invoice", "producer", '{"id": "str"}'))
    _assert_retryable_busy(result, investigation_id=inv_id, entity="Invoice", role="producer")


def test_wiring_obligation_declare_contention_returns_retryable_busy(isolated_store):
    inv_id = _start_investigation("wiring-declare-busy")
    with _short_lock_timeout(), _held_lock(server._inv_dir(inv_id) / "findings.jsonl"):
        result = _json(
            server.wiring_obligation_declare(inv_id, "Publisher", "send", "emit the event")
        )
    _assert_retryable_busy(result, investigation_id=inv_id, class_name="Publisher", method_name="send")


def test_wiring_obligation_resolve_contention_returns_retryable_busy(isolated_store):
    inv_id = _start_investigation("wiring-resolve-busy")
    finding_id = _json(
        server.wiring_obligation_declare(inv_id, "Publisher", "send", "emit the event")
    )["finding_id"]
    with _short_lock_timeout(), _held_lock(server._inv_dir(inv_id) / "findings.jsonl"):
        result = _json(server.wiring_obligation_resolve(inv_id, finding_id, "confirmed in publisher.py:12"))
    _assert_retryable_busy(result, investigation_id=inv_id, finding_id=finding_id)


def test_memory_retract_contention_returns_retryable_busy(isolated_store):
    inv_id = _start_investigation("retract-busy")
    finding_id = _store_finding(inv_id)
    retractions_path = server._inv_dir(inv_id) / "retractions.jsonl"
    with _short_lock_timeout(), _held_lock(retractions_path):
        result = _json(server.memory_retract(inv_id, finding_id, dry_run=False))
    _assert_retryable_busy(result, investigation_id=inv_id, target=finding_id)
    assert server._read_jsonl(retractions_path) == []


def test_memory_restore_contention_returns_retryable_busy(isolated_store):
    inv_id = _start_investigation("restore-busy")
    finding_id = _store_finding(inv_id)
    applied = _json(server.memory_retract(inv_id, finding_id, dry_run=False))
    assert applied["applied"] is True
    retractions_path = server._inv_dir(inv_id) / "retractions.jsonl"
    with _short_lock_timeout(), _held_lock(retractions_path):
        result = _json(server.memory_restore(inv_id, finding_id=finding_id))
    _assert_retryable_busy(result, investigation_id=inv_id, finding_id=finding_id)
    active = [row for row in server._read_jsonl(retractions_path) if row.get("active") is True]
    inactive = [row for row in server._read_jsonl(retractions_path) if row.get("active") is False]
    assert len(active) == 1
    assert inactive == []


@pytest.mark.parametrize(
    ("tool_name", "call"),
    [
        ("memory_promote", lambda inv_id, finding_id: server.memory_promote(inv_id, finding_id, "hot")),
        ("memory_demote", lambda inv_id, finding_id: server.memory_demote(inv_id, finding_id, "cold")),
    ],
)
def test_memory_tier_change_contention_returns_retryable_busy(isolated_store, tool_name, call):
    inv_id = _start_investigation(f"{tool_name}-busy")
    tier = "cold" if tool_name == "memory_promote" else "warm"
    finding_id = _store_finding(inv_id, tier=tier)
    with _short_lock_timeout(), _held_lock(server._inv_dir(inv_id) / ".lock"):
        result = _json(call(inv_id, finding_id))
    _assert_retryable_busy(result, investigation_id=inv_id, finding_id=finding_id)


def test_audit_log_global_contention_returns_retryable_busy(isolated_store):
    inv_id = _start_investigation("audit-busy")
    audit_path = server.MEMORY_DIR.parent / "audit" / f"{server.datetime.now(server.timezone.utc).strftime('%Y-%m-%d')}.jsonl"
    with _short_lock_timeout(), _held_lock(audit_path):
        result = _json(server.audit_log("unit_test", "{}", "output", investigation_id=inv_id))
    _assert_retryable_busy(result, investigation_id=inv_id, tool="unit_test")


def test_wiring_obligation_declare_reloads_manifest_under_lock(isolated_store, monkeypatch):
    inv_id = _start_investigation("manifest-race")

    original_load = server._load_manifest_fresh
    original_save = server._save_manifest
    original_timeout = inv_store._STORE_LOCK_TIMEOUT_S
    load_count = 0
    load_count_lock = threading.Lock()
    first_save_entered = threading.Event()
    release_first_save = threading.Event()
    inv_store._STORE_LOCK_TIMEOUT_S = 15.0
    try:
        def _wrapped_load(investigation_id: str):
            nonlocal load_count
            with load_count_lock:
                load_count += 1
            return original_load(investigation_id)

        def _wrapped_save(manifest: dict) -> None:
            first_save_entered.set()
            if not release_first_save.wait(1.0):
                raise AssertionError("timed out waiting to release manifest save")
            original_save(manifest)

        monkeypatch.setattr(server, "_load_manifest_fresh", _wrapped_load)
        monkeypatch.setattr(server, "_save_manifest", _wrapped_save)

        results: dict[str, dict] = {}

        def _declare(suffix: str) -> None:
            results[suffix] = _json(
                server.wiring_obligation_declare(inv_id, f"Publisher{suffix}", f"send{suffix}", "emit the event")
            )

        threads = [threading.Thread(target=_declare, args=(suffix,)) for suffix in ("a", "b")]
        for thread in threads:
            thread.start()

        assert first_save_entered.wait(1.0), "expected the first manifest save to block"
        time.sleep(0.1)
        assert load_count == 1, f"second writer should not load a stale manifest snapshot (count={load_count})"

        release_first_save.set()
        for thread in threads:
            thread.join(15.0)
        if any(thread.is_alive() for thread in threads):
            release_first_save.set()
            for thread in threads:
                thread.join(15.0)
        assert all(not thread.is_alive() for thread in threads), "manifest writers must finish"
        assert all(row.get("stored") for row in results.values()), results

        manifest = server._load_manifest_fresh(inv_id)
        assert manifest is not None
        assert manifest["finding_counts"]["gap"] == 2
    finally:
        inv_store._STORE_LOCK_TIMEOUT_S = original_timeout


def test_memory_promote_preserves_concurrent_append_during_rewrite(isolated_store, monkeypatch):
    inv_id = _start_investigation("rewrite-race")
    finding_id = _store_finding(inv_id, tier="cold")
    # promote reports ok only once the point is indexed; stub a landed upsert.
    monkeypatch.setattr(server, "_qdrant_upsert", lambda *a, **k: True)
    findings_path = server._inv_dir(inv_id) / "findings.jsonl"
    original_timeout = inv_store._STORE_LOCK_TIMEOUT_S
    inv_store._STORE_LOCK_TIMEOUT_S = 15.0
    append_started = threading.Event()
    append_finished = threading.Event()
    blocked_while_rewriting = []

    def _append_late() -> None:
        append_started.set()
        with server._locked_file(server._inv_dir(inv_id) / ".lock", "a+", exclusive=True):
            server._append_jsonl(findings_path, {
                "id": "late-append",
                "investigation_id": inv_id,
                "record_type": "observed",
                "type": "observed",
                "text": "late concurrent append",
                "tier": "warm",
            })
        append_finished.set()

    append_thread = threading.Thread(target=_append_late)

    # Deterministic interleave: the rewrite has read findings.jsonl and is about
    # to write it back. Start the appender exactly there and give it the chance
    # to append. Under the lock it cannot, and the rewrite must not wait on it;
    # without the lock it appends at once and the rewrite then overwrites it.
    # (The old version slept 50 ms before appending, after the rewrite had
    # finished, so removing the lock passed 5/5.)
    real_atomic_write = inv_store._atomic_write_text

    def _write_after_interleave(path, data):
        if Path(path) == findings_path and not append_thread.is_alive() and not append_finished.is_set():
            append_thread.start()
            assert append_started.wait(5.0)
            blocked_while_rewriting.append(not append_finished.wait(1.0))
        return real_atomic_write(path, data)

    monkeypatch.setattr(inv_store, "_atomic_write_text", _write_after_interleave)

    try:
        result = _json(server.memory_promote(inv_id, finding_id, "hot"))
        assert result["ok"] is True, result
        assert blocked_while_rewriting == [True], "the append ran inside the rewrite window"

        assert append_finished.wait(15.0), "concurrent append must complete"
        append_thread.join(15.0)
        assert not append_thread.is_alive(), "append writer must finish"

        rows = server._read_jsonl(findings_path)
        ids = {row.get("id") for row in rows}
        assert finding_id in ids
        assert "late-append" in ids
        promoted = next(row for row in rows if row.get("id") == finding_id)
        assert promoted.get("tier") == "hot"
    finally:
        inv_store._STORE_LOCK_TIMEOUT_S = original_timeout


def _append_worker(root_str: str, path_str: str, barrier, proc_idx: int, writes_per_proc: int) -> None:
    import inv_store as _inv_store

    root = Path(root_str)
    path = Path(path_str)
    _inv_store._get_memory_dir = lambda: root
    _inv_store._STORE_LOCK_TIMEOUT_S = 10.0
    payload = "x" * 10000
    barrier.wait()
    for seq in range(writes_per_proc):
        _inv_store._append_jsonl(path, {"proc": proc_idx, "seq": seq, "payload": payload})


@pytest.mark.skipif(sys.platform == "win32", reason="requires fork for shared barrier timing")
def test_append_jsonl_survives_real_process_contention(tmp_path, monkeypatch):
    root = tmp_path / "memory"
    root.mkdir()
    monkeypatch.setattr(inv_store, "_get_memory_dir", lambda: root)
    monkeypatch.setattr(inv_store, "_STORE_LOCK_TIMEOUT_S", 10.0)
    path = inv_store._inv_dir("stress-case") / "findings.jsonl"

    proc_count = 6
    writes_per_proc = 120
    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(proc_count)
    procs = [
        ctx.Process(
            target=_append_worker,
            args=(str(root), str(path), barrier, proc_idx, writes_per_proc),
        )
        for proc_idx in range(proc_count)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(30)
    alive = [proc.pid for proc in procs if proc.is_alive()]
    if alive:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
                proc.join(5)
    assert not alive, f"writers hung under contention: {alive}"
    assert all(proc.exitcode == 0 for proc in procs), [proc.exitcode for proc in procs]

    raw_lines = path.read_text().splitlines()
    assert len(raw_lines) == proc_count * writes_per_proc
    rows = [json.loads(line) for line in raw_lines]
    unique = {(row["proc"], row["seq"]) for row in rows}
    assert len(unique) == proc_count * writes_per_proc
