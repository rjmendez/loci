"""Tests for the loci_health self-diagnosis tool + the embed warm-ping (#2, #7).

Both are read-only + fail-open: no live backends are required. Backend reachability
is stubbed via backends._alive; the warm-ping is exercised with an injected embed_fn.
"""
import json
import sys
import threading
import time
from pathlib import Path

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import backends  # noqa: E402
import embed_ops  # noqa: E402
import server  # noqa: E402


_EXPECTED_KEYS = {
    "code_version", "ladybug", "ollama_reachable", "vllm_reachable",
    "qdrant_reachable", "embed_model", "rerank_model", "warm",
}


def test_loci_health_returns_expected_keys():
    out = json.loads(server.loci_health())
    assert isinstance(out, dict)
    # Subset (not ==) so additive fields don't break this contract.
    assert _EXPECTED_KEYS <= set(out.keys())
    # the graph-store health state is one of the documented values
    assert out["ladybug"] in (
        "available", "contended", "unavailable", "latched", "backoff"
    )
    # booleans for reachability, strings for model names / version
    for k in ("ollama_reachable", "vllm_reachable", "qdrant_reachable", "warm"):
        assert isinstance(out[k], bool)
    for k in ("code_version", "embed_model", "rerank_model"):
        assert isinstance(out[k], str)


def test_loci_health_probes_independent_and_fail_open(monkeypatch):
    # _alive raises for the ollama endpoint, is up for vllm, down for qdrant.
    def fake_alive(url, timeout=1.0):
        if "11434" in (url or ""):
            raise RuntimeError("boom: probe blew up")
        if "8000" in (url or ""):
            return True
        return False

    monkeypatch.setattr(backends, "_alive", fake_alive)
    # Resolvers now accept a probe_timeout arg (loci_health passes a short one).
    monkeypatch.setattr(backends, "ollama_url", lambda *a, **k: "http://localhost:11434")
    monkeypatch.setattr(backends, "vllm_url", lambda *a, **k: "http://localhost:8000")
    monkeypatch.setattr(backends, "qdrant", lambda: ("http://localhost:6333", ""))

    out = json.loads(server.loci_health())
    # The raising ollama probe must NOT mask the others (independent + fail-open).
    assert out["ollama_reachable"] is False   # swallowed -> default
    assert out["vllm_reachable"] is True
    assert out["qdrant_reachable"] is False
    assert _EXPECTED_KEYS <= set(out.keys())


def test_loci_health_never_raises_when_resolvers_throw(monkeypatch):
    # Every probe is independent and fail-open: the full key set survives any resolver raising.
    monkeypatch.setattr(backends, "ollama_url", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(backends, "vllm_url", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(backends, "qdrant", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    out = json.loads(server.loci_health())
    assert _EXPECTED_KEYS <= set(out.keys())


def test_loci_health_probes_are_bounded_short_timeout(monkeypatch):
    # Every probe, the resolvers' own included, must be short-timeout so a first call cannot block.
    for var in ("OLLAMA_BASE_URL", "OLLAMA_URL", "VLLM_BASE_URL", "QDRANT_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(backends, "_config", lambda: {})
    backends.ollama_url.cache_clear()
    backends.vllm_url.cache_clear()

    seen_timeouts = []

    def recording_alive(url, timeout=1.0):
        seen_timeouts.append(timeout)
        return False

    monkeypatch.setattr(backends, "_alive", recording_alive)

    out = json.loads(server.loci_health())

    # The resolvers' internal probe must actually run, not short-circuit.
    assert seen_timeouts, "expected reachability probes to have run"
    assert all(t <= 0.5 for t in seen_timeouts), (
        f"all loci_health probe timeouts must be <= 0.5s, got {seen_timeouts}")
    # Backends down -> all reachability False, full key set still returned (fail-open).
    assert out["ollama_reachable"] is False
    assert out["vllm_reachable"] is False
    assert out["qdrant_reachable"] is False
    assert _EXPECTED_KEYS <= set(out.keys())
    # Reset so a later real caller re-resolves cleanly.
    backends.ollama_url.cache_clear()
    backends.vllm_url.cache_clear()


def test_code_version_first_compute_is_thread_safe(monkeypatch):
    # Double-checked locking: concurrent cold callers spawn `git rev-parse` exactly once.
    import subprocess as _sp

    monkeypatch.setattr(server, "_code_version_cache", None)
    calls = []
    barrier = threading.Barrier(8)

    class _Result:
        returncode = 0
        stdout = "deadbeef\n"

    def fake_run(*a, **k):
        calls.append(1)
        # Slow enough that racers pile up on the lock before the cache is set.
        import time as _t
        _t.sleep(0.05)
        return _Result()

    monkeypatch.setattr(_sp, "run", fake_run)

    results = []

    def worker():
        barrier.wait()
        results.append(server._code_version())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1  # subprocess forked once despite 8 concurrent callers
    assert results == ["deadbeef"] * 8
    assert server._code_version_cache == "deadbeef"


def _reset_warm():
    with embed_ops._warm_lock:
        embed_ops._warm_started = False


def test_warm_fires_once_and_flips_warmed():
    _reset_warm()
    assert embed_ops.warmed() is False
    calls = []
    started = embed_ops.warm(embed_fn=lambda t: calls.append(t) or [])
    assert started is True
    assert embed_ops.warmed() is True
    # idempotent: a second call does not re-fire
    assert embed_ops.warm(embed_fn=lambda t: calls.append(t) or []) is False


def test_warm_is_best_effort_never_raises_when_endpoint_down():
    _reset_warm()
    ran = threading.Event()

    def boom(texts):
        ran.set()
        raise RuntimeError("endpoint down / cold")

    # warm() must return without raising even though the embed call raises.
    assert embed_ops.warm(embed_fn=boom) is True
    # the daemon thread actually ran and swallowed the exception (no crash).
    assert ran.wait(timeout=5) is True
    # warmed() reflects the fired state regardless of the underlying failure.
    assert embed_ops.warmed() is True


def test_warm_does_not_latch_when_thread_start_fails(monkeypatch):
    # A thread that fails to start must not latch _warm_started, or no call can retry.
    _reset_warm()

    class _BoomThread:
        def __init__(self, *a, **k):
            pass

        def start(self):
            raise RuntimeError("cannot spawn thread")

    monkeypatch.setattr(embed_ops.threading, "Thread", _BoomThread)
    assert embed_ops.warm(embed_fn=lambda t: []) is False
    assert embed_ops.warmed() is False

    # With threading restored, a subsequent call succeeds and latches.
    monkeypatch.undo()
    _reset_warm()
    assert embed_ops.warm(embed_fn=lambda t: []) is True
    assert embed_ops.warmed() is True


def test_health_initializes_the_lazy_graph_store_instead_of_reporting_unavailable(monkeypatch):
    """A healthy-but-not-yet-opened graph must report 'available', not 'unavailable'.

    Production shape: a fresh server process. The store singleton is created lazily on
    the first graph op, so before this fix the first loci_health of every process read
    'unavailable' for a perfectly good graph — and callers switch the code-graph lane
    off on that false negative.
    """
    class _HealthyStore:
        def readable_probe(self):
            return True

        def lock_holder_pid(self):
            return 4242

    # Fresh-process state: nothing initialized, nothing latched, no backoff pending.
    monkeypatch.setattr(server, "_ladybug_store", None)
    monkeypatch.setattr(server, "_ladybug_failed", False)
    monkeypatch.setattr(server, "_ladybug_last_attempt", 0.0)

    opened = []

    def fake_get(backfill=True):
        store = _HealthyStore()
        opened.append(store)
        monkeypatch.setattr(server, "_ladybug_store", store)
        return store

    monkeypatch.setattr(server, "_get_ladybug", fake_get)

    assert server._ladybug_health_state() == "available"
    assert opened, "health must attempt the lazy initialization, not read the global and give up"


def test_health_reports_latched_without_attempting_init(monkeypatch):
    """A permanent latch is a predetermined answer — settle it without opening anything."""
    attempts = []
    monkeypatch.setattr(server, "_ladybug_store", None)
    monkeypatch.setattr(server, "_ladybug_failed", True)
    monkeypatch.setattr(server, "_ladybug_last_attempt", 0.0)
    monkeypatch.setattr(server, "_get_ladybug", lambda *a, **k: attempts.append(k) or None)

    assert server._ladybug_health_state() == "latched"
    assert not attempts, "latched is predetermined; health must not try to open the store"


def test_health_reports_backoff_without_attempting_init(monkeypatch):
    """Likewise for the transient-failure backoff window."""
    attempts = []
    monkeypatch.setattr(server, "_ladybug_store", None)
    monkeypatch.setattr(server, "_ladybug_failed", False)
    monkeypatch.setattr(server, "_ladybug_last_attempt", time.monotonic())
    monkeypatch.setattr(server, "_get_ladybug", lambda *a, **k: attempts.append(k) or None)

    assert server._ladybug_health_state() == "backoff"
    assert not attempts, "inside the backoff window health must not hammer the open"


def test_health_opens_the_store_without_running_the_backfill(monkeypatch):
    """loci_health must stay a read: the backfill writes, and writes take the writer lease."""
    kwargs = {}

    def fake_get(backfill=True):
        kwargs["backfill"] = backfill
        store = type("S", (), {"readable_probe": lambda self: True})()
        monkeypatch.setattr(server, "_ladybug_store", store)
        return store

    monkeypatch.setattr(server, "_ladybug_store", None)
    monkeypatch.setattr(server, "_ladybug_failed", False)
    monkeypatch.setattr(server, "_ladybug_last_attempt", 0.0)
    monkeypatch.setattr(server, "_get_ladybug", fake_get)

    assert server._ladybug_health_state() == "available"
    assert kwargs["backfill"] is False


def test_backfill_deferred_by_health_still_runs_for_the_next_real_caller(monkeypatch):
    """Deferring it must not lose it — the first caller that wants the graph gets it."""
    ran = []
    monkeypatch.setattr(server, "_ladybug_store", object())
    monkeypatch.setattr(server, "_ladybug_backfilled", False)
    monkeypatch.setattr(server, "_ladybug_backfill_if_empty", lambda store: ran.append(store))

    server._get_ladybug(backfill=False)
    assert ran == [], "health's open must not backfill"

    server._get_ladybug()
    assert len(ran) == 1, "the next real caller must run the deferred backfill"

    server._get_ladybug()
    assert len(ran) == 1, "and only once per process"


def test_backfill_claim_is_serialized_by_its_own_lock(monkeypatch):
    """The one-time claim must be made under a lock, or two threads both take the writer lease.

    Asserted on the lock itself rather than by racing threads: a timing race reproduces only
    intermittently, and a test that usually passes is not a guard.
    """
    class _RecordingLock:
        def __init__(self):
            self.entered = 0
            self._lock = threading.Lock()

        def __enter__(self):
            self.entered += 1
            return self._lock.__enter__()

        def __exit__(self, *exc):
            return self._lock.__exit__(*exc)

    lock = _RecordingLock()
    ran = []
    monkeypatch.setattr(server, "_ladybug_store", object())
    monkeypatch.setattr(server, "_ladybug_backfilled", False)
    monkeypatch.setattr(server, "_ladybug_backfill_lock", lock)
    monkeypatch.setattr(server, "_ladybug_backfill_if_empty", lambda store: ran.append(store))

    server._get_ladybug()
    assert lock.entered == 1, "the one-time claim was made without holding the lock"
    assert len(ran) == 1

    server._get_ladybug()
    assert lock.entered == 1, "an already-claimed backfill must not re-enter the lock"
    assert len(ran) == 1


def test_backfill_lock_is_not_the_store_lock(monkeypatch):
    """_ladybug_lock is non-reentrant and is already held on one of the call paths."""
    assert server._ladybug_backfill_lock is not server._ladybug_lock


def test_health_reports_backoff_armed_by_the_open_it_just_attempted(monkeypatch):
    """A failed open arms the backoff; health must re-read that, not call it unavailable."""
    monkeypatch.setattr(server, "_ladybug_store", None)
    monkeypatch.setattr(server, "_ladybug_failed", False)
    monkeypatch.setattr(server, "_ladybug_last_attempt", 0.0)

    def failing_open(backfill=True):
        monkeypatch.setattr(server, "_ladybug_last_attempt", time.monotonic())
        return None

    monkeypatch.setattr(server, "_get_ladybug", failing_open)

    assert server._ladybug_health_state() == "backoff"
