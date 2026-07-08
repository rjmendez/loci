"""Tests for the loci_health self-diagnosis tool + the embed warm-ping (#2, #7).

Both are read-only + fail-open: no live backends are required. Backend reachability
is stubbed via backends._alive; the warm-ping is exercised with an injected embed_fn.
"""
import json
import sys
import threading
from pathlib import Path

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import backends  # noqa: E402
import embed_ops  # noqa: E402
import server  # noqa: E402


_EXPECTED_KEYS = {
    "code_version", "kuzu", "ollama_reachable", "vllm_reachable",
    "qdrant_reachable", "embed_model", "rerank_model", "warm",
}


def test_loci_health_returns_expected_keys():
    out = json.loads(server.loci_health())
    assert isinstance(out, dict)
    # Subset (not ==) so additive fields don't break this contract.
    assert _EXPECTED_KEYS <= set(out.keys())
    # kuzu is one of the documented states
    assert out["kuzu"] in ("available", "unavailable", "latched", "backoff")
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
    monkeypatch.setattr(backends, "ollama_url", lambda: "http://localhost:11434")
    monkeypatch.setattr(backends, "vllm_url", lambda: "http://localhost:8000")
    monkeypatch.setattr(backends, "qdrant", lambda: ("http://localhost:6333", ""))

    out = json.loads(server.loci_health())
    # The raising ollama probe must NOT mask the others (independent + fail-open).
    assert out["ollama_reachable"] is False   # swallowed -> default
    assert out["vllm_reachable"] is True
    assert out["qdrant_reachable"] is False
    assert _EXPECTED_KEYS <= set(out.keys())


def test_loci_health_never_raises_when_resolvers_throw(monkeypatch):
    # Even if every backend endpoint resolver raises, the tool still returns a dict
    # with the full key set (each probe is independent + fail-open).
    monkeypatch.setattr(backends, "ollama_url", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(backends, "vllm_url", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(backends, "qdrant", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    out = json.loads(server.loci_health())
    assert _EXPECTED_KEYS <= set(out.keys())


def test_code_version_first_compute_is_thread_safe(monkeypatch):
    # Concurrent callers before the cache is filled must spawn `git rev-parse`
    # exactly ONCE (double-checked locking), and all agree on the result.
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
    # If the daemon thread cannot be created/started, warm() must NOT latch
    # _warm_started, so warmed() stays False and a later call can retry.
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
