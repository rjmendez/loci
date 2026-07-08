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
    assert set(out.keys()) == _EXPECTED_KEYS
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
    assert set(out.keys()) == _EXPECTED_KEYS


def test_loci_health_never_raises_when_backends_import_missing(monkeypatch):
    # Even if the whole backends module resolution blows up, the tool returns a dict.
    monkeypatch.setattr(backends, "ollama_url", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(backends, "vllm_url", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    monkeypatch.setattr(backends, "qdrant", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    out = json.loads(server.loci_health())
    assert set(out.keys()) == _EXPECTED_KEYS


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
