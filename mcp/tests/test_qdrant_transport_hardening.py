"""Timeout/retry hardening for qdrant_ops transport paths."""
from __future__ import annotations

import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import qdrant_ops as Q  # noqa: E402


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture(autouse=True, scope="module")
def _module_globals_are_restored():
    # Direct assignment (Q._OLLAMA_BASE = ...) leaked embed.local into every
    # later test in the session. Tests must set module globals via monkeypatch.
    before = Q._OLLAMA_BASE
    yield
    assert Q._OLLAMA_BASE == before, "a test leaked qdrant_ops._OLLAMA_BASE"


@pytest.fixture(autouse=True)
def _reset_transport_state(monkeypatch):
    for op in ("embed", "qdrant_query"):
        Q._transport_breakers[op]["timeouts"] = 0
        Q._transport_breakers[op]["opened_until"] = 0.0
    Q._endpoint_ready_cache.clear()
    Q._embed_cache.clear()
    monkeypatch.setattr(Q, "_endpoint_ready", lambda _url: True)


def _install_requests(monkeypatch, side_effects):
    timeout_type = type("Timeout", (Exception,), {})
    conn_type = type("ConnectionError", (Exception,), {})
    queue = list(side_effects)

    def fake_post(_url, json=None, headers=None, timeout=None):  # noqa: ARG001
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    fake_requests = types.SimpleNamespace(
        post=fake_post,
        exceptions=types.SimpleNamespace(Timeout=timeout_type, ConnectionError=conn_type),
    )
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    return timeout_type, conn_type


def test_embed_retries_timeout_then_recovers(monkeypatch):
    monkeypatch.setattr(Q, "_OLLAMA_BASE", "http://embed.local:11434")
    monkeypatch.setattr(Q, "_EMBED_RETRY_ATTEMPTS", 2)
    monkeypatch.setattr(Q, "_TRANSPORT_BACKOFF_BASE_S", 0.01)
    monkeypatch.setattr(Q, "_TRANSPORT_BACKOFF_CAP_S", 0.02)
    sleep_calls = []
    monkeypatch.setattr(Q.time, "sleep", lambda s: sleep_calls.append(s))

    timeout_type, _ = _install_requests(monkeypatch, [])
    _install_requests(
        monkeypatch,
        [timeout_type("timed out"), _Resp({"data": [{"embedding": [0.1, 0.2]}]})],
    )

    out = Q._embed("hello")
    assert out == [0.1, 0.2]
    assert sleep_calls and sleep_calls[0] <= 0.02
    assert Q._transport_breakers["embed"]["timeouts"] == 0


def test_embed_brownout_opens_and_short_circuits(monkeypatch):
    monkeypatch.setattr(Q, "_OLLAMA_BASE", "http://embed.local:11434")
    monkeypatch.setattr(Q, "_EMBED_RETRY_ATTEMPTS", 1)
    monkeypatch.setattr(Q, "_TRANSPORT_BROWNOUT_THRESHOLD", 2)
    monkeypatch.setattr(Q, "_TRANSPORT_BROWNOUT_SECONDS", 60.0)

    timeout_type, _ = _install_requests(monkeypatch, [])
    calls = {"n": 0}

    def fake_post(_url, json=None, headers=None, timeout=None):  # noqa: ARG001
        calls["n"] += 1
        raise timeout_type("timed out")

    fake_requests = types.SimpleNamespace(
        post=fake_post,
        exceptions=types.SimpleNamespace(Timeout=timeout_type, ConnectionError=type("ConnectionError", (Exception,), {})),
    )
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    assert Q._embed("a") is None
    assert Q._embed("b") is None
    assert calls["n"] == 2
    # Brownout now open; this must not hit requests.post again.
    assert Q._embed("c") is None
    assert calls["n"] == 2


def test_embed_readiness_gate_blocks_high_cost_call(monkeypatch):
    monkeypatch.setattr(Q, "_OLLAMA_BASE", "http://embed.local:11434")
    monkeypatch.setattr(Q, "_endpoint_ready", lambda _url: False)
    requests_mock = types.SimpleNamespace(post=lambda *a, **k: _Resp({}))  # pragma: no cover - must not be called
    monkeypatch.setitem(sys.modules, "requests", requests_mock)
    assert Q._embed("hello") is None


def test_query_retry_helper_retries_then_returns(monkeypatch):
    monkeypatch.setattr(Q, "_QDRANT_QUERY_RETRY_ATTEMPTS", 2)
    monkeypatch.setattr(Q.time, "sleep", lambda _s: None)
    seq = [TimeoutError("timed out"), "ok"]

    def _call():
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    assert Q._query_points_with_retry(_call, attempts=2, op="qdrant_query") == "ok"


def test_query_retry_helper_reports_timeout_when_exhausted(monkeypatch):
    monkeypatch.setattr(Q.time, "sleep", lambda _s: None)
    with pytest.raises(RuntimeError, match="qdrant_query_timeout"):
        Q._query_points_with_retry(lambda: (_ for _ in ()).throw(TimeoutError("timed out")), attempts=2, op="qdrant_query")
