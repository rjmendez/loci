"""llm_local -> LiteLLM tier (LOCI_LLM_VIA_LITELLM).

The auto-router can answer from a cloud tier, so this tier must pass the same egress
policy as the direct cloud fallback, and any refusal or failure must land back on the
direct Ollama path. No network: requests.post is faked and records every URL.
"""
import json
import os
import sys
import types
from unittest import mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import llm_local as L  # noqa: E402


class _Resp:
    def __init__(self, payload, exc=None):
        self._payload, self._exc = payload, exc

    def raise_for_status(self):
        if self._exc is not None:
            raise self._exc

    def json(self):
        return self._payload


def _chat(text, model="cf-glm"):
    return {"model": model, "choices": [{"message": {"content": text}}]}


@pytest.fixture
def net(monkeypatch):
    """Fake requests; routes by URL suffix and records calls."""
    calls = []
    routes = {"/chat/completions": _Resp(_chat("via litellm")),
              "/api/generate": _Resp({"response": "direct ollama"})}

    def post(url, json=None, headers=None, timeout=None):  # noqa: A002
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        for suffix, resp in routes.items():
            if url.endswith(suffix):
                if isinstance(resp, Exception):
                    raise resp
                return resp
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=post))
    monkeypatch.setattr(L, "_gen_env", lambda: "http://fake-ollama:11434")
    import backends
    monkeypatch.setattr(backends, "ollama_gen_model", lambda: "qwen2.5:3b")
    monkeypatch.setattr(L, "_read_gpu_load", lambda: None)
    for name in ("LOCI_LLM_VIA_LITELLM", "LOCI_CLOUD_TIER_ALLOW_PROMPT_EXPORT",
                 "LOCI_LITELLM_URL", "LOCI_LITELLM_MODEL", "LOCI_LITELLM_API_KEY",
                 "LITELLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    net_ = types.SimpleNamespace(calls=calls, routes=routes)
    return net_


def _egress_allowed(monkeypatch, **overrides):
    import backends
    values = {"cloud_tier_enabled": True, "cloud_max_tokens_per_call": 0,
              "cloud_deny_roles": set(), "cloud_allowed_roles": set()}
    values.update(overrides)
    for attr, value in values.items():
        monkeypatch.setattr(backends, attr, (lambda v=value: v))
    monkeypatch.setattr(L, "_consume_cloud_daily_budget", lambda max_tokens: None)
    monkeypatch.setenv("LOCI_CLOUD_TIER_ALLOW_PROMPT_EXPORT", "1")


def _urls(net):
    return [c["url"] for c in net.calls]


def test_flag_off_stays_on_ollama(net, monkeypatch):
    _egress_allowed(monkeypatch)
    r = L.generate("hello")
    assert r["text"] == "direct ollama"
    assert all("/chat/completions" not in u for u in _urls(net))


def test_flag_on_routes_through_litellm(net, monkeypatch):
    _egress_allowed(monkeypatch)
    monkeypatch.setenv("LOCI_LLM_VIA_LITELLM", "1")
    monkeypatch.setenv("LITELLM_API_KEY", "k")
    r = L.generate("hello", max_tokens=33, temperature=0.1)
    assert r["ok"] and r["text"] == "via litellm" and r["tier"] == "litellm"
    assert r["model"] == "cf-glm"
    assert _urls(net) == ["http://127.0.0.1:4000/v1/chat/completions"]
    body = net.calls[0]["json"]
    assert body["model"] == "auto-router" and body["max_tokens"] == 33
    assert net.calls[0]["headers"] == {"Authorization": "Bearer k"}


def test_json_format_maps_to_response_format(net, monkeypatch):
    _egress_allowed(monkeypatch)
    monkeypatch.setenv("LOCI_LLM_VIA_LITELLM", "1")
    net.routes["/chat/completions"] = _Resp(_chat('{"a": 1}'))
    r = L.generate("hello", fmt="json")
    assert json.loads(r["text"]) == {"a": 1}
    assert net.calls[0]["json"]["response_format"] == {"type": "json_object"}


def test_non_json_answer_falls_back_to_ollama(net, monkeypatch):
    _egress_allowed(monkeypatch)
    monkeypatch.setenv("LOCI_LLM_VIA_LITELLM", "1")
    net.routes["/api/generate"] = _Resp({"response": '{"ok": true}'})
    r = L.generate("hello", fmt="json")  # litellm answers "via litellm", not JSON
    assert r["text"] == '{"ok": true}' and r.get("tier") != "litellm"


@pytest.mark.parametrize("override,env", [
    ({"cloud_tier_enabled": False}, {}),
    ({}, {"LOCI_CLOUD_TIER_ALLOW_PROMPT_EXPORT": ""}),
    ({"cloud_max_tokens_per_call": 10}, {}),
    ({"cloud_deny_roles": {"triage"}}, {}),
    ({"cloud_allowed_roles": {"redteam"}}, {}),
])
def test_egress_policy_blocks_litellm_tier(net, monkeypatch, override, env):
    _egress_allowed(monkeypatch, **override)
    monkeypatch.setenv("LOCI_LLM_VIA_LITELLM", "1")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(L, "_heuristic_route", lambda p: {"role": "triage"})
    r = L.generate("hello", max_tokens=256)
    assert r["text"] == "direct ollama"
    assert all("/chat/completions" not in u for u in _urls(net))


def test_budget_refusal_blocks_litellm_tier(net, monkeypatch):
    _egress_allowed(monkeypatch)
    monkeypatch.setattr(L, "_consume_cloud_daily_budget", lambda max_tokens: "budget exhausted")
    monkeypatch.setenv("LOCI_LLM_VIA_LITELLM", "1")
    assert L.generate("hello")["text"] == "direct ollama"
    assert all("/chat/completions" not in u for u in _urls(net))


def test_explicit_model_and_think_stay_direct(net, monkeypatch):
    _egress_allowed(monkeypatch)
    monkeypatch.setenv("LOCI_LLM_VIA_LITELLM", "1")
    assert L.generate("hello", model="granite3-guardian:2b")["text"] == "direct ollama"
    assert L.generate("hello", think=True, max_tokens=1200)["text"] == "direct ollama"
    assert all("/chat/completions" not in u for u in _urls(net))


def test_litellm_failure_falls_back_to_ollama(net, monkeypatch):
    _egress_allowed(monkeypatch)
    monkeypatch.setenv("LOCI_LLM_VIA_LITELLM", "1")
    net.routes["/chat/completions"] = ConnectionError("refused")
    r = L.generate("hello")
    assert r["text"] == "direct ollama"
    assert _urls(net)[0].endswith("/chat/completions") and _urls(net)[1].endswith("/api/generate")


def test_credentials_are_redacted_before_egress(net, monkeypatch):
    _egress_allowed(monkeypatch)
    monkeypatch.setenv("LOCI_LLM_VIA_LITELLM", "1")
    L.generate("use ghp_abcdefghijklmnopqrstuvwxyz0123 please")
    sent = net.calls[0]["json"]["messages"][0]["content"]
    assert "ghp_" not in sent and "[REDACTED]" in sent
