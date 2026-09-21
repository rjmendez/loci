"""Tests for llm_local — the local (Ollama) generation primitive.

No live Ollama: requests.post is monkeypatched with a fake response object. Covers the
happy path, HTTP-error fail-open, JSON-format validation, and that keep_alive + format
are actually placed in the posted payload.
"""
import pytest
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import llm_local as L  # noqa: E402


class _FakeResp:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, payload, raise_exc=None):
        self._payload = payload
        self._raise = raise_exc

    def raise_for_status(self):
        if self._raise is not None:
            raise self._raise

    def json(self):
        return self._payload


def _install_post(monkeypatch, resp=None, exc=None, capture=None):
    """Patch llm_local's `import requests` so requests.post returns `resp` / raises `exc`,
    optionally recording the posted json into `capture`."""
    import types

    def fake_post(url, json=None, timeout=None):  # noqa: A002 - mirror requests' kwarg
        if capture is not None:
            capture["url"] = url
            capture["json"] = json
            capture["timeout"] = timeout
        if exc is not None:
            raise exc
        return resp

    fake_requests = types.SimpleNamespace(post=fake_post)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)


def _ensure_base(monkeypatch):
    # generate() short-circuits when no base URL is configured; give it one.
    monkeypatch.setattr(L, "_gen_env", lambda _v="http://fake-ollama:11434": _v)


@pytest.fixture(autouse=True)
def _no_vllm_fallback(monkeypatch):
    """Keep these tests on the Ollama path and off the network.

    generate() now falls through to the vLLM tier when Ollama fails, so every
    fail-open test below would otherwise attempt a real request to whatever
    backends resolves. The fallback has its own tests in
    test_llm_local_fallback.py; here it is disabled so a failure means what the
    test name says it means.
    """
    monkeypatch.setattr(L, "_try_vllm", lambda *a, **k: None)
    # Pin the model: generate() otherwise resolves it from ~/.loci/backends.toml.
    import backends
    monkeypatch.setattr(backends, "ollama_gen_model", lambda: "qwen2.5:3b")


def test_happy_path_ok_true(monkeypatch):
    _ensure_base(monkeypatch)
    _install_post(monkeypatch, resp=_FakeResp({"response": "hello world"}))
    r = L.generate("say hi")
    assert r["ok"] is True
    assert r["text"] == "hello world"
    assert r["model"] == "qwen2.5:3b"


def test_http_error_fails_open(monkeypatch):
    _ensure_base(monkeypatch)
    _install_post(monkeypatch, resp=_FakeResp({}, raise_exc=RuntimeError("500 boom")))
    r = L.generate("say hi")
    assert r["ok"] is False
    assert r["text"] == ""
    assert r["model"] == "qwen2.5:3b"


def test_exception_never_raises(monkeypatch):
    _ensure_base(monkeypatch)
    _install_post(monkeypatch, exc=TimeoutError("timed out"))
    r = L.generate("say hi")  # must not raise
    assert r["ok"] is False
    assert r["text"] == ""
    assert r["model"] == "qwen2.5:3b"
    # The failure must carry its reason; without one a misconfigured endpoint needs a live diagnosis.
    assert "timed out" in r["why"]


def test_json_fmt_invalid_body_not_ok(monkeypatch):
    _ensure_base(monkeypatch)
    _install_post(monkeypatch, resp=_FakeResp({"response": "not-json {oops"}))
    r = L.generate("give me json", fmt="json")
    assert r["ok"] is False          # body did not parse as JSON
    assert r["text"] == "not-json {oops"  # text preserved for debugging


def test_json_fmt_valid_body_ok(monkeypatch):
    _ensure_base(monkeypatch)
    _install_post(monkeypatch, resp=_FakeResp({"response": '{"a": 1}'}))
    r = L.generate("give me json", fmt="json")
    assert r["ok"] is True
    assert json.loads(r["text"]) == {"a": 1}


def test_payload_includes_keep_alive_and_format(monkeypatch):
    _ensure_base(monkeypatch)
    cap = {}
    _install_post(monkeypatch, resp=_FakeResp({"response": "{}"}), capture=cap)
    L.generate("p", fmt="json", max_tokens=99, temperature=0.5, keep_alive="1h")
    body = cap["json"]
    assert body["keep_alive"] == "1h"          # critical: model stays resident
    assert body["format"] == "json"
    assert body["stream"] is False
    assert body["options"]["num_predict"] == 99
    assert body["options"]["temperature"] == 0.5
    assert cap["url"].endswith("/api/generate")


def test_payload_omits_format_when_not_json(monkeypatch):
    _ensure_base(monkeypatch)
    cap = {}
    _install_post(monkeypatch, resp=_FakeResp({"response": "hi"}), capture=cap)
    L.generate("p")
    assert "format" not in cap["json"]
    assert cap["json"]["keep_alive"] == "30m"  # default pin


def test_payload_includes_think_false_always(monkeypatch):
    """think:false must be sent regardless of fmt -- production classify/compress/verify
    calls all use fmt in different ways, and any of them can hit a thinking-capable model."""
    _ensure_base(monkeypatch)
    cap = {}
    _install_post(monkeypatch, resp=_FakeResp({"response": "hi"}), capture=cap)
    L.generate("p")
    assert cap["json"]["think"] is False


def test_falls_back_to_thinking_field_when_response_empty(monkeypatch):
    """Mirrors the ab_eval_local_model.py harness fix: a thinking-capable model can still
    route its JSON answer into `thinking` even with think=False sent."""
    _ensure_base(monkeypatch)
    _install_post(monkeypatch, resp=_FakeResp({"response": "", "thinking": '{"a": 1}'}))
    r = L.generate("give me json", fmt="json")
    assert r["ok"] is True
    assert json.loads(r["text"]) == {"a": 1}


def test_prefers_response_over_thinking_when_both_present(monkeypatch):
    _ensure_base(monkeypatch)
    _install_post(monkeypatch, resp=_FakeResp({"response": '{"a": 2}', "thinking": "reasoning trace"}))
    r = L.generate("give me json", fmt="json")
    assert json.loads(r["text"]) == {"a": 2}


def test_no_base_url_fails_open(monkeypatch):
    """No endpoint at all -- which means stubbing the RESOLVER too.

    This set only _OLLAMA and passed because _resolve_ollama() fell back to an
    endpoint that happened to be incapable of generating. Once generation was
    pointed at a host that works, the test failed -- correctly, because it had
    been asserting "the resolved backend is broken", not "there is no backend".
    """
    monkeypatch.setattr(L, "_gen_env", lambda: "")
    monkeypatch.setattr(L, "_resolve_ollama", lambda: "")
    r = L.generate("say hi")
    assert r["ok"] is False and r["text"] == ""
    assert "no Ollama endpoint" in r["why"]


def test_embedding_model_config_is_replaced_by_local_generation_model(monkeypatch):
    _ensure_base(monkeypatch)
    import backends
    monkeypatch.setattr(backends, "ollama_gen_model", lambda: "nomic-embed-text:latest")

    import types
    cap = {}

    class _TagsResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": "nomic-embed-text:latest"}, {"name": "qwen2.5:3b"}]}

    def fake_post(url, json=None, timeout=None):  # noqa: A002
        cap["model"] = json.get("model")
        return _FakeResp({"response": "hi"})

    fake_requests = types.SimpleNamespace(get=lambda *a, **k: _TagsResp(), post=fake_post)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    r = L.generate("say hi")
    assert r["ok"] is True
    assert r["model"] == "qwen2.5:3b"
    assert cap["model"] == "qwen2.5:3b"


def test_generate_retries_with_discovered_model_after_initial_model_failure(monkeypatch):
    _ensure_base(monkeypatch)
    import backends
    monkeypatch.setattr(backends, "ollama_gen_model", lambda: "bad-model:1")

    import types
    calls = {"post": []}

    class _TagsResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": "nomic-embed-text:latest"}, {"name": "qwen2.5:3b"}]}

    def fake_post(url, json=None, timeout=None):  # noqa: A002
        calls["post"].append(json.get("model"))
        if len(calls["post"]) == 1:
            raise RuntimeError("model not found")
        return _FakeResp({"response": "ok"})

    fake_requests = types.SimpleNamespace(get=lambda *a, **k: _TagsResp(), post=fake_post)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    r = L.generate("say hi")
    assert r["ok"] is True
    assert r["model"] == "qwen2.5:3b"
    assert calls["post"] == ["bad-model:1", "qwen2.5:3b"]


def test_timeout_kwarg_default_and_override(monkeypatch):
    """generate() posts with _TIMEOUT unless the caller passes its own deadline."""
    _ensure_base(monkeypatch)
    cap = {}
    _install_post(monkeypatch, resp=_FakeResp({"response": "hi"}), capture=cap)
    L.generate("say hi")
    assert cap["timeout"] == L._TIMEOUT
    L.generate("say hi", timeout=7)
    assert cap["timeout"] == 7.0
