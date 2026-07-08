"""Tests for llm_local — the local (Ollama) generation primitive.

No live Ollama: requests.post is monkeypatched with a fake response object. Covers the
happy path, HTTP-error fail-open, JSON-format validation, and that keep_alive + format
are actually placed in the posted payload.
"""
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
    monkeypatch.setattr(L, "_OLLAMA", "http://fake-ollama:11434")


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
    assert r == {"text": "", "ok": False, "model": "qwen2.5:3b"}


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


def test_no_base_url_fails_open(monkeypatch):
    monkeypatch.setattr(L, "_OLLAMA", "")
    r = L.generate("say hi")
    assert r["ok"] is False and r["text"] == ""
