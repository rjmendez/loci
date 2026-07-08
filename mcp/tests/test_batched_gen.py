"""Tests for batched_gen — the batched vLLM/TGI client with Ollama fallback.

No live servers: the HTTP client is stubbed via `client_fn`, and the Ollama fallback is
stubbed by monkeypatching the lazily-imported `llm_local` module. Mirrors test_embed_ops
style: deterministic stubs, fail-open assertions, per-prompt isolation.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import batched_gen as B  # noqa: E402


# ---- stub HTTP client emulating a requests.Session against /v1/completions -------------

class _Resp:
    def __init__(self, text="", *, status_ok=True, body=None):
        self._text = text
        self._status_ok = status_ok
        self._body = body

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("HTTP 500")

    def json(self):
        if self._body is not None:
            return self._body
        return {"choices": [{"text": self._text}]}


class _StubClient:
    """Records posts; returns a canned completion per prompt. `fail_on` prompts raise."""
    def __init__(self, fail_on=None, echo=True):
        self.calls = []
        self.fail_on = set(fail_on or ())
        self.echo = echo

    def post(self, url, json=None, timeout=None):
        prompt = (json or {}).get("prompt", "")
        self.calls.append({"url": url, "json": json})
        if prompt in self.fail_on:
            raise ConnectionError("simulated network failure")
        return _Resp(text=f"echo:{prompt}" if self.echo else "")


def _client_fn_for(client):
    return lambda: client


# ---- helpers to force VLLM_BASE_URL state ----------------------------------------------

def _set_vllm(monkeypatch, url):
    monkeypatch.setattr(B, "_VLLM", url)


# ---- batched happy path ----------------------------------------------------------------

def test_batched_happy_path(monkeypatch):
    _set_vllm(monkeypatch, "http://vllm.local:8000")
    client = _StubClient()
    prompts = ["a", "b", "c"]
    out = B.generate_batch(prompts, client_fn=_client_fn_for(client))
    assert len(out) == len(prompts)
    assert all(r["ok"] for r in out)
    assert [r["text"] for r in out] == ["echo:a", "echo:b", "echo:c"]
    # Hit the OpenAI-compatible completions endpoint, once per prompt.
    assert all(c["url"].endswith("/v1/completions") for c in client.calls)
    assert len(client.calls) == 3
    assert client.calls[0]["json"]["model"] == B._DEFAULT_MODEL


def test_batched_respects_model_and_max_tokens(monkeypatch):
    _set_vllm(monkeypatch, "http://vllm.local:8000")
    client = _StubClient()
    B.generate_batch(["x"], model="my-model", max_tokens=99,
                     client_fn=_client_fn_for(client))
    body = client.calls[0]["json"]
    assert body["model"] == "my-model"
    assert body["max_tokens"] == 99


def test_batched_json_format_validates(monkeypatch):
    _set_vllm(monkeypatch, "http://vllm.local:8000")

    class _JsonClient(_StubClient):
        def post(self, url, json=None, timeout=None):
            self.calls.append({"url": url, "json": json})
            prompt = (json or {}).get("prompt")
            text = '{"k": 1}' if prompt == "good" else "not json"
            return _Resp(text=text)

    client = _JsonClient()
    out = B.generate_batch(["good", "bad"], fmt="json",
                           client_fn=_client_fn_for(client))
    assert out[0]["ok"] is True and out[1]["ok"] is False   # non-JSON downgraded
    assert client.calls[0]["json"]["response_format"] == {"type": "json_object"}


# ---- fallback to Ollama when VLLM_BASE_URL unset ---------------------------------------

class _FakeLLMLocal:
    """Stand-in for the sibling llm_local module (only .generate is used)."""
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = set(fail_on or ())

    def generate(self, prompt, model=None, fmt=None, max_tokens=256):
        self.calls.append({"prompt": prompt, "model": model,
                           "fmt": fmt, "max_tokens": max_tokens})
        if prompt in self.fail_on:
            return {"text": "", "ok": False, "model": model}
        return {"text": f"ollama:{prompt}", "ok": True, "model": model or "qwen2.5:3b"}


def _install_fake_llm_local(monkeypatch, fake):
    # batched_gen does `import llm_local` lazily; put our fake on sys.modules so that
    # import resolves to it without any real Ollama call.
    monkeypatch.setitem(sys.modules, "llm_local", fake)


def test_fallback_when_vllm_unset(monkeypatch):
    _set_vllm(monkeypatch, "")            # no batched server configured
    fake = _FakeLLMLocal()
    _install_fake_llm_local(monkeypatch, fake)
    # A client_fn is provided but must be IGNORED on the fallback path.
    sentinel = _StubClient()
    out = B.generate_batch(["a", "b"], client_fn=_client_fn_for(sentinel))
    assert [r["text"] for r in out] == ["ollama:a", "ollama:b"]
    assert all(r["ok"] for r in out)
    assert len(fake.calls) == 2 and sentinel.calls == []   # went to Ollama, not HTTP


def test_fallback_forwards_fmt_and_max_tokens(monkeypatch):
    _set_vllm(monkeypatch, "")
    fake = _FakeLLMLocal()
    _install_fake_llm_local(monkeypatch, fake)
    B.generate_batch(["p"], max_tokens=42, fmt="json",
                     client_fn=None)
    assert fake.calls[0]["fmt"] == "json"
    assert fake.calls[0]["max_tokens"] == 42
    assert fake.calls[0]["model"] is None   # no model named -> llm_local default used


def test_fallback_when_batched_server_down(monkeypatch):
    # VLLM set, but EVERY request raises -> whole batch degrades to Ollama, not all-failed.
    _set_vllm(monkeypatch, "http://vllm.local:8000")
    fake = _FakeLLMLocal()
    _install_fake_llm_local(monkeypatch, fake)
    client = _StubClient(fail_on={"a", "b"})   # server unreachable for all prompts
    out = B.generate_batch(["a", "b"], client_fn=_client_fn_for(client))
    assert [r["text"] for r in out] == ["ollama:a", "ollama:b"]
    assert all(r["ok"] for r in out)


# ---- per-prompt failure isolation ------------------------------------------------------

def test_per_prompt_failure_isolation_batched(monkeypatch):
    _set_vllm(monkeypatch, "http://vllm.local:8000")
    client = _StubClient(fail_on={"b"})        # only middle prompt fails
    out = B.generate_batch(["a", "b", "c"], client_fn=_client_fn_for(client))
    assert len(out) == 3
    assert out[0] == {"text": "echo:a", "ok": True}
    assert out[1] == {"text": "", "ok": False}   # isolated failure
    assert out[2] == {"text": "echo:c", "ok": True}


def test_per_prompt_failure_isolation_fallback(monkeypatch):
    _set_vllm(monkeypatch, "")
    fake = _FakeLLMLocal(fail_on={"b"})
    _install_fake_llm_local(monkeypatch, fake)
    out = B.generate_batch(["a", "b", "c"])
    assert out[0]["ok"] is True and out[0]["text"] == "ollama:a"
    assert out[1] == {"text": "", "ok": False}
    assert out[2]["ok"] is True


def test_fallback_import_failure_degrades_whole_batch(monkeypatch):
    # If llm_local cannot even be imported, fail-open: aligned all-failed result, no raise.
    _set_vllm(monkeypatch, "")
    monkeypatch.setitem(sys.modules, "llm_local", None)  # import llm_local -> ImportError
    # also block the package-relative fallback import
    out = B.generate_batch(["a", "b"])
    assert out == [{"text": "", "ok": False}, {"text": "", "ok": False}]


# ---- edge cases ------------------------------------------------------------------------

def test_empty_prompts_returns_empty(monkeypatch):
    _set_vllm(monkeypatch, "http://vllm.local:8000")
    assert B.generate_batch([]) == []
    assert B.generate_batch(None) == []


def test_client_fn_construction_failure_degrades_to_ollama(monkeypatch):
    _set_vllm(monkeypatch, "http://vllm.local:8000")
    fake = _FakeLLMLocal()
    _install_fake_llm_local(monkeypatch, fake)

    def _boom():
        raise RuntimeError("cannot build client")

    out = B.generate_batch(["a"], client_fn=_boom)
    assert out == [{"text": "ollama:a", "ok": True}]  # fell back cleanly
