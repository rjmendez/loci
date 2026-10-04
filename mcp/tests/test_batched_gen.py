"""Tests for batched_gen: concurrent fan-out generation over the local Ollama tier.

No live servers: the Ollama tier is stubbed by monkeypatching the lazily-imported
`llm_local` module. Deterministic stubs, fail-open assertions, per-prompt isolation.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import batched_gen as B  # noqa: E402


class _FakeLLMLocal:
    """Stand-in for the sibling llm_local module (only .generate is used)."""
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = set(fail_on or ())

    def generate(self, prompt, model=None, fmt=None, max_tokens=256, think=False):
        self.calls.append({"prompt": prompt, "model": model,
                           "fmt": fmt, "max_tokens": max_tokens, "think": think})
        if prompt in self.fail_on:
            return {"text": "", "ok": False, "model": model}
        return {"text": f"ollama:{prompt}", "ok": True, "model": model or "qwen2.5:3b"}


def _install_fake_llm_local(monkeypatch, fake):
    # batched_gen imports llm_local lazily, so the fake must sit on sys.modules.
    monkeypatch.setitem(sys.modules, "llm_local", fake)


def test_batch_goes_through_llm_local_in_order(monkeypatch):
    fake = _FakeLLMLocal()
    _install_fake_llm_local(monkeypatch, fake)
    out = B.generate_batch(["a", "b"])
    assert [r["text"] for r in out] == ["ollama:a", "ollama:b"]
    assert all(r["ok"] for r in out)
    assert len(fake.calls) == 2


def test_forwards_fmt_max_tokens_and_think(monkeypatch):
    fake = _FakeLLMLocal()
    _install_fake_llm_local(monkeypatch, fake)
    B.generate_batch(["p"], max_tokens=42, fmt="json", think=True)
    assert fake.calls[0]["fmt"] == "json"
    assert fake.calls[0]["max_tokens"] == 42
    assert fake.calls[0]["think"] is True
    assert fake.calls[0]["model"] is None   # no model named -> llm_local default used


def test_named_model_is_forwarded(monkeypatch):
    fake = _FakeLLMLocal()
    _install_fake_llm_local(monkeypatch, fake)
    B.generate_batch(["p"], model="qwen3:4b")
    assert fake.calls[0]["model"] == "qwen3:4b"


def test_client_fn_and_endpoint_role_are_accepted_and_ignored(monkeypatch):
    """They selected the removed vLLM path; callers still pass them."""
    fake = _FakeLLMLocal()
    _install_fake_llm_local(monkeypatch, fake)

    def _boom():
        raise RuntimeError("client_fn must not be called")

    out = B.generate_batch(["a"], client_fn=_boom, endpoint_role="code")
    assert out == [{"text": "ollama:a", "ok": True}]


def test_per_prompt_failure_isolation(monkeypatch):
    fake = _FakeLLMLocal(fail_on={"b"})
    _install_fake_llm_local(monkeypatch, fake)
    out = B.generate_batch(["a", "b", "c"])
    assert out[0]["ok"] is True and out[0]["text"] == "ollama:a"
    assert out[1] == {"text": "", "ok": False}
    assert out[2]["ok"] is True


def test_import_failure_degrades_whole_batch(monkeypatch):
    # If llm_local cannot even be imported, fail-open: aligned all-failed result, no raise.
    monkeypatch.setitem(sys.modules, "llm_local", None)  # import llm_local -> ImportError
    out = B.generate_batch(["a", "b"])
    assert out == [{"text": "", "ok": False}, {"text": "", "ok": False}]


def test_empty_prompts_returns_empty():
    assert B.generate_batch([]) == []
    assert B.generate_batch(None) == []


def test_non_string_prompts_are_stringified(monkeypatch):
    fake = _FakeLLMLocal()
    _install_fake_llm_local(monkeypatch, fake)
    out = B.generate_batch([1, None])
    assert [r["text"] for r in out] == ["ollama:1", "ollama:None"]


def test_dispatches_concurrently(monkeypatch):
    """Prompts must run concurrently, not one-at-a-time.

    Regression guard for the 2026-09 fix: _via_ollama used to loop serially even though
    the module is documented/used as a concurrent fan-out client. A slow fake generate()
    (sleep per call) proves concurrency by wall-clock: N calls at S seconds each must take
    close to S seconds total (thread-pooled), not N*S (serial).
    """
    import time

    class _SlowFakeLLMLocal:
        def __init__(self, delay):
            self.delay = delay
            self.calls = 0

        def generate(self, prompt, model=None, fmt=None, max_tokens=256, think=False):
            self.calls += 1
            time.sleep(self.delay)
            return {"text": f"ollama:{prompt}", "ok": True, "model": model}

    delay = 0.2
    n = 5
    fake = _SlowFakeLLMLocal(delay)
    _install_fake_llm_local(monkeypatch, fake)

    t0 = time.monotonic()
    out = B.generate_batch([f"p{i}" for i in range(n)])
    elapsed = time.monotonic() - t0

    assert fake.calls == n
    assert all(r["ok"] for r in out)
    # Serial execution would take n * delay (~1.0s here); concurrent execution should
    # complete well under half that. Generous margin for CI/thread-scheduling jitter.
    assert elapsed < (n * delay) * 0.6, f"expected concurrent dispatch, took {elapsed:.2f}s"
