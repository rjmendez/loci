"""Tests for text_ops — generation-tier basic ops. Generation is STUBBED (no live Ollama)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import text_ops as T  # noqa: E402


def _gen_returns(value, ok=True):
    """Build a stub gen_fn that always returns a fixed text/ok, matching the contract:
    gen_fn(prompt, *, fmt=None, max_tokens=256) -> {'text':str,'ok':bool}."""
    def _stub(prompt, *, fmt=None, max_tokens=256):
        return {"text": value, "ok": ok}
    return _stub


def test_compress_routes_through_compress_model_when_no_gen_fn_given(monkeypatch):
    """compress()'s default gen_fn (no explicit gen_fn passed) must resolve
    backends.ollama_compress_model() and pass it to llm_local.generate explicitly."""
    calls = {}

    def fake_generate(prompt, model="", fmt=None, max_tokens=256):
        calls["model"] = model
        return {"text": "condensed", "ok": True}

    import llm_local
    monkeypatch.setattr(llm_local, "generate", fake_generate)
    import backends
    monkeypatch.setattr(backends, "ollama_compress_model", lambda: "strong-compress-model:27b")

    result = T.compress("x" * 50, max_chars=10)
    assert result["degraded"] is False
    assert calls["model"] == "strong-compress-model:27b"


def test_compress_falls_back_to_shared_generate_when_backends_unresolvable(monkeypatch):
    """If backends.ollama_compress_model() itself errors, compress() must still call
    generate() with an empty model (shared gen_model fallback), not raise or no-op."""
    calls = {}

    def fake_generate(prompt, model="", fmt=None, max_tokens=256):
        calls["model"] = model
        return {"text": "condensed", "ok": True}

    import llm_local
    monkeypatch.setattr(llm_local, "generate", fake_generate)

    def _boom():
        raise RuntimeError("no backends module")

    import backends
    monkeypatch.setattr(backends, "ollama_compress_model", _boom)

    result = T.compress("x" * 50, max_chars=10)
    assert result["degraded"] is False
    assert calls["model"] == ""


def test_classify_routes_through_classify_model_when_no_gen_fn_given(monkeypatch):
    """classify() should resolve backends.ollama_classify_model() for model selection
    when no explicit gen_fn is provided."""
    calls = {}

    def fake_generate(prompt, model="", fmt=None, max_tokens=256):
        calls["model"] = model
        return {"text": "bug", "ok": True}

    import llm_local
    monkeypatch.setattr(llm_local, "generate", fake_generate)
    import backends
    monkeypatch.setattr(backends, "ollama_classify_model", lambda: "tiny-classifier:1b")

    result = T.classify("App crashes on launch after update.", ["bug", "feature"])
    assert result["label"] == "bug"
    assert result["degraded"] is False
    assert calls["model"] == "tiny-classifier:1b"


def test_classify_does_not_consult_compress_model(monkeypatch):
    """classify() must stay on the shared gen_model -- it must not call
    ollama_compress_model at all, confirming the two ops route independently."""
    calls = {"compress_model_called": False}

    def fake_generate(prompt, model="", fmt=None, max_tokens=256):
        return {"text": "bug", "ok": True}

    def _spy():
        calls["compress_model_called"] = True
        return "should-not-be-used:1b"

    import llm_local
    monkeypatch.setattr(llm_local, "generate", fake_generate)
    import backends
    monkeypatch.setattr(backends, "ollama_compress_model", _spy)

    result = T.classify("some bug report", ["bug", "feature"])
    assert result["label"] == "bug"
    assert calls["compress_model_called"] is False


def test_classify_explicit_single_label_mention_skips_generation():
    def _boom(*a, **k):
        raise AssertionError("gen_fn must not be called when exactly one label is explicit")

    # Explicit mention of exactly one candidate label should route through the
    # deterministic code path instead of invoking local generation.
    result = T.classify("This is clearly a bug, not a crash fix request.", ["bug", "feature"], gen_fn=_boom)
    assert result == {"label": "bug", "degraded": False}


# --- classify -------------------------------------------------------------

def test_classify_valid_label():
    gen = _gen_returns("bug")
    r = T.classify("the app crashes on launch", ["bug", "feature", "question"], gen_fn=gen)
    assert r == {"label": "bug", "degraded": False}


def test_classify_valid_label_case_and_punctuation_normalized():
    gen = _gen_returns("  Bug.  ")  # noisy casing/whitespace/punctuation around a real label
    r = T.classify("crash", ["bug", "feature"], gen_fn=gen)
    assert r == {"label": "bug", "degraded": False}  # mapped back to canonical spelling


def test_classify_out_of_set_label_is_degraded():
    gen = _gen_returns("banana")  # not in the label set
    r = T.classify("some text", ["bug", "feature"], gen_fn=gen)
    assert r == {"label": None, "degraded": True}


def test_classify_generation_not_ok_is_degraded():
    gen = _gen_returns("bug", ok=False)  # generation failed -> fall back
    r = T.classify("some text", ["bug", "feature"], gen_fn=gen)
    assert r == {"label": None, "degraded": True}


def test_classify_empty_inputs_degraded_without_calling_gen():
    def _boom(*a, **k):
        raise AssertionError("gen_fn must not be called on empty inputs")
    assert T.classify("", ["bug"], gen_fn=_boom) == {"label": None, "degraded": True}
    assert T.classify("text", [], gen_fn=_boom) == {"label": None, "degraded": True}


def test_classify_gen_fn_raising_is_fail_open():
    def _raises(prompt, *, fmt=None, max_tokens=256):
        raise RuntimeError("ollama down")
    r = T.classify("text", ["bug", "feature"], gen_fn=_raises)
    assert r == {"label": None, "degraded": True}


# --- compress -------------------------------------------------------------

def test_compress_happy_path_within_budget():
    summary = "Short condensed summary."
    gen = _gen_returns(summary)
    long_text = "x" * 5000
    r = T.compress(long_text, max_chars=600, gen_fn=gen)
    assert r == {"text": summary, "degraded": False}
    assert len(r["text"]) <= 600


def test_compress_already_within_budget_returns_unchanged_without_gen():
    def _boom(*a, **k):
        raise AssertionError("gen_fn must not be called when text already fits")
    r = T.compress("already short", max_chars=600, gen_fn=_boom)
    assert r == {"text": "already short", "degraded": False}


def test_compress_fail_open_when_gen_not_ok_char_truncates():
    gen = _gen_returns("ignored because ok is False", ok=False)
    long_text = "abcdefghij" * 100  # 1000 chars
    r = T.compress(long_text, max_chars=50, gen_fn=gen)
    assert r["degraded"] is True
    assert r["text"] == long_text[:50]
    assert len(r["text"]) == 50


def test_compress_model_overruns_budget_is_clamped_and_degraded():
    over = "y" * 100  # longer than the 40-char budget
    gen = _gen_returns(over)
    r = T.compress("z" * 500, max_chars=40, gen_fn=gen)
    assert r["degraded"] is True
    assert r["text"] == over[:40]
    assert len(r["text"]) == 40


def test_compress_gen_fn_raising_is_fail_open():
    def _raises(prompt, *, fmt=None, max_tokens=256):
        raise RuntimeError("ollama down")
    long_text = "q" * 300
    r = T.compress(long_text, max_chars=100, gen_fn=_raises)
    assert r["degraded"] is True and r["text"] == long_text[:100]
