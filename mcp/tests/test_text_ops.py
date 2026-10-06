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


def _recording_generate(monkeypatch, text):
    """llm_local.generate stub that records exactly which keyword arguments it was called with."""
    calls = []

    def fake_generate(prompt, **kw):
        calls.append(kw)
        return {"text": text, "ok": True}

    import llm_local
    monkeypatch.setattr(llm_local, "generate", fake_generate)
    return calls


def test_compress_never_names_a_model_so_the_pool_picks(monkeypatch):
    """A call that names a model bypasses the pool: it is never logged, tagged or graded. Not even a leftover
    env var or config key may put a model on the call."""
    calls = _recording_generate(monkeypatch, "condensed")
    monkeypatch.setenv("LOCI_OLLAMA_COMPRESS_MODEL", "strong-compress-model:27b")
    import backends
    monkeypatch.setattr(backends, "_config", lambda: {"ollama": {"compress_model": "cfg-compress:7b"}})

    result = T.compress("x" * 50, max_chars=10)
    assert result == {"text": "condensed", "degraded": False}
    assert calls == [{"fmt": None, "max_tokens": 32}]


def test_classify_never_names_a_model_so_the_pool_picks(monkeypatch):
    calls = _recording_generate(monkeypatch, "bug")
    monkeypatch.setenv("LOCI_OLLAMA_CLASSIFY_MODEL", "tiny-classifier:1b")
    import backends
    monkeypatch.setattr(backends, "_config", lambda: {"ollama": {"classify_model": "cfg-classify:1b"}})

    result = T.classify("App crashes on launch after update.", ["bug", "feature"])
    assert result == {"label": "bug", "degraded": False}
    assert calls == [{"fmt": None, "max_tokens": 32}]


def test_classify_explicit_single_label_mention_skips_generation():
    def _boom(*a, **k):
        raise AssertionError("gen_fn must not be called when exactly one label is explicit")

    # Explicit mention of exactly one candidate label should route through the
    # deterministic code path instead of invoking local generation.
    result = T.classify("This is clearly a bug, not a crash fix request.", ["bug", "feature"], gen_fn=_boom)
    assert result == {"label": "bug", "degraded": False}



def test_classify_explicit_mention_returns_the_canonical_label_spelling():
    # Regression (hypothesis: text="0", labels=["0 "]): the fast path matched on
    # the stripped label and returned "0", which is not one of the caller's labels.
    def _boom(*a, **k):
        raise AssertionError("gen_fn must not be called when exactly one label is explicit")

    assert T.classify("0", ["0 "], gen_fn=_boom) == {"label": "0 ", "degraded": False}
    result = T.classify("route this to Billing Team now", [" Billing Team", "support"], gen_fn=_boom)
    assert result == {"label": " Billing Team", "degraded": False}


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
