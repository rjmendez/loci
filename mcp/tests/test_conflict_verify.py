"""Tests for conflict_verify — semantic corroboration for conflict detection."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import conflict_verify as C  # noqa: E402


def _gen_fn(text, why=None):
    """Build a stub gen_fn matching llm_local.generate's return contract."""
    def _fn(prompt, model="", fmt=None, max_tokens=256, temperature=0.2, keep_alive="30m"):
        if why is not None:
            return {"text": "", "ok": False, "model": model, "why": why}
        return {"text": text, "ok": True, "model": model}
    return _fn


def test_empty_findings_short_circuit_without_calling_gen_fn():
    calls = []

    def _fn(*a, **k):
        calls.append((a, k))
        return {"text": '{"verdict":"contradict","reason":"x"}', "ok": True}

    result = C.judge_conflict("   ", "non-empty", gen_fn=_fn)
    assert result == {"verdict": None, "reason": "", "ok": True, "error": None}
    assert calls == []


def test_contradict_verdict_is_reported():
    result = C.judge_conflict(
        "the port is open",
        "the port is not open",
        gen_fn=_gen_fn('{"verdict":"contradict","reason":"same port, opposite polarity"}'),
    )
    assert result == {
        "verdict": "contradict",
        "reason": "same port, opposite polarity",
        "ok": True,
        "error": None,
    }


def test_agree_and_same_topic_verdicts_are_accepted():
    agree = C.judge_conflict("a", "b", gen_fn=_gen_fn('{"verdict":"agree","reason":"same fact"}'))
    assert agree["verdict"] == "agree"

    same_topic = C.judge_conflict(
        "a",
        "b",
        gen_fn=_gen_fn('{"verdict":"same_topic_no_conflict","reason":"related but compatible"}'),
    )
    assert same_topic["verdict"] == "same_topic_no_conflict"


def test_json_extraction_tolerates_fenced_output():
    result = C.judge_conflict(
        "a",
        "b",
        gen_fn=_gen_fn("```json\n"
                       '{"verdict":"agree","reason":"compatible"}\n'
                       "```"),
    )
    assert result["verdict"] == "agree"
    assert result["ok"] is True


def test_not_ok_generation_fails_open():
    result = C.judge_conflict("a", "b", gen_fn=_gen_fn("", why="no Ollama endpoint resolved"))
    assert result == {"verdict": None, "reason": "", "ok": False,
                      "error": "no Ollama endpoint resolved"}


def test_generate_raising_fails_open():
    def _boom(*a, **k):
        raise RuntimeError("connection refused")

    result = C.judge_conflict("a", "b", gen_fn=_boom)
    assert result["verdict"] is None
    assert result["ok"] is False
    assert "generate() raised" in result["error"]


def test_invalid_or_unparseable_json_fails_open():
    bad_json = C.judge_conflict("a", "b", gen_fn=_gen_fn('{"verdict":"contradict"'))
    assert bad_json["verdict"] is None
    assert bad_json["ok"] is False
    assert "unparseable response" in bad_json["error"]

    bad_verdict = C.judge_conflict("a", "b", gen_fn=_gen_fn('{"verdict":"maybe","reason":"x"}'))
    assert bad_verdict["verdict"] is None
    assert bad_verdict["ok"] is False
    assert "invalid verdict" in bad_verdict["error"]


def test_lazy_generate_routes_through_verify_model(monkeypatch):
    calls = {}

    def fake_generate(prompt, model="", fmt=None, max_tokens=256, temperature=0.2, keep_alive="30m"):
        calls["model"] = model
        calls["fmt"] = fmt
        calls["temperature"] = temperature
        return {"text": '{"verdict":"agree","reason":"x"}', "ok": True}

    import llm_local
    monkeypatch.setattr(llm_local, "generate", fake_generate)
    import backends
    monkeypatch.setattr(backends, "ollama_verify_model", lambda: "strong-verify-model:27b")

    result = C._lazy_generate("prompt", fmt="json", max_tokens=64)
    assert result["ok"] is True
    assert calls["model"] == "strong-verify-model:27b"
    assert calls["fmt"] == "json"
    assert calls["temperature"] == 0.0


def test_lazy_generate_falls_back_to_empty_model_when_backend_lookup_errors(monkeypatch):
    calls = {}

    def fake_generate(prompt, model="", fmt=None, max_tokens=256, temperature=0.2, keep_alive="30m"):
        calls["model"] = model
        return {"text": '{"verdict":"agree","reason":"x"}', "ok": True}

    import llm_local
    monkeypatch.setattr(llm_local, "generate", fake_generate)
    import backends
    monkeypatch.setattr(backends, "ollama_verify_model", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    result = C._lazy_generate("prompt", fmt="json")
    assert result["ok"] is True
    assert calls["model"] == ""
