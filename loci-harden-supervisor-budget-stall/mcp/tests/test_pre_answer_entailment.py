"""Tests for pre_answer_entailment — advisory exact-claim support checking."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pre_answer_entailment as E  # noqa: E402


def _ok(text):
    def _fn(prompt, *, fmt=None, max_tokens=256):
        assert fmt == "json"
        assert "CLAIM:" in prompt and "CITED EVIDENCE:" in prompt
        return {"text": text, "ok": True}
    return _fn


def _not_ok(prompt, *, fmt=None, max_tokens=256):
    return {"text": "", "ok": False, "why": "no Ollama endpoint resolved"}


def _raises(prompt, *, fmt=None, max_tokens=256):
    raise RuntimeError("connection refused")


_EVIDENCE = [{
    "role": "support",
    "evidence_id": "f-1",
    "record_type": "observed",
    "source": "test",
    "ts": "2026-09-16T00:00:00Z",
    "text": "EDR alert fired on host-b. Analyst noted this may indicate suspicious activity.",
}]


def test_confirmed_verdict_is_returned_when_model_confirms_exact_support():
    result = E.check_claim_entailment(
        "Host-b executed malware.",
        _EVIDENCE,
        gen_fn=_ok('{"verdict": "confirmed", "rationale": "The cited evidence says so.", "confidence": 0.8}'),
    )
    assert result["available"] is True
    assert result["verdict"] == "confirmed"
    assert result["rationale"] == "The cited evidence says so."
    assert result["confidence"] == 0.8
    assert result["degraded"] is False


def test_unknown_verdict_is_coerced_to_uncertain_without_marking_unavailable():
    result = E.check_claim_entailment(
        "claim",
        _EVIDENCE,
        gen_fn=_ok('{"verdict": "maybe", "rationale": "mixed", "confidence": 0.5}'),
    )
    assert result["available"] is True
    assert result["verdict"] == "uncertain"
    assert result["degraded"] is False


def test_not_ok_result_marks_check_unavailable_fail_open():
    result = E.check_claim_entailment("claim", _EVIDENCE, gen_fn=_not_ok)
    assert result["available"] is False
    assert result["verdict"] is None
    assert result["degraded"] is True
    assert result["error"] == "no Ollama endpoint resolved"


def test_generate_error_marks_check_unavailable_fail_open():
    result = E.check_claim_entailment("claim", _EVIDENCE, gen_fn=_raises)
    assert result["available"] is False
    assert result["verdict"] is None
    assert result["degraded"] is True
    assert "generate() raised" in result["error"]


def test_unparseable_response_marks_check_unavailable():
    result = E.check_claim_entailment("claim", _EVIDENCE, gen_fn=_ok("not json"))
    assert result["available"] is False
    assert result["verdict"] is None
    assert result["degraded"] is True
    assert "unparseable response" in result["error"]


def test_empty_claim_or_evidence_short_circuits_unavailable_without_calling_model():
    calls = []

    def _fn(*a, **k):
        calls.append((a, k))
        return {"text": "{}", "ok": True}

    result = E.check_claim_entailment("   ", _EVIDENCE, gen_fn=_fn)
    assert result["available"] is False
    result = E.check_claim_entailment("claim", [], gen_fn=_fn)
    assert result["available"] is False
    assert calls == []


def test_default_gen_fn_reuses_verify_lazy_generate(monkeypatch):
    captured = {}

    def _fake(prompt, *, fmt=None, max_tokens=256):
        captured["fmt"] = fmt
        captured["max_tokens"] = max_tokens
        return {
            "text": '{"verdict": "refuted", "rationale": "Evidence only reports an alert.", "confidence": 0.7}',
            "ok": True,
        }

    import verify
    monkeypatch.setattr(verify, "_lazy_generate", _fake)
    result = E.check_claim_entailment("claim", _EVIDENCE)
    assert result["available"] is True
    assert result["verdict"] == "refuted"
    assert captured["fmt"] == "json"
    assert captured["max_tokens"] == 220
