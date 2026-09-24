"""Tests for pre_answer_entailment — advisory exact-claim support checking."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

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


def test_reflex_fastpath_confirms_exact_high_overlap_without_model_call():
    calls = []

    def _fn(*a, **k):
        calls.append((a, k))
        return {"text": "{}", "ok": True}

    evidence = [{
        "role": "support",
        "evidence_id": "f-2",
        "record_type": "observed",
        "source": "test",
        "text": "Host-b executed malware and established persistence.",
    }]
    result = E.check_claim_entailment(
        "Host-b executed malware.",
        evidence,
        gen_fn=_fn,
    )
    assert result["available"] is True
    assert result["verdict"] == "confirmed"
    assert "reflex_fastpath" in result["rationale"]
    assert calls == []


def test_reflex_fastpath_is_deterministic_on_repeated_calls():
    calls = []

    def _fn(*a, **k):
        calls.append((a, k))
        return {"text": "{}", "ok": True}

    evidence = [{
        "role": "support",
        "evidence_id": "f-2b",
        "record_type": "observed",
        "source": "test",
        "text": "Host-b executed malware and established persistence.",
    }]
    first = E.check_claim_entailment("Host-b executed malware.", evidence, gen_fn=_fn)
    second = E.check_claim_entailment("Host-b executed malware.", evidence, gen_fn=_fn)

    assert first["available"] is True
    assert first["verdict"] == "confirmed"
    assert second == first
    assert "reflex_fastpath" in first["rationale"]
    assert calls == []


def test_reflex_fastpath_fails_open_on_malformed_evidence():
    calls = []

    def _fn(*a, **k):
        calls.append((a, k))
        return {"text": "{}", "ok": True}

    malformed = [
        None,
        "not a dict",
        {"role": "support", "evidence_id": "f-4b", "record_type": "observed", "source": "test"},
        {"role": "support", "text": None},
    ]
    result = E.check_claim_entailment("Host-b executed malware.", malformed, gen_fn=_fn)

    assert result["available"] is False
    assert result["verdict"] is None
    assert result["degraded"] is True
    assert calls == []


def test_reflex_fastpath_refutes_strong_contradiction_without_model_call():
    calls = []

    def _fn(*a, **k):
        calls.append((a, k))
        return {"text": "{}", "ok": True}

    evidence = [{
        "role": "contradiction",
        "evidence_id": "f-3",
        "record_type": "observed",
        "source": "test",
        "text": "Host-b did not execute malware during the inspected window.",
    }]
    result = E.check_claim_entailment(
        "Host-b executed malware during the inspected window.",
        evidence,
        gen_fn=_fn,
    )
    assert result["available"] is True
    assert result["verdict"] == "refuted"
    assert "reflex_fastpath" in result["rationale"]
    assert calls == []


def test_reflex_fastpath_prevalidated_support_short_circuits_model_call():
    calls = []

    def _fn(*a, **k):
        calls.append((a, k))
        return {"text": "{}", "ok": True}

    evidence = [{
        "role": "support",
        "evidence_id": "f-4",
        "record_type": "observed",
        "source": "test",
        "validation_status": "validated",
        "text": "Host-b executed malware and wrote startup registry keys.",
    }]
    result = E.check_claim_entailment(
        "Host-b executed malware.",
        evidence,
        gen_fn=_fn,
    )
    assert result["available"] is True
    assert result["verdict"] == "confirmed"
    assert "reflex_fastpath" in result["rationale"]
    assert calls == []


# --- negation must never be confirmed by the lexical reflex -------------------------

def _recording(text):
    calls = []

    def _fn(prompt, *, fmt=None, max_tokens=256):
        calls.append(prompt)
        return {"text": text, "ok": True}

    return _fn, calls


_MODEL_REFUTES = '{"verdict": "refuted", "rationale": "the evidence negates the claim", "confidence": 0.6}'


@pytest.mark.parametrize("row", [
    # exact containment, full overlap: the claim appears verbatim inside a negation
    {"role": "support", "text": "No evidence that Host-b executed malware."},
    {"role": "support", "text": "It is false that host-b executed malware."},
    {"role": "support", "text": "Host-b executed malware? Not observed."},
    # prevalidated support with strong overlap but opposite polarity
    {"role": "support", "validation_status": "validated", "text": "Host-b never executed malware."},
    {"role": "support", "prevalidated": True, "text": "Host-b did not execute malware."},
])
def test_reflex_fastpath_never_confirms_negated_support(row):
    evidence = [dict(row, evidence_id="f-neg", record_type="observed", source="test")]
    assert E._reflex_arc_fastpath("Host-b executed malware.", evidence) is None

    fn, calls = _recording(_MODEL_REFUTES)
    result = E.check_claim_entailment("Host-b executed malware.", evidence, gen_fn=fn)
    assert len(calls) == 1                      # the model, not the reflex, decides
    assert result["verdict"] == "refuted"
    assert result["confidence"] == 0.6
    assert "reflex_fastpath" not in result["rationale"]


def test_reflex_fastpath_never_confirms_a_negated_claim_from_affirmative_evidence():
    evidence = [{"role": "support", "validation_status": "validated",
                 "text": "Host-b did execute malware from the startup folder."}]
    assert E._reflex_arc_fastpath("Host-b did not execute malware.", evidence) is None


def test_reflex_fastpath_confirms_when_claim_and_evidence_share_the_negation():
    # Positive twin: matching polarity is still an exact, model-free confirmation.
    evidence = [{"role": "support", "evidence_id": "f-5", "record_type": "observed", "source": "test",
                 "text": "Host-b did not execute malware during the window."}]
    result = E._reflex_arc_fastpath("Host-b did not execute malware.", evidence)
    assert result == {
        "available": True,
        "verdict": "confirmed",
        "rationale": "reflex_fastpath: exact high-overlap support evidence; model bypassed.",
        "confidence": 0.97,
        "degraded": False,
        "error": "",
    }


def test_reflex_fastpath_skips_malformed_rows_and_still_uses_the_good_one():
    # Direct call: the isinstance/text filters let a good row through malformed ones.
    good = {"role": "support", "evidence_id": "f-6", "record_type": "observed", "source": "test",
            "text": "Host-b executed malware and established persistence."}
    rows = [None, "not a dict", 7, {"role": "support", "text": None}, {"role": None}, good]
    result = E._reflex_arc_fastpath("Host-b executed malware.", rows)
    assert result is not None
    assert (result["verdict"], result["confidence"]) == ("confirmed", 0.97)

    fn, calls = _recording("{}")
    via_tool = E.check_claim_entailment("Host-b executed malware.", rows, gen_fn=fn)
    assert via_tool == result
    assert calls == []


@pytest.mark.parametrize("raw_conf", ["NaN", "Infinity", '"inf"', "1" + "0" * 400],
                         ids=["NaN", "Infinity", "str-inf", "int-10e400"])
def test_nonfinite_model_confidence_marks_the_check_unavailable(raw_conf):
    raw = '{"verdict": "confirmed", "rationale": "ok", "confidence": %s}' % raw_conf
    result = E.check_claim_entailment("Host-b executed malware.", _EVIDENCE, gen_fn=_ok(raw))
    assert result["available"] is False
    assert result["verdict"] is None
    assert result["confidence"] == 0.0
    assert result["degraded"] is True
    assert result["error"].startswith("non-finite confidence")


def test_coerce_confidence_rejects_nonfinite_and_keeps_finite():
    assert [E._coerce_confidence(x) for x in (float("nan"), float("inf"), "-inf", 10 ** 400)] == [0.0] * 4
    assert [E._coerce_confidence(x) for x in (0.25, "0.5", 3, -1)] == [0.25, 0.5, 1.0, 0.0]
