"""Tests for the semantic contradiction check (deep_think -> loci merge).

Reproduces the two failure cases the lexical check got wrong in the dt-loci-006
training run, using injected embed/llm stubs — no network, no async:

  - true positive:  "LIMITED to 1 expansion" vs "NOT limited to 1 expansion"
                    (lexical MISSED it — too little surface overlap).
  - false positive: "code_memory_correlate works" vs "memory_retract over-captures"
                    (lexical FLAGGED it — shared jargon + a "should not"). These
                    pass the embedding subject gate, so it is the LLM polarity
                    judge that must reject them.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memcheck.checks.contradiction_llm import run_contradiction_llm


LIMITED = (
    "deep_think_fan_out adaptive width expansion is LIMITED to 1 expansion to cap "
    "API spend, triggered when confidence_score < threshold."
)
NOT_LIMITED = (
    "deep_think_fan_out adaptive width expansion is NOT limited to 1 expansion; it "
    "can expand unboundedly and is not capped to control API spend."
)
CORR_WORKS = (
    "code_memory_correlate correctly anchored on the fabricated entity and surfaced "
    "the contaminated finding. This part works."
)
RETRACT_OVERCAPTURES = (
    "entity-anchored memory_retract over-captures: it should not tombstone meta "
    "findings that merely mention the entity."
)


def _f(text, id):
    return {"text": text, "type": "observed", "id": id, "confidence": "medium"}


def _embed(texts):
    """Deterministic 4-d vectors: same-subject pairs are near-collinear."""
    out = []
    for t in texts:
        if "NOT limited" in t:
            out.append([0.98, 0.2, 0.0, 0.0])
        elif "LIMITED to 1 expansion" in t:
            out.append([1.0, 0.0, 0.0, 0.0])
        elif "over-captures" in t:
            out.append([0.2, 0.95, 0.0, 0.0])
        elif "correctly anchored" in t:
            out.append([0.0, 1.0, 0.0, 0.0])
        else:
            out.append([0.0, 0.0, 1.0, 0.0])
    return out


class _LLM:
    """Stub polarity judge: true only for the real expansion contradiction."""

    def __init__(self):
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        if "1 expansion" in prompt and "not limited" in prompt.lower():
            return '{"contradict": true, "claim": "adaptive expansion is limited to 1"}'
        return '{"contradict": false, "claim": ""}'


class TestSemanticContradiction(unittest.TestCase):
    def test_catches_semantic_negation_lexical_missed(self):
        llm = _LLM()
        findings = [_f(LIMITED, "a"), _f(NOT_LIMITED, "b")]
        verdicts = run_contradiction_llm(findings, embed_fn=_embed, llm_fn=llm)
        self.assertEqual(len(verdicts), 1)
        v = verdicts[0]
        self.assertEqual(v.verdict_type, "contradiction")
        self.assertEqual(v.source, "llm")
        self.assertEqual(sorted(v.refs), ["a", "b"])

    def test_rejects_false_positive_via_llm_judge(self):
        # These pass the embedding subject gate (shared jargon) but the LLM judge
        # must reject them — they describe different tools, no factual conflict.
        llm = _LLM()
        findings = [_f(CORR_WORKS, "c"), _f(RETRACT_OVERCAPTURES, "d")]
        verdicts = run_contradiction_llm(findings, embed_fn=_embed, llm_fn=llm)
        self.assertEqual(verdicts, [])
        self.assertGreaterEqual(llm.calls, 1)  # the judge was actually consulted

    def test_mixed_set_flags_only_the_real_pair(self):
        llm = _LLM()
        findings = [
            _f(LIMITED, "a"),
            _f(NOT_LIMITED, "b"),
            _f(CORR_WORKS, "c"),
            _f(RETRACT_OVERCAPTURES, "d"),
        ]
        verdicts = run_contradiction_llm(findings, embed_fn=_embed, llm_fn=llm)
        self.assertEqual(len(verdicts), 1)
        self.assertEqual(sorted(verdicts[0].refs), ["a", "b"])

    def test_subject_gate_skips_unrelated_pairs(self):
        # Two orthogonal-subject findings never reach the LLM judge.
        llm = _LLM()
        findings = [_f(LIMITED, "a"), _f("totally unrelated topic about widgets", "z")]
        verdicts = run_contradiction_llm(findings, embed_fn=_embed, llm_fn=llm)
        self.assertEqual(verdicts, [])
        self.assertEqual(llm.calls, 0)

    def test_fail_open_when_embeddings_unavailable(self):
        llm = _LLM()
        findings = [_f(LIMITED, "a"), _f(NOT_LIMITED, "b")]
        verdicts = run_contradiction_llm(findings, embed_fn=lambda ts: [], llm_fn=llm)
        self.assertEqual(verdicts, [])
        self.assertEqual(llm.calls, 0)

    def test_provisional_when_established_finding_contradicted(self):
        # A high-confidence observed finding contradicted by one datum -> provisional.
        llm = _LLM()
        findings = [
            {"text": LIMITED, "type": "observed", "id": "a", "confidence": "high"},
            {"text": NOT_LIMITED, "type": "observed", "id": "b", "confidence": "medium"},
        ]
        verdicts = run_contradiction_llm(findings, embed_fn=_embed, llm_fn=llm)
        self.assertEqual(len(verdicts), 1)
        self.assertTrue(verdicts[0].provisional)


if __name__ == "__main__":
    unittest.main()
