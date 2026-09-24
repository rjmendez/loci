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

from memcheck.checks.contradiction_llm import (
    DEFAULT_SUBJECT_THRESHOLD,
    run_contradiction_llm,
    verify_and_merge,
)
from memcheck.verdict import make_signature, new_verdict


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
        self.prompts = []

    def __call__(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
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
        self.assertEqual(v.refs, ["a", "b"])
        self.assertEqual(v.subject_kind, "memory")
        self.assertEqual(v.decision, "flag")
        self.assertEqual(v.subject_signature, make_signature("memory", "a|b"))
        # Two medium/observed findings (protection 0.66 < 0.75): enforced, not provisional.
        self.assertIs(v.provisional, False)
        self.assertEqual(v.confidence, 0.80)
        self.assertEqual(
            v.rationale,
            "LLM judge confirmed factual contradiction (subject cosine=0.98)"
            "; disputed claim: adaptive expansion is limited to 1",
        )
        self.assertEqual(llm.calls, 1)

    def test_rejects_false_positive_via_llm_judge(self):
        # Shared jargon passes the embedding subject gate; the LLM judge must still reject these.
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
        v = verdicts[0]
        self.assertIs(v.provisional, True)
        # high (0.9) * observed (1.1) = 0.99 protection; provisional softens 0.80 -> 0.55.
        self.assertEqual(v.confidence, 0.55)
        self.assertTrue(v.rationale.endswith(
            " - PROVISIONAL: established finding (prot=0.99)"
            " requires corroboration before the verdict is enforced"
        ))

    def test_not_provisional_just_below_protection_threshold(self):
        # medium (0.6) * observed (1.1) = 0.66 on both sides: below 0.75.
        llm = _LLM()
        findings = [
            {"text": LIMITED, "type": "observed", "id": "a", "confidence": "medium"},
            {"text": NOT_LIMITED, "type": "inferred", "id": "b", "confidence": "low"},
        ]
        (v,) = run_contradiction_llm(findings, embed_fn=_embed, llm_fn=llm)
        self.assertIs(v.provisional, False)
        self.assertEqual(v.confidence, 0.80)
        self.assertNotIn("PROVISIONAL", v.rationale)


def _unit_vectors(texts):
    """One-hot-ish vectors keyed by position; the cosine is supplied separately."""
    return [[float(i)] for i in range(len(texts))]


def _table_cosine(table):
    """cosine_fn stub: look the pair up by the one-element 'index' vectors."""
    def cos(a, b):
        key = tuple(sorted((int(a[0]), int(b[0]))))
        value = table[key]
        if isinstance(value, Exception):
            raise value
        return value
    return cos


class _AlwaysContradicts:
    def __init__(self, raise_on=()):
        self.prompts = []
        self._raise_on = raise_on

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if any(marker in prompt for marker in self._raise_on):
            raise RuntimeError("judge unavailable")
        return '{"contradict": true, "claim": "x"}'


def _plain(text, id):
    return {"text": text, "id": id}


class TestGateAndJudge(unittest.TestCase):
    def test_pair_exactly_at_threshold_is_judged(self):
        llm = _AlwaysContradicts()
        findings = [_plain("alpha", "a"), _plain("beta", "b")]
        verdicts = run_contradiction_llm(
            findings, embed_fn=_unit_vectors, llm_fn=llm,
            cosine_fn=_table_cosine({(0, 1): DEFAULT_SUBJECT_THRESHOLD}),
        )
        self.assertEqual([v.refs for v in verdicts], [["a", "b"]])
        self.assertEqual(len(llm.prompts), 1)

    def test_pair_just_below_threshold_is_not_judged(self):
        llm = _AlwaysContradicts()
        findings = [_plain("alpha", "a"), _plain("beta", "b")]
        verdicts = run_contradiction_llm(
            findings, embed_fn=_unit_vectors, llm_fn=llm,
            cosine_fn=_table_cosine({(0, 1): DEFAULT_SUBJECT_THRESHOLD - 0.01}),
        )
        self.assertEqual(verdicts, [])
        self.assertEqual(llm.prompts, [])

    def test_max_pairs_judges_the_most_similar_first(self):
        llm = _AlwaysContradicts()
        findings = [_plain("alpha", "a"), _plain("beta", "b"), _plain("gamma", "c")]
        cos = _table_cosine({(0, 1): 0.70, (0, 2): 0.95, (1, 2): 0.80})
        verdicts = run_contradiction_llm(
            findings, embed_fn=_unit_vectors, llm_fn=llm, cosine_fn=cos, max_pairs=2,
        )
        self.assertEqual([v.refs for v in verdicts], [["a", "c"], ["b", "c"]])
        self.assertEqual(len(llm.prompts), 2)
        self.assertIn("subject cosine=0.95", verdicts[0].rationale)

    def test_judge_failure_on_one_pair_does_not_stop_the_others(self):
        llm = _AlwaysContradicts(raise_on=("FINDING A: alpha",))
        findings = [_plain("alpha", "a"), _plain("beta", "b"), _plain("gamma", "c")]
        cos = _table_cosine({(0, 1): 0.90, (0, 2): 0.95, (1, 2): 0.80})
        verdicts = run_contradiction_llm(
            findings, embed_fn=_unit_vectors, llm_fn=llm, cosine_fn=cos,
        )
        self.assertEqual([v.refs for v in verdicts], [["b", "c"]])
        self.assertEqual(len(llm.prompts), 3)

    def test_cosine_failure_skips_only_that_pair(self):
        llm = _AlwaysContradicts()
        findings = [_plain("alpha", "a"), _plain("beta", "b"), _plain("gamma", "c")]
        cos = _table_cosine({(0, 1): ValueError("dim"), (0, 2): 0.10, (1, 2): 0.90})
        verdicts = run_contradiction_llm(
            findings, embed_fn=_unit_vectors, llm_fn=llm, cosine_fn=cos,
        )
        self.assertEqual([v.refs for v in verdicts], [["b", "c"]])

    def test_fenced_json_reply_is_parsed(self):
        findings = [_plain("alpha", "a"), _plain("beta", "b")]
        verdicts = run_contradiction_llm(
            findings, embed_fn=_unit_vectors,
            llm_fn=lambda p: 'Sure.\n```json\n{"contradict": true, "claim": "c"}\n```',
            cosine_fn=_table_cosine({(0, 1): 0.9}),
        )
        self.assertEqual([v.refs for v in verdicts], [["a", "b"]])
        self.assertTrue(verdicts[0].rationale.endswith("; disputed claim: c"))

    def test_unparseable_or_negative_reply_is_not_a_contradiction(self):
        for reply in ("no idea", '{"contradict": false}', "", None):
            with self.subTest(reply=reply):
                verdicts = run_contradiction_llm(
                    [_plain("alpha", "a"), _plain("beta", "b")],
                    embed_fn=_unit_vectors, llm_fn=lambda p, r=reply: r,
                    cosine_fn=_table_cosine({(0, 1): 0.9}),
                )
                self.assertEqual(verdicts, [])


class TestVerifyAndMerge(unittest.TestCase):
    def _lexical(self):
        return [new_verdict(
            subject_kind="memory", subject_signature="s", subject_excerpt="e",
            verdict_type="contradiction", decision="flag", confidence=0.55,
            rationale="lexical", source="rule", refs=["x", "y"],
        )]

    def test_embed_failure_keeps_lexical_verdicts(self):
        lexical = self._lexical()

        def boom(texts):
            raise RuntimeError("embed endpoint down")

        llm = _AlwaysContradicts()
        out = verify_and_merge(
            [_plain("alpha", "a"), _plain("beta", "b")], lexical,
            embed_fn=boom, llm_fn=llm, cosine_fn=_table_cosine({}),
        )
        self.assertIs(out, lexical)
        self.assertEqual(llm.prompts, [])

    def test_mismatched_embeddings_keep_lexical_verdicts(self):
        lexical = self._lexical()
        out = verify_and_merge(
            [_plain("alpha", "a"), _plain("beta", "b")], lexical,
            embed_fn=lambda ts: [[0.0]], llm_fn=_AlwaysContradicts(),
            cosine_fn=_table_cosine({}),
        )
        self.assertIs(out, lexical)

    def test_working_embeddings_supersede_lexical_verdicts(self):
        out = verify_and_merge(
            [_plain("alpha", "a"), _plain("beta", "b")], self._lexical(),
            embed_fn=_unit_vectors, llm_fn=_AlwaysContradicts(),
            cosine_fn=_table_cosine({(0, 1): 0.9}),
        )
        self.assertEqual([(v.source, v.refs) for v in out], [("llm", ["a", "b"])])

    def test_working_embeddings_with_no_llm_contradiction_drop_lexical(self):
        out = verify_and_merge(
            [_plain("alpha", "a"), _plain("beta", "b")], self._lexical(),
            embed_fn=_unit_vectors, llm_fn=lambda p: '{"contradict": false}',
            cosine_fn=_table_cosine({(0, 1): 0.9}),
        )
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
