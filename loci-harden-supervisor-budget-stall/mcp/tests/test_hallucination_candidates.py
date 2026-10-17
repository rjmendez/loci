"""_hallucination_candidates: the blanket-provenance path, and the normal one.

This function had no test at all, and its only call site swallows every
exception into `candidates = []`. A NameError inside it therefore reads exactly
like "no candidates found" — the feature disables itself and the caller reports
a clean result. That is the failure these tests exist to make loud.
"""
import unittest

from memcheck.verdict import Verdict
from server import _hallucination_candidates


def _v(vtype, refs):
    return Verdict(id="v-" + "-".join(refs), subject_kind="memory",
                   subject_signature=refs[0], subject_excerpt="", verdict_type=vtype,
                   decision="flag", confidence=1.0, rationale="", source="rule", refs=list(refs))


def _f(fid, text, rtype="observed"):
    return {"id": fid, "text": text, "record_type": rtype}


class HallucinationCandidatesTest(unittest.TestCase):

    def test_receipted_counter_finding_yields_a_candidate(self):
        """The documented case: unsupported positive vs a finding that IS receipted."""
        findings = [_f("a", "the relay is up"), _f("b", "the relay is down")]
        # only "a" is unsupported, so "b" counts as receipted
        out = _hallucination_candidates(findings, [], [_v("unsupported_observed", ["a"])],
                                        [_v("contradiction", ["a", "b"])])
        self.assertEqual([c["finding_id"] for c in out], ["a"])
        self.assertEqual(out[0]["contradicted_by"], "b")

    def test_blanket_unsupported_still_produces_candidates(self):
        """With no audit lane at all, EVERY observed finding is flagged unsupported.

        The old `other in unsupported_ids: continue` skip then discarded every
        observed-vs-observed contradiction, so the strongest hallucination signal
        could never fire. An empty audit lane must not silently disable detection.
        """
        findings = [_f("a", "the relay is up"), _f("b", "the relay is down")]
        unsupported = [_v("unsupported_observed", ["a"]), _v("unsupported_observed", ["b"])]
        out = _hallucination_candidates(findings, [], unsupported,
                                        [_v("contradiction", ["a", "b"])])
        self.assertTrue(out, "blanket-unsupported must not suppress the contradiction pass")
        self.assertIn(out[0]["finding_id"], {"a", "b"})

    def test_partial_unsupported_keeps_the_receipted_gate(self):
        """Not blanket: an unsupported-vs-unsupported pair has no receipted counter."""
        findings = [_f("a", "up"), _f("b", "down"), _f("c", "sideways")]
        unsupported = [_v("unsupported_observed", ["a"]), _v("unsupported_observed", ["b"])]
        # c is observed and NOT unsupported -> not blanket
        out = _hallucination_candidates(findings, [], unsupported,
                                        [_v("contradiction", ["a", "b"])])
        self.assertEqual(out, [], "a-vs-b both unsupported and c is receipted: no candidate")

    def test_no_contradictions_is_empty_not_an_exception(self):
        self.assertEqual(_hallucination_candidates([_f("a", "x")], [], [], []), [])


if __name__ == "__main__":
    unittest.main()
