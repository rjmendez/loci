"""The lexical lane must not count function words as evidence.

investigation_pre_answer_check is the tool an agent calls BEFORE asserting a
memory-derived claim. Its lexical lane scored overlap normalised by the claim
side only, and tokenize() dropped 21 domain words but no stopwords — so a short
claim could clear the 0.45 gate on shared function words alone.

Measured on 400 real findings before the fix: 68.8% of generic short claims were
reported as supported. After: 1.2%. True-positive control over 300 claims drawn
from the findings they should match: unchanged at 96.7%.
"""
import unittest

import server


class TokenizeDropsNonEvidenceTest(unittest.TestCase):

    def test_stopwords_are_not_evidence(self):
        self.assertEqual(server.tokenize("the and of is was it this"), set())

    def test_domain_generic_words_are_still_dropped(self):
        self.assertEqual(server.tokenize("the host reported a result"), set())

    def test_content_words_survive(self):
        self.assertEqual(
            server.tokenize("the qdrant retention purge deleted findings"),
            {"qdrant", "retention", "purge", "deleted", "findings"},
        )


class LexicalGateTest(unittest.TestCase):
    """0.45 is the gate in _pre_answer_lexical_refs."""

    GATE = 0.45

    def _score(self, claim, evidence):
        return server._lexical_match_score(server.tokenize(claim),
                                           server.tokenize(evidence))

    def test_unrelated_evidence_no_longer_clears_the_gate(self):
        """The measured regression: shared stopwords declared support.

        'the server is down' scored 0.67 against a document about cultural-noise
        characterisation, on the overlap {'down', 'the'}.
        """
        score = self._score(
            "the server is down",
            "LITERATURE: how a continuous cultural-noise background is characterised",
        )
        self.assertLess(score, self.GATE)

    def test_a_claim_of_only_stopwords_cannot_be_supported(self):
        """It should be unassessable lexically, not trivially supported."""
        self.assertEqual(server.tokenize("it was the result"), set())
        self.assertEqual(self._score("it was the result", "anything at all here"), 0.0)

    def test_genuinely_matching_evidence_still_clears_the_gate(self):
        """The false-negative control: the fix must not break real matches."""
        score = self._score(
            "the retention purge deleted indexed findings",
            "The Qdrant retention purge deleted every indexed finding older than "
            "thirty days on each server start.",
        )
        self.assertGreaterEqual(score, self.GATE)


if __name__ == "__main__":
    unittest.main()
