"""The semantic lane must corroborate a hit before it counts as support.

investigation_pre_answer_check compared a filtered dense-similarity score to one
absolute constant (_QDRANT_SUPPORT_MIN_SCORE = 0.55) and appended anything above
it straight into support_refs. A filtered top-k search cannot return "no match":
query_points with a must-match on investigation_id always returns that
investigation's k nearest points, however unrelated the claim. So the gate asked
"is 0.55 big?" when the only answerable question is "is this hit distinctive, and
does it mention what the claim mentions?".

MEASURED on the live corpus (~/.hermes/memory-sessions, 141 investigations),
300 claims copied verbatim OUT of a different investigation:

    false support   264/300 = 88.0%  ->  1.7%
    and in 264/264 of those the lexical lane returned nothing: the semantic
    lane alone flipped the verdict.

Control, 300 claims that ARE backed by the investigation they are checked
against (a finding's own text, its own indexed point excluded):

    true support   100.0%  ->  80.3%

Retuning 0.55 cannot fix this: 96.7% of the negatives scored inside the positive
range. Neither half of the replacement rule separates the classes alone either --
best-ref lexical overlap is NEG p95 0.162 vs POS p05 0.070, best-ref margin over
the pool median is NEG p95 0.1013 vs POS p05 0.0528. The conjunction does.

The fixture in fixtures/semantic_gate_probe.json is those 600 probes (scores,
per-ref lexical overlap, pool size, label -- no embeddings, no finding text).
Regenerate with fixtures/measure_semantic_gate.py (needs a live Qdrant + embedder;
it lives beside the fixture rather than in scripts/ because scripts/ is a pinned
callgraph corpus).

No test here needs a live Qdrant or Ollama: the semantic lane is monkeypatched
and the lexical pool comes from a tmp MEMORY_DIR.
"""
import json
import statistics
import tempfile
import unittest
import uuid
from pathlib import Path

import server

FIXTURE = Path(__file__).parent / "fixtures" / "semantic_gate_probe.json"

# The real false support measured on the live corpus: this text was reported as
# supporting the M5Cardputer claim below, at score 0.63.
UNRELATED_FINDING_TEXT = (
    "GTSAM FIXED + VERIFIED HEALTHY: the dama-gtsam-fusion configMap was patched "
    "and the pose graph now converges on every window."
)
CLAIM = (
    "The M5Cardputer firmware passes an unsanitised filename to FatFS, so an "
    "arduino-esp32 SD path traversal escapes the upload directory."
)


def _ref(score, overlap, pool_size, pool_median, evidence_id="ev-1"):
    """A ref exactly as _search_qdrant_claim_evidence now returns one."""
    return {
        "evidence_id": evidence_id,
        "record_type": "finding",
        "source": "test",
        "ts": "2026-08-25T00:00:00Z",
        "origin": "qdrant",
        "score": score,
        "lexical_overlap": overlap,
        "pool_median": pool_median,
        "pool_size": pool_size,
        "margin": round(score - pool_median, 4),
        "snippet": "",
    }


class SemanticGateTestCase(unittest.TestCase):
    """Shared harness: tmp MEMORY_DIR + a patched semantic lane."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_dir = server.MEMORY_DIR
        self._orig_search = server._search_qdrant_claim_evidence
        server.MEMORY_DIR = Path(self._tmp.name)

    def tearDown(self):
        server._search_qdrant_claim_evidence = self._orig_search
        server.MEMORY_DIR = self._orig_dir
        self._tmp.cleanup()

    def _investigation(self, finding_text, numeric_confidence=None):
        inv_id = f"semgate-{uuid.uuid4().hex[:8]}"
        server.investigation_start(investigation_id=inv_id, title="semantic gate")
        stored = json.loads(server.investigation_store(
            investigation_id=inv_id,
            finding_type="observed",
            text=finding_text,
            source="test",
            confidence="high",
            numeric_confidence=numeric_confidence,
        ))
        return inv_id, stored

    def _patch_semantic(self, refs):
        server._search_qdrant_claim_evidence = lambda *a, **k: (
            list(refs), {"enabled": True, "available": True,
                         "query_attempted": True, "error": None},
        )

    def _check(self, inv_id, claim=CLAIM):
        return json.loads(server.investigation_pre_answer_check(
            investigation_id=inv_id, claims=claim, record=False,
        ))


class UncorroboratedSemanticHitTest(SemanticGateTestCase):

    def test_uncorroborated_semantic_hit_does_not_support(self):
        """T1 -- the reproduced defect, with the real text and the real score.

        The finding shares no content tokens with the claim, so the lexical lane
        returns nothing and the semantic hit is the only voice in the room. It is
        neither distinctive (0.63 vs a 0.55 pool median) nor on-topic (overlap
        well under 0.15), so it must not assert support.

        FAILS with the fix reverted: 0.63 >= 0.55 -> supported True, ref in
        support_refs.
        """
        inv_id, _ = self._investigation(UNRELATED_FINDING_TEXT)
        self._patch_semantic([_ref(0.63, 0.06, 25, 0.55)])
        result = self._check(inv_id)
        claim = result["claim_results"][0]

        self.assertFalse(claim["supported"])
        self.assertEqual(claim["support_refs"], [])
        self.assertEqual([r["evidence_id"] for r in claim["semantic_candidates"]], ["ev-1"])
        self.assertEqual(claim["support_basis"], "semantic_candidate_only")


class CorroboratedSemanticHitTest(SemanticGateTestCase):

    def test_corroborated_semantic_hit_still_supports(self):
        """T2 -- REGRESSION CONTROL. Passes both before and after, by design.

        A distinctive hit (0.78 against a 0.66 pool median) whose text actually
        mentions what the claim mentions. It exists to catch a fix that
        over-rejects, not to prove the defect.
        """
        inv_id, _ = self._investigation(UNRELATED_FINDING_TEXT)
        overlap = server._lexical_match_score(
            server.tokenize(CLAIM),
            server.tokenize("The M5Cardputer firmware hands an unsanitised filename "
                            "straight to FatFS on the SD card."),
        )
        # getattr so this control still runs against the pre-fix module.
        self.assertGreaterEqual(
            overlap, getattr(server, "_SEMANTIC_SUPPORT_MIN_OVERLAP", 0.15))
        self._patch_semantic([_ref(0.78, round(overlap, 4), 25, 0.66)])
        result = self._check(inv_id)
        claim = result["claim_results"][0]

        self.assertTrue(claim["supported"])
        self.assertEqual([r["evidence_id"] for r in claim["support_refs"]], ["ev-1"])
        if "support_basis" in claim:  # absent on the pre-fix module
            self.assertEqual(claim["support_basis"], "semantic_corroborated")


class FlatNeighbourhoodTest(SemanticGateTestCase):

    def test_flat_neighbourhood_is_not_support(self):
        """T3 -- distinctiveness, pinned independently of the overlap half.

        High lexical overlap, but the whole neighbourhood sits at 0.60-0.62: the
        hit is no closer to the claim than 24 other points, so its rank-1 status
        carries no information. Measured NEG p95 margin 0.1013 vs POS p05 0.0528.

        FAILS with the fix reverted: 0.62 >= 0.55 -> supported True.
        """
        inv_id, _ = self._investigation(UNRELATED_FINDING_TEXT)
        self._patch_semantic([_ref(0.62, 0.55, 25, 0.61)])
        result = self._check(inv_id)
        claim = result["claim_results"][0]

        self.assertFalse(claim["supported"])
        self.assertEqual(claim["support_refs"], [])
        self.assertEqual(claim["support_basis"], "semantic_candidate_only")


class SmallPoolTest(SemanticGateTestCase):

    def test_small_pool_denies_semantic_only_support(self):
        """T4 -- the degradation path.

        With 3 points indexed, "distinctive within its neighbourhood" is not a
        statistic. 46.3% of the negative probes landed on investigations with
        fewer than 8 indexed points; admitting those on overlap alone measured
        3.0% false support vs 1.7% denied, for +1.4pt of true support.

        FAILS with the fix reverted: 0.64 >= 0.55 -> supported True.
        """
        inv_id, _ = self._investigation(UNRELATED_FINDING_TEXT)
        self._patch_semantic([_ref(0.64, 0.40, 3, 0.50)])
        result = self._check(inv_id)
        claim = result["claim_results"][0]

        self.assertFalse(claim["supported"])
        self.assertEqual(claim["support_refs"], [])
        self.assertEqual([r["evidence_id"] for r in claim["semantic_candidates"]], ["ev-1"])


class ConfidenceChainTest(SemanticGateTestCase):

    def test_uncorroborated_semantic_ids_do_not_enter_the_confidence_chain(self):
        """T5 -- an unrelated neighbour also contaminated the reported confidence.

        Uncorroborated refs used to reach matched_ids, which feeds
        min_chain_confidence, so the tool reported a confidence chain walked over
        a finding that supports nothing.

        FAILS with the fix reverted: the id is in matched_evidence_ids and
        min_chain_confidence comes back as that finding's 0.2.
        """
        inv_id, stored = self._investigation(UNRELATED_FINDING_TEXT, numeric_confidence=0.2)
        finding_id = stored["finding_id"]
        self._patch_semantic([_ref(0.63, 0.06, 25, 0.55, evidence_id=finding_id)])
        result = self._check(inv_id)

        self.assertNotIn(finding_id, result["matched_evidence_ids"])
        self.assertIsNone(result["min_chain_confidence"])
        self.assertIsNone(result["confidence_summary"])
        self.assertFalse(result["claim_results"][0]["supported"])


class MeasuredOperatingPointTest(unittest.TestCase):
    """T6 -- replay the 600 live probes offline and pin the measured numbers."""

    @classmethod
    def setUpClass(cls):
        cls.rows = json.loads(FIXTURE.read_text())["rows"]

    @staticmethod
    def _supported(row):
        """The tool's verdict for one probe row, expressed against whatever rule
        the module currently ships.

        With the fix reverted _semantic_ref_corroborated does not exist and the
        fallback reproduces the shipped rule exactly -- any ref over the
        pre-filter is support -- so this test fails on the measured rate, not on
        a missing attribute.
        """
        if row["lexical_support_refs"]:
            return True
        corroborated = getattr(server, "_semantic_ref_corroborated", lambda ref: True)
        pool_median = statistics.median(row["scores"])
        for ref in row["refs"]:
            if ref["score"] < server._QDRANT_SUPPORT_MIN_SCORE:
                continue
            if corroborated({
                "score": ref["score"],
                "lexical_overlap": ref["lexical_overlap"],
                "pool_median": pool_median,
                "pool_size": len(row["scores"]),
            }):
                return True
        return False

    def _rate(self, label):
        rows = [r for r in self.rows if r["label"] == label]
        self.assertEqual(len(rows), 300, "fixture must carry 300 probes per class")
        return sum(1 for r in rows if self._supported(r)) / len(rows)

    def test_false_support_stays_under_five_percent(self):
        """Claims copied verbatim out of a DIFFERENT investigation. Was 88.0%."""
        rate = self._rate("negative")
        self.assertLessEqual(rate, 0.05, f"false support {rate:.1%} (measured 1.7%)")

    def test_true_support_stays_above_seventy_eight_percent(self):
        """Claims the investigation really does back. Was 100%, now 80.3%."""
        rate = self._rate("positive")
        self.assertGreaterEqual(rate, 0.78, f"true support {rate:.1%} (measured 80.3%)")


if __name__ == "__main__":
    unittest.main()
