import sys
import unittest
from pathlib import Path

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

from flybrain_scope_engine import ComparativeClaimTieringEngine, ResearchPriorityScoring  # noqa: E402


class ResearchPriorityScoringTest(unittest.TestCase):
    def setUp(self):
        self.scorer = ResearchPriorityScoring()

    def test_categorical_confidence_labels_are_safe(self):
        base = {
            "payoff": 0.8,
            "scope_compatibility": 0.7,
            "evidence_strength": 0.6,
            "risk": 0.1,
            "latency": 0.2,
            "dataset": "fw",
        }
        expected = {"high": 0.9, "medium": 0.6, "low": 0.3}
        scores = {}
        for label, confidence in expected.items():
            result = self.scorer.score({**base, "confidence": label})
            self.assertAlmostEqual(result["components"]["confidence"], confidence)
            self.assertGreaterEqual(result["score"], 0.0)
            self.assertLessEqual(result["score"], 1.0)
            scores[label] = result["score"]
        self.assertGreater(scores["high"], scores["medium"])
        self.assertGreater(scores["medium"], scores["low"])

    def test_normalized_dataset_names_score_identically_near_threshold(self):
        payload = {
            "payoff": 0.8,
            "confidence": "medium",
            "scope_compatibility": 0.8,
            "evidence_strength": 0.7,
            "risk": 0.0,
            "latency": 0.0,
        }
        flywire = self.scorer.score({**payload, "dataset": "flywire"})
        mixed_case = self.scorer.score({**payload, "dataset": "FlyWire"})
        alias = self.scorer.score({**payload, "dataset": "FW"})
        spaced = self.scorer.score({**payload, "dataset": "fly wire"})

        self.assertEqual(flywire["score"], mixed_case["score"])
        self.assertEqual(flywire["score"], alias["score"])
        self.assertEqual(flywire["score"], spaced["score"])
        self.assertEqual(flywire["priority"], mixed_case["priority"])
        self.assertEqual(flywire["priority"], alias["priority"])
        self.assertEqual(flywire["priority"], spaced["priority"])
        self.assertGreaterEqual(flywire["score"], 0.75)

    def test_malformed_confidence_never_crashes_and_falls_back(self):
        base = {
            "payoff": 0.6,
            "scope_compatibility": 0.5,
            "evidence_strength": 0.4,
            "risk": 0.1,
            "latency": 0.1,
            "dataset": "fw",
        }
        malformed_values = ["", "not-a-number", {}, [], float("nan"), float("inf"), True]
        for value in malformed_values:
            result = self.scorer.score({**base, "confidence": value})
            self.assertEqual(result["components"]["confidence"], 0.0)
            self.assertGreaterEqual(result["score"], 0.0)
            self.assertLessEqual(result["score"], 1.0)
            self.assertIn(result["priority"], {"low", "medium", "high"})

    def test_priority_thresholds_are_deterministic_with_rounding(self):
        almost_high = self.scorer.score({
            "dataset": "fw",
            "payoff": 0.7998857142857143,  # raw ~= 0.74996 -> rounded score 0.75
            "confidence": "medium",
            "scope_compatibility": 0.8,
            "evidence_strength": 0.7,
            "risk": 0.0,
            "latency": 0.0,
        })
        almost_medium = self.scorer.score({
            "dataset": "unknown",
            "payoff": 0.6284571428571428,  # raw ~= 0.44996 -> rounded score 0.45
            "confidence": "low",
            "scope_compatibility": 0.4,
            "evidence_strength": 0.4,
            "risk": 0.0,
            "latency": 0.0,
        })

        self.assertEqual(almost_high["score"], 0.75)
        self.assertEqual(almost_high["priority"], "high")
        self.assertEqual(almost_medium["score"], 0.45)
        self.assertEqual(almost_medium["priority"], "medium")


class ComparativeClaimTieringEngineTest(unittest.TestCase):
    def setUp(self):
        self.engine = ComparativeClaimTieringEngine()

    def test_confidence_labels_and_malformed_values_do_not_crash(self):
        base = {
            "dataset": "FlyWire",
            "dataset_version": "flywire783",
            "sex": "female",
            "life_stage": "adult",
            "annotation_completeness": 0.95,
            "circuit_class": "Kenyon cell",
            "experience_window": "naive",
            "evidence_strength": "validated",
        }
        for confidence in ("high", "medium", "low", "bad-token", {"oops": 1}, float("nan")):
            result = self.engine.tier({**base, "confidence": confidence})
            self.assertIn(result["tier"], {"T0", "T1", "T2", "T3"})
            self.assertGreaterEqual(result["score"], 0.0)
            self.assertLessEqual(result["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
