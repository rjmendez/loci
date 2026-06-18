"""Tests for memcheck/backend.py — cosine_similarity and InMemoryBackend."""

import asyncio
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memcheck.backend import InMemoryBackend, cosine_similarity
from memcheck.verdict import new_verdict


def _verdict(kind="claim", decision="flag", confidence=0.8, text="some subject text"):
    return new_verdict(
        subject_kind=kind,
        subject_signature="sig:" + text[:20],
        subject_excerpt=text,
        verdict_type="contradiction",
        decision=decision,
        confidence=confidence,
        rationale="test",
        source="rule",
    )


def run(coro):
    return asyncio.run(coro)


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0)

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_opposite_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)

    def test_empty_vectors_return_zero(self):
        self.assertEqual(cosine_similarity([], []), 0.0)

    def test_mismatched_length_returns_zero(self):
        self.assertEqual(cosine_similarity([1.0], [1.0, 2.0]), 0.0)

    def test_zero_vector_returns_zero(self):
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 0.0]), 0.0)

    def test_negative_components(self):
        a = [-1.0, -1.0]
        b = [-1.0, -1.0]
        self.assertAlmostEqual(cosine_similarity(a, b), 1.0)

    def test_unnormalized_vectors(self):
        a = [3.0, 0.0]
        b = [5.0, 0.0]
        self.assertAlmostEqual(cosine_similarity(a, b), 1.0)

    def test_symmetry(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        self.assertAlmostEqual(cosine_similarity(a, b), cosine_similarity(b, a))


class TestInMemoryBackend(unittest.TestCase):
    def setUp(self):
        self.backend = InMemoryBackend()

    def test_empty_stats(self):
        stats = run(self.backend.stats())
        self.assertEqual(stats["total_verdicts"], 0)
        self.assertEqual(stats["recurring_blocks"], 0)

    def test_record_increments_total(self):
        run(self.backend.record(_verdict()))
        stats = run(self.backend.stats())
        self.assertEqual(stats["total_verdicts"], 1)

    def test_record_multiple(self):
        run(self.backend.record(_verdict(text="first finding")))
        run(self.backend.record(_verdict(text="second finding")))
        stats = run(self.backend.stats())
        self.assertEqual(stats["total_verdicts"], 2)

    def test_recall_empty_store(self):
        results = run(self.backend.recall("query", [1.0, 0.0], "claim", top_k=5))
        self.assertEqual(results, [])

    def test_recall_filters_by_kind(self):
        run(self.backend.record_with_embedding(_verdict(kind="claim"), [1.0, 0.0]))
        run(self.backend.record_with_embedding(_verdict(kind="code"), [1.0, 0.0]))
        results = run(self.backend.recall("query", [1.0, 0.0], "claim", top_k=5))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].verdict.subject_kind, "claim")

    def test_recall_returns_top_k(self):
        for i in range(5):
            run(self.backend.record_with_embedding(
                _verdict(kind="claim", text=f"finding {i}"),
                [float(i), 0.0]
            ))
        results = run(self.backend.recall("query", [1.0, 0.0], "claim", top_k=2))
        self.assertEqual(len(results), 2)

    def test_recall_sorted_by_similarity_desc(self):
        run(self.backend.record_with_embedding(_verdict(kind="m", text="a"), [1.0, 0.0]))
        run(self.backend.record_with_embedding(_verdict(kind="m", text="b"), [0.0, 1.0]))
        # Query is [1, 0] — "a" (same direction) should rank higher than "b" (orthogonal)
        results = run(self.backend.recall("q", [1.0, 0.0], "m", top_k=5))
        self.assertGreater(results[0].similarity, results[1].similarity)

    def test_coalesce_near_duplicate_increments_occurrences(self):
        embedding = [1.0, 0.0, 0.0]
        run(self.backend.record_with_embedding(_verdict(text="same subject"), embedding))
        run(self.backend.record_with_embedding(_verdict(text="same subject"), embedding))
        stats = run(self.backend.stats())
        # Should coalesce (similarity == 1.0 >= 0.97) → still just 1 stored verdict
        self.assertEqual(stats["total_verdicts"], 1)

    def test_different_kind_no_coalesce(self):
        embedding = [1.0, 0.0]
        run(self.backend.record_with_embedding(_verdict(kind="claim"), embedding))
        run(self.backend.record_with_embedding(_verdict(kind="code"), embedding))
        stats = run(self.backend.stats())
        self.assertEqual(stats["total_verdicts"], 2)

    def test_coalesce_keeps_higher_confidence(self):
        embedding = [1.0, 0.0]
        run(self.backend.record_with_embedding(_verdict(confidence=0.5), embedding))
        run(self.backend.record_with_embedding(_verdict(confidence=0.9), embedding))
        results = run(self.backend.recall("q", [1.0, 0.0], "claim", top_k=5))
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].verdict.confidence, 0.9)

    def test_forget_by_excerpt_and_kind(self):
        v = _verdict(text="target finding")
        run(self.backend.record(v))
        removed = run(self.backend.forget("target finding", "claim"))
        self.assertEqual(removed, 1)
        stats = run(self.backend.stats())
        self.assertEqual(stats["total_verdicts"], 0)

    def test_forget_nonexistent_returns_zero(self):
        removed = run(self.backend.forget("does not exist", "claim"))
        self.assertEqual(removed, 0)

    def test_forget_kind_mismatch_does_not_remove(self):
        run(self.backend.record(_verdict(kind="claim", text="target")))
        removed = run(self.backend.forget("target", "code"))
        self.assertEqual(removed, 0)
        stats = run(self.backend.stats())
        self.assertEqual(stats["total_verdicts"], 1)

    def test_recurring_blocks_count(self):
        embedding = [1.0, 0.0]
        # Record same verdict twice to get occurrences >= 2 with decision=flag (blocking)
        run(self.backend.record_with_embedding(
            _verdict(decision="flag"), embedding
        ))
        run(self.backend.record_with_embedding(
            _verdict(decision="flag"), embedding
        ))
        stats = run(self.backend.stats())
        self.assertEqual(stats["recurring_blocks"], 1)


if __name__ == "__main__":
    unittest.main()
