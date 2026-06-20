"""Network-free unit tests for the grounding gate.

Patches ground_gate.embed so the keep/drop logic is exercised without an
embeddings endpoint. Run: python3 -m unittest discover deep_think_loci/tests
"""
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "grounding"))
import ground_gate  # noqa: E402

# deterministic unit vectors keyed by text — query is orthogonal to bleed, parallel to on-topic
VECS = {
    "query": [1.0, 0.0],
    "on-topic": [1.0, 0.0],     # cos 1.00 -> keep
    "near": [0.8, 0.6],         # cos 0.80 -> keep
    "bleed": [0.0, 1.0],        # cos 0.00 -> drop
}


def fake_embed(texts):
    return np.array([VECS[t] for t in texts], dtype=float)


class GroundGateTests(unittest.TestCase):
    @patch.object(ground_gate, "embed", side_effect=fake_embed)
    def test_keeps_on_topic_drops_bleed(self, _):
        cands = [
            {"id": "a", "text": "on-topic"},
            {"id": "b", "text": "bleed"},
            {"id": "c", "text": "near"},
        ]
        r = ground_gate.gate("query", cands, threshold=0.59)
        self.assertEqual({k["id"] for k in r["kept"]}, {"a", "c"})
        self.assertEqual({k["id"] for k in r["dropped"]}, {"b"})
        self.assertEqual(r["n_kept"], 2)
        self.assertEqual(r["n_dropped"], 1)
        self.assertIn("cosine", r["mode"])
        # cosines are reported and rounded
        self.assertAlmostEqual(next(k["cos"] for k in r["kept"] if k["id"] == "a"), 1.0, places=2)

    @patch.object(ground_gate, "embed", side_effect=fake_embed)
    def test_threshold_boundary(self, _):
        # 'near' is cos 0.80; a threshold above it must drop it
        r = ground_gate.gate("query", [{"id": "c", "text": "near"}], threshold=0.85)
        self.assertEqual(r["n_kept"], 0)
        self.assertEqual(r["n_dropped"], 1)

    def test_empty_input_is_graceful(self):
        r = ground_gate.gate("query", [])
        self.assertEqual(r["mode"], "empty")
        self.assertEqual(r["n_in"], 0)
        self.assertEqual(r["kept"], [])

    @patch.object(ground_gate, "embed", side_effect=fake_embed)
    def test_bare_string_candidates_normalized(self, _):
        r = ground_gate.gate("query", ["on-topic", "bleed"], threshold=0.59)
        self.assertEqual(r["n_kept"], 1)
        self.assertEqual(r["n_dropped"], 1)


if __name__ == "__main__":
    unittest.main()
