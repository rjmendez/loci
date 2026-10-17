"""Causal edges from author-declared derived_from links.

The shipped producer inferred causality from text: it required B to contain A's
UUID literally, or A's first 60 characters as a substring plus a causal keyword.
Meanwhile investigation_store was already recording derived_from — the same
relation, stated by the author instead of guessed.

Measured over 136 live investigations: heuristic 62 edges, declared 524.
"""
import unittest

import server


def _f(fid, text="some finding text", **kw):
    return {"id": fid, "text": text, **kw}


class DeclaredCausalEdgesTest(unittest.TestCase):

    def test_a_declared_link_becomes_an_edge(self):
        edges = server._declared_causal_edges([
            _f("a", "the purge deleted the index"),
            _f("b", "coverage collapsed", derived_from=["a"]),
        ])
        self.assertEqual(len(edges), 1)
        e = edges[0]
        self.assertEqual((e["source_id"], e["target_id"]), ("a", "b"))
        self.assertEqual(e["method"], "declared_lineage")
        self.assertEqual(e["edge_type"], "caused_by")
        self.assertGreater(e["confidence"], 0.5,
                           "declared lineage must outrank the heuristic's 0.5")

    def test_a_bare_string_is_accepted_not_only_a_list(self):
        edges = server._declared_causal_edges([_f("a"), _f("b", derived_from="a")])
        self.assertEqual([(e["source_id"], e["target_id"]) for e in edges], [("a", "b")])

    def test_multiple_parents_each_produce_an_edge(self):
        edges = server._declared_causal_edges([
            _f("a"), _f("b"), _f("c", derived_from=["a", "b"]),
        ])
        self.assertEqual(sorted(e["source_id"] for e in edges), ["a", "b"])

    def test_a_claim_string_with_no_matching_finding_is_skipped(self):
        """derived_from also accepts free text; that has no node to point at."""
        edges = server._declared_causal_edges([
            _f("b", derived_from=["the relay was already known to be down"]),
        ])
        self.assertEqual(edges, [])

    def test_a_parent_outside_the_investigation_is_skipped(self):
        edges = server._declared_causal_edges([_f("b", derived_from=["not-here"])])
        self.assertEqual(edges, [])

    def test_self_reference_is_skipped(self):
        edges = server._declared_causal_edges([_f("a", derived_from=["a"])])
        self.assertEqual(edges, [])

    def test_findings_without_lineage_produce_nothing(self):
        self.assertEqual(server._declared_causal_edges([_f("a"), _f("b")]), [])


class MergeCausalEdgesTest(unittest.TestCase):
    """Both lanes are kept; declared wins a collision.

    The heuristic is not inert — it produces 62 edges on the live corpus — so
    this is a union, not a replacement.
    """

    def _e(self, src, tgt, method, conf):
        return {"source_id": src, "target_id": tgt, "method": method,
                "confidence": conf, "edge_type": "caused_by"}

    def test_non_colliding_edges_are_both_kept(self):
        merged = server._merge_causal_edges(
            [self._e("a", "b", "heuristic", 0.5)],
            [self._e("c", "d", "declared_lineage", 0.9)],
        )
        self.assertEqual(
            sorted((e["source_id"], e["target_id"]) for e in merged),
            [("a", "b"), ("c", "d")],
        )

    def test_declared_replaces_an_inferred_edge_for_the_same_pair(self):
        merged = server._merge_causal_edges(
            [self._e("a", "b", "heuristic", 0.5)],
            [self._e("a", "b", "declared_lineage", 0.9)],
        )
        self.assertEqual(len(merged), 1, "one relation must not yield two edges")
        self.assertEqual(merged[0]["method"], "declared_lineage")

    def test_an_empty_declared_set_leaves_inferred_edges_alone(self):
        merged = server._merge_causal_edges([self._e("a", "b", "heuristic", 0.5)], [])
        self.assertEqual([e["method"] for e in merged], ["heuristic"])


if __name__ == "__main__":
    unittest.main()
