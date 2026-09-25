"""Tests for memcheck/checks/ — pure Python, no network or async."""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memcheck.checks.contradiction import run_contradiction
from memcheck.checks.provenance import run_provenance
from memcheck.checks.contagion import find_contamination


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(text, id=None, type="observed", source=""):
    """Build a minimal finding dict."""
    d = {"text": text, "type": type, "source": source}
    if id is not None:
        d["id"] = id
    return d


def _audit(tool, text=""):
    return {"tool": tool, "text": text}


# ---------------------------------------------------------------------------
# Contradiction checks
# ---------------------------------------------------------------------------

class TestRunContradiction(unittest.TestCase):
    def test_empty_findings_returns_empty(self):
        self.assertEqual(run_contradiction([]), [])

    def test_none_findings_returns_empty(self):
        self.assertEqual(run_contradiction(None), [])

    def test_single_finding_no_contradiction(self):
        self.assertEqual(run_contradiction([_f("the host is alive")]), [])

    def test_two_same_polarity_no_flag(self):
        # Both positive — no negation polarity mismatch
        f1 = _f("the service is running on port 443")
        f2 = _f("the service is available on port 443")
        self.assertEqual(run_contradiction([f1, f2]), [])

    def test_two_both_negated_no_flag(self):
        f1 = _f("the service is not running on port 443")
        f2 = _f("the service is not available on port 443")
        self.assertEqual(run_contradiction([f1, f2]), [])

    def test_opposite_polarity_high_overlap_flags(self):
        f1 = _f("the service is running on port 8080 endpoint alive")
        f2 = _f("the service is not running on port 8080 endpoint alive")
        results = run_contradiction([f1, f2])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].verdict_type, "contradiction")
        self.assertEqual(results[0].decision, "flag")

    def test_opposite_polarity_low_overlap_no_flag(self):
        # Very different texts — overlap below min_overlap
        f1 = _f("the database connection succeeded")
        f2 = _f("network firewall not configured")
        self.assertEqual(run_contradiction([f1, f2]), [])

    def test_symmetric_pair_one_verdict_only(self):
        # A vs B and B vs A should produce exactly one verdict, not two
        f1 = {"id": "a", "text": "the server is running on port 9000 http endpoint alive"}
        f2 = {"id": "b", "text": "the server is not running on port 9000 http endpoint alive"}
        results = run_contradiction([f1, f2])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].refs, ["a", "b"])

    def test_duplicate_id_pair_is_deduped(self):
        # The same finding id re-stored later (e.g. an edit) must not produce a
        # second verdict for the same (a, b) pair: the pair key is id-based.
        pos = {"id": "a", "text": "the server is running on port 9000 http endpoint alive"}
        neg = {"id": "b", "text": "the server is not running on port 9000 http endpoint alive"}
        pos_again = {"id": "a", "text": "the server is running on port 9000 http endpoint alive"}
        results = run_contradiction([pos, neg, pos_again])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].refs, ["a", "b"])

    def test_malformed_finding_skipped(self):
        good = _f("the service is running on port 443 endpoint alive")
        not_negated = _f("the service is not available on port 443 endpoint alive")
        # Non-dict entry should be skipped without raising, and without
        # dropping the valid pair around it (ids keep their input index).
        results = run_contradiction([good, "not a dict", not_negated])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].refs, ["finding:0", "finding:2"])
        self.assertEqual(results[0].decision, "flag")

    def _cap_findings(self, with_ids=True):
        texts = [
            "alpha beta gamma delta",
            "not alpha beta gamma delta",
            "epsilon zeta theta iota",
            "not epsilon zeta theta iota",
        ]
        ids = ["old-pos", "old-neg", "new-pos", "new-neg"]
        return [
            {"id": i, "text": t} if with_ids else {"text": t}
            for i, t in zip(ids, texts)
        ]

    def test_max_findings_cap(self):
        # Both the oldest pair and the newest pair contradict; max_findings=2
        # keeps only the most recent two.
        findings = self._cap_findings()
        self.assertEqual(len(run_contradiction(findings)), 2)  # uncapped control
        results = run_contradiction(findings, max_findings=2)
        self.assertEqual([v.refs for v in results], [["new-pos", "new-neg"]])

    def test_max_findings_cap_keeps_original_index_for_derived_ids(self):
        results = run_contradiction(self._cap_findings(with_ids=False), max_findings=2)
        self.assertEqual([v.refs for v in results], [["finding:2", "finding:3"]])

    def test_ordinary_findings_are_not_provisional(self):
        # observed + no confidence -> protection 0.55 < 0.75: enforced verdict.
        f1 = {"id": "a", "text": "the cache is populated with entries records rows"}
        f2 = {"id": "b", "text": "the cache is not populated with entries records rows"}
        (v,) = run_contradiction([f1, f2])
        self.assertIs(v.provisional, False)
        self.assertEqual(v.confidence, 0.55)
        self.assertNotIn("PROVISIONAL", v.rationale)

    def test_established_finding_makes_contradiction_provisional(self):
        # high (0.9) * observed (1.1) -> protection 0.99 >= 0.75.
        f1 = {"id": "a", "text": "the cache is populated with entries records rows",
              "confidence": "high", "type": "observed"}
        f2 = {"id": "b", "text": "the cache is not populated with entries records rows"}
        (v,) = run_contradiction([f1, f2])
        self.assertIs(v.provisional, True)
        self.assertEqual(v.confidence, 0.40)
        self.assertIn("PROVISIONAL", v.rationale)
        self.assertIn("prot=0.99", v.rationale)

    def test_refs_contain_finding_ids(self):
        f1 = {"id": "alpha", "text": "the cache is populated with entries records rows"}
        f2 = {"id": "beta", "text": "the cache is not populated with entries records rows"}
        results = run_contradiction([f1, f2])
        self.assertEqual(len(results), 1)
        self.assertIn("alpha", results[0].refs)
        self.assertIn("beta", results[0].refs)


# ---------------------------------------------------------------------------
# Provenance checks
# ---------------------------------------------------------------------------

class TestRunProvenance(unittest.TestCase):
    def test_empty_findings_returns_empty(self):
        self.assertEqual(run_provenance([], [_audit("tool_a", "some text")]), [])

    def test_none_findings_returns_empty(self):
        self.assertEqual(run_provenance(None, []), [])

    def test_non_observed_findings_skipped(self):
        # Only "observed" type is checked
        f = {"text": "something important", "type": "hypothesis", "source": ""}
        self.assertEqual(run_provenance([f], []), [])

    def test_observed_with_empty_audit_flagged(self):
        f = {"text": "the host contacted external.example.com endpoint", "type": "observed", "source": "netscan"}
        results = run_provenance([f], [])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].verdict_type, "unsupported_observed")
        self.assertEqual(results[0].decision, "warn")

    def test_observed_supported_by_receipt_not_flagged(self):
        f = {"text": "the host queried external.example.com endpoint", "type": "observed", "source": "netscan"}
        audit = {"tool": "netscan", "text": "the host queried external.example.com endpoint output"}
        self.assertEqual(run_provenance([f], [audit]), [])

    def test_observed_name_mismatch_but_strong_text_overlap(self):
        # No source in finding → supported by text overlap alone
        f = {"text": "the binary executable.exe launched from temp path directory", "type": "observed", "source": ""}
        audit = {"tool": "procmon", "text": "the binary executable.exe launched from temp path directory output"}
        self.assertEqual(run_provenance([f], [audit]), [])

    def test_observed_name_match_but_no_text_overlap_flagged(self):
        f = {"text": "the process contacted endpoint external.example.com alive", "type": "observed", "source": "procmon"}
        audit = {"tool": "procmon", "text": "unrelated firewall database query result"}
        results = run_provenance([f], [audit])
        self.assertEqual(len(results), 1)

    def test_observed_name_match_with_stopword_only_overlap_is_still_flagged(self):
        f = {"text": "the breach", "type": "observed", "source": "procmon"}
        audit = {"tool": "procmon", "text": "the audit bucket allowed unrestricted access"}
        results = run_provenance([f], [audit])
        self.assertEqual(len(results), 1)

    def test_malformed_audit_entry_skipped(self):
        f = {"text": "the host contacted external.example.com endpoint", "type": "observed", "source": "netscan"}
        results = run_provenance([f], ["not a dict", None])
        self.assertEqual(len(results), 1)  # Still flagged — malformed entries don't count as receipts

    def test_malformed_finding_skipped(self):
        results = run_provenance(["not a dict", None], [])
        self.assertEqual(results, [])

    def test_finding_with_empty_text_skipped(self):
        f = {"text": "", "type": "observed", "source": "tool"}
        self.assertEqual(run_provenance([f], []), [])

    def test_only_one_verdict_per_unsupported_finding(self):
        f = {"text": "the host contacted external.example.com endpoint", "type": "observed", "source": "netscan"}
        results = run_provenance([f], [])
        self.assertEqual(len(results), 1)


# ---------------------------------------------------------------------------
# Contagion checks
# ---------------------------------------------------------------------------

class TestFindContamination(unittest.TestCase):
    def _no_entities(self, text):
        return {}

    def _entities_from_urls(self, text):
        import re
        urls = re.findall(r"https?://\S+", text)
        return {"urls": urls} if urls else {}

    def test_empty_seeds_returns_empty(self):
        result = find_contamination([], [], entities_of=self._no_entities)
        self.assertEqual(result["contaminated_ids"], [])
        self.assertEqual(result["reasons"], {})

    def test_seeds_always_included(self):
        result = find_contamination(["seed-1"], [], entities_of=self._no_entities)
        self.assertIn("seed-1", result["contaminated_ids"])
        self.assertIn("seed-1", result["reasons"])
        self.assertIn("seed", result["reasons"]["seed-1"])

    def test_seed_not_in_findings_still_returned(self):
        # Seed id doesn't match any finding — still in output as a seed
        findings = [{"id": "f1", "text": "unrelated text"}]
        result = find_contamination(["missing-seed"], findings, entities_of=self._no_entities)
        self.assertIn("missing-seed", result["contaminated_ids"])

    def test_entity_anchor_contaminates_sharing_url(self):
        seed_text = "the server called http://fake-hallucinated.internal/api endpoint"
        findings = [
            {"id": "seed", "text": seed_text},
            {"id": "child", "text": "infrastructure built on http://fake-hallucinated.internal/api endpoint"},
        ]
        result = find_contamination(
            ["seed"], findings, entities_of=self._entities_from_urls
        )
        self.assertIn("child", result["contaminated_ids"])
        child_reasons = result["reasons"].get("child", [])
        self.assertTrue(any("entity:" in r for r in child_reasons))

    def test_no_shared_entity_not_contaminated(self):
        findings = [
            {"id": "seed", "text": "http://fake.internal/a endpoint"},
            {"id": "unrelated", "text": "http://real.example.com/b endpoint"},
        ]
        result = find_contamination(
            ["seed"], findings, entities_of=self._entities_from_urls
        )
        self.assertNotIn("unrelated", result["contaminated_ids"])

    def test_semantic_neighbor_included(self):
        result = find_contamination(
            ["seed"], [], entities_of=self._no_entities, semantic_neighbor_ids=["sem-1", "sem-2"]
        )
        self.assertIn("sem-1", result["contaminated_ids"])
        self.assertIn("sem-2", result["contaminated_ids"])
        self.assertIn("semantic", result["reasons"].get("sem-1", []))

    def test_derivation_chain_propagates(self):
        findings = [
            {"id": "seed", "text": "hallucinated endpoint"},
            {"id": "child", "text": "uses seed", "derived_from": "seed"},
            {"id": "grandchild", "text": "uses child", "derived_from": "child"},
        ]
        result = find_contamination(["seed"], findings, entities_of=self._no_entities)
        self.assertIn("child", result["contaminated_ids"])
        self.assertIn("grandchild", result["contaminated_ids"])

    def test_derivation_cycle_guarded(self):
        # A -> B -> A cycle should not loop infinitely
        findings = [
            {"id": "a", "text": "first", "derived_from": "b"},
            {"id": "b", "text": "second", "derived_from": "a"},
        ]
        # No seed overlap → just seeds returned, no infinite loop, and the
        # cycle itself must not mark a or b contaminated.
        result = find_contamination(["seed"], findings, entities_of=self._no_entities)
        self.assertEqual(result["contaminated_ids"], ["seed"])
        self.assertEqual(result["reasons"], {"seed": ["seed"]})

    def test_derivation_cycle_that_reaches_seed_is_contaminated(self):
        findings = [
            {"id": "a", "text": "first", "derived_from": ["b", "seed"]},
            {"id": "b", "text": "second", "derived_from": "a"},
        ]
        result = find_contamination(["seed"], findings, entities_of=self._no_entities)
        self.assertEqual(result["contaminated_ids"], ["seed", "a", "b"])
        self.assertEqual(result["reasons"]["a"], ["derived_from:seed"])
        self.assertEqual(result["reasons"]["b"], ["derived_from:a"])

    def test_derived_from_as_string(self):
        findings = [
            {"id": "seed", "text": "hallucinated"},
            {"id": "child", "text": "derived", "derived_from": "seed"},
        ]
        result = find_contamination(["seed"], findings, entities_of=self._no_entities)
        self.assertIn("child", result["contaminated_ids"])

    def test_derived_from_as_list(self):
        findings = [
            {"id": "seed", "text": "hallucinated"},
            {"id": "child", "text": "derived", "derived_from": ["seed", "other"]},
        ]
        result = find_contamination(["seed"], findings, entities_of=self._no_entities)
        self.assertIn("child", result["contaminated_ids"])

    def test_malformed_finding_skipped(self):
        result = find_contamination(
            ["seed"], ["not a dict", None], entities_of=self._no_entities
        )
        self.assertIn("seed", result["contaminated_ids"])

    def test_entities_of_raises_does_not_propagate(self):
        def bad_extractor(text):
            raise RuntimeError("extractor failed")

        findings = [
            {"id": "seed", "text": "seed text"},
            {"id": "f1", "text": "some text", "derived_from": "seed"},
            {"id": "f2", "text": "unrelated text"},
        ]
        # Should not raise; a finding whose extraction failed still takes part
        # in derivation propagation, and the seed is kept.
        result = find_contamination(["seed"], findings, entities_of=bad_extractor)
        self.assertEqual(result["contaminated_ids"], ["seed", "f1"])
        self.assertEqual(
            result["reasons"], {"seed": ["seed"], "f1": ["derived_from:seed"]}
        )

    def test_entities_of_raising_for_one_finding_keeps_the_others_anchored(self):
        def flaky(text):
            if "boom" in text:
                raise RuntimeError("extractor failed")
            return self._entities_from_urls(text)

        findings = [
            {"id": "seed", "text": "calls http://fake.internal/api endpoint"},
            {"id": "bad", "text": "boom http://fake.internal/api endpoint"},
            {"id": "child", "text": "built on http://fake.internal/api endpoint"},
        ]
        result = find_contamination(["seed"], findings, entities_of=flaky)
        self.assertEqual(result["contaminated_ids"], ["seed", "child"])
        self.assertEqual(result["reasons"]["child"], ["entity:http://fake.internal/api"])

    def test_seeds_appear_first_in_output(self):
        findings = [
            {"id": "seed", "text": "http://h.internal/a endpoint"},
            {"id": "extra", "text": "http://h.internal/a endpoint"},
        ]
        result = find_contamination(
            ["seed"], findings, entities_of=self._entities_from_urls
        )
        self.assertEqual(result["contaminated_ids"][0], "seed")


if __name__ == "__main__":
    unittest.main()
