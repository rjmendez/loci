"""
Tests for the 7-tier PolicyTier ladder, TIER_MODEL_MAP, resolve_model_for_tier,
and StagePolicy.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from policy import (
    TIER_MODEL_MAP,
    PolicyTier,
    StagePolicy,
    resolve_model_for_tier,
)

_REQUIRED_CONSTRAINT_KEYS = {
    "max_context_chars",
    "max_retrieval_hops",
    "max_refinement_steps",
    "max_perspectives",
    "max_fanout",
    "session_limit",
}

_ALL_TIERS = [
    PolicyTier.ZERO,
    PolicyTier.NANO,
    PolicyTier.LIGHT,
    PolicyTier.STANDARD,
    PolicyTier.TURBO,
    PolicyTier.HEAVY,
    PolicyTier.RESEARCH,
]


class TestTierConstraints(unittest.TestCase):
    """All 7 tiers expose a complete constraints dict."""

    def test_all_tiers_have_required_constraint_keys(self):
        for tier in _ALL_TIERS:
            with self.subTest(tier=tier.value):
                c = tier.constraints()
                missing = _REQUIRED_CONSTRAINT_KEYS - set(c.keys())
                self.assertFalse(missing, f"{tier.value} missing keys: {missing}")

    def test_zero_tier_max_context_chars_is_zero(self):
        self.assertEqual(PolicyTier.ZERO.constraints()["max_context_chars"], 0)

    def test_nano_tier_max_context_chars_is_512(self):
        self.assertEqual(PolicyTier.NANO.constraints()["max_context_chars"], 512)

    def test_research_tier_session_limit_is_1(self):
        self.assertEqual(PolicyTier.RESEARCH.constraints()["session_limit"], 1)

    def test_non_research_tiers_have_unlimited_session(self):
        for tier in _ALL_TIERS:
            if tier == PolicyTier.RESEARCH:
                continue
            with self.subTest(tier=tier.value):
                self.assertEqual(tier.constraints()["session_limit"], -1)

    def test_tier_ordering_by_max_context(self):
        """Context budget increases monotonically from ZERO through RESEARCH."""
        prev = -1
        for tier in _ALL_TIERS:
            cur = tier.constraints()["max_context_chars"]
            self.assertGreaterEqual(cur, prev, f"{tier.value} context not >= previous")
            prev = cur


class TestBackwardCompatMedium(unittest.TestCase):
    """MEDIUM alias maps to STANDARD semantics."""

    def test_medium_enum_value_parseable(self):
        t = PolicyTier("medium")
        self.assertEqual(t, PolicyTier.MEDIUM)

    def test_medium_same_model_list_as_standard(self):
        self.assertEqual(TIER_MODEL_MAP[PolicyTier.MEDIUM], TIER_MODEL_MAP[PolicyTier.STANDARD])

    def test_medium_constraints_equal_standard(self):
        self.assertEqual(PolicyTier.MEDIUM.constraints(), PolicyTier.STANDARD.constraints())


class TestResolveModelForTier(unittest.TestCase):
    """resolve_model_for_tier picks a model correctly."""

    def test_zero_tier_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, r"ZERO"):
            resolve_model_for_tier(PolicyTier.ZERO)

    def test_nano_tier_returns_ollama_model(self):
        model = resolve_model_for_tier(PolicyTier.NANO)
        self.assertTrue(model.startswith("ollama/"), f"Expected ollama/ prefix, got {model}")

    def test_light_returns_first_model(self):
        model = resolve_model_for_tier(PolicyTier.LIGHT)
        self.assertIn(model, TIER_MODEL_MAP[PolicyTier.LIGHT])

    def test_available_filter_selects_matching(self):
        available = ["gpt-5.4-mini", "claude-sonnet-4.6"]
        model = resolve_model_for_tier(PolicyTier.STANDARD, available=available)
        self.assertIn(model, available)

    def test_available_filter_falls_back_when_none_match(self):
        model = resolve_model_for_tier(PolicyTier.HEAVY, available=["nonexistent-model"])
        self.assertIn(model, TIER_MODEL_MAP[PolicyTier.HEAVY])


class TestStagePolicy(unittest.TestCase):
    """StagePolicy tier_for respects stage mapping and outer ceiling."""

    def _sp(self, stages, outer):
        return StagePolicy(stages=stages, outer_tier=outer)

    def test_stage_in_stages_returns_correct_tier(self):
        sp = self._sp({"decompose": PolicyTier.NANO, "synthesis": PolicyTier.HEAVY}, PolicyTier.HEAVY)
        self.assertEqual(sp.tier_for("decompose"), PolicyTier.NANO)

    def test_stage_not_in_stages_returns_outer_tier(self):
        sp = self._sp({"decompose": PolicyTier.NANO}, PolicyTier.STANDARD)
        self.assertEqual(sp.tier_for("unknown_stage"), PolicyTier.STANDARD)

    def test_stage_exceeds_outer_is_clamped(self):
        # synthesis=HEAVY but outer=LIGHT → must clamp to LIGHT
        sp = self._sp({"synthesis": PolicyTier.HEAVY}, PolicyTier.LIGHT)
        self.assertEqual(sp.tier_for("synthesis"), PolicyTier.LIGHT)

    def test_stage_equal_to_outer_is_not_clamped(self):
        sp = self._sp({"synthesis": PolicyTier.STANDARD}, PolicyTier.STANDARD)
        self.assertEqual(sp.tier_for("synthesis"), PolicyTier.STANDARD)

    def test_stage_below_outer_is_not_upgraded(self):
        sp = self._sp({"reflect": PolicyTier.NANO}, PolicyTier.HEAVY)
        self.assertEqual(sp.tier_for("reflect"), PolicyTier.NANO)


class TestStagePolicyForSwarmReason(unittest.TestCase):
    """StagePolicy.for_swarm_reason factory produces correct defaults."""

    def test_default_outer_is_standard(self):
        sp = StagePolicy.for_swarm_reason()
        self.assertEqual(sp.outer_tier, PolicyTier.STANDARD)

    def test_synthesis_stage_defaults_to_heavy(self):
        sp = StagePolicy.for_swarm_reason(outer=PolicyTier.HEAVY)
        self.assertEqual(sp.tier_for("synthesis"), PolicyTier.HEAVY)

    def test_synthesis_clamped_to_light_when_outer_is_light(self):
        sp = StagePolicy.for_swarm_reason(outer=PolicyTier.LIGHT)
        # synthesis=HEAVY > LIGHT → clamped
        self.assertEqual(sp.tier_for("synthesis"), PolicyTier.LIGHT)

    def test_decompose_is_nano_when_outer_allows(self):
        sp = StagePolicy.for_swarm_reason(outer=PolicyTier.STANDARD)
        self.assertEqual(sp.tier_for("decompose"), PolicyTier.NANO)

    def test_all_swarm_stages_present(self):
        sp = StagePolicy.for_swarm_reason()
        for stage in ("decompose", "fanout", "self_consistency", "escalation", "synthesis"):
            with self.subTest(stage=stage):
                self.assertIn(stage, sp.stages)


class TestStagePolicyForInvestigation(unittest.TestCase):
    """StagePolicy.for_investigation always sets pre_answer_check=ZERO."""

    def test_pre_answer_check_is_zero_regardless_of_outer(self):
        for outer in [PolicyTier.LIGHT, PolicyTier.STANDARD, PolicyTier.HEAVY]:
            with self.subTest(outer=outer.value):
                sp = StagePolicy.for_investigation(outer=outer)
                # ZERO <= any outer, so it should never be clamped up
                self.assertEqual(sp.tier_for("pre_answer_check"), PolicyTier.ZERO)

    def test_reason_stage_is_standard(self):
        sp = StagePolicy.for_investigation(outer=PolicyTier.HEAVY)
        self.assertEqual(sp.tier_for("reason"), PolicyTier.STANDARD)

    def test_reflect_and_verify_are_nano(self):
        sp = StagePolicy.for_investigation(outer=PolicyTier.STANDARD)
        self.assertEqual(sp.tier_for("reflect"), PolicyTier.NANO)
        self.assertEqual(sp.tier_for("verify"), PolicyTier.NANO)


if __name__ == "__main__":
    unittest.main()
