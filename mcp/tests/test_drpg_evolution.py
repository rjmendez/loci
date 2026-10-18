"""
Tests for mcp/drpg_evolution.py.

Covers:
- TrajectoryStore push + read roundtrip
- TrajectoryScorer.score with known steps (math verified)
- PolicyPromoter.decide at boundary thresholds
- DrpgSuggestionGenerator.generate with high backtrack count
- evolve_policy end-to-end with a temp dir

Mutation-verification notes (run each to confirm red, then reverted):
- TrajectoryScorer.score: removed `- abs(r_ovp)` term → test_scorer_known_math fails
- PolicyPromoter.decide: changed `> 0.7` to `>= 0.7` → test_promoter_boundary_* fail
- DrpgSuggestionGenerator.generate: removed backtrack branch → test_suggestion_backtrack fails
- evolve_policy: removed `policy.metadata.version += 1` → test_evolve_promote_bumps_version fails
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from policy import Policy, PolicyBackend, PolicyTier, PolicyRegion
from drpg_evolution import (
    TrajectoryStore,
    TrajectoryScorer,
    PolicyPromoter,
    DrpgSuggestionGenerator,
    evolve_policy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(action: str = "SEARCH", r_ret=0.0, r_ovp=0.0, r_bkp=0.0,
          r_ref=0.0, r_step=0.0, r_cor=0.0) -> dict:
    return {
        "action": action,
        "reward": {
            "r_ret": r_ret,
            "r_ovp": r_ovp,
            "r_bkp": r_bkp,
            "r_ref": r_ref,
            "r_step": r_step,
            "r_cor": r_cor,
        },
    }


def _stored_policy(backend: PolicyBackend, title: str = "Test") -> Policy:
    p = Policy(
        title=title,
        tier=PolicyTier.MEDIUM,
        region=PolicyRegion.MEMORY,
        config={"discovery_mode": True},
    )
    backend.store(p)
    return p


# ---------------------------------------------------------------------------
# TrajectoryStore
# ---------------------------------------------------------------------------

class TestTrajectoryStore:

    def test_push_creates_jsonl_file(self, tmp_path):
        store = TrajectoryStore(tmp_path / "traj")
        store.push("p1", {"step": 0, "action": "SEARCH", "reward": {"r_ret": 1.0}})
        assert (tmp_path / "traj" / "p1.jsonl").exists()

    def test_read_returns_empty_for_missing_policy(self, tmp_path):
        store = TrajectoryStore(tmp_path / "traj")
        assert store.read("nonexistent") == []

    def test_push_and_read_roundtrip(self, tmp_path):
        store = TrajectoryStore(tmp_path / "traj")
        d1 = {"step": 0, "action": "SEARCH", "reward": {"r_ret": 0.5}}
        d2 = {"step": 1, "action": "BACKTRACK", "reward": {"r_bkp": -0.2}}
        store.push("pol-abc", d1)
        store.push("pol-abc", d2)

        rows = store.read("pol-abc")
        assert len(rows) == 2
        assert rows[0]["action"] == "SEARCH"
        assert rows[0]["reward"]["r_ret"] == 0.5
        assert rows[1]["action"] == "BACKTRACK"

    def test_multiple_policies_are_independent(self, tmp_path):
        store = TrajectoryStore(tmp_path / "traj")
        store.push("alpha", {"step": 0, "action": "SEARCH"})
        store.push("beta", {"step": 0, "action": "REFUSE"})

        alpha = store.read("alpha")
        beta = store.read("beta")
        assert len(alpha) == 1
        assert alpha[0]["action"] == "SEARCH"
        assert len(beta) == 1
        assert beta[0]["action"] == "REFUSE"

    def test_list_policies_returns_ids(self, tmp_path):
        store = TrajectoryStore(tmp_path / "traj")
        store.push("x1", {"step": 0})
        store.push("x2", {"step": 0})

        ids = set(store.list_policies())
        assert "x1" in ids
        assert "x2" in ids

    def test_list_policies_empty_when_no_files(self, tmp_path):
        store = TrajectoryStore(tmp_path / "traj")
        assert store.list_policies() == []


# ---------------------------------------------------------------------------
# TrajectoryScorer
# ---------------------------------------------------------------------------

class TestTrajectoryScorer:

    def test_score_empty_steps_returns_zero(self):
        scorer = TrajectoryScorer()
        assert scorer.score([]) == 0.0

    def test_scorer_known_math_single_step(self):
        """
        Formula: r_ret - abs(r_ovp) - abs(r_bkp) + r_ref - abs(r_step) + r_cor
        With: r_ret=1.0, r_ovp=-0.1, r_bkp=0.0, r_ref=0.0, r_step=-0.05, r_cor=0.5
        Expected per-step: 1.0 - 0.1 - 0.0 + 0.0 - 0.05 + 0.5 = 1.35
        Normalized by 1: 1.35
        """
        scorer = TrajectoryScorer()
        step = _step(r_ret=1.0, r_ovp=-0.1, r_bkp=0.0, r_ref=0.0, r_step=-0.05, r_cor=0.5)
        assert scorer.score([step]) == pytest.approx(1.35, abs=1e-9)

    def test_scorer_known_math_two_steps(self):
        """
        Step 1: 1.0 - 0.1 - 0.0 + 0.0 - 0.05 + 0.5 = 1.35
        Step 2: 0.5 - 0.2 - 0.2 + 0.0 - 0.1 + 0.0 = 0.0
        Total = 1.35; normalized by 2 = 0.675
        """
        scorer = TrajectoryScorer()
        s1 = _step(r_ret=1.0, r_ovp=-0.1, r_bkp=0.0, r_ref=0.0, r_step=-0.05, r_cor=0.5)
        s2 = _step(r_ret=0.5, r_ovp=-0.2, r_bkp=-0.2, r_ref=0.0, r_step=-0.1, r_cor=0.0)
        assert scorer.score([s1, s2]) == pytest.approx(0.675, abs=1e-9)

    def test_scorer_treats_abs_of_negative_ovp(self):
        """abs(r_ovp) with r_ovp=0.2 and r_ovp=-0.2 must produce the same deduction."""
        scorer = TrajectoryScorer()
        s_pos = _step(r_ovp=0.2)
        s_neg = _step(r_ovp=-0.2)
        assert scorer.score([s_pos]) == pytest.approx(scorer.score([s_neg]), abs=1e-9)


# ---------------------------------------------------------------------------
# PolicyPromoter
# ---------------------------------------------------------------------------

class TestPolicyPromoter:

    def _policy(self):
        return Policy(
            title="Promo Test",
            tier=PolicyTier.LIGHT,
            region=PolicyRegion.MEMORY,
        )

    def test_promote_above_threshold(self):
        promoter = PolicyPromoter()
        result = promoter.decide(0.71, self._policy())
        assert result["action"] == "promote"
        assert result["score"] == pytest.approx(0.71)

    def test_keep_at_boundary_above_demote(self):
        promoter = PolicyPromoter()
        result = promoter.decide(0.5, self._policy())
        assert result["action"] == "keep"
        assert result["score"] == pytest.approx(0.5)

    def test_demote_below_threshold(self):
        promoter = PolicyPromoter()
        result = promoter.decide(0.19, self._policy())
        assert result["action"] == "demote"
        assert result["score"] == pytest.approx(0.19)

    def test_exactly_at_promote_threshold_is_keep(self):
        """score == 0.7 is NOT > 0.7, so should be keep."""
        promoter = PolicyPromoter()
        result = promoter.decide(0.7, self._policy())
        assert result["action"] == "keep"

    def test_exactly_at_demote_threshold_is_keep(self):
        """score == 0.2 is NOT < 0.2, so should be keep."""
        promoter = PolicyPromoter()
        result = promoter.decide(0.2, self._policy())
        assert result["action"] == "keep"

    def test_decision_includes_reason_string(self):
        promoter = PolicyPromoter()
        result = promoter.decide(0.8, self._policy())
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0


# ---------------------------------------------------------------------------
# DrpgSuggestionGenerator
# ---------------------------------------------------------------------------

class TestDrpgSuggestionGenerator:

    def _policy(self):
        return Policy(title="Sug Test", tier=PolicyTier.MEDIUM, config={"discovery_mode": True})

    def test_empty_steps_returns_no_suggestions(self):
        gen = DrpgSuggestionGenerator()
        result = gen.generate([], self._policy())
        assert result == []

    def test_high_backtrack_rate_produces_hop_suggestion(self):
        """31% backtrack rate (4/13 → above 30% threshold) must trigger hop suggestion."""
        gen = DrpgSuggestionGenerator()
        steps = [_step(action="BACKTRACK")] * 4 + [_step(action="SEARCH")] * 9
        suggestions = gen.generate(steps, self._policy())
        affects = [s["affects"] for s in suggestions]
        assert "max_retrieval_hops" in affects

    def test_low_backtrack_rate_no_hop_suggestion(self):
        """10% backtrack rate (1/10) must NOT trigger hop suggestion."""
        gen = DrpgSuggestionGenerator()
        steps = [_step(action="BACKTRACK")] * 1 + [_step(action="SEARCH")] * 9
        suggestions = gen.generate(steps, self._policy())
        affects = [s["affects"] for s in suggestions]
        assert "max_retrieval_hops" not in affects

    def test_high_overlap_produces_dedup_suggestion(self):
        gen = DrpgSuggestionGenerator()
        steps = [_step(r_ovp=-0.4)] * 5
        suggestions = gen.generate(steps, self._policy())
        affects = [s["affects"] for s in suggestions]
        assert "dedup_filter" in affects

    def test_suggestion_confidence_is_float_in_range(self):
        gen = DrpgSuggestionGenerator()
        steps = [_step(action="BACKTRACK")] * 5 + [_step(action="SEARCH")] * 5
        suggestions = gen.generate(steps, self._policy())
        for s in suggestions:
            assert isinstance(s["confidence"], float)
            assert 0.0 < s["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# evolve_policy end-to-end
# ---------------------------------------------------------------------------

class TestEvolvePolicy:

    def test_evolve_returns_error_for_missing_policy(self, tmp_path):
        backend = PolicyBackend(tmp_path / "policies")
        store = TrajectoryStore(tmp_path / "traj")
        result = evolve_policy("does-not-exist", backend, store)
        assert result["action"] == "error"
        assert result["error"] == "policy_not_found"
        assert result["policy_id"] == "does-not-exist"

    def test_evolve_keep_with_moderate_score(self, tmp_path):
        backend = PolicyBackend(tmp_path / "policies")
        store = TrajectoryStore(tmp_path / "traj")
        policy = _stored_policy(backend)

        # Push steps that score approximately 0.5 (keep range)
        # Per-step: r_ret=0.5, everything else 0 → per-step = 0.5, normalized = 0.5
        for _ in range(3):
            store.push(policy.policy_id, _step(r_ret=0.5))

        result = evolve_policy(policy.policy_id, backend, store)
        assert result["policy_id"] == policy.policy_id
        assert result["action"] == "keep"
        assert result["steps_analyzed"] == 3
        assert result["score"] == pytest.approx(0.5, abs=1e-9)

    def test_evolve_promote_bumps_version_and_sets_source(self, tmp_path):
        backend = PolicyBackend(tmp_path / "policies")
        store = TrajectoryStore(tmp_path / "traj")
        policy = _stored_policy(backend)
        original_version = policy.metadata.version

        # Push steps scoring > 0.7: r_ret=1.0, r_cor=0.5 → per-step = 1.5 → mean = 1.5
        for _ in range(2):
            store.push(policy.policy_id, _step(r_ret=1.0, r_cor=0.5))

        result = evolve_policy(policy.policy_id, backend, store)
        assert result["action"] == "promote"

        reloaded = backend.load(policy.policy_id)
        assert reloaded.metadata.source == "drpg-evolved"
        assert reloaded.metadata.version == original_version + 1

    def test_evolve_demote_does_not_modify_policy(self, tmp_path):
        backend = PolicyBackend(tmp_path / "policies")
        store = TrajectoryStore(tmp_path / "traj")
        policy = _stored_policy(backend)
        original_version = policy.metadata.version

        # Push steps scoring < 0.2: everything zero except r_ovp=-0.5 → per-step = -0.5
        for _ in range(3):
            store.push(policy.policy_id, _step(r_ovp=-0.5))

        result = evolve_policy(policy.policy_id, backend, store)
        assert result["action"] == "demote"

        reloaded = backend.load(policy.policy_id)
        assert reloaded.metadata.version == original_version  # unchanged on demote

    def test_evolve_returns_suggestions_list(self, tmp_path):
        backend = PolicyBackend(tmp_path / "policies")
        store = TrajectoryStore(tmp_path / "traj")
        policy = _stored_policy(backend)

        store.push(policy.policy_id, _step(r_ret=0.5))
        result = evolve_policy(policy.policy_id, backend, store)
        assert isinstance(result["suggestions"], list)

    def test_evolve_steps_analyzed_matches_trajectory_length(self, tmp_path):
        backend = PolicyBackend(tmp_path / "policies")
        store = TrajectoryStore(tmp_path / "traj")
        policy = _stored_policy(backend)

        for i in range(7):
            store.push(policy.policy_id, _step(r_ret=0.3))

        result = evolve_policy(policy.policy_id, backend, store)
        assert result["steps_analyzed"] == 7
