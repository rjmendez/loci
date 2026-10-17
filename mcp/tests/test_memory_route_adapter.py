"""
Tests for mcp/memory_route_adapter.py.

Covers:
- RouteOptimizationResult.from_dict: parses all fields correctly
- RouteOptimizationResult.to_policy_dict: tier mapping from score_delta + baseline_objective
- adapt_route_optimization_to_policy: full adaptation path (success, no-op, bad JSON)
- evaluate_counterfactual_gate: go/no-go logic with various changed fractions
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory_route_adapter import (
    RouteOptimizationResult,
    adapt_route_optimization_to_policy,
    evaluate_counterfactual_gate,
)
from policy import Policy, PolicyBackend, PolicyTier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_optimize_json(
    optimized=True,
    reason="ok",
    baseline_objective=0.4,
    best_objective=0.8,
    score_delta=0.4,
    recommendation=None,
    provenance_aggregation=None,
    top_candidates=None,
) -> str:
    if recommendation is None:
        recommendation = {"top_k": 5, "deduplicate": True, "dedup_threshold": 0.8}
    return json.dumps({
        "optimized": optimized,
        "reason": reason,
        "baseline_objective": baseline_objective,
        "best_objective": best_objective,
        "score_delta": score_delta,
        "recommendation": recommendation,
        "provenance_aggregation": provenance_aggregation or {"sources": ["audit"]},
        "top_candidates": top_candidates or [],
    })


def _make_simulate_json(changed: int, total: int) -> str:
    results = [{"decision_id": f"d{i}"} for i in range(total)]
    return json.dumps({"changed": changed, "results": results})


# ---------------------------------------------------------------------------
# RouteOptimizationResult.from_dict
# ---------------------------------------------------------------------------

class TestRouteOptimizationResultFromDict:
    def test_parses_all_fields(self):
        d = {
            "optimized": True,
            "reason": "ok",
            "baseline_objective": 0.3,
            "best_objective": 0.75,
            "score_delta": 0.45,
            "recommendation": {"top_k": 7, "deduplicate": False, "dedup_threshold": 0.9},
            "provenance_aggregation": {"sources": ["audit"]},
            "top_candidates": [{"policy": {"top_k": 7}}],
        }
        r = RouteOptimizationResult.from_dict(d)
        assert r.optimized is True
        assert r.reason == "ok"
        assert r.baseline_objective == pytest.approx(0.3)
        assert r.best_objective == pytest.approx(0.75)
        assert r.score_delta == pytest.approx(0.45)
        assert r.recommendation == {"top_k": 7, "deduplicate": False, "dedup_threshold": 0.9}
        assert r.provenance_aggregation == {"sources": ["audit"]}
        assert len(r.top_candidates) == 1

    def test_defaults_on_empty_dict(self):
        r = RouteOptimizationResult.from_dict({})
        assert r.optimized is False
        assert r.reason == ""
        assert r.baseline_objective == pytest.approx(0.0)
        assert r.best_objective == pytest.approx(0.0)
        assert r.score_delta == pytest.approx(0.0)
        assert r.recommendation is None
        assert r.provenance_aggregation == {}
        assert r.top_candidates == []

    def test_numeric_coercion(self):
        r = RouteOptimizationResult.from_dict({
            "baseline_objective": "0.2",
            "best_objective": "0.6",
            "score_delta": "0.4",
        })
        assert isinstance(r.baseline_objective, float)
        assert isinstance(r.score_delta, float)


# ---------------------------------------------------------------------------
# RouteOptimizationResult.to_policy_dict — tier mapping
# ---------------------------------------------------------------------------

class TestToPolicyDict:
    def test_high_delta_low_baseline_gives_light_tier(self):
        r = RouteOptimizationResult.from_dict({
            "optimized": True,
            "score_delta": 0.35,
            "baseline_objective": 0.4,
            "best_objective": 0.75,
            "recommendation": {"top_k": 5, "deduplicate": True, "dedup_threshold": 0.8},
        })
        d = r.to_policy_dict()
        assert d is not None
        assert d["tier"] == "light"

    def test_medium_delta_gives_medium_tier(self):
        r = RouteOptimizationResult.from_dict({
            "optimized": True,
            "score_delta": 0.15,
            "baseline_objective": 0.7,
            "best_objective": 0.85,
            "recommendation": {"top_k": 5, "deduplicate": True, "dedup_threshold": 0.8},
        })
        d = r.to_policy_dict()
        assert d is not None
        assert d["tier"] == "medium"

    def test_low_delta_gives_heavy_tier(self):
        r = RouteOptimizationResult.from_dict({
            "optimized": True,
            "score_delta": 0.05,
            "baseline_objective": 0.9,
            "best_objective": 0.95,
            "recommendation": {"top_k": 5, "deduplicate": True, "dedup_threshold": 0.8},
        })
        d = r.to_policy_dict()
        assert d is not None
        assert d["tier"] == "heavy"

    def test_no_recommendation_returns_none(self):
        r = RouteOptimizationResult.from_dict({"optimized": False, "recommendation": None})
        assert r.to_policy_dict() is None

    def test_policy_dict_contains_required_fields(self):
        r = RouteOptimizationResult.from_dict({
            "optimized": True,
            "score_delta": 0.35,
            "baseline_objective": 0.4,
            "best_objective": 0.75,
            "recommendation": {"top_k": 5, "deduplicate": True, "dedup_threshold": 0.8},
        })
        d = r.to_policy_dict()
        assert "title" in d
        assert d["region"] == "memory"
        assert "auto-generated" in d["labels"]
        assert "memory-routing" in d["labels"]
        assert d["config"]["top_k"] == 5
        assert d["config"]["discovery_mode"] is False


# ---------------------------------------------------------------------------
# adapt_route_optimization_to_policy
# ---------------------------------------------------------------------------

class TestAdaptRouteOptimizationToPolicy:
    def _make_backend(self):
        tmp = tempfile.mkdtemp()
        return PolicyBackend(policy_dir=Path(tmp))

    def test_optimized_true_stores_policy_and_returns_id(self):
        optimize_json = _make_optimize_json(
            optimized=True,
            score_delta=0.35,
            baseline_objective=0.4,
        )
        backend = self._make_backend()
        result = adapt_route_optimization_to_policy(optimize_json, backend=backend)
        assert result["adapted"] is True
        assert result["policy_id"] is not None
        assert result["tier"] in ("light", "medium", "heavy")
        assert result["reason"] == "stored_ok"

    def test_optimized_false_returns_adapted_false(self):
        optimize_json = _make_optimize_json(
            optimized=False,
            reason="insufficient_decisions",
            recommendation=None,
        )
        backend = self._make_backend()
        result = adapt_route_optimization_to_policy(optimize_json, backend=backend)
        assert result["adapted"] is False
        assert result["policy_id"] is None
        assert "insufficient_decisions" in result["reason"]

    def test_bad_json_returns_adapted_false_with_reason(self):
        result = adapt_route_optimization_to_policy("{not valid json}", backend=self._make_backend())
        assert result["adapted"] is False
        assert "json_parse_error" in result["reason"]

    def test_stored_policy_has_correct_tier(self):
        optimize_json = _make_optimize_json(
            optimized=True,
            score_delta=0.05,
            baseline_objective=0.9,
        )
        backend = self._make_backend()
        result = adapt_route_optimization_to_policy(optimize_json, backend=backend)
        assert result["adapted"] is True
        assert result["tier"] == "heavy"


# ---------------------------------------------------------------------------
# evaluate_counterfactual_gate
# ---------------------------------------------------------------------------

class TestEvaluateCounterfactualGate:
    def test_zero_changed_is_safe(self):
        simulate_json = _make_simulate_json(changed=0, total=20)
        result = evaluate_counterfactual_gate(simulate_json, max_changed_fraction=0.05)
        assert result["safe_to_downgrade"] is True
        assert result["changed_fraction"] == pytest.approx(0.0)
        assert result["changed"] == 0
        assert result["total"] == 20

    def test_five_of_twenty_is_not_safe(self):
        simulate_json = _make_simulate_json(changed=5, total=20)
        result = evaluate_counterfactual_gate(simulate_json, max_changed_fraction=0.05)
        assert result["safe_to_downgrade"] is False
        assert result["changed_fraction"] == pytest.approx(0.25)
        assert "too disruptive" in result["reason"]

    def test_bad_json_returns_not_safe(self):
        result = evaluate_counterfactual_gate("{not json}", max_changed_fraction=0.05)
        assert result["safe_to_downgrade"] is False
        assert result["changed"] == -1
        assert "json_parse_error" in result["reason"]

    def test_empty_results_is_safe(self):
        # 0 changed out of 0 results → fraction=0.0 → safe
        simulate_json = json.dumps({"changed": 0, "results": []})
        result = evaluate_counterfactual_gate(simulate_json, max_changed_fraction=0.05)
        assert result["safe_to_downgrade"] is True
        assert result["changed_fraction"] == pytest.approx(0.0)
        assert result["total"] == 0

    def test_custom_threshold_respected(self):
        simulate_json = _make_simulate_json(changed=2, total=20)  # 10%
        # 10% > 5% → not safe
        assert evaluate_counterfactual_gate(simulate_json, max_changed_fraction=0.05)["safe_to_downgrade"] is False
        # 10% <= 15% → safe
        assert evaluate_counterfactual_gate(simulate_json, max_changed_fraction=0.15)["safe_to_downgrade"] is True
