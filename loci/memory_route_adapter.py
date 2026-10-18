"""
Adapter: memory_route_policy_optimize -> Loci Policy object.

Converts the routing optimizer's recommendation into a structured
Policy stored via PolicyBackend, feeding it into the DRPG evolution loop.

Also provides counterfactual_gate: wraps memory_route_counterfactual_simulate
to validate tier downgrades before committing them.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class RouteOptimizationResult:
    """Parsed output from memory_route_policy_optimize."""
    optimized: bool
    reason: str
    baseline_objective: float
    best_objective: float
    score_delta: float
    recommendation: Optional[Dict[str, Any]]  # {top_k, deduplicate, dedup_threshold}
    provenance_aggregation: Dict[str, Any] = field(default_factory=dict)
    top_candidates: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RouteOptimizationResult":
        return cls(
            optimized=d.get("optimized", False),
            reason=d.get("reason", ""),
            baseline_objective=float(d.get("baseline_objective", 0.0)),
            best_objective=float(d.get("best_objective", 0.0)),
            score_delta=float(d.get("score_delta", 0.0)),
            recommendation=d.get("recommendation"),
            provenance_aggregation=d.get("provenance_aggregation", {}),
            top_candidates=d.get("top_candidates", []),
        )

    def to_policy_dict(self) -> Optional[Dict[str, Any]]:
        """
        Convert recommendation to a Policy-compatible dict.

        Maps routing optimizer metrics to PolicyTier and EVO-RAG reward weights:
        - high stability score -> HEAVY tier (trust existing routing)
        - high compression + low removal -> MEDIUM tier (tightening ok)
        - high score_delta + low baseline -> LIGHT tier (aggressive optimization)
        """
        if not self.recommendation:
            return None

        # Map score_delta and stability to tier
        if self.score_delta > 0.3 and self.baseline_objective < 0.5:
            tier = "light"
        elif self.score_delta > 0.1:
            tier = "medium"
        else:
            tier = "heavy"

        return {
            "title": f"memory_route_policy_optimize result (delta={self.score_delta:.3f})",
            "tier": tier,
            "region": "memory",
            "labels": ["auto-generated", "memory-routing"],
            "config": {
                "top_k": self.recommendation.get("top_k"),
                "deduplicate": self.recommendation.get("deduplicate"),
                "dedup_threshold": self.recommendation.get("dedup_threshold"),
                "baseline_objective": self.baseline_objective,
                "best_objective": self.best_objective,
                "score_delta": self.score_delta,
                "discovery_mode": False,  # optimization result -> refinement phase
                "provenance": self.provenance_aggregation,
            },
            "metadata": {
                "source": "loci.memory_route_policy_optimize",
            },
        }


def adapt_route_optimization_to_policy(
    optimize_result_json: str,
    backend=None,  # PolicyBackend instance, optional (for testing)
) -> Dict[str, Any]:
    """
    Parse memory_route_policy_optimize output and store as a Policy.

    Args:
        optimize_result_json: JSON string from memory_route_policy_optimize
        backend: PolicyBackend instance; if None, creates one at default path

    Returns:
        {
            "adapted": bool,
            "policy_id": str or None,
            "tier": str or None,
            "score_delta": float,
            "reason": str,
        }
    """
    try:
        result_dict = json.loads(optimize_result_json)
    except json.JSONDecodeError as e:
        return {"adapted": False, "policy_id": None, "tier": None, "score_delta": 0.0,
                "reason": f"json_parse_error: {e}"}

    result = RouteOptimizationResult.from_dict(result_dict)

    if not result.optimized or not result.recommendation:
        return {
            "adapted": False,
            "policy_id": None,
            "tier": None,
            "score_delta": result.score_delta,
            "reason": result.reason or "no_recommendation",
        }

    policy_dict = result.to_policy_dict()
    if not policy_dict:
        return {"adapted": False, "policy_id": None, "tier": None,
                "score_delta": result.score_delta, "reason": "empty_policy_dict"}

    from loci.policy import Policy, PolicyBackend

    if backend is None:
        backend = PolicyBackend()

    policy = Policy.from_dict(policy_dict)
    backend.store(policy)

    logger.info(f"Stored route optimization as policy {policy.policy_id} (tier={policy.tier.value})")
    return {
        "adapted": True,
        "policy_id": policy.policy_id,
        "tier": policy.tier.value,
        "score_delta": result.score_delta,
        "reason": "stored_ok",
    }


def evaluate_counterfactual_gate(
    simulate_result_json: str,
    max_changed_fraction: float = 0.05,
) -> Dict[str, Any]:
    """
    Parse memory_route_counterfactual_simulate output and return go/no-go for tier downgrade.

    A downgrade is safe if the fraction of changed routing decisions is below max_changed_fraction.

    Args:
        simulate_result_json: JSON string from memory_route_counterfactual_simulate
        max_changed_fraction: max fraction of decisions that may change (default 5%)

    Returns:
        {
            "safe_to_downgrade": bool,
            "changed_fraction": float,
            "changed": int,
            "total": int,
            "reason": str,
        }
    """
    try:
        result = json.loads(simulate_result_json)
    except json.JSONDecodeError as e:
        return {"safe_to_downgrade": False, "changed_fraction": 1.0,
                "changed": -1, "total": -1, "reason": f"json_parse_error: {e}"}

    changed = result.get("changed", 0)
    total_results = len(result.get("results", []))
    changed_fraction = changed / total_results if total_results > 0 else 0.0
    safe = changed_fraction <= max_changed_fraction

    return {
        "safe_to_downgrade": safe,
        "changed_fraction": round(changed_fraction, 4),
        "changed": changed,
        "total": total_results,
        "reason": (
            f"changed_fraction={changed_fraction:.2%} <= {max_changed_fraction:.2%}"
            if safe
            else f"changed_fraction={changed_fraction:.2%} > {max_changed_fraction:.2%} -- too disruptive"
        ),
    }
