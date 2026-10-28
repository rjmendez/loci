"""
Policy enforcement hooks for the investigation reasoning chain.

Integrates PolicyTier constraints and EVO-RAG reward signals into:
- investigation_pre_answer_check: r_ref refusal reward gate
- investigation_reason: perspectives depth capping
- investigation_verify_all: heuristic pre-pass tier routing
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)


def get_investigation_policy_config() -> Dict[str, Any]:
    """Load investigation policy config from env vars."""
    return {
        "enabled": os.getenv("LOCI_POLICY_ENABLE", "0").lower() in ("1", "true", "yes"),
        "trajectory_dir": Path(os.getenv("LOCI_TRAJECTORY_DIR", "~/.loci/trajectories")).expanduser(),
        "pre_answer_refusal_threshold": float(os.getenv("LOCI_PRE_ANSWER_REFUSAL_THRESHOLD", "0.3")),
        "verify_heuristic_skip_threshold": float(os.getenv("LOCI_VERIFY_SKIP_THRESHOLD", "0.85")),
    }


def cap_perspectives_to_tier(requested_perspectives: int, policy_tier: str) -> int:
    """
    Cap investigation_reason perspectives to PolicyTier.max_retrieval_hops.

    Tier -> max perspectives:
      light  -> 1
      medium -> 3
      heavy  -> 5 (no cap beyond param limit)

    Returns capped value; logs if capping occurred.
    """
    tier_caps = {"light": 1, "medium": 3, "heavy": 5}
    cap = tier_caps.get(policy_tier.lower(), 5)
    capped = min(requested_perspectives, cap)
    if capped < requested_perspectives:
        logger.info(
            f"investigation_reason: perspectives capped {requested_perspectives}->{capped} "
            f"by {policy_tier} tier policy"
        )
    return capped


def evaluate_pre_answer_for_refusal(
    support_summary: Dict[str, Any],
    refusal_threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Evaluate whether investigation_pre_answer_check evidence warrants a refusal.

    Maps to EVO-RAG r_ref: reward for valid refusals when evidence is insufficient.

    Args:
        support_summary: dict with keys like 'supported_count', 'unsupported_count',
                         'ambiguous_count', 'min_chain_confidence'
        refusal_threshold: if supported_ratio <= this, recommend refusal

    Returns:
        {
            "recommend_refusal": bool,
            "r_ref": float,          # 1.0 if refusal recommended, -0.5 if refused unnecessarily
            "supported_ratio": float,
            "reason": str,
        }
    """
    supported = support_summary.get("supported_count", 0)
    unsupported = support_summary.get("unsupported_count", 0)
    ambiguous = support_summary.get("ambiguous_count", 0)
    total = supported + unsupported + ambiguous

    if total == 0:
        return {
            "recommend_refusal": True,
            "r_ref": 1.0,
            "supported_ratio": 0.0,
            "reason": "no_claims_to_evaluate",
        }

    supported_ratio = supported / total
    recommend_refusal = supported_ratio <= refusal_threshold

    # r_ref: positive when refusal is justified, negative when refusing unnecessarily
    r_ref = 1.0 if recommend_refusal else -0.5 * (1.0 - supported_ratio)

    return {
        "recommend_refusal": recommend_refusal,
        "r_ref": round(r_ref, 3),
        "supported_ratio": round(supported_ratio, 3),
        "reason": (
            f"supported_ratio={supported_ratio:.2f} <= threshold={refusal_threshold}"
            if recommend_refusal
            else f"supported_ratio={supported_ratio:.2f} > threshold={refusal_threshold}"
        ),
    }


def filter_findings_for_verification(
    findings: List[Dict[str, Any]],
    skip_threshold: float = 0.85,
) -> Dict[str, Any]:
    """
    Heuristic pre-pass for investigation_verify_all.

    Splits findings into:
    - skip: already high-confidence (numeric_confidence >= skip_threshold)
    - verify: borderline or unverified findings that need full verify_finding call

    Args:
        findings: list of finding dicts with optional 'numeric_confidence' field
        skip_threshold: confidence above which to skip verification

    Returns:
        {
            "to_verify": [findings needing verify_finding],
            "skipped": [findings already high-confidence],
            "skip_count": int,
            "verify_count": int,
            "estimated_call_reduction": float,  # fraction of calls saved
        }
    """
    to_verify = []
    skipped = []

    for finding in findings:
        confidence = finding.get("numeric_confidence") or finding.get("confidence", 0.0)
        # Normalize string confidences
        if isinstance(confidence, str):
            conf_map = {"high": 0.9, "medium": 0.6, "low": 0.3, "unknown": 0.0}
            confidence = conf_map.get(confidence.lower(), 0.0)

        if isinstance(confidence, (int, float)) and confidence >= skip_threshold:
            skipped.append(finding)
        else:
            to_verify.append(finding)

    total = len(findings)
    skip_count = len(skipped)
    verify_count = len(to_verify)
    reduction = skip_count / total if total > 0 else 0.0

    return {
        "to_verify": to_verify,
        "skipped": skipped,
        "skip_count": skip_count,
        "verify_count": verify_count,
        "estimated_call_reduction": round(reduction, 3),
    }


def push_refusal_step_to_trajectory(
    policy_id: str,
    investigation_id: str,
    r_ref: float,
    trajectory_dir: Path,
) -> bool:
    """Push an r_ref refusal step to the DRPG TrajectoryStore."""
    try:
        from loci.drpg import TrajectoryStore

        store = TrajectoryStore(trajectory_dir)
        store.push(policy_id, {
            "step": 0,
            "action": "REFUSE" if r_ref > 0 else "ANSWER",
            "reward": {
                "r_ret": 0.0,
                "r_mem": 0.0,
                "r_ovp": 0.0,
                "r_bkp": 0.0,
                "r_ref": r_ref,
                "r_step": -0.01,
                "r_cor": 0.0,
            },
            "policy_id": policy_id,
            "investigation_id": investigation_id,
        })
        return True
    except Exception as e:
        logger.warning(f"Failed to push refusal step to trajectory: {e}")
        return False
