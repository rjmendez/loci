"""
LiteLLM hooks for Loci policy integration.

Provides:
1. loci_policy_hook: pre-call hook deciding whether to apply policy
2. grounding_hook_augmented: enhanced grounding with policy backing
3. tier_router_surface: surfaces policy violations in routing decisions
4. eval_hook_stub: captures step-level feedback for DRPG evolution
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from policy import Policy, PolicyBackend, PolicyRegion, get_policy_config


logger = logging.getLogger(__name__)


class PolicyDecision:
    """Result of policy decision: whether to apply and why."""
    
    def __init__(self, apply: bool, policy_id: Optional[str] = None, reason: str = ""):
        self.apply = apply
        self.policy_id = policy_id
        self.reason = reason
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "apply": self.apply,
            "policy_id": self.policy_id,
            "reason": self.reason,
        }


class PolicyViolation:
    """Represents a policy constraint violation."""
    
    def __init__(self, violation_type: str, message: str, severity: str = "warning"):
        self.violation_type = violation_type
        self.message = message
        self.severity = severity  # "warning" or "error"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.violation_type,
            "message": self.message,
            "severity": self.severity,
        }


async def loci_policy_hook(
    model: str,
    messages: List[Dict[str, str]],
    kwargs: Dict[str, Any],
    litellm_params: Dict[str, Any],
    logger_arg: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Pre-call hook: decide whether to apply policy based on model and config.
    
    Returns:
    {
        "apply": bool,
        "policy_id": str or None,
        "reason": str,
        "violations": [{"type": str, "message": str, "severity": str}] or []
    }
    """
    violations: List[PolicyViolation] = []
    
    # Get policy config from environment
    policy_config = get_policy_config()
    
    if not policy_config.enabled:
        return {
            "apply": False,
            "policy_id": None,
            "reason": "policy_disabled_via_env",
            "violations": [],
        }
    
    # Check model restrictions
    if policy_config.allowed_models and model not in policy_config.allowed_models:
        violations.append(PolicyViolation(
            "model_not_allowed",
            f"Model '{model}' not in allowed list: {policy_config.allowed_models}",
            severity="warning",
        ))
        return {
            "apply": False,
            "policy_id": None,
            "reason": "model_not_allowed",
            "violations": [v.to_dict() for v in violations],
        }

    # Check tool_call gating
    tool_call_name = kwargs.get("tool_call_name")
    if tool_call_name and policy_config.enabled:
        allowed_tools: Optional[List[str]] = None
        allowed_tools_env = os.environ.get("LOCI_POLICY_ALLOWED_TOOLS")
        if allowed_tools_env:
            allowed_tools = [t.strip() for t in allowed_tools_env.split(",")]
        elif policy_config.allowed_models:
            # Reuse allowed_models list as fallback tool allowlist when no TOOLS env set
            allowed_tools = policy_config.allowed_models

        if allowed_tools is not None and tool_call_name not in allowed_tools:
            violations.append(PolicyViolation(
                "tool_not_allowed",
                f"Tool '{tool_call_name}' not in allowed tools list",
                severity="error",
            ))
            return {
                "apply": False,
                "policy_id": None,
                "reason": "tool_not_allowed",
                "violations": [v.to_dict() for v in violations],
            }

    # Try to load a policy for this model/tier
    # For now, we'll just indicate policy is ready to apply
    # In production, this would select a specific policy from the backend
    policy_id = kwargs.get("policy_id", None)
    
    return {
        "apply": True,
        "policy_id": policy_id,
        "reason": "policy_enabled",
        "violations": [v.to_dict() for v in violations],
    }


async def grounding_hook_augmented(
    model: str,
    messages: List[Dict[str, str]],
    kwargs: Dict[str, Any],
    litellm_params: Dict[str, Any],
    logger_arg: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Augmented grounding hook: inject policy-backed evidence into system message.
    
    Adds MEMORY MATCH section with:
    - Policy tier and discovery phase status
    - 7D reward vector weights from policy
    - Matching findings (from Qdrant/grounding)
    
    Returns modified messages + metadata.
    """
    # Get policy if one is specified
    policy_id = kwargs.get("policy_id")
    policy: Optional[Policy] = None
    
    if policy_id:
        backend = PolicyBackend()
        policy = backend.load(policy_id)
    
    # Build augmented system message
    augmented_system = _build_augmented_system_message(policy, kwargs.get("grounding_context", {}))
    
    # Inject or update system message
    updated_messages = list(messages)
    if policy and augmented_system:
        if updated_messages and updated_messages[0].get("role") == "system":
            updated_messages[0]["content"] += "\n\n" + augmented_system
        else:
            updated_messages.insert(0, {"role": "system", "content": augmented_system})
    
    return {
        "messages": updated_messages,
        "policy_id": policy_id,
        "augmented": bool(augmented_system),
    }


def _build_augmented_system_message(policy: Optional[Policy], context: Dict[str, Any]) -> str:
    """Build augmented system message with policy backing."""
    if not policy:
        return ""
    
    lines = ["POLICY-AWARE MEMORY MATCH:"]
    
    # Add policy tier and phase
    lines.append(f"- Tier: {policy.tier.value}")
    lines.append(f"- Phase: {'discovery' if policy.config.get('discovery_mode') else 'refinement'}")
    
    # Add reward vector weights
    weights = policy.get_reward_vector_weights()
    lines.append(f"- Reward Vector: {json.dumps(weights, indent=0)}")
    
    # Add constraints
    constraints = policy.get_constraints()
    lines.append(f"- Max Context: {constraints.get('max_context_chars')} chars")
    lines.append(f"- Max Hops: {constraints.get('max_retrieval_hops')}")
    
    # Add matching findings if present
    findings = context.get("findings", [])
    if findings:
        lines.append(f"- Findings: {len(findings)} matches")
        for finding in findings[:3]:  # Limit to top 3
            score = finding.get("score", 0)
            text = finding.get("text", "")[:100]
            lines.append(f"  - (score={score:.2f}) {text}")
    
    return "\n".join(lines)


async def tier_router_surface(
    model: str,
    messages: List[Dict[str, str]],
    kwargs: Dict[str, Any],
    litellm_params: Dict[str, Any],
    logger_arg: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Rewrites the model field to enforce policy tier cost constraints.

    If a policy is in effect, resolves the requested model against the policy
    tier's allowed list and downgrades to the first allowed model when the
    requested model exceeds the tier budget.

    Returns:
    {
        "tier_decision": str,
        "downgraded": bool,
        "model": str,
        "original_model": str or None,
        "violations": [...]
    }
    """
    from policy import resolve_model_for_tier  # avoid circular import

    violations: List[PolicyViolation] = []
    policy_id = kwargs.get("policy_id")

    if not policy_id:
        return {
            "tier_decision": "default",
            "downgraded": False,
            "model": model,
            "original_model": None,
            "violations": [],
        }

    backend = PolicyBackend()
    policy = backend.load(policy_id)

    if not policy:
        return {
            "tier_decision": "default",
            "downgraded": False,
            "model": model,
            "original_model": None,
            "violations": [{"type": "policy_not_found", "message": f"Policy {policy_id} not found"}],
        }

    resolved_model, was_rewritten = resolve_model_for_tier(model, policy.tier)

    if was_rewritten:
        violations.append(PolicyViolation(
            "model_downgraded",
            f"Model '{model}' downgraded to '{resolved_model}' per {policy.tier.value} tier policy",
            severity="warning",
        ))

    return {
        "tier_decision": policy.tier.value,
        "downgraded": was_rewritten,
        "model": resolved_model,
        "original_model": model if was_rewritten else None,
        "violations": [v.to_dict() for v in violations],
    }


class StepFeedback:
    """Single step's feedback for DRPG evolution."""
    
    def __init__(
        self,
        step: int,
        action: str,
        reward_vector: Dict[str, float],
        policy_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ):
        self.step = step
        self.action = action  # SEARCH, BACKTRACK, ANSWER, REFUSE
        self.reward_vector = reward_vector
        self.policy_id = policy_id
        self.timestamp = timestamp or datetime.utcnow().isoformat() + "Z"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "reward": self.reward_vector,
            "policy_id": self.policy_id,
            "timestamp": self.timestamp,
        }


async def eval_hook_stub(
    model: str,
    messages: List[Dict[str, str]],
    kwargs: Dict[str, Any],
    litellm_params: Dict[str, Any],
    logger_arg: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Hook: captures step-level feedback for DRPG policy evolution.

    Persists reward vector to JSONL trajectory file under LOCI_TRAJECTORY_DIR
    (defaults to ~/.loci/trajectories) and also logs via logger.

    Expected kwargs:
    - policy_id: str
    - step: int
    - action: str (SEARCH, BACKTRACK, ANSWER, REFUSE)
    - reward_vector: Dict[str, float] (r_ret, r_rem, r_ovp, r_bkp, r_ref, r_step, r_cor)

    Returns:
    {
        "logged": bool,
        "persisted": bool,
        "step_id": str,
        "feedback": {...}
    }
    """
    from drpg_evolution import TrajectoryStore

    step = kwargs.get("step", 0)
    action = kwargs.get("action", "unknown")
    reward_vector = kwargs.get("reward_vector", {})
    policy_id = kwargs.get("policy_id")

    feedback = StepFeedback(
        step=step,
        action=action,
        reward_vector=reward_vector,
        policy_id=policy_id,
    )

    _log = logger_arg or logger
    _log.info(f"DRPG Feedback: {json.dumps(feedback.to_dict())}")

    persisted = False
    if policy_id:
        try:
            traj_dir = os.environ.get(
                "LOCI_TRAJECTORY_DIR", str(Path.home() / ".loci" / "trajectories")
            )
            store = TrajectoryStore(Path(traj_dir))
            store.push(policy_id, feedback.to_dict())
            persisted = True
        except Exception as exc:
            _log.warning("eval_hook_stub: trajectory persist failed (fail-open): %r", exc)

    return {
        "logged": True,
        "persisted": persisted,
        "step_id": f"{policy_id}:step:{step}",
        "feedback": feedback.to_dict(),
    }
