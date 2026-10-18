"""
Loci policy.md backend — policy storage, validation, and tier-aware tracking.

Provides:
1. PolicyTier: enum for light/medium/heavy tiers
2. PolicyScope: region definitions (memory, context, routing)
3. Policy: main policy document class
4. PolicyBackend: storage and retrieval
5. get_policy_config(): environment variable resolution
6. validate_policy(): constraint checking
"""

import os
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import json


class PolicyTier(str, Enum):
    """Policy tier constraints for retrieval and context limits."""
    LIGHT = "light"       # 2K context, 3 hops, 2 refinement steps
    MEDIUM = "medium"     # 8K context, 7 hops, 4 refinement steps
    HEAVY = "heavy"       # 64K context, 15 hops, 10 refinement steps

    def constraints(self) -> Dict[str, int]:
        """Return constraint values for this tier."""
        tiers = {
            PolicyTier.LIGHT: {
                "max_context_chars": 2048,
                "max_retrieval_hops": 3,
                "max_refinement_steps": 2,
            },
            PolicyTier.MEDIUM: {
                "max_context_chars": 8192,
                "max_retrieval_hops": 7,
                "max_refinement_steps": 4,
            },
            PolicyTier.HEAVY: {
                "max_context_chars": 65536,
                "max_retrieval_hops": 15,
                "max_refinement_steps": 10,
            },
        }
        return tiers.get(self, {})


TIER_MODEL_MAP: Dict[PolicyTier, List[str]] = {
    PolicyTier.LIGHT: [
        "claude-haiku-4.5",
        "gpt-5-mini",
        "mistral-7b-instruct",
        "llama-3-8b-instruct",
    ],
    PolicyTier.MEDIUM: [
        "claude-sonnet-4.6",
        "gpt-5.4-mini",
        "llama-3-70b-instruct",
        "mistral-large",
    ],
    PolicyTier.HEAVY: [
        "claude-opus-4.8",
        "gpt-5.4",
        "gpt-5.5",
        "claude-sonnet-4.6",  # fallback if opus unavailable
    ],
}


def get_tier_model_map() -> Dict[PolicyTier, List[str]]:
    """
    Get the tier→model map, respecting LOCI_TIER_MODEL_MAP env var override.

    LOCI_TIER_MODEL_MAP should be a JSON string:
    {"light": ["model-a", "model-b"], "medium": [...], "heavy": [...]}
    """
    override = os.getenv("LOCI_TIER_MODEL_MAP")
    if override:
        try:
            data = json.loads(override)
            return {
                PolicyTier(k): v
                for k, v in data.items()
                if k in ("light", "medium", "heavy")
            }
        except (json.JSONDecodeError, ValueError):
            pass  # fall through to default
    return TIER_MODEL_MAP


def resolve_model_for_tier(requested_model: str, tier: PolicyTier) -> Tuple[str, bool]:
    """
    Given a requested model and a tier constraint, return (resolved_model, was_rewritten).

    If requested_model is already in the tier's allowed list, returns it unchanged.
    If it's in an allowed or lower tier, keeps it (within cost budget).
    If it's in a higher tier or unknown, downgrades to the first model in the allowed tier.
    """
    tier_map = get_tier_model_map()
    allowed = tier_map.get(tier, [])
    tier_order = {PolicyTier.LIGHT: 0, PolicyTier.MEDIUM: 1, PolicyTier.HEAVY: 2}

    # Find which tier the requested model belongs to
    requested_tier = None
    for t, models in tier_map.items():
        if requested_model in models:
            requested_tier = t
            break

    # If model is in an allowed or lower tier, keep it
    if requested_tier is not None and tier_order[requested_tier] <= tier_order[tier]:
        return requested_model, False

    # Otherwise downgrade to first model in allowed tier
    if allowed:
        return allowed[0], True
    return requested_model, False


class PolicyRegion(str, Enum):
    """Policy region/scope definitions."""
    MEMORY = "memory"         # Affects grounding_hook retrieval strategy
    CONTEXT = "context"       # Affects system message injection
    ROUTING = "routing"       # Affects tier_router circuit-breaker decisions
    TOOL_CALL = "tool_call"   # Affects MCP tool invocation gating


@dataclass
class PolicyMetadata:
    """Policy metadata: creation, modification, provenance."""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    created_by: str = "loci-system"
    version: int = 1
    source: Optional[str] = None  # e.g., "REA_HERMES_ENHMENT", "DRPG-evolution"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Policy:
    """Policy.md object: defines retrieval strategy, tier constraints, and evidence backing."""
    
    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    type: str = "policy.md"
    tier: PolicyTier = PolicyTier.MEDIUM
    region: PolicyRegion = PolicyRegion.MEMORY
    labels: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: PolicyMetadata = field(default_factory=PolicyMetadata)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Policy":
        """Deserialize policy from dict."""
        tier_str = data.get("tier", "medium")
        region_str = data.get("region", "memory")
        
        try:
            tier = PolicyTier(tier_str) if isinstance(tier_str, str) else tier_str
        except ValueError:
            tier = PolicyTier.MEDIUM
        
        try:
            region = PolicyRegion(region_str) if isinstance(region_str, str) else region_str
        except ValueError:
            region = PolicyRegion.MEMORY
        
        metadata_dict = data.get("metadata", {})
        metadata = PolicyMetadata(**metadata_dict) if metadata_dict else PolicyMetadata()
        
        return cls(
            policy_id=data.get("policy_id", str(uuid.uuid4())),
            title=data.get("title", ""),
            type=data.get("type", "policy.md"),
            tier=tier,
            region=region,
            labels=data.get("labels", []),
            config=data.get("config", {}),
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize policy to dict."""
        return {
            "policy_id": self.policy_id,
            "title": self.title,
            "type": self.type,
            "tier": self.tier.value,
            "region": self.region.value,
            "labels": self.labels,
            "config": self.config,
            "metadata": self.metadata.to_dict(),
        }

    def get_constraints(self) -> Dict[str, int]:
        """Get tier-based constraints for this policy."""
        return self.tier.constraints()

    def get_reward_vector_weights(self) -> Dict[str, float]:
        """
        Get 7D reward vector weights based on policy phase.
        
        From EVO-RAG research:
        - r_ret: Retrieval_Bonus
        - r_rem: Retrieval_Action_Penalty
        - r_ovp: Subquery_Overlap_Penalty
        - r_bkp: Backtrack_Penalty
        - r_ref: Refusal_Reward
        - r_step: Step_Penalty
        - r_cor: Answer_Correctness
        """
        is_discovery = self.config.get("discovery_mode", True)
        
        if is_discovery:
            # Discovery phase: high retrieval incentive, low search penalty
            return {
                "r_ret": 1.0,      # High incentive to retrieve
                "r_rem": -0.05,    # Low search penalty
                "r_ovp": -0.1,     # Moderate overlap penalty
                "r_bkp": -0.2,     # Moderate backtrack penalty
                "r_ref": -0.5,     # Discourage refusal during discovery
                "r_step": -0.02,   # Low step penalty
                "r_cor": 0.5,      # Moderate correctness reward
            }
        else:
            # Refinement phase: efficiency focus, high refusal reward when evidence insufficient
            return {
                "r_ret": 0.5,      # Moderate retrieval reward
                "r_rem": -0.2,     # Higher search penalty
                "r_ovp": -0.15,    # Higher overlap penalty
                "r_bkp": -0.3,     # Higher backtrack penalty
                "r_ref": 1.0,      # High reward for refusal with insufficient evidence
                "r_step": -0.1,    # Higher step penalty
                "r_cor": 1.0,      # High correctness reward
            }


@dataclass
class PolicyConfig:
    """Environment-derived policy configuration."""
    enabled: bool = False
    max_chars: int = 8192
    allowed_models: Optional[List[str]] = None

    @classmethod
    def from_env(cls) -> "PolicyConfig":
        """Load policy config from environment variables."""
        enabled = os.getenv("LOCI_POLICY_ENABLE", "0").lower() in ("1", "true", "yes")
        max_chars = int(os.getenv("LOCI_POLICY_MAX_CHARS", "8192"))
        
        allowed_models_str = os.getenv("LOCI_POLICY_ALLOWED_MODELS", None)
        allowed_models = None
        if allowed_models_str:
            allowed_models = [m.strip() for m in allowed_models_str.split(",")]
        
        return cls(
            enabled=enabled,
            max_chars=max_chars,
            allowed_models=allowed_models,
        )


def get_policy_config() -> PolicyConfig:
    """Get policy configuration from environment."""
    return PolicyConfig.from_env()


def validate_policy(policy_dict: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Validate policy document against constraints.
    
    Returns list of errors (empty if valid).
    """
    errors = []
    
    # Check required fields
    if not policy_dict.get("title"):
        errors.append({"code": "missing_title", "message": "Policy must have a title"})
    
    # Check tier
    tier_str = policy_dict.get("tier", "")
    try:
        PolicyTier(tier_str)
    except ValueError:
        errors.append({
            "code": "invalid_tier",
            "message": f"Tier must be one of: light, medium, heavy. Got: {tier_str}"
        })
    
    # Check region
    region_str = policy_dict.get("region", "")
    try:
        PolicyRegion(region_str)
    except ValueError:
        errors.append({
            "code": "invalid_region",
            "message": f"Region must be one of: memory, context, routing, tool_call. Got: {region_str}"
        })
    
    # Check tier/config consistency
    if policy_dict.get("tier") and policy_dict.get("config"):
        try:
            tier = PolicyTier(policy_dict["tier"])
            constraints = tier.constraints()
            config = policy_dict["config"]
            
            max_chars = config.get("max_context_chars")
            if max_chars and max_chars > constraints.get("max_context_chars", max_chars):
                errors.append({
                    "code": "tier_config_mismatch",
                    "message": f"max_context_chars ({max_chars}) exceeds {tier.value} tier limit ({constraints['max_context_chars']})"
                })
            
            max_hops = config.get("max_retrieval_hops")
            if max_hops and max_hops > constraints.get("max_retrieval_hops", max_hops):
                errors.append({
                    "code": "tier_config_mismatch",
                    "message": f"max_retrieval_hops ({max_hops}) exceeds {tier.value} tier limit ({constraints['max_retrieval_hops']})"
                })
        except ValueError:
            pass  # Already reported in tier validation
    
    return errors


class PolicyBackend:
    """Backend for storing and retrieving policies."""
    
    def __init__(self, policy_dir: Optional[Path] = None):
        """Initialize backend with optional policy directory."""
        if policy_dir is None:
            policy_dir = Path.home() / ".loci" / "policies"
        
        self.policy_dir = Path(policy_dir)
        self.policy_dir.mkdir(parents=True, exist_ok=True)
    
    def store(self, policy: Policy) -> Dict[str, Any]:
        """
        Store a policy document.
        
        Returns: {"policy_id": "...", "status": "stored"}
        Raises: ValueError if policy validation fails
        """
        # Validate
        errors = validate_policy(policy.to_dict())
        if errors:
            raise ValueError(f"Policy validation failed: {errors}")
        
        # Save to file
        policy_file = self.policy_dir / f"{policy.policy_id}.json"
        with open(policy_file, "w") as f:
            json.dump(policy.to_dict(), f, indent=2)
        
        return {
            "policy_id": policy.policy_id,
            "status": "stored",
        }
    
    def load(self, policy_id: str) -> Optional[Policy]:
        """
        Load a policy by ID.
        
        Returns: Policy object or None if not found
        """
        policy_file = self.policy_dir / f"{policy_id}.json"
        
        if not policy_file.exists():
            return None
        
        with open(policy_file, "r") as f:
            data = json.load(f)
        
        return Policy.from_dict(data)
    
    def list_policies(self, labels: Optional[List[str]] = None, region: Optional[str] = None) -> List[Policy]:
        """
        List policies, optionally filtered by labels or region.
        
        Returns: List of Policy objects
        """
        policies = []
        
        for policy_file in self.policy_dir.glob("*.json"):
            with open(policy_file, "r") as f:
                data = json.load(f)
            
            policy = Policy.from_dict(data)
            
            # Apply filters
            if labels and not any(label in policy.labels for label in labels):
                continue
            if region and policy.region.value != region:
                continue
            
            policies.append(policy)
        
        return policies
