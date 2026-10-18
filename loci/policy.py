"""
Loci policy.md backend — policy storage, validation, and tier-aware tracking.

Provides:
1. PolicyTier: 7-tier enum (ZERO/NANO/LIGHT/STANDARD/TURBO/HEAVY/RESEARCH)
2. TIER_MODEL_MAP: model lists per tier
3. _TIER_ORDER: numeric ordering for tier comparisons
4. resolve_model_for_tier(): tier → model selection (with rewrite tracking)
5. StagePolicy: per-stage tier assignment for multi-stage tools
6. PolicyScope: region definitions (memory, context, routing)
7. Policy: main policy document class
8. PolicyBackend: storage and retrieval
9. get_policy_config(): environment variable resolution
10. validate_policy(): constraint checking
"""

import os
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
import json

from loci.mtime_cache import MtimeCache

# ---------------------------------------------------------------------------
# Module-level mtime-keyed policy cache
# ---------------------------------------------------------------------------

_POLICY_CACHE: "MtimeCache[Policy]" = MtimeCache(maxsize=256)


class PolicyTier(str, Enum):
    """7-tier policy ladder for retrieval and context limits."""
    ZERO     = "zero"      # No LLM — graph/cache/index only. r_step=0.
    NANO     = "nano"      # Local Ollama only. Zero API cost.
    LIGHT    = "light"     # haiku/flash. ~$0.25/1M tokens.
    STANDARD = "standard"  # sonnet/gpt-4o-mini. ~$0.60/1M.
    TURBO    = "turbo"     # sonnet+extended-think / o3-mini. ~$3/1M.
    HEAVY    = "heavy"     # opus/gpt-4o. ~$15/1M.
    RESEARCH = "research"  # Full swarm pipeline. ~$50+/run. session_limit=1.

    # Backward-compat alias — STANDARD replaces MEDIUM
    MEDIUM   = "medium"

    def constraints(self) -> Dict[str, int]:
        """Return constraint values for this tier."""
        tiers = {
            PolicyTier.ZERO:     {"max_context_chars": 0,      "max_retrieval_hops": 0,  "max_refinement_steps": 0,  "max_perspectives": 0, "max_fanout": 0,  "session_limit": -1},
            PolicyTier.NANO:     {"max_context_chars": 512,    "max_retrieval_hops": 1,  "max_refinement_steps": 1,  "max_perspectives": 1, "max_fanout": 4,  "session_limit": -1},
            PolicyTier.LIGHT:    {"max_context_chars": 2048,   "max_retrieval_hops": 3,  "max_refinement_steps": 2,  "max_perspectives": 1, "max_fanout": 8,  "session_limit": -1},
            PolicyTier.STANDARD: {"max_context_chars": 8192,   "max_retrieval_hops": 7,  "max_refinement_steps": 4,  "max_perspectives": 3, "max_fanout": 16, "session_limit": -1},
            PolicyTier.TURBO:    {"max_context_chars": 32768,  "max_retrieval_hops": 12, "max_refinement_steps": 6,  "max_perspectives": 4, "max_fanout": 24, "session_limit": -1},
            PolicyTier.HEAVY:    {"max_context_chars": 65536,  "max_retrieval_hops": 15, "max_refinement_steps": 10, "max_perspectives": 5, "max_fanout": 32, "session_limit": -1},
            PolicyTier.RESEARCH: {"max_context_chars": 524288, "max_retrieval_hops": 30, "max_refinement_steps": 20, "max_perspectives": 5, "max_fanout": 64, "session_limit": 1},
        }
        # MEDIUM is an alias for STANDARD
        result = tiers.get(self)
        if result is None and self == PolicyTier.MEDIUM:
            return tiers[PolicyTier.STANDARD]
        return result or {}


TIER_MODEL_MAP: Dict[PolicyTier, List[str]] = {
    PolicyTier.ZERO:     [],  # no model calls
    PolicyTier.NANO:     ["ollama/llama3.2:3b", "ollama/mistral:7b", "ollama/phi3:mini"],
    PolicyTier.LIGHT:    ["claude-haiku-4.5", "gpt-5-mini", "mistral-7b-instruct"],
    PolicyTier.STANDARD: ["claude-sonnet-4.6", "gpt-5.4-mini", "llama-3-70b-instruct"],
    PolicyTier.MEDIUM:   ["claude-sonnet-4.6", "gpt-5.4-mini", "llama-3-70b-instruct"],  # alias
    PolicyTier.TURBO:    ["claude-sonnet-4.6-extended", "o3-mini", "gpt-5.4-mini-thinking"],
    PolicyTier.HEAVY:    ["claude-opus-4.8", "gpt-5.4", "gpt-5.5"],
    PolicyTier.RESEARCH: ["claude-opus-4.8", "gpt-5.5"],  # full swarm uses these
}

# Numeric ordering for tier comparisons
_TIER_ORDER: Dict[PolicyTier, int] = {
    PolicyTier.ZERO: 0,
    PolicyTier.NANO: 1,
    PolicyTier.LIGHT: 2,
    PolicyTier.STANDARD: 3,
    PolicyTier.MEDIUM: 3,   # same level as STANDARD
    PolicyTier.TURBO: 4,
    PolicyTier.HEAVY: 5,
    PolicyTier.RESEARCH: 6,
}


def get_tier_model_map() -> Dict[PolicyTier, List[str]]:
    """
    Get the tier→model map, delegating to model_selector for Ollama support.

    Falls back to LOCI_TIER_MODEL_MAP env var override (JSON string mapping tier
    names to model lists), then to the compiled TIER_MODEL_MAP default.
    """
    try:
        from loci.model_selector import get_tier_model_map as _get
        raw = _get()
        result = {}
        for k, v in raw.items():
            try:
                result[PolicyTier(k)] = v
            except ValueError:
                pass
        if result:
            return result
    except ImportError:
        pass
    override = os.getenv("LOCI_TIER_MODEL_MAP")
    if override:
        try:
            data = json.loads(override)
            result = {}
            for k, v in data.items():
                try:
                    result[PolicyTier(k)] = v
                except ValueError:
                    pass  # unknown tier key — ignore silently
            if result:
                return result
        except (json.JSONDecodeError, ValueError):
            pass  # fall through to default
    return TIER_MODEL_MAP


def resolve_model_for_tier(
    tier: PolicyTier,
    available: Optional[List[str]] = None,
) -> str:
    """
    Return the best model for *tier* from the tier model map.

    If *available* is provided, the first model in the tier's list that also
    appears in *available* is preferred; when none matches, the first model in
    the tier map is returned as a fallback.

    Raises ValueError for PolicyTier.ZERO (that tier forbids all LLM calls).
    """
    if tier is PolicyTier.ZERO:
        raise ValueError(
            f"PolicyTier.ZERO forbids LLM calls — no model can be selected. "
            f"Use a higher tier or handle the ZERO case before calling resolve_model_for_tier."
        )
    tier_map = get_tier_model_map()
    candidates = tier_map.get(tier, [])
    if available and candidates:
        for model in candidates:
            if model in available:
                return model
    if candidates:
        return candidates[0]
    return ""


def downgrade_model_to_tier(requested_model: str, tier: PolicyTier) -> Tuple[str, bool]:
    """
    Given a requested model and a tier constraint, return (resolved_model, was_rewritten).

    This is the cost-enforcement path: if *requested_model* already belongs to
    *tier* or a cheaper tier it is returned unchanged; otherwise the first
    model in the tier's allowed list is substituted.
    Delegates to model_selector.resolve_for_loci when available for Ollama-aware routing.
    """
    try:
        from loci.model_selector import resolve_for_loci
        resolved, downgraded = resolve_for_loci(requested_model, tier.value)
        return resolved, downgraded
    except ImportError:
        pass

    tier_map = get_tier_model_map()
    allowed = tier_map.get(tier, [])

    requested_tier = None
    for t, models in tier_map.items():
        if requested_model in models:
            requested_tier = t
            break

    if requested_tier is not None and _TIER_ORDER.get(requested_tier, 3) <= _TIER_ORDER.get(tier, 3):
        return requested_model, False

    if allowed:
        return allowed[0], True
    return requested_model, False


# ---------------------------------------------------------------------------
# StagePolicy — per-stage tier assignment
# ---------------------------------------------------------------------------

@dataclass
class StagePolicy:
    """Per-stage tier assignment for multi-stage tools like swarm_reason."""
    stages: Dict[str, PolicyTier] = field(default_factory=dict)
    outer_tier: PolicyTier = PolicyTier.STANDARD

    def tier_for(self, stage: str) -> PolicyTier:
        """Get tier for a stage, clamped to outer_tier ceiling."""
        stage_tier = self.stages.get(stage, self.outer_tier)
        if _TIER_ORDER.get(stage_tier, 3) > _TIER_ORDER.get(self.outer_tier, 3):
            return self.outer_tier
        return stage_tier

    @classmethod
    def for_swarm_reason(cls, outer: PolicyTier = PolicyTier.STANDARD) -> "StagePolicy":
        """Default stage policy for swarm_reason."""
        return cls(
            stages={
                "decompose":        PolicyTier.NANO,
                "fanout":           PolicyTier.LIGHT,
                "self_consistency": PolicyTier.LIGHT,
                "escalation":       PolicyTier.STANDARD,
                "synthesis":        PolicyTier.HEAVY,
            },
            outer_tier=outer,
        )

    @classmethod
    def for_investigation(cls, outer: PolicyTier = PolicyTier.STANDARD) -> "StagePolicy":
        """Default stage policy for the investigation pipeline."""
        return cls(
            stages={
                "pre_answer_check": PolicyTier.ZERO,
                "reason":           PolicyTier.STANDARD,
                "reflect":          PolicyTier.NANO,
                "verify":           PolicyTier.NANO,
                "adversarial":      PolicyTier.STANDARD,
            },
            outer_tier=outer,
        )


class PolicyRegion(str, Enum):
    """Policy region/scope definitions."""
    MEMORY    = "memory"     # Affects grounding_hook retrieval strategy
    CONTEXT   = "context"    # Affects system message injection
    ROUTING   = "routing"    # Affects tier_router circuit-breaker decisions
    TOOL_CALL = "tool_call"  # Affects MCP tool invocation gating


# Sets used by validate_policy() error messages
_VALID_TIERS: frozenset = frozenset(t.value for t in PolicyTier)
_VALID_REGIONS: frozenset = frozenset(r.value for r in PolicyRegion)


@dataclass(slots=True)
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
    tier: PolicyTier = PolicyTier.STANDARD
    region: PolicyRegion = PolicyRegion.MEMORY
    labels: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: PolicyMetadata = field(default_factory=PolicyMetadata)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Policy":
        """Deserialize policy from dict."""
        tier_str = data.get("tier", "standard")
        region_str = data.get("region", "memory")
        
        try:
            tier = PolicyTier(tier_str) if isinstance(tier_str, str) else tier_str
        except ValueError:
            tier = PolicyTier.STANDARD
        
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
        Get 7D reward vector weights based on policy tier and phase.
        
        From EVO-RAG research:
        - r_ret: Retrieval_Bonus
        - r_rem: Retrieval_Action_Penalty
        - r_ovp: Subquery_Overlap_Penalty
        - r_bkp: Backtrack_Penalty
        - r_ref: Refusal_Reward
        - r_step: Step_Penalty
        - r_cor: Answer_Correctness
        """
        tier = self.tier
        is_discovery = self.config.get("discovery_mode", True)

        # ZERO: no model call — all zeros
        if tier == PolicyTier.ZERO:
            return {k: 0.0 for k in ("r_ret", "r_rem", "r_ovp", "r_bkp", "r_ref", "r_step", "r_cor")}

        # NANO: lightweight discovery weights
        if tier == PolicyTier.NANO:
            return {
                "r_ret":  0.8,
                "r_rem":  -0.03,
                "r_ovp":  -0.05,
                "r_bkp":  -0.1,
                "r_ref":  -0.3,
                "r_step": -0.01,
                "r_cor":  0.4,
            }

        # TURBO: refinement-leaning weights (between STANDARD and HEAVY)
        if tier == PolicyTier.TURBO:
            return {
                "r_ret":  0.6,
                "r_rem":  -0.15,
                "r_ovp":  -0.12,
                "r_bkp":  -0.25,
                "r_ref":  0.7,
                "r_step": -0.06,
                "r_cor":  0.9,
            }

        # RESEARCH: full discovery with high backtrack penalty (expensive)
        if tier == PolicyTier.RESEARCH:
            return {
                "r_ret":  1.0,
                "r_rem":  -0.05,
                "r_ovp":  -0.1,
                "r_bkp":  -0.6,   # high: backtracking is expensive at this tier
                "r_ref":  -0.5,
                "r_step": -0.02,
                "r_cor":  0.8,
            }

        # LIGHT / STANDARD / MEDIUM / HEAVY — phase-driven weights
        if is_discovery:
            return {
                "r_ret":  1.0,
                "r_rem":  -0.05,
                "r_ovp":  -0.1,
                "r_bkp":  -0.2,
                "r_ref":  -0.5,
                "r_step": -0.02,
                "r_cor":  0.5,
            }
        else:
            return {
                "r_ret":  0.5,
                "r_rem":  -0.2,
                "r_ovp":  -0.15,
                "r_bkp":  -0.3,
                "r_ref":  1.0,
                "r_step": -0.1,
                "r_cor":  1.0,
            }


@dataclass
class PolicyConfig:
    """Environment-derived policy configuration."""
    enabled: bool = False
    max_chars: int = 8192
    allowed_models: Optional[List[str]] = None
    cache_confidence_threshold: float = 0.85
    cache_ttl_hours: float = 24.0

    @classmethod
    def from_env(cls) -> "PolicyConfig":
        """Load policy config from environment variables."""
        enabled = os.getenv("LOCI_POLICY_ENABLE", "0").lower() in ("1", "true", "yes")
        max_chars = int(os.getenv("LOCI_POLICY_MAX_CHARS", "8192"))
        
        allowed_models_str = os.getenv("LOCI_POLICY_ALLOWED_MODELS", None)
        allowed_models = None
        if allowed_models_str:
            allowed_models = [m.strip() for m in allowed_models_str.split(",")]
        
        cache_confidence_threshold = float(os.getenv("LOCI_CACHE_CONFIDENCE", "0.85"))
        cache_ttl_hours = float(os.getenv("LOCI_CACHE_TTL_HOURS", "24.0"))
        
        return cls(
            enabled=enabled,
            max_chars=max_chars,
            allowed_models=allowed_models,
            cache_confidence_threshold=cache_confidence_threshold,
            cache_ttl_hours=cache_ttl_hours,
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
            "message": f"Tier must be one of: {', '.join(sorted(_VALID_TIERS))}. Got: {tier_str}"
        })
    
    # Check region
    region_str = policy_dict.get("region", "")
    try:
        PolicyRegion(region_str)
    except ValueError:
        errors.append({
            "code": "invalid_region",
            "message": f"Region must be one of: {', '.join(sorted(_VALID_REGIONS))}. Got: {region_str}"
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
        with open(policy_file, "w", encoding="utf-8") as f:
            json.dump(policy.to_dict(), f, indent=2)
        
        # Invalidate cache so next load picks up the freshly written file
        _POLICY_CACHE.invalidate(policy.policy_id)
        
        return {
            "policy_id": policy.policy_id,
            "status": "stored",
        }
    
    def load(self, policy_id: str) -> Optional[Policy]:
        """
        Load a policy by ID.
        
        Returns: Policy object or None if not found.
        Uses mtime-keyed cache to avoid redundant file reads.
        """
        policy_file = self.policy_dir / f"{policy_id}.json"
        if not policy_file.exists():
            return None
        try:
            mtime = policy_file.stat().st_mtime
        except OSError:
            return None
        cached = _POLICY_CACHE.get(policy_id, mtime)
        if cached is not None:
            return cached
        with open(policy_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        policy = Policy.from_dict(data)
        _POLICY_CACHE.set(policy_id, policy, mtime)
        return policy
    
    def list_policies(self, labels: Optional[List[str]] = None, region: Optional[str] = None) -> List[Policy]:
        """
        List policies, optionally filtered by labels or region.
        
        Returns: List of Policy objects. Skips re-reading files whose mtime matches the cache.
        """
        policies = []
        
        for policy_file in self.policy_dir.glob("*.json"):
            policy_id = policy_file.stem
            try:
                mtime = policy_file.stat().st_mtime
            except OSError:
                continue
            cached = _POLICY_CACHE.get(policy_id, mtime)
            if cached is not None:
                policy = cached
            else:
                with open(policy_file, "r", encoding="utf-8") as f:
                    policy = Policy.from_dict(json.load(f))
                _POLICY_CACHE.set(policy_id, policy, mtime)
            
            # Apply filters
            if labels and not any(label in policy.labels for label in labels):
                continue
            if region and policy.region.value != region:
                continue
            
            policies.append(policy)
        
        return policies
