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

import math
import os
import random
import re
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Callable, Mapping, Optional, Dict, List, Any
import json
import tomllib

from loci.mtime_cache import MtimeCache

# ---------------------------------------------------------------------------
# Module-level mtime-keyed policy cache
# ---------------------------------------------------------------------------

_POLICY_CACHE: "MtimeCache[Policy]" = MtimeCache(maxsize=256)

# A policy id names a file under the store, so it is restricted to a safe filename:
# lowercase alphanumeric first, then alphanumerics, '_', '.', '-', at most 64 characters.
# Default ids (uuid4) match.  Nothing with a separator, drive colon or empty value does.
POLICY_ID_PATTERN = r"[a-z0-9][a-z0-9_.-]{0,63}"
_POLICY_ID_RE = re.compile(POLICY_ID_PATTERN)


def loci_home() -> Path:
    """Root of the Loci state directory: $LOCI_HOME when set, else ~/.loci.

    Resolved at call time (never at import) so a test that sets LOCI_HOME, or HOME,
    after this module is imported still redirects every default store path.
    """
    override = os.environ.get("LOCI_HOME")
    if override:
        return Path(override)
    return Path.home() / ".loci"


def validate_policy_id(policy_id: Any) -> str:
    """Return *policy_id* when it is a safe store key; raise ValueError('invalid policy_id: ...') otherwise."""
    if not isinstance(policy_id, str) or _POLICY_ID_RE.fullmatch(policy_id) is None:
        raise ValueError(f"invalid policy_id: {policy_id!r} (must match {POLICY_ID_PATTERN})")
    return policy_id


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
    Get the tier→model map, respecting LOCI_TIER_MODEL_MAP env var override.

    LOCI_TIER_MODEL_MAP should be a JSON string mapping tier names to model lists.
    Only keys that map to a valid PolicyTier are included; unknown keys are silently
    ignored.  When the env var is set, only the tiers present in it are returned
    (no merge with defaults).  Falls back to TIER_MODEL_MAP on parse error.

    This is the policy map only; the model pool's local pick is added by
    ``loci.model_selector.candidates`` (see ``resolve_model_for_tier``).
    """
    return _parse_tier_map(os.getenv("LOCI_TIER_MODEL_MAP")) or TIER_MODEL_MAP


#: A tier map for callers that talk to the LiteLLM proxy (litellm_call): its values are the proxy's own model group
#: names. A separate variable, because LOCI_TIER_MODEL_MAP also feeds every other tier lookup (pool candidates,
#: swarm stages) and changing it to group names would change those too.
PROXY_TIER_MAP_ENV = "LOCI_LITELLM_TIER_MAP"


def _parse_tier_map(raw: Optional[str]) -> Optional[Dict[PolicyTier, List[str]]]:
    """A JSON tier→models map as a dict keyed by PolicyTier, or None when *raw* is unset, unparseable or names no
    valid tier. Unknown tier keys are ignored; the tiers present are the only ones returned (no merge)."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
        result = {}
        for k, v in data.items():
            try:
                result[PolicyTier(k)] = v
            except ValueError:
                pass  # unknown tier key — ignore silently
        return result or None
    except (json.JSONDecodeError, ValueError, AttributeError):
        return None


def get_proxy_tier_model_map() -> Dict[PolicyTier, List[str]]:
    """The tier→model map for calls through the LiteLLM proxy: LOCI_LITELLM_TIER_MAP when it parses, else the
    ordinary map (``get_tier_model_map``)."""
    return _parse_tier_map(os.getenv(PROXY_TIER_MAP_ENV)) or get_tier_model_map()


def _tier_candidates(tier: PolicyTier) -> List[str]:
    """The model pool's local pick for *tier* (when a pool serves it) followed by the policy map's entries."""
    try:
        from loci.model_selector import candidates
    except ImportError:
        return list(get_tier_model_map().get(tier, []))
    return candidates(tier.value)


def _candidates(tier: PolicyTier, pool: bool) -> List[str]:
    """Candidates for *tier*: with *pool* the pool's local pick leads (see ``_tier_candidates``); without it
    only the policy map's entries, which a caller that must name proxy aliases sets through LOCI_TIER_MODEL_MAP."""
    return _tier_candidates(tier) if pool else list(get_proxy_tier_model_map().get(tier, []))


def resolve_model_for_tier(
    tier: PolicyTier,
    available: Optional[List[str]] = None,
) -> str:
    """
    Return the best model for *tier*: the model pool's local pick when one serves the
    tier, then the tier model map's entries.

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
    candidates = _tier_candidates(tier)
    if available and candidates:
        for model in candidates:
            if model in available:
                return model
    if candidates:
        return candidates[0]
    return ""


class DowngradeResult(tuple):
    """``(model, was_rewritten)`` -- it unpacks and compares as that pair -- plus ``tier_unresolved``.

    ``tier_unresolved`` is True when the tier named no usable model and the cost ceiling was
    enforced anyway (see ``downgrade_model_to_tier``): *model* is then the cheapest model allowed
    at or below the tier, or "" when there is none.  A caller must not call "".
    """

    def __new__(cls, model: str, was_rewritten: bool, tier_unresolved: bool = False) -> "DowngradeResult":
        self = super().__new__(cls, (model, was_rewritten))
        self.tier_unresolved = tier_unresolved
        return self

    @property
    def model(self) -> str:
        return self[0]

    @property
    def was_rewritten(self) -> bool:
        return self[1]

    @property
    def usable_model(self) -> str:
        """The model a caller may actually call: "" when the tier forbids every model.

        That is the ZERO case: tier_unresolved with the request left alone (was_rewritten False).
        Every other result, including an unresolved one that fell back to the cheapest allowed
        model, is usable as returned.
        """
        return "" if self.tier_unresolved and not self.was_rewritten else self.model


def _cheapest_allowed(tier: Any, pool: bool = True) -> str:
    """The first candidate of the cheapest tier that has one, among tiers at or below *tier*; "" if none.

    An unknown *tier* (not a PolicyTier) has no known ceiling, so the whole ladder is searched
    and the cheapest candidate anywhere wins.
    """
    ceiling = _TIER_ORDER[tier] if isinstance(tier, PolicyTier) else max(_TIER_ORDER.values())
    for candidate_tier in sorted(_TIER_ORDER, key=lambda t: _TIER_ORDER[t]):
        if _TIER_ORDER[candidate_tier] > ceiling:
            break
        models = _candidates(candidate_tier, pool)
        if models:
            return models[0]
    return ""


def downgrade_model_to_tier(requested_model: str, tier: PolicyTier, *, pool: bool = True) -> DowngradeResult:
    """
    Given a requested model and a tier constraint, return (resolved_model, was_rewritten).

    This is the cost-enforcement path: if *requested_model* already belongs to
    *tier* or a cheaper tier it is returned unchanged; otherwise the first
    candidate for the tier (see ``resolve_model_for_tier``) is substituted.
    A model the pool serves for a tier counts as belonging to that tier. A model no
    tier lists is substituted, as it always was.

    Fail closed: when the tier has no candidate at all (a tier the model map or its
    LOCI_TIER_MODEL_MAP override does not list, or a tier that is not a PolicyTier) the requested
    model is NOT passed through.  The result is the cheapest allowed model at or below the tier
    (or "" if there is none), ``was_rewritten`` is True and ``tier_unresolved`` is True.

    The ZERO tier is the one exception to the pair: it still returns ``(requested_model, False)``,
    because mcp/tests/test_model_selector.py pins that, but with ``tier_unresolved`` True and
    ``usable_model`` "" -- callers must use ``usable_model`` (tier_router_surface and litellm_call do).

    ``pool=False`` leaves the model pool out: candidates are only the tier map's entries and a request is
    judged only against the map, never against the pool's local pick. litellm_call uses it so that a rewrite
    names a model the LiteLLM proxy serves (LOCI_LITELLM_TIER_MAP lists the proxy's group names, falling back to
    LOCI_TIER_MODEL_MAP), not an Ollama model the pool prefers.
    """
    if not isinstance(tier, PolicyTier):
        try:
            tier = PolicyTier(tier)
        except ValueError:
            return DowngradeResult(_cheapest_allowed(tier, pool), True, True)
    tier_map = get_tier_model_map() if pool else get_proxy_tier_model_map()
    allowed = _candidates(tier, pool)

    requested_tier = None
    for t, models in tier_map.items():
        if requested_model in models:
            requested_tier = t
            break
    if requested_tier is None and pool:
        try:
            from loci.model_selector import local_model
        except ImportError:
            local_model = None
        if local_model is not None:
            for t in tier_map:
                if requested_model == local_model(t.value):
                    requested_tier = t
                    break

    if requested_tier is not None and _TIER_ORDER.get(requested_tier, 3) <= _TIER_ORDER.get(tier, 3):
        return DowngradeResult(requested_model, False)

    if allowed:
        return DowngradeResult(allowed[0], True)
    if tier is PolicyTier.ZERO:
        # ZERO forbids every model.  The legacy pair (requested, False) is kept because an existing
        # test (mcp/tests/test_model_selector.py) pins it; tier_unresolved=True and usable_model == ""
        # tell callers that the request must not be called.
        return DowngradeResult(requested_model, False, True)
    return DowngradeResult(_cheapest_allowed(tier, pool), True, True)


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

    def get_learned_genes(self) -> Dict[str, float]:
        """The valid learned gene scales stored in ``config['learned_genes']`` (see ``LEARNED_GENES_KEY``)."""
        return read_learned_genes(self.config)[0]

    def get_effective_constraints(
        self,
        *,
        environ: Optional[Mapping[str, str]] = None,
        registry_path: Optional[Path] = None,
    ) -> Dict[str, int]:
        """``get_constraints()`` with the learned gene scales applied, ONLY when the learned-genes feature is ON.

        The feature is OFF unless ``LOCI_LEARNED_GENES=1`` AND ``learned_genes`` is not registered default-off
        in ``scripts/policy_flags.toml`` (see ``learned_feature_on``).  Off, this returns exactly
        ``get_constraints()``.  On, a limit can only get tighter: never above the tier value, never below 1
        for a limit the tier allows.  Nothing calls this yet.
        """
        base = self.get_constraints()
        if not learned_feature_on(LEARNED_GENES_FLAG, environ=environ, registry_path=registry_path):
            return base
        return apply_genes(base, self.get_learned_genes())

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
            policy_dir = loci_home() / "policies"

        self.policy_dir = Path(policy_dir)
        self.policy_dir.mkdir(parents=True, exist_ok=True)

    def _policy_file(self, policy_id: str) -> Path:
        """The file for *policy_id*; ValueError('invalid policy_id') unless it is a safe name inside policy_dir."""
        validate_policy_id(policy_id)
        base = self.policy_dir.resolve()
        path = (base / f"{policy_id}.json").resolve()
        if path.parent != base:
            raise ValueError(f"invalid policy_id: {policy_id!r} (resolves outside the policy directory)")
        return path
    
    def store(self, policy: Policy, *, writer: Optional[Callable[[Path, bytes], None]] = None) -> Dict[str, Any]:
        """
        Store a policy document.

        ``writer(path, data)`` replaces the plain file write when given (the bounded evolver passes its
        sandboxed temp+rename writer); the validation, the path check and the cache invalidation are the same.

        Returns: {"policy_id": "...", "status": "stored"}
        Raises: ValueError if policy validation fails
        """
        # Validate
        errors = validate_policy(policy.to_dict())
        if errors:
            raise ValueError(f"Policy validation failed: {errors}")
        
        # Save to file
        policy_file = self._policy_file(policy.policy_id)
        if writer is not None:
            writer(policy_file, json.dumps(policy.to_dict(), indent=2).encode("utf-8"))
        else:
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
        Raises: ValueError('invalid policy_id') for an id that is not a safe store key.
        Uses mtime-keyed cache to avoid redundant file reads.
        """
        policy_file = self._policy_file(policy_id)
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


# ---------------------------------------------------------------------------
# Bounded policy genes (Tripwire-Governor S18)
# ---------------------------------------------------------------------------
#
# A gene is a SCALE in [floor, 1.0] on one tier limit.  1.0 is the safe-side human default: the tier value
# a person wrote.  The learner may only tighten (move toward the floor), at most 10% per gene per patch, and
# never past the floor.  There are at most MAX_GENES.  The owner envelope's caps (spend caps, ladder
# thresholds, kill switches, path denylist) are NOT genes: check_cap_change refuses to widen them.
# Everything here is pure; nothing is wired to a caller, and the learned values take effect only behind the
# learned_genes flag (default OFF, registered in scripts/policy_flags.toml).

MAX_GENES = 8
GENE_STEP_MAX = Fraction(1, 10)          # a patch may tighten a gene by at most 10% of its current value
LEARNED_GENES_KEY = "learned_genes"      # Policy.config key holding {gene name: scale}
LEARNED_GENES_FLAG = "learned_genes"     # name in scripts/policy_flags.toml
LEARNED_GENES_ENV = "LOCI_LEARNED_GENES"  # must be "1" for the genes to take effect
DEFAULT_FLAG_REGISTRY = Path(__file__).resolve().parents[1] / "scripts" / "policy_flags.toml"
_DRAW_OVERSHOOT = 0.1                    # exploration overshoots both ends of the log range, then clips to them


class GeneBoundsError(ValueError):
    """A gene name or value is outside what the evolver may produce."""


class CapViolation(ValueError):
    """A change would widen a cap of the owner envelope (or tighten it by more than one step)."""


@dataclass(frozen=True)
class GeneSpec:
    """One evolvable limit: ``tier_key`` of Policy.get_constraints() scaled by a value in [floor, default]."""
    name: str
    tier_key: str
    floor: float
    default: float = 1.0


GENES: tuple = (
    GeneSpec("retrieval_hops_scale", "max_retrieval_hops", 0.5),
    GeneSpec("refinement_steps_scale", "max_refinement_steps", 0.5),
    GeneSpec("perspectives_scale", "max_perspectives", 0.5),
    GeneSpec("fanout_scale", "max_fanout", 0.4),
    GeneSpec("context_chars_scale", "max_context_chars", 0.25),
)
GENE_BY_NAME: Dict[str, GeneSpec] = {g.name: g for g in GENES}


def validate_gene_set(genes: tuple) -> tuple:
    """GeneBoundsError unless ``genes`` has 1..MAX_GENES specs with unique names and 0 < floor < default."""
    if not 0 < len(genes) <= MAX_GENES:
        raise GeneBoundsError(f"{len(genes)} genes; at most {MAX_GENES} are allowed")
    if len({g.name for g in genes}) != len(genes):
        raise GeneBoundsError("gene names must be unique")
    for g in genes:
        if not (0.0 < g.floor < g.default):
            raise GeneBoundsError(f"gene {g.name}: need 0 < floor < default, got floor={g.floor!r} default={g.default!r}")
    return genes


validate_gene_set(GENES)


def default_genome() -> Dict[str, float]:
    """Every gene at its human default (the do-nothing genome)."""
    return {g.name: g.default for g in GENES}


def clamp_gene(name: str, value: float) -> float:
    """``value`` clipped to [floor, default] of gene ``name``; GeneBoundsError for an unknown gene or non-finite value."""
    spec = GENE_BY_NAME.get(name)
    if spec is None:
        raise GeneBoundsError(f"unknown gene {name!r}")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise GeneBoundsError(f"gene {name}: {value!r} is not a finite number")
    return min(max(float(value), spec.floor), spec.default)


def draw_gene(name: str, rng: random.Random) -> float:
    """One exploratory draw for a gene: log-uniform over [floor, default], overshooting each end by 10% of the
    log range and then clipped, so the bounds themselves are reachable and never exceeded."""
    spec = GENE_BY_NAME[name]
    t = rng.uniform(-_DRAW_OVERSHOOT, 1.0 + _DRAW_OVERSHOOT)
    value = spec.floor * (spec.default / spec.floor) ** t
    return min(max(value, spec.floor), spec.default)


def draw_genome(rng: random.Random) -> Dict[str, float]:
    return {g.name: draw_gene(g.name, rng) for g in GENES}


def _q(x: Any) -> Fraction:
    """Exact value of a number as written: 0.9 is 9/10, not its binary neighbour."""
    return x if isinstance(x, Fraction) else Fraction(str(x))


def read_learned_genes(config: Mapping[str, Any]) -> tuple:
    """``(valid genes, problems)`` from ``config['learned_genes']``.

    An entry that is unknown, not a finite number, or outside [floor, default] is NOT used (the human default
    applies instead) and is named in ``problems`` so the degradation is reported, not silent.
    """
    raw = config.get(LEARNED_GENES_KEY) if isinstance(config, Mapping) else None
    if raw is None:
        return {}, []
    if not isinstance(raw, Mapping):
        return {}, [f"{LEARNED_GENES_KEY} is not an object"]
    genes: Dict[str, float] = {}
    problems: List[str] = []
    for name, value in raw.items():
        spec = GENE_BY_NAME.get(name)
        if spec is None:
            problems.append(f"unknown gene {name!r}")
        elif (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
              or not spec.floor <= value <= spec.default):
            problems.append(f"gene {name}={value!r} outside [{spec.floor}, {spec.default}]")
        else:
            genes[name] = float(value)
    return genes, problems


def apply_genes(constraints: Mapping[str, int], genes: Mapping[str, float]) -> Dict[str, int]:
    """``constraints`` with each gene's scale applied: floor(limit * scale), at least 1 when the limit is positive.

    A gene outside [floor, default] is ignored here as well (defence in depth); the result never exceeds the input.
    """
    out = dict(constraints)
    for name, scale in genes.items():
        spec = GENE_BY_NAME.get(name)
        if spec is None or isinstance(scale, bool) or not isinstance(scale, (int, float)) \
                or not math.isfinite(scale) or not spec.floor <= scale <= spec.default:
            continue
        limit = out.get(spec.tier_key)
        if isinstance(limit, int) and limit > 0:
            out[spec.tier_key] = max(1, min(limit, math.floor(limit * scale)))
    return out


def check_genome_step(current: Mapping[str, float], proposed: Mapping[str, float]) -> bool:
    """Validate a patch from ``current`` to ``proposed``; True when it WIDENS at least one gene.

    GeneBoundsError for an unknown gene, a value outside [floor, default], or a tightening of more than
    10% of the current value.  Genes absent from ``current`` start at their default.
    """
    widening = False
    for name, new in proposed.items():
        spec = GENE_BY_NAME.get(name)
        if clamp_gene(name, new) != new:
            raise GeneBoundsError(f"gene {name}: {new!r} is outside [{spec.floor}, {spec.default}]")
        old = current.get(name, spec.default)
        if _q(new) > _q(old):
            widening = True
        elif _q(new) < _q(old) * (1 - GENE_STEP_MAX):
            raise GeneBoundsError(f"gene {name}: {old!r} -> {new!r} tightens by more than {GENE_STEP_MAX * 100}% in one patch")
    return widening


def check_cap_change(caps: Mapping[str, float], proposed: Mapping[str, float]) -> Dict[str, float]:
    """The owner envelope's caps are not genes.  Return ``proposed`` merged over ``caps`` if every change only
    tightens (lowers) a cap by at most 10%; CapViolation('... widen ...') for a raise, a new cap, or a cut over 10%.
    """
    merged = dict(caps)
    for name, new in proposed.items():
        if name not in caps:
            raise CapViolation(f"cap {name!r} is not in the envelope; the learner cannot add caps")
        if isinstance(new, bool) or not isinstance(new, (int, float)) or not math.isfinite(new) or new < 0:
            raise CapViolation(f"cap {name}: {new!r} is not a finite non-negative number")
        old = caps[name]
        if _q(new) > _q(old):
            raise CapViolation(f"cap {name}: {old!r} -> {new!r} would widen the cap; only the owner may widen")
        if _q(new) < _q(old) * (1 - GENE_STEP_MAX):
            raise CapViolation(f"cap {name}: {old!r} -> {new!r} tightens by more than {GENE_STEP_MAX * 100}% in one step")
        merged[name] = new
    return merged


def registered_off(name: str, registry_path: Optional[Path] = None) -> bool:
    """Is ``name`` registered as a default-off learned flag?  An unreadable or missing registry counts as YES
    (fail closed): a feature whose switch cannot be read is off."""
    try:
        with open(registry_path or DEFAULT_FLAG_REGISTRY, "rb") as fh:
            flags = tomllib.load(fh).get("flag", [])
    except (OSError, tomllib.TOMLDecodeError):
        return True
    return any(isinstance(f, dict) and f.get("name") == name for f in flags)


def learned_feature_on(
    name: str,
    *,
    environ: Optional[Mapping[str, str]] = None,
    registry_path: Optional[Path] = None,
) -> bool:
    """A learned feature is ON only when ``LOCI_LEARNED_GENES=1`` is set AND it is not registered default-off.
    Removing the registry entry is the owner's act; the environment alone cannot turn a registered feature on."""
    env = os.environ if environ is None else environ
    if name != LEARNED_GENES_FLAG:
        return False
    if env.get(LEARNED_GENES_ENV) != "1":
        return False
    return not registered_off(name, registry_path)
