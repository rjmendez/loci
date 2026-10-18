"""
Loci: policy-aware retrieval and memory library.

Usage:
    from loci import PolicyTier, PolicyBackend, TrajectoryStore, LiteLLMClient
    from loci.policy import Policy, PolicyConfig
    from loci.cache import check_memory_cache
    from loci.drpg import evolve_policy
"""
from loci.policy import Policy, PolicyTier, PolicyRegion, PolicyBackend, PolicyConfig, get_policy_config, validate_policy, TIER_MODEL_MAP, get_tier_model_map, resolve_model_for_tier, downgrade_model_to_tier
from loci.drpg import TrajectoryStore, TrajectoryScorer, PolicyPromoter, DrpgSuggestionGenerator, evolve_policy
from loci.cache import check_memory_cache, format_cache_response, CacheHit
from loci.graph_cache import GraphCache, make_cache_key
from loci.litellm_client import call_litellm_proxy, LiteLLMResponse
from loci.investigation_policy import cap_perspectives_to_tier, evaluate_pre_answer_for_refusal, filter_findings_for_verification
from loci.memory_route_adapter import adapt_route_optimization_to_policy, evaluate_counterfactual_gate

__all__ = [
    "Policy", "PolicyTier", "PolicyRegion", "PolicyBackend", "PolicyConfig",
    "get_policy_config", "validate_policy", "TIER_MODEL_MAP", "get_tier_model_map",
    "resolve_model_for_tier", "downgrade_model_to_tier", "TrajectoryStore", "TrajectoryScorer",
    "PolicyPromoter", "DrpgSuggestionGenerator", "evolve_policy", "check_memory_cache",
    "format_cache_response", "CacheHit", "GraphCache", "make_cache_key",
    "call_litellm_proxy", "LiteLLMResponse",
    "cap_perspectives_to_tier", "evaluate_pre_answer_for_refusal",
    "filter_findings_for_verification", "adapt_route_optimization_to_policy",
    "evaluate_counterfactual_gate",
]
