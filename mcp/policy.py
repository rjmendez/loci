"""Backward-compat shim. Import from loci.policy directly."""
from loci.policy import *  # noqa: F401, F403
from loci.policy import (  # noqa: F401
    Policy, PolicyTier, PolicyRegion, PolicyBackend, PolicyConfig,
    PolicyMetadata, get_policy_config, validate_policy,
    TIER_MODEL_MAP, get_tier_model_map, resolve_model_for_tier,
)
