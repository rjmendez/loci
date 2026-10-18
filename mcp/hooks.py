"""Backward-compat shim. Import from loci.hooks directly."""
from loci.hooks import *  # noqa: F401, F403
from loci.hooks import (  # noqa: F401
    loci_policy_hook, grounding_hook_augmented, tier_router_surface, eval_hook_stub,
    PolicyDecision, PolicyViolation, StepFeedback,
)

