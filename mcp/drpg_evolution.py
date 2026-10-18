"""Backward-compat shim. Import from loci.drpg directly."""
from loci.drpg import *  # noqa: F401, F403
from loci.drpg import (  # noqa: F401
    TrajectoryStore, TrajectoryScorer, PolicyPromoter,
    DrpgSuggestionGenerator, evolve_policy, evolve_all_policies,
    _default_trajectory_store, _REWARD_DIMS, _REWARD_WEIGHTS_DEFAULT,
)
