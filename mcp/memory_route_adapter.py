"""Backward-compat shim. Import from loci.memory_route_adapter directly."""
from loci.memory_route_adapter import *  # noqa: F401, F403
from loci.memory_route_adapter import (  # noqa: F401
    RouteOptimizationResult, adapt_route_optimization_to_policy, evaluate_counterfactual_gate,
)
