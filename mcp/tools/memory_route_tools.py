"""Memory route policy MCP tools: mcp__loci__memory_route_to_policy, mcp__loci__counterfactual_gate."""
from __future__ import annotations

import json

from loci.memory_route_adapter import adapt_route_optimization_to_policy, evaluate_counterfactual_gate


def register(mcp_server):
    """Register memory routing tools on the given FastMCP server."""

    @mcp_server.tool()
    def memory_route_to_policy(optimize_result_json: str) -> str:
        """Convert memory_route_policy_optimize output to a stored Policy document."""
        try:
            result = adapt_route_optimization_to_policy(optimize_result_json)
            return json.dumps(result)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @mcp_server.tool()
    def counterfactual_gate(
        simulate_result_json: str,
        max_changed_fraction: float = 0.05,
    ) -> str:
        """Evaluate whether a tier downgrade is safe based on counterfactual simulation results."""
        try:
            result = evaluate_counterfactual_gate(simulate_result_json, max_changed_fraction)
            return json.dumps(result)
        except Exception as exc:
            return json.dumps({"error": str(exc)})
