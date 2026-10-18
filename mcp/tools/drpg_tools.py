"""DRPG MCP tools: mcp__loci__policy_evolve, mcp__loci__policy_trajectory_push."""
from __future__ import annotations

import json

from loci.drpg import evolve_policy as _evolve_policy, _default_trajectory_store


def register(mcp_server):
    """Register DRPG evolution tools on the given FastMCP server."""

    @mcp_server.tool()
    def policy_evolve(policy_id: str) -> str:
        """Run DRPG evolution loop for a policy: score trajectory, promote/demote, return decision."""
        try:
            from loci.policy import PolicyBackend
            backend = PolicyBackend()
            store = _default_trajectory_store()
            result = _evolve_policy(policy_id, backend, store)
            return json.dumps(result)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @mcp_server.tool()
    def policy_trajectory_push(
        policy_id: str,
        step: int,
        action: str,
        reward_vector: dict,
    ) -> str:
        """Push a reward step to the trajectory store for a given policy_id."""
        try:
            store = _default_trajectory_store()
            step_dict = {
                "step": step,
                "action": action,
                "reward": reward_vector,
                "policy_id": policy_id,
            }
            store.push(policy_id, step_dict)
            return json.dumps({"pushed": True, "policy_id": policy_id, "step": step})
        except Exception as exc:
            return json.dumps({"error": str(exc)})
