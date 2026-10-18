"""DRPG MCP tools: mcp__loci__policy_evolve, mcp__loci__policy_trajectory_push, mcp__loci__policy_evolve_all."""
from __future__ import annotations

import asyncio
import json

from loci.drpg import evolve_policy as _evolve_policy, evolve_all_policies as _evolve_all, _default_trajectory_store


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

    @mcp_server.tool()
    def policy_evolve_all(policy_ids: list = None) -> str:
        """Evolve all (or a given list of) policies in parallel via ProcessPoolExecutor.

        If policy_ids is omitted or empty, all policies found in the trajectory store
        are evolved. Returns a summary with total, promoted, demoted, and error counts.
        """
        try:
            from pathlib import Path
            import os
            store = _default_trajectory_store()
            ids = policy_ids if policy_ids else store.list_policies()
            traj_dir = Path(os.environ.get("LOCI_TRAJECTORY_DIR", str(Path.home() / ".loci" / "trajectories")))
            from loci.policy import PolicyBackend
            backend = PolicyBackend()
            policy_dir = backend.base_dir
            results = asyncio.run(_evolve_all(ids, policy_dir, traj_dir))
            promoted = sum(1 for r in results if r.get("action") == "promote")
            demoted = sum(1 for r in results if r.get("action") == "demote")
            errors = sum(1 for r in results if r.get("action") == "error")
            return json.dumps({
                "total": len(results),
                "promoted": promoted,
                "demoted": demoted,
                "errors": errors,
                "results": results,
            })
        except Exception as exc:
            return json.dumps({"error": str(exc)})
