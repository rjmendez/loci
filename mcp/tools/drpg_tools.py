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
        """Evolve all (or a given list of) policies serially, at most 64 per call.

        If policy_ids is omitted or empty, all policies found in the trajectory store
        are evolved.  Returns total, promoted, demoted, errors, policies_evolved (results that
        are not errors), rows_read (trajectory rows analyzed) and per-policy results, plus
        ``status``: 0 on success, 1 with an ``error`` when nothing could be read
        ('rows_read=0'), more than 64 policies were requested, or anything else failed.
        """
        try:
            from loci.policy import PolicyBackend
            store = _default_trajectory_store()
            ids = policy_ids if policy_ids else store.list_policies()
            backend = PolicyBackend()
            results = asyncio.run(_evolve_all(ids, backend.policy_dir, store.base_dir))
            promoted = sum(1 for r in results if r.get("action") == "promote")
            demoted = sum(1 for r in results if r.get("action") == "demote")
            errors = sum(1 for r in results if r.get("action") == "error")
            rows_read = sum(int(r.get("steps_analyzed", 0)) for r in results)
            summary = {
                "total": len(results),
                "promoted": promoted,
                "demoted": demoted,
                "errors": errors,
                "policies_evolved": len(results) - errors,
                "rows_read": rows_read,
                "results": results,
                "status": 0,
            }
            if rows_read == 0:
                summary["status"] = 1
                summary["error"] = "rows_read=0"
            return json.dumps(summary)
        except Exception as exc:
            return json.dumps({"error": str(exc), "status": 1})
