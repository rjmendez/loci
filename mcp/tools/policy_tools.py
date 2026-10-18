"""Policy MCP tools: mcp__loci__policy_store, mcp__loci__policy_load."""
from __future__ import annotations

import json
from typing import Any, Optional

from loci.policy import Policy, PolicyBackend, PolicyTier, PolicyRegion, validate_policy


def register(mcp_server):
    """Register policy tools on the given FastMCP server."""

    @mcp_server.tool()
    def policy_store(
        title: str,
        tier: str = "medium",
        region: str = "memory",
        labels: Optional[list[str]] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> str:
        """Store a policy.md document (tier-aware retrieval/routing constraints) and return its policy_id."""
        try:
            policy = Policy(
                title=title,
                tier=PolicyTier(tier),
                region=PolicyRegion(region),
                labels=labels or [],
                config=config or {},
            )
            backend = PolicyBackend()
            result = backend.store(policy)
            return json.dumps(result)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})

    @mcp_server.tool()
    def policy_load(policy_id: str) -> str:
        """Load a stored policy.md document by policy_id; returns its full serialized form or not_found."""
        backend = PolicyBackend()
        policy = backend.load(policy_id)
        if policy is None:
            return json.dumps({"error": "not_found", "policy_id": policy_id})
        return json.dumps(policy.to_dict())
