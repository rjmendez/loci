"""LiteLLM MCP tools: mcp__loci__litellm_call, mcp__loci__litellm_health."""
from __future__ import annotations

import json
from typing import Dict, List

from loci.litellm_client import call_litellm_proxy, check_proxy_health
from loci.policy import downgrade_model_to_tier, PolicyTier


def register(mcp_server):
    """Register LiteLLM proxy tools on the given FastMCP server."""

    @mcp_server.tool()
    def litellm_call(
        model: str,
        messages: List[Dict[str, str]],
        tier: str = "medium",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Make a governed LiteLLM proxy call with tier-based model routing."""
        tier_enum = PolicyTier(tier) if tier else PolicyTier.MEDIUM
        decision = downgrade_model_to_tier(model, tier_enum, pool=False)   # name only models the proxy serves (LOCI_TIER_MODEL_MAP)
        resolved_model, was_rewritten = decision
        if getattr(decision, "tier_unresolved", False) and not decision.usable_model:
            return json.dumps({
                "error": f"tier {tier_enum.value} allows no model; the requested model was not called",
                "model": "",
                "original_model": model,
            })
        try:
            response = call_litellm_proxy(
                model=resolved_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            result = response.to_dict()
            if was_rewritten:
                result["model_rewritten"] = True
                result["original_model"] = model
            return json.dumps(result)
        except Exception as exc:
            return json.dumps({"error": str(exc), "model": resolved_model})
    @mcp_server.tool()
    def litellm_health() -> str:
        """Check LiteLLM proxy health: status ok / auth_required / error / unreachable, and whether it answered."""
        return json.dumps(check_proxy_health())
