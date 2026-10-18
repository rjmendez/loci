"""LiteLLM MCP tools: mcp__loci__litellm_call, mcp__loci__litellm_health."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loci.litellm_client import call_litellm_proxy, resolve_model_for_tier_simple, LITELLM_PROXY_URL


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
        resolved_model, was_rewritten = resolve_model_for_tier_simple(model, tier)
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
        """Check LiteLLM proxy health; returns proxy URL and reachability status."""
        import urllib.request, urllib.error
        url = f"{LITELLM_PROXY_URL.rstrip('/')}/health"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return json.dumps({"status": "ok", "proxy_url": LITELLM_PROXY_URL, "http_status": resp.status})
        except urllib.error.URLError as exc:
            return json.dumps({"status": "unreachable", "proxy_url": LITELLM_PROXY_URL, "error": str(exc)})
        except Exception as exc:
            return json.dumps({"status": "error", "proxy_url": LITELLM_PROXY_URL, "error": str(exc)})
