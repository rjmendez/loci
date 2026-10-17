"""
Lightweight LiteLLM proxy client for Loci MCP.

Allows Copilot CLI and other MCP clients to make governed, cost-optimal
LiteLLM calls without importing the full litellm package.
Uses raw HTTP to the LiteLLM proxy endpoint.
"""

import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

LITELLM_PROXY_URL = os.getenv("LOCI_LITELLM_PROXY_URL", "http://localhost:4000")
LITELLM_API_KEY = os.getenv("LOCI_LITELLM_API_KEY", "")
LITELLM_TIMEOUT_S = float(os.getenv("LOCI_LITELLM_TIMEOUT_S", "30"))

# Inline tier→model map (mirrors policy.py TIER_MODEL_MAP, kept here to avoid circular imports)
_TIER_MODEL_MAP = {
    "light":  ["claude-haiku-4.5", "gpt-5-mini", "mistral-7b-instruct"],
    "medium": ["claude-sonnet-4.6", "gpt-5.4-mini", "llama-3-70b-instruct"],
    "heavy":  ["claude-opus-4.8", "gpt-5.4", "gpt-5.5"],
}

@dataclass
class LiteLLMResponse:
    model: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "content": self.content,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
            "latency_ms": round(self.latency_ms, 1),
        }


def resolve_model_for_tier_simple(requested_model: str, tier: str) -> tuple:
    """Resolve model against tier constraint. Returns (resolved_model, was_rewritten)."""
    tier_order = {"light": 0, "medium": 1, "heavy": 2}
    allowed = _TIER_MODEL_MAP.get(tier, [])

    requested_tier = None
    for t, models in _TIER_MODEL_MAP.items():
        if requested_model in models:
            requested_tier = t
            break

    if requested_tier is not None and tier_order.get(requested_tier, 1) <= tier_order.get(tier, 1):
        return requested_model, False

    if allowed:
        return allowed[0], True
    return requested_model, False


def call_litellm_proxy(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> LiteLLMResponse:
    """
    Make a completion call to the LiteLLM proxy via HTTP.
    Raises urllib.error.URLError on network failure.
    Raises ValueError on non-200 response.
    """
    url = f"{LITELLM_PROXY_URL.rstrip('/')}/v1/chat/completions"

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(payload)),
    }
    if LITELLM_API_KEY:
        headers["Authorization"] = f"Bearer {LITELLM_API_KEY}"

    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=LITELLM_TIMEOUT_S) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise ValueError(f"LiteLLM proxy error {e.code}: {body}") from e
    latency_ms = (time.monotonic() - t0) * 1000

    choice = raw["choices"][0]["message"]["content"]
    usage = raw.get("usage", {})

    return LiteLLMResponse(
        model=raw.get("model", model),
        content=choice,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        latency_ms=latency_ms,
        raw=raw,
    )
