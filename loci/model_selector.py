"""
Ollama-aware model selection by PolicyTier.

Maps PolicyTier levels to specific local Ollama model names,
with cloud API fallbacks. Reads LOCI_TIER_MODEL_MAP env var
as JSON override, or falls back to machine-specific defaults.
"""

from __future__ import annotations
import json
import os
from typing import Dict, List, Optional

_DEFAULT_OLLAMA_TIER_MAP: Dict[str, List[str]] = {
    "zero": [],
    "nano": [
        "ollama/openthai-systemone:0.8b-q8_0",
        "ollama/antares-1b:Q8_0",
        "ollama/qwen2.5:3b",
    ],
    "light": [
        "ollama/qwen2.5:3b",
        "ollama/huihui_ai/qwen3.5-abliterated:4b",
        "ollama/granite3-guardian:2b",
        "claude-haiku-4.5",
    ],
    "standard": [
        "ollama/ministral3-8b-reasoning-uncensored:Q4_K_M",
        "ollama/qwen3.5-9b-defiant-fable-heretic:Q4_K_M",
        "ollama/foundation-sec-8b-reasoning:Q4_K_M",
        "claude-sonnet-4.6",
    ],
    "medium": [
        "ollama/ministral3-8b-reasoning-uncensored:Q4_K_M",
        "ollama/qwen3.5-9b-defiant-fable-heretic:Q4_K_M",
    ],
    "turbo": [
        "ollama/gemma4-12b-heretic:Q4_K_M",
        "claude-sonnet-4.6",
    ],
    "heavy": [
        "ollama/qwen3.8-27b-heretic:64k",
        "claude-opus-4.8",
        "gpt-5.4",
    ],
    "research": [
        "ollama/qwen3.8-27b-heretic:64k",
        "claude-opus-4.8",
    ],
}

_TASK_MODEL_OVERRIDES: Dict[str, Dict[str, str]] = {
    "security_review": {
        "standard": "ollama/foundation-sec-8b-reasoning:Q4_K_M",
        "light": "ollama/vulnllm-r-7b:Q4_K_M",
    },
    "rtl_verify": {
        "standard": "ollama/verireason-rtl-7b:Q4_K_M",
    },
    "code": {
        "standard": "ollama/qwen2.5-coder-agent:latest",
        "light": "ollama/qwen2.5-coder:7b",
    },
    "triage": {
        "nano": "ollama/antares-1b:Q8_0",
        "light": "ollama/qwen2.5:3b",
    },
}


def get_tier_model_map() -> Dict[str, List[str]]:
    """Load tier->model map from env override or defaults."""
    override = os.getenv("LOCI_TIER_MODEL_MAP")
    if override:
        try:
            data = json.loads(override)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return _DEFAULT_OLLAMA_TIER_MAP


def select_model(
    tier: str,
    task_type: Optional[str] = None,
    prefer_local: bool = True,
) -> str:
    """
    Select the best model for a given tier and task type.

    Args:
        tier: PolicyTier value string (nano/light/standard/heavy etc.)
        task_type: Optional task hint (security_review/code/triage/rtl_verify)
        prefer_local: If True, prefer Ollama local models over API

    Returns:
        Model name string (Ollama-prefixed or API name)
    """
    tier_lower = tier.lower()

    if task_type and task_type in _TASK_MODEL_OVERRIDES:
        override = _TASK_MODEL_OVERRIDES[task_type].get(tier_lower)
        if override:
            return override

    tier_map = get_tier_model_map()
    candidates = tier_map.get(tier_lower, tier_map.get("standard", []))

    if not candidates:
        return "ollama/antares-1b:Q8_0"

    if prefer_local:
        local = [m for m in candidates if m.startswith("ollama/")]
        if local:
            return local[0]

    return candidates[0]


def resolve_for_loci(
    requested_model: str,
    tier: str,
    task_type: Optional[str] = None,
) -> tuple[str, bool]:
    """
    Resolve model: if requested model is in a higher tier than allowed,
    downgrade to tier-appropriate model.

    Returns (resolved_model, was_downgraded).
    """
    tier_order = {
        "zero": 0, "nano": 1, "light": 2, "standard": 3,
        "medium": 3, "turbo": 4, "heavy": 5, "research": 6,
    }
    requested_tier_order = 5

    tier_map = get_tier_model_map()
    for t, models in tier_map.items():
        if requested_model in models:
            requested_tier_order = tier_order.get(t, 5)
            break

    allowed_order = tier_order.get(tier.lower(), 3)

    if requested_tier_order <= allowed_order:
        return requested_model, False

    resolved = select_model(tier, task_type)
    return resolved, resolved != requested_model