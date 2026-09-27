#!/usr/bin/env python3
"""Named local model tags for optional specialist routing.

These constants are additive catalog entries only. Importing this module must not
change any existing default tier, environment variable, or runtime routing path.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Last resort when mcp/backends.py cannot be imported: small enough to load on one GPU.
_ONE_GPU_FALLBACK_MODEL = "qwen2.5:3b"


def _backends_model(resolver: str) -> str:
    """Resolve a role model the same way scripts/swarm_escalate.py does. Fail-open."""
    try:
        mcp_dir = str(Path(__file__).resolve().parents[1] / "mcp")
        if mcp_dir not in sys.path:
            sys.path.insert(0, mcp_dir)
        import backends

        return str(getattr(backends, resolver)() or "").strip() or _ONE_GPU_FALLBACK_MODEL
    except Exception:
        return _ONE_GPU_FALLBACK_MODEL


CODE_SPECIALIST_MODEL = "qwen2.5-coder:7b"
MATH_SPECIALIST_MODEL = "hf.co/bartowski/Qwen2.5-Math-7B-Instruct-GGUF:Q4_K_M"
# 2026-09-17 quality-first benchmark: qwen3.8:latest and llama-guard3:8b both
# passed all 15 adversarial safety cases. Latency was recorded but not used to
# break the quality tie. Same-day live swarm sanity check on 3 borderline safety
# prompts: qwen3.8 answered buffer-overflow/SQLi in bounded educational terms and
# refused padlock guidance; llama-guard3 timed out at 120s on all 3 prompts; the
# live-pulled granite3-guardian:2b returned a terse "No" once and otherwise
# failed the swarm JSON contract on those prompts, so it stays out of answer-stage
# safety routing for now. qwen3.8:latest was dropped 2026-09-26: at 17.7 GB it splits
# across both 11/12 GB GPUs and its loads time out.
SAFETY_SPECIALIST_MODELS = (
    "llama-guard3:8b",
)
TOOL_CALLING_SPECIALIST_MODEL = "hf.co/eaddario/Watt-Tool-8B-GGUF:Q4_K_M"

SWARM_CHEAP_FANOUT_MODELS = (
    # 2026-09-17 live /api/tags roster confirms qwen2.5:3b is already pulled on
    # the Ollama host, and it remains the unchanged runtime cheap-tier default.
    "qwen2.5:3b",
    "heretic-llama31-8b-instruct:latest",
    "llama3.1-agent:latest",
)
SWARM_GUARDIAN_MODELS = SAFETY_SPECIALIST_MODELS
# Same resolution as swarm_escalate's defaults: env -> backends.toml -> one-GPU fallback.
SWARM_ESCALATION_MODEL = _backends_model("swarm_escalate_model")
SWARM_SYNTHESIS_MODEL = _backends_model("swarm_synthesize_model")

SPECIALIST_MODELS = {
    "code": CODE_SPECIALIST_MODEL,
    "math": MATH_SPECIALIST_MODEL,
    "safety": SAFETY_SPECIALIST_MODELS,
    "tool_calling": TOOL_CALLING_SPECIALIST_MODEL,
}

SWARM_ROLE_MODELS = {
    "cheap_fanout": SWARM_CHEAP_FANOUT_MODELS,
    "guardian": SWARM_GUARDIAN_MODELS,
    "escalation": SWARM_ESCALATION_MODEL,
    "synthesis": SWARM_SYNTHESIS_MODEL,
}

__all__ = [
    "CODE_SPECIALIST_MODEL",
    "MATH_SPECIALIST_MODEL",
    "SAFETY_SPECIALIST_MODELS",
    "TOOL_CALLING_SPECIALIST_MODEL",
    "SPECIALIST_MODELS",
    "SWARM_CHEAP_FANOUT_MODELS",
    "SWARM_GUARDIAN_MODELS",
    "SWARM_ESCALATION_MODEL",
    "SWARM_SYNTHESIS_MODEL",
    "SWARM_ROLE_MODELS",
]
