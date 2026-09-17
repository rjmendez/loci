#!/usr/bin/env python3
"""Named local model tags for optional specialist routing.

These constants are additive catalog entries only. Importing this module must not
change any existing default tier, environment variable, or runtime routing path.
"""
from __future__ import annotations


CODE_SPECIALIST_MODEL = "qwen2.5-coder:7b"
MATH_SPECIALIST_MODEL = "hf.co/bartowski/Qwen2.5-Math-7B-Instruct-GGUF:Q4_K_M"
# 2026-09-17 quality-first benchmark: qwen3.8:latest and llama-guard3:8b both
# passed all 15 adversarial safety cases. Latency was recorded but not used to
# break the quality tie.
SAFETY_SPECIALIST_MODELS = (
    "llama-guard3:8b",
    "qwen3.8:latest",
)
TOOL_CALLING_SPECIALIST_MODEL = "hf.co/eaddario/Watt-Tool-8B-GGUF:Q4_K_M"

SWARM_CHEAP_FANOUT_MODELS = (
    "heretic-llama31-8b-instruct:latest",
    "llama3.1-agent:latest",
)
SWARM_GUARDIAN_MODELS = SAFETY_SPECIALIST_MODELS
SWARM_ESCALATION_MODEL = "qwen3.8:latest"
SWARM_SYNTHESIS_MODEL = "qwen3.8:latest"

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
