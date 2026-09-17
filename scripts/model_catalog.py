#!/usr/bin/env python3
"""Named local model tags for optional specialist routing.

These constants are additive catalog entries only. Importing this module must not
change any existing default tier, environment variable, or runtime routing path.
"""
from __future__ import annotations


CODE_SPECIALIST_MODEL = "qwen2.5-coder:7b"
MATH_SPECIALIST_MODEL = "hf.co/bartowski/Qwen2.5-Math-7B-Instruct-GGUF:Q4_K_M"
# 2026-09-17 host audit: the resident guardrail-labelled tags timed out on every
# real prompt, so keep a responsive local fallback in the catalog until those
# dedicated models are actually serviceable on this host.
SAFETY_SPECIALIST_MODEL = "qwen2.5:3b"
TOOL_CALLING_SPECIALIST_MODEL = "hf.co/eaddario/Watt-Tool-8B-GGUF:Q4_K_M"

SWARM_CHEAP_FANOUT_MODEL = "qwen2.5:3b"
SWARM_GUARDIAN_MODEL = "qwen2.5:3b"
SWARM_ESCALATION_MODEL = "heretic-llama31-8b-instruct:latest"
SWARM_SYNTHESIS_MODEL = "qwen2.5:3b"

SPECIALIST_MODELS = {
    "code": CODE_SPECIALIST_MODEL,
    "math": MATH_SPECIALIST_MODEL,
    "safety": SAFETY_SPECIALIST_MODEL,
    "tool_calling": TOOL_CALLING_SPECIALIST_MODEL,
}

SWARM_ROLE_MODELS = {
    "cheap_fanout": SWARM_CHEAP_FANOUT_MODEL,
    "guardian": SWARM_GUARDIAN_MODEL,
    "escalation": SWARM_ESCALATION_MODEL,
    "synthesis": SWARM_SYNTHESIS_MODEL,
}

__all__ = [
    "CODE_SPECIALIST_MODEL",
    "MATH_SPECIALIST_MODEL",
    "SAFETY_SPECIALIST_MODEL",
    "TOOL_CALLING_SPECIALIST_MODEL",
    "SPECIALIST_MODELS",
    "SWARM_CHEAP_FANOUT_MODEL",
    "SWARM_GUARDIAN_MODEL",
    "SWARM_ESCALATION_MODEL",
    "SWARM_SYNTHESIS_MODEL",
    "SWARM_ROLE_MODELS",
]
