#!/usr/bin/env python3
"""Named local model tags for optional specialist routing.

These constants are additive catalog entries only. Importing this module must not
change any existing default tier, environment variable, or runtime routing path.
"""
from __future__ import annotations


CODE_SPECIALIST_MODEL = "qwen2.5-coder:7b"
MATH_SPECIALIST_MODEL = "hf.co/bartowski/Qwen2.5-Math-7B-Instruct-GGUF:Q4_K_M"
SAFETY_SPECIALIST_MODEL = "llama-guard3:8b"
TOOL_CALLING_SPECIALIST_MODEL = "hf.co/eaddario/Watt-Tool-8B-GGUF:Q4_K_M"

SPECIALIST_MODELS = {
    "code": CODE_SPECIALIST_MODEL,
    "math": MATH_SPECIALIST_MODEL,
    "safety": SAFETY_SPECIALIST_MODEL,
    "tool_calling": TOOL_CALLING_SPECIALIST_MODEL,
}

__all__ = [
    "CODE_SPECIALIST_MODEL",
    "MATH_SPECIALIST_MODEL",
    "SAFETY_SPECIALIST_MODEL",
    "TOOL_CALLING_SPECIALIST_MODEL",
    "SPECIALIST_MODELS",
]
