from __future__ import annotations

import importlib
import pathlib
import sys


REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))


def test_specialist_model_catalog_is_additive_only():
    swarm = importlib.import_module("swarm_escalate")
    before = (
        swarm._DEFAULT_CHEAP_MODEL,
        swarm._DEFAULT_ESCALATE_MODEL,
        swarm._DEFAULT_SYNTHESIZE_MODEL,
        swarm._DEFAULT_DECOMPOSE_MODEL,
    )

    catalog = importlib.import_module("model_catalog")

    assert catalog.CODE_SPECIALIST_MODEL == "qwen2.5-coder:7b"
    assert catalog.MATH_SPECIALIST_MODEL == "hf.co/bartowski/Qwen2.5-Math-7B-Instruct-GGUF:Q4_K_M"
    assert catalog.SAFETY_SPECIALIST_MODEL == "llama-guard3:8b"
    assert catalog.TOOL_CALLING_SPECIALIST_MODEL == "hf.co/eaddario/Watt-Tool-8B-GGUF:Q4_K_M"
    assert catalog.SPECIALIST_MODELS == {
        "code": catalog.CODE_SPECIALIST_MODEL,
        "math": catalog.MATH_SPECIALIST_MODEL,
        "safety": catalog.SAFETY_SPECIALIST_MODEL,
        "tool_calling": catalog.TOOL_CALLING_SPECIALIST_MODEL,
    }
    assert (
        swarm._DEFAULT_CHEAP_MODEL,
        swarm._DEFAULT_ESCALATE_MODEL,
        swarm._DEFAULT_SYNTHESIZE_MODEL,
        swarm._DEFAULT_DECOMPOSE_MODEL,
    ) == before
