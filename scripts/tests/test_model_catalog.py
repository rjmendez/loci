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
    assert catalog.SAFETY_SPECIALIST_MODELS == (
        "llama-guard3:8b",
        "qwen3.8:latest",
    )
    assert catalog.TOOL_CALLING_SPECIALIST_MODEL == "hf.co/eaddario/Watt-Tool-8B-GGUF:Q4_K_M"
    assert catalog.SPECIALIST_MODELS == {
        "code": catalog.CODE_SPECIALIST_MODEL,
        "math": catalog.MATH_SPECIALIST_MODEL,
        "safety": catalog.SAFETY_SPECIALIST_MODELS,
        "tool_calling": catalog.TOOL_CALLING_SPECIALIST_MODEL,
    }
    assert catalog.SWARM_CHEAP_FANOUT_MODELS == (
        "qwen2.5:3b",
        "heretic-llama31-8b-instruct:latest",
        "llama3.1-agent:latest",
    )
    assert catalog.SWARM_GUARDIAN_MODELS == (
        "llama-guard3:8b",
        "qwen3.8:latest",
    )
    assert catalog.SWARM_ESCALATION_MODEL == "qwen3.8:latest"
    assert catalog.SWARM_SYNTHESIS_MODEL == "qwen3.8:latest"
    assert catalog.SWARM_ROLE_MODELS == {
        "cheap_fanout": catalog.SWARM_CHEAP_FANOUT_MODELS,
        "guardian": catalog.SWARM_GUARDIAN_MODELS,
        "escalation": catalog.SWARM_ESCALATION_MODEL,
        "synthesis": catalog.SWARM_SYNTHESIS_MODEL,
    }
    assert (
        swarm._DEFAULT_CHEAP_MODEL,
        swarm._DEFAULT_ESCALATE_MODEL,
        swarm._DEFAULT_SYNTHESIZE_MODEL,
        swarm._DEFAULT_DECOMPOSE_MODEL,
    ) == before
