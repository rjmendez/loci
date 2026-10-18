"""
Tests for loci.model_selector -- Ollama-aware model selection by PolicyTier.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from loci.model_selector import (
    get_tier_model_map,
    select_model,
    resolve_for_loci,
)


def test_select_model_nano_returns_small():
    model = select_model("nano")
    assert model.startswith("ollama/")
    assert any(x in model for x in ["0.8b", "1b", "3b", "antares", "openthai", "qwen2.5"])


def test_select_model_heavy_returns_large():
    model = select_model("heavy", prefer_local=False)
    assert any(x in model for x in ["27b", "opus", "gpt-5"])


def test_select_model_heavy_prefers_local():
    model = select_model("heavy", prefer_local=True)
    assert model.startswith("ollama/")
    assert "27b" in model


def test_task_override_security():
    model = select_model("standard", task_type="security_review")
    assert "foundation-sec" in model


def test_task_override_code():
    model = select_model("standard", task_type="code")
    assert "coder" in model


def test_task_override_triage():
    model = select_model("nano", task_type="triage")
    assert "antares-1b" in model


def test_resolve_for_loci_downgrade():
    resolved, was_downgraded = resolve_for_loci(
        "ollama/gemma4-12b-heretic:Q4_K_M", "nano"
    )
    assert was_downgraded
    assert resolved.startswith("ollama/")
    assert any(x in resolved for x in ["antares", "openthai", "qwen2.5"])


def test_resolve_for_loci_no_downgrade():
    resolved, was_downgraded = resolve_for_loci("ollama/antares-1b:Q8_0", "heavy")
    assert not was_downgraded
    assert resolved == "ollama/antares-1b:Q8_0"


def test_env_override():
    override = {
        "nano": ["custom-nano-model"],
        "standard": ["custom-standard-model"],
    }
    with patch.dict(os.environ, {"LOCI_TIER_MODEL_MAP": json.dumps(override)}):
        tier_map = get_tier_model_map()
    assert tier_map["nano"] == ["custom-nano-model"]
    assert tier_map["standard"] == ["custom-standard-model"]


def test_env_override_invalid_json_falls_back():
    with patch.dict(os.environ, {"LOCI_TIER_MODEL_MAP": "not-json"}):
        tier_map = get_tier_model_map()
    assert "nano" in tier_map
    assert len(tier_map["nano"]) > 0


def test_select_model_unknown_tier_falls_back_to_standard():
    model = select_model("unknown_tier")
    assert model
    assert "ollama/" in model or "claude" in model or "gpt" in model


def test_select_model_zero_tier_returns_fallback():
    model = select_model("zero")
    assert model == "ollama/antares-1b:Q8_0"