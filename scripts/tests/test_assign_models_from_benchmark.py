from __future__ import annotations

import pathlib
import sys


REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import assign_models_from_benchmark as A  # noqa: E402


def test_parse_ollama_list_reads_name_column():
    text = """NAME                 ID      SIZE
qwen2.5:3b           abc     1.9 GB
local-heretic:8b     def     4.9 GB
"""
    assert A.parse_ollama_list(text) == {"qwen2.5:3b", "local-heretic:8b"}


def test_benchmark_winners_collects_unique_by_role():
    payload = {
        "summary": [
            {"role": "synthesis", "quality_winners": ["qwen3.8:latest", "qwen3.8:latest"]},
            {"role": "guardian", "quality_winners": ["llama-guard3:8b"]},
        ]
    }
    assert A.benchmark_winners(payload) == {
        "synthesis": ["qwen3.8:latest"],
        "guardian": ["llama-guard3:8b"],
    }


def test_choose_redteam_prefers_installed_heretic_from_winners():
    winners = {
        "synthesis": ["qwen3.8:latest", "local-heretic-qwen38-27b:q4km"],
        "escalation": ["qwen3.8:latest"],
    }
    installed = {"qwen3.8:latest", "local-heretic-qwen38-27b:q4km"}
    model, present = A.choose_redteam(winners, installed)
    assert model == "local-heretic-qwen38-27b:q4km"
    assert present is True
