"""Tests for the `swarm_reason` MCP tool wrapper over scripts/swarm_escalate.py."""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import llm_tools  # noqa: E402
import server  # noqa: E402


class _FakeConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_swarm_reason_delegates_and_serializes(monkeypatch):
    captured = {}

    def fake_run_swarm(config, deps=None):
        captured["config"] = config
        captured["deps"] = deps
        return {
            "schema_version": 1,
            "topic": config.topic,
            "findings": [{
                "subtask": "check auth",
                "answer": "auth is enabled",
                "confidence": "medium",
                "tier_reached": "cheap",
                "model": config.cheap_model,
                "ok": True,
            }],
            "summary": "auth is enabled",
            "stats": {"fanout_count": 1, "escalated_count": 0, "escalation_rate": 0.0},
            "degraded": False,
        }

    fake_module = SimpleNamespace(
        SwarmConfig=_FakeConfig,
        run_swarm=fake_run_swarm,
        validate_swarm_result=lambda result: [],
        _DEFAULT_CHEAP_MODEL="cheap-default",
        _DEFAULT_ESCALATE_MODEL="escalate-default",
        _DEFAULT_SYNTHESIZE_MODEL="synthesize-default",
    )
    monkeypatch.setattr(llm_tools, "_load_swarm_escalate", lambda: fake_module)

    out = json.loads(server.swarm_reason(
        "why is auth failing",
        fanout_count=7,
        seeds=3,
        cheap_model="cheap-x",
        escalate_model="strong-y",
        synthesize_model="synth-z",
        decompose_model="decomp-a",
        subtasks=["check auth", "check logs"],
        escalate_confidences=["low", "medium"],
        synthesize_think=True,
        safety_check=True,
        self_consistency_samples=3,
        escalate_with_prior_context=True,
        reduce_group_size=8,
        stigmergic_consensus=True,
        stigmergic_ttl_minutes=45,
    ))

    assert out["summary"] == "auth is enabled"
    assert captured["deps"] is None
    assert captured["config"].topic == "why is auth failing"
    assert captured["config"].fanout_count == 7
    assert captured["config"].seeds == 3
    assert captured["config"].auto_parallel is False
    assert captured["config"].cheap_model == "cheap-x"
    assert captured["config"].escalate_model == "strong-y"
    assert captured["config"].synthesize_model == "synth-z"
    assert captured["config"].decompose_model == "decomp-a"
    assert captured["config"].subtasks == ["check auth", "check logs"]
    assert captured["config"].escalate_confidences == ("low", "medium")
    assert captured["config"].synthesize_think is True
    assert captured["config"].safety_check is True
    assert captured["config"].self_consistency_samples == 3
    assert captured["config"].escalate_with_prior_context is True
    assert captured["config"].reduce_group_size == 8
    assert captured["config"].stigmergic_consensus is True
    assert captured["config"].stigmergic_ttl_minutes == 45


def test_swarm_reason_fails_open_on_wrapper_exception(monkeypatch):
    monkeypatch.setattr(llm_tools, "_load_swarm_escalate",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    out = json.loads(server.swarm_reason("topic x", fanout_count=4))

    assert out["schema_version"] == 1
    assert out["topic"] == "topic x"
    assert out["degraded"] is True
    assert out["stats"] == {"fanout_count": 4, "escalated_count": 0, "escalation_rate": 0.0}
    assert out["findings"][0]["subtask"] == "topic x"
    assert out["findings"][0]["ok"] is False
    assert out["synthesis"]["degraded"] is True
    assert "boom" in out["synthesis"]["error"]


def test_llm_tools_register_includes_swarm_reason():
    registered = []

    class FakeMCP:
        def tool(self):
            return lambda fn: registered.append(fn.__name__) or fn

    llm_tools.register(FakeMCP())
    assert "swarm_reason" in registered
