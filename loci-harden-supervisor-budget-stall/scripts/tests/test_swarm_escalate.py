from __future__ import annotations

import math
import pathlib
import sys


REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import swarm_escalate as S  # noqa: E402


def _config(*, subtasks=None, fanout_count: int = 4) -> S.SwarmConfig:
    return S.SwarmConfig(
        topic="swarm reasoning topic",
        cheap_model="cheap-model:latest",
        escalate_model="strong-model:latest",
        synthesize_model="synth-model:latest",
        decompose_model="planner-model:latest",
        subtasks=subtasks,
        fanout_count=fanout_count,
    )


def _assert_valid(result: dict) -> None:
    assert S.validate_swarm_result(result) == []
    assert result["schema_version"] == 1


def test_swarm_normal_flow_with_escalation():
    calls = []

    def _generate_batch(prompts, model=None, max_tokens=256, fmt=None):  # noqa: ARG001
        calls.append((model, list(prompts)))
        if model == "planner-model:latest":
            return [{
                "ok": True,
                "text": """{"subtasks":["Check auth MFA policy","Check cache TTL","Check auth MFA controls","Inspect logging"]}""",
            }]
        if model == "cheap-model:latest":
            return [
                {"ok": True, "text": '{"answer":"Auth requires MFA.","confidence":"high"}'},
                {"ok": True, "text": '{"answer":"Cache TTL is five minutes.","confidence":"low"}'},
                {"ok": True, "text": '{"answer":"Auth does not require MFA.","confidence":"high"}'},
                {"ok": True, "text": '{"answer":"Audit logs are enabled.","confidence":"high"}'},
            ]
        if model == "strong-model:latest":
            return [
                {"ok": True, "text": '{"answer":"Auth requires MFA for privileged access.","confidence":"high"}'},
                {"ok": True, "text": '{"answer":"Cache TTL is five minutes by default.","confidence":"high"}'},
                {"ok": True, "text": '{"answer":"Auth requires MFA for privileged access.","confidence":"high"}'},
            ]
        if model == "synth-model:latest":
            return [{
                "ok": True,
                "text": '{"schema_version":1,"topic":"swarm reasoning topic","summary":"Escalation reconciled auth and cache answers."}',
            }]
        raise AssertionError(f"unexpected model: {model}")

    result = S.run_swarm(_config(), deps={"generate_batch": _generate_batch})

    _assert_valid(result)
    assert result["stats"] == {"fanout_count": 4, "escalated_count": 3, "escalation_rate": 0.75}
    assert result["summary"] == "Escalation reconciled auth and cache answers."
    assert result["findings"][0]["tier_reached"] == "escalated"
    assert result["findings"][1]["tier_reached"] == "escalated"
    assert result["findings"][2]["tier_reached"] == "escalated"
    assert result["findings"][3]["tier_reached"] == "cheap"
    assert result["tiers"]["escalate"]["attempted"] == 3
    assert any(model == "strong-model:latest" and len(prompts) == 3 for model, prompts in calls)


def test_swarm_all_cheap_success_no_escalation():
    calls = []

    def _generate_batch(prompts, model=None, max_tokens=256, fmt=None):  # noqa: ARG001
        calls.append((model, list(prompts)))
        if model == "cheap-model:latest":
            return [
                {"ok": True, "text": '{"answer":"Auth requires MFA.","confidence":"high"}'},
                {"ok": True, "text": '{"answer":"Cache TTL is five minutes.","confidence":"high"}'},
                {"ok": True, "text": '{"answer":"Logs are retained for 30 days.","confidence":"high"}'},
            ]
        if model == "synth-model:latest":
            return [{"ok": True, "text": '{"summary":"Cheap tier handled all subtasks."}'}]
        raise AssertionError(f"unexpected model: {model}")

    result = S.run_swarm(_config(subtasks=["Check auth", "Check cache", "Check logging"], fanout_count=3), deps={"generate_batch": _generate_batch})

    _assert_valid(result)
    assert result["decomposition"]["source"] == "supplied"
    assert result["stats"]["escalated_count"] == 0
    assert result["tiers"]["escalate"]["attempted"] == 0
    assert all(item["tier_reached"] == "cheap" for item in result["findings"])
    assert not any(model == "strong-model:latest" for model, _ in calls)


def test_dead_cheap_tier_fails_open_with_valid_result():
    def _generate_batch(prompts, model=None, max_tokens=256, fmt=None):  # noqa: ARG001
        if model == "planner-model:latest":
            return [{"ok": True, "text": '{"subtasks":["Check auth","Check cache"]}'}]
        if model == "cheap-model:latest":
            raise RuntimeError("cheap tier offline")
        if model == "strong-model:latest":
            return [
                {"ok": False, "text": "", "why": "strong tier offline"},
                {"ok": False, "text": "", "why": "strong tier offline"},
            ]
        if model == "synth-model:latest":
            return [{"ok": False, "text": "", "why": "synth offline"}]
        raise AssertionError(f"unexpected model: {model}")

    result = S.run_swarm(_config(fanout_count=2), deps={"generate_batch": _generate_batch})

    _assert_valid(result)
    assert result["degraded"] is True
    assert result["stats"] == {"fanout_count": 2, "escalated_count": 2, "escalation_rate": 1.0}
    assert all(item["confidence"] == "low" for item in result["findings"])
    assert all(item["ok"] is False for item in result["findings"])
    assert result["tiers"]["escalate"]["failed_open"] == 2
    assert "Swarm processed 2 subtasks" in result["summary"]


def test_escalation_rate_computed_correctly():
    def _generate_batch(prompts, model=None, max_tokens=256, fmt=None):  # noqa: ARG001
        if model == "cheap-model:latest":
            return [
                {"ok": True, "text": '{"answer":"A","confidence":"high"}'},
                {"ok": True, "text": '{"answer":"B","confidence":"low"}'},
                {"ok": True, "text": '{"answer":"C","confidence":"high"}'},
            ]
        if model == "strong-model:latest":
            return [{"ok": True, "text": '{"answer":"B2","confidence":"high"}'}]
        if model == "synth-model:latest":
            return [{"ok": True, "text": '{"summary":"One escalation happened."}'}]
        raise AssertionError(f"unexpected model: {model}")

    result = S.run_swarm(_config(subtasks=["one", "two", "three"], fanout_count=3), deps={"generate_batch": _generate_batch})

    _assert_valid(result)
    assert result["stats"]["fanout_count"] == 3
    assert result["stats"]["escalated_count"] == 1
    assert math.isclose(result["stats"]["escalation_rate"], 1 / 3)


def test_schema_validation_catches_shape_errors_and_passes_real_result():
    errors = S.validate_swarm_result({"schema_version": 2, "topic": "", "findings": [], "summary": "", "stats": {}})
    assert errors

    def _generate_batch(prompts, model=None, max_tokens=256, fmt=None):  # noqa: ARG001
        if model == "cheap-model:latest":
            return [{"ok": True, "text": '{"answer":"Done","confidence":"high"}'}]
        if model == "synth-model:latest":
            return [{"ok": True, "text": '{"summary":"Valid final JSON."}'}]
        raise AssertionError(f"unexpected model: {model}")

    result = S.run_swarm(_config(subtasks=["single"], fanout_count=1), deps={"generate_batch": _generate_batch})

    _assert_valid(result)


def test_huge_subtasks_are_truncated_in_prompts_not_in_returned_findings():
    """Regression test: a live run against 10 real diff-review subtasks (each a multi-KB
    diff pasted verbatim as the "subtask") produced garbled cheap-tier answers and a
    context-overflowed synthesis call that never returned valid JSON, forcing the
    mechanical fallback summary. Root cause: the full subtask text was re-embedded,
    uncapped, into both the per-subtask answer prompt and the synthesis prompt. This
    test pins the fix: prompts sent to the model must be bounded regardless of how large
    a subtask's text is, while the findings returned to the caller keep full fidelity."""
    huge_subtask = "DIFF CONTEXT " + ("x" * 5000)
    seen_prompts = {"cheap": [], "synth": []}

    def _generate_batch(prompts, model=None, max_tokens=256, fmt=None):  # noqa: ARG001
        if model == "cheap-model:latest":
            seen_prompts["cheap"].extend(prompts)
            return [{"ok": True, "text": '{"answer":"Looks fine.","confidence":"high"}'} for _ in prompts]
        if model == "synth-model:latest":
            seen_prompts["synth"].extend(prompts)
            return [{"ok": True, "text": '{"summary":"All clear."}'}]
        raise AssertionError(f"unexpected model: {model}")

    result = S.run_swarm(
        _config(subtasks=[huge_subtask], fanout_count=1),
        deps={"generate_batch": _generate_batch},
    )

    _assert_valid(result)
    # The prompt actually sent to the cheap tier must be bounded, not a multi-KB blob.
    assert len(seen_prompts["cheap"][0]) < 2000
    assert "[truncated]" in seen_prompts["cheap"][0]
    # Same for whatever gets re-embedded into the synthesis prompt.
    assert len(seen_prompts["synth"][0]) < 2000
    # But the finding returned to the caller keeps the full, untruncated subtask text —
    # truncation is a prompt-construction concern only, never a data-loss concern.
    assert result["findings"][0]["subtask"] == huge_subtask
    assert result["findings"][0]["answer"] == "Looks fine."

