from __future__ import annotations

import json
import math
import pathlib
import re
import sys
import time


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

    def _generate_batch(prompts, model=None, max_tokens=256, fmt=None, think=False):  # noqa: ARG001
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

    def _generate_batch(prompts, model=None, max_tokens=256, fmt=None, think=False):  # noqa: ARG001
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


def test_stigmergic_gate_accepts_corroborated_duplicates_without_escalation():
    calls = []

    def _generate_batch(prompts, model=None, max_tokens=256, fmt=None, think=False):  # noqa: ARG001
        calls.append((model, list(prompts)))
        if model == "cheap-model:latest":
            return [
                {"ok": True, "text": '{"answer":"Auth requires MFA because mcp/server.py:42 enforces it.","confidence":"high"}'},
                {"ok": True, "text": '{"answer":"Auth requires MFA because mcp/server.py:42 enforces it.","confidence":"high"}'},
            ]
        if model == "synth-model:latest":
            return [{"ok": True, "text": '{"summary":"Corroborated cheap answers were accepted."}'}]
        raise AssertionError(f"unexpected model: {model}")

    config = _config(subtasks=["Check auth MFA", "Verify auth MFA"], fanout_count=2)
    config.stigmergic_consensus = True
    result = S.run_swarm(config, deps={"generate_batch": _generate_batch})

    _assert_valid(result)
    assert result["stats"]["escalated_count"] == 0
    assert result["stigmergic_consensus"]["stats"]["accepted_clusters"] == 1
    assert not any(model == "strong-model:latest" for model, _ in calls)


def test_stigmergic_gate_escalates_under_corroborated_high_confidence():
    def _generate_batch(prompts, model=None, max_tokens=256, fmt=None, think=False):  # noqa: ARG001
        if model == "cheap-model:latest":
            return [{"ok": True, "text": '{"answer":"Cache TTL appears to be five minutes.","confidence":"high"}'}]
        if model == "strong-model:latest":
            return [{"ok": True, "text": '{"answer":"Cache TTL is five minutes by default.","confidence":"high"}'}]
        if model == "synth-model:latest":
            return [{"ok": True, "text": '{"summary":"Single cheap answer was escalated."}'}]
        raise AssertionError(f"unexpected model: {model}")

    config = _config(subtasks=["Check cache TTL"], fanout_count=1)
    config.stigmergic_consensus = True
    result = S.run_swarm(config, deps={"generate_batch": _generate_batch})

    _assert_valid(result)
    assert result["stats"]["escalated_count"] == 1
    assert result["triage"]["escalation_reasons"]["0"]
    assert result["stigmergic_consensus"]["clusters"][0]["state"] == "degraded"


def test_dead_cheap_tier_fails_open_with_valid_result():
    def _generate_batch(prompts, model=None, max_tokens=256, fmt=None, think=False):  # noqa: ARG001
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
    def _generate_batch(prompts, model=None, max_tokens=256, fmt=None, think=False):  # noqa: ARG001
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

    def _generate_batch(prompts, model=None, max_tokens=256, fmt=None, think=False):  # noqa: ARG001
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

    def _generate_batch(prompts, model=None, max_tokens=256, fmt=None, think=False):  # noqa: ARG001
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


def _seed_from_prompt(prompt: str) -> int:
    match = re.search(r"Independent seed:\s*(\d+)/", prompt)
    assert match, prompt
    return int(match.group(1))


def test_multi_seed_dispatches_seed_rounds_concurrently():
    planner_calls = []

    def _generate_batch(prompts, model=None, max_tokens=256, fmt=None, think=False):  # noqa: ARG001
        if model == "planner-model:latest":
            seed = _seed_from_prompt(prompts[0])
            planner_calls.append(seed)
            time.sleep(0.25)
            return [{"ok": True, "text": f'{{"subtasks":["seed {seed} task"]}}'}]
        if model == "cheap-model:latest":
            return [{"ok": True, "text": '{"answer":"done","confidence":"high"}'} for _ in prompts]
        if model == "synth-model:latest":
            return [{"ok": True, "text": '{"summary":"multi-seed ok"}'}]
        raise AssertionError(f"unexpected model: {model}")

    config = _config(fanout_count=1)
    config.seeds = 4

    start = time.perf_counter()
    result = S.run_swarm(config, deps={"generate_batch": _generate_batch})
    elapsed = time.perf_counter() - start

    _assert_valid(result)
    assert elapsed < 0.65, f"expected concurrent seeds, took {elapsed:.2f}s"
    assert sorted(planner_calls) == [1, 2, 3, 4]
    assert result["multi_seed"]["completed_seeds"] == 4
    assert result["summary"] == "multi-seed ok"


def test_multi_seed_seed_failure_does_not_sink_run(monkeypatch):
    original = S._run_single_seed

    def _patched(config, batch_fn, *, seed_index=0, seed_count=1):
        if seed_index == 1:
            raise RuntimeError("seed exploded")
        return original(config, batch_fn, seed_index=seed_index, seed_count=seed_count)

    monkeypatch.setattr(S, "_run_single_seed", _patched)

    def _generate_batch(prompts, model=None, max_tokens=256, fmt=None, think=False):  # noqa: ARG001
        if model == "planner-model:latest":
            seed = _seed_from_prompt(prompts[0])
            return [{"ok": True, "text": f'{{"subtasks":["seed {seed} task"]}}'}]
        if model == "cheap-model:latest":
            return [{"ok": True, "text": '{"answer":"done","confidence":"high"}'} for _ in prompts]
        if model == "synth-model:latest":
            return [{"ok": True, "text": '{"summary":"survived one dead seed"}'}]
        raise AssertionError(f"unexpected model: {model}")

    config = _config(fanout_count=1)
    config.seeds = 3
    result = S.run_swarm(config, deps={"generate_batch": _generate_batch})

    _assert_valid(result)
    assert result["degraded"] is False
    assert result["summary"] == "survived one dead seed"
    assert result["multi_seed"]["completed_seeds"] == 2
    assert result["multi_seed"]["failed_seeds"] == 1
    assert result["multi_seed"]["errors"][0]["seed"] == 1
    assert len(result["findings"]) == 2


def test_multi_seed_dedupes_findings_and_bounds_synthesis_prompt():
    seen_synth_prompts = []

    def _generate_batch(prompts, model=None, max_tokens=256, fmt=None, think=False):  # noqa: ARG001
        if model == "planner-model:latest":
            seed = _seed_from_prompt(prompts[0])
            subtasks = ["common task"] + [f"seed {seed} task {idx}" for idx in range(1, 12)]
            return [{"ok": True, "text": json.dumps({"subtasks": subtasks})}]
        if model == "cheap-model:latest":
            rows = []
            for prompt in prompts:
                if "Subtask: common task" in prompt:
                    answer = "shared answer " + ("x" * 1200)
                else:
                    match = re.search(r"Subtask:\s*(seed \d+ task \d+)", prompt)
                    assert match, prompt
                    answer = (match.group(1) + " -> " + ("y" * 1200))
                rows.append({"ok": True, "text": json.dumps({"answer": answer, "confidence": "high"})})
            return rows
        if model == "synth-model:latest":
            seen_synth_prompts.extend(prompts)
            return [{"ok": True, "text": '{"summary":"bounded multi-seed synthesis"}'}]
        raise AssertionError(f"unexpected model: {model}")

    config = _config(fanout_count=12)
    config.seeds = 3
    result = S.run_swarm(config, deps={"generate_batch": _generate_batch})

    _assert_valid(result)
    assert result["summary"] == "bounded multi-seed synthesis"
    assert result["multi_seed"]["merge"]["input_count"] == 36
    assert result["multi_seed"]["merge"]["deduped_count"] == 2
    assert result["multi_seed"]["merge"]["merged_count"] == 34
    assert result["multi_seed"]["synthesis_input"]["dropped_count"] > 0
    assert len(seen_synth_prompts[0]) < 21000
    assert result["stats"]["fanout_count"] == 34


def test_safety_check_default_preserves_swarm_result_shape():
    def _generate_batch(prompts, model=None, max_tokens=256, fmt=None, think=False):  # noqa: ARG001
        if model == "cheap-model:latest":
            return [{"ok": True, "text": '{"answer":"Safe answer.","confidence":"high"}'}]
        if model == "synth-model:latest":
            return [{"ok": True, "text": '{"summary":"Safe synthesis."}'}]
        raise AssertionError(f"unexpected model: {model}")

    def _guardian(_text):  # pragma: no cover - must not be called
        raise AssertionError("guardian should not run by default")

    result = S.run_swarm(
        _config(subtasks=["check safety"], fanout_count=1),
        deps={"generate_batch": _generate_batch, "guardian_check": _guardian},
    )

    _assert_valid(result)
    assert "safety_flag" not in result
    assert result["findings"][0]["answer"] == "Safe answer."


def test_safety_check_adds_advisory_swarm_annotation_without_rewriting_summary():
    seen = []

    def _generate_batch(prompts, model=None, max_tokens=256, fmt=None, think=False):  # noqa: ARG001
        if model == "cheap-model:latest":
            return [{"ok": True, "text": '{"answer":"Keep the finding.","confidence":"high"}'}]
        if model == "synth-model:latest":
            return [{"ok": True, "text": '{"summary":"Quoted attack text remains evidence."}'}]
        raise AssertionError(f"unexpected model: {model}")

    def _guardian(text):
        seen.append(text)
        return {"flagged": True, "verdict": "Yes", "ok": True, "error": None}

    config = _config(subtasks=["check quoted attack text"], fanout_count=1)
    config.safety_check = True
    result = S.run_swarm(config, deps={"generate_batch": _generate_batch, "guardian_check": _guardian})

    _assert_valid(result)
    assert seen == ["Quoted attack text remains evidence."]
    assert result["summary"] == "Quoted attack text remains evidence."
    assert result["findings"][0]["answer"] == "Keep the finding."
    assert result["safety_flag"] == {"flagged": True, "why": "guardian verdict Yes"}


def test_safety_check_guardian_errors_fail_open_for_swarm():
    def _generate_batch(prompts, model=None, max_tokens=256, fmt=None, think=False):  # noqa: ARG001
        if model == "cheap-model:latest":
            return [{"ok": True, "text": '{"answer":"Safe answer.","confidence":"high"}'}]
        if model == "synth-model:latest":
            return [{"ok": True, "text": '{"summary":"Safe synthesis."}'}]
        raise AssertionError(f"unexpected model: {model}")

    def _guardian(_text):
        raise RuntimeError("guardian offline")

    config = _config(subtasks=["check safety"], fanout_count=1)
    config.safety_check = True
    result = S.run_swarm(config, deps={"generate_batch": _generate_batch, "guardian_check": _guardian})

    _assert_valid(result)
    assert result["summary"] == "Safe synthesis."
    assert result["safety_flag"]["flagged"] is False
    assert "failed open" in result["safety_flag"]["why"]


def test_resolve_config_default_no_opt_in_preserves_legacy_models():
    config = S._resolve_config(S.parse_args(["topic only"]))

    assert config.escalate_model == S._DEFAULT_ESCALATE_MODEL
    assert config.synthesize_model == S._DEFAULT_SYNTHESIZE_MODEL
    assert config.decompose_max_tokens == 1200
    assert config.synthesize_max_tokens == 1400


def test_resolve_config_opt_in_without_override_upgrades_models():
    config = S._resolve_config(S.parse_args(["topic only", "--seeds", "2"]))

    assert config.escalate_model == S._TIER_ESCALATE_MODEL
    assert config.synthesize_model == S._TIER_SYNTHESIZE_MODEL
    assert config.decompose_max_tokens == S._TIER_DECOMPOSE_MAX_TOKENS
    assert config.synthesize_max_tokens == S._TIER_SYNTHESIZE_MAX_TOKENS


def test_resolve_config_opt_in_preserves_explicit_non_default_overrides():
    config = S._resolve_config(S.parse_args([
        "topic only",
        "--seeds", "2",
        "--escalate-model", "strong-explicit:latest",
        "--synthesize-model", "synth-explicit:latest",
    ]))

    assert config.escalate_model == "strong-explicit:latest"
    assert config.synthesize_model == "synth-explicit:latest"
    assert config.decompose_max_tokens == S._TIER_DECOMPOSE_MAX_TOKENS
    assert config.synthesize_max_tokens == S._TIER_SYNTHESIZE_MAX_TOKENS


def test_resolve_config_opt_in_preserves_explicit_legacy_default_overrides():
    config = S._resolve_config(S.parse_args([
        "topic only",
        "--seeds", "2",
        "--escalate-model", S._DEFAULT_ESCALATE_MODEL,
        "--synthesize-model", S._DEFAULT_SYNTHESIZE_MODEL,
    ]))

    assert config.escalate_model == S._DEFAULT_ESCALATE_MODEL
    assert config.synthesize_model == S._DEFAULT_SYNTHESIZE_MODEL
    assert config.decompose_max_tokens == S._TIER_DECOMPOSE_MAX_TOKENS
    assert config.synthesize_max_tokens == S._TIER_SYNTHESIZE_MAX_TOKENS


def test_self_consistency_majority_replaces_low_confidence_before_escalation():
    cheap_calls = 0

    def _generate_batch(prompts, model=None, max_tokens=256, fmt=None, think=False):  # noqa: ARG001
        nonlocal cheap_calls
        if model == "cheap-model:latest":
            cheap_calls += 1
            if cheap_calls == 1:
                return [{"ok": True, "text": '{"answer":"uncertain draft","confidence":"low"}'}]
            assert len(prompts) == 3
            return [
                {"ok": True, "text": '{"answer":"majority answer","confidence":"high"}'},
                {"ok": True, "text": '{"answer":"minority answer","confidence":"low"}'},
                {"ok": True, "text": '{"answer":"majority answer","confidence":"high"}'},
            ]
        if model == "synth-model:latest":
            return [{"ok": True, "text": '{"summary":"Self-consistency avoided escalation."}'}]
        if model == "strong-model:latest":  # pragma: no cover - must not be called
            raise AssertionError("majority high-confidence answer should avoid escalation")
        raise AssertionError(f"unexpected model: {model}")

    config = _config(subtasks=["low confidence subtask"], fanout_count=1)
    config.self_consistency_samples = 3
    result = S.run_swarm(config, deps={"generate_batch": _generate_batch})

    _assert_valid(result)
    assert result["findings"][0]["answer"] == "majority answer"
    assert result["stats"]["escalated_count"] == 0
    assert result["tiers"]["self_consistency"]["attempted"] == 1
    assert result["tiers"]["self_consistency"]["majority"] == 1


def test_escalate_with_prior_context_includes_cheap_answer_in_prompt():
    captured = []

    def _generate_batch(prompts, model=None, max_tokens=256, fmt=None, think=False):  # noqa: ARG001
        if model == "cheap-model:latest":
            return [{"ok": True, "text": '{"answer":"cheap weaker draft","confidence":"low"}'}]
        if model == "strong-model:latest":
            captured.extend(prompts)
            return [{"ok": True, "text": '{"answer":"improved strong answer","confidence":"high"}'}]
        if model == "synth-model:latest":
            return [{"ok": True, "text": '{"summary":"Prior-aware escalation succeeded."}'}]
        raise AssertionError(f"unexpected model: {model}")

    config = _config(subtasks=["critique this"], fanout_count=1)
    config.escalate_with_prior_context = True
    result = S.run_swarm(config, deps={"generate_batch": _generate_batch})

    _assert_valid(result)
    assert result["findings"][0]["answer"] == "improved strong answer"
    assert "PRIOR_WEAK_ANSWER: cheap weaker draft" in captured[0]
    assert "Critique" in captured[0] or "critique" in captured[0]


def test_hierarchical_reduce_groups_findings_and_fails_open_per_group():
    final_prompts = []

    def _generate_batch(prompts, model=None, max_tokens=256, fmt=None, think=False):  # noqa: ARG001
        if model == "cheap-model:latest":
            return [
                {"ok": True, "text": json.dumps({"answer": f"answer {idx}", "confidence": "high"})}
                for idx, _ in enumerate(prompts)
            ]
        if model == "synth-model:latest" and len(prompts) == 3:
            return [
                {"ok": True, "text": '{"summary":"group one summary","confidence":"high"}'},
                {"ok": False, "text": "", "why": "group model failed"},
                {"ok": True, "text": '{"summary":"group three summary","confidence":"medium"}'},
            ]
        if model == "synth-model:latest":
            final_prompts.extend(prompts)
            return [{"ok": True, "text": '{"summary":"Reduced synthesis complete."}'}]
        raise AssertionError(f"unexpected model: {model}")

    config = _config(subtasks=[f"task {idx}" for idx in range(5)], fanout_count=5)
    config.reduce_group_size = 2
    result = S.run_swarm(config, deps={"generate_batch": _generate_batch})

    _assert_valid(result)
    assert result["summary"] == "Reduced synthesis complete."
    assert result["synthesis"]["reduce"]["group_count"] == 3
    assert result["synthesis"]["reduce"]["failed_open"] == 1
    assert "group one summary" in final_prompts[0]
    assert "Reduced finding group 2" in final_prompts[0]
    assert len(result["findings"]) == 5
