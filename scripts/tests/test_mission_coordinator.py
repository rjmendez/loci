"""The mission coordinator is a gate: whatever it cannot verify must come back blocked or rejected, never accepted.

The supervisor underneath is fail-open on purpose (a generator that fails or answers garbage yields a default plan and
neutral all-true verdicts, flagged ``fail_open``), so the wrapper has to read those flags. A run where the model was
down used to come back ``accepted``.
"""
import json

import pytest

from scripts import mission_coordinator as MC
from scripts.mission_coordinator import MissionCoordinator, mission_gate


TASK = "Identify local spider sightings around Example NC within 5 miles."
SOURCES = [
    {"name": "iNaturalist", "capability": "GPS/photo verified wildlife observations"},
    {"name": "Wikipedia", "capability": "background taxonomy only"},
]
FINDING = {"source": "iNaturalist", "claim": "A. aurantia seen nearby", "evidence": "photo + GPS 2.1 miles"}
CLEAN = {"finding_index": 0, "on_task": True, "supported": True, "source_appropriate": True, "guidance": "", "rationale": "ok"}


def _plan(task=TASK):
    return {
        "task": task,
        "fail_open": False,
        "decision_tree": [{
            "sub_need": "GPS/photo-verified local observations",
            "preferred_sources": ["iNaturalist"],
            "fallback_sources": ["Wikipedia"],
            "rationale": "Local photo/GPS evidence belongs in iNaturalist.",
        }],
        "assignments": [{"sub_need": "GPS/photo-verified local observations", "source": "iNaturalist", "rationale": "best match"}],
        "source_policy": "Use iNaturalist for local observations; Wikipedia is background only.",
    }


def _gen_fn_for(task=TASK, plan=None, verdicts=None):
    def gen_fn(**kwargs):
        if "planning source use" in kwargs.get("prompt", ""):
            return json.dumps(plan or _plan(task))
        return json.dumps({"fail_open": False, "verdicts": verdicts or [CLEAN]})
    return gen_fn


def _verdict(**overrides):
    return {**CLEAN, **overrides}


# ------------------------------------------------------------------------------------------- decisions

def test_mission_gate_accepts_clean_run():
    result = mission_gate(TASK, [FINDING], SOURCES, gen_fn=_gen_fn_for())
    assert result["status"] == "accepted"
    assert result["replan"] == {"needed": False, "reason": "no correction or route change required"}
    assert result["route"]["preferred_sources"] == ["iNaturalist"]
    assert result["verdicts"][0]["supported"] is True


def test_a_single_finding_dict_is_one_finding():
    assert mission_gate(TASK, FINDING, SOURCES, gen_fn=_gen_fn_for())["status"] == "accepted"


def test_mission_gate_rejects_bad_source_match_and_unsupported_claim():
    findings = [{"source": "Wikipedia", "claim": "A. aurantia seen nearby", "evidence": "common in North America"}]
    verdicts = [_verdict(supported=False, source_appropriate=False, guidance="Use iNaturalist GPS/photo observations instead.",
                         rationale="Wikipedia is background only and does not prove local observation")]
    result = mission_gate(TASK, findings, SOURCES, gen_fn=_gen_fn_for(verdicts=verdicts))
    assert result["status"] == "rejected"
    assert result["reason"] == "mission gate failed: outputs drifted from task objective or lacked verified evidence"
    assert result["verdicts"][0]["source_appropriate"] is False
    assert result["replan"] == {"needed": True, "reason": "unsupported, off-task, or wrong-source finding detected"}


@pytest.mark.parametrize("flag", ["on_task", "supported", "source_appropriate"])
def test_any_one_failed_check_rejects(flag):
    result = mission_gate(TASK, [FINDING], SOURCES, gen_fn=_gen_fn_for(verdicts=[_verdict(**{flag: False})]))
    assert result["status"] == "rejected"


def test_a_rejection_without_a_worker_makes_no_repair_attempt():
    result = mission_gate(TASK, [FINDING], SOURCES, gen_fn=_gen_fn_for(verdicts=[_verdict(on_task=False)]))
    assert result["status"] == "rejected" and "attempted" not in result["replan"]


# ------------------------------------------------------------------------------------------- repair

def _flipping_gen_fn(first_verdict, then_verdict, flagged_reviews=2):
    """Plan, then ``first_verdict`` for the first ``flagged_reviews`` reviews and ``then_verdict`` for every later one.
    Two by default: the gate reviews once, then supervise_and_correct reviews again before it repairs anything."""
    reviews = []

    def gen_fn(**kwargs):
        if "planning source use" in kwargs.get("prompt", ""):
            return json.dumps(_plan())
        reviews.append(1)
        return json.dumps({"fail_open": False, "verdicts": [first_verdict if len(reviews) <= flagged_reviews else then_verdict]})
    gen_fn.reviews = reviews
    return gen_fn


def test_a_worker_repair_that_brings_the_finding_back_is_accepted():
    workers = []

    def worker_fn(**ctx):
        workers.append(ctx)
        return {**ctx["finding"], "claim": "narrowed to what the photo shows"}
    gen = _flipping_gen_fn(_verdict(on_task=False, guidance="stay on task"), CLEAN)
    result = mission_gate(TASK, [FINDING], SOURCES, gen_fn=gen, worker_fn=worker_fn)
    assert result["status"] == "accepted" and result["reason"] == "worker repair brought outputs back within mission scope"
    assert len(workers) == 1 and workers[0]["guidance"] == "stay on task"
    assert result["replan"]["attempted"] is True and result["replan"]["converged"] is True
    assert result["verdicts"][0]["on_task"] is True


def test_a_finding_that_is_flagged_and_then_clean_with_nothing_repaired_is_not_accepted():
    """The model changed its mind between two reviews; the finding did not change. That is not a repair."""
    workers = []
    gen = _flipping_gen_fn(_verdict(on_task=False), CLEAN, flagged_reviews=1)
    result = mission_gate(TASK, [FINDING], SOURCES, gen_fn=gen, worker_fn=lambda **ctx: workers.append(1) or dict(ctx["finding"]))
    assert result["status"] == "rejected" and workers == []
    assert result["replan"]["converged"] is False
    assert result["replan"]["reason"].startswith("reviews disagreed (flagged, then clean on re-review")


def test_a_worker_repair_that_does_not_fix_it_is_rejected_with_the_attempt_recorded():
    gen = _flipping_gen_fn(_verdict(on_task=False), _verdict(on_task=False))
    result = mission_gate(TASK, [FINDING], SOURCES, gen_fn=gen, worker_fn=lambda **ctx: dict(ctx["finding"]))
    assert result["status"] == "rejected"
    assert (result["replan"]["attempted"], result["replan"]["converged"]) == (True, False)
    assert result["verdicts"][0]["on_task"] is False


def test_a_claim_the_evidence_check_refutes_is_not_repaired_it_hard_fails_under_the_strict_tripwire():
    """The model says clean but the evidence check says unsupported: the tripwire is on evidence, not on opinion."""
    workers = []
    result = mission_gate(TASK, [FINDING], SOURCES, gen_fn=_gen_fn_for(), evidence_fn=lambda *a, **k: False,
                          worker_fn=lambda **ctx: workers.append(1) or dict(ctx["finding"]))
    assert result["status"] == "rejected" and workers == []
    assert result["replan"]["attempted"] is True and result["replan"]["converged"] is False
    assert result["verdicts"][0]["tripwire_triggered"] is True


def test_a_refuted_claim_is_rejected_with_the_tripwire_on_its_verdict_when_there_is_no_worker():
    result = mission_gate(TASK, [FINDING], SOURCES, gen_fn=_gen_fn_for(), evidence_fn=lambda *a, **k: False)
    assert result["status"] == "rejected" and result["verdicts"][0]["tripwire_triggered"] is True
    assert result["verdicts"][0]["supported"] is False


def test_the_evidence_check_confirming_the_claim_lets_it_through():
    result = mission_gate(TASK, [FINDING], SOURCES, gen_fn=_gen_fn_for(), evidence_fn=lambda *a, **k: True)
    assert result["status"] == "accepted" and result["verdicts"][0]["evidence_checked"] is True


def test_an_evidence_check_that_could_not_decide_blocks_instead_of_passing():
    result = mission_gate(TASK, [FINDING], SOURCES, gen_fn=_gen_fn_for(), evidence_fn=lambda *a, **k: None)
    assert result["status"] == "blocked"
    assert result["reason"] == "evidence check inconclusive for 1 finding(s): the claim was not verified"


def test_every_inconclusive_finding_is_counted():
    second = {"source": "iNaturalist", "claim": "second", "evidence": "photo"}
    verdicts = [CLEAN, _verdict(finding_index=1)]
    result = mission_gate(TASK, [FINDING, second], SOURCES, gen_fn=_gen_fn_for(verdicts=verdicts), evidence_fn=lambda *a, **k: None)
    assert result["status"] == "blocked"
    assert result["reason"] == "evidence check inconclusive for 2 finding(s): the claim was not verified"
    assert len(result["verdicts"]) == 2


def test_a_models_own_unsupported_verdict_is_repairable_without_an_evidence_check():
    gen = _flipping_gen_fn(_verdict(supported=False), CLEAN)
    result = mission_gate(TASK, [FINDING], SOURCES, gen_fn=gen, worker_fn=lambda **ctx: {**ctx["finding"], "claim": "qualified"})
    assert result["status"] == "accepted" and result["replan"]["converged"] is True


def test_a_repair_that_raises_is_a_rejection_not_a_crash():
    def worker_fn(**ctx):
        raise RuntimeError("worker died")
    gen = _flipping_gen_fn(_verdict(on_task=False), CLEAN)
    result = mission_gate(TASK, [FINDING], SOURCES, gen_fn=gen, worker_fn=worker_fn)
    assert result["status"] == "rejected" and result["replan"]["converged"] is False


def test_a_repair_call_that_itself_raises_is_a_rejection_not_a_crash(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("supervisor blew up")
    monkeypatch.setattr(MC, "supervise_and_correct", boom)
    gen = _flipping_gen_fn(_verdict(on_task=False), CLEAN)
    result = mission_gate(TASK, [FINDING], SOURCES, gen_fn=gen, worker_fn=lambda **ctx: dict(ctx["finding"]))
    assert result["status"] == "rejected"
    assert (result["replan"]["attempted"], result["replan"]["converged"]) == (True, False)
    assert result["replan"]["result"]["error"] == "supervisor blew up"


def test_a_repair_that_degrades_to_fail_open_cannot_accept(monkeypatch):
    monkeypatch.setattr(MC, "supervise_and_correct", lambda *a, **k: {"fail_open": True, "converged": True, "rounds": 1, "verdicts": [CLEAN]})
    gen = _flipping_gen_fn(_verdict(on_task=False), CLEAN)
    result = mission_gate(TASK, [FINDING], SOURCES, gen_fn=gen, worker_fn=lambda **ctx: dict(ctx["finding"]))
    assert result["status"] == "rejected" and result["replan"]["converged"] is False


def test_the_replanned_routing_plan_comes_from_the_audit_trail_when_there_is_one():
    plan_obj = {"task": "t", "decision_tree": []}
    assert MissionCoordinator._replanned_routing_plan({"audit_trail": [{}, {"replanned_routing_plan": plan_obj}]}, "fallback") is plan_obj
    for repair in ({}, {"audit_trail": []}, {"audit_trail": "x"}, {"audit_trail": ["not a dict"]}, {"audit_trail": [{}]}):
        assert MissionCoordinator._replanned_routing_plan(repair, "fallback") == "fallback"


# ------------------------------------------------------------------------------------------- fail closed

def test_no_task_and_no_generator_block():
    blocked = mission_gate("", [FINDING], SOURCES, gen_fn=_gen_fn_for())
    assert blocked["status"] == "blocked" and blocked["reason"] == "missing task objective"
    assert blocked["replan"] == {"needed": True, "reason": "missing task objective"}
    missing = mission_gate(TASK, [FINDING], SOURCES, gen_fn=None)
    assert missing["status"] == "blocked"
    assert missing["reason"] == "generator function missing; fail-closed mission gate cannot validate outputs"


def test_no_findings_block_instead_of_passing_vacuously():
    for findings in ([], None):
        result = mission_gate(TASK, findings, SOURCES, gen_fn=_gen_fn_for())
        assert result["status"] == "blocked" and result["reason"] == "no findings to evaluate"


def test_malformed_findings_block_and_say_how_many_were_dropped():
    result = mission_gate(TASK, ["a string", FINDING, 7], SOURCES, gen_fn=_gen_fn_for())
    assert result["status"] == "blocked" and result["reason"].startswith("malformed findings: 2 item(s)")
    assert mission_gate(TASK, "not-a-list", SOURCES, gen_fn=_gen_fn_for())["reason"].startswith("malformed findings: 1 item(s)")


def test_a_generator_that_answers_garbage_blocks_and_does_not_accept():
    """The supervisor turns this into a default plan and neutral all-true verdicts; that must not be a pass."""
    result = mission_gate(TASK, [FINDING], SOURCES, gen_fn=lambda **kw: "I am not JSON")
    assert result["status"] == "blocked"
    assert result["reason"].startswith("routing plan degraded (supervisor fail-open)")


def test_a_generator_that_raises_blocks():
    def gen_fn(**kw):
        raise RuntimeError("model down")
    result = mission_gate(TASK, [FINDING], SOURCES, gen_fn=gen_fn)
    assert result["status"] == "blocked" and result["reason"].startswith("routing plan degraded")


def test_garbage_verdicts_after_a_good_plan_block():
    def gen_fn(**kw):
        return json.dumps(_plan()) if "planning source use" in kw.get("prompt", "") else "also not JSON"
    result = mission_gate(TASK, [FINDING], SOURCES, gen_fn=gen_fn)
    assert result["status"] == "blocked"
    assert result["reason"] == "verification degraded (supervisor fail-open): findings could not be verified"
    assert result["route"]["preferred_sources"] == ["iNaturalist"]           # the plan that was made is still reported


def test_exceptions_in_routing_or_supervision_block_with_the_cause(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(MC, "plan_source_routing", boom)
    assert mission_gate(TASK, [FINDING], SOURCES, gen_fn=_gen_fn_for())["reason"] == "routing plan failed: boom"
    monkeypatch.undo()
    monkeypatch.setattr(MC, "supervise_findings", boom)
    result = mission_gate(TASK, [FINDING], SOURCES, gen_fn=_gen_fn_for())
    assert result["status"] == "blocked" and result["reason"] == "mission gate could not evaluate findings: boom"
    assert result["route"]["preferred_sources"] == ["iNaturalist"]


def test_a_non_dict_plan_or_result_blocks(monkeypatch):
    monkeypatch.setattr(MC, "plan_source_routing", lambda *a, **k: None)
    assert mission_gate(TASK, [FINDING], SOURCES, gen_fn=_gen_fn_for())["reason"].startswith("routing plan degraded")
    monkeypatch.undo()
    monkeypatch.setattr(MC, "supervise_findings", lambda *a, **k: None)
    assert mission_gate(TASK, [FINDING], SOURCES, gen_fn=_gen_fn_for())["reason"].startswith("verification degraded")


def test_a_fail_open_supervisor_result_blocks_even_when_every_verdict_looks_clean(monkeypatch):
    monkeypatch.setattr(MC, "supervise_findings", lambda *a, **k: {"fail_open": True, "verdicts": [CLEAN]})
    result = mission_gate(TASK, [FINDING], SOURCES, gen_fn=_gen_fn_for())
    assert result["status"] == "blocked" and result["verdicts"] == []


def test_the_coordinator_is_robust_to_partial_input():
    result = MissionCoordinator(task=None, findings="not-a-list", sources={"name": "iNaturalist"}, gen_fn=_gen_fn_for(TASK)).run()
    assert result["status"] == "blocked" and result["reason"] == "missing task objective"
    assert result["verdicts"] == [] and result["route"]["preferred_sources"] == []


# ------------------------------------------------------------------------------------------- normalising

def test_sources_keep_only_named_objects_and_accept_the_alternative_keys():
    got = MissionCoordinator._normalize_sources([
        {"name": " iNat ", "capability": " photos "}, {"source": "GBIF", "description": "records"},
        {"name": ""}, {"capability": "no name"}, "not a dict", None])
    assert got == [{"name": "iNat", "capability": "photos"}, {"name": "GBIF", "capability": "records"}]
    assert MissionCoordinator._normalize_sources("nope") == [] and MissionCoordinator._normalize_sources(None) == []


def test_findings_are_copied_so_the_callers_objects_are_not_shared():
    original = {"claim": "x"}
    kept, dropped = MissionCoordinator._normalize_findings([original])
    kept[0]["claim"] = "changed"
    assert original == {"claim": "x"} and dropped == 0


def test_the_task_is_stripped_and_a_non_string_task_is_stringified():
    assert MissionCoordinator("  spiders  ", [FINDING]).task == "spiders"
    assert MissionCoordinator(42, [FINDING]).task == "42"
    assert MissionCoordinator(None, [FINDING]).task == ""


def test_the_route_prefers_assignments_then_the_decision_tree():
    c = MissionCoordinator(TASK, [FINDING])
    assert c._route_payload({"assignments": [{"source": "A"}, "junk", {"source": " "}], "decision_tree": [{"preferred_sources": ["B"]}]})["preferred_sources"] == ["A"]
    assert c._route_payload({"assignments": [], "decision_tree": [{"preferred_sources": ["B", " "]}, "junk"]})["preferred_sources"] == ["B"]
    assert c._route_payload(None) == {"task": TASK, "preferred_sources": [], "source_policy": "", "decision_tree": [], "assignments": []}


# ------------------------------------------------------------------------------------------- CLI

def _cli(argv, gen_fn, capsys):
    code = MC._main(argv, gen_fn=gen_fn)
    return code, json.loads(capsys.readouterr().out)


PAYLOAD = json.dumps({"task": TASK, "sources": SOURCES, "findings": [FINDING]})


def test_the_cli_exit_code_follows_the_gate_decision(capsys):
    code, out = _cli(["--payload", PAYLOAD], _gen_fn_for(), capsys)
    assert (code, out["status"]) == (MC.EXIT_ACCEPTED, "accepted") and MC.EXIT_ACCEPTED == 0
    code, out = _cli(["--payload", PAYLOAD], _gen_fn_for(verdicts=[_verdict(supported=False)]), capsys)
    assert (code, out["status"]) == (MC.EXIT_REJECTED, "rejected") and MC.EXIT_REJECTED == 1
    code, out = _cli(["--payload", PAYLOAD], lambda **kw: "garbage", capsys)
    assert (code, out["status"]) == (MC.EXIT_BLOCKED, "blocked") and MC.EXIT_BLOCKED == 2


def test_the_cli_takes_the_three_options_when_there_is_no_payload(capsys):
    code, out = _cli(["--task", TASK, "--sources", json.dumps(SOURCES), "--findings", json.dumps([FINDING])], _gen_fn_for(), capsys)
    assert code == 0 and out["status"] == "accepted"


def test_the_cli_blocks_on_bad_input_instead_of_falling_back(capsys):
    for argv in (["--payload", "{not json"], ["--payload", "[1, 2]"], ["--task", TASK, "--findings", "{oops"], ["--sources", "nope"]):
        code, out = _cli(argv, _gen_fn_for(), capsys)
        assert code == MC.EXIT_BLOCKED and out["status"] == "blocked" and out["reason"].startswith("bad input:")


def test_the_cli_blocks_when_no_generator_can_be_found(monkeypatch, capsys):
    monkeypatch.setattr(MC, "_default_gen_fn", lambda: None)
    code, out = _cli(["--payload", PAYLOAD], None, capsys)
    assert code == MC.EXIT_BLOCKED and out["reason"].startswith("generator function missing")


def test_the_cli_does_not_invent_a_routing_plan_of_its_own(monkeypatch, capsys):
    """The first version answered every call, plan and verdict alike, with a canned plan built from the sources."""
    seen = []
    monkeypatch.setattr(MC, "_default_gen_fn", lambda: (lambda **kw: seen.append(kw.get("prompt", "")) or "garbage"))
    code, out = _cli(["--payload", PAYLOAD], None, capsys)
    assert seen and code == MC.EXIT_BLOCKED


def test_the_default_generator_is_the_local_model_or_none():
    gen = MC._default_gen_fn()
    assert gen is None or callable(gen)
