import json

from scripts.mission_coordinator import MissionCoordinator, mission_gate


TASK = "Identify local spider sightings around Example NC within 5 miles."
SOURCES = [
    {"name": "iNaturalist", "capability": "GPS/photo verified wildlife observations"},
    {"name": "Wikipedia", "capability": "background taxonomy only"},
]


def _gen_fn_for(task, plan=None, verdicts=None):
    def gen_fn(**kwargs):
        prompt = kwargs.get("prompt", "")
        if "planning source use" in prompt:
            payload = plan or {
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
            return json.dumps(payload)
        return json.dumps({"fail_open": False, "verdicts": verdicts or [
            {"finding_index": 0, "on_task": True, "supported": True, "source_appropriate": True, "guidance": "", "rationale": "matches task and evidence"}
        ]})
    return gen_fn


def test_mission_gate_accepts_clean_run():
    findings = [{"source": "iNaturalist", "claim": "A. aurantia seen nearby", "evidence": "photo + GPS 2.1 miles"}]
    result = mission_gate(
        TASK,
        findings,
        SOURCES,
        gen_fn=_gen_fn_for(TASK, verdicts=[{
            "finding_index": 0,
            "on_task": True,
            "supported": True,
            "source_appropriate": True,
            "guidance": "",
            "rationale": "on task and source matched",
        }]),
    )

    assert result["status"] == "accepted"
    assert result["replan"]["needed"] is False
    assert result["route"]["preferred_sources"] == ["iNaturalist"]
    assert result["verdicts"][0]["supported"] is True


def test_mission_gate_rejects_bad_source_match_and_unsupported_claim():
    findings = [{"source": "Wikipedia", "claim": "A. aurantia seen nearby", "evidence": "common in North America"}]
    verdicts = [{
        "finding_index": 0,
        "on_task": True,
        "supported": False,
        "source_appropriate": False,
        "guidance": "Use iNaturalist GPS/photo observations instead.",
        "rationale": "Wikipedia is background only and does not prove local observation",
    }]
    result = mission_gate(
        TASK,
        findings,
        SOURCES,
        gen_fn=_gen_fn_for(TASK, verdicts=verdicts),
    )

    assert result["status"] == "rejected"
    assert result["reason"].startswith("mission gate failed")
    assert result["verdicts"][0]["source_appropriate"] is False
    assert result["replan"]["needed"] is True


def test_mission_gate_blocks_missing_generator_and_empty_task():
    blocked = mission_gate("", [{"source": "Wikipedia", "claim": "x", "evidence": "y"}], SOURCES)
    assert blocked["status"] == "blocked"
    assert "missing task objective" in blocked["reason"]

    missing = mission_gate(TASK, [{"source": "Wikipedia", "claim": "x", "evidence": "y"}], SOURCES, gen_fn=None)
    assert missing["status"] == "blocked"
    assert "generator function missing" in missing["reason"]


def test_mission_coordinator_class_is_robust_to_partial_input():
    result = MissionCoordinator(task=None, findings="not-a-list", sources={"name": "iNaturalist"}, gen_fn=_gen_fn_for(TASK)).run()
    assert result["status"] in {"blocked", "accepted", "rejected"}
    assert isinstance(result["verdicts"], list)
    assert isinstance(result["route"], dict)
