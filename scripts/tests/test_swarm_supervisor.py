import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.swarm_supervisor import plan_source_routing, supervise_and_correct, supervise_findings


SOURCES = [
    {"name": "DuckDuckGo", "capability": "general web search; recent/local mentions; no structured GPS filtering"},
    {"name": "Wikipedia", "capability": "general encyclopedic taxonomy/background; no GPS filtering"},
    {"name": "iNaturalist", "capability": "GPS-filterable photo-verified species observations near a location"},
]
TASK = "Identify spider species observed within 5 miles of PLACEHOLDER_LOCATION: Example NC."


def test_plan_source_routing_structure_prioritizes_inaturalist():
    def gen_fn(**_kwargs):
        return {"text": json.dumps({
            "task": TASK,
            "fail_open": False,
            "decision_tree": [
                {"sub_need": "GPS/photo-verified local observations", "preferred_sources": ["iNaturalist"], "fallback_sources": ["DuckDuckGo"], "rationale": "iNaturalist has structured nearby observations with photos and coordinates."},
                {"sub_need": "taxonomy/background", "preferred_sources": ["Wikipedia"], "fallback_sources": ["DuckDuckGo"], "rationale": "Wikipedia fits general taxonomy but not local evidence."},
            ],
            "assignments": [{"sub_need": "GPS/photo-verified local observations", "source": "iNaturalist", "rationale": "best evidence fit"}],
            "source_policy": "Use iNaturalist for local proof; use Wikipedia only for background.",
        })}

    plan = plan_source_routing(TASK, SOURCES, gen_fn)

    assert plan["fail_open"] is False
    assert plan["decision_tree"][0]["preferred_sources"] == ["iNaturalist"]
    assert any(item["source"] == "iNaturalist" for item in plan["assignments"])


def test_plan_source_routing_fail_open_on_parse_failure_uses_all_sources():
    def gen_fn(**_kwargs):
        return {"text": "not json"}

    plan = plan_source_routing(TASK, SOURCES, gen_fn)

    assert plan["fail_open"] is True
    assert plan["decision_tree"][0]["preferred_sources"] == ["DuckDuckGo", "Wikipedia", "iNaturalist"]
    assert {item["source"] for item in plan["assignments"]} == {"DuckDuckGo", "Wikipedia", "iNaturalist"}


def test_plan_source_routing_max_depth_one_is_flat_behavior():
    payload = {
        "task": TASK,
        "fail_open": False,
        "decision_tree": [{
            "sub_need": "GPS/photo-verified local observations",
            "preferred_sources": ["iNaturalist"],
            "fallback_sources": ["DuckDuckGo"],
            "fallback_condition": "ignored at depth one",
            "children": [{"sub_need": "ignored child", "preferred_sources": ["DuckDuckGo"]}],
            "rationale": "iNaturalist has structured nearby observations.",
        }],
        "assignments": [{"sub_need": "GPS/photo-verified local observations", "source": "iNaturalist"}],
        "source_policy": "Use iNaturalist for local proof.",
    }

    def gen_fn(**_kwargs):
        return {"text": json.dumps(payload)}

    implicit = plan_source_routing(TASK, SOURCES, gen_fn)
    explicit = plan_source_routing(TASK, SOURCES, gen_fn, max_depth=1)

    assert json.dumps(explicit, sort_keys=True) == json.dumps(implicit, sort_keys=True)
    assert "children" not in explicit["decision_tree"][0]
    assert "fallback_condition" not in explicit["decision_tree"][0]


def test_plan_source_routing_recursive_expansion_adds_children_and_fallback():
    calls = []

    def gen_fn(**kwargs):
        prompt = kwargs["prompt"]
        calls.append(prompt)
        if "planning source use" in prompt:
            return {"text": json.dumps({
                "task": TASK,
                "fail_open": False,
                "decision_tree": [{"sub_need": "GPS/photo-verified local observations", "preferred_sources": ["iNaturalist"], "fallback_sources": ["DuckDuckGo"], "rationale": "best fit"}],
                "source_policy": "Use iNaturalist.",
            })}
        return {"text": json.dumps({
            "sub_need": "GPS/photo-verified local observations",
            "preferred_sources": ["iNaturalist"],
            "fallback_condition": "if fewer than 5 GPS-tagged observations are returned",
            "fallback_sources": ["DuckDuckGo"],
            "rationale": "structured local observations first",
            "children": [{"sub_need": "photo-verified species IDs", "preferred_sources": ["iNaturalist"], "fallback_condition": "if IDs conflict", "fallback_sources": ["Wikipedia"], "rationale": "confirm taxonomy after local proof"}],
        })}

    plan = plan_source_routing(TASK, SOURCES, gen_fn, max_depth=2)

    node = plan["decision_tree"][0]
    assert plan["fail_open"] is False
    assert node["fallback_condition"] == "if fewer than 5 GPS-tagged observations are returned"
    assert node["children"][0]["sub_need"] == "photo-verified species IDs"
    assert len(calls) == 2


def test_plan_source_routing_node_count_cap_falls_back_to_flat_plan():
    def gen_fn(**kwargs):
        if "planning source use" in kwargs["prompt"]:
            return {"text": json.dumps({"task": TASK, "fail_open": False, "decision_tree": [{"sub_need": "GPS/photo-verified local observations", "preferred_sources": ["iNaturalist"], "rationale": "best fit"}], "source_policy": "Use iNaturalist."})}
        return {"text": json.dumps({"sub_need": "GPS/photo-verified local observations", "preferred_sources": ["iNaturalist"], "children": [{"sub_need": "child one", "preferred_sources": ["iNaturalist"]}, {"sub_need": "child two", "preferred_sources": ["DuckDuckGo"]}]})}

    plan = plan_source_routing(TASK, SOURCES, gen_fn, max_depth=2, max_nodes=2)

    assert plan["fail_open"] is True
    assert plan["decision_tree"] == [{"sub_need": "GPS/photo-verified local observations", "preferred_sources": ["iNaturalist"], "fallback_sources": [], "rationale": "best fit"}]


def test_supervise_findings_flags_bad_source_and_unsupported_count_and_clean_good_finding():
    plan = {"decision_tree": [{"sub_need": "GPS/photo-verified local observations", "preferred_sources": ["iNaturalist"], "fallback_sources": [], "rationale": "Only iNaturalist has local GPS/photo evidence."}]}
    findings = [
        {"source": "iNaturalist", "claim": "A. aurantia observed nearby.", "evidence": "photo yes, GPS 2.1 miles"},
        {"source": "Wikipedia", "claim": "A. aurantia observed nearby.", "evidence": "common in North America"},
        {"source": "DuckDuckGo", "claim": "Exactly 47 confirmed sightings.", "evidence": "blog mentions spiders generally"},
    ]

    def gen_fn(**_kwargs):
        return json.dumps({"fail_open": False, "verdicts": [
            {"finding_index": 0, "on_task": True, "supported": True, "source_appropriate": True, "guidance": "", "rationale": "matches plan and evidence"},
            {"finding_index": 1, "on_task": True, "supported": False, "source_appropriate": False, "guidance": "Redirect to iNaturalist GPS/photo observations.", "rationale": "Wikipedia cannot prove local observation"},
            {"finding_index": 2, "on_task": True, "supported": False, "source_appropriate": False, "guidance": "Remove exact count unless backed by observation records.", "rationale": "no count evidence"},
        ]})

    result = supervise_findings(TASK, findings, plan, gen_fn)

    assert result["fail_open"] is False
    assert result["verdicts"][0] == {"finding_index": 0, "on_task": True, "supported": True, "source_appropriate": True, "guidance": "", "rationale": "matches plan and evidence"}
    assert result["verdicts"][1]["source_appropriate"] is False
    assert result["verdicts"][1]["supported"] is False
    assert "iNaturalist" in result["verdicts"][1]["guidance"]
    assert result["verdicts"][2]["supported"] is False
    assert "exact count" in result["verdicts"][2]["guidance"]


def test_supervise_findings_fail_open_neutral_on_parse_failure():
    findings = [{"source": "DuckDuckGo", "claim": "something", "evidence": "snippet"}]

    def gen_fn(**_kwargs):
        return {"text": "nonsense"}

    result = supervise_findings(TASK, findings, {}, gen_fn)

    assert result["fail_open"] is True
    assert result["verdicts"] == [{"finding_index": 0, "on_task": True, "supported": True, "source_appropriate": True, "guidance": "", "rationale": result["verdicts"][0]["rationale"]}]


def test_supervise_and_correct_redispatches_only_flagged_findings():
    plan = {"decision_tree": [{"sub_need": "GPS/photo-verified local observations", "preferred_sources": ["iNaturalist"]}]}
    findings = [
        {"source": "iNaturalist", "sub_need": "GPS/photo-verified local observations", "claim": "supported", "evidence": "photo"},
        {"source": "Wikipedia", "sub_need": "GPS/photo-verified local observations", "claim": "local proof", "evidence": "background"},
    ]
    worker_calls = []

    def gen_fn(**kwargs):
        payload = json.loads(kwargs.get("prompt", "{}").split("Findings: ", 1)[1])
        verdicts = []
        for idx, finding in enumerate(payload):
            clean = finding["source"] == "iNaturalist"
            verdicts.append({"finding_index": idx, "on_task": True, "supported": clean, "source_appropriate": clean, "guidance": "" if clean else "Redirect to iNaturalist GPS/photo observations.", "rationale": "clean" if clean else "wrong source"})
        return json.dumps({"fail_open": False, "verdicts": verdicts})

    def worker_fn(**kwargs):
        worker_calls.append(kwargs["finding"])
        return {"source": "iNaturalist", "sub_need": kwargs["finding"]["sub_need"], "claim": "local proof corrected", "evidence": "photo yes, GPS 2.1 miles"}

    result = supervise_and_correct(TASK, findings, plan, gen_fn, worker_fn, max_rounds=2)

    assert result["converged"] is True
    assert worker_calls == [findings[1]]
    assert result["findings"][0] == findings[0]
    assert result["findings"][1]["source"] == "iNaturalist"
    assert result["audit_trail"][1]["rounds"][0]["corrected_finding"]["source"] == "iNaturalist"


def test_supervise_and_correct_marks_unconverged_at_max_rounds():
    plan = {"decision_tree": [{"sub_need": "GPS/photo-verified local observations", "preferred_sources": ["iNaturalist"]}]}
    findings = [{"source": "Wikipedia", "sub_need": "GPS/photo-verified local observations", "claim": "local proof", "evidence": "background"}]

    def gen_fn(**_kwargs):
        return json.dumps({"fail_open": False, "verdicts": [{"finding_index": 0, "on_task": True, "supported": False, "source_appropriate": False, "guidance": "Still wrong; use iNaturalist.", "rationale": "wrong source"}]})

    def worker_fn(**kwargs):
        return dict(kwargs["finding"])

    result = supervise_and_correct(TASK, findings, plan, gen_fn, worker_fn, max_rounds=1)

    assert result["converged"] is False
    assert result["audit_trail"][0]["unconverged"] is True
    assert len(result["audit_trail"][0]["rounds"]) == 2


def test_supervise_and_correct_worker_exception_fails_open():
    plan = {"decision_tree": [{"sub_need": "GPS/photo-verified local observations", "preferred_sources": ["iNaturalist"]}]}
    findings = [{"source": "Wikipedia", "sub_need": "GPS/photo-verified local observations", "claim": "local proof", "evidence": "background"}]

    def gen_fn(**_kwargs):
        return json.dumps({"fail_open": False, "verdicts": [{"finding_index": 0, "on_task": True, "supported": False, "source_appropriate": False, "guidance": "Use iNaturalist.", "rationale": "wrong source"}]})

    def worker_fn(**_kwargs):
        raise RuntimeError("worker unavailable")

    result = supervise_and_correct(TASK, findings, plan, gen_fn, worker_fn)

    assert result["fail_open"] is True
    assert result["findings"] == findings
    assert "worker unavailable" in result["error"]


def test_supervise_findings_evidence_fn_none_preserves_current_behavior_and_fake_can_flip():
    finding = {"source": "iNaturalist", "claim": "observed nearby", "evidence": "photo"}

    def gen_fn(**_kwargs):
        return json.dumps({"fail_open": False, "verdicts": [{"finding_index": 0, "on_task": True, "supported": True, "source_appropriate": True, "guidance": "", "rationale": "model says supported"}]})

    baseline = supervise_findings(TASK, [finding], {}, gen_fn)
    grounded = supervise_findings(TASK, [finding], {}, gen_fn, evidence_fn=lambda **_kwargs: {"supported": False, "rationale": "retrieved evidence does not contain the claim"})

    assert baseline == supervise_findings(TASK, [finding], {}, gen_fn, evidence_fn=None)
    assert baseline["verdicts"][0]["supported"] is True
    assert grounded["verdicts"][0]["supported"] is False
    assert "retrieved evidence" in grounded["verdicts"][0]["rationale"]
