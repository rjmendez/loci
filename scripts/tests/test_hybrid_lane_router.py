from __future__ import annotations

import importlib.util
import json
import pathlib
import sys


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "hybrid_lane_router.py"


def _load():
    spec = importlib.util.spec_from_file_location("hybrid_lane_router", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _base_request() -> dict[str, object]:
    return {
        "task_id": "task-1",
        "session_id": "session-1",
        "priority": "P1",
        "critical_path": False,
        "task_class": "reasoning",
        "idempotency_key": "1234567890abcdef",
        "max_latency_ms": 1500,
        "max_cost_usd": 0.2,
        "quality_floor": 0.7,
        "evidence_required": "standard",
    }


def test_validate_request_flags_missing_fields():
    mod = _load()
    errors = mod.validate_dispatch_request({"task_id": "t"})
    assert "missing:session_id" in errors
    assert "missing:task_class" in errors


def test_critical_path_routes_to_reserved_slot():
    mod = _load()
    req = _base_request()
    req["critical_path"] = True
    decision = mod.choose_lane(req)
    assert decision.lane_id == "copilot-critical"
    assert decision.reason == "critical_path"


def test_speculative_requires_confidence():
    mod = _load()
    req = _base_request()
    req["task_class"] = "speculative"

    low = mod.choose_lane(req, upstream_confidence=0.5)
    high = mod.choose_lane(req, upstream_confidence=0.9)

    assert low.lane_id == "local-batch"
    assert high.lane_id == "local-speculative"


def test_brownout_l3_forces_local_for_reasoning():
    mod = _load()
    req = _base_request()
    req["task_class"] = "reasoning"
    decision = mod.choose_lane(req, brownout_level="L3")
    assert decision.lane_id == "local-batch"
    assert decision.degraded is True


def test_main_emits_json_decision(tmp_path, capsys):
    mod = _load()
    path = tmp_path / "req.json"
    path.write_text(json.dumps(_base_request()), encoding="utf-8")
    rc = mod.main(["--request-json", str(path)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["lane_id"] == "local-batch"
