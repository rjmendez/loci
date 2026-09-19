from __future__ import annotations

import importlib.util
import json
import pathlib
import sys


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "chaos_hardening_gate.py"


def _load():
    spec = importlib.util.spec_from_file_location("chaos_hardening_gate", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_chaos_gate_passes_with_healthy_metrics(tmp_path):
    mod = _load()
    events = [
        {"timed_out": False, "retry_count": 1, "success": True, "duplicate_effect": False, "provenance_complete": True},
        {"timed_out": False, "retry_count": 0, "success": True, "duplicate_effect": False, "provenance_complete": True},
        {"status": "ok", "attempts": 2, "duplicate_effect": False, "evidence_provenance_tier": "tool_verified"},
    ]
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in events), encoding="utf-8")

    report = {"summary": {"candidate_bypass_count": 0}}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    rc = mod.main(["--chaos-events", str(path), "--adversarial-report", str(report_path)])
    assert rc == 0


def test_chaos_gate_fails_when_thresholds_are_breached(tmp_path, capsys):
    mod = _load()
    events = [
        {"timed_out": True, "retry_count": 2, "success": False, "duplicate_effect": True, "provenance_complete": False},
        {"timed_out": False, "retry_count": 1, "success": False, "duplicate_effect": False, "provenance_complete": False},
    ]
    path = tmp_path / "events.json"
    path.write_text(json.dumps(events), encoding="utf-8")

    report = {"summary": {"classification_counts": {"candidate_bypass": 2}}}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    rc = mod.main([
        "--chaos-events",
        str(path),
        "--adversarial-report",
        str(report_path),
        "--max-candidate-bypass",
        "0",
    ])
    assert rc == 1
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False
    failed = set(out["summary"]["failed_gates"])
    assert "timeouts" in failed
    assert "retries" in failed
    assert "duplicate_effects" in failed
    assert "provenance_completeness" in failed
    assert "adversarial_candidate_bypass" in failed


def test_fail_on_skipped_turns_unknown_metrics_into_failure(tmp_path):
    mod = _load()
    events = [{"event": "noop"}]
    path = tmp_path / "events.json"
    path.write_text(json.dumps(events), encoding="utf-8")

    rc = mod.main(["--chaos-events", str(path), "--fail-on-skipped"])
    assert rc == 1


def test_main_requires_an_input():
    mod = _load()
    assert mod.main([]) == 2
