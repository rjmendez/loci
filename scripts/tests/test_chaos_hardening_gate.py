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


def test_chaos_gate_passes_with_healthy_metrics(tmp_path, capsys):
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
    out = json.loads(capsys.readouterr().out)
    # Every gate was measured and passed: a gate that could not read its fields
    # is "skipped", which also exits 0, so the statuses are the claim here.
    assert out["ok"] is True
    assert out["summary"]["passed"] == 5
    assert out["summary"]["skipped"] == 0
    assert out["summary"]["failed"] == 0
    observed = {g["name"]: g["observed"] for g in out["gates"]}
    assert observed["timeouts"] == {"timeouts": 0, "measured": 2, "timeout_rate": 0.0}
    assert observed["retries"] == {"retried_records": 2, "recovered_after_retry": 2, "recovery_rate": 1.0}
    assert observed["duplicate_effects"] == {"duplicate_effects": 0, "measured": 3, "duplicate_effect_rate": 0.0}
    assert observed["provenance_completeness"] == {"complete": 3, "measured": 3, "completeness_rate": 1.0}
    assert observed["adversarial_candidate_bypass"] == {"candidate_bypass_count": 0}


def _gate_for_report(mod, tmp_path, capsys, report, *extra):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    rc = mod.main(["--adversarial-report", str(report_path), *extra])
    out = json.loads(capsys.readouterr().out)
    (gate,) = out["gates"]
    return rc, gate


def test_report_without_any_bypass_count_is_not_a_pass(tmp_path, capsys):
    # A summary with no candidate_bypass_count, no classification_counts and no
    # findings list says nothing about bypasses. It used to be read as 0 -> pass.
    mod = _load()
    rc, gate = _gate_for_report(mod, tmp_path, capsys, {"summary": {"total_probes": 12}})
    assert gate["status"] == "skipped"
    assert gate["observed"] == {}
    assert rc == 0

    rc, gate = _gate_for_report(mod, tmp_path, capsys, {"summary": {"total_probes": 12}}, "--fail-on-skipped")
    assert gate["status"] == "skipped"
    assert rc == 1


def test_report_counted_from_findings_list(tmp_path, capsys):
    mod = _load()
    report = {"summary": {}, "findings": [
        {"classification": "candidate_bypass"}, {"classification": "blocked"}, {"classification": " Candidate_Bypass "},
    ]}
    rc, gate = _gate_for_report(mod, tmp_path, capsys, report, "--max-candidate-bypass", "2")
    assert (rc, gate["status"], gate["observed"]) == (0, "pass", {"candidate_bypass_count": 2})

    report = {"summary": {}, "findings": []}
    rc, gate = _gate_for_report(mod, tmp_path, capsys, report, "--fail-on-skipped")
    assert (rc, gate["status"], gate["observed"]) == (0, "pass", {"candidate_bypass_count": 0})


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
