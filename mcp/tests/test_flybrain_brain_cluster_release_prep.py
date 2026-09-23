import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_brain_cluster_release_prep as prep  # noqa: E402


def _report(*, accuracy: float, abstain: float, calibration: float, decision: float) -> dict[str, object]:
    return {
        "gate_report": {
            "metrics": {
                "sample_count": 120,
                "accuracy": accuracy,
                "abstain_rate": abstain,
                "mean_calibration_error": calibration,
            }
        },
        "shadow_report": {
            "metrics": {
                "decision_match_rate": decision,
                "confidence_drift": {"mean_abs_delta": 0.03},
                "fail_closed_rate": {"delta": 0.01},
                "routing_entropy": {"mean_abs_delta": 0.02},
                "expert_collapse": {
                    "candidate": {"concentration": 0.45},
                    "concentration_delta": 0.03,
                },
            }
        },
    }


def _write_run(run_root, name: str, objective: str, report: dict[str, object]) -> None:
    directory = run_root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "p0-report.json").write_text(json.dumps(report), encoding="utf-8")
    sample_payload = {
        "samples": [],
        "metadata": {"objective": objective},
    }
    (directory / f"{name}-samples.json").write_text(json.dumps(sample_payload), encoding="utf-8")


def test_calibrate_thresholds_from_runs_groups_by_objective(tmp_path):
    runs = tmp_path / "runs"
    out = tmp_path / "out"
    _write_run(runs, "run-a", "connectivity_tier", _report(accuracy=0.91, abstain=0.05, calibration=0.08, decision=0.97))
    _write_run(runs, "run-b", "connectivity_tier", _report(accuracy=0.89, abstain=0.06, calibration=0.09, decision=0.96))
    _write_run(runs, "run-c", "neurotransmitter_dominance", _report(accuracy=0.86, abstain=0.07, calibration=0.12, decision=0.94))
    _write_run(runs, "run-d", "neurotransmitter_dominance", _report(accuracy=0.87, abstain=0.06, calibration=0.11, decision=0.95))

    summary = prep.calibrate_thresholds_from_runs(
        prep.ReleasePrepConfig(
            runs_root=runs,
            output_dir=out,
            min_reports_per_objective=2,
        )
    )

    assert summary["status"] == "ok"
    objectives = summary["objectives_calibrated"]
    assert sorted(objectives) == ["connectivity_tier", "neurotransmitter_dominance"]
    for objective, row in objectives.items():
        threshold_path = row["threshold_file"]
        assert os.path.exists(threshold_path)
        payload = json.loads((out / f"braincluster-thresholds-{objective}.json").read_text(encoding="utf-8"))
        assert payload["schema_version"] == "braincluster-threshold-calibration/v1"


def test_calibrate_thresholds_from_runs_rejects_when_insufficient_reports(tmp_path):
    runs = tmp_path / "runs"
    out = tmp_path / "out"
    _write_run(runs, "run-a", "connectivity_tier", _report(accuracy=0.91, abstain=0.05, calibration=0.08, decision=0.97))

    try:
        prep.calibrate_thresholds_from_runs(
            prep.ReleasePrepConfig(
                runs_root=runs,
                output_dir=out,
                min_reports_per_objective=2,
            )
        )
    except ValueError as exc:
        assert "minimum report count" in str(exc)
        return
    raise AssertionError("Expected ValueError when no objective reaches minimum reports.")
