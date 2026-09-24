import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_brain_cluster_release_prep as prep  # noqa: E402


def _report(*, accuracy: float, abstain: float, calibration: float, decision: float,
            dataset: str | None = None, objective: str | None = None,
            baseline_pass: bool | None = None) -> dict[str, object]:
    report: dict[str, object] = {
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
    notes: dict[str, object] = {}
    if dataset is not None:
        notes["dataset_symbol"] = dataset
    if objective is not None:
        notes["objective"] = objective
    if notes:
        report["dataset_manifest"] = {"notes": notes}
    if baseline_pass is not None:
        report["trivial_baseline"] = {
            "pass": baseline_pass,
            "model_heldout_accuracy": 0.9,
            "best_trivial_rule": "majority",
            "best_trivial_accuracy": 0.7 if baseline_pass else 0.95,
            "required_accuracy": 0.71 if baseline_pass else 0.96,
            "heldout_count": 30,
        }
    return report


def _write_run(run_root, name: str, report: dict[str, object], *, sample_metadata: dict | None = None) -> None:
    directory = run_root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "p0-report.json").write_text(json.dumps(report), encoding="utf-8")
    if sample_metadata is not None:
        payload = {"samples": [], "metadata": sample_metadata}
        (directory / f"{name}-samples.json").write_text(json.dumps(payload), encoding="utf-8")


def _good(**kwargs):
    return _report(accuracy=0.9, abstain=0.05, calibration=0.08, decision=0.96, **kwargs)


def _calibrate(tmp_path, min_reports: int = 2):
    return prep.calibrate_thresholds_from_runs(
        prep.ReleasePrepConfig(runs_root=tmp_path / "runs", output_dir=tmp_path / "out",
                               min_reports_per_objective=min_reports)
    )


def test_groups_by_dataset_and_objective(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs, "fw-a", _good(dataset="fw", objective="connectivity_tier", baseline_pass=True))
    _write_run(runs, "fw-b", _good(dataset="fw", objective="connectivity_tier", baseline_pass=False))
    _write_run(runs, "fw-nt-a", _good(), sample_metadata={"schema_version": "flybrain-fw-training-samples/v1",
                                                          "objective": "neurotransmitter_dominance"})
    _write_run(runs, "fw-nt-b", _good(), sample_metadata={"dataset_symbol": "fw",
                                                          "objective": "neurotransmitter_dominance"})

    summary = _calibrate(tmp_path)

    assert summary["schema_version"] == prep.RELEASE_PREP_SCHEMA_VERSION
    assert sorted(summary["groups_calibrated"]) == ["fw/connectivity_tier", "fw/neurotransmitter_dominance"]
    row = summary["groups_calibrated"]["fw/connectivity_tier"]
    assert row["dataset_symbol"] == "fw" and row["objective"] == "connectivity_tier"
    assert row["trivial_baseline_pass_count"] == 1 and row["trivial_baseline_all_passed"] is False
    path = tmp_path / "out" / "braincluster-thresholds-fw-connectivity_tier.json"
    assert row["threshold_file"] == str(path.resolve())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "braincluster-threshold-calibration/v1"
    assert payload["dataset_symbol"] == "fw" and payload["objective"] == "connectivity_tier"
    assert payload["trivial_baseline"]["recorded_count"] == 2
    assert {run["best_trivial_accuracy"] for run in payload["trivial_baseline"]["runs"]} == {0.7, 0.95}


def test_other_dataset_does_not_count_toward_fw_minimum(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs, "fw-a", _good(dataset="fw", objective="connectivity_tier"))
    _write_run(runs, "hb-a", _good(dataset="hb", objective="connectivity_tier"))
    _write_run(runs, "hb-b", _good(dataset="hb", objective="connectivity_tier"))

    summary = _calibrate(tmp_path)

    assert list(summary["groups_calibrated"]) == ["hb/connectivity_tier"]
    skipped = summary["groups_skipped"]["fw/connectivity_tier"]
    assert skipped["reason"] == "insufficient_reports" and skipped["report_count"] == 1
    assert not (tmp_path / "out" / "braincluster-thresholds-fw-connectivity_tier.json").exists()


@pytest.mark.parametrize(
    "dataset,objective",
    [(None, "connectivity_tier"), ("unknown", "connectivity_tier"), ("mouse", "connectivity_tier"),
     ("fw", "custom"), ("fw", None)],
)
def test_unresolved_dataset_or_objective_fails_closed(tmp_path, dataset, objective):
    runs = tmp_path / "runs"
    for name in ("a", "b", "c"):
        _write_run(runs, name, _good(dataset=dataset, objective=objective))
    with pytest.raises(ValueError, match="resolved \\(dataset, objective\\)"):
        _calibrate(tmp_path)


def test_unresolved_runs_are_listed_not_pooled(tmp_path):
    runs = tmp_path / "runs"
    _write_run(runs, "fw-a", _good(dataset="fw", objective="connectivity_tier"))
    _write_run(runs, "fw-b", _good(dataset="fw", objective="connectivity_tier"))
    _write_run(runs, "anon", _good(objective="connectivity_tier"))
    summary = _calibrate(tmp_path)
    assert summary["groups_calibrated"]["fw/connectivity_tier"]["report_count"] == 2
    assert [row["dataset_symbol"] for row in summary["unresolved_runs"]] == ["unknown"]


def test_rejects_when_insufficient_reports(tmp_path):
    _write_run(tmp_path / "runs", "run-a", _good(dataset="fw", objective="connectivity_tier"))
    with pytest.raises(ValueError, match="minimum report count"):
        _calibrate(tmp_path)


def test_cli_error_payload_uses_v2_schema(tmp_path, capsys):
    code = prep.main(["--runs-root", str(tmp_path / "missing"), "--output-dir", str(tmp_path / "out")])
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == prep.RELEASE_PREP_SCHEMA_VERSION and payload["status"] == "error"
