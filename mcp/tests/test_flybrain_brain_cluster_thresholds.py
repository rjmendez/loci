import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_brain_cluster_thresholds as fbthr  # noqa: E402


def _reports() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i in range(1, 11):
        rows.append(
            {
                "gate_report": {
                    "metrics": {
                        "sample_count": 40 + i,
                        "accuracy": 0.84 + (i * 0.01),
                        "abstain_rate": 0.02 + (i * 0.005),
                        "mean_calibration_error": 0.05 + (i * 0.01),
                    },
                },
                "shadow_report": {
                    "metrics": {
                        "decision_match_rate": 0.90 + (i * 0.009),
                        "confidence_drift": {"mean_abs_delta": 0.02 + (i * 0.004)},
                        "fail_closed_rate": {"delta": 0.00 + (i * 0.006)},
                        "routing_entropy": {"mean_abs_delta": 0.01 + (i * 0.003)},
                        "expert_collapse": {
                            "candidate": {"concentration": 0.40 + (i * 0.03)},
                            "concentration_delta": 0.02 + (i * 0.01),
                        },
                    },
                },
            }
        )
    return rows


def test_calibration_is_deterministic_for_same_input():
    first = fbthr.calibrate_thresholds_from_reports(_reports())
    second = fbthr.calibrate_thresholds_from_reports(_reports())
    assert first == second
    assert first["schema_version"] == "braincluster-threshold-calibration/v1"
    assert first["input_report_count"] == 10


def test_calibration_thresholds_have_expected_shape_and_ranges():
    result = fbthr.calibrate_thresholds_from_reports(_reports(), lower_quantile=0.2, upper_quantile=0.8, safety_margin=0.01)
    gate = result["thresholds"]["gate"]
    shadow = result["thresholds"]["shadow_replay"]

    assert 0.0 <= gate["min_accuracy"] <= 1.0
    assert 0.0 <= gate["max_abstain_rate"] <= 1.0
    assert 0.0 <= gate["max_confidence_calibration_error"] <= 1.0
    assert gate["min_samples"] > 0

    assert 0.0 <= shadow["min_decision_match_rate"] <= 1.0
    assert shadow["max_mean_abs_confidence_drift"] >= 0.0
    assert shadow["max_fail_closed_rate_delta"] >= 0.0
    assert shadow["max_mean_abs_routing_entropy_delta"] >= 0.0
    assert 0.0 <= shadow["max_selected_expert_concentration"] <= 1.0
    assert shadow["max_selected_expert_concentration_delta"] >= 0.0


def test_calibration_rejects_malformed_reports():
    with pytest.raises(ValueError, match="report payload"):
        fbthr.calibrate_thresholds_from_reports({"unexpected": []})


def test_calibration_rejects_invalid_quantiles():
    with pytest.raises(ValueError, match="quantiles must satisfy"):
        fbthr.calibrate_thresholds_from_reports(_reports(), lower_quantile=0.9, upper_quantile=0.8)
