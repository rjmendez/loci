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
    assert first["calibration_id"] == "braincluster-thresholds-" + first["input_fingerprint"][:16]


def test_calibration_fingerprint_changes_with_params_and_rows():
    base = fbthr.calibrate_thresholds_from_reports(_reports())
    other_margin = fbthr.calibrate_thresholds_from_reports(_reports(), safety_margin=0.03)
    fewer_rows = fbthr.calibrate_thresholds_from_reports(_reports()[:9])
    assert len({base["input_fingerprint"], other_margin["input_fingerprint"], fewer_rows["input_fingerprint"]}) == 3


# Hand-computed for _reports(): each metric is linear in i = 1..10 (v = a + b*i),
# and _quantile(p=0.2 / 0.8) takes the inclusive 20th / 80th percentile, which
# for ten evenly spaced points sits at i = 2.8 / 8.2. So with a 0.01 margin:
#   min_accuracy       = (0.84 + 0.01*2.8)  - 0.01 = 0.858
#   max_abstain_rate   = (0.02 + 0.005*8.2) + 0.01 = 0.071
#   max_calib_error    = (0.05 + 0.01*8.2)  + 0.01 = 0.142
#   min_samples        = round(40 + 2.8)           = 43
#   min_decision_match = (0.90 + 0.009*2.8) - 0.01 = 0.9152
#   drift / fail-closed / entropy / concentration / concentration delta at 8.2:
#     0.02+0.004*8.2+0.01, 0.006*8.2+0.01, 0.01+0.003*8.2+0.01,
#     0.40+0.03*8.2+0.01, 0.02+0.01*8.2+0.01
EXPECTED_GATE = {
    "min_accuracy": 0.858,
    "max_abstain_rate": 0.071,
    "max_confidence_calibration_error": 0.142,
    "min_samples": 43,
}
EXPECTED_SHADOW = {
    "min_decision_match_rate": 0.9152,
    "max_mean_abs_confidence_drift": 0.0628,
    "max_fail_closed_rate_delta": 0.0592,
    "max_mean_abs_routing_entropy_delta": 0.0446,
    "max_selected_expert_concentration": 0.656,
    "max_selected_expert_concentration_delta": 0.112,
}


def test_calibration_thresholds_have_expected_shape_and_ranges():
    result = fbthr.calibrate_thresholds_from_reports(_reports(), lower_quantile=0.2, upper_quantile=0.8, safety_margin=0.01)
    gate = result["thresholds"]["gate"]
    shadow = result["thresholds"]["shadow_replay"]

    assert set(gate) >= set(EXPECTED_GATE)
    assert set(shadow) >= set(EXPECTED_SHADOW)
    for key, value in EXPECTED_GATE.items():
        assert gate[key] == pytest.approx(value, abs=1e-9), key
    for key, value in EXPECTED_SHADOW.items():
        assert shadow[key] == pytest.approx(value, abs=1e-9), key


def test_calibration_clamps_rates_into_the_unit_interval():
    # A margin larger than the headroom must clamp, not produce a rate above 1
    # or a minimum below 0.
    result = fbthr.calibrate_thresholds_from_reports(_reports(), lower_quantile=0.2, upper_quantile=0.8, safety_margin=0.95)
    gate = result["thresholds"]["gate"]
    shadow = result["thresholds"]["shadow_replay"]
    assert gate["min_accuracy"] == 0.0
    assert gate["max_abstain_rate"] == pytest.approx(1.0)
    assert shadow["min_decision_match_rate"] == 0.0
    assert shadow["max_selected_expert_concentration"] == pytest.approx(1.0)
    assert shadow["max_mean_abs_confidence_drift"] == pytest.approx(0.0328 + 0.02 + 0.95)


def test_calibration_rejects_malformed_reports():
    with pytest.raises(ValueError, match="report payload"):
        fbthr.calibrate_thresholds_from_reports({"unexpected": []})


def test_calibration_rejects_invalid_quantiles():
    with pytest.raises(ValueError, match="quantiles must satisfy"):
        fbthr.calibrate_thresholds_from_reports(_reports(), lower_quantile=0.9, upper_quantile=0.8)
