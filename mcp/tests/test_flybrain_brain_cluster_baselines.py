import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_brain_cluster_baselines as fbb  # noqa: E402
import flybrain_brain_cluster_pipeline as fbcp  # noqa: E402


def _fw_connectivity_rows(n: int = 60) -> list[tuple[str, str]]:
    """fw-style tautological rows: the label is a threshold on total_pre_count, which is in the input."""
    rows = []
    for i in range(n):
        total = 5 + (i * 37) % 400
        label = "high_connectivity" if total >= 250 else "baseline_connectivity"
        rows.append((f"dataset flywire783 pre_root {1000 + i} dominant_neuropil ME_R total_pre_count {total} "
                     f"dominant_count {total // 2} dominance_ratio 0.500000", label))
    return rows


def _fw_nt_rows(n: int = 60) -> list[tuple[str, str]]:
    rows = []
    names = ("ach", "gaba", "glut")
    for i in range(n):
        top = names[i % 3]
        scores = {name: (0.7 if name == top else 0.1) for name in names}
        text = "dataset flywire783 " + " ".join(f"{name}_avg {scores[name]:.6f}" for name in names)
        rows.append((text, f"dominant_{top}"))
    return rows


def test_parse_input_features_pairs_tokens():
    assert fbb.parse_input_features("dataset fw total 12 ratio 0.5") == {"dataset": "fw", "total": "12", "ratio": "0.5"}


def test_threshold_rule_recovers_tautological_label():
    rows = _fw_connectivity_rows()
    rule = fbb.fit_threshold_rule(rows[:40])
    assert rule is not None and rule["feature"] in {"total_pre_count", "dominant_count"}
    report = fbb.evaluate_trivial_baselines(rows[:40], rows[40:])
    assert report["best_rule"] == fbb.RULE_THRESHOLD
    assert report["best_accuracy"] >= 0.95


def test_argmax_rule_for_neurotransmitter_scores():
    rows = _fw_nt_rows()
    rule = fbb.fit_argmax_rule(rows[:40])
    assert rule == {"label_to_feature": {"dominant_ach": "ach_avg", "dominant_gaba": "gaba_avg",
                                         "dominant_glut": "glut_avg"}}
    report = fbb.evaluate_trivial_baselines(rows[:40], rows[40:])
    assert report["rules"][fbb.RULE_ARGMAX]["heldout_accuracy"] == pytest.approx(1.0)


def test_majority_only_when_no_numeric_or_score_features():
    train = [("color red", "a")] * 6 + [("color blue", "b")] * 2
    held = [("color red", "a"), ("color blue", "b")]
    report = fbb.evaluate_trivial_baselines(train, held)
    assert report["rules"][fbb.RULE_THRESHOLD]["applicable"] is False
    assert report["rules"][fbb.RULE_ARGMAX]["applicable"] is False
    assert report["best_rule"] == fbb.RULE_MAJORITY
    assert report["best_accuracy"] == pytest.approx(0.5)


def test_gate_requires_margin_over_best_trivial_rule():
    baselines = {"best_rule": "majority", "best_accuracy": 0.8}
    ok = fbb.trivial_baseline_gate(model_heldout_accuracy=0.9, baselines=baselines, heldout_count=10)
    assert ok["pass"] is True and ok["required_accuracy"] == pytest.approx(0.81)
    tie = fbb.trivial_baseline_gate(model_heldout_accuracy=0.8, baselines=baselines, heldout_count=10)
    assert tie["pass"] is False and "does not beat the trivial baseline" in tie["failure_reasons"][0]
    strict = fbb.trivial_baseline_gate(model_heldout_accuracy=0.9, baselines=baselines, heldout_count=10,
                                       config={"min_margin": 0.2})
    assert strict["pass"] is False
    empty = fbb.trivial_baseline_gate(model_heldout_accuracy=None, baselines=None, heldout_count=0)
    assert empty["pass"] is False and len(empty["failure_reasons"]) == 2


def test_gate_config_validation():
    with pytest.raises(ValueError):
        fbb.BaselineGateConfig(min_margin=-0.1)
    with pytest.raises(ValueError, match="unknown baseline gate keys"):
        fbb.BaselineGateConfig.from_value({"margin": 0.1})


# ---------------------------------------------------------------------------
# Pipeline integration (AC6 gate, dataset notes, grouped split)
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, payload) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _run(tmp_path: Path, payload, **kwargs):
    return fbcp.run_brain_cluster_p0_dry_run(
        _write(tmp_path, payload),
        output_dir=tmp_path / "out",
        state_path=tmp_path / "state" / "promotion-state.json",
        split_seed=kwargs.pop("split_seed", "seed-e2e"),
        **kwargs,
    )


def _fw_payload(n: int = 80) -> dict:
    samples = []
    for i, (text, label) in enumerate(_fw_connectivity_rows(n)):
        samples.append({
            "sample_id": f"fw783-pre-{1000 + i}",
            "region_id": "me_r" if i % 2 else "lo_r",
            "input_text": text,
            "expected_label": label,
            "expected_confidence": 0.9,
            "provenance_refs": [f"flywire783:pre_pt_root_id:{1000 + i}"],
            "metadata": {"dataset": "flywire", "dataset_version": "flywire783", "task_type": "connectivity"},
        })
    return {"samples": samples, "metadata": {"schema_version": "flybrain-fw-training-samples/v1",
                                             "objective": "connectivity_tier"}}


def test_tautological_fw_connectivity_fails_trivial_baseline_gate(tmp_path):
    report = _run(tmp_path, _fw_payload())
    baseline = report["trivial_baseline"]
    assert baseline["best_trivial_rule"] == fbb.RULE_THRESHOLD
    assert baseline["best_trivial_accuracy"] >= 0.95
    assert baseline["pass"] is False
    assert report["pass_gate"] is False and report["promoted"] is False
    assert report["gate_report"]["trivial_baseline"]["pass"] is False
    assert any("trivial baseline" in reason for reason in report["gate_report"]["failure_reasons"])
    written = json.loads((tmp_path / "out" / "trivial-baseline-report.json").read_text(encoding="utf-8"))
    assert written["best_trivial_accuracy"] == baseline["best_trivial_accuracy"]
    notes = report["dataset_manifest"]["notes"]
    assert notes["dataset_symbol"] == "fw" and notes["dataset_version"] == "flywire783"
    assert notes["objective"] == "connectivity_tier"
    assert report["dataset"]["source"] == "payload_schema_version"


def test_model_without_signal_is_not_promoted(tmp_path):
    from test_flybrain_brain_cluster_pipeline import _sample_payload

    report = _run(tmp_path, _sample_payload(with_cue=False))
    assert report["trivial_baseline"]["best_trivial_rule"] == fbb.RULE_MAJORITY
    assert report["trivial_baseline"]["pass"] is False
    assert report["promoted"] is False


def test_baseline_margin_is_configurable(tmp_path):
    from test_flybrain_brain_cluster_pipeline import _sample_payload

    report = _run(tmp_path, _sample_payload(), baseline_gate={"min_margin": 0.5})
    assert report["trivial_baseline"]["config"]["min_margin"] == 0.5
    assert report["trivial_baseline"]["pass"] is False
    assert report["promoted"] is False


def test_pipeline_split_keeps_groups_together(tmp_path):
    from test_flybrain_brain_cluster_pipeline import _sample_payload

    payload = _sample_payload()
    for index, sample in enumerate(payload["samples"]):
        sample["metadata"]["split_group"] = f"pair-{index // 2}"
    report = _run(tmp_path, payload)
    manifest = report["dataset_manifest"]
    split_of = {sid: name for name in ("train_ids", "val_ids", "test_ids") for sid in manifest[name]}
    for index in range(0, 40, 2):
        assert split_of[f"s{index + 1:03d}"] == split_of[f"s{index + 2:03d}"]
    split = manifest["notes"]["split"]
    assert split["strategy"] == "grouped"
    assert split["group_keys"] == ["split_group"]
    assert split["multi_sample_components"] == 20
    assert manifest["notes"]["dataset_symbol"] == "unknown"


def test_pipeline_rejects_mixed_datasets(tmp_path):
    payload = _fw_payload(40)
    payload["metadata"] = {}
    payload["samples"][0]["metadata"]["dataset"] = "banc"
    with pytest.raises(ValueError, match="mix datasets"):
        _run(tmp_path, payload)


def test_pipeline_rejects_version_off_pin(tmp_path):
    payload = _fw_payload(40)
    for sample in payload["samples"]:
        sample["metadata"]["dataset_version"] = "flywire630"
    with pytest.raises(ValueError, match="registry pin"):
        _run(tmp_path, payload)
