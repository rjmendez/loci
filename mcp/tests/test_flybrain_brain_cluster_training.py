import os
import sys
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_brain_cluster_training as fbct  # noqa: E402


def _samples() -> list[fbct.TrainingSample]:
    return [
        fbct.TrainingSample(
            sample_id=f"s{i}",
            region_id="grounding" if i % 2 == 0 else "provenance",
            input_text=f"input {i}",
            expected_label="accept" if i % 3 else "reject",
            expected_confidence=0.8 if i % 2 == 0 else 0.6,
            provenance_refs=(f"finding-{i}",),
        )
        for i in range(1, 21)
    ]


def test_manifest_build_is_deterministic():
    samples = _samples()
    first = fbct.build_dataset_manifest(samples, split_seed="seed-1")
    second = fbct.build_dataset_manifest(samples, split_seed="seed-1")
    assert first.manifest_id == second.manifest_id
    assert first.samples_sha256 == second.samples_sha256
    assert first.train_ids == second.train_ids
    assert first.val_ids == second.val_ids
    assert first.test_ids == second.test_ids


def test_manifest_rejects_duplicate_sample_id():
    samples = _samples()
    dup = fbct.TrainingSample(
        sample_id=samples[0].sample_id,
        region_id="grounding",
        input_text="dup",
        expected_label="accept",
        expected_confidence=0.5,
        provenance_refs=("finding-dup",),
    )
    with pytest.raises(ValueError, match="duplicate sample_id"):
        fbct.build_dataset_manifest([*samples, dup], split_seed="seed-2")


def test_golden_set_limits_per_region():
    golden = fbct.build_golden_set(_samples(), max_per_region=3)
    grounding = [s for s in golden if s.region_id == "grounding"]
    provenance = [s for s in golden if s.region_id == "provenance"]
    assert len(grounding) == 3
    assert len(provenance) == 3


def test_evaluate_predictions_passes_thresholds():
    golden = fbct.build_golden_set(_samples(), max_per_region=10)
    predictions = [
        fbct.PredictionRecord(
            sample_id=s.sample_id,
            predicted_label=s.expected_label,
            predicted_confidence=s.expected_confidence,
            abstained=False,
        )
        for s in golden
    ]
    summary = fbct.evaluate_predictions(
        golden,
        predictions,
        thresholds=fbct.PromotionGateThresholds(
            min_accuracy=0.8,
            max_abstain_rate=0.2,
            max_confidence_calibration_error=0.1,
            min_samples=10,
        ),
    )
    assert summary.pass_promotion_gate is True
    assert summary.failures == ()


def test_evaluate_predictions_fails_on_abstain_and_accuracy():
    golden = fbct.build_golden_set(_samples(), max_per_region=5)
    predictions = []
    for idx, sample in enumerate(golden):
        if idx < 3:
            predictions.append(
                fbct.PredictionRecord(
                    sample_id=sample.sample_id,
                    predicted_label="wrong",
                    predicted_confidence=0.2,
                    abstained=False,
                )
            )
            continue
        predictions.append(
            fbct.PredictionRecord(
                sample_id=sample.sample_id,
                predicted_label=sample.expected_label,
                predicted_confidence=0.4,
                abstained=True,
            )
        )

    summary = fbct.evaluate_predictions(
        golden,
        predictions,
        thresholds=fbct.PromotionGateThresholds(
            min_accuracy=0.7,
            max_abstain_rate=0.1,
            max_confidence_calibration_error=0.2,
            min_samples=6,
        ),
    )
    assert summary.pass_promotion_gate is False
    assert any("accuracy below threshold" in failure for failure in summary.failures)
    assert any("abstain rate above threshold" in failure for failure in summary.failures)


def test_train_region_expert_from_manifest_writes_versioned_artifacts(tmp_path):
    samples = _samples()
    manifest = fbct.build_dataset_manifest(samples, split_seed="train-seed")
    result = fbct.train_region_expert_from_manifest(
        samples,
        manifest=manifest,
        region_id="grounding",
        output_dir=tmp_path,
    )

    assert result.schema_version == "flybrain-brain-cluster-region-expert/v1"
    assert os.path.exists(result.artifact_path)
    assert os.path.exists(result.metrics_path)
    assert result.train_count > 0
    assert result.model_fingerprint
    assert result.metrics_fingerprint

    with open(result.artifact_path, "r", encoding="utf-8") as fh:
        model_payload = json.load(fh)
    with open(result.metrics_path, "r", encoding="utf-8") as fh:
        metrics_payload = json.load(fh)

    assert model_payload["schema_version"] == "flybrain-brain-cluster-region-expert/v1"
    assert model_payload["artifact_kind"] == "region_expert_model"
    assert model_payload["dataset_fingerprint"] == manifest.samples_sha256
    assert model_payload["region_id"] == "grounding"
    assert model_payload["model"]["model_type"] == "multinomial_naive_bayes"
    assert "accept" in model_payload["label_space"] or "reject" in model_payload["label_space"]
    assert metrics_payload["schema_version"] == "flybrain-brain-cluster-region-expert/v1-metrics"
    assert metrics_payload["manifest_id"] == manifest.manifest_id
    assert metrics_payload["val_metrics"]["sample_count"] >= 0


def test_train_region_expert_fingerprint_is_deterministic(tmp_path):
    samples = _samples()
    manifest = fbct.build_dataset_manifest(samples, split_seed="det-seed")
    first = fbct.train_region_expert_from_manifest(
        samples,
        manifest=manifest,
        region_id="provenance",
        output_dir=tmp_path,
    )
    second = fbct.train_region_expert_from_manifest(
        samples,
        manifest=manifest,
        region_id="provenance",
        output_dir=tmp_path,
    )
    assert first.model_fingerprint == second.model_fingerprint
    assert first.metrics_fingerprint == second.metrics_fingerprint


def test_train_router_from_labeled_data_writes_artifacts(tmp_path):
    routing_samples = [
        fbct.RouterTrainingSample(
            sample_id=f"r{i}",
            task_type="verification" if i % 2 == 0 else "grounding",
            risk_tier="high" if i % 3 == 0 else "medium",
            preferred_region="provenance_expert" if i % 2 == 0 else "grounding_expert",
            expected_expert="safety_expert" if i % 3 == 0 else "provenance_expert",
            metadata={"latency_tier": "interactive" if i % 2 == 0 else "batch"},
        )
        for i in range(1, 16)
    ]
    result = fbct.train_router_from_labeled_data(
        routing_samples,
        split_seed="router-seed",
        output_dir=tmp_path,
    )
    assert result.schema_version == "flybrain-brain-cluster-router/v1"
    assert result.train_count > 0
    assert os.path.exists(result.artifact_path)
    assert os.path.exists(result.metrics_path)

    with open(result.artifact_path, "r", encoding="utf-8") as fh:
        model_payload = json.load(fh)
    with open(result.metrics_path, "r", encoding="utf-8") as fh:
        metrics_payload = json.load(fh)
    assert model_payload["artifact_kind"] == "router_model"
    assert model_payload["split_seed"] == "router-seed"
    assert model_payload["model"]["model_type"] == "multinomial_naive_bayes"
    assert set(model_payload["label_space"]) == {"provenance_expert", "safety_expert"}
    assert metrics_payload["schema_version"] == "flybrain-brain-cluster-router/v1-metrics"
    assert metrics_payload["val_split_size"] >= 0


def test_train_swarm_consensus_student_writes_artifacts(tmp_path):
    samples = _samples()
    manifest = fbct.build_dataset_manifest(samples, split_seed="swarm-seed")
    experts = {}
    for region_id in sorted(manifest.region_counts):
        experts[region_id] = fbct.train_region_expert_from_manifest(
            samples,
            manifest=manifest,
            region_id=region_id,
            output_dir=tmp_path / "region",
        )
    result = fbct.train_swarm_consensus_student(
        samples,
        manifest=manifest,
        expert_results=experts,
        output_dir=tmp_path / "swarm-student",
        min_consensus_samples=4,
    )
    assert result.schema_version == "flybrain-brain-cluster-swarm-student/v1"
    assert result.consensus_train_count >= 4
    assert os.path.exists(result.artifact_path)
    assert os.path.exists(result.metrics_path)
    with open(result.artifact_path, "r", encoding="utf-8") as fh:
        model_payload = json.load(fh)
    assert model_payload["artifact_kind"] == "swarm_consensus_student_model"
    assert model_payload["consensus_policy"]["min_vote_share"] == 0.6


def test_train_swarm_consensus_student_is_deterministic(tmp_path):
    samples = _samples()
    manifest = fbct.build_dataset_manifest(samples, split_seed="swarm-det")
    experts = {}
    for region_id in sorted(manifest.region_counts):
        experts[region_id] = fbct.train_region_expert_from_manifest(
            samples,
            manifest=manifest,
            region_id=region_id,
            output_dir=tmp_path / "region",
        )
    first = fbct.train_swarm_consensus_student(
        samples,
        manifest=manifest,
        expert_results=experts,
        output_dir=tmp_path / "student-a",
        min_consensus_samples=4,
    )
    second = fbct.train_swarm_consensus_student(
        samples,
        manifest=manifest,
        expert_results=experts,
        output_dir=tmp_path / "student-b",
        min_consensus_samples=4,
    )
    assert first.model_fingerprint == second.model_fingerprint
    assert first.metrics_fingerprint == second.metrics_fingerprint


def test_gate_runner_pass_report():
    golden = fbct.build_golden_set(_samples(), max_per_region=10)
    predictions = [
        fbct.PredictionRecord(
            sample_id=s.sample_id,
            predicted_label=s.expected_label,
            predicted_confidence=s.expected_confidence,
            abstained=False,
        )
        for s in golden
    ]
    report = fbct.run_gate(
        golden,
        predictions,
        thresholds=fbct.PromotionGateThresholds(
            min_accuracy=0.8,
            max_abstain_rate=0.2,
            max_confidence_calibration_error=0.1,
            min_samples=10,
        ),
    )
    assert report["pass"] is True
    assert report["status"] == "pass"
    assert report["exit_code"] == 0
    assert report["failure_reasons"] == []
    assert set(report) == {"schema_version", "pass", "status", "deterministic", "thresholds", "metrics", "failure_reasons", "exit_code"}


def test_gate_runner_fail_report():
    golden = fbct.build_golden_set(_samples(), max_per_region=5)
    bad_predictions = []
    for idx, sample in enumerate(golden):
        if idx < 3:
            bad_predictions.append(
                fbct.PredictionRecord(
                    sample_id=sample.sample_id,
                    predicted_label="wrong",
                    predicted_confidence=0.2,
                    abstained=False,
                )
            )
            continue
        bad_predictions.append(
            fbct.PredictionRecord(
                sample_id=sample.sample_id,
                predicted_label=sample.expected_label,
                predicted_confidence=0.4,
                abstained=True,
            )
        )
    report = fbct.run_gate(
        golden,
        bad_predictions,
        thresholds=fbct.PromotionGateThresholds(
            min_accuracy=0.7,
            max_abstain_rate=0.1,
            max_confidence_calibration_error=0.2,
            min_samples=6,
        ),
    )
    assert report["pass"] is False
    assert report["status"] == "fail"
    assert report["exit_code"] == 1
    assert any("accuracy below threshold" in reason for reason in report["failure_reasons"])
    assert any("abstain rate above threshold" in reason for reason in report["failure_reasons"])


def test_gate_runner_cli_reports_malformed_input():
    exit_code = fbct.main(["--samples", '{"bad": "payload"}', "--predictions", "[]"])
    assert exit_code == 2


def test_gate_runner_report_shape_is_deterministic():
    golden = fbct.build_golden_set(_samples(), max_per_region=10)
    predictions = [
        fbct.PredictionRecord(
            sample_id=s.sample_id,
            predicted_label=s.expected_label,
            predicted_confidence=s.expected_confidence,
            abstained=False,
        )
        for s in golden
    ]
    report = fbct.run_gate(golden, predictions)
    round_trip = __import__("json").loads(fbct._serialize_report(report))
    assert round_trip == report
    assert fbct._serialize_report(report).endswith("\n")
