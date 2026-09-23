import os
import sys

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
