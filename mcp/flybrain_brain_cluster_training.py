from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrainingSample:
    sample_id: str
    region_id: str
    input_text: str
    expected_label: str
    expected_confidence: float
    provenance_refs: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id is required")
        if not self.region_id.strip():
            raise ValueError("region_id is required")
        if not self.input_text.strip():
            raise ValueError("input_text is required")
        if not self.expected_label.strip():
            raise ValueError("expected_label is required")
        if not (0.0 <= float(self.expected_confidence) <= 1.0):
            raise ValueError("expected_confidence must be in [0.0, 1.0]")
        if not self.provenance_refs:
            raise ValueError("provenance_refs must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "region_id": self.region_id,
            "input_text": self.input_text,
            "expected_label": self.expected_label,
            "expected_confidence": float(self.expected_confidence),
            "provenance_refs": list(self.provenance_refs),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: str
    manifest_id: str
    sample_count: int
    region_counts: Mapping[str, int]
    split_seed: str
    samples_sha256: str
    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    notes: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "sample_count": self.sample_count,
            "region_counts": dict(self.region_counts),
            "split_seed": self.split_seed,
            "samples_sha256": self.samples_sha256,
            "train_ids": list(self.train_ids),
            "val_ids": list(self.val_ids),
            "test_ids": list(self.test_ids),
            "notes": dict(self.notes),
        }


@dataclass(frozen=True)
class PredictionRecord:
    sample_id: str
    predicted_label: str
    predicted_confidence: float
    abstained: bool = False

    def validate(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id is required")
        if not (0.0 <= float(self.predicted_confidence) <= 1.0):
            raise ValueError("predicted_confidence must be in [0.0, 1.0]")


@dataclass(frozen=True)
class PromotionGateThresholds:
    min_accuracy: float = 0.85
    max_abstain_rate: float = 0.15
    max_confidence_calibration_error: float = 0.08
    min_samples: int = 30


@dataclass(frozen=True)
class EvaluationSummary:
    sample_count: int
    accuracy: float
    abstain_rate: float
    mean_calibration_error: float
    pass_promotion_gate: bool
    failures: tuple[str, ...]


def deterministic_split_ids(
    sample_ids: Sequence[str],
    *,
    split_seed: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if not sample_ids:
        return tuple(), tuple(), tuple()
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    decorated: list[tuple[str, str]] = []
    for sample_id in sample_ids:
        key = _sha256_hex(f"{split_seed}:{sample_id.strip()}")
        decorated.append((key, sample_id))
    ordered = [sample_id for _, sample_id in sorted(decorated, key=lambda item: item[0])]
    n = len(ordered)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return tuple(ordered[:train_end]), tuple(ordered[train_end:val_end]), tuple(ordered[val_end:])


def build_dataset_manifest(
    samples: Sequence[TrainingSample],
    *,
    split_seed: str,
    schema_version: str = "flybrain-brain-cluster-dataset/v1",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    notes: Mapping[str, Any] | None = None,
) -> DatasetManifest:
    if not samples:
        raise ValueError("samples must be non-empty")
    validated: list[TrainingSample] = []
    region_counts: dict[str, int] = {}
    seen_ids: set[str] = set()
    for sample in samples:
        sample.validate()
        if sample.sample_id in seen_ids:
            raise ValueError(f"duplicate sample_id: {sample.sample_id}")
        seen_ids.add(sample.sample_id)
        validated.append(sample)
        region_counts[sample.region_id] = region_counts.get(sample.region_id, 0) + 1

    canonical_samples = [sample.as_dict() for sample in sorted(validated, key=lambda s: s.sample_id)]
    digest = _sha256_hex(_stable_json(canonical_samples))
    train_ids, val_ids, test_ids = deterministic_split_ids(
        [sample.sample_id for sample in validated],
        split_seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    manifest_payload = {
        "schema_version": schema_version,
        "split_seed": split_seed,
        "samples_sha256": digest,
        "sample_count": len(validated),
        "region_counts": region_counts,
        "train_ids": list(train_ids),
        "val_ids": list(val_ids),
        "test_ids": list(test_ids),
    }
    manifest_id = f"fbc-dataset-{_sha256_hex(_stable_json(manifest_payload))[:16]}"
    return DatasetManifest(
        schema_version=schema_version,
        manifest_id=manifest_id,
        sample_count=len(validated),
        region_counts=region_counts,
        split_seed=split_seed,
        samples_sha256=digest,
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        notes=dict(notes or {}),
    )


def build_golden_set(
    samples: Sequence[TrainingSample],
    *,
    max_per_region: int = 20,
) -> tuple[TrainingSample, ...]:
    if max_per_region <= 0:
        raise ValueError("max_per_region must be > 0")
    by_region: dict[str, list[TrainingSample]] = {}
    for sample in samples:
        sample.validate()
        by_region.setdefault(sample.region_id, []).append(sample)
    golden: list[TrainingSample] = []
    for region_id in sorted(by_region):
        ordered = sorted(by_region[region_id], key=lambda s: s.sample_id)
        golden.extend(ordered[:max_per_region])
    return tuple(golden)


def evaluate_predictions(
    golden_samples: Sequence[TrainingSample],
    predictions: Sequence[PredictionRecord],
    *,
    thresholds: PromotionGateThresholds | None = None,
) -> EvaluationSummary:
    thresholds = thresholds or PromotionGateThresholds()
    if not golden_samples:
        raise ValueError("golden_samples must be non-empty")
    for sample in golden_samples:
        sample.validate()
    pred_map: dict[str, PredictionRecord] = {}
    for record in predictions:
        record.validate()
        pred_map[record.sample_id] = record

    total = len(golden_samples)
    correct = 0
    abstained = 0
    calibration_errors: list[float] = []
    missing = 0

    for sample in golden_samples:
        pred = pred_map.get(sample.sample_id)
        if pred is None:
            missing += 1
            continue
        if pred.abstained:
            abstained += 1
            continue
        if pred.predicted_label == sample.expected_label:
            correct += 1
        calibration_errors.append(abs(float(pred.predicted_confidence) - float(sample.expected_confidence)))

    abstain_rate = (abstained + missing) / total
    evaluated = total - abstained - missing
    accuracy = (correct / evaluated) if evaluated > 0 else 0.0
    mean_calibration_error = (sum(calibration_errors) / len(calibration_errors)) if calibration_errors else 1.0

    failures: list[str] = []
    if total < thresholds.min_samples:
        failures.append(f"insufficient sample count ({total} < {thresholds.min_samples})")
    if accuracy < thresholds.min_accuracy:
        failures.append(f"accuracy below threshold ({accuracy:.3f} < {thresholds.min_accuracy:.3f})")
    if abstain_rate > thresholds.max_abstain_rate:
        failures.append(f"abstain rate above threshold ({abstain_rate:.3f} > {thresholds.max_abstain_rate:.3f})")
    if mean_calibration_error > thresholds.max_confidence_calibration_error:
        failures.append(
            "calibration error above threshold "
            f"({mean_calibration_error:.3f} > {thresholds.max_confidence_calibration_error:.3f})"
        )

    return EvaluationSummary(
        sample_count=total,
        accuracy=accuracy,
        abstain_rate=abstain_rate,
        mean_calibration_error=mean_calibration_error,
        pass_promotion_gate=(len(failures) == 0),
        failures=tuple(failures),
    )

