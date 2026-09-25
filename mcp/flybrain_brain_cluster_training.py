from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
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
    # Structured learner inputs (numeric or categorical values) used by the
    # feature backends in ``flybrain_learners``; the legacy NB reads only
    # ``input_text``. Omitted from ``as_dict`` when empty so legacy dataset
    # digests (``samples_sha256``) are unchanged.
    features: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id is required")
        validate_sample_features(self.features)
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
        payload = {
            "sample_id": self.sample_id,
            "region_id": self.region_id,
            "input_text": self.input_text,
            "expected_label": self.expected_label,
            "expected_confidence": float(self.expected_confidence),
            "provenance_refs": list(self.provenance_refs),
            "metadata": dict(self.metadata),
        }
        if self.features:
            payload["features"] = dict(self.features)
        return payload


_FEATURE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_sample_features(features: Mapping[str, Any]) -> None:
    """Feature names are identifiers; values are finite numbers, bools, strings or None."""
    if not isinstance(features, Mapping):
        raise ValueError("features must be a mapping")
    for key, value in features.items():
        if not isinstance(key, str) or not _FEATURE_NAME_RE.fullmatch(key):
            raise ValueError(f"invalid feature name: {key!r}")
        if value is None or isinstance(value, (str, bool)):
            continue
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"feature {key} must be finite (use None for missing)")
            continue
        raise ValueError(f"feature {key} has unsupported type {type(value).__name__}")


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

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_accuracy": float(self.min_accuracy),
            "max_abstain_rate": float(self.max_abstain_rate),
            "max_confidence_calibration_error": float(self.max_confidence_calibration_error),
            "min_samples": int(self.min_samples),
        }


@dataclass(frozen=True)
class EvaluationSummary:
    sample_count: int
    accuracy: float
    abstain_rate: float
    mean_calibration_error: float
    pass_promotion_gate: bool
    failures: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "accuracy": float(self.accuracy),
            "abstain_rate": float(self.abstain_rate),
            "mean_calibration_error": float(self.mean_calibration_error),
            "pass_promotion_gate": bool(self.pass_promotion_gate),
            "failures": list(self.failures),
        }


@dataclass(frozen=True)
class RegionExpertTrainingResult:
    artifact_id: str
    schema_version: str
    artifact_path: str
    metrics_path: str
    region_id: str
    model_fingerprint: str
    metrics_fingerprint: str
    train_count: int
    val_count: int
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class RouterTrainingSample:
    sample_id: str
    task_type: str
    risk_tier: str
    preferred_region: str
    expected_expert: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id is required")
        if not self.task_type.strip():
            raise ValueError("task_type is required")
        if not self.risk_tier.strip():
            raise ValueError("risk_tier is required")
        if not self.expected_expert.strip():
            raise ValueError("expected_expert is required")

    def as_feature_text(self) -> str:
        features = [
            f"task:{self.task_type.strip().lower()}",
            f"risk:{self.risk_tier.strip().lower()}",
        ]
        preferred = self.preferred_region.strip().lower()
        if preferred:
            features.append(f"preferred:{preferred}")
        for key in sorted(self.metadata):
            value = str(self.metadata[key]).strip().lower()
            if value:
                features.append(f"meta:{key.strip().lower()}={value}")
        return " ".join(features)


@dataclass(frozen=True)
class RouterTrainingResult:
    artifact_id: str
    schema_version: str
    artifact_path: str
    metrics_path: str
    model_fingerprint: str
    metrics_fingerprint: str
    train_count: int
    val_count: int
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class SwarmStudentTrainingResult:
    artifact_id: str
    schema_version: str
    artifact_path: str
    metrics_path: str
    model_fingerprint: str
    metrics_fingerprint: str
    consensus_train_count: int
    consensus_eval_count: int
    metrics: Mapping[str, Any]


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


# Metadata values that carry no grouping information (never tie samples together).
_UNKNOWN_GROUP_VALUES = frozenset({"", "unknown", "none", "nan", "null", "<na>", "n/a", "na"})
# Always honoured, whatever the dataset registry lists: a builder-declared group.
GENERIC_SPLIT_GROUP_KEY = "split_group"


def _group_value(metadata: Mapping[str, Any], key: str) -> str | None:
    if key not in metadata:
        return None
    raw = metadata[key]
    if raw is None or isinstance(raw, (dict, list, tuple, set)):
        return None
    if isinstance(raw, float) and raw != raw:
        return None
    text = str(raw).strip()
    if text.lower() in _UNKNOWN_GROUP_VALUES:
        return None
    return text


def split_group_components(
    samples: Sequence[TrainingSample],
    *,
    group_keys: Sequence[str],
) -> dict[str, str]:
    """Map each sample_id to its split component id.

    Two samples are in the same component when they share a known value of
    any key in ``group_keys`` (transitively: union-find over all keys), e.g. the
    same cell type, the same hemilineage, or the same left/right pair. A sample
    with no known group value is its own component, keyed by its sample_id.
    """
    keys = tuple(dict.fromkeys(str(key).strip() for key in group_keys if str(key).strip()))
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Deterministic: the lexicographically smaller root wins.
            if rb < ra:
                ra, rb = rb, ra
            parent[rb] = ra

    for sample in samples:
        node = f"s:{sample.sample_id.strip()}"
        find(node)
        metadata = sample.metadata if isinstance(sample.metadata, Mapping) else {}
        for key in keys:
            value = _group_value(metadata, key)
            if value is not None:
                union(f"g:{key}={value}", node)

    members: dict[str, list[str]] = {}
    for sample in samples:
        members.setdefault(find(f"s:{sample.sample_id.strip()}"), []).append(sample.sample_id)
    smallest_group: dict[str, str] = {}
    for node in list(parent):
        if node.startswith("g:"):
            root = find(node)
            if root not in smallest_group or node < smallest_group[root]:
                smallest_group[root] = node
    component_of: dict[str, str] = {}
    for root, ids in members.items():
        # A component tied by any group value is named after its smallest group
        # value; an ungrouped sample keeps its own id (identical to the legacy split).
        component_id = smallest_group.get(root) or ids[0].strip()
        for sample_id in ids:
            component_of[sample_id] = component_id
    return component_of


def grouped_split_ids(
    samples: Sequence[TrainingSample],
    *,
    group_keys: Sequence[str],
    split_seed: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    """Deterministic split that never lets a group straddle train/val/test.

    Components (``split_group_components``) are ordered by
    ``sha256(split_seed:component_id)`` and assigned whole: a component goes to
    train while fewer than ``int(n * train_ratio)`` samples are in train, then
    to val, then to test. With no grouping metadata every component is one
    sample and the result equals ``deterministic_split_ids``.

    Returns ``(train, val, test, info)``; ``info`` records the grouping, the
    achieved split sizes, and how many multi-sample components the legacy
    per-sample split would have straddled (the leakage it prevents).
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")
    keys = tuple(dict.fromkeys(str(key).strip() for key in group_keys if str(key).strip()))
    ids = [sample.sample_id for sample in samples]
    component_of = split_group_components(samples, group_keys=keys)
    members: dict[str, list[str]] = {}
    for sample_id in ids:
        members.setdefault(component_of[sample_id], []).append(sample_id)
    for component_id in members:
        members[component_id].sort(key=lambda sid: _sha256_hex(f"{split_seed}:{sid.strip()}"))
    ordered_components = sorted(members, key=lambda cid: _sha256_hex(f"{split_seed}:{cid}"))

    n = len(ids)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    assigned = 0
    for component_id in ordered_components:
        bucket = train if assigned < train_end else val if assigned < val_end else test
        bucket.extend(members[component_id])
        assigned += len(members[component_id])

    multi = {cid: rows for cid, rows in members.items() if len(rows) > 1}
    legacy_train, legacy_val, _ = deterministic_split_ids(
        ids, split_seed=split_seed, train_ratio=train_ratio, val_ratio=val_ratio
    )
    legacy_split = {sid: 0 for sid in legacy_train}
    legacy_split.update({sid: 1 for sid in legacy_val})
    straddling = sum(1 for rows in multi.values() if len({legacy_split.get(sid, 2) for sid in rows}) > 1)
    info = {
        "strategy": "grouped",
        "group_keys": list(keys),
        "component_count": len(members),
        "multi_sample_components": len(multi),
        "grouped_sample_count": sum(len(rows) for rows in multi.values()),
        "largest_component_size": max((len(rows) for rows in members.values()), default=0),
        "achieved_counts": {"train": len(train), "val": len(val), "test": len(test)},
        "per_sample_split_straddling_components": straddling,
    }
    return tuple(train), tuple(val), tuple(test), info


def assert_grouped_split(
    samples: Sequence[TrainingSample],
    manifest: "DatasetManifest",
    *,
    group_keys: Sequence[str],
) -> None:
    """Fail closed if any known group value appears in more than one split."""
    split_of: dict[str, str] = {}
    for name, split_ids in (("train", manifest.train_ids), ("val", manifest.val_ids), ("test", manifest.test_ids)):
        for sample_id in split_ids:
            split_of[sample_id] = name
    seen: dict[tuple[str, str], set[str]] = {}
    for sample in samples:
        split = split_of.get(sample.sample_id)
        if split is None:
            continue
        metadata = sample.metadata if isinstance(sample.metadata, Mapping) else {}
        for key in group_keys:
            value = _group_value(metadata, key)
            if value is not None:
                seen.setdefault((key, value), set()).add(split)
    straddling = sorted(f"{key}={value}" for (key, value), splits in seen.items() if len(splits) > 1)
    if straddling:
        raise ValueError(f"split groups straddle splits: {', '.join(straddling[:10])}")


def build_dataset_manifest(
    samples: Sequence[TrainingSample],
    *,
    split_seed: str,
    schema_version: str = "flybrain-brain-cluster-dataset/v1",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    notes: Mapping[str, Any] | None = None,
    group_keys: Sequence[str] | None = None,
) -> DatasetManifest:
    """Validate samples and split them deterministically.

    With ``group_keys`` the split is grouped (``grouped_split_ids``): samples
    that share a known value of any listed metadata key always land in the same
    split, and ``notes["split"]`` records the grouping and its leakage stats.
    ``group_keys=None`` keeps the legacy per-sample split.
    """
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
    manifest_notes = dict(notes or {})
    if group_keys is None:
        train_ids, val_ids, test_ids = deterministic_split_ids(
            [sample.sample_id for sample in validated],
            split_seed=split_seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
    else:
        train_ids, val_ids, test_ids, split_info = grouped_split_ids(
            validated,
            group_keys=group_keys,
            split_seed=split_seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
        manifest_notes["split"] = split_info

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
        notes=manifest_notes,
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


def _as_thresholds_dict(thresholds: PromotionGateThresholds | Mapping[str, Any] | None) -> dict[str, Any]:
    if thresholds is None:
        thresholds = PromotionGateThresholds()
    if isinstance(thresholds, PromotionGateThresholds):
        threshold_obj = thresholds
        return {
            "min_accuracy": float(threshold_obj.min_accuracy),
            "max_abstain_rate": float(threshold_obj.max_abstain_rate),
            "max_confidence_calibration_error": float(threshold_obj.max_confidence_calibration_error),
            "min_samples": int(threshold_obj.min_samples),
        }
    if not isinstance(thresholds, Mapping):
        raise ValueError("thresholds must be a PromotionGateThresholds or mapping")
    normalized: dict[str, Any] = {}
    for key in ("min_accuracy", "max_abstain_rate", "max_confidence_calibration_error", "min_samples"):
        if key in thresholds:
            normalized[key] = thresholds[key]
    if not normalized:
        raise ValueError("thresholds mapping must include at least one supported threshold")
    default = PromotionGateThresholds()
    return {
        "min_accuracy": float(normalized.get("min_accuracy", default.min_accuracy)),
        "max_abstain_rate": float(normalized.get("max_abstain_rate", default.max_abstain_rate)),
        "max_confidence_calibration_error": float(normalized.get("max_confidence_calibration_error", default.max_confidence_calibration_error)),
        "min_samples": int(normalized.get("min_samples", default.min_samples)),
    }


def _normalize_training_sample(raw: TrainingSample | Mapping[str, Any]) -> TrainingSample:
    if isinstance(raw, TrainingSample):
        raw.validate()
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError("training sample must be a TrainingSample or mapping")
    missing = [key for key in ("sample_id", "region_id", "input_text", "expected_label", "expected_confidence") if key not in raw]
    if missing:
        raise ValueError(f"training sample missing required keys: {', '.join(missing)}")
    provenance = raw.get("provenance_refs")
    if provenance is None:
        provenance = raw.get("provenance")
    if isinstance(provenance, str):
        provenance_refs = (provenance,)
    elif provenance is None:
        provenance_refs = ()
    else:
        provenance_refs = tuple(str(item) for item in provenance)
    sample = TrainingSample(
        sample_id=str(raw["sample_id"]),
        region_id=str(raw["region_id"]),
        input_text=str(raw["input_text"]),
        expected_label=str(raw["expected_label"]),
        expected_confidence=float(raw["expected_confidence"]),
        provenance_refs=provenance_refs,
        metadata=dict(raw.get("metadata") or {}),
        features=dict(raw.get("features") or {}),
    )
    sample.validate()
    return sample


def _normalize_prediction_record(raw: PredictionRecord | Mapping[str, Any]) -> PredictionRecord:
    if isinstance(raw, PredictionRecord):
        raw.validate()
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError("prediction record must be a PredictionRecord or mapping")
    if "sample_id" not in raw:
        raise ValueError("prediction record missing sample_id")
    if "predicted_label" not in raw:
        raise ValueError("prediction record missing predicted_label")
    if "predicted_confidence" not in raw:
        raise ValueError("prediction record missing predicted_confidence")
    record = PredictionRecord(
        sample_id=str(raw["sample_id"]),
        predicted_label=str(raw["predicted_label"]),
        predicted_confidence=float(raw["predicted_confidence"]),
        abstained=bool(raw.get("abstained", False)),
    )
    record.validate()
    return record


def _load_json_value(source: str | Path | Mapping[str, Any] | Sequence[Any] | None, *, label: str) -> Any:
    if source is None:
        raise ValueError(f"{label} is required")
    if isinstance(source, (dict, list, tuple)):
        return source
    text = str(source).strip()
    if not text:
        raise ValueError(f"{label} is required")
    if text.startswith("{") or text.startswith("["):
        return json.loads(text)
    path = Path(text)
    if not path.exists():
        raise ValueError(f"{label} not found: {text}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label}: {exc.msg}") from exc


def _extract_items(payload: Mapping[str, Any] | Sequence[Any], *, key: str) -> list[Any]:
    if isinstance(payload, Mapping):
        if key in payload:
            return list(payload[key])
        if key.rstrip("s") in payload:
            return list(payload[key.rstrip("s")])
        if key == "samples" and "golden_set" in payload:
            return list(payload["golden_set"])
        if key == "predictions" and "prediction_records" in payload:
            return list(payload["prediction_records"])
        raise ValueError(f"payload missing '{key}'")
    return list(payload)


def _serialize_report(report: Mapping[str, Any]) -> str:
    return json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"


def run_gate(
    samples: Sequence[TrainingSample | Mapping[str, Any]] | Mapping[str, Any] | str | Path,
    predictions: Sequence[PredictionRecord | Mapping[str, Any]] | Mapping[str, Any] | str | Path,
    *,
    thresholds: PromotionGateThresholds | Mapping[str, Any] | None = None,
    max_per_region: int = 20,
) -> dict[str, Any]:
    if isinstance(samples, (str, bytes, Path)):
        sample_payload = _load_json_value(samples, label="samples")
    elif isinstance(samples, Mapping):
        sample_payload = samples
    else:
        sample_payload = list(samples)
    if isinstance(predictions, (str, bytes, Path)):
        prediction_payload = _load_json_value(predictions, label="predictions")
    elif isinstance(predictions, Mapping):
        prediction_payload = predictions
    else:
        prediction_payload = list(predictions)
    sample_list = [_normalize_training_sample(item) for item in _extract_items(sample_payload, key="samples")]
    if not sample_list:
        raise ValueError("samples must be non-empty")
    golden = build_golden_set(sample_list, max_per_region=max_per_region)
    pred_list = [_normalize_prediction_record(item) for item in _extract_items(prediction_payload, key="predictions")]
    threshold_values = _as_thresholds_dict(thresholds)
    summary = evaluate_predictions(golden, pred_list, thresholds=PromotionGateThresholds(**threshold_values))
    report = {
        "schema_version": "braincluster-golden-set-gate/v1",
        "pass": bool(summary.pass_promotion_gate),
        "status": "pass" if summary.pass_promotion_gate else "fail",
        "deterministic": True,
        "thresholds": threshold_values,
        "metrics": {
            "sample_count": int(summary.sample_count),
            "accuracy": float(summary.accuracy),
            "abstain_rate": float(summary.abstain_rate),
            "mean_calibration_error": float(summary.mean_calibration_error),
        },
        "failure_reasons": list(summary.failures),
        "exit_code": 0 if summary.pass_promotion_gate else 1,
    }
    return report


def run_golden_set_gate(
    samples: Sequence[TrainingSample | Mapping[str, Any]] | Mapping[str, Any] | str | Path,
    predictions: Sequence[PredictionRecord | Mapping[str, Any]] | Mapping[str, Any] | str | Path,
    *,
    thresholds: PromotionGateThresholds | Mapping[str, Any] | None = None,
    max_per_region: int = 20,
) -> dict[str, Any]:
    return run_gate(samples, predictions, thresholds=thresholds, max_per_region=max_per_region)


def _parse_thresholds(raw: str | Mapping[str, Any] | None) -> PromotionGateThresholds | None:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return PromotionGateThresholds(**_as_thresholds_dict(raw))
    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("{"):
        values = json.loads(text)
        return PromotionGateThresholds(**_as_thresholds_dict(values))
    candidate = Path(text)
    if candidate.exists():
        values = json.loads(candidate.read_text(encoding="utf-8"))
        return PromotionGateThresholds(**_as_thresholds_dict(values))
    raise ValueError(f"thresholds source is not valid JSON nor a readable file: {raw!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic brain-cluster golden-set gate.")
    parser.add_argument("--samples", "--golden-set", dest="samples", help="Path or inline JSON for golden-set samples")
    parser.add_argument("--predictions", "--prediction-set", dest="predictions", help="Path or inline JSON for prediction records")
    parser.add_argument("--thresholds", "--thresholds-file", dest="thresholds", help="JSON string or path to threshold settings")
    parser.add_argument("--max-per-region", type=int, default=20, help="Maximum samples to keep per region before evaluation")
    parser.add_argument("--report", help="Optional path to write the JSON report")
    args = parser.parse_args(argv)
    try:
        report = run_gate(
            args.samples,
            args.predictions,
            thresholds=_parse_thresholds(args.thresholds),
            max_per_region=args.max_per_region,
        )
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        threshold_payload = _as_thresholds_dict(PromotionGateThresholds())
        payload = {
            "schema_version": "braincluster-golden-set-gate/v1",
            "pass": False,
            "status": "error",
            "deterministic": True,
            "thresholds": threshold_payload,
            "metrics": {
                "sample_count": 0,
                "accuracy": 0.0,
                "abstain_rate": 0.0,
                "mean_calibration_error": 0.0,
            },
            "failure_reasons": [f"malformed input: {exc}"],
            "exit_code": 2,
        }
        output = _serialize_report(payload)
        if args.report:
            Path(args.report).write_text(output, encoding="utf-8")
        print(output, end="")
        return 2
    output = _serialize_report(report)
    if args.report:
        Path(args.report).write_text(output, encoding="utf-8")
    print(output, end="")
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9_]+", text.lower()))


def _fit_multinomial_nb(
    rows: Sequence[tuple[str, str]],
    *,
    alpha: float = 1.0,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("training rows must be non-empty")
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0")

    label_doc_counts: dict[str, int] = {}
    label_token_counts: dict[str, dict[str, int]] = {}
    label_token_totals: dict[str, int] = {}
    vocab: set[str] = set()

    for text, label in rows:
        label = label.strip()
        if not label:
            raise ValueError("label is required")
        label_doc_counts[label] = label_doc_counts.get(label, 0) + 1
        token_counts = label_token_counts.setdefault(label, {})
        token_total = label_token_totals.get(label, 0)
        for token in _tokenize(text):
            vocab.add(token)
            token_counts[token] = token_counts.get(token, 0) + 1
            token_total += 1
        label_token_totals[label] = token_total

    labels = tuple(sorted(label_doc_counts))
    total_docs = sum(label_doc_counts.values())
    vocab_size = max(1, len(vocab))

    log_prior: dict[str, float] = {}
    log_probs: dict[str, dict[str, float]] = {}
    unknown_token_log_prob: dict[str, float] = {}
    for label in labels:
        log_prior[label] = math.log(label_doc_counts[label] / total_docs)
        token_counts = label_token_counts[label]
        denom = label_token_totals.get(label, 0) + alpha * vocab_size
        probs: dict[str, float] = {}
        for token in sorted(vocab):
            probs[token] = math.log((token_counts.get(token, 0) + alpha) / denom)
        log_probs[label] = probs
        unknown_token_log_prob[label] = math.log(alpha / denom)

    return {
        "model_type": "multinomial_naive_bayes",
        "alpha": alpha,
        "labels": labels,
        "vocab": tuple(sorted(vocab)),
        "log_prior": log_prior,
        "log_probs": log_probs,
        "unknown_token_log_prob": unknown_token_log_prob,
        "train_rows": len(rows),
    }


def _predict_multinomial_nb(model: Mapping[str, Any], text: str) -> tuple[str, float]:
    labels = tuple(model["labels"])
    if not labels:
        raise ValueError("model has no labels")
    log_prior = model["log_prior"]
    log_probs = model["log_probs"]
    unknown = model["unknown_token_log_prob"]

    token_freq: dict[str, int] = {}
    for token in _tokenize(text):
        token_freq[token] = token_freq.get(token, 0) + 1

    scores: dict[str, float] = {}
    for label in labels:
        score = float(log_prior[label])
        token_weights = log_probs[label]
        default_weight = float(unknown[label])
        for token, freq in token_freq.items():
            score += freq * float(token_weights.get(token, default_weight))
        scores[label] = score

    best_score = max(scores.values())
    winners = [label for label, score in scores.items() if score == best_score]
    predicted_label = sorted(winners)[0]

    max_log = max(scores.values())
    exp_scores = {label: math.exp(score - max_log) for label, score in scores.items()}
    exp_total = sum(exp_scores.values()) or 1.0
    confidence = float(exp_scores[predicted_label] / exp_total)
    return predicted_label, confidence


def _classification_metrics(
    rows: Sequence[tuple[str, str]],
    model: Mapping[str, Any],
) -> dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "mean_calibration_error": 1.0,
            "labels": {},
        }

    labels = tuple(sorted(set(model["labels"]) | {label for _, label in rows}))
    per_label_counts: dict[str, dict[str, int]] = {
        label: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for label in labels
    }

    correct = 0
    calibration_errors: list[float] = []
    for text, actual in rows:
        predicted, confidence = _predict_multinomial_nb(model, text)
        calibration_errors.append(abs(confidence - (1.0 if predicted == actual else 0.0)))
        if predicted == actual:
            correct += 1
            per_label_counts[actual]["tp"] += 1
        else:
            per_label_counts[predicted]["fp"] += 1
            per_label_counts[actual]["fn"] += 1
        per_label_counts[actual]["support"] += 1

    details: dict[str, Any] = {}
    f1_total = 0.0
    for label in labels:
        c = per_label_counts[label]
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1_total += f1
        details[label] = {
            "support": c["support"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "sample_count": len(rows),
        "accuracy": correct / len(rows),
        "macro_f1": f1_total / len(labels) if labels else 0.0,
        "mean_calibration_error": sum(calibration_errors) / len(calibration_errors),
        "labels": details,
    }


def _write_versioned_json(
    output_dir: str | Path,
    *,
    filename_prefix: str,
    payload: Mapping[str, Any],
) -> tuple[str, str]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    payload_text = _stable_json(payload)
    fingerprint = _sha256_hex(payload_text)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    file_path = output_root / f"{filename_prefix}-{ts}-{fingerprint[:12]}.json"
    file_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(file_path), fingerprint


def train_region_expert_from_manifest(
    samples: Sequence[TrainingSample],
    *,
    manifest: DatasetManifest,
    region_id: str,
    output_dir: str | Path,
    model_schema_version: str = "flybrain-brain-cluster-region-expert/v1",
    alpha: float = 1.0,
    random_seed: int = 0,
) -> RegionExpertTrainingResult:
    del random_seed  # training is deterministic by construction

    normalized_region = region_id.strip()
    if not normalized_region:
        raise ValueError("region_id is required")
    selected: dict[str, TrainingSample] = {}
    for sample in samples:
        sample.validate()
        if sample.region_id == normalized_region:
            selected[sample.sample_id] = sample

    train_rows: list[tuple[str, str]] = []
    val_rows: list[tuple[str, str]] = []
    for sample_id in manifest.train_ids:
        sample = selected.get(sample_id)
        if sample is not None:
            train_rows.append((sample.input_text, sample.expected_label))
    for sample_id in manifest.val_ids:
        sample = selected.get(sample_id)
        if sample is not None:
            val_rows.append((sample.input_text, sample.expected_label))
    if not train_rows:
        raise ValueError(f"no train samples for region_id='{normalized_region}' in manifest splits")

    model = _fit_multinomial_nb(train_rows, alpha=alpha)
    train_metrics = _classification_metrics(train_rows, model)
    val_metrics = _classification_metrics(val_rows, model) if val_rows else {
        "sample_count": 0,
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "mean_calibration_error": 1.0,
        "labels": {},
        "warning": "no validation samples in selected split",
    }
    metrics_payload = {
        "schema_version": f"{model_schema_version}-metrics",
        "region_id": normalized_region,
        "manifest_id": manifest.manifest_id,
        "dataset_fingerprint": manifest.samples_sha256,
        "train_split_size": len(train_rows),
        "val_split_size": len(val_rows),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    model_payload = {
        "schema_version": model_schema_version,
        "artifact_kind": "region_expert_model",
        "region_id": normalized_region,
        "manifest_id": manifest.manifest_id,
        "dataset_schema_version": manifest.schema_version,
        "dataset_fingerprint": manifest.samples_sha256,
        "split_seed": manifest.split_seed,
        "label_space": tuple(sorted({label for _, label in train_rows})),
        "model": model,
    }

    artifact_path, model_fingerprint = _write_versioned_json(
        output_dir,
        filename_prefix=f"region-expert-{normalized_region}-model",
        payload=model_payload,
    )
    metrics_path, metrics_fingerprint = _write_versioned_json(
        output_dir,
        filename_prefix=f"region-expert-{normalized_region}-metrics",
        payload=metrics_payload,
    )
    artifact_id = f"region-expert-{normalized_region}-{model_fingerprint[:12]}"
    return RegionExpertTrainingResult(
        artifact_id=artifact_id,
        schema_version=model_schema_version,
        artifact_path=artifact_path,
        metrics_path=metrics_path,
        region_id=normalized_region,
        model_fingerprint=model_fingerprint,
        metrics_fingerprint=metrics_fingerprint,
        train_count=len(train_rows),
        val_count=len(val_rows),
        metrics=metrics_payload,
    )


def train_router_from_labeled_data(
    routing_samples: Sequence[RouterTrainingSample],
    *,
    split_seed: str,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    schema_version: str = "flybrain-brain-cluster-router/v1",
    alpha: float = 1.0,
    random_seed: int = 0,
) -> RouterTrainingResult:
    del random_seed  # training is deterministic by construction

    validated: list[RouterTrainingSample] = []
    seen_ids: set[str] = set()
    for sample in routing_samples:
        sample.validate()
        if sample.sample_id in seen_ids:
            raise ValueError(f"duplicate routing sample_id: {sample.sample_id}")
        seen_ids.add(sample.sample_id)
        validated.append(sample)
    if not validated:
        raise ValueError("routing_samples must be non-empty")

    ordered_ids = [s.sample_id for s in sorted(validated, key=lambda s: s.sample_id)]
    val_ratio = max(0.0, (1.0 - train_ratio) / 2.0)
    train_ids, val_ids, _ = deterministic_split_ids(
        ordered_ids,
        split_seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    sample_by_id = {s.sample_id: s for s in validated}
    train_rows = [(sample_by_id[i].as_feature_text(), sample_by_id[i].expected_expert) for i in train_ids if i in sample_by_id]
    val_rows = [(sample_by_id[i].as_feature_text(), sample_by_id[i].expected_expert) for i in val_ids if i in sample_by_id]
    if not train_rows:
        raise ValueError("router training split produced zero train rows")
    model = _fit_multinomial_nb(train_rows, alpha=alpha)

    train_metrics = _classification_metrics(train_rows, model)
    val_metrics = _classification_metrics(val_rows, model) if val_rows else {
        "sample_count": 0,
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "mean_calibration_error": 1.0,
        "labels": {},
        "warning": "no validation samples in selected split",
    }

    data_fingerprint = _sha256_hex(_stable_json([s.as_feature_text() + "|" + s.expected_expert for s in validated]))
    model_payload = {
        "schema_version": schema_version,
        "artifact_kind": "router_model",
        "split_seed": split_seed,
        "train_ratio": train_ratio,
        "dataset_fingerprint": data_fingerprint,
        "label_space": tuple(sorted({s.expected_expert for s in validated})),
        "model": model,
    }
    metrics_payload = {
        "schema_version": f"{schema_version}-metrics",
        "split_seed": split_seed,
        "dataset_fingerprint": data_fingerprint,
        "train_split_size": len(train_rows),
        "val_split_size": len(val_rows),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    artifact_path, model_fingerprint = _write_versioned_json(
        output_dir,
        filename_prefix="router-model",
        payload=model_payload,
    )
    metrics_path, metrics_fingerprint = _write_versioned_json(
        output_dir,
        filename_prefix="router-metrics",
        payload=metrics_payload,
    )
    artifact_id = f"router-{model_fingerprint[:12]}"
    return RouterTrainingResult(
        artifact_id=artifact_id,
        schema_version=schema_version,
        artifact_path=artifact_path,
        metrics_path=metrics_path,
        model_fingerprint=model_fingerprint,
        metrics_fingerprint=metrics_fingerprint,
        train_count=len(train_rows),
        val_count=len(val_rows),
        metrics=metrics_payload,
    )


def train_swarm_consensus_student(
    samples: Sequence[TrainingSample],
    *,
    manifest: DatasetManifest,
    expert_results: Mapping[str, RegionExpertTrainingResult],
    output_dir: str | Path,
    min_vote_share: float = 0.6,
    min_mean_confidence: float = 0.55,
    min_consensus_samples: int = 30,
    alpha: float = 1.0,
    schema_version: str = "flybrain-brain-cluster-swarm-student/v1",
) -> SwarmStudentTrainingResult:
    if not (0.5 <= float(min_vote_share) <= 1.0):
        raise ValueError("min_vote_share must be in [0.5, 1.0]")
    if not (0.0 <= float(min_mean_confidence) <= 1.0):
        raise ValueError("min_mean_confidence must be in [0.0, 1.0]")
    if int(min_consensus_samples) < 1:
        raise ValueError("min_consensus_samples must be >= 1")
    if not expert_results:
        raise ValueError("expert_results must be non-empty")

    sample_by_id = {sample.sample_id: sample for sample in samples}
    expert_models: dict[str, Mapping[str, Any]] = {}
    for region_id, result in expert_results.items():
        payload = json.loads(Path(result.artifact_path).read_text(encoding="utf-8"))
        model = payload.get("model")
        if not isinstance(model, Mapping):
            raise ValueError(f"region expert artifact missing model payload for region_id='{region_id}'")
        expert_models[region_id] = model

    def _consensus_label(sample: TrainingSample) -> tuple[str, float, float]:
        votes: dict[str, int] = {}
        confidences: dict[str, list[float]] = {}
        for model in expert_models.values():
            label, conf = _predict_multinomial_nb(model, sample.input_text)
            votes[label] = votes.get(label, 0) + 1
            confidences.setdefault(label, []).append(float(conf))
        ranked = sorted(votes.items(), key=lambda item: (-item[1], item[0]))
        top_label, top_votes = ranked[0]
        vote_share = float(top_votes) / float(len(expert_models))
        mean_conf = sum(confidences.get(top_label, [0.0])) / max(1, len(confidences.get(top_label, [])))
        return top_label, vote_share, mean_conf

    train_ids = tuple(manifest.train_ids) + tuple(manifest.val_ids)
    eval_ids = tuple(manifest.test_ids)
    consensus_train_rows: list[tuple[str, str]] = []
    consensus_eval_rows: list[tuple[str, str]] = []
    consensus_audit: list[dict[str, Any]] = []

    for sample_id in train_ids:
        sample = sample_by_id.get(sample_id)
        if sample is None:
            continue
        label, vote_share, mean_conf = _consensus_label(sample)
        if vote_share >= float(min_vote_share) and mean_conf >= float(min_mean_confidence):
            consensus_train_rows.append((sample.input_text, label))
            consensus_audit.append(
                {
                    "sample_id": sample_id,
                    "split": "train",
                    "consensus_label": label,
                    "vote_share": vote_share,
                    "mean_confidence": mean_conf,
                }
            )

    for sample_id in eval_ids:
        sample = sample_by_id.get(sample_id)
        if sample is None:
            continue
        label, vote_share, mean_conf = _consensus_label(sample)
        if vote_share >= float(min_vote_share) and mean_conf >= float(min_mean_confidence):
            consensus_eval_rows.append((sample.input_text, label))
            consensus_audit.append(
                {
                    "sample_id": sample_id,
                    "split": "eval",
                    "consensus_label": label,
                    "vote_share": vote_share,
                    "mean_confidence": mean_conf,
                }
            )

    if len(consensus_train_rows) < int(min_consensus_samples):
        raise ValueError(
            f"insufficient consensus samples for swarm student "
            f"({len(consensus_train_rows)} < {int(min_consensus_samples)})"
        )

    model = _fit_multinomial_nb(consensus_train_rows, alpha=alpha)
    train_metrics = _classification_metrics(consensus_train_rows, model)
    eval_metrics = _classification_metrics(consensus_eval_rows, model) if consensus_eval_rows else {
        "sample_count": 0,
        "accuracy": 0.0,
        "macro_f1": 0.0,
        "mean_calibration_error": 1.0,
        "labels": {},
        "warning": "no consensus evaluation samples in test split",
    }

    model_payload = {
        "schema_version": schema_version,
        "artifact_kind": "swarm_consensus_student_model",
        "manifest_id": manifest.manifest_id,
        "dataset_fingerprint": manifest.samples_sha256,
        "consensus_policy": {
            "min_vote_share": float(min_vote_share),
            "min_mean_confidence": float(min_mean_confidence),
            "min_consensus_samples": int(min_consensus_samples),
        },
        "label_space": tuple(sorted({label for _, label in consensus_train_rows})),
        "model": model,
    }
    metrics_payload = {
        "schema_version": f"{schema_version}-metrics",
        "manifest_id": manifest.manifest_id,
        "dataset_fingerprint": manifest.samples_sha256,
        "consensus_train_count": len(consensus_train_rows),
        "consensus_eval_count": len(consensus_eval_rows),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "consensus_audit": consensus_audit[: min(200, len(consensus_audit))],
    }
    artifact_path, model_fingerprint = _write_versioned_json(
        output_dir,
        filename_prefix="swarm-student-model",
        payload=model_payload,
    )
    metrics_path, metrics_fingerprint = _write_versioned_json(
        output_dir,
        filename_prefix="swarm-student-metrics",
        payload=metrics_payload,
    )
    artifact_id = f"swarm-student-{model_fingerprint[:12]}"
    return SwarmStudentTrainingResult(
        artifact_id=artifact_id,
        schema_version=schema_version,
        artifact_path=artifact_path,
        metrics_path=metrics_path,
        model_fingerprint=model_fingerprint,
        metrics_fingerprint=metrics_fingerprint,
        consensus_train_count=len(consensus_train_rows),
        consensus_eval_count=len(consensus_eval_rows),
        metrics=metrics_payload,
    )
