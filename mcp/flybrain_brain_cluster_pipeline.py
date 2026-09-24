from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import flybrain_brain_cluster as fbc
import flybrain_brain_cluster_baselines as fbb
import flybrain_brain_cluster_training as fbct


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_samples(raw: Sequence[fbct.TrainingSample | Mapping[str, Any]] | Mapping[str, Any] | str | Path) -> list[fbct.TrainingSample]:
    return _parse_samples_and_metadata(raw)[0]


def _parse_samples_and_metadata(
    raw: Sequence[fbct.TrainingSample | Mapping[str, Any]] | Mapping[str, Any] | str | Path,
) -> tuple[list[fbct.TrainingSample], dict[str, Any]]:
    """Samples plus the builder payload's top-level ``metadata`` (empty for bare lists)."""
    if isinstance(raw, (str, bytes, Path)):
        payload = fbct._load_json_value(raw, label="samples")
    else:
        payload = raw
    rows = fbct._extract_items(payload, key="samples") if isinstance(payload, Mapping) else list(payload)
    samples = [fbct._normalize_training_sample(item) for item in rows]
    if not samples:
        raise ValueError("samples must be non-empty")
    metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
    return samples, dict(metadata) if isinstance(metadata, Mapping) else {}


UNKNOWN_DATASET = "unknown"
# Per-sample ``metadata.dataset`` spellings used by builders -> registry symbol.
_SAMPLE_DATASET_ALIASES = {"flywire": "fw", "flywire783": "fw", "hemibrain": "hb"}
_SCHEMA_DATASET_RE = re.compile(r"^flybrain-([a-z0-9]+)-training-samples/v[0-9]+$")


def _resolve_dataset(
    samples: Sequence[fbct.TrainingSample],
    payload_metadata: Mapping[str, Any],
    explicit_symbol: str | None,
) -> dict[str, Any]:
    """Resolve ``{dataset_symbol, dataset_version, source}`` for a run (roadmap B3).

    Sources, in order: the explicit argument, ``payload.metadata.dataset_symbol``,
    the payload schema_version, then every sample's ``metadata.dataset``. Mixed
    datasets in one run fail closed; nothing resolvable gives ``unknown``, which
    release_prep refuses to calibrate.
    """
    from flybrain_dataset_registry import get_dataset, normalize_symbol

    candidates: list[tuple[str, str]] = []
    if explicit_symbol:
        candidates.append(("argument", normalize_symbol(explicit_symbol)))
    if payload_metadata.get("dataset_symbol"):
        candidates.append(("payload_metadata", normalize_symbol(str(payload_metadata["dataset_symbol"]))))
    schema_match = _SCHEMA_DATASET_RE.match(str(payload_metadata.get("schema_version") or ""))
    if schema_match:
        candidates.append(("payload_schema_version", normalize_symbol(schema_match.group(1))))
    sample_values = {
        str(sample.metadata.get("dataset", "")).strip().lower()
        for sample in samples
        if isinstance(sample.metadata, Mapping)
    }
    sample_values.discard("")
    if sample_values:
        mapped = {_SAMPLE_DATASET_ALIASES.get(value, value) for value in sample_values}
        if len(mapped) > 1:
            raise ValueError(f"samples mix datasets {sorted(mapped)}; runs must be single-dataset")
        if not all(str(sample.metadata.get("dataset", "")).strip() for sample in samples):
            raise ValueError("some samples carry metadata.dataset and some do not; runs must be single-dataset")
        candidates.append(("sample_metadata", normalize_symbol(next(iter(mapped)))))
    symbols = {symbol for _, symbol in candidates}
    if len(symbols) > 1:
        raise ValueError(f"dataset sources disagree: {candidates}")
    if not symbols:
        return {"dataset_symbol": UNKNOWN_DATASET, "dataset_version": None, "source": None}
    symbol = symbols.pop()
    versions = {
        str(sample.metadata.get("dataset_version")).strip()
        for sample in samples
        if isinstance(sample.metadata, Mapping) and sample.metadata.get("dataset_version")
    }
    if payload_metadata.get("dataset_version"):
        versions.add(str(payload_metadata["dataset_version"]).strip())
    if len(versions) > 1:
        raise ValueError(f"samples mix dataset versions {sorted(versions)}; runs must be single-version")
    pinned = get_dataset(symbol).pinned_version
    version = versions.pop() if versions else pinned
    if pinned and version != pinned:
        raise ValueError(f"dataset version {version!r} does not match the registry pin {pinned!r} for {symbol}")
    return {"dataset_symbol": symbol, "dataset_version": version, "source": candidates[0][0]}


def _split_group_keys_for(dataset_symbol: str) -> tuple[str, ...]:
    from flybrain_dataset_registry import split_group_keys

    keys = [fbct.GENERIC_SPLIT_GROUP_KEY]
    if dataset_symbol != UNKNOWN_DATASET:
        keys.extend(split_group_keys(dataset_symbol))
    return tuple(dict.fromkeys(keys))


def _heldout_ids(manifest: fbct.DatasetManifest) -> tuple[str, tuple[str, ...]]:
    if manifest.test_ids:
        return "test", tuple(manifest.test_ids)
    return "val", tuple(manifest.val_ids)


def _trivial_baseline_report(
    samples: Sequence[fbct.TrainingSample],
    *,
    manifest: fbct.DatasetManifest,
    predictions: Sequence[fbct.PredictionRecord],
    config: fbb.BaselineGateConfig | Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Model vs trivial rules on the held-out split (test, else val). Abstentions count as wrong."""
    sample_by_id = {sample.sample_id: sample for sample in samples}
    split_name, heldout = _heldout_ids(manifest)
    train_rows = [(sample_by_id[i].input_text, sample_by_id[i].expected_label) for i in manifest.train_ids]
    heldout_rows = [(sample_by_id[i].input_text, sample_by_id[i].expected_label) for i in heldout]
    pred_by_id = {record.sample_id: record for record in predictions}
    model_accuracy: float | None = None
    if heldout:
        correct = 0
        for sample_id in heldout:
            record = pred_by_id.get(sample_id)
            if record is not None and not record.abstained and record.predicted_label == sample_by_id[sample_id].expected_label:
                correct += 1
        model_accuracy = correct / float(len(heldout))
    baselines = fbb.evaluate_trivial_baselines(train_rows, heldout_rows) if train_rows and heldout_rows else None
    report = fbb.trivial_baseline_gate(
        model_heldout_accuracy=model_accuracy,
        baselines=baselines,
        heldout_count=len(heldout),
        config=config,
    )
    report["heldout_split"] = split_name
    return report


def _derive_router_samples(samples: Sequence[fbct.TrainingSample]) -> list[fbct.RouterTrainingSample]:
    router_samples: list[fbct.RouterTrainingSample] = []
    for sample in sorted(samples, key=lambda s: s.sample_id):
        task_type = str(sample.metadata.get("task_type", "verification")).strip() or "verification"
        risk_tier = str(sample.metadata.get("risk_tier", "medium")).strip() or "medium"
        expert_id = f"{sample.region_id}_expert"
        router_samples.append(
            fbct.RouterTrainingSample(
                sample_id=f"route-{sample.sample_id}",
                task_type=task_type,
                risk_tier=risk_tier,
                preferred_region=expert_id,
                expected_expert=expert_id,
                metadata={"region_id": sample.region_id},
            )
        )
    return router_samples


def _build_router_runtime_payload(
    router_samples: Sequence[fbct.RouterTrainingSample],
    *,
    policy_version: str,
    model_fingerprint: str,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, int]] = {}
    for row in router_samples:
        grouped.setdefault(row.task_type, {})
        current = grouped[row.task_type]
        current[row.expected_expert] = current.get(row.expected_expert, 0) + 1
    routes_by_task_type: dict[str, list[str]] = {}
    for task_type in sorted(grouped):
        counts = grouped[task_type]
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        routes_by_task_type[task_type] = [expert_id for expert_id, _ in ordered]
    if "default" not in routes_by_task_type:
        all_experts = sorted({row.expected_expert for row in router_samples})
        routes_by_task_type["default"] = all_experts or ["generalist"]
    default_route = routes_by_task_type.get("default", [])
    fanout_k = max(1, min(3, len(default_route)))
    return {
        "schema_version": "braincluster-router-runtime/v1",
        "policy_version": policy_version,
        "routes_by_task_type": routes_by_task_type,
        "swarm_policy": {
            "enabled": True,
            "execution_mode": "parallel_fanout",
            "fanout_k": fanout_k,
            "consensus": "gated_majority_then_confidence",
        },
        "training_artifact": {"model_fingerprint": model_fingerprint},
    }


def _build_experts_runtime_payload(
    *,
    manifest: fbct.DatasetManifest,
    expert_results: Mapping[str, fbct.RegionExpertTrainingResult],
    shadow_overrides: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for region_id in sorted(manifest.region_counts):
        expert_id = f"{region_id}_expert"
        result = expert_results[region_id]
        val_metrics = result.metrics.get("val_metrics", {}) if isinstance(result.metrics, Mapping) else {}
        confidence = float(val_metrics.get("accuracy", 0.82))
        confidence = max(0.0, min(1.0, confidence if confidence > 0.0 else 0.82))
        shadow_behavior = {
            "confidence": confidence,
            "provenance_refs": [f"dataset:{manifest.manifest_id}:{region_id}"],
            "replay_fingerprint_mode": "match",
        }
        if shadow_overrides and expert_id in shadow_overrides:
            shadow_behavior.update(dict(shadow_overrides[expert_id]))
        rows.append(
            {
                "expert_id": expert_id,
                "region": region_id,
                "training_artifact": {
                    "model_fingerprint": result.model_fingerprint,
                    "metrics_fingerprint": result.metrics_fingerprint,
                },
                "shadow_replay": shadow_behavior,
            }
        )
    return {
        "schema_version": "braincluster-experts-runtime/v1",
        "experts": rows,
    }


def _write_brain_cluster_manifest(
    *,
    bundle_dir: Path,
    artifact_id: str,
    artifact_version: str,
    router_payload: Mapping[str, Any],
    experts_payload: Mapping[str, Any],
) -> Path:
    artifact_root = bundle_dir / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    router_path = artifact_root / "router.json"
    experts_path = artifact_root / "experts.json"
    router_path.write_text(_stable_json(dict(router_payload)) + "\n", encoding="utf-8")
    experts_path.write_text(_stable_json(dict(experts_payload)) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": fbc.BRAIN_CLUSTER_ARTIFACT_SCHEMA_VERSION,
        "artifact": {
            "id": artifact_id,
            "version": artifact_version,
            "relative_root": "artifacts",
        },
        "integrity": {
            "manifest_sha256": "",
            "files": [
                {
                    "logical_name": "router",
                    "relative_path": "router.json",
                    "sha256": hashlib.sha256(router_path.read_bytes()).hexdigest(),
                    "size_bytes": router_path.stat().st_size,
                },
                {
                    "logical_name": "experts",
                    "relative_path": "experts.json",
                    "sha256": hashlib.sha256(experts_path.read_bytes()).hexdigest(),
                    "size_bytes": experts_path.stat().st_size,
                },
            ],
        },
    }
    canonical = json.loads(json.dumps(manifest))
    canonical["integrity"]["manifest_sha256"] = ""
    manifest["integrity"]["manifest_sha256"] = _sha256_hex(_stable_json(canonical))

    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _build_gate_predictions(
    samples: Sequence[fbct.TrainingSample],
    *,
    manifest: fbct.DatasetManifest,
    expert_results: Mapping[str, fbct.RegionExpertTrainingResult],
) -> list[fbct.PredictionRecord]:
    sample_by_id = {sample.sample_id: sample for sample in samples}
    model_by_region: dict[str, Mapping[str, Any]] = {}
    for region_id, result in expert_results.items():
        payload = json.loads(Path(result.artifact_path).read_text(encoding="utf-8"))
        model_by_region[region_id] = payload["model"]

    predictions: list[fbct.PredictionRecord] = []
    for sample in sorted(samples, key=lambda row: row.sample_id):
        model = model_by_region.get(sample.region_id)
        if model is None:
            predictions.append(
                fbct.PredictionRecord(
                    sample_id=sample.sample_id,
                    predicted_label="abstain",
                    predicted_confidence=0.0,
                    abstained=True,
                )
            )
            continue
        label, confidence = fbct._predict_multinomial_nb(model, sample.input_text)
        predictions.append(
            fbct.PredictionRecord(
                sample_id=sample.sample_id,
                predicted_label=label,
                predicted_confidence=confidence,
                abstained=False,
            )
        )
    return predictions


def _build_shadow_fixtures(
    samples: Sequence[fbct.TrainingSample],
    *,
    manifest: fbct.DatasetManifest,
    max_fixtures: int,
) -> list[dict[str, Any]]:
    sample_by_id = {sample.sample_id: sample for sample in samples}
    fixture_ids = list(manifest.test_ids) or list(manifest.val_ids) or list(manifest.train_ids)
    rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(fixture_ids[: max(1, max_fixtures)]):
        sample = sample_by_id[sample_id]
        expert_id = f"{sample.region_id}_expert"
        rows.append(
            {
                "fixture_id": f"fx-{index + 1:04d}-{sample.sample_id}",
                "route_seed": f"{manifest.split_seed}:{sample.sample_id}",
                "task": {
                    "request_id": f"req-{sample.sample_id}",
                    "cluster_id": "flybrain-cluster",
                    "objective": f"validate sample {sample.sample_id}",
                    "task_type": str(sample.metadata.get("task_type", "verification")) or "verification",
                    "risk_tier": str(sample.metadata.get("risk_tier", "medium")) or "medium",
                    "inputs": {"sample_id": sample.sample_id, "expected_label": sample.expected_label},
                    "router_features": {"preferred_region": expert_id},
                    "constraints": {"provenance_required": True},
                },
            }
        )
    return rows


@dataclass(frozen=True)
class BrainClusterDryRunResult:
    schema_version: str
    pass_gate: bool
    pass_shadow: bool
    promoted: bool
    rolled_back: bool
    artifact_manifest_path: str
    artifact_manifest_sha256: str
    gate_report: Mapping[str, Any]
    shadow_report: Mapping[str, Any]
    promotion_state: Mapping[str, Any]
    dataset_manifest: Mapping[str, Any]
    region_artifacts: Mapping[str, Any]
    router_artifact: Mapping[str, Any]
    swarm_student_artifact: Mapping[str, Any]
    trivial_baseline: Mapping[str, Any] = field(default_factory=dict)
    dataset: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        status = "pass" if (self.pass_gate and self.pass_shadow and self.promoted and not self.rolled_back) else "fail"
        return {
            "schema_version": self.schema_version,
            "status": status,
            "pass": status == "pass",
            "pass_gate": self.pass_gate,
            "pass_shadow": self.pass_shadow,
            "promoted": self.promoted,
            "rolled_back": self.rolled_back,
            "artifact_manifest_path": self.artifact_manifest_path,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "gate_report": dict(self.gate_report),
            "shadow_report": dict(self.shadow_report),
            "promotion_state": dict(self.promotion_state),
            "dataset_manifest": dict(self.dataset_manifest),
            "region_artifacts": dict(self.region_artifacts),
            "router_artifact": dict(self.router_artifact),
            "swarm_student_artifact": dict(self.swarm_student_artifact),
            "trivial_baseline": dict(self.trivial_baseline),
            "dataset": dict(self.dataset),
        }


def run_brain_cluster_p0_dry_run(
    samples: Sequence[fbct.TrainingSample | Mapping[str, Any]] | Mapping[str, Any] | str | Path,
    *,
    output_dir: str | Path,
    state_path: str | Path,
    split_seed: str,
    gate_thresholds: fbct.PromotionGateThresholds | Mapping[str, Any] | None = None,
    shadow_thresholds: fbc.ShadowReplayThresholds | Mapping[str, Any] | None = None,
    max_per_region: int = 20,
    max_shadow_fixtures: int = 50,
    rollback_on_shadow_failure: bool = True,
    shadow_fixtures: Sequence[Mapping[str, Any]] | None = None,
    candidate_shadow_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    dataset_symbol: str | None = None,
    baseline_gate: fbb.BaselineGateConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_samples, payload_metadata = _parse_samples_and_metadata(samples)
    dataset_info = _resolve_dataset(normalized_samples, payload_metadata, dataset_symbol)
    baseline_config = fbb.BaselineGateConfig.from_value(baseline_gate)
    label_counts: dict[str, int] = {}
    for sample in normalized_samples:
        label_counts[sample.expected_label] = label_counts.get(sample.expected_label, 0) + 1
    objective = str(payload_metadata.get("objective", "")).strip()
    if not objective:
        objective = str(normalized_samples[0].metadata.get("objective", "")).strip() if normalized_samples else ""
    if not objective:
        if all(label.startswith("dominant_") for label in label_counts):
            objective = "neurotransmitter_dominance"
        elif {"high_connectivity", "baseline_connectivity"}.issuperset(set(label_counts)):
            objective = "connectivity_tier"
        else:
            objective = "custom"
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    state_file = Path(state_path)

    group_keys = _split_group_keys_for(dataset_info["dataset_symbol"])
    dataset_manifest = fbct.build_dataset_manifest(
        normalized_samples,
        split_seed=split_seed,
        notes={
            "objective": objective,
            "dataset_symbol": dataset_info["dataset_symbol"],
            "dataset_version": dataset_info["dataset_version"],
            "label_counts": dict(sorted(label_counts.items())),
        },
        group_keys=group_keys,
    )
    fbct.assert_grouped_split(normalized_samples, dataset_manifest, group_keys=group_keys)
    (output_root / "dataset-manifest.json").write_text(
        json.dumps(dataset_manifest.as_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    region_artifacts: dict[str, fbct.RegionExpertTrainingResult] = {}
    region_dir = output_root / "region-experts"
    for region_id in sorted(dataset_manifest.region_counts):
        region_artifacts[region_id] = fbct.train_region_expert_from_manifest(
            normalized_samples,
            manifest=dataset_manifest,
            region_id=region_id,
            output_dir=region_dir,
        )

    router_samples = _derive_router_samples(normalized_samples)
    router_result = fbct.train_router_from_labeled_data(
        router_samples,
        split_seed=split_seed,
        output_dir=output_root / "router",
    )
    swarm_student_result = fbct.train_swarm_consensus_student(
        normalized_samples,
        manifest=dataset_manifest,
        expert_results=region_artifacts,
        output_dir=output_root / "swarm-student",
        min_vote_share=0.5,
        min_mean_confidence=0.0,
        min_consensus_samples=max(1, min(10, len(dataset_manifest.train_ids) + len(dataset_manifest.val_ids))),
    )

    artifact_version = f"{dataset_manifest.manifest_id}-{router_result.model_fingerprint[:12]}"
    bundle_dir = output_root / "artifact-bundle"
    router_payload = _build_router_runtime_payload(
        router_samples,
        policy_version=f"braincluster-router-runtime/{router_result.model_fingerprint[:12]}",
        model_fingerprint=router_result.model_fingerprint,
    )
    baseline_experts_payload = _build_experts_runtime_payload(
        manifest=dataset_manifest,
        expert_results=region_artifacts,
        shadow_overrides=None,
    )
    experts_payload = _build_experts_runtime_payload(
        manifest=dataset_manifest,
        expert_results=region_artifacts,
        shadow_overrides=candidate_shadow_overrides,
    )
    baseline_manifest_candidate_path = _write_brain_cluster_manifest(
        bundle_dir=output_root / "baseline-artifact-bundle",
        artifact_id="braincluster-expert-router",
        artifact_version=f"{artifact_version}-baseline",
        router_payload=router_payload,
        experts_payload=baseline_experts_payload,
    )
    candidate_manifest_path = _write_brain_cluster_manifest(
        bundle_dir=bundle_dir,
        artifact_id="braincluster-expert-router",
        artifact_version=artifact_version,
        router_payload=router_payload,
        experts_payload=experts_payload,
    )

    # Fail-closed validation of generated candidate artifacts before state transitions.
    candidate_bundle = fbc.load_brain_cluster_artifacts(candidate_manifest_path)
    fbc.stage_brain_cluster_candidate_manifest(state_file, candidate_manifest_path=candidate_manifest_path)

    predictions = _build_gate_predictions(normalized_samples, manifest=dataset_manifest, expert_results=region_artifacts)
    gate_thresholds = gate_thresholds or fbct.PromotionGateThresholds(
        min_accuracy=0.8,
        max_abstain_rate=0.2,
        max_confidence_calibration_error=0.2,
        min_samples=30,
    )
    gate_report = fbct.run_golden_set_gate(
        normalized_samples,
        predictions,
        thresholds=gate_thresholds,
        max_per_region=max_per_region,
    )
    # AC6: the promotion gate also requires beating the trivial rules on held-out data.
    trivial_baseline = _trivial_baseline_report(
        normalized_samples,
        manifest=dataset_manifest,
        predictions=predictions,
        config=baseline_config,
    )
    (output_root / "trivial-baseline-report.json").write_text(
        json.dumps(trivial_baseline, indent=2, sort_keys=True), encoding="utf-8"
    )
    gate_report = dict(gate_report)
    gate_report["trivial_baseline"] = {
        key: trivial_baseline[key]
        for key in ("pass", "model_heldout_accuracy", "best_trivial_rule", "best_trivial_accuracy",
                    "required_accuracy", "heldout_count", "heldout_split", "config")
    }
    if not trivial_baseline["pass"]:
        gate_report["failure_reasons"] = list(gate_report.get("failure_reasons", [])) + list(
            trivial_baseline["failure_reasons"]
        )
        gate_report["pass"] = False
        gate_report["status"] = "fail"
        gate_report["exit_code"] = 1
    pass_gate = bool(gate_report.get("pass", False))

    baseline_manifest_path: str | None = None
    prior_state = fbc.read_brain_cluster_promotion_state(state_file)
    if prior_state.promoted:
        baseline_manifest_path = prior_state.promoted.manifest_path
    if baseline_manifest_path is None:
        baseline_manifest_path = str(baseline_manifest_candidate_path)

    fixtures_payload = list(shadow_fixtures) if shadow_fixtures is not None else _build_shadow_fixtures(
        normalized_samples,
        manifest=dataset_manifest,
        max_fixtures=max_shadow_fixtures,
    )
    shadow_report = fbc.run_brain_cluster_shadow_replay(
        fixtures_payload,
        baseline_manifest_path=baseline_manifest_path,
        candidate_manifest_path=str(candidate_manifest_path),
        thresholds=shadow_thresholds,
    )
    pass_shadow = bool(shadow_report.get("pass", False))

    promoted = False
    rolled_back = False
    if pass_gate and pass_shadow:
        promoted_state = fbc.promote_brain_cluster_candidate(state_file)
        promoted = promoted_state.promoted is not None and promoted_state.promoted.manifest_sha256 == candidate_bundle.manifest.manifest_sha256
    elif rollback_on_shadow_failure and prior_state.promoted is not None and prior_state.previous_promoted is not None:
        rollback_state = fbc.rollback_brain_cluster_promoted(state_file)
        rolled_back = rollback_state.promoted is not None

    final_state = fbc.read_brain_cluster_promotion_state(state_file)
    result = BrainClusterDryRunResult(
        schema_version="braincluster-p0-dry-run/v1",
        pass_gate=pass_gate,
        pass_shadow=pass_shadow,
        promoted=promoted,
        rolled_back=rolled_back,
        artifact_manifest_path=str(candidate_manifest_path),
        artifact_manifest_sha256=candidate_bundle.manifest.manifest_sha256,
        gate_report=gate_report,
        shadow_report=shadow_report,
        promotion_state=final_state.as_dict(),
        dataset_manifest=dataset_manifest.as_dict(),
        region_artifacts={
            region_id: {
                "artifact_id": result.artifact_id,
                "artifact_path": result.artifact_path,
                "metrics_path": result.metrics_path,
                "model_fingerprint": result.model_fingerprint,
                "metrics_fingerprint": result.metrics_fingerprint,
                "train_count": result.train_count,
                "val_count": result.val_count,
            }
            for region_id, result in sorted(region_artifacts.items())
        },
        router_artifact={
            "artifact_id": router_result.artifact_id,
            "artifact_path": router_result.artifact_path,
            "metrics_path": router_result.metrics_path,
            "model_fingerprint": router_result.model_fingerprint,
            "metrics_fingerprint": router_result.metrics_fingerprint,
            "train_count": router_result.train_count,
            "val_count": router_result.val_count,
        },
        swarm_student_artifact={
            "artifact_id": swarm_student_result.artifact_id,
            "artifact_path": swarm_student_result.artifact_path,
            "metrics_path": swarm_student_result.metrics_path,
            "model_fingerprint": swarm_student_result.model_fingerprint,
            "metrics_fingerprint": swarm_student_result.metrics_fingerprint,
            "consensus_train_count": swarm_student_result.consensus_train_count,
            "consensus_eval_count": swarm_student_result.consensus_eval_count,
        },
        trivial_baseline=trivial_baseline,
        dataset=dataset_info,
    )
    return result.as_dict()


def _parse_thresholds_file(raw: str | None, *, section: str) -> Mapping[str, Any] | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("{"):
        payload = json.loads(text)
    else:
        payload = json.loads(Path(text).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("thresholds payload must be a JSON object")
    if section in payload and isinstance(payload.get(section), Mapping):
        return dict(payload[section])
    thresholds = payload.get("thresholds")
    if isinstance(thresholds, Mapping):
        key = "gate" if section == "gate" else "shadow_replay"
        nested = thresholds.get(key)
        if isinstance(nested, Mapping):
            return dict(nested)
    return dict(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic brain-cluster P0 train/gate/shadow/promotion dry run.")
    parser.add_argument("--samples", required=True, help="Path to JSON payload containing samples[]")
    parser.add_argument("--output-dir", required=True, help="Directory for generated artifacts and reports")
    parser.add_argument("--state-path", required=True, help="Promotion-state JSON path")
    parser.add_argument("--split-seed", required=True, help="Deterministic split seed")
    parser.add_argument("--gate-thresholds", help="JSON string or path for golden-set gate thresholds")
    parser.add_argument("--shadow-thresholds", help="JSON string or path for shadow replay thresholds")
    parser.add_argument("--max-per-region", type=int, default=20)
    parser.add_argument("--max-shadow-fixtures", type=int, default=50)
    parser.add_argument("--no-rollback-on-shadow-failure", action="store_true")
    parser.add_argument("--dataset", help="Dataset symbol (default: read from the samples payload)")
    parser.add_argument("--baseline-margin", type=float, default=fbb.BaselineGateConfig().min_margin,
                        help="Held-out accuracy the model must add over the best trivial rule")
    parser.add_argument("--report", help="Optional report output path")
    args = parser.parse_args(argv)

    try:
        report = run_brain_cluster_p0_dry_run(
            args.samples,
            output_dir=args.output_dir,
            state_path=args.state_path,
            split_seed=args.split_seed,
            gate_thresholds=_parse_thresholds_file(args.gate_thresholds, section="gate"),
            shadow_thresholds=_parse_thresholds_file(args.shadow_thresholds, section="shadow"),
            max_per_region=args.max_per_region,
            max_shadow_fixtures=args.max_shadow_fixtures,
            rollback_on_shadow_failure=not args.no_rollback_on_shadow_failure,
            dataset_symbol=args.dataset,
            baseline_gate={"min_margin": args.baseline_margin},
        )
    except (ValueError, TypeError, KeyError, json.JSONDecodeError, fbc.BrainClusterArtifactError, fbc.BrainClusterPromotionStateError) as exc:
        payload = {
            "schema_version": "braincluster-p0-dry-run/v1",
            "status": "error",
            "pass": False,
            "error": str(exc),
        }
        output = _stable_json(payload) + "\n"
        if args.report:
            Path(args.report).write_text(output, encoding="utf-8")
        print(output, end="")
        return 2

    output = _stable_json(report) + "\n"
    if args.report:
        Path(args.report).write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if report.get("pass", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
