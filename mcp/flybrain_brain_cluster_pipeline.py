from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import flybrain_brain_cluster as fbc
import flybrain_brain_cluster_training as fbct


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_samples(raw: Sequence[fbct.TrainingSample | Mapping[str, Any]] | Mapping[str, Any] | str | Path) -> list[fbct.TrainingSample]:
    if isinstance(raw, (str, bytes, Path)):
        payload = fbct._load_json_value(raw, label="samples")
    else:
        payload = raw
    rows = fbct._extract_items(payload, key="samples") if isinstance(payload, Mapping) else list(payload)
    samples = [fbct._normalize_training_sample(item) for item in rows]
    if not samples:
        raise ValueError("samples must be non-empty")
    return samples


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
    return {
        "schema_version": "braincluster-router-runtime/v1",
        "policy_version": policy_version,
        "routes_by_task_type": routes_by_task_type,
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
) -> dict[str, Any]:
    normalized_samples = _parse_samples(samples)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    state_file = Path(state_path)

    dataset_manifest = fbct.build_dataset_manifest(normalized_samples, split_seed=split_seed)
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
    )
    return result.as_dict()


def _parse_thresholds_file(raw: str | None) -> Mapping[str, Any] | None:
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
    parser.add_argument("--report", help="Optional report output path")
    args = parser.parse_args(argv)

    try:
        report = run_brain_cluster_p0_dry_run(
            args.samples,
            output_dir=args.output_dir,
            state_path=args.state_path,
            split_seed=args.split_seed,
            gate_thresholds=_parse_thresholds_file(args.gate_thresholds),
            shadow_thresholds=_parse_thresholds_file(args.shadow_thresholds),
            max_per_region=args.max_per_region,
            max_shadow_fixtures=args.max_shadow_fixtures,
            rollback_on_shadow_failure=not args.no_rollback_on_shadow_failure,
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
