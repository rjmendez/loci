import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_brain_cluster_pipeline as fbcp  # noqa: E402


def _sample_payload() -> dict[str, list[dict[str, object]]]:
    samples = []
    for i in range(1, 41):
        region = "grounding" if i % 2 == 0 else "provenance"
        label = "accept" if i % 4 else "reject"
        samples.append(
            {
                "sample_id": f"s{i:03d}",
                "region_id": region,
                "input_text": f"sample text {i} for {region}",
                "expected_label": label,
                "expected_confidence": 0.82 if label == "accept" else 0.68,
                "provenance_refs": [f"finding-{i}"],
                "metadata": {
                    "task_type": "verification" if i % 3 else "grounding",
                    "risk_tier": "high" if i % 5 == 0 else "medium",
                },
            }
        )
    return {"samples": samples}


def _run(tmp_path: Path, *, split_seed: str, **kwargs: object) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    samples_path = tmp_path / "samples.json"
    samples_path.write_text(__import__("json").dumps(_sample_payload()), encoding="utf-8")
    return fbcp.run_brain_cluster_p0_dry_run(
        str(samples_path),
        output_dir=tmp_path / "out",
        state_path=tmp_path / "state" / "promotion-state.json",
        split_seed=split_seed,
        **kwargs,
    )


def test_p0_dry_run_executes_end_to_end_and_promotes(tmp_path):
    report = _run(tmp_path, split_seed="seed-e2e")
    assert report["schema_version"] == "braincluster-p0-dry-run/v1"
    assert report["pass_gate"] is True
    assert report["pass_shadow"] is True
    assert report["promoted"] is True
    assert report["rolled_back"] is False
    assert report["promotion_state"]["promoted"] is not None
    assert report["dataset_manifest"]["notes"]["objective"] == "custom"
    assert report["dataset_manifest"]["notes"]["label_counts"] == {"accept": 30, "reject": 10}
    assert report["swarm_student_artifact"]["consensus_train_count"] > 0
    assert Path(report["artifact_manifest_path"]).exists()
    assert report["gate_report"]["exit_code"] == 0
    assert report["shadow_report"]["exit_code"] == 0


def test_router_runtime_payload_includes_parallel_swarm_policy():
    samples = _sample_payload()["samples"]
    normalized = [fbcp.fbct._normalize_training_sample(item) for item in samples]
    router_samples = fbcp._derive_router_samples(normalized)
    payload = fbcp._build_router_runtime_payload(
        router_samples,
        policy_version="braincluster-router-runtime/test",
        model_fingerprint="abc123",
    )
    swarm = payload["swarm_policy"]
    assert swarm["enabled"] is True
    assert swarm["execution_mode"] == "parallel_fanout"
    assert swarm["fanout_k"] >= 1
    assert swarm["consensus"] == "gated_majority_then_confidence"


def test_p0_dry_run_is_deterministic_for_same_seed(tmp_path):
    first = _run(tmp_path / "first", split_seed="seed-determinism")
    second = _run(tmp_path / "second", split_seed="seed-determinism")

    assert first["dataset_manifest"]["manifest_id"] == second["dataset_manifest"]["manifest_id"]
    assert first["dataset_manifest"]["samples_sha256"] == second["dataset_manifest"]["samples_sha256"]
    assert first["dataset_manifest"]["train_ids"] == second["dataset_manifest"]["train_ids"]
    assert first["dataset_manifest"]["val_ids"] == second["dataset_manifest"]["val_ids"]
    assert first["dataset_manifest"]["test_ids"] == second["dataset_manifest"]["test_ids"]
    assert first["artifact_manifest_sha256"] == second["artifact_manifest_sha256"]
    assert first["gate_report"]["metrics"] == second["gate_report"]["metrics"]
    assert first["shadow_report"]["metrics"] == second["shadow_report"]["metrics"]
    assert first["router_artifact"]["model_fingerprint"] == second["router_artifact"]["model_fingerprint"]
    assert first["router_artifact"]["metrics_fingerprint"] == second["router_artifact"]["metrics_fingerprint"]
    assert first["swarm_student_artifact"]["model_fingerprint"] == second["swarm_student_artifact"]["model_fingerprint"]
    assert first["swarm_student_artifact"]["metrics_fingerprint"] == second["swarm_student_artifact"]["metrics_fingerprint"]

    first_regions = first["region_artifacts"]
    second_regions = second["region_artifacts"]
    assert sorted(first_regions) == sorted(second_regions)
    for region_id in sorted(first_regions):
        assert first_regions[region_id]["model_fingerprint"] == second_regions[region_id]["model_fingerprint"]
        assert first_regions[region_id]["metrics_fingerprint"] == second_regions[region_id]["metrics_fingerprint"]


def test_p0_dry_run_failure_injection_missing_replay_fingerprint_triggers_fail_closed(tmp_path):
    report = _run(
        tmp_path,
        split_seed="seed-failure",
        candidate_shadow_overrides={
            "grounding_expert": {"replay_fingerprint_mode": "missing"},
            "provenance_expert": {"replay_fingerprint_mode": "missing"},
        },
    )
    assert report["pass_shadow"] is False
    assert report["promoted"] is False
    assert report["shadow_report"]["exit_code"] == 1
    assert any("fail-closed rate delta above threshold" in reason for reason in report["shadow_report"]["failure_reasons"])


def test_p0_dry_run_rejects_malformed_shadow_fixture_payload(tmp_path):
    with pytest.raises(ValueError, match="fixture task missing required fields"):
        _run(
            tmp_path,
            split_seed="seed-bad-fixture",
            shadow_fixtures=[{"fixture_id": "broken", "task": {"cluster_id": "missing-fields"}}],
        )


def test_parse_thresholds_file_supports_calibration_bundle_shape(tmp_path):
    payload = {
        "schema_version": "braincluster-threshold-calibration/v1",
        "thresholds": {
            "gate": {"min_accuracy": 0.71},
            "shadow_replay": {"min_decision_match_rate": 0.88},
        },
    }
    path = tmp_path / "thresholds.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    gate = fbcp._parse_thresholds_file(str(path), section="gate")
    shadow = fbcp._parse_thresholds_file(str(path), section="shadow")
    assert gate["min_accuracy"] == 0.71
    assert shadow["min_decision_match_rate"] == 0.88
