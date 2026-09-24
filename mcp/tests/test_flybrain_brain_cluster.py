import os
import sys
import hashlib
import json
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_brain_cluster as fbc  # noqa: E402


class _StubExpert:
    def __init__(self, expert_id: str, confidence: float, provenance_refs: tuple[str, ...], *, fp_mode: str = "match"):
        self.expert_id = expert_id
        self.region = expert_id
        self._confidence = confidence
        self._refs = provenance_refs
        self._fp_mode = fp_mode

    def infer(self, task: fbc.ClusterTaskEnvelope, provenance: fbc.ReplayProvenanceEnvelope) -> fbc.ExpertOutput:
        if self._fp_mode == "mismatch":
            output_fp = "deadbeef"
        elif self._fp_mode == "match":
            output_fp = provenance.replay_fingerprint
        else:
            output_fp = ""
        return fbc.ExpertOutput(
            expert_id=self.expert_id,
            confidence=self._confidence,
            claims=[f"claim from {self.expert_id}"],
            artifacts={"replay_fingerprint": output_fp},
            provenance_refs=self._refs,
        )


def _task() -> fbc.ClusterTaskEnvelope:
    return fbc.ClusterTaskEnvelope(
        request_id="req-1",
        cluster_id="flybrain-cluster",
        objective="verify scoped claim",
        task_type="verification",
        risk_tier="high",
        inputs={"claim": "X"},
        router_features={"preferred_region": "provenance_expert"},
        constraints={"latency_tier": "interactive"},
    )


def _manifest_digest(manifest: dict[str, object]) -> str:
    canonical = json.loads(json.dumps(manifest))
    canonical["integrity"]["manifest_sha256"] = ""
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_artifact_bundle(
    root: Path,
    *,
    artifact_version: str = "2026.09.23",
    router_payload: dict[str, object] | None = None,
    experts_payload: dict[str, object] | None = None,
) -> Path:
    artifact_root = root / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    router_payload = router_payload or {
        "policy_version": "braincluster-router/v2",
        "routes_by_task_type": {"verification": ["r1"]},
    }
    experts_payload = experts_payload or {"experts": [{"expert_id": "r1", "region": "r1"}]}
    router_path = artifact_root / "router.json"
    experts_path = artifact_root / "experts.json"
    router_path.write_text(json.dumps(router_payload), encoding="utf-8")
    experts_path.write_text(json.dumps(experts_payload), encoding="utf-8")

    manifest = {
        "schema_version": fbc.BRAIN_CLUSTER_ARTIFACT_SCHEMA_VERSION,
        "artifact": {"id": "braincluster-expert-router", "version": artifact_version, "relative_root": "artifacts"},
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
    manifest["integrity"]["manifest_sha256"] = _manifest_digest(manifest)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_router_is_deterministic_for_same_input():
    router = fbc.DeterministicRegionRouter(
        routes_by_task_type={"verification": ["provenance_expert", "safety_expert"], "default": ["generalist"]}
    )
    first = router.plan(_task())
    second = router.plan(_task())
    assert first.primary == second.primary
    assert first.alternates == second.alternates
    assert first.topk_scores == second.topk_scores
    assert first.routing_entropy == second.routing_entropy


def test_provenance_envelope_fingerprint_is_stable():
    task = _task()
    first = fbc.ReplayProvenanceEnvelope.from_task(task, policy_version="p1", route_seed="s1")
    second = fbc.ReplayProvenanceEnvelope.from_task(task, policy_version="p1", route_seed="s1")
    assert first.replay_fingerprint == second.replay_fingerprint


def test_coordinator_accepts_when_all_gates_pass():
    router = fbc.DeterministicRegionRouter(routes_by_task_type={"verification": ["provenance_expert"]})
    experts = [_StubExpert("provenance_expert", 0.92, ("finding-1",))]
    gates = [fbc.ConfidenceGate(min_confidence=0.7), fbc.ProvenanceRefsGate(), fbc.ReplayFingerprintGate()]
    coordinator = fbc.BrainClusterCoordinator(router=router, experts=experts, gates=gates)

    result = coordinator.run(_task())
    assert result.decision == fbc.ClusterDecision.ACCEPT
    assert result.selected_expert == "provenance_expert"
    assert fbc.ClusterState.ACCEPTED in result.state_history
    assert result.final_state == fbc.ClusterState.FINALIZED


def test_coordinator_retries_to_alternate_when_provenance_missing():
    router = fbc.DeterministicRegionRouter(
        routes_by_task_type={"verification": ["provenance_expert", "safety_expert"]}
    )
    experts = [
        _StubExpert("provenance_expert", 0.91, ()),
        _StubExpert("safety_expert", 0.89, ("finding-2",)),
    ]
    gates = [fbc.ProvenanceRefsGate(on_missing=fbc.ClusterDecision.RETRY), fbc.ConfidenceGate(min_confidence=0.7)]
    coordinator = fbc.BrainClusterCoordinator(router=router, experts=experts, gates=gates, max_attempts=2)

    result = coordinator.run(_task())
    assert result.decision == fbc.ClusterDecision.ACCEPT
    assert result.selected_expert == "safety_expert"
    assert fbc.ClusterState.RETRY_EXPERT in result.state_history


def test_coordinator_fail_closed_on_replay_fingerprint_mismatch():
    router = fbc.DeterministicRegionRouter(routes_by_task_type={"verification": ["provenance_expert"]})
    experts = [_StubExpert("provenance_expert", 0.95, ("finding-1",), fp_mode="mismatch")]
    gates = [fbc.ReplayFingerprintGate()]
    coordinator = fbc.BrainClusterCoordinator(router=router, experts=experts, gates=gates)

    result = coordinator.run(_task())
    assert result.decision == fbc.ClusterDecision.FAIL_CLOSED
    assert result.gate_reason.startswith("replay fingerprint mismatch")


def test_coordinator_fail_closed_on_missing_replay_fingerprint():
    router = fbc.DeterministicRegionRouter(routes_by_task_type={"verification": ["provenance_expert"]})
    experts = [_StubExpert("provenance_expert", 0.95, ("finding-1",), fp_mode="missing")]
    gates = [fbc.ReplayFingerprintGate()]
    coordinator = fbc.BrainClusterCoordinator(router=router, experts=experts, gates=gates)

    result = coordinator.run(_task())
    assert result.decision == fbc.ClusterDecision.FAIL_CLOSED
    assert result.gate_reason.startswith("missing replay fingerprint")


def test_coordinator_fail_closed_on_low_confidence():
    router = fbc.DeterministicRegionRouter(routes_by_task_type={"verification": ["provenance_expert"]})
    experts = [_StubExpert("provenance_expert", 0.4, ("finding-1",))]
    gates = [fbc.ConfidenceGate(min_confidence=0.7)]
    coordinator = fbc.BrainClusterCoordinator(router=router, experts=experts, gates=gates)

    result = coordinator.run(_task())
    assert result.decision == fbc.ClusterDecision.FAIL_CLOSED
    assert fbc.ClusterState.FAIL_CLOSED in result.state_history


def test_selected_expert_is_none_when_all_candidates_unavailable():
    router = fbc.DeterministicRegionRouter(
        routes_by_task_type={"verification": ["missing_expert_a", "missing_expert_b"]}
    )
    coordinator = fbc.BrainClusterCoordinator(router=router, experts=[], gates=[], max_attempts=2)

    result = coordinator.run(_task())
    assert result.decision == fbc.ClusterDecision.FAIL_CLOSED
    assert result.selected_expert is None


def test_artifact_loader_accepts_valid_manifest(tmp_path):
    manifest_path = _write_artifact_bundle(tmp_path)
    loaded = fbc.load_brain_cluster_artifacts(manifest_path)
    assert loaded.manifest.schema_version == fbc.BRAIN_CLUSTER_ARTIFACT_SCHEMA_VERSION
    assert loaded.manifest.artifact_id == "braincluster-expert-router"
    assert loaded.manifest.artifact_version == "2026.09.23"
    assert loaded.router["policy_version"] == "braincluster-router/v2"
    assert loaded.experts["experts"][0]["expert_id"] == "r1"


def test_artifact_loader_rejects_missing_artifact_file(tmp_path):
    manifest_path = _write_artifact_bundle(tmp_path)
    (tmp_path / "artifacts" / "experts.json").unlink()
    with pytest.raises(fbc.BrainClusterArtifactError) as exc:
        fbc.load_brain_cluster_artifacts(manifest_path)
    assert exc.value.code == fbc.BrainClusterArtifactErrorCode.ARTIFACT_MISSING.value


def test_artifact_loader_rejects_bad_hash(tmp_path):
    manifest_path = _write_artifact_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["integrity"]["files"][0]["sha256"] = "0" * 64
    manifest["integrity"]["manifest_sha256"] = _manifest_digest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(fbc.BrainClusterArtifactError) as exc:
        fbc.load_brain_cluster_artifacts(manifest_path)
    assert exc.value.code == fbc.BrainClusterArtifactErrorCode.INTEGRITY_MISMATCH.value


def test_artifact_loader_rejects_bad_size(tmp_path):
    manifest_path = _write_artifact_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["integrity"]["files"][1]["size_bytes"] = int(manifest["integrity"]["files"][1]["size_bytes"]) + 7
    manifest["integrity"]["manifest_sha256"] = _manifest_digest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(fbc.BrainClusterArtifactError) as exc:
        fbc.load_brain_cluster_artifacts(manifest_path)
    assert exc.value.code == fbc.BrainClusterArtifactErrorCode.INTEGRITY_MISMATCH.value


def test_artifact_loader_rejects_malformed_manifest(tmp_path):
    manifest_path = _write_artifact_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["integrity"]["files"] = [{"logical_name": "router"}]
    manifest["integrity"]["manifest_sha256"] = _manifest_digest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(fbc.BrainClusterArtifactError) as exc:
        fbc.load_brain_cluster_artifacts(manifest_path)
    assert exc.value.code == fbc.BrainClusterArtifactErrorCode.MANIFEST_INVALID.value


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m["artifact"].update({"relative_root": "../escape"}),
        lambda m: m["integrity"]["files"][0].update({"relative_path": "../escape-router.json"}),
    ],
)
def test_artifact_loader_rejects_path_escape_attempts(tmp_path, mutate):
    manifest_path = _write_artifact_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest["integrity"]["manifest_sha256"] = _manifest_digest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(fbc.BrainClusterArtifactError) as exc:
        fbc.load_brain_cluster_artifacts(manifest_path)
    assert exc.value.code == fbc.BrainClusterArtifactErrorCode.PATH_ESCAPE.value


def test_promotion_happy_path_stage_promote_and_read(tmp_path):
    state_path = tmp_path / "promotion-state.json"
    manifest_path = _write_artifact_bundle(tmp_path / "v1", artifact_version="2026.09.23")

    staged = fbc.stage_brain_cluster_candidate_manifest(state_path, candidate_manifest_path=manifest_path)
    assert staged.schema_version == fbc.BRAIN_CLUSTER_PROMOTION_STATE_SCHEMA_VERSION
    assert staged.candidate is not None
    assert staged.promoted is None

    promoted = fbc.promote_brain_cluster_candidate(state_path)
    assert promoted.candidate is None
    assert promoted.promoted is not None
    assert promoted.promoted.artifact_version == "2026.09.23"
    assert promoted.previous_promoted is None

    read_back = fbc.read_brain_cluster_promotion_state(state_path)
    assert read_back.promoted is not None
    assert read_back.promoted.manifest_path == promoted.promoted.manifest_path


def test_promotion_state_rejects_previous_promoted_without_promoted():
    with pytest.raises(fbc.BrainClusterPromotionStateError) as exc:
        fbc.BrainClusterPromotionState(
            schema_version=fbc.BRAIN_CLUSTER_PROMOTION_STATE_SCHEMA_VERSION,
            updated_at="2026-09-24T00:00:00Z",
            previous_promoted=fbc.BrainClusterPromotionPointer(
                manifest_path="/home/rjmendez/development/loci/mcp/tests/fixtures/manifest.json",
                manifest_sha256="a" * 64,
                artifact_id="braincluster-expert-router",
                artifact_version="2026.09.23",
                recorded_at="2026-09-24T00:00:00Z",
            ),
        )
    assert exc.value.code == fbc.BrainClusterPromotionStateErrorCode.STATE_INVALID.value
    assert exc.value.message == "previous_promoted cannot be set without promoted."


def test_promotion_invalid_transition_when_candidate_matches_current_promoted(tmp_path):
    state_path = tmp_path / "promotion-state.json"
    manifest_path = _write_artifact_bundle(tmp_path / "v1", artifact_version="2026.09.23")

    fbc.stage_brain_cluster_candidate_manifest(state_path, candidate_manifest_path=manifest_path)
    fbc.promote_brain_cluster_candidate(state_path)
    fbc.stage_brain_cluster_candidate_manifest(state_path, candidate_manifest_path=manifest_path)

    with pytest.raises(fbc.BrainClusterPromotionStateError) as exc:
        fbc.promote_brain_cluster_candidate(state_path)
    assert exc.value.code == fbc.BrainClusterPromotionStateErrorCode.INVALID_TRANSITION.value


def test_promotion_missing_candidate_transition(tmp_path):
    state_path = tmp_path / "promotion-state.json"

    with pytest.raises(fbc.BrainClusterPromotionStateError) as exc:
        fbc.promote_brain_cluster_candidate(state_path)
    assert exc.value.code == fbc.BrainClusterPromotionStateErrorCode.CANDIDATE_MISSING.value


def test_promotion_fails_closed_when_validation_fails(tmp_path):
    state_path = tmp_path / "promotion-state.json"
    manifest_path = _write_artifact_bundle(tmp_path / "v1", artifact_version="2026.09.23")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["integrity"]["files"][0]["sha256"] = "f" * 64
    manifest["integrity"]["manifest_sha256"] = _manifest_digest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    fbc.stage_brain_cluster_candidate_manifest(state_path, candidate_manifest_path=manifest_path)
    with pytest.raises(fbc.BrainClusterPromotionStateError) as exc:
        fbc.promote_brain_cluster_candidate(state_path)
    assert exc.value.code == fbc.BrainClusterPromotionStateErrorCode.VALIDATION_FAILED.value


def test_promotion_rollback_to_prior_promoted_version(tmp_path):
    state_path = tmp_path / "promotion-state.json"
    manifest_v1 = _write_artifact_bundle(tmp_path / "v1", artifact_version="2026.09.23")
    manifest_v2 = _write_artifact_bundle(tmp_path / "v2", artifact_version="2026.09.24")

    fbc.stage_brain_cluster_candidate_manifest(state_path, candidate_manifest_path=manifest_v1)
    first = fbc.promote_brain_cluster_candidate(state_path)
    assert first.promoted is not None
    assert first.promoted.artifact_version == "2026.09.23"

    fbc.stage_brain_cluster_candidate_manifest(state_path, candidate_manifest_path=manifest_v2)
    second = fbc.promote_brain_cluster_candidate(state_path)
    assert second.promoted is not None
    assert second.promoted.artifact_version == "2026.09.24"
    assert second.previous_promoted is not None
    assert second.previous_promoted.artifact_version == "2026.09.23"

    rolled_back = fbc.rollback_brain_cluster_promoted(state_path)
    assert rolled_back.promoted is not None
    assert rolled_back.promoted.artifact_version == "2026.09.23"
    assert rolled_back.previous_promoted is not None
    assert rolled_back.previous_promoted.artifact_version == "2026.09.24"


def _shadow_fixtures() -> list[dict[str, object]]:
    return [
        {
            "fixture_id": "fx-1",
            "route_seed": "seed-a",
            "task": {
                "request_id": "req-1",
                "cluster_id": "flybrain-cluster",
                "objective": "verify alpha",
                "task_type": "verification",
                "risk_tier": "high",
                "inputs": {"claim": "alpha"},
                "router_features": {"preferred_region": "r1"},
            },
        },
        {
            "fixture_id": "fx-2",
            "route_seed": "seed-b",
            "task": {
                "request_id": "req-2",
                "cluster_id": "flybrain-cluster",
                "objective": "verify beta",
                "task_type": "verification",
                "risk_tier": "medium",
                "inputs": {"claim": "beta"},
                "router_features": {"preferred_region": "r2"},
            },
        },
    ]


def test_shadow_replay_report_is_deterministic(tmp_path):
    baseline_manifest = _write_artifact_bundle(
        tmp_path / "baseline",
        artifact_version="2026.09.23",
        router_payload={
            "policy_version": "braincluster-router/v2",
            "routes_by_task_type": {"verification": ["r1", "r2"]},
        },
        experts_payload={
            "experts": [
                {"expert_id": "r1", "region": "r1", "shadow_replay": {"confidence": 0.88}},
                {"expert_id": "r2", "region": "r2", "shadow_replay": {"confidence": 0.84}},
            ]
        },
    )
    candidate_manifest = _write_artifact_bundle(
        tmp_path / "candidate",
        artifact_version="2026.09.24",
        router_payload={
            "policy_version": "braincluster-router/v2",
            "routes_by_task_type": {"verification": ["r1", "r2"]},
        },
        experts_payload={
            "experts": [
                {"expert_id": "r1", "region": "r1", "shadow_replay": {"confidence": 0.88}},
                {"expert_id": "r2", "region": "r2", "shadow_replay": {"confidence": 0.84}},
            ]
        },
    )
    fixtures = _shadow_fixtures()

    first = fbc.run_brain_cluster_shadow_replay(
        fixtures,
        baseline_manifest_path=baseline_manifest,
        candidate_manifest_path=candidate_manifest,
    )
    second = fbc.run_brain_cluster_shadow_replay(
        fixtures,
        baseline_manifest_path=baseline_manifest,
        candidate_manifest_path=candidate_manifest,
    )
    assert first == second
    assert first["pass"] is True
    assert first["status"] == "pass"
    assert first["metrics"]["decision_match_rate"] == 1.0
    serialized = fbc.serialize_brain_cluster_shadow_replay_report(first)
    assert serialized.endswith("\n")
    assert json.loads(serialized) == first


def test_shadow_replay_threshold_failures_include_clear_reasons(tmp_path):
    baseline_manifest = _write_artifact_bundle(
        tmp_path / "baseline",
        artifact_version="2026.09.23",
        router_payload={
            "policy_version": "braincluster-router/v2",
            "routes_by_task_type": {"verification": ["r1", "r2"]},
        },
        experts_payload={
            "experts": [
                {"expert_id": "r1", "region": "r1", "shadow_replay": {"confidence": 0.94}},
                {"expert_id": "r2", "region": "r2", "shadow_replay": {"confidence": 0.92}},
            ]
        },
    )
    candidate_manifest = _write_artifact_bundle(
        tmp_path / "candidate",
        artifact_version="2026.09.24",
        router_payload={
            "policy_version": "braincluster-router/v3",
            "routes_by_task_type": {"verification": ["r1", "r2"]},
        },
        experts_payload={
            "experts": [
                {"expert_id": "r1", "region": "r1", "shadow_replay": {"confidence": 0.20}},
                {"expert_id": "r2", "region": "r2", "shadow_replay": {"confidence": 0.20}},
            ]
        },
    )

    report = fbc.run_brain_cluster_shadow_replay(
        _shadow_fixtures(),
        baseline_manifest_path=baseline_manifest,
        candidate_manifest_path=candidate_manifest,
        thresholds={
            "min_decision_match_rate": 1.0,
            "max_mean_abs_confidence_drift": 0.01,
            "max_fail_closed_rate_delta": 0.0,
            "max_mean_abs_routing_entropy_delta": 0.0,
            "max_selected_expert_concentration": 0.4,
            "max_selected_expert_concentration_delta": 0.0,
        },
    )
    assert report["pass"] is False
    assert report["status"] == "fail"
    assert report["exit_code"] == 1
    reasons = report["failure_reasons"]
    assert any("decision match rate below threshold" in reason for reason in reasons)
    assert any("confidence drift above threshold" in reason for reason in reasons)
    assert any("fail-closed rate delta above threshold" in reason for reason in reasons)
    assert any("candidate expert selection concentration above threshold" in reason for reason in reasons)
    assert any(flag["code"] == "expert_collapse_high" for flag in report["issue_flags"])


def test_shadow_replay_rejects_malformed_fixtures(tmp_path):
    baseline_manifest = _write_artifact_bundle(tmp_path / "baseline", artifact_version="2026.09.23")
    candidate_manifest = _write_artifact_bundle(tmp_path / "candidate", artifact_version="2026.09.24")
    malformed = [{"fixture_id": "broken", "task": {"cluster_id": "missing fields"}}]
    with pytest.raises(ValueError, match="fixture task missing required fields"):
        fbc.run_brain_cluster_shadow_replay(
            malformed,
            baseline_manifest_path=baseline_manifest,
            candidate_manifest_path=candidate_manifest,
        )
