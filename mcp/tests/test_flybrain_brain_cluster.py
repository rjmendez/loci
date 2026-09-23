import os
import sys

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


def test_coordinator_fail_closed_on_low_confidence():
    router = fbc.DeterministicRegionRouter(routes_by_task_type={"verification": ["provenance_expert"]})
    experts = [_StubExpert("provenance_expert", 0.4, ("finding-1",))]
    gates = [fbc.ConfidenceGate(min_confidence=0.7)]
    coordinator = fbc.BrainClusterCoordinator(router=router, experts=experts, gates=gates)

    result = coordinator.run(_task())
    assert result.decision == fbc.ClusterDecision.FAIL_CLOSED
    assert fbc.ClusterState.FAIL_CLOSED in result.state_history
