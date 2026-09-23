from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

from replay_fingerprint import flybrain_replay_fingerprint

CLUSTER_TOOL_NAME = "flybrain_brain_cluster"


class ClusterDecision(str, Enum):
    ACCEPT = "accept"
    RETRY = "retry"
    ESCALATE = "escalate"
    FAIL_CLOSED = "fail_closed"


class ClusterState(str, Enum):
    RECEIVED = "received"
    ROUTED = "routed"
    RUNNING_EXPERT = "running_expert"
    GATED = "gated"
    ACCEPTED = "accepted"
    RETRY_EXPERT = "retry_expert"
    ESCALATED = "escalated"
    FAIL_CLOSED = "fail_closed"
    FINALIZED = "finalized"


@dataclass(frozen=True)
class ClusterTaskEnvelope:
    request_id: str
    cluster_id: str
    objective: str
    task_type: str
    risk_tier: str
    inputs: Mapping[str, Any]
    router_features: Mapping[str, Any] = field(default_factory=dict)
    constraints: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "cluster_id": self.cluster_id,
            "objective": self.objective,
            "task_type": self.task_type,
            "risk_tier": self.risk_tier,
            "inputs": dict(self.inputs),
            "router_features": dict(self.router_features),
            "constraints": dict(self.constraints),
        }


@dataclass(frozen=True)
class ReplayProvenanceEnvelope:
    request_id: str
    cluster_id: str
    policy_version: str
    route_seed: str
    replay_fingerprint: str
    deterministic: bool = True
    degraded: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_task(
        cls,
        task: ClusterTaskEnvelope,
        *,
        policy_version: str,
        route_seed: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ReplayProvenanceEnvelope":
        request = {
            "request_id": task.request_id,
            "objective": task.objective,
            "task_type": task.task_type,
            "risk_tier": task.risk_tier,
            "inputs": dict(task.inputs),
            "router_features": dict(task.router_features),
            "constraints": dict(task.constraints),
            "route_seed": route_seed,
            "policy_version": policy_version,
        }
        dataset_scope = {"cluster_id": task.cluster_id, "risk_tier": task.risk_tier}
        fingerprint = flybrain_replay_fingerprint(CLUSTER_TOOL_NAME, request, dataset_scope)
        return cls(
            request_id=task.request_id,
            cluster_id=task.cluster_id,
            policy_version=policy_version,
            route_seed=route_seed,
            replay_fingerprint=fingerprint,
            deterministic=True,
            degraded=False,
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True)
class RoutePlan:
    primary: str
    alternates: tuple[str, ...]
    topk_scores: Mapping[str, float]
    routing_entropy: float
    policy_version: str

    def ordered_experts(self) -> list[str]:
        return [self.primary, *self.alternates]


@dataclass(frozen=True)
class ExpertOutput:
    expert_id: str
    confidence: float
    claims: Sequence[str]
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    provenance_refs: Sequence[str] = field(default_factory=tuple)
    warnings: Sequence[str] = field(default_factory=tuple)
    degraded: bool = False


@dataclass(frozen=True)
class GateResult:
    decision: ClusterDecision
    reason: str
    warnings: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class ClusterRunResult:
    decision: ClusterDecision
    final_state: ClusterState
    state_history: tuple[ClusterState, ...]
    route_plan: RoutePlan
    provenance: ReplayProvenanceEnvelope
    selected_expert: str | None
    output: ExpertOutput | None
    gate_reason: str
    warnings: tuple[str, ...] = field(default_factory=tuple)


class RegionExpert(Protocol):
    expert_id: str
    region: str

    def infer(self, task: ClusterTaskEnvelope, provenance: ReplayProvenanceEnvelope) -> ExpertOutput:
        ...


class RegionRouter(Protocol):
    def plan(self, task: ClusterTaskEnvelope) -> RoutePlan:
        ...


class ClusterGate(Protocol):
    def evaluate(
        self,
        task: ClusterTaskEnvelope,
        provenance: ReplayProvenanceEnvelope,
        route: RoutePlan,
        output: ExpertOutput,
    ) -> GateResult:
        ...


def _normalized_entropy(scores: Mapping[str, float]) -> float:
    if not scores:
        return 0.0
    positive = [max(float(v), 0.0) for v in scores.values()]
    total = sum(positive)
    if total <= 0.0:
        return 0.0
    probs = [v / total for v in positive if v > 0.0]
    if len(probs) <= 1:
        return 0.0
    entropy = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(len(probs))
    if max_entropy <= 0.0:
        return 0.0
    return entropy / max_entropy


class DeterministicRegionRouter:
    def __init__(
        self,
        *,
        routes_by_task_type: Mapping[str, Sequence[str]],
        policy_version: str = "braincluster-router/v1",
    ):
        self._routes = {key: tuple(value) for key, value in routes_by_task_type.items()}
        self.policy_version = policy_version

    def plan(self, task: ClusterTaskEnvelope) -> RoutePlan:
        candidates = list(self._routes.get(task.task_type, self._routes.get("default", ("generalist",))))
        preferred = str(task.router_features.get("preferred_region", "")).strip()
        risk = str(task.risk_tier).strip().lower()

        scores: dict[str, float] = {}
        for index, expert_id in enumerate(candidates):
            score = 1.0 / (index + 1)
            if preferred and expert_id == preferred:
                score += 0.35
            if risk == "high" and "safety" in expert_id:
                score += 0.5
            scores[expert_id] = score

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if not ordered:
            ordered = [("generalist", 1.0)]
            scores = {"generalist": 1.0}

        primary = ordered[0][0]
        alternates = tuple(item[0] for item in ordered[1:])
        entropy = _normalized_entropy(scores)
        return RoutePlan(
            primary=primary,
            alternates=alternates,
            topk_scores=scores,
            routing_entropy=entropy,
            policy_version=self.policy_version,
        )


class ConfidenceGate:
    def __init__(self, *, min_confidence: float = 0.65):
        self.min_confidence = float(min_confidence)

    def evaluate(
        self,
        task: ClusterTaskEnvelope,
        provenance: ReplayProvenanceEnvelope,
        route: RoutePlan,
        output: ExpertOutput,
    ) -> GateResult:
        if output.confidence < self.min_confidence:
            return GateResult(
                decision=ClusterDecision.FAIL_CLOSED,
                reason=f"expert confidence below threshold ({output.confidence:.3f} < {self.min_confidence:.3f})",
            )
        return GateResult(decision=ClusterDecision.ACCEPT, reason="confidence gate passed")


class ProvenanceRefsGate:
    def __init__(self, *, on_missing: ClusterDecision = ClusterDecision.RETRY):
        self.on_missing = on_missing

    def evaluate(
        self,
        task: ClusterTaskEnvelope,
        provenance: ReplayProvenanceEnvelope,
        route: RoutePlan,
        output: ExpertOutput,
    ) -> GateResult:
        if output.provenance_refs:
            return GateResult(decision=ClusterDecision.ACCEPT, reason="provenance refs present")
        return GateResult(decision=self.on_missing, reason="missing provenance refs")


class ReplayFingerprintGate:
    def evaluate(
        self,
        task: ClusterTaskEnvelope,
        provenance: ReplayProvenanceEnvelope,
        route: RoutePlan,
        output: ExpertOutput,
    ) -> GateResult:
        artifact_fp = str(output.artifacts.get("replay_fingerprint", "")).strip()
        if artifact_fp and artifact_fp != provenance.replay_fingerprint:
            return GateResult(
                decision=ClusterDecision.FAIL_CLOSED,
                reason="replay fingerprint mismatch between envelope and expert output",
            )
        return GateResult(decision=ClusterDecision.ACCEPT, reason="replay fingerprint gate passed")


class BrainClusterCoordinator:
    def __init__(
        self,
        *,
        router: RegionRouter,
        experts: Sequence[RegionExpert],
        gates: Sequence[ClusterGate] | None = None,
        max_attempts: int = 3,
    ):
        self.router = router
        self.experts = {expert.expert_id: expert for expert in experts}
        self.gates = list(gates or [])
        self.max_attempts = max(1, int(max_attempts))

    def run(
        self,
        task: ClusterTaskEnvelope,
        *,
        route_seed: str = "default",
    ) -> ClusterRunResult:
        history: list[ClusterState] = [ClusterState.RECEIVED]
        warnings: list[str] = []
        route = self.router.plan(task)
        history.append(ClusterState.ROUTED)
        provenance = ReplayProvenanceEnvelope.from_task(
            task,
            policy_version=route.policy_version,
            route_seed=route_seed,
            metadata={"routing_entropy": route.routing_entropy},
        )

        selected_expert: str | None = None
        output: ExpertOutput | None = None
        gate_reason = "no experts were attempted"
        attempts = 0

        for expert_id in route.ordered_experts():
            if attempts >= self.max_attempts:
                warnings.append("max attempts reached before evaluating all alternates")
                break
            attempts += 1
            selected_expert = expert_id
            expert = self.experts.get(expert_id)
            if expert is None:
                warnings.append(f"expert '{expert_id}' unavailable")
                continue

            history.append(ClusterState.RUNNING_EXPERT)
            output = expert.infer(task, provenance)
            history.append(ClusterState.GATED)
            gate = self._run_gates(task, provenance, route, output)
            warnings.extend(gate.warnings)
            gate_reason = gate.reason
            if gate.decision == ClusterDecision.ACCEPT:
                history.append(ClusterState.ACCEPTED)
                history.append(ClusterState.FINALIZED)
                return ClusterRunResult(
                    decision=ClusterDecision.ACCEPT,
                    final_state=ClusterState.FINALIZED,
                    state_history=tuple(history),
                    route_plan=route,
                    provenance=provenance,
                    selected_expert=selected_expert,
                    output=output,
                    gate_reason=gate_reason,
                    warnings=tuple(warnings),
                )
            if gate.decision == ClusterDecision.RETRY:
                history.append(ClusterState.RETRY_EXPERT)
                continue
            if gate.decision == ClusterDecision.ESCALATE:
                history.append(ClusterState.ESCALATED)
                history.append(ClusterState.FINALIZED)
                return ClusterRunResult(
                    decision=ClusterDecision.ESCALATE,
                    final_state=ClusterState.FINALIZED,
                    state_history=tuple(history),
                    route_plan=route,
                    provenance=provenance,
                    selected_expert=selected_expert,
                    output=output,
                    gate_reason=gate_reason,
                    warnings=tuple(warnings),
                )
            history.append(ClusterState.FAIL_CLOSED)
            history.append(ClusterState.FINALIZED)
            return ClusterRunResult(
                decision=ClusterDecision.FAIL_CLOSED,
                final_state=ClusterState.FINALIZED,
                state_history=tuple(history),
                route_plan=route,
                provenance=provenance,
                selected_expert=selected_expert,
                output=output,
                gate_reason=gate_reason,
                warnings=tuple(warnings),
            )

        history.append(ClusterState.FAIL_CLOSED)
        history.append(ClusterState.FINALIZED)
        return ClusterRunResult(
            decision=ClusterDecision.FAIL_CLOSED,
            final_state=ClusterState.FINALIZED,
            state_history=tuple(history),
            route_plan=route,
            provenance=provenance,
            selected_expert=selected_expert,
            output=output,
            gate_reason=gate_reason,
            warnings=tuple(warnings),
        )

    def _run_gates(
        self,
        task: ClusterTaskEnvelope,
        provenance: ReplayProvenanceEnvelope,
        route: RoutePlan,
        output: ExpertOutput,
    ) -> GateResult:
        for gate in self.gates:
            verdict = gate.evaluate(task, provenance, route, output)
            if verdict.decision != ClusterDecision.ACCEPT:
                return verdict
        return GateResult(decision=ClusterDecision.ACCEPT, reason="all gates passed")

