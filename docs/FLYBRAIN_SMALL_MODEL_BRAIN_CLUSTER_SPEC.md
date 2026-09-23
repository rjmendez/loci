# FlyBrain Small-Model Brain Cluster Spec

This spec defines an internal Loci architecture for region-specialized small models ("brain cluster") with deterministic routing, replay-safe provenance, and fail-closed gates. It is intentionally scoped to Loci internals and excludes Hugbot or dama-gotchi integration.

## Goals

- Build specialist experts that handle narrow FlyBrain reasoning responsibilities.
- Route deterministically, with bounded retries and explicit fallback decisions.
- Keep every decision replayable with stable fingerprints.
- Fail closed on low-confidence, missing provenance, or replay mismatch conditions.

## Architecture

1. **Task envelope**: typed request contract carrying objective, task type, risk tier, inputs, routing features, and constraints.
2. **Router**: deterministic policy producing primary + alternate experts and routing entropy.
3. **Coordinator**: state machine that executes experts and gates outputs.
4. **Gates**: independent checks (confidence, provenance refs, replay fingerprint consistency).
5. **Provenance envelope**: immutable replay metadata bound to policy version and route seed.

## Reference implementation

Runtime module:

- `mcp/flybrain_brain_cluster.py`
- `mcp/flybrain_brain_cluster_training.py`

Primary types:

- `ClusterTaskEnvelope`
- `ReplayProvenanceEnvelope`
- `RoutePlan`
- `ExpertOutput`
- `GateResult`
- `ClusterRunResult`

Primary contracts:

- `RegionExpert` protocol
- `RegionRouter` protocol
- `ClusterGate` protocol

Provided defaults:

- `DeterministicRegionRouter`
- `BrainClusterCoordinator`
- `ConfidenceGate`
- `ProvenanceRefsGate`
- `ReplayFingerprintGate`

## Coordinator state model

Execution states:

- `received`
- `routed`
- `running_expert`
- `gated`
- `accepted`
- `retry_expert`
- `escalated`
- `fail_closed`
- `finalized`

Canonical flow:

- `received -> routed -> running_expert -> gated -> accepted -> finalized`

Retry flow:

- `received -> routed -> running_expert -> gated -> retry_expert -> running_expert -> ...`

Hard-stop flow:

- `received -> routed -> running_expert -> gated -> fail_closed -> finalized`

## Failure behavior

| Failure mode | Expected behavior |
|---|---|
| Expert unavailable | Warning recorded; coordinator tries alternates until `max_attempts`, then fail closed. |
| Low confidence | `ConfidenceGate` returns `fail_closed`. |
| Missing provenance references | `ProvenanceRefsGate` returns configurable decision (`retry` by default). |
| Replay fingerprint mismatch | `ReplayFingerprintGate` returns `fail_closed`. |
| No viable expert after retries | Coordinator returns `fail_closed` with terminal `finalized` state. |

## Metrics to track

- Routing entropy (`RoutePlan.routing_entropy`)
- Expert utilization distribution (per `selected_expert`)
- Gate rejection rate by gate type
- Fail-closed rate and reason frequency
- Replay mismatch incidence

## Phase rollout

1. **P0 simulation**: run cluster offline with replay fixtures only.
2. **P1 shadow**: run cluster in parallel, non-authoritative outputs.
3. **P2 constrained live**: allow narrow low-risk paths behind canary gates and rollback triggers.

## Test coverage

Tests:

- `mcp/tests/test_flybrain_brain_cluster.py`
- `mcp/tests/test_flybrain_brain_cluster_training.py`

Current checks verify:

- deterministic routing for identical inputs
- stable replay fingerprint generation
- accept path when gates pass
- retry-to-alternate behavior
- fail-closed behavior for low confidence, missing fingerprint, and replay mismatch
- deterministic dataset manifest generation and train/val/test splits
- golden-set construction per region and promotion-gate threshold evaluation

## Training/test scaffold (P0 -> P1)

`mcp/flybrain_brain_cluster_training.py` provides deterministic scaffolding so expert training and canary evaluation can be wired without external integrations:

- `TrainingSample`: strict per-sample contract (region, label, confidence target, provenance refs).
- `build_dataset_manifest(...)`: canonical sample digest + deterministic split IDs with a versioned manifest id.
- `build_golden_set(...)`: bounded per-region canary sample selection.
- `PredictionRecord` + `evaluate_predictions(...)`: promotion-gate metrics (accuracy, abstain rate, calibration error).
- `PromotionGateThresholds`: explicit pass/fail thresholds for promotion decisions.

This keeps early training and test workflows replayable and fail-closed while the expert models and router training loops are implemented.
