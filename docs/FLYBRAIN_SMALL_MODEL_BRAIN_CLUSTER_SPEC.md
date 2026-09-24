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
4. **Swarm policy**: bounded parallel fan-out (`fanout_k`) with deterministic consensus and fail-closed gating.
5. **Gates**: independent checks (confidence, provenance refs, replay fingerprint consistency).
6. **Provenance envelope**: immutable replay metadata bound to policy version and route seed.

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
- `run_shadow_replay_fixtures_with_artifact(...)`
- `compare_shadow_replay_runs(...)`
- `run_brain_cluster_shadow_replay(...)`

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
- Decision match rate between baseline/candidate artifacts
- Confidence drift summary (signed/absolute deltas)
- Fail-closed rate delta (candidate minus baseline)
- Expert collapse concentration delta

## Shadow replay harness (P1)

`run_brain_cluster_shadow_replay(...)` executes replay fixtures through baseline and candidate artifact manifests, compares per-fixture decisions, and emits a deterministic JSON report:

- `schema_version`: `braincluster-shadow-replay-report/v1`
- `pass` / `status` / `exit_code`
- `thresholds`: explicit pass/fail thresholds
- `metrics`: decision match, confidence drift, fail-closed delta, routing entropy drift, collapse concentration
- `failure_reasons`: threshold failure strings
- `issue_flags`: drift/entropy/collapse flags
- `fixtures`: per-fixture baseline-vs-candidate rows

`serialize_brain_cluster_shadow_replay_report(...)` emits stable machine-readable JSON with sorted keys and newline termination for pipeline use.

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
- deterministic shadow replay report output
- threshold-based pass/fail replay gating and failure reasons
- malformed replay fixture handling

## Training/test scaffold (P0 -> P1)

`mcp/flybrain_brain_cluster_training.py` provides deterministic scaffolding so expert training and canary evaluation can be wired without external integrations:

- `TrainingSample`: strict per-sample contract (region, label, confidence target, provenance refs).
- `build_dataset_manifest(...)`: canonical sample digest + deterministic split IDs with a versioned manifest id.
- `build_golden_set(...)`: bounded per-region canary sample selection.
- `PredictionRecord` + `evaluate_predictions(...)`: promotion-gate metrics (accuracy, abstain rate, calibration error).
- `PromotionGateThresholds`: explicit pass/fail thresholds for promotion decisions.
- `train_swarm_consensus_student(...)`: deterministic student training from
  multi-expert consensus votes, so swarm knowledge is distilled into a bounded
  artifact with explicit consensus policy metadata.

This keeps early training and test workflows replayable and fail-closed while the expert models and router training loops are implemented.

`mcp/flybrain_brain_cluster_pipeline.py` now provides a P0 execution harness that
composes the scaffold into one deterministic dry-run:

- train region experts and router model from the dataset manifest
- emit fail-closed runtime artifact manifest (`router.json` + `experts.json`)
- run golden-set gate and shadow replay thresholds
- stage candidate + promote only when both checks pass (rollback path explicit)

`mcp/flybrain_brain_cluster_thresholds.py` calibrates threshold candidates from
held-out dry-run reports and emits a versioned threshold bundle keyed by an
input fingerprint. This keeps threshold changes measurable and reproducible.

`mcp/flybrain_brain_cluster_fw_samples.py` builds deterministic `TrainingSample`
payloads directly from the local FlyWire snapshot at
`<storage_root>\snapshots\fw\flywire783\metadata\files\per_neuron_neuropil_count_pre_783.feather`
so train/gate/shadow runs can execute against real local data without manual
dataset reformatting.

The builder now supports two objectives:
- `connectivity_tier`: binary label (`high_connectivity` / `baseline_connectivity`) from per-neuron neuropil counts.
- `neurotransmitter_dominance`: multiclass label (`dominant_ach`, `dominant_gaba`, `dominant_glut`, `dominant_da`, `dominant_ser`, `dominant_oct`) derived from proofread connection neurotransmitter probabilities, with deterministic region mapping from dominant neuropil.

Hardening guards in the builder enforce:
- minimum distinct label count before training sample emission
- maximum dominant-label share to block extreme class-collapse datasets

`mcp/flybrain_brain_cluster_release_prep.py` calibrates gate/shadow threshold bundles
from historical `p0-report.json` runs, grouped by `(dataset_symbol, objective)`. It writes
one versioned `braincluster-thresholds-<dataset>-<objective>.json` per group (report schema
`braincluster-release-prep/v2`). Runs whose dataset cannot be resolved, or whose objective
is `custom` or `unknown`, are listed under `unresolved_runs` and never calibrated. One
dataset's runs therefore never count toward another dataset's minimum report count.

Every p0 run splits train/val/test by group (`grouped_split_ids`). Samples that share a
known value of any `split_group_keys` metadata key from the registry (cell type,
hemilineage, or a left/right pair via `split_group`) land in the same split.
`dataset_manifest.notes.split` records the grouping and how many groups a per-neuron split
would have straddled.

The promotion gate also runs the trivial-baseline check (AC6,
`flybrain_brain_cluster_baselines`). Three rules are fitted on the train split and scored on
the held-out split (test, else val): majority class, a one-feature threshold on a numeric
`input_text` feature, and argmax over score features in `input_text`. The gate fails unless
the model's held-out accuracy is at least the best trivial accuracy plus `min_margin`
(default 0.01, CLI `--baseline-margin`). Both numbers are recorded in
`gate_report.trivial_baseline`, in `trivial-baseline-report.json`, and in the
`trivial_baseline` block of each threshold bundle.
