# FlyBrain Pilot SLOs and Tripwires

## Purpose

This pilot sets explicit service-level objectives (SLOs) for FlyBrain local-harness work in Loci. The goal is not to optimize throughput alone; it is to ensure every claim stays scoped, every write stays inside the configured storage root, and every operational queue item is either actively moving or intentionally blocked.

This doc is grounded in the repo's current FlyBrain safety and provenance controls:

- `docs/FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md` requires explicit dataset/version/query scope for FlyBrain-derived claims.
- `docs/FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md` hard-stops any path escape or destructive write outside the allowlisted root.
- `docs/FLYBRAIN_RISK_REGISTER.md` and `docs/FLYBRAIN_FAIL_BEHAVIOR_MATRIX.md` distinguish safety-critical hard stops from degraded-but-allowed warnings.
- `mcp/provenance_firewall.py` requires independent human/tool/deterministic evidence before a `model_asserted` claim can be treated as credible.
- `mcp/replay_fingerprint.py` creates deterministic replay fingerprints for FlyBrain requests and dataset scope.
- `mcp/investigation_tools.py` provides the operational queue states (`queued`, `claimed`, `done`, `blocked`, `cancelled`) and lease semantics for stop-the-line coordination.
- `docs/sleep-consolidation-scheduler-spec.md` explicitly tracks `queue_pressure` as a live operational signal.

## Pilot operating principles

1. Safety is the first-class SLO. A path-policy violation or integrity failure is never a “soft warning” for pilot execution.
2. Provenance is required for every FlyBrain-derived claim. If the claim cannot be replayed or scoped, it is not eligible for a confident pilot result.
3. The queue is a control plane, not just status metadata. `investigation_queue_complete(..., state="blocked")` is the explicit stop-the-line action when a work item becomes unsafe or ungrounded.
4. Warning states are allowed only when the system remains in a bounded, recoverable posture. Hard-stop states require immediate pause and escalation.

## 1) SLO definitions and formulas

### 1.1 Provenance completeness SLO

Definition: every FlyBrain-derived finding must carry a complete provenance envelope before it is treated as evidence.

Formula:

`Provenance completeness = (FlyBrain findings with tool_name + request + dataset_scope + result_contract + replay_fingerprint) / (all FlyBrain-derived findings) × 100%`

Target for pilot:

- Green: 100%
- Acceptable with warning: 98–99.9%
- No-go: < 98%

Required fields are the same minimum envelope described in `FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md`: tool name, resolved entity, dataset scope, query semantics, result contract (`count`, `count_status`, warnings), and replay fingerprint.

### 1.2 Scope fidelity SLO

Definition: FlyBrain claims must remain within the dataset, version, sex, stage, anatomy, and evidence-family boundaries actually observed.

Formula:

`Scope fidelity = (FlyBrain claims that remain explicitly scoped and avoid overgeneralization) / (all FlyBrain-derived claims) × 100%`

Target for pilot:

- Green: >= 99%
- Warning: 95–98%
- No-go: < 95%

A claim that says “this is generally true” without direct support across the matching scope fails this SLO and is blocked from final publication.

### 1.3 Replay determinism SLO

Definition: equivalent FlyBrain requests must produce the same deterministic replay fingerprint and dataset-scope readback.

Formula:

`Replay determinism = (repeated runs with matching replay_fingerprint and equivalent dataset_scope) / (all repeated FlyBrain runs) × 100%`

Target for pilot:

- Green: 100%
- Warning: 99–99.9%
- No-go: < 99%

This is the operational guardrail for `replay_fingerprint` behavior in `mcp/replay_fingerprint.py` and the general provenance guidance around “replayable evidence.”

### 1.4 Queue health / work discipline SLO

Definition: active investigation work should be either moving forward or intentionally resolved; blocked items should not accumulate without active operator ownership.

Formula:

`Queue pressure = blocked_items / max(1, claimed_items + blocked_items)`

Additional check:

`Stale claimed ratio = stale_claimed_items / total_claimed_items`

Target for pilot:

- Green: queue pressure < 0.20 and stale claimed ratio < 0.10
- Warning: queue pressure 0.20–0.35 or stale claimed ratio 0.10–0.25
- No-go: queue pressure > 0.35 or any sustained stale claimed item without human reassignment

This is grounded in the queue semantics in `mcp/investigation_tools.py` and the scheduler-doc signal `queue_pressure` described in `docs/sleep-consolidation-scheduler-spec.md`.

### 1.5 Storage safety SLO

Definition: every file system operation must remain under the configured allowlist root and under the configured storage budget guardrails.

Formula:

`Storage safety = 1 - (path violations + integrity mismatches + budget hard-stop writes) / total write operations`

Measured by:

- allowlist path validation (`LOCI_FLYBRAIN_STORAGE_ROOT`)
- write-path deny rules in `docs/FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md`
- disk usage thresholds from `docs/FLYBRAIN_WAIT_*` style storage guardrails and `docs/FLYBRAIN_RISK_REGISTER.md`

Target for pilot:

- Green: 100% safe writes, zero policy violations, storage under soft cap (160 GB)
- Warning: usage 150–180 GB with active cleanup and no new large writes
- No-go: any out-of-root write, any path escape, or storage >= 200 GB hard-stop

## 2) Pilot go / no-go thresholds

### Green go

A pilot run may continue when all are true:

- Provenance completeness >= 100% for stored FlyBrain-derived claims
- Scope fidelity >= 99%
- Replay determinism >= 99%
- Queue pressure < 0.20 and no stale claimed work
- No path-policy violations and storage below soft cap
- No unresolved `model_asserted` findings without independent support
- No integrity mismatch or quarantine issue on promoted artifacts

### Yellow / conditional go

The pilot may continue only with explicit operator acknowledgement when:

- One warning-level signal is present for a bounded period
- Cleanup or re-scoping is active and tracked in investigation notes
- There is no impact on safety, provenance, or evidence integrity

Examples:

- Storage at 150–160 GB with cleanup in progress
- Queue pressure 0.20–0.35 while a backlog is being drained
- One or two low-risk provenance warnings on legacy records that are explicitly marked degraded

Conditions for yellow status:

- must attach corrective action within 24 hours
- must not involve any hard-stop trigger
- must be reviewed in the current investigation before next phase gate

### Hard no-go

Pilot stops immediately when any hard-stop trigger occurs. This includes:

- any path escape or destructive write outside the allowlisted root
- any FlyBrain claim stored without provenance
- any `model_asserted` claim lacking independent evidence
- any integrity mismatch or manifest validation failure on promoted artifacts
- any storage pressure at or above 200 GB hard-stop threshold
- any queue item requiring a stop-the-line block that is not explicitly handled by an owner

## 3) Stop-the-line tripwires and escalation actions

### 3.1 Hard-stop tripwires

| Tripwire | Why it is a hard stop | Required action |
|---|---|---|
| Path escape / root violation | Filesystem safety policy is explicit: any failed path check is a hard failure. | Abort the run immediately. Do not write. Log deny reason and fix configuration before retry. |
| Missing FlyBrain provenance envelope | A FlyBrain claim without dataset/version/query scope is not replayable or defensible. | Do not persist or publish the claim. Add provenance metadata or narrow the claim to a degraded, non-final state. |
| `model_asserted` without independent evidence | `mcp/provenance_firewall.py` requires independent evidence. | Mark the claim as unsupported, downgrade it, and require independent evidence before reuse. |
| Artifact integrity mismatch / quarantine | Tampered, partial, or malformed artifacts are not valid evidence. | Quarantine the artifact, keep prior active data, and re-validate before promotion or sync. |
| Budget hard stop (>= 200 GB) | Storage policy defines this as the hard stop for nonessential writes. | Freeze nonessential writes and keep only the minimal active run set. |
| `investigation_queue_complete(..., state="blocked")` on a safety-critical item | Queue is the operational stop-the-line mechanism. | Reassign owner, document the hold reason, and do not continue until the item is resolved or explicitly requeued. |

Escalation flow for a hard-stop item:

1. Block the current item using the investigation queue (`state="blocked"`), or fail the run before further writes.
2. Record the reason in the active investigation via `investigation_note(..., field="checked_source")` or a queued item note.
3. Escalate to the owning operator for root-cause repair.
4. Resume only after the issue is fixed and the required safety or provenance check passes cleanly.

### 3.2 Warning tripwires

Warning-level conditions do not stop the runway, but they require tracking and bounded recovery.

| Warning tripwire | Typical condition | Recovery action |
|---|---|---|
| Storage watch | 150–160 GB usage | Pause large downloads, prune stale cache, archive unused outputs. |
| Queue pressure | 0.20–0.35 | Drain backlog, claim and complete work with explicit ownership, avoid issuing new work without review. |
| Partial provenance | missing nonessential metadata | Re-run or narrow the claim; attach the missing provenance envelope within the current investigation cycle. |
| Count-status or query-semantics warnings | `count_status != exact`, grouped counts confused with pairwise counts | Re-run with corrected query settings and label query semantics in the final answer. |
| Legacy findings with degraded metadata | older records without complete provenance | Keep them tagged as degraded, not final evidence. |

A warning stays warning only while the run remains within the bounded safer space and the action is documented. Nothing in warning state should be treated as general truth.

## 4) Distinction: warning vs hard-stop tripwires

The difference is simple and must remain explicit:

- Warning tripwires indicate degraded but recoverable conditions. They do not invalidate the current run by themselves. Example: a storage watch condition at 150 GB or a queue backlog that is reclaimable.
- Hard-stop tripwires indicate the system is no longer safe to continue under the current evidence and operating posture. Example: a path escape, missing provenance, or invalid artifact promotion.

Operational rule: warnings trigger remediation and review; hard stops trigger suspension and escalation.

In other words:

`warning = recoverable, bounded, and still within acceptable pilot operating posture`
`hard stop = unsafe, ungrounded, or un-replayable; continuation is not permitted`

## 5) Measurement sources in the current code and docs

These are the active sources for pilot measurement and enforcement:

- `docs/FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md` — minimum provenance envelope for FlyBrain claims
- `docs/FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md` — allowlist-only storage boundaries and destructive-write rules
- `docs/FLYBRAIN_RISK_REGISTER.md` — operational risk register and stop/go decisions
- `docs/FLYBRAIN_FAIL_BEHAVIOR_MATRIX.md` — fail-closed vs degraded classification
- `docs/FLYBRAIN_PHASE1_FEATURE_GATING.md` — gating thresholds for phase progression and local rollout
- `docs/sleep-consolidation-scheduler-spec.md` — queue pressure and workload control heuristics
- `mcp/provenance_firewall.py` — evidence firewall for `model_asserted` claims
- `mcp/replay_fingerprint.py` — deterministic request/dataset fingerprinting
- `mcp/investigation_tools.py` — queue ownership and stop-the-line workflow (`queued`, `claimed`, `done`, `blocked`, `cancelled`)
- `mcp/investigation_tools.py::_coordination_lease_expired` and `_coordination_manifest` — lease discipline for worker ownership

These are not optional “extra” docs. They are the current operating evidence base for pilot safety and rollout decisions.

## 6) Recommended pilot gate summary

Pilot gate summary for operational decision-making:

- Proceed with green go only if all SLOs are green and no hard-stop tripwire is active.
- Continue in yellow only if the condition is degraded but bounded and all active corrective actions remain explicit.
- Stop immediately on any hard-stop tripwire and leave the queue or run in a blocked state until the issue is fixed.

The working rule is intentionally conservative:

- if the claim cannot be replayed, it is not a valid pilot claim
- if the path cannot be validated, it is not a valid pilot write
- if the queue cannot be owned or drained, the work is not in a safe pilot state

## Status

Grounded in the current repo and ready to use as the pilot control document for FlyBrain safety, provenance, and queue discipline.
