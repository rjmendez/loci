# FlyBrain Pilot Go / No-Go Checklist

## Purpose

This checklist defines the required pre-merge, pilot-launch, and run-time decision gates for FlyBrain work in Loci. It is aligned with the current FlyBrain safety, provenance, queue, rollout, and operational policy docs, and it is intended to serve as the canonical go / no-go decision record before merge, pilot launch, and promotion.

Primary references:
- `docs/FLYBRAIN_GUIDE.md`
- `docs/FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md`
- `docs/FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md`
- `docs/FLYBRAIN_QUEUE_STATE_MACHINE_SPEC.md`
- `docs/FLYBRAIN_OPERATOR_TRAINING_CHECKLIST.md`
- `docs/FLYBRAIN_HARNESS_ROLLOUT_MILESTONES.md`
- `docs/FLYBRAIN_PHASE1_FEATURE_GATING.md`
- `docs/FLYBRAIN_PILOT_SLOS_AND_TRIPWIRES.md`
- `docs/FLYBRAIN_QUEUE_AND_PROMOTION_ROLLBACK_PLAYBOOK.md`
- `docs/FLYBRAIN_HARNESS_MANIFEST_PROVENANCE_SCHEMA.md`

Default operating rule: no go unless the evidence is present, replayable, and safe.

## Decision posture

A FlyBrain pilot is eligible to proceed only when all of the following remain true:
- every read/write path resolves under `LOCI_FLYBRAIN_STORAGE_ROOT` and passes the path guard,
- every FlyBrain-derived claim keeps explicit dataset/version/query provenance,
- the queue remains in a valid operating state with active ownership and no stale or unsafe holds,
- the active graph or snapshot pointer is the last known-good, verified artifact,
- there is no unresolved integrity mismatch, quarantine, or unverified promotion state.

Any hard-stop failure triggers immediate stop-the-line; no override is allowed without documented corrective action and fresh evidence.

## 1) Pre-merge readiness checks

Complete these before merge or pilot promotion. A merge is not ready when the evidence is incomplete, ambiguous, or fail-open.

### 1.1 Safety and path integrity
- [ ] All FlyBrain entrypoints route through the validated path-guard utility.
- [ ] `LOCI_FLYBRAIN_STORAGE_ROOT` is set explicitly; no hardcoded drive letter, bare drive root, or symlink-based fallback path is used.
- [ ] All candidate paths resolve under `$LOCI_FLYBRAIN_STORAGE_ROOT\graph`, `\snapshots`, `\cache`, `\backups`, or `\logs` only.
- [ ] Destructive operations are disabled or explicitly validated against the allowlist and root guard.
- [ ] Log entries record allow/deny decisions for each path and operation.
- [ ] Storage pressure stays below the warning threshold or a cleanup plan is attached and approved.

Objective evidence requirement:
- `logs\pipelines\<run_id>\preflight.json` showing the allowlisted root and path inspection result.
- path validation log or denied-path record with reason and timestamp.

### 1.2 Provenance and replayability
- [ ] Every FlyBrain-derived claim, finding, or result packet includes the minimum provenance envelope: tool name, input entity, dataset scope, query semantics, result contract, and replay fingerprint.
- [ ] Dataset scope is explicit for `hb`, `fw`, or any cross-dataset comparison; no implicit generalization across sex, stage, anatomy, or query default.
- [ ] Query semantics are preserved: `group_by_class`, `weight`, `exclude_dbs`, paging, and count-status warnings are documented.
- [ ] Model-only or inferred-only claims are either downgraded or supported by independent tool/human/deterministic evidence.
- [ ] No `model_asserted` claim is treated as final without provenance and verification checks.

Objective evidence requirement:
- finding metadata or manifest entries with `flybrain_provenance` or equivalent provenance bundle,
- replay fingerprint or run manifest for the same query path,
- verification results showing no unsupported model-only claim is being published as final.

### 1.3 Queue and coordination integrity
- [ ] Queue state is valid: `queued`, `claimed`, `done`, `blocked`, `cancelled` only.
- [ ] No item remains `claimed` past its valid lease window without a live owner and re-lease check.
- [ ] Any blocked item is explicitly recorded with owner, reason, and follow-up action.
- [ ] Claim/reclaim behavior matches the canonical state machine; no ownership bypass is allowed.
- [ ] Dependencies are stored and interpreted only by a documented scheduler policy, not hidden auto-unlock logic.

Objective evidence requirement:
- current queue snapshot from `investigation_queue_status` / `list` with item states, owners, lease expiry, and notes,
- proof that blocked or stale items are not silently left active.

### 1.4 Artifact integrity and promotion safety
- [ ] Candidate manifests include all required fields and relative paths only.
- [ ] Every artifact has an existence check and a valid lowercase SHA-256.
- [ ] Manifest provenance, artifact hash, and dataset version pins align.
- [ ] No candidate pointer points to a quarantined, missing, or stale artifact.
- [ ] Promotion updates are atomic pointer metadata updates; no in-place mutation of the active graph state.

Objective evidence requirement:
- `manifest.json` plus `manifest.sha256` for the relevant snapshot or graph candidate,
- integrity verification log,
- promotion pointer metadata showing candidate before/after and active pointer status.

### 1.5 Budget, rollback, and restore readiness
- [ ] Current storage use is under soft-cap or an approved cleanup/remediation plan is in place.
- [ ] Backup and restore plan has been reviewed for the dataset or graph being promoted.
- [ ] Restore-drill evidence exists for the latest relevant project state or data path.
- [ ] Rollback path is known: pointer revert, manifest restore, or snapshot rewind.

Objective evidence requirement:
- storage usage snapshot and threshold classification,
- recent backup verification or restore-drill log,
- rollback playbook entries or saved pointer snapshot.

## 2) Pilot launch readiness checks

Run these before any pilot launch or execution lane starts.

### 2.1 Environment and configuration gate
- [ ] `LOCI_FLYBRAIN_STORAGE_ROOT` is defined and valid.
- [ ] `LOCI_FLYBRAIN_RUN_MODE` is set to the intended state (`dry-run`, `plan`, or `execute`).
- [ ] `LOCI_FLYBRAIN_DATASET_SCOPE` and version pins are explicitly approved for the current pilot scope.
- [ ] All write targets are under the configured root and no ad hoc machine-specific paths are used.
- [ ] Default retry/backoff values are documented and consistent with the harness plan.

Objective evidence requirement:
- env/config export or run manifest captured under `logs\pipelines\<run_id>`,
- dry-run output showing scope and path checks pass.

### 2.2 Plan gate readiness
- [ ] `dry-run` succeeded without path, scope, or storage misconfiguration.
- [ ] `plan` generated a deterministic stage graph and idempotency keys.
- [ ] Predicted writes remain inside the canonical allowlisted subtrees.
- [ ] Queue items needed for launch are created in `queued` or `claimed` state, not ambiguous or orphaned states.
- [ ] Human reviewer has signed off on the plan package before `execute` begins.

Objective evidence requirement:
- `logs\pipelines\<run_id>\plan.json` and stage-state artifacts,
- plan review note with reviewer name and timestamp.

### 2.3 Execution preflight gate
- [ ] Snapshot or source artifact manifest is in place and hash-verified.
- [ ] Candidate graph build/import is staged, not silently swapped into active state.
- [ ] Query pack is defined for at least a minimal smoke set and a representative negative-scope test.
- [ ] No open integrity warnings or quarantine issues remain on the candidate artifact.
- [ ] Pilot operator confirms queue/ownership state is ready for execution work.

Objective evidence requirement:
- manifest and hash outputs for the candidate artifact,
- smoke query pack artifact or run-result bundle,
- queue snapshot showing clear ownership and no unresolved stale claim.

## 3) Go / no-go gates with objective evidence requirements

### Gate G0: Pre-merge / branch readiness
Go only when all are true:
- path guard passes,
- manifest provenance is complete,
- artifact integrity is verified,
- queue state is valid,
- budget and rollback are documented,
- no open hard-stop issue remains.

No-go triggers:
- path escape or deny event,
- checksum mismatch,
- manifest schema violation,
- unapproved dataset pin,
- stale or unsafe queue ownership,
- missing rollback or backup evidence.

Required evidence:
- preflight log, manifest proof, validation transcript, queue snapshot, backup/restore status.

### Gate G1: Pilot launch gate
Go only when:
- `dry-run` and `plan` both passed with unchanged scope,
- all required env vars are present,
- queue ownership and lease state are stable,
- storage remains below the warning threshold or cleanup is active and authorized,
- smoke query pack is defined and ready.

No-go triggers:
- `plan` is not reproducible,
- hidden scope drift,
- active queue item remains unowned or stale,
- any path, provenance, or manifest problem is discovered in preflight.

Required evidence:
- `preflight.json`, `plan.json`, queue snapshot, scope pin record, and run-mode config snapshot.

### Gate G2: Execution / promote gate
Go only when:
- candidate artifact has verified integrity,
- manifest aligns with dataset version and source metadata,
- promotion update is an atomic pointer swap only,
- smoke queries, counts, and result contracts match expected behavior,
- the queue shows a clean claimed/done state for the active task.

No-go triggers:
- promotion pointer drift or inconsistent active pointer,
- integrity mismatch or quarantine,
- query contract violation or unscoped answer,
- `model_asserted` claim without independent support.

Required evidence:
- candidate manifest and checksum output,
- comparison of active pointer vs candidate pointer,
- result hashes or query-pack output with expected values,
- signed review note.

### Gate G3: Pilot hold / continue gate
The pilot may continue in a limited or degraded mode only when:
- the issue is warning-level and bounded,
- remediation is explicit and tracked,
- the risk does not affect safety, provenance, artifact integrity, or queue ownership,
- an operator lead approves the exception in writing.

No-go triggers:
- any hard-stop trigger in FlyBrain safety, provenance, storage, or queue semantics,
- recovery action not documented by evidence,
- repeated or unresolved warning-level drift across a full pilot cycle.

Required evidence:
- issue log, corrective action record, reviewer approval, and updated run notes.

## 4) Required sign-off roles and artifacts

### Required roles
- Operator: owns run preflight, execution, queue-task handling, and run logs.
- Secondary reviewer: verifies the plan package, scope boundaries, and non-safety drift before execution.
- Operator lead / approver: signs off on go / no-go for launch and promotion gates; may not waive hard-stop conditions.
- Incident owner: handles contamination, quarantine, or rollback actions when an integrity issue or queue fault occurs.
- Queue owner: assumes responsibility for any claimed work item and executes stop-the-line handling when required.

### Required artifacts for every launch
- `logs\pipelines\<run_id>\preflight.json`
- `logs\pipelines\<run_id>\plan.json`
- `logs\pipelines\<run_id>\summary.json` or equivalent execution output
- `snapshots\<dataset>\<version>\manifest\manifest.json`
- `snapshots\<dataset>\<version>\manifest\manifest.sha256`
- query pack or smoke-results file for the launch
- queue snapshot or item list showing states, owners, and lease expiry
- backup/restore verification record for the relevant graph or snapshot

### Required sign-off record
Each launch or promotion record must include:
- operator name,
- reviewer name,
- approver name,
- run ID,
- gate pass/fail state,
- evidence artifact locations,
- date/time,
- explicit remediation or abort note if any gate fails.

## 5) Explicit abort criteria and contingency actions

### Hard abort criteria
Abort immediately if any of the following occur:
- path-policy violation or any write outside `LOCI_FLYBRAIN_STORAGE_ROOT`,
- integrity mismatch or manifest validation failure,
- promotion pointer references a quarantined, missing, or stale artifact,
- queue item enters blocked or stale state without a documented owner and remediation plan,
- FlyBrain claim lacks required provenance or is stored as final without a replayable evidence trail,
- storage crosses the hard-stop threshold or a critical budget condition occurs without a documented override,
- any `model_asserted` claim is treated as final without independent support,
- scope drift crosses dataset, stage, or anatomy boundaries without a fresh, explicit evidence review.

### Immediate contingency actions
1. Stop all new writes and freeze the current job.
2. Mark the active queue item as `blocked` with a clear stop-the-line note if needed.
3. Quarantine the artifact or candidate graph; keep the prior known-good artifact active.
4. Preserve evidence: manifest, hash, pointer state, queue snapshot, and run logs.
5. Notify the operator lead and incident owner.
6. Revert to the prior good state using the last verified pointer, manifest, or snapshot.
7. Re-run only the smallest safe validation after the root cause is corrected.

### Recovery flow after abort
- confirm the root cause and document it,
- remove or quarantine the faulty artifact set,
- restore the last known-good graph or snapshot,
- re-run `dry-run` / `plan` on the minimal safe subset,
- resume only when the same gate passes with fresh evidence,
- keep a written record of the reason for resume and the evidence that made it safe.

## 6) Operational rule summary

The pilot must not proceed on optimism. It proceeds only on objective evidence.

Required default stance:
- warning states are recoverable only when bounded and explicitly tracked,
- hard-stop states require immediate suspension and clear corrective action,
- evidence must be stored and replayable, not inferred after the fact,
- all operational work remains under the allowlisted storage root and queue discipline,
- the final answer or rollout decision is only as strong as the proof behind it.

Status: ready for FlyBrain pilot gating and operator sign-off workflow.
