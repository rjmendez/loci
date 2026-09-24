# FlyBrain Queue and Promotion Rollback Playbook

## Purpose

Use this playbook when FlyBrain harness queue operations, manifest validation, or promotion state begin to violate the repository contract. The goal is to stop forward progress, preserve evidence, revert to the last known-good state, and re-enable only after verification has proven the root cause is corrected.

This playbook aligns with:

- `docs/FLYBRAIN_QUEUE_STATE_MACHINE_SPEC.md`
- `docs/FLYBRAIN_HARNESS_MANIFEST_PROVENANCE_SCHEMA.md`
- `docs/FLYBRAIN_ARTIFACT_INTEGRITY_QUARANTINE_POLICY.md`
- `docs/loci-validated-knowledge-promotion.md`
- `docs/FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md`

## Default incident posture

When any of the following are true, enter containment mode immediately:

- queue API behavior diverges from the canonical state machine,
- ownership or lease validation fails,
- a manifest is accepted without required provenance or integrity checks,
- a graph or knowledge promotion pointer points to an unverified or quarantined artifact,
- any promotion writes a pointer or state transition before the manifest and artifact are validated.

Default policy:

- freeze queue claims,
- freeze promotion writes,
- keep the prior known-good graph active,
- do not auto-recover by bypassing validation,
- preserve forensic evidence before any mutation.

---

## 1. Queue schema or behavior regression

### Trigger conditions

- `investigation_queue_enqueue`, `investigation_queue_claim`, or `investigation_queue_complete` accepts invalid or out-of-spec payloads.
- queue items are created with a state outside `queued`, `claimed`, `done`, `blocked`, `cancelled`.
- `scope_kind` or `scope_targets` validation is bypassed or widened silently.
- `dependencies` unexpectedly unlock items without a scheduler policy decision.
- `lease_expires_at` is missing, malformed, or not recomputed correctly.
- manifest queue data no longer matches `manifest["coordination"]["items"]` expectations.

### Immediate containment actions

1. Stop all new queue claims and completions in the affected environment.
2. Disable any scheduler or orchestration path that auto-advances queued items to `claimed`.
3. Snapshot the queue state: item IDs, states, owner sessions, lease expiry values, dependency lists, and manifest hashes.
4. Switch the queue to read-only for investigation; do not mutate any items while the root cause is under review.
5. Place dependent jobs behind a manual gate until the queue semantics are restored.

### Rollback steps

1. Revert the queue schema or behavior change to the last known-good release or patch.
2. Restore the prior queue manifest or snapshot from the last verified backup.
3. If a migration introduced the issue, revert or neutralize the migration before reloading the queue.
4. Reconcile the queue against the canonical state machine:
   - `queued` is the canonical enqueue state,
   - `claimed` requires a valid owner and live lease,
   - terminal states are sticky and not claimable,
   - repeated `complete` to the same final state is idempotent.
5. Restore dependency handling to an explicit scheduler policy only; do not auto-unlock without an explicit rule.

### Verification before re-enable

- Every queue item state matches one of `queued`, `claimed`, `done`, `blocked`, `cancelled`.
- `claimed` items have a single owner and valid `lease_expires_at`.
- No final-state item is still being claimed by the scheduler.
- `dependencies` are stored as metadata only unless scheduler logic explicitly enforces gating.
- A targeted queue replay or dry run succeeds for a small known-good subset before production re-enable.

### Post-incident evidence checklist

- manifest diff showing queue schema before/after rollback
- affected item ID list and their transition history
- validation logs for queue enqueue/claim/complete calls
- unit/regression test evidence covering canonical state transitions
- release/commit hash for the reverted schema or behavior change

---

## 2. Queue ownership and lease regression

### Trigger conditions

- two sessions claim the same item while a lease remains active,
- a claim is accepted despite `lease_expires_at` being in the future,
- ownership is lost, overwritten, or manually mutated without a valid lease,
- `investigation_queue_complete` allows completion without the current owner,
- a claimed item is reclaimed by another session before expiry,
- duplicate work or silent cancellation is observed due to ownership drift.

### Immediate containment actions

1. Freeze all claims from any session other than the verified current owner.
2. Treat all active claims as invalid unless they have a valid owner and a future lease timestamp.
3. Expire or suspend any stale or malformed lease before the scheduler reuses the item.
4. Stop retry logic that bypasses ownership enforcement.
5. Disable reclaim attempts while the live lease is still active.

### Rollback steps

1. Revert to the last known-good ownership and lease logic.
2. Restore the manifest from a snapshot where `owner_session` and `lease_expires_at` are consistent.
3. Remove any unexpected manual owner mutation and rebuild the queue state from the source-of-truth backup.
4. Force a reconciliation pass for all claimed items so there is exactly one owner per live claim.
5. Reject any claim that tries to take active ownership without the lease expiring first.

### Verification before re-enable

- every `claimed` item has exactly one owner and a future `lease_expires_at`,
- `lease_expires_at <= now` is treated as expired and reclaimable only after expiry,
- `complete` without the owning session remains rejected for an active claim,
- same-owner claim remains idempotent and renews only the lease,
- no silently accepted ownership takeover occurs during a live lease window.

### Post-incident evidence checklist

- ownership and lease audit table for all claimed items
- time-ordered manifest values for `owner_session` and `lease_expires_at`
- session logs for each claim, renew, and completion action
- proof that active leases cannot be reclaimed prematurely
- reproduction or regression test for the ownership bug

---

## 3. Manifest provenance validation regression

### Trigger conditions

- a manifest is accepted without required fields from `FLYBRAIN_HARNESS_MANIFEST_PROVENANCE_SCHEMA.md`,
- `path_template` is absolute or uses a hardcoded drive path instead of `$LOCI_FLYBRAIN_STORAGE_ROOT\...`,
- `artifact.relative_root` or file entries contain absolute paths or drive-root escapes,
- `sha256` is missing, malformed, or not lowercase 64-hex,
- `refresh.decision` is invalid or stale evidence remains active after `next_check_due` passes,
- manifest scope or provenance does not match downstream claims.

### Immediate containment actions

1. Stop all manifest ingestion and all downstream use of the affected artifact set.
2. Quarantine the manifest and all associated files before any read, sync, or promotion logic uses them.
3. Put the manifest in a non-usable state: do not read, index, or promote it.
4. Preserve the original files and manifest hash values for forensic review.
5. Disable any auto-approval path that accepts a manifest without full validation.

### Rollback steps

1. Revert to the last known-good manifest version or last verified artifact snapshot.
2. Remove the invalid manifest from active use and restore the prior active pointer or snapshot.
3. Rebuild a fresh manifest only after file existence, size, and hash verification passes.
4. Re-apply the schema and provenance gate before loading the dataset into active memory or graph state.
5. If no valid replacement can be produced quickly, restore the previous manifest/pointer and hold the dataset read-only.

### Verification before re-enable

- every required field exists and matches the canonical schema,
- every path field is relative and rooted under `$LOCI_FLYBRAIN_STORAGE_ROOT`,
- every file has a valid `sha256` and existence check,
- `scope.claim_tier`, `dataset.version_id`, and supporting evidence align with the claim boundary,
- `refresh.decision` and `next_check_due` are current and valid,
- no quarantined or stale manifest is allowed to proceed to graph or memory promotion.

### Post-incident evidence checklist

- manifest JSON before and after failure
- full integrity verification report for affected files
- hash and size mismatch records
- provenance/scope-boundary audit trail
- quarantine event JSON(s)
- release or commit hash for the reverted validation gate

---

## 4. Graph promotion pointer or state failure

### Trigger conditions

- a promotion pointer references a quarantined, missing, or unverified artifact,
- a graph or active-index pointer changes without a verified manifest,
- the active pointer is replaced while the prior good pointer still exists,
- the active graph points to the wrong dataset version, scope, or provenance,
- graph promotion state is inconsistent between manifest, pointer, and filesystem,
- a pointer is mutated without integrity or provenance validation.

### Immediate containment actions

1. Stop all promotion writes immediately.
2. Freeze the current active pointer and keep the last known-good graph runnable.
3. Disable any auto-swap logic that updates active pointer state without validation.
4. Quarantine or isolate the new candidate pointer and files until diagnosis completes.
5. Keep a read-only copy of the active graph and a rollback pointer to the last valid version.

### Rollback steps

1. Revert the active pointer to the previous verified graph or snapshot.
2. Restore the last known-good manifest and pointer metadata for the active graph.
3. If the candidate graph was partially written, remove or quarantine it and re-fetch a fresh copy under a clean staging path.
4. Ensure all write paths remain under the canonical storage root and do not use hardcoded locations.
5. If pointer updates were persisted in a registry or index, restore the registry entry and disable mutation while validation is redone.

### Verification before re-enable

- the active graph pointer resolves to a verified, non-quarantine artifact,
- manifest `integrity.verification.status` is `verified` and matches the runtime artifact state,
- `scope`, `dataset.version_id`, and evidence lineage align with the active graph,
- a promotion dry run succeeds against the target before pointer switchover,
- a smoke test confirms the graph is readable and queries correctly without stale or mismatched pointers.

### Post-incident evidence checklist

- pointer before/after values
- manifest verification status for active and candidate graph
- quarantine event for the failed candidate
- hash and file list comparisons between active and candidate graph
- smoke-test results for read/query compatibility
- exact rollback commit or pointer restoration record

---

## 5. Cross-cutting incident procedure

Use this for any queue, manifest, or promotion regression.

### Phase A: Detect and classify

- confirm the symptom: queue issue, ownership issue, manifest issue, or graph pointer issue,
- capture the exact failing API, item IDs, manifest IDs, and artifact paths,
- determine whether the issue is single-queue, single-dataset, or system-wide,
- classify as one of `queue_schema`, `lease_regression`, `manifest_validation`, or `promotion_pointer_failure`.

### Phase B: Contain and freeze

- stop new claims, new promotions, and new manifest approvals,
- freeze scheduler execution and disable automatic release lanes,
- keep the last known-good artifact, manifest, and pointer live,
- preserve evidence before mutating any state.

### Phase C: Revert

- reset only the affected component(s): queue API, ownership logic, validation layer, or pointer state,
- restore the previous verified state from backup, snapshot, or commit,
- keep the rollback minimal and reversible.

### Phase D: Verify

Before re-enabling, confirm all of the following:

- queue semantics match the canonical state machine,
- ownership and lease enforcement are active,
- manifests validate against the required schema and integrity rules,
- all promoted artifacts are verified and non-quarantined,
- active graph pointers resolve to the last known-good state,
- no stale `next_check_due`, `claim_tier`, or integrity mismatch remains unaddressed.

### Phase E: Re-enable with guardrails

- re-enable in controlled sequence: queue first, then manifest validation, then promotion writes,
- run targeted dry runs with limited blast radius before broad activation,
- require explicit review for every production override.

---

## 6. Operational guardrails

- never bypass provenance failure with a local exception,
- never allow a promotion pointer to move before validation completes,
- never allow queue takeover while a lease is still valid,
- treat all queue and promotion regressions as unsafe until proven otherwise,
- keep manual overrides out of the mainline control path unless explicitly approved and logged.

---

## 7. Recovery standards before declaring stable

A rollback is not complete until all of the following are true:

- no active out-of-spec queue items remain,
- no invalid queue transitions are still accepted,
- all manifests in use pass validation,
- all active pointers resolve to verified artifacts,
- no quarantined or stale artifact is still promoted,
- an operator has reviewed the final evidence and signed off on re-enable.

---

## 8. Evidence retention and audit standard

Retain the following until incident closure:

- queue item snapshots,
- manifest versions and hash values,
- ownership and lease provenance,
- active and candidate pointer values,
- quarantine records and failure reasons,
- operator override and approval records.

This is the minimum chain of custody required to prove the incident was contained, rolled back, and recovered without silent promotion of invalid state.
