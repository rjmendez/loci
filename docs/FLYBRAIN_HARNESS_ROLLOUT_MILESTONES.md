# FlyBrain harness rollout milestones (inventory -> reproducible local query harness)

## Purpose

Define the minimal safe rollout sequence from dry-run inventory to the first reproducible local query harness run, using only config/env-driven storage conventions.

This sequencing is aligned with:

- [FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md](./FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md)
- [FLYBRAIN_HARNESS_STORAGE_LAYOUT.md](./FLYBRAIN_HARNESS_STORAGE_LAYOUT.md)
- [FLYBRAIN_FETCH_SYNC_PIPELINES_PLAN.md](./FLYBRAIN_FETCH_SYNC_PIPELINES_PLAN.md)
- [FLYBRAIN_NEO4J_NEUPRINT_LOCAL_STACK_PLAN.md](./FLYBRAIN_NEO4J_NEUPRINT_LOCAL_STACK_PLAN.md)
- [FLYBRAIN_HARNESS_MANIFEST_PROVENANCE_SCHEMA.md](./FLYBRAIN_HARNESS_MANIFEST_PROVENANCE_SCHEMA.md)

## Current roadmap status (2026-09-23)

The rollout is now in a staged operational posture, not a blank-slate plan:

- **M0 dry-run inventory gate:** complete. Storage-path checks, dataset scope validation, and deterministic preflight proof are in place.
- **M1 plan freeze gate:** complete. Stage plans, idempotency keys, and write-intent manifests are persisted before execute mode.
- **M2 snapshot + manifest gate:** complete. Snapshot acquisition, checksum verification, and manifest-schema enforcement are part of the active contract.
- **M3 first reproducible local query harness gate:** partial/in progress. The harness and guardrails are in place, but the final training/eval chain remains blocked by the dataset-builder and specialist-training outputs.
- **Merge/deploy gate for testing:** PR stack is active, but production deployment stays behind merge readiness. The current stack is ordered: the base PR must land first, then the stacked PR can merge and deploy to test.

### Current execution blockers

The remaining roadmap blockers are not infrastructure-only; they are product/validation gates:

1. `bc-trainlog-dataset-builder` must produce a reproducible dataset artifact with fingerprints, split metadata, and label balance notes.
2. `bc-trainlog-train-specialists` must train the specialist models from the canonical corpus.
3. `bc-trainlog-eval-and-gate` must validate the trained specialists with holdout/shadow checks and fail-closed promotion criteria.

This keeps the rollout honest: no broad promotion or deployment before the output artifact and evaluation gate are green.

## Roadmap updates vs. the original plan

The original milestones remain valid, but the operational sequence is now explicit:

1. **Preflight + inventory** (complete)
2. **Plan freeze + resume-safe staging** (complete)
3. **Snapshot / manifest verification** (complete)
4. **Reproducible harness query run** (active)
5. **Specialist dataset build** (active required dependency)
6. **Specialist training + eval gate** (active required dependency)
7. **Stacked PR merge + test deployment** (next operational gate)

This order is intentionally conservative: it preserves fail-closed behavior and keeps deployment from racing ahead of data quality and evaluation evidence.

## Required variable contract

No hardcoded drive letters or machine-specific absolute paths are allowed.

Required:

- `LOCI_FLYBRAIN_STORAGE_ROOT`
- `LOCI_FLYBRAIN_RUN_MODE` (`dry-run` | `plan` | `execute`)

Rollout controls:

- `LOCI_FLYBRAIN_DATASET_SCOPE` (phase-1 default: `hb,fw`)
- `LOCI_FLYBRAIN_DATASET_VERSION_PINS` (for example `hb=neuprint_JRC_Hemibrain_1point2point1;fw=flywire783`)
- `LOCI_FLYBRAIN_MAX_RETRIES`
- `LOCI_FLYBRAIN_RETRY_BACKOFF_SECONDS`
- `LOCI_FLYBRAIN_CONTINUE_ON_DATASET_FAILURE`

All writable targets must resolve under `$LOCI_FLYBRAIN_STORAGE_ROOT\` and pass path-policy guards before side effects.

## Minimal milestone boundaries

### Milestone 0 - Dry-run inventory gate

**Goal:** Prove storage, scope, and policy preconditions without touching dataset payload trees.

**Allowed side effects:**

- `logs\pipelines\<run_id>\preflight.json`
- `logs\pipelines\<run_id>\inventory.json`

**Exit criteria:**

1. Path guard validation passes for all resolved descendants.
2. Scope/pin set is phase-1 compliant (`hb` + `fw` only unless explicitly expanded).
3. Budget/state classification is recorded with no hard-stop violation.

### Milestone 1 - Plan freeze gate

**Goal:** Materialize deterministic write intent and resume/idempotency keys before execute mode.

**Allowed side effects:**

- `logs\pipelines\<run_id>\plan.json`
- `logs\pipelines\<run_id>\stage_state\*.json`

**Exit criteria:**

1. Stage graph is complete for each pinned dataset/version.
2. Idempotency keys are persisted per stage using deterministic input fingerprints.
3. Predicted writes stay inside allowed subtrees (`snapshots`, `cache`, `graph`, `logs`).

### Milestone 2 - Execute snapshot + manifest gate

**Goal:** Acquire and verify phase-1 source artifacts with resumable/idempotent semantics, without promoting a graph yet.

**Allowed side effects:**

- `cache\downloads\<dataset>\<version>\*`
- `snapshots\<dataset>\<version>\source\*`
- `snapshots\<dataset>\<version>\manifest\manifest.json`
- `snapshots\<dataset>\<version>\manifest\manifest.sha256`

**Exit criteria:**

1. Required files are present with checksum/manifest integrity pass.
2. Resume state is present for interrupted reruns.
3. Manifest schema fields are complete and path-safe (no absolute path leakage).

### Milestone 3 - First reproducible local query harness gate

**Goal:** Stand up the first reproducible local query harness run against pinned local data.

**Allowed side effects (in addition to Milestone 2):**

- `graph\hb\<version_id>\neo4j-store\candidate-<build_id>\*`
- `graph\hb\<version_id>\promotion\*` (only after candidate checks pass)
- `logs\harness\<harness_run_id>\query_pack.json`
- `logs\harness\<harness_run_id>\results.json`

**Exit criteria:**

1. Candidate graph build/import passes structural and smoke-query checks.
2. Promotion pointer metadata is written atomically (no in-place active-store mutation).
3. A reproducible query pack is persisted with:
   - dataset/version pins,
   - run mode and config fingerprint,
   - query inputs,
   - result hashes and timestamps.
4. Re-running the same query pack against the same promoted dataset/version yields matching structural result hashes (or documented, expected variance contract).

## Stop/go policy between milestones

Progression is strictly sequential: M0 -> M1 -> M2 -> M3.

Always stop (no auto-advance) on:

- path-policy violation,
- required checksum mismatch,
- manifest schema failure,
- unapproved scope/version pin,
- promotion pointer integrity risk.

Retry/resume is allowed only for transient fetch/lock failures and must reuse existing stage idempotency keys.

## Canonical writable layout reminders

All path templates are under `$LOCI_FLYBRAIN_STORAGE_ROOT\`:

- `graph\`
- `snapshots\`
- `cache\`
- `backups\`
- `logs\`

The rollout never writes outside this root and never relies on machine-specific drive lettering.

---

Status: planned and ready to drive phased implementation from dry-run inventory to first reproducible local query harness.
