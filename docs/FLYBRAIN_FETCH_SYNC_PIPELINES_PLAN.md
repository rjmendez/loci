# FlyBrain fetch/sync pipeline plan (resumable + idempotent)

## Purpose

Define a variable-driven, safety-bounded pipeline for local FlyBrain dataset fetch and sync operations that is:

- resumable across failures and restarts,
- idempotent across retries,
- aligned to current phase-1 dataset scope and manifest schema,
- operable through clear run modes (`dry-run`, `plan`, `execute`).

This plan is aligned with:

- [FLYBRAIN_HARNESS_STORAGE_LAYOUT.md](./FLYBRAIN_HARNESS_STORAGE_LAYOUT.md)
- [FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md](./FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md)
- [FLYBRAIN_DATASET_PROVENANCE_MATRIX.md](./FLYBRAIN_DATASET_PROVENANCE_MATRIX.md)
- [FLYBRAIN_HARNESS_MANIFEST_PROVENANCE_SCHEMA.md](./FLYBRAIN_HARNESS_MANIFEST_PROVENANCE_SCHEMA.md)
- [FLYBRAIN_NEO4J_NEUPRINT_LOCAL_STACK_PLAN.md](./FLYBRAIN_NEO4J_NEUPRINT_LOCAL_STACK_PLAN.md)

## Config contract (variable-driven only)

No hardcoded drive letters or machine-specific absolute paths are permitted.

Required:

- `LOCI_FLYBRAIN_STORAGE_ROOT` (canonical writable root)
- `LOCI_FLYBRAIN_RUN_MODE` (`dry-run` | `plan` | `execute`)

Recommended pipeline controls:

- `LOCI_FLYBRAIN_DATASET_SCOPE` (default phase-1: `hb,fw`)
- `LOCI_FLYBRAIN_DATASET_VERSION_PINS` (for example `hb=neuprint_JRC_Hemibrain_1point2point1;fw=flywire783`)
- `LOCI_FLYBRAIN_MAX_RETRIES` (integer; default 3)
- `LOCI_FLYBRAIN_RETRY_BACKOFF_SECONDS` (base delay; default 10)
- `LOCI_FLYBRAIN_CONTINUE_ON_DATASET_FAILURE` (`false` by default for strict runs)
- `LOCI_FLYBRAIN_BUDGET_SOFT_CAP_GB` / `LOCI_FLYBRAIN_BUDGET_HARD_CAP_GB` (policy-aligned overrides only)

All paths must be resolved and validated through shared storage/path-guard helpers before side effects.

## Phase-1 dataset scope

The pipeline must enforce the currently approved initial scope:

1. `hb` local graph snapshot (`neuprint_JRC_Hemibrain_1point2point1`)
2. `fw` metadata snapshot (`flywire783`)

Explicitly out of phase-1:

- full local FlyWire adjacency/chunkedgraph mirror,
- adding `mc`, `mv`, `BANC`, or other large mirrors without a scope/policy update.

## Canonical writable layout touched by this pipeline

All path templates are rooted at `$LOCI_FLYBRAIN_STORAGE_ROOT\`.

- `snapshots\<dataset_symbol>\<version_id>\source\` (downloaded raw payloads)
- `snapshots\<dataset_symbol>\<version_id>\manifest\` (artifact manifests/checksums)
- `cache\downloads\<dataset_symbol>\<version_id>\` (resumable partials/chunks/temp indexes)
- `graph\hb\<version_id>\neo4j-store\candidate-<build_id>\` (hb only, execute mode)
- `graph\hb\<version_id>\promotion\` (hb promotion pointer/metadata)
- `logs\pipelines\<run_id>\` (run journal, per-stage telemetry, failures)

## Pipeline stages

### Stage 0 — preflight and policy gate

Inputs:

- run mode
- dataset scope + pins
- storage policy thresholds

Actions:

1. Resolve and validate root/layout paths.
2. Validate every resolved path target against allowlist + deny rules.
3. Evaluate storage pressure level (watch/soft/critical/hard-stop).
4. Validate dataset scope is phase-1 compliant.

Outputs/metadata:

- `logs\pipelines\<run_id>\preflight.json`
- run journal entry with resolved config fingerprint

Hard-fail boundaries:

- invalid root/path guard failure
- hard-stop budget with write-requiring mode
- unapproved dataset/version target

### Stage 1 — planning and idempotency key derivation

Actions:

1. For each dataset/version pin, derive deterministic execution plan.
2. Compute idempotency keys (see below).
3. Discover prior stage state files for resume.

Outputs:

- `logs\pipelines\<run_id>\plan.json`
- optional plan artifact in `dry-run`/`plan` mode

### Stage 2 — source discovery + remote provenance fetch

Actions:

1. Resolve source descriptors (dataset symbol, version id, source URI, license).
2. Capture remote file list/expected checksums where available.
3. Persist source envelope for replay.

Outputs:

- `snapshots\<dataset>\<version>\manifest\source_provenance.json`
- run journal stage record

Retry/resume:

- Safe to rerun fully (overwrite-by-content hash or compare-and-skip).

### Stage 3 — resumable acquisition into cache

Actions:

1. Download/fetch into `cache\downloads\...\*.partial`.
2. Track chunk progress and byte offsets in stage-state metadata.
3. On success, seal partial into content-addressed final filename in `source\`.

Outputs:

- partial state: `cache\downloads\<dataset>\<version>\download_state.json`
- completed files: `snapshots\<dataset>\<version>\source\...`

Retry/resume:

- resume by byte range/chunk map where source supports it;
- otherwise restart file, preserving prior failure diagnostics.

### Stage 4 — integrity verification + manifest build

Actions:

1. Hash acquired files and compare to expected checksums (if provided).
2. Write manifest using `fbh-manifest/v1`.
3. Compute and store `integrity.manifest_sha256`.

Outputs:

- `snapshots\<dataset>\<version>\manifest\manifest.json`
- `snapshots\<dataset>\<version>\manifest\manifest.sha256`

Hard-fail boundaries:

- checksum mismatch
- missing required schema fields
- absolute/hardcoded path leakage in manifest

### Stage 5 — dataset-specific sync/promotion

`hb` lane (execute mode):

1. Build/import into new candidate graph path.
2. Run integrity/smoke checks.
3. Promote by atomic pointer/manifest update in `promotion\`.

`fw` lane (execute mode):

1. Materialize metadata snapshot in versioned snapshot tree.
2. No graph promotion; mark as metadata-ready.

Retry/resume:

- candidate build ids are immutable units; failed candidate remains for forensics;
- promotion step is retriable and idempotent via compare-and-set pointer semantics.

### Stage 6 — finalize, retention, and run closure

Actions:

1. Mark per-dataset final status.
2. Optionally prune only ephemeral cache descendants (policy-safe).
3. Emit run summary and next-check schedule metadata.

Outputs:

- `logs\pipelines\<run_id>\summary.json`
- dataset refresh decision updates in manifest(s)

## Idempotency model

### Idempotency key structure

Use deterministic keys per dataset stage:

`<dataset_symbol>|<version_id>|<stage_name>|<source_fingerprint>|<pipeline_version>`

Where:

- `source_fingerprint` = hash of source URI + expected file inventory/checksums (if known),
- `pipeline_version` = explicit pipeline code/version marker.

### Idempotent write rules

1. Never write directly to promoted/active graph paths.
2. Write new artifacts to candidate or temporary paths first.
3. Promote only via atomic pointer/rename metadata update.
4. Treat existing matching artifact+checksum as success (`already_materialized`).
5. Use per-stage state files to decide skip/resume/retry.

## Retry and resume semantics

Retry classes:

- transient network/service errors -> retry with exponential backoff + jitter
- integrity/schema/path-policy failures -> no automatic retry (operator action required)
- storage-threshold gate failures -> retry only after space recovery

Resume sources:

- `logs\pipelines\<run_id>\stage_state\*.json`
- `cache\downloads\<dataset>\<version>\download_state.json`
- existing manifests and hash files under `snapshots\...\manifest\`

Resume decision order:

1. If finalized success marker exists for stage idempotency key, skip stage.
2. Else if resumable state exists, continue from checkpoint.
3. Else start stage fresh.

## Run modes and side-effect boundaries

### `dry-run`

- Validate config, paths, budget gates, dataset scope, and planning logic.
- Compute idempotency keys and predicted writes.
- No dataset content writes.
- Allowed writes: run diagnostics under `logs\pipelines\<run_id>\` only.

### `plan`

- Everything in `dry-run`, plus write concrete plan artifacts and action graph.
- Optional remote metadata probes allowed.
- No graph promotion and no destructive/pruning actions.

### `execute`

- Full staged fetch/sync with integrity checks and (hb) promotion.
- Destructive actions limited to explicit ephemeral cache descendants and must pass policy guard + root protection.

## Failure handling boundaries

Per-dataset failure isolation:

- one dataset can fail while another completes only when `LOCI_FLYBRAIN_CONTINUE_ON_DATASET_FAILURE=true`;
- default strict mode stops on first dataset failure after preserving logs/state.

Non-recoverable boundaries (always stop):

- path guard/policy violation,
- root-protection violation attempt,
- manifest schema violation after verification stage,
- checksum mismatch on required files for pinned artifact,
- promotion pointer corruption risk.

Recoverable boundaries (retry/resume possible):

- remote fetch timeout/connection reset,
- transient source throttling,
- temporary lock contention on cache/log targets.

## Required metadata writes

At minimum each run must persist:

1. run descriptor (`run_id`, pipeline version, mode, config fingerprint),
2. per-stage status timeline with start/end timestamps,
3. per-dataset idempotency keys and outcomes,
4. source provenance envelope (`system`, `access_method`, `uri`, `retrieved_at`, license),
5. integrity artifacts (file hashes + manifest hash),
6. refresh decision fields (`decision`, `checked_at`, `next_check_due`, optional `supersedes_manifest_id`),
7. failure records (`error_class`, `error_code`, `retryable`, `last_attempt_at`).

## Operator sequence

1. Run `dry-run` to validate policy/scope/path gates.
2. Run `plan` to materialize concrete action plan and verify expected writes.
3. Run `execute` only after plan review.
4. On failure, rerun with same dataset/version pins and run context to resume idempotently.

---

Status: planned and aligned for implementation under current phase-1 FlyBrain harness constraints.
