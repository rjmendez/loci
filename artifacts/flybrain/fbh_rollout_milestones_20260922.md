# FlyBrain rollout milestones artifact (2026-09-22)

## Decision

Adopt a four-gate rollout from dry-run inventory to first reproducible local query harness, with explicit stop/go boundaries and env-driven storage contracts only.

## Milestones

1. **M0 - Dry-run inventory gate**
   - Preflight and inventory only; log-only writes under `logs\pipelines\<run_id>\`.
2. **M1 - Plan freeze gate**
   - Deterministic stage plan + idempotency keys persisted; no dataset payload writes yet.
3. **M2 - Execute snapshot + manifest gate**
   - Resumable acquisition and manifest/integrity completion under `cache\` and `snapshots\`.
4. **M3 - First reproducible local query harness gate**
   - Candidate hb graph build, safe promotion pointer update, and reproducible query pack + results with hash replay contract.

## Variable contract

Required:

- `LOCI_FLYBRAIN_STORAGE_ROOT`
- `LOCI_FLYBRAIN_RUN_MODE`

Pinned rollout controls:

- `LOCI_FLYBRAIN_DATASET_SCOPE`
- `LOCI_FLYBRAIN_DATASET_VERSION_PINS`
- `LOCI_FLYBRAIN_MAX_RETRIES`
- `LOCI_FLYBRAIN_RETRY_BACKOFF_SECONDS`
- `LOCI_FLYBRAIN_CONTINUE_ON_DATASET_FAILURE`

No hardcoded drive letters or machine-specific absolute paths are permitted.

## Safety boundaries

Always stop on path-policy failures, checksum/manifest failures, unapproved scope pins, or promotion-integrity risk. Retry/resume is allowed only for transient fetch/lock failures using the same idempotency keys.

## References

- `docs/FLYBRAIN_HARNESS_ROLLOUT_MILESTONES.md`
- `docs/FLYBRAIN_FETCH_SYNC_PIPELINES_PLAN.md`
- `docs/FLYBRAIN_NEO4J_NEUPRINT_LOCAL_STACK_PLAN.md`
- `docs/FLYBRAIN_HARNESS_STORAGE_LAYOUT.md`
- `docs/FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md`
