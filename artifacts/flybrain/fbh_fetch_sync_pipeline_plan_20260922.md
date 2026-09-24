# FlyBrain fetch/sync pipeline plan artifact (2026-09-22)

## Decision

Adopt a staged fetch/sync pipeline with explicit run modes (`dry-run`, `plan`, `execute`) and deterministic idempotency/resume behavior for phase-1 local FlyBrain artifacts.

## Scope alignment

- Storage root and writes are rooted at `LOCI_FLYBRAIN_STORAGE_ROOT`.
- Allowed phase-1 datasets: `hb` (`neuprint_JRC_Hemibrain_1point2point1`) and `fw` (`flywire783`).
- Manifest contract: `fbh-manifest/v1` with required provenance, scope, refresh, and integrity fields.
- Path safety and destructive-op boundaries follow `docs/FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md`.

## Pipeline stages

1. Preflight policy gate (paths, budget, scope)
2. Plan/idempotency derivation
3. Source provenance fetch
4. Resumable acquisition into cache + source snapshot sealing
5. Integrity verification + manifest write
6. Dataset-specific sync/promotion (`hb` candidate->promotion pointer, `fw` metadata finalize)
7. Finalize + retention-safe closure

## Resume + idempotency

- Stage idempotency key:  
  `<dataset_symbol>|<version_id>|<stage_name>|<source_fingerprint>|<pipeline_version>`
- Resume state persisted in stage-state files and download state files.
- Existing checksum-matching artifacts are treated as `already_materialized` and skipped.
- Promotion is atomic and compare-and-set guarded.

## Run modes

- `dry-run`: validation and predicted writes only; log-only side effects.
- `plan`: writes plan artifacts and key derivations, but no promotion/destructive actions.
- `execute`: full fetch/sync/promotion with integrity enforcement.

## Failure boundaries

Always stop on:

- path-policy violations
- checksum mismatches for required files
- manifest schema violations
- root/destructive safety violations
- promotion corruption risk

Retry/resume allowed for:

- transient remote/network errors
- source throttling
- temporary lock contention

## Metadata contract to write

- run descriptor + config fingerprint
- per-stage timeline
- dataset idempotency keys and outcomes
- source provenance envelope
- file and manifest checksums
- refresh decision metadata
- structured failure records

## References

- `docs/FLYBRAIN_FETCH_SYNC_PIPELINES_PLAN.md`
- `docs/FLYBRAIN_HARNESS_STORAGE_LAYOUT.md`
- `docs/FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md`
- `docs/FLYBRAIN_DATASET_PROVENANCE_MATRIX.md`
- `docs/FLYBRAIN_HARNESS_MANIFEST_PROVENANCE_SCHEMA.md`
