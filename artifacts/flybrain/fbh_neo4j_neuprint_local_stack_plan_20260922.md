# FlyBrain harness local Neo4j/neuPrint stack plan (2026-09-22)

## Scope

Define the deployment and import plan for a hemibrain-scale local Neo4j/neuPrint stack under the configured FlyBrain storage root.

## Root/storage contract

- Canonical root env: `LOCI_FLYBRAIN_STORAGE_ROOT`
- All runtime/import/cache/log paths must resolve under that root.
- No hardcoded absolute machine paths.

## Dataset/version pin (phase 1)

- `dataset_symbol`: `hb`
- `version_id`: `neuprint_JRC_Hemibrain_1point2point1`
- Upgrade policy: explicit pin bump PR only

## Resource profiles

- Minimal: 8 vCPU, 32 GB RAM, ~40-60 GB graph storage, ~25-40 GB import scratch
- Recommended: 16 vCPU, 64 GB RAM, ~60-90 GB graph storage, ~40-70 GB import scratch

## Import stages

1. Preflight path/storage guard checks
2. Acquire pinned snapshot + manifest/checksums
3. Verify and normalize import inputs
4. Offline candidate graph build (no in-place active-store mutation)
5. Atomic promotion pointer switch
6. Post-promotion scratch cleanup + rollback window retention

## Operational boundaries

- Enforce allowlist-only write-path policy.
- Block nonessential imports in critical/hard-stop storage states.
- Never run destructive ops on root/dataset anchors.
- Promotion requires manifest completeness + checksum pass.

## Source links

- `docs/FLYBRAIN_NEO4J_NEUPRINT_LOCAL_STACK_PLAN.md`
- `docs/FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md`
- `docs/FLYBRAIN_DATASET_PROVENANCE_MATRIX.md`
