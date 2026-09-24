# FlyBrain artifact integrity + quarantine policy

## Purpose

Define the verification gate for local FlyBrain artifacts so partial downloads, size/hash mismatches, and corrupted payloads are blocked before sync/promotion.

This policy aligns with:

- `docs/FLYBRAIN_HARNESS_MANIFEST_PROVENANCE_SCHEMA.md`
- `docs/FLYBRAIN_HARNESS_STORAGE_LAYOUT.md`
- `docs/FLYBRAIN_NEO4J_NEUPRINT_LOCAL_STACK_PLAN.md`

## Variable-driven path contract

All paths must remain variable-driven and rooted under:

- `LOCI_FLYBRAIN_STORAGE_ROOT`

Verification and quarantine paths:

- staging: `$LOCI_FLYBRAIN_STORAGE_ROOT\cache\imports\<dataset_symbol>\<version_id>\`
- quarantine: `$LOCI_FLYBRAIN_STORAGE_ROOT\cache\quarantine\<dataset_symbol>\<version_id>\<manifest_id>\<event_ts>\`
- quarantine report: `<quarantine_path>\quarantine_event.json`

No hardcoded drive-letter paths are allowed.

## Manifest/provenance alignment

Use `fbh-manifest/v1` as the baseline and apply this integrity profile:

- required per file:
  - `integrity.files[*].relative_path`
  - `integrity.files[*].sha256`
  - `integrity.files[*].size_bytes` (required when source metadata provides expected size)
- recommended extension:
  - `integrity.files[*].hashes.sha256` (same value as `sha256`)
  - `integrity.files[*].hashes.sha512` (optional secondary strong hash)
  - `integrity.verification.status` (`pending`, `verified`, `quarantined`)
  - `integrity.verification.verified_at`
  - `integrity.verification.failure_reason` (when quarantined)

`sha256` remains the interoperability floor for all readers/writers.

## Accepted hash algorithms

### Allowed for pass/fail integrity gates

1. `sha256` (required)
2. `sha512` (optional, additive)
3. `blake3` (optional, additive; only if both writer and verifier support it)

### Not accepted as sole pass/fail proof

- `md5`
- `sha1`

If upstream only provides weak hashes, local verification still requires local `sha256`.

## Verification order (must run in order)

1. **Path + manifest preflight**
   - validate all candidate paths under `LOCI_FLYBRAIN_STORAGE_ROOT`
   - reject absolute or drive-hardcoded manifest file paths
2. **Presence check**
   - every listed `relative_path` must exist in staging
3. **Expected-size check**
   - if `size_bytes` exists, enforce exact byte equality
4. **Partial-download checks**
   - reject `.partial`, `.tmp`, or in-progress marker files
   - reject `observed_size < expected_size`
   - reject zero-byte file when expected size is non-zero
   - reject archive/container parse failures (for zip/tar/parquet/neo4j dumps)
5. **Hash verification**
   - verify `sha256` for every file
   - if secondary hashes are provided, verify all provided algorithms
6. **Aggregate manifest decision**
   - all files must pass to set `integrity.verification.status=verified`
   - any failure sets status to `quarantined` and blocks sync/promotion

## Expected-size handling rules

- **Known expected size:** strict equality required.
- **Expected size omitted by source:** allow hash-only verification, but mark as `size_unavailable` in verification metadata.
- **Transferred bytes exceed expected size:** treat as corruption; quarantine.
- **Content-Length mismatch during fetch:** treat as partial/corrupt; quarantine before unpack/import.

## Mismatch and failure actions

On any size/hash/parse mismatch:

1. Do not promote the artifact.
2. Move failed payload(s) into quarantine path under `cache\quarantine\...`.
3. Emit `quarantine_event.json` with:
   - `manifest_id`, `dataset_symbol`, `version_id`
   - `relative_path`
   - `expected_size_bytes`, `observed_size_bytes`
   - expected vs observed hash values
   - `failure_reason`
   - `detected_at`
4. Mark manifest verification status as `quarantined`.
5. Keep the last known-good promoted dataset active (`refresh.decision` stays `no_change` unless rollback action is explicitly required).

No fail-open path is permitted.

## Quarantine retention and revalidation policy

- Quarantine data is immutable evidence; do not overwrite prior events.
- Retain quarantine events for at least 30 days or until incident closure (whichever is longer).
- Revalidation requires a fresh fetch into a new staging path; never restore from quarantined payload directly.
- Revalidation pass criteria:
  1. strict size match (if available),
  2. full required hash match (`sha256`),
  3. parse/smoke-open success,
  4. manifest verification status updated to `verified`.
- Only verified artifacts may proceed into sync/promotion stages.

## Sync pipeline integration

Gate mapping for planned local sync pipeline:

1. **Acquire lane:** download/export into `$LOCI_FLYBRAIN_STORAGE_ROOT\cache\imports\...`
2. **Verify lane (this policy):** run size/hash/partial checks; quarantine failures
3. **Promote lane:** only consumes `verified` manifests
4. **Runtime/sync lane:** reads promoted pointer; never reads quarantined paths

This keeps integrity enforcement consistent with non-destructive blue/green promotion planning.
