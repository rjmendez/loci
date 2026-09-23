# FlyBrain artifact integrity + quarantine policy decision (2026-09-22)

## Decision

Adopt a strict verification gate for local FlyBrain artifacts before sync/promotion:

1. path/manifest preflight,
2. file presence,
3. expected-size checks,
4. partial-download detection,
5. required `sha256` hash verification.

Any failure quarantines the artifact and blocks promotion.

## Variable-driven path contract

- storage root: `LOCI_FLYBRAIN_STORAGE_ROOT`
- staging: `$LOCI_FLYBRAIN_STORAGE_ROOT\cache\imports\<dataset>\<version>\`
- quarantine: `$LOCI_FLYBRAIN_STORAGE_ROOT\cache\quarantine\<dataset>\<version>\<manifest_id>\<event_ts>\`

No hardcoded drive-letter paths.

## Hash and size policy

- Required gate algorithm: `sha256`
- Optional additive algorithms: `sha512`, `blake3`
- Disallowed as sole proof: `md5`, `sha1`
- If `size_bytes` is available in manifest/source metadata, byte-exact equality is mandatory.

## Quarantine and revalidation

- Quarantine is immutable evidence; keep event records (`quarantine_event.json`) with mismatch details.
- Retain quarantined artifacts for at least 30 days or incident closure.
- Revalidation must re-fetch into a fresh staging path; quarantined payloads are never promoted directly.
- Promotion is allowed only after full verification status is `verified`.

## Alignment links

- `docs/FLYBRAIN_ARTIFACT_INTEGRITY_QUARANTINE_POLICY.md`
- `docs/FLYBRAIN_HARNESS_MANIFEST_PROVENANCE_SCHEMA.md`
- `docs/FLYBRAIN_NEO4J_NEUPRINT_LOCAL_STACK_PLAN.md`
