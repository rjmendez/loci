# FlyBrain harness manifest/provenance schema decision (2026-09-22)

## Decision

Adopt `fbh-manifest/v1` as the local artifact contract for dataset snapshots and metadata bundles.

The schema captures:

- reproducibility (`dataset.symbol`, `version_id`, source/access metadata),
- refresh tracking (`refresh` decision and cadence),
- integrity checks (`manifest_sha256`, per-file SHA-256),
- claim scoping (`scope` with sex/stage/anatomy/evidence tier),
- lineage (`derived_from`, ingest pipeline run metadata).

## Path safety alignment

- All path fields are variable-driven from `LOCI_FLYBRAIN_STORAGE_ROOT`.
- No hardcoded drive letters are allowed in manifest fields.
- File records use relative paths under the artifact root.

## Required use in harness writers

1. Emit one manifest per ingest/update.
2. Populate required reproducibility + scope fields before publishing.
3. Compute and persist per-file SHA-256.
4. Record refresh decision (`no_change`, `patch_refresh`, `major_bump`, `rollback`).
5. Link superseding manifests via `refresh.supersedes_manifest_id`.

## Required use in harness readers

1. Reject manifests with missing required fields.
2. Reject absolute/hardcoded path fields.
3. Verify file hashes before trusting artifact content.
4. Enforce claim scoping boundaries in downstream summaries.

## Examples and validation

The complete schema, examples, and validation fragment are documented in:

- `docs/FLYBRAIN_HARNESS_MANIFEST_PROVENANCE_SCHEMA.md`

This artifact is the rollout decision record; the docs page is the normative writer/reader guide.

Integrity failure handling, partial-download detection, and quarantine/revalidation behavior are defined in:

- `docs/FLYBRAIN_ARTIFACT_INTEGRITY_QUARANTINE_POLICY.md`
