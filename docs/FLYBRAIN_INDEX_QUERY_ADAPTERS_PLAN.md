# FlyBrain index and query adapters plan

## Purpose

Define the adapter layer that lets Loci route FlyBrain reads through local, reproducible harness data before falling back to remote/live sources.

This plan is aligned to the current storage guardrails and dataset scope:

- `docs/FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md`
- `docs/FLYBRAIN_HARNESS_STORAGE_LAYOUT.md`
- `docs/FLYBRAIN_FETCH_SYNC_PIPELINES_PLAN.md`
- `docs/FLYBRAIN_DATASET_PROVENANCE_MATRIX.md`

## Core decision

Use a single adapter registry with explicit dataset, version, and query semantics, instead of ad hoc logic spread across one-off scripts or notebooks.

All paths remain under the configured root variable:

- `LOCI_FLYBRAIN_STORAGE_ROOT`

No machine-specific drive letter or hardcoded `F:\` path is allowed in code or documentation.

## Adapter model

Each adapter exposes the same contract:

- `dataset_symbol`
- `version_id`
- `scope_tags` (sex, stage, anatomy, access method)
- `supports_query(query_kind)`
- `resolve_path()`
- `open_snapshot()`
- `query()`
- `provenance_envelope()`

## Required adapters

### 1. `HbGraphLocalAdapter`

Use for the primary local hemibrain graph snapshot.

Required config:

- dataset symbol: `hb`
- version pin: `neuprint_JRC_Hemibrain_1point2point1`
- root: `$LOCI_FLYBRAIN_STORAGE_ROOT\graph\hb\neuprint_JRC_Hemibrain_1point2point1\`

Supported query classes:

- connectivity lookup
- class-neighbor traversal
- deterministic graph replay / smoke verification

Execution gates (must all pass before execute):

- manifest canonical self-hash matches `integrity.manifest_sha256`
- each `integrity.files` entry passes path containment + sha256/size checks
- `integrity.verification.status == "verified"` with `verified_at`
- `promotion/active_pointer.json` exists and `active=true`
- `refresh.decision != "rollback"` and `next_check_due` is not expired

Not supported:

- full-brain or non-hemibrain generalization without an explicit scope note

### 2. `FwMetadataLocalAdapter`

Use for FlyWire metadata snapshots and entity normalization.

Required config:

- dataset symbol: `fw`
- version pin: `flywire783`
- root: `$LOCI_FLYBRAIN_STORAGE_ROOT\snapshots\fw\flywire783\metadata\`

Supported query classes:

- entity normalization
- annotation lookup
- provenance and version check
- metadata-level comparison

Not supported:

- local full FlyWire adjacency mirror in phase 1

Execution gates (must all pass before execute):

- manifest canonical self-hash matches `integrity.manifest_sha256`
- `integrity.files` is non-empty and passes path containment + sha256/size checks
- `integrity.verification.status == "verified"` with `verified_at`
- `refresh.decision != "rollback"` and `next_check_due` is not expired

### 3. `RemoteFallbackAdapter`

Use only when the local adapter has no compatible data for the requested query or the user explicitly requests a live check.

Rules:

- tag results as `remote_fallback`
- record dataset, version, and query settings in the provenance envelope
- never silently replace a local scope with a remote one without explicit notes

## Query routing rules

### Priority order

1. Exact local dataset/version match in the configured storage root
2. Local metadata or snapshot index with provenance
3. Remote live query if the user explicitly requests or policy allows fallback

### Local-first routing

Prefer the local adapter when all of the following hold:

- dataset symbol matches a pinned local snapshot
- the requested query type is supported by that local snapshot
- the query scope respects the dataset boundary (sex, stage, anatomy)

### Auto-fallback guardrails

Fallback is allowed only when:

- the local dataset does not support the requested operation,
- the request is explicitly marked as an external validation pass,
- the result persists a provenance note indicating the source change.

## Required index layers

### Manifest index

Keep a manifest index at:

- `$LOCI_FLYBRAIN_STORAGE_ROOT\snapshots\<dataset>\<version>\manifest\`

This stores:

- dataset symbol
- version id
- source URI
- checksum manifest
- refresh policy
- prepared query compatibility tags

### Query capability index

Keep an adapter capability map for each dataset and version:

- `connectivity_supported`
- `metadata_supported`
- `graph_snapshot_supported`
- `scope_tags`
- `known_limitations`

This makes routing deterministic and avoids implicit assumptions about dataset equivalence.

## Provenance contract

Every query response should carry an envelope with:

- dataset symbol
- version identifier
- access path
- query kind
- optional `source=local` or `source=remote_fallback`
- scope tags
- result warnings or `count_status` notes

This is where Loci keeps the dataset boundary explicit instead of flattening multiple evidence sources into one vague fact.

## Failure handling

Stop and require operator attention when:

- root path validation fails
- manifest is missing or malformed
- requested dataset version does not match the pinned local snapshot
- query scope violates a known dataset boundary
- adapter capability says a local result is unsupported but remote fallback was not explicitly approved
- any manifest/hash/file-integrity guard fails
- hb active promotion pointer is missing/inactive
- verification status is not `verified`

Retry/resume is allowed only for transient issues such as:

- lock contention
- temporary network access to remote fallback
- staged download/cache recovery

Fail-closed rule:

- do not return success-shaped local results when integrity/promotion gates fail.
- return explicit adapter error and require operator remediation.

## Implementation checklist

- define adapter registry in code
- add dataset capability manifest for `hb` and `fw`
- implement local path resolution through the shared storage helper
- add adapter-specific provenance writer
- validate query routing with deterministic unit tests
- prune stale capability metadata on version pin updates

Guardrail test references:

- `mcp/tests/test_flybrain_hb_adapter.py`
- `mcp/tests/test_flybrain_fw_metadata_adapter.py`

## Recommended next implementation step

Implement the adapter registry and a minimal `hb` / `fw` capability manifest behind the storage root helper before adding a deeper query execution layer.

---

## Concrete hb adapter contract

See [FLYBRAIN_HB_LOCAL_ADAPTER_CONTRACT.md](./FLYBRAIN_HB_LOCAL_ADAPTER_CONTRACT.md) for the execution-ready hb adapter contract, result envelope, provenance envelope, error codes, and rollback conditions.

---

Status: ready for implementation as the next FlyBrain harness integration layer.
