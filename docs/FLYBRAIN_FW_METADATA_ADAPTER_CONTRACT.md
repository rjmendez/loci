# FlyBrain local `fw` metadata adapter contract

## Scope

This document defines the concrete, execution-ready contract for the local FlyWire metadata adapter used by the FlyBrain harness in phase 1. It is intentionally narrow: local read-only metadata enrichment only, pinned to `fw=flywire783`, and routed through `LOCI_FLYBRAIN_STORAGE_ROOT`.

This adapter does not implement local FlyWire graph traversal or adjacency queries. The harness may use it only for:

- entity normalization and canonical ID resolution,
- annotation/class lookup,
- metadata-level provenance hydration,
- scope validation for dataset-bound claims,
- local evidence packaging for reporting and replay.

It must refuse any operation that implies local FlyWire graph analytics or remote fallback masquerading as local evidence.

## Canonical dataset mapping

The adapter is pinned to the exact dataset used by the phase-1 local harness:

- `dataset_symbol`: `fw`
- `version_id`: `flywire783`
- `storage_root`: `$LOCI_FLYBRAIN_STORAGE_ROOT/snapshots/fw/flywire783/metadata/`
- `manifest`: `manifest.json` under that root
- `access_layer`: `metadata_only`
- `supported_by`: local snapshot only, never a live remote graph

This mapping is the only valid local `fw` metadata dataset for the adapter. Requests for any other version or any dataset outside `fw` must fail with `DATASET_PIN_MISMATCH`.

## Supported local read-only operations

The adapter must support only these query kinds:

1. `entity_lookup`
   - Input: a free-form label, `FlyWire` entity ID, or canonical name.
   - Output: a normalized entity row with `canonical_id`, `label`, `aliases`, `dataset_scope`, and `provenance`.

2. `class_lookup`
   - Input: class label or identifier.
   - Output: class metadata including `class_id`, `label`, `synonyms`, `superclasses`, and completeness notes.

3. `annotation_lookup`
   - Input: entity or class identifier with one or more annotation keys.
   - Output: field/value metadata rows or a map of annotation names to values.

4. `provenance_lookup`
   - Input: a canonical entity/class ID.
   - Output: local manifest provenance, snapshot lineage, and source metadata for the requested entity or cluster.

5. `scope_validation`
   - Input: a proposed claim scope with `sex`, `stage`, `anatomy`, `dataset`, and `evidence_family`.
   - Output: `allowed` or `rejected` plus explicit rejection reason and required metadata fields.

The adapter must reject all other query kinds with `UNSUPPORTED_QUERY_KIND`.

## Explicit non-supported operations

The following operations are not allowed in this phase and must always fail closed:

- any adjacency traversal (`neighbors`, `connectivity`, `graph_walk`, `synapse_path`),
- any full-graph query over FlyWire local topology,
- any cross-dataset join that mixes `fw` metadata with `hb` graph IDs without an explicit mapping class,
- any write, mutation, or cache eviction in the local metadata snapshot,
- any remote fallback that is reported as local evidence.

These must raise `UNSUPPORTED_QUERY_KIND`, `OUT_OF_SCOPE`, or `REMOTE_FALLBACK_BLOCKED` depending on the request.

## Safety gates (normative)

Every request must pass the following gates before the adapter executes:

### Gate A: root and path containment

- Resolve the root from `LOCI_FLYBRAIN_STORAGE_ROOT` only.
- Reject non-absolute or empty values.
- Reject UNC paths, drive-root paths, symlinks, and windows reparse points.
- Normalize every path and ensure the resolved metadata root remains under the configured root.
- Fail with `ROOT_NOT_CONFIGURED`, `ROOT_OUTSIDE_ALLOWLIST`, or `PATH_ESCAPE`.

### Gate B: dataset pin enforcement

- Require `request.dataset_symbol == "fw"`.
- Require `request.version_id == "flywire783"`.
- Require `request.access_layer == "metadata_only"`.
- Require the request to target `snapshots/fw/flywire783/metadata` exactly.
- Fail with `DATASET_PIN_MISMATCH` if any pin does not match.

### Gate C: read-only execution guard

- The adapter must be read-only; no writes, no temp files, no manifest mutation.
- If the request attempts any mutation, return `READ_ONLY_VIOLATION` and do not touch the snapshot.

### Gate D: manifest validity and integrity

- The snapshot root must contain `manifest.json`.
- The manifest must validate against the FlyBrain harness schema version (`fbh-manifest/v1`).
- Manifest fields must include dataset symbol, version ID, source, scope, integrity, refresh, and lineage.
- `integrity.verification.status` must be `verified`.
- `integrity.verification.verified_at` is required when status is `verified`.
- `integrity.manifest_sha256` must be lowercase 64-hex and must match a canonical self-hash digest.
- Canonical self-hash rule: set `integrity.manifest_sha256=""` in the canonical JSON payload, serialize with stable key ordering, then compute sha256.
- `integrity.files` must be non-empty and each entry must:
  - use a safe relative path (no absolute/drive/traversal),
  - be unique after case-folding,
  - avoid `.partial`, `.tmp`, `.inprogress` paths,
  - include lowercase 64-hex `sha256`,
  - resolve under `artifact.relative_root`,
  - exist on disk,
  - match `size_bytes` when present,
  - match file SHA256.
- Any mismatch must immediately fail with `MANIFEST_INVALID` or `INTEGRITY_MISMATCH`.

### Gate E: capability gating

- The `query_kind` must be one of the supported local metadata operations.
- Any request for a prohibited operation or unregistered capability must fail `UNSUPPORTED_QUERY_KIND`.

### Gate F: provenance boundary

- Every successful query must emit a provenance envelope with `dataset_symbol`, `version_id`, `query_kind`, `scope_tags`, `manifest_id`, `manifest_sha256`, `access_method`, and `replay_fingerprint`.
- No result may be emitted without a provenance envelope.
- Fallback to remote or live data is forbidden in this adapter. Fail `REMOTE_FALLBACK_BLOCKED` instead of silently degrading.

## Error codes

The adapter must return deterministic error objects using the following codes:

- `ROOT_NOT_CONFIGURED`
- `ROOT_OUTSIDE_ALLOWLIST`
- `PATH_ESCAPE`
- `DATASET_PIN_MISMATCH`
- `REQUEST_INVALID`
- `UNSUPPORTED_QUERY_KIND`
- `MANIFEST_MISSING`
- `MANIFEST_INVALID`
- `INTEGRITY_MISMATCH`
- `READ_ONLY_VIOLATION`
- `OUT_OF_SCOPE`
- `REMOTE_FALLBACK_BLOCKED`
- `QUERY_TIMEOUT`
- `DATASET_UNAVAILABLE`
- `PROMOTION_STATE_INVALID`

All error responses must have the same top-level shape:

```json
{
  "ok": false,
  "error": {
    "code": "DATASET_PIN_MISMATCH",
    "message": "Dataset/version pin mismatch for fw metadata adapter",
    "request_id": "<uuid>",
    "dataset_symbol": "fw",
    "version_id": "flywire783",
    "details": {
      "expected_symbol": "fw",
      "expected_version": "flywire783",
      "received_symbol": "fw",
      "received_version": "flywire782"
    }
  }
}
```

## Rollback and fail-closed behavior

This adapter is strictly local-read-only and must never promote a snapshot during execution. Rollback means the harness keeps the last known-good manifest pointer and rejects the request without changing the active local metadata state.

Rollback conditions:

- any root/path guard failure,
- any manifest validation or integrity mismatch,
- any version pin mismatch,
- any unsupported query kind,
- any request that would require remote fallback,
- any write or mutation detected,
- any request that tries to cross dataset scope with a non-allowed mapping class.
- `refresh.decision == "rollback"` or expired `refresh.next_check_due`.

On rollback:

1. do not modify any files,
2. do not update the active promoted pointer,
3. do not mark the snapshot as ready,
4. return an error envelope and leave the prior snapshot pointer unchanged,
5. record a `PROMOTION_STATE_INVALID` or `MANIFEST_INVALID` reason in the caller-facing diagnostics when a candidate snapshot was rejected.

Operator remediation:

1. fix or re-fetch missing/corrupt metadata files,
2. regenerate `manifest.json` and canonical `integrity.manifest_sha256`,
3. re-verify files and set `integrity.verification.status=verified` with `verified_at`,
4. rerun adapter health/execute after integrity gates pass.

## Exact request contract

```json
{
  "request_id": "urn:uuid:...",
  "dataset_symbol": "fw",
  "version_id": "flywire783",
  "query_kind": "entity_lookup",
  "scope_tags": {
    "sex": "female",
    "stage": "adult",
    "anatomy": "whole_brain_fafb_aligned",
    "evidence_family": "metadata_annotation",
    "access_layer": "metadata_only"
  },
  "query_payload": {
    "input": "MBON",
    "match_mode": "contains",
    "limit": 25
  },
  "allow_remote_fallback": false,
  "caller": "flybrain_harness_local_query_router"
}
```

Required request fields:

- `request_id`
- `dataset_symbol`
- `version_id`
- `query_kind`
- `scope_tags`
- `query_payload`
- `allow_remote_fallback`
- `caller`

The adapter must reject any request missing these fields with `REQUEST_INVALID`.

## Exact response contract

Successful responses must have this top-level shape:

```json
{
  "ok": true,
  "dataset_symbol": "fw",
  "version_id": "flywire783",
  "query_kind": "entity_lookup",
  "source": "local",
  "count_status": "exact",
  "result_rows": [
    {
      "canonical_id": "FBbt:00000000",
      "label": "MBON",
      "aliases": ["MBON", "mushroom body output neuron"],
      "score": 0.91,
      "entity_type": "class",
      "dataset_scope": {
        "dataset_symbol": "fw",
        "version_id": "flywire783",
        "sex": "female",
        "stage": "adult",
        "anatomy": "whole_brain_fafb_aligned"
      }
    }
  ],
  "warnings": [],
  "provenance": {
    "dataset_symbol": "fw",
    "version_id": "flywire783",
    "query_kind": "entity_lookup",
    "scope_tags": {
      "sex": "female",
      "stage": "adult",
      "anatomy": "whole_brain_fafb_aligned",
      "evidence_family": "metadata_annotation",
      "access_layer": "metadata_only"
    },
    "manifest_id": "fbh-fw-flywire783-2026-09-22T18:50:00Z",
    "manifest_sha256": "<sha256 of manifest>",
    "source_path": "snapshots/fw/flywire783/metadata",
    "access_method": "local_snapshot_read_only",
    "replay_fingerprint": "<sha256-of-canonical-query-input-and-scope>",
    "replay_fingerprint_version": "v1",
    "generated_at": "2026-09-22T18:50:00Z"
  }
}
```

Required response fields:

- `ok`
- `dataset_symbol`
- `version_id`
- `query_kind`
- `source`
- `count_status`
- `result_rows`
- `warnings`
- `provenance`

`count_status` must be one of:

- `exact` when the local metadata index has a complete answer set for the request,
- `unavailable` when the adapter was blocked by a local integrity or capability issue and cannot provide an exact answer,
- `partial` only when a bounded local query has a known subset and the metadata announces the subset as partial; otherwise prefer `exact` or `unavailable`.

`warnings` is an array of strings; it must never be null.

## Provenance envelope contract

This adapter must emit a deterministic provenance block for every successful result. The envelope is a canonical subset of the broader FlyBrain provenance schema and must include:

```json
{
  "dataset_symbol": "fw",
  "version_id": "flywire783",
  "query_kind": "entity_lookup",
  "scope_tags": {
    "sex": "female",
    "stage": "adult",
    "anatomy": "whole_brain_fafb_aligned",
    "access_layer": "metadata_only"
  },
  "manifest_id": "fbh-fw-flywire783-2026-09-22T18:50:00Z",
  "manifest_sha256": "<sha256>",
  "source_path": "snapshots/fw/flywire783/metadata",
  "access_method": "local_snapshot_read_only",
  "replay_fingerprint": "<sha256>",
  "replay_fingerprint_version": "v1",
  "generated_at": "2026-09-22T18:50:00Z"
}
```

Recommended normalization:

- compute `replay_fingerprint` using a canonical JSON object of `(tool_name, request, dataset_scope)`,
- keep it stable under key ordering,
- allow the harness to replay the exact same local read-only query for audit and regression.

The harness should use the same canonical replay logic as `mcp/replay_fingerprint.py` so the adapter remains compatible with the repository's FlyBrain provenance tooling.

## Local metadata map to pinned `fw` snapshot

The adapter must resolve the dataset as follows:

```text
LOCI_FLYBRAIN_STORAGE_ROOT
└── snapshots
    └── fw
        └── flywire783
            └── metadata
                ├── manifest.json
                ├── entities.json
                ├── classes.json
                ├── annotations.json
                └── provenance.json
```

The adapter must not access or imply any other `fw` subtree such as a live mirror, remote cache, or temporary staging directory. It must treat any additional dataset version or path under a different `fw` folder as out of scope unless explicitly promoted by the harness with a new manifest and version bump.

The expected `fw` snapshot contract is therefore:

- `snapshot_root = <root>/snapshots/fw/flywire783/metadata`
- `manifest_id` and `manifest_sha256` must match the snapshot manifest exactly,
- `dataset_symbol` and `version_id` must equal `fw` and `flywire783`,
- `scope_tags` must declare the access layer as `metadata_only`,
- `source` must be `local`, never `remote_fallback`.

## Implementation skeleton

The implementation should follow this skeleton:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FW_DATASET_SYMBOL = "fw"
FW_VERSION_ID = "flywire783"
FW_ACCESS_LAYER = "metadata_only"

@dataclass(frozen=True)
class FwMetadataQueryRequest:
    request_id: str
    dataset_symbol: str
    version_id: str
    query_kind: str
    scope_tags: dict[str, Any]
    query_payload: dict[str, Any]
    allow_remote_fallback: bool = False
    caller: str = "flybrain_harness"

@dataclass(frozen=True)
class FwMetadataError:
    code: str
    message: str
    request_id: str
    dataset_symbol: str
    version_id: str
    details: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class FwMetadataResponse:
    ok: bool
    dataset_symbol: str
    version_id: str
    query_kind: str
    source: str
    count_status: str
    result_rows: list[dict[str, Any]]
    warnings: list[str]
    provenance: dict[str, Any]
    error: FwMetadataError | None = None

class FwMetadataLocalAdapter:
    supported_query_kinds = {
        "entity_lookup",
        "class_lookup",
        "annotation_lookup",
        "provenance_lookup",
        "scope_validation",
    }

    def __init__(self, storage_root: str | Path | None = None):
        self.storage_root = self._resolve_root(storage_root)
        self.snapshot_root = Path(self.storage_root) / "snapshots" / "fw" / "flywire783" / "metadata"

    def execute(self, request: FwMetadataQueryRequest) -> FwMetadataResponse:
        self._assert_root_safe()
        self._assert_read_only(request)
        self._assert_dataset_pin(request)
        self._assert_supported_kind(request.query_kind)
        self._assert_manifest_valid()
        return self._dispatch(request)

    def _resolve_root(self, storage_root: str | Path | None):
        # resolve via LOCI_FLYBRAIN_STORAGE_ROOT; reject invalid roots
        raise NotImplementedError

    def _assert_root_safe(self):
        # reject symlinks, UNC, drive roots, and any path escape outside root
        raise NotImplementedError

    def _assert_read_only(self, request):
        # ensure the request does not attempt writes or mutation
        raise NotImplementedError

    def _assert_dataset_pin(self, request):
        # enforce fw + flywire783 + metadata_only
        raise NotImplementedError

    def _assert_supported_kind(self, query_kind):
        # enforce supported metadata-only operations only
        raise NotImplementedError

    def _assert_manifest_valid(self):
        # load manifest.json, compute checksums, verify snapshot integrity
        raise NotImplementedError

    def _dispatch(self, request):
        # route to entity_lookup/class_lookup/... in local-only mode
        raise NotImplementedError
```

The full implementation may populate rows from local snapshot indexes (`entities.json`, `classes.json`, `annotations.json`, or any equivalent pinned metadata materialization). It must not materialize a local FlyWire adjacency graph or treat any remote response as local evidence.

## Execution gate checklist before first live execute run

The adapter should not be marked live until all checks below pass:

- [ ] `LOCI_FLYBRAIN_STORAGE_ROOT` is set and valid
- [ ] local `fw` snapshot is present at `snapshots/fw/flywire783/metadata`
- [ ] `manifest.json` validates and `manifest_sha256` matches
- [ ] all expected metadata indexes are readable
- [ ] `entity_lookup` returns a deterministic row shape
- [ ] `class_lookup` returns a deterministic row shape
- [ ] `scope_validation` rejects out-of-scope requests
- [ ] unsupported query kinds fail with `UNSUPPORTED_QUERY_KIND`
- [ ] no request can silently fall back to remote data
- [ ] prior good snapshot pointer remains intact on any failure

## Implementation notes for rollout

- Keep the adapter read-only and dataset-pinned; the only mutable state should be the manifest pointer in the harness control layer, not in the metadata adapter itself.
- Keep the payload canonical, deterministic, and replayable; do not embed machine-specific file paths into the returned rows.
- Treat any unverified or partially downloaded snapshot as non-promotable and reject the request under `MANIFEST_INVALID`/`INTEGRITY_MISMATCH`.
- Treat `fw` as metadata-only evidence unless a later phase explicitly adds a full local graph capability with a separate dataset contract.

## Decision summary

This adapter is intentionally narrow, safe, and execution-oriented. It exists to normalize and compare FlyWire metadata with local provenance in a way that is deterministic, scoped, and audit-friendly, while preventing the phase-1 harness from claiming unsupported local FlyWire graph behavior.

## Targeted test coverage references

- `mcp/tests/test_flybrain_fw_metadata_adapter.py`
  - `test_fw_adapter_execute_success_with_valid_manifest`
  - `test_fw_adapter_rejects_manifest_hash_mismatch`
  - `test_fw_adapter_rejects_escape_relative_path`
