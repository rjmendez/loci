# FlyBrain hb local adapter contract

## Purpose

This document defines the concrete local read-only contract for the pinned
hemibrain adapter used by the FlyBrain harness.

It is the execution target for the phase-1 local graph path:

- dataset symbol: `hb`
- version pin: `neuprint_JRC_Hemibrain_1point2point1`
- graph root: `$LOCI_FLYBRAIN_STORAGE_ROOT\graph\hb\neuprint_JRC_Hemibrain_1point2point1\`
- manifest root: `$LOCI_FLYBRAIN_STORAGE_ROOT\snapshots\hb\neuprint_JRC_Hemibrain_1point2point1\manifest\`

## Supported local read-only operations

The adapter exposes only these operations:

1. `capabilities()`
   - returns the pinned dataset/version, supported query kinds, and scope tags.
2. `resolve_snapshot()`
   - resolves the manifest and promotion pointer for the pinned hb dataset.
3. `validate_compatibility(request)`
   - performs all safety gates before execution.
4. `health()`
   - returns a health summary for the pinned snapshot and manifest.
5. `execute(request)`
   - executes a read-only local query when every gate passes.

No write operation is supported by this adapter.

## Hard scope contract

The hb adapter only accepts the phase-1 local graph scope:

- `sex=female`
- `stage=adult`
- `anatomy=hemibrain_region`
- `evidence_family=connectome_structural`
- `access_layer=local_graph_snapshot`

Any request outside this scope is rejected with a fail-closed error.

## Supported query kinds

The adapter accepts these query kinds only:

- `connectivity_lookup`
- `neighborhood_traversal`
- `replay_smoke`
- `scope_smoke`

The adapter must reject any other query kind with
`UNSUPPORTED_QUERY_KIND`.

## Safety gates

Before any execution, the adapter must validate all of the following:

1. **Root configuration**
   - `LOCI_FLYBRAIN_STORAGE_ROOT` or a supported alias must resolve to an
     absolute local path.
   - The configured root must not be a drive root, UNC path, or reparse-point
     escape.
2. **Dataset pin**
   - `dataset_symbol == "hb"`
   - `version_id == "neuprint_JRC_Hemibrain_1point2point1"`
3. **Manifest presence**
   - `manifest.json` must exist under the pinned manifest root.
4. **Manifest validity**
   - `schema_version == "fbh-manifest/v1"`
   - `dataset.symbol == "hb"`
   - `dataset.version_id == "neuprint_JRC_Hemibrain_1point2point1"`
   - `integrity.manifest_sha256` must be present
   - `integrity.verification.status == "verified"`
   - if `integrity.verification.status == "verified"`, `integrity.verification.verified_at` is required
   - `refresh.decision` must be one of: `no_change`, `patch_refresh`, `major_bump`, `rollback`
5. **Manifest integrity verification**
   - recompute the manifest digest using canonical JSON with
     `integrity.manifest_sha256=""` during computation
   - require exact digest match with stored `integrity.manifest_sha256`
   - require `integrity.files` to be non-empty and each entry to:
     - use a safe relative path (no absolute path, no drive prefix, no `..`)
     - be unique after case-fold normalization
     - avoid partial suffixes (`.partial`, `.tmp`, `.inprogress`)
     - include lowercase 64-hex `sha256`
     - resolve under the pinned artifact root and exist as a file
     - match `size_bytes` when present as a non-negative integer
     - match the listed sha256 digest
6. **Promotion state**
   - the active promotion pointer must exist
   - the active pointer JSON must include `{"active": true}`
   - the active pointer must remain under the pinned hb graph root
7. **Scope alignment**
   - request scope tags must match the hard scope contract above
8. **Query guard**
   - request must be read-only
   - request must not allow remote fallback through this adapter
   - request must have a supported query kind
   - request payload must include a non-empty `query` field
9. **Limit guard**
   - `max_rows > 0`
   - `max_depth >= 0`
   - `max_weight >= 0` when present

## Rollback conditions

The adapter must mark rollback as required, and refuse execution, when any of
these conditions hold:

- `integrity.verification.status != "verified"`
- the active promotion pointer is missing
- manifest scope does not match the phase-1 hb scope
- `refresh.decision == "rollback"`
- `refresh.next_check_due` is in the past
- manifest or promotion state is missing or malformed
- any compatibility check fails on a safety-critical gate

Rollback-required states are never success-shaped. They must be surfaced as
explicit errors with a rollback flag.

## Error codes

The adapter uses explicit fail-closed error codes:

- `ROOT_NOT_CONFIGURED`
- `ROOT_NOT_ABSOLUTE`
- `ROOT_OUT_OF_BOUNDS`
- `ROOT_REPARSE_POINT`
- `DATASET_PIN_MISMATCH`
- `UNSUPPORTED_QUERY_KIND`
- `OUT_OF_SCOPE`
- `SCOPE_VIOLATION`
- `READ_ONLY_REQUIRED`
- `MANIFEST_MISSING`
- `MANIFEST_INVALID`
- `INTEGRITY_MISMATCH`
- `PROMOTION_STATE_INVALID`
- `ROLLBACK_REQUIRED`
- `QUERY_LIMIT_EXCEEDED`
- `BACKEND_UNAVAILABLE`
- `UNIMPLEMENTED_BACKEND`

Any error in the rollback family below must also set `rollback_required=true`:

- `MANIFEST_INVALID`
- `INTEGRITY_MISMATCH`
- `PROMOTION_STATE_INVALID`
- `ROLLBACK_REQUIRED`

## Operator remediation for fail-closed states

When the adapter fails closed, do not bypass checks. Fix data/control state,
then retry:

1. restore or rebuild corrupted/missing integrity files,
2. regenerate `manifest.json` and canonical self-hash value,
3. set verification to `status="verified"` with `verified_at`,
4. restore `promotion\active_pointer.json` with `active=true`,
5. rerun local adapter health/compatibility checks before execute.

## Query result contract

Successful hb query responses must use this exact envelope shape:

```json
{
  "status": "ok",
  "count_status": "exact",
  "dataset_symbol": "hb",
  "version_id": "neuprint_JRC_Hemibrain_1point2point1",
  "query_kind": "connectivity_lookup",
  "source": "local",
  "row_count": 1,
  "columns": ["source_id", "target_id", "weight"],
  "rows": [
    {
      "source_id": "...",
      "target_id": "...",
      "weight": 3
    }
  ],
  "warnings": [],
  "provenance": {
    "schema_version": "hb-provenance/v1",
    "dataset_symbol": "hb",
    "version_id": "neuprint_JRC_Hemibrain_1point2point1",
    "source": "local",
    "access_method": "local_graph_snapshot",
    "access_path": "$LOCI_FLYBRAIN_STORAGE_ROOT\\graph\\hb\\neuprint_JRC_Hemibrain_1point2point1",
    "query_kind": "connectivity_lookup",
    "scope_tags": {
      "sex": "female",
      "stage": "adult",
      "anatomy": "hemibrain_region",
      "evidence_family": "connectome_structural",
      "access_layer": "local_graph_snapshot"
    },
    "manifest_id": "fbh-hb-...",
    "manifest_sha256": "...",
    "manifest_path": "...\\snapshots\\hb\\neuprint_JRC_Hemibrain_1point2point1\\manifest\\manifest.json",
    "graph_root": "...\\graph\\hb\\neuprint_JRC_Hemibrain_1point2point1",
    "active_pointer_path": "...\\graph\\hb\\neuprint_JRC_Hemibrain_1point2point1\\promotion\\active_pointer.json",
    "replay_fingerprint": "64-hex-digest",
    "generated_at": "2026-09-23T00:00:00+00:00",
    "result_contract_version": "hb-query-result/v1"
  },
  "compatibility_checks": [
    {
      "name": "dataset_pin",
      "passed": true,
      "code": null,
      "detail": null,
      "rollback_required": false
    }
  ],
  "rollback_required": false,
  "error_code": null,
  "error_message": null,
  "query_result_version": "hb-query-result/v1"
}
```

Contract rules:

- `row_count` must match `rows.length`.
- `columns` must list the normalized column names in the result.
- `source` must be `local` for this adapter.
- `count_status` must be `exact` on success.
- `error_code` and `error_message` must be `null` on success.

## Provenance envelope contract

The provenance envelope is mandatory on every success-shaped query result.
It must contain:

- `schema_version = "hb-provenance/v1"`
- `dataset_symbol`
- `version_id`
- `source = "local"`
- `access_method = "local_graph_snapshot"`
- `access_path`
- `query_kind`
- `scope_tags`
- `manifest_id`
- `manifest_sha256`
- `manifest_path`
- `graph_root`
- `active_pointer_path`
- `replay_fingerprint`
- `generated_at`
- `result_contract_version = "hb-query-result/v1"`

The replay fingerprint must be deterministic for the request and dataset scope.

## Implementation skeleton

The repository implementation lives in `mcp/flybrain_hb_adapter.py` and exposes:

- `HbScopeTags`
- `HbQueryRequest`
- `HbCompatibilityCheck`
- `HbSnapshotHandle`
- `HbProvenanceEnvelope`
- `HbQueryResult`
- `HbCapabilityManifest`
- `HbHealthStatus`
- `HbValidationReport`
- `HbAdapterError`
- `LocalHbAdapter`
- `build_hb_adapter()`

The adapter is intentionally backend-injected. The skeleton validates the
contract and fail-closed gates first, then defers actual graph retrieval to the
caller-provided local backend.

## Targeted test coverage references

- `mcp/tests/test_flybrain_hb_adapter.py`
  - `test_manifest_not_verified_requires_rollback`
  - `test_validation_rejects_scope_and_fallback`
  - `test_execute_returns_local_provenance_and_rows`

## Mapping to the pinned hb dataset

The adapter maps the pinned local hemibrain dataset to these fixed paths:

- graph root: `$LOCI_FLYBRAIN_STORAGE_ROOT\graph\hb\neuprint_JRC_Hemibrain_1point2point1\`
- manifest root: `$LOCI_FLYBRAIN_STORAGE_ROOT\snapshots\hb\neuprint_JRC_Hemibrain_1point2point1\manifest\`
- active promotion pointer: `$LOCI_FLYBRAIN_STORAGE_ROOT\graph\hb\neuprint_JRC_Hemibrain_1point2point1\promotion\active_pointer.json`

Those paths are the concrete contract boundary for the hb adapter.

---

Status: execution-ready local hb adapter contract and implementation skeleton.
