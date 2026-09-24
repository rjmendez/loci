# FlyBrain Harness Manifest + Provenance Schema

## Purpose

Define one JSON contract for local FlyBrain harness artifacts so each snapshot is:

- reproducible (re-fetch/rebuild possible),
- integrity-checkable (file and manifest hashes),
- refresh-trackable (status and decision history),
- claim-scoped (sex/stage/anatomy/evidence boundaries recorded).

This schema is aligned with:

- `docs/FLYBRAIN_HARNESS_STORAGE_LAYOUT.md`
- `docs/FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md`
- `docs/FLYBRAIN_ARTIFACT_INTEGRITY_QUARANTINE_POLICY.md`
- `docs/FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md`
- `docs/FLYBRAIN_DATASET_PROVENANCE_MATRIX.md`
- `docs/FLYBRAIN_HB_FW_LOCAL_INDEX_QUERY_IMPORT_STRATEGY.md`
- `docs/FLYBRAIN_FETCH_SYNC_PIPELINES_PLAN.md`

## Root rule

All path fields are variable-driven. Do not write hardcoded drive-letter paths.

Required root variable:

- `LOCI_FLYBRAIN_STORAGE_ROOT`

Allowed path style examples:

- `$LOCI_FLYBRAIN_STORAGE_ROOT\graph\hb\neuprint_JRC_Hemibrain_1point2point1\`
- `$LOCI_FLYBRAIN_STORAGE_ROOT\snapshots\fw\flywire783\metadata\`

Not allowed:

- `F:\...`
- `C:\...`

## hb/fw execution-contract alignment audit (pre-run)

The phase-1 execution contract requires that local query responses can always report:

- dataset + version pin,
- local-vs-fallback source,
- manifest identity (`manifest_id`, `manifest_sha256`),
- scope boundaries (sex/stage/anatomy/evidence family),
- deterministic replay key.

Coverage in this schema:

| Execution-contract need | Manifest field(s) | Required in v1 | Notes |
|---|---|---|---|
| Dataset/version pin identity | `dataset.symbol`, `dataset.version_id` | Yes | Must match configured phase pin (`hb` or `fw`). |
| Source access-path provenance | `dataset.source.system`, `dataset.source.access_method`, `dataset.source.uri` | Yes | `access_method` must be explicit and stable. |
| Scope boundaries | `scope.sex`, `scope.stage`, `scope.anatomy`, `scope.evidence_family` | Yes | Prevents over-generalization in downstream claims. |
| Manifest identity for query envelopes | `manifest_id`, `integrity.manifest_sha256` | Yes | `manifest_sha256` computation rule is fixed below (no recursion ambiguity). |
| Promotion/readiness gate state | `integrity.verification.*`, `refresh.decision`, `refresh.checked_at` | Yes | Required to block fail-open use of unverified artifacts. |
| Reproducibility and replay linkage | `lineage.pipeline.run_id`, `lineage.pipeline.job_version` | Yes | Query-level `replay_fingerprint` remains response-side, not manifest-side. |
| Adapter capability boundary (`hb` graph vs `fw` metadata) | `artifact.kind`, `artifact.relative_root` | Yes | `kind` + path subtree enforce phase-1 lane boundaries. |

## Canonical schema (v1)

```json
{
  "schema_version": "fbh-manifest/v1",
  "manifest_id": "fbh-hb-neuprint_JRC_Hemibrain_1point2point1-2026-09-22T18:50:00Z",
  "generated_at": "2026-09-22T18:50:00Z",
  "storage_root_env": "LOCI_FLYBRAIN_STORAGE_ROOT",
  "artifact": {
    "kind": "dataset_snapshot",
    "relative_root": "graph\\hb\\neuprint_JRC_Hemibrain_1point2point1\\",
    "path_template": "$LOCI_FLYBRAIN_STORAGE_ROOT\\graph\\hb\\neuprint_JRC_Hemibrain_1point2point1\\"
  },
  "dataset": {
    "symbol": "hb",
    "label": "Neuprint web interface - hemibrain:v1.2.1",
    "version_id": "neuprint_JRC_Hemibrain_1point2point1",
    "source": {
      "system": "virtual-fly-brain",
      "access_method": "list_connectome_datasets",
      "uri": "vfb://dataset/hb/neuprint_JRC_Hemibrain_1point2point1",
      "retrieved_at": "2026-09-22T18:10:00Z",
      "license": {
        "spdx_id": "CC-BY-4.0",
        "url": "https://creativecommons.org/licenses/by/4.0/"
      }
    }
  },
  "scope": {
    "sex": "female",
    "stage": "adult",
    "anatomy": "hemibrain_region",
    "evidence_family": "connectome_structural",
    "claim_tier": "T1_dataset_version_specific"
  },
  "integrity": {
    "manifest_sha256": "3f2c0db9818f44f3d9b67ff3d6cf115d5c29ce7e1e08ecfc1f0974737cc26f96",
    "files": [
      {
        "relative_path": "nodes.parquet",
        "size_bytes": 87423123,
        "sha256": "66942828f1f2f8146f6aa0ac8cfed724611f8b34495bfedfe9f74efd14577231",
        "last_modified": "2026-09-22T18:44:39Z"
      }
    ],
    "verification": {
      "status": "verified",
      "verified_at": "2026-09-22T18:49:00Z",
      "failure_reason": null
    }
  },
  "refresh": {
    "cadence": "quarterly_integrity_recheck",
    "decision": "no_change",
    "checked_at": "2026-09-22T18:50:00Z",
    "supersedes_manifest_id": null,
    "next_check_due": "2026-12-22T00:00:00Z"
  },
  "lineage": {
    "derived_from": [
      {
        "type": "remote_dataset_release",
        "id": "neuprint_JRC_Hemibrain_1point2point1"
      }
    ],
    "pipeline": {
      "job_name": "fbh_snapshot_ingest",
      "job_version": "2026.09.22.1",
      "run_id": "run-9f1a826e",
      "run_mode": "execute"
    }
  },
  "notes": [
    "Phase-1 local graph-first rollout scope."
  ]
}
```

## Required fields

Top-level required fields:

- `schema_version`
- `manifest_id`
- `generated_at`
- `storage_root_env`
- `artifact`
- `dataset`
- `scope`
- `integrity`
- `refresh`
- `lineage`

Nested required fields:

- `artifact.kind`
- `artifact.relative_root`
- `artifact.path_template`
- `dataset.symbol`
- `dataset.version_id`
- `dataset.source.system`
- `dataset.source.access_method`
- `dataset.source.uri`
- `dataset.source.retrieved_at`
- `dataset.source.license.spdx_id`
- `scope.sex`
- `scope.stage`
- `scope.anatomy`
- `scope.evidence_family`
- `scope.claim_tier`
- `integrity.manifest_sha256`
- `integrity.files` (non-empty)
- `integrity.files[*].relative_path`
- `integrity.files[*].sha256`
- `integrity.verification.status`
- `refresh.decision`
- `refresh.checked_at`
- `lineage.derived_from` (non-empty)
- `lineage.pipeline.job_name`
- `lineage.pipeline.job_version`
- `lineage.pipeline.run_id`

Notes:

- Paths must remain under the canonical root subtrees: `graph\`, `snapshots\`, `cache\`, `backups\`, or `logs\`.
- `datasets\...` is not a canonical writable layout in phase 1.
- `dataset.symbol` must be one of the known VFB symbols and phase-1 execution currently allows only `hb` and `fw`. Verified local-snapshot manifests (read-only adapter use, not phase-1 query execution) also exist for `BANC` (`relative_root` `snapshots/BANC/banc_888`, see `FLYBRAIN_BANC_ADAPTER_CONTRACT.md`) and `l1em` (`snapshots/l1em/catmaid_l1em`, see `FLYBRAIN_L1EM_ADAPTER_CONTRACT.md`). Both adapters require the `manifest/manifest.sha256` sidecar to be present and to match.

## Resolved ambiguities and edge-case rules (required before first execute run)

### 1. Manifest self-hash ambiguity

`integrity.manifest_sha256` cannot be computed over a manifest payload that already contains the final hash value.

Normative rule:

1. Build canonical JSON with UTF-8 + stable key ordering.
2. Set `integrity.manifest_sha256` to an empty string (`""`) for hash computation.
3. Compute sha256 over that canonical payload.
4. Write the resulting digest into `integrity.manifest_sha256`.
5. Also write sidecar `manifest.sha256` with the same value.

Readers must recompute using the same rule; otherwise hash verification is undefined.

### 2. Relative path safety

For `artifact.relative_root` and each `integrity.files[*].relative_path`:

- must not be absolute,
- must not contain drive letters,
- must not contain `..` segments,
- must not start with `\` or `/`,
- must not resolve outside `LOCI_FLYBRAIN_STORAGE_ROOT` after normalization.

Implementation note (current guard behavior):

- validate with normalized relative paths only (`_is_safe_relative_path`)
- reject duplicates after case-folding (Windows-safe)
- reject partial artifact suffixes: `.partial`, `.tmp`, `.inprogress`
- require each listed file to exist under `artifact.relative_root`
- require lowercase 64-hex `sha256` and matching digest
- enforce `size_bytes` when present as a non-negative integer

### 3. Duplicate-path collision handling

Writers must reject manifests containing duplicate file paths after case-folded normalization (Windows-safe) to prevent alias collisions (for example `Nodes.parquet` vs `nodes.parquet`).

### 4. Partial-download artifacts

Manifests must not include `.partial`, `.tmp`, `.inprogress`, or zero-byte artifacts when a non-zero `size_bytes` is expected. Such artifacts are quarantine-only and non-promotable.

### 5. Verification status gating

`integrity.verification.status` is required and must be one of:

- `pending`
- `verified`
- `quarantined`

Promotion and query use are allowed only for `verified` manifests.
When `status == "verified"`, `verified_at` is required. Missing `verified_at`
is a hard validation error.

### 6. hb/fw lane boundary

- `hb` manifests must resolve under `graph\hb\<version>\...` for graph-capable artifacts.
- `fw` manifests must resolve under `snapshots\fw\<version>\metadata\...` for metadata-only artifacts in phase 1.

A `fw` manifest must never be treated as proof of local full-graph adjacency capability in phase 1.

### 7. Refresh-chain consistency

- If `refresh.decision == "no_change"`, `refresh.supersedes_manifest_id` must be `null`.
- If `refresh.decision != "no_change"`, `refresh.supersedes_manifest_id` must point to a prior manifest for the same `(dataset.symbol, dataset.version_id)` lineage.

### 8. Timestamp and timezone consistency

`generated_at`, `retrieved_at`, `checked_at`, `verified_at`, and `next_check_due` must be RFC3339 UTC (`...Z` or `+00:00`) and parseable without locale-specific assumptions.

## Validation guidance

### Writer checks (before commit/store)

1. `path_template` must begin with `$LOCI_FLYBRAIN_STORAGE_ROOT\`.
2. `artifact.relative_root` and every `integrity.files[*].relative_path` must be safe relative paths (no drive/root/traversal).
3. `dataset.version_id` must be version-bearing and match pinned scope (for example `flywire783`, `neuprint_JRC_Hemibrain_1point2point1`).
4. `refresh.decision` must be one of: `no_change`, `patch_refresh`, `major_bump`, `rollback`.
5. `sha256` values must be lowercase 64-hex.
6. `scope.claim_tier` must align with evidence breadth (`T0`..`T3` from provenance guidance).
7. If `refresh.decision != "no_change"`, set `supersedes_manifest_id` to the prior manifest.
8. `integrity.files` must be non-empty, unique after case-folding, and free of partial-temp suffixes.
9. `integrity.verification.status` must be set, and `verified_at` is required when status is `verified`.
10. Compute `integrity.manifest_sha256` using the self-hash rule above and mirror it to `manifest.sha256`.

### Reader checks (before using claim/evidence)

1. Reject if required fields are missing.
2. Reject if path fields are absolute, traversal-based, or hardcoded drive paths.
3. Verify every listed file hash before using artifact data.
4. Enforce `integrity.verification.status == "verified"` before promotion/query use.
5. Enforce scope boundary in downstream claims:
   - no cross-sex/stage/anatomy generalization unless evidence supports it.
6. Use `dataset.symbol + dataset.version_id + dataset.source.access_method` as the minimum reproducibility key.
7. Treat `refresh.next_check_due` in the past as stale until revalidated.
8. Apply integrity quarantine policy before artifact use:
   - enforce expected-size equality when `size_bytes` is present,
   - enforce required `sha256` verification,
   - quarantine and block promotion on any mismatch.

## Fail-closed runtime outcomes

A manifest that fails any required integrity or verification gate must not be
used in a success-shaped local query response.

- hash mismatch or file checksum/size mismatch: fail closed
- `integrity.verification.status != "verified"`: fail closed
- unsafe/escaping path entry in `integrity.files`: fail closed
- stale `refresh.next_check_due`: fail closed until refresh/revalidation

Operator remediation:

1. rebuild or re-fetch the snapshot artifacts,
2. regenerate `manifest.json` + `manifest.sha256` with canonical self-hash,
3. re-run integrity verification to set `verification.status=verified` and
   `verified_at`,
4. retry promotion/query only after all gates pass.

## Targeted test coverage references

These guardrails are covered by focused tests:

- `mcp/tests/test_flybrain_hb_adapter.py`
  - verified-status gating and rollback-required behavior
  - manifest self-hash validation path
- `mcp/tests/test_flybrain_fw_metadata_adapter.py`
  - manifest self-hash mismatch fail-closed behavior
  - integrity file path escape rejection (`..\\outside.json`)

## Minimal JSON Schema fragment for automation

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "FlyBrainHarnessManifestV1",
  "type": "object",
  "required": [
    "schema_version",
    "manifest_id",
    "generated_at",
    "storage_root_env",
    "artifact",
    "dataset",
    "scope",
    "integrity",
    "refresh",
    "lineage"
  ],
  "properties": {
    "schema_version": {
      "const": "fbh-manifest/v1"
    },
    "storage_root_env": {
      "const": "LOCI_FLYBRAIN_STORAGE_ROOT"
    },
    "artifact": {
      "type": "object",
      "required": [
        "kind",
        "relative_root",
        "path_template"
      ],
      "properties": {
        "path_template": {
          "type": "string",
          "pattern": "^\$LOCI_FLYBRAIN_STORAGE_ROOT\\"
        }
      }
    },
    "dataset": {
      "type": "object",
      "required": ["symbol", "version_id", "source"],
      "properties": {
        "symbol": {
          "type": "string",
          "enum": ["fw", "mc", "BANC", "hb", "mv", "ol", "fafb", "l1em"]
        },
        "source": {
          "type": "object",
          "required": ["system", "access_method", "uri", "retrieved_at", "license"],
          "properties": {
            "license": {
              "type": "object",
              "required": ["spdx_id"]
            }
          }
        }
      }
    },
    "scope": {
      "type": "object",
      "required": ["sex", "stage", "anatomy", "evidence_family", "claim_tier"]
    },
    "integrity": {
      "type": "object",
      "required": ["manifest_sha256", "files", "verification"],
      "properties": {
        "manifest_sha256": {
          "type": "string",
          "pattern": "^[a-f0-9]{64}$"
        },
        "files": {
          "type": "array",
          "minItems": 1,
          "items": {
            "type": "object",
            "required": [
              "relative_path",
              "sha256"
            ],
            "properties": {
              "sha256": {
                "type": "string",
                "pattern": "^[a-f0-9]{64}$"
              }
            }
          }
        },
        "verification": {
          "type": "object",
          "required": ["status"],
          "properties": {
            "status": {
              "type": "string",
              "enum": ["pending", "verified", "quarantined"]
            }
          }
        }
      }
    },
    "lineage": {
      "type": "object",
      "required": ["derived_from", "pipeline"],
      "properties": {
        "derived_from": {
          "type": "array",
          "minItems": 1
        },
        "pipeline": {
          "type": "object",
          "required": ["job_name", "job_version", "run_id"]
        }
      }
    }
  }
}
```

## Writer/reader interoperability notes

- Writers should preserve unknown fields for forward compatibility.
- Readers should hard-fail on invalid required fields, but soft-ignore unknown extensions.
- For upgrades, bump `schema_version` and keep prior manifest immutable; link versions through `refresh.supersedes_manifest_id`.
- Query response envelopes should carry `manifest_id`, `manifest_sha256`, and `replay_fingerprint` so runtime answers remain traceable to one manifest state.
