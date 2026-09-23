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
    "evidence_family": "connectivity_structural",
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
    ]
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
      "run_id": "run-9f1a826e"
    }
  },
  "notes": [
    "Phase-1 local graph-first rollout scope."
  ]
}
```

## Required fields

- `schema_version`
- `manifest_id`
- `generated_at`
- `storage_root_env`
- `artifact.kind`
- `artifact.relative_root`
- `artifact.path_template`
- `dataset.symbol`
- `dataset.version_id`
- `dataset.source.system`
- `dataset.source.access_method`
- `dataset.source.retrieved_at`
- `dataset.source.license.spdx_id`
- `scope.sex`
- `scope.stage`
- `scope.anatomy`
- `scope.evidence_family`
- `integrity.files[*].relative_path`
- `integrity.files[*].sha256`
- `refresh.decision`
- `refresh.checked_at`
- `lineage.derived_from`

Notes:
- Paths must remain under the canonical root subtrees: `graph\`, `snapshots\`, `cache\`, `backups\`, or `logs\`.
- `datasets\...` is not the canonical writable layout for the phase-1 harness.

## Validation guidance

### Writer checks (before commit/store)

1. `path_template` must begin with `$LOCI_FLYBRAIN_STORAGE_ROOT\`.
2. `artifact.relative_root` and every `integrity.files[*].relative_path` must be relative (no drive, no root).
3. `dataset.version_id` must be version-bearing and match pinned scope (for example `flywire783`, `neuprint_JRC_Hemibrain_1point2point1`).
4. `refresh.decision` must be one of: `no_change`, `patch_refresh`, `major_bump`, `rollback`.
5. `sha256` values must be lowercase 64-hex.
6. `scope.claim_tier` must align with evidence breadth (`T0`..`T3` from provenance guidance).
7. If `refresh.decision != "no_change"`, set `supersedes_manifest_id` to the prior manifest.

### Reader checks (before using claim/evidence)

1. Reject if required fields are missing.
2. Reject if path fields are absolute or hardcoded drive paths.
3. Verify every listed file hash before using artifact data.
4. Enforce scope boundary in downstream claims:
   - no cross-sex/stage/anatomy generalization unless evidence supports it.
5. Use `dataset.symbol + version_id + access_method` as the minimum reproducibility key.
6. Treat `refresh.next_check_due` in the past as stale until revalidated.
7. Apply integrity quarantine policy before artifact use:
   - enforce expected-size equality when `size_bytes` is present,
   - enforce required `sha256` verification,
   - quarantine and block promotion on any mismatch.

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
        "relative_root",
        "path_template"
      ],
      "properties": {
        "path_template": {
          "type": "string",
          "pattern": "^\\$LOCI_FLYBRAIN_STORAGE_ROOT\\\\"
        }
      }
    },
    "integrity": {
      "type": "object",
      "properties": {
        "files": {
          "type": "array",
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
