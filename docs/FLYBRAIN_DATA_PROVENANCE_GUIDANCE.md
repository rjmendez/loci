# FlyBrain data provenance guidance

This page explains the minimum provenance needed for FlyBrain facts stored in Loci.

The main rule is simple: if a claim came from a FlyBrain query, the memory entry should also record where that claim came from, how it was asked, and what scope it belongs to.

## Why this matters

A FlyBrain result can change meaning when any of these differ:

- dataset symbol
- dataset version or release label
- sex
- life stage
- anatomy scope
- query mode
- evidence family (curated vs predicted vs connectivity)

Without that metadata, a later answer can accidentally treat a local dataset result as if it were a general biological fact.

## Minimum metadata to keep

When storing a FlyBrain-derived claim, record:

1. Tool used
   - example: `query_connectivity`, `search_terms`, `run_query`
2. Input and resolved entity
   - original string and resolved class/ID
3. Dataset scope
   - included and excluded datasets
   - version-bearing identifiers
4. Query semantics
   - `group_by_class`, `weight`, paging, filters, warnings
5. Result contract
   - count, `count_status`, warnings, row count
6. Scope tags
   - sex, stage, anatomy coverage, evidence family

## Example

```json
{
  "flybrain_provenance": {
    "tool_name": "query_connectivity",
    "request": {
      "upstream_type_input": "FBbt_00003686",
      "weight": 20,
      "group_by_class": true,
      "exclude_dbs": ["hb", "fafb"]
    },
    "dataset_scope": {
      "included_symbols": ["fw", "mc", "ol", "mv"],
      "excluded_symbols": ["hb", "fafb"],
      "version_ids_seen": ["flywire783", "male_cns_v1_0"]
    },
    "scope": {
      "sex": "unspecified",
      "stage": "adult-biased",
      "anatomy_scope": "Kenyon cell class and subclasses"
    },
    "result_contract": {
      "count": 1858,
      "count_status": "exact",
      "returned_rows": 5,
      "warnings": []
    },
    "evidence_family": "connectivity_rollup"
  }
}
```

This is enough to re-check a claim later without guessing.

## Claim tiers

Use the evidence to constrain the language:

- T0: dataset-run specific
  - "In this dataset/query path..."
- T1: dataset-version specific
  - "In FlyWire v783..."
- T2: matched cross-dataset scope
  - "Observed in both adult female FlyWire and BANC under compatible settings..."
- T3: broader generalization
  - only with direct evidence across the claimed sex/stage/anatomy scopes

If the evidence does not support the wider claim, keep the wording narrow.

## Required boundaries to keep explicit

Do not generalize across:

- female vs male data without direct support
- adult vs larval data without direct support
- one anatomy scope to another
- curated findings to predicted confidence findings
- grouped vs ungrouped connectivity counts

## A simple rule for writing final answers

Use this pattern:

- weakly scoped: "In this dataset/version/query path..."
- moderately scoped: "In the adult female FlyWire dataset..."
- stronger cross-dataset: "Observed across adult female datasets with matching scope..."
- general: only when the evidence spans the full scope being claimed

## Implementation note

The repo now tracks deterministic fingerprints and provenance fields for FlyBrain memory entries, including replay fingerprints and normalized dataset scope. That lets later checks confirm whether a claim can be replayed and whether the dataset scope still matches the evidence.

For local harness artifact manifests (dataset snapshots/metadata bundles), use the canonical writer/reader contract in `docs/FLYBRAIN_HARNESS_MANIFEST_PROVENANCE_SCHEMA.md`.

## Final guidance

The safe default is not to say "this is generally true." The safe default is to say what the evidence actually supports, and to carry the dataset and query path along with it.

---

Status: ready for FlyBrain provenance capture and replay checks.
