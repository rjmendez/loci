# FlyBrain Data Provenance Guidance for Reproducible Loci Claims

**Goal**: define what must be captured when a FlyBrain-derived claim is stored in Loci so the claim can be re-run, re-checked, and safely compared across dataset versions and access paths.

**Audience**: next PR implementing or tightening FlyBrain claim storage, memory replay, and reasoning checks.

## 1) Evidence-backed constraints from current tool surfaces

These are not assumptions; they are directly observable in current VFB/FlyBrain tool behavior and existing repo docs.

| Surface | Observed behavior | Reproducibility implication |
|---|---|---|
| `list_connectome_datasets` | Returns dataset identity as `symbol`, `short_form`, and version-bearing labels (examples observed: `fw` -> `flywire783`, `mc` -> `male_cns_v1_0`, `hb` -> `neuprint_JRC_Hemibrain_1point2point1`, `l1em` -> `catmaid_l1em`). | Claims must store both dataset symbol(s) and resolved version-bearing identifier(s); symbol alone is not enough for replay. |
| `search_terms` | Returns ontology class hits with facets including stage tags such as `Adult` and type tags (`Neuron`, `Class`, etc.). | Stage is discoverable at resolution time; stage tags used for claim scope must be stored with the resolved class identity. |
| `query_connectivity` | Result semantics depend on invocation mode (`group_by_class` rollups are non-additive by design). Zero-result and warning outcomes vary by class typing and access path. `exclude_dbs` materially changes scope. | Claims must persist query mode flags and the include/exclude dataset set; otherwise later re-runs can produce non-comparable numbers. |
| `run_query` | Provides `count` + `count_status`; may emit warnings like "offset/limit were ignored" for some query types. | Persist `count_status`, warnings, and paging semantics; never treat `count` as comparable without its status and query-type behavior. |
| `get_known_neurotransmitters` vs `get_predicted_neurotransmitters` | Known NT is curated (no confidence). Predicted NT includes `dataset` and confidence statistics (`mean_confidence`, percentages, counts). | Provenance must encode evidence family (`curated` vs `predicted`) and confidence availability; mixing them into one confidence scale is invalid. |
| `docs/FLYBRAIN_CROSS_DATASET_LESSONS.md` | Existing guidance already flags dataset heterogeneity and sex/stage caveats. | Store claim scope explicitly (dataset/version/sex/stage); do not elevate to "general" without cross-scope evidence. |

## 2) Minimum reproducibility envelope for every FlyBrain claim

For any claim derived from FlyBrain tools, store metadata that can fully reconstruct the access path and scope.

```json
{
  "flybrain_provenance": {
    "tool_name": "query_connectivity",
    "tool_variant": "virtual-fly-brain-query_connectivity",
    "executed_at": "2026-09-22T20:56:00Z",
    "request": {
      "upstream_type_input": "FBbt_00003686",
      "downstream_type_input": null,
      "weight": 20,
      "group_by_class": true,
      "limit": 5,
      "offset": 0,
      "exclude_dbs": ["hb", "fafb"]
    },
    "resolution": {
      "upstream": {"query": "FBbt_00003686", "id": "FBbt_00003686", "label": "Kenyon cell"},
      "downstream": null
    },
    "dataset_scope": {
      "included_symbols": ["BANC", "fw", "ol", "mv", "mc", "l1em"],
      "excluded_symbols": ["hb", "fafb"],
      "version_ids_seen": ["flywire783", "male_cns_v1_0", "neuprint_JRC_Manc_1_2_1"]
    },
    "result_contract": {
      "count": 1858,
      "count_status": "exact",
      "returned_rows": 5,
      "ranked_by": "total_weight",
      "warnings": []
    },
    "scope": {
      "sex": "unspecified",
      "stage": "adult-biased",
      "anatomy_scope": "Kenyon cell class and subclasses"
    },
    "evidence_family": "connectivity_rollup"
  }
}
```

## 3) Required claim tiering (dataset/version/sex/stage)

Use this tiering when writing findings or synthesizing answers. Promote only when explicit evidence supports the wider scope.

| Tier | Allowed wording | Required support | Disallowed leap |
|---|---|---|---|
| **T0: Dataset-run specific** | "In this run/query result..." | One executed call + full reproducibility envelope | Calling it a property of the neuron class globally |
| **T1: Dataset-version specific** | "In FlyWire v783..." / "in male_cns_v1_0..." | One dataset with explicit version identifier + scoped entities | Extrapolating to other versions of same dataset family |
| **T2: Cross-dataset, same stage/sex scope** | "Observed in both flywire783 and male_cns_v1_0 under comparable query settings..." | >=2 datasets, harmonized query parameters, explicit stage/sex compatibility note | Generalizing across sex or developmental stage without direct evidence |
| **T3: Cross-sex or cross-stage generalization** | "Supported across male/female and/or adult/larval datasets..." | Direct support from each claimed scope (e.g., adult female + adult male, or adult + larval) with caveat on structural differences | Any universal statement ("all Drosophila", "generally true") without multi-scope evidence |

## 4) Access-path fields that must never be dropped

These fields are mandatory because they change interpretability or replayability:

1. **Entity resolution path**: original input string, resolved ID, resolved label, and whether match came from exact ID, exact label, synonym, or broad match.
2. **Dataset selection**: explicit include/exclude symbols and resolved version-bearing IDs (from dataset list + result payloads).
3. **Query semantics flags**: `group_by_class`, aggregation mode, confidence filters, paging inputs, and any warning that semantics/paging were ignored.
4. **Result quality flags**: `count_status`, warnings, empty-result notes, and any tool-level caveat text.
5. **Scope tags for interpretation**: sex, stage, anatomical scope, and evidence family (`curated` vs `predicted` vs `connectivity`).

If any item above is unavailable at capture time, the finding must be tagged as partial provenance and capped at **T0**.

## 5) PR-ready implementation guidance for Loci

For the next implementation PR, wire the following into FlyBrain claim storage paths:

1. Add a `metadata.flybrain_provenance` block (shape above) on `investigation_store` writes for FlyBrain-derived findings.
2. Add normalization for dataset identity:
   - store `symbol` (human-facing stable short key),
   - store version-bearing `short_form`/dataset identifiers when returned,
   - persist tool-returned dataset labels verbatim for audit readability.
3. Add a deterministic claim-tier calculator:
   - default to **T0**,
   - promote only when required evidence matrix is satisfied,
   - auto-add caveat text for sex/stage mismatch or unknown scope.
4. Add pre-answer guardrails:
   - block T2/T3 wording unless supporting provenance rows exist in memory,
   - downgrade final wording when `count_status != "exact"` or warnings indicate degraded semantics.
5. Add replay hooks:
   - persist raw request payload + essential response summary for deterministic re-run,
   - include a normalized hash of `(tool_name, request, dataset_scope)` to de-duplicate equivalent evidence calls.

### Implementation note (v1 now wired)

The current implementation writes deterministic v1 fingerprints in these fields:

- `metadata.flybrain_provenance.replay_fingerprint`
- `metadata.flybrain_provenance.replay_fingerprint_version` (currently `"v1"`)
- `metadata.provenance_access_path_fingerprint` (for explicit provenance-bearing findings)
- `metadata.provenance_access_path_fingerprint_version` (currently `"v1"`)

FlyBrain audit entries also carry:

- `replay_fingerprint`
- `replay_fingerprint_version`
- `replay_dataset_scope` (normalized symbols/version IDs parsed from tool output)

## 6) Non-generalization policy (must be explicit in outputs)

- Do not generalize adult findings to larval datasets without direct larval evidence.
- Do not generalize female-derived findings to male wiring without direct male evidence (and vice versa).
- Do not generalize from curated NT declarations to predicted NT confidence behavior.
- Do not compare counts across grouped and non-grouped connectivity modes as if they are equivalent measurements.

When these constraints are violated by missing metadata, emit constrained language ("in this dataset/version/query path") rather than broad biological claims.

## 7) Implementation status note (2026-09-22)

- `investigation_store` now enforces fail-closed validation for FlyBrain `claim_scope` metadata when either `metadata.claim_scope` or `metadata.flybrain_provenance` is present.
- The required tuple keys are validated as:
  `(dataset, dataset_version, sex, life_stage, annotation_completeness, circuit_class, experience_window)`.
- Validated scope is preserved through the memory pipeline by forwarding `claim_scope` into Mnemosyne metadata and surfacing it on Mnemosyne recall rows.

---

**Status**: ready as a handoff artifact for the next FlyBrain provenance PR.
