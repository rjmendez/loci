# Fly-Brain Dataset Provenance Matrix

## Goal

Build a reproducible, evidence-bounded dataset matrix for FlyBrain analysis so future PRs and memory entries can state exactly which dataset/version/access path they used and what scope they are valid for.

This artifact follows the repository guidance in `docs/FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md` and the cross-dataset synthesis in `docs/FLYBRAIN_CROSS_DATASET_LESSONS.md`, and it is grounded in the actual tool surface returned by the FlyBrain/VFB tooling used in this repo.

## Evidence used

- `virtual-fly-brain-list_connectome_datasets` output (current dataset catalog)
- `docs/FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md`
- `docs/FLYBRAIN_CROSS_DATASET_LESSONS.md`
- `virtual-fly-brain-search_terms` for entity resolution and class-level scoping

## Provenance standard

For each dataset, this matrix records:

- dataset symbol
- canonical label
- version signal (the stable version-bearing identifier available in the current tool surface)
- access method
- freshness signal
- explicit scope boundary (sex / stage / anatomical coverage)
- evidence notes for downstream claim limits

Important: in the current VFB tool surface, the freshness signal is not a timestamp. The authoritative freshness signal is the version-bearing identifier embedded in dataset labels and short-form IDs (for example `flywire783`, `male_cns_v1_0`, `neuprint_JRC_Hemibrain_1point2point1`). This is the safe, replayable signal available today.

## Matrix

| dataset symbol | canonical label | version signal | access method | freshness signal | explicit scope boundary | allowed use / not allowed use |
|---|---|---|---|---|---|---|
| `fw` | FlyWire web interface v783 | `flywire783` | VFB/FlyBrain dataset list + connectivity queries on the FlyWire dataset | version-bearing label `v783` and short-form `flywire783` | Adult female brain / FAFB-aligned scope; treat as female-adult evidence unless direct male or larval evidence is added | Valid for adult female FlyWire claims; not a general statement about all FlyBrain data or all sexes/stages |
| `mc` | Neuprint web interface - male-cns:v1.0 | `male_cns_v1_0` | Neuprint-style dataset access through the FlyBrain tooling | version-bearing label `v1.0` and short-form `male_cns_v1_0` | Male whole-CNS scope; different sex and broader anatomy than female brain-only data | Valid for male CNS structural claims; do not apply to female-brain or larval conclusions without direct evidence |
| `BANC` | BANC web interface v888 | `BANC888` | VFB dataset interface / BANC access path | version-bearing label `v888` and short-form `BANC888` | Female whole-CNS scope; broader than a single brain region and not equivalent to a single adult dataset subset | Valid for BANC-female claims; not interchangeable with FlyWire adult female brain or male datasets |
| `hb` | Neuprint web interface - hemibrain:v1.2.1 | `neuprint_JRC_Hemibrain_1point2point1` | Neuprint access path used in FlyBrain connectome queries | version-bearing identifier `1point2point1` and `hb` symbol | Adult hemibrain region; scope is narrower than whole CNS | Valid for hemibrain-specific structural claims; do not generalize to full-brain or larval claims without direct evidence |
| `mv` | Neuprint web interface - MANC:v1.2.1 | `neuprint_JRC_Manc_1_2_1` | Neuprint access path | version-bearing identifier `1.2.1` and `mv` symbol | Male VNC-centric / male connectome scope; not the same as female or mixed adult brain data | Valid for male VNC / MANC-specific wiring claims; not a direct substitute for female adult brain mappings |
| `ol` | Neuprint web interface - JRC_Optic-Lobe:v1.0.1 | `neuprint_JRC_OpticLobe_v1_0_1` | Neuprint access path for optic-lobe-specific connectomics | version-bearing label `v1.0.1` | Optic-lobe-only scope; not whole-brain or whole-CNS | valid for optic-lobe analyses only; do not treat as generic brain-wide connectivity |
| `fafb` | VFB CATMAID Adult Brain (FAFB) | `catmaid_fafb` | VFB CATMAID access path | source label `Adult Brain (FAFB)` with stable CATMAID dataset identity, not a numbered release in the current list | Adult brain scope; tied to the CATMAID FAFB source rather than a full whole-CNS dataset | valid within FAFB adult-brain claims; not interchangeable with full-CNS or larval scope |
| `l1em` | VFB CATMAID L1 CNS | `catmaid_l1em` | VFB CATMAID access path | version-bearing L1 label, stable dataset symbol `l1em` | L1 larval CNS scope; explicitly developmental-stage-specific | valid for L1 larval CNS claims only; do not claim adult generalization without direct adult corroboration |

## Scope boundaries and claim safety

The repo guidance is explicit: do not treat any one dataset as a general truth for all fly-brain subjects.

1. Sex is not interchangeable.
   - FlyWire and BANC are described in the repo as female-oriented datasets.
   - Male CNS and MANC are separate male datasets.
   - Cross-sex claims require direct mixed-sex evidence.

2. Stage is not interchangeable.
   - `l1em` is explicitly larval (L1) scope.
   - The adult datasets are adult-specific unless a specific query or annotation states otherwise.
   - Larval-to-adult transfer is only valid with adult corroboration.

3. Anatomy is not interchangeable.
   - `ol` is optic-lobe specific.
   - `hb` is hemibrain-specific.
   - `fafb` is adult-brain CATMAID scope.
   - `mc` and `BANC` are whole-CNS or broad-brain datasets.

4. Query semantics and dataset filters matter.
   - The docs require recording include/exclude dataset sets, grouping mode, and result contract (`count`, `count_status`, warnings).
   - `query_connectivity` results can vary materially by `group_by_class`, `weight`, `exclude_dbs`, and paging semantics.

## Access-path and freshness guidance for future PR work

Use the following rule when writing FlyBrain-derived findings or summarizing PR results:

- Persist dataset symbol + version-bearing identifier + access path.
- Record whether the data were obtained via `list_connectome_datasets`, `query_connectivity`, `run_query`, or `search_terms`.
- Persist `exclude_dbs`, `group_by_class`, and result warnings when relevant.
- Cap wording to dataset-local or dataset-version-local scope unless the evidence spans multiple datasets with matching sex/stage/anatomy constraints.

## Validation summary

This matrix is validated against the current repo evidence:

- `virtual-fly-brain-list_connectome_datasets` returned the current dataset catalog and version-bearing labels.
- `docs/FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md` explicitly states that version-bearing labels and dataset symbols are required for reproducibility and that dataset/version/sex/stage scope must be stored.
- `docs/FLYBRAIN_CROSS_DATASET_LESSONS.md` explicitly calls out dataset heterogeneity, sex and stage constraints, and the invalidity of over-generalizing from one dataset to another.

No speculative data points are included beyond the observed dataset labels and the scope notes already documented in repo materials.

## Recommended PR-ready phrase

"For all FlyBrain-derived claims, cite both the dataset symbol and its version-bearing identifier, and bind the claim to explicit sex/stage/anatomy scope; without that boundary, the claim remains dataset-local and must not be generalized."

---

Status: complete and ready for use in future fly-brain PR and memory work.
