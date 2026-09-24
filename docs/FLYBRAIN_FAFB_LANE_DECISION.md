# FlyBrain FAFB lane decision

Status: **DEFERRED** (no adapter, no sample builder, no data pulled)
Decided: 2026-09-23
Scope: registry symbol `fafb` (snapshot dir `snapshots/fafb/catmaid_fafb`)

## Question

Does a separate `fafb` lane, built from the VFB-hosted FAFB CATMAID tracing, add value over the
existing `fw` lane (FlyWire v783)? Both come from the same EM volume: FAFB, Zheng et al. 2018,
one adult female brain.

## Evidence

### What the CATMAID FAFB instance holds

We probed the public instance anonymously at https://fafb.catmaid.virtualflybrain.org on
2026-09-23. We read metadata endpoints only and saved nothing under `F:\.flybrain`.

| Probe | Result |
|---|---|
| `GET /projects/` | 1 project ("Adult Brain"), 1 stack `fafb`, dims 253952 x 155648 x 7063 |
| `GET /1/stats/nodecount` | 242 contributing users, about 23.2 M skeleton nodes in total |
| `GET /1/annotations/` | 6,013 annotations: 329 `FBbt:` terms, 28 `Paper: ...` tags, `publication_link` / `publication_timestamp` tags, and LH/PN cell-type names such as `LHAV3f1#1_L` |
| Any `license` annotation | none |
| Bulk export endpoint | none. The API is per-query (skeletons, connectors, annotations). No versioned dump file with a Content-Length exists. |

The instance holds manually traced neurons from about 25 published studies: the mushroom body
(Bates and Schlegel 2020, Zheng 2018/2022), the lateral horn (Dolan 2019), the CX (Sayin 2019),
gustatory circuits (Engert 2022) and others. The data is **sparse and partial**. Only published
arbors are present, and many synapses attach to untraced fragments.

The phase-5 P0 smoke report (`F:\.flybrain\logs\phase5-dataset-expansion-20260923T205010Z\p0-smoke-report.md`)
had already found zero rows for a class-level Tm1->T3 connectivity query. The optic lobe is
essentially untraced there, so connectivity labels would be missing for whole regions.

### What `fw` already provides from the same volume

- `proofread_connections_783.feather` (852 MB): whole-brain, proofread neuron-to-neuron edges
  with per-edge neuropil and NT predictions. That covers roughly 139k neurons (Codex FAFB tile
  stats, `branch-qc-20260923T204457Z-r2/external-size-evidence-report.md`).
- `per_neuron_neuropil_count_{pre,post}_783.feather`: FlyWire neuropil region assignment for
  every proofread neuron.
- Cell-type identity was cross-matched to hemibrain and to the older FAFB CATMAID tracings by
  Schlegel et al. 2024. The BANC repo clone already references that table as
  `BANC-project/data/meta/fafb_schlegel_et_al_2024_meta.tsv` (currently a 0-byte LFS stub).

### Objective-by-objective comparison

| Objective | What CATMAID `fafb` could add | Why it does not beat `fw` |
|---|---|---|
| `connectivity_tier` | Manually traced synapse counts for a few thousand published neurons | Tracing is partial, so counts are biased low and depend on which study traced the neuron. The same biological neurons appear in `fw` with complete proofread edges. |
| `region_specialization_tier` | Region assignment from CATMAID volumes | Assigning regions needs skeleton or connector node coordinates, which means per-skeleton pulls. The DATA-PULL RULES forbid skeletons. `fw` already ships neuropil counts per neuron. |
| `neurotransmitter_dominance` | None: CATMAID has no NT predictions | Not in the registry allow-list for `fafb`, and not available from the source anyway |

### Blocking issues under the DATA-PULL RULES

1. **No selective bulk product.** No edgelist, annotation or NT file can be sized with HEAD
   before download. Building one would mean scraping thousands of per-skeleton API calls, which
   is effectively a skeleton/connector mirror. The rules forbid skeletons and bulk mirrors.
2. **License is not clear at the export level.** VFB's per-neuron term pages mark FAFB neuron
   data as CC-BY-4.0. The CATMAID instance itself carries no license annotation, and the hosting
   site footer says "All Rights Reserved". Each neuron also requires citing its own source
   publication. That makes per-row citation mandatory, and our manifest schema has no way to
   express it for an API scrape.
3. **The same neurons would sit in two lanes.** The CATMAID skeletons and the FlyWire v783
   neurons are the same cells in the same animal. A `fafb` lane used as a held-out or A/B arm
   against `fw` (AB plan AC1-AC5) would put identical biological units in train and test. That
   breaks the evaluation independence assumed by safety constraints SC1-SC4 and the label-hygiene
   rule.

## Decision

`fafb` is **deferred as a redundant alias of `fw`**. For any structural objective it contributes
only a sparse, partial, same-animal subset of what `fw` already holds, with no license-clear bulk
export. We pulled no data, and the existing metadata-only snapshot
(`snapshots/fafb/catmaid_fafb/{source,metadata}`) is left untouched.

The one real value in the CATMAID instance is **annotation provenance**: paper tags, `FBbt:`
terms, and LH/MB type names. That belongs to the `fw` lane as a label-enrichment table. The
source to use is the Schlegel et al. 2024 FlyWire annotation release, keyed to v783 root IDs,
not the CATMAID skeleton IDs.

## Follow-ups (not in this change)

1. **Registry change request** (integrator, `mcp/flybrain_dataset_registry.py`):
   - Keep `fafb` at `DatasetStatus.PLANNED`. Do not activate it.
   - Change its `description` to say it is deferred as an alias of `fw` (same FAFB volume) and
     point to this document.
   - Optionally add an alias marker, e.g. `alias_of="fw"`, or a `DEFERRED` status, if the
     registry grows one. Until then, `build_training_samples("fafb", ...)` fails closed with
     `BUILDER_NOT_REGISTERED`, because no `flybrain_brain_cluster_fafb_samples` module exists.
     That is the intended behavior.
   - Reduce `supported_objectives` for `fafb` to `()`, or leave it as is. Either choice is safe,
     because no builder is registered.
2. **`fw` enrichment** (a separate `fw` lane task): pull the FlyWire annotation TSV from the
   Schlegel et al. 2024 release, license CC-BY 4.0, cited per the flywire_annotations repo. It
   adds cell-type and hemibrain-match labels keyed to v783 root IDs. That would support
   `region_specialization_tier` for `fw` without a new lane.
3. **Revisit triggers.** Reopen this decision only if one of these happens: VFB publishes a
   versioned, license-stated bulk export (edgelist or annotations) for the CATMAID instance; or
   a downstream objective specifically needs *manual-tracing vs. automated-segmentation*
   agreement. That would be a QA/calibration use, not a training lane.

## Citations (if any CATMAID FAFB data is used in future)

- Zheng Z, et al. (2018) A Complete Electron Microscopy Volume of the Brain of Adult Drosophila
  melanogaster. Cell 174(3):730-743.e22. https://doi.org/10.1016/j.cell.2018.06.019
- Saalfeld S, Cardona A, Hartenstein V, Tomancak P (2009) CATMAID. Bioinformatics
  25(15):1984-1986. https://doi.org/10.1093/bioinformatics/btp266
- The source publication tagged on each neuron (`Paper: ...` annotation).
