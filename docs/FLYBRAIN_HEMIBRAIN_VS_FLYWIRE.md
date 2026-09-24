# Hemibrain v1.2.1 vs. FlyWire v783 — Comparison Reference

**Scope of this document:** A rigorous, evidence-backed comparison of the two principal adult Drosophila brain connectome datasets available through the VFB/Loci tool surface, intended to guide scoped claim-making and prevent cross-dataset overgeneralization.

**Reading path:** After reading [FLYBRAIN_GUIDE.md](./FLYBRAIN_GUIDE.md) and [FLYBRAIN_DATASET_PROVENANCE_MATRIX.md](./FLYBRAIN_DATASET_PROVENANCE_MATRIX.md), use this document when a claim involves both `hb` (hemibrain) and `fw` (FlyWire) data, or when you need to decide which dataset is appropriate for a given question.

---

## 1. Comparison Matrix

| Dimension | Hemibrain v1.2.1 (`hb`) | FlyWire v783 (`fw`) |
|---|---|---|
| **Anatomical coverage** | ~one-third of the adult female central brain; excludes both optic lobes in full and excludes the ventral nerve cord | Whole adult female brain: full central brain + both optic lobes; excludes ventral nerve cord |
| **Sex / stage scope** | Adult female (right hemisphere of the central brain) | Adult female (whole brain) |
| **Neuron count** | ~22,000–26,000 (≈25,842 in v1.2.1) | ~139,255 proofread neurons |
| **Synapse count** | ~9.5 million total synapse contacts; ~3.8 million connection pairs | ~54.5 million annotated synapses |
| **Dataset fraction** | ~35% of the central brain volume represented | Whole-brain extent (central brain + optic lobes); completeness not 100% but the stated goal is all neurons |
| **Annotation model** | Expert-assigned neuron type labels integrated into a Neo4j graph (neuPrint); annotations are fixed per release | Hierarchical community+expert annotation: flow, superclass, cell class, nerve, lineage, morphology, neurotransmitter predictions; annotations versioned via CAVE and openly published |
| **Cell-type catalog status** | Mature, stable; cell type assignments for covered regions are high-quality and well-established | Larger and ongoing; flagship paper (Nature 2024) covers >139k neurons; optic lobe types from companion dataset (Matsliah et al. 2024) merged in |
| **Update cadence** | Fixed; last substantive update circa 2021–2022; no continuous extension expected | Continuous proofreading; v783 is the frozen cite-able snapshot (used for the flagship 2024 paper); future snapshots may be released |
| **Versioning model** | Stable frozen release; neuron body IDs (`hb` body IDs) are stable across queries to v1.2.1 | Chunkedgraph-based live proofreading; root IDs can shift with edits; v783 snapshot pins IDs for reproducibility |
| **Primary query interface** | neuPrint (Neo4j Cypher); `neuprint-python` (Python); `hemibrainr` (R); REST API | CAVE client (Python `caveclient`), Codex web interface, `fafbseg-py` / `fafbseg` (R); also queryable via VFB tool surface (symbol `fw`) |
| **VFB symbol / version ID** | `hb` / `neuprint_JRC_Hemibrain_1point2point1` | `fw` / `flywire783` |
| **Proofreading approach** | Expert-focused, regionally concentrated on core covered areas | Crowdsourced + expert; probabilistic uncertainty targeting; algorithmic prioritization of uncertain regions |
| **False merge/split rate (post-proofreading)** | Low in core proofread regions; higher uncertainty at coverage boundaries; no recent comprehensive update | Major branches: <1% false merge, <1–2% false split after expert proofreading of significant branches; twigs and finest processes have higher residual error (Nature 2024 supplementary) |
| **Uncertainty tracking** | Basic quality-control metrics; no continuous probabilistic uncertainty model | Quantitative uncertainty from AI segmentation used to drive proofreading priorities; more mature error characterization |
| **Neurotransmitter annotations** | Curated where available via VFB/FBbt ontology; predicted values via `get_predicted_neurotransmitters` | Community-curated + model-predicted with confidence scores; dominant neurotransmitter assignment included in v783 annotation files |
| **Cross-dataset matching** | `hemibrainr` package provides matching between hemibrain and FlyWire neuron identities via NBLAST morphology | FlyWire v783 annotation files include VirtualFlyBrain IDs and cross-references to published cell-type identities |
| **License / access** | Publicly accessible via neuPrint (requires free token); data download available | Publicly accessible via Codex (flywire.ai) and CAVE (requires free token); annotation files on GitHub; snapshot data as CSV/Parquet |
| **Best use in Loci** | Stable reference for central brain regions within its coverage; well-understood error profile; safe for reproducibility-critical comparisons within those regions | Best for whole-brain and optic-lobe questions; most complete adult female neuron catalog; preferred for new analyses requiring broad coverage |

**Primary citations:**
- Hemibrain: Scheffer et al., *eLife* 2020 (DOI: 10.7554/eLife.57443); neuPrint v1.2.1 dataset at Janelia Research Campus.
- FlyWire: Dorkenwald et al. and Schlegel et al., *Nature* October 2024 (FlyWire Consortium flagship papers); CAVE: Dorkenwald et al., *Nature Methods* 2024 (DOI: 10.1038/s41592-024-02426-z); annotation files: github.com/flyconnectome/flywire_annotations.

---

## 2. Scope Constraints and Cross-Dataset Guardrails

These guardrails apply whenever a claim involves both datasets, or when generalizing from one to the other.

### 2.1 Anatomy — do not treat hemibrain coverage as whole-brain coverage

Hemibrain covers approximately one-third of the adult female central brain. It does not include both optic lobes fully, and it does not include the ventral nerve cord. FlyWire covers the entire brain (central brain + optic lobes) but also excludes the VNC.

**Guardrail:** A hemibrain connectivity result is a central-brain claim, not a whole-brain claim. Optic-lobe questions require FlyWire or the dedicated Optic Lobe dataset (`ol`). VNC questions require MANC (`mv`) or male-CNS (`mc`).

### 2.2 Neuron identity — body IDs and root IDs are not interchangeable

Hemibrain body IDs (`hb`) and FlyWire root IDs (`fw`) refer to different reconstructions of different specimens. A neuron labeled "LC4" in hemibrain and "LC4" in FlyWire are the same cell type, but their connectivity matrices, synapse counts, and partner lists are distinct observations from two separate EM datasets.

**Guardrail:** Do not mix `hb` body IDs and `fw` root IDs in the same connectivity query or count. Use `hemibrainr`-based morphological matching or VFB cross-reference IDs when explicitly claiming cross-dataset type correspondence.

### 2.3 Synapse counts — scale difference is not an accuracy signal

FlyWire reports ~54.5 million synapses vs. hemibrain's ~9.5 million. This reflects coverage difference (whole-brain vs. partial central brain), not that hemibrain is less accurate. Synapse counts within hemibrain's covered regions are valid on their own terms.

**Guardrail:** When comparing synapse counts across datasets, normalize by the region or neuron population being compared, not by raw totals.

### 2.4 Annotation model — hierarchical vs. flat type labels

FlyWire uses a hierarchical annotation model (flow → superclass → cell class → type). Hemibrain uses a flatter type-name model. Cell type names do not always map 1:1.

**Guardrail:** When asserting that a cell type name used in hemibrain matches a FlyWire annotation, cite the explicit cross-reference source (e.g., hemibrainr matching, VFB ontology ID, or paper citation). Do not assume label equivalence by string match alone.

### 2.5 Versioning — hemibrain is frozen, FlyWire is a snapshot

Hemibrain v1.2.1 is a stable, frozen dataset with stable body IDs. FlyWire's live proofreading means that the ID space can change; v783 is the frozen snapshot that corresponds to the flagship 2024 papers. Future FlyWire snapshots may have different IDs.

**Guardrail:** For reproducibility, always cite the FlyWire snapshot version (v783 or later) alongside queries. Hemibrain claims can cite v1.2.1 as stable. Do not assume FlyWire IDs obtained outside the v783 snapshot are equivalent to v783 IDs.

### 2.6 Sex — both are adult female; do not generalize to male

Both hemibrain and FlyWire are adult female datasets. Claims from either dataset apply to the adult female brain only unless explicitly corroborated in a male dataset (e.g., male-CNS `mc` or MANC `mv`).

**Guardrail:** Do not treat hemibrain or FlyWire results as sex-universal facts. Tag all claims with `sex: female, stage: adult`.

### 2.7 Error profile — uncertainty differs by region and dataset

Hemibrain has a well-characterized but regionally variable error profile, concentrated in its core proofread area. FlyWire's post-proofreading error rate is <1–2% for major branches but higher for fine processes ("twigs") and boundary regions. The two datasets have different error modes that are not directly comparable without considering region and branch-class.

**Guardrail:** Do not assert that one dataset is universally more accurate without specifying the region and branch class. For the regions hemibrain does cover, its error rate is well-characterized and valid. For whole-brain or optic-lobe questions, FlyWire post-proofreading rates are the better reference.

---

## 3. What Each Dataset Is Best For

| Question type | Recommended dataset | Rationale |
|---|---|---|
| Central brain connectivity (learning, navigation, decision circuits) | Either; hemibrain for historical reference and stability, FlyWire for broader context | Both cover the central brain; hemibrain is stable and well-cited; FlyWire provides more neurons and partners |
| Optic lobe / visual system circuits | FlyWire (`fw`) or Optic Lobe (`ol`) | Hemibrain does not fully cover the optic lobes |
| VNC / motor circuits | MANC (`mv`) or male-CNS (`mc`) | Neither hemibrain nor FlyWire covers the VNC |
| Whole-brain cell type catalog | FlyWire (`fw`) | Only dataset with ~139k neurons covering the full adult female brain |
| Stable, frozen reference for reproducibility | Hemibrain v1.2.1 (`hb`) | Frozen release; body IDs are stable; long track record of validated analyses |
| Cross-sex comparison (male vs. female) | Requires separate male dataset + female dataset | Do not infer sex-universal claims from either hemibrain or FlyWire alone |
| Neurotransmitter assignment (curated) | `get_known_neurotransmitters` via VFB ontology (dataset-agnostic) | Ontology-curated, no confidence score |
| Neurotransmitter assignment (predicted) | `get_predicted_neurotransmitters` with `exclude_dbs` set explicitly | Use `split_by_dataset=true` to see per-dataset agreement |
| Larval circuit questions | CATMAID L1 CNS (`l1em`) | Adult datasets (hemibrain, FlyWire) are not valid for larval claims |
| New analyses requiring large partner sets | FlyWire (`fw`) | ~6x more neurons and ~5.7x more synapses than hemibrain |
| Analyses that cite a specific 2020–2022 paper's hemibrain result | Hemibrain v1.2.1 (`hb`) | Match the dataset version to the citation to avoid version drift |

---

## 4. Practical Integration Recommendations for Loci

### 4.1 Dataset selection logic

When routing a connectivity or anatomy question, apply this selection order:

1. Check whether the question is optic-lobe–specific → use `fw` or `ol`.
2. Check whether the question is VNC-specific → use `mv` or `mc`.
3. Check whether the question is larval → use `l1em`.
4. For adult female central-brain questions: both `hb` and `fw` are valid; prefer `fw` for breadth, `hb` for stability when citing existing literature.

### 4.2 Provenance tagging for hemibrain vs. FlyWire claims

All claims derived from either dataset must carry the FlyBrain provenance envelope (see [FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md](./FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md)) with at minimum:

- `dataset_symbol`: `hb` or `fw`
- `version_id`: `neuprint_JRC_Hemibrain_1point2point1` or `flywire783`
- `sex`: `female`
- `stage`: `adult`
- `anatomy_scope`: the specific region or neuron class queried
- `evidence_family`: `connectivity`, `predicted_neurotransmitter`, `curated_neurotransmitter`, etc.

### 4.3 Cross-dataset claim gating

Gate or down-rank a claim when:

- It asserts equivalence between an `hb` body ID and an `fw` root ID without citing a morphological matching source.
- It generalizes optic-lobe or VNC connectivity from a hemibrain result.
- It presents a synapse count comparison between datasets without normalizing by region or neuron population.
- It claims a cell type "is the same" across datasets based only on label string matching.
- It asserts sex-universal or stage-universal validity from either dataset alone.

### 4.4 VFB tool surface usage notes

Within the Loci VFB tool surface:
- `list_connectome_datasets` returns both `hb` (`neuprint_JRC_Hemibrain_1point2point1`) and `fw` (`flywire783`) as active entries — both are queryable.
- `query_connectivity` results vary by `exclude_dbs`; when comparing datasets, run separate queries per dataset rather than relying on a single merged result.
- `get_predicted_neurotransmitters` with `split_by_dataset=true` is the correct call for cross-dataset neurotransmitter comparison.
- Empty results from one dataset do not indicate absence of biology; they may indicate the neuron type is outside that dataset's coverage.

### 4.5 Documentation and memory update guidance

When storing a FlyBrain-derived finding in Loci memory:
- For hemibrain results: mark `dataset_version: v1.2.1`, note that the dataset is frozen and has not been updated since ~2022.
- For FlyWire results: mark `dataset_version: v783` (the snapshot), note that the live proofreading version may differ and that root IDs are snapshot-specific.
- Do not store a claim as "confirmed in hemibrain and FlyWire" without both an explicit `hb` query result and an explicit `fw` query result with matched scope.

---

## 5. Where This Document Lives in the Doc Set

This document is a canonical reference within the active FlyBrain doc set. It is indexed from [FLYBRAIN_GUIDE.md](./FLYBRAIN_GUIDE.md) and from [FLYBRAIN_DATASET_PROVENANCE_MATRIX.md](./FLYBRAIN_DATASET_PROVENANCE_MATRIX.md).

| Relationship | Document |
|---|---|
| Entry point / reading path | [FLYBRAIN_GUIDE.md](./FLYBRAIN_GUIDE.md) |
| Dataset-level scope map (all datasets) | [FLYBRAIN_DATASET_PROVENANCE_MATRIX.md](./FLYBRAIN_DATASET_PROVENANCE_MATRIX.md) |
| Provenance envelope requirements | [FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md](./FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md) |
| Terminology | [FLYBRAIN_REASONING_GLOSSARY.md](./FLYBRAIN_REASONING_GLOSSARY.md) |
| Archive / redirect ledger | [FLYBRAIN_ARCHIVE.md](./FLYBRAIN_ARCHIVE.md) |

---

## 6. Evidence Sources

| Claim | Source |
|---|---|
| Hemibrain neuron count (~25,842), synapse count (~9.5M), coverage (~35% central brain) | Scheffer et al., *eLife* 2020; Janelia FlyEM hemibrain project page; github.com/FlyBrainLab/datasets/hemibrain/v1.2/README.md |
| FlyWire v783 neuron count (~139,255), synapse count (~54.5M), whole-brain scope | Dorkenwald et al. + Schlegel et al., *Nature* 2024; flywire.ai; NeuroTrailblazers dataset catalog |
| FlyWire annotation hierarchy (flow, superclass, cell class, etc.) | github.com/flyconnectome/flywire_annotations; Zenodo supplemental files (10877326) |
| FlyWire CAVE versioning and chunkedgraph model | Dorkenwald et al., *Nature Methods* 2024 (DOI: 10.1038/s41592-024-02426-z) |
| FlyWire post-proofreading error rates (<1% false merge/split for major branches) | FlyWire Nature 2024 flagship papers (supplementary material) |
| Hemibrain query interface (neuPrint, Cypher, neuprint-python) | neuprint.janelia.org; VFB neuPrint tutorial (virtualflybrain.org) |
| FlyWire query interface (CAVE, Codex, fafbseg-py) | flywire.ai/apps; github.com/seung-lab/CAVEclient |
| Cross-dataset neuron matching via hemibrainr | natverse.org/hemibrainr; github.com/natverse/hemibrainr |
| VFB dataset symbols and version IDs (`hb`, `fw`, etc.) | `virtual-fly-brain-list_connectome_datasets` tool output (current session) |
| Optic lobe companion annotations (Matsliah et al. 2024) | Nature 2024 FlyWire companion paper series |
| Hemibrain last update circa 2021–2022 | Janelia FlyEM project documentation |

---

Status: canonical reference document; version 1.0; produced for the `flybrain-hemibrain-vs-flywire` task.
