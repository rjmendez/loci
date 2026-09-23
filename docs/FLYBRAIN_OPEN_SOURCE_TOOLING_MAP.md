> Archived / retired from active navigation
>
> This page is kept only for historical continuity and searchability. It is no longer part of the default FlyBrain reading path.
>
> Canonical entry point: [FLYBRAIN_GUIDE.md](./FLYBRAIN_GUIDE.md)
> Archive home: [FLYBRAIN_ARCHIVE.md](./FLYBRAIN_ARCHIVE.md)
> For current guidance, use the guide and the evidence/provenance docs before digging into historical detail.
>
# Fly-Brain Open-Source Tooling Map

**Purpose**: Concrete, evidence-backed map of the open-source tooling landscape for
fly-brain data access, query, and simulation — grouped by capability and anchored to
specific Loci integration points.

**Audience**: next PR adding new FlyBrain-sourced data pipelines, simulation hooks, or
ontology-driven reasoning to Loci.

**Related docs**:
- `docs/FLYBRAIN_GUIDE.md` — orientation and core message
- `docs/FLYBRAIN_DATASET_PROVENANCE_MATRIX.md` — dataset-by-dataset scope bounds
- `docs/FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md` — minimum reproducibility envelope
- `docs/FLYBRAIN_INSPIRATION_MAP.md` — concept-to-Loci-subsystem placement

---

## 1. Data Access / Query APIs

### Virtual Fly Brain (VFB) MCP + REST API

| Attribute | Value |
|---|---|
| **Purpose** | Unified neuroinformatics hub: ontology-grounded search, 3D image data, connectomics queries, expression data, and annotation across all major Drosophila datasets |
| **License** | CC BY 4.0 (ontology data); server code open-source at [github.com/VirtualFlyBrain](https://github.com/VirtualFlyBrain) |
| **Maturity** | High — production system cited in >200 publications; actively maintained through 2026; REST API (`v3-cached.virtualflybrain.org`), MCP server, Python client (VFBconnect), R client all available |
| **Key integration surface** | 11 MCP tools already wired into this session (`virtual-fly-brain-*`): `query_connectivity`, `search_terms`, `get_term_info`, `run_query`, `list_connectome_datasets`, `get_hierarchy`, `resolve_entity`, `resolve_combination`, `list_search_facets`, `get_known_neurotransmitters`, `get_predicted_neurotransmitters` |
| **Datasets covered** | FlyWire v783 (`fw`), Hemibrain v1.2.1 (`hb`), MANC v1.2.1 (`mv`), Male-CNS v1.0 (`mc`), BANC v888 (`BANC`), Optic Lobe v1.0.1 (`ol`), FAFB CATMAID (`fafb`), L1 larval CNS (`l1em`) — as enumerated by `list_connectome_datasets` |

**Loci integration points**:
1. **Provenance-bound memory writes**: `query_connectivity` and `run_query` results can be stored with full provenance envelopes (see `FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md`) and retrieved via `rag_context_search` or `investigation_search`.
2. **Ontology-grounded claim validation**: `get_term_info` + `get_hierarchy` resolve FBbt IDs into typed scope labels (sex/stage/anatomy) that feed into `investigation_pre_answer_check` boundary checks.

**References**: [virtualflybrain.org/docs/apis](https://www.virtualflybrain.org/docs/apis/) · [VFB3-MCP repo](https://github.com/VirtualFlyBrain/VFB3-MCP) · Court et al. 2023, *Frontiers in Physiology* 14:1076533

---

### neuPrint / neuprint-python

| Attribute | Value |
|---|---|
| **Purpose** | Graph-structured connectome database and query platform for Janelia EM datasets (Hemibrain, MANC, Optic Lobe, Male-CNS); Python client `neuprint-python` provides programmatic access to neurons, synapses, ROIs, and paths |
| **License** | BSD (neuprint-python) |
| **Maturity** | High — production system from Janelia Research Campus; neuprint-python on PyPI, actively maintained, widely used across the connectomics community |
| **Key integration surface** | REST/Cypher queries to neuPrint server; `neuprint-python` client for neurons, synapses, pathfinding, ROI-based queries |

**Loci integration points**:
1. **Augmented dataset coverage**: `neuprint-python` queries can supply ROI-level synapse counts and path queries not exposed via the VFB MCP surface, which can then be stored as typed memory entries with `investigation_store`.
2. **Cross-dataset corroboration**: Hemibrain (`hb`) and MANC (`mv`) data retrieved via neuprint-python can be compared against VFB connectivity results to drive `conflict_list` / `conflict_resolve` checks.

**References**: [github.com/connectome-neuprint/neuprint-python](https://github.com/connectome-neuprint/neuprint-python) · [neuprint.janelia.org](https://neuprint.janelia.org) · [virtualflybrain.org/docs/tools/neuprint](https://www.virtualflybrain.org/docs/tools/neuprint/)

---

### fafbseg (FAFBseg-py)

| Attribute | Value |
|---|---|
| **Purpose** | Python library for working with FAFB and FlyWire datasets: fetch segmentation IDs and meshes/skeletons, map coordinates, query FlyWire annotations and neuroglancer URLs, transform between brain templates |
| **License** | GPL-3.0 |
| **Maturity** | High — v3.2.2 on PyPI; actively maintained by Jefferis lab; deep integration with navis and the natverse ecosystem |
| **Key integration surface** | `fafbseg.flywire` module for FlyWire neuron lookup, annotation reading, materialization version handling; `fafbseg.xform` for template transforms |

**Loci integration points**:
1. **Materialization-version pinning**: `fafbseg` materialization version maps to the VFB `flywire783` version signal; storing both together in provenance ensures claims are reproducible as the FlyWire proofreading version advances.
2. **Coordinate-to-neuron resolution**: raw EM coordinates from image stacks can be resolved to neuron segment IDs, which then feed downstream VFB `search_terms` / `get_term_info` queries for ontology-linked storage.

**References**: [fafbseg-py.readthedocs.io](https://fafbseg-py.readthedocs.io/) · [pypi.org/project/fafbseg](https://pypi.org/project/fafbseg/)

---

## 2. Connectomics Analysis

### navis

| Attribute | Value |
|---|---|
| **Purpose** | General-purpose Python library for neuron and connectome analysis and visualization: skeleton/mesh loading, morphometrics (NBLAST, Strahler, cable length, tortuosity), graph-theoretic connectivity, multi-format I/O, 3D plotting |
| **License** | GPL-3.0 |
| **Maturity** | High — v1.12.0 (July 2024); active development with Rust-compiled `navis-fastcore` backend for NBLAST; well-documented; integrates with neuPrint, CATMAID, NeuroMorpho, Blender, NEURON |
| **Key integration surface** | `navis.nblast()` for morphology comparison; `navis.interfaces.neuprint` and `.catmaid` for dataset connectors; `navis.plot3d` / `plot2d` for visualization |

**Loci integration points**:
1. **Morphology-similarity scoring as memory confidence signal**: NBLAST similarity scores between neuron pairs can be stored as structured memory entries and used to weight retrieval confidence in `memory_confidence` / `memory_promote`.
2. **Cross-dataset skeleton alignment**: navis template-transform pipelines (using `flybrains` space transforms) resolve the same neuron in different dataset coordinate spaces, supporting multi-dataset claim merging.

**References**: [github.com/navis-org/navis](https://github.com/navis-org/navis) · [navis.readthedocs.io](https://navis.readthedocs.io/) · ~130 stars on GitHub

---

### cocoa / coconat / coconatfly (Jefferis Lab Comparative Connectomics)

| Attribute | Value |
|---|---|
| **Purpose** | Comparative connectomics across multiple fly datasets: `cocoa` (Python), `coconat`/`coconatfly` (R/natverse) provide dataset-agnostic query interfaces, hemibrain vs FlyWire cross-dataset neuron-matching, and connectivity comparison |
| **License** | MIT (cocoa); GPL-3.0 (coconat) |
| **Maturity** | Medium — actively developed, marked experimental; `coconatfly` under rapid development as of 2024; Python `cocoa` lags R ecosystem feature parity |
| **Key integration surface** | `cocoa` Python API for cross-dataset neuron match queries; `coconatfly` R package for hemibrain ↔ FlyWire comparison |

**Loci integration points**:
1. **Cross-dataset entity canonicalization**: cocoa's neuron-matching output (cross-dataset ID pairs with similarity) can populate `entity_list` entries with multi-dataset provenance, making the same neuron traceable across `fw`, `hb`, and `mc`.
2. **Conflict resolution input**: when cocoa flags a neuron as divergent between datasets, that signal directly feeds `conflict_list` / `conflict_resolve` — converting connectivity disagreement into a first-class Loci conflict finding.

**References**: [github.com/flyconnectome/cocoa](https://github.com/flyconnectome/cocoa) · [github.com/natverse/coconatfly](https://github.com/natverse/coconatfly) · [flyconnecto.me/tools](https://flyconnecto.me/tools/)

---

## 3. Morphology / Visualization

### Neuroglancer

| Attribute | Value |
|---|---|
| **Purpose** | Browser-based 3D viewer for large volumetric and segmentation neuroimaging data; used by FlyWire, Janelia, and VFB for interactive exploration of EM stacks, meshes, and annotations |
| **License** | Apache 2.0 |
| **Maturity** | High — production system at multiple major labs; actively maintained by Google/Seung lab; widely deployed across all major connectomics projects |
| **Key integration surface** | URL-based state sharing (embed neuron/segment lists in shareable URLs); `cloudvolume` Python library for programmatic I/O; fafbseg generates FlyWire Neuroglancer URLs natively |

**Loci integration points**:
1. **Visualization permalinks as evidence citations**: Neuroglancer state URLs generated by fafbseg or VFB can be stored as `source_url` fields in memory entries, providing a direct visual reference for claims about specific neurons or volumes.
2. **Annotation handoff**: Neuroglancer annotation layers can be exported as structured JSON and ingested as memory entries, connecting manual expert annotations to the Loci store.

**References**: [github.com/google/neuroglancer](https://github.com/google/neuroglancer)

---

### CloudVolume

| Attribute | Value |
|---|---|
| **Purpose** | Python library for reading/writing cloud-hosted neuroimaging datasets in Precomputed, N5, Zarr, and other formats; foundational I/O layer under FlyWire and Neuroglancer |
| **License** | BSD |
| **Maturity** | High — production system from Seung lab (Princeton); actively maintained; widely used across connectomics community |
| **Key integration surface** | `CloudVolume` Python API for voxel, mesh, and skeleton I/O; direct access to FlyWire and BANC volumes |

**Loci integration points**:
1. **Volume-level provenance**: segment metadata fetched via `CloudVolume` (e.g., mesh version, agglomeration level) can be stored alongside VFB query results to tie morphological data to the same materialization checkpoint.
2. **Large-batch skeleton ingestion**: `CloudVolume.skeleton.get()` can retrieve skeletons for a neuron population, enabling bulk NBLAST runs that feed morphology-similarity findings into Loci memory.

**References**: [github.com/seung-lab/cloud-volume](https://github.com/seung-lab/cloud-volume)

---

### CATMAID + pymaid

| Attribute | Value |
|---|---|
| **Purpose** | CATMAID is a collaborative annotation and reconstruction tool for EM stacks; `pymaid` is a Python client for CATMAID servers hosting FAFB and L1-larval datasets used in VFB |
| **License** | AGPL-3.0 (CATMAID); GPL-3.0 (pymaid) |
| **Maturity** | High — CATMAID is a decade-old production system; `pymaid` actively maintained and part of navis/natverse ecosystem |
| **Key integration surface** | `pymaid.get_neuron()`, `.get_connectors()`, `.get_annotations()` for FAFB and l1em dataset access; interoperates with navis |

**Loci integration points**:
1. **Historical/FAFB claims**: CATMAID `fafb` and `l1em` datasets have different provenance than FlyWire; `pymaid` queries tagged with CATMAID stack ID map to the VFB dataset symbols `fafb` / `l1em` and can be stored with matching provenance scope.
2. **Annotation layer ingest**: CATMAID neuron annotations (cell types, literature references) can be batch-fetched via pymaid and stored as typed memory entries, enriching ontology-linked entity records.

**References**: [github.com/navis-org/pymaid](https://github.com/navis-org/pymaid) · [catmaid.readthedocs.io](https://catmaid.readthedocs.io/)

---

## 4. Simulation

### eonsystemspbc/fly-brain (Whole-Brain Spiking Simulator)

| Attribute | Value |
|---|---|
| **Purpose** | Leaky integrate-and-fire simulator of the whole adult fly brain built from the FlyWire connectome (~138k neurons, ~5M synapses); supports multiple backends: Brian2, Brian2-CUDA, PyTorch, GeNN, experimental NEST GPU |
| **License** | MIT |
| **Maturity** | Medium — actively developed community project; no numbered release yet; backends vary in maturity; well-suited for benchmarking and optogenetic manipulation studies |
| **Key integration surface** | Python runner scripts with connectome CSV/JSON input; Brian2 integration for spike propagation; optogenetic silence/activate interfaces |

**Loci integration points**:
1. **Simulation-derived hypotheses as findings**: spike propagation results (e.g., "silencing DM1 PN reduces MB Kenyon cell activation by X%") can be stored as typed `hypothesis` findings in Loci with `investigation_store`, linked to the FlyWire dataset version and simulation parameter set.
2. **Parameter-space tracing**: simulation run configurations (neuron count, synapse weights, backend) stored in memory entries allow `investigation_as_of` to replay which hypothesis was tested under which conditions.

**References**: [github.com/eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain)

---

### Brian2

| Attribute | Value |
|---|---|
| **Purpose** | General-purpose Python-based spiking neural network simulator; widely used for implementing connectome-derived neural models of Drosophila circuits |
| **License** | CeCILL-2.1 (open-source compatible) |
| **Maturity** | High — v2.7+ stable; well-documented; broad community adoption for computational neuroscience |
| **Key integration surface** | `brian2` Python API; connects to FlyWire-derived adjacency matrices for circuit-level simulations |

**Loci integration points**:
1. **Circuit simulation results as provenance-tagged claims**: Brian2 simulation outputs (membrane potential traces, spike trains) stored as structured findings with provenance tied to FlyWire dataset version and parameter set.
2. **Model-evidence linking**: simulation parameter sweeps stored in Loci enable `conflict_list` checks when simulation predictions diverge from anatomical connectivity claims.

**References**: [brian2.readthedocs.io](https://brian2.readthedocs.io/) · [briansimulator.org](https://briansimulator.org/)

---

## 5. Annotation / Ontology

### FlyBase Drosophila Anatomy Ontology (FBbt)

| Attribute | Value |
|---|---|
| **Purpose** | Structured, OWL2-formalized ontology of Drosophila anatomy: ~15,000 terms covering brain regions, cell types, developmental stages, with extensive synonymy for literature-linked retrieval |
| **License** | CC BY 4.0 |
| **Maturity** | High — actively maintained by FlyBase; version-tracked OBO/OWL/JSON releases on GitHub; foundational to VFB, FlyBase, and all major annotation systems |
| **Key integration surface** | OBO/OWL files on GitHub; SPARQL endpoint; VFB `get_term_info` resolves FBbt IDs to typed, synonym-enriched records; `get_hierarchy` traverses `part_of` and `subclass_of` relations |

**Loci integration points**:
1. **Ontology-grounded claim scoping**: every VFB query result carries an FBbt ID as the entity identifier; storing that ID alongside claims allows `investigation_entity_lookup` / `entity_timeline` to retrieve all findings for a typed anatomical entity across sessions.
2. **Scope validation in pre-answer checks**: `get_hierarchy` traversal provides the full anatomical containment chain (e.g., "optic lobe" ⊂ "adult brain"), enabling `investigation_pre_answer_check` to flag when a claim about a sub-region is incorrectly generalized to the whole brain.

**References**: [github.com/FlyBase/drosophila-anatomy-developmental-ontology](https://github.com/FlyBase/drosophila-anatomy-developmental-ontology) · [obofoundry.org/ontology/fbbt](https://obofoundry.org/ontology/fbbt)

---

### FlyBase Gene Ontology Integration (FBgn + GO)

| Attribute | Value |
|---|---|
| **Purpose** | FlyBase gene/allele ontology links (`FBgn_*` IDs) and GO term associations for molecular function and expression annotation; used in VFB for linking anatomy terms to gene expression data |
| **License** | CC BY 4.0 |
| **Maturity** | High — FlyBase is a core model-organism database; data updated with each Drosophila genome release |
| **Key integration surface** | FlyBase REST API (`api.flybase.org`); VFB `search_terms` with `filter_types=["gene"]`; cross-links anatomy terms to expression data |

**Loci integration points**:
1. **Gene-expression provenance**: neurotransmitter predictions from `get_predicted_neurotransmitters` reference GO secretion terms (same ID space as known NT data); these GO IDs can anchor gene-level findings in memory entries for cross-evidence reasoning.
2. **Expression-anatomy link chains**: FBgn → GO → FBbt triples can be stored as causal edges (`causal_edges_list`) to support "what gene is expressed in this circuit?" queries.

**References**: [api.flybase.org](https://api.flybase.org/) · [flybase.org](https://flybase.org/)

---

## Ranked Shortlist — Adoption Recommendations

| Rank | Tool / Project | Why adopt now | Key risk |
|---|---|---|---|
| **1** | **VFB MCP** (`virtual-fly-brain-*`) | Already wired into this session; covers all 8 active datasets; ontology-grounded; directly feeds provenance envelope required by `FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md`; no new dependencies needed | VFB REST tier can return stale cache (`force_refresh` caveat); tool surface may lag dataset updates by weeks |
| **2** | **neuprint-python** | Fills the ROI-level synapse-count and pathfinding gaps not exposed through VFB MCP; BSD license; well-documented; trivially installable | Requires neuPrint server authentication token; hemibrain/MANC-only scope — does not cover FlyWire |
| **3** | **navis** | Best-in-class for morphology comparison (NBLAST), skeleton I/O, and template-space transforms; integrates with both neuPrint and CATMAID surfaces; GPU-accelerated via `navis-fastcore` | GPL-3.0 — copyleft; requires explicit dependency decision; heavier install than pure-data tools |
| **4** | **fafbseg** | Only tool that natively resolves FlyWire materialization versions (maps to `flywire783` provenance signal); coordinate-to-segment resolution fills an upstream gap | GPL-3.0; depends on Chunkedgraph server that may require FlyWire token for some operations |
| **5** | **Brian2 + eonsystemspbc/fly-brain** | Enables simulation-derived hypothesis generation directly from FlyWire adjacency data; Brian2 is stable/well-documented; spiking output directly testable against connectivity claims | eonsystemspbc simulator is not yet versioned; simulation is compute-intensive and not reproducible without pinned connectome snapshots — requires strict provenance handling |

---

## Gaps and observations

- **No Python tool yet covers BANC v888 at the same maturity as FlyWire / Hemibrain tools.** BANC data is available through VFB MCP, but a dedicated Python client is absent; direct REST access to the BANC Chunkedgraph server is possible but undocumented here.
- **Annotation roundtrip is not yet formalized.** The Loci store can hold VFB-derived annotations, but there is no automated sync path from Loci memory back to CATMAID or FlyWire annotation layers.
- **Simulation provenance is the weakest link.** Brian2 and the eonsystemspbc simulator do not natively emit structured provenance. Any simulation result stored in Loci must carry an explicit parameter manifest written by the calling code.
- **cocoa / coconatfly are experimental** — use for exploratory cross-dataset matching only; do not rely on API stability for production memory entries.

---

## Integration checklist for new fly-brain PRs

For every new data pipeline or simulation hook that touches this tooling:

- [ ] Record dataset symbol + version-bearing identifier (see `FLYBRAIN_DATASET_PROVENANCE_MATRIX.md`).
- [ ] Persist `exclude_dbs`, query mode flags, and result contract for any `query_connectivity` / `run_query` call.
- [ ] Tag claims with sex/stage/anatomy scope before writing to `investigation_store`.
- [ ] Store Neuroglancer or VFB permalink as `source_url` field when a visual reference is available.
- [ ] For simulation-derived findings, embed the full parameter manifest in the finding's `evidence` field.
- [ ] For ontology-derived claims, store the FBbt ID and ontology version alongside the claim.

---

*Last updated: 2026-09-22. Evidence sources: `virtual-fly-brain-list_connectome_datasets` (live), web search (2024–2026 sources cited inline), existing repo docs.*
