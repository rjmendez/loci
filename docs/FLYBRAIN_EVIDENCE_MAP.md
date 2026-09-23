# FlyBrain evidence map

This is the concise traceability map for the short FlyBrain docs in this repo. It links the central claims to the supporting implementation docs, dataset-boundary notes, and tool surfaces that back them up, while calling out where the evidence is partial or intentionally scoped.

## Evidence status legend

- Direct tool surface: the active VFB/MCP tool surface and immediate dataset metadata used in the current repo session.
- Repo-backed design note: implementation or guidance docs in this repo that define the intended workflow or boundary conditions.
- External reference: published project documentation or papers that corroborate the broader tool or dataset context.
- Partial / scope-limited: a statement is valid only within a specific dataset, stage, anatomy, or query path and should not be treated as universal biology without further evidence.

## Key claim -> supporting evidence map

| Claim / doc point | Supporting evidence | Evidence status | Wording guard |
|---|---|---|---|
| FlyBrain claims are only meaningful when dataset, stage, anatomy, and query path are explicit. | [FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md](./FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md) defines the required provenance envelope; [FLYBRAIN_DATASET_PROVENANCE_MATRIX.md](./FLYBRAIN_DATASET_PROVENANCE_MATRIX.md) provides the dataset boundary map; [FLYBRAIN_CROSS_DATASET_LESSONS.md](./FLYBRAIN_CROSS_DATASET_LESSONS.md) explains why cross-dataset generalization fails without scope checks. | Repo-backed design note | Keep language dataset-scoped unless the evidence spans the claimed scope directly. |
| Loci should carry provenance with each FlyBrain-derived claim rather than flattening it into a vague fact. | [FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md](./FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md) includes the minimum metadata required, plus example JSON and claim tiers. [REASONING_LOOP_PROVENANCE.md](./REASONING_LOOP_PROVENANCE.md) reinforces that claims must stay traceable to evidence and contradiction history. | Repo-backed design note | Do not treat a claim as general if the provenance envelope is missing or incomplete. |
| Hemibrain (`hb`) and FlyWire (`fw`) differ in coverage, annotation model, versioning, error profile, and query ergonomics; cross-dataset claims require explicit matching. | [FLYBRAIN_HEMIBRAIN_VS_FLYWIRE.md](./FLYBRAIN_HEMIBRAIN_VS_FLYWIRE.md) provides the full comparison matrix, scope guardrails, and Loci integration guidance. Grounded in Scheffer et al. *eLife* 2020, Dorkenwald/Schlegel et al. *Nature* 2024, CAVE (*Nature Methods* 2024), and `virtual-fly-brain-list_connectome_datasets` tool output. | Direct tool surface + external reference | Gate cross-dataset claims (body-ID equivalence, synapse-count comparison, type-label matching) per the guardrails in that doc. |
| The workflow is ontology lookup -> scope check -> evidence comparison -> provenance capture. | The active VFB surface in this session exposes `search_terms`, `get_term_info`, `get_hierarchy`, `query_connectivity`, `run_query`, and dataset metadata via `list_connectome_datasets`; this is summarized in [FLYBRAIN_OPEN_SOURCE_TOOLING_MAP.md](./FLYBRAIN_OPEN_SOURCE_TOOLING_MAP.md). | Direct tool surface + repo-backed summary | This is a workflow pattern, not a claim that every query result is universally valid. |
| FlyBrain data access is built on a multi-dataset, ontology-grounded stack. | [FLYBRAIN_OPEN_SOURCE_TOOLING_MAP.md](./FLYBRAIN_OPEN_SOURCE_TOOLING_MAP.md) enumerates VFB, neuPrint, fafbseg, navis, CATMAID/pymaid, and the ontology layers. The active session also contains the live `virtual-fly-brain-*` tool surface. | Direct tool surface + external reference | The list is a tooling map, not a guarantee that each project is equally suitable for a given Loci feature. |
| The architecture mapping from FlyBrain organization to Loci structure is conceptual, not biological proof. | [FLYBRAIN_IO_TO_LOCI_MAPPING.md](./FLYBRAIN_IO_TO_LOCI_MAPPING.md) explicitly frames itself as a translation of structure and function into Loci design patterns. This should be read as an engineering analogy. | Repo-backed design note | Use it for implementation scaffolding; do not present it as a direct empirical validation of circuit biology. |
| Comparative cross-species material is useful for scope guardrails, not generic biological universalization. | [FLYBRAIN_EVOLUTIONARY_COMPARATIVE.md](./FLYBRAIN_EVOLUTIONARY_COMPARATIVE.md) classifies mechanisms as conserved, fly-specific, or context-bound and references the design implications for Loci. | Repo-backed design note + external reference | Use the comparative material to decide whether a mechanism is safe to generalize, rather than assuming a fly-specific pattern is universal. |
| Open-source FlyBrain tooling is mature but its coverage and freshness vary by dataset and query mode. | [FLYBRAIN_OPEN_SOURCE_TOOLING_MAP.md](./FLYBRAIN_OPEN_SOURCE_TOOLING_MAP.md) includes dataset coverage, maturity, and adoption notes; [FLYBRAIN_DATASET_PROVENANCE_MATRIX.md](./FLYBRAIN_DATASET_PROVENANCE_MATRIX.md) keeps the dataset boundaries explicit. | Repo-backed design note + external reference | Avoid broad statements like "this result holds across all FlyBrain data" unless the dataset + query scope is directly matched. |

| Neuromodulatory identity claims (dopamine / serotonin / octopamine) must carry dataset symbol, version, evidence family (curated vs predicted), and cross-dataset agreement before storage or promotion. | [FLYBRAIN_NEUROMODULATORY_COMPARISON.md](./FLYBRAIN_NEUROMODULATORY_COMPARISON.md) provides the full cross-dataset comparison, discordance inventory, and per-system robustness assessment. The comparison is backed by live `virtual-fly-brain-get_known_neurotransmitters` and `virtual-fly-brain-get_predicted_neurotransmitters` calls with `split_by_dataset=true` and all datasets included. | Direct tool surface + repo-backed design note | Do not treat a neuromodulatory identity claim as high-confidence without specifying the evidence family (curated vs predicted), the dataset(s), and whether cross-dataset agreement exists. Specific flagged cases (AL2b2, PMPD, PLP) must not be promoted without human review. |
| Cross-dataset discordances in neuromodulatory identity are real, dataset-local, and not resolvable by averaging. | [FLYBRAIN_NEUROMODULATORY_COMPARISON.md](./FLYBRAIN_NEUROMODULATORY_COMPARISON.md) §Cross-system comparison: discordance types observed. Key cases: serotonergic PMPD (dopamine in flywire783 / serotonin in male_cns), octopaminergic AL2b2 (three-dataset acetylcholine at 0.86–0.92 confidence vs ontology label), serotonergic PLP (two-dataset glutamate prediction). | Direct tool surface | These are not query errors. They signal that sex, anatomy, or classification boundaries must be part of the claim envelope. |
| The three neuromodulatory systems (dopaminergic, serotonergic, octopaminergic) differ in population scale, circuit diffuseness, and cross-dataset evidence richness; their robustness rankings differ accordingly. | [FLYBRAIN_NEUROMODULATORY_COMPARISON.md](./FLYBRAIN_NEUROMODULATORY_COMPARISON.md) §Cross-system comparison. Dopaminergic: 17,488 instances, 5-dataset agreement, highest robustness. Serotonergic: 2,188 instances, 4-dataset agreement for SE population, moderate subclass uncertainty. Octopaminergic: 377 instances, 3–4 dataset agreement for VUM family, notable AL2b2 anomaly. | Direct tool surface | Treat octopaminergic subclass predictions with more caution than dopaminergic ones, proportional to the smaller evidence base. |

## Evidence sources worth checking when a claim is challenged

1. [FLYBRAIN_GUIDE.md](./FLYBRAIN_GUIDE.md) — the short, canonical framing for the problem statement.
2. [FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md](./FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md) — the minimum replayable provenance requirements.
3. [FLYBRAIN_DATASET_PROVENANCE_MATRIX.md](./FLYBRAIN_DATASET_PROVENANCE_MATRIX.md) — the dataset-level scope boundaries.
4. [FLYBRAIN_NEUROMODULATORY_COMPARISON.md](./FLYBRAIN_NEUROMODULATORY_COMPARISON.md) — cross-dataset evidence comparison for dopaminergic, serotonergic, and octopaminergic circuits; includes discordance inventory and Loci routing implications.
5. [FLYBRAIN_HEMIBRAIN_VS_FLYWIRE.md](./FLYBRAIN_HEMIBRAIN_VS_FLYWIRE.md) — detailed hemibrain vs. FlyWire comparison matrix with cross-dataset guardrails.
6. [FLYBRAIN_OPEN_SOURCE_TOOLING_MAP.md](./FLYBRAIN_OPEN_SOURCE_TOOLING_MAP.md) — the tool landscape and adoption notes.
7. [FLYBRAIN_IO_TO_LOCI_MAPPING.md](./FLYBRAIN_IO_TO_LOCI_MAPPING.md) — the implementation analogy from anatomy to Loci architecture.
8. The active VFB tool surface (`search_terms`, `get_term_info`, `get_hierarchy`, `query_connectivity`, `run_query`, `list_connectome_datasets`, `get_known_neurotransmitters`, `get_predicted_neurotransmitters`) — the concrete live evidence path for a claim.

## Partial-evidence / scope-limited claims to keep explicit

- Dataset-specific results should be phrased as "in this dataset/query path" unless matched elsewhere.
- Cross-dataset comparisons are strongest only when the relevant sex, stage, anatomy, and query semantics align.
- Architecture mappings are design aids; they are not direct evidence that a biological circuit works as described.
- Tooling coverage does not guarantee semantic equivalence across datasets or query types.

## Reading pattern

When a FlyBrain claim needs to be checked, start with the guide, then read the provenance guidance and the dataset matrix before relying on a broader statement. If a design or tool claim is being asserted, confirm it against the specific tool or mapping doc rather than the short summary alone.
