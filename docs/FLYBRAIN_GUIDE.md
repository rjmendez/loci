# FlyBrain Guide for Loci

This is the short, user-facing version of the FlyBrain material in this repo. If you only read one page, read this one first, then use the provenance, dataset, and mapping docs below as follow-up references.

## Why this matters

FlyBrain is the practical test for one of Loci's core promises: when evidence comes from different datasets, stages, or interpretations, the system should keep each claim scoped instead of flattening it into a vague "fact."

This matters for ordinary questions such as:

- Is this connectivity pattern real in the dataset I am using, or only in a different one?
- Can I safely generalize from adult female data to male or larval data?
- Which result should I trust when connectivity, ontology, and predictions disagree?

The active VFB tool surface in this repo supports this workflow pattern: `search_terms` finds the relevant neuron or anatomy, `get_term_info` and `get_hierarchy` confirm scope, `query_connectivity` and `run_query` compare evidence, and [FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md](./FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md) captures the dataset and version trail. For dataset boundaries, see [FLYBRAIN_DATASET_PROVENANCE_MATRIX.md](./FLYBRAIN_DATASET_PROVENANCE_MATRIX.md); for the design mapping, see [FLYBRAIN_IO_TO_LOCI_MAPPING.md](./FLYBRAIN_IO_TO_LOCI_MAPPING.md). Where the evidence is partial or scope-limited, the docs keep that boundary explicit instead of treating the result as universal.

This is not a shortcut to a generic biology answer. It is a boundary check: the value is in keeping claims traceable, useful, and honest about what the evidence does and does not support.

## User story: when FlyBrain matters in Loci

Use FlyBrain when the question is not just "what is the circuit?" but "what is the evidence, and how far can I trust it?" That is the real user outcome: you can compare neuron classes or connectivity patterns without accidentally turning a dataset-local result into a general fact.

Typical Loci use cases:

- You need to decide whether a connectivity pattern holds across adult female, male, or larval data.
- You want to check whether a claim stays valid after a different dataset or query path is used.
- You are evaluating whether an anatomical interpretation is safe to generalize or should stay scoped to one region, stage, or dataset.
- You need a clean evidence trail before storing a claim in memory or answering a user with confidence.

A practical workflow is straightforward:

1. Start with `search_terms` to find the neuron class or anatomy.
2. Use `get_term_info` or `get_hierarchy` to confirm the entity and its scope boundary.
3. Compare evidence with `query_connectivity` or `run_query` across the relevant datasets.
4. Store the result with the provenance envelope from `FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md` so later checks can replay the exact query context.

Important caveat: VFB/FlyBrain queries are part of an external tool surface and are not interchangeable across datasets or query modes. Empty results can mean "no result in this query," not "no biology exists"; `count_status` warnings, dataset exclusions, and stale cached responses are part of the evidence. The safe default is to scope the claim to the dataset, version, sex/stage/anatomy boundary, and query semantics actually used.

This is the difference between a vague biological statement and a claim Loci can defend: the result is tied to its dataset, version, anatomy scope, and query semantics instead of being treated as universally true.

## The core message

The most important idea is not a biological fact. It is a method rule:

- Different FlyBrain datasets are not interchangeable.
- A claim is only as good as its dataset, version, sex, stage, anatomy, and query path.
- Loci should store provenance with each claim, not just the conclusion.

That rule is the reason the follow-up FlyBrain notes exist.

## Active follow-up docs

The live, active reading path stays short. Use these first; deeper historical material lives in [FLYBRAIN_ARCHIVE.md](./FLYBRAIN_ARCHIVE.md).

| Document | Core user-facing message | Best for |
|---|---|---|
| [FLYBRAIN_NEUROMODULATORY_COMPARISON.md](./FLYBRAIN_NEUROMODULATORY_COMPARISON.md) | Evidence-backed cross-dataset comparison of dopaminergic, serotonergic, and octopaminergic circuits, with confidence/scope notes, discordance inventory, and Loci routing implications. | Neuromodulatory claim evaluation, routing/confidence design. |
| [FLYBRAIN_EVIDENCE_MAP.md](./FLYBRAIN_EVIDENCE_MAP.md) | Claim-to-evidence traceability map across the FlyBrain docs and tool surface. | Checking what supports a specific FlyBrain claim. |
| [FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md](./FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md) | Every FlyBrain-derived claim should carry provenance: dataset, version, query settings, and result contract. | Provenance and replay requirements. |
| [FLYBRAIN_DATASET_PROVENANCE_MATRIX.md](./FLYBRAIN_DATASET_PROVENANCE_MATRIX.md) | Dataset-by-dataset scope map for sex, stage, and anatomy coverage. | Checking what a dataset can and cannot support. |
| [FLYBRAIN_HEMIBRAIN_VS_FLYWIRE.md](./FLYBRAIN_HEMIBRAIN_VS_FLYWIRE.md) | Rigorous comparison of hemibrain v1.2.1 (`hb`) and FlyWire v783 (`fw`): coverage, annotation model, versioning, query ergonomics, error modes, and cross-dataset guardrails. | Deciding between hemibrain and FlyWire; gating cross-dataset claims. |
| [FLYBRAIN_REASONING_GLOSSARY.md](./FLYBRAIN_REASONING_GLOSSARY.md) | Reference glossary for the FlyBrain and Loci terminology used across the evidence docs. | Terminology lookup while reading deeper sources. |
| [FLYBRAIN_ARCHIVE.md](./FLYBRAIN_ARCHIVE.md) | Redirect index and archive home for the de-emphasized deep notes. | Reaching the retained historical material without dead ends. |

## Evidence map and traceability

For a concise claim-by-claim map from the short FlyBrain docs to the supporting source material, see [FLYBRAIN_EVIDENCE_MAP.md](./FLYBRAIN_EVIDENCE_MAP.md). The map links each central claim to the repo docs, tool surface, and dataset-boundary notes that support it, and it marks areas where the evidence is partial or intentionally scoped.

## Doc maintenance plan

The FlyBrain doc set is intentionally layered: short, canonical docs for normal readers and deeper appendix material for technical follow-up. The repo's current retention policy is documented in [FLYBRAIN_PRUNE_PLAN.md](./FLYBRAIN_PRUNE_PLAN.md): keep the entry points short, summarize the long-lived design notes, and only retire a deep file after it is replaced with an explicit redirect or a summary pointer.

This keeps navigation coherent without deleting history or leaving orphan pages behind.

## Redirect map

Legacy FlyBrain entry points should resolve to the short canonical docs above. The live redirection ledger lives in [FLYBRAIN_ARCHIVE.md](./FLYBRAIN_ARCHIVE.md), which maps older dense pages to their current canonical destinations without leaving dead ends or orphaned pages.

## Read in this order

1. Start here: this page.
2. Keep [FLYBRAIN_REASONING_GLOSSARY.md](./FLYBRAIN_REASONING_GLOSSARY.md) open for quick term lookup while reading the deeper docs.
3. If you need the evidence and scope rules, read [FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md](./FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md).
4. If you need the dataset boundaries, read [FLYBRAIN_DATASET_PROVENANCE_MATRIX.md](./FLYBRAIN_DATASET_PROVENANCE_MATRIX.md).
5. If you are comparing hemibrain (`hb`) and FlyWire (`fw`) specifically — coverage, annotation model, error modes, query ergonomics, or cross-dataset guardrails — read [FLYBRAIN_HEMIBRAIN_VS_FLYWIRE.md](./FLYBRAIN_HEMIBRAIN_VS_FLYWIRE.md).
6. If you want the claim-to-evidence map, read [FLYBRAIN_EVIDENCE_MAP.md](./FLYBRAIN_EVIDENCE_MAP.md).
7. If you need deeper historical or comparative context, start at [FLYBRAIN_ARCHIVE.md](./FLYBRAIN_ARCHIVE.md). The archive contains the dense FlyBrain notes that were intentionally retired from the active path.

## Why it matters for Loci

The FlyBrain material is proving ground for a few Loci habits we want to keep:

- evidence-bounded claims;
- explicit dataset/version scope;
- typed memory with provenance;
- routing that responds to context and confidence rather than raw text alone.

The most important Loci takeaway is: a finding is only trustworthy when its evidence path is still traceable.

## Archive and appendix material

The deeper FlyBrain notes are kept as appendix/archive material for historical and technical reference. They are not the starting point for normal reading.

See [FLYBRAIN_ARCHIVE.md](./FLYBRAIN_ARCHIVE.md) for the archive index and pointers to the retained deep notes.
