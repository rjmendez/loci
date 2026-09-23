# FlyBrain Archive and Appendix Index

This is the de-emphasized archive index. It keeps the dense FlyBrain notes discoverable without making them the normal reading path.

## Active canonical docs

These are the docs to read first for normal work. They are intentionally short and operational.

- [FLYBRAIN_GUIDE.md](./FLYBRAIN_GUIDE.md) — the main entry point for FlyBrain in Loci.
- [FLYBRAIN_EVIDENCE_MAP.md](./FLYBRAIN_EVIDENCE_MAP.md) — claim-to-evidence traceability map across the FlyBrain docs, tool surface, and provenance notes.
- [FLYBRAIN_REASONING_GLOSSARY.md](./FLYBRAIN_REASONING_GLOSSARY.md) — short term lookup for dataset, provenance, and reasoning terminology.
- [FLYBRAIN_HEMIBRAIN_VS_FLYWIRE.md](./FLYBRAIN_HEMIBRAIN_VS_FLYWIRE.md) — rigorous hemibrain v1.2.1 vs. FlyWire v783 comparison: coverage, annotation model, versioning, query ergonomics, error modes, and cross-dataset guardrails.
- [FLYBRAIN_PRUNE_PLAN.md](./FLYBRAIN_PRUNE_PLAN.md) — current retention and redirect policy for the FlyBrain doc set.

## Redirect map (legacy -> canonical)

This is the live redirect table for FlyBrain doc entry points. Dense pages should resolve here when they are summarized or retired, so readers always land on the short canonical path instead of a dead end.

| Legacy path | Redirect target | Notes |
|---|---|---|
| `FLYBRAIN_CROSS_DATASET_LESSONS.md` | [FLYBRAIN_GUIDE.md](./FLYBRAIN_GUIDE.md), [FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md](./FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md) | Keep the conceptual summary in the guide; use provenance guidance for the evidence rules. |
| `FLYBRAIN_RESEARCH_PRIORITIES.md` | [FLYBRAIN_GUIDE.md](./FLYBRAIN_GUIDE.md), [FLYBRAIN_ARCHIVE.md](./FLYBRAIN_ARCHIVE.md) | Planning doc; canonical entry next and deeper archive context. |
| `FLYBRAIN_IO_TO_LOCI_MAPPING.md` | [FLYBRAIN_GUIDE.md](./FLYBRAIN_GUIDE.md), [ARCHITECTURE.md](./ARCHITECTURE.md) | Architecture reference should resolve to the short guide plus the implementation mapping. |
| `FLYBRAIN_INSPIRATION_MAP.md` | [FLYBRAIN_GUIDE.md](./FLYBRAIN_GUIDE.md), [FLYBRAIN_ARCHIVE.md](./FLYBRAIN_ARCHIVE.md) | Concept-to-subsystem map retained as a deep reference rather than a front-door page. |
| `FLYBRAIN_OPEN_SOURCE_TOOLING_MAP.md` | [FLYBRAIN_GUIDE.md](./FLYBRAIN_GUIDE.md), [FLYBRAIN_ARCHIVE.md](./FLYBRAIN_ARCHIVE.md) | Tooling guidance remains archival; the guide is the reader-facing summary. |
| `FLYBRAIN_EVOLUTIONARY_COMPARATIVE.md` | [FLYBRAIN_GUIDE.md](./FLYBRAIN_GUIDE.md), [FLYBRAIN_CROSS_DATASET_LESSONS.md](./FLYBRAIN_CROSS_DATASET_LESSONS.md) | Comparative material is specialized; route through the guide and the lessons summary. |

## Final retired state

The dense FlyBrain pages remain available for historical continuity, but they are intentionally retired from the active navigation path. Use the short canonical docs above first; only follow the archived deep notes when you need historical context or implementation detail.

The redirect ledger is the single source of truth for these retired entry points:

- [FLYBRAIN_CROSS_DATASET_LESSONS.md](./FLYBRAIN_CROSS_DATASET_LESSONS.md) → [FLYBRAIN_GUIDE.md](./FLYBRAIN_GUIDE.md) and [FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md](./FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md)
- [FLYBRAIN_RESEARCH_PRIORITIES.md](./FLYBRAIN_RESEARCH_PRIORITIES.md) → [FLYBRAIN_GUIDE.md](./FLYBRAIN_GUIDE.md) and [FLYBRAIN_ARCHIVE.md](./FLYBRAIN_ARCHIVE.md)
- [FLYBRAIN_IO_TO_LOCI_MAPPING.md](./FLYBRAIN_IO_TO_LOCI_MAPPING.md) → [FLYBRAIN_GUIDE.md](./FLYBRAIN_GUIDE.md) and [ARCHITECTURE.md](./ARCHITECTURE.md)
- [FLYBRAIN_INSPIRATION_MAP.md](./FLYBRAIN_INSPIRATION_MAP.md) → [FLYBRAIN_GUIDE.md](./FLYBRAIN_GUIDE.md) and [FLYBRAIN_ARCHIVE.md](./FLYBRAIN_ARCHIVE.md)
- [FLYBRAIN_OPEN_SOURCE_TOOLING_MAP.md](./FLYBRAIN_OPEN_SOURCE_TOOLING_MAP.md) → [FLYBRAIN_GUIDE.md](./FLYBRAIN_GUIDE.md) and [FLYBRAIN_ARCHIVE.md](./FLYBRAIN_ARCHIVE.md)
- [FLYBRAIN_EVOLUTIONARY_COMPARATIVE.md](./FLYBRAIN_EVOLUTIONARY_COMPARATIVE.md) → [FLYBRAIN_GUIDE.md](./FLYBRAIN_GUIDE.md) and [FLYBRAIN_CROSS_DATASET_LESSONS.md](./FLYBRAIN_CROSS_DATASET_LESSONS.md)

## Planning and active follow-up

- [FLYBRAIN_RESEARCH_PRIORITIES.md](./FLYBRAIN_RESEARCH_PRIORITIES.md) — ranked research paths, execution order, and Loci subsystem hooks. Useful for planning, but not the default reader path.

## Archive / dense or retired references

These files are still valid references, but they sit behind the canonical docs because they are deeper, more specialized, or better treated as appendix material.

- [FLYBRAIN_CROSS_DATASET_LESSONS.md](./FLYBRAIN_CROSS_DATASET_LESSONS.md) — cross-dataset lessons and scoping warnings.
- [FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md](./FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md) — provenance, replay, and dataset-boundary rules for FlyBrain-derived claims.
- [FLYBRAIN_DATASET_PROVENANCE_MATRIX.md](./FLYBRAIN_DATASET_PROVENANCE_MATRIX.md) — dataset-by-dataset scope and allowed use.
- [FLYBRAIN_EVOLUTIONARY_COMPARATIVE.md](./FLYBRAIN_EVOLUTIONARY_COMPARATIVE.md) — comparative conservation vs. fly-specific design patterns.
- [FLYBRAIN_INSPIRATION_MAP.md](./FLYBRAIN_INSPIRATION_MAP.md) — concept-to-subsystem placement guide for new FlyBrain-inspired ideas.
- [FLYBRAIN_IO_TO_LOCI_MAPPING.md](./FLYBRAIN_IO_TO_LOCI_MAPPING.md) — architecture translation from fly-brain organization to Loci implementation patterns.
- [FLYBRAIN_OPEN_SOURCE_TOOLING_MAP.md](./FLYBRAIN_OPEN_SOURCE_TOOLING_MAP.md) — open-source tooling map and adoption guidance.

## Reading guidance

Use the canonical docs first. Use the archive only when you need implementation detail, historical context, a dataset boundary check, or a specialized comparative note. The archive is not the live source of truth for a current tool result: dataset availability, `count_status`, and cached VFB responses still need to be checked against the active tool output and the dataset matrix.

The current retention policy is tracked in [FLYBRAIN_PRUNE_PLAN.md](./FLYBRAIN_PRUNE_PLAN.md) and the repo-wide [DOCS_RETENTION_POLICY.md](./DOCS_RETENTION_POLICY.md): keep the shallow path short, keep dense material available, and redirect any retired page back to the canonical docs instead of leaving dead ends.
