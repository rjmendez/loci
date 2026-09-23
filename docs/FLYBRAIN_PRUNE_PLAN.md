# FlyBrain Doc Prune and Redirect Plan

This plan keeps the FlyBrain documentation set readable without destroying historical material. The canonical docs stay short and discoverable; the dense notes stay available as archive material or become summarized pointers when they are no longer primary reading paths. The repo-wide policy lives in [DOCS_RETENTION_POLICY.md](./DOCS_RETENTION_POLICY.md); this file applies the same rules to the FlyBrain slice.

## Principles

- Keep the canonical path shallow: `FLYBRAIN_GUIDE.md` remains the main landing page; `FLYBRAIN_ARCHIVE.md` remains the appendix index.
- Keep evidence-heavy content, but lower its default prominence. Dense pages are technical reference, not the default entry point.
- Summarize before retiring. A document should not disappear without a redirect target or a replacement summary.
- Preserve historical links and avoid orphan docs. Any future retirement must be paired with a redirect pointer from the old file and a canonical target in the active docs set.

## Decision matrix

| File | Decision | Rationale | Navigation target |
|---|---|---|---|
| `docs/FLYBRAIN_GUIDE.md` | Keep canonical | This is the clear user-facing entry point for FlyBrain in Loci, with the evidence map linked from the canonical path. | Main landing page for FlyBrain readers. |
| `docs/FLYBRAIN_EVIDENCE_MAP.md` | Keep canonical | This provides a concise claim-to-evidence traceability map without adding noise to the main guide. | Linked from the guide and archive index. |
| `docs/FLYBRAIN_ARCHIVE.md` | Keep archive index | It already acts as the explicit appendix. This should remain the stable place for deep reading. | Archive index. |
| `docs/FLYBRAIN_REASONING_GLOSSARY.md` | Keep | Short, reference-friendly glossary that makes the terminology consistent across Loci and FlyBrain docs. | Linked directly from the guide and the provenance guidance. |
| `docs/FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md` | Keep | Core evidence-scoping and replay rules; this is a working artifact, not background fluff. | Treated as required reading after the guide. |
| `docs/FLYBRAIN_DATASET_PROVENANCE_MATRIX.md` | Keep | Dataset scope is essential to avoid cross-dataset overgeneralization. | Treated as the boundary-check reference. |
| `docs/FLYBRAIN_CROSS_DATASET_LESSONS.md` | Summarize and keep in archive | The lessons are important, but they are already represented in the guide and provenance material. Full detail should stay in the archive, not in the higher-traffic path. | Redirect to guide + archive. |
| `docs/FLYBRAIN_RESEARCH_PRIORITIES.md` | Summarize and keep as archive reference | Valuable backlog and scoring context, but it is not a general user guide and is too dense for primary navigation. Keep the full ranking in the archive and mirror the key priorities in the guide. | Redirect to guide summary and archive. |
| `docs/FLYBRAIN_IO_TO_LOCI_MAPPING.md` | Summarize and keep as archive reference | This is the architecture translation, which is important but should not compete with the short conceptual overview. | Redirect to guide + archive reference. |
| `docs/FLYBRAIN_INSPIRATION_MAP.md` | Summarize and keep as archive reference | Good concept-to-subsystem placement material, but it is easier to consume as a narrow reference once the architecture doc is known. | Redirect to guide + archive. |
| `docs/FLYBRAIN_OPEN_SOURCE_TOOLING_MAP.md` | Summarize and keep as archive reference | It is useful as a capability map, but the long table is noisy for casual readers and should not be a default landing page. | Redirect to guide + archive. |
| `docs/FLYBRAIN_EVOLUTIONARY_COMPARATIVE.md` | Keep in archive; summarize only if needed | This is a specialist comparative document with value for scope guards, but it is clearly not a core reader path. | Redirect to archive and the cross-dataset guidance when the comparative question arises. |

## Retire / redirect plan

No file is deleted by this plan. The goal is to make the high-traffic path shorter and to ensure that any future retirement is explicit and safe.

| Current file | Action when retired | Redirect target | Notes |
|---|---|---|---|
| `docs/FLYBRAIN_RESEARCH_PRIORITIES.md` | Replace with a short summary pointer | `docs/FLYBRAIN_GUIDE.md` plus `docs/FLYBRAIN_ARCHIVE.md` | Full ranking remains in the archive until a shorter summary is introduced. |
| `docs/FLYBRAIN_IO_TO_LOCI_MAPPING.md` | Replace with a short architecture pointer | `docs/FLYBRAIN_GUIDE.md` and `docs/ARCHITECTURE.md` | This should never be orphaned; keep link from guide to deep mapping. |
| `docs/FLYBRAIN_INSPIRATION_MAP.md` | Replace with a short concept pointer | `docs/FLYBRAIN_GUIDE.md` and `docs/FLYBRAIN_ARCHIVE.md` | Good candidate for a small summary card rather than full prose. |
| `docs/FLYBRAIN_OPEN_SOURCE_TOOLING_MAP.md` | Replace with a curated tool shortlist pointer | `docs/FLYBRAIN_GUIDE.md` and archive quick links | Keep a small shortlist of the most relevant tools; move the long-form map to archive. |
| `docs/FLYBRAIN_EVOLUTIONARY_COMPARATIVE.md` | Replace with a compact comparative note | `docs/FLYBRAIN_GUIDE.md` and `docs/FLYBRAIN_CROSS_DATASET_LESSONS.md` | Only retire if a more concise summary replaces it. |
| `docs/FLYBRAIN_CROSS_DATASET_LESSONS.md` | Replace with a short lesson summary | `docs/FLYBRAIN_GUIDE.md` and `docs/FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md` | Keep the deep historical context in archive if needed. |

## Redirect map

This table is the concrete redirect ledger for the FlyBrain docs. It is intentionally small, explicit, and aligned to the current prune plan so each dense page has a stable target in the canonical path.

| Legacy path | Redirect target | Canonical destination |
|---|---|---|
| `docs/FLYBRAIN_CROSS_DATASET_LESSONS.md` | `docs/FLYBRAIN_GUIDE.md` and `docs/FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md` | Short guide + evidence rules |
| `docs/FLYBRAIN_RESEARCH_PRIORITIES.md` | `docs/FLYBRAIN_GUIDE.md` and `docs/FLYBRAIN_ARCHIVE.md` | Guide entry + archive planning context |
| `docs/FLYBRAIN_IO_TO_LOCI_MAPPING.md` | `docs/FLYBRAIN_GUIDE.md` and `docs/ARCHITECTURE.md` | Guide overview + implementation mapping |
| `docs/FLYBRAIN_INSPIRATION_MAP.md` | `docs/FLYBRAIN_GUIDE.md` and `docs/FLYBRAIN_ARCHIVE.md` | Guide overview + archive reference |
| `docs/FLYBRAIN_OPEN_SOURCE_TOOLING_MAP.md` | `docs/FLYBRAIN_GUIDE.md` and `docs/FLYBRAIN_ARCHIVE.md` | Guide overview + archive tool shortlist |
| `docs/FLYBRAIN_EVOLUTIONARY_COMPARATIVE.md` | `docs/FLYBRAIN_GUIDE.md` and `docs/FLYBRAIN_CROSS_DATASET_LESSONS.md` | Guide overview + comparative summary |

## Downstream todo mapping

These are the concrete follow-up tasks that should be scheduled after this plan is accepted:

- `docs-flybrain-prune-summary-guide`: add a one-page summary section in `docs/FLYBRAIN_GUIDE.md` that mirrors the top-level priorities and keeps the canonical path short.
- `docs-flybrain-evidence-map`: maintain a concise claim-to-evidence map in `docs/FLYBRAIN_EVIDENCE_MAP.md` and keep it linked from the guide and archive.
- `docs-flybrain-archive-redirect-map`: maintain the redirect targets in `docs/FLYBRAIN_ARCHIVE.md` whenever a dense doc is summarized or retired.
- `docs-flybrain-research-priorities-summarize`: collapse `FLYBRAIN_RESEARCH_PRIORITIES.md` into a short executive summary and keep the full ranking in archive.
- `docs-flybrain-io-map-summarize`: reduce `FLYBRAIN_IO_TO_LOCI_MAPPING.md` to a short architecture summary plus archive pointer.
- `docs-flybrain-tooling-summarize`: replace the long open-source tooling map with a curated shortlist and a pointer to the archive version.
- `docs-flybrain-evolutionary-pointers`: add an explicit pointer from the guide to the comparative material instead of leaving it as a stand-alone long page.

## Expected final state

The FlyBrain docs should resolve to these three layers:

1. README / guide: short, canonical, and actionable.
2. Archive index: the home for deep historical and technical notes.
3. Redirect links: explicit pointers so a file can be summarized or retired without leaving dead ends.

This keeps the repo navigable, reduces doc drag, and preserves evidence trails without destructive cleanup.
