# FlyBrain local research reference

This is the index of what exists locally, where it lives, how trustworthy each part is, and how to cite it. Keep it current (see "Refresh" at the end). The plan lives in [FLYBRAIN_ROADMAP.md](FLYBRAIN_ROADMAP.md).

Paths are WSL paths. On Windows, `/mnt/f/.flybrain` is `F:\.flybrain`.

## 1. Datasets (read-only snapshots)

Every snapshot has a self-hashed `manifest/manifest.json` plus a `manifest.sha256` sidecar. Adapters verify both before reading and refuse on mismatch. Never write into `snapshots/`.

| Symbol | Dataset | Pinned version | Local snapshot | Size | Licence | Cite |
|---|---|---|---|---:|---|---|
| `fw` | FlyWire whole-brain connectome | `flywire783` | `/mnt/f/.flybrain/snapshots/fw/flywire783/` | 11.9 GB | CC BY 4.0 per the Zenodo records pulled (registry: **UNREVIEWED**, needs review) | Dorkenwald et al. 2024; Schlegel et al. 2024 (Nature). Exact data DOIs are in the manifest |
| `hb` | Hemibrain | `neuprint_JRC_Hemibrain_1point2point1` | `/mnt/f/.flybrain/snapshots/hb/neuprint_JRC_Hemibrain_1point2point1/` | 0.5 GB | registry: **UNREVIEWED**, needs review | Scheffer et al. 2020 (eLife) |
| `banc` | BANC, brain and nerve cord (female) | `banc_888` | `/mnt/f/.flybrain/snapshots/BANC/banc_888/` | 29.5 GB | CC BY 4.0 | Bates AS et al. (2026). *Distributed control circuits across a brain-and-cord connectome.* Nature. doi:10.1038/s41586-026-10735-w. Data: Harvard Dataverse doi:10.7910/DVN/7WTH1N |
| `mc` | male CNS (brain + VNC) | `male-cns_v1.0` | `/mnt/f/.flybrain/snapshots/mc/male-cns_v1.0/` | 39.5 GB | CC BY 4.0 | Berg S et al. (2026). *Sexual dimorphism in the complete connectome of the Drosophila male central nervous system.* Cell. bioRxiv doi:10.1101/2025.10.09.680999 |
| `mv` | MANC, male VNC | `manc_v1.0` | `/mnt/f/.flybrain/snapshots/mv/manc_v1.0/` | 13.2 GB | CC BY 4.0 | Takemura S et al. (2024). eLife 13:RP97769. doi:10.7554/eLife.97769; Marin EC et al. (2024). eLife 13:RP97766 |
| `ol` | Optic lobe (male) | `optic_lobe_v1.1` | `/mnt/f/.flybrain/snapshots/ol/optic_lobe_v1.1/` | 5.5 GB | CC BY 4.0 | Nern A et al. (2025). *Connectome-driven neural inventory of a complete visual system.* Nature 641:1225–1237. doi:10.1038/s41586-025-08746-0 |
| `l1em` | L1 larval connectome | `catmaid_l1em` (Winding 2023 supplementary) | `/mnt/f/.flybrain/snapshots/l1em/catmaid_l1em/` | 0.2 GB | CC BY 4.0 (Europe PMC author manuscript) | Winding M et al. (2023). *The connectome of an insect brain.* Science. doi:10.1126/science.add9330 (registry citation field: **missing**, to add) |
| `fafb` | FAFB CATMAID tracing | `catmaid_fafb` | metadata only | — | UNREVIEWED | Deferred as largely redundant with FlyWire (`FLYBRAIN_FAFB_LANE_DECISION.md`) |

Pull reports with the file-by-file sha256, licence evidence and what was skipped and why are in `/mnt/f/.flybrain/logs/data-pull-20260924T010254Z/data-pull-report.md`. Adapter contracts are in `docs/FLYBRAIN_<SYMBOL>_ADAPTER_CONTRACT.md`.

**Label provenance warnings** (from the research synthesis, section 2):

- Several annotation columns in these releases are connectivity-defined: optic-lobe types, larval S2 types and clusters, MANC systematic types and hemilineages, male-CNS types, and FlyWire CBLAST types.
- Every dataset's NT column is a classifier output (the Eckstein/synister family). The exceptions are literature-backed fields such as male-CNS type-level `nt_ground_truth` and optic-lobe `consensusNt` with `ntReference`.

## 2. Results and reports

| What | Location | Status | Use it as |
|---|---|---|---|
| Real-model results, 4 finished datasets plus location-free super_class | `docs/FLYBRAIN_REAL_MODELS_INTERIM.md` (PR #393) | **interim, unaudited** | Working numbers only. Do not quote as final |
| Per-lane reports (JSON + markdown: baselines, CIs, ablations, calibration, controls) | `/mnt/f/.flybrain/logs/real-models-20260924T174122Z/<dataset>/<target>/<run>/report.{json,md}` | interim | Primary evidence for each lane |
| Model artifacts (manifested, fail-closed load) | `/mnt/f/.flybrain/cache/models/…`, created at the integrate step | pending | Only lanes that pass the gates get artifacts, at canary level |
| Wiring-feature caches (fingerprinted parquet) | `/mnt/f/.flybrain/cache/wiring-features/<dataset>/<fingerprint>.parquet` | interim | Reproducible inputs; the fingerprint is recorded in each report |
| Roadmap review (earlier planning critique) | `/mnt/f/.flybrain/logs/roadmap-review-20260924T001615Z/` | historical | Background only |
| Phase 1–5 planning artifacts | `/mnt/f/.flybrain/logs/phase*-2026092*` | historical, superseded | Background only |
| Phase 6 future roadmap | `/mnt/f/.flybrain/logs/phase6-future-roadmap-20260923T210237Z/` | **withdrawn, out of scope** | Do not use (scope bleed from dama-gotchi/hugbot) |

## 3. Literature

- **Synthesis:** `/mnt/f/.flybrain/logs/research-20260924T2245Z/SYNTHESIS.md`. It covers the findings compared with prior work, methods to adopt, parallel efforts, roadmap implications, errata (6b) and references (7).
- **Structured data:** `research.json` in the same directory, with every source, its key result, which findings it links to, and its verification status. Per-topic notes are in `*.md` there too.
- **Verification:** 150 citations were re-fetched by an independent checker. 137 verified, 4 misattributed (corrected in the synthesis), 9 unverifiable (paywalled), 0 fabricated.
- **Cite from the reference list, not from memory.** Claims marked `[unverifiable]` are context only.

## 4. Status and provenance legend

| Tag | Meaning | Loci tier when ingested |
|---|---|---|
| `measured` / `curated` label | independent of wiring | — (label property) |
| `connectivity_defined` label | recovering it from wiring is partly circular; report as "recovery" | — |
| `model_predicted` label | a pass measures distillation, not biology | — |
| Harness-computed number (gated, audited) | reproducible from manifest + fingerprint | `tool_verified` |
| Harness-computed number (interim) | not yet audited | `tool_verified` **plus** an `interim` marker; mark `superseded` when replaced |
| Literature claim from the synthesis | cited, verified source | imported / human-curated tier with the citation attached |
| Model prediction for a neuron | output of a gated model | `model_asserted` (never independent evidence) |
| Withdrawn or historical document | not current | not ingested, or ingested as `superseded` |

## 5. How to cite a number from here

Always include:

- dataset symbol and pinned version;
- target and its label provenance;
- model and report path (or artifact fingerprint);
- split type (grouped);
- CI;
- status (interim or audited).

Example:

> BANC `banc_888`, super_class (curated), wiring-only HGB 0.893 [0.851–0.936] on a grouped split. Source: `real-models-20260924T174122Z/banc/…/quick_super_class_wiring_only.log`. Status: interim.

## 6. Ingesting into Loci

Planned; see the roadmap. Run it after each workflow ships, and only for audited material, except where it is explicitly marked interim. Steps:

1. Ingest this reference, the roadmap and the final results doc through the docs indexer. Loci's docs root must include the repo's `docs/` (`LOCI_DOCS_ROOTS`).
2. Store headline lane results as findings in a `flybrain-reference` investigation. Each finding carries its provenance tier (table above), dataset version, report path and status.
3. Store literature claims with their citation and verification status.
4. When a later run replaces a result, resolve the old finding as `superseded`, pointing to the new one. Do not retract it; retraction is for results that were wrong.

## 7. Open errata

E1–E10 are open provenance questions for dataset teams plus published numbers we could not retrieve. They are listed in `FLYBRAIN_ROADMAP.md` and in the synthesis, section 6b. Two licence gaps found while building this reference, the registry licence for `fw` and `hb`, are tracked here until they are reviewed.

## Refresh

Update this file whenever:

- a workflow ships (model, benchmark or serving);
- a snapshot or pin changes;
- a result's status changes (interim → audited, or superseded);
- an erratum is resolved.

When you update it, keep the status column honest.
