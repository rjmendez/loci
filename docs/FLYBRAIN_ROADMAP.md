# FlyBrain roadmap

Canonical roadmap for FlyBrain models, the benchmark and the FlyBrain Research MCP, as of 2026-09-24. `FLYBRAIN_HARNESS_ROLLOUT_MILESTONES.md` covers the data-harness rollout (M0–M3). This file covers everything built on top of it.

Local research reference (what is where, how trustworthy it is, how to cite it): [FLYBRAIN_REFERENCE.md](FLYBRAIN_REFERENCE.md).

Evidence:

- Interim results: `FLYBRAIN_REAL_MODELS_INTERIM.md`.
- Literature synthesis: [flybrain-research/SYNTHESIS.md](flybrain-research/SYNTHESIS.md) (150 citations checked: 137 verified, 4 corrected, 9 unverifiable, 0 fabricated).

## Scope

FlyBrain is a connectome evidence harness plus small models that predict neuron properties from wiring, with calibrated confidence and label provenance. It is not a behaviour simulator, a robot controller or an NT-from-wiring predictor. Integration with dama-gotchi or hugbot is out of scope unless decided separately (the Phase 6 roadmap was withdrawn for that reason).

## Principles

The literature synthesis and the interim results back all of these.

1. **Label provenance comes first.** Every target is tagged `measured`, `curated_morphology`, `connectivity_defined` or `model_predicted`. Only the first two can pass a gate, be promoted or become a canary. The other two are reported as "recovery" or "distillation".
2. **Grouped evaluation:**
   - No type, hemilineage, homolog group, left/right pair or matched type crosses a split.
   - The test set is used once.
   - Statistics are computed at group level (effective n = number of groups).
3. **Honest baselines.** A model must beat the best trivial rule *and* a size/degree-only model, backed by group-bootstrap CIs, a group-permutation null and a graded split curve (random → type → hemilineage → hemisphere → animal).
4. **One global model per (dataset, target).** Specialists must earn their place against it on the same split.
5. **Calibrated confidence with abstention.** Debiased ECE, a grouped calibration fold, and back-off to the parent class when unsure.

## Pipeline (in order)

### 1. Real models: workflow `flybrain-real-models` (running; draft PR #393)

- **Datasets:** l1em, BANC, optic lobe, MANC and male-cns are done. FlyWire and the graph model are still running.
- **Rigor stage:**
  - provenance gates;
  - literature NT ground truth (`drosophila_neurotransmitters`, confidence ≥ 4, CNN training types removed);
  - size/degree gate;
  - group statistics;
  - normalised, edge-thresholded features (≥ 5 synapses);
  - hierarchical super_class → cell_class;
  - FlyWire hemilineage target;
  - the fw NT pipeline bug fix.
- **Re-evaluation of every lane,** plus the GNN vs HGB comparison (inductive, multiple seeds, both GPUs).
- **Cross-animal transfer** (male-cns ↔ BANC, and → FlyWire): reviewed matches only, dimorphic and sex-specific types excluded, a left→right ceiling and a graph-matching baseline.
- **Integration:** candidate artifacts at canary level only. The swarm consensus-student is deprecated; the per-region router is deferred.

### 2. FlyBrain Research MCP, Phase A: the reference layer (queued; starts when the real-model workflow ships)

A standalone, read-only MCP server that is **not** inside Loci. It serves evidence, not predictions, and is the main way agents query FlyBrain. The data is small and static: the reference catalog, snapshot manifests, the lanes summary, and `docs/flybrain-research/research.json`.

Tools:

- **`datasets()`**
  - versions, licences, citations and manifest paths;
  - which annotation columns are connectivity-defined or model-predicted.
- **`provenance(dataset, column)`**
  - where a label comes from;
  - what that means for using it (for example, "recovery, not discovery").
- **`literature(query)`**
  - search over the 150 checked sources;
  - each hit has its verification status, key result and linked findings (F1–F10).
- **`results(dataset?, target?)`**
  - lane results with CI, gate, label provenance and **status** (interim, audited or superseded).
- **`findings()` and `errata()`**
  - current conclusions and open questions E1–E10.

Rules:

- **Tamper-resistant claims.** Every response pairs its raw numbers with a one-line `claim` field that has the caveats built in (split type, provenance, status), so agents cannot quote a number without its context.
- **Honest status.** Only audited results are reported as audited; interim ones say so.
- **Server standards:** async tools with time limits (the #390 lesson), bearer-token auth, and a `0.0.0.0` bind with a token so Windows clients work. It shares no stores with Loci.

### 3. Benchmark (queued; starts when the test-honesty workflow finishes)

1. **Stereotypy map and sexual dimorphism.** These run first because they change features and pooling.
   - Per-class edge reproducibility sets the thresholds and weights.
   - The published dimorphic and sex-specific type lists decide pooling exclusions. "Is this type dimorphic?" becomes a measured target.
2. **A grouped-split fly-connectome benchmark:** fixed splits across fw, BANC, mc and mv; provenance tags; baselines; a dataset card; a leaderboard script.
3. **NTAC re-test on grouped hold-outs,** with no seeds from test types or left/right pairs.
4. **Per-dataset vs pooled-across-animals vs pooled + hierarchy,** scored in-domain, leave-one-animal-out and against the left→right ceiling.
5. **A super_class-routed hierarchy** for fine classes.
6. **An NT specialist** with its own inputs (lineage, synapse-level or morphology features, wiring → lineage → NT), scored on literature ground truth only.
7. **Single model vs ensemble vs self-training.** Pseudo-labels are tagged `model_predicted`, and scores come from real held-out labels only.
8. **Rigor extras:**
   - a REFORMS report template;
   - multi-seed and multi-split stability for every comparison;
   - a label-noise ceiling;
   - hierarchy mistake-severity metrics;
   - hierarchical conformal abstention;
   - a sparse "AND of ~5 partner types" baseline;
   - a homophily check.

As the benchmark ships, its results, splits and dataset card are added to the Research MCP's `results()` and `datasets()`.

### 4. FlyBrain Research MCP, Phase B: predictions (queued after the benchmark)

This extends the same server. Predictions wait for the audits, calibration and conformal abstention so that agents can rely on them.

- **Shared core:** a `flybrain_serving` library used by both the MCP and an HTTP API (FastAPI).
- **Core behaviour:**
  - serves only gated lanes, from manifest-verified artifacts;
  - reads per-dataset feature stores;
  - every response carries label provenance, gate status, calibrated probability, abstention/back-off to the parent class, model fingerprint, dataset version, CC-BY attribution and the `claim` field.
- **Tools:** `predict`, `explain`, `compare` (cross-animal, via reviewed matches) and `similar`.
- **Candidate tools, not yet confirmed:**
  - `anomalies`/`typicality` (a QA queue);
  - `match_candidates`;
  - `nt_uncertainty`;
  - `score_submission` (benchmark scoring).
- **Edge exports:**
  - ONNX (checked against the original models) plus a Raspberry Pi runner;
  - ESP32 via emlearn/m2cgen C export of logistic regression or small distilled trees. The ESP32 receives feature vectors and does no feature computation.

### 5. Reference upkeep (after each workflow ships)

- Refresh [FLYBRAIN_REFERENCE.md](FLYBRAIN_REFERENCE.md): statuses (interim → audited, superseded), new reports and artifacts, pins, and resolved errata. The Research MCP reads from these same sources, so updating them updates the MCP.
- **Loci ingestion is minimal.** The Research MCP is the primary way to query FlyBrain; Loci does not hold copies. Optionally, keep a handful of headline, **audited** findings in a `flybrain-reference` investigation for cross-project recall:
  - tag them `tool_verified`, with the report path, dataset version and status;
  - literature claims carry their citation;
  - model predictions are `model_asserted`;
  - resolve replaced results as `superseded`, not retracted;
  - never ingest interim numbers or withdrawn material (the Phase 6 roadmap).
- Resolve the registry licence gaps for `fw` and `hb`, and add the missing `l1em` citation.

## Backlog

- **Harness package** (added 2026-09-24):
  - What: extract the evaluation harness into a standalone, domain-agnostic package for graph node-labelling. It covers grouped/cluster splits, provenance tags and gates, trivial and size/degree baselines, group bootstrap and permutation, the graded split curve, debiased calibration and REFORMS reports.
  - First internal use: evaluating Loci's own gates (entailment/pre-answer checks, `classify_text`, provenance tiers) on its code and memory graphs.
  - Candidate external domains: protein-interaction and gene networks, fraud/AML account graphs, knowledge graphs and code graphs.
- **Confidence-gated escalation** (Mowst-style): a cheap model first, escalating uncertain cases. This also applies to Loci's LLM tiers.
- **Tooling reuse audit:** the sjcabs harmonised schema, coconatfly/natverse and pinned CAVE snapshots could replace custom adapter code.
- **Cross-organism harness check** (C. elegans, Lu 2026; MICrONS), after the benchmark ships.
- **Per-dataset NT label-uncertainty notes** for connectome-simulation groups, alongside the errata.

## Deferred, cut or deprecated

| Item | Status | Reason |
|---|---|---|
| Per-region expert router ("brain cluster") | Deferred | Global models match or beat per-region ones at this data scale. MoE needs far more data per expert. Revisit only if a leave-one-region-out test shows region structure. |
| Swarm consensus-student | Deprecated | It trains NB on NB experts' own consensus (model-predicted labels), so it is circular and cannot pass a gate. Ensembling and self-training are evaluated separately. |
| Naive Bayes experts | Weak baseline only | Fails nearly every gate. |
| `flow` target | Dropped | Trivial (missing primary-dendrite width gives 0.947). |
| `connectivity_tier` | Sanity lane only | Mostly tracks neuron size. |
| Larva → adult transfer | Not attempted | No larval MBIN–MBON connection survives metamorphosis. |
| Phase 6 "mobile agents / embodied robotics" mission | Withdrawn | Scope bleed from dama-gotchi and hugbot. |

## Errata (deferred)

Open provenance questions for dataset teams, and published numbers we could not retrieve. None blocks current work. The full table is in section 6b of the research synthesis.

| # | Item | Status |
|---|---|---|
| E1 | BANC `super_class` / `flow` rule | open |
| E2 | Location of BANC NT ground truth | open |
| E3 | Whether larval `io_class` depends on connectivity | open |
| E4 | Which FlyWire types were split by connectivity (CBLAST) | open |
| E5 | male-CNS `consensusNt` construction (88.4% figure unverified) | open |
| E6 | Training-type lists for each NT CNN | open |
| E7 | Matsliah 2024 per-type F-scores | open |
| E8 | Nern 2025 held-out type-level NT accuracy | open |
| E9 | Eckstein 2024 hemilineage-split accuracy | open |
| E10 | Park-Marcotte C1→C3 accuracy drop | open |

## Operations

- **Heavy jobs:** `flybrain-slot`, which runs 3 at a time. The old global lock serialised jobs for hours. The source is `scripts/flybrain-slot`; install it with `install -m 755 scripts/flybrain-slot ~/.local/bin/flybrain-slot`.
- **Loci MCP secrets:** `scripts/ops/loci-mcp-envfile.sh` moves secret `Environment=` lines from the user unit into a mode-600 `EnvironmentFile`. Run it with `--dry-run` first.
- **GPUs:** both are used (index 0 = RTX 2080 Ti, index 1 = RTX 4070 Ti with `CUDA_DEVICE_ORDER=PCI_BUS_ID`). Pin one card per job and keep at least 1 GB free on each so Loci's embedder can load. Ollama's large models degrade to Loci fallbacks (#390/#391) during training.
- **Storage:** writes go only to `/mnt/f/.flybrain/cache` and run logs. Snapshots are read-only.
