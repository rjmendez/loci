# FlyBrain real models: interim results (2026-09-24, lanes v2 added 2026-09-25)

These are interim results from the `flybrain-real-models` workflow. Five dataset tracks (l1em, BANC, optic lobe, MANC, male-cns) are finished. FlyWire (fw) and the graph model are still running. The numbers have **not** been through the workflow's leakage or reproducibility audits. The last row of the location-free table below comes from fw's report while that track is still running. Reports, logs and model artifacts live under `/mnt/f/.flybrain/logs/real-models-20260924T174122Z/`.

**Update 2026-09-25:** the rigor re-evaluation (lanes v2) is in the next section. The per-dataset tables further down are the **old harness** (single-shuffle gate, no size/degree baseline, raw features); where a lane has been re-run, the lanes-v2 verdict supersedes them.

## Method

- **Features:** per-neuron features from the connectome only: in/out partner-category composition, 2-hop composition, reciprocity, degree statistics, and neuropil distributions where stated.
  - Features that define a target's label are excluded for that target.
  - Partner categories of held-out neurons are masked.
  - Body/root ids are never features.
- **Learners:** logistic regression and sklearn `HistGradientBoosting`, both with calibrated probabilities. Naive Bayes is reported only as a weak baseline.
- **Splits:** grouped, so no cell type, hemilineage, homolog group or left/right pair crosses train/val/test. The test set is used once.
- **Gate:** the model must beat the best trivial rule (majority, threshold stump or one-column lookup) with a bootstrap CI. The gate also includes a label-shuffle control and a random-vs-grouped split gap.

## Lanes v2: the rigor re-evaluation (2026-09-25, partial)

Every lane is being re-run with the research-driven rules. The live table, with every lane that has finished since this doc was committed, is `/mnt/f/.flybrain/logs/real-models-20260924T174122Z/lanes-v2.md` (`.json`). A watcher re-aggregates it whenever a report lands.

**What changed from the table below:**

- **Features (R5):** degree columns are within-dataset percentile ranks, and 2-hop composition uses only pairs with ≥ 5 synapses [Schlegel 2024]. Partner-NT features are hard-blocked for every NT target.
- **Gate (R3/R4):** a model must beat all of the following:
  - the best trivial rule, on accuracy and macro-F1;
  - a paired cluster-bootstrap gain;
  - a size/degree-only HGB (by the margin, with a paired CI above 0) [Bernett 2024];
  - a 100-permutation group-block null (p ≤ 0.05) [Ojala & Garriga 2010].

  The gate applies only to measured and curated_morphology labels (R1). Connectivity-defined lanes read as "recovery of connectivity-derived annotations", and model-predicted lanes as distillation.
- **Size baseline inputs:** degree columns excluded as features, such as connectivity_tier's, still reach the size baseline through a side channel (`EvalDataset.aux`). The same channel feeds the degree-decile axis and the hemisphere split level.
- **Calibration (R6):** a dedicated grouped calibration fold. Reported metrics are debiased L2 ECE, ECE-sweep, classwise ECE, log-loss and Brier, each with a cluster-bootstrap CI.

**Finished lanes at commit time.** "Groups" is the number of test groups (the effective n). The size column is the size/degree-only HGB.

| Lane | Provenance | Groups | Trivial | Size | Best model acc [95% CI] / macro-F1 | Perm p | Verdict | Old |
|---|---|---:|---:|---:|---|---:|---|---|
| mc super_class | curated | 527 | 0.739 | 0.690 | hgb 0.979 [0.970–0.987] / 0.80; debiased ECE 0.010 | 0.01 | **pass** | pass |
| ol super_class | curated | 158 | 0.855 | 0.846 | hgb 0.966 [0.946–0.981] / 0.92 | 0.01 | **pass** | pass |
| l1em sensory_modality | curated | 32 | 0.426 | 0.593 | hgb 0.722 [0.582–0.860] / 0.72 | 0.01 | **fail** (size baseline) | pass |
| ol nt_ground_truth (consensus) | measured? | 23 | 0.581 | 0.645 | hgb 0.758 [0.539–0.911] / 0.69 | 0.01 | **fail** (paired gain, size) | fail |
| ol nt_literature (strict R2) | measured | 12 | 0.750 | 0.707 | hgb 0.707 [0.234–0.961] / 0.28 | 0.69 | **fail** (every criterion) | new |
| mc nt_literature (strict R2) | measured | **2** | – | – | not evaluable: the 103 strict types merge into 25 grouped components (type + hemilineage + supertype), 2 of them in test | – | **not evaluable** | new |
| l1em io_class | conn-defined | 209 | 0.805 | 0.897 | hgb 0.945 [0.916–0.971] | 0.01 | not gated (criteria met) | pass |
| l1em connectivity_tier | conn-defined | 209 | 0.726 | **0.997** | hgb 0.874 | 0.01 | not gated; loses to the size baseline by construction | pass |
| ol cell_family | conn-defined | 159 | 0.252 | 0.354 | hgb 0.742 [0.643–0.830] / 0.53 | 0.01 | not gated (criteria met) | pass |

**Reading:**

- **super_class from wiring survives every new control** in male-cns and the optic lobe:
  - It beats a size/degree-only model by 22–35 points (mc) and 6–19 points (ol), both paired CIs.
  - Permutation p is 1/101.
  - mc calibration is excellent (debiased ECE 0.010).
  - The mc split curve falls from 0.994 (random) to 0.989 (type), 0.978 (type + hemilineage) and 0.978 (+ supertype). Left→right transfer is 0.994.
- **l1em sensory modality loses its pass.** Wiring beats the trivial rule, but not a size/degree-only model: the paired gain CI touches 0 with 32 test groups.
- **Literature NT (R2) is a negative result wherever it can be evaluated:**
  - ol strict: 12 test types, p = 0.27 (logreg) / 0.69 (hgb). A random split scores 0.97 against 0.55 on the type split: type memorisation.
  - mc strict: cannot be evaluated on the primary split. The type-only level of the split curve reaches 0.93 [0.81–1.00] against a 0.59 majority over 19 types. That level lets hemilineages cross train/test, and lineage largely fixes NT [Lacin 2019; Eckstein 2024], so it is not evidence of wiring→NT.
- **connectivity_tier is its own definition.** On l1em a size/degree-only model reaches 0.997, so the tier is not a wiring result.

**Graph model** (male-cns, fully inductive):

- **Setup:**
  - Every node sharing a held-out cell type or hemilineage has its partner category masked, and is removed from the inductive training graph.
  - R5 features; graph edges with ≥ 5 synapses.
  - GPUs per R8: 2080 Ti and 4070 Ti, one card per job, capped at 0.6 of the card.
- **super_class over 5 seeds:**

  | Model | Accuracy | Macro-F1 |
  |---|---|---|
  | GraphSAGE inductive | 0.982 ± 0.002 | 0.888 ± 0.004 |
  | GraphSAGE transductive | 0.982 ± 0.002 | 0.881 ± 0.006 |
  | hgb | 0.981 ± 0.003 | **0.908 ± 0.005** |
  | HGB + Correct-and-Smooth (train labels only) [Huang 2020] | 0.975 ± 0.003 | 0.878 |

  - The GNN's accuracy gain over hgb has a CI crossing 0 in 5 of 5 seeds.
  - The GNN's macro-F1 is about 0.02–0.03 lower in every seed.
  - C&S is significantly worse than hgb in 5 of 5 seeds.
  - Removing message passing does not hurt validation accuracy.
  - This agrees with the synthesis: 2-hop composition already carries what a 2-layer GraphSAGE learns [Hamilton 2017; SYNTHESIS F7/F9].
- cell_class and nt_ground_truth (3 seeds each) were still queued at commit time.

**Honest caveats:**

- **Compute:** the shared box ran at load 80–160 for most of the day. The WSL VM also crashed once (OOM from another agent's 45 GB job), and all jobs were restarted.
- **Pending lanes:** 30 lanes were still queued at commit time behind other agents' jobs. `lanes-v2.md` lists them under "Pending"; nothing pending is reported as a result here.
- **Split-curve bias:** levels other than the primary grouping are optimistic for masked targets, because features were masked for the primary split only.
- **ol super_class random-split control:** the hgb random-split control reads 0.79, below the grouped 0.966. It is reported as measured; it is not used by the gate.

## Label provenance

A parallel literature review (`/mnt/f/.flybrain/logs/research-20260924T2245Z/SYNTHESIS.md`) found that several published labels are themselves defined by connectivity or are classifier output. The provenance column below uses these tags:

| Tag | Meaning |
|---|---|
| measured | direct measurement |
| curated | curated from morphology or position |
| conn-defined | defined partly from connectivity, so recovering it from wiring is partly circular |
| predicted | another model's output, so a pass measures distillation, not biology |

## Location-free super_class

The question is whether wiring alone recovers super_class. Super_class is curated from soma position and nerve entry/exit, and anatomical location features can encode that rule, so the table removes them. The wiring-only feature set keeps degree, in/out composition, 2-hop composition and reciprocity; it drops neuropil, optic-lobe layer and column, and BANC morphology.

| Dataset | All features | Neuropil removed | **Wiring only** | Trivial rule (wiring view) |
|---|---:|---:|---:|---:|
| FlyWire (fw)† | 0.961 | 0.967 | **0.967** | 0.671 |
| BANC | 0.961 | 0.921 | **0.893** [0.851–0.936], macro-F1 0.888 | 0.298 |
| Optic lobe (ol) | 0.992 | — | **0.913**, macro-F1 0.711 | 0.705 |
| male-cns (mc) | 0.979 | — | **0.976** [0.966–0.984], macro-F1 0.798 | 0.654 |

Best model shown (hgb); logreg also passes on every wiring-only run; ECE 0.02–0.05. †fw's neuropil-removed run is already wiring-only; the fw track is still running.

Reading:

- Wiring alone recovers super_class at **0.89–0.98** on held-out grouped splits in all four datasets tested (FlyWire, BANC, optic lobe, male-cns).
- In male-cns, region features add nothing: 0.979 with them vs 0.976 without, and macro-F1 is unchanged (0.80).
- In FlyWire, neuropil adds nothing.
- In BANC, anatomy contributed about 7 points: roughly 4 from neuropil and 3 from morphology.
- In the optic lobe, accuracy holds but macro-F1 drops from 0.985 to 0.711. The minority classes (visual projection and centrifugal types) depend on where synapses sit, which is part of how they are defined.

## Finished tracks

### l1em: L1 larva (Winding 2023)

| Target | Provenance | Best | Held-out acc [CI] | Trivial | Gate |
|---|---|---|---:|---:|---|
| io_class (sensory/ascending/DN/RGN/interneuron) | curated* | hgb | 0.945 [0.916–0.969] | 0.805 | pass (5/5 seeds) |
| sensory_modality | curated | hgb | 0.759 [0.621–0.887] | 0.426 | pass (4/5 seeds); ECE 0.17, 54 test neurons |
| connectivity_tier | size proxy | hgb | 0.892 [0.853–0.928] | 0.726 | pass |

The S2 cell types, `level_7_cluster` and ascending modality were rejected as targets because the paper defines them by connectivity. \*The literature review flags io_class as possibly connectivity-assisted; that is open question E3.

### BANC v888

| Target | Provenance | Best | Held-out acc [CI] | Trivial | Gate |
|---|---|---|---:|---:|---|
| super_class | curated | hgb | 0.961 [0.948–0.973] | 0.377 | pass |
| cell_class (~60 classes) | curated | logreg | 0.625 [0.508–0.765] | 0.051 | pass, but ECE 0.42 |
| flow | curated (rule undocumented) | logreg | 0.996 | 0.947 | fail: trivial on missing primary-dendrite width |
| connectivity_tier | size proxy | hgb | 0.883 [0.859–0.907] | 0.753 | pass |
| NT dominance | predicted (synister v2) | logreg | 0.622 [0.538–0.721] | 0.476 | distillation only; hgb 0.79 random vs 0.44 grouped (lineage memorisation) |

### Optic lobe v1.1

| Target | Provenance | Best | Held-out acc [CI] | Trivial | Gate |
|---|---|---|---:|---:|---|
| cell_family (unseen types) | curated? (families are anatomical name classes; types are conn-defined) | hgb | 0.756 [0.655–0.844] | 0.252 | pass; random split 0.964 |
| super_class | curated (location-defined) | hgb | 0.992 → 0.913 wiring-only | 0.855 / 0.705 | pass |
| connectivity_tier | size proxy | hgb | 0.775 [0.715–0.838] | 0.692 | narrow pass |
| NT (predictedNt) | predicted | hgb | 0.784 | 0.529 | distillation only |
| NT (consensusNt with ntReference) | literature-backed | hgb | 0.697 [0.473–0.875] | 0.581 | fail: 134 types, 23 in test |

### male-cns v1.0 (mc)

Split groups are cell type, real hemilineage and supertype, which keeps sister types together. There are at most 20 neurons per type.

| Target | Provenance | Best | Held-out acc [CI] | Trivial | Gate |
|---|---|---|---:|---:|---|
| super_class | curated | hgb / logreg | 0.979 [0.970–0.987]; wiring-only 0.976 | 0.739 (primary-neuropil lookup) | pass; random-vs-grouped gap 1.5 pts |
| cell_class | conn-defined (male-CNS types used NBLAST + connectivity) | logreg | 0.916 [0.866–0.951] | 0.656 | pass (hgb fails the shuffle rule); ECE 0.17–0.43, confidences unreliable |
| region_specialization_tier | — | hgb | 0.844 [0.812–0.877] | 0.700 | pass, with all ROI, neuropil and division inputs excluded |
| connectivity_tier | size proxy | hgb | 0.856 [0.833–0.881] | 0.709 | pass, but a 2-feature sparsity control reaches 0.68 (about 45% of the lift), so this is quasi-tautological |
| NT (type-level `nt_ground_truth`) | literature/curated per type (classifier columns never loaded; hemilineage excluded) | logreg | 0.556 [0.371–0.801]; wiring-only 0.46–0.48 | 0.381 / 0.317 | fail (paired CI crosses 0). The wiring-only pass does not hold on validation; random split 0.78–0.90 is type memorisation |

### MANC v1.0 (mv)

| Target | Provenance | Best | Held-out acc [CI] | Trivial | Gate |
|---|---|---|---:|---:|---|
| cell_class | curated | logreg | 0.891 [0.808–0.948], macro-F1 0.67 | 0.765 | pass; signal is polarity and reciprocity |
| hemilineage (34 classes, unseen types) | conn-defined (MANC assignment used connectivity clustering and NT predictions) | logreg | 0.493 [0.433–0.555], ECE 0.03 | 0.097 | pass; re-test on FlyWire hemilineages |
| NT | predicted (no GT column in MANC v1.0) | — | 0.56–0.58, CI ±0.15 | 0.532 | fail; earlier NB "wins" were same-hemilineage leakage |
| connectivity / region tier | — | logreg | 0.829 / 0.805 | 0.715 / 0.700 | preliminary (full runs queued) |

## What holds so far

1. **Cell identity from wiring is real and reproducible.** super_class works across 4 datasets and 2 life stages, and holds without location features. cell_class holds in MANC, and in BANC with poor calibration. Cell families of never-seen optic-lobe types reach 0.76.
2. **Lineage is partly predictable from wiring.** This needs a clean label; FlyWire hemilineages come next.
3. **NT from wiring does not hold up against ground truth.** "Passes" appear only on classifier-predicted labels (distillation). Literature ground-truth sets are small, so CIs are wide. The next stage uses the `drosophila_neurotransmitters` repository (>900 types).
4. **Naive Bayes over text tokens fails nearly everywhere.** Logistic regression and gradient-boosted trees pass. Boosted trees memorise lineage on random splits, which is why grouped splits are required.

## Findings so far and how they compare with prior work

### What we have learned

**Biology**

1. **Wiring encodes cell identity.** super_class from wiring alone reaches 0.89–0.98 across FlyWire, BANC, optic lobe and male-cns. Other identity results:
   - cell_class: MANC 0.89; BANC 0.63 against a 0.05 trivial rule.
   - cell families of never-seen optic-lobe types: 0.76.
   - larval io_class: 0.95.

   Location or morphology matters only where the class is defined by location (optic-lobe projection and centrifugal classes; BANC about +7 points).
2. **Lineage leaves a partial trace in wiring.** MANC hemilineage reaches 0.49 across 34 classes, against a 0.10 trivial rule. That label is partly connectivity-defined, so the check is being repeated on FlyWire hemilineages.
3. **NT is not recoverable from wiring against ground truth.** Every apparent pass was on classifier-predicted labels (distillation) or leaked through lineage: BANC boosted trees score 0.79 on a random split and 0.44 grouped.
4. **Some targets are proxies:**
   - BANC `flow` is solved by one missing morphology field.
   - `connectivity_tier` mostly tracks neuron size.
   - Optic-lobe super_class with location features is partly the curation rule.

**Method**

5. **Label provenance matters most.** Many published labels are connectivity-defined or model output. Tagging provenance changed how about half of the results should be read.
6. **Grouped splits change the answers.** Random splits inflate results by 10–40 points: optic-lobe family 0.96 vs 0.76, MANC hemilineage 0.84 vs 0.49. They also hid real leaks.
7. **Honest learners on real features win.** Logistic regression and gradient-boosted trees pass. Naive Bayes over text tokens and the swarm consensus-student did not.
8. **Gates are only as good as the tests behind them.** A parallel audit found about 180 hollow tests in the Loci repo.

### Improve next (in the running workflow)

- **Labels:** gates that enforce provenance, and literature NT ground truth.
- **Baselines and statistics:** a size/degree-only baseline, group bootstrap and permutation, and a graded split curve.
- **Features and calibration:** edge-thresholded, normalized features; debiased calibration; a hierarchical super_class → cell_class model.
- **Compute:** 3 concurrent heavy-job slots instead of a single lock, and both GPUs, keeping 1 GB free for Loci's embedder.

### Expand next (queued benchmark workflow)

- **Cross-animal:** transfer with measured ceilings, and a pooled per-task model across animals.
- **Specialists:** a super_class-routed hierarchy, and an NT specialist with its own inputs.
- **Studies:** stereotypy and sexual-dimorphism analyses that decide what can safely be pooled.
- **The benchmark:** a public grouped-split benchmark with provenance tags, and an NTAC re-test on grouped hold-outs.

### Has anyone released a similar model?

Parts of this exist; we did not find this combination. Sources were verified in `/mnt/f/.flybrain/logs/research-20260924T2245Z/SYNTHESIS.md`.

| Work | What it is | How it differs from ours |
|---|---|---|
| NTAC (Schwartzman 2025) | Connectivity-based cell typing on FlyWire: >95% with 2% seed labels, about 52–70% unseeded | Seeded protocol leaks members of the same type; no grouped hold-out |
| Matsliah 2024; Nern 2025 | Optic-lobe typing; about 5 partner types / top-5 connections define a type | These define the types from connectivity (label source, not a predictive model) |
| flywire-gnn (Dhakane, unreviewed) | FlyWire super_class with MLP/GraphSAGE, about 0.98 | Random split, with neuropil and predicted-NT features (the leaks our gates block) |
| NeuNet (2024) | Hemibrain classification from skeleton + connectome, 0.917 (connectome branch alone 0.587) | Morphology-led; not a grouped evaluation |
| Elabbady 2025 (MICrONS, mouse) | Cell typing from perisomatic features: 91% cross-validated, 82% on expert validation | Mouse and non-wiring, but shows the same drop under stricter validation |
| Eckstein 2024 (synister) | NT from EM synapse images, 94% per neuron | The source of the datasets' NT columns; images, not wiring |
| ACDC (Lee / Matsliah 2026) | Topology-only graph alignment across datasets | Matching, not classification; our transfer baseline |
| Lappalainen 2024 (flyvis); Shiu 2024 | Connectome-constrained dynamical models | Consume predicted NT, so NT provenance matters to them |
| Lu 2026 (C. elegans) | Graph vs non-graph neuron-classification benchmark | Worm, 3 classes; a candidate cross-organism check |

**What appears new:**

- a wiring-only classifier evaluated with grouped hold-outs plus trivial and size baselines;
- label-provenance gating;
- evidence that NT is not recoverable from wiring against ground truth;
- once the transfer stage runs, wiring-only transfer across animals with measured ceilings.

This is based on one literature pass. Repeat the search before publishing.

## Next (in the running workflow)

- Finish the queued lanes-v2 runs (the watcher keeps `lanes-v2.md` current). Then run the r5:10 2-hop sensitivity on the headline lanes, and the graph cell_class / nt_ground_truth seeds.
- The earlier rigor-stage items below are now implemented (provenance gate, literature NT, size/degree gate, group bootstrap and permutation, split curve, R5 features, debiased calibration). The hierarchical super_class → cell_class model exists as a report-only harness hook (R7), and no lane has turned it on yet.

- Finish the mc, fw and graph-model tracks.
- **Rigor stage:**
  - label-provenance tags enforced by the gates;
  - literature NT ground truth;
  - a size/degree-only baseline gate;
  - group bootstrap and group permutation;
  - a graded split curve;
  - edge-thresholded, normalised features;
  - debiased calibration;
  - a hierarchical super_class → cell_class model;
  - the fw NT pipeline bug fix.
- **Cross-animal transfer:** reviewed matches only, dimorphic types excluded, a left→right ceiling and a graph-matching baseline.
- The per-region expert router is deferred, and the swarm consensus-student is deprecated.
