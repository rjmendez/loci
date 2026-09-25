# FlyBrain real models: interim results (2026-09-24)

These are interim results from the `flybrain-real-models` workflow. Four dataset tracks (l1em, BANC, optic lobe, MANC) are finished. male-cns (mc), FlyWire (fw) and the graph model are still running. The numbers have **not** been through the workflow's leakage or reproducibility audits. The last row of the location-free table below comes from fw's report while that track is still running. Reports, logs and model artifacts live under `/mnt/f/.flybrain/logs/real-models-20260924T174122Z/`.

## Method

- **Features:** per-neuron features from the connectome only: in/out partner-category composition, 2-hop composition, reciprocity, degree statistics, and neuropil distributions where stated.
  - Features that define a target's label are excluded for that target.
  - Partner categories of held-out neurons are masked.
  - Body/root ids are never features.
- **Learners:** logistic regression and sklearn `HistGradientBoosting`, both with calibrated probabilities. Naive Bayes is reported only as a weak baseline.
- **Splits:** grouped, so no cell type, hemilineage, homolog group or left/right pair crosses train/val/test. The test set is used once.
- **Gate:** the model must beat the best trivial rule (majority, threshold stump or one-column lookup) with a bootstrap CI. The gate also includes a label-shuffle control and a random-vs-grouped split gap.

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
