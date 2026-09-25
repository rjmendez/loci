# FlyBrain `fw` (FlyWire 783) adapter contract

The metadata-only lookup adapter is specified in
[`FLYBRAIN_FW_METADATA_ADAPTER_CONTRACT.md`](FLYBRAIN_FW_METADATA_ADAPTER_CONTRACT.md).
This document covers the **training / evaluation** side of `fw`: which snapshot
files a model may read, how they are verified, which targets exist, and what
the evaluation guarantees.

## Real models

Code: `mcp/flybrain_fw_targets.py` (targets, snapshot verification, features,
eval datasets, CLI) and `mcp/flybrain_brain_cluster_fw_samples.py`
(`build_fw_structured_samples`, a sample-dict view of the same data; the legacy
`build_fw_training_samples` is unchanged and remains **circular**; roadmap B1).
Tests: `mcp/tests/test_flybrain_fw_targets.py` (synthetic snapshot only).

### Snapshot access (fail closed, read only)

`open_fw_snapshot(storage_root, required_roles=..., stamp_dir=...)` checks the
following before any product is read:

1. `manifest/manifest.json` and `manifest/manifest.sha256` exist, and the sidecar equals `integrity.manifest_sha256`.
2. The manifest self-hash (the same rule as the other adapters) matches.
3. `verification.status == verified`, the dataset is pinned to `fw/flywire783`, `refresh.decision != rollback` and `next_check_due` has not passed.
4. Every listed file has a safe relative path, is not partial or temporary, exists, and has the listed size.
5. Each required product is sha256-verified. Hash stamps go to `<storage_root>/cache/hash-stamps/fw`. A stamp dir inside `snapshots/` is refused.

Products used:

| role | file |
|---|---|
| `edgelist_per_neuropil` | `source/proofread_connections_783.feather`: pre, post, neuropil, syn_count (the per-edge NT columns are never read) |
| `neuron_annotations_hierarchical` | `source/nature_schlegel2024/Supplementary_Data_1_neuron_annotations.tsv` |
| `per_neuron_neuropil_count_pre` | `source/per_neuron_neuropil_count_pre_783.feather` (used only for the connectivity_tier label) |

Nothing is written under `snapshots/`. Features are cached under
`/mnt/f/.flybrain/cache/wiring-features/fw/`. The transfer bridge table goes to
`.../wiring-features/fw/bridge/`, and reports go to
`/mnt/f/.flybrain/logs/real-models-20260924T174122Z/fw/<target>/<run_label>/`.

### Targets

The report target name is also the label-exclusion objective.

| target | label | partner category (features) | held-out categories masked | ground truth? |
|---|---|---|---|---|
| `super_class` | Schlegel `super_class`, harmonized vocab | super_class | yes | yes (curated) |
| `super_class_no_neuropil` | same, with every neuropil/region feature also excluded | super_class | yes | yes |
| `flow` | Schlegel `flow` (afferent / intrinsic / efferent) | super_class | yes | yes |
| `cell_class` | Schlegel `cell_class`, classes with >= 100 neurons | cell_class | yes | yes |
| `cell_class_no_neuropil` | same, no neuropil features | cell_class | yes | yes |
| `neurotransmitter_dominance` | `top_nt`, the per-neuron argmax of Eckstein 2024 synapse-level NT **predictions** | super_class | no (the category is not the label) | **no**: the label is another model's output |
| `nt_ground_truth` | `known_nt` (literature), a single classical transmitter; ambiguous co-transmitter strings are dropped | super_class | no | yes |
| `connectivity_tier` | top quartile of total presynapse count (threshold computed over all annotated neurons) | super_class | no | yes (a definition) |

Two objectives, `super_class_no_neuropil` and `cell_class_no_neuropil`, are
registered by `flybrain_fw_targets` itself. They take the registered type-hierarchy
patterns and add the region patterns. They exist because fw `super_class`
(optic, central, visual_projection) and the optic `cell_class` values (`ME>LO`,
`LA>ME`, ...) are defined by where a neuron's arbors lie, so the neuropil
fraction features are close to definitional for them.

### Features

`flybrain_wiring_features.build_wiring_features` runs over the connections table
with `unique_pairs=False` (one row per pre/post/neuropil) and `two_hop=True`.
It produces degree, out/in partner composition, out/in neuropil fractions (top
20), reciprocity, and 2-hop composition. The objective's exclusions are applied
at build time, and constant columns are dropped. Body and root ids are never
features. The nb learner reads a text view: decile-binned feature tokens
(`name name__qK`), with bin edges fitted on the train rows only.

### Sampling and splits

- Rows with Schlegel `status` of outlier_seg or outlier_bio are not samples. They still count as graph nodes.
- Groups are union-find over `cell_type`, `hemilineage` (ito_lee) and `hemibrain_type`. No group spans train, val and test.
- Each grouped component is capped at `max_per_component=200` rows, so giant types (photoreceptors, T4/T5, Kenyon cells) cannot fill whole classes or whole test splits. Each class is then capped at `max_per_class` (10k; 4k for cell_class) by taking whole capped components in sha256 order. Classes below `min_class_count` (100; 50 for nt_ground_truth) are dropped.
- When the partner category is the label or determines it, the val and test node categories are masked before the features are built, and the split hash is recorded in `notes.masked_split_ids_sha256`. `run_evaluation` refuses a different split.
- Neurons with no type, no hemilineage and no hemibrain type are singleton groups. An untyped left/right homolog pair can therefore straddle splits (a residual caveat).

### Evaluation

Every target goes through `flybrain_model_eval.run_evaluation` with nb, logreg
(C in {0.1, 1}) and hgb (lr 0.1, 300 iterations, class_weight in {None,
balanced}). Tuning uses 4-fold grouped CV inside train, and all models are
calibrated with temperature scaling on the out-of-fold predictions. The
evaluation reports:

- the majority class, the best trivial rule of the model's view, and test accuracy with a grouped-bootstrap 95% CI;
- macro-F1 and ECE;
- a label-shuffle control;
- a random-split control;
- a drop-one-family ablation on val.

CLI:

```
python flybrain_fw_targets.py <target> [<target> ...] --run-label v2 [--write-bridge]
```

Wrap the command in `flock /tmp/flybrain-heavy.lock`.

### Results (run label `v2`)

All numbers are from the held-out grouped test split, evaluated once per final
config. The reports are at
`/mnt/f/.flybrain/logs/real-models-20260924T174122Z/fw/<target>/v2/report.{json,md}`,
the models at `.../models/<name>/`, and this table at `.../fw/results_v2.{md,json}`.
"random-split acc" is the per-sample random-split control; it is optimistic by
construction and is shown only to measure the grouped-vs-random gap.

| target | model | majority | best trivial (rule) | acc [95% CI] | macro-F1 | ECE | shuffle acc | random-split acc | test components | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| super_class | nb | 0.228 | 0.671 (text:lookup) | 0.890 [0.850, 0.925] | 0.826 | 0.038 | 0.132 | 0.880 | 694 | pass |
| super_class | logreg | 0.228 | 0.591 (features:lookup) | 0.942 [0.905, 0.969] | 0.919 | 0.044 | 0.246 | 0.971 | 694 | pass |
| super_class | hgb | 0.228 | 0.591 (features:lookup) | 0.961 [0.932, 0.982] | 0.925 | 0.025 | 0.220 | 0.986 | 694 | pass |
| super_class_no_neuropil | nb | 0.228 | 0.671 (text:lookup) | 0.875 [0.834, 0.909] | 0.812 | 0.040 | 0.150 | 0.863 | 694 | pass |
| super_class_no_neuropil | logreg | 0.228 | 0.591 (features:lookup) | 0.941 [0.907, 0.965] | 0.913 | 0.055 | 0.234 | 0.962 | 694 | pass |
| super_class_no_neuropil | hgb | 0.228 | 0.591 (features:lookup) | 0.967 [0.948, 0.983] | 0.939 | 0.024 | 0.202 | 0.985 | 694 | pass |
| flow | nb | 0.525 | 0.860 (text:lookup) | 0.824 [0.678, 0.930] | 0.709 | 0.065 | 0.572 | 0.876 | 361 | fail |
| flow | logreg | 0.525 | 0.887 (features:threshold) | 0.943 [0.878, 0.983] | 0.920 | 0.019 | 0.519 | 0.970 | 361 | fail |
| flow | hgb | 0.525 | 0.887 (features:threshold) | 0.963 [0.933, 0.987] | 0.934 | 0.010 | 0.525 | 0.988 | 361 | pass |
| cell_class | nb | 0.117 | 0.368 (text:lookup) | 0.801 [0.673, 0.910] | 0.576 | 0.099 | 0.003 | 0.902 | 888 | pass |
| cell_class | logreg | 0.117 | 0.296 (features:lookup) | 0.874 [0.770, 0.948] | 0.578 | 0.106 | 0.148 | 0.975 | 888 | fail |
| cell_class | hgb | 0.117 | 0.296 (features:lookup) | 0.883 [0.790, 0.956] | 0.642 | 0.091 | 0.039 | 0.981 | 888 | pass |
| cell_class_no_neuropil | nb | 0.117 | 0.368 (text:lookup) | 0.753 [0.622, 0.869] | 0.511 | 0.083 | 0.004 | 0.877 | 888 | pass |
| cell_class_no_neuropil | logreg | 0.117 | 0.260 (features:lookup) | 0.854 [0.765, 0.930] | 0.597 | 0.126 | 0.159 | 0.971 | 888 | fail |
| cell_class_no_neuropil | hgb | 0.117 | 0.260 (features:lookup) | 0.854 [0.755, 0.943] | 0.611 | 0.086 | 0.035 | 0.981 | 888 | pass |
| neurotransmitter_dominance | nb | 0.248 | 0.431 (text:lookup) | 0.381 [0.285, 0.484] | 0.288 | 0.102 | 0.228 | 0.477 | 726 | fail |
| neurotransmitter_dominance | logreg | 0.248 | 0.370 (features:lookup) | 0.468 [0.371, 0.571] | 0.335 | 0.092 | 0.227 | 0.634 | 726 | pass |
| neurotransmitter_dominance | hgb | 0.248 | 0.370 (features:lookup) | 0.567 [0.452, 0.671] | 0.436 | 0.080 | 0.287 | 0.789 | 726 | fail |
| nt_ground_truth | nb | 0.828 | 0.844 (text:lookup) | 0.644 [0.418, 0.861] | 0.237 | 0.201 | 0.706 | 0.703 | 27 | fail |
| nt_ground_truth | logreg | 0.828 | 0.844 (features:lookup) | 0.688 [0.500, 0.872] | 0.201 | 0.175 | 0.826 | 0.911 | 27 | fail |
| nt_ground_truth | hgb | 0.828 | 0.844 (features:lookup) | 0.721 [0.495, 0.936] | 0.317 | 0.475 | 0.611 | 0.981 | 27 | fail |
| connectivity_tier | nb | 0.398 | 0.699 (text:lookup) | 0.652 [0.526, 0.796] | 0.650 | 0.050 | 0.403 | 0.695 | 341 | fail |
| connectivity_tier | logreg | 0.398 | 0.632 (features:lookup) | 0.673 [0.568, 0.781] | 0.672 | 0.062 | 0.425 | 0.745 | 341 | fail |
| connectivity_tier | hgb | 0.398 | 0.632 (features:lookup) | 0.805 [0.743, 0.872] | 0.801 | 0.040 | 0.396 | 0.885 | 341 | pass |

How to read the table:

- **super_class, flow and cell_class can be learned from wiring, even with no neuropil features.** Held-out categories are masked and whole cell types, hemilineages and hemibrain types are held out. Removing every neuropil feature does not hurt super_class (hgb 0.967 vs 0.961), and costs cell_class about 3 points. The signal is mostly partner composition plus degree. The drop-one-family ablations on val are small because the families are redundant.
- **The label-shuffle gate is fragile under class caps.** It compares the shuffle accuracy with the *train-majority* rule's accuracy on test. When the per-class caps make the train majority differ from the test majority, a shuffled model that happens to predict the test's most common class passes that bar without any leakage. Two examples:
  - `neurotransmitter_dominance` hgb: shuffle 0.287 vs "majority" 0.248, while the test's most common class is at 0.418.
  - `cell_class*` logreg: shuffle 0.15 vs 0.117.

  Those models fail only on this criterion. The paired accuracy gain CIs are well above 0 (NT hgb [0.090, 0.317]; cell_class logreg [0.447, 0.695]). The gates are reported exactly as the harness computed them.
- **flow logreg fails** because its paired gain CI over the one-feature threshold stump includes 0 ([-0.028, 0.171]).
- **neurotransmitter_dominance** has non-ground-truth labels: `top_nt` is the argmax of Eckstein 2024's synapse-image classifier. The best model reaches 0.567 (logreg 0.468 passes the gate). Wiring carries some transmitter signal for unseen cell types and hemilineages, but far less than the random-split number (0.789) suggests.
- **nt_ground_truth is a negative result.** Every model is below the majority class (0.828 acetylcholine). Only 164 grouped components carry a single-classical `known_nt`, and only 27 of them are in test, so the literature labels are effectively per cell type, and the random-split 0.981 is pure type memorization. The earlier v1 run (`nt_ground_truth/final`, before the per-component cap) failed the same way: 18 test components; model 0.10-0.19 vs trivial 0.78-0.84. That v1 result is why the per-component cap was introduced, so the v2 test split for this one target is not fully untouched.
- **connectivity_tier passes only with hgb** (0.805 vs 0.632), with every degree, count and n_neuropils feature excluded. A val-only follow-up (`fw/checks/conn_entropy_ablation.json`) also dropped the entropy features, then recip and the `other` fractions. Val accuracy went 0.757 → 0.741 → 0.707, and composition-only features alone reached 0.724, so the signal is not a hidden size proxy.
- **Calibration.** Temperature scaling fitted on grouped out-of-fold probabilities sometimes raises test ECE a little (super_class hgb 0.014 raw → 0.025). nt_ground_truth hgb is badly calibrated (0.475) because of the val/test distribution shift.
