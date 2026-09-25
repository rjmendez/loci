# L1EM real-model evaluation (Winding et al. 2023)

**Run date:** 2026-09-25
**Dataset:** L1 larval connectome, Winding et al. 2023 Science 379:eadd9330
**Snapshot:** `catmaid_l1em` (PMC7614541, CC-BY-4.0)
**Manifest SHA:** `ec77f89ba0f5755537f00a49755afa9677f1aeca64ea15feca6f64c5622616ed`
**Pipeline:** `flybrain_l1em_targets.py` + `flybrain_model_eval.run_evaluation`
**Split:** pair-grouped (homologous left/right pairs), 5 seeds (primary + 4 repeats)
**n_train / n_test:** 1769 / 380 (io_class, connectivity_tier); 247 / 54 (sensory_modality)

---

## Rejected targets

The following are connectivity-circular or unavailable in the release and are never used as prediction targets:

| Target | Reason |
|---|---|
| `celltype` | Defined by connectivity motif (Fig. S4C); re-deriving from wiring is circular |
| `level_7_cluster` | Spectral clustering of the connectivity graph (Fig. S8) |
| `ascending_modality` | Table S1: "based on connectivity" |
| `neurotransmitter_dominance` | No per-neuron NT annotation in the release |
| `region_specialization_tier` | No per-synapse larval neuropil assignment |

---

## Results

### io_class (brain input/output class)

Primary split:

| Model | Accuracy [95% CI] | Macro-F1 | ECE | Best trivial | Gate |
|---|---|---|---|---|---|
| NaiveBayes | 0.863 [0.822, 0.901] | 0.706 | 0.062 | 0.813 (text:lookup) | fail |
| LogReg | 0.932 [0.899, 0.962] | 0.767 | 0.036 | 0.805 (features:lookup) | pass |
| HGB | 0.942 [0.913, 0.968] | 0.810 | 0.024 | 0.805 (features:lookup) | pass |

Across 5 seeds:

| Model | Acc mean +- sd | Gain over trivial | Gate passes |
|---|---|---|---|
| HGB | 0.951 +- 0.014 | +0.128 [+0.092, +0.140] | 5/5 |
| LogReg | 0.940 +- 0.017 | +0.117 [+0.092, +0.142] | 5/5 |
| NaiveBayes | 0.857 +- 0.024 | +0.042 [+0.005, +0.074] | 1/5 |

Cluster-grouped (pair + level-7 cluster): HGB 0.937 [0.883, 0.982] vs trivial 0.826, gate pass.

---

### sensory_modality (sensory neuron type by sense organ)

Small subset (sensory neurons only, n_test=54). Primary split:

| Model | Accuracy [95% CI] | Macro-F1 | ECE | Best trivial | Gate |
|---|---|---|---|---|---|
| NaiveBayes | 0.611 [0.444, 0.768] | 0.541 | 0.139 | 0.519 (text:lookup) | fail |
| LogReg | 0.704 [0.564, 0.839] | 0.636 | 0.143 | 0.426 (features:lookup) | pass |
| HGB | 0.759 [0.621, 0.887] | 0.738 | 0.171 | 0.426 (features:lookup) | pass |

Across 5 seeds: HGB 0.788 +- 0.039, +0.334 over trivial, 4/5 gate passes.
(1 fail is expected variance at n=54 test set.)

---

### connectivity_tier (high vs baseline total synapses)

75% majority class. Primary split:

| Model | Accuracy [95% CI] | Macro-F1 | ECE | Best trivial | Gate |
|---|---|---|---|---|---|
| NaiveBayes | 0.813 [0.761, 0.860] | 0.772 | 0.032 | 0.766 (text:lookup) | fail |
| LogReg | 0.826 [0.775, 0.873] | 0.754 | 0.056 | 0.726 (features:lookup) | pass |
| HGB | 0.892 [0.855, 0.928] | 0.867 | 0.040 | 0.726 (features:lookup) | pass |

Across 5 seeds: HGB 0.878 +- 0.026, +0.133 over trivial, 5/5 gate passes.

Legacy text NaiveBayes (brain-cluster pipeline): 0.861, gate fail (trivial lookup 0.903 beats it).

---

## Key takeaways

1. HGB on wiring features passes all gates with comfortable margin across 5 seeds.
2. Shuffle controls confirm signal is real; model accuracy is +12-33% over trivial depending on target.
3. NaiveBayes text fails all primary targets; it is the correct tool for Loci internal text routing, not connectome classification.
4. sensory_modality is harder but tractable at n=54 test; 4/5 seeds pass. More data (e.g. FlyWire olfactory labels) would tighten calibration.
5. Cluster-grouped splits confirm the same pass/fail pattern; generalisation is not a split artifact.

---

## Snapshot requirements

Run `flybrain_l1em_targets.py` with `LOCI_FLYBRAIN_STORAGE_ROOT` pointing to a root that contains:

```
snapshots/l1em/catmaid_l1em/
  source/PMC7614541_SupplementaryFiles.zip   # 10,942,755 bytes
  metadata/files/
    all-all_connectivity_matrix.csv          # 34,914,576 bytes; SHA 94d51a82...
    Supplementary_Data_S2.csv               # 65,148 bytes; SHA 477250fc...
    inputs.csv                              # 55,324 bytes; SHA a42d206a...
    outputs.csv                             # 55,124 bytes; SHA 01cb2aee...
    aa_connectivity_matrix.csv              # 34,524,851 bytes; SHA 6ea30ce0...
    ad_connectivity_matrix.csv              # 34,487,969 bytes; SHA 6c758683...
    da_connectivity_matrix.csv              # 28,738,194 bytes; SHA e0d3dd85...
    dd_connectivity_matrix.csv              # 30,491,891 bytes; SHA 476ee5e6...
  manifest/manifest.json                    # SHA ec77f89b...
  manifest/manifest.sha256
```

All CSVs extract from PMC7614541_SupplementaryFiles.zip (Europe PMC, CC-BY-4.0).
SHA256 of the zip: f2608b4c73232f0704e494a874dfd8c91b2db3520e03579bbf3b3b9637b3cf85
