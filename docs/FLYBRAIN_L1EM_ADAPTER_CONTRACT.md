# FlyBrain l1em (L1 larval connectome) adapter contract

Status: **planned dataset, contract + builder implemented, not promoted.**
Modules: `mcp/flybrain_l1em_adapter.py`, `mcp/flybrain_brain_cluster_l1em_samples.py`.
Tests: `mcp/tests/test_flybrain_l1em_adapter.py`, `mcp/tests/test_flybrain_brain_cluster_l1em_samples.py`.
These use synthetic fixtures and run offline. Each file also has a smoke test that runs only when `LOCI_FLYBRAIN_STORAGE_ROOT` contains the real snapshot.

## 1. Scope

- Dataset: Winding M, Pedigo BD, Barnes CL, et al. (2023) *The connectome of an insect brain.* Science 379(6636):eadd9330. doi:10.1126/science.add9330.
- Organism and stage: *Drosophila melanogaster*, first-instar larva (L1), brain only. Sex is mixed or unspecified.
- Registry: symbol `l1em`, snapshot dir `l1em`, pinned version `catmaid_l1em`, `OrganismStage.LARVAL`, `RegionVocabulary.CATMAID_ANNOTATION`.
- Claim tier: T1, meaning claims are specific to this dataset and version. Only dataset-local larval claims are allowed.
- This is structural evidence only. It is not a brain simulation.

## 2. Source and license

| Item | Value |
|---|---|
| Distribution used | Europe PMC supplementary bundle of author manuscript PMC7614541 (EMS175448) |
| URL | `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7614541/supplementaryFiles` |
| Retrieved (UTC) | 2026-09-24T00:24:23Z |
| License | **CC-BY-4.0**. The PMC article states "This work is licensed under a CC BY 4.0 International license", and the Europe PMC core record gives `license = cc by`, `isOpenAccess = Y`. |
| Required citation | Winding et al. 2023, Science 379:eadd9330, doi:10.1126/science.add9330 (author manuscript PMC7614541) |

We did **not** use the GitHub mirror `brain-networks/larval-drosophila-connectome`. It has no license file (this was the open question in the roadmap review), and its `Supplementary-Data-S1.zip` is 1,107,380 bytes, so it is not byte-identical to the Europe PMC copy (1,054,841 bytes). The PMC link on `pmc.ncbi.nlm.nih.gov` returns a bot-challenge HTML page to non-browser clients, so it cannot be scripted.

The adapter accepts only licenses in `L1EM_ALLOWED_LICENSES = {"CC-BY-4.0"}`. Any other license value, including `UNREVIEWED`, raises `LICENSE_UNVERIFIED`.

## 3. Snapshot layout and manifest

Root: `$LOCI_FLYBRAIN_STORAGE_ROOT/snapshots/l1em/catmaid_l1em/`

- `source/PMC7614541_SupplementaryFiles.zip`: the bundle exactly as downloaded (11 MB).
- `metadata/files/`: data files extracted from the bundle. We extracted only Data S1 to S4. The figures, the PDF and the docx were not extracted.
- `manifest/manifest.json` and `manifest/manifest.sha256`: `fbh-manifest/v1`, with `artifact.relative_root = snapshots/l1em/catmaid_l1em`. Every file entry records `source_url`, `sha256`, `size_bytes`, `retrieved_at`, `license`, `citation`, `version` and `extracted_from`.
- `metadata/manifest.json` is the older phase-5 manifest, which covers only the hosted page. It was **left untouched**, and the adapter does not read it.

| relative_path | bytes | sha256 |
|---|---|---|
| source/PMC7614541_SupplementaryFiles.zip | 10942755 | f2608b4c73232f0704e494a874dfd8c91b2db3520e03579bbf3b3b9637b3cf85 |
| metadata/files/Supplementary_Data_S1.zip | 1054841 | 258d79eb8e0891b6c42ee67e33b8efcf6505c641cdac7f1f7bc33f985857a695 |
| metadata/files/Supplementary_Data_S2.csv | 65148 | 477250fc559684f0e898a12016f41221bfa320012e378ab7ae8c5125c7776f7e |
| metadata/files/Supplementary_Data_S3.csv | 59709 | e7b4908a0a52627f5e97b9496c2cb7e2d64ddfdf4d98dffdf8bb8ccebe82999d |
| metadata/files/Supplementary_Data_S4.csv | 34029 | 885f22873b1410e7817ca517aa591d80c0b76ad9e65a84202827576b5c18927c |
| metadata/files/all-all_connectivity_matrix.csv | 34914576 | 94d51a821217048acf711f4feb71693a8487f2547540c6f2f25896c60e2348e3 |
| metadata/files/aa_connectivity_matrix.csv | 34524851 | 6ea30ce009da9baf5280dac80f37c2d9d21d1620e38a04eea2051920781b3e36 |
| metadata/files/ad_connectivity_matrix.csv | 34487969 | 6c758683f657e787fca5626dcab2a8e31fbcaf687ad6c9b2cb4fdd6b584a426f |
| metadata/files/da_connectivity_matrix.csv | 28738194 | e0d3dd85a53516271eeb310af0c0cd590a4efc1380b65f9aed6b34edecfcb317 |
| metadata/files/dd_connectivity_matrix.csv | 30491891 | 476ee5e609091cdc623ba0620a6353f4aa652d145ed74929f07c04bbe6536d0c |
| metadata/files/inputs.csv | 55324 | a42d206abe2963dd99f2edef9bad3dfa5e6b0b36ec1c87d99aafd21432a45851 |
| metadata/files/outputs.csv | 55124 | 01cb2aeebb5b78bb36dabddd8220a3b5ba3fc6ce728b7860bdcaaee44f091276 |

Total on disk: about 175 MB. `manifest_sha256` is `8f0f7d1eab952ae68c8247f307851d9494be5f69105f0b7a8b0e7fe1c66fd4ca`.

## 4. Tables the adapter reads (`L1EM_REQUIRED_FILES`)

| Role | File | Contract |
|---|---|---|
| `all_all_matrix` | `all-all_connectivity_matrix.csv` | Square matrix whose first column is the index. Row labels and column labels are the same integer skids in the same order. Values are finite, non-negative integers. **Rows are presynaptic and columns are postsynaptic**: the column sums match `inputs.csv` (r = 0.96). `all-all` equals `aa + ad + da + dd` exactly. The matrix has 2952 neurons, 110,677 non-zero edges and 352,611 synapses. |
| `annotations` | `Supplementary_Data_S2.csv` | Columns `left_id, right_id, celltype, additional_annotations, level_7_cluster`. A missing side is written as `no pair`. Each row must have at least one skid, and no skid may appear more than once. There are 1372 rows covering 2610 skids, 2606 of which are in the matrix. |
| `inputs` / `outputs` | `inputs.csv` / `outputs.csv` | Header `,axon_input,dendrite_input` / `,axon_output,dendrite_output`. Counts must be >= 0 and skids must be unique. |

Matrix neurons with no S2 annotation (346 of them) go into `L1emSnapshot.unannotated_skids` and are left out of samples.

## 5. Fail-closed behavior (`L1emErrorCode`)

Every failure raises `L1emAdapterError(ValueError)`. The error carries `.code` and `.details`, and its message starts with `[CODE] `.

| Code | Trigger |
|---|---|
| `ROOT_NOT_CONFIGURED` | The storage root cannot be resolved, or `snapshot_root` is relative. |
| `PATH_ESCAPE` | `relative_root` or a file path is unsafe (absolute path, drive letter, or `..`), or it resolves outside the snapshot. |
| `MANIFEST_MISSING` | `manifest/manifest.json` or its `manifest/manifest.sha256` sidecar is absent. A sidecar that does not match the manifest self-hash raises `INTEGRITY_MISMATCH`. Every file entry must carry an integer `size_bytes`, which is checked even when `verify_integrity=False`. |
| `MANIFEST_INVALID` | The file is not JSON, has the wrong `schema_version`, is missing a section, has a bad status, hash format or refresh decision, lists duplicate or partial files, or has no citation. |
| `MANIFEST_STALE` | `refresh.decision == rollback`, or `next_check_due` has passed. |
| `INTEGRITY_MISMATCH` | The manifest is not `verified`, the manifest self-hash does not match, or a file's size or sha256 does not match. |
| `DATASET_PIN_MISMATCH` | The symbol or version is not `l1em/catmaid_l1em`, or `relative_root` is not `snapshots/l1em/catmaid_l1em`. |
| `LICENSE_UNVERIFIED` | `dataset.source.license.spdx_id` is not in the allow-list. |
| `REQUIRED_FILE_MISSING` | A listed file is missing on disk, or a required role is not listed in the manifest. |
| `SCHEMA_MISMATCH` | A CSV header or row is invalid, or input features differ from `L1EM_INPUT_FEATURES`. |
| `MATRIX_INVALID` | The matrix is not square, has mismatched labels, has duplicate skids, or has non-integer or negative values. |
| `ANNOTATION_CONFLICT` | A skid is annotated twice. |
| `OBJECTIVE_UNSUPPORTED` | The builder was called directly with an objective other than `connectivity_tier`. |
| `CROSS_STAGE_UNSUPPORTED` | A region id without the `l1_` prefix, any call to `map_to_adult_neuropil`, a scope with an adult stage or target stage, an adult region vocabulary, or a manifest whose `scope.stage` is not larval. |

The adapter never makes network calls. There is no remote fallback.

## 6. Region vocabulary and cross-stage rule

- `region_id = "l1_" + sanitize(celltype)`, taken from the S2 cell types. The current set is `l1_ascending, l1_cn, l1_dn_sez, l1_dn_vnc, l1_kc, l1_lhn, l1_ln, l1_mb_fbn, l1_mb_ffn, l1_mbin, l1_mbon, l1_pn, l1_pn_somato, l1_pre_dn_sez, l1_pre_dn_vnc, l1_rgn, l1_sensory`.
- These are larval CATMAID cell-type classes. They are **not neuropils**, and they are never mapped to FlyWire, hemibrain, BANC, MANC or optic-lobe vocabularies. `map_to_adult_neuropil()` always raises `CROSS_STAGE_UNSUPPORTED`.
- Cross-stage identity transfer (larval to adult neuron or type identity) is **unsupported**. Payload metadata and every sample's metadata record `cross_stage_identity_transfer: "unsupported"`.
- Names that look shared, such as "KC" or "MBON", refer to larval cells and do not imply homology with adult types.

## 7. Objectives

| Objective | Registry allows | l1em status |
|---|---|---|
| `connectivity_tier` | yes | **Implemented** and registered |
| `region_specialization_tier` | no (narrowed at integration) | **Unsupported**; the registry lists only `connectivity_tier` for l1em, so dispatch gives `OBJECTIVE_NOT_SUPPORTED`. This release has no per-synapse larval neuropil assignment. The only region key is cell type, which is also the routing key, so any label based on it would be constant within an expert or recoverable from the input. The roadmap review makes the same point. |
| `neurotransmitter_dominance` | no | **Unsupported.** Data S1 to S4 have no per-neuron NT annotations or predictions with provenance, and dispatch gives `OBJECTIVE_NOT_SUPPORTED`. |

### 7.1 `connectivity_tier` definition

- Unit: one annotated neuron (skid) in the all-all matrix.
- Candidates: annotated neurons with `total_synapses >= min_total_synapses`. The default is 10. `total_synapses` is the row sum plus the column sum of the all-all matrix.
- Label: `high_connectivity` if `total_synapses >= quantile(candidates.total_synapses, q)`, otherwise `baseline_connectivity`. The default `q` is 0.75, and the quantile uses linear interpolation (the same as numpy and pandas).
- Confidence: the same formula as fw, `clip(0.55 + 0.4 * |total - thr| / max(total, thr, 1), 0.5, 0.99)`.
- Regions: at least `min_region_samples` per region (default 10), with at most `max_regions` regions (default 20), ranked by size and then by name. Samples are ordered by `(region_id, skid)` and taken round-robin through the shared `balance_and_cap_samples`.
- Label gates come from the shared `assemble_training_payload`: `min_distinct_labels = 2` and `max_label_share = 0.9`.
- Real snapshot result: 2952 source rows, 2527 candidates, 2527 selected across 17 regions. The threshold is 348 synapses. Label counts are 1894 baseline and 633 high, so the dominant share is 0.750.

### 7.2 Label hygiene and leakage check

The model input `input_text` is built only from `L1EM_INPUT_FEATURES`:

`celltype, annotation, hemisphere, paired, cluster, axon_output_fraction, axon_input_fraction`

The format is `dataset catmaid_l1em stage larva_l1 celltype <..> annotation <..> hemisphere <left|right> paired <yes|no> cluster <level_7_cluster> axon_output_fraction <x.xxxx|na> axon_input_fraction <x.xxxx|na>`.

`L1EM_FORBIDDEN_INPUT_FEATURES` never reaches `input_text`: `total_synapses, in_synapses, out_synapses, axon_output, dendrite_output, axon_input, dendrite_input, partner_count, skid`. This is enforced in three ways:

1. An import-time assertion that the two feature sets do not overlap.
2. `render_input_text()` rejects any feature set other than exactly `L1EM_INPUT_FEATURES` and raises `SCHEMA_MISMATCH`.
3. The tests check the key order, check that no forbidden key appears, and check that no raw synapse count appears as a value.

The raw axon and dendrite counts in `inputs.csv` and `outputs.csv` nearly sum to the label quantity, so they are forbidden. Only their scale-free fractions are used.

Leakage evidence on the real snapshot (2527 samples):

- The fractions are only weakly monotone with the label quantity. The Spearman correlation with `total_synapses` is -0.32 for `axon_output_fraction` and -0.18 for `axon_input_fraction`.
- In-sample group-majority accuracy is an upper bound on what a lookup table can reach, not a held-out score. The majority class alone gives 0.750, cell type 0.819, `level_7_cluster` 0.894, and the annotation string 0.895. No single feature determines the label.
- `annotation` and `cluster` values are shared within homologous left/right pairs, and they are often near-unique per pair. Their high in-sample score comes from that memorization. It is not label derivation.
- **Evaluation must split by pair.** Each sample's `metadata.split_group` is `l1em-pair-<min(skid, pair_skid)>`. The grouped-split trainer (tracked in the roadmap) should use it. A random split would put left/right homologs on both sides of the split.
- `level_7_cluster` comes from the paper's connectivity-based spectral clustering. It encodes connection pattern, not synapse volume. It is kept, but it must be reported as a caveat on any score. The planned A/B should include a variant that drops it.

## 8. Payload

The builder returns `assemble_training_payload(...)`, so the payload passes `validate_training_payload`. The schema is `flybrain-l1em-training-samples/v1`. Beyond the shared keys, the metadata includes `dataset_version`, `stage`, `region_vocabulary`, `cross_stage_identity_transfer`, `manifest_id`, `manifest_sha256`, `license`, `citation`, `annotation_rows`, `unannotated_neurons_excluded`, `high_connectivity_threshold`, `label_definition`, `input_features`, `forbidden_input_features`, and every threshold from the config. `input_fingerprint` hashes the manifest sha256, every config value that affects output, and the input feature list.

Each sample's `provenance_refs` contains `catmaid_l1em:skid:<skid>`, `source:all-all_connectivity_matrix.csv` and `manifest_sha256:<sha>`.

## 9. Usage

```python
from flybrain_brain_cluster_samples import build_training_samples
from flybrain_brain_cluster_l1em_samples import L1emSampleBuildConfig

payload = build_training_samples(
    "l1em", "connectivity_tier",
    L1emSampleBuildConfig(storage_root=r"<root>"),
    allow_planned=True,  # until the registry marks l1em active
)
```

## 10. Promotion gates (not met yet)

- SC3: at least 2 calibrated threshold reports for (l1em, connectivity_tier).
- A grouped-split (by pair) baseline must beat the majority baseline (0.750) and the cell-type-only rule, and the results must be recorded.
- Do not redistribute derived models without attributing Winding et al. 2023 under CC-BY-4.0.

## 11. Real models

Modules: `mcp/flybrain_brain_cluster_l1em_samples.py` (structured features; the legacy text builder above is unchanged) and `mcp/flybrain_l1em_targets.py` (targets and runner). Tests: `mcp/tests/test_flybrain_l1em_real_models.py`, which runs offline on a synthetic snapshot.
Run: `python -m flybrain_l1em_targets [--targets ...] [--repeats 4]` from `mcp/`. Reports go to `<report-root>/l1em/<target>/[<run_label>/]report.{json,md}` with saved models under `models/`. The run also writes `l1em/l1em_real_models.json` and `l1em/SUMMARY.md`. Run of 2026-09-24: `/mnt/f/.flybrain/logs/real-models-20260924T174122Z/l1em/`.

### 11.1 Which labels can be targets

The paper defines most S2 cell types by connectivity, so a wiring model would only re-derive those definitions. They are rejected as targets (`REJECTED_TARGETS`), and no feature model reads them:

| Candidate | Verdict | Why (Supplementary Material) |
|---|---|---|
| `celltype` (all 17 classes) | **rejected** | Fig. S4C gives each interneuron class a connectivity-motif definition. PN is postsynaptic to SN. PN-somato is postsynaptic to AN. LN is local within a sensory cohort (Fig. S10). LHN is downstream of olfactory, thermo or visual PNs. CN is downstream of LHN and MBON. MB-FBN and MB-FFN are defined by feedback or feedforward around MBIN and MBON. pre-DN-SEZ and pre-DN-VNC are presynaptic to the DNs. The KC and MBON motifs are also connectivity-based. |
| `level_7_cluster` | **rejected** | This is the paper's spectral clustering of the connectivity graph (Fig. S8). The earlier "cluster" lookup was circular. |
| ascending-neuron modality | **rejected** | Table S1 says the AN modalities are "based on connectivity". |
| `l1em_io_class` | **kept** | Brain input and output classes are defined by anatomy. SN enters through a nerve. AN enters from the VNC. DN-SEZ, DN-VNC and RGN are defined by where their axon outputs land: SEZ, VNC or ring gland (Fig. S3C). Every other class is `interneuron`. |
| `l1em_sensory_modality` | **kept** | SN modality comes from the sense organ or nerve and was "previously described for sensory neurons". thermo-cold and thermo-warm are merged. Classes with fewer than 10 neurons are dropped (none are dropped on the real data). |
| `connectivity_tier` | kept, weak | This is the legacy label (total synapses >= q0.75). It is only meaningful under the exclusions in 11.2. |

Caveat for `l1em_io_class`: S2 collapses overlapping classes by priority, so a DN-SEZ or RGN that also meets a higher-priority interneuron motif is labelled `interneuron`. That is label noise, not leakage.

### 11.2 Features and leakage controls

- **Wiring** (`flybrain_wiring_features` over `all-all` exported to `cache/l1em/edges/all_all-<key>.parquet`, keyed by `manifest_sha256`). Families: `degree`, `out_comp`/`in_comp`, `recip`, and `out2_comp`/`in2_comp` (io_class and connectivity_tier). The partner category is the **io class**. A connectivity-defined class is never used, because it would leak the class it is defined against: for example, `pre-DN-*` would leak DN.
- **l1em-only**: `etype_out`/`etype_in` give the aa/ad/da/dd share of output and input synapses. The four split matrices must be listed in the manifest and must sum to all-all exactly, or the run fails closed. `cmpt` holds the axon share of output and input. `ase` holds 8 + 8 adjacency-spectral-embedding coordinates of log1p(all-all). This embedding is label-free, uses ARPACK with a fixed start vector, and fixes signs.
- **Exclusions**: `l1em_io_class` and `l1em_sensory_modality` are registered in `fwf` with the `cell_class` patterns plus `*celltype*`, `*annotation*`, `*modality*`, `*cluster*`, `*io_class*`, `*pair_skid*`, `*hemisphere*` and `*skid*`. `connectivity_tier` uses the registered degree exclusions, which remove all 13 `degree__*` columns. It also drops `ase`, because the embedding norm tracks degree. `proxy_audit` fails closed if any remaining feature has |Spearman rho| >= 0.9 with the total synapse count. The real maximum is 0.544, for `out2_comp__interneuron`.
- **Masking**: `l1em_io_class` uses its own label as the partner category. The split is planned first (`plan_grouped_split`). The io class of every val and test neuron is then hidden (`mask_category_ids`, 758 nodes on the primary split), and `notes.masked_split_ids_sha256` pins the split. A unit test shows that permuting held-out labels changes no feature, and that the unmasked build would change features.
- **Text view for NB**: `binned_text` uses train-fitted quantile bins in the form `key key_q<b>`, so the NB tokenizer keeps the key with the value. The `legacy-text` run uses the old `input_text` for comparison.
- **Protocol**: every run uses `flybrain_model_eval.run_evaluation`. The split is grouped by homologous pair (`split_group`), 70/15/15. Tuning uses 5-fold grouped CV inside train, scored by log loss. Probabilities are calibrated with temperature scaling (multiclass) or isotonic regression (binary) on out-of-fold predictions. The test set is used once. The run reports cluster-bootstrap CIs, the label shuffle, a random-split control, and drop-one-family ablation on val. Because the data are small, each target is re-run on 4 more split seeds.

### 11.3 Results (held-out test, primary split `flybrain-real-models-v1`)

| target | model | majority | best trivial (rule) | acc [95% CI] | macro-F1 | ECE | shuffle | gate | 5-seed acc (mean +- sd) | gate passes |
|---|---|---|---|---|---|---|---|---|---|---|
| l1em_io_class | nb | 0.697 | 0.813 (text lookup) | 0.863 [0.822, 0.901] | 0.706 | 0.062 | 0.642 | fail | 0.857 +- 0.024 | 1/5 |
| l1em_io_class | logreg | 0.697 | 0.805 (lookup `cmpt__axon_input_share`) | 0.932 [0.899, 0.962] | 0.767 | 0.036 | 0.697 | **pass** | 0.940 +- 0.017 | 5/5 |
| l1em_io_class | hgb | 0.697 | 0.805 (same) | 0.945 [0.916, 0.969] | 0.846 | 0.028 | 0.697 | **pass** | 0.952 +- 0.011 | 5/5 |
| l1em_sensory_modality | nb | 0.241 | 0.519 (text lookup) | 0.611 [0.444, 0.768] | 0.541 | 0.139 | 0.111 | fail | 0.643 +- 0.083 | 0/5 |
| l1em_sensory_modality | logreg | 0.241 | 0.426 (lookup `degree__log1p_out_n_annotated_partners`) | 0.704 [0.564, 0.839] | 0.636 | 0.143 | 0.167 | **pass** | 0.661 +- 0.043 | 2/5 |
| l1em_sensory_modality | hgb | 0.241 | 0.426 (same) | 0.759 [0.621, 0.887] | 0.738 | 0.171 | 0.241 | **pass** | 0.788 +- 0.039 | 4/5 |
| connectivity_tier | nb | 0.711 | 0.766 (text lookup) | 0.813 [0.761, 0.860] | 0.772 | 0.032 | 0.687 | fail | 0.811 +- 0.020 | 2/5 |
| connectivity_tier | logreg | 0.711 | 0.726 (lookup `recip__partner_frac`) | 0.826 [0.775, 0.873] | 0.754 | 0.056 | 0.705 | **pass** | 0.821 +- 0.016 | 5/5 |
| connectivity_tier | hgb | 0.711 | 0.726 (same) | 0.892 [0.853, 0.928] | 0.861 | 0.059 | 0.711 | **pass** | 0.876 +- 0.021 | 5/5 |

Sizes: io_class and connectivity_tier have 1769 train, 378 val and 380 test samples, with 209 test components. sensory_modality has 247 train, 53 val and 54 test samples, with 32 test components.

### 11.4 Skeptic's notes

- **Controls behave.** For io_class and connectivity_tier, every label-shuffle run, on all 5 seeds, collapses to the majority rate or below. For sensory_modality, a shuffled model lands near chance (0.09 to 0.24). That is above the tiny test-majority rate on some seeds, as explained below. Accuracy on a random split is close to the grouped result (io_class hgb 0.945 vs 0.945; connectivity_tier hgb 0.887 vs 0.892), so pair grouping is not hiding a large leak.
- **io_class is real but anatomically easy.** Sensory neurons receive almost only axonic input inside the brain, and DNs send most of their output out of it. A NaN-tolerant single-feature stump on val (`cmpt__axon_input_share`) reaches 0.839, against 0.944 for logreg. No single family carries the model: the largest drop in the val ablation is 0.016, for `etype_out`. The model also passes when the split is grouped by pair **and** level-7 cluster (logreg and hgb 0.940 vs 0.826 trivial). That split has only 11 test components, so its CI is wide.
- **sensory_modality is small.** The signal comes mostly from `ase` (-0.075 val accuracy when dropped) and `degree` (-0.057). The data are few, so calibration is weak (ECE 0.14 to 0.17). The single hgb seed that failed the gate (repeat 1: 0.778 vs 0.444) failed **only** on the shuffle criterion. Its test-majority rate is 0.13 with 7 classes, so a shuffled model at chance (0.185) exceeds majority + 0.02. See `shared_edits_needed`. The pair + cluster grouping is **degenerate** for this target: SN clusters are modality-pure (Cramer's V 0.71 in the paper), and only 3 test components remain. All models fail there, and that result should not be read as evidence either way. Pairs alone may not group near-identical, unpaired SNs of one organ, which is a residual optimism risk.
- **connectivity_tier signal is not quantization.** Fractions from few synapses leak count through their granularity. The val-only `granularity_probe` (`connectivity_tier/granularity_probe.json`) scores 0.717 on exact-0/1/missing counts alone, the same as majority (0.717). Adding the smallest positive fraction (about 1/count) gives 0.741. The full feature set gives 0.833. The NaN-tolerant stump scores 0.706. The remaining signal is network position: `in2_comp` alone scores 0.783. The harness threshold rule skips columns with any NaN, so the reported feature-view "best trivial" for this target has no stump. The NaN-tolerant stump above shows this does not change the verdict.
- **The legacy text NB (`legacy-text` run) fails the gate.** It scores 0.861 against a 0.903 text lookup, which reads `cluster` and `annotation`. The new feature models are the ones to keep.
- One exploratory diagnostic (hgb on test with feature subsets) ran before the val-only probe was written. No configuration was chosen from it. All numbers quoted above come from val or the harness's single test pass.

### 11.5 Artifacts

`models/<nb|logreg|hgb>/manifest.json` under each target's report directory. Load with `flybrain_learners.load_learner(dir, trusted_root=...)`. Wiring-feature caches live in `/mnt/f/.flybrain/cache/wiring-features/l1em/` and the edge list in `/mnt/f/.flybrain/cache/l1em/edges/`. Nothing is written under `snapshots/`. Attribute Winding et al. 2023 (CC-BY-4.0) when redistributing any derived model.
