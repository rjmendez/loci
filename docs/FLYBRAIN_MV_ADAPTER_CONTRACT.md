# FlyBrain MANC (`mv`) Adapter + Sample Builder Contract

Status: built and checked against the real local snapshot. The registry still lists `mv` as `planned`, so callers must pass `allow_planned=True` until the integrator promotes it.

Modules:

- `mcp/flybrain_mv_adapter.py`: fail-closed snapshot and manifest verification, the local readers, and the ROI and NT vocabularies.
- `mcp/flybrain_brain_cluster_mv_samples.py`: the training-sample builder. It is registered for `connectivity_tier`, `neurotransmitter_dominance` and `region_specialization_tier`.

Tests:

- `mcp/tests/test_flybrain_mv_adapter.py`
- `mcp/tests/test_flybrain_brain_cluster_mv_samples.py`

Both files use small synthetic fixtures. Each also has real-data smoke tests, which are skipped unless `LOCI_FLYBRAIN_STORAGE_ROOT` contains the snapshot. The smoke tests open the snapshot read-only: they use size checks only, and they never write hash stamps.

## 1. Dataset pin and scope

| Field | Value |
|---|---|
| Registry symbol | `mv` (same casing on disk) |
| Version | `manc_v1.0`. The public bucket `gs://flyem-manc-exports` has only a `v1.0/` prefix. MANC v1.2.x is available only through the neuPrint API and was not pulled. |
| Snapshot root | `$LOCI_FLYBRAIN_STORAGE_ROOT/snapshots/mv/manc_v1.0/`, resolved with `snapshot_version_root("mv")` |
| Scope | adult male *D. melanogaster*, ventral nerve cord only, `connectome_structural`, claim tier `T1_dataset_version_specific` |
| License | CC BY 4.0. Evidence: the Janelia FlyEM MANC page says "The MANC is licensed under CC-BY", and the manifest records `license_evidence`. |
| Citation | Takemura et al. 2024, eLife 13:RP97769; Marin et al. 2024, eLife 13:RP97766; Cheong et al. 2024, eLife 13:RP96084. Data: FlyEM MANC v1.0, `gs://flyem-manc-exports/v1.0`. |

## 2. Local products

The manifest lists 17 files, about 12 GB in total. Every listed file is checked for presence, path safety and exact `size_bytes` on each open. Only the products below are opened or hashed by default:

| Role | Snapshot-relative path | Bytes | Used by |
|---|---|---:|---|
| `meta` | `metadata/manc-v1.0-neuron-properties.feather` | 17,188,218 | every objective |
| `edgelist` | `source/manc-traced-adjacencies-v1.0/traced-connections.csv` | 75,262,163 | `neurotransmitter_dominance` only (the out-partner tier input) |

`meta` holds one row per body in the segmentation: 102,369 rows, about 75% of which are untraced fragments with a null `status`. It already carries everything the builders need. That includes the per-neuron neuPrint NT predictions (`predictedNt`, `predictedNtProb`, `nt*Prob`) and the per-ROI `roiInfo` struct. The adapter never opens these files: `Neuprint_Neurons_manc_v1.ftr` (918 MB, every body), the synapse, synapse-set and synapse-connection exports (up to 3.9 GB each), `traced-connections-per-roi.csv`, or the bz2 synapse-partner table. They are still size-checked as manifest entries.

## 3. Adapter guarantees (`flybrain_mv_adapter`)

`open_mv_snapshot(storage_root=None, *, required_roles=("meta", "edgelist"), verify_hashes=None, now=None, use_stamp_cache=True)` checks the following, in this order:

1. The storage root comes from `build_flybrain_harness_layout(create=False)`. A relative or unset root raises `ROOT_NOT_CONFIGURED`. The registry pin must equal the adapter pin (`DATASET_PIN_MISMATCH`), and the snapshot path must stay under the root (`PATH_ESCAPE`).
2. The snapshot directory exists (`SNAPSHOT_MISSING`) and `manifest/manifest.json` parses (`MANIFEST_MISSING` / `MANIFEST_INVALID`).
3. The `manifest/manifest.sha256` sidecar exists (`MANIFEST_MISSING`) and matches `integrity.manifest_sha256` (`INTEGRITY_MISMATCH`). This check runs before any file content is hashed.
4. `validate_mv_manifest` checks, in order:
   - the schema (`fbh-manifest/v1`) and required sections;
   - the dataset pin (`mv` / `manc_v1.0`);
   - the licence, which must be `CC-BY-4.0` (`LICENSE_NOT_PERMITTED`);
   - the scope, which must be adult `ventral_nerve_cord`;
   - `artifact.relative_root`;
   - the verification status, which must be `verified`;
   - the manifest self-hash;
   - the refresh decision (not `rollback`) and the `next_check_due` window (`PROMOTION_STATE_INVALID`).
5. Each file entry: a safe relative path; no duplicates after case-folding; no `.partial`/`.tmp`/`.inprogress` entries; a 64-hex sha256; the file is present; the size is an integer (not a bool) and matches exactly. Every required role must be listed (`PRODUCT_MISSING`).
6. Content hashing runs only after all the checks above pass, and it uses `flybrain_hash_stamps`:
   - `verify_hashes=None` (default): hash only the required products. A hash is skipped when a write-once stamp matches the file's current `(size, mtime_ns)`.
   - `True`: rehash every listed file (about 12 GB).
   - `False`: size checks only.
   - `use_stamp_cache=False`: stamps under `manifest/hash-stamps/` are neither read nor written, so the open is strictly read-only on the snapshot.

`MvSnapshot.hash_verification` records `hashed` / `stamp` / `size_only` for each file.

Readers (local files only; pyarrow; no network):

- `load_mv_neurons(path, statuses=("Traced",))`: an Arrow dataset scan with a `status` filter and column projection. `roiInfo` is not decoded, and untraced bodies are never materialised. `bodyId` is returned as a string, sorted by zero-padded id. It fails closed on missing columns, null or duplicate ids, and non-Arrow files (`SCHEMA_MISMATCH`).
- `load_mv_neuropil_synweights(path, body_ids)`: decodes `roiInfo` for the requested bodies only, with an Arrow `isin` filter. It returns a dense `synweight` matrix over the neuropil ROIs; nerves, the cervical connective and tracts are excluded. Every struct field must be in the ROI vocabulary (`REGION_VOCABULARY_UNKNOWN`). Negative or non-finite weights, and requested bodies that are missing, fail closed.
- `load_mv_out_partner_counts(path)`: parses 3 CSV columns and returns the distinct traced post partners and the summed weight for each `bodyId_pre`. A repeated `(pre, post)` pair, null values, an empty file or missing columns fail closed.
- `check_nt_argmax(frame)`: `predictedNt` must be the argmax of `ntAcetylcholineProb`/`ntGabaProb`/`ntGlutamateProb`/`ntUnknownProb`, and `predictedNtProb` must equal that maximum. Otherwise it raises `SCHEMA_MISMATCH`, or `NT_VOCABULARY_UNKNOWN` for a value outside the vocabulary. On the real data, all 23,042 traced neurons that have a prediction pass.

Error codes (`MvAdapterErrorCode`): `ROOT_NOT_CONFIGURED`, `PATH_ESCAPE`, `DATASET_PIN_MISMATCH`, `SNAPSHOT_MISSING`, `MANIFEST_MISSING`, `MANIFEST_INVALID`, `MANIFEST_EXISTS`, `LICENSE_NOT_PERMITTED`, `INTEGRITY_MISMATCH`, `PROMOTION_STATE_INVALID`, `PRODUCT_MISSING`, `SCHEMA_MISMATCH`, `REGION_VOCABULARY_UNKNOWN`, `NT_VOCABULARY_UNKNOWN`. `MvAdapterError` subclasses `ValueError`, and its message starts with `[CODE]`.

`build_mv_manifest` / `write_mv_manifest` build a manifest with the self-hash applied and write it together with its sidecar. They refuse to overwrite an existing manifest (`MANIFEST_EXISTS`).

## 4. ROI vocabulary (`map_mv_roi`)

The vocabulary is explicit and covers all 61 MANC v1.0 ROIs (`all_ROIs.txt`). The side suffix `(L)`/`(R)` is stripped, and `region_id = vnc_<sanitized base>`:

| Kind | Base names | Example region ids |
|---|---|---|
| neuropil (13 ids) | `LegNp(T1..T3)`, `NTct(UTct-T1)`, `WTct(UTct-T2)`, `HTct(UTct-T3)`, `IntTct`, `LTct`, `ANm`, `mVAC(T1..T3)`, `Ov` | `vnc_legnp_t1`, `vnc_htct_utct_t3`, `vnc_anm` |
| connective | `CV` | excluded from region and share computations |
| tract | `GF` | excluded |
| nerve (18) | `ADMN`, `AbN1-4`, `AbNT`, `CvN`, `DMetaN`, `DProN`, `MesoAN`, `MesoLN`, `MetaLN`, `PDMN`, `PrN`, `ProAN`, `ProCN`, `ProLN`, `VProN` | excluded |

Any other name, such as `NotPrimary` or a BANC-style `ITO_*` label, raises `REGION_VOCABULARY_UNKNOWN`. NT vocabulary: `acetylcholine -> ach`, `gaba -> gaba`, `glutamate -> glut` (shared with fw and BANC). `unknown` is a real class of the MANC classifier but never becomes a label.

## 5. Candidates and regions

- `status == "Traced"` (23,200 bodies). Rows with other statuses are removed by the Arrow scan filter.
- The neuron has a cell `type` (`require_type=True`). This drops 1,536 traced bodies.
- `class` is not `Glia`.
- The neuron has at least one synapse in a neuropil ROI. This drops 14 bodies.
- `region_id` is the neuron's primary neuropil: the neuropil ROI with the largest `roiInfo.synweight`, with the side suffix removed. All 13 neuropil region ids qualify on real data.

## 6. Objectives and label definitions

Each label comes from a measured or classifier quantity. None of them is an annotation that appears in the input.

### 6.1 `connectivity_tier`

The label is `high_connectivity` when `downstream` is at or above the `high_connectivity_quantile` (default 0.75) over all candidates with `downstream >= min_total_count`; otherwise it is `baseline_connectivity`. `downstream` is neuPrint's count of output synaptic connections. On real data the threshold is 2906. The confidence uses fw's distance-from-threshold formula, clipped to [0.5, 0.99]. `label_source` records `downstream` and `pre`.

### 6.2 `neurotransmitter_dominance`

The label is `dominant_<ach|gaba|glut>` from `predictedNt`, the MANC classifier. The confidence is `predictedNtProb` clipped to [0.5, 0.99]. The builder drops `unknown` predictions (91 on real data) and neurons with `pre < min_total_count`.

### 6.3 `region_specialization_tier` (supported, non-circular)

`share` is the primary neuropil ROI's synweight divided by the neuron's total neuropil synweight. It uses the side-specific ROIs, so `LegNp(T1)(L)` and `LegNp(T1)(R)` are distinct. The nerves, `CV` and `GF` are excluded. The label is `region_specialized` when `share >= specialization_share_threshold` (default 0.9); otherwise it is `region_distributed`. Neurons with total neuropil synweight below `min_total_count` are dropped. The confidence uses the distance-from-threshold formula.

This objective is supported for `mv` but not for l1em, because MANC has measured per-neuropil synapse counts for every neuron, independent of any annotation. The MANC annotations that encode how widely a neuron spreads are kept out of the input:

- the intrinsic `subclass` / `prefix` letters (IR, II, BR, BI, CR, CI, ...), where R means restricted and I means intersegmental;
- the `target` / `origin` neuromere lists;
- `longTract`;
- entry and exit nerves.

In an exploratory check on real data, the label was computed with the same 0.9 threshold over all about 21.6k candidates, and a train-to-test lookup was scored on a hemilineage/type-grouped 70/30 split. `target` alone predicted the label at 0.94 held-out accuracy, `subclass` at 0.91 and `origin` at 0.90, against a majority baseline of 0.46. These features would make the task a table lookup.

## 7. Label hygiene: leakage and trivial-baseline checks

The input text is `dataset manc10 <key> <value> ...`, with categorical values only.

| Objective | Input features | Also forbidden (examples) |
|---|---|---|
| `connectivity_tier` | primary_neuropil, soma_neuromere, soma_side, class, birthtime, hemilineage | downstream, upstream, pre, post, synweight, size, n_post_partners, out_partner_tier, roiInfo |
| `neurotransmitter_dominance` | primary_neuropil, soma_neuromere, soma_side, class, birthtime, out_partner_tier | predictedNt(Prob), nt*Prob, transmission, **hemilineage** |
| `region_specialization_tier` | primary_neuropil, soma_neuromere, soma_side, class, birthtime | subclass, prefix, target, origin, long_tract, entry/exit nerve, n_neuropils, **hemilineage** |

All objectives also forbid these identity keys: `root`/`body_id`, `cell_type`/`type`, `instance`, `systematic_type`, `group`, `split_group`.

- **No body id in the input.** Unlike BANC, the mv input carries no `root <id>`. MANC bodyIds are not random: low ids went to large bodies that were proofread early. With the id in the input, the shared `threshold` trivial rule predicted `connectivity_tier` from the id alone at **0.774** held-out accuracy (5,000 samples, grouped split). A threshold on the id also reached 0.76 in-sample for `region_specialization_tier`.
- **Hemilineage is not an input for NT.** Each VNC hemilineage uses one fast transmitter (Lacin et al. 2019). On real data, a hemilineage lookup gets 0.83 in-sample over all typed traced neurons, and 0.93 over the 13,631 neurons with an annotated hemilineage and an ach/gaba/glut prediction. Hemilineage is also kept out of the `region_specialization_tier` input, because hemilineages have stereotyped projection patterns. It stays in sample metadata as a split-group key.
- `assert_no_label_leakage` fails the build if any `input_text` key is forbidden, or is not on the objective's allow-list.
- `trivial_feature_check` runs on every build, after capping. It computes the in-sample accuracy of a per-value majority lookup for each input feature and for `region_id`, which is the expert routing key. It also fits the shared threshold stump (`fit_threshold_rule`). The build fails if the best of these exceeds `max_single_feature_accuracy` (default 0.9). The report is written to `metadata.trivial_feature_check`.

### 7.0 Integrator re-run (current code)

Two changes landed after the measurements in 7.1:

- **Capped subsets are chosen in sha256(bodyId) order** (`hash_ordered_preselect`, fingerprint `cap_order: sha256-body-id/v1`). The 7.1 builds kept each region's lowest bodyIds, which are the large, early-proofread bodies, so they over-sampled high_connectivity (2,059 of 5,000 against the 25% the quantile defines).
- **The promotion gate now has a categorical `lookup` rule**, so the AC6 bar is no longer the majority label for mv's category-only inputs.

Full p0 dry run per objective (default config, 5,000 samples, pipeline grouped split with seed `integrator-real-eval`, held-out = test split, `verify_hashes=False`, `use_stamp_cache=False`):

| Objective | Labels (5,000) | Held-out n | Majority | Gate best (rule, feature) | Model held-out | Trivial-baseline gate | Other p0 gate failures |
|---|---|---:|---:|---|---:|---|---|
| `connectivity_tier` | baseline 3,910 / high 1,090 | 587 | 0.699 | 0.750 (lookup, birthtime) | 0.692 | fail | none |
| `neurotransmitter_dominance` | ach 2,722 / gaba 1,287 / glut 991 | 595 | 0.540 | 0.652 (lookup, class) | 0.603 | fail | accuracy 0.596 < 0.80 |
| `region_specialization_tier` | distributed 3,307 / specialized 1,693 | 588 | 0.755 | 0.881 (lookup, class) | 0.866 | fail | calibration 0.230 > 0.20 |

With an unbiased sample, no mv objective beats the best one-feature lookup, so none passes AC6. The 7.1 conclusion that "the model beats the stronger per-feature lookup on every objective" came from the id-ordered cap and no longer holds. Peak RSS was 0.5 GB (connectivity, region) and 1.5 GB (NT, which reads the edge list).

### 7.1 Numbers on the real snapshot (2026-09-24)

Setup:

- Default config, 5,000 samples after capping.
- Grouped split from the pipeline (`split_group` + registry keys `cell_type`, `hemilineage`), seed `mv-eval`.
- `held-out` means the test split of about 710 samples.
- The model is the p0 region-expert pipeline (`run_brain_cluster_p0_dry_run`, output in a temp dir).
- The best categorical lookup was fitted on train and scored on test. This check is stronger than the shared trivial rules, which only see numeric features.

| Objective | Labels (after cap) | In-sample best single feature | Held-out majority | Held-out best categorical lookup | Model held-out | AC6 gate (majority + 0.01) |
|---|---|---|---|---|---|---|
| `connectivity_tier` | baseline 2941 / high 2059 | birthtime 0.726 | 0.407 | birthtime 0.740 | **0.795** | pass |
| `neurotransmitter_dominance` | ach 2624 / gaba 1413 / glut 963 | class 0.559 | 0.432 | class 0.439 | **0.524** | pass |
| `region_specialization_tier` | distributed 3740 / specialized 1260 | primary_neuropil 0.811 | 0.688 | primary_neuropil 0.756 | **0.891** | pass |

On every objective, the model beats both the shared trivial rules and the stronger per-feature lookup. The p0 golden-set gate still fails on its absolute thresholds:

- `connectivity_tier`: accuracy 0.788 < 0.80.
- `neurotransmitter_dominance`: accuracy 0.638 < 0.80 and calibration 0.220 > 0.20.
- `region_specialization_tier`: calibration 0.256 > 0.20.

That is expected before promotion; see section 9.

Uncapped (all about 21.5k candidates), the in-sample best single feature is birthtime 0.817 (connectivity), out_partner_tier 0.560 (NT) and primary_neuropil 0.774 (region). All are below the 0.9 guard.

`birthtime` is informative for connectivity: primary (embryonic) neurons are larger. This is biology, not leakage: birthtime is an annotation of developmental origin, not a synapse count.

Grouped split: the union over `split_group`, `cell_type` and `hemilineage` yields about 380 components on 5k samples, and the largest has about 305 samples. The per-sample split would have straddled about 190 of those groups. `hemilineage` placeholders (`TBD`, `None`, `Unknown`, `NA`) are normalised to `unknown` so that they never tie neurons together. `split_group` is `manc_group_<group>`, from MANC's left/right and serial homolog `group` column.

## 8. Provenance in the payload metadata

These fields come in addition to the shared payload keys:

- `dataset_version`, `region_vocabulary` (`manc_neuropil`), `input_features`, `forbidden_input_features`, `leakage_check`, `trivial_feature_check`.
- `high_connectivity_threshold` / `specialization_share_threshold`.
- `input_paths`, `provenance_mode` (`manifest_snapshot` | `explicit_paths`), `manifest_id`, `manifest_sha256`, `product_sha256`, `hash_verification`.
- `filter_stats` (`dropped_untyped`, `dropped_excluded_class`, `dropped_no_neuropil_synapses`, `dropped_nt_unknown`), `license`.

Each sample's metadata carries `root_id` (the bodyId as a string), `cell_type`, `hemilineage`, `split_group`, `primary_neuropil_roi`, `task_type`, `risk_tier` and `label_source`. The input fingerprint covers the product hashes, the manifest hash, every config value that affects output, and `INPUT_TEXT_FORMAT` (`mv-input-text/v1`). The output is byte-identical across runs.

## 9. Operational notes

- CLI: `python mcp/flybrain_brain_cluster_mv_samples.py --objective <obj> --output <file> --allow-planned [--storage-root R] [--no-stamp-cache] [--max-single-feature-accuracy X]`. Errors are printed as JSON with exit code 2.
- A default open hashes at most 92 MB (meta and edgelist); this took about 0.5 s on the real snapshot. Later opens use stamps. A full build took 2 to 5 s per objective.
- The multi-GB neuPrint synapse exports are listed and size-checked, but no code path reads them. Adding a reader for them needs its own column-projected, filtered design.
- Before promotion (registry `planned` to `active`), these still apply: SC3 thresholds per (dataset, objective), the grouped-split trivial-baseline gate (AC6), the p0 golden-set accuracy and calibration thresholds, and the AB plan AC1-AC5.

## 10. Real models (grouped, gated, calibrated)

Run date: 2026-09-24. Run label: `wiring-v1`. Each report is at `/mnt/f/.flybrain/logs/real-models-20260924T174122Z/mv/<target>/wiring_v1/report.{json,md}`, and the saved models are in `models/<name>/`.

### 10.1 Code

- `mcp/flybrain_brain_cluster_mv_samples.py` gains the pieces below. Existing callers are unaffected: with defaults, payloads and fingerprints are byte-identical.
  - `STRUCTURED_FEATURES[objective]`: the categorical inputs for the feature learners. It is a subset of the legacy input keys: `out_partner_tier` is removed everywhere, and `primary_neuropil` is removed for region specialization.
  - `MvSampleBuildConfig.attach_features` (default `False`) puts those inputs in `sample["features"]`.
  - `load_mv_candidates(config)` and `collect_mv_samples(objective, config)` return every labelled candidate, with no cap and no balancing. The majority rate is therefore the true class prior.
- `mcp/flybrain_mv_targets.py` (new) builds the `EvalDataset` for each target and runs `flybrain_model_eval.run_evaluation`. CLI: `python mcp/flybrain_mv_targets.py --target <t> [--run-label L] [--families degree,out_comp,...] [--quick] [--summary out.jsonl]`.
- Snapshot access is read-only:
  - Inputs are opened with `open_mv_snapshot(required_roles=(meta, edgelist), use_stamp_cache=False)`, so nothing is written into `snapshots/`.
  - The per-ROI edge list (`traced-connections-per-roi.csv`) has no adapter role. Before it is read, its sha256 is checked against its entry (role `edgelist_per_roi`) in the verified manifest.
- Wiring features come from `flybrain_wiring_features`, run on `traced-connections.csv`:
  - Node table: all 23,200 Traced bodies. Partner category: harmonized class.
  - Families: `degree`, `out_comp`/`in_comp`, `out2_comp`/`in2_comp` (two-hop, with the return path removed), `recip` and `out_np`/`in_np`.
  - The neuropil families use a derived table, cached under `/mnt/f/.flybrain/cache/mv-derived/`. It keeps only the 13 VNC base neuropils, with the side stripped. `CV`, `GF`, the nerves and `NotPrimary` are dropped, so "passes through the neck connective or a nerve" never becomes a feature.
- Two tagged composition sets are added per target: `*_comp_hl` (partner hemilineage) for the hemilineage target, and `*_comp_pnt` (partner predicted NT) for the NT target. When the partner category is the label, or determines it, the set is masked for the val and test samples and for every node that shares a held-out cell type, homolog `group`, `serial` set or hemilineage. The notes record `masked_split_ids_sha256`, which `run_evaluation` checks.
- NB view: fused `key key_<value>` tokens. Numeric features are cut at deciles fitted on the train split only. The legacy NB tokenizer splits `key value` into separate tokens, so without fusing, the numeric value tokens would be shared across all features.
- `report["mv_shuffle_null"]`: a supplementary label-shuffle control (see 10.4). It is reported next to the harness gate and does not replace it.

### 10.2 Targets, labels and exclusions

| Target | Label (independent of inputs) | n | Split keys (union-find) | Never an input (beyond the registered `fwf` patterns and ids/type/group/serial) |
|---|---|---:|---|---|
| `cell_class` | curated MANC `class`, harmonized (6 classes) | 21,650 | cell_type, hemilineage, group, serial | any annotation. somaNeuromere/somaSide/birthtime are null exactly for DNs and sensory neurons. Also excluded: nerve/CV/GF synapses, subclass, modality |
| `hemilineage` | developmental hemilineage (lineage ground truth). 34 classes, each with >= 50 neurons and >= 4 types | 13,381 | cell_type, group, serial (a target cannot group on its own label) | class, subclass, partner predicted NT, NT scores |
| `neurotransmitter_dominance` | MANC `predictedNt` (ach/gaba/glut). This is a classifier output: MANC v1.0 has no NT ground-truth column | 21,151 | cell_type, hemilineage, group, serial | hemilineage, partner-hemilineage composition, NT probabilities, transmission |
| `connectivity_tier` | `downstream` >= q75 (definition unchanged) | all candidates | same as NT | the whole `degree` family, `*n_neuropils*`, pre/post/upstream/downstream/size |
| `region_specialization_tier` | top side-specific neuropil share >= 0.9 (definition unchanged) | all candidates | same as NT | every `*_np__*` column, primary_neuropil, subclass/target/origin/long tract/nerves, hemilineage |

Models:

- Backends: nb (alpha grid), logreg (C in {0.1, 1}) and hgb (two configs; one for hemilineage).
- Tuning: grouped K-fold inside train on macro-F1 (5-fold; 3-fold for hemilineage).
- Calibration: temperature scaling, fitted on grouped out-of-fold probabilities.
- Test: scored once per final config, with 1,000-draw cluster-bootstrap CIs.

### 10.3 Results (held-out grouped test split)

"Best trivial" is the best rule in the model's own view, which is what the harness gates on. Rules named `features:*` apply to logreg and hgb. Rules named `text:*` are lookups over the binned tokens and apply to nb.

| Target | Model | Majority | Best trivial (rule) | Test acc [95% CI] | Macro-F1 | ECE | Shuffle | Gate |
|---|---|---:|---|---|---:|---:|---:|---|
| cell_class | logreg | 0.675 | 0.731 (threshold) | **0.891** [0.808, 0.948] | 0.668 | 0.021 | 0.675 | pass |
| cell_class | hgb | 0.675 | 0.731 (threshold) | 0.877 [0.759, 0.951] | 0.642 | 0.034 | 0.675 | pass |
| cell_class | nb | 0.675 | 0.765 (text lookup) | 0.685 [0.520, 0.837] | 0.536 | 0.085 | 0.590 | fail |
| hemilineage | logreg | 0.029 | 0.097 (lookup) | **0.493** [0.433, 0.555] | 0.461 | 0.031 | 0.045 | pass |
| hemilineage | hgb | 0.029 | 0.097 (lookup) | 0.365 [0.318, 0.415] | 0.340 | 0.103 | 0.032 | pass |
| hemilineage | nb | 0.029 | 0.135 (text lookup) | 0.366 [0.302, 0.433] | 0.324 | 0.068 | 0.059 | fail (harness shuffle) |
| neurotransmitter_dominance | logreg | 0.520 | 0.532 (lookup) | 0.580 [0.423, 0.734] | 0.499 | 0.052 | 0.571 | fail |
| neurotransmitter_dominance | hgb | 0.520 | 0.532 (lookup) | 0.560 [0.366, 0.756] | 0.431 | 0.095 | 0.520 | fail |
| neurotransmitter_dominance | nb | 0.520 | 0.595 (text lookup) | 0.349 [0.207, 0.574] | 0.317 | 0.145 | 0.414 | fail |

The full connectivity_tier and region_specialization_tier runs were queued behind the shared heavy-job lock when this was written; their reports land in the same tree. A preliminary logreg-only check was run outside the lock (<5 min, 2-fold CV, no ablation, no report written) on the same split:

- connectivity_tier: 0.829 [0.762, 0.903] vs lookup 0.715 (majority 0.633). The shuffle control collapsed.
- region_specialization_tier: 0.805 [0.727, 0.873] vs threshold 0.700 (majority 0.400). The harness shuffle check fails here, with a shuffle score of 0.46 against a bar of 0.42; the prior-shift effect in 10.4 explains this.

### 10.4 Reading the results

**`cell_class` passes. Most of the signal is polarity and reciprocity.**

- The best single rule is a threshold on in-degree, or a lookup on the binned out/(in+out) weight ratio, which scores 0.765 in the text view. Sensory neurons get almost no VNC input, and motor neurons make no VNC output.
- logreg still beats that rule by 0.13 (paired-bootstrap gain CI [0.07, 0.34]) and roughly doubles its macro-F1 (0.67 vs 0.29).
- Ablation (hgb, scored on val): dropping `recip` costs 0.26 and dropping `degree` costs 0.08, while dropping `out_comp` improves accuracy by 0.08.
- A random per-neuron split reaches 0.94-0.97. The gap to the grouped split is the optimism that grouping removes.

**`hemilineage` from wiring is a real, non-trivial result.**

- Baselines: 34 classes, majority 0.03, best one-feature lookup 0.10.
- logreg reaches 0.49 (CI [0.43, 0.56]) on cell types it never saw, with ECE 0.03 after temperature scaling.
- The signal is spread across families. The largest ablation drop is the masked partner-hemilineage composition (`out_comp_hl`, -0.04); neuropil and degree each cost about 0.01.
- A random split reaches 0.84, so most of the easy accuracy there comes from same-type siblings.

**`neurotransmitter_dominance` (predicted transmitter) does not beat the trivial rules. This is a negative result.**

- The split groups by hemilineage, and each hemilineage uses one transmitter, so every test hemilineage is unseen in training.
- The class prior moves between splits: val has 39 GABA neurons and test has 453. Only 145 test components exist, dominated by a few large hemilineages, so the CI is about ±0.15.
- hgb looks strong on val (0.84) but reaches only 0.56 on test.
- The legacy lookup-by-class result (section 7.0: class lookup 0.652 beat the NB expert's 0.603) comes from the capped, balanced sample. In the uncapped, prior-true sample, class is not even the best single feature: the best is a lookup on a binned out-partner count, at 0.53.
- No wiring model generalises NT to new hemilineages above the trivial rules. Dale-type homophily (the masked `*_comp_pnt` features) adds nothing in ablation (-0.001).
- On a random split, hgb reaches 0.87. The earlier model-beats-baseline numbers came from same-hemilineage leakage.

**The harness shuffle check misfires under prior shift.**

- The check compares one shuffled refit against the train-majority label scored on test. When the test prior differs from train, a shuffled model that spreads its predictions can beat that label by chance: NT logreg scores 0.571 against 0.520, and hemilineage nb 0.059 against 0.029.
- `mv_shuffle_null` refits 3 permutations per model. Its bar is `max(majority, sum_c q_c p_c)`: the accuracy of feature-independent guessing with the same output mix.
- Every model collapses to within 0.02 of that bar, so these shuffle failures are not leakage. The harness gate is still reported unchanged.
