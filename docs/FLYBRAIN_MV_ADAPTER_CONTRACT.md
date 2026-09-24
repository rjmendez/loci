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
