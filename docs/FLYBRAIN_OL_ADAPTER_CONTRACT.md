# FlyBrain Optic-Lobe (ol) Adapter + Sample Builder Contract

Status: built and checked against the real local snapshot (read-only). The registry still lists `ol` as `planned`, so callers must pass `allow_planned=True` until the integrator promotes it.

Modules:

- `mcp/flybrain_ol_adapter.py`: fail-closed snapshot/manifest verification, local readers, and the ROI and NT vocabularies
- `mcp/flybrain_brain_cluster_ol_samples.py`: training-sample builder, registered for `connectivity_tier` and `neurotransmitter_dominance`, plus the grouped-split derivability audit

Tests:

- `mcp/tests/test_flybrain_ol_adapter.py`
- `mcp/tests/test_flybrain_brain_cluster_ol_samples.py`

Both use tiny synthetic fixtures (a 15-row neuPrint `Neuron` feather and a one-row `Meta` CSV). Each also has a smoke test that is skipped unless `LOCI_FLYBRAIN_STORAGE_ROOT` contains the real snapshot. The smoke tests open with `verify_hashes=False`, so they never write hash stamps into the real snapshot.

## 1. Dataset pin and scope

| Field | Value |
|---|---|
| Registry symbol | `ol` |
| Version | `optic_lobe_v1.1` (neuPrint `optic-lobe:v1.1`; flat export `2024-09-11-a7d912`, minconf 0.5) |
| Snapshot root | `$LOCI_FLYBRAIN_STORAGE_ROOT\snapshots\ol\optic_lobe_v1.1\` (resolved via `snapshot_version_root("ol")`) |
| Scope | adult male *D. melanogaster*, right optic lobe, `connectome_structural`, claim tier `T1_dataset_version_specific` |
| License | CC BY 4.0 (manifest `license_evidence`: Janelia FlyEM optic-lobe project page) |
| Citation | Nern A, Loesche F, Takemura S, Burnett LE, Dreher M, et al. (2025). *Connectome-driven neural inventory of a complete visual system.* Nature 641:1225-1237. doi:10.1038/s41586-025-08746-0 |

`ol` images the right optic lobe of the same male specimen later released as `mc` (male-cns v1.0). The adapter never joins `ol` rows to `mc`; body ids and types can differ between the two releases.

## 2. Products used

The pull job (manifest `fbh-ol-optic_lobe_v1.1-2026-09-24T01:32:01Z`, sidecar `c072d711...b806`) lists 15 files. The builders need two of them:

| Role (= manifest `role`) | Snapshot-relative path | Bytes | sha256 |
|---|---|---:|---|
| `neuron_annotations` | `metadata/Neuprint_Neurons.feather` | 687,066,010 | `a95962efd13a35a771f7713208bab2f5061aba1d7bbe43d1d068ba796dc8d991` |
| `neuprint_meta` | `metadata/Neuprint_Meta.csv` | 580,574 | `1719cccbd0842e35bee3d7d6906a35ced0ccbaf36e64ad5fcee0da5ee5ef5417` |

Both sha256 values were recomputed from the bytes on disk during the explicit-path real builds (2026-09-24) and match the manifest.

The other 13 listed files are size-checked on every open and hashed only with `verify_hashes=True`. They include the flat-connectome weights, the synapse partner and point tables (up to 1.6 GB each), and the per-ROI connection table. No builder reads them.

`Neuprint_Neurons.feather` holds 10,314,236 bodies, and only 53,987 of them have a `type`. `load_ol_neurons` never loads the whole table. It reads the schema first, then runs an Arrow dataset scan with:

- column projection to the 15 `NEURON_COLUMNS`
- a pushed-down filter: `type` is non-null and non-empty, and `status` is in `allowed_statuses`

A real build peaks at about 1 GB RSS. It takes about 20 s, plus the sha256 of the 687 MB file the first time a file state is opened in `manifest_snapshot` mode (that result is then stamped).

## 3. Adapter guarantees (`flybrain_ol_adapter`)

`open_ol_snapshot(storage_root=None, *, required_roles=..., verify_hashes=None)` returns an `OlSnapshot` or raises `OlAdapterError`. The error has `.code` set to an `OlAdapterErrorCode`, and its message is prefixed `[CODE] `. The check order and the `verify_hashes` semantics are identical to BANC (`docs/FLYBRAIN_BANC_ADAPTER_CONTRACT.md` §3):

1. The manifest schema and the dataset pins.
2. The manifest self-hash.
3. The `manifest.sha256` sidecar. A missing sidecar fails closed.
4. Path safety and the exact `size_bytes` of every listed file.
5. Content hashing through `flybrain_hash_stamps`.

`OlSnapshot.hash_verification` records `hashed`, `stamp` or `size_only` for each file.

The ol adapter adds one gate that BANC does not have. If a manifest entry sits at one of the adapter's product paths and its `role` differs from the adapter's role for that path, the open fails with `MANIFEST_INVALID`. This catches a manifest that points a product at the wrong file.

| Code | Raised when |
|---|---|
| `ROOT_NOT_CONFIGURED` | The storage root is missing, relative, UNC, a drive root, or crosses a symlink or reparse point. |
| `PATH_ESCAPE` | `artifact.relative_root` or a file path is unsafe, or resolves outside the snapshot. |
| `DATASET_PIN_MISMATCH` | The manifest symbol, version or relative_root is not `ol`/`optic_lobe_v1.1`/`snapshots/ol/optic_lobe_v1.1`; the registry pin has drifted; or the neuPrint `Meta` row is not `optic-lobe` / `v1.1`. |
| `SNAPSHOT_MISSING` / `MANIFEST_MISSING` | The snapshot directory, `manifest.json` or the sidecar is absent. |
| `MANIFEST_INVALID` | A field is missing or invalid, a path is a duplicate after case-folding, a `.partial` file is listed, a sha is malformed, a size is not an integer (bools are rejected), or a product role does not match. |
| `LICENSE_NOT_PERMITTED` | The SPDX id is not `CC-BY-4.0`. |
| `INTEGRITY_MISMATCH` | The status is not `verified`, the self-hash or sidecar is wrong, or a size or sha256 does not match. |
| `PROMOTION_STATE_INVALID` | `refresh.decision == "rollback"`, or `next_check_due` has passed (the real manifest is due 2026-12-23). |
| `PRODUCT_MISSING` | A role is unknown, a required product is not listed, or a listed or explicit file does not exist. |
| `SCHEMA_MISMATCH` | A required column is missing or renamed (raw neuPrint names with type suffixes, such as `bodyId:long`, are matched exactly); the file is not Arrow IPC; `bodyId` is null or duplicated; `roiInfo` is not a JSON object; a count is negative, fractional or a string; the Meta CSV does not have exactly one row; or `predictedNtConfidence` is outside [0, 1]. |
| `REGION_VOCABULARY_UNKNOWN` | A `roiInfo` key or a primary ROI is outside the explicit vocabulary, or the release's `primaryRois` differ from it. |
| `NT_VOCABULARY_UNKNOWN` | `predictedNt` is not one of the 8 classes. `unclear` is not a class: the builder drops it first. |
| `MANIFEST_EXISTS` | `write_ol_manifest` would overwrite an existing manifest. |

Body ids are uint64 in the file. They are cast to strings, so 20-digit ids stay exact. There is no network access, and a test asserts that neither module imports an HTTP or socket library.

## 4. ROI vocabulary (explicit)

`OL_PRIMARY_ROIS` is the release's `primaryRois` list: 89 names from `Neuprint_Meta.csv`. `load_ol_release_meta` fails closed if the release list differs. `map_ol_roi` maps each name as follows:

| Primary ROI | `division` | `region_id` example |
|---|---|---|
| `ME`, `LO`, `LOP`, `AME`, `LA` with `(R)` or `(L)` | `optic_lobe` | `ME(R)` becomes `ol_me_r`; `LO(L)` becomes `ol_lo_l` |
| `vnc-shell` | `ventral_nerve_cord` | `vnc_vnc_shell` |
| every other primary ROI | `central_brain` | `PLP(R)` becomes `cb_plp_r`; `GNG` becomes `cb_gng` |

Mushroom-body lobes get an `mb_` prefix and `'` becomes `prime`. So `aL(R)` maps to `cb_mb_al_r` and `a'L(R)` maps to `cb_mb_aprimel_r`, and neither collides with the antennal lobe `AL(R)` (`cb_al_r`). Slug uniqueness is asserted at import.

`parse_roi_info` classifies every `roiInfo` key:

| Key | Treatment |
|---|---|
| a primary ROI | Kept, with its pre/post/synweight counts. |
| `OL(R)`, `OL(L)` | Skipped. These super-level ROIs double-count their children. |
| `(ME\|LO\|LOP)_R_layer_N` | Layer synweight, kept. |
| `(ME\|LO\|LOP)_R_col_X_Y` | Counted toward the column span. |
| anything else | `REGION_VOCABULARY_UNKNOWN`. |

On real data every typed neuron's `roiInfo` falls in these classes.

A sample's `region_id` is the primary ROI with the largest synweight, with ties broken by name. A neuron with no primary-ROI synapses is dropped (`dropped_no_primary_roi`, 0 on real data). This is the synaptic-mass neuropil, like fw, and unlike BANC's root-point neuropil.

## 5. Objectives and label definitions

The builder is `build_ol_training_samples(objective, config)`, with config type `OlSampleBuildConfig`. It is registered at import. The payload goes through `assemble_training_payload`, so fw's label-diversity and label-concentration gates apply. The schema is `flybrain-ol-training-samples/v1`.

**Common filters:**

- The neuron is typed.
- `status` is in `allowed_statuses`, default `("Traced",)`. This drops 2,523 of the 53,987 typed bodies: 1,350 with an empty status ("Out of scope"), 1,137 `Anchor`, 35 `Orphan`, and 1 `Unimportant`.
- It has a primary-ROI synapse.
- Its region passes `min_region_samples` (25) and `max_regions` (20).

**Deterministic subsampling:** fw's `balance_and_cap_samples` takes each region's samples in `sample_id` order. In this release the low body ids are the large, early-traced bodies: bodyId vs `downstream` has Spearman -0.65. An id-ordered cap would therefore oversample big neurons. On the first draft this pushed capped `high_connectivity` from 25% to 39%, and capped ach from 67% to 85%.

The builder therefore pre-selects with `_hash_round_robin`. That is the same round-robin over sorted regions, but within each region it takes samples in `sha256("ol11:<bodyId>")` order. The shared helper then only reorders the result. The output is still fully deterministic.

### 5.1 `connectivity_tier`

- **Measure:** the neuPrint `downstream` count, which is the number of output connections (postsynaptic partner sites) summed over all partners. Neurons below `min_total_count` (10) are dropped (721 on real data).
- **Label:** `high_connectivity` if the measure is at or above the global `high_connectivity_quantile` (0.75) over retained candidates, otherwise `baseline_connectivity`.
- **Confidence:** fw's formula, `clip(0.55 + 0.4 * |x - thr| / max(x, thr, 1), 0.5, 0.99)`.
- **Real build (default config, 2026-09-24):**
  - 50,743 candidates; threshold 1,247.
  - Capped 5,000 samples: 2,634 baseline and 2,366 high (dominant share 0.527).
  - 15 regions: `cb_{aotu,avlp,lal,lh,plp,pvlp,slp,sps}_r`, `cb_ib`, `ol_{la,lo,lop,me}_r`, `ol_lo_l`, `ol_me_l`.
  - By division: 2,764 optic lobe and 2,236 central brain.
  - The round-robin gives the small central-brain regions equal turns, which is why high is about 47% in the capped build versus 25% overall.

### 5.2 `neurotransmitter_dominance`

- **Label:** `dominant_<code>` from the neuron's own synapse-classifier call `predictedNt`.
- **Codes:** `ach, gaba, glut, da, ser, oct, his`, plus `tyr`, which is unused in this release.
- **Filters:** `unclear` calls are dropped (7,126), as are rows with a missing call. `totalNtPredictions` must be at least `min_total_count`.
- **Confidence:** `clip(predictedNtConfidence, 0.5, 0.99)`.
- The type-level calls (`celltypePredictedNt`, `consensusNt`, `ntReference`, `otherNt`) are never read, because they are label-derived per type.
- **Real build:**
  - 44,338 candidates.
  - Capped 5,000 samples: ach 3,791, glut 787, gaba 373, his 33, da 7, oct 6, ser 3 (dominant share 0.758).
  - 6 regions: `cb_plp_r`, `cb_pvlp_r`, `ol_lo_r`, `ol_lop_r`, `ol_me_l`, `ol_me_r`.

### 5.3 `region_specialization_tier`: not built

The registry allow-list for ol is now `connectivity_tier` and `neurotransmitter_dominance` only, so dispatch for this objective fails closed with `OBJECTIVE_NOT_SUPPORTED` (as l1em does). The reasoning below is unchanged.

The only region signal in this release is `roiInfo`, which already defines `region_id` (routing) and the arbor input features. Any "how concentrated is the neuron in its region" label would be a function of the same vector as the inputs and the routing key, which is circular. (l1em narrowed its allow-list for the same reason.)

A non-circular version would need a region signal that is independent of `roiInfo`. One example is soma position against a neuropil atlas, which is not in the pulled tables.

## 6. Label hygiene and leakage check

**Inputs.** `input_text` is `dataset ol11` followed by `key value` pairs, in this fixed order:

| Objective | Input features (`INPUT_FEATURES`) |
|---|---|
| both | `side`, `soma`, `hex_column`, `primary_neuropil`, `input_neuropil`, `output_neuropil`, `arbor`, `dominant_layer`, `hemilineage` |
| NT only | also `polarity_tier` and `column_span_tier` |

What each feature holds:

- `side`: the instance suffix `_R`/`_L`.
- `soma`: whether `somaLocation` is present.
- `hex_column`: whether `assignedOlHex1` is set, which marks columnar types.
- `input_neuropil`, `output_neuropil`: the dominant primary ROI by post count and by pre count.
- `arbor`: the primary ROIs holding at least 10% of synweight, joined with `+`.
- `dominant_layer`: the OL layer with the most synweight, for example `me_r_layer_03`.
- `polarity_tier`: pre/(pre+post) binned at 0.1/0.3.
- `column_span_tier`: the number of distinct column ROIs, binned at 2/10.

Every value is a categorical token. The id and the type are never input:

- **Body id:** it is a size proxy (Spearman -0.65 with the connectivity measure). It stays in `sample_id`, `provenance_refs` and `metadata.body_id`.
- **`type` and `instance`:** these are excluded because the instance name contains the type. `cell_type` is the split-group key. The label is highly type-determined (on real data 99.97% of NT candidates share their type's majority NT call, and 91% of connectivity candidates share their type's majority tier), so type in the input would turn both tasks into type memorisation.

**Forbidden inputs** (`FORBIDDEN_INPUT_FEATURES`):

| Objective | Forbidden keys |
|---|---|
| both | `body`, `body_id`, `root`, `cell_type`, `type`, `instance`, `flywire_type`, `mcns_serial` |
| connectivity | also `downstream`, `upstream`, `pre`, `post`, `synweight`, `size`, `total_nt_predictions` (a presynapse count), `polarity_tier`, `column_span_tier`, `n_columns` (magnitude proxies) |
| NT | also the neuron and cell-type NT fields (`predicted_nt*`, `celltype_*`, `consensus_nt`, `nt_reference`, `other_nt*`, `total_nt_predictions`) |

`assert_no_label_leakage` runs on every sample before capping. It rejects any forbidden or unexpected key, and any odd token count. The payload records `leakage_check: "passed"`.

**Grouped-split derivability audit.** `label_derivability_audit(samples)` splits by `cell_type` with `grouped_split_ids`, the same split as the pipeline, and fits on train. It then scores val and test together, on types that do not appear in train:

- `gate`: the promotion gate's trivial rules (`evaluate_trivial_baselines`). Because ol inputs are all categorical, the threshold and argmax rules never apply. The gate's `lookup` rule (added by the integrator) fits one value-to-majority-label table per input feature, so the gate's bar is now the best single-feature lookup rather than the majority label.
- `single_feature`: a value-to-majority-label lookup per input feature. This is a stronger trivial rule than the gate has.
- `all_features_lookup`: a lookup on the full input tuple, which is sparse on unseen types.

The CLI prints the audit with `--audit`. Real results (2026-09-24, default config, seed `ol-derivability-audit`):

| Build | Held-out n | Majority (gate bar) | Best single feature | All-features lookup |
|---|---:|---:|---|---:|
| connectivity, capped 5,000 | 1,478 | 0.524 | `output_neuropil` 0.774 (`primary_neuropil` 0.770, `input_neuropil` 0.771, `arbor` 0.696, `hemilineage` 0.695) | 0.553 |
| connectivity, all 50,514 | 14,696 | 0.766 | `arbor` 0.801 | 0.770 |
| NT, capped 5,000 | 1,412 | 0.873 | `hemilineage` 0.882 (`column_span_tier` 0.875; the rest at or below majority) | 0.809 |
| NT, all 44,242 | 12,878 | 0.764 | `column_span_tier` 0.775 | 0.732 |

How to read these numbers:

- **No input determines either label.** Every rule stays far from 1.0 on unseen cell types, so neither objective is tautological.
- **Connectivity is learnable but not trivial.** Neuropil identity carries real signal, because central-brain projection neurons are larger than columnar OL neurons. A trained model must beat 0.524 by the gate margin, and should be compared against the 0.774 single-feature lookup.
- **NT is barely predictable from coarse anatomy on unseen types.** In the capped build the best trivial rule is within 0.01 of majority, and majority is 0.873. Expect the AC6 gate to fail unless a model finds signal these features do not expose. That result is an honest negative, not a bug.
- **Promotion evidence:** the NT objective should not be promoted on the strength of this builder alone.

**Integrator p0 dry run (current code).** Default config, 5,000 samples, pipeline grouped split with seed `integrator-real-eval`, held-out = test split, `verify_hashes=False`:

| Objective | Held-out n | Majority | Gate best (rule, feature) | Model held-out | Trivial-baseline gate |
|---|---:|---:|---|---:|---|
| `connectivity_tier` | 734 | 0.488 | 0.819 (lookup, primary_neuropil) | 0.781 | fail |
| `neurotransmitter_dominance` | 726 | 0.700 | 0.731 (lookup, arbor) | 0.708 | fail |

Neither objective passes AC6: the p0 region experts do not beat a one-feature neuropil or arbor lookup on unseen cell types. Peak RSS was about 1.4 GB per build.

## 7. Provenance in the payload metadata

On top of the dispatcher-required keys, the payload metadata includes:

- `dataset_version`, `neuprint_release` (`optic-lobe:v1.1`)
- `region_vocabulary: "optic_lobe_neuropil"`, `region_division_counts`
- `input_features`, `forbidden_input_features`, `leakage_check`
- `high_connectivity_threshold`
- `typed_rows`
- `input_paths`, `provenance_mode` (`manifest_snapshot` or `explicit_paths`), `manifest_id`, `manifest_sha256`, `product_sha256`
- `filter_stats`: `dropped_status`, `dropped_below_min_total`, `dropped_no_primary_roi`, `dropped_nt_unclear`, `dropped_nt_missing`
- `license`
- the config knobs

`input_fingerprint` covers the paths, the product and manifest sha256 values, and every knob: `allowed_statuses`, `arbor_min_share`, both tier-edge tuples, and the input-feature list.

Explicit paths (`neurons_path`, `meta_path`) must be given together or not at all; a partial set raises `ValueError`. In explicit mode each file's sha256 is computed on the fly, and the release pin and ROI vocabulary are still checked.

## 8. Operational notes

- CLI: `python flybrain_brain_cluster_ol_samples.py --objective <obj> --output <json> --allow-planned [--storage-root F:\.flybrain] [--audit]`
- The default `verify_hashes=None` writes a hash stamp under `manifest/hash-stamps/` the first time each product is verified. Use `verify_hashes=False` (as the smoke tests do) for a read-only run.
- Tests re-register the builder per test and unregister it at module import, following the BANC pattern.
- Before promotion (registry `planned` to `active`): SC3 thresholds per (dataset, objective), the grouped-split trivial-baseline gate (AC6; see §6 for the expected NT outcome), and AB AC1-AC5 still apply.
