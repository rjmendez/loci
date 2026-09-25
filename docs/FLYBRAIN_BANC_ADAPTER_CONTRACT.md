# FlyBrain BANC Adapter + Sample Builder Contract

Status: built against a real local snapshot. The registry still lists `banc` as `planned`, so callers must pass `allow_planned=True` until the integrator promotes it.

Modules:

- `mcp/flybrain_banc_adapter.py`: fail-closed snapshot/manifest verification, local readers, and the region and NT vocabularies
- `mcp/flybrain_brain_cluster_banc_samples.py`: training-sample builder, registered for `connectivity_tier` and `neurotransmitter_dominance`

Tests:

- `mcp/tests/test_flybrain_banc_adapter.py`
- `mcp/tests/test_flybrain_brain_cluster_banc_samples.py`

Both use synthetic fixtures only. Each also has a smoke test that is skipped unless `LOCI_FLYBRAIN_STORAGE_ROOT` contains the real snapshot.

## 1. Dataset pin and scope

| Field | Value |
|---|---|
| Registry symbol | `banc` (manifest `dataset.symbol` = `BANC`, the on-disk casing; the adapter compares after `normalize_symbol`) |
| Version | `banc_888` (CAVE materialization 888) |
| Snapshot root | `$LOCI_FLYBRAIN_STORAGE_ROOT\snapshots\BANC\banc_888\` (resolved via `snapshot_version_root("banc")`) |
| Scope | adult female *D. melanogaster*, brain + ventral nerve cord, `connectome_structural`, claim tier `T1_dataset_version_specific` |
| License | CC BY 4.0 (Harvard Dataverse dataset license, API `latestVersion.license`; BANC-project README "License") |
| Citation | Bates AS, Phelps JS, Kim M, Yang HHJ, et al. (2026). *Distributed control circuits across a brain-and-cord connectome.* Nature. doi:10.1038/s41586-026-10735-w. Data: BANC v888, Harvard Dataverse doi:10.7910/DVN/7WTH1N |

## 2. Local products (selective pull, 2026-09-24)

These files were pulled from the public GCS bucket `lee-lab_brain-and-nerve-cord-fly-connectome`, the bucket that `banc_data_locations.md` and the Dataverse docs point to. The pull followed these rules:

- Check the size with a HEAD request first.
- Download with `curl -C -` into `*.partial`.
- Check the remote md5 (`x-goog-hash`) before renaming the file.
- Compute the sha256 and append an entry to `manifest/pull-ledger.jsonl`.

| Role | Snapshot-relative path | Bytes | sha256 |
|---|---|---:|---|
| `edgelist_v3` | `source/compiled_data/banc_888_edgelist_simple_v3.feather` | 359,161,658 | `8c296e946f3c69a8c7222f30ad75fa8a98eeb189124fec6df829c9125f4be64b` |
| `nt_prediction` | `source/compiled_data/banc_888_neurotransmitter_prediction_v2.csv` | 21,107,592 | `bb0f4afa48a05d90008c6d801e053ff2667fb1ba68e29501c92f49514540da1d` |
| `meta` | `metadata/banc_888_meta.feather` | 57,503,026 | `86ccf5df0c67419f8c5f43e93a7ed38d23a080e9f7fde26737290252f3780098` |

The total pull was 437.8 MB, well under the 25 GB dataset budget. F: had about 286 GB free before the pull, above the 200 GB floor.

The manifest is at `manifest/manifest.json`. Its sidecar `manifest.sha256` holds `e6b8e909bc6ff11326667405b9aaeb2a06ed61229435f3974079e52a7c7db981`. The manifest was written by `build_banc_manifest` + `write_banc_manifest` and follows `FLYBRAIN_HARNESS_MANIFEST_PROVENANCE_SCHEMA.md` (self-hash rule, `verified` status, `next_check_due` 2026-12-23).

Products that were not pulled:

- `banc_888_synapses_v3_enriched.parquet` is 19.73 GB. That is over the 15 GB single-file cap. (The Dataverse doc lists it as 5.6 GB; the bucket copy has grown since.)
- `banc_nt_prediction_v3_w_sizethresh_10_*.parquet` (5.4 GB) and `synapse_neuropil_lookup_v3.parquet` (2.4 GB) are per-synapse tables. The builders don't need them because they work from per-neuron summaries.
- The influence shards, meshes, skeletons, NBLAST tables and template volumes were skipped under the pull policy.

Data drift: the GCS `compiled_data` copies are newer than the per-file Dataverse docs.

| File | Rows now | Rows in the Dataverse doc |
|---|---:|---:|
| Edgelist | 13,620,865 | 13,507,098 |
| Meta | 188,508 | 188,162 |

The manifest sha256 values pin the exact bytes that were used. If the bucket is re-pulled, the result must be written as a new manifest that sets `refresh.decision` and `supersedes_manifest_id`. It must not overwrite the current manifest; `write_banc_manifest` refuses with `MANIFEST_EXISTS`.

The git clone at `source/BANC-project` came from an earlier pull and is not listed in `integrity.files`. The adapter does not read it.

## 3. Adapter guarantees (`flybrain_banc_adapter`)

`open_banc_snapshot(storage_root=None, *, required_roles=..., verify_hashes=None)` returns a `BancSnapshot` (resolved product paths, manifest id/sha, license, `provenance()`), or raises `BancAdapterError`. The error has `.code` set to a `BancAdapterErrorCode`, and its message is prefixed `[CODE] `.

Every open checks, in order: manifest schema and pins, the manifest self-hash, the `manifest.sha256` sidecar, then path safety and the exact `size_bytes` of every listed file. File contents are hashed only after all of those pass:

| `verify_hashes` | Content hashing |
|---|---|
| `None` (default) | sha256 of the files behind `required_roles` only. Skipped when a stamp under `manifest/hash-stamps/` matches the file's current `(size, mtime_ns)`. Other listed files, such as the 19.7 GB enriched synapse table, get the size check only. |
| `True` (explicit verify) | sha256 of every listed file, always. Use it for the quarterly integrity recheck and after any suspected tampering. |
| `False` | Size checks only (smoke runs). |

Stamps come from `flybrain_hash_stamps` (schema `fbh-hash-stamp/v1`). One is written per verified file state, and only after the recomputed sha256 matched the manifest and the file did not change while it was hashed. A valid stamp is never rewritten. The stamp name is derived from the relative path, the expected sha256, the size and the mtime_ns, so a size or mtime change, or a new expected hash, forces a rehash. A byte change that keeps both the size and the nanosecond mtime is not caught by a stamped open; `verify_hashes=True` catches it. A stamp caches an earlier verification. It is not a signature, and it is only as trustworthy as the snapshot directory. If the stamp directory is not writable, the open still succeeds and `BancSnapshot.warnings` says so. `BancSnapshot.hash_verification` records `hashed`, `stamp` or `size_only` for each listed file.

| Code | Raised when |
|---|---|
| `ROOT_NOT_CONFIGURED` | The storage root is missing, relative, UNC, a drive root, or crosses a symlink or reparse point (from `flybrain_harness_storage`). |
| `PATH_ESCAPE` | `artifact.relative_root` or a file path is absolute, uses a drive letter, contains `..`, or resolves outside the snapshot. |
| `DATASET_PIN_MISMATCH` | The manifest symbol/version/relative_root is not BANC/`banc_888`/`snapshots/BANC/banc_888`, or the registry pin has drifted. |
| `SNAPSHOT_MISSING` / `MANIFEST_MISSING` | The snapshot directory, `manifest/manifest.json`, or the `manifest/manifest.sha256` sidecar is absent (a missing sidecar fails closed; it is not a warning). |
| `MANIFEST_INVALID` | A required field is missing, a status or decision value is invalid, a path is a duplicate after case-folding, a `.partial`/`.tmp`/`.inprogress` file is listed, a sha is malformed, or a size is non-integer. |
| `LICENSE_NOT_PERMITTED` | `dataset.source.license.spdx_id` is not in `{"CC-BY-4.0"}` (for example `UNREVIEWED`). |
| `INTEGRITY_MISMATCH` | The status is not `verified`, the manifest self-hash is wrong, the sidecar does not match, or a file's size or sha256 does not match. |
| `PROMOTION_STATE_INVALID` | `refresh.decision == "rollback"`, or `next_check_due` has passed. |
| `PRODUCT_MISSING` | A required role is not listed in the manifest, or a listed or explicit file does not exist. |
| `SCHEMA_MISMATCH` | A required column is missing, the file is not Arrow IPC, an id is null or duplicated in meta, NT rows for the same root conflict, or an NT score is outside [0, 1]. |
| `REGION_VOCABULARY_UNKNOWN` | A `root_region` prefix or curated `region` value is not in the explicit map. |
| `NT_VOCABULARY_UNKNOWN` | The predicted neurotransmitter is not one of the 8 BANC classes. |
| `MANIFEST_EXISTS` | Writing a manifest would overwrite an existing one. |

The readers keep all 18- and 19-digit root ids as strings:

- `load_banc_meta`: meta feather or parquet. Requires `banc_888_id, proofread, side, root_region, region, hemilineage, flow, super_class, cell_class`, and `banc_888_id` must be unique.
- `load_banc_outgoing_totals`: reads only the Arrow schema first, then sums `count` per `pre` and counts distinct `post` partners. Output is sorted by root id.
- `load_banc_nt_predictions`: the published CSV repeats a root once per SeaTable anchor row (551 roots span 1,754 rows). Repeats with identical prediction columns are collapsed into one row, with `nt_row_multiplicity` recording the count. Repeats that disagree fail closed.

The adapter makes no network calls. A test asserts that neither module imports an HTTP or socket library.

## 4. Region vocabulary: brain versus nerve cord (explicit)

`region_id` is built from BANC's `root_region`, the neuropil that contains the neuron's root point (soma or primary neurite). Unlike fw, it is not taken from the dominant synaptic neuropil, because BANC's per-synapse neuropil lookup was not pulled.

| `root_region` prefix | `cns_division` | `subdivision` |
|---|---|---|
| `ITO_optic_` | `brain` | `optic_lobe` |
| `ITO_midbrain_` | `brain` | `central_brain` |
| `COURT_vnc_` | `nerve_cord` | `ventral_nerve_cord` |
| `MANC_vnc_` | `nerve_cord` | `ventral_nerve_cord` |

How a label becomes a `region_id`:

- The neuropil is the part after the prefix, with any trailing `_L`/`_R` removed and recorded as the side.
- `region_id = sanitize("<division>_<neuropil>")`. For example, `ITO_optic_ME_R` becomes `brain_me`, `COURT_vnc_ProNM-T1` becomes `nerve_cord_pronm_t1`, and `MANC_vnc_LNp_T1_L` becomes `nerve_cord_lnp_t1`.
- Any other prefix raises `REGION_VOCABULARY_UNKNOWN`.

The curated `region` column (`optic_lobe` | `central_brain` | `ventral_nerve_cord`) is used as a cross-check:

- If it disagrees with the division/subdivision derived from `root_region`, the neuron is dropped and counted in `filter_stats.dropped_division_conflict`. On the real data this is about 9.7k of about 155k candidates, mostly optic-lobe roots whose curated region is `central_brain`.
- Neurons with no `root_region` are dropped and counted in `dropped_missing_root_region`.
- Any other curated value raises an error.

Output payloads carry `region_vocabulary: "banc_neuropil"` and `region_division_counts`. With the defaults (top 20 regions by count, round-robin cap of 5,000), the real build selects 20 regions, with 2,750 brain samples and 2,250 nerve-cord samples (integration re-run 2026-09-24, after the proofread fix below).

## 5. Objectives and label definitions

The builder is `build_banc_training_samples(objective, config)`, with config type `BancSampleBuildConfig`. It is registered at import through `register_banc_sample_builders()`. The payload is assembled with `assemble_training_payload`, so fw's label-diversity and label-concentration gates apply unchanged. The schema is `flybrain-banc-training-samples/v1`.

Common filters:

- `proofread_only=True`. The published meta stores `proofread` as the strings `"TRUE"`/`"FALSE"`; the builder parses them explicitly (`_proofread_mask`) and fails closed on any other token. An earlier draft used `astype(bool)`, which treated `"FALSE"` as true and made this filter a no-op.
- `exclude_super_classes=("glia", "not_a_neuron", "trachea")`
- `min_region_samples`, `max_regions`
- Deterministic ordering: region, then zero-padded root id, then fw's round-robin `balance_and_cap_samples`

### 5.1 `connectivity_tier` (same label semantics as fw)

- **Measure:** `total_out_synapses` is the sum of `count` over the v3 edgelist rows where the neuron is `pre`. The edgelist is rolled up at synapse size of at least 10 voxels, with autapses excluded. Only neurons with a measure of at least `min_total_count` (10) are kept.
- **Label:** `high_connectivity` if the measure is at or above the global quantile `high_connectivity_quantile` (0.75) over the retained candidates, otherwise `baseline_connectivity`. On the real data the threshold is 278 synapses.
- **Confidence:** same as fw, `clip(0.55 + 0.4 * |x - thr| / max(x, thr, 1), 0.5, 0.99)`.
- **Real build:** 5,000 samples, 3,162 baseline and 1,838 high (dominant share 0.632); 7,397 unproofread candidates dropped.

### 5.2 `neurotransmitter_dominance` (fw semantics, BANC vocabulary)

- **Label:** `dominant_<code>` from the per-neuron classifier call `neurotransmitter_predicted`, which is the argmax of the per-synapse probability sums.
- **Codes:** `ach, gaba, glut, da, ser, oct` match fw. `his` (histamine) and `tyr` (tyramine) exist only in BANC and are kept.
- **Confidence:** `clip(neurotransmitter_score, 0.5, 0.99)`.
- **Filters:** at least `min_total_count` classified presynapses. Roots with more than one anchor row in the CSV are excluded (`dropped_multi_anchor`, 551 on the real data), because they are probable merge or multi-soma segments.
- **Classifier version mismatch:** the per-neuron CSV uses the v2 classifier (size of at least 5). No per-neuron v3 summary has been deposited yet. The `out_partner_tier` input feature comes from the v3 edgelist. This mix is intentional and recorded in `input_paths` and `product_sha256`.
- **Real build:** 5,000 samples. Counts by label: ach 2,797, gaba 991, glut 624, da 378, his 96, ser 64, oct 42, tyr 8 (dominant share 0.559); 7,936 unproofread candidates dropped.

## 6. Label hygiene and leakage check

Unlike fw, which puts `total_pre_count` and the NT averages directly into `input_text`, BANC's `input_text` never contains the quantity a label is computed from.

**Inputs.** `input_text` is `dataset banc888 root <id>` followed by `key value` pairs, in this fixed order:

| Objective | Input features (`INPUT_FEATURES`) |
|---|---|
| `connectivity_tier` | cns_division, subdivision, root_neuropil, side, super_class, cell_class, flow, hemilineage |
| `neurotransmitter_dominance` | the same, plus `out_partner_tier` (distinct downstream partners binned at 10/100 into t0/t1/t2) |

**Forbidden inputs** (`FORBIDDEN_INPUT_FEATURES`):

| Objective | Forbidden keys | Reason |
|---|---|---|
| `connectivity_tier` | total_out_synapses, pre_count, count, norm, n_post_partners, output_connections, input_connections, out_partner_tier, cell_type | Label sources and close proxies. `output_connections` in meta is itself a synapse count. |
| `neurotransmitter_dominance` | the 8 per-NT counts, neurotransmitter_predicted/score, cell_type_neurotransmitter_*, neurotransmitter_verified, known_nt, top_nt, cell_type | Label sources, and label-derived cell-type consensus. `cell_type` is excluded from both objectives because it keys the consensus NT. |

**Runtime check.** `assert_no_label_leakage` runs on every sample before capping. It fails if any key is forbidden or not in the allow-list. The payload records `leakage_check: "passed"`, `input_features` and `forbidden_input_features`. The label-source values are kept only in `sample.metadata.label_source`. Training reads `input_text`, not metadata.

**Audit (2026-09-24, real 5,000-sample builds, run before the proofread fix; re-run it on the current build before promotion).** This is in-sample, best single-feature lookup accuracy, which is an optimistic upper bound:

| Objective | Majority baseline | Best feature | Other features |
|---|---:|---|---|
| connectivity | 0.638 | hemilineage 0.779 | super_class 0.736, cell_class 0.726, root_neuropil 0.683; the rest are at or below 0.642 |
| NT | 0.548 | hemilineage 0.798 | cell_class 0.600; the rest are at or below 0.570; `out_partner_tier` 0.548 |

No single feature determines the label. Hemilineage is a strong prior for NT identity, which is expected biology (lineage-transmitter coupling), but it is not derived from the label. Re-run this audit whenever `INPUT_FEATURES` changes.

## 7. Provenance in the payload metadata

On top of the dispatcher-required keys, the payload metadata includes:

- `dataset_version`
- `region_vocabulary`, `region_division_counts`
- `input_paths`
- `provenance_mode`: `manifest_snapshot`, or `explicit_paths`, where each file's sha256 is computed on the fly
- `manifest_id`, `manifest_sha256`, `product_sha256`
- `filter_stats`
- `license`
- `high_connectivity_threshold`
- the config knobs

`input_fingerprint` covers the paths, the product and manifest sha256 values, every config knob and the input-feature list.

Explicit paths must be given for every required product or for none; a partial set raises `ValueError`. The required products are:

| Objective | Required products |
|---|---|
| connectivity | edgelist + meta |
| NT | edgelist + meta + nt_prediction |

## 8. Operational notes

- A real build takes about 10 s (connectivity) or about 25 s (NT) on the F: snapshot, plus sha256 verification of the required products the first time each file state is opened (stamped after that). Pass `verify_hashes=True` for a full rehash of every listed file, and `verify_hashes=False` only for smoke runs.
- CLI: `python flybrain_brain_cluster_banc_samples.py --objective <obj> --output <json> --allow-planned [--storage-root F:\.flybrain]`
- Tests re-register the builder per test and unregister it at module import, so the dispatcher's stub-`banc` tests keep an empty slot in the same pytest session.
- Before promotion (registry `planned` to `active`, license `UNREVIEWED` to `CC-BY-4.0`): SC3 (thresholds from at least 2 eligible reports per (dataset, objective)), the grouped-split trivial-baseline gate (AC6) and the AB plan AC1-AC5 still apply. `region_specialization_tier` is on BANC's registry allow-list but has no builder yet (`BUILDER_NOT_REGISTERED`).

## 9. Real models (2026-09-24, `flybrain_banc_targets`)

`mcp/flybrain_banc_targets.py` builds structured model inputs for five BANC targets and runs `flybrain_model_eval.run_evaluation`: grouped split by `cell_type` + `hemilineage` (union-find), grouped CV tuning, calibration, one held-out test pass, cluster bootstrap, label shuffle, and a drop-one-family ablation. Reports and models are written to `/mnt/f/.flybrain/logs/real-models-20260924T174122Z/banc/<target>/<run>/`.

**Targets and labels**

| Target | Label | Samples |
|---|---|---|
| `super_class` | curated `super_class`, harmonized to the shared vocabulary (`harmonize_super_class`); `unknown`/`non_neuronal`/`other` dropped | proofread, wired; at least 200 neurons and 5 cell types per class; hash-ordered cap per class |
| `cell_class` | curated `cell_class` (slug) | same, with at least 150 neurons per class |
| `flow` | curated `afferent` / `intrinsic` / `efferent` | same |
| `connectivity_tier` | unchanged legacy builder (section 5.1) | legacy builder, `max_samples` raised |
| `neurotransmitter_dominance` | unchanged legacy builder (section 5.2). The labels are **v2 classifier predictions, not ground truth.** | legacy builder |

**Inputs (feature families)**

- `degree`, `recip`: from `edgelist_simple_v3` (weight = `count`).
- `out_np`, `in_np`: top-30 neuropil fractions plus other, entropy and count. They come from one streamed pass over the 198.8M-row `synapses_v3_enriched.parquet` (one synapse = weight 1), computed once with a constant partner category so the table is label-independent (cache `wiring-features/banc/82b4222f…`, 487 s, 7 GB RSS, under the heavy lock).
- `out_comp`, `in_comp`, `out2_comp`, `in2_comp`: partner composition by harmonized super_class. For `super_class`, `cell_class` and `flow`, the partner category of **every node outside the train split is masked**, including neurons that are not in the evaluation sample. This is stricter than masking val and test only: an unsampled neuron of a held-out type would otherwise leak its label through homophily. The 2-hop return path is removed by the foundation.
- `morph`: `banc_888_metrics.feather` (cable, volume, L2 nodes, branch/end points, axon/dendrite length, mitochondria, pd_width, segregation index, projection score, and derived ratios). For `connectivity_tier` the extensive size measures are dropped as size proxies (`LOCAL_EXCLUSIONS`).
- `annot`: curated super_class, cell_class, flow and side, for `connectivity_tier` and NT only.
- The registered `flybrain_wiring_features` exclusions apply on top of all of this: for example `degree__*` and `*n_neuropils*` are dropped for `connectivity_tier`. IDs are never features.
- The NB text view is `name name_qK` tokens, with quantile bins fitted on train rows only.

**Snapshot hygiene.** `open_banc_inputs` validates the manifest with `verify_hashes=False` (structure and sizes, so no stamp is written into the snapshot). It then sha256-verifies each file it reads against the manifest, keeping stamps in `/mnt/f/.flybrain/cache/hash-stamps/BANC-banc_888`. All five files matched on first hashing.

**Results (run `quick`: capped samples, one config per model, 3-fold grouped CV).** The full-grid `v1` runs were queued behind the shared heavy lock for more than 2 h and never started, so these are the reported numbers. Test accuracy has a cluster-bootstrap 95% CI; the gate is the harness gate.

| Target | Model | Majority | Best trivial (rule) | Model acc [95% CI] | Macro-F1 | ECE | Shuffle | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| super_class | nb | 0.073 | 0.445 (text lookup) | 0.819 [0.741, 0.879] | 0.810 | 0.060 | 0.090 | pass |
| super_class | logreg | 0.073 | 0.377 (feature lookup) | 0.960 [0.942, 0.974] | 0.946 | 0.037 | 0.091 | pass |
| super_class | hgb | 0.073 | 0.377 | 0.961 [0.948, 0.973] | 0.954 | 0.014 | 0.075 | pass |
| super_class (no neuropil) | hgb | 0.073 | 0.356 | 0.921 [0.881, 0.956] | 0.909 | 0.024 | 0.074 | pass |
| super_class (no neuropil) | logreg | 0.073 | 0.356 | 0.912 [0.866, 0.948] | 0.895 | 0.034 | 0.096 | fail (shuffle limit 0.093) |
| cell_class | logreg | 0.000 | 0.051 | 0.625 [0.508, 0.765] | 0.555 | **0.419** | 0.005 | pass |
| flow | logreg | 0.549 | 0.947 (lookup on missing `pd_width`) | 0.996 [0.990, 0.999] | 0.993 | 0.009 | 0.591 | fail (shuffle) |
| flow | hgb | 0.549 | 0.947 | 0.990 [0.984, 0.995] | 0.986 | 0.008 | 0.604 | fail (shuffle) |
| connectivity_tier | hgb | 0.597 | 0.753 (lookup `annot__super_class`) | 0.883 [0.859, 0.907] | 0.877 | 0.025 | 0.596 | pass |
| connectivity_tier | logreg | 0.597 | 0.753 | 0.789 [0.748, 0.832] | 0.779 | 0.055 | 0.588 | fail (paired CI includes 0) |
| NT (v2 predictions) | logreg | 0.415 | 0.476 (lookup `annot__cell_class`) | 0.622 [0.538, 0.721] | 0.554 | 0.129 | 0.402 | pass |
| NT (v2 predictions) | hgb | 0.415 | 0.476 | 0.435 [0.309, 0.587] | 0.213 | 0.272 | 0.415 | fail |

**Reading the results**

- **super_class** is learnable from wiring and morphology across unseen cell types and hemilineages. The random-split accuracy (0.971) is close to the grouped accuracy (0.961), so memorizing groups adds little. Dropping neuropil costs 4 points. Neuropil location partly *defines* the intrinsic, ascending and descending classes, so the no-neuropil row is the stricter claim.
- **flow** is nearly trivial: one missing-value lookup (afferents have no primary dendrite) reaches 0.947. The harness gate fails only on its single label permutation. Over 10 permutations (`shuffle_multi_flow.json`), the shuffle accuracy averages 0.48 (logreg) and 0.50 (hgb), below the 0.549 majority rate. With two dominant feature clusters, one permutation can map a whole cluster to the right class by chance. This is control variance, not leakage, but the harness verdict stands as fail.
- **connectivity_tier** hgb beats the lookup rule by 9–17 points (paired CI). No single family is essential (largest drop: partner composition, −3 points). Entropy-type features correlate with neuron size, so treat this as "predictable from connectivity shape", not independent of size.
- **NT**: logreg generalizes across hemilineage groups; hgb does not (random split 0.79 against grouped 0.44), and isotonic calibration does not rescue it. The labels are classifier predictions.
- **cell_class**: the train majority class does not occur in test (grouped split), so the majority rate is 0. Accuracy is real but calibration is poor (ECE 0.42). This run has no ablation and no hgb, because the unlocked 5-minute budget did not fit about 60 classes.

**Reproduce:** `python -m flybrain_banc_targets --target <t> --run-label v1` (full grid; run it under `flock /tmp/flybrain-heavy.lock`). The quick-run scripts are next to the reports.
