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

`open_banc_snapshot(storage_root=None, *, required_roles=..., verify_hashes=True)` returns a `BancSnapshot` (resolved product paths, manifest id/sha, license, `provenance()`), or raises `BancAdapterError`. The error has `.code` set to a `BancAdapterErrorCode`, and its message is prefixed `[CODE] `.

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

- A real build takes about 10 s (connectivity) or about 25 s (NT) on the F: snapshot, plus about 3 s of sha256 verification. Pass `verify_hashes=False` only for smoke runs.
- CLI: `python flybrain_brain_cluster_banc_samples.py --objective <obj> --output <json> --allow-planned [--storage-root F:\.flybrain]`
- Tests re-register the builder per test and unregister it at module import, so the dispatcher's stub-`banc` tests keep an empty slot in the same pytest session.
- Before promotion (registry `planned` to `active`, license `UNREVIEWED` to `CC-BY-4.0`): SC3 (thresholds from at least 2 eligible reports per objective) and the AB plan AC1-AC5 still apply. `region_specialization_tier` is on BANC's registry allow-list but has no builder yet (`BUILDER_NOT_REGISTERED`).
