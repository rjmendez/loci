# FlyBrain MaleCNS (`mc`) Adapter + Sample Builder Contract

Status: built and checked against the real local snapshot. The registry still lists `mc` as `planned`, so callers must pass `allow_planned=True` until the integrator promotes it.

Modules:

- `mcp/flybrain_mc_adapter.py`: fail-closed snapshot and manifest verification, streaming local readers, and the primary-ROI and NT vocabularies
- `mcp/flybrain_brain_cluster_mc_samples.py`: the training-sample builder, registered for `connectivity_tier`, `neurotransmitter_dominance` and `region_specialization_tier`, plus `trivial_baseline_report`

Tests:

- `mcp/tests/test_flybrain_mc_adapter.py`
- `mcp/tests/test_flybrain_brain_cluster_mc_samples.py`

Both test files use small synthetic fixtures. The smoke tests against real data are skipped unless `LOCI_FLYBRAIN_STORAGE_ROOT` contains the snapshot. They run with `verify_hashes=False`, so they write nothing into it.

## 1. Dataset pin and scope

| Field | Value |
|---|---|
| Registry symbol | `mc` (manifest `dataset.symbol` = `mc`) |
| Version | `male-cns_v1.0` (neuPrint `male-cns:v1.0`, released 2026-06-08; supersedes v0.9) |
| Snapshot root | `$LOCI_FLYBRAIN_STORAGE_ROOT\snapshots\mc\male-cns_v1.0\`, resolved with `snapshot_version_root("mc")` |
| Scope | One adult male *D. melanogaster*: the complete CNS (brain and ventral nerve cord). The manifest `scope` must be `sex=male`, `stage=adult`, `anatomy=brain_and_ventral_nerve_cord`; anything else raises `DATASET_PIN_MISMATCH`. |
| Licence | CC BY 4.0. Evidence: the male-cns.janelia.org site footer, recorded in the manifest as `license_evidence`. |
| Citation | Berg S, Beckett IR, Costa M, Schlegel P, et al. (2026). *Sexual dimorphism in the complete connectome of the Drosophila male central nervous system.* Cell (preprint: bioRxiv doi:10.1101/2025.10.09.680999). Data: FlyEM MaleCNS v1.0, gs://flyem-male-cns/v1.0 |

`mc` is not MANC. MANC (the male VNC only) is `mv`. The optic-lobe dataset `ol` images the same male specimen but has its own version pin. Rows from `ol` are never merged into `mc` rows unless there is an explicit bodyId crosswalk.

## 2. Products the adapter reads

The 2026-09-24 selective pull wrote the manifest (`manifest/manifest.json`, sidecar `7ed0c554...3a54`). It lists 21 files (about 37 GB). The adapter maps five of them to roles. The role names are the same as the `role` field the pull wrote into each manifest entry. If a listed entry's `role` disagrees with the adapter's map, the open fails with `MANIFEST_INVALID`.

| Role | Snapshot-relative path | Bytes | Used for |
|---|---|---:|---|
| `meta` | `metadata/body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14,483,314 | candidates: type, status, superclass, class, sides, hemilineages (211,577 bodies) |
| `nt_prediction` | `metadata/body-neurotransmitters-male-cns-v1.0.feather` | 43,282,834 | NT label (1,835,518 bodies) |
| `edgelist` | `source/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1,051,241,946 | connectivity label; `out_partner_tier` input (151,856,684 edges) |
| `neuron_roi_info` | `source/neuprint-inputs/Neuprint_Neurons.feather` | 4,648,794,426 | per-neuron `roiInfo`, which gives the primary neuropil and the specialization label (88,404,403 bodies) |
| `dataset_meta` | `metadata/neuprint-inputs/Neuprint_Meta.csv` | 1,247,784 | `primaryRois`, the ROI vocabulary cross-check |

These listed files are not read. They are size-checked on every open and hashed only on an explicit verify:

- the synapse-level tables (`syn-points` 13 GB, `syn-partners*`, `tbar-neurotransmitters`)
- `body-stats`
- the significant-only and traced-only weight tables
- `Neuprint_Neuron_Connections.feather`
- the NBLAST cross-matches
- the neuroglancer JSON

The builder does not need the synapse-level tables. Per-neuron `roiInfo` already carries per-ROI pre/post counts.

## 3. Adapter guarantees (`flybrain_mc_adapter`)

`open_mc_snapshot(storage_root=None, *, required_roles=..., verify_hashes=None, stamp_dir=None)` returns an `McSnapshot` or raises `McAdapterError`. The snapshot holds the resolved product paths, the manifest id and sha256, the licence, `hash_verification` and `provenance()`. The error has `.code` set to an `McAdapterErrorCode`, and its message is prefixed with `[CODE] `.

Checks run in the order used by BANC:

1. manifest schema and pins
2. manifest self-hash
3. `manifest.sha256` sidecar
4. per-entry path safety, `role` consistency and exact `size_bytes` for every listed file

File contents are hashed only after all of these pass.

| `verify_hashes` | Content hashing |
|---|---|
| `None` (default) | sha256 of the files behind `required_roles` only, skipped when a write-once stamp (`flybrain_hash_stamps`) matches the file's current `(size, mtime_ns)`. Other listed files get the size check only. |
| `True` | sha256 of every listed file (about 37 GB). Use this for the quarterly recheck. |
| `False` | Size checks only (smoke runs). |

By default, stamps go to `<snapshot>/manifest/hash-stamps/`. The `stamp_dir` argument (or `McSampleBuildConfig.stamp_dir`, or `--stamp-dir` on the CLI) puts them somewhere else, so a verified run can leave the snapshot untouched.

Measured on the real snapshot on F:, with a scratch stamp dir: the first open hashed the 5 required products (about 5.7 GB) in 27 s, and every hash matched the manifest. The second open used the stamps and took 0.47 s.

| Code | Raised when |
|---|---|
| `ROOT_NOT_CONFIGURED` | The storage root is missing, relative, UNC, a drive root, or crosses a symlink or reparse point. |
| `PATH_ESCAPE` | `artifact.relative_root` or a file path is absolute, uses a drive letter, contains `..`, or resolves outside the snapshot. |
| `DATASET_PIN_MISMATCH` | The manifest symbol, version, relative_root or scope is not `mc`/`male-cns_v1.0`/`snapshots/mc/male-cns_v1.0`/adult male CNS; the registry pin has drifted; or `Neuprint_Meta.csv` is not `male-cns:v1.0`. |
| `SNAPSHOT_MISSING` / `MANIFEST_MISSING` | The snapshot dir, `manifest.json` or the sidecar is absent. |
| `MANIFEST_INVALID` | A required field is missing; a status or decision is invalid; a path is a case-folded duplicate; a `.partial`/`.tmp` file is listed; a sha is malformed; a size is not an integer (booleans are rejected); or a `role` disagrees with the product map. |
| `LICENSE_NOT_PERMITTED` | The SPDX id is not `CC-BY-4.0`. |
| `INTEGRITY_MISMATCH` | The status is not `verified`, the self-hash or sidecar is wrong, or a file's size or sha256 is wrong. |
| `PROMOTION_STATE_INVALID` | The refresh decision is `rollback`, or `next_check_due` (2026-12-23 on the real manifest) has passed. |
| `PRODUCT_MISSING` | A required role is unlisted or unknown, or a file is absent. |
| `SCHEMA_MISMATCH` | A required column is missing; the file is not Arrow IPC; an id is null or duplicated; `:ID(Body-ID)` differs from `bodyId:long`; `roiInfo` is not a JSON object; or an NT confidence is outside [0, 1]. |
| `REGION_VOCABULARY_UNKNOWN` | An ROI is outside the 144-ROI map, or the snapshot's `primaryRois` differs from the map. |
| `NT_VOCABULARY_UNKNOWN` | An NT call is outside the 7 MaleCNS classes. |
| `MANIFEST_EXISTS` | `write_mc_manifest` would overwrite a manifest. A re-pull must write a new, superseding manifest. |

**Readers.** Every reader returns body ids as decimal strings. None of them loads a large table whole:

- `load_mc_annotations`: the whole annotation table (14 MB). Dictionary columns such as `statusLabel` are decoded to strings.
- `load_mc_nt_predictions(path, body_ids=...)`: a pyarrow-dataset scan with column projection and an `isin(body_ids)` filter.
- `load_mc_outgoing_totals(path, body_ids=...)`: streams the 152 M-edge table batch by batch (`body_pre`, `weight` only, filtered). Each batch is grouped on its own, and the partial results are then merged. Returns `total_out_synapses = sum(weight)` and `n_post_partners` (edge rows) per `body_pre`. On real bodies, `total_out_synapses` equals neuPrint's per-neuron `downstream` (checked in the smoke test).
- `load_mc_neuron_roi_summary(path, body_ids=...)`: streams `Neuprint_Neurons.feather`, reading 7 of its 56 columns with the id filter, and reduces each `roiInfo` JSON to a `RoiSummary` (see section 4). With about 162k typed bodies the scan takes about 12 s with a warm page cache (about 60 s cold) and about 1.5 GB RSS. The 88 M-row table is never materialized.
- `load_mc_primary_rois` / `check_mc_roi_vocabulary`: parse `primaryRois` from `Neuprint_Meta.csv` and fail closed unless it equals the adapter's map exactly.

The adapter makes no network calls. A test asserts that neither module imports an HTTP, socket or neuprint-client library.

## 4. Region vocabulary: explicit primary-ROI map

`MC_PRIMARY_ROIS` lists all 144 neuPrint primary ROIs of `male-cns:v1.0`. Each maps to `(division, subdivision, neuropil, side, region_id)`, grouped as in the `roiHierarchy` of `Neuprint_Meta.csv`:

| division | subdivision | neuropils (sides `(L)/(R)` where they exist) |
|---|---|---|
| `brain` | `central_brain` | AL, GNG, SCL, LH, PED, CentralBrain-unspecified; CX: AB, EB, FB, NO, PB; INP: IB, ICL, ATL, CRE; LX: BU, LAL; MB: CA, a'L, aL, b'L, bL, gL; PENP: CAN, FLA, PRW, SAD; SNP: SIP, SLP, SMP; VLNP: AOTU, AVLP, PLP, PVLP, WED; VMNP: EPA, GOR, IPS, SPS, VES |
| `brain` | `optic_lobe` | ME, LO, LOP, AME, LA, Optic-unspecified |
| `nerve_cord` | `ventral_nerve_cord` | ANm, HTct(UTct-T3), IntTct, LTct, LegNp(T1/T2/T3), NTct(UTct-T1), Ov, WTct(UTct-T2), mVAC(T1/T2/T3), VNC-unspecified |
| `nerve_cord` | `ventral_nerve_cord_nerve` | ADMN, AbN1-4, AbNT, CvN, DMetaN, DProN, MesoAN, MesoLN, MetaLN, PDMN, PrN, ProAN, ProCN, ProLN, VProN |
| `neck_connective` | `cervical_connective` | CV-unspecified |

`region_id` is `<division>_<neuropil slug>`, without the side. For example, `ME(R)` becomes `brain_me` and `LegNp(T1)(L)` becomes `nerve_cord_legnp_t1`.

Mushroom-body lobes get explicit slugs (`aL` → `mb_al`, `a'L` → `mb_apl`, `bL`, `b'L`, `gL`), because plain sanitizing would collide `aL` with the antennal lobe `AL`. The map is checked for collisions at import time.

**Primary neuropil of a neuron** (`summarize_roi_info`):

- Only primary-ROI keys of `roiInfo` count. Parent and child ROIs such as `CentralBrain` or `AL-DA1(L)` would double count.
- Per ROI, `synapses = pre + post`. The primary neuropil (`top_roi`) is the ROI with the most synapses; ties break on the ROI name.
- The summary also records `top_roi_synapses`, `total_primary_synapses` and `n_primary_rois`.
- Malformed JSON raises `SCHEMA_MISMATCH`.

The real snapshot's `primaryRois` equals the map exactly (144/144).

## 5. Sample builder

`build_mc_training_samples(objective, config)` uses config type `McSampleBuildConfig`. It registers at import (`register_mc_sample_builders()`), and the dispatcher loads it lazily as `flybrain_brain_cluster_mc_samples`. Payloads go through `assemble_training_payload` (the fw diversity and concentration gates). The schema is `flybrain-mc-training-samples/v1`, and `region_vocabulary` is `male_cns_roi`.

**Candidates.** A body is a candidate when all of these hold:

- its annotation `type` is set (it is a typed neuron)
- `status` is in `allowed_statuses` (`Traced`, `Anchor`)
- `superclass` is not in `exclude_super_classes` (`glia`)
- it has at least `min_primary_synapses` (100) pre+post synapses in primary ROIs

On the real snapshot this leaves about 157k of the 88.4 M bodies in `Neuprint_Neurons` (drops: 47,071 untyped, 1,985 other statuses, about 4.4k with fewer than 100 primary synapses). These body ids are the filter for every large scan.

**Shared behaviour:**

- The top `max_regions` (20) regions with at least `min_region_samples` (25) samples are kept.
- Ordering is deterministic: region, then zero-padded body id, then fw's round-robin `balance_and_cap_samples` (`max_samples`, default 5,000).
- `input_fingerprint` covers the paths, the product and manifest sha256 values, every config knob and the input-feature list.
- Explicit paths must be given for every required product or for none.

| Objective | Required roles |
|---|---|
| `connectivity_tier` | dataset_meta, edgelist, meta, neuron_roi_info |
| `neurotransmitter_dominance` | dataset_meta, edgelist, meta, neuron_roi_info, nt_prediction |
| `region_specialization_tier` | dataset_meta, meta, neuron_roi_info |

### 5.1 Label definitions (`LABEL_DEFINITIONS`, also written to payload metadata)

| Objective | Label | Confidence |
|---|---|---|
| `connectivity_tier` | `high_connectivity` if `sum(weight)` over the neuron's outgoing minconf-0.5 neuron-neuron edges is at or above `quantile(high_connectivity_quantile=0.75)` over the candidates (real threshold: 1,816), otherwise `baseline_connectivity`. Neurons below `min_total_count` (10) are dropped. | fw formula `clip(0.55 + 0.4*abs(x-thr)/max(x,thr,1), 0.5, 0.99)` |
| `neurotransmitter_dominance` | `dominant_<code>` of the per-body classifier call `predicted_nt`. Codes: ach, gaba, glut, da, ser, oct, his (MaleCNS has no tyramine class). `unclear` calls are dropped (13,817 candidates), as are bodies with fewer than `min_total_count` T-bar predictions. `consensus_nt` and `ground_truth` go into `metadata.label_source` for audit only. | `clip(predicted_nt_confidence, 0.5, 0.99)` |
| `region_specialization_tier` | `region_specialized` if `top_roi_synapses / total_primary_synapses >= specialization_share` (0.75, about the candidate median of 0.746), otherwise `region_distributed` | `clip(0.55 + 0.4*abs(share-cut)/span, 0.5, 0.99)` |

`region_specialization_tier` is **not circular**. The input names *which* primary neuropil holds the most synapses, and that neuropil is also the routing `region_id`. The input never says *how dominant* that neuropil is. The share, the per-ROI counts and the ROI count are forbidden inputs.

The label varies inside every expert. In the l1em case, by contrast, the only region key is the cell type, which is also the routing key, so any region label would be constant within an expert. The share has a real biological spread: its median is 0.37 for ascending neurons and 0.99 for VNC motor neurons.

### 5.2 Inputs, forbidden inputs, split groups

`input_text` is `dataset mcns10` followed by `key value` pairs, in this fixed order:

| Objective | `INPUT_FEATURES` |
|---|---|
| `connectivity_tier`, `region_specialization_tier` | cns_division, subdivision, primary_neuropil, side, super_class, cell_class, hemilineage |
| `neurotransmitter_dominance` | the same, plus `out_partner_tier` (downstream partners binned at 100/1,000 into t0/t1/t2) |

How the features are built:

- `side` is `somaSide` (L/R/M), falling back to `rootSide`.
- `hemilineage` is the first real lineage: `itoleeHl` (brain), then `trumanHl` (VNC). The sentinels `primary`, `putative_primary`, `no_lineage` and `TBD` stay as input tokens but never become a split group.

**The body id is not in `input_text`.** This is a deliberate deviation from banc/fw. FlyEM body ids are not random: low ids are large bodies that were proofread early. Before this was fixed, the promotion gate's one-feature stump on `root` alone scored 0.850 held-out accuracy for `connectivity_tier` and 0.646 for `region_specialization_tier`, against majority baselines of 0.469 and 0.590. `root`, `root_id`, `body` and `body_id` are forbidden inputs for every objective. The id stays in `sample_id`, `provenance_refs` and `metadata.root_id`.

`FORBIDDEN_INPUT_FEATURES` for every objective includes these identity proxies: cell_type, type, instance, group, flywire/hemibrain/manc type, synonyms and the body id. Each objective also forbids its label sources:

- **connectivity:** synapse counts, partner counts, downstream/upstream, and `out_partner_tier`
- **NT:** every classifier output (`predicted_nt*`, `celltype_*`, `consensus_nt`, `ground_truth`, per-NT probabilities)
- **region specialization:** the share, ROI counts, `roi_info`, and synapse counts

`assert_no_label_leakage` checks every sample before capping. It rejects forbidden keys and any key outside the allow-list.

**Split groups.** `metadata.cell_type` (the annotation `type`) and `metadata.hemilineage` carry the registry `split_group_keys` for mc (`cell_type`, `hemilineage`). They never appear in `input_text`.

## 6. Trivial-baseline check on the real snapshot (2026-09-24)

### 6.0 Integrator re-run (current code)

Two changes landed after the first measurements below:

- **Capped subsets are chosen in sha256(body id) order** (`hash_ordered_preselect`, fingerprint `cap_order: sha256-body-id/v1`). The earlier builds kept each region's lowest body ids. FlyEM ids track body size, so those builds over-sampled large neurons (high_connectivity 2,657 of 5,000, although the 0.75 quantile defines 25% of candidates). Round-robin region balancing still lifts the high share above 25%, because small regions contribute proportionally more.
- **The promotion gate now has a categorical `lookup` rule** (one table per input feature, best held-out feature reported), so it no longer reduces to "beat majority" for mc's category-only inputs.

Full p0 dry run per objective (default config, 5,000 samples, pipeline grouped split with seed `integrator-real-eval`, held-out = test split, `verify_hashes=False`):

| Objective | Labels (5,000) | Held-out n | Majority | Gate best (rule, feature) | Model held-out | Trivial-baseline gate | p0 gate |
|---|---|---:|---:|---|---:|---|---|
| `connectivity_tier` | baseline 2,928 / high 2,072 | 732 | 0.430 | 0.653 (lookup, primary_neuropil) | 0.766 | pass | pass |
| `neurotransmitter_dominance` | ach 2,750, glut 863, gaba 834, da 500, ser 39, his 11, oct 3 | 734 | 0.636 | 0.644 (lookup, side) | 0.553 | fail | fail (also accuracy 0.730 < 0.80) |
| `region_specialization_tier` | distributed 3,035 / specialized 1,965 | 749 | 0.636 | 0.708 (lookup, cell_class) | 0.738 | pass | pass |

Peak RSS was about 1.7 GB per build and each build took 13-20 s warm. mc stays `planned`: SC3 needs at least two eligible threshold reports per (dataset, objective), and AB AC1-AC5 still apply. The tables below are the adapter track's first measurements (id-ordered cap, gate without the lookup rule) and are kept for the record.

`trivial_baseline_report(payload)` reproduces the pipeline's grouped split (union-find over `split_group`, `cell_type` and `hemilineage`, the same 0.7/0.15/0.15 ratios) and runs these on the held-out test split:

- `evaluate_trivial_baselines`, the promotion gate's rules: majority, numeric stump and score argmax
- a single-feature lookup audit (majority label per feature value, fitted on train), which the gate does not model
- an all-feature lookup table

The builds used the defaults (5,000 samples each, `verify_hashes=False`, final code with no body id in the input):

| Objective | Labels (5,000) | Held-out n | Gate best (rule) | Best single-feature lookup | All-feature lookup |
|---|---|---:|---|---|---:|
| `connectivity_tier` | baseline 2,343 / high 2,657 | 725 | 0.469 (majority) | cell_class 0.705 (cns_division 0.674, super_class 0.619, hemilineage 0.546, primary_neuropil 0.527, side 0.473) | 0.546 |
| `neurotransmitter_dominance` | ach 2,920, gaba 799, glut 707, da 498, ser 43, oct 26, his 7 | 750 | 0.656 (majority) | super_class / side 0.657 (the rest are at or below 0.657) | 0.661 |
| `region_specialization_tier` | distributed 3,774 / specialized 1,226 | 731 | 0.590 (majority) | primary_neuropil 0.602 (the rest are at or below 0.599) | 0.592 |

Split statistics: about 946 components (about 536 with more than one sample; the largest has 460 samples). A per-sample split would have let about 369 of those components straddle splits.

Region divisions:

- connectivity: brain 3,500 / nerve_cord 1,500
- NT: brain 3,750 / nerve_cord 1,250
- region specialization: brain 3,500 / nerve_cord 1,500

Reading the table:

- **No label is trivially derivable.** No single feature and no all-feature lookup comes near 1.0 on held-out neurons.
- **The gate under-reports `connectivity_tier`.** Its best rule is majority (0.469), but a one-feature `cell_class` lookup reaches 0.705. A model that only memorizes `cell_class` would pass the gate at its 0.01 margin. See the integrator notes.
- **`neurotransmitter_dominance` may not be learnable under the grouped split.** Held-out cell types and hemilineages are unseen in training, and every lookup sits at the majority rate. Expect the trivial-baseline gate to fail this objective until inputs carry more transferable signal. That is the honest outcome, not a builder bug.
- **Hemilineage adds little held-out signal.** It is a split-group key, so its values never reach both train and test. Within a split it remains a strong NT prior; that is biology (lineage-transmitter coupling).

To reproduce:

```
python flybrain_brain_cluster_mc_samples.py --objective <obj> --output <payload.json> \
    --baseline-report <report.json> --allow-planned [--storage-root F:\.flybrain] [--stamp-dir <scratch>]
```

## 7. Operational notes

- A real build takes about 13-22 s with a warm cache, plus first-open hashing of about 5.7 GB of required products (27 s on F:). Peak RSS is about 1.5-2 GB.
- The tests re-register the builder for each test and unregister it when the module is imported. This keeps the dispatcher's stub-`mc` tests working in the same session.
- Before promotion (`planned` to `active`): SC3 threshold reports per (dataset, objective), the grouped-split trivial-baseline gate (AC6) and AB plan AC1-AC5 still apply.
