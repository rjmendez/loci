"""MaleCNS v1.0 (``mc``) real-model targets: independent labels + structured wiring features.

This module turns the manifest-verified ``mc`` snapshot into
``flybrain_model_eval.EvalDataset`` objects and runs the evaluation harness
(``run_evaluation``) on them. Targets:

* ``nt_ground_truth``: the ``ground_truth`` column of the body-neurotransmitter
  table (only rows where it is set). In MaleCNS v1.0 it is assigned per cell
  type (every one of the 3,318 annotated types carries exactly one value), from
  transmitter identities known outside the synapse classifier (literature,
  transcriptomics and, in the VNC, hemilineage-transmitter assignments). It was
  the classifier's training target, so every classifier output
  (``predicted_nt*``, ``celltype_*``, ``consensus_nt``, which equals
  ``ground_truth`` on 100% of labelled rows) is excluded; none of them is ever
  loaded into the feature frame. Label provenance: ``measured`` but
  uncertain (it is the synister_malecns training table: confidence >= 3 plus
  hemilineage-inferred VNC labels, SYNTHESIS E5); superseded by
  ``nt_literature``.
* ``nt_literature`` (+ ``_binary``, ``_all``, ``_all_binary``): the R2
  literature ground truth (``flybrain_nt_ground_truth``: drosophila_neurotransmitters,
  confidence >= 4), mapped onto male-CNS types by name and by the male-CNS GT
  table's ``gt_celltype -> cell_type_mcns`` column. ``nt_literature`` drops the
  synister_malecns training types (R2); ``nt_literature_all`` keeps them.
* ``super_class``: the curated ``superclass`` annotation, harmonized with
  ``flybrain_wiring_features.harmonize_super_class`` (curated_morphology).
* ``cell_class``: the curated ``class`` annotation (only rows where it is set;
  mostly sensory modality and central-brain classes). connectivity_defined:
  male-CNS types used NBLAST + connectivity [Berg 2025], so this is reported as
  recovery of connectivity-derived annotations and never gated.
* ``connectivity_tier`` / ``region_specialization_tier``: the existing labels,
  built by ``flybrain_brain_cluster_mc_samples`` (unchanged definitions);
  connectivity_defined (statistics of the connectome itself), never gated.

Label provenance (R1) per target is ``TARGET_LABEL_PROVENANCE``, checked against
``flybrain_target_registry`` at import; ``run_target`` evaluates through
``flybrain_target_registry.run_gated_evaluation``.

Features (``<family>__<name>``; ablations drop one family at a time):

* wiring (``flybrain_wiring_features`` on the 151.9 M-edge minconf-0.5 table,
  node table = all 211,577 annotated bodies, partner category = harmonized
  super class): ``degree``, ``out_comp``, ``in_comp``, ``recip``, ``out2_comp``,
  ``in2_comp``. The mc pair table has no neuropil column, so ``out_np``/``in_np``
  do not exist here; per-neuropil information comes from ``roi`` instead.
* ``roi`` (this module, from ``Neuprint_Neurons.feather`` ``roiInfo``): the
  fraction of the neuron's pre and post synapses in each of the top-K base
  neuropils (hemispheres merged), an ``other`` fraction, pre/post entropy and
  the left-hemisphere share. ``degree__roi_pre_share`` and
  ``degree__roi_n_primary_rois`` are size/polarity quantities and live in the
  ``degree`` family so ``connectivity_tier`` drops them with the other degree
  features.
* categorical columns with plain names (so the registered exclusion patterns
  apply to them): ``cns_division``, ``subdivision``, ``primary_neuropil`` (the
  top primary ROI, measured), ``side``, ``super_class``, ``cell_class``,
  ``hemilineage``.

Leakage rules on top of the registered exclusions (``EXTRA_EXCLUDED``):
``hemilineage`` is never an input for ``nt_ground_truth`` (the VNC ground truth
is assigned per hemilineage) nor for ``super_class`` / ``cell_class`` (its
sentinel values mark non-intrinsic neurons). Cell type, supertype and body id
are never inputs; cell type, hemilineage and supertype are the grouped-split
keys (union-find). When the partner category (super class) is the target or is
determined by it (``super_class``, ``cell_class``), the partner category of
every node that shares a held-out component's cell type, hemilineage or
supertype is masked before the wiring features are computed.

Samples are capped per cell type (``max_per_type``, sha256 order) so the
columnar optic-lobe types (up to ~2,000 copies) do not dominate. Nothing is
written into ``snapshots/``; caches go under ``<cache_root>`` and reports under
``<report_root>/mc/<target>/<run_label>/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import flybrain_nt_ground_truth as ntgt
import flybrain_target_registry as ftr
import flybrain_wiring_features as fwf
from flybrain_mc_adapter import (
    MC_PRIMARY_ROIS,
    NEURON_ID_COLUMN,
    NEURON_ROI_INFO_COLUMN,
    ROLE_DATASET_META,
    ROLE_EDGELIST,
    ROLE_META,
    ROLE_NEURON_ROI_INFO,
    ROLE_NT_PREDICTION,
    McAdapterError,
    McAdapterErrorCode,
    check_mc_roi_vocabulary,
    load_mc_annotations,
    load_mc_nt_predictions,
    map_mc_primary_roi,
    nt_short_code,
    open_mc_snapshot,
)

MC_SYMBOL = "mc"
TARGET_NT_GROUND_TRUTH = "nt_ground_truth"
TARGET_SUPER_CLASS = "super_class"
TARGET_CELL_CLASS = "cell_class"
TARGET_CONNECTIVITY = "connectivity_tier"
TARGET_REGION_SPECIALIZATION = "region_specialization_tier"
NT_LITERATURE_TARGETS: tuple[str, ...] = ntgt.NT_LITERATURE_TARGETS
MC_REAL_TARGETS: tuple[str, ...] = (
    TARGET_NT_GROUND_TRUTH,
    TARGET_SUPER_CLASS,
    TARGET_CELL_CLASS,
    TARGET_CONNECTIVITY,
    TARGET_REGION_SPECIALIZATION,
    *NT_LITERATURE_TARGETS,
)
LEGACY_TARGETS = frozenset({TARGET_CONNECTIVITY, TARGET_REGION_SPECIALIZATION})
_P = ftr.LabelProvenance
# R1 label provenance per target (must equal flybrain_target_registry; checked at import).
TARGET_LABEL_PROVENANCE: Mapping[str, str] = {
    TARGET_NT_GROUND_TRUTH: _P.MEASURED.value,
    TARGET_SUPER_CLASS: _P.CURATED_MORPHOLOGY.value,
    TARGET_CELL_CLASS: _P.CONNECTIVITY_DEFINED.value,
    TARGET_CONNECTIVITY: _P.CONNECTIVITY_DEFINED.value,
    TARGET_REGION_SPECIALIZATION: _P.CONNECTIVITY_DEFINED.value,
    **{t: _P.MEASURED.value for t in NT_LITERATURE_TARGETS},
}
ftr.check_module_provenance("mc", TARGET_LABEL_PROVENANCE)
# Partner category (super class) is the target or a function of it -> mask held-out nodes.
MASKED_TARGETS = frozenset({TARGET_SUPER_CLASS, TARGET_CELL_CLASS})

GROUP_KEYS: tuple[str, ...] = ("cell_type", "hemilineage", "supertype")
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "cns_division", "subdivision", "primary_neuropil", "side", "super_class", "cell_class", "hemilineage",
)
# Inputs dropped on top of the registered exclusion patterns (reason in the module docstring).
EXTRA_EXCLUDED: Mapping[str, frozenset[str]] = {
    TARGET_NT_GROUND_TRUTH: frozenset({"hemilineage"}),
    **{t: frozenset({"hemilineage"}) for t in NT_LITERATURE_TARGETS},
    TARGET_SUPER_CLASS: frozenset({"hemilineage"}),
    TARGET_CELL_CLASS: frozenset({"hemilineage"}),
    TARGET_CONNECTIVITY: frozenset(),
    TARGET_REGION_SPECIALIZATION: frozenset(),
}
# Never inputs for any target (identity proxies / split keys / label sources).
NEVER_FEATURES = frozenset({
    "root_id", "body_id", "bodyId", "node_id", "cell_type", "type", "supertype", "instance", "group",
    "hemilineage_group", "label", "sample_id", "superclass_raw", "ground_truth",
})
HEMILINEAGE_SENTINELS = frozenset({"primary", "putative_primary", "no_lineage", "tbd", "unknown"})
NON_LABEL_SUPER_CLASSES = frozenset({fwf.UNKNOWN, "non_neuronal", "other"})
TEXT_TAG = "mcns10"
SPARSITY_FAMILY = "sparsity"

DEFAULT_CACHE_ROOT = "/mnt/f/.flybrain/cache"
DEFAULT_STAMP_DIR = "/mnt/f/.flybrain/cache/hash-stamps/mc"
DEFAULT_REPORT_ROOT = "/mnt/f/.flybrain/logs/real-models-20260924T174122Z"
ROI_CACHE_SCHEMA = "flybrain-mc-roi-long/v1"
_SIDE_TOKENS = {"l": "left", "r": "right", "m": "midline"}
_ROLES = (ROLE_META, ROLE_DATASET_META, ROLE_NEURON_ROI_INFO, ROLE_EDGELIST, ROLE_NT_PREDICTION)


@dataclass(frozen=True)
class McRealModelConfig:
    storage_root: str | Path | None = None
    # Write-once hash stamps outside the snapshot (nothing is written into snapshots/).
    stamp_dir: str | Path | None = DEFAULT_STAMP_DIR
    verify_hashes: bool | None = None
    cache_root: str | Path | None = DEFAULT_CACHE_ROOT
    max_per_type: int = 20
    min_primary_synapses: int = 100
    min_types_per_class: int = 5
    top_k_rois: int = 24
    two_hop: bool = True
    allowed_statuses: tuple[str, ...] = ("Traced", "Anchor")
    cap_salt: str = "mc-real-models/v1"
    # Legacy (connectivity / region) builds: take every candidate, then cap per type here.
    legacy_max_samples: int = 400_000
    legacy_max_regions: int = 500

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in out.items()}


# =========================================================================== inputs


@dataclass(frozen=True)
class McInputs:
    paths: Mapping[str, Path]
    manifest_sha256: str | None
    product_sha256: Mapping[str, str]
    hash_verification: Mapping[str, Any] = field(default_factory=dict)

    def provenance(self) -> dict[str, Any]:
        return {"manifest_sha256": self.manifest_sha256, "product_sha256": dict(self.product_sha256),
                "paths": {k: str(v) for k, v in sorted(self.paths.items())}}


def open_inputs(config: McRealModelConfig, *, paths: Mapping[str, str | Path] | None = None) -> McInputs:
    """Manifest-verified snapshot paths, or explicit local paths (tests) hashed on the fly."""
    if paths is not None:
        from flybrain_mc_adapter import sha256_file

        resolved = {role: Path(p) for role, p in paths.items()}
        for role, p in resolved.items():
            if not p.is_file():
                raise McAdapterError(McAdapterErrorCode.PRODUCT_MISSING, f"mc {role} file not found", {"path": str(p)})
        return McInputs(resolved, None, {role: sha256_file(p) for role, p in sorted(resolved.items())})
    snapshot = open_mc_snapshot(config.storage_root, required_roles=_ROLES, verify_hashes=config.verify_hashes,
                                stamp_dir=config.stamp_dir)
    prov = snapshot.provenance()
    return McInputs({role: snapshot.path(role) for role in _ROLES}, prov["manifest_sha256"],
                    dict(prov["product_sha256"]), dict(snapshot.hash_verification))


def _missing(value: Any) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "<na>"}


def _clean(value: Any) -> str:
    return "unknown" if _missing(value) else "_".join(str(value).strip().lower().split())


def _side(soma_side: Any, root_side: Any) -> str:
    for value in (soma_side, root_side):
        if not _missing(value) and str(value).strip().lower() in _SIDE_TOKENS:
            return _SIDE_TOKENS[str(value).strip().lower()]
    return "unknown"


def _hemilineage(ito: Any, truman: Any) -> tuple[str, str | None]:
    """(input token, split-group value or None); same rule as the legacy mc builder."""
    tokens = [_clean(v) for v in (ito, truman) if not _missing(v)]
    for token in tokens:
        if token not in HEMILINEAGE_SENTINELS:
            return token, token
    return (tokens[0] if tokens else "unknown"), None


def load_node_table(meta_path: Path) -> pd.DataFrame:
    """Every annotated body (the wiring node table) with harmonized labels and split keys."""
    meta = load_mc_annotations(meta_path, columns=("type", "status", "superclass", "class", "somaSide", "rootSide",
                                                   "itoleeHl", "trumanHl"), optional_columns=("supertype",))
    if "supertype" not in meta.columns:
        meta["supertype"] = None
    hl = [_hemilineage(a, b) for a, b in zip(meta["itoleeHl"], meta["trumanHl"])]
    out = pd.DataFrame({
        "root_id": meta["bodyId"].astype(str),
        "cell_type": [None if _missing(v) else str(v).strip() for v in meta["type"]],
        "status": meta["status"].fillna("").astype(str).str.strip(),
        "superclass_raw": meta["superclass"],
        "super_class": [fwf.harmonize_super_class(v) for v in meta["superclass"]],
        "cell_class": [None if _missing(v) else _clean(v) for v in meta["class"]],
        "side": [_side(a, b) for a, b in zip(meta["somaSide"], meta["rootSide"])],
        "hemilineage": [t for t, _ in hl],
        "hemilineage_group": [g for _, g in hl],
        "supertype": [None if _missing(v) else str(v).strip() for v in meta["supertype"]],
    })
    return out


# =========================================================================== roi features


def parse_roi_info(raw: Any) -> list[tuple[str, int, int]]:
    """``roiInfo`` JSON -> [(primary roi, pre, post)] for primary ROIs with any synapse (fail closed)."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)) or not str(raw).strip():
        return []
    try:
        info = json.loads(raw)
    except ValueError as exc:
        raise McAdapterError(McAdapterErrorCode.SCHEMA_MISMATCH, f"roiInfo is not valid JSON: {exc}") from exc
    if not isinstance(info, dict):
        raise McAdapterError(McAdapterErrorCode.SCHEMA_MISMATCH, "roiInfo must be a JSON object")
    out = []
    for roi, stats in info.items():
        if roi not in MC_PRIMARY_ROIS:
            continue
        if not isinstance(stats, dict):
            raise McAdapterError(McAdapterErrorCode.SCHEMA_MISMATCH, "roiInfo entry must be an object", {"roi": roi})
        pre, post = int(stats.get("pre", 0) or 0), int(stats.get("post", 0) or 0)
        if pre or post:
            out.append((roi, pre, post))
    return sorted(out)


def _ids_digest(ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(ids, key=lambda v: (len(v), v)):
        digest.update(f"{value}\n".encode("utf-8"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_snapshot_path(path: Path) -> None:
    if "snapshots" in path.resolve(strict=False).parts:
        raise ValueError(f"refusing to write under a snapshots directory: {path}")


def load_roi_long(neuron_path: Path, body_ids: Sequence[str], *, product_sha256: str | None,
                  cache_root: str | Path | None = DEFAULT_CACHE_ROOT) -> pd.DataFrame:
    """Per-body primary-ROI pre/post counts in long form (root_id, roi, pre, post), cached.

    Streams ``Neuprint_Neurons.feather`` (~88 M rows) with column projection
    and a body-id filter; the parquet cache is keyed by the product sha256 (or
    size+mtime when none is given) and the requested id set.
    """
    import pyarrow as pa
    import pyarrow.dataset as ds

    ids = [str(v) for v in body_ids]
    stat = os.stat(neuron_path)
    key_payload = {"schema": ROI_CACHE_SCHEMA, "product_sha256": product_sha256,
                   "size": int(stat.st_size), "mtime_ns": None if product_sha256 else int(stat.st_mtime_ns),
                   "ids_sha256": _ids_digest(ids)}
    key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()
    cache_path: Path | None = None
    if cache_root is not None:
        cache_dir = Path(cache_root) / "mc-roi"
        _reject_snapshot_path(cache_dir)
        cache_path = cache_dir / f"{key[:32]}.parquet"
        sidecar = cache_path.with_suffix(".json")
        if cache_path.is_file() and sidecar.is_file():
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            if meta.get("key") == key and meta.get("parquet_sha256") == _file_sha256(cache_path):
                return pd.read_parquet(cache_path)
    dataset = ds.dataset(str(neuron_path), format="ipc")
    value_set = pa.array([int(v) for v in ids], type=pa.int64())
    roots: list[str] = []
    rois: list[str] = []
    pres: list[int] = []
    posts: list[int] = []
    seen: set[str] = set()
    for batch in dataset.to_batches(columns=[NEURON_ID_COLUMN, NEURON_ROI_INFO_COLUMN],
                                    filter=ds.field(NEURON_ID_COLUMN).isin(value_set)):
        if batch.num_rows == 0:
            continue
        body = batch.column(NEURON_ID_COLUMN)
        if body.null_count:
            raise McAdapterError(McAdapterErrorCode.SCHEMA_MISMATCH, "neuron table has null body ids")
        for bid, raw in zip(body.cast(pa.int64()).to_pylist(), batch.column(NEURON_ROI_INFO_COLUMN).to_pylist()):
            rid = str(bid)
            if rid in seen:
                raise McAdapterError(McAdapterErrorCode.SCHEMA_MISMATCH, "neuron table body id is not unique",
                                     {"root_id": rid})
            seen.add(rid)
            for roi, pre, post in parse_roi_info(raw):
                roots.append(rid)
                rois.append(roi)
                pres.append(pre)
                posts.append(post)
    frame = pd.DataFrame({"root_id": roots, "roi": rois, "pre": np.asarray(pres, dtype=np.int64),
                          "post": np.asarray(posts, dtype=np.int64)})
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".parquet.tmp")
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, cache_path)
        cache_path.with_suffix(".json").write_text(json.dumps(
            {**key_payload, "key": key, "rows": int(len(frame)), "bodies_found": len(seen),
             "parquet_sha256": _file_sha256(cache_path)}, indent=2, sort_keys=True), encoding="utf-8")
    return frame


def _entropy_bits(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(totals > 0, matrix / np.where(totals > 0, totals, 1.0), 0.0)
        logs = np.where(p > 0, np.log2(np.where(p > 0, p, 1.0)), 0.0)
    return np.where(totals[:, 0] > 0, -(p * logs).sum(axis=1), np.nan)


def roi_feature_frame(roi_long: pd.DataFrame, *, top_k: int, rank_ids: Iterable[str] | None = None) -> pd.DataFrame:
    """Per-body ROI features + the top-ROI categorical columns (one row per body in ``roi_long``).

    ``rank_ids`` (default: every body) fixes which bodies' synapse totals pick
    the top-K base neuropils; it uses no labels.
    """
    if roi_long.empty:
        raise ValueError("no roiInfo rows for the requested bodies")
    vocab = {roi: map_mc_primary_roi(roi) for roi in sorted(set(roi_long["roi"]))}
    long = roi_long.assign(region=[vocab[r].region_id for r in roi_long["roi"]],
                           hemi=[vocab[r].side or "none" for r in roi_long["roi"]],
                           syn=roi_long["pre"] + roi_long["post"])
    ids = sorted(set(long["root_id"]), key=lambda v: (len(v), v))
    pre = long.pivot_table(index="root_id", columns="region", values="pre", aggfunc="sum", fill_value=0).reindex(ids)
    post = long.pivot_table(index="root_id", columns="region", values="post", aggfunc="sum", fill_value=0).reindex(ids)
    pre, post = pre.fillna(0).astype(np.float64), post.fillna(0).astype(np.float64)
    ranked_rows = pre.index if rank_ids is None else pre.index.intersection(pd.Index(list(rank_ids)))
    totals = (pre.loc[ranked_rows] + post.loc[ranked_rows]).sum(axis=0)
    regions = sorted(totals.index, key=lambda r: (-float(totals[r]), r))
    top = regions[: int(top_k)]
    rest = [r for r in pre.columns if r not in set(top)]
    feats: dict[str, np.ndarray] = {}
    for prefix, mat in (("pre", pre), ("post", post)):
        row_total = mat.sum(axis=1).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            for region in top:
                col = mat[region].to_numpy() if region in mat.columns else np.zeros(len(mat))
                feats[f"roi__{prefix}_frac__{region}"] = np.where(row_total > 0, col / np.where(row_total > 0, row_total, 1), np.nan)
            other = mat[rest].sum(axis=1).to_numpy() if rest else np.zeros(len(mat))
            feats[f"roi__{prefix}_frac__other"] = np.where(row_total > 0, other / np.where(row_total > 0, row_total, 1), np.nan)
        feats[f"roi__{prefix}_entropy"] = _entropy_bits(mat.to_numpy())
    by_hemi = long.pivot_table(index="root_id", columns="hemi", values="syn", aggfunc="sum", fill_value=0).reindex(ids).fillna(0)
    left = by_hemi["left"].to_numpy() if "left" in by_hemi.columns else np.zeros(len(ids))
    right = by_hemi["right"].to_numpy() if "right" in by_hemi.columns else np.zeros(len(ids))
    with np.errstate(divide="ignore", invalid="ignore"):
        feats["roi__left_share"] = np.where(left + right > 0, left / np.where(left + right > 0, left + right, 1), np.nan)
        tot_pre, tot_post = pre.sum(axis=1).to_numpy(), post.sum(axis=1).to_numpy()
        feats["degree__roi_pre_share"] = np.where(tot_pre + tot_post > 0, tot_pre / np.maximum(tot_pre + tot_post, 1), np.nan)
    feats["degree__roi_n_primary_rois"] = long.groupby("root_id")["roi"].nunique().reindex(ids).to_numpy(dtype=np.float64)
    # Top primary ROI (same rule as the adapter: most pre+post, ties on the name).
    top_rows = long.sort_values(["root_id", "syn", "roi"], ascending=[True, False, True], kind="mergesort") \
        .drop_duplicates("root_id").set_index("root_id").reindex(ids)
    mapped = [vocab[r] for r in top_rows["roi"]]
    out = pd.DataFrame(feats)
    out.insert(0, "root_id", ids)
    out["cns_division"] = [m.division for m in mapped]
    out["subdivision"] = [m.subdivision for m in mapped]
    out["primary_neuropil"] = [m.region_id[len(m.division) + 1:] for m in mapped]
    out["total_primary_synapses"] = long.groupby("root_id")["syn"].sum().reindex(ids).to_numpy(dtype=np.int64)
    return out


# =========================================================================== populations + labels


def _hash_rank(salt: str, value: str) -> str:
    return hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()


def cap_per_type(frame: pd.DataFrame, *, max_per_type: int, salt: str) -> pd.DataFrame:
    """Keep at most ``max_per_type`` rows per cell type, chosen in sha256(salt:root_id) order."""
    ranked = frame.assign(_h=[_hash_rank(salt, r) for r in frame["root_id"]])
    ranked = ranked.sort_values(["cell_type", "_h"], kind="mergesort")
    ranked = ranked.groupby("cell_type", sort=False, group_keys=False).head(int(max_per_type))
    return ranked.drop(columns=["_h"]).sort_values("root_id", key=lambda s: s.str.zfill(24), kind="mergesort") \
        .reset_index(drop=True)


def base_population(nodes: pd.DataFrame, roi_feats: pd.DataFrame, config: McRealModelConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    """Typed, allowed-status, neuronal bodies with >= min_primary_synapses primary-ROI synapses."""
    stats: dict[str, int] = {"annotated_bodies": int(len(nodes))}
    frame = nodes[nodes["cell_type"].notna()]
    stats["typed"] = int(len(frame))
    allowed = {s.lower() for s in config.allowed_statuses}
    frame = frame[frame["status"].str.lower().isin(allowed)]
    stats["allowed_status"] = int(len(frame))
    frame = frame[frame["super_class"] != "non_neuronal"]
    stats["neuronal"] = int(len(frame))
    frame = frame.merge(roi_feats, on="root_id", how="inner")
    frame = frame[frame["total_primary_synapses"] >= int(config.min_primary_synapses)]
    stats["enough_primary_synapses"] = int(len(frame))
    return frame.reset_index(drop=True), stats


def _keep_classes_with_types(frame: pd.DataFrame, min_types: int) -> tuple[pd.DataFrame, dict[str, int]]:
    types_per = frame.groupby("label")["cell_type"].nunique()
    keep = sorted(types_per[types_per >= int(min_types)].index)
    dropped = {str(k): int(v) for k, v in types_per.items() if k not in set(keep)}
    return frame[frame["label"].isin(keep)].reset_index(drop=True), dropped


def label_frame(target: str, population: pd.DataFrame, *, nt_ground_truth: pd.DataFrame | None,
                config: McRealModelConfig, nt_literature: pd.DataFrame | None = None
                ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """``population`` rows that carry the target label (column ``label``), classes with enough types, capped.

    ``nt_literature``: (root_id, label) rows from ``flybrain_nt_ground_truth.label_neurons`` for the
    ``nt_literature*`` targets.
    """
    info: dict[str, Any] = {}
    if target in NT_LITERATURE_TARGETS:
        if nt_literature is None:
            raise ValueError(f"{target} needs the literature NT labels (flybrain_nt_ground_truth)")
        lit = nt_literature[["root_id", "label"]].astype({"root_id": str})
        frame = population.merge(lit, on="root_id", how="inner")
    elif target == TARGET_NT_GROUND_TRUTH:
        if nt_ground_truth is None:
            raise ValueError("nt_ground_truth target needs the body-neurotransmitter table")
        gt = nt_ground_truth[["root_id", "ground_truth"]]
        gt = gt[gt["ground_truth"].map(lambda v: not _missing(v))]
        frame = population.merge(gt, on="root_id", how="inner")
        frame["label"] = [nt_short_code(v) for v in frame["ground_truth"]]
        frame = frame.drop(columns=["ground_truth"])
        # Label is per cell type in this release; fail closed if that stops being true.
        per_type = frame.groupby("cell_type")["label"].nunique()
        info["types_with_multiple_labels"] = int((per_type > 1).sum())
    elif target == TARGET_SUPER_CLASS:
        frame = population[~population["super_class"].isin(NON_LABEL_SUPER_CLASSES)].copy()
        frame["label"] = frame["super_class"]
    elif target == TARGET_CELL_CLASS:
        frame = population[population["cell_class"].notna()].copy()
        frame["label"] = frame["cell_class"]
    else:
        raise ValueError(f"label_frame does not build {target!r} (legacy targets use the mc sample builder)")
    frame, dropped = _keep_classes_with_types(frame, config.min_types_per_class)
    info["classes_dropped_few_types"] = dropped
    info["rows_before_cap"] = int(len(frame))
    frame = cap_per_type(frame, max_per_type=config.max_per_type, salt=config.cap_salt)
    info["rows"] = int(len(frame))
    info["types"] = int(frame["cell_type"].nunique())
    info["label_counts"] = {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()}
    info["types_per_label"] = {str(k): int(v) for k, v in frame.groupby("label")["cell_type"].nunique().items()}
    return frame, info


def legacy_label_frame(target: str, population: pd.DataFrame, config: McRealModelConfig, *,
                       explicit_paths: Mapping[str, str | Path] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """connectivity_tier / region_specialization_tier labels from the unchanged mc sample builder."""
    import flybrain_brain_cluster_mc_samples as mcs

    kwargs: dict[str, Any] = {}
    if explicit_paths is not None:
        kwargs = {"meta_path": explicit_paths[ROLE_META], "edgelist_path": explicit_paths.get(ROLE_EDGELIST),
                  "neuron_path": explicit_paths[ROLE_NEURON_ROI_INFO],
                  "dataset_meta_path": explicit_paths[ROLE_DATASET_META]}
        if target == TARGET_REGION_SPECIALIZATION:
            kwargs.pop("edgelist_path")
    cfg = mcs.McSampleBuildConfig(objective=target, storage_root=config.storage_root, stamp_dir=config.stamp_dir,
                                  verify_hashes=config.verify_hashes, max_samples=int(config.legacy_max_samples),
                                  max_regions=int(config.legacy_max_regions),
                                  min_primary_synapses=int(config.min_primary_synapses),
                                  max_label_share=1.0, **kwargs)
    payload = mcs.build_mc_training_samples(target, cfg)
    rows = pd.DataFrame({
        "root_id": [str(s["metadata"]["root_id"]) for s in payload["samples"]],
        "label": [str(s["expected_label"]) for s in payload["samples"]],
    })
    frame = population.merge(rows, on="root_id", how="inner")
    info = {"legacy_samples": int(len(rows)), "rows_before_cap": int(len(frame)),
            "legacy_label_definition": payload["metadata"].get("label_definition"),
            "legacy_fingerprint": payload["metadata"].get("input_fingerprint") or payload["metadata"].get("fingerprint"),
            "high_connectivity_threshold": payload["metadata"].get("high_connectivity_threshold")}
    frame = cap_per_type(frame, max_per_type=config.max_per_type, salt=config.cap_salt)
    info["rows"] = int(len(frame))
    info["types"] = int(frame["cell_type"].nunique())
    info["label_counts"] = {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()}
    return frame, info


# =========================================================================== feature selection


def allowed_columns(target: str, columns: Iterable[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Columns a model may see for ``target`` + why each other column was dropped."""
    cols = [c for c in columns if c not in NEVER_FEATURES and c != "total_primary_synapses" and c != "status"]
    registry = set(fwf.excluded_feature_names(cols, target))
    extra = {c for c in cols if c in EXTRA_EXCLUDED.get(target, frozenset())}
    kept = [c for c in cols if c not in registry and c not in extra]
    fwf.assert_features_allowed(kept, target)
    return kept, {"registry": sorted(registry), "mc_extra": sorted(extra)}


def input_text(row: Mapping[str, Any], categorical: Sequence[str]) -> str:
    """Legacy-style ``key value`` text over the allowed categorical columns (NB view)."""
    parts = [f"dataset {TEXT_TAG}"]
    for key in categorical:
        value = row.get(key)
        parts.append(f"{key} {fwf.slug(value) if not _missing(value) else 'unknown'}")
    return " ".join(parts)


# =========================================================================== eval dataset


@dataclass
class McTargetBuild:
    target: str
    data: Any  # flybrain_model_eval.EvalDataset
    info: dict[str, Any]


def _group(value: Any) -> str | None:
    return None if _missing(value) else str(value)


def _held_out_mask_ids(nodes: pd.DataFrame, frame: pd.DataFrame, held_out: set[str]) -> list[str]:
    """Every node sharing a held-out sample's cell type, hemilineage group or supertype (plus the samples)."""
    rows = frame[frame["root_id"].isin(held_out)]
    types = set(rows["cell_type"].dropna())
    lineages = set(rows["hemilineage_group"].dropna())
    supertypes = set(rows["supertype"].dropna())
    hit = nodes["cell_type"].isin(types) | nodes["hemilineage_group"].isin(lineages) | nodes["supertype"].isin(supertypes)
    return sorted(set(nodes.loc[hit, "root_id"]) | held_out)


def build_target_dataset(target: str, config: McRealModelConfig, eval_config: Any, *,
                         inputs: McInputs | None = None, explicit_paths: Mapping[str, str | Path] | None = None,
                         wiring_params: fwf.WiringFeatureParams | None = None,
                         feature_filter: Sequence[str] | None = None, log=print,
                         nt_root: str | Path = ntgt.DEFAULT_NT_GT_ROOT) -> McTargetBuild:
    """Labels + grouped-split plan + (masked) wiring and ROI features -> ``EvalDataset``.

    ``feature_filter``: optional family names to keep (e.g. to run a wiring-only variant).
    """
    import flybrain_model_eval as fme

    if target not in MC_REAL_TARGETS:
        raise ValueError(f"target must be one of: {', '.join(MC_REAL_TARGETS)}")
    fwf.objective_exclusions(target)  # fail closed before any work
    notes_extra: dict[str, Any] = ftr.provenance_notes(MC_SYMBOL, target)
    t0 = time.time()
    inputs = inputs or open_inputs(config, paths=explicit_paths)
    check_mc_roi_vocabulary(inputs.paths[ROLE_DATASET_META])
    nodes = load_node_table(inputs.paths[ROLE_META])
    log(f"[mc] nodes {len(nodes)} ({time.time() - t0:.0f}s)")
    roi_long = load_roi_long(inputs.paths[ROLE_NEURON_ROI_INFO], nodes["root_id"].tolist(),
                             product_sha256=inputs.product_sha256.get(ROLE_NEURON_ROI_INFO),
                             cache_root=config.cache_root)
    typed_ids = nodes.loc[nodes["cell_type"].notna(), "root_id"]
    roi_feats = roi_feature_frame(roi_long, top_k=config.top_k_rois, rank_ids=typed_ids)
    del roi_long
    population, pop_stats = base_population(nodes, roi_feats, config)
    log(f"[mc] population {len(population)} ({time.time() - t0:.0f}s)")

    if target in LEGACY_TARGETS:
        frame, label_info = legacy_label_frame(target, population, config, explicit_paths=explicit_paths)
    else:
        nt = None
        lit = None
        if target == TARGET_NT_GROUND_TRUTH:
            nt = load_mc_nt_predictions(inputs.paths[ROLE_NT_PREDICTION], body_ids=population["root_id"].tolist())
            nt = nt.rename(columns={"body": "root_id"})[["root_id", "ground_truth"]]  # nothing else leaves here
        if target in NT_LITERATURE_TARGETS:
            labelled = ntgt.label_neurons(population[["root_id", "cell_type"]], dataset=MC_SYMBOL,
                                          source=ntgt.open_nt_ground_truth(nt_root), target=target,
                                          id_column="root_id")
            lit = labelled.frame
            notes_extra["nt_literature_coverage"] = labelled.coverage
        frame, label_info = label_frame(target, population, nt_ground_truth=nt, config=config, nt_literature=lit)
        if label_info.get("types_with_multiple_labels"):
            label_info["warning"] = "ground_truth differs within some cell types"
    if frame["label"].nunique() < 2:
        raise ValueError(f"{target}: fewer than two classes after filtering")
    log(f"[mc] {target}: {len(frame)} samples, {label_info.get('types')} types ({time.time() - t0:.0f}s)")

    frame = frame.assign(sample_id=[f"mc-{target}-{r}" for r in frame["root_id"]])
    group_values = [{"cell_type": _group(ct), "hemilineage": _group(hg), "supertype": _group(st)}
                    for ct, hg, st in zip(frame["cell_type"], frame["hemilineage_group"], frame["supertype"])]
    plan_ids = frame["sample_id"].tolist()
    plan = fme.plan_grouped_split(plan_ids, group_values, GROUP_KEYS, eval_config)
    notes: dict[str, Any] = dict(notes_extra)
    mask_ids: list[str] | None = None
    if target in MASKED_TARGETS:
        held_out = {sid.rsplit("-", 1)[1] for sid in plan["val"] + plan["test"]}
        mask_ids = _held_out_mask_ids(nodes, frame, held_out)
        notes["masked_split_ids_sha256"] = fme.split_ids_sha256(plan)
        notes["masked_partner_category_nodes"] = len(mask_ids)

    edges = fwf.EdgeSource(path=str(inputs.paths[ROLE_EDGELIST]), pre="body_pre", post="body_post", weight="weight",
                           provenance={"manifest_sha256": inputs.manifest_sha256,
                                       "sha256": inputs.product_sha256.get(ROLE_EDGELIST)})
    params = wiring_params or fwf.WiringFeatureParams(two_hop=bool(config.two_hop))
    wiring = fwf.build_wiring_features(dataset=MC_SYMBOL, objective=target, edges=edges, nodes=nodes,
                                       id_column="root_id", category_column="superclass_raw",
                                       vocab_map=fwf.harmonize_super_class, params=params,
                                       cache_root=config.cache_root, mask_category_ids=mask_ids)
    log(f"[mc] wiring {wiring.frame.shape} cache={wiring.cache_path} ({time.time() - t0:.0f}s)")
    frame = frame.merge(wiring.frame, left_on="root_id", right_on=fwf.NODE_ID_COLUMN, how="left")
    families = {"degree", "out_comp", "in_comp", "recip", "out2_comp", "in2_comp", "roi"}
    if feature_filter is not None and SPARSITY_FAMILY in set(feature_filter):
        # Control only (never in the default set): how many partner categories a neuron touches,
        # a size proxy hidden inside the composition vectors (contract doc, "Real models").
        for side in ("out", "in"):
            comp = [c for c in frame.columns if c.startswith(f"{side}_comp__") and not c.endswith("__entropy")]
            frame[f"{SPARSITY_FAMILY}__n_nonzero_{side}_comp"] = (frame[comp].fillna(0.0) > 0).sum(axis=1).astype(float)
        families.add(SPARSITY_FAMILY)
    candidates = [c for c in frame.columns if c.split("__", 1)[0] in families or c in CATEGORICAL_COLUMNS]
    if feature_filter is not None:
        keep_fams = set(feature_filter)
        candidates = [c for c in candidates if fwf.feature_family(c) in keep_fams]
    feature_columns, dropped = allowed_columns(target, candidates)
    categorical = [c for c in CATEGORICAL_COLUMNS if c in feature_columns]
    for column in categorical:
        frame[column] = frame[column].map(lambda v: "unknown" if _missing(v) else fwf.slug(v)).astype(object)
    frame = frame.reset_index(drop=True)
    records = frame[categorical].to_dict(orient="records") if categorical else [{}] * len(frame)
    texts = tuple(input_text(row, categorical) for row in records)
    notes.update({
        "dataset_version": "male-cns_v1.0",
        "manifest_sha256": inputs.manifest_sha256,
        "product_sha256": dict(inputs.product_sha256),
        "wiring_fingerprint": wiring.fingerprint,
        "wiring_cache_path": wiring.cache_path,
        "wiring_excluded_features": list(wiring.excluded_features),
        "dropped_features": dropped,
        "population": pop_stats,
        "labels": label_info,
        "config": config.as_dict(),
        "feature_filter": None if feature_filter is None else sorted(feature_filter),
        "text_keys": ["dataset", *categorical],
    })
    # Group values come from the split plan's inputs (the hemilineage *group*, not the input token).
    plan_groups = dict(zip(plan_ids, group_values))
    data = fme.EvalDataset(
        dataset=MC_SYMBOL, target=target, sample_ids=tuple(frame["sample_id"]),
        labels=frame["label"].astype(str).to_numpy(dtype=object),
        features=frame[feature_columns].copy(), group_keys=GROUP_KEYS,
        group_values=tuple(plan_groups[sid] for sid in frame["sample_id"]), text=texts, notes=notes)
    info = {"elapsed_seconds": round(time.time() - t0, 1), "n_samples": len(frame), "n_features": len(feature_columns),
            "plan_counts": {k: len(v) for k, v in plan.items()}, **{k: notes[k] for k in ("labels", "population")}}
    return McTargetBuild(target, data, info)


# =========================================================================== harness


def default_models() -> tuple[Any, ...]:
    import flybrain_model_eval as fme

    return (
        fme.ModelSpec("nb", ({"alpha": 0.5}, {"alpha": 1.0}), "temperature"),
        fme.ModelSpec("logreg", ({"C": 0.1}, {"C": 1.0}, {"C": 1.0, "class_weight": "balanced"}), "temperature"),
        fme.ModelSpec("hgb", ({"learning_rate": 0.1, "max_iter": 300},
                              {"learning_rate": 0.1, "max_iter": 300, "class_weight": "balanced"}), "temperature"),
    )


def run_target(target: str, config: McRealModelConfig = McRealModelConfig(), *, eval_config: Any = None,
               feature_filter: Sequence[str] | None = None, log=print) -> dict[str, Any]:
    import flybrain_model_eval as fme

    eval_config = eval_config or fme.EvalConfig(models=default_models(), report_root=DEFAULT_REPORT_ROOT,
                                                run_label="wiring-roi-v1", n_threads=8)
    build = build_target_dataset(target, config, eval_config, feature_filter=feature_filter, log=log)
    log(f"[mc] {target}: evaluating {build.info['n_samples']} samples x {build.info['n_features']} features")
    report = ftr.run_gated_evaluation(build.data, eval_config)
    report["mc_build"] = build.info
    if eval_config.report_root:
        ftr.write_gated_report(report, eval_config.report_root, run_label=eval_config.run_label)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    import flybrain_model_eval as fme

    parser = argparse.ArgumentParser(description="Train + evaluate mc real-model targets (grouped, gated).")
    parser.add_argument("--target", action="append", choices=MC_REAL_TARGETS, help="repeatable; default all")
    parser.add_argument("--storage-root", default=os.environ.get("LOCI_FLYBRAIN_STORAGE_ROOT", "/mnt/f/.flybrain"))
    parser.add_argument("--stamp-dir", default=DEFAULT_STAMP_DIR)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--report-root", default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--run-label", default="wiring-roi-v1")
    parser.add_argument("--families", help="comma-separated feature families/columns to keep (variant runs)")
    parser.add_argument("--max-per-type", type=int, default=20)
    parser.add_argument("--models", default="nb,logreg,hgb")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--summary-out", help="append summary rows (JSONL) here")
    args = parser.parse_args(argv)
    config = McRealModelConfig(storage_root=args.storage_root, stamp_dir=args.stamp_dir, cache_root=args.cache_root,
                               max_per_type=args.max_per_type)
    wanted = set(args.models.split(","))
    models = tuple(m for m in default_models() if m.backend in wanted)
    eval_config = fme.EvalConfig(models=models, report_root=args.report_root, run_label=args.run_label,
                                 n_bootstrap=args.n_bootstrap, ablation=not args.no_ablation, n_threads=8,
                                 cv_folds=args.cv_folds)
    families = None if not args.families else [f.strip() for f in args.families.split(",") if f.strip()]
    failures = 0
    for target in args.target or MC_REAL_TARGETS:
        t0 = time.time()
        try:
            report = run_target(target, config, eval_config=eval_config, feature_filter=families,
                                log=lambda m: print(m, flush=True))
        except Exception:  # keep going: one target failing must not cost the others their lock slot
            import traceback

            print(f"[mc] {target} FAILED", flush=True)
            traceback.print_exc()
            failures += 1
            continue
        for row in report["summary"]:
            row = {**row, "run_label": args.run_label}
            print(json.dumps(row, sort_keys=True), flush=True)
            if args.summary_out:
                with open(args.summary_out, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"[mc] {target} done in {time.time() - t0:.0f}s maxrss_MB "
              f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}", flush=True)
    return 1 if failures else 0


__all__ = [
    "CATEGORICAL_COLUMNS",
    "EXTRA_EXCLUDED",
    "GROUP_KEYS",
    "MC_REAL_TARGETS",
    "McInputs",
    "McRealModelConfig",
    "McTargetBuild",
    "allowed_columns",
    "base_population",
    "build_target_dataset",
    "cap_per_type",
    "default_models",
    "label_frame",
    "legacy_label_frame",
    "load_node_table",
    "load_roi_long",
    "open_inputs",
    "parse_roi_info",
    "roi_feature_frame",
    "run_target",
]


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
