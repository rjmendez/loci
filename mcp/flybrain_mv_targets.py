"""MANC v1.0 (``mv``) real-model targets: structured wiring features + the grouped evaluation harness.

This module turns the manifest-verified ``mv`` snapshot into
``flybrain_model_eval.EvalDataset`` objects and runs ``run_evaluation`` on
them (nb / logreg / hgb, grouped split, trivial baselines, bootstrap CIs,
label-shuffle and random-split controls, drop-one-family ablation). Targets:

* ``cell_class``: the curated MANC ``class`` (intrinsic / sensory / ascending /
  descending / motor / efferent), harmonized with
  ``flybrain_wiring_features.harmonize_super_class``, predicted from wiring
  only. The partner category of the wiring features *is* this label, so it is
  masked (val + test samples and every node sharing a held-out cell type,
  homolog group or serial group) before any feature is computed.
* ``hemilineage``: the developmental hemilineage (Truman/Lacin nomenclature;
  ground truth from lineage tracing, not from connectivity) of VNC-born
  neurons, predicted from wiring + soma position + birth time. Split groups
  are cell type, homolog group and serial group (hemilineage cannot be a split
  key when it is the label). Partner hemilineage composition (``*_comp_hl``)
  is masked the same way.
* ``neurotransmitter_dominance``: MANC's per-neuron transmitter *prediction*
  (``predictedNt``, ach / gaba / glut; there is no NT ground-truth column in
  MANC v1.0) from wiring + class / birthtime / soma position. Hemilineage is
  never an input (one fast transmitter per hemilineage) and is a split key.
  Partner predicted-NT composition (``*_comp_pnt``) is masked like above.
* ``connectivity_tier`` / ``region_specialization_tier``: the existing labels
  from ``flybrain_brain_cluster_mv_samples.collect_mv_samples`` (unchanged
  definitions, all candidates, no cap or balancing).

Wiring features (``flybrain_wiring_features``) use the traced-to-traced
adjacency (``traced-connections.csv``, one row per pair) with the node table
= every Traced body (partner category: harmonized class), plus a derived
neuropil edge table built from ``traced-connections-per-roi.csv``: only VNC
neuropil ROIs are kept (nerves, the neck connective ``CV``, the ``GF`` tract
and ``NotPrimary`` are dropped) and the side suffix is stripped, so
``out_np``/``in_np`` are fractions over the 13 base neuropils. The per-ROI
file is not an adapter role; its sha256 is checked against the verified
manifest entry (role ``edgelist_per_roi``) before it is read.

Leakage rules on top of the registered exclusions (``EXTRA_EXCLUDED``): body
ids, cell type, instance, systematic type, homolog/serial group, tags and
synonyms are never features; ``cell_class`` sees no annotation at all (soma
neuromere / side and birthtime are null exactly for descending and sensory
neurons); ``hemilineage`` never sees class/subclass; NT never sees
hemilineage or the partner-hemilineage composition. Nothing is written into
``snapshots/``: derived tables and wiring caches go under ``<cache_root>``,
reports under ``<report_root>/mv/<target>/<run_label>/``.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import flybrain_wiring_features as fwf
from flybrain_brain_cluster_mv_samples import (
    STRUCTURED_FEATURES,
    MvSampleBuildConfig,
    _clean,
    _hemilineage,
    _side,
    collect_mv_samples,
    load_mv_candidates,
)
from flybrain_mv_adapter import (
    MV_MANIFEST_RELATIVE_PATH,
    MV_VERSION_ID,
    ROI_KIND_NEUROPIL,
    ROLE_EDGELIST,
    ROLE_META,
    MvAdapterError,
    MvAdapterErrorCode,
    load_mv_neurons,
    map_mv_roi,
    nt_short_code,
    open_mv_snapshot,
    sha256_file,
)

MV_SYMBOL = "mv"
TARGET_CELL_CLASS = "cell_class"
TARGET_HEMILINEAGE = "hemilineage"
TARGET_NT = "neurotransmitter_dominance"
TARGET_CONNECTIVITY = "connectivity_tier"
TARGET_REGION_SPECIALIZATION = "region_specialization_tier"
MV_REAL_TARGETS: tuple[str, ...] = (
    TARGET_CELL_CLASS,
    TARGET_HEMILINEAGE,
    TARGET_NT,
    TARGET_CONNECTIVITY,
    TARGET_REGION_SPECIALIZATION,
)
LEGACY_TARGETS = frozenset({TARGET_NT, TARGET_CONNECTIVITY, TARGET_REGION_SPECIALIZATION})

ROLE_EDGELIST_PER_ROI = "edgelist_per_roi"
PER_ROI_RELATIVE_PATH = "source/manc-traced-adjacencies-v1.0/traced-connections-per-roi.csv"
NEUROPIL_EDGES_SCHEMA = "flybrain-mv-neuropil-edges/v1"
NOT_PRIMARY_ROI = "NotPrimary"

DEFAULT_STORAGE_ROOT = "/mnt/f/.flybrain"
DEFAULT_CACHE_ROOT = "/mnt/f/.flybrain/cache"
DEFAULT_REPORT_ROOT = "/mnt/f/.flybrain/logs/real-models-20260924T174122Z"
DEFAULT_RUN_LABEL = "wiring-v1"

# Group keys (union-find) per target. ``serial_group`` = MANC ``serial`` (serially
# homologous neurons across neuromeres), ``split_group`` = MANC ``group``
# (left/right homologs).
DEFAULT_GROUP_KEYS: tuple[str, ...] = ("cell_type", "hemilineage", "split_group", "serial_group")
GROUP_KEYS: Mapping[str, tuple[str, ...]] = {
    TARGET_CELL_CLASS: DEFAULT_GROUP_KEYS,
    TARGET_HEMILINEAGE: ("cell_type", "split_group", "serial_group"),
    TARGET_NT: DEFAULT_GROUP_KEYS,
    TARGET_CONNECTIVITY: DEFAULT_GROUP_KEYS,
    TARGET_REGION_SPECIALIZATION: DEFAULT_GROUP_KEYS,
}

# Categorical annotation features per target (plain names so the registered
# exclusion patterns apply to them).
CATEGORICAL_FEATURES: Mapping[str, tuple[str, ...]] = {
    TARGET_CELL_CLASS: (),
    TARGET_HEMILINEAGE: ("soma_neuromere", "soma_side", "birthtime"),
    TARGET_NT: STRUCTURED_FEATURES[TARGET_NT],
    TARGET_CONNECTIVITY: STRUCTURED_FEATURES[TARGET_CONNECTIVITY],
    TARGET_REGION_SPECIALIZATION: STRUCTURED_FEATURES[TARGET_REGION_SPECIALIZATION],
}

# Partner-category feature sets: (tag, node-table category column, masked for this target?).
# tag "" is the base set (all families); tagged sets contribute composition families only.
WIRING_SETS: Mapping[str, tuple[tuple[str, str, bool], ...]] = {
    TARGET_CELL_CLASS: (("", "cls", True),),
    TARGET_HEMILINEAGE: (("", "cls", False), ("hl", "hl", True)),
    TARGET_NT: (("", "cls", False), ("pnt", "pnt", True)),
    TARGET_CONNECTIVITY: (("", "cls", False),),
    TARGET_REGION_SPECIALIZATION: (("", "cls", False),),
}
WIRING_FAMILIES = ("degree", "out_comp", "in_comp", "out_np", "in_np", "recip", "out2_comp", "in2_comp")
COMPOSITION_FAMILIES = ("out_comp", "in_comp", "out2_comp", "in2_comp")

# Never features for any target (identity / homolog keys), matched case-insensitively.
NEVER_FEATURES: tuple[str, ...] = (
    "root", "body*", "*body_id*", "cell_type", "type", "instance", "systematic*", "group", "split_group",
    "serial*", "tag", "synonyms", "sample_id", "label", "status*",
)
# Per-target exclusions on top of fwf's registered patterns (fnmatch, lower-case).
EXTRA_EXCLUDED: Mapping[str, tuple[str, ...]] = {
    # Soma neuromere/side and birthtime are null exactly for descending + sensory neurons.
    TARGET_CELL_CLASS: ("soma_*", "birthtime", "hemilineage", "*_hl__*", "subclass", "modality", "*nerve*",
                        "long_tract", "origin", "target", "prefix", "receptor*", "position*", "*_pnt__*"),
    TARGET_HEMILINEAGE: ("class", "subclass", "*_pnt__*", "predicted*", "nt*", "prefix", "*nerve*", "target",
                         "origin", "long_tract"),
    TARGET_NT: ("hemilineage", "*_hl__*", "predicted*", "nt*", "transmission"),
    TARGET_CONNECTIVITY: ("downstream", "upstream", "pre", "post", "size", "synweight", "*_hl__*", "*_pnt__*"),
    TARGET_REGION_SPECIALIZATION: ("subclass", "prefix", "target", "origin", "long_tract", "*nerve*",
                                   "serial_motif", "hemilineage", "primary_neuropil", "*_hl__*", "*_pnt__*"),
}
NON_LABEL_CLASSES = frozenset({fwf.UNKNOWN, "non_neuronal", "other"})


@dataclass(frozen=True)
class MvRealModelConfig:
    storage_root: str = DEFAULT_STORAGE_ROOT
    cache_root: str = DEFAULT_CACHE_ROOT
    # None: hash the required products (no stamps are read or written: use_stamp_cache=False).
    verify_hashes: bool | None = None
    two_hop: bool = True
    top_k_neuropils: int = 20
    min_class_samples: int = 50
    min_class_groups: int = 4
    nb_bins: int = 10
    # legacy-objective builder knobs (unchanged defaults)
    min_total_count: int = 10
    min_region_samples: int = 25
    high_connectivity_quantile: float = 0.75
    specialization_share_threshold: float = 0.9

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MvInputs:
    snapshot_root: Path | None
    paths: Mapping[str, Path]
    manifest_sha256: str | None
    product_sha256: Mapping[str, str]
    hash_verification: Mapping[str, str] = field(default_factory=dict)


# =========================================================================== inputs


def _manifest_entry(snapshot_root: Path, role: str) -> Mapping[str, Any]:
    manifest = json.loads((snapshot_root / MV_MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8"))
    entries = [f for f in manifest["integrity"]["files"] if f.get("role") == role]
    if len(entries) != 1:
        raise MvAdapterError(MvAdapterErrorCode.PRODUCT_MISSING, f"manifest lists {len(entries)} files with role {role!r}")
    return entries[0]


def verify_manifest_listed_file(snapshot_root: Path, role: str) -> tuple[Path, str]:
    """Full sha256 of a manifest-listed file the adapter has no role for (fail closed on any mismatch).

    Call only after ``open_mv_snapshot`` verified the manifest self-hash and sidecar.
    """
    entry = _manifest_entry(snapshot_root, role)
    path = (snapshot_root / str(entry["relative_path"])).resolve(strict=False)
    if snapshot_root.resolve() not in path.parents:
        raise MvAdapterError(MvAdapterErrorCode.PATH_ESCAPE, "manifest path escapes the snapshot", {"path": str(path)})
    if not path.is_file():
        raise MvAdapterError(MvAdapterErrorCode.PRODUCT_MISSING, f"{role} file missing", {"path": str(path)})
    if path.stat().st_size != int(entry["size_bytes"]):
        raise MvAdapterError(MvAdapterErrorCode.INTEGRITY_MISMATCH, f"{role} size mismatch", {"path": str(path)})
    digest = sha256_file(path)
    if digest != str(entry["sha256"]):
        raise MvAdapterError(MvAdapterErrorCode.INTEGRITY_MISMATCH, f"{role} sha256 mismatch", {"path": str(path)})
    return path, digest


def open_inputs(config: MvRealModelConfig, *, paths: Mapping[str, str | Path] | None = None) -> MvInputs:
    """Manifest-verified (read-only: no hash stamps written) meta + edge list + per-ROI edge list.

    ``paths`` (tests): explicit ``{meta, edgelist, edgelist_per_roi}`` files; no manifest.
    """
    if paths is not None:
        resolved = {role: Path(p) for role, p in paths.items()}
        for role in (ROLE_META, ROLE_EDGELIST, ROLE_EDGELIST_PER_ROI):
            if role not in resolved or not resolved[role].is_file():
                raise MvAdapterError(MvAdapterErrorCode.PRODUCT_MISSING, f"explicit {role} path missing")
        return MvInputs(None, resolved, None, {role: sha256_file(p) for role, p in sorted(resolved.items())})
    snapshot = open_mv_snapshot(config.storage_root, required_roles=(ROLE_META, ROLE_EDGELIST),
                                verify_hashes=config.verify_hashes, use_stamp_cache=False)
    per_roi, per_roi_sha = verify_manifest_listed_file(snapshot.snapshot_root, ROLE_EDGELIST_PER_ROI)
    return MvInputs(
        snapshot_root=snapshot.snapshot_root,
        paths={ROLE_META: snapshot.path(ROLE_META), ROLE_EDGELIST: snapshot.path(ROLE_EDGELIST),
               ROLE_EDGELIST_PER_ROI: per_roi},
        manifest_sha256=snapshot.manifest_sha256,
        product_sha256={**dict(snapshot.product_sha256), ROLE_EDGELIST_PER_ROI: per_roi_sha},
        hash_verification=dict(snapshot.hash_verification),
    )


def _reject_snapshot_path(path: Path) -> None:
    if "snapshots" in Path(path).resolve(strict=False).parts:
        raise ValueError(f"refusing to write under a snapshots directory: {path}")


def _sha256_file(path: Path) -> str:
    return sha256_file(Path(path))


def derive_neuropil_edges(per_roi_path: Path, *, source_sha256: str, cache_root: str | Path) -> Path:
    """Per-ROI traced edges -> parquet (pre, post, neuropil, weight) over VNC neuropils only (cached).

    ``neuropil`` is the side-stripped base neuropil region id (``vnc_legnp_t1``).
    Nerve, connective (``CV``) and tract (``GF``) rows and ``NotPrimary`` are
    dropped; any other unknown ROI name fails closed (``map_mv_roi``).
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    out_dir = Path(cache_root) / "mv-derived"
    _reject_snapshot_path(out_dir)
    out = out_dir / f"neuropil-edges-{source_sha256[:16]}.parquet"
    sidecar = out.with_suffix(".json")
    if out.is_file() and sidecar.is_file():
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if meta.get("source_sha256") == source_sha256 and meta.get("parquet_sha256") == _sha256_file(out) \
                and meta.get("schema") == NEUROPIL_EDGES_SCHEMA:
            return out
    table = pacsv.read_csv(
        str(per_roi_path),
        convert_options=pacsv.ConvertOptions(
            include_columns=["bodyId_pre", "bodyId_post", "roi", "weight"],
            column_types={"bodyId_pre": pa.int64(), "bodyId_post": pa.int64(), "roi": pa.string(),
                          "weight": pa.int64()}),
    )
    for col in ("bodyId_pre", "bodyId_post", "roi", "weight"):
        if table.column(col).null_count:
            raise MvAdapterError(MvAdapterErrorCode.SCHEMA_MISMATCH, f"per-ROI edge list has null {col}")
    rois = pc.unique(table.column("roi")).to_pylist()
    mapping: dict[str, str | None] = {}
    for roi in rois:
        if roi == NOT_PRIMARY_ROI:
            mapping[roi] = None
            continue
        mapped = map_mv_roi(roi)  # unknown names raise REGION_VOCABULARY_UNKNOWN
        mapping[roi] = mapped.region_id if mapped.kind == ROI_KIND_NEUROPIL else None
    keep_rois = sorted(r for r, v in mapping.items() if v is not None)
    kept = table.filter(pc.is_in(table.column("roi"), value_set=pa.array(keep_rois, pa.string())))
    region = pa.array([mapping[r] for r in keep_rois], pa.string())
    idx = pc.index_in(kept.column("roi"), value_set=pa.array(keep_rois, pa.string()))
    neuropil = pc.take(region, idx)
    result = pa.table({"pre": kept.column("bodyId_pre"), "post": kept.column("bodyId_post"), "neuropil": neuropil,
                       "weight": kept.column("weight")})
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    pq.write_table(result, str(tmp))
    os.replace(tmp, out)
    sidecar.write_text(json.dumps({
        "schema": NEUROPIL_EDGES_SCHEMA,
        "source_path": str(per_roi_path),
        "source_sha256": source_sha256,
        "rows_in": int(table.num_rows),
        "rows_kept": int(result.num_rows),
        "dropped_rois": sorted(r for r, v in mapping.items() if v is None),
        "neuropils": sorted(set(v for v in mapping.values() if v)),
        "parquet_sha256": _sha256_file(out),
    }, indent=2, sort_keys=True), encoding="utf-8")
    return out


# =========================================================================== node table / labels


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and value != value) or str(value).strip().lower() in {
        "", "nan", "none", "<na>", "null"}


def _group_id(prefix: str, value: Any) -> str | None:
    if _missing(value):
        return None
    try:
        return f"{prefix}_{int(float(value))}"
    except (TypeError, ValueError):
        return None


def _nt_code(value: Any) -> str:
    if _missing(value) or str(value).strip().lower() == "unknown":
        return fwf.UNKNOWN
    return nt_short_code(str(value))


def load_node_table(meta_path: Path) -> pd.DataFrame:
    """Every Traced body (untyped and glia included) with its partner categories + group keys."""
    frame = load_mv_neurons(meta_path, statuses=("Traced",),
                            columns=("type", "class", "hemilineage", "predictedNt", "group"))
    serial = _load_optional_column(meta_path, "serial", frame["bodyId"].tolist())
    return pd.DataFrame({
        "root_id": frame["bodyId"].astype(str),
        "cell_type": [None if _missing(v) else str(v).strip() for v in frame["type"]],
        "cls": [fwf.harmonize_super_class(v) for v in frame["class"]],
        "hl": [_hemilineage(v) for v in frame["hemilineage"]],
        "hemilineage": [None if _hemilineage(v) == "unknown" else _hemilineage(v) for v in frame["hemilineage"]],
        "pnt": [_nt_code(v) for v in frame["predictedNt"]],
        "split_group": [_group_id("manc_group", v) for v in frame["group"]],
        "serial_group": [_group_id("manc_serial", serial.get(b)) for b in frame["bodyId"]],
    })


def _load_optional_column(meta_path: Path, column: str, body_ids: Sequence[str]) -> dict[str, Any]:
    """``{bodyId: value}`` for a meta column outside the adapter's required set (empty if absent)."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(meta_path), format="feather")
    if column not in dataset.schema.names:
        return {}
    table = dataset.to_table(columns=["bodyId", column], filter=pc.field("status") == "Traced")
    return dict(zip((str(v) for v in table.column("bodyId").to_pylist()), table.column(column).to_pylist()))


def _categorical_row(row: Any) -> dict[str, str]:
    return {
        "soma_neuromere": _clean(row.somaNeuromere),
        "soma_side": _side(row.somaSide, row.rootSide),
        "birthtime": _clean(row.birthtime),
        "class": _clean(row.class_),
    }


def _filter_classes(frame: pd.DataFrame, *, min_samples: int, min_groups: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep labels with >= ``min_samples`` samples and >= ``min_groups`` distinct cell types."""
    counts = frame.groupby("label")["root_id"].count()
    types = frame.groupby("label")["cell_type"].nunique()
    keep = sorted(l for l in counts.index if counts[l] >= min_samples and types[l] >= min_groups)
    dropped = {str(l): int(counts[l]) for l in counts.index if l not in set(keep)}
    return frame[frame["label"].isin(keep)].reset_index(drop=True), {"dropped_labels": dropped, "kept_labels": keep}


def label_frame(target: str, config: MvRealModelConfig, inputs: MvInputs, nodes: pd.DataFrame
                ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One row per sample: root_id, label, categorical features, group keys."""
    storage = None if inputs.snapshot_root is None else config.storage_root
    explicit = {} if inputs.snapshot_root is not None else {
        "meta_path": inputs.paths[ROLE_META], "edgelist_path": inputs.paths[ROLE_EDGELIST]}
    common = dict(storage_root=storage, verify_hashes=config.verify_hashes, use_stamp_cache=False,
                  min_total_count=config.min_total_count, min_region_samples=config.min_region_samples,
                  high_connectivity_quantile=config.high_connectivity_quantile,
                  specialization_share_threshold=config.specialization_share_threshold, max_label_share=1.0,
                  max_single_feature_accuracy=1.0, attach_features=True, **explicit)
    group_cols = nodes.set_index("root_id")[["split_group", "serial_group"]]
    info: dict[str, Any] = {}
    if target in LEGACY_TARGETS:
        collected = collect_mv_samples(target, MvSampleBuildConfig(objective=target, **common))
        rows = []
        for sample in collected["samples"]:
            meta = sample["metadata"]
            rows.append({"root_id": str(meta["root_id"]), "label": sample["expected_label"],
                         "cell_type": meta["cell_type"], "hemilineage": meta["hemilineage"],
                         **{k: sample["features"][k] for k in STRUCTURED_FEATURES[target]}})
        frame = pd.DataFrame(rows)
        info.update({"builder": "flybrain_brain_cluster_mv_samples.collect_mv_samples",
                     "extra_stats": collected["extra_stats"], "filter_stats": collected["stats"]})
    else:
        loaded = load_mv_candidates(MvSampleBuildConfig(objective=TARGET_CONNECTIVITY, **common))
        cand = loaded["frame"]
        rows = []
        for row in cand.itertuples(index=False):
            base = {"root_id": str(row.bodyId), "cell_type": str(row.type).strip(),
                    "hemilineage": _hemilineage(row.hemilineage), **_categorical_row(row)}
            if target == TARGET_CELL_CLASS:
                base["label"] = fwf.harmonize_super_class(row.class_)
            else:
                base["label"] = _hemilineage(row.hemilineage)
            rows.append(base)
        frame = pd.DataFrame(rows)
        if target == TARGET_CELL_CLASS:
            frame = frame[~frame["label"].isin(NON_LABEL_CLASSES)]
        else:
            frame = frame[frame["label"] != "unknown"]
        frame, kept = _filter_classes(frame, min_samples=config.min_class_samples, min_groups=config.min_class_groups)
        info.update({"builder": "flybrain_mv_targets.label_frame", "filter_stats": loaded["stats"], **kept})
    frame = frame.join(group_cols, on="root_id")
    frame["cell_type"] = frame["cell_type"].map(lambda v: None if _missing(v) or v == "unknown" else v)
    frame["hemilineage"] = frame["hemilineage"].map(lambda v: None if _missing(v) or v == "unknown" else v)
    frame = frame.sort_values("root_id", key=lambda s: s.str.zfill(20), kind="mergesort").reset_index(drop=True)
    info["label_counts"] = {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()}
    info["n_samples"] = int(len(frame))
    info["n_cell_types"] = int(frame["cell_type"].nunique())
    return frame, info


# =========================================================================== features


def allowed_columns(target: str, columns: Iterable[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Columns a model may see for ``target`` + why every other column was dropped."""
    cols = list(columns)
    never = sorted(c for c in cols if any(fnmatch.fnmatchcase(c.lower(), p) for p in NEVER_FEATURES))
    rest = [c for c in cols if c not in set(never)]
    registry = set(fwf.excluded_feature_names(rest, target))
    extra = {c for c in rest if any(fnmatch.fnmatchcase(c.lower(), p) for p in EXTRA_EXCLUDED.get(target, ()))}
    kept = [c for c in rest if c not in registry and c not in extra]
    fwf.assert_features_allowed(kept, target)
    return kept, {"never": never, "registry": sorted(registry), "mv_extra": sorted(extra)}


def held_out_mask_ids(nodes: pd.DataFrame, frame: pd.DataFrame, held_out: set[str], group_keys: Sequence[str]
                      ) -> list[str]:
    """Held-out samples + every node sharing one of their group-key values (cell type, homolog/serial group)."""
    rows = frame[frame["root_id"].isin(held_out)]
    hit = pd.Series(False, index=nodes.index)
    for key in group_keys:
        if key in nodes.columns and key in rows.columns:
            values = set(rows[key].dropna())
            if values:
                hit |= nodes[key].isin(values)
    return sorted(set(nodes.loc[hit, "root_id"]) | set(held_out))


def binned_text(frame: pd.DataFrame, numeric: Sequence[str], categorical: Sequence[str], train_rows: np.ndarray,
                *, bins: int) -> tuple[str, ...]:
    """NB view: ``key key_<value>`` tokens (fused so the NB tokenizer keeps key and value together).

    Numeric features are cut at quantiles fitted on the train rows only; NaN -> ``na``.
    """
    parts: list[list[str]] = []
    for col in categorical:
        vals = frame[col].map(lambda v: "unknown" if _missing(v) else fwf.slug(v))
        parts.append([f"{col} {col}_{v}" for v in vals])
    for col in numeric:
        values = frame[col].to_numpy(dtype=float)
        train = values[train_rows]
        train = train[np.isfinite(train)]
        edges = np.unique(np.quantile(train, np.linspace(0, 1, bins + 1)[1:-1])) if len(train) else np.zeros(0)
        idx = np.searchsorted(edges, values, side="right")
        parts.append([f"{col} {col}_{'na' if not np.isfinite(v) else f'b{i}'}" for v, i in zip(values, idx)])
    return tuple(" ".join(row) for row in zip(*parts)) if parts else tuple("" for _ in range(len(frame)))


@dataclass
class MvTargetBuild:
    target: str
    data: Any  # flybrain_model_eval.EvalDataset
    info: dict[str, Any]


def build_target_dataset(target: str, config: MvRealModelConfig, eval_config: Any, *,
                         inputs: MvInputs | None = None, explicit_paths: Mapping[str, str | Path] | None = None,
                         feature_filter: Sequence[str] | None = None, log=print) -> MvTargetBuild:
    """Labels + grouped-split plan + (masked) wiring features -> ``EvalDataset``."""
    import flybrain_model_eval as fme

    if target not in MV_REAL_TARGETS:
        raise ValueError(f"target must be one of: {', '.join(MV_REAL_TARGETS)}")
    fwf.objective_exclusions(target)  # fail closed before any work
    t0 = time.time()
    inputs = inputs or open_inputs(config, paths=explicit_paths)
    nodes = load_node_table(inputs.paths[ROLE_META])
    frame, label_info = label_frame(target, config, inputs, nodes)
    if frame["label"].nunique() < 2:
        raise ValueError(f"{target}: fewer than two classes after filtering")
    log(f"[mv] {target}: {len(frame)} samples, {label_info['n_cell_types']} types ({time.time() - t0:.0f}s)")

    keys = GROUP_KEYS[target]
    frame = frame.assign(sample_id=[f"mv-{target}-{r}" for r in frame["root_id"]])
    group_values = [{k: row[k] for k in keys} for row in frame[list(keys)].to_dict(orient="records")]
    plan = fme.plan_grouped_split(frame["sample_id"].tolist(), group_values, keys, eval_config)
    held_out = {sid.rsplit("-", 1)[1] for sid in plan["val"] + plan["test"]}

    neuropil_path = derive_neuropil_edges(inputs.paths[ROLE_EDGELIST_PER_ROI],
                                          source_sha256=inputs.product_sha256[ROLE_EDGELIST_PER_ROI],
                                          cache_root=config.cache_root)
    edges = fwf.EdgeSource(path=str(inputs.paths[ROLE_EDGELIST]), pre="bodyId_pre", post="bodyId_post",
                           weight="weight", format="csv",
                           provenance={"manifest_sha256": inputs.manifest_sha256,
                                       "sha256": inputs.product_sha256.get(ROLE_EDGELIST)})
    np_edges = fwf.EdgeSource(path=str(neuropil_path), pre="pre", post="post", weight="weight", neuropil="neuropil",
                              unique_pairs=False, format="parquet",
                              provenance={"manifest_sha256": inputs.manifest_sha256,
                                          "source_sha256": inputs.product_sha256.get(ROLE_EDGELIST_PER_ROI),
                                          "schema": NEUROPIL_EDGES_SCHEMA})
    notes: dict[str, Any] = {"wiring": []}
    merged = frame
    masked_any = False
    for tag, category, masked in WIRING_SETS[target]:
        mask_ids = held_out_mask_ids(nodes, frame, held_out, keys) if masked else None
        params = fwf.WiringFeatureParams(two_hop=bool(config.two_hop), top_k_neuropils=int(config.top_k_neuropils),
                                         category_tag=tag)
        wiring = fwf.build_wiring_features(
            dataset=MV_SYMBOL, objective=target, edges=edges, nodes=nodes, id_column="root_id",
            category_column=category, params=params, neuropil_edges=np_edges if not tag else None,
            cache_root=config.cache_root, mask_category_ids=mask_ids)
        cols = [c for c in wiring.frame.columns if c != fwf.NODE_ID_COLUMN]
        if tag:  # tagged sets only add their composition families
            cols = [c for c in cols if fwf.feature_family(c) in {f"{fam}_{tag}" for fam in COMPOSITION_FAMILIES}]
        sub = wiring.frame[[fwf.NODE_ID_COLUMN, *cols]]
        merged = merged.merge(sub, left_on="root_id", right_on=fwf.NODE_ID_COLUMN, how="left").drop(
            columns=[fwf.NODE_ID_COLUMN])
        masked_any = masked_any or masked
        notes["wiring"].append({"tag": tag, "category": category, "masked": masked,
                                "masked_nodes": None if mask_ids is None else len(mask_ids),
                                "fingerprint": wiring.fingerprint, "cache_path": wiring.cache_path,
                                "excluded_by_objective": len(wiring.excluded_features),
                                "n_columns": len(cols)})
        log(f"[mv] wiring set {tag or 'base'}({category}) masked={masked}: {len(cols)} cols ({time.time() - t0:.0f}s)")
    if masked_any:
        notes["masked_split_ids_sha256"] = fme.split_ids_sha256(plan)

    wiring_cols = [c for c in merged.columns if fwf.feature_family(c).split("_hl")[0].split("_pnt")[0]
                   in WIRING_FAMILIES and "__" in c]
    candidates = wiring_cols + [c for c in CATEGORICAL_FEATURES[target] if c in merged.columns]
    if feature_filter is not None:
        wanted = set(feature_filter)
        candidates = [c for c in candidates if fwf.feature_family(c) in wanted or c in wanted]
    feature_columns, dropped = allowed_columns(target, candidates)
    categorical = [c for c in CATEGORICAL_FEATURES[target] if c in feature_columns]
    numeric = [c for c in feature_columns if c not in set(categorical)]
    for col in categorical:
        merged[col] = merged[col].map(lambda v: "unknown" if _missing(v) else fwf.slug(v)).astype(object)
    merged = merged.reset_index(drop=True)
    position = {sid: i for i, sid in enumerate(merged["sample_id"])}
    train_rows = np.asarray(sorted(position[s] for s in plan["train"]), dtype=np.int64)
    texts = binned_text(merged, numeric, categorical, train_rows, bins=int(config.nb_bins))
    notes.update({
        "dataset_version": MV_VERSION_ID,
        "manifest_sha256": inputs.manifest_sha256,
        "product_sha256": dict(inputs.product_sha256),
        "hash_verification": dict(inputs.hash_verification),
        "neuropil_edges": str(neuropil_path),
        "dropped_features": dropped,
        "labels": label_info,
        "group_keys": list(keys),
        "config": config.as_dict(),
        "feature_filter": None if feature_filter is None else sorted(feature_filter),
        "nb_text": f"fused key_value tokens; numeric features cut at {config.nb_bins}-quantiles of the train split",
    })
    plan_groups = dict(zip(frame["sample_id"], group_values))
    data = fme.EvalDataset(
        dataset=MV_SYMBOL, target=target, sample_ids=tuple(merged["sample_id"]),
        labels=merged["label"].astype(str).to_numpy(dtype=object),
        features=merged[feature_columns].copy(), group_keys=keys,
        group_values=tuple(plan_groups[sid] for sid in merged["sample_id"]), text=texts, notes=notes)
    info = {"elapsed_seconds": round(time.time() - t0, 1), "n_samples": int(len(merged)),
            "n_features": len(feature_columns), "n_categorical": len(categorical),
            "plan_counts": {k: len(v) for k, v in plan.items()}, "labels": label_info,
            "families": fwf.feature_families(feature_columns)}
    return MvTargetBuild(target, data, info)


# =========================================================================== harness


def default_models(target: str | None = None) -> tuple[Any, ...]:
    import flybrain_model_eval as fme

    many_classes = target == TARGET_HEMILINEAGE
    hgb_grid = ({"learning_rate": 0.1, "max_iter": 200 if many_classes else 300},)
    if not many_classes:
        hgb_grid = hgb_grid + ({"learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 15,
                                "l2_regularization": 1.0},)
    return (
        fme.ModelSpec("nb", ({"alpha": 0.5}, {"alpha": 1.0}), "temperature"),
        fme.ModelSpec("logreg", ({"C": 0.1}, {"C": 1.0}), "temperature"),
        fme.ModelSpec("hgb", hgb_grid, "temperature"),
    )


def default_eval_config(target: str, *, report_root: str | None = DEFAULT_REPORT_ROOT,
                        run_label: str = DEFAULT_RUN_LABEL, quick: bool = False) -> Any:
    import flybrain_model_eval as fme

    return fme.EvalConfig(models=default_models(target), cv_folds=3 if (quick or target == TARGET_HEMILINEAGE) else 5,
                          n_bootstrap=200 if quick else 1000, report_root=report_root, run_label=run_label,
                          n_threads=8)


def shuffle_null_control(data: Any, report: Mapping[str, Any], eval_config: Any, *, n_permutations: int = 3,
                         tolerance: float = 0.02) -> dict[str, Any]:
    """Supplementary label-shuffle control that is robust to a train/test prior shift.

    The harness compares one shuffled refit against the *train*-majority label
    scored on test. Under a grouped split the test class prior can differ
    from the train prior (whole hemilineages move together), and then a model
    that learned nothing but spreads its predictions over the classes can
    beat that single label by chance. Here the best params of every model are
    refit on ``n_permutations`` permutations of the train labels and scored on
    test; the null bar is ``max(majority, chance)`` with
    ``chance = sum_c q_c * p_c`` (q: the shuffled model's predicted class mix,
    p: the test prior), the accuracy of feature-independent guessing with the
    same output mix. Reported next to (never instead of) the harness gate.
    """
    import flybrain_learners as fl
    import flybrain_model_eval as fme

    plan = fme.plan_grouped_split(list(data.sample_ids), list(data.group_values), data.group_keys, eval_config)
    position = {sid: i for i, sid in enumerate(data.sample_ids)}
    rows = {k: np.asarray(sorted(position[s] for s in plan[k]), dtype=np.int64) for k in ("train", "test")}

    def inputs(r: np.ndarray) -> pd.DataFrame:
        frame = data.features.iloc[r].reset_index(drop=True).copy()
        if data.text is not None:
            frame[fl.TEXT_COLUMN] = [data.text[i] for i in r]
        return frame

    X_train, X_test = inputs(rows["train"]), inputs(rows["test"])
    y_train, y_test = data.labels[rows["train"]], data.labels[rows["test"]]
    labels, counts = np.unique(y_test, return_counts=True)
    prior = dict(zip(labels.tolist(), (counts / counts.sum()).tolist()))
    out: dict[str, Any] = {"n_permutations": int(n_permutations), "test_prior": prior, "models": {}}
    for label, entry in report.get("models", {}).items():
        params = dict(entry["tuning"]["best_params"])
        accs, chances = [], []
        for k in range(int(n_permutations)):
            rng = np.random.default_rng(int(eval_config.seed) + 104729 * (k + 1))
            shuffled = y_train.copy()
            rng.shuffle(shuffled)
            learner = fl.make_learner(entry["backend"], seed=int(eval_config.seed), **params).fit(X_train, shuffled)
            pred = np.asarray(learner.predict(X_test), dtype=object)
            accs.append(float(np.mean(pred == y_test)))
            vals, cnt = np.unique(pred, return_counts=True)
            chances.append(float(sum(c / len(pred) * prior.get(v, 0.0) for v, c in zip(vals.tolist(), cnt.tolist()))))
        gate = entry["gate"]
        majority = float(gate["majority_accuracy"])
        bar = max(majority, float(np.mean(chances)))
        out["models"][label] = {
            "accuracies": accs, "mean_accuracy": float(np.mean(accs)),
            "chance_same_output_mix": float(np.mean(chances)), "majority_accuracy": majority, "null_bar": bar,
            "collapsed_to_null": float(np.mean(accs)) <= bar + tolerance,
            "gate_without_harness_shuffle": bool(gate["trivial_baseline_gate"]["pass"]
                                                 and gate["beats_trivial_macro_f1"] and gate["paired_gain_significant"]),
        }
    return out


def run_target(target: str, config: MvRealModelConfig = MvRealModelConfig(), *, eval_config: Any = None,
               feature_filter: Sequence[str] | None = None, inputs: MvInputs | None = None,
               explicit_paths: Mapping[str, str | Path] | None = None, shuffle_null: bool = True,
               log=print) -> dict[str, Any]:
    import flybrain_model_eval as fme
    from threadpoolctl import threadpool_limits

    eval_config = eval_config or default_eval_config(target)
    build = build_target_dataset(target, config, eval_config, feature_filter=feature_filter, inputs=inputs,
                                 explicit_paths=explicit_paths, log=log)
    log(f"[mv] {target}: evaluating {build.info['n_samples']} samples x {build.info['n_features']} features")
    report = fme.run_evaluation(build.data, eval_config)
    report["mv_build"] = build.info
    if shuffle_null:
        with threadpool_limits(limits=int(eval_config.n_threads)):
            report["mv_shuffle_null"] = shuffle_null_control(build.data, report, eval_config)
        log(f"[mv] {target}: shuffle null " + json.dumps(
            {m: round(v["mean_accuracy"], 3) for m, v in report["mv_shuffle_null"]["models"].items()}))
    if eval_config.report_root:
        fme.write_report(report, eval_config.report_root, run_label=eval_config.run_label)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train + evaluate mv (MANC v1.0) real-model targets.")
    parser.add_argument("--target", action="append", choices=MV_REAL_TARGETS, help="repeatable; default all")
    parser.add_argument("--storage-root", default=os.environ.get("LOCI_FLYBRAIN_STORAGE_ROOT", DEFAULT_STORAGE_ROOT))
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--report-root", default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    parser.add_argument("--families", default=None, help="comma-separated feature families to keep")
    parser.add_argument("--quick", action="store_true", help="3-fold CV, 200 bootstrap draws")
    parser.add_argument("--summary", default=None, help="append summary rows (JSON lines) here")
    args = parser.parse_args(argv)
    config = MvRealModelConfig(storage_root=args.storage_root, cache_root=args.cache_root)
    inputs = open_inputs(config)
    families = None if not args.families else [f.strip() for f in args.families.split(",") if f.strip()]
    for target in args.target or MV_REAL_TARGETS:
        eval_config = default_eval_config(target, report_root=args.report_root, run_label=args.run_label,
                                          quick=args.quick)
        report = run_target(target, config, eval_config=eval_config, feature_filter=families, inputs=inputs)
        for row in report["summary"]:
            line = json.dumps({**row, "run_label": args.run_label}, sort_keys=True, default=str)
            print(line, flush=True)
            if args.summary:
                with open(args.summary, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
    return 0


__all__ = [
    "CATEGORICAL_FEATURES",
    "EXTRA_EXCLUDED",
    "GROUP_KEYS",
    "MV_REAL_TARGETS",
    "MvInputs",
    "MvRealModelConfig",
    "MvTargetBuild",
    "NEVER_FEATURES",
    "WIRING_SETS",
    "allowed_columns",
    "binned_text",
    "build_target_dataset",
    "default_eval_config",
    "default_models",
    "derive_neuropil_edges",
    "held_out_mask_ids",
    "label_frame",
    "load_node_table",
    "open_inputs",
    "run_target",
    "verify_manifest_listed_file",
]


if __name__ == "__main__":
    raise SystemExit(main())
