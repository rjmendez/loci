"""BANC v888 "real model" targets: structured wiring features, honest labels, grouped evaluation.

Targets (all evaluated through ``flybrain_target_registry.run_gated_evaluation``,
i.e. ``flybrain_model_eval.run_evaluation`` behind the R1 provenance gate):

* ``super_class``: curated BANC ``super_class`` harmonized onto the shared
  cross-dataset vocabulary (``flybrain_wiring_features.harmonize_super_class``).
  curated_morphology, provenance uncertain (the BANC rule is undocumented, E1).
* ``cell_class``: curated BANC ``cell_class`` (classes with enough neurons and
  enough distinct cell types to be split by group). connectivity_defined
  (uncertain): BANC types/classes were transferred through NBLAST +
  connectivity co-clustering matches, so this is never gated.
* ``flow``: curated ``afferent`` / ``intrinsic`` / ``efferent``;
  curated_morphology, provenance uncertain (E1).
* ``connectivity_tier`` and ``neurotransmitter_dominance``: the existing sample
  builder (``flybrain_brain_cluster_banc_samples``) defines the samples and
  labels unchanged; only the model inputs are new. NT labels are the BANC
  **v2 classifier predictions** (argmax of per-neuron scores), not annotated
  ground truth: that lane is a *distillation of synister_banc*
  (model_predicted) and connectivity_tier a connectivity-derived statistic;
  neither is ever gated.
* ``nt_literature`` (+ ``_binary``, ``_all``, ``_all_binary``): the R2
  literature ground truth (``flybrain_nt_ground_truth``, confidence >= 4)
  mapped onto proofread BANC neurons by ``cell_type`` (measured, gated).
  ``nt_literature`` removes the synister_banc training types (every type in
  the BANC GT tables minus ``banc_cell_types_removed_for_nt``);
  ``nt_literature_all`` keeps them. At most ``NT_MAX_PER_TYPE`` neurons per type.

Model inputs (never the curated annotation that is the target):

* ``base`` wiring features: one build over all BANC meta neurons with a
  constant partner category, so it does not depend on any label and is cached
  once: ``degree__*``, ``recip__*`` from ``edgelist_simple_v3`` (weight =
  synapse ``count``) and ``out_np__*`` / ``in_np__*`` (top-30 neuropil
  fractions, entropy) from a streamed pass over the 198.8M-row enriched
  synapse table (``neuropil`` column, one synapse = weight 1).
* ``comp`` wiring features: partner composition by harmonized super_class
  (``out_comp`` / ``in_comp``) and its 2-hop version (``out2_comp`` /
  ``in2_comp``). For super_class / cell_class / flow (the partner category is
  the target or determines it) the category of **every node outside the train
  split** is masked, including neurons that are not in the evaluation sample
  (a stricter mask than val+test only: an unsampled neuron of a held-out
  cell type would otherwise leak its label through homophily).
* ``morph__*`` from ``banc_888_metrics.feather`` (cable length, volume,
  branch/end points, axon/dendrite split, segregation index, ...). For
  ``connectivity_tier`` the extensive size measures are dropped as size
  proxies (``LOCAL_EXCLUSIONS``) on top of the registered exclusions.
* ``annot__*`` (connectivity_tier / NT only): curated super_class, cell_class,
  flow and side as categoricals, as the legacy text model used.

Every feature table passes ``flybrain_wiring_features`` exclusions for the
target (``build_wiring_features`` drops them; ``run_evaluation`` re-checks and
fails closed). Snapshot reads are read-only: the manifest is validated with
``verify_hashes=False`` (structure + sizes, no stamp writes) and each file
actually read is then sha256-verified against the manifest with a hash-stamp
cache OUTSIDE the snapshot tree (``DEFAULT_STAMP_DIR``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import flybrain_model_eval as fme
import flybrain_nt_ground_truth as ntgt
import flybrain_target_registry as ftr
import flybrain_wiring_features as fwf
from flybrain_banc_adapter import (
    BANC_PRODUCT_PATHS,
    BANC_VERSION_ID,
    ROLE_EDGELIST_V3,
    ROLE_META,
    ROLE_NT_PREDICTION,
    BancAdapterError,
    BancAdapterErrorCode,
    load_banc_meta,
    manifest_digest,
    open_banc_snapshot,
)
from flybrain_hash_stamps import METHOD_SIZE_ONLY, HashStampCache, verify_file_sha256

DATASET = "banc"
DEFAULT_STORAGE_ROOT = "/mnt/f/.flybrain"
DEFAULT_CACHE_ROOT = "/mnt/f/.flybrain/cache"
DEFAULT_STAMP_DIR = "/mnt/f/.flybrain/cache/hash-stamps/BANC-banc_888"

REL_EDGELIST = BANC_PRODUCT_PATHS[ROLE_EDGELIST_V3]
REL_META = BANC_PRODUCT_PATHS[ROLE_META]
REL_NT = BANC_PRODUCT_PATHS[ROLE_NT_PREDICTION]
REL_METRICS = "metadata/banc_888_metrics.feather"
REL_ENRICHED = "source/compiled_data/banc_888_synapses_v3_enriched.parquet"

TARGET_SUPER_CLASS = "super_class"
TARGET_CELL_CLASS = "cell_class"
TARGET_FLOW = "flow"
TARGET_CONNECTIVITY = "connectivity_tier"
TARGET_NT = "neurotransmitter_dominance"
ANNOTATION_TARGETS: tuple[str, ...] = (TARGET_SUPER_CLASS, TARGET_CELL_CLASS, TARGET_FLOW)
LEGACY_TARGETS: tuple[str, ...] = (TARGET_CONNECTIVITY, TARGET_NT)
NT_LITERATURE_TARGETS: tuple[str, ...] = ntgt.NT_LITERATURE_TARGETS
BANC_TARGETS: tuple[str, ...] = ANNOTATION_TARGETS + LEGACY_TARGETS + NT_LITERATURE_TARGETS
NT_MAX_PER_TYPE = 20
_P = ftr.LabelProvenance
# R1 label provenance per target (must equal flybrain_target_registry; checked at import).
TARGET_LABEL_PROVENANCE: Mapping[str, str] = {
    TARGET_SUPER_CLASS: _P.CURATED_MORPHOLOGY.value,
    TARGET_CELL_CLASS: _P.CONNECTIVITY_DEFINED.value,
    TARGET_FLOW: _P.CURATED_MORPHOLOGY.value,
    TARGET_CONNECTIVITY: _P.CONNECTIVITY_DEFINED.value,
    TARGET_NT: _P.MODEL_PREDICTED.value,
    **{t: _P.MEASURED.value for t in NT_LITERATURE_TARGETS},
}
ftr.check_module_provenance(DATASET, TARGET_LABEL_PROVENANCE)
# Partner category (harmonized super_class) is the target or determines it -> mask non-train nodes.
MASKED_TARGETS: frozenset[str] = frozenset(ANNOTATION_TARGETS)
GROUP_KEYS: tuple[str, ...] = ("cell_type", "hemilineage")
FLOW_LABELS: tuple[str, ...] = ("afferent", "efferent", "intrinsic")
DROP_SUPER_CLASSES: frozenset[str] = frozenset({fwf.UNKNOWN, "non_neuronal", "other"})

# metrics column -> (feature name, transform, extensive?)
MORPH_COLUMNS: Mapping[str, tuple[str, str, bool]] = {
    "l2_cable_length_um": ("morph__log_cable_length_um", "log1p", True),
    "volume_nm3": ("morph__log_volume_nm3", "log1p", True),
    "l2_nodes": ("morph__log_l2_nodes", "log1p", True),
    "branchpoints": ("morph__log_branchpoints", "log1p", True),
    "endpoints": ("morph__log_endpoints", "log1p", True),
    "axon_length": ("morph__log_axon_length", "log1p", True),
    "dend_length": ("morph__log_dend_length", "log1p", True),
    "mitochondria": ("morph__log_mitochondria", "log1p", True),
    "pd_width": ("morph__pd_width", "identity", False),
    "segregation_index": ("morph__segregation_index", "identity", False),
    "projection_score": ("morph__projection_score", "identity", False),
}
MORPH_DERIVED: tuple[str, ...] = ("morph__axon_fraction", "morph__branchpoints_per_um", "morph__mito_per_um")

# BANC-local exclusions applied on top of the registered objective exclusions.
LOCAL_EXCLUSIONS: Mapping[str, tuple[str, ...]] = {
    # Extensive size measures track total synapse count closely (bigger arbor, more synapses).
    TARGET_CONNECTIVITY: tuple(name for name, _, extensive in MORPH_COLUMNS.values() if extensive)
    + ("morph__branchpoints_per_um", "morph__mito_per_um"),
}

ANNOT_COLUMNS: tuple[str, ...] = ("super_class", "cell_class", "flow", "side")

ABLATION_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "degree": ("degree__*",),
    "recip": ("recip__*",),
    "neuropil": ("out_np__*", "in_np__*"),
    "partner_comp": ("out_comp__*", "in_comp__*"),
    "partner_comp_2hop": ("out2_comp__*", "in2_comp__*"),
    "morph": ("morph__*",),
    "annot": ("annot__*",),
}


# =========================================================================== inputs


@dataclass(frozen=True)
class BancInputs:
    snapshot_root: Path
    paths: Mapping[str, Path]
    manifest_id: str
    manifest_sha256: str
    product_sha256: Mapping[str, str]
    hash_methods: Mapping[str, str]

    def path(self, rel: str) -> Path:
        return self.paths[rel]

    def provenance(self) -> dict[str, Any]:
        return {"manifest_id": self.manifest_id, "manifest_sha256": self.manifest_sha256,
                "product_sha256": dict(sorted(self.product_sha256.items())),
                "hash_methods": dict(sorted(self.hash_methods.items()))}


def _reject_snapshot_dir(path: Path, what: str) -> None:
    if "snapshots" in Path(path).resolve(strict=False).parts:
        raise ValueError(f"refusing to use a {what} under a snapshots directory: {path}")


def open_banc_inputs(rel_paths: Sequence[str], *, storage_root: str | Path | None = DEFAULT_STORAGE_ROOT,
                     stamp_dir: str | Path = DEFAULT_STAMP_DIR, verify_hashes: bool | None = None) -> BancInputs:
    """Validate the pinned BANC manifest and sha256-verify ``rel_paths``; never writes into the snapshot.

    ``verify_hashes``: None = hash unless a stamp in ``stamp_dir`` (outside the
    snapshot) matches the file's (size, mtime_ns); True = always hash;
    False = size checks only (tests / smoke runs).
    """
    _reject_snapshot_dir(Path(stamp_dir), "hash-stamp directory")
    # verify_hashes=False: manifest self-hash, sidecar, pin, licence, path-safety and every
    # file's size are checked; no content hash, so the in-snapshot stamp cache is never written.
    snapshot = open_banc_snapshot(storage_root, required_roles=(ROLE_EDGELIST_V3, ROLE_META), verify_hashes=False)
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    if manifest_digest(manifest) != snapshot.manifest_sha256:
        raise BancAdapterError(BancAdapterErrorCode.INTEGRITY_MISMATCH, "manifest changed after validation")
    listed = {str(e["relative_path"]).replace("\\", "/"): str(e["sha256"]) for e in manifest["integrity"]["files"]}
    cache = HashStampCache(stamp_dir)
    paths: dict[str, Path] = {}
    shas: dict[str, str] = {}
    methods: dict[str, str] = {}
    for rel in dict.fromkeys(rel_paths):
        if rel not in listed:
            raise BancAdapterError(BancAdapterErrorCode.PRODUCT_MISSING, "file is not listed in the BANC manifest",
                                   {"relative_path": rel})
        path = (snapshot.snapshot_root / rel).resolve(strict=False)
        if not path.is_relative_to(snapshot.snapshot_root.resolve(strict=False)):
            raise BancAdapterError(BancAdapterErrorCode.PATH_ESCAPE, "path escapes the snapshot", {"relative_path": rel})
        if verify_hashes is False:
            methods[rel] = METHOD_SIZE_ONLY
        else:
            check = verify_file_sha256(path, relative_path=rel, expected_sha256=listed[rel], cache=cache,
                                       force=verify_hashes is True)
            if not check.matched:
                raise BancAdapterError(BancAdapterErrorCode.INTEGRITY_MISMATCH, "file sha256 mismatch",
                                       {"relative_path": rel, "expected": listed[rel], "actual": check.actual_sha256})
            methods[rel] = check.method
        paths[rel] = path
        shas[rel] = listed[rel]
    return BancInputs(snapshot.snapshot_root, paths, snapshot.manifest_id, snapshot.manifest_sha256, shas, methods)


# =========================================================================== node table


def _clean(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>", "unknown"}:
        return None
    return text


def _proofread(values: Sequence[Any]) -> np.ndarray:
    from flybrain_brain_cluster_banc_samples import _proofread_mask

    return np.asarray(_proofread_mask(list(values)), dtype=bool)


def load_node_table(meta_path: Path, metrics_path: Path | None = None) -> pd.DataFrame:
    """All BANC meta neurons: ids, curated annotations, proofread flag, group keys, morphology features."""
    meta = load_banc_meta(meta_path, columns=("banc_888_id", "proofread", "side", "super_class", "cell_class",
                                              "flow", "hemilineage"), optional_columns=("cell_type",))
    if "cell_type" not in meta.columns:
        meta["cell_type"] = None
    meta["proofread_bool"] = _proofread(meta["proofread"].tolist())
    for column in ("cell_type", "hemilineage"):
        meta[column] = [_clean(v) for v in meta[column].tolist()]
    meta["super_class_h"] = [fwf.harmonize_super_class(v) for v in meta["super_class"].tolist()]
    if metrics_path is not None:
        meta = meta.merge(morphology_features(metrics_path), on="banc_888_id", how="left")
    return meta.reset_index(drop=True)


def morphology_features(metrics_path: Path) -> pd.DataFrame:
    import pyarrow.feather as feather

    wanted = ["banc_888_id", *MORPH_COLUMNS]
    table = feather.read_table(str(metrics_path), columns=wanted)
    raw = table.to_pandas()
    raw["banc_888_id"] = raw["banc_888_id"].astype(str)
    if raw["banc_888_id"].duplicated().any():
        raise BancAdapterError(BancAdapterErrorCode.SCHEMA_MISMATCH, "metrics banc_888_id is not unique")
    out = pd.DataFrame({"banc_888_id": raw["banc_888_id"]})
    for column, (name, transform, _) in MORPH_COLUMNS.items():
        values = pd.to_numeric(raw[column], errors="coerce").astype(float)
        values = values.where(np.isfinite(values))
        out[name] = np.log1p(values.clip(lower=0)) if transform == "log1p" else values
    axon = pd.to_numeric(raw["axon_length"], errors="coerce").astype(float)
    dend = pd.to_numeric(raw["dend_length"], errors="coerce").astype(float)
    cable = pd.to_numeric(raw["l2_cable_length_um"], errors="coerce").astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["morph__axon_fraction"] = np.where((axon + dend) > 0, axon / (axon + dend), np.nan)
        out["morph__branchpoints_per_um"] = np.where(cable > 0, raw["branchpoints"].astype(float) / cable, np.nan)
        out["morph__mito_per_um"] = np.where(cable > 0, pd.to_numeric(raw["mitochondria"], errors="coerce") / cable,
                                             np.nan)
    return out


# =========================================================================== labels


def _stable_rank(values: Sequence[str], salt: str) -> list[str]:
    return [hashlib.sha256(f"{salt}:{v}".encode("utf-8")).hexdigest() for v in values]


def annotation_labels(nodes: pd.DataFrame, target: str) -> pd.Series:
    """Per-node label for an annotation target (None = not eligible). Proofread neurons only."""
    if target == TARGET_SUPER_CLASS:
        labels = [None if v in DROP_SUPER_CLASSES else v for v in nodes["super_class_h"].tolist()]
    elif target == TARGET_FLOW:
        labels = [fwf.slug(v) if _clean(v) is not None and fwf.slug(v) in FLOW_LABELS else None
                  for v in nodes["flow"].tolist()]
    elif target == TARGET_CELL_CLASS:
        labels = [fwf.slug(v) if _clean(v) is not None else None for v in nodes["cell_class"].tolist()]
    else:
        raise ValueError(f"not an annotation target: {target!r}")
    out = pd.Series(labels, index=nodes.index, dtype=object)
    out[~nodes["proofread_bool"].to_numpy()] = None
    return out


def nt_literature_labels(nodes: pd.DataFrame, target: str, *, nt_root: str | Path = ntgt.DEFAULT_NT_GT_ROOT
                         ) -> tuple[pd.Series, dict[str, Any]]:
    """Per-node literature NT (R2) for proofread neurons (None elsewhere) + coverage statistics."""
    source = ntgt.open_nt_ground_truth(nt_root)
    typed = pd.DataFrame({"banc_888_id": nodes["banc_888_id"].astype(str), "cell_type": nodes["cell_type"]})
    labelled = ntgt.label_neurons(typed, dataset=DATASET, source=source, target=target, id_column="banc_888_id")
    mapping = dict(zip(labelled.frame["banc_888_id"].astype(str), labelled.frame["label"].astype(str)))
    out = pd.Series([mapping.get(i) for i in typed["banc_888_id"]], index=nodes.index, dtype=object)
    out[~nodes["proofread_bool"].to_numpy()] = None
    coverage = dict(labelled.coverage)
    coverage["proofread_neurons_labelled"] = int(out.notna().sum())
    return out, coverage


def cap_per_type(frame: pd.DataFrame, *, max_per_type: int, salt: str) -> pd.DataFrame:
    """At most ``max_per_type`` rows per cell type, in sha256(salt:id) order (type-level labels)."""
    ranked = frame.assign(_r=_stable_rank(frame["banc_888_id"].astype(str).tolist(), salt))
    ranked = ranked.sort_values(["cell_type", "_r"], kind="mergesort")
    capped = ranked.groupby("cell_type", sort=False, dropna=False).head(int(max_per_type)).drop(columns=["_r"])
    return capped.sort_values("banc_888_id", kind="mergesort").reset_index(drop=True)


@dataclass(frozen=True)
class SelectionConfig:
    min_class_count: int = 200
    min_class_cell_types: int = 5
    max_per_class: int = 5000
    salt: str = "flybrain-banc-real-models-v1"


def select_annotation_samples(nodes: pd.DataFrame, labels: pd.Series, has_wiring: np.ndarray,
                              config: SelectionConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Eligible (labeled, proofread, wired) nodes; small / single-type classes dropped; per-class hash cap."""
    frame = nodes.assign(label=labels)
    eligible = frame["label"].notna().to_numpy() & np.asarray(has_wiring, dtype=bool)
    frame = frame[eligible]
    counts = frame["label"].value_counts()
    type_counts = frame.dropna(subset=["cell_type"]).groupby("label")["cell_type"].nunique()
    kept_classes = sorted(c for c, n in counts.items()
                          if int(n) >= config.min_class_count
                          and int(type_counts.get(c, 0)) >= config.min_class_cell_types)
    dropped = {str(c): int(n) for c, n in counts.items() if c not in set(kept_classes)}
    frame = frame[frame["label"].isin(kept_classes)].copy()
    frame["_rank"] = _stable_rank(frame["banc_888_id"].tolist(), config.salt)
    frame = (frame.sort_values(["label", "_rank"], kind="mergesort").groupby("label", sort=True)
             .head(int(config.max_per_class)).drop(columns=["_rank"]))
    frame = frame.sort_values("banc_888_id", kind="mergesort").reset_index(drop=True)
    info = {"eligible_before_class_filter": int(eligible.sum()),
            "class_counts_before_cap": {str(k): int(v) for k, v in counts.sort_index().items()},
            "dropped_classes": dict(sorted(dropped.items())),
            "kept_classes": kept_classes,
            "class_counts_after_cap": {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()},
            "selection": asdict(config)}
    return frame, info


# =========================================================================== wiring features


def edge_sources(inputs: BancInputs, *, neuropil: bool = True) -> tuple[fwf.EdgeSource, fwf.EdgeSource | None]:
    prov = {"manifest_sha256": inputs.manifest_sha256}
    edges = fwf.EdgeSource(path=str(inputs.path(REL_EDGELIST)), pre="pre", post="post", weight="count",
                           unique_pairs=True, format="ipc",
                           provenance={**prov, "sha256": inputs.product_sha256[REL_EDGELIST]})
    np_edges = None
    if neuropil:
        np_edges = fwf.EdgeSource(path=str(inputs.path(REL_ENRICHED)), pre="pre_root_id", post="post_root_id",
                                  weight=None, neuropil="neuropil", unique_pairs=False, format="parquet",
                                  provenance={**prov, "sha256": inputs.product_sha256[REL_ENRICHED]})
    return edges, np_edges


BASE_FAMILIES = ("degree__", "recip__", "out_np__", "in_np__")
COMP_FAMILIES = ("out_comp__", "in_comp__", "out2_comp__", "in2_comp__")


def base_wiring_features(inputs: BancInputs, nodes: pd.DataFrame, *, objective: str, neuropil: bool = True,
                         cache_root: str | Path | None = DEFAULT_CACHE_ROOT,
                         top_k_neuropils: int = 30) -> fwf.WiringFeatureResult:
    """Label-independent features (constant partner category) -> degree, recip, neuropil families."""
    edges, np_edges = edge_sources(inputs, neuropil=neuropil)
    table = pd.DataFrame({"banc_888_id": nodes["banc_888_id"].astype(str), "_const": "neuron"})
    result = fwf.build_wiring_features(
        dataset=DATASET, objective=objective, edges=edges, nodes=table, id_column="banc_888_id",
        category_column="_const", params=fwf.WiringFeatureParams(top_k_neuropils=top_k_neuropils, reciprocity=True,
                                                                  two_hop=False),
        neuropil_edges=np_edges, cache_root=cache_root)
    return _keep_families(result, BASE_FAMILIES)


def composition_features(inputs: BancInputs, nodes: pd.DataFrame, *, objective: str,
                         mask_ids: Sequence[str] | None, cache_root: str | Path | None = DEFAULT_CACHE_ROOT,
                         ) -> fwf.WiringFeatureResult:
    """Partner composition (1- and 2-hop) by harmonized super_class, with optional category masking."""
    edges, _ = edge_sources(inputs, neuropil=False)
    table = pd.DataFrame({"banc_888_id": nodes["banc_888_id"].astype(str), "super_class": nodes["super_class"]})
    result = fwf.build_wiring_features(
        dataset=DATASET, objective=objective, edges=edges, nodes=table, id_column="banc_888_id",
        category_column="super_class", vocab_map=fwf.harmonize_super_class,
        params=fwf.WiringFeatureParams(reciprocity=False, two_hop=True), cache_root=cache_root,
        mask_category_ids=mask_ids)
    return _keep_families(result, COMP_FAMILIES)


def _keep_families(result: fwf.WiringFeatureResult, prefixes: Sequence[str]) -> fwf.WiringFeatureResult:
    cols = [fwf.NODE_ID_COLUMN] + [c for c in result.frame.columns if c.startswith(tuple(prefixes))]
    return fwf.WiringFeatureResult(frame=result.frame[cols], fingerprint=result.fingerprint,
                                   cache_path=result.cache_path, objective=result.objective,
                                   excluded_features=result.excluded_features, meta=result.meta)


def local_exclusions(columns: Sequence[str], target: str) -> list[str]:
    patterns = LOCAL_EXCLUSIONS.get(target, ())
    return sorted(c for c in columns if c in set(patterns))


# =========================================================================== text view (nb)


def binned_text(frame: pd.DataFrame, train_rows: np.ndarray, *, n_bins: int = 5) -> list[str]:
    """``name name_qK`` tokens per feature; quantile edges fitted on train rows only."""
    parts: list[list[str]] = []
    for name in sorted(frame.columns):
        col = frame[name]
        if pd.api.types.is_numeric_dtype(col.dtype):
            x = col.to_numpy(dtype=np.float64)
            train = x[train_rows]
            train = train[np.isfinite(train)]
            if len(train) == 0:
                continue
            edges = np.unique(np.quantile(train, np.linspace(0, 1, n_bins + 1)[1:-1]))
            bins = np.searchsorted(edges, x, side="right")
            values = [f"{name}_qna" if not math.isfinite(v) else f"{name}_q{b}" for v, b in zip(x, bins)]
        else:
            values = [f"{name}_{fwf.slug(v) if pd.notna(v) else 'na'}" for v in col.tolist()]
        parts.append([f"{name} {v}" for v in values])
    return [" ".join(row) for row in zip(*parts)] if parts else [""] * len(frame)


# =========================================================================== datasets


@dataclass(frozen=True)
class BancTargetConfig:
    storage_root: str | None = DEFAULT_STORAGE_ROOT
    cache_root: str | None = DEFAULT_CACHE_ROOT
    stamp_dir: str = DEFAULT_STAMP_DIR
    verify_hashes: bool | None = None
    neuropil: bool = True
    morphology: bool = True
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    legacy_max_samples: int = 30000
    drop_families: tuple[str, ...] = ()  # e.g. ("neuropil",) for a no-neuropil variant
    # nt_literature*: smaller classes are kept (labels are per type), capped per type afterwards
    nt_selection: SelectionConfig = field(default_factory=lambda: SelectionConfig(
        min_class_count=40, min_class_cell_types=3, max_per_class=100_000, salt="flybrain-banc-nt-literature-v1"))
    nt_root: str = ntgt.DEFAULT_NT_GT_ROOT


def _required_files(config: BancTargetConfig, target: str) -> list[str]:
    files = [REL_EDGELIST, REL_META]
    if target == TARGET_NT:
        files.append(REL_NT)
    if config.morphology:
        files.append(REL_METRICS)
    if config.neuropil:
        files.append(REL_ENRICHED)
    return files


def _drop_families(columns: Sequence[str], families: Sequence[str]) -> list[str]:
    import fnmatch

    patterns = [p for fam in families for p in ABLATION_FAMILIES.get(fam, (fam,))]
    return [c for c in columns if not any(fnmatch.fnmatchcase(c, p) for p in patterns)]


def build_eval_dataset(target: str, config: BancTargetConfig, eval_config: fme.EvalConfig,
                       *, inputs: BancInputs | None = None, nodes: pd.DataFrame | None = None,
                       log=print) -> fme.EvalDataset:
    if target not in BANC_TARGETS:
        raise ValueError(f"target must be one of {BANC_TARGETS}")
    t0 = time.time()
    if inputs is None:
        inputs = open_banc_inputs(_required_files(config, target), storage_root=config.storage_root,
                                  stamp_dir=config.stamp_dir, verify_hashes=config.verify_hashes)
    if nodes is None:
        nodes = load_node_table(inputs.path(REL_META), inputs.path(REL_METRICS) if config.morphology else None)
    log(f"[banc:{target}] inputs verified {inputs.hash_methods}; nodes={len(nodes)} ({time.time() - t0:.0f}s)")
    base = base_wiring_features(inputs, nodes, objective=target, neuropil=config.neuropil,
                                cache_root=config.cache_root)
    log(f"[banc:{target}] base features {base.frame.shape} cache={base.cache_path} ({time.time() - t0:.0f}s)")
    notes: dict[str, Any] = {"dataset_version": BANC_VERSION_ID, "provenance": inputs.provenance(),
                             "base_features_fingerprint": base.fingerprint,
                             **ftr.provenance_notes(DATASET, target)}

    if target in ANNOTATION_TARGETS or target in NT_LITERATURE_TARGETS:
        base_idx = base.frame.set_index(fwf.NODE_ID_COLUMN)
        wired = base_idx.reindex(nodes["banc_888_id"].astype(str))
        has_wiring = (wired.filter(like="recip__").notna().any(axis=1)
                      | wired.filter(like="out_np__").notna().any(axis=1)).to_numpy()
        if "degree__out_weight_total" in wired.columns:
            has_wiring = has_wiring | ((wired["degree__out_weight_total"].fillna(0)
                                        + wired["degree__in_weight_total"].fillna(0)) > 0).to_numpy()
    if target in NT_LITERATURE_TARGETS:
        labels, coverage = nt_literature_labels(nodes, target, nt_root=config.nt_root)
        samples, selection = select_annotation_samples(nodes, labels, has_wiring, config.nt_selection)
        samples = cap_per_type(samples, max_per_type=NT_MAX_PER_TYPE, salt=config.nt_selection.salt)
        selection["class_counts_after_type_cap"] = {str(k): int(v) for k, v in
                                                    samples["label"].value_counts().sort_index().items()}
        ids = samples["banc_888_id"].astype(str).tolist()
        group_values = samples[list(GROUP_KEYS)].to_dict(orient="records")
        label_col = "label"
        notes.update({"selection": selection, "nt_literature_coverage": coverage,
                      "label_source": ntgt.NT_LITERATURE_SPECS[target].describe()})
    elif target in ANNOTATION_TARGETS:
        labels = annotation_labels(nodes, target)
        samples, selection = select_annotation_samples(nodes, labels, has_wiring, config.selection)
        ids = samples["banc_888_id"].astype(str).tolist()
        group_values = samples[list(GROUP_KEYS)].to_dict(orient="records")
        label_col = "label"
        notes.update({"selection": selection,
                      "label_source": f"BANC meta curated {target}" + (
                          " harmonized via flybrain_wiring_features.harmonize_super_class"
                          if target == TARGET_SUPER_CLASS else "")})
    else:
        samples, legacy_meta = legacy_samples_frame(target, config)
        ids = samples["banc_888_id"].tolist()
        group_values = samples[list(GROUP_KEYS)].to_dict(orient="records")
        label_col = "label"
        notes.update({"legacy_builder": legacy_meta,
                      "label_source": ("per-neuron argmax of BANC neurotransmitter_prediction_v2 scores "
                                       "(classifier predictions, NOT annotated ground truth)"
                                       if target == TARGET_NT else
                                       "legacy connectivity_tier: total outgoing v3 synapse count >= global q75")})

    plan = fme.plan_grouped_split(ids, group_values, GROUP_KEYS, eval_config)
    mask_ids: list[str] | None = None
    if target in MASKED_TARGETS:
        train = set(plan["train"])
        mask_ids = [i for i in nodes["banc_888_id"].astype(str).tolist() if i not in train]
        notes["masked_split_ids_sha256"] = fme.split_ids_sha256(plan)
        notes["category_mask"] = {"policy": "every node outside the train split (incl. unsampled neurons)",
                                  "masked_nodes": len(mask_ids)}
    comp = composition_features(inputs, nodes, objective=target, mask_ids=mask_ids, cache_root=config.cache_root)
    log(f"[banc:{target}] comp features {comp.frame.shape} ({time.time() - t0:.0f}s)")
    notes["comp_features_fingerprint"] = comp.fingerprint

    frame = samples[["banc_888_id", label_col, *GROUP_KEYS]].copy()
    frame["banc_888_id"] = frame["banc_888_id"].astype(str)
    frame = frame.merge(base.frame, left_on="banc_888_id", right_on=fwf.NODE_ID_COLUMN, how="left").drop(
        columns=[fwf.NODE_ID_COLUMN])
    frame = frame.merge(comp.frame, left_on="banc_888_id", right_on=fwf.NODE_ID_COLUMN, how="left").drop(
        columns=[fwf.NODE_ID_COLUMN])
    if config.morphology:
        morph_cols = [c for c in nodes.columns if c.startswith("morph__")]
        frame = frame.merge(nodes[["banc_888_id", *morph_cols]].assign(banc_888_id=lambda d: d["banc_888_id"].astype(str)),
                            on="banc_888_id", how="left")
    if target in LEGACY_TARGETS:
        annot = nodes[["banc_888_id", *ANNOT_COLUMNS]].copy()
        annot["banc_888_id"] = annot["banc_888_id"].astype(str)
        for column in ANNOT_COLUMNS:
            annot[f"annot__{column}"] = [fwf.slug(v) if _clean(v) is not None else fwf.UNKNOWN
                                         for v in annot.pop(column).tolist()]
        frame = frame.merge(annot, on="banc_888_id", how="left")
    if len(frame) != len(samples):
        raise ValueError("feature join changed the sample count")

    reserved = {"banc_888_id", label_col, *GROUP_KEYS}
    feature_cols = [c for c in frame.columns if c not in reserved]
    dropped_local = local_exclusions(feature_cols, target)
    feature_cols = [c for c in feature_cols if c not in set(dropped_local)]
    feature_cols = _drop_families(feature_cols, config.drop_families)
    fwf.assert_features_allowed(feature_cols, target)
    feature_cols = sorted(feature_cols)
    notes.update({"local_exclusions": dropped_local, "dropped_families": list(config.drop_families),
                  "registered_exclusions_dropped": sorted(set(base.excluded_features) | set(comp.excluded_features)),
                  "n_features": len(feature_cols)})

    position = {sid: i for i, sid in enumerate(ids)}
    train_rows = np.asarray(sorted(position[s] for s in plan["train"]), dtype=np.int64)
    frame["__text__"] = binned_text(frame[feature_cols], train_rows)
    data = fme.EvalDataset.from_frame(frame, dataset=DATASET, target=target, id_column="banc_888_id",
                                      label_column=label_col, feature_columns=feature_cols,
                                      group_columns=list(GROUP_KEYS), text_column="__text__", notes=notes)
    log(f"[banc:{target}] dataset n={len(ids)} features={len(feature_cols)} ({time.time() - t0:.0f}s)")
    return data


def legacy_samples_frame(target: str, config: BancTargetConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Samples + labels from the existing BANC builder (unchanged semantics), as a frame."""
    from flybrain_brain_cluster_banc_samples import BancSampleBuildConfig, build_banc_training_samples

    payload = build_banc_training_samples(target, BancSampleBuildConfig(
        objective=target, storage_root=config.storage_root, verify_hashes=False,
        max_samples=int(config.legacy_max_samples)))
    rows = []
    for sample in payload["samples"]:
        meta = sample["metadata"]
        rows.append({"banc_888_id": str(meta["root_id"]), "label": sample["expected_label"],
                     "cell_type": _clean(meta.get("cell_type")), "hemilineage": _clean(meta.get("hemilineage"))})
    frame = pd.DataFrame(rows)
    keep = {k: payload["metadata"].get(k) for k in ("input_fingerprint", "label_counts", "selected_count",
                                                     "high_connectivity_threshold", "filter_stats")}
    return frame, keep


# =========================================================================== runner


def default_eval_config(run_label: str = "v1", report_root: str | None = fme.DEFAULT_REPORT_ROOT, *,
                        cv_folds: int = 4, n_bootstrap: int = 1000, n_threads: int = 8) -> fme.EvalConfig:
    return fme.EvalConfig(
        models=(
            fme.ModelSpec("nb", ({"alpha": 1.0},), "temperature"),
            fme.ModelSpec("logreg", ({"C": 0.1}, {"C": 1.0, "class_weight": "balanced"}), "temperature"),
            fme.ModelSpec("hgb", ({"max_iter": 300, "learning_rate": 0.1},
                                  {"max_iter": 400, "learning_rate": 0.1, "class_weight": "balanced"}), "isotonic"),
        ),
        cv_folds=cv_folds, n_bootstrap=n_bootstrap, n_threads=n_threads, report_root=report_root,
        run_label=run_label, ablation_families={k: list(v) for k, v in ABLATION_FAMILIES.items()},
    )


def run_target(target: str, config: BancTargetConfig, eval_config: fme.EvalConfig, *, log=print) -> dict[str, Any]:
    import dataclasses
    import fnmatch

    data = build_eval_dataset(target, config, eval_config, log=log)
    if eval_config.ablation_families is not None:  # only families that exist for this target
        cols = list(data.features.columns)
        present = {name: list(patterns) for name, patterns in eval_config.ablation_families.items()
                   if any(fnmatch.fnmatchcase(c, p) for c in cols for p in patterns)}
        eval_config = dataclasses.replace(eval_config, ablation_families=present)
    report = ftr.run_gated_evaluation(data, eval_config)
    for row in report["summary"]:
        log(json.dumps(row, sort_keys=True, default=str))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BANC v888 real-model evaluation (grouped, calibrated, gated).")
    parser.add_argument("--target", choices=BANC_TARGETS, required=True)
    parser.add_argument("--storage-root", default=DEFAULT_STORAGE_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--report-root", default=fme.DEFAULT_REPORT_ROOT)
    parser.add_argument("--run-label", default="v1")
    parser.add_argument("--no-neuropil", action="store_true")
    parser.add_argument("--no-morphology", action="store_true")
    parser.add_argument("--drop-family", action="append", default=[])
    parser.add_argument("--max-per-class", type=int, default=5000)
    parser.add_argument("--min-class-count", type=int, default=200)
    parser.add_argument("--legacy-max-samples", type=int, default=30000)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--models", default="nb,logreg,hgb")
    parser.add_argument("--no-ablation", action="store_true")
    args = parser.parse_args(argv)
    config = BancTargetConfig(storage_root=args.storage_root, cache_root=args.cache_root,
                              neuropil=not args.no_neuropil, morphology=not args.no_morphology,
                              selection=SelectionConfig(min_class_count=args.min_class_count,
                                                        max_per_class=args.max_per_class),
                              legacy_max_samples=args.legacy_max_samples, drop_families=tuple(args.drop_family))
    eval_config = default_eval_config(args.run_label, args.report_root, cv_folds=args.cv_folds,
                                      n_bootstrap=args.n_bootstrap)
    wanted = {m.strip() for m in args.models.split(",") if m.strip()}
    import dataclasses

    eval_config = dataclasses.replace(eval_config, models=tuple(m for m in eval_config.models if m.backend in wanted),
                                      ablation=not args.no_ablation)
    run_target(args.target, config, eval_config, log=lambda msg: print(msg, flush=True))
    return 0


__all__ = [
    "ABLATION_FAMILIES",
    "ANNOTATION_TARGETS",
    "BANC_TARGETS",
    "BancInputs",
    "BancTargetConfig",
    "GROUP_KEYS",
    "LEGACY_TARGETS",
    "LOCAL_EXCLUSIONS",
    "MASKED_TARGETS",
    "SelectionConfig",
    "annotation_labels",
    "base_wiring_features",
    "binned_text",
    "build_eval_dataset",
    "composition_features",
    "default_eval_config",
    "legacy_samples_frame",
    "load_node_table",
    "local_exclusions",
    "morphology_features",
    "open_banc_inputs",
    "run_target",
    "select_annotation_samples",
    "NT_LITERATURE_TARGETS",
    "NT_MAX_PER_TYPE",
    "TARGET_LABEL_PROVENANCE",
    "cap_per_type",
    "nt_literature_labels",
]


if __name__ == "__main__":
    raise SystemExit(main())
