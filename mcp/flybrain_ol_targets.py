"""Optic-lobe v1.1 "real model" targets: labels, structured features, eval datasets.

This module turns the local, manifest-verified optic-lobe snapshot into
``flybrain_model_eval.EvalDataset`` objects for five targets and runs the
shared harness (``run_evaluation``) on them. Nothing is written into the
snapshot: the snapshot is opened with ``verify_hashes=False`` (manifest
self-hash + sidecar + every listed file's size; no hash stamps), and the extra
products read here (the pair-level edge list and the flat-connectome body
stats) are sha256-checked against the manifest by reading them.

Targets (label source, all per neuron; samples are typed ``Traced`` neurons):

* ``cell_family``: the neuron's cell-type family from ``ol_cell_family``
  (documented prefix map + merges; families with < ``min_types`` labelled
  types pooled into ``other``; ``*_unclear`` / ``*TBD*`` / parenthesised
  provisional types are never labels).
* ``super_class``: the flat-connectome ``class`` column (optic /
  visual_projection / visual_centrifugal / central / descending_neuron ...),
  harmonized through ``flybrain_wiring_features.harmonize_super_class``;
  ``*_tbc`` calls dropped; classes with < ``min_types`` types dropped.
* ``connectivity_tier``: the legacy definition (neuPrint ``downstream`` >= the
  0.75 quantile over candidates with downstream >= 10).
* ``neurotransmitter_dominance``: the neuron's own synapse-classifier call
  ``predictedNt`` (PREDICTED; ``unclear`` dropped, ``totalNtPredictions`` >= 10).
  This lane is a *distillation of the optic-lobe synapse NT classifier*
  [Nern 2025] (model_predicted), never an NT accuracy, and never gated.
* ``nt_ground_truth``: the type-level ``consensusNt`` restricted to types
  whose ``ntReference`` is set (literature / experimental reference, e.g.
  Davis et al. 2020 RNA-seq; CONSENSUS, not per-neuron ground truth).
  measured with provenance uncertain (consensus mixes predictions with
  curation); superseded by ``nt_literature``.
* ``nt_literature`` (+ ``_binary``, ``_all``, ``_all_binary``): the R2
  literature ground truth (``flybrain_nt_ground_truth``, confidence >= 4)
  mapped by type name. ``nt_literature`` drops the classifier's training
  types (Nern et al. table, ``Part_of_training_data == yes``).

Label provenance (R1): ``cell_family`` is connectivity_defined (optic-lobe
types were finalised by connectivity [Matsliah 2024; Nern 2025]: recovery of
connectivity-derived annotations), ``super_class`` curated_morphology,
``connectivity_tier`` connectivity_defined. ``TARGET_LABEL_PROVENANCE`` is
checked against ``flybrain_target_registry`` at import and every evaluation
goes through ``run_gated_evaluation``.

Split: grouped by ``cell_type`` (the registry split key), so every test
neuron belongs to a cell type the model never saw. The per-type sample cap
(``max_per_type``, sha256(body id) order) keeps the 2,000-neuron columnar
types from dominating.

Features (``<family>__<name>``; the family is what the ablation drops):

* wiring (``flybrain_wiring_features`` over the 22M-row pair-level edge list):
  ``out_comp`` / ``in_comp`` (weighted partner-family composition),
  ``out2_comp`` / ``in2_comp`` (2-hop), ``recip``; ``degree`` is built but the
  connectivity exclusions drop it for that target. Partner categories are
  the partner's cell FAMILY. The categories of every node whose cell type is
  in the val or test split are MASKED (treated as unannotated) for every
  target, so no held-out type's identity reaches any node's features through
  same-type partners (``mask_category_ids``; the split plan's sha256 is
  pinned in the dataset notes and re-checked by ``run_evaluation``).
* anatomy (the neuron's own ``roiInfo``): ``roi`` (per-ROI input/output
  fractions, optic-lobe share, entropy, output/input ratio), ``layer``
  (per-layer synweight fractions), ``col`` (log column span). No absolute
  counts.

Exclusions: ``flybrain_wiring_features`` rules for the target (``cell_family``
is registered here with the type-hierarchy patterns plus family/lineage
names) and ``OL_EXTRA_EXCLUSIONS`` (for ``connectivity_tier``: column span,
ROI entropy and the output/input ratio, all size proxies). Body ids, types,
instances, hemilineage and ``assignedOlHex*`` are never features.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import flybrain_nt_ground_truth as ntgt
import flybrain_target_registry as ftr
import flybrain_wiring_features as fwf
from flybrain_ol_adapter import (
    OL_NT_UNCLEAR,
    OL_VERSION_ID,
    ROLE_NEUPRINT_META,
    ROLE_NEURONS,
    OlAdapterError,
    OlAdapterErrorCode,
    is_safe_relative_path,
    layer_slug,
    load_ol_release_meta,
    map_ol_roi,
    nt_short_code,
    open_ol_snapshot,
    parse_roi_info,
    sha256_file,
)

OL_SYMBOL = "ol"
TARGET_CELL_FAMILY = "cell_family"
TARGET_SUPER_CLASS = "super_class"
TARGET_CONNECTIVITY = "connectivity_tier"
TARGET_NT_PREDICTED = "neurotransmitter_dominance"
TARGET_NT_CONSENSUS = "nt_ground_truth"
NT_LITERATURE_TARGETS: tuple[str, ...] = ntgt.NT_LITERATURE_TARGETS
OL_TARGETS: tuple[str, ...] = (TARGET_CELL_FAMILY, TARGET_SUPER_CLASS, TARGET_CONNECTIVITY, TARGET_NT_PREDICTED,
                               TARGET_NT_CONSENSUS, *NT_LITERATURE_TARGETS)
_P = ftr.LabelProvenance
# R1 label provenance per target (must equal flybrain_target_registry; checked at import).
TARGET_LABEL_PROVENANCE: Mapping[str, str] = {
    TARGET_CELL_FAMILY: _P.CONNECTIVITY_DEFINED.value,
    TARGET_SUPER_CLASS: _P.CURATED_MORPHOLOGY.value,
    TARGET_CONNECTIVITY: _P.CONNECTIVITY_DEFINED.value,
    TARGET_NT_PREDICTED: _P.MODEL_PREDICTED.value,
    TARGET_NT_CONSENSUS: _P.MEASURED.value,
    **{t: _P.MEASURED.value for t in NT_LITERATURE_TARGETS},
}
ftr.check_module_provenance(OL_SYMBOL, TARGET_LABEL_PROVENANCE)
GROUP_KEYS: tuple[str, ...] = ("cell_type",)
ROLE_EDGELIST = "edgelist"
ROLE_BODY_STATS = "body_stats"
FEATURE_SETS: tuple[str, ...] = ("full", "wiring_only", "anatomy_only")
WIRING_FAMILIES = ("degree", "out_comp", "in_comp", "out2_comp", "in2_comp", "recip")
ANATOMY_FAMILIES = ("roi", "layer", "col")
OTHER_FAMILY = "other"
UNCLEAR_CATEGORY = "unclear"
CENTRAL_FAMILY = "central_brain"

# Raw neuPrint columns read from Neuprint_Neurons.feather (projection only).
NEURON_TABLE_COLUMNS: Mapping[str, str] = {
    "body_id": "bodyId:long",
    "cell_type": "type:string",
    "instance": "instance:string",
    "status": "status:string",
    "pre": "pre:int",
    "post": "post:int",
    "downstream": "downstream:int",
    "predicted_nt": "predictedNt:string",
    "predicted_nt_confidence": "predictedNtConfidence:float",
    "total_nt_predictions": "totalNtPredictions:float",
    "consensus_nt": "consensusNt:string",
    "nt_reference": "ntReference:string",
    "hemilineage": "hemilineage:string",
    "soma_location": "somaLocation:point{srid:9157}",
    "assigned_ol_hex1": "assignedOlHex1:float",
    "roi_info": "roiInfo:string",
}
_OPTIONAL_NEURON_COLUMNS = frozenset({"consensus_nt", "nt_reference"})

# ---------------------------------------------------------------------------
# exclusions
# ---------------------------------------------------------------------------

_FAMILY_EXTRA_PATTERNS = ("*family*", "*families*", "*hemilineage*", "*lineage*", "ito*", "truman*", "*instance*",
                          "*hex*", "*serial*", "*flywire*")


def register_ol_exclusions() -> None:
    """Register ``cell_family`` (idempotent; never replaces someone else's rule)."""
    if TARGET_CELL_FAMILY not in fwf.registered_objectives():
        base = fwf.objective_exclusions("cell_class").patterns
        fwf.register_objective_exclusions(
            TARGET_CELL_FAMILY, (*base, *_FAMILY_EXTRA_PATTERNS),
            reason="label is a grouping of the curated cell type; type/class/lineage annotations determine it")


register_ol_exclusions()

# Extra, ol-specific exclusions applied after the fwf rule (feature names that
# the shared patterns do not catch but that are size proxies / label sources).
OL_EXTRA_EXCLUSIONS: Mapping[str, tuple[str, ...]] = {
    TARGET_CONNECTIVITY: ("col__*", "roi__entropy_bits", "roi__out_in_ratio"),
    TARGET_CELL_FAMILY: (),
    TARGET_SUPER_CLASS: (),
    TARGET_NT_PREDICTED: (),
    TARGET_NT_CONSENSUS: (),
    **{t: () for t in NT_LITERATURE_TARGETS},
}


def ol_excluded(names: Sequence[str], target: str) -> list[str]:
    patterns = OL_EXTRA_EXCLUSIONS.get(target, ())
    return sorted(n for n in names if any(fnmatch.fnmatchcase(n, p) for p in patterns))


def assert_ol_features_allowed(names: Sequence[str], target: str) -> None:
    fwf.assert_features_allowed(list(names), target)
    offenders = ol_excluded(names, target)
    if offenders:
        raise fwf.LabelLeakageError(f"ol features excluded for {target!r}: {', '.join(offenders[:20])}")


# ---------------------------------------------------------------------------
# family map
# ---------------------------------------------------------------------------

# Prefix merges (documented in docs/FLYBRAIN_OL_ADAPTER_CONTRACT.md "Real models").
FAMILY_MERGES: Mapping[str, str] = {
    # medulla visual projection sub-series -> MeVP (MeVPMe kept: 13 types, own series in Nern et al.)
    "MeVPLo": "MeVP", "MeVPLp": "MeVP", "MeVPaMe": "MeVP", "MeVPOL": "MeVP",
    "LoVCLo": "LoVC", "MeVCMe": "MeVC",
    # lobula plate tangentials
    "HSE": "HS", "HSN": "HS", "HSS": "HS", "HST": "HS", "VSm": "VS",
    # descending neurons
    "DNg": "DN", "DNge": "DN", "DNp": "DN", "DNpe": "DN", "DNc": "DN", "DNb": "DN", "DNa": "DN",
    "AN": "AN",
}
_NON_LABEL_RE = re.compile(r"(unclear|tbd)", re.IGNORECASE)
_PREFIX_RE = re.compile(r"^([A-Za-z]+)")
_CENTRAL_CLASSES = frozenset({"central", "ascending_neuron", "motor"})


def is_provisional_type(cell_type: str) -> bool:
    text = str(cell_type).strip()
    return not text or bool(_NON_LABEL_RE.search(text)) or text.startswith("(")


def raw_family(cell_type: str, type_class: str | None = None) -> str | None:
    """Family before rare-family pooling; None for provisional types (never a label)."""
    text = str(cell_type).strip()
    if is_provisional_type(text):
        return None
    match = _PREFIX_RE.match(text)
    prefix = match.group(1) if match else text
    if prefix.startswith("DN"):
        return "DN"
    if type_class in _CENTRAL_CLASSES:
        return CENTRAL_FAMILY
    return FAMILY_MERGES.get(prefix, prefix)


@dataclass(frozen=True)
class FamilyMap:
    type_to_family: Mapping[str, str]  # labelled types only
    raw_to_family: Mapping[str, str]
    family_type_counts: Mapping[str, int]
    min_types: int
    sha256: str

    def family(self, cell_type: str) -> str | None:
        return self.type_to_family.get(str(cell_type))

    def as_dict(self) -> dict[str, Any]:
        return {"min_types": self.min_types, "sha256": self.sha256,
                "family_type_counts": dict(sorted(self.family_type_counts.items())),
                "pooled_into_other": sorted(k for k, v in self.raw_to_family.items() if v == OTHER_FAMILY),
                "type_to_family": dict(sorted(self.type_to_family.items()))}


def build_family_map(type_classes: Mapping[str, str | None], *, min_types: int = 5) -> FamilyMap:
    """``type_classes``: cell type -> majority flat-connectome class (None if unknown)."""
    raw = {t: raw_family(t, c) for t, c in type_classes.items()}
    counts: dict[str, int] = {}
    for fam in raw.values():
        if fam is not None:
            counts[fam] = counts.get(fam, 0) + 1
    raw_to_family = {fam: (fam if n >= int(min_types) else OTHER_FAMILY) for fam, n in counts.items()}
    type_to_family = {t: raw_to_family[f] for t, f in raw.items() if f is not None}
    final_counts: dict[str, int] = {}
    for fam in type_to_family.values():
        final_counts[fam] = final_counts.get(fam, 0) + 1
    digest = hashlib.sha256(json.dumps({"min_types": int(min_types), "map": sorted(type_to_family.items())},
                                       sort_keys=True).encode("utf-8")).hexdigest()
    return FamilyMap(type_to_family, raw_to_family, final_counts, int(min_types), digest)


def partner_category(cell_type: str, family_map: FamilyMap) -> str:
    """Wiring partner category: the family; provisional types -> ``unclear``."""
    fam = family_map.family(cell_type)
    return UNCLEAR_CATEGORY if fam is None else fam


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


@dataclass
class OlStructuredInputs:
    neurons: pd.DataFrame  # every typed body (any status), canonical columns
    body_class: pd.Series  # body_id (str) -> flat-connectome class
    edges: fwf.EdgeSource
    provenance: Mapping[str, Any]


def _manifest_entry(manifest: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    entries = [f for f in manifest.get("integrity", {}).get("files", []) if f.get("role") == role]
    if len(entries) != 1:
        raise OlAdapterError(OlAdapterErrorCode.PRODUCT_MISSING, f"manifest lists {len(entries)} files for role {role!r}")
    return entries[0]


def _verified_extra(snapshot_root: Path, manifest: Mapping[str, Any], role: str, *, hash_check: bool) -> tuple[Path, str]:
    entry = _manifest_entry(manifest, role)
    rel = str(entry["relative_path"])
    if not is_safe_relative_path(rel):
        raise OlAdapterError(OlAdapterErrorCode.PATH_ESCAPE, f"unsafe path for {role}", {"path": rel})
    path = (snapshot_root / rel).resolve(strict=False)
    if snapshot_root.resolve(strict=False) not in path.parents:
        raise OlAdapterError(OlAdapterErrorCode.PATH_ESCAPE, f"{role} resolves outside the snapshot", {"path": str(path)})
    if not path.is_file():
        raise OlAdapterError(OlAdapterErrorCode.PRODUCT_MISSING, f"{role} file missing", {"path": str(path)})
    if path.stat().st_size != int(entry["size_bytes"]):
        raise OlAdapterError(OlAdapterErrorCode.INTEGRITY_MISMATCH, f"{role} size differs from manifest")
    sha = str(entry["sha256"])
    if hash_check:
        actual = sha256_file(path)
        if actual != sha:
            raise OlAdapterError(OlAdapterErrorCode.INTEGRITY_MISMATCH, f"{role} sha256 differs from manifest",
                                 {"expected": sha, "actual": actual})
    return path, sha


def _read_neurons(path: Path) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(path), format="ipc")
    names = set(dataset.schema.names)
    wanted = {k: v for k, v in NEURON_TABLE_COLUMNS.items() if v in names}
    missing = [v for k, v in NEURON_TABLE_COLUMNS.items() if v not in names and k not in _OPTIONAL_NEURON_COLUMNS]
    if missing:
        raise OlAdapterError(OlAdapterErrorCode.SCHEMA_MISMATCH, "neuron table lacks columns",
                             {"missing": missing, "path": str(path)})
    type_field = ds.field(NEURON_TABLE_COLUMNS["cell_type"])
    table = dataset.to_table(columns=list(wanted.values()), filter=pc.is_valid(type_field) & (type_field != ""))
    table = table.rename_columns(list(wanted))
    table = table.set_column(table.schema.get_field_index("body_id"), "body_id", table.column("body_id").cast(pa.string()))
    frame = table.to_pandas()
    for key in _OPTIONAL_NEURON_COLUMNS:
        if key not in frame.columns:
            frame[key] = None
    frame["cell_type"] = frame["cell_type"].astype(str).str.strip()
    frame["status"] = frame["status"].fillna("").astype(str)
    if frame["body_id"].duplicated().any():
        raise OlAdapterError(OlAdapterErrorCode.SCHEMA_MISMATCH, "neuron table bodyId is not unique")
    return frame.reset_index(drop=True)


def _read_body_class(path: Path) -> pd.Series:
    import pyarrow as pa
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(path), format="ipc")
    for col in ("body", "class"):
        if col not in dataset.schema.names:
            raise OlAdapterError(OlAdapterErrorCode.SCHEMA_MISMATCH, f"body stats lack column {col!r}")
    table = dataset.to_table(columns=["body", "class"], filter=ds.field("class").is_valid())
    body = table.column("body").cast(pa.string()).to_pylist()
    return pd.Series(table.column("class").to_pylist(), index=body, dtype=object)


def load_ol_structured_inputs(
    storage_root: str | Path | None = None,
    *,
    verify_extra_hashes: bool = True,
    neurons_path: str | Path | None = None,
    meta_path: str | Path | None = None,
    edges_path: str | Path | None = None,
    body_stats_path: str | Path | None = None,
) -> OlStructuredInputs:
    """Open the snapshot read-only (or explicit paths) and load node table + class + edge source."""
    explicit = (neurons_path, meta_path, edges_path, body_stats_path)
    if any(p is not None for p in explicit):
        if not all(p is not None for p in explicit):
            raise ValueError("explicit ol paths must be given for neurons, meta, edges and body stats together")
        paths = {k: Path(v).resolve(strict=True) for k, v in zip(("neurons", "meta", ROLE_EDGELIST, ROLE_BODY_STATS),
                                                                   explicit)}
        provenance = {"mode": "explicit_paths", "manifest_id": None, "manifest_sha256": None,
                      "product_sha256": {k: sha256_file(p) for k, p in sorted(paths.items())}}
    else:
        snapshot = open_ol_snapshot(storage_root, required_roles=(ROLE_NEURONS, ROLE_NEUPRINT_META),
                                    verify_hashes=False)
        manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
        edges, edges_sha = _verified_extra(snapshot.snapshot_root, manifest, ROLE_EDGELIST, hash_check=verify_extra_hashes)
        stats, stats_sha = _verified_extra(snapshot.snapshot_root, manifest, ROLE_BODY_STATS,
                                           hash_check=verify_extra_hashes)
        paths = {"neurons": snapshot.path(ROLE_NEURONS), "meta": snapshot.path(ROLE_NEUPRINT_META),
                 ROLE_EDGELIST: edges, ROLE_BODY_STATS: stats}
        provenance = {"mode": "manifest_snapshot", "manifest_id": snapshot.manifest_id,
                      "manifest_sha256": snapshot.manifest_sha256,
                      "product_sha256": {**dict(snapshot.product_sha256), ROLE_EDGELIST: edges_sha,
                                         ROLE_BODY_STATS: stats_sha},
                      "hash_verification": {**dict(snapshot.hash_verification),
                                            ROLE_EDGELIST: "hashed" if verify_extra_hashes else "size_only",
                                            ROLE_BODY_STATS: "hashed" if verify_extra_hashes else "size_only"}}
    release = load_ol_release_meta(paths["meta"])
    neurons = _read_neurons(paths["neurons"])
    body_class = _read_body_class(paths[ROLE_BODY_STATS])
    edge_source = fwf.EdgeSource(
        path=str(paths[ROLE_EDGELIST]), pre="body_pre", post="body_post", weight="weight", unique_pairs=True,
        format="ipc", provenance={"manifest_sha256": provenance["manifest_sha256"],
                                  "sha256": provenance["product_sha256"][ROLE_EDGELIST]})
    provenance = {**provenance, "dataset_version": OL_VERSION_ID, "neuprint_release": f"{release.dataset}:{release.tag}",
                  "paths": {k: str(v) for k, v in sorted(paths.items())}}
    return OlStructuredInputs(neurons=neurons, body_class=body_class, edges=edge_source, provenance=provenance)


def type_majority_class(inputs: OlStructuredInputs) -> dict[str, str | None]:
    cls = inputs.neurons["body_id"].map(inputs.body_class)
    frame = pd.DataFrame({"t": inputs.neurons["cell_type"], "c": cls})
    out: dict[str, str | None] = {}
    for t, group in frame.groupby("t", sort=True):
        counts = group["c"].dropna().astype(str).value_counts()
        out[str(t)] = None if counts.empty else sorted(counts[counts == counts.max()].index)[0]
    return out


# ---------------------------------------------------------------------------
# anatomy (roiInfo) features
# ---------------------------------------------------------------------------

_LAYER_ORDER = tuple([f"me_r_layer_{i:02d}" for i in range(1, 11)] + [f"lo_r_layer_{i:02d}" for i in range(1, 8)]
                     + [f"lop_r_layer_{i:02d}" for i in range(1, 5)])


def anatomy_features(roi_infos: Sequence[Any], body_ids: Sequence[str], *, top_rois: Sequence[str] | None = None,
                     n_top_rois: int = 24) -> tuple[pd.DataFrame, list[str]]:
    """Per-neuron ROI / layer / column features from ``roiInfo`` (fractions only; no counts)."""
    summaries = [parse_roi_info(raw) for raw in roi_infos]
    if top_rois is None:
        totals: dict[str, int] = {}
        for s in summaries:
            for roi, c in s.primary.items():
                totals[roi] = totals.get(roi, 0) + c.synweight
        top_rois = [r for r, _ in sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[: int(n_top_rois)]]
    top = list(top_rois)
    slugs = {r: map_ol_roi(r).slug for r in top}
    rows = []
    for s in summaries:
        pre_total = sum(c.pre for c in s.primary.values())
        post_total = sum(c.post for c in s.primary.values())
        syn_total = sum(c.synweight for c in s.primary.values())
        row: dict[str, float] = {}
        for direction, total, attr in (("in", post_total, "post"), ("out", pre_total, "pre")):
            for r in top:
                c = s.primary.get(r)
                row[f"roi__{slugs[r]}_{direction}"] = (getattr(c, attr) / total if c else 0.0) if total else math.nan
            covered = sum(getattr(s.primary[r], attr) for r in top if r in s.primary)
            row[f"roi__other_{direction}"] = (1.0 - covered / total) if total else math.nan
            ol = sum(getattr(c, attr) for roi, c in s.primary.items() if map_ol_roi(roi).division == "optic_lobe")
            row[f"roi__optic_lobe_share_{direction}"] = ol / total if total else math.nan
        if syn_total:
            p = np.asarray([c.synweight for c in s.primary.values()], dtype=float) / syn_total
            p = p[p > 0]
            row["roi__entropy_bits"] = float(-(p * np.log2(p)).sum())
        else:
            row["roi__entropy_bits"] = math.nan
        row["roi__out_in_ratio"] = pre_total / (pre_total + post_total) if pre_total + post_total else math.nan
        layer_weights = {layer_slug(k): v for k, v in s.layers.items()}
        layer_total = sum(layer_weights.values())
        for name in _LAYER_ORDER:
            row[f"layer__{name}"] = layer_weights.get(name, 0) / layer_total if layer_total else math.nan
        row["layer__ol_layer_share"] = layer_total / syn_total if syn_total else math.nan
        row["col__log1p_column_span"] = math.log1p(s.n_columns)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.insert(0, "body_id", [str(b) for b in body_ids])
    return frame, top


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OlTargetConfig:
    target: str
    max_per_type: int = 20
    min_types: int = 5
    min_synapses: int = 10
    min_total_count: int = 10
    high_connectivity_quantile: float = 0.75
    allowed_statuses: tuple[str, ...] = ("Traced",)
    feature_set: str = "full"
    two_hop: bool = True
    n_top_rois: int = 24
    cache_root: str | None = fwf.DEFAULT_CACHE_ROOT
    salt: str = "ol11-real-models"
    nt_root: str = ntgt.DEFAULT_NT_GT_ROOT  # R2 literature NT (nt_literature* targets)

    def validate(self) -> None:
        if self.target not in OL_TARGETS:
            raise ValueError(f"target must be one of {', '.join(OL_TARGETS)}")
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {', '.join(FEATURE_SETS)}")
        if self.max_per_type < 1 or self.min_types < 1:
            raise ValueError("max_per_type and min_types must be >= 1")
        if not 0.0 < self.high_connectivity_quantile < 1.0:
            raise ValueError("high_connectivity_quantile must be in (0, 1)")


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or not str(value).strip() \
        or str(value).strip().lower() in {"nan", "none", "<na>"}


def label_frame(inputs: OlStructuredInputs, config: OlTargetConfig, family_map: FamilyMap) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Candidate neurons with a label for ``config.target`` (before the per-type cap)."""
    n = inputs.neurons
    status_ok = n["status"].isin(set(config.allowed_statuses))
    frame = n[status_ok & ~n["cell_type"].map(is_provisional_type)].copy()
    stats: dict[str, Any] = {"typed_bodies": int(len(n)), "status_and_type_ok": int(len(frame))}
    size = frame["pre"].fillna(0).astype("int64") + frame["post"].fillna(0).astype("int64")
    frame = frame[size >= int(config.min_synapses)]
    stats["min_synapses_ok"] = int(len(frame))
    target = config.target
    if target == TARGET_CELL_FAMILY:
        frame["label"] = frame["cell_type"].map(family_map.family)
        definition = "cell-type family (FamilyMap; rare families pooled into 'other')"
    elif target == TARGET_SUPER_CLASS:
        raw = frame["body_id"].map(inputs.body_class)
        tbc = raw.astype(str).str.lower().str.endswith("_tbc")
        frame["label"] = [None if (_missing(v) or t) else fwf.harmonize_super_class(v) for v, t in zip(raw, tbc)]
        stats["dropped_tbc_or_missing_class"] = int(frame["label"].isna().sum())
        definition = "flat-connectome body class, harmonize_super_class; *_tbc dropped"
    elif target == TARGET_CONNECTIVITY:
        down = frame["downstream"].fillna(0).astype("int64")
        frame = frame[down >= int(config.min_total_count)].copy()
        down = frame["downstream"].astype("int64")
        threshold = float(np.quantile(down.to_numpy(dtype=float), config.high_connectivity_quantile))
        frame["label"] = np.where(down >= threshold, "high_connectivity", "baseline_connectivity")
        stats["high_connectivity_threshold"] = threshold
        definition = (f"neuPrint downstream >= {config.high_connectivity_quantile} quantile ({threshold:.0f}) over "
                      f"candidates with downstream >= {config.min_total_count}")
    elif target == TARGET_NT_PREDICTED:
        pred = frame["predicted_nt"].astype(str).str.strip().str.lower()
        ok = ~frame["predicted_nt"].isna() & (pred != "") & (pred != OL_NT_UNCLEAR) & (pred != "nan")
        ok &= frame["total_nt_predictions"].fillna(0).astype(float) >= float(config.min_total_count)
        frame = frame[ok].copy()
        frame["label"] = ["dominant_" + nt_short_code(v) for v in frame["predicted_nt"]]
        definition = "PREDICTED: per-neuron synapse-classifier predictedNt (unclear dropped)"
    elif target == TARGET_NT_CONSENSUS:
        if frame["nt_reference"].isna().all() and frame["consensus_nt"].isna().all():
            raise OlAdapterError(OlAdapterErrorCode.SCHEMA_MISMATCH, "consensusNt / ntReference columns are absent")
        cons = frame["consensus_nt"].astype(str).str.strip().str.lower()
        ok = ~frame["nt_reference"].map(_missing) & ~frame["consensus_nt"].map(_missing) & (cons != OL_NT_UNCLEAR)
        frame = frame[ok].copy()
        frame["label"] = ["nt_" + nt_short_code(v) for v in frame["consensus_nt"]]
        definition = "CONSENSUS: type-level consensusNt where ntReference is set (literature/experiment-backed)"
    elif target in NT_LITERATURE_TARGETS:
        labelled = ntgt.label_neurons(frame[["body_id", "cell_type"]], dataset=OL_SYMBOL,
                                      source=ntgt.open_nt_ground_truth(config.nt_root), target=target,
                                      id_column="body_id")
        mapping = dict(zip(labelled.frame["body_id"].astype(str), labelled.frame["label"].astype(str)))
        frame["label"] = [mapping.get(str(b)) for b in frame["body_id"]]
        stats["nt_literature_coverage"] = labelled.coverage
        definition = ntgt.NT_LITERATURE_SPECS[target].describe()
    else:  # pragma: no cover - validated
        raise ValueError(target)
    frame = frame[frame["label"].notna()].copy()
    # classes need >= min_types distinct types to be learnable under a type-grouped split
    type_counts = frame.groupby("label")["cell_type"].nunique()
    keep = set(type_counts[type_counts >= int(config.min_types)].index)
    stats["dropped_classes_few_types"] = {str(k): int(v) for k, v in type_counts.items() if k not in keep}
    frame = frame[frame["label"].isin(keep)].copy()
    stats["candidates"] = int(len(frame))
    stats["label_definition"] = definition
    return frame.reset_index(drop=True), stats


def cap_per_type(frame: pd.DataFrame, *, max_per_type: int, salt: str) -> pd.DataFrame:
    """At most ``max_per_type`` neurons per cell type, in sha256(salt:body_id) order (deterministic, id-order-free)."""
    key = frame["body_id"].map(lambda b: hashlib.sha256(f"{salt}:{b}".encode("utf-8")).hexdigest())
    ordered = frame.assign(_k=key).sort_values(["cell_type", "_k"], kind="stable")
    capped = ordered.groupby("cell_type", sort=True).head(int(max_per_type)).drop(columns="_k")
    return capped.sort_values("body_id", key=lambda s: s.str.zfill(24), kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# text (legacy NB view)
# ---------------------------------------------------------------------------

def _text_keys(target: str) -> tuple[str, ...]:
    import flybrain_brain_cluster_ol_samples as ols

    if target == TARGET_CONNECTIVITY:
        return tuple(ols.INPUT_FEATURES["connectivity_tier"])
    if target in (TARGET_NT_PREDICTED, TARGET_NT_CONSENSUS, *NT_LITERATURE_TARGETS):
        return tuple(ols.INPUT_FEATURES["neurotransmitter_dominance"])
    return ols.STRUCTURED_TEXT_FEATURES


def input_texts(frame: pd.DataFrame, target: str) -> list[str]:
    import flybrain_brain_cluster_ol_samples as ols

    tokens = ols.anatomy_tokens(frame)
    keys = _text_keys(target)
    return [ols.render_input_text(tokens[str(b)], keys) for b in frame["body_id"]]


# ---------------------------------------------------------------------------
# eval dataset
# ---------------------------------------------------------------------------


def _feature_columns(columns: Sequence[str], feature_set: str) -> list[str]:
    fams = WIRING_FAMILIES if feature_set == "wiring_only" else ANATOMY_FAMILIES if feature_set == "anatomy_only" \
        else WIRING_FAMILIES + ANATOMY_FAMILIES
    return [c for c in columns if fwf.feature_family(c) in fams]


def build_ol_eval_dataset(inputs: OlStructuredInputs, config: OlTargetConfig, eval_config: Any,
                          *, family_map: FamilyMap | None = None) -> tuple[Any, dict[str, Any]]:
    """Labels -> per-type cap -> split plan -> masked wiring + anatomy features -> ``EvalDataset``."""
    import flybrain_model_eval as fme

    config.validate()
    target = config.target
    if family_map is None:
        family_map = build_family_map(type_majority_class(inputs), min_types=config.min_types)
    candidates, stats = label_frame(inputs, config, family_map)
    samples = cap_per_type(candidates, max_per_type=config.max_per_type, salt=config.salt)
    if samples.empty or samples["label"].nunique() < 2:
        raise ValueError(f"{target}: fewer than two labels after filtering")
    short = {TARGET_CELL_FAMILY: "fam", TARGET_SUPER_CLASS: "cls", TARGET_CONNECTIVITY: "out",
             TARGET_NT_PREDICTED: "nt", TARGET_NT_CONSENSUS: "ntc", ntgt.TARGET_NT_LITERATURE: "ntl",
             ntgt.TARGET_NT_LITERATURE_BINARY: "ntlb", ntgt.TARGET_NT_LITERATURE_ALL: "ntla",
             ntgt.TARGET_NT_LITERATURE_ALL_BINARY: "ntlab"}[target]
    samples["sample_id"] = [f"ol11-{short}-{b}" for b in samples["body_id"]]
    group_values = [{"cell_type": t} for t in samples["cell_type"]]
    plan = fme.plan_grouped_split(samples["sample_id"].tolist(), group_values, GROUP_KEYS, eval_config)
    held_ids = set(plan["val"]) | set(plan["test"])
    held_types = set(samples.loc[samples["sample_id"].isin(held_ids), "cell_type"])
    nodes = inputs.neurons[["body_id", "cell_type"]].copy()
    nodes["partner_category"] = [partner_category(t, family_map) for t in nodes["cell_type"]]
    mask = nodes.loc[nodes["cell_type"].isin(held_types), "body_id"].tolist()

    wiring = fwf.build_wiring_features(
        dataset=OL_SYMBOL, objective=target, edges=inputs.edges, nodes=nodes, id_column="body_id",
        category_column="partner_category", params=fwf.WiringFeatureParams(two_hop=config.two_hop, top_k_neuropils=0),
        cache_root=config.cache_root, mask_category_ids=mask)
    anatomy, top_rois = anatomy_features(samples["roi_info"].tolist(), samples["body_id"].tolist(),
                                         n_top_rois=config.n_top_rois)
    anatomy, anat_dropped = fwf.apply_objective_exclusions(anatomy.rename(columns={"body_id": fwf.NODE_ID_COLUMN}),
                                                           target)
    frame = samples.merge(wiring.frame, left_on="body_id", right_on=fwf.NODE_ID_COLUMN, how="left", validate="1:1")
    frame = frame.drop(columns=[fwf.NODE_ID_COLUMN]).merge(
        anatomy, left_on="body_id", right_on=fwf.NODE_ID_COLUMN, how="left", validate="1:1").drop(
        columns=[fwf.NODE_ID_COLUMN])
    all_features = [c for c in list(wiring.frame.columns) + list(anatomy.columns) if c != fwf.NODE_ID_COLUMN]
    extra_dropped = ol_excluded(all_features, target)
    candidates_cols = [c for c in all_features if c not in set(extra_dropped)]
    feature_cols = sorted(_feature_columns(candidates_cols, config.feature_set))
    # constant columns carry nothing and only slow the learners
    feature_cols = [c for c in feature_cols if frame[c].nunique(dropna=False) > 1]
    assert_ol_features_allowed(feature_cols, target)
    frame["input_text"] = input_texts(frame, target)
    wiring_meta = {k: v for k, v in wiring.meta.items() if k not in ("columns",)}
    notes = {
        **ftr.provenance_notes(OL_SYMBOL, target),
        "masked_split_ids_sha256": fme.split_ids_sha256(plan),
        "masking": {"held_out_types": len(held_types), "masked_nodes": len(mask),
                    "rule": "partner category hidden for every node of a val/test cell type"},
        "label_definition": stats["label_definition"],
        "label_stats": stats,
        "sampling": {"max_per_type": config.max_per_type, "order": f"sha256({config.salt}:body_id)",
                     "samples": int(len(frame)), "types": int(frame["cell_type"].nunique()),
                     "allowed_statuses": list(config.allowed_statuses), "min_synapses": config.min_synapses},
        "feature_set": config.feature_set,
        "excluded_features": {"fwf": sorted(set(wiring.excluded_features) | set(anat_dropped)), "ol_extra": extra_dropped},
        "top_rois": list(top_rois),
        "family_map_sha256": family_map.sha256,
        "family_type_counts": dict(sorted(family_map.family_type_counts.items())),
        "wiring": {"fingerprint": wiring.fingerprint, "cache_path": wiring.cache_path, "meta": wiring_meta},
        "provenance": dict(inputs.provenance),
    }
    data = fme.EvalDataset.from_frame(
        frame, dataset=OL_SYMBOL, target=target, id_column="sample_id", label_column="label",
        feature_columns=feature_cols, group_columns=list(GROUP_KEYS), text_column="input_text", notes=notes)
    return data, {"plan": plan, "frame": frame, "family_map": family_map, "feature_columns": feature_cols}


# ---------------------------------------------------------------------------
# harness driver
# ---------------------------------------------------------------------------


def default_models(n_classes: int) -> tuple[Any, ...]:
    import flybrain_model_eval as fme

    calibration = "isotonic" if n_classes == 2 else "temperature"
    return (
        fme.ModelSpec("nb", ({},), calibration, name="nb"),
        fme.ModelSpec("logreg", ({"C": 0.1, "max_iter": 3000}, {"C": 1.0, "max_iter": 3000},
                                 {"C": 1.0, "class_weight": "balanced", "max_iter": 3000}), calibration, name="logreg"),
        fme.ModelSpec("hgb", ({"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 20},
                              {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 40,
                               "l2_regularization": 1.0}), calibration, name="hgb"),
    )


def run_ol_target(inputs: OlStructuredInputs, target: str, *, feature_set: str = "full",
                  report_root: str | None = None, family_map: FamilyMap | None = None,
                  target_overrides: Mapping[str, Any] | None = None, eval_overrides: Mapping[str, Any] | None = None,
                  models: Sequence[Any] | None = None) -> dict[str, Any]:
    import flybrain_model_eval as fme

    config = OlTargetConfig(target=target, feature_set=feature_set, **dict(target_overrides or {}))
    eval_kwargs: dict[str, Any] = {"run_label": "" if feature_set == "full" else feature_set}
    if report_root is not None:
        eval_kwargs["report_root"] = report_root
    eval_kwargs.update(dict(eval_overrides or {}))
    eval_config = fme.EvalConfig(**eval_kwargs)
    data, extra = build_ol_eval_dataset(inputs, config, eval_config, family_map=family_map)
    n_classes = len(set(data.labels.tolist()))
    if models is None:
        models = default_models(n_classes)
        if feature_set != "full":  # the NB text view does not change with the feature set
            models = tuple(m for m in models if m.backend != "nb")
    eval_config = replace(eval_config, models=tuple(models))
    report = ftr.run_gated_evaluation(data, eval_config)
    if eval_config.report_root and target == TARGET_CELL_FAMILY:
        out = fme.report_dir(eval_config.report_root, OL_SYMBOL, target, eval_config.run_label)
        (out / "family_map.json").write_text(json.dumps(extra["family_map"].as_dict(), indent=2, sort_keys=True) + "\n",
                                             encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train + evaluate ol real-model targets with the shared harness.")
    parser.add_argument("--storage-root", default="/mnt/f/.flybrain")
    parser.add_argument("--targets", default=",".join(OL_TARGETS))
    parser.add_argument("--feature-sets", default="full")
    parser.add_argument("--report-root", default=None)
    parser.add_argument("--max-per-type", type=int, default=20)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--no-extra-hash", action="store_true", help="size-check (not hash) edges/body stats")
    args = parser.parse_args(argv)
    inputs = load_ol_structured_inputs(args.storage_root, verify_extra_hashes=not args.no_extra_hash)
    family_map = build_family_map(type_majority_class(inputs))
    rows = []
    for target in [t for t in args.targets.split(",") if t]:
        for feature_set in [f for f in args.feature_sets.split(",") if f]:
            report = run_ol_target(inputs, target, feature_set=feature_set, report_root=args.report_root,
                                   family_map=family_map, target_overrides={"max_per_type": args.max_per_type},
                                   eval_overrides={"n_bootstrap": args.n_bootstrap})
            for row in report["summary"]:
                rows.append({**row, "feature_set": feature_set})
            print(json.dumps({"target": target, "feature_set": feature_set, "summary": report["summary"],
                              "elapsed": report["elapsed_seconds"]}, default=str), flush=True)
    print(json.dumps({"rows": rows}, default=str))
    return 0


__all__ = [
    "ANATOMY_FAMILIES", "FAMILY_MERGES", "FEATURE_SETS", "FamilyMap", "GROUP_KEYS", "OL_EXTRA_EXCLUSIONS", "OL_TARGETS",
    "OlStructuredInputs", "OlTargetConfig", "WIRING_FAMILIES", "anatomy_features", "assert_ol_features_allowed",
    "build_family_map", "build_ol_eval_dataset", "cap_per_type", "default_models", "input_texts", "is_provisional_type",
    "label_frame", "load_ol_structured_inputs", "ol_excluded", "partner_category", "raw_family",
    "register_ol_exclusions", "run_ol_target", "type_majority_class",
]


if __name__ == "__main__":
    raise SystemExit(main())
