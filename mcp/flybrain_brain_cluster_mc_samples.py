"""MaleCNS v1.0 (registry ``mc``) brain-cluster training-sample builder.

Registered with ``flybrain_brain_cluster_samples`` for ``connectivity_tier``,
``neurotransmitter_dominance`` and ``region_specialization_tier``. Reads only the
local, manifest-verified snapshot (``flybrain_mc_adapter``) or explicit local
paths; never the network.

Candidates are *typed* neurons only: the body-annotation ``type`` is set and
``status`` is ``Traced`` or ``Anchor`` (~162k of the ~88 M bodies in
``Neuprint_Neurons.feather``; the large tables are streamed with that body-id
filter). Each neuron's region is its *primary neuropil*: the neuPrint primary
ROI holding the most of its pre + post synapses (explicit 144-ROI vocabulary,
``map_mc_primary_roi``). ``region_id`` (the routing key) is that neuropil
without the hemisphere, e.g. ``brain_me`` or ``nerve_cord_legnp_t1``.

Label definitions (none is computable from the model input):

* ``connectivity_tier``: ``high_connectivity`` when the neuron's total outgoing
  synapse weight (``weight`` summed per ``body_pre`` in the minconf-0.5
  neuron-neuron table) is at or above the configured global quantile of the
  candidates, else ``baseline_connectivity``. Confidence uses fw's
  distance-from-threshold formula.
* ``neurotransmitter_dominance``: ``dominant_<code>`` from the per-body
  classifier call ``predicted_nt`` (ach/gaba/glut/da/ser/oct/his); bodies the
  classifier calls ``unclear`` are dropped. Confidence = clipped
  ``predicted_nt_confidence``.
* ``region_specialization_tier``: ``region_specialized`` when at least
  ``specialization_share`` (default 0.75) of the neuron's primary-ROI synapses
  (pre + post) sit in its primary neuropil, else ``region_distributed``.
  The input names *which* neuropil is largest, never how large it is: the
  share, the per-ROI counts and the number of ROIs are forbidden inputs.

``assert_no_label_leakage`` enforces the per-objective input allow-list.
Identity proxies are never inputs: cell type (the strongest proxy for every
label) is the grouped-split key, carried in sample metadata with
``hemilineage``, and the body id is not in ``input_text`` at all (on its own
it predicts connectivity; see ``_IDENTITY_KEYS``).
``trivial_baseline_report`` reproduces the promotion gate's trivial rules on a
grouped split and adds a single-feature lookup audit.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from flybrain_brain_cluster_samples import (
    assemble_training_payload,
    balance_and_cap_samples,
    hash_ordered_preselect,
    register_sample_builder,
    stable_json,
    unregister_sample_builder,
)
from flybrain_dataset_registry import (
    OBJECTIVE_CONNECTIVITY_TIER,
    OBJECTIVE_NEUROTRANSMITTER_DOMINANCE,
    OBJECTIVE_REGION_SPECIALIZATION_TIER,
)
from flybrain_mc_adapter import (
    MC_NT_ABSTAIN,
    MC_VERSION_ID,
    ROLE_DATASET_META,
    ROLE_EDGELIST,
    ROLE_META,
    ROLE_NEURON_ROI_INFO,
    ROLE_NT_PREDICTION,
    McAdapterError,
    McAdapterErrorCode,
    check_mc_roi_vocabulary,
    load_mc_annotations,
    load_mc_neuron_roi_summary,
    load_mc_nt_predictions,
    load_mc_outgoing_totals,
    map_mc_primary_roi,
    nt_short_code,
    open_mc_snapshot,
    sha256_file,
)

MC_SYMBOL = "mc"
MC_SAMPLE_OBJECTIVES: tuple[str, ...] = (
    OBJECTIVE_CONNECTIVITY_TIER,
    OBJECTIVE_NEUROTRANSMITTER_DOMINANCE,
    OBJECTIVE_REGION_SPECIALIZATION_TIER,
)
_DATASET_TAG = "mcns10"
# How the capped subset is chosen (recorded in the input fingerprint).
CAP_ORDER = "sha256-body-id/v1"

LABEL_HIGH = "high_connectivity"
LABEL_BASELINE = "baseline_connectivity"
LABEL_SPECIALIZED = "region_specialized"
LABEL_DISTRIBUTED = "region_distributed"
LABEL_DEFINITIONS: Mapping[str, str] = {
    OBJECTIVE_CONNECTIVITY_TIER: (
        "high_connectivity iff sum(weight) over the neuron's outgoing minconf-0.5 neuron-neuron edges >= "
        "quantile(high_connectivity_quantile) over candidates; else baseline_connectivity"
    ),
    OBJECTIVE_NEUROTRANSMITTER_DOMINANCE: (
        "dominant_<code> of the per-body classifier call predicted_nt (body-neurotransmitters table); "
        "'unclear' calls are dropped"
    ),
    OBJECTIVE_REGION_SPECIALIZATION_TIER: (
        "region_specialized iff (pre+post synapses in the primary neuropil) / (pre+post synapses in all primary "
        "ROIs) >= specialization_share; else region_distributed"
    ),
}

_BASE_FEATURES: tuple[str, ...] = (
    "cns_division",
    "subdivision",
    "primary_neuropil",
    "side",
    "super_class",
    "cell_class",
    "hemilineage",
)
# Model-input feature keys per objective (order = order in input_text).
INPUT_FEATURES: Mapping[str, tuple[str, ...]] = {
    OBJECTIVE_CONNECTIVITY_TIER: _BASE_FEATURES,
    OBJECTIVE_NEUROTRANSMITTER_DOMINANCE: (*_BASE_FEATURES, "out_partner_tier"),
    OBJECTIVE_REGION_SPECIALIZATION_TIER: _BASE_FEATURES,
}
# Identity proxies: never an input for any objective (cell type is the split key).
# The body id is one of them: FlyEM body ids are not random (low ids are large,
# early-proofread bodies), so a one-feature stump on the id alone scored 0.85
# held-out accuracy for connectivity_tier on the real snapshot. Unlike banc/fw,
# mc input_text therefore carries no ``root`` token; the id stays in sample_id,
# provenance_refs and metadata.
_IDENTITY_KEYS = frozenset(
    {
        "cell_type", "type", "instance", "group", "flywire_type", "hemibrain_type", "manc_type", "synonyms",
        "body", "body_id", "root", "root_id",
    }
)
_COUNT_KEYS = frozenset(
    {
        "total_out_synapses",
        "n_post_partners",
        "weight",
        "pre",
        "post",
        "downstream",
        "upstream",
        "synweight",
        "total_nt_predictions",
    }
)
# Keys that must never reach input_text for an objective (label sources / proxies).
FORBIDDEN_INPUT_FEATURES: Mapping[str, frozenset[str]] = {
    OBJECTIVE_CONNECTIVITY_TIER: _IDENTITY_KEYS | _COUNT_KEYS | {"out_partner_tier"},
    OBJECTIVE_NEUROTRANSMITTER_DOMINANCE: _IDENTITY_KEYS
    | {
        "predicted_nt",
        "predicted_nt_confidence",
        "total_nt_predictions",
        "celltype_predicted_nt",
        "celltype_predicted_nt_confidence",
        "celltype_total_nt_predictions",
        "consensus_nt",
        "ground_truth",
        "top_nt",
        "known_nt",
        "nt_acetylcholine_prob",
        "nt_dopamine_prob",
        "nt_gaba_prob",
        "nt_glutamate_prob",
        "nt_histamine_prob",
        "nt_octopamine_prob",
        "nt_serotonin_prob",
    },
    OBJECTIVE_REGION_SPECIALIZATION_TIER: _IDENTITY_KEYS
    | _COUNT_KEYS
    | {
        "top_roi_synapses",
        "total_primary_synapses",
        "primary_share",
        "n_primary_rois",
        "roi_info",
        "out_partner_tier",
    },
}
for _objective, _features in INPUT_FEATURES.items():  # import-time guard
    if set(_features) & FORBIDDEN_INPUT_FEATURES[_objective]:  # pragma: no cover
        raise RuntimeError(f"mc input features overlap the forbidden set for {_objective}")

# Sample-metadata keys (never input_text) that group related neurons for the split.
SPLIT_GROUP_KEYS: tuple[str, ...] = ("cell_type", "hemilineage")
# Hemilineage annotations that are not a lineage (primary neurons / unassigned):
# informative as an input token, but never a split group.
HEMILINEAGE_SENTINELS = frozenset({"primary", "putative_primary", "no_lineage", "tbd", "unknown"})
META_OPTIONAL_COLUMNS: tuple[str, ...] = ("subclass", "somaNeuromere", "group")
_SIDE_TOKENS = {"l": "left", "r": "right", "m": "midline"}


@dataclass(frozen=True)
class McSampleBuildConfig:
    objective: str = OBJECTIVE_CONNECTIVITY_TIER
    storage_root: str | Path | None = None
    # Explicit local overrides. When all needed paths are given the manifest is not consulted.
    meta_path: str | Path | None = None
    nt_path: str | Path | None = None
    edgelist_path: str | Path | None = None
    neuron_path: str | Path | None = None
    dataset_meta_path: str | Path | None = None
    # None: stamp-cached verification of the required products (see
    # flybrain_hash_stamps); True: full rehash; False: size checks only.
    verify_hashes: bool | None = None
    # Hash-stamp directory override (default: <snapshot>/manifest/hash-stamps).
    stamp_dir: str | Path | None = None
    max_samples: int = 5000
    min_total_count: int = 10
    min_region_samples: int = 25
    high_connectivity_quantile: float = 0.75
    specialization_share: float = 0.75
    min_primary_synapses: int = 100
    max_regions: int = 20
    min_distinct_labels: int = 2
    max_label_share: float = 0.9
    allowed_statuses: tuple[str, ...] = ("Traced", "Anchor")
    partner_tier_edges: tuple[int, ...] = (100, 1000)
    exclude_super_classes: tuple[str, ...] = ("glia",)


def _validate_config(config: McSampleBuildConfig) -> None:
    if config.objective not in MC_SAMPLE_OBJECTIVES:
        raise ValueError(f"objective must be one of: {', '.join(MC_SAMPLE_OBJECTIVES)}")
    if config.max_samples <= 0:
        raise ValueError("max_samples must be > 0")
    if config.min_total_count < 1:
        raise ValueError("min_total_count must be >= 1")
    if config.min_primary_synapses < 1:
        raise ValueError("min_primary_synapses must be >= 1")
    if config.min_region_samples < 1:
        raise ValueError("min_region_samples must be >= 1")
    if not (0.0 < config.high_connectivity_quantile < 1.0):
        raise ValueError("high_connectivity_quantile must be in (0, 1)")
    if not (0.0 < config.specialization_share < 1.0):
        raise ValueError("specialization_share must be in (0, 1)")
    if config.max_regions < 1:
        raise ValueError("max_regions must be >= 1")
    if config.min_distinct_labels < 1:
        raise ValueError("min_distinct_labels must be >= 1")
    if not (0.0 < config.max_label_share <= 1.0):
        raise ValueError("max_label_share must be in (0, 1]")
    if not config.allowed_statuses:
        raise ValueError("allowed_statuses must not be empty")
    edges = tuple(int(e) for e in config.partner_tier_edges)
    if not edges or list(edges) != sorted(set(edges)) or edges[0] < 1:
        raise ValueError("partner_tier_edges must be strictly increasing positive integers")


def _required_roles(objective: str) -> tuple[str, ...]:
    roles = [ROLE_META, ROLE_DATASET_META, ROLE_NEURON_ROI_INFO]
    if objective in (OBJECTIVE_CONNECTIVITY_TIER, OBJECTIVE_NEUROTRANSMITTER_DOMINANCE):
        roles.append(ROLE_EDGELIST)  # label source (connectivity) / out_partner_tier input (NT)
    if objective == OBJECTIVE_NEUROTRANSMITTER_DOMINANCE:
        roles.append(ROLE_NT_PREDICTION)
    return tuple(sorted(roles))


def _resolve_inputs(config: McSampleBuildConfig) -> tuple[dict[str, Path], dict[str, Any]]:
    roles = _required_roles(config.objective)
    explicit = {
        ROLE_META: config.meta_path,
        ROLE_NT_PREDICTION: config.nt_path,
        ROLE_EDGELIST: config.edgelist_path,
        ROLE_NEURON_ROI_INFO: config.neuron_path,
        ROLE_DATASET_META: config.dataset_meta_path,
    }
    if all(explicit[role] is not None for role in roles):
        paths = {role: Path(explicit[role]).resolve(strict=False) for role in roles}
        for role, path in paths.items():
            if not path.is_file():
                raise McAdapterError(McAdapterErrorCode.PRODUCT_MISSING, f"mc {role} file not found",
                                     {"path": str(path)})
        return paths, {"mode": "explicit_paths", "manifest_id": None, "manifest_sha256": None,
                       "product_sha256": {role: sha256_file(p) for role, p in sorted(paths.items())},
                       "hash_verification": {}}
    if any(explicit[role] is not None for role in roles):
        raise ValueError("explicit mc paths must be given for every required product or none: " + ", ".join(roles))
    snapshot = open_mc_snapshot(config.storage_root, required_roles=roles, verify_hashes=config.verify_hashes,
                                stamp_dir=config.stamp_dir)
    prov = snapshot.provenance()
    return {role: snapshot.path(role) for role in roles}, {
        "mode": "manifest_snapshot",
        "manifest_id": prov["manifest_id"],
        "manifest_sha256": prov["manifest_sha256"],
        "product_sha256": prov["product_sha256"],
        "hash_verification": dict(snapshot.hash_verification),
    }


def _is_missing(value: Any) -> bool:
    if value is None or (isinstance(value, float) and value != value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "<na>"}


def _clean(value: Any) -> str:
    if _is_missing(value):
        return "unknown"
    return "_".join(str(value).strip().lower().split())


def _side(soma_side: Any, root_side: Any) -> str:
    for value in (soma_side, root_side):
        if not _is_missing(value):
            token = str(value).strip().lower()
            if token in _SIDE_TOKENS:
                return _SIDE_TOKENS[token]
    return "unknown"


def _hemilineage(ito: Any, truman: Any) -> tuple[str, str | None]:
    """Return ``(input token, split-group value or None)``.

    The first real lineage wins (brain ``itoleeHl`` before VNC ``trumanHl``).
    Sentinels (``putative_primary``, ``TBD``...) stay informative input tokens
    but never tie neurons together in the split.
    """
    tokens = [_clean(v) for v in (ito, truman) if not _is_missing(v)]
    for token in tokens:
        if token not in HEMILINEAGE_SENTINELS:
            return token, token
    return (tokens[0] if tokens else "unknown"), None


def _partner_tier(n_partners: int, edges: Sequence[int]) -> str:
    for index, edge in enumerate(edges):
        if n_partners < edge:
            return f"t{index}"
    return f"t{len(edges)}"


def _input_text(objective: str, features: Mapping[str, str]) -> str:
    parts = [f"dataset {_DATASET_TAG}"]
    parts.extend(f"{key} {features[key]}" for key in INPUT_FEATURES[objective])
    return " ".join(parts)


def assert_no_label_leakage(samples: Sequence[Mapping[str, Any]], objective: str) -> None:
    """Fail closed if any forbidden (label-source) or unexpected key appears in an input_text."""
    forbidden = FORBIDDEN_INPUT_FEATURES[objective]
    allowed = set(INPUT_FEATURES[objective]) | {"dataset"}
    for sample in samples:
        tokens = str(sample["input_text"]).split()
        keys = set(tokens[0::2])
        leaked = sorted(keys & forbidden)
        unknown = sorted(keys - allowed)
        if leaked or unknown:
            raise ValueError(
                f"label leakage guard failed for {sample['sample_id']}: forbidden={leaked} unexpected={unknown}"
            )


def _candidate_meta(meta, config: McSampleBuildConfig):
    """Typed, allowed-status, non-excluded neurons from the body annotations."""
    stats = {}
    typed = meta["type"].map(lambda v: not _is_missing(v))
    stats["dropped_untyped"] = int((~typed).sum())
    frame = meta[typed]
    allowed = {s.strip().lower() for s in config.allowed_statuses}
    status_ok = frame["status"].fillna("").astype(str).str.strip().str.lower().isin(allowed)
    stats["dropped_status"] = int((~status_ok).sum())
    frame = frame[status_ok]
    excluded = {str(v).strip().lower() for v in config.exclude_super_classes}
    non_neuronal = frame["superclass"].fillna("").astype(str).str.strip().str.lower().isin(excluded)
    stats["dropped_non_neuronal"] = int(non_neuronal.sum())
    frame = frame[~non_neuronal].rename(columns={"bodyId": "root_id"})
    return frame.reset_index(drop=True), stats


def _annotate_regions(frame, config: McSampleBuildConfig):
    """Attach the primary-neuropil vocabulary; drop neurons without enough primary-ROI synapses."""
    has_roi = frame["top_roi"].notna()
    enough = frame["total_primary_synapses"] >= int(config.min_primary_synapses)
    stats = {
        "dropped_no_primary_roi": int((~has_roi).sum()),
        "dropped_few_primary_synapses": int((has_roi & ~enough).sum()),
    }
    out = frame[has_roi & enough].copy()
    mapped = [map_mc_primary_roi(roi) for roi in out["top_roi"]]
    out["cns_division"] = [m.division for m in mapped]
    out["subdivision"] = [m.subdivision for m in mapped]
    out["primary_neuropil"] = [m.region_id[len(m.division) + 1:] for m in mapped]
    out["region_id"] = [m.region_id for m in mapped]
    return out, stats


def _rank_regions(frame, *, min_region_samples: int, max_regions: int) -> list[str]:
    counts = frame.groupby("region_id", sort=True)["root_id"].count().to_dict()
    eligible = {region for region, count in counts.items() if int(count) >= int(min_region_samples)}
    if not eligible:
        raise ValueError(
            "No regions satisfy min_region_samples. Lower --min-region-samples or inspect source distribution."
        )
    return sorted(eligible, key=lambda region: (-int(counts[region]), region))[: int(max_regions)]


def _sort_key_frame(frame):
    frame = frame.copy()
    frame["_root_sort"] = frame["root_id"].str.zfill(24)
    frame = frame.sort_values(["region_id", "_root_sort"], kind="mergesort")
    return frame.drop(columns=["_root_sort"])


def _features(row: Any, config: McSampleBuildConfig) -> dict[str, str]:
    hemilineage, _ = _hemilineage(row.itoleeHl, row.trumanHl)
    return {
        "cns_division": _clean(row.cns_division),
        "subdivision": _clean(row.subdivision),
        "primary_neuropil": _clean(row.primary_neuropil),
        "side": _side(row.somaSide, row.rootSide),
        "super_class": _clean(row.superclass),
        "cell_class": _clean(row.cell_class_raw),
        "hemilineage": hemilineage,
        "out_partner_tier": _partner_tier(int(getattr(row, "n_post_partners", 0) or 0), config.partner_tier_edges),
    }


def _group_fields(row: Any) -> dict[str, str | None]:
    _, group = _hemilineage(row.itoleeHl, row.trumanHl)
    return {"cell_type": str(row.type).strip(), "hemilineage": group}


def _division_counts(samples: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        division = str(sample["metadata"]["cns_division"])
        counts[division] = counts.get(division, 0) + 1
    return dict(sorted(counts.items()))


def _load_base(config: McSampleBuildConfig, paths: Mapping[str, Path]):
    check_mc_roi_vocabulary(paths[ROLE_DATASET_META])
    meta = load_mc_annotations(paths[ROLE_META], optional_columns=META_OPTIONAL_COLUMNS)
    candidates, stats = _candidate_meta(meta, config)
    if candidates.empty:
        raise ValueError("No typed mc neurons remain after status / super-class filtering.")
    body_ids = [int(value) for value in candidates["root_id"]]
    roi = load_mc_neuron_roi_summary(paths[ROLE_NEURON_ROI_INFO], body_ids=body_ids)
    frame = candidates.merge(roi, on="root_id", how="left")
    missing_row = frame["n_primary_rois"].isna()
    stats["dropped_missing_neuron_row"] = int(missing_row.sum())
    frame = frame[~missing_row]
    frame["top_roi_synapses"] = frame["top_roi_synapses"].fillna(0).astype("int64")
    frame["total_primary_synapses"] = frame["total_primary_synapses"].fillna(0).astype("int64")
    frame["n_primary_rois"] = frame["n_primary_rois"].fillna(0).astype("int64")
    # itertuples cannot expose the reserved word ``class`` as an attribute.
    frame = frame.rename(columns={"class": "cell_class_raw"})
    return frame, body_ids, stats, int(len(meta))


def _sample(objective: str, row: Any, feats: Mapping[str, str], label: str, confidence: float,
            source_label: str, extra_meta: Mapping[str, Any]) -> dict[str, Any]:
    root_id = str(row.root_id)
    prefix = {"connectivity_tier": "pre", "neurotransmitter_dominance": "nt", "region_specialization_tier": "roi"}
    return {
        "sample_id": f"{_DATASET_TAG}-{prefix[objective]}-{root_id}",
        "region_id": str(row.region_id),
        "input_text": _input_text(objective, feats),
        "expected_label": label,
        "expected_confidence": round(confidence, 6),
        "provenance_refs": [f"{_DATASET_TAG}:body:{root_id}", f"source:{source_label}"],
        "metadata": {
            "dataset": MC_SYMBOL,
            "dataset_version": MC_VERSION_ID,
            "root_id": root_id,
            "cns_division": feats["cns_division"],
            "primary_roi": str(row.top_roi),
            **_group_fields(row),
            **dict(extra_meta),
        },
    }


def _build_connectivity(config: McSampleBuildConfig, paths: Mapping[str, Path], source_label: str):
    frame, body_ids, stats, source_rows = _load_base(config, paths)
    totals, scanned_edges = load_mc_outgoing_totals(paths[ROLE_EDGELIST], body_ids=body_ids)
    frame = frame.merge(totals, on="root_id", how="left")
    frame["total_out_synapses"] = frame["total_out_synapses"].fillna(0).astype("int64")
    frame["n_post_partners"] = frame["n_post_partners"].fillna(0).astype("int64")
    low = frame["total_out_synapses"] < int(config.min_total_count)
    stats["dropped_below_min_total_count"] = int(low.sum())
    frame = frame[~low]
    frame, region_stats = _annotate_regions(frame, config)
    stats.update(region_stats)
    if frame.empty:
        raise ValueError("No mc neurons remain after count / region filtering.")
    threshold = float(frame["total_out_synapses"].quantile(config.high_connectivity_quantile))
    ranked = _rank_regions(frame, min_region_samples=config.min_region_samples, max_regions=config.max_regions)
    selected = _sort_key_frame(frame[frame["region_id"].isin(ranked)])
    samples = []
    for row in selected.itertuples(index=False):
        total = int(row.total_out_synapses)
        label = LABEL_HIGH if float(total) >= threshold else LABEL_BASELINE
        score_delta = abs(float(total) - threshold) / max(float(total), threshold, 1.0)
        confidence = min(0.99, max(0.5, 0.55 + (0.4 * score_delta)))
        feats = _features(row, config)
        samples.append(_sample(OBJECTIVE_CONNECTIVITY_TIER, row, feats, label, confidence, source_label, {
            "task_type": "connectivity",
            "risk_tier": "high" if label == LABEL_HIGH else "medium",
            "label_source": {"total_out_synapses": total, "n_post_partners": int(row.n_post_partners)},
        }))
    stats.update({"source_rows": source_rows, "candidate_rows": int(len(frame)),
                  "high_connectivity_threshold": threshold, "scanned_edges": scanned_edges})
    return samples, stats


def _build_neurotransmitter(config: McSampleBuildConfig, paths: Mapping[str, Path], source_label: str):
    frame, body_ids, stats, source_rows = _load_base(config, paths)
    nt = load_mc_nt_predictions(paths[ROLE_NT_PREDICTION], body_ids=body_ids).rename(columns={"body": "root_id"})
    nt = nt.drop(columns=["cell_type"])
    totals, scanned_edges = load_mc_outgoing_totals(paths[ROLE_EDGELIST], body_ids=body_ids)
    frame = frame.merge(nt, on="root_id", how="left").merge(
        totals[["root_id", "n_post_partners"]], on="root_id", how="left")
    frame["n_post_partners"] = frame["n_post_partners"].fillna(0).astype("int64")
    calls = frame["predicted_nt"].fillna(MC_NT_ABSTAIN).astype(str).str.strip().str.lower()
    unclear = calls == MC_NT_ABSTAIN
    stats["dropped_nt_unclear"] = int(unclear.sum())
    frame = frame[~unclear]
    low = frame["total_nt_predictions"].fillna(0).astype("int64") < int(config.min_total_count)
    stats["dropped_below_min_total_count"] = int(low.sum())
    frame = frame[~low]
    frame, region_stats = _annotate_regions(frame, config)
    stats.update(region_stats)
    if frame.empty:
        raise ValueError("No mc neurons remain after NT / count / region filtering.")
    ranked = _rank_regions(frame, min_region_samples=config.min_region_samples, max_regions=config.max_regions)
    selected = _sort_key_frame(frame[frame["region_id"].isin(ranked)])
    samples = []
    for row in selected.itertuples(index=False):
        code = nt_short_code(str(row.predicted_nt))
        score = float(row.predicted_nt_confidence)
        if not (score == score) or score < 0.0 or score > 1.0:
            raise McAdapterError(McAdapterErrorCode.SCHEMA_MISMATCH, "predicted_nt_confidence outside [0, 1]",
                                 {"root_id": str(row.root_id), "score": score})
        confidence = min(0.99, max(0.5, score))
        label = f"dominant_{code}"
        feats = _features(row, config)
        samples.append(_sample(OBJECTIVE_NEUROTRANSMITTER_DOMINANCE, row, feats, label, confidence, source_label, {
            "dominant_neurotransmitter": label,
            "task_type": "neurotransmitter",
            "risk_tier": "high" if confidence >= 0.85 else "medium",
            "label_source": {
                "predicted_nt_confidence": round(score, 6),
                "total_nt_predictions": int(row.total_nt_predictions),
                "consensus_nt": None if _is_missing(row.consensus_nt) else str(row.consensus_nt),
                "ground_truth": None if _is_missing(row.ground_truth) else str(row.ground_truth),
            },
        }))
    stats.update({"source_rows": source_rows, "candidate_rows": int(len(frame)),
                  "high_connectivity_threshold": None, "scanned_edges": scanned_edges})
    return samples, stats


def _build_region_specialization(config: McSampleBuildConfig, paths: Mapping[str, Path], source_label: str):
    frame, _body_ids, stats, source_rows = _load_base(config, paths)
    stats["dropped_below_min_total_count"] = 0
    frame, region_stats = _annotate_regions(frame, config)
    stats.update(region_stats)
    if frame.empty:
        raise ValueError("No mc neurons remain after region filtering.")
    ranked = _rank_regions(frame, min_region_samples=config.min_region_samples, max_regions=config.max_regions)
    selected = _sort_key_frame(frame[frame["region_id"].isin(ranked)])
    cut = float(config.specialization_share)
    samples = []
    for row in selected.itertuples(index=False):
        share = int(row.top_roi_synapses) / float(int(row.total_primary_synapses))
        label = LABEL_SPECIALIZED if share >= cut else LABEL_DISTRIBUTED
        span = (1.0 - cut) if label == LABEL_SPECIALIZED else cut
        confidence = min(0.99, max(0.5, 0.55 + 0.4 * (abs(share - cut) / max(span, 1e-9))))
        feats = _features(row, config)
        samples.append(_sample(OBJECTIVE_REGION_SPECIALIZATION_TIER, row, feats, label, confidence, source_label, {
            "task_type": "region_specialization",
            "risk_tier": "medium",
            "label_source": {
                "primary_share": round(share, 6),
                "top_roi_synapses": int(row.top_roi_synapses),
                "total_primary_synapses": int(row.total_primary_synapses),
                "n_primary_rois": int(row.n_primary_rois),
            },
        }))
    stats.update({"source_rows": source_rows, "candidate_rows": int(len(frame)),
                  "high_connectivity_threshold": None, "scanned_edges": 0})
    return samples, stats


_BUILDERS = {
    OBJECTIVE_CONNECTIVITY_TIER: (_build_connectivity, ROLE_EDGELIST),
    OBJECTIVE_NEUROTRANSMITTER_DOMINANCE: (_build_neurotransmitter, ROLE_NT_PREDICTION),
    OBJECTIVE_REGION_SPECIALIZATION_TIER: (_build_region_specialization, ROLE_NEURON_ROI_INFO),
}
_FILTER_STAT_KEYS: tuple[str, ...] = (
    "dropped_untyped",
    "dropped_status",
    "dropped_non_neuronal",
    "dropped_missing_neuron_row",
    "dropped_no_primary_roi",
    "dropped_few_primary_synapses",
    "dropped_below_min_total_count",
    "dropped_nt_unclear",
)


def build_mc_training_samples(objective: str, config: McSampleBuildConfig) -> dict[str, Any]:
    if config.objective != objective:
        raise ValueError(f"config.objective {config.objective!r} != requested objective {objective!r}")
    _validate_config(config)
    try:
        import pyarrow  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pyarrow is required to build mc training samples") from exc

    paths, provenance = _resolve_inputs(config)
    build, primary_role = _BUILDERS[objective]
    source_path = paths[primary_role]
    all_samples, stats = build(config, paths, source_path.name)
    assert_no_label_leakage(all_samples, objective)
    # Body ids track size/proofreading order, so pick the capped subset in
    # sha256(body id) order instead of sample_id order (see hash_ordered_preselect).
    preselected = hash_ordered_preselect(all_samples, max_samples=int(config.max_samples), salt=_DATASET_TAG,
                                         id_of=lambda sample: str(sample["metadata"]["root_id"]))
    capped = balance_and_cap_samples(preselected, max_samples=int(config.max_samples))
    fingerprint_inputs = {
        "cap_order": CAP_ORDER,
        "paths": {role: str(path) for role, path in sorted(paths.items())},
        "product_sha256": provenance["product_sha256"],
        "manifest_sha256": provenance["manifest_sha256"],
        "max_samples": config.max_samples,
        "min_total_count": config.min_total_count,
        "min_region_samples": config.min_region_samples,
        "high_connectivity_quantile": config.high_connectivity_quantile,
        "specialization_share": config.specialization_share,
        "min_primary_synapses": config.min_primary_synapses,
        "max_regions": config.max_regions,
        "min_distinct_labels": config.min_distinct_labels,
        "max_label_share": config.max_label_share,
        "allowed_statuses": sorted(config.allowed_statuses),
        "partner_tier_edges": list(config.partner_tier_edges),
        "exclude_super_classes": sorted(config.exclude_super_classes),
        "input_features": list(INPUT_FEATURES[objective]),
    }
    extra = {
        "dataset_version": MC_VERSION_ID,
        "region_vocabulary": "male_cns_roi",
        "region_division_counts": _division_counts(capped),
        "label_definition": LABEL_DEFINITIONS[objective],
        "input_features": list(INPUT_FEATURES[objective]),
        "forbidden_input_features": sorted(FORBIDDEN_INPUT_FEATURES[objective]),
        "split_group_keys": list(SPLIT_GROUP_KEYS),
        "leakage_check": "passed",
        "high_connectivity_threshold": stats["high_connectivity_threshold"],
        "specialization_share": float(config.specialization_share)
        if objective == OBJECTIVE_REGION_SPECIALIZATION_TIER else None,
        "min_total_count": int(config.min_total_count),
        "min_primary_synapses": int(config.min_primary_synapses),
        "min_region_samples": int(config.min_region_samples),
        "max_samples": int(config.max_samples),
        "max_regions": int(config.max_regions),
        "allowed_statuses": sorted(config.allowed_statuses),
        "scanned_edges": int(stats["scanned_edges"]),
        "input_paths": {role: str(path) for role, path in sorted(paths.items())},
        "provenance_mode": provenance["mode"],
        "manifest_id": provenance["manifest_id"],
        "manifest_sha256": provenance["manifest_sha256"],
        "product_sha256": provenance["product_sha256"],
        "hash_verification": provenance["hash_verification"],
        "filter_stats": {key: int(stats.get(key, 0)) for key in _FILTER_STAT_KEYS},
        "license": "CC-BY-4.0",
    }
    return assemble_training_payload(
        capped,
        symbol=MC_SYMBOL,
        objective=objective,
        source_path=str(source_path),
        source_rows=int(stats["source_rows"]),
        candidate_rows=int(stats["candidate_rows"]),
        min_distinct_labels=int(config.min_distinct_labels),
        max_label_share=float(config.max_label_share),
        fingerprint_inputs=fingerprint_inputs,
        extra_metadata=extra,
    )


# ---------------------------------------------------------------------------
# Trivial-baseline audit (reproduces the promotion gate's rules on a grouped split)
# ---------------------------------------------------------------------------


def _lookup_accuracy(train_rows, heldout_rows, keys: Sequence[str]) -> float:
    """Fit ``features[keys] -> majority label`` on train; unseen keys fall back to the train majority."""
    from flybrain_brain_cluster_baselines import parse_input_features

    def key_of(text: str) -> tuple[str, ...]:
        features = parse_input_features(text)
        return tuple(features.get(k, "") for k in keys)

    overall: dict[str, int] = {}
    table: dict[tuple[str, ...], dict[str, int]] = {}
    for text, label in train_rows:
        overall[label] = overall.get(label, 0) + 1
        bucket = table.setdefault(key_of(text), {})
        bucket[label] = bucket.get(label, 0) + 1

    def best(counts: Mapping[str, int]) -> str:
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    fallback = best(overall)
    rules = {key: best(counts) for key, counts in table.items()}
    if not heldout_rows:
        return 0.0
    correct = sum(1 for text, label in heldout_rows if rules.get(key_of(text), fallback) == label)
    return correct / float(len(heldout_rows))


def trivial_baseline_report(payload: Mapping[str, Any], *, split_seed: str = "mc-baseline-audit") -> dict[str, Any]:
    """Grouped split (registry keys + ``split_group``) and trivial-rule accuracies on the held-out split.

    ``gate_rules`` is exactly ``evaluate_trivial_baselines`` (majority /
    numeric stump / score argmax), which the promotion gate uses.
    ``lookup_audit`` adds single-feature and all-feature lookup tables (a
    rule the gate does not model): if any single input feature predicts the
    label on held-out neurons almost perfectly, the objective is trivial.
    """
    import flybrain_brain_cluster_training as fbct
    from flybrain_brain_cluster_baselines import evaluate_trivial_baselines

    samples = [
        fbct.TrainingSample(
            sample_id=str(s["sample_id"]),
            region_id=str(s["region_id"]),
            input_text=str(s["input_text"]),
            expected_label=str(s["expected_label"]),
            expected_confidence=float(s["expected_confidence"]),
            provenance_refs=tuple(s["provenance_refs"]),
            metadata=dict(s["metadata"]),
        )
        for s in payload["samples"]
    ]
    keys = (fbct.GENERIC_SPLIT_GROUP_KEY, *SPLIT_GROUP_KEYS)
    train_ids, val_ids, test_ids, info = fbct.grouped_split_ids(samples, group_keys=keys, split_seed=split_seed)
    by_id = {s.sample_id: (s.input_text, s.expected_label) for s in samples}
    heldout_name, heldout = ("test", test_ids) if test_ids else ("val", val_ids)
    train_rows = [by_id[i] for i in train_ids]
    heldout_rows = [by_id[i] for i in heldout]
    gate = evaluate_trivial_baselines(train_rows, heldout_rows)
    objective = str(payload["metadata"]["objective"])
    features = INPUT_FEATURES[objective]
    lookup = {name: round(_lookup_accuracy(train_rows, heldout_rows, (name,)), 6) for name in features}
    lookup["all_input_features"] = round(_lookup_accuracy(train_rows, heldout_rows, features), 6)
    best_single = max((v, k) for k, v in lookup.items() if k != "all_input_features")
    return {
        "objective": objective,
        "split": info,
        "heldout_split": heldout_name,
        "train_count": len(train_rows),
        "heldout_count": len(heldout_rows),
        "gate_rules": gate,
        "lookup_audit": lookup,
        "best_single_feature_lookup": {"feature": best_single[1], "accuracy": best_single[0]},
    }


# ---------------------------------------------------------------------------
# Structured features (opt-in; the payload builders above are unchanged)
# ---------------------------------------------------------------------------


def attach_structured_features(samples: Sequence[Mapping[str, Any]], feature_frame, *, objective: str,
                               id_column: str = "node_id") -> list[dict[str, Any]]:
    """Return copies of ``samples`` with a ``features`` dict from ``feature_frame`` (joined on the body id).

    ``feature_frame`` is e.g. ``flybrain_wiring_features.build_wiring_features(...).frame``
    or ``flybrain_mc_targets.roi_feature_frame(...)``. Columns the objective's
    registered exclusions (``flybrain_wiring_features``) forbid are dropped
    first, and the survivors are re-checked with ``assert_features_allowed``
    (fail closed). ``input_text`` is left untouched, so existing callers that
    read only ``input_text`` see the same samples. A sample whose body is
    missing from the frame fails closed. See ``flybrain_mc_targets`` for the
    full real-model path (new targets, masking, harness runs).
    """
    import flybrain_wiring_features as fwf

    columns = [c for c in feature_frame.columns if c != id_column]
    dropped = set(fwf.excluded_feature_names(columns, objective))
    kept = [c for c in columns if c not in dropped]
    fwf.assert_features_allowed(kept, objective)
    lookup = feature_frame.set_index(feature_frame[id_column].astype(str))[kept]
    if lookup.index.duplicated().any():
        raise ValueError("feature_frame has duplicate body ids")
    out = []
    for sample in samples:
        root_id = str(sample["metadata"]["root_id"])
        if root_id not in lookup.index:
            raise ValueError(f"no structured features for body {root_id}")
        features = {}
        for name, value in lookup.loc[root_id].items():
            if hasattr(value, "item"):
                value = value.item()
            features[name] = None if isinstance(value, float) and value != value else value
        out.append({**dict(sample), "features": features})
    return out


def register_mc_sample_builders(*, replace: bool = False) -> None:
    register_sample_builder(
        MC_SYMBOL,
        MC_SAMPLE_OBJECTIVES,
        build_mc_training_samples,
        config_type=McSampleBuildConfig,
        replace=replace,
    )


def unregister_mc_sample_builders() -> None:
    for objective in MC_SAMPLE_OBJECTIVES:
        unregister_sample_builder(MC_SYMBOL, objective)


def main(argv: Sequence[str] | None = None) -> int:
    from flybrain_brain_cluster_samples import build_training_samples

    parser = argparse.ArgumentParser(
        description="Build deterministic brain-cluster training samples from local MaleCNS v1.0 (mc).")
    parser.add_argument("--storage-root", help="FlyBrain storage root (defaults to LOCI_FLYBRAIN_STORAGE_ROOT)")
    parser.add_argument("--objective", choices=MC_SAMPLE_OBJECTIVES, default=OBJECTIVE_CONNECTIVITY_TIER)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--min-total-count", type=int, default=10)
    parser.add_argument("--min-region-samples", type=int, default=25)
    parser.add_argument("--high-connectivity-quantile", type=float, default=0.75)
    parser.add_argument("--specialization-share", type=float, default=0.75)
    parser.add_argument("--max-regions", type=int, default=20)
    parser.add_argument("--stamp-dir", help="Hash-stamp directory (default: <snapshot>/manifest/hash-stamps)")
    parser.add_argument("--baseline-report", help="Also write trivial_baseline_report(payload) JSON here")
    parser.add_argument("--allow-planned", action="store_true", help="Required while mc is 'planned' in the registry.")
    args = parser.parse_args(argv)
    config = McSampleBuildConfig(
        objective=args.objective,
        storage_root=args.storage_root,
        stamp_dir=args.stamp_dir,
        max_samples=args.max_samples,
        min_total_count=args.min_total_count,
        min_region_samples=args.min_region_samples,
        high_connectivity_quantile=args.high_connectivity_quantile,
        specialization_share=args.specialization_share,
        max_regions=args.max_regions,
    )
    schema = "flybrain-mc-training-samples/v1"
    try:
        payload = build_training_samples(MC_SYMBOL, args.objective, config, allow_planned=args.allow_planned)
    except (ValueError, RuntimeError, FileNotFoundError, KeyError, TypeError) as exc:
        print(stable_json({"schema_version": schema, "status": "error", "pass": False, "error": str(exc)}))
        return 2
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(stable_json(payload) + "\n", encoding="utf-8")
    meta = payload["metadata"]
    summary = {
        "schema_version": schema,
        "status": "ok",
        "pass": True,
        "output": str(output),
        "sample_count": len(payload["samples"]),
        "regions": len(meta["selected_regions"]),
        "objective": meta["objective"],
        "label_counts": meta["label_counts"],
        "region_division_counts": meta["region_division_counts"],
    }
    if args.baseline_report:
        report = trivial_baseline_report(payload)
        report_path = Path(args.baseline_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(stable_json(report) + "\n", encoding="utf-8")
        summary["baseline_report"] = str(report_path)
        summary["best_trivial_accuracy"] = report["gate_rules"]["best_accuracy"]
        summary["best_single_feature_lookup"] = report["best_single_feature_lookup"]
    print(json.dumps(summary, sort_keys=True))
    return 0


register_mc_sample_builders()


if __name__ == "__main__":
    raise SystemExit(main())
