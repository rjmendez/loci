"""MANC v1.0 (``mv``, male adult ventral nerve cord) brain-cluster training-sample builder.

Registered with ``flybrain_brain_cluster_samples`` for ``connectivity_tier``,
``neurotransmitter_dominance`` and ``region_specialization_tier``. Reads only
the local, manifest-verified snapshot (``flybrain_mv_adapter``) or explicit
local paths; never the network.

Candidates are neurons with neuPrint ``status == "Traced"`` and a cell
``type`` (annotated), excluding glia, with at least one synapse in a VNC
neuropil ROI. ``region_id`` is the neuron's primary neuropil: the neuropil ROI
(side suffix removed) with the largest ``roiInfo`` synweight.

Label definitions (see docs/FLYBRAIN_MV_ADAPTER_CONTRACT.md):

* ``connectivity_tier``: ``high_connectivity`` when the neuron's ``downstream``
  count (output synaptic connections, neuPrint) is at or above the configured
  quantile over all candidates, else ``baseline_connectivity``. Confidence uses
  fw's distance-from-threshold formula.
* ``neurotransmitter_dominance``: ``dominant_<code>`` from the MANC per-neuron
  classifier (``predictedNt``; ach/gaba/glut). ``unknown`` predictions are
  dropped. The adapter checks ``predictedNt`` is the argmax of the probability
  columns. Confidence = clipped ``predictedNtProb``.
* ``region_specialization_tier``: ``region_specialized`` when the primary
  neuropil ROI holds at least ``specialization_share_threshold`` of the neuron's
  neuropil synweight (side-specific ROIs, nerves / connective / tracts
  excluded), else ``region_distributed``.

The model input (``input_text``) never contains the quantity a label is
computed from, nor the annotations that encode it (e.g. MANC intrinsic-neuron
``subclass`` letters, ``target``/``origin`` neuromere lists for region
specialization; ``hemilineage`` for NT, since hemilineage fixes the fast
transmitter). ``assert_no_label_leakage`` enforces the allow-list per
objective, and ``trivial_feature_check`` fails the build when any single input
feature (or the region id, or a numeric threshold stump) predicts the
label above ``max_single_feature_accuracy`` in-sample.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from flybrain_brain_cluster_baselines import fit_threshold_rule
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
from flybrain_mv_adapter import (
    MV_NT_UNKNOWN,
    MV_VERSION_ID,
    ROLE_EDGELIST,
    ROLE_META,
    MvAdapterError,
    MvAdapterErrorCode,
    check_nt_argmax,
    load_mv_neurons,
    load_mv_neuropil_synweights,
    load_mv_out_partner_counts,
    map_mv_roi,
    nt_short_code,
    open_mv_snapshot,
    sha256_file,
)

MV_SYMBOL = "mv"
MV_SAMPLE_OBJECTIVES: tuple[str, ...] = (
    OBJECTIVE_CONNECTIVITY_TIER,
    OBJECTIVE_NEUROTRANSMITTER_DOMINANCE,
    OBJECTIVE_REGION_SPECIALIZATION_TIER,
)
_DATASET_TAG = "manc10"
# How the capped subset is chosen (recorded in the input fingerprint).
CAP_ORDER = "sha256-body-id/v1"
# Bumped whenever input_text rendering changes (part of the input fingerprint).
INPUT_TEXT_FORMAT = "mv-input-text/v1"
_SAMPLE_PREFIX = {
    OBJECTIVE_CONNECTIVITY_TIER: "pre",
    OBJECTIVE_NEUROTRANSMITTER_DOMINANCE: "nt",
    OBJECTIVE_REGION_SPECIALIZATION_TIER: "spec",
}
LABEL_HIGH = "high_connectivity"
LABEL_BASELINE = "baseline_connectivity"
LABEL_SPECIALIZED = "region_specialized"
LABEL_DISTRIBUTED = "region_distributed"

# Model-input feature keys per objective (order = order in input_text).
INPUT_FEATURES: Mapping[str, tuple[str, ...]] = {
    OBJECTIVE_CONNECTIVITY_TIER: (
        "primary_neuropil",
        "soma_neuromere",
        "soma_side",
        "class",
        "birthtime",
        "hemilineage",
    ),
    OBJECTIVE_NEUROTRANSMITTER_DOMINANCE: (
        "primary_neuropil",
        "soma_neuromere",
        "soma_side",
        "class",
        "birthtime",
        "out_partner_tier",
    ),
    OBJECTIVE_REGION_SPECIALIZATION_TIER: (
        "primary_neuropil",
        "soma_neuromere",
        "soma_side",
        "class",
        "birthtime",
    ),
}
# Annotation columns that name the neuron or its homolog group; never inputs.
# ``root`` (the numeric bodyId) is one of them: MANC bodyIds are not random
# (low ids were assigned to large, early-proofread bodies), so a one-feature
# threshold on the id alone predicted connectivity_tier at 0.77 held-out
# accuracy on the real snapshot. Unlike BANC, mv input_text carries no id.
_IDENTITY_FEATURES = frozenset(
    {"root", "body_id", "cell_type", "type", "instance", "systematic_type", "group", "split_group"}
)
# Keys that must never reach input_text for an objective (label sources / proxies).
FORBIDDEN_INPUT_FEATURES: Mapping[str, frozenset[str]] = {
    OBJECTIVE_CONNECTIVITY_TIER: frozenset(
        {
            "downstream",
            "upstream",
            "pre",
            "post",
            "synweight",
            "size",
            "roi_info",
            "neuropil_synweight",
            "n_post_partners",
            "traced_out_weight",
            "out_partner_tier",
            "input_rois",
            "output_rois",
        }
    )
    | _IDENTITY_FEATURES,
    OBJECTIVE_NEUROTRANSMITTER_DOMINANCE: frozenset(
        {
            "predicted_nt",
            "predicted_nt_prob",
            "nt_acetylcholine_prob",
            "nt_gaba_prob",
            "nt_glutamate_prob",
            "nt_unknown_prob",
            "transmission",
            # Hemilineage fixes the fast transmitter (one NT per hemilineage);
            # it stays a split-group key in metadata, never an input.
            "hemilineage",
        }
    )
    | _IDENTITY_FEATURES,
    OBJECTIVE_REGION_SPECIALIZATION_TIER: frozenset(
        {
            "roi_info",
            "input_rois",
            "output_rois",
            "neuropil_synweight",
            "top_neuropil_share",
            "n_neuropils",
            # MANC annotations that encode how many neuromeres/neuropils a
            # neuron spans: intrinsic subclass letters (IR/II/BR/BI/CR/CI...),
            # the systematic prefix, target/origin neuromere lists, long tracts
            # and entry/exit nerves.
            "subclass",
            "prefix",
            "target",
            "origin",
            "long_tract",
            "entry_nerve",
            "exit_nerve",
            "serial_motif",
            # Stereotyped per-hemilineage projection patterns; split key only.
            "hemilineage",
        }
    )
    | _IDENTITY_FEATURES,
}
for _objective, _features in INPUT_FEATURES.items():
    _overlap = set(_features) & FORBIDDEN_INPUT_FEATURES[_objective]
    if _overlap:  # pragma: no cover - module invariant
        raise AssertionError(f"{_objective}: input features overlap forbidden set: {sorted(_overlap)}")

# neuPrint soma/root side tokens -> side vocabulary.
_SIDE_MAP: Mapping[str, str] = {
    "lhs": "left",
    "rhs": "right",
    "midline": "midline",
    "mid": "midline",
    "bil": "bilateral",
}
# Hemilineage placeholders that must not tie neurons together or act as a value.
_UNKNOWN_HEMILINEAGE = frozenset({"tbd", "none", "unknown", "na", "nan", ""})


@dataclass(frozen=True)
class MvSampleBuildConfig:
    objective: str = OBJECTIVE_CONNECTIVITY_TIER
    storage_root: str | Path | None = None
    # Explicit local overrides. When all needed paths are given the manifest is not consulted.
    meta_path: str | Path | None = None
    edgelist_path: str | Path | None = None
    # None: stamp-cached verification of the required products (see
    # flybrain_hash_stamps); True: full rehash; False: size checks only.
    verify_hashes: bool | None = None
    # False: never read or write hash stamps (strictly read-only snapshot open).
    use_stamp_cache: bool = True
    max_samples: int = 5000
    # connectivity: min downstream; NT: min presynapse count (pre); region: min neuropil synweight.
    min_total_count: int = 10
    min_region_samples: int = 25
    high_connectivity_quantile: float = 0.75
    specialization_share_threshold: float = 0.9
    max_regions: int = 20
    min_distinct_labels: int = 2
    max_label_share: float = 0.9
    statuses: tuple[str, ...] = ("Traced",)
    require_type: bool = True
    exclude_classes: tuple[str, ...] = ("glia",)
    partner_tier_edges: tuple[int, ...] = (10, 100)
    # Fail the build if one input feature alone predicts the label this well (in-sample).
    max_single_feature_accuracy: float = 0.9


def _validate_config(config: MvSampleBuildConfig) -> None:
    if config.objective not in MV_SAMPLE_OBJECTIVES:
        raise ValueError(f"objective must be one of: {', '.join(MV_SAMPLE_OBJECTIVES)}")
    if config.max_samples <= 0:
        raise ValueError("max_samples must be > 0")
    if config.min_total_count < 1:
        raise ValueError("min_total_count must be >= 1")
    if config.min_region_samples < 1:
        raise ValueError("min_region_samples must be >= 1")
    if not (0.0 < config.high_connectivity_quantile < 1.0):
        raise ValueError("high_connectivity_quantile must be in (0, 1)")
    if not (0.0 < config.specialization_share_threshold < 1.0):
        raise ValueError("specialization_share_threshold must be in (0, 1)")
    if config.max_regions < 1:
        raise ValueError("max_regions must be >= 1")
    if config.min_distinct_labels < 1:
        raise ValueError("min_distinct_labels must be >= 1")
    if not (0.0 < config.max_label_share <= 1.0):
        raise ValueError("max_label_share must be in (0, 1]")
    if not (0.0 < config.max_single_feature_accuracy <= 1.0):
        raise ValueError("max_single_feature_accuracy must be in (0, 1]")
    if not config.statuses:
        raise ValueError("statuses must not be empty")
    edges = tuple(int(e) for e in config.partner_tier_edges)
    if not edges or list(edges) != sorted(set(edges)) or edges[0] < 1:
        raise ValueError("partner_tier_edges must be strictly increasing positive integers")


def _required_roles(objective: str) -> tuple[str, ...]:
    # Only the NT objective uses the edge list (out-partner tier input feature).
    if objective == OBJECTIVE_NEUROTRANSMITTER_DOMINANCE:
        return (ROLE_META, ROLE_EDGELIST)
    return (ROLE_META,)


def _resolve_inputs(config: MvSampleBuildConfig) -> tuple[dict[str, Path], dict[str, Any]]:
    roles = _required_roles(config.objective)
    explicit = {ROLE_META: config.meta_path, ROLE_EDGELIST: config.edgelist_path}
    if all(explicit[role] is not None for role in roles):
        paths = {role: Path(explicit[role]).resolve(strict=False) for role in roles}
        for role, path in paths.items():
            if not path.is_file():
                raise MvAdapterError(MvAdapterErrorCode.PRODUCT_MISSING, f"mv {role} file not found",
                                     {"path": str(path)})
        return paths, {"mode": "explicit_paths", "manifest_id": None, "manifest_sha256": None,
                       "product_sha256": {role: sha256_file(p) for role, p in sorted(paths.items())},
                       "hash_verification": {}}
    if any(explicit[role] is not None for role in roles):
        raise ValueError("explicit mv paths must be given for every required product or none: " + ", ".join(roles))
    snapshot = open_mv_snapshot(config.storage_root, required_roles=roles, verify_hashes=config.verify_hashes,
                                use_stamp_cache=config.use_stamp_cache)
    prov = snapshot.provenance()
    return {role: snapshot.path(role) for role in roles}, {
        "mode": "manifest_snapshot",
        "manifest_id": prov["manifest_id"],
        "manifest_sha256": prov["manifest_sha256"],
        "product_sha256": prov["product_sha256"],
        "hash_verification": dict(snapshot.hash_verification),
    }


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "unknown"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>", "null"}:
        return "unknown"
    return "_".join(text.lower().split())


def _side(soma_side: Any, root_side: Any) -> str:
    for raw in (soma_side, root_side):
        token = _clean(raw)
        if token != "unknown":
            return _SIDE_MAP.get(token, token)
    return "unknown"


def _hemilineage(value: Any) -> str:
    token = _clean(value)
    return "unknown" if token in _UNKNOWN_HEMILINEAGE else token


def _split_group(value: Any) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "unknown"
    try:
        return f"manc_group_{int(value)}"
    except (TypeError, ValueError):
        return "unknown"


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
        if len(tokens) % 2:
            raise ValueError(f"label leakage guard failed for {sample['sample_id']}: odd token count")
        keys = set(tokens[0::2])
        leaked = sorted(keys & forbidden)
        unknown = sorted(keys - allowed)
        if leaked or unknown:
            raise ValueError(
                f"label leakage guard failed for {sample['sample_id']}: forbidden={leaked} unexpected={unknown}"
            )


def _lookup_accuracy(values: Sequence[str], labels: Sequence[str]) -> float:
    by_value: dict[str, dict[str, int]] = {}
    for value, label in zip(values, labels):
        counts = by_value.setdefault(value, {})
        counts[label] = counts.get(label, 0) + 1
    correct = sum(max(counts.values()) for counts in by_value.values())
    return correct / float(len(labels)) if labels else 0.0


def trivial_feature_check(
    samples: Sequence[Mapping[str, Any]], objective: str, *, max_accuracy: float
) -> dict[str, Any]:
    """In-sample single-feature predictability of the label (an optimistic upper bound).

    For every input feature, and for ``region_id`` (the expert routing key), the
    accuracy of predicting each value's majority label; plus the best one-feature
    numeric threshold stump over ``input_text`` (``fit_threshold_rule``; mv
    inputs are categorical, so it normally finds no numeric feature). Raises ``ValueError`` when any of them
    exceeds ``max_accuracy``: such a label would be (nearly) derivable from one
    input and the model could not add anything a lookup table cannot.
    """
    labels = [str(sample["expected_label"]) for sample in samples]
    if not labels:
        raise ValueError("trivial_feature_check needs samples")
    majority = _lookup_accuracy(["*"] * len(labels), labels)
    parsed = []
    for sample in samples:
        tokens = str(sample["input_text"]).split()
        parsed.append(dict(zip(tokens[0::2], tokens[1::2])))
    lookup: dict[str, float] = {}
    for key in INPUT_FEATURES[objective]:
        lookup[key] = round(_lookup_accuracy([row.get(key, "") for row in parsed], labels), 6)
    lookup["region_id"] = round(_lookup_accuracy([str(s["region_id"]) for s in samples], labels), 6)
    stump = fit_threshold_rule([(str(s["input_text"]), str(s["expected_label"])) for s in samples])
    stump_summary = None
    if stump is not None:
        stump_summary = {"feature": stump["feature"], "train_accuracy": round(float(stump["train_accuracy"]), 6)}
    best_feature, best_accuracy = sorted(lookup.items(), key=lambda item: (-item[1], item[0]))[0]
    if stump_summary is not None and stump_summary["train_accuracy"] > best_accuracy:
        best_feature, best_accuracy = f"threshold:{stump_summary['feature']}", stump_summary["train_accuracy"]
    report = {
        "sample_count": len(labels),
        "majority_accuracy": round(majority, 6),
        "single_feature_lookup_accuracy": dict(sorted(lookup.items())),
        "threshold_stump": stump_summary,
        "best_single_feature": best_feature,
        "best_single_feature_accuracy": round(float(best_accuracy), 6),
        "max_allowed_accuracy": float(max_accuracy),
    }
    if best_accuracy > float(max_accuracy):
        raise ValueError(
            f"label is trivially derivable from one input: {best_feature} predicts {objective} with in-sample "
            f"accuracy {best_accuracy:.3f} > {max_accuracy:.3f}"
        )
    return report


def _load_candidates(config: MvSampleBuildConfig, meta_path: Path):
    """Traced (typed, non-excluded-class) neurons with a primary neuropil."""
    import numpy as np

    frame = load_mv_neurons(meta_path, statuses=config.statuses)
    source_rows = int(len(frame))
    stats = {"dropped_untyped": 0, "dropped_excluded_class": 0, "dropped_no_neuropil_synapses": 0}
    if config.require_type:
        typed = frame["type"].map(lambda v: _clean(v) != "unknown")
        stats["dropped_untyped"] = int((~typed).sum())
        frame = frame[typed]
    excluded = {_clean(value) for value in config.exclude_classes}
    class_tokens = frame["class"].map(_clean)
    drop_class = class_tokens.isin(excluded)
    stats["dropped_excluded_class"] = int(drop_class.sum())
    frame = frame[~drop_class].reset_index(drop=True)
    weights = load_mv_neuropil_synweights(meta_path, frame["bodyId"].tolist())
    matrix = weights.weights
    totals = matrix.sum(axis=1) if matrix.size else np.zeros(len(frame), dtype="int64")
    has_neuropil = totals > 0
    stats["dropped_no_neuropil_synapses"] = int((~has_neuropil).sum())
    top_index = matrix.argmax(axis=1) if matrix.size else np.zeros(len(frame), dtype="int64")
    top_weight = matrix.max(axis=1) if matrix.size else np.zeros(len(frame), dtype="int64")
    rois = [map_mv_roi(name) for name in weights.neuropil_rois]
    frame = frame.assign(
        neuropil_synweight=totals.astype("int64"),
        top_neuropil_roi=[weights.neuropil_rois[i] for i in top_index],
        top_neuropil_share=np.where(totals > 0, top_weight / np.maximum(totals, 1), 0.0),
        n_neuropils=(matrix > 0).sum(axis=1).astype("int64") if matrix.size else 0,
        region_id=[rois[i].region_id for i in top_index],
    )
    frame = frame[has_neuropil].reset_index(drop=True)
    return frame, source_rows, stats


def _rank_regions(frame, *, min_region_samples: int, max_regions: int) -> list[str]:
    counts = frame.groupby("region_id", sort=True)["bodyId"].count().to_dict()
    eligible = {region for region, count in counts.items() if int(count) >= int(min_region_samples)}
    if not eligible:
        raise ValueError(
            "No regions satisfy min_region_samples. Lower --min-region-samples or inspect source distribution."
        )
    return sorted(eligible, key=lambda region: (-int(counts[region]), region))[: int(max_regions)]


def _sorted_selection(frame, ranked: Sequence[str]):
    selected = frame[frame["region_id"].isin(list(ranked))].copy()
    selected["_sort"] = selected["bodyId"].str.zfill(20)
    selected = selected.sort_values(["region_id", "_sort"], kind="mergesort")
    return selected.drop(columns=["_sort"])


def _features(row: Any, config: MvSampleBuildConfig) -> dict[str, str]:
    return {
        "primary_neuropil": str(row.region_id)[len("vnc_"):] if str(row.region_id).startswith("vnc_")
        else str(row.region_id),
        "soma_neuromere": _clean(row.somaNeuromere),
        "soma_side": _side(row.somaSide, row.rootSide),
        "class": _clean(row.class_),
        "birthtime": _clean(row.birthtime),
        "hemilineage": _hemilineage(row.hemilineage),
        "out_partner_tier": _partner_tier(int(getattr(row, "n_post_partners", 0) or 0), config.partner_tier_edges),
    }


def _base_metadata(row: Any, feats: Mapping[str, str], task_type: str) -> dict[str, Any]:
    return {
        "dataset": MV_SYMBOL,
        "dataset_version": MV_VERSION_ID,
        "root_id": str(row.bodyId),
        "cns_division": "nerve_cord",
        # Split-group keys (registry split_group_keys + the generic split_group):
        # never part of input_text.
        "cell_type": _clean(row.type) if _clean(row.type) == "unknown" else str(row.type).strip(),
        "hemilineage": _hemilineage(row.hemilineage),
        "split_group": _split_group(row.group),
        "primary_neuropil_roi": str(row.top_neuropil_roi),
        "task_type": task_type,
    }


def _confidence_from_threshold(value: float, threshold: float) -> float:
    delta = abs(float(value) - threshold) / max(float(value), threshold, 1e-9)
    return min(0.99, max(0.5, 0.55 + (0.4 * delta)))


def _build_connectivity(config: MvSampleBuildConfig, frame):
    frame = frame[frame["downstream"].fillna(0).astype("int64") >= int(config.min_total_count)]
    if frame.empty:
        raise ValueError("No rows remain after min_total_count filtering on downstream.")
    threshold = float(frame["downstream"].astype("float64").quantile(config.high_connectivity_quantile))
    ranked = _rank_regions(frame, min_region_samples=config.min_region_samples, max_regions=config.max_regions)
    samples = []
    for row in _sorted_selection(frame, ranked).itertuples(index=False):
        total = int(row.downstream)
        label = LABEL_HIGH if float(total) >= threshold else LABEL_BASELINE
        feats = _features(row, config)
        body = str(row.bodyId)
        metadata = _base_metadata(row, feats, "connectivity")
        metadata.update({
            "risk_tier": "high" if label == LABEL_HIGH else "medium",
            "label_source": {"downstream": total, "pre": int(row.pre)},
        })
        samples.append(_sample(OBJECTIVE_CONNECTIVITY_TIER, body, row, feats, label,
                               _confidence_from_threshold(total, threshold), metadata))
    return samples, {"candidate_rows": int(len(frame)), "high_connectivity_threshold": threshold}


def _build_neurotransmitter(config: MvSampleBuildConfig, frame, edgelist_path: Path):
    partners, edge_rows = load_mv_out_partner_counts(edgelist_path)
    frame = frame.merge(partners[["bodyId", "n_post_partners"]], on="bodyId", how="left")
    frame["n_post_partners"] = frame["n_post_partners"].fillna(0).astype("int64")
    frame = frame[frame["predictedNt"].notna()]
    check_nt_argmax(frame)
    unknown = frame["predictedNt"].astype(str).str.strip().str.lower() == MV_NT_UNKNOWN
    dropped_unknown = int(unknown.sum())
    frame = frame[~unknown]
    frame = frame[frame["pre"].fillna(0).astype("int64") >= int(config.min_total_count)]
    if frame.empty:
        raise ValueError("No rows remain after NT / min_total_count (presynapse) filtering.")
    ranked = _rank_regions(frame, min_region_samples=config.min_region_samples, max_regions=config.max_regions)
    samples = []
    for row in _sorted_selection(frame, ranked).itertuples(index=False):
        code = nt_short_code(str(row.predictedNt))
        score = float(row.predictedNtProb)
        if not (score == score) or score < 0.0 or score > 1.0:
            raise MvAdapterError(MvAdapterErrorCode.SCHEMA_MISMATCH, "predictedNtProb outside [0, 1]",
                                 {"bodyId": str(row.bodyId), "score": score})
        label = f"dominant_{code}"
        feats = _features(row, config)
        body = str(row.bodyId)
        confidence = min(0.99, max(0.5, score))
        metadata = _base_metadata(row, feats, "neurotransmitter")
        metadata.update({
            "dominant_neurotransmitter": label,
            "risk_tier": "high" if confidence >= 0.85 else "medium",
            "label_source": {"predicted_nt_prob": round(score, 6), "pre": int(row.pre)},
        })
        samples.append(_sample(OBJECTIVE_NEUROTRANSMITTER_DOMINANCE, body, row, feats, label, confidence, metadata))
    return samples, {"candidate_rows": int(len(frame)), "dropped_nt_unknown": dropped_unknown,
                     "edge_rows": int(edge_rows)}


def _build_region_specialization(config: MvSampleBuildConfig, frame):
    frame = frame[frame["neuropil_synweight"] >= int(config.min_total_count)]
    if frame.empty:
        raise ValueError("No rows remain after min_total_count filtering on neuropil synweight.")
    threshold = float(config.specialization_share_threshold)
    ranked = _rank_regions(frame, min_region_samples=config.min_region_samples, max_regions=config.max_regions)
    samples = []
    for row in _sorted_selection(frame, ranked).itertuples(index=False):
        share = float(row.top_neuropil_share)
        label = LABEL_SPECIALIZED if share >= threshold else LABEL_DISTRIBUTED
        feats = _features(row, config)
        body = str(row.bodyId)
        metadata = _base_metadata(row, feats, "region_specialization")
        metadata.update({
            "risk_tier": "medium",
            "label_source": {"top_neuropil_share": round(share, 6),
                             "neuropil_synweight": int(row.neuropil_synweight),
                             "n_neuropils": int(row.n_neuropils)},
        })
        samples.append(_sample(OBJECTIVE_REGION_SPECIALIZATION_TIER, body, row, feats, label,
                               _confidence_from_threshold(share, threshold), metadata))
    return samples, {"candidate_rows": int(len(frame)), "specialization_share_threshold": threshold}


def _sample(objective: str, body: str, row: Any, feats: Mapping[str, str], label: str, confidence: float,
            metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": f"{_DATASET_TAG}-{_SAMPLE_PREFIX[objective]}-{body}",
        "region_id": str(row.region_id),
        "input_text": _input_text(objective, feats),
        "expected_label": label,
        "expected_confidence": round(float(confidence), 6),
        "provenance_refs": [f"{_DATASET_TAG}:body_id:{body}", "source:manc-v1.0-neuron-properties.feather"],
        "metadata": dict(metadata),
    }


def build_mv_training_samples(objective: str, config: MvSampleBuildConfig) -> dict[str, Any]:
    if config.objective != objective:
        raise ValueError(f"config.objective {config.objective!r} != requested objective {objective!r}")
    _validate_config(config)
    try:
        import pyarrow  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pyarrow is required to build mv training samples") from exc

    paths, provenance = _resolve_inputs(config)
    meta_path = paths[ROLE_META]
    frame, source_rows, stats = _load_candidates(config, meta_path)
    frame = frame.rename(columns={"class": "class_"})
    if frame.empty:
        raise ValueError("No mv neurons remain after status / type / class / neuropil filtering.")
    extra_stats: dict[str, Any]
    if objective == OBJECTIVE_NEUROTRANSMITTER_DOMINANCE:
        all_samples, extra_stats = _build_neurotransmitter(config, frame, paths[ROLE_EDGELIST])
    elif objective == OBJECTIVE_REGION_SPECIALIZATION_TIER:
        all_samples, extra_stats = _build_region_specialization(config, frame)
    else:
        all_samples, extra_stats = _build_connectivity(config, frame)
    assert_no_label_leakage(all_samples, objective)
    # MANC bodyIds track size/proofreading order, so pick the capped subset in
    # sha256(bodyId) order instead of sample_id order (see hash_ordered_preselect).
    preselected = hash_ordered_preselect(all_samples, max_samples=int(config.max_samples), salt=_DATASET_TAG,
                                         id_of=lambda sample: str(sample["metadata"]["root_id"]))
    capped = balance_and_cap_samples(preselected, max_samples=int(config.max_samples))
    trivial = trivial_feature_check(capped, objective, max_accuracy=float(config.max_single_feature_accuracy))
    fingerprint_inputs = {
        "cap_order": CAP_ORDER,
        "paths": {role: str(path) for role, path in sorted(paths.items())},
        "product_sha256": provenance["product_sha256"],
        "manifest_sha256": provenance["manifest_sha256"],
        "max_samples": config.max_samples,
        "min_total_count": config.min_total_count,
        "min_region_samples": config.min_region_samples,
        "high_connectivity_quantile": config.high_connectivity_quantile,
        "specialization_share_threshold": config.specialization_share_threshold,
        "max_regions": config.max_regions,
        "min_distinct_labels": config.min_distinct_labels,
        "max_label_share": config.max_label_share,
        "statuses": sorted(config.statuses),
        "require_type": config.require_type,
        "exclude_classes": sorted(config.exclude_classes),
        "partner_tier_edges": list(config.partner_tier_edges),
        "max_single_feature_accuracy": config.max_single_feature_accuracy,
        "input_features": list(INPUT_FEATURES[objective]),
        "input_text_format": INPUT_TEXT_FORMAT,
    }
    filter_stats = {
        "dropped_untyped": int(stats["dropped_untyped"]),
        "dropped_excluded_class": int(stats["dropped_excluded_class"]),
        "dropped_no_neuropil_synapses": int(stats["dropped_no_neuropil_synapses"]),
        "dropped_nt_unknown": int(extra_stats.get("dropped_nt_unknown", 0)),
    }
    extra = {
        "dataset_version": MV_VERSION_ID,
        "region_vocabulary": "manc_neuropil",
        "region_division_counts": {"nerve_cord": len(capped)},
        "input_features": list(INPUT_FEATURES[objective]),
        "forbidden_input_features": sorted(FORBIDDEN_INPUT_FEATURES[objective]),
        "leakage_check": "passed",
        "trivial_feature_check": trivial,
        "high_connectivity_threshold": extra_stats.get("high_connectivity_threshold"),
        "specialization_share_threshold": extra_stats.get("specialization_share_threshold"),
        "min_total_count": int(config.min_total_count),
        "min_region_samples": int(config.min_region_samples),
        "max_samples": int(config.max_samples),
        "max_regions": int(config.max_regions),
        "statuses": sorted(config.statuses),
        "require_type": bool(config.require_type),
        "input_paths": {role: str(path) for role, path in sorted(paths.items())},
        "provenance_mode": provenance["mode"],
        "manifest_id": provenance["manifest_id"],
        "manifest_sha256": provenance["manifest_sha256"],
        "product_sha256": provenance["product_sha256"],
        "hash_verification": provenance["hash_verification"],
        "filter_stats": filter_stats,
        "license": "CC-BY-4.0",
    }
    return assemble_training_payload(
        capped,
        symbol=MV_SYMBOL,
        objective=objective,
        source_path=str(meta_path),
        source_rows=int(source_rows),
        candidate_rows=int(extra_stats["candidate_rows"]),
        min_distinct_labels=int(config.min_distinct_labels),
        max_label_share=float(config.max_label_share),
        fingerprint_inputs=fingerprint_inputs,
        extra_metadata=extra,
    )


def register_mv_sample_builders(*, replace: bool = False) -> None:
    register_sample_builder(
        MV_SYMBOL,
        MV_SAMPLE_OBJECTIVES,
        build_mv_training_samples,
        config_type=MvSampleBuildConfig,
        replace=replace,
    )


def unregister_mv_sample_builders() -> None:
    for objective in MV_SAMPLE_OBJECTIVES:
        unregister_sample_builder(MV_SYMBOL, objective)


def main(argv: Sequence[str] | None = None) -> int:
    from flybrain_brain_cluster_samples import build_training_samples

    parser = argparse.ArgumentParser(
        description="Build deterministic brain-cluster training samples from local MANC v1.0 (mv)."
    )
    parser.add_argument("--storage-root", help="FlyBrain storage root (defaults to LOCI_FLYBRAIN_STORAGE_ROOT)")
    parser.add_argument("--objective", choices=MV_SAMPLE_OBJECTIVES, default=OBJECTIVE_CONNECTIVITY_TIER)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--min-total-count", type=int, default=10)
    parser.add_argument("--min-region-samples", type=int, default=25)
    parser.add_argument("--high-connectivity-quantile", type=float, default=0.75)
    parser.add_argument("--specialization-share-threshold", type=float, default=0.9)
    parser.add_argument("--max-regions", type=int, default=20)
    parser.add_argument("--max-label-share", type=float, default=0.9)
    parser.add_argument("--max-single-feature-accuracy", type=float, default=0.9,
                        help="Fail when one input feature alone predicts the label above this (in-sample).")
    parser.add_argument("--no-stamp-cache", action="store_true",
                        help="Do not read or write hash stamps (strictly read-only snapshot open).")
    parser.add_argument("--allow-planned", action="store_true", help="Required while mv is 'planned' in the registry.")
    args = parser.parse_args(argv)
    config = MvSampleBuildConfig(
        objective=args.objective,
        storage_root=args.storage_root,
        max_samples=args.max_samples,
        min_total_count=args.min_total_count,
        min_region_samples=args.min_region_samples,
        high_connectivity_quantile=args.high_connectivity_quantile,
        specialization_share_threshold=args.specialization_share_threshold,
        max_regions=args.max_regions,
        max_label_share=args.max_label_share,
        max_single_feature_accuracy=args.max_single_feature_accuracy,
        use_stamp_cache=not args.no_stamp_cache,
    )
    schema = "flybrain-mv-training-samples/v1"
    try:
        payload = build_training_samples(MV_SYMBOL, args.objective, config, allow_planned=args.allow_planned)
    except (ValueError, RuntimeError, FileNotFoundError, KeyError, TypeError) as exc:
        print(stable_json({"schema_version": schema, "status": "error", "pass": False, "error": str(exc)}))
        return 2
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(stable_json(payload) + "\n", encoding="utf-8")
    meta = payload["metadata"]
    print(json.dumps({
        "schema_version": schema,
        "status": "ok",
        "pass": True,
        "output": str(output),
        "sample_count": len(payload["samples"]),
        "regions": len(meta["selected_regions"]),
        "objective": meta["objective"],
        "label_counts": meta["label_counts"],
        "trivial_feature_check": meta["trivial_feature_check"],
    }, sort_keys=True))
    return 0


register_mv_sample_builders()


if __name__ == "__main__":
    raise SystemExit(main())
