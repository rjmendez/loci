"""FlyEM optic-lobe v1.1 brain-cluster training-sample builder.

Registered with ``flybrain_brain_cluster_samples`` for ``connectivity_tier``
and ``neurotransmitter_dominance``. Reads only the local, manifest-verified
snapshot (``flybrain_ol_adapter``) or explicit local paths; never the network.

Candidates are the typed neurons of the neuPrint ``Neuron`` table (non-empty
``type``) with an allowed tracing status (default ``Traced``). Labels:

* ``connectivity_tier``: ``high_connectivity`` when the neuron's neuPrint
  ``downstream`` count (output connections summed over all partners) is at or
  above the configured global quantile over retained candidates, else
  ``baseline_connectivity``. Confidence uses fw's distance-from-threshold formula.
* ``neurotransmitter_dominance``: ``dominant_<code>`` from the neuron's own
  synapse-classifier call ``predictedNt`` (``unclear`` calls are dropped),
  confidence = clipped ``predictedNtConfidence``.

``region_specialization_tier`` is not built: the only region signal in this
release is ``roiInfo``, which already defines ``region_id`` and the arbor input
features, so any specialization label would be a function of the inputs.

``input_text`` holds only anatomy/annotation features (dominant neuropils,
arbor neuropil set, dominant OL layer, side, soma, hex-column assignment,
hemilineage; NT adds polarity and column-span tiers). It never holds the label
source, the cell type (split-group key; it would let the model memorise the
type -> label map), the instance name (contains the type), or the body id
(bodyId order tracks size in this release: Spearman -0.65 with ``downstream``).
``assert_no_label_leakage`` enforces the allow-list per objective.
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
)
from flybrain_ol_adapter import (
    DEFAULT_ALLOWED_STATUSES,
    OL_NT_UNCLEAR,
    OL_VERSION_ID,
    ROLE_NEUPRINT_META,
    ROLE_NEURONS,
    OlAdapterError,
    OlAdapterErrorCode,
    layer_slug,
    load_ol_neurons,
    load_ol_release_meta,
    map_ol_roi,
    nt_short_code,
    open_ol_snapshot,
    parse_roi_info,
    sha256_file,
)

OL_SYMBOL = "ol"
OL_SAMPLE_OBJECTIVES: tuple[str, ...] = (OBJECTIVE_CONNECTIVITY_TIER, OBJECTIVE_NEUROTRANSMITTER_DOMINANCE)
_DATASET_TAG = "ol11"
_REQUIRED_ROLES: tuple[str, ...] = (ROLE_NEURONS, ROLE_NEUPRINT_META)

_COMMON_FEATURES: tuple[str, ...] = (
    "side",
    "soma",
    "hex_column",
    "primary_neuropil",
    "input_neuropil",
    "output_neuropil",
    "arbor",
    "dominant_layer",
    "hemilineage",
)
# Model-input feature keys per objective (order = order in input_text).
INPUT_FEATURES: Mapping[str, tuple[str, ...]] = {
    OBJECTIVE_CONNECTIVITY_TIER: _COMMON_FEATURES,
    OBJECTIVE_NEUROTRANSMITTER_DOMINANCE: (*_COMMON_FEATURES, "polarity_tier", "column_span_tier"),
}
_ALWAYS_FORBIDDEN = frozenset(
    {"body", "body_id", "root", "cell_type", "type", "instance", "flywire_type", "mcns_serial"}
)
# Keys that must never reach input_text for an objective (label sources / proxies).
FORBIDDEN_INPUT_FEATURES: Mapping[str, frozenset[str]] = {
    OBJECTIVE_CONNECTIVITY_TIER: _ALWAYS_FORBIDDEN
    | frozenset(
        {
            "downstream",
            "upstream",
            "pre",
            "post",
            "synweight",
            "size",
            "total_nt_predictions",
            "polarity_tier",
            "column_span_tier",
            "n_columns",
        }
    ),
    OBJECTIVE_NEUROTRANSMITTER_DOMINANCE: _ALWAYS_FORBIDDEN
    | frozenset(
        {
            "predicted_nt",
            "predicted_nt_confidence",
            "total_nt_predictions",
            "celltype_predicted_nt",
            "celltype_predicted_nt_confidence",
            "celltype_total_nt_predictions",
            "consensus_nt",
            "nt_reference",
            "other_nt",
            "other_nt_reference",
        }
    ),
}


@dataclass(frozen=True)
class OlSampleBuildConfig:
    objective: str = OBJECTIVE_CONNECTIVITY_TIER
    storage_root: str | Path | None = None
    # Explicit local overrides. When both are given the manifest is not consulted.
    neurons_path: str | Path | None = None
    meta_path: str | Path | None = None
    # None: stamp-cached verification of the required products (see
    # flybrain_hash_stamps); True: full rehash; False: size checks only.
    verify_hashes: bool | None = None
    max_samples: int = 5000
    min_total_count: int = 10
    min_region_samples: int = 25
    high_connectivity_quantile: float = 0.75
    max_regions: int = 20
    min_distinct_labels: int = 2
    max_label_share: float = 0.9
    allowed_statuses: tuple[str, ...] = DEFAULT_ALLOWED_STATUSES
    arbor_min_share: float = 0.1
    polarity_tier_edges: tuple[float, ...] = (0.1, 0.3)
    column_span_tier_edges: tuple[int, ...] = (2, 10)


def _validate_config(config: OlSampleBuildConfig) -> None:
    if config.objective not in OL_SAMPLE_OBJECTIVES:
        raise ValueError(f"objective must be one of: {', '.join(OL_SAMPLE_OBJECTIVES)}")
    if config.max_samples <= 0:
        raise ValueError("max_samples must be > 0")
    if config.min_total_count < 1:
        raise ValueError("min_total_count must be >= 1")
    if config.min_region_samples < 1:
        raise ValueError("min_region_samples must be >= 1")
    if not (0.0 < config.high_connectivity_quantile < 1.0):
        raise ValueError("high_connectivity_quantile must be in (0, 1)")
    if config.max_regions < 1:
        raise ValueError("max_regions must be >= 1")
    if config.min_distinct_labels < 1:
        raise ValueError("min_distinct_labels must be >= 1")
    if not (0.0 < config.max_label_share <= 1.0):
        raise ValueError("max_label_share must be in (0, 1]")
    if not config.allowed_statuses or any(not str(s).strip() for s in config.allowed_statuses):
        raise ValueError("allowed_statuses must be non-empty strings")
    if not (0.0 < config.arbor_min_share <= 1.0):
        raise ValueError("arbor_min_share must be in (0, 1]")
    polarity = tuple(float(e) for e in config.polarity_tier_edges)
    if not polarity or list(polarity) != sorted(set(polarity)) or polarity[0] <= 0.0 or polarity[-1] >= 1.0:
        raise ValueError("polarity_tier_edges must be strictly increasing values in (0, 1)")
    columns = tuple(int(e) for e in config.column_span_tier_edges)
    if not columns or list(columns) != sorted(set(columns)) or columns[0] < 1:
        raise ValueError("column_span_tier_edges must be strictly increasing positive integers")


def _resolve_inputs(config: OlSampleBuildConfig) -> tuple[dict[str, Path], dict[str, Any]]:
    explicit = {ROLE_NEURONS: config.neurons_path, ROLE_NEUPRINT_META: config.meta_path}
    if all(explicit[role] is not None for role in _REQUIRED_ROLES):
        paths = {role: Path(explicit[role]).resolve(strict=False) for role in _REQUIRED_ROLES}
        for role, path in paths.items():
            if not path.is_file():
                raise OlAdapterError(OlAdapterErrorCode.PRODUCT_MISSING, f"ol {role} file not found",
                                     {"path": str(path)})
        return paths, {"mode": "explicit_paths", "manifest_id": None, "manifest_sha256": None,
                       "product_sha256": {role: sha256_file(p) for role, p in sorted(paths.items())}}
    if any(explicit[role] is not None for role in _REQUIRED_ROLES):
        raise ValueError("explicit ol paths must be given for every required product or none: "
                         + ", ".join(_REQUIRED_ROLES))
    snapshot = open_ol_snapshot(config.storage_root, required_roles=_REQUIRED_ROLES,
                                verify_hashes=config.verify_hashes)
    prov = snapshot.provenance()
    return {role: snapshot.path(role) for role in _REQUIRED_ROLES}, {
        "mode": "manifest_snapshot",
        "manifest_id": prov["manifest_id"],
        "manifest_sha256": prov["manifest_sha256"],
        "product_sha256": prov["product_sha256"],
    }


def _missing(value: Any) -> bool:
    if value is None or (isinstance(value, float) and value != value):
        return True
    text = str(value).strip()
    return not text or text.lower() in {"nan", "none", "<na>"}


def _clean(value: Any) -> str:
    if _missing(value):
        return "unknown"
    return "_".join(str(value).strip().lower().split())


def _tier(value: float, edges: Sequence[float]) -> str:
    for index, edge in enumerate(edges):
        if value < edge:
            return f"t{index}"
    return f"t{len(edges)}"


def _side(instance: Any) -> str:
    text = "" if _missing(instance) else str(instance).strip()
    if text.endswith("_R"):
        return "right"
    if text.endswith("_L"):
        return "left"
    return "unknown"


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


def _int(value: Any, *, column: str, body_id: str) -> int:
    if _missing(value):
        return 0
    number = float(value)
    if number < 0 or number != int(number):
        raise OlAdapterError(OlAdapterErrorCode.SCHEMA_MISMATCH, f"{column} is not a non-negative integer",
                             {"body_id": body_id, "value": repr(value)})
    return int(number)


def _annotate(frame, config: OlSampleBuildConfig) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Parse roiInfo once per neuron; drop neurons with no primary-ROI synapses."""
    rows: list[dict[str, Any]] = []
    dropped_no_primary_roi = 0
    for row in frame.itertuples(index=False):
        body_id = str(row.body_id)
        summary = parse_roi_info(row.roi_info)
        dominant = summary.dominant("synweight")
        if dominant is None:
            dropped_no_primary_roi += 1
            continue
        region = map_ol_roi(dominant)
        pre = _int(row.pre, column="pre", body_id=body_id)
        post = _int(row.post, column="post", body_id=body_id)
        dom_in = summary.dominant("post")
        dom_out = summary.dominant("pre")
        layer = summary.dominant_layer()
        arbor = summary.arbor(config.arbor_min_share)
        features = {
            "side": _side(row.instance),
            "soma": "no" if _missing(row.soma_location) else "yes",
            "hex_column": "no" if _missing(row.assigned_ol_hex1) else "yes",
            "primary_neuropil": region.slug,
            "input_neuropil": map_ol_roi(dom_in).slug if dom_in else "none",
            "output_neuropil": map_ol_roi(dom_out).slug if dom_out else "none",
            "arbor": "+".join(map_ol_roi(roi).slug for roi in arbor) or "none",
            "dominant_layer": layer_slug(layer) if layer else "none",
            "hemilineage": _clean(row.hemilineage),
            "polarity_tier": _tier(pre / float(pre + post), config.polarity_tier_edges) if pre + post else "none",
            "column_span_tier": _tier(float(summary.n_columns), [float(e) for e in config.column_span_tier_edges]),
        }
        rows.append(
            {
                "body_id": body_id,
                "cell_type": str(row.cell_type).strip(),
                "region_id": region.region_id,
                "division": region.division,
                "dominant_roi": region.roi,
                "features": features,
                "row": row,
            }
        )
    return rows, {"dropped_no_primary_roi": dropped_no_primary_roi}


def _rank_regions(rows: Sequence[Mapping[str, Any]], *, min_region_samples: int, max_regions: int) -> list[str]:
    counts: dict[str, int] = {}
    for item in rows:
        counts[item["region_id"]] = counts.get(item["region_id"], 0) + 1
    eligible = {region for region, count in counts.items() if count >= int(min_region_samples)}
    if not eligible:
        raise ValueError(
            "No regions satisfy min_region_samples. Lower --min-region-samples or inspect source distribution."
        )
    return sorted(eligible, key=lambda region: (-counts[region], region))[: int(max_regions)]


def _select(rows: Sequence[Mapping[str, Any]], config: OlSampleBuildConfig) -> list[Mapping[str, Any]]:
    ranked = set(_rank_regions(rows, min_region_samples=config.min_region_samples, max_regions=config.max_regions))
    chosen = [item for item in rows if item["region_id"] in ranked]
    return sorted(chosen, key=lambda item: (item["region_id"], item["body_id"].zfill(24)))


def _hash_round_robin(samples: Sequence[dict[str, Any]], *, max_samples: int) -> list[dict[str, Any]]:
    """Cap-sized subset in sha256(body_id) order per region (shared ``hash_ordered_preselect``).

    In optic-lobe v1.1 low body ids are the large, early-traced bodies
    (bodyId vs ``downstream`` Spearman -0.65), so the shared cap helper's
    sample_id order would bias a capped build toward high-connectivity neurons.
    """
    return hash_ordered_preselect(samples, max_samples=max_samples, salt=_DATASET_TAG,
                                  id_of=lambda sample: str(sample["metadata"]["body_id"]))


def _division_counts(samples: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        division = str(sample["metadata"]["division"])
        counts[division] = counts.get(division, 0) + 1
    return dict(sorted(counts.items()))


def _sample(objective: str, item: Mapping[str, Any], *, kind: str, label: str, confidence: float,
            risk_tier: str, source_label: str, label_source: Mapping[str, Any]) -> dict[str, Any]:
    body_id = item["body_id"]
    return {
        "sample_id": f"{_DATASET_TAG}-{kind}-{body_id}",
        "region_id": item["region_id"],
        "input_text": _input_text(objective, item["features"]),
        "expected_label": label,
        "expected_confidence": round(confidence, 6),
        "provenance_refs": [f"{_DATASET_TAG}:body_id:{body_id}", f"source:{source_label}"],
        "metadata": {
            "dataset": OL_SYMBOL,
            "dataset_version": OL_VERSION_ID,
            "body_id": body_id,
            # Split-group key (registry split_group_keys for ol); never in input_text.
            "cell_type": item["cell_type"],
            "hemilineage": item["features"]["hemilineage"],
            "division": item["division"],
            "dominant_roi": item["dominant_roi"],
            "task_type": "connectivity" if objective == OBJECTIVE_CONNECTIVITY_TIER else "neurotransmitter",
            "risk_tier": risk_tier,
            "label_source": dict(label_source),
        },
    }


def _build_connectivity(config: OlSampleBuildConfig, frame, source_label: str):
    downstream = frame["downstream"].fillna(0).astype("int64")
    kept = frame[downstream >= int(config.min_total_count)]
    dropped_below = int(len(frame) - len(kept))
    if kept.empty:
        raise ValueError("No rows remain after min_total_count filtering.")
    rows, stats = _annotate(kept, config)
    if not rows:
        raise ValueError("No ol neurons remain after ROI parsing.")
    totals = sorted(_int(item["row"].downstream, column="downstream", body_id=item["body_id"]) for item in rows)
    import numpy as np

    threshold = float(np.quantile(np.asarray(totals, dtype="float64"), config.high_connectivity_quantile))
    samples = []
    for item in _select(rows, config):
        total = _int(item["row"].downstream, column="downstream", body_id=item["body_id"])
        label = "high_connectivity" if float(total) >= threshold else "baseline_connectivity"
        score_delta = abs(float(total) - threshold) / max(float(total), threshold, 1.0)
        confidence = min(0.99, max(0.5, 0.55 + (0.4 * score_delta)))
        samples.append(_sample(
            OBJECTIVE_CONNECTIVITY_TIER, item, kind="out", label=label, confidence=confidence,
            risk_tier="high" if label == "high_connectivity" else "medium", source_label=source_label,
            label_source={"downstream": total, "pre": _int(item["row"].pre, column="pre", body_id=item["body_id"])},
        ))
    stats.update({"candidate_rows": len(rows), "high_connectivity_threshold": threshold,
                  "dropped_below_min_total": dropped_below, "dropped_nt_unclear": 0, "dropped_nt_missing": 0})
    return samples, stats


def _build_neurotransmitter(config: OlSampleBuildConfig, frame, source_label: str):
    missing = frame["predicted_nt"].isna() | (frame["predicted_nt"].astype(str).str.strip() == "")
    unclear = ~missing & (frame["predicted_nt"].astype(str).str.strip().str.lower() == OL_NT_UNCLEAR)
    dropped_missing = int(missing.sum())
    dropped_unclear = int(unclear.sum())
    frame = frame[~missing & ~unclear]
    enough = frame["total_nt_predictions"].fillna(0).astype("float64") >= float(config.min_total_count)
    dropped_below = int((~enough).sum())
    frame = frame[enough]
    if frame.empty:
        raise ValueError("No rows remain after min_total_count filtering on classified presynapse count.")
    rows, stats = _annotate(frame, config)
    if not rows:
        raise ValueError("No ol neurons remain after ROI parsing.")
    samples = []
    for item in _select(rows, config):
        row = item["row"]
        code = nt_short_code(str(row.predicted_nt))
        score = float(row.predicted_nt_confidence) if not _missing(row.predicted_nt_confidence) else float("nan")
        if not (score == score) or score < 0.0 or score > 1.0:
            raise OlAdapterError(OlAdapterErrorCode.SCHEMA_MISMATCH, "predictedNtConfidence outside [0, 1]",
                                 {"body_id": item["body_id"], "score": score})
        confidence = min(0.99, max(0.5, score))
        label = f"dominant_{code}"
        samples.append(_sample(
            OBJECTIVE_NEUROTRANSMITTER_DOMINANCE, item, kind="nt", label=label, confidence=confidence,
            risk_tier="high" if confidence >= 0.85 else "medium", source_label=source_label,
            label_source={"predicted_nt_confidence": round(score, 6),
                          "total_nt_predictions": int(float(row.total_nt_predictions))},
        ))
    stats.update({"candidate_rows": len(rows), "high_connectivity_threshold": None,
                  "dropped_below_min_total": dropped_below, "dropped_nt_unclear": dropped_unclear,
                  "dropped_nt_missing": dropped_missing})
    return samples, stats


def build_ol_training_samples(objective: str, config: OlSampleBuildConfig) -> dict[str, Any]:
    if config.objective != objective:
        raise ValueError(f"config.objective {config.objective!r} != requested objective {objective!r}")
    _validate_config(config)
    try:
        import pyarrow  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pyarrow is required to build ol training samples") from exc

    paths, provenance = _resolve_inputs(config)
    release = load_ol_release_meta(paths[ROLE_NEUPRINT_META])
    frame, source_rows, typed_rows = load_ol_neurons(paths[ROLE_NEURONS], statuses=config.allowed_statuses)
    source_path = paths[ROLE_NEURONS]
    if objective == OBJECTIVE_NEUROTRANSMITTER_DOMINANCE:
        all_samples, stats = _build_neurotransmitter(config, frame, source_path.name)
    else:
        all_samples, stats = _build_connectivity(config, frame, source_path.name)
    assert_no_label_leakage(all_samples, objective)
    preselected = _hash_round_robin(all_samples, max_samples=int(config.max_samples))
    capped = balance_and_cap_samples(preselected, max_samples=int(config.max_samples))
    fingerprint_inputs = {
        "paths": {role: str(path) for role, path in sorted(paths.items())},
        "product_sha256": provenance["product_sha256"],
        "manifest_sha256": provenance["manifest_sha256"],
        "max_samples": config.max_samples,
        "min_total_count": config.min_total_count,
        "min_region_samples": config.min_region_samples,
        "high_connectivity_quantile": config.high_connectivity_quantile,
        "max_regions": config.max_regions,
        "min_distinct_labels": config.min_distinct_labels,
        "max_label_share": config.max_label_share,
        "allowed_statuses": sorted(config.allowed_statuses),
        "arbor_min_share": config.arbor_min_share,
        "polarity_tier_edges": list(config.polarity_tier_edges),
        "column_span_tier_edges": list(config.column_span_tier_edges),
        "input_features": list(INPUT_FEATURES[objective]),
    }
    extra = {
        "dataset_version": OL_VERSION_ID,
        "neuprint_release": f"{release.dataset}:{release.tag}",
        "region_vocabulary": "optic_lobe_neuropil",
        "region_division_counts": _division_counts(capped),
        "input_features": list(INPUT_FEATURES[objective]),
        "forbidden_input_features": sorted(FORBIDDEN_INPUT_FEATURES[objective]),
        "leakage_check": "passed",
        "high_connectivity_threshold": stats["high_connectivity_threshold"],
        "min_total_count": int(config.min_total_count),
        "min_region_samples": int(config.min_region_samples),
        "max_samples": int(config.max_samples),
        "max_regions": int(config.max_regions),
        "allowed_statuses": sorted(config.allowed_statuses),
        "input_paths": {role: str(path) for role, path in sorted(paths.items())},
        "provenance_mode": provenance["mode"],
        "manifest_id": provenance["manifest_id"],
        "manifest_sha256": provenance["manifest_sha256"],
        "product_sha256": provenance["product_sha256"],
        "typed_rows": int(typed_rows),
        "filter_stats": {
            "dropped_status": int(typed_rows - len(frame)),
            "dropped_below_min_total": int(stats["dropped_below_min_total"]),
            "dropped_no_primary_roi": int(stats["dropped_no_primary_roi"]),
            "dropped_nt_unclear": int(stats["dropped_nt_unclear"]),
            "dropped_nt_missing": int(stats["dropped_nt_missing"]),
        },
        "license": "CC-BY-4.0",
    }
    return assemble_training_payload(
        capped,
        symbol=OL_SYMBOL,
        objective=objective,
        source_path=str(source_path),
        source_rows=int(source_rows),
        candidate_rows=int(stats["candidate_rows"]),
        min_distinct_labels=int(config.min_distinct_labels),
        max_label_share=float(config.max_label_share),
        fingerprint_inputs=fingerprint_inputs,
        extra_metadata=extra,
    )


# ---------------------------------------------------------------------------
# Label-derivability audit (grouped split, held-out)
# ---------------------------------------------------------------------------


def label_derivability_audit(
    samples: Sequence[Mapping[str, Any]],
    *,
    split_seed: str = "ol-derivability-audit",
    group_keys: Sequence[str] = ("cell_type",),
) -> dict[str, Any]:
    """Held-out accuracy of trivial rules on a grouped (cell-type) split.

    * ``gate``: ``flybrain_brain_cluster_baselines.evaluate_trivial_baselines``
      (majority / numeric threshold / score argmax) - the rules the promotion
      gate uses.
    * ``single_feature``: for each input feature, a lookup table value ->
      majority train label (unseen values fall back to the global majority).
    * ``all_features``: the same lookup keyed on the full input_text minus
      the dataset tag (an upper bound for memorising feature combinations).

    Train = grouped train split; held-out = val + test. Pure, deterministic.
    """
    import flybrain_brain_cluster_baselines as fbb
    from flybrain_brain_cluster_training import TrainingSample, grouped_split_ids

    typed = [
        TrainingSample(
            sample_id=str(s["sample_id"]),
            region_id=str(s["region_id"]),
            input_text=str(s["input_text"]),
            expected_label=str(s["expected_label"]),
            expected_confidence=float(s["expected_confidence"]),
            provenance_refs=tuple(s["provenance_refs"]),
            metadata=dict(s["metadata"]),
        )
        for s in samples
    ]
    train_ids, val_ids, test_ids, split_info = grouped_split_ids(typed, group_keys=group_keys, split_seed=split_seed)
    by_id = {s.sample_id: s for s in typed}
    train = [(by_id[i].input_text, by_id[i].expected_label) for i in train_ids]
    heldout = [(by_id[i].input_text, by_id[i].expected_label) for i in (*val_ids, *test_ids)]
    if not train or not heldout:
        raise ValueError("audit needs non-empty train and held-out splits")
    gate = fbb.evaluate_trivial_baselines(train, heldout)

    def _majority(labels: Sequence[str]) -> str:
        counts: dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    global_majority = _majority([label for _, label in train])

    def _lookup_accuracy(key_fn) -> float:
        table: dict[str, list[str]] = {}
        for text, label in train:
            table.setdefault(key_fn(text), []).append(label)
        rule = {key: _majority(labels) for key, labels in table.items()}
        correct = sum(1 for text, label in heldout if rule.get(key_fn(text), global_majority) == label)
        return correct / float(len(heldout))

    feature_keys = [key for key in fbb.parse_input_features(train[0][0]) if key != "dataset"]
    single = {
        key: round(_lookup_accuracy(lambda text, k=key: fbb.parse_input_features(text).get(k, "")), 4)
        for key in feature_keys
    }
    combo = _lookup_accuracy(lambda text: text)
    best_feature = sorted(single.items(), key=lambda kv: (-kv[1], kv[0]))[0] if single else (None, None)
    return {
        "split": {"group_keys": list(group_keys), "train": len(train), "heldout": len(heldout),
                  "components": split_info["component_count"]},
        "gate_best_rule": gate["best_rule"],
        "gate_best_accuracy": round(gate["best_accuracy"], 4),
        "gate_rules": {name: row["heldout_accuracy"] for name, row in gate["rules"].items()},
        "single_feature": single,
        "best_single_feature": {"feature": best_feature[0], "accuracy": best_feature[1]},
        "all_features_lookup": round(combo, 4),
    }


def register_ol_sample_builders(*, replace: bool = False) -> None:
    register_sample_builder(
        OL_SYMBOL,
        OL_SAMPLE_OBJECTIVES,
        build_ol_training_samples,
        config_type=OlSampleBuildConfig,
        replace=replace,
    )


def unregister_ol_sample_builders() -> None:
    for objective in OL_SAMPLE_OBJECTIVES:
        unregister_sample_builder(OL_SYMBOL, objective)


def main(argv: Sequence[str] | None = None) -> int:
    from flybrain_brain_cluster_samples import build_training_samples

    parser = argparse.ArgumentParser(
        description="Build deterministic brain-cluster training samples from local FlyEM optic-lobe v1.1."
    )
    parser.add_argument("--storage-root", help="FlyBrain storage root (defaults to LOCI_FLYBRAIN_STORAGE_ROOT)")
    parser.add_argument("--objective", choices=OL_SAMPLE_OBJECTIVES, default=OBJECTIVE_CONNECTIVITY_TIER)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--min-total-count", type=int, default=10)
    parser.add_argument("--min-region-samples", type=int, default=25)
    parser.add_argument("--high-connectivity-quantile", type=float, default=0.75)
    parser.add_argument("--max-regions", type=int, default=20)
    parser.add_argument("--audit", action="store_true", help="Also print the grouped-split derivability audit.")
    parser.add_argument("--allow-planned", action="store_true", help="Required while ol is 'planned' in the registry.")
    args = parser.parse_args(argv)
    config = OlSampleBuildConfig(
        objective=args.objective,
        storage_root=args.storage_root,
        max_samples=args.max_samples,
        min_total_count=args.min_total_count,
        min_region_samples=args.min_region_samples,
        high_connectivity_quantile=args.high_connectivity_quantile,
        max_regions=args.max_regions,
    )
    schema = "flybrain-ol-training-samples/v1"
    try:
        payload = build_training_samples(OL_SYMBOL, args.objective, config, allow_planned=args.allow_planned)
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
    if args.audit:
        summary["derivability_audit"] = label_derivability_audit(payload["samples"])
    print(json.dumps(summary, sort_keys=True))
    return 0


register_ol_sample_builders()


if __name__ == "__main__":
    raise SystemExit(main())
