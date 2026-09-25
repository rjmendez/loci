"""BANC (brain + nerve cord, v888) brain-cluster training-sample builder.

Registered with ``flybrain_brain_cluster_samples`` for ``connectivity_tier``
and ``neurotransmitter_dominance``. Reads only the local, manifest-verified
snapshot (``flybrain_banc_adapter``) or explicit local paths; never the network.

Label semantics follow fw where the data supports it:

* ``connectivity_tier``: ``high_connectivity`` when a neuron's total outgoing
  v3 synapse count (edgelist ``count`` summed per ``pre``) is at or above the
  configured global quantile, else ``baseline_connectivity``. Confidence uses
  fw's distance-from-threshold formula.
* ``neurotransmitter_dominance``: ``dominant_<code>`` from the per-neuron
  classifier argmax (``neurotransmitter_predicted``), confidence = clipped
  ``neurotransmitter_score``. Codes: ach/gaba/glut/da/ser/oct (as fw) plus
  his/tyr (BANC-only classes).

Unlike fw, the model input (``input_text``) never contains the quantity the
label is computed from; ``assert_no_label_leakage`` enforces this per objective.
Regions come from the explicit brain/nerve-cord map in ``map_banc_region``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from flybrain_banc_adapter import (
    BANC_VERSION_ID,
    ROLE_EDGELIST_V3,
    ROLE_META,
    ROLE_NT_PREDICTION,
    BancAdapterError,
    BancAdapterErrorCode,
    curated_region_division,
    load_banc_meta,
    load_banc_nt_predictions,
    load_banc_outgoing_totals,
    map_banc_region,
    nt_short_code,
    open_banc_snapshot,
    sha256_file,
)
from flybrain_brain_cluster_samples import (
    assemble_training_payload,
    balance_and_cap_samples,
    register_sample_builder,
    stable_json,
    unregister_sample_builder,
)
from flybrain_dataset_registry import (
    OBJECTIVE_CONNECTIVITY_TIER,
    OBJECTIVE_NEUROTRANSMITTER_DOMINANCE,
)

BANC_SYMBOL = "banc"
BANC_SAMPLE_OBJECTIVES: tuple[str, ...] = (OBJECTIVE_CONNECTIVITY_TIER, OBJECTIVE_NEUROTRANSMITTER_DOMINANCE)
_DATASET_TAG = "banc888"

# Model-input feature keys per objective (order = order in input_text).
INPUT_FEATURES: Mapping[str, tuple[str, ...]] = {
    OBJECTIVE_CONNECTIVITY_TIER: (
        "cns_division",
        "subdivision",
        "root_neuropil",
        "side",
        "super_class",
        "cell_class",
        "flow",
        "hemilineage",
    ),
    OBJECTIVE_NEUROTRANSMITTER_DOMINANCE: (
        "cns_division",
        "subdivision",
        "root_neuropil",
        "side",
        "super_class",
        "cell_class",
        "flow",
        "hemilineage",
        "out_partner_tier",
    ),
}
# Keys that must never reach input_text for an objective (label sources / proxies).
FORBIDDEN_INPUT_FEATURES: Mapping[str, frozenset[str]] = {
    OBJECTIVE_CONNECTIVITY_TIER: frozenset(
        {
            "total_out_synapses",
            "pre_count",
            "count",
            "norm",
            "n_post_partners",
            "output_connections",
            "input_connections",
            "out_partner_tier",
            "cell_type",
        }
    ),
    OBJECTIVE_NEUROTRANSMITTER_DOMINANCE: frozenset(
        {
            "acetylcholine",
            "dopamine",
            "gaba",
            "glutamate",
            "histamine",
            "octopamine",
            "serotonin",
            "tyramine",
            "neurotransmitter_predicted",
            "neurotransmitter_score",
            "cell_type_neurotransmitter_predicted",
            "cell_type_neurotransmitter_score",
            "neurotransmitter_verified",
            "known_nt",
            "top_nt",
            "cell_type",
        }
    ),
}


# Meta columns carried into sample metadata (never into input_text) so the
# training split can keep related neurons together (registry split_group_keys).
SPLIT_GROUP_META_COLUMNS: tuple[str, ...] = ("cell_type",)


def _split_group_fields(row: Any) -> dict[str, str]:
    return {"cell_type": _clean(getattr(row, "cell_type", None)), "hemilineage": _clean(row.hemilineage)}


@dataclass(frozen=True)
class BancSampleBuildConfig:
    objective: str = OBJECTIVE_CONNECTIVITY_TIER
    storage_root: str | Path | None = None
    # Explicit local overrides. When all needed paths are given the manifest is not consulted.
    edgelist_path: str | Path | None = None
    nt_path: str | Path | None = None
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
    proofread_only: bool = True
    partner_tier_edges: tuple[int, ...] = (10, 100)
    exclude_super_classes: tuple[str, ...] = ("glia", "not_a_neuron", "trachea")


def _validate_config(config: BancSampleBuildConfig) -> None:
    if config.objective not in BANC_SAMPLE_OBJECTIVES:
        raise ValueError(f"objective must be one of: {', '.join(BANC_SAMPLE_OBJECTIVES)}")
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
    edges = tuple(int(e) for e in config.partner_tier_edges)
    if not edges or list(edges) != sorted(set(edges)) or edges[0] < 1:
        raise ValueError("partner_tier_edges must be strictly increasing positive integers")


def _required_roles(objective: str) -> tuple[str, ...]:
    # Both objectives use the edgelist: connectivity for the label, NT for the
    # out-partner tier input feature.
    if objective == OBJECTIVE_NEUROTRANSMITTER_DOMINANCE:
        return (ROLE_EDGELIST_V3, ROLE_META, ROLE_NT_PREDICTION)
    return (ROLE_EDGELIST_V3, ROLE_META)


def _resolve_inputs(config: BancSampleBuildConfig) -> tuple[dict[str, Path], dict[str, Any]]:
    roles = _required_roles(config.objective)
    explicit = {
        ROLE_EDGELIST_V3: config.edgelist_path,
        ROLE_META: config.meta_path,
        ROLE_NT_PREDICTION: config.nt_path,
    }
    if all(explicit[role] is not None for role in roles):
        paths = {role: Path(explicit[role]).resolve(strict=False) for role in roles}
        for role, path in paths.items():
            if not path.is_file():
                raise BancAdapterError(BancAdapterErrorCode.PRODUCT_MISSING, f"BANC {role} file not found",
                                       {"path": str(path)})
        return paths, {"mode": "explicit_paths", "manifest_id": None, "manifest_sha256": None,
                       "product_sha256": {role: sha256_file(p) for role, p in sorted(paths.items())}}
    if any(explicit[role] is not None for role in roles):
        raise ValueError("explicit BANC paths must be given for every required product or none: "
                         + ", ".join(roles))
    snapshot = open_banc_snapshot(config.storage_root, required_roles=roles, verify_hashes=config.verify_hashes)
    prov = snapshot.provenance()
    return {role: snapshot.path(role) for role in roles}, {
        "mode": "manifest_snapshot",
        "manifest_id": prov["manifest_id"],
        "manifest_sha256": prov["manifest_sha256"],
        "product_sha256": prov["product_sha256"],
    }


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "unknown"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return "unknown"
    return "_".join(text.lower().split())


def _partner_tier(n_partners: int, edges: Sequence[int]) -> str:
    for index, edge in enumerate(edges):
        if n_partners < edge:
            return f"t{index}"
    return f"t{len(edges)}"


def _input_text(objective: str, root_id: str, features: Mapping[str, str]) -> str:
    parts = [f"dataset {_DATASET_TAG}", f"root {root_id}"]
    parts.extend(f"{key} {features[key]}" for key in INPUT_FEATURES[objective])
    return " ".join(parts)


def assert_no_label_leakage(samples: Sequence[Mapping[str, Any]], objective: str) -> None:
    """Fail closed if any forbidden (label-source) key appears in an input_text."""
    forbidden = FORBIDDEN_INPUT_FEATURES[objective]
    allowed = set(INPUT_FEATURES[objective]) | {"dataset", "root"}
    for sample in samples:
        tokens = str(sample["input_text"]).split()
        keys = set(tokens[0::2])
        leaked = sorted(keys & forbidden)
        unknown = sorted(keys - allowed)
        if leaked or unknown:
            raise ValueError(
                f"label leakage guard failed for {sample['sample_id']}: forbidden={leaked} unexpected={unknown}"
            )


def _annotate_regions(frame, id_column: str) -> tuple[Any, dict[str, int]]:
    """Attach division/neuropil/region_id from root_region; drop missing + conflicting rows."""
    stats = {"dropped_missing_root_region": 0, "dropped_division_conflict": 0}
    records = []
    for row in frame.itertuples(index=False):
        root_region = getattr(row, "root_region")
        if root_region is None or (isinstance(root_region, float) and root_region != root_region) or not str(root_region).strip():
            stats["dropped_missing_root_region"] += 1
            records.append(None)
            continue
        mapped = map_banc_region(str(root_region))
        curated = curated_region_division(getattr(row, "region"))
        if curated is not None and curated != (mapped.division, mapped.subdivision):
            stats["dropped_division_conflict"] += 1
            records.append(None)
            continue
        records.append(mapped)
    keep = [mapped is not None for mapped in records]
    out = frame.loc[keep].copy()
    kept = [mapped for mapped in records if mapped is not None]
    out["cns_division"] = [m.division for m in kept]
    out["subdivision"] = [m.subdivision for m in kept]
    out["root_neuropil"] = [m.neuropil for m in kept]
    out["region_id"] = [m.region_id for m in kept]
    return out, stats


def _rank_regions(frame, *, min_region_samples: int, max_regions: int) -> list[str]:
    counts = frame.groupby("region_id", sort=True)["root_id"].count().to_dict()
    eligible = {region for region, count in counts.items() if int(count) >= int(min_region_samples)}
    if not eligible:
        raise ValueError(
            "No regions satisfy min_region_samples. Lower --min-region-samples or inspect source distribution."
        )
    return sorted(eligible, key=lambda region: (-int(counts[region]), region))[: int(max_regions)]


def _features(row: Any, config: BancSampleBuildConfig) -> dict[str, str]:
    return {
        "cns_division": _clean(row.cns_division),
        "subdivision": _clean(row.subdivision),
        "root_neuropil": _clean(row.root_neuropil),
        "side": _clean(row.side),
        "super_class": _clean(row.super_class),
        "cell_class": _clean(row.cell_class),
        "flow": _clean(row.flow),
        "hemilineage": _clean(row.hemilineage),
        "out_partner_tier": _partner_tier(int(getattr(row, "n_post_partners", 0) or 0), config.partner_tier_edges),
    }


_TRUE_TOKENS = frozenset({"true", "t", "1", "yes", "y"})
_FALSE_TOKENS = frozenset({"false", "f", "0", "no", "n", ""})


def _proofread_mask(values) -> list[bool]:
    """Parse the meta ``proofread`` column explicitly.

    The published banc_888_meta.feather stores it as the strings ``"TRUE"`` /
    ``"FALSE"``; ``astype(bool)`` would turn ``"FALSE"`` into ``True`` and make
    ``proofread_only`` a silent no-op. Unknown tokens fail closed.
    """
    out: list[bool] = []
    for value in values:
        if value is None or (isinstance(value, float) and value != value):
            out.append(False)
            continue
        if isinstance(value, bool) or type(value).__name__ == "bool_":
            out.append(bool(value))
            continue
        token = str(value).strip().lower()
        if token in _TRUE_TOKENS:
            out.append(True)
        elif token in _FALSE_TOKENS:
            out.append(False)
        else:
            raise BancAdapterError(BancAdapterErrorCode.SCHEMA_MISMATCH, "meta proofread value is not boolean",
                                   {"value": str(value)})
    return out


def _join_meta(frame, meta, config: BancSampleBuildConfig):
    joined = frame.merge(meta.rename(columns={"banc_888_id": "root_id"}), on="root_id", how="inner")
    unmatched = int(len(frame) - len(joined))
    excluded = {str(value).strip().lower() for value in config.exclude_super_classes}
    non_neuronal = joined["super_class"].fillna("").astype(str).str.strip().str.lower().isin(excluded)
    dropped_non_neuronal = int(non_neuronal.sum())
    joined = joined[~non_neuronal]
    if config.proofread_only:
        before = len(joined)
        joined = joined[_proofread_mask(joined["proofread"].tolist())]
        dropped_unproofread = int(before - len(joined))
    else:
        dropped_unproofread = 0
    return joined, {
        "dropped_meta_unmatched": unmatched,
        "dropped_non_neuronal": dropped_non_neuronal,
        "dropped_unproofread": dropped_unproofread,
    }


def _sort_key_frame(frame):
    frame = frame.copy()
    frame["_root_sort"] = frame["root_id"].str.zfill(24)
    frame = frame.sort_values(["region_id", "_root_sort"], kind="mergesort")
    return frame.drop(columns=["_root_sort"])


def _division_counts(samples: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        division = str(sample["metadata"]["cns_division"])
        counts[division] = counts.get(division, 0) + 1
    return dict(sorted(counts.items()))


def _build_connectivity(config: BancSampleBuildConfig, paths: Mapping[str, Path], source_label: str):
    totals, edge_rows = load_banc_outgoing_totals(paths[ROLE_EDGELIST_V3])
    meta = load_banc_meta(paths[ROLE_META], optional_columns=SPLIT_GROUP_META_COLUMNS)
    frame = totals[totals["total_out_synapses"] >= int(config.min_total_count)]
    if frame.empty:
        raise ValueError("No rows remain after min_total_count filtering.")
    frame, join_stats = _join_meta(frame, meta, config)
    frame, region_stats = _annotate_regions(frame, "root_id")
    if frame.empty:
        raise ValueError("No BANC neurons remain after meta join / region mapping.")
    threshold = float(frame["total_out_synapses"].quantile(config.high_connectivity_quantile))
    ranked = _rank_regions(frame, min_region_samples=config.min_region_samples, max_regions=config.max_regions)
    selected = _sort_key_frame(frame[frame["region_id"].isin(ranked)])
    samples = []
    for row in selected.itertuples(index=False):
        total = int(row.total_out_synapses)
        label = "high_connectivity" if float(total) >= threshold else "baseline_connectivity"
        score_delta = abs(float(total) - threshold) / max(float(total), threshold, 1.0)
        confidence = min(0.99, max(0.5, 0.55 + (0.4 * score_delta)))
        feats = _features(row, config)
        root_id = str(row.root_id)
        samples.append(
            {
                "sample_id": f"{_DATASET_TAG}-pre-{root_id}",
                "region_id": str(row.region_id),
                "input_text": _input_text(OBJECTIVE_CONNECTIVITY_TIER, root_id, feats),
                "expected_label": label,
                "expected_confidence": round(confidence, 6),
                "provenance_refs": [f"{_DATASET_TAG}:root_id:{root_id}", f"source:{source_label}"],
                "metadata": {
                    "dataset": "banc",
                    "dataset_version": BANC_VERSION_ID,
                    "root_id": root_id,
                    "cns_division": feats["cns_division"],
                    **_split_group_fields(row),
                    "root_region": str(row.root_region),
                    "task_type": "connectivity",
                    "risk_tier": "high" if label == "high_connectivity" else "medium",
                    "label_source": {"total_out_synapses": total, "n_post_partners": int(row.n_post_partners)},
                },
            }
        )
    stats = {
        "source_rows": int(edge_rows),
        "candidate_rows": int(len(frame)),
        "high_connectivity_threshold": threshold,
        "dropped_multi_anchor": 0,
        **join_stats,
        **region_stats,
    }
    return samples, stats


def _build_neurotransmitter(config: BancSampleBuildConfig, paths: Mapping[str, Path], source_label: str):
    nt = load_banc_nt_predictions(paths[ROLE_NT_PREDICTION])
    meta = load_banc_meta(paths[ROLE_META], optional_columns=SPLIT_GROUP_META_COLUMNS)
    totals, _edge_rows = load_banc_outgoing_totals(paths[ROLE_EDGELIST_V3])
    source_rows = int(len(nt))
    multi_anchor = nt["nt_row_multiplicity"] > 1
    dropped_multi_anchor = int(multi_anchor.sum())
    nt = nt[~multi_anchor]
    frame = nt[nt["neurotransmitter_predicted"].notna() & (nt["count"].fillna(0).astype("int64") >= int(config.min_total_count))]
    if frame.empty:
        raise ValueError("No rows remain after min_total_count filtering on classified presynapse count.")
    frame = frame.rename(columns={"count": "classified_presynapses"})
    frame = frame.merge(totals[["root_id", "n_post_partners"]], on="root_id", how="left")
    frame["n_post_partners"] = frame["n_post_partners"].fillna(0).astype("int64")
    frame, join_stats = _join_meta(frame, meta, config)
    frame, region_stats = _annotate_regions(frame, "root_id")
    if frame.empty:
        raise ValueError("No BANC neurons remain after meta join / region mapping.")
    ranked = _rank_regions(frame, min_region_samples=config.min_region_samples, max_regions=config.max_regions)
    selected = _sort_key_frame(frame[frame["region_id"].isin(ranked)])
    samples = []
    for row in selected.itertuples(index=False):
        code = nt_short_code(str(row.neurotransmitter_predicted))
        score = float(row.neurotransmitter_score)
        if not (score == score) or score < 0.0 or score > 1.0:
            raise BancAdapterError(BancAdapterErrorCode.SCHEMA_MISMATCH, "neurotransmitter_score outside [0, 1]",
                                   {"root_id": str(row.root_id), "score": score})
        confidence = min(0.99, max(0.5, score))
        label = f"dominant_{code}"
        feats = _features(row, config)
        root_id = str(row.root_id)
        samples.append(
            {
                "sample_id": f"{_DATASET_TAG}-nt-{root_id}",
                "region_id": str(row.region_id),
                "input_text": _input_text(OBJECTIVE_NEUROTRANSMITTER_DOMINANCE, root_id, feats),
                "expected_label": label,
                "expected_confidence": round(confidence, 6),
                "provenance_refs": [f"{_DATASET_TAG}:root_id:{root_id}", f"source:{source_label}"],
                "metadata": {
                    "dataset": "banc",
                    "dataset_version": BANC_VERSION_ID,
                    "root_id": root_id,
                    "cns_division": feats["cns_division"],
                    **_split_group_fields(row),
                    "root_region": str(row.root_region),
                    "dominant_neurotransmitter": label,
                    "task_type": "neurotransmitter",
                    "risk_tier": "high" if confidence >= 0.85 else "medium",
                    "label_source": {"neurotransmitter_score": round(score, 6),
                                     "classified_presynapses": int(row.classified_presynapses)},
                },
            }
        )
    stats = {
        "source_rows": source_rows,
        "candidate_rows": int(len(frame)),
        "high_connectivity_threshold": None,
        "dropped_multi_anchor": dropped_multi_anchor,
        **join_stats,
        **region_stats,
    }
    return samples, stats


def build_banc_training_samples(objective: str, config: BancSampleBuildConfig) -> dict[str, Any]:
    if config.objective != objective:
        raise ValueError(f"config.objective {config.objective!r} != requested objective {objective!r}")
    _validate_config(config)
    try:
        import pyarrow  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pyarrow is required to build BANC training samples") from exc

    paths, provenance = _resolve_inputs(config)
    primary_role = ROLE_NT_PREDICTION if objective == OBJECTIVE_NEUROTRANSMITTER_DOMINANCE else ROLE_EDGELIST_V3
    source_path = paths[primary_role]
    if objective == OBJECTIVE_NEUROTRANSMITTER_DOMINANCE:
        all_samples, stats = _build_neurotransmitter(config, paths, source_path.name)
    else:
        all_samples, stats = _build_connectivity(config, paths, source_path.name)
    assert_no_label_leakage(all_samples, objective)
    capped = balance_and_cap_samples(all_samples, max_samples=int(config.max_samples))
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
        "proofread_only": config.proofread_only,
        "partner_tier_edges": list(config.partner_tier_edges),
        "exclude_super_classes": sorted(config.exclude_super_classes),
        "input_features": list(INPUT_FEATURES[objective]),
    }
    extra = {
        "dataset_version": BANC_VERSION_ID,
        "region_vocabulary": "banc_neuropil",
        "region_division_counts": _division_counts(capped),
        "input_features": list(INPUT_FEATURES[objective]),
        "forbidden_input_features": sorted(FORBIDDEN_INPUT_FEATURES[objective]),
        "leakage_check": "passed",
        "high_connectivity_threshold": stats["high_connectivity_threshold"],
        "min_total_count": int(config.min_total_count),
        "min_region_samples": int(config.min_region_samples),
        "max_samples": int(config.max_samples),
        "max_regions": int(config.max_regions),
        "proofread_only": bool(config.proofread_only),
        "input_paths": {role: str(path) for role, path in sorted(paths.items())},
        "provenance_mode": provenance["mode"],
        "manifest_id": provenance["manifest_id"],
        "manifest_sha256": provenance["manifest_sha256"],
        "product_sha256": provenance["product_sha256"],
        "filter_stats": {
            key: int(stats[key])
            for key in (
                "dropped_meta_unmatched",
                "dropped_non_neuronal",
                "dropped_unproofread",
                "dropped_missing_root_region",
                "dropped_division_conflict",
                "dropped_multi_anchor",
            )
        },
        "license": "CC-BY-4.0",
    }
    return assemble_training_payload(
        capped,
        symbol=BANC_SYMBOL,
        objective=objective,
        source_path=str(source_path),
        source_rows=int(stats["source_rows"]),
        candidate_rows=int(stats["candidate_rows"]),
        min_distinct_labels=int(config.min_distinct_labels),
        max_label_share=float(config.max_label_share),
        fingerprint_inputs=fingerprint_inputs,
        extra_metadata=extra,
    )


def attach_structured_features(payload: Mapping[str, Any], features, *, objective: str,
                               id_column: str = "node_id") -> dict[str, Any]:
    """Return a copy of a built payload whose samples carry ``features`` (structured model inputs).

    ``features`` is a DataFrame with ``id_column`` (BANC root id as str) plus
    feature columns, e.g. ``flybrain_wiring_features.build_wiring_features(...).frame``.
    Fails closed (``LabelLeakageError``) when any column is excluded for
    ``objective``; NaN becomes None. ``input_text`` and labels are unchanged, so
    existing text-model callers see the same payload.
    """
    import math

    import flybrain_wiring_features as fwf

    columns = [c for c in features.columns if c != id_column]
    fwf.assert_features_allowed(columns, objective)
    ids = features[id_column].astype(str)
    if ids.duplicated().any():
        raise ValueError(f"features.{id_column} is not unique")
    table = features[columns].set_axis(ids.tolist(), axis=0)
    samples = []
    missing = 0
    for sample in payload["samples"]:
        root_id = str(sample["metadata"]["root_id"])
        values: dict[str, Any] = {name: None for name in columns}
        if root_id in table.index:
            for name, value in table.loc[root_id].items():
                if hasattr(value, "item"):
                    value = value.item()
                if isinstance(value, float) and not math.isfinite(value):
                    value = None
                values[str(name)] = value
        else:
            missing += 1
        samples.append({**sample, "features": values})
    metadata = dict(payload["metadata"])
    metadata["structured_features"] = {"objective": objective, "columns": sorted(columns), "missing_rows": missing}
    return {**payload, "samples": samples, "metadata": metadata}


def register_banc_sample_builders(*, replace: bool = False) -> None:
    register_sample_builder(
        BANC_SYMBOL,
        BANC_SAMPLE_OBJECTIVES,
        build_banc_training_samples,
        config_type=BancSampleBuildConfig,
        replace=replace,
    )


def unregister_banc_sample_builders() -> None:
    for objective in BANC_SAMPLE_OBJECTIVES:
        unregister_sample_builder(BANC_SYMBOL, objective)


def main(argv: Sequence[str] | None = None) -> int:
    from flybrain_brain_cluster_samples import build_training_samples

    parser = argparse.ArgumentParser(description="Build deterministic brain-cluster training samples from local BANC v888.")
    parser.add_argument("--storage-root", help="FlyBrain storage root (defaults to LOCI_FLYBRAIN_STORAGE_ROOT)")
    parser.add_argument("--objective", choices=BANC_SAMPLE_OBJECTIVES, default=OBJECTIVE_CONNECTIVITY_TIER)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--min-total-count", type=int, default=10)
    parser.add_argument("--min-region-samples", type=int, default=25)
    parser.add_argument("--high-connectivity-quantile", type=float, default=0.75)
    parser.add_argument("--max-regions", type=int, default=20)
    parser.add_argument("--allow-planned", action="store_true", help="Required while banc is 'planned' in the registry.")
    args = parser.parse_args(argv)
    config = BancSampleBuildConfig(
        objective=args.objective,
        storage_root=args.storage_root,
        max_samples=args.max_samples,
        min_total_count=args.min_total_count,
        min_region_samples=args.min_region_samples,
        high_connectivity_quantile=args.high_connectivity_quantile,
        max_regions=args.max_regions,
    )
    schema = "flybrain-banc-training-samples/v1"
    try:
        payload = build_training_samples(BANC_SYMBOL, args.objective, config, allow_planned=args.allow_planned)
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
        "region_division_counts": meta["region_division_counts"],
    }, sort_keys=True))
    return 0


register_banc_sample_builders()


if __name__ == "__main__":
    raise SystemExit(main())
