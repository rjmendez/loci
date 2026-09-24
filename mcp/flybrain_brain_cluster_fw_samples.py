from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from flybrain_harness_storage import build_flybrain_harness_layout

_REGION_SANITIZE_RE = re.compile(r"[^a-z0-9]+")
_NT_COLUMNS = ("ach_avg", "gaba_avg", "glut_avg", "da_avg", "ser_avg", "oct_avg")


@dataclass(frozen=True)
class FwSampleBuildConfig:
    storage_root: str | Path | None = None
    source_path: str | Path | None = None
    companion_pre_path: str | Path | None = None
    objective: str = "connectivity_tier"
    max_samples: int = 5000
    min_total_count: int = 10
    min_region_samples: int = 25
    high_connectivity_quantile: float = 0.75
    max_regions: int = 20
    min_distinct_labels: int = 2
    max_label_share: float = 0.9


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sanitize_region(raw: str) -> str:
    value = _REGION_SANITIZE_RE.sub("_", str(raw).strip().lower()).strip("_")
    return value or "unknown_region"


def _resolve_source_path(config: FwSampleBuildConfig) -> Path:
    if config.source_path is not None:
        candidate = Path(config.source_path).resolve(strict=False)
    else:
        layout = build_flybrain_harness_layout(root_override=config.storage_root, create=False)
        if config.objective == "neurotransmitter_dominance":
            candidate = (layout.snapshots / "fw" / "flywire783" / "metadata" / "files" / "proofread_connections_783.feather").resolve(strict=False)
        else:
            candidate = (layout.snapshots / "fw" / "flywire783" / "metadata" / "files" / "per_neuron_neuropil_count_pre_783.feather").resolve(strict=False)
    if not candidate.exists():
        raise FileNotFoundError(f"FlyWire source file not found: {candidate}")
    return candidate


def _resolve_companion_pre_path(config: FwSampleBuildConfig) -> Path:
    if config.companion_pre_path is not None:
        candidate = Path(config.companion_pre_path).resolve(strict=False)
    else:
        layout = build_flybrain_harness_layout(root_override=config.storage_root, create=False)
        candidate = (layout.snapshots / "fw" / "flywire783" / "metadata" / "files" / "per_neuron_neuropil_count_pre_783.feather").resolve(strict=False)
    if not candidate.exists():
        raise FileNotFoundError(f"FlyWire pre-neuropil companion file not found: {candidate}")
    return candidate


def _validate_config(config: FwSampleBuildConfig) -> None:
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
    if config.objective not in {"connectivity_tier", "neurotransmitter_dominance"}:
        raise ValueError("objective must be one of: connectivity_tier, neurotransmitter_dominance")


def _row_to_connectivity_sample(
    *,
    pre_root_id: int,
    dominant_neuropil: str,
    dominant_count: int,
    total_count: int,
    high_threshold: float,
    source_path: Path,
) -> dict[str, Any]:
    label = "high_connectivity" if float(total_count) >= high_threshold else "baseline_connectivity"
    score_delta = abs(float(total_count) - high_threshold) / max(float(total_count), high_threshold, 1.0)
    confidence = min(0.99, max(0.5, 0.55 + (0.4 * score_delta)))
    dominance_ratio = float(dominant_count) / max(float(total_count), 1.0)
    region_id = _sanitize_region(dominant_neuropil)
    sample_id = f"fw783-pre-{pre_root_id}"
    return {
        "sample_id": sample_id,
        "region_id": region_id,
        "input_text": (
            f"dataset flywire783 pre_root {pre_root_id} dominant_neuropil {dominant_neuropil} "
            f"total_pre_count {int(total_count)} dominant_count {int(dominant_count)} "
            f"dominance_ratio {dominance_ratio:.6f}"
        ),
        "expected_label": label,
        "expected_confidence": round(confidence, 6),
        "provenance_refs": [
            f"flywire783:pre_pt_root_id:{pre_root_id}",
            f"source:{source_path.name}",
        ],
        "metadata": {
            "dataset": "flywire",
            "dataset_version": "flywire783",
            "source_file": str(source_path),
            "pre_pt_root_id": str(pre_root_id),
            "dominant_neuropil": dominant_neuropil,
            "total_pre_count": int(total_count),
            "dominant_count": int(dominant_count),
            "task_type": "connectivity",
            "risk_tier": "high" if label == "high_connectivity" else "medium",
        },
    }


def _row_to_neurotransmitter_sample(
    *,
    pre_root_id: int,
    dominant_neuropil: str,
    total_syn_count: int,
    nt_scores: Mapping[str, float],
    source_path: Path,
) -> dict[str, Any]:
    ordered = sorted(((nt, float(score)) for nt, score in nt_scores.items()), key=lambda item: (-item[1], item[0]))
    primary_nt, primary_score = ordered[0]
    secondary_score = ordered[1][1] if len(ordered) > 1 else 0.0
    confidence = min(0.99, max(0.5, float(primary_score)))
    margin = max(0.0, float(primary_score) - float(secondary_score))
    label = f"dominant_{primary_nt.replace('_avg', '')}"
    region_id = _sanitize_region(dominant_neuropil)
    sample_id = f"fw783-nt-{pre_root_id}"
    return {
        "sample_id": sample_id,
        "region_id": region_id,
        "input_text": (
            f"dataset flywire783 pre_root {pre_root_id} dominant_neuropil {dominant_neuropil} "
            f"total_syn_count {int(total_syn_count)} ach_avg {float(nt_scores['ach_avg']):.6f} "
            f"gaba_avg {float(nt_scores['gaba_avg']):.6f} glut_avg {float(nt_scores['glut_avg']):.6f} "
            f"da_avg {float(nt_scores['da_avg']):.6f} ser_avg {float(nt_scores['ser_avg']):.6f} "
            f"oct_avg {float(nt_scores['oct_avg']):.6f} nt_margin {margin:.6f}"
        ),
        "expected_label": label,
        "expected_confidence": round(confidence, 6),
        "provenance_refs": [
            f"flywire783:pre_pt_root_id:{pre_root_id}",
            f"source:{source_path.name}",
        ],
        "metadata": {
            "dataset": "flywire",
            "dataset_version": "flywire783",
            "source_file": str(source_path),
            "pre_pt_root_id": str(pre_root_id),
            "dominant_neuropil": dominant_neuropil,
            "total_syn_count": int(total_syn_count),
            "dominant_neurotransmitter": label,
            "task_type": "neurotransmitter",
            "risk_tier": "high" if confidence >= 0.85 else "medium",
        },
    }


def _balance_and_cap_samples(samples: Sequence[dict[str, Any]], *, max_samples: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(str(sample["region_id"]), []).append(sample)
    for region_id in grouped:
        grouped[region_id].sort(key=lambda item: str(item["sample_id"]))

    selected: list[dict[str, Any]] = []
    region_order = sorted(grouped)
    index = 0
    while len(selected) < max_samples and region_order:
        region_id = region_order[index % len(region_order)]
        bucket = grouped[region_id]
        if bucket:
            selected.append(bucket.pop(0))
        if not bucket:
            region_order = [r for r in region_order if grouped[r]]
            index = 0
            continue
        index += 1
    return selected


def _rank_regions(merged, *, min_region_samples: int, max_regions: int) -> list[str]:
    region_counts = merged.groupby("region_id", sort=True)["pre_pt_root_id"].count().to_dict()
    eligible_regions = {
        region_id
        for region_id, count in region_counts.items()
        if int(count) >= int(min_region_samples)
    }
    if not eligible_regions:
        raise ValueError(
            "No regions satisfy min_region_samples. Lower --min-region-samples or inspect source distribution."
        )
    return sorted(
        eligible_regions,
        key=lambda region: (-int(region_counts[region]), region),
    )[: int(max_regions)]


def _build_connectivity_samples(config: FwSampleBuildConfig, source_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.feather as feather

    table = feather.read_table(str(source_path), columns=["pre_pt_root_id", "neuropil", "count"])
    frame = table.to_pandas()
    frame = frame[frame["count"].astype("int64") >= int(config.min_total_count)]
    if frame.empty:
        raise ValueError("No rows remain after min_total_count filtering.")

    totals = (
        frame.groupby("pre_pt_root_id", sort=True)["count"]
        .sum()
        .rename("total_count")
        .to_frame()
        .reset_index()
    )
    dominant_idx = frame.groupby("pre_pt_root_id", sort=True)["count"].idxmax()
    dominant = (
        frame.loc[dominant_idx, ["pre_pt_root_id", "neuropil", "count"]]
        .rename(columns={"count": "dominant_count", "neuropil": "dominant_neuropil"})
        .reset_index(drop=True)
    )
    merged = totals.merge(dominant, on="pre_pt_root_id", how="inner")
    if merged.empty:
        raise ValueError("Unable to derive dominant neuropil rows from FlyWire input.")

    high_threshold = float(merged["total_count"].quantile(config.high_connectivity_quantile))
    merged["region_id"] = merged["dominant_neuropil"].map(_sanitize_region)
    ranked_regions = _rank_regions(merged, min_region_samples=config.min_region_samples, max_regions=config.max_regions)
    filtered = merged[merged["region_id"].isin(ranked_regions)].copy()
    filtered = filtered.sort_values(["region_id", "pre_pt_root_id"], kind="mergesort")

    all_samples = [
        _row_to_connectivity_sample(
            pre_root_id=int(row.pre_pt_root_id),
            dominant_neuropil=str(row.dominant_neuropil),
            dominant_count=int(row.dominant_count),
            total_count=int(row.total_count),
            high_threshold=high_threshold,
            source_path=source_path,
        )
        for row in filtered.itertuples(index=False)
    ]
    return all_samples, {
        "source_rows": int(table.num_rows),
        "candidate_rows": int(len(merged)),
        "high_connectivity_threshold": float(high_threshold),
        "objective": "connectivity_tier",
    }


def _build_neurotransmitter_samples(config: FwSampleBuildConfig, source_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.feather as feather

    pre_companion_path = _resolve_companion_pre_path(config)
    nt_table = feather.read_table(str(source_path), columns=["pre_pt_root_id", "neuropil", "syn_count", *_NT_COLUMNS])
    nt_frame = nt_table.to_pandas()
    nt_frame = nt_frame[nt_frame["syn_count"].astype("int64") >= int(config.min_total_count)]
    if nt_frame.empty:
        raise ValueError("No rows remain after min_total_count filtering on syn_count.")

    for col in _NT_COLUMNS:
        nt_frame[col] = nt_frame[col].astype("float64")
    nt_frame["syn_count"] = nt_frame["syn_count"].astype("int64")
    nt_frame["weighted_total"] = nt_frame["syn_count"].astype("float64")
    for col in _NT_COLUMNS:
        nt_frame[f"weighted_{col}"] = nt_frame["syn_count"] * nt_frame[col]

    agg_cols = {f"weighted_{col}": "sum" for col in _NT_COLUMNS}
    agg_cols["syn_count"] = "sum"
    grouped = nt_frame.groupby("pre_pt_root_id", sort=True).agg(agg_cols).reset_index()
    grouped = grouped.rename(columns={"syn_count": "total_syn_count"})
    for col in _NT_COLUMNS:
        grouped[col] = grouped[f"weighted_{col}"] / grouped["total_syn_count"].clip(lower=1)

    companion_table = feather.read_table(str(pre_companion_path), columns=["pre_pt_root_id", "neuropil", "count"])
    companion_frame = companion_table.to_pandas()
    dominant_idx = companion_frame.groupby("pre_pt_root_id", sort=True)["count"].idxmax()
    dominant = (
        companion_frame.loc[dominant_idx, ["pre_pt_root_id", "neuropil"]]
        .rename(columns={"neuropil": "dominant_neuropil"})
        .reset_index(drop=True)
    )
    merged = grouped.merge(dominant, on="pre_pt_root_id", how="left")
    merged["dominant_neuropil"] = merged["dominant_neuropil"].fillna("UNKNOWN")
    merged["region_id"] = merged["dominant_neuropil"].map(_sanitize_region)
    ranked_regions = _rank_regions(merged, min_region_samples=config.min_region_samples, max_regions=config.max_regions)
    filtered = merged[merged["region_id"].isin(ranked_regions)].copy()
    filtered = filtered.sort_values(["region_id", "pre_pt_root_id"], kind="mergesort")

    all_samples = [
        _row_to_neurotransmitter_sample(
            pre_root_id=int(row.pre_pt_root_id),
            dominant_neuropil=str(row.dominant_neuropil),
            total_syn_count=int(row.total_syn_count),
            nt_scores={col: float(getattr(row, col)) for col in _NT_COLUMNS},
            source_path=source_path,
        )
        for row in filtered.itertuples(index=False)
    ]
    return all_samples, {
        "source_rows": int(nt_table.num_rows),
        "candidate_rows": int(len(merged)),
        "high_connectivity_threshold": None,
        "objective": "neurotransmitter_dominance",
        "companion_pre_path": str(pre_companion_path),
    }


def build_fw_training_samples(config: FwSampleBuildConfig) -> dict[str, Any]:
    _validate_config(config)
    try:
        import pyarrow.feather as feather  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pyarrow is required to build FlyWire training samples") from exc

    source_path = _resolve_source_path(config)
    if config.objective == "neurotransmitter_dominance":
        all_samples, build_meta = _build_neurotransmitter_samples(config, source_path)
    else:
        all_samples, build_meta = _build_connectivity_samples(config, source_path)
    capped = _balance_and_cap_samples(all_samples, max_samples=int(config.max_samples))
    if not capped:
        raise ValueError("No training samples produced after region filtering and capping.")

    label_counts: dict[str, int] = {}
    for sample in capped:
        label = str(sample["expected_label"])
        label_counts[label] = label_counts.get(label, 0) + 1
    if len(label_counts) < int(config.min_distinct_labels):
        raise ValueError(
            f"label diversity below minimum ({len(label_counts)} < {config.min_distinct_labels}); "
            "adjust objective filters or caps."
        )
    dominant_share = max(label_counts.values()) / float(len(capped))
    if dominant_share > float(config.max_label_share):
        raise ValueError(
            f"label concentration too high ({dominant_share:.3f} > {config.max_label_share:.3f}); "
            "dataset is too imbalanced for stable training."
        )

    payload = {
        "samples": capped,
        "metadata": {
            "schema_version": "flybrain-fw-training-samples/v1",
            "source_path": str(source_path),
            "objective": build_meta["objective"],
            "source_rows": int(build_meta["source_rows"]),
            "candidate_rows": int(build_meta["candidate_rows"]),
            "selected_count": int(len(capped)),
            "selected_regions": sorted({str(s["region_id"]) for s in capped}),
            "label_counts": dict(sorted(label_counts.items())),
            "high_connectivity_threshold": build_meta["high_connectivity_threshold"],
            "min_total_count": int(config.min_total_count),
            "min_region_samples": int(config.min_region_samples),
            "max_samples": int(config.max_samples),
            "max_regions": int(config.max_regions),
            "min_distinct_labels": int(config.min_distinct_labels),
            "max_label_share": float(config.max_label_share),
            "companion_pre_path": build_meta.get("companion_pre_path"),
            "dominant_label_share": float(dominant_share),
            "input_fingerprint": hashlib.sha256(
                _stable_json(
                    {
                        "objective": config.objective,
                        "source_path": str(source_path),
                        "companion_pre_path": build_meta.get("companion_pre_path"),
                        "max_samples": config.max_samples,
                        "min_total_count": config.min_total_count,
                        "min_region_samples": config.min_region_samples,
                        "high_connectivity_quantile": config.high_connectivity_quantile,
                        "max_regions": config.max_regions,
                        "min_distinct_labels": config.min_distinct_labels,
                        "max_label_share": config.max_label_share,
                    }
                ).encode("utf-8")
            ).hexdigest(),
        },
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build deterministic brain-cluster training samples from FlyWire raw data."
    )
    parser.add_argument("--storage-root", help="FlyBrain storage root (for example F:\\.flybrain)")
    parser.add_argument(
        "--objective",
        choices=("connectivity_tier", "neurotransmitter_dominance"),
        default="connectivity_tier",
        help="Training objective to materialize from FlyWire data.",
    )
    parser.add_argument(
        "--source-path",
        help="Optional direct source path. Defaults are objective-specific FlyWire snapshot files.",
    )
    parser.add_argument(
        "--companion-pre-path",
        help="Optional pre-neuropil source for region mapping (used by neurotransmitter_dominance objective).",
    )
    parser.add_argument("--output", required=True, help="Output JSON file containing samples[]")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--min-total-count", type=int, default=10)
    parser.add_argument("--min-region-samples", type=int, default=25)
    parser.add_argument("--high-connectivity-quantile", type=float, default=0.75)
    parser.add_argument("--max-regions", type=int, default=20)
    parser.add_argument("--min-distinct-labels", type=int, default=2)
    parser.add_argument("--max-label-share", type=float, default=0.9)
    args = parser.parse_args(argv)

    config = FwSampleBuildConfig(
        storage_root=args.storage_root,
        source_path=args.source_path,
        companion_pre_path=args.companion_pre_path,
        objective=args.objective,
        max_samples=args.max_samples,
        min_total_count=args.min_total_count,
        min_region_samples=args.min_region_samples,
        high_connectivity_quantile=args.high_connectivity_quantile,
        max_regions=args.max_regions,
        min_distinct_labels=args.min_distinct_labels,
        max_label_share=args.max_label_share,
    )
    try:
        payload = build_fw_training_samples(config)
    except (ValueError, RuntimeError, FileNotFoundError, KeyError, TypeError) as exc:
        print(
            _stable_json(
                {
                    "schema_version": "flybrain-fw-training-samples/v1",
                    "status": "error",
                    "pass": False,
                    "error": str(exc),
                }
            )
        )
        return 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_stable_json(payload) + "\n", encoding="utf-8")
    print(
        _stable_json(
            {
                "schema_version": "flybrain-fw-training-samples/v1",
                "status": "ok",
                "pass": True,
                "output": str(output_path),
                "sample_count": len(payload["samples"]),
                "regions": len(payload["metadata"]["selected_regions"]),
                "objective": payload["metadata"]["objective"],
                "high_connectivity_threshold": payload["metadata"]["high_connectivity_threshold"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
