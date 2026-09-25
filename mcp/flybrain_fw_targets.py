"""FlyWire 783 real-model targets: annotation-derived labels + structured wiring features.

This module builds ``flybrain_model_eval.EvalDataset`` objects for three
FlyWire 783 targets and runs the shared evaluation harness:

* ``super_class``: annotation ``super_class`` harmonized onto the shared coarse
  vocabulary with ``flybrain_wiring_features.harmonize_super_class``.
* ``nt_type``: annotation neurotransmitter type, normalized to ``ach`` /
  ``gaba`` / ``glut`` / ``da`` / ``ser`` / ``oct``.
* ``connectivity_tier``: ``high_connectivity`` for the top quartile of the
  neuron's outgoing weighted synapse mass, else ``baseline_connectivity``.

Features are intentionally simple and dataset-local:

* ``degree__out_degree`` / ``degree__in_degree`` from proofread connections.
* ``nt_out__*``: per-neuron outgoing NT probabilities, weighted by
  ``syn_count`` and averaged over outgoing edges.
* ``pre_np__*`` / ``post_np__*``: top-K per-neuropil pre/post fractions,
  plus ``other`` and entropy.

Splits are grouped by ``cell_type`` so every test neuron comes from a held-out
cell type, matching the other FlyBrain real-model datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import flybrain_model_eval as fme
import flybrain_learners as fl
import flybrain_wiring_features as fwf

DATASET = "fw"
VERSION_ID = "flywire783"
TARGET_SUPER_CLASS = "super_class"
TARGET_NT_TYPE = "nt_type"
TARGET_CONNECTIVITY = "connectivity_tier"
FW_TARGETS: tuple[str, ...] = (TARGET_SUPER_CLASS, TARGET_NT_TYPE, TARGET_CONNECTIVITY)
GROUP_KEYS: tuple[str, ...] = ("cell_type",)
DEFAULT_REPORT_ROOT = "/home/rjmendez/.loci/flybrain-real-models/reports"
DEFAULT_ANNOTATIONS = "/mnt/f/.flybrain/snapshots/fw/flywire783/metadata/files/annotations/flywire_annotations_v783.csv"
DEFAULT_CONNECTIONS = "/mnt/f/.flybrain/snapshots/fw/flywire783/metadata/files/proofread_connections_783.feather"
DEFAULT_PRE_NEUROPIL = "/mnt/f/.flybrain/snapshots/fw/flywire783/metadata/files/per_neuron_neuropil_count_pre_783.feather"
DEFAULT_POST_NEUROPIL = "/mnt/f/.flybrain/snapshots/fw/flywire783/metadata/files/per_neuron_neuropil_count_post_783.feather"
NT_COLUMNS: tuple[str, ...] = ("ach_avg", "gaba_avg", "glut_avg", "da_avg", "ser_avg", "oct_avg")
ANNOTATION_COLUMNS: tuple[str, ...] = ("root_id", "super_class", "cell_type", "hemilineage", "nt_type")
NT_SHORT_CODES: Mapping[str, str] = {
    "acetylcholine": "ach",
    "gaba": "gaba",
    "glutamate": "glut",
    "dopamine": "da",
    "serotonin": "ser",
    "octopamine": "oct",
}
NON_LABEL_SUPER_CLASSES = frozenset({fwf.UNKNOWN, "non_neuronal", "other"})


def _register_fw_targets() -> None:
    if TARGET_NT_TYPE not in fwf.registered_objectives():
        nt_rule = fwf.objective_exclusions("nt_ground_truth")
        fwf.register_objective_exclusions(
            TARGET_NT_TYPE,
            nt_rule.patterns,
            reason="label is the annotation neurotransmitter type; NT prediction scores and names near-copy it",
        )


_register_fw_targets()


@dataclass(frozen=True)
class FwTargetConfig:
    annotations_path: str | Path = DEFAULT_ANNOTATIONS
    connections_path: str | Path = DEFAULT_CONNECTIONS
    pre_neuropil_path: str | Path = DEFAULT_PRE_NEUROPIL
    post_neuropil_path: str | Path = DEFAULT_POST_NEUROPIL
    top_k_neuropils: int = 16
    min_label_neurons: int = 50
    min_label_cell_types: int = 5
    connectivity_quantile: float = 0.75
    max_per_cell_type: int = 1
    cap_salt: str = "fw-real-models/v1"

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in out.items()}


@dataclass
class FwContext:
    annotations: pd.DataFrame
    features: pd.DataFrame
    feature_columns: tuple[str, ...]
    proofread_root_ids: tuple[str, ...]
    config: FwTargetConfig
    neuropils_pre: tuple[str, ...]
    neuropils_post: tuple[str, ...]
    connectivity_threshold: float

    @classmethod
    def load(cls, config: FwTargetConfig, *, log=print) -> "FwContext":
        started = time.time()
        ann_path = Path(config.annotations_path)
        conn_path = Path(config.connections_path)
        pre_path = Path(config.pre_neuropil_path)
        post_path = Path(config.post_neuropil_path)
        annotations = pd.read_csv(ann_path, usecols=list(ANNOTATION_COLUMNS), low_memory=False)
        annotations["root_id"] = pd.to_numeric(annotations["root_id"], errors="coerce").astype("Int64")
        annotations = annotations.dropna(subset=["root_id"]).copy()
        annotations["root_id"] = annotations["root_id"].astype("int64").astype(str)
        annotations["cell_type"] = [_clean_group(v) for v in annotations["cell_type"].tolist()]
        annotations["hemilineage"] = [_clean_opt(v) for v in annotations["hemilineage"].tolist()]
        annotations["super_class_h"] = [fwf.harmonize_super_class(v) for v in annotations["super_class"].tolist()]
        annotations["nt_type_h"] = [_normalize_nt(v) for v in annotations["nt_type"].tolist()]
        log(f"[fw] annotations rows={len(annotations)} path={ann_path}")

        conn = pd.read_feather(conn_path, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count", *NT_COLUMNS])
        conn["pre_pt_root_id"] = conn["pre_pt_root_id"].astype("int64").astype(str)
        conn["post_pt_root_id"] = conn["post_pt_root_id"].astype("int64").astype(str)
        conn["syn_count"] = pd.to_numeric(conn["syn_count"], errors="coerce").fillna(0).astype("float32")
        for column in NT_COLUMNS:
            conn[column] = pd.to_numeric(conn[column], errors="coerce").fillna(0).astype("float32")
        proofread_ids = tuple(sorted(pd.unique(pd.concat([conn["pre_pt_root_id"], conn["post_pt_root_id"]]))))
        annotations = annotations[annotations["root_id"].isin(proofread_ids)].copy()
        annotations = annotations[annotations["cell_type"].notna()].copy()
        log(f"[fw] proofread annotated neurons={len(annotations)} unique_cell_types={annotations['cell_type'].nunique()} "
            f"({time.time() - started:.0f}s)")

        features, threshold, top_pre, top_post = build_feature_frame(
            annotations["root_id"].tolist(), conn, pre_path, post_path, top_k=config.top_k_neuropils,
            quantile=config.connectivity_quantile, log=log
        )
        feature_columns = tuple(sorted(c for c in features.columns if c != "root_id"))
        features = features[["root_id", *feature_columns]].copy()
        log(f"[fw] feature frame={features.shape} q{config.connectivity_quantile:.2f}={threshold:.3f} "
            f"({time.time() - started:.0f}s)")
        return cls(annotations, features, feature_columns, proofread_ids, config, top_pre, top_post, threshold)


def _clean_group(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None
    return text


def _clean_opt(value: Any) -> str | None:
    return _clean_group(value)


def _normalize_nt(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    return NT_SHORT_CODES.get(text)


def _entropy_from_frame(frame: pd.DataFrame) -> pd.Series:
    arr = frame.to_numpy(dtype=np.float64, copy=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        bits = -(arr * np.log2(np.clip(arr, 1e-12, None)))
    return pd.Series(np.where(np.isfinite(bits).sum(axis=1) >= 0, bits.sum(axis=1), np.nan), index=frame.index)


def _slug_neuropil(value: Any) -> str:
    return fwf.slug(str(value).replace("(", "_").replace(")", "_"))


def _neuropil_fraction_features(path: Path, *, id_column: str, prefix: str, root_ids: Sequence[str], top_k: int,
                                log=print) -> tuple[pd.DataFrame, tuple[str, ...]]:
    table = pd.read_feather(path, columns=[id_column, "neuropil", "count"])
    table[id_column] = table[id_column].astype("int64").astype(str)
    table["count"] = pd.to_numeric(table["count"], errors="coerce").fillna(0).astype("float32")
    table = table[table[id_column].isin(root_ids)].copy()
    totals = table.groupby(id_column, sort=False)["count"].sum().rename("__total__")
    top = (
        table.groupby("neuropil", sort=False)["count"]
        .sum()
        .sort_values(ascending=False)
        .head(int(top_k))
        .index.tolist()
    )
    top_frame = table[table["neuropil"].isin(top)].pivot_table(
        index=id_column, columns="neuropil", values="count", aggfunc="sum", fill_value=0
    )
    top_frame = top_frame.reindex(columns=top, fill_value=0)
    fractions = top_frame.div(totals, axis=0).fillna(0)
    fractions.columns = [f"{prefix}__{_slug_neuropil(c)}" for c in fractions.columns]
    if not fractions.empty:
        entropy = _entropy_from_frame(fractions)
        fractions[f"{prefix}__other"] = np.maximum(0.0, 1.0 - fractions.sum(axis=1))
        fractions[f"{prefix}__entropy_bits"] = entropy
        fractions[f"{prefix}__n_neuropils"] = table.groupby(id_column, sort=False)["neuropil"].nunique().reindex(
            fractions.index
        ).astype("float32")
        fractions[f"{prefix}__dominant_fraction"] = fractions[
            [c for c in fractions.columns if c.startswith(f"{prefix}__") and c not in {
                f"{prefix}__other", f"{prefix}__entropy_bits", f"{prefix}__n_neuropils", f"{prefix}__dominant_fraction",
            }]
        ].max(axis=1)
    out = fractions.reset_index().rename(columns={id_column: "root_id"})
    log(f"[fw] {prefix} neuropils rows={len(out)} top_k={len(top)} path={path}")
    return out, tuple(_slug_neuropil(v) for v in top)


def build_feature_frame(root_ids: Sequence[str], conn: pd.DataFrame, pre_path: Path, post_path: Path, *, top_k: int,
                        quantile: float = 0.75, log=print) -> tuple[pd.DataFrame, float, tuple[str, ...], tuple[str, ...]]:
    out_degree = conn.groupby("pre_pt_root_id", sort=False)["syn_count"].sum().rename("degree__out_degree")
    in_degree = conn.groupby("post_pt_root_id", sort=False)["syn_count"].sum().rename("degree__in_degree")
    weighted = conn[["pre_pt_root_id", "syn_count", *NT_COLUMNS]].copy()
    for column in NT_COLUMNS:
        weighted[column] = weighted[column] * weighted["syn_count"]
    nt = weighted.groupby("pre_pt_root_id", sort=False).sum()
    nt_weighted_total = nt["syn_count"].rename("weighted_total_syn_count")
    for column in NT_COLUMNS:
        nt[column] = nt[column] / nt["syn_count"].clip(lower=1.0)
    nt = nt.drop(columns=["syn_count"]).rename(columns={c: f"nt_out__{c.replace('_avg', '')}" for c in NT_COLUMNS})

    base = pd.DataFrame({"root_id": pd.Index(root_ids, dtype=object)})
    base = base.merge(out_degree.rename_axis("root_id").reset_index(), on="root_id", how="left")
    base = base.merge(in_degree.rename_axis("root_id").reset_index(), on="root_id", how="left")
    base = base.merge(nt.rename_axis("root_id").reset_index(), on="root_id", how="left")
    base = base.merge(nt_weighted_total.rename_axis("root_id").reset_index(), on="root_id", how="left")
    pre_frame, top_pre = _neuropil_fraction_features(pre_path, id_column="pre_pt_root_id", prefix="pre_np",
                                                     root_ids=root_ids, top_k=top_k, log=log)
    post_frame, top_post = _neuropil_fraction_features(post_path, id_column="post_pt_root_id", prefix="post_np",
                                                       root_ids=root_ids, top_k=top_k, log=log)
    base = base.merge(pre_frame, on="root_id", how="left").merge(post_frame, on="root_id", how="left")
    for column in base.columns:
        if column != "root_id":
            base[column] = pd.to_numeric(base[column], errors="coerce").fillna(0.0).astype("float32")
    threshold = float(base["weighted_total_syn_count"].quantile(float(quantile)))
    return base, threshold, top_pre, top_post


def default_models() -> tuple[fme.ModelSpec, ...]:
    return (
        fme.ModelSpec(
            "logreg",
            (
                {"C": 1.0, "max_iter": 1000},
            ),
            None,
            "logreg",
        ),
    )


def binned_text(features: pd.DataFrame, train_rows: Sequence[int], *, n_bins: int = 5) -> list[str]:
    columns = sorted(features.columns)
    train_rows = np.asarray(list(train_rows), dtype=np.int64)
    parts: list[list[str]] = []
    for name in columns:
        col = features[name]
        if pd.api.types.is_numeric_dtype(col.dtype):
            values = col.to_numpy(dtype=np.float64)
            train = values[train_rows]
            train = train[np.isfinite(train)]
            if len(train) == 0:
                edges = np.asarray([], dtype=np.float64)
            else:
                qs = np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1]
                edges = np.unique(np.quantile(train, qs))
            bucket = np.digitize(np.where(np.isfinite(values), values, -1e18), edges, right=False)
            tokens = [f"{name} {name}_q{int(v)}" if np.isfinite(x) else f"{name} {name}_missing"
                      for x, v in zip(values, bucket, strict=False)]
        else:
            vals = col.astype(object).where(col.notna(), fwf.UNKNOWN).astype(str).tolist()
            tokens = [f"{name} {fwf.slug(v)}" for v in vals]
        parts.append(tokens)
    return [" ".join(row) for row in zip(*parts, strict=False)]


def eval_config(*, split_seed: str, report_root: str | None = DEFAULT_REPORT_ROOT, run_label: str = "",
                models: Sequence[fme.ModelSpec] | None = None, full: bool = False,
                n_bootstrap: int | None = None, n_threads: int = 8) -> fme.EvalConfig:
    return fme.EvalConfig(
        models=tuple(models or default_models()),
        split_seed=split_seed,
        cv_folds=0,
        tune_metric="macro_f1",
        seed=0,
        n_bootstrap=5 if n_bootstrap is None and full else (3 if n_bootstrap is None else int(n_bootstrap)),
        min_margin=0.01,
        shuffle_control=True,
        random_split_control=False,
        ablation=False,
        n_threads=n_threads,
        report_root=report_root,
        run_label=run_label,
        save_models=False,
        calibration_protocol="oof",
        calibration_bootstrap=0,
        size_baseline=False,
        require_size_baseline=False,
        n_permutations=0,
        require_permutation_null=False,
        split_curve=False,
    )


def _target_frame(target: str, ctx: FwContext) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = ctx.annotations.merge(ctx.features, on="root_id", how="left", validate="one_to_one")
    notes: dict[str, Any] = {
        "dataset_version": VERSION_ID,
        "config": ctx.config.as_dict(),
        "feature_columns": list(ctx.feature_columns),
        "n_features": len(ctx.feature_columns),
        "top_pre_neuropils": list(ctx.neuropils_pre),
        "top_post_neuropils": list(ctx.neuropils_post),
        "proofread_annotated_neurons": int(len(ctx.annotations)),
    }
    if target == TARGET_SUPER_CLASS:
        frame["label"] = frame["super_class_h"]
        frame = frame[~frame["label"].isin(NON_LABEL_SUPER_CLASSES)].copy()
        notes["label_definition"] = "annotation super_class harmonized via flybrain_wiring_features.harmonize_super_class"
    elif target == TARGET_NT_TYPE:
        frame["label"] = frame["nt_type_h"]
        frame = frame[frame["label"].notna()].copy()
        notes["label_definition"] = "annotation nt_type normalized to ach/gaba/glut/da/ser/oct"
    elif target == TARGET_CONNECTIVITY:
        frame["label"] = np.where(
            frame["weighted_total_syn_count"] >= float(ctx.connectivity_threshold),
            "high_connectivity",
            "baseline_connectivity",
        )
        notes["label_definition"] = "top quartile of outgoing weighted synapse mass from proofread_connections_783.feather"
        notes["high_connectivity_threshold"] = float(ctx.connectivity_threshold)
    else:
        raise ValueError(f"unknown fw target {target!r} (supported: {', '.join(FW_TARGETS)})")

    before = len(frame)
    if target != TARGET_CONNECTIVITY:
        frame, kept_notes = _drop_small_label_groups(
            frame,
            min_neurons=ctx.config.min_label_neurons,
            min_cell_types=ctx.config.min_label_cell_types,
        )
        notes.update(kept_notes)
    before_cap = len(frame)
    frame = _cap_per_cell_type(frame, max_per_cell_type=ctx.config.max_per_cell_type, salt=ctx.config.cap_salt)
    frame = frame.sort_values("root_id", kind="stable").reset_index(drop=True)
    notes.update({
        "rows_before_label_filters": int(before),
        "rows_before_cap": int(before_cap),
        "n_samples": int(len(frame)),
        "label_counts": {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()},
        "sampling": {"max_per_cell_type": int(ctx.config.max_per_cell_type),
                     "order": f"sha256({ctx.config.cap_salt}:cell_type:root_id)"},
    })
    return frame, notes


def _drop_small_label_groups(frame: pd.DataFrame, *, min_neurons: int, min_cell_types: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    counts = frame["label"].value_counts()
    n_types = frame.groupby("label", sort=True)["cell_type"].nunique()
    keep = sorted(
        label for label in counts.index
        if int(counts[label]) >= int(min_neurons) and int(n_types.get(label, 0)) >= int(min_cell_types)
    )
    out = frame[frame["label"].isin(keep)].copy()
    dropped = sorted(set(frame["label"].astype(str)) - set(str(v) for v in keep))
    notes = {
        "min_label_neurons": int(min_neurons),
        "min_label_cell_types": int(min_cell_types),
        "dropped_small_labels": {
            str(label): {"neurons": int(counts[label]), "cell_types": int(n_types.get(label, 0))}
            for label in dropped
        },
    }
    return out, notes


def _stable_cap_order(cell_type: str, root_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{cell_type}:{root_id}".encode("utf-8")).hexdigest()


def _cap_per_cell_type(frame: pd.DataFrame, *, max_per_cell_type: int, salt: str) -> pd.DataFrame:
    if max_per_cell_type <= 0:
        return frame
    ranked = frame.copy()
    ranked["__cap_order__"] = [
        _stable_cap_order(str(cell_type), str(root_id), salt)
        for cell_type, root_id in zip(ranked["cell_type"].tolist(), ranked["root_id"].tolist(), strict=False)
    ]
    ranked = ranked.sort_values(["cell_type", "__cap_order__"], kind="stable")
    ranked["__cap_rank__"] = ranked.groupby("cell_type", sort=False).cumcount()
    ranked = ranked[ranked["__cap_rank__"] < int(max_per_cell_type)].copy()
    return ranked.drop(columns=["__cap_order__", "__cap_rank__"])


def build_eval_dataset(target: str, ctx: FwContext, config: fme.EvalConfig) -> fme.EvalDataset:
    frame, notes = _target_frame(target, ctx)
    feature_cols = [c for c in ctx.feature_columns if c in frame.columns and c != "weighted_total_syn_count"]
    if target == TARGET_CONNECTIVITY:
        # keep the raw label source out of the feature frame; the registered
        # connectivity exclusions will also drop every degree__* proxy.
        feature_cols = [c for c in feature_cols if c != "weighted_total_syn_count"]
    data_frame = frame[["root_id", "label", *GROUP_KEYS, *feature_cols]].copy()
    kept, dropped = fwf.apply_objective_exclusions(data_frame[feature_cols], target)
    feature_cols = sorted(kept.columns.tolist())
    group_values = frame[list(GROUP_KEYS)].to_dict(orient="records")
    plan = fme.plan_grouped_split(frame["root_id"].tolist(), group_values, GROUP_KEYS, config)
    pos = {sid: i for i, sid in enumerate(frame["root_id"].tolist())}
    train_rows = np.asarray(sorted(pos[sid] for sid in plan["train"]), dtype=np.int64)
    text = binned_text(kept, train_rows)
    data_frame = pd.concat([data_frame[["root_id", "label", *GROUP_KEYS]].reset_index(drop=True),
                            kept.reset_index(drop=True)], axis=1)
    data_frame["__text__"] = text
    fwf.assert_features_allowed(feature_cols, target)
    notes["excluded_features"] = dropped
    notes["text_mode"] = "binned"
    return fme.EvalDataset.from_frame(
        data_frame,
        dataset=DATASET,
        target=target,
        id_column="root_id",
        label_column="label",
        feature_columns=feature_cols,
        group_columns=list(GROUP_KEYS),
        text_column="__text__",
        notes=notes,
    )


def _best_model_row(report: Mapping[str, Any]) -> Mapping[str, Any]:
    best = str(report.get("best_on_val") or "")
    for row in report["summary"]:
        if str(row["model"]) == best:
            return row
    return report["summary"][0]


def run_target(target: str, ctx: FwContext, *, report_root: str = DEFAULT_REPORT_ROOT, seeds: int = 5,
               base_seed: str = "flybrain-fw-real-models-v1", n_threads: int = 8, log=print) -> dict[str, Any]:
    runs = []
    for i in range(int(seeds)):
        label = "" if i == 0 else f"repeat-{i}"
        cfg = eval_config(
            split_seed=base_seed if i == 0 else f"{base_seed}-seed{i}",
            report_root=report_root,
            run_label=label,
            n_threads=n_threads,
            full=False,
        )
        started = time.time()
        report = fme.run_evaluation(build_eval_dataset(target, ctx, cfg), cfg)
        best = dict(_best_model_row(report))
        best["best_on_val"] = report.get("best_on_val")
        best["elapsed_seconds"] = round(time.time() - started, 1)
        runs.append({"seed_index": i, "split_seed": cfg.split_seed, "best": best, "summary": report["summary"]})
        log(json.dumps({"target": target, "seed_index": i, "best": best}, sort_keys=True, default=str))
    passes = sum(1 for run in runs if run["best"]["gate"] == "pass")
    return {
        "target": target,
        "seeds": int(seeds),
        "pass_count": int(passes),
        "target_pass": bool(passes >= 3),
        "primary": runs[0],
        "runs": runs,
    }


def run_all(*, report_root: str = DEFAULT_REPORT_ROOT, targets: Sequence[str] = FW_TARGETS, seeds: int = 5,
            top_k_neuropils: int = 16, min_label_neurons: int = 50, min_label_cell_types: int = 5,
            max_per_cell_type: int = 1,
            n_threads: int = 8) -> dict[str, Any]:
    started = time.time()
    config = FwTargetConfig(
        top_k_neuropils=top_k_neuropils,
        min_label_neurons=min_label_neurons,
        min_label_cell_types=min_label_cell_types,
        max_per_cell_type=max_per_cell_type,
    )
    ctx = FwContext.load(config)
    results = {target: run_target(target, ctx, report_root=report_root, seeds=seeds, n_threads=n_threads)
               for target in targets}
    payload = {
        "schema_version": "flybrain-fw-real-models/v1",
        "dataset": DATASET,
        "dataset_version": VERSION_ID,
        "config": config.as_dict(),
        "elapsed_seconds": round(time.time() - started, 1),
        "results": results,
    }
    out = Path(report_root) / DATASET
    out.mkdir(parents=True, exist_ok=True)
    (out / "fw_real_models.json").write_text(
        json.dumps(fl._jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (out / "SUMMARY.md").write_text(render_summary(payload), encoding="utf-8")
    return payload


def _f(value: Any) -> str:
    return "-" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{float(value):.3f}"


def render_summary(payload: Mapping[str, Any]) -> str:
    lines = [
        "# FlyWire 783 real models",
        "",
        f"dataset `{payload['dataset']}` version `{payload['dataset_version']}`",
        "",
        "| target | primary model | acc | best trivial | macro-F1 | n_test | gate | passes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for target, result in payload["results"].items():
        best = result["primary"]["best"]
        lines.append(
            f"| {target} | {best['model']} | {_f(best['model_acc'])} | {_f(best['best_trivial'])} | "
            f"{_f(best['macro_f1'])} | {best['n_test']} | {best['gate']} | {result['pass_count']}/{result['seeds']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate FlyWire 783 super_class / nt_type / connectivity_tier targets.")
    parser.add_argument("--targets", nargs="*", choices=list(FW_TARGETS), default=list(FW_TARGETS))
    parser.add_argument("--report-root", default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--top-k-neuropils", type=int, default=16)
    parser.add_argument("--min-label-neurons", type=int, default=50)
    parser.add_argument("--min-label-cell-types", type=int, default=5)
    parser.add_argument("--max-per-cell-type", type=int, default=1)
    parser.add_argument("--n-threads", type=int, default=8)
    args = parser.parse_args(argv)
    payload = run_all(
        report_root=args.report_root,
        targets=args.targets,
        seeds=args.seeds,
        top_k_neuropils=args.top_k_neuropils,
        min_label_neurons=args.min_label_neurons,
        min_label_cell_types=args.min_label_cell_types,
        max_per_cell_type=args.max_per_cell_type,
        n_threads=args.n_threads,
    )
    for target, result in payload["results"].items():
        print(json.dumps({"target": target, "primary": result["primary"]["best"], "pass_count": result["pass_count"],
                          "seeds": result["seeds"], "target_pass": result["target_pass"]}, default=str, sort_keys=True))
    return 0


__all__ = [
    "DATASET",
    "DEFAULT_ANNOTATIONS",
    "DEFAULT_CONNECTIONS",
    "DEFAULT_POST_NEUROPIL",
    "DEFAULT_PRE_NEUROPIL",
    "DEFAULT_REPORT_ROOT",
    "FW_TARGETS",
    "FwContext",
    "FwTargetConfig",
    "GROUP_KEYS",
    "TARGET_CONNECTIVITY",
    "TARGET_NT_TYPE",
    "TARGET_SUPER_CLASS",
    "build_eval_dataset",
    "build_feature_frame",
    "default_models",
    "eval_config",
    "render_summary",
    "run_all",
    "run_target",
]


if __name__ == "__main__":
    raise SystemExit(main())
