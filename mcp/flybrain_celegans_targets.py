"""C. elegans connectome targets evaluated with the FlyBrain real-model harness.

Targets
-------
* ``coarse_role``: sensory / interneuron / motor, from Witvliet/Cook neuron-class
  metadata. Modulatory, glia and muscle neurons are excluded from this target.
* ``nt_family``: mature-adult single-family neurotransmitter identity. Only the
  two well-supported single-family labels (acetylcholine / glutamate) are kept
  by default; sparse families and ambiguous codes are reported and dropped.

Features are built directly from the stage-8 NemaNode edge list because the
release is tiny enough to work comfortably in pandas: weighted in/out degree,
reciprocity, partner-role composition, and stable/variable connection-class
fractions. For ``coarse_role``, partner-role features mask every held-out
neuron's role before feature construction, mirroring the label-masking pattern
used by the other FlyBrain target modules.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _import_flybrain_modules() -> tuple[Any, Any]:
    try:
        import flybrain_model_eval as fme  # type: ignore
        import flybrain_wiring_features as fwf  # type: ignore
        return fme, fwf
    except ModuleNotFoundError:
        pass
    candidates = (
        Path("/home/rjmendez/development/flybrain/flybrain"),
        Path("/mnt/c/Users/rjmendez-admin/development/flybrain/flybrain"),
    )
    for candidate in candidates:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            try:
                import flybrain_model_eval as fme  # type: ignore
                import flybrain_wiring_features as fwf  # type: ignore
                return fme, fwf
            except ModuleNotFoundError:
                continue
    raise ModuleNotFoundError(
        "could not import flybrain_model_eval / flybrain_wiring_features; set PYTHONPATH or clone "
        "/home/rjmendez/development/flybrain"
    )


fme, fwf = _import_flybrain_modules()

DATASET = "celegans"
DEFAULT_DATA_ROOT = "/mnt/f/.flybrain/snapshots/celegans/witvliet2021_cook2019_r1/sources"
DEFAULT_REPORT_ROOT = "/home/rjmendez/.loci/flybrain-real-models/reports"
DEFAULT_STAGE = 8
GROUP_KEYS: tuple[str, ...] = ()
TARGET_COARSE_ROLE = "coarse_role"
TARGET_NT_FAMILY = "nt_family"
CELEGANS_TARGETS: tuple[str, ...] = (TARGET_COARSE_ROLE, TARGET_NT_FAMILY)
CORE_ROLE_MAP: Mapping[str, str] = {"sensory": "sensory", "inter": "interneuron", "motor": "motor"}
NT_FAMILY_MAP: Mapping[str, str] = {
    "a": "acetylcholine",
    "l": "glutamate",
    "g": "gaba",
    "d": "dopamine",
    "s": "serotonin",
    "o": "octopamine",
    "t": "tyramine",
}
ABLATION_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "degree": ("degree__*",),
    "recip": ("recip__*",),
    "partner_comp_out": ("out_comp__*",),
    "partner_comp_in": ("in_comp__*",),
    "stability": ("stability__*",),
}


def _register(objective: str, patterns: Sequence[str], reason: str) -> None:
    if objective in fwf.registered_objectives():
        return
    fwf.register_objective_exclusions(objective, tuple(patterns), reason=reason)


_register(
    TARGET_COARSE_ROLE,
    fwf.objective_exclusions("cell_class").patterns + ("*coarse_role*", "*coarse_type*", "*raw_typ*", "*classes*"),
    "label is curated neuron role metadata; class/type fields and copied role columns determine it",
)
_register(
    TARGET_NT_FAMILY,
    fwf.objective_exclusions("nt_ground_truth").patterns + ("*nt_family*", "*raw_nt*", "*nt_code*"),
    "label is the per-neuron neurotransmitter annotation; NT codes or predictions would define it",
)


@dataclass(frozen=True)
class TargetSpec:
    name: str
    partner_category: str
    mask_partner_category: bool
    description: str


TARGET_SPECS: Mapping[str, TargetSpec] = {
    TARGET_COARSE_ROLE: TargetSpec(
        TARGET_COARSE_ROLE,
        "coarse_role",
        True,
        "sensory / interneuron / motor from curated neuron class metadata",
    ),
    TARGET_NT_FAMILY: TargetSpec(
        TARGET_NT_FAMILY,
        "coarse_role",
        False,
        "single-family neurotransmitter label from nemanode neurons.json",
    ),
}


@dataclass
class CelegansContext:
    data_root: Path
    stage: int
    classes: pd.DataFrame
    neurons: pd.DataFrame
    nodes: pd.DataFrame
    edges: pd.DataFrame
    classifications: pd.DataFrame
    unmatched_edge_nodes: tuple[str, ...]

    @classmethod
    def load(cls, *, data_root: str | Path = DEFAULT_DATA_ROOT, stage: int = DEFAULT_STAGE) -> "CelegansContext":
        root = Path(data_root)
        classes = pd.read_csv(root / "nature2021/tables/_neurons.csv").rename(
            columns={"class": "classes", "type": "coarse_type", "integration": "developmental_origin"}
        )
        neurons = pd.DataFrame(json.loads((root / "nemanode/neurons.json").read_text(encoding="utf-8")))
        raw_edges = pd.DataFrame(
            json.loads((root / f"nemanode/connections/witvliet_2020_{int(stage)}.json").read_text(encoding="utf-8"))
        )
        raw_edges["weight"] = [float(sum(v) if isinstance(v, list) else v or 0.0) for v in raw_edges["syn"]]
        edges = raw_edges.groupby(["pre", "post"], as_index=False).agg(weight=("weight", "sum"))
        classifications = pd.read_csv(root / "nature2021/tables/connection_classifications.csv").rename(
            columns={"classification": "stability"}
        )
        edge_nodes = sorted(set(edges["pre"]).union(edges["post"]))
        nodes = neurons[neurons["name"].isin(edge_nodes)].copy()
        unmatched = tuple(sorted(set(edge_nodes) - set(nodes["name"])))
        nodes = nodes.merge(classes, on="classes", how="left", validate="many_to_one")
        if nodes["coarse_type"].isna().any():
            missing = sorted(nodes.loc[nodes["coarse_type"].isna(), "classes"].dropna().unique().tolist())
            raise ValueError(f"missing coarse role metadata for classes: {missing}")
        nodes["coarse_role"] = [CORE_ROLE_MAP.get(str(v), "other") for v in nodes["coarse_type"]]
        nodes["sample_id"] = nodes["name"].astype(str)
        nodes["split_group"] = nodes["classes"].astype(str)
        node_set = set(nodes["name"])
        edges = edges[edges["pre"].isin(node_set) & edges["post"].isin(node_set)].copy()
        return cls(root, int(stage), classes, neurons, nodes.sort_values("name").reset_index(drop=True),
                   edges.reset_index(drop=True), classifications, unmatched)


def default_models() -> tuple[Any, ...]:
    return (
        fme.ModelSpec("logreg", ({"C": 0.03}, {"C": 0.1}, {"C": 0.3}, {"C": 1.0}), "temperature", "logreg"),
        fme.ModelSpec(
            "hgb",
            (
                {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 5, "l2_regularization": 1.0},
                {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 10, "l2_regularization": 0.0},
            ),
            "temperature",
            "hgb",
        ),
    )


def eval_config(*, split_seed: str, report_root: str | Path | None, run_label: str,
                full: bool = True) -> Any:
    return fme.EvalConfig(
        models=default_models(),
        split_seed=split_seed,
        cv_folds=3,
        tune_metric="macro_f1",
        seed=0,
        n_bootstrap=200 if full else 100,
        min_margin=0.01,
        shuffle_control=False,
        random_split_control=False,
        ablation=False,
        ablation_families=ABLATION_FAMILIES,
        n_threads=4,
        report_root=str(report_root) if report_root is not None else None,
        run_label=run_label,
        save_models=False,
    )


def entropy_bits(frame: pd.DataFrame) -> pd.Series:
    arr = frame.to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(arr > 0, arr * np.log2(arr), 0.0)
    return pd.Series(-term.sum(axis=1), index=frame.index, dtype=np.float64)


def composition_features(edges: pd.DataFrame, *, index_name: str, group_name: str, prefix: str) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()
    table = edges.groupby([index_name, group_name])["weight"].sum().unstack(fill_value=0.0)
    totals = table.sum(axis=1).replace(0.0, np.nan)
    fractions = table.div(totals, axis=0).fillna(0.0)
    fractions = fractions.rename(columns={c: f"{prefix}__{fwf.slug(c)}" for c in fractions.columns})
    fractions[f"{prefix}__entropy"] = entropy_bits(fractions)
    return fractions


def stability_features(edges: pd.DataFrame, classifications: pd.DataFrame) -> pd.DataFrame:
    merged = edges.merge(classifications, on=["pre", "post"], how="left")
    merged["stability"] = merged["stability"].fillna("unknown")
    out = composition_features(merged.rename(columns={"pre": "node"}), index_name="node", group_name="stability",
                               prefix="stability__out")
    inn = composition_features(merged.rename(columns={"post": "node"}), index_name="node",
                               group_name="stability", prefix="stability__in")
    return out.join(inn, how="outer")


def build_feature_frame(ctx: CelegansContext, *, partner_category: Mapping[str, str]) -> pd.DataFrame:
    node_index = pd.Index(ctx.nodes["name"].tolist(), name="name")
    edges = ctx.edges.copy()
    reverse = edges.rename(columns={"pre": "post", "post": "pre", "weight": "reverse_weight"})
    edges = edges.merge(reverse, on=["pre", "post"], how="left")
    edges["reciprocal"] = edges["reverse_weight"].notna()

    frame = pd.DataFrame(index=node_index)
    out_stats = edges.groupby("pre").agg(
        **{
            "degree__out_degree": ("post", "nunique"),
            "degree__out_weight": ("weight", "sum"),
            "degree__out_mean_weight": ("weight", "mean"),
            "degree__out_max_weight": ("weight", "max"),
        }
    )
    in_stats = edges.groupby("post").agg(
        **{
            "degree__in_degree": ("pre", "nunique"),
            "degree__in_weight": ("weight", "sum"),
            "degree__in_mean_weight": ("weight", "mean"),
            "degree__in_max_weight": ("weight", "max"),
        }
    )
    frame = frame.join(out_stats).join(in_stats)

    out_recip = edges[edges["reciprocal"]].groupby("pre").agg(recip_weight=("weight", "sum"),
                                                                recip_partners=("post", "nunique"))
    in_recip = edges[edges["reciprocal"]].groupby("post").agg(recip_weight=("weight", "sum"),
                                                                recip_partners=("pre", "nunique"))
    frame["recip__out_weight_frac"] = out_recip["recip_weight"].reindex(node_index).fillna(0.0) / frame[
        "degree__out_weight"
    ].replace(0.0, np.nan)
    frame["recip__out_partner_frac"] = out_recip["recip_partners"].reindex(node_index).fillna(0.0) / frame[
        "degree__out_degree"
    ].replace(0.0, np.nan)
    frame["recip__in_weight_frac"] = in_recip["recip_weight"].reindex(node_index).fillna(0.0) / frame[
        "degree__in_weight"
    ].replace(0.0, np.nan)
    frame["recip__in_partner_frac"] = in_recip["recip_partners"].reindex(node_index).fillna(0.0) / frame[
        "degree__in_degree"
    ].replace(0.0, np.nan)

    out_weight = frame["degree__out_weight"].fillna(0.0)
    in_weight = frame["degree__in_weight"].fillna(0.0)
    frame["degree__out_share"] = out_weight / (out_weight + in_weight + 1e-9)
    frame["degree__total_weight"] = out_weight + in_weight
    for column in (
        "degree__out_degree",
        "degree__out_weight",
        "degree__out_mean_weight",
        "degree__out_max_weight",
        "degree__in_degree",
        "degree__in_weight",
        "degree__in_mean_weight",
        "degree__in_max_weight",
        "degree__total_weight",
    ):
        frame[f"{column}__log1p"] = np.log1p(frame[column].fillna(0.0))

    comp_edges = ctx.edges.copy()
    comp_edges["post_category"] = [partner_category.get(str(v), fwf.UNKNOWN) for v in comp_edges["post"]]
    comp_edges["pre_category"] = [partner_category.get(str(v), fwf.UNKNOWN) for v in comp_edges["pre"]]
    out_comp = composition_features(comp_edges, index_name="pre", group_name="post_category", prefix="out_comp")
    in_comp = composition_features(comp_edges.rename(columns={"post": "node"}), index_name="node",
                                   group_name="pre_category", prefix="in_comp")
    stability = stability_features(ctx.edges, ctx.classifications)
    frame = frame.join(out_comp).join(in_comp).join(stability)
    return frame.fillna(0.0).reindex(sorted(frame.columns), axis=1)


def target_frame(target: str, ctx: CelegansContext, *, min_nt_family_count: int = 10) -> tuple[pd.DataFrame, dict[str, Any]]:
    nodes = ctx.nodes.copy()
    notes: dict[str, Any] = {
        "stage": int(ctx.stage),
        "n_stage_nodes": int(len(nodes)),
        "unmatched_edge_nodes": list(ctx.unmatched_edge_nodes),
    }
    if target == TARGET_COARSE_ROLE:
        frame = nodes[nodes["coarse_role"] != "other"].copy()
        dropped = nodes.loc[nodes["coarse_role"] == "other", "coarse_type"].value_counts().sort_index().to_dict()
        frame["label"] = frame["coarse_role"]
        notes.update(
            {
                "label_definition": "curated neuron class collapsed to sensory / interneuron / motor",
                "dropped_other_types": {str(k): int(v) for k, v in dropped.items()},
                "split_strategy": "per-neuron random split (dataset too small for grouped holdout)",
            }
        )
    elif target == TARGET_NT_FAMILY:
        frame = nodes[~nodes["coarse_type"].isin(["muscle", "glia"])].copy()
        frame["raw_nt"] = frame["nt"].astype(str)
        frame["nt_family"] = [NT_FAMILY_MAP.get(str(v), None) for v in frame["nt"]]
        counts = frame["nt_family"].value_counts(dropna=True)
        keep = sorted(counts[counts >= int(min_nt_family_count)].index.tolist())
        dropped_sparse = {str(k): int(v) for k, v in counts[counts < int(min_nt_family_count)].sort_index().items()}
        ambiguous = frame[frame["nt_family"].isna()]
        frame = frame[frame["nt_family"].isin(keep)].copy()
        frame["label"] = frame["nt_family"]
        notes.update(
            {
                "label_definition": "single-family neurotransmitter code from nemanode neurons.json",
                "kept_families": keep,
                "dropped_sparse_families": dropped_sparse,
                "dropped_ambiguous_or_unassigned_codes": {
                    str(k): int(v) for k, v in ambiguous["raw_nt"].value_counts().sort_index().items()
                },
                "min_class_neurons": int(min_nt_family_count),
                "split_strategy": "per-neuron random split after dropping sparse / ambiguous families",
            }
        )
    else:
        raise ValueError(f"unknown target {target!r}; expected one of {', '.join(CELEGANS_TARGETS)}")
    frame = frame.sort_values("name", kind="stable").reset_index(drop=True)
    notes.update(
        {
            "target_description": TARGET_SPECS[target].description,
            "n_samples": int(len(frame)),
            "label_counts": {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()},
            "n_split_groups": int(len(frame)) if not GROUP_KEYS else int(frame[list(GROUP_KEYS)].drop_duplicates().shape[0]),
        }
    )
    return frame, notes


def build_eval_dataset(target: str, ctx: CelegansContext, config: Any, *, min_nt_family_count: int = 10) -> Any:
    spec = TARGET_SPECS[target]
    frame, notes = target_frame(target, ctx, min_nt_family_count=min_nt_family_count)
    if GROUP_KEYS:
        group_values = frame[list(GROUP_KEYS)].to_dict(orient="records")
        plan = fme.plan_grouped_split(frame["sample_id"].tolist(), group_values, GROUP_KEYS, config)
    else:
        train_ids, val_ids, test_ids = fme.deterministic_split_ids(
            list(frame["sample_id"]), split_seed=config.split_seed,
            train_ratio=config.train_ratio, val_ratio=config.val_ratio,
        )
        plan = {"train": train_ids, "val": val_ids, "test": test_ids}
    heldout = set(plan["val"]) | set(plan["test"])
    partner_category = dict(zip(ctx.nodes["name"], ctx.nodes[spec.partner_category], strict=False))
    if spec.mask_partner_category:
        for name in heldout:
            partner_category[str(name)] = fwf.UNKNOWN
    features = build_feature_frame(ctx, partner_category=partner_category)
    feature_block = features.loc[frame["name"]].reset_index(drop=True)
    fwf.assert_features_allowed(feature_block.columns, target)
    notes.update(
        {
            "feature_columns": list(feature_block.columns),
            "feature_count": int(feature_block.shape[1]),
            "partner_category": spec.partner_category,
            "partner_category_masked_for_val_test": bool(spec.mask_partner_category),
            "masked_nodes": int(sum(1 for name in frame["name"] if name in heldout)) if spec.mask_partner_category else 0,
            "masked_split_ids_sha256": fme.split_ids_sha256(plan),
            "source_files": {
                "classes_csv": str(ctx.data_root / "nature2021/tables/_neurons.csv"),
                "neurons_json": str(ctx.data_root / "nemanode/neurons.json"),
                "connections_json": str(ctx.data_root / f"nemanode/connections/witvliet_2020_{ctx.stage}.json"),
                "connection_classifications": str(ctx.data_root / "nature2021/tables/connection_classifications.csv"),
            },
        }
    )
    data = pd.concat([frame[["sample_id", "label", *GROUP_KEYS]].reset_index(drop=True), feature_block], axis=1)
    return fme.EvalDataset.from_frame(
        data,
        dataset=DATASET,
        target=target,
        id_column="sample_id",
        label_column="label",
        feature_columns=list(feature_block.columns),
        group_columns=list(GROUP_KEYS),
        notes=notes,
    )


def best_model_accuracy(report: Mapping[str, Any]) -> float:
    return float(report["models"][report["best_on_val"]]["test"]["accuracy"])


def best_model_macro_f1(report: Mapping[str, Any]) -> float:
    return float(report["models"][report["best_on_val"]]["test"]["macro_f1"])


def run_target(target: str, ctx: CelegansContext, *, report_root: str | Path = DEFAULT_REPORT_ROOT,
               split_seed: str = "celegans-real-models-v1", n_repeats: int = 2,
               min_nt_family_count: int = 10) -> dict[str, Any]:
    base_label = f"adult_stage_{ctx.stage}"
    primary_cfg = eval_config(split_seed=split_seed, report_root=report_root, run_label=base_label, full=True)
    primary = fme.run_evaluation(build_eval_dataset(target, ctx, primary_cfg, min_nt_family_count=min_nt_family_count),
                                 primary_cfg)
    repeats = []
    for i in range(1, int(n_repeats) + 1):
        seed = f"{split_seed}-repeat-{i}"
        cfg = eval_config(split_seed=seed, report_root=report_root, run_label=f"{base_label}-repeat-{i}", full=False)
        result = fme.run_evaluation(build_eval_dataset(target, ctx, cfg, min_nt_family_count=min_nt_family_count), cfg)
        repeats.append(
            {
                "split_seed": seed,
                "best_on_val": result["best_on_val"],
                "accuracy": best_model_accuracy(result),
                "macro_f1": best_model_macro_f1(result),
                "label_counts": result["notes"]["label_counts"],
            }
        )
    accuracies = [best_model_accuracy(primary), *[float(r["accuracy"]) for r in repeats]]
    macro_f1s = [best_model_macro_f1(primary), *[float(r["macro_f1"]) for r in repeats]]
    return {
        "target": target,
        "stage": int(ctx.stage),
        "dataset_stats": primary["notes"],
        "primary": {
            "best_on_val": primary["best_on_val"],
            "summary": primary["summary"],
            "split": primary["split"],
            "report_dir": str(fme.report_dir(report_root, DATASET, target, base_label)),
        },
        "repeats": repeats,
        "aggregate": {
            "n_runs": len(accuracies),
            "best_model_accuracy_mean": float(np.mean(accuracies)),
            "best_model_accuracy_std": float(np.std(accuracies, ddof=0)),
            "best_model_macro_f1_mean": float(np.mean(macro_f1s)),
            "best_model_macro_f1_std": float(np.std(macro_f1s, ddof=0)),
        },
    }


def summary_path(report_root: str | Path) -> Path:
    path = Path(report_root) / DATASET / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_all(*, data_root: str | Path = DEFAULT_DATA_ROOT, report_root: str | Path = DEFAULT_REPORT_ROOT,
            targets: Sequence[str] = CELEGANS_TARGETS, stage: int = DEFAULT_STAGE,
            split_seed: str = "celegans-real-models-v1", n_repeats: int = 2,
            min_nt_family_count: int = 10) -> dict[str, Any]:
    started = time.time()
    ctx = CelegansContext.load(data_root=data_root, stage=stage)
    results = {
        target: run_target(target, ctx, report_root=report_root, split_seed=split_seed, n_repeats=n_repeats,
                           min_nt_family_count=min_nt_family_count)
        for target in targets
    }
    payload = {
        "dataset": DATASET,
        "stage": int(stage),
        "data_root": str(data_root),
        "report_root": str(report_root),
        "targets": list(targets),
        "elapsed_seconds": round(time.time() - started, 2),
        "results": results,
    }
    path = summary_path(report_root)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--report-root", default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--stage", type=int, default=DEFAULT_STAGE)
    parser.add_argument("--target", choices=("all", *CELEGANS_TARGETS), default="all")
    parser.add_argument("--split-seed", default="celegans-real-models-v1")
    parser.add_argument("--n-repeats", type=int, default=2)
    parser.add_argument("--min-nt-family-count", type=int, default=10)
    args = parser.parse_args(argv)
    targets = CELEGANS_TARGETS if args.target == "all" else (args.target,)
    payload = run_all(
        data_root=args.data_root,
        report_root=args.report_root,
        targets=targets,
        stage=args.stage,
        split_seed=args.split_seed,
        n_repeats=args.n_repeats,
        min_nt_family_count=args.min_nt_family_count,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
