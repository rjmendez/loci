"""Non-circular l1em (Winding et al. 2023 L1 larva) targets and the real-model evaluation runner.

Which S2 columns can be a target for a *wiring* model
-------------------------------------------------------
Winding et al. 2023 (Supplementary Material, Fig. S4C and S10) give every brain
interneuron class a **connectivity-motif definition**: PN = postsynaptic to
sensory neurons, PN-somato = postsynaptic to ascending neurons, LN = local
within a sensory cohort, LHN = downstream of olfactory/thermo/visual PNs,
CN = downstream of LHN and MBON, MB-FBN / MB-FFN = feedback / feedforward
around MBIN and MBON, pre-DN-SEZ / pre-DN-VNC = presynaptic to DN-SEZ /
DN-VNC, and the KC / MBON motifs (KC -> MBON, MBIN a-a KC). ``level_7_cluster``
is the paper's spectral clustering of the connectivity graph, and the
ascending-neuron modalities are "based on connectivity" (Table S1). Predicting
any of those from wiring features re-derives the definition, so they are
**rejected as targets** (``REJECTED_TARGETS``) and never used as features.

The targets kept here are defined by anatomy, not by brain connectivity:

* ``l1em_io_class`` -- brain input/output class. ``sensory`` (enters through
  a nerve), ``ascending`` (enters from the VNC), ``dn_sez`` / ``dn_vnc`` /
  ``rgn`` (axon output in the SEZ / VNC / ring gland; Fig. S3C) and
  ``interneuron`` for every other class. Caveat: S2 collapses overlapping
  classes by priority ([SN, AN, LN, MBIN, KC, MBON] > [PN, DN-VNC] > MB-FBN >
  LHN > CN > [DN-SEZ, PN-somato] > [RGN, MB-FFN] > pre-DN-VNC > pre-DN-SEZ), so
  a DN-SEZ or RGN that also meets an interneuron motif of higher priority is
  labelled ``interneuron`` here (label noise, not leakage).
* ``l1em_sensory_modality`` -- modality of brain sensory neurons
  (``additional_annotations``; "previously described for sensory neurons",
  i.e. by sense organ / nerve). thermo-cold + thermo-warm are merged; classes
  with fewer than ``min_class_neurons`` neurons are dropped (recorded).
* ``connectivity_tier`` -- the legacy label (total synapses >= the 0.75
  quantile), rebuilt with the legacy builder, under the registered
  connectivity exclusions (every ``degree__*`` column, counts, totals), and
  without the spectral embedding (its norm tracks degree). An audit fails
  closed if any remaining feature has |Spearman rho| >= 0.9 with the total.

Partner categories for wiring features never use a connectivity-defined
class: ``l1em_io_class`` uses its own label as the partner category, masked
for val + test nodes (``plan_grouped_split`` -> ``mask_category_ids``, the
foundation's required pattern); ``l1em_sensory_modality`` and
``connectivity_tier`` use the io class (not their target, nothing masked).

Every evaluation goes through ``flybrain_model_eval.run_evaluation`` (grouped
split by homologous pair, trivial baselines, tuning by grouped CV inside
train, calibrated probabilities, one test pass, bootstrap CIs, label-shuffle
and random-split controls, family ablation). Because the dataset is small, each
target is also re-run on ``n_repeats`` extra split seeds and the spread is
reported (``repeats.json``).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import flybrain_brain_cluster_l1em_samples as l1s
import flybrain_learners as fl
import flybrain_model_eval as fme
import flybrain_target_registry as ftr
import flybrain_wiring_features as fwf
from flybrain_l1em_adapter import L1EM_DATASET_SYMBOL, load_l1em_snapshot

TARGET_IO_CLASS = "l1em_io_class"
TARGET_SENSORY_MODALITY = "l1em_sensory_modality"
TARGET_CONNECTIVITY = "connectivity_tier"
L1EM_TARGETS: tuple[str, ...] = (TARGET_IO_CLASS, TARGET_SENSORY_MODALITY, TARGET_CONNECTIVITY)
DEFAULT_REPORT_ROOT = fme.DEFAULT_REPORT_ROOT
DEFAULT_CACHE_ROOT = l1s.L1EM_DEFAULT_CACHE_ROOT
DEFAULT_SNAPSHOT_ROOT = "/mnt/f/.flybrain/snapshots/l1em/catmaid_l1em"
PAIR_GROUP_KEYS: tuple[str, ...] = ("split_group",)
CLUSTER_GROUP_KEYS: tuple[str, ...] = ("split_group", "level_7_cluster")
IO_CLASS_OF_CELLTYPE: Mapping[str, str] = {
    "sensory": "sensory",
    "ascending": "ascending",
    "DN-SEZ": "dn_sez",
    "DN-VNC": "dn_vnc",
    "RGN": "rgn",
}
IO_INTERNEURON = "interneuron"
# Classes whose S2 label is a connectivity-motif definition (Fig. S4C, S10).
CONNECTIVITY_DEFINED_CELLTYPES: frozenset[str] = frozenset({
    "PN", "PN-somato", "LN", "LHN", "CN", "MB-FBN", "MB-FFN", "pre-DN-SEZ", "pre-DN-VNC", "KC", "MBON", "MBIN",
})
REJECTED_TARGETS: Mapping[str, str] = {
    "celltype": "Fig. S4C defines every interneuron class (PN, PN-somato, LN, LHN, CN, MB-FBN, MB-FFN, pre-DN-SEZ, "
                "pre-DN-VNC, KC/MBON motifs) by connectivity to other classes; a wiring model would re-derive it.",
    "level_7_cluster": "spectral clustering of the connectivity graph (Fig. S8); circular for any wiring model.",
    "ascending_modality": "Table S1: ascending-neuron modalities are 'based on connectivity'.",
    "neurotransmitter_dominance": "no per-neuron NT annotation in the release.",
    "region_specialization_tier": "no per-synapse larval neuropil assignment in the release.",
}
MODALITY_MERGE: Mapping[str, str] = {"thermo-cold": "thermo", "thermo-warm": "thermo"}
PROXY_RHO_LIMIT = 0.9

_ANNOTATION_PATTERNS = ("*celltype*", "*annotation*", "*modality*", "*cluster*", "*io_class*", "*pair_skid*",
                        "*hemisphere*", "*skid*")


def _register(objective: str, extra: Sequence[str], reason: str) -> None:
    if objective in fwf.registered_objectives():
        return
    base = fwf.objective_exclusions("cell_class").patterns
    fwf.register_objective_exclusions(objective, tuple(base) + tuple(extra), reason=reason)


_register(TARGET_IO_CLASS, _ANNOTATION_PATTERNS,
          "label is the S2 celltype collapsed to anatomical input/output classes; celltype, annotation and "
          "cluster fields (and any cell-type hierarchy level) determine it")
_register(TARGET_SENSORY_MODALITY, _ANNOTATION_PATTERNS + ("*organ*", "*_order*"),
          "label is the S2 additional_annotations modality of sensory neurons; annotation, celltype and "
          "cluster fields determine or near-copy it")


# =========================================================================== specs


@dataclass(frozen=True)
class TargetSpec:
    name: str
    partner_category: str  # node-table column used as partner category in wiring features
    mask_partner_category: bool  # True when the partner category is the target itself
    two_hop: bool
    structured_families: tuple[str, ...]
    calibration: str
    hgb_min_leaf: int
    description: str
    label_provenance: str = ""  # R1; must equal flybrain_target_registry (checked at import)


TARGET_SPECS: Mapping[str, TargetSpec] = {
    # io_class: larval S2 types are connectivity-defined (R1); whether the collapsed io classes are is
    # open (SYNTHESIS E3), so the lane is reported as recovery of connectivity-derived annotations.
    TARGET_IO_CLASS: TargetSpec(
        TARGET_IO_CLASS, "io_class", True, True, ("etype", "cmpt", "ase"), "temperature", 10,
        "anatomical brain input/output class (sensory / ascending / DN-SEZ / DN-VNC / RGN / interneuron)",
        ftr.LabelProvenance.CONNECTIVITY_DEFINED.value),
    TARGET_SENSORY_MODALITY: TargetSpec(
        TARGET_SENSORY_MODALITY, "io_class", False, False, ("etype", "cmpt", "ase"), "temperature", 5,
        "sense-organ modality of brain sensory neurons", ftr.LabelProvenance.CURATED_MORPHOLOGY.value),
    TARGET_CONNECTIVITY: TargetSpec(
        TARGET_CONNECTIVITY, "io_class", False, True, ("etype", "cmpt"), "isotonic", 20,
        "legacy connectivity_tier (total synapses >= q0.75), degree proxies excluded",
        ftr.LabelProvenance.CONNECTIVITY_DEFINED.value),
}
ftr.check_module_provenance(L1EM_DATASET_SYMBOL, {k: v.label_provenance for k, v in TARGET_SPECS.items()})


def io_class(celltype: str) -> str:
    return IO_CLASS_OF_CELLTYPE.get(str(celltype), IO_INTERNEURON)


def sensory_modality(annotation: str) -> str:
    value = str(annotation).strip()
    return MODALITY_MERGE.get(value, value)


def default_models(target: str) -> tuple[fme.ModelSpec, ...]:
    spec = TARGET_SPECS[target]
    leaf = spec.hgb_min_leaf
    return (
        fme.ModelSpec("nb", ({"alpha": 1.0}, {"alpha": 0.3}), spec.calibration, "nb"),
        fme.ModelSpec("logreg", ({"C": 0.03}, {"C": 0.1}, {"C": 0.3}, {"C": 1.0}), spec.calibration, "logreg"),
        fme.ModelSpec("hgb", (
            {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": leaf, "l2_regularization": 1.0},
            {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 2 * leaf, "l2_regularization": 0.0},
        ), spec.calibration, "hgb"),
    )


# =========================================================================== context


@dataclass
class L1emContext:
    """Snapshot, matrices, node table, edge source and label-free features, loaded once per run."""

    snapshot: Any
    matrices: l1s.L1emMatrices
    nodes: pd.DataFrame
    edges: Any
    structured: pd.DataFrame  # every family, index = skid
    cache_root: str | None
    snapshot_root: str | None
    storage_root: str | None

    @classmethod
    def load(cls, *, snapshot_root: str | None = DEFAULT_SNAPSHOT_ROOT, storage_root: str | None = None,
             cache_root: str | None = DEFAULT_CACHE_ROOT, verify_integrity: bool = True) -> "L1emContext":
        snapshot = load_l1em_snapshot(storage_root, snapshot_root=snapshot_root, verify_integrity=verify_integrity)
        matrices = l1s.load_l1em_matrices(snapshot)
        nodes = l1s.l1em_node_table(snapshot, matrices)
        nodes["io_class"] = [io_class(c) for c in nodes["celltype"]]
        edges = l1s.export_l1em_edge_table(matrices, cache_root=cache_root or DEFAULT_CACHE_ROOT)
        structured = l1s.structured_features(nodes, matrices)
        return cls(snapshot, matrices, nodes, edges, structured, cache_root, snapshot_root, storage_root)


# =========================================================================== samples per target


def target_frame(target: str, ctx: L1emContext, *, min_total_synapses: int = 10,
                 min_class_neurons: int = 10) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rows = samples (sample_id, skid, label, split_group, level_7_cluster, total_synapses) + label notes."""
    nodes = ctx.nodes
    if target == TARGET_CONNECTIVITY:
        payload = l1s.build_l1em_training_samples(
            TARGET_CONNECTIVITY,
            l1s.L1emSampleBuildConfig(storage_root=ctx.storage_root, snapshot_root=ctx.snapshot_root,
                                      min_total_synapses=min_total_synapses))
        legacy = pd.DataFrame({
            "skid": [int(s["metadata"]["skid"]) for s in payload["samples"]],
            "label": [s["expected_label"] for s in payload["samples"]],
            "legacy_text": [s["input_text"] for s in payload["samples"]],
        })
        frame = legacy.merge(nodes, on="skid", how="left", validate="one_to_one")
        if frame["sample_id"].isna().any():
            raise ValueError("legacy connectivity samples reference skids missing from the node table")
        notes = {"label_definition": payload["metadata"]["label_definition"],
                 "high_connectivity_threshold": payload["metadata"]["high_connectivity_threshold"],
                 "legacy_input_fingerprint": payload["metadata"].get("input_fingerprint")}
    else:
        frame = nodes[nodes["total_synapses"] >= int(min_total_synapses)].copy()
        if target == TARGET_IO_CLASS:
            frame["label"] = frame["io_class"]
            notes = {"label_definition": "S2 celltype -> {sensory, ascending, dn_sez, dn_vnc, rgn}; every other "
                                         "class -> interneuron",
                     "label_caveat": "S2 priority-collapses overlapping classes; DN-SEZ/RGN neurons that also meet a "
                                     "higher-priority interneuron motif are labelled interneuron"}
        elif target == TARGET_SENSORY_MODALITY:
            frame = frame[frame["celltype"] == "sensory"].copy()
            frame["label"] = [sensory_modality(a) for a in frame["annotation"]]
            counts = frame["label"].value_counts()
            dropped = sorted(counts[counts < int(min_class_neurons)].index.tolist())
            frame = frame[~frame["label"].isin(dropped)].copy()
            notes = {"label_definition": "S2 additional_annotations of celltype == sensory (sense organ / nerve); "
                                         "thermo-cold + thermo-warm merged",
                     "dropped_small_classes": {k: int(counts[k]) for k in dropped},
                     "min_class_neurons": int(min_class_neurons)}
        else:
            raise ValueError(f"unknown l1em target {target!r} (supported: {', '.join(L1EM_TARGETS)})")
    frame = frame.sort_values("skid", kind="stable").reset_index(drop=True)
    notes.update({"min_total_synapses": int(min_total_synapses), "n_samples": int(len(frame)),
                  "label_counts": {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()}})
    return frame, notes


def binned_text(features: pd.DataFrame, train_rows: Sequence[int], *, n_bins: int = 5) -> list[str]:
    """``key key_q<b>`` tokens per feature; quantile edges fitted on ``train_rows`` only.

    The value token carries its key because the NB tokenizer splits on
    whitespace; the text-view baselines read the same ``key value`` pairs.
    """
    columns = sorted(features.columns)
    train_rows = np.asarray(train_rows, dtype=np.int64)
    parts: list[list[str]] = []
    for name in columns:
        col = features[name]
        if pd.api.types.is_numeric_dtype(col.dtype):
            x = col.to_numpy(dtype=np.float64)
            fit = x[train_rows]
            fit = fit[np.isfinite(fit)]
            edges = np.unique(np.quantile(fit, np.linspace(0, 1, n_bins + 1)[1:-1])) if len(fit) else np.zeros(0)
            bins = np.searchsorted(edges, x, side="right")
            parts.append([f"{name} {name}_q{int(b)}" if np.isfinite(v) else f"{name} {name}_qna"
                          for v, b in zip(x, bins)])
        else:
            parts.append([f"{name} {name}_{fwf.slug(v)}" if pd.notna(v) else f"{name} {name}_na" for v in col.tolist()])
    return [" ".join(row) for row in zip(*parts)] if parts else [""] * len(features)


def proxy_audit(features: pd.DataFrame, totals: Sequence[float], *, limit: float = PROXY_RHO_LIMIT) -> dict[str, Any]:
    """|Spearman rho| of every numeric feature with the label quantity; fail closed at ``limit``."""
    t = pd.Series(np.asarray(totals, dtype=np.float64)).rank()
    rhos = {}
    for name in features.columns:
        col = features[name]
        if not pd.api.types.is_numeric_dtype(col.dtype):
            continue
        x = pd.Series(col.to_numpy(dtype=np.float64))
        ok = x.notna().to_numpy()
        if ok.sum() < 3 or x[ok].nunique() < 2:
            continue
        rho = float(np.corrcoef(x[ok].rank().to_numpy(), t[ok].rank().to_numpy())[0, 1])
        if math.isfinite(rho):
            rhos[name] = rho
    ranked = sorted(rhos.items(), key=lambda kv: -abs(kv[1]))
    offenders = [name for name, rho in ranked if abs(rho) >= limit]
    if offenders:
        raise fwf.LabelLeakageError(f"features are near-monotone proxies of total synapses (|rho| >= {limit}): "
                                    f"{', '.join(offenders)}")
    return {"limit": limit, "max_abs_rho": abs(ranked[0][1]) if ranked else 0.0,
            "top": [{"feature": k, "spearman_rho": round(v, 4)} for k, v in ranked[:8]]}


def build_eval_dataset(target: str, ctx: L1emContext, config: fme.EvalConfig, *,
                       group_keys: Sequence[str] = PAIR_GROUP_KEYS, min_total_synapses: int = 10,
                       min_class_neurons: int = 10, text_mode: str = "binned") -> fme.EvalDataset:
    """Samples + features for one target and one split seed (features masked for that exact split)."""
    spec = TARGET_SPECS[target]
    frame, notes = target_frame(target, ctx, min_total_synapses=min_total_synapses,
                                min_class_neurons=min_class_neurons)
    group_values = frame[list(group_keys)].to_dict(orient="records")
    plan = fme.plan_grouped_split(frame["sample_id"].tolist(), group_values, group_keys, config)
    heldout = set(plan["val"]) | set(plan["test"])
    mask = [int(s) for s, sid in zip(frame["skid"], frame["sample_id"]) if sid in heldout] \
        if spec.mask_partner_category else []
    wiring = l1s.l1em_wiring_features(objective=target, edges=ctx.edges, nodes=ctx.nodes,
                                      category_column=spec.partner_category, two_hop=spec.two_hop,
                                      mask_skids=mask, cache_root=ctx.cache_root)
    wf = wiring.frame.set_index(fwf.NODE_ID_COLUMN)
    wf.index = wf.index.astype(int)
    families = [c for c in ctx.structured.columns if fwf.feature_family(c).split("_")[0] in spec.structured_families]
    features = pd.concat([wf.loc[frame["skid"]].reset_index(drop=True),
                          ctx.structured.loc[frame["skid"], families].reset_index(drop=True)], axis=1)
    kept, dropped_extra = fwf.apply_objective_exclusions(features, target)
    features = kept.reindex(sorted(kept.columns), axis=1)
    fwf.assert_features_allowed(features.columns, target)
    audit = proxy_audit(features, frame["total_synapses"]) if target == TARGET_CONNECTIVITY else None

    train_pos = frame.index[frame["sample_id"].isin(set(plan["train"]))].to_numpy()
    if text_mode == "binned":
        text = binned_text(features, train_pos)
    elif text_mode == "legacy":
        if "legacy_text" not in frame.columns:
            raise ValueError("legacy text exists only for connectivity_tier")
        text = frame["legacy_text"].tolist()
    else:
        raise ValueError("text_mode must be 'binned' or 'legacy'")
    data_frame = pd.concat([frame[["sample_id", "label", *group_keys]].reset_index(drop=True), features], axis=1)
    data_frame["__text__"] = text
    notes = {
        **notes,
        "target_description": spec.description,
        **ftr.provenance_notes(L1EM_DATASET_SYMBOL, target),
        "partner_category": spec.partner_category,
        "partner_category_masked_for_val_test": bool(spec.mask_partner_category),
        "masked_nodes": len(mask),
        "masked_split_ids_sha256": fme.split_ids_sha256(plan),
        "two_hop": spec.two_hop,
        "structured_families": list(spec.structured_families),
        "wiring_fingerprint": wiring.fingerprint,
        "wiring_cache_path": wiring.cache_path,
        "wiring_excluded_features": list(wiring.excluded_features),
        "excluded_after_join": dropped_extra,
        "text_mode": text_mode,
        "proxy_audit": audit,
        "manifest_sha256": ctx.snapshot.manifest_sha256,
        "edge_table": dict(ctx.edges.provenance),
        "rejected_targets": dict(REJECTED_TARGETS),
        "citation": ctx.snapshot.citation,
        "license": ctx.snapshot.license_spdx,
    }
    data = fme.EvalDataset.from_frame(
        data_frame, dataset=L1EM_DATASET_SYMBOL, target=target, id_column="sample_id", label_column="label",
        feature_columns=list(features.columns), group_columns=list(group_keys), text_column="__text__", notes=notes)
    data.aux = fme.size_side_aux(data.features, size_frame=wiring.size_frame, id_column=fwf.NODE_ID_COLUMN,
                                 ids=frame["skid"].tolist(),
                                 side=frame["hemisphere"].tolist() if "hemisphere" in frame.columns else None)
    data.__post_init__()  # re-validate aux against the features
    return data


def granularity_probe(data: fme.EvalDataset, config: fme.EvalConfig, *, backend: str = "hgb",
                      params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Val-only diagnostic for connectivity_tier: how much of the signal is sampling granularity?

    Scale-free fractions still leak a little count information through
    quantization (a neuron with 12 synapses has fractions in steps of 1/12 and
    many exact zeros). The probe fits ``backend`` on (a) only the per-row
    counts of exact-0 / exact-1 / missing fractions, (b) those plus the
    smallest positive fraction (~ 1 / count), (c) each feature family alone and
    (d) all features minus each family, all scored on **val** (test untouched).
    """
    params = dict(params or {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 20,
                             "l2_regularization": 1.0})
    plan = fme.plan_grouped_split(list(data.sample_ids), list(data.group_values), data.group_keys, config)
    pos = {sid: i for i, sid in enumerate(data.sample_ids)}
    tr = np.asarray(sorted(pos[s] for s in plan["train"]))
    va = np.asarray(sorted(pos[s] for s in plan["val"]))
    groups = np.asarray(["|".join(str(data.group_values[i].get(k)) for k in data.group_keys) for i in tr],
                        dtype=object)
    num = data.features.select_dtypes("number")
    y = data.labels

    def score(frame: pd.DataFrame) -> float:
        learner = fl.make_learner(backend, seed=config.seed, **params).fit(frame.iloc[tr], y[tr], groups=groups)
        return fme.accuracy(y[va].tolist(), learner.predict(frame.iloc[va]))

    sparsity = pd.DataFrame({"n_zero": (num == 0).sum(axis=1), "n_one": (num == 1).sum(axis=1),
                             "n_missing": num.isna().sum(axis=1)})
    with_min = sparsity.assign(min_positive_fraction=num.where(num > 0).min(axis=1))
    families = sorted({fwf.feature_family(c) for c in num.columns})
    return {
        "evaluated_on": "val", "backend": backend, "params": params,
        "majority_val": fme.accuracy(y[va].tolist(), [fme._majority(y[tr].tolist())] * len(va)),
        "sparsity_counts_only": score(sparsity),
        "sparsity_plus_min_positive_fraction": score(with_min),
        "all_features": score(num),
        "only_family": {f: score(num[[c for c in num.columns if fwf.feature_family(c) == f]]) for f in families},
        "drop_family": {f: score(num[[c for c in num.columns if fwf.feature_family(c) != f]]) for f in families},
    }


# =========================================================================== runner


def eval_config(*, split_seed: str = "flybrain-real-models-v1", report_root: str | None = DEFAULT_REPORT_ROOT,
                run_label: str = "", models: Sequence[fme.ModelSpec] = (), full: bool = True,
                n_threads: int = 8) -> fme.EvalConfig:
    return fme.EvalConfig(
        models=tuple(models), split_seed=split_seed, cv_folds=5, tune_metric="log_loss", seed=0,
        n_bootstrap=1000 if full else 500, min_margin=0.01, shuffle_control=True, random_split_control=full,
        ablation=full, n_threads=n_threads, report_root=report_root, run_label=run_label, save_models=full)


def run_target(target: str, ctx: L1emContext, *, report_root: str | None = DEFAULT_REPORT_ROOT, n_repeats: int = 4,
               base_seed: str = "flybrain-real-models-v1", cluster_grouped: bool = True,
               legacy_text: bool = True) -> dict[str, Any]:
    """Primary run (full controls, saved models) + repeated split seeds + optional robustness runs."""
    models = default_models(target)
    out: dict[str, Any] = {"target": target}
    primary_cfg = eval_config(split_seed=base_seed, report_root=report_root, models=models)
    primary = ftr.run_gated_evaluation(build_eval_dataset(target, ctx, primary_cfg), primary_cfg)
    out["primary"] = {"summary": primary["summary"], "best_on_val": primary["best_on_val"],
                      "ablation": primary.get("ablation"), "split": primary["split"]}
    repeats = []
    for i in range(1, int(n_repeats) + 1):
        cfg = eval_config(split_seed=f"{base_seed}-l1em-r{i}", report_root=report_root, run_label=f"repeat-{i}",
                          models=models, full=False)
        try:
            rep = ftr.run_gated_evaluation(build_eval_dataset(target, ctx, cfg), cfg)
            repeats.append({"split_seed": cfg.split_seed, "summary": rep["summary"]})
        except ValueError as exc:  # e.g. an empty split on a tiny target
            repeats.append({"split_seed": cfg.split_seed, "error": str(exc)})
    out["repeats"] = repeats
    out["repeat_stats"] = repeat_stats([primary["summary"]] + [r["summary"] for r in repeats if "summary" in r])
    if cluster_grouped and target != TARGET_CONNECTIVITY:
        cfg = eval_config(split_seed=base_seed, report_root=report_root, run_label="cluster-grouped",
                          models=models, full=False)
        try:
            rep = ftr.run_gated_evaluation(build_eval_dataset(target, ctx, cfg, group_keys=CLUSTER_GROUP_KEYS), cfg)
            out["cluster_grouped"] = {"summary": rep["summary"], "split": rep["split"]}
        except ValueError as exc:
            out["cluster_grouped"] = {"error": str(exc)}
    if target == TARGET_CONNECTIVITY:
        probe_cfg = eval_config(split_seed=base_seed, report_root=None, models=models, full=False)
        out["granularity_probe"] = granularity_probe(build_eval_dataset(target, ctx, probe_cfg), probe_cfg)
    if legacy_text and target == TARGET_CONNECTIVITY:
        cfg = eval_config(split_seed=base_seed, report_root=report_root, run_label="legacy-text",
                          models=(fme.ModelSpec("nb", ({"alpha": 1.0},), "isotonic", "nb_legacy_text"),), full=False)
        rep = ftr.run_gated_evaluation(build_eval_dataset(target, ctx, cfg, text_mode="legacy"), cfg)
        out["legacy_text"] = {"summary": rep["summary"]}
    return out


def repeat_stats(summaries: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Mean / sd / min / max over split seeds of accuracy, gain over best trivial, macro-F1 and gate passes."""
    by_model: dict[str, list[Mapping[str, Any]]] = {}
    for rows in summaries:
        for row in rows:
            by_model.setdefault(str(row["model"]), []).append(row)
    stats = {}
    for model, rows in sorted(by_model.items()):
        def agg(values: list[float]) -> dict[str, float]:
            arr = np.asarray(values, dtype=np.float64)
            return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "min": float(arr.min()), "max": float(arr.max())}
        stats[model] = {
            "n_seeds": len(rows),
            "model_acc": agg([r["model_acc"] for r in rows]),
            "best_trivial": agg([r["best_trivial"] for r in rows]),
            "gain_over_best_trivial": agg([r["model_acc"] - r["best_trivial"] for r in rows]),
            "majority": agg([r["majority"] for r in rows]),
            "macro_f1": agg([r["macro_f1"] for r in rows]),
            "ece": agg([r["ece"] for r in rows]),
            "shuffle_acc": agg([r["shuffle_acc"] for r in rows if r.get("shuffle_acc") is not None] or [float("nan")]),
            "gate_passes": int(sum(r["gate"] == "pass" for r in rows)),
        }
    return stats


def run_all(*, targets: Sequence[str] = L1EM_TARGETS, report_root: str = DEFAULT_REPORT_ROOT,
            cache_root: str = DEFAULT_CACHE_ROOT, snapshot_root: str = DEFAULT_SNAPSHOT_ROOT, n_repeats: int = 4,
            verify_integrity: bool = True) -> dict[str, Any]:
    started = time.time()
    ctx = L1emContext.load(snapshot_root=snapshot_root, cache_root=cache_root, verify_integrity=verify_integrity)
    results = {t: run_target(t, ctx, report_root=report_root, n_repeats=n_repeats) for t in targets}
    payload = {
        "schema_version": "flybrain-l1em-real-models/v1",
        "dataset": L1EM_DATASET_SYMBOL,
        "manifest_sha256": ctx.snapshot.manifest_sha256,
        "rejected_targets": dict(REJECTED_TARGETS),
        "results": results,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    out = Path(report_root) / L1EM_DATASET_SYMBOL
    l1s._refuse_snapshot_path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "l1em_real_models.json").write_text(
        json.dumps(fl._jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (out / "SUMMARY.md").write_text(render_summary(payload), encoding="utf-8")
    return payload


def _f(value: Any) -> str:
    return "-" if value is None or (isinstance(value, float) and not math.isfinite(value)) else f"{value:.3f}"


def render_summary(payload: Mapping[str, Any]) -> str:
    lines = ["# l1em real models", "", f"manifest_sha256 `{payload['manifest_sha256']}`", "",
             "Rejected targets (circular or unavailable):", ""]
    lines += [f"- `{k}`: {v}" for k, v in payload["rejected_targets"].items()]
    for target, res in payload["results"].items():
        lines += ["", f"## {target}", "", "Primary split (pair-grouped):", "",
                  "| model | majority | best trivial (rule) | acc [95% CI] | macro-F1 | ECE | shuffle | gate |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in res["primary"]["summary"]:
            lines.append(f"| {r['model']} | {_f(r['majority'])} | {_f(r['best_trivial'])} ({r['best_trivial_rule']}) | "
                         f"{_f(r['model_acc'])} {r['model_ci']} | {_f(r['macro_f1'])} | {_f(r['ece'])} | "
                         f"{_f(r['shuffle_acc'])} | {r['gate']} |")
        lines += ["", "Across split seeds (primary + repeats):", "",
                  "| model | seeds | acc mean +- sd | gain over best trivial mean [min, max] | macro-F1 mean | gate passes |",
                  "|---|---|---|---|---|---|"]
        for model, s in res["repeat_stats"].items():
            g = s["gain_over_best_trivial"]
            lines.append(f"| {model} | {s['n_seeds']} | {_f(s['model_acc']['mean'])} +- {_f(s['model_acc']['sd'])} | "
                         f"{g['mean']:+.3f} [{g['min']:+.3f}, {g['max']:+.3f}] | {_f(s['macro_f1']['mean'])} | "
                         f"{s['gate_passes']}/{s['n_seeds']} |")
        for key, title in (("cluster_grouped", "Grouped by pair + level-7 cluster"), ("legacy_text", "Legacy text NB")):
            if key in res:
                lines += ["", f"{title}:", ""]
                if "error" in res[key]:
                    lines.append(f"- error: {res[key]['error']}")
                else:
                    for r in res[key]["summary"]:
                        lines.append(f"- {r['model']}: acc {_f(r['model_acc'])} {r['model_ci']} vs best trivial "
                                     f"{_f(r['best_trivial'])} ({r['best_trivial_rule']}), gate {r['gate']}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="l1em real-model evaluation (reports under --report-root/l1em/)")
    parser.add_argument("--targets", nargs="*", default=list(L1EM_TARGETS), choices=list(L1EM_TARGETS))
    parser.add_argument("--report-root", default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--snapshot-root", default=DEFAULT_SNAPSHOT_ROOT)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--no-verify", action="store_true", help="skip sha256 re-hash (sizes are still checked)")
    args = parser.parse_args(argv)
    payload = run_all(targets=args.targets, report_root=args.report_root, cache_root=args.cache_root,
                      snapshot_root=args.snapshot_root, n_repeats=args.repeats, verify_integrity=not args.no_verify)
    for target, res in payload["results"].items():
        for row in res["primary"]["summary"]:
            print(json.dumps(fl._jsonable(row), sort_keys=True))
    return 0


__all__ = [
    "CONNECTIVITY_DEFINED_CELLTYPES",
    "IO_CLASS_OF_CELLTYPE",
    "L1EM_TARGETS",
    "L1emContext",
    "REJECTED_TARGETS",
    "TARGET_CONNECTIVITY",
    "TARGET_IO_CLASS",
    "TARGET_SENSORY_MODALITY",
    "TARGET_SPECS",
    "TargetSpec",
    "binned_text",
    "build_eval_dataset",
    "default_models",
    "eval_config",
    "granularity_probe",
    "io_class",
    "proxy_audit",
    "repeat_stats",
    "run_all",
    "run_target",
    "sensory_modality",
    "target_frame",
]

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
