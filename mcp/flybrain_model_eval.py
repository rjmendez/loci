"""One evaluation harness for FlyBrain brain-cluster learners (roadmap B1/B2/AC6).

``run_evaluation(data, config)``:

1. **Leakage check (fail closed).** Every feature column, and every key in
   ``input_text`` when a text model is evaluated, must pass
   ``flybrain_wiring_features.assert_features_allowed(names, target)``; a
   feature column identical to the label also fails.
2. **Grouped split** (``grouped_split_ids``: union-find over the dataset's
   ``split_group_keys``) into train / val / test; ``assert_grouped_split``
   re-checks that no group straddles two splits.
3. **Trivial baselines** fitted on train, scored on val and test, for each
   view a model sees: the *feature view* (majority, one-feature threshold
   stump, label->score argmax, one-feature lookup; vectorized, same rules
   and tie-breaks as ``flybrain_brain_cluster_baselines``) and the *text
   view* (``evaluate_trivial_baselines`` on ``input_text``).
4. **Tuning** per model spec: grouped K-fold CV inside train (groups = split
   components) over the param grid; the best params by ``tune_metric``.
5. **Final fit** on train (grouped early stopping inside the learner) and
   probability calibration fitted on the tuning pass's grouped out-of-fold
   probabilities (isotonic / sigmoid / temperature).
6. **Held-out test, once per final config**: accuracy and macro-F1 with
   grouped (cluster) bootstrap 95% CIs, paired bootstrap CI of the gain over
   the best trivial rule and over majority, ECE + reliability table (raw and
   calibrated), log loss, Brier, confusion matrix, per-class P/R/F1.
7. **Controls**: label-shuffle (train labels permuted, refit, scored on
   test; must collapse to the majority rate), random-vs-grouped split gap,
   and a drop-one-family feature ablation for the best-on-val model, scored
   on **val** (the test split is not reused for ablations).
8. **Gate** per model (``flybrain_eval_stats.assemble_gate``; any missing
   criterion fails closed): beats the best trivial rule of its view on test
   accuracy by ``min_margin`` (``trivial_baseline_gate``) AND on macro-F1,
   the paired-bootstrap lower bound of the accuracy gain is > 0, it beats
   the **size/degree-only HGB** by the margin with a paired CI above 0 (R3),
   and the **group-permutation null** (>= 100 block permutations of train
   labels over split components) gives p <= ``permutation_alpha`` (R4). The
   single label shuffle is still reported but no longer gates.
9. **Report**: ``report.json`` + ``report.md`` (+ saved model artifacts)
   under ``<report_root>/<dataset>/<target>/``.

Rigor additions (research rules; see ``flybrain_eval_stats``):

* R1: ``notes["label_provenance"]`` in {measured, curated_morphology,
  connectivity_defined, model_predicted}. The gate verdict is computed for
  every target but ``gate_applies`` is True only for measured and
  curated_morphology labels; ``claim`` says what a pass may be called.
* R3: size/degree baseline (columns from features or ``EvalDataset.aux``
  matching ``config.size_patterns``), accuracy by degree decile, and for NT
  targets a binary ACh vs GABA+Glu view, a hemilineage -> NT oracle and a
  two-stage wiring -> hemilineage -> NT model (``report["nt_hooks"]``).
* R4: cluster bootstrap over split components (effective n = groups), the
  group-permutation null, and a graded split curve (random -> type ->
  hemilineage -> hemisphere) for the best-on-val model.
* R6: ``calibration_protocol="grouped_fold"`` (train / calibrate / test),
  equal-width / equal-mass / sweep / debiased L2 / classwise ECE, log loss
  and Brier with cluster-bootstrap CIs, and Dirichlet or top-label
  calibration (``ModelSpec.calibration`` in {"dirichlet", "top_label"}).
* R7: ``config.hierarchy_parent`` compares a parent -> child conditional
  model with the flat model (report only).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import flybrain_brain_cluster_baselines as fbb
import flybrain_eval_stats as es
import flybrain_learners as fl
import flybrain_wiring_features as fwf
from flybrain_brain_cluster_training import (
    DatasetManifest,
    TrainingSample,
    assert_grouped_split,
    deterministic_split_ids,
    grouped_split_ids,
    split_group_components,
)

EVAL_REPORT_SCHEMA_VERSION = "flybrain-model-eval-report/v1"
DEFAULT_REPORT_ROOT = "/mnt/f/.flybrain/logs/real-models-20260924T174122Z"
VIEW_FEATURES = "features"
VIEW_TEXT = "text"


# =========================================================================== data


@dataclass
class EvalDataset:
    """Samples for one (dataset, target). ``features`` holds feature columns only (row i = sample i)."""

    dataset: str
    target: str
    sample_ids: tuple[str, ...]
    labels: np.ndarray
    features: pd.DataFrame
    group_keys: tuple[str, ...]
    group_values: tuple[Mapping[str, Any], ...]
    text: tuple[str, ...] | None = None
    notes: Mapping[str, Any] = field(default_factory=dict)
    # Side-channel columns (row i = sample i) that NO model ever sees: size/degree columns for the
    # size baseline when they are excluded as features, hemilineage for the NT oracle, side for the
    # hemisphere split level, a parent label (super_class) for the hierarchy prototype.
    aux: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        n = len(self.sample_ids)
        if n == 0:
            raise ValueError("EvalDataset is empty")
        if len(set(self.sample_ids)) != n:
            raise ValueError("duplicate sample ids")
        if len(self.labels) != n or len(self.features) != n or len(self.group_values) != n:
            raise ValueError("sample_ids, labels, features and group_values must have equal length")
        if self.text is not None and len(self.text) != n:
            raise ValueError("text length differs from sample count")
        self.labels = np.asarray([str(v) for v in self.labels], dtype=object)
        self.features = self.features.reset_index(drop=True)
        if fl.TEXT_COLUMN in self.features.columns:
            raise ValueError(f"{fl.TEXT_COLUMN} is reserved; pass text separately")
        if self.aux is not None:
            if len(self.aux) != n:
                raise ValueError("aux length differs from sample count")
            self.aux = self.aux.reset_index(drop=True)
            clash = sorted(set(self.aux.columns) & set(self.features.columns))
            if clash:
                raise ValueError(f"aux columns duplicate feature columns: {clash}")

    @classmethod
    def from_samples(cls, samples: Sequence[TrainingSample | Mapping[str, Any]], *, dataset: str, target: str,
                     group_keys: Sequence[str], notes: Mapping[str, Any] | None = None,
                     aux_keys: Sequence[str] = ()) -> "EvalDataset":
        rows = [s.as_dict() if isinstance(s, TrainingSample) else dict(s) for s in samples]
        ids = tuple(str(r["sample_id"]) for r in rows)
        labels = np.asarray([str(r["expected_label"]) for r in rows], dtype=object)
        features = pd.DataFrame.from_records([dict(r.get("features") or {}) for r in rows])
        if len(features) != len(rows):  # every sample has an empty features dict
            features = pd.DataFrame(index=range(len(rows)))
        features = features.reindex(sorted(features.columns), axis=1)
        keys = tuple(group_keys)
        groups = tuple({k: (r.get("metadata") or {}).get(k) for k in keys} for r in rows)
        text = tuple(str(r.get("input_text", "")) for r in rows)
        aux = None
        if aux_keys:
            aux = pd.DataFrame.from_records([{k: (r.get("metadata") or {}).get(k) for k in aux_keys} for r in rows],
                                            columns=list(aux_keys))
        return cls(dataset, target, ids, labels, features, keys, groups, text, dict(notes or {}), aux)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, *, dataset: str, target: str, id_column: str, label_column: str,
                   feature_columns: Sequence[str], group_columns: Sequence[str],
                   text_column: str | None = None, notes: Mapping[str, Any] | None = None,
                   aux_columns: Sequence[str] = ()) -> "EvalDataset":
        overlap = set(feature_columns) & {id_column, label_column}
        if overlap:
            raise ValueError(f"feature_columns include id/label columns: {sorted(overlap)}")
        if label_column in set(aux_columns):
            raise ValueError("aux_columns must not include the label column")
        keys = tuple(group_columns)
        groups = tuple(frame[list(keys)].to_dict(orient="records")) if keys else tuple({} for _ in range(len(frame)))
        return cls(
            dataset, target,
            tuple(str(v) for v in frame[id_column].tolist()),
            frame[label_column].astype(str).to_numpy(dtype=object),
            frame[list(feature_columns)].reset_index(drop=True),
            keys, groups,
            None if text_column is None else tuple(str(v) for v in frame[text_column].tolist()),
            dict(notes or {}),
            frame[list(aux_columns)].reset_index(drop=True) if aux_columns else None,
        )

    def row_metadata(self) -> list[dict[str, Any]]:
        """Group values merged with aux columns (for split-curve levels and oracles); NaN -> None."""
        out = [dict(m) for m in self.group_values]
        if self.aux is not None:
            for col in self.aux.columns:
                for i, v in enumerate(self.aux[col].tolist()):
                    out[i].setdefault(col, None if (isinstance(v, float) and math.isnan(v)) else v)
        return out

    def light_samples(self) -> list[TrainingSample]:
        return [
            TrainingSample(sample_id=sid, region_id="eval", input_text="-", expected_label=str(label),
                           expected_confidence=1.0, provenance_refs=("eval",), metadata=dict(meta))
            for sid, label, meta in zip(self.sample_ids, self.labels, self.group_values)
        ]


HEMISPHERE_AUX = "hemisphere"
TOTAL_DEGREE_AUX = "size__total_degree"
_SIDE_NORMAL = {"left": "left", "l": "left", "lhs": "left", "right": "right", "r": "right", "rhs": "right"}


def normalize_hemisphere(value: Any) -> str:
    """left / right / other (midline, unknown, missing) for the hemisphere split level."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    return _SIDE_NORMAL.get(str(value).strip().lower(), "other")


def size_side_aux(features: pd.DataFrame, *, size_frame: pd.DataFrame | None = None, id_column: str | None = None,
                  ids: Sequence[Any] | None = None, side: Sequence[Any] | None = None,
                  extra: Mapping[str, Sequence[Any]] | None = None) -> pd.DataFrame | None:
    """Row-aligned ``EvalDataset.aux``: size/degree columns not already features, total degree, hemisphere.

    ``size_frame`` is a per-node table (``id_column`` + degree columns, e.g.
    ``WiringFeatureResult.size_frame``) aligned to ``ids``. Size columns that are
    already model features are skipped (aux may not duplicate features);
    ``size__total_degree`` (out + in weight total, raw or rank) is always added
    when both are present so the degree-decile axis is the same for every
    target. None of these ever reaches a model [R3].
    """
    n = len(features)
    out = pd.DataFrame(index=range(n))
    if size_frame is not None:
        if id_column is None or ids is None:
            raise ValueError("size_frame needs id_column and ids")
        table = size_frame.assign(**{id_column: size_frame[id_column].astype(str)}).drop_duplicates(id_column)
        aligned = table.set_index(id_column).reindex([str(i) for i in ids]).reset_index(drop=True)
        if len(aligned) != n:
            raise ValueError("size_frame alignment changed the row count")
        for col in aligned.columns:
            if col not in features.columns and pd.api.types.is_numeric_dtype(aligned[col].dtype):
                out[col] = aligned[col].astype(np.float64).to_numpy()
        for suffix in ("", "_rank"):
            o, i = f"degree__out_weight_total{suffix}", f"degree__in_weight_total{suffix}"
            if o in aligned.columns and i in aligned.columns:
                out[TOTAL_DEGREE_AUX] = (aligned[o].astype(np.float64).fillna(0.0)
                                         + aligned[i].astype(np.float64).fillna(0.0)).to_numpy()
                break
    if side is not None:
        side = list(side)
        if len(side) != n:
            raise ValueError("side length differs from features")
        out[HEMISPHERE_AUX] = [normalize_hemisphere(v) for v in side]
    for name, values in (extra or {}).items():
        values = list(values)
        if len(values) != n:
            raise ValueError(f"aux column {name!r} length differs from features")
        out[name] = values
    out = out[[c for c in out.columns if c not in features.columns]]
    return out if len(out.columns) else None


@dataclass(frozen=True)
class ModelSpec:
    backend: str
    param_grid: tuple[Mapping[str, Any], ...] = ({},)
    calibration: str | None = "isotonic"
    name: str = ""

    @property
    def label(self) -> str:
        return self.name or self.backend


@dataclass(frozen=True)
class EvalConfig:
    models: tuple[ModelSpec, ...] = (ModelSpec("logreg", ({"C": 0.1}, {"C": 1.0}), "isotonic"),
                                     ModelSpec("hgb", ({},), "isotonic"))
    split_seed: str = "flybrain-real-models-v1"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    cv_folds: int = 5
    tune_metric: str = "macro_f1"
    seed: int = 0
    n_bootstrap: int = 1000
    min_margin: float = 0.01
    ece_bins: int = 15
    shuffle_control: bool = True
    shuffle_tolerance: float = 0.02
    random_split_control: bool = True
    ablation: bool = True
    ablation_families: Mapping[str, Sequence[str]] | None = None
    n_threads: int = 8
    report_root: str | None = DEFAULT_REPORT_ROOT
    run_label: str = ""
    save_models: bool = True
    # ---- R6 calibration protocol: "grouped_fold" = base fitted on train minus a grouped calibration
    # fold, calibrator fitted on that fold (train/calibrate/test); "oof" = legacy grouped OOF calibration.
    calibration_protocol: str = "grouped_fold"
    calibration_fraction: float = 0.2
    calibration_bootstrap: int = 200
    # ---- R3 size/degree-only baseline (mandatory gate) + degree-decile table
    size_baseline: bool = True
    require_size_baseline: bool = True
    size_patterns: tuple[str, ...] = es.DEFAULT_SIZE_PATTERNS
    size_baseline_params: Mapping[str, Any] = field(default_factory=lambda: {"max_iter": 200})
    degree_column: str | None = None
    # ---- R4 group-permutation null (replaces the single shuffle in the gate)
    n_permutations: int = 100
    permutation_alpha: float = 0.05
    permutation_n_jobs: int = 4
    require_permutation_null: bool = True
    # ---- R4 graded split curve (None = es.default_split_levels)
    split_curve: bool = True
    split_levels: tuple[es.SplitLevel, ...] | None = None
    # ---- R3 NT hooks (None = on when the target is an NT target) and R7 hierarchy prototype
    nt_hooks: bool | None = None
    hemilineage_column: str = "hemilineage"
    hierarchy_parent: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["models"] = [asdict(m) for m in self.models]
        return fl._jsonable(out)


# =========================================================================== metrics


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Macro-F1 over labels present in ``y_true`` or ``y_pred`` (sklearn default)."""
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    index = {c: i for i, c in enumerate(labels)}
    k = len(labels)
    t = np.asarray([index[v] for v in y_true])
    p = np.asarray([index[v] for v in y_pred])
    cm = np.bincount(t * k + p, minlength=k * k).reshape(k, k)
    return _macro_f1_from_counts(np.diag(cm), cm.sum(axis=0) - np.diag(cm), cm.sum(axis=1) - np.diag(cm))


def _macro_f1_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> Any:
    """Works on (K,) or (B, K) arrays; labels with tp+fp+fn == 0 are not counted."""
    tp, fp, fn = (np.asarray(a, dtype=np.float64) for a in (tp, fp, fn))
    denom = 2 * tp + fp + fn
    present = denom > 0
    f1 = np.where(present, 2 * tp / np.where(present, denom, 1.0), 0.0)
    n_present = present.sum(axis=-1)
    result = np.where(n_present > 0, f1.sum(axis=-1) / np.maximum(n_present, 1), 0.0)
    return float(result) if np.ndim(result) == 0 else result


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(np.asarray(y_true, dtype=object) == np.asarray(y_pred, dtype=object)))


def confusion(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    labels = sorted(set(y_true) | set(y_pred))
    index = {c: i for i, c in enumerate(labels)}
    k = len(labels)
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[index[t], index[p]] += 1
    per_class = {}
    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": int(cm[i, :].sum()), "precision": precision, "recall": recall, "f1": f1}
    return {"labels": labels, "matrix": cm.tolist(), "per_class": per_class}


def grouped_bootstrap(y_true: Sequence[str], preds: Mapping[str, Sequence[str]], components: Sequence[str], *,
                      n_bootstrap: int, seed: int, pairs: Sequence[tuple[str, str]] = ()) -> dict[str, Any]:
    """Cluster bootstrap over split components (correlated neurons resample together).

    Returns 95% percentile CIs for accuracy and macro-F1 of each prediction
    set and for the paired accuracy / macro-F1 difference of each ``(a, b)``.
    """
    comps = sorted(set(components))
    c_index = {c: i for i, c in enumerate(comps)}
    comp_of = np.asarray([c_index[c] for c in components])
    n_c = len(comps)
    labels = sorted(set(y_true).union(*[set(p) for p in preds.values()]))
    l_index = {c: i for i, c in enumerate(labels)}
    k = len(labels)
    t = np.asarray([l_index[v] for v in y_true])
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n_c, np.full(n_c, 1.0 / n_c), size=int(n_bootstrap)).astype(np.float64)
    n_per_c = np.bincount(comp_of, minlength=n_c).astype(np.float64)
    total = weights @ n_per_c
    stats: dict[str, dict[str, np.ndarray]] = {}
    for name, pred in preds.items():
        p = np.asarray([l_index[v] for v in pred])
        correct = np.bincount(comp_of, weights=(p == t).astype(np.float64), minlength=n_c)
        tp_c = np.zeros((n_c, k))
        fp_c = np.zeros((n_c, k))
        fn_c = np.zeros((n_c, k))
        hit = p == t
        np.add.at(tp_c, (comp_of[hit], t[hit]), 1.0)
        np.add.at(fp_c, (comp_of[~hit], p[~hit]), 1.0)
        np.add.at(fn_c, (comp_of[~hit], t[~hit]), 1.0)
        stats[name] = {
            "acc": np.where(total > 0, (weights @ correct) / np.maximum(total, 1.0), 0.0),
            "f1": _macro_f1_from_counts(weights @ tp_c, weights @ fp_c, weights @ fn_c),
        }

    def ci(values: np.ndarray) -> list[float]:
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    out: dict[str, Any] = {"n_bootstrap": int(n_bootstrap), "n_components": n_c, "method": "cluster_bootstrap_by_split_component",
                           "per_prediction": {}, "paired": {}}
    for name, s in stats.items():
        out["per_prediction"][name] = {"accuracy_ci95": ci(s["acc"]), "macro_f1_ci95": ci(s["f1"])}
    for a, b in pairs:
        diff_acc = stats[a]["acc"] - stats[b]["acc"]
        diff_f1 = stats[a]["f1"] - stats[b]["f1"]
        out["paired"][f"{a}-minus-{b}"] = {
            "accuracy_diff_ci95": ci(diff_acc),
            "macro_f1_diff_ci95": ci(diff_f1),
            "p_accuracy_diff_le_0": float(np.mean(diff_acc <= 0)),
        }
    return out


# =========================================================================== feature-view baselines


def _majority(labels: Sequence[str]) -> str:
    counts = pd.Series(list(labels)).value_counts()
    top = counts.max()
    return sorted(counts[counts == top].index)[0]


def _fit_threshold_features(frame: pd.DataFrame, y: np.ndarray, classes: Sequence[str]) -> dict[str, Any] | None:
    """Vectorized ``fbb.fit_threshold_rule`` over numeric columns with no missing train values."""
    if len(set(y.tolist())) < 2:
        return None
    index = {c: i for i, c in enumerate(classes)}
    codes = np.asarray([index[v] for v in y])
    k = len(classes)
    total = np.bincount(codes, minlength=k)
    constant_correct = int(total.max())
    best: tuple[int, str, float, int, int] | None = None
    for name in sorted(frame.columns):
        col = frame[name]
        if not (pd.api.types.is_numeric_dtype(col.dtype) or pd.api.types.is_bool_dtype(col.dtype)):
            continue
        x = col.to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(x)):
            continue
        order = np.argsort(x, kind="stable")
        xs, cs = x[order], codes[order]
        onehot = np.zeros((len(xs), k), dtype=np.int64)
        onehot[np.arange(len(xs)), cs] = 1
        left = np.cumsum(onehot, axis=0)[:-1]
        boundary = xs[1:] != xs[:-1]
        if not boundary.any():
            continue
        right = total[None, :] - left
        correct = left.max(axis=1) + right.max(axis=1)
        correct = np.where(boundary, correct, -1)
        i = int(np.argmax(correct))
        if best is None or int(correct[i]) > best[0]:
            best = (int(correct[i]), name, float((xs[i] + xs[i + 1]) / 2.0),
                    int(np.argmax(right[i])), int(np.argmax(left[i])))
    if best is None or best[0] <= constant_correct:
        return None
    correct, name, threshold, above, below = best
    return {"feature": name, "threshold": threshold, "above_label": classes[above], "below_label": classes[below],
            "train_accuracy": correct / float(len(y))}


def _apply_threshold_features(rule: Mapping[str, Any], frame: pd.DataFrame) -> list[str]:
    x = pd.to_numeric(frame[rule["feature"]], errors="coerce").to_numpy(dtype=np.float64)
    out = np.where(x >= rule["threshold"], rule["above_label"], rule["below_label"]).astype(object)
    out[~np.isfinite(x)] = ""
    return out.tolist()


def _fit_argmax_features(frame: pd.DataFrame, classes: Sequence[str]) -> dict[str, Any] | None:
    numeric = {c for c in frame.columns
               if pd.api.types.is_numeric_dtype(frame[c].dtype) and np.all(np.isfinite(frame[c].to_numpy(dtype=np.float64)))}
    mapping: dict[str, str] = {}
    for label in classes:
        stem = fbb._label_stem(label)
        for suffix in fbb._SCORE_SUFFIXES:
            if f"{stem}{suffix}" in numeric:
                mapping[label] = f"{stem}{suffix}"
                break
    return {"label_to_feature": dict(sorted(mapping.items()))} if len(mapping) >= 2 else None


def _apply_argmax_features(rule: Mapping[str, Any], frame: pd.DataFrame) -> list[str]:
    labels = list(rule["label_to_feature"])
    scores = np.column_stack([pd.to_numeric(frame[rule["label_to_feature"][l]], errors="coerce").to_numpy(dtype=np.float64)
                              for l in labels])
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    best = scores.max(axis=1, keepdims=True)
    out = []
    for row, top in zip(scores, best[:, 0]):
        winners = sorted(labels[j] for j in range(len(labels)) if row[j] == top)
        out.append(winners[0] if np.isfinite(top) else "")
    return out


def _lookup_key(col: pd.Series) -> pd.Series:
    return col.astype(object).where(col.notna(), "__nan__").astype(str)


def _lookup_tables(frame: pd.DataFrame, y: np.ndarray) -> dict[str, dict[str, str]]:
    tables = {}
    ys = pd.Series(y, dtype=object)
    for name in sorted(frame.columns):
        keys = _lookup_key(frame[name])
        if keys.nunique() < 2:
            continue
        counts = pd.crosstab(keys.to_numpy(), ys.to_numpy())
        counts = counts.reindex(sorted(counts.columns), axis=1)
        tables[name] = counts.idxmax(axis=1).to_dict()
    return tables


def feature_view_baselines(train: pd.DataFrame, y_train: np.ndarray, heldout: Mapping[str, tuple[pd.DataFrame, np.ndarray]],
                           ) -> dict[str, Any]:
    """Fit majority / threshold / argmax / lookup on train features; score each held-out split.

    Returns ``{"rules": {...}, "splits": {split: {"best_rule", "best_accuracy", "predictions": {rule: [...]}}}}``.
    Like ``fbb.evaluate_trivial_baselines`` the lookup rule reports the
    feature with the best accuracy *on that held-out split* (a strict bar).
    """
    classes = sorted(set(y_train.tolist()))
    majority = _majority(y_train.tolist())
    threshold = _fit_threshold_features(train, y_train, classes)
    argmax = _fit_argmax_features(train, classes)
    tables = _lookup_tables(train, y_train)
    rules = {
        fbb.RULE_MAJORITY: {"applicable": True, "params": {"label": majority}},
        fbb.RULE_THRESHOLD: {"applicable": threshold is not None, "params": threshold},
        fbb.RULE_ARGMAX: {"applicable": argmax is not None, "params": argmax},
        fbb.RULE_LOOKUP: {"applicable": bool(tables), "params": None if not tables else {"features": sorted(tables)}},
    }
    splits: dict[str, Any] = {}
    for split, (frame, y) in heldout.items():
        truth = y.tolist()
        preds: dict[str, list[str]] = {fbb.RULE_MAJORITY: [majority] * len(truth)}
        if threshold is not None:
            preds[fbb.RULE_THRESHOLD] = _apply_threshold_features(threshold, frame)
        if argmax is not None:
            preds[fbb.RULE_ARGMAX] = _apply_argmax_features(argmax, frame)
        lookup_acc: dict[str, float] = {}
        lookup_pred: dict[str, list[str]] = {}
        for name, table in tables.items():
            keys = _lookup_key(frame[name]).tolist()
            lookup_pred[name] = [table.get(key, majority) for key in keys]
            lookup_acc[name] = accuracy(truth, lookup_pred[name])
        if lookup_acc:
            best_feature = sorted(lookup_acc.items(), key=lambda item: (-item[1], item[0]))[0][0]
            preds[fbb.RULE_LOOKUP] = lookup_pred[best_feature]
        accs = {rule: accuracy(truth, p) for rule, p in preds.items()}
        best_rule = sorted(accs.items(), key=lambda item: (-item[1], item[0]))[0][0]
        splits[split] = {
            "accuracy": accs,
            "macro_f1": {rule: macro_f1(truth, p) for rule, p in preds.items()},
            "lookup_best_feature": None if not lookup_acc else best_feature,
            "lookup_per_feature_accuracy": {k: round(v, 6) for k, v in sorted(lookup_acc.items())},
            "best_rule": best_rule,
            "best_accuracy": accs[best_rule],
            "predictions": preds,
        }
    return {"view": VIEW_FEATURES, "rules": fl._jsonable(rules), "splits": splits}


def features_as_text(frame: pd.DataFrame) -> list[str]:
    """Serialize feature rows as ``key value`` text (the baselines module's input format)."""
    parts = []
    for name in sorted(frame.columns):
        col = frame[name]
        if pd.api.types.is_numeric_dtype(col.dtype):
            values = [repr(float(v)) if pd.notna(v) else "nan" for v in col.tolist()]
        else:
            values = [fwf.slug(v) if pd.notna(v) else "nan" for v in col.tolist()]
        parts.append([f"{name} {v}" for v in values])
    return [" ".join(row) for row in zip(*parts)] if parts else [""] * len(frame)


def text_view_baselines(train_text: Sequence[str], y_train: np.ndarray,
                        heldout: Mapping[str, tuple[Sequence[str], np.ndarray]]) -> dict[str, Any]:
    """``fbb.evaluate_trivial_baselines`` on ``input_text`` plus the winning rule's predictions."""
    train_rows = list(zip(train_text, y_train.tolist()))
    splits = {}
    for split, (texts, y) in heldout.items():
        rows = list(zip(texts, y.tolist()))
        result = fbb.evaluate_trivial_baselines(train_rows, rows)
        preds: dict[str, list[str]] = {}
        majority = result["rules"][fbb.RULE_MAJORITY]["params"]["label"]
        preds[fbb.RULE_MAJORITY] = [majority] * len(rows)
        if result["rules"][fbb.RULE_THRESHOLD]["applicable"]:
            params = result["rules"][fbb.RULE_THRESHOLD]["params"]
            preds[fbb.RULE_THRESHOLD] = [fbb.apply_threshold_rule(params, t) or "" for t in texts]
        if result["rules"][fbb.RULE_ARGMAX]["applicable"]:
            params = result["rules"][fbb.RULE_ARGMAX]["params"]
            preds[fbb.RULE_ARGMAX] = [fbb.apply_argmax_rule(params, t) or "" for t in texts]
        if result["rules"][fbb.RULE_LOOKUP]["applicable"]:
            feature = result["rules"][fbb.RULE_LOOKUP]["params"]["feature"]
            table = fbb.fit_lookup_rules(train_rows)[feature]
            preds[fbb.RULE_LOOKUP] = [fbb.apply_lookup_rule(table, feature, t) for t in texts]
        truth = y.tolist()
        splits[split] = {
            "accuracy": {rule: float(r["heldout_accuracy"]) for rule, r in result["rules"].items() if r["applicable"]},
            "macro_f1": {rule: macro_f1(truth, p) for rule, p in preds.items()},
            "best_rule": result["best_rule"],
            "best_accuracy": float(result["best_accuracy"]),
            "predictions": preds,
            "module_result": {k: v for k, v in result.items() if k != "rules"} | {
                "rules": {rule: {kk: vv for kk, vv in r.items()} for rule, r in result["rules"].items()}},
        }
    return {"view": VIEW_TEXT, "splits": splits}


# =========================================================================== leakage


def check_leakage(data: EvalDataset, *, uses_text: bool) -> dict[str, Any]:
    fwf.assert_features_allowed(list(data.features.columns), data.target)
    identical = [c for c in data.features.columns
                 if data.features[c].astype(str).tolist() == data.labels.tolist()]
    if identical:
        raise fwf.LabelLeakageError(f"feature columns identical to the label: {identical}")
    text_keys: list[str] = []
    if uses_text:
        if data.text is None:
            raise ValueError("a text model was requested but the dataset has no input_text")
        keys: set[str] = set()
        for text in data.text:
            keys.update(fbb.parse_input_features(text))
        text_keys = sorted(keys)
        fwf.assert_features_allowed(text_keys, data.target)
    rule = fwf.objective_exclusions(data.target)
    return {"checked_feature_columns": len(data.features.columns), "checked_text_keys": text_keys,
            "exclusion_rule": rule.as_dict(), "global_patterns": list(fwf.GLOBAL_EXCLUDED_PATTERNS), "pass": True}


# =========================================================================== harness


def _light(sample_ids: Sequence[str], group_values: Sequence[Mapping[str, Any]]) -> list[TrainingSample]:
    return [TrainingSample(sample_id=str(sid), region_id="eval", input_text="-", expected_label="-",
                           expected_confidence=1.0, provenance_refs=("eval",), metadata=dict(meta))
            for sid, meta in zip(sample_ids, group_values)]


def plan_grouped_split(sample_ids: Sequence[str], group_values: Sequence[Mapping[str, Any]], group_keys: Sequence[str],
                       config: "EvalConfig") -> dict[str, tuple[str, ...]]:
    """The exact train/val/test ids ``run_evaluation`` will use (depends only on ids, groups and config).

    Call it BEFORE building features whenever a feature uses other nodes'
    labels (e.g. partner composition by the target's own category): pass
    ``val + test`` to ``build_wiring_features(mask_category_ids=...)`` and put
    ``split_ids_sha256(plan)`` into ``EvalDataset.notes["masked_split_ids_sha256"]``;
    ``run_evaluation`` then fails closed if the split it computes differs.
    """
    train, val, test, _ = grouped_split_ids(_light(sample_ids, group_values), group_keys=tuple(group_keys),
                                            split_seed=config.split_seed, train_ratio=config.train_ratio,
                                            val_ratio=config.val_ratio)
    return {"train": train, "val": val, "test": test}


def split_ids_sha256(split: Mapping[str, Sequence[str]]) -> str:
    payload = json.dumps({k: sorted(split[k]) for k in ("train", "val", "test")}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split(data: EvalDataset, config: EvalConfig) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    samples = data.light_samples()
    train_ids, val_ids, test_ids, info = grouped_split_ids(
        samples, group_keys=data.group_keys, split_seed=config.split_seed,
        train_ratio=config.train_ratio, val_ratio=config.val_ratio)
    info = {**info, "ids_sha256": split_ids_sha256({"train": train_ids, "val": val_ids, "test": test_ids})}
    expected = data.notes.get("masked_split_ids_sha256") if isinstance(data.notes, Mapping) else None
    if expected is not None and expected != info["ids_sha256"]:
        raise ValueError("features were masked for a different split than the one being evaluated "
                         "(notes.masked_split_ids_sha256 != split ids sha256)")
    manifest = DatasetManifest("eval", "eval", len(samples), {}, config.split_seed, "", train_ids, val_ids, test_ids)
    assert_grouped_split(samples, manifest, group_keys=data.group_keys)
    component_of = split_group_components(samples, group_keys=data.group_keys)
    components = np.asarray([component_of[sid] for sid in data.sample_ids], dtype=object)
    position = {sid: i for i, sid in enumerate(data.sample_ids)}
    idx = {name: np.asarray(sorted(position[s] for s in ids), dtype=np.int64)
           for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids))}
    for name, rows in idx.items():
        if len(rows) == 0:
            raise ValueError(f"grouped split produced an empty {name} split")
    return idx, components, info


def _inputs(data: EvalDataset, rows: np.ndarray, columns: Sequence[str] | None = None) -> pd.DataFrame:
    frame = data.features.iloc[rows] if columns is None else data.features.iloc[rows][list(columns)]
    frame = frame.reset_index(drop=True).copy()
    if data.text is not None:
        frame[fl.TEXT_COLUMN] = [data.text[i] for i in rows]
    return frame


def _score(y: Sequence[str], proba: np.ndarray, classes: Sequence[str], bins: int) -> dict[str, Any]:
    pred = [classes[i] for i in np.argmax(proba, axis=1)]
    return {
        "accuracy": accuracy(list(y), pred),
        "macro_f1": macro_f1(list(y), pred),
        "ece": fl.expected_calibration_error(proba, y, classes, n_bins=bins),
        "log_loss": fl.multiclass_log_loss(proba, y, classes),
        "brier": fl.brier_score(proba, y, classes),
        "predictions": pred,
    }


def _tune(spec: ModelSpec, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, classes: Sequence[str],
          config: EvalConfig) -> dict[str, Any]:
    trials = []
    best = None
    for params in spec.param_grid:
        if config.cv_folds and config.cv_folds >= 2:
            oof = fl.out_of_fold_proba(spec.backend, params, X, y, groups, classes=classes,
                                       n_splits=config.cv_folds, seed=config.seed)
            scored = _score(y.tolist(), oof, classes, config.ece_bins)
            metric = -scored["log_loss"] if config.tune_metric == "log_loss" else scored[config.tune_metric]
        else:
            oof, scored, metric = None, {}, 0.0
        trial = {"params": fl._jsonable(dict(params)), "cv_metric": metric,
                 "cv": {k: v for k, v in scored.items() if k != "predictions"}}
        trials.append(trial)
        if best is None or metric > best[0]:
            best = (metric, dict(params), oof)
    assert best is not None
    return {"trials": trials, "best_params": best[1], "best_cv_metric": best[0], "oof": best[2]}


def _fit_final(spec: ModelSpec, params: Mapping[str, Any], X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
               classes: Sequence[str], oof: np.ndarray | None, config: EvalConfig) -> tuple[fl.Learner, fl.Learner]:
    base = fl.make_learner(spec.backend, seed=config.seed, **dict(params)).fit(X, y, groups=groups)
    if spec.calibration is None:
        return base, base
    if oof is None:
        oof = fl.out_of_fold_proba(spec.backend, params, X, y, groups, classes=classes,
                                   n_splits=max(2, config.cv_folds), seed=config.seed)
    if spec.calibration in es.EXT_CALIBRATION_METHODS:
        return base, es.ExtCalibratedLearner.from_proba(base, oof, y, method=spec.calibration, classes=classes,
                                                        groups=groups, source="grouped_cv_oof", seed=config.seed)
    calibrated = fl.CalibratedLearner.from_oof(base, oof, y, method=spec.calibration, classes=classes,
                                               cv=max(2, config.cv_folds))
    return base, calibrated


CALIBRATION_PROTOCOLS = ("grouped_fold", "oof")


def _fit_final_protocol(spec: ModelSpec, params: Mapping[str, Any], X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                        classes: Sequence[str], oof: np.ndarray | None, config: EvalConfig
                        ) -> tuple[fl.Learner, fl.Learner, dict[str, Any]]:
    """Final fit + calibration per ``config.calibration_protocol`` (R6).

    ``grouped_fold``: whole train components go to a calibration fold
    (``config.calibration_fraction`` of train rows); the base learner is fitted
    on the rest and the calibrator on the base's probabilities for the fold,
    i.e. train / calibrate / test [Ovadia 2019]. Falls back to ``oof`` (recorded)
    when the fold cannot be formed (too few components, one class left).
    """
    if spec.calibration is None:
        base = fl.make_learner(spec.backend, seed=config.seed, **dict(params)).fit(X, y, groups=groups)
        return base, base, {"protocol": "none", "fit_rows": int(len(y))}
    if config.calibration_protocol == "grouped_fold":
        reason = None
        try:
            fit_rows, cal_rows = es.calibration_fold(np.arange(len(y)), groups, fraction=config.calibration_fraction,
                                                     seed=config.split_seed)
            if len(set(y[fit_rows].tolist())) < 2 or len(cal_rows) < 2:
                reason = "calibration fold left fewer than two classes in the fit rows or fewer than two rows"
        except ValueError as exc:
            reason = str(exc)
        if reason is None:
            X_fit, X_cal = X.iloc[fit_rows].reset_index(drop=True), X.iloc[cal_rows].reset_index(drop=True)
            base = fl.make_learner(spec.backend, seed=config.seed, **dict(params)).fit(X_fit, y[fit_rows],
                                                                                      groups=groups[fit_rows])
            if spec.calibration in es.EXT_CALIBRATION_METHODS:
                final: fl.Learner = es.ExtCalibratedLearner.from_proba(
                    base, base.predict_proba(X_cal), y[cal_rows], method=spec.calibration, groups=groups[cal_rows],
                    source="grouped_calibration_fold", seed=config.seed)
            else:
                final = fl.CalibratedLearner(base, method=spec.calibration, cv=max(2, config.cv_folds)).fit_prefit(
                    X_cal, y[cal_rows])
            info = {"protocol": "grouped_fold", "fit_rows": int(len(fit_rows)), "calibration_rows": int(len(cal_rows)),
                    "calibration_components": int(len(set(groups[cal_rows].tolist()))),
                    "calibration_fraction": float(config.calibration_fraction)}
            return base, final, info
        base, final = _fit_final(spec, params, X, y, groups, classes, oof, config)
        return base, final, {"protocol": "oof", "fallback_reason": reason, "fit_rows": int(len(y))}
    base, final = _fit_final(spec, params, X, y, groups, classes, oof, config)
    return base, final, {"protocol": "oof", "fit_rows": int(len(y))}


def _families(columns: Sequence[str], config: EvalConfig) -> dict[str, list[str]]:
    if config.ablation_families is not None:
        return {name: sorted(c for c in columns if any(fnmatch.fnmatchcase(c, p) for p in patterns))
                for name, patterns in config.ablation_families.items()}
    return fwf.feature_families(columns)


def run_evaluation(data: EvalDataset, config: EvalConfig = EvalConfig()) -> dict[str, Any]:
    """Run the full protocol; returns the report dict (and writes it when ``config.report_root`` is set)."""
    from threadpoolctl import threadpool_limits

    started = time.time()
    if not config.models:
        raise ValueError("config.models is empty")
    if config.calibration_protocol not in CALIBRATION_PROTOCOLS:
        raise ValueError(f"calibration_protocol must be one of {CALIBRATION_PROTOCOLS}")
    allowed_cal = set(fl.CALIBRATION_METHODS) | set(es.EXT_CALIBRATION_METHODS)
    for s in config.models:
        if s.calibration is not None and s.calibration not in allowed_cal:
            raise ValueError(f"{s.label}: calibration must be one of {sorted(allowed_cal)} or None")
    specs = list(config.models)
    uses_text = any(fl.make_learner(s.backend).uses_text for s in specs)
    uses_features = any(not fl.make_learner(s.backend).uses_text for s in specs)
    leakage = check_leakage(data, uses_text=uses_text)
    if uses_features and data.features.shape[1] == 0:
        raise ValueError("feature models requested but the dataset has no feature columns")

    idx, components, split_info = _split(data, config)
    y = {name: data.labels[rows] for name, rows in idx.items()}
    classes = tuple(sorted(set(y["train"].tolist())))
    feature_cols = list(data.features.columns)
    provenance = es.provenance_claim(data.notes)

    report: dict[str, Any] = {
        "schema_version": EVAL_REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": data.dataset,
        "target": data.target,
        "config": config.as_dict(),
        "notes": fl._jsonable(dict(data.notes)),
        "label_provenance": provenance,
        "leakage_check": leakage,
        "split": {**split_info,
                  "counts": {k: int(len(v)) for k, v in idx.items()},
                  "label_counts": {k: pd.Series(v).value_counts().sort_index().to_dict() for k, v in y.items()},
                  "test_labels_unseen_in_train": sorted(set(y["test"].tolist()) - set(classes)),
                  "n_test_components": int(len(set(components[idx["test"]].tolist()))),
                  "effective_n": {k: es.effective_n(components[v]) for k, v in idx.items()}},
        "n_features": len(feature_cols),
        "feature_families": fwf.feature_families(feature_cols),
        "aux_columns": [] if data.aux is None else list(data.aux.columns),
    }

    with threadpool_limits(limits=int(config.n_threads)):
        _set_torch_threads(int(config.n_threads))
        baselines: dict[str, Any] = {}
        if uses_features:
            baselines[VIEW_FEATURES] = feature_view_baselines(
                data.features.iloc[idx["train"]].reset_index(drop=True), y["train"],
                {s: (data.features.iloc[idx[s]].reset_index(drop=True), y[s]) for s in ("val", "test")})
        if uses_text:
            baselines[VIEW_TEXT] = text_view_baselines(
                [data.text[i] for i in idx["train"]], y["train"],  # type: ignore[index]
                {s: ([data.text[i] for i in idx[s]], y[s]) for s in ("val", "test")})  # type: ignore[index]

        X_train = _inputs(data, idx["train"])
        X_val = _inputs(data, idx["val"])
        X_test = _inputs(data, idx["test"])
        g_train = components[idx["train"]]
        g_test = components[idx["test"]].tolist()

        # ---- R3: size/degree-only baseline (label-independent, fitted once on train)
        size = _size_baseline(data, idx, y, g_train, config) if config.size_baseline else {"available": False,
                                                                                           "reason": "disabled"}
        size_preds = size.pop("_predictions", None)
        baselines["size_degree"] = size
        degree_name, degree_all = es.degree_values(_size_frame(data), config.degree_column)

        models: dict[str, Any] = {}
        fitted: dict[str, tuple[ModelSpec, dict[str, Any], fl.Learner]] = {}
        test_probas: dict[str, np.ndarray] = {}
        for spec in specs:
            t0 = time.time()
            tuning = _tune(spec, X_train, y["train"], g_train, classes, config)
            base, final, cal_info = _fit_final_protocol(spec, tuning["best_params"], X_train, y["train"], g_train,
                                                        classes, tuning["oof"], config)
            view = VIEW_TEXT if final.uses_text else VIEW_FEATURES
            val_raw = _score(y["val"].tolist(), base.predict_proba(X_val), base.classes_, config.ece_bins)
            val_final = _score(y["val"].tolist(), final.predict_proba(X_val), final.classes_, config.ece_bins)
            # ---- the single held-out test evaluation of this final config
            test_raw_proba = base.predict_proba(X_test)
            test_proba = final.predict_proba(X_test)
            test = _score(y["test"].tolist(), test_proba, final.classes_, config.ece_bins)
            test_raw = _score(y["test"].tolist(), test_raw_proba, base.classes_, config.ece_bins)
            models[spec.label] = {
                "backend": spec.backend,
                "view": view,
                "calibration": spec.calibration,
                "calibration_protocol": cal_info,
                "tuning": {k: v for k, v in tuning.items() if k != "oof"},
                "fit_info": fl._jsonable(final.fit_info),
                "describe": final.describe() if spec.backend != "nb" else {"backend": "nb", "classes": list(final.classes_)},
                "val": {"raw": _strip(val_raw), "calibrated": _strip(val_final)},
                "test": {
                    **_strip(test),
                    "raw_uncalibrated": _strip(test_raw),
                    "reliability_table": fl.reliability_table(test_proba, y["test"], final.classes_, n_bins=config.ece_bins),
                    "confusion": confusion(y["test"].tolist(), test["predictions"]),
                    # ---- R6: debiased / classwise ECE, log loss, Brier with cluster-bootstrap CIs
                    "calibration": {
                        "calibrated": es.calibration_metrics_with_ci(
                            test_proba, y["test"], final.classes_, g_test, n_bootstrap=config.calibration_bootstrap,
                            seed=config.seed, n_bins=config.ece_bins),
                        "raw": es.calibration_metrics_with_ci(
                            test_raw_proba, y["test"], base.classes_, g_test, n_bootstrap=config.calibration_bootstrap,
                            seed=config.seed, n_bins=config.ece_bins),
                    },
                },
                "fit_seconds": round(time.time() - t0, 3),
            }
            fitted[spec.label] = (spec, tuning["best_params"], final)
            test_probas[spec.label] = test_proba
            models[spec.label]["_test_predictions"] = test["predictions"]

        # ---- bootstrap CIs + paired gains, gate inputs
        for label, entry in models.items():
            view_base = baselines[entry["view"]]["splits"]["test"]
            best_rule = view_base["best_rule"]
            model_pred = entry.pop("_test_predictions")
            preds = {"model": model_pred, "best_trivial": view_base["predictions"][best_rule],
                     "majority": view_base["predictions"][fbb.RULE_MAJORITY]}
            pairs = [("model", "best_trivial"), ("model", "majority")]
            if size_preds is not None:
                preds["size_baseline"] = size_preds["test"]
                pairs.append(("model", "size_baseline"))
            boot = grouped_bootstrap(y["test"].tolist(), preds, g_test, n_bootstrap=config.n_bootstrap,
                                     seed=config.seed, pairs=tuple(pairs))
            entry["test"]["bootstrap"] = boot
            gate = fbb.trivial_baseline_gate(
                model_heldout_accuracy=entry["test"]["accuracy"],
                baselines={"best_rule": f"{entry['view']}:{best_rule}", "best_accuracy": view_base["best_accuracy"]},
                heldout_count=int(len(idx["test"])), config={"min_margin": config.min_margin})
            trivial_f1 = max(view_base["macro_f1"].values())
            paired = boot["paired"]["model-minus-best_trivial"]
            entry["gate"] = {
                "trivial_baseline_gate": {k: v for k, v in gate.items() if k != "baselines"},
                "best_trivial_rule": f"{entry['view']}:{best_rule}",
                "best_trivial_accuracy": view_base["best_accuracy"],
                "majority_accuracy": view_base["accuracy"][fbb.RULE_MAJORITY],
                "best_trivial_macro_f1": trivial_f1,
                "beats_trivial_macro_f1": entry["test"]["macro_f1"] > trivial_f1 + config.min_margin,
                "paired_accuracy_gain_ci95": paired["accuracy_diff_ci95"],
                "paired_gain_significant": paired["accuracy_diff_ci95"][0] > 0,
            }
            entry["gate"].update(_size_gate(entry["test"]["accuracy"], size, boot, config))
            # ---- R3: accuracy by degree decile (edges from train)
            if degree_all is not None:
                dec_preds = {"model": model_pred, "best_trivial": preds["best_trivial"], "majority": preds["majority"]}
                if size_preds is not None:
                    dec_preds["size_baseline"] = size_preds["test"]
                entry["test"]["degree_deciles"] = {
                    "degree_column": degree_name,
                    "rows": es.degree_decile_table(degree_all[idx["train"]], degree_all[idx["test"]], y["test"].tolist(),
                                                   dec_preds, g_test)}

        # ---- controls
        for label, (spec, params, final) in fitted.items():
            entry = models[label]
            if config.shuffle_control:
                rng = np.random.default_rng(config.seed + 7919)
                shuffled = y["train"].copy()
                rng.shuffle(shuffled)
                control = fl.make_learner(spec.backend, seed=config.seed, **dict(params)).fit(X_train, shuffled, groups=g_train)
                shuffle_acc = accuracy(y["test"].tolist(), control.predict(X_test))
                majority_acc = entry["gate"]["majority_accuracy"]
                entry["shuffle_control"] = {
                    "evaluated_on": "test",
                    "accuracy": shuffle_acc,
                    "majority_accuracy": majority_acc,
                    "collapsed_to_majority": shuffle_acc <= majority_acc + config.shuffle_tolerance,
                }
            if config.n_permutations > 0:
                entry["permutation_null"] = _permutation_control(spec, params, X_train, y["train"], g_train, X_test,
                                                                 y["test"], config)
            if config.random_split_control:
                entry["random_split_control"] = _random_split_control(data, spec, params, config)
            ok_shuffle = entry.get("shuffle_control", {}).get("collapsed_to_majority", True)
            g = entry["gate"]
            g["shuffle_ok"] = ok_shuffle
            g.update(_null_gate(entry.get("permutation_null"), ok_shuffle, config))
            verdict = es.assemble_gate({
                "beats_trivial_accuracy": g["trivial_baseline_gate"]["pass"],
                "beats_trivial_macro_f1": g["beats_trivial_macro_f1"],
                "paired_gain_significant": g["paired_gain_significant"],
                "beats_size_baseline": g["beats_size_baseline"],
                "null_control_ok": g["null_control_ok"],
            })
            g["pass"] = verdict["pass"]
            g["failed_criteria"] = verdict["failed"]
            g["gate_applies"] = provenance["gate_applies"]
            g["claim"] = provenance["claim"]

        best_label = sorted(models, key=lambda m: (-models[m]["val"]["calibrated"]["macro_f1"], m))[0]
        report["best_on_val"] = best_label
        spec_best, params_best, final_best = fitted[best_label]
        if config.ablation:
            report["ablation"] = _ablation(data, spec_best, params_best, idx, y, g_train, feature_cols, config) \
                if not fl.make_learner(spec_best.backend).uses_text else {"skipped": "text model"}
        if config.split_curve:
            report["split_curve"] = _split_curve(data, spec_best, params_best, config)
        if _nt_hooks_enabled(data, config):
            report["nt_hooks"] = _nt_hooks(data, spec_best, params_best, idx, components, config,
                                           direct_test_predictions=final_best.predict(X_test))
        if config.hierarchy_parent:
            report["hierarchy"] = _hierarchy(data, spec_best, params_best, idx, components, config,
                                             flat_proba=test_probas[best_label], flat_classes=final_best.classes_)

        if config.report_root and config.save_models:
            out_dir = report_dir(config.report_root, data.dataset, data.target, config.run_label)
            for label, (spec, params, final) in fitted.items():
                manifest = fl.save_learner(final, out_dir / "models" / label, overwrite=True,
                                           metrics={"test": {k: v for k, v in models[label]["test"].items()
                                                             if k in ("accuracy", "macro_f1", "ece", "log_loss")},
                                                    "gate": models[label]["gate"]})
                models[label]["artifact"] = {"path": manifest["path"], "manifest_sha256": manifest["manifest_sha256"]}

    for view, result in baselines.items():
        for split in result.get("splits", {}).values():
            split.pop("predictions", None)
    report["baselines"] = baselines
    report["models"] = models
    report["summary"] = summary_rows(report)
    report["elapsed_seconds"] = round(time.time() - started, 2)
    if config.report_root:
        write_report(report, config.report_root, run_label=config.run_label)
    return report


# =========================================================================== rigor helpers (R3/R4/R6/R7)


def _size_frame(data: EvalDataset) -> pd.DataFrame:
    """Features and aux side by side (the size baseline and degree axis may read either)."""
    if data.aux is None:
        return data.features
    return pd.concat([data.features, data.aux], axis=1)


def _size_baseline(data: EvalDataset, idx: Mapping[str, np.ndarray], y: Mapping[str, np.ndarray], g_train: np.ndarray,
                   config: EvalConfig) -> dict[str, Any]:
    """R3: HGB on degree / synapse-count / cable-size columns only [Bernett 2024; Subramonian 2024].

    Columns come from features AND aux (so a size proxy excluded as a model
    feature can still define the baseline: the bar only gets higher).
    """
    frame = _size_frame(data)
    cols = [c for c in es.size_columns(frame.columns, config.size_patterns)
            if pd.api.types.is_numeric_dtype(frame[c].dtype) or pd.api.types.is_bool_dtype(frame[c].dtype)]
    if not cols:
        return {"available": False, "reason": f"no numeric column matches {list(config.size_patterns)}"}
    sub = frame[cols].astype(np.float64)
    if len(set(y["train"].tolist())) < 2:
        return {"available": False, "reason": "fewer than two train classes"}
    learner = fl.make_learner("hgb", seed=config.seed, **dict(config.size_baseline_params))
    learner.fit(sub.iloc[idx["train"]].reset_index(drop=True), y["train"], groups=g_train)
    preds = {s: learner.predict(sub.iloc[idx[s]].reset_index(drop=True)) for s in ("val", "test")}
    return {
        "available": True,
        "backend": "hgb",
        "params": fl._jsonable(dict(config.size_baseline_params)),
        "columns": cols,
        "columns_from_aux": sorted(c for c in cols if data.aux is not None and c in data.aux.columns),
        "accuracy": {s: accuracy(y[s].tolist(), p) for s, p in preds.items()},
        "macro_f1": {s: macro_f1(y[s].tolist(), p) for s, p in preds.items()},
        "_predictions": preds,
    }


def _size_gate(model_acc: float, size: Mapping[str, Any], boot: Mapping[str, Any], config: EvalConfig) -> dict[str, Any]:
    if not size.get("available"):
        return {"size_baseline_accuracy": None, "paired_gain_over_size_ci95": None,
                "beats_size_baseline": None if config.require_size_baseline else True,
                "size_baseline_note": f"size baseline unavailable ({size.get('reason')})"
                                      + ("; required, so the gate fails closed" if config.require_size_baseline
                                         else "; not required by config")}
    size_acc = float(size["accuracy"]["test"])
    ci = boot["paired"]["model-minus-size_baseline"]["accuracy_diff_ci95"]
    return {"size_baseline_accuracy": size_acc, "paired_gain_over_size_ci95": ci,
            "beats_size_baseline": bool(model_acc >= size_acc + config.min_margin and ci[0] > 0)}


def _permutation_control(spec: ModelSpec, params: Mapping[str, Any], X_train: pd.DataFrame, y_train: np.ndarray,
                         g_train: np.ndarray, X_test: pd.DataFrame, y_test: np.ndarray, config: EvalConfig) -> dict[str, Any]:
    """R4: group-block permutation null of the final config (no retuning per permutation, documented).

    The observed statistic is the same pipeline refitted on the TRUE labels
    (base learner, full train, no calibration), so observed and null differ
    only in the labels.
    """
    from threadpoolctl import threadpool_limits

    reference = fl.make_learner(spec.backend, seed=config.seed, **dict(params)).fit(X_train, y_train, groups=g_train)
    ref_pred = reference.predict(X_test)
    observed = accuracy(y_test.tolist(), ref_pred)
    observed_f1 = macro_f1(y_test.tolist(), ref_pred)

    def fit_predict(labels: np.ndarray) -> list[str]:
        if len(set(labels.tolist())) < 2:
            return [labels[0]] * len(X_test)
        return fl.make_learner(spec.backend, seed=config.seed, **dict(params)).fit(
            X_train, labels, groups=g_train).predict(X_test)

    n_jobs = max(1, int(config.permutation_n_jobs))
    with threadpool_limits(limits=max(1, int(config.n_threads) // n_jobs)):
        result = es.permutation_null(fit_predict, y_train, g_train, y_test, observed_accuracy=observed,
                                     observed_macro_f1=observed_f1, n_permutations=config.n_permutations,
                                     seed=config.seed + 104729, n_jobs=n_jobs)
    result["evaluated_on"] = "test"
    result["note"] = "final params refitted per permutation without retuning; observed = same refit on true labels"
    return result


MIN_PERMUTATIONS = 100


def _null_gate(perm: Mapping[str, Any] | None, ok_shuffle: bool, config: EvalConfig) -> dict[str, Any]:
    if perm is not None and perm["n_permutations"] >= MIN_PERMUTATIONS:
        ok = perm["p_value_accuracy"] <= config.permutation_alpha
        return {"null_control": "group_permutation", "permutation_p_value": perm["p_value_accuracy"],
                "null_control_ok": bool(ok)}
    n = 0 if perm is None else perm["n_permutations"]
    if config.require_permutation_null:
        return {"null_control": "group_permutation", "permutation_p_value": None if perm is None else perm["p_value_accuracy"],
                "null_control_ok": None,
                "null_control_note": f"{n} permutations < {MIN_PERMUTATIONS} required (R4); the gate fails closed"}
    return {"null_control": "single_shuffle (permutation null not required by config)",
            "permutation_p_value": None if perm is None else perm["p_value_accuracy"], "null_control_ok": bool(ok_shuffle)}


def _fit_predict_rows(data: EvalDataset, spec: ModelSpec, params: Mapping[str, Any], train: np.ndarray, test: np.ndarray,
                      labels: np.ndarray, groups: np.ndarray | None, config: EvalConfig) -> list[str]:
    learner = fl.make_learner(spec.backend, seed=config.seed, **dict(params))
    learner.fit(_inputs(data, train), labels[train], groups=None if groups is None else groups[train])
    return learner.predict(_inputs(data, test))


def _split_curve(data: EvalDataset, spec: ModelSpec, params: Mapping[str, Any], config: EvalConfig) -> dict[str, Any]:
    """R4: the best-on-val config refitted (no retuning, no calibration) on a graded series of splits.

    Features were built (and partner categories masked) for the PRIMARY
    split only, so non-primary levels of a masked target are optimistic.
    """
    metadata = data.row_metadata()
    available = sorted({k for m in metadata for k in m})
    levels = list(config.split_levels) if config.split_levels is not None else \
        es.default_split_levels(data.group_keys, available)
    rows = []
    for level in levels:
        needed = list(level.group_keys) + ([level.holdout_key] if level.holdout_key else [])
        missing = [k for k in needed if k not in available]
        if missing:
            rows.append({"level": level.name, "skipped": f"missing metadata keys {missing}"})
            continue
        plan = es.split_curve_plan(data.sample_ids, metadata, level, split_seed=config.split_seed,
                                   train_ratio=config.train_ratio, val_ratio=config.val_ratio)
        tr, te = plan["train"], plan["test"]
        if len(tr) == 0 or len(te) == 0 or len(set(data.labels[tr].tolist())) < 2:
            rows.append({"level": level.name, "skipped": "empty split or fewer than two train classes",
                         "n_train": int(len(tr)), "n_test": int(len(te))})
            continue
        pred = _fit_predict_rows(data, spec, params, tr, te, data.labels, plan["components"], config)
        truth = data.labels[te].tolist()
        maj = _majority(data.labels[tr].tolist())
        boot = grouped_bootstrap(truth, {"model": pred}, plan["components"][te].tolist(),
                                 n_bootstrap=min(config.n_bootstrap, 500), seed=config.seed)
        rows.append({"level": level.name, "kind": plan["kind"], "group_keys": list(level.group_keys),
                     "holdout_key": plan.get("holdout_key"), "n_train": int(len(tr)), "n_test": int(len(te)),
                     "n_test_components": int(len(set(plan["components"][te].tolist()))),
                     "accuracy": accuracy(truth, pred), "macro_f1": macro_f1(truth, pred),
                     "accuracy_ci95": boot["per_prediction"]["model"]["accuracy_ci95"],
                     "majority_accuracy": accuracy(truth, [maj] * len(truth))})
    masked = isinstance(data.notes, Mapping) and data.notes.get("masked_split_ids_sha256") is not None
    return {"model": spec.label, "params": fl._jsonable(dict(params)), "levels": rows,
            "caveat": ("features were masked for the primary split only; levels other than the primary grouping are "
                       "optimistic for this masked target") if masked else None}


def _nt_hooks_enabled(data: EvalDataset, config: EvalConfig) -> bool:
    """On for NT targets (``nt_*`` or anything naming a neurotransmitter) unless ``config.nt_hooks`` says otherwise."""
    if config.nt_hooks is not None:
        return bool(config.nt_hooks)
    target = data.target.lower()
    return target.startswith("nt_") or "neurotransmitter" in target


def _nt_hooks(data: EvalDataset, spec: ModelSpec, params: Mapping[str, Any], idx: Mapping[str, np.ndarray],
              components: np.ndarray, config: EvalConfig, *, direct_test_predictions: Sequence[str]) -> dict[str, Any]:
    """R3 NT extras: binary ACh vs GABA+Glu view, hemilineage -> NT oracle, two-stage wiring -> hemilineage -> NT."""
    out: dict[str, Any] = {"model": spec.label, "binary": _nt_binary_view(data, spec, params, idx, components, config)}
    metadata = data.row_metadata()
    col = config.hemilineage_column
    if not any(col in m for m in metadata):
        out["hemilineage_oracle"] = {"skipped": f"no {col!r} in group keys or aux"}
        out["two_stage"] = {"skipped": f"no {col!r} in group keys or aux"}
        return out
    hl = np.asarray([m.get(col) for m in metadata], dtype=object)
    tr, te = idx["train"], idx["test"]
    y_test = data.labels[te].tolist()
    direct = list(direct_test_predictions)
    oracle = es.hemilineage_nt_oracle(hl[tr], data.labels[tr], hl[te], data.labels[te])
    boot = grouped_bootstrap(y_test, {"direct": direct,
                                      "oracle_train": oracle["train_lookup"]["predictions"],
                                      "oracle_loo": oracle["leave_one_out"]["predictions"]},
                             components[te].tolist(), n_bootstrap=config.n_bootstrap, seed=config.seed,
                             pairs=(("direct", "oracle_train"),))
    out["hemilineage_oracle"] = {
        "train_lookup": {k: v for k, v in oracle["train_lookup"].items() if k != "predictions"},
        "leave_one_out_upper_bound": {k: v for k, v in oracle["leave_one_out"].items() if k != "predictions"},
        "n_train_hemilineages": oracle["n_train_hemilineages"], "n_test_hemilineages": oracle["n_test_hemilineages"],
        "direct_model_accuracy": accuracy(y_test, direct),
        "bootstrap": boot,
    }
    known_tr = np.asarray([i for i in tr if es._known(hl[i])], dtype=np.int64)
    if len(known_tr) == 0 or len({str(hl[i]) for i in known_tr}) < 2:
        out["two_stage"] = {"skipped": "fewer than two known train hemilineages"}
        return out
    hl_labels = np.asarray([str(v) for v in hl], dtype=object)
    stage1 = _fit_predict_rows(data, spec, params, known_tr, te, hl_labels, components, config)
    table = oracle["table"]
    majority = _majority(data.labels[tr].tolist())
    two_stage = [table.get(h, majority) for h in stage1]
    known_te = [es._known(hl[i]) for i in te]
    hl_true = [str(hl[i]) for i in te]
    boot2 = grouped_bootstrap(y_test, {"two_stage": two_stage, "direct": direct},
                              components[te].tolist(), n_bootstrap=config.n_bootstrap, seed=config.seed,
                              pairs=(("direct", "two_stage"),))
    out["two_stage"] = {
        "stage1_target": col, "stage1_train_rows": int(len(known_tr)),
        "stage1_test_accuracy_on_known": float(np.mean([a == b for a, b, k in zip(stage1, hl_true, known_te) if k]))
        if any(known_te) else None,
        "accuracy": accuracy(y_test, two_stage), "macro_f1": macro_f1(y_test, two_stage),
        "direct_accuracy": accuracy(y_test, direct),
        "bootstrap": boot2,
        "note": "stage 2 maps the predicted TRAIN hemilineage to its train majority NT; under a hemilineage-grouped "
                "split every test hemilineage is unseen, so this measures 'NT of the most similar train lineage'",
    }
    return out


def _nt_binary_view(data: EvalDataset, spec: ModelSpec, params: Mapping[str, Any], idx: Mapping[str, np.ndarray],
                    components: np.ndarray, config: EvalConfig) -> dict[str, Any]:
    """ACh vs GABA+Glu on the SAME split (rows with other transmitters dropped, counts recorded)."""
    binary = np.asarray([es.binary_nt_label(v) for v in data.labels], dtype=object)
    keep = {s: np.asarray([i for i in idx[s] if binary[i] is not None], dtype=np.int64) for s in ("train", "val", "test")}
    dropped = {s: int(len(idx[s]) - len(keep[s])) for s in keep}
    if len(keep["train"]) == 0 or len(keep["test"]) == 0 or len(set(binary[keep["train"]].tolist())) < 2:
        return {"skipped": "fewer than two binary classes in train or empty test", "dropped_rows": dropped}
    labels = np.asarray([b if b is not None else "" for b in binary], dtype=object)
    pred = _fit_predict_rows(data, spec, params, keep["train"], keep["test"], labels, components, config)
    truth = labels[keep["test"]].tolist()
    comps = components[keep["test"]].tolist()
    maj = _majority(labels[keep["train"]].tolist())
    preds = {"model": pred, "majority": [maj] * len(truth)}
    feat = None
    if data.features.shape[1]:
        feat = feature_view_baselines(data.features.iloc[keep["train"]].reset_index(drop=True), labels[keep["train"]],
                                      {"test": (data.features.iloc[keep["test"]].reset_index(drop=True), labels[keep["test"]])})
        best_rule = feat["splits"]["test"]["best_rule"]
        preds["best_trivial"] = feat["splits"]["test"]["predictions"][best_rule]
    pairs = [("model", "majority")] + ([("model", "best_trivial")] if "best_trivial" in preds else [])
    boot = grouped_bootstrap(truth, preds, comps, n_bootstrap=config.n_bootstrap, seed=config.seed, pairs=tuple(pairs))
    return {"labels": [es.NT_BINARY_EXC, es.NT_BINARY_INH], "dropped_rows": dropped,
            "n_train": int(len(keep["train"])), "n_test": int(len(keep["test"])),
            "accuracy": accuracy(truth, pred), "macro_f1": macro_f1(truth, pred),
            "majority_accuracy": accuracy(truth, preds["majority"]),
            "best_trivial_rule": None if feat is None else feat["splits"]["test"]["best_rule"],
            "best_trivial_accuracy": None if feat is None else feat["splits"]["test"]["best_accuracy"],
            "bootstrap": boot}


def _hierarchy(data: EvalDataset, spec: ModelSpec, params: Mapping[str, Any], idx: Mapping[str, np.ndarray],
               components: np.ndarray, config: EvalConfig, *, flat_proba: np.ndarray,
               flat_classes: Sequence[str]) -> dict[str, Any]:
    """R7 prototype: parent -> child conditional model vs the flat global model on the same split (report only)."""
    col = str(config.hierarchy_parent)
    metadata = data.row_metadata()
    if not any(col in m for m in metadata):
        return {"skipped": f"no {col!r} in group keys or aux"}
    if fl.make_learner(spec.backend).uses_text:
        return {"skipped": "text model"}
    parents = np.asarray([m.get(col) for m in metadata], dtype=object)
    tr, te = idx["train"], idx["test"]
    parent_of = es.parent_map(data.labels[tr], parents[tr])
    for label in set(data.labels[tr].tolist()) - set(parent_of):
        parent_of[label] = "__unknown_parent__"
    learner = fl.make_learner("hierarchical", seed=config.seed, base_backend=spec.backend, base_params=dict(params),
                              parent_of=parent_of)
    learner.fit(_inputs(data, tr), data.labels[tr], groups=components[tr])
    hier_proba = learner.predict_proba(_inputs(data, te))
    y_test = data.labels[te].tolist()
    hier_pred = [learner.classes_[i] for i in hier_proba.argmax(axis=1)]
    flat_pred = [flat_classes[i] for i in np.asarray(flat_proba).argmax(axis=1)]
    eval_parent_of = {**es.parent_map(data.labels, parents), **parent_of}
    comps = components[te].tolist()

    def block(proba: np.ndarray, pred: list[str], cls: Sequence[str]) -> dict[str, Any]:
        return {"accuracy": accuracy(y_test, pred), "macro_f1": macro_f1(y_test, pred),
                **es.hierarchy_metrics(y_test, pred, eval_parent_of),
                "calibration": es.calibration_metrics_with_ci(proba, y_test, cls, comps,
                                                              n_bootstrap=config.calibration_bootstrap,
                                                              seed=config.seed, n_bins=config.ece_bins)}

    boot = grouped_bootstrap(y_test, {"hierarchical": hier_pred, "flat": flat_pred}, comps,
                             n_bootstrap=config.n_bootstrap, seed=config.seed, pairs=(("hierarchical", "flat"),))
    return {"parent_column": col, "model": spec.label, "n_parents": len(set(parent_of.values())),
            "flat": block(np.asarray(flat_proba), flat_pred, flat_classes),
            "hierarchical": block(hier_proba, hier_pred, learner.classes_),
            "bootstrap": boot, "fit_info": learner.fit_info,
            "note": "R7: reported, not shipped; the flat model is the calibrated final model, the hierarchy is uncalibrated"}


def _strip(scored: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in scored.items() if k != "predictions"}


def _set_torch_threads(n: int) -> None:
    try:
        import torch

        torch.set_num_threads(max(1, n))
    except Exception:  # pragma: no cover - torch optional
        pass


def _random_split_control(data: EvalDataset, spec: ModelSpec, params: Mapping[str, Any], config: EvalConfig) -> dict[str, Any]:
    """Same model on a per-sample (ungrouped) split: the leakage gap B2 asks to report."""
    train_ids, _, test_ids = deterministic_split_ids(list(data.sample_ids), split_seed=config.split_seed,
                                                     train_ratio=config.train_ratio, val_ratio=config.val_ratio)
    position = {sid: i for i, sid in enumerate(data.sample_ids)}
    tr = np.asarray(sorted(position[s] for s in train_ids))
    te = np.asarray(sorted(position[s] for s in test_ids))
    learner = fl.make_learner(spec.backend, seed=config.seed, **dict(params))
    learner.fit(_inputs(data, tr), data.labels[tr])
    return {"split": "per_sample_random", "test_accuracy": accuracy(data.labels[te].tolist(), learner.predict(_inputs(data, te))),
            "test_count": int(len(te)), "note": "optimistic by construction; compare with the grouped test accuracy"}


def _ablation(data: EvalDataset, spec: ModelSpec, params: Mapping[str, Any], idx: Mapping[str, np.ndarray],
              y: Mapping[str, np.ndarray], g_train: np.ndarray, feature_cols: Sequence[str], config: EvalConfig) -> dict[str, Any]:
    families = _families(feature_cols, config)
    rows = []

    def fit_eval(columns: Sequence[str]) -> dict[str, float]:
        learner = fl.make_learner(spec.backend, seed=config.seed, **dict(params))
        learner.fit(_inputs(data, idx["train"], columns), y["train"], groups=g_train)
        pred = learner.predict(_inputs(data, idx["val"], columns))
        return {"accuracy": accuracy(y["val"].tolist(), pred), "macro_f1": macro_f1(y["val"].tolist(), pred)}

    full = fit_eval(feature_cols)
    for family, members in families.items():
        kept = [c for c in feature_cols if c not in set(members)]
        if not kept:
            rows.append({"family": family, "n_features": len(members), "skipped": "no features left"})
            continue
        scored = fit_eval(kept)
        rows.append({"family": family, "n_features": len(members), **scored,
                     "delta_accuracy": scored["accuracy"] - full["accuracy"],
                     "delta_macro_f1": scored["macro_f1"] - full["macro_f1"]})
    return {"model": spec.label, "evaluated_on": "val", "full": full,
            "drop_one_family": sorted(rows, key=lambda r: r.get("delta_accuracy", 0.0))}


# =========================================================================== reporting


def report_dir(report_root: str | Path, dataset: str, target: str, run_label: str = "") -> Path:
    base = Path(report_root) / fwf.slug(dataset) / fwf.slug(target)
    path = base / fwf.slug(run_label) if run_label else base
    if "snapshots" in path.resolve(strict=False).parts:
        raise ValueError(f"refusing to write reports under a snapshots directory: {path}")
    return path


def summary_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per model: the fields the orchestrator's results table needs."""
    rows = []
    for label, entry in report.get("models", {}).items():
        gate = entry.get("gate", {})
        ci = entry["test"].get("bootstrap", {}).get("per_prediction", {}).get("model", {}).get("accuracy_ci95")
        rows.append({
            "dataset": report["dataset"],
            "target": report["target"],
            "model": label,
            "majority": gate.get("majority_accuracy"),
            "best_trivial": gate.get("best_trivial_accuracy"),
            "best_trivial_rule": gate.get("best_trivial_rule"),
            "model_acc": entry["test"]["accuracy"],
            "model_ci": None if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]",
            "macro_f1": entry["test"]["macro_f1"],
            "ece": entry["test"]["ece"],
            "shuffle_acc": entry.get("shuffle_control", {}).get("accuracy"),
            "n_train": report["split"]["counts"]["train"],
            "n_test": report["split"]["counts"]["test"],
            "gate": "pass" if gate.get("pass") else "fail",
            # ---- rigor columns (R1/R3/R4/R6)
            "size_baseline": gate.get("size_baseline_accuracy"),
            "perm_p": gate.get("permutation_p_value"),
            "ece_sweep": entry["test"].get("calibration", {}).get("calibrated", {}).get("ece_sweep"),
            "effective_n_test": report["split"].get("effective_n", {}).get("test"),
            "failed_criteria": gate.get("failed_criteria"),
            "label_provenance": report.get("label_provenance", {}).get("label_provenance"),
            "gate_applies": bool(gate.get("gate_applies", False)),
            "claim": gate.get("claim"),
        })
    return rows


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci_str(ci: Sequence[float] | None) -> str:
    return "" if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def _render_rigor_sections(report: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    cal_rows = [(label, entry["test"]["calibration"]) for label, entry in report["models"].items()
                if "calibration" in entry["test"]]
    if cal_rows:
        lines += ["", "## Calibration (test, cluster-bootstrap 95% CI)", "",
                  "| model | stage | ECE width | ECE mass | ECE_sweep | debiased L2 | classwise | log loss | Brier |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for label, cal in cal_rows:
            for stage in ("raw", "calibrated"):
                m = cal[stage]
                cells = [f"{m[k]:.3f} {_ci_str(m.get(k + '_ci95'))}" for k in es.CALIBRATION_METRIC_KEYS]
                lines.append(f"| {label} | {stage} | " + " | ".join(cells) + " |")
    for label, entry in report["models"].items():
        dec = entry["test"].get("degree_deciles")
        if not dec:
            continue
        keys = [k for k in dec["rows"][0] if k.startswith("acc_")] if dec["rows"] else []
        lines += ["", f"## Accuracy by degree decile ({label}; axis {dec['degree_column']}, edges from train)", "",
                  "| decile | n | groups | " + " | ".join(k[4:] for k in keys) + " |",
                  "|---|---|---|" + "---|" * len(keys)]
        for row in dec["rows"]:
            lines.append(f"| {row['decile']} | {row['n']} | {row['n_components']} | "
                         + " | ".join(f"{row[k]:.3f}" for k in keys) + " |")
    curve = report.get("split_curve")
    if curve:
        lines += ["", f"## Split curve ({curve['model']}, refit without retuning)", "",
                  "| level | n train | n test | test groups | acc [95% CI] | macro-F1 | majority |", "|---|---|---|---|---|---|---|"]
        for row in curve["levels"]:
            if "skipped" in row:
                lines.append(f"| {row['level']} | skipped: {row['skipped']} | | | | | |")
            else:
                lines.append(f"| {row['level']} | {row['n_train']} | {row['n_test']} | {row['n_test_components']} | "
                             f"{row['accuracy']:.3f} {_ci_str(row['accuracy_ci95'])} | {row['macro_f1']:.3f} | "
                             f"{row['majority_accuracy']:.3f} |")
        if curve.get("caveat"):
            lines.append(f"\nCaveat: {curve['caveat']}.")
    nt = report.get("nt_hooks")
    if nt:
        lines += ["", f"## NT hooks ({nt['model']})", ""]
        b = nt.get("binary", {})
        if "skipped" in b:
            lines.append(f"- binary ACh vs GABA+Glu: skipped ({b['skipped']})")
        else:
            lines.append(f"- binary ACh vs GABA+Glu: acc {b['accuracy']:.3f}, macro-F1 {b['macro_f1']:.3f}, majority "
                         f"{b['majority_accuracy']:.3f}, best trivial {_fmt(b['best_trivial_accuracy'])} "
                         f"({b['best_trivial_rule']}); n test {b['n_test']}, dropped {b['dropped_rows']}")
        o = nt.get("hemilineage_oracle", {})
        if "skipped" in o:
            lines.append(f"- hemilineage -> NT oracle: skipped ({o['skipped']})")
        else:
            lines.append(f"- hemilineage -> NT oracle: train lookup {o['train_lookup']['accuracy']:.3f} (coverage "
                         f"{o['train_lookup']['coverage']:.2f}); within-test leave-one-out upper bound "
                         f"{o['leave_one_out_upper_bound']['accuracy']:.3f} (coverage "
                         f"{o['leave_one_out_upper_bound']['coverage']:.2f}); direct model {o['direct_model_accuracy']:.3f}")
        t = nt.get("two_stage", {})
        if "skipped" in t:
            lines.append(f"- two-stage wiring -> hemilineage -> NT: skipped ({t['skipped']})")
        else:
            lines.append(f"- two-stage wiring -> hemilineage -> NT: acc {t['accuracy']:.3f} vs direct "
                         f"{t['direct_accuracy']:.3f}; paired direct-minus-two-stage CI95 "
                         f"{_ci_str(t['bootstrap']['paired']['direct-minus-two_stage']['accuracy_diff_ci95'])}")
    hier = report.get("hierarchy")
    if hier and "skipped" not in hier:
        lines += ["", f"## Hierarchy prototype (parent = {hier['parent_column']}; report only)", "",
                  "| model | acc | macro-F1 | parent acc | errors within parent | ECE_sweep | log loss |", "|---|---|---|---|---|---|---|"]
        for name in ("flat", "hierarchical"):
            h = hier[name]
            lines.append(f"| {name} | {h['accuracy']:.3f} | {h['macro_f1']:.3f} | {h['parent_accuracy']:.3f} | "
                         f"{_fmt(h['errors_within_true_parent'])} | {h['calibration']['ece_sweep']:.3f} | "
                         f"{h['calibration']['log_loss']:.3f} |")
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [f"# {report['dataset']} / {report['target']}", "",
             f"Generated {report['generated_at']}. Split: grouped by {', '.join(report['split']['group_keys']) or '(none)'}; "
             f"counts {report['split']['counts']}; {report['split']['n_test_components']} test components "
             f"(effective n = groups: {report['split'].get('effective_n')}).", ""]
    prov = report.get("label_provenance") or {}
    if prov:
        lines += [f"Label provenance: **{prov.get('label_provenance')}**; gate applies: {prov.get('gate_applies')}. "
                  f"A pass here means: {prov.get('claim')}.", ""]
    lines += ["## Held-out test", "",
              "| model | majority | best trivial (rule) | size/degree | acc [95% CI] | macro-F1 | ECE | ECE_sweep | shuffle acc "
              "| perm p | gate |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in report["summary"]:
        lines.append(f"| {row['model']} | {_fmt(row['majority'])} | {_fmt(row['best_trivial'])} ({row['best_trivial_rule']}) | "
                     f"{_fmt(row.get('size_baseline'))} | "
                     f"{_fmt(row['model_acc'])} {row['model_ci'] or ''} | {_fmt(row['macro_f1'])} | {_fmt(row['ece'])} | "
                     f"{_fmt(row.get('ece_sweep'))} | {_fmt(row['shuffle_acc'])} | {_fmt(row.get('perm_p'))} | {row['gate']} |")
    lines += ["", "## Gate detail", ""]
    for label, entry in report["models"].items():
        g = entry["gate"]
        lines.append(f"- **{label}**: paired acc gain CI95 {[round(x, 4) for x in g['paired_accuracy_gain_ci95']]}, "
                     f"macro-F1 {entry['test']['macro_f1']:.3f} vs trivial {g['best_trivial_macro_f1']:.3f}, "
                     f"shuffle ok {g['shuffle_ok']}, raw ECE {entry['test']['raw_uncalibrated']['ece']:.3f} -> "
                     f"calibrated {entry['test']['ece']:.3f}"
                     + (f", random-split acc {entry['random_split_control']['test_accuracy']:.3f}"
                        if "random_split_control" in entry else ""))
        if g.get("failed_criteria"):
            lines.append(f"  - failed criteria: {', '.join(g['failed_criteria'])}")
        if g.get("paired_gain_over_size_ci95") is not None:
            lines.append(f"  - gain over size/degree baseline CI95 {[round(x, 4) for x in g['paired_gain_over_size_ci95']]}")
        for key in ("size_baseline_note", "null_control_note"):
            if g.get(key):
                lines.append(f"  - {g[key]}")
        failures = g["trivial_baseline_gate"].get("failure_reasons") or []
        for reason in failures:
            lines.append(f"  - {reason}")
    lines += _render_rigor_sections(report)
    abl = report.get("ablation")
    if abl and "drop_one_family" in abl:
        lines += ["", f"## Feature ablation ({abl['model']}, on {abl['evaluated_on']})", "",
                  f"Full: acc {abl['full']['accuracy']:.3f}, macro-F1 {abl['full']['macro_f1']:.3f}", "",
                  "| dropped family | n | acc | d acc | macro-F1 | d macro-F1 |", "|---|---|---|---|---|---|"]
        for row in abl["drop_one_family"]:
            if "skipped" in row:
                lines.append(f"| {row['family']} | {row['n_features']} | skipped: {row['skipped']} | | | |")
            else:
                lines.append(f"| {row['family']} | {row['n_features']} | {row['accuracy']:.3f} | {row['delta_accuracy']:+.3f} | "
                             f"{row['macro_f1']:.3f} | {row['delta_macro_f1']:+.3f} |")
    lines += ["", "## Leakage check", "",
              f"Exclusion rule for `{report['target']}`: {report['leakage_check']['exclusion_rule']['reason']}. "
              f"{report['leakage_check']['checked_feature_columns']} feature columns checked.", ""]
    return "\n".join(lines) + "\n"


def write_report(report: Mapping[str, Any], report_root: str | Path, *, run_label: str = "") -> dict[str, str]:
    out = report_dir(report_root, report["dataset"], report["target"], run_label)
    out.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(fl._jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    (out / "report.json").write_text(payload, encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return {"json": str(out / "report.json"), "markdown": str(out / "report.md"),
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}


__all__ = [
    "HEMISPHERE_AUX",
    "TOTAL_DEGREE_AUX",
    "normalize_hemisphere",
    "size_side_aux",
    "DEFAULT_REPORT_ROOT",
    "EVAL_REPORT_SCHEMA_VERSION",
    "EvalConfig",
    "EvalDataset",
    "ModelSpec",
    "accuracy",
    "check_leakage",
    "confusion",
    "feature_view_baselines",
    "features_as_text",
    "grouped_bootstrap",
    "macro_f1",
    "plan_grouped_split",
    "render_markdown",
    "report_dir",
    "run_evaluation",
    "split_ids_sha256",
    "summary_rows",
    "text_view_baselines",
    "write_report",
]
