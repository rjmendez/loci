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
8. **Gate** per model: beats the best trivial rule of its view on test
   accuracy by ``min_margin`` (``trivial_baseline_gate``) AND on macro-F1,
   the paired-bootstrap lower bound of the accuracy gain is > 0, and the
   shuffle control collapsed.
9. **Report**: ``report.json`` + ``report.md`` (+ saved model artifacts)
   under ``<report_root>/<dataset>/<target>/``.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import flybrain_brain_cluster_baselines as fbb
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

    @classmethod
    def from_samples(cls, samples: Sequence[TrainingSample | Mapping[str, Any]], *, dataset: str, target: str,
                     group_keys: Sequence[str], notes: Mapping[str, Any] | None = None) -> "EvalDataset":
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
        return cls(dataset, target, ids, labels, features, keys, groups, text, dict(notes or {}))

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, *, dataset: str, target: str, id_column: str, label_column: str,
                   feature_columns: Sequence[str], group_columns: Sequence[str],
                   text_column: str | None = None, notes: Mapping[str, Any] | None = None) -> "EvalDataset":
        overlap = set(feature_columns) & {id_column, label_column}
        if overlap:
            raise ValueError(f"feature_columns include id/label columns: {sorted(overlap)}")
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
        )

    def light_samples(self) -> list[TrainingSample]:
        return [
            TrainingSample(sample_id=sid, region_id="eval", input_text="-", expected_label=str(label),
                           expected_confidence=1.0, provenance_refs=("eval",), metadata=dict(meta))
            for sid, label, meta in zip(self.sample_ids, self.labels, self.group_values)
        ]


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
    calibrated = fl.CalibratedLearner.from_oof(base, oof, y, method=spec.calibration, classes=classes,
                                               cv=max(2, config.cv_folds))
    return base, calibrated


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

    report: dict[str, Any] = {
        "schema_version": EVAL_REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": data.dataset,
        "target": data.target,
        "config": config.as_dict(),
        "notes": fl._jsonable(dict(data.notes)),
        "leakage_check": leakage,
        "split": {**split_info,
                  "counts": {k: int(len(v)) for k, v in idx.items()},
                  "label_counts": {k: pd.Series(v).value_counts().sort_index().to_dict() for k, v in y.items()},
                  "test_labels_unseen_in_train": sorted(set(y["test"].tolist()) - set(classes)),
                  "n_test_components": int(len(set(components[idx["test"]].tolist())))},
        "n_features": len(feature_cols),
        "feature_families": fwf.feature_families(feature_cols),
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
        models: dict[str, Any] = {}
        fitted: dict[str, tuple[ModelSpec, dict[str, Any], fl.Learner]] = {}
        for spec in specs:
            t0 = time.time()
            tuning = _tune(spec, X_train, y["train"], g_train, classes, config)
            base, final = _fit_final(spec, tuning["best_params"], X_train, y["train"], g_train, classes,
                                     tuning["oof"], config)
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
                "tuning": {k: v for k, v in tuning.items() if k != "oof"},
                "fit_info": fl._jsonable(final.fit_info),
                "describe": final.describe() if spec.backend != "nb" else {"backend": "nb", "classes": list(final.classes_)},
                "val": {"raw": _strip(val_raw), "calibrated": _strip(val_final)},
                "test": {
                    **_strip(test),
                    "raw_uncalibrated": _strip(test_raw),
                    "reliability_table": fl.reliability_table(test_proba, y["test"], final.classes_, n_bins=config.ece_bins),
                    "confusion": confusion(y["test"].tolist(), test["predictions"]),
                },
                "fit_seconds": round(time.time() - t0, 3),
            }
            fitted[spec.label] = (spec, tuning["best_params"], final)
            models[spec.label]["_test_predictions"] = test["predictions"]

        # ---- bootstrap CIs + paired gains, gate
        for label, entry in models.items():
            view_base = baselines[entry["view"]]["splits"]["test"]
            best_rule = view_base["best_rule"]
            preds = {"model": entry.pop("_test_predictions"), "best_trivial": view_base["predictions"][best_rule],
                     "majority": view_base["predictions"][fbb.RULE_MAJORITY]}
            boot = grouped_bootstrap(y["test"].tolist(), preds, g_test, n_bootstrap=config.n_bootstrap,
                                     seed=config.seed, pairs=(("model", "best_trivial"), ("model", "majority")))
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
            if config.random_split_control:
                entry["random_split_control"] = _random_split_control(data, spec, params, config)
            ok_shuffle = entry.get("shuffle_control", {}).get("collapsed_to_majority", True)
            g = entry["gate"]
            g["shuffle_ok"] = ok_shuffle
            g["pass"] = bool(g["trivial_baseline_gate"]["pass"] and g["beats_trivial_macro_f1"]
                             and g["paired_gain_significant"] and ok_shuffle)

        best_label = sorted(models, key=lambda m: (-models[m]["val"]["calibrated"]["macro_f1"], m))[0]
        report["best_on_val"] = best_label
        if config.ablation:
            spec, params, _ = fitted[best_label]
            report["ablation"] = _ablation(data, spec, params, idx, y, g_train, feature_cols, config) \
                if not fl.make_learner(spec.backend).uses_text else {"skipped": "text model"}

        if config.report_root and config.save_models:
            out_dir = report_dir(config.report_root, data.dataset, data.target, config.run_label)
            for label, (spec, params, final) in fitted.items():
                manifest = fl.save_learner(final, out_dir / "models" / label, overwrite=True,
                                           metrics={"test": {k: v for k, v in models[label]["test"].items()
                                                             if k in ("accuracy", "macro_f1", "ece", "log_loss")},
                                                    "gate": models[label]["gate"]})
                models[label]["artifact"] = {"path": manifest["path"], "manifest_sha256": manifest["manifest_sha256"]}

    for view, result in baselines.items():
        for split in result["splits"].values():
            split.pop("predictions", None)
    report["baselines"] = baselines
    report["models"] = models
    report["summary"] = summary_rows(report)
    report["elapsed_seconds"] = round(time.time() - started, 2)
    if config.report_root:
        write_report(report, config.report_root, run_label=config.run_label)
    return report


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
        })
    return rows


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [f"# {report['dataset']} / {report['target']}", "",
             f"Generated {report['generated_at']}. Split: grouped by {', '.join(report['split']['group_keys']) or '(none)'}; "
             f"counts {report['split']['counts']}; {report['split']['n_test_components']} test components.", "",
             "## Held-out test", "",
             "| model | majority | best trivial (rule) | acc [95% CI] | macro-F1 | ECE | shuffle acc | gate |",
             "|---|---|---|---|---|---|---|---|"]
    for row in report["summary"]:
        lines.append(f"| {row['model']} | {_fmt(row['majority'])} | {_fmt(row['best_trivial'])} ({row['best_trivial_rule']}) | "
                     f"{_fmt(row['model_acc'])} {row['model_ci'] or ''} | {_fmt(row['macro_f1'])} | {_fmt(row['ece'])} | "
                     f"{_fmt(row['shuffle_acc'])} | {row['gate']} |")
    lines += ["", "## Gate detail", ""]
    for label, entry in report["models"].items():
        g = entry["gate"]
        lines.append(f"- **{label}**: paired acc gain CI95 {[round(x, 4) for x in g['paired_accuracy_gain_ci95']]}, "
                     f"macro-F1 {entry['test']['macro_f1']:.3f} vs trivial {g['best_trivial_macro_f1']:.3f}, "
                     f"shuffle ok {g['shuffle_ok']}, raw ECE {entry['test']['raw_uncalibrated']['ece']:.3f} -> "
                     f"calibrated {entry['test']['ece']:.3f}"
                     + (f", random-split acc {entry['random_split_control']['test_accuracy']:.3f}"
                        if "random_split_control" in entry else ""))
        failures = g["trivial_baseline_gate"].get("failure_reasons") or []
        for reason in failures:
            lines.append(f"  - {reason}")
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
