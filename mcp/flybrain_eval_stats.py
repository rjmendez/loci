"""Rigor statistics for the FlyBrain evaluation harness (research rules R1, R3, R4, R6, R7).

Pure functions plus two learner wrappers; ``flybrain_model_eval.run_evaluation``
wires them into the protocol. Citations refer to
``docs/flybrain-research/SYNTHESIS.md``.

* **R1 label provenance**: ``provenance_claim``. Gates apply only to
  ``measured`` and ``curated_morphology`` labels. Connectivity-defined labels are
  reported as recovery of connectivity-derived annotations, model-predicted
  labels as distillation of a classifier [Kapoor 2023 REFORMS; Codex FAQ].
* **R3 field baselines**: ``size_columns`` (the harness fits an HGB that
  sees only degree / synapse-count / cable-size features) and
  ``degree_decile_table`` [Bernett 2024; Subramonian 2024]; NT helpers
  ``normalize_nt``, ``binary_nt_label``, ``hemilineage_nt_oracle`` [Eckstein
  2024: 88% of hemilineages have one dominant NT; Lacin 2019].
* **R4 statistics**: ``cluster_bootstrap_indices`` (effective n = number of
  groups [Varoquaux 2017]), ``group_permuted_labels`` / ``permutation_null``
  (a block permutation over split components [Ojala & Garriga 2010]) and
  ``SplitLevel`` / ``split_curve_plan`` for the graded random -> type ->
  hemilineage -> hemisphere curve [Shchur 2018; Kapoor 2022 L3.2].
* **R6 calibration**: equal-width, equal-mass and sweep ECE [Roelofs 2022],
  a debiased L2 ECE [Kumar 2019], classwise ECE [Kull 2019], log loss and
  Brier with cluster-bootstrap CIs; Dirichlet (ODIR) [Kull 2019] and top-label
  histogram binning [Gupta & Ramdas 2022] calibrators; ``calibration_fold``
  carves a grouped calibration fold out of train [Ovadia 2019].
* **R7 hierarchy**: ``HierarchicalLearner`` (P(child) = P(parent) *
  P(child | parent), the WordTree pattern [Redmon 2017]) and
  ``hierarchy_metrics``. It is for reporting only and cannot be saved.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

import flybrain_learners as fl

_EPS = 1e-12

# =========================================================================== R1 provenance

LABEL_PROVENANCE = ("measured", "curated_morphology", "connectivity_defined", "model_predicted")
GATED_PROVENANCE = frozenset({"measured", "curated_morphology"})
PROVENANCE_UNSPECIFIED = "unspecified"


def provenance_claim(notes: Mapping[str, Any] | None) -> dict[str, Any]:
    """What a pass on this target may be called, from ``notes['label_provenance']``.

    Optional notes: ``label_classifier`` (the classifier distilled, for
    model_predicted) and ``provenance_uncertain`` (bool, e.g. BANC flow).
    An undeclared or unknown provenance is fail-closed: the gate is reported
    but does not apply.
    """
    notes = notes or {}
    raw = str(notes.get("label_provenance") or "").strip().lower()
    uncertain = bool(notes.get("provenance_uncertain", False))
    if raw not in LABEL_PROVENANCE:
        return {"label_provenance": raw or PROVENANCE_UNSPECIFIED, "provenance_uncertain": uncertain,
                "gate_applies": False,
                "claim": "label provenance not declared: the gate verdict is informational only"}
    if raw in GATED_PROVENANCE:
        claim = f"accuracy against {raw.replace('_', ' ')} labels"
        if uncertain:
            claim += " (provenance uncertain)"
        return {"label_provenance": raw, "provenance_uncertain": uncertain, "gate_applies": True, "claim": claim}
    if raw == "connectivity_defined":
        claim = "recovery of connectivity-derived annotations (not biological accuracy)"
    else:
        classifier = str(notes.get("label_classifier") or "the source classifier")
        claim = f"distillation of {classifier} (not ground-truth accuracy)"
    return {"label_provenance": raw, "provenance_uncertain": uncertain, "gate_applies": False, "claim": claim}


# =========================================================================== helpers


def _class_index(y: Sequence[Any], classes: Sequence[str]) -> np.ndarray:
    index = {c: i for i, c in enumerate(classes)}
    return np.asarray([index.get(str(v), -1) for v in y], dtype=np.int64)


def _top_label(proba: np.ndarray, y: Sequence[Any], classes: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    proba = np.asarray(proba, dtype=np.float64)
    if proba.shape[0] != len(y):
        raise ValueError("proba rows and labels differ")
    if len(y) == 0:
        return np.zeros(0), np.zeros(0)
    t = _class_index(y, classes)
    pred = proba.argmax(axis=1)
    return proba.max(axis=1), (pred == t).astype(np.float64)


def _equal_mass_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int) -> list[tuple[int, float, float]]:
    """(count, mean confidence, accuracy) per equal-mass bin, sorted by confidence."""
    order = np.argsort(conf, kind="stable")
    out = []
    for chunk in np.array_split(order, max(1, min(int(n_bins), len(order)))):
        if len(chunk):
            out.append((len(chunk), float(conf[chunk].mean()), float(correct[chunk].mean())))
    return out


# =========================================================================== R6 calibration metrics


def ece_equal_width(proba: np.ndarray, y: Sequence[Any], classes: Sequence[str], *, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width bins (the legacy harness metric, biased upward for small n)."""
    return fl.expected_calibration_error(proba, y, classes, n_bins=n_bins)


def ece_equal_mass(proba: np.ndarray, y: Sequence[Any], classes: Sequence[str], *, n_bins: int = 15) -> float:
    """Top-label ECE with equal-mass (quantile) bins [Roelofs 2022; Nixon 2019]."""
    conf, correct = _top_label(proba, y, classes)
    if len(conf) == 0:
        return 0.0
    n = float(len(conf))
    return float(sum(c * abs(a - m) for c, m, a in _equal_mass_bins(conf, correct, n_bins)) / n)


def ece_sweep(proba: np.ndarray, y: Sequence[Any], classes: Sequence[str], *, max_bins: int | None = None) -> dict[str, Any]:
    """ECE_sweep [Roelofs 2022]: equal-mass binning with the LARGEST bin count whose bin accuracies are monotone.

    Returns ``{"ece": L1 ECE at that bin count, "n_bins": b}``.
    """
    conf, correct = _top_label(proba, y, classes)
    n = len(conf)
    if n == 0:
        return {"ece": 0.0, "n_bins": 0}
    order = np.argsort(conf, kind="stable")
    cs, ks = conf[order], correct[order]
    ccum = np.concatenate([[0.0], np.cumsum(cs)])
    kcum = np.concatenate([[0.0], np.cumsum(ks)])
    limit = n if max_bins is None else min(n, int(max_bins))
    best_b, best_ece = 1, abs(ks.mean() - cs.mean())
    for b in range(2, limit + 1):
        edges = np.linspace(0, n, b + 1).round().astype(np.int64)
        counts = np.diff(edges)
        if np.any(counts == 0):
            break
        acc = (kcum[edges[1:]] - kcum[edges[:-1]]) / counts
        if np.any(np.diff(acc) < 0):
            break
        mean_conf = (ccum[edges[1:]] - ccum[edges[:-1]]) / counts
        best_b, best_ece = b, float(np.sum(counts * np.abs(acc - mean_conf)) / n)
    return {"ece": float(best_ece), "n_bins": int(best_b)}


def debiased_l2_ece(proba: np.ndarray, y: Sequence[Any], classes: Sequence[str], *, n_bins: int = 15) -> float:
    """Debiased top-label L2 calibration error [Kumar 2019], equal-mass bins.

    Each bin's squared gap has its sampling variance ``acc (1 - acc) / (n_b - 1)``
    subtracted; the result is ``sqrt(max(0, sum))``.
    """
    conf, correct = _top_label(proba, y, classes)
    n = float(len(conf))
    if n == 0:
        return 0.0
    total = 0.0
    for count, mean_conf, acc in _equal_mass_bins(conf, correct, n_bins):
        bias = acc * (1.0 - acc) / (count - 1) if count > 1 else 0.0
        total += (count / n) * ((acc - mean_conf) ** 2 - bias)
    return float(math.sqrt(max(0.0, total)))


def classwise_ece(proba: np.ndarray, y: Sequence[Any], classes: Sequence[str], *, n_bins: int = 15) -> float:
    """Classwise ECE [Kull 2019]: mean over classes of the one-vs-rest equal-width-bin ECE of p_k."""
    proba = np.asarray(proba, dtype=np.float64)
    if len(y) == 0 or proba.shape[1] == 0:
        return 0.0
    t = _class_index(y, classes)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    n = float(len(t))
    per_class = []
    for k in range(proba.shape[1]):
        p = proba[:, k]
        hit = (t == k).astype(np.float64)
        bins = np.clip(np.digitize(p, edges[1:-1], right=True), 0, int(n_bins) - 1)
        sum_p = np.bincount(bins, weights=p, minlength=n_bins)
        sum_h = np.bincount(bins, weights=hit, minlength=n_bins)
        per_class.append(float(np.sum(np.abs(sum_h - sum_p)) / n))
    return float(np.mean(per_class))


def calibration_metrics(proba: np.ndarray, y: Sequence[Any], classes: Sequence[str], *, n_bins: int = 15) -> dict[str, Any]:
    sweep = ece_sweep(proba, y, classes)
    return {
        "ece_equal_width": ece_equal_width(proba, y, classes, n_bins=n_bins),
        "ece_equal_mass": ece_equal_mass(proba, y, classes, n_bins=n_bins),
        "ece_sweep": sweep["ece"],
        "ece_sweep_bins": sweep["n_bins"],
        "debiased_l2_ece": debiased_l2_ece(proba, y, classes, n_bins=n_bins),
        "classwise_ece": classwise_ece(proba, y, classes, n_bins=n_bins),
        "log_loss": fl.multiclass_log_loss(proba, y, classes),
        "brier": fl.brier_score(proba, y, classes),
    }


CALIBRATION_METRIC_KEYS = ("ece_equal_width", "ece_equal_mass", "ece_sweep", "debiased_l2_ece", "classwise_ece",
                           "log_loss", "brier")


# =========================================================================== R4 cluster bootstrap


def effective_n(components: Sequence[Any]) -> int:
    """Effective sample size for grouped data = the number of distinct groups [Varoquaux 2017]."""
    return int(len(set(map(str, components))))


def cluster_bootstrap_indices(components: Sequence[Any], *, n_bootstrap: int, seed: int) -> list[np.ndarray]:
    """Row-index resamples that draw whole groups with replacement (a cluster bootstrap)."""
    comps = np.asarray([str(c) for c in components], dtype=object)
    unique = sorted(set(comps.tolist()))
    rows_of = {c: np.flatnonzero(comps == c) for c in unique}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(n_bootstrap)):
        picks = rng.integers(0, len(unique), size=len(unique))
        out.append(np.concatenate([rows_of[unique[i]] for i in picks]))
    return out


def _ci(values: Sequence[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]


def calibration_metrics_with_ci(proba: np.ndarray, y: Sequence[Any], classes: Sequence[str], components: Sequence[Any], *,
                                n_bootstrap: int = 200, seed: int = 0, n_bins: int = 15) -> dict[str, Any]:
    """``calibration_metrics`` plus 95% cluster-bootstrap CIs (groups resampled whole)."""
    proba = np.asarray(proba, dtype=np.float64)
    labels = np.asarray([str(v) for v in y], dtype=object)
    point = calibration_metrics(proba, labels, classes, n_bins=n_bins)
    samples: dict[str, list[float]] = {k: [] for k in CALIBRATION_METRIC_KEYS}
    for rows in cluster_bootstrap_indices(components, n_bootstrap=n_bootstrap, seed=seed):
        m = calibration_metrics(proba[rows], labels[rows], classes, n_bins=n_bins)
        for key in CALIBRATION_METRIC_KEYS:
            samples[key].append(m[key])
    ci = {f"{k}_ci95": _ci(v) for k, v in samples.items()} if n_bootstrap else {}
    return {**point, **ci, "n_bootstrap": int(n_bootstrap), "effective_n": effective_n(components),
            "method": "cluster_bootstrap_by_split_component"}


# =========================================================================== R6 calibrators


def fit_dirichlet(proba: np.ndarray, y: Sequence[Any], classes: Sequence[str], *, reg: float = 1e-2,
                  max_iter: int = 500) -> dict[str, Any]:
    """Dirichlet calibration with ODIR regularization [Kull 2019].

    Multinomial logistic regression on log-probabilities, z = W log p + b,
    initialized at the identity; ``reg`` penalizes the off-diagonal of W and b
    (scaled by K(K-1) and K). Returns a JSON-able state.
    """
    from scipy.optimize import minimize

    proba = np.asarray(proba, dtype=np.float64)
    k = len(classes)
    t = _class_index(y, classes)
    keep = t >= 0
    L = np.log(np.clip(proba[keep], _EPS, 1.0))
    t = t[keep]
    n = len(t)
    if n < 2:
        raise ValueError("Dirichlet calibration needs at least two labelled rows")
    onehot = np.zeros((n, k))
    onehot[np.arange(n), t] = 1.0
    off = 1.0 - np.eye(k)
    lam_w = float(reg) / max(1, k * (k - 1))
    lam_b = float(reg) / max(1, k)

    def loss(theta: np.ndarray) -> tuple[float, np.ndarray]:
        W = theta[: k * k].reshape(k, k)
        b = theta[k * k:]
        z = L @ W.T + b
        z = z - z.max(axis=1, keepdims=True)
        logsm = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        nll = -float((logsm * onehot).sum()) / n
        G = (np.exp(logsm) - onehot) / n
        gW = G.T @ L + 2 * lam_w * W * off
        gb = G.sum(axis=0) + 2 * lam_b * b
        return nll + lam_w * float(((W * off) ** 2).sum()) + lam_b * float((b ** 2).sum()), np.concatenate([gW.ravel(), gb])

    theta0 = np.concatenate([np.eye(k).ravel(), np.zeros(k)])
    result = minimize(loss, theta0, jac=True, method="L-BFGS-B", options={"maxiter": int(max_iter)})
    return {"method": "dirichlet", "classes": list(classes), "reg": float(reg), "W": result.x[: k * k].reshape(k, k).tolist(),
            "b": result.x[k * k:].tolist(), "converged": bool(result.success)}


def fit_dirichlet_cv(proba: np.ndarray, y: Sequence[Any], classes: Sequence[str], groups: Sequence[Any] | None = None, *,
                     regs: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0), n_splits: int = 3, seed: int = 0) -> dict[str, Any]:
    """Pick the ODIR strength by grouped K-fold log loss inside the calibration data, then refit on all of it."""
    proba = np.asarray(proba, dtype=np.float64)
    labels = np.asarray([str(v) for v in y], dtype=object)
    if groups is None or len(set(map(str, groups))) < n_splits or len(regs) == 1:
        state = fit_dirichlet(proba, labels, classes, reg=float(regs[min(1, len(regs) - 1)]))
        state["reg_selection"] = "default (too few groups for grouped CV)"
        return state
    scores = {}
    for reg in regs:
        total = 0.0
        for tr, te in fl.grouped_folds(groups, n_splits=n_splits, seed=seed):
            state = fit_dirichlet(proba[tr], labels[tr], classes, reg=reg)
            total += fl.multiclass_log_loss(apply_ext_calibrator(state, proba[te]), labels[te], classes) * len(te)
        scores[float(reg)] = total / len(labels)
    best = sorted(scores.items(), key=lambda kv: (kv[1], -kv[0]))[0][0]
    state = fit_dirichlet(proba, labels, classes, reg=best)
    state["reg_selection"] = {"grouped_cv_log_loss": scores, "chosen": best}
    return state


def fit_top_label(proba: np.ndarray, y: Sequence[Any], classes: Sequence[str], *, points_per_bin: int = 50) -> dict[str, Any]:
    """Top-label histogram binning [Gupta & Ramdas 2022].

    For each predicted class with at least ``2 * points_per_bin`` rows, the
    top-label confidence is equal-mass binned (``points_per_bin`` rows per
    bin) and replaced by the bin's accuracy; rarer predicted classes share one
    pooled binning. The other classes are rescaled to fill ``1 - conf``, so a
    row's argmax can change only when the calibrated confidence drops below a
    rescaled runner-up.
    """
    conf, correct = _top_label(proba, y, classes)
    pred = np.asarray(proba).argmax(axis=1) if len(conf) else np.zeros(0, dtype=np.int64)
    keep = _class_index(y, classes) >= 0
    conf, correct, pred = conf[keep], correct[keep], pred[keep]

    def binning(c: np.ndarray, k: np.ndarray) -> dict[str, list[float]]:
        n_bins = max(1, len(c) // max(1, int(points_per_bin)))
        order = np.argsort(c, kind="stable")
        chunks = [ch for ch in np.array_split(order, n_bins) if len(ch)]
        thresholds = [float((c[chunks[i][-1]] + c[chunks[i + 1][0]]) / 2.0) for i in range(len(chunks) - 1)]
        return {"thresholds": thresholds, "values": [float(k[ch].mean()) for ch in chunks]}

    per_class: dict[str, Any] = {}
    pooled_rows = np.zeros(len(conf), dtype=bool)
    for j, label in enumerate(classes):
        rows = pred == j
        if rows.sum() >= 2 * points_per_bin:
            per_class[label] = binning(conf[rows], correct[rows])
        else:
            pooled_rows |= rows
    pooled = binning(conf[pooled_rows], correct[pooled_rows]) if pooled_rows.any() else binning(conf, correct) \
        if len(conf) else {"thresholds": [], "values": [1.0]}
    return {"method": "top_label", "classes": list(classes), "points_per_bin": int(points_per_bin),
            "per_class": per_class, "pooled": pooled}


def apply_ext_calibrator(state: Mapping[str, Any], proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    if state["method"] == "dirichlet":
        W = np.asarray(state["W"], dtype=np.float64)
        b = np.asarray(state["b"], dtype=np.float64)
        z = np.log(np.clip(proba, _EPS, 1.0)) @ W.T + b
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)
    if state["method"] == "top_label":
        classes = list(state["classes"])
        out = proba.copy()
        if len(out) == 0:
            return out
        top = out.argmax(axis=1)
        conf = out[np.arange(len(out)), top]
        new = np.empty_like(conf)
        for i, (j, c) in enumerate(zip(top, conf)):
            spec = state["per_class"].get(classes[j], state["pooled"])
            new[i] = spec["values"][int(np.searchsorted(spec["thresholds"], c, side="right"))]
        new = np.clip(new, 1e-6, 1.0 - 1e-6)
        k = out.shape[1]
        rest = 1.0 - conf
        scale = np.where(rest > _EPS, (1.0 - new) / np.where(rest > _EPS, rest, 1.0), 0.0)
        out = out * scale[:, None]
        flat = rest <= _EPS
        if flat.any() and k > 1:
            out[flat] = ((1.0 - new[flat]) / (k - 1))[:, None]
        out[np.arange(len(out)), top] = new
        return out / out.sum(axis=1, keepdims=True)
    raise ValueError(f"unknown extended calibrator {state['method']!r}")


EXT_CALIBRATION_METHODS = ("dirichlet", "top_label")


class ExtCalibratedLearner(fl.Learner):
    """A fitted base learner plus a Dirichlet or top-label calibrator (JSON state). Saved and loaded like any learner."""

    backend = "calibrated_ext"
    default_params = {"method": "dirichlet", "base_backend": ""}

    def __init__(self, *, seed: int = 0, **params: Any) -> None:
        super().__init__(seed=seed, **params)
        if self.params["method"] not in EXT_CALIBRATION_METHODS:
            raise ValueError(f"extended calibration method must be one of {EXT_CALIBRATION_METHODS}")
        self.base: fl.Learner | None = None
        self.calibrator: dict[str, Any] = {}

    @property
    def model_files(self) -> tuple[str, ...]:  # type: ignore[override]
        base_files = () if self.base is None else tuple(f"base/{n}" for n in self.base.model_files)
        return ("calibrator.json",) + base_files

    @classmethod
    def from_proba(cls, fitted_base: fl.Learner, raw_proba: np.ndarray, y: Sequence[Any], *, method: str,
                   classes: Sequence[str] | None = None, groups: Sequence[Any] | None = None, source: str,
                   seed: int = 0) -> "ExtCalibratedLearner":
        """Calibrate ``fitted_base`` from probabilities it (or grouped CV) produced on rows it was NOT fitted on."""
        if not fitted_base.classes_:
            raise ValueError("base learner must be fitted")
        classes = tuple(classes or fitted_base.classes_)
        learner = cls(seed=seed, method=method, base_backend=fitted_base.backend)
        learner.base = fitted_base
        raw = fl.align_proba(np.asarray(raw_proba), classes, fitted_base.classes_)
        if method == "dirichlet":
            state = fit_dirichlet_cv(raw, y, fitted_base.classes_, groups, seed=seed)
        else:
            state = fit_top_label(raw, y, fitted_base.classes_)
        learner.calibrator = state
        learner.classes_ = fitted_base.classes_
        learner.schema = fitted_base.schema
        learner.uses_text = fitted_base.uses_text
        calibrated = apply_ext_calibrator(state, raw)
        learner.fit_info = {"calibration_source": source, "calibration_rows": int(len(raw)),
                            "ece_before": fl.expected_calibration_error(raw, y, learner.classes_),
                            "ece_after_in_sample": fl.expected_calibration_error(calibrated, y, learner.classes_),
                            "base_fit_info": fl._jsonable(fitted_base.fit_info)}
        learner.training_fingerprint = {**fitted_base.training_fingerprint, "calibration": dict(learner.params)}
        return learner

    def fit(self, X: pd.DataFrame, y: Sequence[Any], *, groups: Sequence[Any] | None = None) -> "ExtCalibratedLearner":
        raise NotImplementedError("use ExtCalibratedLearner.from_proba with held-out or grouped out-of-fold probabilities")

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.base is not None
        return apply_ext_calibrator(self.calibrator, self.base.predict_proba(X))

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out["base"] = None if self.base is None else self.base.describe()
        out["calibrator_method"] = self.params["method"]
        return out

    def _save_model(self, directory: Path) -> None:
        (directory / "calibrator.json").write_text(json.dumps(fl._jsonable(self.calibrator), sort_keys=True), encoding="utf-8")
        (directory / "base").mkdir()
        fl._write_learner_files(self.base, directory / "base")

    def _load_model(self, directory: Path) -> None:
        self.calibrator = json.loads((directory / "calibrator.json").read_text(encoding="utf-8"))
        self.base = fl._read_learner_files(directory / "base")
        self.uses_text = self.base.uses_text


if ExtCalibratedLearner.backend not in fl.available_backends():
    fl.register_backend(ExtCalibratedLearner.backend, ExtCalibratedLearner)


def calibration_fold(train_rows: np.ndarray, components: Sequence[Any], *, fraction: float, seed: Any
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Split train rows into (fit rows, calibration rows) by whole split components [Ovadia 2019].

    Components are ordered by ``sha256(seed:component)`` and dealt to the
    calibration fold until it holds ``fraction`` of the train rows.
    """
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("calibration fraction must be in (0, 1)")
    train_rows = np.asarray(train_rows, dtype=np.int64)
    comps = np.asarray([str(components[i]) for i in train_rows], dtype=object)
    unique = sorted(set(comps.tolist()), key=lambda c: hashlib.sha256(f"{seed}:cal:{c}".encode()).hexdigest())
    if len(unique) < 2:
        raise ValueError("a grouped calibration fold needs at least two train components")
    sizes = pd.Series(comps).value_counts().to_dict()
    target = float(fraction) * len(train_rows)
    cal: set[str] = set()
    filled = 0
    for c in unique[:-1]:  # never take every component
        if filled >= target:
            break
        cal.add(c)
        filled += sizes[c]
    in_cal = np.asarray([c in cal for c in comps])
    return train_rows[~in_cal], train_rows[in_cal]


# =========================================================================== R3 size / degree baseline

DEFAULT_SIZE_PATTERNS = (
    "degree__*", "size__*", "sparsity__*", "*total_synapses*", "*n_synapses*", "*synapse_count*",
    "*cable_length*", "morph__*cable*", "morph__*volume*", "morph__*area*", "*n_pre*", "*n_post*",
)


def size_columns(columns: Sequence[str], patterns: Sequence[str] = DEFAULT_SIZE_PATTERNS) -> list[str]:
    import fnmatch

    return sorted(c for c in columns if any(fnmatch.fnmatchcase(c.lower(), p.lower()) for p in patterns))


def degree_values(frame: pd.DataFrame, column: str | None = None) -> tuple[str | None, np.ndarray | None]:
    """The size axis for decile tables: ``column`` if given, else total synapse weight, else the first size column."""
    if column is not None:
        if column not in frame.columns:
            return None, None
        return column, pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if "size__total_degree" in frame.columns:  # fme.size_side_aux: out + in weight total (raw or rank)
        return "size__total_degree", pd.to_numeric(frame["size__total_degree"], errors="coerce").to_numpy(
            dtype=np.float64)
    if {"degree__out_weight_total", "degree__in_weight_total"} <= set(frame.columns):
        total = (pd.to_numeric(frame["degree__out_weight_total"], errors="coerce").fillna(0)
                 + pd.to_numeric(frame["degree__in_weight_total"], errors="coerce").fillna(0))
        return "degree__out_weight_total+degree__in_weight_total", total.to_numpy(dtype=np.float64)
    cols = [c for c in size_columns(frame.columns) if pd.api.types.is_numeric_dtype(frame[c].dtype)]
    if not cols:
        return None, None
    return cols[0], pd.to_numeric(frame[cols[0]], errors="coerce").to_numpy(dtype=np.float64)


def degree_decile_table(train_degree: np.ndarray, test_degree: np.ndarray, y_true: Sequence[str],
                        predictions: Mapping[str, Sequence[str]], components: Sequence[Any], *, n_bins: int = 10
                        ) -> list[dict[str, Any]]:
    """Accuracy per size decile [Subramonian 2024]; decile edges come from TRAIN degrees only."""
    train_degree = np.asarray(train_degree, dtype=np.float64)
    finite = train_degree[np.isfinite(train_degree)]
    qs = np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1)[1:-1])) if len(finite) else np.zeros(0)
    test_degree = np.asarray(test_degree, dtype=np.float64)
    bins = np.where(np.isfinite(test_degree), np.searchsorted(qs, test_degree, side="right"), -1)
    y = np.asarray(list(y_true), dtype=object)
    comps = np.asarray([str(c) for c in components], dtype=object)
    lo_edges = np.concatenate([[-np.inf], qs])
    hi_edges = np.concatenate([qs, [np.inf]])
    rows = []
    for b in sorted(set(bins.tolist())):
        mask = bins == b
        row: dict[str, Any] = {"decile": "missing" if b < 0 else int(b), "n": int(mask.sum()),
                               "n_components": int(len(set(comps[mask].tolist())))}
        if b >= 0:
            row["degree_lo"] = None if not np.isfinite(lo_edges[b]) else float(lo_edges[b])
            row["degree_hi"] = None if not np.isfinite(hi_edges[b]) else float(hi_edges[b])
        for name, pred in predictions.items():
            p = np.asarray(list(pred), dtype=object)
            row[f"acc_{name}"] = float(np.mean(p[mask] == y[mask]))
        rows.append(row)
    return rows


# =========================================================================== R3 NT helpers

NT_ACH, NT_GABA, NT_GLU = "acetylcholine", "gaba", "glutamate"
NT_BINARY_EXC, NT_BINARY_INH = "ach", "gaba_glu"
_NT_ALIASES = {
    NT_ACH: ("acetylcholine", "ach", "cholinergic", "dominant_acetylcholine", "dominant_ach"),
    NT_GABA: ("gaba", "gabaergic", "dominant_gaba"),
    NT_GLU: ("glutamate", "glu", "glut", "glutamatergic", "dominant_glutamate", "dominant_glu"),
}
_NT_LOOKUP = {alias: nt for nt, aliases in _NT_ALIASES.items() for alias in aliases}


def normalize_nt(label: Any) -> str | None:
    """Map a dataset NT spelling to acetylcholine / gaba / glutamate; anything else -> None."""
    key = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    return _NT_LOOKUP.get(key)


def binary_nt_label(label: Any) -> str | None:
    """ACh vs (GABA + Glu) [synthesis sec. 5 add 2]; other transmitters -> None (dropped from the view)."""
    nt = normalize_nt(label)
    if nt is None:
        return None
    return NT_BINARY_EXC if nt == NT_ACH else NT_BINARY_INH


def _known(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value)) and \
        str(value).strip().lower() not in ("", "nan", "none", "unknown", "null", "<na>")


def hemilineage_nt_oracle(train_hl: Sequence[Any], y_train: Sequence[str], test_hl: Sequence[Any],
                          y_test: Sequence[str]) -> dict[str, Any]:
    """Two hemilineage -> NT references [Lacin 2019; Eckstein 2024].

    * ``train_lookup``: the train-set hemilineage -> majority NT table applied
      to test (unseen hemilineages -> train majority). An honest baseline; under
      a hemilineage-grouped split it collapses to the majority.
    * ``leave_one_out``: within the TEST set, each neuron gets the majority NT
      of the OTHER test neurons of its hemilineage. An upper bound on how much
      NT the lineage alone explains; not a deployable rule.
    """
    y_train = [str(v) for v in y_train]
    y_test = [str(v) for v in y_test]
    majority = pd.Series(y_train).value_counts().sort_index().idxmax() if y_train else ""
    table: dict[str, dict[str, int]] = {}
    for h, lab in zip(train_hl, y_train):
        if _known(h):
            table.setdefault(str(h), {}).setdefault(lab, 0)
            table[str(h)][lab] += 1

    def top(counts: Mapping[str, int]) -> str:
        best = max(counts.values())
        return sorted(k for k, v in counts.items() if v == best)[0]

    lookup = {h: top(c) for h, c in table.items()}
    pred_train = [lookup.get(str(h), majority) if _known(h) else majority for h in test_hl]
    covered = [(_known(h) and str(h) in lookup) for h in test_hl]
    test_table: dict[str, dict[str, int]] = {}
    for h, lab in zip(test_hl, y_test):
        if _known(h):
            test_table.setdefault(str(h), {}).setdefault(lab, 0)
            test_table[str(h)][lab] += 1
    pred_loo = []
    loo_covered = []
    for h, lab in zip(test_hl, y_test):
        if _known(h):
            counts = dict(test_table[str(h)])
            counts[lab] -= 1
            counts = {k: v for k, v in counts.items() if v > 0}
            if counts:
                pred_loo.append(top(counts))
                loo_covered.append(True)
                continue
        pred_loo.append(majority)
        loo_covered.append(False)
    def acc(p: Sequence[str]) -> float:
        return float(np.mean(np.asarray(p, dtype=object) == np.asarray(y_test, dtype=object))) if y_test else 0.0

    return {
        "train_lookup": {"accuracy": acc(pred_train), "coverage": float(np.mean(covered)) if covered else 0.0,
                         "predictions": pred_train},
        "leave_one_out": {"accuracy": acc(pred_loo), "coverage": float(np.mean(loo_covered)) if loo_covered else 0.0,
                          "predictions": pred_loo},
        "n_train_hemilineages": len(lookup),
        "n_test_hemilineages": len(test_table),
        "table": lookup,
    }


# =========================================================================== R4 permutation null


def group_permuted_labels(y: Sequence[str], components: Sequence[Any], rng: np.random.Generator) -> np.ndarray:
    """A block permutation of labels over groups.

    Label blocks (one per component, in a random component order) are laid
    end to end over the rows of the components in a second random order. The
    label marginals are kept exactly, and a component that received a stretch
    of one donor stays label-homogeneous, so grouped (within-type) label
    correlation survives, which a per-row shuffle destroys [Ojala & Garriga 2010].
    """
    y = np.asarray([str(v) for v in y], dtype=object)
    comps = np.asarray([str(c) for c in components], dtype=object)
    unique = sorted(set(comps.tolist()))
    rows_of = {c: np.flatnonzero(comps == c) for c in unique}
    donors = [unique[i] for i in rng.permutation(len(unique))]
    slots = [unique[i] for i in rng.permutation(len(unique))]
    sequence = np.concatenate([y[rows_of[c]] for c in donors]) if unique else np.zeros(0, dtype=object)
    out = np.empty_like(y)
    pos = 0
    for c in slots:
        rows = rows_of[c]
        out[rows] = sequence[pos:pos + len(rows)]
        pos += len(rows)
    return out


def permutation_null(fit_predict: Callable[[np.ndarray], Sequence[str]], y_train: Sequence[str],
                     train_components: Sequence[Any], y_test: Sequence[str], *, observed_accuracy: float,
                     observed_macro_f1: float | None = None, n_permutations: int = 100, seed: int = 0,
                     n_jobs: int = 1) -> dict[str, Any]:
    """Refit on ``n_permutations`` group-permuted train labels and score on test.

    ``p_value = (1 + #{null >= observed}) / (1 + n)``. Label vectors are drawn
    sequentially from one seeded generator; the refits can run on threads.
    """
    from flybrain_model_eval import macro_f1  # local import: avoid a cycle at module import

    rng = np.random.default_rng(seed)
    permuted = [group_permuted_labels(y_train, train_components, rng) for _ in range(int(n_permutations))]
    truth = np.asarray([str(v) for v in y_test], dtype=object)

    def one(labels: np.ndarray) -> tuple[float, float]:
        pred = np.asarray(list(fit_predict(labels)), dtype=object)
        return float(np.mean(pred == truth)), float(macro_f1(truth.tolist(), pred.tolist()))

    if n_jobs and n_jobs > 1 and permuted:
        from joblib import Parallel, delayed

        scores = Parallel(n_jobs=int(n_jobs), backend="threading")(delayed(one)(p) for p in permuted)
    else:
        scores = [one(p) for p in permuted]
    accs = np.asarray([s[0] for s in scores], dtype=np.float64)
    f1s = np.asarray([s[1] for s in scores], dtype=np.float64)
    n = len(accs)
    out: dict[str, Any] = {
        "n_permutations": n,
        "method": "group_block_permutation_of_train_labels",
        "observed_accuracy": float(observed_accuracy),
        "null_accuracy_mean": float(accs.mean()) if n else None,
        "null_accuracy_q95": float(np.quantile(accs, 0.95)) if n else None,
        "null_accuracy_max": float(accs.max()) if n else None,
        "p_value_accuracy": float((1 + np.sum(accs >= observed_accuracy - 1e-12)) / (1 + n)),
        "null_accuracies": [round(float(a), 6) for a in accs],
    }
    if observed_macro_f1 is not None:
        out["observed_macro_f1"] = float(observed_macro_f1)
        out["null_macro_f1_mean"] = float(f1s.mean()) if n else None
        out["p_value_macro_f1"] = float((1 + np.sum(f1s >= observed_macro_f1 - 1e-12)) / (1 + n))
    return out


# =========================================================================== R4 graded split curve

SIDE_KEYS = ("side", "hemisphere", "soma_side")


@dataclass(frozen=True)
class SplitLevel:
    """One rung of the split curve.

    ``group_keys`` = () is a per-sample random split. ``holdout_key`` makes a
    transfer split instead: train on rows whose value is in ``train_values``,
    test on rows in ``test_values`` (e.g. side left -> right).
    """

    name: str
    group_keys: tuple[str, ...] = ()
    holdout_key: str | None = None
    train_values: tuple[str, ...] = ()
    test_values: tuple[str, ...] = ()
    notes: Mapping[str, Any] = field(default_factory=dict)


def default_split_levels(group_keys: Sequence[str], available_keys: Sequence[str]) -> list[SplitLevel]:
    """random -> cumulative group keys (type, then + hemilineage, ...) -> hemisphere (left -> right) when a side column exists."""
    levels = [SplitLevel("random")]
    acc: list[str] = []
    for key in group_keys:
        acc.append(key)
        levels.append(SplitLevel("+".join(acc), tuple(acc)))
    side = next((k for k in SIDE_KEYS if k in available_keys), None)
    if side is not None:
        levels.append(SplitLevel(f"hemisphere({side}:left->right)", (), side, ("left", "l"), ("right", "r")))
    return levels


def split_curve_plan(sample_ids: Sequence[str], metadata: Sequence[Mapping[str, Any]], level: SplitLevel, *,
                     split_seed: str, train_ratio: float, val_ratio: float) -> dict[str, Any]:
    """Row indices ``train`` / ``test`` and grouping info for one level (val is left unused)."""
    from flybrain_brain_cluster_training import (
        TrainingSample, deterministic_split_ids, grouped_split_ids, split_group_components)

    position = {sid: i for i, sid in enumerate(sample_ids)}
    samples = [TrainingSample(sample_id=str(sid), region_id="eval", input_text="-", expected_label="-",
                              expected_confidence=1.0, provenance_refs=("eval",), metadata=dict(meta))
               for sid, meta in zip(sample_ids, metadata)]
    if level.holdout_key is not None:
        values = [str(m.get(level.holdout_key, "")).strip().lower() for m in metadata]
        tr_vals = {v.lower() for v in level.train_values}
        te_vals = {v.lower() for v in level.test_values}
        train = np.flatnonzero([v in tr_vals for v in values])
        test = np.flatnonzero([v in te_vals for v in values])
        return {"train": train, "test": test, "components": np.asarray([str(s) for s in sample_ids], dtype=object),
                "kind": "holdout", "holdout_key": level.holdout_key}
    if not level.group_keys:
        tr, _, te = deterministic_split_ids(list(map(str, sample_ids)), split_seed=split_seed,
                                            train_ratio=train_ratio, val_ratio=val_ratio)
        comps = np.asarray([str(s) for s in sample_ids], dtype=object)
    else:
        tr, _, te, _ = grouped_split_ids(samples, group_keys=level.group_keys, split_seed=split_seed,
                                         train_ratio=train_ratio, val_ratio=val_ratio)
        comp_of = split_group_components(samples, group_keys=level.group_keys)
        comps = np.asarray([comp_of[str(s)] for s in sample_ids], dtype=object)
    return {"train": np.asarray(sorted(position[s] for s in tr), dtype=np.int64),
            "test": np.asarray(sorted(position[s] for s in te), dtype=np.int64),
            "components": comps, "kind": "grouped" if level.group_keys else "random"}


# =========================================================================== R7 hierarchy


class HierarchicalLearner(fl.Learner):
    """P(child | x) = P(parent | x) * P(child | x, parent) [Redmon 2017; scHPL, Michielsen 2021].

    ``parent_of`` maps every child label to its parent (e.g. cell_class ->
    super_class, built from TRAIN rows). A parent with one child needs no
    child model. For reporting only (R7: report, don't ship): it cannot be saved.
    """

    backend = "hierarchical"
    default_params = {"base_backend": "hgb", "base_params": {}, "parent_of": {}}

    def __init__(self, *, seed: int = 0, **params: Any) -> None:
        super().__init__(seed=seed, **params)
        self.parent_model: fl.Learner | None = None
        self.children: dict[str, fl.Learner | str] = {}

    def _fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None) -> None:
        parent_of = {str(k): str(v) for k, v in dict(self.params["parent_of"]).items()}
        missing = sorted(set(y.tolist()) - set(parent_of))
        if missing:
            raise ValueError(f"no parent for child labels: {missing[:5]}")
        parents = np.asarray([parent_of[v] for v in y], dtype=object)
        backend, params = self.params["base_backend"], dict(self.params["base_params"])
        self.parent_model = fl.make_learner(backend, seed=self.seed, **params).fit(X, parents, groups=groups)
        self.children = {}
        for parent in self.parent_model.classes_:
            rows = np.flatnonzero(parents == parent)
            kids = sorted(set(y[rows].tolist()))
            if len(kids) == 1:
                self.children[parent] = kids[0]
            else:
                self.children[parent] = fl.make_learner(backend, seed=self.seed, **params).fit(
                    X.iloc[rows], y[rows], groups=None if groups is None else groups[rows])
        self.fit_info = {"n_parents": len(self.children),
                         "single_child_parents": sorted(p for p, c in self.children.items() if isinstance(c, str))}

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        assert self.parent_model is not None
        index = {c: i for i, c in enumerate(self.classes_)}
        out = np.zeros((len(X), len(self.classes_)))
        pp = self.parent_model.predict_proba(X)
        for j, parent in enumerate(self.parent_model.classes_):
            child = self.children[parent]
            if isinstance(child, str):
                out[:, index[child]] += pp[:, j]
            else:
                cp = child.predict_proba(X)
                for k, label in enumerate(child.classes_):
                    out[:, index[label]] += pp[:, j] * cp[:, k]
        return out

    def _save_model(self, directory: Path) -> None:
        raise NotImplementedError("HierarchicalLearner is a report-only prototype (R7); it is not saved")

    def _load_model(self, directory: Path) -> None:  # pragma: no cover
        raise NotImplementedError


if HierarchicalLearner.backend not in fl.available_backends():
    fl.register_backend(HierarchicalLearner.backend, HierarchicalLearner)


def parent_map(children: Sequence[Any], parents: Sequence[Any]) -> dict[str, str]:
    """child -> most frequent parent over the given rows (ties -> lexicographically first)."""
    counts: dict[str, dict[str, int]] = {}
    for c, p in zip(children, parents):
        if _known(p):
            counts.setdefault(str(c), {}).setdefault(str(p), 0)
            counts[str(c)][str(p)] += 1
    out = {}
    for c, pc in counts.items():
        best = max(pc.values())
        out[c] = sorted(k for k, v in pc.items() if v == best)[0]
    return out


def hierarchy_metrics(y_true: Sequence[str], y_pred: Sequence[str], parent_of: Mapping[str, str]) -> dict[str, Any]:
    """Parent-level accuracy and mistake severity: the share of errors that stay inside the true parent."""
    y_true = [str(v) for v in y_true]
    y_pred = [str(v) for v in y_pred]
    tp = [parent_of.get(v) for v in y_true]
    pp = [parent_of.get(v) for v in y_pred]
    errors = [(t, p) for t, p, a, b in zip(tp, pp, y_true, y_pred) if a != b]
    return {
        "parent_accuracy": float(np.mean([a == b and a is not None for a, b in zip(tp, pp)])) if y_true else 0.0,
        "n_errors": len(errors),
        "errors_within_true_parent": float(np.mean([t == p and t is not None for t, p in errors])) if errors else None,
    }


# =========================================================================== gate assembly

GATE_CRITERIA = ("beats_trivial_accuracy", "beats_trivial_macro_f1", "paired_gain_significant",
                 "beats_size_baseline", "null_control_ok")


def assemble_gate(checks: Mapping[str, Any]) -> dict[str, Any]:
    """AND of every criterion in ``GATE_CRITERIA``; a missing or None criterion fails closed.

    Returns ``{"pass": bool, "failed": [...]}``. Every criterion must be present
    as True for a pass, so removing a check anywhere upstream fails the gate
    rather than silently passing it.
    """
    failed = [name for name in GATE_CRITERIA if checks.get(name) is not True]
    return {"pass": not failed, "failed": failed}


__all__ = [
    "CALIBRATION_METRIC_KEYS", "DEFAULT_SIZE_PATTERNS", "EXT_CALIBRATION_METHODS", "ExtCalibratedLearner",
    "GATED_PROVENANCE", "GATE_CRITERIA", "HierarchicalLearner", "LABEL_PROVENANCE", "NT_BINARY_EXC", "NT_BINARY_INH",
    "SplitLevel", "apply_ext_calibrator", "assemble_gate", "binary_nt_label", "calibration_fold", "calibration_metrics",
    "calibration_metrics_with_ci", "classwise_ece", "cluster_bootstrap_indices", "debiased_l2_ece",
    "default_split_levels", "degree_decile_table", "degree_values", "ece_equal_mass", "ece_equal_width", "ece_sweep",
    "effective_n", "fit_dirichlet", "fit_dirichlet_cv", "fit_top_label", "group_permuted_labels",
    "hemilineage_nt_oracle", "hierarchy_metrics", "normalize_nt", "parent_map", "permutation_null",
    "provenance_claim", "size_columns", "split_curve_plan",
]
