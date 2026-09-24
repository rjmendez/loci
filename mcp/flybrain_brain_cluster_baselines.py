"""Trivial-rule baselines and the promotion gate that must beat them (roadmap B1 / AC6).

A brain-cluster model is only worth promoting if it does better on held-out
samples than the obvious rule a person would write without any training:

* ``majority``: always predict the most common training label.
* ``threshold``: a one-feature decision stump on a numeric feature that is
  visible in ``input_text`` (e.g. fw ``total_pre_count``, the very quantity the
  connectivity_tier label is thresholded from).
* ``argmax``: pick the label whose score feature is largest in ``input_text``
  (e.g. fw ``ach_avg`` / ``gaba_avg`` / ... for neurotransmitter_dominance).

Rules only see what the model sees (``input_text``). Each rule is fitted on
the train split and scored on the held-out split. The gate fails unless the
model's held-out accuracy is at least ``best trivial accuracy + min_margin``.
A tautological objective (label recoverable from one input feature) therefore
fails the gate: its trivial accuracy is ~1.0 and nothing can beat it.

Pure python, deterministic, no I/O.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

TRIVIAL_BASELINE_SCHEMA_VERSION = "braincluster-trivial-baseline/v1"
RULE_MAJORITY = "majority"
RULE_THRESHOLD = "threshold"
RULE_ARGMAX = "argmax"

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LABEL_PREFIXES = ("dominant_",)
_SCORE_SUFFIXES = ("", "_avg", "_score", "_prob", "_probability", "_mean")


@dataclass(frozen=True)
class BaselineGateConfig:
    """``min_margin``: absolute accuracy the model must add over the best trivial rule."""

    min_margin: float = 0.01
    min_heldout_samples: int = 1

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.min_margin) <= 1.0):
            raise ValueError("min_margin must be in [0, 1]")
        if int(self.min_heldout_samples) < 1:
            raise ValueError("min_heldout_samples must be >= 1")

    def as_dict(self) -> dict[str, Any]:
        return {"min_margin": float(self.min_margin), "min_heldout_samples": int(self.min_heldout_samples)}

    @classmethod
    def from_value(cls, value: "BaselineGateConfig | Mapping[str, Any] | None") -> "BaselineGateConfig":
        if value is None:
            return cls()
        if isinstance(value, BaselineGateConfig):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("baseline gate config must be a BaselineGateConfig or mapping")
        unknown = sorted(set(value) - {"min_margin", "min_heldout_samples"})
        if unknown:
            raise ValueError(f"unknown baseline gate keys: {', '.join(unknown)}")
        default = cls()
        return cls(
            min_margin=float(value.get("min_margin", default.min_margin)),
            min_heldout_samples=int(value.get("min_heldout_samples", default.min_heldout_samples)),
        )


def parse_input_features(text: str) -> dict[str, str]:
    """Parse ``key value key value ...`` input text; later duplicates are ignored."""
    tokens = str(text).split()
    features: dict[str, str] = {}
    for index in range(0, len(tokens) - 1, 2):
        key, value = tokens[index], tokens[index + 1]
        if _KEY_RE.match(key) and key not in features:
            features[key] = value
    return features


def _as_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _numeric_rows(rows: Sequence[tuple[str, str]]) -> dict[str, list[float]]:
    """Numeric features present and finite in every row."""
    parsed = [parse_input_features(text) for text, _ in rows]
    if not parsed:
        return {}
    common = set(parsed[0])
    for features in parsed[1:]:
        common &= set(features)
    out: dict[str, list[float]] = {}
    for key in sorted(common):
        values = [_as_float(features[key]) for features in parsed]
        if all(value is not None for value in values):
            out[key] = [float(value) for value in values]  # type: ignore[arg-type]
    return out


def _majority_label(labels: Sequence[str]) -> str:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _accuracy(predicted: Sequence[str], actual: Sequence[str]) -> float:
    if not actual:
        return 0.0
    return sum(1 for p, a in zip(predicted, actual) if p == a) / float(len(actual))


def _best_label(counts: Mapping[str, int]) -> tuple[str, int]:
    if not counts:
        return "", 0
    label, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return label, count


def fit_threshold_rule(rows: Sequence[tuple[str, str]]) -> dict[str, Any] | None:
    """Best single-feature stump over numeric input features (train rows only).

    Predicts ``above_label`` when ``feature >= threshold`` else ``below_label``;
    each side's label is its majority label. Returns ``None`` when no numeric
    feature exists or no split beats predicting one label everywhere.
    """
    labels = [label for _, label in rows]
    numeric = _numeric_rows(rows)
    if not numeric or len(set(labels)) < 2:
        return None
    total_counts: dict[str, int] = {}
    for label in labels:
        total_counts[label] = total_counts.get(label, 0) + 1
    _, constant_correct = _best_label(total_counts)
    best: tuple[int, str, float, str, str] | None = None
    for feature in sorted(numeric):
        pairs = sorted(zip(numeric[feature], labels), key=lambda item: item[0])
        left: dict[str, int] = {}
        right = dict(total_counts)
        for index in range(1, len(pairs)):
            value, label = pairs[index - 1]
            left[label] = left.get(label, 0) + 1
            right[label] -= 1
            if pairs[index][0] == value:
                continue
            below_label, below_correct = _best_label(left)
            above_label, above_correct = _best_label({k: v for k, v in right.items() if v > 0})
            correct = below_correct + above_correct
            threshold = (value + pairs[index][0]) / 2.0
            candidate = (correct, feature, threshold, above_label, below_label)
            if best is None or correct > best[0]:
                best = candidate
    if best is None or best[0] <= constant_correct:
        return None
    correct, feature, threshold, above_label, below_label = best
    return {
        "feature": feature,
        "threshold": float(threshold),
        "above_label": above_label,
        "below_label": below_label,
        "train_accuracy": correct / float(len(rows)),
    }


def apply_threshold_rule(rule: Mapping[str, Any], text: str) -> str | None:
    value = _as_float(parse_input_features(text).get(str(rule["feature"]), ""))
    if value is None:
        return None
    return str(rule["above_label"]) if value >= float(rule["threshold"]) else str(rule["below_label"])


def _label_stem(label: str) -> str:
    for prefix in _LABEL_PREFIXES:
        if label.startswith(prefix):
            return label[len(prefix):]
    return label


def fit_argmax_rule(rows: Sequence[tuple[str, str]]) -> dict[str, Any] | None:
    """Map labels to score features in ``input_text`` (``dominant_ach`` -> ``ach_avg``).

    Applicable only when at least two training labels have a score feature.
    """
    numeric = _numeric_rows(rows)
    if not numeric:
        return None
    mapping: dict[str, str] = {}
    for label in sorted({label for _, label in rows}):
        stem = _label_stem(label)
        for suffix in _SCORE_SUFFIXES:
            key = f"{stem}{suffix}"
            if key in numeric:
                mapping[label] = key
                break
    if len(mapping) < 2:
        return None
    return {"label_to_feature": dict(sorted(mapping.items()))}


def apply_argmax_rule(rule: Mapping[str, Any], text: str) -> str | None:
    features = parse_input_features(text)
    scored: list[tuple[float, str]] = []
    for label, key in dict(rule["label_to_feature"]).items():
        value = _as_float(features.get(key, ""))
        if value is not None:
            scored.append((value, label))
    if not scored:
        return None
    best_value = max(value for value, _ in scored)
    return sorted(label for value, label in scored if value == best_value)[0]


def evaluate_trivial_baselines(
    train_rows: Sequence[tuple[str, str]],
    heldout_rows: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Fit every applicable trivial rule on ``train_rows``; score on ``heldout_rows``.

    Rows are ``(input_text, expected_label)``. A rule that cannot make a
    prediction for a held-out row counts that row as wrong.
    """
    if not train_rows:
        raise ValueError("trivial baselines need at least one train row")
    actual = [label for _, label in heldout_rows]
    rules: dict[str, Any] = {}

    majority = _majority_label([label for _, label in train_rows])
    rules[RULE_MAJORITY] = {
        "applicable": True,
        "params": {"label": majority},
        "heldout_accuracy": _accuracy([majority] * len(actual), actual),
    }

    threshold = fit_threshold_rule(train_rows)
    if threshold is None:
        rules[RULE_THRESHOLD] = {"applicable": False, "params": None, "heldout_accuracy": None}
    else:
        predicted = [apply_threshold_rule(threshold, text) or "" for text, _ in heldout_rows]
        rules[RULE_THRESHOLD] = {
            "applicable": True,
            "params": threshold,
            "heldout_accuracy": _accuracy(predicted, actual),
        }

    argmax = fit_argmax_rule(train_rows)
    if argmax is None:
        rules[RULE_ARGMAX] = {"applicable": False, "params": None, "heldout_accuracy": None}
    else:
        predicted = [apply_argmax_rule(argmax, text) or "" for text, _ in heldout_rows]
        rules[RULE_ARGMAX] = {
            "applicable": True,
            "params": argmax,
            "heldout_accuracy": _accuracy(predicted, actual),
        }

    applicable = [(name, row["heldout_accuracy"]) for name, row in rules.items() if row["applicable"]]
    best_rule, best_accuracy = sorted(applicable, key=lambda item: (-float(item[1]), item[0]))[0]
    return {
        "train_count": len(train_rows),
        "heldout_count": len(heldout_rows),
        "rules": rules,
        "best_rule": best_rule,
        "best_accuracy": float(best_accuracy),
    }


def trivial_baseline_gate(
    *,
    model_heldout_accuracy: float | None,
    baselines: Mapping[str, Any] | None,
    heldout_count: int,
    config: BaselineGateConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pass only if the model beats the best trivial rule by ``min_margin`` on held-out data."""
    cfg = BaselineGateConfig.from_value(config)
    failures: list[str] = []
    required: float | None = None
    best = None if baselines is None else float(baselines["best_accuracy"])
    if heldout_count < cfg.min_heldout_samples:
        failures.append(f"held-out split too small for the trivial-baseline gate ({heldout_count} < "
                        f"{cfg.min_heldout_samples})")
    if best is not None:
        required = best + float(cfg.min_margin)
    if model_heldout_accuracy is None or best is None:
        failures.append("trivial-baseline gate has no held-out measurement")
    elif required is not None and float(model_heldout_accuracy) < required:
        failures.append(
            "model does not beat the trivial baseline on held-out data "
            f"({float(model_heldout_accuracy):.3f} < {best:.3f} [{baselines['best_rule']}] + margin "
            f"{float(cfg.min_margin):.3f})"
        )
    return {
        "schema_version": TRIVIAL_BASELINE_SCHEMA_VERSION,
        "pass": not failures,
        "config": cfg.as_dict(),
        "model_heldout_accuracy": None if model_heldout_accuracy is None else float(model_heldout_accuracy),
        "best_trivial_rule": None if baselines is None else baselines["best_rule"],
        "best_trivial_accuracy": best,
        "required_accuracy": required,
        "heldout_count": int(heldout_count),
        "baselines": dict(baselines) if baselines is not None else None,
        "failure_reasons": failures,
    }


__all__ = [
    "BaselineGateConfig",
    "RULE_ARGMAX",
    "RULE_MAJORITY",
    "RULE_THRESHOLD",
    "TRIVIAL_BASELINE_SCHEMA_VERSION",
    "apply_argmax_rule",
    "apply_threshold_rule",
    "evaluate_trivial_baselines",
    "fit_argmax_rule",
    "fit_threshold_rule",
    "parse_input_features",
    "trivial_baseline_gate",
]
