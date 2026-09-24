"""Pluggable brain-cluster expert learners behind one interface.

Backends (``make_learner(name, ...)``):

* ``nb``: the legacy multinomial Naive Bayes from
  ``flybrain_brain_cluster_training`` (``_fit_multinomial_nb``), unchanged. It
  reads only the text column (``TEXT_COLUMN``, i.e. a sample's ``input_text``).
* ``logreg``: sklearn ``LogisticRegression`` over standardized numeric
  features (median-imputed, with missing indicators) plus one-hot categorical
  features.
* ``hgb``: sklearn ``HistGradientBoostingClassifier`` with native categorical
  support. When ``groups`` are passed to ``fit`` the number of boosting
  iterations is chosen by early stopping on an internal *grouped* validation
  split, then the model is refit on all rows with that iteration count.
* ``mlp``: a small plain-torch MLP over the same numeric/one-hot encoding as
  ``logreg`` (JSON preprocessor + torch ``state_dict``; no pickle).

Every learner takes a ``pandas.DataFrame`` of features (numeric or categorical
columns, plus an optional ``TEXT_COLUMN``) and a sequence of string labels.
``predict_proba`` returns an ``(n, len(classes_))`` array aligned to the sorted
``classes_``. Seeds are explicit and every backend is deterministic on CPU.

Calibration: ``CalibratedLearner`` fits a probability calibrator
(``isotonic`` / ``sigmoid`` / ``temperature``) on grouped out-of-fold
predictions (``fit``) or on a held-out calibration split (``fit_prefit``).
Calibrators are stored as JSON, never pickled. ``expected_calibration_error``
and ``reliability_table`` measure the result.

Artifacts: ``save_learner`` writes a directory with the model file(s)
(``model.joblib`` for sklearn, ``model.pt`` state_dict for torch,
``model.json`` for NB), ``meta.json`` (backend, params, seed), ``schema.json``
(feature schema), ``labels.json``, ``fingerprint.json`` (training
fingerprint), ``metrics.json`` and a ``manifest.json`` that lists the sha256
of every other file.

PICKLE TRUST BOUNDARY. ``model.joblib`` is a pickle: unpickling it runs
arbitrary code. ``load_learner`` therefore (1) refuses any artifact directory
that is not inside ``trusted_root`` (the FlyBrain harness root, resolved,
symlinks rejected), (2) verifies ``manifest.json`` (optionally pinned by
``expected_manifest_sha256``) and the sha256 of every listed file, and fails
if any file is missing, modified or *unlisted*, and only then (3) unpickles.
The manifest is not a signature: anyone who can write inside the harness root
can rewrite it, so artifacts must only ever be loaded from the harness root,
which only the harness writes. Torch weights load with ``weights_only=True``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

LEARNER_ARTIFACT_SCHEMA_VERSION = "flybrain-learner-artifact/v1"
TEXT_COLUMN = "__input_text__"
MISSING_CATEGORY = "__missing__"
MANIFEST_FILE = "manifest.json"
ARTIFACT_FILES = ("meta.json", "schema.json", "labels.json", "fingerprint.json", "metrics.json")
CALIBRATION_METHODS = ("isotonic", "sigmoid", "temperature")
_FEATURE_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EPS = 1e-12


# --------------------------------------------------------------------------- utils


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays and tuples so ``json`` accepts them (NaN -> None)."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def align_proba(proba: np.ndarray, from_classes: Sequence[str], to_classes: Sequence[str]) -> np.ndarray:
    """Reorder/extend probability columns from ``from_classes`` to ``to_classes`` (missing -> 0)."""
    out = np.zeros((proba.shape[0], len(to_classes)), dtype=np.float64)
    index = {label: i for i, label in enumerate(to_classes)}
    for j, label in enumerate(from_classes):
        if label in index:
            out[:, index[label]] = proba[:, j]
    return out


def _labels_array(y: Sequence[Any]) -> np.ndarray:
    labels = np.asarray([str(v) for v in y], dtype=object)
    if any(not str(v).strip() for v in labels):
        raise ValueError("labels must be non-empty strings")
    return labels


def data_fingerprint(frame: pd.DataFrame, y: Sequence[Any]) -> str:
    """sha256 over column names, row-wise content hashes and labels (order sensitive)."""
    digest = hashlib.sha256()
    digest.update(_stable_json([str(c) for c in frame.columns]).encode("utf-8"))
    if len(frame):
        hashed = pd.util.hash_pandas_object(frame, index=False)
        digest.update(np.asarray(hashed, dtype=np.uint64).tobytes())
    digest.update(_stable_json([str(v) for v in y]).encode("utf-8"))
    return digest.hexdigest()


# --------------------------------------------------------------------------- feature schema


@dataclass(frozen=True)
class FeatureSchema:
    """The feature columns a learner was fitted on, split by kind (sorted, fixed at fit)."""

    numeric: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()
    uses_text: bool = False

    @property
    def columns(self) -> tuple[str, ...]:
        return self.numeric + self.categorical

    def as_dict(self) -> dict[str, Any]:
        return {"numeric": list(self.numeric), "categorical": list(self.categorical), "uses_text": self.uses_text}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureSchema":
        return cls(
            numeric=tuple(str(v) for v in payload.get("numeric", ())),
            categorical=tuple(str(v) for v in payload.get("categorical", ())),
            uses_text=bool(payload.get("uses_text", False)),
        )

    @classmethod
    def infer(cls, frame: pd.DataFrame, *, uses_text: bool = False) -> "FeatureSchema":
        numeric: list[str] = []
        categorical: list[str] = []
        for column in frame.columns:
            name = str(column)
            if name == TEXT_COLUMN:
                continue
            if not _FEATURE_KEY_RE.match(name):
                raise ValueError(f"invalid feature name {name!r}: must match {_FEATURE_KEY_RE.pattern}")
            dtype = frame[column].dtype
            if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
                numeric.append(name)
            else:
                categorical.append(name)
        return cls(numeric=tuple(sorted(numeric)), categorical=tuple(sorted(categorical)), uses_text=uses_text)

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Project ``frame`` onto the schema. Missing columns fail closed."""
        missing = [c for c in self.columns if c not in frame.columns]
        if missing:
            raise ValueError(f"frame is missing schema feature columns: {', '.join(missing[:10])}")
        out: dict[str, Any] = {}
        for name in self.numeric:
            out[name] = pd.to_numeric(frame[name], errors="coerce").astype("float64").to_numpy()
        for name in self.categorical:
            col = frame[name]
            values = col.astype(object).where(col.notna(), MISSING_CATEGORY).astype(str)
            out[name] = values.to_numpy(dtype=object)
        return pd.DataFrame(out, index=frame.index, columns=list(self.columns))


def features_frame(samples: Sequence[Mapping[str, Any]], *, include_text: bool = True) -> pd.DataFrame:
    """Build a learner input frame from sample dicts (``features`` + ``input_text``)."""
    rows = [dict(sample.get("features") or {}) for sample in samples]
    frame = pd.DataFrame.from_records(rows) if rows else pd.DataFrame()
    if len(frame) != len(rows):  # all feature dicts empty
        frame = pd.DataFrame(index=range(len(rows)))
    frame = frame.reindex(sorted(frame.columns), axis=1)
    if include_text:
        frame[TEXT_COLUMN] = [str(sample.get("input_text", "")) for sample in samples]
    return frame


# --------------------------------------------------------------------------- base learner


class Learner:
    """Common interface. Subclasses implement ``_fit``/``_predict_proba``/``_save_model``/``_load_model``."""

    backend: str = ""
    model_files: tuple[str, ...] = ()
    uses_text: bool = False
    default_params: Mapping[str, Any] = {}

    def __init__(self, *, seed: int = 0, **params: Any) -> None:
        unknown = sorted(set(params) - set(self.default_params))
        if unknown:
            raise ValueError(f"{self.backend}: unknown params {', '.join(unknown)}")
        self.seed = int(seed)
        self.params: dict[str, Any] = {**dict(self.default_params), **params}
        self.classes_: tuple[str, ...] = ()
        self.schema: FeatureSchema | None = None
        self.fit_info: dict[str, Any] = {}
        self.training_fingerprint: dict[str, Any] = {}

    # public API -----------------------------------------------------------
    def get_params(self) -> dict[str, Any]:
        return dict(self.params)

    def fit(self, X: pd.DataFrame, y: Sequence[Any], *, groups: Sequence[Any] | None = None) -> "Learner":
        labels = _labels_array(y)
        if len(labels) != len(X):
            raise ValueError("X and y lengths differ")
        if len(labels) == 0:
            raise ValueError("cannot fit on zero rows")
        if self.uses_text:
            if TEXT_COLUMN not in X.columns:
                raise ValueError(f"{self.backend} needs the text column {TEXT_COLUMN!r}")
            self.schema = FeatureSchema(uses_text=True)
        else:
            feature_frame = X.drop(columns=[TEXT_COLUMN], errors="ignore")
            self.schema = FeatureSchema.infer(feature_frame)
            if not self.schema.columns:
                raise ValueError(f"{self.backend} needs at least one feature column")
        self.classes_ = tuple(sorted(set(labels.tolist())))
        if len(self.classes_) < 2 and not self.uses_text:
            raise ValueError(f"{self.backend} needs at least two classes to fit")
        group_arr = None if groups is None else np.asarray([str(g) for g in groups], dtype=object)
        if group_arr is not None and len(group_arr) != len(labels):
            raise ValueError("groups length differs from y")
        self._fit(X, labels, group_arr)
        self.training_fingerprint = {
            "backend": self.backend,
            "params": _jsonable(self.params),
            "seed": self.seed,
            "n_rows": int(len(labels)),
            "classes": list(self.classes_),
            "schema": self.schema.as_dict(),
            "data_sha256": data_fingerprint(self._inputs(X), labels.tolist()),
            "library_versions": _library_versions(self.backend),
        }
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.classes_:
            raise ValueError("learner is not fitted")
        proba = np.asarray(self._predict_proba(X), dtype=np.float64)
        proba = np.clip(proba, 0.0, 1.0)
        totals = proba.sum(axis=1, keepdims=True)
        totals[totals <= 0] = 1.0
        return proba / totals

    def predict(self, X: pd.DataFrame) -> list[str]:
        proba = self.predict_proba(X)
        return [self.classes_[i] for i in np.argmax(proba, axis=1)]

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "params": _jsonable(self.params),
            "seed": self.seed,
            "classes": list(self.classes_),
            "schema": None if self.schema is None else self.schema.as_dict(),
            "fit_info": _jsonable(self.fit_info),
            "training_fingerprint": _jsonable(self.training_fingerprint),
        }

    def save(self, directory: str | Path, *, metrics: Mapping[str, Any] | None = None,
             overwrite: bool = False) -> dict[str, Any]:
        return save_learner(self, directory, metrics=metrics, overwrite=overwrite)

    @staticmethod
    def load(directory: str | Path, *, trusted_root: str | Path,
             expected_manifest_sha256: str | None = None) -> "Learner":
        return load_learner(directory, trusted_root=trusted_root, expected_manifest_sha256=expected_manifest_sha256)

    # helpers ------------------------------------------------------------------
    def _inputs(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.schema is not None
        if self.uses_text:
            return X[[TEXT_COLUMN]].astype(str)
        return self.schema.prepare(X)

    # subclass hooks -------------------------------------------------------------
    def _fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None) -> None:  # pragma: no cover
        raise NotImplementedError

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _save_model(self, directory: Path) -> None:  # pragma: no cover
        raise NotImplementedError

    def _load_model(self, directory: Path) -> None:  # pragma: no cover
        raise NotImplementedError


def _library_versions(backend: str) -> dict[str, str]:
    import sklearn

    versions = {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}
    if backend == "mlp":
        import torch

        versions["torch"] = torch.__version__
    return versions


def _inner_group_split(groups: np.ndarray, y: np.ndarray, *, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Deterministic grouped holdout of ~``fraction`` of rows; None when not possible.

    Rows of the inner validation split whose label is absent from the inner
    train split are dropped (a learner cannot score an unseen class).
    """
    unique = sorted(set(groups.tolist()))
    if len(unique) < 4:
        return None
    order = sorted(unique, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode("utf-8")).hexdigest())
    target = max(1, int(round(len(groups) * fraction)))
    counts = pd.Series(groups).value_counts().to_dict()
    chosen: set[str] = set()
    taken = 0
    for g in order:
        if taken >= target:
            break
        chosen.add(g)
        taken += counts[g]
    is_val = np.asarray([g in chosen for g in groups.tolist()])
    train_idx = np.flatnonzero(~is_val)
    train_labels = set(y[train_idx].tolist())
    val_idx = np.asarray([i for i in np.flatnonzero(is_val) if y[i] in train_labels], dtype=np.int64)
    if len(train_idx) == 0 or len(val_idx) == 0 or len(train_labels) < 2:
        return None
    return train_idx, val_idx


# --------------------------------------------------------------------------- nb (legacy)


class NaiveBayesLearner(Learner):
    """Legacy multinomial NB over ``input_text`` tokens (``_fit_multinomial_nb`` unchanged)."""

    backend = "nb"
    model_files = ("model.json",)
    uses_text = True
    default_params = {"alpha": 1.0}

    def _fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None) -> None:
        from flybrain_brain_cluster_training import _fit_multinomial_nb

        rows = list(zip(X[TEXT_COLUMN].astype(str).tolist(), y.tolist()))
        self.model = _fit_multinomial_nb(rows, alpha=float(self.params["alpha"]))
        self.fit_info = {"vocab_size": len(self.model["vocab"])}

    def _scores(self, text: str) -> dict[str, float]:
        # Same arithmetic as flybrain_brain_cluster_training._predict_multinomial_nb,
        # returning every label's log score instead of only the argmax.
        from flybrain_brain_cluster_training import _tokenize

        token_freq: dict[str, int] = {}
        for token in _tokenize(text):
            token_freq[token] = token_freq.get(token, 0) + 1
        scores: dict[str, float] = {}
        for label in self.model["labels"]:
            score = float(self.model["log_prior"][label])
            weights = self.model["log_probs"][label]
            default = float(self.model["unknown_token_log_prob"][label])
            for token, freq in token_freq.items():
                score += freq * float(weights.get(token, default))
            scores[label] = score
        return scores

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        labels = list(self.model["labels"])
        out = np.zeros((len(X), len(self.classes_)), dtype=np.float64)
        col = {label: self.classes_.index(label) for label in labels}
        for i, text in enumerate(X[TEXT_COLUMN].astype(str).tolist()):
            scores = self._scores(text)
            top = max(scores.values())
            exp = {label: math.exp(s - top) for label, s in scores.items()}
            total = sum(exp.values()) or 1.0
            for label, value in exp.items():
                out[i, col[label]] = value / total
        return out

    def _save_model(self, directory: Path) -> None:
        payload = dict(self.model)
        payload["labels"] = list(payload["labels"])
        payload["vocab"] = list(payload["vocab"])
        _write_json(directory / "model.json", payload)

    def _load_model(self, directory: Path) -> None:
        payload = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        payload["labels"] = tuple(payload["labels"])
        payload["vocab"] = tuple(payload["vocab"])
        self.model = payload


# --------------------------------------------------------------------------- sklearn backends


class _SklearnLearner(Learner):
    model_files = ("model.joblib",)

    def _save_model(self, directory: Path) -> None:
        import joblib

        joblib.dump(self.model, directory / "model.joblib", compress=3)

    def _load_model(self, directory: Path) -> None:
        import joblib

        self.model = joblib.load(directory / "model.joblib")  # manifest verified by load_learner first

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.model.predict_proba(self._design(X))
        return align_proba(np.asarray(proba), [str(c) for c in self.model.classes_], self.classes_)

    def _design(self, X: pd.DataFrame) -> Any:
        return self.schema.prepare(X)  # type: ignore[union-attr]


class LogRegLearner(_SklearnLearner):
    backend = "logreg"
    default_params = {"C": 1.0, "class_weight": None, "max_iter": 2000, "min_category_count": 5}

    def _fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None) -> None:
        import warnings

        from sklearn.compose import ColumnTransformer
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        assert self.schema is not None
        transformers = []
        if self.schema.numeric:
            transformers.append((
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                          ("scale", StandardScaler())]),
                list(self.schema.numeric),
            ))
        if self.schema.categorical:
            transformers.append((
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=int(self.params["min_category_count"]),
                              sparse_output=True),
                list(self.schema.categorical),
            ))
        classifier = LogisticRegression(
            C=float(self.params["C"]),
            class_weight=self.params["class_weight"],
            max_iter=int(self.params["max_iter"]),
            random_state=self.seed,
        )
        self.model = Pipeline([("prep", ColumnTransformer(transformers, sparse_threshold=0.3)), ("clf", classifier)])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.model.fit(self.schema.prepare(X), y)
        self.fit_info = {
            "n_iter": int(np.max(classifier.n_iter_)),
            "converged": not any(issubclass(w.category, ConvergenceWarning) for w in caught),
        }


class HgbLearner(_SklearnLearner):
    backend = "hgb"
    default_params = {
        "learning_rate": 0.05,
        "max_iter": 500,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "class_weight": None,
        "max_bins": 255,
        "min_category_count": 5,
        "inner_val_fraction": 0.15,
        "n_iter_no_change": 20,
    }

    def _build(self, max_iter: int, early_stopping: bool) -> Any:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OrdinalEncoder

        assert self.schema is not None
        n_num = len(self.schema.numeric)
        n_cat = len(self.schema.categorical)
        transformers = []
        if n_num:
            transformers.append(("num", "passthrough", list(self.schema.numeric)))
        if n_cat:
            transformers.append((
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan,
                               min_frequency=int(self.params["min_category_count"]),
                               max_categories=int(self.params["max_bins"]) - 1),
                list(self.schema.categorical),
            ))
        classifier = HistGradientBoostingClassifier(
            learning_rate=float(self.params["learning_rate"]),
            max_iter=int(max_iter),
            max_leaf_nodes=int(self.params["max_leaf_nodes"]),
            min_samples_leaf=int(self.params["min_samples_leaf"]),
            l2_regularization=float(self.params["l2_regularization"]),
            class_weight=self.params["class_weight"],
            max_bins=int(self.params["max_bins"]),
            categorical_features=[False] * n_num + [True] * n_cat,
            early_stopping=early_stopping,
            n_iter_no_change=int(self.params["n_iter_no_change"]),
            validation_fraction=float(self.params["inner_val_fraction"]),
            random_state=self.seed,
        )
        return Pipeline([("prep", ColumnTransformer(transformers)), ("clf", classifier)])

    def _fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None) -> None:
        assert self.schema is not None
        design = self.schema.prepare(X)
        max_iter = int(self.params["max_iter"])
        split = None if groups is None else _inner_group_split(
            groups, y, fraction=float(self.params["inner_val_fraction"]), seed=self.seed)
        if split is not None:
            train_idx, val_idx = split
            probe = self._build(max_iter, early_stopping=True)
            prep = probe.named_steps["prep"].fit(design.iloc[train_idx])
            clf = probe.named_steps["clf"]
            clf.fit(prep.transform(design.iloc[train_idx]), y[train_idx],
                    X_val=prep.transform(design.iloc[val_idx]), y_val=y[val_idx])
            stopped_early = int(clf.n_iter_) < max_iter
            best_iter = max(1, int(clf.n_iter_) - int(self.params["n_iter_no_change"])) if stopped_early else max_iter
            self.model = self._build(best_iter, early_stopping=False)
            self.model.fit(design, y)
            self.fit_info = {"early_stopping": "grouped_inner_val", "probe_n_iter": int(clf.n_iter_),
                             "n_iter": best_iter, "inner_train_rows": int(len(train_idx)),
                             "inner_val_rows": int(len(val_idx))}
        else:
            self.model = self._build(max_iter, early_stopping=len(y) >= 200)
            self.model.fit(design, y)
            self.fit_info = {"early_stopping": "random_inner_val" if len(y) >= 200 else "off",
                             "n_iter": int(self.model.named_steps["clf"].n_iter_)}


# --------------------------------------------------------------------------- torch mlp


class _DenseEncoder:
    """JSON-serializable numeric standardizer + categorical one-hot (top categories)."""

    def __init__(self, schema: FeatureSchema, min_category_count: int = 5, max_categories: int = 64) -> None:
        self.schema = schema
        self.min_category_count = int(min_category_count)
        self.max_categories = int(max_categories)
        self.state: dict[str, Any] = {}

    def fit(self, design: pd.DataFrame) -> "_DenseEncoder":
        numeric = {}
        for name in self.schema.numeric:
            values = design[name].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if len(finite) else 0.0
            filled = np.where(np.isfinite(values), values, median)
            std = float(filled.std()) or 1.0
            numeric[name] = {"median": median, "mean": float(filled.mean()), "std": std}
        categorical = {}
        for name in self.schema.categorical:
            counts = design[name].value_counts()
            kept = sorted(v for v, c in counts.items() if c >= self.min_category_count)
            kept = sorted(kept, key=lambda v: (-int(counts[v]), v))[: self.max_categories]
            categorical[name] = sorted(kept)
        self.state = {"numeric": numeric, "categorical": categorical}
        return self

    def transform(self, design: pd.DataFrame) -> np.ndarray:
        blocks = []
        for name in self.schema.numeric:
            spec = self.state["numeric"][name]
            values = design[name].to_numpy(dtype=np.float64)
            missing = ~np.isfinite(values)
            filled = np.where(missing, spec["median"], values)
            blocks.append(((filled - spec["mean"]) / spec["std"])[:, None])
            blocks.append(missing.astype(np.float64)[:, None])
        for name in self.schema.categorical:
            vocab = self.state["categorical"][name]
            index = {v: i for i, v in enumerate(vocab)}
            onehot = np.zeros((len(design), len(vocab) + 1), dtype=np.float64)
            codes = np.asarray([index.get(v, len(vocab)) for v in design[name].tolist()], dtype=np.int64)
            onehot[np.arange(len(design)), codes] = 1.0
            blocks.append(onehot)
        return np.hstack(blocks).astype(np.float32) if blocks else np.zeros((len(design), 0), dtype=np.float32)


class MlpLearner(Learner):
    backend = "mlp"
    model_files = ("model.pt", "encoder.json")
    default_params = {
        "hidden": 128,
        "layers": 2,
        "dropout": 0.1,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 200,
        "batch_size": 512,
        "patience": 15,
        "class_weight": None,
        "device": "cpu",
        "min_category_count": 5,
        "max_categories": 64,
        "inner_val_fraction": 0.15,
    }

    def _net(self, n_in: int, n_out: int) -> Any:
        import torch.nn as nn

        layers: list[Any] = []
        width = n_in
        for _ in range(int(self.params["layers"])):
            layers += [nn.Linear(width, int(self.params["hidden"])), nn.ReLU(), nn.Dropout(float(self.params["dropout"]))]
            width = int(self.params["hidden"])
        layers.append(nn.Linear(width, n_out))
        return nn.Sequential(*layers)

    def _fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None) -> None:
        import torch

        assert self.schema is not None
        torch.manual_seed(self.seed)
        design = self.schema.prepare(X)
        self.encoder = _DenseEncoder(self.schema, int(self.params["min_category_count"]),
                                     int(self.params["max_categories"])).fit(design)
        features = self.encoder.transform(design)
        targets = np.asarray([self.classes_.index(v) for v in y.tolist()], dtype=np.int64)
        device = torch.device(str(self.params["device"]))
        split = None if groups is None else _inner_group_split(
            groups, y, fraction=float(self.params["inner_val_fraction"]), seed=self.seed)
        train_idx, val_idx = split if split is not None else (np.arange(len(y)), np.zeros(0, dtype=np.int64))
        weight = None
        if self.params["class_weight"] == "balanced":
            counts = np.bincount(targets[train_idx], minlength=len(self.classes_)).astype(np.float64)
            weight = torch.tensor(len(train_idx) / (len(self.classes_) * np.maximum(counts, 1.0)),
                                  dtype=torch.float32, device=device)
        best_epochs = self._train(features, targets, train_idx, val_idx, int(self.params["epochs"]), weight, device)
        # Refit on every row for the selected epoch count (deterministic reseed).
        torch.manual_seed(self.seed)
        self._train(features, targets, np.arange(len(y)), np.zeros(0, dtype=np.int64), best_epochs, weight, device)
        self.fit_info = {"epochs": best_epochs,
                         "early_stopping": "grouped_inner_val" if len(val_idx) else "off",
                         "n_inputs": int(features.shape[1])}

    def _train(self, features, targets, train_idx, val_idx, epochs, weight, device) -> int:
        import torch
        import torch.nn.functional as F

        generator = torch.Generator().manual_seed(self.seed)
        self.net = self._net(features.shape[1], len(self.classes_)).to(device)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=float(self.params["lr"]),
                                      weight_decay=float(self.params["weight_decay"]))
        xt = torch.from_numpy(features[train_idx]).to(device)
        yt = torch.from_numpy(targets[train_idx]).to(device)
        xv = torch.from_numpy(features[val_idx]).to(device) if len(val_idx) else None
        yv = torch.from_numpy(targets[val_idx]).to(device) if len(val_idx) else None
        batch = int(self.params["batch_size"])
        best_loss, best_epoch, best_state, stale = math.inf, epochs, None, 0
        for epoch in range(1, epochs + 1):
            self.net.train()
            perm = torch.randperm(len(train_idx), generator=generator).to(device)
            for start in range(0, len(train_idx), batch):
                idx = perm[start:start + batch]
                optimizer.zero_grad()
                loss = F.cross_entropy(self.net(xt[idx]), yt[idx], weight=weight)
                loss.backward()
                optimizer.step()
            if xv is not None:
                self.net.eval()
                with torch.no_grad():
                    val_loss = float(F.cross_entropy(self.net(xv), yv))
                if val_loss < best_loss - 1e-6:
                    best_loss, best_epoch, stale = val_loss, epoch, 0
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    stale += 1
                    if stale >= int(self.params["patience"]):
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return best_epoch

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        features = torch.from_numpy(self.encoder.transform(self.schema.prepare(X)))  # type: ignore[union-attr]
        device = next(self.net.parameters()).device
        self.net.eval()
        with torch.no_grad():
            logits = self.net(features.to(device))
            return torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64)

    def _save_model(self, directory: Path) -> None:
        import torch

        state = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        torch.save(state, directory / "model.pt")
        _write_json(directory / "encoder.json", _jsonable(self.encoder.state))

    def _load_model(self, directory: Path) -> None:
        import torch

        assert self.schema is not None
        self.encoder = _DenseEncoder(self.schema, int(self.params["min_category_count"]), int(self.params["max_categories"]))
        self.encoder.state = json.loads((directory / "encoder.json").read_text(encoding="utf-8"))
        n_in = 2 * len(self.schema.numeric) + sum(len(v) + 1 for v in self.encoder.state["categorical"].values())
        self.net = self._net(n_in, len(self.classes_))
        self.net.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
        self.net.eval()


# --------------------------------------------------------------------------- registry

_BACKENDS: dict[str, type[Learner]] = {}


def register_backend(name: str, cls: type[Learner], *, replace: bool = False) -> None:
    """Register a learner backend (e.g. a graph model). ``cls.backend`` must equal ``name``."""
    if not _FEATURE_KEY_RE.match(name):
        raise ValueError(f"invalid backend name {name!r}")
    if getattr(cls, "backend", None) != name:
        raise ValueError(f"backend class {cls.__name__} declares backend={cls.backend!r}, expected {name!r}")
    if name in _BACKENDS and not replace:
        raise ValueError(f"backend already registered: {name}")
    _BACKENDS[name] = cls


def available_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def make_learner(backend: str, *, seed: int = 0, **params: Any) -> Learner:
    if backend == "calibrated":
        raise ValueError("build a CalibratedLearner explicitly")
    if backend not in _BACKENDS:
        raise ValueError(f"unknown learner backend {backend!r}; available: {', '.join(available_backends())}")
    return _BACKENDS[backend](seed=seed, **params)


for _name, _cls in (("nb", NaiveBayesLearner), ("logreg", LogRegLearner), ("hgb", HgbLearner), ("mlp", MlpLearner)):
    register_backend(_name, _cls)


# --------------------------------------------------------------------------- calibration metrics


def expected_calibration_error(proba: np.ndarray, y: Sequence[Any], classes: Sequence[str], *, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins."""
    table = reliability_table(proba, y, classes, n_bins=n_bins)
    n = sum(row["count"] for row in table)
    if n == 0:
        return 0.0
    return float(sum(row["count"] * abs(row["accuracy"] - row["mean_confidence"]) for row in table if row["count"]) / n)


def reliability_table(proba: np.ndarray, y: Sequence[Any], classes: Sequence[str], *, n_bins: int = 15) -> list[dict[str, Any]]:
    proba = np.asarray(proba, dtype=np.float64)
    labels = [str(v) for v in y]
    if proba.shape[0] != len(labels):
        raise ValueError("proba rows and labels differ")
    confidence = proba.max(axis=1) if len(labels) else np.zeros(0)
    predicted = [classes[i] for i in proba.argmax(axis=1)] if len(labels) else []
    correct = np.asarray([p == a for p, a in zip(predicted, labels)], dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    bins = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, int(n_bins) - 1)
    rows = []
    for b in range(int(n_bins)):
        mask = bins == b
        count = int(mask.sum())
        rows.append({
            "bin_lo": float(edges[b]),
            "bin_hi": float(edges[b + 1]),
            "count": count,
            "mean_confidence": float(confidence[mask].mean()) if count else 0.0,
            "accuracy": float(correct[mask].mean()) if count else 0.0,
        })
    return rows


def multiclass_log_loss(proba: np.ndarray, y: Sequence[Any], classes: Sequence[str]) -> float:
    index = {c: i for i, c in enumerate(classes)}
    if len(y) == 0:
        return 0.0
    picked = np.asarray([proba[r, index[str(v)]] if str(v) in index else 0.0 for r, v in enumerate(y)])
    return float(-np.mean(np.log(np.clip(picked, _EPS, 1.0))))


def brier_score(proba: np.ndarray, y: Sequence[Any], classes: Sequence[str]) -> float:
    index = {c: i for i, c in enumerate(classes)}
    onehot = np.zeros_like(proba, dtype=np.float64)
    for r, v in enumerate(y):
        if str(v) in index:
            onehot[r, index[str(v)]] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1))) if len(y) else 0.0


# --------------------------------------------------------------------------- calibrators (JSON state)


def fit_calibrator(proba: np.ndarray, y: Sequence[Any], classes: Sequence[str], method: str) -> dict[str, Any]:
    """Fit a calibrator on (probabilities, true labels); returns a JSON-able state."""
    if method not in CALIBRATION_METHODS:
        raise ValueError(f"calibration method must be one of {CALIBRATION_METHODS}")
    proba = np.asarray(proba, dtype=np.float64)
    labels = [str(v) for v in y]
    if len(labels) < 2:
        raise ValueError("calibration needs at least two rows")
    index = {c: i for i, c in enumerate(classes)}
    targets = np.asarray([index.get(v, -1) for v in labels])
    if method == "temperature":
        from scipy.optimize import minimize_scalar

        logp = np.log(np.clip(proba, _EPS, 1.0))
        valid = targets >= 0

        def nll(log_t: float) -> float:
            z = logp[valid] / math.exp(log_t)
            z = z - z.max(axis=1, keepdims=True)
            logsm = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            return float(-logsm[np.arange(valid.sum()), targets[valid]].mean())

        result = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
        return {"method": method, "classes": list(classes), "temperature": float(math.exp(result.x))}
    per_class = []
    for k in range(len(classes)):
        target = (targets == k).astype(np.float64)
        p = proba[:, k]
        if target.min() == target.max():
            per_class.append({"constant": float(target[0])})
            continue
        if method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, target)
            per_class.append({"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()})
        else:
            from sklearn.linear_model import LogisticRegression

            logit = np.log(np.clip(p, _EPS, 1 - 1e-12) / np.clip(1 - p, _EPS, 1.0))[:, None]
            platt = LogisticRegression(C=1e4, max_iter=1000).fit(logit, target)
            per_class.append({"a": float(platt.coef_[0, 0]), "b": float(platt.intercept_[0])})
    return {"method": method, "classes": list(classes), "per_class": per_class}


def apply_calibrator(state: Mapping[str, Any], proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=np.float64)
    method = state["method"]
    if method == "temperature":
        z = np.log(np.clip(proba, _EPS, 1.0)) / float(state["temperature"])
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)
    out = np.zeros_like(proba)
    for k, spec in enumerate(state["per_class"]):
        p = proba[:, k]
        if "constant" in spec:
            out[:, k] = spec["constant"]
        elif method == "isotonic":
            out[:, k] = np.interp(p, np.asarray(spec["x"]), np.asarray(spec["y"]))
        else:
            logit = np.log(np.clip(p, _EPS, 1 - 1e-12) / np.clip(1 - p, _EPS, 1.0))
            out[:, k] = 1.0 / (1.0 + np.exp(-(spec["a"] * logit + spec["b"])))
    totals = out.sum(axis=1, keepdims=True)
    fallback = totals[:, 0] <= _EPS
    out[fallback] = proba[fallback]
    totals[fallback] = 1.0
    return out / np.where(totals <= _EPS, 1.0, totals)


def grouped_folds(groups: Sequence[Any], *, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic GroupKFold-style folds: groups hashed with ``seed`` then dealt round-robin by size."""
    groups_arr = np.asarray([str(g) for g in groups], dtype=object)
    unique = sorted(set(groups_arr.tolist()))
    k = min(int(n_splits), len(unique))
    if k < 2:
        raise ValueError(f"grouped CV needs at least 2 distinct groups (got {len(unique)})")
    counts = pd.Series(groups_arr).value_counts().to_dict()
    order = sorted(unique, key=lambda g: (-counts[g], hashlib.sha256(f"{seed}:{g}".encode("utf-8")).hexdigest()))
    loads = [0] * k
    fold_of: dict[str, int] = {}
    for g in order:  # greedy balance: largest groups first into the lightest fold
        f = min(range(k), key=lambda i: (loads[i], i))
        fold_of[g] = f
        loads[f] += counts[g]
    fold_ids = np.asarray([fold_of[g] for g in groups_arr.tolist()])
    return [(np.flatnonzero(fold_ids != f), np.flatnonzero(fold_ids == f)) for f in range(k)]


def out_of_fold_proba(backend: str, params: Mapping[str, Any], X: pd.DataFrame, y: Sequence[Any],
                      groups: Sequence[Any], *, classes: Sequence[str], n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Grouped out-of-fold probabilities aligned to ``classes``."""
    labels = _labels_array(y)
    groups_arr = np.asarray([str(g) for g in groups], dtype=object)
    oof = np.zeros((len(labels), len(classes)), dtype=np.float64)
    for train_idx, test_idx in grouped_folds(groups_arr, n_splits=n_splits, seed=seed):
        fold = make_learner(backend, seed=seed, **dict(params))
        fold.fit(X.iloc[train_idx], labels[train_idx], groups=groups_arr[train_idx])
        oof[test_idx] = align_proba(fold.predict_proba(X.iloc[test_idx]), fold.classes_, classes)
    return oof


class CalibratedLearner(Learner):
    """A fitted base learner plus a JSON calibrator on its probabilities."""

    backend = "calibrated"

    def __init__(self, base: Learner, *, method: str = "isotonic", cv: int = 5) -> None:
        if method not in CALIBRATION_METHODS:
            raise ValueError(f"calibration method must be one of {CALIBRATION_METHODS}")
        super().__init__(seed=base.seed)
        self.base = base
        self.params = {"method": method, "cv": int(cv), "base_backend": base.backend}
        self.calibrator: dict[str, Any] = {}

    @property
    def model_files(self) -> tuple[str, ...]:  # type: ignore[override]
        return ("calibrator.json",) + tuple(f"base/{name}" for name in self.base.model_files)

    def fit(self, X: pd.DataFrame, y: Sequence[Any], *, groups: Sequence[Any] | None = None) -> "CalibratedLearner":
        """Grouped-CV calibration: OOF probs from ``cv`` grouped folds fit the calibrator; base refit on all rows."""
        if groups is None:
            raise ValueError("CalibratedLearner.fit needs groups (grouped CV); use fit_prefit for a held-out split")
        labels = _labels_array(y)
        classes = tuple(sorted(set(labels.tolist())))
        oof = out_of_fold_proba(self.base.backend, self.base.get_params(), X, labels, groups,
                                classes=classes, n_splits=int(self.params["cv"]), seed=self.seed)
        self.base.fit(X, labels, groups=groups)
        self._finish(fit_calibrator(oof, labels, classes, self.params["method"]), source="grouped_cv_oof",
                     n=len(labels), raw=oof, labels=labels)
        return self

    @classmethod
    def from_oof(cls, fitted_base: Learner, oof_proba: np.ndarray, y: Sequence[Any], *, method: str,
                 classes: Sequence[str], cv: int) -> "CalibratedLearner":
        """Wrap a base already fitted on all train rows, calibrating on grouped OOF probs computed elsewhere.

        ``oof_proba`` must come from grouped CV over the same train rows (e.g.
        the tuning pass), aligned to ``classes``.
        """
        if not fitted_base.classes_:
            raise ValueError("base learner must be fitted")
        learner = cls(fitted_base, method=method, cv=cv)
        labels = _labels_array(y)
        learner._finish(fit_calibrator(oof_proba, labels, tuple(classes), method), source="grouped_cv_oof",
                        n=len(labels), raw=np.asarray(oof_proba), labels=labels)
        return learner

    def fit_prefit(self, X_cal: pd.DataFrame, y_cal: Sequence[Any]) -> "CalibratedLearner":
        """Calibrate an already-fitted base on a held-out calibration split (e.g. the grouped val split)."""
        if not self.base.classes_:
            raise ValueError("base learner must be fitted before fit_prefit")
        labels = _labels_array(y_cal)
        raw = self.base.predict_proba(X_cal)
        self._finish(fit_calibrator(raw, labels, self.base.classes_, self.params["method"]), source="prefit_heldout",
                     n=len(labels), raw=raw, labels=labels)
        return self

    def _finish(self, state: dict[str, Any], *, source: str, n: int, raw: np.ndarray, labels: np.ndarray) -> None:
        self.calibrator = state
        self.classes_ = self.base.classes_
        self.schema = self.base.schema
        self.uses_text = self.base.uses_text
        calibrated = apply_calibrator(state, align_proba(raw, state["classes"], self.classes_))
        self.fit_info = {
            "calibration_source": source,
            "calibration_rows": int(n),
            "ece_before": expected_calibration_error(align_proba(raw, state["classes"], self.classes_), labels, self.classes_),
            "ece_after_in_sample": expected_calibration_error(calibrated, labels, self.classes_),
            "base_fit_info": _jsonable(self.base.fit_info),
        }
        self.training_fingerprint = {**self.base.training_fingerprint, "calibration": dict(self.params)}

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = align_proba(self.base.predict_proba(X), self.base.classes_, self.calibrator["classes"])
        return align_proba(apply_calibrator(self.calibrator, raw), self.calibrator["classes"], self.classes_)

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out["base"] = self.base.describe()
        out["calibrator_method"] = self.params["method"]
        return out

    def _save_model(self, directory: Path) -> None:
        _write_json(directory / "calibrator.json", _jsonable(self.calibrator))
        (directory / "base").mkdir()
        _write_learner_files(self.base, directory / "base")

    def _load_model(self, directory: Path) -> None:
        self.calibrator = json.loads((directory / "calibrator.json").read_text(encoding="utf-8"))
        self.base = _read_learner_files(directory / "base")


# --------------------------------------------------------------------------- artifacts


def _write_learner_files(learner: Learner, directory: Path) -> None:
    if not learner.classes_:
        raise ValueError("cannot save an unfitted learner")
    meta = {
        "backend": learner.backend,
        "params": _jsonable(learner.params),
        "seed": learner.seed,
        "fit_info": _jsonable(learner.fit_info),
        "model_files": list(learner.model_files),
    }
    _write_json(directory / "meta.json", meta)
    _write_json(directory / "schema.json", learner.schema.as_dict() if learner.schema else {})
    _write_json(directory / "labels.json", list(learner.classes_))
    _write_json(directory / "fingerprint.json", _jsonable(learner.training_fingerprint))
    learner._save_model(directory)


def _read_learner_files(directory: Path) -> Learner:
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    backend = str(meta["backend"])
    if backend == "calibrated":
        learner: Learner = CalibratedLearner.__new__(CalibratedLearner)
        Learner.__init__(learner, seed=int(meta["seed"]))
        learner.params = dict(meta["params"])
    else:
        if backend not in _BACKENDS:
            raise ValueError(f"artifact backend {backend!r} is not registered")
        learner = _BACKENDS[backend](seed=int(meta["seed"]), **dict(meta["params"]))
    learner.classes_ = tuple(json.loads((directory / "labels.json").read_text(encoding="utf-8")))
    schema_payload = json.loads((directory / "schema.json").read_text(encoding="utf-8"))
    learner.schema = FeatureSchema.from_dict(schema_payload) if schema_payload else None
    learner.fit_info = dict(meta.get("fit_info") or {})
    learner.training_fingerprint = json.loads((directory / "fingerprint.json").read_text(encoding="utf-8"))
    learner._load_model(directory)
    if isinstance(learner, CalibratedLearner):
        learner.uses_text = learner.base.uses_text
    return learner


def _listed_files(directory: Path) -> list[str]:
    out = []
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        for name in sorted(files):
            path = Path(root) / name
            if path.is_symlink():
                raise ValueError(f"artifact contains a symlink: {path}")
            out.append(path.relative_to(directory).as_posix())
    return out


def save_learner(learner: Learner, directory: str | Path, *, metrics: Mapping[str, Any] | None = None,
                 overwrite: bool = False) -> dict[str, Any]:
    """Write the artifact directory and its manifest; returns the manifest (with ``manifest_sha256``)."""
    target = Path(directory)
    if target.exists():
        if not overwrite:
            raise ValueError(f"artifact directory already exists: {target}")
        if any(target.iterdir()) and not (target / MANIFEST_FILE).is_file():
            raise ValueError(f"refusing to overwrite a directory that is not a learner artifact: {target}")
        import shutil

        shutil.rmtree(target)
    target.mkdir(parents=True)
    _write_learner_files(learner, target)
    _write_json(target / "metrics.json", _jsonable(dict(metrics or {})))
    files = {name: sha256_file(target / name) for name in _listed_files(target)}
    manifest = {
        "schema_version": LEARNER_ARTIFACT_SCHEMA_VERSION,
        "backend": learner.backend,
        "classes": list(learner.classes_),
        "training_fingerprint_sha256": hashlib.sha256(
            _stable_json(_jsonable(learner.training_fingerprint)).encode("utf-8")).hexdigest(),
        "files": files,
    }
    _write_json(target / MANIFEST_FILE, manifest)
    return {**manifest, "manifest_sha256": sha256_file(target / MANIFEST_FILE), "path": str(target)}


def _resolve_inside(directory: Path, trusted_root: Path) -> Path:
    root = Path(trusted_root).resolve(strict=True)
    for probe in [Path(directory), *Path(directory).parents]:
        if probe.is_symlink():
            raise ValueError(f"artifact path traverses a symlink: {probe}")
    resolved = Path(directory).resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"artifact {resolved} is outside the trusted root {root}; refusing to load (pickle boundary)")
    return resolved


def verify_artifact(directory: str | Path, *, trusted_root: str | Path,
                    expected_manifest_sha256: str | None = None) -> dict[str, Any]:
    """Fail-closed integrity check of an artifact directory. Reads no pickle."""
    resolved = _resolve_inside(Path(directory), Path(trusted_root))
    manifest_path = resolved / MANIFEST_FILE
    if not manifest_path.is_file():
        raise ValueError(f"artifact has no {MANIFEST_FILE}: {resolved}")
    manifest_sha = sha256_file(manifest_path)
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise ValueError("manifest sha256 does not match the pinned value")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != LEARNER_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported artifact schema_version: {manifest.get('schema_version')!r}")
    listed = dict(manifest.get("files") or {})
    present = [name for name in _listed_files(resolved) if name != MANIFEST_FILE]
    unlisted = sorted(set(present) - set(listed))
    if unlisted:
        raise ValueError(f"artifact contains files not in the manifest: {', '.join(unlisted[:10])}")
    for name, digest in sorted(listed.items()):
        path = resolved / name
        if ".." in Path(name).parts or Path(name).is_absolute():
            raise ValueError(f"unsafe manifest path: {name}")
        if not path.is_file():
            raise ValueError(f"artifact file missing: {name}")
        if sha256_file(path) != digest:
            raise ValueError(f"artifact file hash mismatch: {name}")
    return {**manifest, "manifest_sha256": manifest_sha, "path": str(resolved)}


def load_learner(directory: str | Path, *, trusted_root: str | Path,
                 expected_manifest_sha256: str | None = None) -> Learner:
    """Verify the manifest and every file hash, THEN deserialize (see module docstring)."""
    info = verify_artifact(directory, trusted_root=trusted_root, expected_manifest_sha256=expected_manifest_sha256)
    learner = _read_learner_files(Path(info["path"]))
    if list(learner.classes_) != list(info["classes"]):
        raise ValueError("artifact labels.json disagrees with its manifest")
    return learner


__all__ = [
    "CALIBRATION_METHODS",
    "CalibratedLearner",
    "FeatureSchema",
    "HgbLearner",
    "LEARNER_ARTIFACT_SCHEMA_VERSION",
    "Learner",
    "LogRegLearner",
    "MISSING_CATEGORY",
    "MlpLearner",
    "NaiveBayesLearner",
    "TEXT_COLUMN",
    "align_proba",
    "apply_calibrator",
    "available_backends",
    "brier_score",
    "data_fingerprint",
    "expected_calibration_error",
    "features_frame",
    "fit_calibrator",
    "grouped_folds",
    "load_learner",
    "make_learner",
    "multiclass_log_loss",
    "out_of_fold_proba",
    "register_backend",
    "reliability_table",
    "save_learner",
    "sha256_file",
    "verify_artifact",
]
