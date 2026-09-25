"""FlyBrain realism critic — scores a sampled connectome against the HGB io_class gate.

Usage
-----
    critic = RealismCritic(model_path)
    connectome = sample_connectome(genome, seed=42)
    sbm = genome  # or a genome dict with compartment_bias
    score = critic.score(connectome, sbm)   # float in [0, 1]
    passed = critic.gate(connectome, sbm)   # True if score >= 0.90

The score is the mean probability that each neuron is assigned to its declared
type by the HGB classifier.  A score of 1.0 means every neuron looks exactly
like its type; 0.0 means every neuron looks like the wrong type.

The realism gate passes at REALISM_GATE = 0.90 (below the 0.942 real l1em
baseline, above chance for 6 classes which is 0.167).
"""
from __future__ import annotations

import hashlib
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from flybrain_wiring_features import EdgeSource, WiringFeatureParams, compute_wiring_features

REALISM_GATE: float = 0.90
_CRITIC_ASE_DIM: int = 8
_DEFAULT_COMPARTMENT_BIAS: dict[str, float] = {
    "axon_out_frac": 0.75,
    "dendrite_in_frac": 0.68,
}


class RealismCritic:
    """Wraps a trained HGB io_class Pipeline and scores sampled connectomes."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._model = joblib.load(self.model_path)
        self._cell_types: tuple[str, ...] = self._extract_cell_types()
        self._feature_columns: tuple[str, ...] = self._extract_feature_columns()
        self._class_index: dict[str, int] = {
            str(name): idx for idx, name in enumerate(self._cell_types)
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def cell_types(self) -> tuple[str, ...]:
        return self._cell_types

    def score(self, connectome: Mapping[str, Any], sbm: Mapping[str, Any] | None = None) -> float:
        """Return mean predicted probability of the declared type for each neuron."""
        if sbm is None:
            sbm = connectome.get("genome", {})
        frame = self._extract_critic_frame(connectome, sbm)
        return self._realism_score(frame, connectome["node_types"])

    def gate(
        self,
        connectome: Mapping[str, Any],
        sbm: Mapping[str, Any] | None = None,
        threshold: float = REALISM_GATE,
    ) -> bool:
        """Return True if the connectome passes the realism gate."""
        return self.score(connectome, sbm) >= threshold

    # ------------------------------------------------------------------
    # Feature extraction (mirrors GrowerOptimizer internals)
    # ------------------------------------------------------------------

    def _extract_critic_frame(
        self,
        connectome: Mapping[str, Any],
        sbm: Mapping[str, Any],
    ) -> pd.DataFrame:
        weights = self._adjacency_matrix(connectome)
        node_types = list(connectome["node_types"])
        wiring = self._wiring_feature_frame(connectome, node_types)
        structured = self._structured_feature_frame(weights, node_types, sbm)
        features = pd.concat(
            [wiring.reset_index(drop=True), structured.reset_index(drop=True)], axis=1
        )
        return self._align_features(features)

    def _wiring_feature_frame(
        self,
        connectome: Mapping[str, Any],
        node_types: Sequence[str],
    ) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as tmpdir:
            edge_path = Path(tmpdir) / f"edges_{connectome.get('seed', 0)}.parquet"
            self._write_edge_table(edge_path, connectome["adjacency"])
            frame, _stats = compute_wiring_features(
                edges=EdgeSource(
                    path=str(edge_path),
                    pre="pre",
                    post="post",
                    weight="weight",
                    unique_pairs=True,
                    format="parquet",
                ),
                node_ids=[str(i) for i in range(connectome["n_neurons"])],
                node_categories=list(node_types),
                params=WiringFeatureParams(
                    top_k_neuropils=0, reciprocity=True, two_hop=True, category_tag=""
                ),
            )
        return frame

    def _structured_feature_frame(
        self,
        weights: np.ndarray,
        node_types: Sequence[str],
        sbm: Mapping[str, Any],
    ) -> pd.DataFrame:
        node_count = len(node_types)
        biases: Mapping[str, Mapping[str, float]] = sbm.get("compartment_bias", {})

        def _axon_out(t: str) -> float:
            return float(biases.get(t, {}).get("axon_out_frac", _DEFAULT_COMPARTMENT_BIAS["axon_out_frac"]))

        def _dendrite_in(t: str) -> float:
            return float(biases.get(t, {}).get("dendrite_in_frac", _DEFAULT_COMPARTMENT_BIAS["dendrite_in_frac"]))

        cmpt = pd.DataFrame(
            {
                "cmpt__axon_output_share": [_axon_out(t) for t in node_types],
                "cmpt__axon_input_share": [1.0 - _dendrite_in(t) for t in node_types],
            }
        )

        out_total = weights.sum(axis=1)
        in_total = weights.sum(axis=0)
        etype_acc = {n: np.zeros(node_count, dtype=np.float64) for n in ("aa", "ad", "da", "dd")}
        in_acc = {n: np.zeros(node_count, dtype=np.float64) for n in ("aa", "ad", "da", "dd")}
        rows, cols = np.nonzero(weights)
        for pre_idx, post_idx in zip(rows, cols):
            w = float(weights[pre_idx, post_idx])
            pre_type = node_types[int(pre_idx)]
            post_type = node_types[int(post_idx)]
            ao, din = _axon_out(pre_type), _dendrite_in(post_type)
            fracs = {"aa": ao * (1.0 - din), "ad": ao * din, "da": (1.0 - ao) * (1.0 - din), "dd": (1.0 - ao) * din}
            for n, f in fracs.items():
                c = w * f
                etype_acc[n][pre_idx] += c
                in_acc[n][post_idx] += c

        etype = {}
        with np.errstate(divide="ignore", invalid="ignore"):
            for n in ("aa", "ad", "da", "dd"):
                etype[f"etype_out__{n}"] = np.where(out_total > 0.0, etype_acc[n] / out_total, 0.0)
                etype[f"etype_in__{n}"] = np.where(in_total > 0.0, in_acc[n] / in_total, 0.0)

        ase = self._spectral_features(weights)
        return pd.concat([pd.DataFrame(etype), cmpt, ase], axis=1)

    def _spectral_features(self, weights: np.ndarray) -> pd.DataFrame:
        n = weights.shape[0]
        out = np.zeros((n, _CRITIC_ASE_DIM), dtype=np.float64)
        inn = np.zeros((n, _CRITIC_ASE_DIM), dtype=np.float64)
        if n >= 2 and np.any(weights > 0.0):
            t = np.log1p(weights)
            k = min(_CRITIC_ASE_DIM, n - 1)
            if k > 0:
                try:
                    if n <= (_CRITIC_ASE_DIM + 1):
                        u, s, vt = np.linalg.svd(t, full_matrices=False)
                        u, s, vt = u[:, :k], s[:k], vt[:k, :]
                    else:
                        from scipy import sparse
                        from scipy.sparse.linalg import svds
                        u, s, vt = svds(
                            sparse.csr_matrix(t), k=k,
                            v0=np.full(n, 1.0 / math.sqrt(n)), solver="arpack",
                        )
                    sqrt_s = np.sqrt(np.maximum(s, 0.0))
                    out[:, :k] = u[:, :k] * sqrt_s[np.newaxis, :k]
                    inn[:, :k] = (vt[:k, :] * sqrt_s[:k, np.newaxis]).T
                except Exception:
                    pass
        cols_out = {f"ase_out_{i}": out[:, i] for i in range(_CRITIC_ASE_DIM)}
        cols_in = {f"ase_in_{i}": inn[:, i] for i in range(_CRITIC_ASE_DIM)}
        return pd.DataFrame({**cols_out, **cols_in})

    def _align_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        aligned = frame.copy()
        for col in self._feature_columns:
            if col not in aligned.columns:
                aligned[col] = 0.0
        aligned = aligned.loc[:, list(self._feature_columns)]
        return aligned.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64)

    def _realism_score(
        self, frame: pd.DataFrame, node_types: Sequence[str]
    ) -> float:
        probabilities = np.asarray(self._model.predict_proba(frame), dtype=np.float64)
        if probabilities.shape[0] != len(node_types):
            raise ValueError("critic returned a different row count than node_types")
        scores = []
        for i, ntype in enumerate(node_types):
            cls_idx = self._class_index.get(str(ntype))
            scores.append(0.0 if cls_idx is None else float(probabilities[i, cls_idx]))
        return float(np.mean(scores)) if scores else 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _adjacency_matrix(self, connectome: Mapping[str, Any]) -> np.ndarray:
        n = connectome["n_neurons"]
        w = np.zeros((n, n), dtype=np.float64)
        for pre, post, weight in connectome["adjacency"]:
            w[int(pre), int(post)] += float(weight)
        return w

    def _write_edge_table(
        self, path: Path, adjacency: Sequence[tuple[int, int, int]]
    ) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        if adjacency:
            pre, post, weight = zip(*adjacency)
        else:
            pre, post, weight = (), (), ()
        pq.write_table(
            pa.table({
                "pre": pa.array(list(pre), type=pa.int64()),
                "post": pa.array(list(post), type=pa.int64()),
                "weight": pa.array(list(weight), type=pa.float64()),
            }),
            path,
        )

    def _extract_cell_types(self) -> tuple[str, ...]:
        clf = getattr(getattr(self._model, "named_steps", {}), "get", lambda *_: None)("clf")
        if clf is not None and hasattr(clf, "classes_"):
            return tuple(str(c) for c in clf.classes_)
        if hasattr(self._model, "classes_"):
            return tuple(str(c) for c in self._model.classes_)
        raise ValueError("cannot extract cell_types from critic model")

    def _extract_feature_columns(self) -> tuple[str, ...]:
        names = getattr(self._model, "feature_names_in_", None)
        if names is not None:
            return tuple(str(n) for n in names)
        schema_path = self.model_path.with_name("schema.json")
        if schema_path.is_file():
            import json
            payload = json.loads(schema_path.read_text())
            numeric = [str(n) for n in payload.get("numeric", [])]
            categorical = [str(n) for n in payload.get("categorical", [])]
            if numeric or categorical:
                return tuple(numeric + categorical)
        raise ValueError("critic model has no feature_names_in_ and no schema.json sibling")
