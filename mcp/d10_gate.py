"""D10 pair-interaction gate: numpy-only inference, pinned artifact, shadow logging.

investigation_reason keeps a finding for the reasoning prompt when the cosine of its
nomic embedding to the question is at least ``ground_threshold`` (0.59). This module
scores the same (question, finding) pairs with a small MLP over pair features and,
in SHADOW mode only, logs both gates' decisions side by side. It never changes what
the live gate keeps: the caller ignores everything this module returns.

The model was chosen offline (docs/d10_shadow_gate.md). Its labels are structural
proxies (same ``dt_target`` tag = grounded), from four recovered deep-think runs, on
finding-finding pairs; the live pairs are question-finding pairs. Shadow logging is
how we find out whether any of that transfers before it decides anything.

* ``pair_features`` is the one definition of the feature vector, used by the trainer
  (deep_think_loci/grounding/train_d10_pair_gate.py) and by inference, so the two
  cannot drift: ``[u, v, |u-v|, u*v]`` of the unit-normalised embeddings plus ten
  tabular columns (``TABULAR_COLUMNS``).
* ``load_gate`` reads ``models/d10_pair_gate/{manifest.json, weights.npz}``. The
  manifest's sha256 must equal ``PINNED_MANIFEST_SHA256`` and the weights' sha256 must
  equal the manifest's, or it refuses. ``np.load(allow_pickle=False)``; no pickle.
* ``record_shadow`` is the fail-open shadow path. Off unless ``LOCI_D10_SHADOW=1``.
  Errors are swallowed and counted (``shadow_error_count``), never raised.

Log rows (``<data home>/instrumentation/d10_shadow.jsonl``) carry ids, scores, enums
and counts only, like every other instrumentation log (see instrumentation_log.py).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from grounding_gate import CONTEXT_CAP  # investigation_reason puts at most 12 kept findings in the prompt

logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path(__file__).resolve().parent / "models" / "d10_pair_gate"
MANIFEST_NAME = "manifest.json"
WEIGHTS_NAME = "weights.npz"
ARTIFACT_SCHEMA = "loci-d10-pair-gate/v1"
# sha256 of the committed manifest.json. The manifest in turn pins the weights'
# sha256, so this one constant binds the code to one trained artifact. Retraining
# means re-exporting and updating this value in the same commit.
PINNED_MANIFEST_SHA256 = "7f2c4c4ab20d57953c59d1811ece3af42eaaf7eb4977483517c68c1f1c82f2e4"

MAX_TEXT_CHARS = 2000   # the grounding dataset builder and memcheck.llm.embed_texts truncate here
TABULAR_COLUMNS = ("cos", "cos2", "token_jaccard", "len_ratio", "log_len_min", "log_len_max",
                   "neg_any", "neg_xor", "log_num_shared", "log_num_only_one")
WEIGHT_KEYS = ("scaler_mean", "scaler_scale", "W1", "b1", "W2", "b2")

SHADOW_ENV = "LOCI_D10_SHADOW"
MAX_PAIRS_ENV = "LOCI_D10_SHADOW_MAX_PAIRS"
DEFAULT_MAX_PAIRS = 512
SHADOW_LOG_NAME = "d10_shadow.jsonl"
SHADOW_SCHEMA = 1

_NEG = re.compile(r"\b(?:not|no|never|none|cannot|without|neither|nor)\b|n't\b", re.IGNORECASE)
_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")


class D10GateError(ValueError):
    """The artifact is missing, altered or malformed; the gate refuses to load it."""


# --------------------------------------------------------------------------- features


def _token_jaccard(x: str, y: str) -> float:
    sa, sb = set(x.split()), set(y.split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _len_ratio(x: str, y: str) -> float:
    la, lb = len(x), len(y)
    return min(la, lb) / (max(la, lb) + 1)


def text_flags(x: str, y: str) -> list[float]:
    """Columns 2..9 of ``TABULAR_COLUMNS``: cheap, symmetric, no embedding."""
    x, y = x[:MAX_TEXT_CHARS], y[:MAX_TEXT_CHARS]
    nx, ny = bool(_NEG.search(x)), bool(_NEG.search(y))
    numx, numy = set(_NUM.findall(x)), set(_NUM.findall(y))
    return [_token_jaccard(x, y), _len_ratio(x, y),
            math.log1p(min(len(x), len(y))), math.log1p(max(len(x), len(y))),
            float(nx or ny), float(nx != ny),
            math.log1p(len(numx & numy)), math.log1p(len(numx ^ numy))]


def unit_rows(emb: Any) -> np.ndarray:
    """Row-normalise raw embeddings the way the dataset builder did (``/ (norm + 1e-9)``)."""
    v = np.asarray(emb, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {v.shape}")
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def pair_features(emb_a: Any, emb_b: Any, texts_a: Sequence[str], texts_b: Sequence[str]) -> np.ndarray:
    """Feature matrix ``[u, v, |u-v|, u*v, tabular]`` (4d + 10 columns) for pairs (a_i, b_i).

    Embeddings may be raw; they are unit-normalised here. The cosine column is the dot
    product of the normalised vectors.
    """
    u, v = unit_rows(emb_a), unit_rows(emb_b)
    if u.shape != v.shape or len(texts_a) != len(u) or len(texts_b) != len(u):
        raise ValueError("pair_features: embeddings and texts must have one row per pair")
    cos = np.sum(u * v, axis=1)
    flags = np.array([text_flags(x, y) for x, y in zip(texts_a, texts_b)], dtype=np.float64)
    flags = flags.reshape(len(u), len(TABULAR_COLUMNS) - 2)
    return np.column_stack([u, v, np.abs(u - v), u * v, cos, cos ** 2, flags])


# --------------------------------------------------------------------------- inference


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # 0.5 * (1 + tanh(z/2)) is the logistic function without exp overflow.
    return 0.5 * (1.0 + np.tanh(0.5 * z))


@dataclass(frozen=True)
class PairGate:
    """StandardScaler + one ReLU hidden layer + logistic output, as exported from sklearn."""

    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    tau: float
    version: str
    manifest_sha256: str

    @property
    def n_features(self) -> int:
        return int(self.W1.shape[0])

    def score(self, X: Any) -> np.ndarray:
        """P(grounded) per row of a feature matrix from ``pair_features``."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"expected (n, {self.n_features}) features, got {X.shape}")
        z = (X - self.scaler_mean) / self.scaler_scale
        h = np.maximum(z @ self.W1 + self.b1, 0.0)
        return _sigmoid(h @ self.W2 + self.b2).reshape(-1)

    def score_pairs(self, emb_a: Any, emb_b: Any, texts_a: Sequence[str], texts_b: Sequence[str]) -> np.ndarray:
        return self.score(pair_features(emb_a, emb_b, texts_a, texts_b))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_gate(directory: Path | str = ARTIFACT_DIR, *,
              expected_manifest_sha256: str | None = PINNED_MANIFEST_SHA256) -> PairGate:
    """Load and verify the artifact. Raises ``D10GateError`` on any mismatch.

    ``expected_manifest_sha256=None`` skips only the pin (for tests that export their
    own artifact); the weights are always checked against the manifest.
    """
    d = Path(directory)
    mpath, wpath = d / MANIFEST_NAME, d / WEIGHTS_NAME
    if not mpath.is_file() or not wpath.is_file():
        raise D10GateError(f"no D10 artifact at {d}")
    raw = mpath.read_bytes()
    msha = hashlib.sha256(raw).hexdigest()
    if expected_manifest_sha256 is not None and msha != expected_manifest_sha256:
        raise D10GateError(f"{MANIFEST_NAME} sha256 {msha[:12]} is not the pinned "
                           f"{expected_manifest_sha256[:12]}; refusing to load")
    manifest = json.loads(raw)
    if manifest.get("schema") != ARTIFACT_SCHEMA:
        raise D10GateError(f"artifact schema {manifest.get('schema')!r} is not {ARTIFACT_SCHEMA}")
    want = (manifest.get("weights") or {}).get("sha256")
    got = sha256_file(wpath)
    if not want or got != want:
        raise D10GateError(f"{WEIGHTS_NAME} sha256 {got[:12]} does not match the manifest; refusing to load")
    with np.load(wpath, allow_pickle=False) as z:
        if sorted(z.files) != sorted(WEIGHT_KEYS):
            raise D10GateError(f"{WEIGHTS_NAME} holds {sorted(z.files)}, expected {sorted(WEIGHT_KEYS)}")
        arrays = {k: np.asarray(z[k], dtype=np.float64) for k in WEIGHT_KEYS}
    n_in = int(manifest["feature_dim"])
    hidden = int(manifest["recipe"]["hidden"])
    shapes = {"scaler_mean": (n_in,), "scaler_scale": (n_in,), "W1": (n_in, hidden), "b1": (hidden,),
              "W2": (hidden, 1), "b2": (1,)}
    for k, shape in shapes.items():
        if arrays[k].shape != shape or not np.all(np.isfinite(arrays[k])):
            raise D10GateError(f"{k}: shape {arrays[k].shape} (want {shape}) or non-finite values")
    if n_in != 4 * int(manifest["embed_dim"]) + len(TABULAR_COLUMNS):
        raise D10GateError(f"feature_dim {n_in} does not fit embed_dim {manifest['embed_dim']}")
    if np.any(arrays["scaler_scale"] <= 0):
        raise D10GateError("scaler_scale must be positive")
    tau = float(manifest["threshold"]["tau"])
    return PairGate(tau=tau, version=f"{manifest.get('version', '?')}+{msha[:12]}", manifest_sha256=msha, **arrays)


# --------------------------------------------------------------------------- shadow

_lock = threading.Lock()
_errors = 0
_cached: PairGate | None = None


def shadow_error_count() -> int:
    """Shadow-path failures swallowed since process start."""
    return _errors


def _count_error(exc: BaseException) -> None:
    global _errors
    with _lock:
        _errors += 1
    logger.debug("d10 shadow gate failed (fail-open, live decision unaffected): %r", exc)


def _gate() -> PairGate:
    """Load once per process; a failed load is retried on the next call (and counted)."""
    global _cached
    if _cached is None:
        _cached = load_gate()
    return _cached


def _max_pairs() -> int:
    try:
        return max(1, int(os.environ.get(MAX_PAIRS_ENV, "") or DEFAULT_MAX_PAIRS))
    except ValueError:
        return DEFAULT_MAX_PAIRS


def _rank_desc(values: Sequence[float]) -> list[int]:
    """1-based rank of each value, highest first; ties keep input order (stable)."""
    order = sorted(range(len(values)), key=lambda i: -values[i])
    rank = [0] * len(values)
    for r, i in enumerate(order, start=1):
        rank[i] = r
    return rank


def mlp_decisions(scores: Sequence[float], tau: float) -> tuple[list[bool], list[bool]]:
    """Per pair: kept (score >= tau) and in the prompt (kept and among the CONTEXT_CAP highest kept)."""
    keep = [float(s) >= tau for s in scores]
    rank = _rank_desc([float(s) if k else -math.inf for s, k in zip(scores, keep)])
    return keep, [bool(k and r <= CONTEXT_CAP) for k, r in zip(keep, rank)]


def record_shadow(log_path: Path, *, investigation_id: str, question: str, finding_ids: Sequence[str],
                  finding_texts: Sequence[str], vectors: Sequence[Sequence[float]], cosines: Sequence[Any],
                  cos_threshold: float, context_ids: Sequence[str]) -> bool:
    """Score each (question, finding) pair and append one log row per pair. Never raises.

    ``vectors`` are the embeddings investigation_reason already computed: question first,
    then one per finding. ``cosines`` and ``context_ids`` are the live gate's scores and
    the ids it put in the prompt; they are recorded, never recomputed or altered.
    Returns True when rows were written.
    """
    try:
        t0 = time.perf_counter_ns()
        gate = _gate()
        n = len(finding_ids)
        if n == 0 or len(vectors) != n + 1 or len(finding_texts) != n or len(cosines) != n:
            raise ValueError("record_shadow: vectors, texts, cosines and ids disagree in length")
        emb = np.asarray(vectors, dtype=np.float64)
        q = np.repeat(emb[:1], n, axis=0)
        # u = question (the claim being grounded), v = finding (the evidence).
        scores = gate.score_pairs(q, emb[1:], [question] * n, list(finding_texts))
        latency_us = (time.perf_counter_ns() - t0) / 1000.0

        cos = [float(c) if c is not None and math.isfinite(float(c)) else None for c in cosines]
        cos_rank = _rank_desc([c if c is not None else -math.inf for c in cos])
        mlp = [float(s) for s in scores]
        mlp_keep, mlp_ctx = mlp_decisions(mlp, gate.tau)
        in_context = set(str(i) for i in context_ids)
        call_id = uuid.uuid4().hex[:16]
        ts = datetime.now(timezone.utc).isoformat()
        limit = _max_pairs()
        rows = []
        for i in sorted(range(n), key=lambda i: cos_rank[i])[:limit]:
            rows.append({
                "schema": SHADOW_SCHEMA,
                "event": "d10_shadow",
                "ts": ts,
                "call_id": call_id,
                "investigation_id": str(investigation_id),
                "finding_id": str(finding_ids[i]),
                "n_pairs": n,
                "n_logged": min(n, limit),
                "cos": None if cos[i] is None else round(cos[i], 4),
                "cos_threshold": float(cos_threshold),
                "cos_keep": bool(cos[i] is not None and cos[i] >= cos_threshold),
                "cos_rank": cos_rank[i],
                "cos_in_context": str(finding_ids[i]) in in_context,
                "mlp_score": round(mlp[i], 6),
                "mlp_tau": round(gate.tau, 6),
                "mlp_keep": mlp_keep[i],
                "mlp_in_context": mlp_ctx[i],
                "gate_version": gate.version,
                "latency_us_call": round(latency_us, 1),
                "latency_us_per_pair": round(latency_us / n, 2),
                "shadow_errors_total": _errors,
            })
        from instrumentation_log import append_rows

        if not append_rows(Path(log_path), rows):
            raise OSError("instrumentation append failed")
        return True
    except Exception as exc:  # noqa: BLE001 - the shadow path must never fail the tool
        _count_error(exc)
        return False
