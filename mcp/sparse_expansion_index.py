"""Sparse-expansion binary index inspired by the fly mushroom body (offline study).

NOT wired into the server. This is design-study code for an offline A/B
evaluation (``eval/fly_sparse_index/``); nothing in ``server.py`` imports it.

The mechanism is the FlyHash family (Dasgupta, Stevens & Navlakha 2017,
Science 358:793):

1. **Input projection** ``D -> G`` "glomeruli": random Gaussian, PCA (fitted
   on the indexed vectors, optionally whitened) or identity (``G == D``).
2. **Sparse binary expansion** ``G -> K`` "Kenyon cells": each KC sums a few
   glomeruli ("claws"). The wiring comes from an :class:`ExpansionSpec`:
   either idealised random (uniform glomeruli, fixed claw count) or a
   parameter file exported by a connectome project, loaded read-only with
   sha256 verification (see :func:`load_spec`).
3. **k-winners-take-all** (APL-like global inhibition): the top
   ``round(wta_fraction * K)`` KCs of each item become 1, the rest 0.
4. **Search** over bit-packed codes by Hamming distance or Jaccard
   similarity, plus an optional **novelty score** (minimum normalised
   Hamming distance to every stored code).

Pure numpy, deterministic under ``seed``. Loci does not import any
connectome code: the only contract with the data source is the JSON + ``.npz``
file pair described in :func:`load_spec`.

Spec-file contract (``fly-mb-params/v1``)
-----------------------------------------
``<name>.json`` with ``schema_version``, ``glomeruli`` (list of G names),
``matrices.<key>.binary_key`` / ``shape``, ``primary`` (default matrix key),
``matrix_file.name`` + ``matrix_file.sha256`` (the sibling ``.npz``) and
``params_sha256`` (sha256 of the canonical JSON with that field blanked).
The caller must pin the sha256 of the JSON file bytes; any mismatch fails
closed with :class:`SpecIntegrityError`.

Scaling beyond the measured matrix (``mode="matrix"``)
------------------------------------------------------
``K <= N`` (N = measured KCs with at least one claw): K columns are drawn
without replacement. ``K > N``: all N columns are used once, and the extra
``K - N`` columns are drawn with replacement and then randomised by
degree-preserving checkerboard swaps (10 per connection), which keeps every
extra KC's claw count and every glomerulus' connection count exactly while
breaking duplicate columns. ``mode="sample"`` instead draws each KC's claw
count from the measured distribution (conditioned on >= 1 claw) and its
glomeruli without replacement with probability proportional to each
glomerulus' share of all measured connections.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

SPEC_SCHEMA_VERSION = "fly-mb-params/v1"
_PROJECTIONS = ("pca", "random", "identity")
_MODES = ("sample", "matrix")
_METRICS = ("hamming", "jaccard")


class SpecIntegrityError(ValueError):
    """A spec file failed verification. Never fall back to a default spec."""


@dataclass(frozen=True)
class ExpansionSpec:
    """Wiring statistics for the G -> K expansion.

    ``claw_pmf[c]`` is the probability that a KC has ``c`` claws.
    ``glomerulus_weights`` (length G, sums to 1) are claw sampling weights.
    ``matrix`` (G x N uint8) is an optional measured connectivity matrix.
    """

    name: str
    n_glomeruli: int
    claw_pmf: np.ndarray
    glomerulus_weights: np.ndarray
    matrix: Optional[np.ndarray] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        w = np.asarray(self.glomerulus_weights, dtype=np.float64)
        pmf = np.asarray(self.claw_pmf, dtype=np.float64)
        if w.shape != (self.n_glomeruli,) or self.n_glomeruli < 1:
            raise ValueError("glomerulus_weights must have length n_glomeruli >= 1")
        if (w < 0).any() or not np.isclose(w.sum(), 1.0):
            raise ValueError("glomerulus_weights must be non-negative and sum to 1")
        if pmf.ndim != 1 or (pmf < 0).any() or not np.isclose(pmf.sum(), 1.0) or pmf[1:].sum() <= 0:
            raise ValueError("claw_pmf must be a pmf with mass on >= 1 claw")
        if self.matrix is not None and self.matrix.shape[0] != self.n_glomeruli:
            raise ValueError("matrix must have n_glomeruli rows")

    @property
    def mean_claws(self) -> float:
        pmf = np.asarray(self.claw_pmf, dtype=np.float64)
        pos = pmf.copy()
        pos[0] = 0.0
        pos /= pos.sum()
        return float((np.arange(len(pos)) * pos).sum())

    def with_fixed_claws(self, claws: int) -> "ExpansionSpec":
        """Same glomerulus weights, every KC gets exactly ``claws`` claws."""
        if claws < 1 or claws > self.n_glomeruli:
            raise ValueError("claws must be in [1, n_glomeruli]")
        pmf = np.zeros(claws + 1)
        pmf[claws] = 1.0
        return replace(self, name=f"{self.name}+claws{claws}", claw_pmf=pmf, matrix=None)

    def with_uniform_glomeruli(self) -> "ExpansionSpec":
        """Same claw distribution, glomeruli sampled uniformly (a control)."""
        g = self.n_glomeruli
        return replace(self, name=f"{self.name}+uniformglom", glomerulus_weights=np.full(g, 1.0 / g), matrix=None)


def idealised_spec(n_glomeruli: int, claws_per_kc: int) -> ExpansionSpec:
    """Idealised FlyHash wiring: uniform glomeruli, fixed claws per KC."""
    if claws_per_kc < 1 or claws_per_kc > n_glomeruli:
        raise ValueError("claws_per_kc must be in [1, n_glomeruli]")
    pmf = np.zeros(claws_per_kc + 1)
    pmf[claws_per_kc] = 1.0
    return ExpansionSpec(
        name=f"idealised-random-G{n_glomeruli}-c{claws_per_kc}",
        n_glomeruli=n_glomeruli,
        claw_pmf=pmf,
        glomerulus_weights=np.full(n_glomeruli, 1.0 / n_glomeruli),
        provenance={"kind": "idealised_random", "reference": "Dasgupta, Stevens & Navlakha 2017"},
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _params_digest(payload: Mapping[str, Any]) -> str:
    canonical = json.loads(json.dumps(payload))
    canonical["params_sha256"] = ""
    return _sha256_bytes(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def spec_from_matrix(matrix: np.ndarray, *, name: str, provenance: Optional[Mapping[str, Any]] = None) -> ExpansionSpec:
    """Build a spec from a binary G x N matrix (glomeruli with no claw dropped)."""
    m = (np.asarray(matrix) > 0).astype(np.uint8)
    if m.ndim != 2 or m.size == 0:
        raise SpecIntegrityError("matrix must be a non-empty 2-D array")
    rows = m.sum(axis=1)
    m = m[rows > 0]
    if m.shape[0] == 0 or m.sum() == 0:
        raise SpecIntegrityError("matrix has no connections")
    claws = m.sum(axis=0).astype(np.int64)
    pmf = np.bincount(claws).astype(np.float64)
    pmf /= pmf.sum()
    weights = m.sum(axis=1).astype(np.float64)
    weights /= weights.sum()
    return ExpansionSpec(name=name, n_glomeruli=int(m.shape[0]), claw_pmf=pmf, glomerulus_weights=weights,
                         matrix=m, provenance=dict(provenance or {}))


def load_spec(json_path: str | Path, *, expected_sha256: str, matrix_key: Optional[str] = None) -> ExpansionSpec:
    """Load a ``fly-mb-params/v1`` spec read-only; fail closed on any mismatch.

    Checks, in order: the pinned sha256 of the JSON bytes, the schema version,
    the ``params_sha256`` self-digest, that the matrix file is a plain sibling
    file name, the ``.npz`` sha256, and the matrix shape. The ``.npz`` is
    loaded with ``allow_pickle=False``.
    """
    path = Path(json_path)
    expected = str(expected_sha256 or "").strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise SpecIntegrityError("expected_sha256 must be a 64-hex sha256 of the spec JSON")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SpecIntegrityError(f"spec file unreadable: {exc}") from exc
    actual = _sha256_bytes(raw)
    if actual != expected:
        raise SpecIntegrityError(f"spec sha256 mismatch: got {actual}, pinned {expected}")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpecIntegrityError(f"spec JSON invalid: {exc}") from exc
    if payload.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise SpecIntegrityError(f"unsupported schema_version {payload.get('schema_version')!r}")
    if payload.get("params_sha256") != _params_digest(payload):
        raise SpecIntegrityError("params_sha256 self-digest mismatch")
    mfile = payload.get("matrix_file") or {}
    name = str(mfile.get("name") or "")
    if not name or Path(name).name != name or name in (".", ".."):
        raise SpecIntegrityError("matrix_file.name must be a plain sibling file name")
    npz_path = path.parent / name
    try:
        npz_bytes = npz_path.read_bytes()
    except OSError as exc:
        raise SpecIntegrityError(f"matrix file unreadable: {exc}") from exc
    if _sha256_bytes(npz_bytes) != mfile.get("sha256"):
        raise SpecIntegrityError("matrix file sha256 mismatch")
    key = matrix_key or payload.get("primary")
    entry = (payload.get("matrices") or {}).get(key)
    if not entry:
        raise SpecIntegrityError(f"matrix {key!r} not in spec")
    import io

    with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as data:
        if entry["binary_key"] not in data.files:
            raise SpecIntegrityError(f"npz lacks {entry['binary_key']!r}")
        matrix = np.array(data[entry["binary_key"]])
    if list(matrix.shape) != list(entry.get("shape") or []) or matrix.shape[0] != len(payload.get("glomeruli") or []):
        raise SpecIntegrityError("matrix shape does not match the spec")
    return spec_from_matrix(matrix, name=f"measured-{key}", provenance={
        "kind": "measured",
        "matrix_key": key,
        "spec_sha256": actual,
        "params_sha256": payload["params_sha256"],
        "threshold_synapses": payload.get("threshold_synapses"),
        "source": {k: {kk: v.get(kk) for kk in ("dataset_symbol", "version_id", "citation", "licence")}
                   for k, v in (payload.get("provenance") or {}).items() if isinstance(v, dict)},
    })


# ---------------------------------------------------------------- connectivity


def _sample_connectivity(spec: ExpansionSpec, n_kc: int, rng: np.random.Generator) -> np.ndarray:
    g = spec.n_glomeruli
    pmf = np.asarray(spec.claw_pmf, dtype=np.float64).copy()
    pmf[0] = 0.0  # a KC with no glomerular claw carries no signal
    pmf /= pmf.sum()
    claws = np.minimum(rng.choice(len(pmf), size=n_kc, p=pmf), g)
    w = np.asarray(spec.glomerulus_weights, dtype=np.float64)
    conn = np.zeros((n_kc, g), dtype=np.uint8)
    for i, c in enumerate(claws):
        conn[i, rng.choice(g, size=int(c), replace=False, p=w)] = 1
    return conn


def _degree_preserving_shuffle(block: np.ndarray, rng: np.random.Generator, swaps_per_edge: int = 10) -> np.ndarray:
    """Checkerboard swaps on a (K x G) 0/1 block: row and column sums unchanged."""
    b = block.copy()
    n_rows = b.shape[0]
    if n_rows < 2:
        return b
    for _ in range(swaps_per_edge * int(b.sum())):
        r1, r2 = rng.choice(n_rows, size=2, replace=False)
        only1 = np.flatnonzero((b[r1] == 1) & (b[r2] == 0))
        only2 = np.flatnonzero((b[r2] == 1) & (b[r1] == 0))
        if only1.size == 0 or only2.size == 0:
            continue
        a = only1[rng.integers(only1.size)]
        c = only2[rng.integers(only2.size)]
        b[r1, a], b[r1, c], b[r2, c], b[r2, a] = 0, 1, 0, 1
    return b


def _matrix_connectivity(spec: ExpansionSpec, n_kc: int, rng: np.random.Generator) -> np.ndarray:
    if spec.matrix is None:
        raise ValueError("mode='matrix' needs a spec with a measured matrix")
    cols = spec.matrix.T[spec.matrix.sum(axis=0) > 0].astype(np.uint8)  # (N, G)
    n = cols.shape[0]
    if n_kc <= n:
        return cols[np.sort(rng.choice(n, size=n_kc, replace=False))]
    extra = cols[rng.choice(n, size=n_kc - n, replace=True)]
    return np.vstack([cols, _degree_preserving_shuffle(extra, rng)])


def build_connectivity(spec: ExpansionSpec, n_kc: int, *, mode: str = "sample", seed: int = 0) -> np.ndarray:
    """(K x G) 0/1 connectivity. Deterministic for a given seed."""
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}")
    if n_kc < 1:
        raise ValueError("n_kc must be >= 1")
    rng = np.random.default_rng(seed)
    if mode == "matrix":
        return _matrix_connectivity(spec, n_kc, rng)
    return _sample_connectivity(spec, n_kc, rng)


# ---------------------------------------------------------------- codes and search


def k_winners(activations: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of the top-``k`` entries per row; ties broken by lower index."""
    a = np.asarray(activations, dtype=np.float64)
    n, m = a.shape
    if not 1 <= k <= m:
        raise ValueError("k must be in [1, n_columns]")
    order = np.argsort(-a, axis=1, kind="stable")[:, :k]
    mask = np.zeros((n, m), dtype=bool)
    np.put_along_axis(mask, order, True, axis=1)
    return mask


def pack_codes(bits: np.ndarray) -> np.ndarray:
    """Pack a boolean (n, K) array into uint64 words (n, ceil(K/64))."""
    bits = np.asarray(bits, dtype=bool)
    n, k = bits.shape
    pad = (-k) % 64
    if pad:
        bits = np.hstack([bits, np.zeros((n, pad), dtype=bool)])
    packed = np.packbits(bits, axis=1, bitorder="little")
    return np.ascontiguousarray(packed).view(np.uint64).reshape(n, -1)


def popcount(words: np.ndarray) -> np.ndarray:
    return np.bitwise_count(words).sum(axis=-1, dtype=np.int64)


def hamming_distances(queries: np.ndarray, codes: np.ndarray, *, block: int = 32) -> np.ndarray:
    """(nq, n) Hamming distances between packed code sets."""
    out = np.empty((queries.shape[0], codes.shape[0]), dtype=np.int64)
    for s in range(0, queries.shape[0], block):
        q = queries[s:s + block]
        out[s:s + block] = popcount(q[:, None, :] ^ codes[None, :, :])
    return out


def jaccard_similarity(queries: np.ndarray, codes: np.ndarray, *, block: int = 32) -> np.ndarray:
    out = np.empty((queries.shape[0], codes.shape[0]), dtype=np.float64)
    for s in range(0, queries.shape[0], block):
        q = queries[s:s + block]
        inter = popcount(q[:, None, :] & codes[None, :, :])
        union = popcount(q[:, None, :] | codes[None, :, :])
        out[s:s + block] = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    return out


def top_k_from_scores(scores: np.ndarray, k: int, *, larger_is_better: bool) -> np.ndarray:
    """Indices of the best ``k`` per row, ties broken by lower index (stable)."""
    key = -scores if larger_is_better else scores
    k = min(k, scores.shape[1])
    return np.argsort(key, axis=1, kind="stable")[:, :k]


class SparseExpansionIndex:
    """Mushroom-body-style sparse binary index (offline design study)."""

    def __init__(
        self,
        spec: ExpansionSpec,
        n_kc: int,
        *,
        wta_fraction: float = 0.05,
        projection: str = "pca",
        whiten: bool = False,
        mode: str = "sample",
        metric: str = "hamming",
        seed: int = 0,
    ) -> None:
        if projection not in _PROJECTIONS:
            raise ValueError(f"projection must be one of {_PROJECTIONS}")
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}")
        if not 0.0 < wta_fraction <= 1.0:
            raise ValueError("wta_fraction must be in (0, 1]")
        self.spec = spec
        self.n_kc = int(n_kc)
        self.k_active = max(1, int(round(wta_fraction * n_kc)))
        self.projection = projection
        self.whiten = whiten
        self.mode = mode
        self.metric = metric
        self.seed = int(seed)
        self.connectivity = build_connectivity(spec, self.n_kc, mode=mode, seed=self.seed).astype(np.float32)
        self._mean: Optional[np.ndarray] = None
        self._proj: Optional[np.ndarray] = None
        self._codes = np.zeros((0, (self.n_kc + 63) // 64), dtype=np.uint64)

    # -- projection -----------------------------------------------------
    def fit(self, x: np.ndarray) -> "SparseExpansionIndex":
        """Fit the D -> G projection on ``x`` (the vectors to be indexed)."""
        x = np.asarray(x, dtype=np.float64)
        d = x.shape[1]
        g = self.spec.n_glomeruli
        self._mean = x.mean(axis=0)
        rng = np.random.default_rng(self.seed + 1_000_003)
        if self.projection == "identity":
            if g != d:
                raise ValueError("identity projection needs n_glomeruli == input dim")
            proj = np.eye(d)
        elif self.projection == "random":
            proj = rng.standard_normal((d, g)) / np.sqrt(d)
        else:
            if g > min(x.shape):
                raise ValueError("PCA needs n_glomeruli <= min(n_items, dim)")
            _, s, vt = np.linalg.svd(x - self._mean, full_matrices=False)
            proj = vt[:g].T
            if self.whiten:
                proj = proj / np.maximum(s[:g] / np.sqrt(max(x.shape[0] - 1, 1)), 1e-12)
            # assign components to glomeruli in a seeded random order, so a
            # heavily sampled glomerulus is not systematically the top PC
            proj = proj[:, rng.permutation(g)]
        self._proj = proj.astype(np.float32)
        return self

    def glomeruli(self, x: np.ndarray) -> np.ndarray:
        if self._proj is None or self._mean is None:
            raise RuntimeError("call fit() first")
        return (np.asarray(x, dtype=np.float32) - self._mean.astype(np.float32)) @ self._proj

    # -- codes -------------------------------------------------------------
    def encode_bits(self, x: np.ndarray) -> np.ndarray:
        kc = self.glomeruli(x) @ self.connectivity.T
        return k_winners(kc, self.k_active)

    def encode(self, x: np.ndarray) -> np.ndarray:
        return pack_codes(self.encode_bits(x))

    def add(self, x: np.ndarray) -> None:
        self._codes = np.vstack([self._codes, self.encode(x)])

    def __len__(self) -> int:
        return int(self._codes.shape[0])

    @property
    def code_bytes_per_item(self) -> int:
        return int(self._codes.shape[1] * 8) if len(self) else ((self.n_kc + 63) // 64) * 8

    @property
    def memory_bytes(self) -> int:
        """Stored codes + connectivity (as index lists) + projection."""
        conn = int(self.connectivity.sum()) * 2
        proj = 0 if self._proj is None else int(self._proj.nbytes + self._mean.nbytes)
        return int(self._codes.nbytes) + conn + proj

    # -- search ------------------------------------------------------------
    def search(self, q: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Top-``k`` stored items per query: (indices, scores).

        Scores are Hamming distances (smaller is better) or Jaccard
        similarities (larger is better) depending on ``metric``.
        """
        if not len(self):
            raise RuntimeError("index is empty")
        qc = self.encode(q)
        if self.metric == "hamming":
            scores = hamming_distances(qc, self._codes)
            idx = top_k_from_scores(scores, k, larger_is_better=False)
        else:
            scores = jaccard_similarity(qc, self._codes)
            idx = top_k_from_scores(scores, k, larger_is_better=True)
        return idx, np.take_along_axis(scores, idx, axis=1)

    def novelty(self, q: np.ndarray) -> np.ndarray:
        """Minimum normalised Hamming distance to any stored code, in [0, 1]."""
        if not len(self):
            raise RuntimeError("index is empty")
        d = hamming_distances(self.encode(q), self._codes)
        return d.min(axis=1) / float(2 * self.k_active)


__all__ = [
    "ExpansionSpec",
    "SparseExpansionIndex",
    "SpecIntegrityError",
    "build_connectivity",
    "hamming_distances",
    "idealised_spec",
    "jaccard_similarity",
    "k_winners",
    "load_spec",
    "pack_codes",
    "spec_from_matrix",
    "top_k_from_scores",
]
