"""GraphSAGE-style node classifier for FlyBrain connectomes (plain torch, full batch).

Why a graph model: the per-neuron wiring features (``flybrain_wiring_features``)
summarize a neuron's partners once. A message-passing model lets every neuron
read its partners' *features* (and their partners', two layers deep), so
structure that a single summary row cannot express becomes learnable.

Pieces
------
* ``GraphTopology`` / ``build_graph_topology``: the annotated-annotated edge
  list of a connectome (pre -> post, synapse weight) streamed from any pyarrow
  edge table with column projection, restricted to the node table, cached as
  a write-once ``.npz`` keyed by a fingerprint of the inputs (never under
  ``snapshots/``).
* ``GraphContext``: topology + the node feature frame (one row per graph node,
  the SAME label-excluded wiring features every other learner sees) + the set
  of held-out node ids.
* ``GraphSageLearner`` (backend ``graphsage``, registered with
  ``flybrain_learners``): input projection, ``layers`` SAGE layers with
  separate in-edge (presynaptic partners) and out-edge (postsynaptic partners)
  mean aggregation over ``log1p(weight)``-normalized sparse adjacency, a
  residual + LayerNorm per layer, dropout, and a jumping-knowledge head. Full
  batch (MaleCNS: 212k nodes / 26M annotated pairs fits in ~2 GB of GPU
  memory). Early stopping on a grouped inner split of the training rows, then
  a refit on all training rows for the selected epoch count.

  ``fit(X, y, groups)`` takes the usual feature frame plus the reserved column
  ``GRAPH_NODE_COLUMN`` (the node id; never a feature: it only says which
  graph node a row is). The feature columns of ``X`` select which context
  columns the network reads for *every* node (so a drop-one-family ablation
  is just a narrower ``X``) and must equal the context's values on those rows.

  Modes. ``transductive``: message passing runs over the full graph (held-out
  nodes' *features* flow, never their labels: labels only enter the loss, and
  the wiring features were built with held-out nodes' categories masked).
  ``inductive``: while training, every held-out node (``GraphContext.heldout``)
  and every edge touching one is removed from the graph (and the inner
  early-stopping nodes are removed while training on the inner split);
  inference then runs on the full graph, as for new, unseen neurons.

* ``neighbor_vote_predictions``: the graph-view trivial rule (weighted vote of
  the training labels of a node's direct partners, majority fallback). A graph
  model must beat it, not only the one-column rules.
* ``run_graph_evaluation``: the ``flybrain_model_eval`` protocol for graph
  models (same grouped split, same feature-view baselines, cluster bootstrap,
  gate, label-shuffle control, drop-one-family ablation, report format), with
  the transductive and inductive GNN and an ``hgb`` learner on the SAME split
  and features, plus a paired bootstrap of GNN minus hgb.
* ``prepare_mc_task`` + ``main``: the male-cns experiments (``nt_ground_truth``,
  ``super_class``, ``cell_class``); see ``python flybrain_graph_model.py -h``.

Artifacts use the foundation format (``flybrain_learners.save_learner``):
``model.pt`` is a torch ``state_dict`` loaded with ``weights_only=True``;
``encoder.json`` / ``graph.json`` are JSON. A loaded learner predicts only
after ``attach_graph`` with a context whose topology fingerprint matches.

Determinism: CPU runs are deterministic for a seed; CUDA sparse matmul uses
atomics, so GPU runs agree only up to float noise.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import flybrain_brain_cluster_baselines as fbb
import flybrain_learners as fl
import flybrain_model_eval as fme
import flybrain_wiring_features as fwf

GRAPH_MODEL_SCHEMA_VERSION = "flybrain-graph-model/v1"
GRAPH_TOPOLOGY_SCHEMA_VERSION = "flybrain-graph-topology/v1"
GRAPH_NODE_COLUMN = "__graph_node__"
VIEW_GRAPH = "graph"
RULE_NEIGHBOR_VOTE = "neighbor_vote"
MODES = ("transductive", "inductive")
DEFAULT_GRAPH_REPORT_ROOT = "/mnt/f/.flybrain/logs/real-models-20260924T174122Z/graph"
DEFAULT_STORAGE_ROOT = "/mnt/f/.flybrain"


# =========================================================================== topology


@dataclass(frozen=True)
class GraphTopology:
    """Directed weighted graph over a fixed node list (edge ``src[e] -> dst[e]``, weight = synapse count)."""

    node_ids: tuple[str, ...]
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    fingerprint: str
    meta: Mapping[str, Any] = field(default_factory=dict)
    _index: dict[str, int] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        n = len(self.node_ids)
        if n == 0:
            raise ValueError("graph has no nodes")
        if len(set(self.node_ids)) != n:
            raise ValueError("graph node ids are not unique")
        if not (len(self.src) == len(self.dst) == len(self.weight)):
            raise ValueError("src, dst and weight lengths differ")
        if len(self.src) and (min(self.src.min(), self.dst.min()) < 0 or max(self.src.max(), self.dst.max()) >= n):
            raise ValueError("edge endpoint out of range")
        if len(self.weight) and (not np.all(np.isfinite(self.weight)) or np.any(self.weight < 0)):
            raise ValueError("edge weights must be finite and non-negative")
        self._index.update({node: i for i, node in enumerate(self.node_ids)})

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return int(len(self.src))

    def filtered(self, min_weight: float) -> "GraphTopology":
        """Edges with weight >= ``min_weight`` (same nodes); fingerprint derived from this one's."""
        if float(min_weight) <= 0:
            return self
        keep = self.weight >= float(min_weight)
        fp = hashlib.sha256(f"{self.fingerprint}|min_weight>={float(min_weight)!r}".encode("utf-8")).hexdigest()
        meta = {**dict(self.meta), "parent_fingerprint": self.fingerprint, "min_weight": float(min_weight),
                "stats": {**dict(self.meta.get("stats", {})), "edges_kept": int(keep.sum())}}
        return GraphTopology(self.node_ids, self.src[keep], self.dst[keep], self.weight[keep], fp, meta)

    def indices(self, ids: Iterable[Any]) -> np.ndarray:
        out = []
        missing = []
        for value in ids:
            key = str(value)
            if key in self._index:
                out.append(self._index[key])
            else:
                missing.append(key)
        if missing:
            raise ValueError(f"{len(missing)} node ids are not in the graph (e.g. {missing[:3]})")
        return np.asarray(out, dtype=np.int64)


def topology_fingerprint(*, dataset: str, source: Mapping[str, Any], node_ids: Sequence[str],
                         min_weight: float) -> str:
    digest = hashlib.sha256()
    for node in node_ids:
        digest.update(f"{node}\n".encode("utf-8"))
    payload = {"schema_version": GRAPH_TOPOLOGY_SCHEMA_VERSION, "dataset": dataset, "source": source,
               "nodes_sha256": digest.hexdigest(), "min_weight": float(min_weight)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
                          .encode("utf-8")).hexdigest()


def graph_from_arrays(node_ids: Sequence[Any], src: Sequence[int], dst: Sequence[int], weight: Sequence[float] | None = None,
                      *, dataset: str = "synthetic", drop_self_loops: bool = True) -> GraphTopology:
    """Build a topology from in-memory index arrays (tests, small graphs). Duplicate pairs are summed."""
    ids = tuple(str(v) for v in node_ids)
    s = np.asarray(src, dtype=np.int64)
    d = np.asarray(dst, dtype=np.int64)
    w = np.ones(len(s), dtype=np.float64) if weight is None else np.asarray(weight, dtype=np.float64)
    s, d, w = _canonical_edges(s, d, w, len(ids), drop_self_loops=drop_self_loops, unique=False)
    source = {"kind": "arrays", "edges_sha256": hashlib.sha256(
        s.tobytes() + d.tobytes() + w.astype(np.float32).tobytes()).hexdigest()}
    fp = topology_fingerprint(dataset=dataset, source=source, node_ids=ids, min_weight=0.0)
    return GraphTopology(ids, s, d, w.astype(np.float32), fp, {"dataset": dataset, "source": source})


def _canonical_edges(src: np.ndarray, dst: np.ndarray, w: np.ndarray, n: int, *, drop_self_loops: bool,
                     unique: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if drop_self_loops:
        keep = src != dst
        src, dst, w = src[keep], dst[keep], w[keep]
    keys = src.astype(np.int64) * n + dst.astype(np.int64)
    if unique:
        order = np.argsort(keys, kind="stable")
        keys, w = keys[order], w[order]
        if len(keys) > 1 and np.any(keys[1:] == keys[:-1]):
            raise ValueError("edge table declared unique_pairs=True but contains duplicate (pre, post) rows")
    else:
        keys, inverse = np.unique(keys, return_inverse=True)
        w = np.bincount(inverse, weights=w, minlength=len(keys))
    return (keys // n).astype(np.int64), (keys % n).astype(np.int64), np.asarray(w, dtype=np.float64)


def build_graph_topology(*, dataset: str, edges: fwf.EdgeSource, node_ids: Sequence[Any], min_weight: float = 0.0,
                         cache_root: str | Path | None = fwf.DEFAULT_CACHE_ROOT, use_cache: bool = True,
                         batch_rows: int = 1 << 20, drop_self_loops: bool = True) -> GraphTopology:
    """Stream ``edges`` once, keep pairs whose endpoints are both in ``node_ids`` (weight >= ``min_weight``).

    Cached under ``<cache_root>/graph-topology/<dataset>/<fingerprint>.npz`` with
    a JSON sidecar holding the file's sha256 (a mismatch is a miss).
    """
    ids = tuple(str(v) for v in node_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("node ids are not unique")
    source = {"kind": "edge_source", "edges": edges.identity(), "drop_self_loops": bool(drop_self_loops)}
    fp = topology_fingerprint(dataset=dataset, source=source, node_ids=ids, min_weight=min_weight)
    cache_path: Path | None = None
    if cache_root is not None:
        cache_dir = Path(cache_root) / "graph-topology" / fwf.slug(dataset)
        fwf._reject_snapshot_path(cache_dir)
        cache_path = cache_dir / f"{fp[:32]}.npz"
        sidecar = cache_path.with_suffix(".json")
        if use_cache and cache_path.is_file() and sidecar.is_file():
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == fp and meta.get("npz_sha256") == fl.sha256_file(cache_path):
                with np.load(cache_path, allow_pickle=False) as payload:
                    return GraphTopology(ids, payload["src"].astype(np.int64), payload["dst"].astype(np.int64),
                                         payload["weight"].astype(np.float32), fp, meta)
    n = len(ids)
    columns = [edges.pre, edges.post] + ([edges.weight] if edges.weight else [])
    table, batches = fwf._iter_batches(edges, columns, batch_rows)
    value_set = fwf._node_value_set(ids, table.schema.field(edges.pre).type)
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    scanned = 0
    for batch in batches:
        if batch.num_rows == 0:
            continue
        scanned += batch.num_rows
        pre = fwf._index_of(batch.column(edges.pre), value_set)
        post = fwf._index_of(batch.column(edges.post), value_set)
        w = fwf._weights(batch, edges.weight)
        keep = (pre >= 0) & (post >= 0) & (w >= float(min_weight))
        if keep.any():
            src_parts.append(pre[keep].astype(np.int32))
            dst_parts.append(post[keep].astype(np.int32))
            w_parts.append(w[keep].astype(np.float32))
    src = np.concatenate(src_parts).astype(np.int64) if src_parts else np.zeros(0, dtype=np.int64)
    dst = np.concatenate(dst_parts).astype(np.int64) if dst_parts else np.zeros(0, dtype=np.int64)
    w64 = np.concatenate(w_parts).astype(np.float64) if w_parts else np.zeros(0)
    del src_parts, dst_parts, w_parts
    src, dst, w64 = _canonical_edges(src, dst, w64, n, drop_self_loops=drop_self_loops, unique=bool(edges.unique_pairs))
    meta = {
        "schema_version": GRAPH_TOPOLOGY_SCHEMA_VERSION,
        "fingerprint": fp,
        "dataset": dataset,
        "source": source,
        "min_weight": float(min_weight),
        "stats": {"edges_scanned": int(scanned), "edges_kept": int(len(src)), "nodes": n,
                  "nodes_with_edges": int(len(np.union1d(src, dst)))},
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_name(cache_path.name + ".tmp.npz")
        np.savez(tmp, src=src.astype(np.int32), dst=dst.astype(np.int32), weight=w64.astype(np.float32))
        os.replace(tmp, cache_path)
        meta["npz_sha256"] = fl.sha256_file(cache_path)
        cache_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return GraphTopology(ids, src, dst, w64.astype(np.float32), fp, meta)


# =========================================================================== context


@dataclass
class GraphContext:
    """Topology + node features (row i = topology node i) + held-out node ids."""

    topology: GraphTopology
    features: pd.DataFrame
    heldout: frozenset[str] = frozenset()
    _adj_cache: dict[Any, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if len(self.features) != self.topology.n_nodes:
            raise ValueError("feature frame rows must equal the number of graph nodes")
        bad = [c for c in self.features.columns if c in (fwf.NODE_ID_COLUMN, GRAPH_NODE_COLUMN, fl.TEXT_COLUMN)]
        if bad:
            raise ValueError(f"context features must not include reserved columns: {bad}")
        self.features = self.features.reset_index(drop=True)
        self.heldout = frozenset(str(v) for v in self.heldout)
        unknown = [h for h in self.heldout if h not in self.topology._index]
        if unknown:
            raise ValueError(f"held-out ids not in the graph: {unknown[:3]}")

    @classmethod
    def from_node_frame(cls, topology: GraphTopology, frame: pd.DataFrame, *, id_column: str = fwf.NODE_ID_COLUMN,
                        heldout: Iterable[Any] = ()) -> "GraphContext":
        """Align a per-node feature frame (with an id column) to the topology's node order; missing nodes fail."""
        keyed = frame.assign(**{id_column: frame[id_column].astype(str)}).set_index(id_column)
        if keyed.index.duplicated().any():
            raise ValueError("node feature frame has duplicate ids")
        missing = [n for n in topology.node_ids if n not in keyed.index]
        if missing:
            raise ValueError(f"{len(missing)} graph nodes have no feature row (e.g. {missing[:3]})")
        aligned = keyed.loc[list(topology.node_ids)].reset_index(drop=True)
        return cls(topology, aligned, frozenset(str(v) for v in heldout))

    def with_heldout(self, ids: Iterable[Any]) -> "GraphContext":
        out = GraphContext(self.topology, self.features, frozenset(str(v) for v in ids))
        out._adj_cache = self._adj_cache  # adjacency depends only on topology + exclusion mask
        return out

    def heldout_mask(self) -> np.ndarray:
        mask = np.zeros(self.topology.n_nodes, dtype=bool)
        if self.heldout:
            mask[self.topology.indices(sorted(self.heldout))] = True
        return mask

    def adjacency(self, exclude: np.ndarray | None, device: Any) -> "Adjacency":
        """Row-normalized in/out adjacency on ``device``; nodes in ``exclude`` lose every edge.

        At most two variants are cached (least recently built evicted first); each variant is 4 CSR
        matrices (~0.3 GB each for 26M edges with int64 indices).
        """
        key_mask = b"" if exclude is None or not exclude.any() else np.packbits(exclude).tobytes()
        key = (str(device), hashlib.sha256(key_mask).hexdigest())
        if key not in self._adj_cache:
            while len(self._adj_cache) >= 2:  # keep the most recent other variant only
                self._adj_cache.pop(next(iter(self._adj_cache)))
            self._adj_cache[key] = _build_adjacency(self.topology, exclude, device)
            _release_device_cache(device)
        return self._adj_cache[key]

    def clear_adjacency_cache(self) -> None:
        devices = {k[0] for k in self._adj_cache}
        self._adj_cache.clear()
        for device in devices:
            _release_device_cache(device)


@dataclass(frozen=True)
class Adjacency:
    """``a_in`` row i averages presynaptic partners (j -> i); ``a_out`` postsynaptic (i -> j); ``*_t`` transposes."""

    a_in: Any
    a_in_t: Any
    a_out: Any
    a_out_t: Any

    def to_dense(self) -> tuple[np.ndarray, np.ndarray]:
        return self.a_in.to_dense().cpu().numpy(), self.a_out.to_dense().cpu().numpy()


def _csr(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, n: int, device: Any) -> Any:
    import torch

    order = np.argsort(rows * n + cols, kind="stable")
    crow = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=n), out=crow[1:])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.sparse_csr_tensor(torch.from_numpy(crow), torch.from_numpy(cols[order].astype(np.int64)),
                                       torch.from_numpy(vals[order].astype(np.float32)), size=(n, n)).to(device)


def _build_adjacency(topology: GraphTopology, exclude: np.ndarray | None, device: Any) -> Adjacency:
    src, dst = topology.src, topology.dst
    w = np.log1p(topology.weight.astype(np.float64))
    if exclude is not None and exclude.any():
        keep = ~(exclude[src] | exclude[dst])
        src, dst, w = src[keep], dst[keep], w[keep]
    n = topology.n_nodes
    in_total = np.bincount(dst, weights=w, minlength=n)
    out_total = np.bincount(src, weights=w, minlength=n)
    v_in = w / np.where(in_total[dst] > 0, in_total[dst], 1.0)    # A_in[dst, src]
    v_out = w / np.where(out_total[src] > 0, out_total[src], 1.0)  # A_out[src, dst]
    return Adjacency(a_in=_csr(dst, src, v_in, n, device), a_in_t=_csr(src, dst, v_in, n, device),
                     a_out=_csr(src, dst, v_out, n, device), a_out_t=_csr(dst, src, v_out, n, device))


def _release_device_cache(device: Any) -> None:
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.empty_cache()


_SPMM_FN: Any = None


def _spmm(a: Any, a_t: Any, h: Any) -> Any:
    """``a @ h`` whose backward uses the precomputed transpose (no per-step sparse transposes/allocations)."""
    global _SPMM_FN
    import torch

    if not torch.is_grad_enabled() or not h.requires_grad:
        return a @ h
    if _SPMM_FN is None:
        class SpmmFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, h, a, a_t):  # type: ignore[override]
                ctx.a_t = a_t
                return a @ h

            @staticmethod
            def backward(ctx, grad):  # type: ignore[override]
                return ctx.a_t @ grad, None, None

        _SPMM_FN = SpmmFn
    return _SPMM_FN.apply(h, a, a_t)


def neighbor_vote_predictions(topology: GraphTopology, train_nodes: np.ndarray, train_labels: Sequence[str],
                              query_nodes: np.ndarray, *, classes: Sequence[str], fallback: str,
                              exclude: np.ndarray | None = None) -> list[str]:
    """Trivial graph rule: log1p-weighted vote of direct partners' TRAINING labels (both directions).

    Nodes with no labelled partner get ``fallback`` (the training majority).
    Ties break to the lexicographically first class (``classes`` is sorted).
    """
    k = len(classes)
    index = {c: i for i, c in enumerate(classes)}
    label = np.full(topology.n_nodes, -1, dtype=np.int64)
    label[np.asarray(train_nodes, dtype=np.int64)] = [index[str(v)] for v in train_labels]
    src, dst = topology.src, topology.dst
    w = np.log1p(topology.weight.astype(np.float64))
    if exclude is not None and exclude.any():
        keep = ~(exclude[src] | exclude[dst])
        src, dst, w = src[keep], dst[keep], w[keep]
    votes = np.zeros(topology.n_nodes * k)
    for target, other in ((dst, src), (src, dst)):
        m = label[other] >= 0
        votes += np.bincount(target[m] * k + label[other][m], weights=w[m], minlength=topology.n_nodes * k)
    votes = votes.reshape(topology.n_nodes, k)[np.asarray(query_nodes, dtype=np.int64)]
    has = votes.sum(axis=1) > 0
    best = np.argmax(votes, axis=1)
    return [classes[b] if h else fallback for b, h in zip(best.tolist(), has.tolist())]


def _sym_norm_adjacency(topology: GraphTopology) -> Any:
    """Undirected ``D^-1/2 (A + A^T) D^-1/2`` over ``log1p(synapses)`` (scipy CSR), as in C&S [Huang 2020]."""
    import scipy.sparse as sp

    n = topology.n_nodes
    w = np.log1p(topology.weight.astype(np.float64))
    a = sp.csr_matrix((w, (topology.src, topology.dst)), shape=(n, n))
    a = (a + a.T).tocsr()
    deg = np.asarray(a.sum(axis=1)).ravel()
    inv = np.where(deg > 0, 1.0 / np.sqrt(np.where(deg > 0, deg, 1.0)), 0.0)
    return (sp.diags(inv) @ a @ sp.diags(inv)).tocsr()


def correct_and_smooth(adjacency: Any, base_proba: np.ndarray, train_nodes: np.ndarray, train_y: np.ndarray, *,
                       alpha_correct: float, alpha_smooth: float, iterations: int = 50) -> np.ndarray:
    """Correct-and-Smooth [Huang 2020] with autoscale, using TRAINING labels only.

    ``base_proba`` (n_nodes x k) are a feature-only model's probabilities for
    every graph node; ``train_y`` are class indices of ``train_nodes``.
    Correct: propagate the training residuals ``Y - Z`` and add them back,
    scaled to the mean training residual. Smooth: propagate the corrected
    scores with training rows clamped to their one-hot labels. Held-out labels
    never enter; returns row-normalized scores (n_nodes x k).
    """
    n, k = base_proba.shape
    train_nodes = np.asarray(train_nodes, dtype=np.int64)
    y_train = np.zeros((len(train_nodes), k))
    y_train[np.arange(len(train_nodes)), np.asarray(train_y, dtype=np.int64)] = 1.0
    z = base_proba.astype(np.float64)
    e0 = np.zeros((n, k))
    e0[train_nodes] = y_train - z[train_nodes]
    e = e0.copy()
    for _ in range(int(iterations)):
        e = (1.0 - alpha_correct) * e0 + alpha_correct * (adjacency @ e)
        e[train_nodes] = e0[train_nodes]
    sigma = float(np.abs(e0[train_nodes]).sum(axis=1).mean()) if len(train_nodes) else 0.0
    norm = np.abs(e).sum(axis=1)
    scale = np.where(norm > 1e-12, sigma / np.where(norm > 1e-12, norm, 1.0), 0.0)
    corrected = z + scale[:, None] * e
    g0 = corrected.copy()
    g0[train_nodes] = y_train
    g = g0.copy()
    for _ in range(int(iterations)):
        g = (1.0 - alpha_smooth) * g0 + alpha_smooth * (adjacency @ g)
    g = np.clip(g, 0.0, None)
    total = g.sum(axis=1, keepdims=True)
    return np.where(total > 0, g / np.where(total > 0, total, 1.0), 1.0 / k)


DEFAULT_CS_GRID: tuple[tuple[float, float], ...] = tuple((a, b) for a in (0.5, 0.8, 0.95) for b in (0.5, 0.8, 0.95))


# =========================================================================== network


def _make_net(n_in: int, hidden: int, n_out: int, layers: int, dropout: float, directed: bool) -> Any:
    import torch
    import torch.nn as nn

    class SageLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin_self = nn.Linear(hidden, hidden)
            self.lin_in = nn.Linear(hidden, hidden, bias=False)
            self.lin_out = nn.Linear(hidden, hidden, bias=False)
            self.norm = nn.LayerNorm(hidden)

        def forward(self, h: Any, adj: Adjacency) -> Any:
            m_in = _spmm(adj.a_in, adj.a_in_t, h)
            m_out = _spmm(adj.a_out, adj.a_out_t, h)
            msg = self.lin_in(m_in) + self.lin_out(m_out) if directed else self.lin_in(0.5 * (m_in + m_out))
            return self.norm(self.lin_self(h) + msg)

    class SageNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Linear(n_in, hidden)
            self.layers = nn.ModuleList([SageLayer() for _ in range(layers)])
            self.head = nn.Linear(hidden * (layers + 1), n_out)
            self.drop = nn.Dropout(dropout)

        def forward(self, x: Any, adj: Adjacency) -> Any:
            h = self.drop(torch.relu(self.inp(x)))
            outs = [h]
            for layer in self.layers:
                h = h + self.drop(torch.relu(layer(h, adj)))  # residual
                outs.append(h)
            return self.head(torch.cat(outs, dim=1))  # jumping knowledge

    return SageNet()


_CAPPED_DEVICES: dict[str, float] = {}


class GpuUnavailableError(RuntimeError):
    """The requested CUDA device lacks free memory (shared box): refuse rather than spill to system RAM."""


def _resolve_device(name: str, max_fraction: float | None = None, min_free_gb: float | None = None) -> Any:
    """CPU unless a CUDA device is requested and available. On CUDA, cap this process's share of the
    device memory (``torch.cuda.set_per_process_memory_fraction``) so an oversized graph raises an OOM
    instead of spilling into WDDM shared system memory under WSL (slow, and it has hung the driver)."""
    import torch

    if not str(name).startswith("cuda"):
        return torch.device(str(name))
    if not torch.cuda.is_available():
        return torch.device("cpu")
    device = torch.device(str(name))
    if min_free_gb is not None and str(device) not in _CAPPED_DEVICES:
        free, total = torch.cuda.mem_get_info(device)
        if free < float(min_free_gb) * (1 << 30):
            raise GpuUnavailableError(
                f"{device} has {free / (1 << 30):.1f} GiB free of {total / (1 << 30):.1f} GiB (< {min_free_gb} GiB needed); "
                "another process holds it. Use device='cpu' or wait.")
    if max_fraction is not None and _CAPPED_DEVICES.get(str(device)) != float(max_fraction):
        torch.cuda.set_per_process_memory_fraction(float(max_fraction), device)
        _CAPPED_DEVICES[str(device)] = float(max_fraction)
    return device


# =========================================================================== learner


class GraphSageLearner(fl.Learner):
    """Full-batch directed GraphSAGE node classifier (see module docstring)."""

    backend = "graphsage"
    model_files = ("model.pt", "encoder.json", "graph.json")
    default_params = {
        "hidden": 128,
        "layers": 2,
        "dropout": 0.3,
        "lr": 5e-3,
        "weight_decay": 5e-4,
        "epochs": 400,
        "patience": 40,
        "min_epochs": 20,
        "stop_metric": "macro_f1",
        "class_weight": None,
        "device": "cpu",
        "max_gpu_memory_fraction": 0.6,
        "min_free_gpu_gb": 4.0,
        "mode": "transductive",
        "directed": True,
        "inner_val_fraction": 0.15,
        "refit": True,
        "min_category_count": 5,
        "max_categories": 64,
    }

    def __init__(self, *, seed: int = 0, **params: Any) -> None:
        super().__init__(seed=seed, **params)
        if self.params["mode"] not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.params["class_weight"] not in (None, "balanced"):
            raise ValueError("class_weight must be None or 'balanced'")
        if self.params["stop_metric"] not in ("loss", "macro_f1"):
            raise ValueError("stop_metric must be 'loss' or 'macro_f1'")
        if int(self.params["layers"]) < 0:
            raise ValueError("layers must be >= 0")
        self.context: GraphContext | None = None
        self.graph_binding: dict[str, Any] = {}
        self._fit_nodes: np.ndarray | None = None
        self._logits: np.ndarray | None = None

    # graph binding --------------------------------------------------------
    def attach_graph(self, context: GraphContext) -> "GraphSageLearner":
        bound = self.graph_binding.get("topology_fingerprint")
        if bound is not None and bound != context.topology.fingerprint:
            raise ValueError("context topology differs from the one this learner was fitted on")
        if self.schema is not None and self.classes_:
            missing = [c for c in self.schema.columns if c not in context.features.columns]
            if missing:
                raise ValueError(f"context lacks fitted feature columns: {missing[:5]}")
        self.context = context
        self._logits = None
        return self

    def _require_context(self) -> GraphContext:
        if self.context is None:
            raise ValueError("graphsage needs a graph: call attach_graph(GraphContext) first")
        return self.context

    def fit(self, X: pd.DataFrame, y: Sequence[Any], *, groups: Sequence[Any] | None = None) -> "GraphSageLearner":
        ctx = self._require_context()
        if GRAPH_NODE_COLUMN not in X.columns:
            raise ValueError(f"graphsage needs the node column {GRAPH_NODE_COLUMN!r}")
        nodes = ctx.topology.indices(X[GRAPH_NODE_COLUMN].tolist())
        if len(set(nodes.tolist())) != len(nodes):
            raise ValueError("duplicate graph nodes in the training rows")
        leaked = sorted(set(X[GRAPH_NODE_COLUMN].astype(str)) & ctx.heldout)
        if leaked:
            raise ValueError(f"{len(leaked)} training rows are held-out nodes (e.g. {leaked[:3]})")
        features = X.drop(columns=[GRAPH_NODE_COLUMN, fl.TEXT_COLUMN], errors="ignore")
        _check_rows_match_context(features, ctx, nodes)
        self._fit_nodes = nodes
        self.graph_binding = {}
        super().fit(features, y, groups=groups)
        self.graph_binding = {
            "topology_fingerprint": ctx.topology.fingerprint,
            "n_nodes": ctx.topology.n_nodes,
            "n_edges": ctx.topology.n_edges,
            "n_inputs": int(self.fit_info.get("n_inputs", 0)),
            "mode": self.params["mode"],
            "n_heldout_nodes": len(ctx.heldout),
        }
        self.training_fingerprint["graph"] = dict(self.graph_binding)
        self.training_fingerprint["library_versions"] = {**self.training_fingerprint.get("library_versions", {}),
                                                         "torch": _torch_version()}
        return self

    # training -------------------------------------------------------------
    def _encode_all(self, ctx: GraphContext) -> np.ndarray:
        assert self.schema is not None
        return self.encoder.transform(self.schema.prepare(ctx.features))

    def _fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None) -> None:
        import torch

        ctx = self._require_context()
        assert self.schema is not None and self._fit_nodes is not None
        t_start = time.time()
        torch.manual_seed(self.seed)
        heldout = ctx.heldout_mask()
        design = self.schema.prepare(ctx.features)
        # Encoder statistics from non-held-out nodes only (features carry no labels, but keep inductive clean).
        self.encoder = fl._DenseEncoder(self.schema, int(self.params["min_category_count"]),
                                        int(self.params["max_categories"])).fit(design.loc[~heldout])
        x_all = self.encoder.transform(design)
        targets = np.asarray([self.classes_.index(v) for v in y.tolist()], dtype=np.int64)
        device = _resolve_device(self.params["device"], float(self.params["max_gpu_memory_fraction"]),
                                 float(self.params["min_free_gpu_gb"]))
        split = None if groups is None else fl._inner_group_split(
            groups, y, fraction=float(self.params["inner_val_fraction"]), seed=self.seed)
        tr, va = split if split is not None else (np.arange(len(y)), np.zeros(0, dtype=np.int64))
        weight = None
        if self.params["class_weight"] == "balanced":
            counts = np.bincount(targets[tr], minlength=len(self.classes_)).astype(np.float64)
            weight = torch.tensor(len(tr) / (len(self.classes_) * np.maximum(counts, 1.0)), dtype=torch.float32,
                                  device=device)
        inductive = self.params["mode"] == "inductive"
        x_t = torch.from_numpy(x_all).to(device)
        nodes = self._fit_nodes
        if len(va):
            train_excl = heldout.copy()
            if inductive:
                train_excl[nodes[va]] = True
            adj_train = ctx.adjacency(train_excl if inductive else None, device)
            adj_eval = ctx.adjacency(heldout if inductive else None, device)
            best_epochs, best_loss = self._train(x_t, adj_train, adj_eval, nodes[tr], targets[tr], nodes[va],
                                                 targets[va], int(self.params["epochs"]), weight, device)
        else:
            best_epochs, best_loss = int(self.params["epochs"]), None
        refit = bool(self.params["refit"]) or not len(va)
        if refit:  # retrain on every training row for the selected epoch count
            torch.manual_seed(self.seed)
            adj_final = ctx.adjacency(heldout if inductive else None, device)
            self._train(x_t, adj_final, adj_final, nodes, targets, np.zeros(0, dtype=np.int64),
                        np.zeros(0, dtype=np.int64), best_epochs, weight, device)
        # else: keep the early-stopped net (trained on the inner-train rows only; cheaper on CPU)
        self.fit_info = {
            "epochs": int(best_epochs),
            "inner_val_loss": best_loss,
            "stop_metric": self.params["stop_metric"],
            "train_seconds": round(time.time() - t_start, 2),
            "early_stopping": "grouped_inner_val" if len(va) else "off",
            "refit_on_all_train_rows": refit,
            "n_inputs": int(x_all.shape[1]),
            "mode": self.params["mode"],
            "device": str(device),
            "n_fit_nodes": int(len(nodes)),
            "n_heldout_nodes_removed_in_training": int(heldout.sum()) if inductive else 0,
        }
        self._logits = None

    def _train(self, x: Any, adj_train: Any, adj_eval: Any, tr_nodes: np.ndarray, tr_y: np.ndarray,
               va_nodes: np.ndarray, va_y: np.ndarray, epochs: int, weight: Any, device: Any) -> tuple[int, float | None]:
        import torch
        import torch.nn.functional as F

        self.net = _make_net(x.shape[1], int(self.params["hidden"]), len(self.classes_), int(self.params["layers"]),
                             float(self.params["dropout"]), bool(self.params["directed"])).to(device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=float(self.params["lr"]),
                                weight_decay=float(self.params["weight_decay"]))
        tr_idx = torch.from_numpy(tr_nodes).to(device)
        tr_t = torch.from_numpy(tr_y).to(device)
        va_idx = torch.from_numpy(va_nodes).to(device) if len(va_nodes) else None
        va_t = torch.from_numpy(va_y).to(device) if len(va_nodes) else None
        # Early stopping on the grouped inner split. Grouped splits shift class priors a lot (whole
        # lineages move together), so val *loss* often bottoms out after 1-2 epochs at the prior;
        # ``stop_metric='macro_f1'`` (default) selects on inner-val macro-F1 (ties: lower loss), never
        # before ``min_epochs``.
        by_f1 = self.params["stop_metric"] == "macro_f1"
        min_epochs = min(int(self.params["min_epochs"]), epochs)
        best_score, best_loss, best_epoch, best_state, stale = -math.inf, math.inf, epochs, None, 0
        for epoch in range(1, epochs + 1):
            self.net.train()
            opt.zero_grad()
            logits = self.net(x, adj_train)
            loss = F.cross_entropy(logits[tr_idx], tr_t, weight=weight)
            loss.backward()
            opt.step()
            if va_idx is not None:
                self.net.eval()
                with torch.no_grad():
                    val_logits = self.net(x, adj_eval)[va_idx]
                    val_loss = float(F.cross_entropy(val_logits, va_t, weight=weight))
                    score = _torch_macro_f1(val_logits.argmax(dim=1), va_t, len(self.classes_)) if by_f1 else -val_loss
                if epoch < min_epochs:
                    continue
                if score > best_score + 1e-6 or (by_f1 and abs(score - best_score) <= 1e-6 and val_loss < best_loss):
                    best_score, best_loss, best_epoch, stale = score, val_loss, epoch, 0
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    stale += 1
                    if stale >= int(self.params["patience"]):
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        del opt
        _release_device_cache(device)
        return best_epoch, (None if best_loss == math.inf else float(best_loss))

    # inference ------------------------------------------------------------
    def node_logits(self) -> np.ndarray:
        """Logits for every graph node (full graph), cached until the next fit/attach."""
        import torch

        ctx = self._require_context()
        if self._logits is None:
            device = next(self.net.parameters()).device
            want = _resolve_device(self.params["device"], float(self.params["max_gpu_memory_fraction"]))
            if device != want:
                self.net.to(want)
                device = want
            x = torch.from_numpy(self._encode_all(ctx)).to(device)
            self.net.eval()
            with torch.no_grad():
                self._logits = self.net(x, ctx.adjacency(None, device)).float().cpu().numpy().astype(np.float64)
        return self._logits

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ctx = self._require_context()
        if GRAPH_NODE_COLUMN not in X.columns:
            raise ValueError(f"graphsage needs the node column {GRAPH_NODE_COLUMN!r}")
        logits = self.node_logits()[ctx.topology.indices(X[GRAPH_NODE_COLUMN].tolist())]
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # artifacts ------------------------------------------------------------
    def _save_model(self, directory: Path) -> None:
        import torch

        torch.save({k: v.detach().cpu() for k, v in self.net.state_dict().items()}, directory / "model.pt")
        fl._write_json(directory / "encoder.json", fl._jsonable(self.encoder.state))
        fl._write_json(directory / "graph.json", fl._jsonable({"schema_version": GRAPH_MODEL_SCHEMA_VERSION,
                                                               **self.graph_binding}))

    def _load_model(self, directory: Path) -> None:
        import torch

        assert self.schema is not None
        binding = json.loads((directory / "graph.json").read_text(encoding="utf-8"))
        if binding.pop("schema_version", None) != GRAPH_MODEL_SCHEMA_VERSION:
            raise ValueError("unsupported graph.json schema_version")
        self.graph_binding = binding
        self.encoder = fl._DenseEncoder(self.schema, int(self.params["min_category_count"]),
                                        int(self.params["max_categories"]))
        self.encoder.state = json.loads((directory / "encoder.json").read_text(encoding="utf-8"))
        n_in = 2 * len(self.schema.numeric) + sum(len(v) + 1 for v in self.encoder.state["categorical"].values())
        if int(binding.get("n_inputs", n_in)) != n_in:
            raise ValueError("graph.json n_inputs disagrees with encoder.json")
        self.net = _make_net(n_in, int(self.params["hidden"]), len(self.classes_), int(self.params["layers"]),
                             float(self.params["dropout"]), bool(self.params["directed"]))
        self.net.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
        self.net.eval()
        self.context = None
        self._logits = None


def _torch_macro_f1(pred: Any, target: Any, k: int) -> float:
    """Macro-F1 over classes present in target or prediction (same convention as fme.macro_f1)."""
    import torch

    cm = torch.bincount(target * k + pred, minlength=k * k).reshape(k, k).double()
    tp = torch.diagonal(cm)
    denom = 2 * tp + (cm.sum(dim=0) - tp) + (cm.sum(dim=1) - tp)
    present = denom > 0
    if not bool(present.any()):
        return 0.0
    return float((2 * tp[present] / denom[present]).mean())


def _torch_version() -> str:
    import torch

    return str(torch.__version__)


def _check_rows_match_context(features: pd.DataFrame, ctx: GraphContext, nodes: np.ndarray) -> None:
    missing = [c for c in features.columns if c not in ctx.features.columns]
    if missing:
        raise ValueError(f"feature columns absent from the graph context: {missing[:5]}")
    ref = ctx.features.iloc[nodes][list(features.columns)].reset_index(drop=True)
    got = features.reset_index(drop=True)
    for col in features.columns:
        a, b = got[col], ref[col]
        if pd.api.types.is_numeric_dtype(a.dtype) and pd.api.types.is_numeric_dtype(b.dtype):
            av = a.to_numpy(dtype=np.float64)
            bv = b.to_numpy(dtype=np.float64)
            same = np.isclose(av, bv, rtol=1e-6, atol=1e-9, equal_nan=True)
        else:
            same = (a.astype(object).where(a.notna(), None) == b.astype(object).where(b.notna(), None)).to_numpy()
            same |= (a.isna() & b.isna()).to_numpy()
        if not bool(np.all(same)):
            raise ValueError(f"feature column {col!r} differs from the graph context on the given nodes")


fl.register_backend("graphsage", GraphSageLearner, replace=True)


def attach_graph(learner: fl.Learner, context: GraphContext) -> fl.Learner:
    """Attach ``context`` to a (possibly calibrated/loaded) graphsage learner."""
    inner = learner
    while isinstance(inner, fl.CalibratedLearner):
        inner = inner.base
    if not isinstance(inner, GraphSageLearner):
        raise ValueError("not a graphsage learner")
    inner.attach_graph(context)
    return learner


# =========================================================================== evaluation


@dataclass(frozen=True)
class GraphEvalConfig:
    eval: fme.EvalConfig = fme.EvalConfig(models=(), report_root=DEFAULT_GRAPH_REPORT_ROOT)
    gnn_grid: tuple[Mapping[str, Any], ...] = ({},)
    modes: tuple[str, ...] = MODES
    device: str = "cpu"
    calibration: str | None = "temperature"
    hgb: fme.ModelSpec | None = fme.ModelSpec("hgb", ({"max_iter": 300}, {"max_iter": 300, "class_weight": "balanced"}),
                                              "isotonic")
    ablation_models: tuple[str, ...] = ("gnn", "hgb")
    gnn_ablation: str = "full"  # "full" (graph + every family) | "graph" (message passing + direction only)
    reload_check: bool = True
    verbose: bool = False
    # HGB + correct-and-smooth baseline (train labels only) [Huang 2020]; () disables it.
    cs_grid: tuple[tuple[float, float], ...] = DEFAULT_CS_GRID
    cs_iterations: int = 50

    def as_dict(self) -> dict[str, Any]:
        return fl._jsonable({"eval": self.eval.as_dict(), "gnn_grid": [dict(g) for g in self.gnn_grid],
                             "modes": list(self.modes), "device": self.device, "calibration": self.calibration,
                             "hgb": None if self.hgb is None else {"backend": self.hgb.backend,
                                                                   "param_grid": [dict(p) for p in self.hgb.param_grid],
                                                                   "calibration": self.hgb.calibration},
                             "ablation_models": list(self.ablation_models), "gnn_ablation": self.gnn_ablation,
                             "reload_check": self.reload_check,
                             "cs_grid": [list(p) for p in self.cs_grid], "cs_iterations": self.cs_iterations})


def _gnn_inputs(data: fme.EvalDataset, rows: np.ndarray, columns: Sequence[str] | None = None) -> pd.DataFrame:
    frame = data.features.iloc[rows] if columns is None else data.features.iloc[rows][list(columns)]
    frame = frame.reset_index(drop=True).copy()
    frame[GRAPH_NODE_COLUMN] = [data.sample_ids[i] for i in rows]
    return frame


def _graph_view(feature_view: Mapping[str, Any], vote: Mapping[str, list[str]], y: Mapping[str, np.ndarray]) -> dict[str, Any]:
    splits = {}
    for split, base in feature_view["splits"].items():
        truth = y[split].tolist()
        preds = {**base["predictions"], RULE_NEIGHBOR_VOTE: vote[split]}
        accs = {**base["accuracy"], RULE_NEIGHBOR_VOTE: fme.accuracy(truth, vote[split])}
        f1s = {**base["macro_f1"], RULE_NEIGHBOR_VOTE: fme.macro_f1(truth, vote[split])}
        best = sorted(accs.items(), key=lambda item: (-item[1], item[0]))[0][0]
        splits[split] = {**{k: v for k, v in base.items() if k not in ("predictions", "accuracy", "macro_f1")},
                         "accuracy": accs, "macro_f1": f1s, "best_rule": best, "best_accuracy": accs[best],
                         "predictions": preds}
    return {"view": VIEW_GRAPH, "rules": {**feature_view["rules"], RULE_NEIGHBOR_VOTE: {
        "applicable": True, "params": {"weights": "log1p(synapses)", "directions": "in+out",
                                       "labels": "train split only", "fallback": "train majority"}}},
            "splits": splits}


def _gate(entry: dict[str, Any], view_base: Mapping[str, Any], boot: Mapping[str, Any], n_test: int,
          config: fme.EvalConfig) -> dict[str, Any]:
    best_rule = view_base["best_rule"]
    gate = fbb.trivial_baseline_gate(
        model_heldout_accuracy=entry["test"]["accuracy"],
        baselines={"best_rule": f"{entry['view']}:{best_rule}", "best_accuracy": view_base["best_accuracy"]},
        heldout_count=int(n_test), config={"min_margin": config.min_margin})
    trivial_f1 = max(view_base["macro_f1"].values())
    paired = boot["paired"]["model-minus-best_trivial"]
    return {
        "trivial_baseline_gate": {k: v for k, v in gate.items() if k != "baselines"},
        "best_trivial_rule": f"{entry['view']}:{best_rule}",
        "best_trivial_accuracy": view_base["best_accuracy"],
        "majority_accuracy": view_base["accuracy"][fbb.RULE_MAJORITY],
        "best_trivial_macro_f1": trivial_f1,
        "beats_trivial_macro_f1": entry["test"]["macro_f1"] > trivial_f1 + config.min_margin,
        "paired_accuracy_gain_ci95": paired["accuracy_diff_ci95"],
        "paired_gain_significant": paired["accuracy_diff_ci95"][0] > 0,
    }


def _hgb_correct_and_smooth(final_h: fl.Learner, ctx: GraphContext, feature_cols: Sequence[str],
                            nodes: Mapping[str, np.ndarray], y: Mapping[str, np.ndarray], classes: Sequence[str],
                            config: "GraphEvalConfig", cfg: fme.EvalConfig) -> dict[str, Any]:
    """HGB probabilities for every graph node, then C&S with train labels only; (alpha_c, alpha_s) tuned on val."""
    base_all = final_h.predict_proba(ctx.features[list(feature_cols)].reset_index(drop=True))
    order = [list(final_h.classes_).index(c) if c in final_h.classes_ else None for c in classes]
    z = np.zeros((ctx.topology.n_nodes, len(classes)))
    for j, src in enumerate(order):
        if src is not None:
            z[:, j] = base_all[:, src]
    adjacency = _sym_norm_adjacency(ctx.topology)
    index = {c: i for i, c in enumerate(classes)}
    train_y = np.asarray([index[str(v)] for v in y["train"]], dtype=np.int64)
    trials = []
    best = None
    for a_c, a_s in config.cs_grid:
        scores = correct_and_smooth(adjacency, z, nodes["train"], train_y, alpha_correct=a_c, alpha_smooth=a_s,
                                    iterations=config.cs_iterations)
        scored = fme._score(y["val"].tolist(), scores[nodes["val"]], list(classes), cfg.ece_bins)
        metric = -scored["log_loss"] if cfg.tune_metric == "log_loss" else scored[cfg.tune_metric]
        trials.append({"params": {"alpha_correct": a_c, "alpha_smooth": a_s}, "val_metric": metric,
                       "val": fme._strip(scored)})
        if best is None or metric > best[0]:
            best = (metric, (a_c, a_s), scores, scored)
    assert best is not None
    scores = best[2]
    # ---- the single held-out test evaluation of the selected (alpha_c, alpha_s)
    test = fme._score(y["test"].tolist(), scores[nodes["test"]], list(classes), cfg.ece_bins)
    entry = {
        "backend": "hgb+correct_and_smooth", "view": VIEW_GRAPH, "calibration": None,
        "calibration_source": "none (C&S scores row-normalized; ECE reported for completeness only)",
        "tuning": {"selection": f"val {cfg.tune_metric}", "trials": trials,
                   "best_params": {"alpha_correct": best[1][0], "alpha_smooth": best[1][1]}},
        "describe": {"method": "Correct-and-Smooth [Huang 2020] over the calibrated hgb; symmetric-normalized "
                               "undirected log1p(synapse) adjacency; autoscale; labels = train split only",
                     "iterations": config.cs_iterations},
        "fit_info": {},
        "val": {"raw": fme._strip(best[3]), "calibrated": fme._strip(best[3])},
        "test": {**fme._strip(test), "raw_uncalibrated": fme._strip(test),
                 "confusion": fme.confusion(y["test"].tolist(), test["predictions"])},
    }
    return {"entry": entry, "test_predictions": test["predictions"]}


def _fit_gnn(params: Mapping[str, Any], mode: str, ctx: GraphContext, X: pd.DataFrame, y: np.ndarray,
             groups: np.ndarray, *, seed: int, device: str) -> GraphSageLearner:
    learner = fl.make_learner("graphsage", seed=seed, **{**dict(params), "mode": mode, "device": device})
    assert isinstance(learner, GraphSageLearner)
    learner.attach_graph(ctx)
    return learner.fit(X, y, groups=groups)


def run_graph_evaluation(data: fme.EvalDataset, context: GraphContext, config: GraphEvalConfig = GraphEvalConfig(),
                         ) -> dict[str, Any]:
    """The ``flybrain_model_eval`` protocol for the graph model + an hgb comparison on the same split."""
    from threadpoolctl import threadpool_limits

    started = time.time()
    cfg = config.eval

    def say(msg: str) -> None:
        if config.verbose:
            print(f"  [eval {time.time() - started:7.1f}s] {msg}", flush=True)

    leakage = fme.check_leakage(data, uses_text=False)
    fwf.assert_features_allowed(list(context.features.columns), data.target)
    if data.features.shape[1] == 0:
        raise ValueError("graph evaluation needs feature columns")
    for mode in config.modes:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
    idx, components, split_info = fme._split(data, cfg)
    y = {name: data.labels[rows] for name, rows in idx.items()}
    classes = tuple(sorted(set(y["train"].tolist())))
    feature_cols = list(data.features.columns)
    heldout_ids = [data.sample_ids[i] for i in np.concatenate([idx["val"], idx["test"]])]
    # The context may already hold every node sharing a held-out group (prepare_mc_task mask_policy="group"):
    # keep those too, so inductive training never sees any node of a val/test cell type or hemilineage.
    ctx = context.with_heldout(set(map(str, heldout_ids)) | set(context.heldout))
    topo = ctx.topology
    nodes = {name: topo.indices([data.sample_ids[i] for i in rows]) for name, rows in idx.items()}

    report: dict[str, Any] = {
        "schema_version": fme.EVAL_REPORT_SCHEMA_VERSION,
        "graph_model_schema_version": GRAPH_MODEL_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": data.dataset,
        "target": data.target,
        "config": cfg.as_dict(),
        "graph_config": config.as_dict(),
        "notes": fl._jsonable(dict(data.notes)),
        "leakage_check": {**leakage, "context_feature_columns_checked": len(context.features.columns),
                          "heldout_nodes": len(heldout_ids),
                          "heldout_nodes_incl_group_members": len(ctx.heldout),
                          "heldout_rule": "val+test node labels never enter training loss; wiring-feature partner "
                                          "categories masked for val+test (masked_split_ids_sha256); inductive mode "
                                          "also removes val+test nodes and their edges from the training graph"},
        "graph": {"fingerprint": topo.fingerprint, "n_nodes": topo.n_nodes, "n_edges": topo.n_edges,
                  "meta": fl._jsonable({k: v for k, v in dict(topo.meta).items() if k != "source"})},
        "split": {**split_info,
                  "counts": {k: int(len(v)) for k, v in idx.items()},
                  "label_counts": {k: pd.Series(v).value_counts().sort_index().to_dict() for k, v in y.items()},
                  "test_labels_unseen_in_train": sorted(set(y["test"].tolist()) - set(classes)),
                  "n_test_components": int(len(set(components[idx["test"]].tolist())))},
        "n_features": len(feature_cols),
        "feature_families": fwf.feature_families(feature_cols),
    }
    g_train = components[idx["train"]]
    g_test = components[idx["test"]].tolist()

    with threadpool_limits(limits=int(cfg.n_threads)):
        fme._set_torch_threads(int(cfg.n_threads))
        feat_view = fme.feature_view_baselines(
            data.features.iloc[idx["train"]].reset_index(drop=True), y["train"],
            {s: (data.features.iloc[idx[s]].reset_index(drop=True), y[s]) for s in ("val", "test")})
        majority = fme._majority(y["train"].tolist())
        vote = {s: neighbor_vote_predictions(topo, nodes["train"], y["train"].tolist(), nodes[s], classes=classes,
                                             fallback=majority) for s in ("val", "test")}
        baselines = {fme.VIEW_FEATURES: feat_view, VIEW_GRAPH: _graph_view(feat_view, vote, y)}

        models: dict[str, Any] = {}
        fitted: dict[str, tuple[str, dict[str, Any], fl.Learner]] = {}
        preds: dict[str, list[str]] = {}
        Xg = {s: _gnn_inputs(data, idx[s]) for s in ("train", "val", "test")}

        for mode in config.modes:
            label = f"graphsage_{mode}"
            t0 = time.time()
            trials = []
            best = None
            for params in config.gnn_grid:
                learner = _fit_gnn(params, mode, ctx, Xg["train"], y["train"], g_train, seed=cfg.seed, device=config.device)
                scored = fme._score(y["val"].tolist(), learner.predict_proba(Xg["val"]), learner.classes_, cfg.ece_bins)
                metric = -scored["log_loss"] if cfg.tune_metric == "log_loss" else scored[cfg.tune_metric]
                trials.append({"params": fl._jsonable(dict(params)), "val_metric": metric, "val": fme._strip(scored),
                               "fit_info": fl._jsonable(learner.fit_info)})
                say(f"{label} {dict(params)}: val {cfg.tune_metric} {metric:.4f} acc {scored['accuracy']:.4f} "
                    f"epochs {learner.fit_info['epochs']} ({learner.fit_info['train_seconds']}s)")
                if best is None or metric > best[0]:
                    best = (metric, dict(params), learner)
            assert best is not None
            base = best[2]
            final: fl.Learner = base
            if config.calibration:
                final = fl.CalibratedLearner(base, method=config.calibration, cv=0).fit_prefit(Xg["val"], y["val"])
            val_raw = fme._score(y["val"].tolist(), base.predict_proba(Xg["val"]), base.classes_, cfg.ece_bins)
            val_cal = fme._score(y["val"].tolist(), final.predict_proba(Xg["val"]), final.classes_, cfg.ece_bins)
            # ---- the single held-out test evaluation of this final config
            test_raw_proba = base.predict_proba(Xg["test"])
            test_proba = final.predict_proba(Xg["test"])
            test = fme._score(y["test"].tolist(), test_proba, final.classes_, cfg.ece_bins)
            test_raw = fme._score(y["test"].tolist(), test_raw_proba, base.classes_, cfg.ece_bins)
            models[label] = {
                "backend": "graphsage", "mode": mode, "view": VIEW_GRAPH, "calibration": config.calibration,
                "calibration_source": "grouped val split (also used for param selection; inner grouped split of "
                                      "train for early stopping)",
                "tuning": {"selection": f"val {cfg.tune_metric}", "trials": trials, "best_params": best[1]},
                "fit_info": fl._jsonable(final.fit_info),
                "describe": final.describe(),
                "val": {"raw": fme._strip(val_raw), "calibrated": fme._strip(val_cal)},
                "test": {**fme._strip(test), "raw_uncalibrated": fme._strip(test_raw),
                         "reliability_table": fl.reliability_table(test_proba, y["test"], final.classes_, n_bins=cfg.ece_bins),
                         "confusion": fme.confusion(y["test"].tolist(), test["predictions"])},
                "fit_seconds": round(time.time() - t0, 3),
            }
            preds[label] = test["predictions"]
            fitted[label] = ("graphsage", {**best[1], "mode": mode}, final)

        if config.hgb is not None:
            spec = config.hgb
            t0 = time.time()
            X_train = fme._inputs(data, idx["train"])
            X_val = fme._inputs(data, idx["val"])
            X_test = fme._inputs(data, idx["test"])
            tuning = fme._tune(spec, X_train, y["train"], g_train, classes, cfg)
            base_h, final_h = fme._fit_final(spec, tuning["best_params"], X_train, y["train"], g_train, classes,
                                             tuning["oof"], cfg)
            val_raw = fme._score(y["val"].tolist(), base_h.predict_proba(X_val), base_h.classes_, cfg.ece_bins)
            val_cal = fme._score(y["val"].tolist(), final_h.predict_proba(X_val), final_h.classes_, cfg.ece_bins)
            test_proba = final_h.predict_proba(X_test)
            test = fme._score(y["test"].tolist(), test_proba, final_h.classes_, cfg.ece_bins)
            test_raw = fme._score(y["test"].tolist(), base_h.predict_proba(X_test), base_h.classes_, cfg.ece_bins)
            models[spec.label] = {
                "backend": spec.backend, "view": fme.VIEW_FEATURES, "calibration": spec.calibration,
                "calibration_source": "grouped CV out-of-fold probabilities on train",
                "tuning": {k: v for k, v in tuning.items() if k != "oof"},
                "fit_info": fl._jsonable(final_h.fit_info), "describe": final_h.describe(),
                "val": {"raw": fme._strip(val_raw), "calibrated": fme._strip(val_cal)},
                "test": {**fme._strip(test), "raw_uncalibrated": fme._strip(test_raw),
                         "reliability_table": fl.reliability_table(test_proba, y["test"], final_h.classes_, n_bins=cfg.ece_bins),
                         "confusion": fme.confusion(y["test"].tolist(), test["predictions"])},
                "fit_seconds": round(time.time() - t0, 3),
            }
            preds[spec.label] = test["predictions"]
            fitted[spec.label] = (spec.backend, dict(tuning["best_params"]), final_h)
            say(f"hgb best {tuning['best_params']} done")
            if config.cs_grid:
                t0 = time.time()
                cs = _hgb_correct_and_smooth(final_h, ctx, feature_cols, nodes, y, classes, config, cfg)
                models["hgb_cs"] = {**cs["entry"], "fit_seconds": round(time.time() - t0, 3)}
                preds["hgb_cs"] = cs["test_predictions"]
                say(f"hgb+C&S best {cs['entry']['tuning']['best_params']} done")
        size = fme._size_baseline(data, idx, y, g_train, cfg)
        if size.get("available"):
            report["size_baseline"] = {k: v for k, v in size.items() if k != "_predictions"}

        # ---- bootstrap CIs + paired gains, gate
        hgb_label = None if config.hgb is None else config.hgb.label
        for label, entry in models.items():
            view_base = baselines[entry["view"]]["splits"]["test"]
            p = {"model": preds[label], "best_trivial": view_base["predictions"][view_base["best_rule"]],
                 "majority": view_base["predictions"][fbb.RULE_MAJORITY]}
            pairs = [("model", "best_trivial"), ("model", "majority")]
            if hgb_label and label != hgb_label:
                p["hgb"] = preds[hgb_label]
                pairs.append(("model", "hgb"))
            boot = fme.grouped_bootstrap(y["test"].tolist(), p, g_test, n_bootstrap=cfg.n_bootstrap, seed=cfg.seed,
                                         pairs=tuple(pairs))
            entry["test"]["bootstrap"] = boot
            entry["gate"] = _gate(entry, view_base, boot, len(idx["test"]), cfg)
            if "model-minus-hgb" in boot["paired"]:
                entry["vs_hgb"] = {"accuracy_diff": entry["test"]["accuracy"] - models[hgb_label]["test"]["accuracy"],
                                   "macro_f1_diff": entry["test"]["macro_f1"] - models[hgb_label]["test"]["macro_f1"],
                                   **boot["paired"]["model-minus-hgb"]}

        say("models fitted; running controls")
        # ---- controls: label shuffle (refit on permuted train labels, scored on test)
        for label, (backend, params, _final) in fitted.items():
            entry = models[label]
            if cfg.shuffle_control:
                rng = np.random.default_rng(cfg.seed + 7919)
                shuffled = y["train"].copy()
                rng.shuffle(shuffled)
                if backend == "graphsage":
                    ctrl = _fit_gnn({k: v for k, v in params.items() if k != "mode"}, params["mode"], ctx,
                                    Xg["train"], shuffled, g_train, seed=cfg.seed, device=config.device)
                    shuffle_pred = ctrl.predict(Xg["test"])
                else:
                    ctrl = fl.make_learner(backend, seed=cfg.seed, **params).fit(
                        fme._inputs(data, idx["train"]), shuffled, groups=g_train)
                    shuffle_pred = ctrl.predict(fme._inputs(data, idx["test"]))
                shuffle_acc = fme.accuracy(y["test"].tolist(), shuffle_pred)
                majority_acc = entry["gate"]["majority_accuracy"]
                entry["shuffle_control"] = {"evaluated_on": "test", "accuracy": shuffle_acc,
                                            "majority_accuracy": majority_acc,
                                            "collapsed_to_majority": shuffle_acc <= majority_acc + cfg.shuffle_tolerance}
            if backend != "graphsage" and cfg.random_split_control:
                spec = config.hgb
                assert spec is not None
                entry["random_split_control"] = fme._random_split_control(data, spec, params, cfg)
            ok_shuffle = entry.get("shuffle_control", {}).get("collapsed_to_majority", True)
            g = entry["gate"]
            g["shuffle_ok"] = ok_shuffle
            g["pass"] = bool(g["trivial_baseline_gate"]["pass"] and g["beats_trivial_macro_f1"]
                             and g["paired_gain_significant"] and ok_shuffle)

        if "hgb_cs" in models:
            g = models["hgb_cs"]["gate"]
            g["shuffle_ok"] = None
            g["pass"] = bool(g["trivial_baseline_gate"]["pass"] and g["beats_trivial_macro_f1"]
                             and g["paired_gain_significant"])
            g["note"] = "report-only baseline: no shuffle control (C&S has no trainable parameters beyond hgb)"
        gnn_labels = [m for m in models if models[m]["backend"] == "graphsage"]
        best_gnn = sorted(gnn_labels, key=lambda m: (-models[m]["val"]["calibrated"]["macro_f1"], m))[0] if gnn_labels else None
        report["best_on_val"] = sorted(models, key=lambda m: (-models[m]["val"]["calibrated"]["macro_f1"], m))[0]
        report["best_gnn_on_val"] = best_gnn
        if cfg.ablation and best_gnn and "gnn" in config.ablation_models:
            _, params, _ = fitted[best_gnn]
            report["ablation"] = _gnn_ablation(data, ctx, params, idx, y, g_train, feature_cols, cfg, config.device,
                                               label=best_gnn, families=config.gnn_ablation == "full")
        say("controls done; ablations")
        if cfg.ablation and hgb_label and "hgb" in config.ablation_models:
            report["ablation_hgb"] = fme._ablation(data, config.hgb, fitted[hgb_label][1], idx, y, g_train,
                                                   feature_cols, cfg)

        if cfg.report_root and cfg.save_models:
            out_dir = fme.report_dir(cfg.report_root, data.dataset, data.target, cfg.run_label)
            for label, (backend, params, final) in fitted.items():
                manifest = fl.save_learner(final, out_dir / "models" / label, overwrite=True,
                                           metrics={"test": {k: v for k, v in models[label]["test"].items()
                                                             if k in ("accuracy", "macro_f1", "ece", "log_loss")},
                                                    "gate": models[label]["gate"]})
                models[label]["artifact"] = {"path": manifest["path"], "manifest_sha256": manifest["manifest_sha256"]}
                if backend == "graphsage" and config.reload_check:
                    loaded = fl.load_learner(out_dir / "models" / label, trusted_root=cfg.report_root,
                                             expected_manifest_sha256=manifest["manifest_sha256"])
                    attach_graph(loaded, ctx)
                    diff = float(np.max(np.abs(loaded.predict_proba(Xg["test"]) - final.predict_proba(Xg["test"]))))
                    models[label]["artifact"]["reload_max_abs_proba_diff"] = diff

    for result in baselines.values():
        for split in result["splits"].values():
            split.pop("predictions", None)
    report["baselines"] = baselines
    report["models"] = models
    report["summary"] = fme.summary_rows(report)
    report["elapsed_seconds"] = round(time.time() - started, 2)
    if cfg.report_root:
        paths = fme.write_report(report, cfg.report_root, run_label=cfg.run_label)
        with open(paths["markdown"], "a", encoding="utf-8") as handle:
            handle.write(render_graph_markdown(report))
    return report


def _gnn_ablation(data: fme.EvalDataset, ctx: GraphContext, params: Mapping[str, Any], idx: Mapping[str, np.ndarray],
                  y: Mapping[str, np.ndarray], g_train: np.ndarray, feature_cols: Sequence[str], cfg: fme.EvalConfig,
                  device: str, *, label: str, families: bool = True) -> dict[str, Any]:
    """Drop-one-family and no-message-passing ablations of the best GNN, scored on val (test untouched)."""
    mode = params["mode"]
    base_params = {k: v for k, v in params.items() if k != "mode"}

    def fit_eval(columns: Sequence[str], overrides: Mapping[str, Any] | None = None) -> dict[str, float]:
        learner = _fit_gnn({**base_params, **dict(overrides or {})}, mode, ctx, _gnn_inputs(data, idx["train"], columns),
                           y["train"], g_train, seed=cfg.seed, device=device)
        pred = learner.predict(_gnn_inputs(data, idx["val"], columns))
        return {"accuracy": fme.accuracy(y["val"].tolist(), pred), "macro_f1": fme.macro_f1(y["val"].tolist(), pred)}

    full = fit_eval(feature_cols)
    rows = []
    for name, overrides in (("message_passing(layers=0)", {"layers": 0}), ("edge_direction(undirected)", {"directed": False})):
        scored = fit_eval(feature_cols, overrides)
        rows.append({"family": name, "n_features": 0, **scored, "delta_accuracy": scored["accuracy"] - full["accuracy"],
                     "delta_macro_f1": scored["macro_f1"] - full["macro_f1"]})
    for family, members in (fme._families(feature_cols, cfg).items() if families else ()):
        kept = [c for c in feature_cols if c not in set(members)]
        if not kept:
            rows.append({"family": family, "n_features": len(members), "skipped": "no features left"})
            continue
        scored = fit_eval(kept)
        rows.append({"family": family, "n_features": len(members), **scored,
                     "delta_accuracy": scored["accuracy"] - full["accuracy"],
                     "delta_macro_f1": scored["macro_f1"] - full["macro_f1"]})
    return {"model": label, "evaluated_on": "val", "full": full,
            "drop_one_family": sorted(rows, key=lambda r: r.get("delta_accuracy", 0.0))}


def render_graph_markdown(report: Mapping[str, Any]) -> str:
    g = report["graph"]
    lines = ["", "## Graph model", "",
             f"Graph: {g['n_nodes']} nodes, {g['n_edges']} directed annotated pairs (fingerprint {g['fingerprint'][:12]}). "
             f"GNN baselines use the graph view = feature-view rules + `{RULE_NEIGHBOR_VOTE}` "
             "(training-label vote of direct partners).", ""]
    test_graph = report["baselines"][VIEW_GRAPH]["splits"]["test"]
    lines.append("Graph-view trivial accuracies on test: " + ", ".join(
        f"{k} {v:.3f}" for k, v in sorted(test_graph["accuracy"].items())))
    lines += ["", "| model | vs hgb acc diff [95% CI] | vs hgb macro-F1 diff [95% CI] | epochs | reload max |dp| |",
              "|---|---|---|---|---|"]
    for label, entry in report["models"].items():
        if entry["backend"] != "graphsage":
            continue
        vs = entry.get("vs_hgb")
        fit = entry.get("fit_info", {}).get("base_fit_info", entry.get("fit_info", {}))
        acc = "-" if not vs else f"{vs['accuracy_diff']:+.3f} [{vs['accuracy_diff_ci95'][0]:+.3f}, {vs['accuracy_diff_ci95'][1]:+.3f}]"
        f1 = "-" if not vs else f"{vs['macro_f1_diff']:+.3f} [{vs['macro_f1_diff_ci95'][0]:+.3f}, {vs['macro_f1_diff_ci95'][1]:+.3f}]"
        reload = entry.get("artifact", {}).get("reload_max_abs_proba_diff")
        lines.append(f"| {label} | {acc} | {f1} | {fit.get('epochs', '-')} | {'-' if reload is None else f'{reload:.2e}'} |")
    abl = report.get("ablation_hgb")
    if abl and "drop_one_family" in abl:
        lines += ["", f"## Feature ablation (hgb, on {abl['evaluated_on']})", "",
                  f"Full: acc {abl['full']['accuracy']:.3f}, macro-F1 {abl['full']['macro_f1']:.3f}", "",
                  "| dropped family | n | acc | d acc | macro-F1 | d macro-F1 |", "|---|---|---|---|---|---|"]
        for row in abl["drop_one_family"]:
            if "skipped" in row:
                lines.append(f"| {row['family']} | {row['n_features']} | skipped | | | |")
            else:
                lines.append(f"| {row['family']} | {row['n_features']} | {row['accuracy']:.3f} | {row['delta_accuracy']:+.3f} | "
                             f"{row['macro_f1']:.3f} | {row['delta_macro_f1']:+.3f} |")
    return "\n".join(lines) + "\n"


# =========================================================================== male-cns tasks

MC_TARGETS = ("nt_ground_truth", "super_class", "cell_class")
_DROP_SUPER = {"unknown", "non_neuronal", "other"}


def prepare_mc_task(target: str, *, storage_root: str = DEFAULT_STORAGE_ROOT, cache_root: str = fwf.DEFAULT_CACHE_ROOT,
                    eval_config: fme.EvalConfig = fme.EvalConfig(models=()), min_class_count: int = 100,
                    min_edge_weight: float = 0.0, mask_policy: str = "group",
                    log=print) -> tuple[fme.EvalDataset, GraphContext]:
    """Labels, grouped split plan, masked wiring features and graph for one male-cns target.

    Nothing is written into the snapshot: hash stamps go to
    ``<cache_root>/hash-stamps/mc``; features and topology to ``<cache_root>``.
    Held-out (val + test) nodes' super-class is masked in every node's
    partner-composition features for EVERY target (the hierarchy and NT are
    correlated, and same-type partners share them).

    ``mask_policy="group"`` (default, fully inductive): the mask AND the
    context's held-out set cover every node that shares a held-out sample's
    cell type or hemilineage (labelled or not), so no same-type / same-lineage
    partner of a test neuron exposes its category in message passing and the
    inductive GNN never trains on such a node. ``"samples"`` masks only the
    sampled val/test nodes (the v1 behaviour).
    """
    if mask_policy not in ("group", "samples"):
        raise ValueError("mask_policy must be 'group' or 'samples'")
    import flybrain_mc_adapter as mca

    if target not in MC_TARGETS:
        raise ValueError(f"target must be one of {MC_TARGETS}")
    t0 = time.time()
    roles = (mca.ROLE_META, mca.ROLE_EDGELIST) + ((mca.ROLE_NT_PREDICTION,) if target == "nt_ground_truth" else ())
    snap = mca.open_mc_snapshot(storage_root, required_roles=roles, stamp_dir=Path(cache_root) / "hash-stamps" / "mc")
    log(f"snapshot verified {snap.manifest_sha256[:12]} ({time.time() - t0:.0f}s)")
    meta = mca.load_mc_annotations(snap.path(mca.ROLE_META), columns=("superclass", "class", "type", "itoleeHl"))
    if target == "super_class":
        labels = pd.Series([fwf.harmonize_super_class(v) for v in meta["superclass"]], index=meta.index)
        labels[labels.isin(_DROP_SUPER)] = None
    elif target == "cell_class":
        labels = meta["class"].map(lambda v: fwf.slug(v).removesuffix("_tbc") if isinstance(v, str) and v.strip() else None)
    else:
        nt = mca.load_mc_nt_predictions(snap.path(mca.ROLE_NT_PREDICTION), body_ids=meta["bodyId"].tolist())
        gt = nt.set_index("body")["ground_truth"]
        raw = meta["bodyId"].map(gt)
        labels = raw.map(lambda v: mca.MC_NT_SHORT_CODES.get(str(v).strip().lower()) if isinstance(v, str) else None)
    frame = meta.assign(label=labels.values)
    frame = frame[frame["label"].notna()].copy()
    counts = frame["label"].value_counts()
    frame = frame[frame["label"].isin(counts[counts >= int(min_class_count)].index)].reset_index(drop=True)
    frame["cell_type"] = frame["type"]
    frame["hemilineage"] = frame["itoleeHl"]
    group_keys = ("cell_type", "hemilineage")
    log(f"{target}: {len(frame)} labelled nodes, classes {frame['label'].value_counts().to_dict()}; "
        f"dropped classes < {min_class_count}: {sorted(counts[counts < int(min_class_count)].index)}")
    plan = fme.plan_grouped_split(frame["bodyId"].tolist(), frame[list(group_keys)].to_dict(orient="records"),
                                  group_keys, eval_config)
    edges = fwf.EdgeSource(path=str(snap.path(mca.ROLE_EDGELIST)), pre="body_pre", post="body_post", weight="weight",
                           provenance={"manifest_sha256": snap.manifest_sha256})
    held_samples = {str(v) for v in list(plan["val"]) + list(plan["test"])}
    if mask_policy == "group":
        held = frame[frame["bodyId"].astype(str).isin(held_samples)]
        hit = np.zeros(len(meta), dtype=bool)
        for key, column in (("cell_type", "type"), ("hemilineage", "itoleeHl")):
            values = {v for v in held[key].dropna().tolist() if str(v).strip()}
            hit |= meta[column].isin(values).to_numpy()
        mask_ids = sorted(held_samples | set(meta.loc[hit, "bodyId"].astype(str)))
    else:
        mask_ids = sorted(held_samples)
    feats = fwf.build_wiring_features(dataset="mc", objective=target, edges=edges, nodes=meta, id_column="bodyId",
                                      category_column="superclass", vocab_map=fwf.harmonize_super_class,
                                      params=fwf.WiringFeatureParams(two_hop=True), cache_root=cache_root,
                                      mask_category_ids=mask_ids)
    log(f"mask_policy={mask_policy}: {len(mask_ids)} nodes masked (sampled val+test {len(held_samples)})")
    log(f"features {feats.frame.shape} excluded {len(feats.excluded_features)} ({time.time() - t0:.0f}s)")
    topo = build_graph_topology(dataset="mc", edges=edges, node_ids=feats.frame[fwf.NODE_ID_COLUMN].tolist(),
                                cache_root=cache_root).filtered(min_edge_weight)
    log(f"graph {topo.n_nodes} nodes {topo.n_edges} edges ({time.time() - t0:.0f}s)")
    in_graph = [m for m in mask_ids if m in topo._index]
    context = GraphContext.from_node_frame(topo, feats.frame, heldout=in_graph if mask_policy == "group" else ())
    fcols = [c for c in feats.frame.columns if c != fwf.NODE_ID_COLUMN]
    merged = frame[["bodyId", "label", *group_keys]].merge(feats.frame, left_on="bodyId", right_on=fwf.NODE_ID_COLUMN,
                                                           how="left", validate="one_to_one")
    data = fme.EvalDataset.from_frame(
        merged, dataset="mc", target=target, id_column="bodyId", label_column="label", feature_columns=fcols,
        group_columns=list(group_keys),
        notes={"masked_split_ids_sha256": fme.split_ids_sha256(plan), "features_fingerprint": feats.fingerprint,
               "features_cache": feats.cache_path, "excluded_features": list(feats.excluded_features),
               "topology_fingerprint": topo.fingerprint, "manifest_sha256": snap.manifest_sha256,
               "min_class_count": int(min_class_count), "label_source": {
                   "super_class": "annotations.superclass harmonized (unknown/non_neuronal/other dropped)",
                   "cell_class": "annotations.class (slug, _tbc merged)",
                   "nt_ground_truth": "body-neurotransmitters.ground_truth (short codes; predictions never used)"}[target],
               "partner_category": "harmonized superclass, masked for held-out nodes (mask_policy)",
               "mask_policy": mask_policy, "masked_partner_category_nodes": len(mask_ids),
               "inductive_heldout_nodes": len(in_graph) if mask_policy == "group" else None,
               "wiring_params": dict(feats.meta.get("params") or {})})
    return data, context


CPU_GNN_GRID: tuple[Mapping[str, Any], ...] = (
    {"hidden": 64, "dropout": 0.3, "class_weight": "balanced", "epochs": 150, "patience": 20, "lr": 0.01,
     "refit": False},
)

DEFAULT_GNN_GRID: tuple[Mapping[str, Any], ...] = (
    {"hidden": 128, "dropout": 0.3, "class_weight": None},
    {"hidden": 128, "dropout": 0.3, "class_weight": "balanced"},
    {"hidden": 256, "dropout": 0.5, "class_weight": "balanced"},
)


GPU_MIN_FREE_GB = 2.0  # R8: a card qualifies only with >= 2 GB free at start; never wait for one
GPU_KEEP_FREE_GB = 1.0  # R8: leave >= 1 GB free on the card (Loci's embedding model must stay loadable)


def select_device(requested: str, *, min_free_gb: float = GPU_MIN_FREE_GB,
                  keep_free_gb: float = GPU_KEEP_FREE_GB) -> dict[str, Any]:
    """R8 device policy: the requested CUDA device if it has >= ``min_free_gb`` free now, else CPU (never wait).

    Returns ``{"device", "name", "memory_fraction", "free_gb", "total_gb", "note"}``. The memory fraction
    caps this process so that at least ``keep_free_gb`` stays free on the card (and never above 0.6).
    Pin the physical card with ``CUDA_VISIBLE_DEVICES`` (``CUDA_DEVICE_ORDER=PCI_BUS_ID``) and pass cuda:0.
    """
    info: dict[str, Any] = {"device": "cpu", "name": "cpu", "memory_fraction": None, "free_gb": None,
                            "total_gb": None, "note": None,
                            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if not str(requested).startswith("cuda"):
        return info
    import torch

    if not torch.cuda.is_available():
        info["note"] = "cpu_fallback: cuda unavailable"
        return info
    device = torch.device(str(requested))
    free, total = torch.cuda.mem_get_info(device)
    free_gb, total_gb = free / (1 << 30), total / (1 << 30)
    info.update(free_gb=round(free_gb, 2), total_gb=round(total_gb, 2))
    if free_gb < float(min_free_gb):
        info["note"] = f"cpu_fallback: gpus busy ({free_gb:.1f} GiB free < {min_free_gb} GiB)"
        return info
    fraction = min(0.6, max(0.05, (free_gb - float(keep_free_gb)) / total_gb))
    info.update(device=str(device), name=torch.cuda.get_device_name(device), memory_fraction=round(fraction, 3))
    return info


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FlyBrain graph-model experiments on male-cns")
    parser.add_argument("--target", choices=MC_TARGETS, required=True)
    parser.add_argument("--device", default="cuda:0",
                        help="cuda:N (falls back to cpu when < 2 GB free: R8, never wait) or cpu")
    parser.add_argument("--storage-root", default=DEFAULT_STORAGE_ROOT)
    parser.add_argument("--cache-root", default=fwf.DEFAULT_CACHE_ROOT)
    parser.add_argument("--report-root", default=DEFAULT_GRAPH_REPORT_ROOT)
    parser.add_argument("--run-label", default="v1")
    parser.add_argument("--seeds", default="0", help="comma-separated model seeds (same split); one report each")
    parser.add_argument("--mask-policy", choices=("group", "samples"), default="group")
    parser.add_argument("--min-class-count", type=int, default=100)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--quick", action="store_true", help="one GNN config, no ablations (smoke)")
    parser.add_argument("--no-ablation", action="store_true", help="skip ablations (seeds > first)")
    parser.add_argument("--light-controls", action="store_true",
                        help="skip shuffle / random-split controls on every seed (a second process adding seeds)")
    parser.add_argument("--grid", choices=("default", "cpu"), default="default",
                        help="cpu: one small balanced config (for CPU-only runs)")
    parser.add_argument("--gnn-ablation", choices=("full", "graph"), default="full",
                        help="graph: only message-passing/direction ablations for the GNN (hgb still ablates families)")
    parser.add_argument("--min-edge-weight", type=float, default=0.0,
                        help="drop graph edges with fewer synapses (features are unaffected)")
    args = parser.parse_args(argv)
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    dev = select_device(args.device)
    log(f"device {dev['device']} ({dev['name']}) memory_fraction={dev['memory_fraction']} "
        f"free={dev['free_gb']}/{dev['total_gb']} GiB CUDA_VISIBLE_DEVICES={dev['cuda_visible_devices']} "
        f"note={dev['note']}")
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    base_cfg = fme.EvalConfig(models=(), cv_folds=args.cv_folds, n_bootstrap=args.n_bootstrap, n_threads=args.n_threads,
                              report_root=args.report_root, run_label=args.run_label, ablation=False)
    data, context = prepare_mc_task(args.target, storage_root=args.storage_root, cache_root=args.cache_root,
                                    eval_config=base_cfg, min_class_count=args.min_class_count,
                                    min_edge_weight=args.min_edge_weight, mask_policy=args.mask_policy, log=log)
    grid = CPU_GNN_GRID if (args.grid == "cpu" or dev["device"] == "cpu") else DEFAULT_GNN_GRID
    extra = {"max_gpu_memory_fraction": dev["memory_fraction"], "min_free_gpu_gb": GPU_MIN_FREE_GB} \
        if dev["device"] != "cpu" else {}
    grid = tuple({**g, **extra} for g in (grid[:1] if args.quick else grid))
    for i, seed in enumerate(seeds):
        label = args.run_label if len(seeds) == 1 else f"{args.run_label}-s{seed}"
        ablate = not (args.quick or args.no_ablation) and i == 0
        # Later seeds only re-measure seed variance: the legacy shuffle and random-split controls (and the
        # ablation) run on the first seed; the GNN / hgb / hgb+C&S comparison runs on every seed.
        first = i == 0 and not args.light_controls
        ecfg = dataclasses.replace(base_cfg, seed=seed, run_label=label, ablation=ablate,
                                   shuffle_control=first, random_split_control=first)
        data.notes = {**dict(data.notes), "device": dev, "seed": seed}
        gcfg = GraphEvalConfig(eval=ecfg, gnn_grid=grid, device=dev["device"], verbose=True,
                               gnn_ablation=args.gnn_ablation)
        log(f"seed {seed}: run_label={label} ablation={ablate} grid={len(grid)} device={dev['device']}")
        report = run_graph_evaluation(data, context, gcfg)
        for row in report["summary"]:
            print(json.dumps(fl._jsonable({**row, "seed": seed})), flush=True)
        for name, entry in report["models"].items():
            if "vs_hgb" in entry:
                print(name, "vs hgb", json.dumps(fl._jsonable(entry["vs_hgb"])), flush=True)
        log(f"seed {seed} done; report under {fme.report_dir(args.report_root, 'mc', args.target, label)}")
        context.clear_adjacency_cache()
    return 0


__all__ = [
    "CPU_GNN_GRID",
    "DEFAULT_CS_GRID",
    "correct_and_smooth",
    "select_device",
    "DEFAULT_GNN_GRID",
    "DEFAULT_GRAPH_REPORT_ROOT",
    "GRAPH_MODEL_SCHEMA_VERSION",
    "GRAPH_NODE_COLUMN",
    "GraphContext",
    "GraphEvalConfig",
    "GpuUnavailableError",
    "GraphSageLearner",
    "GraphTopology",
    "MODES",
    "RULE_NEIGHBOR_VOTE",
    "VIEW_GRAPH",
    "attach_graph",
    "build_graph_topology",
    "graph_from_arrays",
    "main",
    "neighbor_vote_predictions",
    "prepare_mc_task",
    "render_graph_markdown",
    "run_graph_evaluation",
]


if __name__ == "__main__":
    raise SystemExit(main())
