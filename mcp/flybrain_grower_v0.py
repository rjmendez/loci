"""FlyBrain Grower v0 - public genome/connectome contract."""
from __future__ import annotations
import json, math, time
from pathlib import Path
from typing import Any, Mapping
import numpy as np

GENOME_SCHEMA_VERSION = "flybrain-grower-genome/v0"
_REQUIRED_STATS_KEYS = {"cell_types","neuron_counts","connection_probability","mean_synapse_count","std_synapse_count"}
_DEFAULT_COMPARTMENT_BIAS = {"axon_out_frac": 0.75, "dendrite_in_frac": 0.68}
_L1EM_NT_BY_TYPE = {
    "sensory": "acetylcholine",
    "ascending": "unknown",
    "PN": "acetylcholine",
    "PN-somato": "acetylcholine",
    "LHN": "acetylcholine",
    "KC": "acetylcholine",
    "MBON": "acetylcholine",
    "MBIN": "dopamine_or_octopamine",
    "LN": "gaba",
    "CN": "gaba",
    "interneuron": "unknown",
    "DN-SEZ": "unknown",
    "DN-VNC": "unknown",
    "pre-DN-SEZ": "unknown",
    "pre-DN-VNC": "unknown",
    "RGN": "unknown",
    "MB-FBN": "acetylcholine",
    "MB-FFN": "acetylcholine",
}


def genome_from_data(type_stats_path):
    """Load l1em_type_stats.json and return a fly-constrained genome dict."""
    path = Path(type_stats_path)
    stats = json.loads(path.read_text())
    missing = _REQUIRED_STATS_KEYS - set(stats.keys())
    if missing:
        raise ValueError(f"type_stats missing required fields: {sorted(missing)}")
    cell_types = list(stats["cell_types"])
    if not cell_types:
        raise ValueError("cell_types must be non-empty")
    type_counts = {t: int(stats["neuron_counts"][t]) for t in cell_types}
    n_neurons = sum(type_counts.values())
    if n_neurons < 2:
        raise ValueError(f"n_neurons must be >= 2, got {n_neurons}")
    connection_probs = {}
    for pre in cell_types:
        connection_probs[pre] = {}
        for post in cell_types:
            p = float(stats["connection_probability"][pre][post])
            connection_probs[pre][post] = max(0.0, min(1.0, p))
    syn_mean_raw = stats["mean_synapse_count"]
    syn_std_raw = stats["std_synapse_count"]
    synapse_count_params = {}
    for pre in cell_types:
        synapse_count_params[pre] = {}
        for post in cell_types:
            mean = float(syn_mean_raw[pre][post])
            std = float(syn_std_raw[pre][post])
            if not math.isfinite(mean) or mean <= 0.0:
                mean = 1.0
            if not math.isfinite(std) or std <= 0.0:
                std = math.sqrt(mean)
            synapse_count_params[pre][post] = {"mean": mean, "std": std}
    compartment_bias = {t: dict(_DEFAULT_COMPARTMENT_BIAS) for t in cell_types}
    genome = {
        "schema_version": GENOME_SCHEMA_VERSION,
        "cell_types": cell_types,
        "type_counts": type_counts,
        "connection_probs": connection_probs,
        "synapse_count_params": synapse_count_params,
        "compartment_bias": compartment_bias,
        "nt_type": {t: _L1EM_NT_BY_TYPE.get(t, "unknown") for t in cell_types},
        "metadata": {
            "source_path": str(path),
            "source_dataset": stats.get("dataset", "unknown"),
            "source_snapshot": stats.get("snapshot", "unknown"),
            "source_schema_version": stats.get("schema_version", "unknown"),
            "constraint_tier": "fly_constrained",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    validate_genome(genome)
    return genome


def validate_genome(genome):
    """Validate genome against spec section 3. Raises ValueError on failure."""
    for key in ("schema_version","cell_types","type_counts","connection_probs","synapse_count_params"):
        if key not in genome:
            raise ValueError(f"genome missing required key: {key!r}")
    cell_types = list(genome["cell_types"])
    if not cell_types:
        raise ValueError("cell_types must be non-empty")
    type_counts = genome["type_counts"]
    if set(type_counts.keys()) != set(cell_types):
        raise ValueError("type_counts keys must match cell_types exactly")
    n = sum(int(v) for v in type_counts.values())
    if n < 2:
        raise ValueError(f"sum(type_counts) must be >= 2, got {n}")
    conn = genome["connection_probs"]
    syn = genome["synapse_count_params"]
    nt_type = genome.get("nt_type")
    for pre in cell_types:
        if pre not in conn:
            raise ValueError(f"connection_probs missing pre-type {pre!r}")
        if pre not in syn:
            raise ValueError(f"synapse_count_params missing pre-type {pre!r}")
        for post in cell_types:
            if post not in conn[pre]:
                raise ValueError(f"connection_probs missing [{pre!r}][{post!r}]")
            if post not in syn[pre]:
                raise ValueError(f"synapse_count_params missing [{pre!r}][{post!r}]")
            p = float(conn[pre][post])
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"connection_probs[{pre!r}][{post!r}]={p} not in [0,1]")
            params = syn[pre][post]
            if "mean" not in params or "std" not in params:
                raise ValueError(f"synapse_count_params[{pre!r}][{post!r}] must have mean and std")
    if nt_type is not None:
        if not isinstance(nt_type, Mapping):
            raise ValueError("nt_type must be a mapping when present")
        if set(nt_type.keys()) != set(cell_types):
            raise ValueError("nt_type keys must match cell_types exactly")


def sample_connectome(genome, seed=None):
    """Sample one connectome from a genome using the SBM process.
    
    Uses Bernoulli edge existence per (pre_type, post_type) block, then
    NegBinom (overdispersed) or Poisson synapse counts per edge.
    No autapses within same type.
    """
    validate_genome(genome)
    cell_types = list(genome["cell_types"])
    type_counts = {t: int(genome["type_counts"][t]) for t in cell_types}
    conn_probs = genome["connection_probs"]
    syn_params = genome["synapse_count_params"]
    resolved_seed = int(seed) if seed is not None else int(np.random.default_rng().integers(2**31))
    rng = np.random.default_rng(resolved_seed)
    type_to_ids = {}
    node_types = []
    cursor = 0
    for t in cell_types:
        count = type_counts.get(t, 0)
        type_to_ids[t] = list(range(cursor, cursor + count))
        node_types.extend([t] * count)
        cursor += count
    adjacency = []
    for pre_type in cell_types:
        pre_ids = np.asarray(type_to_ids[pre_type], dtype=np.int64)
        if pre_ids.size == 0:
            continue
        for post_type in cell_types:
            post_ids = np.asarray(type_to_ids[post_type], dtype=np.int64)
            if post_ids.size == 0:
                continue
            prob = float(conn_probs[pre_type][post_type])
            if prob <= 0.0:
                continue
            exists = rng.random((pre_ids.size, post_ids.size)) < prob
            if pre_type == post_type:
                diag = np.arange(min(pre_ids.size, post_ids.size))
                exists[diag, diag] = False
            edge_rows, edge_cols = np.where(exists)
            if edge_rows.size == 0:
                continue
            params = syn_params[pre_type][post_type]
            counts = _sample_synapse_counts(mean=float(params["mean"]), std=float(params["std"]), size=int(edge_rows.size), rng=rng)
            keep = counts > 0
            if not np.any(keep):
                continue
            for pre_id, post_id, w in zip(pre_ids[edge_rows[keep]], post_ids[edge_cols[keep]], counts[keep]):
                adjacency.append((int(pre_id), int(post_id), int(w)))
    return {
        "adjacency": adjacency,
        "cell_type_map": {idx: node_types[idx] for idx in range(len(node_types))},
        "node_types": tuple(node_types),
        "n_neurons": len(node_types),
        "genome": dict(genome),
        "seed": resolved_seed,
    }


def _sample_synapse_counts(*, mean, std, size, rng):
    if size < 1:
        return np.zeros(0, dtype=np.int64)
    if not (math.isfinite(mean) and mean > 0.0):
        raise ValueError(f"synapse-count mean must be finite and > 0, got {mean}")
    if not (math.isfinite(std) and std > 0.0):
        raise ValueError(f"synapse-count std must be finite and > 0, got {std}")
    variance = std * std
    if variance <= mean * (1.0 + 1e-9):
        return rng.poisson(mean, size=size).astype(np.int64)
    r = (mean * mean) / (variance - mean)
    p = r / (r + mean)
    if not (math.isfinite(r) and r > 0.0 and math.isfinite(p) and 0.0 < p < 1.0):
        raise ValueError(f"invalid NegBinom params from mean={mean}, std={std}: r={r}, p={p}")
    return rng.negative_binomial(r, p, size=size).astype(np.int64)
