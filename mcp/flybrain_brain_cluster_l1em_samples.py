"""Brain-cluster training samples from the L1 larval connectome (l1em).

Objective ``connectivity_tier`` only. One sample per annotated neuron in the
Winding et al. 2023 all-all matrix. Label: ``high_connectivity`` when the
neuron's total synapse count (all-all row sum + column sum) is at or above the
configured quantile of the candidate set, else ``baseline_connectivity``.

Label hygiene: the model input (``input_text``) is built only from
``L1EM_INPUT_FEATURES`` (cell type, modality annotation, hemisphere, pairing,
level-7 cluster and two scale-free axon/dendrite fractions). Synapse counts,
in/out totals, raw compartment counts and partner counts are in
``L1EM_FORBIDDEN_INPUT_FEATURES`` and never reach the model input. See
``docs/FLYBRAIN_L1EM_ADAPTER_CONTRACT.md``.

Unsupported here (not registered): ``neurotransmitter_dominance`` (no NT
annotations in the release) and ``region_specialization_tier`` (no larval
per-synapse neuropil assignment). Region ids are larval cell types
(``l1_<celltype>``), never adult neuropils.

Structured features (real models): ``load_l1em_matrices``, ``export_l1em_edge_table``,
``l1em_node_table``, ``l1em_wiring_features`` and ``structured_features`` give the
feature learners in ``flybrain_l1em_targets`` a numeric per-neuron view. They
do not change the legacy text builder or its payload.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from flybrain_brain_cluster_samples import (
    assemble_training_payload,
    balance_and_cap_samples,
    register_sample_builder,
)
from flybrain_l1em_adapter import (
    L1EM_DATASET_SYMBOL,
    L1EM_STAGE,
    L1EM_SUPPORTED_OBJECTIVES,
    L1EM_VERSION_ID,
    L1emAdapterError,
    L1emErrorCode,
    L1emNeuron,
    assert_larval_region,
    load_l1em_snapshot,
    require_supported_objective,
)

L1EM_INPUT_FEATURES: tuple[str, ...] = (
    "celltype",
    "annotation",
    "hemisphere",
    "paired",
    "cluster",
    "axon_output_fraction",
    "axon_input_fraction",
)
L1EM_FORBIDDEN_INPUT_FEATURES: frozenset[str] = frozenset(
    {
        "total_synapses",
        "in_synapses",
        "out_synapses",
        "axon_output",
        "dendrite_output",
        "axon_input",
        "dendrite_input",
        "partner_count",
        "skid",
    }
)
L1EM_LABEL_HIGH = "high_connectivity"
L1EM_LABEL_BASELINE = "baseline_connectivity"
L1EM_LABEL_DEFINITION = (
    "high_connectivity iff (all-all row sum + column sum) >= quantile(q) over candidate neurons; "
    "else baseline_connectivity"
)
if set(L1EM_INPUT_FEATURES) & L1EM_FORBIDDEN_INPUT_FEATURES:  # pragma: no cover - import-time guard
    raise RuntimeError("l1em input features overlap the forbidden (label-derivable) set")


@dataclass(frozen=True)
class L1emSampleBuildConfig:
    objective: str = "connectivity_tier"
    storage_root: str | Path | None = None
    snapshot_root: str | Path | None = None
    verify_integrity: bool = True
    max_samples: int = 5000
    min_total_synapses: int = 10
    min_region_samples: int = 10
    high_connectivity_quantile: float = 0.75
    max_regions: int = 20
    min_distinct_labels: int = 2
    max_label_share: float = 0.9


def _validate_config(config: L1emSampleBuildConfig) -> None:
    require_supported_objective(config.objective)
    if config.max_samples <= 0:
        raise ValueError("max_samples must be > 0")
    if config.min_total_synapses < 1:
        raise ValueError("min_total_synapses must be >= 1")
    if config.min_region_samples < 1:
        raise ValueError("min_region_samples must be >= 1")
    if not (0.0 < config.high_connectivity_quantile < 1.0):
        raise ValueError("high_connectivity_quantile must be in (0, 1)")
    if config.max_regions < 1:
        raise ValueError("max_regions must be >= 1")
    if config.min_distinct_labels < 1:
        raise ValueError("min_distinct_labels must be >= 1")
    if not (0.0 < config.max_label_share <= 1.0):
        raise ValueError("max_label_share must be in (0, 1]")


def input_features(neuron: L1emNeuron) -> dict[str, str]:
    """The only neuron fields allowed into ``input_text`` (see leakage guard)."""
    return {
        "celltype": neuron.celltype,
        "annotation": neuron.annotation,
        "hemisphere": neuron.hemisphere,
        "paired": "yes" if neuron.paired else "no",
        "cluster": neuron.cluster,
        "axon_output_fraction": _fmt_fraction(neuron.axon_output_fraction),
        "axon_input_fraction": _fmt_fraction(neuron.axon_input_fraction),
    }


def render_input_text(features: Mapping[str, str]) -> str:
    keys = tuple(features)
    if keys != L1EM_INPUT_FEATURES:
        raise L1emAdapterError(
            L1emErrorCode.SCHEMA_MISMATCH,
            "input features must be exactly L1EM_INPUT_FEATURES (label-leakage guard)",
            {"got": list(keys), "expected": list(L1EM_INPUT_FEATURES)},
        )
    body = " ".join(f"{key} {_token(value)}" for key, value in features.items())
    return f"dataset {L1EM_VERSION_ID} stage {L1EM_STAGE} {body}"


def _fmt_fraction(value: float | None) -> str:
    return "na" if value is None else f"{float(value):.4f}"


def _token(value: str) -> str:
    return "_".join(str(value).split()) or "na"


def _quantile(values: list[int], q: float) -> float:
    """Linear-interpolated quantile (matches pandas/numpy default)."""
    ordered = sorted(float(v) for v in values)
    if not ordered:
        raise ValueError("cannot compute quantile of an empty set")
    position = (len(ordered) - 1) * float(q)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _row_to_sample(neuron: L1emNeuron, *, threshold: float, manifest_sha256: str, source_name: str) -> dict[str, Any]:
    total = neuron.total_synapses
    label = L1EM_LABEL_HIGH if float(total) >= threshold else L1EM_LABEL_BASELINE
    score_delta = abs(float(total) - threshold) / max(float(total), threshold, 1.0)
    confidence = min(0.99, max(0.5, 0.55 + (0.4 * score_delta)))
    region_id = assert_larval_region(neuron.region_id)
    return {
        "sample_id": f"l1em-skid-{neuron.skid}",
        "region_id": region_id,
        "input_text": render_input_text(input_features(neuron)),
        "expected_label": label,
        "expected_confidence": round(confidence, 6),
        "provenance_refs": [
            f"{L1EM_VERSION_ID}:skid:{neuron.skid}",
            f"source:{source_name}",
            f"manifest_sha256:{manifest_sha256}",
        ],
        "metadata": {
            "dataset": L1EM_DATASET_SYMBOL,
            "dataset_version": L1EM_VERSION_ID,
            "stage": L1EM_STAGE,
            "skid": str(neuron.skid),
            "celltype": neuron.celltype,
            "hemisphere": neuron.hemisphere,
            "pair_skid": None if neuron.pair_skid is None else str(neuron.pair_skid),
            # Homologous left/right pairs share annotations; split on this key.
            "split_group": f"l1em-pair-{min(neuron.skid, neuron.pair_skid or neuron.skid)}",
            "total_synapses": int(total),
            "in_synapses": int(neuron.in_synapses),
            "out_synapses": int(neuron.out_synapses),
            "task_type": "connectivity",
            "risk_tier": "high" if label == L1EM_LABEL_HIGH else "medium",
            "cross_stage_identity_transfer": "unsupported",
        },
    }


def build_l1em_training_samples(objective: str, config: L1emSampleBuildConfig) -> dict[str, Any]:
    if objective != config.objective:
        raise ValueError(f"objective {objective!r} != config.objective {config.objective!r}")
    _validate_config(config)
    snapshot = load_l1em_snapshot(
        config.storage_root, snapshot_root=config.snapshot_root, verify_integrity=config.verify_integrity
    )
    candidates = [n for n in snapshot.neurons if n.total_synapses >= int(config.min_total_synapses)]
    if not candidates:
        raise ValueError("No annotated l1em neurons remain after min_total_synapses filtering.")
    threshold = _quantile([n.total_synapses for n in candidates], config.high_connectivity_quantile)

    region_counts: dict[str, int] = {}
    for neuron in candidates:
        region_counts[neuron.region_id] = region_counts.get(neuron.region_id, 0) + 1
    eligible = [r for r, c in region_counts.items() if c >= int(config.min_region_samples)]
    if not eligible:
        raise ValueError("No regions satisfy min_region_samples. Lower min_region_samples or inspect source.")
    ranked = sorted(eligible, key=lambda r: (-region_counts[r], r))[: int(config.max_regions)]
    keep = set(ranked)
    selected = sorted((n for n in candidates if n.region_id in keep), key=lambda n: (n.region_id, n.skid))

    source_path = snapshot.files["all_all_matrix"]
    rows = [
        _row_to_sample(n, threshold=threshold, manifest_sha256=snapshot.manifest_sha256, source_name=source_path.name)
        for n in selected
    ]
    capped = balance_and_cap_samples(rows, max_samples=int(config.max_samples))
    return assemble_training_payload(
        capped,
        symbol=L1EM_DATASET_SYMBOL,
        objective=objective,
        source_path=str(source_path),
        source_rows=snapshot.matrix_rows,
        candidate_rows=len(candidates),
        min_distinct_labels=config.min_distinct_labels,
        max_label_share=config.max_label_share,
        fingerprint_inputs={
            "manifest_sha256": snapshot.manifest_sha256,
            "max_samples": config.max_samples,
            "min_total_synapses": config.min_total_synapses,
            "min_region_samples": config.min_region_samples,
            "high_connectivity_quantile": config.high_connectivity_quantile,
            "max_regions": config.max_regions,
            "min_distinct_labels": config.min_distinct_labels,
            "max_label_share": config.max_label_share,
            "input_features": list(L1EM_INPUT_FEATURES),
        },
        extra_metadata={
            "dataset_version": L1EM_VERSION_ID,
            "stage": L1EM_STAGE,
            "region_vocabulary": "catmaid_annotation",
            "cross_stage_identity_transfer": "unsupported",
            "manifest_id": snapshot.manifest_id,
            "manifest_sha256": snapshot.manifest_sha256,
            "license": snapshot.license_spdx,
            "citation": snapshot.citation,
            "annotation_rows": snapshot.annotation_rows,
            "unannotated_neurons_excluded": len(snapshot.unannotated_skids),
            "high_connectivity_threshold": float(threshold),
            "label_definition": L1EM_LABEL_DEFINITION,
            "input_features": list(L1EM_INPUT_FEATURES),
            "forbidden_input_features": sorted(L1EM_FORBIDDEN_INPUT_FEATURES),
            "min_total_synapses": int(config.min_total_synapses),
            "min_region_samples": int(config.min_region_samples),
            "max_samples": int(config.max_samples),
            "max_regions": int(config.max_regions),
        },
    )


# ---------------------------------------------------------------------------
# Structured (numeric) features for the real-model evaluation.
#
# The legacy text builder above is unchanged. Everything below feeds the
# feature learners run by ``flybrain_l1em_targets`` through
# ``flybrain_model_eval``: a node table, the all-all edge list exported to the
# feature cache, wiring features from ``flybrain_wiring_features`` and three
# l1em-only families. No function here puts a skid, a synapse total or an S2
# annotation column into a feature; per-objective label exclusions are applied
# by the wiring-feature and eval modules on top of that.
#
# Families (column ``<family>__<name>``):
#   * ``etype_out`` / ``etype_in``: fraction of the neuron's output / input
#     synapses that are axo-axonic (aa), axo-dendritic (ad), dendro-axonic (da)
#     or dendro-dendritic (dd), from the four split matrices. Scale free.
#   * ``cmpt``: axon share of output and of input from ``outputs.csv`` /
#     ``inputs.csv`` (the scale-free fractions the legacy text already used).
#   * ``ase``: adjacency spectral embedding of log1p(all-all), ``dim`` output and
#     ``dim`` input coordinates. Label free (no annotation is read), computed on
#     the whole graph. Its norm tracks degree, so connectivity_tier never uses it.
# ---------------------------------------------------------------------------

L1EM_EDGE_TYPES: tuple[str, ...] = ("aa", "ad", "da", "dd")
L1EM_EDGE_TYPE_FILES: Mapping[str, str] = {t: f"metadata/files/{t}_connectivity_matrix.csv" for t in L1EM_EDGE_TYPES}
L1EM_STRUCTURED_SCHEMA_VERSION = "flybrain-l1em-structured/v1"
L1EM_DEFAULT_CACHE_ROOT = "/mnt/f/.flybrain/cache"
L1EM_ASE_DIM = 8
L1EM_STRUCTURED_FAMILIES: tuple[str, ...] = ("etype", "cmpt", "ase")


@dataclass(frozen=True)
class L1emMatrices:
    """The all-all matrix (rows presynaptic) and its aa/ad/da/dd split, one skid order."""

    skids: tuple[int, ...]
    all_all: Any  # numpy (n, n) float64
    by_type: Mapping[str, Any]
    manifest_sha256: str

    def index_of(self) -> dict[int, int]:
        return {skid: i for i, skid in enumerate(self.skids)}


def _refuse_snapshot_path(path: Path) -> None:
    if "snapshots" in Path(path).resolve(strict=False).parts:
        raise ValueError(f"refusing to write l1em cache files under a snapshots directory: {path}")


def _listed_manifest_paths(snapshot: Any) -> set[str]:
    """Relative paths the (already validated) manifest lists; re-hashed to match the snapshot."""
    from flybrain_l1em_adapter import manifest_digest

    manifest = json.loads(Path(snapshot.manifest_path).read_text(encoding="utf-8"))
    if manifest_digest(manifest) != snapshot.manifest_sha256:
        raise L1emAdapterError(
            L1emErrorCode.INTEGRITY_MISMATCH, "manifest changed after the snapshot was validated",
            {"manifest_path": str(snapshot.manifest_path)},
        )
    return {str(e.get("relative_path") or "").replace("\\", "/").casefold() for e in manifest["integrity"]["files"]}


def _read_square_matrix(path: Path, *, order: Sequence[int] | None = None) -> tuple[tuple[int, ...], Any]:
    import numpy as np
    import pandas as pd

    try:
        frame = pd.read_csv(path, index_col=0)
        rows = [int(v) for v in frame.index]
        cols = [int(str(v)) for v in frame.columns]
    except (OSError, TypeError, ValueError) as exc:
        raise L1emAdapterError(L1emErrorCode.MATRIX_INVALID, f"matrix unreadable: {exc}", {"path": str(path)}) from exc
    if not rows or rows != cols or len(set(rows)) != len(rows):
        raise L1emAdapterError(
            L1emErrorCode.MATRIX_INVALID, "matrix must be square with identical, unique row/column skids",
            {"path": str(path)},
        )
    if order is not None and tuple(rows) != tuple(order):
        if set(rows) != set(order):
            raise L1emAdapterError(
                L1emErrorCode.MATRIX_INVALID, "split matrix covers different skids than all-all", {"path": str(path)}
            )
        frame.index = rows
        frame.columns = cols
        frame = frame.loc[list(order), list(order)]
        rows = list(order)
    values = frame.to_numpy(dtype="float64")
    if not np.isfinite(values).all() or (values < 0).any() or (np.mod(values, 1.0) != 0).any():
        raise L1emAdapterError(
            L1emErrorCode.MATRIX_INVALID, "matrix values must be finite non-negative integers", {"path": str(path)}
        )
    return tuple(rows), values


def load_l1em_matrices(snapshot: Any, *, require_edge_types: bool = True) -> L1emMatrices:
    """Read all-all (+ the aa/ad/da/dd split) from a validated snapshot; fail closed on any mismatch.

    The split files must be listed in the snapshot manifest (so the adapter has
    size- and, with ``verify_integrity``, sha256-checked them) and must sum to
    all-all exactly.
    """
    import numpy as np

    skids, all_all = _read_square_matrix(Path(snapshot.files["all_all_matrix"]))
    by_type: dict[str, Any] = {}
    if require_edge_types:
        listed = _listed_manifest_paths(snapshot)
        for edge_type, rel in L1EM_EDGE_TYPE_FILES.items():
            if rel.casefold() not in listed:
                raise L1emAdapterError(
                    L1emErrorCode.REQUIRED_FILE_MISSING, f"manifest does not list the {edge_type} split matrix",
                    {"relative_path": rel},
                )
            _, by_type[edge_type] = _read_square_matrix(Path(snapshot.snapshot_root) / rel, order=skids)
        total = sum(by_type.values())
        if not np.array_equal(total, all_all):
            raise L1emAdapterError(
                L1emErrorCode.MATRIX_INVALID, "aa + ad + da + dd does not equal the all-all matrix",
                {"max_abs_diff": float(np.abs(total - all_all).max())},
            )
    return L1emMatrices(skids=skids, all_all=all_all, by_type=by_type, manifest_sha256=snapshot.manifest_sha256)


def export_l1em_edge_table(matrices: L1emMatrices, *, cache_root: str | Path = L1EM_DEFAULT_CACHE_ROOT) -> Any:
    """Write the non-zero all-all entries as ``(pre, post, weight)`` parquet under the cache; return an EdgeSource.

    The file is keyed by the manifest sha256 and reused (sidecar sha256
    verified) so the wiring-feature fingerprint stays stable across runs.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from flybrain_wiring_features import EdgeSource

    cache_dir = Path(cache_root) / "l1em" / "edges"
    _refuse_snapshot_path(cache_dir)
    skid_digest = hashlib.sha256(",".join(str(s) for s in matrices.skids).encode("utf-8")).hexdigest()
    key = hashlib.sha256(f"{matrices.manifest_sha256}:{skid_digest}".encode("utf-8")).hexdigest()[:24]
    path = cache_dir / f"all_all-{key}.parquet"
    sidecar = path.with_suffix(".json")
    digest = None
    if path.is_file() and sidecar.is_file():
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        if meta.get("parquet_sha256") == _sha256_path(path) and meta.get("manifest_sha256") == matrices.manifest_sha256:
            digest = meta["parquet_sha256"]
    if digest is None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        pre_i, post_i = np.nonzero(matrices.all_all)
        skid_arr = np.asarray(matrices.skids, dtype=np.int64)
        table = pa.table({
            "pre": pa.array(skid_arr[pre_i], type=pa.int64()),
            "post": pa.array(skid_arr[post_i], type=pa.int64()),
            "weight": pa.array(matrices.all_all[pre_i, post_i], type=pa.float64()),
        })
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp)
        os.replace(tmp, path)
        digest = _sha256_path(path)
        sidecar.write_text(json.dumps({
            "schema_version": L1EM_STRUCTURED_SCHEMA_VERSION,
            "manifest_sha256": matrices.manifest_sha256,
            "source": "all-all_connectivity_matrix.csv",
            "rows": int(table.num_rows),
            "parquet_sha256": digest,
        }, indent=2, sort_keys=True), encoding="utf-8")
    return EdgeSource(
        path=str(path), pre="pre", post="post", weight="weight", unique_pairs=True, format="parquet",
        provenance={"dataset": L1EM_DATASET_SYMBOL, "manifest_sha256": matrices.manifest_sha256,
                    "source": "all-all_connectivity_matrix.csv", "edge_table_sha256": digest},
    )


def l1em_node_table(snapshot: Any, matrices: L1emMatrices | None = None) -> Any:
    """One row per annotated neuron that is in the matrix (metadata only; never a feature table)."""
    import pandas as pd

    present = None if matrices is None else set(matrices.skids)
    rows = []
    for n in snapshot.neurons:
        if present is not None and n.skid not in present:
            continue
        rows.append({
            "skid": int(n.skid),
            "node_id": str(n.skid),
            "sample_id": f"l1em-skid-{n.skid}",
            "celltype": n.celltype,
            "annotation": n.annotation,
            "hemisphere": n.hemisphere,
            "pair_skid": None if n.pair_skid is None else int(n.pair_skid),
            "level_7_cluster": n.cluster,
            "split_group": f"l1em-pair-{min(n.skid, n.pair_skid or n.skid)}",
            "total_synapses": int(n.total_synapses),
            "in_synapses": int(n.in_synapses),
            "out_synapses": int(n.out_synapses),
            "axon_output_fraction": n.axon_output_fraction,
            "axon_input_fraction": n.axon_input_fraction,
        })
    if not rows:
        raise ValueError("no annotated l1em neurons in the matrix")
    return pd.DataFrame(rows).sort_values("skid", kind="stable").reset_index(drop=True)


def edge_type_features(matrices: L1emMatrices, skids: Sequence[int]) -> Any:
    """``etype_out__<t>`` / ``etype_in__<t>``: share of output / input synapses of each connection type."""
    import numpy as np
    import pandas as pd

    if not matrices.by_type:
        raise ValueError("edge-type features need the aa/ad/da/dd matrices (load with require_edge_types=True)")
    index = matrices.index_of()
    rows = np.asarray([index[int(s)] for s in skids], dtype=np.int64)
    out_total = matrices.all_all.sum(axis=1)[rows]
    in_total = matrices.all_all.sum(axis=0)[rows]
    features: dict[str, Any] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for edge_type in L1EM_EDGE_TYPES:
            mat = matrices.by_type[edge_type]
            features[f"etype_out__{edge_type}"] = np.where(out_total > 0, mat.sum(axis=1)[rows] / out_total, np.nan)
            features[f"etype_in__{edge_type}"] = np.where(in_total > 0, mat.sum(axis=0)[rows] / in_total, np.nan)
    return pd.DataFrame(features, index=pd.Index([int(s) for s in skids], name="skid"))


def compartment_features(nodes: Any) -> Any:
    """``cmpt__axon_output_share`` / ``cmpt__axon_input_share`` from the node table's scale-free fractions."""
    import pandas as pd

    frame = pd.DataFrame({
        "cmpt__axon_output_share": pd.to_numeric(nodes["axon_output_fraction"], errors="coerce").to_numpy(),
        "cmpt__axon_input_share": pd.to_numeric(nodes["axon_input_fraction"], errors="coerce").to_numpy(),
    }, index=pd.Index(nodes["skid"].astype(int).tolist(), name="skid"))
    return frame


def spectral_embedding_features(matrices: L1emMatrices, skids: Sequence[int], *, dim: int = L1EM_ASE_DIM) -> Any:
    """Adjacency spectral embedding of log1p(all-all): ``ase__out_kk`` (left) and ``ase__in_kk`` (right).

    Uses no annotation at all. Deterministic: truncated ARPACK SVD of the
    sparse matrix with a fixed start vector (BLAS capped at 8 threads), and
    each component's sign is fixed so its largest-magnitude left coordinate
    is positive.
    """
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from scipy.sparse.linalg import svds
    from threadpoolctl import threadpool_limits

    n = len(matrices.skids)
    k = min(int(dim), n - 1)
    if k < 1:
        raise ValueError("dim must be >= 1 and the matrix must have at least 2 nodes")
    weights = sparse.csr_matrix(np.log1p(matrices.all_all))
    with threadpool_limits(limits=8):
        u, s, vt = svds(weights, k=k, v0=np.full(n, 1.0 / np.sqrt(n)), solver="arpack")
    order = np.argsort(-s, kind="stable")
    u, s, v = u[:, order], s[order], vt[order].T
    signs = np.sign(u[np.argmax(np.abs(u), axis=0), np.arange(k)])
    signs[signs == 0] = 1.0
    scale = np.sqrt(s) * signs
    out_emb, in_emb = u * scale, v * scale
    index = matrices.index_of()
    rows = np.asarray([index[int(x)] for x in skids], dtype=np.int64)
    features = {f"ase__out_{j:02d}": out_emb[rows, j] for j in range(k)}
    features.update({f"ase__in_{j:02d}": in_emb[rows, j] for j in range(k)})
    return pd.DataFrame(features, index=pd.Index([int(x) for x in skids], name="skid"))


def structured_features(nodes: Any, matrices: L1emMatrices, *, families: Sequence[str] = L1EM_STRUCTURED_FAMILIES,
                        ase_dim: int = L1EM_ASE_DIM) -> Any:
    """Concatenate the requested l1em-only families for ``nodes`` (index = skid, node-table order)."""
    import pandas as pd

    unknown = set(families) - set(L1EM_STRUCTURED_FAMILIES)
    if unknown:
        raise ValueError(f"unknown structured families: {sorted(unknown)}")
    skids = nodes["skid"].astype(int).tolist()
    parts = []
    if "etype" in families:
        parts.append(edge_type_features(matrices, skids))
    if "cmpt" in families:
        parts.append(compartment_features(nodes))
    if "ase" in families:
        parts.append(spectral_embedding_features(matrices, skids, dim=ase_dim))
    if not parts:
        return pd.DataFrame(index=pd.Index(skids, name="skid"))
    return pd.concat(parts, axis=1).loc[skids]


def l1em_wiring_features(*, objective: str, edges: Any, nodes: Any, category_column: str, two_hop: bool = False,
                         mask_skids: Sequence[int] = (), cache_root: str | Path | None = L1EM_DEFAULT_CACHE_ROOT,
                         use_cache: bool = True) -> Any:
    """``flybrain_wiring_features.build_wiring_features`` over the l1em edge table.

    Partner categories come from ``nodes[category_column]``; ``mask_skids``
    hides the category of those nodes (pass the val + test skids whenever the
    category is, or determines, the target). ``objective`` exclusions are
    applied by the foundation before the frame is returned.
    """
    from flybrain_wiring_features import WiringFeatureParams, build_wiring_features

    table = nodes[["node_id", category_column]].copy()
    return build_wiring_features(
        dataset=L1EM_DATASET_SYMBOL, objective=objective, edges=edges, nodes=table, id_column="node_id",
        category_column=category_column,
        params=WiringFeatureParams(top_k_neuropils=0, reciprocity=True, two_hop=bool(two_hop),
                                   category_tag=""),
        cache_root=cache_root, use_cache=use_cache,
        mask_category_ids=[str(int(s)) for s in mask_skids] or None,
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

register_sample_builder(
    L1EM_DATASET_SYMBOL,
    L1EM_SUPPORTED_OBJECTIVES,
    build_l1em_training_samples,
    config_type=L1emSampleBuildConfig,
)

__all__ = [
    "L1EM_ASE_DIM",
    "L1EM_EDGE_TYPES",
    "L1EM_EDGE_TYPE_FILES",
    "L1EM_FORBIDDEN_INPUT_FEATURES",
    "L1EM_INPUT_FEATURES",
    "L1EM_LABEL_DEFINITION",
    "L1EM_STRUCTURED_FAMILIES",
    "L1emMatrices",
    "L1emSampleBuildConfig",
    "build_l1em_training_samples",
    "compartment_features",
    "edge_type_features",
    "export_l1em_edge_table",
    "input_features",
    "l1em_node_table",
    "l1em_wiring_features",
    "load_l1em_matrices",
    "render_input_text",
    "spectral_embedding_features",
    "structured_features",
]
