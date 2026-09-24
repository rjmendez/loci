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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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


register_sample_builder(
    L1EM_DATASET_SYMBOL,
    L1EM_SUPPORTED_OBJECTIVES,
    build_l1em_training_samples,
    config_type=L1emSampleBuildConfig,
)

__all__ = [
    "L1EM_FORBIDDEN_INPUT_FEATURES",
    "L1EM_INPUT_FEATURES",
    "L1EM_LABEL_DEFINITION",
    "L1emSampleBuildConfig",
    "build_l1em_training_samples",
    "input_features",
    "render_input_text",
]
