"""Dataset-agnostic dispatch for brain-cluster training-sample builders.

``build_training_samples(symbol, objective, config)`` resolves the dataset via
``flybrain_dataset_registry``, enforces the per-dataset objective allow-list and
lifecycle status, delegates to the registered builder, and validates that the
returned payload has the shape produced by the FlyWire builder
(``{"samples": [...], "metadata": {...}}``) so training / thresholds / pipeline
accept it unchanged.

Dataset builders register themselves with ``register_sample_builder`` at import
time from a module named ``flybrain_brain_cluster_<symbol>_samples``; the
dispatcher imports that module lazily on first use, so new datasets never have
to edit this file. Builders must read local snapshots only (no network).
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.util
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import flybrain_brain_cluster_fw_samples as fw_samples
from flybrain_dataset_registry import (
    DatasetErrorCode,
    DatasetRegistryError,
    normalize_symbol,
    require_active,
    require_objective,
)

TRAINING_SAMPLE_REQUIRED_KEYS: tuple[str, ...] = (
    "sample_id",
    "region_id",
    "input_text",
    "expected_label",
    "expected_confidence",
    "provenance_refs",
    "metadata",
)
TRAINING_PAYLOAD_REQUIRED_METADATA_KEYS: tuple[str, ...] = (
    "schema_version",
    "objective",
    "source_path",
    "source_rows",
    "candidate_rows",
    "selected_count",
    "selected_regions",
    "label_counts",
    "dominant_label_share",
    "input_fingerprint",
)
SCHEMA_VERSION_RE = re.compile(r"^flybrain-[a-z0-9]+-training-samples/v[0-9]+$")
BUILDER_MODULE_TEMPLATE = "flybrain_brain_cluster_{symbol}_samples"


def training_schema_version(symbol: str, version: int = 1) -> str:
    """Canonical payload schema string, e.g. ``flybrain-banc-training-samples/v1``."""
    return f"flybrain-{normalize_symbol(symbol)}-training-samples/v{int(version)}"


class SampleBuilder(Protocol):
    def __call__(self, objective: str, config: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RegisteredBuilder:
    symbol: str
    objective: str
    builder: SampleBuilder
    config_type: type


_BUILDERS: dict[tuple[str, str], RegisteredBuilder] = {}


def register_sample_builder(
    symbol: str,
    objectives: Iterable[str],
    builder: SampleBuilder,
    *,
    config_type: type,
    replace: bool = False,
) -> None:
    """Register ``builder(objective, config) -> payload`` for each objective.

    Every objective must be in the registry's allow-list for ``symbol``.
    """
    canonical = normalize_symbol(symbol)
    objective_list = sorted(set(objectives))
    if not objective_list:
        raise DatasetRegistryError(DatasetErrorCode.OBJECTIVE_NOT_SUPPORTED, "no objectives given", symbol=canonical)
    for objective in objective_list:
        require_objective(canonical, objective)
    for objective in objective_list:
        key = (canonical, objective)
        existing = _BUILDERS.get(key)
        if existing is not None and not replace and existing.builder is not builder:
            raise DatasetRegistryError(
                DatasetErrorCode.BUILDER_ALREADY_REGISTERED,
                f"builder already registered for {canonical}/{objective}",
                symbol=canonical,
            )
    for objective in objective_list:
        _BUILDERS[(canonical, objective)] = RegisteredBuilder(canonical, objective, builder, config_type)


def unregister_sample_builder(symbol: str, objective: str) -> None:
    _BUILDERS.pop((normalize_symbol(symbol), objective), None)


def registered_builders() -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {}
    for symbol, objective in sorted(_BUILDERS):
        out.setdefault(symbol, []).append(objective)
    return {symbol: tuple(objs) for symbol, objs in out.items()}


def _lookup_builder(symbol: str, objective: str) -> RegisteredBuilder:
    key = (symbol, objective)
    if key not in _BUILDERS:
        module_name = BUILDER_MODULE_TEMPLATE.format(symbol=symbol)
        if importlib.util.find_spec(module_name) is not None:
            importlib.import_module(module_name)  # module registers itself on import
    entry = _BUILDERS.get(key)
    if entry is None:
        raise DatasetRegistryError(
            DatasetErrorCode.BUILDER_NOT_REGISTERED,
            f"no sample builder registered for {symbol}/{objective}",
            symbol=symbol,
        )
    return entry


def _default_config(config_type: type, objective: str) -> Any:
    if dataclasses.is_dataclass(config_type) and any(f.name == "objective" for f in dataclasses.fields(config_type)):
        return config_type(objective=objective)
    return config_type()


def validate_training_payload(payload: Any, *, objective: str | None = None, symbol: str | None = None) -> None:
    """Fail-closed shape check shared by every dataset builder's output.

    With ``symbol`` the payload must also be stamped for that dataset
    (``schema_version`` prefix and, when present, ``dataset_symbol``), so a
    builder cannot hand back another dataset's samples under the wrong key.
    """

    def fail(message: str) -> None:
        raise DatasetRegistryError(DatasetErrorCode.PAYLOAD_INVALID, message)

    if not isinstance(payload, Mapping):
        fail("payload must be a mapping")
    samples = payload.get("samples")
    metadata = payload.get("metadata")
    if not isinstance(samples, list) or not samples:
        fail("payload.samples must be a non-empty list")
    if not isinstance(metadata, Mapping):
        fail("payload.metadata must be a mapping")
    missing_meta = [key for key in TRAINING_PAYLOAD_REQUIRED_METADATA_KEYS if key not in metadata]
    if missing_meta:
        fail(f"payload.metadata missing keys: {', '.join(missing_meta)}")
    if not SCHEMA_VERSION_RE.match(str(metadata["schema_version"])):
        fail(f"payload.metadata.schema_version invalid: {metadata['schema_version']!r}")
    if symbol is not None:
        expected_prefix = f"flybrain-{symbol}-training-samples/"
        if not str(metadata["schema_version"]).startswith(expected_prefix):
            fail(f"payload.metadata.schema_version {metadata['schema_version']!r} is not for dataset {symbol!r}")
        if "dataset_symbol" in metadata and metadata["dataset_symbol"] != symbol:
            fail(f"payload.metadata.dataset_symbol {metadata['dataset_symbol']!r} != dispatched dataset {symbol!r}")
    if objective is not None and metadata["objective"] != objective:
        fail(f"payload objective {metadata['objective']!r} != requested {objective!r}")
    if int(metadata["selected_count"]) != len(samples):
        fail("payload.metadata.selected_count does not match len(samples)")
    seen: set[str] = set()
    label_counts: dict[str, int] = {}
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            fail(f"samples[{index}] must be a mapping")
        missing = [key for key in TRAINING_SAMPLE_REQUIRED_KEYS if key not in sample]
        if missing:
            fail(f"samples[{index}] missing keys: {', '.join(missing)}")
        sample_id = str(sample["sample_id"])
        if sample_id in seen:
            fail(f"duplicate sample_id {sample_id!r}")
        seen.add(sample_id)
        if not isinstance(sample["provenance_refs"], list) or not sample["provenance_refs"]:
            fail(f"samples[{index}].provenance_refs must be a non-empty list")
        if not isinstance(sample["metadata"], Mapping):
            fail(f"samples[{index}].metadata must be a mapping")
        confidence = float(sample["expected_confidence"])
        if not 0.0 <= confidence <= 1.0:
            fail(f"samples[{index}].expected_confidence out of [0, 1]")
        label = str(sample["expected_label"])
        label_counts[label] = label_counts.get(label, 0) + 1
    if dict(metadata["label_counts"]) != label_counts:
        fail("payload.metadata.label_counts does not match samples")


def build_training_samples(
    symbol: str,
    objective: str,
    config: Any = None,
    *,
    allow_planned: bool = False,
) -> dict[str, Any]:
    """Dispatch to the registered builder for ``symbol``/``objective``.

    Planned datasets are refused unless ``allow_planned=True`` (pre-promotion
    dry runs). ``config=None`` builds the registered config type's defaults.
    """
    canonical = normalize_symbol(symbol)
    require_active(canonical, allow_planned=allow_planned)
    require_objective(canonical, objective)
    entry = _lookup_builder(canonical, objective)
    if config is None:
        config = _default_config(entry.config_type, objective)
    if not isinstance(config, entry.config_type):
        raise DatasetRegistryError(
            DatasetErrorCode.CONFIG_TYPE_MISMATCH,
            f"{canonical} builder expects {entry.config_type.__name__}, got {type(config).__name__}",
            symbol=canonical,
        )
    config_objective = getattr(config, "objective", objective)
    if config_objective != objective:
        raise DatasetRegistryError(
            DatasetErrorCode.CONFIG_OBJECTIVE_MISMATCH,
            f"config.objective {config_objective!r} != requested objective {objective!r}",
            symbol=canonical,
        )
    payload = entry.builder(objective, config)
    validate_training_payload(payload, objective=objective, symbol=canonical)
    return payload


# ---------------------------------------------------------------------------
# Shared helpers for dataset builders (keep outputs consistent with fw).
# ---------------------------------------------------------------------------

def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sanitize_region(raw: str) -> str:
    return fw_samples._sanitize_region(raw)


def balance_and_cap_samples(samples: Sequence[dict[str, Any]], *, max_samples: int) -> list[dict[str, Any]]:
    """Deterministic round-robin across regions (sorted), identical to fw."""
    return fw_samples._balance_and_cap_samples(samples, max_samples=max_samples)


def hash_ordered_preselect(
    samples: Sequence[dict[str, Any]],
    *,
    max_samples: int,
    salt: str,
    id_of: Callable[[Mapping[str, Any]], str] | None = None,
) -> list[dict[str, Any]]:
    """Pick the cap-sized subset in sha256(salt:id) order per region, round-robin over sorted regions.

    ``balance_and_cap_samples`` takes each region's samples in ``sample_id``
    order. When sample ids embed a body id that tracks neuron size or
    proofreading order (FlyEM MaleCNS, MANC and optic-lobe body ids all do),
    a capped build keeps the lowest ids and skews the label balance (for
    example far more high_connectivity than the 25% a 0.75 quantile defines).
    This pre-selection chooses an id-order-independent subset of exactly the
    size the cap allows; passing the result through ``balance_and_cap_samples``
    then only reorders it. Deterministic and independent of input order.
    ``id_of`` defaults to ``sample_id``.
    """
    key_of = id_of or (lambda sample: str(sample["sample_id"]))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(str(sample["region_id"]), []).append(sample)

    def _order(sample: Mapping[str, Any]) -> tuple[str, str]:
        digest = hashlib.sha256(f"{salt}:{key_of(sample)}".encode("utf-8")).hexdigest()
        return digest, str(sample["sample_id"])

    for bucket in grouped.values():
        bucket.sort(key=_order)
    limit = int(max_samples)
    selected: list[dict[str, Any]] = []
    cursors = {region: 0 for region in grouped}
    regions = sorted(grouped)
    while len(selected) < limit:
        progressed = False
        for region in regions:
            if len(selected) >= limit:
                break
            if cursors[region] < len(grouped[region]):
                selected.append(grouped[region][cursors[region]])
                cursors[region] += 1
                progressed = True
        if not progressed:
            break
    return selected


def assemble_training_payload(
    samples: Sequence[dict[str, Any]],
    *,
    symbol: str,
    objective: str,
    source_path: str,
    source_rows: int,
    candidate_rows: int,
    min_distinct_labels: int,
    max_label_share: float,
    fingerprint_inputs: Mapping[str, Any],
    extra_metadata: Mapping[str, Any] | None = None,
    schema_version: int = 1,
) -> dict[str, Any]:
    """Apply fw's label-diversity/concentration gates and build the payload.

    ``samples`` should already be balanced/capped. ``fingerprint_inputs`` must
    contain every config value that affects output (objective is added).
    """
    capped = list(samples)
    if not capped:
        raise ValueError("No training samples produced after region filtering and capping.")
    label_counts: dict[str, int] = {}
    for sample in capped:
        label = str(sample["expected_label"])
        label_counts[label] = label_counts.get(label, 0) + 1
    if len(label_counts) < int(min_distinct_labels):
        raise ValueError(
            f"label diversity below minimum ({len(label_counts)} < {min_distinct_labels}); "
            "adjust objective filters or caps."
        )
    dominant_share = max(label_counts.values()) / float(len(capped))
    if dominant_share > float(max_label_share):
        raise ValueError(
            f"label concentration too high ({dominant_share:.3f} > {max_label_share:.3f}); "
            "dataset is too imbalanced for stable training."
        )
    canonical = normalize_symbol(symbol)
    fingerprint = hashlib.sha256(
        stable_json({**dict(fingerprint_inputs), "objective": objective, "symbol": canonical}).encode("utf-8")
    ).hexdigest()
    metadata: dict[str, Any] = dict(extra_metadata or {})
    metadata.update(
        {
            "schema_version": training_schema_version(canonical, schema_version),
            "dataset_symbol": canonical,
            "source_path": str(source_path),
            "objective": objective,
            "source_rows": int(source_rows),
            "candidate_rows": int(candidate_rows),
            "selected_count": int(len(capped)),
            "selected_regions": sorted({str(s["region_id"]) for s in capped}),
            "label_counts": dict(sorted(label_counts.items())),
            "min_distinct_labels": int(min_distinct_labels),
            "max_label_share": float(max_label_share),
            "dominant_label_share": float(dominant_share),
            "input_fingerprint": fingerprint,
        }
    )
    return {"samples": capped, "metadata": metadata}


# ---------------------------------------------------------------------------
# Built-in fw registration (delegates unchanged).
# ---------------------------------------------------------------------------

def _fw_builder(objective: str, config: fw_samples.FwSampleBuildConfig) -> dict[str, Any]:
    return fw_samples.build_fw_training_samples(config)


register_sample_builder(
    "fw",
    fw_samples.FW_SUPPORTED_OBJECTIVES,
    _fw_builder,
    config_type=fw_samples.FwSampleBuildConfig,
)

__all__ = [
    "BUILDER_MODULE_TEMPLATE",
    "RegisteredBuilder",
    "SampleBuilder",
    "TRAINING_PAYLOAD_REQUIRED_METADATA_KEYS",
    "TRAINING_SAMPLE_REQUIRED_KEYS",
    "assemble_training_payload",
    "balance_and_cap_samples",
    "build_training_samples",
    "register_sample_builder",
    "registered_builders",
    "sanitize_region",
    "stable_json",
    "training_schema_version",
    "unregister_sample_builder",
    "validate_training_payload",
]
