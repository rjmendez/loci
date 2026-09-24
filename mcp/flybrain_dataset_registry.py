"""Typed, static registry of FlyBrain dataset symbols.

The registry is the single source of truth for:

* the canonical (lowercase) dataset symbol used in code, CLIs and payloads,
* the on-disk snapshot directory name (which may differ in case, e.g. ``BANC``),
* the pinned snapshot version directory,
* organism stage and region vocabulary type,
* the per-dataset objective allow-list consumed by sample builders,
* lifecycle status (``active`` or ``planned``),
* the sample-metadata keys that group related neurons for train/validation/test
  splitting (``split_group_keys``), and the citation each licence requires.

It performs no I/O at import time and never touches the network. Directory
names on disk are normalized here; they are never renamed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from flybrain_harness_storage import build_flybrain_harness_layout

OBJECTIVE_CONNECTIVITY_TIER = "connectivity_tier"
OBJECTIVE_NEUROTRANSMITTER_DOMINANCE = "neurotransmitter_dominance"
OBJECTIVE_REGION_SPECIALIZATION_TIER = "region_specialization_tier"

KNOWN_OBJECTIVES: tuple[str, ...] = (
    OBJECTIVE_CONNECTIVITY_TIER,
    OBJECTIVE_NEUROTRANSMITTER_DOMINANCE,
    OBJECTIVE_REGION_SPECIALIZATION_TIER,
)

LICENSE_UNREVIEWED = "UNREVIEWED"
LICENSE_CC_BY_4_0 = "CC-BY-4.0"
# Symbols that must never be promoted to ACTIVE (see _validate_registry).
NEVER_ACTIVE_SYMBOLS: frozenset[str] = frozenset({"fafb"})


class DatasetErrorCode(str, Enum):
    UNKNOWN_DATASET = "UNKNOWN_DATASET"
    UNKNOWN_OBJECTIVE = "UNKNOWN_OBJECTIVE"
    OBJECTIVE_NOT_SUPPORTED = "OBJECTIVE_NOT_SUPPORTED"
    DATASET_NOT_ACTIVE = "DATASET_NOT_ACTIVE"
    VERSION_UNPINNED = "VERSION_UNPINNED"
    BUILDER_NOT_REGISTERED = "BUILDER_NOT_REGISTERED"
    BUILDER_ALREADY_REGISTERED = "BUILDER_ALREADY_REGISTERED"
    CONFIG_TYPE_MISMATCH = "CONFIG_TYPE_MISMATCH"
    CONFIG_OBJECTIVE_MISMATCH = "CONFIG_OBJECTIVE_MISMATCH"
    PAYLOAD_INVALID = "PAYLOAD_INVALID"


class DatasetRegistryError(ValueError):
    """Fail-closed registry/dispatch error with a stable ``code``.

    Subclasses ``ValueError`` so existing CLI ``except ValueError`` paths keep
    reporting it as a structured error.
    """

    def __init__(self, code: DatasetErrorCode, message: str, *, symbol: str | None = None) -> None:
        self.code = code
        self.symbol = symbol
        super().__init__(f"[{code.value}] {message}")


class DatasetStatus(str, Enum):
    ACTIVE = "active"
    PLANNED = "planned"


class OrganismStage(str, Enum):
    ADULT = "adult"
    LARVAL = "larval"


class RegionVocabulary(str, Enum):
    FLYWIRE_NEUROPIL = "flywire_neuropil"
    HEMIBRAIN_ROI = "hemibrain_roi"
    BANC_NEUROPIL = "banc_neuropil"
    CATMAID_ANNOTATION = "catmaid_annotation"
    MANC_NEUROPIL = "manc_neuropil"
    OPTIC_LOBE_NEUROPIL = "optic_lobe_neuropil"
    MALE_CNS_ROI = "male_cns_roi"


@dataclass(frozen=True)
class DatasetSpec:
    symbol: str
    snapshot_dir_name: str
    pinned_version: str | None
    organism_stage: OrganismStage
    region_vocabulary: RegionVocabulary
    license: str
    supported_objectives: tuple[str, ...]
    status: DatasetStatus
    description: str = ""
    # Sample-metadata keys whose shared (known) values tie neurons together for
    # train/validation/test splitting. The split unions every listed key, so no
    # value of any key straddles two splits.
    split_group_keys: tuple[str, ...] = ()
    citation: str = ""

    @property
    def is_active(self) -> bool:
        return self.status is DatasetStatus.ACTIVE

    def supports(self, objective: str) -> bool:
        return objective in self.supported_objectives

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "snapshot_dir_name": self.snapshot_dir_name,
            "pinned_version": self.pinned_version,
            "organism_stage": self.organism_stage.value,
            "region_vocabulary": self.region_vocabulary.value,
            "license": self.license,
            "supported_objectives": list(self.supported_objectives),
            "status": self.status.value,
            "description": self.description,
            "split_group_keys": list(self.split_group_keys),
            "citation": self.citation,
        }


_C = OBJECTIVE_CONNECTIVITY_TIER
_N = OBJECTIVE_NEUROTRANSMITTER_DOMINANCE
_R = OBJECTIVE_REGION_SPECIALIZATION_TIER

# NOTE: fw's objective set is consumed by flybrain_brain_cluster_fw_samples;
# adding an objective here widens what the fw builder accepts, so only add one
# together with a matching fw builder branch.
_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec("fw", "fw", "flywire783", OrganismStage.ADULT, RegionVocabulary.FLYWIRE_NEUROPIL,
                LICENSE_UNREVIEWED, (_C, _N), DatasetStatus.ACTIVE,
                "FlyWire FAFB whole-brain v783 (metadata tables).",
                # The fw builder does not emit these keys yet, so fw splits stay
                # per-neuron until it reads the Schlegel cell-type annotations.
                split_group_keys=("cell_type", "hemilineage")),
    DatasetSpec("hb", "hb", "neuprint_JRC_Hemibrain_1point2point1", OrganismStage.ADULT,
                RegionVocabulary.HEMIBRAIN_ROI, LICENSE_UNREVIEWED, (_C, _R), DatasetStatus.ACTIVE,
                "Janelia hemibrain v1.2.1 via local neuPrint export.",
                split_group_keys=("cell_type",)),
    # banc/l1em/mc/mv/ol licences were verified at pull time (evidence in each
    # snapshot's manifest). All stay PLANNED until the promotion gates (SC3
    # threshold reports per (dataset, objective), grouped-split trivial-baseline
    # gate, AB AC1-AC5) pass; until then dispatch needs allow_planned=True.
    DatasetSpec("banc", "BANC", "banc_888", OrganismStage.ADULT, RegionVocabulary.BANC_NEUROPIL,
                LICENSE_CC_BY_4_0, (_C, _N, _R), DatasetStatus.PLANNED,
                "Brain and nerve cord (BANC) materialization 888; dir is uppercase on disk. "
                "Builders: connectivity_tier, neurotransmitter_dominance "
                "(region_specialization_tier has no builder and fails closed).",
                split_group_keys=("cell_type", "hemilineage"),
                citation="Bates AS, Phelps JS, Kim M, Yang HHJ, et al. (2026). Distributed control circuits "
                         "across a brain-and-cord connectome. Nature. doi:10.1038/s41586-026-10735-w. "
                         "Data: Harvard Dataverse doi:10.7910/DVN/7WTH1N (CC BY 4.0)."),
    DatasetSpec("l1em", "l1em", "catmaid_l1em", OrganismStage.LARVAL, RegionVocabulary.CATMAID_ANNOTATION,
                LICENSE_CC_BY_4_0, (_C,), DatasetStatus.PLANNED,
                "L1 larval connectome (Winding et al. 2023, Europe PMC supplementary data). "
                "connectivity_tier only: no NT annotations and no per-synapse larval neuropil.",
                # split_group is the l1em builder's left/right homolog pair key.
                split_group_keys=("split_group",)),
    # fafb is a redundant alias of fw (same FAFB EM volume). It must never be
    # promoted; an empty objective list makes every dispatch fail closed.
    DatasetSpec("fafb", "fafb", "catmaid_fafb", OrganismStage.ADULT, RegionVocabulary.CATMAID_ANNOTATION,
                LICENSE_UNREVIEWED, (), DatasetStatus.PLANNED,
                "Deferred: redundant alias of fw (same FAFB EM volume; sparse manually traced CATMAID "
                "subset, no license-clear bulk export). See docs/FLYBRAIN_FAFB_LANE_DECISION.md."),
    # mc / mv / ol match the snapshots pulled 2026-09-24 (manifests under
    # snapshots/<symbol>/<version>/manifest/). mc and ol image the same male
    # specimen; ol stays its own pinned version and is never merged into mc
    # rows without an explicit bodyId crosswalk.
    DatasetSpec("mc", "mc", "male-cns_v1.0", OrganismStage.ADULT, RegionVocabulary.MALE_CNS_ROI,
                LICENSE_CC_BY_4_0, (_C, _N, _R), DatasetStatus.PLANNED,
                "Janelia FlyEM male adult CNS (MaleCNS, neuPrint male-cns:v1.0, released 2026-06-08): "
                "brain + ventral nerve cord of one male. Builders: connectivity_tier, neurotransmitter_dominance, "
                "region_specialization_tier (docs/FLYBRAIN_MC_ADAPTER_CONTRACT.md).",
                split_group_keys=("cell_type", "hemilineage"),
                citation="Berg S, Beckett IR, Costa M, Schlegel P, et al. (2026). Sexual dimorphism in the "
                         "complete connectome of the Drosophila male central nervous system. Cell "
                         "(preprint: bioRxiv doi:10.1101/2025.10.09.680999). Data: FlyEM MaleCNS v1.0, "
                         "gs://flyem-male-cns/v1.0 (CC BY 4.0)."),
    DatasetSpec("mv", "mv", "manc_v1.0", OrganismStage.ADULT, RegionVocabulary.MANC_NEUROPIL,
                LICENSE_CC_BY_4_0, (_C, _N, _R), DatasetStatus.PLANNED,
                "Janelia FlyEM Male Adult Nerve Cord (MANC) v1.0 flat exports: male ventral nerve cord "
                "only (v1.2.x is neuPrint-API only and was not pulled). Builders: connectivity_tier, "
                "neurotransmitter_dominance, region_specialization_tier (docs/FLYBRAIN_MV_ADAPTER_CONTRACT.md).",
                split_group_keys=("cell_type", "hemilineage"),
                citation="Takemura S, Hayworth KJ, Huang GB, et al. (2024). A connectome of the male "
                         "Drosophila ventral nerve cord. eLife 13:RP97769. doi:10.7554/eLife.97769; "
                         "Marin EC, Morris BJ, Stuerner T, et al. (2024). eLife 13:RP97766. "
                         "doi:10.7554/eLife.97766; Cheong HSJ, Eichler K, Stuerner T, et al. (2024). "
                         "eLife 13:RP96084. doi:10.7554/eLife.96084. Data: FlyEM MANC v1.0, "
                         "gs://flyem-manc-exports/v1.0 (CC BY 4.0)."),
    DatasetSpec("ol", "ol", "optic_lobe_v1.1", OrganismStage.ADULT, RegionVocabulary.OPTIC_LOBE_NEUROPIL,
                LICENSE_CC_BY_4_0, (_C, _N), DatasetStatus.PLANNED,
                "Janelia FlyEM male right optic lobe (neuPrint optic-lobe:v1.1; flat export "
                "2024-09-11-a7d912, minconf 0.5). Builders: connectivity_tier, neurotransmitter_dominance "
                "(docs/FLYBRAIN_OL_ADAPTER_CONTRACT.md). No region_specialization_tier: roiInfo is the only "
                "region signal and already defines region_id and the inputs, so any such label is circular.",
                split_group_keys=("cell_type",),
                citation="Nern A, Loesche F, Takemura S, Burnett LE, Dreher M, et al. (2025). "
                         "Connectome-driven neural inventory of a complete visual system. Nature "
                         "641:1225-1237. doi:10.1038/s41586-025-08746-0. Data: FlyEM optic-lobe:v1.1, "
                         "gs://flyem-optic-lobe/v1.1 (CC BY 4.0)."),
)

DATASET_REGISTRY: dict[str, DatasetSpec] = {spec.symbol: spec for spec in _SPECS}


def _validate_registry() -> None:
    for symbol, spec in DATASET_REGISTRY.items():
        if symbol != symbol.lower() or not symbol.isalnum():
            raise AssertionError(f"registry symbol must be lowercase alphanumeric: {symbol!r}")
        if spec.snapshot_dir_name.lower() != symbol:
            raise AssertionError(f"snapshot_dir_name must case-fold to symbol: {spec.snapshot_dir_name!r}")
        unknown = [obj for obj in spec.supported_objectives if obj not in KNOWN_OBJECTIVES]
        if unknown:
            raise AssertionError(f"{symbol}: unknown objectives {unknown}")
        if spec.is_active and not spec.pinned_version:
            raise AssertionError(f"{symbol}: active datasets must pin a version")
        if spec.is_active and symbol in NEVER_ACTIVE_SYMBOLS:
            raise AssertionError(f"{symbol}: dataset is deferred and must not be active")
        if spec.is_active and not spec.supported_objectives:
            raise AssertionError(f"{symbol}: active datasets must support at least one objective")
        if spec.license != LICENSE_UNREVIEWED and not spec.pinned_version:
            raise AssertionError(f"{symbol}: a reviewed licence must be tied to a pinned snapshot version")
        if any(not key or key != key.strip() for key in spec.split_group_keys):
            raise AssertionError(f"{symbol}: split_group_keys must be non-empty, trimmed strings")


_validate_registry()


def normalize_symbol(raw: str) -> str:
    """Return the canonical lowercase symbol (accepts on-disk casing such as ``BANC``)."""
    if not isinstance(raw, str) or not raw.strip():
        raise DatasetRegistryError(DatasetErrorCode.UNKNOWN_DATASET, "dataset symbol must be a non-empty string")
    symbol = raw.strip().lower()
    if symbol not in DATASET_REGISTRY:
        raise DatasetRegistryError(
            DatasetErrorCode.UNKNOWN_DATASET,
            f"unknown dataset symbol {raw!r}; known: {', '.join(sorted(DATASET_REGISTRY))}",
            symbol=symbol,
        )
    return symbol


def get_dataset(symbol: str) -> DatasetSpec:
    return DATASET_REGISTRY[normalize_symbol(symbol)]


def list_datasets(status: DatasetStatus | str | None = None) -> tuple[DatasetSpec, ...]:
    wanted = DatasetStatus(status) if status is not None else None
    return tuple(
        DATASET_REGISTRY[key]
        for key in sorted(DATASET_REGISTRY)
        if wanted is None or DATASET_REGISTRY[key].status is wanted
    )


def supported_objectives(symbol: str) -> tuple[str, ...]:
    return get_dataset(symbol).supported_objectives


def require_objective(symbol: str, objective: str) -> DatasetSpec:
    spec = get_dataset(symbol)
    if objective not in KNOWN_OBJECTIVES:
        raise DatasetRegistryError(
            DatasetErrorCode.UNKNOWN_OBJECTIVE,
            f"unknown objective {objective!r}; known: {', '.join(KNOWN_OBJECTIVES)}",
            symbol=spec.symbol,
        )
    if not spec.supports(objective):
        raise DatasetRegistryError(
            DatasetErrorCode.OBJECTIVE_NOT_SUPPORTED,
            f"objective {objective!r} not supported for {spec.symbol}; "
            f"supported: {', '.join(spec.supported_objectives)}",
            symbol=spec.symbol,
        )
    return spec


def require_active(symbol: str, *, allow_planned: bool = False) -> DatasetSpec:
    spec = get_dataset(symbol)
    if not spec.is_active and not allow_planned:
        raise DatasetRegistryError(
            DatasetErrorCode.DATASET_NOT_ACTIVE,
            f"dataset {spec.symbol} is {spec.status.value}; pass allow_planned=True for pre-promotion runs",
            symbol=spec.symbol,
        )
    return spec


def snapshot_version_root(symbol: str, storage_root: str | Path | None = None) -> Path:
    """``<root>/snapshots/<snapshot_dir_name>/<pinned_version>`` (no existence check, no mkdir)."""
    spec = get_dataset(symbol)
    if not spec.pinned_version:
        raise DatasetRegistryError(
            DatasetErrorCode.VERSION_UNPINNED,
            f"dataset {spec.symbol} has no pinned snapshot version",
            symbol=spec.symbol,
        )
    layout = build_flybrain_harness_layout(root_override=storage_root, create=False)
    return layout.snapshots / spec.snapshot_dir_name / spec.pinned_version


def split_group_keys(symbol: str) -> tuple[str, ...]:
    """Sample-metadata keys whose shared values must never straddle a split."""
    return get_dataset(symbol).split_group_keys
