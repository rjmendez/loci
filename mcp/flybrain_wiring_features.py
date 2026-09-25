"""Dataset-agnostic per-neuron wiring features + per-objective label-exclusion rules.

Inputs
------
* an edge table (``EdgeSource``: feather/IPC or parquet with ``pre``, ``post``
  and optionally ``weight`` and ``neuropil`` columns), streamed with
  ``pyarrow.dataset`` column projection; 150M-edge tables are never
  materialized as Python objects,
* optionally a second edge table carrying the neuropil column
  (``neuropil_edges``, e.g. BANC ``edgelist_split_v3`` or fw
  ``proofread_connections``) when the pair-level table has none,
* a node annotation table (``pandas.DataFrame``: one row per neuron, an id
  column matching the edge ids and a partner-category column such as
  ``super_class``), optionally harmonized through ``vocab_map`` (see
  ``harmonize_super_class`` / ``SUPER_CLASS_VOCAB``).

Features (column ``<family>__<name>``; the family is what ablations drop)
-----------------------------------------------------------------------
* ``degree``: out/in total weight (all partners), annotated-partner counts,
  mean/max per-pair weight, log1p versions, out/(in+out) weight ratio.
* ``out_comp`` / ``in_comp``: weighted fraction of output/input going to each
  partner category (plus ``unannotated`` and ``unknown``) and its entropy.
* ``out_np`` / ``in_np``: top-K neuropil weight fractions (K fixed globally by
  total weight), ``other`` fraction, entropy (bits) and neuropil count.
* ``recip``: reciprocated fraction of out/in weight and of partners
  (annotated partners only).
* ``out2_comp`` / ``in2_comp`` (``two_hop=True``): partner composition of the
  partners (row-normalized W @ composition).

R5 robust features (research synthesis item 6; [Schlegel 2024; Scheffer 2025;
Stürner 2025]), opt-in so every v1 fingerprint and cache entry is unchanged:

* ``two_hop_min_weight`` (R5 default 5, sensitivity 10): both 2-hop hops use
  only annotated pairs whose pair-aggregated weight reaches the threshold.
* ``degree_transform='rank'``: extensive degree / weight / neuropil-count
  columns become within-dataset percentile ranks (``*_rank``); log1p copies go.
* ``robust_wiring_params()`` builds the R5 params; ``FLYBRAIN_WIRING_PROFILE``
  (``v1`` | ``r5`` | ``r5:10``) or ``with wiring_profile('r5'):`` upgrades the
  params of any unchanged caller.
* Partner-NT hard block: an NT objective (``is_nt_objective``) refuses an
  NT-derived partner category (builder and cache loader), and every
  ``*__gaba`` / ``*_pnt__*`` / ``*inhibit*``-style column is excluded for it.

Partner counts, reciprocity and 2-hop use only edges whose two endpoints are
in the node table (deduplicated per pair), so they mean the same thing on a
per-pair table (mc/BANC simple) and on a per-neuropil table (fw).

Caching: the full (unexcluded) table is written once to
``<cache_root>/wiring-features/<dataset>/<fingerprint>.parquet`` (+ ``.json``
sidecar). The fingerprint covers the edge files (path, size, mtime, declared
provenance such as a snapshot manifest sha256), the node table content, the
vocab map and the params. Cache writes refuse any path inside a
``snapshots`` directory.

Label exclusion (enforced, not documented): ``build_wiring_features`` and
``load_wiring_features`` require an ``objective`` and drop every column that
matches the objective's declared exclusion patterns (plus the global id
patterns) before returning. ``assert_features_allowed`` fails closed when a
caller puts an excluded feature into a sample; the eval harness calls it.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

WIRING_FEATURES_SCHEMA_VERSION = "flybrain-wiring-features/v1"
DEFAULT_CACHE_ROOT = "/mnt/f/.flybrain/cache"
UNANNOTATED = "unannotated"
UNKNOWN = "unknown"
NODE_ID_COLUMN = "node_id"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


# =========================================================================== exclusions


class LabelLeakageError(ValueError):
    """A feature that defines (or is a monotone proxy of) the label reached a model."""


@dataclass(frozen=True)
class ObjectiveExclusion:
    objective: str
    patterns: tuple[str, ...]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"objective": self.objective, "patterns": list(self.patterns), "reason": self.reason}


# Identifiers are never features for any objective.
GLOBAL_EXCLUDED_PATTERNS: tuple[str, ...] = (
    "id", "*_id", "*_ids", "*root_id*", "*body_id*", "bodyid", "body", "*pt_root*", "root_*",
    "supervoxel*", "nucleus*", "sample_id", "node_id", "*_match", "*_match_*", "seed_*",
)

_NT_PATTERNS = (
    "*_avg", "nt__*", "nt_*", "*_nt", "*_nt_*", "*conf_nt*", "*neurotransmitter*", "*transmitter*",
    "ach", "ach_*", "gaba", "gaba_*", "glut", "glut_*", "da", "da_*", "dopamine*", "ser", "ser_*",
    "serotonin*", "oct", "oct_*", "octopamine*", "his", "his_*", "histamine*", "tyr*", "*nt_prob*",
    "*nt_score*", "predicted_nt*", "*neuropeptide*",
)
# R5 hard block: partner-NT-derived features (e.g. the "fraction of inhibitory
# inputs" of [SYNTHESIS F3], or a composition family built on partner
# predicted NT such as MANC ``*_comp_pnt__*``) never reach an NT model. These
# name patterns catch such columns however they were built and merged in;
# ``build_wiring_features`` / ``load_wiring_features`` additionally refuse an
# NT objective whose partner category itself is an NT column (see
# ``assert_partner_category_allowed``).
NT_VALUE_VOCAB: frozenset[str] = frozenset({
    "ach", "acetylcholine", "cholinergic", "gaba", "gabaergic", "glu", "glut", "glutamate", "glutamatergic",
    "da", "dopamine", "dopaminergic", "ser", "5ht", "5_ht", "serotonin", "serotonergic", "oct", "octopamine",
    "octopaminergic", "his", "histamine", "histaminergic", "tyr", "tyramine", "tyraminergic",
    "inhibitory", "excitatory", "inhib", "excit",
})
_PARTNER_NT_PATTERNS = tuple(f"*__{v}" for v in sorted(NT_VALUE_VOCAB)) + tuple(
    f"*__{v}_*" for v in sorted(NT_VALUE_VOCAB)) + (
    "*_pnt__*", "*_pnt_*", "*comp_pnt*", "*comp_nt*", "*_nt__*", "pnt", "pnt_*", "*_pnt",
    "*predictednt*", "*consensusnt*", "*topnt*", "*top_nt*", "*known_nt*", "*synister*",
    "*inhibit*", "*excitat*", "*inhib_*", "*excit_*", "*_sign__*", "*dale*",
)
_NT_PATTERNS = _NT_PATTERNS + _PARTNER_NT_PATTERNS
_NT_NAME_RE = re.compile(r"(^|[^a-z])(p?nt|neurotransmitter|transmitter)([^a-z]|$)")
NT_OBJECTIVES: set[str] = {"neurotransmitter_dominance", "nt_ground_truth"}


def register_nt_objective(objective: str) -> None:
    """Mark ``objective`` as an NT target: the partner-NT hard block then applies to it."""
    name = str(objective).strip()
    if not name:
        raise ValueError("objective is required")
    NT_OBJECTIVES.add(name)


def is_nt_objective(objective: str) -> bool:
    """True for registered NT objectives and (fail closed) any objective whose name says NT."""
    name = str(objective).strip().lower()
    return name in NT_OBJECTIVES or bool(_NT_NAME_RE.search(name))


def _nt_like_name(name: str) -> bool:
    lowered = slug(name)
    return bool(_NT_NAME_RE.search(lowered)) or lowered in NT_VALUE_VOCAB or any(
        fnmatch.fnmatchcase(lowered, p) for p in _NT_PATTERNS)


def partner_category_is_nt(category_column: str, categories: Iterable[Any], *, category_tag: str = "",
                           min_fraction: float = 0.5) -> bool:
    """Heuristic, fail-closed: is this partner category an NT (or E/I sign) annotation?

    True when the column name or tag looks like an NT field, or when at least
    ``min_fraction`` of the labelled nodes carry an NT-vocabulary value (so a
    renamed NT column is still caught).
    """
    if _nt_like_name(category_column) or (category_tag and _nt_like_name(category_tag)):
        return True
    labelled = [slug(v) for v in categories
                if not (v is None or (isinstance(v, float) and math.isnan(v)))
                and slug(v) not in {UNKNOWN, UNANNOTATED, "other"}]
    if not labelled:
        return False
    hits = sum(1 for v in labelled if v in NT_VALUE_VOCAB)
    return hits / len(labelled) >= float(min_fraction)


def assert_partner_category_allowed(objective: str, category_column: str, categories: Iterable[Any], *,
                                    category_tag: str = "") -> None:
    """Hard R5 block: an NT objective may not use partner-NT-derived features (raises ``LabelLeakageError``)."""
    if is_nt_objective(objective) and partner_category_is_nt(category_column, categories, category_tag=category_tag):
        raise LabelLeakageError(
            f"objective {objective!r} is an NT target and partner category {category_column!r}"
            f"{' (tag ' + repr(category_tag) + ')' if category_tag else ''} is NT-derived; partner-NT features "
            "(e.g. fraction of inhibitory inputs) are banned from every NT model [SYNTHESIS F3; R5]"
        )
_REGION_PATTERNS = (
    "*_np__*", "np__*", "*neuropil*", "*roi*", "region*", "*_region", "cns_division", "subdivision",
    "*position*", "neuromere", "*_side_index",
)
_TYPE_HIERARCHY_PATTERNS = (
    "super_class", "superclass", "super_class_*", "cell_class", "class", "cell_sub_class", "sub_class",
    "*cell_type*", "type", "*_type", "flow", "cell_function*", "*cluster*", "nerve", "*_nerve",
    "body_part_*", "peripheral_target_type",
)
_DEGREE_PATTERNS = (
    "degree__*", "*partner_tier*", "*n_partners*", "*weight_total*", "*_count", "*_counts", "n_*",
    "*synapse*", "*connections*", "*pre_count*", "*post_count*", "total_*", "*_tier",
    "*n_neuropils*",
)

_EXCLUSIONS: dict[str, ObjectiveExclusion] = {}


def register_objective_exclusions(objective: str, patterns: Sequence[str], *, reason: str,
                                  replace: bool = False) -> ObjectiveExclusion:
    """Declare the feature-name patterns (fnmatch, case-insensitive) that define ``objective``'s label."""
    name = str(objective).strip()
    if not name:
        raise ValueError("objective is required")
    if not str(reason).strip():
        raise ValueError("an exclusion reason is required (why these features define the label)")
    if name in _EXCLUSIONS and not replace:
        raise ValueError(f"exclusions already registered for objective {name!r}")
    entry = ObjectiveExclusion(name, tuple(dict.fromkeys(str(p).strip().lower() for p in patterns if str(p).strip())),
                               str(reason).strip())
    _EXCLUSIONS[name] = entry
    return entry


def registered_objectives() -> tuple[str, ...]:
    return tuple(sorted(_EXCLUSIONS))


def objective_exclusions(objective: str) -> ObjectiveExclusion:
    if objective not in _EXCLUSIONS:
        raise ValueError(
            f"no label-exclusion rules registered for objective {objective!r}; register them with "
            f"register_objective_exclusions() before building features (registered: {', '.join(registered_objectives())})"
        )
    return _EXCLUSIONS[objective]


def excluded_feature_names(names: Iterable[str], objective: str) -> list[str]:
    patterns = GLOBAL_EXCLUDED_PATTERNS + objective_exclusions(objective).patterns
    out = []
    for name in names:
        lowered = str(name).lower()
        if any(fnmatch.fnmatchcase(lowered, pattern) for pattern in patterns):
            out.append(str(name))
    return sorted(out)


def assert_features_allowed(names: Iterable[str], objective: str) -> None:
    """Fail closed if any feature name matches the objective's (or the global id) exclusions."""
    offenders = excluded_feature_names(names, objective)
    if offenders:
        rule = objective_exclusions(objective)
        raise LabelLeakageError(
            f"features excluded for objective {objective!r} ({rule.reason}): {', '.join(offenders[:20])}"
        )


def apply_objective_exclusions(frame: pd.DataFrame, objective: str) -> tuple[pd.DataFrame, list[str]]:
    dropped = excluded_feature_names([c for c in frame.columns if c != NODE_ID_COLUMN], objective)
    return frame.drop(columns=dropped), dropped


for _objective, _patterns, _reason in (
    ("connectivity_tier", _DEGREE_PATTERNS,
     "label is a threshold on a neuron's synapse count; degree, weight sums and partner tiers are monotone proxies"),
    ("neurotransmitter_dominance", _NT_PATTERNS,
     "label is the argmax of NT prediction scores; NT scores/predictions define it"),
    ("nt_ground_truth", _NT_PATTERNS,
     "label is an annotated transmitter; NT predictions and their scores are near-copies of it"),
    ("region_specialization_tier", _REGION_PATTERNS,
     "label is derived from the neuron's per-neuropil synapse distribution"),
    ("primary_region", _REGION_PATTERNS, "label is the neuron's top neuropil"),
    ("super_class", _TYPE_HIERARCHY_PATTERNS,
     "label is a level of the curated cell-type hierarchy; other levels (class/type/flow/nerve) determine it"),
    ("cell_class", _TYPE_HIERARCHY_PATTERNS, "label is a level of the curated cell-type hierarchy"),
    ("flow", _TYPE_HIERARCHY_PATTERNS, "flow is a deterministic function of super_class / cell class"),
    ("hemilineage", _TYPE_HIERARCHY_PATTERNS + ("*hemilineage*", "*lineage*", "ito*", "truman*"),
     "label is the hemilineage annotation; lineage fields and cell types determine it"),
):
    register_objective_exclusions(_objective, _patterns, reason=_reason)


# =========================================================================== vocab


def slug(value: Any) -> str:
    text = _SLUG_RE.sub("_", str(value).strip().lower()).strip("_")
    return text or UNKNOWN


SUPER_CLASS_VOCAB: dict[str, str] = {
    # FlyWire 783
    "central": "intrinsic_central", "optic": "intrinsic_optic", "sensory": "sensory",
    "visual_projection": "visual_projection", "visual_centrifugal": "visual_centrifugal",
    "ascending": "ascending", "descending": "descending", "motor": "motor", "endocrine": "endocrine",
    # BANC 888
    "central_brain_intrinsic": "intrinsic_central", "optic_lobe_intrinsic": "intrinsic_optic",
    "ventral_nerve_cord_intrinsic": "intrinsic_vnc", "sensory_ascending": "sensory",
    "sensory_descending": "sensory", "visceral_circulatory": "endocrine",
    "ascending_visceral_circulatory": "endocrine", "glia": "non_neuronal", "trachea": "non_neuronal",
    "not_a_neuron": "non_neuronal",
    # male-cns v1.0 / optic-lobe
    "cb_intrinsic": "intrinsic_central", "ol_intrinsic": "intrinsic_optic", "vnc_intrinsic": "intrinsic_vnc",
    "cb_sensory": "sensory", "ol_sensory": "sensory", "vnc_sensory": "sensory",
    "descending_neuron": "descending", "ascending_neuron": "ascending", "cb_motor": "motor",
    "vnc_motor": "motor", "cb_endocrine": "endocrine", "vnc_endocrine": "endocrine",
    "cb_efferent": "efferent", "vnc_efferent": "efferent", "efferent_ascending": "efferent",
    "efferent_descending": "efferent", "ens": "other", "vnc": UNKNOWN,
    # MANC v1.0
    "intrinsic_neuron": "intrinsic_vnc", "sensory_neuron": "sensory", "motor_neuron": "motor",
    "efferent_neuron": "efferent", "sensory_ascending_neuron": "sensory", "efferent_ascending_neuron": "efferent",
    "sensory_descending_neuron": "sensory",
}


def harmonize_super_class(value: Any) -> str:
    """Map a dataset's super-class spelling onto the shared coarse vocabulary."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return UNKNOWN
    key = slug(value)
    if key.endswith("_tbc"):
        key = key[: -len("_tbc")]
    return SUPER_CLASS_VOCAB.get(key, key)


def _harmonizer(vocab_map: Mapping[str, str] | Callable[[Any], str] | None) -> Callable[[Any], str]:
    if vocab_map is None:
        return lambda v: UNKNOWN if v is None or (isinstance(v, float) and math.isnan(v)) else slug(v)
    if callable(vocab_map):
        return lambda v: slug(vocab_map(v))  # type: ignore[operator]
    mapping = {slug(k): slug(v) for k, v in vocab_map.items()}
    return lambda v: UNKNOWN if v is None or (isinstance(v, float) and math.isnan(v)) else mapping.get(slug(v), slug(v))


# =========================================================================== specs


@dataclass(frozen=True)
class EdgeSource:
    path: str
    pre: str
    post: str
    weight: str | None = None
    neuropil: str | None = None
    unique_pairs: bool = True
    format: str = "auto"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def file_format(self) -> str:
        if self.format != "auto":
            return self.format
        return "parquet" if str(self.path).endswith(".parquet") else "ipc"

    def identity(self) -> dict[str, Any]:
        stat = os.stat(self.path)
        return {
            "path": str(Path(self.path).resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "columns": [self.pre, self.post, self.weight, self.neuropil],
            "unique_pairs": bool(self.unique_pairs),
            "format": self.file_format(),
            "provenance": json.loads(json.dumps(dict(self.provenance), sort_keys=True, default=str)),
        }


@dataclass(frozen=True)
class WiringFeatureParams:
    top_k_neuropils: int = 20
    reciprocity: bool = True
    two_hop: bool = False
    min_edge_weight: float = 0.0
    category_tag: str = ""
    max_pair_rows: int = 120_000_000
    batch_rows: int = 1 << 20
    # --- R5 robust features (defaults reproduce the v1 features bit for bit) ---
    # 2-hop composition is built only from annotated pairs whose (pair-aggregated)
    # weight is >= this many synapses, on BOTH hops (i -> j and j -> partners of j).
    # Edges > 10 synapses reproduce > 90% of the time across animals [Schlegel 2024];
    # FIB-SEM detects > 40% more synapses than TEM [Scheffer 2025]. R5: 5, sensitivity 10.
    two_hop_min_weight: float = 0.0
    # 'raw' (v1) keeps raw degree counts/weight sums; 'rank' replaces every
    # extensive degree / neuropil-count column with its within-dataset percentile
    # rank in (0, 1] (scale-free, so FIB-SEM vs TEM weight scaling cancels
    # [Stürner 2025: normalized DNa02 slope 0.69 vs raw 0.42]).
    degree_transform: str = "raw"

    def __post_init__(self) -> None:
        if self.degree_transform not in DEGREE_TRANSFORMS:
            raise ValueError(f"degree_transform must be one of {DEGREE_TRANSFORMS}")
        if float(self.two_hop_min_weight) < 0 or not math.isfinite(float(self.two_hop_min_weight)):
            raise ValueError("two_hop_min_weight must be a finite number >= 0")
        if float(self.two_hop_min_weight) > 0 and not self.two_hop:
            raise ValueError("two_hop_min_weight > 0 requires two_hop=True")

    def as_dict(self) -> dict[str, Any]:
        """Fingerprint payload. R5 fields are recorded only when they differ from the
        v1 defaults, so every v1 fingerprint (and cache entry) is unchanged."""
        out = asdict(self)
        for key, legacy in _R5_LEGACY_DEFAULTS.items():
            if out[key] == legacy:
                del out[key]
        return out

    def is_robust(self) -> bool:
        return any(getattr(self, k) != v for k, v in _R5_LEGACY_DEFAULTS.items())


DEGREE_TRANSFORMS = ("raw", "rank")
_R5_LEGACY_DEFAULTS: dict[str, Any] = {"two_hop_min_weight": 0.0, "degree_transform": "raw"}
R5_TWO_HOP_MIN_WEIGHT = 5.0
R5_SENSITIVITY_TWO_HOP_MIN_WEIGHT = 10.0


def robust_wiring_params(base: WiringFeatureParams | None = None, *,
                         two_hop_min_weight: float = R5_TWO_HOP_MIN_WEIGHT, degree_transform: str = "rank",
                         **overrides: Any) -> WiringFeatureParams:
    """The R5 configuration: 2-hop on edges >= 5 synapses (10 for the sensitivity run) and
    rank-normalized degree features. ``base`` supplies every other field (top-K, tag, ...)."""
    from dataclasses import replace

    base = base or WiringFeatureParams()
    return replace(base, two_hop=True, two_hop_min_weight=float(two_hop_min_weight),
                   degree_transform=degree_transform, **overrides)


# ---- feature profiles: let every dataset track request R5 features without code changes.
# ``FLYBRAIN_WIRING_PROFILE`` = "v1" (default) | "r5" (2-hop >= 5, rank) | "r5:<min_weight>"
# (e.g. "r5:10" for the sensitivity run). ``wiring_profile(...)`` does the same in-process.
# The profile rewrites the params BEFORE fingerprinting, so it is recorded in the
# fingerprint and in the cache sidecar; the v1 profile leaves params untouched.
WIRING_PROFILE_ENV = "FLYBRAIN_WIRING_PROFILE"
_PROFILE_OVERRIDE: list[str] = []


def parse_wiring_profile(text: str | None) -> dict[str, Any] | None:
    """``None`` for v1, else ``{'two_hop_min_weight': w, 'degree_transform': 'rank', 'name': ...}``."""
    raw = (text or "").strip().lower()
    if raw in ("", "v1", "legacy", "raw"):
        return None
    name, _, arg = raw.partition(":")
    if name != "r5":
        raise ValueError(f"unknown wiring profile {text!r}; use v1, r5 or r5:<min_weight>")
    weight = float(arg) if arg else R5_TWO_HOP_MIN_WEIGHT
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"wiring profile {text!r}: min weight must be > 0")
    return {"name": f"r5:{weight:g}", "two_hop_min_weight": weight, "degree_transform": "rank"}


def active_wiring_profile() -> dict[str, Any] | None:
    return parse_wiring_profile(_PROFILE_OVERRIDE[-1] if _PROFILE_OVERRIDE else os.environ.get(WIRING_PROFILE_ENV))


def apply_wiring_profile(params: WiringFeatureParams, profile: dict[str, Any] | None = None) -> WiringFeatureParams:
    """Upgrade ``params`` to the active profile. Params that are already non-v1 are kept as given;
    the 2-hop threshold is only set when the caller asked for 2-hop features."""
    from dataclasses import replace

    profile = active_wiring_profile() if profile is None else profile
    if not profile or params.is_robust():
        return params
    return replace(params, degree_transform=profile["degree_transform"],
                   two_hop_min_weight=profile["two_hop_min_weight"] if params.two_hop else 0.0)


class wiring_profile:  # noqa: N801 - context manager used like a function
    """``with wiring_profile('r5:10'): ...`` -- scoped override of ``FLYBRAIN_WIRING_PROFILE``."""

    def __init__(self, profile: str) -> None:
        parse_wiring_profile(profile)
        self.profile = profile

    def __enter__(self) -> "wiring_profile":
        _PROFILE_OVERRIDE.append(self.profile)
        return self

    def __exit__(self, *exc: Any) -> None:
        _PROFILE_OVERRIDE.pop()


@dataclass(frozen=True)
class WiringFeatureResult:
    frame: pd.DataFrame
    fingerprint: str
    cache_path: str | None
    objective: str
    excluded_features: tuple[str, ...]
    meta: Mapping[str, Any]
    # node id + every degree / size column BEFORE objective exclusions (R3 size baseline and degree
    # deciles read these through EvalDataset.aux; they never reach a model).
    size_frame: pd.DataFrame | None = None

    def feature_families(self) -> dict[str, list[str]]:
        return feature_families(c for c in self.frame.columns if c != NODE_ID_COLUMN)


def feature_family(name: str) -> str:
    return name.split("__", 1)[0] if "__" in name else name


def feature_families(names: Iterable[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name in names:
        out.setdefault(feature_family(str(name)), []).append(str(name))
    return {k: sorted(v) for k, v in sorted(out.items())}


# =========================================================================== streaming core


def _iter_batches(source: EdgeSource, columns: Sequence[str], batch_rows: int):
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(source.path), format=source.file_format())
    names = set(dataset.schema.names)
    missing = [c for c in columns if c not in names]
    if missing:
        raise ValueError(f"edge table {source.path} lacks columns: {', '.join(missing)}")
    return dataset, dataset.to_batches(columns=list(columns), batch_size=int(batch_rows))


def _node_value_set(node_ids: Sequence[Any], edge_type: Any):
    import pyarrow as pa

    if pa.types.is_integer(edge_type):
        return pa.array([int(v) for v in node_ids], type=pa.int64())
    if pa.types.is_string(edge_type) or pa.types.is_large_string(edge_type):
        return pa.array([str(v) for v in node_ids], type=pa.string())
    raise ValueError(f"unsupported edge id type {edge_type}")


def _index_of(column, value_set) -> np.ndarray:
    import pyarrow as pa
    import pyarrow.compute as pc

    if pa.types.is_integer(column.type) and column.type != pa.int64():
        column = pc.cast(column, pa.int64())  # safe cast: raises on uint64 overflow
    if pa.types.is_large_string(column.type):
        column = pc.cast(column, pa.string())
    return pc.fill_null(pc.index_in(column, value_set=value_set), -1).to_numpy(zero_copy_only=False).astype(np.int64)


def _weights(batch, column: str | None) -> np.ndarray:
    if column is None:
        return np.ones(batch.num_rows, dtype=np.float64)
    arr = batch.column(column)
    if arr.null_count:
        raise ValueError(f"edge weight column {column} has nulls")
    return arr.to_numpy(zero_copy_only=False).astype(np.float64)


class _NeuropilAccumulator:
    def __init__(self, n_nodes: int) -> None:
        self.n = n_nodes
        self.codes: dict[str, int] = {}
        self.out: list[np.ndarray] = []
        self.inn: list[np.ndarray] = []

    def _global_codes(self, column) -> np.ndarray:
        import pyarrow.compute as pc

        encoded = pc.dictionary_encode(pc.fill_null(column.cast("string"), UNKNOWN))
        local = encoded.dictionary.to_pylist()
        remap = np.empty(len(local), dtype=np.int64)
        for i, raw in enumerate(local):
            key = slug(raw)
            if key not in self.codes:
                self.codes[key] = len(self.codes)
                self.out.append(np.zeros(self.n))
                self.inn.append(np.zeros(self.n))
            remap[i] = self.codes[key]
        return remap[encoded.indices.to_numpy(zero_copy_only=False)]

    def add(self, pre_idx: np.ndarray, post_idx: np.ndarray, w: np.ndarray, column) -> None:
        codes = self._global_codes(column)
        for target, store in ((pre_idx, self.out), (post_idx, self.inn)):
            mask = target >= 0
            if not mask.any():
                continue
            t, c, ww = target[mask], codes[mask], w[mask]
            for code in np.unique(c):
                sel = c == code
                store[code] += np.bincount(t[sel], weights=ww[sel], minlength=self.n)

    def matrices(self) -> tuple[list[str], np.ndarray, np.ndarray]:
        names = sorted(self.codes, key=lambda k: self.codes[k])
        if not names:
            return [], np.zeros((self.n, 0)), np.zeros((self.n, 0))
        return names, np.stack(self.out, axis=1), np.stack(self.inn, axis=1)


def _entropy_bits(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(totals > 0, matrix / totals, 0.0)
        logs = np.where(p > 0, np.log2(np.where(p > 0, p, 1.0)), 0.0)
    ent = -(p * logs).sum(axis=1)
    return np.where(totals[:, 0] > 0, ent, np.nan)


def _fractions(matrix: np.ndarray, totals: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(totals[:, None] > 0, matrix / totals[:, None], np.nan)


def _category_mass(rows: np.ndarray, cats: np.ndarray, w: np.ndarray, n: int, n_cat: int) -> np.ndarray:
    """(n, n_cat) matrix: summed weight of the edges of each ``rows`` node per partner category."""
    return np.bincount(rows * n_cat + cats, weights=w, minlength=n * n_cat).reshape(n, n_cat)


def _two_hop_composition(n: int, node_cat: np.ndarray, keys: np.ndarray, src: np.ndarray, dst: np.ndarray,
                         pw: np.ndarray, comp_out_j: np.ndarray, comp_in_j: np.ndarray,
                         out_den_j: np.ndarray, in_den_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalized W @ (partner composition), minus each node's i -> j -> i return path.

    ``keys``/``src``/``dst``/``pw`` are the (sorted, unique) pairs that define W;
    ``comp_*_j`` are the per-node compositions reached on the second hop and
    ``*_den_j`` the totals they were normalized by (so the return-path share of
    node i inside j's composition is ``w_ji / den_j``).
    """
    from scipy import sparse

    W = sparse.csr_matrix((pw, (src, dst)), shape=(n, n))
    row = np.asarray(W.sum(axis=1)).ravel()
    col = np.asarray(W.sum(axis=0)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        out2 = np.asarray(sparse.diags(np.where(row > 0, 1.0 / row, 0.0)) @ W @ comp_out_j)
        in2 = np.asarray(sparse.diags(np.where(col > 0, 1.0 / col, 0.0)) @ W.T.tocsr() @ comp_in_j)
    if len(keys):
        # Remove the i -> j -> i return path: j's composition counts i's own
        # category, which would put the node's own label into its features.
        reverse = dst * n + src
        pos = np.minimum(np.searchsorted(keys, reverse), len(keys) - 1)
        w_rev = np.where(keys[pos] == reverse, pw[pos], 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            self_out = np.bincount(src, weights=np.where(
                (row[src] > 0) & (out_den_j[dst] > 0), (pw / row[src]) * (w_rev / out_den_j[dst]), 0.0), minlength=n)
            self_in = np.bincount(dst, weights=np.where(
                (col[dst] > 0) & (in_den_j[src] > 0), (pw / col[dst]) * (w_rev / in_den_j[src]), 0.0), minlength=n)
        del reverse, pos, w_rev
        out2[np.arange(n), node_cat] = np.maximum(out2[np.arange(n), node_cat] - self_out, 0.0)
        in2[np.arange(n), node_cat] = np.maximum(in2[np.arange(n), node_cat] - self_in, 0.0)
    out2 = np.where(row[:, None] > 0, out2, np.nan)
    in2 = np.where(col[:, None] > 0, in2, np.nan)
    return out2, in2


# Extensive (size-scaled) columns that ``degree_transform='rank'`` turns into
# within-dataset percentile ranks; their log1p copies are dropped (rank-identical).
_RANKED_EXTENSIVE = (
    "degree__out_weight_total", "degree__in_weight_total", "degree__out_n_annotated_partners",
    "degree__in_n_annotated_partners", "degree__out_mean_pair_weight", "degree__in_mean_pair_weight",
    "degree__out_max_pair_weight", "degree__in_max_pair_weight", "out_np__n_neuropils", "in_np__n_neuropils",
)
_DROPPED_UNDER_RANK = (
    "degree__log1p_out_weight_total", "degree__log1p_in_weight_total",
    "degree__log1p_out_n_annotated_partners", "degree__log1p_in_n_annotated_partners",
)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Average-tie percentile rank in (0, 1] over the finite entries; NaN stays NaN.

    Exact zeros (no edges at all) map to 0.0, so ``rank > 0`` iff ``raw > 0``: the
    absence of inputs/outputs stays visible and callers can still test for it.
    """
    from scipy.stats import rankdata

    values = np.asarray(values, dtype=np.float64)
    out = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    m = int(finite.sum())
    if m:
        out[finite] = rankdata(values[finite], method="average") / m
    out[finite & (values == 0)] = 0.0
    return out


def _rank_transform(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, values in features.items():
        if name in _DROPPED_UNDER_RANK:
            continue
        if name in _RANKED_EXTENSIVE:
            out[f"{name}_rank"] = percentile_rank(values)
        else:
            out[name] = values
    return out


def compute_wiring_features(
    *,
    edges: EdgeSource,
    node_ids: Sequence[Any],
    node_categories: Sequence[str],
    params: WiringFeatureParams = WiringFeatureParams(),
    neuropil_edges: EdgeSource | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stream the edge table(s) once and return (feature frame indexed like ``node_ids``, stats)."""
    import pyarrow as pa

    n = len(node_ids)
    if n == 0:
        raise ValueError("node table is empty")
    categories = sorted(set(node_categories) | {UNANNOTATED, UNKNOWN})
    cat_code = {c: i for i, c in enumerate(categories)}
    n_cat = len(categories)
    node_cat = np.asarray([cat_code[c] for c in node_categories], dtype=np.int64)
    unannotated_code = cat_code[UNANNOTATED]

    out_total = np.zeros(n)
    in_total = np.zeros(n)
    out_comp = np.zeros(n * n_cat)
    in_comp = np.zeros(n * n_cat)
    np_acc = _NeuropilAccumulator(n)
    pair_keys: list[np.ndarray] = []
    pair_w: list[np.ndarray] = []
    pair_rows = 0
    scanned = 0
    min_w = float(params.min_edge_weight)

    columns = [edges.pre, edges.post] + [c for c in (edges.weight, edges.neuropil) if c]
    dataset, batches = _iter_batches(edges, columns, params.batch_rows)
    value_set = _node_value_set(node_ids, dataset.schema.field(edges.pre).type)
    for batch in batches:
        if batch.num_rows == 0:
            continue
        scanned += batch.num_rows
        pre_idx = _index_of(batch.column(edges.pre), value_set)
        post_idx = _index_of(batch.column(edges.post), value_set)
        w = _weights(batch, edges.weight)
        if min_w > 0:
            keep = w >= min_w
            pre_idx, post_idx, w = pre_idx[keep], post_idx[keep], w[keep]
            batch = batch.filter(pa.array(keep))
        post_cat = np.where(post_idx >= 0, node_cat[np.maximum(post_idx, 0)], unannotated_code)
        pre_cat = np.where(pre_idx >= 0, node_cat[np.maximum(pre_idx, 0)], unannotated_code)
        m = pre_idx >= 0
        out_total += np.bincount(pre_idx[m], weights=w[m], minlength=n)
        out_comp += np.bincount(pre_idx[m] * n_cat + post_cat[m], weights=w[m], minlength=n * n_cat)
        m = post_idx >= 0
        in_total += np.bincount(post_idx[m], weights=w[m], minlength=n)
        in_comp += np.bincount(post_idx[m] * n_cat + pre_cat[m], weights=w[m], minlength=n * n_cat)
        both = (pre_idx >= 0) & (post_idx >= 0)
        if both.any():
            pair_rows += int(both.sum())
            if pair_rows > int(params.max_pair_rows):
                raise ValueError(
                    f"annotated pair rows exceed max_pair_rows={params.max_pair_rows}; raise it (memory ~16 B/row) "
                    "or restrict the node table"
                )
            pair_keys.append(pre_idx[both] * n + post_idx[both])
            pair_w.append(w[both].astype(np.float32))
        if edges.neuropil and neuropil_edges is None:
            np_acc.add(pre_idx, post_idx, w, batch.column(edges.neuropil))

    np_scanned = 0
    if neuropil_edges is not None:
        if not neuropil_edges.neuropil:
            raise ValueError("neuropil_edges must declare a neuropil column")
        cols = [neuropil_edges.pre, neuropil_edges.post, neuropil_edges.neuropil] + (
            [neuropil_edges.weight] if neuropil_edges.weight else [])
        np_dataset, np_batches = _iter_batches(neuropil_edges, cols, params.batch_rows)
        np_value_set = _node_value_set(node_ids, np_dataset.schema.field(neuropil_edges.pre).type)
        for batch in np_batches:
            if batch.num_rows == 0:
                continue
            np_scanned += batch.num_rows
            w = _weights(batch, neuropil_edges.weight)
            np_acc.add(_index_of(batch.column(neuropil_edges.pre), np_value_set),
                       _index_of(batch.column(neuropil_edges.post), np_value_set), w,
                       batch.column(neuropil_edges.neuropil))

    features: dict[str, np.ndarray] = {}
    tag = f"_{slug(params.category_tag)}" if params.category_tag else ""

    # -- composition
    out_comp_m = out_comp.reshape(n, n_cat)
    in_comp_m = in_comp.reshape(n, n_cat)
    out_frac = _fractions(out_comp_m, out_total)
    in_frac = _fractions(in_comp_m, in_total)
    for k, name in enumerate(categories):
        features[f"out_comp{tag}__{name}"] = out_frac[:, k]
        features[f"in_comp{tag}__{name}"] = in_frac[:, k]
    features[f"out_comp{tag}__entropy"] = _entropy_bits(out_comp_m)
    features[f"in_comp{tag}__entropy"] = _entropy_bits(in_comp_m)

    # -- pairs (annotated endpoints only), deduplicated
    if pair_keys:
        keys = np.concatenate(pair_keys)
        pw = np.concatenate(pair_w).astype(np.float64)
    else:
        keys = np.zeros(0, dtype=np.int64)
        pw = np.zeros(0)
    del pair_keys, pair_w
    if not edges.unique_pairs:
        keys, inverse = np.unique(keys, return_inverse=True)
        pw = np.bincount(inverse, weights=pw, minlength=len(keys))
        del inverse
    else:
        order = np.argsort(keys, kind="stable")
        keys, pw = keys[order], pw[order]
        del order
        if len(keys) > 1 and np.any(keys[1:] == keys[:-1]):
            raise ValueError("edge table declared unique_pairs=True but contains duplicate (pre, post) rows")
    src = keys // n
    dst = keys % n
    out_partners = np.bincount(src, minlength=n).astype(np.float64)
    in_partners = np.bincount(dst, minlength=n).astype(np.float64)
    out_pair_w = np.bincount(src, weights=pw, minlength=n)
    in_pair_w = np.bincount(dst, weights=pw, minlength=n)
    out_max = np.zeros(n)
    in_max = np.zeros(n)
    if len(pw):
        np.maximum.at(out_max, src, pw)
        np.maximum.at(in_max, dst, pw)

    with np.errstate(divide="ignore", invalid="ignore"):
        features["degree__out_weight_total"] = out_total
        features["degree__in_weight_total"] = in_total
        features["degree__log1p_out_weight_total"] = np.log1p(out_total)
        features["degree__log1p_in_weight_total"] = np.log1p(in_total)
        features["degree__out_n_annotated_partners"] = out_partners
        features["degree__in_n_annotated_partners"] = in_partners
        features["degree__log1p_out_n_annotated_partners"] = np.log1p(out_partners)
        features["degree__log1p_in_n_annotated_partners"] = np.log1p(in_partners)
        features["degree__out_mean_pair_weight"] = np.where(out_partners > 0, out_pair_w / out_partners, np.nan)
        features["degree__in_mean_pair_weight"] = np.where(in_partners > 0, in_pair_w / in_partners, np.nan)
        features["degree__out_max_pair_weight"] = np.where(out_partners > 0, out_max, np.nan)
        features["degree__in_max_pair_weight"] = np.where(in_partners > 0, in_max, np.nan)
        total_io = out_total + in_total
        features["degree__out_in_weight_ratio"] = np.where(total_io > 0, out_total / total_io, np.nan)

    if params.reciprocity:
        reverse = dst * n + src
        pos = np.searchsorted(keys, reverse)
        pos = np.minimum(pos, max(len(keys) - 1, 0))
        recip = (keys[pos] == reverse) if len(keys) else np.zeros(0, dtype=bool)
        r_out_w = np.bincount(src[recip], weights=pw[recip], minlength=n)
        r_in_w = np.bincount(dst[recip], weights=pw[recip], minlength=n)
        r_out_n = np.bincount(src[recip], minlength=n)
        with np.errstate(divide="ignore", invalid="ignore"):
            features["recip__out_weight_frac"] = np.where(out_pair_w > 0, r_out_w / out_pair_w, np.nan)
            features["recip__in_weight_frac"] = np.where(in_pair_w > 0, r_in_w / in_pair_w, np.nan)
            features["recip__partner_frac"] = np.where(out_partners > 0, r_out_n / out_partners, np.nan)
        del reverse, pos, recip

    two_hop_stats: dict[str, Any] = {}
    if params.two_hop and len(keys):
        thr = float(params.two_hop_min_weight)
        if thr > 0:
            # R5: both hops use strong pairs only; j's composition is the composition
            # of j's own strong (annotated-table) partners, not of all its edges.
            strong = pw >= thr
            s_keys, s_pw = keys[strong], pw[strong]
            s_src, s_dst = src[strong], dst[strong]
            s_row = np.bincount(s_src, weights=s_pw, minlength=n)
            s_col = np.bincount(s_dst, weights=s_pw, minlength=n)
            comp_out_j = _fractions(_category_mass(s_src, node_cat[s_dst], s_pw, n, n_cat), s_row)
            comp_in_j = _fractions(_category_mass(s_dst, node_cat[s_src], s_pw, n, n_cat), s_col)
            out2, in2 = _two_hop_composition(n, node_cat, s_keys, s_src, s_dst, s_pw,
                                             np.nan_to_num(comp_out_j), np.nan_to_num(comp_in_j), s_row, s_col)
            two_hop_stats = {"two_hop_min_weight": thr, "two_hop_pairs": int(strong.sum()),
                             "two_hop_weight_fraction": float(s_pw.sum() / pw.sum()) if pw.sum() > 0 else 0.0}
            del s_keys, s_pw, s_src, s_dst, strong, comp_out_j, comp_in_j
        else:
            out2, in2 = _two_hop_composition(n, node_cat, keys, src, dst, pw, np.nan_to_num(out_frac),
                                             np.nan_to_num(in_frac), out_total, in_total)
        for k, name in enumerate(categories):
            features[f"out2_comp{tag}__{name}"] = out2[:, k]
            features[f"in2_comp{tag}__{name}"] = in2[:, k]
        del out2, in2

    # -- neuropil
    np_names, np_out, np_in = np_acc.matrices()
    top_neuropils: list[str] = []
    if np_names:
        totals = np_out.sum(axis=0) + np_in.sum(axis=0)
        ranked = sorted(range(len(np_names)), key=lambda i: (-totals[i], np_names[i]))
        top = ranked[: int(params.top_k_neuropils)]
        top_neuropils = [np_names[i] for i in top]
        for prefix, mat in (("out_np", np_out), ("in_np", np_in)):
            row_total = mat.sum(axis=1)
            frac = _fractions(mat, row_total)
            for i in top:
                features[f"{prefix}__{np_names[i]}"] = frac[:, i]
            rest = [i for i in range(len(np_names)) if i not in set(top)]
            features[f"{prefix}__other"] = frac[:, rest].sum(axis=1) if rest else np.where(row_total > 0, 0.0, np.nan)
            features[f"{prefix}__entropy"] = _entropy_bits(mat)
            features[f"{prefix}__n_neuropils"] = np.where(row_total > 0, (mat > 0).sum(axis=1).astype(float), np.nan)

    if params.degree_transform == "rank":
        features = _rank_transform(features)
    frame = pd.DataFrame(features)
    frame = frame.reindex(sorted(frame.columns), axis=1)
    stats = {
        "edges_scanned": int(scanned),
        "neuropil_edges_scanned": int(np_scanned),
        "annotated_pair_rows": int(pair_rows),
        "annotated_pairs": int(len(keys)),
        "categories": categories,
        "neuropils_seen": len(np_names),
        "top_neuropils": top_neuropils,
        "nodes": int(n),
        "nodes_with_outputs": int((out_total > 0).sum()),
        "nodes_with_inputs": int((in_total > 0).sum()),
        "degree_transform": params.degree_transform,
        **two_hop_stats,
    }
    return frame, stats


# =========================================================================== cached builder


def _reject_snapshot_path(path: Path) -> None:
    resolved = path.resolve(strict=False)
    if "snapshots" in resolved.parts:
        raise ValueError(f"refusing to write features under a snapshots directory: {resolved}")


def _nodes_digest(ids: Sequence[str], cats: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for node_id, cat in zip(ids, cats):
        digest.update(f"{node_id}\t{cat}\n".encode("utf-8"))
    return digest.hexdigest()


def wiring_fingerprint(*, dataset: str, edges: EdgeSource, neuropil_edges: EdgeSource | None,
                       nodes_digest: str, params: WiringFeatureParams, vocab_digest: str) -> str:
    payload = {
        "schema_version": WIRING_FEATURES_SCHEMA_VERSION,
        "dataset": dataset,
        "edges": edges.identity(),
        "neuropil_edges": None if neuropil_edges is None else neuropil_edges.identity(),
        "nodes_sha256": nodes_digest,
        "vocab_sha256": vocab_digest,
        "params": params.as_dict(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_wiring_features(
    *,
    dataset: str,
    objective: str,
    edges: EdgeSource,
    nodes: pd.DataFrame,
    id_column: str,
    category_column: str,
    vocab_map: Mapping[str, str] | Callable[[Any], str] | None = None,
    params: WiringFeatureParams = WiringFeatureParams(),
    neuropil_edges: EdgeSource | None = None,
    cache_root: str | Path | None = DEFAULT_CACHE_ROOT,
    use_cache: bool = True,
    mask_category_ids: Iterable[Any] | None = None,
) -> WiringFeatureResult:
    """Build (or load from cache) wiring features, then drop the columns ``objective`` excludes.

    Returns a frame with a ``node_id`` column (the node table's id as a
    decimal/string) and feature columns only; label-defining features for
    ``objective`` never leave this function.

    ``mask_category_ids``: nodes whose partner category is hidden (treated as
    ``unannotated``). REQUIRED for honesty whenever ``category_column`` is the
    target label itself (or determines it): pass the val + test node ids from
    ``flybrain_model_eval.plan_grouped_split`` so held-out labels never reach
    any node's features (transductive leakage through same-type partners).
    ``two_hop`` also removes each node's i->j->i return path, so a node's own
    category never enters its own features.

    R5: ``params.two_hop_min_weight`` / ``params.degree_transform`` (or the
    ``FLYBRAIN_WIRING_PROFILE`` env var / ``wiring_profile`` context, applied
    here before fingerprinting) select strong-edge 2-hop and rank-normalized
    degree features. NT objectives raise ``LabelLeakageError`` when the
    partner category is NT-derived.
    """
    rule = objective_exclusions(objective)  # unknown objective -> fail closed before any work
    params = apply_wiring_profile(params)
    if not re.match(r"^[a-z0-9_]+$", dataset or ""):
        raise ValueError("dataset must be a lowercase symbol")
    if id_column not in nodes.columns or category_column not in nodes.columns:
        raise ValueError(f"node table needs columns {id_column!r} and {category_column!r}")
    if nodes[id_column].isna().any():
        raise ValueError("node table has null ids")
    ids = [str(v) for v in nodes[id_column].tolist()]
    if len(set(ids)) != len(ids):
        raise ValueError("node table ids are not unique")
    harmonize = _harmonizer(vocab_map)
    raw_cats = nodes[category_column].tolist()
    cats = [harmonize(v) for v in raw_cats]
    # R5 hard block, before any work: NT objectives never see partner-NT features.
    category_is_nt = (partner_category_is_nt(category_column, raw_cats, category_tag=params.category_tag)
                      or partner_category_is_nt(category_column, cats, category_tag=params.category_tag))
    if category_is_nt and is_nt_objective(objective):
        assert_partner_category_allowed(objective, category_column, cats + raw_cats, category_tag=params.category_tag)
    masked = {str(v) for v in (mask_category_ids or ())}
    unknown_masks = masked - set(ids)
    if unknown_masks:
        raise ValueError(f"mask_category_ids not in the node table: {sorted(unknown_masks)[:5]}")
    cats = [UNANNOTATED if node_id in masked else cat for node_id, cat in zip(ids, cats)]
    vocab_digest = hashlib.sha256(json.dumps(sorted(set(zip(map(str, nodes[category_column].tolist()), cats))),
                                             sort_keys=True).encode("utf-8")).hexdigest()
    order = sorted(range(len(ids)), key=lambda i: (len(ids[i]), ids[i]))
    ids = [ids[i] for i in order]
    cats = [cats[i] for i in order]
    fingerprint = wiring_fingerprint(dataset=dataset, edges=edges, neuropil_edges=neuropil_edges,
                                     nodes_digest=_nodes_digest(ids, cats), params=params, vocab_digest=vocab_digest)
    cache_path: Path | None = None
    meta: dict[str, Any]
    full: pd.DataFrame | None = None
    if cache_root is not None:
        cache_dir = Path(cache_root) / "wiring-features" / dataset
        _reject_snapshot_path(cache_dir)
        cache_path = cache_dir / f"{fingerprint[:32]}.parquet"
        if use_cache and cache_path.is_file() and cache_path.with_suffix(".json").is_file():
            meta = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
            # A sidecar/parquet mismatch (partial write, tamper) is treated as a miss and rebuilt.
            if meta.get("fingerprint") == fingerprint and meta.get("parquet_sha256") == _sha256_file(cache_path):
                full = pd.read_parquet(cache_path)
    if full is None:
        frame, stats = compute_wiring_features(edges=edges, node_ids=ids, node_categories=cats, params=params,
                                               neuropil_edges=neuropil_edges)
        frame.insert(0, NODE_ID_COLUMN, ids)
        full = frame
        meta = {
            "schema_version": WIRING_FEATURES_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "dataset": dataset,
            "category_column": category_column,
            "params": params.as_dict(),
            "edges": edges.identity(),
            "neuropil_edges": None if neuropil_edges is None else neuropil_edges.identity(),
            "stats": stats,
            "masked_category_nodes": len(masked),
            "partner_category_nt": bool(category_is_nt),
            "r5": {"two_hop_min_weight": float(params.two_hop_min_weight),
                   "degree_transform": params.degree_transform},
            "columns": list(frame.columns),
        }
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".parquet.tmp")
            full.to_parquet(tmp, index=False)
            os.replace(tmp, cache_path)
            meta["parquet_sha256"] = _sha256_file(cache_path)
            cache_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    meta = {**meta, "partner_category_nt": bool(category_is_nt)}
    kept, dropped = apply_objective_exclusions(full, objective)
    return WiringFeatureResult(
        frame=kept.reset_index(drop=True),
        fingerprint=fingerprint,
        cache_path=None if cache_path is None else str(cache_path),
        objective=objective,
        excluded_features=tuple(dropped),
        meta={**meta, "exclusion_rule": rule.as_dict()},
        size_frame=size_columns_frame(full),
    )


def load_wiring_features(path: str | Path, *, objective: str) -> WiringFeatureResult:
    """Load a cached feature parquet (sidecar sha256 verified) with ``objective`` exclusions applied."""
    rule = objective_exclusions(objective)
    parquet = Path(path)
    meta = json.loads(parquet.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("parquet_sha256") and meta["parquet_sha256"] != _sha256_file(parquet):
        raise ValueError(f"cached wiring features hash mismatch: {parquet}")
    if is_nt_objective(objective):
        tag = str((meta.get("params") or {}).get("category_tag") or "")
        if meta.get("partner_category_nt") or partner_category_is_nt(
                str(meta.get("category_column") or ""), (meta.get("stats") or {}).get("categories") or (),
                category_tag=tag):
            raise LabelLeakageError(
                f"objective {objective!r} is an NT target and cached features {parquet.name} were built on an "
                f"NT-derived partner category ({meta.get('category_column')!r}); partner-NT features are banned "
                "from every NT model [SYNTHESIS F3; R5]")
    full = pd.read_parquet(parquet)
    kept, dropped = apply_objective_exclusions(full, objective)
    return WiringFeatureResult(frame=kept, fingerprint=str(meta["fingerprint"]), cache_path=str(parquet),
                               objective=objective, excluded_features=tuple(dropped),
                               meta={**meta, "exclusion_rule": rule.as_dict()}, size_frame=size_columns_frame(full))


def size_columns_frame(full: pd.DataFrame) -> pd.DataFrame:
    """Node id + the degree family and neuropil counts of a full (pre-exclusion) feature frame.

    These are size / degree proxies [R3; Bernett 2024]: the harness uses them for
    the size-only baseline and the degree-decile axis via EvalDataset.aux
    (never as model features), so an objective that excludes them as features
    (connectivity_tier) still gets a size bar.
    """
    cols = [c for c in full.columns if c != NODE_ID_COLUMN
            and (feature_family(c) == "degree" or c.endswith("__n_neuropils") or c.endswith("__n_neuropils_rank"))]
    return full[[NODE_ID_COLUMN, *cols]].reset_index(drop=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "size_columns_frame",
    "DEFAULT_CACHE_ROOT",
    "EdgeSource",
    "GLOBAL_EXCLUDED_PATTERNS",
    "LabelLeakageError",
    "NODE_ID_COLUMN",
    "ObjectiveExclusion",
    "SUPER_CLASS_VOCAB",
    "UNANNOTATED",
    "UNKNOWN",
    "WIRING_FEATURES_SCHEMA_VERSION",
    "WiringFeatureParams",
    "WiringFeatureResult",
    "DEGREE_TRANSFORMS",
    "WIRING_PROFILE_ENV",
    "active_wiring_profile",
    "apply_wiring_profile",
    "parse_wiring_profile",
    "wiring_profile",
    "NT_OBJECTIVES",
    "NT_VALUE_VOCAB",
    "R5_SENSITIVITY_TWO_HOP_MIN_WEIGHT",
    "R5_TWO_HOP_MIN_WEIGHT",
    "apply_objective_exclusions",
    "assert_features_allowed",
    "assert_partner_category_allowed",
    "is_nt_objective",
    "partner_category_is_nt",
    "percentile_rank",
    "register_nt_objective",
    "robust_wiring_params",
    "build_wiring_features",
    "compute_wiring_features",
    "excluded_feature_names",
    "feature_families",
    "feature_family",
    "harmonize_super_class",
    "load_wiring_features",
    "objective_exclusions",
    "register_objective_exclusions",
    "registered_objectives",
    "slug",
    "wiring_fingerprint",
]
