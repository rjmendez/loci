"""FlyWire 783 (``fw``) real-model targets: labels, groups, wiring features, eval datasets.

The legacy fw objectives in ``flybrain_brain_cluster_fw_samples`` were circular
(the label was a threshold on / argmax of numbers written into the model's own
``input_text``). This module builds non-circular targets instead:

========================================  ===========================================  ===============
target (= label-exclusion objective)      label                                        partner category
========================================  ===========================================  ===============
``super_class``                           Schlegel 2024 ``super_class``                super_class (masked)
``super_class_no_neuropil``               same, neuropil features removed as well      super_class (masked)
``flow``                                  Schlegel 2024 ``flow``                       super_class (masked)
``cell_class``                            Schlegel 2024 ``cell_class`` (>= N neurons)  cell_class (masked)
``cell_class_no_neuropil``                same, neuropil features removed as well      cell_class (masked)
``neurotransmitter_dominance``            Schlegel ``top_nt`` = per-neuron argmax of   super_class
                                          Eckstein 2024 synapse NT *predictions*
                                          (NOT ground truth)
``nt_ground_truth``                       Schlegel ``known_nt`` (literature), single   super_class
                                          classical transmitter only
``connectivity_tier``                     top quartile of the neuron's total           super_class
                                          presynapse count (per-neuron neuropil counts)
``hemilineage``                           Schlegel 2024 ``ito_lee_hemilineage``        hemilineage (masked)
                                          (cell-body-fibre tracts; the headline
                                          hemilineage target, curated_morphology)
``nt_literature`` (+ ``_binary``,         drosophila_neurotransmitters, conf >= 4,     super_class
``_all``, ``_all_binary``)                mapped by cell_type (R2;
                                          ``flybrain_nt_ground_truth``)
========================================  ===========================================  ===============

Every target carries a ``label_provenance`` (R1, ``flybrain_target_registry``);
evaluations go through ``ftr.run_gated_evaluation``, so the gate can only pass
measured / curated-morphology targets. ``neurotransmitter_dominance`` is a
distillation of the Eckstein 2024 classifier and ``connectivity_tier`` a
connectivity-derived statistic: both are reported, never gated.

Features are ``flybrain_wiring_features`` streamed over
``proofread_connections_783.feather`` (pre, post, neuropil, syn_count; one row
per (pre, post, neuropil), so ``unique_pairs=False``) with 2-hop composition.
Label-defining columns are removed by the objective's registered exclusions.
When the partner category is (or determines) the label, it is masked before
the features are built (``plan_grouped_split`` -> ``mask_category_ids`` ->
``masked_split_ids_sha256``) for every node that shares a held-out sample's
cell type, hemibrain type, type family or (except for the hemilineage target)
hemilineage, not only for the sampled val/test neurons: an unsampled neuron of
a held-out type would otherwise reveal the label through same-type partners.

Groups: ``cell_type``, ``hemilineage`` (ito_lee; the catch-all
``putative_primary`` and ambiguous ``*_or_*`` values are not group keys),
``hemibrain_type`` and ``type_family`` (union-find), so no cell type,
hemilineage, hemibrain type or sister-type family straddles train/val/test.
``type_family`` closes the sister-type leak behind the first
``nt_ground_truth`` run (val accuracy 0.98 vs test 0.10): optic-lobe neurons
have no ito_lee hemilineage and no hemibrain type, so ``T5c`` and ``C3`` sat in
val while ``T4a-d`` / ``T5a,b,d`` and ``C2`` were in train, and one Kenyon-cell
component (1,979 of 2,825 test rows) was the whole test set. The per-component
cap (``max_per_component``) removes the second problem. The ``hemilineage``
target cannot group by its own label and uses the other three keys.

Snapshot access is read-only and fail-closed: ``open_fw_snapshot`` checks the
manifest sidecar, the manifest self-hash, the verification status and refresh
decision, every listed path/size, then content-hashes the products it needs,
keeping hash stamps OUTSIDE the snapshot (default ``<root>/cache/hash-stamps/fw``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

import flybrain_nt_ground_truth as ntgt
import flybrain_target_registry as ftr
import flybrain_wiring_features as fwf
from flybrain_hash_stamps import HashStampCache, sha256_file, verify_file_sha256

FW_SNAPSHOT_RELATIVE_ROOT = "snapshots/fw/flywire783"
FW_MANIFEST_RELATIVE_PATH = "manifest/manifest.json"
FW_MANIFEST_SIDECAR_RELATIVE_PATH = "manifest/manifest.sha256"
DEFAULT_STORAGE_ROOT = "/mnt/f/.flybrain"
DEFAULT_REPORT_ROOT = "/mnt/f/.flybrain/logs/real-models-20260924T174122Z"

ROLE_CONNECTIONS = "edgelist_per_neuropil"
ROLE_ANNOTATIONS = "neuron_annotations_hierarchical"
ROLE_PRE_COUNTS = "per_neuron_neuropil_count_pre"
FW_PRODUCT_PATHS: dict[str, str] = {
    ROLE_CONNECTIONS: "source/proofread_connections_783.feather",
    ROLE_ANNOTATIONS: "source/nature_schlegel2024/Supplementary_Data_1_neuron_annotations.tsv",
    ROLE_PRE_COUNTS: "source/per_neuron_neuropil_count_pre_783.feather",
}

GROUP_COLUMNS: tuple[str, ...] = ("cell_type", "hemilineage", "hemibrain_type", "type_family")
HEMILINEAGE_GROUP_COLUMNS: tuple[str, ...] = ("cell_type", "hemibrain_type", "type_family")
# Catch-all hemilineage values: never a label and never a split key.
HEMILINEAGE_SENTINELS = frozenset({"putative_primary", "primary", "unknown", "na", "no_lineage", "tbd"})
# Sister types from one developmental origin whose names do not share a stem.
TYPE_FAMILY_MERGES: Mapping[str, str] = {"T4": "T4/T5", "T5": "T4/T5"}
CLASSICAL_NT: tuple[str, ...] = ("acetylcholine", "glutamate", "gaba", "dopamine", "serotonin", "octopamine",
                                 "histamine")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_BLOCKED_SUFFIXES = (".partial", ".tmp", ".inprogress")


class FwSnapshotError(RuntimeError):
    """The fw snapshot failed a fail-closed integrity check."""


# =========================================================================== snapshot


@dataclass(frozen=True)
class FwSnapshot:
    snapshot_root: Path
    manifest_sha256: str
    product_paths: Mapping[str, Path]
    product_sha256: Mapping[str, str]
    hash_verification: Mapping[str, str]
    warnings: tuple[str, ...] = ()

    def path(self, role: str) -> Path:
        if role not in self.product_paths:
            raise FwSnapshotError(f"role {role!r} was not opened (required_roles)")
        return self.product_paths[role]

    def provenance(self, role: str) -> dict[str, str]:
        return {"manifest_sha256": self.manifest_sha256, "sha256": self.product_sha256[role],
                "relative_path": FW_PRODUCT_PATHS[role]}


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Self-hash rule from FLYBRAIN_HARNESS_MANIFEST_PROVENANCE_SCHEMA.md (same as the other adapters)."""
    canonical = json.loads(json.dumps(manifest))
    integrity = canonical.get("integrity")
    if not isinstance(integrity, dict):
        raise FwSnapshotError("manifest.integrity must be an object")
    integrity["manifest_sha256"] = ""
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _inside_snapshots(path: Path) -> bool:
    return "snapshots" in Path(path).resolve(strict=False).parts


def open_fw_snapshot(
    storage_root: str | Path = DEFAULT_STORAGE_ROOT,
    *,
    required_roles: Sequence[str] = tuple(FW_PRODUCT_PATHS),
    stamp_dir: str | Path | None = None,
    verify_hashes: bool | None = None,
    now: datetime | None = None,
) -> FwSnapshot:
    """Verify the pinned flywire783 snapshot and return the required product paths.

    ``verify_hashes``: None hashes the required products (stamp cache honoured),
    True re-hashes them ignoring stamps, False checks sizes only. Stamps are
    never written inside ``snapshots/`` (the default stamp dir is
    ``<storage_root>/cache/hash-stamps/fw``).
    """
    unknown = [r for r in required_roles if r not in FW_PRODUCT_PATHS]
    if unknown:
        raise FwSnapshotError(f"unknown fw product roles: {unknown}")
    root = Path(storage_root).resolve(strict=False)
    snapshot_root = root / FW_SNAPSHOT_RELATIVE_ROOT
    manifest_path = snapshot_root / FW_MANIFEST_RELATIVE_PATH
    sidecar_path = snapshot_root / FW_MANIFEST_SIDECAR_RELATIVE_PATH
    if not manifest_path.is_file():
        raise FwSnapshotError(f"fw manifest not found: {manifest_path}")
    if not sidecar_path.is_file():
        raise FwSnapshotError(f"fw manifest.sha256 sidecar is missing: {sidecar_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FwSnapshotError(f"fw manifest unreadable: {exc}") from exc
    if not isinstance(manifest, dict):
        raise FwSnapshotError("fw manifest must be a JSON object")
    integrity = manifest.get("integrity") if isinstance(manifest.get("integrity"), dict) else {}
    expected = str(integrity.get("manifest_sha256") or "")
    if not _SHA_RE.fullmatch(expected):
        raise FwSnapshotError("integrity.manifest_sha256 must be lowercase 64-hex")
    if sidecar_path.read_text(encoding="utf-8").strip() != expected:
        raise FwSnapshotError("manifest.sha256 sidecar does not match manifest")
    actual = _manifest_digest(manifest)
    if actual != expected:
        raise FwSnapshotError(f"manifest self-hash mismatch (expected {expected}, actual {actual})")
    status = str((integrity.get("verification") or {}).get("status") or "")
    if status != "verified":
        raise FwSnapshotError(f"manifest verification status must be 'verified' (got {status!r})")
    dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), dict) else {}
    if dataset.get("symbol") != "fw" or dataset.get("version_id") != "flywire783":
        raise FwSnapshotError("manifest is not pinned to fw/flywire783")
    refresh = manifest.get("refresh") if isinstance(manifest.get("refresh"), dict) else {}
    if str(refresh.get("decision") or "").lower() == "rollback":
        raise FwSnapshotError("refresh.decision indicates rollback")
    due = refresh.get("next_check_due")
    if due:
        due_dt = datetime.fromisoformat(str(due).replace("Z", "+00:00"))
        if due_dt < (now or datetime.now(timezone.utc)):
            raise FwSnapshotError(f"refresh.next_check_due has elapsed ({due})")

    files = integrity.get("files")
    if not isinstance(files, list) or not files:
        raise FwSnapshotError("integrity.files must be a non-empty list")
    listed: dict[str, tuple[Path, str]] = {}
    for entry in files:
        rel = str((entry or {}).get("relative_path") or "").replace("\\", "/").strip()
        sha = str((entry or {}).get("sha256") or "").strip()
        size = (entry or {}).get("size_bytes")
        if not rel or rel.startswith("/") or ".." in rel.split("/") or ":" in rel:
            raise FwSnapshotError(f"unsafe integrity path {rel!r}")
        if rel.lower().endswith(_BLOCKED_SUFFIXES):
            raise FwSnapshotError(f"partial/temporary artifact listed: {rel}")
        if rel.casefold() in {k.casefold() for k in listed}:
            raise FwSnapshotError(f"duplicate integrity path {rel}")
        if not _SHA_RE.fullmatch(sha):
            raise FwSnapshotError(f"file sha256 must be lowercase 64-hex: {rel}")
        path = (snapshot_root / rel).resolve(strict=False)
        if not path.is_relative_to(snapshot_root.resolve(strict=False)):
            raise FwSnapshotError(f"integrity path escapes the snapshot: {rel}")
        if not path.is_file():
            raise FwSnapshotError(f"integrity file is missing: {rel}")
        if isinstance(size, bool) or not isinstance(size, int) or path.stat().st_size != size:
            raise FwSnapshotError(f"file size mismatch: {rel}")
        listed[rel] = (path, sha)

    stamp_root = Path(stamp_dir) if stamp_dir is not None else root / "cache" / "hash-stamps" / "fw"
    if _inside_snapshots(stamp_root):
        raise FwSnapshotError(f"refusing to keep hash stamps inside a snapshots directory: {stamp_root}")
    cache = HashStampCache(stamp_root)
    report: dict[str, str] = {}
    warnings: list[str] = []
    paths: dict[str, Path] = {}
    shas: dict[str, str] = {}
    for role in required_roles:
        rel = FW_PRODUCT_PATHS[role]
        if rel not in listed:
            raise FwSnapshotError(f"manifest does not list required product {role!r} ({rel})")
        path, sha = listed[rel]
        if verify_hashes is False:
            report[rel] = "size_only"
        else:
            check = verify_file_sha256(path, relative_path=rel, expected_sha256=sha, cache=cache,
                                       force=verify_hashes is True, hasher=sha256_file)
            if not check.matched:
                raise FwSnapshotError(f"sha256 mismatch for {rel}: expected {sha}, got {check.actual_sha256}")
            report[rel] = check.method
            if check.warning:
                warnings.append(check.warning)
        paths[role] = path
        shas[role] = sha
    return FwSnapshot(snapshot_root, expected, paths, shas, dict(sorted(report.items())), tuple(warnings))


# =========================================================================== annotations + labels

_ANNOTATION_COLUMNS = ("root_id", "flow", "super_class", "cell_class", "cell_sub_class", "cell_type",
                       "hemibrain_type", "ito_lee_hemilineage", "top_nt", "known_nt", "known_nt_source",
                       "side", "status")


def type_family(cell_type: Any) -> str | None:
    """Sister-type family used as an extra split key (conservative: it only ever merges groups).

    A hyphenated suffix is dropped (``KCapbp-ap2`` -> ``KCapbp``), then trailing
    lower-case letters after a digit, capital or ``)`` (``T4a`` -> ``T4``,
    ``KCab`` -> ``KC``, ``Dm3a`` -> ``Dm3``, ``(M_lPNm12,M_lPNm13)a`` -> ``(M_lPNm12,M_lPNm13)``),
    then ``TYPE_FAMILY_MERGES`` (T4/T5 share their progenitors). ``LHPV2a1`` and
    ``ORN_VC5`` are unchanged.
    """
    if cell_type is None or (isinstance(cell_type, float) and cell_type != cell_type):
        return None
    text = str(cell_type).strip()
    if not text or text.lower() in {"na", "nan", "none", "unknown"}:
        return None
    stem = text.split("-", 1)[0] if not text.startswith("(") else text
    stem = re.sub(r"_[a-z]$", "", stem)  # LHPV2a1_a / LHPV2a1_b
    stem = re.sub(r"(?<=[0-9A-Z)])[a-z]+$", "", stem) or text
    return "fam:" + TYPE_FAMILY_MERGES.get(stem, stem)


def _lineage_group(value: Any) -> str | None:
    """Hemilineage as a label / split key: sentinels and ambiguous ``A_or_B`` assignments -> None."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    text = str(value).strip()
    if not text or text.lower() in HEMILINEAGE_SENTINELS or "_or_" in text.lower():
        return None
    return text


def load_fw_annotations(path: str | Path) -> pd.DataFrame:
    """Schlegel 2024 neuron annotations (one row per proofread neuron), ids as int64.

    ``hemilineage`` holds only real lineage assignments (label and split key);
    the file's value is kept verbatim in ``hemilineage_raw``. ``type_family``
    is the sister-type split key (``type_family``).
    """
    frame = pd.read_csv(path, sep="\t", usecols=list(_ANNOTATION_COLUMNS), dtype=str, keep_default_na=False,
                        na_values=[""])
    frame["root_id"] = frame["root_id"].astype("int64")
    if not frame["root_id"].is_unique:
        raise ValueError("annotation root_id is not unique")
    frame = frame.rename(columns={"ito_lee_hemilineage": "hemilineage"})
    for col in ("cell_type", "hemibrain_type", "hemilineage"):
        frame[col] = frame[col].where(frame[col].notna() & ~frame[col].isin(["na", "NA", "unknown"]), None)
    frame["hemilineage_raw"] = frame["hemilineage"]
    frame["hemilineage"] = frame["hemilineage"].map(_lineage_group)
    frame["type_family"] = frame["cell_type"].map(type_family)
    return frame.sort_values("root_id", kind="mergesort").reset_index(drop=True)


def parse_known_nt(value: Any) -> str | None:
    """Single classical transmitter from a ``known_nt`` string; None when absent or ambiguous."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    tokens = [t.strip().lower() for t in str(value).split(",")]
    classical = sorted({t for t in tokens if t in CLASSICAL_NT})
    return classical[0] if len(classical) == 1 else None


def total_presynapse_counts(pre_counts_path: str | Path) -> pd.Series:
    """Total presynapse count per neuron (sum over neuropils) from per_neuron_neuropil_count_pre."""
    import pyarrow.feather as feather

    table = feather.read_table(str(pre_counts_path), columns=["pre_pt_root_id", "count"]).to_pandas()
    return table.groupby("pre_pt_root_id", sort=True)["count"].sum().astype("int64")


def _label_super_class(ann: pd.DataFrame, _ctx: Mapping[str, Any]) -> pd.Series:
    return ann["super_class"].map(lambda v: None if pd.isna(v) else fwf.harmonize_super_class(v))


def _label_flow(ann: pd.DataFrame, _ctx: Mapping[str, Any]) -> pd.Series:
    return ann["flow"].map(lambda v: None if pd.isna(v) else fwf.slug(v))


def _label_cell_class(ann: pd.DataFrame, _ctx: Mapping[str, Any]) -> pd.Series:
    return ann["cell_class"].map(lambda v: None if pd.isna(v) else fwf.slug(v))


def _label_top_nt(ann: pd.DataFrame, _ctx: Mapping[str, Any]) -> pd.Series:
    return ann["top_nt"].map(lambda v: None if pd.isna(v) or str(v).lower() not in CLASSICAL_NT else str(v).lower())


def _label_known_nt(ann: pd.DataFrame, _ctx: Mapping[str, Any]) -> pd.Series:
    return ann["known_nt"].map(parse_known_nt)


def _label_hemilineage(ann: pd.DataFrame, _ctx: Mapping[str, Any]) -> pd.Series:
    return ann["hemilineage"].map(lambda v: None if v is None or (isinstance(v, float) and v != v) else str(v))


def _label_nt_literature(ann: pd.DataFrame, ctx: Mapping[str, Any]) -> pd.Series:
    labels: Mapping[int, str] = ctx["nt_literature_labels"]
    return ann["root_id"].map(lambda r: labels.get(int(r)))


def connectivity_tier_labels(totals: pd.Series, quantile: float = 0.75) -> tuple[pd.Series, float]:
    threshold = float(np.quantile(totals.to_numpy(dtype=float), quantile))
    labels = np.where(totals.to_numpy(dtype=float) >= threshold, "high_connectivity", "baseline_connectivity")
    return pd.Series(labels, index=totals.index, dtype=object), threshold


def _label_connectivity(ann: pd.DataFrame, ctx: Mapping[str, Any]) -> pd.Series:
    totals = ctx["presynapse_totals"].reindex(ann["root_id"]).fillna(0).astype("int64")
    totals.index = ann.index
    labels, threshold = connectivity_tier_labels(totals, float(ctx.get("quantile", 0.75)))
    ctx["connectivity_threshold"] = threshold  # type: ignore[index]
    return labels


# =========================================================================== target specs


def _register_strict_objectives() -> None:
    region = fwf.objective_exclusions("region_specialization_tier").patterns
    for name, base in (("super_class_no_neuropil", "super_class"), ("cell_class_no_neuropil", "cell_class")):
        if name in fwf.registered_objectives():
            continue
        rule = fwf.objective_exclusions(base)
        fwf.register_objective_exclusions(
            name, tuple(rule.patterns) + tuple(region),
            reason=(f"{rule.reason}; additionally every neuropil/region feature, because fw super_class and "
                    "optic cell_class ('ME>LO', 'LA>ME', ...) are defined by where a neuron's arbors lie"))


_register_strict_objectives()


@dataclass(frozen=True)
class FwTargetSpec:
    target: str
    label_fn: Callable[[pd.DataFrame, Mapping[str, Any]], pd.Series]
    category_column: str
    vocab: Any
    mask_heldout: bool
    label_provenance: str  # R1: measured / curated_morphology / connectivity_defined / model_predicted
    description: str
    min_class_count: int = 100
    # classes with fewer distinct grouped components cannot be learned under a grouped split
    min_class_components: int = 1
    max_per_class: int = 10_000
    needs_pre_counts: bool = False
    needs_nt_literature: bool = False
    # A few giant cell types (photoreceptors, T4/T5, Kenyon cells) would otherwise fill whole
    # classes and whole test splits; cap each grouped component so many components are sampled.
    max_per_component: int = 200
    group_columns: tuple[str, ...] = GROUP_COLUMNS

    @property
    def ground_truth(self) -> bool:
        """Back-compat flag: True only for measured labels."""
        return self.label_provenance == ftr.LabelProvenance.MEASURED.value


_P = ftr.LabelProvenance
_NT_LIT_SPECS = tuple(
    FwTargetSpec(target, _label_nt_literature, "super_class", fwf.harmonize_super_class, False, _P.MEASURED.value,
                 spec.describe(), min_class_count=20, min_class_components=3, needs_nt_literature=True,
                 max_per_component=50)
    for target, spec in ntgt.NT_LITERATURE_SPECS.items())

FW_TARGETS: dict[str, FwTargetSpec] = {
    spec.target: spec
    for spec in (
        FwTargetSpec("super_class", _label_super_class, "super_class", fwf.harmonize_super_class, True,
                     _P.CURATED_MORPHOLOGY.value,
                     "Schlegel 2024 super_class (harmonized vocab) from wiring incl. neuropil fractions"),
        FwTargetSpec("super_class_no_neuropil", _label_super_class, "super_class", fwf.harmonize_super_class, True,
                     _P.CURATED_MORPHOLOGY.value,
                     "super_class from partner composition / reciprocity / 2-hop / degree only"),
        FwTargetSpec("flow", _label_flow, "super_class", fwf.harmonize_super_class, True, _P.CURATED_MORPHOLOGY.value,
                     "Schlegel 2024 flow (afferent/intrinsic/efferent)"),
        FwTargetSpec("cell_class", _label_cell_class, "cell_class", None, True, _P.CURATED_MORPHOLOGY.value,
                     "Schlegel 2024 cell_class (classes with >= 100 neurons); optic classes are neuropil "
                     "paths, so neuropil features are near-definitional here", max_per_class=4000),
        FwTargetSpec("cell_class_no_neuropil", _label_cell_class, "cell_class", None, True,
                     _P.CURATED_MORPHOLOGY.value, "cell_class without any neuropil feature", max_per_class=4000),
        FwTargetSpec("hemilineage", _label_hemilineage, "hemilineage", None, True, _P.CURATED_MORPHOLOGY.value,
                     "Schlegel 2024 ito_lee_hemilineage (cell-body-fibre tracts; putative_primary and 'A_or_B' "
                     "assignments dropped); partner hemilineage composition masked for held-out groups",
                     min_class_count=30, min_class_components=3, max_per_class=2000, max_per_component=100,
                     group_columns=HEMILINEAGE_GROUP_COLUMNS),
        FwTargetSpec("neurotransmitter_dominance", _label_top_nt, "super_class", fwf.harmonize_super_class, False,
                     _P.MODEL_PREDICTED.value, "top_nt = per-neuron argmax of Eckstein 2024 synapse-level NT "
                                               "PREDICTIONS (label is a model output, not ground truth)"),
        FwTargetSpec("nt_ground_truth", _label_known_nt, "super_class", fwf.harmonize_super_class, False,
                     _P.MEASURED.value, "known_nt (literature / Schlegel 2024), single classical transmitter only; "
                                        "superseded by nt_literature", min_class_count=50, min_class_components=3,
                     max_per_component=50),
        FwTargetSpec("connectivity_tier", _label_connectivity, "super_class", fwf.harmonize_super_class, False,
                     _P.CONNECTIVITY_DEFINED.value, "top quartile of total presynapse count "
                     "(per_neuron_neuropil_count_pre); degree / count / n_neuropils features excluded",
                     needs_pre_counts=True),
        *_NT_LIT_SPECS,
    )
}


# Every fw target spec must agree with the central registry (fail closed at import).
ftr.check_module_provenance("fw", {name: spec.label_provenance for name, spec in FW_TARGETS.items()})


def _hash_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def component_ids(frame: pd.DataFrame, group_columns: Sequence[str] = GROUP_COLUMNS) -> list[str]:
    """Union-find component per row (same rule as the grouped split)."""
    from flybrain_brain_cluster_training import split_group_components
    import flybrain_model_eval as fme

    ids = frame["root_id"].astype(str).tolist()
    comp = split_group_components(fme._light(ids, frame[list(group_columns)].to_dict(orient="records")),
                                  group_keys=tuple(group_columns))
    return [comp[i] for i in ids]


def select_target_rows(ann: pd.DataFrame, spec: FwTargetSpec, ctx: Mapping[str, Any]) -> pd.DataFrame:
    """Labelled rows for ``spec``: drop outliers / unlabeled / rare classes, cap big classes by whole components."""
    frame = ann.copy()
    frame["label"] = spec.label_fn(frame, ctx).to_numpy()
    frame = frame[frame["status"].isna()]  # outlier_seg / outlier_bio neurons are not samples
    frame = frame[frame["label"].notna() & ~frame["label"].isin([fwf.UNKNOWN, "unknown", "non_neuronal"])]
    counts = frame["label"].value_counts()
    frame = frame[frame["label"].isin(counts[counts >= int(spec.min_class_count)].index)].copy()
    frame["_component"] = component_ids(frame, spec.group_columns)
    if int(spec.min_class_components) > 1:
        comps = frame.groupby("label")["_component"].nunique()
        frame = frame[frame["label"].isin(comps[comps >= int(spec.min_class_components)].index)].copy()
    frame["_order"] = [(_hash_key(f"fw-cap:{c}"), _hash_key(f"fw-cap:{r}"))
                       for c, r in zip(frame["_component"], frame["root_id"].astype(str))]
    frame = frame.sort_values("_order", kind="mergesort")
    frame = frame.groupby("_component", sort=False).head(int(spec.max_per_component))
    kept = []
    for label, part in frame.groupby("label", sort=True):
        part = part.sort_values("_order", kind="mergesort")
        kept.append(part.head(int(spec.max_per_class)))
    if not kept:
        raise ValueError(f"{spec.target}: no class survives the selection filters")
    out = pd.concat(kept).sort_values("root_id", kind="mergesort").reset_index(drop=True)
    return out.drop(columns=["_order"])


def held_out_mask_ids(annotations: pd.DataFrame, rows: pd.DataFrame, held_out: Sequence[str],
                      group_columns: Sequence[str]) -> list[str]:
    """Held-out samples + every annotated node sharing any of their group values (cell type, lineage, ...)."""
    held = rows[rows["root_id"].astype(str).isin(set(held_out))]
    hit = np.zeros(len(annotations), dtype=bool)
    for column in group_columns:
        values = set(held[column].dropna())
        if values:
            hit |= annotations[column].isin(values).to_numpy()
    ids = set(annotations.loc[hit, "root_id"].astype(str)) | set(held_out)
    return sorted(ids)


def nt_literature_labels(annotations: pd.DataFrame, target: str, *,
                         nt_root: str | Path = ntgt.DEFAULT_NT_GT_ROOT) -> tuple[dict[int, str], dict[str, Any]]:
    """root_id -> literature NT for an ``nt_literature*`` target + coverage statistics (R2)."""
    source = ntgt.open_nt_ground_truth(nt_root)
    labelled = ntgt.label_neurons(annotations[["root_id", "cell_type", "hemibrain_type"]], dataset="fw",
                                  source=source, target=target, id_column="root_id",
                                  secondary_type_column="hemibrain_type")
    return {int(r): str(l) for r, l in zip(labelled.frame["root_id"], labelled.frame["label"])}, labelled.coverage


# =========================================================================== text view (nb)


def binned_text(features: pd.DataFrame, fit_rows: np.ndarray, n_bins: int = 10) -> list[str]:
    """``name name__qK`` tokens with decile edges fitted on ``fit_rows`` (train) only.

    The value token carries the feature name so the NB tokenizer (``[a-z0-9_]+``)
    keeps it distinct per feature; missing values become ``name__na``.
    """
    cols = sorted(features.columns)
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    parts: list[list[str]] = []
    for name in cols:
        values = features[name].to_numpy(dtype=float)
        train = values[fit_rows]
        train = train[np.isfinite(train)]
        edges = np.unique(np.quantile(train, qs)) if len(train) else np.zeros(0)
        bins = np.searchsorted(edges, values, side="right")
        tokens = [f"{name} {name}__q{b}" if np.isfinite(v) else f"{name} {name}__na" for v, b in zip(values, bins)]
        parts.append(tokens)
    return [" ".join(row) for row in zip(*parts)] if parts else [""] * len(features)


# =========================================================================== eval dataset


@dataclass
class FwBuild:
    data: Any  # flybrain_model_eval.EvalDataset
    plan: Mapping[str, Sequence[str]]
    features: fwf.WiringFeatureResult
    rows: pd.DataFrame
    info: dict[str, Any] = field(default_factory=dict)


def edge_source(snapshot: FwSnapshot) -> fwf.EdgeSource:
    return fwf.EdgeSource(path=str(snapshot.path(ROLE_CONNECTIONS)), pre="pre_pt_root_id", post="post_pt_root_id",
                          weight="syn_count", neuropil="neuropil", unique_pairs=False, format="ipc",
                          provenance=snapshot.provenance(ROLE_CONNECTIONS))


def build_fw_eval_dataset(
    target: str,
    *,
    snapshot: FwSnapshot,
    annotations: pd.DataFrame,
    config: Any,
    presynapse_totals: pd.Series | None = None,
    cache_root: str | Path = fwf.DEFAULT_CACHE_ROOT,
    params: fwf.WiringFeatureParams | None = None,
    with_text: bool = True,
    nt_root: str | Path = ntgt.DEFAULT_NT_GT_ROOT,
) -> FwBuild:
    """Samples + masked wiring features + grouped split for one fw target (the REQUIRED mask pattern)."""
    import flybrain_model_eval as fme

    spec = FW_TARGETS[target]
    prov = ftr.target_provenance("fw", target)  # fail closed before any work
    ctx: dict[str, Any] = {"presynapse_totals": presynapse_totals}
    if spec.needs_pre_counts and presynapse_totals is None:
        raise ValueError(f"{target} needs presynapse_totals")
    coverage = None
    if spec.needs_nt_literature:
        ctx["nt_literature_labels"], coverage = nt_literature_labels(annotations, target, nt_root=nt_root)
    rows = select_target_rows(annotations, spec, ctx)
    ids = rows["root_id"].astype(str).tolist()
    keys = tuple(spec.group_columns)
    groups = rows[list(keys)].to_dict(orient="records")
    plan = fme.plan_grouped_split(ids, groups, keys, config)
    mask = held_out_mask_ids(annotations, rows, list(plan["val"]) + list(plan["test"]), keys) \
        if spec.mask_heldout else None
    feats = fwf.build_wiring_features(
        dataset="fw", objective=target, edges=edge_source(snapshot), nodes=annotations[["root_id", spec.category_column]],
        id_column="root_id", category_column=spec.category_column, vocab_map=spec.vocab,
        params=params or fwf.WiringFeatureParams(two_hop=True), cache_root=cache_root, mask_category_ids=mask)
    fcols = [c for c in feats.frame.columns if c != fwf.NODE_ID_COLUMN]
    # constant columns carry no signal and confuse the trivial stumps' tie-breaks
    feat = feats.frame.set_index(fwf.NODE_ID_COLUMN).reindex(ids)
    fcols = [c for c in fcols if feat[c].nunique(dropna=False) > 1]
    frame = rows.reset_index(drop=True).copy()
    frame["root_id"] = frame["root_id"].astype(str)
    frame = pd.concat([frame, feat[fcols].reset_index(drop=True)], axis=1)
    notes: dict[str, Any] = {
        "label_source": spec.description,
        **prov.notes(),
        "label_is_ground_truth": spec.ground_truth,
        "group_keys": list(keys),
        "partner_category": spec.category_column,
        "partner_category_masked_for_val_test": spec.mask_heldout,
        "masked_category_nodes": None if mask is None else len(mask),
        "mask_rule": None if mask is None else ("held-out samples + every annotated node sharing their "
                                                + ", ".join(keys)),
        "components": {"selected": int(rows["_component"].nunique()),
                       "per_label": {str(k): int(v) for k, v in
                                     rows.groupby("label")["_component"].nunique().sort_index().items()}},
        "features_fingerprint": feats.fingerprint,
        "features_cache": feats.cache_path,
        "excluded_features": list(feats.excluded_features),
        "fw_manifest_sha256": snapshot.manifest_sha256,
        "product_sha256": dict(snapshot.product_sha256),
        "min_class_count": spec.min_class_count,
        "max_per_class": spec.max_per_class,
        "max_per_component": spec.max_per_component,
        "cap_rule": ("<= max_per_component rows per grouped component (sha256 order), then per class whole "
                     "(capped) components in sha256('fw-cap:'+component) order up to max_per_class"),
        "label_counts_selected": rows["label"].value_counts().sort_index().to_dict(),
    }
    if spec.mask_heldout:
        notes["masked_split_ids_sha256"] = fme.split_ids_sha256(plan)
    if "connectivity_threshold" in ctx:
        notes["connectivity_threshold_presynapses"] = ctx["connectivity_threshold"]
    if coverage is not None:
        notes["nt_literature_coverage"] = coverage
    text_col = None
    if with_text:
        position = {sid: i for i, sid in enumerate(ids)}
        train_rows = np.asarray(sorted(position[s] for s in plan["train"]), dtype=np.int64)
        frame["input_text"] = binned_text(frame[fcols], train_rows)
        text_col = "input_text"
        notes["text_view"] = "decile-binned feature tokens, edges fitted on train rows only"
    data = fme.EvalDataset.from_frame(frame, dataset="fw", target=target, id_column="root_id", label_column="label",
                                      feature_columns=fcols, group_columns=list(keys),
                                      text_column=text_col, notes=notes)
    data.aux = fme.size_side_aux(data.features, size_frame=feats.size_frame, id_column=fwf.NODE_ID_COLUMN,
                                 ids=frame["root_id"], side=frame["side"] if "side" in frame.columns else None)
    data.__post_init__()  # re-validate aux against the features
    return FwBuild(data=data, plan=plan, features=feats, rows=rows, info={"n_features": len(fcols)})


# =========================================================================== transfer bridge


def hemibrain_type_group(value: Any) -> str | None:
    """Canonical hemibrain type group: sorted, de-duplicated comma tokens joined by '|'."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    tokens = sorted({t.strip() for t in re.split(r"[,;]", str(value).strip("()")) if t.strip()})
    return "|".join(tokens) or None


def write_bridge_table(annotations: pd.DataFrame, *, snapshot: FwSnapshot, features_fingerprint: str,
                       features_cache: str | None, cache_root: str | Path = fwf.DEFAULT_CACHE_ROOT) -> Path:
    """hemibrain_type bridging columns keyed like the wiring-feature cache (``node_id``) for the transfer track.

    Not a training input: these columns are identifiers of the curated type
    hierarchy and are excluded as features for every type-level objective.
    """
    out_dir = Path(cache_root) / "wiring-features" / "fw" / "bridge"
    if _inside_snapshots(out_dir):
        raise ValueError(f"refusing to write under snapshots: {out_dir}")
    frame = pd.DataFrame({
        "node_id": annotations["root_id"].astype(str),
        "hemibrain_type": annotations["hemibrain_type"],
        "hemibrain_type_group": annotations["hemibrain_type"].map(hemibrain_type_group),
        "cell_type": annotations["cell_type"],
        "cell_class": annotations["cell_class"],
        "super_class": annotations["super_class"].map(lambda v: None if pd.isna(v) else fwf.harmonize_super_class(v)),
        "flow": annotations["flow"],
        "hemilineage": annotations.get("hemilineage_raw", annotations["hemilineage"]),
        "side": annotations["side"],
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{features_fingerprint[:32]}.bridge.parquet"
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    meta = {
        "schema_version": "flybrain-fw-bridge/v1",
        "features_fingerprint": features_fingerprint,
        "features_cache": features_cache,
        "fw_manifest_sha256": snapshot.manifest_sha256,
        "annotations_sha256": snapshot.product_sha256.get(ROLE_ANNOTATIONS),
        "rows": int(len(frame)),
        "with_hemibrain_type": int(frame["hemibrain_type"].notna().sum()),
        "parquet_sha256": sha256_file(path),
        "use": "join key for hb<->fw transfer; never a training feature",
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return path


# =========================================================================== driver


def default_eval_config(run_label: str = "v2", report_root: str = DEFAULT_REPORT_ROOT, **overrides: Any) -> Any:
    import flybrain_model_eval as fme

    models = (
        fme.ModelSpec("nb", ({"alpha": 1.0},), "temperature"),
        fme.ModelSpec("logreg", ({"C": 0.1}, {"C": 1.0}), "temperature"),
        fme.ModelSpec("hgb", ({"max_iter": 300, "learning_rate": 0.1},
                              {"max_iter": 300, "learning_rate": 0.1, "class_weight": "balanced"}), "temperature"),
    )
    kwargs = dict(models=models, cv_folds=4, n_bootstrap=1000, run_label=run_label, n_threads=8,
                  report_root=report_root)
    kwargs.update(overrides)
    return fme.EvalConfig(**kwargs)


def run_target(target: str, *, storage_root: str = DEFAULT_STORAGE_ROOT, run_label: str = "v2",
               report_root: str = DEFAULT_REPORT_ROOT, cache_root: str = fwf.DEFAULT_CACHE_ROOT,
               write_bridge: bool = False, nt_root: str = ntgt.DEFAULT_NT_GT_ROOT,
               **config_overrides: Any) -> dict[str, Any]:
    t0 = time.time()
    spec = FW_TARGETS[target]
    roles = [ROLE_CONNECTIONS, ROLE_ANNOTATIONS] + ([ROLE_PRE_COUNTS] if spec.needs_pre_counts else [])
    snap = open_fw_snapshot(storage_root, required_roles=roles,
                            stamp_dir=Path(cache_root) / "hash-stamps" / "fw")
    ann = load_fw_annotations(snap.path(ROLE_ANNOTATIONS))
    totals = total_presynapse_counts(snap.path(ROLE_PRE_COUNTS)) if spec.needs_pre_counts else None
    config = default_eval_config(run_label=run_label, report_root=report_root, **config_overrides)
    build = build_fw_eval_dataset(target, snapshot=snap, annotations=ann, config=config,
                                  presynapse_totals=totals, cache_root=cache_root, nt_root=nt_root)
    print(f"[{target}] samples {len(build.rows)} features {build.info['n_features']} "
          f"({round(time.time() - t0)} s)", flush=True)
    if write_bridge:
        bridge = write_bridge_table(ann, snapshot=snap, features_fingerprint=build.features.fingerprint,
                                    features_cache=build.features.cache_path, cache_root=cache_root)
        print(f"[{target}] bridge {bridge}", flush=True)
    report = ftr.run_gated_evaluation(build.data, config)
    for row in report["summary"]:
        print(json.dumps(row), flush=True)
    print(f"[{target}] elapsed {round(time.time() - t0)} s maxrss_MB "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f}", flush=True)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate fw real-model targets through the eval harness.")
    parser.add_argument("targets", nargs="+", choices=sorted(FW_TARGETS))
    parser.add_argument("--storage-root", default=DEFAULT_STORAGE_ROOT)
    parser.add_argument("--report-root", default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--cache-root", default=fwf.DEFAULT_CACHE_ROOT)
    parser.add_argument("--run-label", default="v2")
    parser.add_argument("--write-bridge", action="store_true")
    args = parser.parse_args(argv)
    for target in args.targets:
        run_target(target, storage_root=args.storage_root, run_label=args.run_label, report_root=args.report_root,
                   cache_root=args.cache_root, write_bridge=args.write_bridge)
    return 0


__all__ = [
    "CLASSICAL_NT",
    "FW_PRODUCT_PATHS",
    "FW_TARGETS",
    "FwBuild",
    "FwSnapshot",
    "FwSnapshotError",
    "FwTargetSpec",
    "GROUP_COLUMNS",
    "HEMILINEAGE_GROUP_COLUMNS",
    "HEMILINEAGE_SENTINELS",
    "TYPE_FAMILY_MERGES",
    "held_out_mask_ids",
    "nt_literature_labels",
    "type_family",
    "ROLE_ANNOTATIONS",
    "ROLE_CONNECTIONS",
    "ROLE_PRE_COUNTS",
    "binned_text",
    "build_fw_eval_dataset",
    "component_ids",
    "connectivity_tier_labels",
    "default_eval_config",
    "edge_source",
    "hemibrain_type_group",
    "load_fw_annotations",
    "open_fw_snapshot",
    "parse_known_nt",
    "run_target",
    "select_target_rows",
    "total_presynapse_counts",
    "write_bridge_table",
]


if __name__ == "__main__":
    raise SystemExit(main())
