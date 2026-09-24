"""Fail-closed local adapter for the L1 larval connectome (Winding et al. 2023).

Reads the pinned ``l1em/catmaid_l1em`` snapshot that holds the paper's
supplementary data (Europe PMC author manuscript PMC7614541, CC-BY-4.0):

- ``metadata/files/all-all_connectivity_matrix.csv``: square synapse-count
  matrix, rows = presynaptic skeleton id (skid), columns = postsynaptic skid.
- ``metadata/files/Supplementary_Data_S2.csv``: left/right homologous pairs with
  ``celltype``, ``additional_annotations`` and ``level_7_cluster``.
- ``metadata/files/inputs.csv`` / ``outputs.csv``: per-neuron axon/dendrite
  input and output synapse counts.

Guarantees:

- local snapshot reads only, no network access;
- the manifest (``manifest/manifest.json``, ``fbh-manifest/v1``) must be
  verified, pinned to ``l1em``/``catmaid_l1em``, CC-BY licensed, and every file
  it lists must match its size and sha256 before any table is parsed;
- every failure raises :class:`L1emAdapterError` with a stable
  :class:`L1emErrorCode`;
- region ids come from the larval CATMAID cell-type vocabulary (``l1_`` prefix)
  and are never mapped to adult neuropils. Cross-stage identity transfer is
  refused with ``CROSS_STAGE_UNSUPPORTED``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from flybrain_dataset_registry import snapshot_version_root
from flybrain_harness_storage import build_flybrain_harness_layout

L1EM_DATASET_SYMBOL = "l1em"
L1EM_VERSION_ID = "catmaid_l1em"
L1EM_STAGE = "larva_l1"
L1EM_MANIFEST_SCHEMA_VERSION = "fbh-manifest/v1"
L1EM_SNAPSHOT_RELATIVE_ROOT = "snapshots/l1em/catmaid_l1em"
L1EM_MANIFEST_RELATIVE_PATH = "manifest/manifest.json"
L1EM_MANIFEST_SIDECAR_RELATIVE_PATH = "manifest/manifest.sha256"
L1EM_ALLOWED_LICENSES: frozenset[str] = frozenset({"CC-BY-4.0"})
L1EM_REGION_PREFIX = "l1_"
L1EM_REQUIRED_FILES: Mapping[str, str] = {
    "all_all_matrix": "metadata/files/all-all_connectivity_matrix.csv",
    "annotations": "metadata/files/Supplementary_Data_S2.csv",
    "inputs": "metadata/files/inputs.csv",
    "outputs": "metadata/files/outputs.csv",
}
L1EM_ANNOTATION_COLUMNS: tuple[str, ...] = (
    "left_id",
    "right_id",
    "celltype",
    "additional_annotations",
    "level_7_cluster",
)
L1EM_INPUT_COLUMNS: tuple[str, ...] = ("axon_input", "dendrite_input")
L1EM_OUTPUT_COLUMNS: tuple[str, ...] = ("axon_output", "dendrite_output")
L1EM_NO_PAIR = "no pair"
# Objectives this release can support, and why the others cannot.
L1EM_SUPPORTED_OBJECTIVES: tuple[str, ...] = ("connectivity_tier",)
L1EM_UNSUPPORTED_OBJECTIVES: Mapping[str, str] = {
    "neurotransmitter_dominance": (
        "Winding et al. 2023 supplementary data carry no per-neuron neurotransmitter "
        "annotations or predictions with provenance."
    ),
    "region_specialization_tier": (
        "No per-synapse larval neuropil assignment in this release; the only region key "
        "is the CATMAID cell type, which is also the routing key."
    ),
}
# Adult-only vocabularies that must never be applied to larval data.
_ADULT_VOCABULARIES = frozenset(
    {"flywire_neuropil", "hemibrain_roi", "banc_neuropil", "manc_neuropil", "optic_lobe_neuropil"}
)
_SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")
_REGION_SANITIZE_RE = re.compile(r"[^a-z0-9]+")
_BLOCKED_ARTIFACT_SUFFIXES = (".partial", ".tmp", ".inprogress")


class L1emErrorCode(str, Enum):
    ROOT_NOT_CONFIGURED = "ROOT_NOT_CONFIGURED"
    PATH_ESCAPE = "PATH_ESCAPE"
    MANIFEST_MISSING = "MANIFEST_MISSING"
    MANIFEST_INVALID = "MANIFEST_INVALID"
    MANIFEST_STALE = "MANIFEST_STALE"
    INTEGRITY_MISMATCH = "INTEGRITY_MISMATCH"
    DATASET_PIN_MISMATCH = "DATASET_PIN_MISMATCH"
    LICENSE_UNVERIFIED = "LICENSE_UNVERIFIED"
    REQUIRED_FILE_MISSING = "REQUIRED_FILE_MISSING"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    MATRIX_INVALID = "MATRIX_INVALID"
    ANNOTATION_CONFLICT = "ANNOTATION_CONFLICT"
    OBJECTIVE_UNSUPPORTED = "OBJECTIVE_UNSUPPORTED"
    CROSS_STAGE_UNSUPPORTED = "CROSS_STAGE_UNSUPPORTED"


class L1emAdapterError(ValueError):
    """Typed, fail-closed adapter error. Message is prefixed ``[CODE] ``."""

    def __init__(self, code: L1emErrorCode, message: str, details: Mapping[str, Any] | None = None) -> None:
        self.code = L1emErrorCode(code)
        self.message = message
        self.details = dict(details or {})
        super().__init__(f"[{self.code.value}] {message}")


@dataclass(frozen=True)
class L1emNeuron:
    skid: int
    celltype: str
    annotation: str
    hemisphere: str  # "left" | "right"
    pair_skid: int | None
    cluster: str
    out_synapses: int  # row sum of the all-all matrix (presynaptic)
    in_synapses: int  # column sum of the all-all matrix (postsynaptic)
    axon_output: float | None = None
    dendrite_output: float | None = None
    axon_input: float | None = None
    dendrite_input: float | None = None

    @property
    def paired(self) -> bool:
        return self.pair_skid is not None

    @property
    def total_synapses(self) -> int:
        return int(self.out_synapses) + int(self.in_synapses)

    @property
    def region_id(self) -> str:
        return larval_region_id(self.celltype)

    @property
    def axon_output_fraction(self) -> float | None:
        return _fraction(self.axon_output, self.dendrite_output)

    @property
    def axon_input_fraction(self) -> float | None:
        return _fraction(self.axon_input, self.dendrite_input)


@dataclass(frozen=True)
class L1emSnapshot:
    snapshot_root: Path
    manifest_path: Path
    manifest_id: str
    manifest_sha256: str
    license_spdx: str
    citation: str
    matrix_rows: int
    annotation_rows: int
    neurons: tuple[L1emNeuron, ...]
    unannotated_skids: tuple[int, ...]
    files: Mapping[str, Path] = field(default_factory=dict)

    def neuron(self, skid: int) -> L1emNeuron:
        for item in self.neurons:
            if item.skid == int(skid):
                return item
        raise KeyError(skid)


# ---------------------------------------------------------------------------
# Region vocabulary (larval only).
# ---------------------------------------------------------------------------

def larval_region_id(celltype: str) -> str:
    """Larval CATMAID cell-type region key, e.g. ``pre-DN-VNC`` -> ``l1_pre_dn_vnc``."""
    value = _REGION_SANITIZE_RE.sub("_", str(celltype).strip().lower()).strip("_")
    if not value:
        raise L1emAdapterError(L1emErrorCode.SCHEMA_MISMATCH, "empty celltype cannot form a larval region id")
    return f"{L1EM_REGION_PREFIX}{value}"


def assert_larval_region(region_id: str) -> str:
    text = str(region_id)
    if not text.startswith(L1EM_REGION_PREFIX) or len(text) <= len(L1EM_REGION_PREFIX):
        raise L1emAdapterError(
            L1emErrorCode.CROSS_STAGE_UNSUPPORTED,
            f"region {text!r} is not in the larval l1em vocabulary",
            {"region_id": text, "required_prefix": L1EM_REGION_PREFIX},
        )
    return text


def map_to_adult_neuropil(region_id: str, *, target_vocabulary: str = "flywire_neuropil") -> str:
    """Always refuses: larval cell-type regions have no adult neuropil identity."""
    raise L1emAdapterError(
        L1emErrorCode.CROSS_STAGE_UNSUPPORTED,
        "cross-stage identity transfer from larval l1em regions to adult neuropils is unsupported",
        {"region_id": str(region_id), "target_vocabulary": str(target_vocabulary)},
    )


def validate_scope(scope: Mapping[str, Any]) -> dict[str, Any]:
    """Fail-closed claim-scope check: only larval, dataset-local claims are allowed."""
    stage = str(scope.get("stage") or L1EM_STAGE).strip().lower()
    vocabulary = str(scope.get("region_vocabulary") or "catmaid_annotation").strip().lower()
    target_stage = str(scope.get("target_stage") or stage).strip().lower()
    if not stage.startswith("larva") or not target_stage.startswith("larva"):
        raise L1emAdapterError(
            L1emErrorCode.CROSS_STAGE_UNSUPPORTED,
            "l1em evidence supports larval dataset-local claims only",
            {"stage": stage, "target_stage": target_stage},
        )
    if vocabulary in _ADULT_VOCABULARIES:
        raise L1emAdapterError(
            L1emErrorCode.CROSS_STAGE_UNSUPPORTED,
            "adult region vocabularies cannot be applied to l1em",
            {"region_vocabulary": vocabulary},
        )
    return {
        "allowed": True,
        "dataset_symbol": L1EM_DATASET_SYMBOL,
        "version_id": L1EM_VERSION_ID,
        "stage": stage,
        "region_vocabulary": vocabulary,
        "cross_stage_identity_transfer": "unsupported",
    }


def require_supported_objective(objective: str) -> str:
    if objective in L1EM_SUPPORTED_OBJECTIVES:
        return objective
    reason = L1EM_UNSUPPORTED_OBJECTIVES.get(objective, "unknown objective for l1em")
    raise L1emAdapterError(
        L1emErrorCode.OBJECTIVE_UNSUPPORTED,
        f"objective {objective!r} is unsupported for l1em: {reason}",
        {"objective": objective, "supported": list(L1EM_SUPPORTED_OBJECTIVES)},
    )


# ---------------------------------------------------------------------------
# Snapshot + manifest.
# ---------------------------------------------------------------------------

def resolve_l1em_snapshot_root(
    storage_root: str | Path | None = None,
    *,
    snapshot_root: str | Path | None = None,
) -> Path:
    """Explicit ``snapshot_root`` wins; otherwise ``<root>/snapshots/l1em/catmaid_l1em``."""
    if snapshot_root is not None:
        candidate = Path(snapshot_root)
        if not candidate.is_absolute():
            raise L1emAdapterError(
                L1emErrorCode.ROOT_NOT_CONFIGURED,
                "snapshot_root must be absolute",
                {"snapshot_root": str(snapshot_root)},
            )
        return candidate.resolve(strict=False)
    try:
        layout = build_flybrain_harness_layout(root_override=storage_root, create=False)
    except ValueError as exc:
        raise L1emAdapterError(L1emErrorCode.ROOT_NOT_CONFIGURED, str(exc)) from exc
    candidate = snapshot_version_root(L1EM_DATASET_SYMBOL, layout.root).resolve(strict=False)
    if not candidate.is_relative_to(layout.root.resolve(strict=False)):
        raise L1emAdapterError(
            L1emErrorCode.PATH_ESCAPE,
            "l1em snapshot root escapes the storage root",
            {"snapshot_root": str(candidate)},
        )
    return candidate


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Self-hash rule from FLYBRAIN_HARNESS_MANIFEST_PROVENANCE_SCHEMA.md."""
    canonical = json.loads(json.dumps(manifest))
    integrity = canonical.get("integrity")
    if not isinstance(integrity, dict):
        raise L1emAdapterError(L1emErrorCode.MANIFEST_INVALID, "manifest integrity section must be an object")
    integrity["manifest_sha256"] = ""
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_l1em_manifest(snapshot_root: Path, *, verify_files: bool = True) -> dict[str, Any]:
    """Load and validate the pinned manifest; returns the parsed manifest."""
    manifest_path = snapshot_root / L1EM_MANIFEST_RELATIVE_PATH
    if not manifest_path.is_file():
        raise L1emAdapterError(
            L1emErrorCode.MANIFEST_MISSING,
            "l1em manifest is missing for the pinned snapshot",
            {"manifest_path": str(manifest_path)},
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise L1emAdapterError(
            L1emErrorCode.MANIFEST_INVALID, f"manifest is unreadable: {exc}", {"manifest_path": str(manifest_path)}
        ) from exc
    if not isinstance(manifest, dict):
        raise L1emAdapterError(L1emErrorCode.MANIFEST_INVALID, "manifest must be a JSON object")
    if manifest.get("schema_version") != L1EM_MANIFEST_SCHEMA_VERSION:
        raise L1emAdapterError(
            L1emErrorCode.MANIFEST_INVALID,
            "manifest schema_version mismatch",
            {"schema_version": manifest.get("schema_version")},
        )
    sections = {name: manifest.get(name) for name in ("artifact", "dataset", "scope", "integrity", "refresh", "lineage")}
    missing = sorted(name for name, value in sections.items() if not isinstance(value, dict) or not value)
    if missing:
        raise L1emAdapterError(L1emErrorCode.MANIFEST_INVALID, "manifest sections missing", {"missing": missing})
    dataset, artifact, scope = sections["dataset"], sections["artifact"], sections["scope"]
    integrity, refresh = sections["integrity"], sections["refresh"]
    if dataset.get("symbol") != L1EM_DATASET_SYMBOL or dataset.get("version_id") != L1EM_VERSION_ID:
        raise L1emAdapterError(
            L1emErrorCode.DATASET_PIN_MISMATCH,
            "manifest is not pinned to l1em/catmaid_l1em",
            {"symbol": dataset.get("symbol"), "version_id": dataset.get("version_id")},
        )
    relative_root = str(artifact.get("relative_root") or "").replace("\\", "/").strip("/")
    if not _is_safe_relative_path(relative_root):
        raise L1emAdapterError(L1emErrorCode.PATH_ESCAPE, "artifact.relative_root is unsafe", {"relative_root": relative_root})
    if relative_root != L1EM_SNAPSHOT_RELATIVE_ROOT:
        raise L1emAdapterError(
            L1emErrorCode.DATASET_PIN_MISMATCH,
            "artifact.relative_root does not match the pinned l1em snapshot root",
            {"relative_root": relative_root, "expected": L1EM_SNAPSHOT_RELATIVE_ROOT},
        )
    if not str(scope.get("stage") or "").startswith("larva"):
        raise L1emAdapterError(
            L1emErrorCode.CROSS_STAGE_UNSUPPORTED, "manifest scope.stage must be larval", {"stage": scope.get("stage")}
        )
    source = dataset.get("source") if isinstance(dataset.get("source"), dict) else {}
    license_info = source.get("license") if isinstance(source.get("license"), dict) else {}
    spdx = str(license_info.get("spdx_id") or "").strip()
    if spdx not in L1EM_ALLOWED_LICENSES:
        raise L1emAdapterError(
            L1emErrorCode.LICENSE_UNVERIFIED,
            "l1em snapshot license is not a reviewed research-use license",
            {"spdx_id": spdx, "allowed": sorted(L1EM_ALLOWED_LICENSES)},
        )
    if not str(source.get("citation") or "").strip():
        raise L1emAdapterError(L1emErrorCode.MANIFEST_INVALID, "dataset.source.citation is required")

    verification = integrity.get("verification") if isinstance(integrity.get("verification"), dict) else {}
    status = str(verification.get("status") or "").strip().lower()
    if status not in {"pending", "verified", "quarantined"}:
        raise L1emAdapterError(L1emErrorCode.MANIFEST_INVALID, "integrity.verification.status invalid", {"status": status})
    if status != "verified":
        raise L1emAdapterError(L1emErrorCode.INTEGRITY_MISMATCH, "manifest is not verified", {"status": status})
    if not str(verification.get("verified_at") or "").strip():
        raise L1emAdapterError(L1emErrorCode.MANIFEST_INVALID, "integrity.verification.verified_at is required")
    expected = str(integrity.get("manifest_sha256") or "").strip()
    if not _SHA256_HEX_RE.fullmatch(expected):
        raise L1emAdapterError(L1emErrorCode.MANIFEST_INVALID, "integrity.manifest_sha256 must be lowercase 64-hex")
    actual = manifest_digest(manifest)
    if actual != expected:
        raise L1emAdapterError(
            L1emErrorCode.INTEGRITY_MISMATCH, "manifest sha256 mismatch", {"expected": expected, "actual": actual}
        )

    decision = str(refresh.get("decision") or "").strip().lower()
    if decision not in {"no_change", "patch_refresh", "major_bump", "rollback"}:
        raise L1emAdapterError(L1emErrorCode.MANIFEST_INVALID, "refresh.decision invalid", {"decision": decision})
    if decision == "rollback":
        raise L1emAdapterError(L1emErrorCode.MANIFEST_STALE, "refresh.decision indicates rollback")
    due = refresh.get("next_check_due")
    if due:
        try:
            due_dt = datetime.fromisoformat(str(due).replace("Z", "+00:00"))
        except ValueError as exc:
            raise L1emAdapterError(L1emErrorCode.MANIFEST_INVALID, "refresh.next_check_due unparseable") from exc
        if due_dt < datetime.now(timezone.utc):
            raise L1emAdapterError(
                L1emErrorCode.MANIFEST_STALE, "refresh.next_check_due has elapsed", {"next_check_due": due}
            )

    files = integrity.get("files")
    if not isinstance(files, list) or not files:
        raise L1emAdapterError(L1emErrorCode.MANIFEST_INVALID, "integrity.files must be a non-empty list")
    root = snapshot_root.resolve(strict=False)
    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise L1emAdapterError(L1emErrorCode.MANIFEST_INVALID, "integrity.files entries must be objects")
        rel = str(entry.get("relative_path") or "").strip()
        digest = str(entry.get("sha256") or "").strip()
        size = entry.get("size_bytes")
        if not _is_safe_relative_path(rel):
            raise L1emAdapterError(L1emErrorCode.PATH_ESCAPE, "unsafe file path in manifest", {"relative_path": rel})
        folded = rel.replace("\\", "/").casefold()
        if folded in seen:
            raise L1emAdapterError(L1emErrorCode.MANIFEST_INVALID, "duplicate file path", {"relative_path": rel})
        seen.add(folded)
        if folded.endswith(_BLOCKED_ARTIFACT_SUFFIXES):
            raise L1emAdapterError(L1emErrorCode.MANIFEST_INVALID, "partial artifact listed", {"relative_path": rel})
        if not _SHA256_HEX_RE.fullmatch(digest):
            raise L1emAdapterError(L1emErrorCode.MANIFEST_INVALID, "file sha256 must be 64-hex", {"relative_path": rel})
        path = (root / rel).resolve(strict=False)
        if not path.is_relative_to(root):
            raise L1emAdapterError(L1emErrorCode.PATH_ESCAPE, "file escapes snapshot root", {"relative_path": rel})
        if not path.is_file():
            raise L1emAdapterError(L1emErrorCode.REQUIRED_FILE_MISSING, "manifest file is missing", {"relative_path": rel})
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise L1emAdapterError(
                L1emErrorCode.MANIFEST_INVALID, "size_bytes must be a non-negative integer", {"relative_path": rel}
            )
        if path.stat().st_size != size:
            raise L1emAdapterError(L1emErrorCode.INTEGRITY_MISMATCH, "file size mismatch", {"relative_path": rel})
        if verify_files:
            got = _sha256_file(path)
            if got != digest:
                raise L1emAdapterError(
                    L1emErrorCode.INTEGRITY_MISMATCH,
                    "file sha256 mismatch",
                    {"relative_path": rel, "expected": digest, "actual": got},
                )
    sidecar = snapshot_root / L1EM_MANIFEST_SIDECAR_RELATIVE_PATH
    if not sidecar.is_file():
        raise L1emAdapterError(
            L1emErrorCode.MANIFEST_MISSING, "manifest.sha256 sidecar is missing", {"sidecar": str(sidecar)}
        )
    if sidecar.read_text(encoding="utf-8").strip() != expected:
        raise L1emAdapterError(L1emErrorCode.INTEGRITY_MISMATCH, "manifest.sha256 sidecar does not match manifest")
    for role, rel in sorted(L1EM_REQUIRED_FILES.items()):
        if rel.casefold() not in seen:
            raise L1emAdapterError(
                L1emErrorCode.REQUIRED_FILE_MISSING,
                f"manifest does not list required {role} file",
                {"role": role, "relative_path": rel},
            )
    return manifest


def load_l1em_snapshot(
    storage_root: str | Path | None = None,
    *,
    snapshot_root: str | Path | None = None,
    verify_integrity: bool = True,
) -> L1emSnapshot:
    """Validate the manifest, then parse the neuron table. Read-only, local-only."""
    root = resolve_l1em_snapshot_root(storage_root, snapshot_root=snapshot_root)
    manifest = load_l1em_manifest(root, verify_files=verify_integrity)
    files = {role: root / rel for role, rel in L1EM_REQUIRED_FILES.items()}

    skids, out_syn, in_syn = _read_matrix_totals(files["all_all_matrix"])
    annotations, annotation_rows = _read_annotations(files["annotations"])
    inputs = _read_compartment_counts(files["inputs"], L1EM_INPUT_COLUMNS)
    outputs = _read_compartment_counts(files["outputs"], L1EM_OUTPUT_COLUMNS)

    neurons: list[L1emNeuron] = []
    unannotated: list[int] = []
    for skid in sorted(skids):
        info = annotations.get(skid)
        if info is None:
            unannotated.append(skid)
            continue
        ax_in, de_in = inputs.get(skid, (None, None))
        ax_out, de_out = outputs.get(skid, (None, None))
        neurons.append(
            L1emNeuron(
                skid=skid,
                celltype=info["celltype"],
                annotation=info["annotation"],
                hemisphere=info["hemisphere"],
                pair_skid=info["pair_skid"],
                cluster=info["cluster"],
                out_synapses=out_syn[skid],
                in_synapses=in_syn[skid],
                axon_output=ax_out,
                dendrite_output=de_out,
                axon_input=ax_in,
                dendrite_input=de_in,
            )
        )
    source = manifest["dataset"].get("source", {})
    return L1emSnapshot(
        snapshot_root=root,
        manifest_path=root / L1EM_MANIFEST_RELATIVE_PATH,
        manifest_id=str(manifest.get("manifest_id") or ""),
        manifest_sha256=str(manifest["integrity"]["manifest_sha256"]),
        license_spdx=str(source["license"]["spdx_id"]),
        citation=str(source["citation"]),
        matrix_rows=len(skids),
        annotation_rows=annotation_rows,
        neurons=tuple(neurons),
        unannotated_skids=tuple(unannotated),
        files=files,
    )


# ---------------------------------------------------------------------------
# Table parsers.
# ---------------------------------------------------------------------------

def _read_matrix_totals(path: Path) -> tuple[list[int], dict[int, int], dict[int, int]]:
    import numpy as np
    import pandas as pd

    try:
        frame = pd.read_csv(path, index_col=0)
    except Exception as exc:  # pragma: no cover - parser specific
        raise L1emAdapterError(L1emErrorCode.MATRIX_INVALID, f"matrix unreadable: {exc}", {"path": str(path)}) from exc
    try:
        rows = [int(v) for v in frame.index]
        cols = [int(str(v)) for v in frame.columns]
    except (TypeError, ValueError) as exc:
        raise L1emAdapterError(L1emErrorCode.MATRIX_INVALID, "matrix labels must be integer skids") from exc
    if not rows or rows != cols:
        raise L1emAdapterError(
            L1emErrorCode.MATRIX_INVALID,
            "matrix must be square with identical row/column skid order",
            {"rows": len(rows), "cols": len(cols)},
        )
    if len(set(rows)) != len(rows):
        raise L1emAdapterError(L1emErrorCode.MATRIX_INVALID, "matrix has duplicate skids")
    try:
        values = frame.to_numpy(dtype="float64")
    except (TypeError, ValueError) as exc:
        raise L1emAdapterError(L1emErrorCode.MATRIX_INVALID, "matrix values must be numeric") from exc
    if not np.isfinite(values).all() or (values < 0).any() or (np.mod(values, 1.0) != 0).any():
        raise L1emAdapterError(L1emErrorCode.MATRIX_INVALID, "matrix values must be finite non-negative integers")
    out_totals = values.sum(axis=1)
    in_totals = values.sum(axis=0)
    return (
        rows,
        {skid: int(out_totals[i]) for i, skid in enumerate(rows)},
        {skid: int(in_totals[i]) for i, skid in enumerate(rows)},
    )


def _read_annotations(path: Path) -> tuple[dict[int, dict[str, Any]], int]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        missing = [col for col in L1EM_ANNOTATION_COLUMNS if col not in header]
        if missing:
            raise L1emAdapterError(
                L1emErrorCode.SCHEMA_MISMATCH, "annotation table missing columns", {"missing": missing, "path": str(path)}
            )
        out: dict[int, dict[str, Any]] = {}
        rows = 0
        for line_no, row in enumerate(reader, start=2):
            rows += 1
            left = _parse_skid(row["left_id"], line_no)
            right = _parse_skid(row["right_id"], line_no)
            if left is None and right is None:
                raise L1emAdapterError(
                    L1emErrorCode.SCHEMA_MISMATCH, "annotation row has neither left nor right skid", {"line": line_no}
                )
            celltype = str(row["celltype"] or "").strip()
            if not celltype:
                raise L1emAdapterError(L1emErrorCode.SCHEMA_MISMATCH, "annotation row has empty celltype", {"line": line_no})
            base = {
                "celltype": celltype,
                "annotation": str(row["additional_annotations"] or "").strip() or "no official annotation",
                "cluster": str(row["level_7_cluster"] or "").strip() or "unclustered",
            }
            for skid, hemisphere, partner in ((left, "left", right), (right, "right", left)):
                if skid is None:
                    continue
                if skid in out:
                    raise L1emAdapterError(
                        L1emErrorCode.ANNOTATION_CONFLICT, "skid annotated more than once", {"skid": skid, "line": line_no}
                    )
                out[skid] = {**base, "hemisphere": hemisphere, "pair_skid": partner}
    if not out:
        raise L1emAdapterError(L1emErrorCode.SCHEMA_MISMATCH, "annotation table is empty", {"path": str(path)})
    return out, rows


def _read_compartment_counts(path: Path, columns: tuple[str, ...]) -> dict[int, tuple[float, float]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or tuple(h.strip() for h in header[1:3]) != columns:
            raise L1emAdapterError(
                L1emErrorCode.SCHEMA_MISMATCH,
                "compartment table header mismatch",
                {"path": str(path), "expected": list(columns), "got": header},
            )
        out: dict[int, tuple[float, float]] = {}
        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue
            try:
                skid = int(row[0])
                first, second = float(row[1]), float(row[2])
            except (IndexError, ValueError) as exc:
                raise L1emAdapterError(
                    L1emErrorCode.SCHEMA_MISMATCH, "compartment row unparseable", {"path": str(path), "line": line_no}
                ) from exc
            if first < 0 or second < 0:
                raise L1emAdapterError(
                    L1emErrorCode.SCHEMA_MISMATCH, "compartment counts must be >= 0", {"line": line_no}
                )
            if skid in out:
                raise L1emAdapterError(L1emErrorCode.ANNOTATION_CONFLICT, "duplicate skid", {"skid": skid})
            out[skid] = (first, second)
    return out


def _parse_skid(raw: Any, line_no: int) -> int | None:
    text = str(raw or "").strip()
    if not text or text.lower() == L1EM_NO_PAIR:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise L1emAdapterError(
            L1emErrorCode.SCHEMA_MISMATCH, "skid must be an integer or 'no pair'", {"line": line_no, "value": text}
        ) from exc


def _fraction(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    denominator = float(first) + float(second)
    if denominator <= 0:
        return None
    return float(first) / denominator


def _is_safe_relative_path(raw: str) -> bool:
    text = str(raw or "").strip()
    if not text:
        return False
    normalized = text.replace("\\", "/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        return False
    return all(part != ".." for part in normalized.split("/"))


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


__all__ = [
    "L1EM_ALLOWED_LICENSES",
    "L1EM_DATASET_SYMBOL",
    "L1EM_REQUIRED_FILES",
    "L1EM_SUPPORTED_OBJECTIVES",
    "L1EM_UNSUPPORTED_OBJECTIVES",
    "L1EM_VERSION_ID",
    "L1emAdapterError",
    "L1emErrorCode",
    "L1emNeuron",
    "L1emSnapshot",
    "assert_larval_region",
    "larval_region_id",
    "load_l1em_manifest",
    "load_l1em_snapshot",
    "manifest_digest",
    "map_to_adult_neuropil",
    "require_supported_objective",
    "resolve_l1em_snapshot_root",
    "validate_scope",
]
