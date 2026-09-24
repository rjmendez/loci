"""Fail-closed local adapter for the pinned MaleCNS v1.0 snapshot (registry symbol ``mc``).

MaleCNS (Berg, Beckett, Costa, Schlegel et al. 2026) is the Janelia FlyEM
connectome of the complete central nervous system (brain + ventral nerve cord)
of one adult male *Drosophila*. This module is the only place the harness
touches the on-disk ``mc`` snapshot:

* ``open_mc_snapshot`` validates the storage root, the manifest
  (``fbh-manifest/v1`` self-hash + sidecar, verification status, refresh
  window, licence, per-file ``role``), the size of every listed file and the
  sha256 of every required product before returning resolved paths. Content
  hashes are cached as write-once stamps keyed by (size, mtime_ns)
  (``flybrain_hash_stamps``), so the 37 GB snapshot is not rehashed on every
  open.
* ``load_mc_*`` readers check required columns and return pandas frames with
  body ids as decimal strings. The large tables (``Neuprint_Neurons.feather``
  holds ~88 M bodies; the edge list ~152 M edges) are streamed batch by batch
  with column projection and a body-id filter, so they are never materialized
  in memory.
* ``map_mc_primary_roi`` is the explicit MaleCNS primary-ROI vocabulary
  (144 ROIs: central brain, optic lobes, VNC neuropils, VNC nerves, neck
  connective). ``check_mc_roi_vocabulary`` fails closed unless that vocabulary
  equals the ``primaryRois`` list in the snapshot's ``Neuprint_Meta.csv``.

No network access: everything is read from
``$LOCI_FLYBRAIN_STORAGE_ROOT/snapshots/mc/male-cns_v1.0``. See
``docs/FLYBRAIN_MC_ADAPTER_CONTRACT.md``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from flybrain_dataset_registry import get_dataset, normalize_symbol, snapshot_version_root
from flybrain_hash_stamps import METHOD_SIZE_ONLY, HashStampCache, verify_file_sha256
from flybrain_harness_storage import build_flybrain_harness_layout

MC_DATASET_SYMBOL = "mc"
MC_VERSION_ID = "male-cns_v1.0"
MC_SNAPSHOT_DIR_NAME = "mc"
MC_SNAPSHOT_RELATIVE_ROOT = "snapshots/mc/male-cns_v1.0"
MC_NEUPRINT_DATASET = "male-cns:v1.0"
MC_MANIFEST_SCHEMA_VERSION = "fbh-manifest/v1"
MC_MANIFEST_RELATIVE_PATH = "manifest/manifest.json"
MC_MANIFEST_SIDECAR_RELATIVE_PATH = "manifest/manifest.sha256"
MC_ALLOWED_LICENSES = frozenset({"CC-BY-4.0"})
MC_CITATION = (
    "Berg S, Beckett IR, Costa M, Schlegel P, et al. (2026). Sexual dimorphism in the complete connectome "
    "of the Drosophila male central nervous system. Cell (preprint: bioRxiv doi:10.1101/2025.10.09.680999). "
    "Data: FlyEM MaleCNS v1.0, gs://flyem-male-cns/v1.0 (CC BY 4.0)."
)

# Product roles -> snapshot-relative paths. Role names equal the ``role`` field
# the pull wrote into each manifest entry; a listed role that disagrees fails closed.
ROLE_META = "meta"
ROLE_NT_PREDICTION = "nt_prediction"
ROLE_EDGELIST = "edgelist"
ROLE_NEURON_ROI_INFO = "neuron_roi_info"
ROLE_DATASET_META = "dataset_meta"
MC_PRODUCT_PATHS: Mapping[str, str] = {
    ROLE_META: "metadata/body-annotations-male-cns-v1.0-minconf-0.5.feather",
    ROLE_NT_PREDICTION: "metadata/body-neurotransmitters-male-cns-v1.0.feather",
    ROLE_EDGELIST: "source/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
    ROLE_NEURON_ROI_INFO: "source/neuprint-inputs/Neuprint_Neurons.feather",
    ROLE_DATASET_META: "metadata/neuprint-inputs/Neuprint_Meta.csv",
}

META_REQUIRED_COLUMNS: tuple[str, ...] = (
    "bodyId",
    "type",
    "status",
    "superclass",
    "class",
    "somaSide",
    "rootSide",
    "itoleeHl",
    "trumanHl",
)
NT_REQUIRED_COLUMNS: tuple[str, ...] = (
    "body",
    "cell_type",
    "total_nt_predictions",
    "predicted_nt_confidence",
    "predicted_nt",
    "ground_truth",
    "celltype_predicted_nt",
    "consensus_nt",
)
EDGELIST_REQUIRED_COLUMNS: tuple[str, ...] = ("body_pre", "body_post", "weight")
# neuPrint bulk-import column names carry their Neo4j type suffixes.
NEURON_ID_COLUMN = ":ID(Body-ID)"
NEURON_BODYID_COLUMN = "bodyId:long"
NEURON_ROI_INFO_COLUMN = "roiInfo:string"
NEURON_COUNT_COLUMNS: Mapping[str, str] = {
    "pre:int": "pre",
    "post:int": "post",
    "downstream:int": "downstream",
    "upstream:int": "upstream",
}
NEURON_REQUIRED_COLUMNS: tuple[str, ...] = (NEURON_ID_COLUMN, *NEURON_COUNT_COLUMNS, NEURON_ROI_INFO_COLUMN)
DATASET_META_PRIMARY_ROIS_COLUMN = "primaryRois:string[]"
DATASET_META_REQUIRED_COLUMNS: tuple[str, ...] = ("dataset:string", "tag:string", DATASET_META_PRIMARY_ROIS_COLUMN)

# Full-name NT -> short code (same codes as fw / banc). MaleCNS predicts 7
# classes (no tyramine). ``unclear`` is the classifier's abstention and is not a label.
MC_NT_SHORT_CODES: Mapping[str, str] = {
    "acetylcholine": "ach",
    "gaba": "gaba",
    "glutamate": "glut",
    "dopamine": "da",
    "serotonin": "ser",
    "octopamine": "oct",
    "histamine": "his",
}
MC_NT_ABSTAIN = "unclear"

# ---------------------------------------------------------------------------
# Primary-ROI vocabulary (neuPrint male-cns:v1.0 ``primaryRois``; hierarchy
# from ``roiHierarchy`` in Neuprint_Meta.csv). ``(L)``/``(R)`` suffixes are
# the hemisphere; each base neuropil lists the side suffixes it exists with.
# ---------------------------------------------------------------------------

DIVISION_BRAIN = "brain"
DIVISION_NERVE_CORD = "nerve_cord"
DIVISION_NECK = "neck_connective"
SUBDIVISION_CENTRAL_BRAIN = "central_brain"
SUBDIVISION_OPTIC_LOBE = "optic_lobe"
SUBDIVISION_VNC_NEUROPIL = "ventral_nerve_cord"
SUBDIVISION_VNC_NERVE = "ventral_nerve_cord_nerve"
SUBDIVISION_NECK = "cervical_connective"

_BOTH = ("L", "R")
_NONE: tuple[str, ...] = ()
# (division, subdivision) -> {base neuropil: sides}
_MC_ROI_TABLE: Mapping[tuple[str, str], Mapping[str, tuple[str, ...]]] = {
    (DIVISION_BRAIN, SUBDIVISION_CENTRAL_BRAIN): {
        "AL": _BOTH, "GNG": _NONE, "SCL": _BOTH, "LH": _BOTH, "PED": _BOTH, "CentralBrain-unspecified": _NONE,
        # CX
        "AB": _BOTH, "EB": _NONE, "FB": _NONE, "NO": _NONE, "PB": _NONE,
        # INP
        "IB": _NONE, "ICL": _BOTH, "ATL": _BOTH, "CRE": _BOTH,
        # LX
        "BU": _BOTH, "LAL": _BOTH,
        # MB
        "CA": _BOTH, "a'L": _BOTH, "aL": _BOTH, "b'L": _BOTH, "bL": _BOTH, "gL": _BOTH,
        # PENP
        "CAN": _BOTH, "FLA": _BOTH, "PRW": _NONE, "SAD": _NONE,
        # SNP
        "SIP": _BOTH, "SLP": _BOTH, "SMP": _BOTH,
        # VLNP
        "AOTU": _BOTH, "AVLP": _BOTH, "PLP": _BOTH, "PVLP": _BOTH, "WED": _BOTH,
        # VMNP
        "EPA": _BOTH, "GOR": _BOTH, "IPS": _BOTH, "SPS": _BOTH, "VES": _BOTH,
    },
    (DIVISION_BRAIN, SUBDIVISION_OPTIC_LOBE): {
        "ME": _BOTH, "LO": _BOTH, "LOP": _BOTH, "AME": _BOTH, "LA": _BOTH, "Optic-unspecified": _BOTH,
    },
    (DIVISION_NERVE_CORD, SUBDIVISION_VNC_NEUROPIL): {
        "ANm": _NONE, "HTct(UTct-T3)": _BOTH, "IntTct": _NONE, "LTct": _NONE,
        "LegNp(T1)": _BOTH, "LegNp(T2)": _BOTH, "LegNp(T3)": _BOTH,
        "NTct(UTct-T1)": _BOTH, "Ov": _BOTH, "WTct(UTct-T2)": _BOTH,
        "mVAC(T1)": _BOTH, "mVAC(T2)": _BOTH, "mVAC(T3)": _BOTH, "VNC-unspecified": _NONE,
    },
    (DIVISION_NERVE_CORD, SUBDIVISION_VNC_NERVE): {
        name: _BOTH
        for name in (
            "ADMN", "AbN1", "AbN2", "AbN3", "AbN4", "AbNT", "CvN", "DMetaN", "DProN", "MesoAN", "MesoLN",
            "MetaLN", "PDMN", "PrN", "ProAN", "ProCN", "ProLN", "VProN",
        )
    },
    (DIVISION_NECK, SUBDIVISION_NECK): {"CV-unspecified": _NONE},
}
# Case-only / punctuation-only collisions (aL vs AL, a'L vs aL) get explicit slugs.
_NEUROPIL_SLUG_OVERRIDES: Mapping[str, str] = {
    "aL": "mb_al",
    "a'L": "mb_apl",
    "bL": "mb_bl",
    "b'L": "mb_bpl",
    "gL": "mb_gl",
}
_SIDE_NAMES = {"L": "left", "R": "right"}
_REGION_SANITIZE_RE = re.compile(r"[^a-z0-9]+")


def _sanitize(raw: str) -> str:
    value = _REGION_SANITIZE_RE.sub("_", str(raw).strip().lower()).strip("_")
    return value or "unknown_region"


@dataclass(frozen=True)
class McRoi:
    roi: str
    division: str
    subdivision: str
    neuropil: str
    side: str | None
    region_id: str


def _build_roi_vocabulary() -> dict[str, McRoi]:
    out: dict[str, McRoi] = {}
    slugs: dict[str, str] = {}
    for (division, subdivision), neuropils in _MC_ROI_TABLE.items():
        for neuropil, sides in neuropils.items():
            slug = _NEUROPIL_SLUG_OVERRIDES.get(neuropil) or _sanitize(neuropil)
            region_id = f"{_sanitize(division)}_{slug}"
            if region_id in slugs and slugs[region_id] != neuropil:
                raise AssertionError(f"mc region_id collision: {neuropil!r} and {slugs[region_id]!r} -> {region_id}")
            slugs[region_id] = neuropil
            for suffix in sides or (None,):
                roi = neuropil if suffix is None else f"{neuropil}({suffix})"
                if roi in out:
                    raise AssertionError(f"duplicate mc primary ROI {roi!r}")
                out[roi] = McRoi(roi, division, subdivision, neuropil, _SIDE_NAMES.get(suffix or ""), region_id)
    return out


MC_PRIMARY_ROIS: Mapping[str, McRoi] = _build_roi_vocabulary()


class McAdapterErrorCode(str, Enum):
    ROOT_NOT_CONFIGURED = "ROOT_NOT_CONFIGURED"
    PATH_ESCAPE = "PATH_ESCAPE"
    DATASET_PIN_MISMATCH = "DATASET_PIN_MISMATCH"
    SNAPSHOT_MISSING = "SNAPSHOT_MISSING"
    MANIFEST_MISSING = "MANIFEST_MISSING"
    MANIFEST_INVALID = "MANIFEST_INVALID"
    MANIFEST_EXISTS = "MANIFEST_EXISTS"
    LICENSE_NOT_PERMITTED = "LICENSE_NOT_PERMITTED"
    INTEGRITY_MISMATCH = "INTEGRITY_MISMATCH"
    PROMOTION_STATE_INVALID = "PROMOTION_STATE_INVALID"
    PRODUCT_MISSING = "PRODUCT_MISSING"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    REGION_VOCABULARY_UNKNOWN = "REGION_VOCABULARY_UNKNOWN"
    NT_VOCABULARY_UNKNOWN = "NT_VOCABULARY_UNKNOWN"


class McAdapterError(ValueError):
    """Typed fail-closed error; message is prefixed with ``[CODE]``."""

    def __init__(self, code: McAdapterErrorCode, message: str, details: Mapping[str, Any] | None = None):
        self.code = code
        self.message = message
        self.details = dict(details or {})
        super().__init__(f"[{code.value}] {message}")


def _fail(code: McAdapterErrorCode, message: str, **details: Any) -> None:
    raise McAdapterError(code, message, details)


def map_mc_primary_roi(roi: str) -> McRoi:
    """Map a neuPrint primary ROI name (``LegNp(T1)(L)``, ``EB``) onto the explicit vocabulary.

    ``region_id`` is ``<division>_<neuropil slug>`` without the side
    (``ME(R)`` -> ``brain_me``, ``LegNp(T1)(L)`` -> ``nerve_cord_legnp_t1``).
    Anything outside the 144-ROI vocabulary raises ``REGION_VOCABULARY_UNKNOWN``.
    """
    text = str(roi or "").strip()
    if text not in MC_PRIMARY_ROIS:
        raise McAdapterError(
            McAdapterErrorCode.REGION_VOCABULARY_UNKNOWN,
            f"ROI {text!r} is not a MaleCNS v1.0 primary ROI",
            {"roi": text},
        )
    return MC_PRIMARY_ROIS[text]


def nt_short_code(name: str) -> str:
    """Full NT name -> short code. ``unclear`` and unknown names raise ``NT_VOCABULARY_UNKNOWN``."""
    text = str(name or "").strip().lower()
    if text not in MC_NT_SHORT_CODES:
        raise McAdapterError(
            McAdapterErrorCode.NT_VOCABULARY_UNKNOWN,
            f"neurotransmitter {name!r} is not in the MaleCNS NT vocabulary",
            {"neurotransmitter": name, "known": sorted(MC_NT_SHORT_CODES)},
        )
    return MC_NT_SHORT_CODES[text]


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

_SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")
_BLOCKED_ARTIFACT_SUFFIXES = (".partial", ".tmp", ".inprogress")
_REFRESH_DECISIONS = frozenset({"no_change", "patch_refresh", "major_bump", "rollback"})


def is_safe_relative_path(raw: str) -> bool:
    text = str(raw or "").strip()
    if not text:
        return False
    normalized = text.replace("\\", "/")
    if normalized.startswith("/"):
        return False
    if re.match(r"^[A-Za-z]:", normalized):
        return False
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    return bool(parts) and not any(part == ".." for part in parts)


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Self-hash rule from FLYBRAIN_HARNESS_MANIFEST_PROVENANCE_SCHEMA.md."""
    canonical = json.loads(json.dumps(manifest))
    integrity = canonical.get("integrity")
    if not isinstance(integrity, dict):
        raise McAdapterError(McAdapterErrorCode.MANIFEST_INVALID, "manifest.integrity must be an object")
    integrity["manifest_sha256"] = ""
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_mc_snapshot_root(storage_root: str | Path | None = None) -> tuple[Path, Path]:
    """Return ``(storage_root, snapshot_root)`` after root + path-escape validation."""
    try:
        layout = build_flybrain_harness_layout(root_override=storage_root, create=False)
    except ValueError as exc:
        raise McAdapterError(McAdapterErrorCode.ROOT_NOT_CONFIGURED, str(exc)) from exc
    spec = get_dataset(MC_DATASET_SYMBOL)
    if spec.pinned_version != MC_VERSION_ID or spec.snapshot_dir_name != MC_SNAPSHOT_DIR_NAME:
        raise McAdapterError(
            McAdapterErrorCode.DATASET_PIN_MISMATCH,
            "registry pin for mc does not match the adapter pin",
            {"registry_version": spec.pinned_version, "adapter_version": MC_VERSION_ID},
        )
    snapshot_root = snapshot_version_root(MC_DATASET_SYMBOL, layout.root).resolve(strict=False)
    if not snapshot_root.is_relative_to(layout.root.resolve(strict=False)):
        raise McAdapterError(
            McAdapterErrorCode.PATH_ESCAPE,
            "mc snapshot path escapes the configured storage root",
            {"snapshot_root": str(snapshot_root)},
        )
    return layout.root, snapshot_root


def build_mc_manifest(
    *,
    files: Sequence[Mapping[str, Any]],
    source_uri: str,
    retrieved_at: str,
    license_spdx: str = "CC-BY-4.0",
    run_id: str,
    generated_at: str | None = None,
    verified_at: str | None = None,
    next_check_due: str | None = None,
    notes: Iterable[str] = (),
    extra_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a verified ``fbh-manifest/v1`` dict for the mc snapshot (self-hash applied).

    Shaped like the manifest the 2026-09-24 selective pull wrote; used by tests
    and by any re-pull (which must be written as a new, superseding manifest).
    """
    generated = generated_at or _utc_now()
    entries = []
    for item in sorted(files, key=lambda f: str(f["relative_path"])):
        entry = {
            "relative_path": str(item["relative_path"]).replace("\\", "/"),
            "size_bytes": int(item["size_bytes"]),
            "sha256": str(item["sha256"]),
        }
        for key in ("url", "retrieved_at", "role", "md5_b64", "remote_crc32c_b64"):
            if key in item:
                entry[key] = item[key]
        entries.append(entry)
    source: dict[str, Any] = {
        "system": "flyem-male-cns-gcs-public",
        "access_method": "https_public_bucket_selective_pull",
        "uri": source_uri,
        "retrieved_at": retrieved_at,
        "license": {"spdx_id": license_spdx, "url": "https://creativecommons.org/licenses/by/4.0/"},
        "citation": MC_CITATION,
        "doi": "10.1101/2025.10.09.680999",
        "neuprint_dataset": MC_NEUPRINT_DATASET,
    }
    source.update(dict(extra_source or {}))
    manifest: dict[str, Any] = {
        "schema_version": MC_MANIFEST_SCHEMA_VERSION,
        "manifest_id": f"fbh-mc-{MC_VERSION_ID}-{generated}",
        "generated_at": generated,
        "storage_root_env": "LOCI_FLYBRAIN_STORAGE_ROOT",
        "artifact": {
            "kind": "dataset_snapshot",
            "relative_root": MC_SNAPSHOT_RELATIVE_ROOT,
            "path_template": "$LOCI_FLYBRAIN_STORAGE_ROOT\\snapshots\\mc\\male-cns_v1.0\\",
        },
        "dataset": {
            "symbol": "mc",
            "label": "Janelia FlyEM male adult CNS (MaleCNS) connectome v1.0 (selective tabular products)",
            "version_id": MC_VERSION_ID,
            "source": source,
        },
        "scope": {
            "sex": "male",
            "stage": "adult",
            "anatomy": "brain_and_ventral_nerve_cord",
            "evidence_family": "connectome_structural",
            "claim_tier": "T1_dataset_version_specific",
        },
        "integrity": {
            "manifest_sha256": "",
            "files": entries,
            "verification": {"status": "verified", "verified_at": verified_at or generated, "failure_reason": None},
        },
        "refresh": {
            "cadence": "quarterly_integrity_recheck",
            "decision": "no_change",
            "checked_at": generated,
            "supersedes_manifest_id": None,
            "next_check_due": next_check_due,
        },
        "lineage": {
            "derived_from": [{"type": "remote_dataset_release", "id": f"neuprint:{MC_NEUPRINT_DATASET}"}],
            "pipeline": {
                "job_name": "fbh_mc_selective_pull",
                "job_version": "2026.09.24.1",
                "run_id": run_id,
                "run_mode": "execute",
            },
        },
        "notes": list(notes),
    }
    manifest["integrity"]["manifest_sha256"] = manifest_digest(manifest)
    return manifest


def write_mc_manifest(snapshot_root: Path, manifest: Mapping[str, Any]) -> Path:
    """Write manifest.json + manifest.sha256; refuses to overwrite (MANIFEST_EXISTS)."""
    target = Path(snapshot_root) / MC_MANIFEST_RELATIVE_PATH
    sidecar = Path(snapshot_root) / MC_MANIFEST_SIDECAR_RELATIVE_PATH
    for path in (target, sidecar):
        if path.exists():
            raise McAdapterError(
                McAdapterErrorCode.MANIFEST_EXISTS,
                "refusing to overwrite an existing mc manifest; supersede with a new version instead",
                {"path": str(path)},
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sidecar.write_text(str(manifest["integrity"]["manifest_sha256"]) + "\n", encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Snapshot open / verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class McSnapshot:
    storage_root: Path
    snapshot_root: Path
    manifest_path: Path
    manifest_id: str
    manifest_sha256: str
    product_paths: Mapping[str, Path]
    product_sha256: Mapping[str, str]
    license_spdx: str
    warnings: tuple[str, ...] = field(default_factory=tuple)
    # relative_path -> "hashed" | "stamp" | "size_only" (see flybrain_hash_stamps).
    hash_verification: Mapping[str, str] = field(default_factory=dict)

    def path(self, role: str) -> Path:
        if role not in self.product_paths:
            raise McAdapterError(McAdapterErrorCode.PRODUCT_MISSING, f"product role {role!r} not in snapshot")
        return self.product_paths[role]

    def provenance(self) -> dict[str, Any]:
        return {
            "dataset_symbol": MC_DATASET_SYMBOL,
            "version_id": MC_VERSION_ID,
            "manifest_id": self.manifest_id,
            "manifest_sha256": self.manifest_sha256,
            "source_path": MC_SNAPSHOT_RELATIVE_ROOT,
            "access_method": "local_snapshot_read_only",
            "license": self.license_spdx,
            "product_sha256": dict(sorted(self.product_sha256.items())),
        }


def validate_mc_manifest(
    manifest: Mapping[str, Any],
    snapshot_root: Path,
    *,
    required_roles: Sequence[str] = tuple(MC_PRODUCT_PATHS),
    verify_hashes: bool | None = None,
    now: datetime | None = None,
    stamp_cache: HashStampCache | None = None,
    hash_report: dict[str, str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, str]:
    """Fail-closed manifest check. Returns ``{relative_path: sha256}`` for listed files.

    Every listed file is always checked for presence, path safety and exact
    ``size_bytes``. Content hashing depends on ``verify_hashes``:

    * ``True`` (explicit verify): sha256 of every listed file, always (~37 GB).
    * ``None`` (default): sha256 of the files backing ``required_roles`` only,
      skipped when ``stamp_cache`` holds a write-once stamp for the file's
      current ``(size, mtime_ns)``. Other listed files are size-checked only.
    * ``False``: size checks only (smoke runs).

    ``hash_report`` (if given) receives ``relative_path -> method``.
    """
    if manifest.get("schema_version") != MC_MANIFEST_SCHEMA_VERSION:
        _fail(McAdapterErrorCode.MANIFEST_INVALID, "manifest schema_version mismatch",
              schema_version=manifest.get("schema_version"))
    for key in ("manifest_id", "generated_at", "artifact", "dataset", "scope", "integrity", "refresh", "lineage"):
        if key not in manifest:
            _fail(McAdapterErrorCode.MANIFEST_INVALID, f"manifest missing required field {key!r}")
    dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), dict) else {}
    try:
        symbol = normalize_symbol(str(dataset.get("symbol") or ""))
    except Exception:
        symbol = str(dataset.get("symbol"))
    if symbol != MC_DATASET_SYMBOL or dataset.get("version_id") != MC_VERSION_ID:
        _fail(McAdapterErrorCode.DATASET_PIN_MISMATCH, "manifest dataset pin does not match mc male-cns_v1.0",
              symbol=dataset.get("symbol"), version_id=dataset.get("version_id"))
    source = dataset.get("source") if isinstance(dataset.get("source"), dict) else {}
    license_info = source.get("license") if isinstance(source.get("license"), dict) else {}
    spdx = str(license_info.get("spdx_id") or "").strip()
    if spdx not in MC_ALLOWED_LICENSES:
        _fail(McAdapterErrorCode.LICENSE_NOT_PERMITTED, "manifest license is missing or not permitted",
              spdx_id=spdx, allowed=sorted(MC_ALLOWED_LICENSES))
    scope = manifest.get("scope") if isinstance(manifest.get("scope"), dict) else {}
    if scope.get("sex") != "male" or scope.get("stage") != "adult" or scope.get("anatomy") != "brain_and_ventral_nerve_cord":
        _fail(McAdapterErrorCode.DATASET_PIN_MISMATCH, "manifest scope is not adult male brain_and_ventral_nerve_cord",
              scope=dict(scope))
    artifact = manifest.get("artifact") if isinstance(manifest.get("artifact"), dict) else {}
    relative_root = str(artifact.get("relative_root") or "").replace("\\", "/").strip("/")
    if not is_safe_relative_path(relative_root):
        _fail(McAdapterErrorCode.PATH_ESCAPE, "artifact.relative_root is unsafe", relative_root=relative_root)
    if relative_root != MC_SNAPSHOT_RELATIVE_ROOT:
        _fail(McAdapterErrorCode.DATASET_PIN_MISMATCH, "artifact.relative_root does not match the mc pin",
              relative_root=relative_root, expected=MC_SNAPSHOT_RELATIVE_ROOT)

    integrity = manifest.get("integrity") if isinstance(manifest.get("integrity"), dict) else {}
    verification = integrity.get("verification") if isinstance(integrity.get("verification"), dict) else {}
    status = str(verification.get("status") or "").strip().lower()
    if status not in {"pending", "verified", "quarantined"}:
        _fail(McAdapterErrorCode.MANIFEST_INVALID, "integrity.verification.status invalid", status=status)
    if status != "verified":
        _fail(McAdapterErrorCode.INTEGRITY_MISMATCH, "manifest verification status must be verified", status=status)
    if not str(verification.get("verified_at") or "").strip():
        _fail(McAdapterErrorCode.MANIFEST_INVALID, "verified_at is required when status is verified")
    expected = str(integrity.get("manifest_sha256") or "").strip()
    if not _SHA256_HEX_RE.fullmatch(expected):
        _fail(McAdapterErrorCode.MANIFEST_INVALID, "integrity.manifest_sha256 must be lowercase 64-hex")
    actual = manifest_digest(manifest)
    if actual != expected:
        _fail(McAdapterErrorCode.INTEGRITY_MISMATCH, "manifest self-hash mismatch", expected=expected, actual=actual)

    refresh = manifest.get("refresh") if isinstance(manifest.get("refresh"), dict) else {}
    decision = str(refresh.get("decision") or "").strip().lower()
    if decision not in _REFRESH_DECISIONS:
        _fail(McAdapterErrorCode.MANIFEST_INVALID, "refresh.decision invalid", decision=decision)
    if decision == "rollback":
        _fail(McAdapterErrorCode.PROMOTION_STATE_INVALID, "refresh.decision indicates rollback")
    due = refresh.get("next_check_due")
    if due:
        try:
            due_dt = datetime.fromisoformat(str(due).replace("Z", "+00:00"))
        except ValueError:
            _fail(McAdapterErrorCode.MANIFEST_INVALID, "refresh.next_check_due not RFC3339", next_check_due=due)
        if due_dt < (now or datetime.now(timezone.utc)):
            _fail(McAdapterErrorCode.PROMOTION_STATE_INVALID, "refresh.next_check_due has elapsed", next_check_due=due)

    files = integrity.get("files")
    if not isinstance(files, list) or not files:
        _fail(McAdapterErrorCode.MANIFEST_INVALID, "integrity.files must be a non-empty list")
    root = Path(snapshot_root).resolve(strict=False)
    role_of_path = {rel: role for role, rel in MC_PRODUCT_PATHS.items()}
    seen: set[str] = set()
    hashes: dict[str, str] = {}
    listed: list[tuple[str, Path, str]] = []
    for entry in files:
        if not isinstance(entry, dict):
            _fail(McAdapterErrorCode.MANIFEST_INVALID, "integrity.files entries must be objects")
        rel = str(entry.get("relative_path") or "").strip()
        sha = str(entry.get("sha256") or "").strip()
        size = entry.get("size_bytes")
        if not is_safe_relative_path(rel):
            _fail(McAdapterErrorCode.PATH_ESCAPE, "integrity file path is unsafe", relative_path=rel)
        folded = rel.replace("\\", "/").casefold()
        if folded in seen:
            _fail(McAdapterErrorCode.MANIFEST_INVALID, "duplicate integrity path after case-folding", relative_path=rel)
        seen.add(folded)
        if rel.lower().endswith(_BLOCKED_ARTIFACT_SUFFIXES):
            _fail(McAdapterErrorCode.MANIFEST_INVALID, "partial/temporary artifact listed", relative_path=rel)
        if not _SHA256_HEX_RE.fullmatch(sha):
            _fail(McAdapterErrorCode.MANIFEST_INVALID, "file sha256 must be lowercase 64-hex", relative_path=rel)
        normalized_rel = rel.replace("\\", "/")
        listed_role = entry.get("role")
        if normalized_rel in role_of_path and listed_role is not None and listed_role != role_of_path[normalized_rel]:
            _fail(McAdapterErrorCode.MANIFEST_INVALID, "manifest role disagrees with the adapter product map",
                  relative_path=rel, listed_role=listed_role, expected_role=role_of_path[normalized_rel])
        path = (root / Path(rel)).resolve(strict=False)
        if not path.is_relative_to(root):
            _fail(McAdapterErrorCode.PATH_ESCAPE, "integrity file escapes snapshot root", relative_path=rel)
        if not path.is_file():
            _fail(McAdapterErrorCode.PRODUCT_MISSING, "integrity file is missing", relative_path=rel, path=str(path))
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            _fail(McAdapterErrorCode.MANIFEST_INVALID, "size_bytes must be a non-negative integer", relative_path=rel)
        if path.stat().st_size != size:
            _fail(McAdapterErrorCode.INTEGRITY_MISMATCH, "file size mismatch", relative_path=rel,
                  expected=size, actual=path.stat().st_size)
        hashes[normalized_rel] = sha
        listed.append((normalized_rel, path, sha))
    for role in required_roles:
        rel = MC_PRODUCT_PATHS[role]
        if rel not in hashes:
            _fail(McAdapterErrorCode.PRODUCT_MISSING, f"manifest does not list required product {role!r}",
                  role=role, relative_path=rel)

    # Content hashing runs only after every metadata, path and size check passed.
    required_paths = {MC_PRODUCT_PATHS[role] for role in required_roles}
    for rel, path, sha in listed:
        if verify_hashes is False or (verify_hashes is None and rel not in required_paths):
            if hash_report is not None:
                hash_report[rel] = METHOD_SIZE_ONLY
            continue
        check = verify_file_sha256(
            path,
            relative_path=rel,
            expected_sha256=sha,
            cache=stamp_cache,
            force=verify_hashes is True,
            hasher=sha256_file,
        )
        if not check.matched:
            _fail(McAdapterErrorCode.INTEGRITY_MISMATCH, "file sha256 mismatch", relative_path=rel,
                  expected=sha, actual=check.actual_sha256)
        if hash_report is not None:
            hash_report[rel] = check.method
        if check.warning and warnings is not None:
            warnings.append(check.warning)
    return hashes


def open_mc_snapshot(
    storage_root: str | Path | None = None,
    *,
    required_roles: Sequence[str] = tuple(MC_PRODUCT_PATHS),
    verify_hashes: bool | None = None,
    now: datetime | None = None,
    stamp_dir: str | Path | None = None,
) -> McSnapshot:
    """Resolve + verify the pinned mc snapshot. Raises ``McAdapterError`` on any gap.

    The manifest self-hash and the ``manifest.sha256`` sidecar are checked on
    every open, before any file content is hashed. File contents are then
    hashed as described in ``validate_mc_manifest``. Stamps live under
    ``<snapshot>/manifest/hash-stamps/`` unless ``stamp_dir`` points elsewhere
    (e.g. a scratch dir when the snapshot must stay untouched).
    """
    for role in required_roles:
        if role not in MC_PRODUCT_PATHS:
            _fail(McAdapterErrorCode.PRODUCT_MISSING, f"unknown product role {role!r}", role=role)
    root, snapshot_root = resolve_mc_snapshot_root(storage_root)
    if not snapshot_root.is_dir():
        _fail(McAdapterErrorCode.SNAPSHOT_MISSING, "mc snapshot directory not found", snapshot_root=str(snapshot_root))
    manifest_path = snapshot_root / MC_MANIFEST_RELATIVE_PATH
    if not manifest_path.is_file():
        _fail(McAdapterErrorCode.MANIFEST_MISSING, "mc manifest.json not found", manifest_path=str(manifest_path))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise McAdapterError(McAdapterErrorCode.MANIFEST_INVALID, f"manifest unreadable: {exc}") from exc
    if not isinstance(manifest, dict):
        _fail(McAdapterErrorCode.MANIFEST_INVALID, "manifest must be a JSON object")
    sidecar = snapshot_root / MC_MANIFEST_SIDECAR_RELATIVE_PATH
    # Fail closed: the sidecar is the only integrity anchor outside manifest.json.
    if not sidecar.is_file():
        _fail(McAdapterErrorCode.MANIFEST_MISSING, "manifest.sha256 sidecar is missing", sidecar=str(sidecar))
    integrity = manifest.get("integrity") if isinstance(manifest.get("integrity"), dict) else {}
    if sidecar.read_text(encoding="utf-8").strip() != str(integrity.get("manifest_sha256") or ""):
        _fail(McAdapterErrorCode.INTEGRITY_MISMATCH, "manifest.sha256 sidecar does not match manifest")
    warnings: list[str] = []
    hash_report: dict[str, str] = {}
    cache = HashStampCache(Path(stamp_dir)) if stamp_dir is not None else HashStampCache.for_snapshot(snapshot_root)
    hashes = validate_mc_manifest(
        manifest,
        snapshot_root,
        required_roles=required_roles,
        verify_hashes=verify_hashes,
        now=now,
        stamp_cache=cache,
        hash_report=hash_report,
        warnings=warnings,
    )
    product_paths = {role: (snapshot_root / MC_PRODUCT_PATHS[role]).resolve(strict=False) for role in required_roles}
    product_sha = {role: hashes[MC_PRODUCT_PATHS[role]] for role in required_roles}
    return McSnapshot(
        storage_root=root,
        snapshot_root=snapshot_root,
        manifest_path=manifest_path,
        manifest_id=str(manifest.get("manifest_id")),
        manifest_sha256=str(manifest["integrity"]["manifest_sha256"]),
        product_paths=product_paths,
        product_sha256=product_sha,
        license_spdx=str(manifest["dataset"]["source"]["license"]["spdx_id"]),
        warnings=tuple(warnings),
        hash_verification=dict(sorted(hash_report.items())),
    )


# ---------------------------------------------------------------------------
# Readers (local files only)
# ---------------------------------------------------------------------------


def _require_file(path: Path, role: str) -> Path:
    candidate = Path(path).resolve(strict=False)
    if not candidate.is_file():
        _fail(McAdapterErrorCode.PRODUCT_MISSING, f"mc {role} file not found", path=str(candidate))
    return candidate


def _feather_columns(path: Path) -> list[str]:
    """Read only the Arrow IPC schema (no column decompression)."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    try:
        with pa.memory_map(str(path), "r") as source:
            return list(ipc.open_file(source).schema.names)
    except (OSError, pa.ArrowInvalid) as exc:
        raise McAdapterError(
            McAdapterErrorCode.SCHEMA_MISMATCH, f"not a readable Arrow/feather v2 file: {exc}", {"path": str(path)}
        ) from exc


def _check_columns(available: Iterable[str], required: Sequence[str], role: str, path: Path) -> None:
    have = set(available)
    missing = [col for col in required if col not in have]
    if missing:
        _fail(McAdapterErrorCode.SCHEMA_MISMATCH, f"mc {role} missing required columns: {', '.join(missing)}",
              role=role, path=str(path), missing=missing)


def _id_array(body_ids: Iterable[Any] | None):
    """Normalize a body-id filter to a sorted, unique int64 Arrow array (or None)."""
    if body_ids is None:
        return None
    import pyarrow as pa

    values = sorted({int(value) for value in body_ids})
    return pa.array(values, type=pa.int64())


def _scan(path: Path, columns: Sequence[str], id_column: str, body_ids):
    """Stream record batches with column projection and an optional id filter."""
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(path), format="ipc")
    expression = None if body_ids is None else ds.field(id_column).isin(body_ids)
    return dataset.to_batches(columns=list(columns), filter=expression)


def _ids_to_str(column):
    import pyarrow as pa

    return column.cast(pa.int64()).cast(pa.string())


def load_mc_annotations(path: Path, *, columns: Sequence[str] = META_REQUIRED_COLUMNS,
                        optional_columns: Sequence[str] = ()):
    """Per-body annotation table (~212k rows) with ``bodyId`` as a decimal string."""
    import pyarrow.feather as feather

    source = _require_file(path, "meta")
    names = _feather_columns(source)
    wanted = list(dict.fromkeys(["bodyId", *columns]))
    _check_columns(names, wanted, "meta", source)
    wanted += [col for col in optional_columns if col in names and col not in wanted]
    table = feather.read_table(str(source), columns=wanted)
    if table.column("bodyId").null_count:
        _fail(McAdapterErrorCode.SCHEMA_MISMATCH, "meta has null bodyId", path=str(source))
    table = table.set_column(table.schema.get_field_index("bodyId"), "bodyId", _ids_to_str(table.column("bodyId")))
    for index, fld in enumerate(table.schema):
        # statusLabel-style dictionary columns -> plain strings for pandas.
        if str(fld.type).startswith("dictionary"):
            table = table.set_column(index, fld.name, table.column(fld.name).cast("string"))
    frame = table.to_pandas()
    if frame["bodyId"].duplicated().any():
        _fail(McAdapterErrorCode.SCHEMA_MISMATCH, "meta bodyId is not unique", path=str(source))
    return frame


def load_mc_nt_predictions(path: Path, *, body_ids: Iterable[Any] | None = None):
    """Per-body NT predictions (``body`` as string), optionally filtered to ``body_ids``."""
    import pyarrow as pa

    source = _require_file(path, "nt_prediction")
    _check_columns(_feather_columns(source), NT_REQUIRED_COLUMNS, "nt_prediction", source)
    batches = list(_scan(source, NT_REQUIRED_COLUMNS, "body", _id_array(body_ids)))
    table = pa.Table.from_batches(batches) if batches else None
    if table is None or table.num_rows == 0:
        import pandas as pd

        return pd.DataFrame({col: pd.Series(dtype="object") for col in NT_REQUIRED_COLUMNS})
    if table.column("body").null_count:
        _fail(McAdapterErrorCode.SCHEMA_MISMATCH, "nt_prediction has null body", path=str(source))
    table = table.set_column(table.schema.get_field_index("body"), "body", _ids_to_str(table.column("body")))
    frame = table.to_pandas()
    if frame["body"].duplicated().any():
        _fail(McAdapterErrorCode.SCHEMA_MISMATCH, "nt_prediction body is not unique", path=str(source))
    return frame.sort_values("body", kind="mergesort").reset_index(drop=True)


def load_mc_outgoing_totals(path: Path, *, body_ids: Iterable[Any] | None = None):
    """Aggregate the neuron-neuron weight table per presynaptic body, streaming.

    Returns ``(frame, scanned_edges)``: ``frame`` has ``root_id`` (str),
    ``total_out_synapses`` (sum of ``weight``) and ``n_post_partners``
    (edge rows, i.e. distinct ``body_post``), sorted by root id. Only rows
    whose ``body_pre`` is in ``body_ids`` are read (all rows when ``None``);
    each record batch is aggregated on its own so memory stays bounded.
    """
    import pyarrow as pa

    source = _require_file(path, "edgelist")
    _check_columns(_feather_columns(source), EDGELIST_REQUIRED_COLUMNS, "edgelist", source)
    partials = []
    scanned = 0
    for batch in _scan(source, ("body_pre", "weight"), "body_pre", _id_array(body_ids)):
        if batch.num_rows == 0:
            continue
        scanned += batch.num_rows
        if batch.column("body_pre").null_count or batch.column("weight").null_count:
            _fail(McAdapterErrorCode.SCHEMA_MISMATCH, "edgelist has null body_pre/weight values", path=str(source))
        slim = pa.table({"body_pre": batch.column("body_pre"), "weight": batch.column("weight").cast(pa.int64())})
        partials.append(slim.group_by("body_pre").aggregate([("weight", "sum"), ("weight", "count")]))
    import pandas as pd

    if not partials:
        return pd.DataFrame({"root_id": pd.Series(dtype="object"),
                             "total_out_synapses": pd.Series(dtype="int64"),
                             "n_post_partners": pd.Series(dtype="int64")}), 0
    merged = pa.concat_tables(partials).group_by("body_pre").aggregate(
        [("weight_sum", "sum"), ("weight_count", "sum")]
    )
    frame = pd.DataFrame(
        {
            "root_id": _ids_to_str(merged.column("body_pre")).to_pylist(),
            "total_out_synapses": merged.column("weight_sum_sum").to_numpy().astype("int64"),
            "n_post_partners": merged.column("weight_count_sum").to_numpy().astype("int64"),
        }
    )
    frame["_sort"] = frame["root_id"].str.zfill(24)
    frame = frame.sort_values("_sort", kind="mergesort").drop(columns=["_sort"]).reset_index(drop=True)
    return frame, int(scanned)


@dataclass(frozen=True)
class RoiSummary:
    """Primary-ROI summary of one body's ``roiInfo`` (synapses = pre + post)."""

    top_roi: str | None
    top_synapses: int
    total_primary_synapses: int
    n_primary_rois: int
    unknown_primary_like: tuple[str, ...] = ()


def summarize_roi_info(raw: str | None) -> RoiSummary:
    """Reduce a neuPrint ``roiInfo`` JSON object to its primary-ROI summary.

    Only keys in the explicit primary vocabulary count (primary ROIs partition
    the CNS; parent/child ROIs such as ``CentralBrain`` or ``AL-DA1(L)`` would
    double count). ``top_roi`` is the primary ROI with the most pre + post
    synapses; ties break on the ROI name. Malformed JSON fails closed.
    """
    if raw is None or (isinstance(raw, float) and raw != raw) or not str(raw).strip():
        return RoiSummary(None, 0, 0, 0)
    try:
        info = json.loads(raw)
    except ValueError as exc:
        raise McAdapterError(McAdapterErrorCode.SCHEMA_MISMATCH, f"roiInfo is not valid JSON: {exc}") from exc
    if not isinstance(info, dict):
        _fail(McAdapterErrorCode.SCHEMA_MISMATCH, "roiInfo must be a JSON object")
    counts: list[tuple[int, str]] = []
    for roi, stats in info.items():
        if roi not in MC_PRIMARY_ROIS:
            continue
        if not isinstance(stats, dict):
            _fail(McAdapterErrorCode.SCHEMA_MISMATCH, "roiInfo entry must be an object", roi=roi)
        synapses = int(stats.get("pre", 0) or 0) + int(stats.get("post", 0) or 0)
        if synapses > 0:
            counts.append((synapses, roi))
    if not counts:
        return RoiSummary(None, 0, 0, 0)
    counts.sort(key=lambda item: (-item[0], item[1]))
    return RoiSummary(counts[0][1], counts[0][0], sum(c for c, _ in counts), len(counts))


def load_mc_neuron_roi_summary(path: Path, *, body_ids: Iterable[Any] | None = None):
    """Stream ``Neuprint_Neurons.feather`` for ``body_ids`` and summarize ``roiInfo``.

    Returns a frame with ``root_id`` (str), ``pre``, ``post``, ``downstream``,
    ``upstream``, ``top_roi``, ``top_roi_synapses``, ``total_primary_synapses``
    and ``n_primary_rois``, sorted by root id. Only the needed columns are
    read and rows are filtered by ``:ID(Body-ID)`` batch by batch, so the
    ~88 M-row table is never materialized. When the file also has
    ``bodyId:long`` it must equal ``:ID(Body-ID)`` on every read row.
    """
    import pandas as pd
    import pyarrow.compute as pc

    source = _require_file(path, "neuron_roi_info")
    names = _feather_columns(source)
    _check_columns(names, NEURON_REQUIRED_COLUMNS, "neuron_roi_info", source)
    columns = list(NEURON_REQUIRED_COLUMNS)
    check_body = NEURON_BODYID_COLUMN in names
    if check_body:
        columns.append(NEURON_BODYID_COLUMN)
    records: dict[str, list[Any]] = {key: [] for key in (
        "root_id", "pre", "post", "downstream", "upstream", "top_roi", "top_roi_synapses",
        "total_primary_synapses", "n_primary_rois")}
    for batch in _scan(source, columns, NEURON_ID_COLUMN, _id_array(body_ids)):
        if batch.num_rows == 0:
            continue
        ids = batch.column(NEURON_ID_COLUMN)
        if ids.null_count:
            _fail(McAdapterErrorCode.SCHEMA_MISMATCH, "neuron table has null body ids", path=str(source))
        if check_body:
            body = batch.column(NEURON_BODYID_COLUMN)
            if body.null_count or not pc.all(pc.equal(ids, body)).as_py():
                _fail(McAdapterErrorCode.SCHEMA_MISMATCH, "neuron table :ID(Body-ID) != bodyId:long", path=str(source))
        records["root_id"].extend(_ids_to_str(ids).to_pylist())
        for raw_name, short in NEURON_COUNT_COLUMNS.items():
            records[short].extend(int(v or 0) for v in batch.column(raw_name).to_pylist())
        for raw in batch.column(NEURON_ROI_INFO_COLUMN).to_pylist():
            summary = summarize_roi_info(raw)
            records["top_roi"].append(summary.top_roi)
            records["top_roi_synapses"].append(summary.top_synapses)
            records["total_primary_synapses"].append(summary.total_primary_synapses)
            records["n_primary_rois"].append(summary.n_primary_rois)
    frame = pd.DataFrame(records)
    for column in ("pre", "post", "downstream", "upstream", "top_roi_synapses", "total_primary_synapses",
                   "n_primary_rois"):
        frame[column] = frame[column].astype("int64")
    if frame["root_id"].duplicated().any():
        _fail(McAdapterErrorCode.SCHEMA_MISMATCH, "neuron table body id is not unique", path=str(source))
    frame["_sort"] = frame["root_id"].str.zfill(24)
    return frame.sort_values("_sort", kind="mergesort").drop(columns=["_sort"]).reset_index(drop=True)


def load_mc_primary_rois(path: Path) -> tuple[str, ...]:
    """``primaryRois`` from the neuPrint ``Meta`` export (one dataset row, ``;``-separated)."""
    import pandas as pd

    source = _require_file(path, "dataset_meta")
    try:
        frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    except (OSError, ValueError) as exc:
        raise McAdapterError(McAdapterErrorCode.SCHEMA_MISMATCH, f"dataset meta unreadable: {exc}",
                             {"path": str(source)}) from exc
    _check_columns(frame.columns, DATASET_META_REQUIRED_COLUMNS, "dataset_meta", source)
    if len(frame) != 1:
        _fail(McAdapterErrorCode.SCHEMA_MISMATCH, "dataset meta must have exactly one row", rows=int(len(frame)))
    row = frame.iloc[0]
    dataset = f"{row['dataset:string']}:{row['tag:string']}"
    if dataset != MC_NEUPRINT_DATASET:
        _fail(McAdapterErrorCode.DATASET_PIN_MISMATCH, "dataset meta is not male-cns:v1.0", dataset=dataset)
    rois = tuple(part.strip() for part in str(row[DATASET_META_PRIMARY_ROIS_COLUMN]).split(";") if part.strip())
    if not rois:
        _fail(McAdapterErrorCode.SCHEMA_MISMATCH, "dataset meta primaryRois is empty")
    return rois


def check_mc_roi_vocabulary(path: Path) -> tuple[str, ...]:
    """Fail closed unless the snapshot's ``primaryRois`` equal ``MC_PRIMARY_ROIS`` exactly."""
    rois = load_mc_primary_rois(path)
    have = set(rois)
    known = set(MC_PRIMARY_ROIS)
    if have != known:
        _fail(
            McAdapterErrorCode.REGION_VOCABULARY_UNKNOWN,
            "snapshot primaryRois differ from the adapter's explicit MaleCNS vocabulary",
            unknown=sorted(have - known),
            missing=sorted(known - have),
        )
    return rois


__all__ = [
    "DIVISION_BRAIN",
    "DIVISION_NECK",
    "DIVISION_NERVE_CORD",
    "MC_ALLOWED_LICENSES",
    "MC_CITATION",
    "MC_DATASET_SYMBOL",
    "MC_MANIFEST_RELATIVE_PATH",
    "MC_NT_ABSTAIN",
    "MC_NT_SHORT_CODES",
    "MC_PRIMARY_ROIS",
    "MC_PRODUCT_PATHS",
    "MC_SNAPSHOT_RELATIVE_ROOT",
    "MC_VERSION_ID",
    "McAdapterError",
    "McAdapterErrorCode",
    "McRoi",
    "McSnapshot",
    "ROLE_DATASET_META",
    "ROLE_EDGELIST",
    "ROLE_META",
    "ROLE_NEURON_ROI_INFO",
    "ROLE_NT_PREDICTION",
    "RoiSummary",
    "build_mc_manifest",
    "check_mc_roi_vocabulary",
    "load_mc_annotations",
    "load_mc_neuron_roi_summary",
    "load_mc_nt_predictions",
    "load_mc_outgoing_totals",
    "load_mc_primary_rois",
    "manifest_digest",
    "map_mc_primary_roi",
    "nt_short_code",
    "open_mc_snapshot",
    "resolve_mc_snapshot_root",
    "sha256_file",
    "summarize_roi_info",
    "validate_mc_manifest",
    "write_mc_manifest",
]
