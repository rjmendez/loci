"""Fail-closed local adapter for the pinned MANC v1.0 snapshot (dataset symbol ``mv``).

MANC (Male Adult Nerve Cord; Takemura et al. 2024, Marin et al. 2024,
Cheong et al. 2024) is the ventral nerve cord of one adult male *Drosophila*.
This module is the only place the harness touches the on-disk MANC snapshot:

* ``open_mv_snapshot`` validates the storage root, the manifest
  (``fbh-manifest/v1`` self-hash + sidecar, verification status, refresh
  window, license), the size of every listed file and the sha256 of every
  required product before returning resolved paths. Content hashes are cached
  as write-once stamps keyed by (size, mtime_ns) (``flybrain_hash_stamps``),
  so the 12 GB snapshot is not rehashed on every open.
* ``load_mv_neurons`` reads the per-body neuron-properties table with an Arrow
  filter on ``status`` and column projection. The file holds every body in the
  segmentation (about 102k rows, most of them untraced fragments); only the
  requested statuses are materialised. ``bodyId`` is returned as a string.
* ``load_mv_neuropil_synweights`` turns the ``roiInfo`` struct column into a
  dense per-neuropil synapse-weight matrix (no JSON parsing, no synapse tables).
* ``load_mv_out_partner_counts`` aggregates the traced-to-traced edge list
  with a three-column CSV projection.
* ``map_mv_roi`` is the explicit MANC ROI vocabulary. Unknown ROIs raise
  instead of being silently bucketed.

The multi-GB neuPrint synapse / synapse-set exports are listed in the manifest
and size-checked, but no reader here opens them.

No network access: everything is read from
``$LOCI_FLYBRAIN_STORAGE_ROOT/snapshots/mv/manc_v1.0``. See
``docs/FLYBRAIN_MV_ADAPTER_CONTRACT.md``.
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

MV_DATASET_SYMBOL = "mv"
MV_VERSION_ID = "manc_v1.0"
MV_SNAPSHOT_DIR_NAME = "mv"
MV_SNAPSHOT_RELATIVE_ROOT = "snapshots/mv/manc_v1.0"
MV_MANIFEST_SCHEMA_VERSION = "fbh-manifest/v1"
MV_MANIFEST_RELATIVE_PATH = "manifest/manifest.json"
MV_MANIFEST_SIDECAR_RELATIVE_PATH = "manifest/manifest.sha256"
MV_ALLOWED_LICENSES = frozenset({"CC-BY-4.0"})
MV_CITATION = (
    "Takemura S, Hayworth KJ, Huang GB, et al. (2024). A connectome of the male Drosophila ventral nerve "
    "cord. eLife 13:RP97769. doi:10.7554/eLife.97769; Marin EC, Morris BJ, Stuerner T, et al. (2024). "
    "eLife 13:RP97766. doi:10.7554/eLife.97766; Cheong HSJ, Eichler K, Stuerner T, et al. (2024). "
    "eLife 13:RP96084. doi:10.7554/eLife.96084. Data: FlyEM MANC v1.0, gs://flyem-manc-exports/v1.0 "
    "(CC BY 4.0)."
)

# Product roles -> snapshot-relative paths (relative to MV_SNAPSHOT_RELATIVE_ROOT).
ROLE_META = "meta"
ROLE_EDGELIST = "edgelist"
MV_PRODUCT_PATHS: Mapping[str, str] = {
    ROLE_META: "metadata/manc-v1.0-neuron-properties.feather",
    ROLE_EDGELIST: "source/manc-traced-adjacencies-v1.0/traced-connections.csv",
}
# Opened by default: the builders need meta (+ edgelist for the NT out-partner feature).
MV_DEFAULT_REQUIRED_ROLES: tuple[str, ...] = (ROLE_META, ROLE_EDGELIST)

META_REQUIRED_COLUMNS: tuple[str, ...] = (
    "bodyId",
    "status",
    "type",
    "class",
    "pre",
    "downstream",
    "upstream",
    "somaNeuromere",
    "somaSide",
    "rootSide",
    "hemilineage",
    "birthtime",
    "group",
    "predictedNt",
    "predictedNtProb",
    "ntAcetylcholineProb",
    "ntGabaProb",
    "ntGlutamateProb",
    "ntUnknownProb",
    "roiInfo",
)
EDGELIST_REQUIRED_COLUMNS: tuple[str, ...] = ("bodyId_pre", "bodyId_post", "weight")

# neuPrint ``predictedNt`` value -> short code (codes shared with fw/banc).
# ``unknown`` is a real class of the MANC classifier; it is never a label.
MV_NT_SHORT_CODES: Mapping[str, str] = {
    "acetylcholine": "ach",
    "gaba": "gaba",
    "glutamate": "glut",
}
MV_NT_UNKNOWN = "unknown"
# predictedNt value -> probability column; predictedNt must be the argmax of these.
MV_NT_PROBABILITY_COLUMNS: Mapping[str, str] = {
    "acetylcholine": "ntAcetylcholineProb",
    "gaba": "ntGabaProb",
    "glutamate": "ntGlutamateProb",
    "unknown": "ntUnknownProb",
}

# ---------------------------------------------------------------------------
# ROI vocabulary (MANC v1.0 all_ROIs.txt; side suffix "(L)"/"(R)" stripped first)
# ---------------------------------------------------------------------------

ROI_KIND_NEUROPIL = "neuropil"
ROI_KIND_NERVE = "nerve"
ROI_KIND_CONNECTIVE = "connective"
ROI_KIND_TRACT = "tract"

# base name -> (kind, family). Families group neuropils for documentation only;
# region ids keep the neuromere (``legnp_t1`` != ``legnp_t2``).
MV_ROI_VOCABULARY: Mapping[str, tuple[str, str]] = {
    # Neuropils
    "ANm": (ROI_KIND_NEUROPIL, "abdominal_neuromeres"),
    "IntTct": (ROI_KIND_NEUROPIL, "tectulum"),
    "LTct": (ROI_KIND_NEUROPIL, "tectulum"),
    "NTct(UTct-T1)": (ROI_KIND_NEUROPIL, "upper_tectulum"),
    "WTct(UTct-T2)": (ROI_KIND_NEUROPIL, "upper_tectulum"),
    "HTct(UTct-T3)": (ROI_KIND_NEUROPIL, "upper_tectulum"),
    "LegNp(T1)": (ROI_KIND_NEUROPIL, "leg_neuropil"),
    "LegNp(T2)": (ROI_KIND_NEUROPIL, "leg_neuropil"),
    "LegNp(T3)": (ROI_KIND_NEUROPIL, "leg_neuropil"),
    "mVAC(T1)": (ROI_KIND_NEUROPIL, "ventral_association_centre"),
    "mVAC(T2)": (ROI_KIND_NEUROPIL, "ventral_association_centre"),
    "mVAC(T3)": (ROI_KIND_NEUROPIL, "ventral_association_centre"),
    "Ov": (ROI_KIND_NEUROPIL, "ovoid"),
    # Connective / tract
    "CV": (ROI_KIND_CONNECTIVE, "cervical_connective"),
    "GF": (ROI_KIND_TRACT, "giant_fiber"),
    # Nerves
    "ADMN": (ROI_KIND_NERVE, "nerve"),
    "AbN1": (ROI_KIND_NERVE, "nerve"),
    "AbN2": (ROI_KIND_NERVE, "nerve"),
    "AbN3": (ROI_KIND_NERVE, "nerve"),
    "AbN4": (ROI_KIND_NERVE, "nerve"),
    "AbNT": (ROI_KIND_NERVE, "nerve"),
    "CvN": (ROI_KIND_NERVE, "nerve"),
    "DMetaN": (ROI_KIND_NERVE, "nerve"),
    "DProN": (ROI_KIND_NERVE, "nerve"),
    "MesoAN": (ROI_KIND_NERVE, "nerve"),
    "MesoLN": (ROI_KIND_NERVE, "nerve"),
    "MetaLN": (ROI_KIND_NERVE, "nerve"),
    "PDMN": (ROI_KIND_NERVE, "nerve"),
    "PrN": (ROI_KIND_NERVE, "nerve"),
    "ProAN": (ROI_KIND_NERVE, "nerve"),
    "ProCN": (ROI_KIND_NERVE, "nerve"),
    "ProLN": (ROI_KIND_NERVE, "nerve"),
    "VProN": (ROI_KIND_NERVE, "nerve"),
}

_SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")
_ROI_SIDE_RE = re.compile(r"\((L|R)\)$")
_REGION_SANITIZE_RE = re.compile(r"[^a-z0-9]+")
_BLOCKED_ARTIFACT_SUFFIXES = (".partial", ".tmp", ".inprogress")
_REFRESH_DECISIONS = frozenset({"no_change", "patch_refresh", "major_bump", "rollback"})


class MvAdapterErrorCode(str, Enum):
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


class MvAdapterError(ValueError):
    """Typed fail-closed error; message is prefixed with ``[CODE]``."""

    def __init__(self, code: MvAdapterErrorCode, message: str, details: Mapping[str, Any] | None = None):
        self.code = code
        self.message = message
        self.details = dict(details or {})
        super().__init__(f"[{code.value}] {message}")


def _fail(code: MvAdapterErrorCode, message: str, **details: Any) -> None:
    raise MvAdapterError(code, message, details)


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MvRoi:
    roi: str
    base: str
    kind: str
    family: str
    side: str | None
    region_id: str


def _sanitize(raw: str) -> str:
    value = _REGION_SANITIZE_RE.sub("_", str(raw).strip().lower()).strip("_")
    return value or "unknown_region"


def map_mv_roi(roi: str) -> MvRoi:
    """Map a MANC ROI name onto the explicit vocabulary.

    ``region_id`` is ``vnc_<base>`` with the side suffix removed
    (``LegNp(T1)(L)`` -> ``vnc_legnp_t1``, ``HTct(UTct-T3)(R)`` ->
    ``vnc_htct_utct_t3``, ``ANm`` -> ``vnc_anm``). Unknown names raise
    ``REGION_VOCABULARY_UNKNOWN``.
    """
    text = str(roi or "").strip()
    if not text:
        _fail(MvAdapterErrorCode.REGION_VOCABULARY_UNKNOWN, "ROI name is empty")
    side = None
    base = text
    match = _ROI_SIDE_RE.search(text)
    if match:
        side = "left" if match.group(1) == "L" else "right"
        base = text[: match.start()]
    if base not in MV_ROI_VOCABULARY:
        _fail(MvAdapterErrorCode.REGION_VOCABULARY_UNKNOWN, f"ROI {text!r} is not in the MANC vocabulary",
              roi=text, known=sorted(MV_ROI_VOCABULARY))
    kind, family = MV_ROI_VOCABULARY[base]
    return MvRoi(roi=text, base=base, kind=kind, family=family, side=side, region_id=_sanitize(f"vnc_{base}"))


def nt_short_code(name: str) -> str:
    """``acetylcholine`` -> ``ach`` etc. ``unknown`` and anything else raise."""
    text = str(name or "").strip().lower()
    if text not in MV_NT_SHORT_CODES:
        _fail(MvAdapterErrorCode.NT_VOCABULARY_UNKNOWN, f"neurotransmitter {name!r} has no MANC label code",
              neurotransmitter=name, known=sorted(MV_NT_SHORT_CODES))
    return MV_NT_SHORT_CODES[text]


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


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
        _fail(MvAdapterErrorCode.MANIFEST_INVALID, "manifest.integrity must be an object")
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


def resolve_mv_snapshot_root(storage_root: str | Path | None = None) -> tuple[Path, Path]:
    """Return ``(storage_root, snapshot_root)`` after root + path-escape validation."""
    try:
        layout = build_flybrain_harness_layout(root_override=storage_root, create=False)
    except ValueError as exc:
        raise MvAdapterError(MvAdapterErrorCode.ROOT_NOT_CONFIGURED, str(exc)) from exc
    spec = get_dataset(MV_DATASET_SYMBOL)
    if spec.pinned_version != MV_VERSION_ID or spec.snapshot_dir_name != MV_SNAPSHOT_DIR_NAME:
        _fail(MvAdapterErrorCode.DATASET_PIN_MISMATCH, "registry pin for mv does not match the adapter pin",
              registry_version=spec.pinned_version, adapter_version=MV_VERSION_ID)
    snapshot_root = snapshot_version_root(MV_DATASET_SYMBOL, layout.root).resolve(strict=False)
    if not snapshot_root.is_relative_to(layout.root.resolve(strict=False)):
        _fail(MvAdapterErrorCode.PATH_ESCAPE, "mv snapshot path escapes the configured storage root",
              snapshot_root=str(snapshot_root))
    return layout.root, snapshot_root


def build_mv_manifest(
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
    """Build a verified ``fbh-manifest/v1`` dict for the MANC snapshot (self-hash applied)."""
    generated = generated_at or _utc_now()
    entries = []
    for item in sorted(files, key=lambda f: str(f["relative_path"])):
        entry = {
            "relative_path": str(item["relative_path"]).replace("\\", "/"),
            "size_bytes": int(item["size_bytes"]),
            "sha256": str(item["sha256"]),
        }
        for key in ("url", "retrieved_at", "role", "md5_b64"):
            if key in item:
                entry[key] = item[key]
        entries.append(entry)
    source: dict[str, Any] = {
        "system": "flyem-manc-gcs-public",
        "access_method": "https_public_bucket_selective_pull",
        "uri": source_uri,
        "retrieved_at": retrieved_at,
        "license": {"spdx_id": license_spdx, "url": "https://creativecommons.org/licenses/by/4.0/"},
        "citation": MV_CITATION,
        "doi": "10.7554/eLife.97769",
    }
    source.update(dict(extra_source or {}))
    manifest: dict[str, Any] = {
        "schema_version": MV_MANIFEST_SCHEMA_VERSION,
        "manifest_id": f"fbh-mv-{MV_VERSION_ID}-{generated}",
        "generated_at": generated,
        "storage_root_env": "LOCI_FLYBRAIN_STORAGE_ROOT",
        "artifact": {
            "kind": "dataset_snapshot",
            "relative_root": MV_SNAPSHOT_RELATIVE_ROOT,
            "path_template": "$LOCI_FLYBRAIN_STORAGE_ROOT\\snapshots\\mv\\manc_v1.0\\",
        },
        "dataset": {
            "symbol": "mv",
            "label": "Male Adult Nerve Cord (MANC) connectome v1.0 (FlyEM flat exports, selective products)",
            "version_id": MV_VERSION_ID,
            "source": source,
        },
        "scope": {
            "sex": "male",
            "stage": "adult",
            "anatomy": "ventral_nerve_cord",
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
            "derived_from": [{"type": "remote_dataset_release", "id": "gs://flyem-manc-exports/v1.0"}],
            "pipeline": {
                "job_name": "fbh_manc_selective_pull",
                "job_version": "2026.09.24.1",
                "run_id": run_id,
                "run_mode": "execute",
            },
        },
        "notes": list(notes),
    }
    manifest["integrity"]["manifest_sha256"] = manifest_digest(manifest)
    return manifest


def write_mv_manifest(snapshot_root: Path, manifest: Mapping[str, Any]) -> Path:
    """Write manifest.json + manifest.sha256; refuses to overwrite (MANIFEST_EXISTS)."""
    target = Path(snapshot_root) / MV_MANIFEST_RELATIVE_PATH
    sidecar = Path(snapshot_root) / MV_MANIFEST_SIDECAR_RELATIVE_PATH
    for path in (target, sidecar):
        if path.exists():
            _fail(MvAdapterErrorCode.MANIFEST_EXISTS,
                  "refusing to overwrite an existing mv manifest; supersede with a new version instead",
                  path=str(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sidecar.write_text(str(manifest["integrity"]["manifest_sha256"]) + "\n", encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Snapshot open / verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MvSnapshot:
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
            _fail(MvAdapterErrorCode.PRODUCT_MISSING, f"product role {role!r} not in snapshot", role=role)
        return self.product_paths[role]

    def provenance(self) -> dict[str, Any]:
        return {
            "dataset_symbol": MV_DATASET_SYMBOL,
            "version_id": MV_VERSION_ID,
            "manifest_id": self.manifest_id,
            "manifest_sha256": self.manifest_sha256,
            "source_path": MV_SNAPSHOT_RELATIVE_ROOT,
            "access_method": "local_snapshot_read_only",
            "license": self.license_spdx,
            "product_sha256": dict(sorted(self.product_sha256.items())),
        }


def validate_mv_manifest(
    manifest: Mapping[str, Any],
    snapshot_root: Path,
    *,
    required_roles: Sequence[str] = MV_DEFAULT_REQUIRED_ROLES,
    verify_hashes: bool | None = None,
    now: datetime | None = None,
    stamp_cache: HashStampCache | None = None,
    hash_report: dict[str, str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, str]:
    """Fail-closed manifest check. Returns ``{relative_path: sha256}`` for listed files.

    Every listed file is always checked for presence, path safety and exact
    ``size_bytes``. Content hashing depends on ``verify_hashes``:

    * ``True`` (explicit verify): sha256 of every listed file, always (about
      12 GB for this snapshot).
    * ``None`` (default): sha256 of the files backing ``required_roles`` only,
      skipped when ``stamp_cache`` holds a write-once stamp for the file's
      current ``(size, mtime_ns)``. Other listed files are size-checked only.
    * ``False``: size checks only (smoke runs).

    ``hash_report`` (if given) receives ``relative_path -> method``.
    """
    if manifest.get("schema_version") != MV_MANIFEST_SCHEMA_VERSION:
        _fail(MvAdapterErrorCode.MANIFEST_INVALID, "manifest schema_version mismatch",
              schema_version=manifest.get("schema_version"))
    for key in ("manifest_id", "generated_at", "artifact", "dataset", "scope", "integrity", "refresh", "lineage"):
        if key not in manifest:
            _fail(MvAdapterErrorCode.MANIFEST_INVALID, f"manifest missing required field {key!r}")
    dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), dict) else {}
    try:
        symbol = normalize_symbol(str(dataset.get("symbol") or ""))
    except Exception:
        symbol = str(dataset.get("symbol"))
    if symbol != MV_DATASET_SYMBOL or dataset.get("version_id") != MV_VERSION_ID:
        _fail(MvAdapterErrorCode.DATASET_PIN_MISMATCH, "manifest dataset pin does not match mv manc_v1.0",
              symbol=dataset.get("symbol"), version_id=dataset.get("version_id"))
    source = dataset.get("source") if isinstance(dataset.get("source"), dict) else {}
    license_info = source.get("license") if isinstance(source.get("license"), dict) else {}
    spdx = str(license_info.get("spdx_id") or "").strip()
    if spdx not in MV_ALLOWED_LICENSES:
        _fail(MvAdapterErrorCode.LICENSE_NOT_PERMITTED, "manifest license is missing or not permitted",
              spdx_id=spdx, allowed=sorted(MV_ALLOWED_LICENSES))
    scope = manifest.get("scope") if isinstance(manifest.get("scope"), dict) else {}
    if str(scope.get("stage") or "").strip().lower() != "adult" or \
            str(scope.get("anatomy") or "").strip().lower() != "ventral_nerve_cord":
        _fail(MvAdapterErrorCode.DATASET_PIN_MISMATCH, "manifest scope must be adult ventral_nerve_cord",
              stage=scope.get("stage"), anatomy=scope.get("anatomy"))
    artifact = manifest.get("artifact") if isinstance(manifest.get("artifact"), dict) else {}
    relative_root = str(artifact.get("relative_root") or "").replace("\\", "/").strip("/")
    if not is_safe_relative_path(relative_root):
        _fail(MvAdapterErrorCode.PATH_ESCAPE, "artifact.relative_root is unsafe", relative_root=relative_root)
    if relative_root != MV_SNAPSHOT_RELATIVE_ROOT:
        _fail(MvAdapterErrorCode.DATASET_PIN_MISMATCH, "artifact.relative_root does not match the mv pin",
              relative_root=relative_root, expected=MV_SNAPSHOT_RELATIVE_ROOT)

    integrity = manifest.get("integrity") if isinstance(manifest.get("integrity"), dict) else {}
    verification = integrity.get("verification") if isinstance(integrity.get("verification"), dict) else {}
    status = str(verification.get("status") or "").strip().lower()
    if status not in {"pending", "verified", "quarantined"}:
        _fail(MvAdapterErrorCode.MANIFEST_INVALID, "integrity.verification.status invalid", status=status)
    if status != "verified":
        _fail(MvAdapterErrorCode.INTEGRITY_MISMATCH, "manifest verification status must be verified", status=status)
    if not str(verification.get("verified_at") or "").strip():
        _fail(MvAdapterErrorCode.MANIFEST_INVALID, "verified_at is required when status is verified")
    expected = str(integrity.get("manifest_sha256") or "").strip()
    if not _SHA256_HEX_RE.fullmatch(expected):
        _fail(MvAdapterErrorCode.MANIFEST_INVALID, "integrity.manifest_sha256 must be lowercase 64-hex")
    actual = manifest_digest(manifest)
    if actual != expected:
        _fail(MvAdapterErrorCode.INTEGRITY_MISMATCH, "manifest self-hash mismatch", expected=expected, actual=actual)

    refresh = manifest.get("refresh") if isinstance(manifest.get("refresh"), dict) else {}
    decision = str(refresh.get("decision") or "").strip().lower()
    if decision not in _REFRESH_DECISIONS:
        _fail(MvAdapterErrorCode.MANIFEST_INVALID, "refresh.decision invalid", decision=decision)
    if decision == "rollback":
        _fail(MvAdapterErrorCode.PROMOTION_STATE_INVALID, "refresh.decision indicates rollback")
    due = refresh.get("next_check_due")
    if due:
        try:
            due_dt = datetime.fromisoformat(str(due).replace("Z", "+00:00"))
        except ValueError:
            _fail(MvAdapterErrorCode.MANIFEST_INVALID, "refresh.next_check_due not RFC3339", next_check_due=due)
        if due_dt < (now or datetime.now(timezone.utc)):
            _fail(MvAdapterErrorCode.PROMOTION_STATE_INVALID, "refresh.next_check_due has elapsed",
                  next_check_due=due)

    files = integrity.get("files")
    if not isinstance(files, list) or not files:
        _fail(MvAdapterErrorCode.MANIFEST_INVALID, "integrity.files must be a non-empty list")
    root = Path(snapshot_root).resolve(strict=False)
    seen: set[str] = set()
    hashes: dict[str, str] = {}
    listed: list[tuple[str, Path, str]] = []
    for entry in files:
        if not isinstance(entry, dict):
            _fail(MvAdapterErrorCode.MANIFEST_INVALID, "integrity.files entries must be objects")
        rel = str(entry.get("relative_path") or "").strip()
        sha = str(entry.get("sha256") or "").strip()
        size = entry.get("size_bytes")
        if not is_safe_relative_path(rel):
            _fail(MvAdapterErrorCode.PATH_ESCAPE, "integrity file path is unsafe", relative_path=rel)
        folded = rel.replace("\\", "/").casefold()
        if folded in seen:
            _fail(MvAdapterErrorCode.MANIFEST_INVALID, "duplicate integrity path after case-folding",
                  relative_path=rel)
        seen.add(folded)
        if rel.lower().endswith(_BLOCKED_ARTIFACT_SUFFIXES):
            _fail(MvAdapterErrorCode.MANIFEST_INVALID, "partial/temporary artifact listed", relative_path=rel)
        if not _SHA256_HEX_RE.fullmatch(sha):
            _fail(MvAdapterErrorCode.MANIFEST_INVALID, "file sha256 must be lowercase 64-hex", relative_path=rel)
        path = (root / Path(rel)).resolve(strict=False)
        if not path.is_relative_to(root):
            _fail(MvAdapterErrorCode.PATH_ESCAPE, "integrity file escapes snapshot root", relative_path=rel)
        if not path.is_file():
            _fail(MvAdapterErrorCode.PRODUCT_MISSING, "integrity file is missing", relative_path=rel, path=str(path))
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            _fail(MvAdapterErrorCode.MANIFEST_INVALID, "size_bytes must be a non-negative integer",
                  relative_path=rel)
        actual_size = path.stat().st_size
        if actual_size != size:
            _fail(MvAdapterErrorCode.INTEGRITY_MISMATCH, "file size mismatch", relative_path=rel,
                  expected=size, actual=actual_size)
        normalized_rel = rel.replace("\\", "/")
        hashes[normalized_rel] = sha
        listed.append((normalized_rel, path, sha))
    for role in required_roles:
        rel = MV_PRODUCT_PATHS[role]
        if rel not in hashes:
            _fail(MvAdapterErrorCode.PRODUCT_MISSING, f"manifest does not list required product {role!r}",
                  role=role, relative_path=rel)

    # Content hashing runs only after every metadata, path and size check passed.
    required_paths = {MV_PRODUCT_PATHS[role] for role in required_roles}
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
            _fail(MvAdapterErrorCode.INTEGRITY_MISMATCH, "file sha256 mismatch", relative_path=rel,
                  expected=sha, actual=check.actual_sha256)
        if hash_report is not None:
            hash_report[rel] = check.method
        if check.warning and warnings is not None:
            warnings.append(check.warning)
    return hashes


def open_mv_snapshot(
    storage_root: str | Path | None = None,
    *,
    required_roles: Sequence[str] = MV_DEFAULT_REQUIRED_ROLES,
    verify_hashes: bool | None = None,
    now: datetime | None = None,
    use_stamp_cache: bool = True,
) -> MvSnapshot:
    """Resolve + verify the pinned MANC snapshot. Raises ``MvAdapterError`` on any gap.

    The manifest self-hash and the ``manifest.sha256`` sidecar are checked on
    every open, before any file content is hashed. File contents are then
    hashed as described in ``validate_mv_manifest``. ``use_stamp_cache=False``
    neither reads nor writes stamps under ``manifest/hash-stamps/`` (the open
    is then strictly read-only on the snapshot, at the cost of rehashing the
    required products).
    """
    for role in required_roles:
        if role not in MV_PRODUCT_PATHS:
            _fail(MvAdapterErrorCode.PRODUCT_MISSING, f"unknown product role {role!r}", role=role)
    root, snapshot_root = resolve_mv_snapshot_root(storage_root)
    if not snapshot_root.is_dir():
        _fail(MvAdapterErrorCode.SNAPSHOT_MISSING, "mv snapshot directory not found", snapshot_root=str(snapshot_root))
    manifest_path = snapshot_root / MV_MANIFEST_RELATIVE_PATH
    if not manifest_path.is_file():
        _fail(MvAdapterErrorCode.MANIFEST_MISSING, "mv manifest.json not found", manifest_path=str(manifest_path))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MvAdapterError(MvAdapterErrorCode.MANIFEST_INVALID, f"manifest unreadable: {exc}") from exc
    if not isinstance(manifest, dict):
        _fail(MvAdapterErrorCode.MANIFEST_INVALID, "manifest must be a JSON object")
    sidecar = snapshot_root / MV_MANIFEST_SIDECAR_RELATIVE_PATH
    warnings: list[str] = []
    # Fail closed: the sidecar is the only integrity anchor outside manifest.json.
    if not sidecar.is_file():
        _fail(MvAdapterErrorCode.MANIFEST_MISSING, "manifest.sha256 sidecar is missing", sidecar=str(sidecar))
    integrity = manifest.get("integrity") if isinstance(manifest.get("integrity"), dict) else {}
    if sidecar.read_text(encoding="utf-8").strip() != str(integrity.get("manifest_sha256") or ""):
        _fail(MvAdapterErrorCode.INTEGRITY_MISMATCH, "manifest.sha256 sidecar does not match manifest")
    hash_report: dict[str, str] = {}
    hashes = validate_mv_manifest(
        manifest,
        snapshot_root,
        required_roles=required_roles,
        verify_hashes=verify_hashes,
        now=now,
        stamp_cache=HashStampCache.for_snapshot(snapshot_root) if use_stamp_cache else None,
        hash_report=hash_report,
        warnings=warnings,
    )
    product_paths = {role: (snapshot_root / MV_PRODUCT_PATHS[role]).resolve(strict=False) for role in required_roles}
    product_sha = {role: hashes[MV_PRODUCT_PATHS[role]] for role in required_roles}
    return MvSnapshot(
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
        _fail(MvAdapterErrorCode.PRODUCT_MISSING, f"mv {role} file not found", path=str(candidate))
    return candidate


def _feather_schema(path: Path):
    """Read only the Arrow IPC schema (no column decompression)."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    try:
        with pa.memory_map(str(path), "r") as source:
            return ipc.open_file(source).schema
    except (OSError, pa.ArrowInvalid) as exc:
        raise MvAdapterError(
            MvAdapterErrorCode.SCHEMA_MISMATCH, f"not a readable Arrow/feather v2 file: {exc}", {"path": str(path)}
        ) from exc


def _check_columns(available: Iterable[str], required: Sequence[str], role: str, path: Path) -> None:
    have = set(available)
    missing = [col for col in required if col not in have]
    if missing:
        _fail(MvAdapterErrorCode.SCHEMA_MISMATCH, f"mv {role} missing required columns: {', '.join(missing)}",
              role=role, path=str(path), missing=missing)


def _csv_header(path: Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        first = handle.readline()
    return [col.strip().strip('"') for col in first.rstrip("\r\n").split(",")] if first else []


def roi_names_in_meta(path: Path) -> list[str]:
    """ROI names present as fields of the ``roiInfo`` struct column (schema only)."""
    import pyarrow as pa

    source = _require_file(path, "meta")
    schema = _feather_schema(source)
    _check_columns(schema.names, ("roiInfo",), "meta", source)
    roi_type = schema.field("roiInfo").type
    if not pa.types.is_struct(roi_type):
        _fail(MvAdapterErrorCode.SCHEMA_MISMATCH, "meta roiInfo must be a struct column", path=str(source),
              type=str(roi_type))
    return [child.name for child in roi_type]


def load_mv_neurons(
    path: Path,
    *,
    statuses: Sequence[str] = ("Traced",),
    columns: Sequence[str] = tuple(col for col in META_REQUIRED_COLUMNS if col != "roiInfo"),
):
    """Per-body neuron properties for the given ``status`` values only.

    Uses an Arrow dataset scan with a ``status`` filter and column projection,
    so untraced fragments (about 75% of the 102k rows) are never materialised
    and ``roiInfo`` is not decoded here. ``bodyId`` is returned as a string.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    source = _require_file(path, "meta")
    schema = _feather_schema(source)
    wanted = list(dict.fromkeys(["bodyId", "status", *columns]))
    _check_columns(schema.names, wanted, "meta", source)
    if not statuses:
        _fail(MvAdapterErrorCode.SCHEMA_MISMATCH, "at least one status must be selected")
    dataset = ds.dataset(str(source), format="feather")
    table = dataset.to_table(columns=wanted, filter=pc.field("status").isin(list(statuses)))
    ids = table.column("bodyId")
    if ids.null_count:
        _fail(MvAdapterErrorCode.SCHEMA_MISMATCH, "meta has null bodyId", path=str(source))
    table = table.set_column(table.schema.get_field_index("bodyId"), "bodyId", ids.cast(pa.string()))
    frame = table.to_pandas()
    if frame["bodyId"].duplicated().any():
        _fail(MvAdapterErrorCode.SCHEMA_MISMATCH, "meta bodyId is not unique", path=str(source))
    frame = frame.assign(_sort=frame["bodyId"].str.zfill(20)).sort_values("_sort", kind="mergesort")
    return frame.drop(columns=["_sort"]).reset_index(drop=True)


@dataclass(frozen=True)
class MvNeuropilWeights:
    """Dense per-neuropil synapse weights for a set of bodies.

    ``weights[i][j]`` is the ``roiInfo[neuropil_rois[j]].synweight`` of
    ``body_ids[i]`` (pre + post synapses in that ROI; null -> 0). Only
    neuropil ROIs are included; nerves, the cervical connective and tracts are
    excluded (``excluded_rois``).
    """

    body_ids: tuple[str, ...]
    neuropil_rois: tuple[str, ...]
    weights: Any  # numpy int64 array, shape (len(body_ids), len(neuropil_rois))
    excluded_rois: tuple[str, ...]


def load_mv_neuropil_synweights(path: Path, body_ids: Sequence[str]) -> MvNeuropilWeights:
    """Decode ``roiInfo`` for ``body_ids`` only into a neuropil synweight matrix.

    Every ROI field of the struct must be in ``MV_ROI_VOCABULARY``
    (``REGION_VOCABULARY_UNKNOWN`` otherwise). Negative weights fail closed.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    source = _require_file(path, "meta")
    roi_names = roi_names_in_meta(source)
    mapped = [map_mv_roi(name) for name in roi_names]
    neuropils = [roi for roi in mapped if roi.kind == ROI_KIND_NEUROPIL]
    if not neuropils:
        _fail(MvAdapterErrorCode.SCHEMA_MISMATCH, "roiInfo has no neuropil ROIs", path=str(source))
    wanted = [str(b) for b in body_ids]
    if not wanted:
        return MvNeuropilWeights((), tuple(r.roi for r in neuropils), np.zeros((0, len(neuropils)), dtype="int64"),
                                 tuple(r.roi for r in mapped if r.kind != ROI_KIND_NEUROPIL))
    try:
        id_values = pa.array([int(b) for b in wanted], pa.int64())
    except ValueError as exc:
        raise MvAdapterError(MvAdapterErrorCode.SCHEMA_MISMATCH, f"bodyId is not an integer: {exc}") from exc
    dataset = ds.dataset(str(source), format="feather")
    table = dataset.to_table(columns=["bodyId", "roiInfo"], filter=pc.field("bodyId").isin(id_values))
    got_ids = [str(v) for v in table.column("bodyId").to_pylist()]
    if len(set(got_ids)) != len(got_ids):
        _fail(MvAdapterErrorCode.SCHEMA_MISMATCH, "meta bodyId is not unique", path=str(source))
    missing = sorted(set(wanted) - set(got_ids))
    if missing:
        _fail(MvAdapterErrorCode.SCHEMA_MISMATCH, "roiInfo rows missing for requested bodies",
              path=str(source), examples=missing[:5])
    roi_col = table.column("roiInfo").combine_chunks() if table.num_rows else None
    columns = []
    for roi in neuropils:
        child = pc.struct_field(roi_col, roi.roi)
        if pa.types.is_struct(child.type):
            names = [f.name for f in child.type]
            if "synweight" not in names:
                _fail(MvAdapterErrorCode.SCHEMA_MISMATCH, f"roiInfo[{roi.roi}] has no synweight", path=str(source))
            child = pc.struct_field(child, "synweight")
        values = pc.fill_null(pc.cast(child, pa.float64()), 0.0).to_numpy(zero_copy_only=False)
        columns.append(values)
    matrix = np.stack(columns, axis=1) if columns else np.zeros((len(got_ids), 0))
    if not np.all(np.isfinite(matrix)) or (matrix < 0).any():
        _fail(MvAdapterErrorCode.SCHEMA_MISMATCH, "roiInfo synweight must be finite and non-negative",
              path=str(source))
    order = {body: index for index, body in enumerate(got_ids)}
    rows = [order[body] for body in wanted]
    return MvNeuropilWeights(
        body_ids=tuple(wanted),
        neuropil_rois=tuple(r.roi for r in neuropils),
        weights=matrix[rows].astype("int64"),
        excluded_rois=tuple(r.roi for r in mapped if r.kind != ROI_KIND_NEUROPIL),
    )


def load_mv_out_partner_counts(path: Path):
    """Aggregate the traced-to-traced edge list per presynaptic body.

    Returns ``(frame, edge_rows)``; ``frame`` has ``bodyId`` (str),
    ``n_post_partners`` (distinct traced ``bodyId_post``) and
    ``traced_out_weight`` (sum of ``weight``), sorted by bodyId. Only the
    ``bodyId_pre``/``bodyId_post``/``weight`` columns are parsed.
    """
    import pyarrow as pa
    import pyarrow.csv as pacsv

    source = _require_file(path, "edgelist")
    header = _csv_header(source)
    _check_columns(header, EDGELIST_REQUIRED_COLUMNS, "edgelist", source)
    try:
        table = pacsv.read_csv(
            str(source),
            convert_options=pacsv.ConvertOptions(
                include_columns=list(EDGELIST_REQUIRED_COLUMNS),
                column_types={"bodyId_pre": pa.int64(), "bodyId_post": pa.int64(), "weight": pa.int64()},
            ),
        )
    except (pa.ArrowInvalid, OSError) as exc:
        raise MvAdapterError(MvAdapterErrorCode.SCHEMA_MISMATCH, f"edgelist unreadable: {exc}",
                             {"path": str(source)}) from exc
    if table.num_rows == 0:
        _fail(MvAdapterErrorCode.SCHEMA_MISMATCH, "edgelist is empty", path=str(source))
    for col in EDGELIST_REQUIRED_COLUMNS:
        if table.column(col).null_count:
            _fail(MvAdapterErrorCode.SCHEMA_MISMATCH, f"edgelist has null {col} values", path=str(source))
    pairs = table.group_by(["bodyId_pre", "bodyId_post"]).aggregate([("weight", "sum")])
    if pairs.num_rows != table.num_rows:
        _fail(MvAdapterErrorCode.SCHEMA_MISMATCH, "edgelist repeats (bodyId_pre, bodyId_post) pairs",
              path=str(source), rows=int(table.num_rows), distinct_pairs=int(pairs.num_rows))
    grouped = table.group_by("bodyId_pre").aggregate([("weight", "sum"), ("weight", "count")])
    frame = grouped.to_pandas().rename(
        columns={"bodyId_pre": "bodyId", "weight_sum": "traced_out_weight", "weight_count": "n_post_partners"}
    )
    frame["bodyId"] = frame["bodyId"].astype("int64").astype(str)
    frame["traced_out_weight"] = frame["traced_out_weight"].astype("int64")
    frame["n_post_partners"] = frame["n_post_partners"].astype("int64")
    frame = frame.assign(_sort=frame["bodyId"].str.zfill(20)).sort_values("_sort", kind="mergesort")
    return frame.drop(columns=["_sort"]).reset_index(drop=True), int(table.num_rows)


def check_nt_argmax(frame) -> int:
    """Fail closed unless ``predictedNt`` is the argmax of the four probability
    columns and ``predictedNtProb`` equals that maximum (tolerance 1e-9).

    Rows with a null ``predictedNt`` are skipped. Returns the number checked.
    """
    import numpy as np

    rows = frame[frame["predictedNt"].notna()]
    if rows.empty:
        return 0
    names = list(MV_NT_PROBABILITY_COLUMNS)
    probs = rows[[MV_NT_PROBABILITY_COLUMNS[name] for name in names]].to_numpy(dtype="float64")
    if not np.all(np.isfinite(probs)):
        _fail(MvAdapterErrorCode.SCHEMA_MISMATCH, "NT probabilities missing for a neuron with predictedNt")
    predicted = rows["predictedNt"].astype(str).str.strip().str.lower().to_numpy()
    unknown = sorted(set(predicted) - set(names))
    if unknown:
        _fail(MvAdapterErrorCode.NT_VOCABULARY_UNKNOWN, "predictedNt outside the MANC NT vocabulary",
              values=unknown[:5], known=names)
    argmax = np.array(names)[probs.argmax(axis=1)]
    top = probs.max(axis=1)
    reported = rows["predictedNtProb"].to_numpy(dtype="float64")
    bad = (argmax != predicted) | ~np.isfinite(reported) | (np.abs(top - reported) > 1e-9)
    if bad.any():
        examples = rows.loc[bad, "bodyId"].astype(str).tolist()[:5]
        _fail(MvAdapterErrorCode.SCHEMA_MISMATCH, "predictedNt/predictedNtProb disagree with the NT probabilities",
              examples=examples, mismatches=int(bad.sum()))
    return int(len(rows))


__all__ = [
    "EDGELIST_REQUIRED_COLUMNS",
    "META_REQUIRED_COLUMNS",
    "MV_ALLOWED_LICENSES",
    "MV_CITATION",
    "MV_DATASET_SYMBOL",
    "MV_DEFAULT_REQUIRED_ROLES",
    "MV_MANIFEST_RELATIVE_PATH",
    "MV_MANIFEST_SIDECAR_RELATIVE_PATH",
    "MV_NT_PROBABILITY_COLUMNS",
    "MV_NT_SHORT_CODES",
    "MV_NT_UNKNOWN",
    "MV_PRODUCT_PATHS",
    "MV_ROI_VOCABULARY",
    "MV_SNAPSHOT_RELATIVE_ROOT",
    "MV_VERSION_ID",
    "MvAdapterError",
    "MvAdapterErrorCode",
    "MvNeuropilWeights",
    "MvRoi",
    "MvSnapshot",
    "ROI_KIND_CONNECTIVE",
    "ROI_KIND_NERVE",
    "ROI_KIND_NEUROPIL",
    "ROI_KIND_TRACT",
    "ROLE_EDGELIST",
    "ROLE_META",
    "build_mv_manifest",
    "check_nt_argmax",
    "load_mv_neurons",
    "load_mv_neuropil_synweights",
    "load_mv_out_partner_counts",
    "manifest_digest",
    "map_mv_roi",
    "nt_short_code",
    "open_mv_snapshot",
    "resolve_mv_snapshot_root",
    "roi_names_in_meta",
    "sha256_file",
    "validate_mv_manifest",
    "write_mv_manifest",
]
