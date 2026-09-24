"""Fail-closed local adapter for the pinned FlyEM optic-lobe v1.1 snapshot.

optic-lobe:v1.1 (Nern et al. 2025) is the right optic lobe of one male
*Drosophila* (the same EM specimen later released as male-cns v1.0; ``ol`` is
its own pinned version and is never merged with ``mc`` rows). This module is
the only place the harness touches the on-disk ``ol`` snapshot:

* ``open_ol_snapshot`` validates the storage root, the manifest
  (``fbh-manifest/v1`` self-hash + sidecar, verification status, refresh
  window, license), the size of every listed file and the sha256 of every
  required product before returning resolved paths. Content hashes are cached
  as write-once stamps keyed by (size, mtime_ns) (``flybrain_hash_stamps``).
* ``load_ol_neurons`` reads the neuPrint ``Neuron`` table with column
  projection and an Arrow scan filter, so only typed neurons with an allowed
  tracing status are materialised (the file holds ~10.3M bodies, ~54k typed).
  Body ids are kept as strings.
* ``load_ol_release_meta`` reads the neuPrint ``Meta`` row and checks the
  release pin and the primary-ROI list against the explicit vocabulary here.
* ``parse_roi_info`` / ``map_ol_roi`` form the explicit ROI vocabulary map.
  Unknown ROI names raise instead of being silently bucketed.

No network access: everything is read from
``$LOCI_FLYBRAIN_STORAGE_ROOT/snapshots/ol/optic_lobe_v1.1``. See
``docs/FLYBRAIN_OL_ADAPTER_CONTRACT.md``.
"""

from __future__ import annotations

import csv
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

OL_DATASET_SYMBOL = "ol"
OL_VERSION_ID = "optic_lobe_v1.1"
OL_SNAPSHOT_DIR_NAME = "ol"
OL_SNAPSHOT_RELATIVE_ROOT = "snapshots/ol/optic_lobe_v1.1"
OL_MANIFEST_SCHEMA_VERSION = "fbh-manifest/v1"
OL_MANIFEST_RELATIVE_PATH = "manifest/manifest.json"
OL_MANIFEST_SIDECAR_RELATIVE_PATH = "manifest/manifest.sha256"
OL_ALLOWED_LICENSES = frozenset({"CC-BY-4.0"})
# neuPrint Meta row values that pin the release.
OL_NEUPRINT_DATASET = "optic-lobe"
OL_NEUPRINT_TAG = "v1.1"
OL_CITATION = (
    "Nern A, Loesche F, Takemura S, Burnett LE, Dreher M, et al. (2025). Connectome-driven neural "
    "inventory of a complete visual system. Nature 641:1225-1237. doi:10.1038/s41586-025-08746-0. "
    "Data: FlyEM optic-lobe:v1.1, gs://flyem-optic-lobe/v1.1 (CC BY 4.0)."
)

# Product roles -> snapshot-relative paths. Role names match the ``role``
# field the pull job wrote into manifest.integrity.files.
ROLE_NEURONS = "neuron_annotations"
ROLE_NEUPRINT_META = "neuprint_meta"
OL_PRODUCT_PATHS: Mapping[str, str] = {
    ROLE_NEURONS: "metadata/Neuprint_Neurons.feather",
    ROLE_NEUPRINT_META: "metadata/Neuprint_Meta.csv",
}

# Canonical name -> raw neuPrint CSV-import column name (type suffix included).
# The raw names are matched exactly: a renamed or retyped column fails closed.
NEURON_COLUMNS: Mapping[str, str] = {
    "body_id": "bodyId:long",
    "cell_type": "type:string",
    "instance": "instance:string",
    "status": "status:string",
    "pre": "pre:int",
    "post": "post:int",
    "downstream": "downstream:int",
    "upstream": "upstream:int",
    "predicted_nt": "predictedNt:string",
    "predicted_nt_confidence": "predictedNtConfidence:float",
    "total_nt_predictions": "totalNtPredictions:float",
    "hemilineage": "hemilineage:string",
    "soma_location": "somaLocation:point{srid:9157}",
    "assigned_ol_hex1": "assignedOlHex1:float",
    "roi_info": "roiInfo:string",
}
NEUPRINT_META_REQUIRED_COLUMNS: tuple[str, ...] = ("dataset:string", "tag:string", "primaryRois:string[]")
DEFAULT_ALLOWED_STATUSES: tuple[str, ...] = ("Traced",)

# neuPrint predictedNt vocabulary -> short code (shared with fw/banc).
OL_NT_SHORT_CODES: Mapping[str, str] = {
    "acetylcholine": "ach",
    "gaba": "gaba",
    "glutamate": "glut",
    "dopamine": "da",
    "serotonin": "ser",
    "octopamine": "oct",
    "histamine": "his",
    "tyramine": "tyr",
}
# Low-confidence neuPrint call; never a label (builders drop it).
OL_NT_UNCLEAR = "unclear"

# ---------------------------------------------------------------------------
# ROI vocabulary (explicit). The primary ROI list is the release's
# ``primaryRois`` (Neuprint_Meta.csv); load_ol_release_meta checks it matches.
# ---------------------------------------------------------------------------

DIVISION_OPTIC_LOBE = "optic_lobe"
DIVISION_CENTRAL_BRAIN = "central_brain"
DIVISION_VENTRAL_NERVE_CORD = "ventral_nerve_cord"

OL_OPTIC_NEUROPILS: tuple[str, ...] = ("AME", "LA", "LO", "LOP", "ME")
_CENTRAL_SIDED: tuple[str, ...] = (
    "AB", "AL", "AMMC", "AOTU", "ATL", "AVLP", "BU", "CA", "CAN", "CRE", "EPA", "FLA", "GA", "GOR", "ICL",
    "IPS", "LAL", "LH", "PED", "PLP", "PVLP", "ROB", "RUB", "SCL", "SIP", "SLP", "SMP", "SPS", "VES", "WED",
    "a'L", "aL", "b'L", "bL", "gL",
)
_CENTRAL_UNPAIRED: tuple[str, ...] = ("EB", "FB", "GNG", "IB", "NO", "PB", "PRW", "SAD")
_VNC_ROIS: tuple[str, ...] = ("vnc-shell",)
_MB_LOBES = frozenset({"a'L", "aL", "b'L", "bL", "gL"})

OL_PRIMARY_ROIS: frozenset[str] = frozenset(
    [f"{name}({side})" for name in OL_OPTIC_NEUROPILS + _CENTRAL_SIDED for side in ("L", "R")]
    + list(_CENTRAL_UNPAIRED)
    + list(_VNC_ROIS)
)
# Super-level ROIs that contain primary ROIs (double counting): recognised, skipped.
OL_AGGREGATE_ROIS: frozenset[str] = frozenset({"OL(L)", "OL(R)"})
_LAYER_RE = re.compile(r"^(ME|LO|LOP)_R_layer_(\d{1,2})$")
_COLUMN_RE = re.compile(r"^(ME|LO|LOP)_R_col_(\d{1,2})_(\d{1,2})$")
_SIDED_RE = re.compile(r"^(.+)\((L|R)\)$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")

_SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")
_BLOCKED_ARTIFACT_SUFFIXES = (".partial", ".tmp", ".inprogress")
_REFRESH_DECISIONS = frozenset({"no_change", "patch_refresh", "major_bump", "rollback"})


class OlAdapterErrorCode(str, Enum):
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


class OlAdapterError(ValueError):
    """Typed fail-closed error; message is prefixed with ``[CODE]``."""

    def __init__(self, code: OlAdapterErrorCode, message: str, details: Mapping[str, Any] | None = None):
        self.code = code
        self.message = message
        self.details = dict(details or {})
        super().__init__(f"[{code.value}] {message}")


def _fail(code: OlAdapterErrorCode, message: str, **details: Any) -> None:
    raise OlAdapterError(code, message, details)


# ---------------------------------------------------------------------------
# ROI vocabulary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OlRegion:
    roi: str
    division: str
    neuropil: str
    side: str | None
    slug: str
    region_id: str


def _slug(text: str) -> str:
    return _SLUG_RE.sub("_", text.replace("'", "prime").lower()).strip("_")


def map_ol_roi(roi: str) -> OlRegion:
    """Map a neuPrint primary ROI onto the explicit optic-lobe / central-brain / VNC vocabulary.

    ``slug`` is a collision-free lowercase token (``ME(R)`` -> ``me_r``,
    ``a'L(R)`` -> ``mb_aprimel_r`` so it never collides with ``AL(R)`` ->
    ``al_r``). ``region_id`` prefixes the division: ``ol_me_r``, ``cb_plp_r``,
    ``vnc_vnc_shell``. Names outside ``OL_PRIMARY_ROIS`` raise
    ``REGION_VOCABULARY_UNKNOWN``.
    """
    text = str(roi or "").strip()
    if text not in OL_PRIMARY_ROIS:
        _fail(OlAdapterErrorCode.REGION_VOCABULARY_UNKNOWN, f"ROI {text!r} is not an optic-lobe v1.1 primary ROI",
              roi=text)
    side_match = _SIDED_RE.match(text)
    neuropil, side = (side_match.group(1), side_match.group(2)) if side_match else (text, None)
    if neuropil in OL_OPTIC_NEUROPILS:
        division, prefix = DIVISION_OPTIC_LOBE, "ol"
    elif text in _VNC_ROIS:
        division, prefix = DIVISION_VENTRAL_NERVE_CORD, "vnc"
    else:
        division, prefix = DIVISION_CENTRAL_BRAIN, "cb"
    slug = _slug(neuropil)
    if neuropil in _MB_LOBES:
        slug = f"mb_{slug}"
    if side is not None:
        slug = f"{slug}_{side.lower()}"
    return OlRegion(
        roi=text,
        division=division,
        neuropil=neuropil,
        side={"L": "left", "R": "right", None: None}[side],
        slug=slug,
        region_id=f"{prefix}_{slug}",
    )


def _check_vocabulary() -> None:
    slugs = [map_ol_roi(roi).slug for roi in sorted(OL_PRIMARY_ROIS)]
    if len(set(slugs)) != len(slugs):
        raise AssertionError("optic-lobe ROI slugs are not unique")


_check_vocabulary()


@dataclass(frozen=True)
class RoiCounts:
    pre: int
    post: int
    synweight: int


@dataclass(frozen=True)
class OlRoiSummary:
    """Per-neuron ``roiInfo`` split into primary ROIs, OL layers and a column count."""

    primary: Mapping[str, RoiCounts]
    layers: Mapping[str, int]
    n_columns: int

    def dominant(self, measure: str) -> str | None:
        """Primary ROI with the largest ``measure`` (> 0); ties broken by ROI name."""
        best = [(getattr(c, measure), roi) for roi, c in self.primary.items() if getattr(c, measure) > 0]
        if not best:
            return None
        top = max(value for value, _ in best)
        return sorted(roi for value, roi in best if value == top)[0]

    def arbor(self, min_share: float) -> tuple[str, ...]:
        """Primary ROIs holding at least ``min_share`` of the primary synweight, sorted."""
        total = sum(c.synweight for c in self.primary.values())
        if total <= 0:
            return ()
        return tuple(sorted(roi for roi, c in self.primary.items() if c.synweight / total >= min_share))

    def dominant_layer(self) -> str | None:
        best = [(weight, name) for name, weight in self.layers.items() if weight > 0]
        if not best:
            return None
        top = max(weight for weight, _ in best)
        return sorted(name for weight, name in best if weight == top)[0]


def _count(value: Any, *, key: str, roi: str) -> int:
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or value < 0 \
            or int(value) != value:
        _fail(OlAdapterErrorCode.SCHEMA_MISMATCH, "roiInfo count is not a non-negative integer",
              roi=roi, key=key, value=repr(value))
    return int(value)


def parse_roi_info(raw: str | None) -> OlRoiSummary:
    """Parse one neuPrint ``roiInfo`` JSON string (explicit vocabulary, fail closed)."""
    if raw is None or (isinstance(raw, float) and raw != raw) or not str(raw).strip():
        return OlRoiSummary(primary={}, layers={}, n_columns=0)
    try:
        info = json.loads(raw)
    except ValueError as exc:
        raise OlAdapterError(OlAdapterErrorCode.SCHEMA_MISMATCH, f"roiInfo is not JSON: {exc}") from exc
    if not isinstance(info, dict):
        _fail(OlAdapterErrorCode.SCHEMA_MISMATCH, "roiInfo must be a JSON object")
    primary: dict[str, RoiCounts] = {}
    layers: dict[str, int] = {}
    n_columns = 0
    for roi, counts in info.items():
        if not isinstance(counts, dict):
            _fail(OlAdapterErrorCode.SCHEMA_MISMATCH, "roiInfo entry must be an object", roi=roi)
        if roi in OL_PRIMARY_ROIS:
            primary[roi] = RoiCounts(
                pre=_count(counts.get("pre"), key="pre", roi=roi),
                post=_count(counts.get("post"), key="post", roi=roi),
                synweight=_count(counts.get("synweight"), key="synweight", roi=roi),
            )
        elif _LAYER_RE.match(roi):
            layers[roi] = _count(counts.get("synweight"), key="synweight", roi=roi)
        elif _COLUMN_RE.match(roi):
            n_columns += 1
        elif roi in OL_AGGREGATE_ROIS:
            continue
        else:
            _fail(OlAdapterErrorCode.REGION_VOCABULARY_UNKNOWN, f"roiInfo key {roi!r} is not in the optic-lobe "
                  "v1.1 vocabulary", roi=roi)
    return OlRoiSummary(primary=primary, layers=dict(sorted(layers.items())), n_columns=n_columns)


def layer_slug(layer_roi: str) -> str:
    """``ME_R_layer_03`` -> ``me_r_layer_03`` (zero-padded to two digits)."""
    match = _LAYER_RE.match(str(layer_roi))
    if not match:
        _fail(OlAdapterErrorCode.REGION_VOCABULARY_UNKNOWN, f"{layer_roi!r} is not an optic-lobe layer ROI")
    return f"{match.group(1).lower()}_r_layer_{int(match.group(2)):02d}"


def nt_short_code(name: str) -> str:
    text = str(name or "").strip().lower()
    if text not in OL_NT_SHORT_CODES:
        _fail(OlAdapterErrorCode.NT_VOCABULARY_UNKNOWN, f"neurotransmitter {name!r} is not in the OL NT vocabulary",
              neurotransmitter=name, known=sorted(OL_NT_SHORT_CODES))
    return OL_NT_SHORT_CODES[text]


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
        raise OlAdapterError(OlAdapterErrorCode.MANIFEST_INVALID, "manifest.integrity must be an object")
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


def resolve_ol_snapshot_root(storage_root: str | Path | None = None) -> tuple[Path, Path]:
    """Return ``(storage_root, snapshot_root)`` after root + path-escape validation."""
    try:
        layout = build_flybrain_harness_layout(root_override=storage_root, create=False)
    except ValueError as exc:
        raise OlAdapterError(OlAdapterErrorCode.ROOT_NOT_CONFIGURED, str(exc)) from exc
    spec = get_dataset(OL_DATASET_SYMBOL)
    if spec.pinned_version != OL_VERSION_ID or spec.snapshot_dir_name != OL_SNAPSHOT_DIR_NAME:
        _fail(OlAdapterErrorCode.DATASET_PIN_MISMATCH, "registry pin for ol does not match the adapter pin",
              registry_version=spec.pinned_version, adapter_version=OL_VERSION_ID)
    snapshot_root = snapshot_version_root(OL_DATASET_SYMBOL, layout.root).resolve(strict=False)
    if not snapshot_root.is_relative_to(layout.root.resolve(strict=False)):
        _fail(OlAdapterErrorCode.PATH_ESCAPE, "ol snapshot path escapes the configured storage root",
              snapshot_root=str(snapshot_root))
    return layout.root, snapshot_root


def build_ol_manifest(
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
    """Build a verified ``fbh-manifest/v1`` dict for the ol snapshot (self-hash applied)."""
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
        "system": "flyem-optic-lobe-gcs-public",
        "access_method": "https_public_bucket_selective_pull",
        "uri": source_uri,
        "retrieved_at": retrieved_at,
        "license": {"spdx_id": license_spdx, "url": "https://creativecommons.org/licenses/by/4.0/"},
        "citation": OL_CITATION,
        "doi": "10.1038/s41586-025-08746-0",
    }
    source.update(dict(extra_source or {}))
    manifest: dict[str, Any] = {
        "schema_version": OL_MANIFEST_SCHEMA_VERSION,
        "manifest_id": f"fbh-ol-{OL_VERSION_ID}-{generated}",
        "generated_at": generated,
        "storage_root_env": "LOCI_FLYBRAIN_STORAGE_ROOT",
        "artifact": {
            "kind": "dataset_snapshot",
            "relative_root": OL_SNAPSHOT_RELATIVE_ROOT,
            "path_template": "$LOCI_FLYBRAIN_STORAGE_ROOT\\snapshots\\ol\\optic_lobe_v1.1\\",
        },
        "dataset": {
            "symbol": "ol",
            "label": "FlyEM male optic lobe connectome optic-lobe:v1.1 (selective tabular products)",
            "version_id": OL_VERSION_ID,
            "source": source,
        },
        "scope": {
            "sex": "male",
            "stage": "adult",
            "anatomy": "right_optic_lobe",
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
            "derived_from": [{"type": "remote_dataset_release", "id": "gs://flyem-optic-lobe/v1.1#optic-lobe:v1.1"}],
            "pipeline": {
                "job_name": "fbh_ol_selective_pull",
                "job_version": "2026.09.24.1",
                "run_id": run_id,
                "run_mode": "execute",
            },
        },
        "notes": list(notes),
    }
    manifest["integrity"]["manifest_sha256"] = manifest_digest(manifest)
    return manifest


def write_ol_manifest(snapshot_root: Path, manifest: Mapping[str, Any]) -> Path:
    """Write manifest.json + manifest.sha256; refuses to overwrite (MANIFEST_EXISTS)."""
    target = Path(snapshot_root) / OL_MANIFEST_RELATIVE_PATH
    sidecar = Path(snapshot_root) / OL_MANIFEST_SIDECAR_RELATIVE_PATH
    for path in (target, sidecar):
        if path.exists():
            _fail(OlAdapterErrorCode.MANIFEST_EXISTS,
                  "refusing to overwrite an existing ol manifest; supersede with a new version instead",
                  path=str(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sidecar.write_text(str(manifest["integrity"]["manifest_sha256"]) + "\n", encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Snapshot open / verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OlSnapshot:
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
            raise OlAdapterError(OlAdapterErrorCode.PRODUCT_MISSING, f"product role {role!r} not in snapshot")
        return self.product_paths[role]

    def provenance(self) -> dict[str, Any]:
        return {
            "dataset_symbol": OL_DATASET_SYMBOL,
            "version_id": OL_VERSION_ID,
            "manifest_id": self.manifest_id,
            "manifest_sha256": self.manifest_sha256,
            "source_path": OL_SNAPSHOT_RELATIVE_ROOT,
            "access_method": "local_snapshot_read_only",
            "license": self.license_spdx,
            "product_sha256": dict(sorted(self.product_sha256.items())),
        }


def validate_ol_manifest(
    manifest: Mapping[str, Any],
    snapshot_root: Path,
    *,
    required_roles: Sequence[str] = tuple(OL_PRODUCT_PATHS),
    verify_hashes: bool | None = None,
    now: datetime | None = None,
    stamp_cache: HashStampCache | None = None,
    hash_report: dict[str, str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, str]:
    """Fail-closed manifest check. Returns ``{relative_path: sha256}`` for listed files.

    Same contract as ``flybrain_banc_adapter.validate_banc_manifest``: every
    listed file is checked for presence, path safety and exact ``size_bytes``;
    content hashing follows ``verify_hashes`` (``True`` all files, ``None``
    required products unless a hash stamp matches, ``False`` size only). In
    addition, a listed product whose manifest ``role`` disagrees with the
    adapter's role for that path fails closed.
    """
    if manifest.get("schema_version") != OL_MANIFEST_SCHEMA_VERSION:
        _fail(OlAdapterErrorCode.MANIFEST_INVALID, "manifest schema_version mismatch",
              schema_version=manifest.get("schema_version"))
    for key in ("manifest_id", "generated_at", "artifact", "dataset", "scope", "integrity", "refresh", "lineage"):
        if key not in manifest:
            _fail(OlAdapterErrorCode.MANIFEST_INVALID, f"manifest missing required field {key!r}")
    dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), dict) else {}
    try:
        symbol = normalize_symbol(str(dataset.get("symbol") or ""))
    except Exception:
        symbol = str(dataset.get("symbol"))
    if symbol != OL_DATASET_SYMBOL or dataset.get("version_id") != OL_VERSION_ID:
        _fail(OlAdapterErrorCode.DATASET_PIN_MISMATCH, "manifest dataset pin does not match ol optic_lobe_v1.1",
              symbol=dataset.get("symbol"), version_id=dataset.get("version_id"))
    source = dataset.get("source") if isinstance(dataset.get("source"), dict) else {}
    license_info = source.get("license") if isinstance(source.get("license"), dict) else {}
    spdx = str(license_info.get("spdx_id") or "").strip()
    if spdx not in OL_ALLOWED_LICENSES:
        _fail(OlAdapterErrorCode.LICENSE_NOT_PERMITTED, "manifest license is missing or not permitted",
              spdx_id=spdx, allowed=sorted(OL_ALLOWED_LICENSES))
    artifact = manifest.get("artifact") if isinstance(manifest.get("artifact"), dict) else {}
    relative_root = str(artifact.get("relative_root") or "").replace("\\", "/").strip("/")
    if not is_safe_relative_path(relative_root):
        _fail(OlAdapterErrorCode.PATH_ESCAPE, "artifact.relative_root is unsafe", relative_root=relative_root)
    if relative_root != OL_SNAPSHOT_RELATIVE_ROOT:
        _fail(OlAdapterErrorCode.DATASET_PIN_MISMATCH, "artifact.relative_root does not match the ol pin",
              relative_root=relative_root, expected=OL_SNAPSHOT_RELATIVE_ROOT)

    integrity = manifest.get("integrity") if isinstance(manifest.get("integrity"), dict) else {}
    verification = integrity.get("verification") if isinstance(integrity.get("verification"), dict) else {}
    status = str(verification.get("status") or "").strip().lower()
    if status not in {"pending", "verified", "quarantined"}:
        _fail(OlAdapterErrorCode.MANIFEST_INVALID, "integrity.verification.status invalid", status=status)
    if status != "verified":
        _fail(OlAdapterErrorCode.INTEGRITY_MISMATCH, "manifest verification status must be verified", status=status)
    if not str(verification.get("verified_at") or "").strip():
        _fail(OlAdapterErrorCode.MANIFEST_INVALID, "verified_at is required when status is verified")
    expected = str(integrity.get("manifest_sha256") or "").strip()
    if not _SHA256_HEX_RE.fullmatch(expected):
        _fail(OlAdapterErrorCode.MANIFEST_INVALID, "integrity.manifest_sha256 must be lowercase 64-hex")
    actual = manifest_digest(manifest)
    if actual != expected:
        _fail(OlAdapterErrorCode.INTEGRITY_MISMATCH, "manifest self-hash mismatch", expected=expected, actual=actual)

    refresh = manifest.get("refresh") if isinstance(manifest.get("refresh"), dict) else {}
    decision = str(refresh.get("decision") or "").strip().lower()
    if decision not in _REFRESH_DECISIONS:
        _fail(OlAdapterErrorCode.MANIFEST_INVALID, "refresh.decision invalid", decision=decision)
    if decision == "rollback":
        _fail(OlAdapterErrorCode.PROMOTION_STATE_INVALID, "refresh.decision indicates rollback")
    due = refresh.get("next_check_due")
    if due:
        try:
            due_dt = datetime.fromisoformat(str(due).replace("Z", "+00:00"))
        except ValueError:
            _fail(OlAdapterErrorCode.MANIFEST_INVALID, "refresh.next_check_due not RFC3339", next_check_due=due)
        if due_dt < (now or datetime.now(timezone.utc)):
            _fail(OlAdapterErrorCode.PROMOTION_STATE_INVALID, "refresh.next_check_due has elapsed",
                  next_check_due=due)

    files = integrity.get("files")
    if not isinstance(files, list) or not files:
        _fail(OlAdapterErrorCode.MANIFEST_INVALID, "integrity.files must be a non-empty list")
    role_of_path = {path: role for role, path in OL_PRODUCT_PATHS.items()}
    root = Path(snapshot_root).resolve(strict=False)
    seen: set[str] = set()
    hashes: dict[str, str] = {}
    listed: list[tuple[str, Path, str]] = []
    for entry in files:
        if not isinstance(entry, dict):
            _fail(OlAdapterErrorCode.MANIFEST_INVALID, "integrity.files entries must be objects")
        rel = str(entry.get("relative_path") or "").strip()
        sha = str(entry.get("sha256") or "").strip()
        size = entry.get("size_bytes")
        if not is_safe_relative_path(rel):
            _fail(OlAdapterErrorCode.PATH_ESCAPE, "integrity file path is unsafe", relative_path=rel)
        folded = rel.replace("\\", "/").casefold()
        if folded in seen:
            _fail(OlAdapterErrorCode.MANIFEST_INVALID, "duplicate integrity path after case-folding",
                  relative_path=rel)
        seen.add(folded)
        if rel.lower().endswith(_BLOCKED_ARTIFACT_SUFFIXES):
            _fail(OlAdapterErrorCode.MANIFEST_INVALID, "partial/temporary artifact listed", relative_path=rel)
        if not _SHA256_HEX_RE.fullmatch(sha):
            _fail(OlAdapterErrorCode.MANIFEST_INVALID, "file sha256 must be lowercase 64-hex", relative_path=rel)
        normalized_rel = rel.replace("\\", "/")
        listed_role = entry.get("role")
        if normalized_rel in role_of_path and listed_role is not None and listed_role != role_of_path[normalized_rel]:
            _fail(OlAdapterErrorCode.MANIFEST_INVALID, "manifest role does not match the adapter role for this path",
                  relative_path=rel, manifest_role=listed_role, adapter_role=role_of_path[normalized_rel])
        path = (root / Path(rel)).resolve(strict=False)
        if not path.is_relative_to(root):
            _fail(OlAdapterErrorCode.PATH_ESCAPE, "integrity file escapes snapshot root", relative_path=rel)
        if not path.is_file():
            _fail(OlAdapterErrorCode.PRODUCT_MISSING, "integrity file is missing", relative_path=rel, path=str(path))
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            _fail(OlAdapterErrorCode.MANIFEST_INVALID, "size_bytes must be a non-negative integer",
                  relative_path=rel)
        if path.stat().st_size != size:
            _fail(OlAdapterErrorCode.INTEGRITY_MISMATCH, "file size mismatch", relative_path=rel,
                  expected=size, actual=path.stat().st_size)
        hashes[normalized_rel] = sha
        listed.append((normalized_rel, path, sha))
    for role in required_roles:
        rel = OL_PRODUCT_PATHS[role]
        if rel not in hashes:
            _fail(OlAdapterErrorCode.PRODUCT_MISSING, f"manifest does not list required product {role!r}",
                  role=role, relative_path=rel)

    # Content hashing runs only after every metadata, path and size check passed.
    required_paths = {OL_PRODUCT_PATHS[role] for role in required_roles}
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
            _fail(OlAdapterErrorCode.INTEGRITY_MISMATCH, "file sha256 mismatch", relative_path=rel,
                  expected=sha, actual=check.actual_sha256)
        if hash_report is not None:
            hash_report[rel] = check.method
        if check.warning and warnings is not None:
            warnings.append(check.warning)
    return hashes


def open_ol_snapshot(
    storage_root: str | Path | None = None,
    *,
    required_roles: Sequence[str] = tuple(OL_PRODUCT_PATHS),
    verify_hashes: bool | None = None,
    now: datetime | None = None,
) -> OlSnapshot:
    """Resolve + verify the pinned ol snapshot. Raises ``OlAdapterError`` on any gap.

    The manifest self-hash and the ``manifest.sha256`` sidecar are checked on
    every open, before any file content is hashed; see ``validate_ol_manifest``.
    """
    for role in required_roles:
        if role not in OL_PRODUCT_PATHS:
            _fail(OlAdapterErrorCode.PRODUCT_MISSING, f"unknown product role {role!r}", role=role)
    root, snapshot_root = resolve_ol_snapshot_root(storage_root)
    if not snapshot_root.is_dir():
        _fail(OlAdapterErrorCode.SNAPSHOT_MISSING, "ol snapshot directory not found", snapshot_root=str(snapshot_root))
    manifest_path = snapshot_root / OL_MANIFEST_RELATIVE_PATH
    if not manifest_path.is_file():
        _fail(OlAdapterErrorCode.MANIFEST_MISSING, "ol manifest.json not found", manifest_path=str(manifest_path))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise OlAdapterError(OlAdapterErrorCode.MANIFEST_INVALID, f"manifest unreadable: {exc}") from exc
    if not isinstance(manifest, dict):
        _fail(OlAdapterErrorCode.MANIFEST_INVALID, "manifest must be a JSON object")
    sidecar = snapshot_root / OL_MANIFEST_SIDECAR_RELATIVE_PATH
    warnings: list[str] = []
    if not sidecar.is_file():
        _fail(OlAdapterErrorCode.MANIFEST_MISSING, "manifest.sha256 sidecar is missing", sidecar=str(sidecar))
    integrity = manifest.get("integrity") if isinstance(manifest.get("integrity"), dict) else {}
    if sidecar.read_text(encoding="utf-8").strip() != str(integrity.get("manifest_sha256") or ""):
        _fail(OlAdapterErrorCode.INTEGRITY_MISMATCH, "manifest.sha256 sidecar does not match manifest")
    hash_report: dict[str, str] = {}
    hashes = validate_ol_manifest(
        manifest,
        snapshot_root,
        required_roles=required_roles,
        verify_hashes=verify_hashes,
        now=now,
        stamp_cache=HashStampCache.for_snapshot(snapshot_root),
        hash_report=hash_report,
        warnings=warnings,
    )
    product_paths = {role: (snapshot_root / OL_PRODUCT_PATHS[role]).resolve(strict=False) for role in required_roles}
    product_sha = {role: hashes[OL_PRODUCT_PATHS[role]] for role in required_roles}
    return OlSnapshot(
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
        _fail(OlAdapterErrorCode.PRODUCT_MISSING, f"ol {role} file not found", path=str(candidate))
    return candidate


def _feather_columns(path: Path) -> list[str]:
    """Read only the Arrow IPC schema (no column decompression)."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    try:
        with pa.memory_map(str(path), "r") as source:
            return list(ipc.open_file(source).schema.names)
    except (OSError, pa.ArrowInvalid) as exc:
        raise OlAdapterError(
            OlAdapterErrorCode.SCHEMA_MISMATCH, f"not a readable Arrow/feather v2 file: {exc}", {"path": str(path)}
        ) from exc


def _check_columns(available: Iterable[str], required: Sequence[str], role: str, path: Path) -> None:
    have = set(available)
    missing = [col for col in required if col not in have]
    if missing:
        _fail(OlAdapterErrorCode.SCHEMA_MISMATCH, f"ol {role} missing required columns: {', '.join(missing)}",
              role=role, path=str(path), missing=missing)


@dataclass(frozen=True)
class OlReleaseMeta:
    dataset: str
    tag: str
    primary_rois: tuple[str, ...]


def load_ol_release_meta(path: Path) -> OlReleaseMeta:
    """Read the one-row neuPrint ``Meta`` CSV and check the release pin + ROI vocabulary."""
    source = _require_file(path, "neuprint_meta")
    csv.field_size_limit(1 << 30)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _check_columns(reader.fieldnames or [], NEUPRINT_META_REQUIRED_COLUMNS, "neuprint_meta", source)
        rows = list(reader)
    if len(rows) != 1:
        _fail(OlAdapterErrorCode.SCHEMA_MISMATCH, "neuprint_meta must hold exactly one row", rows=len(rows))
    row = rows[0]
    dataset, tag = str(row["dataset:string"]).strip(), str(row["tag:string"]).strip()
    if (dataset, tag) != (OL_NEUPRINT_DATASET, OL_NEUPRINT_TAG):
        _fail(OlAdapterErrorCode.DATASET_PIN_MISMATCH, "neuprint_meta dataset/tag does not match optic-lobe v1.1",
              dataset=dataset, tag=tag)
    primary = tuple(sorted(part for part in str(row["primaryRois:string[]"]).split(";") if part))
    if set(primary) != OL_PRIMARY_ROIS:
        _fail(OlAdapterErrorCode.REGION_VOCABULARY_UNKNOWN,
              "release primaryRois differ from the adapter's explicit ROI vocabulary",
              unexpected=sorted(set(primary) - OL_PRIMARY_ROIS), missing=sorted(OL_PRIMARY_ROIS - set(primary)))
    return OlReleaseMeta(dataset=dataset, tag=tag, primary_rois=primary)


def load_ol_neurons(path: Path, *, statuses: Sequence[str] = DEFAULT_ALLOWED_STATUSES):
    """Typed neurons from ``Neuprint_Neurons.feather`` with canonical column names.

    Only the ``NEURON_COLUMNS`` projection is read, and the Arrow scan filter
    keeps rows whose ``type`` is non-empty and whose ``status`` is in
    ``statuses``, so the ~10M untyped fragments are never materialised.
    Returns ``(frame, source_rows, typed_rows)``: ``source_rows`` is the
    table's full row count, ``typed_rows`` the typed rows before the status
    filter. ``body_id`` is a string and must be unique.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    source = _require_file(path, "neuron_annotations")
    names = _feather_columns(source)
    _check_columns(names, tuple(NEURON_COLUMNS.values()), "neuron_annotations", source)
    allowed = sorted({str(status) for status in statuses})
    if not allowed:
        raise ValueError("statuses must be non-empty")
    try:
        dataset = ds.dataset(str(source), format="ipc")
        source_rows = int(dataset.count_rows())
        type_field = ds.field(NEURON_COLUMNS["cell_type"])
        typed = pc.is_valid(type_field) & (type_field != "")
        typed_rows = int(dataset.count_rows(filter=typed))
        table = dataset.to_table(
            columns=list(NEURON_COLUMNS.values()),
            filter=typed & ds.field(NEURON_COLUMNS["status"]).isin(allowed),
        )
    except (OSError, pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError) as exc:
        raise OlAdapterError(OlAdapterErrorCode.SCHEMA_MISMATCH, f"neuron table scan failed: {exc}",
                             {"path": str(source)}) from exc
    table = table.rename_columns(list(NEURON_COLUMNS))
    body_index = table.schema.get_field_index("body_id")
    table = table.set_column(body_index, "body_id", table.column("body_id").cast(pa.string()))
    frame = table.to_pandas()
    if frame["body_id"].isna().any():
        _fail(OlAdapterErrorCode.SCHEMA_MISMATCH, "neuron table has null bodyId", path=str(source))
    if frame["body_id"].duplicated().any():
        _fail(OlAdapterErrorCode.SCHEMA_MISMATCH, "neuron table bodyId is not unique", path=str(source))
    return frame, source_rows, typed_rows


__all__ = [
    "DEFAULT_ALLOWED_STATUSES",
    "DIVISION_CENTRAL_BRAIN",
    "DIVISION_OPTIC_LOBE",
    "DIVISION_VENTRAL_NERVE_CORD",
    "NEURON_COLUMNS",
    "OL_AGGREGATE_ROIS",
    "OL_ALLOWED_LICENSES",
    "OL_CITATION",
    "OL_DATASET_SYMBOL",
    "OL_MANIFEST_RELATIVE_PATH",
    "OL_NT_SHORT_CODES",
    "OL_NT_UNCLEAR",
    "OL_PRIMARY_ROIS",
    "OL_PRODUCT_PATHS",
    "OL_SNAPSHOT_RELATIVE_ROOT",
    "OL_VERSION_ID",
    "OlAdapterError",
    "OlAdapterErrorCode",
    "OlRegion",
    "OlReleaseMeta",
    "OlRoiSummary",
    "OlSnapshot",
    "ROLE_NEUPRINT_META",
    "ROLE_NEURONS",
    "RoiCounts",
    "build_ol_manifest",
    "layer_slug",
    "load_ol_neurons",
    "load_ol_release_meta",
    "manifest_digest",
    "map_ol_roi",
    "nt_short_code",
    "open_ol_snapshot",
    "parse_roi_info",
    "resolve_ol_snapshot_root",
    "sha256_file",
    "validate_ol_manifest",
    "write_ol_manifest",
]
