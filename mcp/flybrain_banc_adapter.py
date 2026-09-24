"""Fail-closed local adapter for the pinned BANC v888 snapshot.

BANC (Brain-And-Nerve-Cord, Bates/Phelps/Kim/Yang et al. 2026) is the first
adult *Drosophila* connectome that spans brain + ventral nerve cord. This
module is the only place the harness touches the on-disk BANC snapshot:

* ``open_banc_snapshot`` validates the storage root, the manifest
  (``fbh-manifest/v1`` self-hash + sidecar, verification status, refresh
  window, license), the size of every listed file and the sha256 of every
  required product before returning resolved paths. Content hashes are cached
  as write-once stamps keyed by (size, mtime_ns) (``flybrain_hash_stamps``),
  so a 29 GB snapshot is not rehashed on every open.
* ``load_banc_*`` readers check required columns and return pandas frames with
  18-19 digit root ids kept as strings (never float-promoted).
* ``map_banc_region`` is the explicit brain-vs-nerve-cord region vocabulary map.
  Unknown vocabularies raise instead of being silently bucketed.

No network access: everything is read from
``$LOCI_FLYBRAIN_STORAGE_ROOT/snapshots/BANC/banc_888``. See
``docs/FLYBRAIN_BANC_ADAPTER_CONTRACT.md``.
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

BANC_DATASET_SYMBOL = "banc"
BANC_VERSION_ID = "banc_888"
BANC_SNAPSHOT_DIR_NAME = "BANC"
BANC_SNAPSHOT_RELATIVE_ROOT = "snapshots/BANC/banc_888"
BANC_MANIFEST_SCHEMA_VERSION = "fbh-manifest/v1"
BANC_MANIFEST_RELATIVE_PATH = "manifest/manifest.json"
BANC_MANIFEST_SIDECAR_RELATIVE_PATH = "manifest/manifest.sha256"
BANC_ALLOWED_LICENSES = frozenset({"CC-BY-4.0"})
BANC_CITATION = (
    "Bates AS, Phelps JS, Kim M, Yang HHJ, et al. (2026). Distributed control circuits across a "
    "brain-and-cord connectome. Nature. doi:10.1038/s41586-026-10735-w. Data: BANC v888, Harvard "
    "Dataverse doi:10.7910/DVN/7WTH1N (CC BY 4.0)."
)

# Product roles -> snapshot-relative paths (relative to BANC_SNAPSHOT_RELATIVE_ROOT).
ROLE_EDGELIST_V3 = "edgelist_v3"
ROLE_NT_PREDICTION = "nt_prediction"
ROLE_META = "meta"
BANC_PRODUCT_PATHS: Mapping[str, str] = {
    ROLE_EDGELIST_V3: "source/compiled_data/banc_888_edgelist_simple_v3.feather",
    ROLE_NT_PREDICTION: "source/compiled_data/banc_888_neurotransmitter_prediction_v2.csv",
    ROLE_META: "metadata/banc_888_meta.feather",
}

EDGELIST_REQUIRED_COLUMNS: tuple[str, ...] = ("pre", "post", "count", "pre_count", "post_count")
NT_PREDICTION_REQUIRED_COLUMNS: tuple[str, ...] = (
    "root_id",
    "acetylcholine",
    "dopamine",
    "gaba",
    "glutamate",
    "histamine",
    "octopamine",
    "serotonin",
    "tyramine",
    "neurotransmitter_predicted",
    "neurotransmitter_score",
    "count",
)
META_REQUIRED_COLUMNS: tuple[str, ...] = (
    "banc_888_id",
    "proofread",
    "side",
    "root_region",
    "region",
    "hemilineage",
    "flow",
    "super_class",
    "cell_class",
)

# Full-name NT -> short code; codes shared with fw (ach/gaba/glut/da/ser/oct)
# plus the two BANC-only classes (histamine, tyramine).
BANC_NT_SHORT_CODES: Mapping[str, str] = {
    "acetylcholine": "ach",
    "gaba": "gaba",
    "glutamate": "glut",
    "dopamine": "da",
    "serotonin": "ser",
    "octopamine": "oct",
    "histamine": "his",
    "tyramine": "tyr",
}

# Explicit brain-vs-nerve-cord vocabulary. ``root_region`` values are
# "<atlas>_<part>_<neuropil>[_<L|R>]"; the prefix decides the CNS division.
DIVISION_BRAIN = "brain"
DIVISION_NERVE_CORD = "nerve_cord"
BANC_ROOT_REGION_PREFIXES: Mapping[str, tuple[str, str]] = {
    "ITO_optic_": (DIVISION_BRAIN, "optic_lobe"),
    "ITO_midbrain_": (DIVISION_BRAIN, "central_brain"),
    "COURT_vnc_": (DIVISION_NERVE_CORD, "ventral_nerve_cord"),
    "MANC_vnc_": (DIVISION_NERVE_CORD, "ventral_nerve_cord"),
}
# Curated per-neuron ``region`` column -> (division, subdivision).
BANC_CURATED_REGIONS: Mapping[str, tuple[str, str]] = {
    "optic_lobe": (DIVISION_BRAIN, "optic_lobe"),
    "central_brain": (DIVISION_BRAIN, "central_brain"),
    "ventral_nerve_cord": (DIVISION_NERVE_CORD, "ventral_nerve_cord"),
}

_SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")
_SIDE_SUFFIX_RE = re.compile(r"_(L|R)$")
_REGION_SANITIZE_RE = re.compile(r"[^a-z0-9]+")
_BLOCKED_ARTIFACT_SUFFIXES = (".partial", ".tmp", ".inprogress")
_REFRESH_DECISIONS = frozenset({"no_change", "patch_refresh", "major_bump", "rollback"})


class BancAdapterErrorCode(str, Enum):
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


class BancAdapterError(ValueError):
    """Typed fail-closed error; message is prefixed with ``[CODE]``."""

    def __init__(self, code: BancAdapterErrorCode, message: str, details: Mapping[str, Any] | None = None):
        self.code = code
        self.message = message
        self.details = dict(details or {})
        super().__init__(f"[{code.value}] {message}")


# ---------------------------------------------------------------------------
# Region vocabulary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BancRegion:
    root_region: str
    division: str
    subdivision: str
    neuropil: str
    side: str | None
    region_id: str


def _sanitize(raw: str) -> str:
    value = _REGION_SANITIZE_RE.sub("_", str(raw).strip().lower()).strip("_")
    return value or "unknown_region"


def map_banc_region(root_region: str) -> BancRegion:
    """Map a BANC ``root_region`` label onto the explicit brain/nerve-cord vocabulary.

    ``region_id`` is ``<division>_<neuropil>`` with the hemisphere suffix
    removed (``ITO_optic_ME_R`` -> ``brain_me``, ``COURT_vnc_ProNM-T1`` ->
    ``nerve_cord_pronm_t1``). Unknown prefixes raise
    ``REGION_VOCABULARY_UNKNOWN``.
    """
    text = str(root_region or "").strip()
    if not text:
        raise BancAdapterError(BancAdapterErrorCode.REGION_VOCABULARY_UNKNOWN, "root_region is empty")
    for prefix in sorted(BANC_ROOT_REGION_PREFIXES, key=lambda p: (-len(p), p)):
        if text.startswith(prefix):
            division, subdivision = BANC_ROOT_REGION_PREFIXES[prefix]
            remainder = text[len(prefix):]
            side_match = _SIDE_SUFFIX_RE.search(remainder)
            side = None
            if side_match:
                side = "left" if side_match.group(1) == "L" else "right"
                remainder = remainder[: side_match.start()]
            if not remainder:
                break
            return BancRegion(
                root_region=text,
                division=division,
                subdivision=subdivision,
                neuropil=remainder,
                side=side,
                region_id=_sanitize(f"{division}_{remainder}"),
            )
    raise BancAdapterError(
        BancAdapterErrorCode.REGION_VOCABULARY_UNKNOWN,
        f"root_region {text!r} does not match a known BANC vocabulary prefix",
        {"root_region": text, "known_prefixes": sorted(BANC_ROOT_REGION_PREFIXES)},
    )


def curated_region_division(region: str | None) -> tuple[str, str] | None:
    """Map the curated ``region`` column; ``None``/NaN -> ``None``; unknown raises."""
    if region is None or (isinstance(region, float) and region != region):
        return None
    text = str(region).strip()
    if not text:
        return None
    if text not in BANC_CURATED_REGIONS:
        raise BancAdapterError(
            BancAdapterErrorCode.REGION_VOCABULARY_UNKNOWN,
            f"curated region {text!r} is not in the BANC division map",
            {"region": text, "known": sorted(BANC_CURATED_REGIONS)},
        )
    return BANC_CURATED_REGIONS[text]


def nt_short_code(name: str) -> str:
    text = str(name or "").strip().lower()
    if text not in BANC_NT_SHORT_CODES:
        raise BancAdapterError(
            BancAdapterErrorCode.NT_VOCABULARY_UNKNOWN,
            f"neurotransmitter {name!r} is not in the BANC NT vocabulary",
            {"neurotransmitter": name, "known": sorted(BANC_NT_SHORT_CODES)},
        )
    return BANC_NT_SHORT_CODES[text]


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
        raise BancAdapterError(BancAdapterErrorCode.MANIFEST_INVALID, "manifest.integrity must be an object")
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


def resolve_banc_snapshot_root(storage_root: str | Path | None = None) -> tuple[Path, Path]:
    """Return ``(storage_root, snapshot_root)`` after root + path-escape validation."""
    try:
        layout = build_flybrain_harness_layout(root_override=storage_root, create=False)
    except ValueError as exc:
        raise BancAdapterError(BancAdapterErrorCode.ROOT_NOT_CONFIGURED, str(exc)) from exc
    spec = get_dataset(BANC_DATASET_SYMBOL)
    if spec.pinned_version != BANC_VERSION_ID or spec.snapshot_dir_name != BANC_SNAPSHOT_DIR_NAME:
        raise BancAdapterError(
            BancAdapterErrorCode.DATASET_PIN_MISMATCH,
            "registry pin for banc does not match the adapter pin",
            {"registry_version": spec.pinned_version, "adapter_version": BANC_VERSION_ID},
        )
    snapshot_root = snapshot_version_root(BANC_DATASET_SYMBOL, layout.root).resolve(strict=False)
    if not snapshot_root.is_relative_to(layout.root.resolve(strict=False)):
        raise BancAdapterError(
            BancAdapterErrorCode.PATH_ESCAPE,
            "BANC snapshot path escapes the configured storage root",
            {"snapshot_root": str(snapshot_root)},
        )
    return layout.root, snapshot_root


def build_banc_manifest(
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
    """Build a verified ``fbh-manifest/v1`` dict for the BANC snapshot (self-hash applied)."""
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
        "system": "banc-gcs-public",
        "access_method": "https_public_bucket_selective_pull",
        "uri": source_uri,
        "retrieved_at": retrieved_at,
        "license": {"spdx_id": license_spdx, "url": "https://creativecommons.org/licenses/by/4.0/"},
        "citation": BANC_CITATION,
        "doi": "10.7910/DVN/7WTH1N",
    }
    source.update(dict(extra_source or {}))
    manifest: dict[str, Any] = {
        "schema_version": BANC_MANIFEST_SCHEMA_VERSION,
        "manifest_id": f"fbh-BANC-{BANC_VERSION_ID}-{generated}",
        "generated_at": generated,
        "storage_root_env": "LOCI_FLYBRAIN_STORAGE_ROOT",
        "artifact": {
            "kind": "dataset_snapshot",
            "relative_root": BANC_SNAPSHOT_RELATIVE_ROOT,
            "path_template": "$LOCI_FLYBRAIN_STORAGE_ROOT\\snapshots\\BANC\\banc_888\\",
        },
        "dataset": {
            "symbol": "BANC",
            "label": "Brain-And-Nerve-Cord connectome v888 (selective products)",
            "version_id": BANC_VERSION_ID,
            "source": source,
        },
        "scope": {
            "sex": "female",
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
            "derived_from": [{"type": "remote_dataset_release", "id": "doi:10.7910/DVN/7WTH1N#banc_888"}],
            "pipeline": {
                "job_name": "fbh_banc_selective_pull",
                "job_version": "2026.09.23.1",
                "run_id": run_id,
                "run_mode": "execute",
            },
        },
        "notes": list(notes),
    }
    manifest["integrity"]["manifest_sha256"] = manifest_digest(manifest)
    return manifest


def write_banc_manifest(snapshot_root: Path, manifest: Mapping[str, Any]) -> Path:
    """Write manifest.json + manifest.sha256; refuses to overwrite (MANIFEST_EXISTS)."""
    target = Path(snapshot_root) / BANC_MANIFEST_RELATIVE_PATH
    sidecar = Path(snapshot_root) / BANC_MANIFEST_SIDECAR_RELATIVE_PATH
    for path in (target, sidecar):
        if path.exists():
            raise BancAdapterError(
                BancAdapterErrorCode.MANIFEST_EXISTS,
                "refusing to overwrite an existing BANC manifest; supersede with a new version instead",
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
class BancSnapshot:
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
            raise BancAdapterError(BancAdapterErrorCode.PRODUCT_MISSING, f"product role {role!r} not in snapshot")
        return self.product_paths[role]

    def provenance(self) -> dict[str, Any]:
        return {
            "dataset_symbol": BANC_DATASET_SYMBOL,
            "version_id": BANC_VERSION_ID,
            "manifest_id": self.manifest_id,
            "manifest_sha256": self.manifest_sha256,
            "source_path": BANC_SNAPSHOT_RELATIVE_ROOT,
            "access_method": "local_snapshot_read_only",
            "license": self.license_spdx,
            "product_sha256": dict(sorted(self.product_sha256.items())),
        }


def _fail(code: BancAdapterErrorCode, message: str, **details: Any) -> None:
    raise BancAdapterError(code, message, details)


def validate_banc_manifest(
    manifest: Mapping[str, Any],
    snapshot_root: Path,
    *,
    required_roles: Sequence[str] = tuple(BANC_PRODUCT_PATHS),
    verify_hashes: bool | None = None,
    now: datetime | None = None,
    stamp_cache: HashStampCache | None = None,
    hash_report: dict[str, str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, str]:
    """Fail-closed manifest check. Returns ``{relative_path: sha256}`` for listed files.

    Every listed file is always checked for presence, path safety and exact
    ``size_bytes``. Content hashing depends on ``verify_hashes``:

    * ``True`` (explicit verify): sha256 of every listed file, always.
    * ``None`` (default): sha256 of the files backing ``required_roles`` only,
      skipped when ``stamp_cache`` holds a write-once stamp for the file's
      current ``(size, mtime_ns)``. Other listed files are size-checked only.
    * ``False``: size checks only (smoke runs).

    ``hash_report`` (if given) receives ``relative_path -> method``.
    """
    if manifest.get("schema_version") != BANC_MANIFEST_SCHEMA_VERSION:
        _fail(BancAdapterErrorCode.MANIFEST_INVALID, "manifest schema_version mismatch",
              schema_version=manifest.get("schema_version"))
    for key in ("manifest_id", "generated_at", "artifact", "dataset", "scope", "integrity", "refresh", "lineage"):
        if key not in manifest:
            _fail(BancAdapterErrorCode.MANIFEST_INVALID, f"manifest missing required field {key!r}")
    dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), dict) else {}
    try:
        symbol = normalize_symbol(str(dataset.get("symbol") or ""))
    except Exception:
        symbol = str(dataset.get("symbol"))
    if symbol != BANC_DATASET_SYMBOL or dataset.get("version_id") != BANC_VERSION_ID:
        _fail(BancAdapterErrorCode.DATASET_PIN_MISMATCH, "manifest dataset pin does not match BANC banc_888",
              symbol=dataset.get("symbol"), version_id=dataset.get("version_id"))
    source = dataset.get("source") if isinstance(dataset.get("source"), dict) else {}
    license_info = source.get("license") if isinstance(source.get("license"), dict) else {}
    spdx = str(license_info.get("spdx_id") or "").strip()
    if spdx not in BANC_ALLOWED_LICENSES:
        _fail(BancAdapterErrorCode.LICENSE_NOT_PERMITTED, "manifest license is missing or not permitted",
              spdx_id=spdx, allowed=sorted(BANC_ALLOWED_LICENSES))
    artifact = manifest.get("artifact") if isinstance(manifest.get("artifact"), dict) else {}
    relative_root = str(artifact.get("relative_root") or "").replace("\\", "/").strip("/")
    if not is_safe_relative_path(relative_root):
        _fail(BancAdapterErrorCode.PATH_ESCAPE, "artifact.relative_root is unsafe", relative_root=relative_root)
    if relative_root != BANC_SNAPSHOT_RELATIVE_ROOT:
        _fail(BancAdapterErrorCode.DATASET_PIN_MISMATCH, "artifact.relative_root does not match the BANC pin",
              relative_root=relative_root, expected=BANC_SNAPSHOT_RELATIVE_ROOT)

    integrity = manifest.get("integrity") if isinstance(manifest.get("integrity"), dict) else {}
    verification = integrity.get("verification") if isinstance(integrity.get("verification"), dict) else {}
    status = str(verification.get("status") or "").strip().lower()
    if status not in {"pending", "verified", "quarantined"}:
        _fail(BancAdapterErrorCode.MANIFEST_INVALID, "integrity.verification.status invalid", status=status)
    if status != "verified":
        _fail(BancAdapterErrorCode.INTEGRITY_MISMATCH, "manifest verification status must be verified", status=status)
    if not str(verification.get("verified_at") or "").strip():
        _fail(BancAdapterErrorCode.MANIFEST_INVALID, "verified_at is required when status is verified")
    expected = str(integrity.get("manifest_sha256") or "").strip()
    if not _SHA256_HEX_RE.fullmatch(expected):
        _fail(BancAdapterErrorCode.MANIFEST_INVALID, "integrity.manifest_sha256 must be lowercase 64-hex")
    actual = manifest_digest(manifest)
    if actual != expected:
        _fail(BancAdapterErrorCode.INTEGRITY_MISMATCH, "manifest self-hash mismatch", expected=expected, actual=actual)

    refresh = manifest.get("refresh") if isinstance(manifest.get("refresh"), dict) else {}
    decision = str(refresh.get("decision") or "").strip().lower()
    if decision not in _REFRESH_DECISIONS:
        _fail(BancAdapterErrorCode.MANIFEST_INVALID, "refresh.decision invalid", decision=decision)
    if decision == "rollback":
        _fail(BancAdapterErrorCode.PROMOTION_STATE_INVALID, "refresh.decision indicates rollback")
    due = refresh.get("next_check_due")
    if due:
        try:
            due_dt = datetime.fromisoformat(str(due).replace("Z", "+00:00"))
        except ValueError:
            _fail(BancAdapterErrorCode.MANIFEST_INVALID, "refresh.next_check_due not RFC3339", next_check_due=due)
        if due_dt < (now or datetime.now(timezone.utc)):
            _fail(BancAdapterErrorCode.PROMOTION_STATE_INVALID, "refresh.next_check_due has elapsed", next_check_due=due)

    files = integrity.get("files")
    if not isinstance(files, list) or not files:
        _fail(BancAdapterErrorCode.MANIFEST_INVALID, "integrity.files must be a non-empty list")
    root = Path(snapshot_root).resolve(strict=False)
    seen: set[str] = set()
    hashes: dict[str, str] = {}
    listed: list[tuple[str, Path, str]] = []
    for entry in files:
        if not isinstance(entry, dict):
            _fail(BancAdapterErrorCode.MANIFEST_INVALID, "integrity.files entries must be objects")
        rel = str(entry.get("relative_path") or "").strip()
        sha = str(entry.get("sha256") or "").strip()
        size = entry.get("size_bytes")
        if not is_safe_relative_path(rel):
            _fail(BancAdapterErrorCode.PATH_ESCAPE, "integrity file path is unsafe", relative_path=rel)
        folded = rel.replace("\\", "/").casefold()
        if folded in seen:
            _fail(BancAdapterErrorCode.MANIFEST_INVALID, "duplicate integrity path after case-folding", relative_path=rel)
        seen.add(folded)
        if rel.lower().endswith(_BLOCKED_ARTIFACT_SUFFIXES):
            _fail(BancAdapterErrorCode.MANIFEST_INVALID, "partial/temporary artifact listed", relative_path=rel)
        if not _SHA256_HEX_RE.fullmatch(sha):
            _fail(BancAdapterErrorCode.MANIFEST_INVALID, "file sha256 must be lowercase 64-hex", relative_path=rel)
        path = (root / Path(rel)).resolve(strict=False)
        if not path.is_relative_to(root):
            _fail(BancAdapterErrorCode.PATH_ESCAPE, "integrity file escapes snapshot root", relative_path=rel)
        if not path.is_file():
            _fail(BancAdapterErrorCode.PRODUCT_MISSING, "integrity file is missing", relative_path=rel, path=str(path))
        if not isinstance(size, int) or size < 0:
            _fail(BancAdapterErrorCode.MANIFEST_INVALID, "size_bytes must be a non-negative integer", relative_path=rel)
        if path.stat().st_size != size:
            _fail(BancAdapterErrorCode.INTEGRITY_MISMATCH, "file size mismatch", relative_path=rel,
                  expected=size, actual=path.stat().st_size)
        normalized_rel = rel.replace("\\", "/")
        hashes[normalized_rel] = sha
        listed.append((normalized_rel, path, sha))
    for role in required_roles:
        rel = BANC_PRODUCT_PATHS[role]
        if rel not in hashes:
            _fail(BancAdapterErrorCode.PRODUCT_MISSING, f"manifest does not list required product {role!r}",
                  role=role, relative_path=rel)

    # Content hashing runs only after every metadata, path and size check passed.
    required_paths = {BANC_PRODUCT_PATHS[role] for role in required_roles}
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
            _fail(BancAdapterErrorCode.INTEGRITY_MISMATCH, "file sha256 mismatch", relative_path=rel,
                  expected=sha, actual=check.actual_sha256)
        if hash_report is not None:
            hash_report[rel] = check.method
        if check.warning and warnings is not None:
            warnings.append(check.warning)
    return hashes


def open_banc_snapshot(
    storage_root: str | Path | None = None,
    *,
    required_roles: Sequence[str] = tuple(BANC_PRODUCT_PATHS),
    verify_hashes: bool | None = None,
    now: datetime | None = None,
) -> BancSnapshot:
    """Resolve + verify the pinned BANC snapshot. Raises ``BancAdapterError`` on any gap.

    The manifest self-hash and the ``manifest.sha256`` sidecar are checked on
    every open, before any file content is hashed. File contents are then
    hashed as described in ``validate_banc_manifest``: by default only the
    required products, and only when no stamp under ``manifest/hash-stamps/``
    matches the file's current (size, mtime_ns). ``verify_hashes=True`` forces a
    full rehash of every listed file.
    """
    for role in required_roles:
        if role not in BANC_PRODUCT_PATHS:
            _fail(BancAdapterErrorCode.PRODUCT_MISSING, f"unknown product role {role!r}", role=role)
    root, snapshot_root = resolve_banc_snapshot_root(storage_root)
    if not snapshot_root.is_dir():
        _fail(BancAdapterErrorCode.SNAPSHOT_MISSING, "BANC snapshot directory not found", snapshot_root=str(snapshot_root))
    manifest_path = snapshot_root / BANC_MANIFEST_RELATIVE_PATH
    if not manifest_path.is_file():
        _fail(BancAdapterErrorCode.MANIFEST_MISSING, "BANC manifest.json not found", manifest_path=str(manifest_path))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BancAdapterError(BancAdapterErrorCode.MANIFEST_INVALID, f"manifest unreadable: {exc}") from exc
    if not isinstance(manifest, dict):
        _fail(BancAdapterErrorCode.MANIFEST_INVALID, "manifest must be a JSON object")
    sidecar = snapshot_root / BANC_MANIFEST_SIDECAR_RELATIVE_PATH
    warnings: list[str] = []
    # Fail closed: the sidecar is written with the manifest (write_banc_manifest)
    # and is the only integrity anchor outside manifest.json itself. It is
    # checked before any file content is hashed.
    if not sidecar.is_file():
        _fail(BancAdapterErrorCode.MANIFEST_MISSING, "manifest.sha256 sidecar is missing", sidecar=str(sidecar))
    integrity = manifest.get("integrity") if isinstance(manifest.get("integrity"), dict) else {}
    if sidecar.read_text(encoding="utf-8").strip() != str(integrity.get("manifest_sha256") or ""):
        _fail(BancAdapterErrorCode.INTEGRITY_MISMATCH, "manifest.sha256 sidecar does not match manifest")
    hash_report: dict[str, str] = {}
    hashes = validate_banc_manifest(
        manifest,
        snapshot_root,
        required_roles=required_roles,
        verify_hashes=verify_hashes,
        now=now,
        stamp_cache=HashStampCache.for_snapshot(snapshot_root),
        hash_report=hash_report,
        warnings=warnings,
    )
    product_paths = {role: (snapshot_root / BANC_PRODUCT_PATHS[role]).resolve(strict=False) for role in required_roles}
    product_sha = {role: hashes[BANC_PRODUCT_PATHS[role]] for role in required_roles}
    return BancSnapshot(
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
        _fail(BancAdapterErrorCode.PRODUCT_MISSING, f"BANC {role} file not found", path=str(candidate))
    return candidate


def _feather_columns(path: Path) -> list[str]:
    """Read only the Arrow IPC schema (no column decompression)."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    try:
        with pa.memory_map(str(path), "r") as source:
            return list(ipc.open_file(source).schema.names)
    except (OSError, pa.ArrowInvalid) as exc:
        raise BancAdapterError(
            BancAdapterErrorCode.SCHEMA_MISMATCH, f"not a readable Arrow/feather v2 file: {exc}", {"path": str(path)}
        ) from exc


def _check_columns(available: Iterable[str], required: Sequence[str], role: str, path: Path) -> None:
    have = set(available)
    missing = [col for col in required if col not in have]
    if missing:
        _fail(BancAdapterErrorCode.SCHEMA_MISMATCH, f"BANC {role} missing required columns: {', '.join(missing)}",
              role=role, path=str(path), missing=missing)


def load_banc_meta(path: Path, *, columns: Sequence[str] = META_REQUIRED_COLUMNS,
                   optional_columns: Sequence[str] = ()):
    """Per-neuron metadata (feather or parquet) with ``banc_888_id`` as string.

    ``optional_columns`` are read when the file has them and otherwise skipped.
    """
    import pyarrow as pa
    import pyarrow.feather as feather
    import pyarrow.parquet as pq

    source = _require_file(path, "meta")
    wanted = list(dict.fromkeys(["banc_888_id", *columns]))
    if source.suffix.lower() == ".parquet":
        names = pq.read_schema(str(source)).names
        _check_columns(names, wanted, "meta", source)
        wanted += [col for col in optional_columns if col in names and col not in wanted]
        table = pq.read_table(str(source), columns=wanted)
    else:
        names = _feather_columns(source)
        _check_columns(names, wanted, "meta", source)
        wanted += [col for col in optional_columns if col in names and col not in wanted]
        table = feather.read_table(str(source), columns=wanted)
    table = table.set_column(
        table.schema.get_field_index("banc_888_id"), "banc_888_id", table.column("banc_888_id").cast(pa.string())
    )
    frame = table.to_pandas()
    if frame["banc_888_id"].isna().any():
        _fail(BancAdapterErrorCode.SCHEMA_MISMATCH, "meta has null banc_888_id", path=str(source))
    if frame["banc_888_id"].duplicated().any():
        _fail(BancAdapterErrorCode.SCHEMA_MISMATCH, "meta banc_888_id is not unique", path=str(source))
    return frame


def load_banc_outgoing_totals(path: Path):
    """Aggregate the v3 edgelist per presynaptic neuron.

    Returns a frame with ``root_id`` (str), ``total_out_synapses`` (sum of
    ``count``) and ``n_post_partners`` (distinct ``post``), sorted by root id.
    """
    import pyarrow as pa
    import pyarrow.feather as feather

    source = _require_file(path, "edgelist_v3")
    names = _feather_columns(source)
    _check_columns(names, EDGELIST_REQUIRED_COLUMNS, "edgelist_v3", source)
    table = feather.read_table(str(source), columns=["pre", "post", "count"])
    if table.num_rows == 0:
        _fail(BancAdapterErrorCode.SCHEMA_MISMATCH, "edgelist is empty", path=str(source))
    pre = table.column("pre").cast(pa.string())
    counts = table.column("count").cast(pa.int64())
    if counts.null_count or pre.null_count:
        _fail(BancAdapterErrorCode.SCHEMA_MISMATCH, "edgelist has null pre/count values", path=str(source))
    slim = pa.table({"pre": pre, "count": counts})
    grouped = slim.group_by("pre").aggregate([("count", "sum"), ("count", "count")])
    frame = grouped.to_pandas().rename(
        columns={"pre": "root_id", "count_sum": "total_out_synapses", "count_count": "n_post_partners"}
    )
    frame["total_out_synapses"] = frame["total_out_synapses"].astype("int64")
    frame["n_post_partners"] = frame["n_post_partners"].astype("int64")
    frame = frame.sort_values("root_id", kind="mergesort").reset_index(drop=True)
    return frame, int(table.num_rows)


def load_banc_nt_predictions(path: Path):
    """Per-neuron NT prediction CSV (v2 classifier) with ids as strings.

    Adds ``nt_row_multiplicity`` (how many CSV rows shared the root_id).
    """
    import pandas as pd

    source = _require_file(path, "nt_prediction")
    header = pd.read_csv(source, nrows=0).columns.tolist()
    _check_columns(header, NT_PREDICTION_REQUIRED_COLUMNS, "nt_prediction", source)
    frame = pd.read_csv(
        source,
        usecols=list(NT_PREDICTION_REQUIRED_COLUMNS),
        dtype={"root_id": str, "neurotransmitter_predicted": str},
    )
    if frame["root_id"].isna().any():
        _fail(BancAdapterErrorCode.SCHEMA_MISMATCH, "nt_prediction has null root_id", path=str(source))
    # The published CSV repeats a root_id once per SeaTable anchor (supervoxel /
    # cell_type row). Repeats with identical prediction columns are collapsed and
    # counted in ``nt_row_multiplicity``; repeats that disagree fail closed.
    multiplicity = frame.groupby("root_id", sort=False)["root_id"].transform("size").astype("int64")
    frame = frame.assign(nt_row_multiplicity=multiplicity)
    collapsed = frame.drop_duplicates()
    if collapsed["root_id"].duplicated().any():
        conflicting = sorted(collapsed.loc[collapsed["root_id"].duplicated(), "root_id"].unique().tolist())[:5]
        _fail(BancAdapterErrorCode.SCHEMA_MISMATCH, "nt_prediction has conflicting rows for the same root_id",
              path=str(source), examples=conflicting)
    return collapsed.reset_index(drop=True)


__all__ = [
    "BANC_ALLOWED_LICENSES",
    "BANC_CITATION",
    "BANC_CURATED_REGIONS",
    "BANC_DATASET_SYMBOL",
    "BANC_MANIFEST_RELATIVE_PATH",
    "BANC_NT_SHORT_CODES",
    "BANC_PRODUCT_PATHS",
    "BANC_ROOT_REGION_PREFIXES",
    "BANC_SNAPSHOT_RELATIVE_ROOT",
    "BANC_VERSION_ID",
    "BancAdapterError",
    "BancAdapterErrorCode",
    "BancRegion",
    "BancSnapshot",
    "DIVISION_BRAIN",
    "DIVISION_NERVE_CORD",
    "ROLE_EDGELIST_V3",
    "ROLE_META",
    "ROLE_NT_PREDICTION",
    "build_banc_manifest",
    "curated_region_division",
    "load_banc_meta",
    "load_banc_nt_predictions",
    "load_banc_outgoing_totals",
    "manifest_digest",
    "map_banc_region",
    "nt_short_code",
    "open_banc_snapshot",
    "resolve_banc_snapshot_root",
    "validate_banc_manifest",
    "write_banc_manifest",
]
