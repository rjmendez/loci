from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from flybrain_harness_storage import build_flybrain_harness_layout, resolve_flybrain_storage_root
from replay_fingerprint import flybrain_replay_fingerprint

HB_DATASET_SYMBOL = "hb"
HB_VERSION_ID = "neuprint_JRC_Hemibrain_1point2point1"
HB_MANIFEST_SCHEMA_VERSION = "fbh-manifest/v1"
HB_QUERY_RESULT_VERSION = "hb-query-result/v1"
HB_PROVENANCE_VERSION = "hb-provenance/v1"
HB_ACCESS_METHOD = "local_graph_snapshot"
HB_SOURCE = "local"
HB_SCOPE_SEX = "female"
HB_SCOPE_STAGE = "adult"
HB_SCOPE_ANATOMY = "hemibrain_region"
HB_SCOPE_EVIDENCE_FAMILY = "connectome_structural"
HB_ACCESS_LAYER = "local_graph_snapshot"
HB_ALLOWED_QUERY_KINDS = ("connectivity_lookup", "neighborhood_traversal", "replay_smoke", "scope_smoke")
HB_ROLLBACK_ERRORS = {"MANIFEST_INVALID", "INTEGRITY_MISMATCH", "PROMOTION_STATE_INVALID", "ROLLBACK_REQUIRED"}
HB_MANIFEST_RELATIVE = Path("snapshots") / HB_DATASET_SYMBOL / HB_VERSION_ID / "manifest" / "manifest.json"
HB_ACTIVE_POINTER_RELATIVE = Path("graph") / HB_DATASET_SYMBOL / HB_VERSION_ID / "promotion" / "active_pointer.json"
HB_GRAPH_ROOT_RELATIVE = Path("graph") / HB_DATASET_SYMBOL / HB_VERSION_ID
_SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")
_BLOCKED_ARTIFACT_SUFFIXES = (".partial", ".tmp", ".inprogress")


class HbAdapterErrorCode(str, Enum):
    ROOT_NOT_CONFIGURED = "ROOT_NOT_CONFIGURED"
    ROOT_NOT_ABSOLUTE = "ROOT_NOT_ABSOLUTE"
    ROOT_OUT_OF_BOUNDS = "ROOT_OUT_OF_BOUNDS"
    ROOT_REPARSE_POINT = "ROOT_REPARSE_POINT"
    DATASET_PIN_MISMATCH = "DATASET_PIN_MISMATCH"
    UNSUPPORTED_QUERY_KIND = "UNSUPPORTED_QUERY_KIND"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    SCOPE_VIOLATION = "SCOPE_VIOLATION"
    READ_ONLY_REQUIRED = "READ_ONLY_REQUIRED"
    MANIFEST_MISSING = "MANIFEST_MISSING"
    MANIFEST_INVALID = "MANIFEST_INVALID"
    INTEGRITY_MISMATCH = "INTEGRITY_MISMATCH"
    PROMOTION_STATE_INVALID = "PROMOTION_STATE_INVALID"
    ROLLBACK_REQUIRED = "ROLLBACK_REQUIRED"
    QUERY_LIMIT_EXCEEDED = "QUERY_LIMIT_EXCEEDED"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    UNIMPLEMENTED_BACKEND = "UNIMPLEMENTED_BACKEND"


@dataclass(frozen=True)
class HbScopeTags:
    sex: str = HB_SCOPE_SEX
    stage: str = HB_SCOPE_STAGE
    anatomy: str = HB_SCOPE_ANATOMY
    evidence_family: str = HB_SCOPE_EVIDENCE_FAMILY
    access_layer: str = HB_ACCESS_LAYER

    def to_dict(self) -> dict[str, str]:
        return {
            "sex": self.sex,
            "stage": self.stage,
            "anatomy": self.anatomy,
            "evidence_family": self.evidence_family,
            "access_layer": self.access_layer,
        }


@dataclass(frozen=True)
class HbQueryRequest:
    query_kind: str
    query_payload: Mapping[str, Any] = field(default_factory=dict)
    dataset_symbol: str = HB_DATASET_SYMBOL
    version_id: str = HB_VERSION_ID
    scope_tags: HbScopeTags = field(default_factory=HbScopeTags)
    allow_remote_fallback: bool = False
    max_rows: int = 500
    max_depth: int = 2
    max_weight: int | None = None
    readonly: bool = True
    request_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_kind": self.query_kind,
            "query_payload": dict(self.query_payload),
            "dataset_symbol": self.dataset_symbol,
            "version_id": self.version_id,
            "scope_tags": self.scope_tags.to_dict(),
            "allow_remote_fallback": self.allow_remote_fallback,
            "max_rows": self.max_rows,
            "max_depth": self.max_depth,
            "max_weight": self.max_weight,
            "readonly": self.readonly,
            "request_id": self.request_id,
        }


@dataclass(frozen=True)
class HbCompatibilityCheck:
    name: str
    passed: bool
    code: str | None = None
    detail: str | None = None
    rollback_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "code": self.code,
            "detail": self.detail,
            "rollback_required": self.rollback_required,
        }


@dataclass(frozen=True)
class HbManifestSummary:
    manifest_id: str
    manifest_sha256: str
    manifest_path: str
    graph_root: str
    active_pointer_path: str
    verification_status: str
    refresh_decision: str
    next_check_due: str | None = None
    rollback_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "manifest_sha256": self.manifest_sha256,
            "manifest_path": self.manifest_path,
            "graph_root": self.graph_root,
            "active_pointer_path": self.active_pointer_path,
            "verification_status": self.verification_status,
            "refresh_decision": self.refresh_decision,
            "next_check_due": self.next_check_due,
            "rollback_required": self.rollback_required,
        }


@dataclass(frozen=True)
class HbSnapshotHandle:
    root: Path
    graph_root: Path
    manifest_path: Path
    active_pointer_path: Path
    manifest_id: str
    manifest_sha256: str
    verification_status: str
    refresh_decision: str
    next_check_due: str | None = None
    rollback_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "graph_root": str(self.graph_root),
            "manifest_path": str(self.manifest_path),
            "active_pointer_path": str(self.active_pointer_path),
            "manifest_id": self.manifest_id,
            "manifest_sha256": self.manifest_sha256,
            "verification_status": self.verification_status,
            "refresh_decision": self.refresh_decision,
            "next_check_due": self.next_check_due,
            "rollback_required": self.rollback_required,
        }


@dataclass(frozen=True)
class HbProvenanceEnvelope:
    schema_version: str
    dataset_symbol: str
    version_id: str
    source: str
    access_method: str
    access_path: str
    query_kind: str
    scope_tags: dict[str, str]
    manifest_id: str
    manifest_sha256: str
    manifest_path: str
    graph_root: str
    active_pointer_path: str
    replay_fingerprint: str
    generated_at: str
    result_contract_version: str = HB_QUERY_RESULT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_symbol": self.dataset_symbol,
            "version_id": self.version_id,
            "source": self.source,
            "access_method": self.access_method,
            "access_path": self.access_path,
            "query_kind": self.query_kind,
            "scope_tags": dict(self.scope_tags),
            "manifest_id": self.manifest_id,
            "manifest_sha256": self.manifest_sha256,
            "manifest_path": self.manifest_path,
            "graph_root": self.graph_root,
            "active_pointer_path": self.active_pointer_path,
            "replay_fingerprint": self.replay_fingerprint,
            "generated_at": self.generated_at,
            "result_contract_version": self.result_contract_version,
        }

@dataclass(frozen=True)
class HbQueryResult:
    status: str
    count_status: str
    dataset_symbol: str
    version_id: str
    query_kind: str
    source: str
    row_count: int
    columns: list[str]
    rows: list[dict[str, Any]]
    warnings: list[str]
    provenance: HbProvenanceEnvelope
    compatibility_checks: list[HbCompatibilityCheck]
    rollback_required: bool = False
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "count_status": self.count_status,
            "dataset_symbol": self.dataset_symbol,
            "version_id": self.version_id,
            "query_kind": self.query_kind,
            "source": self.source,
            "row_count": self.row_count,
            "columns": list(self.columns),
            "rows": [dict(row) for row in self.rows],
            "warnings": list(self.warnings),
            "provenance": self.provenance.to_dict(),
            "compatibility_checks": [check.to_dict() for check in self.compatibility_checks],
            "rollback_required": self.rollback_required,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "query_result_version": HB_QUERY_RESULT_VERSION,
        }

    @classmethod
    def error(cls, *, request: HbQueryRequest, provenance: HbProvenanceEnvelope, code: HbAdapterErrorCode | str, message: str, compatibility_checks: Sequence[HbCompatibilityCheck] = (), rollback_required: bool = False) -> "HbQueryResult":
        error_code = code.value if isinstance(code, HbAdapterErrorCode) else str(code)
        return cls(
            status="error",
            count_status="unavailable",
            dataset_symbol=request.dataset_symbol,
            version_id=request.version_id,
            query_kind=request.query_kind,
            source=HB_SOURCE,
            row_count=0,
            columns=[],
            rows=[],
            warnings=[message],
            provenance=provenance,
            compatibility_checks=list(compatibility_checks),
            rollback_required=rollback_required or (error_code in HB_ROLLBACK_ERRORS),
            error_code=error_code,
            error_message=message,
        )


@dataclass(frozen=True)
class HbCapabilityManifest:
    dataset_symbol: str
    version_id: str
    schema_version: str
    source: str
    access_method: str
    supported_operations: tuple[str, ...]
    supported_query_kinds: tuple[str, ...]
    scope_tags: HbScopeTags
    manifest_root: str
    graph_root: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_symbol": self.dataset_symbol,
            "version_id": self.version_id,
            "schema_version": self.schema_version,
            "source": self.source,
            "access_method": self.access_method,
            "supported_operations": list(self.supported_operations),
            "supported_query_kinds": list(self.supported_query_kinds),
            "scope_tags": self.scope_tags.to_dict(),
            "manifest_root": self.manifest_root,
            "graph_root": self.graph_root,
        }


@dataclass(frozen=True)
class HbHealthStatus:
    ok: bool
    snapshot: HbSnapshotHandle | None
    checks: list[HbCompatibilityCheck]
    capabilities: HbCapabilityManifest

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "snapshot": None if self.snapshot is None else self.snapshot.to_dict(),
            "checks": [check.to_dict() for check in self.checks],
            "capabilities": self.capabilities.to_dict(),
        }


@dataclass(frozen=True)
class HbValidationReport:
    ok: bool
    request: HbQueryRequest
    checks: list[HbCompatibilityCheck]
    snapshot: HbSnapshotHandle | None
    rollback_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "request": self.request.to_dict(),
            "checks": [check.to_dict() for check in self.checks],
            "snapshot": None if self.snapshot is None else self.snapshot.to_dict(),
            "rollback_required": self.rollback_required,
        }


class HbAdapterError(RuntimeError):
    def __init__(self, code: HbAdapterErrorCode | str, message: str, *, checks: Sequence[HbCompatibilityCheck] = (), request: HbQueryRequest | None = None, snapshot: HbSnapshotHandle | None = None, rollback_required: bool = False) -> None:
        self.code = code.value if isinstance(code, HbAdapterErrorCode) else str(code)
        self.message = message
        self.checks = list(checks)
        self.request = request
        self.snapshot = snapshot
        self.rollback_required = rollback_required or (self.code in HB_ROLLBACK_ERRORS)
        super().__init__(f"{self.code}: {self.message}")


def _is_safe_relative_path(raw: str) -> bool:
    text = str(raw or "").strip()
    if not text:
        return False
    normalized = text.replace("\\", "/")
    if normalized.startswith("/") or normalized.startswith("\\"):
        return False
    if re.match(r"^[A-Za-z]:", normalized):
        return False
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        return False
    return True


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _manifest_digest(manifest: dict[str, Any]) -> str:
    canonical = json.loads(json.dumps(manifest))
    integrity = canonical.get("integrity")
    if not isinstance(integrity, dict):
        raise HbAdapterError(HbAdapterErrorCode.MANIFEST_INVALID, "Manifest integrity section must be an object")
    integrity["manifest_sha256"] = ""
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class LocalHbAdapter:
    """Read-only local hb adapter for the FlyBrain harness."""

    def __init__(self, *, root_override: str | Path | None = None, env: Mapping[str, str] | None = None, backend: Callable[[HbQueryRequest, HbSnapshotHandle], Sequence[Mapping[str, Any]]] | None = None) -> None:
        self._env = env if env is not None else os.environ
        self._root = resolve_flybrain_storage_root(root_override=root_override, env=self._env)
        self._layout = build_flybrain_harness_layout(root_override=self._root, env=self._env, create=False)
        self._backend = backend

    @property
    def root(self) -> Path:
        return self._root

    @property
    def layout(self):
        return self._layout

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _manifest_path(self) -> Path:
        return self._root / HB_MANIFEST_RELATIVE

    def _active_pointer_path(self) -> Path:
        return self._root / HB_ACTIVE_POINTER_RELATIVE

    def _graph_root(self) -> Path:
        return self._root / HB_GRAPH_ROOT_RELATIVE

    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise HbAdapterError(HbAdapterErrorCode.MANIFEST_MISSING, f"Missing required file: {path}")
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HbAdapterError(HbAdapterErrorCode.MANIFEST_INVALID, f"Invalid JSON at {path}: {exc.msg}") from exc
        if not isinstance(loaded, dict):
            raise HbAdapterError(HbAdapterErrorCode.MANIFEST_INVALID, f"Expected JSON object at {path}")
        return loaded

    def _manifest_summary(self) -> HbManifestSummary:
        manifest = self._load_json(self._manifest_path())
        active_pointer_path = self._active_pointer_path()
        if manifest.get("schema_version") != HB_MANIFEST_SCHEMA_VERSION:
            raise HbAdapterError(HbAdapterErrorCode.MANIFEST_INVALID, f"Manifest schema_version must be {HB_MANIFEST_SCHEMA_VERSION}")
        dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), dict) else {}
        artifact = manifest.get("artifact") if isinstance(manifest.get("artifact"), dict) else {}
        source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
        integrity = manifest.get("integrity") if isinstance(manifest.get("integrity"), dict) else {}
        refresh = manifest.get("refresh") if isinstance(manifest.get("refresh"), dict) else {}
        lineage = manifest.get("lineage") if isinstance(manifest.get("lineage"), dict) else {}
        scope = manifest.get("scope") if isinstance(manifest.get("scope"), dict) else {}
        if not dataset or not artifact or not source or not integrity or not refresh or not lineage:
            raise HbAdapterError(
                HbAdapterErrorCode.MANIFEST_INVALID,
                "Manifest must include dataset, artifact, source, scope, integrity, refresh, and lineage objects",
            )
        if dataset.get("symbol") != HB_DATASET_SYMBOL or dataset.get("version_id") != HB_VERSION_ID:
            raise HbAdapterError(HbAdapterErrorCode.DATASET_PIN_MISMATCH, f"Expected {HB_DATASET_SYMBOL}/{HB_VERSION_ID}")
        artifact_relative_root = str(artifact.get("relative_root") or "").strip()
        if not _is_safe_relative_path(artifact_relative_root):
            raise HbAdapterError(HbAdapterErrorCode.ROOT_OUT_OF_BOUNDS, "Manifest artifact.relative_root must be a safe relative path")
        expected_artifact_root = str(HB_GRAPH_ROOT_RELATIVE).replace("\\", "/")
        if artifact_relative_root.replace("\\", "/") != expected_artifact_root:
            raise HbAdapterError(HbAdapterErrorCode.OUT_OF_SCOPE, f"Manifest artifact.relative_root must equal {expected_artifact_root}")
        artifact_root = (self._root / Path(artifact_relative_root)).resolve(strict=False)
        if not artifact_root.is_relative_to(self._root.resolve(strict=False)):
            raise HbAdapterError(HbAdapterErrorCode.ROOT_OUT_OF_BOUNDS, "Manifest artifact.relative_root escapes storage root")
        verification = integrity.get("verification") if isinstance(integrity.get("verification"), dict) else {}
        verification_status = str(verification.get("status") or "").strip().lower()
        if verification_status not in {"pending", "verified", "quarantined"}:
            raise HbAdapterError(HbAdapterErrorCode.MANIFEST_INVALID, f"Unexpected integrity.verification.status: {verification_status or '<missing>'}")
        if verification_status == "verified" and not str(verification.get("verified_at") or "").strip():
            raise HbAdapterError(HbAdapterErrorCode.MANIFEST_INVALID, "integrity.verification.verified_at is required when status is verified")
        refresh_decision = str(refresh.get("decision") or "").strip().lower()
        next_check_due = refresh.get("next_check_due")
        manifest_id = str(manifest.get("manifest_id") or "").strip()
        manifest_sha256 = str(integrity.get("manifest_sha256") or "").strip()
        if not manifest_id or not manifest_sha256:
            raise HbAdapterError(HbAdapterErrorCode.MANIFEST_INVALID, "Manifest must include manifest_id and integrity.manifest_sha256")
        if not _SHA256_HEX_RE.fullmatch(manifest_sha256):
            raise HbAdapterError(HbAdapterErrorCode.MANIFEST_INVALID, "integrity.manifest_sha256 must be lowercase 64-hex")
        computed_manifest_sha256 = _manifest_digest(manifest)
        if computed_manifest_sha256 != manifest_sha256:
            raise HbAdapterError(
                HbAdapterErrorCode.INTEGRITY_MISMATCH,
                "Manifest SHA256 mismatch against canonical self-hash",
                rollback_required=True,
            )
        files = integrity.get("files")
        if not isinstance(files, list) or not files:
            raise HbAdapterError(HbAdapterErrorCode.MANIFEST_INVALID, "integrity.files must be a non-empty list")
        seen_paths: set[str] = set()
        for entry in files:
            if not isinstance(entry, dict):
                raise HbAdapterError(HbAdapterErrorCode.MANIFEST_INVALID, "integrity.files entries must be objects")
            relative_path = str(entry.get("relative_path") or "").strip()
            file_sha256 = str(entry.get("sha256") or "").strip()
            size_bytes = entry.get("size_bytes")
            if not _is_safe_relative_path(relative_path):
                raise HbAdapterError(HbAdapterErrorCode.ROOT_OUT_OF_BOUNDS, f"Unsafe integrity.files relative path: {relative_path!r}")
            folded = relative_path.replace("\\", "/").casefold()
            if folded in seen_paths:
                raise HbAdapterError(HbAdapterErrorCode.MANIFEST_INVALID, f"Duplicate integrity.files relative path after case folding: {relative_path}")
            seen_paths.add(folded)
            if any(relative_path.lower().endswith(suffix) for suffix in _BLOCKED_ARTIFACT_SUFFIXES):
                raise HbAdapterError(HbAdapterErrorCode.MANIFEST_INVALID, f"Partial artifact path not permitted in integrity.files: {relative_path}")
            if not _SHA256_HEX_RE.fullmatch(file_sha256):
                raise HbAdapterError(HbAdapterErrorCode.MANIFEST_INVALID, f"Invalid sha256 for integrity file entry: {relative_path}")
            file_path = (artifact_root / Path(relative_path)).resolve(strict=False)
            if not file_path.is_relative_to(artifact_root):
                raise HbAdapterError(HbAdapterErrorCode.ROOT_OUT_OF_BOUNDS, f"Integrity file path escapes artifact root: {relative_path}")
            if not file_path.exists() or not file_path.is_file():
                raise HbAdapterError(HbAdapterErrorCode.MANIFEST_MISSING, f"Missing integrity file: {file_path}")
            if isinstance(size_bytes, int) and size_bytes >= 0 and file_path.stat().st_size != size_bytes:
                raise HbAdapterError(HbAdapterErrorCode.INTEGRITY_MISMATCH, f"size_bytes mismatch for integrity file: {relative_path}", rollback_required=True)
            if _sha256_file(file_path) != file_sha256:
                raise HbAdapterError(HbAdapterErrorCode.INTEGRITY_MISMATCH, f"SHA256 mismatch for integrity file: {relative_path}", rollback_required=True)
        if scope.get("sex") != HB_SCOPE_SEX or scope.get("stage") != HB_SCOPE_STAGE or scope.get("anatomy") != HB_SCOPE_ANATOMY:
            raise HbAdapterError(HbAdapterErrorCode.OUT_OF_SCOPE, "Pinned manifest scope does not match phase-1 hb scope")
        if refresh_decision not in {"no_change", "patch_refresh", "major_bump", "rollback"}:
            raise HbAdapterError(HbAdapterErrorCode.MANIFEST_INVALID, f"Unexpected refresh.decision value: {refresh_decision or '<missing>'}")
        rollback_required = verification_status != "verified" or refresh_decision == "rollback"
        if verification_status != "verified":
            raise HbAdapterError(HbAdapterErrorCode.INTEGRITY_MISMATCH, f"Manifest verification status is {verification_status or '<missing>'}, expected verified", rollback_required=True)
        if not active_pointer_path.exists():
            raise HbAdapterError(HbAdapterErrorCode.PROMOTION_STATE_INVALID, f"Missing active promotion pointer: {active_pointer_path}", rollback_required=True)
        active_pointer = self._load_json(active_pointer_path)
        if not bool(active_pointer.get("active")):
            raise HbAdapterError(HbAdapterErrorCode.PROMOTION_STATE_INVALID, "Active promotion pointer is not active", rollback_required=True)
        if next_check_due:
            try:
                due_dt = datetime.fromisoformat(str(next_check_due).replace("Z", "+00:00"))
            except ValueError as exc:
                raise HbAdapterError(HbAdapterErrorCode.MANIFEST_INVALID, f"Invalid next_check_due timestamp: {next_check_due}") from exc
            if due_dt < datetime.now(timezone.utc):
                raise HbAdapterError(HbAdapterErrorCode.ROLLBACK_REQUIRED, "Manifest next_check_due has elapsed; revalidate or rollback before use", rollback_required=True)
        return HbManifestSummary(manifest_id=manifest_id, manifest_sha256=manifest_sha256, manifest_path=str(self._manifest_path()), graph_root=str(self._graph_root()), active_pointer_path=str(active_pointer_path), verification_status=verification_status, refresh_decision=refresh_decision, next_check_due=str(next_check_due) if next_check_due is not None else None, rollback_required=rollback_required)

    def capabilities(self) -> HbCapabilityManifest:
        layout = self._layout
        return HbCapabilityManifest(dataset_symbol=HB_DATASET_SYMBOL, version_id=HB_VERSION_ID, schema_version=HB_MANIFEST_SCHEMA_VERSION, source=HB_SOURCE, access_method=HB_ACCESS_METHOD, supported_operations=("capabilities", "resolve_snapshot", "validate_compatibility", "health", "execute"), supported_query_kinds=HB_ALLOWED_QUERY_KINDS, scope_tags=HbScopeTags(), manifest_root=str(layout.snapshots / HB_DATASET_SYMBOL / HB_VERSION_ID / "manifest"), graph_root=str(layout.graph / HB_DATASET_SYMBOL / HB_VERSION_ID))

    def resolve_snapshot(self) -> HbSnapshotHandle:
        summary = self._manifest_summary()
        return HbSnapshotHandle(root=self._root, graph_root=self._graph_root(), manifest_path=Path(summary.manifest_path), active_pointer_path=Path(summary.active_pointer_path), manifest_id=summary.manifest_id, manifest_sha256=summary.manifest_sha256, verification_status=summary.verification_status, refresh_decision=summary.refresh_decision, next_check_due=summary.next_check_due, rollback_required=summary.rollback_required)

    def health(self) -> HbHealthStatus:
        checks: list[HbCompatibilityCheck] = []
        snapshot: HbSnapshotHandle | None = None
        ok = True
        try:
            snapshot = self.resolve_snapshot()
            checks.append(HbCompatibilityCheck(name="snapshot_resolved", passed=True))
        except HbAdapterError as exc:
            ok = False
            checks.append(HbCompatibilityCheck(name="snapshot_resolved", passed=False, code=exc.code, detail=exc.message, rollback_required=exc.rollback_required))
        return HbHealthStatus(ok=ok, snapshot=snapshot, checks=checks, capabilities=self.capabilities())

    def validate_compatibility(self, request: HbQueryRequest) -> HbValidationReport:
        checks: list[HbCompatibilityCheck] = []
        snapshot: HbSnapshotHandle | None = None
        rollback_required = False
        if request.dataset_symbol != HB_DATASET_SYMBOL or request.version_id != HB_VERSION_ID:
            checks.append(HbCompatibilityCheck(name="dataset_pin", passed=False, code=HbAdapterErrorCode.DATASET_PIN_MISMATCH.value, detail=f"Expected {HB_DATASET_SYMBOL}/{HB_VERSION_ID}"))
            return HbValidationReport(ok=False, request=request, checks=checks, snapshot=None, rollback_required=False)
        checks.append(HbCompatibilityCheck(name="read_only", passed=bool(request.readonly), code=None if request.readonly else HbAdapterErrorCode.READ_ONLY_REQUIRED.value, detail=None if request.readonly else "hb adapter only supports read-only operations"))
        checks.append(HbCompatibilityCheck(name="remote_fallback", passed=not request.allow_remote_fallback, code=None if not request.allow_remote_fallback else HbAdapterErrorCode.OUT_OF_SCOPE.value, detail=None if not request.allow_remote_fallback else "local hb adapter does not permit remote fallback"))
        checks.append(HbCompatibilityCheck(name="query_kind", passed=request.query_kind in HB_ALLOWED_QUERY_KINDS, code=None if request.query_kind in HB_ALLOWED_QUERY_KINDS else HbAdapterErrorCode.UNSUPPORTED_QUERY_KIND.value, detail=None if request.query_kind in HB_ALLOWED_QUERY_KINDS else f"Unsupported hb query kind: {request.query_kind}"))
        scope = request.scope_tags
        checks.append(HbCompatibilityCheck(name="scope_tags", passed=scope.sex == HB_SCOPE_SEX and scope.stage == HB_SCOPE_STAGE and scope.anatomy == HB_SCOPE_ANATOMY, code=None if (scope.sex == HB_SCOPE_SEX and scope.stage == HB_SCOPE_STAGE and scope.anatomy == HB_SCOPE_ANATOMY) else HbAdapterErrorCode.SCOPE_VIOLATION.value, detail=None if (scope.sex == HB_SCOPE_SEX and scope.stage == HB_SCOPE_STAGE and scope.anatomy == HB_SCOPE_ANATOMY) else "hb scope must remain female/adult/hemibrain_region"))
        checks.append(HbCompatibilityCheck(name="evidence_family", passed=scope.evidence_family == HB_SCOPE_EVIDENCE_FAMILY, code=None if scope.evidence_family == HB_SCOPE_EVIDENCE_FAMILY else HbAdapterErrorCode.OUT_OF_SCOPE.value, detail=None if scope.evidence_family == HB_SCOPE_EVIDENCE_FAMILY else f"Expected evidence_family={HB_SCOPE_EVIDENCE_FAMILY}"))
        checks.append(HbCompatibilityCheck(name="access_layer", passed=scope.access_layer == HB_ACCESS_LAYER, code=None if scope.access_layer == HB_ACCESS_LAYER else HbAdapterErrorCode.OUT_OF_SCOPE.value, detail=None if scope.access_layer == HB_ACCESS_LAYER else f"Expected access_layer={HB_ACCESS_LAYER}"))
        try:
            snapshot = self.resolve_snapshot()
            checks.append(HbCompatibilityCheck(name="snapshot", passed=True))
            rollback_required = rollback_required or snapshot.rollback_required
        except HbAdapterError as exc:
            checks.append(HbCompatibilityCheck(name="snapshot", passed=False, code=exc.code, detail=exc.message, rollback_required=exc.rollback_required))
            rollback_required = rollback_required or exc.rollback_required
        payload = dict(request.query_payload)
        checks.append(HbCompatibilityCheck(name="payload.query", passed=bool(str(payload.get("query") or "").strip()), code=None if str(payload.get("query") or "").strip() else HbAdapterErrorCode.MANIFEST_INVALID.value, detail=None if str(payload.get("query") or "").strip() else "Missing required payload field: query"))
        checks.append(HbCompatibilityCheck(name="max_rows", passed=request.max_rows > 0, code=None if request.max_rows > 0 else HbAdapterErrorCode.QUERY_LIMIT_EXCEEDED.value, detail=None if request.max_rows > 0 else "max_rows must be positive"))
        checks.append(HbCompatibilityCheck(name="max_depth", passed=request.max_depth >= 0, code=None if request.max_depth >= 0 else HbAdapterErrorCode.QUERY_LIMIT_EXCEEDED.value, detail=None if request.max_depth >= 0 else "max_depth must be non-negative"))
        checks.append(HbCompatibilityCheck(name="max_weight", passed=request.max_weight is None or request.max_weight >= 0, code=None if request.max_weight is None or request.max_weight >= 0 else HbAdapterErrorCode.QUERY_LIMIT_EXCEEDED.value, detail=None if request.max_weight is None or request.max_weight >= 0 else "max_weight must be non-negative when provided"))
        ok = all(check.passed for check in checks)
        if not ok:
            rollback_required = rollback_required or any(check.rollback_required for check in checks)
        return HbValidationReport(ok=ok, request=request, checks=checks, snapshot=snapshot, rollback_required=rollback_required)

    def _provenance(self, request: HbQueryRequest, snapshot: HbSnapshotHandle) -> HbProvenanceEnvelope:
        dataset_scope = {"included_symbols": [HB_DATASET_SYMBOL], "version_ids_seen": [HB_VERSION_ID], "excluded_symbols": []}
        return HbProvenanceEnvelope(schema_version=HB_PROVENANCE_VERSION, dataset_symbol=request.dataset_symbol, version_id=request.version_id, source=HB_SOURCE, access_method=HB_ACCESS_METHOD, access_path=str(snapshot.graph_root), query_kind=request.query_kind, scope_tags=request.scope_tags.to_dict(), manifest_id=snapshot.manifest_id, manifest_sha256=snapshot.manifest_sha256, manifest_path=str(snapshot.manifest_path), graph_root=str(snapshot.graph_root), active_pointer_path=str(snapshot.active_pointer_path), replay_fingerprint=flybrain_replay_fingerprint("hb_local_adapter", request.to_dict(), dataset_scope), generated_at=self._now_iso())

    def execute(self, request: HbQueryRequest) -> HbQueryResult:
        report = self.validate_compatibility(request)
        if not report.ok or report.snapshot is None:
            first_failed = next((check for check in report.checks if not check.passed), None)
            code = first_failed.code if first_failed and first_failed.code else HbAdapterErrorCode.MANIFEST_INVALID.value
            message = first_failed.detail if first_failed and first_failed.detail else "hb adapter compatibility validation failed"
            snapshot = report.snapshot or HbSnapshotHandle(root=self._root, graph_root=self._graph_root(), manifest_path=self._manifest_path(), active_pointer_path=self._active_pointer_path(), manifest_id="", manifest_sha256="", verification_status="unavailable", refresh_decision="unavailable", next_check_due=None, rollback_required=True)
            raise HbAdapterError(code, message, checks=report.checks, request=request, snapshot=snapshot, rollback_required=report.rollback_required)
        snapshot = report.snapshot
        provenance = self._provenance(request, snapshot)
        if self._backend is None:
            raise HbAdapterError(HbAdapterErrorCode.UNIMPLEMENTED_BACKEND, "No local hb backend has been wired into the adapter yet", checks=report.checks, request=request, snapshot=snapshot, rollback_required=report.rollback_required)
        try:
            rows = list(self._backend(request, snapshot))
        except HbAdapterError:
            raise
        except Exception as exc:
            raise HbAdapterError(HbAdapterErrorCode.BACKEND_UNAVAILABLE, f"hb backend execution failed: {exc}", checks=report.checks, request=request, snapshot=snapshot, rollback_required=report.rollback_required) from exc
        if len(rows) > request.max_rows:
            raise HbAdapterError(HbAdapterErrorCode.QUERY_LIMIT_EXCEEDED, f"Query returned {len(rows)} rows which exceeds max_rows={request.max_rows}", checks=report.checks, request=request, snapshot=snapshot, rollback_required=report.rollback_required)
        columns = sorted({str(key) for row in rows for key in row.keys()})
        normalized_rows = [{str(key): row.get(key) for key in columns} for row in rows]
        return HbQueryResult(status="ok", count_status="exact", dataset_symbol=request.dataset_symbol, version_id=request.version_id, query_kind=request.query_kind, source=HB_SOURCE, row_count=len(normalized_rows), columns=columns, rows=normalized_rows, warnings=[], provenance=provenance, compatibility_checks=report.checks, rollback_required=report.rollback_required)


def build_hb_adapter(*, root_override: str | Path | None = None, env: Mapping[str, str] | None = None, backend: Callable[[HbQueryRequest, HbSnapshotHandle], Sequence[Mapping[str, Any]]] | None = None) -> LocalHbAdapter:
    return LocalHbAdapter(root_override=root_override, env=env, backend=backend)
