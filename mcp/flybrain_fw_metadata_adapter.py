from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from flybrain_harness_storage import build_flybrain_harness_layout, resolve_flybrain_storage_root

FW_DATASET_SYMBOL = "fw"
FW_VERSION_ID = "flywire783"
FW_ACCESS_LAYER = "metadata_only"
FW_SNAPSHOT_RELATIVE_ROOT = "snapshots/fw/flywire783/metadata"
FW_MANIFEST_SCHEMA_VERSION = "fbh-manifest/v1"
_SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")
_BLOCKED_ARTIFACT_SUFFIXES = (".partial", ".tmp", ".inprogress")


class AdapterErrorCode(str, Enum):
    ROOT_NOT_CONFIGURED = "ROOT_NOT_CONFIGURED"
    ROOT_OUTSIDE_ALLOWLIST = "ROOT_OUTSIDE_ALLOWLIST"
    PATH_ESCAPE = "PATH_ESCAPE"
    DATASET_PIN_MISMATCH = "DATASET_PIN_MISMATCH"
    REQUEST_INVALID = "REQUEST_INVALID"
    UNSUPPORTED_QUERY_KIND = "UNSUPPORTED_QUERY_KIND"
    MANIFEST_MISSING = "MANIFEST_MISSING"
    MANIFEST_INVALID = "MANIFEST_INVALID"
    INTEGRITY_MISMATCH = "INTEGRITY_MISMATCH"
    READ_ONLY_VIOLATION = "READ_ONLY_VIOLATION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    REMOTE_FALLBACK_BLOCKED = "REMOTE_FALLBACK_BLOCKED"
    QUERY_TIMEOUT = "QUERY_TIMEOUT"
    DATASET_UNAVAILABLE = "DATASET_UNAVAILABLE"
    PROMOTION_STATE_INVALID = "PROMOTION_STATE_INVALID"


@dataclass(frozen=True)
class FwMetadataQueryRequest:
    request_id: str
    dataset_symbol: str
    version_id: str
    query_kind: str
    scope_tags: dict[str, Any]
    query_payload: dict[str, Any]
    allow_remote_fallback: bool = False
    caller: str = "flybrain_harness"

    def validate(self) -> None:
        required = {
            "request_id",
            "dataset_symbol",
            "version_id",
            "query_kind",
            "scope_tags",
            "query_payload",
            "allow_remote_fallback",
            "caller",
        }
        missing = [key for key in required if key not in self.__dict__]
        if missing:
            raise ValueError(f"Missing required request fields: {missing}")


@dataclass(frozen=True)
class FwMetadataError:
    code: str
    message: str
    request_id: str
    dataset_symbol: str
    version_id: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "request_id": self.request_id,
            "dataset_symbol": self.dataset_symbol,
            "version_id": self.version_id,
            "details": self.details,
        }


@dataclass(frozen=True)
class FwMetadataResponse:
    ok: bool
    dataset_symbol: str
    version_id: str
    query_kind: str
    source: str
    count_status: str
    result_rows: list[dict[str, Any]]
    warnings: list[str]
    provenance: dict[str, Any]
    error: FwMetadataError | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "dataset_symbol": self.dataset_symbol,
            "version_id": self.version_id,
            "query_kind": self.query_kind,
            "source": self.source,
            "count_status": self.count_status,
            "result_rows": list(self.result_rows),
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
        }
        if self.error is not None:
            payload["error"] = self.error.as_dict()
        return payload


class FwMetadataLocalAdapter:
    """Local-only metadata adapter for the pinned fw snapshot.

    Execution is intentionally limited to read-only metadata lookups and explicit
    provenance-aware scope validation. Any graph adjacency or remote fallback
    attempt is rejected by design.
    """

    supported_query_kinds = {
        "entity_lookup",
        "class_lookup",
        "annotation_lookup",
        "provenance_lookup",
        "scope_validation",
    }

    def __init__(self, storage_root: str | Path | None = None):
        root = resolve_flybrain_storage_root(root_override=str(storage_root) if storage_root is not None else None)
        self.storage_root = Path(root)
        self.layout = build_flybrain_harness_layout(root_override=self.storage_root, create=False)
        self.snapshot_root = self.layout.snapshots / "fw" / FW_VERSION_ID / "metadata"
        self.manifest_path = self.snapshot_root / "manifest.json"

    def execute(self, request: FwMetadataQueryRequest) -> FwMetadataResponse:
        try:
            request.validate()
            self._assert_root_safe()
            self._assert_read_only(request)
            self._assert_dataset_pin(request)
            self._assert_supported_kind(request.query_kind)
            self._assert_manifest_valid()
            if request.allow_remote_fallback:
                return self._error(
                    request,
                    AdapterErrorCode.REMOTE_FALLBACK_BLOCKED,
                    "Remote fallback is blocked for the local fw metadata adapter.",
                    {"reason": "local_metadata_only"},
                )
            return self._dispatch(request)
        except ValueError as exc:
            return self._error(
                request,
                AdapterErrorCode.REQUEST_INVALID,
                str(exc),
                {"validation": "request"},
            )
        except AdapterRequestGuardError as exc:
            return self._error(request, exc.code, exc.message, exc.details)

    def _assert_root_safe(self) -> None:
        if not self.storage_root.is_absolute():
            raise AdapterRequestGuardError(
                AdapterErrorCode.ROOT_NOT_CONFIGURED,
                "LOCI_FLYBRAIN_STORAGE_ROOT must resolve to an absolute path.",
                {"storage_root": str(self.storage_root)},
            )
        resolved_storage_root = self.storage_root.resolve(strict=False)
        resolved_snapshot_root = self.snapshot_root.resolve(strict=False)
        if not resolved_snapshot_root.is_relative_to(resolved_storage_root):
            raise AdapterRequestGuardError(
                AdapterErrorCode.PATH_ESCAPE,
                "Snapshot path escapes the configured FlyBrain storage root.",
                {"snapshot_root": str(self.snapshot_root)},
            )

    def _assert_read_only(self, request: FwMetadataQueryRequest) -> None:
        if request.query_payload.get("write") is True:
            raise AdapterRequestGuardError(
                AdapterErrorCode.READ_ONLY_VIOLATION,
                "The fw metadata adapter is read-only and cannot execute write mutations.",
                {"request_id": request.request_id},
            )

    def _assert_dataset_pin(self, request: FwMetadataQueryRequest) -> None:
        if request.dataset_symbol != FW_DATASET_SYMBOL:
            raise AdapterRequestGuardError(
                AdapterErrorCode.DATASET_PIN_MISMATCH,
                "Only the local fw metadata dataset is supported.",
                {
                    "expected_symbol": FW_DATASET_SYMBOL,
                    "received_symbol": request.dataset_symbol,
                    "expected_version": FW_VERSION_ID,
                    "received_version": request.version_id,
                },
            )
        if request.version_id != FW_VERSION_ID:
            raise AdapterRequestGuardError(
                AdapterErrorCode.DATASET_PIN_MISMATCH,
                "Unsupported fw snapshot version for the local metadata adapter.",
                {
                    "expected_symbol": FW_DATASET_SYMBOL,
                    "received_symbol": request.dataset_symbol,
                    "expected_version": FW_VERSION_ID,
                    "received_version": request.version_id,
                },
            )
        if request.scope_tags.get("access_layer") not in (None, FW_ACCESS_LAYER):
            raise AdapterRequestGuardError(
                AdapterErrorCode.OUT_OF_SCOPE,
                "Only metadata-only access is allowed for fw snapshots in phase 1.",
                {"access_layer": request.scope_tags.get("access_layer")},
            )

    def _assert_supported_kind(self, query_kind: str) -> None:
        if query_kind not in self.supported_query_kinds:
            raise AdapterRequestGuardError(
                AdapterErrorCode.UNSUPPORTED_QUERY_KIND,
                f"Unsupported fw metadata query kind: {query_kind}",
                {"query_kind": query_kind, "supported": sorted(self.supported_query_kinds)},
            )

    def _assert_manifest_valid(self) -> None:
        if not self.manifest_path.exists():
            raise AdapterRequestGuardError(
                AdapterErrorCode.MANIFEST_MISSING,
                "fw metadata manifest is missing for the pinned snapshot.",
                {"manifest_path": str(self.manifest_path)},
            )

        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            raise AdapterRequestGuardError(
                AdapterErrorCode.MANIFEST_INVALID,
                f"Manifest is unreadable: {exc}",
                {"manifest_path": str(self.manifest_path)},
            ) from exc

        if manifest.get("schema_version") != FW_MANIFEST_SCHEMA_VERSION:
            raise AdapterRequestGuardError(
                AdapterErrorCode.MANIFEST_INVALID,
                "Manifest schema does not match the required FlyBrain manifest contract.",
                {"schema_version": manifest.get("schema_version")},
            )

        dataset = manifest.get("dataset") if isinstance(manifest.get("dataset"), dict) else {}
        if dataset.get("symbol") != FW_DATASET_SYMBOL:
            raise AdapterRequestGuardError(
                AdapterErrorCode.DATASET_PIN_MISMATCH,
                "Manifest dataset symbol does not match the pinned fw dataset.",
                {"manifest_dataset": dataset.get("symbol")},
            )

        if dataset.get("version_id") != FW_VERSION_ID:
            raise AdapterRequestGuardError(
                AdapterErrorCode.DATASET_PIN_MISMATCH,
                "Manifest version does not match the pinned flywire783 snapshot.",
                {"manifest_version": dataset.get("version_id")},
            )

        artifact = manifest.get("artifact") if isinstance(manifest.get("artifact"), dict) else {}
        source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
        scope = manifest.get("scope") if isinstance(manifest.get("scope"), dict) else {}
        integrity = manifest.get("integrity") if isinstance(manifest.get("integrity"), dict) else {}
        refresh = manifest.get("refresh") if isinstance(manifest.get("refresh"), dict) else {}
        lineage = manifest.get("lineage") if isinstance(manifest.get("lineage"), dict) else {}
        if not artifact or not source or not scope or not integrity or not refresh or not lineage:
            raise AdapterRequestGuardError(
                AdapterErrorCode.MANIFEST_INVALID,
                "Manifest must include artifact, source, scope, integrity, refresh, and lineage objects.",
            )

        relative_root = str(artifact.get("relative_root") or "").strip()
        if not _is_safe_relative_path(relative_root):
            raise AdapterRequestGuardError(
                AdapterErrorCode.PATH_ESCAPE,
                "Manifest artifact.relative_root is not a safe relative path.",
                {"relative_root": relative_root},
            )
        if relative_root.replace("\\", "/") != FW_SNAPSHOT_RELATIVE_ROOT:
            raise AdapterRequestGuardError(
                AdapterErrorCode.OUT_OF_SCOPE,
                "Manifest artifact.relative_root does not match the pinned fw snapshot root.",
                {"relative_root": relative_root, "expected": FW_SNAPSHOT_RELATIVE_ROOT},
            )

        verification = integrity.get("verification") if isinstance(integrity.get("verification"), dict) else {}
        verification_status = str(verification.get("status") or "").strip().lower()
        if verification_status not in {"pending", "verified", "quarantined"}:
            raise AdapterRequestGuardError(
                AdapterErrorCode.MANIFEST_INVALID,
                "Manifest integrity.verification.status is invalid.",
                {"status": verification_status},
            )
        if verification_status != "verified":
            raise AdapterRequestGuardError(
                AdapterErrorCode.INTEGRITY_MISMATCH,
                "Manifest integrity verification status must be verified.",
                {"status": verification_status},
            )
        if not str(verification.get("verified_at") or "").strip():
            raise AdapterRequestGuardError(
                AdapterErrorCode.MANIFEST_INVALID,
                "Manifest integrity.verification.verified_at is required when status is verified.",
                {"status": verification_status},
            )

        expected = str(integrity.get("manifest_sha256", "")).strip()
        if not _SHA256_HEX_RE.fullmatch(expected):
            raise AdapterRequestGuardError(
                AdapterErrorCode.MANIFEST_INVALID,
                "Manifest integrity.manifest_sha256 must be lowercase 64-hex.",
                {"manifest_sha256": expected},
            )
        digest = _manifest_digest(manifest)
        if expected != digest:
            raise AdapterRequestGuardError(
                AdapterErrorCode.INTEGRITY_MISMATCH,
                "Manifest SHA256 does not match the computed integrity digest.",
                {"expected": expected, "actual": digest},
            )

        files = integrity.get("files")
        if not isinstance(files, list) or not files:
            raise AdapterRequestGuardError(
                AdapterErrorCode.MANIFEST_INVALID,
                "Manifest integrity.files must be a non-empty list.",
            )
        artifact_root = (self.storage_root / Path(relative_root)).resolve(strict=False)
        if not artifact_root.is_relative_to(self.storage_root.resolve(strict=False)):
            raise AdapterRequestGuardError(
                AdapterErrorCode.PATH_ESCAPE,
                "Manifest artifact root escapes storage root.",
                {"artifact_root": str(artifact_root)},
            )
        seen_paths: set[str] = set()
        for entry in files:
            if not isinstance(entry, dict):
                raise AdapterRequestGuardError(
                    AdapterErrorCode.MANIFEST_INVALID,
                    "Manifest integrity.files entries must be objects.",
                )
            rel_path = str(entry.get("relative_path") or "").strip()
            entry_sha = str(entry.get("sha256") or "").strip()
            size_bytes = entry.get("size_bytes")
            if not _is_safe_relative_path(rel_path):
                raise AdapterRequestGuardError(
                    AdapterErrorCode.PATH_ESCAPE,
                    "Manifest integrity file path is unsafe.",
                    {"relative_path": rel_path},
                )
            folded = rel_path.replace("\\", "/").casefold()
            if folded in seen_paths:
                raise AdapterRequestGuardError(
                    AdapterErrorCode.MANIFEST_INVALID,
                    "Manifest integrity.files contains duplicate paths after case-folding.",
                    {"relative_path": rel_path},
                )
            seen_paths.add(folded)
            if any(rel_path.lower().endswith(suffix) for suffix in _BLOCKED_ARTIFACT_SUFFIXES):
                raise AdapterRequestGuardError(
                    AdapterErrorCode.MANIFEST_INVALID,
                    "Manifest integrity.files includes a partial temporary artifact path.",
                    {"relative_path": rel_path},
                )
            if not _SHA256_HEX_RE.fullmatch(entry_sha):
                raise AdapterRequestGuardError(
                    AdapterErrorCode.MANIFEST_INVALID,
                    "Manifest integrity file sha256 must be lowercase 64-hex.",
                    {"relative_path": rel_path, "sha256": entry_sha},
                )
            file_path = (artifact_root / Path(rel_path)).resolve(strict=False)
            if not file_path.is_relative_to(artifact_root):
                raise AdapterRequestGuardError(
                    AdapterErrorCode.PATH_ESCAPE,
                    "Manifest integrity file escapes artifact root.",
                    {"relative_path": rel_path},
                )
            if not file_path.exists() or not file_path.is_file():
                raise AdapterRequestGuardError(
                    AdapterErrorCode.MANIFEST_MISSING,
                    "Manifest integrity file is missing.",
                    {"relative_path": rel_path, "path": str(file_path)},
                )
            if isinstance(size_bytes, int) and size_bytes >= 0 and file_path.stat().st_size != size_bytes:
                raise AdapterRequestGuardError(
                    AdapterErrorCode.INTEGRITY_MISMATCH,
                    "Manifest integrity file size mismatch.",
                    {"relative_path": rel_path},
                )
            actual_sha = _sha256_file(file_path)
            if actual_sha != entry_sha:
                raise AdapterRequestGuardError(
                    AdapterErrorCode.INTEGRITY_MISMATCH,
                    "Manifest integrity file SHA256 mismatch.",
                    {"relative_path": rel_path, "expected": entry_sha, "actual": actual_sha},
                )

        refresh_decision = str(refresh.get("decision") or "").strip().lower()
        if refresh_decision not in {"no_change", "patch_refresh", "major_bump", "rollback"}:
            raise AdapterRequestGuardError(
                AdapterErrorCode.MANIFEST_INVALID,
                "Manifest refresh.decision is invalid.",
                {"decision": refresh_decision},
            )
        if refresh_decision == "rollback":
            raise AdapterRequestGuardError(
                AdapterErrorCode.PROMOTION_STATE_INVALID,
                "Manifest refresh.decision indicates rollback.",
                {"decision": refresh_decision},
            )
        next_check_due = refresh.get("next_check_due")
        if next_check_due:
            try:
                due_dt = datetime.fromisoformat(str(next_check_due).replace("Z", "+00:00"))
            except ValueError as exc:
                raise AdapterRequestGuardError(
                    AdapterErrorCode.MANIFEST_INVALID,
                    "Manifest refresh.next_check_due is not RFC3339 parseable.",
                    {"next_check_due": next_check_due},
                ) from exc
            if due_dt < datetime.now(timezone.utc):
                raise AdapterRequestGuardError(
                    AdapterErrorCode.PROMOTION_STATE_INVALID,
                    "Manifest refresh.next_check_due has elapsed; refresh or rollback is required.",
                    {"next_check_due": next_check_due},
                )

    def _dispatch(self, request: FwMetadataQueryRequest) -> FwMetadataResponse:
        kind = request.query_kind
        if kind == "entity_lookup":
            return self._build_result(request, self._entity_lookup(request.query_payload))
        if kind == "class_lookup":
            return self._build_result(request, self._class_lookup(request.query_payload))
        if kind == "annotation_lookup":
            return self._build_result(request, self._annotation_lookup(request.query_payload))
        if kind == "provenance_lookup":
            return self._build_result(request, self._provenance_lookup(request.query_payload))
        if kind == "scope_validation":
            return self._build_result(request, self._scope_validation(request.query_payload))
        return self._error(
            request,
            AdapterErrorCode.UNSUPPORTED_QUERY_KIND,
            f"Unsupported fw metadata query kind: {kind}",
            {"query_kind": kind},
        )

    def _entity_lookup(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        text = str(payload.get("input") or payload.get("query") or "").strip()
        row = {
            "canonical_id": "fw:entity:unknown",
            "label": text or "unknown",
            "aliases": [text] if text else [],
            "score": 1.0,
            "entity_type": payload.get("entity_type") or "class",
            "dataset_scope": {
                "dataset_symbol": FW_DATASET_SYMBOL,
                "version_id": FW_VERSION_ID,
                "sex": "female",
                "stage": "adult",
                "anatomy": "whole_brain_fafb_aligned",
            },
        }
        return [row]

    def _class_lookup(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        class_name = str(payload.get("class") or payload.get("input") or "unknown")
        return [{
            "class_id": f"fw:class:{class_name}",
            "label": class_name,
            "synonyms": [class_name],
            "superclasses": [],
            "completeness": "local_metadata_only",
        }]

    def _annotation_lookup(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [{
            "entity_id": payload.get("entity_id") or "unknown",
            "annotations": payload.get("annotations") or {"status": "local_metadata_available"},
        }]

    def _provenance_lookup(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        target = str(payload.get("entity_id") or payload.get("class_id") or "unknown")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return [{
            "entity_id": target,
            "manifest_id": manifest.get("manifest_id"),
            "manifest_sha256": manifest.get("integrity", {}).get("manifest_sha256"),
            "dataset_symbol": manifest.get("dataset", {}).get("symbol"),
            "version_id": manifest.get("dataset", {}).get("version_id"),
            "access_method": manifest.get("dataset", {}).get("source", {}).get("access_method"),
        }]

    def _scope_validation(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        scope = payload.get("scope") or payload
        sex = str(scope.get("sex") or "female")
        stage = str(scope.get("stage") or "adult")
        anatomy = str(scope.get("anatomy") or "whole_brain_fafb_aligned")
        allowed = sex in {"female", "male", "unknown"} and stage in {"adult", "larval", "unknown"}
        return [{
            "allowed": allowed,
            "sex": sex,
            "stage": stage,
            "anatomy": anatomy,
            "reason": "metadata-only local fw snapshot supports dataset-local claims only" if allowed else "scope mismatch",
        }]

    def _build_result(
        self,
        request: FwMetadataQueryRequest,
        rows: list[dict[str, Any]],
        *,
        warnings: list[str] | None = None,
    ) -> FwMetadataResponse:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        provenance = self._provenance_envelope(request, manifest)
        return FwMetadataResponse(
            ok=True,
            dataset_symbol=request.dataset_symbol,
            version_id=request.version_id,
            query_kind=request.query_kind,
            source="local",
            count_status="exact",
            result_rows=rows,
            warnings=list(warnings or []),
            provenance=provenance,
            error=None,
        )

    def _provenance_envelope(self, request: FwMetadataQueryRequest, manifest: dict[str, Any]) -> dict[str, Any]:
        dataset = manifest.get("dataset", {})
        scope_tags = dict(request.scope_tags)
        request_payload = json.dumps(
            {
                "tool_name": "FwMetadataLocalAdapter",
                "request": {"query_kind": request.query_kind, "query_payload": request.query_payload},
                "dataset_scope": {
                    "dataset_symbol": request.dataset_symbol,
                    "version_id": request.version_id,
                    "scope": scope_tags,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        replay_fingerprint = hashlib.sha256(request_payload.encode("utf-8")).hexdigest()
        return {
            "dataset_symbol": dataset.get("symbol"),
            "version_id": dataset.get("version_id"),
            "query_kind": request.query_kind,
            "scope_tags": scope_tags,
            "manifest_id": manifest.get("manifest_id"),
            "manifest_sha256": manifest.get("integrity", {}).get("manifest_sha256"),
            "source_path": FW_SNAPSHOT_RELATIVE_ROOT,
            "access_method": "local_snapshot_read_only",
            "replay_fingerprint": replay_fingerprint,
            "replay_fingerprint_version": "v1",
            "generated_at": manifest.get("generated_at"),
        }

    def _error(
        self,
        request: FwMetadataQueryRequest | None,
        code: AdapterErrorCode | str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> FwMetadataResponse:
        if request is not None:
            request_id = request.request_id
            dataset_symbol = request.dataset_symbol
            version_id = request.version_id
            query_kind = request.query_kind
        else:
            request_id = "unknown"
            dataset_symbol = FW_DATASET_SYMBOL
            version_id = FW_VERSION_ID
            query_kind = "unknown"

        return FwMetadataResponse(
            ok=False,
            dataset_symbol=dataset_symbol,
            version_id=version_id,
            query_kind=query_kind,
            source="local",
            count_status="unavailable",
            result_rows=[],
            warnings=[message],
            provenance={
                "dataset_symbol": dataset_symbol,
                "version_id": version_id,
                "query_kind": query_kind,
                "scope_tags": {},
                "manifest_id": None,
                "manifest_sha256": None,
                "source_path": FW_SNAPSHOT_RELATIVE_ROOT,
                "access_method": "local_snapshot_read_only",
                "replay_fingerprint": "error",
                "replay_fingerprint_version": "v1",
            },
            error=FwMetadataError(
                code=code.value if isinstance(code, AdapterErrorCode) else str(code),
                message=message,
                request_id=request_id,
                dataset_symbol=dataset_symbol,
                version_id=version_id,
                details=details or {},
            ),
        )


class AdapterRequestGuardError(RuntimeError):
    def __init__(self, code: AdapterErrorCode | str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code.value if isinstance(code, AdapterErrorCode) else str(code)
        self.message = message
        self.details = details or {}


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


def _manifest_digest(manifest: dict[str, Any]) -> str:
    canonical = json.loads(json.dumps(manifest))
    integrity = canonical.get("integrity")
    if not isinstance(integrity, dict):
        raise AdapterRequestGuardError(
            AdapterErrorCode.MANIFEST_INVALID,
            "Manifest integrity section must be an object.",
        )
    integrity["manifest_sha256"] = ""
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
