from __future__ import annotations

"""Brain-cluster coordinator + fail-closed expert/router artifact loader.

Artifact manifest contract:
  - schema_version == braincluster-artifact-manifest/v1
  - artifact.id / artifact.version / artifact.relative_root
  - integrity.files[] entries with logical_name, relative_path, sha256, size_bytes
  - integrity.manifest_sha256 canonical self-hash
"""

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from uuid import uuid4

from replay_fingerprint import flybrain_replay_fingerprint

CLUSTER_TOOL_NAME = "flybrain_brain_cluster"
BRAIN_CLUSTER_ARTIFACT_SCHEMA_VERSION = "braincluster-artifact-manifest/v1"
BRAIN_CLUSTER_PROMOTION_STATE_SCHEMA_VERSION = "braincluster-promotion-state/v1"
_SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")
_BLOCKED_ARTIFACT_SUFFIXES = (".partial", ".tmp", ".inprogress")


class BrainClusterArtifactErrorCode(str, Enum):
    MANIFEST_MISSING = "MANIFEST_MISSING"
    MANIFEST_INVALID = "MANIFEST_INVALID"
    PATH_ESCAPE = "PATH_ESCAPE"
    ARTIFACT_MISSING = "ARTIFACT_MISSING"
    ARTIFACT_INVALID = "ARTIFACT_INVALID"
    INTEGRITY_MISMATCH = "INTEGRITY_MISMATCH"


class BrainClusterArtifactError(RuntimeError):
    def __init__(
        self,
        code: BrainClusterArtifactErrorCode | str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = code.value if isinstance(code, BrainClusterArtifactErrorCode) else str(code)
        self.message = message
        self.details = dict(details or {})
        super().__init__(f"{self.code}: {self.message}")


class BrainClusterPromotionStateErrorCode(str, Enum):
    STATE_INVALID = "STATE_INVALID"
    STATE_IO_ERROR = "STATE_IO_ERROR"
    MANIFEST_MISSING = "MANIFEST_MISSING"
    CANDIDATE_MISSING = "CANDIDATE_MISSING"
    PROMOTED_MISSING = "PROMOTED_MISSING"
    ROLLBACK_UNAVAILABLE = "ROLLBACK_UNAVAILABLE"
    INVALID_TRANSITION = "INVALID_TRANSITION"
    VALIDATION_FAILED = "VALIDATION_FAILED"


class BrainClusterPromotionStateError(RuntimeError):
    def __init__(
        self,
        code: BrainClusterPromotionStateErrorCode | str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = code.value if isinstance(code, BrainClusterPromotionStateErrorCode) else str(code)
        self.message = message
        self.details = dict(details or {})
        super().__init__(f"{self.code}: {self.message}")


@dataclass(frozen=True)
class BrainClusterPromotionPointer:
    manifest_path: str
    manifest_sha256: str
    artifact_id: str
    artifact_version: str
    recorded_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "artifact_id": self.artifact_id,
            "artifact_version": self.artifact_version,
            "recorded_at": self.recorded_at,
        }


@dataclass(frozen=True)
class BrainClusterPromotionState:
    schema_version: str
    updated_at: str | None
    candidate: BrainClusterPromotionPointer | None = None
    promoted: BrainClusterPromotionPointer | None = None
    previous_promoted: BrainClusterPromotionPointer | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "updated_at": self.updated_at,
            "candidate": self.candidate.as_dict() if self.candidate else None,
            "promoted": self.promoted.as_dict() if self.promoted else None,
            "previous_promoted": self.previous_promoted.as_dict() if self.previous_promoted else None,
        }


@dataclass(frozen=True)
class BrainClusterArtifactManifest:
    schema_version: str
    artifact_id: str
    artifact_version: str
    artifact_relative_root: str
    manifest_path: str
    manifest_sha256: str
    files: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "artifact_version": self.artifact_version,
            "artifact_relative_root": self.artifact_relative_root,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "files": [dict(item) for item in self.files],
        }


@dataclass(frozen=True)
class BrainClusterArtifactBundle:
    manifest: BrainClusterArtifactManifest
    router: Mapping[str, Any]
    experts: Any


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
        raise BrainClusterArtifactError(
            BrainClusterArtifactErrorCode.MANIFEST_INVALID,
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


def _load_json_object(path: Path, *, missing_code: BrainClusterArtifactErrorCode) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise BrainClusterArtifactError(
            missing_code,
            "Required JSON file is missing.",
            details={"path": str(path)},
        )
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BrainClusterArtifactError(
            BrainClusterArtifactErrorCode.ARTIFACT_INVALID,
            f"Invalid JSON at {path}: {exc.msg}",
            details={"path": str(path)},
        ) from exc
    if not isinstance(loaded, dict):
        raise BrainClusterArtifactError(
            BrainClusterArtifactErrorCode.ARTIFACT_INVALID,
            "Expected JSON object artifact payload.",
            details={"path": str(path)},
        )
    return loaded


def _validate_brain_cluster_manifest(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
) -> tuple[BrainClusterArtifactManifest, dict[str, Path]]:
    if manifest.get("schema_version") != BRAIN_CLUSTER_ARTIFACT_SCHEMA_VERSION:
        raise BrainClusterArtifactError(
            BrainClusterArtifactErrorCode.MANIFEST_INVALID,
            "Manifest schema_version does not match the brain-cluster artifact contract.",
            details={"schema_version": manifest.get("schema_version")},
        )

    artifact = manifest.get("artifact") if isinstance(manifest.get("artifact"), dict) else {}
    integrity = manifest.get("integrity") if isinstance(manifest.get("integrity"), dict) else {}
    artifact_id = str(artifact.get("id") or "").strip()
    artifact_version = str(artifact.get("version") or "").strip()
    relative_root = str(artifact.get("relative_root") or "").strip()
    if not artifact_id or not artifact_version:
        raise BrainClusterArtifactError(
            BrainClusterArtifactErrorCode.MANIFEST_INVALID,
            "Manifest artifact.id and artifact.version are required.",
        )
    if not _is_safe_relative_path(relative_root):
        raise BrainClusterArtifactError(
            BrainClusterArtifactErrorCode.PATH_ESCAPE,
            "Manifest artifact.relative_root is not a safe relative path.",
            details={"relative_root": relative_root},
        )

    manifest_root = manifest_path.parent.resolve(strict=False)
    artifact_root = (manifest_root / Path(relative_root)).resolve(strict=False)
    if not artifact_root.is_relative_to(manifest_root):
        raise BrainClusterArtifactError(
            BrainClusterArtifactErrorCode.PATH_ESCAPE,
            "Manifest artifact root escapes manifest directory.",
            details={"artifact_root": str(artifact_root)},
        )

    expected_manifest_sha256 = str(integrity.get("manifest_sha256") or "").strip()
    if not _SHA256_HEX_RE.fullmatch(expected_manifest_sha256):
        raise BrainClusterArtifactError(
            BrainClusterArtifactErrorCode.MANIFEST_INVALID,
            "Manifest integrity.manifest_sha256 must be lowercase 64-hex.",
            details={"manifest_sha256": expected_manifest_sha256},
        )
    actual_manifest_sha256 = _manifest_digest(manifest)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise BrainClusterArtifactError(
            BrainClusterArtifactErrorCode.INTEGRITY_MISMATCH,
            "Manifest SHA256 does not match canonical self-hash.",
            details={"expected": expected_manifest_sha256, "actual": actual_manifest_sha256},
        )

    files = integrity.get("files")
    if not isinstance(files, list) or not files:
        raise BrainClusterArtifactError(
            BrainClusterArtifactErrorCode.MANIFEST_INVALID,
            "Manifest integrity.files must be a non-empty list.",
        )

    seen_paths: set[str] = set()
    seen_logical_names: set[str] = set()
    resolved_files: dict[str, Path] = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise BrainClusterArtifactError(
                BrainClusterArtifactErrorCode.MANIFEST_INVALID,
                "Manifest integrity.files entries must be objects.",
            )
        logical_name = str(entry.get("logical_name") or "").strip()
        relative_path = str(entry.get("relative_path") or "").strip()
        sha256_value = str(entry.get("sha256") or "").strip()
        size_bytes = entry.get("size_bytes")
        if not logical_name:
            raise BrainClusterArtifactError(
                BrainClusterArtifactErrorCode.MANIFEST_INVALID,
                "Manifest integrity file entry logical_name is required.",
            )
        if not relative_path:
            raise BrainClusterArtifactError(
                BrainClusterArtifactErrorCode.MANIFEST_INVALID,
                "Manifest integrity file entry relative_path is required.",
            )
        folded_name = logical_name.casefold()
        if folded_name in seen_logical_names:
            raise BrainClusterArtifactError(
                BrainClusterArtifactErrorCode.MANIFEST_INVALID,
                "Manifest integrity.files contains duplicate logical_name entries.",
                details={"logical_name": logical_name},
            )
        seen_logical_names.add(folded_name)
        if not _is_safe_relative_path(relative_path):
            raise BrainClusterArtifactError(
                BrainClusterArtifactErrorCode.PATH_ESCAPE,
                "Manifest integrity file path is unsafe.",
                details={"logical_name": logical_name, "relative_path": relative_path},
            )
        folded_path = relative_path.replace("\\", "/").casefold()
        if folded_path in seen_paths:
            raise BrainClusterArtifactError(
                BrainClusterArtifactErrorCode.MANIFEST_INVALID,
                "Manifest integrity.files contains duplicate paths after case-folding.",
                details={"relative_path": relative_path},
            )
        seen_paths.add(folded_path)
        if any(relative_path.lower().endswith(suffix) for suffix in _BLOCKED_ARTIFACT_SUFFIXES):
            raise BrainClusterArtifactError(
                BrainClusterArtifactErrorCode.MANIFEST_INVALID,
                "Manifest integrity.files includes a temporary artifact suffix.",
                details={"relative_path": relative_path},
            )
        if not _SHA256_HEX_RE.fullmatch(sha256_value):
            raise BrainClusterArtifactError(
                BrainClusterArtifactErrorCode.MANIFEST_INVALID,
                "Manifest integrity file sha256 must be lowercase 64-hex.",
                details={"logical_name": logical_name, "sha256": sha256_value},
            )
        if not isinstance(size_bytes, int) or size_bytes < 0:
            raise BrainClusterArtifactError(
                BrainClusterArtifactErrorCode.MANIFEST_INVALID,
                "Manifest integrity file size_bytes must be a non-negative integer.",
                details={"logical_name": logical_name, "size_bytes": size_bytes},
            )
        artifact_path = (artifact_root / Path(relative_path)).resolve(strict=False)
        if not artifact_path.is_relative_to(artifact_root):
            raise BrainClusterArtifactError(
                BrainClusterArtifactErrorCode.PATH_ESCAPE,
                "Manifest integrity file escapes artifact root.",
                details={"logical_name": logical_name, "relative_path": relative_path},
            )
        if not artifact_path.exists() or not artifact_path.is_file():
            raise BrainClusterArtifactError(
                BrainClusterArtifactErrorCode.ARTIFACT_MISSING,
                "Manifest integrity file is missing.",
                details={"logical_name": logical_name, "path": str(artifact_path)},
            )
        if artifact_path.stat().st_size != size_bytes:
            raise BrainClusterArtifactError(
                BrainClusterArtifactErrorCode.INTEGRITY_MISMATCH,
                "Manifest integrity file size mismatch.",
                details={"logical_name": logical_name, "relative_path": relative_path},
            )
        file_sha = _sha256_file(artifact_path)
        if file_sha != sha256_value:
            raise BrainClusterArtifactError(
                BrainClusterArtifactErrorCode.INTEGRITY_MISMATCH,
                "Manifest integrity file SHA256 mismatch.",
                details={"logical_name": logical_name, "relative_path": relative_path, "expected": sha256_value, "actual": file_sha},
            )
        resolved_files[folded_name] = artifact_path

    required = {"router", "experts"}
    missing_required = sorted(name for name in required if name not in resolved_files)
    if missing_required:
        raise BrainClusterArtifactError(
            BrainClusterArtifactErrorCode.MANIFEST_INVALID,
            "Manifest integrity.files must include router and experts artifacts.",
            details={"missing_logical_names": missing_required},
        )

    manifest_summary = BrainClusterArtifactManifest(
        schema_version=str(manifest.get("schema_version") or ""),
        artifact_id=artifact_id,
        artifact_version=artifact_version,
        artifact_relative_root=relative_root,
        manifest_path=str(manifest_path),
        manifest_sha256=expected_manifest_sha256,
        files=tuple(dict(item) for item in files),
    )
    return manifest_summary, resolved_files


def load_brain_cluster_artifacts(manifest_path: str | Path) -> BrainClusterArtifactBundle:
    manifest_file = Path(manifest_path)
    if not manifest_file.exists() or not manifest_file.is_file():
        raise BrainClusterArtifactError(
            BrainClusterArtifactErrorCode.MANIFEST_MISSING,
            "Brain-cluster artifact manifest is missing.",
            details={"manifest_path": str(manifest_file)},
        )
    try:
        manifest_payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BrainClusterArtifactError(
            BrainClusterArtifactErrorCode.MANIFEST_INVALID,
            f"Manifest is not valid JSON: {exc.msg}",
            details={"manifest_path": str(manifest_file)},
        ) from exc
    if not isinstance(manifest_payload, dict):
        raise BrainClusterArtifactError(
            BrainClusterArtifactErrorCode.MANIFEST_INVALID,
            "Manifest must be a JSON object.",
            details={"manifest_path": str(manifest_file)},
        )
    manifest, resolved_files = _validate_brain_cluster_manifest(manifest_payload, manifest_path=manifest_file)
    router = _load_json_object(
        resolved_files["router"],
        missing_code=BrainClusterArtifactErrorCode.ARTIFACT_MISSING,
    )
    experts = _load_json_object(
        resolved_files["experts"],
        missing_code=BrainClusterArtifactErrorCode.ARTIFACT_MISSING,
    )
    return BrainClusterArtifactBundle(manifest=manifest, router=router, experts=experts)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _state_pointer_from_payload(payload: Any, *, field_name: str) -> BrainClusterPromotionPointer | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise BrainClusterPromotionStateError(
            BrainClusterPromotionStateErrorCode.STATE_INVALID,
            "Promotion-state pointer entry must be a JSON object or null.",
            details={"field": field_name},
        )
    manifest_path = str(payload.get("manifest_path") or "").strip()
    manifest_sha256 = str(payload.get("manifest_sha256") or "").strip()
    artifact_id = str(payload.get("artifact_id") or "").strip()
    artifact_version = str(payload.get("artifact_version") or "").strip()
    recorded_at = str(payload.get("recorded_at") or "").strip()
    if not all((manifest_path, manifest_sha256, artifact_id, artifact_version, recorded_at)):
        raise BrainClusterPromotionStateError(
            BrainClusterPromotionStateErrorCode.STATE_INVALID,
            "Promotion-state pointer is missing required fields.",
            details={"field": field_name},
        )
    if not _SHA256_HEX_RE.fullmatch(manifest_sha256):
        raise BrainClusterPromotionStateError(
            BrainClusterPromotionStateErrorCode.STATE_INVALID,
            "Promotion-state pointer manifest_sha256 must be lowercase 64-hex.",
            details={"field": field_name, "manifest_sha256": manifest_sha256},
        )
    return BrainClusterPromotionPointer(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        artifact_id=artifact_id,
        artifact_version=artifact_version,
        recorded_at=recorded_at,
    )


def _state_pointer_from_manifest(manifest: BrainClusterArtifactManifest, *, recorded_at: str) -> BrainClusterPromotionPointer:
    return BrainClusterPromotionPointer(
        manifest_path=str(Path(manifest.manifest_path).resolve(strict=False)),
        manifest_sha256=manifest.manifest_sha256,
        artifact_id=manifest.artifact_id,
        artifact_version=manifest.artifact_version,
        recorded_at=recorded_at,
    )


def _persist_brain_cluster_promotion_state(state_path: Path, state: BrainClusterPromotionState) -> None:
    state_path = state_path.resolve(strict=False)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = f"{state_path.name}.{uuid4().hex}.partial"
    temp_path = state_path.parent / temp_name
    payload = state.as_dict()
    try:
        temp_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        temp_path.replace(state_path)
    except OSError as exc:
        raise BrainClusterPromotionStateError(
            BrainClusterPromotionStateErrorCode.STATE_IO_ERROR,
            "Failed to persist promotion-state JSON.",
            details={"state_path": str(state_path)},
        ) from exc
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def read_brain_cluster_promotion_state(state_path: str | Path) -> BrainClusterPromotionState:
    resolved_state_path = Path(state_path).resolve(strict=False)
    if not resolved_state_path.exists():
        return BrainClusterPromotionState(
            schema_version=BRAIN_CLUSTER_PROMOTION_STATE_SCHEMA_VERSION,
            updated_at=None,
            candidate=None,
            promoted=None,
            previous_promoted=None,
        )
    try:
        payload = json.loads(resolved_state_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BrainClusterPromotionStateError(
            BrainClusterPromotionStateErrorCode.STATE_IO_ERROR,
            "Failed to read promotion-state JSON.",
            details={"state_path": str(resolved_state_path)},
        ) from exc
    except json.JSONDecodeError as exc:
        raise BrainClusterPromotionStateError(
            BrainClusterPromotionStateErrorCode.STATE_INVALID,
            f"Promotion-state JSON is invalid: {exc.msg}",
            details={"state_path": str(resolved_state_path)},
        ) from exc
    if not isinstance(payload, dict):
        raise BrainClusterPromotionStateError(
            BrainClusterPromotionStateErrorCode.STATE_INVALID,
            "Promotion-state payload must be a JSON object.",
            details={"state_path": str(resolved_state_path)},
        )
    schema_version = str(payload.get("schema_version") or "").strip()
    if schema_version != BRAIN_CLUSTER_PROMOTION_STATE_SCHEMA_VERSION:
        raise BrainClusterPromotionStateError(
            BrainClusterPromotionStateErrorCode.STATE_INVALID,
            "Promotion-state schema_version is unsupported.",
            details={"schema_version": schema_version},
        )
    updated_at_raw = payload.get("updated_at")
    updated_at = None if updated_at_raw is None else str(updated_at_raw).strip()
    if updated_at == "":
        updated_at = None
    return BrainClusterPromotionState(
        schema_version=schema_version,
        updated_at=updated_at,
        candidate=_state_pointer_from_payload(payload.get("candidate"), field_name="candidate"),
        promoted=_state_pointer_from_payload(payload.get("promoted"), field_name="promoted"),
        previous_promoted=_state_pointer_from_payload(payload.get("previous_promoted"), field_name="previous_promoted"),
    )


def stage_brain_cluster_candidate_manifest(
    state_path: str | Path,
    *,
    candidate_manifest_path: str | Path,
) -> BrainClusterPromotionState:
    candidate_path = Path(candidate_manifest_path).resolve(strict=False)
    if not candidate_path.exists() or not candidate_path.is_file():
        raise BrainClusterPromotionStateError(
            BrainClusterPromotionStateErrorCode.MANIFEST_MISSING,
            "Candidate manifest path is missing.",
            details={"candidate_manifest_path": str(candidate_path)},
        )
    state = read_brain_cluster_promotion_state(state_path)
    staged_at = _utc_now_iso()
    candidate_pointer = BrainClusterPromotionPointer(
        manifest_path=str(candidate_path),
        manifest_sha256=_sha256_file(candidate_path),
        artifact_id="staged-candidate",
        artifact_version="pending-validation",
        recorded_at=staged_at,
    )
    next_state = BrainClusterPromotionState(
        schema_version=BRAIN_CLUSTER_PROMOTION_STATE_SCHEMA_VERSION,
        updated_at=staged_at,
        candidate=candidate_pointer,
        promoted=state.promoted,
        previous_promoted=state.previous_promoted,
    )
    _persist_brain_cluster_promotion_state(Path(state_path), next_state)
    return next_state


def promote_brain_cluster_candidate(state_path: str | Path) -> BrainClusterPromotionState:
    state = read_brain_cluster_promotion_state(state_path)
    if state.candidate is None:
        raise BrainClusterPromotionStateError(
            BrainClusterPromotionStateErrorCode.CANDIDATE_MISSING,
            "No staged candidate exists to promote.",
        )
    try:
        bundle = load_brain_cluster_artifacts(state.candidate.manifest_path)
    except BrainClusterArtifactError as exc:
        raise BrainClusterPromotionStateError(
            BrainClusterPromotionStateErrorCode.VALIDATION_FAILED,
            "Candidate manifest failed fail-closed validation.",
            details={"candidate_manifest_path": state.candidate.manifest_path, "artifact_error_code": exc.code},
        ) from exc
    if state.promoted and state.promoted.manifest_sha256 == bundle.manifest.manifest_sha256:
        raise BrainClusterPromotionStateError(
            BrainClusterPromotionStateErrorCode.INVALID_TRANSITION,
            "Candidate is already the active promoted artifact version.",
            details={"candidate_manifest_path": state.candidate.manifest_path},
        )
    promoted_at = _utc_now_iso()
    promoted_pointer = _state_pointer_from_manifest(bundle.manifest, recorded_at=promoted_at)
    next_state = BrainClusterPromotionState(
        schema_version=BRAIN_CLUSTER_PROMOTION_STATE_SCHEMA_VERSION,
        updated_at=promoted_at,
        candidate=None,
        promoted=promoted_pointer,
        previous_promoted=state.promoted,
    )
    _persist_brain_cluster_promotion_state(Path(state_path), next_state)
    return next_state


def rollback_brain_cluster_promoted(state_path: str | Path) -> BrainClusterPromotionState:
    state = read_brain_cluster_promotion_state(state_path)
    if state.promoted is None:
        raise BrainClusterPromotionStateError(
            BrainClusterPromotionStateErrorCode.PROMOTED_MISSING,
            "No active promoted artifact exists to roll back.",
        )
    if state.previous_promoted is None:
        raise BrainClusterPromotionStateError(
            BrainClusterPromotionStateErrorCode.ROLLBACK_UNAVAILABLE,
            "No prior promoted artifact is available for rollback.",
        )
    try:
        bundle = load_brain_cluster_artifacts(state.previous_promoted.manifest_path)
    except BrainClusterArtifactError as exc:
        raise BrainClusterPromotionStateError(
            BrainClusterPromotionStateErrorCode.VALIDATION_FAILED,
            "Rollback target failed fail-closed validation.",
            details={"rollback_manifest_path": state.previous_promoted.manifest_path, "artifact_error_code": exc.code},
        ) from exc
    rolled_back_at = _utc_now_iso()
    promoted_pointer = _state_pointer_from_manifest(bundle.manifest, recorded_at=rolled_back_at)
    next_state = BrainClusterPromotionState(
        schema_version=BRAIN_CLUSTER_PROMOTION_STATE_SCHEMA_VERSION,
        updated_at=rolled_back_at,
        candidate=None,
        promoted=promoted_pointer,
        previous_promoted=state.promoted,
    )
    _persist_brain_cluster_promotion_state(Path(state_path), next_state)
    return next_state


class ClusterDecision(str, Enum):
    ACCEPT = "accept"
    RETRY = "retry"
    ESCALATE = "escalate"
    FAIL_CLOSED = "fail_closed"


class ClusterState(str, Enum):
    RECEIVED = "received"
    ROUTED = "routed"
    RUNNING_EXPERT = "running_expert"
    GATED = "gated"
    ACCEPTED = "accepted"
    RETRY_EXPERT = "retry_expert"
    ESCALATED = "escalated"
    FAIL_CLOSED = "fail_closed"
    FINALIZED = "finalized"


@dataclass(frozen=True)
class ClusterTaskEnvelope:
    request_id: str
    cluster_id: str
    objective: str
    task_type: str
    risk_tier: str
    inputs: Mapping[str, Any]
    router_features: Mapping[str, Any] = field(default_factory=dict)
    constraints: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "cluster_id": self.cluster_id,
            "objective": self.objective,
            "task_type": self.task_type,
            "risk_tier": self.risk_tier,
            "inputs": dict(self.inputs),
            "router_features": dict(self.router_features),
            "constraints": dict(self.constraints),
        }


@dataclass(frozen=True)
class ReplayProvenanceEnvelope:
    request_id: str
    cluster_id: str
    policy_version: str
    route_seed: str
    replay_fingerprint: str
    deterministic: bool = True
    degraded: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_task(
        cls,
        task: ClusterTaskEnvelope,
        *,
        policy_version: str,
        route_seed: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ReplayProvenanceEnvelope":
        request = {
            "request_id": task.request_id,
            "objective": task.objective,
            "task_type": task.task_type,
            "risk_tier": task.risk_tier,
            "inputs": dict(task.inputs),
            "router_features": dict(task.router_features),
            "constraints": dict(task.constraints),
            "route_seed": route_seed,
            "policy_version": policy_version,
        }
        dataset_scope = {"cluster_id": task.cluster_id, "risk_tier": task.risk_tier}
        fingerprint = flybrain_replay_fingerprint(CLUSTER_TOOL_NAME, request, dataset_scope)
        return cls(
            request_id=task.request_id,
            cluster_id=task.cluster_id,
            policy_version=policy_version,
            route_seed=route_seed,
            replay_fingerprint=fingerprint,
            deterministic=True,
            degraded=False,
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True)
class RoutePlan:
    primary: str
    alternates: tuple[str, ...]
    topk_scores: Mapping[str, float]
    routing_entropy: float
    policy_version: str

    def ordered_experts(self) -> list[str]:
        return [self.primary, *self.alternates]


@dataclass(frozen=True)
class ExpertOutput:
    expert_id: str
    confidence: float
    claims: Sequence[str]
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    provenance_refs: Sequence[str] = field(default_factory=tuple)
    warnings: Sequence[str] = field(default_factory=tuple)
    degraded: bool = False


@dataclass(frozen=True)
class GateResult:
    decision: ClusterDecision
    reason: str
    warnings: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class ClusterRunResult:
    decision: ClusterDecision
    final_state: ClusterState
    state_history: tuple[ClusterState, ...]
    route_plan: RoutePlan
    provenance: ReplayProvenanceEnvelope
    selected_expert: str | None
    output: ExpertOutput | None
    gate_reason: str
    warnings: tuple[str, ...] = field(default_factory=tuple)


class RegionExpert(Protocol):
    expert_id: str
    region: str

    def infer(self, task: ClusterTaskEnvelope, provenance: ReplayProvenanceEnvelope) -> ExpertOutput:
        ...


class RegionRouter(Protocol):
    def plan(self, task: ClusterTaskEnvelope) -> RoutePlan:
        ...


class ClusterGate(Protocol):
    def evaluate(
        self,
        task: ClusterTaskEnvelope,
        provenance: ReplayProvenanceEnvelope,
        route: RoutePlan,
        output: ExpertOutput,
    ) -> GateResult:
        ...


def _normalized_entropy(scores: Mapping[str, float]) -> float:
    if not scores:
        return 0.0
    positive = [max(float(v), 0.0) for v in scores.values()]
    total = sum(positive)
    if total <= 0.0:
        return 0.0
    probs = [v / total for v in positive if v > 0.0]
    if len(probs) <= 1:
        return 0.0
    entropy = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(len(probs))
    if max_entropy <= 0.0:
        return 0.0
    return entropy / max_entropy


class DeterministicRegionRouter:
    def __init__(
        self,
        *,
        routes_by_task_type: Mapping[str, Sequence[str]],
        policy_version: str = "braincluster-router/v1",
    ):
        self._routes = {key: tuple(value) for key, value in routes_by_task_type.items()}
        self.policy_version = policy_version

    def plan(self, task: ClusterTaskEnvelope) -> RoutePlan:
        candidates = list(self._routes.get(task.task_type, self._routes.get("default", ("generalist",))))
        preferred = str(task.router_features.get("preferred_region", "")).strip()
        risk = str(task.risk_tier).strip().lower()

        scores: dict[str, float] = {}
        for index, expert_id in enumerate(candidates):
            score = 1.0 / (index + 1)
            if preferred and expert_id == preferred:
                score += 0.35
            if risk == "high" and "safety" in expert_id:
                score += 0.5
            scores[expert_id] = score

        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        if not ordered:
            ordered = [("generalist", 1.0)]
            scores = {"generalist": 1.0}

        primary = ordered[0][0]
        alternates = tuple(item[0] for item in ordered[1:])
        entropy = _normalized_entropy(scores)
        return RoutePlan(
            primary=primary,
            alternates=alternates,
            topk_scores=scores,
            routing_entropy=entropy,
            policy_version=self.policy_version,
        )


class ConfidenceGate:
    def __init__(self, *, min_confidence: float = 0.65):
        self.min_confidence = float(min_confidence)

    def evaluate(
        self,
        task: ClusterTaskEnvelope,
        provenance: ReplayProvenanceEnvelope,
        route: RoutePlan,
        output: ExpertOutput,
    ) -> GateResult:
        if output.confidence < self.min_confidence:
            return GateResult(
                decision=ClusterDecision.FAIL_CLOSED,
                reason=f"expert confidence below threshold ({output.confidence:.3f} < {self.min_confidence:.3f})",
            )
        return GateResult(decision=ClusterDecision.ACCEPT, reason="confidence gate passed")


class ProvenanceRefsGate:
    def __init__(self, *, on_missing: ClusterDecision = ClusterDecision.RETRY):
        self.on_missing = on_missing

    def evaluate(
        self,
        task: ClusterTaskEnvelope,
        provenance: ReplayProvenanceEnvelope,
        route: RoutePlan,
        output: ExpertOutput,
    ) -> GateResult:
        if output.provenance_refs:
            return GateResult(decision=ClusterDecision.ACCEPT, reason="provenance refs present")
        return GateResult(decision=self.on_missing, reason="missing provenance refs")


class ReplayFingerprintGate:
    def __init__(self, *, require_present: bool = True):
        self.require_present = require_present

    def evaluate(
        self,
        task: ClusterTaskEnvelope,
        provenance: ReplayProvenanceEnvelope,
        route: RoutePlan,
        output: ExpertOutput,
    ) -> GateResult:
        artifact_fp = str(output.artifacts.get("replay_fingerprint", "")).strip()
        if self.require_present and not artifact_fp:
            return GateResult(
                decision=ClusterDecision.FAIL_CLOSED,
                reason="missing replay fingerprint in expert output artifacts",
            )
        if artifact_fp and artifact_fp != provenance.replay_fingerprint:
            return GateResult(
                decision=ClusterDecision.FAIL_CLOSED,
                reason="replay fingerprint mismatch between envelope and expert output",
            )
        return GateResult(decision=ClusterDecision.ACCEPT, reason="replay fingerprint gate passed")


class BrainClusterCoordinator:
    def __init__(
        self,
        *,
        router: RegionRouter,
        experts: Sequence[RegionExpert],
        gates: Sequence[ClusterGate] | None = None,
        max_attempts: int = 3,
    ):
        self.router = router
        self.experts = {expert.expert_id: expert for expert in experts}
        self.gates = list(gates or [])
        self.max_attempts = max(1, int(max_attempts))

    def run(
        self,
        task: ClusterTaskEnvelope,
        *,
        route_seed: str = "default",
    ) -> ClusterRunResult:
        history: list[ClusterState] = [ClusterState.RECEIVED]
        warnings: list[str] = []
        route = self.router.plan(task)
        history.append(ClusterState.ROUTED)
        provenance = ReplayProvenanceEnvelope.from_task(
            task,
            policy_version=route.policy_version,
            route_seed=route_seed,
            metadata={"routing_entropy": route.routing_entropy},
        )

        selected_expert: str | None = None
        output: ExpertOutput | None = None
        gate_reason = "no experts were attempted"
        attempts = 0

        for expert_id in route.ordered_experts():
            if attempts >= self.max_attempts:
                warnings.append("max attempts reached before evaluating all alternates")
                break
            attempts += 1
            expert = self.experts.get(expert_id)
            if expert is None:
                warnings.append(f"expert '{expert_id}' unavailable")
                continue

            selected_expert = expert_id
            history.append(ClusterState.RUNNING_EXPERT)
            output = expert.infer(task, provenance)
            history.append(ClusterState.GATED)
            gate = self._run_gates(task, provenance, route, output)
            warnings.extend(gate.warnings)
            gate_reason = gate.reason
            if gate.decision == ClusterDecision.ACCEPT:
                history.append(ClusterState.ACCEPTED)
                history.append(ClusterState.FINALIZED)
                return ClusterRunResult(
                    decision=ClusterDecision.ACCEPT,
                    final_state=ClusterState.FINALIZED,
                    state_history=tuple(history),
                    route_plan=route,
                    provenance=provenance,
                    selected_expert=selected_expert,
                    output=output,
                    gate_reason=gate_reason,
                    warnings=tuple(warnings),
                )
            if gate.decision == ClusterDecision.RETRY:
                history.append(ClusterState.RETRY_EXPERT)
                continue
            if gate.decision == ClusterDecision.ESCALATE:
                history.append(ClusterState.ESCALATED)
                history.append(ClusterState.FINALIZED)
                return ClusterRunResult(
                    decision=ClusterDecision.ESCALATE,
                    final_state=ClusterState.FINALIZED,
                    state_history=tuple(history),
                    route_plan=route,
                    provenance=provenance,
                    selected_expert=selected_expert,
                    output=output,
                    gate_reason=gate_reason,
                    warnings=tuple(warnings),
                )
            history.append(ClusterState.FAIL_CLOSED)
            history.append(ClusterState.FINALIZED)
            return ClusterRunResult(
                decision=ClusterDecision.FAIL_CLOSED,
                final_state=ClusterState.FINALIZED,
                state_history=tuple(history),
                route_plan=route,
                provenance=provenance,
                selected_expert=selected_expert,
                output=output,
                gate_reason=gate_reason,
                warnings=tuple(warnings),
            )

        history.append(ClusterState.FAIL_CLOSED)
        history.append(ClusterState.FINALIZED)
        return ClusterRunResult(
            decision=ClusterDecision.FAIL_CLOSED,
            final_state=ClusterState.FINALIZED,
            state_history=tuple(history),
            route_plan=route,
            provenance=provenance,
            selected_expert=selected_expert,
            output=output,
            gate_reason=gate_reason,
            warnings=tuple(warnings),
        )

    def _run_gates(
        self,
        task: ClusterTaskEnvelope,
        provenance: ReplayProvenanceEnvelope,
        route: RoutePlan,
        output: ExpertOutput,
    ) -> GateResult:
        for gate in self.gates:
            verdict = gate.evaluate(task, provenance, route, output)
            if verdict.decision != ClusterDecision.ACCEPT:
                return verdict
        return GateResult(decision=ClusterDecision.ACCEPT, reason="all gates passed")


@dataclass(frozen=True)
class ShadowReplayFixture:
    fixture_id: str
    task: ClusterTaskEnvelope
    route_seed: str = "shadow-replay"


@dataclass(frozen=True)
class ShadowReplayThresholds:
    min_decision_match_rate: float = 0.95
    max_mean_abs_confidence_drift: float = 0.12
    max_fail_closed_rate_delta: float = 0.05
    max_mean_abs_routing_entropy_delta: float = 0.20
    max_selected_expert_concentration: float = 1.00
    max_selected_expert_concentration_delta: float = 0.20

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_decision_match_rate": float(self.min_decision_match_rate),
            "max_mean_abs_confidence_drift": float(self.max_mean_abs_confidence_drift),
            "max_fail_closed_rate_delta": float(self.max_fail_closed_rate_delta),
            "max_mean_abs_routing_entropy_delta": float(self.max_mean_abs_routing_entropy_delta),
            "max_selected_expert_concentration": float(self.max_selected_expert_concentration),
            "max_selected_expert_concentration_delta": float(self.max_selected_expert_concentration_delta),
        }


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _load_json_value(source: str | Path | Mapping[str, Any] | Sequence[Any], *, label: str) -> Any:
    if isinstance(source, Mapping):
        return dict(source)
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        return list(source)
    text = str(source).strip()
    if not text:
        raise ValueError(f"{label} is required")
    if text.startswith("{") or text.startswith("["):
        return json.loads(text)
    path = Path(text)
    if not path.exists() or not path.is_file():
        raise ValueError(f"{label} not found: {text}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label}: {exc.msg}") from exc


def _extract_items(payload: Mapping[str, Any] | Sequence[Any], *, key: str) -> list[Any]:
    if isinstance(payload, Mapping):
        if key in payload:
            values = payload[key]
        elif key.rstrip("s") in payload:
            values = payload[key.rstrip("s")]
        else:
            raise ValueError(f"payload missing '{key}'")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            raise ValueError(f"payload field '{key}' must be an array")
        return list(values)
    return list(payload)


def _normalize_task_envelope(raw: ClusterTaskEnvelope | Mapping[str, Any]) -> ClusterTaskEnvelope:
    if isinstance(raw, ClusterTaskEnvelope):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError("fixture task must be an object")
    required = ("request_id", "cluster_id", "objective", "task_type", "risk_tier", "inputs")
    missing = [field for field in required if field not in raw]
    if missing:
        raise ValueError(f"fixture task missing required fields: {', '.join(missing)}")
    inputs = raw.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("fixture task inputs must be an object")
    router_features = raw.get("router_features", {})
    constraints = raw.get("constraints", {})
    if not isinstance(router_features, Mapping):
        raise ValueError("fixture task router_features must be an object")
    if not isinstance(constraints, Mapping):
        raise ValueError("fixture task constraints must be an object")
    return ClusterTaskEnvelope(
        request_id=str(raw["request_id"]),
        cluster_id=str(raw["cluster_id"]),
        objective=str(raw["objective"]),
        task_type=str(raw["task_type"]),
        risk_tier=str(raw["risk_tier"]),
        inputs=dict(inputs),
        router_features=dict(router_features),
        constraints=dict(constraints),
    )


def _normalize_shadow_fixture(raw: ShadowReplayFixture | Mapping[str, Any], *, index: int) -> ShadowReplayFixture:
    if isinstance(raw, ShadowReplayFixture):
        return raw
    if not isinstance(raw, Mapping):
        raise ValueError(f"fixture at index {index} must be an object")
    fixture_id = str(raw.get("fixture_id") or f"fixture-{index + 1}").strip()
    if not fixture_id:
        raise ValueError(f"fixture at index {index} has empty fixture_id")
    if "task" not in raw:
        raise ValueError(f"fixture '{fixture_id}' missing task")
    route_seed = str(raw.get("route_seed") or "shadow-replay").strip() or "shadow-replay"
    task = _normalize_task_envelope(raw["task"])
    return ShadowReplayFixture(fixture_id=fixture_id, task=task, route_seed=route_seed)


def load_shadow_replay_fixtures(
    fixtures: Sequence[ShadowReplayFixture | Mapping[str, Any]] | Mapping[str, Any] | str | Path,
) -> tuple[ShadowReplayFixture, ...]:
    payload = _load_json_value(fixtures, label="fixtures") if isinstance(fixtures, (str, Path)) else fixtures
    if isinstance(payload, Mapping):
        raw_items = _extract_items(payload, key="fixtures")
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        raw_items = list(payload)
    else:
        raise ValueError("fixtures payload must be an array or object")
    normalized = tuple(_normalize_shadow_fixture(item, index=i) for i, item in enumerate(raw_items))
    if not normalized:
        raise ValueError("fixtures must be non-empty")
    seen: set[str] = set()
    for fixture in normalized:
        if fixture.fixture_id in seen:
            raise ValueError(f"duplicate fixture_id: {fixture.fixture_id}")
        seen.add(fixture.fixture_id)
    return normalized


def _as_thresholds_obj(thresholds: ShadowReplayThresholds | Mapping[str, Any] | None) -> ShadowReplayThresholds:
    if thresholds is None:
        return ShadowReplayThresholds()
    if isinstance(thresholds, ShadowReplayThresholds):
        return thresholds
    if not isinstance(thresholds, Mapping):
        raise ValueError("thresholds must be ShadowReplayThresholds or mapping")
    defaults = ShadowReplayThresholds()
    return ShadowReplayThresholds(
        min_decision_match_rate=float(thresholds.get("min_decision_match_rate", defaults.min_decision_match_rate)),
        max_mean_abs_confidence_drift=float(
            thresholds.get("max_mean_abs_confidence_drift", defaults.max_mean_abs_confidence_drift)
        ),
        max_fail_closed_rate_delta=float(thresholds.get("max_fail_closed_rate_delta", defaults.max_fail_closed_rate_delta)),
        max_mean_abs_routing_entropy_delta=float(
            thresholds.get("max_mean_abs_routing_entropy_delta", defaults.max_mean_abs_routing_entropy_delta)
        ),
        max_selected_expert_concentration=float(
            thresholds.get("max_selected_expert_concentration", defaults.max_selected_expert_concentration)
        ),
        max_selected_expert_concentration_delta=float(
            thresholds.get(
                "max_selected_expert_concentration_delta",
                defaults.max_selected_expert_concentration_delta,
            )
        ),
    )


class _ArtifactReplayExpert:
    def __init__(self, expert_id: str, *, behavior: Mapping[str, Any] | None = None):
        self.expert_id = expert_id
        self.region = expert_id
        self._behavior = dict(behavior or {})

    def infer(self, task: ClusterTaskEnvelope, provenance: ReplayProvenanceEnvelope) -> ExpertOutput:
        per_task_conf = self._behavior.get("confidence_by_task_type", {})
        if isinstance(per_task_conf, Mapping) and task.task_type in per_task_conf:
            confidence = float(per_task_conf[task.task_type])
        else:
            confidence = float(self._behavior.get("confidence", 0.82))
        confidence = max(0.0, min(1.0, confidence))

        mode = str(self._behavior.get("replay_fingerprint_mode", "match")).strip().lower()
        if mode == "missing":
            replay_fp = ""
        elif mode == "mismatch":
            replay_fp = hashlib.sha256(f"{provenance.replay_fingerprint}:mismatch".encode("utf-8")).hexdigest()
        else:
            replay_fp = provenance.replay_fingerprint

        refs_raw = self._behavior.get("provenance_refs")
        if refs_raw is None:
            provenance_refs = (f"{task.cluster_id}:{self.expert_id}:{task.request_id}",)
        elif isinstance(refs_raw, Sequence) and not isinstance(refs_raw, (str, bytes, bytearray)):
            provenance_refs = tuple(str(item) for item in refs_raw if str(item).strip())
        elif str(refs_raw).strip():
            provenance_refs = (str(refs_raw),)
        else:
            provenance_refs = tuple()

        return ExpertOutput(
            expert_id=self.expert_id,
            confidence=confidence,
            claims=[f"{self.expert_id} handled {task.task_type}"],
            artifacts={"replay_fingerprint": replay_fp},
            provenance_refs=provenance_refs,
        )


def _build_coordinator_from_bundle(bundle: BrainClusterArtifactBundle) -> BrainClusterCoordinator:
    router_payload = bundle.router if isinstance(bundle.router, Mapping) else {}
    routes = router_payload.get("routes_by_task_type", {}) if isinstance(router_payload, Mapping) else {}
    normalized_routes: dict[str, tuple[str, ...]] = {}
    if isinstance(routes, Mapping):
        for key, value in routes.items():
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                experts = tuple(str(item).strip() for item in value if str(item).strip())
                if experts:
                    normalized_routes[str(key)] = experts

    expert_rows = bundle.experts.get("experts", []) if isinstance(bundle.experts, Mapping) else []
    experts: list[_ArtifactReplayExpert] = []
    for row in expert_rows if isinstance(expert_rows, Sequence) else []:
        if not isinstance(row, Mapping):
            continue
        expert_id = str(row.get("expert_id") or "").strip()
        if not expert_id:
            continue
        behavior = row.get("shadow_replay")
        behavior_map = behavior if isinstance(behavior, Mapping) else {}
        experts.append(_ArtifactReplayExpert(expert_id, behavior=behavior_map))

    if not normalized_routes:
        route_ids = tuple(expert.expert_id for expert in experts)
        if route_ids:
            normalized_routes["default"] = route_ids
        else:
            normalized_routes["default"] = ("generalist",)

    gates: list[ClusterGate] = [
        ConfidenceGate(min_confidence=0.65),
        ProvenanceRefsGate(on_missing=ClusterDecision.FAIL_CLOSED),
        ReplayFingerprintGate(),
    ]
    return BrainClusterCoordinator(
        router=DeterministicRegionRouter(
            routes_by_task_type=normalized_routes,
            policy_version=str(router_payload.get("policy_version") or "braincluster-router/v1"),
        ),
        experts=experts,
        gates=gates,
        max_attempts=max(1, len(experts) + 1),
    )


def run_shadow_replay_fixtures_with_artifact(
    fixtures: Sequence[ShadowReplayFixture],
    *,
    manifest_path: str | Path,
) -> tuple[dict[str, Any], ...]:
    bundle = load_brain_cluster_artifacts(manifest_path)
    coordinator = _build_coordinator_from_bundle(bundle)
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        result = coordinator.run(fixture.task, route_seed=fixture.route_seed)
        rows.append(
            {
                "fixture_id": fixture.fixture_id,
                "decision": result.decision.value,
                "selected_expert": result.selected_expert,
                "confidence": float(result.output.confidence) if result.output else None,
                "fail_closed": bool(result.decision == ClusterDecision.FAIL_CLOSED),
                "routing_entropy": float(result.route_plan.routing_entropy),
                "routing_scores": {str(k): float(v) for k, v in sorted(result.route_plan.topk_scores.items())},
                "gate_reason": result.gate_reason,
                "policy_version": result.route_plan.policy_version,
                "replay_fingerprint": result.provenance.replay_fingerprint,
            }
        )
    return tuple(rows)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _rate_true(values: Sequence[bool]) -> float:
    if not values:
        return 0.0
    return float(sum(1 for item in values if item) / len(values))


def _selected_expert_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    counts = Counter(str(row.get("selected_expert") or "none") for row in rows)
    if total <= 0:
        return {"total": 0, "concentration": 0.0, "unique_experts": 0, "counts": {}}
    concentration = max(counts.values()) / total
    return {
        "total": total,
        "concentration": float(concentration),
        "unique_experts": len(counts),
        "counts": {key: int(counts[key]) for key in sorted(counts)},
    }


def compare_shadow_replay_runs(
    fixtures: Sequence[ShadowReplayFixture],
    *,
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    thresholds: ShadowReplayThresholds | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if len(fixtures) != len(baseline_rows) or len(fixtures) != len(candidate_rows):
        raise ValueError("fixtures/baseline_rows/candidate_rows size mismatch")
    threshold_obj = _as_thresholds_obj(thresholds)
    fixture_comparisons: list[dict[str, Any]] = []
    decision_matches = 0
    confidence_deltas: list[float] = []
    entropy_deltas: list[float] = []
    baseline_fail_closed: list[bool] = []
    candidate_fail_closed: list[bool] = []

    for fixture, baseline, candidate in zip(fixtures, baseline_rows, candidate_rows, strict=True):
        baseline_decision = str(baseline.get("decision") or "")
        candidate_decision = str(candidate.get("decision") or "")
        is_match = baseline_decision == candidate_decision
        if is_match:
            decision_matches += 1

        baseline_conf = baseline.get("confidence")
        candidate_conf = candidate.get("confidence")
        conf_delta: float | None = None
        if baseline_conf is not None and candidate_conf is not None:
            conf_delta = float(candidate_conf) - float(baseline_conf)
            confidence_deltas.append(conf_delta)

        baseline_entropy = baseline.get("routing_entropy")
        candidate_entropy = candidate.get("routing_entropy")
        entropy_delta: float | None = None
        if baseline_entropy is not None and candidate_entropy is not None:
            entropy_delta = float(candidate_entropy) - float(baseline_entropy)
            entropy_deltas.append(entropy_delta)

        base_fc = bool(baseline.get("fail_closed", baseline_decision == ClusterDecision.FAIL_CLOSED.value))
        cand_fc = bool(candidate.get("fail_closed", candidate_decision == ClusterDecision.FAIL_CLOSED.value))
        baseline_fail_closed.append(base_fc)
        candidate_fail_closed.append(cand_fc)

        fixture_comparisons.append(
            {
                "fixture_id": fixture.fixture_id,
                "decision_match": is_match,
                "baseline_decision": baseline_decision,
                "candidate_decision": candidate_decision,
                "baseline_selected_expert": baseline.get("selected_expert"),
                "candidate_selected_expert": candidate.get("selected_expert"),
                "baseline_confidence": baseline_conf,
                "candidate_confidence": candidate_conf,
                "confidence_delta": conf_delta,
                "baseline_routing_entropy": baseline_entropy,
                "candidate_routing_entropy": candidate_entropy,
                "routing_entropy_delta": entropy_delta,
                "baseline_fail_closed": base_fc,
                "candidate_fail_closed": cand_fc,
                "baseline_gate_reason": baseline.get("gate_reason"),
                "candidate_gate_reason": candidate.get("gate_reason"),
            }
        )

    total = len(fixtures)
    decision_match_rate = (decision_matches / total) if total > 0 else 0.0
    baseline_fail_closed_rate = _rate_true(baseline_fail_closed)
    candidate_fail_closed_rate = _rate_true(candidate_fail_closed)
    fail_closed_rate_delta = candidate_fail_closed_rate - baseline_fail_closed_rate

    mean_signed_conf = _mean(confidence_deltas)
    mean_abs_conf = _mean([abs(item) for item in confidence_deltas]) if confidence_deltas else 0.0
    max_abs_conf = max((abs(item) for item in confidence_deltas), default=0.0)

    mean_signed_entropy = _mean(entropy_deltas)
    mean_abs_entropy = _mean([abs(item) for item in entropy_deltas]) if entropy_deltas else 0.0
    max_abs_entropy = max((abs(item) for item in entropy_deltas), default=0.0)

    baseline_collapse = _selected_expert_summary(baseline_rows)
    candidate_collapse = _selected_expert_summary(candidate_rows)
    concentration_delta = float(candidate_collapse["concentration"]) - float(baseline_collapse["concentration"])

    failures: list[str] = []
    issue_flags: list[dict[str, Any]] = []

    if decision_match_rate < threshold_obj.min_decision_match_rate:
        failures.append(
            "decision match rate below threshold "
            f"({decision_match_rate:.3f} < {threshold_obj.min_decision_match_rate:.3f})"
        )
    if mean_abs_conf > threshold_obj.max_mean_abs_confidence_drift:
        failures.append(
            "confidence drift above threshold "
            f"({mean_abs_conf:.3f} > {threshold_obj.max_mean_abs_confidence_drift:.3f})"
        )
    if fail_closed_rate_delta > threshold_obj.max_fail_closed_rate_delta:
        failures.append(
            "fail-closed rate delta above threshold "
            f"({fail_closed_rate_delta:.3f} > {threshold_obj.max_fail_closed_rate_delta:.3f})"
        )
    if mean_abs_entropy > threshold_obj.max_mean_abs_routing_entropy_delta:
        failures.append(
            "routing entropy drift above threshold "
            f"({mean_abs_entropy:.3f} > {threshold_obj.max_mean_abs_routing_entropy_delta:.3f})"
        )
    if float(candidate_collapse["concentration"]) > threshold_obj.max_selected_expert_concentration:
        failures.append(
            "candidate expert selection concentration above threshold "
            f"({float(candidate_collapse['concentration']):.3f} > {threshold_obj.max_selected_expert_concentration:.3f})"
        )
    if concentration_delta > threshold_obj.max_selected_expert_concentration_delta:
        failures.append(
            "candidate expert concentration delta above threshold "
            f"({concentration_delta:.3f} > {threshold_obj.max_selected_expert_concentration_delta:.3f})"
        )

    if decision_match_rate < 1.0:
        issue_flags.append({"code": "decision_drift", "severity": "warning", "value": float(1.0 - decision_match_rate)})
    if mean_abs_conf > 0.0:
        issue_flags.append({"code": "confidence_drift", "severity": "warning", "value": float(mean_abs_conf)})
    if mean_abs_entropy > 0.0:
        issue_flags.append({"code": "routing_entropy_drift", "severity": "warning", "value": float(mean_abs_entropy)})
    if concentration_delta > 0.0:
        issue_flags.append({"code": "expert_collapse_delta", "severity": "warning", "value": float(concentration_delta)})
    if float(candidate_collapse["concentration"]) >= 0.95:
        issue_flags.append(
            {
                "code": "expert_collapse_high",
                "severity": "critical",
                "value": float(candidate_collapse["concentration"]),
            }
        )

    return {
        "schema_version": "braincluster-shadow-replay-report/v1",
        "deterministic": True,
        "pass": len(failures) == 0,
        "status": "pass" if len(failures) == 0 else "fail",
        "thresholds": threshold_obj.as_dict(),
        "metrics": {
            "fixture_count": total,
            "decision_match_rate": float(decision_match_rate),
            "confidence_drift": {
                "compared_count": len(confidence_deltas),
                "mean_signed_delta": float(mean_signed_conf),
                "mean_abs_delta": float(mean_abs_conf),
                "max_abs_delta": float(max_abs_conf),
            },
            "fail_closed_rate": {
                "baseline": float(baseline_fail_closed_rate),
                "candidate": float(candidate_fail_closed_rate),
                "delta": float(fail_closed_rate_delta),
            },
            "routing_entropy": {
                "compared_count": len(entropy_deltas),
                "mean_signed_delta": float(mean_signed_entropy),
                "mean_abs_delta": float(mean_abs_entropy),
                "max_abs_delta": float(max_abs_entropy),
            },
            "expert_collapse": {
                "baseline": baseline_collapse,
                "candidate": candidate_collapse,
                "concentration_delta": float(concentration_delta),
            },
        },
        "failure_reasons": failures,
        "issue_flags": issue_flags,
        "fixtures": fixture_comparisons,
        "exit_code": 0 if len(failures) == 0 else 1,
    }


def run_brain_cluster_shadow_replay(
    fixtures: Sequence[ShadowReplayFixture | Mapping[str, Any]] | Mapping[str, Any] | str | Path,
    *,
    baseline_manifest_path: str | Path,
    candidate_manifest_path: str | Path,
    thresholds: ShadowReplayThresholds | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_fixtures = load_shadow_replay_fixtures(fixtures)
    baseline_rows = run_shadow_replay_fixtures_with_artifact(
        normalized_fixtures,
        manifest_path=baseline_manifest_path,
    )
    candidate_rows = run_shadow_replay_fixtures_with_artifact(
        normalized_fixtures,
        manifest_path=candidate_manifest_path,
    )
    return compare_shadow_replay_runs(
        normalized_fixtures,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        thresholds=thresholds,
    )


def serialize_brain_cluster_shadow_replay_report(report: Mapping[str, Any]) -> str:
    return _stable_json(dict(report)) + "\n"
