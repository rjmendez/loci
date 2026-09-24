import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_fw_metadata_adapter as ffw  # noqa: E402


def _manifest_digest(manifest: dict[str, object]) -> str:
    canonical = json.loads(json.dumps(manifest))
    canonical["integrity"]["manifest_sha256"] = ""
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_valid_snapshot(root: Path) -> None:
    metadata_root = root / "snapshots" / "fw" / ffw.FW_VERSION_ID / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    metadata_file = metadata_root / "classes.json"
    metadata_file.write_text('{"ok":true}', encoding="utf-8")
    file_sha = hashlib.sha256(metadata_file.read_bytes()).hexdigest()
    manifest = {
        "schema_version": ffw.FW_MANIFEST_SCHEMA_VERSION,
        "manifest_id": "fw-test-manifest",
        "dataset": {
            "symbol": ffw.FW_DATASET_SYMBOL,
            "version_id": ffw.FW_VERSION_ID,
            "source": {"access_method": "local_snapshot_read_only"},
        },
        "artifact": {"relative_root": ffw.FW_SNAPSHOT_RELATIVE_ROOT},
        "source": {"access_method": "local_snapshot_read_only"},
        "scope": {"access_layer": ffw.FW_ACCESS_LAYER},
        "lineage": {"family": "fw"},
        "integrity": {
            "manifest_sha256": "",
            "files": [{"relative_path": "classes.json", "sha256": file_sha, "size_bytes": metadata_file.stat().st_size}],
            "verification": {"status": "verified", "verified_at": "2099-01-01T00:00:00+00:00"},
        },
        "refresh": {
            "decision": "no_change",
            "checked_at": "2099-01-01T00:00:00+00:00",
            "next_check_due": "2099-01-02T00:00:00+00:00",
            "supersedes_manifest_id": None,
        },
    }
    manifest["integrity"]["manifest_sha256"] = _manifest_digest(manifest)
    (metadata_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _request() -> ffw.FwMetadataQueryRequest:
    return ffw.FwMetadataQueryRequest(
        request_id="req-1",
        dataset_symbol=ffw.FW_DATASET_SYMBOL,
        version_id=ffw.FW_VERSION_ID,
        query_kind="scope_validation",
        scope_tags={"access_layer": ffw.FW_ACCESS_LAYER},
        query_payload={"scope": {"sex": "female", "stage": "adult"}},
    )


def test_fw_adapter_execute_success_with_valid_manifest(tmp_path):
    root = tmp_path / "flybrain-root"
    _write_valid_snapshot(root)
    adapter = ffw.FwMetadataLocalAdapter(storage_root=root)
    response = adapter.execute(_request()).as_dict()
    assert response["ok"] is True
    assert response["count_status"] == "exact"
    assert response["provenance"]["manifest_id"] == "fw-test-manifest"


def test_fw_adapter_rejects_manifest_hash_mismatch(tmp_path):
    root = tmp_path / "flybrain-root"
    _write_valid_snapshot(root)
    manifest_path = root / "snapshots" / "fw" / ffw.FW_VERSION_ID / "metadata" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["integrity"]["manifest_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    adapter = ffw.FwMetadataLocalAdapter(storage_root=root)
    response = adapter.execute(_request()).as_dict()
    assert response["ok"] is False
    assert response["error"]["code"] == ffw.AdapterErrorCode.INTEGRITY_MISMATCH.value


def test_fw_adapter_rejects_escape_relative_path(tmp_path):
    root = tmp_path / "flybrain-root"
    _write_valid_snapshot(root)
    manifest_path = root / "snapshots" / "fw" / ffw.FW_VERSION_ID / "metadata" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["integrity"]["files"][0]["relative_path"] = "..\\outside.json"
    manifest["integrity"]["manifest_sha256"] = _manifest_digest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    adapter = ffw.FwMetadataLocalAdapter(storage_root=root)
    response = adapter.execute(_request()).as_dict()
    assert response["ok"] is False
    assert response["error"]["code"] == ffw.AdapterErrorCode.PATH_ESCAPE.value


def test_fw_adapter_rejects_missing_or_malformed_integrity_files_entries(tmp_path):
    root = tmp_path / "flybrain-root"
    _write_valid_snapshot(root)
    manifest_path = root / "snapshots" / "fw" / ffw.FW_VERSION_ID / "metadata" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["integrity"]["files"] = [{"relative_path": "classes.json"}]
    manifest["integrity"]["manifest_sha256"] = _manifest_digest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    adapter = ffw.FwMetadataLocalAdapter(storage_root=root)
    response = adapter.execute(_request()).as_dict()
    assert response["ok"] is False
    assert response["error"]["code"] == ffw.AdapterErrorCode.MANIFEST_INVALID.value


def test_fw_adapter_rejects_checksum_mismatch_for_referenced_file(tmp_path):
    root = tmp_path / "flybrain-root"
    _write_valid_snapshot(root)
    metadata_file = root / "snapshots" / "fw" / ffw.FW_VERSION_ID / "metadata" / "classes.json"
    metadata_file.write_text('{"ok":false}', encoding="utf-8")
    manifest_path = root / "snapshots" / "fw" / ffw.FW_VERSION_ID / "metadata" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["integrity"]["files"][0]["sha256"] = hashlib.sha256(metadata_file.read_bytes()).hexdigest()
    manifest["integrity"]["manifest_sha256"] = _manifest_digest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    adapter = ffw.FwMetadataLocalAdapter(storage_root=root)
    response = adapter.execute(_request()).as_dict()
    assert response["ok"] is False
    assert response["error"]["code"] == ffw.AdapterErrorCode.INTEGRITY_MISMATCH.value


def test_fw_adapter_rejects_unverified_manifest_status(tmp_path):
    root = tmp_path / "flybrain-root"
    _write_valid_snapshot(root)
    manifest_path = root / "snapshots" / "fw" / ffw.FW_VERSION_ID / "metadata" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["integrity"]["verification"]["status"] = "quarantined"
    manifest["integrity"]["manifest_sha256"] = _manifest_digest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    adapter = ffw.FwMetadataLocalAdapter(storage_root=root)
    response = adapter.execute(_request()).as_dict()
    assert response["ok"] is False
    assert response["error"]["code"] == ffw.AdapterErrorCode.INTEGRITY_MISMATCH.value
