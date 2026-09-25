import json
import os
import sys
import hashlib
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_hb_adapter as fha  # noqa: E402


def _manifest_digest(manifest: dict[str, object]) -> str:
    canonical = json.loads(json.dumps(manifest))
    canonical["integrity"]["manifest_sha256"] = ""
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_manifest(root: Path, *, verification_status: str = "verified", refresh_decision: str = "no_change", next_check_due: str = "2099-01-01T00:00:00+00:00") -> None:
    manifest_dir = root / "snapshots" / "hb" / fha.HB_VERSION_ID / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    active_dir = root / "graph" / "hb" / fha.HB_VERSION_ID / "promotion"
    active_dir.mkdir(parents=True, exist_ok=True)
    (active_dir / "active_pointer.json").write_text(json.dumps({"active": True}), encoding="utf-8")
    graph_root = root / "graph" / "hb" / fha.HB_VERSION_ID
    graph_root.mkdir(parents=True, exist_ok=True)
    nodes_file = graph_root / "nodes.parquet"
    nodes_file.write_bytes(b"hb-nodes")
    nodes_sha256 = hashlib.sha256(nodes_file.read_bytes()).hexdigest()
    manifest = {
        "schema_version": fha.HB_MANIFEST_SCHEMA_VERSION,
        "manifest_id": "fbh-hb-test",
        "dataset": {"symbol": "hb", "version_id": fha.HB_VERSION_ID},
        "artifact": {"relative_root": f"graph/hb/{fha.HB_VERSION_ID}"},
        "source": {"access_method": "local_graph_snapshot"},
        "lineage": {"family": "hb"},
        "scope": {"sex": "female", "stage": "adult", "anatomy": "hemibrain_region"},
        "integrity": {
            "manifest_sha256": "",
            "files": [{"relative_path": "nodes.parquet", "sha256": nodes_sha256, "size_bytes": nodes_file.stat().st_size}],
            "verification": {"status": verification_status},
        },
        "refresh": {
            "decision": refresh_decision,
            "checked_at": "2099-01-01T00:00:00+00:00",
            "next_check_due": next_check_due,
            "supersedes_manifest_id": None,
        },
    }
    if verification_status == "verified":
        manifest["integrity"]["verification"]["verified_at"] = "2099-01-01T00:00:00+00:00"
    manifest["integrity"]["manifest_sha256"] = _manifest_digest(manifest)
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_capabilities_and_snapshot_contract(tmp_path):
    root = tmp_path / "flybrain-root"
    _write_manifest(root)
    adapter = fha.build_hb_adapter(root_override=root)

    caps = adapter.capabilities().to_dict()
    assert caps["dataset_symbol"] == "hb"
    assert caps["version_id"] == fha.HB_VERSION_ID
    assert "execute" in caps["supported_operations"]
    assert caps["supported_query_kinds"] == list(fha.HB_ALLOWED_QUERY_KINDS)

    snapshot = adapter.resolve_snapshot().to_dict()
    assert snapshot["manifest_id"] == "fbh-hb-test"
    assert snapshot["verification_status"] == "verified"
    assert snapshot["rollback_required"] is False


def test_validation_rejects_scope_and_fallback(tmp_path):
    root = tmp_path / "flybrain-root"
    _write_manifest(root)
    adapter = fha.build_hb_adapter(root_override=root)

    request = fha.HbQueryRequest(
        query_kind="connectivity_lookup",
        query_payload={"query": "MATCH (n) RETURN n LIMIT 1"},
        scope_tags=fha.HbScopeTags(sex="male"),
        allow_remote_fallback=True,
    )
    report = adapter.validate_compatibility(request)
    assert report.ok is False
    codes = {check.code for check in report.checks if not check.passed}
    assert fha.HbAdapterErrorCode.SCOPE_VIOLATION.value in codes
    assert fha.HbAdapterErrorCode.OUT_OF_SCOPE.value in codes


def test_execute_returns_local_provenance_and_rows(tmp_path):
    root = tmp_path / "flybrain-root"
    _write_manifest(root)

    def backend(request, snapshot):
        return [{"source_id": "A", "target_id": "B", "weight": 3}]

    adapter = fha.build_hb_adapter(root_override=root, backend=backend)
    result = adapter.execute(
        fha.HbQueryRequest(
            query_kind="connectivity_lookup",
            query_payload={"query": "MATCH (n) RETURN n LIMIT 1"},
        )
    )

    data = result.to_dict()
    assert data["status"] == "ok"
    assert data["count_status"] == "exact"
    assert data["row_count"] == 1
    assert data["source"] == "local"
    assert data["provenance"]["dataset_symbol"] == "hb"
    assert data["provenance"]["version_id"] == fha.HB_VERSION_ID
    assert data["provenance"]["result_contract_version"] == fha.HB_QUERY_RESULT_VERSION
    assert data["rows"][0]["source_id"] == "A"
    assert data["rows"][0]["target_id"] == "B"
    assert data["rows"][0]["weight"] == 3


def test_manifest_not_verified_requires_rollback(tmp_path):
    root = tmp_path / "flybrain-root"
    _write_manifest(root, verification_status="quarantined")
    adapter = fha.build_hb_adapter(root_override=root)

    with pytest.raises(fha.HbAdapterError) as excinfo:
        adapter.resolve_snapshot()
    assert excinfo.value.code == fha.HbAdapterErrorCode.INTEGRITY_MISMATCH.value
    assert excinfo.value.rollback_required is True


def test_manifest_self_hash_mismatch_blocks_snapshot(tmp_path):
    root = tmp_path / "flybrain-root"
    _write_manifest(root)
    manifest_path = root / "snapshots" / "hb" / fha.HB_VERSION_ID / "manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["integrity"]["manifest_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    adapter = fha.build_hb_adapter(root_override=root)
    with pytest.raises(fha.HbAdapterError) as excinfo:
        adapter.resolve_snapshot()
    assert excinfo.value.code == fha.HbAdapterErrorCode.INTEGRITY_MISMATCH.value
    assert excinfo.value.rollback_required is True


def test_manifest_rejects_malformed_integrity_files_entries(tmp_path):
    root = tmp_path / "flybrain-root"
    _write_manifest(root)
    manifest_path = root / "snapshots" / "hb" / fha.HB_VERSION_ID / "manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["integrity"]["files"] = [{"relative_path": "nodes.parquet"}]
    manifest["integrity"]["manifest_sha256"] = _manifest_digest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    adapter = fha.build_hb_adapter(root_override=root)
    with pytest.raises(fha.HbAdapterError) as excinfo:
        adapter.resolve_snapshot()
    assert excinfo.value.code == fha.HbAdapterErrorCode.MANIFEST_INVALID.value


def test_manifest_rejects_unsafe_relative_path(tmp_path):
    root = tmp_path / "flybrain-root"
    _write_manifest(root)
    manifest_path = root / "snapshots" / "hb" / fha.HB_VERSION_ID / "manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["integrity"]["files"][0]["relative_path"] = "../outside.parquet"
    manifest["integrity"]["manifest_sha256"] = _manifest_digest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    adapter = fha.build_hb_adapter(root_override=root)
    with pytest.raises(fha.HbAdapterError) as excinfo:
        adapter.resolve_snapshot()
    assert excinfo.value.code == fha.HbAdapterErrorCode.ROOT_OUT_OF_BOUNDS.value


def test_manifest_detects_ref_file_checksum_mismatch(tmp_path):
    root = tmp_path / "flybrain-root"
    _write_manifest(root)
    graph_root = root / "graph" / "hb" / fha.HB_VERSION_ID
    nodes_file = graph_root / "nodes.parquet"
    nodes_file.write_bytes(b"tampered-hb-nodes")
    manifest_path = root / "snapshots" / "hb" / fha.HB_VERSION_ID / "manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["integrity"]["files"][0]["sha256"] = hashlib.sha256(b"tampered-hb-nodes").hexdigest()
    manifest["integrity"]["manifest_sha256"] = _manifest_digest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    adapter = fha.build_hb_adapter(root_override=root)
    with pytest.raises(fha.HbAdapterError) as excinfo:
        adapter.resolve_snapshot()
    assert excinfo.value.code == fha.HbAdapterErrorCode.INTEGRITY_MISMATCH.value


def test_unverified_manifest_blocks_execution(tmp_path):
    root = tmp_path / "flybrain-root"
    _write_manifest(root, verification_status="quarantined")

    def backend(request, snapshot):
        return [{"source_id": "A", "target_id": "B", "weight": 3}]

    adapter = fha.build_hb_adapter(root_override=root, backend=backend)
    with pytest.raises(fha.HbAdapterError) as excinfo:
        adapter.execute(
            fha.HbQueryRequest(
                query_kind="connectivity_lookup",
                query_payload={"query": "MATCH (n) RETURN n LIMIT 1"},
            )
        )
    assert excinfo.value.code == fha.HbAdapterErrorCode.INTEGRITY_MISMATCH.value
    assert excinfo.value.rollback_required is True


def test_disabled_promotion_pointer_blocks_execution(tmp_path):
    root = tmp_path / "flybrain-root"
    _write_manifest(root)
    active_pointer = root / "graph" / "hb" / fha.HB_VERSION_ID / "promotion" / "active_pointer.json"
    active_pointer.write_text(json.dumps({"active": False}), encoding="utf-8")

    def backend(request, snapshot):
        return [{"source_id": "A", "target_id": "B", "weight": 3}]

    adapter = fha.build_hb_adapter(root_override=root, backend=backend)
    with pytest.raises(fha.HbAdapterError) as excinfo:
        adapter.execute(
            fha.HbQueryRequest(
                query_kind="connectivity_lookup",
                query_payload={"query": "MATCH (n) RETURN n LIMIT 1"},
            )
        )
    assert excinfo.value.code == fha.HbAdapterErrorCode.PROMOTION_STATE_INVALID.value
    assert excinfo.value.rollback_required is True


# ---------------------------------------------------------------------------
# Manifest-integrity checks, one per guard. Each fixture differs from the
# valid one in exactly the property the guard exists for, so removing that
# guard lets the snapshot resolve and the test fails.
# ---------------------------------------------------------------------------


def _manifest_path(root: Path) -> Path:
    return root / "snapshots" / "hb" / fha.HB_VERSION_ID / "manifest" / "manifest.json"


def _rewrite_manifest(root: Path, mutate) -> None:
    path = _manifest_path(root)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest["integrity"]["manifest_sha256"] = _manifest_digest(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _resolve_error(root: Path) -> fha.HbAdapterError:
    with pytest.raises(fha.HbAdapterError) as excinfo:
        fha.build_hb_adapter(root_override=root).resolve_snapshot()
    return excinfo.value


def test_same_size_tamper_is_caught_by_the_sha256_check(tmp_path):
    # The older tamper test changed the file length, so the size check caught
    # it and the SHA256 compare could be deleted with every test still green.
    root = tmp_path / "flybrain-root"
    _write_manifest(root)
    nodes_file = root / "graph" / "hb" / fha.HB_VERSION_ID / "nodes.parquet"
    original = nodes_file.read_bytes()
    tampered = b"hb-nodez"
    assert len(tampered) == len(original) and tampered != original
    nodes_file.write_bytes(tampered)

    err = _resolve_error(root)
    assert err.code == fha.HbAdapterErrorCode.INTEGRITY_MISMATCH.value
    assert err.message == "SHA256 mismatch for integrity file: nodes.parquet"
    assert err.rollback_required is True


def test_resize_is_caught_by_the_size_check(tmp_path):
    root = tmp_path / "flybrain-root"
    _write_manifest(root)
    (root / "graph" / "hb" / fha.HB_VERSION_ID / "nodes.parquet").write_bytes(b"hb-nodes-longer")

    err = _resolve_error(root)
    assert err.code == fha.HbAdapterErrorCode.INTEGRITY_MISMATCH.value
    assert err.message == "size_bytes mismatch for integrity file: nodes.parquet"


@pytest.mark.parametrize("field, value", [("sex", "male"), ("stage", "larva"), ("anatomy", "optic_lobe")])
def test_manifest_scope_outside_phase1_is_rejected(tmp_path, field, value):
    root = tmp_path / "flybrain-root"
    _write_manifest(root)
    _rewrite_manifest(root, lambda m: m["scope"].__setitem__(field, value))

    err = _resolve_error(root)
    assert err.code == fha.HbAdapterErrorCode.OUT_OF_SCOPE.value
    assert err.message == "Pinned manifest scope does not match phase-1 hb scope"


def test_integrity_file_symlinked_outside_the_artifact_root_is_rejected(tmp_path):
    # A path that is lexically safe ("nodes.parquet") but resolves outside the
    # artifact root through a symlink. The manifest carries the real SHA256 and
    # size of the outside file, so only the resolved-path containment check
    # stops it.
    root = tmp_path / "flybrain-root"
    _write_manifest(root)
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"not-hb-data")
    nodes_file = root / "graph" / "hb" / fha.HB_VERSION_ID / "nodes.parquet"
    nodes_file.unlink()
    try:
        nodes_file.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation requires OS privileges")

    def point_at_outside(m):
        m["integrity"]["files"][0]["sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
        m["integrity"]["files"][0]["size_bytes"] = outside.stat().st_size

    _rewrite_manifest(root, point_at_outside)

    err = _resolve_error(root)
    assert err.code == fha.HbAdapterErrorCode.ROOT_OUT_OF_BOUNDS.value
    assert err.message == "Integrity file path escapes artifact root: nodes.parquet"


@pytest.mark.parametrize("suffix", [".partial", ".tmp", ".inprogress"])
def test_partial_artifacts_are_not_accepted_as_integrity_files(tmp_path, suffix):
    # The partial file exists and its hash and size are right, so without the
    # suffix guard the snapshot would resolve on a half-written artifact.
    root = tmp_path / "flybrain-root"
    _write_manifest(root)
    graph_root = root / "graph" / "hb" / fha.HB_VERSION_ID
    partial = graph_root / f"edges.parquet{suffix}"
    partial.write_bytes(b"half-written")

    def add_partial(m):
        m["integrity"]["files"].append({
            "relative_path": partial.name,
            "sha256": hashlib.sha256(partial.read_bytes()).hexdigest(),
            "size_bytes": partial.stat().st_size,
        })

    _rewrite_manifest(root, add_partial)

    err = _resolve_error(root)
    assert err.code == fha.HbAdapterErrorCode.MANIFEST_INVALID.value
    assert err.message == f"Partial artifact path not permitted in integrity.files: {partial.name}"


def test_rollback_refresh_decision_marks_the_snapshot_rollback_required(tmp_path):
    root = tmp_path / "flybrain-root"
    _write_manifest(root, refresh_decision="rollback")

    snapshot = fha.build_hb_adapter(root_override=root).resolve_snapshot()
    assert snapshot.refresh_decision == "rollback"
    assert snapshot.rollback_required is True


def test_elapsed_next_check_due_requires_rollback(tmp_path):
    root = tmp_path / "flybrain-root"
    _write_manifest(root, next_check_due="2000-01-01T00:00:00+00:00")

    err = _resolve_error(root)
    assert err.code == fha.HbAdapterErrorCode.ROLLBACK_REQUIRED.value
    assert err.rollback_required is True


def test_the_valid_fixture_resolves_cleanly(tmp_path):
    # Positive twin for every negative test above: the unmodified fixture passes
    # all of the same guards.
    root = tmp_path / "flybrain-root"
    _write_manifest(root)
    snapshot = fha.build_hb_adapter(root_override=root).resolve_snapshot()
    assert (snapshot.verification_status, snapshot.refresh_decision, snapshot.rollback_required) == (
        "verified", "no_change", False)
