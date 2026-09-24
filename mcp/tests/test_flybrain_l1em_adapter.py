import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_l1em_adapter as l1em  # noqa: E402
from flybrain_l1em_adapter import L1emAdapterError, L1emErrorCode  # noqa: E402

# (skid, celltype, hemisphere, pair_skid, cluster, annotation)
FIXTURE_NEURONS = [
    (101, "KC", "left", 102, "3", "KC"),
    (102, "KC", "right", 101, "3", "KC"),
    (103, "KC", "left", 104, "3", "KC"),
    (104, "KC", "right", 103, "3", "KC"),
    (105, "PN", "left", 106, "7", "olfactory 2nd_order PN"),
    (106, "PN", "right", 105, "7", "olfactory 2nd_order PN"),
    (107, "PN", "left", None, "7", "no official annotation"),
    (108, "PN", "right", None, "8", "no official annotation"),
    (109, "pre-DN-VNC", "left", 110, "12", "no official annotation"),
    (110, "pre-DN-VNC", "right", 109, "12", "no official annotation"),
    (111, "pre-DN-VNC", "left", 112, "12", "no official annotation"),
    (112, "pre-DN-VNC", "right", 111, "12", "no official annotation"),
]
UNANNOTATED = [113]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix(skids):
    # Deterministic synthetic counts; row = presynaptic.
    return [[((a * 7 + b * 3) % 11) * (1 + (a % 4)) if a != b else 0 for b in skids] for a in skids]


def write_l1em_fixture(tmp_path: Path, *, manifest_overrides=None, skip_manifest=False, license_spdx="CC-BY-4.0"):
    """Write a synthetic l1em snapshot under <tmp>/snapshots/l1em/catmaid_l1em."""
    root = tmp_path / "snapshots" / "l1em" / "catmaid_l1em"
    files_dir = root / "metadata" / "files"
    files_dir.mkdir(parents=True)
    skids = [n[0] for n in FIXTURE_NEURONS] + UNANNOTATED
    matrix = _matrix(skids)
    with (files_dir / "all-all_connectivity_matrix.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + skids)
        for skid, row in zip(skids, matrix):
            writer.writerow([skid] + [float(v) for v in row])
    with (files_dir / "Supplementary_Data_S2.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["left_id", "right_id", "celltype", "additional_annotations", "level_7_cluster"])
        done = set()
        for skid, celltype, side, pair, cluster, annotation in FIXTURE_NEURONS:
            if skid in done:
                continue
            left, right = (skid, pair) if side == "left" else (pair, skid)
            writer.writerow([left or "no pair", right or "no pair", celltype, annotation, cluster])
            done.update({skid, pair})
    with (files_dir / "inputs.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "axon_input", "dendrite_input"])
        for skid in skids:
            writer.writerow([skid, float(skid % 5), float(10 + skid % 3)])
    with (files_dir / "outputs.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "axon_output", "dendrite_output"])
        for skid in skids:
            writer.writerow([skid, float(20 + skid % 7), float(skid % 2)])
    if skip_manifest:
        return root
    entries = []
    for rel in sorted(l1em.L1EM_REQUIRED_FILES.values()):
        path = root / rel
        entries.append({"relative_path": rel, "sha256": _sha(path), "size_bytes": path.stat().st_size})
    now = datetime.now(timezone.utc)
    manifest = {
        "schema_version": "fbh-manifest/v1",
        "manifest_id": "fbh-l1em-catmaid_l1em-fixture",
        "generated_at": now.isoformat(),
        "storage_root_env": "LOCI_FLYBRAIN_STORAGE_ROOT",
        "artifact": {"kind": "dataset_snapshot", "relative_root": "snapshots/l1em/catmaid_l1em", "path_template": "x"},
        "dataset": {
            "symbol": "l1em",
            "version_id": "catmaid_l1em",
            "source": {
                "system": "fixture",
                "access_method": "fixture",
                "uri": "fixture://l1em",
                "retrieved_at": now.isoformat(),
                "license": {"spdx_id": license_spdx},
                "citation": "Winding et al. 2023 Science doi:10.1126/science.add9330",
            },
        },
        "scope": {"sex": "mixed_or_unspecified", "stage": "larva_l1", "anatomy": "x", "evidence_family": "x", "claim_tier": "x"},
        "integrity": {"manifest_sha256": "", "files": entries, "verification": {"status": "verified", "verified_at": now.isoformat()}},
        "refresh": {"decision": "no_change", "checked_at": now.isoformat(), "next_check_due": (now + timedelta(days=30)).isoformat()},
        "lineage": {"derived_from": [{"type": "fixture", "id": "x"}], "pipeline": {"job_name": "x", "job_version": "x", "run_id": "x"}},
    }
    for dotted, value in (manifest_overrides or {}).items():
        target = manifest
        keys = dotted.split(".")
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    manifest["integrity"]["manifest_sha256"] = l1em.manifest_digest(manifest)
    (root / "manifest").mkdir()
    (root / "manifest" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "manifest" / "manifest.sha256").write_text(manifest["integrity"]["manifest_sha256"] + "\n", encoding="utf-8")
    return root


def _persist_manifest(path: Path, data) -> None:
    """Rewrite a re-hashed manifest together with its sidecar."""
    path.write_text(json.dumps(data))
    (path.parent / "manifest.sha256").write_text(data["integrity"]["manifest_sha256"] + "\n")


def _code(excinfo) -> L1emErrorCode:
    return excinfo.value.code


def test_load_snapshot_parses_neurons(tmp_path):
    write_l1em_fixture(tmp_path)
    snap = l1em.load_l1em_snapshot(tmp_path)
    assert snap.matrix_rows == 13
    assert [n.skid for n in snap.neurons] == sorted(n[0] for n in FIXTURE_NEURONS)
    assert snap.unannotated_skids == (113,)
    kc = snap.neuron(101)
    assert kc.hemisphere == "left" and kc.pair_skid == 102 and kc.region_id == "l1_kc"
    assert snap.neuron(107).paired is False
    skids = [n[0] for n in FIXTURE_NEURONS] + UNANNOTATED
    matrix = _matrix(skids)
    i = skids.index(105)
    assert snap.neuron(105).out_synapses == sum(matrix[i])
    assert snap.neuron(105).in_synapses == sum(row[i] for row in matrix)
    assert snap.license_spdx == "CC-BY-4.0"


def test_storage_root_resolution_uses_registry_layout(tmp_path):
    write_l1em_fixture(tmp_path)
    assert l1em.resolve_l1em_snapshot_root(tmp_path) == (tmp_path / "snapshots" / "l1em" / "catmaid_l1em").resolve()


def test_missing_manifest_fails_closed(tmp_path):
    write_l1em_fixture(tmp_path, skip_manifest=True)
    with pytest.raises(L1emAdapterError) as excinfo:
        l1em.load_l1em_snapshot(tmp_path)
    assert _code(excinfo) is L1emErrorCode.MANIFEST_MISSING
    assert str(excinfo.value).startswith("[MANIFEST_MISSING] ")


def test_tampered_file_is_integrity_mismatch(tmp_path):
    root = write_l1em_fixture(tmp_path)
    with (root / "metadata" / "files" / "inputs.csv").open("a") as handle:
        handle.write("999,1.0,1.0\n")
    with pytest.raises(L1emAdapterError) as excinfo:
        l1em.load_l1em_snapshot(tmp_path)
    assert _code(excinfo) is L1emErrorCode.INTEGRITY_MISMATCH


def test_manifest_hash_tamper_detected(tmp_path):
    root = write_l1em_fixture(tmp_path)
    path = root / "manifest" / "manifest.json"
    data = json.loads(path.read_text())
    data["notes"] = ["edited after hashing"]
    path.write_text(json.dumps(data))
    with pytest.raises(L1emAdapterError) as excinfo:
        l1em.load_l1em_snapshot(tmp_path)
    assert _code(excinfo) is L1emErrorCode.INTEGRITY_MISMATCH


@pytest.mark.parametrize(
    "overrides, code",
    [
        ({"dataset.symbol": "fafb"}, L1emErrorCode.DATASET_PIN_MISMATCH),
        ({"artifact.relative_root": "snapshots/fw/flywire783"}, L1emErrorCode.DATASET_PIN_MISMATCH),
        ({"artifact.relative_root": "../escape"}, L1emErrorCode.PATH_ESCAPE),
        ({"scope.stage": "adult"}, L1emErrorCode.CROSS_STAGE_UNSUPPORTED),
        ({"integrity.verification.status": "pending"}, L1emErrorCode.INTEGRITY_MISMATCH),
        ({"refresh.decision": "rollback"}, L1emErrorCode.MANIFEST_STALE),
        ({"refresh.next_check_due": "2000-01-01T00:00:00Z"}, L1emErrorCode.MANIFEST_STALE),
        ({"dataset.source.citation": ""}, L1emErrorCode.MANIFEST_INVALID),
        ({"schema_version": "fbh-manifest/v0"}, L1emErrorCode.MANIFEST_INVALID),
    ],
)
def test_manifest_guards(tmp_path, overrides, code):
    write_l1em_fixture(tmp_path, manifest_overrides=overrides)
    with pytest.raises(L1emAdapterError) as excinfo:
        l1em.load_l1em_snapshot(tmp_path)
    assert _code(excinfo) is code


def test_unreviewed_license_refused(tmp_path):
    write_l1em_fixture(tmp_path, license_spdx="UNREVIEWED")
    with pytest.raises(L1emAdapterError) as excinfo:
        l1em.load_l1em_snapshot(tmp_path)
    assert _code(excinfo) is L1emErrorCode.LICENSE_UNVERIFIED


def test_required_file_must_be_listed(tmp_path):
    root = write_l1em_fixture(tmp_path)
    path = root / "manifest" / "manifest.json"
    data = json.loads(path.read_text())
    data["integrity"]["files"] = [f for f in data["integrity"]["files"] if "outputs" not in f["relative_path"]]
    data["integrity"]["manifest_sha256"] = l1em.manifest_digest(data)
    _persist_manifest(path, data)
    with pytest.raises(L1emAdapterError) as excinfo:
        l1em.load_l1em_snapshot(tmp_path)
    assert _code(excinfo) is L1emErrorCode.REQUIRED_FILE_MISSING


def _rewrite_file_and_rehash(root: Path, rel: str, text: str) -> None:
    (root / rel).write_text(text)
    path = root / "manifest" / "manifest.json"
    data = json.loads(path.read_text())
    for entry in data["integrity"]["files"]:
        if entry["relative_path"] == rel:
            entry["sha256"] = _sha(root / rel)
            entry["size_bytes"] = (root / rel).stat().st_size
    data["integrity"]["manifest_sha256"] = l1em.manifest_digest(data)
    _persist_manifest(path, data)


def test_non_square_matrix_rejected(tmp_path):
    root = write_l1em_fixture(tmp_path)
    _rewrite_file_and_rehash(root, "metadata/files/all-all_connectivity_matrix.csv", ",1,2\n1,0.0,1.0\n3,1.0,0.0\n")
    with pytest.raises(L1emAdapterError) as excinfo:
        l1em.load_l1em_snapshot(tmp_path)
    assert _code(excinfo) is L1emErrorCode.MATRIX_INVALID


def test_negative_matrix_values_rejected(tmp_path):
    root = write_l1em_fixture(tmp_path)
    _rewrite_file_and_rehash(root, "metadata/files/all-all_connectivity_matrix.csv", ",1,2\n1,0.0,-1.0\n2,1.0,0.0\n")
    with pytest.raises(L1emAdapterError) as excinfo:
        l1em.load_l1em_snapshot(tmp_path)
    assert _code(excinfo) is L1emErrorCode.MATRIX_INVALID


def test_annotation_schema_and_conflicts(tmp_path):
    root = write_l1em_fixture(tmp_path)
    rel = "metadata/files/Supplementary_Data_S2.csv"
    _rewrite_file_and_rehash(root, rel, "left_id,right_id,celltype\n101,102,KC\n")
    with pytest.raises(L1emAdapterError) as excinfo:
        l1em.load_l1em_snapshot(tmp_path)
    assert _code(excinfo) is L1emErrorCode.SCHEMA_MISMATCH

    header = "left_id,right_id,celltype,additional_annotations,level_7_cluster\n"
    _rewrite_file_and_rehash(root, rel, header + "101,102,KC,KC,3\n101,no pair,KC,KC,3\n")
    with pytest.raises(L1emAdapterError) as excinfo:
        l1em.load_l1em_snapshot(tmp_path)
    assert _code(excinfo) is L1emErrorCode.ANNOTATION_CONFLICT


def test_region_vocabulary_is_larval_only():
    assert l1em.larval_region_id("pre-DN-VNC") == "l1_pre_dn_vnc"
    assert l1em.assert_larval_region("l1_kc") == "l1_kc"
    for adult in ("MB_CA_R", "fw_mb", "l1_"):
        with pytest.raises(L1emAdapterError) as excinfo:
            l1em.assert_larval_region(adult)
        assert excinfo.value.code is L1emErrorCode.CROSS_STAGE_UNSUPPORTED
    with pytest.raises(L1emAdapterError) as excinfo:
        l1em.map_to_adult_neuropil("l1_kc")
    assert excinfo.value.code is L1emErrorCode.CROSS_STAGE_UNSUPPORTED


def test_scope_validation_refuses_cross_stage():
    assert l1em.validate_scope({"stage": "larva_l1"})["allowed"] is True
    for scope in ({"stage": "adult"}, {"stage": "larva_l1", "target_stage": "adult"}, {"region_vocabulary": "flywire_neuropil"}):
        with pytest.raises(L1emAdapterError) as excinfo:
            l1em.validate_scope(scope)
        assert excinfo.value.code is L1emErrorCode.CROSS_STAGE_UNSUPPORTED


def test_unsupported_objectives_are_explicit():
    assert l1em.require_supported_objective("connectivity_tier") == "connectivity_tier"
    for objective in ("neurotransmitter_dominance", "region_specialization_tier", "bogus"):
        with pytest.raises(L1emAdapterError) as excinfo:
            l1em.require_supported_objective(objective)
        assert excinfo.value.code is L1emErrorCode.OBJECTIVE_UNSUPPORTED


_REAL_ROOT = os.environ.get("LOCI_FLYBRAIN_STORAGE_ROOT", "")


@pytest.mark.skipif(
    not _REAL_ROOT
    or not (Path(_REAL_ROOT) / "snapshots" / "l1em" / "catmaid_l1em" / "manifest" / "manifest.json").is_file(),
    reason="real l1em snapshot not available",
)
def test_real_snapshot_smoke():
    snap = l1em.load_l1em_snapshot(_REAL_ROOT)
    assert snap.matrix_rows == 2952
    assert len(snap.neurons) + len(snap.unannotated_skids) == 2952
    assert all(n.region_id.startswith("l1_") for n in snap.neurons)


def test_missing_sidecar_fails_closed(tmp_path):
    root = write_l1em_fixture(tmp_path)
    (root / "manifest" / "manifest.sha256").unlink()
    with pytest.raises(L1emAdapterError) as excinfo:
        l1em.load_l1em_snapshot(tmp_path)
    assert _code(excinfo) is L1emErrorCode.MANIFEST_MISSING


def test_rehashed_manifest_without_matching_sidecar_rejected(tmp_path):
    root = write_l1em_fixture(tmp_path)
    path = root / "manifest" / "manifest.json"
    data = json.loads(path.read_text())
    data["notes"] = ["edited and self-hash recomputed"]
    data["integrity"]["manifest_sha256"] = l1em.manifest_digest(data)
    path.write_text(json.dumps(data))
    with pytest.raises(L1emAdapterError) as excinfo:
        l1em.load_l1em_snapshot(tmp_path)
    assert _code(excinfo) is L1emErrorCode.INTEGRITY_MISMATCH


def test_missing_size_bytes_rejected_even_without_hash_verification(tmp_path):
    root = write_l1em_fixture(tmp_path)
    path = root / "manifest" / "manifest.json"
    data = json.loads(path.read_text())
    for entry in data["integrity"]["files"]:
        entry.pop("size_bytes", None)
    data["integrity"]["manifest_sha256"] = l1em.manifest_digest(data)
    _persist_manifest(path, data)
    with pytest.raises(L1emAdapterError) as excinfo:
        l1em.load_l1em_snapshot(tmp_path, verify_integrity=False)
    assert _code(excinfo) is L1emErrorCode.MANIFEST_INVALID
