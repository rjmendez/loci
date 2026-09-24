import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_ol_adapter as ol  # noqa: E402
from flybrain_ol_adapter import OlAdapterError, OlAdapterErrorCode  # noqa: E402


def roi(pre, post):
    return {"pre": pre, "post": post, "downstream": pre * 3, "upstream": post, "synweight": pre + post}


def roi_info(**by_roi):
    """``roi_info(ME_R=(10, 90), ...)`` -> neuPrint roiInfo JSON with OL(R) aggregate + extras."""
    info = {}
    for key, value in by_roi.items():
        name = key.replace("_R", "(R)").replace("_L", "(L)") if key[-2:] in ("_R", "_L") else key
        info[name] = roi(*value) if isinstance(value, tuple) else value
    ol_total = [info[k] for k in info if k in {"ME(R)", "LO(R)", "LOP(R)", "AME(R)", "LA(R)"}]
    if ol_total:
        info["OL(R)"] = roi(sum(v["pre"] for v in ol_total), sum(v["post"] for v in ol_total))
    return json.dumps(info, sort_keys=True)


# (body, type, instance, status, pre, post, downstream, predictedNt, conf, totalNt, hemilineage, soma, hex, roiInfo)
_NEURONS = [
    (10001, "Mi1", "Mi1_R", "Traced", 120, 300, 500, "acetylcholine", 0.9, 110.0, None, "{x:1, y:2, z:3}", 10.0,
     roi_info(ME_R=(120, 300), ME_R_layer_05={"synweight": 200}, ME_R_layer_09={"synweight": 150},
              ME_R_col_10_12={"post": 3, "synweight": 3})),
    (10002, "Mi1", "Mi1_R", "Traced", 12, 60, 20, "acetylcholine", 0.8, 11.0, None, "{x:1, y:2, z:4}", 11.0,
     roi_info(ME_R=(12, 60), ME_R_layer_05={"synweight": 50})),
    (10003, "Tm1", "Tm1_R", "Traced", 200, 150, 900, "acetylcholine", 0.7, 190.0, None, "{x:1, y:2, z:5}", 12.0,
     roi_info(ME_R=(40, 100), LO_R=(160, 50), LO_R_layer_2={"synweight": 120}, ME_R_layer_01={"synweight": 90})),
    (10004, "Tm2", "Tm2_R", "Traced", 30, 40, 30, "glutamate", 0.6, 28.0, None, "{x:1, y:2, z:6}", 13.0,
     roi_info(LO_R=(30, 40), LO_R_layer_1={"synweight": 70})),
    (10005, "T4a", "T4a_R", "Traced", 250, 80, 800, "acetylcholine", 0.95, 240.0, None, "{x:1, y:2, z:7}", None,
     roi_info(LOP_R=(250, 80), LOP_R_layer_1={"synweight": 300})),
    (10006, "T4b", "T4b_R", "Traced", 3, 20, 5, "gaba", 0.5, 3.0, None, "{x:1, y:2, z:8}", None,
     roi_info(LOP_R=(3, 20), LOP_R_layer_2={"synweight": 23})),
    (10007, "LC4", "LC4_R", "Traced", 500, 700, 1500, "acetylcholine", 0.88, 480.0, "VLPl2_lateral", None, None,
     roi_info(LO_R=(20, 400), PVLP_R=(480, 100), LO_R_layer_4={"synweight": 500})),
    (10008, "LPLC2", "LPLC2_R", "Traced", 90, 400, 60, "unclear", 0.3, 85.0, "VLPl2_lateral", "{x:1, y:2, z:9}", None,
     roi_info(PVLP_R=(80, 50), LOP_R=(10, 350))),
    (10009, None, None, "", 5, 5, 5, "gaba", 0.9, 5.0, None, None, None, roi_info(ME_R=(5, 5))),  # untyped
    (10010, "Mi9", "Mi9_R", "Orphan", 50, 50, 70, "glutamate", 0.9, 45.0, None, None, 14.0,
     roi_info(ME_R=(50, 50))),  # status filtered
    (10011, "C3", "C3_R", "Traced", 110, 120, 300, "gaba", 0.92, 100.0, None, "{x:1, y:2, z:10}", None,
     roi_info(ME_R=(100, 110), LA_R=(10, 10), ME_R_layer_10={"synweight": 80})),
    (10012, "Dm3", "Dm3_R", "Traced", 40, 30, 90, "glutamate", 0.7, 35.0, None, "{x:1, y:2, z:11}", None,
     json.dumps({"OL(R)": roi(40, 30), "ME_R_col_01_02": {"pre": 1, "synweight": 1}})),  # no primary ROI
    (10013, "", "x", "Traced", 5, 5, 5, "gaba", 0.9, 5.0, None, None, None, roi_info(ME_R=(5, 5))),  # empty type
    (10014, "Mi4", "Mi4_R", "Traced", 25, 90, 40, "gaba", 0.75, 22.0, None, None, 15.0,
     roi_info(ME_R=(25, 90), ME_R_layer_05={"synweight": 60})),
    (10015, "CT1", "CT1_L", "Traced", 900, 2000, 2000, "gaba", 0.78, 880.0, None, "{x:9, y:9, z:9}", None,
     roi_info(ME_L=(400, 900), LO_R=(500, 1100), LO_R_layer_5={"synweight": 900})),
]
_EXTRA_COLUMNS = {
    "consensusNt:string": "acetylcholine",
    "celltypePredictedNt:string": "acetylcholine",
    "size:long": 12345,
    ":LABEL": "Segment;Neuron",
}


def write_neurons(path, rows=_NEURONS, *, drop=(), rename=None):
    cols = list(zip(*rows))
    data = {
        "bodyId:long": pa.array(cols[0], pa.uint64()),
        "type:string": list(cols[1]),
        "instance:string": list(cols[2]),
        "status:string": list(cols[3]),
        "pre:int": pa.array(cols[4], pa.int32()),
        "post:int": pa.array(cols[5], pa.int32()),
        "downstream:int": pa.array(cols[6], pa.int32()),
        "upstream:int": pa.array(cols[5], pa.int32()),
        "predictedNt:string": list(cols[7]),
        "predictedNtConfidence:float": pa.array(cols[8], pa.float64()),
        "totalNtPredictions:float": pa.array(cols[9], pa.float64()),
        "hemilineage:string": pa.array(cols[10], pa.string()),
        "somaLocation:point{srid:9157}": pa.array(cols[11], pa.string()),
        "assignedOlHex1:float": pa.array(cols[12], pa.float64()),
        "roiInfo:string": list(cols[13]),
    }
    for name, value in _EXTRA_COLUMNS.items():
        data[name] = [value] * len(rows)
    for name in drop:
        data.pop(name)
    if rename:
        for old, new in rename.items():
            data[new] = data.pop(old)
    feather.write_feather(pa.table(data), str(path), chunksize=4)


def write_meta(path, *, dataset="optic-lobe", tag="v1.1", primary=None):
    rois = sorted(ol.OL_PRIMARY_ROIS) if primary is None else primary
    header = "dataset:string,tag:string,voxelSize:float[],primaryRois:string[],roiInfo:string"
    row = f'{dataset},{tag},8.0;8.0;8.0,{";".join(rois)},"{{""ME(R)"": {{""pre"": 1}}}}"'
    Path(path).write_text(header + "\n" + row + "\n", encoding="utf-8")


def build_snapshot(storage_root, *, manifest_overrides=None, write_manifest=True, neuron_rows=_NEURONS):
    """Create a synthetic ol snapshot + verified manifest under storage_root."""
    snap = Path(storage_root) / "snapshots" / "ol" / "optic_lobe_v1.1"
    (snap / "metadata").mkdir(parents=True, exist_ok=True)
    write_neurons(snap / ol.OL_PRODUCT_PATHS[ol.ROLE_NEURONS], rows=neuron_rows)
    write_meta(snap / ol.OL_PRODUCT_PATHS[ol.ROLE_NEUPRINT_META])
    (snap / "metadata" / "optic-lobe-v1.1.json").write_text("{}\n", encoding="utf-8")
    files = []
    for role, rel in sorted(ol.OL_PRODUCT_PATHS.items()):
        path = snap / rel
        files.append({"relative_path": rel, "size_bytes": path.stat().st_size, "sha256": ol.sha256_file(path),
                      "role": role})
    descriptor = snap / "metadata" / "optic-lobe-v1.1.json"
    files.append({"relative_path": "metadata/optic-lobe-v1.1.json", "size_bytes": descriptor.stat().st_size,
                  "sha256": ol.sha256_file(descriptor), "role": "release_descriptor"})
    manifest = ol.build_ol_manifest(
        files=files,
        source_uri="synthetic://ol",
        retrieved_at="2026-09-24T00:00:00Z",
        run_id="run-test",
        generated_at="2026-09-24T00:00:00Z",
        next_check_due="2099-01-01T00:00:00Z",
    )
    if manifest_overrides:
        manifest_overrides(manifest)
        manifest["integrity"]["manifest_sha256"] = ol.manifest_digest(manifest)
    if write_manifest:
        ol.write_ol_manifest(snap, manifest)
    return snap, manifest


def _code(exc_info):
    return exc_info.value.code


# ---------------------------------------------------------------------------
# ROI / NT vocabularies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,division,neuropil,side,slug,region_id",
    [
        ("ME(R)", "optic_lobe", "ME", "right", "me_r", "ol_me_r"),
        ("LOP(L)", "optic_lobe", "LOP", "left", "lop_l", "ol_lop_l"),
        ("PLP(R)", "central_brain", "PLP", "right", "plp_r", "cb_plp_r"),
        ("AL(R)", "central_brain", "AL", "right", "al_r", "cb_al_r"),
        ("aL(R)", "central_brain", "aL", "right", "mb_al_r", "cb_mb_al_r"),
        ("a'L(L)", "central_brain", "a'L", "left", "mb_aprimel_l", "cb_mb_aprimel_l"),
        ("GNG", "central_brain", "GNG", None, "gng", "cb_gng"),
        ("vnc-shell", "ventral_nerve_cord", "vnc-shell", None, "vnc_shell", "vnc_vnc_shell"),
    ],
)
def test_map_ol_roi_explicit_vocabulary(raw, division, neuropil, side, slug, region_id):
    mapped = ol.map_ol_roi(raw)
    assert (mapped.division, mapped.neuropil, mapped.side, mapped.slug, mapped.region_id) == (
        division, neuropil, side, slug, region_id,
    )


@pytest.mark.parametrize("raw", ["", "OL(R)", "ME", "ME(X)", "ME_R_layer_01", "FB(R)", "LO(r)"])
def test_map_ol_roi_unknown_fails_closed(raw):
    with pytest.raises(OlAdapterError) as exc:
        ol.map_ol_roi(raw)
    assert _code(exc) is OlAdapterErrorCode.REGION_VOCABULARY_UNKNOWN
    assert str(exc.value).startswith("[REGION_VOCABULARY_UNKNOWN] ")


def test_primary_roi_vocabulary_matches_release_size_and_slugs_unique():
    assert len(ol.OL_PRIMARY_ROIS) == 89  # Neuprint_Meta.csv primaryRois (optic-lobe v1.1)
    slugs = [ol.map_ol_roi(r).slug for r in ol.OL_PRIMARY_ROIS]
    assert len(set(slugs)) == len(slugs)


def test_parse_roi_info_splits_primary_layers_columns():
    summary = ol.parse_roi_info(_NEURONS[2][13])  # Tm1
    assert set(summary.primary) == {"ME(R)", "LO(R)"}
    assert summary.dominant("synweight") == "LO(R)"
    assert summary.dominant("post") == "ME(R)"
    assert summary.dominant("pre") == "LO(R)"
    assert summary.arbor(0.1) == ("LO(R)", "ME(R)")
    assert summary.dominant_layer() == "LO_R_layer_2"
    assert ol.layer_slug("LO_R_layer_2") == "lo_r_layer_02"
    assert ol.parse_roi_info(_NEURONS[0][13]).n_columns == 1
    empty = ol.parse_roi_info(None)
    assert empty.dominant("synweight") is None and empty.arbor(0.1) == ()


def test_parse_roi_info_tie_break_is_by_name():
    summary = ol.parse_roi_info(json.dumps({"LO(R)": roi(5, 5), "ME(R)": roi(5, 5)}))
    assert summary.dominant("synweight") == "LO(R)"


@pytest.mark.parametrize(
    "raw,code",
    [
        ("not json", OlAdapterErrorCode.SCHEMA_MISMATCH),
        ("[1, 2]", OlAdapterErrorCode.SCHEMA_MISMATCH),
        (json.dumps({"ME(R)": 5}), OlAdapterErrorCode.SCHEMA_MISMATCH),
        (json.dumps({"ME(R)": {"pre": -1}}), OlAdapterErrorCode.SCHEMA_MISMATCH),
        (json.dumps({"ME(R)": {"pre": 1.5}}), OlAdapterErrorCode.SCHEMA_MISMATCH),
        (json.dumps({"ME(R)": {"pre": "7"}}), OlAdapterErrorCode.SCHEMA_MISMATCH),
        (json.dumps({"LA_R_layer_1": {"synweight": 1}}), OlAdapterErrorCode.REGION_VOCABULARY_UNKNOWN),
        (json.dumps({"Medulla": {"pre": 1}}), OlAdapterErrorCode.REGION_VOCABULARY_UNKNOWN),
    ],
)
def test_parse_roi_info_fails_closed(raw, code):
    with pytest.raises(OlAdapterError) as exc:
        ol.parse_roi_info(raw)
    assert _code(exc) is code


def test_nt_short_codes():
    assert ol.nt_short_code("Acetylcholine") == "ach"
    assert ol.nt_short_code("histamine") == "his"
    for bad in ("unclear", "nitric_oxide", ""):
        with pytest.raises(OlAdapterError) as exc:
            ol.nt_short_code(bad)
        assert _code(exc) is OlAdapterErrorCode.NT_VOCABULARY_UNKNOWN


# ---------------------------------------------------------------------------
# Snapshot / manifest
# ---------------------------------------------------------------------------


def test_open_snapshot_verified(tmp_path):
    snap, manifest = build_snapshot(tmp_path)
    opened = ol.open_ol_snapshot(tmp_path)
    assert opened.snapshot_root == snap.resolve()
    assert opened.manifest_sha256 == manifest["integrity"]["manifest_sha256"]
    assert opened.license_spdx == "CC-BY-4.0"
    assert set(opened.product_paths) == set(ol.OL_PRODUCT_PATHS)
    prov = opened.provenance()
    assert prov["dataset_symbol"] == "ol" and prov["version_id"] == "optic_lobe_v1.1"
    assert opened.hash_verification["metadata/Neuprint_Neurons.feather"] == "hashed"
    assert opened.hash_verification["metadata/optic-lobe-v1.1.json"] == "size_only"


def test_hash_stamps_skip_rehash_until_file_changes(tmp_path):
    snap, _ = build_snapshot(tmp_path)
    ol.open_ol_snapshot(tmp_path)
    second = ol.open_ol_snapshot(tmp_path)
    assert second.hash_verification["metadata/Neuprint_Neurons.feather"] == "stamp"
    forced = ol.open_ol_snapshot(tmp_path, verify_hashes=True)
    assert set(forced.hash_verification.values()) == {"hashed"}
    size_only = ol.open_ol_snapshot(tmp_path, verify_hashes=False)
    assert set(size_only.hash_verification.values()) == {"size_only"}
    stamps = list((snap / "manifest" / "hash-stamps").glob("*.json"))
    assert len(stamps) == 3  # 2 products + descriptor (forced verify)


def test_same_size_tamper_detected_after_mtime_change(tmp_path):
    snap, _ = build_snapshot(tmp_path)
    ol.open_ol_snapshot(tmp_path)
    meta_path = snap / ol.OL_PRODUCT_PATHS[ol.ROLE_NEUPRINT_META]
    data = meta_path.read_bytes().replace(b"v1.1", b"v1.2")
    meta_path.write_bytes(data)
    stat = meta_path.stat()
    os.utime(meta_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(OlAdapterError) as exc:
        ol.open_ol_snapshot(tmp_path)
    assert _code(exc) is OlAdapterErrorCode.INTEGRITY_MISMATCH


def test_open_snapshot_missing_manifest(tmp_path):
    build_snapshot(tmp_path, write_manifest=False)
    with pytest.raises(OlAdapterError) as exc:
        ol.open_ol_snapshot(tmp_path)
    assert _code(exc) is OlAdapterErrorCode.MANIFEST_MISSING


def test_open_snapshot_missing_dir(tmp_path):
    with pytest.raises(OlAdapterError) as exc:
        ol.open_ol_snapshot(tmp_path)
    assert _code(exc) is OlAdapterErrorCode.SNAPSHOT_MISSING


def test_missing_sidecar_fails_closed(tmp_path):
    snap, _ = build_snapshot(tmp_path)
    (snap / "manifest" / "manifest.sha256").unlink()
    with pytest.raises(OlAdapterError) as exc:
        ol.open_ol_snapshot(tmp_path)
    assert _code(exc) is OlAdapterErrorCode.MANIFEST_MISSING


def test_sidecar_mismatch(tmp_path):
    snap, _ = build_snapshot(tmp_path)
    (snap / "manifest" / "manifest.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(OlAdapterError) as exc:
        ol.open_ol_snapshot(tmp_path)
    assert _code(exc) is OlAdapterErrorCode.INTEGRITY_MISMATCH


def test_manifest_self_hash_mismatch(tmp_path):
    snap, manifest = build_snapshot(tmp_path, write_manifest=False)
    manifest["notes"] = ["edited after hashing"]
    ol.write_ol_manifest(snap, manifest)
    with pytest.raises(OlAdapterError) as exc:
        ol.open_ol_snapshot(tmp_path)
    assert _code(exc) is OlAdapterErrorCode.INTEGRITY_MISMATCH


def _files(m):
    return m["integrity"]["files"]


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda m: m["dataset"]["source"]["license"].update(spdx_id="UNREVIEWED"), OlAdapterErrorCode.LICENSE_NOT_PERMITTED),
        (lambda m: m["dataset"].update(version_id="optic_lobe_v1.0"), OlAdapterErrorCode.DATASET_PIN_MISMATCH),
        (lambda m: m["dataset"].update(symbol="mc"), OlAdapterErrorCode.DATASET_PIN_MISMATCH),
        (lambda m: m["integrity"]["verification"].update(status="pending"), OlAdapterErrorCode.INTEGRITY_MISMATCH),
        (lambda m: m["refresh"].update(decision="rollback"), OlAdapterErrorCode.PROMOTION_STATE_INVALID),
        (lambda m: m["refresh"].update(next_check_due="2020-01-01T00:00:00Z"), OlAdapterErrorCode.PROMOTION_STATE_INVALID),
        (lambda m: m["artifact"].update(relative_root="snapshots/mc/male-cns_v1.0"), OlAdapterErrorCode.DATASET_PIN_MISMATCH),
        (lambda m: _files(m).append(dict(_files(m)[0], relative_path="../outside.csv")), OlAdapterErrorCode.PATH_ESCAPE),
        (lambda m: _files(m).append(dict(_files(m)[0], relative_path="x.feather.partial")), OlAdapterErrorCode.MANIFEST_INVALID),
        (lambda m: _files(m).append(dict(_files(m)[0], relative_path=_files(m)[0]["relative_path"].upper())),
         OlAdapterErrorCode.MANIFEST_INVALID),
        (lambda m: _files(m).pop(0), OlAdapterErrorCode.PRODUCT_MISSING),
        (lambda m: _files(m)[0].update(size_bytes=1), OlAdapterErrorCode.INTEGRITY_MISMATCH),
        (lambda m: _files(m)[0].update(size_bytes=True), OlAdapterErrorCode.MANIFEST_INVALID),
        (lambda m: _files(m)[0].update(sha256="A" * 64), OlAdapterErrorCode.MANIFEST_INVALID),
        (lambda m: [f.update(role="edgelist") for f in _files(m) if f["relative_path"].endswith("Neurons.feather")],
         OlAdapterErrorCode.MANIFEST_INVALID),
        (lambda m: m.update(schema_version="fbh-manifest/v0"), OlAdapterErrorCode.MANIFEST_INVALID),
    ],
)
def test_manifest_gates_fail_closed(tmp_path, mutate, code):
    build_snapshot(tmp_path, manifest_overrides=mutate)
    with pytest.raises(OlAdapterError) as exc:
        ol.open_ol_snapshot(tmp_path)
    assert _code(exc) is code


def test_tampered_product_fails_integrity(tmp_path):
    snap, _ = build_snapshot(tmp_path)
    meta_path = snap / ol.OL_PRODUCT_PATHS[ol.ROLE_NEUPRINT_META]
    meta_path.write_bytes(meta_path.read_bytes().replace(b"v1.1", b"v9.9"))
    with pytest.raises(OlAdapterError) as exc:
        ol.open_ol_snapshot(tmp_path)
    assert _code(exc) is OlAdapterErrorCode.INTEGRITY_MISMATCH


def test_required_roles_subset_and_unknown_role(tmp_path):
    build_snapshot(tmp_path)
    opened = ol.open_ol_snapshot(tmp_path, required_roles=(ol.ROLE_NEUPRINT_META,))
    assert set(opened.product_paths) == {ol.ROLE_NEUPRINT_META}
    with pytest.raises(OlAdapterError) as exc:
        opened.path(ol.ROLE_NEURONS)
    assert _code(exc) is OlAdapterErrorCode.PRODUCT_MISSING
    with pytest.raises(OlAdapterError) as exc:
        ol.open_ol_snapshot(tmp_path, required_roles=("syn_points",))
    assert _code(exc) is OlAdapterErrorCode.PRODUCT_MISSING


def test_write_manifest_never_overwrites(tmp_path):
    snap, manifest = build_snapshot(tmp_path)
    before = (snap / "manifest" / "manifest.json").read_text()
    with pytest.raises(OlAdapterError) as exc:
        ol.write_ol_manifest(snap, manifest)
    assert _code(exc) is OlAdapterErrorCode.MANIFEST_EXISTS
    assert (snap / "manifest" / "manifest.json").read_text() == before


def test_relative_storage_root_rejected():
    with pytest.raises(OlAdapterError) as exc:
        ol.open_ol_snapshot("relative/root")
    assert _code(exc) is OlAdapterErrorCode.ROOT_NOT_CONFIGURED


def test_validate_manifest_uses_injected_clock(tmp_path):
    snap, manifest = build_snapshot(tmp_path)
    late = datetime(2100, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(OlAdapterError) as exc:
        ol.validate_ol_manifest(manifest, snap, now=late)
    assert _code(exc) is OlAdapterErrorCode.PROMOTION_STATE_INVALID


def test_manifest_json_is_schema_shaped(tmp_path):
    _, manifest = build_snapshot(tmp_path)
    assert manifest["schema_version"] == "fbh-manifest/v1"
    assert manifest["artifact"]["relative_root"] == "snapshots/ol/optic_lobe_v1.1"
    assert manifest["artifact"]["path_template"].startswith("$LOCI_FLYBRAIN_STORAGE_ROOT\\")
    assert manifest["scope"]["anatomy"] == "right_optic_lobe"
    assert manifest["dataset"]["source"]["citation"] == ol.OL_CITATION


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def test_load_neurons_filters_typed_and_status(tmp_path):
    path = tmp_path / "neurons.feather"
    write_neurons(path)
    frame, source_rows, typed_rows = ol.load_ol_neurons(path)
    assert source_rows == len(_NEURONS)
    assert typed_rows == len(_NEURONS) - 2  # 10009 null type, 10013 empty type
    ids = set(frame["body_id"])
    assert "10009" not in ids and "10013" not in ids and "10010" not in ids
    assert all(isinstance(v, str) for v in frame["body_id"])
    assert list(frame.columns) == list(ol.NEURON_COLUMNS)
    widened, _, _ = ol.load_ol_neurons(path, statuses=("Traced", "Orphan"))
    assert "10010" in set(widened["body_id"])


def test_load_neurons_large_ids_stay_exact(tmp_path):
    rows = [(18446744073709551557, *_NEURONS[0][1:])]
    path = tmp_path / "big.feather"
    write_neurons(path, rows=rows)
    frame, _, _ = ol.load_ol_neurons(path)
    assert frame["body_id"].tolist() == ["18446744073709551557"]


def test_load_neurons_missing_or_renamed_column(tmp_path):
    path = tmp_path / "neurons.feather"
    write_neurons(path, drop=("roiInfo:string",))
    with pytest.raises(OlAdapterError) as exc:
        ol.load_ol_neurons(path)
    assert _code(exc) is OlAdapterErrorCode.SCHEMA_MISMATCH
    write_neurons(path, rename={"predictedNt:string": "predictedNt"})
    with pytest.raises(OlAdapterError) as exc:
        ol.load_ol_neurons(path)
    assert _code(exc) is OlAdapterErrorCode.SCHEMA_MISMATCH


def test_load_neurons_duplicate_ids(tmp_path):
    path = tmp_path / "neurons.feather"
    write_neurons(path, rows=[_NEURONS[0], _NEURONS[0]])
    with pytest.raises(OlAdapterError) as exc:
        ol.load_ol_neurons(path)
    assert _code(exc) is OlAdapterErrorCode.SCHEMA_MISMATCH


def test_load_neurons_not_feather_and_missing(tmp_path):
    bad = tmp_path / "bad.feather"
    bad.write_text("not arrow", encoding="utf-8")
    with pytest.raises(OlAdapterError) as exc:
        ol.load_ol_neurons(bad)
    assert _code(exc) is OlAdapterErrorCode.SCHEMA_MISMATCH
    with pytest.raises(OlAdapterError) as exc:
        ol.load_ol_neurons(tmp_path / "absent.feather")
    assert _code(exc) is OlAdapterErrorCode.PRODUCT_MISSING


def test_release_meta_pin_and_vocabulary(tmp_path):
    path = tmp_path / "meta.csv"
    write_meta(path)
    meta = ol.load_ol_release_meta(path)
    assert (meta.dataset, meta.tag) == ("optic-lobe", "v1.1")
    assert set(meta.primary_rois) == ol.OL_PRIMARY_ROIS
    write_meta(path, tag="v1.0")
    with pytest.raises(OlAdapterError) as exc:
        ol.load_ol_release_meta(path)
    assert _code(exc) is OlAdapterErrorCode.DATASET_PIN_MISMATCH
    write_meta(path, primary=sorted(ol.OL_PRIMARY_ROIS | {"NEW(R)"}))
    with pytest.raises(OlAdapterError) as exc:
        ol.load_ol_release_meta(path)
    assert _code(exc) is OlAdapterErrorCode.REGION_VOCABULARY_UNKNOWN
    path.write_text("dataset:string,tag:string\noptic-lobe,v1.1\n", encoding="utf-8")
    with pytest.raises(OlAdapterError) as exc:
        ol.load_ol_release_meta(path)
    assert _code(exc) is OlAdapterErrorCode.SCHEMA_MISMATCH


def test_no_network_imports():
    source = Path(ol.__file__).read_text(encoding="utf-8")
    for token in ("requests", "urllib", "http.client", "socket", "curl"):
        assert f"import {token}" not in source


_REAL_ROOT = os.environ.get("LOCI_FLYBRAIN_STORAGE_ROOT", "")


@pytest.mark.skipif(
    not _REAL_ROOT
    or not (Path(_REAL_ROOT) / "snapshots" / "ol" / "optic_lobe_v1.1" / "manifest" / "manifest.json").is_file(),
    reason="real ol snapshot not present under LOCI_FLYBRAIN_STORAGE_ROOT",
)
def test_real_snapshot_smoke():
    # verify_hashes=False: size + manifest checks only, and no hash stamps are written.
    opened = ol.open_ol_snapshot(_REAL_ROOT, verify_hashes=False)
    assert opened.license_spdx == "CC-BY-4.0"
    for path in opened.product_paths.values():
        assert path.is_file()
    meta = ol.load_ol_release_meta(opened.path(ol.ROLE_NEUPRINT_META))
    assert (meta.dataset, meta.tag) == ("optic-lobe", "v1.1")
