import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_mc_adapter as mc  # noqa: E402
from flybrain_mc_adapter import McAdapterError, McAdapterErrorCode  # noqa: E402


def bid(idx):
    return 10000 + idx


# (idx, type, status, superclass, class, somaSide, rootSide, itoleeHl, trumanHl)
_META_ROWS = [
    (1, "T1a", "Traced", "ol_intrinsic", None, "R", None, None, None),
    (2, "T1a", "Traced", "ol_intrinsic", None, "L", None, None, None),
    (3, "Mi1", "Traced", "ol_intrinsic", None, "R", None, None, None),
    (4, "ORN_DA1", "Traced", "cb_sensory", "olfactory", "L", None, None, None),
    (5, "DA1_lPN", "Traced", "cb_intrinsic", "ALPN", "R", None, "ALad1", None),
    (6, "DA1_lPN", "Traced", "cb_intrinsic", "ALPN", "L", None, "ALad1", None),
    (7, "IN09A001", "Traced", "vnc_intrinsic", None, "L", None, None, "09A"),
    (8, "MNfl01", "Traced", "vnc_motor", None, "R", None, None, "15B"),
    (9, "IN12B002", "Traced", "vnc_intrinsic", None, "L", None, None, "12B"),
    (10, "AN19A", "Traced", "ascending_neuron", None, "R", None, "putative_primary", "19A"),
    (11, None, "Traced", "cb_intrinsic", None, "L", None, None, None),  # untyped
    (12, "orphan_t", "Orphan", "cb_intrinsic", None, "L", None, None, None),  # status
    (13, "DNa01", "Traced", "descending_neuron", None, None, "R", "putative_primary", None),
    (14, "EPG", "Traced", "cb_intrinsic", "CX", "M", None, "DM1_CX_d2", None),
    (15, "tiny", "Traced", "cb_intrinsic", None, "L", None, None, None),  # few synapses
    (16, "SMP_anchor", "Anchor", "cb_intrinsic", None, "R", None, None, "TBD"),
]

# idx -> {primary or other roi: (pre, post)}
_ROI_INFO = {
    1: {"ME(R)": (50, 200), "LO(R)": (5, 20), "OL(R)": (55, 220)},
    2: {"ME(L)": (40, 150), "LOP(L)": (30, 90)},
    3: {"ME(R)": (300, 400)},
    4: {"AL(L)": (500, 100), "AL-DA1(L)": (500, 100), "CentralBrain": (500, 100)},
    5: {"AL(R)": (100, 300), "LH(R)": (200, 50), "CA(R)": (100, 30)},
    6: {"AL(L)": (90, 280), "LH(L)": (180, 60)},
    7: {"LegNp(T1)(L)": (300, 500)},
    8: {"LegNp(T1)(R)": (5, 600)},
    9: {"LegNp(T1)(L)": (100, 100), "LegNp(T2)(L)": (80, 90)},
    10: {"LegNp(T1)(R)": (200, 50), "GNG": (150, 100), "CV-unspecified": (10, 10)},
    11: {"SMP(L)": (100, 100)},
    12: {"SMP(L)": (100, 100)},
    13: {"GNG": (100, 400), "LegNp(T1)(R)": (100, 20)},
    14: {"EB": (300, 300), "PB": (100, 100)},
    15: {"AL(L)": (1, 2)},
    16: {"SMP(R)": (100, 100)},
    90: {"ME(R)": (1, 1)},  # untyped fragment present only in the neuron table
}

# pre idx -> [(post idx, weight)]
_EDGES = {
    1: [(2, 30), (3, 20)],
    2: [(1, 10)],
    3: [(1, 200), (2, 150), (4, 20)],
    4: [(5, 400), (6, 300)],
    5: [(6, 50), (14, 60)],
    6: [(5, 40)],
    7: [(8, 300), (9, 200)],
    8: [(7, 3)],
    9: [(7, 60), (10, 30)],
    10: [(13, 250), (9, 100)],
    11: [(1, 50)],
    12: [(1, 50)],
    13: [(8, 90), (7, 30)],
    14: [(5, 20)],
    15: [(4, 12)],
    16: [(14, 70)],
    90: [(1, 1)],
}

# idx -> (predicted_nt, confidence, total_nt_predictions)
_NT = {
    1: ("acetylcholine", 0.9, 55),
    2: ("acetylcholine", 0.7, 70),
    3: ("glutamate", 0.8, 300),
    4: ("acetylcholine", 0.95, 500),
    5: ("acetylcholine", 0.9, 400),
    6: ("acetylcholine", 0.85, 270),
    7: ("gaba", 0.8, 300),
    8: ("glutamate", 0.6, 5),
    9: ("gaba", 0.75, 180),
    10: ("acetylcholine", 0.8, 350),
    11: ("gaba", 0.9, 100),
    12: ("gaba", 0.9, 100),
    13: ("acetylcholine", 0.5, 200),
    14: ("unclear", 0.4, 400),
    15: ("glutamate", 0.9, 1),
    16: ("dopamine", 0.7, 100),
    90: ("unclear", 0.3, 1),
}


def write_meta(path, rows=_META_ROWS):
    cols = list(zip(*rows))
    table = pa.table(
        {
            "bodyId": pa.array([bid(i) for i in cols[0]], pa.int64()),
            "type": list(cols[1]),
            "status": list(cols[2]),
            "statusLabel": pa.array(["Roughly traced" for _ in rows]).dictionary_encode(),
            "superclass": list(cols[3]),
            "class": pa.array(list(cols[4]), pa.string()),
            "somaSide": list(cols[5]),
            "rootSide": pa.array(list(cols[6]), pa.string()),
            "itoleeHl": pa.array(list(cols[7]), pa.string()),
            "trumanHl": pa.array(list(cols[8]), pa.string()),
            "instance": [f"{t}_x" if t else None for t in cols[1]],
            "group": pa.array([float(bid(i)) for i in cols[0]], pa.float64()),
            "subclass": pa.array([None for _ in rows], pa.string()),
        }
    )
    feather.write_feather(table, str(path))


def write_nt(path, nt=_NT):
    idx = sorted(nt)
    table = pa.table(
        {
            "body": pa.array([bid(i) for i in idx], pa.int64()),
            "cell_type": [None for _ in idx],
            "total_nt_predictions": pa.array([nt[i][2] for i in idx], pa.int32()),
            "predicted_nt_confidence": [nt[i][1] for i in idx],
            "predicted_nt": [nt[i][0] for i in idx],
            "ground_truth": pa.array([None for _ in idx], pa.string()),
            "celltype_total_nt_predictions": pa.array([nt[i][2] for i in idx], pa.int32()),
            "celltype_predicted_nt": [nt[i][0] for i in idx],
            "celltype_predicted_nt_confidence": [nt[i][1] for i in idx],
            "consensus_nt": [nt[i][0] for i in idx],
        }
    )
    feather.write_feather(table, str(path))


def write_edgelist(path, edges=_EDGES, chunksize=None):
    pre, post, weight = [], [], []
    for p, targets in sorted(edges.items()):
        for q, w in targets:
            pre.append(bid(p))
            post.append(bid(q))
            weight.append(w)
    table = pa.table(
        {
            "body_pre": pa.array(pre, pa.int64()),
            "body_post": pa.array(post, pa.int64()),
            "weight": pa.array(weight, pa.int64()),
        }
    )
    feather.write_feather(table, str(path), chunksize=chunksize)


def _roi_json(info):
    return json.dumps(
        {roi: {"pre": pre, "post": post, "downstream": pre * 4, "upstream": post, "synweight": pre * 4 + post}
         for roi, (pre, post) in sorted(info.items())}
    )


def write_neurons(path, roi_info=_ROI_INFO, *, bodyid_offset=0, chunksize=None):
    idx = sorted(roi_info)
    pre = [sum(p for r, (p, _) in roi_info[i].items() if r in mc.MC_PRIMARY_ROIS) for i in idx]
    post = [sum(q for r, (_, q) in roi_info[i].items() if r in mc.MC_PRIMARY_ROIS) for i in idx]
    table = pa.table(
        {
            ":ID(Body-ID)": pa.array([bid(i) for i in idx], pa.int64()),
            "post:int": pa.array(post, pa.int32()),
            "pre:int": pa.array(pre, pa.int32()),
            "downstream:int": pa.array([p * 4 for p in pre], pa.int32()),
            "upstream:int": pa.array(post, pa.int32()),
            "synweight:int": pa.array([p * 4 + q for p, q in zip(pre, post)], pa.int32()),
            "bodyId:long": pa.array([bid(i) + bodyid_offset for i in idx], pa.int64()),
            "type:string": [None for _ in idx],
            "roiInfo:string": [_roi_json(roi_info[i]) for i in idx],
        }
    )
    feather.write_feather(table, str(path), chunksize=chunksize)


def write_dataset_meta(path, rois=None, dataset="male-cns", tag="v1.0"):
    rois = sorted(mc.MC_PRIMARY_ROIS) if rois is None else rois
    header = "dataset:string,tag:string,voxelSize:float[],primaryRois:string[],superLevelRois:string[],:Label"
    joined = ";".join(rois)
    line = f'{dataset},{tag},8.0;8.0;8.0,"{joined}","{joined}",Meta'
    Path(path).write_text(header + "\n" + line + "\n", encoding="utf-8")


_WRITERS = {
    mc.ROLE_META: write_meta,
    mc.ROLE_NT_PREDICTION: write_nt,
    mc.ROLE_EDGELIST: write_edgelist,
    mc.ROLE_NEURON_ROI_INFO: write_neurons,
    mc.ROLE_DATASET_META: write_dataset_meta,
}


def build_snapshot(storage_root, *, manifest_overrides=None, write_manifest=True, extra_files=()):
    """Create a synthetic mc snapshot + verified manifest under storage_root."""
    snap = Path(storage_root) / "snapshots" / "mc" / "male-cns_v1.0"
    for role, rel in mc.MC_PRODUCT_PATHS.items():
        (snap / rel).parent.mkdir(parents=True, exist_ok=True)
        _WRITERS[role](snap / rel)
    files = []
    for role, rel in sorted(mc.MC_PRODUCT_PATHS.items()):
        path = snap / rel
        files.append({"relative_path": rel, "size_bytes": path.stat().st_size, "sha256": mc.sha256_file(path),
                      "role": role})
    for rel, payload in extra_files:
        path = snap / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        files.append({"relative_path": rel, "size_bytes": path.stat().st_size, "sha256": mc.sha256_file(path),
                      "role": "synapse_points"})
    manifest = mc.build_mc_manifest(
        files=files,
        source_uri="synthetic://mc",
        retrieved_at="2026-09-24T00:00:00Z",
        run_id="run-test",
        generated_at="2026-09-24T00:00:00Z",
        next_check_due="2099-01-01T00:00:00Z",
    )
    if manifest_overrides:
        manifest_overrides(manifest)
        manifest["integrity"]["manifest_sha256"] = mc.manifest_digest(manifest)
    if write_manifest:
        mc.write_mc_manifest(snap, manifest)
    return snap, manifest


def _code(exc_info):
    return exc_info.value.code


# ---------------------------------------------------------------------------
# ROI / NT vocabularies
# ---------------------------------------------------------------------------


def test_primary_roi_vocabulary_is_complete_and_collision_free():
    assert len(mc.MC_PRIMARY_ROIS) == 144
    by_region = {}
    for roi in mc.MC_PRIMARY_ROIS.values():
        by_region.setdefault(roi.region_id, set()).add(roi.neuropil)
    assert all(len(neuropils) == 1 for neuropils in by_region.values())
    divisions = {roi.division for roi in mc.MC_PRIMARY_ROIS.values()}
    assert divisions == {"brain", "nerve_cord", "neck_connective"}


@pytest.mark.parametrize(
    "raw,division,subdivision,neuropil,side,region_id",
    [
        ("ME(R)", "brain", "optic_lobe", "ME", "right", "brain_me"),
        ("Optic-unspecified(L)", "brain", "optic_lobe", "Optic-unspecified", "left", "brain_optic_unspecified"),
        ("AL(L)", "brain", "central_brain", "AL", "left", "brain_al"),
        ("aL(R)", "brain", "central_brain", "aL", "right", "brain_mb_al"),
        ("a'L(L)", "brain", "central_brain", "a'L", "left", "brain_mb_apl"),
        ("EB", "brain", "central_brain", "EB", None, "brain_eb"),
        ("GNG", "brain", "central_brain", "GNG", None, "brain_gng"),
        ("LegNp(T1)(L)", "nerve_cord", "ventral_nerve_cord", "LegNp(T1)", "left", "nerve_cord_legnp_t1"),
        ("HTct(UTct-T3)(R)", "nerve_cord", "ventral_nerve_cord", "HTct(UTct-T3)", "right", "nerve_cord_htct_utct_t3"),
        ("ANm", "nerve_cord", "ventral_nerve_cord", "ANm", None, "nerve_cord_anm"),
        ("ProLN(L)", "nerve_cord", "ventral_nerve_cord_nerve", "ProLN", "left", "nerve_cord_proln"),
        ("CV-unspecified", "neck_connective", "cervical_connective", "CV-unspecified", None,
         "neck_connective_cv_unspecified"),
    ],
)
def test_map_primary_roi_explicit_vocabulary(raw, division, subdivision, neuropil, side, region_id):
    mapped = mc.map_mc_primary_roi(raw)
    assert (mapped.division, mapped.subdivision, mapped.neuropil, mapped.side, mapped.region_id) == (
        division, subdivision, neuropil, side, region_id,
    )


@pytest.mark.parametrize("raw", ["", "ME", "EB(L)", "AMMC(R)", "CentralBrain", "AL-DA1(L)", "LegNp(T1)", "me(R)"])
def test_map_primary_roi_unknown_fails_closed(raw):
    with pytest.raises(McAdapterError) as exc:
        mc.map_mc_primary_roi(raw)
    assert _code(exc) is McAdapterErrorCode.REGION_VOCABULARY_UNKNOWN
    assert str(exc.value).startswith("[REGION_VOCABULARY_UNKNOWN] ")


def test_nt_short_codes():
    assert mc.nt_short_code("Acetylcholine") == "ach"
    assert mc.nt_short_code("histamine") == "his"
    for bad in ("unclear", "tyramine", "nitric_oxide", ""):
        with pytest.raises(McAdapterError) as exc:
            mc.nt_short_code(bad)
        assert _code(exc) is McAdapterErrorCode.NT_VOCABULARY_UNKNOWN


def test_roi_vocabulary_check_against_dataset_meta(tmp_path):
    path = tmp_path / "meta.csv"
    write_dataset_meta(path)
    assert set(mc.check_mc_roi_vocabulary(path)) == set(mc.MC_PRIMARY_ROIS)
    write_dataset_meta(path, rois=[*sorted(mc.MC_PRIMARY_ROIS), "NEW(L)"])
    with pytest.raises(McAdapterError) as exc:
        mc.check_mc_roi_vocabulary(path)
    assert _code(exc) is McAdapterErrorCode.REGION_VOCABULARY_UNKNOWN
    assert exc.value.details["unknown"] == ["NEW(L)"]
    write_dataset_meta(path, rois=sorted(mc.MC_PRIMARY_ROIS)[1:])
    with pytest.raises(McAdapterError) as exc:
        mc.check_mc_roi_vocabulary(path)
    assert exc.value.details["missing"] == [sorted(mc.MC_PRIMARY_ROIS)[0]]
    write_dataset_meta(path, tag="v0.9")
    with pytest.raises(McAdapterError) as exc:
        mc.check_mc_roi_vocabulary(path)
    assert _code(exc) is McAdapterErrorCode.DATASET_PIN_MISMATCH


def test_summarize_roi_info():
    summary = mc.summarize_roi_info(_roi_json(_ROI_INFO[4]))
    # AL-DA1(L) and CentralBrain are non-primary and must not double count.
    assert (summary.top_roi, summary.top_synapses, summary.total_primary_synapses, summary.n_primary_rois) == (
        "AL(L)", 600, 600, 1)
    tie = mc.summarize_roi_info(json.dumps({"SMP(R)": {"pre": 5, "post": 5}, "SIP(R)": {"pre": 10}}))
    assert tie.top_roi == "SIP(R)" and tie.total_primary_synapses == 20
    assert mc.summarize_roi_info(None).top_roi is None
    assert mc.summarize_roi_info(json.dumps({"CentralBrain": {"pre": 3}})).n_primary_rois == 0
    with pytest.raises(McAdapterError) as exc:
        mc.summarize_roi_info("{not json")
    assert _code(exc) is McAdapterErrorCode.SCHEMA_MISMATCH
    with pytest.raises(McAdapterError):
        mc.summarize_roi_info("[1, 2]")


# ---------------------------------------------------------------------------
# Snapshot / manifest
# ---------------------------------------------------------------------------


def test_open_snapshot_verified(tmp_path):
    snap, manifest = build_snapshot(tmp_path)
    opened = mc.open_mc_snapshot(tmp_path)
    assert opened.snapshot_root == snap.resolve()
    assert opened.manifest_sha256 == manifest["integrity"]["manifest_sha256"]
    assert opened.license_spdx == "CC-BY-4.0"
    assert set(opened.product_paths) == set(mc.MC_PRODUCT_PATHS)
    prov = opened.provenance()
    assert prov["dataset_symbol"] == "mc" and prov["version_id"] == "male-cns_v1.0"
    assert set(opened.hash_verification.values()) == {"hashed"}
    # write-once stamps land under the snapshot's manifest dir by default
    assert len(list((snap / "manifest" / "hash-stamps").glob("*.json"))) == len(mc.MC_PRODUCT_PATHS)
    again = mc.open_mc_snapshot(tmp_path)
    assert set(again.hash_verification.values()) == {"stamp"}


def test_unrequired_listed_files_are_size_checked_only(tmp_path):
    build_snapshot(tmp_path, extra_files=[("source/flat-connectome/syn-points.feather", b"x" * 64)])
    opened = mc.open_mc_snapshot(tmp_path, required_roles=(mc.ROLE_META,))
    assert opened.hash_verification["source/flat-connectome/syn-points.feather"] == "size_only"
    assert opened.hash_verification[mc.MC_PRODUCT_PATHS[mc.ROLE_META]] == "hashed"
    assert opened.hash_verification[mc.MC_PRODUCT_PATHS[mc.ROLE_EDGELIST]] == "size_only"
    full = mc.open_mc_snapshot(tmp_path, verify_hashes=True)
    assert set(full.hash_verification.values()) == {"hashed"}
    smoke = mc.open_mc_snapshot(tmp_path, verify_hashes=False)
    assert set(smoke.hash_verification.values()) == {"size_only"}


def test_stamp_dir_override_keeps_snapshot_untouched(tmp_path):
    snap, _ = build_snapshot(tmp_path / "root")
    stamps = tmp_path / "stamps"
    opened = mc.open_mc_snapshot(tmp_path / "root", stamp_dir=stamps)
    assert set(opened.hash_verification.values()) == {"hashed"}
    assert not (snap / "manifest" / "hash-stamps").exists()
    assert len(list(stamps.glob("*.json"))) == len(mc.MC_PRODUCT_PATHS)
    again = mc.open_mc_snapshot(tmp_path / "root", stamp_dir=stamps)
    assert set(again.hash_verification.values()) == {"stamp"}


def test_open_snapshot_missing_manifest(tmp_path):
    build_snapshot(tmp_path, write_manifest=False)
    with pytest.raises(McAdapterError) as exc:
        mc.open_mc_snapshot(tmp_path)
    assert _code(exc) is McAdapterErrorCode.MANIFEST_MISSING


def test_open_snapshot_missing_dir(tmp_path):
    with pytest.raises(McAdapterError) as exc:
        mc.open_mc_snapshot(tmp_path)
    assert _code(exc) is McAdapterErrorCode.SNAPSHOT_MISSING


def test_tampered_product_fails_integrity(tmp_path):
    snap, _ = build_snapshot(tmp_path)
    path = snap / mc.MC_PRODUCT_PATHS[mc.ROLE_DATASET_META]
    data = bytearray(path.read_bytes())
    data[0] = ord("X")
    path.write_bytes(bytes(data))
    with pytest.raises(McAdapterError) as exc:
        mc.open_mc_snapshot(tmp_path)
    assert _code(exc) is McAdapterErrorCode.INTEGRITY_MISMATCH


def test_manifest_self_hash_mismatch(tmp_path):
    snap, manifest = build_snapshot(tmp_path, write_manifest=False)
    manifest["notes"] = ["edited after hashing"]
    mc.write_mc_manifest(snap, manifest)
    with pytest.raises(McAdapterError) as exc:
        mc.open_mc_snapshot(tmp_path)
    assert _code(exc) is McAdapterErrorCode.INTEGRITY_MISMATCH


def _first_product(m, role):
    return next(f for f in m["integrity"]["files"] if f["relative_path"] == mc.MC_PRODUCT_PATHS[role])


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda m: m["dataset"]["source"]["license"].update(spdx_id="UNREVIEWED"),
         McAdapterErrorCode.LICENSE_NOT_PERMITTED),
        (lambda m: m["dataset"].update(version_id="male-cns_v0.9"), McAdapterErrorCode.DATASET_PIN_MISMATCH),
        (lambda m: m["dataset"].update(symbol="mv"), McAdapterErrorCode.DATASET_PIN_MISMATCH),
        (lambda m: m["scope"].update(sex="female"), McAdapterErrorCode.DATASET_PIN_MISMATCH),
        (lambda m: m["scope"].update(anatomy="ventral_nerve_cord"), McAdapterErrorCode.DATASET_PIN_MISMATCH),
        (lambda m: m["integrity"]["verification"].update(status="pending"), McAdapterErrorCode.INTEGRITY_MISMATCH),
        (lambda m: m["refresh"].update(decision="rollback"), McAdapterErrorCode.PROMOTION_STATE_INVALID),
        (lambda m: m["refresh"].update(next_check_due="2020-01-01T00:00:00Z"),
         McAdapterErrorCode.PROMOTION_STATE_INVALID),
        (lambda m: m["artifact"].update(relative_root="snapshots/mv/manc_v1.0"), McAdapterErrorCode.DATASET_PIN_MISMATCH),
        (lambda m: m["integrity"]["files"].append(dict(m["integrity"]["files"][0], relative_path="../outside.csv")),
         McAdapterErrorCode.PATH_ESCAPE),
        (lambda m: m["integrity"]["files"].append(dict(m["integrity"]["files"][0], relative_path="x.feather.partial")),
         McAdapterErrorCode.MANIFEST_INVALID),
        (lambda m: m["integrity"]["files"].append(dict(m["integrity"]["files"][0],
                                                       relative_path=m["integrity"]["files"][0]["relative_path"].upper())),
         McAdapterErrorCode.MANIFEST_INVALID),
        (lambda m: _first_product(m, mc.ROLE_META).update(role="nt_prediction"), McAdapterErrorCode.MANIFEST_INVALID),
        (lambda m: _first_product(m, mc.ROLE_META).update(size_bytes=True), McAdapterErrorCode.MANIFEST_INVALID),
        (lambda m: m["integrity"]["files"].remove(_first_product(m, mc.ROLE_NEURON_ROI_INFO)),
         McAdapterErrorCode.PRODUCT_MISSING),
        (lambda m: _first_product(m, mc.ROLE_EDGELIST).update(size_bytes=1), McAdapterErrorCode.INTEGRITY_MISMATCH),
    ],
)
def test_manifest_gates_fail_closed(tmp_path, mutate, code):
    build_snapshot(tmp_path, manifest_overrides=mutate)
    with pytest.raises(McAdapterError) as exc:
        mc.open_mc_snapshot(tmp_path)
    assert _code(exc) is code


def test_size_mismatch_fails_before_any_hashing(tmp_path):
    snap, _ = build_snapshot(tmp_path, manifest_overrides=lambda m: _first_product(m, mc.ROLE_EDGELIST).update(
        size_bytes=1))
    with pytest.raises(McAdapterError):
        mc.open_mc_snapshot(tmp_path)
    assert not (snap / "manifest" / "hash-stamps").exists()


def test_required_roles_subset(tmp_path):
    nt_rel = mc.MC_PRODUCT_PATHS[mc.ROLE_NT_PREDICTION]
    build_snapshot(tmp_path, manifest_overrides=lambda m: m["integrity"]["files"].__setitem__(
        slice(None), [f for f in m["integrity"]["files"] if f["relative_path"] != nt_rel]))
    roles = (mc.ROLE_META, mc.ROLE_EDGELIST)
    opened = mc.open_mc_snapshot(tmp_path, required_roles=roles)
    assert set(opened.product_paths) == set(roles)
    with pytest.raises(McAdapterError) as exc:
        mc.open_mc_snapshot(tmp_path)
    assert _code(exc) is McAdapterErrorCode.PRODUCT_MISSING
    with pytest.raises(McAdapterError) as exc:
        mc.open_mc_snapshot(tmp_path, required_roles=("synapse_points",))
    assert _code(exc) is McAdapterErrorCode.PRODUCT_MISSING


def test_write_manifest_never_overwrites(tmp_path):
    snap, manifest = build_snapshot(tmp_path)
    before = (snap / "manifest" / "manifest.json").read_text()
    with pytest.raises(McAdapterError) as exc:
        mc.write_mc_manifest(snap, manifest)
    assert _code(exc) is McAdapterErrorCode.MANIFEST_EXISTS
    assert (snap / "manifest" / "manifest.json").read_text() == before


def test_sidecar_mismatch(tmp_path):
    snap, _ = build_snapshot(tmp_path)
    (snap / "manifest" / "manifest.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(McAdapterError) as exc:
        mc.open_mc_snapshot(tmp_path)
    assert _code(exc) is McAdapterErrorCode.INTEGRITY_MISMATCH


def test_missing_sidecar_fails_closed(tmp_path):
    snap, _ = build_snapshot(tmp_path)
    (snap / "manifest" / "manifest.sha256").unlink()
    with pytest.raises(McAdapterError) as exc:
        mc.open_mc_snapshot(tmp_path)
    assert _code(exc) is McAdapterErrorCode.MANIFEST_MISSING


def test_relative_storage_root_rejected():
    with pytest.raises(McAdapterError) as exc:
        mc.open_mc_snapshot("relative/root")
    assert _code(exc) is McAdapterErrorCode.ROOT_NOT_CONFIGURED


def test_validate_manifest_uses_injected_clock(tmp_path):
    snap, manifest = build_snapshot(tmp_path)
    with pytest.raises(McAdapterError) as exc:
        mc.validate_mc_manifest(manifest, snap, now=datetime(2100, 1, 1, tzinfo=timezone.utc))
    assert _code(exc) is McAdapterErrorCode.PROMOTION_STATE_INVALID


def test_manifest_json_is_schema_shaped(tmp_path):
    snap, _ = build_snapshot(tmp_path)
    manifest = json.loads((snap / "manifest" / "manifest.json").read_text())
    for key in ("schema_version", "manifest_id", "generated_at", "storage_root_env", "artifact", "dataset",
                "scope", "integrity", "refresh", "lineage"):
        assert key in manifest
    assert manifest["artifact"]["path_template"].startswith("$LOCI_FLYBRAIN_STORAGE_ROOT\\")
    assert manifest["dataset"]["symbol"] == "mc"
    assert manifest["dataset"]["source"]["license"]["spdx_id"] == "CC-BY-4.0"
    assert "10.1101/2025.10.09.680999" in manifest["dataset"]["source"]["citation"]
    assert manifest["scope"] == {**manifest["scope"], "sex": "male", "anatomy": "brain_and_ventral_nerve_cord"}


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def test_annotations_reader(tmp_path):
    path = tmp_path / "meta.feather"
    write_meta(path)
    frame = mc.load_mc_annotations(path, optional_columns=("subclass", "absent_column"))
    assert len(frame) == len(_META_ROWS)
    assert frame["bodyId"].iloc[0] == str(bid(1))
    assert frame["bodyId"].map(type).eq(str).all()
    assert "subclass" in frame.columns and "absent_column" not in frame.columns
    dup = tmp_path / "dup.feather"
    write_meta(dup, rows=_META_ROWS + [_META_ROWS[0]])
    with pytest.raises(McAdapterError) as exc:
        mc.load_mc_annotations(dup)
    assert _code(exc) is McAdapterErrorCode.SCHEMA_MISMATCH


def test_annotations_missing_column(tmp_path):
    path = tmp_path / "meta.feather"
    feather.write_feather(pa.table({"bodyId": [1], "type": ["x"]}), str(path))
    with pytest.raises(McAdapterError) as exc:
        mc.load_mc_annotations(path)
    assert _code(exc) is McAdapterErrorCode.SCHEMA_MISMATCH
    assert "superclass" in str(exc.value)


def test_nt_reader_filters_by_body(tmp_path):
    path = tmp_path / "nt.feather"
    write_nt(path)
    frame = mc.load_mc_nt_predictions(path, body_ids=[bid(3), bid(1), 999])
    assert list(frame["body"]) == [str(bid(1)), str(bid(3))]
    assert list(frame["predicted_nt"]) == ["acetylcholine", "glutamate"]
    assert len(mc.load_mc_nt_predictions(path)) == len(_NT)
    assert mc.load_mc_nt_predictions(path, body_ids=[999]).empty


def test_nt_reader_rejects_duplicate_bodies(tmp_path):
    path = tmp_path / "nt.feather"
    write_nt(path)
    table = feather.read_table(str(path))
    feather.write_feather(pa.concat_tables([table, table.slice(0, 1)]), str(path))
    with pytest.raises(McAdapterError) as exc:
        mc.load_mc_nt_predictions(path)
    assert _code(exc) is McAdapterErrorCode.SCHEMA_MISMATCH


@pytest.mark.parametrize("chunksize", [None, 3])
def test_outgoing_totals_streaming(tmp_path, chunksize):
    path = tmp_path / "edges.feather"
    write_edgelist(path, chunksize=chunksize)
    frame, scanned = mc.load_mc_outgoing_totals(path)
    assert scanned == sum(len(v) for v in _EDGES.values())
    got = dict(zip(frame["root_id"], zip(frame["total_out_synapses"], frame["n_post_partners"])))
    assert got[str(bid(3))] == (370, 3)
    assert got[str(bid(8))] == (3, 1)
    assert list(frame["root_id"]) == sorted(frame["root_id"], key=lambda r: r.zfill(24))
    subset, scanned_subset = mc.load_mc_outgoing_totals(path, body_ids=[bid(3), bid(8)])
    assert list(subset["root_id"]) == [str(bid(3)), str(bid(8))]
    assert scanned_subset == 4


def test_edgelist_missing_column_and_not_feather(tmp_path):
    path = tmp_path / "edges.feather"
    feather.write_feather(pa.table({"body_pre": [1], "weight": [3]}), str(path))
    with pytest.raises(McAdapterError) as exc:
        mc.load_mc_outgoing_totals(path)
    assert _code(exc) is McAdapterErrorCode.SCHEMA_MISMATCH and "body_post" in str(exc.value)
    path.write_text("body_pre,body_post,weight\n1,2,3\n")
    with pytest.raises(McAdapterError) as exc:
        mc.load_mc_outgoing_totals(path)
    assert _code(exc) is McAdapterErrorCode.SCHEMA_MISMATCH


@pytest.mark.parametrize("chunksize", [None, 4])
def test_neuron_roi_summary_streaming(tmp_path, chunksize):
    path = tmp_path / "neurons.feather"
    write_neurons(path, chunksize=chunksize)
    frame = mc.load_mc_neuron_roi_summary(path, body_ids=[bid(5), bid(10), bid(4)])
    assert list(frame["root_id"]) == [str(bid(4)), str(bid(5)), str(bid(10))]
    row = frame.set_index("root_id").loc[str(bid(5))]
    assert (row["top_roi"], row["top_roi_synapses"], row["total_primary_synapses"], row["n_primary_rois"]) == (
        "AL(R)", 400, 780, 3)
    assert frame.set_index("root_id").loc[str(bid(4)), "total_primary_synapses"] == 600
    assert len(mc.load_mc_neuron_roi_summary(path)) == len(_ROI_INFO)


def test_neuron_roi_summary_id_column_mismatch(tmp_path):
    path = tmp_path / "neurons.feather"
    write_neurons(path, bodyid_offset=1)
    with pytest.raises(McAdapterError) as exc:
        mc.load_mc_neuron_roi_summary(path, body_ids=[bid(1)])
    assert _code(exc) is McAdapterErrorCode.SCHEMA_MISMATCH


def test_neuron_table_missing_column(tmp_path):
    path = tmp_path / "neurons.feather"
    feather.write_feather(pa.table({":ID(Body-ID)": [1], "roiInfo:string": ["{}"]}), str(path))
    with pytest.raises(McAdapterError) as exc:
        mc.load_mc_neuron_roi_summary(path)
    assert _code(exc) is McAdapterErrorCode.SCHEMA_MISMATCH and "pre:int" in str(exc.value)


def test_missing_product_file(tmp_path):
    for reader in (mc.load_mc_annotations, mc.load_mc_nt_predictions, mc.load_mc_outgoing_totals,
                   mc.load_mc_neuron_roi_summary, mc.load_mc_primary_rois):
        with pytest.raises(McAdapterError) as exc:
            reader(tmp_path / "nope.feather")
        assert _code(exc) is McAdapterErrorCode.PRODUCT_MISSING


# ---------------------------------------------------------------------------
# Real snapshot (skipped unless LOCI_FLYBRAIN_STORAGE_ROOT holds it). Read-only:
# size checks only, so no hash stamps are written into the snapshot.
# ---------------------------------------------------------------------------

_REAL_ROOT = os.environ.get("LOCI_FLYBRAIN_STORAGE_ROOT", "")
_REAL_SNAPSHOT = Path(_REAL_ROOT) / "snapshots" / "mc" / "male-cns_v1.0" if _REAL_ROOT else None
real_snapshot = pytest.mark.skipif(
    _REAL_SNAPSHOT is None or not (_REAL_SNAPSHOT / "manifest" / "manifest.json").is_file(),
    reason="real mc snapshot not present under LOCI_FLYBRAIN_STORAGE_ROOT",
)


@real_snapshot
def test_real_snapshot_smoke():
    opened = mc.open_mc_snapshot(_REAL_ROOT, verify_hashes=False)
    assert opened.license_spdx == "CC-BY-4.0"
    for path in opened.product_paths.values():
        assert path.is_file()
    assert len(mc.check_mc_roi_vocabulary(opened.path(mc.ROLE_DATASET_META))) == 144


@real_snapshot
def test_real_snapshot_filtered_readers():
    opened = mc.open_mc_snapshot(_REAL_ROOT, verify_hashes=False)
    ids = [10001, 10002, 10003]
    roi = mc.load_mc_neuron_roi_summary(opened.path(mc.ROLE_NEURON_ROI_INFO), body_ids=ids)
    assert list(roi["root_id"]) == ["10001", "10002", "10003"]
    assert roi["top_roi"].map(lambda r: r in mc.MC_PRIMARY_ROIS).all()
    totals, _ = mc.load_mc_outgoing_totals(opened.path(mc.ROLE_EDGELIST), body_ids=ids)
    # The flat weight table and neuPrint's per-neuron ``downstream`` agree.
    merged = totals.merge(roi, on="root_id")
    assert (merged["total_out_synapses"] == merged["downstream"]).all()
    nt = mc.load_mc_nt_predictions(opened.path(mc.ROLE_NT_PREDICTION), body_ids=ids)
    assert list(nt["body"]) == ["10001", "10002", "10003"]
