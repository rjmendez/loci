import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_mv_adapter as mv  # noqa: E402
from flybrain_hash_stamps import HASH_STAMP_RELATIVE_DIR  # noqa: E402
from flybrain_mv_adapter import MvAdapterError, MvAdapterErrorCode  # noqa: E402

FIXTURE_ROIS = ("ANm", "CV", "IntTct", "LTct", "LegNp(T1)(L)", "LegNp(T1)(R)", "ProLN(L)")
_ROI_STRUCT = pa.struct(
    [(name, pa.int64()) for name in ("downstream", "post", "pre", "synweight", "upstream")]
)
_NT_ORDER = ("acetylcholine", "gaba", "glutamate", "unknown")
_NT_PROBS = {
    "acetylcholine": (0.8, 0.1, 0.05, 0.05),
    "gaba": (0.1, 0.8, 0.05, 0.05),
    "glutamate": (0.05, 0.1, 0.8, 0.05),
    "unknown": (0.1, 0.1, 0.2, 0.6),
    None: (None, None, None, None),
}
BIG_BODY = 841369767527

# idx -> (status, type, class, pre, downstream, somaNeuromere, somaSide, rootSide, hemilineage,
#         birthtime, group, predictedNt, {roi: synweight})
NEURONS = {
    1: ("Traced", "IN01A001", "intrinsic neuron", 200, 3000, "T1", "LHS", None, "01A", "secondary", 100.0,
        "acetylcholine", {"LegNp(T1)(L)": 1000}),
    2: ("Traced", "IN01A001", "intrinsic neuron", 50, 400, "T1", "RHS", None, "01A", "secondary", 100.0,
        "acetylcholine", {"LegNp(T1)(R)": 300, "LTct": 100}),
    3: ("Traced", "IN09A002", "intrinsic neuron", 300, 5000, "T1", "LHS", None, "09A", "primary", 101.0,
        "gaba", {"LegNp(T1)(L)": 900, "LegNp(T1)(R)": 50}),
    4: ("Traced", "IN09A002", "intrinsic neuron", 20, 100, "T1", "RHS", None, "09A", "primary", 101.0,
        "gaba", {"LegNp(T1)(R)": 45, "IntTct": 40}),
    5: ("Traced", "DNa01", "descending neuron", 500, 8000, None, None, "RHS", None, None, 102.0,
        "acetylcholine", {"LTct": 600, "IntTct": 500, "CV": 900}),
    6: ("Traced", "DNa01", "descending neuron", 400, 7000, None, None, "LHS", None, None, 102.0,
        "acetylcholine", {"LTct": 700, "IntTct": 200, "CV": 500}),
    7: ("Traced", "SNta01", "sensory neuron", 30, 150, None, None, "LHS", "TBD", None, None,
        "glutamate", {"LegNp(T1)(L)": 200, "ProLN(L)": 50}),
    8: ("Traced", "SNta01", "sensory neuron", 25, 120, None, None, "RHS", "TBD", None, None,
        "unknown", {"LegNp(T1)(R)": 150}),
    9: ("Traced", "MNfl01", "motor neuron", 15, 60, "T1", "LHS", None, "15B", "secondary", 103.0,
        "glutamate", {"LegNp(T1)(L)": 500}),
    10: ("Traced", "AN01A001", "ascending neuron", 250, 2500, "A1", "RHS", None, "11A", "secondary", 104.0,
         "acetylcholine", {"ANm": 300, "LTct": 200, "IntTct": 150}),
    11: ("Traced", "AN01A001", "ascending neuron", 220, 2200, "A1", "LHS", None, "11A", "secondary", 104.0,
         "acetylcholine", {"ANm": 400, "LTct": 100}),
    12: ("Traced", "IN06A003", "intrinsic neuron", 100, 1500, "A2", "RHS", None, "06A", "secondary", 105.0,
         "gaba", {"ANm": 800}),
    13: ("Traced", None, "intrinsic neuron", 90, 900, "T1", "LHS", None, "01A", "secondary", None,
         "acetylcholine", {"LegNp(T1)(L)": 400}),  # untyped
    14: ("Traced", "Glia01", "Glia", 10, 50, None, None, None, None, None, None,
         "glutamate", {"LegNp(T1)(L)": 30}),  # glia
    15: ("Orphan", "IN01A001", "intrinsic neuron", 99, 999, "T1", "LHS", None, "01A", "secondary", 100.0,
         "acetylcholine", {"LegNp(T1)(L)": 99}),  # not traced
    16: ("Traced", "IN04B004", "intrinsic neuron", 5, 5, "T1", "LHS", None, "04B", "secondary", 107.0,
         "acetylcholine", {"CV": 10}),  # no neuropil synapses
    17: ("Traced", "IN12B005", "intrinsic neuron", 80, 900, "T2", "LHS", None, "12B", "secondary", 106.0,
         "glutamate", {"LegNp(T1)(L)": 60, "LTct": 30}),
}

# pre idx -> [(post idx, weight)]
EDGES = {
    1: [(2, 30), (3, 12), (9, 4)],
    2: [(1, 6)],
    3: [(1, 40), (2, 22), (4, 9), (9, 3)],
    4: [(3, 2)],
    5: [(6, 50), (10, 12)],
    6: [(5, 40)],
    7: [(9, 8), (1, 2)],
    9: [(1, 1)],
    10: [(11, 20), (12, 5)],
    11: [(10, 15)],
    12: [(10, 9), (11, 2)],
    17: [(1, 7)],
}


def body(idx):
    return BIG_BODY if idx == 17 else 10000 + idx


def write_meta(path, neurons=NEURONS, roi_names=FIXTURE_ROIS, drop_columns=(), probs_override=None):
    ids = sorted(neurons)
    cols = {name: [] for name in mv.META_REQUIRED_COLUMNS}
    cols["instance"] = []
    for idx in ids:
        (status, ntype, klass, pre, down, neuromere, soma_side, root_side, hl, birth, group, nt, rois) = neurons[idx]
        probs = (probs_override or {}).get(idx, _NT_PROBS[nt])
        cols["bodyId"].append(body(idx))
        cols["status"].append(status)
        cols["type"].append(ntype)
        cols["instance"].append(f"{ntype}_{idx}" if ntype else None)
        cols["class"].append(klass)
        cols["pre"].append(pre)
        cols["downstream"].append(down)
        cols["upstream"].append(down // 2)
        cols["somaNeuromere"].append(neuromere)
        cols["somaSide"].append(soma_side)
        cols["rootSide"].append(root_side)
        cols["hemilineage"].append(hl)
        cols["birthtime"].append(birth)
        cols["group"].append(group)
        cols["predictedNt"].append(nt)
        cols["predictedNtProb"].append(max(p for p in probs if p is not None) if nt else None)
        for name, prob in zip(_NT_ORDER, probs):
            cols[mv.MV_NT_PROBABILITY_COLUMNS[name]].append(prob)
        info = {}
        for roi in roi_names:
            weight = rois.get(roi)
            info[roi] = None if weight is None else {
                "downstream": weight, "post": weight // 2, "pre": weight // 4, "synweight": weight, "upstream": weight // 2,
            }
        cols["roiInfo"].append(info)
    roi_type = pa.struct([(roi, _ROI_STRUCT) for roi in roi_names])
    arrays = {}
    for name, values in cols.items():
        if name in drop_columns:
            continue
        if name == "roiInfo":
            arrays[name] = pa.array(values, type=roi_type)
        elif name in ("bodyId", "pre", "downstream", "upstream"):
            arrays[name] = pa.array(values, pa.int64())
        elif name in ("group", "predictedNtProb") or name.startswith("nt"):
            arrays[name] = pa.array(values, pa.float64())
        else:
            arrays[name] = pa.array(values, pa.string())
    feather.write_feather(pa.table(arrays), str(path))


def write_edgelist(path, edges=EDGES, duplicate=False):
    lines = ["bodyId_pre,bodyId_post,weight"]
    for pre, targets in sorted(edges.items()):
        for post, weight in targets:
            lines.append(f"{body(pre)},{body(post)},{weight}")
    if duplicate:
        lines.append(lines[1])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


EXTRA_REL = "source/neuprint_manc_v1.0/neuprint_manc_v1.0_ftr/all_ROIs.txt"


def build_snapshot(storage_root, *, manifest_overrides=None, write_manifest=True, meta_kwargs=None):
    """Synthetic mv snapshot + verified manifest under storage_root."""
    snap = Path(storage_root) / "snapshots" / "mv" / "manc_v1.0"
    for rel in (*mv.MV_PRODUCT_PATHS.values(), EXTRA_REL):
        (snap / rel).parent.mkdir(parents=True, exist_ok=True)
    write_meta(snap / mv.MV_PRODUCT_PATHS[mv.ROLE_META], **(meta_kwargs or {}))
    write_edgelist(snap / mv.MV_PRODUCT_PATHS[mv.ROLE_EDGELIST])
    (snap / EXTRA_REL).write_text("\n".join(FIXTURE_ROIS) + "\n", encoding="utf-8")
    files = []
    for role, rel in [*sorted(mv.MV_PRODUCT_PATHS.items()), ("roi_list", EXTRA_REL)]:
        path = snap / rel
        files.append({"relative_path": rel, "size_bytes": path.stat().st_size, "sha256": mv.sha256_file(path),
                      "role": role})
    manifest = mv.build_mv_manifest(
        files=files,
        source_uri="synthetic://manc",
        retrieved_at="2026-09-24T00:00:00Z",
        run_id="run-test",
        generated_at="2026-09-24T00:00:00Z",
        next_check_due="2099-01-01T00:00:00Z",
    )
    if manifest_overrides:
        manifest_overrides(manifest)
        manifest["integrity"]["manifest_sha256"] = mv.manifest_digest(manifest)
    if write_manifest:
        mv.write_mv_manifest(snap, manifest)
    return snap, manifest


def _rewrite_manifest(snap, manifest):
    (snap / mv.MV_MANIFEST_RELATIVE_PATH).write_text(json.dumps(manifest), encoding="utf-8")
    (snap / mv.MV_MANIFEST_SIDECAR_RELATIVE_PATH).write_text(manifest["integrity"]["manifest_sha256"] + "\n")


def _code(exc_info):
    return exc_info.value.code


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,kind,side,region_id",
    [
        ("LegNp(T1)(L)", mv.ROI_KIND_NEUROPIL, "left", "vnc_legnp_t1"),
        ("LegNp(T3)(R)", mv.ROI_KIND_NEUROPIL, "right", "vnc_legnp_t3"),
        ("HTct(UTct-T3)(R)", mv.ROI_KIND_NEUROPIL, "right", "vnc_htct_utct_t3"),
        ("NTct(UTct-T1)(L)", mv.ROI_KIND_NEUROPIL, "left", "vnc_ntct_utct_t1"),
        ("mVAC(T2)(L)", mv.ROI_KIND_NEUROPIL, "left", "vnc_mvac_t2"),
        ("ANm", mv.ROI_KIND_NEUROPIL, None, "vnc_anm"),
        ("IntTct", mv.ROI_KIND_NEUROPIL, None, "vnc_inttct"),
        ("Ov(R)", mv.ROI_KIND_NEUROPIL, "right", "vnc_ov"),
        ("CV", mv.ROI_KIND_CONNECTIVE, None, "vnc_cv"),
        ("GF(L)", mv.ROI_KIND_TRACT, "left", "vnc_gf"),
        ("ProLN(R)", mv.ROI_KIND_NERVE, "right", "vnc_proln"),
        ("AbNT", mv.ROI_KIND_NERVE, None, "vnc_abnt"),
    ],
)
def test_map_mv_roi_explicit_vocabulary(raw, kind, side, region_id):
    mapped = mv.map_mv_roi(raw)
    assert (mapped.kind, mapped.side, mapped.region_id) == (kind, side, region_id)


@pytest.mark.parametrize("raw", ["", "NotPrimary", "LegNp(T4)(L)", "ITO_midbrain_AL_L", "LegNp(T1)(X)"])
def test_map_mv_roi_unknown_fails_closed(raw):
    with pytest.raises(MvAdapterError) as exc:
        mv.map_mv_roi(raw)
    assert _code(exc) is MvAdapterErrorCode.REGION_VOCABULARY_UNKNOWN


def test_all_published_rois_are_mapped():
    # MANC v1.0 all_ROIs.txt (61 names).
    published = (
        "ADMN(L) ADMN(R) Ov(L) Ov(R) ANm AbN1(L) AbN1(R) AbN2(L) AbN2(R) AbN3(L) AbN3(R) AbN4(L) AbN4(R) AbNT CV "
        "CvN(L) CvN(R) DMetaN(L) DMetaN(R) DProN(L) DProN(R) GF(L) GF(R) HTct(UTct-T3)(L) HTct(UTct-T3)(R) "
        "LegNp(T1)(L) LegNp(T1)(R) LegNp(T2)(L) LegNp(T2)(R) LegNp(T3)(L) LegNp(T3)(R) IntTct LTct MesoAN(L) "
        "MesoAN(R) MesoLN(L) MesoLN(R) MetaLN(L) MetaLN(R) NTct(UTct-T1)(L) NTct(UTct-T1)(R) PDMN(L) PDMN(R) "
        "PrN(L) PrN(R) ProCN(L) ProCN(R) ProAN(L) ProAN(R) ProLN(L) ProLN(R) VProN(L) VProN(R) WTct(UTct-T2)(L) "
        "WTct(UTct-T2)(R) mVAC(T1)(L) mVAC(T1)(R) mVAC(T2)(L) mVAC(T2)(R) mVAC(T3)(L) mVAC(T3)(R)"
    ).split()
    assert len(published) == 61
    kinds = {}
    for name in published:
        kinds.setdefault(mv.map_mv_roi(name).kind, set()).add(mv.map_mv_roi(name).region_id)
    assert len(kinds[mv.ROI_KIND_NEUROPIL]) == 13


def test_nt_short_codes():
    assert [mv.nt_short_code(n) for n in ("acetylcholine", "GABA", "glutamate")] == ["ach", "gaba", "glut"]
    for bad in ("unknown", "histamine", ""):
        with pytest.raises(MvAdapterError) as exc:
            mv.nt_short_code(bad)
        assert _code(exc) is MvAdapterErrorCode.NT_VOCABULARY_UNKNOWN


# ---------------------------------------------------------------------------
# Snapshot open / manifest
# ---------------------------------------------------------------------------


def test_open_snapshot_verifies_and_stamps(tmp_path):
    snap, manifest = build_snapshot(tmp_path)
    opened = mv.open_mv_snapshot(tmp_path)
    assert opened.snapshot_root == snap.resolve()
    assert opened.license_spdx == "CC-BY-4.0"
    assert set(opened.product_paths) == {mv.ROLE_META, mv.ROLE_EDGELIST}
    assert opened.hash_verification[mv.MV_PRODUCT_PATHS[mv.ROLE_META]] == "hashed"
    assert opened.hash_verification[EXTRA_REL] == "size_only"
    prov = opened.provenance()
    assert prov["dataset_symbol"] == "mv" and prov["version_id"] == "manc_v1.0"
    assert prov["manifest_sha256"] == manifest["integrity"]["manifest_sha256"]
    again = mv.open_mv_snapshot(tmp_path)
    assert again.hash_verification[mv.MV_PRODUCT_PATHS[mv.ROLE_META]] == "stamp"
    forced = mv.open_mv_snapshot(tmp_path, verify_hashes=True)
    assert set(forced.hash_verification.values()) == {"hashed"}
    smoke = mv.open_mv_snapshot(tmp_path, verify_hashes=False)
    assert set(smoke.hash_verification.values()) == {"size_only"}


def test_open_without_stamp_cache_is_read_only(tmp_path):
    snap, _ = build_snapshot(tmp_path)
    opened = mv.open_mv_snapshot(tmp_path, use_stamp_cache=False)
    assert opened.hash_verification[mv.MV_PRODUCT_PATHS[mv.ROLE_META]] == "hashed"
    assert not (snap / HASH_STAMP_RELATIVE_DIR).exists()
    again = mv.open_mv_snapshot(tmp_path, use_stamp_cache=False)
    assert again.hash_verification[mv.MV_PRODUCT_PATHS[mv.ROLE_META]] == "hashed"


def test_required_roles_limit_hashing(tmp_path):
    build_snapshot(tmp_path)
    opened = mv.open_mv_snapshot(tmp_path, required_roles=(mv.ROLE_META,))
    assert set(opened.product_paths) == {mv.ROLE_META}
    assert opened.hash_verification[mv.MV_PRODUCT_PATHS[mv.ROLE_EDGELIST]] == "size_only"
    with pytest.raises(MvAdapterError) as exc:
        opened.path(mv.ROLE_EDGELIST)
    assert _code(exc) is MvAdapterErrorCode.PRODUCT_MISSING
    with pytest.raises(MvAdapterError) as exc:
        mv.open_mv_snapshot(tmp_path, required_roles=("synapses",))
    assert _code(exc) is MvAdapterErrorCode.PRODUCT_MISSING


def test_missing_snapshot_and_manifest(tmp_path):
    with pytest.raises(MvAdapterError) as exc:
        mv.open_mv_snapshot(tmp_path)
    assert _code(exc) is MvAdapterErrorCode.SNAPSHOT_MISSING
    build_snapshot(tmp_path, write_manifest=False)
    with pytest.raises(MvAdapterError) as exc:
        mv.open_mv_snapshot(tmp_path)
    assert _code(exc) is MvAdapterErrorCode.MANIFEST_MISSING


def test_relative_storage_root_rejected():
    with pytest.raises(MvAdapterError) as exc:
        mv.open_mv_snapshot("relative/root")
    assert _code(exc) is MvAdapterErrorCode.ROOT_NOT_CONFIGURED


def test_sidecar_missing_and_mismatch(tmp_path):
    snap, _ = build_snapshot(tmp_path)
    sidecar = snap / mv.MV_MANIFEST_SIDECAR_RELATIVE_PATH
    sidecar.write_text("0" * 64 + "\n")
    with pytest.raises(MvAdapterError) as exc:
        mv.open_mv_snapshot(tmp_path)
    assert _code(exc) is MvAdapterErrorCode.INTEGRITY_MISMATCH
    sidecar.unlink()
    with pytest.raises(MvAdapterError) as exc:
        mv.open_mv_snapshot(tmp_path)
    assert _code(exc) is MvAdapterErrorCode.MANIFEST_MISSING


def test_manifest_self_hash_tamper(tmp_path):
    snap, manifest = build_snapshot(tmp_path)
    manifest["notes"] = ["tampered"]
    (snap / mv.MV_MANIFEST_RELATIVE_PATH).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(MvAdapterError) as exc:
        mv.open_mv_snapshot(tmp_path)
    assert _code(exc) is MvAdapterErrorCode.INTEGRITY_MISMATCH


def _set(path, value):
    def apply(manifest):
        node = manifest
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return apply


@pytest.mark.parametrize(
    "override,code",
    [
        (_set(("dataset", "source", "license", "spdx_id"), "CC-BY-NC-4.0"), MvAdapterErrorCode.LICENSE_NOT_PERMITTED),
        (_set(("dataset", "symbol"), "mc"), MvAdapterErrorCode.DATASET_PIN_MISMATCH),
        (_set(("dataset", "version_id"), "manc_v1.2.1"), MvAdapterErrorCode.DATASET_PIN_MISMATCH),
        (_set(("scope", "anatomy"), "brain_and_ventral_nerve_cord"), MvAdapterErrorCode.DATASET_PIN_MISMATCH),
        (_set(("scope", "stage"), "larval"), MvAdapterErrorCode.DATASET_PIN_MISMATCH),
        (_set(("artifact", "relative_root"), "snapshots/mv/../BANC"), MvAdapterErrorCode.PATH_ESCAPE),
        (_set(("artifact", "relative_root"), "snapshots/mv/manc_v1.2"), MvAdapterErrorCode.DATASET_PIN_MISMATCH),
        (_set(("integrity", "verification", "status"), "pending"), MvAdapterErrorCode.INTEGRITY_MISMATCH),
        (_set(("integrity", "verification", "status"), "bogus"), MvAdapterErrorCode.MANIFEST_INVALID),
        (_set(("refresh", "decision"), "rollback"), MvAdapterErrorCode.PROMOTION_STATE_INVALID),
        (_set(("refresh", "next_check_due"), "2000-01-01T00:00:00Z"), MvAdapterErrorCode.PROMOTION_STATE_INVALID),
        (_set(("schema_version",), "fbh-manifest/v0"), MvAdapterErrorCode.MANIFEST_INVALID),
    ],
)
def test_manifest_policy_fails_closed(tmp_path, override, code):
    build_snapshot(tmp_path, manifest_overrides=override)
    with pytest.raises(MvAdapterError) as exc:
        mv.open_mv_snapshot(tmp_path)
    assert _code(exc) is code


def _edit_files(mutator):
    def apply(manifest):
        mutator(manifest["integrity"]["files"])

    return apply


@pytest.mark.parametrize(
    "mutator,code",
    [
        (lambda files: files[0].update(relative_path="../escape.feather"), MvAdapterErrorCode.PATH_ESCAPE),
        (lambda files: files[0].update(relative_path="C:/abs.feather"), MvAdapterErrorCode.PATH_ESCAPE),
        (lambda files: files[0].update(size_bytes=files[0]["size_bytes"] + 1), MvAdapterErrorCode.INTEGRITY_MISMATCH),
        (lambda files: files[0].update(size_bytes=True), MvAdapterErrorCode.MANIFEST_INVALID),
        (lambda files: files[0].update(sha256="ABC"), MvAdapterErrorCode.MANIFEST_INVALID),
        (lambda files: files.append(dict(files[0])), MvAdapterErrorCode.MANIFEST_INVALID),
        (lambda files: files.append({**files[0], "relative_path": "metadata/x.feather.partial"}),
         MvAdapterErrorCode.MANIFEST_INVALID),
        (lambda files: files.append({**files[0], "relative_path": "metadata/missing.feather"}),
         MvAdapterErrorCode.PRODUCT_MISSING),
        (lambda files: files.remove(next(f for f in files if f["role"] == "meta")),
         MvAdapterErrorCode.PRODUCT_MISSING),
        (lambda files: files.clear(), MvAdapterErrorCode.MANIFEST_INVALID),
    ],
)
def test_integrity_entries_fail_closed(tmp_path, mutator, code):
    build_snapshot(tmp_path, manifest_overrides=_edit_files(mutator))
    with pytest.raises(MvAdapterError) as exc:
        mv.open_mv_snapshot(tmp_path)
    assert _code(exc) is code


def test_content_change_with_same_size_is_caught(tmp_path):
    snap, _ = build_snapshot(tmp_path)
    path = snap / mv.MV_PRODUCT_PATHS[mv.ROLE_EDGELIST]
    data = bytearray(path.read_bytes())
    data[-2] = ord("9") if data[-2] != ord("9") else ord("8")
    path.write_bytes(bytes(data))
    with pytest.raises(MvAdapterError) as exc:
        mv.open_mv_snapshot(tmp_path, use_stamp_cache=False)
    assert _code(exc) is MvAdapterErrorCode.INTEGRITY_MISMATCH
    # size-only smoke opens do not hash, by contract
    mv.open_mv_snapshot(tmp_path, verify_hashes=False)


def test_write_manifest_refuses_overwrite(tmp_path):
    snap, manifest = build_snapshot(tmp_path)
    before = (snap / mv.MV_MANIFEST_RELATIVE_PATH).read_text()
    with pytest.raises(MvAdapterError) as exc:
        mv.write_mv_manifest(snap, manifest)
    assert _code(exc) is MvAdapterErrorCode.MANIFEST_EXISTS
    assert (snap / mv.MV_MANIFEST_RELATIVE_PATH).read_text() == before


def test_validate_manifest_uses_injected_clock(tmp_path):
    snap, manifest = build_snapshot(tmp_path)
    with pytest.raises(MvAdapterError) as exc:
        mv.validate_mv_manifest(manifest, snap, now=datetime(2100, 1, 1, tzinfo=timezone.utc))
    assert _code(exc) is MvAdapterErrorCode.PROMOTION_STATE_INVALID


def test_build_manifest_carries_scope_license_and_citation(tmp_path):
    _, manifest = build_snapshot(tmp_path)
    assert manifest["dataset"]["symbol"] == "mv"
    assert manifest["scope"] == {
        "sex": "male", "stage": "adult", "anatomy": "ventral_nerve_cord",
        "evidence_family": "connectome_structural", "claim_tier": "T1_dataset_version_specific",
    }
    assert "eLife 13:RP97769" in manifest["dataset"]["source"]["citation"]
    assert manifest["artifact"]["relative_root"] == "snapshots/mv/manc_v1.0"


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def test_load_neurons_filters_status_and_keeps_string_ids(tmp_path):
    path = tmp_path / "meta.feather"
    write_meta(path)
    frame = mv.load_mv_neurons(path)
    assert str(BIG_BODY) in set(frame["bodyId"])
    assert str(body(15)) not in set(frame["bodyId"])  # Orphan
    assert frame["bodyId"].map(type).eq(str).all()
    assert frame["bodyId"].tolist() == sorted(frame["bodyId"], key=lambda b: b.zfill(20))
    assert "roiInfo" not in frame.columns
    orphans = mv.load_mv_neurons(path, statuses=("Orphan",))
    assert orphans["bodyId"].tolist() == [str(body(15))]


def test_load_neurons_schema_mismatch(tmp_path):
    path = tmp_path / "meta.feather"
    write_meta(path, drop_columns=("somaNeuromere",))
    with pytest.raises(MvAdapterError) as exc:
        mv.load_mv_neurons(path)
    assert _code(exc) is MvAdapterErrorCode.SCHEMA_MISMATCH
    bad = tmp_path / "not.feather"
    bad.write_text("not arrow")
    with pytest.raises(MvAdapterError) as exc:
        mv.load_mv_neurons(bad)
    assert _code(exc) is MvAdapterErrorCode.SCHEMA_MISMATCH
    with pytest.raises(MvAdapterError) as exc:
        mv.load_mv_neurons(tmp_path / "absent.feather")
    assert _code(exc) is MvAdapterErrorCode.PRODUCT_MISSING


def test_neuropil_synweights_matrix(tmp_path):
    path = tmp_path / "meta.feather"
    write_meta(path)
    ids = [str(body(5)), str(body(1)), str(body(16))]
    weights = mv.load_mv_neuropil_synweights(path, ids)
    assert weights.body_ids == tuple(ids)
    assert "CV" not in weights.neuropil_rois and "ProLN(L)" not in weights.neuropil_rois
    assert set(weights.excluded_rois) == {"CV", "ProLN(L)"}
    rows = {b: dict(zip(weights.neuropil_rois, row.tolist())) for b, row in zip(ids, weights.weights)}
    assert rows[str(body(5))]["LTct"] == 600 and rows[str(body(5))]["IntTct"] == 500
    assert rows[str(body(1))]["LegNp(T1)(L)"] == 1000
    assert sum(rows[str(body(16))].values()) == 0  # CV only


def test_neuropil_synweights_unknown_roi_fails_closed(tmp_path):
    path = tmp_path / "meta.feather"
    write_meta(path, roi_names=(*FIXTURE_ROIS, "NotPrimary"))
    with pytest.raises(MvAdapterError) as exc:
        mv.load_mv_neuropil_synweights(path, [str(body(1))])
    assert _code(exc) is MvAdapterErrorCode.REGION_VOCABULARY_UNKNOWN


def test_neuropil_synweights_missing_body_fails_closed(tmp_path):
    path = tmp_path / "meta.feather"
    write_meta(path)
    with pytest.raises(MvAdapterError) as exc:
        mv.load_mv_neuropil_synweights(path, [str(body(1)), "999"])
    assert _code(exc) is MvAdapterErrorCode.SCHEMA_MISMATCH


def test_out_partner_counts(tmp_path):
    path = tmp_path / "edges.csv"
    write_edgelist(path)
    frame, rows = mv.load_mv_out_partner_counts(path)
    assert rows == sum(len(v) for v in EDGES.values())
    by = frame.set_index("bodyId")
    assert by.loc[str(body(3)), "n_post_partners"] == 4
    assert by.loc[str(body(3)), "traced_out_weight"] == 74
    assert by.loc[str(BIG_BODY), "n_post_partners"] == 1


def test_out_partner_counts_fail_closed(tmp_path):
    dup = tmp_path / "dup.csv"
    write_edgelist(dup, duplicate=True)
    with pytest.raises(MvAdapterError) as exc:
        mv.load_mv_out_partner_counts(dup)
    assert _code(exc) is MvAdapterErrorCode.SCHEMA_MISMATCH
    bad = tmp_path / "bad.csv"
    bad.write_text("pre,post,weight\n1,2,3\n")
    with pytest.raises(MvAdapterError) as exc:
        mv.load_mv_out_partner_counts(bad)
    assert _code(exc) is MvAdapterErrorCode.SCHEMA_MISMATCH
    empty = tmp_path / "empty.csv"
    empty.write_text("bodyId_pre,bodyId_post,weight\n")
    with pytest.raises(MvAdapterError) as exc:
        mv.load_mv_out_partner_counts(empty)
    assert _code(exc) is MvAdapterErrorCode.SCHEMA_MISMATCH


def test_nt_argmax_check(tmp_path):
    path = tmp_path / "meta.feather"
    write_meta(path)
    frame = mv.load_mv_neurons(path)
    assert mv.check_nt_argmax(frame) == int(frame["predictedNt"].notna().sum())
    swapped = tmp_path / "swapped.feather"
    write_meta(swapped, probs_override={1: (0.1, 0.8, 0.05, 0.05)})  # predictedNt=ach but gaba is max
    with pytest.raises(MvAdapterError) as exc:
        mv.check_nt_argmax(mv.load_mv_neurons(swapped))
    assert _code(exc) is MvAdapterErrorCode.SCHEMA_MISMATCH
    frame = mv.load_mv_neurons(path)
    frame.loc[frame.index[0], "predictedNt"] = "serotonin"
    with pytest.raises(MvAdapterError) as exc:
        mv.check_nt_argmax(frame)
    assert _code(exc) is MvAdapterErrorCode.NT_VOCABULARY_UNKNOWN


def test_adapter_has_no_network_imports():
    source = Path(mv.__file__).read_text(encoding="utf-8")
    for token in ("requests", "urllib", "http.client", "socket", "curl", "neuprint"):
        assert f"import {token}" not in source


# ---------------------------------------------------------------------------
# Real snapshot (read-only; skipped unless LOCI_FLYBRAIN_STORAGE_ROOT holds it)
# ---------------------------------------------------------------------------

_REAL_ROOT = os.environ.get("LOCI_FLYBRAIN_STORAGE_ROOT", "")
_REAL_MANIFEST = Path(_REAL_ROOT) / "snapshots" / "mv" / "manc_v1.0" / "manifest" / "manifest.json"
real_snapshot = pytest.mark.skipif(
    not _REAL_ROOT or not _REAL_MANIFEST.is_file(),
    reason="real mv snapshot not present under LOCI_FLYBRAIN_STORAGE_ROOT",
)


@real_snapshot
def test_real_snapshot_smoke():
    opened = mv.open_mv_snapshot(_REAL_ROOT, verify_hashes=False)
    assert opened.license_spdx == "CC-BY-4.0"
    assert set(opened.hash_verification.values()) == {"size_only"}
    meta = opened.path(mv.ROLE_META)
    rois = mv.roi_names_in_meta(meta)
    assert len(rois) == 61
    assert all(mv.map_mv_roi(name) for name in rois)
    traced = mv.load_mv_neurons(meta)
    assert len(traced) > 20000
    assert traced["status"].eq("Traced").all()
    sample = traced["bodyId"].head(50).tolist()
    weights = mv.load_mv_neuropil_synweights(meta, sample)
    assert weights.weights.shape == (50, 23)
    assert mv.check_nt_argmax(traced) > 20000
