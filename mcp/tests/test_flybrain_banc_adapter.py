import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_banc_adapter as banc  # noqa: E402
from flybrain_banc_adapter import BancAdapterError, BancAdapterErrorCode  # noqa: E402

_ID = "7205759400000000{:02d}"

# (idx, root_region, curated region, super_class, cell_class, flow, hemilineage, side, proofread)
_META_ROWS = [
    (1, "ITO_optic_ME_R", "optic_lobe", "optic_lobe_intrinsic", "medulla_intrinsic", "intrinsic", None, "right", True),
    (2, "ITO_optic_ME_L", "optic_lobe", "optic_lobe_intrinsic", "transmedullary", "intrinsic", None, "left", True),
    (3, "ITO_optic_ME_R", "optic_lobe", "visual_projection", "lobula_columnar", "intrinsic", None, "right", True),
    (4, "ITO_midbrain_AL_L", "central_brain", "sensory", "olfactory_receptor_neuron", "afferent", None, "left", True),
    (5, "ITO_midbrain_AL_R", "central_brain", "central_brain_intrinsic", "antennal_lobe_local", "intrinsic", "ALl1", "right", True),
    (6, "ITO_midbrain_AL_L", "central_brain", "central_brain_intrinsic", "antennal_lobe_local", "intrinsic", "ALl1", "left", True),
    (7, "COURT_vnc_ProNM-T1", "ventral_nerve_cord", "ventral_nerve_cord_intrinsic", "single_leg_neuromere", "intrinsic", "09A", "left", True),
    (8, "COURT_vnc_ProNM-T1", "ventral_nerve_cord", "motor", "leg_motor", "efferent", "15B", "right", True),
    (9, "MANC_vnc_LNp_T1_L", "ventral_nerve_cord", "ventral_nerve_cord_intrinsic", "single_leg_neuromere", "intrinsic", "12B", "left", True),
    (10, "MANC_vnc_LNp_T1_R", "ventral_nerve_cord", "ascending", "ascending_neuron", "intrinsic", "19A", "right", True),
    (11, None, "central_brain", "central_brain_intrinsic", None, "intrinsic", None, "left", True),  # no root_region
    (12, "ITO_optic_ME_R", "ventral_nerve_cord", "ascending", None, "intrinsic", None, "right", True),  # conflict
    (13, "ITO_midbrain_AL_L", "central_brain", "glia", "astrocyte", None, None, "left", True),  # non-neuronal
    (14, "COURT_vnc_ProNM-T1", "ventral_nerve_cord", "motor", "leg_motor", "efferent", "15B", "left", False),  # unproofread
    (15, "MANC_vnc_LNp_T1_L", "ventral_nerve_cord", "descending", "descending_neuron", "intrinsic", None, "left", True),
    (16, "ITO_midbrain_AL_R", "central_brain", "central_brain_intrinsic", "antennal_lobe_local", "intrinsic", "ALv1", "right", True),
]

# pre idx -> list of (post idx, count)
_EDGES = {
    1: [(2, 40), (3, 25), (4, 5)],
    2: [(1, 6), (3, 4)],
    3: [(1, 90), (2, 70), (5, 30)],
    4: [(5, 3), (6, 8)],
    5: [(4, 55), (6, 60), (16, 20)],
    6: [(5, 2)],
    7: [(8, 80), (9, 60)],
    8: [(7, 5)],
    9: [(7, 12), (10, 3)],
    10: [(9, 150), (15, 40)],
    11: [(1, 50)],
    12: [(1, 50)],
    13: [(4, 50)],
    14: [(7, 50)],
    15: [(9, 7)],
    16: [(5, 9), (6, 33)],
}

# idx -> (predicted nt, score, count)
_NT = {
    1: ("acetylcholine", 0.91, 40),
    2: ("glutamate", 0.66, 22),
    3: ("acetylcholine", 0.72, 80),
    4: ("acetylcholine", 0.95, 12),
    5: ("gaba", 0.88, 120),
    6: ("gaba", 0.61, 9),
    7: ("glutamate", 0.8, 70),
    8: ("glutamate", 0.55, 5),
    9: ("gaba", 0.77, 15),
    10: ("acetylcholine", 0.83, 150),
    11: ("acetylcholine", 0.9, 50),
    12: ("acetylcholine", 0.9, 50),
    13: ("gaba", 0.9, 50),
    14: ("glutamate", 0.9, 50),
    15: ("histamine", 0.52, 7),
    16: ("dopamine", 0.7, 30),
}
_NT_ORDER = ("acetylcholine", "dopamine", "gaba", "glutamate", "histamine", "octopamine", "serotonin", "tyramine")


def rid(idx):
    return _ID.format(idx)


def write_meta(path, rows=_META_ROWS):
    cols = list(zip(*rows))
    table = pa.table(
        {
            "banc_888_id": [rid(i) for i in cols[0]],
            "root_region": list(cols[1]),
            "region": list(cols[2]),
            "super_class": list(cols[3]),
            "cell_class": list(cols[4]),
            "flow": list(cols[5]),
            "hemilineage": list(cols[6]),
            "side": list(cols[7]),
            "proofread": list(cols[8]),
            "cell_type": ["ct" for _ in rows],
            "output_connections": [1 for _ in rows],
        }
    )
    feather.write_feather(table, str(path))


def write_edgelist(path, edges=_EDGES):
    pre, post, count = [], [], []
    for p, targets in sorted(edges.items()):
        for q, c in targets:
            pre.append(rid(p))
            post.append(rid(q))
            count.append(c)
    post_totals = {}
    pre_totals = {}
    for p, q, c in zip(pre, post, count):
        post_totals[q] = post_totals.get(q, 0) + c
        pre_totals[p] = pre_totals.get(p, 0) + c
    table = pa.table(
        {
            "pre": pre,
            "post": post,
            "count": pa.array(count, pa.int32()),
            "norm": [c / post_totals[q] for q, c in zip(post, count)],
            "post_count": pa.array([post_totals[q] for q in post], pa.int32()),
            "pre_count": pa.array([pre_totals[p] for p in pre], pa.int32()),
        }
    )
    feather.write_feather(table, str(path))


def write_nt(path, nt=_NT, duplicate_idx=(5,), conflicting_idx=()):
    header = ["root_id", *_NT_ORDER, "neurotransmitter_predicted", "neurotransmitter_score", "count",
              "supervoxel_id", "position", "cell_type", "cell_type_neurotransmitter_predicted",
              "cell_type_neurotransmitter_score"]
    lines = [",".join(header)]
    for idx, (name, score, count) in sorted(nt.items()):
        per_nt = [str(count if n == name else 0) for n in _NT_ORDER]
        row = [rid(idx), *per_nt, name, str(score), str(count), f"9{idx:03d}", f'"1, 2, {idx}"', "", "", ""]
        lines.append(",".join(row))
        if idx in duplicate_idx:
            dup = list(row)
            dup[-5] = f"8{idx:03d}"
            lines.append(",".join(dup))
        if idx in conflicting_idx:
            bad = list(row)
            bad[len(_NT_ORDER) + 2] = "0.1"
            lines.append(",".join(bad))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_snapshot(storage_root, *, manifest_overrides=None, write_manifest=True):
    """Create a synthetic BANC snapshot + verified manifest under storage_root."""
    snap = Path(storage_root) / "snapshots" / "BANC" / "banc_888"
    (snap / "source" / "compiled_data").mkdir(parents=True, exist_ok=True)
    (snap / "metadata").mkdir(parents=True, exist_ok=True)
    write_edgelist(snap / banc.BANC_PRODUCT_PATHS[banc.ROLE_EDGELIST_V3])
    write_nt(snap / banc.BANC_PRODUCT_PATHS[banc.ROLE_NT_PREDICTION])
    write_meta(snap / banc.BANC_PRODUCT_PATHS[banc.ROLE_META])
    files = []
    for role, rel in sorted(banc.BANC_PRODUCT_PATHS.items()):
        path = snap / rel
        files.append({"relative_path": rel, "size_bytes": path.stat().st_size, "sha256": banc.sha256_file(path),
                      "role": role})
    manifest = banc.build_banc_manifest(
        files=files,
        source_uri="synthetic://banc",
        retrieved_at="2026-09-23T00:00:00Z",
        run_id="run-test",
        generated_at="2026-09-23T00:00:00Z",
        next_check_due="2099-01-01T00:00:00Z",
    )
    if manifest_overrides:
        manifest_overrides(manifest)
        manifest["integrity"]["manifest_sha256"] = banc.manifest_digest(manifest)
    if write_manifest:
        banc.write_banc_manifest(snap, manifest)
    return snap, manifest


def _code(exc_info):
    return exc_info.value.code


# ---------------------------------------------------------------------------
# Region / NT vocabularies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,division,subdivision,neuropil,side,region_id",
    [
        ("ITO_optic_ME_R", "brain", "optic_lobe", "ME", "right", "brain_me"),
        ("ITO_optic_LOP_L", "brain", "optic_lobe", "LOP", "left", "brain_lop"),
        ("ITO_midbrain_MB_CA_L", "brain", "central_brain", "MB_CA", "left", "brain_mb_ca"),
        ("ITO_midbrain_GNG", "brain", "central_brain", "GNG", None, "brain_gng"),
        ("COURT_vnc_ProNM-T1", "nerve_cord", "ventral_nerve_cord", "ProNM-T1", None, "nerve_cord_pronm_t1"),
        ("MANC_vnc_LNp_T1_L", "nerve_cord", "ventral_nerve_cord", "LNp_T1", "left", "nerve_cord_lnp_t1"),
        ("MANC_vnc_HTct_UTct_T3_R", "nerve_cord", "ventral_nerve_cord", "HTct_UTct_T3", "right", "nerve_cord_htct_utct_t3"),
    ],
)
def test_map_banc_region_explicit_vocabulary(raw, division, subdivision, neuropil, side, region_id):
    mapped = banc.map_banc_region(raw)
    assert (mapped.division, mapped.subdivision, mapped.neuropil, mapped.side, mapped.region_id) == (
        division, subdivision, neuropil, side, region_id,
    )


@pytest.mark.parametrize("raw", ["", "FAFB_brain_ME_R", "ITO_optic_", "ITO_optic__R", "vnc_T1"])
def test_map_banc_region_unknown_fails_closed(raw):
    with pytest.raises(BancAdapterError) as exc:
        banc.map_banc_region(raw)
    assert _code(exc) is BancAdapterErrorCode.REGION_VOCABULARY_UNKNOWN
    assert str(exc.value).startswith("[REGION_VOCABULARY_UNKNOWN] ")


def test_curated_region_division():
    assert banc.curated_region_division("ventral_nerve_cord") == ("nerve_cord", "ventral_nerve_cord")
    assert banc.curated_region_division(None) is None
    assert banc.curated_region_division(float("nan")) is None
    with pytest.raises(BancAdapterError) as exc:
        banc.curated_region_division("neck")
    assert _code(exc) is BancAdapterErrorCode.REGION_VOCABULARY_UNKNOWN


def test_nt_short_codes():
    assert banc.nt_short_code("Acetylcholine") == "ach"
    assert banc.nt_short_code("tyramine") == "tyr"
    with pytest.raises(BancAdapterError) as exc:
        banc.nt_short_code("nitric_oxide")
    assert _code(exc) is BancAdapterErrorCode.NT_VOCABULARY_UNKNOWN


# ---------------------------------------------------------------------------
# Snapshot / manifest
# ---------------------------------------------------------------------------


def test_open_snapshot_verified(tmp_path):
    snap, manifest = build_snapshot(tmp_path)
    opened = banc.open_banc_snapshot(tmp_path)
    assert opened.snapshot_root == snap.resolve()
    assert opened.manifest_sha256 == manifest["integrity"]["manifest_sha256"]
    assert opened.license_spdx == "CC-BY-4.0"
    assert set(opened.product_paths) == set(banc.BANC_PRODUCT_PATHS)
    prov = opened.provenance()
    assert prov["dataset_symbol"] == "banc" and prov["version_id"] == "banc_888"
    assert (snap / "manifest" / "manifest.sha256").read_text().strip() == manifest["integrity"]["manifest_sha256"]


def test_open_snapshot_missing_manifest(tmp_path):
    build_snapshot(tmp_path, write_manifest=False)
    with pytest.raises(BancAdapterError) as exc:
        banc.open_banc_snapshot(tmp_path)
    assert _code(exc) is BancAdapterErrorCode.MANIFEST_MISSING


def test_open_snapshot_missing_dir(tmp_path):
    with pytest.raises(BancAdapterError) as exc:
        banc.open_banc_snapshot(tmp_path)
    assert _code(exc) is BancAdapterErrorCode.SNAPSHOT_MISSING


def test_tampered_product_fails_integrity(tmp_path):
    snap, _ = build_snapshot(tmp_path)
    nt_path = snap / banc.BANC_PRODUCT_PATHS[banc.ROLE_NT_PREDICTION]
    data = bytearray(nt_path.read_bytes())
    data[-3] = ord("9") if data[-3] != ord("9") else ord("8")
    nt_path.write_bytes(bytes(data))
    with pytest.raises(BancAdapterError) as exc:
        banc.open_banc_snapshot(tmp_path)
    assert _code(exc) is BancAdapterErrorCode.INTEGRITY_MISMATCH


def test_manifest_self_hash_mismatch(tmp_path):
    snap, manifest = build_snapshot(tmp_path, write_manifest=False)
    manifest["notes"] = ["edited after hashing"]
    banc.write_banc_manifest(snap, manifest)
    with pytest.raises(BancAdapterError) as exc:
        banc.open_banc_snapshot(tmp_path)
    assert _code(exc) is BancAdapterErrorCode.INTEGRITY_MISMATCH


@pytest.mark.parametrize(
    "mutate,code",
    [
        (lambda m: m["dataset"]["source"]["license"].update(spdx_id="UNREVIEWED"), BancAdapterErrorCode.LICENSE_NOT_PERMITTED),
        (lambda m: m["dataset"].update(version_id="banc_626"), BancAdapterErrorCode.DATASET_PIN_MISMATCH),
        (lambda m: m["dataset"].update(symbol="fw"), BancAdapterErrorCode.DATASET_PIN_MISMATCH),
        (lambda m: m["integrity"]["verification"].update(status="pending"), BancAdapterErrorCode.INTEGRITY_MISMATCH),
        (lambda m: m["refresh"].update(decision="rollback"), BancAdapterErrorCode.PROMOTION_STATE_INVALID),
        (lambda m: m["refresh"].update(next_check_due="2020-01-01T00:00:00Z"), BancAdapterErrorCode.PROMOTION_STATE_INVALID),
        (lambda m: m["artifact"].update(relative_root="snapshots/fw/flywire783"), BancAdapterErrorCode.DATASET_PIN_MISMATCH),
        (lambda m: m["integrity"]["files"].append(dict(m["integrity"]["files"][0], relative_path="../outside.csv")),
         BancAdapterErrorCode.PATH_ESCAPE),
        (lambda m: m["integrity"]["files"].append(dict(m["integrity"]["files"][0], relative_path="x.feather.partial")),
         BancAdapterErrorCode.MANIFEST_INVALID),
        (lambda m: m["integrity"]["files"].append(dict(m["integrity"]["files"][0],
                                                       relative_path=m["integrity"]["files"][0]["relative_path"].upper())),
         BancAdapterErrorCode.MANIFEST_INVALID),
        (lambda m: m["integrity"]["files"].pop(0), BancAdapterErrorCode.PRODUCT_MISSING),
        (lambda m: m["integrity"]["files"][0].update(size_bytes=1), BancAdapterErrorCode.INTEGRITY_MISMATCH),
    ],
)
def test_manifest_gates_fail_closed(tmp_path, mutate, code):
    build_snapshot(tmp_path, manifest_overrides=mutate)
    with pytest.raises(BancAdapterError) as exc:
        banc.open_banc_snapshot(tmp_path)
    assert _code(exc) is code


def test_required_roles_subset(tmp_path):
    build_snapshot(tmp_path, manifest_overrides=lambda m: m["integrity"]["files"].__setitem__(
        slice(None), [f for f in m["integrity"]["files"] if "neurotransmitter" not in f["relative_path"]]))
    opened = banc.open_banc_snapshot(tmp_path, required_roles=(banc.ROLE_EDGELIST_V3, banc.ROLE_META))
    assert set(opened.product_paths) == {banc.ROLE_EDGELIST_V3, banc.ROLE_META}
    with pytest.raises(BancAdapterError) as exc:
        banc.open_banc_snapshot(tmp_path)
    assert _code(exc) is BancAdapterErrorCode.PRODUCT_MISSING


def test_write_manifest_never_overwrites(tmp_path):
    snap, manifest = build_snapshot(tmp_path)
    before = (snap / "manifest" / "manifest.json").read_text()
    with pytest.raises(BancAdapterError) as exc:
        banc.write_banc_manifest(snap, manifest)
    assert _code(exc) is BancAdapterErrorCode.MANIFEST_EXISTS
    assert (snap / "manifest" / "manifest.json").read_text() == before


def test_sidecar_mismatch(tmp_path):
    snap, _ = build_snapshot(tmp_path)
    (snap / "manifest" / "manifest.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(BancAdapterError) as exc:
        banc.open_banc_snapshot(tmp_path)
    assert _code(exc) is BancAdapterErrorCode.INTEGRITY_MISMATCH


def test_relative_storage_root_rejected():
    with pytest.raises(BancAdapterError) as exc:
        banc.open_banc_snapshot("relative/root")
    assert _code(exc) is BancAdapterErrorCode.ROOT_NOT_CONFIGURED


def test_validate_manifest_uses_injected_clock(tmp_path):
    snap, manifest = build_snapshot(tmp_path)
    with pytest.raises(BancAdapterError) as exc:
        banc.validate_banc_manifest(manifest, snap, now=datetime(2100, 1, 1, tzinfo=timezone.utc))
    assert _code(exc) is BancAdapterErrorCode.PROMOTION_STATE_INVALID


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def test_outgoing_totals(tmp_path):
    path = tmp_path / "edges.feather"
    write_edgelist(path)
    frame, rows = banc.load_banc_outgoing_totals(path)
    assert rows == sum(len(v) for v in _EDGES.values())
    got = dict(zip(frame["root_id"], zip(frame["total_out_synapses"], frame["n_post_partners"])))
    assert got[rid(3)] == (190, 3)
    assert got[rid(6)] == (2, 1)
    assert list(frame["root_id"]) == sorted(frame["root_id"])
    assert frame["root_id"].map(type).eq(str).all()


def test_edgelist_missing_column(tmp_path):
    path = tmp_path / "edges.feather"
    feather.write_feather(pa.table({"pre": ["1"], "post": ["2"], "count": [3]}), str(path))
    with pytest.raises(BancAdapterError) as exc:
        banc.load_banc_outgoing_totals(path)
    assert _code(exc) is BancAdapterErrorCode.SCHEMA_MISMATCH
    assert "pre_count" in str(exc.value)


def test_edgelist_not_feather(tmp_path):
    path = tmp_path / "edges.feather"
    path.write_text("pre,post\n1,2\n")
    with pytest.raises(BancAdapterError) as exc:
        banc.load_banc_outgoing_totals(path)
    assert _code(exc) is BancAdapterErrorCode.SCHEMA_MISMATCH


def test_missing_product_file(tmp_path):
    with pytest.raises(BancAdapterError) as exc:
        banc.load_banc_meta(tmp_path / "nope.feather")
    assert _code(exc) is BancAdapterErrorCode.PRODUCT_MISSING


def test_meta_reader_and_duplicate_ids(tmp_path):
    path = tmp_path / "meta.feather"
    write_meta(path)
    frame = banc.load_banc_meta(path)
    assert len(frame) == len(_META_ROWS)
    assert frame["banc_888_id"].iloc[0] == rid(1)
    dup = tmp_path / "dup.feather"
    write_meta(dup, rows=_META_ROWS + [_META_ROWS[0]])
    with pytest.raises(BancAdapterError) as exc:
        banc.load_banc_meta(dup)
    assert _code(exc) is BancAdapterErrorCode.SCHEMA_MISMATCH


def test_nt_reader_collapses_identical_duplicates(tmp_path):
    path = tmp_path / "nt.csv"
    write_nt(path, duplicate_idx=(5,))
    frame = banc.load_banc_nt_predictions(path)
    assert frame["root_id"].is_unique
    assert int(frame.loc[frame["root_id"] == rid(5), "nt_row_multiplicity"].iloc[0]) == 2
    assert int(frame.loc[frame["root_id"] == rid(1), "nt_row_multiplicity"].iloc[0]) == 1
    # ids are strings, not float-promoted
    assert frame["root_id"].iloc[0] == rid(1)


def test_nt_reader_rejects_conflicting_duplicates(tmp_path):
    path = tmp_path / "nt.csv"
    write_nt(path, duplicate_idx=(), conflicting_idx=(3,))
    with pytest.raises(BancAdapterError) as exc:
        banc.load_banc_nt_predictions(path)
    assert _code(exc) is BancAdapterErrorCode.SCHEMA_MISMATCH


def test_nt_reader_missing_column(tmp_path):
    path = tmp_path / "nt.csv"
    path.write_text("root_id,acetylcholine\n1,2\n")
    with pytest.raises(BancAdapterError) as exc:
        banc.load_banc_nt_predictions(path)
    assert _code(exc) is BancAdapterErrorCode.SCHEMA_MISMATCH


def test_manifest_json_is_schema_shaped(tmp_path):
    snap, _ = build_snapshot(tmp_path)
    manifest = json.loads((snap / "manifest" / "manifest.json").read_text())
    for key in ("schema_version", "manifest_id", "generated_at", "storage_root_env", "artifact", "dataset",
                "scope", "integrity", "refresh", "lineage"):
        assert key in manifest
    assert manifest["artifact"]["path_template"].startswith("$LOCI_FLYBRAIN_STORAGE_ROOT\\")
    assert manifest["dataset"]["symbol"] == "BANC"
    assert manifest["dataset"]["source"]["license"]["spdx_id"] == "CC-BY-4.0"
    assert "10.7910/DVN/7WTH1N" in manifest["dataset"]["source"]["citation"]
    assert manifest["scope"]["anatomy"] == "brain_and_ventral_nerve_cord"


_REAL_ROOT = os.environ.get("LOCI_FLYBRAIN_STORAGE_ROOT", "")


@pytest.mark.skipif(
    not _REAL_ROOT or not (Path(_REAL_ROOT) / "snapshots" / "BANC" / "banc_888" / "manifest" / "manifest.json").is_file(),
    reason="real BANC snapshot not present under LOCI_FLYBRAIN_STORAGE_ROOT",
)
def test_real_snapshot_smoke():
    opened = banc.open_banc_snapshot(_REAL_ROOT, verify_hashes=False)
    assert opened.license_spdx == "CC-BY-4.0"
    for path in opened.product_paths.values():
        assert path.is_file()


def test_missing_sidecar_fails_closed(tmp_path):
    snap, _ = build_snapshot(tmp_path)
    (snap / "manifest" / "manifest.sha256").unlink()
    with pytest.raises(BancAdapterError) as exc:
        banc.open_banc_snapshot(tmp_path)
    assert _code(exc) is BancAdapterErrorCode.MANIFEST_MISSING
