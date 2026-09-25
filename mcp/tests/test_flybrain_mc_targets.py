"""Synthetic tests for flybrain_mc_targets (mc real-model targets) and the mc structured-feature hook."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_brain_cluster_mc_samples as mcs  # noqa: E402
import flybrain_mc_adapter as mca  # noqa: E402
import flybrain_mc_targets as mct  # noqa: E402
import flybrain_model_eval as fme  # noqa: E402
import flybrain_wiring_features as fwf  # noqa: E402

# type -> (superclass, class, ground_truth NT, primary ROIs, itoleeHl, supertype)
_TYPES = {
    "Mi1": ("ol_intrinsic", None, "acetylcholine", ("ME(R)", "LO(R)"), None, "st_mi"),
    "Mi4": ("ol_intrinsic", None, "gaba", ("ME(R)", "LOP(R)"), None, "st_mi"),
    "Tm1": ("ol_intrinsic", None, "acetylcholine", ("ME(L)", "LO(L)"), None, None),
    "Tm9": ("ol_intrinsic", None, "acetylcholine", ("ME(L)", "LO(L)"), None, None),
    "Dm8": ("ol_intrinsic", None, "glutamate", ("ME(R)",), None, None),
    "LC10": ("visual_projection", None, "acetylcholine", ("LO(R)", "AOTU(R)"), None, None),
    "LC4": ("visual_projection", None, "acetylcholine", ("LO(L)", "PVLP(L)"), None, None),
    "DA1_lPN": ("cb_intrinsic", "ALPN", "acetylcholine", ("AL(R)", "LH(R)", "CA(R)"), "ALad1", None),
    "VA1_PN": ("cb_intrinsic", "ALPN", "acetylcholine", ("AL(L)", "LH(L)"), "ALad1", None),
    "EPG": ("cb_intrinsic", "CX", "acetylcholine", ("EB", "PB"), "DM1", None),
    "PEN": ("cb_intrinsic", "CX", "acetylcholine", ("PB", "NO"), "DM1", None),
    "Delta7": ("cb_intrinsic", "CX", "glutamate", ("PB",), "DM2", None),
    "ORN_DA1": ("cb_sensory", "olfactory", "acetylcholine", ("AL(L)",), None, None),
    "ORN_VA1": ("cb_sensory", "olfactory", "acetylcholine", ("AL(R)",), None, None),
    "IN09A": ("vnc_intrinsic", None, "gaba", ("LegNp(T1)(L)", "LegNp(T2)(L)"), None, None),
    "IN12B": ("vnc_intrinsic", None, "gaba", ("LegNp(T1)(R)",), None, None),
    "IN03A": ("vnc_intrinsic", None, "acetylcholine", ("LegNp(T3)(R)", "LegNp(T2)(R)"), None, None),
    "DNa01": ("descending_neuron", None, "acetylcholine", ("GNG", "LegNp(T1)(R)"), None, None),
    "DNa02": ("descending_neuron", None, "acetylcholine", ("GNG", "LegNp(T1)(L)"), None, None),
    "AN19": ("ascending_neuron", None, "glutamate", ("LegNp(T1)(R)", "GNG"), None, None),
}
_TRUMAN = {"IN09A": "09A", "IN12B": "12B", "IN03A": "03A"}
PER_TYPE = 8


def bid(i):
    return 20000 + i


def _bodies():
    rows = []
    i = 0
    for t, (sc, cls, gt, rois, ito, st) in _TYPES.items():
        for k in range(PER_TYPE):
            i += 1
            rows.append({"idx": i, "type": t, "superclass": sc, "class": cls, "gt": gt, "rois": rois,
                         "ito": ito, "truman": _TRUMAN.get(t), "supertype": st, "side": "L" if k % 2 else "R"})
    # untyped fragments + a glia body (node table only)
    rows.append({"idx": i + 1, "type": None, "superclass": "cb_intrinsic", "class": None, "gt": None,
                 "rois": ("SMP(L)",), "ito": None, "truman": None, "supertype": None, "side": "L"})
    rows.append({"idx": i + 2, "type": "glia_x", "superclass": "glia", "class": None, "gt": None,
                 "rois": ("SMP(R)",), "ito": None, "truman": None, "supertype": None, "side": "R"})
    return rows


def _write_all(root: Path):
    rng = np.random.default_rng(3)
    bodies = _bodies()
    meta = pa.table({
        "bodyId": pa.array([bid(b["idx"]) for b in bodies], pa.int64()),
        "type": pa.array([b["type"] for b in bodies], pa.string()),
        "status": ["Glia" if b["superclass"] == "glia" else "Traced" for b in bodies],
        "statusLabel": pa.array(["Roughly traced" for _ in bodies]).dictionary_encode(),
        "superclass": [b["superclass"] for b in bodies],
        "class": pa.array([b["class"] for b in bodies], pa.string()),
        "somaSide": [b["side"] for b in bodies],
        "rootSide": pa.array([None for _ in bodies], pa.string()),
        "itoleeHl": pa.array([b["ito"] for b in bodies], pa.string()),
        "trumanHl": pa.array([b["truman"] for b in bodies], pa.string()),
        "supertype": pa.array([b["supertype"] for b in bodies], pa.string()),
    })
    paths = {
        mca.ROLE_META: root / "meta.feather",
        mca.ROLE_NT_PREDICTION: root / "nt.feather",
        mca.ROLE_EDGELIST: root / "edges.feather",
        mca.ROLE_NEURON_ROI_INFO: root / "neurons.feather",
        mca.ROLE_DATASET_META: root / "Neuprint_Meta.csv",
    }
    feather.write_feather(meta, str(paths[mca.ROLE_META]))
    nt = pa.table({
        "body": pa.array([bid(b["idx"]) for b in bodies], pa.int64()),
        "cell_type": pa.array([b["type"] for b in bodies], pa.string()),
        "total_nt_predictions": pa.array([100 for _ in bodies], pa.int32()),
        "predicted_nt_confidence": [0.9 for _ in bodies],
        "predicted_nt": [b["gt"] or "unclear" for b in bodies],
        "ground_truth": pa.array([b["gt"] for b in bodies], pa.string()),
        "celltype_total_nt_predictions": pa.array([100 for _ in bodies], pa.int32()),
        "celltype_predicted_nt": [b["gt"] or "unclear" for b in bodies],
        "celltype_predicted_nt_confidence": [0.9 for _ in bodies],
        "consensus_nt": [b["gt"] or "unclear" for b in bodies],
    })
    feather.write_feather(nt, str(paths[mca.ROLE_NT_PREDICTION]))
    roi_json, pres, posts = [], [], []
    for b in bodies:
        info = {}
        for j, roi in enumerate(b["rois"]):
            info[roi] = {"pre": int(rng.integers(20, 400)) // (j + 1), "post": int(rng.integers(50, 600)) // (j + 1)}
        info["CentralBrain"] = {"pre": 999, "post": 999}  # non-primary: ignored
        roi_json.append(json.dumps(info))
        pres.append(sum(v["pre"] for k, v in info.items() if k in mca.MC_PRIMARY_ROIS))
        posts.append(sum(v["post"] for k, v in info.items() if k in mca.MC_PRIMARY_ROIS))
    neurons = pa.table({
        ":ID(Body-ID)": pa.array([bid(b["idx"]) for b in bodies] + [bid(999)], pa.int64()),
        "post:int": pa.array(posts + [1], pa.int32()),
        "pre:int": pa.array(pres + [1], pa.int32()),
        "downstream:int": pa.array([p * 4 for p in pres] + [4], pa.int32()),
        "upstream:int": pa.array(posts + [1], pa.int32()),
        "bodyId:long": pa.array([bid(b["idx"]) for b in bodies] + [bid(999)], pa.int64()),
        "roiInfo:string": roi_json + [json.dumps({"ME(R)": {"pre": 1, "post": 1}})],
    })
    feather.write_feather(neurons, str(paths[mca.ROLE_NEURON_ROI_INFO]))
    ids = [bid(b["idx"]) for b in bodies] + [bid(999)]
    pairs = set()
    for _ in range(1500):
        a, c = rng.choice(len(ids), size=2, replace=False)
        pairs.add((ids[a], ids[c]))
    pairs = sorted(pairs)
    feather.write_feather(pa.table({
        "body_pre": pa.array([p for p, _ in pairs], pa.int64()),
        "body_post": pa.array([q for _, q in pairs], pa.int64()),
        "weight": pa.array([int(rng.integers(1, 40)) for _ in pairs], pa.int64()),
    }), str(paths[mca.ROLE_EDGELIST]))
    rois = sorted(mca.MC_PRIMARY_ROIS)
    joined = ";".join(rois)
    paths[mca.ROLE_DATASET_META].write_text(
        "dataset:string,tag:string,voxelSize:float[],primaryRois:string[],superLevelRois:string[],:Label\n"
        f'male-cns,v1.0,8.0;8.0;8.0,"{joined}","{joined}",Meta\n', encoding="utf-8")
    return paths


@pytest.fixture()
def synth(tmp_path):
    paths = _write_all(tmp_path)
    config = mct.McRealModelConfig(storage_root=None, stamp_dir=None, cache_root=str(tmp_path / "cache"),
                                   max_per_type=6, min_primary_synapses=10, min_types_per_class=2, top_k_rois=6)
    return tmp_path, paths, config


def _eval_config(tmp_path, **kw):
    base = dict(models=(fme.ModelSpec("logreg", ({"C": 1.0},), "temperature"),),
                cv_folds=2, n_bootstrap=50, report_root=str(tmp_path / "reports"), save_models=False,
                random_split_control=False, n_threads=2)
    base.update(kw)
    return fme.EvalConfig(**base)


def _build(tmp_path, paths, config, target, **kw):
    return mct.build_target_dataset(target, config, _eval_config(tmp_path), explicit_paths=paths,
                                    log=lambda *_: None, **kw)


# --------------------------------------------------------------------------- units


def test_parse_roi_info_keeps_primary_only_and_fails_closed():
    rows = mct.parse_roi_info(json.dumps({"ME(R)": {"pre": 3, "post": 4}, "OL(R)": {"pre": 9, "post": 9},
                                          "LO(R)": {"pre": 0, "post": 0}}))
    assert rows == [("ME(R)", 3, 4)]
    assert mct.parse_roi_info(None) == []
    with pytest.raises(mca.McAdapterError):
        mct.parse_roi_info("{not json")
    with pytest.raises(mca.McAdapterError):
        mct.parse_roi_info(json.dumps({"ME(R)": 5}))


def test_roi_feature_frame_fractions_and_top_roi():
    long = pd.DataFrame({"root_id": ["1", "1", "1", "2"], "roi": ["ME(R)", "ME(L)", "LO(R)", "EB"],
                         "pre": [10, 10, 0, 5], "post": [35, 5, 40, 5]})
    out = mct.roi_feature_frame(long, top_k=1).set_index("root_id")
    pre_cols = [c for c in out.columns if c.startswith("roi__pre_frac__")]
    assert np.allclose(out.loc["1", pre_cols].sum(), 1.0)
    assert out.loc["1", "roi__pre_frac__brain_me"] == pytest.approx(1.0)  # hemispheres merged
    assert out.loc["1", "roi__left_share"] == pytest.approx(15 / 100)
    assert np.isnan(out.loc["2", "roi__left_share"])  # EB has no hemisphere
    assert out.loc["1", "primary_neuropil"] == "me" and out.loc["1", "subdivision"] == "optic_lobe"
    assert out.loc["1", "degree__roi_n_primary_rois"] == 3
    assert out.loc["1", "total_primary_synapses"] == 100


def test_cap_per_type_is_deterministic_and_order_free():
    frame = pd.DataFrame({"root_id": [str(i) for i in range(50)], "cell_type": ["A"] * 30 + ["B"] * 20})
    a = mct.cap_per_type(frame, max_per_type=5, salt="s")
    b = mct.cap_per_type(frame.sample(frac=1.0, random_state=1), max_per_type=5, salt="s")
    assert a["root_id"].tolist() == b["root_id"].tolist()
    assert a.groupby("cell_type").size().tolist() == [5, 5]
    assert a["root_id"].tolist() != mct.cap_per_type(frame, max_per_type=5, salt="t")["root_id"].tolist()


@pytest.mark.parametrize("target", mct.MC_REAL_TARGETS)
def test_allowed_columns_respect_registry_and_extra_rules(target):
    columns = ["degree__out_weight_total", "degree__roi_pre_share", "out_comp__sensory", "recip__partner_frac",
               "roi__pre_frac__brain_me", "roi__left_share", *mct.CATEGORICAL_COLUMNS, "cell_type", "supertype",
               "root_id", "node_id"]
    kept, dropped = mct.allowed_columns(target, columns)
    fwf.assert_features_allowed(kept, target)
    assert not {"cell_type", "supertype", "root_id", "node_id"} & set(kept)
    assert not set(kept) & mct.EXTRA_EXCLUDED[target]
    if target == "connectivity_tier":
        assert not [c for c in kept if c.startswith("degree__")]
    if target == "region_specialization_tier":
        assert not [c for c in kept if "roi" in c or c in {"primary_neuropil", "cns_division", "subdivision"}]
    if target in ("super_class", "cell_class"):
        assert not {"super_class", "cell_class", "hemilineage"} & set(kept)
    if target == "nt_ground_truth":
        assert "hemilineage" not in kept and "super_class" in kept


def test_held_out_mask_covers_non_sample_nodes_of_held_out_groups():
    nodes = pd.DataFrame({"root_id": ["1", "2", "3", "4", "5"], "cell_type": ["A", "A", "B", "C", None],
                          "hemilineage_group": [None, None, "hl1", None, "hl1"],
                          "supertype": [None, None, None, "st", None]})
    frame = nodes.iloc[[0, 2]].copy()  # samples: 1 (type A), 3 (hl1)
    ids = mct._held_out_mask_ids(nodes, frame, {"1", "3"})
    assert ids == ["1", "2", "3", "5"]  # 2 shares type A, 5 shares hemilineage hl1


# --------------------------------------------------------------------------- end to end (synthetic)


def test_nt_ground_truth_dataset_has_no_classifier_outputs(synth):
    tmp_path, paths, config = synth
    build = _build(tmp_path, paths, config, "nt_ground_truth")
    data = build.data
    cols = list(data.features.columns)
    fwf.assert_features_allowed(cols, "nt_ground_truth")
    assert not [c for c in cols if "nt" in c.split("__")[-1].split("_") or "predicted" in c or "consensus" in c]
    assert "hemilineage" not in cols and "cell_type" not in cols
    assert set(data.labels) == {"ach", "gaba", "glut"}
    assert data.group_keys == mct.GROUP_KEYS
    # glia and untyped bodies are never samples; max_per_type caps every type
    assert all(str(bid(len(_TYPES) * PER_TYPE + k)) not in " ".join(data.sample_ids) for k in (1, 2))
    assert max(pd.Series([g["cell_type"] for g in data.group_values]).value_counts()) <= config.max_per_type
    # text view only carries allowed keys
    keys = {tok for text in data.text for tok in text.split()[0::2]}
    fwf.assert_features_allowed(sorted(keys), "nt_ground_truth")
    assert "hemilineage" not in keys


def test_split_never_straddles_groups(synth):
    tmp_path, paths, config = synth
    build = _build(tmp_path, paths, config, "nt_ground_truth")
    data = build.data
    plan = fme.plan_grouped_split(list(data.sample_ids), list(data.group_values), data.group_keys, _eval_config(tmp_path))
    where = {}
    for split, ids in plan.items():
        for sid in ids:
            where[sid] = split
    seen: dict[tuple[str, str], str] = {}
    for sid, groups in zip(data.sample_ids, data.group_values):
        for key, value in groups.items():
            if value is None:
                continue
            assert seen.setdefault((key, value), where[sid]) == where[sid]


def test_super_class_masks_held_out_partner_categories(synth):
    tmp_path, paths, config = synth
    build = _build(tmp_path, paths, config, "super_class")
    notes = build.data.notes
    assert notes["masked_partner_category_nodes"] > 0
    assert "masked_split_ids_sha256" in notes
    cols = list(build.data.features.columns)
    assert not {"super_class", "cell_class", "hemilineage"} & set(cols)
    assert [c for c in cols if c.startswith("out_comp__")]  # wiring features present
    # unmasked build differs from the masked one (the mask reached the features)
    unmasked = _build(tmp_path, paths, config, "nt_ground_truth")
    assert notes["wiring_fingerprint"] != unmasked.data.notes["wiring_fingerprint"]


@pytest.mark.parametrize("target", ["connectivity_tier", "region_specialization_tier"])
def test_legacy_targets_reuse_unchanged_labels(synth, target):
    tmp_path, paths, config = synth
    config = mct.McRealModelConfig(**{**config.as_dict(), "legacy_max_samples": 10_000})
    build = _build(tmp_path, paths, config, target)
    cols = list(build.data.features.columns)
    fwf.assert_features_allowed(cols, target)
    if target == "connectivity_tier":
        assert set(build.data.labels) <= {mcs.LABEL_HIGH, mcs.LABEL_BASELINE}
        assert not [c for c in cols if c.startswith("degree__")]
    else:
        assert set(build.data.labels) <= {mcs.LABEL_SPECIALIZED, mcs.LABEL_DISTRIBUTED}
        assert not [c for c in cols if c.startswith("roi__")]


def test_run_evaluation_end_to_end_writes_only_under_report_root(synth):
    tmp_path, paths, config = synth
    ec = _eval_config(tmp_path, ablation=True)
    build = mct.build_target_dataset("nt_ground_truth", config, ec, explicit_paths=paths, log=lambda *_: None)
    report = fme.run_evaluation(build.data, ec)
    assert report["summary"] and report["summary"][0]["target"] == "nt_ground_truth"
    assert (tmp_path / "reports" / "mc" / "nt_ground_truth" / "report.json").is_file()
    assert "shuffle_control" in report["models"]["logreg"]
    assert not list(tmp_path.glob("**/snapshots"))
    # R1: the mc ground_truth column is measured but uncertain (E5); the notes carry it for the harness gate
    assert build.data.notes["label_provenance"] == "measured" and build.data.notes["provenance_uncertain"] is True
    assert report["label_provenance"]["gate_applies"] is True


def test_every_mc_target_declares_provenance():
    import flybrain_target_registry as ftr

    assert set(mct.TARGET_LABEL_PROVENANCE) == set(mct.MC_REAL_TARGETS)
    assert mct.TARGET_LABEL_PROVENANCE["cell_class"] == "connectivity_defined"
    assert not ftr.target_provenance("mc", "connectivity_tier").gateable


def test_feature_filter_and_sparsity_control(synth):
    tmp_path, paths, config = synth
    wiring_fams = ("degree", "out_comp", "in_comp", "recip", "out2_comp", "in2_comp")
    wiring_only = _build(tmp_path, paths, config, "nt_ground_truth", feature_filter=wiring_fams)
    assert {fwf.feature_family(c) for c in wiring_only.data.features.columns} <= set(wiring_fams)
    control = _build(tmp_path, paths, config, "connectivity_tier", feature_filter=("sparsity",))
    assert sorted(control.data.features.columns) == ["sparsity__n_nonzero_in_comp", "sparsity__n_nonzero_out_comp"]
    full = _build(tmp_path, paths, config, "connectivity_tier")
    assert not [c for c in full.data.features.columns if c.startswith("sparsity__")]  # control only


def test_cache_refuses_snapshot_paths(tmp_path):
    bad = tmp_path / "snapshots" / "mc"
    source = tmp_path / "neurons.feather"
    source.write_bytes(b"x")
    with pytest.raises(ValueError, match="snapshots"):
        mct.load_roi_long(source, ["1"], product_sha256="x", cache_root=bad)
    assert not bad.exists()


# --------------------------------------------------------------------------- mc_samples structured hook


def test_attach_structured_features_drops_excluded_and_keeps_text():
    samples = [{"sample_id": "s1", "input_text": "dataset mcns10 side left", "metadata": {"root_id": "7"}},
               {"sample_id": "s2", "input_text": "dataset mcns10 side right", "metadata": {"root_id": "8"}}]
    frame = pd.DataFrame({"node_id": ["7", "8"], "degree__out_weight_total": [5.0, 9.0],
                          "out_comp__sensory": [0.5, float("nan")]})
    out = mcs.attach_structured_features(samples, frame, objective="connectivity_tier")
    assert out[0]["input_text"] == samples[0]["input_text"]
    assert out[0]["features"] == {"out_comp__sensory": 0.5}
    assert out[1]["features"] == {"out_comp__sensory": None}
    assert "features" not in samples[0]  # inputs untouched
    with pytest.raises(ValueError):
        mcs.attach_structured_features([{"metadata": {"root_id": "9"}}], frame, objective="connectivity_tier")
