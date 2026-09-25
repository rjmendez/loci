"""Tests for flybrain_mv_targets (MANC real-model targets) on small synthetic fixtures."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import flybrain_brain_cluster_baselines as fbb  # noqa: E402
import flybrain_model_eval as fme  # noqa: E402
import flybrain_mv_targets as mt  # noqa: E402
import flybrain_wiring_features as fwf  # noqa: E402
from flybrain_mv_adapter import MvAdapterError, MvAdapterErrorCode  # noqa: E402
from test_flybrain_mv_adapter import write_edgelist, write_meta  # noqa: E402

_CLASSES = ("intrinsic neuron", "sensory neuron", "descending neuron", "ascending neuron", "motor neuron")
_HEMILINEAGES = ("01A", "03A", "06B", "09A", "12B", "19A")
_NT_BY_HL = {"01A": "acetylcholine", "03A": "acetylcholine", "06B": "gaba", "09A": "gaba", "12B": "glutamate",
             "19A": "glutamate"}
_ROIS = ("ANm", "CV", "IntTct", "LTct", "LegNp(T1)(L)", "LegNp(T1)(R)", "ProLN(L)")


def _synthetic(n_types=36, per_type=3, seed=7):
    """MANC-like neurons (the adapter-test tuple layout) + edges, with class/hemilineage structure in wiring."""
    rng = np.random.default_rng(seed)
    neurons, edges = {}, {}
    idx = 1
    for t in range(n_types):
        klass = _CLASSES[t % len(_CLASSES)]
        hl = _HEMILINEAGES[t % len(_HEMILINEAGES)] if klass in ("intrinsic neuron", "ascending neuron",
                                                                 "motor neuron") else "TBD"
        nt = _NT_BY_HL.get(hl, ("acetylcholine", "gaba", "glutamate")[t % 3])
        vnc_born = hl != "TBD"
        for k in range(per_type):
            rois = {"LegNp(T1)(L)": int(rng.integers(20, 400)), "LTct": int(rng.integers(0, 200))}
            if t % 4 == 0:
                rois["ANm"] = int(rng.integers(10, 300))
            if klass in ("descending neuron", "ascending neuron"):
                rois["CV"] = int(rng.integers(10, 100))
            neurons[idx] = ("Traced", f"T{t:03d}", klass, int(rng.integers(20, 400)), int(rng.integers(20, 5000)),
                            "T1" if vnc_born else None, ("LHS", "RHS")[k % 2] if vnc_born else None,
                            None if vnc_born else "RHS", hl, "secondary" if vnc_born else None, float(1000 + t),
                            nt, rois)
            idx += 1
    ids = sorted(neurons)
    for pre in ids:
        targets = rng.choice([i for i in ids if i != pre], size=6, replace=False)
        edges[pre] = [(int(post), int(rng.integers(1, 30))) for post in targets]
    return neurons, edges


def _body(idx):
    return 10000 + idx


def _write_per_roi(path, edges, neurons, *, extra_rows=()):
    lines = ["bodyId_pre,bodyId_post,roi,weight"]
    for pre, targets in sorted(edges.items()):
        for post, weight in targets:
            rois = sorted(neurons[pre][12])
            roi = rois[post % len(rois)]
            lines.append(f"{_body(pre)},{_body(post)},{roi},{weight}")
            lines.append(f"{_body(pre)},{_body(post)},NotPrimary,1")
    lines.extend(extra_rows)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def synthetic_paths(tmp_path):
    neurons, edges = _synthetic()
    meta, edgelist, per_roi = tmp_path / "meta.feather", tmp_path / "edges.csv", tmp_path / "per_roi.csv"
    write_meta(meta, neurons=neurons, roi_names=_ROIS)
    write_edgelist(edgelist, edges=edges)
    _write_per_roi(per_roi, edges, neurons)
    return {"meta": meta, "edgelist": edgelist, "edgelist_per_roi": per_roi}


def _config(tmp_path, **overrides):
    base = dict(cache_root=str(tmp_path / "cache"), min_class_samples=2, min_class_groups=2, min_total_count=1,
                min_region_samples=1, nb_bins=4)
    base.update(overrides)
    return mt.MvRealModelConfig(**base)


# --------------------------------------------------------------------------- derived neuropil edges


def test_derive_neuropil_edges_keeps_only_neuropils_and_strips_side(tmp_path, synthetic_paths):
    out = mt.derive_neuropil_edges(synthetic_paths["edgelist_per_roi"], source_sha256="a" * 64,
                                   cache_root=tmp_path / "cache")
    table = pd.read_parquet(out)
    assert set(table.columns) == {"pre", "post", "neuropil", "weight"}
    assert set(table["neuropil"]) <= {"vnc_legnp_t1", "vnc_ltct", "vnc_anm", "vnc_inttct"}
    meta = json.loads(out.with_suffix(".json").read_text())
    assert "CV" in meta["dropped_rois"] and "NotPrimary" in meta["dropped_rois"]
    assert meta["rows_kept"] < meta["rows_in"]
    # cache hit returns the same file without rewriting it
    mtime = out.stat().st_mtime_ns
    assert mt.derive_neuropil_edges(synthetic_paths["edgelist_per_roi"], source_sha256="a" * 64,
                                    cache_root=tmp_path / "cache") == out
    assert out.stat().st_mtime_ns == mtime


def test_derive_neuropil_edges_unknown_roi_fails_closed(tmp_path, synthetic_paths):
    bad = tmp_path / "bad.csv"
    bad.write_text("bodyId_pre,bodyId_post,roi,weight\n1,2,ITO_FOO,3\n", encoding="utf-8")
    with pytest.raises(MvAdapterError) as exc:
        mt.derive_neuropil_edges(bad, source_sha256="b" * 64, cache_root=tmp_path / "cache")
    assert exc.value.code is MvAdapterErrorCode.REGION_VOCABULARY_UNKNOWN


def test_derive_neuropil_edges_refuses_snapshot_dir(tmp_path, synthetic_paths):
    with pytest.raises(ValueError, match="snapshots"):
        mt.derive_neuropil_edges(synthetic_paths["edgelist_per_roi"], source_sha256="c" * 64,
                                 cache_root=tmp_path / "snapshots" / "x")


def test_verify_manifest_listed_file(tmp_path):
    snap = tmp_path / "snap"
    (snap / "manifest").mkdir(parents=True)
    data = snap / "source" / "x.csv"
    data.parent.mkdir(parents=True)
    data.write_text("a,b\n1,2\n", encoding="utf-8")
    from flybrain_mv_adapter import sha256_file

    entry = {"relative_path": "source/x.csv", "role": "edgelist_per_roi", "size_bytes": data.stat().st_size,
             "sha256": sha256_file(data)}
    manifest = snap / "manifest" / "manifest.json"
    manifest.write_text(json.dumps({"integrity": {"files": [entry]}}), encoding="utf-8")
    path, digest = mt.verify_manifest_listed_file(snap, "edgelist_per_roi")
    assert path == data.resolve() and digest == entry["sha256"]
    manifest.write_text(json.dumps({"integrity": {"files": [{**entry, "sha256": "0" * 64}]}}), encoding="utf-8")
    with pytest.raises(MvAdapterError) as exc:
        mt.verify_manifest_listed_file(snap, "edgelist_per_roi")
    assert exc.value.code is MvAdapterErrorCode.INTEGRITY_MISMATCH
    manifest.write_text(json.dumps({"integrity": {"files": []}}), encoding="utf-8")
    with pytest.raises(MvAdapterError):
        mt.verify_manifest_listed_file(snap, "edgelist_per_roi")


# --------------------------------------------------------------------------- leakage rules


@pytest.mark.parametrize("target,column", [
    ("cell_class", "soma_neuromere"), ("cell_class", "birthtime"), ("cell_class", "class"),
    ("cell_class", "out_comp_hl__06b"),
    ("hemilineage", "class"), ("hemilineage", "out_comp_pnt__ach"), ("hemilineage", "subclass"),
    ("neurotransmitter_dominance", "hemilineage"), ("neurotransmitter_dominance", "in2_comp_hl__06b"),
    ("neurotransmitter_dominance", "predicted_nt_prob"),
    ("connectivity_tier", "degree__out_weight_total"), ("connectivity_tier", "out_np__n_neuropils"),
    ("region_specialization_tier", "out_np__vnc_legnp_t1"), ("region_specialization_tier", "primary_neuropil"),
])
def test_allowed_columns_drops_label_sources(target, column):
    kept, dropped = mt.allowed_columns(target, [column, "recip__partner_frac"])
    assert column not in kept
    assert "recip__partner_frac" in kept
    assert column in dropped["registry"] + dropped["mv_extra"]


@pytest.mark.parametrize("target", mt.MV_REAL_TARGETS)
def test_identity_columns_never_features(target):
    kept, dropped = mt.allowed_columns(target, ["type", "cell_type", "group", "serial_group", "body_id",
                                                "instance", "systematic_type", "recip__in_weight_frac"])
    assert kept == ["recip__in_weight_frac"]


def test_structured_categoricals_pass_registered_exclusions():
    for target in mt.MV_REAL_TARGETS:
        kept, _ = mt.allowed_columns(target, mt.CATEGORICAL_FEATURES[target])
        assert kept == list(mt.CATEGORICAL_FEATURES[target])


def test_held_out_mask_covers_shared_groups():
    nodes = pd.DataFrame({"root_id": ["1", "2", "3", "4", "5"], "cell_type": ["A", "A", "B", None, "C"],
                          "split_group": [None, None, "g1", "g1", None], "hemilineage": [None] * 5})
    frame = pd.DataFrame({"root_id": ["1", "3", "5"], "cell_type": ["A", "B", "C"],
                          "split_group": [None, "g1", None], "hemilineage": [None] * 3})
    mask = mt.held_out_mask_ids(nodes, frame, {"1", "3"}, ("cell_type", "split_group", "hemilineage"))
    assert mask == ["1", "2", "3", "4"]  # 2 shares type A, 4 shares group g1; 5 is train


def test_binned_text_fits_bins_on_train_only():
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 100.0, np.nan], "c": ["a", "b", None, "a", "b", "a"]})
    texts = mt.binned_text(frame, ["x"], ["c"], np.array([0, 1, 2, 3]), bins=2)
    parsed = [fbb.parse_input_features(t) for t in texts]
    assert parsed[0] == {"c": "c_a", "x": "x_b0"}
    assert parsed[2]["c"] == "c_unknown"
    assert parsed[4]["x"] == parsed[3]["x"] == "x_b1"  # the 100.0 outlier did not move the train edges
    assert parsed[5]["x"] == "x_na"


# --------------------------------------------------------------------------- labels + end-to-end


def test_label_frames(tmp_path, synthetic_paths):
    config = _config(tmp_path)
    inputs = mt.open_inputs(config, paths=synthetic_paths)
    nodes = mt.load_node_table(inputs.paths["meta"])
    cls, info = mt.label_frame("cell_class", config, inputs, nodes)
    assert set(cls["label"]) == {"intrinsic_vnc", "sensory", "descending", "ascending", "motor"}
    assert info["n_samples"] == len(cls)
    hl, _ = mt.label_frame("hemilineage", config, inputs, nodes)
    assert set(hl["label"]) <= set(h.lower() for h in _HEMILINEAGES)
    assert hl["label"].nunique() >= 2
    nt, _ = mt.label_frame("neurotransmitter_dominance", config, inputs, nodes)
    assert set(nt["label"]) <= {"dominant_ach", "dominant_gaba", "dominant_glut"}
    assert "hemilineage" in nt.columns  # split key only; never a feature (see allowed_columns)
    assert nodes["root_id"].is_unique


def _quick_eval(tmp_path):
    return fme.EvalConfig(models=(fme.ModelSpec("nb", ({"alpha": 1.0},), None),
                                  fme.ModelSpec("logreg", ({"C": 1.0},), "temperature")),
                          cv_folds=2, n_bootstrap=20, report_root=str(tmp_path / "reports"), run_label="t",
                          n_threads=1, ablation=True, random_split_control=False, train_ratio=0.6, val_ratio=0.2)


@pytest.mark.parametrize("target", ["cell_class", "hemilineage", "neurotransmitter_dominance"])
def test_build_target_dataset_masks_and_excludes(tmp_path, synthetic_paths, target):
    config = _config(tmp_path)
    build = mt.build_target_dataset(target, config, _quick_eval(tmp_path), explicit_paths=synthetic_paths,
                                    log=lambda *_: None)
    data = build.data
    cols = list(data.features.columns)
    fwf.assert_features_allowed(cols, target)
    assert not {"cell_type", "type", "group", "split_group", "serial_group"} & set(cols)
    if target != "neurotransmitter_dominance":
        assert "class" not in cols
    if target == "cell_class":
        assert not any(c.startswith("soma") or c == "birthtime" for c in cols)
    if target == "neurotransmitter_dominance":
        assert "hemilineage" not in cols and not any("_hl__" in c for c in cols)
        assert any("_pnt__" in c for c in cols)
    if target == "hemilineage":
        assert any("_hl__" in c for c in cols)
        assert "hemilineage" not in data.group_keys
    # every masked build records the split it was masked for, and the harness recomputes the same split
    assert data.notes["masked_split_ids_sha256"]
    wiring = [w for w in data.notes["wiring"] if w["masked"]]
    assert wiring and all(w["masked_nodes"] > 0 for w in wiring)
    # the masked partner category of held-out nodes never shows up: a held-out node's own class is hidden,
    # so the composition columns are computed from train partners only (checked via the sha guard)
    fme.check_leakage(data, uses_text=True)


def test_run_target_end_to_end_writes_report(tmp_path, synthetic_paths):
    config = _config(tmp_path)
    report = mt.run_target("cell_class", config, eval_config=_quick_eval(tmp_path), explicit_paths=synthetic_paths,
                           log=lambda *_: None)
    assert {row["model"] for row in report["summary"]} == {"nb", "logreg"}
    assert report["leakage_check"]["pass"] is True
    out = tmp_path / "reports" / "mv" / "cell_class" / "t"
    assert (out / "report.json").is_file() and (out / "report.md").is_file()
    assert report["mv_build"]["n_samples"] == sum(report["split"]["counts"].values())
    null = report["mv_shuffle_null"]
    assert set(null["models"]) == {"nb", "logreg"}
    assert abs(sum(null["test_prior"].values()) - 1.0) < 1e-9
    for entry in null["models"].values():
        assert len(entry["accuracies"]) == null["n_permutations"]
        assert entry["null_bar"] >= entry["majority_accuracy"]
    saved = json.loads((out / "report.json").read_text())
    assert "mv_shuffle_null" in saved and "mv_build" in saved
    for row in report["summary"]:
        assert row["gate"] in ("pass", "fail")
        assert row["shuffle_acc"] is not None


def test_masked_split_mismatch_fails_closed(tmp_path, synthetic_paths):
    config = _config(tmp_path)
    build = mt.build_target_dataset("cell_class", config, _quick_eval(tmp_path), explicit_paths=synthetic_paths,
                                    log=lambda *_: None)
    other = fme.EvalConfig(**{**_quick_eval(tmp_path).__dict__, "split_seed": "different-seed"})
    with pytest.raises(ValueError, match="masked for a different split"):
        fme.run_evaluation(build.data, other)


def test_unknown_target_fails_closed(tmp_path, synthetic_paths):
    with pytest.raises(ValueError, match="target must be one of"):
        mt.build_target_dataset("subclass", _config(tmp_path), _quick_eval(tmp_path), explicit_paths=synthetic_paths)


def test_open_inputs_explicit_paths_require_all_roles(tmp_path, synthetic_paths):
    paths = dict(synthetic_paths)
    paths.pop("edgelist_per_roi")
    with pytest.raises(MvAdapterError):
        mt.open_inputs(_config(tmp_path), paths=paths)


def test_module_has_no_network_imports():
    source = Path(mt.__file__).read_text(encoding="utf-8")
    for token in ("requests", "urllib", "httpx", "socket", "neuprint"):
        assert f"import {token}" not in source
