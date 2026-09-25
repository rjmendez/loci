"""Offline tests for the flybrain_model_eval harness (synthetic data only, tmp report root)."""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_brain_cluster_baselines as fbb  # noqa: E402
import flybrain_eval_stats as es  # noqa: E402
import flybrain_model_eval as fme  # noqa: E402
import flybrain_wiring_features as fwf  # noqa: E402


AUX = ("size__total_synapses", "hemilineage", "side")


def _interaction_dataset(n_groups=90, per_group=10, seed=0, target="super_class", extra=None, aux=AUX, notes=None):
    """Label = sign pattern of (x1, x2): no single-feature rule does well, a tree model does.

    Aux (never a model input): a label-independent size column, a hemilineage
    shared by every 9th type, and alternating left/right sides.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        cx, cy = rng.normal(size=2) * 1.5
        for j in range(per_group):
            x1 = cx + 0.3 * rng.normal()
            x2 = cy + 0.3 * rng.normal()
            label = "a" if (x1 > 0 and x2 > 0) else "b" if (x1 < 0 and x2 < 0) else "c"
            rows.append({"sample_id": f"s{g}_{j}", "label": label, "cell_type": f"type{g}",
                         "wire__x1": x1, "wire__x2": x2, "noise__z": rng.normal(),
                         "annot_colour": ["red", "green"][int(rng.integers(0, 2))],
                         "size__total_synapses": float(rng.lognormal(5, 1)), "hemilineage": f"hl{g % 9}",
                         "side": "left" if j % 2 == 0 else "right"})
    frame = pd.DataFrame(rows)
    if extra:
        extra(frame)
    feature_cols = [c for c in frame.columns if c not in ("sample_id", "label", "cell_type", *AUX)]
    return fme.EvalDataset.from_frame(frame, dataset="toy", target=target, id_column="sample_id", label_column="label",
                                      feature_columns=feature_cols, group_columns=["cell_type"],
                                      aux_columns=list(aux), notes=notes)


def _config(tmp_path, **kwargs):
    base = dict(models=(fme.ModelSpec("hgb", ({"max_iter": 150},), "isotonic"),
                        fme.ModelSpec("logreg", ({"C": 0.1}, {"C": 1.0}), "temperature")),
                cv_folds=3, n_bootstrap=200, report_root=str(tmp_path / "reports"), n_threads=2,
                calibration_bootstrap=50)
    base.update(kwargs)
    return fme.EvalConfig(**base)


def _fast(tmp_path, **kwargs):
    """No permutation null (not required) and no optional extras: for tests about other parts of the protocol."""
    base = dict(n_permutations=0, require_permutation_null=False, split_curve=False, ablation=False,
                random_split_control=False)
    base.update(kwargs)
    return _config(tmp_path, **base)


def test_full_protocol_on_interaction_data(tmp_path):
    data = _interaction_dataset(notes={"label_provenance": "curated_morphology"})
    levels = (es.SplitLevel("random"), es.SplitLevel("type", ("cell_type",)),
              es.SplitLevel("type+hemilineage", ("cell_type", "hemilineage")),
              es.SplitLevel("hemisphere", (), "side", ("left",), ("right",)))
    report = fme.run_evaluation(data, _config(tmp_path, split_levels=levels))
    rows = {r["model"]: r for r in report["summary"]}
    hgb = rows["hgb"]
    assert hgb["gate"] == "pass", report["models"]["hgb"]["gate"]
    gate = report["models"]["hgb"]["gate"]
    assert gate["failed_criteria"] == [] and gate["gate_applies"] is True
    # R3 size/degree baseline: read from aux, label-independent here, so it is beaten
    assert report["baselines"]["size_degree"]["columns"] == ["size__total_synapses"]
    assert report["baselines"]["size_degree"]["columns_from_aux"] == ["size__total_synapses"]
    assert gate["beats_size_baseline"] is True and hgb["size_baseline"] < hgb["model_acc"] - 0.1
    deciles = report["models"]["hgb"]["test"]["degree_deciles"]
    assert deciles["degree_column"] == "size__total_synapses"
    assert sum(r["n"] for r in deciles["rows"]) == report["split"]["counts"]["test"]
    assert {"acc_model", "acc_size_baseline", "acc_best_trivial", "acc_majority"} <= set(deciles["rows"][0])
    # R4 group-permutation null (>= 100) gates; effective n = groups
    perm = report["models"]["hgb"]["permutation_null"]
    assert perm["n_permutations"] == 100 and perm["p_value_accuracy"] <= 0.02
    assert gate["null_control"] == "group_permutation" and gate["null_control_ok"] is True
    assert hgb["perm_p"] == perm["p_value_accuracy"]
    assert report["split"]["effective_n"]["test"] == report["split"]["n_test_components"]
    # R4 split curve: random is per-sample, hemisphere trains on left and tests on right
    curve = {r["level"]: r for r in report["split_curve"]["levels"]}
    assert curve["random"]["n_test_components"] == curve["random"]["n_test"]
    assert curve["type"]["n_test_components"] < curve["type"]["n_test"]
    assert curve["type+hemilineage"]["n_test_components"] <= 3  # 9 hemilineages merge the 90 types
    assert curve["hemisphere"]["n_train"] + curve["hemisphere"]["n_test"] == len(data.sample_ids)
    # R6 calibration: grouped calibration fold + debiased metrics with CIs
    cal = report["models"]["hgb"]["calibration_protocol"]
    assert cal["protocol"] == "grouped_fold" and 0 < cal["calibration_rows"] < report["split"]["counts"]["train"]
    metrics = report["models"]["hgb"]["test"]["calibration"]["calibrated"]
    for key in es.CALIBRATION_METRIC_KEYS:
        lo, hi = metrics[f"{key}_ci95"]
        assert lo <= hi
    assert hgb["ece_sweep"] == metrics["ece_sweep"]
    # R1 provenance columns
    assert hgb["label_provenance"] == "curated_morphology" and hgb["gate_applies"] is True
    assert "nt_hooks" not in report  # not an NT target
    assert hgb["model_acc"] > hgb["best_trivial"] + 0.1
    assert report["models"]["hgb"]["shuffle_control"]["collapsed_to_majority"]
    # logistic regression cannot represent the sign interaction: an honest negative
    assert rows["logreg"]["model_acc"] < hgb["model_acc"]
    for key in ("dataset", "target", "model", "majority", "best_trivial", "best_trivial_rule", "model_acc", "model_ci",
                "macro_f1", "ece", "shuffle_acc", "n_train", "n_test", "gate"):
        assert key in hgb
    # grouped split: no cell type in two splits
    assert report["split"]["group_keys"] == ["cell_type"]
    assert report["split"]["n_test_components"] < report["split"]["counts"]["test"]
    # calibration + CIs recorded
    test = report["models"]["hgb"]["test"]
    lo, hi = test["bootstrap"]["per_prediction"]["model"]["accuracy_ci95"]
    assert lo <= test["accuracy"] <= hi
    assert len(test["reliability_table"]) == 15
    assert "raw_uncalibrated" in test and "confusion" in test
    # ablation on val, per family
    ablation = report["ablation"]
    assert ablation["evaluated_on"] == "val"
    families = {r["family"] for r in ablation["drop_one_family"]}
    assert families == {"wire", "noise", "annot_colour"}
    wire = next(r for r in ablation["drop_one_family"] if r["family"] == "wire")
    assert wire["delta_accuracy"] < -0.1
    assert "random_split_control" in report["models"]["hgb"]
    # files
    out = tmp_path / "reports" / "toy" / "super_class"
    saved = json.loads((out / "report.json").read_text())
    assert saved["summary"] == json.loads(json.dumps(report["summary"]))
    assert "| hgb |" in (out / "report.md").read_text()
    artifact = report["models"]["hgb"]["artifact"]
    import flybrain_learners as fl

    loaded = fl.load_learner(artifact["path"], trusted_root=tmp_path / "reports",
                             expected_manifest_sha256=artifact["manifest_sha256"])
    assert loaded.classes_ == ("a", "b", "c")


def test_tautological_feature_fails_gate(tmp_path):
    def leak(frame):
        frame["score_hint"] = frame["label"].map({"a": 0.0, "b": 1.0, "c": 2.0})

    data = _interaction_dataset(extra=leak, target="flow")
    report = fme.run_evaluation(data, _fast(tmp_path, models=(fme.ModelSpec("hgb", ({"max_iter": 100},), None),),
                                              ablation=False, random_split_control=False))
    row = report["summary"][0]
    assert row["best_trivial"] >= 0.99 and row["best_trivial_rule"] == "features:lookup"
    assert row["gate"] == "fail"


def test_label_defining_feature_is_rejected(tmp_path):
    def leak(frame):
        frame["degree__out_weight_total"] = np.arange(len(frame), dtype=float)

    data = _interaction_dataset(extra=leak, target="connectivity_tier")
    with pytest.raises(fwf.LabelLeakageError):
        fme.run_evaluation(data, _config(tmp_path))
    assert not (tmp_path / "reports").exists()


def test_id_feature_and_label_copy_rejected(tmp_path):
    data = _interaction_dataset(extra=lambda f: f.__setitem__("root_id", np.arange(len(f))))
    with pytest.raises(fwf.LabelLeakageError):
        fme.run_evaluation(data, _config(tmp_path))
    data = _interaction_dataset(extra=lambda f: f.__setitem__("copied", f["label"]))
    with pytest.raises(fwf.LabelLeakageError, match="identical to the label"):
        fme.run_evaluation(data, _config(tmp_path))


def test_text_model_with_leaking_input_text_is_rejected(tmp_path):
    samples = []
    for i in range(60):
        label = "high" if i % 2 else "low"
        samples.append({"sample_id": f"s{i}", "expected_label": label, "metadata": {"cell_type": f"t{i // 3}"},
                        "input_text": f"cns_division brain out_partner_tier {label}_tier", "features": {"wire__x": float(i)}})
    data = fme.EvalDataset.from_samples(samples, dataset="toy", target="connectivity_tier", group_keys=["cell_type"])
    with pytest.raises(fwf.LabelLeakageError, match="out_partner_tier"):
        fme.run_evaluation(data, _config(tmp_path, models=(fme.ModelSpec("nb", ({},), None),)))


def test_nb_text_model_runs_with_text_baselines(tmp_path):
    rng = np.random.default_rng(1)
    samples = []
    for i in range(300):
        colour = ["red", "green", "blue"][i % 3]
        label = "x" if colour == "red" or rng.random() < 0.2 else "y"
        samples.append({"sample_id": f"s{i}", "expected_label": label, "metadata": {"cell_type": f"t{i // 5}"},
                        "input_text": f"colour {colour} shape {['sq', 'tri'][i % 2]}"})
    data = fme.EvalDataset.from_samples(samples, dataset="toy", target="super_class", group_keys=["cell_type"])
    report = fme.run_evaluation(data, _fast(tmp_path, models=(fme.ModelSpec("nb", ({"alpha": 1.0},), "sigmoid"),),
                                              ablation=True))
    row = report["summary"][0]
    assert row["best_trivial_rule"].startswith("text:")
    # lookup on colour is the whole signal, so NB cannot beat it: the gate must say so
    assert row["gate"] == "fail"
    assert report["ablation"] == {"skipped": "text model"}


def test_feature_view_baselines_match_module_on_text_serialization():
    rng = np.random.default_rng(3)
    n = 240
    frame = pd.DataFrame({"a_num": rng.integers(0, 20, size=n).astype(float), "b_num": rng.normal(size=n).round(2),
                          "colour": rng.choice(["red", "green", "blue"], size=n)})
    y = np.where(frame["a_num"] > 12, "dominant_hi", np.where(frame["colour"] == "red", "dominant_lo", "dominant_mid"))
    y = np.asarray(y, dtype=object)
    tr, te = np.arange(0, 160), np.arange(160, n)
    ours = fme.feature_view_baselines(frame.iloc[tr].reset_index(drop=True), y[tr],
                                      {"test": (frame.iloc[te].reset_index(drop=True), y[te])})
    text = fme.features_as_text(frame)
    module = fbb.evaluate_trivial_baselines(list(zip([text[i] for i in tr], y[tr])), list(zip([text[i] for i in te], y[te])))
    for rule in ("majority", "threshold", "lookup"):
        assert ours["splits"]["test"]["accuracy"][rule] == pytest.approx(module["rules"][rule]["heldout_accuracy"]), rule
    assert ours["rules"]["threshold"]["params"]["feature"] == module["rules"]["threshold"]["params"]["feature"]
    assert ours["rules"]["threshold"]["params"]["threshold"] == pytest.approx(module["rules"]["threshold"]["params"]["threshold"])
    assert ours["splits"]["test"]["best_accuracy"] == pytest.approx(module["best_accuracy"])


def test_argmax_rule_on_score_features():
    frame = pd.DataFrame({"ach_avg": [0.9, 0.1, 0.2, 0.8], "gaba_avg": [0.1, 0.9, 0.7, 0.3]})
    y = np.asarray(["dominant_ach", "dominant_gaba", "dominant_gaba", "dominant_ach"], dtype=object)
    result = fme.feature_view_baselines(frame, y, {"test": (frame, y)})
    assert result["splits"]["test"]["accuracy"]["argmax"] == 1.0


def test_grouped_bootstrap_and_metrics():
    y = ["a", "b", "a", "b", "a", "c"]
    comps = ["g1", "g1", "g2", "g3", "g4", "g5"]
    boot = fme.grouped_bootstrap(y, {"perfect": y, "const": ["a"] * 6}, comps, n_bootstrap=300, seed=0,
                                 pairs=(("perfect", "const"),))
    assert boot["per_prediction"]["perfect"]["accuracy_ci95"] == [1.0, 1.0]
    assert boot["paired"]["perfect-minus-const"]["accuracy_diff_ci95"][0] >= 0
    assert fme.macro_f1(["a", "b"], ["a", "a"]) == pytest.approx((2 / 3 + 0) / 2)
    cm = fme.confusion(["a", "b", "b"], ["a", "a", "b"])
    assert cm["matrix"] == [[1, 0], [1, 1]]


def test_report_dir_refuses_snapshots(tmp_path):
    with pytest.raises(ValueError, match="snapshots"):
        fme.report_dir(tmp_path / "snapshots", "toy", "x")


def test_eval_dataset_validation():
    with pytest.raises(ValueError, match="duplicate"):
        fme.EvalDataset("d", "super_class", ("a", "a"), np.asarray(["x", "y"]), pd.DataFrame({"f": [1, 2]}), (), ({}, {}))


def test_plan_grouped_split_matches_run_and_mismatch_fails(tmp_path):
    data = _interaction_dataset(n_groups=30)
    cfg = _fast(tmp_path, models=(fme.ModelSpec("logreg", ({},), None),), ablation=False, random_split_control=False,
                  shuffle_control=False, report_root=None)
    plan = fme.plan_grouped_split(data.sample_ids, data.group_values, data.group_keys, cfg)
    data.notes = {"masked_split_ids_sha256": fme.split_ids_sha256(plan)}
    report = fme.run_evaluation(data, cfg)
    assert report["split"]["ids_sha256"] == fme.split_ids_sha256(plan)
    assert report["split"]["counts"]["test"] == len(plan["test"])
    data.notes = {"masked_split_ids_sha256": "0" * 64}
    with pytest.raises(ValueError, match="masked for a different split"):
        fme.run_evaluation(data, cfg)
