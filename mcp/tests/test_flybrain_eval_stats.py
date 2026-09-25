"""Tests for the rigor layer (flybrain_eval_stats + its wiring in flybrain_model_eval).

Every gate criterion has a test that fails if that criterion is removed or
short-circuited: the gate-assembly unit tests (each criterion alone flips the
verdict) plus integration tests in which exactly ONE criterion fails.
Synthetic data only; reports go to tmp_path.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_eval_stats as es  # noqa: E402
import flybrain_learners as fl  # noqa: E402
import flybrain_model_eval as fme  # noqa: E402


def _cfg(tmp_path, **kwargs):
    base = dict(models=(fme.ModelSpec("hgb", ({"max_iter": 100},), None),), cv_folds=3, n_bootstrap=200,
                report_root=str(tmp_path / "reports"), n_threads=2, calibration_bootstrap=30, split_curve=False,
                ablation=False, random_split_control=False, n_permutations=0, require_permutation_null=False)
    base.update(kwargs)
    return fme.EvalConfig(**base)


def _sign_dataset(n_groups=90, per_group=10, seed=0, target="super_class", size_mode="noise", notes=None):
    """Sign-pattern label of (x1, x2) per type; a tree model beats every one-feature rule.

    ``size_mode="noise"``: aux size is label-independent. ``"none"``: no size column anywhere.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        cx, cy = rng.normal(size=2) * 1.5
        for j in range(per_group):
            x1, x2 = cx + 0.3 * rng.normal(), cy + 0.3 * rng.normal()
            label = "a" if (x1 > 0 and x2 > 0) else "b" if (x1 < 0 and x2 < 0) else "c"
            rows.append({"sample_id": f"s{g}_{j}", "label": label, "cell_type": f"type{g}", "wire__x1": x1,
                         "wire__x2": x2, "size__total_synapses": float(rng.lognormal(5, 1))})
    frame = pd.DataFrame(rows)
    aux = ["size__total_synapses"] if size_mode == "noise" else []
    return fme.EvalDataset.from_frame(frame, dataset="toy", target=target, id_column="sample_id", label_column="label",
                                      feature_columns=["wire__x1", "wire__x2"], group_columns=["cell_type"],
                                      aux_columns=aux, notes=notes)


# =========================================================================== gate assembly (mutation-proof)


def test_assemble_gate_requires_every_criterion():
    all_true = {name: True for name in es.GATE_CRITERIA}
    assert es.assemble_gate(all_true) == {"pass": True, "failed": []}
    assert set(es.GATE_CRITERIA) == {"beats_trivial_accuracy", "beats_trivial_macro_f1", "paired_gain_significant",
                                     "beats_size_baseline", "null_control_ok"}
    for name in es.GATE_CRITERIA:
        for bad in (False, None):
            verdict = es.assemble_gate({**all_true, name: bad})
            assert verdict == {"pass": False, "failed": [name]}, name
        missing = {k: v for k, v in all_true.items() if k != name}
        assert es.assemble_gate(missing)["failed"] == [name]
    # truthy non-bools are not accepted (a stray float cannot pass a criterion)
    assert es.assemble_gate({**all_true, "null_control_ok": 0.01})["pass"] is False


def test_null_gate_uses_permutations_and_fails_closed(tmp_path):
    cfg = _cfg(tmp_path, require_permutation_null=True)
    ok = fme._null_gate({"n_permutations": 100, "p_value_accuracy": 0.0099}, False, cfg)
    assert ok["null_control_ok"] is True and ok["null_control"] == "group_permutation"
    bad = fme._null_gate({"n_permutations": 100, "p_value_accuracy": 0.2}, True, cfg)
    assert bad["null_control_ok"] is False  # a collapsed single shuffle does not rescue it
    few = fme._null_gate({"n_permutations": 99, "p_value_accuracy": 0.01}, True, cfg)
    assert few["null_control_ok"] is None and "fails closed" in few["null_control_note"]
    assert fme._null_gate(None, True, cfg)["null_control_ok"] is None
    legacy = fme._null_gate(None, False, _cfg(tmp_path, require_permutation_null=False))
    assert legacy["null_control_ok"] is False


def test_size_gate_logic(tmp_path):
    boot = {"paired": {"model-minus-size_baseline": {"accuracy_diff_ci95": [0.01, 0.2]}}}
    size = {"available": True, "accuracy": {"test": 0.7}}
    cfg = _cfg(tmp_path)
    assert fme._size_gate(0.8, size, boot, cfg)["beats_size_baseline"] is True
    assert fme._size_gate(0.705, size, boot, cfg)["beats_size_baseline"] is False  # inside the margin
    flat = {"paired": {"model-minus-size_baseline": {"accuracy_diff_ci95": [-0.01, 0.2]}}}
    assert fme._size_gate(0.8, size, flat, cfg)["beats_size_baseline"] is False  # CI crosses 0
    missing = fme._size_gate(0.9, {"available": False, "reason": "x"}, boot, cfg)
    assert missing["beats_size_baseline"] is None and "fails closed" in missing["size_baseline_note"]
    optional = fme._size_gate(0.9, {"available": False, "reason": "x"}, boot, _cfg(tmp_path, require_size_baseline=False))
    assert optional["beats_size_baseline"] is True


# =========================================================================== integration: one failing criterion at a time


def _size_proxy_dataset(n_groups=300, per_group=5, seed=2):
    """Label = size band (a non-monotone function of size). The model sees a noisy copy of log size, so it
    beats every one-feature trivial rule, but the size/degree-only HGB is as good: R3 must fail it."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        centre = rng.normal(5.0, 1.0)
        for j in range(per_group):
            log_size = centre + 0.05 * rng.normal()
            label = "mid" if 4.4 < log_size < 5.6 else "edge"
            rows.append({"sample_id": f"s{g}_{j}", "label": label, "cell_type": f"t{g}",
                         "wire__s": log_size + 0.02 * rng.normal(), "wire__noise": rng.normal(),
                         "size__total_synapses": float(np.exp(log_size))})
    frame = pd.DataFrame(rows)
    return fme.EvalDataset.from_frame(frame, dataset="toy", target="super_class", id_column="sample_id",
                                      label_column="label", feature_columns=["wire__s", "wire__noise"],
                                      group_columns=["cell_type"], aux_columns=["size__total_synapses"])


def test_size_baseline_gate_blocks_a_size_proxy_model(tmp_path):
    data = _size_proxy_dataset()
    report = fme.run_evaluation(data, _cfg(tmp_path))
    gate = report["models"]["hgb"]["gate"]
    assert report["models"]["hgb"]["test"]["accuracy"] > gate["best_trivial_accuracy"] + 0.05
    assert gate["trivial_baseline_gate"]["pass"] and gate["paired_gain_significant"]
    assert gate["beats_size_baseline"] is False
    assert gate["failed_criteria"] == ["beats_size_baseline"]
    assert report["summary"][0]["gate"] == "fail"
    # the same run without the size gate would pass: the size gate is what blocks it
    loose = fme.run_evaluation(data, _cfg(tmp_path, size_baseline=False, require_size_baseline=False, report_root=None))
    assert loose["models"]["hgb"]["gate"]["pass"] is True


def test_missing_size_columns_fail_closed(tmp_path):
    data = _sign_dataset(size_mode="none")
    report = fme.run_evaluation(data, _cfg(tmp_path))
    gate = report["models"]["hgb"]["gate"]
    assert report["baselines"]["size_degree"]["available"] is False
    assert gate["failed_criteria"] == ["beats_size_baseline"]
    ok = fme.run_evaluation(data, _cfg(tmp_path, require_size_baseline=False, report_root=None))
    assert ok["models"]["hgb"]["gate"]["pass"] is True


def test_too_few_permutations_fail_closed(tmp_path):
    data = _sign_dataset()
    report = fme.run_evaluation(data, _cfg(tmp_path, n_permutations=5, require_permutation_null=True))
    gate = report["models"]["hgb"]["gate"]
    assert report["models"]["hgb"]["permutation_null"]["n_permutations"] == 5
    assert gate["failed_criteria"] == ["null_control_ok"]


def test_permutation_null_rejects_a_model_without_signal(tmp_path):
    """Group-constant labels unrelated to the (group-clustered) features: the observed accuracy sits inside the
    group-permutation null, so p is large and the null criterion fails."""
    rng = np.random.default_rng(5)
    rows = []
    for g in range(60):
        label = ["x", "y"][int(rng.integers(0, 2))]
        centre = rng.normal(size=2)
        for j in range(8):
            rows.append({"sample_id": f"s{g}_{j}", "label": label, "cell_type": f"t{g}",
                         "wire__a": centre[0] + 0.1 * rng.normal(), "wire__b": centre[1] + 0.1 * rng.normal()})
    data = fme.EvalDataset.from_frame(pd.DataFrame(rows), dataset="toy", target="super_class", id_column="sample_id",
                                      label_column="label", feature_columns=["wire__a", "wire__b"],
                                      group_columns=["cell_type"])
    cfg = _cfg(tmp_path, models=(fme.ModelSpec("logreg", ({"C": 1.0},), None),), n_permutations=100,
               require_permutation_null=True, require_size_baseline=False, size_baseline=False, report_root=None)
    report = fme.run_evaluation(data, cfg)
    perm = report["models"]["logreg"]["permutation_null"]
    assert perm["p_value_accuracy"] > 0.05
    assert "null_control_ok" in report["models"]["logreg"]["gate"]["failed_criteria"]


# =========================================================================== R4 statistics


def test_group_permuted_labels_keep_marginals_and_group_homogeneity():
    rng = np.random.default_rng(0)
    comps = np.repeat([f"g{i}" for i in range(40)], 5)
    labels = np.repeat(rng.choice(["a", "b", "c"], size=40), 5)
    out = es.group_permuted_labels(labels, comps, np.random.default_rng(1))
    assert sorted(out.tolist()) == sorted(labels.tolist())  # exact marginals
    homogeneous = [len(set(out[comps == c].tolist())) == 1 for c in set(comps.tolist())]
    assert all(homogeneous)  # equal-size groups: every group receives one donor block
    assert not np.array_equal(out, labels)
    # a per-row shuffle would break homogeneity; this is what makes the null group-level
    unequal = np.asarray(["g0"] * 3 + ["g1"] * 7 + ["g2"] * 5)
    out2 = es.group_permuted_labels(np.asarray(["a"] * 3 + ["b"] * 7 + ["c"] * 5), unequal, np.random.default_rng(3))
    assert sorted(out2.tolist()) == sorted(["a"] * 3 + ["b"] * 7 + ["c"] * 5)


def test_permutation_null_p_value_formula():
    y_train = np.asarray(["a", "b"] * 20)
    comps = np.asarray([f"g{i // 2}" for i in range(40)])
    y_test = ["a", "b", "a", "b"]
    result = es.permutation_null(lambda labels: ["a", "b", "a", "b"], y_train, comps, y_test,
                                 observed_accuracy=1.0, n_permutations=100, seed=0)
    assert result["n_permutations"] == 100 and result["p_value_accuracy"] == 1.0  # the null always ties
    result = es.permutation_null(lambda labels: ["b", "a", "b", "a"], y_train, comps, y_test,
                                 observed_accuracy=1.0, n_permutations=100, seed=0, n_jobs=2)
    assert result["p_value_accuracy"] == pytest.approx(1 / 101)


def test_cluster_bootstrap_resamples_whole_groups():
    comps = ["a"] * 5 + ["b"] * 3 + ["c"]
    for rows in es.cluster_bootstrap_indices(comps, n_bootstrap=50, seed=0):
        drawn = [comps[i] for i in rows]
        for c in set(drawn):
            assert drawn.count(c) % {"a": 5, "b": 3, "c": 1}[c] == 0
    assert es.effective_n(comps) == 3


def test_split_curve_plan_levels():
    ids = [f"s{i}" for i in range(200)]
    meta = [{"cell_type": f"t{i // 10}", "hemilineage": f"h{i // 50}", "side": "left" if i % 2 else "right"}
            for i in range(200)]
    levels = es.default_split_levels(["cell_type", "hemilineage"], ["cell_type", "hemilineage", "side"])
    assert [lv.name for lv in levels] == ["random", "cell_type", "cell_type+hemilineage",
                                          "hemisphere(side:left->right)"]
    plans = {lv.name: es.split_curve_plan(ids, meta, lv, split_seed="x", train_ratio=0.7, val_ratio=0.15)
             for lv in levels}
    type_plan = plans["cell_type"]
    train_types = {meta[i]["cell_type"] for i in type_plan["train"]}
    assert not train_types & {meta[i]["cell_type"] for i in type_plan["test"]}
    hl_plan = plans["cell_type+hemilineage"]
    assert not {meta[i]["hemilineage"] for i in hl_plan["train"]} & {meta[i]["hemilineage"] for i in hl_plan["test"]}
    side = plans["hemisphere(side:left->right)"]
    assert {meta[i]["side"] for i in side["train"]} == {"left"} and {meta[i]["side"] for i in side["test"]} == {"right"}


# =========================================================================== R6 calibration


def _calibrated_sample(n=4000, k=3, seed=0, temperature=1.0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, k)) * 2.0
    p_true = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.asarray([rng.choice(k, p=row) for row in p_true])
    shown = np.exp(logits / temperature)
    shown = shown / shown.sum(axis=1, keepdims=True)
    classes = [f"c{i}" for i in range(k)]
    return shown, [classes[i] for i in y], classes


def test_ece_variants_on_calibrated_and_overconfident_scores():
    proba, y, classes = _calibrated_sample()
    assert es.ece_sweep(proba, y, classes)["ece"] < 0.04
    assert es.ece_equal_mass(proba, y, classes) < 0.04
    assert es.debiased_l2_ece(proba, y, classes) < 0.03
    assert es.classwise_ece(proba, y, classes) < 0.03
    over, y2, _ = _calibrated_sample(temperature=0.3)
    assert es.ece_sweep(over, y2, classes)["ece"] > 0.1
    assert es.debiased_l2_ece(over, y2, classes) > 0.1
    assert es.classwise_ece(over, y2, classes) > es.classwise_ece(proba, y, classes)


def test_debiased_ece_removes_small_sample_bias():
    """With few rows, plain binned ECE of a PERFECTLY calibrated model is well above 0; the debiased estimate is not."""
    plain, debiased = [], []
    for seed in range(30):
        proba, y, classes = _calibrated_sample(n=150, seed=seed)
        plain.append(es.ece_equal_width(proba, y, classes))
        debiased.append(es.debiased_l2_ece(proba, y, classes))
    assert np.mean(plain) > 0.06
    assert np.mean(debiased) < np.mean(plain) / 2


def test_ece_sweep_picks_largest_monotone_binning():
    conf = np.linspace(0.34, 0.99, 300)
    proba = np.column_stack([conf, (1 - conf) / 2, (1 - conf) / 2])
    y = ["c0"] * 300  # always right: accuracy 1 in every bin is monotone, so the sweep reaches n bins
    out = es.ece_sweep(proba, y, ["c0", "c1", "c2"])
    assert out["n_bins"] == 300 and out["ece"] == pytest.approx(1 - conf.mean())


def test_calibration_metrics_ci_contain_point():
    proba, y, classes = _calibrated_sample(n=600)
    comps = [f"g{i // 6}" for i in range(600)]
    out = es.calibration_metrics_with_ci(proba, y, classes, comps, n_bootstrap=100, seed=0)
    assert out["effective_n"] == 100
    for key in es.CALIBRATION_METRIC_KEYS:
        lo, hi = out[f"{key}_ci95"]
        assert lo <= hi
    assert out["log_loss_ci95"][0] <= out["log_loss"] <= out["log_loss_ci95"][1]


def test_dirichlet_and_top_label_fix_overconfidence():
    over, y, classes = _calibrated_sample(n=3000, seed=1, temperature=0.3)
    test_over, y_test, _ = _calibrated_sample(n=3000, seed=2, temperature=0.3)
    groups = [f"g{i // 10}" for i in range(3000)]
    dirichlet = es.fit_dirichlet_cv(over, y, classes, groups)
    fixed = es.apply_ext_calibrator(dirichlet, test_over)
    assert np.allclose(fixed.sum(axis=1), 1.0)
    assert fl.multiclass_log_loss(fixed, y_test, classes) < fl.multiclass_log_loss(test_over, y_test, classes) - 0.1
    assert es.ece_sweep(fixed, y_test, classes)["ece"] < es.ece_sweep(test_over, y_test, classes)["ece"] / 2
    top = es.fit_top_label(over, y, classes, points_per_bin=100)
    fixed_top = es.apply_ext_calibrator(top, test_over)
    assert np.allclose(fixed_top.sum(axis=1), 1.0)
    assert es.ece_sweep(fixed_top, y_test, classes)["ece"] < es.ece_sweep(test_over, y_test, classes)["ece"] / 2


def test_calibration_fold_is_grouped():
    comps = np.asarray([f"g{i // 7}" for i in range(700)], dtype=object)
    fit_rows, cal_rows = es.calibration_fold(np.arange(700), comps, fraction=0.2, seed="s")
    assert set(fit_rows.tolist()).isdisjoint(cal_rows.tolist())
    assert len(fit_rows) + len(cal_rows) == 700
    assert set(comps[fit_rows].tolist()).isdisjoint(comps[cal_rows].tolist())
    assert 0.15 < len(cal_rows) / 700 < 0.25
    with pytest.raises(ValueError):
        es.calibration_fold(np.arange(5), np.asarray(["g"] * 5), fraction=0.2, seed="s")


@pytest.mark.parametrize("method", ["dirichlet", "top_label", "temperature"])
def test_harness_calibration_methods_and_artifact_roundtrip(tmp_path, method):
    data = _sign_dataset()
    cfg = _cfg(tmp_path, models=(fme.ModelSpec("hgb", ({"max_iter": 100},), method),), run_label=method)
    report = fme.run_evaluation(data, cfg)
    entry = report["models"]["hgb"]
    assert entry["calibration_protocol"]["protocol"] == "grouped_fold"
    assert entry["fit_info"]["calibration_source"] in ("grouped_calibration_fold", "prefit_heldout")
    loaded = fl.load_learner(entry["artifact"]["path"], trusted_root=tmp_path / "reports",
                             expected_manifest_sha256=entry["artifact"]["manifest_sha256"])
    X = data.features.iloc[:20]
    assert np.allclose(loaded.predict_proba(X).sum(axis=1), 1.0)
    # oof protocol still works for the extended methods
    legacy = fme.run_evaluation(data, _cfg(tmp_path, models=cfg.models, calibration_protocol="oof", report_root=None))
    assert legacy["models"]["hgb"]["calibration_protocol"]["protocol"] == "oof"


def test_unknown_calibration_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="calibration"):
        fme.run_evaluation(_sign_dataset(), _cfg(tmp_path, models=(fme.ModelSpec("hgb", ({},), "bogus"),)))
    with pytest.raises(ValueError, match="calibration_protocol"):
        fme.run_evaluation(_sign_dataset(), _cfg(tmp_path, calibration_protocol="bogus"))


# =========================================================================== R3 size / degree deciles


def test_degree_decile_table_uses_train_edges():
    train = np.arange(100, dtype=float)
    test = np.asarray([-5.0, 5.0, 55.0, 95.0, 500.0, np.nan])
    y = ["a", "a", "b", "b", "b", "a"]
    rows = es.degree_decile_table(train, test, y, {"m": ["a", "b", "b", "b", "a", "a"]}, list("uvwxyz"))
    by = {r["decile"]: r for r in rows}
    assert by[0]["n"] == 2 and by[0]["acc_m"] == 0.5  # -5 and 5 fall in the first train decile
    assert by[9]["n"] == 2 and by[9]["acc_m"] == 0.5  # 95 and 500 in the last
    assert by["missing"]["n"] == 1
    assert sum(r["n"] for r in rows) == 6


def test_size_columns_and_degree_axis():
    cols = ["degree__out_weight_total", "degree__in_weight_total", "out_comp__x", "morph__cable_length_um", "size__v"]
    assert es.size_columns(cols) == ["degree__in_weight_total", "degree__out_weight_total", "morph__cable_length_um",
                                     "size__v"]
    frame = pd.DataFrame({"degree__out_weight_total": [1.0, 2.0], "degree__in_weight_total": [3.0, np.nan]})
    name, values = es.degree_values(frame)
    assert name.startswith("degree__out_weight_total+") and values.tolist() == [4.0, 2.0]


# =========================================================================== R3 NT hooks


def _nt_dataset(seed=3):
    """NT set by hemilineage (Lacin 2019 style); wiring carries a noisy hemilineage signature."""
    rng = np.random.default_rng(seed)
    nts = ["acetylcholine", "gaba", "glutamate", "serotonin"]
    rows = []
    for h in range(18):
        nt = nts[h % 4] if h % 6 else "acetylcholine"
        sig = rng.normal(size=2) * 2
        for t in range(5):
            for j in range(8):
                label = nt if rng.random() > 0.1 else nts[int(rng.integers(0, 3))]
                rows.append({"sample_id": f"h{h}t{t}n{j}", "label": label, "cell_type": f"h{h}_t{t}",
                             "wire__a": sig[0] + rng.normal(), "wire__b": sig[1] + rng.normal(),
                             "size__total_synapses": float(rng.lognormal(5, 1)), "hemilineage": f"hl{h}"})
    frame = pd.DataFrame(rows)
    return fme.EvalDataset.from_frame(frame, dataset="toy", target="nt_ground_truth", id_column="sample_id",
                                      label_column="label", feature_columns=["wire__a", "wire__b"],
                                      group_columns=["cell_type"], aux_columns=["size__total_synapses", "hemilineage"],
                                      notes={"label_provenance": "measured"})


def test_nt_hooks_binary_oracle_two_stage(tmp_path):
    data = _nt_dataset()
    report = fme.run_evaluation(data, _cfg(tmp_path, models=(fme.ModelSpec("logreg", ({"C": 1.0},), None),)))
    nt = report["nt_hooks"]
    binary = nt["binary"]
    assert binary["labels"] == ["ach", "gaba_glu"]
    test_rows = report["split"]["label_counts"]["test"]
    assert binary["dropped_rows"]["test"] == test_rows.get("serotonin", 0)
    assert binary["n_test"] == report["split"]["counts"]["test"] - test_rows.get("serotonin", 0)
    oracle = nt["hemilineage_oracle"]
    # types are nested in hemilineages and grouped by type only, so test hemilineages were seen in train
    assert oracle["train_lookup"]["coverage"] == 1.0 and oracle["train_lookup"]["accuracy"] > 0.8
    assert oracle["leave_one_out_upper_bound"]["accuracy"] > 0.8
    two = nt["two_stage"]
    assert two["stage1_target"] == "hemilineage" and 0.0 <= two["accuracy"] <= 1.0
    assert "direct-minus-two_stage" in two["bootstrap"]["paired"]
    md = (tmp_path / "reports" / "toy" / "nt_ground_truth" / "report.md").read_text()
    assert "## NT hooks" in md and "binary ACh vs GABA+Glu" in md


def test_nt_hooks_off_for_other_targets_and_forceable(tmp_path):
    data = _sign_dataset()
    assert "nt_hooks" not in fme.run_evaluation(data, _cfg(tmp_path, report_root=None))
    forced = fme.run_evaluation(data, _cfg(tmp_path, nt_hooks=True, report_root=None))
    assert "skipped" in forced["nt_hooks"]["binary"]  # labels a/b/c are not transmitters


def test_normalize_nt_and_oracle_unit():
    assert es.normalize_nt("ACh") == "acetylcholine" and es.normalize_nt("GABA") == "gaba"
    assert es.normalize_nt("dominant_glutamate") == "glutamate" and es.normalize_nt("dopamine") is None
    assert es.binary_nt_label("glutamate") == "gaba_glu" and es.binary_nt_label("acetylcholine") == "ach"
    out = es.hemilineage_nt_oracle(["h1", "h1", "h2"], ["gaba", "gaba", "ach"], ["h1", "h3", "h3", "h3"],
                                   ["gaba", "ach", "ach", "gaba"])
    assert out["train_lookup"]["predictions"] == ["gaba", "gaba", "gaba", "gaba"]  # unseen h3 -> train majority
    assert out["train_lookup"]["coverage"] == 0.25
    # leave-one-out: h1 has a single test neuron (no others) -> fallback; h3 rows see the other two
    assert out["leave_one_out"]["coverage"] == 0.75
    assert out["leave_one_out"]["predictions"][1:] == ["ach", "ach", "ach"]


# =========================================================================== R7 hierarchy + R1 provenance


def _hier_dataset(seed=4):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(3):
        for c in range(3):
            for t in range(6):
                centre = np.asarray([p * 3.0, c * 1.5]) + rng.normal(size=2) * 0.3
                for j in range(6):
                    rows.append({"sample_id": f"p{p}c{c}t{t}n{j}", "label": f"class{p}{c}", "cell_type": f"p{p}c{c}t{t}",
                                 "wire__p": centre[0] + 0.3 * rng.normal(), "wire__c": centre[1] + 0.3 * rng.normal(),
                                 "size__total_synapses": float(rng.lognormal(5, 1)), "super_class": f"super{p}"})
    return fme.EvalDataset.from_frame(pd.DataFrame(rows), dataset="toy", target="cell_class", id_column="sample_id",
                                      label_column="label", feature_columns=["wire__p", "wire__c"],
                                      group_columns=["cell_type"], aux_columns=["size__total_synapses", "super_class"])


def test_hierarchy_prototype_reports_flat_vs_conditional(tmp_path):
    data = _hier_dataset()
    report = fme.run_evaluation(data, _cfg(tmp_path, models=(fme.ModelSpec("logreg", ({"C": 1.0},), "temperature"),),
                                           hierarchy_parent="super_class"))
    hier = report["hierarchy"]
    assert hier["n_parents"] == 3
    for name in ("flat", "hierarchical"):
        block = hier[name]
        assert 0 <= block["accuracy"] <= 1 and block["parent_accuracy"] >= block["accuracy"]
        assert "ece_sweep_ci95" in block["calibration"]
    assert hier["hierarchical"]["accuracy"] > 0.5
    assert "hierarchical-minus-flat" in hier["bootstrap"]["paired"]


def test_hierarchical_learner_probabilities_and_no_save(tmp_path):
    data = _hier_dataset()
    parent_of = {f"class{p}{c}": f"super{p}" for p in range(3) for c in range(3)}
    learner = fl.make_learner("hierarchical", base_backend="logreg", base_params={"C": 1.0}, parent_of=parent_of)
    learner.fit(data.features, data.labels)
    proba = learner.predict_proba(data.features)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.mean(np.asarray(learner.predict(data.features)) == data.labels) > 0.8
    with pytest.raises(NotImplementedError):
        fl.save_learner(learner, tmp_path / "h")
    with pytest.raises(ValueError, match="no parent"):
        fl.make_learner("hierarchical", base_backend="logreg", parent_of={}).fit(data.features, data.labels)


@pytest.mark.parametrize("prov,applies,word", [
    ("measured", True, "measured"),
    ("curated_morphology", True, "curated morphology"),
    ("connectivity_defined", False, "connectivity-derived"),
    ("model_predicted", False, "distillation"),
    (None, False, "not declared"),
    ("made_up", False, "not declared"),
])
def test_provenance_claims(prov, applies, word):
    out = es.provenance_claim({} if prov is None else {"label_provenance": prov, "label_classifier": "synister"})
    assert out["gate_applies"] is applies and word in out["claim"]


def test_provenance_flows_into_gate_and_summary(tmp_path):
    data = _sign_dataset(notes={"label_provenance": "model_predicted", "label_classifier": "Eckstein 2024 CNN"})
    report = fme.run_evaluation(data, _cfg(tmp_path))
    row = report["summary"][0]
    assert row["gate"] == "pass"  # the statistics pass ...
    assert row["gate_applies"] is False and "distillation of Eckstein 2024 CNN" in row["claim"]  # ... but it is not gated
    md = (tmp_path / "reports" / "toy" / "super_class" / "report.md").read_text()
    assert "Label provenance: **model_predicted**" in md


def test_aux_is_never_a_model_input(tmp_path):
    data = _sign_dataset()
    report = fme.run_evaluation(data, _cfg(tmp_path, report_root=None))
    schema = report["models"]["hgb"]["describe"]["schema"]
    assert "size__total_synapses" not in str(schema)
    assert report["aux_columns"] == ["size__total_synapses"]
    with pytest.raises(ValueError, match="aux columns duplicate"):
        fme.EvalDataset("d", "super_class", ("a", "b"), np.asarray(["x", "y"]), pd.DataFrame({"f": [1, 2]}), (),
                        ({}, {}), aux=pd.DataFrame({"f": [1, 2]}))


def test_normalize_nt_accepts_prefixed_dataset_spellings():
    assert es.normalize_nt("nt_ach") == es.normalize_nt("acetylcholine")
    assert es.normalize_nt("nt_glut") == es.normalize_nt("glutamate")
    assert es.normalize_nt("nt_gaba") == es.normalize_nt("GABA")
    assert es.normalize_nt("nt_his") is None and es.normalize_nt("nt_oct") is None
    assert es.binary_nt_label("nt_glut") == es.binary_nt_label("gaba")
