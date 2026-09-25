"""Offline tests for flybrain_target_registry (R1 label provenance + the provenance gate)."""

import dataclasses
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_banc_targets as banc  # noqa: E402
import flybrain_fw_targets as fw  # noqa: E402
import flybrain_l1em_targets as l1em  # noqa: E402
import flybrain_mc_targets as mc  # noqa: E402
import flybrain_model_eval as fme  # noqa: E402
import flybrain_mv_targets as mv  # noqa: E402
import flybrain_ol_targets as ol  # noqa: E402
import flybrain_target_registry as ftr  # noqa: E402

P = ftr.LabelProvenance


def test_every_module_target_has_a_registered_provenance():
    declared = {
        "fw": {k: s.label_provenance for k, s in fw.FW_TARGETS.items()},
        "mc": dict(mc.TARGET_LABEL_PROVENANCE),
        "mv": dict(mv.TARGET_LABEL_PROVENANCE),
        "banc": dict(banc.TARGET_LABEL_PROVENANCE),
        "ol": dict(ol.TARGET_LABEL_PROVENANCE),
        "l1em": {k: s.label_provenance for k, s in l1em.TARGET_SPECS.items()},
    }
    assert set(mc.TARGET_LABEL_PROVENANCE) == set(mc.MC_REAL_TARGETS)
    assert set(mv.TARGET_LABEL_PROVENANCE) == set(mv.MV_REAL_TARGETS)
    assert set(banc.TARGET_LABEL_PROVENANCE) == set(banc.BANC_TARGETS)
    assert set(ol.TARGET_LABEL_PROVENANCE) == set(ol.OL_TARGETS)
    for dataset, table in declared.items():
        for target, value in table.items():
            assert ftr.target_provenance(dataset, target).provenance.value == value, (dataset, target)
    # and nothing registered that no module can evaluate
    assert {k for k in ftr.TARGET_PROVENANCE} == {(d, t) for d, table in declared.items() for t in table}


@pytest.mark.parametrize("dataset,target,expected", [
    ("fw", "super_class", P.CURATED_MORPHOLOGY), ("fw", "flow", P.CURATED_MORPHOLOGY),
    ("fw", "hemilineage", P.CURATED_MORPHOLOGY), ("fw", "neurotransmitter_dominance", P.MODEL_PREDICTED),
    ("fw", "nt_literature", P.MEASURED), ("fw", "connectivity_tier", P.CONNECTIVITY_DEFINED),
    ("mc", "cell_class", P.CONNECTIVITY_DEFINED), ("mc", "super_class", P.CURATED_MORPHOLOGY),
    ("mv", "hemilineage", P.CONNECTIVITY_DEFINED), ("mv", "neurotransmitter_dominance", P.MODEL_PREDICTED),
    ("banc", "neurotransmitter_dominance", P.MODEL_PREDICTED), ("banc", "flow", P.CURATED_MORPHOLOGY),
    ("ol", "cell_family", P.CONNECTIVITY_DEFINED), ("ol", "neurotransmitter_dominance", P.MODEL_PREDICTED),
    ("l1em", "l1em_io_class", P.CONNECTIVITY_DEFINED),
])
def test_r1_assignments(dataset, target, expected):
    prov = ftr.target_provenance(dataset, target)
    assert prov.provenance is expected
    assert prov.gateable == (expected in (P.MEASURED, P.CURATED_MORPHOLOGY))


def test_uncertain_and_frames():
    banc_flow = ftr.target_provenance("banc", "flow")
    assert banc_flow.provenance_uncertain and banc_flow.gateable
    nt = ftr.target_provenance("banc", "neurotransmitter_dominance")
    assert nt.reporting_frame.startswith("distillation of synister_banc")
    assert ftr.target_provenance("ol", "cell_family").reporting_frame == "recovery of connectivity-derived annotations"
    with pytest.raises(ValueError):
        ftr.TargetProvenance("x", "y", P.MODEL_PREDICTED, "src", ("A 2020",))  # classifier missing
    with pytest.raises(ftr.ProvenanceGateError):
        ftr.target_provenance("fw", "made_up_target")
    with pytest.raises(ftr.ProvenanceGateError):
        ftr.check_module_provenance("fw", {"super_class": "measured"})


def test_notes_match_the_harness_reader():
    import flybrain_eval_stats as es

    for (dataset, target), prov in ftr.TARGET_PROVENANCE.items():
        claim = es.provenance_claim(prov.notes())
        assert claim["label_provenance"] == prov.provenance.value
        assert claim["gate_applies"] == prov.gateable, (dataset, target)


def _report(dataset, target, harness_pass=True):
    return {"dataset": dataset, "target": target,
            "models": {"hgb": {"gate": {"pass": harness_pass}}, "logreg": {"gate": {"pass": False}}},
            "summary": [{"model": "hgb", "gate": "pass" if harness_pass else "fail"},
                        {"model": "logreg", "gate": "fail"}]}


def test_gate_refuses_non_gateable_targets():
    report = ftr.apply_provenance_gate(_report("fw", "connectivity_tier"))
    gate = report["models"]["hgb"]["gate"]
    assert gate["harness_pass"] is True and gate["pass"] is False and gate["verdict"] == ftr.VERDICT_NOT_GATED
    rows = {r["model"]: r for r in report["summary"]}
    assert rows["hgb"]["gate"] == "not_gated" and rows["hgb"]["label_provenance"] == "connectivity_defined"
    assert rows["hgb"]["reporting_frame"] == "recovery of connectivity-derived annotations"
    with pytest.raises(ftr.ProvenanceGateError, match="never promoted"):
        ftr.assert_promotable(report, "hgb")
    # idempotent: a second application keeps the harness verdict
    again = ftr.apply_provenance_gate(report)
    assert again["models"]["hgb"]["gate"]["harness_pass"] is True


def test_gate_passes_gateable_targets_and_promotion_needs_a_pass():
    report = ftr.apply_provenance_gate(_report("fw", "super_class"))
    assert report["models"]["hgb"]["gate"]["pass"] is True
    assert {r["model"]: r["gate"] for r in report["summary"]} == {"hgb": "pass", "logreg": "fail"}
    ftr.assert_promotable(report, "hgb")
    with pytest.raises(ftr.ProvenanceGateError, match="did not pass"):
        ftr.assert_promotable(report, "logreg")
    with pytest.raises(ftr.ProvenanceGateError, match="apply_provenance_gate"):
        ftr.assert_promotable(_report("fw", "super_class"), "hgb")
    distilled = ftr.apply_provenance_gate(_report("mv", "neurotransmitter_dominance"))
    assert distilled["summary"][0]["reporting_frame"].startswith("distillation of the MANC NT classifier")


def _toy_dataset(target, dataset="fw"):
    rng = np.random.default_rng(0)
    rows = []
    for g in range(40):
        centre = rng.normal(size=2) * 2
        for j in range(8):
            x = centre + 0.2 * rng.normal(size=2)
            rows.append({"sample_id": f"s{g}_{j}", "label": "a" if x[0] > 0 else "b", "cell_type": f"t{g}",
                         "wire__x": x[0], "wire__y": x[1]})
    frame = pd.DataFrame(rows)
    return fme.EvalDataset.from_frame(frame, dataset=dataset, target=target, id_column="sample_id",
                                      label_column="label", feature_columns=["wire__x", "wire__y"],
                                      group_columns=["cell_type"])


def _fast_config(tmp_path):
    wanted = dict(models=(fme.ModelSpec("logreg", ({"C": 1.0},), "sigmoid"),), cv_folds=2, n_bootstrap=50,
                  n_threads=2, report_root=str(tmp_path / "reports"), ablation=False, random_split_control=False,
                  n_permutations=0, require_permutation_null=False, split_curve=False, calibration_bootstrap=20,
                  size_baseline=False, require_size_baseline=False, nt_hooks=False)
    names = {f.name for f in dataclasses.fields(fme.EvalConfig)}
    return fme.EvalConfig(**{k: v for k, v in wanted.items() if k in names})


def test_run_gated_evaluation_end_to_end(tmp_path):
    config = _fast_config(tmp_path)
    report = ftr.run_gated_evaluation(_toy_dataset("connectivity_tier"), config)
    assert report["notes"]["label_provenance"] == "connectivity_defined"
    assert report["label_provenance"]["gate_applies"] is False
    row = report["summary"][0]
    assert row["gate"] == "not_gated" and row["label_provenance"] == "connectivity_defined"
    out = tmp_path / "reports" / "fw" / "connectivity_tier"
    assert "Label provenance: `connectivity_defined`" in (out / "report.md").read_text()
    saved = json.loads((out / "report.json").read_text())
    assert saved["models"]["logreg"]["gate"]["verdict"] == "not_gated"
    metrics = json.loads((out / "models" / "logreg" / "metrics.json").read_text())
    assert metrics["gate"]["pass"] is False and metrics["label_provenance"]["label_provenance"] == "connectivity_defined"
    # the re-stamped artifact still verifies against the manifest recorded in the report
    import flybrain_learners as fl

    fl.load_learner(out / "models" / "logreg", trusted_root=out / "models",
                    expected_manifest_sha256=report["models"]["logreg"]["artifact"]["manifest_sha256"])
    with pytest.raises(ftr.ProvenanceGateError):
        ftr.run_gated_evaluation(_toy_dataset("not_registered"), config)
