"""R3 size/degree side channel: excluded degree columns still reach the size baseline, never a model."""

import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_eval_stats as es  # noqa: E402
import flybrain_model_eval as fme  # noqa: E402
import flybrain_wiring_features as fwf  # noqa: E402

EDGES = [(1, 2, 10.0), (1, 3, 2.0), (2, 4, 8.0), (2, 5, 1.0), (3, 6, 9.0), (2, 1, 6.0)]
NODES = pd.DataFrame({"root_id": [1, 2, 3, 4, 5, 6], "super_class": ["A", "A", "B", "C", "B", "C"]})


def _source(tmp_path):
    path = tmp_path / "edges.feather"
    feather.write_feather(pa.table({"pre": [r[0] for r in EDGES], "post": [r[1] for r in EDGES],
                                    "weight": [r[2] for r in EDGES]}), str(path))
    return fwf.EdgeSource(path=str(path), pre="pre", post="post", weight="weight", unique_pairs=True)


def test_size_frame_keeps_degree_columns_the_objective_excludes(tmp_path):
    result = fwf.build_wiring_features(dataset="toy", objective="connectivity_tier", edges=_source(tmp_path),
                                       nodes=NODES, id_column="root_id", category_column="super_class",
                                       params=fwf.WiringFeatureParams(), cache_root=None)
    assert not [c for c in result.frame.columns if c.startswith("degree__")]  # excluded as features
    assert "degree__out_weight_total" in result.size_frame.columns  # but kept for the size bar
    assert "degree__out_weight_total" in result.excluded_features
    row = result.size_frame.set_index(fwf.NODE_ID_COLUMN).loc["1"]
    assert row["degree__out_weight_total"] == pytest.approx(12.0)


def test_size_frame_follows_the_rank_profile(tmp_path):
    result = fwf.build_wiring_features(dataset="toy", objective="connectivity_tier", edges=_source(tmp_path),
                                       nodes=NODES, id_column="root_id", category_column="super_class",
                                       params=fwf.robust_wiring_params(), cache_root=None)
    assert "degree__out_weight_total_rank" in result.size_frame.columns
    assert "degree__out_weight_total" not in result.size_frame.columns


def test_size_side_aux_aligns_skips_features_and_normalizes_side():
    size = pd.DataFrame({fwf.NODE_ID_COLUMN: ["3", "1", "2"], "degree__out_weight_total_rank": [0.2, 0.9, 0.5],
                         "degree__in_weight_total_rank": [0.1, 0.3, 0.0], "degree__recip_x": [1.0, 2.0, 3.0]})
    features = pd.DataFrame({"degree__recip_x": [7.0, 8.0, 9.0], "out_comp__a": [0.1, 0.2, 0.3]})
    aux = fme.size_side_aux(features, size_frame=size, id_column=fwf.NODE_ID_COLUMN, ids=[1, 2, 3],
                            side=["L", "right", None])
    assert "degree__recip_x" not in aux.columns  # already a feature: aux may not duplicate it
    assert aux["degree__out_weight_total_rank"].tolist() == [0.9, 0.5, 0.2]
    assert aux[fme.TOTAL_DEGREE_AUX].tolist() == pytest.approx([1.2, 0.5, 0.3])
    assert aux[fme.HEMISPHERE_AUX].tolist() == ["left", "right", "unknown"]
    name, values = es.degree_values(aux)
    assert name == fme.TOTAL_DEGREE_AUX and values.tolist() == pytest.approx([1.2, 0.5, 0.3])


def test_size_side_aux_returns_none_when_nothing_to_add():
    features = pd.DataFrame({"a": [1.0, 2.0]})
    assert fme.size_side_aux(features) is None


def test_aux_size_columns_raise_the_bar_but_never_reach_models():
    rng = np.random.default_rng(0)
    n = 400
    size = rng.gamma(2.0, 50.0, n)
    labels = np.where(size > np.median(size), "big", "small")  # a pure size proxy label
    noise = pd.DataFrame({"out_comp__a": rng.random(n), "out_comp__b": rng.random(n)})
    aux = pd.DataFrame({"degree__out_weight_total": size, fme.HEMISPHERE_AUX: ["left", "right"] * (n // 2)})
    data = fme.EvalDataset(dataset="toy", target="super_class", sample_ids=tuple(f"s{i}" for i in range(n)),
                           labels=labels, features=noise, group_keys=("g",),
                           group_values=tuple({"g": f"g{i // 4}"} for i in range(n)), aux=aux)
    config = fme.EvalConfig(models=(fme.ModelSpec("logreg", ({"C": 1.0},), None),), cv_folds=2, n_bootstrap=50,
                            n_permutations=0, require_permutation_null=False, split_curve=True, ablation=False,
                            random_split_control=False, report_root=None, save_models=False, n_threads=1,
                            calibration_bootstrap=20)
    report = fme.run_evaluation(data, config)
    size_base = report["baselines"]["size_degree"]
    assert size_base["available"] and "degree__out_weight_total" in size_base["columns_from_aux"]
    assert size_base["accuracy"]["test"] > 0.9
    gate = report["models"]["logreg"]["gate"]
    assert gate["beats_size_baseline"] is False
    levels = {row["level"] for row in report["split_curve"]["levels"]}
    assert any(level.startswith("hemisphere(") for level in levels)
