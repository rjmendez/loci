"""R5 robust wiring features: strong-edge 2-hop, rank normalization and the partner-NT hard block.

Each test is written so that it FAILS if the corresponding safeguard is
removed: the 2-hop threshold (hand-checked values that differ from the
unthresholded ones, and invariance to added weak edges), the rank
normalization (invariance to a global synapse-detection scale factor, no raw
counts left) and the NT block (builder, loader and name-level checks).
"""

import dataclasses
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_wiring_features as fwf  # noqa: E402

# All six nodes are annotated. Strong (>= 5) pairs: 1->2 (10), 2->4 (8), 3->6 (9), 2->1 (6).
EDGES = [
    (1, 2, 10.0),
    (1, 3, 2.0),
    (2, 4, 8.0),
    (2, 5, 1.0),
    (3, 6, 9.0),
    (2, 1, 6.0),
]
NODES = pd.DataFrame({"root_id": [1, 2, 3, 4, 5, 6], "super_class": ["A", "A", "B", "C", "B", "C"]})
LEGACY_PARAM_KEYS = {"top_k_neuropils", "reciprocity", "two_hop", "min_edge_weight", "category_tag",
                     "max_pair_rows", "batch_rows"}


def _write(path, rows):
    feather.write_feather(pa.table({"pre": [r[0] for r in rows], "post": [r[1] for r in rows],
                                    "weight": [float(r[2]) for r in rows]}), str(path))
    return str(path)


def _build(tmp_path, rows=EDGES, *, objective="super_class", params=None, nodes=NODES, category_column="super_class",
           name="edges.feather", unique_pairs=True, cache=False, **kwargs):
    source = fwf.EdgeSource(path=_write(tmp_path / name, rows), pre="pre", post="post", weight="weight",
                            unique_pairs=unique_pairs)
    return fwf.build_wiring_features(dataset="toy", objective=objective, edges=source, nodes=nodes,
                                     id_column="root_id", category_column=category_column,
                                     params=params or fwf.WiringFeatureParams(),
                                     cache_root=(tmp_path / "cache") if cache else None, **kwargs)


def _rows(result):
    return result.frame.set_index(fwf.NODE_ID_COLUMN)


# ------------------------------------------------------------------ strong-edge 2-hop


def test_two_hop_threshold_hand_checked(tmp_path):
    strong = _rows(_build(tmp_path, params=fwf.WiringFeatureParams(two_hop=True, two_hop_min_weight=5)))
    legacy = _rows(_build(tmp_path, params=fwf.WiringFeatureParams(two_hop=True), name="e2.feather"))
    # node 1 -> 2 is its only strong out-edge; 2's strong outputs: C (node 4) 8, A (node 1) 6;
    # the 1 -> 2 -> 1 return path is removed from A.
    assert strong.loc["1", "out2_comp__c"] == pytest.approx(8 / 14)
    assert strong.loc["1", "out2_comp__a"] == pytest.approx(0.0)
    assert strong.loc["1", "out2_comp__b"] == pytest.approx(0.0)
    # Unthresholded (v1): the weak 1 -> 3 edge and 2's weak output to B both leak in.
    assert legacy.loc["1", "out2_comp__c"] == pytest.approx((10 / 12) * (8 / 15) + 2 / 12)
    assert legacy.loc["1", "out2_comp__b"] == pytest.approx((10 / 12) * (1 / 15))
    # in-direction: 6 <- 3 (strong); 3's only input (1 -> 3, w 2) is weak -> nothing reaches node 6.
    assert strong.loc["6", "in2_comp__a"] == pytest.approx(0.0)
    assert legacy.loc["6", "in2_comp__a"] == pytest.approx(1.0)
    # node 5 only has a weak input: no strong 2-hop neighbourhood -> NaN, not 0.
    assert np.isnan(strong.loc["5", "in2_comp__a"])
    # 1-hop composition is untouched by the 2-hop threshold.
    assert strong.loc["1", "out_comp__b"] == pytest.approx(2 / 12)


def test_two_hop_threshold_ignores_added_weak_edges(tmp_path):
    params = fwf.WiringFeatureParams(two_hop=True, two_hop_min_weight=5)
    base = _rows(_build(tmp_path, params=params))
    noise = [(4, 5, 4.0), (5, 3, 1.0), (6, 1, 3.0), (4, 1, 2.0), (3, 2, 4.9)]
    noisy = _rows(_build(tmp_path, EDGES + noise, params=params, name="noisy.feather"))
    two_hop = [c for c in base.columns if c.startswith(("out2_comp__", "in2_comp__"))]
    assert two_hop
    pd.testing.assert_frame_equal(base[two_hop], noisy[two_hop])
    # sensitivity (10): only 1 -> 2 survives, so node 1's second hop is empty (2 has no strong outputs)
    ten = _rows(_build(tmp_path, params=dataclasses.replace(params, two_hop_min_weight=10), name="t.feather"))
    assert ten.loc["1", "out2_comp__c"] == pytest.approx(0.0)
    assert np.isnan(ten.loc["2", "out2_comp__c"])


def test_two_hop_threshold_uses_pair_aggregated_weight(tmp_path):
    # per-neuropil rows (fw-style): 3 + 3 synapses on 1 -> 2 aggregate to a strong 6-synapse pair.
    rows = [(1, 2, 3.0), (1, 2, 3.0), (2, 4, 8.0)]
    res = _rows(_build(tmp_path, rows, unique_pairs=False,
                       params=fwf.WiringFeatureParams(two_hop=True, two_hop_min_weight=5)))
    assert res.loc["1", "out2_comp__c"] == pytest.approx(1.0)


def test_two_hop_stats_recorded(tmp_path):
    res = _build(tmp_path, params=fwf.robust_wiring_params())
    stats = res.meta["stats"]
    assert stats["two_hop_min_weight"] == 5.0 and stats["two_hop_pairs"] == 4
    assert stats["two_hop_weight_fraction"] == pytest.approx(33 / 36)


# ------------------------------------------------------------------ rank normalization


def test_rank_transform_is_scale_free_and_drops_raw_counts(tmp_path):
    params = fwf.WiringFeatureParams(two_hop=True, degree_transform="rank")
    base = _rows(_build(tmp_path, params=params))
    scaled = _rows(_build(tmp_path, [(a, b, w * 1.43) for a, b, w in EDGES], params=params, name="s.feather"))
    for raw in ("degree__out_weight_total", "degree__in_weight_total", "degree__out_n_annotated_partners",
                "degree__log1p_out_weight_total", "degree__out_max_pair_weight", "degree__out_mean_pair_weight"):
        assert raw not in base.columns
    ranked = [c for c in base.columns if c.endswith("_rank")]
    assert "degree__out_weight_total_rank" in ranked and "degree__in_max_pair_weight_rank" in ranked
    vals = base[ranked].to_numpy(dtype=float)
    assert np.nanmin(vals) >= 0 and np.nanmax(vals) <= 1
    # A global synapse-detection factor (FIB-SEM vs TEM) leaves every feature unchanged.
    pd.testing.assert_frame_equal(base, scaled, check_exact=False, rtol=1e-9)
    raw_base = _rows(_build(tmp_path, params=fwf.WiringFeatureParams(two_hop=True), name="r1.feather"))
    raw_scaled = _rows(_build(tmp_path, [(a, b, w * 1.43) for a, b, w in EDGES],
                              params=fwf.WiringFeatureParams(two_hop=True), name="r2.feather"))
    assert not np.allclose(raw_base["degree__out_weight_total"], raw_scaled["degree__out_weight_total"])


def test_percentile_rank():
    out = fwf.percentile_rank(np.array([3.0, np.nan, 1.0, 3.0]))
    assert np.isnan(out[1])
    assert out[2] == pytest.approx(1 / 3) and out[0] == pytest.approx(2.5 / 3) == out[3]
    zeros = fwf.percentile_rank(np.array([0.0, 0.0, 5.0, 2.0]))
    assert list(zeros[:2]) == [0.0, 0.0] and zeros[2] == pytest.approx(1.0) and zeros[3] == pytest.approx(0.75)


def test_rank_columns_still_excluded_for_connectivity_tier(tmp_path):
    res = _build(tmp_path, objective="connectivity_tier", params=fwf.robust_wiring_params())
    assert not any(c.startswith("degree__") for c in res.frame.columns)


# ------------------------------------------------------------------ params / fingerprint


def test_legacy_fingerprint_unchanged_and_r5_recorded(tmp_path):
    assert set(fwf.WiringFeatureParams().as_dict()) == LEGACY_PARAM_KEYS
    assert set(fwf.WiringFeatureParams(two_hop=True).as_dict()) == LEGACY_PARAM_KEYS
    robust = fwf.robust_wiring_params()
    assert robust.two_hop and robust.two_hop_min_weight == 5.0 and robust.degree_transform == "rank"
    assert robust.as_dict()["two_hop_min_weight"] == 5.0 and robust.as_dict()["degree_transform"] == "rank"
    fps = {_build(tmp_path, params=p, name=f"f{i}.feather").fingerprint for i, p in enumerate((
        fwf.WiringFeatureParams(two_hop=True), robust, fwf.robust_wiring_params(two_hop_min_weight=10),
        fwf.robust_wiring_params(degree_transform="raw")))}
    assert len(fps) == 4
    # identical edge file -> the four fingerprints above differ only through params
    e = fwf.EdgeSource(path=_write(tmp_path / "fp.feather", EDGES), pre="pre", post="post", weight="weight")
    kwargs = dict(dataset="toy", edges=e, neuropil_edges=None, nodes_digest="x", vocab_digest="y")
    assert fwf.wiring_fingerprint(params=fwf.WiringFeatureParams(), **kwargs) != fwf.wiring_fingerprint(
        params=fwf.WiringFeatureParams(degree_transform="rank"), **kwargs)


def test_param_validation():
    with pytest.raises(ValueError, match="degree_transform"):
        fwf.WiringFeatureParams(degree_transform="zscore")
    with pytest.raises(ValueError, match="requires two_hop"):
        fwf.WiringFeatureParams(two_hop_min_weight=5)
    with pytest.raises(ValueError, match=">= 0"):
        fwf.WiringFeatureParams(two_hop=True, two_hop_min_weight=-1)
    assert fwf.robust_wiring_params(fwf.WiringFeatureParams(category_tag="hl"), two_hop_min_weight=10).category_tag == "hl"


# ------------------------------------------------------------------ partner-NT hard block

NT_NODES = NODES.assign(pnt=["ach", "gaba", "glut", "ach", "unknown", "gaba"],
                        mystery=["acetylcholine", "gaba", "glutamate", "acetylcholine", None, "gaba"])


@pytest.mark.parametrize("objective", ["neurotransmitter_dominance", "nt_ground_truth", "nt_binary_ach"])
@pytest.mark.parametrize("column", ["pnt", "mystery"])
def test_builder_blocks_partner_nt_for_nt_objectives(tmp_path, objective, column):
    if objective not in fwf.registered_objectives():
        fwf.register_objective_exclusions(objective, ["nt*"], reason="toy NT target")
    try:
        with pytest.raises(fwf.LabelLeakageError, match="partner-NT"):
            _build(tmp_path, objective=objective, nodes=NT_NODES, category_column=column)
    finally:
        if objective == "nt_binary_ach":
            fwf._EXCLUSIONS.pop(objective, None)


def test_builder_blocks_nt_tag_and_allows_non_nt(tmp_path):
    with pytest.raises(fwf.LabelLeakageError):
        _build(tmp_path, objective="nt_ground_truth", params=fwf.WiringFeatureParams(category_tag="pnt"))
    ok = _build(tmp_path, objective="nt_ground_truth")  # super_class partners are fine for NT
    assert ok.meta["partner_category_nt"] is False
    # a non-NT objective may use partner NT; the build records it and the NT loader refuses the cache
    res = _build(tmp_path, objective="hemilineage", nodes=NT_NODES, category_column="mystery", cache=True)
    assert res.meta["partner_category_nt"] is True
    with pytest.raises(fwf.LabelLeakageError, match="NT-derived"):
        fwf.load_wiring_features(res.cache_path, objective="neurotransmitter_dominance")
    fwf.load_wiring_features(res.cache_path, objective="hemilineage")


def test_nt_block_name_patterns():
    for name in ("out_comp_pnt__ach", "in_comp__gaba", "out2_comp__glutamate", "in_comp__inhibitory",
                 "frac_inhibitory_inputs", "in2_comp_nt__x", "partner_excitatory_share", "in_comp__ach_l"):
        with pytest.raises(fwf.LabelLeakageError):
            fwf.assert_features_allowed([name], "nt_ground_truth")
        with pytest.raises(fwf.LabelLeakageError):
            fwf.assert_features_allowed([name], "neurotransmitter_dominance")
    fwf.assert_features_allowed(["out_comp__intrinsic_central", "recip__partner_frac", "out2_comp__descending",
                                 "degree__out_weight_total_rank", "in_comp__entropy"], "nt_ground_truth")
    fwf.assert_features_allowed(["in_comp__gaba"], "super_class")  # block is for NT models only


def test_is_nt_objective_and_detector():
    for name in ("neurotransmitter_dominance", "nt_ground_truth", "nt_binary", "fw_nt", "transmitter_class"):
        assert fwf.is_nt_objective(name), name
    for name in ("connectivity_tier", "count_tier", "hemilineage", "cell_family", "super_class", "l1em_io_class"):
        assert not fwf.is_nt_objective(name), name
    assert fwf.partner_category_is_nt("top_nt", ["x"])
    assert fwf.partner_category_is_nt("predictedNt", ["x"])
    assert fwf.partner_category_is_nt("renamed", ["gaba", "ach", "unknown", None])
    assert fwf.partner_category_is_nt("renamed", ["inhibitory", "excitatory"])
    assert not fwf.partner_category_is_nt("super_class", ["intrinsic_central", "sensory", "gaba_like_other"])
    assert not fwf.partner_category_is_nt("superclass_raw", ["cb_intrinsic", "descending_neuron"])
    try:
        fwf.register_nt_objective("toy_polarity")
        assert fwf.is_nt_objective("toy_polarity")
    finally:
        fwf.NT_OBJECTIVES.discard("toy_polarity")


# ------------------------------------------------------------------ profiles (env var / context)


@pytest.fixture(autouse=True)
def _no_ambient_profile(monkeypatch):
    monkeypatch.delenv(fwf.WIRING_PROFILE_ENV, raising=False)


def test_profile_env_and_context_upgrade_params(tmp_path, monkeypatch):
    base = fwf.WiringFeatureParams(two_hop=True, top_k_neuropils=3)
    v1 = _build(tmp_path, params=base)
    monkeypatch.setenv(fwf.WIRING_PROFILE_ENV, "r5")
    r5 = _build(tmp_path, params=base, name="p.feather")
    assert r5.meta["params"]["two_hop_min_weight"] == 5.0 and r5.meta["params"]["top_k_neuropils"] == 3
    assert "degree__out_weight_total_rank" in r5.frame.columns
    with fwf.wiring_profile("r5:10"):
        r10 = _build(tmp_path, params=base, name="q.feather")
    assert r10.meta["params"]["two_hop_min_weight"] == 10.0
    monkeypatch.setenv(fwf.WIRING_PROFILE_ENV, "v1")
    again = _build(tmp_path, params=base, name="v1again.feather")
    assert again.meta["params"] == v1.meta["params"] == base.as_dict()
    assert "degree__out_weight_total" in again.frame.columns
    assert len({v1.fingerprint, r5.fingerprint, r10.fingerprint}) == 3
    # no 2-hop requested -> the profile only rank-normalizes; explicit non-v1 params are kept as given
    one_hop = fwf.apply_wiring_profile(fwf.WiringFeatureParams(), fwf.parse_wiring_profile("r5"))
    assert one_hop.two_hop_min_weight == 0.0 and one_hop.degree_transform == "rank"
    explicit = fwf.robust_wiring_params(two_hop_min_weight=7)
    assert fwf.apply_wiring_profile(explicit, fwf.parse_wiring_profile("r5:10")) == explicit
    with pytest.raises(ValueError, match="unknown wiring profile"):
        fwf.parse_wiring_profile("r6")
    with pytest.raises(ValueError, match="> 0"):
        fwf.parse_wiring_profile("r5:0")
