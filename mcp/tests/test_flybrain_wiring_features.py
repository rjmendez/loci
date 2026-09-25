"""Offline tests for flybrain_wiring_features on a hand-checkable toy graph."""

import math
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_wiring_features as fwf  # noqa: E402

# Node 6 is not in the node table (an unannotated partner); node 5 has no category.
EDGES = [
    (1, 2, 3, "X"),
    (1, 3, 1, "Y"),
    (2, 1, 2, "X"),
    (3, 4, 4, "Y"),
    (1, 6, 2, "X"),
    (6, 3, 5, "Z"),
]
NODES = pd.DataFrame({"root_id": [1, 2, 3, 4, 5], "super_class": ["A", "A", "B", "B", None]})


def _write_edges(path, rows, *, string_ids=False, fmt="feather", with_np=True):
    data = {
        "pre": [str(r[0]) if string_ids else r[0] for r in rows],
        "post": [str(r[1]) if string_ids else r[1] for r in rows],
        "weight": [r[2] for r in rows],
    }
    if with_np:
        data["neuropil"] = [r[3] for r in rows]
    table = pa.table(data)
    if fmt == "feather":
        feather.write_feather(table, str(path))
    else:
        pq.write_table(table, str(path))
    return str(path)


def _build(tmp_path, objective="super_class", edges=None, **kwargs):
    path = edges or _write_edges(tmp_path / "edges.feather", EDGES)
    source = kwargs.pop("source", None) or fwf.EdgeSource(path=path, pre="pre", post="post", weight="weight",
                                                          neuropil="neuropil")
    return fwf.build_wiring_features(dataset="toy", objective=objective, edges=source, nodes=NODES,
                                     id_column="root_id", category_column="super_class",
                                     cache_root=tmp_path / "cache", **kwargs)


def _row(result, node):
    frame = result.frame.set_index(fwf.NODE_ID_COLUMN)
    return frame.loc[str(node)]


def test_hand_checked_features(tmp_path):
    result = _build(tmp_path, objective="flow", params=fwf.WiringFeatureParams(two_hop=True, top_k_neuropils=2))
    n1 = _row(result, 1)
    assert n1["out_comp__a"] == pytest.approx(0.5)
    assert n1["out_comp__b"] == pytest.approx(1 / 6)
    assert n1["out_comp__unannotated"] == pytest.approx(1 / 3)
    assert n1["degree__out_weight_total"] == pytest.approx(6)
    assert n1["degree__out_n_annotated_partners"] == 2
    assert n1["degree__out_max_pair_weight"] == 3
    assert n1["recip__out_weight_frac"] == pytest.approx(0.75)
    assert n1["recip__partner_frac"] == pytest.approx(0.5)
    assert n1["out_np__x"] == pytest.approx(5 / 6)
    assert n1["out_comp__entropy"] == pytest.approx(-(0.5 * math.log2(0.5) + (1 / 6) * math.log2(1 / 6) + (1 / 3) * math.log2(1 / 3)))
    n3 = _row(result, 3)
    assert n3["in_comp__unannotated"] == pytest.approx(5 / 6)
    assert n3["in_comp__a"] == pytest.approx(1 / 6)
    assert n3["degree__in_weight_total"] == pytest.approx(6)
    n2 = _row(result, 2)
    assert n2["recip__out_weight_frac"] == pytest.approx(1.0)
    # 2 -> 1; node 1 sends A (node 2 itself) 3/6, B 1/6, unannotated 2/6. The 2 -> 1 -> 2
    # return path is removed, so node 2's own category never enters its own features.
    assert n2["out2_comp__a"] == pytest.approx(0.0)
    assert n2["out2_comp__b"] == pytest.approx(1 / 6)
    assert n2["out2_comp__unannotated"] == pytest.approx(1 / 3)
    # in-direction: 3 <- 1 (w1); 1's inputs are A (node 2) 2/2 -> in2 of 3 has A = 1.0 (no return path 3->1)
    assert n3["in2_comp__a"] == pytest.approx(1.0)
    # 1 <- 2 (w2); 2's inputs are only from node 1 (A, itself) -> removed
    assert n1["in2_comp__a"] == pytest.approx(0.0)
    n5 = _row(result, 5)
    assert np.isnan(n5["out_comp__a"]) and n5["degree__out_weight_total"] == 0
    # neuropils: X=10, Y=5, Z=5 (tie Y/Z broken by name); top-2 = x, y, rest -> other
    assert result.meta["stats"]["top_neuropils"] == ["x", "y"]
    assert _row(result, 3)["in_np__other"] == pytest.approx(5 / 6)
    assert "unknown" in result.meta["stats"]["categories"]


def test_string_ids_parquet_and_split_rows_match_pairs(tmp_path):
    base = _build(tmp_path, objective="flow")
    split_rows = [(1, 2, 1, "X"), (1, 2, 2, "Y")] + EDGES[1:]
    path = _write_edges(tmp_path / "split.parquet", split_rows, string_ids=True, fmt="parquet")
    nodes = NODES.assign(root_id=NODES["root_id"].astype(str))
    result = fwf.build_wiring_features(
        dataset="toy", objective="flow", nodes=nodes, id_column="root_id", category_column="super_class",
        edges=fwf.EdgeSource(path=path, pre="pre", post="post", weight="weight", neuropil="neuropil", unique_pairs=False),
        cache_root=None)
    a = base.frame.set_index("node_id")
    b = result.frame.set_index("node_id")
    for col in ("degree__out_n_annotated_partners", "degree__out_max_pair_weight", "recip__out_weight_frac",
                "out_comp__a", "degree__out_weight_total"):
        assert np.allclose(a[col].to_numpy(dtype=float), b[col].to_numpy(dtype=float), equal_nan=True), col
    assert b.loc["1", "out_np__y"] == pytest.approx(3 / 6)


def test_duplicate_pairs_rejected_when_declared_unique(tmp_path):
    path = _write_edges(tmp_path / "dup.feather", EDGES + [(1, 2, 1, "Y")])
    with pytest.raises(ValueError, match="duplicate"):
        _build(tmp_path, edges=path)


def test_separate_neuropil_table_and_no_weight_column(tmp_path):
    pairs = _write_edges(tmp_path / "pairs.feather", EDGES, with_np=False)
    np_path = _write_edges(tmp_path / "np.feather", EDGES)
    result = fwf.build_wiring_features(
        dataset="toy", objective="flow", nodes=NODES, id_column="root_id", category_column="super_class",
        edges=fwf.EdgeSource(path=pairs, pre="pre", post="post"),
        neuropil_edges=fwf.EdgeSource(path=np_path, pre="pre", post="post", weight="weight", neuropil="neuropil"),
        cache_root=None)
    n1 = _row(result, 1)
    assert n1["degree__out_weight_total"] == 3  # unweighted rows
    assert n1["out_np__x"] == pytest.approx(5 / 6)  # weights from the neuropil table


def test_min_edge_weight_filter(tmp_path):
    result = _build(tmp_path, objective="flow", params=fwf.WiringFeatureParams(min_edge_weight=2))
    assert _row(result, 1)["degree__out_weight_total"] == pytest.approx(5)


def test_cache_hit_and_fingerprint_changes(tmp_path, monkeypatch):
    edges = _write_edges(tmp_path / "edges.feather", EDGES)
    first = _build(tmp_path, objective="flow", edges=edges)
    assert first.cache_path and os.path.isfile(first.cache_path)
    monkeypatch.setattr(fwf, "compute_wiring_features", lambda **_: (_ for _ in ()).throw(AssertionError("recomputed")))
    second = _build(tmp_path, objective="flow", edges=edges)
    assert second.fingerprint == first.fingerprint
    pd.testing.assert_frame_equal(first.frame, second.frame)
    with pytest.raises(AssertionError, match="recomputed"):
        _build(tmp_path, objective="flow", edges=edges, params=fwf.WiringFeatureParams(top_k_neuropils=1))
    os.utime(edges, ns=(1, 1))  # edge file changed on disk -> new fingerprint
    with pytest.raises(AssertionError, match="recomputed"):
        _build(tmp_path, objective="flow", edges=edges)
    loaded = fwf.load_wiring_features(first.cache_path, objective="connectivity_tier")
    assert not any(c.startswith("degree__") for c in loaded.frame.columns)


def test_cache_tamper_detected(tmp_path):
    first = _build(tmp_path, objective="flow")
    with open(first.cache_path, "ab") as handle:
        handle.write(b"junk")
    with pytest.raises(ValueError, match="hash mismatch"):
        fwf.load_wiring_features(first.cache_path, objective="flow")


def test_refuses_cache_under_snapshots(tmp_path):
    edges = fwf.EdgeSource(path=_write_edges(tmp_path / "e.feather", EDGES), pre="pre", post="post")
    with pytest.raises(ValueError, match="snapshots"):
        fwf.build_wiring_features(dataset="toy", objective="flow", nodes=NODES, id_column="root_id",
                                  category_column="super_class", edges=edges,
                                  cache_root=tmp_path / "snapshots" / "cache")
    assert not (tmp_path / "snapshots").exists()


# ----------------------------------------------------------------- exclusions


def test_objective_exclusions_are_enforced_by_builder(tmp_path):
    conn = _build(tmp_path, objective="connectivity_tier")
    assert conn.excluded_features and all(c.startswith("degree__") or "n_neuropils" in c for c in conn.excluded_features)
    assert not any(c.startswith("degree__") for c in conn.frame.columns)
    region = _build(tmp_path, objective="region_specialization_tier")
    assert not any("_np__" in c for c in region.frame.columns)
    assert any(c.startswith("degree__") for c in region.frame.columns)
    with pytest.raises(ValueError, match="no label-exclusion rules"):
        _build(tmp_path, objective="made_up_target")


def test_assert_features_allowed():
    with pytest.raises(fwf.LabelLeakageError):
        fwf.assert_features_allowed(["out_comp__a", "root_id"], "super_class")
    with pytest.raises(fwf.LabelLeakageError):
        fwf.assert_features_allowed(["cell_class"], "super_class")
    with pytest.raises(fwf.LabelLeakageError):
        fwf.assert_features_allowed(["gaba_avg"], "neurotransmitter_dominance")
    with pytest.raises(fwf.LabelLeakageError):
        fwf.assert_features_allowed(["out_partner_tier"], "connectivity_tier")
    fwf.assert_features_allowed(["out_comp__a", "in_np__al_l", "hemilineage"], "super_class")


def test_register_custom_objective():
    try:
        fwf.register_objective_exclusions("toy_target", ["secret*"], reason="toy label is copied from secret")
        assert fwf.excluded_feature_names(["secret_sauce", "fine"], "toy_target") == ["secret_sauce"]
        with pytest.raises(ValueError, match="already registered"):
            fwf.register_objective_exclusions("toy_target", [], reason="again")
        with pytest.raises(ValueError, match="reason"):
            fwf.register_objective_exclusions("other_target", [], reason=" ")
    finally:
        fwf._EXCLUSIONS.pop("toy_target", None)


def test_harmonize_super_class():
    assert fwf.harmonize_super_class("cb_intrinsic") == "intrinsic_central"
    assert fwf.harmonize_super_class("central_brain_intrinsic") == "intrinsic_central"
    assert fwf.harmonize_super_class("central") == "intrinsic_central"
    assert fwf.harmonize_super_class("vnc_sensory_tbc") == "sensory"
    assert fwf.harmonize_super_class(None) == "unknown"
    assert fwf.harmonize_super_class("vnc_tbc") == "unknown"
    assert fwf.harmonize_super_class("Brand New") == "brand_new"


def test_vocab_map_applies(tmp_path):
    result = _build(tmp_path, objective="flow", vocab_map={"A": "alpha"})
    assert "out_comp__alpha" in result.frame.columns and "out_comp__a" not in result.frame.columns


def test_mask_category_ids_hides_heldout_labels(tmp_path):
    edges = _write_edges(tmp_path / "edges.feather", EDGES)
    plain = _build(tmp_path, objective="flow", edges=edges)
    masked = _build(tmp_path, objective="flow", edges=edges, mask_category_ids=[3])
    assert masked.fingerprint != plain.fingerprint
    assert _row(masked, 1)["out_comp__b"] == pytest.approx(0.0)
    assert _row(masked, 1)["out_comp__unannotated"] == pytest.approx(0.5)
    assert masked.meta["masked_category_nodes"] == 1
    with pytest.raises(ValueError, match="not in the node table"):
        _build(tmp_path, objective="flow", edges=edges, mask_category_ids=[99])


def test_feature_families():
    fams = fwf.feature_families(["degree__x", "out_comp__a", "out_comp__b", "hemilineage"])
    assert fams == {"degree": ["degree__x"], "hemilineage": ["hemilineage"], "out_comp": ["out_comp__a", "out_comp__b"]}
