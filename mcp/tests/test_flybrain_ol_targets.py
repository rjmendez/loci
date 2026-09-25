import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import flybrain_brain_cluster_ol_samples as ols  # noqa: E402
import flybrain_model_eval as fme  # noqa: E402
import flybrain_ol_targets as olt  # noqa: E402
import flybrain_wiring_features as fwf  # noqa: E402
from test_flybrain_ol_adapter import roi_info, write_meta  # noqa: E402

# Importing the samples module registers the ol builders globally; keep the slot empty for other modules.
ols.unregister_ol_sample_builders()

# family -> (types, class, NT, main ROI, layer)
_FAMILIES = {
    "Mi": (["Mi1", "Mi2", "Mi4", "Mi9"], "optic", "acetylcholine", "ME_R", "ME_R_layer_05"),
    "Tm": (["Tm1", "Tm2", "Tm3", "Tm9"], "optic", "glutamate", "LO_R", "LO_R_layer_2"),
    "LC": (["LC4", "LC6", "LC9", "LC11"], "visual_projection", "acetylcholine", "PVLP_R", "LO_R_layer_4"),
    "LoVC": (["LoVC1", "LoVC2", "LoVC3", "LoVC4"], "visual_centrifugal", "gaba", "PLP_R", "LO_R_layer_5"),
}
_PER_TYPE = 6


def _synthetic_rows(seed=0):
    rng = np.random.default_rng(seed)
    rows, classes = [], {}
    body = 20000
    for fam, (types, cls, nt, main, layer) in _FAMILIES.items():
        for t in types:
            for _ in range(_PER_TYPE):
                body += 1
                pre = int(rng.integers(50, 3000))
                post = int(rng.integers(50, 3000))
                info = json.loads(roi_info(**{main: (pre - 10, post - 10), "ME_R": (10, 10)}))
                info[layer] = {"synweight": pre}
                info["ME_R_col_1_2"] = {"pre": 1, "synweight": 1}
                rows.append({
                    "bodyId:long": body, "type:string": t, "instance:string": f"{t}_R", "status:string": "Traced",
                    "pre:int": pre, "post:int": post, "downstream:int": pre * 4, "upstream:int": post,
                    "predictedNt:string": nt, "predictedNtConfidence:float": 0.9, "totalNtPredictions:float": float(pre),
                    "consensusNt:string": nt, "ntReference:string": "Davis et al 2020",
                    "hemilineage:string": None, "somaLocation:point{srid:9157}": "{x:1}",
                    "assignedOlHex1:float": None, "roiInfo:string": json.dumps(info, sort_keys=True),
                })
                classes[body] = cls
    # an unclear type, an Anchor-status neuron of a real type, an untyped fragment
    extra = [(29001, "Tm_unclear", "Traced", "optic"), (29002, "Mi1", "Anchor", "optic"), (29003, None, "", None)]
    for body_id, t, status, cls in extra:
        rows.append({**rows[0], "bodyId:long": body_id, "type:string": t, "instance:string": f"{t}_R",
                     "status:string": status})
        if cls:
            classes[body_id] = cls
    return rows, classes


def _write_inputs(tmp_path, seed=0):
    rows, classes = _synthetic_rows(seed)
    frame = pd.DataFrame(rows)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = table.set_column(0, "bodyId:long", pa.array(frame["bodyId:long"].tolist(), pa.uint64()))
    neurons = tmp_path / "neurons.feather"
    feather.write_feather(table, str(neurons), chunksize=16)
    meta = tmp_path / "meta.csv"
    write_meta(meta)
    stats = tmp_path / "body_stats.feather"
    feather.write_feather(pa.table({"body": pa.array(list(classes), pa.uint64()),
                                    "class": list(classes.values())}), str(stats))
    # family-structured wiring: Mi -> Tm -> LC, LoVC -> Mi, plus noise
    rng = np.random.default_rng(seed + 1)
    by_fam = {}
    for r in rows:
        t = r["type:string"]
        if t and not t.endswith("unclear"):
            fam = next(f for f, spec in _FAMILIES.items() if t in spec[0])
            by_fam.setdefault(fam, []).append(r["bodyId:long"])
    pairs = {}
    for src, dst in (("Mi", "Tm"), ("Tm", "LC"), ("LoVC", "Mi"), ("Tm", "Tm")):
        for a in by_fam[src]:
            for b in rng.choice(by_fam[dst], size=4, replace=False):
                if a != b:
                    pairs[(a, int(b))] = int(rng.integers(1, 30))
    pairs[(29003, by_fam["Mi"][0])] = 5  # fragment input
    pairs[(29002, by_fam["Tm"][0])] = 7  # anchor-status Mi1 -> Tm
    pre, post = zip(*pairs)
    edges = tmp_path / "edges.feather"
    feather.write_feather(pa.table({"body_pre": pa.array(pre, pa.uint64()), "body_post": pa.array(post, pa.uint64()),
                                    "weight": pa.array(list(pairs.values()), pa.int64())}), str(edges), chunksize=64)
    return olt.load_ol_structured_inputs(neurons_path=neurons, meta_path=meta, edges_path=edges,
                                         body_stats_path=stats)


@pytest.fixture
def inputs(tmp_path):
    return _write_inputs(tmp_path)


def _eval_config(tmp_path, **kw):
    base = dict(cv_folds=2, n_bootstrap=50, report_root=str(tmp_path / "reports"), save_models=False, ablation=False,
                random_split_control=False, train_ratio=0.5, val_ratio=0.25)
    base.update(kw)
    return fme.EvalConfig(**base)


def _config(target, tmp_path, **kw):
    base = dict(target=target, min_types=1, max_per_type=4, cache_root=str(tmp_path / "cache"))
    base.update(kw)
    return olt.OlTargetConfig(**base)


# --------------------------------------------------------------------------- family map


@pytest.mark.parametrize("cell_type,cls,expected", [
    ("Tm5a", "optic", "Tm"), ("TmY9b", "optic", "TmY"), ("MeVPLo1", "visual_projection", "MeVP"),
    ("MeVPMe3", "optic", "MeVPMe"), ("LoVCLo2", "visual_centrifugal", "LoVC"), ("HSE", "visual_projection", "HS"),
    ("DNge017", "descending_neuron", "DN"), ("PLP001", "central", "central_brain"), ("T4a", "optic", "T"),
    ("Tm_unclear", "optic", None), ("aMe_TBD1", "optic", None), ("(PLP256)", "central", None), ("R1-R6", "sensory", "R"),
])
def test_raw_family(cell_type, cls, expected):
    assert olt.raw_family(cell_type, cls) == expected


def test_family_map_pools_rare_families_and_is_deterministic():
    types = {"Mi1": "optic", "Mi4": "optic", "Mi9": "optic", "C2": "optic", "C3": "optic", "Tm_unclear": "optic"}
    fm = olt.build_family_map(types, min_types=3)
    assert fm.type_to_family == {"Mi1": "Mi", "Mi4": "Mi", "Mi9": "Mi", "C2": "other", "C3": "other"}
    assert fm.family("Tm_unclear") is None
    assert olt.partner_category("Tm_unclear", fm) == olt.UNCLEAR_CATEGORY
    assert fm.sha256 == olt.build_family_map(dict(reversed(list(types.items()))), min_types=3).sha256
    assert fm.sha256 != olt.build_family_map(types, min_types=2).sha256


# --------------------------------------------------------------------------- exclusions


def test_cell_family_objective_registered_with_type_and_lineage_patterns():
    rule = fwf.objective_exclusions(olt.TARGET_CELL_FAMILY)
    names = ["cell_type", "super_class", "hemilineage", "family_code", "assigned_ol_hex1", "instance", "out_comp__mi"]
    assert fwf.excluded_feature_names(names, olt.TARGET_CELL_FAMILY) == sorted(names[:-1])
    assert "*family*" in rule.patterns
    olt.register_ol_exclusions()  # idempotent


def test_connectivity_extra_exclusions_fail_closed():
    with pytest.raises(fwf.LabelLeakageError):
        olt.assert_ol_features_allowed(["roi__me_r_in", "col__log1p_column_span"], olt.TARGET_CONNECTIVITY)
    with pytest.raises(fwf.LabelLeakageError):
        olt.assert_ol_features_allowed(["degree__out_weight_total"], olt.TARGET_CONNECTIVITY)
    olt.assert_ol_features_allowed(["roi__me_r_in", "col__log1p_column_span"], olt.TARGET_CELL_FAMILY)


# --------------------------------------------------------------------------- features / labels


def test_anatomy_features_are_fractions_without_counts():
    raw = [roi_info(ME_R=(30, 70), LO_R=(10, 90)), roi_info(PVLP_R=(5, 5))]
    frame, top = olt.anatomy_features(raw, ["1", "2"], n_top_rois=2)
    assert top == ["LO(R)", "ME(R)"]
    row = frame.iloc[0]
    assert row["roi__me_r_out"] == pytest.approx(0.75) and row["roi__lo_r_in"] == pytest.approx(90 / 160)
    assert row["roi__optic_lobe_share_in"] == pytest.approx(1.0)
    assert frame.iloc[1]["roi__other_in"] == pytest.approx(1.0)
    assert frame.iloc[1]["roi__optic_lobe_share_out"] == pytest.approx(0.0)
    numeric = frame.drop(columns="body_id")
    fractions = numeric.drop(columns=["col__log1p_column_span", "roi__entropy_bits"])
    values = fractions.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    assert values.max() <= 1.0 + 1e-9 and values.min() >= -1e-9
    assert not [c for c in numeric.columns if fwf.feature_family(c) not in olt.ANATOMY_FAMILIES]


def test_cap_per_type_is_order_independent(inputs):
    frame = inputs.neurons.assign(label="x")
    a = olt.cap_per_type(frame, max_per_type=2, salt="s")
    b = olt.cap_per_type(frame.iloc[::-1], max_per_type=2, salt="s")
    assert a["body_id"].tolist() == b["body_id"].tolist()
    assert a.groupby("cell_type").size().max() == 2


@pytest.mark.parametrize("target,labels", [
    ("cell_family", {"Mi", "Tm", "LC", "LoVC"}),
    ("super_class", {"intrinsic_optic", "visual_projection", "visual_centrifugal"}),
    ("connectivity_tier", {"high_connectivity", "baseline_connectivity"}),
    ("neurotransmitter_dominance", {"dominant_ach", "dominant_glut", "dominant_gaba"}),
    ("nt_ground_truth", {"nt_ach", "nt_glut", "nt_gaba"}),
])
def test_label_frames(inputs, tmp_path, target, labels):
    fm = olt.build_family_map(olt.type_majority_class(inputs), min_types=1)
    frame, stats = olt.label_frame(inputs, _config(target, tmp_path), fm)
    assert set(frame["label"]) == labels
    assert "Tm_unclear" not in set(frame["cell_type"])  # provisional types are never labelled
    assert set(frame["status"]) == {"Traced"}
    assert stats["label_definition"]


def test_min_types_drops_classes(inputs, tmp_path):
    fm = olt.build_family_map(olt.type_majority_class(inputs), min_types=1)
    frame, stats = olt.label_frame(inputs, _config("super_class", tmp_path, min_types=5), fm)
    assert set(frame["label"]) == {"intrinsic_optic"}  # Mi + Tm: 8 types
    assert stats["dropped_classes_few_types"] == {"visual_projection": 4, "visual_centrifugal": 4}


# --------------------------------------------------------------------------- eval dataset


def _build(inputs, tmp_path, target, **kw):
    fm = olt.build_family_map(olt.type_majority_class(inputs), min_types=1)
    return olt.build_ol_eval_dataset(inputs, _config(target, tmp_path, **kw), _eval_config(tmp_path), family_map=fm)


def test_eval_dataset_masks_every_node_of_heldout_types(inputs, tmp_path):
    data, extra = _build(inputs, tmp_path, "cell_family")
    plan = extra["plan"]
    assert data.notes["masked_split_ids_sha256"] == fme.split_ids_sha256(plan)
    frame = extra["frame"]
    held = set(frame.loc[frame["sample_id"].isin(set(plan["val"]) | set(plan["test"])), "cell_type"])
    expected = inputs.neurons["cell_type"].isin(held).sum()  # includes the Anchor-status Mi1 if Mi1 is held out
    assert data.notes["masking"]["masked_nodes"] == expected
    # no group straddles splits
    split_of = {sid: name for name in ("train", "val", "test") for sid in plan[name]}
    types = frame.groupby("cell_type")["sample_id"].apply(lambda s: {split_of[x] for x in s})
    assert all(len(v) == 1 for v in types)
    # a train neuron's partner composition never names a held-out family member's category via masked nodes
    wiring_cols = [c for c in data.features.columns if c.startswith(("in_comp__", "out_comp__"))]
    assert wiring_cols and "in_comp__unannotated" in data.features.columns
    assert not [c for c in data.features.columns if "body" in c or c in ("cell_type", "label", "sample_id")]


def test_masking_hides_heldout_categories(inputs, tmp_path):
    fm = olt.build_family_map(olt.type_majority_class(inputs), min_types=1)
    nodes = inputs.neurons[["body_id", "cell_type"]].assign(
        partner_category=[olt.partner_category(t, fm) for t in inputs.neurons["cell_type"]])
    tm_ids = nodes.loc[nodes["cell_type"].str.startswith("Tm") & (nodes["cell_type"] != "Tm_unclear"), "body_id"]
    res = fwf.build_wiring_features(dataset="ol", objective="cell_family", edges=inputs.edges, nodes=nodes,
                                    id_column="body_id", category_column="partner_category",
                                    cache_root=str(tmp_path / "c"), mask_category_ids=tm_ids.tolist())
    mi = res.frame[res.frame["node_id"].isin(nodes.loc[nodes["cell_type"].str.startswith("Mi"), "body_id"])]
    # Mi -> Tm edges exist, but every Tm node is masked: the category never appears
    assert "out_comp__tm" not in res.frame.columns and "in_comp__tm" not in res.frame.columns
    assert (mi["out_comp__unannotated"] > 0).any()


def test_connectivity_dataset_excludes_size_proxies(inputs, tmp_path):
    data, _ = _build(inputs, tmp_path, "connectivity_tier")
    cols = list(data.features.columns)
    assert not [c for c in cols if c.startswith(("degree__", "col__"))]
    assert "roi__entropy_bits" not in cols and "roi__out_in_ratio" not in cols
    fwf.assert_features_allowed(cols, "connectivity_tier")
    keys = {tok for text in data.text for tok in text.split()[0::2]}
    assert keys == {"dataset", *ols.INPUT_FEATURES["connectivity_tier"]}


@pytest.mark.parametrize("feature_set,families", [
    ("wiring_only", set(olt.WIRING_FAMILIES)), ("anatomy_only", set(olt.ANATOMY_FAMILIES))])
def test_feature_sets(inputs, tmp_path, feature_set, families):
    data, _ = _build(inputs, tmp_path, "cell_family", feature_set=feature_set)
    assert {fwf.feature_family(c) for c in data.features.columns} <= families


def test_structured_text_has_no_type_derived_tokens(inputs, tmp_path):
    data, _ = _build(inputs, tmp_path, "super_class")
    keys = {tok for text in data.text for tok in text.split()[0::2]}
    assert keys == {"dataset", *ols.STRUCTURED_TEXT_FEATURES}
    assert not keys & {"hemilineage", "hex_column", "cell_type", "type", "instance"}


def test_run_ol_target_end_to_end(inputs, tmp_path):
    report = olt.run_ol_target(
        inputs, "cell_family", report_root=str(tmp_path / "reports"),
        target_overrides={"min_types": 1, "max_per_type": 4, "cache_root": str(tmp_path / "cache")},
        eval_overrides={"cv_folds": 2, "n_bootstrap": 50, "save_models": False, "ablation": False,
                        "random_split_control": False, "train_ratio": 0.5, "val_ratio": 0.25},
        models=(fme.ModelSpec("logreg", ({"C": 1.0},), None),))
    assert report["dataset"] == "ol" and report["target"] == "cell_family"
    assert report["summary"][0]["model"] == "logreg"
    out = tmp_path / "reports" / "ol" / "cell_family"
    assert (out / "report.json").is_file() and (out / "family_map.json").is_file()
    assert "snapshots" not in str(out)
    # R1: optic-lobe families are connectivity-defined -> reported, never gated
    assert report["notes"]["label_provenance"] == "connectivity_defined"
    assert {r["gate"] for r in report["summary"]} == {"not_gated"}
    assert report["summary"][0]["reporting_frame"] == "recovery of connectivity-derived annotations"


def test_explicit_paths_all_or_none(tmp_path):
    with pytest.raises(ValueError):
        olt.load_ol_structured_inputs(neurons_path=tmp_path / "x")


def test_config_validation():
    with pytest.raises(ValueError):
        olt.OlTargetConfig(target="region_specialization_tier").validate()
    with pytest.raises(ValueError):
        olt.OlTargetConfig(target="cell_family", feature_set="all").validate()


def test_samples_module_structured_helpers(inputs):
    frame = inputs.neurons[inputs.neurons["status"] == "Traced"].head(3)
    tokens = ols.anatomy_tokens(frame)
    assert set(tokens) == set(frame["body_id"])
    text = ols.render_input_text(tokens[frame["body_id"].iloc[0]], ols.STRUCTURED_TEXT_FEATURES)
    assert text.startswith("dataset ol11 side ")
    assert len(text.split()) == 2 * (1 + len(ols.STRUCTURED_TEXT_FEATURES))


def test_no_network_imports():
    source = open(olt.__file__, encoding="utf-8").read()
    for name in ("requests", "urllib", "http.client", "socket", "httpx"):
        assert f"import {name}" not in source


def test_text_view_keys_obey_objective_exclusions():
    import flybrain_wiring_features as fwf

    for target in olt.OL_TARGETS:
        keys = olt._text_keys(target)
        assert not fwf.excluded_feature_names(list(keys), target), target
    assert "hemilineage" not in olt._text_keys("nt_literature")
