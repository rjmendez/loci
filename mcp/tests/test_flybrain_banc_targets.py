"""Tests for flybrain_banc_targets (BANC real-model targets) on a synthetic snapshot."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_banc_adapter as banc  # noqa: E402
import flybrain_banc_targets as bt  # noqa: E402
import flybrain_brain_cluster_banc_samples as bs  # noqa: E402
import flybrain_model_eval as fme  # noqa: E402
import flybrain_wiring_features as fwf  # noqa: E402

N = 240
CLASSES = ("central_brain_intrinsic", "optic_lobe_intrinsic", "sensory", "motor")
FLOW = {"central_brain_intrinsic": "intrinsic", "optic_lobe_intrinsic": "intrinsic", "sensory": "afferent",
        "motor": "efferent"}
NEUROPIL = {"central_brain_intrinsic": "SMP", "optic_lobe_intrinsic": "ME", "sensory": "AL", "motor": "LegNp_T1"}
NT_NAMES = ("acetylcholine", "dopamine", "gaba", "glutamate", "histamine", "octopamine", "serotonin", "tyramine")


def rid(i: int) -> str:
    return f"72057594{i:010d}"


def cls_of(i: int) -> str:
    return CLASSES[(i % 24) % 4]


def _synthetic_tables(seed: int = 0):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame({
        "banc_888_id": [rid(i) for i in range(N)],
        "proofread": ["TRUE" if i % 17 else "FALSE" for i in range(N)],
        "side": ["left" if i % 2 else "right" for i in range(N)],
        "root_region": ["ITO_midbrain_SMP_L" if cls_of(i) != "motor" else "COURT_vnc_ProNM-T1" for i in range(N)],
        "region": ["central_brain" if cls_of(i) != "motor" else "ventral_nerve_cord" for i in range(N)],
        "super_class": [cls_of(i) for i in range(N)],
        "cell_class": [f"cc_{cls_of(i)}_{(i % 24) // 12}" for i in range(N)],
        "flow": [FLOW[cls_of(i)] for i in range(N)],
        "hemilineage": [None] * N,
        "cell_type": [f"ct{i % 24}" for i in range(N)],
    })
    edges = {}
    for i in range(N):
        ci = cls_of(i)
        if ci == "motor":
            continue  # motor neurons: inputs only in this toy
        for _ in range(8):
            j = int(rng.integers(0, N))
            if j == i or cls_of(j) == "sensory":
                continue  # sensory neurons receive no input
            edges[(i, j)] = edges.get((i, j), 0) + int(rng.integers(1, 30))
    pre, post, count = zip(*[(rid(a), rid(b), c) for (a, b), c in sorted(edges.items())])
    edgelist = pa.table({"pre": list(pre), "post": list(post), "count": pa.array(count, pa.int32()),
                         "norm": [0.1] * len(pre), "post_count": pa.array([1] * len(pre), pa.int32()),
                         "pre_count": pa.array([1] * len(pre), pa.int32())})
    syn_pre, syn_post, syn_np = [], [], []
    for (a, b), c in sorted(edges.items()):
        for k in range(min(c, 5)):
            syn_pre.append(rid(a))
            syn_post.append(rid(b))
            syn_np.append(NEUROPIL[cls_of(a)] if k % 2 else NEUROPIL[cls_of(b)])
    enriched = pa.table({"pre_root_id": pa.array(syn_pre, pa.large_string()),
                         "post_root_id": pa.array(syn_post, pa.large_string()),
                         "neuropil": syn_np})
    metrics = pd.DataFrame({
        "banc_888_id": [rid(i) for i in range(N)],
        "l2_cable_length_um": rng.uniform(10, 1000, N), "volume_nm3": rng.uniform(1e6, 1e9, N),
        "l2_nodes": rng.uniform(10, 500, N), "branchpoints": rng.integers(1, 200, N).astype(np.int32),
        "endpoints": rng.integers(1, 200, N).astype(np.int32), "axon_length": rng.uniform(1, 500, N),
        "dend_length": rng.uniform(1, 500, N), "mitochondria": rng.uniform(0, 100, N),
        "pd_width": rng.uniform(0, 2, N), "segregation_index": rng.uniform(0, 1, N),
        "projection_score": rng.uniform(0, 1, N),
    })
    lines = [",".join(["root_id", *NT_NAMES, "neurotransmitter_predicted", "neurotransmitter_score", "count"])]
    for i in range(N):
        nt = NT_NAMES[(i % 24) % 3]
        lines.append(",".join([rid(i), *["0"] * 8, nt, "0.8", "50"]))
    return meta, edgelist, enriched, metrics, "\n".join(lines) + "\n"


@pytest.fixture()
def snapshot(tmp_path):
    storage = tmp_path / "flybrain"
    snap = storage / "snapshots" / "BANC" / "banc_888"
    meta, edgelist, enriched, metrics, nt = _synthetic_tables()
    paths = {
        bt.REL_META: snap / bt.REL_META,
        bt.REL_EDGELIST: snap / bt.REL_EDGELIST,
        bt.REL_NT: snap / bt.REL_NT,
        bt.REL_METRICS: snap / bt.REL_METRICS,
        bt.REL_ENRICHED: snap / bt.REL_ENRICHED,
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    feather.write_feather(pa.Table.from_pandas(meta, preserve_index=False), str(paths[bt.REL_META]))
    feather.write_feather(edgelist, str(paths[bt.REL_EDGELIST]))
    feather.write_feather(pa.Table.from_pandas(metrics, preserve_index=False), str(paths[bt.REL_METRICS]))
    pq.write_table(enriched, str(paths[bt.REL_ENRICHED]))
    paths[bt.REL_NT].write_text(nt, encoding="utf-8")
    files = [{"relative_path": rel, "size_bytes": p.stat().st_size, "sha256": banc.sha256_file(p)}
             for rel, p in sorted(paths.items())]
    manifest = banc.build_banc_manifest(files=files, source_uri="synthetic://banc", retrieved_at="2026-09-23T00:00:00Z",
                                        run_id="run-test", generated_at="2026-09-23T00:00:00Z",
                                        next_check_due="2099-01-01T00:00:00Z")
    banc.write_banc_manifest(snap, manifest)
    return storage, snap


def _listing(root: Path):
    return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*"))


def _config(tmp_path, storage, **kw):
    return bt.BancTargetConfig(storage_root=str(storage), cache_root=str(tmp_path / "cache"),
                               stamp_dir=str(tmp_path / "stamps"),
                               selection=bt.SelectionConfig(min_class_count=10, min_class_cell_types=3,
                                                            max_per_class=40), **kw)


def _eval_config(tmp_path, **kw):
    base = dict(models=(fme.ModelSpec("logreg", ({"C": 1.0},), "temperature"),), cv_folds=2, n_bootstrap=50,
                n_threads=2, report_root=str(tmp_path / "reports"), run_label="t", random_split_control=False)
    base.update(kw)
    return fme.EvalConfig(**base)


# --------------------------------------------------------------------------- inputs


def test_open_inputs_verifies_hashes_and_never_writes_into_snapshot(tmp_path, snapshot):
    storage, snap = snapshot
    before = _listing(snap)
    inputs = bt.open_banc_inputs([bt.REL_META, bt.REL_EDGELIST], storage_root=storage,
                                 stamp_dir=tmp_path / "stamps")
    assert set(inputs.hash_methods.values()) == {"hashed"}
    again = bt.open_banc_inputs([bt.REL_META, bt.REL_EDGELIST], storage_root=storage, stamp_dir=tmp_path / "stamps")
    assert set(again.hash_methods.values()) == {"stamp"}
    assert _listing(snap) == before
    assert len(list((tmp_path / "stamps").glob("*.json"))) == 2
    assert inputs.provenance()["manifest_sha256"] == again.manifest_sha256


def test_open_inputs_refuses_stamp_dir_inside_snapshots(tmp_path, snapshot):
    storage, snap = snapshot
    with pytest.raises(ValueError, match="snapshots"):
        bt.open_banc_inputs([bt.REL_META], storage_root=storage, stamp_dir=snap / "manifest" / "stamps")


def test_open_inputs_fails_closed_on_content_change(tmp_path, snapshot):
    storage, snap = snapshot
    path = snap / bt.REL_NT
    data = bytearray(path.read_bytes())
    data[-3] = ord("9") if data[-3] != ord("9") else ord("8")  # same size, different content
    path.write_bytes(bytes(data))
    with pytest.raises(banc.BancAdapterError) as exc:
        bt.open_banc_inputs([bt.REL_NT], storage_root=storage, stamp_dir=tmp_path / "stamps")
    assert exc.value.code == banc.BancAdapterErrorCode.INTEGRITY_MISMATCH


def test_open_inputs_rejects_unlisted_file(tmp_path, snapshot):
    storage, _ = snapshot
    with pytest.raises(banc.BancAdapterError) as exc:
        bt.open_banc_inputs(["metadata/not_listed.feather"], storage_root=storage, stamp_dir=tmp_path / "stamps")
    assert exc.value.code == banc.BancAdapterErrorCode.PRODUCT_MISSING


# --------------------------------------------------------------------------- labels / selection


def test_annotation_labels_harmonize_and_require_proofread():
    nodes = pd.DataFrame({
        "super_class": ["central_brain_intrinsic", "sensory_ascending", "glia", None, "motor"],
        "super_class_h": [fwf.harmonize_super_class(v) for v in
                          ["central_brain_intrinsic", "sensory_ascending", "glia", None, "motor"]],
        "flow": ["intrinsic", "afferent", None, "weird", "efferent"],
        "cell_class": ["Kenyon Cell", None, "astro", "x", "leg_motor"],
        "proofread_bool": [True, True, True, True, False],
    })
    assert bt.annotation_labels(nodes, "super_class").tolist() == ["intrinsic_central", "sensory", None, None, None]
    assert bt.annotation_labels(nodes, "flow").tolist() == ["intrinsic", "afferent", None, None, None]
    assert bt.annotation_labels(nodes, "cell_class").tolist() == ["kenyon_cell", None, "astro", "x", None]
    with pytest.raises(ValueError):
        bt.annotation_labels(nodes, "connectivity_tier")


def test_select_samples_drops_small_and_single_type_classes_and_caps_deterministically():
    ids = [rid(i) for i in range(60)]
    nodes = pd.DataFrame({"banc_888_id": ids,
                          "cell_type": [f"t{i % 6}" for i in range(40)] + ["solo"] * 15 + [None] * 5,
                          "hemilineage": [None] * 60})
    labels = pd.Series(["a"] * 40 + ["b"] * 15 + ["c"] * 5, dtype=object)
    wired = np.ones(60, dtype=bool)
    wired[0] = False
    cfg = bt.SelectionConfig(min_class_count=10, min_class_cell_types=3, max_per_class=20)
    frame, info = bt.select_annotation_samples(nodes, labels, wired, cfg)
    assert info["kept_classes"] == ["a"]
    assert info["dropped_classes"] == {"b": 15, "c": 5}
    assert len(frame) == 20 and rid(0) not in set(frame["banc_888_id"])
    again, _ = bt.select_annotation_samples(nodes.iloc[::-1].reset_index(drop=True),
                                            labels.iloc[::-1].reset_index(drop=True), wired[::-1], cfg)
    assert again["banc_888_id"].tolist() == frame["banc_888_id"].tolist()


def test_local_exclusions_drop_size_proxies_for_connectivity_only():
    cols = ["morph__log_cable_length_um", "morph__segregation_index", "morph__axon_fraction",
            "morph__mito_per_um", "out_comp__sensory"]
    assert bt.local_exclusions(cols, "connectivity_tier") == ["morph__log_cable_length_um", "morph__mito_per_um"]
    assert bt.local_exclusions(cols, "super_class") == []


def test_binned_text_uses_train_quantiles_and_passes_leakage_check():
    frame = pd.DataFrame({"out_comp__sensory": [0.0, 0.1, 0.2, 0.3, 100.0, np.nan],
                          "annot__side": ["l", "r", "l", "r", "l", None]})
    text = bt.binned_text(frame, np.array([0, 1, 2, 3]), n_bins=2)
    assert text[4].split()[1] == "annot__side_l"
    assert "out_comp__sensory_q1" in text[4] and "out_comp__sensory_qna" in text[5]
    keys = set()
    for row in text:
        keys.update(row.split()[0::2])
    fwf.assert_features_allowed(sorted(keys), "neurotransmitter_dominance")


def test_attach_structured_features_keeps_payload_and_fails_closed():
    payload = {"samples": [{"sample_id": "s1", "input_text": "x", "metadata": {"root_id": rid(1)}},
                           {"sample_id": "s2", "input_text": "y", "metadata": {"root_id": rid(2)}}],
               "metadata": {"objective": "flow"}}
    feats = pd.DataFrame({"node_id": [rid(1)], "recip__partner_frac": [np.nan], "out_comp__sensory": [0.5]})
    out = bs.attach_structured_features(payload, feats, objective="flow")
    assert out["samples"][0]["features"] == {"recip__partner_frac": None, "out_comp__sensory": 0.5}
    assert out["samples"][1]["features"] == {"recip__partner_frac": None, "out_comp__sensory": None}
    assert out["metadata"]["structured_features"]["missing_rows"] == 1
    assert out["samples"][0]["input_text"] == "x" and "features" not in payload["samples"][0]
    with pytest.raises(fwf.LabelLeakageError):
        bs.attach_structured_features(payload, feats.assign(degree__out_weight_total=1.0),
                                      objective="connectivity_tier")


# --------------------------------------------------------------------------- end to end


def test_super_class_dataset_masks_every_non_train_node_and_evaluates(tmp_path, snapshot):
    storage, snap = snapshot
    before = _listing(snap)
    config = _config(tmp_path, storage)
    ecfg = _eval_config(tmp_path)
    data = bt.build_eval_dataset("super_class", config, ecfg, log=lambda *_: None)
    cols = list(data.features.columns)
    assert not [c for c in cols if c in {"super_class", "cell_class", "flow", "cell_type"} or c.startswith("annot__")]
    assert any(c.startswith("out_np__") for c in cols) and any(c.startswith("out2_comp__") for c in cols)
    assert any(c.startswith("morph__") for c in cols)
    plan = fme.plan_grouped_split(list(data.sample_ids), list(data.group_values), bt.GROUP_KEYS, ecfg)
    assert data.notes["masked_split_ids_sha256"] == fme.split_ids_sha256(plan)
    assert data.notes["category_mask"]["masked_nodes"] == N - len(plan["train"])
    report = bt.run_target("super_class", config, ecfg, log=lambda *_: None)
    assert report["leakage_check"]["pass"] and report["summary"][0]["target"] == "super_class"
    assert (tmp_path / "reports" / "banc" / "super_class" / "t" / "report.json").is_file()
    assert _listing(snap) == before


def test_masking_hides_train_labels_of_held_out_partners(tmp_path, snapshot):
    storage, _ = snapshot
    config = _config(tmp_path, storage)
    inputs = bt.open_banc_inputs([bt.REL_META, bt.REL_EDGELIST], storage_root=storage,
                                 stamp_dir=tmp_path / "stamps")
    nodes = bt.load_node_table(inputs.path(bt.REL_META))
    everything = nodes["banc_888_id"].tolist()
    masked = bt.composition_features(inputs, nodes, objective="super_class", mask_ids=everything,
                                     cache_root=config.cache_root).frame
    comp_cols = [c for c in masked.columns if c.startswith("out_comp__") and not c.endswith(("unannotated", "entropy"))]
    assert masked[comp_cols].fillna(0).to_numpy().sum() == 0  # all categories hidden -> only 'unannotated'


def test_connectivity_dataset_excludes_degree_and_size_proxies(tmp_path, snapshot):
    storage, _ = snapshot
    config = _config(tmp_path, storage, legacy_max_samples=500)
    data = bt.build_eval_dataset("connectivity_tier", config, _eval_config(tmp_path), log=lambda *_: None)
    cols = list(data.features.columns)
    assert not [c for c in cols if c.startswith("degree__") or "n_neuropils" in c]
    assert "morph__log_cable_length_um" not in cols and "morph__segregation_index" in cols
    assert "annot__super_class" in cols
    assert "masked_split_ids_sha256" not in data.notes
    fwf.assert_features_allowed(cols, "connectivity_tier")


def test_no_neuropil_variant_drops_family(tmp_path, snapshot):
    storage, _ = snapshot
    config = _config(tmp_path, storage, drop_families=("neuropil",))
    data = bt.build_eval_dataset("flow", config, _eval_config(tmp_path), log=lambda *_: None)
    assert not [c for c in data.features.columns if "_np__" in c]
    assert set(data.labels.tolist()) <= {"afferent", "intrinsic", "efferent"}
