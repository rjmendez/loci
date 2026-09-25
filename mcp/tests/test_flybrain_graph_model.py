"""Offline tests for flybrain_graph_model (tiny synthetic graphs, CPU; CUDA tests skip when unavailable)."""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch = pytest.importorskip("torch")

import flybrain_graph_model as fgm  # noqa: E402
import flybrain_learners as fl  # noqa: E402
import flybrain_model_eval as fme  # noqa: E402
import flybrain_wiring_features as fwf  # noqa: E402

CLASSES = ("a", "b", "c")


def _synthetic(n_labelled=360, n_hubs=60, n_groups=36, seed=0):
    """Labelled nodes carry NO class signal in their own features; their class is readable only
    from the features of the (unlabelled) hub neurons that project onto them.

    Groups (cell types) are class-pure, like real cell types. Random labelled->labelled edges
    are class-independent, so the neighbour-label vote is uninformative too.
    """
    rng = np.random.default_rng(seed)
    lab_group = np.arange(n_labelled) % n_groups
    lab_class = np.asarray([0, 0, 1, 2])[lab_group % 4]  # imbalanced: class a is the majority everywhere
    hub_class = np.arange(n_hubs) % 3
    ids = [f"n{i}" for i in range(n_labelled)] + [f"h{j}" for j in range(n_hubs)]
    src, dst, w = [], [], []
    for i in range(n_labelled):
        hubs = rng.choice(np.flatnonzero(hub_class == lab_class[i]), size=3, replace=False)
        for h in hubs:
            src.append(n_labelled + int(h))
            dst.append(i)
            w.append(float(rng.integers(3, 12)))
        for j in rng.choice(n_labelled, size=3, replace=False):
            if j != i:
                src.append(i)
                dst.append(int(j))
                w.append(1.0)
    topo = fgm.graph_from_arrays(ids, src, dst, w)
    n = len(ids)
    feats = pd.DataFrame({
        "sig__a": np.zeros(n), "sig__b": np.zeros(n), "sig__c": np.zeros(n),
        "noise__x": rng.normal(size=n), "noise__y": rng.normal(size=n),
    })
    for k, col in enumerate(("sig__a", "sig__b", "sig__c")):
        feats.loc[n_labelled + np.flatnonzero(hub_class == k), col] = 1.0
    ctx = fgm.GraphContext(topo, feats)
    frame = feats.iloc[:n_labelled].copy()
    frame.insert(0, "node", ids[:n_labelled])
    frame["label"] = [CLASSES[c] for c in lab_class]
    frame["cell_type"] = [f"t{g}" for g in lab_group]
    return ctx, frame


def _rows(frame, idx):
    X = frame.iloc[idx][["sig__a", "sig__b", "sig__c", "noise__x", "noise__y"]].reset_index(drop=True)
    X[fgm.GRAPH_NODE_COLUMN] = frame["node"].iloc[idx].tolist()
    return X, frame["label"].to_numpy(dtype=object)[idx], frame["cell_type"].to_numpy(dtype=object)[idx]


def _split(frame):
    g = frame["cell_type"].str[1:].astype(int)
    test = (g % 5 == 0).to_numpy()
    return np.flatnonzero(~test), np.flatnonzero(test)


FAST = {"hidden": 32, "epochs": 150, "patience": 30, "lr": 0.01, "dropout": 0.1}


def _fit(ctx, frame, tr, **params):
    X, y, g = _rows(frame, tr)
    learner = fl.make_learner("graphsage", seed=0, **{**FAST, **params})
    learner.attach_graph(ctx)
    return learner.fit(X, y, groups=g)


# ---------------------------------------------------------------- topology


def test_graph_from_arrays_sums_duplicates_and_drops_self_loops():
    topo = fgm.graph_from_arrays(["x", "y", "z"], [0, 0, 1, 2], [1, 1, 1, 0], [2, 3, 5, 1])
    pairs = {(int(s), int(d)): float(w) for s, d, w in zip(topo.src, topo.dst, topo.weight)}
    assert pairs == {(0, 1): 5.0, (2, 0): 1.0}
    assert list(topo.indices(["z", "x"])) == [2, 0]
    with pytest.raises(ValueError, match="not in the graph"):
        topo.indices(["nope"])
    with pytest.raises(ValueError, match="not unique"):
        fgm.graph_from_arrays(["x", "x"], [0], [1])


def _write_edges(path, rows):
    import pyarrow as pa
    import pyarrow.feather as feather

    pre, post, w = zip(*rows)
    feather.write_feather(pa.table({"body_pre": pa.array(pre, pa.int64()), "body_post": pa.array(post, pa.int64()),
                                    "weight": pa.array(w, pa.int32())}), str(path))


def test_build_graph_topology_streams_filters_and_caches(tmp_path):
    edges_path = tmp_path / "edges.feather"
    # 99 is not in the node table; (3 -> 3) is a self loop; (1 -> 2, w=1) is below min_weight.
    _write_edges(edges_path, [(1, 2, 1), (2, 3, 4), (3, 1, 7), (99, 1, 50), (3, 3, 9), (1, 3, 2)])
    src = fwf.EdgeSource(path=str(edges_path), pre="body_pre", post="body_post", weight="weight",
                         provenance={"manifest_sha256": "x"})
    cache = tmp_path / "cache"
    topo = fgm.build_graph_topology(dataset="toy", edges=src, node_ids=["1", "2", "3"], min_weight=2, cache_root=cache)
    got = {(topo.node_ids[s], topo.node_ids[d]): float(w) for s, d, w in zip(topo.src, topo.dst, topo.weight)}
    assert got == {("2", "3"): 4.0, ("3", "1"): 7.0, ("1", "3"): 2.0}
    assert topo.meta["stats"]["edges_scanned"] == 6
    files = sorted(p.name for p in (cache / "graph-topology" / "toy").iterdir())
    assert len(files) == 2 and files[0].endswith(".json") and files[1].endswith(".npz")
    again = fgm.build_graph_topology(dataset="toy", edges=src, node_ids=["1", "2", "3"], min_weight=2, cache_root=cache)
    assert again.fingerprint == topo.fingerprint and np.array_equal(again.src, topo.src)
    other = fgm.build_graph_topology(dataset="toy", edges=src, node_ids=["1", "2", "3"], min_weight=1, cache_root=cache)
    assert other.fingerprint != topo.fingerprint and other.n_edges == 4
    with pytest.raises(ValueError, match="snapshots"):
        fgm.build_graph_topology(dataset="toy", edges=src, node_ids=["1", "2"], cache_root=tmp_path / "snapshots" / "c")


def test_adjacency_is_row_normalized_log_weight_and_exclusion_drops_edges():
    topo = fgm.graph_from_arrays(["p", "q", "r"], [0, 1, 0], [2, 2, 1], [1.0, 3.0, 2.0])
    ctx = fgm.GraphContext(topo, pd.DataFrame({"f__x": [0.0, 1.0, 2.0]}))
    adj = ctx.adjacency(None, "cpu")
    dense_in, dense_out = adj.to_dense()
    w1, w3 = np.log1p(1.0), np.log1p(3.0)
    assert dense_in[2, 0] == pytest.approx(w1 / (w1 + w3)) and dense_in[2, 1] == pytest.approx(w3 / (w1 + w3))
    assert np.allclose(dense_out.sum(axis=1), [1.0, 1.0, 0.0])
    assert np.allclose(adj.a_in_t.to_dense().numpy(), dense_in.T)
    assert np.allclose(adj.a_out_t.to_dense().numpy(), dense_out.T)
    e_in, _ = ctx.adjacency(np.array([False, True, False]), "cpu").to_dense()
    assert e_in[:, 1].sum() == 0 and e_in[1].sum() == 0
    assert e_in[2, 0] == pytest.approx(1.0)


def test_custom_spmm_gradient_matches_dense():
    ctx, _ = _synthetic(n_labelled=60, n_hubs=12, n_groups=6)
    adj = ctx.adjacency(None, "cpu")
    dense_in, _ = adj.to_dense()
    h = torch.randn(ctx.topology.n_nodes, 4, dtype=torch.float32, requires_grad=True)
    g = torch.randn(ctx.topology.n_nodes, 4)
    (fgm._spmm(adj.a_in, adj.a_in_t, h) * g).sum().backward()
    ref = torch.from_numpy(dense_in.T.astype(np.float32)) @ g
    assert torch.allclose(h.grad, ref, atol=1e-5)


def test_neighbor_vote_uses_training_labels_only():
    topo = fgm.graph_from_arrays(["a0", "a1", "b0", "q", "lonely"], [0, 1, 2, 3], [3, 3, 3, 2], [5, 5, 1, 1])
    pred = fgm.neighbor_vote_predictions(topo, np.array([0, 1, 2]), ["x", "x", "y"], np.array([3, 4]),
                                         classes=("x", "y"), fallback="y")
    assert pred == ["x", "y"]  # q: two heavy x partners; lonely: no labelled partner -> fallback


# ---------------------------------------------------------------- learner


def test_gnn_reads_partner_features_and_beats_no_message_passing():
    ctx, frame = _synthetic()
    tr, te = _split(frame)
    ctx = ctx.with_heldout(frame["node"].iloc[te])
    Xte, yte, _ = _rows(frame, te)
    for mode in fgm.MODES:
        learner = _fit(ctx, frame, tr, mode=mode)
        acc = np.mean(np.asarray(learner.predict(Xte)) == yte)
        assert acc > 0.9, (mode, acc)
        proba = learner.predict_proba(Xte)
        assert proba.shape == (len(te), 3) and np.allclose(proba.sum(axis=1), 1.0)
        assert learner.fit_info["mode"] == mode and learner.fit_info["early_stopping"] == "grouped_inner_val"
    flat = _fit(ctx, frame, tr, layers=0)
    flat_acc = np.mean(np.asarray(flat.predict(Xte)) == yte)
    assert flat_acc < 0.6  # own features are pure noise: without message passing it is near chance


def test_inductive_training_ignores_heldout_nodes_entirely():
    ctx, frame = _synthetic()
    tr, te = _split(frame)
    heldout = frame["node"].iloc[te].tolist()
    a = _fit(ctx.with_heldout(heldout), frame, tr, mode="inductive")
    feats = ctx.features.copy()
    pos = ctx.topology.indices(heldout)
    feats.loc[pos, "noise__x"] = 1e3  # scramble held-out nodes' features
    feats.loc[pos, "sig__a"] = 1.0
    scrambled = fgm.GraphContext(ctx.topology, feats, frozenset(heldout))
    b = _fit(scrambled, frame, tr, mode="inductive")
    for key, value in a.net.state_dict().items():
        assert torch.equal(value, b.net.state_dict()[key]), key
    # transductive mode does read held-out nodes' features (never labels) during training
    c = _fit(ctx.with_heldout(heldout), frame, tr, mode="transductive")
    d = _fit(scrambled, frame, tr, mode="transductive")
    assert any(not torch.equal(v, d.net.state_dict()[k]) for k, v in c.net.state_dict().items())


def test_heldout_rows_missing_graph_and_mismatched_features_fail_closed():
    ctx, frame = _synthetic(n_labelled=90, n_hubs=15, n_groups=9)
    tr, te = _split(frame)
    X, y, g = _rows(frame, tr)
    learner = fl.make_learner("graphsage", **FAST)
    with pytest.raises(ValueError, match="attach_graph"):
        learner.fit(X, y, groups=g)
    learner.attach_graph(ctx.with_heldout(frame["node"].iloc[tr[:2]]))
    with pytest.raises(ValueError, match="held-out"):
        learner.fit(X, y, groups=g)
    learner.attach_graph(ctx)
    bad = X.copy()
    bad.loc[0, "noise__x"] += 1.0
    with pytest.raises(ValueError, match="differs from the graph context"):
        learner.fit(bad, y, groups=g)
    with pytest.raises(ValueError, match="node column"):
        learner.fit(X.drop(columns=[fgm.GRAPH_NODE_COLUMN]), y, groups=g)
    with pytest.raises(ValueError, match="mode"):
        fl.make_learner("graphsage", mode="semi")


def test_cpu_training_is_deterministic():
    ctx, frame = _synthetic(n_labelled=120, n_hubs=30, n_groups=12)
    tr, te = _split(frame)
    Xte, _, _ = _rows(frame, te)
    a = _fit(ctx, frame, tr).predict_proba(Xte)
    b = _fit(ctx, frame, tr).predict_proba(Xte)
    assert np.array_equal(a, b)


def test_save_load_roundtrip_weights_only_and_graph_binding(tmp_path):
    ctx, frame = _synthetic(n_labelled=120, n_hubs=30, n_groups=12)
    tr, te = _split(frame)
    Xte, yte, _ = _rows(frame, te)
    Xtr, ytr, _ = _rows(frame, tr)
    base = _fit(ctx, frame, tr)
    cal = fl.CalibratedLearner(base, method="temperature", cv=0).fit_prefit(Xtr, ytr)
    root = tmp_path / "harness"
    manifest = fl.save_learner(cal, root / "models" / "gnn")
    assert "base/model.pt" in manifest["files"] and "base/graph.json" in manifest["files"]
    state = torch.load(root / "models" / "gnn" / "base" / "model.pt", weights_only=True)
    assert all(isinstance(v, torch.Tensor) for v in state.values())
    graph_json = json.loads((root / "models" / "gnn" / "base" / "graph.json").read_text())
    assert graph_json["topology_fingerprint"] == ctx.topology.fingerprint
    loaded = fl.load_learner(root / "models" / "gnn", trusted_root=root,
                             expected_manifest_sha256=manifest["manifest_sha256"])
    with pytest.raises(ValueError, match="attach_graph"):
        loaded.predict_proba(Xte)
    fgm.attach_graph(loaded, ctx)
    assert np.allclose(loaded.predict_proba(Xte), cal.predict_proba(Xte), atol=1e-6)
    other, _ = _synthetic(n_labelled=120, n_hubs=30, n_groups=12, seed=5)
    with pytest.raises(ValueError, match="topology differs"):
        fgm.attach_graph(loaded, other)


# ---------------------------------------------------------------- evaluation protocol


def _eval_data(ctx, frame, cfg, *, mask_sha=True):
    plan = fme.plan_grouped_split(frame["node"].tolist(), frame[["cell_type"]].to_dict(orient="records"),
                                  ("cell_type",), cfg)
    notes = {"masked_split_ids_sha256": fme.split_ids_sha256(plan) if mask_sha else "0" * 64}
    return fme.EvalDataset.from_frame(frame, dataset="toy", target="cell_class", id_column="node", label_column="label",
                                      feature_columns=["sig__a", "sig__b", "sig__c", "noise__x", "noise__y"],
                                      group_columns=["cell_type"], notes=notes)


def _gcfg(tmp_path, **kwargs):
    ecfg = fme.EvalConfig(models=(), cv_folds=2, n_bootstrap=100, n_threads=2, report_root=str(tmp_path / "reports"),
                          run_label="t", random_split_control=True)
    return fgm.GraphEvalConfig(eval=ecfg, gnn_grid=(FAST,), device="cpu",
                               hgb=fme.ModelSpec("hgb", ({"max_iter": 50},), "isotonic"), **kwargs)


def test_run_graph_evaluation_end_to_end(tmp_path):
    ctx, frame = _synthetic(n_labelled=600, n_groups=60)  # big enough that the test split holds every class
    cfg = _gcfg(tmp_path)
    report = fgm.run_graph_evaluation(_eval_data(ctx, frame, cfg.eval), ctx, cfg)
    rows = {r["model"]: r for r in report["summary"]}
    assert set(rows) == {"graphsage_transductive", "graphsage_inductive", "hgb"}
    for key in ("majority", "best_trivial", "best_trivial_rule", "model_acc", "model_ci", "macro_f1", "ece",
                "shuffle_acc", "n_train", "n_test", "gate"):
        assert key in rows["hgb"]
    test_graph = report["baselines"]["graph"]["splits"]["test"]
    assert fgm.RULE_NEIGHBOR_VOTE in test_graph["accuracy"]
    for label in ("graphsage_transductive", "graphsage_inductive"):
        r = rows[label]
        assert r["best_trivial_rule"].startswith("graph:")
        assert r["model_acc"] > 0.9 and r["gate"] == "pass", r
        assert report["models"][label]["shuffle_control"]["collapsed_to_majority"]
        vs = report["models"][label]["vs_hgb"]
        assert vs["accuracy_diff"] > 0.2  # hgb sees only the node's own (noise) features
        assert report["models"][label]["artifact"]["reload_max_abs_proba_diff"] < 1e-5
    assert rows["hgb"]["gate"] == "fail"
    abl = {r["family"]: r for r in report["ablation"]["drop_one_family"]}
    assert abl["message_passing(layers=0)"]["delta_accuracy"] < -0.2
    assert "sig" in abl and "noise" in abl  # (no sign assertion: hub identity is learnable from hub noise too)
    assert "drop_one_family" in report["ablation_hgb"]
    out = tmp_path / "reports" / "toy" / "cell_class" / "t"
    md = (out / "report.md").read_text()
    assert "## Graph model" in md and "neighbor_vote" in md
    json.loads((out / "report.json").read_text())


def test_run_graph_evaluation_refuses_features_masked_for_another_split(tmp_path):
    ctx, frame = _synthetic(n_labelled=120, n_hubs=30, n_groups=12)
    cfg = _gcfg(tmp_path)
    with pytest.raises(ValueError, match="masked for a different split"):
        fgm.run_graph_evaluation(_eval_data(ctx, frame, cfg.eval, mask_sha=False), ctx, cfg)


def test_run_graph_evaluation_rejects_label_defining_context_features(tmp_path):
    ctx, frame = _synthetic(n_labelled=120, n_hubs=30, n_groups=12)
    leaky = fgm.GraphContext(ctx.topology, ctx.features.assign(cell_type_code=0.0))
    cfg = _gcfg(tmp_path)
    with pytest.raises(fwf.LabelLeakageError):
        fgm.run_graph_evaluation(_eval_data(ctx, frame, cfg.eval), leaky, cfg)


def _cuda_ok():
    # Opt-in: on a shared WSL box even torch.cuda.is_available() can block on a wedged driver.
    return os.environ.get("FLYBRAIN_CUDA_TESTS") == "1" and torch.cuda.is_available()


@pytest.mark.skipif(os.environ.get("FLYBRAIN_CUDA_TESTS") != "1", reason="CUDA tests are opt-in (FLYBRAIN_CUDA_TESTS=1)")
def test_cuda_fit_matches_cpu_quality():
    if not _cuda_ok():
        pytest.skip("CUDA not available")
    ctx, frame = _synthetic()
    tr, te = _split(frame)
    Xte, yte, _ = _rows(frame, te)
    learner = _fit(ctx, frame, tr, device="cuda:0")
    assert learner.fit_info["device"].startswith("cuda")
    assert np.mean(np.asarray(learner.predict(Xte)) == yte) > 0.9


def test_torch_macro_f1_matches_harness_and_stop_metric_validated():
    rng = np.random.default_rng(1)
    t = rng.integers(0, 4, size=200)
    p = np.where(rng.random(200) < 0.6, t, rng.integers(0, 4, size=200))
    p[p == 3] = 2  # class 3 never predicted
    got = fgm._torch_macro_f1(torch.from_numpy(p), torch.from_numpy(t), 4)
    assert got == pytest.approx(fme.macro_f1([str(v) for v in t], [str(v) for v in p]))
    with pytest.raises(ValueError, match="stop_metric"):
        fl.make_learner("graphsage", stop_metric="accuracy")


def test_filtered_topology_and_no_refit_mode():
    ctx, frame = _synthetic(n_labelled=120, n_hubs=30, n_groups=12)
    topo = ctx.topology
    strong = topo.filtered(5.0)
    assert strong.n_edges < topo.n_edges and np.all(strong.weight >= 5.0)
    assert strong.fingerprint != topo.fingerprint and topo.filtered(0.0) is topo
    tr, te = _split(frame)
    learner = _fit(fgm.GraphContext(strong, ctx.features), frame, tr, refit=False)
    assert learner.fit_info["refit_on_all_train_rows"] is False
    Xte, yte, _ = _rows(frame, te)
    assert np.mean(np.asarray(learner.predict(Xte)) == yte) > 0.8
