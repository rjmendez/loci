"""Offline tests for the l1em structured features, non-circular targets and eval wiring (synthetic snapshot)."""

import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_brain_cluster_l1em_samples as l1s  # noqa: E402
import flybrain_l1em_adapter as l1em  # noqa: E402
import flybrain_l1em_targets as T  # noqa: E402
import flybrain_model_eval as fme  # noqa: E402
import flybrain_wiring_features as fwf  # noqa: E402
from flybrain_l1em_adapter import L1emAdapterError, L1emErrorCode  # noqa: E402

# (celltype, annotation, number of left/right pairs)
_CLASSES = [
    ("sensory", "olfactory", 5),
    ("sensory", "gut", 5),
    ("sensory", "thermo-cold", 2),
    ("sensory", "thermo-warm", 2),
    ("ascending", "noci", 3),
    ("DN-VNC", "no official annotation", 4),
    ("DN-SEZ", "no official annotation", 3),
    ("RGN", "no official annotation", 3),
    ("PN", "olfactory 2nd_order PN", 5),
    ("KC", "KC", 5),
    ("pre-DN-VNC", "no official annotation", 5),
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_matrix(path: Path, skids, values) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([""] + list(skids))
        for skid, row in zip(skids, values):
            writer.writerow([skid] + [float(v) for v in row])


def write_real_models_fixture(tmp_path: Path, *, list_split_files=True, break_split_sum=False):
    """Synthetic l1em snapshot with every io class, sensory modalities and the aa/ad/da/dd split."""
    root = tmp_path / "snapshots" / "l1em" / "catmaid_l1em"
    files = root / "metadata" / "files"
    files.mkdir(parents=True)
    rows, neurons = [], []
    skid = 1000
    for c_index, (celltype, annotation, pairs) in enumerate(_CLASSES):
        for p in range(pairs):
            left, right = skid, skid + 1
            skid += 2
            rows.append([left, right, celltype, annotation, str(c_index)])
            neurons += [(left, c_index), (right, c_index)]
    skids = [s for s, _ in neurons] + [9001]  # one unannotated neuron
    cls = np.asarray([c for _, c in neurons] + [len(_CLASSES)])
    rng = np.random.RandomState(0)
    n = len(skids)
    # class-structured Poisson wiring so the target is learnable
    rate = rng.gamma(1.0, 1.0, size=(len(_CLASSES) + 1, len(_CLASSES) + 1))
    base = rng.poisson(rate[cls][:, cls] * 2.0) + (rng.rand(n, n) < 0.3) * 3
    np.fill_diagonal(base, 0)
    split = {t: np.zeros((n, n), dtype=np.int64) for t in l1s.L1EM_EDGE_TYPES}
    assign = rng.randint(0, 4, size=(n, n))
    for k, t in enumerate(l1s.L1EM_EDGE_TYPES):
        split[t][assign == k] = base[assign == k]
    all_all = sum(split.values())
    if break_split_sum:
        split["dd"][0, 1] += 1
    _write_matrix(files / "all-all_connectivity_matrix.csv", skids, all_all)
    for t in l1s.L1EM_EDGE_TYPES:
        _write_matrix(files / f"{t}_connectivity_matrix.csv", skids, split[t])
    with (files / "Supplementary_Data_S2.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["left_id", "right_id", "celltype", "additional_annotations", "level_7_cluster"])
        writer.writerows(rows)
    for name, cols in (("inputs.csv", ("axon_input", "dendrite_input")), ("outputs.csv", ("axon_output", "dendrite_output"))):
        with (files / name).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["", *cols])
            for s in skids:
                writer.writerow([s, float(s % 5 + 1), float(s % 3 + 1)])
    rels = sorted(l1em.L1EM_REQUIRED_FILES.values())
    if list_split_files:
        rels += sorted(l1s.L1EM_EDGE_TYPE_FILES.values())
    entries = [{"relative_path": r, "sha256": _sha(root / r), "size_bytes": (root / r).stat().st_size} for r in rels]
    now = datetime.now(timezone.utc)
    manifest = {
        "schema_version": "fbh-manifest/v1",
        "manifest_id": "fbh-l1em-real-models-fixture",
        "generated_at": now.isoformat(),
        "storage_root_env": "LOCI_FLYBRAIN_STORAGE_ROOT",
        "artifact": {"kind": "dataset_snapshot", "relative_root": "snapshots/l1em/catmaid_l1em", "path_template": "x"},
        "dataset": {"symbol": "l1em", "version_id": "catmaid_l1em",
                    "source": {"system": "fixture", "access_method": "fixture", "uri": "fixture://l1em",
                               "retrieved_at": now.isoformat(), "license": {"spdx_id": "CC-BY-4.0"},
                               "citation": "Winding et al. 2023 Science doi:10.1126/science.add9330"}},
        "scope": {"sex": "mixed_or_unspecified", "stage": "larva_l1", "anatomy": "x", "evidence_family": "x",
                  "claim_tier": "x"},
        "integrity": {"manifest_sha256": "", "files": entries,
                      "verification": {"status": "verified", "verified_at": now.isoformat()}},
        "refresh": {"decision": "no_change", "checked_at": now.isoformat(),
                    "next_check_due": (now + timedelta(days=30)).isoformat()},
        "lineage": {"derived_from": [{"type": "fixture", "id": "x"}],
                    "pipeline": {"job_name": "x", "job_version": "x", "run_id": "x"}},
    }
    manifest["integrity"]["manifest_sha256"] = l1em.manifest_digest(manifest)
    (root / "manifest").mkdir()
    (root / "manifest" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "manifest" / "manifest.sha256").write_text(manifest["integrity"]["manifest_sha256"] + "\n", encoding="utf-8")
    return root


@pytest.fixture()
def fixture_ctx(tmp_path):
    root = write_real_models_fixture(tmp_path)
    return T.L1emContext.load(snapshot_root=str(root), cache_root=str(tmp_path / "cache"))


def _cfg(**kw):
    values = dict(split_seed="t", report_root=None, models=T.default_models(T.TARGET_IO_CLASS), full=False, n_threads=1)
    values.update(kw)
    return T.eval_config(**values)


# ----------------------------------------------------------------------------- matrices / features


def test_matrices_split_sums_to_all_all(fixture_ctx):
    m = fixture_ctx.matrices
    assert np.array_equal(sum(m.by_type.values()), m.all_all)
    assert len(m.skids) == len(fixture_ctx.nodes) + 1  # the unannotated neuron is in the matrix only


def test_split_file_not_in_manifest_fails_closed(tmp_path):
    root = write_real_models_fixture(tmp_path, list_split_files=False)
    snap = l1em.load_l1em_snapshot(snapshot_root=str(root))
    with pytest.raises(L1emAdapterError) as err:
        l1s.load_l1em_matrices(snap)
    assert err.value.code is L1emErrorCode.REQUIRED_FILE_MISSING


def test_split_not_summing_to_all_all_fails_closed(tmp_path):
    root = write_real_models_fixture(tmp_path, break_split_sum=True)
    snap = l1em.load_l1em_snapshot(snapshot_root=str(root))
    with pytest.raises(L1emAdapterError) as err:
        l1s.load_l1em_matrices(snap)
    assert err.value.code is L1emErrorCode.MATRIX_INVALID


def test_edge_table_cached_reused_and_never_in_snapshots(tmp_path, fixture_ctx):
    first = fixture_ctx.edges
    mtime = os.stat(first.path).st_mtime_ns
    again = l1s.export_l1em_edge_table(fixture_ctx.matrices, cache_root=str(tmp_path / "cache"))
    assert again.path == first.path and os.stat(again.path).st_mtime_ns == mtime
    assert first.provenance["manifest_sha256"] == fixture_ctx.snapshot.manifest_sha256
    import pyarrow.parquet as pq

    table = pq.read_table(first.path).to_pandas()
    assert int(table["weight"].sum()) == int(fixture_ctx.matrices.all_all.sum())
    with pytest.raises(ValueError, match="snapshots"):
        l1s.export_l1em_edge_table(fixture_ctx.matrices, cache_root=str(tmp_path / "snapshots" / "x"))


def test_edge_type_fractions_sum_to_one(fixture_ctx):
    f = l1s.edge_type_features(fixture_ctx.matrices, fixture_ctx.nodes["skid"].tolist())
    out = f[[f"etype_out__{t}" for t in l1s.L1EM_EDGE_TYPES]].sum(axis=1).dropna()
    inn = f[[f"etype_in__{t}" for t in l1s.L1EM_EDGE_TYPES]].sum(axis=1).dropna()
    assert np.allclose(out, 1.0) and np.allclose(inn, 1.0)


def test_spectral_embedding_deterministic_and_label_free(fixture_ctx):
    skids = fixture_ctx.nodes["skid"].tolist()
    a = l1s.spectral_embedding_features(fixture_ctx.matrices, skids, dim=4)
    b = l1s.spectral_embedding_features(fixture_ctx.matrices, skids, dim=4)
    assert a.equals(b)
    assert list(a.columns) == [f"ase__out_{j:02d}" for j in range(4)] + [f"ase__in_{j:02d}" for j in range(4)]


def test_structured_features_contain_no_ids_or_annotations(fixture_ctx):
    cols = list(fixture_ctx.structured.columns)
    assert not fwf.excluded_feature_names(cols, T.TARGET_IO_CLASS)
    assert not any(k in c for c in cols for k in ("skid", "celltype", "annotation", "cluster", "total"))


# ----------------------------------------------------------------------------- targets


def test_io_class_mapping_and_rejected_targets():
    assert T.io_class("sensory") == "sensory" and T.io_class("DN-VNC") == "dn_vnc"
    assert T.io_class("pre-DN-VNC") == T.IO_INTERNEURON and T.io_class("KC") == T.IO_INTERNEURON
    for name in ("celltype", "level_7_cluster", "ascending_modality"):
        assert name in T.REJECTED_TARGETS
    assert not set(T.IO_CLASS_OF_CELLTYPE) & T.CONNECTIVITY_DEFINED_CELLTYPES


def test_exclusions_registered_fail_closed():
    for target in (T.TARGET_IO_CLASS, T.TARGET_SENSORY_MODALITY):
        for bad in ("celltype", "annotation", "level_7_cluster", "l1_class__celltype", "skid"):
            with pytest.raises(fwf.LabelLeakageError):
                fwf.assert_features_allowed([bad], target)
    fwf.assert_features_allowed(["degree__in_weight_total", "degree__out_mean_pair_weight"], T.TARGET_IO_CLASS)
    with pytest.raises(fwf.LabelLeakageError):
        fwf.assert_features_allowed(["degree__in_weight_total"], T.TARGET_CONNECTIVITY)


def test_sensory_modality_merges_thermo_and_drops_small(fixture_ctx):
    frame, notes = T.target_frame(T.TARGET_SENSORY_MODALITY, fixture_ctx, min_class_neurons=5)
    assert set(frame["label"]) == {"olfactory", "gut", "thermo"}
    assert (frame["celltype"] == "sensory").all()
    frame, notes = T.target_frame(T.TARGET_SENSORY_MODALITY, fixture_ctx, min_class_neurons=9)
    assert notes["dropped_small_classes"] == {"thermo": 8}


def test_io_class_features_masked_for_heldout_split(fixture_ctx):
    cfg = _cfg()
    data = T.build_eval_dataset(T.TARGET_IO_CLASS, fixture_ctx, cfg)
    plan = fme.plan_grouped_split(list(data.sample_ids), list(data.group_values), data.group_keys, cfg)
    assert data.notes["masked_split_ids_sha256"] == fme.split_ids_sha256(plan)
    assert data.notes["masked_nodes"] == len(plan["val"]) + len(plan["test"])
    # Permuting the labels of held-out neurons must not change ANY sample's features.
    heldout = {int(s.split("-")[-1]) for s in plan["val"] + plan["test"]}
    nodes = fixture_ctx.nodes.copy()
    rows = nodes["skid"].isin(heldout)
    nodes.loc[rows, "io_class"] = np.where(nodes.loc[rows, "io_class"] == "sensory", "rgn", "sensory")
    other = T.L1emContext(fixture_ctx.snapshot, fixture_ctx.matrices, nodes, fixture_ctx.edges,
                          fixture_ctx.structured, fixture_ctx.cache_root, fixture_ctx.snapshot_root, None)
    permuted = T.build_eval_dataset(T.TARGET_IO_CLASS, other, cfg)
    pd.testing.assert_frame_equal(data.features, permuted.features)


def test_unmasked_io_class_would_leak(fixture_ctx):
    """Control for the test above: without the mask, held-out labels do reach features."""
    nodes = fixture_ctx.nodes
    base = l1s.l1em_wiring_features(objective=T.TARGET_IO_CLASS, edges=fixture_ctx.edges, nodes=nodes,
                                    category_column="io_class", cache_root=None)
    flipped = nodes.copy()
    flipped["io_class"] = np.where(flipped["io_class"] == "sensory", "rgn", "sensory")
    moved = l1s.l1em_wiring_features(objective=T.TARGET_IO_CLASS, edges=fixture_ctx.edges, nodes=flipped,
                                     category_column="io_class", cache_root=None)
    assert not base.frame.equals(moved.frame)


def test_connectivity_tier_has_no_degree_or_embedding(fixture_ctx):
    cfg = _cfg(models=T.default_models(T.TARGET_CONNECTIVITY))
    data = T.build_eval_dataset(T.TARGET_CONNECTIVITY, fixture_ctx, cfg, min_total_synapses=1)
    cols = list(data.features.columns)
    assert cols and not any(c.startswith(("degree__", "ase__")) for c in cols)
    assert data.notes["proxy_audit"]["max_abs_rho"] < T.PROXY_RHO_LIMIT


def test_proxy_audit_fails_closed_on_monotone_proxy():
    totals = np.arange(50, dtype=float)
    frame = pd.DataFrame({"recip__ok": np.sin(totals), "out_comp__proxy": np.log1p(totals)})
    with pytest.raises(fwf.LabelLeakageError, match="out_comp__proxy"):
        T.proxy_audit(frame, totals)


def test_binned_text_uses_train_edges_only():
    frame = pd.DataFrame({"etype_out__aa": np.linspace(0, 1, 20), "recip__partner_frac": [np.nan] + [0.5] * 19})
    train = np.arange(10)
    text = T.binned_text(frame, train, n_bins=4)
    changed = frame.copy()
    changed.loc[10:, "etype_out__aa"] = 100.0
    assert T.binned_text(changed, train, n_bins=4)[:10] == text[:10]
    assert text[0].split()[:2] == ["etype_out__aa", "etype_out__aa_q0"]
    assert "recip__partner_frac_qna" in text[0]


def test_end_to_end_io_class_eval(fixture_ctx):
    models = (fme.ModelSpec("nb", ({"alpha": 1.0},), "temperature", "nb"),
              fme.ModelSpec("logreg", ({"C": 0.3},), "temperature", "logreg"))
    cfg = fme.EvalConfig(models=models, split_seed="t", cv_folds=2, n_bootstrap=50, report_root=None,
                         random_split_control=False, ablation=True, n_threads=1, save_models=False)
    data = T.build_eval_dataset(T.TARGET_IO_CLASS, fixture_ctx, cfg)
    report = fme.run_evaluation(data, cfg)
    rows = report["summary"]
    assert {r["model"] for r in rows} == {"nb", "logreg"}
    assert all(r["dataset"] == "l1em" and r["target"] == T.TARGET_IO_CLASS for r in rows)
    assert report["leakage_check"]["pass"]
    assert "drop_one_family" in report["ablation"]


def test_repeat_stats_aggregates():
    rows = [[{"model": "m", "model_acc": 0.8, "best_trivial": 0.7, "majority": 0.6, "macro_f1": 0.5, "ece": 0.1,
              "shuffle_acc": 0.6, "gate": "pass"}],
            [{"model": "m", "model_acc": 0.6, "best_trivial": 0.7, "majority": 0.6, "macro_f1": 0.4, "ece": 0.2,
              "shuffle_acc": 0.6, "gate": "fail"}]]
    stats = T.repeat_stats(rows)["m"]
    assert stats["n_seeds"] == 2 and stats["gate_passes"] == 1
    assert stats["gain_over_best_trivial"]["min"] == pytest.approx(-0.1)


def test_granularity_probe_scores_on_val_only(fixture_ctx):
    cfg = _cfg(models=T.default_models(T.TARGET_CONNECTIVITY))
    data = T.build_eval_dataset(T.TARGET_CONNECTIVITY, fixture_ctx, cfg, min_total_synapses=1)
    probe = T.granularity_probe(data, cfg, params={"max_iter": 20, "min_samples_leaf": 2})
    assert probe["evaluated_on"] == "val"
    for key in ("sparsity_counts_only", "sparsity_plus_min_positive_fraction", "all_features", "majority_val"):
        assert 0.0 <= probe[key] <= 1.0
    assert set(probe["only_family"]) == set(probe["drop_family"])
