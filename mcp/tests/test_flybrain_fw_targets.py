"""Offline tests for flybrain_fw_targets (synthetic fw snapshot under tmp_path; never touches /mnt/f)."""

import dataclasses
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_brain_cluster_fw_samples as fw_samples  # noqa: E402
import flybrain_fw_targets as ft  # noqa: E402
import flybrain_model_eval as fme  # noqa: E402
import flybrain_wiring_features as fwf  # noqa: E402

CLASSES = ("central", "optic", "sensory")
NEUROPILS = {"central": ("SMP_L", "SLP_L"), "optic": ("ME_L", "LO_L"), "sensory": ("AL_L", "GNG")}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _annotations(n_types=30, per_type=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    rid = 720575940600000000
    for t in range(n_types):
        sc = CLASSES[t % 3]
        for j in range(per_type):
            rid += 1
            rows.append({
                "supervoxel_id": rid + 5, "root_id": rid, "flow": "afferent" if sc == "sensory" else "intrinsic",
                "super_class": sc, "cell_class": {"central": "CX", "optic": "ME>LO", "sensory": "olfactory"}[sc],
                "cell_sub_class": "", "cell_type": f"T{t}", "hemibrain_type": f"HB{t // 2}" if t % 5 else "",
                "ito_lee_hemilineage": f"HL{t % 7}" if sc == "central" else "",
                "top_nt": ["acetylcholine", "gaba", "glutamate"][int(rng.integers(0, 3))],
                "known_nt": ["acetylcholine", "gaba", "glutamate, gaba", ""][t % 4], "known_nt_source": "",
                "side": "left" if j % 2 else "right", "status": "outlier_seg" if (t == 0 and j == 0) else "",
            })
    return pd.DataFrame(rows)


def _edges(ann, seed=0):
    rng = np.random.default_rng(seed)
    ids = ann["root_id"].to_numpy()
    cls = ann["super_class"].to_numpy()
    pre, post, npl, w = [], [], [], []
    for i in range(len(ids)):
        for _ in range(8):
            same = rng.random() < 0.8
            pool = np.flatnonzero((cls == cls[i]) == same)
            j = int(rng.choice(pool))
            if j == i:
                continue
            for neuropil in NEUROPILS[cls[i]][: 1 + int(rng.integers(0, 2))]:
                pre.append(int(ids[i]))
                post.append(int(ids[j]))
                npl.append(neuropil)
                w.append(int(rng.integers(5, 40)))
    frame = pd.DataFrame({"pre_pt_root_id": pre, "post_pt_root_id": post, "neuropil": npl, "syn_count": w})
    frame = frame.groupby(["pre_pt_root_id", "post_pt_root_id", "neuropil"], as_index=False)["syn_count"].sum()
    for col in ("gaba_avg", "ach_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg"):
        frame[col] = 0.1
    return frame


@pytest.fixture()
def fw_root(tmp_path):
    root = tmp_path / "flybrain"
    snap = root / ft.FW_SNAPSHOT_RELATIVE_ROOT
    (snap / "manifest").mkdir(parents=True)
    ann = _annotations()
    edges = _edges(ann)
    files = {
        ft.ROLE_ANNOTATIONS: snap / ft.FW_PRODUCT_PATHS[ft.ROLE_ANNOTATIONS],
        ft.ROLE_CONNECTIONS: snap / ft.FW_PRODUCT_PATHS[ft.ROLE_CONNECTIONS],
        ft.ROLE_PRE_COUNTS: snap / ft.FW_PRODUCT_PATHS[ft.ROLE_PRE_COUNTS],
    }
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    ann.to_csv(files[ft.ROLE_ANNOTATIONS], sep="\t", index=False)
    feather.write_feather(pa.Table.from_pandas(edges, preserve_index=False), str(files[ft.ROLE_CONNECTIONS]))
    pre_counts = edges.groupby(["pre_pt_root_id", "neuropil"], as_index=False)["syn_count"].sum().rename(
        columns={"syn_count": "count"})
    feather.write_feather(pa.Table.from_pandas(pre_counts, preserve_index=False), str(files[ft.ROLE_PRE_COUNTS]))
    manifest = {
        "schema_version": "fbh-manifest/v1",
        "dataset": {"symbol": "fw", "version_id": "flywire783"},
        "refresh": {"decision": "extended", "next_check_due": "2999-01-01T00:00:00Z"},
        "integrity": {
            "manifest_sha256": "",
            "verification": {"status": "verified", "verified_at": "2026-09-24T00:00:00Z"},
            "files": [{"role": role, "relative_path": ft.FW_PRODUCT_PATHS[role], "sha256": _sha(path),
                       "size_bytes": path.stat().st_size} for role, path in files.items()],
        },
    }
    manifest["integrity"]["manifest_sha256"] = ft._manifest_digest(manifest)
    (snap / ft.FW_MANIFEST_RELATIVE_PATH).write_text(json.dumps(manifest), encoding="utf-8")
    (snap / ft.FW_MANIFEST_SIDECAR_RELATIVE_PATH).write_text(manifest["integrity"]["manifest_sha256"] + "\n")
    return root


def _snapshot_listing(root):
    snap = root / "snapshots"
    return sorted(str(p.relative_to(snap)) for p in snap.rglob("*"))


@pytest.fixture()
def small_targets(monkeypatch):
    for name in list(ft.FW_TARGETS):
        monkeypatch.setitem(ft.FW_TARGETS, name, dataclasses.replace(ft.FW_TARGETS[name], min_class_count=5,
                                                                        max_per_class=1000))


def _config(tmp_path, **kw):
    base = dict(models=(fme.ModelSpec("nb", ({"alpha": 1.0},), "temperature"),
                        fme.ModelSpec("logreg", ({"C": 1.0},), "temperature")),
                cv_folds=2, n_bootstrap=50, n_threads=2, report_root=str(tmp_path / "reports"), ablation=False,
                random_split_control=False)
    base.update(kw)
    return fme.EvalConfig(**base)


# --------------------------------------------------------------------------- snapshot


def test_open_fw_snapshot_verifies_and_never_writes_into_snapshot(fw_root, tmp_path):
    before = _snapshot_listing(fw_root)
    snap = ft.open_fw_snapshot(fw_root, stamp_dir=tmp_path / "stamps")
    assert set(snap.hash_verification.values()) == {"hashed"}
    again = ft.open_fw_snapshot(fw_root, stamp_dir=tmp_path / "stamps")
    assert set(again.hash_verification.values()) == {"stamp"}
    assert _snapshot_listing(fw_root) == before
    assert snap.provenance(ft.ROLE_CONNECTIONS)["manifest_sha256"] == snap.manifest_sha256


def test_open_fw_snapshot_fails_closed(fw_root, tmp_path):
    snap_dir = fw_root / ft.FW_SNAPSHOT_RELATIVE_ROOT
    with pytest.raises(ft.FwSnapshotError, match="snapshots"):
        ft.open_fw_snapshot(fw_root, stamp_dir=snap_dir / "manifest" / "stamps")
    with pytest.raises(ft.FwSnapshotError, match="unknown"):
        ft.open_fw_snapshot(fw_root, required_roles=["nope"], stamp_dir=tmp_path / "s")
    # same-size content tamper -> sha mismatch
    target = snap_dir / ft.FW_PRODUCT_PATHS[ft.ROLE_ANNOTATIONS]
    raw = bytearray(target.read_bytes())
    raw[-2] = ord("X") if raw[-2] != ord("X") else ord("Y")
    target.write_bytes(bytes(raw))
    with pytest.raises(ft.FwSnapshotError, match="sha256 mismatch"):
        ft.open_fw_snapshot(fw_root, required_roles=[ft.ROLE_ANNOTATIONS], stamp_dir=tmp_path / "s2")
    # sidecar disagreement
    (snap_dir / ft.FW_MANIFEST_SIDECAR_RELATIVE_PATH).write_text("0" * 64)
    with pytest.raises(ft.FwSnapshotError, match="sidecar"):
        ft.open_fw_snapshot(fw_root, stamp_dir=tmp_path / "s3")


def test_open_fw_snapshot_rejects_edited_manifest(fw_root, tmp_path):
    path = fw_root / ft.FW_SNAPSHOT_RELATIVE_ROOT / ft.FW_MANIFEST_RELATIVE_PATH
    manifest = json.loads(path.read_text())
    manifest["dataset"]["version_id"] = "flywire630"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ft.FwSnapshotError, match="self-hash"):
        ft.open_fw_snapshot(fw_root, stamp_dir=tmp_path / "s")


# --------------------------------------------------------------------------- labels


def test_parse_known_nt_keeps_single_classical_transmitter():
    assert ft.parse_known_nt("acetylcholine") == "acetylcholine"
    assert ft.parse_known_nt("gaba, nitric oxide, allatostatin-c") == "gaba"
    assert ft.parse_known_nt("glutamate, gaba") is None
    assert ft.parse_known_nt("acetylcholine, nitric oxide, dopamine") is None
    assert ft.parse_known_nt(None) is None and ft.parse_known_nt(float("nan")) is None


def test_connectivity_tier_labels_threshold():
    labels, thr = ft.connectivity_tier_labels(pd.Series([1, 2, 3, 4, 100]), 0.75)
    assert thr == 4.0
    assert labels.tolist() == ["baseline_connectivity"] * 3 + ["high_connectivity"] * 2


def test_select_target_rows_drops_outliers_and_caps_whole_components(fw_root, tmp_path):
    snap = ft.open_fw_snapshot(fw_root, stamp_dir=tmp_path / "s")
    ann = ft.load_fw_annotations(snap.path(ft.ROLE_ANNOTATIONS))
    spec = dataclasses.replace(ft.FW_TARGETS["super_class"], min_class_count=5, max_per_class=16)
    rows = ft.select_target_rows(ann, spec, {})
    assert rows["status"].isna().all()
    assert rows["label"].value_counts().max() <= 16
    assert set(rows["label"]) == {"intrinsic_central", "intrinsic_optic", "sensory"}
    # the cap takes components whole except possibly the last one per class
    for label, part in rows.groupby("label"):
        full = ft.select_target_rows(ann, dataclasses.replace(spec, max_per_class=10_000), {})
        full = full[full["label"] == label]
        comp_sizes = full.groupby("_component").size()
        kept = part.groupby("_component").size()
        assert sum(kept[c] < comp_sizes[c] for c in kept.index) <= 1
    capped = ft.select_target_rows(ann, dataclasses.replace(spec, max_per_component=2), {})
    assert capped.groupby("_component").size().max() <= 2
    assert capped["_component"].nunique() == rows["_component"].nunique() or len(capped) < len(rows)
    nt = ft.select_target_rows(ann, dataclasses.replace(ft.FW_TARGETS["nt_ground_truth"], min_class_count=1), {})
    assert set(nt["label"]) <= {"acetylcholine", "gaba"}  # "glutamate, gaba" is ambiguous -> dropped


# --------------------------------------------------------------------------- features + eval


def test_build_eval_dataset_masks_heldout_and_applies_exclusions(fw_root, tmp_path, small_targets):
    snap = ft.open_fw_snapshot(fw_root, stamp_dir=tmp_path / "s")
    ann = ft.load_fw_annotations(snap.path(ft.ROLE_ANNOTATIONS))
    config = _config(tmp_path)
    build = ft.build_fw_eval_dataset("super_class_no_neuropil", snapshot=snap, annotations=ann, config=config,
                                     cache_root=tmp_path / "cache")
    cols = list(build.data.features.columns)
    assert cols and not [c for c in cols if "np__" in c]
    fwf.assert_features_allowed(cols, "super_class_no_neuropil")
    assert build.data.notes["masked_split_ids_sha256"] == fme.split_ids_sha256(build.plan)
    # every held-out sample AND every unsampled node sharing its type / lineage / family is masked
    held = list(build.plan["val"]) + list(build.plan["test"])
    expected = ft.held_out_mask_ids(ann, build.rows, held, ft.GROUP_COLUMNS)
    assert set(held) <= set(expected)
    assert build.features.meta["masked_category_nodes"] == len(expected) == build.data.notes["masked_category_nodes"]
    outlier = str(ann.loc[ann["status"].notna(), "root_id"].iloc[0])  # type T0, never a sample
    assert outlier not in set(build.rows["root_id"].astype(str))
    if "T0" in set(build.rows.loc[build.rows["root_id"].astype(str).isin(held), "cell_type"]):
        assert outlier in set(expected)
    assert build.data.notes["label_provenance"] == "curated_morphology"
    # the with-neuropil variant keeps neuropil fractions (same split -> same masked cache)
    with_np = ft.build_fw_eval_dataset("super_class", snapshot=snap, annotations=ann, config=config,
                                       cache_root=tmp_path / "cache")
    assert any(c.startswith("out_np__") for c in with_np.data.features.columns)
    assert with_np.features.fingerprint == build.features.fingerprint
    # connectivity tier: no degree / count proxies survive
    totals = ft.total_presynapse_counts(snap.path(ft.ROLE_PRE_COUNTS))
    conn = ft.build_fw_eval_dataset("connectivity_tier", snapshot=snap, annotations=ann, config=config,
                                    presynapse_totals=totals, cache_root=tmp_path / "cache")
    assert not [c for c in conn.data.features.columns if c.startswith("degree__") or "n_neuropils" in c]
    assert "masked_split_ids_sha256" not in conn.data.notes
    assert conn.data.notes["connectivity_threshold_presynapses"] > 0
    # the text view carries only allowed keys, as name-prefixed bin tokens
    text = build.data.text[0].split()
    assert all(tok.startswith(key + "__") for key, tok in zip(text[::2], text[1::2]))
    # nothing was written into the snapshot
    assert not list((fw_root / "snapshots").rglob("*.parquet"))


def test_run_evaluation_end_to_end_on_synthetic_fw(fw_root, tmp_path, small_targets):
    snap = ft.open_fw_snapshot(fw_root, stamp_dir=tmp_path / "s")
    ann = ft.load_fw_annotations(snap.path(ft.ROLE_ANNOTATIONS))
    config = _config(tmp_path)
    build = ft.build_fw_eval_dataset("super_class", snapshot=snap, annotations=ann, config=config,
                                     cache_root=tmp_path / "cache")
    report = fme.run_evaluation(build.data, config)
    rows = {r["model"]: r for r in report["summary"]}
    assert set(rows) == {"nb", "logreg"}
    assert all(r["dataset"] == "fw" and r["target"] == "super_class" for r in rows.values())
    assert report["notes"]["partner_category_masked_for_val_test"] is True
    assert (tmp_path / "reports" / "fw" / "super_class" / "report.json").is_file()
    # a mask computed for another split is refused
    other = dataclasses.replace(config, split_seed="another-seed")
    with pytest.raises(ValueError, match="masked"):
        fme.run_evaluation(build.data, other)


def test_binned_text_uses_train_rows_only():
    frame = pd.DataFrame({"a__x": [0.0, 1.0, 2.0, 3.0, 100.0, np.nan]})
    text = ft.binned_text(frame, np.asarray([0, 1, 2, 3]), n_bins=4)
    assert text[4] == "a__x a__x__q3"  # beyond the train range -> top bin, edges not refitted on it
    assert text[5] == "a__x a__x__na"
    assert text[0] == "a__x a__x__q0"


def test_bridge_table_and_hemibrain_groups(fw_root, tmp_path):
    assert ft.hemibrain_type_group("M_adPNm5,M_adPNm4") == "M_adPNm4|M_adPNm5"
    assert ft.hemibrain_type_group("(CL141)") == "CL141"
    assert ft.hemibrain_type_group(None) is None
    snap = ft.open_fw_snapshot(fw_root, stamp_dir=tmp_path / "s")
    ann = ft.load_fw_annotations(snap.path(ft.ROLE_ANNOTATIONS))
    path = ft.write_bridge_table(ann, snapshot=snap, features_fingerprint="f" * 64, features_cache=None,
                                 cache_root=tmp_path / "cache")
    bridge = pd.read_parquet(path)
    assert len(bridge) == len(ann) and "hemibrain_type_group" in bridge.columns
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["parquet_sha256"] == _sha(path)
    with pytest.raises(ValueError, match="snapshots"):
        ft.write_bridge_table(ann, snapshot=snap, features_fingerprint="f" * 64, features_cache=None,
                              cache_root=fw_root / "snapshots" / "x")


def test_structured_samples_keep_legacy_builder_untouched(fw_root, tmp_path, small_targets):
    payload = fw_samples.build_fw_structured_samples("flow", storage_root=fw_root, cache_root=tmp_path / "cache",
                                                     config=_config(tmp_path))
    sample = payload["samples"][0]
    assert sample["features"] and set(payload["metadata"]["group_keys"]) == set(ft.GROUP_COLUMNS)
    assert payload["metadata"]["notes"]["masked_split_ids_sha256"]
    data = fme.EvalDataset.from_samples(payload["samples"], dataset="fw", target="flow",
                                        group_keys=payload["metadata"]["group_keys"])
    assert len(data.sample_ids) == len(payload["samples"])
    assert fw_samples.FW_SUPPORTED_OBJECTIVES  # legacy objective allow-list still present


# --------------------------------------------------------------------------- grouping fix (d), provenance (a)


def test_type_family_groups_sister_types():
    assert ft.type_family("T4a") == ft.type_family("T5c") == "fam:T4/T5"
    assert ft.type_family("KCab") == ft.type_family("KCapbp-ap2") == ft.type_family("KCg-m") == "fam:KC"
    assert ft.type_family("Dm3a") == ft.type_family("Dm3c") == "fam:Dm3"
    assert ft.type_family("LHPV2a1_a") == ft.type_family("LHPV2a1_b") == "fam:LHPV2a1"
    assert ft.type_family("(M_lPNm12,M_lPNm13)a") == ft.type_family("(M_lPNm12,M_lPNm13)b")
    assert ft.type_family("C2") != ft.type_family("C3")
    assert ft.type_family("Mi1") == "fam:Mi1" and ft.type_family("ORN_VC5") == "fam:ORN_VC5"
    assert ft.type_family(None) is None and ft.type_family("na") is None


def test_sister_types_never_straddle_the_split():
    """The first nt_ground_truth run had T5c in val with T4a-d / T5a,b,d in train (val 0.98 vs test 0.10)."""
    rows = []
    rid = 1
    for t in ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d", "KCab", "KCg-m", "KCapbp-ap2"):
        for _ in range(3):
            rows.append({"root_id": rid, "cell_type": t, "hemibrain_type": None, "hemilineage": None,
                         "type_family": ft.type_family(t)})
            rid += 1
    for i in range(30):
        rows.append({"root_id": rid, "cell_type": f"X{i}", "hemibrain_type": None, "hemilineage": None,
                     "type_family": ft.type_family(f"X{i}")})
        rid += 1
    frame = pd.DataFrame(rows)
    comps = ft.component_ids(frame)
    by_type = dict(zip(frame["cell_type"], comps))
    assert len({by_type[t] for t in ("T4a", "T4d", "T5c")}) == 1
    assert len({by_type[t] for t in ("KCab", "KCg-m", "KCapbp-ap2")}) == 1
    plan = fme.plan_grouped_split(frame["root_id"].astype(str).tolist(),
                                  frame[list(ft.GROUP_COLUMNS)].to_dict(orient="records"), ft.GROUP_COLUMNS,
                                  fme.EvalConfig(report_root=None))
    where = {i: s for s in plan for i in plan[s]}
    t45 = {where[str(r)] for r, t in zip(frame["root_id"], frame["cell_type"]) if t.startswith(("T4", "T5"))}
    assert len(t45) == 1


def test_load_annotations_drops_catch_all_lineages(tmp_path):
    ann = _annotations(n_types=6, per_type=2)
    ann.loc[0, "ito_lee_hemilineage"] = "putative_primary"
    ann.loc[1, "ito_lee_hemilineage"] = "VLPl1_or_VLPl5"
    path = tmp_path / "ann.tsv"
    ann.to_csv(path, sep="\t", index=False)
    out = ft.load_fw_annotations(path).set_index("root_id")
    assert pd.isna(out.loc[int(ann.loc[0, "root_id"]), "hemilineage"])
    assert out.loc[int(ann.loc[0, "root_id"]), "hemilineage_raw"] == "putative_primary"
    assert pd.isna(out.loc[int(ann.loc[1, "root_id"]), "hemilineage"])
    assert out["type_family"].notna().all()


def test_every_fw_target_declares_provenance():
    import flybrain_target_registry as ftr

    for name, spec in ft.FW_TARGETS.items():
        assert ftr.target_provenance("fw", name).provenance.value == spec.label_provenance
    assert ft.FW_TARGETS["nt_ground_truth"].ground_truth and not ft.FW_TARGETS["neurotransmitter_dominance"].ground_truth
    assert ft.FW_TARGETS["hemilineage"].group_columns == ft.HEMILINEAGE_GROUP_COLUMNS
    assert "hemilineage" not in ft.HEMILINEAGE_GROUP_COLUMNS


def test_hemilineage_target_masks_partner_lineages(fw_root, tmp_path, monkeypatch):
    monkeypatch.setitem(ft.FW_TARGETS, "hemilineage", dataclasses.replace(
        ft.FW_TARGETS["hemilineage"], min_class_count=2, min_class_components=1))
    snap = ft.open_fw_snapshot(fw_root, stamp_dir=tmp_path / "s")
    ann = ft.load_fw_annotations(snap.path(ft.ROLE_ANNOTATIONS))
    build = ft.build_fw_eval_dataset("hemilineage", snapshot=snap, annotations=ann, config=_config(tmp_path),
                                     cache_root=tmp_path / "cache")
    assert set(build.data.labels) <= {f"HL{i}" for i in range(7)}
    assert build.data.group_keys == ft.HEMILINEAGE_GROUP_COLUMNS
    assert build.data.notes["masked_split_ids_sha256"] == fme.split_ids_sha256(build.plan)
    assert not [c for c in build.data.features.columns if "lineage" in c.lower()]
    assert build.data.notes["label_provenance"] == "curated_morphology"


def _fake_nt_repo(tmp_path, monkeypatch, rows):
    import hashlib

    import flybrain_nt_ground_truth as ntgt

    commit = "a" * 40
    root = tmp_path / "ntgt"
    repo = root / f"repo-{commit[:8]}"
    (repo / ".git").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text(commit)
    cols = ["species", "region", "hemilineage", "cell_type", "neurotransmitter_verified_source",
            "neurotransmitter_verified_evidence", "neurotransmitter_verified_confidence", *ntgt.NT_COLUMNS]
    records = []
    for cell_type, nt in rows:
        rec = {c: 0 for c in ntgt.NT_COLUMNS}
        rec.update({"species": ntgt.ADULT_SPECIES, "region": "central_brain", "hemilineage": "x",
                    "cell_type": cell_type, "neurotransmitter_verified_source": "S",
                    "neurotransmitter_verified_evidence": "immuno", "neurotransmitter_verified_confidence": 5,
                    nt: 1})
        records.append(rec)
    files = {ntgt.GT_DATA: pd.DataFrame(records, columns=cols).to_csv(index=False),
             ntgt.FW_CNN_GT[0]: "cell_type\nT1\n", ntgt.FW_CNN_GT[1]: "cell_type\n"}
    pins = {}
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
        pins[rel] = hashlib.sha256((repo / rel).read_bytes()).hexdigest()
    (root / ntgt.SOURCE_FILE).write_text(json.dumps({"commit": commit}))
    monkeypatch.setattr(ntgt, "NT_GT_COMMIT", commit)
    monkeypatch.setattr(ntgt, "PINNED_SHA256", pins)
    return root


def test_nt_literature_target_on_synthetic_fw(fw_root, tmp_path, monkeypatch):
    nts = ["acetylcholine", "gaba", "glutamate"]
    nt_root = _fake_nt_repo(tmp_path, monkeypatch, [(f"T{t}", nts[t % 3]) for t in range(1, 30)])
    for name in ("nt_literature", "nt_literature_all"):
        monkeypatch.setitem(ft.FW_TARGETS, name, dataclasses.replace(
            ft.FW_TARGETS[name], min_class_count=2, min_class_components=1))
    snap = ft.open_fw_snapshot(fw_root, stamp_dir=tmp_path / "s")
    ann = ft.load_fw_annotations(snap.path(ft.ROLE_ANNOTATIONS))
    full = ft.build_fw_eval_dataset("nt_literature_all", snapshot=snap, annotations=ann, config=_config(tmp_path),
                                    cache_root=tmp_path / "cache", nt_root=nt_root)
    strict = ft.build_fw_eval_dataset("nt_literature", snapshot=snap, annotations=ann, config=_config(tmp_path),
                                      cache_root=tmp_path / "cache", nt_root=nt_root)
    assert set(full.data.labels) == set(nts)
    assert "T1" in set(full.rows["cell_type"]) and "T1" not in set(strict.rows["cell_type"])  # CNN training type
    cov = strict.data.notes["nt_literature_coverage"]
    assert cov["cnn_training"]["matched_types_in_cnn_training"] == 1 and cov["types_used"] == 28
    assert full.data.notes["label_provenance"] == "measured"
    assert not [c for c in full.data.features.columns if "lineage" in c]
    fwf.assert_features_allowed(list(full.data.features.columns), "nt_literature_all")
