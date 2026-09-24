import dataclasses
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import flybrain_brain_cluster_mc_samples as mc_samples  # noqa: E402
import flybrain_brain_cluster_samples as samples  # noqa: E402
import flybrain_mc_adapter as mc  # noqa: E402
from flybrain_dataset_registry import DatasetErrorCode, DatasetRegistryError  # noqa: E402
from flybrain_mc_adapter import McAdapterError, McAdapterErrorCode  # noqa: E402
from test_flybrain_mc_adapter import (  # noqa: E402
    _EDGES,
    _ROI_INFO,
    bid,
    build_snapshot,
    write_dataset_meta,
    write_edgelist,
    write_meta,
    write_neurons,
    write_nt,
)

# Importing the module registers the mc builders globally. Undo that at import
# time so other test modules see an empty slot; each test here re-registers.
mc_samples.unregister_mc_sample_builders()

CONN = "connectivity_tier"
NT = "neurotransmitter_dominance"
ROI = "region_specialization_tier"
ALL = (CONN, NT, ROI)


@pytest.fixture(autouse=True)
def _mc_registered():
    mc_samples.register_mc_sample_builders(replace=True)
    yield
    mc_samples.unregister_mc_sample_builders()


def _write_all(tmp_path):
    paths = {
        "meta_path": tmp_path / "meta.feather",
        "nt_path": tmp_path / "nt.feather",
        "edgelist_path": tmp_path / "edges.feather",
        "neuron_path": tmp_path / "neurons.feather",
        "dataset_meta_path": tmp_path / "Neuprint_Meta.csv",
    }
    if not paths["meta_path"].exists():
        write_meta(paths["meta_path"])
        write_nt(paths["nt_path"])
        write_edgelist(paths["edgelist_path"])
        write_neurons(paths["neuron_path"])
        write_dataset_meta(paths["dataset_meta_path"])
    return paths


def _explicit_config(tmp_path, objective=CONN, **overrides):
    base = dict(
        objective=objective,
        **_write_all(tmp_path),
        max_samples=50,
        min_total_count=1,
        min_primary_synapses=100,
        min_region_samples=1,
        high_connectivity_quantile=0.5,
        max_regions=20,
    )
    base.update(overrides)
    return mc_samples.McSampleBuildConfig(**base)


def _build(tmp_path, objective=CONN, **overrides):
    return samples.build_training_samples("mc", objective, _explicit_config(tmp_path, objective, **overrides),
                                          allow_planned=True)


def _kv(sample):
    tokens = sample["input_text"].split()
    return dict(zip(tokens[0::2], tokens[1::2]))


def _by_root(payload):
    return {s["metadata"]["root_id"]: s for s in payload["samples"]}


def test_registration_covers_supported_objectives():
    assert samples.registered_builders()["mc"] == tuple(sorted(ALL))


def test_builder_module_name_matches_dispatcher_template():
    assert samples.BUILDER_MODULE_TEMPLATE.format(symbol="mc") == mc_samples.__name__


def test_planned_dataset_requires_opt_in(tmp_path):
    with pytest.raises(DatasetRegistryError) as exc:
        samples.build_training_samples("mc", CONN, _explicit_config(tmp_path))
    assert exc.value.code is DatasetErrorCode.DATASET_NOT_ACTIVE


def test_candidate_filters(tmp_path):
    payload = _build(tmp_path)
    stats = payload["metadata"]["filter_stats"]
    roots = set(_by_root(payload))
    assert stats["dropped_untyped"] == 1 and str(bid(11)) not in roots
    assert stats["dropped_status"] == 1 and str(bid(12)) not in roots
    assert stats["dropped_few_primary_synapses"] == 1 and str(bid(15)) not in roots
    assert str(bid(16)) in roots  # Anchor status is allowed
    assert str(bid(90)) not in roots  # untyped fragment only present in the neuron table
    assert payload["metadata"]["candidate_rows"] == 13


def test_connectivity_payload_shape_and_labels(tmp_path):
    payload = _build(tmp_path)
    meta = payload["metadata"]
    assert meta["schema_version"] == "flybrain-mc-training-samples/v1"
    assert meta["dataset_symbol"] == "mc" and meta["dataset_version"] == "male-cns_v1.0"
    assert meta["region_vocabulary"] == "male_cns_roi"
    assert meta["leakage_check"] == "passed"
    assert meta["provenance_mode"] == "explicit_paths"
    assert meta["label_definition"] == mc_samples.LABEL_DEFINITIONS[CONN]
    totals = {str(bid(p)): sum(w for _, w in targets) for p, targets in _EDGES.items()}
    threshold = meta["high_connectivity_threshold"]
    for root, sample in _by_root(payload).items():
        expected = "high_connectivity" if totals[root] >= threshold else "baseline_connectivity"
        assert sample["expected_label"] == expected
        assert sample["metadata"]["label_source"]["total_out_synapses"] == totals[root]
        assert 0.5 <= sample["expected_confidence"] <= 0.99
    assert set(meta["label_counts"]) == {"high_connectivity", "baseline_connectivity"}


@pytest.mark.parametrize("objective", ALL)
def test_input_text_has_no_label_source_or_identity(tmp_path, objective):
    payload = _build(tmp_path, objective)
    for sample in payload["samples"]:
        kv = _kv(sample)
        assert list(kv) == ["dataset", *mc_samples.INPUT_FEATURES[objective]]
        assert not set(kv) & mc_samples.FORBIDDEN_INPUT_FEATURES[objective]
        # the body id is a proxy for size: it is never in the model input
        assert sample["metadata"]["root_id"] not in sample["input_text"]
        assert "root" not in kv and "cell_type" not in kv


def test_region_mapping_is_explicit(tmp_path):
    by_root = _by_root(_build(tmp_path))
    assert by_root[str(bid(1))]["region_id"] == "brain_me"
    assert _kv(by_root[str(bid(1))])["subdivision"] == "optic_lobe"
    assert by_root[str(bid(5))]["region_id"] == "brain_al"
    assert by_root[str(bid(7))]["region_id"] == "nerve_cord_legnp_t1"
    assert _kv(by_root[str(bid(7))])["cns_division"] == "nerve_cord"
    assert by_root[str(bid(10))]["metadata"]["primary_roi"] == "GNG"  # 250 vs 250: tie breaks on name
    assert by_root[str(bid(14))]["region_id"] == "brain_eb"
    assert set(_build(tmp_path)["metadata"]["region_division_counts"]) == {"brain", "nerve_cord"}


def test_side_and_hemilineage_features(tmp_path):
    by_root = _by_root(_build(tmp_path))
    assert _kv(by_root[str(bid(13))])["side"] == "right"  # somaSide missing -> rootSide
    assert _kv(by_root[str(bid(14))])["side"] == "midline"
    # itoleeHl sentinel falls through to the VNC lineage for the split group
    assert _kv(by_root[str(bid(10))])["hemilineage"] == "19a"
    assert by_root[str(bid(10))]["metadata"]["hemilineage"] == "19a"
    # sentinels stay informative inputs but never tie neurons together
    assert _kv(by_root[str(bid(13))])["hemilineage"] == "putative_primary"
    assert by_root[str(bid(13))]["metadata"]["hemilineage"] is None
    assert by_root[str(bid(16))]["metadata"]["hemilineage"] is None
    assert by_root[str(bid(5))]["metadata"]["cell_type"] == "DA1_lPN"
    assert _kv(by_root[str(bid(5))])["cell_class"] == "alpn"


def test_neurotransmitter_payload(tmp_path):
    payload = _build(tmp_path, NT)
    meta = payload["metadata"]
    by_root = _by_root(payload)
    assert meta["filter_stats"]["dropped_nt_unclear"] == 1 and str(bid(14)) not in by_root
    assert by_root[str(bid(7))]["expected_label"] == "dominant_gaba"
    assert by_root[str(bid(3))]["expected_label"] == "dominant_glut"
    assert by_root[str(bid(16))]["expected_label"] == "dominant_da"
    assert by_root[str(bid(13))]["expected_confidence"] == 0.5
    assert by_root[str(bid(4))]["expected_confidence"] == 0.95
    assert by_root[str(bid(4))]["metadata"]["label_source"]["consensus_nt"] == "acetylcholine"
    assert _kv(by_root[str(bid(3))])["out_partner_tier"] == "t0"  # 3 partners < 100
    assert meta["source_path"].endswith("nt.feather")


def test_min_total_count_filters_nt(tmp_path):
    payload = _build(tmp_path, NT, min_total_count=10)
    assert str(bid(8)) not in _by_root(payload)
    # bodies 8 (5 predictions) and 15 (1 prediction)
    assert payload["metadata"]["filter_stats"]["dropped_below_min_total_count"] == 2


def test_unknown_nt_fails_closed(tmp_path):
    paths = _write_all(tmp_path)
    from test_flybrain_mc_adapter import _NT

    write_nt(paths["nt_path"], nt={**_NT, 3: ("nitric_oxide", 0.9, 300)})
    with pytest.raises(McAdapterError) as exc:
        _build(tmp_path, NT)
    assert exc.value.code is McAdapterErrorCode.NT_VOCABULARY_UNKNOWN


def test_region_specialization_labels(tmp_path):
    payload = _build(tmp_path, ROI)
    cut = payload["metadata"]["specialization_share"]
    assert cut == 0.75
    for root, sample in _by_root(payload).items():
        idx = int(root) - 10000
        primary = {r: p + q for r, (p, q) in _ROI_INFO[idx].items() if r in mc.MC_PRIMARY_ROIS}
        share = max(primary.values()) / sum(primary.values())
        assert sample["metadata"]["label_source"]["primary_share"] == round(share, 6)
        expected = "region_specialized" if share >= cut else "region_distributed"
        assert sample["expected_label"] == expected
    by_root = _by_root(payload)
    assert by_root[str(bid(3))]["expected_label"] == "region_specialized"
    assert by_root[str(bid(5))]["expected_label"] == "region_distributed"


def test_deterministic_and_fingerprint_sensitive(tmp_path):
    first = _build(tmp_path)
    second = _build(tmp_path)
    assert samples.stable_json(first) == samples.stable_json(second)
    changed = _build(tmp_path, high_connectivity_quantile=0.6)
    assert changed["metadata"]["input_fingerprint"] != first["metadata"]["input_fingerprint"]
    roi_a = _build(tmp_path, ROI)
    roi_b = _build(tmp_path, ROI, specialization_share=0.6)
    assert roi_a["metadata"]["input_fingerprint"] != roi_b["metadata"]["input_fingerprint"]


def test_balance_cap_round_robin(tmp_path):
    payload = _build(tmp_path, max_samples=4)
    assert len(payload["samples"]) == 4
    assert len({s["region_id"] for s in payload["samples"]}) >= 3


def test_manifest_snapshot_mode(tmp_path):
    build_snapshot(tmp_path)
    for objective in ALL:
        config = mc_samples.McSampleBuildConfig(objective=objective, storage_root=tmp_path, max_samples=50,
                                                min_total_count=1, min_region_samples=1,
                                                high_connectivity_quantile=0.5)
        payload = samples.build_training_samples("mc", objective, config, allow_planned=True)
        meta = payload["metadata"]
        assert meta["provenance_mode"] == "manifest_snapshot"
        assert set(meta["product_sha256"]) == set(mc_samples._required_roles(objective))
        assert meta["manifest_sha256"]
        assert set(meta["hash_verification"].values()) <= {"hashed", "stamp", "size_only"}
    # region_specialization never needs the edge list or the NT table
    assert mc_samples._required_roles(ROI) == tuple(sorted(
        (mc.ROLE_META, mc.ROLE_DATASET_META, mc.ROLE_NEURON_ROI_INFO)))


def test_manifest_snapshot_tamper_fails_closed(tmp_path):
    snap, _ = build_snapshot(tmp_path)
    path = snap / mc.MC_PRODUCT_PATHS[mc.ROLE_DATASET_META]
    path.write_bytes(path.read_bytes().replace(b"male-cns", b"male-cnz"))
    config = mc_samples.McSampleBuildConfig(storage_root=tmp_path, min_region_samples=1)
    with pytest.raises(McAdapterError) as exc:
        samples.build_training_samples("mc", CONN, config, allow_planned=True)
    assert exc.value.code is McAdapterErrorCode.INTEGRITY_MISMATCH


def test_partial_explicit_paths_rejected(tmp_path):
    paths = _write_all(tmp_path)
    config = mc_samples.McSampleBuildConfig(objective=CONN, meta_path=paths["meta_path"])
    with pytest.raises(ValueError, match="every required product or none"):
        samples.build_training_samples("mc", CONN, config, allow_planned=True)


def test_roi_vocabulary_drift_fails_closed(tmp_path):
    paths = _write_all(tmp_path)
    write_dataset_meta(paths["dataset_meta_path"], rois=[*sorted(mc.MC_PRIMARY_ROIS), "NEW(R)"])
    with pytest.raises(McAdapterError) as exc:
        _build(tmp_path)
    assert exc.value.code is McAdapterErrorCode.REGION_VOCABULARY_UNKNOWN


def test_config_type_and_objective_guards(tmp_path):
    with pytest.raises(DatasetRegistryError) as exc:
        samples.build_training_samples("mc", CONN, object(), allow_planned=True)
    assert exc.value.code is DatasetErrorCode.CONFIG_TYPE_MISMATCH
    with pytest.raises(DatasetRegistryError) as exc:
        samples.build_training_samples("mc", NT, _explicit_config(tmp_path, CONN), allow_planned=True)
    assert exc.value.code is DatasetErrorCode.CONFIG_OBJECTIVE_MISMATCH


def test_label_concentration_gate(tmp_path):
    with pytest.raises(ValueError, match="label concentration too high"):
        _build(tmp_path, NT, max_label_share=0.5)


@pytest.mark.parametrize(
    "bad",
    [
        {"max_samples": 0},
        {"min_total_count": 0},
        {"min_primary_synapses": 0},
        {"min_region_samples": 0},
        {"high_connectivity_quantile": 1.0},
        {"specialization_share": 0.0},
        {"max_regions": 0},
        {"max_label_share": 0.0},
        {"allowed_statuses": ()},
        {"partner_tier_edges": (10, 5)},
    ],
)
def test_invalid_config_values(tmp_path, bad):
    with pytest.raises(ValueError):
        _build(tmp_path, **bad)


def test_leakage_guard_rejects_forbidden_and_unexpected_keys():
    with pytest.raises(ValueError, match="label leakage guard"):
        mc_samples.assert_no_label_leakage([{"sample_id": "x", "input_text": "dataset mcns10 root 10001"}], CONN)
    with pytest.raises(ValueError, match="label leakage guard"):
        mc_samples.assert_no_label_leakage(
            [{"sample_id": "y", "input_text": "dataset mcns10 predicted_nt gaba"}], NT)
    with pytest.raises(ValueError, match="label leakage guard"):
        mc_samples.assert_no_label_leakage(
            [{"sample_id": "z", "input_text": "dataset mcns10 n_primary_rois 3"}], ROI)
    with pytest.raises(ValueError, match="label leakage guard"):
        mc_samples.assert_no_label_leakage(
            [{"sample_id": "w", "input_text": "dataset mcns10 out_partner_tier t2"}], CONN)


def test_forbidden_sets_cover_each_label_source():
    assert {"total_out_synapses", "n_post_partners", "downstream", "root"} <= mc_samples.FORBIDDEN_INPUT_FEATURES[CONN]
    assert {"predicted_nt", "consensus_nt", "ground_truth", "celltype_predicted_nt"} <= \
        mc_samples.FORBIDDEN_INPUT_FEATURES[NT]
    assert {"primary_share", "top_roi_synapses", "n_primary_rois"} <= mc_samples.FORBIDDEN_INPUT_FEATURES[ROI]
    for objective in ALL:
        assert "cell_type" in mc_samples.FORBIDDEN_INPUT_FEATURES[objective]


@pytest.mark.parametrize("objective", ALL)
def test_samples_carry_split_group_keys_outside_input(tmp_path, objective):
    from flybrain_dataset_registry import split_group_keys

    payload = _build(tmp_path, objective)
    assert set(split_group_keys("mc")) <= set(payload["metadata"]["split_group_keys"])
    for sample in payload["samples"]:
        assert sample["metadata"]["cell_type"]
        assert "hemilineage" in sample["metadata"]


def test_trivial_baseline_report_runs_on_grouped_split(tmp_path):
    payload = _build(tmp_path)
    report = mc_samples.trivial_baseline_report(payload, split_seed="unit")
    assert report["split"]["strategy"] == "grouped"
    assert report["train_count"] + report["heldout_count"] <= len(payload["samples"])
    # no numeric input feature exists, so the stump and argmax rules never apply
    assert report["gate_rules"]["rules"]["threshold"]["applicable"] is False
    assert report["gate_rules"]["rules"]["argmax"]["applicable"] is False
    assert set(report["lookup_audit"]) == {*mc_samples.INPUT_FEATURES[CONN], "all_input_features"}


def test_default_config_uses_stamp_cached_verification():
    config = mc_samples.McSampleBuildConfig()
    assert config.verify_hashes is None and config.stamp_dir is None
    assert dataclasses.fields(config)


def test_no_network_imports():
    source = Path(mc_samples.__file__).read_text(encoding="utf-8") + Path(mc.__file__).read_text(encoding="utf-8")
    for token in ("requests", "urllib", "http.client", "socket", "curl", "neuprint"):
        assert f"import {token}" not in source


def test_cli_writes_payload_and_baseline_report(tmp_path, capsys):
    build_snapshot(tmp_path / "root")
    out = tmp_path / "out" / "payload.json"
    report = tmp_path / "out" / "baseline.json"
    code = mc_samples.main([
        "--storage-root", str(tmp_path / "root"), "--objective", CONN, "--output", str(out),
        "--min-total-count", "1", "--min-region-samples", "1", "--high-connectivity-quantile", "0.5",
        "--baseline-report", str(report), "--allow-planned",
    ])
    assert code == 0 and out.is_file() and report.is_file()
    assert '"status": "ok"' in capsys.readouterr().out
    assert mc_samples.main(["--storage-root", str(tmp_path / "root"), "--output", str(out)]) == 2


_REAL_ROOT = os.environ.get("LOCI_FLYBRAIN_STORAGE_ROOT", "")


@pytest.mark.skipif(
    not _REAL_ROOT or not (Path(_REAL_ROOT) / "snapshots" / "mc" / "male-cns_v1.0" / "manifest" / "manifest.json").is_file(),
    reason="real mc snapshot not present under LOCI_FLYBRAIN_STORAGE_ROOT",
)
@pytest.mark.parametrize("objective", ALL)
def test_real_snapshot_smoke(objective):
    # verify_hashes=False keeps the run read-only (no stamps written into the snapshot).
    config = mc_samples.McSampleBuildConfig(objective=objective, storage_root=_REAL_ROOT, verify_hashes=False,
                                            max_samples=500)
    payload = samples.build_training_samples("mc", objective, config, allow_planned=True)
    meta = payload["metadata"]
    assert meta["selected_count"] == 500
    assert set(meta["region_division_counts"]) == {"brain", "nerve_cord"}
    assert meta["dominant_label_share"] <= 0.9
    assert all("root" not in _kv(s) for s in payload["samples"])
    report = mc_samples.trivial_baseline_report(payload)
    # no gate rule may recover the label from the input (the body-id stump used to):
    # the input has no numeric feature, and no categorical lookup comes near 1.0
    assert report["gate_rules"]["rules"]["threshold"]["applicable"] is False
    assert report["gate_rules"]["best_rule"] in {"majority", "lookup"}
    assert report["gate_rules"]["best_accuracy"] < 0.9
    assert report["best_single_feature_lookup"]["accuracy"] < 0.9
