import dataclasses
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import flybrain_brain_cluster_ol_samples as ol_samples  # noqa: E402
import flybrain_brain_cluster_samples as samples  # noqa: E402
import flybrain_ol_adapter as ol  # noqa: E402
from flybrain_dataset_registry import DatasetErrorCode, DatasetRegistryError  # noqa: E402
from flybrain_ol_adapter import OlAdapterError, OlAdapterErrorCode  # noqa: E402
from test_flybrain_ol_adapter import _NEURONS, build_snapshot, roi_info, write_meta, write_neurons  # noqa: E402

# Importing the module registers the ol builders globally. Undo that at import
# time so other test modules see an empty slot; each test here re-registers.
ol_samples.unregister_ol_sample_builders()

CONN = "connectivity_tier"
NT = "neurotransmitter_dominance"


@pytest.fixture(autouse=True)
def _ol_registered():
    ol_samples.register_ol_sample_builders(replace=True)
    yield
    ol_samples.unregister_ol_sample_builders()


def _explicit_config(tmp_path, objective=CONN, rows=None, **overrides):
    neurons = tmp_path / "neurons.feather"
    meta = tmp_path / "meta.csv"
    if rows is not None or not neurons.exists():
        write_neurons(neurons, rows=rows if rows is not None else _NEURONS)
        write_meta(meta)
    base = dict(
        objective=objective,
        neurons_path=neurons,
        meta_path=meta,
        max_samples=50,
        min_total_count=10,
        min_region_samples=1,
        high_connectivity_quantile=0.5,
        max_regions=10,
    )
    base.update(overrides)
    return ol_samples.OlSampleBuildConfig(**base)


def _kv(sample):
    tokens = sample["input_text"].split()
    return dict(zip(tokens[0::2], tokens[1::2]))


def _build(tmp_path, objective=CONN, **overrides):
    return samples.build_training_samples("ol", objective, _explicit_config(tmp_path, objective, **overrides),
                                          allow_planned=True)


def test_registration_covers_supported_objectives():
    assert samples.registered_builders()["ol"] == (CONN, NT)


def test_planned_dataset_requires_opt_in(tmp_path):
    with pytest.raises(DatasetRegistryError) as exc:
        samples.build_training_samples("ol", CONN, _explicit_config(tmp_path))
    assert exc.value.code is DatasetErrorCode.DATASET_NOT_ACTIVE


def test_region_specialization_is_not_supported(tmp_path):
    # The registry allow-list excludes it (no non-circular label in this release).
    with pytest.raises(DatasetRegistryError) as exc:
        samples.build_training_samples("ol", "region_specialization_tier", None, allow_planned=True)
    assert exc.value.code is DatasetErrorCode.OBJECTIVE_NOT_SUPPORTED


def test_connectivity_payload_shape_and_labels(tmp_path):
    payload = _build(tmp_path)
    meta = payload["metadata"]
    assert meta["schema_version"] == "flybrain-ol-training-samples/v1"
    assert meta["dataset_symbol"] == "ol"
    assert meta["dataset_version"] == "optic_lobe_v1.1"
    assert meta["neuprint_release"] == "optic-lobe:v1.1"
    assert meta["region_vocabulary"] == "optic_lobe_neuropil"
    assert meta["provenance_mode"] == "explicit_paths"
    assert meta["source_rows"] == len(_NEURONS)
    assert meta["typed_rows"] == len(_NEURONS) - 2
    stats = meta["filter_stats"]
    assert stats["dropped_status"] == 1  # 10010 Orphan
    assert stats["dropped_below_min_total"] == 1  # 10006 downstream 5
    assert stats["dropped_no_primary_roi"] == 1  # 10012 only OL(R) + columns
    ids = {s["metadata"]["body_id"] for s in payload["samples"]}
    assert ids == {"10001", "10002", "10003", "10004", "10005", "10007", "10008", "10011", "10014", "10015"}
    assert meta["selected_regions"] == ["cb_pvlp_r", "ol_lo_r", "ol_lop_r", "ol_me_r"]
    assert set(meta["region_division_counts"]) == {"central_brain", "optic_lobe"}
    threshold = meta["high_connectivity_threshold"]
    assert threshold == 400.0  # median downstream of the 10 retained candidates
    for sample in payload["samples"]:
        total = sample["metadata"]["label_source"]["downstream"]
        expected = "high_connectivity" if total >= threshold else "baseline_connectivity"
        assert sample["expected_label"] == expected
        assert 0.5 <= sample["expected_confidence"] <= 0.99
        assert sample["sample_id"] == f"ol11-out-{sample['metadata']['body_id']}"


def test_region_and_features_from_roi_info(tmp_path):
    payload = _build(tmp_path)
    by_id = {s["metadata"]["body_id"]: s for s in payload["samples"]}
    tm1 = by_id["10003"]
    assert tm1["region_id"] == "ol_lo_r"
    kv = _kv(tm1)
    assert kv["primary_neuropil"] == "lo_r"
    assert kv["input_neuropil"] == "me_r"
    assert kv["output_neuropil"] == "lo_r"
    assert kv["arbor"] == "lo_r+me_r"
    assert kv["dominant_layer"] == "lo_r_layer_02"
    lc4 = _kv(by_id["10007"])
    assert by_id["10007"]["region_id"] == "cb_pvlp_r"
    assert (lc4["soma"], lc4["hemilineage"], lc4["hex_column"]) == ("no", "vlpl2_lateral", "no")
    assert _kv(by_id["10001"])["hex_column"] == "yes"
    ct1 = by_id["10015"]
    assert ct1["region_id"] == "ol_lo_r"  # LO(R) synweight 1600 > ME(L) 1300
    assert _kv(ct1)["side"] == "left"
    assert _kv(ct1)["input_neuropil"] == "lo_r" and _kv(ct1)["arbor"] == "lo_r+me_l"
    assert _kv(by_id["10005"])["dominant_layer"] == "lop_r_layer_01"


def test_connectivity_input_excludes_label_source_type_and_body_id(tmp_path):
    payload = _build(tmp_path)
    for sample in payload["samples"]:
        kv = _kv(sample)
        assert list(kv) == ["dataset", *ol_samples.INPUT_FEATURES[CONN]]
        assert not set(kv) & ol_samples.FORBIDDEN_INPUT_FEATURES[CONN]
        body_id = sample["metadata"]["body_id"]
        values = set(kv.values())
        assert body_id not in values
        assert str(sample["metadata"]["label_source"]["downstream"]) not in values
        assert sample["metadata"]["cell_type"].lower() not in values
        assert sample["metadata"]["cell_type"] not in sample["input_text"].split()
    assert payload["metadata"]["leakage_check"] == "passed"


def test_neurotransmitter_payload(tmp_path):
    payload = _build(tmp_path, NT)
    meta = payload["metadata"]
    stats = meta["filter_stats"]
    assert stats["dropped_nt_unclear"] == 1  # 10008
    assert stats["dropped_below_min_total"] == 1  # 10006 totalNt 3
    labels = {s["metadata"]["body_id"]: s["expected_label"] for s in payload["samples"]}
    assert labels["10011"] == "dominant_gaba"
    assert labels["10004"] == "dominant_glut"
    assert labels["10001"] == "dominant_ach"
    assert "10008" not in labels
    for sample in payload["samples"]:
        kv = _kv(sample)
        assert list(kv) == ["dataset", *ol_samples.INPUT_FEATURES[NT]]
        assert not set(kv) & ol_samples.FORBIDDEN_INPUT_FEATURES[NT]
        assert sample["sample_id"] == f"ol11-nt-{sample['metadata']['body_id']}"
        score = sample["metadata"]["label_source"]["predicted_nt_confidence"]
        assert sample["expected_confidence"] == pytest.approx(min(0.99, max(0.5, score)))
    polarity = {s["metadata"]["body_id"]: _kv(s)["polarity_tier"] for s in payload["samples"]}
    assert polarity["10002"] == "t1"  # 12 / 72 = 0.167
    assert polarity["10005"] == "t2"  # 250 / 330


def test_unknown_nt_fails_closed(tmp_path):
    rows = [(*_NEURONS[0][:7], "nitric_oxide", *_NEURONS[0][8:]), _NEURONS[1]]
    with pytest.raises(OlAdapterError) as exc:
        _build(tmp_path, NT, rows=rows)
    assert exc.value.code is OlAdapterErrorCode.NT_VOCABULARY_UNKNOWN


def test_nt_confidence_out_of_range_fails_closed(tmp_path):
    rows = [(*_NEURONS[0][:8], 1.7, *_NEURONS[0][9:]), _NEURONS[3]]
    with pytest.raises(OlAdapterError) as exc:
        _build(tmp_path, NT, rows=rows)
    assert exc.value.code is OlAdapterErrorCode.SCHEMA_MISMATCH


def test_unknown_roi_key_fails_closed(tmp_path):
    rows = [(*_NEURONS[0][:13], roi_info(ME_R=(1, 400), Medulla={"pre": 1})), _NEURONS[3]]
    with pytest.raises(OlAdapterError) as exc:
        _build(tmp_path, rows=rows)
    assert exc.value.code is OlAdapterErrorCode.REGION_VOCABULARY_UNKNOWN


def test_deterministic_and_fingerprint_sensitive(tmp_path):
    first = _build(tmp_path)
    second = _build(tmp_path)
    assert first == second
    changed = _build(tmp_path, high_connectivity_quantile=0.6)
    assert changed["metadata"]["input_fingerprint"] != first["metadata"]["input_fingerprint"]
    statuses = _build(tmp_path, allowed_statuses=("Traced", "Orphan"))
    assert statuses["metadata"]["input_fingerprint"] != first["metadata"]["input_fingerprint"]
    assert "10010" in {s["metadata"]["body_id"] for s in statuses["samples"]}


def test_cap_is_region_round_robin_and_independent_of_id_order(tmp_path):
    payload = _build(tmp_path, max_samples=4)
    assert payload["metadata"]["selected_count"] == 4
    assert {s["region_id"] for s in payload["samples"]} == {"cb_pvlp_r", "ol_lo_r", "ol_lop_r", "ol_me_r"}
    fake = [
        {"sample_id": f"s-{i}", "region_id": "r1" if i % 2 else "r2", "metadata": {"body_id": str(i)}}
        for i in range(40)
    ]
    picked = {s["sample_id"] for s in ol_samples._hash_round_robin(fake, max_samples=10)}
    assert picked == {s["sample_id"] for s in ol_samples._hash_round_robin(list(reversed(fake)), max_samples=10)}
    lowest = {f"s-{i}" for i in range(10)}
    assert picked != lowest
    assert len(ol_samples._hash_round_robin(fake, max_samples=100)) == 40


def test_manifest_snapshot_mode(tmp_path):
    build_snapshot(tmp_path)
    config = ol_samples.OlSampleBuildConfig(
        objective=NT, storage_root=tmp_path, min_total_count=10, min_region_samples=1, max_regions=10
    )
    payload = samples.build_training_samples("ol", NT, config, allow_planned=True)
    meta = payload["metadata"]
    assert meta["provenance_mode"] == "manifest_snapshot"
    assert len(meta["manifest_sha256"]) == 64
    assert set(meta["product_sha256"]) == {"neuron_annotations", "neuprint_meta"}
    assert meta["license"] == "CC-BY-4.0"


def test_manifest_snapshot_tamper_fails_closed(tmp_path):
    snap, _ = build_snapshot(tmp_path)
    write_neurons(snap / ol.OL_PRODUCT_PATHS[ol.ROLE_NEURONS], rows=_NEURONS[:3])
    config = ol_samples.OlSampleBuildConfig(objective=CONN, storage_root=tmp_path, min_region_samples=1)
    with pytest.raises(OlAdapterError) as exc:
        samples.build_training_samples("ol", CONN, config, allow_planned=True)
    assert exc.value.code is OlAdapterErrorCode.INTEGRITY_MISMATCH


def test_partial_explicit_paths_rejected(tmp_path):
    config = _explicit_config(tmp_path)
    with pytest.raises(ValueError, match="every required product"):
        samples.build_training_samples("ol", CONN, dataclasses.replace(config, meta_path=None), allow_planned=True)


def test_release_pin_checked_in_explicit_mode(tmp_path):
    config = _explicit_config(tmp_path)
    write_meta(config.meta_path, tag="v1.0")
    with pytest.raises(OlAdapterError) as exc:
        samples.build_training_samples("ol", CONN, config, allow_planned=True)
    assert exc.value.code is OlAdapterErrorCode.DATASET_PIN_MISMATCH


def test_config_type_and_objective_guards(tmp_path):
    with pytest.raises(DatasetRegistryError) as exc:
        samples.build_training_samples("ol", NT, _explicit_config(tmp_path, CONN), allow_planned=True)
    assert exc.value.code is DatasetErrorCode.CONFIG_OBJECTIVE_MISMATCH
    with pytest.raises(DatasetRegistryError) as exc:
        samples.build_training_samples("ol", CONN, object(), allow_planned=True)
    assert exc.value.code is DatasetErrorCode.CONFIG_TYPE_MISMATCH


def test_label_concentration_gate(tmp_path):
    with pytest.raises(ValueError, match="label concentration too high"):
        _build(tmp_path, max_label_share=0.51, high_connectivity_quantile=0.05)


@pytest.mark.parametrize(
    "bad",
    [
        dict(max_samples=0),
        dict(high_connectivity_quantile=1.0),
        dict(min_total_count=0),
        dict(allowed_statuses=()),
        dict(arbor_min_share=0.0),
        dict(polarity_tier_edges=(0.3, 0.1)),
        dict(polarity_tier_edges=(0.0, 0.5)),
        dict(column_span_tier_edges=(10, 2)),
    ],
)
def test_invalid_config_values(tmp_path, bad):
    # Every validation message starts with the parameter it rejects, so a
    # deleted check cannot pass on some later, unrelated ValueError.
    (key,) = bad
    with pytest.raises(ValueError, match=rf"^{key} must "):
        _build(tmp_path, **bad)


def test_leakage_guard_rejects_forbidden_and_unexpected_keys():
    for text, objective in (
        ("dataset ol11 downstream 400", CONN),
        ("dataset ol11 body 10001", CONN),
        ("dataset ol11 cell_type Mi1", NT),
        ("dataset ol11 predicted_nt gaba", NT),
        ("dataset ol11 polarity_tier t1", CONN),
        ("dataset ol11 mystery x", NT),
        ("dataset ol11 arbor", NT),
    ):
        with pytest.raises(ValueError, match="label leakage guard"):
            ol_samples.assert_no_label_leakage([{"sample_id": "x", "input_text": text}], objective)


@pytest.mark.parametrize("objective", [CONN, NT])
def test_samples_carry_split_group_key_outside_input(tmp_path, objective):
    payload = _build(tmp_path, objective)
    for sample in payload["samples"]:
        assert sample["metadata"]["cell_type"]
        assert "cell_type" not in _kv(sample)
    types = {s["metadata"]["body_id"]: s["metadata"]["cell_type"] for s in payload["samples"]}
    assert types["10001"] == types["10002"] == "Mi1"


def test_label_derivability_audit_uses_grouped_split(tmp_path):
    rows = []
    for index in range(60):
        cell_type = f"T{index % 12}"
        big = index % 3 == 0
        rows.append((
            20000 + index, cell_type, f"{cell_type}_R", "Traced", 50, 100, 900 if big else 40,
            "acetylcholine" if index % 2 else "glutamate", 0.8, 50.0, None, None, None,
            roi_info(ME_R=(50, 100)) if index % 4 else roi_info(LO_R=(50, 100)),
        ))
    payload = _build(tmp_path, rows=rows, max_samples=100, high_connectivity_quantile=0.8)
    audit = ol_samples.label_derivability_audit(payload["samples"])
    assert audit["split"]["group_keys"] == ["cell_type"]
    assert audit["split"]["train"] + audit["split"]["heldout"] == len(payload["samples"])
    # No numeric features in ol input_text: only majority and the categorical lookup apply.
    assert audit["gate_best_rule"] in {"majority", "lookup"}
    assert audit["gate_rules"]["threshold"] is None and audit["gate_rules"]["argmax"] is None
    assert audit["gate_rules"]["lookup"] is not None
    # The shared gate's lookup rule is at least as strict as this audit's best single feature.
    assert audit["gate_best_accuracy"] >= audit["best_single_feature"]["accuracy"] - 1e-4
    assert set(audit["single_feature"]) == set(ol_samples.INPUT_FEATURES[CONN])
    assert 0.0 <= audit["all_features_lookup"] <= 1.0
    assert audit == ol_samples.label_derivability_audit(payload["samples"])


def test_default_config_uses_stamp_cached_verification():
    assert ol_samples.OlSampleBuildConfig().verify_hashes is None


def test_no_network_imports():
    source = Path(ol_samples.__file__).read_text(encoding="utf-8")
    for token in ("requests", "urllib", "http.client", "socket", "curl"):
        assert f"import {token}" not in source


def test_cli_writes_payload_and_reports_errors(tmp_path, capsys):
    build_snapshot(tmp_path / "root")
    out = tmp_path / "out" / "payload.json"
    code = ol_samples.main(["--storage-root", str(tmp_path / "root"), "--output", str(out), "--allow-planned",
                            "--min-region-samples", "1", "--audit"])
    assert code == 0 and out.is_file()
    assert '"status": "ok"' in capsys.readouterr().out
    code = ol_samples.main(["--storage-root", str(tmp_path / "root"), "--output", str(out)])
    assert code == 2
    assert "DATASET_NOT_ACTIVE" in capsys.readouterr().out


_REAL_ROOT = os.environ.get("LOCI_FLYBRAIN_STORAGE_ROOT", "")


@pytest.mark.skipif(
    not _REAL_ROOT
    or not (Path(_REAL_ROOT) / "snapshots" / "ol" / "optic_lobe_v1.1" / "manifest" / "manifest.json").is_file(),
    reason="real ol snapshot not present under LOCI_FLYBRAIN_STORAGE_ROOT",
)
@pytest.mark.parametrize("objective", [CONN, NT])
def test_real_snapshot_smoke(objective):
    # verify_hashes=False keeps the smoke run read-only (no hash stamps written).
    config = ol_samples.OlSampleBuildConfig(objective=objective, storage_root=_REAL_ROOT, verify_hashes=False,
                                            max_samples=500)
    payload = samples.build_training_samples("ol", objective, config, allow_planned=True)
    meta = payload["metadata"]
    assert meta["selected_count"] == 500
    assert meta["source_rows"] > 10_000_000 and 50_000 < meta["typed_rows"] < 60_000
    assert "optic_lobe" in meta["region_division_counts"]
    assert meta["dominant_label_share"] <= 0.9
    ol_samples.assert_no_label_leakage(payload["samples"], objective)
