import dataclasses
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import flybrain_banc_adapter as banc  # noqa: E402
import flybrain_brain_cluster_banc_samples as banc_samples  # noqa: E402
import flybrain_brain_cluster_samples as samples  # noqa: E402
from flybrain_banc_adapter import BancAdapterError, BancAdapterErrorCode  # noqa: E402
from flybrain_dataset_registry import DatasetErrorCode, DatasetRegistryError  # noqa: E402
from test_flybrain_banc_adapter import (  # noqa: E402
    _META_ROWS,
    build_snapshot,
    rid,
    write_edgelist,
    write_meta,
    write_nt,
)

# Importing the module registers the banc builders globally. Undo that at
# import time so other test modules (e.g. the dispatcher's stub-banc tests)
# see an empty slot; each test here re-registers via the fixture below.
banc_samples.unregister_banc_sample_builders()

CONN = "connectivity_tier"
NT = "neurotransmitter_dominance"


@pytest.fixture(autouse=True)
def _banc_registered():
    banc_samples.register_banc_sample_builders(replace=True)
    yield
    banc_samples.unregister_banc_sample_builders()


def _explicit_config(tmp_path, objective=CONN, **overrides):
    edges = tmp_path / "edges.feather"
    nt = tmp_path / "nt.csv"
    meta = tmp_path / "meta.feather"
    if not edges.exists():
        write_edgelist(edges)
        write_nt(nt)
        write_meta(meta)
    base = dict(
        objective=objective,
        edgelist_path=edges,
        nt_path=nt,
        meta_path=meta,
        max_samples=50,
        min_total_count=1,
        min_region_samples=1,
        high_connectivity_quantile=0.5,
        max_regions=10,
    )
    base.update(overrides)
    return banc_samples.BancSampleBuildConfig(**base)


def _kv(sample):
    tokens = sample["input_text"].split()
    return dict(zip(tokens[0::2], tokens[1::2]))


def test_registration_covers_supported_objectives():
    assert samples.registered_builders()["banc"] == (CONN, NT)


def test_planned_dataset_requires_opt_in(tmp_path):
    with pytest.raises(DatasetRegistryError) as exc:
        samples.build_training_samples("banc", CONN, _explicit_config(tmp_path))
    assert exc.value.code is DatasetErrorCode.DATASET_NOT_ACTIVE


def test_connectivity_payload_shape_and_labels(tmp_path):
    payload = samples.build_training_samples("BANC", CONN, _explicit_config(tmp_path), allow_planned=True)
    meta = payload["metadata"]
    assert meta["schema_version"] == "flybrain-banc-training-samples/v1"
    assert meta["dataset_symbol"] == "banc"
    assert meta["objective"] == CONN
    assert meta["region_vocabulary"] == "banc_neuropil"
    assert meta["provenance_mode"] == "explicit_paths"
    assert set(meta["label_counts"]) == {"high_connectivity", "baseline_connectivity"}
    assert set(meta["region_division_counts"]) == {"brain", "nerve_cord"}
    assert meta["selected_regions"] == ["brain_al", "brain_me", "nerve_cord_lnp_t1", "nerve_cord_pronm_t1"]
    stats = meta["filter_stats"]
    assert stats["dropped_missing_root_region"] == 1  # idx 11
    assert stats["dropped_division_conflict"] == 1  # idx 12
    assert stats["dropped_non_neuronal"] == 1  # idx 13 glia
    assert stats["dropped_unproofread"] == 1  # idx 14
    ids = {s["metadata"]["root_id"] for s in payload["samples"]}
    assert ids.isdisjoint({rid(11), rid(12), rid(13), rid(14)})
    # threshold = median of outgoing totals over retained candidates
    by_id = {s["metadata"]["root_id"]: s for s in payload["samples"]}
    threshold = meta["high_connectivity_threshold"]
    for root_id, sample in by_id.items():
        total = sample["metadata"]["label_source"]["total_out_synapses"]
        expected = "high_connectivity" if total >= threshold else "baseline_connectivity"
        assert sample["expected_label"] == expected
        assert 0.5 <= sample["expected_confidence"] <= 0.99
        assert sample["sample_id"] == f"banc888-pre-{root_id}"


def test_connectivity_input_excludes_label_source(tmp_path):
    payload = samples.build_training_samples("banc", CONN, _explicit_config(tmp_path), allow_planned=True)
    for sample in payload["samples"]:
        kv = _kv(sample)
        assert list(kv) == ["dataset", "root", *banc_samples.INPUT_FEATURES[CONN]]
        assert not set(kv) & banc_samples.FORBIDDEN_INPUT_FEATURES[CONN]
        total = str(sample["metadata"]["label_source"]["total_out_synapses"])
        assert total not in [v for k, v in kv.items() if k != "root"]
    assert payload["metadata"]["leakage_check"] == "passed"


def test_region_mapping_is_explicit_brain_vs_cord(tmp_path):
    payload = samples.build_training_samples("banc", CONN, _explicit_config(tmp_path), allow_planned=True)
    for sample in payload["samples"]:
        kv = _kv(sample)
        division = kv["cns_division"]
        assert sample["region_id"].startswith(division + "_")
        assert sample["metadata"]["cns_division"] == division
        if sample["metadata"]["root_region"].startswith(("COURT_vnc_", "MANC_vnc_")):
            assert division == "nerve_cord" and kv["subdivision"] == "ventral_nerve_cord"
        else:
            assert division == "brain"


def test_neurotransmitter_payload(tmp_path):
    payload = samples.build_training_samples("banc", NT, _explicit_config(tmp_path, NT), allow_planned=True)
    meta = payload["metadata"]
    assert meta["objective"] == NT
    assert meta["high_connectivity_threshold"] is None
    labels = {s["expected_label"] for s in payload["samples"]}
    assert labels <= {f"dominant_{c}" for c in banc.BANC_NT_SHORT_CODES.values()}
    assert {"dominant_ach", "dominant_gaba", "dominant_glut"} <= labels
    assert "dominant_his" in labels and "dominant_da" in labels  # BANC-only class retained
    assert meta["filter_stats"]["dropped_multi_anchor"] == 1  # idx 5 duplicated in the CSV
    by_id = {s["metadata"]["root_id"]: s for s in payload["samples"]}
    assert rid(5) not in by_id
    assert by_id[rid(15)]["expected_confidence"] == 0.52
    assert by_id[rid(1)]["expected_confidence"] == 0.91
    assert by_id[rid(10)]["expected_label"] == "dominant_ach"
    for sample in payload["samples"]:
        kv = _kv(sample)
        assert list(kv) == ["dataset", "root", *banc_samples.INPUT_FEATURES[NT]]
        assert not set(kv) & banc_samples.FORBIDDEN_INPUT_FEATURES[NT]
        assert not any(name in sample["input_text"] for name in banc.BANC_NT_SHORT_CODES)
        assert sample["sample_id"].startswith("banc888-nt-")


def test_min_total_count_filters_nt(tmp_path):
    payload = samples.build_training_samples(
        "banc", NT, _explicit_config(tmp_path, NT, min_total_count=10), allow_planned=True
    )
    ids = {s["metadata"]["root_id"] for s in payload["samples"]}
    assert rid(8) not in ids and rid(15) not in ids and rid(6) not in ids
    assert all(s["metadata"]["label_source"]["classified_presynapses"] >= 10 for s in payload["samples"])


def test_deterministic_and_fingerprint_sensitive(tmp_path):
    config = _explicit_config(tmp_path)
    first = samples.build_training_samples("banc", CONN, config, allow_planned=True)
    second = samples.build_training_samples("banc", CONN, config, allow_planned=True)
    assert first == second
    other = samples.build_training_samples(
        "banc", CONN, dataclasses.replace(config, high_connectivity_quantile=0.6), allow_planned=True
    )
    assert other["metadata"]["input_fingerprint"] != first["metadata"]["input_fingerprint"]


def test_balance_cap_round_robin(tmp_path):
    payload = samples.build_training_samples(
        "banc", CONN, _explicit_config(tmp_path, max_samples=4), allow_planned=True
    )
    assert len(payload["samples"]) == 4
    assert [s["region_id"] for s in payload["samples"]] == [
        "brain_al", "brain_me", "nerve_cord_lnp_t1", "nerve_cord_pronm_t1",
    ]


def test_manifest_snapshot_mode(tmp_path):
    build_snapshot(tmp_path)
    config = banc_samples.BancSampleBuildConfig(
        objective=NT, storage_root=tmp_path, min_total_count=1, min_region_samples=1, max_regions=10
    )
    payload = samples.build_training_samples("banc", NT, config, allow_planned=True)
    meta = payload["metadata"]
    assert meta["provenance_mode"] == "manifest_snapshot"
    assert len(meta["manifest_sha256"]) == 64
    assert set(meta["product_sha256"]) == {"edgelist_v3", "meta", "nt_prediction"}
    assert meta["license"] == "CC-BY-4.0"


def test_manifest_snapshot_tamper_fails_closed(tmp_path):
    snap, _ = build_snapshot(tmp_path)
    edges = snap / banc.BANC_PRODUCT_PATHS[banc.ROLE_EDGELIST_V3]
    write_edgelist(edges, edges={1: [(2, 999)], 2: [(1, 1)]})
    config = banc_samples.BancSampleBuildConfig(objective=CONN, storage_root=tmp_path, min_region_samples=1)
    with pytest.raises(BancAdapterError) as exc:
        samples.build_training_samples("banc", CONN, config, allow_planned=True)
    assert exc.value.code in (BancAdapterErrorCode.INTEGRITY_MISMATCH,)


def test_partial_explicit_paths_rejected(tmp_path):
    config = _explicit_config(tmp_path)
    with pytest.raises(ValueError, match="every required product"):
        samples.build_training_samples(
            "banc", CONN, dataclasses.replace(config, meta_path=None), allow_planned=True
        )


def test_unknown_region_vocabulary_fails_closed(tmp_path):
    rows = list(_META_ROWS)
    rows[0] = (1, "FAFB_brain_ME_R", *rows[0][2:])
    meta = tmp_path / "meta.feather"
    write_edgelist(tmp_path / "edges.feather")
    write_nt(tmp_path / "nt.csv")
    write_meta(meta, rows=rows)
    with pytest.raises(BancAdapterError) as exc:
        samples.build_training_samples("banc", CONN, _explicit_config(tmp_path), allow_planned=True)
    assert exc.value.code is BancAdapterErrorCode.REGION_VOCABULARY_UNKNOWN


def test_config_type_and_objective_guards(tmp_path):
    with pytest.raises(DatasetRegistryError) as exc:
        samples.build_training_samples("banc", NT, _explicit_config(tmp_path, CONN), allow_planned=True)
    assert exc.value.code is DatasetErrorCode.CONFIG_OBJECTIVE_MISMATCH
    with pytest.raises(DatasetRegistryError) as exc:
        samples.build_training_samples("banc", CONN, object(), allow_planned=True)
    assert exc.value.code is DatasetErrorCode.CONFIG_TYPE_MISMATCH
    with pytest.raises(DatasetRegistryError) as exc:
        samples.build_training_samples("banc", "region_specialization_tier", None, allow_planned=True)
    assert exc.value.code is DatasetErrorCode.BUILDER_NOT_REGISTERED


def test_label_concentration_gate(tmp_path):
    with pytest.raises(ValueError, match="label concentration too high"):
        samples.build_training_samples(
            "banc", CONN, _explicit_config(tmp_path, max_label_share=0.51, high_connectivity_quantile=0.1),
            allow_planned=True,
        )


def test_invalid_config_values(tmp_path):
    for bad in (dict(max_samples=0), dict(high_connectivity_quantile=1.0), dict(partner_tier_edges=(100, 10))):
        with pytest.raises(ValueError):
            samples.build_training_samples("banc", CONN, _explicit_config(tmp_path, **bad), allow_planned=True)


def test_leakage_guard_rejects_forbidden_keys():
    bad = [{"sample_id": "x", "input_text": "dataset banc888 root 1 total_out_synapses 400"}]
    with pytest.raises(ValueError, match="label leakage guard"):
        banc_samples.assert_no_label_leakage(bad, CONN)
    bad_nt = [{"sample_id": "y", "input_text": "dataset banc888 root 1 neurotransmitter_predicted gaba"}]
    with pytest.raises(ValueError, match="label leakage guard"):
        banc_samples.assert_no_label_leakage(bad_nt, NT)


def test_no_network_imports():
    source = Path(banc_samples.__file__).read_text(encoding="utf-8") + Path(banc.__file__).read_text(encoding="utf-8")
    for token in ("requests", "urllib", "http.client", "socket", "curl"):
        assert f"import {token}" not in source


_REAL_ROOT = os.environ.get("LOCI_FLYBRAIN_STORAGE_ROOT", "")


@pytest.mark.skipif(
    not _REAL_ROOT or not (Path(_REAL_ROOT) / "snapshots" / "BANC" / "banc_888" / "manifest" / "manifest.json").is_file(),
    reason="real BANC snapshot not present under LOCI_FLYBRAIN_STORAGE_ROOT",
)
@pytest.mark.parametrize("objective", [CONN, NT])
def test_real_snapshot_smoke(objective):
    config = banc_samples.BancSampleBuildConfig(objective=objective, storage_root=_REAL_ROOT, verify_hashes=False,
                                                max_samples=500)
    payload = samples.build_training_samples("banc", objective, config, allow_planned=True)
    meta = payload["metadata"]
    assert meta["selected_count"] == 500
    assert set(meta["region_division_counts"]) == {"brain", "nerve_cord"}
    assert meta["dominant_label_share"] <= 0.9


def test_string_proofread_column_is_parsed_not_truthy(tmp_path):
    """Real banc_888_meta stores proofread as "TRUE"/"FALSE" strings."""
    string_rows = [(*row[:8], "TRUE" if row[8] else "FALSE") for row in _META_ROWS]
    edges = tmp_path / "edges.feather"
    nt = tmp_path / "nt.csv"
    meta = tmp_path / "meta.feather"
    write_edgelist(edges)
    write_nt(nt)
    write_meta(meta, rows=string_rows)
    config = banc_samples.BancSampleBuildConfig(
        objective=CONN,
        edgelist_path=edges,
        nt_path=nt,
        meta_path=meta,
        max_samples=50,
        min_total_count=1,
        min_region_samples=1,
        high_connectivity_quantile=0.5,
        max_regions=10,
    )
    payload = samples.build_training_samples("banc", CONN, config, allow_planned=True)
    assert payload["metadata"]["filter_stats"]["dropped_unproofread"] == 1
    assert rid(14) not in {s["metadata"]["root_id"] for s in payload["samples"]}


def test_unknown_proofread_token_fails_closed():
    with pytest.raises(BancAdapterError) as exc:
        banc_samples._proofread_mask(["TRUE", "maybe"])
    assert exc.value.code is BancAdapterErrorCode.SCHEMA_MISMATCH
    assert banc_samples._proofread_mask([True, False, "FALSE", "true", None]) == [True, False, False, True, False]


@pytest.mark.parametrize("objective", [CONN, NT])
def test_samples_carry_split_group_keys_outside_input(tmp_path, objective):
    payload = samples.build_training_samples("banc", objective, _explicit_config(tmp_path, objective=objective),
                                             allow_planned=True)
    for sample in payload["samples"]:
        assert sample["metadata"]["cell_type"] == "ct"  # synthetic meta fixture
        assert "hemilineage" in sample["metadata"]
        assert "cell_type" not in _kv(sample)
    by_id = {s["metadata"]["root_id"]: s for s in payload["samples"]}
    if rid(5) in by_id:
        assert by_id[rid(5)]["metadata"]["hemilineage"] == "all1"


def test_default_config_uses_stamp_cached_verification():
    assert banc_samples.BancSampleBuildConfig().verify_hashes is None
