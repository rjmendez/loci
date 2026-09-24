import dataclasses
import json
import os
import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import flybrain_brain_cluster_mv_samples as mv_samples  # noqa: E402
import flybrain_brain_cluster_samples as samples  # noqa: E402
import flybrain_brain_cluster_training as fbct  # noqa: E402
from flybrain_dataset_registry import DatasetErrorCode, DatasetRegistryError, split_group_keys  # noqa: E402
from flybrain_hash_stamps import HASH_STAMP_RELATIVE_DIR  # noqa: E402
from flybrain_mv_adapter import MvAdapterError, MvAdapterErrorCode  # noqa: E402
from test_flybrain_mv_adapter import (  # noqa: E402
    BIG_BODY,
    NEURONS,
    body,
    build_snapshot,
    write_edgelist,
    write_meta,
)

# Importing the module registers the mv builders globally. Undo that so other
# test modules see an empty slot; each test here re-registers via the fixture.
mv_samples.unregister_mv_sample_builders()

CONN = "connectivity_tier"
NT = "neurotransmitter_dominance"
SPEC = "region_specialization_tier"
ALL = (CONN, NT, SPEC)


@pytest.fixture(autouse=True)
def _mv_registered():
    mv_samples.register_mv_sample_builders(replace=True)
    yield
    mv_samples.unregister_mv_sample_builders()


def _paths(tmp_path):
    meta = tmp_path / "meta.feather"
    edges = tmp_path / "edges.csv"
    if not meta.exists():
        write_meta(meta)
        write_edgelist(edges)
    return meta, edges


def _config(tmp_path, objective=CONN, **overrides):
    meta, edges = _paths(tmp_path)
    base = dict(
        objective=objective,
        meta_path=meta,
        edgelist_path=edges,
        max_samples=50,
        min_total_count=1,
        min_region_samples=1,
        high_connectivity_quantile=0.5,
        max_regions=10,
        max_label_share=1.0,
        # A dozen synthetic neurons make every high-cardinality feature a perfect
        # lookup table; the guard is exercised separately below.
        max_single_feature_accuracy=1.0,
    )
    base.update(overrides)
    return mv_samples.MvSampleBuildConfig(**base)


def _build(tmp_path, objective=CONN, **overrides):
    return samples.build_training_samples("mv", objective, _config(tmp_path, objective, **overrides),
                                          allow_planned=True)


def _kv(sample):
    tokens = sample["input_text"].split()
    return dict(zip(tokens[0::2], tokens[1::2]))


def _by_body(payload):
    return {s["metadata"]["root_id"]: s for s in payload["samples"]}


def test_registration_covers_supported_objectives():
    assert samples.registered_builders()["mv"] == (CONN, NT, SPEC)


def test_planned_dataset_requires_opt_in(tmp_path):
    with pytest.raises(DatasetRegistryError) as exc:
        samples.build_training_samples("mv", CONN, _config(tmp_path))
    assert exc.value.code is DatasetErrorCode.DATASET_NOT_ACTIVE


def test_config_type_and_objective_mismatch_fail_closed(tmp_path):
    with pytest.raises(DatasetRegistryError) as exc:
        samples.build_training_samples("mv", NT, _config(tmp_path, CONN), allow_planned=True)
    assert exc.value.code is DatasetErrorCode.CONFIG_OBJECTIVE_MISMATCH


@pytest.mark.parametrize("objective", ALL)
def test_payload_shape_and_filters(tmp_path, objective):
    payload = _build(tmp_path, objective)
    meta = payload["metadata"]
    assert meta["schema_version"] == "flybrain-mv-training-samples/v1"
    assert meta["dataset_symbol"] == "mv"
    assert meta["dataset_version"] == "manc_v1.0"
    assert meta["objective"] == objective
    assert meta["region_vocabulary"] == "manc_neuropil"
    assert meta["provenance_mode"] == "explicit_paths"
    assert meta["license"] == "CC-BY-4.0"
    assert meta["source_rows"] == 16  # every Traced body (the Orphan is filtered by the Arrow scan)
    stats = meta["filter_stats"]
    assert stats["dropped_untyped"] == 1  # 13
    assert stats["dropped_excluded_class"] == 1  # 14 glia
    assert stats["dropped_no_neuropil_synapses"] == 1  # 16 (CV only)
    ids = set(_by_body(payload))
    assert ids.isdisjoint({str(body(i)) for i in (13, 14, 15, 16)})
    for sample in payload["samples"]:
        assert sample["region_id"].startswith("vnc_")
        assert sample["sample_id"].startswith("manc10-")
        assert sample["metadata"]["cns_division"] == "nerve_cord"
    assert meta["trivial_feature_check"]["sample_count"] == len(payload["samples"])


def test_connectivity_labels_follow_downstream_quantile(tmp_path):
    payload = _build(tmp_path, CONN)
    threshold = payload["metadata"]["high_connectivity_threshold"]
    downstream = [NEURONS[i][4] for i in NEURONS if i not in (13, 14, 15, 16)]
    assert threshold == pytest.approx(float(statistics.median(downstream)))
    for sample in payload["samples"]:
        total = sample["metadata"]["label_source"]["downstream"]
        expected = "high_connectivity" if total >= threshold else "baseline_connectivity"
        assert sample["expected_label"] == expected
        assert 0.5 <= sample["expected_confidence"] <= 0.99
    assert set(payload["metadata"]["label_counts"]) == {"high_connectivity", "baseline_connectivity"}


def test_neurotransmitter_labels_and_unknown_dropped(tmp_path):
    payload = _build(tmp_path, NT)
    by = _by_body(payload)
    assert payload["metadata"]["filter_stats"]["dropped_nt_unknown"] == 1  # 8
    assert str(body(8)) not in by
    assert by[str(body(1))]["expected_label"] == "dominant_ach"
    assert by[str(body(3))]["expected_label"] == "dominant_gaba"
    assert by[str(body(9))]["expected_label"] == "dominant_glut"
    assert by[str(body(1))]["expected_confidence"] == pytest.approx(0.8)
    # out-partner tier from the traced edge list: body 3 has 4 partners -> t0 with edges (10, 100)
    assert _kv(by[str(body(3))])["out_partner_tier"] == "t0"
    tiered = _build(tmp_path, NT, partner_tier_edges=(2, 4))
    assert _kv(_by_body(tiered)[str(body(3))])["out_partner_tier"] == "t2"
    assert _kv(_by_body(tiered)[str(body(12))])["out_partner_tier"] == "t1"
    assert _kv(_by_body(tiered)[str(body(6))])["out_partner_tier"] == "t0"


def test_nt_argmax_disagreement_fails_closed(tmp_path):
    meta = tmp_path / "bad_meta.feather"
    edges = tmp_path / "edges.csv"
    write_meta(meta, probs_override={3: (0.9, 0.05, 0.03, 0.02)})  # predictedNt=gaba, ach is max
    write_edgelist(edges)
    with pytest.raises(MvAdapterError) as exc:
        _build(tmp_path, NT, meta_path=meta, edgelist_path=edges)
    assert exc.value.code is MvAdapterErrorCode.SCHEMA_MISMATCH


def test_region_specialization_labels_follow_top_neuropil_share(tmp_path):
    payload = _build(tmp_path, SPEC, specialization_share_threshold=0.9)
    by = _by_body(payload)
    # body 1: 1000 in LegNp(T1)(L) only -> share 1.0
    assert by[str(body(1))]["expected_label"] == "region_specialized"
    # body 3: 900 / 950 = 0.947 -> specialized (side-specific ROIs: LegNp(T1)(R) counts separately)
    assert by[str(body(3))]["metadata"]["label_source"]["top_neuropil_share"] == pytest.approx(900 / 950, abs=1e-6)
    assert by[str(body(3))]["expected_label"] == "region_specialized"
    # body 5: LTct 600 / (600 + 500); the 900 CV (connective) synapses are excluded
    assert by[str(body(5))]["metadata"]["label_source"]["top_neuropil_share"] == pytest.approx(600 / 1100, abs=1e-6)
    assert by[str(body(5))]["expected_label"] == "region_distributed"
    assert by[str(body(5))]["region_id"] == "vnc_ltct"
    # body 7: ProLN(L) is a nerve, excluded -> 200 / 200
    assert by[str(body(7))]["metadata"]["label_source"]["top_neuropil_share"] == pytest.approx(1.0)
    for sample in payload["samples"]:
        share = sample["metadata"]["label_source"]["top_neuropil_share"]
        assert sample["expected_label"] == ("region_specialized" if share >= 0.9 else "region_distributed")
    assert payload["metadata"]["specialization_share_threshold"] == 0.9


def test_primary_neuropil_region_ids(tmp_path):
    payload = _build(tmp_path, CONN)
    by = _by_body(payload)
    assert by[str(body(1))]["region_id"] == "vnc_legnp_t1"
    assert by[str(body(2))]["region_id"] == "vnc_legnp_t1"
    assert by[str(body(10))]["region_id"] == "vnc_anm"
    assert by[str(body(10))]["metadata"]["primary_neuropil_roi"] == "ANm"
    assert _kv(by[str(body(10))])["primary_neuropil"] == "anm"
    assert set(payload["metadata"]["selected_regions"]) == {"vnc_anm", "vnc_legnp_t1", "vnc_ltct"}


@pytest.mark.parametrize("objective", ALL)
def test_input_text_excludes_label_sources_and_ids(tmp_path, objective):
    payload = _build(tmp_path, objective)
    for sample in payload["samples"]:
        kv = _kv(sample)
        assert list(kv) == ["dataset", *mv_samples.INPUT_FEATURES[objective]]
        assert not set(kv) & mv_samples.FORBIDDEN_INPUT_FEATURES[objective]
        assert "root" not in kv
        assert sample["metadata"]["root_id"] not in sample["input_text"]
        assert kv["dataset"] == "manc10"
    assert payload["metadata"]["leakage_check"] == "passed"
    if objective in (NT, SPEC):
        assert "hemilineage" not in mv_samples.INPUT_FEATURES[objective]
    if objective == CONN:
        assert "out_partner_tier" in mv_samples.FORBIDDEN_INPUT_FEATURES[CONN]


def test_leakage_guard_rejects_forbidden_and_unexpected_keys():
    good = {"sample_id": "s", "input_text": "dataset manc10 primary_neuropil anm"}
    mv_samples.assert_no_label_leakage([good], SPEC)
    for text in ("dataset manc10 subclass ir", "dataset manc10 root 10001", "dataset manc10 hemilineage 09a",
                 "dataset manc10 favourite_colour blue", "dataset manc10 dangling"):
        with pytest.raises(ValueError):
            mv_samples.assert_no_label_leakage([{"sample_id": "s", "input_text": text}], SPEC)
    with pytest.raises(ValueError):
        mv_samples.assert_no_label_leakage(
            [{"sample_id": "s", "input_text": "dataset manc10 hemilineage 09a"}], NT
        )
    mv_samples.assert_no_label_leakage([{"sample_id": "s", "input_text": "dataset manc10 hemilineage 09a"}], CONN)


def test_trivial_feature_check_fails_on_derivable_label():
    rows = []
    for i in range(40):
        klass = "motor_neuron" if i % 2 else "ascending_neuron"
        rows.append({
            "sample_id": f"s{i}",
            "region_id": "vnc_anm",
            "input_text": f"dataset manc10 primary_neuropil anm soma_neuromere t1 soma_side left class {klass} "
                          f"birthtime primary",
            "expected_label": "region_specialized" if i % 2 else "region_distributed",
        })
    with pytest.raises(ValueError, match="trivially derivable.*class"):
        mv_samples.trivial_feature_check(rows, SPEC, max_accuracy=0.9)
    report = mv_samples.trivial_feature_check(rows, SPEC, max_accuracy=1.0)
    assert report["best_single_feature"] == "class"
    assert report["single_feature_lookup_accuracy"]["class"] == 1.0
    assert report["majority_accuracy"] == 0.5
    assert report["threshold_stump"] is None  # categorical inputs only


@pytest.mark.parametrize("objective", ALL)
def test_builder_wires_trivial_guard(tmp_path, objective):
    payload = _build(tmp_path, objective)
    check = payload["metadata"]["trivial_feature_check"]
    assert set(check["single_feature_lookup_accuracy"]) == {*mv_samples.INPUT_FEATURES[objective], "region_id"}
    assert check["threshold_stump"] is None
    with pytest.raises(ValueError, match="trivially derivable"):
        _build(tmp_path, objective, max_single_feature_accuracy=check["best_single_feature_accuracy"] - 0.01)


def test_split_group_metadata_outside_input(tmp_path):
    payload = _build(tmp_path, CONN)
    by = _by_body(payload)
    assert by[str(body(1))]["metadata"]["cell_type"] == "IN01A001"
    assert by[str(body(1))]["metadata"]["hemilineage"] == "01a"
    assert by[str(body(1))]["metadata"]["split_group"] == "manc_group_100"
    # "TBD" hemilineage and a missing group must not tie neurons together
    assert by[str(body(7))]["metadata"]["hemilineage"] == "unknown"
    assert by[str(body(7))]["metadata"]["split_group"] == "unknown"
    for sample in payload["samples"]:
        kv = _kv(sample)
        assert "cell_type" not in kv and "split_group" not in kv


@pytest.mark.parametrize("objective", ALL)
def test_grouped_split_keeps_groups_together(tmp_path, objective):
    payload = _build(tmp_path, objective)
    training = [
        fbct.TrainingSample(**{k: s[k] for k in ("sample_id", "region_id", "input_text", "expected_label",
                                                  "expected_confidence", "provenance_refs", "metadata")})
        for s in payload["samples"]
    ]
    keys = (fbct.GENERIC_SPLIT_GROUP_KEY, *split_group_keys("mv"))
    manifest = fbct.build_dataset_manifest(training, split_seed="mv-test", group_keys=keys)
    fbct.assert_grouped_split(training, manifest, group_keys=keys)
    assert manifest.notes["split"]["multi_sample_components"] >= 1


@pytest.mark.parametrize("objective", ALL)
def test_output_is_deterministic(tmp_path, objective):
    first = _build(tmp_path, objective)
    second = _build(tmp_path, objective)
    assert samples.stable_json(first) == samples.stable_json(second)


def test_string_sort_handles_mixed_width_body_ids(tmp_path):
    payload = _build(tmp_path, CONN)
    legnp = [s["metadata"]["root_id"] for s in payload["samples"] if s["region_id"] == "vnc_legnp_t1"]
    assert str(BIG_BODY) in legnp
    assert legnp == sorted(legnp, key=lambda b: b.zfill(20))


def test_manifest_snapshot_mode(tmp_path):
    snap, manifest = build_snapshot(tmp_path / "root")
    config = mv_samples.MvSampleBuildConfig(
        objective=NT, storage_root=tmp_path / "root", max_samples=50, min_total_count=1, min_region_samples=1,
        max_label_share=1.0, max_single_feature_accuracy=1.0,
    )
    payload = samples.build_training_samples("mv", NT, config, allow_planned=True)
    meta = payload["metadata"]
    assert meta["provenance_mode"] == "manifest_snapshot"
    assert meta["manifest_sha256"] == manifest["integrity"]["manifest_sha256"]
    assert set(meta["product_sha256"]) == {"meta", "edgelist"}
    assert meta["hash_verification"]["metadata/manc-v1.0-neuron-properties.feather"] == "hashed"
    assert (snap / HASH_STAMP_RELATIVE_DIR).is_dir()
    # connectivity needs only the meta product: the edge list is size-checked only
    conn = samples.build_training_samples("mv", CONN, dataclasses.replace(config, objective=CONN),
                                          allow_planned=True)
    assert set(conn["metadata"]["product_sha256"]) == {"meta"}
    assert conn["metadata"]["hash_verification"][
        "source/manc-traced-adjacencies-v1.0/traced-connections.csv"] == "size_only"


def test_manifest_mode_read_only_without_stamp_cache(tmp_path):
    snap, _ = build_snapshot(tmp_path / "root")
    config = mv_samples.MvSampleBuildConfig(
        objective=SPEC, storage_root=tmp_path / "root", max_samples=50, min_total_count=1, min_region_samples=1,
        max_label_share=1.0, max_single_feature_accuracy=1.0, use_stamp_cache=False,
    )
    samples.build_training_samples("mv", SPEC, config, allow_planned=True)
    assert not (snap / HASH_STAMP_RELATIVE_DIR).exists()


def test_manifest_integrity_failure_propagates(tmp_path):
    snap, _ = build_snapshot(tmp_path / "root")
    (snap / "manifest" / "manifest.sha256").write_text("0" * 64 + "\n")
    config = mv_samples.MvSampleBuildConfig(objective=CONN, storage_root=tmp_path / "root")
    with pytest.raises(MvAdapterError) as exc:
        samples.build_training_samples("mv", CONN, config, allow_planned=True)
    assert exc.value.code is MvAdapterErrorCode.INTEGRITY_MISMATCH


def test_fingerprint_tracks_config_and_inputs(tmp_path):
    base = _build(tmp_path, CONN)["metadata"]["input_fingerprint"]
    assert _build(tmp_path, CONN, high_connectivity_quantile=0.6)["metadata"]["input_fingerprint"] != base
    other = tmp_path / "other"
    other.mkdir()
    meta, edges = _paths(other)
    changed = dict(NEURONS)
    changed[1] = (*NEURONS[1][:4], 3001, *NEURONS[1][5:])
    write_meta(meta, neurons=changed)
    assert _build(other, CONN)["metadata"]["input_fingerprint"] != base


def test_partial_explicit_paths_rejected(tmp_path):
    meta, _ = _paths(tmp_path)
    config = mv_samples.MvSampleBuildConfig(objective=NT, meta_path=meta)
    with pytest.raises(ValueError, match="every required product"):
        mv_samples.build_mv_training_samples(NT, config)
    missing = mv_samples.MvSampleBuildConfig(objective=CONN, meta_path=tmp_path / "absent.feather")
    with pytest.raises(MvAdapterError) as exc:
        mv_samples.build_mv_training_samples(CONN, missing)
    assert exc.value.code is MvAdapterErrorCode.PRODUCT_MISSING


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_samples": 0},
        {"min_total_count": 0},
        {"min_region_samples": 0},
        {"high_connectivity_quantile": 1.0},
        {"specialization_share_threshold": 0.0},
        {"max_regions": 0},
        {"min_distinct_labels": 0},
        {"max_label_share": 0.0},
        {"max_single_feature_accuracy": 0.0},
        {"statuses": ()},
        {"partner_tier_edges": (10, 5)},
    ],
)
def test_config_validation(tmp_path, overrides):
    # Every validation message starts with the parameter it rejects, so a
    # deleted check cannot pass on some later, unrelated ValueError.
    (key,) = overrides
    with pytest.raises(ValueError, match=rf"^{key} must "):
        mv_samples.build_mv_training_samples(CONN, _config(tmp_path, CONN, **overrides))


def test_label_concentration_gate_applies(tmp_path):
    with pytest.raises(ValueError, match="label concentration"):
        _build(tmp_path, NT, max_label_share=0.4)


def test_cli_main_writes_payload(tmp_path, capsys):
    build_snapshot(tmp_path / "root")
    out = tmp_path / "out" / "samples.json"
    code = mv_samples.main([
        "--storage-root", str(tmp_path / "root"), "--objective", SPEC, "--output", str(out),
        "--min-total-count", "1", "--min-region-samples", "1", "--allow-planned", "--no-stamp-cache",
        "--max-label-share", "1.0", "--max-single-feature-accuracy", "0.5",
    ])
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    # A tripped single-feature guard is a structured error, not a crash.
    assert code == 2 and printed["status"] == "error" and "trivially derivable" in printed["error"]
    assert not out.exists()
    code = mv_samples.main(["--storage-root", str(tmp_path / "root"), "--objective", SPEC, "--output", str(out)])
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 2 and "planned" in printed["error"]
    code = mv_samples.main([
        "--storage-root", str(tmp_path / "root"), "--objective", SPEC, "--output", str(out),
        "--min-total-count", "1", "--min-region-samples", "1", "--allow-planned", "--no-stamp-cache",
        "--max-label-share", "1.0", "--max-single-feature-accuracy", "1.0",
    ])
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 0 and printed["status"] == "ok"
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["metadata"]["objective"] == SPEC
    assert printed["sample_count"] == len(written["samples"])


def test_builder_has_no_network_imports():
    source = Path(mv_samples.__file__).read_text(encoding="utf-8")
    for token in ("requests", "urllib", "http.client", "socket", "curl", "neuprint"):
        assert f"import {token}" not in source


# ---------------------------------------------------------------------------
# Real snapshot (read-only; skipped unless LOCI_FLYBRAIN_STORAGE_ROOT holds it)
# ---------------------------------------------------------------------------

_REAL_ROOT = os.environ.get("LOCI_FLYBRAIN_STORAGE_ROOT", "")
_REAL_SNAPSHOT = Path(_REAL_ROOT) / "snapshots" / "mv" / "manc_v1.0"


@pytest.mark.skipif(
    not _REAL_ROOT or not (_REAL_SNAPSHOT / "manifest" / "manifest.json").is_file(),
    reason="real mv snapshot not present under LOCI_FLYBRAIN_STORAGE_ROOT",
)
@pytest.mark.parametrize("objective", ALL)
def test_real_snapshot_smoke(objective):
    stamps_before = (_REAL_SNAPSHOT / HASH_STAMP_RELATIVE_DIR).exists()
    config = mv_samples.MvSampleBuildConfig(objective=objective, storage_root=_REAL_ROOT, verify_hashes=False,
                                            use_stamp_cache=False, max_samples=2000)
    payload = samples.build_training_samples("mv", objective, config, allow_planned=True)
    meta = payload["metadata"]
    assert meta["selected_count"] == 2000
    assert meta["source_rows"] > 20000
    assert len(meta["selected_regions"]) == 13
    assert meta["dominant_label_share"] <= 0.9
    check = meta["trivial_feature_check"]
    assert check["best_single_feature_accuracy"] <= 0.9
    assert check["threshold_stump"] is None
    assert (_REAL_SNAPSHOT / HASH_STAMP_RELATIVE_DIR).exists() == stamps_before
