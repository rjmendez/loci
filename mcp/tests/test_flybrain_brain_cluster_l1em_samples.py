import importlib
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import flybrain_brain_cluster_samples as samples  # noqa: E402
from flybrain_dataset_registry import DatasetErrorCode, DatasetRegistryError  # noqa: E402
from flybrain_l1em_adapter import L1emAdapterError, L1emErrorCode  # noqa: E402
from test_flybrain_l1em_adapter import write_l1em_fixture  # noqa: E402

# The l1em builder module is imported inside a fixture (not at collection time)
# and unregistered on teardown, so the shared dispatcher tests that use "l1em"
# as their lazy-import stub target stay order-independent.
l1em_samples = None
L1EM_FORBIDDEN_INPUT_FEATURES = frozenset()
L1EM_INPUT_FEATURES = ()
L1emSampleBuildConfig = None


@pytest.fixture(scope="module", autouse=True)
def _l1em_builder_registered():
    global l1em_samples, L1EM_FORBIDDEN_INPUT_FEATURES, L1EM_INPUT_FEATURES, L1emSampleBuildConfig
    module = importlib.import_module("flybrain_brain_cluster_l1em_samples")
    if "l1em" not in samples.registered_builders():
        samples.register_sample_builder(
            "l1em",
            ("connectivity_tier",),
            module.build_l1em_training_samples,
            config_type=module.L1emSampleBuildConfig,
        )
    l1em_samples = module
    L1EM_FORBIDDEN_INPUT_FEATURES = module.L1EM_FORBIDDEN_INPUT_FEATURES
    L1EM_INPUT_FEATURES = module.L1EM_INPUT_FEATURES
    L1emSampleBuildConfig = module.L1emSampleBuildConfig
    yield module
    for objective in samples.registered_builders().get("l1em", ()):
        samples.unregister_sample_builder("l1em", objective)


def _config(tmp_path, **overrides):
    values = dict(storage_root=tmp_path, min_total_synapses=1, min_region_samples=2, max_label_share=0.95)
    values.update(overrides)
    return L1emSampleBuildConfig(**values)


def _build(tmp_path, **overrides):
    return samples.build_training_samples(
        "l1em", "connectivity_tier", _config(tmp_path, **overrides), allow_planned=True
    )


def test_builder_registered_for_connectivity_only():
    assert samples.registered_builders()["l1em"] == ("connectivity_tier",)


def test_dispatch_builds_valid_payload(tmp_path):
    write_l1em_fixture(tmp_path)
    payload = _build(tmp_path)
    samples.validate_training_payload(payload, objective="connectivity_tier")
    meta = payload["metadata"]
    assert meta["schema_version"] == "flybrain-l1em-training-samples/v1"
    assert meta["dataset_symbol"] == "l1em"
    assert meta["source_rows"] == 13
    assert meta["candidate_rows"] == 12
    assert meta["unannotated_neurons_excluded"] == 1
    assert meta["selected_regions"] == ["l1_kc", "l1_pn", "l1_pre_dn_vnc"]
    assert set(meta["label_counts"]) == {"high_connectivity", "baseline_connectivity"}
    assert meta["cross_stage_identity_transfer"] == "unsupported"
    assert meta["license"] == "CC-BY-4.0"
    for sample in payload["samples"]:
        assert sample["region_id"].startswith("l1_")
        assert sample["sample_id"].startswith("l1em-skid-")
        assert any(ref.startswith("manifest_sha256:") for ref in sample["provenance_refs"])


def test_output_is_deterministic(tmp_path):
    write_l1em_fixture(tmp_path)
    first = _build(tmp_path)
    second = _build(tmp_path)
    assert first == second
    assert [s["sample_id"] for s in first["samples"]][:3] == ["l1em-skid-101", "l1em-skid-105", "l1em-skid-109"]


def test_label_matches_total_synapse_quantile(tmp_path):
    write_l1em_fixture(tmp_path)
    payload = _build(tmp_path)
    threshold = payload["metadata"]["high_connectivity_threshold"]
    for sample in payload["samples"]:
        total = sample["metadata"]["total_synapses"]
        expected = "high_connectivity" if total >= threshold else "baseline_connectivity"
        assert sample["expected_label"] == expected


def test_input_text_excludes_label_derivable_features(tmp_path):
    write_l1em_fixture(tmp_path)
    payload = _build(tmp_path)
    assert not set(L1EM_INPUT_FEATURES) & L1EM_FORBIDDEN_INPUT_FEATURES
    for sample in payload["samples"]:
        text = sample["input_text"]
        tokens = text.split()
        keys = tokens[4::2]
        assert tokens[:4] == ["dataset", "catmaid_l1em", "stage", "larva_l1"]
        assert tuple(keys) == L1EM_INPUT_FEATURES
        for forbidden in L1EM_FORBIDDEN_INPUT_FEATURES:
            assert forbidden not in keys
        # no raw integer synapse count appears as a value
        meta = sample["metadata"]
        for count_key in ("total_synapses", "in_synapses", "out_synapses"):
            value = str(meta[count_key])
            numeric_values = [v for k, v in zip(keys, tokens[5::2]) if k not in ("cluster",)]
            assert value not in numeric_values
        assert re.search(r"axon_output_fraction (na|[01]\.\d{4})", text)


def test_render_input_text_rejects_extra_features():
    with pytest.raises(L1emAdapterError) as excinfo:
        l1em_samples.render_input_text({"celltype": "KC", "total_synapses": "900"})
    assert excinfo.value.code is L1emErrorCode.SCHEMA_MISMATCH


def test_split_group_pairs_homologs(tmp_path):
    write_l1em_fixture(tmp_path)
    payload = _build(tmp_path)
    groups = {s["metadata"]["skid"]: s["metadata"]["split_group"] for s in payload["samples"]}
    assert groups["101"] == groups["102"] == "l1em-pair-101"
    assert groups["107"] == "l1em-pair-107"


def test_fingerprint_changes_with_config(tmp_path):
    write_l1em_fixture(tmp_path)
    a = _build(tmp_path)["metadata"]["input_fingerprint"]
    b = _build(tmp_path, high_connectivity_quantile=0.5)["metadata"]["input_fingerprint"]
    assert a != b


def test_max_samples_caps_round_robin(tmp_path):
    write_l1em_fixture(tmp_path)
    payload = _build(tmp_path, max_samples=6, max_label_share=1.0, min_distinct_labels=1)
    assert len(payload["samples"]) == 6
    regions = [s["region_id"] for s in payload["samples"]]
    assert regions[:3] == ["l1_kc", "l1_pn", "l1_pre_dn_vnc"]


def test_planned_dataset_requires_allow_planned(tmp_path):
    write_l1em_fixture(tmp_path)
    with pytest.raises(DatasetRegistryError) as excinfo:
        samples.build_training_samples("l1em", "connectivity_tier", _config(tmp_path))
    assert excinfo.value.code is DatasetErrorCode.DATASET_NOT_ACTIVE


def test_neurotransmitter_objective_not_supported():
    with pytest.raises(DatasetRegistryError) as excinfo:
        samples.build_training_samples("l1em", "neurotransmitter_dominance", allow_planned=True)
    assert excinfo.value.code is DatasetErrorCode.OBJECTIVE_NOT_SUPPORTED


def test_region_specialization_not_in_l1em_allow_list():
    # The registry narrows l1em to connectivity_tier, so dispatch refuses before
    # any builder lookup (no per-synapse larval neuropil data exists).
    with pytest.raises(DatasetRegistryError) as excinfo:
        samples.build_training_samples("l1em", "region_specialization_tier", allow_planned=True)
    assert excinfo.value.code is DatasetErrorCode.OBJECTIVE_NOT_SUPPORTED


def test_direct_builder_refuses_unsupported_objective(tmp_path):
    write_l1em_fixture(tmp_path)
    cfg = _config(tmp_path, objective="region_specialization_tier")
    with pytest.raises(L1emAdapterError) as excinfo:
        l1em_samples.build_l1em_training_samples("region_specialization_tier", cfg)
    assert excinfo.value.code is L1emErrorCode.OBJECTIVE_UNSUPPORTED


def test_config_type_and_objective_mismatch(tmp_path):
    with pytest.raises(DatasetRegistryError) as excinfo:
        samples.build_training_samples("l1em", "connectivity_tier", object(), allow_planned=True)
    assert excinfo.value.code is DatasetErrorCode.CONFIG_TYPE_MISMATCH


def test_region_filter_fails_closed(tmp_path):
    write_l1em_fixture(tmp_path)
    with pytest.raises(ValueError, match="min_region_samples"):
        _build(tmp_path, min_region_samples=50)


def test_label_concentration_gate(tmp_path):
    write_l1em_fixture(tmp_path)
    with pytest.raises(ValueError, match="label concentration too high"):
        _build(tmp_path, max_label_share=0.5)


def test_integrity_failure_propagates(tmp_path):
    root = write_l1em_fixture(tmp_path)
    (root / "metadata" / "files" / "outputs.csv").write_text(",axon_output,dendrite_output\n")
    with pytest.raises(L1emAdapterError) as excinfo:
        _build(tmp_path)
    assert excinfo.value.code is L1emErrorCode.INTEGRITY_MISMATCH


@pytest.mark.parametrize(
    "overrides",
    [{"max_samples": 0}, {"min_total_synapses": 0}, {"high_connectivity_quantile": 1.0}, {"max_label_share": 0.0}],
)
def test_config_validation(tmp_path, overrides):
    write_l1em_fixture(tmp_path)
    with pytest.raises(ValueError):
        _build(tmp_path, **overrides)


_REAL_ROOT = os.environ.get("LOCI_FLYBRAIN_STORAGE_ROOT", "")


@pytest.mark.skipif(
    not _REAL_ROOT
    or not (Path(_REAL_ROOT) / "snapshots" / "l1em" / "catmaid_l1em" / "manifest" / "manifest.json").is_file(),
    reason="real l1em snapshot not available",
)
def test_real_snapshot_smoke():
    payload = samples.build_training_samples(
        "l1em", "connectivity_tier", L1emSampleBuildConfig(storage_root=_REAL_ROOT), allow_planned=True
    )
    meta = payload["metadata"]
    assert meta["source_rows"] == 2952
    assert meta["selected_count"] > 2000
    assert meta["dominant_label_share"] <= 0.9
    assert all(r.startswith("l1_") for r in meta["selected_regions"])
