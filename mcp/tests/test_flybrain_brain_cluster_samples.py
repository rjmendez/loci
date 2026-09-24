import os
import sys
from dataclasses import dataclass

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import flybrain_brain_cluster_fw_samples as fw_samples  # noqa: E402
import flybrain_brain_cluster_samples as samples  # noqa: E402
import flybrain_brain_cluster_training as fbct  # noqa: E402
from flybrain_dataset_registry import DatasetErrorCode, DatasetRegistryError  # noqa: E402
from test_flybrain_brain_cluster_fw_samples import _write_fixture, _write_nt_fixture  # noqa: E402


@dataclass(frozen=True)
class _StubConfig:
    objective: str = "connectivity_tier"
    max_samples: int = 10
    symbol: str = "banc"


def _stub_builder(objective, config):
    rows = [
        {
            "sample_id": f"stub-{i}",
            "region_id": samples.sanitize_region(region),
            "input_text": f"dataset stub root {i} region {region}",
            "expected_label": label,
            "expected_confidence": 0.7,
            "provenance_refs": [f"stub:root_id:{i}"],
            "metadata": {"dataset": "stub"},
        }
        for i, (region, label) in enumerate(
            [("GNG", "high_connectivity"), ("GNG", "baseline_connectivity"), ("LegNp(T1)_L", "baseline_connectivity")]
        )
    ]
    capped = samples.balance_and_cap_samples(rows, max_samples=config.max_samples)
    return samples.assemble_training_payload(
        capped,
        symbol=config.symbol,
        objective=objective,
        source_path="synthetic",
        source_rows=3,
        candidate_rows=3,
        min_distinct_labels=2,
        max_label_share=0.9,
        fingerprint_inputs={"max_samples": config.max_samples},
    )


@pytest.fixture(autouse=True)
def _restore_builder_registry():
    """Snapshot/restore the global builder table so stub registrations in this
    module never clobber (or get blocked by) real dataset builders that other
    test modules imported, whatever the collection order."""
    saved = dict(samples._BUILDERS)
    yield
    samples._BUILDERS.clear()
    samples._BUILDERS.update(saved)


@pytest.fixture
def stub_banc():
    samples.unregister_sample_builder("banc", "connectivity_tier")
    samples.register_sample_builder("banc", ["connectivity_tier"], _stub_builder, config_type=_StubConfig)
    yield
    samples.unregister_sample_builder("banc", "connectivity_tier")


def _fw_config(tmp_path, **overrides):
    source = tmp_path / "per_neuron_neuropil_count_pre_783.feather"
    _write_fixture(source)
    base = dict(
        source_path=source,
        max_samples=6,
        min_total_count=1,
        min_region_samples=1,
        high_connectivity_quantile=0.5,
        max_regions=5,
    )
    base.update(overrides)
    return fw_samples.FwSampleBuildConfig(**base)


def test_fw_dispatch_is_identical_to_direct_builder(tmp_path):
    config = _fw_config(tmp_path)
    assert samples.build_training_samples("fw", "connectivity_tier", config) == fw_samples.build_fw_training_samples(config)


def test_fw_dispatch_neurotransmitter(tmp_path):
    pre = tmp_path / "per_neuron_neuropil_count_pre_783.feather"
    nt = tmp_path / "proofread_connections_783.feather"
    _write_fixture(pre)
    _write_nt_fixture(nt)
    config = fw_samples.FwSampleBuildConfig(
        objective="neurotransmitter_dominance",
        source_path=nt,
        companion_pre_path=pre,
        max_samples=10,
        min_total_count=1,
        min_region_samples=1,
        max_regions=5,
    )
    payload = samples.build_training_samples("FW", "neurotransmitter_dominance", config)
    assert payload == fw_samples.build_fw_training_samples(config)


def test_fw_payload_accepted_by_training_normalizer(tmp_path):
    payload = samples.build_training_samples("fw", "connectivity_tier", _fw_config(tmp_path))
    rows = [fbct._normalize_training_sample(item) for item in fbct._extract_items(payload, key="samples")]
    assert len(rows) == payload["metadata"]["selected_count"]


def test_fw_registered_objectives_follow_registry():
    assert samples.registered_builders()["fw"] == ("connectivity_tier", "neurotransmitter_dominance")
    assert fw_samples.FW_SUPPORTED_OBJECTIVES == ("connectivity_tier", "neurotransmitter_dominance")


def test_fw_legacy_validation_message_unchanged():
    with pytest.raises(ValueError, match="objective must be one of: connectivity_tier, neurotransmitter_dominance"):
        fw_samples.build_fw_training_samples(fw_samples.FwSampleBuildConfig(objective="nope"))


def test_config_objective_mismatch_fails_closed(tmp_path):
    with pytest.raises(DatasetRegistryError) as exc:
        samples.build_training_samples("fw", "neurotransmitter_dominance", _fw_config(tmp_path))
    assert exc.value.code is DatasetErrorCode.CONFIG_OBJECTIVE_MISMATCH


def test_config_type_mismatch_fails_closed():
    with pytest.raises(DatasetRegistryError) as exc:
        samples.build_training_samples("fw", "connectivity_tier", _StubConfig())
    assert exc.value.code is DatasetErrorCode.CONFIG_TYPE_MISMATCH


def test_objective_not_in_dataset_allowlist():
    with pytest.raises(DatasetRegistryError) as exc:
        samples.build_training_samples("fw", "region_specialization_tier")
    assert exc.value.code is DatasetErrorCode.OBJECTIVE_NOT_SUPPORTED
    with pytest.raises(DatasetRegistryError) as exc:
        samples.register_sample_builder("l1em", ["neurotransmitter_dominance"], _stub_builder, config_type=_StubConfig)
    assert exc.value.code is DatasetErrorCode.OBJECTIVE_NOT_SUPPORTED


def test_planned_dataset_requires_opt_in(stub_banc):
    with pytest.raises(DatasetRegistryError) as exc:
        samples.build_training_samples("banc", "connectivity_tier", _StubConfig())
    assert exc.value.code is DatasetErrorCode.DATASET_NOT_ACTIVE
    payload = samples.build_training_samples("BANC", "connectivity_tier", _StubConfig(), allow_planned=True)
    assert payload["metadata"]["schema_version"] == "flybrain-banc-training-samples/v1"
    assert payload["metadata"]["dataset_symbol"] == "banc"
    assert payload["metadata"]["selected_regions"] == ["gng", "legnp_t1_l"]
    again = samples.build_training_samples("banc", "connectivity_tier", None, allow_planned=True)
    assert again["metadata"]["input_fingerprint"] == payload["metadata"]["input_fingerprint"]
    assert [s["sample_id"] for s in again["samples"]] == [s["sample_id"] for s in payload["samples"]]


def test_missing_builder_fails_closed():
    with pytest.raises(DatasetRegistryError) as exc:
        samples.build_training_samples("hb", "connectivity_tier")
    assert exc.value.code is DatasetErrorCode.BUILDER_NOT_REGISTERED


def test_duplicate_registration_rejected(stub_banc):
    with pytest.raises(DatasetRegistryError) as exc:
        samples.register_sample_builder("banc", ["connectivity_tier"], lambda o, c: {}, config_type=_StubConfig)
    assert exc.value.code is DatasetErrorCode.BUILDER_ALREADY_REGISTERED


def test_lazy_module_import_registers_builder(monkeypatch):
    target = "flybrain_brain_cluster_mc_samples"
    imported = []

    def _fake_import(name):
        imported.append(name)
        samples.register_sample_builder("mc", ["connectivity_tier"], _stub_builder, config_type=_StubConfig)

    monkeypatch.setattr(samples.importlib.util, "find_spec", lambda name: object() if name == target else None)
    monkeypatch.setattr(samples.importlib, "import_module", _fake_import)
    try:
        payload = samples.build_training_samples(
            "mc", "connectivity_tier", _StubConfig(symbol="mc"), allow_planned=True
        )
        assert imported == [target]
        assert payload["samples"]
        with pytest.raises(DatasetRegistryError) as exc:
            samples.build_training_samples(
                "mc",
                "region_specialization_tier",
                _StubConfig(objective="region_specialization_tier"),
                allow_planned=True,
            )
        assert exc.value.code is DatasetErrorCode.BUILDER_NOT_REGISTERED
    finally:
        samples.unregister_sample_builder("mc", "connectivity_tier")


def test_payload_validation_rejects_bad_shapes():
    good = _stub_builder("connectivity_tier", _StubConfig())
    samples.validate_training_payload(good, objective="connectivity_tier")
    truncated = {"samples": [dict(good["samples"][0])], "metadata": dict(good["metadata"])}
    with pytest.raises(DatasetRegistryError) as exc:
        samples.validate_training_payload(truncated)
    assert exc.value.code is DatasetErrorCode.PAYLOAD_INVALID
    dup = {
        "samples": good["samples"] + [good["samples"][0]],
        "metadata": {**good["metadata"], "selected_count": 4},
    }
    with pytest.raises(DatasetRegistryError, match="duplicate sample_id"):
        samples.validate_training_payload(dup)
    with pytest.raises(DatasetRegistryError, match="objective"):
        samples.validate_training_payload(good, objective="neurotransmitter_dominance")


def test_assemble_rejects_label_concentration():
    rows = _stub_builder("connectivity_tier", _StubConfig())["samples"]
    with pytest.raises(ValueError, match="label concentration too high"):
        samples.assemble_training_payload(
            rows,
            symbol="banc",
            objective="connectivity_tier",
            source_path="s",
            source_rows=3,
            candidate_rows=3,
            min_distinct_labels=2,
            max_label_share=0.5,
            fingerprint_inputs={},
        )


def test_payload_stamped_for_another_dataset_rejected(stub_banc):
    samples.register_sample_builder(
        "mc", ["connectivity_tier"], _stub_builder, config_type=_StubConfig
    )
    with pytest.raises(DatasetRegistryError) as exc:
        samples.build_training_samples("mc", "connectivity_tier", _StubConfig(symbol="banc"), allow_planned=True)
    assert exc.value.code is DatasetErrorCode.PAYLOAD_INVALID


def _id_ordered_samples(n_per_region: int = 200) -> list[dict]:
    # Low ids are "big" neurons (label hi), as in FlyEM body ids.
    out = []
    for region in ("r_a", "r_b"):
        for index in range(n_per_region):
            body = 10_000 + index
            out.append({
                "sample_id": f"t-{region}-{body}",
                "region_id": region,
                "expected_label": "hi" if index < n_per_region // 4 else "lo",
                "metadata": {"root_id": str(body)},
            })
    return out


def test_hash_ordered_preselect_removes_id_order_bias():
    pool = _id_ordered_samples()
    by_id_order = samples.balance_and_cap_samples(pool, max_samples=100)
    assert all(s["expected_label"] == "hi" for s in by_id_order)  # the bias being fixed
    picked = samples.hash_ordered_preselect(pool, max_samples=100, salt="t",
                                            id_of=lambda s: s["metadata"]["root_id"])
    assert len(picked) == 100
    assert {s["region_id"] for s in picked} == {"r_a", "r_b"}
    hi_share = sum(s["expected_label"] == "hi" for s in picked) / 100.0
    assert 0.1 < hi_share < 0.45
    again = samples.hash_ordered_preselect(list(reversed(pool)), max_samples=100, salt="t",
                                           id_of=lambda s: s["metadata"]["root_id"])
    assert [s["sample_id"] for s in again] == [s["sample_id"] for s in picked]
    capped = samples.balance_and_cap_samples(picked, max_samples=100)
    assert {s["sample_id"] for s in capped} == {s["sample_id"] for s in picked}
    assert len(samples.hash_ordered_preselect(pool, max_samples=10_000, salt="t")) == len(pool)
