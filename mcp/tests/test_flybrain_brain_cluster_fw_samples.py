import os
import sys

import pyarrow as pa
import pyarrow.feather as feather

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import flybrain_brain_cluster_fw_samples as fw_samples  # noqa: E402


def _write_fixture(path):
    table = pa.table(
        {
            "pre_pt_root_id": [101, 101, 102, 102, 103, 103, 104, 104, 105, 105, 106, 106],
            "neuropil": ["ME_L", "LO_L", "ME_L", "LO_L", "LO_L", "ME_L", "AL_R", "AL_R", "AL_R", "ME_L", "ME_L", "LO_L"],
            "count": [12, 5, 3, 9, 7, 6, 11, 4, 8, 3, 2, 10],
        }
    )
    feather.write_feather(table, str(path))


def _write_nt_fixture(path):
    table = pa.table(
        {
            "pre_pt_root_id": [101, 101, 102, 102, 103, 104],
            "post_pt_root_id": [201, 202, 203, 204, 205, 206],
            "neuropil": ["ME_L", "LO_L", "AL_R", "AL_R", "LO_L", "ME_L"],
            "syn_count": [8, 5, 10, 2, 9, 7],
            "ach_avg": [0.8, 0.75, 0.1, 0.12, 0.2, 0.15],
            "gaba_avg": [0.05, 0.08, 0.7, 0.65, 0.1, 0.2],
            "glut_avg": [0.1, 0.1, 0.1, 0.1, 0.6, 0.5],
            "da_avg": [0.02, 0.02, 0.03, 0.04, 0.03, 0.05],
            "ser_avg": [0.01, 0.02, 0.04, 0.05, 0.04, 0.04],
            "oct_avg": [0.02, 0.03, 0.03, 0.04, 0.03, 0.06],
        }
    )
    feather.write_feather(table, str(path))


def test_build_fw_training_samples_is_deterministic(tmp_path):
    source = tmp_path / "per_neuron_neuropil_count_pre_783.feather"
    _write_fixture(source)
    config = fw_samples.FwSampleBuildConfig(
        source_path=source,
        max_samples=6,
        min_total_count=1,
        min_region_samples=1,
        high_connectivity_quantile=0.5,
        max_regions=5,
    )

    first = fw_samples.build_fw_training_samples(config)
    second = fw_samples.build_fw_training_samples(config)

    assert first["metadata"]["input_fingerprint"] == second["metadata"]["input_fingerprint"]
    assert first["metadata"]["selected_regions"] == second["metadata"]["selected_regions"]
    assert [row["sample_id"] for row in first["samples"]] == [row["sample_id"] for row in second["samples"]]
    assert [row["expected_label"] for row in first["samples"]] == [row["expected_label"] for row in second["samples"]]


def test_build_fw_training_samples_applies_region_and_sample_caps(tmp_path):
    source = tmp_path / "per_neuron_neuropil_count_pre_783.feather"
    _write_fixture(source)
    payload = fw_samples.build_fw_training_samples(
        fw_samples.FwSampleBuildConfig(
            source_path=source,
            max_samples=4,
            min_total_count=1,
            min_region_samples=2,
            high_connectivity_quantile=0.5,
            max_regions=1,
        )
    )
    assert len(payload["samples"]) == 3
    assert payload["metadata"]["selected_regions"] == ["lo_l"]
    assert set(payload["metadata"]["label_counts"]) <= {"baseline_connectivity", "high_connectivity"}


def test_build_fw_training_samples_neurotransmitter_objective(tmp_path):
    pre_source = tmp_path / "per_neuron_neuropil_count_pre_783.feather"
    nt_source = tmp_path / "proofread_connections_783.feather"
    _write_fixture(pre_source)
    _write_nt_fixture(nt_source)

    payload = fw_samples.build_fw_training_samples(
        fw_samples.FwSampleBuildConfig(
            objective="neurotransmitter_dominance",
            source_path=nt_source,
            companion_pre_path=pre_source,
            max_samples=10,
            min_total_count=1,
            min_region_samples=1,
            max_regions=5,
        )
    )
    labels = {row["expected_label"] for row in payload["samples"]}
    assert payload["metadata"]["objective"] == "neurotransmitter_dominance"
    assert labels <= {"dominant_ach", "dominant_gaba", "dominant_glut", "dominant_da", "dominant_ser", "dominant_oct"}
    assert any(label == "dominant_ach" for label in labels)
    assert any(label == "dominant_gaba" for label in labels)


def test_build_fw_training_samples_rejects_overconcentrated_labels(tmp_path):
    source = tmp_path / "per_neuron_neuropil_count_pre_783.feather"
    _write_fixture(source)
    try:
        fw_samples.build_fw_training_samples(
            fw_samples.FwSampleBuildConfig(
                source_path=source,
                max_samples=6,
                min_total_count=1,
                min_region_samples=1,
                high_connectivity_quantile=0.01,
                max_regions=5,
                max_label_share=0.5,
            )
        )
    except ValueError as exc:
        assert "label concentration too high" in str(exc)
        return
    raise AssertionError("Expected concentrated labels to fail guard")
