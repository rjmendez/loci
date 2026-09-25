"""Structured-feature extension of flybrain_brain_cluster_mv_samples (attach_features / collect_mv_samples)."""

import dataclasses
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import flybrain_brain_cluster_mv_samples as mv_samples  # noqa: E402
import flybrain_brain_cluster_samples as samples  # noqa: E402
import flybrain_model_eval as fme  # noqa: E402
import flybrain_wiring_features as fwf  # noqa: E402
from test_flybrain_mv_adapter import write_edgelist, write_meta  # noqa: E402

mv_samples.unregister_mv_sample_builders()
ALL = ("connectivity_tier", "neurotransmitter_dominance", "region_specialization_tier")


@pytest.fixture(autouse=True)
def _mv_registered():
    mv_samples.register_mv_sample_builders(replace=True)
    yield
    mv_samples.unregister_mv_sample_builders()


def _config(tmp_path, objective, **overrides):
    meta, edges = tmp_path / "meta.feather", tmp_path / "edges.csv"
    if not meta.exists():
        write_meta(meta)
        write_edgelist(edges)
    base = dict(objective=objective, meta_path=meta, edgelist_path=edges, max_samples=50, min_total_count=1,
                min_region_samples=1, high_connectivity_quantile=0.5, max_regions=10, max_label_share=1.0,
                max_single_feature_accuracy=1.0)
    base.update(overrides)
    return mv_samples.MvSampleBuildConfig(**base)


@pytest.mark.parametrize("objective", ALL)
def test_default_payload_has_no_features_and_same_fingerprint(tmp_path, objective):
    legacy = samples.build_training_samples("mv", objective, _config(tmp_path, objective), allow_planned=True)
    assert all("features" not in s for s in legacy["samples"])
    again = samples.build_training_samples("mv", objective, _config(tmp_path, objective), allow_planned=True)
    assert legacy == again


@pytest.mark.parametrize("objective", ALL)
def test_attach_features_adds_allowed_categoricals(tmp_path, objective):
    payload = samples.build_training_samples("mv", objective, _config(tmp_path, objective, attach_features=True),
                                             allow_planned=True)
    plain = samples.build_training_samples("mv", objective, _config(tmp_path, objective), allow_planned=True)
    assert payload["metadata"]["input_fingerprint"] != plain["metadata"]["input_fingerprint"]
    for sample in payload["samples"]:
        assert set(sample["features"]) == set(mv_samples.STRUCTURED_FEATURES[objective])
        assert not set(sample["features"]) & mv_samples.FORBIDDEN_INPUT_FEATURES[objective]
    fwf.assert_features_allowed(list(mv_samples.STRUCTURED_FEATURES[objective]), objective)
    data = fme.EvalDataset.from_samples(payload["samples"], dataset="mv", target=objective,
                                        group_keys=("cell_type", "hemilineage", "split_group"))
    assert list(data.features.columns) == sorted(mv_samples.STRUCTURED_FEATURES[objective])


@pytest.mark.parametrize("objective", ALL)
def test_collect_returns_every_candidate_uncapped(tmp_path, objective):
    config = _config(tmp_path, objective, max_samples=6)
    collected = mv_samples.collect_mv_samples(objective, config)
    capped = samples.build_training_samples("mv", objective, config, allow_planned=True)
    assert len(collected["samples"]) >= len(capped["samples"])
    assert len(capped["samples"]) <= 6
    assert all("features" in s for s in collected["samples"])


def test_collect_rejects_objective_mismatch(tmp_path):
    with pytest.raises(ValueError, match="config.objective"):
        mv_samples.collect_mv_samples("connectivity_tier", _config(tmp_path, "neurotransmitter_dominance"))


def test_region_structured_features_exclude_primary_neuropil():
    feats = mv_samples.STRUCTURED_FEATURES["region_specialization_tier"]
    assert "primary_neuropil" not in feats
    assert fwf.excluded_feature_names(["primary_neuropil"], "region_specialization_tier") == ["primary_neuropil"]


def test_load_candidates_matches_builder_candidates(tmp_path):
    config = _config(tmp_path, "connectivity_tier")
    loaded = mv_samples.load_mv_candidates(config)
    assert "class_" in loaded["frame"].columns
    assert loaded["stats"]["dropped_untyped"] == 1
    assert dataclasses.replace(config).attach_features is False
