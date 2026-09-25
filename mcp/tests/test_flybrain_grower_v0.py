"""Tests for flybrain_grower_v0 — genome_from_data and sample_connectome."""
import json, math, tempfile, os
from pathlib import Path
import numpy as np
import pytest
from flybrain_grower_v0 import genome_from_data, validate_genome, sample_connectome, GENOME_SCHEMA_VERSION

MINIMAL_STATS = {
    "schema_version": "l1em-type-stats/v1",
    "dataset": "test",
    "snapshot": "test_snap",
    "cell_types": ["A", "B", "C"],
    "neuron_counts": {"A": 10, "B": 20, "C": 5},
    "connection_probability": {
        "A": {"A": 0.5, "B": 0.3, "C": 0.1},
        "B": {"A": 0.2, "B": 0.6, "C": 0.2},
        "C": {"A": 0.1, "B": 0.4, "C": 0.7},
    },
    "mean_synapse_count": {
        "A": {"A": 2.0, "B": 1.5, "C": 1.0},
        "B": {"A": 1.8, "B": 3.0, "C": 1.2},
        "C": {"A": 1.0, "B": 2.0, "C": 2.5},
    },
    "std_synapse_count": {
        "A": {"A": 3.0, "B": 2.0, "C": 1.5},
        "B": {"A": 2.5, "B": 4.0, "C": 1.8},
        "C": {"A": 1.5, "B": 3.0, "C": 3.5},
    },
}


@pytest.fixture
def stats_file(tmp_path):
    p = tmp_path / "l1em_type_stats.json"
    p.write_text(json.dumps(MINIMAL_STATS))
    return p


def test_genome_from_data_returns_valid_genome(stats_file):
    g = genome_from_data(stats_file)
    assert g["schema_version"] == GENOME_SCHEMA_VERSION
    assert set(g["cell_types"]) == {"A", "B", "C"}
    assert g["type_counts"] == {"A": 10, "B": 20, "C": 5}
    assert g["metadata"]["constraint_tier"] == "fly_constrained"
    validate_genome(g)  # must not raise


def test_genome_from_data_maps_connection_probs(stats_file):
    g = genome_from_data(stats_file)
    assert 0.0 <= g["connection_probs"]["A"]["B"] <= 1.0
    assert math.isclose(g["connection_probs"]["A"]["B"], 0.3)


def test_genome_from_data_maps_synapse_params(stats_file):
    g = genome_from_data(stats_file)
    p = g["synapse_count_params"]["B"]["B"]
    assert "mean" in p and "std" in p
    assert p["mean"] > 0.0 and p["std"] > 0.0


def test_genome_from_data_includes_nt_type(stats_file):
    g = genome_from_data(stats_file)
    assert "nt_type" in g
    assert set(g["nt_type"].keys()) == set(g["cell_types"])
    assert all(value == "unknown" for value in g["nt_type"].values())


def test_genome_from_data_raises_on_missing_field(tmp_path):
    bad = dict(MINIMAL_STATS)
    del bad["neuron_counts"]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="missing required fields"):
        genome_from_data(p)


def test_validate_genome_raises_on_bad_prob(stats_file):
    g = genome_from_data(stats_file)
    g["connection_probs"]["A"]["B"] = 1.5  # invalid
    with pytest.raises(ValueError, match="not in"):
        validate_genome(g)


def test_sample_connectome_structure(stats_file):
    g = genome_from_data(stats_file)
    c = sample_connectome(g, seed=42)
    assert c["n_neurons"] == 35  # 10+20+5
    assert isinstance(c["adjacency"], list)
    assert len(c["node_types"]) == 35
    assert c["seed"] == 42
    # adjacency entries are (pre, post, weight) with weight > 0
    for pre, post, w in c["adjacency"]:
        assert 0 <= pre < 35 and 0 <= post < 35
        assert w > 0


def test_sample_connectome_no_autapses(stats_file):
    g = genome_from_data(stats_file)
    c = sample_connectome(g, seed=7)
    type_map = c["cell_type_map"]
    for pre, post, _ in c["adjacency"]:
        # same neuron id is never pre and post
        assert pre != post
        # within a same-type block the same neuron cannot connect to itself
        # (diagonal is zeroed; different neurons in same block are fine)


def test_sample_connectome_deterministic(stats_file):
    g = genome_from_data(stats_file)
    c1 = sample_connectome(g, seed=0)
    c2 = sample_connectome(g, seed=0)
    assert c1["adjacency"] == c2["adjacency"]


def test_sample_connectome_different_seeds_differ(stats_file):
    g = genome_from_data(stats_file)
    c1 = sample_connectome(g, seed=1)
    c2 = sample_connectome(g, seed=2)
    # Very unlikely to be identical
    assert c1["adjacency"] != c2["adjacency"]
