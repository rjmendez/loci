import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flybrain_grower_optimizer import GrowerOptimizer, REALISM_GATE  # noqa: E402


class MockBodyAdapter:
    def __init__(self, reward: float = 0.8) -> None:
        self.reward_value = float(reward)
        self.reward_calls = 0

    def reward(self, *, samples, **_kwargs):
        self.reward_calls += 1
        assert samples
        return self.reward_value


class MockCritic:
    def __init__(self, hit_probability: float) -> None:
        self.hit_probability = float(hit_probability)
        self.classes_ = np.asarray(["ascending", "dn_sez", "dn_vnc", "interneuron", "rgn", "sensory"], dtype=object)
        self.feature_names_in_ = np.asarray(["label_hint"], dtype=object)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        base = (1.0 - self.hit_probability) / (len(self.classes_) - 1)
        proba = np.full((len(frame), len(self.classes_)), base, dtype=np.float64)
        label_index = {label: idx for idx, label in enumerate(self.classes_)}
        for row_index, label in enumerate(frame["label_hint"].tolist()):
            proba[row_index, label_index[str(label)]] = self.hit_probability
        return proba


@pytest.fixture()
def patch_joblib_load(monkeypatch):
    def _install(hit_probability: float) -> None:
        monkeypatch.setattr("flybrain_grower_optimizer.joblib.load", lambda _path: MockCritic(hit_probability))

    return _install


def _optimizer(monkeypatch, patch_joblib_load, hit_probability: float = 0.95, reward: float = 0.8) -> GrowerOptimizer:
    patch_joblib_load(hit_probability)
    optimizer = GrowerOptimizer(MockBodyAdapter(reward=reward), realism_critic_path="mock.joblib", n_neurons=24, sigma0=0.2)

    def fake_sample_connectome(_sbm, *, seed: int):
        node_types = ("ascending", "dn_sez", "dn_vnc", "interneuron", "rgn", "sensory") * 4
        return {
            "adjacency": [(0, 1, 3), (1, 2, 2), (2, 3, 1)],
            "cell_type_map": {idx: node_type for idx, node_type in enumerate(node_types)},
            "node_types": node_types,
            "n_neurons": len(node_types),
            "seed": seed,
        }

    monkeypatch.setattr(optimizer, "_sample_connectome", fake_sample_connectome)
    monkeypatch.setattr(
        optimizer,
        "_extract_critic_frame",
        lambda connectome, _sbm: pd.DataFrame({"label_hint": list(connectome["node_types"])}),
    )
    return optimizer


def test_genome_to_sbm_round_trip_is_exact(monkeypatch, patch_joblib_load):
    optimizer = _optimizer(monkeypatch, patch_joblib_load)
    genome = np.linspace(-1.5, 1.5, optimizer.genome_dim)
    sbm = optimizer.genome_to_sbm(genome)

    assert sum(sbm["type_counts"].values()) == optimizer.n_neurons
    assert set(sbm["type_counts"].keys()) == set(optimizer.cell_types)
    np.testing.assert_allclose(optimizer.sbm_to_genome(sbm), genome)


def test_evaluate_genome_hard_gates_low_realism(monkeypatch, patch_joblib_load):
    optimizer = _optimizer(monkeypatch, patch_joblib_load, hit_probability=0.5, reward=0.9)
    genome = np.zeros(optimizer.genome_dim, dtype=np.float64)

    score = optimizer.evaluate_genome(genome, n_samples=3)

    assert score == 0.0
    assert optimizer.body_adapter.reward_calls == 0
    assert 0.5 < REALISM_GATE


def test_evaluate_genome_combines_realism_and_reward(monkeypatch, patch_joblib_load):
    optimizer = _optimizer(monkeypatch, patch_joblib_load, hit_probability=0.95, reward=0.8)
    genome = np.zeros(optimizer.genome_dim, dtype=np.float64)

    score = optimizer.evaluate_genome(genome, n_samples=2)

    assert score == pytest.approx(0.95 * 0.8)
    assert optimizer.body_adapter.reward_calls == 1


def test_run_returns_best_genome(monkeypatch, patch_joblib_load):
    optimizer = _optimizer(monkeypatch, patch_joblib_load, hit_probability=0.95, reward=0.8)
    target = np.full((optimizer.genome_dim,), 0.25, dtype=np.float64)
    monkeypatch.setattr(
        optimizer,
        "evaluate_genome",
        lambda genome, n_samples=10: float(1.0 / (1.0 + np.linalg.norm(np.asarray(genome) - target) ** 2)),
    )

    result = optimizer.run(max_iter=2)

    assert result["best_genome"].shape == (optimizer.genome_dim,)
    assert result["best_score"] > 0.0
    assert result["history"]
    np.testing.assert_allclose(optimizer.sbm_to_genome(result["best_sbm"]), result["best_genome"])
