"""Tests for flybrain_realism_critic.RealismCritic."""
import math
import numpy as np
import pandas as pd
import pytest

from flybrain_realism_critic import RealismCritic, REALISM_GATE

CELL_TYPES_6 = ("ascending", "dn_sez", "dn_vnc", "interneuron", "rgn", "sensory")


class MockCritic:
    def __init__(self, hit_probability=0.95):
        self.hit_probability = float(hit_probability)
        self.classes_ = np.asarray(list(CELL_TYPES_6), dtype=object)
        self.feature_names_in_ = np.asarray(["label_hint"], dtype=object)

    def predict_proba(self, frame):
        n = len(frame)
        n_c = len(self.classes_)
        base = (1.0 - self.hit_probability) / max(n_c - 1, 1)
        proba = np.full((n, n_c), base, dtype=np.float64)
        label_index = {label: idx for idx, label in enumerate(self.classes_)}
        labels = frame["label_hint"].tolist() if "label_hint" in frame.columns else ["sensory"] * n
        for row, label in enumerate(labels):
            idx = label_index.get(str(label))
            if idx is not None:
                proba[row, idx] = self.hit_probability
        return proba


def _tiny_connectome(n=12, seed=0):
    rng = np.random.default_rng(seed)
    node_types = tuple(CELL_TYPES_6[i % len(CELL_TYPES_6)] for i in range(n))
    adjacency = []
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < 0.4:
                adjacency.append((i, j, int(rng.integers(1, 4))))
    return {"adjacency": adjacency, "cell_type_map": {idx: t for idx, t in enumerate(node_types)}, "node_types": node_types, "n_neurons": n, "genome": {}, "seed": seed}


@pytest.fixture()
def make_critic(monkeypatch):
    def factory(hit_probability=0.95):
        monkeypatch.setattr("flybrain_realism_critic.joblib.load", lambda _p: MockCritic(hit_probability))
        rc = RealismCritic("mock.joblib")
        monkeypatch.setattr(rc, "_extract_critic_frame", lambda connectome, sbm: pd.DataFrame({"label_hint": list(connectome["node_types"])}))
        return rc
    return factory


def test_critic_loads_and_exposes_cell_types(make_critic):
    rc = make_critic()
    assert len(rc.cell_types) == 6
    assert "sensory" in rc.cell_types

def test_score_returns_float_in_unit_interval(make_critic):
    rc = make_critic(hit_probability=0.8)
    conn = _tiny_connectome(n=12, seed=42)
    s = rc.score(conn, {})
    assert isinstance(s, float) and 0.0 <= s <= 1.0

def test_score_high_probability_is_close_to_hit_prob(make_critic):
    rc = make_critic(hit_probability=0.95)
    s = rc.score(_tiny_connectome(n=12, seed=7), {})
    assert math.isclose(s, 0.95, abs_tol=1e-6)

def test_gate_passes_above_threshold(make_critic):
    rc = make_critic(hit_probability=0.95)
    assert rc.gate(_tiny_connectome(n=12, seed=3), {}, threshold=REALISM_GATE)

def test_gate_fails_below_threshold(make_critic):
    rc = make_critic(hit_probability=0.5)
    assert not rc.gate(_tiny_connectome(n=12, seed=4), {}, threshold=REALISM_GATE)

def test_score_deterministic(make_critic):
    rc = make_critic()
    conn = _tiny_connectome(n=12, seed=99)
    assert math.isclose(rc.score(conn, {}), rc.score(conn, {}))

def test_score_unknown_type_is_zero(make_critic):
    rc = make_critic()
    conn = dict(_tiny_connectome(n=6, seed=5))
    conn["node_types"] = tuple(["unknown_type"] * 6)
    s = rc.score(conn, {})
    assert math.isclose(s, 0.0)
