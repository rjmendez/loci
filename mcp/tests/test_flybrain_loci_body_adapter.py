"""Tests for flybrain_loci_body_adapter.LociCodeGraphAdapter."""
import numpy as np
import pytest
from flybrain_loci_body_adapter import LociCodeGraphAdapter, OBS_DIM, ACTION_DIM


@pytest.fixture()
def adapter():
    return LociCodeGraphAdapter(mock=True, seed=42)


def test_obs_action_dims(adapter):
    assert adapter.obs_dim == OBS_DIM
    assert adapter.action_dim == ACTION_DIM


def test_reset_returns_correct_shape(adapter):
    obs = adapter.reset(seed=0)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32


def test_obs_values_in_unit_interval(adapter):
    obs = adapter.reset(seed=1)
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0)


def test_step_returns_valid_step_result(adapter):
    adapter.reset(seed=2)
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[0] = 1.0  # code_graph_query
    result = adapter.step(action)
    assert result.obs.shape == (OBS_DIM,)
    assert isinstance(result.reward, float)
    assert isinstance(result.done, bool)


def test_step_wrong_action_shape_raises(adapter):
    adapter.reset()
    with pytest.raises(ValueError, match="expected action shape"):
        adapter.step(np.zeros(2, dtype=np.float32))


def test_done_after_max_steps(adapter):
    adapter._max_steps = 3
    adapter.reset(seed=3)
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[0] = 1.0
    for _ in range(2):
        result = adapter.step(action)
        assert not result.done
    result = adapter.step(action)
    assert result.done


def test_no_op_penalizes_open_investigation(adapter):
    adapter.reset(seed=4)
    adapter._current_node = {
        "node_kind": "symbol", "symbol_kind": "function",
        "lang": "python", "in_investigation": True,
        "caller_count": 0, "callee_count": 0,
        "import_count": 0, "reference_count": 0,
        "finding_confidence": 0.0, "id": "test:1", "name": "fn"
    }
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    action[3] = 1.0  # no_op
    result = adapter.step(action)
    assert result.reward < 0.0


def test_reward_hook_returns_mean_realism(adapter):
    r = adapter.reward(samples=[], realism_scores=[0.8, 0.9, 0.95])
    assert abs(r - 0.8833) < 0.01


def test_reward_hook_empty_returns_zero(adapter):
    assert adapter.reward(samples=[], realism_scores=[]) == 0.0


def test_encode_node_python_flag():
    a = LociCodeGraphAdapter(mock=True)
    node = {"node_kind": "symbol", "symbol_kind": "function", "lang": "python",
            "in_investigation": False, "caller_count": 4, "callee_count": 2,
            "import_count": 1, "reference_count": 0, "finding_confidence": 0.7}
    obs = a._encode_node(node)
    assert obs[12] == 1.0  # is_python
    assert obs[13] == 0.0  # not typescript
    assert obs[11] == pytest.approx(0.7)  # finding_confidence
