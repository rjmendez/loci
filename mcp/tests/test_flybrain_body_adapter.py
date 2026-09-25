import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flybrain_body_adapter import LociMockBodyAdapter, RadioSourceSeekingAdapter, StepResult  # noqa: E402


@pytest.mark.parametrize(
    "adapter_cls,expected_obs_dim,expected_action_dim",
    [
        (LociMockBodyAdapter, 32, 4),
        (RadioSourceSeekingAdapter, 10, 4),
    ],
)
def test_obs_action_dims_match_reset_and_step(adapter_cls, expected_obs_dim, expected_action_dim):
    adapter = adapter_cls()
    assert adapter.obs_dim == expected_obs_dim
    assert adapter.action_dim == expected_action_dim
    obs = adapter.reset(seed=7)
    assert obs.shape == (expected_obs_dim,)
    result = adapter.step(np.zeros(expected_action_dim, dtype=np.float32))
    assert isinstance(result, StepResult)
    assert result.obs.shape == (expected_obs_dim,)


def test_loci_mock_reset_step_contract_and_reward_bounds():
    adapter = LociMockBodyAdapter()
    obs = adapter.reset(seed=11)
    assert obs.dtype == np.float32
    result = adapter.step(np.array([10.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert isinstance(result.reward, float)
    assert result.reward in (0.0, 1.0)
    assert isinstance(result.done, bool)
    assert result.info["body"] == adapter.name()
    assert result.info["expected_route"] in adapter.ROUTES
    assert result.info["selected_route"] in adapter.ROUTES
    assert 0.0 <= result.info["accuracy_so_far"] <= 1.0


def test_radio_reset_step_contract_and_reward_bounds():
    adapter = RadioSourceSeekingAdapter()
    obs = adapter.reset(seed=13)
    assert obs.shape == (10,)
    assert np.all(obs[:8] >= 0.0)
    result = adapter.step(np.array([0.0, 0.0, 8.0, -8.0], dtype=np.float32))
    assert isinstance(result.reward, float)
    assert result.reward <= 0.0
    assert result.reward >= -100.0
    assert isinstance(result.done, bool)
    assert result.info["body"] == adapter.name()
    assert 0.0 <= result.info["speed"] <= adapter.max_speed


def test_loci_mock_dataset_is_balanced():
    adapter = LociMockBodyAdapter()
    assert len(adapter.ROUTING_EXAMPLES) == 100
    counts = {route: 0 for route in adapter.ROUTES}
    for example in adapter.ROUTING_EXAMPLES:
        counts[example.route] += 1
    assert counts == {route: 25 for route in adapter.ROUTES}


@pytest.mark.parametrize(
    "adapter_cls,action_sequence",
    [
        (LociMockBodyAdapter, [np.array([0.0, 1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0, 0.0])]),
        (RadioSourceSeekingAdapter, [np.array([0.0, 0.0, 0.0, 0.0]), np.array([4.0, 1.0, 0.5, -1.0])]),
    ],
)
def test_seed_determinism(adapter_cls, action_sequence):
    first = adapter_cls()
    second = adapter_cls()
    first_obs = first.reset(seed=123)
    second_obs = second.reset(seed=123)
    np.testing.assert_allclose(first_obs, second_obs)

    for action in action_sequence:
        left = first.step(action)
        right = second.step(action)
        np.testing.assert_allclose(left.obs, right.obs)
        assert left.reward == pytest.approx(right.reward)
        assert left.done is right.done
        assert left.info == right.info
