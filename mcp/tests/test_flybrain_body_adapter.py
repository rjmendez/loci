from __future__ import annotations

import pytest

from flybrain_body_adapter import CelegansLocomotionAdapter, StepResult


def test_celegans_dims() -> None:
    adapter = CelegansLocomotionAdapter()

    assert adapter.name == "celegans_locomotion"
    assert adapter.obs_dim == 8
    assert adapter.action_dim == 4


def test_reset_returns_correct_shape() -> None:
    adapter = CelegansLocomotionAdapter()

    observation = adapter.reset(seed=7)

    assert len(observation) == 8
    assert all(-1.0 <= angle <= 1.0 for angle in observation[:6])
    assert 0.0 <= observation[6] <= 1.0
    assert -1.0 <= observation[7] <= 1.0


def test_step_returns_valid_step_result() -> None:
    adapter = CelegansLocomotionAdapter()
    adapter.reset(seed=11)

    result = adapter.step(2)

    assert isinstance(result, StepResult)
    assert len(result.observation) == 8
    assert isinstance(result.reward, float)
    assert isinstance(result.done, bool)
    assert isinstance(result.info, dict)
    assert result.info["steps"] == 1
    assert "distance_to_source" in result.info


def test_dorsal_and_ventral_actions_produce_different_body_angles() -> None:
    dorsal_adapter = CelegansLocomotionAdapter()
    ventral_adapter = CelegansLocomotionAdapter()
    dorsal_adapter.reset(seed=23)
    ventral_adapter.reset(seed=23)

    dorsal = dorsal_adapter.step(0).observation[:6]
    ventral = ventral_adapter.step(1).observation[:6]

    assert dorsal != ventral
    assert dorsal[0] > ventral[0]


def test_seed_determinism() -> None:
    first = CelegansLocomotionAdapter()
    second = CelegansLocomotionAdapter()

    first_obs = first.reset(seed=101)
    second_obs = second.reset(seed=101)
    assert first_obs == pytest.approx(second_obs)

    for action in [0, 2, 1, 3, 2]:
        first_result = first.step(action)
        second_result = second.step(action)
        assert first_result.observation == pytest.approx(second_result.observation)
        assert first_result.reward == pytest.approx(second_result.reward)
        assert first_result.done is second_result.done
        assert first_result.info == second_result.info
