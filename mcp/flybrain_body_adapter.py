from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class StepResult:
    observation: list[float]
    reward: float
    done: bool
    info: dict[str, Any]


class FlyBrainBodyAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def action_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def reset(self, seed: int | None = None) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: int) -> StepResult:
        raise NotImplementedError


class RadioSourceSeekingAdapter(FlyBrainBodyAdapter):
    obs_dim = 10
    action_dim = 4
    max_steps = 100
    arena_size = 10.0
    signal_decay = 2.0
    arrival_radius = 0.1
    max_step_size = 0.25

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._step_count = 0
        self._position = (0.0, 0.0)
        self._source = (0.0, 0.0)
        self._heading = 0.0
        self._speed = 0.5
        self._prev_distance = 0.0

    @property
    def name(self) -> str:
        return "radio_source_seeking"

    def _sample_state(self) -> None:
        self._position = (
            self._rng.uniform(0.5, self.arena_size - 0.5),
            self._rng.uniform(0.5, self.arena_size - 0.5),
        )
        self._source = (
            self._rng.uniform(0.5, self.arena_size - 0.5),
            self._rng.uniform(0.5, self.arena_size - 0.5),
        )
        self._heading = self._rng.uniform(-math.pi, math.pi)
        self._speed = self._rng.uniform(0.25, 0.75)
        self._prev_distance = self._distance_to_source()

    def _distance_to_source(self) -> float:
        return math.dist(self._position, self._source)

    def _signal(self) -> float:
        return math.exp(-self._distance_to_source() / self.signal_decay)

    def _observation(self) -> list[float]:
        dx = (self._source[0] - self._position[0]) / self.arena_size
        dy = (self._source[1] - self._position[1]) / self.arena_size
        return [
            _clamp(self._position[0] / self.arena_size, 0.0, 1.0),
            _clamp(self._position[1] / self.arena_size, 0.0, 1.0),
            math.cos(self._heading),
            math.sin(self._heading),
            _clamp(self._speed, 0.0, 1.0),
            _clamp(self._signal(), 0.0, 1.0),
            _clamp(dx, -1.0, 1.0),
            _clamp(dy, -1.0, 1.0),
            _clamp(dx * math.cos(self._heading) + dy * math.sin(self._heading), -1.0, 1.0),
            _clamp(self._step_count / self.max_steps, 0.0, 1.0),
        ]

    def reset(self, seed: int | None = None) -> list[float]:
        if seed is not None:
            self._rng = random.Random(seed)
        self._step_count = 0
        self._sample_state()
        return self._observation()

    def step(self, action: int) -> StepResult:
        if not 0 <= action < self.action_dim:
            raise ValueError(f"action must be in [0, {self.action_dim}), got {action}")

        if action == 1:
            self._heading += math.pi / 10
        elif action == 2:
            self._heading -= math.pi / 10
        elif action == 3:
            self._speed = _clamp(self._speed * 0.85, 0.0, 1.0)
        else:
            self._speed = _clamp(self._speed + 0.05, 0.0, 1.0)

        step_size = self.max_step_size * self._speed
        self._position = (
            _clamp(self._position[0] + step_size * math.cos(self._heading), 0.0, self.arena_size),
            _clamp(self._position[1] + step_size * math.sin(self._heading), 0.0, self.arena_size),
        )
        self._step_count += 1
        distance = self._distance_to_source()
        progress = _clamp((self._prev_distance - distance) / self.max_step_size, -1.0, 1.0)
        reward = progress - 0.05
        done = distance < self.arrival_radius or self._step_count >= self.max_steps
        if distance < self.arrival_radius:
            reward += 3.0
        self._prev_distance = distance
        return StepResult(
            observation=self._observation(),
            reward=reward,
            done=done,
            info={"distance_to_source": distance, "steps": self._step_count},
        )


class CelegansLocomotionAdapter(FlyBrainBodyAdapter):
    obs_dim = 8
    action_dim = 4
    num_segments = 6
    max_steps = 100
    arena_size = 10.0
    attractant_decay = 2.0
    arrival_radius = 0.1
    max_step_size = 0.25
    _segment_phase = math.pi / 5
    _wave_step = math.pi / 6

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._step_count = 0
        self._position = (0.0, 0.0)
        self._source = (0.0, 0.0)
        self._heading = 0.0
        self._speed = 0.35
        self._wave_phase = 0.0
        self._bend_bias = 0.0
        self._body_angles = [0.0] * self.num_segments
        self._prev_distance = 0.0
        self._gradient_cue = 0.0

    @property
    def name(self) -> str:
        return "celegans_locomotion"

    def _distance_to_source(self) -> float:
        return math.dist(self._position, self._source)

    def _signal_at(self, point: tuple[float, float]) -> float:
        return math.exp(-math.dist(point, self._source) / self.attractant_decay)

    def _gradient_along_heading(self) -> float:
        lookahead = 0.2
        hx = math.cos(self._heading)
        hy = math.sin(self._heading)
        forward = (
            _clamp(self._position[0] + hx * lookahead, 0.0, self.arena_size),
            _clamp(self._position[1] + hy * lookahead, 0.0, self.arena_size),
        )
        backward = (
            _clamp(self._position[0] - hx * lookahead, 0.0, self.arena_size),
            _clamp(self._position[1] - hy * lookahead, 0.0, self.arena_size),
        )
        cue = self._signal_at(forward) - self._signal_at(backward)
        return _clamp(cue, -1.0, 1.0)

    def _update_body_angles(self) -> None:
        amplitude = 0.2 + 0.35 * self._speed + 0.15 * min(1.0, abs(self._bend_bias))
        next_angles: list[float] = []
        for index in range(self.num_segments):
            taper = math.exp(-index / 6)
            angle = amplitude * math.sin(self._wave_phase - index * self._segment_phase)
            angle += self._bend_bias * 0.25 * taper
            next_angles.append(_clamp(angle, -1.0, 1.0))
        self._body_angles = next_angles

    def _observation(self) -> list[float]:
        return [*self._body_angles, _clamp(self._speed, 0.0, 1.0), self._gradient_cue]

    def reset(self, seed: int | None = None) -> list[float]:
        if seed is not None:
            self._rng = random.Random(seed)
        self._step_count = 0
        self._position = (
            self._rng.uniform(1.0, self.arena_size - 1.0),
            self._rng.uniform(1.0, self.arena_size - 1.0),
        )
        while True:
            self._source = (
                self._rng.uniform(1.0, self.arena_size - 1.0),
                self._rng.uniform(1.0, self.arena_size - 1.0),
            )
            if math.dist(self._position, self._source) > 1.0:
                break
        self._heading = self._rng.uniform(-math.pi, math.pi)
        self._speed = self._rng.uniform(0.25, 0.55)
        self._wave_phase = self._rng.uniform(-math.pi, math.pi)
        self._bend_bias = 0.0
        self._update_body_angles()
        self._prev_distance = self._distance_to_source()
        self._gradient_cue = self._gradient_along_heading()
        return self._observation()

    def step(self, action: int) -> StepResult:
        if not 0 <= action < self.action_dim:
            raise ValueError(f"action must be in [0, {self.action_dim}), got {action}")

        previous_gradient_cue = self._gradient_cue

        if action == 0:
            self._bend_bias = _clamp(self._bend_bias + 0.7, -1.0, 1.0)
            self._wave_phase += self._wave_step
        elif action == 1:
            self._bend_bias = _clamp(self._bend_bias - 0.7, -1.0, 1.0)
            self._wave_phase -= self._wave_step
        elif action == 2:
            self._speed = _clamp(self._speed + 0.12, 0.0, 1.0)
            self._wave_phase += self._wave_step / 2
            self._bend_bias *= 0.9
        else:
            self._speed = _clamp(self._speed - 0.12, 0.0, 1.0)
            self._bend_bias *= 0.75

        self._update_body_angles()
        curvature = self._body_angles[0] - self._body_angles[-1]
        self._heading += curvature * 0.3

        propulsion = 0.45 + 0.55 * abs(math.sin(self._wave_phase))
        step_size = self.max_step_size * self._speed * propulsion
        self._position = (
            _clamp(self._position[0] + step_size * math.cos(self._heading), 0.0, self.arena_size),
            _clamp(self._position[1] + step_size * math.sin(self._heading), 0.0, self.arena_size),
        )

        self._step_count += 1
        distance = self._distance_to_source()
        self._gradient_cue = self._gradient_along_heading()
        forward_progress_normalized = _clamp(
            (self._prev_distance - distance) / self.max_step_size,
            -1.0,
            1.0,
        )
        reward = forward_progress_normalized - 0.05
        if self._gradient_cue > previous_gradient_cue:
            reward += 0.2
        done = distance < self.arrival_radius or self._step_count >= self.max_steps
        if distance < self.arrival_radius:
            reward += 3.0
        self._prev_distance = distance
        self._bend_bias *= 0.85

        return StepResult(
            observation=self._observation(),
            reward=reward,
            done=done,
            info={
                "distance_to_source": distance,
                "position": self._position,
                "source_position": self._source,
                "steps": self._step_count,
                "arrived": distance < self.arrival_radius,
            },
        )


__all__ = [
    "CelegansLocomotionAdapter",
    "FlyBrainBodyAdapter",
    "RadioSourceSeekingAdapter",
    "StepResult",
]
