from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import math
from typing import Any, ClassVar

import numpy as np


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]


class FlyBrainBodyAdapter(ABC):
    """Single interface for all T2 bodies. Grown brains interact only through this."""

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Dimensionality of the observation vector."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimensionality of the action vector."""

    @abstractmethod
    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset episode, return initial observation."""

    @abstractmethod
    def step(self, action: np.ndarray) -> StepResult:
        """Apply action, return StepResult."""

    @abstractmethod
    def name(self) -> str:
        """Unique body identifier for provenance."""

    def _coerce_action(self, action: np.ndarray) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if arr.shape != (self.action_dim,):
            raise ValueError(f"expected action shape {(self.action_dim,)}, got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("action must contain only finite values")
        return arr

    def _terminal_obs(self) -> np.ndarray:
        return np.zeros(self.obs_dim, dtype=np.float32)


@dataclass(frozen=True)
class RoutingExample:
    query: str
    route: str


def _stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def _random_embedding(text: str, dim: int) -> np.ndarray:
    rng = np.random.RandomState(_stable_seed(text))
    vec = rng.normal(loc=0.0, scale=1.0, size=dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


def _build_loci_routing_examples() -> tuple[RoutingExample, ...]:
    templates = {
        "memory": (
            "Recall prior incident notes for auth token regression",
            "Find the last decision about schema cache invalidation",
            "Retrieve previous answer about routing policy drift",
            "Surface remembered context for hermes grounding thresholds",
            "Load earlier benchmark notes for graph embedding latency",
        ),
        "code_graph": (
            "Find callers of compute_route_score in the planner",
            "Trace the symbol impact of AdapterFactory across the code graph",
            "Show inbound references to flybrain_model_eval.run_evaluation",
            "Locate the class hierarchy rooted at BodyAdapterBase",
            "Map dependencies for the ingest queue serializer",
        ),
        "investigation": (
            "Investigate why benchmark latency spiked after the last merge",
            "Open a provenance investigation for conflicting routing evidence",
            "Analyze a suspected regression in grouped split reproducibility",
            "Examine why the verifier gate failed on the latest holdout",
            "Triage an unresolved finding about stale manifest promotion",
        ),
        "external": (
            "Look up the MAVLink definition for GLOBAL_POSITION_INT",
            "Check the FlyWire paper for optic lobe release terminology",
            "Search external docs for CMA-ES restart heuristics",
            "Verify the neuPrint column vocabulary from published docs",
            "Fetch reference material for radio source-seeking baselines",
        ),
    }
    examples: list[RoutingExample] = []
    for route, stems in templates.items():
        for idx in range(25):
            stem = stems[idx % len(stems)]
            examples.append(RoutingExample(query=f"{stem} [{idx:02d}]", route=route))
    return tuple(examples)


def _softmax(action: np.ndarray) -> np.ndarray:
    shifted = action - np.max(action)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _wrap_angle(angle: float) -> float:
    return ((angle + math.pi) % (2.0 * math.pi)) - math.pi


class LociMockBodyAdapter(FlyBrainBodyAdapter):
    """Synthetic held-out routing body for early grow-to-spec loop tests."""

    ROUTES: ClassVar[tuple[str, ...]] = ("memory", "code_graph", "investigation", "external")
    ROUTING_EXAMPLES: ClassVar[tuple[RoutingExample, ...]] = _build_loci_routing_examples()
    ROUTING_OBSERVATIONS: ClassVar[tuple[np.ndarray, ...]] = tuple(
        _random_embedding(example.query, 32) for example in ROUTING_EXAMPLES
    )
    MAX_STEPS: ClassVar[int] = len(ROUTING_EXAMPLES)

    def __init__(self) -> None:
        self._rng = np.random.RandomState()
        self._order = np.arange(len(self.ROUTING_EXAMPLES), dtype=np.int64)
        self._cursor = 0
        self._done = False
        self._reward_total = 0.0
        self._seed: int | None = None

    @property
    def obs_dim(self) -> int:
        return 32

    @property
    def action_dim(self) -> int:
        return 4

    def name(self) -> str:
        return "loci_mock_code_graph_v1"

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._seed = seed
        self._rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self._order = self._rng.permutation(len(self.ROUTING_EXAMPLES))
        self._cursor = 0
        self._done = False
        self._reward_total = 0.0
        return self._current_obs()

    def _current_obs(self) -> np.ndarray:
        index = int(self._order[self._cursor])
        return self.ROUTING_OBSERVATIONS[index].copy()

    def step(self, action: np.ndarray) -> StepResult:
        if self._done:
            raise RuntimeError("episode is done; call reset() before step()")
        action_vec = self._coerce_action(action)
        index = int(self._order[self._cursor])
        example = self.ROUTING_EXAMPLES[index]
        selected_index = int(np.argmax(action_vec))
        selected_route = self.ROUTES[selected_index]
        reward = 1.0 if selected_route == example.route else 0.0
        self._reward_total += reward
        self._cursor += 1
        done = self._cursor >= self.MAX_STEPS
        self._done = done
        next_obs = self._terminal_obs() if done else self._current_obs()
        info = {
            "body": self.name(),
            "seed": self._seed,
            "dataset_size": self.MAX_STEPS,
            "step_index": self._cursor - 1,
            "query": example.query,
            "expected_route": example.route,
            "selected_route": selected_route,
            "selected_route_index": selected_index,
            "accuracy_so_far": self._reward_total / self._cursor,
        }
        return StepResult(obs=next_obs, reward=float(reward), done=done, info=info)


class RadioSourceSeekingAdapter(FlyBrainBodyAdapter):
    """Simple 2-D radio source-seeking simulator for free-tier closed-loop experiments."""

    SENSOR_ANGLES: ClassVar[np.ndarray] = np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)
    MAX_STEPS: ClassVar[int] = 200

    def __init__(
        self,
        *,
        sensor_noise_std: float = 0.02,
        turn_rate: float = math.pi / 6.0,
        speed_delta: float = 0.5,
        max_speed: float = 3.0,
        world_limit: float = 50.0,
        source_radius: float = 20.0,
        min_source_radius: float = 5.0,
    ) -> None:
        self.sensor_noise_std = float(sensor_noise_std)
        self.turn_rate = float(turn_rate)
        self.speed_delta = float(speed_delta)
        self.max_speed = float(max_speed)
        self.world_limit = float(world_limit)
        self.source_radius = float(source_radius)
        self.min_source_radius = float(min_source_radius)
        self._rng = np.random.RandomState()
        self._seed: int | None = None
        self._position = np.zeros(2, dtype=np.float64)
        self._source = np.zeros(2, dtype=np.float64)
        self._heading = 0.0
        self._speed = 0.0
        self._prev_heading_delta = 0.0
        self._prev_speed = 0.0
        self._steps = 0
        self._done = False

    @property
    def obs_dim(self) -> int:
        return 10

    @property
    def action_dim(self) -> int:
        return 4

    def name(self) -> str:
        return "radio_source_seeking_v1"

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._seed = seed
        self._rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self._position = np.zeros(2, dtype=np.float64)
        radius = math.sqrt(self._rng.uniform(self.min_source_radius ** 2, self.source_radius ** 2))
        angle = float(self._rng.uniform(-math.pi, math.pi))
        self._source = np.array([radius * math.cos(angle), radius * math.sin(angle)], dtype=np.float64)
        self._heading = float(self._rng.uniform(-math.pi, math.pi))
        self._speed = 0.0
        self._prev_heading_delta = 0.0
        self._prev_speed = 0.0
        self._steps = 0
        self._done = False
        return self._observe()

    def _distance_to_source(self) -> float:
        return float(np.linalg.norm(self._source - self._position))

    def _observe(self) -> np.ndarray:
        delta = self._source - self._position
        distance = max(float(np.linalg.norm(delta)), 1e-6)
        source_bearing = math.atan2(delta[1], delta[0])
        base_strength = 1.0 / (1.0 + distance)
        readings = []
        for sensor_angle in self.SENSOR_ANGLES:
            absolute_sensor_heading = self._heading + float(sensor_angle)
            relative = _wrap_angle(source_bearing - absolute_sensor_heading)
            directional_gain = 0.25 + 0.75 * max(math.cos(relative), 0.0)
            noisy = base_strength * directional_gain + self._rng.normal(0.0, self.sensor_noise_std)
            readings.append(float(np.clip(noisy, 0.0, 1.0)))
        obs = np.asarray(readings + [self._prev_heading_delta, self._prev_speed], dtype=np.float32)
        return obs

    def step(self, action: np.ndarray) -> StepResult:
        if self._done:
            raise RuntimeError("episode is done; call reset() before step()")
        action_vec = self._coerce_action(action)
        probs = _softmax(action_vec)
        heading_delta = float((probs[0] - probs[1]) * self.turn_rate)
        speed_delta = float((probs[2] - probs[3]) * self.speed_delta)
        self._heading = _wrap_angle(self._heading + heading_delta)
        self._speed = float(np.clip(self._speed + speed_delta, 0.0, self.max_speed))
        direction = np.array([math.cos(self._heading), math.sin(self._heading)], dtype=np.float64)
        self._position = np.clip(self._position + direction * self._speed, -self.world_limit, self.world_limit)
        self._prev_heading_delta = heading_delta
        self._prev_speed = self._speed
        self._steps += 1
        distance = self._distance_to_source()
        reward = -distance
        done = distance <= 1.0 or self._steps >= self.MAX_STEPS
        self._done = done
        next_obs = self._terminal_obs() if done else self._observe()
        info = {
            "body": self.name(),
            "seed": self._seed,
            "step_index": self._steps,
            "position": tuple(float(v) for v in self._position),
            "source_position": tuple(float(v) for v in self._source),
            "distance_to_source": distance,
            "heading": self._heading,
            "speed": self._speed,
            "action_probabilities": tuple(float(v) for v in probs),
            "terminated_by": "source_reached" if distance <= 1.0 else ("max_steps" if done else None),
        }
        return StepResult(obs=next_obs, reward=float(reward), done=done, info=info)


__all__ = [
    "FlyBrainBodyAdapter",
    "LociMockBodyAdapter",
    "RadioSourceSeekingAdapter",
    "RoutingExample",
    "StepResult",
]
