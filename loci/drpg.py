"""
DRPG (Dynamic Retrieval-based Policy Generation) evolution loop.

Reads step-level reward trajectories persisted by eval_hook_stub,
scores them, and promotes high-quality policies back to Loci memory.

Architecture:
- TrajectoryStore: reads/writes JSONL trajectory files per policy_id
- TrajectoryScorer: aggregates 7D reward vectors into a scalar quality score
- PolicyPromoter: surfaces promote/demote/keep decisions given trajectory score
- DrpgSuggestionGenerator: generates next-step discovery/refinement suggestions
"""

import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from loci.jsonl_bounds import MAX_FILE_ROWS, append_bounded, tail_lines
from loci.policy import Policy, PolicyBackend, loci_home, validate_policy_id
from loci.mtime_cache import MtimeCache

logger = logging.getLogger(__name__)

# evolve_all_policies is serial and bounded: one process, one thread, at most this many
# policies per call.  A bigger batch is refused, not truncated.
MAX_POLICIES_PER_CALL = 64


class PolicyLimitError(ValueError):
    """evolve_all_policies was given more than MAX_POLICIES_PER_CALL policies."""

# ---------------------------------------------------------------------------
# Module-level mtime-keyed trajectory cache
# ---------------------------------------------------------------------------

_TRAJ_CACHE: MtimeCache[List[Dict[str, Any]]] = MtimeCache(maxsize=64)


class TrajectoryStore:
    """Persists and retrieves step-level reward trajectories per policy_id."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, policy_id: str) -> Path:
        validate_policy_id(policy_id)
        return self.base_dir / f"{policy_id}.jsonl"

    def push(self, policy_id: str, step_feedback_dict: Dict[str, Any], *, strict: bool = False) -> None:
        """Append one step's feedback dict to the trajectory file.

        Reward integrity: when the step carries a ``reward`` vector, every value present must be a
        finite number in [-1, 1] and a non-object reward is refused; with ``strict=True`` all seven
        reward keys must also be present.  Violations raise ValueError('invalid reward: ...') and
        nothing is written.  The file is bounded: past 5000 rows or 1 MiB it is rotated to
        ``<policy_id>.jsonl.<k>`` and a fresh file is started.
        """
        entry = dict(step_feedback_dict)
        _check_reward(entry.get("reward"), strict=strict, present="reward" in entry)
        entry.setdefault("_ts", datetime.utcnow().isoformat() + "Z")
        append_bounded(self._path(policy_id), json.dumps(entry))
        # Invalidate cache so next read picks up the new step
        _TRAJ_CACHE.invalidate(policy_id)

    def read(self, policy_id: str, limit: int = MAX_FILE_ROWS) -> List[Dict[str, Any]]:
        """Return the most recent ``limit`` persisted steps (default: a whole file, 5000) for a policy_id.

        Only the tail of the file is read, so the cost is bounded by ``limit`` and not by file size.
        The default-limit read uses the mtime-keyed cache.
        """
        path = self._path(policy_id)
        if not path.exists():
            return []
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return []
        cacheable = limit == MAX_FILE_ROWS
        if cacheable:
            cached = _TRAJ_CACHE.get(policy_id, mtime)
            if cached is not None:
                return list(cached)
        steps = []
        for raw in tail_lines(path, limit):
            try:
                steps.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        if cacheable:
            _TRAJ_CACHE.set(policy_id, steps, mtime)
        return list(steps)

    def list_policies(self) -> List[str]:
        """Return policy IDs that have trajectory files."""
        ids = []
        for path in self.base_dir.glob("*.jsonl"):
            try:
                ids.append(validate_policy_id(path.stem))
            except ValueError:
                continue  # not a policy id: never hand a stray file name to the store
        return ids


_REWARD_DIMS = ("r_ret", "r_mem", "r_ovp", "r_bkp", "r_ref", "r_step", "r_cor")
# Weights applied AFTER abs() on penalty dims (r_ovp, r_bkp, r_step at indices 2,3,5).
# Formula: r_ret + 0.5*r_mem - abs(r_ovp) - abs(r_bkp) + r_ref - abs(r_step) + r_cor
_REWARD_WEIGHTS_DEFAULT = (1.0, 0.5, -1.0, -1.0, 1.0, -1.0, 1.0)
_ABS_DIM_INDICES = (2, 3, 5)  # r_ovp, r_bkp, r_step require abs before weighting


_REWARD_MIN, _REWARD_MAX = -1.0, 1.0
_LEGACY_WEIGHT_KEYS = {"r_rem": "r_mem"}  # Policy.get_reward_vector_weights() still says r_rem for the memory term


def _check_reward(reward: Any, *, strict: bool, present: bool) -> None:
    """ValueError('invalid reward: ...') unless *reward* is an in-range reward vector (see TrajectoryStore.push)."""
    if not present:
        if strict:
            raise ValueError("invalid reward: step has no reward vector")
        return
    if not isinstance(reward, dict):
        raise ValueError(f"invalid reward: expected an object of reward values, got {type(reward).__name__}")
    for key in _REWARD_DIMS:
        if key not in reward:
            continue
        value = reward[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                or not (_REWARD_MIN <= value <= _REWARD_MAX):
            raise ValueError(f"invalid reward: {key}={value!r} is not a finite number in [-1, 1]")
    if strict:
        missing = [key for key in _REWARD_DIMS if key not in reward]
        if missing:
            raise ValueError(f"invalid reward: missing required key(s) {', '.join(missing)}")


def weights_from_mapping(weights: Mapping[str, float]) -> tuple:
    """A weight tuple in _REWARD_DIMS order from a {reward key: weight} mapping.

    Dimensions the mapping omits weigh 0.0.  The legacy key ``r_rem`` is the memory term
    ``r_mem``: it is mapped with one logged warning (an explicit ``r_mem`` wins over it), never
    dropped silently.  Any other key is a ValueError.
    """
    resolved: Dict[str, float] = {}
    legacy_seen = []
    for key, weight in weights.items():
        canonical = _LEGACY_WEIGHT_KEYS.get(key, key)
        if canonical not in _REWARD_DIMS:
            raise ValueError(f"unknown reward weight key {key!r}")
        if key != canonical:
            legacy_seen.append((key, canonical))
        if key == canonical or canonical not in weights:
            resolved[canonical] = float(weight)
    for legacy, canonical in legacy_seen:
        logger.warning("reward weight key %r is the legacy name of %r; mapped (use %r)", legacy, canonical, canonical)
    return tuple(resolved.get(dim, 0.0) for dim in _REWARD_DIMS)


class TrajectoryScorer:
    """Aggregates a list of step dicts into a scalar quality score."""

    def __init__(self, weights: Union[Sequence[float], Mapping[str, float]] = _REWARD_WEIGHTS_DEFAULT):
        if isinstance(weights, Mapping):
            weights = weights_from_mapping(weights)
        self.weights = tuple(weights)

    @classmethod
    def from_policy(cls, policy: Policy) -> "TrajectoryScorer":
        """A scorer weighted by ``policy.get_reward_vector_weights()`` (its tier and phase)."""
        return cls(policy.get_reward_vector_weights())

    def score(self, steps: List[Dict[str, Any]]) -> float:
        """
        Weighted mean over all steps of:
            r_ret + 0.5*r_mem - abs(r_ovp) - abs(r_bkp) + r_ref - abs(r_step) + r_cor

        Dispatches to _score_scalar for <10 steps, _score_vectorized otherwise.
        Returns 0.0 for empty input.
        """
        if not steps:
            return 0.0
        if len(steps) < 10:
            return self._score_scalar(steps)
        return self._score_vectorized(steps)

    def _score_scalar(self, steps: List[Dict[str, Any]]) -> float:
        """Pure Python weighted sum for small step counts."""
        w = self.weights
        total = 0.0
        for s in steps:
            rv = s.get("reward", s)
            r_ret  = float(rv.get("r_ret",  0.0))
            r_mem  = float(rv.get("r_mem",  0.0))
            r_ovp  = float(rv.get("r_ovp",  0.0))
            r_bkp  = float(rv.get("r_bkp",  0.0))
            r_ref  = float(rv.get("r_ref",  0.0))
            r_step = float(rv.get("r_step", 0.0))
            r_cor  = float(rv.get("r_cor",  0.0))
            total += (
                w[0] * r_ret
                + w[1] * r_mem
                + w[2] * abs(r_ovp)
                + w[3] * abs(r_bkp)
                + w[4] * r_ref
                + w[5] * abs(r_step)
                + w[6] * r_cor
            )
        return total / len(steps)

    def _score_vectorized(self, steps: List[Dict[str, Any]]) -> float:
        """NumPy/CuPy vectorized weighted mean for larger batches.

        Applies abs() to penalty dimensions (r_ovp, r_bkp, r_step) before
        weighting, matching the _score_scalar formula exactly.
        """
        from loci.accelerator import get_array_module
        xp = get_array_module()
        dims = _REWARD_DIMS
        R = xp.array(
            [[float(s.get("reward", s).get(k, 0.0)) for k in dims] for s in steps],
            dtype=xp.float32,
        )
        for i in _ABS_DIM_INDICES:
            R[:, i] = xp.abs(R[:, i])
        w = xp.array(self.weights, dtype=xp.float32)
        return float(xp.mean(R @ w))


class PolicyPromoter:
    """Decides promote / demote / keep given a trajectory quality score."""

    PROMOTE_THRESHOLD = 0.7
    DEMOTE_THRESHOLD = 0.2

    def decide(self, score: float, policy: Policy) -> Dict[str, Any]:
        """
        Returns {"action": "promote"|"demote"|"keep", "score": float, "reason": str}.
        Thresholds: score > 0.7 → promote, score < 0.2 → demote, else keep.
        """
        if score > self.PROMOTE_THRESHOLD:
            return {
                "action": "promote",
                "score": score,
                "reason": f"score {score:.4f} exceeds promote threshold {self.PROMOTE_THRESHOLD}",
            }
        if score < self.DEMOTE_THRESHOLD:
            return {
                "action": "demote",
                "score": score,
                "reason": f"score {score:.4f} below demote threshold {self.DEMOTE_THRESHOLD}",
            }
        return {
            "action": "keep",
            "score": score,
            "reason": f"score {score:.4f} within acceptable range [{self.DEMOTE_THRESHOLD}, {self.PROMOTE_THRESHOLD}]",
        }


class DrpgSuggestionGenerator:
    """Inspects step sequences for patterns and returns actionable suggestions."""

    def generate(self, steps: List[Dict[str, Any]], policy: Policy) -> List[Dict[str, Any]]:
        """
        Returns a list of {"suggestion": str, "confidence": float, "affects": str} dicts.

        Patterns detected:
        - Many BACKTRACK actions → suggest fewer hops
        - High average r_ovp magnitude → suggest dedup filter
        - Many REFUSE actions → suggest moving to refinement phase
        """
        if not steps:
            return []

        suggestions: List[Dict[str, Any]] = []
        total = len(steps)

        backtrack_count = sum(
            1 for s in steps if str(s.get("action", "")).upper() == "BACKTRACK"
        )
        refuse_count = sum(
            1 for s in steps if str(s.get("action", "")).upper() == "REFUSE"
        )
        avg_ovp = (
            sum(abs(float(s.get("reward", {}).get("r_ovp", 0.0))) for s in steps) / total
        )

        if backtrack_count / total > 0.3:
            suggestions.append({
                "suggestion": "Reduce max_retrieval_hops: high backtrack rate indicates over-exploration",
                "confidence": min(0.95, 0.5 + backtrack_count / total),
                "affects": "max_retrieval_hops",
            })

        if avg_ovp > 0.15:
            suggestions.append({
                "suggestion": "Enable semantic dedup filter: high subquery overlap detected",
                "confidence": min(0.9, avg_ovp * 4),
                "affects": "dedup_filter",
            })

        if refuse_count / total > 0.4:
            suggestions.append({
                "suggestion": "Transition to refinement phase: high refusal rate signals discovery saturation",
                "confidence": min(0.9, 0.5 + refuse_count / total),
                "affects": "discovery_mode",
            })

        return suggestions


def evolve_policy(
    policy_id: str,
    backend: PolicyBackend,
    trajectory_store: TrajectoryStore,
) -> Dict[str, Any]:
    """
    Full DRPG pipeline:
      1. Read trajectory from store
      2. Score via TrajectoryScorer, weighted by the policy's own reward weights
         (Policy.get_reward_vector_weights: its tier and discovery/refinement phase)
      3. Decide promote/demote/keep via PolicyPromoter
      4. If promote: update policy metadata and re-store
      5. Return summary dict

    Returns:
        {
            "policy_id": str,
            "action": "promote"|"demote"|"keep",
            "score": float,
            "suggestions": [...],
            "steps_analyzed": int,
        }
    """
    steps = trajectory_store.read(policy_id)

    policy: Optional[Policy] = backend.load(policy_id)
    if policy is None:
        return {
            "policy_id": policy_id,
            "action": "error",
            "score": TrajectoryScorer().score(steps),
            "suggestions": [],
            "steps_analyzed": len(steps),
            "error": "policy_not_found",
        }

    # The policy's own tier/phase weights (Policy.get_reward_vector_weights), not a fixed tuple.
    score = TrajectoryScorer.from_policy(policy).score(steps)

    promoter = PolicyPromoter()
    decision = promoter.decide(score, policy)
    action = decision["action"]

    if action == "promote":
        policy.metadata.source = "drpg-evolved"
        policy.metadata.version += 1
        policy.metadata.updated_at = datetime.utcnow().isoformat() + "Z"
        backend.store(policy)

    suggester = DrpgSuggestionGenerator()
    suggestions = suggester.generate(steps, policy)

    return {
        "policy_id": policy_id,
        "action": action,
        "score": score,
        "suggestions": suggestions,
        "steps_analyzed": len(steps),
    }


def _default_trajectory_store() -> TrajectoryStore:
    traj_dir = os.environ.get("LOCI_TRAJECTORY_DIR") or str(loci_home() / "trajectories")
    return TrajectoryStore(Path(traj_dir))


def _evolve_worker(args):
    """Evolve one policy from a (policy_id, policy_dir, trajectory_dir) tuple; errors become an 'error' result."""
    policy_id, policy_dir_str, traj_dir_str = args
    try:
        backend = PolicyBackend(Path(policy_dir_str))
        store = TrajectoryStore(Path(traj_dir_str))
        return evolve_policy(policy_id, backend, store)
    except Exception as e:
        return {
            "policy_id": policy_id,
            "action": "error",
            "error": str(e),
            "score": 0.0,
            "steps_analyzed": 0,
        }


async def evolve_all_policies(
    policy_ids: List[str],
    policy_dir: Path,
    trajectory_dir: Path,
    max_workers: int = None,
) -> List[Dict[str, Any]]:
    """Evolve the given policies one after another, in sorted policy_id order, in this process.

    Serial by design: no process pool, no threads.  Duplicate ids are evolved once.  More than
    MAX_POLICIES_PER_CALL distinct ids raises PolicyLimitError before anything is evolved.
    ``max_workers`` is accepted for backward compatibility and ignored.
    """
    del max_workers  # compatibility only; evolution is serial
    ids = sorted(set(policy_ids or []))
    if len(ids) > MAX_POLICIES_PER_CALL:
        raise PolicyLimitError(
            f"{len(ids)} policies requested; at most {MAX_POLICIES_PER_CALL} per call"
        )
    return [_evolve_worker((pid, str(policy_dir), str(trajectory_dir))) for pid in ids]
