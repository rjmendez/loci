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

import hashlib
import json
import logging
import math
import os
import random
import shutil
import tempfile
import time
import tomllib
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from loci.degeneracy import DEGENERATE, HINGE_REWARD, MIN_ARM_N, Verdict
from loci.jsonl_bounds import MAX_FILE_ROWS, append_bounded, tail_lines
from loci.policy import (
    GENE_STEP_MAX, GENES, LEARNED_GENES_FLAG, LEARNED_GENES_KEY, Policy, PolicyBackend, _q, apply_genes,
    check_genome_step, loci_home, validate_policy_id,
)
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


# ===========================================================================
# Bounded evolver (Tripwire-Governor S18)
# ===========================================================================
#
# Slow path, off the call path, never wired or scheduled by this module.  The learner may tighten at most
# MAX_GENES gene scales (loci.policy.GENES) by at most 10% per patch, at most one patch per cell per 24 h,
# keeps the previous value plus an evidence hash for a one-command revert, and can write only inside its
# own sandbox (LearnWriter).  Every write is temp + rename.  Dry-run is the caller's default; this module
# never decides to apply.

LEARN_QUOTA_BYTES = 256 * 1024 * 1024    # learn dir quota (design: 256 MiB on C:)
MIN_FREE_BYTES = 2 * 1024 ** 3           # any learner write aborts below 2 GB free
PATCH_COOLDOWN_S = 24 * 3600             # at most one patch per cell per 24 h
MIN_OUTCOMES = MIN_ARM_N                 # 30 labelled outcomes
MIN_DAYS = 3                             # over at least 3 distinct (UTC) days
WINDOW_GAP_S = 3600                      # two windows: different session, at least 1 h apart
ES_POP = 8
ES_SIGMA = 0.2
TRAIN_GAIN_MIN = Fraction(2, 100)
HELDOUT_GAIN_MIN = Fraction(1, 100)
SAVING_WEIGHT = 1.0                      # replay credit per unit of mean relative tightening
HOLDOUT_BUCKETS = 5                      # one bucket in five is held out
LOCK_STALE_S = 25 * 60                   # evolver wall limit is 20 min; a lock older than 25 min is stale
EVOLVER_WALL_S = 20 * 60
CONSOLIDATE_WALL_S = 10 * 60
RESERVED_NAMES = frozenset({"envelope.toml", "stop", "pause"})

COLLECTING = "collecting_evidence"
REVIEW_REQUIRED = "review_required"
APPROVED = "approved_for_promotion"
ROLLBACK_TRIGGERED = "rollback_triggered"


class SandboxViolation(RuntimeError):
    """The learner tried to write or remove a path outside its sandbox or one that is protected."""


class QuotaExceeded(RuntimeError):
    """A write would take the learn dir over its quota."""


class LowDiskSpace(RuntimeError):
    """Free space would fall below MIN_FREE_BYTES."""


class LearnerBusy(RuntimeError):
    """Another learner run holds the lock."""


class TimeLimitExceeded(RuntimeError):
    """The run's wall-clock limit passed."""


class PatchRateLimited(RuntimeError):
    """The cell already received a patch within the last 24 h."""


class ReviewRequired(RuntimeError):
    """A widening patch needs the owner; the learner never applies one."""


class PatchConflict(RuntimeError):
    """The stored policy no longer matches what the patch expected."""


class PatchTampered(RuntimeError):
    """A stored patch no longer hashes to its evidence hash."""


class NothingToRevert(RuntimeError):
    """There is no applied, un-reverted patch."""


class PatchLogCorrupt(RuntimeError):
    """The patch log has a line that cannot be parsed; refuse rather than guess the rate limit."""


# --------------------------------------------------------------------------- filesystem and sandbox

class OsFs:
    """The filesystem operations the learner uses.  Tests substitute a recording subclass."""

    def free_bytes(self, path: Path) -> int:
        p = Path(path)
        while not p.exists() and p.parent != p:
            p = p.parent
        return shutil.disk_usage(p).free

    def tree_size(self, path: Path) -> int:
        total = 0
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
        return total

    def file_size(self, path: Path) -> int:
        try:
            return os.path.getsize(path)
        except OSError:
            return 0

    def exists(self, path: Path) -> bool:
        return Path(path).exists()

    def read_bytes(self, path: Path) -> bytes:
        return Path(path).read_bytes()

    def makedirs(self, path: Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    def write_temp(self, final: Path, data: bytes) -> Path:
        """Write ``data`` to a fresh temp file beside ``final`` (flushed and fsynced); return its path."""
        fd, tmp = tempfile.mkstemp(dir=str(Path(final).parent), prefix=Path(final).name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return Path(tmp)

    def replace(self, tmp: Path, final: Path) -> None:
        os.replace(tmp, final)

    def remove(self, path: Path) -> None:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass

    def create_exclusive(self, path: Path, data: bytes) -> bool:
        """Create ``path`` with ``data`` unless it exists; True when this call created it."""
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return True


def _norm(path: Any) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(str(path))))


def _within(target: str, root: str) -> bool:
    return target == root or target.startswith(root.rstrip("\\/") + os.sep)


class LearnWriter:
    """The only way the learner writes.  Every target is checked against the sandbox first.

    * Allowed: inside ``learn_dir`` or under one of ``also_allowed`` (the policy store, the flag registry).
    * Never: a ``protected`` path (envelope, STOP/PAUSE files, audit paths) or a reserved name under an
      allowed root (``envelope.toml``, ``STOP``, ``PAUSE``, ``audit*``).  Raises SandboxViolation.
    * The learn dir stays at or under ``quota_bytes``; free space after the write stays at or above
      ``min_free_bytes``.  Both are checked for the whole batch BEFORE anything is written.
    * Each file is written to a temp file beside it and renamed over it.
    """

    def __init__(self, learn_dir: Path, *, fs: Optional[OsFs] = None, protected: Sequence[Any] = (),
                 also_allowed: Sequence[Any] = (), quota_bytes: int = LEARN_QUOTA_BYTES,
                 min_free_bytes: int = MIN_FREE_BYTES) -> None:
        from loci.vitals import check_on_c
        self.learn_dir = check_on_c(learn_dir)  # learned data never goes to F:/D: or /mnt/*
        self.fs = fs or OsFs()
        self._protected = [_norm(p) for p in protected]
        self._learn = _norm(self.learn_dir)
        self._roots = [self._learn] + [_norm(p) for p in also_allowed]
        self.quota_bytes = quota_bytes
        self.min_free_bytes = min_free_bytes

    def check_path(self, path: Any) -> Path:
        target = _norm(path)
        for prot in self._protected:
            if _within(target, prot):
                raise SandboxViolation(f"{path} is a protected path (envelope, STOP/PAUSE or audit)")
        for root in self._roots:
            if _within(target, root):
                rel = [part.lower() for part in Path(target[len(root):].lstrip("\\/")).parts]
                if rel and (rel[-1] in RESERVED_NAMES or rel[-1].startswith("audit") or "audit" in rel[:-1]):
                    raise SandboxViolation(f"{path} has a reserved name (envelope, STOP/PAUSE or audit)")
                return Path(path)
        raise SandboxViolation(f"{path} is outside the learner sandbox ({self.learn_dir})")

    def write(self, path: Any, data: bytes) -> None:
        self.write_many([(path, data)])

    def write_many(self, items: Sequence[tuple]) -> None:
        checked = [(self.check_path(p), bytes(d)) for p, d in items]
        if not checked:
            return
        total_new = sum(len(d) for _, d in checked)
        free = self.fs.free_bytes(self.learn_dir)
        if free - total_new < self.min_free_bytes:
            raise LowDiskSpace(f"{free} bytes free; writing {total_new} would leave less than {self.min_free_bytes}")
        delta = sum(len(d) - self.fs.file_size(p) for p, d in checked if _within(_norm(p), self._learn))
        if delta > 0 and self.fs.tree_size(self.learn_dir) + delta > self.quota_bytes:
            raise QuotaExceeded(f"learn dir would exceed its {self.quota_bytes} byte quota (+{delta} bytes)")
        pending: List[tuple] = []
        try:
            for path, data in checked:
                self.fs.makedirs(path.parent)
                pending.append((self.fs.write_temp(path, data), path))
            while pending:
                tmp, final = pending[0]
                self.fs.replace(tmp, final)
                pending.pop(0)
        except BaseException:
            for tmp, _final in pending:
                try:
                    self.fs.remove(tmp)
                except OSError:
                    pass
            raise

    def remove(self, path: Any) -> None:
        self.fs.remove(self.check_path(path))


class RunLock:
    """One learner job at a time: an exclusive-create lock file in the learn dir, stale after LOCK_STALE_S."""

    def __init__(self, writer: LearnWriter, path: Path, clock: Callable[[], float] = time.time) -> None:
        self._writer = writer
        self.path = writer.check_path(path)
        self._clock = clock
        self.held = False

    def acquire(self) -> "RunLock":
        fs = self._writer.fs
        fs.makedirs(self.path.parent)
        body = json.dumps({"pid": os.getpid(), "ts": self._clock()}).encode("utf-8")
        for _ in range(2):
            if fs.create_exclusive(self.path, body):
                self.held = True
                return self
            try:
                ts = float(json.loads(fs.read_bytes(self.path).decode("utf-8"))["ts"])
            except (OSError, ValueError, KeyError, TypeError):
                ts = None  # unreadable lock: treat as stale only after the file is older than we can tell, so refuse
            if ts is not None and self._clock() - ts > LOCK_STALE_S:
                fs.remove(self.path)
                continue
            raise LearnerBusy(f"another learner run holds {self.path}")
        raise LearnerBusy(f"could not take {self.path}")

    def release(self) -> None:
        if self.held:
            self._writer.fs.remove(self.path)
            self.held = False

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self.release()


class Deadline:
    """Cooperative wall-clock limit: ``check()`` raises TimeLimitExceeded once ``seconds`` have passed."""

    def __init__(self, seconds: float, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._end = clock() + seconds
        self.seconds = seconds

    def check(self) -> None:
        if self._clock() > self._end:
            raise TimeLimitExceeded(f"wall-clock limit of {self.seconds:g}s exceeded")


def learner_blocked(vitals_state_path: Any, clock: Callable[[], float] = time.time) -> Optional[str]:
    """None when the vitals ladder is at L0; otherwise the reason the learner must not run.

    A missing, stale, unreadable or corrupt heartbeat is NOT L0 (loci.vitals treats it as L3 for learning).
    """
    from loci.vitals import read_state
    view = read_state(vitals_state_path, clock=clock)
    if view.allows("start_learning"):
        return None
    return f"vitals ladder is not L0 (status={view.status}, level={view.level}); the learner runs at L0 only"


# --------------------------------------------------------------------------- promotion state machine

@dataclass(frozen=True)
class Outcome:
    id: str
    ts: float                 # epoch seconds
    session: Optional[str]    # None when the row names no session: it can never prove independence


@dataclass(frozen=True)
class PromotionStatus:
    state: str
    reasons: tuple
    n: int
    days: int


def parse_ts(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    return None


def row_key(row: Mapping[str, Any]) -> str:
    """A stable id for a trajectory row: its own ``id`` or the sha256 of its canonical JSON."""
    rid = row.get("id")
    if isinstance(rid, str) and rid:
        return rid
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def outcomes_from_rows(rows: Iterable[Mapping[str, Any]], since: float = 0.0) -> tuple:
    """``(outcomes, skipped)``: rows at or after ``since`` (epoch s) with a usable ``_ts`` or ``ts``; the rest are
    counted in ``skipped`` (never dropped silently)."""
    out: List[Outcome] = []
    skipped = 0
    for row in rows:
        ts = parse_ts(row.get("_ts", row.get("ts")))
        if ts is None:
            skipped += 1
            continue
        if ts < since:
            continue
        session = row.get("session_id", row.get("session"))
        out.append(Outcome(row_key(row), ts, str(session) if session not in (None, "") else None))
    return out, skipped


def independent_windows(outcomes: Iterable[Outcome]) -> bool:
    """True when two outcomes come from different sessions and lie at least WINDOW_GAP_S apart."""
    spans: Dict[str, List[float]] = {}
    for o in outcomes:
        if o.session is None:
            continue
        span = spans.setdefault(o.session, [o.ts, o.ts])
        span[0] = min(span[0], o.ts)
        span[1] = max(span[1], o.ts)
    if len(spans) < 2:
        return False
    top = sorted(((hi, s) for s, (lo, hi) in spans.items()), reverse=True)[:2]
    for s, (lo, _hi) in spans.items():
        best_hi = top[0][0] if top[0][1] != s else top[1][0]
        if best_hi - lo >= WINDOW_GAP_S:
            return True
    return False


def evaluate_promotion(outcomes: Iterable[Outcome], *, widening: bool = False, stage0_passed: bool = False,
                       degeneracy: Optional[Verdict] = None) -> PromotionStatus:
    """The state of one patch's evidence.

    Precedence: a DEGENERATE verdict is ``rollback_triggered``; a widening patch is ``review_required`` whatever
    the evidence; otherwise ``approved_for_promotion`` needs ALL of: at least 30 distinct outcomes, over at least
    3 distinct UTC days, two independent windows (different session, at least 1 h apart) and a Stage-0 pass.
    Anything less is ``collecting_evidence`` with the missing pieces listed.
    """
    unique = {o.id: o for o in outcomes}
    n = len(unique)
    days = len({datetime.fromtimestamp(o.ts, tz=timezone.utc).date() for o in unique.values()})
    if degeneracy is not None and degeneracy.label == DEGENERATE:
        return PromotionStatus(ROLLBACK_TRIGGERED, tuple(degeneracy.reasons) or ("degenerate",), n, days)
    if widening:
        return PromotionStatus(REVIEW_REQUIRED, ("widening_patch_needs_owner",), n, days)
    missing = []
    if n < MIN_OUTCOMES:
        missing.append(f"outcomes {n} < {MIN_OUTCOMES}")
    if days < MIN_DAYS:
        missing.append(f"days {days} < {MIN_DAYS}")
    if not independent_windows(unique.values()):
        missing.append("no_independent_windows")
    if not stage0_passed:
        missing.append("stage0_not_passed")
    if missing:
        return PromotionStatus(COLLECTING, tuple(missing), n, days)
    return PromotionStatus(APPROVED, (), n, days)


# --------------------------------------------------------------------------- replay objective and the ES

def split_rows(rows: Sequence[Mapping[str, Any]]) -> tuple:
    """``(train, heldout)``: a row is held out when sha256(row key) falls in 1 of HOLDOUT_BUCKETS buckets.
    Deterministic and independent of row order."""
    train, held = [], []
    for row in rows:
        bucket = int(hashlib.sha256(row_key(row).encode("utf-8")).hexdigest()[:8], 16) % HOLDOUT_BUCKETS
        (held if bucket == 0 else train).append(row)
    return train, held


def replay_score(policy: Policy, genome: Mapping[str, float], current: Mapping[str, float],
                 rows: Sequence[Mapping[str, Any]]) -> float:
    """Mean replay score of ``genome`` over ``rows`` (a deterministic proxy, not a model call).

    Per row: ``-1.0`` (the reward hinge) when the tightened limits would have cut it off, i.e. the row reports
    a ``usage[<tier limit>]`` above the genome's limit, or a gene the genome tightened relative to ``current``
    has no reported usage (unknown is treated as binding).  Otherwise the row's own reward score plus
    SAVING_WEIGHT times the mean relative tightening of the genome versus ``current``.
    """
    if not rows:
        return 0.0
    scorer = TrajectoryScorer.from_policy(policy)
    limits = apply_genes(policy.get_constraints(), genome)
    saving = sum(1.0 - genome.get(g.name, g.default) / current.get(g.name, g.default) for g in GENES) / len(GENES)
    total = 0.0
    for row in rows:
        usage = row.get("usage") if isinstance(row.get("usage"), Mapping) else {}
        binding = False
        for g in GENES:
            used = usage.get(g.tier_key)
            if isinstance(used, bool) or not isinstance(used, (int, float)):
                if genome.get(g.name, g.default) < current.get(g.name, g.default):
                    binding = True
            elif g.tier_key in limits and used > limits[g.tier_key]:
                binding = True
        total += HINGE_REWARD if binding else scorer.score([row]) + SAVING_WEIGHT * saving
    return total / len(rows)


def _round6(x: float) -> float:
    return round(x, 6)


def _ceil6(x: Fraction) -> float:
    return math.ceil(x * 10 ** 6) / 10 ** 6


def _candidate(current: Mapping[str, float], rng: random.Random, sigma: float) -> Dict[str, float]:
    """One log-space ES perturbation of ``current``: tighten-only, at most 10% per gene, inside [floor, default]."""
    out: Dict[str, float] = {}
    for g in GENES:
        cur = current.get(g.name, g.default)
        factor = math.exp(sigma * rng.gauss(0.0, 1.0))
        lowest = max(g.floor, _ceil6(_q(cur) * (1 - GENE_STEP_MAX)))
        value = _round6(min(cur, cur * factor))
        out[g.name] = min(max(value, lowest), cur)
    return out


@dataclass(frozen=True)
class EsResult:
    accepted: bool
    reason: str
    genome: Dict[str, float]      # the best candidate (equal to ``current`` when nothing moved)
    train_gain: float
    heldout_gain: Optional[float]  # None when the held-out rows were never scored
    discarded_nonfinite: int


def _gain_at_least(gain: float, minimum: Fraction) -> bool:
    """``gain >= minimum`` decided on the gain rounded to 12 places, so 0.52 - 0.5 passes 0.02 exactly."""
    return _q(round(gain, 12)) >= minimum


def evolve_genome(current: Mapping[str, float], rng: random.Random,
                  train_score: Callable[[Mapping[str, float]], float],
                  heldout_score: Callable[[Mapping[str, float]], float],
                  *, pop: int = ES_POP, sigma: float = ES_SIGMA) -> EsResult:
    """One evolution-strategy generation.  Pick the best of ``pop`` tighten-only candidates on the TRAIN score;
    accept it only if its train gain over ``current`` is >= 0.02 AND its held-out gain is >= 0.01.  The
    held-out score is computed once, for the winner only.  Non-finite candidate scores are discarded and
    counted."""
    cur = {g.name: current.get(g.name, g.default) for g in GENES}
    base_train = train_score(cur)
    if not math.isfinite(base_train):
        return EsResult(False, "nonfinite_baseline", cur, 0.0, None, 0)
    best, best_score, discarded = None, None, 0
    for _ in range(pop):
        cand = _candidate(cur, rng, sigma)
        score = train_score(cand)
        if not math.isfinite(score):
            discarded += 1
            continue
        if best is None or score > best_score:
            best, best_score = cand, score
    if best is None or best == cur:
        return EsResult(False, "no_change", cur, 0.0, None, discarded)
    train_gain = best_score - base_train
    if not _gain_at_least(train_gain, TRAIN_GAIN_MIN):
        return EsResult(False, "train_gain_below_min", best, train_gain, None, discarded)
    held_gain = heldout_score(best) - heldout_score(cur)
    if not math.isfinite(held_gain):
        return EsResult(False, "nonfinite_heldout", best, train_gain, None, discarded)
    if not _gain_at_least(held_gain, HELDOUT_GAIN_MIN):
        return EsResult(False, "heldout_gain_below_min", best, train_gain, held_gain, discarded)
    return EsResult(True, "accepted", best, train_gain, held_gain, discarded)


# --------------------------------------------------------------------------- patches, rate limit, revert

def evidence_hash(evidence: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of ``evidence`` (the same bytes as scripts/policy_prereg ``record_hash``)."""
    return hashlib.sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Patch:
    id: str
    cell: str
    ts: float
    previous: Dict[str, Optional[float]]   # stored value before the patch; None = the key was absent
    proposed: Dict[str, float]             # stored value after the patch (only the genes that changed)
    evidence: Dict[str, Any]
    evidence_hash: str
    widening: bool

    def to_entry(self) -> Dict[str, Any]:
        return {"type": "patch", "id": self.id, "cell": self.cell, "ts": self.ts, "evidence": self.evidence,
                "evidence_hash": self.evidence_hash, "widening": self.widening}

    @classmethod
    def from_entry(cls, entry: Mapping[str, Any]) -> "Patch":
        ev = entry["evidence"]
        return cls(entry["id"], entry["cell"], entry["ts"], dict(ev["previous"]), dict(ev["proposed"]), dict(ev),
                   entry["evidence_hash"], bool(entry.get("widening", False)))


def plan_patch(cell: str, stored_genes: Mapping[str, float], proposed_genome: Mapping[str, float], now: float,
               details: Optional[Mapping[str, Any]] = None) -> Optional[Patch]:
    """The patch taking the cell's stored genes to ``proposed_genome``, or None when nothing changes.

    GeneBoundsError when a value is out of bounds or tightens by more than 10%; ``widening`` is set when any
    gene would loosen (such a patch can only ever be review_required).
    """
    effective = {g.name: stored_genes.get(g.name, g.default) for g in GENES}
    changed = {name: value for name, value in proposed_genome.items() if value != effective.get(name)}
    if not changed:
        return None
    widening = check_genome_step(effective, changed)
    evidence = {
        "cell": cell, "ts": now,
        "previous": {name: stored_genes.get(name) for name in changed},
        "proposed": dict(changed),
        "details": dict(details or {}),
    }
    digest = evidence_hash(evidence)
    return Patch(digest[:16], cell, now, evidence["previous"], evidence["proposed"], evidence, digest, widening)


class PatchLog:
    """Append-only JSONL of patches, reverts and failed applications, kept in the learn dir."""

    def __init__(self, path: Path, writer: LearnWriter) -> None:
        self.path = writer.check_path(path)
        self._writer = writer

    def entries(self) -> List[Dict[str, Any]]:
        if not self._writer.fs.exists(self.path):
            return []
        out = []
        raw = self._writer.fs.read_bytes(self.path).decode("utf-8")
        for lineno, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                entry["type"]
            except (ValueError, KeyError, TypeError) as exc:
                raise PatchLogCorrupt(f"{self.path}:{lineno}: unreadable patch log line") from exc
            out.append(entry)
        return out

    def append(self, entry: Mapping[str, Any]) -> None:
        old = self._writer.fs.read_bytes(self.path) if self._writer.fs.exists(self.path) else b""
        self._writer.write(self.path, old + json.dumps(entry, sort_keys=True).encode("utf-8") + b"\n")

    def check_rate(self, cell: str, now: float) -> None:
        """PatchRateLimited unless the cell's last patch (reverted or not) is at least 24 h old."""
        stamps = [e["ts"] for e in self.entries() if e["type"] == "patch" and e["cell"] == cell]
        if stamps and now - max(stamps) < PATCH_COOLDOWN_S:
            raise PatchRateLimited(f"cell {cell!r} was patched {now - max(stamps):.0f}s ago; at most 1 patch per 24 h")

    def applied_patches(self) -> List[Patch]:
        """Patches that were applied and not reverted, oldest first."""
        entries = self.entries()
        dead = {e["patch_id"] for e in entries if e["type"] in ("revert", "patch_failed")}
        return [Patch.from_entry(e) for e in entries if e["type"] == "patch" and e["id"] not in dead]


def apply_patch(patch: Patch, *, backend: PolicyBackend, writer: LearnWriter, log: PatchLog, now: float,
                owner_approved: bool = False) -> None:
    """Write ``patch`` into the cell's policy (``config['learned_genes']``).  Never called in dry-run.

    ReviewRequired for a widening patch without ``owner_approved``; PatchRateLimited inside 24 h of the
    cell's last patch; PatchConflict if the stored genes are not the ones the patch started from.  The log
    entry is written FIRST so the rate limit and the revert record survive a failed policy write.
    """
    if patch.widening and not owner_approved:
        raise ReviewRequired(f"patch {patch.id} widens a gene: review_required, the owner must apply it")
    log.check_rate(patch.cell, now)
    policy = backend.load(patch.cell)
    if policy is None:
        raise PatchConflict(f"policy {patch.cell!r} does not exist")
    stored = dict(policy.config.get(LEARNED_GENES_KEY) or {})
    for name, before in patch.previous.items():
        if stored.get(name) != before:
            raise PatchConflict(f"gene {name} of {patch.cell!r} is {stored.get(name)!r}, patch expected {before!r}")
    log.append(patch.to_entry())
    stored.update(patch.proposed)
    policy.config[LEARNED_GENES_KEY] = stored
    policy.metadata.version += 1
    policy.metadata.source = "bounded-evolver"
    policy.metadata.updated_at = datetime.utcnow().isoformat() + "Z"
    try:
        backend.store(policy, writer=writer.write)
    except BaseException:
        try:
            log.append({"type": "patch_failed", "patch_id": patch.id, "ts": now})
        except Exception:
            logger.exception("could not record the failed patch %s", patch.id)
        raise


def flip_flag_off(writer: LearnWriter, registry_path: Path, name: str, reason: str, review_by: date) -> None:
    """Register (or refresh) ``name`` as default-off in the learned-flag registry, with ``review_by``.

    The registry lists features that are OFF and why; removing an entry is the owner's act, never the learner's.
    """
    path = Path(registry_path)
    text = writer.fs.read_bytes(path).decode("utf-8") if writer.fs.exists(path) else ""
    lines = text.splitlines(keepends=True)
    heads = [i for i, ln in enumerate(lines) if ln.strip() == "[[flag]]"]
    ends = [(heads[k + 1] if k + 1 < len(heads) else len(lines)) for k in range(len(heads))]
    new_reason = f"reason = {json.dumps(reason)}\n"
    new_review = f"review_by = {review_by.isoformat()}\n"
    target = None
    for start, end in zip(heads, ends):
        if any(ln.strip() == f"name = {json.dumps(name)}" for ln in lines[start:end]):
            target = (start, end)
            break
    if target is None:
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines += ["\n", "[[flag]]\n", f"name = {json.dumps(name)}\n", new_reason, new_review]
    else:
        start, end = target
        block = [ln for ln in lines[start:end] if not ln.lstrip().startswith(("reason", "review_by"))]
        while block and not block[-1].strip():
            block.pop()
        block += [new_reason, new_review, "\n"] if end < len(lines) else [new_reason, new_review]
        lines[start:end] = block
    new_text = "".join(lines)
    parsed = tomllib.loads(new_text).get("flag", [])
    if not any(f.get("name") == name and f.get("review_by") == review_by for f in parsed):
        raise ValueError(f"refusing to write a registry that does not hold {name!r} with review_by {review_by}")
    writer.write(path, new_text.encode("utf-8"))


def revert_last_patch(*, backend: PolicyBackend, writer: LearnWriter, log: PatchLog, registry_path: Path,
                      now: float, review_days: int = 30) -> Dict[str, Any]:
    """One command: restore the exact previous gene values of the last applied patch, record the revert, and
    register ``learned_genes`` as OFF with a ``review_by`` date.

    PatchTampered when the stored evidence no longer hashes to the stored evidence hash (nothing is touched);
    PatchConflict when the cell's genes are neither the patch's nor the restored values.
    """
    applied = log.applied_patches()
    if not applied:
        raise NothingToRevert("no applied patch to revert")
    patch = applied[-1]
    if evidence_hash(patch.evidence) != patch.evidence_hash:
        raise PatchTampered(f"patch {patch.id} no longer hashes to its evidence hash; refusing to revert from it")
    policy = backend.load(patch.cell)
    if policy is None:
        raise PatchConflict(f"policy {patch.cell!r} does not exist")
    stored = dict(policy.config.get(LEARNED_GENES_KEY) or {})
    restored = dict(stored)
    for name, before in patch.previous.items():
        if before is None:
            restored.pop(name, None)
        else:
            restored[name] = before
    if stored != restored:
        for name, after in patch.proposed.items():
            if stored.get(name) != after:
                raise PatchConflict(f"gene {name} of {patch.cell!r} is {stored.get(name)!r}, expected {after!r}; "
                                    f"the policy changed since the patch")
        if restored:
            policy.config[LEARNED_GENES_KEY] = restored
        else:
            policy.config.pop(LEARNED_GENES_KEY, None)
        policy.metadata.version += 1
        policy.metadata.source = "bounded-evolver-revert"
        policy.metadata.updated_at = datetime.utcnow().isoformat() + "Z"
        backend.store(policy, writer=writer.write)
    log.append({"type": "revert", "patch_id": patch.id, "cell": patch.cell, "ts": now})
    review_by = datetime.fromtimestamp(now, tz=timezone.utc).date() + timedelta(days=review_days)
    flip_flag_off(writer, registry_path, LEARNED_GENES_FLAG,
                  f"learned gene patch {patch.id} on {patch.cell} was reverted; re-review before enabling", review_by)
    return {"patch_id": patch.id, "cell": patch.cell, "restored": {n: patch.previous[n] for n in patch.previous},
            "review_by": review_by.isoformat()}


# --------------------------------------------------------------------------- OS resource limits

LEARNER_MEMORY_BYTES = 512 * 1024 * 1024   # <= 512 MiB, enforced by the job object on Windows
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


class LimitsNotApplied(RuntimeError):
    """A resource cap could not be put in place; the learner refuses to run uncapped."""


class OsLimits:
    """Puts the learner's caps on THIS process: BELOW_NORMAL priority and its own Job Object (<= 512 MiB, no
    child processes, killed with the job) on Windows; ``nice`` on POSIX.  Tests inject a recording fake;
    the real class is exercised by hand in a throwaway subprocess (see the test module header)."""

    def set_low_priority(self) -> str:
        if os.name == "nt":
            import ctypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            if not k32.SetPriorityClass(k32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS):
                raise OSError(ctypes.get_last_error(), "SetPriorityClass failed")
            return "below_normal"
        os.nice(10)
        return "nice+10"

    def bind_memory_job(self, memory_bytes: int) -> str:
        if os.name != "nt":
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            return f"rlimit_as={memory_bytes}"
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in ("r_ops", "w_ops", "o_ops", "r_bytes", "w_bytes", "o_bytes")]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("Basic", BasicLimits), ("Io", IoCounters), ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = ctypes.c_void_p
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        k32.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        job = k32.CreateJobObjectW(None, None)
        if not job:
            raise OSError(ctypes.get_last_error(), "CreateJobObject failed")
        info = ExtendedLimits()
        # JOB_MEMORY 0x200 | PROCESS_MEMORY 0x100 | ACTIVE_PROCESS 0x8 | KILL_ON_JOB_CLOSE 0x2000
        info.Basic.LimitFlags = 0x200 | 0x100 | 0x8 | 0x2000
        info.Basic.ActiveProcessLimit = 1
        info.JobMemoryLimit = memory_bytes
        info.ProcessMemoryLimit = memory_bytes
        if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
        if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
        return f"job_memory={memory_bytes}"


def apply_os_limits(limits: Optional[OsLimits] = None, *, memory_bytes: int = LEARNER_MEMORY_BYTES,
                    environ: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Apply every cap or raise LimitsNotApplied; returns what was applied (reported, never silent)."""
    limits = limits or OsLimits()
    env = os.environ if environ is None else environ
    for name in _THREAD_ENV:
        env[name] = "1"
    try:
        priority = limits.set_low_priority()
        memory = limits.bind_memory_job(memory_bytes)
    except Exception as exc:
        raise LimitsNotApplied(f"resource caps not applied ({type(exc).__name__}: {exc}); refusing to run uncapped") from exc
    return {"priority": priority, "memory": memory, "threads": 1}
