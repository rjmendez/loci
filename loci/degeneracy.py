"""Degeneracy judge for a learned routing arm against the frozen incumbent (Tripwire-Governor S15).

Pure functions, stdlib only, no I/O and no clock. The verdict is one of:

* ``healthy``     -- no check fired and the arm is past burn-in;
* ``degenerate``  -- at least one check fired; the caller swaps in the frozen incumbent and the state
  NEVER auto-resumes (resuming is an explicit operator act, see scripts/policy_watchdog.py);
* ``too_early``   -- fewer than ``MIN_ARM_N`` outcomes for the arm (or the incumbent): no verdict yet.
  A non-finite value is the one exception: it is degenerate at any n, because it cannot be repaired by
  more data.

Checks (design.md section 6, "Degeneracy judge"). Rates are exact ``Fraction`` s, so "more than 1 pp"
is decided without float noise and exactly 1 pp is NOT a breach:

* ``nonfinite``                    any NaN/inf in the arm's cost, reward or quality;
* ``completion_below_incumbent``   completion rate < incumbent's - 1 pp;
* ``refusal_above_incumbent``      refusal-or-empty rate > incumbent's + 1 pp;
* ``cheap_by_refusing``            cheaper per call than the incumbent, refuses more, and NOT cheaper per
  answer actually served: the saving is the refusals, not a better route;
* ``escalation_drift``             escalation rate above the incumbent's by more than 2 SE (pooled
  two-proportion z, strictly greater than 2);
* ``reward_up_quality_down``       mean reward above the incumbent's while mean quality is more than
  1 pp below it (Goodhart: the arm gamed the reward).

``cell_reward`` is the per-call reward ``r = q_hat - lambda * cost_norm`` with the hinge ``r = -1`` when
``q_hat`` is below the cell's quality floor. ``PageHinkley`` is the drift detector the learning loop uses
to reset a cell.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Mapping, Optional

__all__ = [
    "HEALTHY", "DEGENERATE", "TOO_EARLY", "MIN_ARM_N", "HINGE_REWARD", "LAMBDA_MAX",
    "DegeneracyInputError", "ArmStats", "Verdict", "cell_reward", "summarize_arm", "judge_arm",
    "PageHinkley",
]

HEALTHY = "healthy"
DEGENERATE = "degenerate"
TOO_EARLY = "too_early"

MIN_ARM_N = 30  # no cell leaves its prior before 30 labelled outcomes (model_pool.MIN_ARM_N)
PP = Fraction(1, 100)  # the 1 percentage-point guard from the pass rule
ESCALATION_Z = 2.0  # "escalation drift > 2 SE"
QUALITY_TOL = Fraction(1, 100)  # "quality down" means more than 1 pp lower
HINGE_REWARD = -1.0
LAMBDA_MAX = 5.0  # lambda is clipped to [0, 5]


class DegeneracyInputError(ValueError):
    """The judge was handed something it cannot judge (a bad incumbent or a malformed row)."""


@dataclass(frozen=True)
class ArmStats:
    """Outcome counts and means for one arm over the judged window."""

    name: str
    n: int
    completed: int  # calls that finished with a usable response (not error/timeout)
    refused: int  # calls whose answer was a refusal or empty
    escalated: int  # calls the governor escalated
    cost_total: float  # USD over all n calls
    mean_reward: Optional[float]  # None when no call carried a quality label
    mean_quality: Optional[float]

    def __post_init__(self) -> None:
        if self.n < 0 or not (0 <= self.completed <= self.n and 0 <= self.refused <= self.n
                              and 0 <= self.escalated <= self.n):
            raise DegeneracyInputError(f"arm {self.name!r}: counts out of range (n={self.n})")

    def rate(self, count: int) -> Fraction:
        return Fraction(count, self.n) if self.n else Fraction(0)


@dataclass(frozen=True)
class Verdict:
    label: str
    reasons: tuple[str, ...]


def _finite(x: Optional[float]) -> bool:
    return x is None or math.isfinite(x)


def cell_reward(q_hat: float, cost_norm: float, lam: float, quality_floor: float) -> float:
    """``q_hat - lam * cost_norm``; exactly ``-1.0`` when ``q_hat`` is below ``quality_floor`` (the hinge).

    ``lam`` is clipped to [0, LAMBDA_MAX]. A non-finite input yields NaN, which the judge reports as
    ``nonfinite`` rather than hiding.
    """
    if not (math.isfinite(q_hat) and math.isfinite(cost_norm) and math.isfinite(lam)):
        return math.nan
    if q_hat < quality_floor:
        return HINGE_REWARD
    return q_hat - min(max(lam, 0.0), LAMBDA_MAX) * cost_norm


def summarize_arm(name: str, rows: Iterable[Mapping], *, lam: float = 1.0, cost_scale: float = 0.01,
                  quality_floor: float = 0.5) -> ArmStats:
    """Fold policy rows into ``ArmStats``.

    A row is ``{"completed": bool, "refused": bool, "escalated": bool, "cost_usd": float,
    "quality": float | None}``. A refused row counts as quality 0.0 (it answered nothing), so it hits the
    reward hinge; a row with ``quality`` None and not refused is unlabelled and joins no mean.
    ``cost_norm = cost_usd / cost_scale``. NaN/inf in a row is propagated, not filtered.
    """
    if not (cost_scale > 0):
        raise DegeneracyInputError(f"cost_scale must be > 0, got {cost_scale!r}")
    n = completed = refused = escalated = 0
    cost = 0.0
    rewards: list[float] = []
    qualities: list[float] = []
    for row in rows:
        try:
            cost_usd = float(row["cost_usd"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DegeneracyInputError(f"arm {name!r}: row without a numeric cost_usd: {row!r}") from exc
        n += 1
        completed += bool(row.get("completed"))
        is_refused = bool(row.get("refused"))
        refused += is_refused
        escalated += bool(row.get("escalated"))
        cost += cost_usd
        q = 0.0 if is_refused else row.get("quality")
        if q is not None:
            q = float(q)
            qualities.append(q)
            rewards.append(cell_reward(q, cost_usd / cost_scale, lam, quality_floor))
    return ArmStats(
        name=name, n=n, completed=completed, refused=refused, escalated=escalated, cost_total=cost,
        mean_reward=sum(rewards) / len(rewards) if rewards else None,
        mean_quality=sum(qualities) / len(qualities) if qualities else None,
    )


def _escalation_z(arm: ArmStats, inc: ArmStats) -> float:
    """Pooled two-proportion z of (arm - incumbent) escalation rate; 0.0 when there is no variance."""
    pooled = Fraction(arm.escalated + inc.escalated, arm.n + inc.n)
    var = float(pooled * (1 - pooled)) * (1.0 / arm.n + 1.0 / inc.n)
    if var <= 0.0:
        return 0.0
    return float(arm.rate(arm.escalated) - inc.rate(inc.escalated)) / math.sqrt(var)


def judge_arm(arm: ArmStats, incumbent: ArmStats, *, min_n: int = MIN_ARM_N) -> Verdict:
    """Classify ``arm`` against the frozen ``incumbent``. See the module docstring for the checks."""
    if not (_finite(incumbent.cost_total) and _finite(incumbent.mean_reward) and _finite(incumbent.mean_quality)):
        raise DegeneracyInputError(f"incumbent {incumbent.name!r} has a non-finite value; cannot judge against it")
    if not (_finite(arm.cost_total) and _finite(arm.mean_reward) and _finite(arm.mean_quality)):
        return Verdict(DEGENERATE, ("nonfinite",))
    if arm.n < min_n or incumbent.n < min_n:
        return Verdict(TOO_EARLY, ("burn_in",))

    reasons: list[str] = []
    if arm.rate(arm.completed) < incumbent.rate(incumbent.completed) - PP:
        reasons.append("completion_below_incumbent")
    if arm.rate(arm.refused) > incumbent.rate(incumbent.refused) + PP:
        reasons.append("refusal_above_incumbent")

    served_arm, served_inc = arm.n - arm.refused, incumbent.n - incumbent.refused
    if (served_arm > 0 and served_inc > 0
            and arm.cost_total * incumbent.n < incumbent.cost_total * arm.n  # cheaper per call
            and arm.rate(arm.refused) > incumbent.rate(incumbent.refused)  # by refusing more
            and arm.cost_total * served_inc >= incumbent.cost_total * served_arm):  # not cheaper per answer
        reasons.append("cheap_by_refusing")

    if _escalation_z(arm, incumbent) > ESCALATION_Z:
        reasons.append("escalation_drift")

    if (arm.mean_reward is not None and incumbent.mean_reward is not None
            and arm.mean_quality is not None and incumbent.mean_quality is not None
            and arm.mean_reward > incumbent.mean_reward
            and arm.mean_quality < incumbent.mean_quality - float(QUALITY_TOL)):
        reasons.append("reward_up_quality_down")

    return Verdict(DEGENERATE if reasons else HEALTHY, tuple(reasons))


class PageHinkley:
    """Two-sided Page-Hinkley mean-shift detector. ``update(x)`` returns True when a shift is detected,
    then resets itself so the next shift can be detected.

    With the design's ``delta = 0.005`` a persistent shift of +0.1 accumulates ~0.095 per sample, so for
    samples of sd 0.03 and the default ``threshold = 1.5`` it fires within ~20 samples (60 seeds: worst 20),
    while 5,000 stationary samples never fire it (60 seeds: 0 alarms). Calibrated, and pinned, in
    tests/test_degeneracy.py.
    """

    def __init__(self, delta: float = 0.005, threshold: float = 1.5) -> None:
        if delta < 0 or threshold <= 0:
            raise DegeneracyInputError(f"PageHinkley needs delta >= 0 and threshold > 0, got {delta!r}, {threshold!r}")
        self.delta = delta
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self._mean = 0.0
        self._up = 0.0
        self._up_min = 0.0
        self._down = 0.0
        self._down_min = 0.0

    def update(self, x: float) -> bool:
        if not math.isfinite(x):
            raise DegeneracyInputError(f"PageHinkley sample is not finite: {x!r}")
        self.n += 1
        self._mean += (x - self._mean) / self.n
        self._up += x - self._mean - self.delta
        self._down += self._mean - x - self.delta
        self._up_min = min(self._up_min, self._up)
        self._down_min = min(self._down_min, self._down)
        if self._up - self._up_min > self.threshold or self._down - self._down_min > self.threshold:
            self.reset()
            return True
        return False
