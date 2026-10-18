"""The pass-rule arithmetic of the learned-router preregistration (design.md sections 5.5 and 6).

Pure functions on counts, no I/O. Rates and thresholds that decide a boundary are exact ``Fraction`` s, so
"within 1 pp" and "down at least 10%" are decided without float noise.
"""
from __future__ import annotations

import math
from fractions import Fraction
from statistics import NormalDist
from typing import Mapping, Sequence, Union

__all__ = [
    "StatsError", "required_n_per_arm", "wilson_lower_bound", "diff_lower_bound",
    "noninferiority_pvalue", "holm", "evaluate_guards",
]

_N = NormalDist()

Number = Union[int, float, Fraction]


class StatsError(ValueError):
    """An argument outside the domain of the arithmetic (n <= 0, p outside (0,1), ...)."""


def _q(x: Number) -> Fraction:
    """Exact value of a number as the user wrote it: 0.91 means 91/100, not its binary neighbour."""
    return x if isinstance(x, Fraction) else Fraction(str(x))


def required_n_per_arm(p: float, margin: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Smallest n per arm for a one-sided non-inferiority test at true difference 0.

    ``n = ceil((z_{1-alpha} + z_{power})^2 * 2 p (1-p) / margin^2)`` for two arms with the same success
    rate ``p``; ``margin`` is the (positive) tolerated drop, e.g. 0.08 for the preregistered -0.08.
    """
    if not (0.0 < p < 1.0) or margin <= 0 or not (0.0 < alpha < 1.0) or not (0.0 < power < 1.0):
        raise StatsError(f"required_n_per_arm: bad arguments p={p!r} margin={margin!r} alpha={alpha!r} power={power!r}")
    z = _N.inv_cdf(1.0 - alpha) + _N.inv_cdf(power)
    return math.ceil(z * z * 2.0 * p * (1.0 - p) / (margin * margin))


def wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    """Wilson score lower bound of a binomial proportion (``z`` = 1.96 two-sided 95%, 1.6449 one-sided)."""
    if n <= 0 or not (0 <= successes <= n):
        raise StatsError(f"wilson_lower_bound: need 0 <= successes <= n and n > 0, got {successes!r}/{n!r}")
    p = successes / n
    z2 = z * z
    centre = p + z2 / (2.0 * n)
    spread = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, (centre - spread) / (1.0 + z2 / n))


def diff_lower_bound(arm_k: int, arm_n: int, inc_k: int, inc_n: int, alpha: float = 0.05) -> float:
    """One-sided (1 - alpha) lower bound on ``p_arm - p_incumbent`` (Wald standard error of the difference)."""
    if arm_n <= 0 or inc_n <= 0 or not (0 <= arm_k <= arm_n) or not (0 <= inc_k <= inc_n):
        raise StatsError("diff_lower_bound: counts out of range")
    pa, pi = arm_k / arm_n, inc_k / inc_n
    se = math.sqrt(pa * (1.0 - pa) / arm_n + pi * (1.0 - pi) / inc_n)
    return (pa - pi) - _N.inv_cdf(1.0 - alpha) * se


def noninferiority_pvalue(arm_k: int, arm_n: int, inc_k: int, inc_n: int, margin: float) -> float:
    """One-sided p-value of H0: ``p_arm - p_inc <= -margin`` (margin positive, e.g. 0.08)."""
    if margin <= 0 or arm_n <= 0 or inc_n <= 0:
        raise StatsError("noninferiority_pvalue: margin and n must be positive")
    pa, pi = arm_k / arm_n, inc_k / inc_n
    se = math.sqrt(pa * (1.0 - pa) / arm_n + pi * (1.0 - pi) / inc_n)
    if se == 0.0:
        return 0.0 if (pa - pi) > -margin else 1.0
    return 1.0 - _N.cdf(((pa - pi) + margin) / se)


def holm(pvalues: Sequence[float], alpha: float = 0.05) -> tuple[list[bool], list[float]]:
    """Holm step-down. Returns (reject flags, adjusted p-values), both in input order."""
    m = len(pvalues)
    if m == 0:
        raise StatsError("holm: no p-values")
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return [a <= alpha for a in adjusted], adjusted


# Guard thresholds of the pass rule (design.md section 6), as exact fractions.
_PP = Fraction(1, 100)


def evaluate_guards(incumbent: Mapping[str, Number], arm: Mapping[str, Number]) -> dict[str, bool]:
    """The non-statistical guards of the pass rule; every value is True when the guard PASSES.

    Inputs per side: ``cost_per_accepted``, ``completion``, ``refusal``, ``p95_latency``,
    ``cache_read_share`` (rates as fractions 0..1) and, for the arm, ``cap_breaches`` and
    ``governor_escalations`` (counts attributable to it).

    * cost per accepted answer down by at least 10%;
    * completion >= incumbent - 1 pp;  refusal <= incumbent + 1 pp  (exactly 1 pp passes);
    * p95 latency <= 1.25 x incumbent;
    * cache-read share not lower by more than 3 pp;
    * zero cap breaches and zero governor escalations.
    """
    return {
        "cost_down_10pct": _q(arm["cost_per_accepted"]) <= _q(incumbent["cost_per_accepted"]) * Fraction(9, 10),
        "completion": _q(arm["completion"]) >= _q(incumbent["completion"]) - _PP,
        "refusal": _q(arm["refusal"]) <= _q(incumbent["refusal"]) + _PP,
        "p95_latency": _q(arm["p95_latency"]) <= _q(incumbent["p95_latency"]) * Fraction(5, 4),
        "cache_read_share": _q(arm["cache_read_share"]) >= _q(incumbent["cache_read_share"]) - 3 * _PP,
        "no_cap_breaches": _q(arm.get("cap_breaches", 0)) == 0,
        "no_governor_escalations": _q(arm.get("governor_escalations", 0)) == 0,
    }
