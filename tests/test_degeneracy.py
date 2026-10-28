"""loci/degeneracy.py: healthy / degenerate / too_early verdicts, the r=-1 hinge, Page-Hinkley.

Pure functions, so no stubs are needed; every fixture is built in the test.

Mutation log (each applied to loci/degeneracy.py; the named tests went red; then reverted):
- cell_reward without the r=-1 hinge: test_hinge_gives_exactly_minus_one_below_the_floor,
  test_an_arm_that_refuses_everything_gets_reward_minus_one_and_is_degenerate
- PageHinkley default delta 0.005 -> 0.06: test_page_hinkley_fires_within_25_samples_of_a_plus_0_1_shift,
  test_page_hinkley_fires_at_the_pinned_sample_for_seed_0, test_page_hinkley_detects_a_downward_shift_too
- refusal check ">" -> ">=" (so exactly +1 pp breaches): test_refusal_rate_exactly_1pp_above_is_healthy_and_more_is_degenerate
- completion check "<" -> "<=": test_completion_exactly_1pp_below_is_healthy_and_more_is_degenerate
- burn-in "<" -> "<=": test_burn_in_boundary_is_exactly_min_n
- escalation z threshold 2.0 -> 1.9 and -> 2.2: test_escalation_drift_is_decided_at_two_standard_errors
- cheap_by_refusing check deleted: test_cheap_by_refusing_is_degenerate_but_genuinely_cheaper_is_healthy
- reward_up_quality_down check deleted: test_reward_up_while_quality_down_is_degenerate
- the nonfinite short-circuit deleted: test_nonfinite_values_are_degenerate_even_before_burn_in, test_a_nonfinite_learned_arm_is_degenerate (watchdog)
- ArmStats.rate returns float count/n instead of an exact Fraction: test_one_pp_boundaries_are_exact_where_float_arithmetic_is_not
- hinge removed also surfaces in test_an_arm_that_refuses_everything_gets_reward_minus_one_and_is_degenerate
"""
from __future__ import annotations

import math
import random

import pytest

from loci.degeneracy import (
    DEGENERATE, HEALTHY, MIN_ARM_N, TOO_EARLY, ArmStats, DegeneracyInputError, PageHinkley,
    cell_reward, judge_arm, summarize_arm,
)


def _arm(name="arm", n=100, completed=None, refused=0, escalated=0, cost=10.0, reward=0.5, quality=0.8):
    return ArmStats(name=name, n=n, completed=n if completed is None else completed, refused=refused,
                    escalated=escalated, cost_total=cost, mean_reward=reward, mean_quality=quality)


INCUMBENT = _arm("incumbent", completed=90, refused=5, escalated=10, cost=12.5, reward=0.5, quality=0.85)


def _rows(n, *, refused=False, completed=True, escalated=False, cost=0.0, quality=0.9):
    return [{"completed": completed, "refused": refused, "escalated": escalated, "cost_usd": cost,
             "quality": None if refused else quality} for _ in range(n)]


# ---- reward and the hinge ---------------------------------------------------------------------------

def test_hinge_gives_exactly_minus_one_below_the_floor():
    # q 0.4 < floor 0.5: the hinge; the linear value would have been 0.4 - 1.0 * 0.25 = 0.15
    assert cell_reward(0.4, 0.25, 1.0, 0.5) == -1.0


def test_at_or_above_the_floor_the_reward_is_linear_in_quality_and_cost():
    # 0.75 - 2.0 * 0.25 = 0.25 (all exact in binary)
    assert cell_reward(0.75, 0.25, 2.0, 0.5) == 0.25
    # exactly at the floor is not below it: 0.5 - 0.0 = 0.5, not -1
    assert cell_reward(0.5, 0.0, 1.0, 0.5) == 0.5


def test_lambda_is_clipped_to_zero_to_five():
    assert cell_reward(1.0, 0.125, 9.0, 0.5) == 0.375  # lambda 5: 1 - 5 * 0.125
    assert cell_reward(1.0, 0.125, -3.0, 0.5) == 1.0  # lambda 0
    assert cell_reward(1.0, 0.125, 5.0, 0.5) == 0.375  # the cap itself is allowed


def test_nonfinite_reward_inputs_give_nan():
    assert math.isnan(cell_reward(math.nan, 0.1, 1.0, 0.5))
    assert math.isnan(cell_reward(0.9, math.inf, 1.0, 0.5))


# ---- summarising rows -------------------------------------------------------------------------------

def test_an_arm_that_refuses_everything_gets_reward_minus_one_and_is_degenerate():
    healthy_inc = summarize_arm("static", _rows(40, cost=0.25, quality=0.75), cost_scale=1.0)
    refuser = summarize_arm("router", _rows(40, refused=True), cost_scale=1.0)
    assert (refuser.n, refuser.refused, refuser.completed) == (40, 40, 40)
    assert refuser.mean_reward == -1.0
    assert refuser.mean_quality == 0.0
    assert healthy_inc.mean_reward == 0.5  # 0.75 - 1.0 * 0.25: no hinge for the healthy arm
    assert judge_arm(refuser, healthy_inc).label == DEGENERATE


def test_the_refuse_everything_verdict_names_its_reason_exactly():
    inc = summarize_arm("static", _rows(40, cost=0.0, quality=0.9))
    refuser = summarize_arm("router", _rows(40, refused=True))
    verdict = judge_arm(refuser, inc)
    assert verdict.label == "degenerate"
    assert verdict.reasons == ("refusal_above_incumbent",)


def test_summarize_counts_and_means_are_exact():
    rows = (_rows(3, cost=0.5, quality=0.75, escalated=True) + _rows(1, cost=0.5, quality=0.25, completed=False)
            + [{"completed": True, "refused": False, "escalated": False, "cost_usd": 0.0, "quality": None}])
    s = summarize_arm("a", rows, lam=1.0, cost_scale=1.0, quality_floor=0.1)
    assert (s.n, s.completed, s.refused, s.escalated) == (5, 4, 0, 3)
    assert s.cost_total == 2.0
    # 4 labelled rows; rewards 0.75-0.5 (x3) = 0.25 and 0.25-0.5 = -0.25 -> mean 0.125; quality mean 0.625
    assert s.mean_reward == 0.125
    assert s.mean_quality == 0.625


def test_no_labels_means_no_reward_mean():
    s = summarize_arm("a", [{"completed": True, "cost_usd": 0.1, "quality": None}] * 3)
    assert (s.mean_reward, s.mean_quality) == (None, None)


def test_a_row_without_cost_is_refused_by_name():
    with pytest.raises(DegeneracyInputError, match="arm 'a': row without a numeric cost_usd"):
        summarize_arm("a", [{"completed": True}])


def test_cost_scale_must_be_positive():
    with pytest.raises(DegeneracyInputError, match="cost_scale must be > 0, got 0"):
        summarize_arm("a", [], cost_scale=0)


def test_counts_out_of_range_are_refused():
    with pytest.raises(DegeneracyInputError, match="counts out of range"):
        _arm(n=10, refused=11)


# ---- the verdicts -----------------------------------------------------------------------------------

def test_a_cheaper_correct_arm_is_healthy():
    arm = _arm(completed=90, refused=5, escalated=10, cost=9.0, reward=0.6, quality=0.86)
    verdict = judge_arm(arm, INCUMBENT)
    assert (verdict.label, verdict.reasons) == (HEALTHY, ())


def test_before_burn_in_the_verdict_is_too_early():
    arm = _arm(n=10, completed=9, refused=1, escalated=1, cost=1.0)
    verdict = judge_arm(arm, INCUMBENT)
    assert (verdict.label, verdict.reasons) == (TOO_EARLY, ("burn_in",))


def test_burn_in_boundary_is_exactly_min_n():
    assert MIN_ARM_N == 30
    below = _arm(n=29, completed=26, refused=1, escalated=3, cost=3.0)
    at = _arm(n=30, completed=27, refused=1, escalated=3, cost=3.0)
    assert judge_arm(below, INCUMBENT).label == TOO_EARLY
    assert judge_arm(at, INCUMBENT).label == HEALTHY


def test_a_young_incumbent_cannot_be_compared_against():
    young = _arm("incumbent", n=29, completed=26, refused=1, escalated=3, cost=3.0)
    assert judge_arm(_arm(), young).label == TOO_EARLY


def test_completion_exactly_1pp_below_is_healthy_and_more_is_degenerate():
    # incumbent 90/100; 89/100 is exactly 1 pp lower, 88/100 is 2 pp lower
    at_edge = judge_arm(_arm(completed=89, refused=5, escalated=10, cost=12.5), INCUMBENT)
    beyond = judge_arm(_arm(completed=88, refused=5, escalated=10, cost=12.5), INCUMBENT)
    assert (at_edge.label, at_edge.reasons) == (HEALTHY, ())
    assert (beyond.label, beyond.reasons) == (DEGENERATE, ("completion_below_incumbent",))


def test_refusal_rate_exactly_1pp_above_is_healthy_and_more_is_degenerate():
    # incumbent refuses 5/100; 6/100 is exactly +1 pp, 7/100 is +2 pp. Same cost, so no cheap-by-refusing.
    at_edge = judge_arm(_arm(completed=90, refused=6, escalated=10, cost=12.5), INCUMBENT)
    beyond = judge_arm(_arm(completed=90, refused=7, escalated=10, cost=12.5), INCUMBENT)
    assert (at_edge.label, at_edge.reasons) == (HEALTHY, ())
    assert (beyond.label, beyond.reasons) == (DEGENERATE, ("refusal_above_incumbent",))


def test_one_pp_boundaries_are_exact_where_float_arithmetic_is_not():
    # Chosen with a float search: in binary floats 9/100 < 10/100 - 0.01 is True (0.09 < 0.09000000000000001)
    # and 7/100 > 6/100 + 0.01 is True (0.07 > 0.06999999999999999), so a float implementation would call
    # these exactly-1-pp differences a breach. Exact fractions do not.
    inc = _arm("incumbent", completed=10, refused=6, escalated=10, cost=12.5)
    completion_edge = judge_arm(_arm(completed=9, refused=6, escalated=10, cost=12.5), inc)
    refusal_edge = judge_arm(_arm(completed=10, refused=7, escalated=10, cost=12.5), inc)
    assert (completion_edge.label, completion_edge.reasons) == (HEALTHY, ())
    assert (refusal_edge.label, refusal_edge.reasons) == (HEALTHY, ())
    # positive twins: one more point of difference is a breach
    assert judge_arm(_arm(completed=8, refused=6, escalated=10, cost=12.5), inc).reasons == ("completion_below_incumbent",)
    assert judge_arm(_arm(completed=10, refused=8, escalated=10, cost=12.5), inc).reasons == ("refusal_above_incumbent",)


def test_cheap_by_refusing_is_degenerate_but_genuinely_cheaper_is_healthy():
    # Incumbent: 100 calls, 12.5 USD, 5 refused -> 0.1316 per served. The arm refuses 1 more (6, within
    # the 1 pp guard). Cost 12.375 for 100 calls is per served 12.375 / 94 = 0.13165 vs 12.5 / 95 = 0.13158:
    # NOT cheaper per answer, only cheaper because it served less -> cheap_by_refusing.
    cheap_by_refusing = judge_arm(_arm(completed=90, refused=6, escalated=10, cost=12.375), INCUMBENT)
    assert (cheap_by_refusing.label, cheap_by_refusing.reasons) == (DEGENERATE, ("cheap_by_refusing",))
    # Same refusals, but cost 11.0 = 0.11702 per served < 0.13158: a real saving.
    genuinely_cheaper = judge_arm(_arm(completed=90, refused=6, escalated=10, cost=11.0), INCUMBENT)
    assert (genuinely_cheaper.label, genuinely_cheaper.reasons) == (HEALTHY, ())


def test_escalation_drift_is_decided_at_two_standard_errors():
    # incumbent 10/100 escalated. Pooled z for arm k/100 (hand computed):
    #   k=20: p=30/200=0.15, se=sqrt(0.15*0.85*0.02)=0.050498, z=0.10/0.050498=1.980 -> within 2 SE
    #   k=21: p=31/200=0.155, se=sqrt(0.155*0.845*0.02)=0.051181, z=0.11/0.051181=2.149 -> drift
    inc = _arm("incumbent", completed=90, refused=5, escalated=10, cost=12.5)
    within = judge_arm(_arm(completed=90, refused=5, escalated=20, cost=12.5), inc)
    drift = judge_arm(_arm(completed=90, refused=5, escalated=21, cost=12.5), inc)
    assert (within.label, within.reasons) == (HEALTHY, ())
    assert (drift.label, drift.reasons) == (DEGENERATE, ("escalation_drift",))


def test_escalating_less_than_the_incumbent_is_not_drift():
    inc = _arm("incumbent", completed=90, refused=5, escalated=40, cost=12.5)
    assert judge_arm(_arm(completed=90, refused=5, escalated=5, cost=12.5), inc).label == HEALTHY


def test_reward_up_while_quality_down_is_degenerate():
    gamed = _arm(completed=90, refused=5, escalated=10, cost=12.5, reward=0.6, quality=0.80)
    verdict = judge_arm(gamed, INCUMBENT)
    assert (verdict.label, verdict.reasons) == (DEGENERATE, ("reward_up_quality_down",))


def test_reward_up_with_quality_within_1pp_or_reward_not_up_is_healthy():
    drop_half_pp = _arm(completed=90, refused=5, escalated=10, cost=12.5, reward=0.6, quality=0.845)
    reward_lower = _arm(completed=90, refused=5, escalated=10, cost=12.5, reward=0.4, quality=0.80)
    assert judge_arm(drop_half_pp, INCUMBENT).label == HEALTHY
    assert judge_arm(reward_lower, INCUMBENT).label == HEALTHY


def test_nonfinite_values_are_degenerate_even_before_burn_in():
    for field in ("cost", "reward", "quality"):
        bad = _arm(n=1, completed=1, **{field: math.nan})
        verdict = judge_arm(bad, INCUMBENT)
        assert (verdict.label, verdict.reasons) == (DEGENERATE, ("nonfinite",)), field
    assert judge_arm(_arm(n=1, completed=1, cost=math.inf), INCUMBENT).reasons == ("nonfinite",)
    # a NaN quality in a row survives summarising and is reported, not filtered out
    nan_arm = summarize_arm("router", _rows(40, quality=math.nan))
    assert judge_arm(nan_arm, INCUMBENT).reasons == ("nonfinite",)
    # positive twin: the same single-call arm with finite values is merely too early
    assert judge_arm(_arm(n=1, completed=1), INCUMBENT).label == TOO_EARLY


def test_a_nonfinite_incumbent_cannot_be_judged_against():
    bad_inc = _arm("incumbent", reward=math.nan)
    with pytest.raises(DegeneracyInputError, match="incumbent 'incumbent' has a non-finite value"):
        judge_arm(_arm(), bad_inc)


def test_several_failed_checks_are_all_reported_in_a_fixed_order():
    # completion 80/100 (-10 pp), refusal 20/100 (+15 pp), escalation 40/100 (z > 2), reward up quality down
    arm = _arm(completed=80, refused=20, escalated=40, cost=12.5, reward=0.9, quality=0.5)
    assert judge_arm(arm, INCUMBENT).reasons == (
        "completion_below_incumbent", "refusal_above_incumbent", "escalation_drift", "reward_up_quality_down")


# ---- Page-Hinkley -----------------------------------------------------------------------------------
# Stream: 100 samples of N(0.5, 0.03) then the mean shifts. Seeds are fixed, so the firing sample is exact.

def _first_alarm_after_shift(seed, shift, detector=None):
    rng = random.Random(seed)
    ph = detector or PageHinkley()
    for _ in range(100):
        assert ph.update(0.5 + rng.gauss(0.0, 0.03)) is False  # no alarm before the shift
    for i in range(1, 201):
        if ph.update(0.5 + shift + rng.gauss(0.0, 0.03)):
            return i
    return None


def test_page_hinkley_fires_within_25_samples_of_a_plus_0_1_shift():
    firing = [_first_alarm_after_shift(seed, 0.1) for seed in range(60)]
    assert None not in firing
    assert max(firing) == 20  # worst of 60 seeds, under the 25-sample requirement
    assert max(firing) <= 25


def test_page_hinkley_fires_at_the_pinned_sample_for_seed_0():
    assert _first_alarm_after_shift(0, 0.1) == 16


def test_page_hinkley_detects_a_downward_shift_too():
    assert _first_alarm_after_shift(0, -0.1) == 19


def test_page_hinkley_never_fires_on_5000_stationary_samples():
    rng = random.Random(7)
    ph = PageHinkley()
    fired = sum(ph.update(0.5 + rng.gauss(0.0, 0.03)) for _ in range(5000))
    assert fired == 0


def test_page_hinkley_resets_after_firing_so_a_second_shift_is_seen():
    ph = PageHinkley()
    assert _first_alarm_after_shift(0, 0.1, ph) == 16
    assert ph.n == 0  # reset on fire
    rng = random.Random(1)
    for _ in range(100):
        ph.update(0.6 + rng.gauss(0.0, 0.03))
    second = next(i for i in range(1, 200) if ph.update(0.7 + rng.gauss(0.0, 0.03)))
    assert second <= 25


def test_page_hinkley_rejects_nonfinite_samples_and_bad_parameters():
    with pytest.raises(DegeneracyInputError, match="PageHinkley sample is not finite: nan"):
        PageHinkley().update(math.nan)
    with pytest.raises(DegeneracyInputError, match="needs delta >= 0 and threshold > 0"):
        PageHinkley(threshold=0)
