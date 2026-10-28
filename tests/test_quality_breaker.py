"""Quality oracle (S12), second half: Wilson quality breaker, judge spend bucket, drop-oldest queue, pool_grader oracle mode.

Time is a fake clock, money is Decimal, the judge and the pool are recording stubs; nothing reaches a model or the network.

Mutation log (each applied to mcp/quality_validators.py or mcp/pool_grader.py; the named tests went red; reverted):
- ``n_downgraded < min_n`` -> ``< 29`` (n>=30 became n>=29): test_29_samples_never_trip_and_the_30th_does,
  test_should_trip_needs_thirty_on_both_arms
- strict ``>`` margin -> ``>=``: test_a_difference_of_exactly_the_margin_does_not_trip
- raw float subtraction instead of the 12-place Decimal compare: test_a_difference_of_exactly_the_margin_does_not_trip
- trip duration 24 h -> 12 h: test_a_trip_disables_the_downgrade_for_exactly_24_hours
- expiry ``now >= until`` -> ``now > until``: test_a_trip_disables_the_downgrade_for_exactly_24_hours
- the downgraded window not cleared at expiry: test_after_expiry_fresh_evidence_is_needed_to_trip_again
- ``enforce`` ignored (always blocks): test_observe_mode_counts_the_trip_but_never_blocks
- a judge verdict recorded while the gate fails: test_judge_results_are_ignored_while_the_gate_fails
- an excluded call scored as a fail: test_excluded_calls_are_counted_never_scored
- internal error returns True under enforce: test_an_internal_error_fails_closed_when_enforcing
- spend ``>`` cap -> ``>=``: test_the_bucket_may_spend_exactly_its_cap_and_not_a_cent_more
- in-flight holds ignored: test_in_flight_holds_count_against_the_cap
- ``_money`` accepts floats: test_money_is_never_a_float
- reader failure treated as zero spend: test_a_spend_reader_that_fails_closes_the_bucket
- eviction drops the NEWEST item: test_a_full_queue_drops_the_oldest_item_and_counts_it
- oracle mode off still touching the oracle structures / bucket: test_oracle_mode_is_off_by_default_and_inert
- bucket reserve removed from grade_one: test_the_judge_budget_stops_at_ten_judgments_at_049_each
- ``select_judge`` not consulted in ``_judge``: test_a_pool_whose_only_model_is_the_answerer_is_skipped_as_same_model
"""
from __future__ import annotations

import datetime as dt
import json
import queue
import sys
from decimal import Decimal
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parents[1] / "mcp"
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import llm_local  # noqa: E402
import model_pool as M  # noqa: E402
import pool_grader as G  # noqa: E402
import quality_validators as qv  # noqa: E402

DAY = 24 * 3600.0


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def _feed(b, arm, passes, total):
    for i in range(total):
        b.record(arm, i < passes)


def _tripped(b, passes_down=15, n_down=30, passes_base=17, n_base=30):
    _feed(b, qv.ARM_BASELINE, passes_base, n_base)
    _feed(b, qv.ARM_DOWNGRADED, passes_down, n_down)


# ------------------------------------------------------------------------------------------------- Wilson
class TestWilson:
    @pytest.mark.parametrize("k, n, want", [
        (15, 30, 0.331539), (17, 30, 0.391970), (19, 30, 0.455132), (26, 40, 0.495056), (30, 30, 0.886483),
        (28, 30, 0.786762), (0, 29, 0.0), (0, 0, 0.0)])
    def test_lower_bounds_match_the_hand_computed_table(self, k, n, want):
        assert abs(qv.wilson_lower(k, n) - want) < 5e-7

    def test_p_one_half_n_thirty_by_hand(self):
        # centre 0.5 + 3.8416/60 = 0.5640267; spread 1.96*sqrt(0.25/30 + 3.8416/3600) = 0.1900343; / (1 + 3.8416/30)
        assert abs(qv.wilson_lower(15, 30) - (0.5640267 - 0.1900343) / 1.1280533) < 5e-6

    def test_should_trip_needs_thirty_on_both_arms(self):
        assert qv.should_trip(0.0, 0.9, 30, 30) is True
        assert qv.should_trip(0.0, 0.9, 29, 30) is False
        assert qv.should_trip(0.0, 0.9, 30, 29) is False

    def test_a_difference_of_exactly_the_margin_does_not_trip(self):
        assert 0.65 - 0.60 > 0.05                                   # raw floats would trip here: why the compare is rounded
        assert qv.should_trip(0.60, 0.65, 30, 30) is False
        assert qv.should_trip(0.5999, 0.65, 30, 30) is True
        assert qv.should_trip(0.60, 0.6501, 30, 30) is True

    def test_the_hand_computed_pairs(self):
        # baseline 17/30 (LB .39197) vs downgraded 15/30 (LB .331539): 0.0604 below -> trips
        assert qv.should_trip(qv.wilson_lower(15, 30), qv.wilson_lower(17, 30), 30, 30) is True
        # baseline 26/40 (LB .495056) vs downgraded 19/30 (LB .455132): 0.0399 below -> does not
        assert qv.should_trip(qv.wilson_lower(19, 30), qv.wilson_lower(26, 40), 30, 40) is False


class TestBreaker:
    def test_29_samples_never_trip_and_the_30th_does(self):
        b = qv.QualityBreaker(enforce=True, clock=Clock())
        _feed(b, qv.ARM_BASELINE, 17, 30)
        _feed(b, qv.ARM_DOWNGRADED, 0, 29)
        assert (b.downgrade_allowed(), b.stats()["trips"], b.stats()["evaluable"]) == (True, 0, False)
        b.record(qv.ARM_DOWNGRADED, False)
        assert (b.downgrade_allowed(), b.stats()["trips"], b.stats()["evaluable"]) == (False, 1, True)

    def test_a_baseline_of_29_does_not_trip_either(self):
        b = qv.QualityBreaker(enforce=True, clock=Clock())
        _feed(b, qv.ARM_BASELINE, 17, 29)
        _feed(b, qv.ARM_DOWNGRADED, 0, 40)
        assert (b.downgrade_allowed(), b.stats()["trips"]) == (True, 0)

    def test_thirty_samples_with_a_lower_bound_006_below_baseline_trip_for_24_hours(self):
        clock = Clock(5000.0)
        b = qv.QualityBreaker(enforce=True, clock=clock)
        _tripped(b)
        s = b.stats()
        assert (s["tripped"], s["trips"], s["disabled_until"], b.downgrade_allowed()) == (True, 1, 5000.0 + DAY, False)
        assert abs(s["lb_baseline"] - s["lb_downgraded"] - 0.060432) < 1e-5

    def test_a_lower_bound_004_below_baseline_does_not_trip(self):
        b = qv.QualityBreaker(enforce=True, clock=Clock())
        _feed(b, qv.ARM_BASELINE, 26, 40)
        _feed(b, qv.ARM_DOWNGRADED, 19, 30)
        s = b.stats()
        assert abs(s["lb_baseline"] - s["lb_downgraded"] - 0.039924) < 1e-5
        assert (s["tripped"], s["trips"], s["evaluable"], b.downgrade_allowed()) == (False, 0, True, True)

    def test_a_trip_disables_the_downgrade_for_exactly_24_hours(self):
        clock = Clock(1000.0)
        b = qv.QualityBreaker(enforce=True, clock=clock)
        _tripped(b)
        clock.now = 1000.0 + DAY - 1
        assert b.downgrade_allowed() is False
        clock.now = 1000.0 + DAY
        assert (b.downgrade_allowed(), b.stats()["tripped"], b.stats()["resets"]) == (True, False, 1)

    def test_after_expiry_fresh_evidence_is_needed_to_trip_again(self):
        clock = Clock(1000.0)
        b = qv.QualityBreaker(enforce=True, clock=clock, window_s=10 * DAY)
        _tripped(b)
        clock.now = 1000.0 + DAY
        assert b.downgrade_allowed() is True
        s = b.stats()
        assert (s["n_downgraded"], s["n_baseline"]) == (0, 30)       # the old evidence is gone, the baseline stays
        _feed(b, qv.ARM_DOWNGRADED, 0, 29)
        assert (b.downgrade_allowed(), b.stats()["trips"]) == (True, 1)
        b.record(qv.ARM_DOWNGRADED, False)
        assert (b.downgrade_allowed(), b.stats()["trips"]) == (False, 2)

    def test_observe_mode_counts_the_trip_but_never_blocks(self):
        b = qv.QualityBreaker(clock=Clock())
        assert b.enforce is False
        _tripped(b)
        s = b.stats()
        assert (s["tripped"], s["trips"], s["blocked_checks"], b.downgrade_allowed()) == (True, 1, 0, True)

    def test_enforce_counts_each_blocked_check(self):
        b = qv.QualityBreaker(enforce=True, clock=Clock())
        _tripped(b)
        assert [b.downgrade_allowed() for _ in range(3)] == [False, False, False]
        assert b.stats()["blocked_checks"] == 3

    def test_an_all_passing_downgrade_never_trips(self):
        b = qv.QualityBreaker(enforce=True, clock=Clock())
        _tripped(b, passes_down=30, passes_base=30)
        assert (b.downgrade_allowed(), b.stats()["trips"]) == (True, 0)

    def test_the_window_is_rolling(self):
        clock = Clock(0.0)
        b = qv.QualityBreaker(clock=clock, window_s=100.0)
        _feed(b, qv.ARM_BASELINE, 30, 30)
        clock.now = 99.0
        assert b.stats()["n_baseline"] == 30
        clock.now = 100.0
        assert b.stats()["n_baseline"] == 0

    def test_the_sample_buffer_is_bounded(self):
        b = qv.QualityBreaker(clock=Clock(), max_samples=50)
        _feed(b, qv.ARM_BASELINE, 60, 60)
        assert b.stats()["n_baseline"] == 50

    def test_excluded_calls_are_counted_never_scored(self):
        b = qv.QualityBreaker(enforce=True, clock=Clock())
        _feed(b, qv.ARM_BASELINE, 17, 30)
        for _ in range(40):
            b.record(qv.ARM_DOWNGRADED, None)
        s = b.stats()
        assert (s["excluded"], s["n_downgraded"], s["trips"], b.downgrade_allowed()) == (40, 0, 0, True)

    def test_validator_results_feed_the_breaker(self):
        b = qv.QualityBreaker(clock=Clock())
        b.record_validation(qv.ARM_DOWNGRADED, qv.validate(qv.CallOutcome(text="ok", finish_reason="stop")))
        b.record_validation(qv.ARM_DOWNGRADED, qv.validate(qv.CallOutcome(text="", finish_reason="stop")))
        b.record_validation(qv.ARM_DOWNGRADED, qv.validate(qv.CallOutcome(text="", finish_reason="stop", stream=True)))
        s = b.stats()
        assert (s["k_downgraded"], s["n_downgraded"], s["excluded"]) == (1, 2, 1)

    def test_judge_results_are_ignored_while_the_gate_fails(self):
        b = qv.QualityBreaker(clock=Clock())
        b.record_judge(qv.ARM_BASELINE, True, gate_passed=False)
        s = b.stats()
        assert (s["n_baseline"], s["judge_ignored_gate_not_passed"], s["recorded"]) == (0, 1, 0)
        b.record_judge(qv.ARM_BASELINE, True, gate_passed=True)
        s = b.stats()
        assert (s["n_baseline"], s["k_baseline"], s["judge_ignored_gate_not_passed"], s["recorded"]) == (1, 1, 1, 1)

    def test_bad_input_is_refused(self):
        b = qv.QualityBreaker(clock=Clock())
        with pytest.raises(ValueError, match="unknown arm 'canary'"):
            b.record("canary", True)
        with pytest.raises(ValueError, match="passed must be True, False or None"):
            b.record(qv.ARM_BASELINE, 1)
        assert b.stats()["recorded"] == 0

    def test_an_internal_error_fails_closed_when_enforcing(self):
        class BrokenClock:
            broken = False

            def __call__(self):
                if self.broken:
                    raise OSError("clock gone")
                return 1000.0
        clock = BrokenClock()
        enforcing = qv.QualityBreaker(enforce=True, clock=clock)
        observing = qv.QualityBreaker(enforce=False, clock=clock)
        assert (enforcing.downgrade_allowed(), observing.downgrade_allowed()) == (True, True)
        clock.broken = True
        assert (enforcing.downgrade_allowed(), observing.downgrade_allowed()) == (False, True)
        assert (enforcing.counters["errors"], observing.counters["errors"]) == (1, 1)

    def test_a_fresh_breaker_reports_every_counter(self):
        assert qv.QualityBreaker(clock=Clock(7.0)).stats() == {
            "enforce": False, "tripped": False, "disabled_until": None, "n_downgraded": 0, "k_downgraded": 0,
            "n_baseline": 0, "k_baseline": 0, "lb_downgraded": 0.0, "lb_baseline": 0.0, "evaluable": False,
            "recorded": 0, "excluded": 0, "trips": 0, "resets": 0, "judge_ignored_gate_not_passed": 0,
            "errors": 0, "blocked_checks": 0}

    @pytest.mark.parametrize("value, want", [("enforce", True), (" ENFORCE ", True), ("observe", False), ("1", False), ("", False)])
    def test_only_the_word_enforce_makes_it_act(self, monkeypatch, value, want):
        monkeypatch.setenv(qv.BREAKER_ENV, value)
        assert qv.QualityBreaker.from_env(clock=Clock()).enforce is want

    def test_unset_means_observe(self, monkeypatch):
        monkeypatch.delenv(qv.BREAKER_ENV, raising=False)
        assert qv.QualityBreaker.from_env(clock=Clock()).enforce is False


# ------------------------------------------------------------------------------------------------- spend bucket
class TestSpendBucket:
    def test_the_judge_budget_stops_at_ten_judgments_at_049_each(self):
        bucket = qv.JudgeSpendBucket(today=lambda: dt.date(2026, 10, 18))
        done = 0
        with pytest.raises(qv.OracleStarved, match="judge budget 0.50 exhausted: spent 0.490, in flight 0"):
            while True:
                held = bucket.reserve(Decimal("0.049"))
                bucket.settle(held, Decimal("0.049"))
                done += 1
        assert done == 10
        assert bucket.spent() == Decimal("0.490")
        assert bucket.stats()["starved"] == 1
        assert qv.OracleStarved.reason == "oracle_starved"

    def test_the_bucket_may_spend_exactly_its_cap_and_not_a_cent_more(self):
        bucket = qv.JudgeSpendBucket(estimate=Decimal("0.05"), today=lambda: dt.date(2026, 10, 18))
        for _ in range(10):
            bucket.settle(bucket.reserve(), None)
        assert bucket.spent() == Decimal("0.50")
        with pytest.raises(qv.OracleStarved, match="exhausted"):
            bucket.reserve()

    def test_in_flight_holds_count_against_the_cap(self):
        bucket = qv.JudgeSpendBucket(estimate=Decimal("0.05"), today=lambda: dt.date(2026, 10, 18))
        held = [bucket.reserve() for _ in range(10)]
        with pytest.raises(qv.OracleStarved, match="in flight 0.50"):
            bucket.reserve()
        bucket.release(held.pop())
        assert bucket.reserve() == Decimal("0.05")
        assert bucket.spent() == Decimal(0)

    def test_a_measured_cost_replaces_the_hold(self):
        bucket = qv.JudgeSpendBucket(today=lambda: dt.date(2026, 10, 18))
        bucket.settle(bucket.reserve(), Decimal("0.010"))
        assert (bucket.spent(), bucket.stats()["in_flight"]) == (Decimal("0.010"), Decimal(0))

    def test_the_ledger_rolls_over_with_the_local_day(self):
        day = [dt.date(2026, 10, 18)]
        bucket = qv.JudgeSpendBucket(today=lambda: day[0])
        bucket.settle(bucket.reserve(Decimal("0.49")), None)
        assert bucket.spent() == Decimal("0.49")
        day[0] = dt.date(2026, 10, 19)
        assert bucket.spent() == Decimal(0)
        assert bucket.reserve() == Decimal("0.049")

    def test_money_is_never_a_float(self):
        bucket = qv.JudgeSpendBucket()
        with pytest.raises(TypeError, match="cost must be a Decimal"):
            bucket.reserve(0.049)
        with pytest.raises(TypeError, match="cap must be a Decimal"):
            qv.JudgeSpendBucket(0.5)
        with pytest.raises(ValueError, match="non-negative"):
            bucket.reserve(Decimal("-1"))
        assert bucket.reserve(Decimal("0.049")) == Decimal("0.049")

    def test_an_injected_reader_is_the_spend_truth(self):
        spent = [Decimal("0.45")]
        reads = []
        bucket = qv.JudgeSpendBucket(spend_reader=lambda: reads.append(1) or spent[0])
        held = bucket.reserve()                                       # 0.45 + 0.049 = 0.499 <= 0.50
        bucket.settle(held, Decimal("0.049"))
        assert bucket.spent() == Decimal("0.45")                       # settle does not double count when a reader owns the truth
        spent[0] = Decimal("0.46")
        with pytest.raises(qv.OracleStarved, match="spent 0.46"):
            bucket.reserve()
        assert len(reads) == 3                                         # reserve, spent(), reserve: settle never reads

    def test_a_spend_reader_that_fails_closes_the_bucket(self):
        def broken():
            raise OSError("ledger unreadable")
        bucket = qv.JudgeSpendBucket(spend_reader=broken)
        with pytest.raises(qv.OracleStarved, match="spend reader unavailable"):
            bucket.reserve()
        assert (bucket.stats()["reader_errors"], bucket.stats()["starved"]) == (1, 1)
        for bad in (0.1, None, Decimal("-1"), Decimal("NaN")):
            b2 = qv.JudgeSpendBucket(spend_reader=lambda v=bad: v)
            with pytest.raises(qv.OracleStarved, match="spend reader unavailable"):
                b2.reserve()
            assert b2.stats()["reader_errors"] == 1


# ------------------------------------------------------------------------------------------------- queue
class TestDropOldestQueue:
    def test_a_full_queue_drops_the_oldest_item_and_counts_it(self):
        q = qv.DropOldestQueue(3)
        assert [q.put(i) for i in (1, 2, 3)] == [None, None, None]
        assert q.put(4) == 1
        assert (q.qsize(), q.dropped) == (3, 1)
        assert [q.get(timeout=0), q.get(timeout=0), q.get(timeout=0)] == [2, 3, 4]

    def test_an_empty_queue_times_out(self):
        q = qv.DropOldestQueue(2)
        with pytest.raises(queue.Empty):
            q.get(timeout=0)
        q.put("x")
        assert q.get(timeout=0) == "x"

    def test_the_default_bound_is_500_and_a_zero_bound_is_refused(self):
        assert qv.DropOldestQueue().maxsize == 500 == qv.JUDGE_QUEUE_MAX
        with pytest.raises(ValueError, match="maxsize must be at least 1"):
            qv.DropOldestQueue(0)


# ------------------------------------------------------------------------------------------------- pool_grader oracle mode
def _grades(tmp_path):
    path = tmp_path / "instrumentation" / M.GRADES_LOG_NAME
    return [json.loads(line) for line in path.read_text().splitlines()]


def _item(**kw):
    base = {"task": "classify", "prompt": "Labels: a, b\nText: t", "answer": "a", "model": "m1", "decision_id": "d1"}
    return {**base, **kw}


class Judge:
    """Recording judge: returns a canned result and keeps every call."""

    def __init__(self, **result):
        self.result = {"text": '{"correct": true}', "ok": True, "model": "judge-model", **result}
        self.calls = []

    def __call__(self, task, prompt, answer, answerer):
        self.calls.append((task, answerer))
        return dict(self.result)


@pytest.fixture
def pool(monkeypatch, tmp_path):
    M.clear_cache()
    monkeypatch.setenv(M.SHADOW_ENV, "1")
    monkeypatch.setenv(G.GRADE_ENV, "1")
    monkeypatch.delenv(G.ROLE_ENV, raising=False)
    monkeypatch.delenv(G.ORACLE_ENV, raising=False)
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path / "memory-sessions"))
    M._LAST_DECISION.set(None)
    monkeypatch.setattr(G, "_queue", queue.Queue(maxsize=2))
    monkeypatch.setattr(G, "_ensure_worker", lambda: None)
    monkeypatch.setattr(G, "_oracle_bucket", None)
    monkeypatch.setattr(G, "_oracle_queue", None)
    yield tmp_path
    M.clear_cache()


class TestOracleMode:
    @pytest.mark.parametrize("value, want", [("1", True), ("true", True), ("YES", True), ("on", True), ("", False),
                                             ("0", False), ("off", False), ("no", False), ("false", False)])
    def test_only_a_yes_turns_it_on(self, pool, monkeypatch, value, want):
        monkeypatch.setenv(G.ORACLE_ENV, value)
        assert G.oracle_enabled() is want

    def test_oracle_mode_is_off_by_default_and_inert(self, pool, monkeypatch):
        touched = []
        monkeypatch.setattr(G, "oracle_bucket", lambda: touched.append("bucket"))
        monkeypatch.setattr(G, "oracle_queue", lambda: touched.append("queue"))
        assert G.oracle_enabled() is False
        judge = Judge(cost_usd="9.99")
        assert [G.grade_one(_item(decision_id=f"d{i}"), judge) for i in range(30)] == [True] * 30   # no budget applies
        subs = [G.maybe_submit("classify", "p", "a", "m1", f"s{i}") for i in range(3)]
        assert subs == [True, True, False]                            # the old 2-slot queue and its queue_full skip
        assert touched == []
        assert [(r["decision_id"], r["skipped"]) for r in _grades(pool) if "skipped" in r] == [("s2", "queue_full")]

    def test_the_judge_budget_stops_at_ten_judgments_at_049_each_in_the_grader(self, pool, monkeypatch):
        monkeypatch.setenv(G.ORACLE_ENV, "1")
        judge = Judge(cost_usd="0.049")
        verdicts = [G.grade_one(_item(decision_id=f"d{i}"), judge) for i in range(11)]
        assert verdicts == [True] * 10 + [None]
        assert len(judge.calls) == 10                                  # the 11th never reached the judge
        rows = _grades(pool)
        assert len(rows) == 11
        assert (rows[10]["decision_id"], rows[10]["skipped"], rows[10]["model"], rows[10]["task"]) == (
            "d10", "oracle_starved", "m1", "classify")
        assert "correct" not in rows[10]
        assert G.oracle_bucket().spent() == Decimal("0.490")
        assert G.oracle_stats()["bucket"]["starved"] == 1

    def test_a_result_with_no_cost_is_charged_the_per_judgment_estimate(self, pool, monkeypatch):
        monkeypatch.setenv(G.ORACLE_ENV, "1")
        judge = Judge()
        assert [G.grade_one(_item(decision_id=f"d{i}"), judge) for i in range(11)] == [True] * 10 + [None]
        assert G.oracle_bucket().spent() == Decimal("0.490")

    def test_a_free_judgment_never_starves_the_bucket(self, pool, monkeypatch):
        monkeypatch.setenv(G.ORACLE_ENV, "1")
        judge = Judge(cost_usd="0")
        assert [G.grade_one(_item(decision_id=f"d{i}"), judge) for i in range(25)] == [True] * 25
        assert G.oracle_bucket().spent() == Decimal(0)

    def test_a_garbage_cost_falls_back_to_the_estimate(self, pool, monkeypatch):
        monkeypatch.setenv(G.ORACLE_ENV, "1")
        for i, bad in enumerate(("abc", -1, True, float("nan"))):
            assert G.grade_one(_item(decision_id=f"d{i}"), Judge(cost_usd=bad)) is True
        assert G.oracle_bucket().spent() == Decimal("0.196")

    def test_a_failed_judgment_releases_its_hold_and_charges_nothing(self, pool, monkeypatch):
        monkeypatch.setenv(G.ORACLE_ENV, "1")
        failing = Judge(ok=False, why="boom")
        assert [G.grade_one(_item(decision_id=f"f{i}"), failing) for i in range(20)] == [None] * 20
        s = G.oracle_stats()["bucket"]
        assert (G.oracle_bucket().spent(), s["in_flight"], s["released"], s["starved"]) == (Decimal(0), Decimal(0), 20, 0)
        assert {r["skipped"] for r in _grades(pool)} == {"judge_failed"}
        assert G.grade_one(_item(decision_id="ok"), Judge()) is True

    def test_a_full_queue_drops_the_oldest_and_says_so(self, pool, monkeypatch):
        monkeypatch.setenv(G.ORACLE_ENV, "1")
        assert all(G.maybe_submit("classify", "p", "a", "m1", f"d{i}") for i in range(501))
        stats = G.oracle_stats()
        assert (stats["queue_len"], stats["queue_dropped"], G._queue.qsize()) == (500, 1, 0)
        rows = _grades(pool)
        assert [(r["decision_id"], r["skipped"]) for r in rows] == [("d0", "queue_dropped")]
        assert G.oracle_queue().get(timeout=0)["decision_id"] == "d1"

    def _single_model_pool(self, monkeypatch):
        monkeypatch.setattr(M, "entries", lambda: [M.PoolEntry("m1", ("verify",), rank=1.0)])
        monkeypatch.setattr(M, "inventory", lambda base_url=None: {"m1": 10 ** 9})
        monkeypatch.setattr(M, "resident_models", lambda base_url=None: {"m1"})

    def test_a_pool_whose_only_model_is_the_answerer_is_skipped_as_same_model(self, pool, monkeypatch):
        self._single_model_pool(monkeypatch)
        generated = []
        monkeypatch.setattr(llm_local, "generate", lambda prompt, **k: generated.append(k) or {"ok": True, "text": "{}"})
        monkeypatch.setenv(G.ORACLE_ENV, "1")
        assert G.grade_one(_item(model="m1")) is None
        row = _grades(pool)[0]
        assert (row["skipped"], row["model"], "correct" in row) == ("same_model", "m1", False)
        assert generated == []
        assert G.oracle_stats()["bucket"]["released"] == 1

    def test_with_oracle_mode_off_the_same_pool_keeps_its_old_reason(self, pool, monkeypatch):
        self._single_model_pool(monkeypatch)
        generated = []
        monkeypatch.setattr(llm_local, "generate", lambda prompt, **k: generated.append(k) or {"ok": True, "text": "{}"})
        assert G.grade_one(_item(model="m1")) is None
        assert _grades(pool)[0]["skipped"] == "no_other_model"
        assert generated == []

    def test_a_pool_with_a_second_model_is_judged_by_it(self, pool, monkeypatch):
        monkeypatch.setattr(M, "entries", lambda: [M.PoolEntry("m1", ("verify",), rank=1.0),
                                                   M.PoolEntry("m2", ("verify",), rank=0.5)])
        monkeypatch.setattr(M, "inventory", lambda base_url=None: {"m1": 10 ** 9, "m2": 10 ** 9})
        monkeypatch.setattr(M, "resident_models", lambda base_url=None: {"m1", "m2"})
        generated = []
        monkeypatch.setattr(llm_local, "generate", lambda prompt, **k: generated.append(k) or
                            {"ok": True, "text": '{"correct": true}', "model": k["model"]})
        monkeypatch.setenv(G.ORACLE_ENV, "1")
        assert G.grade_one(_item(model="m1")) is True
        assert [(g["model"], g["task"]) for g in generated] == [("m2", "judge")]
        assert _grades(pool)[0]["correct"] is True
