"""RetryLatch: exact backoff times on an injected clock, permanent vs transient, state for health."""
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retry_latch import RetryLatch  # noqa: E402


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


def _latch(clock, wall=None, **kw):
    kw.setdefault("base_s", 30.0)
    kw.setdefault("cap_s", 600.0)
    return RetryLatch("thing", clock=clock, wall=wall or (lambda: 1_700_000_000.0), **kw)


def test_a_fresh_latch_is_ok_and_allows_an_attempt():
    latch = _latch(Clock())
    assert latch.allow() is True
    assert latch.state() == {
        "name": "thing", "ok": True, "permanent": False, "last_error": "", "since": "",
        "attempts": 0, "failures_total": 0, "next_retry_in_s": None,
    }


def test_a_transient_failure_refuses_until_exactly_the_backoff_then_allows():
    clock = Clock(100.0)
    latch = _latch(clock)
    assert latch.failure(OSError("not enough memory")) is False
    assert latch.allow() is False
    clock.t = 129.999
    assert latch.allow() is False
    clock.t = 130.0
    assert latch.allow() is True          # positive twin: the retry IS allowed at the backoff


def test_state_reports_the_countdown_and_the_error():
    clock = Clock(100.0)
    latch = _latch(clock)
    latch.failure(OSError("not enough memory"))
    clock.t = 120.0
    assert latch.state() == {
        "name": "thing", "ok": False, "permanent": False, "last_error": "OSError: not enough memory",
        "since": "2023-11-14T22:13:20+00:00", "attempts": 1, "failures_total": 1, "next_retry_in_s": 10.0,
    }


def test_backoff_doubles_per_consecutive_failure_and_is_capped():
    clock = Clock(0.0)
    latch = _latch(clock, base_s=30.0, cap_s=100.0)
    waits = []
    for _ in range(5):
        latch.failure(OSError("x"))
        waits.append(latch.state()["next_retry_in_s"])
        clock.t += waits[-1]
        assert latch.allow() is True
    assert waits == [30.0, 60.0, 100.0, 100.0, 100.0]


def test_success_clears_the_error_the_attempts_and_the_backoff():
    clock = Clock(100.0)
    latch = _latch(clock)
    latch.failure(OSError("boom"))
    latch.failure(OSError("boom again"))
    latch.success()
    assert latch.state() == {
        "name": "thing", "ok": True, "permanent": False, "last_error": "", "since": "",
        "attempts": 0, "failures_total": 2, "next_retry_in_s": None,
    }
    assert latch.allow() is True
    latch.failure(OSError("later"))          # the backoff restarts from the base, not from where it was
    assert latch.state()["next_retry_in_s"] == 30.0


def test_since_is_the_first_failure_of_the_run_not_the_latest():
    clock = Clock(0.0)
    walls = iter([1_700_000_000.0, 1_700_000_999.0])
    latch = _latch(clock, wall=lambda: next(walls))
    latch.failure(OSError("a"))
    clock.t = 30.0
    latch.failure(OSError("b"))
    assert latch.state()["since"] == "2023-11-14T22:13:20+00:00"
    assert latch.state()["last_error"] == "OSError: b"


def test_an_import_error_is_permanent_and_never_allows_again():
    clock = Clock(100.0)
    latch = _latch(clock)
    assert latch.failure(ImportError("No module named 'fastembed'")) is True
    clock.t = 1e9
    assert latch.allow() is False
    st = latch.state()
    assert (st["ok"], st["permanent"], st["next_retry_in_s"]) == (False, True, None)
    assert st["last_error"] == "ImportError: No module named 'fastembed'"
    assert latch.permanent is True


def test_the_permanent_override_wins_in_both_directions():
    clock = Clock(0.0)
    explicit = _latch(clock)
    assert explicit.failure(RuntimeError("dimension mismatch"), permanent=True) is True
    clock.t = 1e9
    assert explicit.allow() is False
    clock.t = 0.0
    transient_import = _latch(clock)
    assert transient_import.failure(ImportError("DLL load failed"), permanent=False) is False
    clock.t = 30.0
    assert transient_import.allow() is True


def test_reset_forgets_the_failure():
    latch = _latch(Clock())
    latch.failure(ImportError("gone"))
    assert latch.allow() is False
    latch.reset()
    assert latch.allow() is True
    assert latch.state()["ok"] is True


def test_a_failure_logs_at_warning_with_the_error_and_the_retry_delay(caplog):
    latch = _latch(Clock())
    with caplog.at_level(logging.DEBUG, logger="loci-mcp"):
        latch.failure(OSError("not enough memory"))
    rec = [r for r in caplog.records if r.name == "loci-mcp"]
    assert [(r.levelno, r.getMessage()) for r in rec] == [
        (logging.WARNING, "thing failed (attempt 1), retrying in 30s: OSError: not enough memory")]


def test_a_permanent_failure_says_so_at_warning(caplog):
    latch = _latch(Clock())
    with caplog.at_level(logging.DEBUG, logger="loci-mcp"):
        latch.failure(ImportError("no torch"))
    assert [(r.levelno, r.getMessage()) for r in caplog.records if r.name == "loci-mcp"] == [
        (logging.WARNING, "thing unavailable permanently (attempt 1): ImportError: no torch")]


def test_warnings_are_rate_limited_and_report_what_was_suppressed(caplog):
    clock = Clock(0.0)
    latch = _latch(clock, log_every_s=60.0)
    with caplog.at_level(logging.WARNING, logger="loci-mcp"):
        assert latch.warn("k", "hello %s", "a") is True
        clock.t = 59.9
        assert latch.warn("k", "hello %s", "b") is False
        assert latch.warn("k", "hello %s", "c") is False
        assert latch.warn("other", "other key") is True          # keys are independent
        clock.t = 60.0
        assert latch.warn("k", "hello %s", "d") is True           # positive twin: emitted at the interval
    assert [r.getMessage() for r in caplog.records] == [
        "hello a", "other key", "hello d (+2 similar suppressed)"]


def test_recovery_is_logged(caplog):
    latch = _latch(Clock())
    latch.failure(OSError("x"))
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="loci-mcp"):
        latch.success()
        latch.success()                      # a success with nothing outstanding says nothing
    assert [r.getMessage() for r in caplog.records] == ["thing recovered after 1 failed attempt(s)"]


def test_cap_never_drops_below_base_and_zero_failures_means_no_delay():
    latch = _latch(Clock(), base_s=30.0, cap_s=5.0)
    assert latch.delay_after(1) == 30.0
    assert latch.delay_after(0) == 0.0
