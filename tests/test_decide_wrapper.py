"""mcp/decide.py wrapper limits: 2 in flight, a 5 s call budget, a breaker, and counters that report every fallback.

All time is a fake clock (nothing sleeps) and every interleave is a threading.Event/Barrier. The model is a recording
stub passed as ``post``; nothing reaches Ollama, the pool or the network.

Mutation log (each applied to mcp/decide.py; the named tests went red; then reverted):
- TIMEOUT_S = 60.0 (the old default): test_the_default_budget_is_5_seconds_and_each_request_gets_what_is_left,
  test_a_call_that_overruns_the_budget_is_degraded_and_counted, test_decide_accepts_an_explicit_budget_and_the_environment_sets_the_default
- the late-answer check (``lim.clock() > deadline``) removed: test_a_call_that_overruns_the_budget_is_degraded_and_counted
- late-answer check ``>=`` instead of ``>``: test_an_answer_at_exactly_the_budget_is_in_time
- ``remaining <= 0`` made ``< 0``: test_a_spent_budget_skips_the_remaining_questions
- the semaphore acquire made blocking (a third call waits): test_a_third_concurrent_call_gets_the_rules_fallback_at_once,
  test_the_in_flight_cap_is_configurable (both fail after their 10 s guard; the mutant also deadlocks its workers
  because it waits while holding the Limits lock, so the helper threads are daemons)
- the slot never released in finish(): test_a_slot_is_released_when_a_call_degrades
- breaker opens at a threshold of 4: test_three_consecutive_degraded_calls_open_the_breaker
- success does not reset the consecutive count: test_a_good_call_resets_the_consecutive_count
- cooldown ``>=`` made ``>``: test_the_breaker_half_opens_after_exactly_the_cooldown
- the probe is not limited to one: test_only_one_probe_is_let_through_while_half_open
- a degraded probe does not re-open: test_a_degraded_probe_reopens_the_breaker_for_a_fresh_cooldown
- any good call (not only the probe) closes a tripped breaker: test_a_call_already_in_flight_when_the_breaker_opened_does_not_close_it
- an invalid request counted toward the breaker: test_a_malformed_request_is_counted_but_never_trips_or_resets_the_breaker
- a ReadTimeout not recognised as a timeout: test_a_timeout_exception_from_the_server_is_counted_as_a_timeout
"""
from __future__ import annotations

import json
import math
import sys
import threading
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parents[1] / "mcp"
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import decide as D  # noqa: E402

STATE = "Charged twice, refund me"
Q = {"q": {"type": "noul", "instructions": "Is it a refund request?"}}
TWO_Q = {"first": {"type": "noul", "instructions": "one?"}, "second": {"type": "noul", "instructions": "two?"}}
P_YES = 1 / (1 + math.exp(-3.95))        # letters A=-4.0 (no), B=-0.05 (yes): softmax over the two
GUARD = 10                               # seconds a blocked test thread waits before the test fails instead of hanging


def _resp(letters):
    top = [{"token": k, "logprob": v} for k, v in letters.items()]
    return {"message": {"content": ""}, "logprobs": [{"token": " ", "logprob": -3.0, "top_logprobs": top}]}


YES = _resp({"A": -4.0, "B": -0.05})
NO_LETTERS = {"message": {"content": ""}, "logprobs": [{"token": " ", "logprob": -3.0, "top_logprobs": []}]}


class Clock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Server:
    """Recording ``post``: logs (url, timeout), optionally advances the clock, answers from a queue then ``default``."""

    def __init__(self, clock=None, take=0.0, default=YES, script=()):
        self.clock, self.take, self.default, self.script = clock, take, default, list(script)
        self.calls = []

    def __call__(self, url, body, timeout):
        self.calls.append((url, timeout))
        if self.clock is not None:
            self.clock.advance(self.take)
        out = self.script.pop(0) if self.script else self.default
        if isinstance(out, Exception):
            raise out
        return out


def _call(limits, server, questions=Q, **kw):
    return D.decide(STATE, questions, model="m:1", base_url="http://ollama", post=server, limits=limits, **kw)


def _degrade(limits, server_clock=None):
    """One degraded call: the server answers without any answer letter."""
    return _call(limits, Server(server_clock, default=NO_LETTERS))


def _fresh(**kw):
    clock = Clock()
    return D.Limits(clock=clock, **kw), clock


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    for name in (D.MODEL_ENV, D.MAX_INFLIGHT_ENV, D.TIMEOUT_ENV, D.BREAKER_THRESHOLD_ENV, D.BREAKER_COOLDOWN_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(D, "_LIMITS", D.Limits())
    monkeypatch.setattr(D, "_record", lambda *a, **k: None)         # the pool's outcome log is not under test
    monkeypatch.setattr(D, "_placement", lambda model: {})


# ------------------------------------------------------------------------------------------ the 5 s budget

def test_the_default_budget_is_5_seconds_and_each_request_gets_what_is_left():
    limits, clock = _fresh()
    server = Server(clock, take=2.0)
    res = _call(limits, server, TWO_Q)
    assert [t for _, t in server.calls] == [5.0, 3.0]
    assert res["ok"] is True and res["degraded"] is False and res["fallback"] is None
    assert res["answers"]["first"]["noul"] == pytest.approx(P_YES, rel=1e-12)


def test_a_call_that_overruns_the_budget_is_degraded_and_counted():
    limits, clock = _fresh()
    res = _call(limits, Server(clock, take=6.0))
    assert res["ok"] is False and res["degraded"] is True and res["fallback"] is None
    assert res["answers"] == {"q": {"error": "answer arrived after the 5s call budget"}}
    assert res["error"] == "q: answer arrived after the 5s call budget"
    counts = limits.stats()
    assert (counts["timeouts"], counts["degraded"], counts["ok"]) == (1, 1, 0)


def test_an_answer_at_exactly_the_budget_is_in_time():
    limits, clock = _fresh()
    res = _call(limits, Server(clock, take=5.0))
    assert res["ok"] is True and res["answers"]["q"]["noul"] == pytest.approx(P_YES, rel=1e-12)
    counts = limits.stats()
    assert (counts["timeouts"], counts["degraded"], counts["ok"]) == (0, 0, 1)
    res = _call(limits, Server(clock, take=5.01))
    assert res["ok"] is False and limits.stats()["timeouts"] == 1


def test_a_spent_budget_skips_the_remaining_questions():
    limits, clock = _fresh()
    server = Server(clock, take=5.0)
    res = _call(limits, server, TWO_Q)
    assert len(server.calls) == 1                                   # the second question was never sent
    assert res["answers"]["first"]["noul"] == pytest.approx(P_YES, rel=1e-12)
    assert res["answers"]["second"] == {"error": "timeout: the 5s call budget was spent"}
    assert res["ok"] is False and res["degraded"] is True and res["error"] == "second: timeout"
    assert limits.stats()["timeouts"] == 1


def test_decide_accepts_an_explicit_budget_and_the_environment_sets_the_default(monkeypatch):
    limits, _ = _fresh()
    server = Server()
    _call(limits, server, timeout=2.5)
    assert server.calls[0][1] == 2.5                                # the argument is the whole-call budget
    assert limits.timeout == 5.0
    monkeypatch.setenv(D.TIMEOUT_ENV, "7")
    assert D.Limits().timeout == 7.0
    for junk in ("abc", "-1", "0", "nan"):
        monkeypatch.setenv(D.TIMEOUT_ENV, junk)
        assert D.Limits().timeout == 5.0
    monkeypatch.setenv(D.TIMEOUT_ENV, "7")
    assert D.Limits(timeout=3.0).timeout == 3.0                     # an argument beats the environment


def test_a_timeout_exception_from_the_server_is_counted_as_a_timeout():
    class ReadTimeout(Exception):                                   # requests.ReadTimeout is not a TimeoutError
        pass
    limits, _ = _fresh()
    res = _call(limits, Server(script=[ReadTimeout("read timed out")]))
    assert res["answers"] == {"q": {"error": "read timed out"}} and res["ok"] is False
    assert limits.stats()["timeouts"] == 1
    res = _call(limits, Server(script=[RuntimeError("connection reset")]))       # twin: another error is no timeout
    assert res["answers"] == {"q": {"error": "connection reset"}} and res["ok"] is False
    counts = limits.stats()
    assert (counts["timeouts"], counts["degraded"]) == (1, 2)


# ------------------------------------------------------------------------------------------ 2 in flight

class Blocking:
    """``post`` that parks every call inside the server until ``release`` is set; ``entered`` fills as calls arrive."""

    def __init__(self, parties):
        self.entered = threading.Barrier(parties)
        self.release = threading.Event()
        self.calls = []

    def __call__(self, url, body, timeout):
        self.calls.append(timeout)
        self.entered.wait(GUARD)
        assert self.release.wait(GUARD)
        return YES


def _in_background(limits, server, count):
    results, threads = [], []
    for _ in range(count):
        t = threading.Thread(target=lambda: results.append(_call(limits, server)), daemon=True)
        t.start()
        threads.append(t)
    return results, threads


def _call_without_waiting(limits, server):
    """Run one call on its own thread. ``(thread, answers, waited)``; ``waited`` is True if it was still parked after GUARD s.

    A call that wrongly waits for a slot is therefore a failed assertion, not a deadlocked test: the caller releases
    the parked workers afterwards either way.
    """
    answers = []
    caller = threading.Thread(target=lambda: answers.append(_call(limits, server)), daemon=True)
    caller.start()
    caller.join(GUARD)
    return caller, answers, caller.is_alive()


def test_a_third_concurrent_call_gets_the_rules_fallback_at_once():
    limits, _ = _fresh()
    server = Blocking(parties=3)                                    # two workers and this thread
    results, threads = _in_background(limits, server, 2)
    server.entered.wait(GUARD)                                      # both are now inside the server
    assert limits.stats()["inflight"] == 2
    third_server = Server()
    caller, answers, waited = _call_without_waiting(limits, third_server)
    if waited:                                                      # fail cleanly: free the workers, then report
        server.release.set()
        caller.join(GUARD)
        pytest.fail("the third call waited for a slot instead of falling back")
    counts = limits.stats()                                         # taken while both workers are still parked
    server.release.set()
    third = answers[0]
    assert third["fallback"] == "busy" and third["ok"] is False and third["degraded"] is True
    assert third["error"] == "decision model busy: use the rules" and third["answers"] == {}
    assert third_server.calls == []                                 # the model was not asked
    assert (counts["fallback_busy"], counts["inflight"], counts["breaker"]) == (1, 2, "closed")
    for t in threads:
        t.join(GUARD)
    assert [r["ok"] for r in results] == [True, True] and limits.stats()["inflight"] == 0
    after = _call(limits, third_server)                             # twin: with the slots free the same call runs
    assert after["fallback"] is None and after["ok"] is True and len(third_server.calls) == 1


def test_the_in_flight_cap_is_configurable():
    limits = D.Limits(max_inflight=1, clock=Clock())
    server = Blocking(parties=2)
    results, threads = _in_background(limits, server, 1)
    server.entered.wait(GUARD)
    caller, answers, waited = _call_without_waiting(limits, Server())
    server.release.set()
    caller.join(GUARD)
    threads[0].join(GUARD)
    assert waited is False                                          # it did not wait for the one slot
    assert answers[0]["fallback"] == "busy"                         # one slot: the second call is already too many
    assert results[0]["ok"] is True


def test_the_in_flight_cap_has_an_environment_override(monkeypatch):
    assert D.Limits().max_inflight == 2
    monkeypatch.setenv(D.MAX_INFLIGHT_ENV, "3")
    assert D.Limits().max_inflight == 3
    monkeypatch.setenv(D.MAX_INFLIGHT_ENV, "two")
    assert D.Limits().max_inflight == 2


def test_a_slot_is_released_when_a_call_degrades():
    limits = D.Limits(max_inflight=1, threshold=99, clock=Clock())
    for _ in range(3):
        assert _call(limits, Server(default=NO_LETTERS))["fallback"] is None     # degraded, yet each gets the one slot
    assert limits.stats()["inflight"] == 0
    assert _call(limits, Server())["ok"] is True


# ------------------------------------------------------------------------------------------ the breaker

def test_three_consecutive_degraded_calls_open_the_breaker():
    limits, _ = _fresh()
    for _ in range(2):
        _degrade(limits)
    assert limits.stats()["breaker"] == "closed"
    _degrade(limits)
    counts = limits.stats()
    assert (counts["breaker"], counts["breaker_opened"], counts["consecutive_degraded"]) == ("open", 1, 3)
    server = Server()
    res = _call(limits, server)
    assert res["fallback"] == "breaker_open" and res["degraded"] is True and res["answers"] == {}
    assert res["error"] == "decision model breaker open: use the rules" and server.calls == []
    assert limits.stats()["fallback_breaker_open"] == 1


def test_a_good_call_resets_the_consecutive_count():
    limits, _ = _fresh()
    _degrade(limits)
    _degrade(limits)
    assert _call(limits, Server())["ok"] is True
    _degrade(limits)
    _degrade(limits)
    counts = limits.stats()
    assert (counts["breaker"], counts["consecutive_degraded"], counts["breaker_opened"]) == ("closed", 2, 0)


def test_the_breaker_half_opens_after_exactly_the_cooldown():
    limits, clock = _fresh()
    for _ in range(3):
        _degrade(limits)
    clock.advance(59.0)
    server = Server()
    assert _call(limits, server)["fallback"] == "breaker_open" and server.calls == []
    clock.advance(1.0)                                              # 60 s since it opened: the probe goes through
    probe = _call(limits, server)
    assert probe["fallback"] is None and probe["ok"] is True and len(server.calls) == 1
    counts = limits.stats()
    assert (counts["breaker"], counts["breaker_probes"], counts["breaker_closed"]) == ("closed", 1, 1)
    assert _call(limits, server)["ok"] is True and len(server.calls) == 2       # closed again: ordinary calls


def test_a_degraded_probe_reopens_the_breaker_for_a_fresh_cooldown():
    limits, clock = _fresh()
    for _ in range(3):
        _degrade(limits)
    clock.advance(60.0)
    assert _degrade(limits)["fallback"] is None                     # the probe ran and failed
    counts = limits.stats()
    assert (counts["breaker"], counts["breaker_opened"], counts["breaker_probes"]) == ("open", 2, 1)
    clock.advance(59.0)
    server = Server()
    assert _call(limits, server)["fallback"] == "breaker_open" and server.calls == []
    clock.advance(1.0)
    assert _call(limits, server)["ok"] is True                      # twin: a second probe, now good
    assert limits.stats()["breaker"] == "closed"


def test_only_one_probe_is_let_through_while_half_open():
    limits, clock = _fresh()
    for _ in range(3):
        _degrade(limits)
    clock.advance(60.0)
    server = Blocking(parties=2)
    results, threads = _in_background(limits, server, 1)            # the probe
    server.entered.wait(GUARD)
    other = Server()
    second = _call(limits, other)
    assert second["fallback"] == "breaker_open" and other.calls == []
    assert limits.stats()["inflight"] == 1
    server.release.set()
    threads[0].join(GUARD)
    assert results[0]["ok"] is True and limits.stats()["breaker"] == "closed"
    assert _call(limits, other)["ok"] is True                       # twin: after the probe closed it, calls run


def test_a_call_already_in_flight_when_the_breaker_opened_does_not_close_it():
    limits, clock = _fresh(threshold=1)
    server = Blocking(parties=2)
    results, threads = _in_background(limits, server, 1)            # in flight, will succeed
    server.entered.wait(GUARD)
    _degrade(limits)                                                # threshold 1: this opens the breaker
    assert limits.stats()["breaker"] == "open"
    server.release.set()
    threads[0].join(GUARD)
    assert results[0]["ok"] is True
    counts = limits.stats()
    assert (counts["breaker"], counts["breaker_closed"], counts["ok"]) == ("open", 0, 1)
    assert _call(limits, Server())["fallback"] == "breaker_open"


def test_a_malformed_request_is_counted_but_never_trips_or_resets_the_breaker():
    limits, _ = _fresh()
    server = Server()
    bad = {"q": {"type": "other", "instructions": "q"}}
    for _ in range(5):
        res = _call(limits, server, bad)
        assert res["error"] == "question 'q': type must be choice, noul, or score" and res["fallback"] is None
    assert server.calls == []
    counts = limits.stats()
    assert (counts["invalid_request"], counts["breaker"], counts["consecutive_degraded"]) == (5, "closed", 0)
    _degrade(limits)
    _degrade(limits)
    _call(limits, server, bad)                                      # does not reset the two degraded calls
    _degrade(limits)
    assert limits.stats()["breaker"] == "open"


def test_a_missing_model_or_server_is_a_degraded_call_that_counts_toward_the_breaker(monkeypatch):
    limits, _ = _fresh()
    monkeypatch.setattr(D, "_resolve_model", lambda model: "")
    for _ in range(3):
        res = D.decide(STATE, Q, base_url="http://ollama", post=Server(), limits=limits)
        assert res["error"] == f'no decision model: add a pool entry with roles = ["decide"] or set {D.MODEL_ENV}'
    assert limits.stats()["breaker"] == "open"
    monkeypatch.setattr(D, "_resolve_model", lambda model: "m:1")
    assert _call(limits, Server())["fallback"] == "breaker_open"


def test_the_breaker_settings_have_defaults_and_environment_overrides(monkeypatch):
    limits = D.Limits()
    assert (limits.threshold, limits.cooldown) == (3, 60.0)
    monkeypatch.setenv(D.BREAKER_THRESHOLD_ENV, "5")
    monkeypatch.setenv(D.BREAKER_COOLDOWN_ENV, "12.5")
    limits = D.Limits()
    assert (limits.threshold, limits.cooldown) == (5, 12.5)
    assert D.Limits(threshold=2, cooldown=1.0).threshold == 2


# ------------------------------------------------------------------------------------------ reported, not silent

def test_stats_reports_every_kind_of_event_exactly(monkeypatch):
    limits, clock = _fresh()
    monkeypatch.setattr(D, "_LIMITS", limits)
    _call(limits, Server())                                         # ok
    _call(limits, Server(clock, take=6.0))                          # timeout (degraded)
    _call(limits, Server(), {"q": {"type": "other", "instructions": "q"}})     # invalid request
    _degrade(limits)                                                # degraded: opens the breaker at the third
    _degrade(limits)
    _call(limits, Server())                                         # breaker_open fallback
    assert D.stats() == {
        "calls": 6, "ok": 1, "degraded": 3, "invalid_request": 1, "timeouts": 1, "fallback_busy": 0,
        "fallback_breaker_open": 1, "breaker_opened": 1, "breaker_probes": 0, "breaker_closed": 0,
        "breaker": "open", "consecutive_degraded": 3, "inflight": 0, "max_inflight": 2, "timeout_s": 5.0,
        "breaker_threshold": 3, "breaker_cooldown_s": 60.0}


def test_the_stats_command_prints_the_counters(monkeypatch, capsys):
    limits, _ = _fresh()
    monkeypatch.setattr(D, "_LIMITS", limits)
    _call(limits, Server())
    assert D._main(["stats"]) == 0
    assert json.loads(capsys.readouterr().out) == D.stats() and D.stats()["ok"] == 1


def test_the_result_keeps_its_shape_and_only_adds_the_fallback_key():
    limits, _ = _fresh()
    res = _call(limits, Server())
    assert set(res) == {"ok", "degraded", "model", "answers", "error", "ms", "fallback"}
    assert (res["ok"], res["degraded"], res["model"], res["error"], res["fallback"]) == (True, False, "m:1", None, None)


def test_the_module_instance_serves_a_call_that_names_no_limits(monkeypatch):
    limits, _ = _fresh()
    monkeypatch.setattr(D, "_LIMITS", limits)
    server = Server()
    res = D.decide(STATE, Q, model="m:1", base_url="http://ollama", post=server)
    assert res["ok"] is True and server.calls == [("http://ollama/api/chat", 5.0)]
    assert D.stats()["calls"] == 1 and D.stats()["ok"] == 1
