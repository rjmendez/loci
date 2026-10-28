"""scripts/policy_watchdog.py: the pure decision core (swap to the frozen incumbent, silent monitor = alarm).

The clock, the latch file and every sink are injected; sinks are recording lists, never raising stubs.

Mutation log (each applied to scripts/policy_watchdog.py; the named tests went red; then reverted):
- the rows_read == 0 alarm deleted: test_zero_rows_read_is_an_alarm_not_a_quiet_pass
- the latch ignored (already-latched arms judged afresh): test_degenerate_never_auto_resumes_until_an_operator_clears_it
- heartbeat staleness ">" -> ">=": test_a_heartbeat_is_silent_only_past_fifteen_minutes
- heartbeat check skips a name (e.g. nightly): test_each_silent_heartbeat_alarms_by_name
- the swap decision keyed on any verdict (not just DEGENERATE): test_a_healthy_learned_arm_is_left_alone, test_too_early_does_nothing
- the window filter removed: test_rows_older_than_the_window_are_not_judged
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "policy_watchdog.py"


def _load():
    spec = importlib.util.spec_from_file_location("policy_watchdog_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


wd = _load()
NOW = 1_000_000.0
FRESH = {"watchdog": NOW - 10, "governor": NOW - 10, "nightly": NOW - 10}


def _row(arm, *, refused=False, quality=0.75, cost=0.0025, ts=NOW - 60, **extra):
    row = {"arm": arm, "ts": ts, "completed": True, "refused": refused, "escalated": False,
           "cost_usd": cost, "quality": None if refused else quality}
    row.update(extra)
    return row


def _rows(arm, n, **kw):
    return [_row(arm, **kw) for _ in range(n)]


class _Run:
    """One run_once call with recording sinks."""

    def __init__(self, tmp_path, rows, *, heartbeats=None, config=None, now=NOW):
        self.swaps, self.alarms, self.beats = [], [], []
        self.latch = wd.LatchStore(tmp_path / "latch.json")
        self.decision = wd.run_once(
            rows, config=config or wd.WatchdogConfig(incumbent="static", learned=("router",)),
            heartbeats=FRESH if heartbeats is None else heartbeats, now=lambda: now, latch=self.latch,
            swap_sink=self.swaps.append, alarm_sink=self.alarms.append,
            heartbeat_sink=lambda name, ts: self.beats.append((name, ts)))


def _good_incumbent():
    return _rows("static", 40, cost=0.0025, quality=0.75)


def test_a_degenerate_arm_is_swapped_to_the_incumbent_and_latched(tmp_path):
    run = _Run(tmp_path, _good_incumbent() + _rows("router", 40, refused=True, cost=0.0))
    assert run.decision.swap == ("router",)
    assert run.decision.verdicts["router"].reasons == ("refusal_above_incumbent",)
    assert run.swaps == [{"arm": "router", "to": "static", "ts": NOW, "reasons": ["refusal_above_incumbent"],
                          "new": True}]
    assert run.latch.latched() == {"router": {"at": NOW, "reasons": ["refusal_above_incumbent"]}}
    assert run.alarms == []
    assert run.beats == [("watchdog", NOW)]


def test_a_healthy_learned_arm_is_left_alone(tmp_path):
    run = _Run(tmp_path, _good_incumbent() + _rows("router", 40, cost=0.00125, quality=0.75))
    assert run.decision.swap == ()
    assert run.decision.verdicts["router"].label == "healthy"
    assert (run.swaps, run.alarms, run.latch.latched()) == ([], [], {})


def test_too_early_does_nothing(tmp_path):
    run = _Run(tmp_path, _good_incumbent() + _rows("router", 10, refused=True))
    assert run.decision.verdicts["router"].label == "too_early"
    assert (run.decision.swap, run.swaps, run.latch.latched()) == ((), [], {})


def test_degenerate_never_auto_resumes_until_an_operator_clears_it(tmp_path):
    bad = _good_incumbent() + _rows("router", 40, refused=True, cost=0.0)
    healthy = _good_incumbent() + _rows("router", 40, cost=0.00125, quality=0.75)
    _Run(tmp_path, bad)
    # the arm now looks healthy, but the latch holds: still swapped, reported as not new, not re-judged
    again = _Run(tmp_path, healthy)
    assert again.decision.swap == ("router",)
    assert again.decision.verdicts == {}
    assert again.swaps == [{"arm": "router", "to": "static", "ts": NOW, "reasons": ["refusal_above_incumbent"],
                            "new": False}]
    # only the explicit operator act lifts it, and it is recorded
    again.latch.clear("router", "ops-on-call", "root cause fixed in #123", at=NOW + 5)
    after = _Run(tmp_path, healthy)
    assert after.decision.swap == ()
    assert after.decision.verdicts["router"].label == "healthy"
    assert json.loads((tmp_path / "latch.json").read_text())["cleared"][0]["operator"] == "ops-on-call"


def test_clearing_needs_an_operator_and_a_reason_and_a_latched_arm(tmp_path):
    latch = wd.LatchStore(tmp_path / "latch.json")
    latch.latch("router", NOW, ["nonfinite"])
    with pytest.raises(ValueError, match="clearing a latch needs an operator and a reason"):
        latch.clear("router", " ", "why", at=NOW)
    with pytest.raises(KeyError, match="arm 'other' is not latched"):
        latch.clear("other", "op", "why", at=NOW)
    assert list(latch.latched()) == ["router"]  # nothing was cleared by the refusals


def test_zero_rows_read_is_an_alarm_not_a_quiet_pass(tmp_path):
    run = _Run(tmp_path, iter(()))
    assert run.decision.alarms == ("rows_read_zero",)
    assert run.decision.rows_read == 0
    assert run.alarms == [{"alarm": "rows_read_zero", "ts": NOW}]
    # positive twin: the same configuration with rows read raises no such alarm
    ok = _Run(tmp_path, _good_incumbent() + _rows("router", 40, cost=0.00125))
    assert ok.decision.alarms == ()
    assert ok.decision.rows_read == 80


def test_a_window_of_only_malformed_rows_is_an_alarm(tmp_path):
    junk = ["not a dict", {"arm": "router"}, {"arm": "router", "ts": "x", "cost_usd": 1}, {"ts": NOW, "cost_usd": 1},
            {"arm": "router", "ts": NOW}]
    run = _Run(tmp_path, junk)
    assert run.decision.alarms == ("all_rows_malformed",)
    assert (run.decision.rows_read, run.decision.rows_bad) == (5, 5)


@pytest.mark.parametrize("name", wd.HEARTBEAT_NAMES)
def test_each_silent_heartbeat_alarms_by_name(tmp_path, name):
    beats = {**FRESH, name: NOW - 901}
    run = _Run(tmp_path, _good_incumbent(), heartbeats=beats)
    assert run.decision.alarms == (f"heartbeat_silent:{name}",)
    assert run.alarms == [{"alarm": f"heartbeat_silent:{name}", "ts": NOW}]


def test_a_heartbeat_is_silent_only_past_fifteen_minutes(tmp_path):
    exactly = _Run(tmp_path, _good_incumbent(), heartbeats={**FRESH, "governor": NOW - 900})
    never = _Run(tmp_path, _good_incumbent(), heartbeats={**FRESH, "governor": None})
    assert exactly.decision.alarms == ()
    assert never.decision.alarms == ("heartbeat_silent:governor",)


def test_rows_older_than_the_window_are_not_judged(tmp_path):
    old = NOW - 86400 - 1
    stale = _good_incumbent() + _rows("router", 40, refused=True, ts=old)
    run = _Run(tmp_path, stale)
    assert run.decision.verdicts["router"].label == "too_early"  # zero in-window rows for the arm
    assert run.decision.swap == ()
    # positive twin: the same rows inside the window are degenerate
    other = tmp_path / "in_window"
    other.mkdir()
    fresh = _Run(other, _good_incumbent() + _rows("router", 40, refused=True, ts=NOW - 86400 + 1))
    assert fresh.decision.swap == ("router",)


def test_other_arms_are_ignored(tmp_path):
    run = _Run(tmp_path, _good_incumbent() + _rows("router", 40, cost=0.00125) + _rows("experiment", 99, refused=True))
    assert run.decision.swap == ()
    assert run.decision.rows_read == 179


def test_a_nonfinite_incumbent_is_reported_as_unjudgeable_not_swapped(tmp_path):
    inc = _rows("static", 40, cost=float("nan"))
    run = _Run(tmp_path, inc + _rows("router", 40, cost=0.00125))
    assert run.decision.swap == ()
    assert run.decision.alarms == ("unjudgeable:incumbent 'static' has a non-finite value; cannot judge against it",)


def test_a_nonfinite_learned_arm_is_degenerate(tmp_path):
    run = _Run(tmp_path, _good_incumbent() + _rows("router", 40, cost=float("nan")))
    assert run.decision.swap == ("router",)
    assert run.decision.verdicts["router"].reasons == ("nonfinite",)


# ---- the command line -------------------------------------------------------------------------------

def _write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _cli(tmp_path, *extra, clock=NOW):
    argv = ["--rows", str(tmp_path / "rows.jsonl"), "--incumbent", "static", "--learned", "router",
            "--state", str(tmp_path / "latch.json"), "--heartbeat-dir", str(tmp_path / "hb"),
            "--alarm-log", str(tmp_path / "alarms.jsonl"), "--swap-log", str(tmp_path / "swaps.jsonl"), *extra]
    return wd.main(argv, clock=lambda: clock)


def _beat_files(tmp_path, at):
    (tmp_path / "hb").mkdir(exist_ok=True)
    for name in wd.HEARTBEAT_NAMES:
        (tmp_path / "hb" / name).write_text(repr(at), encoding="utf-8")


def test_cli_exit_codes_and_logs(tmp_path, capsys):
    _beat_files(tmp_path, NOW - 5)
    _write_rows(tmp_path / "rows.jsonl", _good_incumbent() + _rows("router", 40, cost=0.00125))
    assert _cli(tmp_path) == 0
    assert (tmp_path / "hb" / "watchdog").read_text() == repr(NOW)  # own heartbeat refreshed
    assert not (tmp_path / "swaps.jsonl").exists()
    assert json.loads(capsys.readouterr().out)["swap"] == []

    _write_rows(tmp_path / "rows.jsonl", _good_incumbent() + _rows("router", 40, refused=True, cost=0.0))
    assert _cli(tmp_path) == 3
    swap = json.loads((tmp_path / "swaps.jsonl").read_text().splitlines()[0])
    assert (swap["arm"], swap["to"], swap["reasons"], swap["new"]) == ("router", "static",
                                                                       ["refusal_above_incumbent"], True)


def test_cli_a_missing_rows_file_is_an_alarm_exit(tmp_path):
    _beat_files(tmp_path, NOW - 5)
    assert _cli(tmp_path) == 2
    assert json.loads((tmp_path / "alarms.jsonl").read_text().splitlines()[0]) == {"alarm": "rows_read_zero", "ts": NOW}


def test_cli_clear_latch_is_the_only_resume(tmp_path):
    _beat_files(tmp_path, NOW - 5)
    _write_rows(tmp_path / "rows.jsonl", _good_incumbent() + _rows("router", 40, refused=True, cost=0.0))
    assert _cli(tmp_path) == 3
    _write_rows(tmp_path / "rows.jsonl", _good_incumbent() + _rows("router", 40, cost=0.00125))
    assert _cli(tmp_path) == 3  # still latched although the rows look healthy
    assert _cli(tmp_path, "--clear-latch", "router", "--operator", "me", "--reason", "fixed") == 0
    assert _cli(tmp_path) == 0
