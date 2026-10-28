"""loci/vitals.py: host vitals sampler and static degradation ladder (Tripwire-Governor slice S9).

Everything runs against injected samplers, a fake clock and tmp_path; the real state file under ~/.loci is
never touched. The only tests that call the real OS samplers are the 0-subprocess recorder and the
Windows cross-checks.

Mutation log (each applied to loci/vitals.py; the named tests went red; then reverted):
- commit threshold ">= limit" -> "> limit": test_commit_charge_boundaries, test_relaxed_commit_hold_boundary
- escalation needs 1 tick instead of 2 (hysteresis removed): test_escalation_needs_two_consecutive_ticks, test_escalation_lands_on_the_lowest_level_the_two_ticks_agree_on
- de-escalation immediate (hysteresis removed): test_deescalation_waits_five_minutes_exactly, test_deescalation_timer_restarts_when_the_hold_level_returns
- de-escalation ">= DEESCALATE_S" -> "> DEESCALATE_S": test_deescalation_waits_five_minutes_exactly
- hold margin removed (relaxed == raw): test_hold_margin_keeps_the_level_inside_the_margin_band
- process counts via subprocess (tasklist) in default_process_counter: test_a_tick_with_the_real_samplers_spawns_no_subprocess
- process count ">" -> ">=": test_process_count_boundaries
- disk "<" -> "<=": test_free_disk_boundaries
- a failed reading read as 0 / fine: test_a_failed_reading_is_degraded_and_floors_the_level, test_degraded_reading_is_never_fine_even_when_the_rest_is_healthy
- stale ">= 30" -> "> 30": test_state_age_boundary_is_stale_at_thirty_seconds
- stale state still trusted for spawning: test_stale_state_blocks_spawning_but_not_serving
- write_atomic writes the target directly (no temp + os.replace): test_atomic_write_replaces_a_complete_temp_file
- check_on_c removed: test_non_c_paths_are_refused_with_not_on_c, test_vitals_and_sinks_refuse_non_c_paths
- min over disks -> first disk: test_disk_reader_reports_the_smallest_free_space
- unknown action allowed instead of denied: test_unknown_action_is_denied_and_known_ones_are_not
- L2 "one delegate" rule removed: test_allows_at_matrix, test_current_level_and_allows_follow_the_ticks
- escalation target = highest instead of lowest agreed level; de-escalation target = lowest instead of highest: the two "lowest/highest" tests
- the escalation counter not reset by a cool tick: test_a_single_spike_never_escalates_but_two_in_a_row_do
- a rising tick no longer voids a running de-escalation window: SURVIVED the first draft of the suite (only a full escalation was
  tested); test_one_hot_tick_inside_a_deescalation_window_restarts_the_window was added and then went red
- heartbeat only written on a transition: test_state_file_content_and_heartbeat_each_tick, test_the_thread_ticks_until_stopped_and_then_no_more
- hold margins removed one signal at a time (commit, process, disk): test_hold_margin_keeps_the_level_inside_the_margin_band (all three)
"""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import pytest

from loci import vitals as vt

GIB = 1024 ** 3
T0 = 1000.0


class Clock:
    def __init__(self, now=T0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class Env:
    """Injected samplers that read mutable values and record every call."""

    def __init__(self, commit=50.0, bash=5, git=1, disk=100.0):
        self.commit, self.bash, self.git, self.disk = commit, bash, git, disk
        self.calls = collections.Counter()

    def read_commit(self):
        self.calls["commit"] += 1
        return self.commit

    def read_procs(self):
        self.calls["procs"] += 1
        return {"bash": self.bash, "git": self.git}

    def read_disk(self):
        self.calls["disk"] += 1
        return self.disk


def make(tmp_path, env=None, clock=None, **kw):
    env = env or Env()
    clock = clock or Clock()
    events = []
    v = vt.Vitals(
        commit_reader=env.read_commit, process_counter=env.read_procs, disk_reader=env.read_disk,
        clock=clock, monotonic=clock, state_path=tmp_path / "state.json", audit_sink=events.append, **kw,
    )
    return v, env, clock, events


def beat(v, clock, n=1, step=5.0):
    """n ticks, each `step` seconds after the previous one; returns the level after each."""
    out = []
    for _ in range(n):
        clock.advance(step)
        out.append(v.tick())
    return out


def R(commit=50.0, bash=5, git=1, disk=100.0, errors=()):
    return vt.Readings(commit, bash, git, disk, tuple(errors))


# ------------------------------------------------------------------------------------------ the pure level function

@pytest.mark.parametrize("commit,level", [
    (0.0, 0), (69.9, 0), (70.0, 1), (79.9, 1), (80.0, 2), (87.9, 2), (88.0, 3), (93.9, 3), (94.0, 4), (100.0, 4),
])
def test_commit_charge_boundaries(commit, level):
    assert vt.level_for(R(commit=commit)) == level


@pytest.mark.parametrize("free,level", [
    (100.0, 0), (15.0, 0), (14.9, 1), (8.0, 1), (7.9, 2), (4.0, 2), (3.9, 3), (0.0, 3),
])
def test_free_disk_boundaries(free, level):
    assert vt.level_for(R(disk=free)) == level


@pytest.mark.parametrize("bash,level", [(0, 0), (149, 0), (150, 0), (151, 2), (400, 2), (401, 3)])
def test_process_count_boundaries(bash, level):
    assert vt.level_for(R(bash=bash)) == level


@pytest.mark.parametrize("git,level", [(0, 0), (20, 0), (21, 2), (50, 2), (51, 3)])
def test_git_process_count_boundaries(git, level):
    assert vt.level_for(R(git=git)) == level


def test_level_is_the_worst_signal_not_the_first():
    # commit says L1, disk says L2, bash says L0: the answer is L2, and it is not the first signal listed.
    assert vt.level_for(R(commit=71.0, bash=5, disk=7.0)) == 2
    assert vt.level_for(R(commit=71.0, bash=5, disk=50.0)) == 1  # positive twin: without the disk signal it is L1
    assert vt.level_for(R(commit=50.0, bash=401, git=21, disk=14.0)) == 3


def test_a_failed_reading_floors_the_level_at_one_and_names_itself():
    r = R(commit=None, errors=[("commit_pct", "OSError: boom")])
    assert vt.level_for(r) == 1
    assert r.degraded == ("commit_pct",)
    healthy = R()
    assert vt.level_for(healthy) == 0  # twin: same other readings, no failure
    assert healthy.degraded == ()


def test_hold_margin_keeps_the_level_inside_the_margin_band():
    # commit: hold L3 down to 83.0 (88 - 5); L1 down to 65.0
    assert vt.level_for(R(commit=83.0), relaxed=True) == 3
    assert vt.level_for(R(commit=82.9), relaxed=True) == 2
    assert vt.level_for(R(commit=65.0), relaxed=True) == 1
    assert vt.level_for(R(commit=64.9), relaxed=True) == 0
    # processes: 10 % off -> bash 135 / 360
    assert vt.level_for(R(bash=135), relaxed=True) == 0
    assert vt.level_for(R(bash=136), relaxed=True) == 2
    assert vt.level_for(R(bash=360), relaxed=True) == 2
    assert vt.level_for(R(bash=361), relaxed=True) == 3
    # disk: +2 GB -> L1 holds below 17.0
    assert vt.level_for(R(disk=17.0), relaxed=True) == 0
    assert vt.level_for(R(disk=16.9), relaxed=True) == 1
    # the raw level of the same readings is lower, so the margin only ever HOLDS a level
    assert vt.level_for(R(commit=83.0)) == 2


def test_relaxed_commit_hold_boundary():
    assert vt.level_for(R(commit=75.0), relaxed=True) == 2
    assert vt.level_for(R(commit=74.9), relaxed=True) == 1


# ------------------------------------------------------------------------------------------ what each level allows

_ALLOWED = {
    # action: (allowed at L0..L4 with nothing running, allowed at L0..L4 with one already running)
    "start_learning": ([True, False, False, False, False], [True, False, False, False, False]),
    "spawn_delegate": ([True, True, True, False, False], [True, True, False, False, False]),
    "new_worktree": ([True, True, True, False, False], [True, True, True, False, False]),
    "keep_agentd_jobs": ([True, True, True, True, False], [True, True, True, True, False]),
    "serve_nonzero_tier": ([True, True, True, True, False], [True, True, True, True, False]),
}


@pytest.mark.parametrize("action", sorted(_ALLOWED))
def test_allows_at_matrix(action):
    idle, busy = _ALLOWED[action]
    assert [vt.allows_at(level, action, 0) for level in range(5)] == idle
    assert [vt.allows_at(level, action, 1) for level in range(5)] == busy


def test_unknown_action_is_denied_and_known_ones_are_not():
    assert vt.allows_at(0, "spawn_delgate") is False  # typo
    assert vt.allows_at(0, "spawn_delegate") is True
    assert vt.ACTIONS == ("start_learning", "spawn_delegate", "new_worktree", "keep_agentd_jobs", "serve_nonzero_tier")


# ------------------------------------------------------------------------------------------ hysteresis (fake clock)

def test_first_sample_sets_the_level_directly_and_is_audited(tmp_path):
    v, env, clock, events = make(tmp_path, Env(commit=95.0))
    assert v.current_level() == 3  # not sampled yet: unknown is not fine
    assert v.tick() == 4
    assert v.current_level() == 4
    assert events == [{
        "ts": T0, "event": "start", "from": None, "to": 4, "raw_level": 4, "drivers": ["commit_pct"], "degraded": [],
        "readings": {"commit_pct": 95.0, "bash": 5, "git": 1, "disk_free_gb": 100.0},
    }]


def test_escalation_needs_two_consecutive_ticks(tmp_path):
    v, env, clock, events = make(tmp_path)
    assert v.tick() == 0
    env.commit = 95.0
    assert beat(v, clock, 1) == [0]       # one hot tick: nothing yet
    assert beat(v, clock, 1) == [4]       # the second consecutive hot tick escalates
    assert events[-1] == {
        "ts": T0 + 10.0, "event": "level_change", "from": 0, "to": 4, "raw_level": 4, "drivers": ["commit_pct"],
        "degraded": [], "readings": {"commit_pct": 95.0, "bash": 5, "git": 1, "disk_free_gb": 100.0},
    }


def test_a_single_spike_never_escalates_but_two_in_a_row_do(tmp_path):
    v, env, clock, events = make(tmp_path)
    v.tick()
    pattern = [95.0, 50.0, 95.0, 50.0, 95.0]  # hot, cool, hot, cool, hot: never two in a row
    levels = []
    for c in pattern:
        env.commit = c
        levels += beat(v, clock)
    assert levels == [0, 0, 0, 0, 0]
    assert [e["event"] for e in events] == ["start"]
    env.commit = 95.0
    assert beat(v, clock) == [4]  # twin: the sixth tick is the second hot one in a row


def test_escalation_lands_on_the_lowest_level_the_two_ticks_agree_on(tmp_path):
    v, env, clock, events = make(tmp_path)
    v.tick()
    env.commit = 95.0   # L4
    assert beat(v, clock) == [0]
    env.commit = 71.0   # L1
    assert beat(v, clock) == [1]
    assert events[-1]["to"] == 1 and events[-1]["raw_level"] == 1


def test_deescalation_waits_five_minutes_exactly(tmp_path):
    v, env, clock, events = make(tmp_path, Env(commit=90.0))
    assert v.tick() == 3
    env.commit = 82.9                       # raw L2 and below 88 - 5 = 83: eligible to step down
    assert beat(v, clock) == [3]            # window opens here, at T0 + 5
    window = clock.now
    assert beat(v, clock, 59) == [3] * 59   # up to window + 295 s: still L3
    assert clock.now == window + 295.0
    assert beat(v, clock) == [2]            # window + 300 s: steps down
    assert events[-1]["event"] == "level_change" and (events[-1]["from"], events[-1]["to"]) == (3, 2)
    assert events[-1]["ts"] == window + 300.0


def test_hysteresis_makes_fast_sampling_irrelevant(tmp_path):
    # the same 300 s window measured with a 1 s tick: still no step down at 299 s, a step at 300 s
    v, env, clock, events = make(tmp_path, Env(commit=90.0))
    v.tick()
    env.commit = 82.9
    assert beat(v, clock, 1, step=1.0) == [3]
    window = clock.now
    assert set(beat(v, clock, 299, step=1.0)) == {3}
    assert clock.now == window + 299.0
    assert beat(v, clock, 1, step=1.0) == [2]


def test_hold_margin_stays_at_the_level_inside_the_band(tmp_path):
    # 83.0 is raw L2 (below 88) but inside the 5 point band: the L3 level is held indefinitely
    v, env, clock, events = make(tmp_path, Env(commit=90.0))
    v.tick()
    env.commit = 83.0
    assert set(beat(v, clock, 120)) == {3}   # 600 s
    assert [e["event"] for e in events] == ["start"]
    env.commit = 82.9                        # twin: one tenth of a point lower and it steps down after 300 s
    beat(v, clock, 1)
    assert beat(v, clock, 60)[-1] == 2


def test_deescalation_timer_restarts_when_the_hold_level_returns(tmp_path):
    v, env, clock, events = make(tmp_path, Env(commit=90.0))
    v.tick()
    env.commit = 82.9
    beat(v, clock, 41)                       # 200 s below, started at the first of these
    env.commit = 84.0                        # back inside the band: hold level 3 again
    assert beat(v, clock) == [3]
    env.commit = 82.9
    assert set(beat(v, clock, 59)) == {3}    # a fresh window: 295 s after it opened is not enough
    assert beat(v, clock) == [3]
    assert beat(v, clock) == [2]


def test_deescalation_steps_to_the_highest_level_justified_during_the_window(tmp_path):
    v, env, clock, events = make(tmp_path, Env(commit=90.0))
    v.tick()
    levels = []
    for c in [76.0, 70.0] * 30 + [70.0]:     # hold L2 (>= 75) and hold L1 alternate; 61 ticks = 300 s
        env.commit = c
        levels += beat(v, clock)
    assert levels[:-1] == [3] * 60
    assert levels[-1] == 2                   # the highest hold level seen in the window, not the last one (L1)


def test_a_new_escalation_cancels_a_running_deescalation_window(tmp_path):
    v, env, clock, events = make(tmp_path, Env(commit=90.0))
    v.tick()
    env.commit = 82.9
    beat(v, clock, 50)                       # 245 s into a step-down window
    env.commit = 96.0
    assert beat(v, clock) == [3]
    assert beat(v, clock) == [4]             # two hot ticks: L4
    env.commit = 50.0
    assert set(beat(v, clock, 60)) == {4}    # the old window is gone: 295 s after the new one opened, still L4
    assert beat(v, clock) == [0]


def test_one_hot_tick_inside_a_deescalation_window_restarts_the_window(tmp_path):
    v, env, clock, events = make(tmp_path, Env(commit=90.0))
    v.tick()
    env.commit = 82.9
    beat(v, clock, 41)                       # 200 s into the window
    env.commit = 96.0
    assert beat(v, clock) == [3]             # one hot tick: not enough to escalate, enough to void the window
    env.commit = 82.9
    assert set(beat(v, clock, 60)) == {3}    # a fresh window opened on the first of these; 295 s later still L3
    assert beat(v, clock) == [2]             # 300 s after the fresh window opened


def test_degraded_reading_is_never_fine_even_when_the_rest_is_healthy(tmp_path):
    v, env, clock, events = make(tmp_path)
    env.commit = float("nan")
    assert v.tick() == 1
    assert v.health()["degraded"] == ["commit_pct"]
    assert set(beat(v, clock, 120)) == {1}   # 600 s later it is still L1: degraded never decays to fine
    env.commit = 50.0
    assert set(beat(v, clock, 60)) == {1}    # recovered, but the hold window is still running
    assert beat(v, clock, 2)[-1] == 0        # twin: once the window is over it is L0 again
    assert v.health()["degraded"] == []


# ------------------------------------------------------------------------------------------ degraded readings

class _Boom(Exception):
    pass


def _raise_boom():
    raise _Boom("sampler offline")


@pytest.mark.parametrize("setup,name,reason_prefix", [
    (lambda e: setattr(e, "read_commit", _raise_boom), "commit_pct", "_Boom: sampler offline"),
    (lambda e: setattr(e, "commit", float("inf")), "commit_pct", "ValueError: reading is not finite"),
    (lambda e: setattr(e, "commit", -1.0), "commit_pct", "ValueError: reading is negative"),
    (lambda e: setattr(e, "commit", "55"), "commit_pct", "TypeError: expected a number, got str"),
    (lambda e: setattr(e, "commit", True), "commit_pct", "TypeError: expected a number, got bool"),
    (lambda e: setattr(e, "read_disk", _raise_boom), "disk_free_gb", "_Boom: sampler offline"),
    (lambda e: setattr(e, "disk", float("nan")), "disk_free_gb", "ValueError: reading is not finite"),
    (lambda e: setattr(e, "bash", 2.5), "bash", "ValueError: count is not a whole number"),
    (lambda e: setattr(e, "git", None), "git", "TypeError: expected a number, got NoneType"),
])
def test_a_failed_reading_is_degraded_and_floors_the_level(tmp_path, setup, name, reason_prefix):
    env = Env()
    setup(env)  # before make(): the sampler callables are captured at construction
    v, env, clock, events = make(tmp_path, env)
    assert v.tick() == 1
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["degraded"] == [name]
    assert state["readings"][name] is None
    assert state["level"] == 1
    assert events[0]["degraded"] == [name]
    sample = v.sample()
    assert [n for n, _ in sample.errors] == [name]
    assert sample.errors[0][1].startswith(reason_prefix)


def test_healthy_samplers_are_not_degraded(tmp_path):
    v, env, clock, events = make(tmp_path)
    assert v.tick() == 0
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["degraded"] == []
    assert state["readings"] == {"commit_pct": 50.0, "bash": 5, "git": 1, "disk_free_gb": 100.0}
    assert v.sample().errors == ()


def test_process_counter_failures_degrade_both_counts(tmp_path):
    env = Env()
    v, env, clock, events = make(tmp_path, env)
    v._process_counter = lambda: _raise_boom()
    assert v.tick() == 1
    assert v.health()["degraded"] == ["bash", "git"]
    v._process_counter = lambda: {"bash": 5}   # a census that forgot git is not a census of zero
    clock.advance(5)
    v.tick()
    assert v.health()["degraded"] == ["git"]
    v._process_counter = lambda: ["bash"]      # not a mapping at all
    clock.advance(5)
    v.tick()
    assert v.health()["degraded"] == ["bash", "git"]
    v._process_counter = env.read_procs        # twin: the real shape is accepted
    clock.advance(5)
    v.tick()
    assert v.health()["degraded"] == []


def test_degraded_changes_are_audited_with_the_reason(tmp_path):
    v, env, clock, events = make(tmp_path)
    v.tick()
    env.commit = -3.0
    beat(v, clock)
    ev = [e for e in events if e["event"] == "degraded_change"]
    assert ev == [{
        "ts": T0 + 5.0, "event": "degraded_change", "from": [], "to": ["commit_pct"], "level": 0,
        "errors": [["commit_pct", "ValueError: reading is negative: -3.0"]],
    }]


# ------------------------------------------------------------------------------------------ state file and heartbeat

def test_state_file_content_and_heartbeat_each_tick(tmp_path):
    v, env, clock, events = make(tmp_path)
    v.tick()
    path = tmp_path / "state.json"
    raw = path.read_bytes()
    assert json.loads(raw) == {
        "v": 1, "ts": T0, "seq": 1, "pid": os.getpid(), "tick_s": 5.0, "level": 0, "raw_level": 0, "degraded": [],
        "readings": {"commit_pct": 50.0, "bash": 5, "git": 1, "disk_free_gb": 100.0},
    }
    assert len(raw) < 4096
    beat(v, clock, 2)
    after = json.loads(path.read_text(encoding="utf-8"))
    assert (after["ts"], after["seq"]) == (T0 + 10.0, 3)   # rewritten every tick, level unchanged


def test_state_stays_under_four_kib_with_every_signal_degraded(tmp_path):
    v, env, clock, events = make(tmp_path)
    v._commit_reader = _raise_boom
    v._disk_reader = _raise_boom
    v._process_counter = _raise_boom
    v.tick()
    assert v.health()["state_write_errors"] == 0
    assert 0 < (tmp_path / "state.json").stat().st_size < 4096


def test_write_atomic_size_cap_boundary(tmp_path):
    p = tmp_path / "s.json"
    vt.write_atomic(p, "x" * 4095)
    assert p.read_text(encoding="utf-8") == "x" * 4095
    with pytest.raises(ValueError, match="4096 bytes, over the 4095 byte cap"):
        vt.write_atomic(p, "y" * 4096)
    assert p.read_text(encoding="utf-8") == "x" * 4095  # the refused write left the old file alone


def test_atomic_write_replaces_a_complete_temp_file(tmp_path, monkeypatch):
    p = tmp_path / "s.json"
    vt.write_atomic(p, "old")
    seen = []
    real_replace = os.replace

    def recording_replace(src, dst):
        seen.append((Path(src).parent, Path(dst), Path(src).read_text(encoding="utf-8"), p.read_text(encoding="utf-8")))
        real_replace(src, dst)

    monkeypatch.setattr(vt.os, "replace", recording_replace)
    vt.write_atomic(p, "new content")
    # at the instant of the rename the temp file already held the full new text and the target still the old
    assert seen == [(tmp_path, p, "new content", "old")]
    assert p.read_text(encoding="utf-8") == "new content"
    assert sorted(x.name for x in tmp_path.iterdir()) == ["s.json"]


def test_failed_rename_leaves_the_old_file_and_no_temp_files(tmp_path, monkeypatch):
    p = tmp_path / "s.json"
    vt.write_atomic(p, "old")

    def failing_replace(src, dst):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(vt.os, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated rename failure"):
        vt.write_atomic(p, "new content")
    assert p.read_text(encoding="utf-8") == "old"
    assert sorted(x.name for x in tmp_path.iterdir()) == ["s.json"]


def test_a_state_write_failure_is_counted_and_the_level_still_updates(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the directory should be", encoding="utf-8")
    v, env, clock, events = make(tmp_path, Env(commit=95.0))
    v.state_path = blocker / "state.json"
    assert v.tick() == 4
    h = v.health()
    assert h["state_write_errors"] == 1 and h["last_error"].startswith("state: ")
    v.state_path = tmp_path / "state.json"   # twin: a writable path works and does not add errors
    clock.advance(5)
    v.tick()
    assert v.health()["state_write_errors"] == 1
    assert json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))["level"] == 4


def test_an_audit_sink_failure_is_counted_and_does_not_stop_the_state_write(tmp_path):
    v, env, clock, events = make(tmp_path, Env(commit=95.0))
    v._audit = lambda ev: _raise_boom()
    assert v.tick() == 4
    h = v.health()
    assert h["audit_errors"] == 1 and h["last_error"] == "audit: _Boom: sampler offline"
    assert json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))["level"] == 4


def test_jsonl_audit_sink_appends_one_sorted_line_per_event(tmp_path):
    sink = vt.JsonlAuditSink(tmp_path / "sub" / "audit.jsonl")
    sink({"event": "start", "to": 0})
    sink({"to": 2, "event": "level_change"})
    lines = (tmp_path / "sub" / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines == ['{"event": "start", "to": 0}', '{"event": "level_change", "to": 2}']


# ------------------------------------------------------------------------------------------ never on another volume

@pytest.mark.parametrize("bad", ["/mnt/f/loci/state.json", "/mnt/F", "/mnt/d/x", "F:\\loci\\state.json", "D:/x/y", "f:/x"])
def test_non_c_paths_are_refused_with_not_on_c(bad):
    with pytest.raises(ValueError, match="not on C:"):
        vt.check_on_c(bad)


@pytest.mark.parametrize("good", ["/mnt/c/loci/state.json", "C:\\loci\\state.json", "c:/x", "/tmp/x/state.json", "/mnt/foo/x", "relative/state.json"])
def test_c_and_driveless_paths_are_accepted(good):
    assert vt.check_on_c(good) == Path(good)


def test_vitals_and_sinks_refuse_non_c_paths(tmp_path):
    with pytest.raises(ValueError, match="not on C:"):
        vt.Vitals(state_path="/mnt/f/loci/state.json", audit_sink=lambda e: None, disk_reader=lambda: 1.0)
    with pytest.raises(ValueError, match="not on C:"):
        vt.write_atomic(Path("/mnt/f/loci/state.json"), "{}")
    with pytest.raises(ValueError, match="not on C:"):
        vt.JsonlAuditSink("/mnt/f/loci/audit.jsonl")
    vt.Vitals(state_path=tmp_path / "state.json", audit_sink=lambda e: None, disk_reader=lambda: 1.0)  # twin
    vt.write_atomic(tmp_path / "ok.json", "{}")
    assert (tmp_path / "ok.json").read_text(encoding="utf-8") == "{}"


def test_tick_interval_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="tick_s must be positive"):
        vt.Vitals(state_path=tmp_path / "s.json", audit_sink=lambda e: None, disk_reader=lambda: 1.0, tick_s=0)
    assert vt.Vitals(state_path=tmp_path / "s.json", audit_sink=lambda e: None, disk_reader=lambda: 1.0, tick_s=0.5).tick_s == 0.5


# ------------------------------------------------------------------------------------------ reading the state elsewhere

def _publish(tmp_path, commit=50.0):
    v, env, clock, events = make(tmp_path, Env(commit=commit))
    v.tick()
    return tmp_path / "state.json"


def test_fresh_state_reports_the_actual_level(tmp_path):
    path = _publish(tmp_path, commit=95.0)
    view = vt.read_state(path, clock=Clock(T0 + 29.0))
    assert (view.status, view.level, view.age_s, view.degraded) == ("ok", 4, 29.0, ())
    assert view.readings == {"commit_pct": 95.0, "bash": 5, "git": 1, "disk_free_gb": 100.0}
    assert view.level_for("spawn") == 4


def test_stale_state_blocks_spawning_but_not_serving(tmp_path):
    path = _publish(tmp_path)  # actual level 0
    fresh = vt.read_state(path, clock=Clock(T0 + 29.0))
    stale = vt.read_state(path, clock=Clock(T0 + 31.0))
    assert fresh.status == "ok" and stale.status == "stale"
    for action in ("spawn_delegate", "start_learning", "new_worktree"):
        assert fresh.allows(action) is True       # 29 s old: the actual level L0 applies
        assert stale.allows(action) is False      # 31 s old: treated as L3
    assert stale.level_for("spawn") == 3
    assert stale.allows("keep_agentd_jobs") is True   # L3 still lets running jobs run
    assert stale.allows("serve_nonzero_tier") is True  # serving is unaffected by a missing sampler
    assert stale.age_s == 31.0 and stale.level == 0


def test_stale_state_does_not_hide_a_hotter_actual_level_from_serving_view(tmp_path):
    path = _publish(tmp_path, commit=95.0)  # actual L4
    assert vt.read_state(path, clock=Clock(T0 + 1.0)).allows("serve_nonzero_tier") is False
    assert vt.read_state(path, clock=Clock(T0 + 60.0)).allows("serve_nonzero_tier") is True  # untrusted -> serving unaffected


def test_state_age_boundary_is_stale_at_thirty_seconds(tmp_path):
    path = _publish(tmp_path)
    assert vt.read_state(path, clock=Clock(T0 + 29.999)).status == "ok"
    assert vt.read_state(path, clock=Clock(T0 + 30.0)).status == "stale"
    assert vt.read_state(path, clock=Clock(T0 + 30.0), stale_after_s=31.0).status == "ok"


def test_a_heartbeat_from_the_future_is_not_trusted(tmp_path):
    path = _publish(tmp_path)
    assert vt.read_state(path, clock=Clock(T0 - 5.0)).status == "ok"      # small skew tolerated
    assert vt.read_state(path, clock=Clock(T0 - 5.5)).status == "stale"   # more than 5 s ahead is not


def test_missing_state_file_means_l3_for_spawning(tmp_path):
    view = vt.read_state(tmp_path / "none.json", clock=Clock())
    assert (view.status, view.level, view.detail) == ("missing", None, "no state file")
    assert view.level_for("spawn") == 3
    assert view.allows("spawn_delegate") is False
    assert view.allows("start_learning") is False
    assert view.allows("serve_nonzero_tier") is True
    assert vt.read_state(_publish(tmp_path), clock=Clock()).allows("spawn_delegate") is True  # twin: present and fresh


@pytest.mark.parametrize("body", [
    "not json", "[]", "{}", '{"ts": 1000.0}', '{"level": 1}', '{"ts": "x", "level": 1}', '{"ts": 1000.0, "level": 7}',
    '{"ts": 1000.0, "level": -1}', '{"ts": 1000.0, "level": true}', '{"ts": 1000.0, "level": 1.5}', '{"ts": NaN, "level": 1}',
    '{"ts": 1000.0, "level": 1, "readings": 5}',
])
def test_corrupt_state_is_not_trusted(tmp_path, body):
    p = tmp_path / "state.json"
    p.write_text(body, encoding="utf-8")
    view = vt.read_state(p, clock=Clock())
    assert view.status == "corrupt"
    assert view.allows("spawn_delegate") is False


def test_oversize_state_file_is_corrupt(tmp_path):
    p = tmp_path / "state.json"
    p.write_text('{"ts": 1000.0, "level": 0, "pad": "' + "x" * 5000 + '"}', encoding="utf-8")
    view = vt.read_state(p, clock=Clock())
    assert view.status == "corrupt" and view.detail == "state file is 4097 bytes or more, cap 4095"


def test_unreadable_state_is_reported_as_such(tmp_path):
    d = tmp_path / "state.json"
    d.mkdir()  # a directory where the file should be
    view = vt.read_state(d, clock=Clock())
    assert view.status == "unreadable"
    assert view.level_for("spawn") == 3


# ------------------------------------------------------------------------------------------ the sampler thread

def test_current_level_and_allows_follow_the_ticks(tmp_path):
    v, env, clock, events = make(tmp_path)
    assert v.allows("spawn_delegate") is False   # unsampled == L3
    v.tick()
    assert v.current_level() == 0 and v.allows("start_learning") is True
    env.commit = 85.0
    beat(v, clock, 2)
    assert v.current_level() == 2
    assert v.allows("start_learning") is False
    assert v.allows("spawn_delegate", active=0) is True
    assert v.allows("spawn_delegate", active=1) is False


def test_the_thread_ticks_until_stopped_and_then_no_more(tmp_path):
    reached = threading.Event()
    n = {"calls": 0}

    def commit():
        n["calls"] += 1
        if n["calls"] >= 3:
            reached.set()
        return 50.0

    env = Env()
    v = vt.Vitals(commit_reader=commit, process_counter=env.read_procs, disk_reader=env.read_disk,
                  state_path=tmp_path / "s.json", audit_sink=lambda e: None, tick_s=0.001)
    def names():
        return [t.name for t in threading.enumerate()].count("loci-vitals")

    assert names() == 0
    v.start()
    try:
        assert names() == 1
        with pytest.raises(RuntimeError, match="already started"):
            v.start()
        assert reached.wait(60) is True
    finally:
        v.stop()
    assert names() == 0
    frozen = n["calls"]
    assert frozen >= 3
    assert v.health()["ticks"] == frozen
    v.stop()  # stopping twice is harmless
    assert n["calls"] == frozen
    assert json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))["seq"] == frozen


def test_a_bug_in_one_tick_is_recorded_and_the_loop_continues(tmp_path):
    reached = threading.Event()
    calls = {"n": 0}

    class FlakyClock(Clock):
        def __call__(self):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _Boom("clock failed once")
            if calls["n"] >= 4:
                reached.set()
            return super().__call__()

    env = Env()
    v = vt.Vitals(commit_reader=env.read_commit, process_counter=env.read_procs, disk_reader=env.read_disk,
                  clock=FlakyClock(), monotonic=Clock(), state_path=tmp_path / "s.json", audit_sink=lambda e: None, tick_s=0.001)
    v.start()
    try:
        assert reached.wait(60) is True
    finally:
        v.stop()
    assert v.health()["ticks"] >= 3
    assert v.health()["last_error"] == "tick: _Boom: clock failed once"


# ------------------------------------------------------------------------------------------ single instance

class FakeInstance:
    def __init__(self, free):
        self.free, self.names, self.released = free, [], 0

    def factory(self, name):
        self.names.append(name)
        return self

    def try_acquire(self):
        return self.free

    def release(self):
        self.released += 1


class FakeVitals:
    created = []

    def __init__(self, **kw):
        self.kw, self.started, self.stopped = kw, 0, 0
        FakeVitals.created.append(self)

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1


def test_a_second_instance_exits_with_already_running(tmp_path):
    FakeVitals.created = []
    inst = FakeInstance(free=False)
    waited = []
    with pytest.raises(SystemExit, match="already_running") as ei:
        vt.main(["--state", str(tmp_path / "s.json"), "--audit", str(tmp_path / "a.jsonl")],
                instance_factory=inst.factory, vitals_factory=FakeVitals, wait=waited.append)
    assert ei.value.code == "already_running"
    assert inst.names == ["Global\\loci_vitals_sampler"]
    assert FakeVitals.created == [] and waited == []   # nothing was started


def test_the_first_instance_runs_and_releases_the_mutex(tmp_path):
    FakeVitals.created = []
    inst = FakeInstance(free=True)
    waited = []
    rc = vt.main(["--state", str(tmp_path / "s.json"), "--audit", str(tmp_path / "a.jsonl"), "--tick", "2.5"],
                 instance_factory=inst.factory, vitals_factory=FakeVitals, wait=waited.append)
    assert rc == 0
    (fv,) = FakeVitals.created
    assert (fv.started, fv.stopped, inst.released) == (1, 1, 1)
    assert waited == [fv]
    assert fv.kw["state_path"] == str(tmp_path / "s.json") and fv.kw["tick_s"] == 2.5
    assert fv.kw["audit_sink"].path == tmp_path / "a.jsonl"


def test_main_releases_the_mutex_when_construction_fails(tmp_path):
    inst = FakeInstance(free=True)
    with pytest.raises(ValueError, match="not on C:"):
        vt.main(["--state", "/mnt/f/x/s.json", "--audit", str(tmp_path / "a.jsonl")],
                instance_factory=inst.factory, wait=lambda v: None)
    assert inst.released == 1


def test_the_real_single_instance_lock_is_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(vt.tempfile, "gettempdir", lambda: str(tmp_path))
    name = f"Local\\loci_vitals_test_{uuid.uuid4().hex}"
    first = vt.default_single_instance(name)
    second = vt.default_single_instance(name)
    assert first.try_acquire() is True
    try:
        assert second.try_acquire() is False   # held
    finally:
        first.release()
    third = vt.default_single_instance(name)
    assert third.try_acquire() is True         # released: free again
    third.release()


# ------------------------------------------------------------------------------------------ default samplers

def test_a_tick_with_the_real_samplers_spawns_no_subprocess(tmp_path, monkeypatch):
    spawned = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: spawned.append(("Popen", a)))
    monkeypatch.setattr(os, "system", lambda *a: spawned.append(("system", a)))
    monkeypatch.setattr(os, "popen", lambda *a, **k: spawned.append(("popen", a)))
    v = vt.Vitals(state_path=tmp_path / "state.json", audit_sink=lambda e: None, clock=Clock(), monotonic=Clock())
    level = v.tick()
    assert spawned == []
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["degraded"] == []                       # the real OS samplers all worked
    assert isinstance(state["readings"]["bash"], int) and isinstance(state["readings"]["git"], int)
    assert level == state["level"]


def test_the_spawn_recorder_does_record_a_spawn(monkeypatch):
    spawned = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: spawned.append(a) or (_ for _ in ()).throw(_Boom("stop")))
    with pytest.raises(_Boom, match="stop"):
        subprocess.run(["noop"])
    assert spawned == [(["noop"],)]


def test_posix_process_census_counts_comm_names(tmp_path):
    for pid, comm in [("101", "bash\n"), ("102", "git\n"), ("103", "BASH\n"), ("104", "python3\n"), ("105", "bash\n")]:
        (tmp_path / pid).mkdir()
        (tmp_path / pid / "comm").write_text(comm, encoding="utf-8")
    (tmp_path / "106").mkdir()       # exited between listing and reading: no comm file
    (tmp_path / "self").mkdir()      # not a pid
    (tmp_path / "self" / "comm").write_text("bash\n", encoding="utf-8")
    assert vt._posix_process_counts(("bash", "git"), proc=tmp_path) == {"bash": 3, "git": 1}


def test_posix_commit_percent_from_meminfo(tmp_path):
    mem = tmp_path / "meminfo"
    mem.write_text("MemTotal:  1 kB\nCommitLimit:    8000000 kB\nCommitted_AS:   2000000 kB\n", encoding="ascii")
    assert vt._posix_commit_pct(mem) == 25.0
    mem.write_text("MemTotal:  1 kB\n", encoding="ascii")
    with pytest.raises(OSError, match="no usable CommitLimit"):
        vt._posix_commit_pct(mem)


def test_disk_reader_reports_the_smallest_free_space(monkeypatch):
    free = {"A:\\": 20 * GIB, "B:\\": 5 * GIB, "C:\\": 40 * GIB}
    asked = []

    def disk_usage(path):
        asked.append(path)
        return collections.namedtuple("U", "total used free")(100 * GIB, 0, free[path])

    monkeypatch.setattr(vt.shutil, "disk_usage", disk_usage)
    assert vt.make_disk_reader(["A:\\", "B:\\", "C:\\"])() == 5.0
    assert asked == ["A:\\", "B:\\", "C:\\"]
    with pytest.raises(ValueError, match="at least one path"):
        vt.make_disk_reader([])


def test_disk_reader_failure_surfaces_as_a_degraded_reading(tmp_path):
    v = vt.Vitals(commit_reader=lambda: 50.0, process_counter=lambda: {"bash": 1, "git": 0},
                  disk_reader=vt.make_disk_reader([tmp_path / "does" / "not" / "exist"]),
                  state_path=tmp_path / "s.json", audit_sink=lambda e: None, clock=Clock(), monotonic=Clock())
    assert v.tick() == 1
    assert v.health()["degraded"] == ["disk_free_gb"]


@pytest.mark.skipif(sys.platform != "win32", reason="GlobalMemoryStatusEx / Toolhelp are Windows APIs; the POSIX paths have fixture tests above")
def test_windows_commit_percent_agrees_with_a_second_api():
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32), ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64), ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64), ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64), ("ullAvailExtendedVirtual", ctypes.c_uint64)]

    ms = MEMORYSTATUSEX()
    ms.dwLength = ctypes.sizeof(ms)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)) != 0
    expected = 100.0 * (1.0 - ms.ullAvailPageFile / ms.ullTotalPageFile)
    assert vt._windows_commit_pct() == pytest.approx(expected, abs=1.5)


@pytest.mark.skipif(sys.platform != "win32", reason="Toolhelp is a Windows API; the POSIX census has a fixture test above")
def test_windows_process_census_counts_a_uniquely_named_process(tmp_path):
    import shutil
    probe_name = f"loci_probe_{uuid.uuid4().hex[:10]}"
    exe = tmp_path / f"{probe_name}.exe"
    shutil.copy2(sys.executable, exe)
    assert vt._windows_process_counts([probe_name]) == {probe_name: 0}
    proc = subprocess.Popen([str(exe), "-c", "import time; time.sleep(60)"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import time
        deadline = time.monotonic() + 30
        while vt._windows_process_counts([probe_name])[probe_name] == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert vt._windows_process_counts([probe_name]) == {probe_name: 1}
    finally:
        proc.kill()
        proc.wait(30)
    assert vt._windows_process_counts([probe_name]) == {probe_name: 0}
