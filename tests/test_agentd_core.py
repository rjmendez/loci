"""loci/agentd.py + loci/agentd_job_win.py: lease, start gates, worktree, deadman, Job Object (S10).

Everything runs against injected fakes (fake clocks, a recording git, a fake job) and tmp_path. The only
tests that touch the OS for real are the Job Object tests at the bottom, which are skipped only off Windows
(a platform condition, not an environment variable) and only ever touch a child process of their own.

Mutation log (each applied to the production code; the named tests went red; then reverted):
- KILL_ON_JOB_CLOSE removed from JobLimits.limit_flags: test_real_close_without_kill_leaves_no_survivors,
  test_limit_flags_are_exactly_active_process_job_memory_and_kill_on_close
- deadman progress counted by file mtime (FileProbe reads st_mtime_ns): test_a_touch_loop_without_transcript_growth_or_cpu_is_not_progress
- clean-repo check skipped in check_start: test_untracked_file_boundary_is_5000_passes_5001_refused,
  test_real_git_untracked_counter_over_the_cap
- wall limit ">=" -> ">": test_wall_limit_boundary_is_exact
- turn limit ">" -> ">=": test_turn_limit_boundary_is_exact
- concurrency cap ignored: test_concurrency_by_governor_level
- run-count seeding from the store dropped: test_run_count_survives_a_restart_and_resets_on_the_next_day
- disk check "<" -> "<=": test_free_disk_boundary_is_ten_gb_exactly
- deadman ">=" -> ">": test_silent_agent_is_killed_at_ninety_seconds
- finalize forgets the leak: test_leak_is_counted_and_reported_in_the_summary
- job start order (assign after resume): test_job_start_order_and_limits_are_exact
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from loci import agentd as ad
from loci import vitals
from loci.agentd_job_win import (
    JOB_OBJECT_LIMIT_ACTIVE_PROCESS,
    JOB_OBJECT_LIMIT_JOB_MEMORY,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    FinishReport,
    JobError,
    JobLimits,
    JobObject,
    JobOsApi,
    SpawnedProcess,
    Win32JobApi,
)

GIB = 1024 ** 3
WIN = sys.platform == "win32"


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class RecordingGit:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = responses or {}

    def __call__(self, args, cwd):
        args = list(args)
        self.calls.append((args, os.fspath(cwd)))
        for prefix, result in self.responses.items():
            if args[:len(prefix)] == list(prefix):
                return result
        if args[0] == "rev-parse":
            return ad.GitResult(0, "deadbeef\n", "")
        return ad.GitResult(0, "", "")

    def verbs(self):
        return [c[0][0] for c in self.calls]


class FakeJob:
    def __init__(self):
        self.started = None
        self.exit = None
        self.cpu = 0.0
        self.killed = 0
        self.finished = 0
        self.start_error = None
        self.report = FinishReport(census=0, leak=False, survivors=0, killed=False)

    def start(self, argv, cwd, env=None, *, spawner=None, stdout=None, stderr=None):
        if self.start_error is not None:
            raise self.start_error
        self.started = SimpleNamespace(argv=list(argv), cwd=cwd, env=env, spawner=spawner)

    def exit_code(self):
        return self.exit

    def cpu_seconds(self):
        return self.cpu

    def kill(self, exit_code=1):
        self.killed += 1

    def finish(self):
        self.finished += 1
        return self.report


class RecordingContainment:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def prepare(self, worktree):
        self.calls.append(("prepare", os.fspath(worktree)))
        if self.fail:
            raise OSError("icacls refused")

    def spawn_suspended(self, argv, cwd, env, stdout=None, stderr=None):
        raise AssertionError("never called: FakeJob.start does not spawn")

    def release(self, worktree):
        self.calls.append(("release", os.fspath(worktree)))


def make(tmp_path, *, level=0, enabled=True, free=50.0, untracked=0, config=None, git=None, store=True,
         containment=None, governor=None, **overrides):
    clock, wall = Clock(1000.0), Clock(86400.0 * 100)
    box = SimpleNamespace(level=level, free=free, untracked=untracked, counter_calls=[])
    jobs = []
    git = git if git is not None else RecordingGit()
    ids = iter(f"r{i}" for i in range(1, 100))

    def job_factory():
        jobs.append(FakeJob())
        return jobs[-1]

    def counter(repo, limit):
        box.counter_calls.append((os.fspath(repo), limit))
        return box.untracked

    cfg = config or ad.AgentdConfig(worktrees_root=tmp_path / "wt", home=tmp_path / "home")
    run_store = ad.RunStore(tmp_path / "runs.jsonl") if store else None
    kwargs = dict(
        job_factory=job_factory,
        governor_level=governor if governor is not None else (lambda: box.level),
        clock=clock, wall_clock=wall, day_of=lambda ts: str(int(ts // 86400)),
        enabled=(lambda: enabled), git=git, untracked_counter=counter,
        free_gb_reader=lambda p: box.free, dir_size_reader=lambda p: 0,
        store=run_store, containment=containment, id_factory=lambda: next(ids))
    kwargs.update(overrides)
    agent = ad.Agentd(cfg, **kwargs)
    return SimpleNamespace(agent=agent, clock=clock, wall=wall, box=box, jobs=jobs, git=git, store=run_store,
                           cfg=cfg)


def req(tmp_path, **kw):
    repo = tmp_path / "repo"
    base = dict(backend="claude_code", argv=["claude", "-p", "task"], cwd=repo, repo=repo)
    base.update(kw)
    return ad.StartRequest(**base)


def beat(env, run, seconds, *, progress=True):
    """Advance the clock and (by default) give the agent CPU progress, then run one supervision step."""
    env.clock.advance(seconds)
    if progress:
        run.job.cpu += 1.0
    return env.agent.tick(run)


# ----------------------------------------------------------------------------- flag and inert default

@pytest.mark.parametrize("value", ["", "0", "true", "yes", " 2 "])
def test_flag_values_other_than_1_are_off(value):
    assert ad.agentd_enabled({"LOCI_AGENTD": value}) is False


def test_flag_1_is_on_and_missing_is_off():
    assert ad.agentd_enabled({"LOCI_AGENTD": "1"}) is True
    assert ad.agentd_enabled({}) is False


def test_default_path_is_inert_when_the_flag_is_off(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCI_AGENTD", raising=False)
    env = make(tmp_path)
    agent = ad.Agentd(env.cfg, job_factory=lambda: env.jobs.append(FakeJob()) or env.jobs[-1], git=env.git,
                      governor_level=lambda: 0, store=env.store, untracked_counter=lambda r, n: 0,
                      free_gb_reader=lambda p: 99.0)
    with pytest.raises(ad.StartRefused, match=r"^flag_off"):
        agent.start(req(tmp_path))
    assert env.git.calls == []
    assert env.jobs == []
    assert not (tmp_path / "runs.jsonl").exists()
    assert not (tmp_path / "wt").exists()
    assert agent.stats()["refusals"] == {"flag_off": 1}


def test_default_path_starts_when_the_flag_is_on(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCI_AGENTD", "1")
    env = make(tmp_path)
    agent = ad.Agentd(env.cfg, job_factory=lambda: env.jobs.append(FakeJob()) or env.jobs[-1], git=env.git,
                      governor_level=lambda: 0, store=env.store, untracked_counter=lambda r, n: 0,
                      free_gb_reader=lambda p: 99.0)
    run = agent.start(req(tmp_path))
    assert run.backend == "claude_code"
    assert env.git.verbs() == ["rev-parse", "worktree"]
    assert len(env.jobs) == 1


# ----------------------------------------------------------------------------- lease: wall and turns

def test_default_lease_is_20_minutes_and_60_turns(tmp_path):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path))
    assert (run.wall_s, run.turns_limit) == (1200.0, 60)


def test_hard_lease_is_60_minutes_and_150_turns(tmp_path):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path, wall_s=3600, turns=150))
    assert (run.wall_s, run.turns_limit) == (3600.0, 150)


def test_wall_over_hard_is_refused_with_its_code(tmp_path):
    env = make(tmp_path)
    with pytest.raises(ad.StartRefused, match=r"^wall_over_hard: 3601s > hard 3600s"):
        env.agent.start(req(tmp_path, wall_s=3601))
    assert env.git.calls == []


def test_turns_over_hard_is_refused_with_its_code(tmp_path):
    env = make(tmp_path)
    with pytest.raises(ad.StartRefused, match=r"^turns_over_hard: 151 > hard 150"):
        env.agent.start(req(tmp_path, turns=151))
    assert env.git.calls == []


def test_wall_limit_boundary_is_exact(tmp_path):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path))
    assert beat(env, run, 1199) is None
    assert run.job.killed == 0
    assert beat(env, run, 1) == "wall_limit"
    assert run.job.killed == 1
    assert run.stop_code == "wall_limit"


def test_turn_limit_boundary_is_exact(tmp_path):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path))
    env.agent.record_turn(run, 60)
    assert beat(env, run, 10) is None
    assert run.job.killed == 0
    env.agent.record_turn(run)
    assert run.turns == 61
    assert beat(env, run, 10) == "turn_limit"
    assert run.job.killed == 1


def test_kill_is_idempotent_and_keeps_the_first_stop_code(tmp_path):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path))
    assert beat(env, run, 1200) == "wall_limit"
    env.agent.kill_run(run, "deadman")
    assert run.stop_code == "wall_limit"
    assert run.job.killed == 1
    assert env.agent.tick(run) == "wall_limit"


# ----------------------------------------------------------------------------- deadman (progress by events, not mtimes)

def test_silent_agent_is_killed_at_ninety_seconds(tmp_path):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path))
    assert beat(env, run, 89, progress=False) is None
    assert run.job.killed == 0
    assert beat(env, run, 1, progress=False) == "deadman"
    assert run.job.killed == 1


def test_agent_with_transcript_growth_every_30_seconds_survives_ten_minutes(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text("")
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path, transcript_path=transcript))
    for i in range(20):
        env.clock.advance(30)
        with open(transcript, "a") as fh:
            fh.write(f'{{"turn": {i}}}\n')
        assert env.agent.tick(run) is None
    assert env.clock() - run.started == 600
    assert run.job.killed == 0


def test_the_same_agent_without_growth_dies_at_the_third_silent_interval(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"turn": 0}\n')
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path, transcript_path=transcript))
    assert beat(env, run, 30, progress=False) is None
    assert beat(env, run, 30, progress=False) is None
    assert beat(env, run, 30, progress=False) == "deadman"


def test_a_touch_loop_without_transcript_growth_or_cpu_is_not_progress(tmp_path):
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text('{"turn": 0}\n')
    size = transcript.stat().st_size
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path, transcript_path=transcript))
    results = []
    for i in range(1, 4):
        env.clock.advance(30)
        stamp = time.time_ns() + i * 10_000_000_000
        os.utime(transcript, ns=(stamp, stamp))
        results.append(env.agent.tick(run))
    assert transcript.stat().st_size == size
    assert results == [None, None, "deadman"]


def test_cpu_progress_counts_only_when_it_adds_up_to_half_a_second(tmp_path):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path))
    run.job.cpu = 0.4
    assert beat(env, run, 30, progress=False) is None   # 0.4 s of CPU: not progress yet
    run.job.cpu = 0.5
    assert beat(env, run, 30, progress=False) is None   # 0.5 s since the baseline: progress, clock restarts
    assert beat(env, run, 60, progress=False) is None   # 60 s silent
    assert beat(env, run, 30, progress=False) == "deadman"


def test_tiny_cpu_trickle_does_not_keep_an_agent_alive(tmp_path):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path))
    codes = []
    for _ in range(3):
        env.clock.advance(30)
        run.job.cpu += 0.1
        codes.append(env.agent.tick(run))
    assert codes == [None, None, "deadman"]


def test_missing_transcript_is_zero_bytes_not_an_error(tmp_path):
    probe = ad.FileProbe(tmp_path / "nope.jsonl", lambda: 1.5)
    assert probe.transcript_bytes() == 0
    assert probe.cpu_seconds() == 1.5


def test_probe_reports_the_file_size_in_bytes(tmp_path):
    f = tmp_path / "t.jsonl"
    f.write_bytes(b"x" * 123)
    assert ad.FileProbe(f, lambda: 0.0).transcript_bytes() == 123


# ----------------------------------------------------------------------------- governor level and concurrency

def test_max_concurrency_table_is_exact():
    assert [ad.max_concurrency(level) for level in range(5)] == [2, 2, 1, 0, 0]


def test_concurrency_by_governor_level(tmp_path):
    env = make(tmp_path, level=0)
    env.agent.start(req(tmp_path))
    env.agent.start(req(tmp_path))
    with pytest.raises(ad.StartRefused, match=r"^concurrency_limit: 2 active, limit 2 at L0"):
        env.agent.start(req(tmp_path))
    assert len(env.jobs) == 2


def test_two_runs_are_fine_at_l1_too(tmp_path):
    env = make(tmp_path, level=1)
    env.agent.start(req(tmp_path))
    env.agent.start(req(tmp_path))
    assert env.agent.stats()["active"] == 2


def test_l2_allows_a_first_run_and_refuses_a_second(tmp_path):
    env = make(tmp_path, level=2)
    env.agent.start(req(tmp_path))
    with pytest.raises(ad.StartRefused, match=r"^governor_level: governor level L2, 1 run\(s\) active"):
        env.agent.start(req(tmp_path))
    assert len(env.jobs) == 1


@pytest.mark.parametrize("level", [3, 4])
def test_l3_and_l4_refuse_even_a_first_run(tmp_path, level):
    env = make(tmp_path, level=level)
    with pytest.raises(ad.StartRefused, match=rf"^governor_level: governor level L{level}, 0 run\(s\) active"):
        env.agent.start(req(tmp_path))
    assert env.git.calls == []


def test_a_finished_run_frees_its_slot(tmp_path):
    env = make(tmp_path, level=2)
    first = env.agent.start(req(tmp_path))
    first.job.exit = 0
    env.agent.finalize(first)
    second = env.agent.start(req(tmp_path))
    assert second.run_id == "r2"


def test_unreadable_governor_fails_closed(tmp_path):
    def broken():
        raise OSError("state file locked")

    env = make(tmp_path, governor=broken)
    with pytest.raises(ad.StartRefused, match=r"^governor_unavailable: OSError: state file locked"):
        env.agent.start(req(tmp_path))
    assert env.git.calls == []


@pytest.mark.parametrize("bad", [None, True, 5, -1, "0", 1.0])
def test_a_nonsense_governor_level_fails_closed(tmp_path, bad):
    env = make(tmp_path, governor=lambda: bad)
    with pytest.raises(ad.StartRefused, match=r"^governor_unavailable: governor level"):
        env.agent.start(req(tmp_path))


def test_default_governor_reads_a_fresh_state_file(tmp_path):
    state = tmp_path / "vitals_state.json"
    state.write_text(json.dumps({"ts": 5000.0, "level": 2}))
    read = ad.default_governor_level(state, clock=lambda: 5001.0)
    assert read() == 2


def test_default_governor_refuses_a_stale_or_missing_state_file(tmp_path):
    state = tmp_path / "vitals_state.json"
    state.write_text(json.dumps({"ts": 5000.0, "level": 0}))
    with pytest.raises(ad.GovernorUnavailable, match=r"vitals state is stale"):
        ad.default_governor_level(state, clock=lambda: 5031.0)()
    with pytest.raises(ad.GovernorUnavailable, match=r"vitals state is missing"):
        ad.default_governor_level(tmp_path / "absent.json", clock=lambda: 5001.0)()


def test_default_governor_when_the_sampler_wrote_through_vitals(tmp_path):
    state = tmp_path / "vitals_state.json"
    vitals.write_atomic(state, json.dumps({"ts": 7000.0, "level": 3}))
    assert ad.default_governor_level(state, clock=lambda: 7000.5)() == 3


# ----------------------------------------------------------------------------- run count per day

def test_twelve_runs_per_day_and_the_thirteenth_is_refused(tmp_path):
    env = make(tmp_path)
    for _ in range(12):
        run = env.agent.start(req(tmp_path))
        run.job.exit = 0
        env.agent.finalize(run)
    with pytest.raises(ad.StartRefused, match=r"^run_count_limit: 12 runs already started on 100"):
        env.agent.start(req(tmp_path))
    assert len(env.jobs) == 12


def test_run_count_survives_a_restart_and_resets_on_the_next_day(tmp_path):
    env = make(tmp_path)
    for _ in range(12):
        run = env.agent.start(req(tmp_path))
        run.job.exit = 0
        env.agent.finalize(run)
    again = make(tmp_path)   # a fresh agentd over the same store file: a restart
    again.wall.now = env.wall.now
    with pytest.raises(ad.StartRefused, match=r"^run_count_limit"):
        again.agent.start(req(tmp_path))
    again.wall.advance(86400)
    assert again.agent.start(req(tmp_path)).run_id == "r1"


def test_a_refused_start_does_not_use_up_a_run(tmp_path):
    env = make(tmp_path, free=1.0)
    for _ in range(13):
        with pytest.raises(ad.StartRefused, match=r"^disk_free_low"):
            env.agent.start(req(tmp_path))
    env.box.free = 50.0
    assert env.agent.start(req(tmp_path)).run_id == "r1"


def test_unreadable_run_store_fails_closed(tmp_path):
    class Broken(ad.RunStore):
        def rows(self):
            raise PermissionError("denied")
            yield  # pragma: no cover

    env = make(tmp_path)
    env.agent.store = Broken(tmp_path / "runs.jsonl")
    with pytest.raises(ad.StartRefused, match=r"^run_count_unknown: PermissionError: denied"):
        env.agent.start(req(tmp_path))


# ----------------------------------------------------------------------------- start gates

def test_free_disk_boundary_is_ten_gb_exactly(tmp_path):
    env = make(tmp_path, free=9.99)
    with pytest.raises(ad.StartRefused, match=r"^disk_free_low: 9\.99 GB free, need 10"):
        env.agent.start(req(tmp_path))
    env.box.free = 10.0
    assert env.agent.start(req(tmp_path)).run_id == "r1"


def test_cwd_equal_to_the_home_folder_is_refused(tmp_path):
    env = make(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    with pytest.raises(ad.StartRefused, match=r"^cwd_is_home: cwd .* is the home folder"):
        env.agent.start(req(tmp_path, cwd=home, repo=tmp_path / "repo"))
    assert env.git.calls == []


def test_a_subfolder_of_home_is_fine(tmp_path):
    env = make(tmp_path)
    (tmp_path / "home" / "proj").mkdir(parents=True)
    assert env.agent.start(req(tmp_path, cwd=tmp_path / "home" / "proj")).run_id == "r1"


def test_untracked_file_boundary_is_5000_passes_5001_refused(tmp_path):
    env = make(tmp_path, untracked=5000)
    run = env.agent.start(req(tmp_path))
    assert run.run_id == "r1"
    env.box.untracked = 5001
    with pytest.raises(ad.StartRefused, match=r"^repo_untracked_limit: 5001 untracked files, cap 5000"):
        env.agent.start(req(tmp_path))
    assert env.box.counter_calls == [(os.fspath(tmp_path / "repo"), 5000)] * 2
    assert len(env.jobs) == 1


def test_an_untracked_counter_that_raises_refuses_the_start(tmp_path):
    def boom(repo, limit):
        raise OSError("git ls-files exited 128")

    env = make(tmp_path, untracked_counter=boom)
    with pytest.raises(ad.StartRefused, match=r"^repo_untracked_limit: cannot count untracked files: OSError"):
        env.agent.start(req(tmp_path))


def test_pause_file_halts_delegation_and_removing_it_resumes(tmp_path):
    pause = tmp_path / "PAUSE"
    cfg = ad.AgentdConfig(worktrees_root=tmp_path / "wt", home=tmp_path / "home", pause_path=pause)
    env = make(tmp_path, config=cfg)
    pause.write_text("")
    with pytest.raises(ad.StartRefused, match=r"^paused"):
        env.agent.start(req(tmp_path))
    pause.unlink()
    assert env.agent.start(req(tmp_path)).run_id == "r1"


@pytest.mark.parametrize("kw", [
    {"backend": "bash"}, {"argv": []}, {"mode": "rwx"}, {"turns": 0}, {"wall_s": 0}, {"wall_s": -5}, {"turns": True},
])
def test_invalid_requests_are_refused(tmp_path, kw):
    env = make(tmp_path)
    with pytest.raises(ad.StartRefused, match=r"^invalid_request"):
        env.agent.start(req(tmp_path, **kw))
    assert env.git.calls == []


def test_gates_are_checked_in_the_documented_order(tmp_path):
    env = make(tmp_path, level=3, free=1.0, untracked=9999)
    with pytest.raises(ad.StartRefused, match=r"^governor_level"):
        env.agent.start(req(tmp_path))
    env.box.level = 0
    with pytest.raises(ad.StartRefused, match=r"^disk_free_low"):
        env.agent.start(req(tmp_path))
    env.box.free = 50.0
    with pytest.raises(ad.StartRefused, match=r"^repo_untracked_limit"):
        env.agent.start(req(tmp_path))
    env.box.untracked = 0
    assert env.agent.start(req(tmp_path)).run_id == "r1"


# ----------------------------------------------------------------------------- read-write gates

def test_rw_is_off_by_default(tmp_path):
    env = make(tmp_path, containment=RecordingContainment())
    with pytest.raises(ad.StartRefused, match=r"^rw_disabled"):
        env.agent.start(req(tmp_path, mode="rw", rw_flag=True))
    assert env.git.calls == []


def test_rw_needs_the_per_run_flag_even_when_enabled(tmp_path):
    cfg = ad.AgentdConfig(worktrees_root=tmp_path / "wt", home=tmp_path / "home", rw_enabled=True)
    env = make(tmp_path, config=cfg, containment=RecordingContainment())
    with pytest.raises(ad.StartRefused, match=r"^rw_disabled"):
        env.agent.start(req(tmp_path, mode="rw", rw_flag=False))


def test_rw_needs_a_containment_object(tmp_path):
    cfg = ad.AgentdConfig(worktrees_root=tmp_path / "wt", home=tmp_path / "home", rw_enabled=True)
    env = make(tmp_path, config=cfg)
    with pytest.raises(ad.StartRefused, match=r"^rw_requires_containment"):
        env.agent.start(req(tmp_path, mode="rw", rw_flag=True))
    assert env.git.calls == []


def test_rw_with_every_gate_open_prepares_spawns_through_and_releases_containment(tmp_path):
    cfg = ad.AgentdConfig(worktrees_root=tmp_path / "wt", home=tmp_path / "home", rw_enabled=True)
    containment = RecordingContainment()
    env = make(tmp_path, config=cfg, containment=containment)
    run = env.agent.start(req(tmp_path, mode="rw", rw_flag=True))
    wt = os.fspath(tmp_path / "wt" / "r1")
    assert containment.calls == [("prepare", wt)]
    assert run.job.started.spawner == containment.spawn_suspended
    run.job.exit = 0
    env.agent.finalize(run)
    assert containment.calls == [("prepare", wt), ("release", wt)]


def test_ro_run_never_touches_containment(tmp_path):
    containment = RecordingContainment()
    env = make(tmp_path, containment=containment)
    run = env.agent.start(req(tmp_path))
    run.job.exit = 0
    env.agent.finalize(run)
    assert containment.calls == []
    assert run.job.started.spawner is None


def test_a_failing_containment_refuses_the_start_and_discards_the_worktree(tmp_path):
    cfg = ad.AgentdConfig(worktrees_root=tmp_path / "wt", home=tmp_path / "home", rw_enabled=True)
    env = make(tmp_path, config=cfg, containment=RecordingContainment(fail=True))
    with pytest.raises(ad.StartRefused, match=r"^containment_failed: OSError: icacls refused"):
        env.agent.start(req(tmp_path, mode="rw", rw_flag=True))
    assert env.git.verbs() == ["rev-parse", "worktree", "worktree", "branch"]
    assert env.jobs[0].started is None


# ----------------------------------------------------------------------------- worktree

def test_worktree_commands_are_exact(tmp_path):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path, base_ref="main"))
    repo = os.fspath(tmp_path / "repo")
    wt = os.fspath(tmp_path / "wt" / "r1")
    assert env.git.calls == [
        (["rev-parse", "--verify", "main^{commit}"], repo),
        (["worktree", "add", "-b", "agentd/r1", wt, "deadbeef"], repo),
    ]
    assert run.base_sha == "deadbeef"
    assert run.worktree == tmp_path / "wt" / "r1"
    assert run.job.started.cwd == tmp_path / "wt" / "r1"
    assert run.job.started.argv == ["claude", "-p", "task"]


def test_an_unresolvable_base_ref_is_refused_before_a_worktree_is_made(tmp_path):
    git = RecordingGit({("rev-parse",): ad.GitResult(128, "", "fatal: bad revision")})
    env = make(tmp_path, git=git)
    with pytest.raises(ad.StartRefused, match=r"^worktree_failed: cannot resolve HEAD: fatal: bad revision"):
        env.agent.start(req(tmp_path))
    assert git.verbs() == ["rev-parse"]
    assert env.jobs == []


def test_a_failed_worktree_add_starts_no_job(tmp_path):
    git = RecordingGit({("worktree", "add"): ad.GitResult(128, "", "fatal: already exists")})
    env = make(tmp_path, git=git)
    with pytest.raises(ad.StartRefused, match=r"^worktree_failed: git worktree add failed: fatal: already exists"):
        env.agent.start(req(tmp_path))
    assert env.jobs == []


def test_a_job_that_cannot_start_discards_the_fresh_worktree(tmp_path):
    env = make(tmp_path)
    env.agent._job_factory = lambda: _failing_job()
    with pytest.raises(ad.StartRefused, match=r"^job_failed: JobError: job start failed: CreateProcess failed"):
        env.agent.start(req(tmp_path))
    wt = os.fspath(tmp_path / "wt" / "r1")
    assert env.git.calls[2:] == [(["worktree", "remove", "--force", wt], os.fspath(tmp_path / "repo")),
                                 (["branch", "-D", "agentd/r1"], os.fspath(tmp_path / "repo"))]
    assert env.agent.stats()["active"] == 0
    assert env.agent.stats()["refusals"] == {"job_failed": 1}


def _failing_job():
    job = FakeJob()
    job.start_error = JobError("job start failed: CreateProcess failed")
    return job


# ----------------------------------------------------------------------------- finalize, summary row, leak

def test_summary_row_is_appended_to_the_store_with_exact_fields(tmp_path):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path, backend="copilot_cli"))
    env.agent.record_turn(run, 3)
    env.clock.advance(12.5)
    run.job.exit = 0
    summary = env.agent.finalize(run)
    rows = list(env.store.rows())
    assert rows[0] == {"event": "start", "run_id": "r1", "backend": "copilot_cli", "mode": "ro", "day": "100",
                       "ts": 8640000.0}
    assert rows[1] == summary == {
        "event": "summary", "run_id": "r1", "backend": "copilot_cli", "mode": "ro", "wall_s": 12.5, "turns": 3,
        "stop_code": "completed", "leak": False, "survivors": 0, "day": "100", "ts": 8640000.0,
        "worktree_kept": True}
    assert len(json.dumps(summary).encode()) < 1024


def test_nonzero_exit_is_exit_nonzero_and_a_kill_keeps_its_own_code(tmp_path):
    env = make(tmp_path)
    failed = env.agent.start(req(tmp_path))
    failed.job.exit = 3
    assert env.agent.finalize(failed)["stop_code"] == "exit_nonzero"
    killed = env.agent.start(req(tmp_path))
    beat(env, killed, 1200)
    assert env.agent.finalize(killed)["stop_code"] == "wall_limit"
    assert env.agent.stats()["stops"] == {"exit_nonzero": 1, "wall_limit": 1}


def test_tick_reports_an_exited_agent_and_supervise_finalizes(tmp_path):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path))
    run.job.exit = 0
    assert env.agent.tick(run) == "exited"
    sleeps = []
    summary = env.agent.supervise(run, sleep=sleeps.append)
    assert summary["stop_code"] == "completed"
    assert sleeps == []


def test_supervise_sleeps_between_ticks_until_the_agent_exits(tmp_path):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path))
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        env.clock.advance(seconds)
        run.job.cpu += 1.0
        if len(sleeps) == 3:
            run.job.exit = 0

    summary = env.agent.supervise(run, sleep=sleep, interval=5.0)
    assert sleeps == [5.0, 5.0, 5.0]
    assert summary["wall_s"] == 15.0
    assert summary["stop_code"] == "completed"


def test_finalize_is_idempotent_and_writes_one_summary_row(tmp_path):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path))
    run.job.exit = 0
    first = env.agent.finalize(run)
    second = env.agent.finalize(run)
    assert second is first
    assert run.job.finished == 1
    assert [r["event"] for r in env.store.rows()] == ["start", "summary"]


def test_leak_is_counted_and_reported_in_the_summary(tmp_path):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path))
    run.job.exit = 0
    run.job.report = FinishReport(census=2, leak=True, survivors=0, killed=True)
    summary = env.agent.finalize(run)
    assert summary["leak"] is True
    assert env.agent.stats()["leaks"] == 1
    clean = env.agent.start(req(tmp_path))
    clean.job.exit = 0
    assert env.agent.finalize(clean)["leak"] is False
    assert env.agent.stats()["leaks"] == 1


def test_the_worktree_is_never_removed_on_a_kill(tmp_path):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path))
    beat(env, run, 1200)
    env.agent.finalize(run)
    assert "remove" not in [c[0][1] for c in env.git.calls if c[0][0] == "worktree"]
    assert env.agent.finalize(run)["worktree_kept"] is True


# ----------------------------------------------------------------------------- the run store

def test_store_drops_writes_past_its_cap_and_counts_them(tmp_path):
    store = ad.RunStore(tmp_path / "s.jsonl", max_bytes=60)
    assert store.append({"a": 1}) is True
    assert store.append({"x": "y" * 80}) is False
    assert store.dropped == 1
    assert list(store.rows()) == [{"a": 1}]


def test_store_skips_and_counts_corrupt_lines(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text('{"a": 1}\nnot json\n\n{"b": 2}\n')
    store = ad.RunStore(path)
    assert list(store.rows()) == [{"a": 1}, {"b": 2}]
    assert store.corrupt_lines == 1


def test_stats_report_store_drops(tmp_path):
    env = make(tmp_path)
    env.agent.store.max_bytes = 10
    run = env.agent.start(req(tmp_path))
    run.job.exit = 0
    env.agent.finalize(run)
    stats = env.agent.stats()
    assert stats["store_dropped"] == 2
    assert stats["enabled"] is True


# ----------------------------------------------------------------------------- real git: the positive twin on a clean repo

def _git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false",
                    *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def clean_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "README.md").write_text("hello\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def test_real_git_clean_fixture_repo_starts_a_run_in_a_real_worktree(tmp_path, clean_repo):
    cfg = ad.AgentdConfig(worktrees_root=tmp_path / "wt", home=tmp_path / "home")
    env = make(tmp_path, config=cfg, git=ad.default_git, untracked_counter=ad.count_untracked,
               free_gb_reader=ad.free_gb)
    run = env.agent.start(req(tmp_path, repo=clean_repo, cwd=clean_repo))
    assert (run.worktree / "README.md").read_text() == "hello\n"
    listing = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=clean_repo, capture_output=True,
                             text=True, check=True).stdout
    assert f"branch refs/heads/agentd/{run.run_id}" in listing
    assert len(run.base_sha) == 40


def test_real_git_untracked_counter_on_a_clean_repo_is_zero(clean_repo):
    assert ad.count_untracked(clean_repo, 5000) == 0


def test_real_git_untracked_counter_over_the_cap(tmp_path, clean_repo):
    for i in range(5001):
        (clean_repo / f"f{i}.txt").write_bytes(b"")
    assert ad.count_untracked(clean_repo, 5000) == 5001
    cfg = ad.AgentdConfig(worktrees_root=tmp_path / "wt", home=tmp_path / "home")
    env = make(tmp_path, config=cfg, git=ad.default_git, untracked_counter=ad.count_untracked)
    with pytest.raises(ad.StartRefused, match=r"^repo_untracked_limit: 5001 untracked files, cap 5000"):
        env.agent.start(req(tmp_path, repo=clean_repo, cwd=clean_repo))
    assert not (tmp_path / "wt").exists()


def test_real_git_untracked_counter_at_the_cap_passes(tmp_path, clean_repo):
    for i in range(5000):
        (clean_repo / f"f{i}.txt").write_bytes(b"")
    assert ad.count_untracked(clean_repo, 5000) == 5000


def test_real_git_untracked_counter_ignores_ignored_files(tmp_path, clean_repo):
    (clean_repo / ".gitignore").write_text("build/\n")
    (clean_repo / "build").mkdir()
    for i in range(10):
        (clean_repo / "build" / f"o{i}.bin").write_bytes(b"")
    assert ad.count_untracked(clean_repo, 5000) == 1   # only .gitignore itself


def test_real_git_untracked_counter_fails_loudly_outside_a_repo(tmp_path):
    with pytest.raises(OSError, match=r"git ls-files exited 128"):
        ad.count_untracked(tmp_path, 5000)


def test_free_gb_reads_the_nearest_existing_parent(tmp_path):
    value = ad.free_gb(tmp_path / "not" / "yet" / "there")
    assert value == pytest.approx(ad.free_gb(tmp_path), abs=0.5)
    assert value > 0


def test_dir_size_sums_file_bytes(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x").write_bytes(b"1" * 100)
    (tmp_path / "y").write_bytes(b"2" * 23)
    assert ad.dir_size(tmp_path) == 123


# ----------------------------------------------------------------------------- JobObject over a recording OS API

class RecordingApi(JobOsApi):
    def __init__(self):
        self.calls = []
        self.census_values = [0]               # before terminate_job; the last value sticks
        self.census_after_kill = [0]           # after terminate_job; the last value sticks
        self.pids = [4243]
        self.killed = False
        self.fail_at = None
        self.exit = None
        self.cpu = 0.0

    def _rec(self, name, *args):
        self.calls.append((name, *args))
        if self.fail_at == name:
            raise OSError(f"{name} refused")

    def create_job(self):
        self._rec("create_job")
        return 11

    def set_extended_limits(self, job, flags, active_process_limit, job_memory_bytes):
        self._rec("set_extended_limits", job, flags, active_process_limit, job_memory_bytes)

    def set_cpu_rate(self, job, percent):
        self._rec("set_cpu_rate", job, percent)

    def spawn_suspended(self, argv, cwd, env, stdout=None, stderr=None):
        self._rec("spawn_suspended", tuple(argv))
        return SpawnedProcess(pid=4242, handle=22)

    def assign(self, job, proc):
        self._rec("assign", job, proc.pid)

    def resume(self, proc):
        self._rec("resume", proc.pid)

    def terminate_process(self, proc, exit_code):
        self._rec("terminate_process", proc.pid, exit_code)

    def exit_code(self, proc):
        return self.exit

    def active_process_count(self, job):
        self.calls.append(("census", job))
        values = self.census_after_kill if self.killed else self.census_values
        return values.pop(0) if len(values) > 1 else values[0]

    def process_ids(self, job):
        return list(self.pids)

    def cpu_seconds(self, job):
        return self.cpu

    def terminate_job(self, job, exit_code):
        self._rec("terminate_job", job, exit_code)
        self.killed = True

    def close_handle(self, handle):
        self._rec("close_handle", handle)

    def names(self):
        return [c[0] for c in self.calls if c[0] != "census"]


def test_limit_flags_are_exactly_active_process_job_memory_and_kill_on_close():
    flags = JobLimits().limit_flags
    assert flags == 0x8 | 0x200 | 0x2000 == 0x2208
    assert flags & JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    assert flags & JOB_OBJECT_LIMIT_ACTIVE_PROCESS
    assert flags & JOB_OBJECT_LIMIT_JOB_MEMORY


def test_job_start_order_and_limits_are_exact():
    api = RecordingApi()
    sink = []
    job = JobObject(api, sink=sink.append)
    proc = job.start(["agent", "--x"], "C:/wt", {"A": "1"})
    assert proc == SpawnedProcess(pid=4242, handle=22)
    assert api.calls == [
        ("create_job",),
        ("set_extended_limits", 11, 0x2208, 64, 6 * GIB),
        ("set_cpu_rate", 11, 50),
        ("spawn_suspended", ("agent", "--x")),
        ("assign", 11, 4242),
        ("resume", 4242),
    ]
    assert job.pid == 4242


def test_job_start_uses_the_containment_spawner_when_given():
    api = RecordingApi()
    seen = []

    def spawner(argv, cwd, env, stdout=None, stderr=None):
        seen.append((tuple(argv), cwd))
        return SpawnedProcess(pid=7, handle=8)

    JobObject(api).start(["a"], "wt", None, spawner=spawner)
    assert seen == [(("a",), "wt")]
    assert "spawn_suspended" not in api.names()
    assert ("assign", 11, 7) in api.calls


def test_job_limits_can_be_tuned_and_are_validated():
    api = RecordingApi()
    JobObject(api, JobLimits(active_process_limit=8, job_memory_bytes=GIB, cpu_percent=25)).start(["a"], "wt")
    assert ("set_extended_limits", 11, 0x2208, 8, GIB) in api.calls
    assert ("set_cpu_rate", 11, 25) in api.calls
    with pytest.raises(ValueError, match=r"cpu_percent must be 1\.\.100, got 101"):
        JobLimits(cpu_percent=101)
    with pytest.raises(ValueError, match=r"active_process_limit must be a positive int, got 0"):
        JobLimits(active_process_limit=0)


@pytest.mark.parametrize("fail_at,expected", [
    ("create_job", []),
    ("set_extended_limits", ["create_job", "set_extended_limits", "close_handle"]),
    ("set_cpu_rate", ["create_job", "set_extended_limits", "set_cpu_rate", "close_handle"]),
    ("spawn_suspended", ["create_job", "set_extended_limits", "set_cpu_rate", "spawn_suspended", "close_handle"]),
    ("assign", ["create_job", "set_extended_limits", "set_cpu_rate", "spawn_suspended", "assign",
                "terminate_process", "close_handle", "close_handle"]),
    ("resume", ["create_job", "set_extended_limits", "set_cpu_rate", "spawn_suspended", "assign", "resume",
                "terminate_process", "close_handle", "close_handle"]),
])
def test_a_failure_during_start_tears_everything_down(fail_at, expected):
    api = RecordingApi()
    api.fail_at = fail_at
    with pytest.raises(JobError, match=rf"job start failed: OSError: {fail_at} refused"):
        JobObject(api).start(["a"], "wt")
    if fail_at == "create_job":
        assert api.names() == ["create_job"]
    else:
        assert api.names() == expected


def test_a_job_can_only_be_started_once():
    api = RecordingApi()
    job = JobObject(api)
    job.start(["a"], "wt")
    with pytest.raises(JobError, match=r"job already started"):
        job.start(["a"], "wt")


def test_finish_with_an_empty_census_is_not_a_leak():
    api = RecordingApi()
    sink = []
    job = JobObject(api, sink=sink.append)
    job.start(["a"], "wt")
    report = job.finish()
    assert report == FinishReport(census=0, leak=False, survivors=0, killed=False)
    assert sink == []
    assert "terminate_job" not in api.names()
    assert api.names()[-2:] == ["close_handle", "close_handle"]


def test_finish_with_a_nonzero_census_logs_a_leak_and_kills_the_job():
    api = RecordingApi()
    api.census_values = [3]
    sink, sleeps = [], []
    job = JobObject(api, sink=sink.append, sleep=sleeps.append)
    job.start(["a"], "wt")
    report = job.finish()
    assert report == FinishReport(census=3, leak=True, survivors=0, killed=True)
    assert sink == [{"event": "leak", "pid": 4242, "census": 3, "pids": [4243]}]
    assert sleeps == [0.05] * 10
    assert job.leaks == 1
    assert ("terminate_job", 11, 1) in api.calls


def test_a_kill_by_agentd_is_not_a_leak_but_survivors_are_reported():
    api = RecordingApi()
    api.census_after_kill = [2]
    sink, sleeps = [], []
    job = JobObject(api, sink=sink.append, sleep=sleeps.append)
    job.start(["a"], "wt")
    job.kill()
    job.kill()
    report = job.finish()
    assert api.names().count("terminate_job") == 1
    assert report == FinishReport(census=2, leak=False, survivors=2, killed=True)
    assert sink == [{"event": "survivors_after_kill", "pid": 4242, "census": 2}]
    assert sleeps == [0.1] * 20


def test_a_kill_that_empties_the_job_has_no_survivors():
    api = RecordingApi()
    api.census_after_kill = [1, 1, 0]
    job = JobObject(api, sleep=lambda s: None)
    job.start(["a"], "wt")
    job.kill()
    assert job.finish().survivors == 0


def test_a_process_that_is_still_counted_for_a_moment_after_exit_is_not_a_leak():
    api = RecordingApi()
    api.census_values = [1, 1, 0]
    sink, sleeps = [], []
    job = JobObject(api, sink=sink.append, sleep=sleeps.append)
    job.start(["a"], "wt")
    report = job.finish()
    assert report == FinishReport(census=0, leak=False, survivors=0, killed=False)
    assert sink == []
    assert sleeps == [0.05, 0.05]
    assert "terminate_job" not in api.names()


def test_close_does_not_terminate_it_leaves_the_killing_to_the_os():
    api = RecordingApi()
    job = JobObject(api)
    job.start(["a"], "wt")
    job.close()
    assert "terminate_job" not in api.names()
    assert api.names()[-2:] == ["close_handle", "close_handle"]
    assert api.calls[-1] == ("close_handle", 11)
    assert job.census() == 0


def test_job_exposes_exit_code_and_cpu_from_the_api():
    api = RecordingApi()
    job = JobObject(api)
    assert (job.exit_code(), job.census(), job.cpu_seconds(), job.pid) == (None, 0, 0.0, None)
    job.start(["a"], "wt")
    api.exit, api.cpu = 5, 2.25
    assert (job.exit_code(), job.cpu_seconds()) == (5, 2.25)


# ----------------------------------------------------------------------------- REAL Windows Job Object (own child only)

winonly = pytest.mark.skipif(not WIN, reason="Job Objects are Windows-only; skipped on other platforms")

SPAWNER = r"""
import subprocess, sys
n, secs, mode = int(sys.argv[1]), sys.argv[2], sys.argv[3]
ok, failed, procs = [], 0, []
for _ in range(n):
    try:
        p = subprocess.Popen(["ping.exe", "-n", secs, "127.0.0.1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ok.append(p.pid); procs.append(p)
    except OSError:
        failed += 1
print(len(ok), failed, *ok, flush=True)
if mode == "wait":
    for p in procs:
        p.wait()
"""


def _alive(pid):
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = ctypes.c_void_p
    handle = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        k32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
        return code.value == 259
    finally:
        k32.CloseHandle(ctypes.c_void_p(handle))


def _wait_until(predicate, timeout=30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _run_spawner(tmp_path, n, secs, mode, limits=JobLimits()):
    out = tmp_path / "spawner.out"
    job = JobObject(Win32JobApi(), limits, sleep=time.sleep)
    with open(out, "w") as fh:
        job.start([sys.executable, "-c", SPAWNER, str(n), str(secs), mode], tmp_path, None, stdout=fh)
    assert _wait_until(lambda: out.read_text().count("\n") >= 1), "spawner never reported"
    fields = out.read_text().split()
    return job, int(fields[0]), int(fields[1]), [int(x) for x in fields[2:]]


@winonly
def test_real_500_child_fork_is_capped_by_the_active_process_limit(tmp_path):
    job, ok, failed, pids = _run_spawner(tmp_path, 500, 31, "nowait")
    try:
        peak = job.census()
        assert ok + failed == 500
        assert ok <= 63                      # 64 slots minus the spawner itself
        assert failed == 500 - ok
        assert failed >= 437
        assert peak <= 64
        assert peak >= ok                    # the children really were alive inside the job
        assert all(_alive(p) for p in pids)
    finally:
        job.finish()
    assert _wait_until(lambda: not any(_alive(p) for p in pids), 10.0)


@winonly
def test_real_ten_children_complete_and_the_census_is_zero_at_exit(tmp_path):
    job, ok, failed, pids = _run_spawner(tmp_path, 10, 2, "wait")
    assert (ok, failed, len(pids)) == (10, 0, 10)
    assert _wait_until(lambda: job.exit_code() is not None)
    assert job.exit_code() == 0
    assert _wait_until(lambda: job.census() == 0, 2.0)   # the job drops a process a few ms after its exit code shows
    report = job.finish()
    assert report == FinishReport(census=0, leak=False, survivors=0, killed=False)
    assert job.leaks == 0


@winonly
def test_real_orphans_after_the_agent_exits_are_a_leak_and_get_killed(tmp_path):
    sink = []
    out = tmp_path / "spawner.out"
    job = JobObject(Win32JobApi(), sink=sink.append, sleep=time.sleep)
    with open(out, "w") as fh:
        job.start([sys.executable, "-c", SPAWNER, "3", "31", "nowait"], tmp_path, None, stdout=fh)
    assert _wait_until(lambda: job.exit_code() is not None)
    pids = [int(x) for x in out.read_text().split()[2:]]
    assert len(pids) == 3
    spawner_pid = job.pid
    report = job.finish()
    assert report == FinishReport(census=3, leak=True, survivors=0, killed=True)
    assert [(e["event"], e["pid"], e["census"], len(e["pids"])) for e in sink] == [("leak", spawner_pid, 3, 3)]
    assert sorted(sink[0]["pids"]) == sorted(pids)
    assert _wait_until(lambda: not any(_alive(p) for p in pids), 10.0)


@winonly
def test_real_close_without_kill_leaves_no_survivors(tmp_path):
    job, ok, failed, pids = _run_spawner(tmp_path, 40, 31, "nowait")
    assert ok == 40 and failed == 0
    assert all(_alive(p) for p in pids)
    job.close()                                   # no terminate: only KILL_ON_JOB_CLOSE can do this
    assert _wait_until(lambda: not any(_alive(p) for p in pids), 5.0)
    assert [p for p in pids if _alive(p)] == []


@winonly
def test_real_cpu_accounting_sees_a_busy_child_but_not_an_idle_one(tmp_path):
    busy = "import time\nt = time.process_time()\nwhile time.process_time() - t < 0.6:\n    pass\n"
    job = JobObject(Win32JobApi(), sleep=time.sleep)
    job.start([sys.executable, "-c", busy], tmp_path)
    assert _wait_until(lambda: job.exit_code() is not None)
    busy_cpu = job.cpu_seconds()
    job.finish()
    idle = JobObject(Win32JobApi(), sleep=time.sleep)
    idle.start([sys.executable, "-S", "-c", "import time; time.sleep(1.0)"], tmp_path)
    assert _wait_until(lambda: idle.exit_code() is not None)
    idle_cpu = idle.cpu_seconds()
    idle.finish()
    assert busy_cpu >= 0.5
    assert idle_cpu < 0.5


@winonly
def test_real_exit_code_of_a_finished_child_is_reported(tmp_path):
    job = JobObject(Win32JobApi(), sleep=time.sleep)
    job.start([sys.executable, "-S", "-c", "raise SystemExit(7)"], tmp_path)
    assert _wait_until(lambda: job.exit_code() is not None)
    assert job.exit_code() == 7
    job.finish()


@winonly
def test_real_agentd_end_to_end_runs_a_process_in_a_job_and_writes_a_summary(tmp_path, clean_repo):
    cfg = ad.AgentdConfig(worktrees_root=tmp_path / "wt", home=tmp_path / "home")
    env = make(tmp_path, config=cfg, git=ad.default_git, untracked_counter=ad.count_untracked,
               job_factory=lambda: JobObject(Win32JobApi(), sleep=time.sleep), clock=time.monotonic)
    script = "import pathlib; pathlib.Path('made.txt').write_text('ok')"
    run = env.agent.start(req(tmp_path, repo=clean_repo, cwd=clean_repo, argv=[sys.executable, "-S", "-c", script]))
    assert _wait_until(lambda: run.job.exit_code() is not None)
    summary = env.agent.finalize(run)
    assert summary["stop_code"] == "completed"
    assert summary["leak"] is False
    assert (run.worktree / "made.txt").read_text() == "ok"


def test_win32_api_is_refused_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(JobError, match=r"Win32JobApi needs Windows"):
        Win32JobApi()


def test_building_a_job_object_makes_no_os_call():
    api = RecordingApi()
    JobObject(api, JobLimits())
    assert api.calls == []
