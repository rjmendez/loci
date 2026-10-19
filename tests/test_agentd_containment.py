"""loci/agentd.py monitor + loci/agentd_audit.py + loci/agentd_token_win.py: containment (S11).

Everything is driven by fake clocks, recording stubs for the volume / commit / process listings / token API,
and tmp_path. The two tests that touch real OS state read the service and Run-key listings on Windows (read
only). NO test creates a restricted token, an ACL or a Job Object: the real restricted-token test the slice
asks for is therefore NOT here, and the docs say path rules are DETECTION until it exists and passes.

Mutation log (each applied to the production code; the named tests went red; then reverted):
- worktree cap 3 GiB -> 3.1 GiB: test_worktree_budget_boundary
- volume floor 8 GB "<" -> "<=": test_volume_floor_boundary
- the persistence listing skipped in final_check: test_persistence_change_fails_the_run, test_no_persistence_change_passes_and_lists_twice
- the deny ACE not added: test_prepare_denies_every_existing_protected_path_then_grants_the_worktree, test_prepare_denies_with_the_restricted_sid_and_full_rights
- agentd reading only the sampler (own commit read dropped): test_agentd_reads_commit_itself_when_the_sampler_is_dead
- diff caps ">" -> ">=": test_diff_line_cap_boundary, test_diff_file_cap_boundary, test_diff_binary_cap_boundary
- check_write prefix compare instead of commonpath: test_a_sibling_directory_with_the_same_prefix_is_outside
- fingerprint without digest: test_same_size_rewrite_with_restored_mtime_is_detected
- quick_check removed from the monitor: test_modifying_envelope_toml_is_a_protected_path_hard_stop
- audit override dropped in finalize: test_a_violation_outranks_how_the_run_ended
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from loci import agentd as ad
from loci import agentd_audit as au
from loci import agentd_token_win as tw
from loci.agentd_job_win import SpawnedProcess

GIB = 1024 ** 3
WIN = sys.platform == "win32"


def gb(text):
    return int(Decimal(text) * GIB)


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


class FakeJob:
    def __init__(self):
        self.exit = None
        self.cpu = 0.0
        self.killed = 0
        self.report = SimpleNamespace(census=0, leak=False, survivors=0, killed=False)

    def start(self, argv, cwd, env=None, *, spawner=None, stdout=None, stderr=None):
        self.started = (list(argv), cwd, spawner)

    def exit_code(self):
        return self.exit

    def cpu_seconds(self):
        return self.cpu

    def kill(self, exit_code=1):
        self.killed += 1

    def finish(self):
        return self.report


def make(tmp_path, *, level=0, commit=10.0, free=50.0, size=0, auditor=None, git=None, config=None,
         containment=None, **overrides):
    clock, wall = Clock(1000.0), Clock(86400.0 * 100)
    box = SimpleNamespace(level=level, commit=commit, free=free, size=size, free_calls=[], size_calls=[],
                          commit_calls=0, sampler_dead=False, total=0)
    jobs = []
    git = git if git is not None else RecordingGit()
    ids = iter(f"r{i}" for i in range(1, 50))

    def governor():
        if box.sampler_dead:
            raise OSError("vitals sampler is dead")
        return box.level

    def commit_reader():
        box.commit_calls += 1
        if isinstance(box.commit, Exception):
            raise box.commit
        return box.commit

    def free_reader(path):
        box.free_calls.append(os.fspath(path))
        if isinstance(box.free, Exception):
            raise box.free
        return box.free

    def dir_size(path):
        box.size_calls.append(os.fspath(path))
        if os.fspath(path) == os.fspath(tmp_path / "wt"):
            return box.total
        if isinstance(box.size, Exception):
            raise box.size
        return box.size

    def job_factory():
        jobs.append(FakeJob())
        return jobs[-1]

    cfg = config or ad.AgentdConfig(worktrees_root=tmp_path / "wt", home=tmp_path / "home")
    kwargs = dict(job_factory=job_factory, governor_level=governor, clock=clock, wall_clock=wall,
                  day_of=lambda ts: str(int(ts // 86400)), enabled=lambda: True, git=git,
                  untracked_counter=lambda repo, limit: 0, free_gb_reader=free_reader, dir_size_reader=dir_size,
                  store=ad.RunStore(tmp_path / "runs.jsonl"), containment=containment, auditor=auditor,
                  commit_reader=commit_reader, id_factory=lambda: next(ids))
    kwargs.update(overrides)
    return SimpleNamespace(agent=ad.Agentd(cfg, **kwargs), clock=clock, wall=wall, box=box, jobs=jobs, git=git,
                           cfg=cfg)


def req(tmp_path, **kw):
    repo = tmp_path / "repo"
    base = dict(backend="claude_code", argv=["claude", "-p", "task"], cwd=repo, repo=repo)
    base.update(kw)
    return ad.StartRequest(**base)


def beat(env, run, seconds):
    """Advance the clock, give the agent CPU progress (so the deadman is quiet), run one supervision step."""
    env.clock.advance(seconds)
    run.job.cpu += 1.0
    return env.agent.tick(run)


# ----------------------------------------------------------------------------- the 30 s monitor: worktree and volume

@pytest.mark.parametrize("size,expected", [
    (3 * GIB, None),
    (gb("2.99"), None),
    (gb("3.01"), "worktree_budget"),
    (3 * GIB + 1, "worktree_budget"),
])
def test_worktree_budget_boundary(tmp_path, size, expected):
    env = make(tmp_path, size=size)
    run = env.agent.start(req(tmp_path))
    assert beat(env, run, 30) == expected
    assert run.job.killed == (1 if expected else 0)
    assert env.box.size_calls[-1] == os.fspath(run.worktree)


@pytest.mark.parametrize("free,expected", [
    (7.9, "disk_floor"),
    (7.999, "disk_floor"),
    (8.0, None),
    (8.1, None),
])
def test_volume_floor_boundary(tmp_path, free, expected):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path))
    env.box.free = free
    assert beat(env, run, 30) == expected
    assert run.job.killed == (1 if expected else 0)
    assert env.box.free_calls[-1] == os.fspath(tmp_path / "wt")


def test_the_monitor_runs_every_30_seconds_and_not_before(tmp_path):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path))
    start_calls = (len(env.box.free_calls), env.box.commit_calls, len(env.box.size_calls))
    assert beat(env, run, 29) is None
    assert (len(env.box.free_calls), env.box.commit_calls, len(env.box.size_calls)) == start_calls
    assert beat(env, run, 1) is None
    assert (len(env.box.free_calls), env.box.commit_calls, len(env.box.size_calls)) == (
        start_calls[0] + 1, 1, start_calls[2] + 1)
    assert beat(env, run, 29) is None
    assert env.box.commit_calls == 1
    assert beat(env, run, 1) is None
    assert env.box.commit_calls == 2


def test_the_monitor_is_not_blocked_by_a_healthy_start(tmp_path):
    env = make(tmp_path, commit=93.9)
    run = env.agent.start(req(tmp_path))
    env.box.free, env.box.size = 8.0, 3 * GIB
    assert [beat(env, run, 30) for _ in range(3)] == [None, None, None]
    assert run.job.killed == 0


def test_killed_run_keeps_its_diff_in_a_real_worktree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.txt").write_text("one\ntwo\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "init")
    env = make(tmp_path, git=ad.default_git)
    run = env.agent.start(req(tmp_path, repo=repo, cwd=repo))
    (run.worktree / "a.txt").write_text("one\nTWO\nthree\n")
    (run.worktree / "new.txt").write_text("fresh\n")
    env.box.size = gb("3.01")
    assert beat(env, run, 30) == "worktree_budget"
    summary = env.agent.finalize(run)
    assert summary["stop_code"] == "worktree_budget"
    assert summary["worktree_kept"] is True
    assert (run.worktree / "a.txt").read_text() == "one\nTWO\nthree\n"
    status = subprocess.run(["git", "status", "--porcelain"], cwd=run.worktree, capture_output=True, text=True,
                            check=True).stdout.splitlines()
    assert sorted(status) == [" M a.txt", "?? new.txt"]


def test_worktrees_total_cap_is_6_gib(tmp_path):
    env = make(tmp_path)
    env.box.total = 6 * GIB
    assert env.agent.start(req(tmp_path)).run_id == "r1"
    env.box.total = 6 * GIB + 1
    with pytest.raises(ad.StartRefused, match=rf"^worktrees_total_budget: worktrees hold {6 * GIB + 1} bytes"):
        env.agent.start(req(tmp_path))


def test_an_unreadable_worktree_size_is_reported_not_ignored(tmp_path):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path))
    env.box.size = PermissionError("denied")
    assert beat(env, run, 30) is None
    assert env.agent.stats()["degraded"] == {"worktree_size_error": 1}


# ----------------------------------------------------------------------------- agentd evaluates commit and disk itself

def test_agentd_reads_commit_itself_when_the_sampler_is_dead(tmp_path):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path))
    env.box.sampler_dead = True
    env.box.commit = 94.0
    assert beat(env, run, 29) is None
    assert run.job.killed == 0
    assert beat(env, run, 1) == "governor_l4"
    assert run.job.killed == 1
    assert env.agent.stats()["degraded"] == {"sampler_dead": 1}


def test_a_dead_sampler_with_commit_just_under_94_does_not_kill(tmp_path):
    env = make(tmp_path, commit=93.9)
    run = env.agent.start(req(tmp_path))
    env.box.sampler_dead = True
    assert beat(env, run, 30) is None
    assert run.job.killed == 0
    assert env.agent.stats()["degraded"] == {"sampler_dead": 1}


def test_the_sampler_saying_l4_kills_even_when_agentds_own_readings_are_fine(tmp_path):
    env = make(tmp_path, commit=10.0)
    run = env.agent.start(req(tmp_path))
    env.box.level = 4
    assert beat(env, run, 30) == "governor_l4"


def test_the_sampler_saying_l3_does_not_kill_a_running_agent(tmp_path):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path))
    env.box.level = 3
    assert beat(env, run, 30) is None


def test_a_failing_commit_reader_is_reported_and_the_sampler_still_counts(tmp_path):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path))
    env.box.commit = OSError("perf counter unavailable")
    assert beat(env, run, 30) is None
    assert env.agent.stats()["degraded"] == {"commit_read_error": 1}
    env.box.level = 4
    assert beat(env, run, 30) == "governor_l4"


def test_blind_for_three_ticks_kills_and_two_do_not(tmp_path):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path))
    env.box.sampler_dead = True
    env.box.commit = OSError("x")
    env.box.free = OSError("y")
    assert beat(env, run, 30) is None
    assert beat(env, run, 30) is None
    assert run.job.killed == 0
    assert beat(env, run, 30) == "monitor_blind"
    assert env.agent.stats()["degraded"] == {"sampler_dead": 3, "commit_read_error": 3, "disk_read_error": 3}


def test_a_good_reading_resets_the_blind_counter(tmp_path):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path))
    env.box.sampler_dead = True
    env.box.commit = OSError("x")
    env.box.free = OSError("y")
    assert [beat(env, run, 30) for _ in range(2)] == [None, None]
    env.box.free = 50.0
    assert beat(env, run, 30) is None
    env.box.free = OSError("y")
    assert [beat(env, run, 30) for _ in range(2)] == [None, None]
    assert run.job.killed == 0


# ----------------------------------------------------------------------------- sandbox: writes outside the worktree

def test_write_inside_the_worktree_passes(tmp_path):
    wt = tmp_path / "wt" / "r1"
    wt.mkdir(parents=True)
    assert au.check_write(wt / "src" / "a.py", wt) is None


def test_write_outside_the_worktree_is_a_sandbox_violation(tmp_path):
    wt = tmp_path / "wt" / "r1"
    wt.mkdir(parents=True)
    with pytest.raises(au.SandboxViolation, match=r"outside worktree: .*elsewhere.txt") as info:
        au.check_write(tmp_path / "elsewhere.txt", wt)
    assert info.value.code == "sandbox_violation"


def test_dotdot_traversal_out_of_the_worktree_is_caught(tmp_path):
    wt = tmp_path / "wt" / "r1"
    wt.mkdir(parents=True)
    with pytest.raises(au.SandboxViolation, match=r"outside worktree"):
        au.check_write(wt / ".." / ".." / "escape.txt", wt)
    assert au.check_write(wt / "a" / ".." / "b.txt", wt) is None


def test_a_sibling_directory_with_the_same_prefix_is_outside(tmp_path):
    wt = tmp_path / "wt" / "r1"
    (tmp_path / "wt" / "r1-evil").mkdir(parents=True)
    wt.mkdir()
    with pytest.raises(au.SandboxViolation, match=r"outside worktree"):
        au.check_write(tmp_path / "wt" / "r1-evil" / "x", wt)


def test_a_write_to_a_protected_path_carries_the_protected_code(tmp_path):
    wt = tmp_path / "wt" / "r1"
    wt.mkdir(parents=True)
    protected = au.default_protected_paths(tmp_path / "home")
    with pytest.raises(au.SandboxViolation, match=r"outside worktree to protected path") as info:
        au.check_write(tmp_path / "home" / ".ssh" / "id_ed25519", wt, protected)
    assert info.value.code == "protected_path"


def test_default_protected_paths_are_exactly_the_decision_3_list(tmp_path):
    home = tmp_path / "home"
    assert au.default_protected_paths(home) == (
        home / ".loci", home / ".ssh", home / ".claude", home / ".hermes", Path("F:/services"))
    assert au.default_protected_paths(home, extra=[tmp_path / "x"])[-1] == tmp_path / "x"


def test_report_write_outside_kills_the_run_and_raises(tmp_path):
    lines = []
    env = make(tmp_path, audit_sink=lines.append, auditor=_auditor(tmp_path))
    run = env.agent.start(req(tmp_path))
    with pytest.raises(au.SandboxViolation, match=r"outside worktree: .*stray.txt"):
        env.agent.report_write(run, tmp_path / "stray.txt")
    assert run.stop_code == "sandbox_violation"
    assert run.job.killed == 1
    assert [json.loads(line)["code"] for line in lines] == ["sandbox_violation"]


def test_report_write_inside_is_fine(tmp_path):
    env = make(tmp_path, auditor=_auditor(tmp_path))
    run = env.agent.start(req(tmp_path))
    assert env.agent.report_write(run, run.worktree / "src" / "a.py") is None
    assert run.stop_code is None
    assert run.job.killed == 0


def test_report_write_to_a_protected_path_is_a_protected_path_stop(tmp_path):
    env = make(tmp_path, auditor=_auditor(tmp_path))
    run = env.agent.start(req(tmp_path))
    with pytest.raises(au.SandboxViolation, match=r"protected path"):
        env.agent.report_write(run, tmp_path / "home" / ".loci" / "envelope.toml")
    assert run.stop_code == "protected_path"


# ----------------------------------------------------------------------------- protected paths: snapshots, envelope.toml

def _protected(home):
    """The decision-3 list WITHOUT F:/services, which on a real machine is a huge live tree no test may walk."""
    return au.default_protected_paths(home)[:-1]


def _home(tmp_path):
    home = tmp_path / "home"
    (home / ".loci").mkdir(parents=True, exist_ok=True)
    (home / ".ssh").mkdir(exist_ok=True)
    (home / ".loci" / "envelope.toml").write_text('[governor]\nhash = "abc"\n')
    (home / ".ssh" / "config").write_text("Host x\n")
    return home


class Listings:
    """A recording stand-in for the scheduled-task / service / Run-key / startup listings."""

    def __init__(self):
        self.state = {kind: set() for kind in au.PERSISTENCE_KINDS}
        self.calls = []

    def listers(self):
        def make_one(kind):
            def lister():
                self.calls.append(kind)
                value = self.state[kind]
                if isinstance(value, Exception):
                    raise value
                return set(value)
            return lister
        return {kind: make_one(kind) for kind in au.PERSISTENCE_KINDS}


def _auditor(tmp_path, listings=None, git=None, **kw):
    home = _home(tmp_path)
    listings = listings if listings is not None else Listings()
    return au.RunAuditor(protected=_protected(home),
                         critical_files=[home / ".loci" / "envelope.toml"], listers=listings.listers(),
                         git=git if git is not None else RecordingGit(), **kw)


def test_modifying_envelope_toml_is_a_protected_path_hard_stop(tmp_path):
    lines = []
    env = make(tmp_path, auditor=_auditor(tmp_path), audit_sink=lines.append)
    run = env.agent.start(req(tmp_path))
    envelope = tmp_path / "home" / ".loci" / "envelope.toml"
    envelope.write_text('[governor]\nhash = "evil"\n')
    assert beat(env, run, 30) == "protected_path"
    assert run.job.killed == 1
    entry = json.loads(lines[0])
    assert (entry["code"], entry["run_id"]) == ("protected_path", "r1")
    assert entry["detail"] == f"critical file changed: {envelope}"
    summary = env.agent.finalize(run)
    assert summary["stop_code"] == "protected_path"
    assert summary["violations"] == ["protected_path"]


def test_an_untouched_envelope_toml_passes_the_same_tick(tmp_path):
    env = make(tmp_path, auditor=_auditor(tmp_path))
    run = env.agent.start(req(tmp_path))
    assert beat(env, run, 30) is None
    assert env.agent.finalize(run)["violations"] == []


def test_deleting_the_envelope_is_detected_too(tmp_path):
    env = make(tmp_path, auditor=_auditor(tmp_path))
    run = env.agent.start(req(tmp_path))
    (tmp_path / "home" / ".loci" / "envelope.toml").unlink()
    assert beat(env, run, 30) == "protected_path"


def test_same_size_rewrite_with_restored_mtime_is_detected(tmp_path):
    f = tmp_path / "cfg.toml"
    f.write_text("aaaa")
    before = au.fingerprint(f)
    stamp = f.stat().st_mtime_ns
    f.write_text("bbbb")
    os.utime(f, ns=(stamp, stamp))
    after = au.fingerprint(f)
    assert before[:2] == after[:2]
    assert before != after


def test_fingerprint_of_a_missing_file_is_none_and_big_files_are_not_hashed(tmp_path):
    assert au.fingerprint(tmp_path / "nope") is None
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * (au.HASH_MAX_BYTES + 1))
    assert au.fingerprint(big)[2] is None
    small = tmp_path / "small.bin"
    small.write_bytes(b"x" * au.HASH_MAX_BYTES)
    assert len(au.fingerprint(small)[2]) == 64


def test_snapshot_diff_sees_added_removed_and_modified_files_but_not_unrelated_ones(tmp_path):
    home = _home(tmp_path)
    other = tmp_path / "unprotected"
    other.mkdir()
    (other / "x").write_text("1")
    roots = _protected(home)
    before = au.snapshot_tree(roots)
    (home / ".ssh" / "config").write_text("Host changed\n")
    (home / ".ssh" / "new_key").write_text("k")
    (home / ".loci" / "envelope.toml").unlink()
    (other / "x").write_text("2")
    changed = au.diff_snapshots(before, au.snapshot_tree(roots))
    def norm(p):
        return os.path.normcase(os.path.realpath(p))

    assert changed == sorted([norm(home / ".ssh" / "config"), norm(home / ".ssh" / "new_key"),
                              norm(home / ".loci" / "envelope.toml")])


def test_an_unchanged_tree_has_an_empty_diff(tmp_path):
    home = _home(tmp_path)
    roots = _protected(home)
    assert au.diff_snapshots(au.snapshot_tree(roots), au.snapshot_tree(roots)) == []


def test_a_truncated_snapshot_is_reported_not_silent(tmp_path):
    home = _home(tmp_path)
    for i in range(5):
        (home / ".ssh" / f"k{i}").write_text("k")
    env = make(tmp_path, auditor=_auditor(tmp_path, max_entries=3))
    run = env.agent.start(req(tmp_path))
    run.job.exit = 0
    summary = env.agent.finalize(run)
    assert summary["violations"] == []
    assert env.agent.stats()["degraded"]["protected_snapshot_truncated"] == 1


# ----------------------------------------------------------------------------- persistence diff

@pytest.mark.parametrize("kind,item", [
    ("scheduled_tasks", "\\evil"), ("services", "EvilSvc"),
    ("run_keys", "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\x=evil.exe"),
    ("startup_folder", "C:\\Users\\u\\Startup\\evil.lnk"),
])
def test_persistence_change_fails_the_run(tmp_path, kind, item):
    listings = Listings()
    listings.state["services"] = {"EventLog"}
    env = make(tmp_path, auditor=_auditor(tmp_path, listings))
    run = env.agent.start(req(tmp_path))
    listings.state[kind].add(item)
    run.job.exit = 0
    summary = env.agent.finalize(run)
    assert summary["stop_code"] == "persistence_change"
    assert summary["runtime_stop"] == "completed"
    assert summary["violations"] == ["persistence_change"]


def test_the_persistence_detail_names_what_was_added(tmp_path):
    listings = Listings()
    auditor = _auditor(tmp_path, listings)
    baseline = auditor.begin(tmp_path / "wt" / "r1", "deadbeef")
    listings.state["scheduled_tasks"].add("\\evil")
    report = auditor.final_check(baseline)
    assert [v.code for v in report.violations] == ["persistence_change"]
    assert json.loads(report.violations[0].detail) == {"scheduled_tasks": {"added": ["\\evil"], "removed": []}}


def test_a_removed_persistence_entry_is_a_change_too(tmp_path):
    listings = Listings()
    listings.state["run_keys"] = {"HKCU\\Run\\keep=a.exe"}
    auditor = _auditor(tmp_path, listings)
    baseline = auditor.begin(tmp_path / "wt" / "r1", "deadbeef")
    listings.state["run_keys"] = set()
    report = auditor.final_check(baseline)
    assert json.loads(report.violations[0].detail) == {"run_keys": {"added": [], "removed": ["HKCU\\Run\\keep=a.exe"]}}


def test_no_persistence_change_passes_and_lists_twice(tmp_path):
    listings = Listings()
    listings.state["services"] = {"EventLog", "Dnscache"}
    env = make(tmp_path, auditor=_auditor(tmp_path, listings))
    run = env.agent.start(req(tmp_path))
    run.job.exit = 0
    summary = env.agent.finalize(run)
    assert summary["stop_code"] == "completed"
    assert summary["violations"] == []
    assert "runtime_stop" not in summary
    assert sorted(listings.calls) == sorted(list(au.PERSISTENCE_KINDS) * 2)


def test_a_listing_that_fails_after_the_run_is_persistence_unknown(tmp_path):
    listings = Listings()
    auditor = _auditor(tmp_path, listings)
    baseline = auditor.begin(tmp_path / "wt" / "r1", "deadbeef")
    listings.state["scheduled_tasks"] = OSError("schtasks timed out")
    report = auditor.final_check(baseline)
    assert [v.code for v in report.violations] == ["persistence_unknown"]
    assert report.violations[0].detail == "scheduled_tasks listing failed: OSError: schtasks timed out"


def test_a_listing_that_fails_at_start_refuses_the_run_and_discards_the_worktree(tmp_path):
    listings = Listings()
    listings.state["services"] = PermissionError("registry denied")
    env = make(tmp_path, auditor=_auditor(tmp_path, listings))
    with pytest.raises(ad.StartRefused, match=r"^audit_setup_failed: PersistenceListingError: services listing failed"):
        env.agent.start(req(tmp_path))
    assert [c[0][:2] for c in env.git.calls[2:]] == [["worktree", "remove"], ["branch", "-D"]]
    assert not hasattr(env.jobs[0], "started")


def test_no_listers_is_reported_as_degraded(tmp_path):
    home = _home(tmp_path)
    auditor = au.RunAuditor(protected=_protected(home), listers={}, git=RecordingGit())
    env = make(tmp_path, auditor=auditor)
    run = env.agent.start(req(tmp_path))
    run.job.exit = 0
    assert env.agent.finalize(run)["violations"] == []
    assert env.agent.stats()["degraded"]["persistence_listers_absent"] == 1


def test_parse_schtasks_csv_keeps_names_only_and_skips_headers_and_info_lines():
    text = ('"\\Microsoft\\Windows\\A","3/1/2026 4:00:00 AM","Ready"\r\n'
            '"TaskName","Next Run Time","Status"\r\n'
            '"\\Backup","N/A","Disabled"\r\n'
            'INFO: There are no scheduled tasks presently available at your access level.\r\n')
    assert au.parse_schtasks_csv(text) == frozenset({"\\Microsoft\\Windows\\A", "\\Backup"})


def test_scheduled_task_lister_asks_schtasks_for_csv_without_a_header():
    asked = []

    def run(argv):
        asked.append(list(argv))
        return '"\\One","N/A","Ready"\n'

    assert au.list_scheduled_tasks(run) == frozenset({"\\One"})
    assert asked == [["schtasks", "/query", "/fo", "csv", "/nh"]]


def test_startup_folder_lister_lists_files_in_both_startup_folders(tmp_path):
    app, prog = tmp_path / "AppData", tmp_path / "ProgramData"
    (app / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup").mkdir(parents=True)
    (app / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / "a.lnk").write_text("")
    (prog / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "StartUp").mkdir(parents=True)
    (prog / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "StartUp" / "b.lnk").write_text("")
    found = au.list_startup_folder({"APPDATA": str(app), "PROGRAMDATA": str(prog)})
    assert sorted(os.path.basename(p) for p in found) == ["a.lnk", "b.lnk"]
    assert au.list_startup_folder({}) == frozenset()


def test_default_listers_cover_the_four_kinds_on_windows_and_none_elsewhere():
    listers = au.default_listers()
    if WIN:
        assert sorted(listers) == sorted(au.PERSISTENCE_KINDS)
    else:
        assert listers == {}


@pytest.mark.skipif(not WIN, reason="registry listings are Windows-only; skipped on other platforms")
def test_real_service_and_run_key_listings_on_windows():
    services = au.list_services()
    assert "EventLog" in services
    run_keys = au.list_run_keys()
    assert all(entry.startswith(("HKCU\\", "HKLM\\")) and "=" in entry for entry in run_keys)


# ----------------------------------------------------------------------------- diff budget

def _change(path="f.py", added=1, removed=0, binary=False, size=0):
    return au.Change(path, added, removed, binary, size)


def test_diff_line_cap_boundary():
    assert au.audit_diff([_change(added=1500, removed=500)]) is None
    violation = au.audit_diff([_change(added=1501, removed=500)])
    assert violation == au.Violation("diff_budget", "2001 changed lines (cap 2000)")


def test_diff_file_cap_boundary():
    assert au.audit_diff([_change(path=f"f{i}") for i in range(40)]) is None
    violation = au.audit_diff([_change(path=f"f{i}") for i in range(41)])
    assert violation == au.Violation("diff_budget", "41 files (cap 40)")


def test_diff_binary_cap_boundary():
    assert au.audit_diff([_change(path="b.bin", added=0, binary=True, size=1024 * 1024)]) is None
    over = au.audit_diff([_change(path="b.bin", added=0, binary=True, size=1024 * 1024 + 1)])
    assert over == au.Violation("diff_budget", "binary b.bin is 1048577 bytes (cap 1048576)")
    assert au.audit_diff([_change(path="b.bin", added=0, binary=True, size=1_153_434)]).code == "diff_budget"


def test_a_big_binary_does_not_count_as_lines_and_a_big_text_file_is_not_a_binary():
    assert au.audit_diff([_change(path="b.bin", added=0, binary=True, size=1024 * 1024)]) is None
    assert au.audit_diff([_change(path="t.txt", added=10, size=5 * 1024 * 1024)]) is None


def test_all_three_budget_problems_are_named_in_one_violation():
    changes = [_change(path=f"f{i}", added=100) for i in range(41)]
    changes.append(_change(path="b.bin", added=0, binary=True, size=2 * 1024 * 1024))
    violation = au.audit_diff(changes)
    assert violation.detail == ("4100 changed lines (cap 2000); 42 files (cap 40); "
                                "binary b.bin is 2097152 bytes (cap 1048576)")


def test_parse_numstat_rows_and_binary_marker():
    assert au.parse_numstat("3\t1\tsrc/a.py\n-\t-\tlogo.png\nnoise\n0\t12\told.txt\n") == [
        (3, 1, "src/a.py"), (None, None, "logo.png"), (0, 12, "old.txt")]


def _numstat_git(numstat):
    return RecordingGit({("-c", "core.quotepath=off", "diff", "--numstat"): ad.GitResult(0, numstat, "")})


@pytest.mark.parametrize("lines,code", [(2000, []), (2001, ["diff_budget"])])
def test_the_auditor_applies_the_line_cap_to_git_output(tmp_path, lines, code):
    auditor = _auditor(tmp_path, git=_numstat_git(f"{lines}\t0\tbig.py\n"))
    baseline = auditor.begin(tmp_path / "wt" / "r1", "deadbeef")
    assert [v.code for v in auditor.final_check(baseline).violations] == code


@pytest.mark.parametrize("files,code", [(40, []), (41, ["diff_budget"])])
def test_the_auditor_applies_the_file_cap_to_git_output(tmp_path, files, code):
    numstat = "".join(f"1\t0\tf{i}.py\n" for i in range(files))
    auditor = _auditor(tmp_path, git=_numstat_git(numstat))
    baseline = auditor.begin(tmp_path / "wt" / "r1", "deadbeef")
    assert [v.code for v in auditor.final_check(baseline).violations] == code


@pytest.mark.parametrize("size,code", [(1024 * 1024, []), (1024 * 1024 + 1, ["diff_budget"])])
def test_the_auditor_applies_the_binary_cap_to_files_on_disk(tmp_path, size, code):
    wt = tmp_path / "wt" / "r1"
    wt.mkdir(parents=True)
    (wt / "blob.bin").write_bytes(b"\0" * size)
    auditor = _auditor(tmp_path, git=_numstat_git("-\t-\tblob.bin\n"))
    baseline = auditor.begin(wt, "deadbeef")
    assert [v.code for v in auditor.final_check(baseline).violations] == code


def test_a_failing_git_diff_fails_closed(tmp_path):
    git = RecordingGit({("-c",): ad.GitResult(128, "", "fatal: bad object deadbeef")})
    auditor = _auditor(tmp_path, git=git)
    baseline = auditor.begin(tmp_path / "wt" / "r1", "deadbeef")
    violations = auditor.final_check(baseline).violations
    assert violations == [au.Violation("diff_budget", "diff audit failed: AuditError: git diff --numstat failed: "
                                                      "fatal: bad object deadbeef")]


def _git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false",
                    *args], cwd=repo, check=True, capture_output=True)


def test_collect_changes_against_a_real_repo_counts_tracked_committed_and_untracked_changes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.txt").write_bytes(b"one\ntwo\nthree\n")
    (repo / "gone.txt").write_bytes(b"x\ny\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    (repo / "a.txt").write_bytes(b"one\nTWO\nthree\nfour\n")          # +2 -1
    (repo / "gone.txt").unlink()                                      # -2
    (repo / "committed.txt").write_bytes(b"c1\nc2\n")
    _git(repo, "add", "committed.txt")
    _git(repo, "commit", "-q", "-m", "agent commit")                  # +2, committed after base
    (repo / "new.txt").write_bytes(b"n1\nn2\nn3")                       # untracked text, 3 lines, no trailing newline
    (repo / "blob.bin").write_bytes(b"\0" * 2000)                     # untracked binary
    changes = au.collect_changes(ad.default_git, repo, base)
    assert sorted((c.path, c.added, c.removed, c.binary, c.size) for c in changes) == [
        ("a.txt", 2, 1, False, 19),("blob.bin", 0, 0, True, 2000), ("committed.txt", 2, 0, False, 6),
        ("gone.txt", 0, 2, False, 0), ("new.txt", 3, 0, False, 8)]


def test_collect_changes_on_a_clean_worktree_is_empty(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "a.txt").write_text("x\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()
    assert au.collect_changes(ad.default_git, repo, base) == []


# ----------------------------------------------------------------------------- audit entries and run summaries

def test_audit_entry_is_one_json_line_under_1_kib():
    line = au.audit_entry("r1", au.Violation("protected_path", "x" * 5000), 12.5)
    assert line.endswith("\n")
    assert len(line.encode("utf-8")) <= 1024
    entry = json.loads(line)
    assert (entry["run_id"], entry["code"], entry["ts"]) == ("r1", "protected_path", 12.5)
    assert entry["detail"] == "x" * len(entry["detail"]) and 900 < len(entry["detail"]) < 1024


def test_a_short_audit_entry_is_unchanged():
    line = au.audit_entry("r1", au.Violation("diff_budget", "41 files (cap 40)"), 3.0)
    assert json.loads(line) == {"run_id": "r1", "code": "diff_budget", "detail": "41 files (cap 40)", "ts": 3.0}


def test_multibyte_detail_is_cut_on_a_character_boundary():
    line = au.audit_entry("r1", au.Violation("protected_path", "é" * 2000), 1.0)
    assert len(line.encode("utf-8")) <= 1024
    assert set(json.loads(line)["detail"]) == {"é"}


def test_a_quote_heavy_detail_still_yields_one_valid_line_under_1_kib():
    detail = ('"' + chr(92)) * 1500
    line = au.audit_entry("r1", au.Violation("diff_budget", detail), 1.0)
    assert len(line.encode("utf-8")) <= 1024
    assert line.count(chr(10)) == 1
    assert json.loads(line)["code"] == "diff_budget"


def test_a_violation_outranks_how_the_run_ended(tmp_path):
    listings = Listings()
    numstat = "".join(f"1\t0\tf{i}.py\n" for i in range(41))
    env = make(tmp_path, auditor=_auditor(tmp_path, listings, git=_numstat_git(numstat)))
    run = env.agent.start(req(tmp_path))
    listings.state["services"].add("EvilSvc")
    (tmp_path / "home" / ".ssh" / "config").write_text("changed\n")
    beat(env, run, 1200)
    summary = env.agent.finalize(run)
    assert summary["runtime_stop"] == "wall_limit"
    assert summary["stop_code"] == "protected_path"
    assert summary["violations"] == ["protected_path", "persistence_change", "diff_budget"]
    assert env.agent.stats()["stops"]["protected_path"] == 1


def test_the_audit_sink_gets_every_violation_and_a_failing_sink_is_reported(tmp_path):
    def broken(line):
        raise OSError("disk full")

    listings = Listings()
    env = make(tmp_path, auditor=_auditor(tmp_path, listings), audit_sink=broken)
    run = env.agent.start(req(tmp_path))
    listings.state["services"].add("EvilSvc")
    run.job.exit = 0
    assert env.agent.finalize(run)["stop_code"] == "persistence_change"
    assert env.agent.stats()["degraded"] == {"audit_write_error": 1}


def test_an_auditor_that_blows_up_is_a_violation_not_a_pass(tmp_path):
    class Exploding(au.RunAuditor):
        def final_check(self, baseline):
            raise RuntimeError("audit crashed")

    home = _home(tmp_path)
    env = make(tmp_path, auditor=Exploding(protected=_protected(home), listers={}, git=RecordingGit()))
    run = env.agent.start(req(tmp_path))
    run.job.exit = 0
    summary = env.agent.finalize(run)
    assert summary["stop_code"] == "diff_budget"
    assert summary["violations"] == ["diff_budget"]


def test_without_an_auditor_the_s10_summary_is_unchanged(tmp_path):
    env = make(tmp_path)
    run = env.agent.start(req(tmp_path))
    run.job.exit = 0
    assert sorted(env.agent.finalize(run)) == sorted([
        "event", "run_id", "backend", "mode", "wall_s", "turns", "stop_code", "leak", "survivors", "day", "ts",
        "worktree_kept"])


# ----------------------------------------------------------------------------- restricted token (recording API, no real tokens)

class RecordingTokenApi(tw.TokenOsApi):
    def __init__(self):
        self.calls = []
        self.fail_on = None
        self.fail_remove = set()
        self.next_token = 100

    def _rec(self, name, *args):
        self.calls.append((name, *args))
        if self.fail_on == (name, args[0] if args else None) or self.fail_on == name:
            raise OSError(f"{name} refused")

    def add_deny_ace(self, path, sid, rights):
        self._rec("deny", path, sid, rights)

    def add_allow_ace(self, path, sid, rights):
        self._rec("allow", path, sid, rights)

    def remove_ace(self, path, sid):
        self.calls.append(("remove", path, sid))
        if path in self.fail_remove:
            raise OSError("remove refused")

    def current_user_sid(self):
        return "S-1-5-21-1-2-3-1001"

    def create_restricted_token(self, flags, disable_sids, restricting_sids):
        self._rec("token", flags, tuple(disable_sids), tuple(restricting_sids))
        self.next_token += 1
        return self.next_token

    def spawn_suspended_as(self, token, argv, cwd, env, stdout=None, stderr=None):
        self.calls.append(("spawn", token, tuple(argv), os.fspath(cwd)))
        return SpawnedProcess(pid=9, handle=10)

    def close_handle(self, handle):
        self.calls.append(("close", handle))


def _contain(tmp_path, api=None, names=(".loci", ".ssh", ".claude")):
    api = api or RecordingTokenApi()
    protected = []
    for name in names:
        p = tmp_path / "home" / name
        p.mkdir(parents=True, exist_ok=True)
        protected.append(p)
    return api, protected, tw.RestrictedTokenContainment(api, protected)


def test_prepare_denies_every_existing_protected_path_then_grants_the_worktree(tmp_path):
    api, protected, c = _contain(tmp_path)
    wt = tmp_path / "wt" / "r1"
    c.prepare(wt)
    assert api.calls == [
        *[("deny", os.fspath(p), "S-1-5-12", "F") for p in protected],
        ("allow", os.fspath(wt), "S-1-5-12", "M"),
        ("token", 1, ("S-1-5-21-1-2-3-1001", "S-1-5-32-544"), ("S-1-1-0", "S-1-5-32-545", "S-1-5-12")),
    ]


def test_prepare_denies_with_the_restricted_sid_and_full_rights(tmp_path):
    api, protected, c = _contain(tmp_path)
    c.prepare(tmp_path / "wt" / "r1")
    denies = [call for call in api.calls if call[0] == "deny"]
    assert len(denies) == 3
    assert {(d[2], d[3]) for d in denies} == {("S-1-5-12", "F")}
    assert tw.RESTRICTED_SID == "S-1-5-12" and tw.DENY_RIGHTS == "F"


def test_a_missing_protected_path_is_skipped_and_recorded(tmp_path):
    api = RecordingTokenApi()
    present = tmp_path / "home" / ".loci"
    present.mkdir(parents=True)
    c = tw.RestrictedTokenContainment(api, [present, tmp_path / "home" / ".hermes"])
    c.prepare(tmp_path / "wt" / "r1")
    assert [call[1] for call in api.calls if call[0] == "deny"] == [os.fspath(present)]
    assert c.skipped == [os.fspath(tmp_path / "home" / ".hermes")]


def test_a_failure_midway_removes_what_was_added_in_reverse(tmp_path):
    api, protected, c = _contain(tmp_path)
    api.fail_on = ("deny", os.fspath(protected[2]))
    with pytest.raises(tw.ContainmentError, match=r"containment setup failed: OSError: deny refused"):
        c.prepare(tmp_path / "wt" / "r1")
    removes = [call for call in api.calls if call[0] == "remove"]
    assert removes == [("remove", os.fspath(protected[1]), "S-1-5-12"), ("remove", os.fspath(protected[0]), "S-1-5-12")]
    assert not any(call[0] in ("allow", "token") for call in api.calls)


def test_a_token_failure_removes_the_acls_and_leaves_nothing_prepared(tmp_path):
    api, protected, c = _contain(tmp_path)
    api.fail_on = "token"
    wt = tmp_path / "wt" / "r1"
    with pytest.raises(tw.ContainmentError, match=r"containment setup failed: OSError: token refused"):
        c.prepare(wt)
    removed = [call[1] for call in api.calls if call[0] == "remove"]
    assert removed == [os.fspath(p) for p in reversed(protected)] + [os.fspath(wt)]
    assert ("close", 101) not in api.calls
    with pytest.raises(tw.ContainmentError, match=r"no restricted token prepared"):
        c.spawn_suspended(["a"], wt, None)


def test_spawn_goes_through_the_token_prepared_for_that_worktree(tmp_path):
    api, _, c = _contain(tmp_path)
    one, two = tmp_path / "wt" / "r1", tmp_path / "wt" / "r2"
    c.prepare(one)
    c.prepare(two)
    proc = c.spawn_suspended(["agent", "-x"], one, None)
    assert proc == SpawnedProcess(pid=9, handle=10)
    assert ("spawn", 101, ("agent", "-x"), os.fspath(one)) in api.calls
    c.spawn_suspended(["agent"], two, None)
    assert ("spawn", 102, ("agent",), os.fspath(two)) in api.calls


def test_deny_aces_are_shared_between_concurrent_runs_and_removed_with_the_last(tmp_path):
    api, protected, c = _contain(tmp_path)
    one, two = tmp_path / "wt" / "r1", tmp_path / "wt" / "r2"
    c.prepare(one)
    c.prepare(two)
    assert len([call for call in api.calls if call[0] == "deny"]) == 3      # not re-added for the second run
    api.calls.clear()
    c.release(one)
    assert api.calls == [("remove", os.fspath(one), "S-1-5-12"), ("close", 101)]
    api.calls.clear()
    c.release(two)
    assert api.calls == [*[("remove", os.fspath(p), "S-1-5-12") for p in reversed(protected)],
                         ("remove", os.fspath(two), "S-1-5-12"), ("close", 102)]


def test_releasing_a_worktree_that_was_never_prepared_does_nothing(tmp_path):
    api, _, c = _contain(tmp_path)
    c.release(tmp_path / "wt" / "ghost")
    assert api.calls == []


def test_release_keeps_going_when_one_removal_fails_and_counts_it(tmp_path):
    api, protected, c = _contain(tmp_path)
    wt = tmp_path / "wt" / "r1"
    c.prepare(wt)
    api.fail_remove = {os.fspath(protected[1])}
    api.calls.clear()
    c.release(wt)
    assert [call[1] for call in api.calls if call[0] == "remove"] == [
        os.fspath(protected[2]), os.fspath(protected[1]), os.fspath(protected[0]), os.fspath(wt)]
    assert c.release_errors == 1


def test_agentd_runs_an_rw_delegate_through_the_restricted_token(tmp_path):
    api, protected, c = _contain(tmp_path)
    cfg = ad.AgentdConfig(worktrees_root=tmp_path / "wt", home=tmp_path / "home", rw_enabled=True)
    env = make(tmp_path, config=cfg, containment=c)
    run = env.agent.start(req(tmp_path, mode="rw", rw_flag=True))
    assert [call[0] for call in api.calls] == ["deny", "deny", "deny", "allow", "token"]
    assert run.job.started[2] == c.spawn_suspended
    run.job.exit = 0
    env.agent.finalize(run)
    assert [call[0] for call in api.calls][-5:] == ["remove", "remove", "remove", "remove", "close"]


def test_win32_token_api_refuses_to_exist_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(tw.ContainmentError, match=r"Win32TokenApi needs Windows"):
        tw.Win32TokenApi()


def test_the_abstract_token_api_has_no_default_behaviour():
    with pytest.raises(NotImplementedError):
        tw.TokenOsApi().add_deny_ace("p", "s", "F")


# ----------------------------------------------------------------------------- DETECTION, not prevention (docs)

DOC = Path(__file__).resolve().parents[1] / "docs" / "agentd.md"


def test_the_docs_label_path_rules_as_detection_until_the_token_test_passes():
    text = DOC.read_text(encoding="utf-8")
    assert "Path rules are DETECTION, not prevention" in text
    assert "restricted-token test has NOT passed" in text
    assert "rw delegation stays OFF" in text


def test_the_code_labels_path_rules_as_detection_too(tmp_path):
    assert au.PATH_RULES_MODE == "detection"
    assert make(tmp_path).agent.stats()["path_rules"] == "detection"
    assert "DETECTION, not prevention" in au.__doc__


def test_rw_delegation_is_still_off_by_default(tmp_path):
    env = make(tmp_path, containment=_contain(tmp_path)[2])
    with pytest.raises(ad.StartRefused, match=r"^rw_disabled"):
        env.agent.start(req(tmp_path, mode="rw", rw_flag=True))
    assert ad.AgentdConfig(worktrees_root=tmp_path).rw_enabled is False
