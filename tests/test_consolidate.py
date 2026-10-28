"""scripts/consolidate.py: nightly consolidation (Tripwire-Governor S18).

Real files under tmp_path; the filesystem seam is a RECORDING OsFs subclass (free space and tree size are
controllable, every write/rename/remove is recorded); clock and vitals heartbeat are injected. Nothing is
scheduled or wired, and no live store is touched.

Mutation log (each applied to scripts/consolidate.py or loci/drpg.py; the named tests went red; then reverted):
- last_sleep.json written in a `finally` (advanced on partial failure):
  test_a_failure_mid_write_leaves_last_sleep_untouched
- last_sleep.json written before the outputs: test_a_failure_mid_write_leaves_last_sleep_untouched,
  test_last_sleep_is_renamed_last_and_every_write_is_temp_then_rename
- the rows_read == 0 refusal deleted: test_zero_rows_read_is_a_failure_and_does_not_advance_the_heartbeat
- heartbeat alarm ">" -> ">=": test_the_heartbeat_alarm_is_past_36_hours_not_at_36_hours
- the early free-space check deleted: test_below_2_gb_free_aborts_with_no_partial_files (the writer's own check then
  still refuses, but with a different message and after the lock was taken)
- the quota check dropped: test_the_quota_aborts_with_exit_6_and_no_files
- dry-run writes anyway: test_dry_run_is_the_default_and_writes_nothing
- the L0 gate dropped: test_consolidation_runs_at_l0_only
- TTL boundary ">" -> ">=": test_ttl_compaction_keeps_exactly_14_days_and_drops_older
- decay exponent dropped (equal weights): test_the_summary_has_the_exact_decayed_mean
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from loci import drpg
from loci.drpg import OsFs, TrajectoryStore

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "consolidate.py"


def _load():
    spec = importlib.util.spec_from_file_location("consolidate_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cons = _load()
NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc).timestamp()
DAY = 86400.0
HOUR = 3600.0
GIB = 1024 ** 3


@pytest.fixture(autouse=True)
def _fresh_cache():
    drpg._TRAJ_CACHE.clear()
    yield
    drpg._TRAJ_CACHE.clear()


class RecordingFs(OsFs):
    def __init__(self, free=10 ** 13, tree=None, fail_replace_on=None):
        self.ops = []
        self.free = free
        self.tree = tree
        self.fail_replace_on = fail_replace_on

    def free_bytes(self, path):
        return self.free

    def tree_size(self, path):
        return super().tree_size(path) if self.tree is None else self.tree

    def write_temp(self, final, data):
        tmp = super().write_temp(final, data)
        self.ops.append(("write_temp", Path(final).name, tmp.name))
        return tmp

    def replace(self, tmp, final):
        self.ops.append(("replace", Path(tmp).name, Path(final).name))
        if self.fail_replace_on == Path(final).name:
            raise OSError(f"injected failure renaming onto {Path(final).name}")
        super().replace(tmp, final)

    def remove(self, path):
        self.ops.append(("remove", Path(path).name))
        super().remove(path)

    def create_exclusive(self, path, data):
        self.ops.append(("create_exclusive", Path(path).name))
        return super().create_exclusive(path, data)


class Limits:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def set_low_priority(self):
        self.calls.append("priority")
        if self.fail:
            raise OSError("denied")
        return "below_normal"

    def bind_memory_job(self, n):
        self.calls.append(("memory", n))
        return f"job_memory={n}"


def iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None).isoformat() + "Z"


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(Path(root).rglob("*")) if p.is_file()}


class Env:
    def __init__(self, tmp_path, level=0, fs=None):
        self.root = tmp_path
        self.traj = tmp_path / "trajectories"
        self.learn = tmp_path / "learn"
        self.vitals = tmp_path / "vitals_state.json"
        self.vitals.write_text(json.dumps({"ts": NOW, "level": level, "degraded": [], "readings": {}}), encoding="utf-8")
        self.fs = fs or RecordingFs()
        self.traj.mkdir()

    def rows(self, pid, rewards, *, ts0=NOW - HOUR):
        store = TrajectoryStore(self.traj)
        for i, r in enumerate(rewards):
            store.push(pid, {"id": f"{pid}-{i}", "reward": {"r_ret": r}, "_ts": iso(ts0 + i)})

    def run(self, **kw):
        kw.setdefault("clock", lambda: NOW)
        kw.setdefault("fs", self.fs)
        kw.setdefault("min_free_bytes", 0)
        return cons.run(trajectory_dir=self.traj, learn_dir=self.learn, vitals_state=self.vitals, **kw)

    def last_sleep(self):
        return json.loads((self.learn / "last_sleep.json").read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------ success path

def test_the_summary_has_the_exact_decayed_mean(tmp_path):
    env = Env(tmp_path)
    env.rows("p1", [0.0, 1.0])  # newest row scores 1.0 (weight 1), the older 0.0 (weight 0.995)
    code, report = env.run(apply=True)
    assert code == 0
    summary = json.loads((env.learn / "consolidated" / "p1.json").read_text(encoding="utf-8"))
    assert summary == {"policy_id": "p1", "rows": 2, "kept": 2, "expired": 0, "undated": 0,
                       "decayed_mean_score": 0.501253133, "as_of": NOW, "schema": 1}  # 1 / 1.995
    assert env.last_sleep() == {"ts": NOW, "rows_read": 2, "policies": 1, "schema": 1}
    assert report["rows_read"] == 2 and report["last_sleep"] == NOW
    replay = [json.loads(x) for x in (env.learn / "replay" / "p1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in replay] == ["p1-0", "p1-1"]
    assert not (env.learn / "consolidate.lock").exists()


def test_ttl_compaction_keeps_exactly_14_days_and_drops_older(tmp_path):
    env = Env(tmp_path)
    lines = [{"id": "old", "reward": {"r_ret": 1.0}, "_ts": iso(NOW - 14 * DAY - 1)},
             {"id": "edge", "reward": {"r_ret": 1.0}, "_ts": iso(NOW - 14 * DAY)},
             {"id": "undated", "reward": {"r_ret": 1.0}},
             {"id": "new", "reward": {"r_ret": 1.0}, "_ts": iso(NOW)}]
    (env.traj / "p1.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
    assert env.run(apply=True)[0] == 0
    kept = [json.loads(x)["id"] for x in (env.learn / "replay" / "p1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert kept == ["edge", "undated", "new"]
    summary = json.loads((env.learn / "consolidated" / "p1.json").read_text(encoding="utf-8"))
    assert (summary["rows"], summary["kept"], summary["expired"], summary["undated"]) == (4, 3, 1, 1)


def test_only_the_newest_rows_are_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(cons, "KEEP_ROWS", 3)
    env = Env(tmp_path)
    env.rows("p1", [0.1, 0.2, 0.3, 0.4, 0.5])
    assert env.run(apply=True)[0] == 0
    kept = [json.loads(x)["id"] for x in (env.learn / "replay" / "p1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert kept == ["p1-2", "p1-3", "p1-4"]
    assert cons.KEEP_ROWS == 3 and cons.TTL_S == 14 * DAY and cons.DECAY == 0.995


def test_dry_run_is_the_default_and_writes_nothing(tmp_path):
    env = Env(tmp_path)
    env.rows("p1", [0.5, 0.5])
    before = snapshot(env.root)
    code, report = env.run()
    assert code == 0 and report["mode"] == "dry-run" and report["files_planned"] == 3 and report["rows_read"] == 2
    assert snapshot(env.root) == before and not env.learn.exists()
    assert env.fs.ops == []


def test_last_sleep_is_renamed_last_and_every_write_is_temp_then_rename(tmp_path):
    env = Env(tmp_path)
    env.rows("p1", [0.5])
    env.rows("p2", [0.5])
    assert env.run(apply=True)[0] == 0
    writes = [op for op in env.fs.ops if op[0] == "write_temp"]
    renames = [op for op in env.fs.ops if op[0] == "replace"]
    assert [w[1] for w in writes] == ["p1.json", "p1.jsonl", "p2.json", "p2.jsonl", "last_sleep.json"]
    assert [r[2] for r in renames] == ["p1.json", "p1.jsonl", "p2.json", "p2.jsonl", "last_sleep.json"]
    for (_, final, tmp), (_, _, renamed_to) in zip(writes, renames):
        assert tmp != final and tmp.startswith(final + ".") and tmp.endswith(".tmp")
    assert [op[0] for op in env.fs.ops][-3:] == ["write_temp", "replace", "remove"]  # then the lock is released
    assert sorted(p.name for p in env.learn.rglob("*") if p.is_file()) == [
        "last_sleep.json", "p1.json", "p1.jsonl", "p2.json", "p2.jsonl"]


# ------------------------------------------------------------------------------------------- failures

def test_a_failure_mid_write_leaves_last_sleep_untouched(tmp_path):
    env = Env(tmp_path, fs=RecordingFs(fail_replace_on="p2.jsonl"))
    env.rows("p1", [0.5])
    env.rows("p2", [0.5])
    env.learn.mkdir()
    old_heartbeat = json.dumps({"ts": NOW - DAY, "rows_read": 7, "policies": 1, "schema": 1}).encode("utf-8")
    (env.learn / "last_sleep.json").write_bytes(old_heartbeat)
    code, report = env.run(apply=True)
    assert code == cons.EXIT_FAILED == 7
    assert report["failed"] == "OSError: injected failure renaming onto p2.jsonl"
    assert (env.learn / "last_sleep.json").read_bytes() == old_heartbeat
    assert not list(env.learn.rglob("*.tmp"))  # the unrenamed temp file was cleaned up
    assert not (env.learn / "consolidate.lock").exists()
    # twin: the same run with the fault gone advances the heartbeat
    env.fs.fail_replace_on = None
    assert env.run(apply=True)[0] == 0
    assert env.last_sleep()["ts"] == NOW


def test_zero_rows_read_is_a_failure_and_does_not_advance_the_heartbeat(tmp_path):
    env = Env(tmp_path)
    (env.traj / "p1.jsonl").write_text("", encoding="utf-8")  # a policy file with no rows
    code, report = env.run(apply=True)
    assert code == cons.EXIT_NO_ROWS == 3
    assert report["refused"] == "rows_read == 0: nothing to consolidate; the heartbeat is NOT advanced"
    assert report["rows_read"] == 0 and not env.learn.exists()
    # no trajectory dir at all is the same failure, and the dir is not created
    gone = cons.run(trajectory_dir=tmp_path / "absent", learn_dir=env.learn, vitals_state=env.vitals, apply=True,
                    clock=lambda: NOW, fs=env.fs, min_free_bytes=0)
    assert gone[0] == 3 and gone[1]["refused"].startswith("rows_read == 0: trajectory dir") and not (tmp_path / "absent").exists()
    # twin: one row is enough
    env.rows("p2", [0.5])
    assert env.run(apply=True)[0] == 0


def test_below_2_gb_free_aborts_with_no_partial_files(tmp_path):
    env = Env(tmp_path, fs=RecordingFs(free=2 * GIB - 1))
    env.rows("p1", [0.5])
    before = snapshot(env.root)
    code, report = cons.run(trajectory_dir=env.traj, learn_dir=env.learn, vitals_state=env.vitals, apply=True,
                            clock=lambda: NOW, fs=env.fs)  # the real 2 GiB floor
    assert code == cons.EXIT_LOW_DISK == 5
    assert report["refused"] == f"free space is below {2 * GIB} bytes; aborting before any write"
    assert snapshot(env.root) == before and not env.learn.exists()
    assert [op for op in env.fs.ops if op[0] in ("write_temp", "replace", "create_exclusive")] == []
    # twin: plenty of space and the same call succeeds
    env.fs.free = 2 * GIB + 10 ** 6
    assert cons.run(trajectory_dir=env.traj, learn_dir=env.learn, vitals_state=env.vitals, apply=True,
                    clock=lambda: NOW, fs=env.fs)[0] == 0


def test_the_floor_also_covers_the_bytes_about_to_be_written(tmp_path):
    env = Env(tmp_path, fs=RecordingFs(free=2 * GIB))  # passes the early check, but the write would dip below
    env.rows("p1", [0.5])
    code, report = cons.run(trajectory_dir=env.traj, learn_dir=env.learn, vitals_state=env.vitals, apply=True,
                            clock=lambda: NOW, fs=env.fs)
    assert code == 5 and "would leave less than" in report["refused"]
    assert not (env.learn / "last_sleep.json").exists() and not list(env.learn.rglob("*.json"))


def test_the_quota_aborts_with_exit_6_and_no_files(tmp_path):
    env = Env(tmp_path, fs=RecordingFs(tree=256 * 1024 * 1024 - 10))
    env.rows("p1", [0.5])
    code, report = cons.run(trajectory_dir=env.traj, learn_dir=env.learn, vitals_state=env.vitals, apply=True,
                            clock=lambda: NOW, fs=env.fs, min_free_bytes=0)
    assert code == cons.EXIT_QUOTA == 6
    assert "exceed its 268435456 byte quota" in report["refused"]
    assert not list(env.learn.rglob("*.json")) and not list(env.learn.rglob("*.jsonl"))
    env.fs.tree = 0
    assert env.run(apply=True)[0] == 0


def test_consolidation_runs_at_l0_only(tmp_path):
    env = Env(tmp_path, level=1)
    env.rows("p1", [0.5])
    before = snapshot(env.root)
    code, report = env.run(apply=True)
    assert code == cons.EXIT_NOT_L0 == 4 and report["refused"].startswith("vitals ladder is not L0 (status=ok, level=1)")
    assert snapshot(env.root) == before
    env.vitals.write_text(json.dumps({"ts": NOW - 60, "level": 0}), encoding="utf-8")  # stale heartbeat
    assert env.run(apply=True)[0] == 4
    env.vitals.write_text(json.dumps({"ts": NOW, "level": 0}), encoding="utf-8")
    assert env.run(apply=True)[0] == 0


def test_one_job_at_a_time(tmp_path):
    env = Env(tmp_path)
    env.rows("p1", [0.5])
    env.learn.mkdir()
    (env.learn / "consolidate.lock").write_text(json.dumps({"pid": 1, "ts": NOW - 60}), encoding="utf-8")
    code, report = env.run(apply=True)
    assert code == cons.EXIT_BUSY == 8 and "another learner run holds" in report["refused"]
    assert not (env.learn / "last_sleep.json").exists()
    (env.learn / "consolidate.lock").write_text(json.dumps({"pid": 1, "ts": NOW - 26 * 60}), encoding="utf-8")
    assert env.run(apply=True)[0] == 0  # a 26-minute-old lock is stale (limit 25 minutes)


def test_the_ten_minute_wall_clock_aborts_without_advancing(tmp_path):
    env = Env(tmp_path)
    env.rows("p1", [0.5])
    ticks = iter([NOW, NOW, NOW] + [NOW + 601] * 20)
    code, report = env.run(apply=True, clock=lambda: next(ticks))
    assert code == cons.EXIT_TIMEOUT == 9 and report["failed"] == "wall-clock limit of 600s exceeded"
    assert not env.learn.exists()


def test_more_than_64_policies_is_refused_not_truncated(tmp_path):
    env = Env(tmp_path)
    for i in range(65):
        (env.traj / f"p{i:02d}.jsonl").write_text(json.dumps({"id": "x", "reward": {"r_ret": 0.5}}) + "\n", encoding="utf-8")
    code, report = env.run(apply=True)
    assert code == 7 and report["failed"] == "PolicyLimitError: 65 policies; at most 64"
    assert not env.learn.exists()


def test_resource_caps_are_applied_when_asked_and_a_failure_refuses(tmp_path, monkeypatch):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "8")

    env = Env(tmp_path)
    env.rows("p1", [0.5])
    ok = Limits()
    code, report = env.run(apply=True, limits=ok, apply_limits=True)
    assert code == 0 and ok.calls == ["priority", ("memory", 512 * 1024 * 1024)]
    assert report["limits"] == {"priority": "below_normal", "memory": "job_memory=536870912", "threads": 1}
    (env.learn / "last_sleep.json").unlink()
    code2, report2 = env.run(apply=True, limits=Limits(fail=True), apply_limits=True)
    assert code2 == cons.EXIT_LIMITS == 10 and "refusing to run uncapped" in report2["refused"]
    assert not (env.learn / "last_sleep.json").exists()


# -------------------------------------------------------------------------------- the heartbeat alarm

def _beat(learn, ts):
    learn.mkdir(exist_ok=True)
    (learn / "last_sleep.json").write_text(json.dumps({"ts": ts, "rows_read": 1}), encoding="utf-8")


def test_the_heartbeat_alarm_is_past_36_hours_not_at_36_hours(tmp_path):
    learn = tmp_path / "learn"
    assert cons.heartbeat_alarm(learn, NOW) == "heartbeat_missing: no last_sleep.json"
    _beat(learn, NOW - 36 * HOUR)
    assert cons.heartbeat_alarm(learn, NOW) is None  # exactly 36 h is still fine
    _beat(learn, NOW - 36 * HOUR - 1)
    assert cons.heartbeat_alarm(learn, NOW) == "heartbeat_stale: last_sleep.json is 129601s old (limit 129600s)"
    _beat(learn, NOW + 301)
    assert cons.heartbeat_alarm(learn, NOW) == "heartbeat_in_future: last_sleep.json is 301s ahead of the clock"
    _beat(learn, NOW + 300)
    assert cons.heartbeat_alarm(learn, NOW) is None
    (learn / "last_sleep.json").write_text("{nope", encoding="utf-8")
    assert cons.heartbeat_alarm(learn, NOW).startswith("heartbeat_unreadable: JSONDecodeError")
    (learn / "last_sleep.json").write_text(json.dumps({"ts": "yesterday"}), encoding="utf-8")
    assert cons.heartbeat_alarm(learn, NOW) == "heartbeat_unreadable: ValueError: ts 'yesterday'"


def test_check_heartbeat_exits_2_on_the_alarm_and_0_when_fresh(tmp_path, capsys):
    learn = tmp_path / "learn"
    assert cons.main(["--learn-dir", str(learn), "--check-heartbeat"], clock=lambda: NOW) == cons.EXIT_ALARM == 2
    assert json.loads(capsys.readouterr().err) == {"alarm": "heartbeat_missing: no last_sleep.json"}
    _beat(learn, NOW - HOUR)
    assert cons.main(["--learn-dir", str(learn), "--check-heartbeat"], clock=lambda: NOW) == 0
    assert json.loads(capsys.readouterr().out) == {"alarm": None}


def test_stopping_the_job_trips_the_alarm_after_36_hours(tmp_path):
    env = Env(tmp_path)
    env.rows("p1", [0.5])
    assert env.run(apply=True)[0] == 0
    assert cons.heartbeat_alarm(env.learn, NOW + 35 * HOUR) is None  # the job "stops" here: nothing advances it
    assert cons.heartbeat_alarm(env.learn, NOW + 36 * HOUR + 1) is not None


def test_a_dry_run_or_a_refused_run_does_not_refresh_the_heartbeat(tmp_path):
    env = Env(tmp_path)
    env.rows("p1", [0.5])
    assert env.run()[0] == 0  # dry-run
    assert cons.heartbeat_alarm(env.learn, NOW) == "heartbeat_missing: no last_sleep.json"


def test_main_is_dry_run_without_apply_and_refuses_non_c_learn_dirs(tmp_path, capsys, monkeypatch):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "8")
    env = Env(tmp_path)
    env.rows("p1", [0.5])
    base = ["--learn-dir", str(env.learn), "--trajectory-dir", str(env.traj), "--vitals-state", str(env.vitals)]
    before = snapshot(env.root)
    limits = Limits()
    assert cons.main(base, clock=lambda: NOW, fs=env.fs, limits=limits) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "dry-run" and snapshot(env.root) == before
    assert limits.calls == ["priority", ("memory", 536870912)]  # the CLI path applies the caps
    with pytest.raises(SystemExit):
        cons.main(["--learn-dir", str(env.learn)], clock=lambda: NOW)
    assert "--trajectory-dir and --vitals-state are required" in capsys.readouterr().err
    assert cons.main(["--learn-dir", "F:/learn", "--check-heartbeat"]) == cons.EXIT_FAILED
    assert "not on C:" in capsys.readouterr().err
