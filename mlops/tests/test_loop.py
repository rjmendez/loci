"""Contract tests for mlops/loop.py — the unattended nightly MLOps loop.

These specify what the loop must do. They used to be characterization tests
that pinned the behaviour of the day, bugs included (a HOLD counted as a
promotion, dry runs that wrote files, failures that never reached FAILED_STEPS
or ALERTS); those pins were replaced by the contract the code now meets.

No external services are used: every subprocess call, every HTTP probe and
every dynamic import performed by the loop is replaced with an in-process fake.
"""

import json
import os
import shutil
import subprocess
import stat
import time
import sys
import types
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlops.loop as loop  # noqa: E402

# The real step functions, captured before the mainenv fixture stubs them, for
# tests that drive a step through main() rather than a hand-written return.
REAL_RUN_MONITOR = loop._run_monitor
REAL_STEPS = {name: getattr(loop, name) for name in (
    "_rebuild_dataset", "_retrain", "_run_canary", "_run_decay", "_run_monitor",
    "_run_embedding_drift", "_run_sft_bake", "_run_active_learn",
    "_emit_embedding_trigger")}

# Explicit, not whatever the importing shell exported: comparing against the
# import-time loop.DEFAULT_OLLAMA made these tests depend on the environment
# pytest was launched from.
OLLAMA_URL = "http://ollama.test:11434"


# ── helpers ───────────────────────────────────────────────────────────────────

class FakeResult:
    """Stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class Runner:
    """Records subprocess.run calls and replays scripted results.

    Results are matched on the *basename of argv[1]* (the script the loop is
    shelling out to). Anything unmatched gets a returncode-0 result.
    """

    def __init__(self):
        self.calls = []
        self.results = {}
        self.side_effects = {}

    def set(self, script_basename, result):
        self.results[script_basename] = result

    def on_call(self, script_basename, fn):
        self.side_effects[script_basename] = fn

    def on_call_cmd(self, script_basename, fn):
        """Like on_call, but the side effect receives the argv — for a fake child
        that writes where its --out flag points, as the real one does."""
        self.side_effects[script_basename] = ("cmd", fn)

    def __call__(self, cmd, *a, **kw):
        self.calls.append(list(cmd))
        key = os.path.basename(cmd[1]) if len(cmd) > 1 else ""
        if key in self.side_effects:
            fx = self.side_effects[key]
            if isinstance(fx, tuple):
                fx[1](list(cmd))
            else:
                fx()
        res = self.results.get(key, FakeResult(0))
        if isinstance(res, list):
            res = res.pop(0) if res else FakeResult(0)
        return res

    def scripts(self):
        return [os.path.basename(c[1]) for c in self.calls if len(c) > 1]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Redirect every module-level path constant into tmp_path.

    loop.py resolves all of its paths at import time from ``__file__``; the
    functions read those module globals on every call, so patching the globals
    is enough to sandbox the whole module.
    """
    repo = tmp_path / "repo"
    mlops = repo / "mlops"
    grounding = repo / "deep_think_loci" / "grounding"
    for d in (mlops / "grounding", mlops / "embedding", mlops / "finetune",
              mlops / "memory", grounding):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(loop, "REPO", repo)
    monkeypatch.setattr(loop, "MLOPS", mlops)
    monkeypatch.setattr(loop, "GROUNDING_DIR", grounding)
    monkeypatch.setattr(loop, "STATE_FILE", mlops / "loop_state.json")
    monkeypatch.setattr(loop, "HISTORY_FILE", mlops / "loop_history.jsonl")
    monkeypatch.setattr(loop, "CANDIDATE_MODEL", mlops / "grounding" / "candidate.joblib")
    monkeypatch.setattr(loop, "LIVE_MODEL", grounding / "grounding_bleed_clf.joblib")
    monkeypatch.setattr(loop, "DATASET", grounding / "grounding_dataset.jsonl")
    monkeypatch.setattr(loop, "ACTIVE_CANDIDATES", mlops / "grounding" / "active_candidates.jsonl")

    runner = Runner()
    # Patch _run, not subprocess.run: _run streams a live child through Popen, and
    # what these tests are about is what the callers do with the result. _run's own
    # bounding and streaming are covered against real children in
    # test_loop_timeouts.py and test_loop_streaming.py.
    monkeypatch.setattr(loop, "_run", runner)

    # Module-level accumulators. main() clears them, but a test calling a single
    # step does not, and a leaked entry makes the next test assert on the
    # previous one's failure.
    loop.FAILED_STEPS.clear()
    loop.ALERTS.clear()

    saved_path = list(sys.path)
    yield types.SimpleNamespace(
        tmp=tmp_path, repo=repo, mlops=mlops, grounding=grounding, run=runner,
    )
    sys.path[:] = saved_path


def write_dataset(env, n_lines):
    env.grounding.joinpath("grounding_dataset.jsonl").write_text(
        "".join(f'{{"i": {i}}}\n' for i in range(n_lines))
    )


def read_history(env):
    p = env.mlops / "loop_history.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# State I/O
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_STATE = {
    "last_run": None,
    "last_dataset_size": 0,
    "runs_seen": [],
    "last_sft_bake": None,
    "last_embedding_tune": None,
    "total_promotions": 0,
}


def test_load_state_missing_file_returns_exact_default(env):
    assert loop._load_state() == DEFAULT_STATE


def test_load_state_default_has_no_loop_run_or_active_learn_keys(env):
    """The default state deliberately omits total_loop_runs / last_active_learn;
    main() only ever reaches those through .get()."""
    s = loop._load_state()
    assert "total_loop_runs" not in s
    assert "last_active_learn" not in s


def test_load_state_corrupt_json_falls_back_to_default(env):
    (env.mlops / "loop_state.json").write_text("{not json")
    assert loop._load_state() == DEFAULT_STATE


def test_load_state_empty_file_falls_back_to_default(env):
    (env.mlops / "loop_state.json").write_text("")
    assert loop._load_state() == DEFAULT_STATE


def test_load_state_backfills_missing_keys_and_keeps_stored_ones(env):
    """main() indexes the default keys directly; a file from an older schema used
    to come back verbatim and KeyError out of the nightly's first line."""
    (env.mlops / "loop_state.json").write_text('{"total_promotions": 5, "extra": 1}')
    assert loop._load_state() == {**DEFAULT_STATE, "total_promotions": 5, "extra": 1}


def test_load_state_non_object_json_falls_back_to_default(env, capsys):
    (env.mlops / "loop_state.json").write_text("[1, 2, 3]")
    assert loop._load_state() == DEFAULT_STATE
    assert "is not a JSON object" in capsys.readouterr().out


def test_save_state_writes_indent_2_json(env):
    loop._save_state({"a": 1, "b": [2]})
    text = (env.mlops / "loop_state.json").read_text()
    assert text == json.dumps({"a": 1, "b": [2]}, indent=2)
    assert "\n  " in text


def test_save_state_keeps_the_old_state_when_the_write_dies(env):
    """A crash part-way through the save must not leave the state file empty.

    _load_state cannot tell a truncated file from a first run: it swallows the
    JSONDecodeError and returns the defaults, so runs_seen=[] re-discovers every
    historical run as new, last_dataset_size=0 disarms the --min-new-pairs veto,
    and every cadence gate opens at once. The state is written once, at the end
    of the run, so losing it costs the whole night's bookkeeping.
    """
    committed = {"last_dataset_size": 4200, "runs_seen": ["r1", "r2"],
                 "total_promotions": 3}
    loop._save_state(committed)
    assert loop._load_state() == {**DEFAULT_STATE, **committed}

    def half_write_text(self, data, *args, **kwargs):
        with open(self, "w") as fh:
            fh.write(data[: len(data) // 2])
        raise OSError("simulated crash mid-write")

    # Its own context: the `env` fixture holds the function-scoped monkeypatch,
    # so undoing that one would put STATE_FILE back to the real repo path.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "write_text", half_write_text)
        with pytest.raises(OSError):
            loop._save_state({"last_dataset_size": 5300,
                              "runs_seen": ["r1", "r2", "r3"],
                              "total_promotions": 4})

    assert loop._load_state() == {**DEFAULT_STATE, **committed}
    assert (env.mlops / "loop_state.json").read_text() == json.dumps(committed, indent=2)


def test_save_state_overwrites_previous_content(env):
    loop._save_state({"a": 1})
    loop._save_state({"b": 2})
    assert json.loads((env.mlops / "loop_state.json").read_text()) == {"b": 2}


def test_append_history_creates_file_and_appends_one_line_per_call(env):
    loop._append_history({"n": 1})
    loop._append_history({"n": 2})
    lines = (env.mlops / "loop_history.jsonl").read_text().splitlines()
    assert [json.loads(l)["n"] for l in lines] == [1, 2]


def test_append_history_creates_a_missing_parent_dir(env, monkeypatch):
    """A missing directory used to raise FileNotFoundError out of the very last
    step of the nightly run."""
    monkeypatch.setattr(loop, "HISTORY_FILE", env.tmp / "nope" / "h.jsonl")
    loop._append_history({"n": 1})
    assert json.loads((env.tmp / "nope" / "h.jsonl").read_text()) == {"n": 1}


def test_append_history_propagates_non_serialisable_record(env):
    with pytest.raises(TypeError):
        loop._append_history({"bad": object()})


# ══════════════════════════════════════════════════════════════════════════════
# Ollama probe
# ══════════════════════════════════════════════════════════════════════════════

def test_ollama_ok_true_and_url_and_timeout(env, monkeypatch):
    seen = {}

    def fake_urlopen(url, timeout=None):
        seen["url"] = url
        seen["timeout"] = timeout
        return object()

    monkeypatch.setattr(loop.urllib.request, "urlopen", fake_urlopen)
    assert loop._ollama_ok("http://h:11434") is True
    assert seen == {"url": "http://h:11434/api/tags", "timeout": 5}


@pytest.mark.parametrize("exc", [
    urllib.error.URLError("down"),
    urllib.error.HTTPError("u", 500, "boom", None, None),
    OSError("refused"),
    ValueError("garbage url"),
])
def test_ollama_ok_false_on_any_exception(env, monkeypatch, exc):
    def boom(url, timeout=None):
        raise exc

    monkeypatch.setattr(loop.urllib.request, "urlopen", boom)
    assert loop._ollama_ok("http://h") is False


def test_ollama_ok_normalises_a_trailing_slash(env, monkeypatch):
    """A caller-supplied --ollama with a trailing slash used to probe
    http://h//api/tags."""
    seen = {}
    monkeypatch.setattr(loop.urllib.request, "urlopen",
                        lambda url, timeout=None: seen.setdefault("url", url))
    assert loop._ollama_ok("http://h/") is True
    assert seen["url"] == "http://h/api/tags"


# ══════════════════════════════════════════════════════════════════════════════
# Run discovery
# ══════════════════════════════════════════════════════════════════════════════

def test_discover_runs_returns_sorted_unseen_parent_dir_names(env):
    for name in ("dt-c", "dt-a", "dt-b"):
        d = env.tmp / "sessions" / name
        d.mkdir(parents=True)
        (d / "findings.jsonl").write_text("{}\n")
    g = str(env.tmp / "sessions" / "*" / "findings.jsonl")
    assert loop._discover_runs(g, []) == ["dt-a", "dt-b", "dt-c"]
    assert loop._discover_runs(g, ["dt-a"]) == ["dt-b", "dt-c"]
    assert loop._discover_runs(g, ["dt-a", "dt-b", "dt-c"]) == []


def test_discover_runs_no_matches_returns_empty_list(env):
    assert loop._discover_runs(str(env.tmp / "nothing" / "*" / "f.jsonl"), []) == []


def test_discover_runs_reports_a_run_id_once(env):
    """Run identity is the parent directory basename, so two same-named session
    dirs under different roots are ONE run. Reporting it twice counted double
    toward --min-new-runs and appended it twice to runs_seen."""
    for root in ("a", "b"):
        d = env.tmp / root / "dt-loci-1"
        d.mkdir(parents=True)
        (d / "findings.jsonl").write_text("{}\n")
    d = env.tmp / "a" / "dt-loci-2"
    d.mkdir(parents=True)
    (d / "findings.jsonl").write_text("{}\n")
    g = str(env.tmp / "*" / "dt-loci-*" / "findings.jsonl")
    assert loop._discover_runs(g, []) == ["dt-loci-1", "dt-loci-2"]


def test_discover_runs_seen_may_be_any_iterable_of_ids(env):
    d = env.tmp / "s" / "dt-1"
    d.mkdir(parents=True)
    (d / "findings.jsonl").write_text("")
    g = str(env.tmp / "s" / "*" / "findings.jsonl")
    assert loop._discover_runs(g, ["dt-1", "dt-1"]) == []


# ══════════════════════════════════════════════════════════════════════════════
# Dataset size
# ══════════════════════════════════════════════════════════════════════════════

def test_current_dataset_size_missing_file_is_zero(env):
    assert loop._current_dataset_size() == 0


def test_current_dataset_size_empty_file_is_zero(env):
    write_dataset(env, 0)
    assert loop._current_dataset_size() == 0


def test_current_dataset_size_counts_lines_including_blanks(env):
    env.grounding.joinpath("grounding_dataset.jsonl").write_text("a\n\nb\n")
    assert loop._current_dataset_size() == 3


def test_current_dataset_size_counts_final_line_without_newline(env):
    env.grounding.joinpath("grounding_dataset.jsonl").write_text("a\nb")
    assert loop._current_dataset_size() == 2


# ══════════════════════════════════════════════════════════════════════════════
# _rebuild_dataset
# ══════════════════════════════════════════════════════════════════════════════

def test_rebuild_dataset_missing_builder_reports_no_rebuild_without_subprocess(env, capsys):
    """No builder means nothing was ingested. Returning the size on disk made that
    indistinguishable from a build that produced it, and the caller marks the runs
    it was supposed to ingest as seen on the strength of that return."""
    write_dataset(env, 4)
    assert loop._rebuild_dataset("g", "http://o") is None
    assert env.run.calls == []
    assert "build_grounding_dataset.py not found" in capsys.readouterr().out


def test_rebuild_dataset_argv_appends_v1_embeddings_to_ollama(env):
    builder = env.repo / "deep_think_loci" / "grounding" / "build_grounding_dataset.py"
    builder.write_text("")
    write_dataset(env, 2)
    loop._rebuild_dataset("GLOB", "http://o:1")
    cmd = env.run.calls[0]
    assert cmd[0] == sys.executable
    assert cmd[1] == str(builder)
    assert cmd[2:] == ["--findings", "GLOB",
                       "--out", str(env.grounding),
                       "--ollama", "http://o:1/v1/embeddings"]


def test_rebuild_dataset_failure_reports_no_rebuild_and_prints_the_exception(env, capsys):
    """The last 500 chars of a traceback land mid-frame, so the log used to read
    "dataset rebuild failed: ^^^^^^^^^^". Report the exception line instead."""
    (env.repo / "deep_think_loci" / "grounding" / "build_grounding_dataset.py").write_text("")
    write_dataset(env, 7)
    tb = ("Traceback (most recent call last):\n"
          '  File "/usr/lib/python3.12/urllib/request.py", line 1347, in do_open\n'
          "    raise URLError(err)\n"
          "    ^^^^^^^^^^^^^^^^^^^\n"
          "urllib.error.URLError: <urlopen error [Errno 111] Connection refused>\n")
    env.run.set("build_grounding_dataset.py", FakeResult(2, stderr=tb))
    assert loop._rebuild_dataset("g", "http://o") is None
    out = capsys.readouterr().out
    assert "dataset rebuild failed" in out
    assert "urllib.error.URLError" in out
    assert "^^^" not in out


def test_rebuild_dataset_success_reports_post_run_size(env, capsys):
    (env.repo / "deep_think_loci" / "grounding" / "build_grounding_dataset.py").write_text("")
    write_dataset(env, 1)
    env.run.on_call("build_grounding_dataset.py", lambda: write_dataset(env, 9))
    assert loop._rebuild_dataset("g", "http://o") == 9
    assert "dataset rebuilt → 9 pairs" in capsys.readouterr().out


# ══════════════════════════════════════════════════════════════════════════════
# _retrain
# ══════════════════════════════════════════════════════════════════════════════

def _metrics_path(env):
    return env.mlops / "grounding" / "train_metrics.json"


def test_retrain_argv_includes_findings_glob_and_dry_run(env):
    loop._retrain("GLOB", "http://o", dry_run=True)
    cmd = env.run.calls[0]
    assert cmd[1] == str(env.mlops / "grounding" / "train.py")
    assert cmd[2:] == ["--dataset", str(env.grounding / "grounding_dataset.jsonl"),
                       "--out", str(_metrics_path(env)),
                       "--ollama", "http://o",
                       "--candidate-out", str(env.mlops / "grounding" / "candidate.joblib"),
                       "--findings-glob", "GLOB",
                       "--dry-run"]


def test_retrain_omits_findings_glob_when_falsy_and_dry_run_when_false(env):
    loop._retrain("", "http://o", dry_run=False)
    cmd = env.run.calls[0]
    assert "--findings-glob" not in cmd
    assert "--dry-run" not in cmd


def test_retrain_returns_none_on_nonzero_exit(env, capsys):
    _metrics_path(env).write_text('{"decision": "PROMOTE"}')
    env.run.set("train.py", FakeResult(3, stderr="bad"))
    assert loop._retrain("g", "o", False) is None
    assert "train.py failed" in capsys.readouterr().out


def test_retrain_returns_none_when_metrics_file_absent(env):
    assert loop._retrain("g", "o", False) is None


def test_retrain_returns_parsed_metrics(env):
    env.run.on_call("train.py",
                    lambda: _metrics_path(env).write_text('{"decision": "PROMOTE", "cv_f1_mean": 0.9}'))
    assert loop._retrain("g", "o", False) == {"decision": "PROMOTE", "cv_f1_mean": 0.9}


def test_retrain_ignores_last_nights_metrics_when_train_writes_nothing(env):
    """train.py exiting 0 without writing train_metrics.json used to make the
    loop re-read LAST night's file and act on its PROMOTE again."""
    _metrics_path(env).write_text('{"decision": "PROMOTE", "stale": true}')
    assert loop._retrain("g", "o", False) is None
    assert loop.FAILED_STEPS == ["train.py"]


def test_retrain_dry_run_writing_no_metrics_is_not_a_failure(env):
    """train.py --dry-run writes no metrics by design."""
    _metrics_path(env).write_text('{"decision": "PROMOTE", "stale": true}')
    assert loop._retrain("g", "o", True) is None
    assert loop.FAILED_STEPS == []


def test_retrain_malformed_metrics_is_a_failed_step_not_a_crash(env):
    """A corrupt metrics file raised JSONDecodeError out of main()."""
    env.run.on_call("train.py", lambda: _metrics_path(env).write_text("{oops"))
    assert loop._retrain("g", "o", False) is None
    assert loop.FAILED_STEPS == ["train.py"]


def test_retrain_does_not_re_print_stdout_after_the_fact(env, capsys):
    """It used to dump the last 1000 chars once train.py had exited, which is the
    one moment the output is no longer useful. _run streams it live instead, so
    printing it again here would only duplicate it."""
    env.run.set("train.py", FakeResult(0, stdout="Y" * 1500))
    loop._retrain("g", "o", False)
    assert capsys.readouterr().out.count("Y") == 0


# ══════════════════════════════════════════════════════════════════════════════
# _run_canary
# ══════════════════════════════════════════════════════════════════════════════

def test_run_canary_without_candidate_returns_none_and_runs_nothing(env, capsys):
    assert loop._run_canary("g", "o", False) is None
    assert env.run.calls == []
    assert "no candidate model to evaluate" in capsys.readouterr().out


def test_run_canary_argv_and_success_shape(env):
    (env.mlops / "grounding" / "candidate.joblib").write_text("m")
    env.run.set("canary.py", FakeResult(0, stdout="all good"))
    out = loop._run_canary("GLOB", "http://o", dry_run=False)
    assert out == {"exit_code": 0, "stdout": "all good"}
    cmd = env.run.calls[0]
    assert cmd[2:] == ["--candidate", str(env.mlops / "grounding" / "candidate.joblib"),
                       "--target", str(env.grounding / "grounding_bleed_clf.joblib"),
                       "--findings", "GLOB",
                       "--ollama", "http://o"]


def test_run_canary_dry_run_flag_appended(env):
    (env.mlops / "grounding" / "candidate.joblib").write_text("m")
    loop._run_canary("g", "o", dry_run=True)
    assert env.run.calls[0][-1] == "--dry-run"


def test_run_canary_drift_exit_1_still_returns_dict_not_none(env, capsys):
    (env.mlops / "grounding" / "candidate.joblib").write_text("m")
    env.run.set("canary.py", FakeResult(1, stdout="drift"))
    assert loop._run_canary("g", "o", False) == {"exit_code": 1, "stdout": "drift"}
    assert "ALERT: canary drift detected" in capsys.readouterr().out


def test_run_canary_exit_2_is_a_rollback_recommendation_not_silence(env, capsys):
    """Was test_run_canary_other_nonzero_exit_has_no_alert, which characterised
    the bug: canary.py exits 2 for ROLLBACK RECOMMENDED -- the loudest thing it
    can say -- and only exit 1 was read, so it went nowhere."""
    (env.mlops / "grounding" / "candidate.joblib").write_text("m")
    env.run.set("canary.py", FakeResult(2, stdout="rollback"))
    assert loop._run_canary("g", "o", False)["exit_code"] == 2
    assert "ROLLBACK" in capsys.readouterr().out
    assert "canary" in loop.ALERTS
    assert "canary" not in loop.FAILED_STEPS, "a rollback recommendation is not a broken step"


def test_run_canary_an_unexpected_exit_is_a_failed_step(env):
    (env.mlops / "grounding" / "candidate.joblib").write_text("m")
    env.run.set("canary.py", FakeResult(4, stderr="Traceback\nValueError: x"))
    loop._run_canary("g", "o", False)
    assert "canary" in loop.FAILED_STEPS


def test_run_canary_hold_exit_is_neither_an_alert_nor_a_failure(env, capsys):
    """HOLD is canary's ordinary 'no' (exit 3). It used to share exit 0 with
    PROMOTE; now it has its own code and is reported as what it is."""
    (env.mlops / "grounding" / "candidate.joblib").write_text("m")
    env.run.set("canary.py", FakeResult(loop.CANARY_HOLD, stdout="hold"))
    assert loop._run_canary("g", "o", False) == {"exit_code": 3, "stdout": "hold"}
    assert "canary HOLD" in capsys.readouterr().out
    assert loop.ALERTS == [] and loop.FAILED_STEPS == []


def test_loop_and_canary_agree_on_the_exit_contract():
    from mlops.grounding import canary as canary_mod
    assert (loop.CANARY_OK, loop.CANARY_DRIFT, loop.CANARY_ROLLBACK, loop.CANARY_HOLD) == (
        canary_mod.EXIT_PROMOTE, canary_mod.EXIT_DRIFT, canary_mod.EXIT_ROLLBACK,
        canary_mod.EXIT_HOLD)
    assert len({0, 1, 2, 3}) == len({canary_mod.EXIT_PROMOTE, canary_mod.EXIT_DRIFT,
                                     canary_mod.EXIT_ROLLBACK, canary_mod.EXIT_HOLD})


def test_run_canary_truncates_stored_stdout_to_500(env):
    (env.mlops / "grounding" / "candidate.joblib").write_text("m")
    env.run.set("canary.py", FakeResult(0, stdout="Z" * 800))
    assert loop._run_canary("g", "o", False)["stdout"] == "Z" * 500


# ══════════════════════════════════════════════════════════════════════════════
# _run_sft_bake
# ══════════════════════════════════════════════════════════════════════════════

def _good_sft(env, nbytes=200):
    (env.mlops / "finetune" / "data").mkdir(parents=True, exist_ok=True)
    (env.mlops / "finetune" / "data" / "sft_pairs.jsonl").write_text("x" * nbytes)


def test_sft_bake_creates_data_dir_and_runs_collect_format_then_bake(env):
    env.run.on_call("format_sft.py", lambda: _good_sft(env))
    assert loop._run_sft_bake("http://o", dry_run=False) is True
    assert (env.mlops / "finetune" / "data").is_dir()
    assert env.run.scripts() == ["collect.py", "format_sft.py", "train_lora.py"]
    assert loop.FAILED_STEPS == []


def test_sft_bake_collect_failure_short_circuits(env, capsys):
    env.run.set("collect.py", FakeResult(1, stderr="nope"))
    assert loop._run_sft_bake("o", False) is False
    assert env.run.scripts() == ["collect.py"]
    assert "SFT step failed" in capsys.readouterr().out


def test_sft_bake_format_failure_returns_false(env):
    env.run.set("format_sft.py", FakeResult(9))
    assert loop._run_sft_bake("o", False) is False
    assert env.run.scripts() == ["collect.py", "format_sft.py"]


def test_sft_bake_missing_pairs_file_returns_false(env, capsys):
    assert loop._run_sft_bake("o", False) is False
    assert "SFT pairs file empty" in capsys.readouterr().out


def test_sft_bake_size_99_bytes_is_too_small(env):
    env.run.on_call("format_sft.py", lambda: _good_sft(env, 99))
    assert loop._run_sft_bake("o", False) is False
    assert "train_lora.py" not in env.run.scripts()


def test_sft_bake_size_exactly_100_bytes_passes_threshold(env):
    env.run.on_call("format_sft.py", lambda: _good_sft(env, 100))
    assert loop._run_sft_bake("o", False) is True
    assert env.run.scripts()[-1] == "train_lora.py"


def test_sft_bake_dry_run_runs_and_writes_nothing(env):
    """collect.py and format_sft.py write into mlops/finetune/data; a dry run
    used to run both and report the step as a success."""
    shutil.rmtree(env.mlops / "finetune")
    assert loop._run_sft_bake("o", dry_run=True) is False
    assert env.run.calls == []
    assert not (env.mlops / "finetune").exists()
    assert loop.FAILED_STEPS == []


def test_sft_bake_argv_of_real_bake(env):
    env.run.on_call("format_sft.py", lambda: _good_sft(env))
    loop._run_sft_bake("http://o", dry_run=False)
    data = env.mlops / "finetune" / "data"
    assert env.run.calls[0][2:] == ["--out", str(data)]
    assert env.run.calls[1][2:] == ["--traces", str(data / "raw_traces.jsonl"),
                                    "--out", str(data / "sft_pairs.jsonl"),
                                    "--mode", "both"]
    assert env.run.calls[2][2:] == ["--sft", str(data / "sft_pairs.jsonl"),
                                    "--backend", "ollama-modelfile"]


def test_sft_bake_failure_is_a_failed_step(env, capsys):
    """A failed bake returned False and nothing else: it never reached
    FAILED_STEPS, so the run exited 0 and said 'done'."""
    env.run.on_call("format_sft.py", lambda: _good_sft(env))
    env.run.set("train_lora.py", FakeResult(1, stderr="Error: model not found"))
    assert loop._run_sft_bake("o", False) is False
    assert loop.FAILED_STEPS == ["SFT bake"]
    assert "train_lora.py bake failed (exit 1): Error: model not found" in capsys.readouterr().out


# ══════════════════════════════════════════════════════════════════════════════
# _run_decay  (dynamic `from memory.decay import apply_decay`)
# ══════════════════════════════════════════════════════════════════════════════

def install_fake(monkeypatch, dotted, **attrs):
    """Register a fake module (and its parent package) in sys.modules."""
    parts = dotted.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    mod = types.ModuleType(dotted)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, dotted, mod)
    if len(parts) > 1:
        monkeypatch.setattr(sys.modules[".".join(parts[:-1])], parts[-1], mod, raising=False)
    return mod


def test_run_decay_returns_stats_and_forwards_kwargs(env, monkeypatch, capsys):
    seen = {}

    def apply_decay(db_path, dry_run):
        seen.update(db_path=db_path, dry_run=dry_run)
        return {"n_rows": 10, "n_decayed": 3, "mean_retention": 0.5}

    install_fake(monkeypatch, "memory.decay", apply_decay=apply_decay)
    out = loop._run_decay("/db.sqlite", dry_run=True)
    assert out == {"n_rows": 10, "n_decayed": 3, "mean_retention": 0.5}
    assert seen == {"db_path": "/db.sqlite", "dry_run": True}
    assert "n_rows=10 n_decayed=3 mean_retention=0.500" in capsys.readouterr().out


def test_run_decay_inserts_mlops_on_sys_path(env, monkeypatch):
    install_fake(monkeypatch, "memory.decay", apply_decay=lambda **k: {})
    loop._run_decay("/db", False)
    assert sys.path[0] == str(env.mlops)


def test_run_decay_swallows_exception_and_returns_empty_dict(env, monkeypatch, capsys):
    def boom(**kw):
        raise RuntimeError("db locked")

    install_fake(monkeypatch, "memory.decay", apply_decay=boom)
    assert loop._run_decay("/db", False) == {}
    assert "decay step failed: db locked" in capsys.readouterr().out


def test_run_decay_import_failure_returns_empty_dict(env, monkeypatch):
    monkeypatch.setitem(sys.modules, "memory", None)
    assert loop._run_decay("/db", False) == {}


def test_run_decay_missing_mean_retention_key_is_reported_as_unknown(env, monkeypatch, capsys):
    """It printed 0.000 -- total loss -- for a value that was simply absent."""
    install_fake(monkeypatch, "memory.decay",
                 apply_decay=lambda **k: {"n_rows": 1, "n_decayed": 0})
    assert loop._run_decay("/db", False) == {"n_rows": 1, "n_decayed": 0}
    assert "mean_retention=n/a" in capsys.readouterr().out


def test_run_decay_null_mean_retention_keeps_the_successful_result(env, monkeypatch, capsys):
    """Formatting a None mean_retention with :.3f used to raise inside the try,
    so a decay that had run was reported as a failure returning {}."""
    stats = {"n_rows": 0, "n_decayed": 0, "mean_retention": None}
    install_fake(monkeypatch, "memory.decay", apply_decay=lambda **k: dict(stats))
    assert loop._run_decay("/db", False) == stats
    assert "mean_retention=n/a" in capsys.readouterr().out
    assert loop.FAILED_STEPS == []


def test_run_decay_error_stub_is_a_failed_step(env, monkeypatch, capsys):
    """apply_decay reports a missing DB as {'error': ...}. That printed
    'n_rows=0 n_decayed=0' and counted as a green step."""
    stub = {"error": "db not found: /nope.db", "n_rows": 0, "n_decayed": 0}
    install_fake(monkeypatch, "memory.decay", apply_decay=lambda **k: dict(stub))
    assert loop._run_decay("/nope.db", True) == stub
    assert loop.FAILED_STEPS == ["decay"]
    assert "decay step failed: db not found: /nope.db" in capsys.readouterr().out


def test_run_decay_on_a_missing_db_is_a_failed_step_with_the_real_decay(env, tmp_path,
                                                                        monkeypatch):
    """Same, through the real apply_decay rather than a stub of it."""
    monkeypatch.setattr(loop, "MLOPS", Path(loop.__file__).resolve().parent)
    monkeypatch.delitem(sys.modules, "memory", raising=False)
    monkeypatch.delitem(sys.modules, "memory.decay", raising=False)
    out = loop._run_decay(str(tmp_path / "absent.db"), True)
    assert out["error"] == f"db not found: {tmp_path / 'absent.db'}"
    assert loop.FAILED_STEPS == ["decay"]


def test_run_decay_non_dict_return_is_a_failed_step(env, monkeypatch):
    install_fake(monkeypatch, "memory.decay", apply_decay=lambda **k: None)
    assert loop._run_decay("/db", False) == {}
    assert loop.FAILED_STEPS == ["decay"]


# ══════════════════════════════════════════════════════════════════════════════
# _run_monitor
# ══════════════════════════════════════════════════════════════════════════════

def test_run_monitor_without_live_model_returns_empty(env, capsys):
    assert loop._run_monitor("g", "o", False) == {}
    assert "monitor skipped — no live model yet" in capsys.readouterr().out


def test_run_monitor_returns_result_and_forwards_kwargs(env, monkeypatch, capsys):
    (env.grounding / "grounding_bleed_clf.joblib").write_text("m")
    seen = {}

    def monitor_live(live_model_path, findings_glob, ollama_url, dry_run):
        seen.update(live_model_path=live_model_path, findings_glob=findings_glob,
                    ollama_url=ollama_url, dry_run=dry_run)
        return {"drift": 0.02, "rollback_recommended": False}

    install_fake(monkeypatch, "mlops.grounding.canary", monitor_live=monitor_live)
    out = loop._run_monitor("GLOB", "http://o", dry_run=True)
    assert out == {"drift": 0.02, "rollback_recommended": False}
    assert seen == {"live_model_path": str(env.grounding / "grounding_bleed_clf.joblib"),
                    "findings_glob": "GLOB", "ollama_url": "http://o", "dry_run": True}
    assert "monitor: drift=0.02 rollback_recommended=False" in capsys.readouterr().out


def test_run_monitor_rollback_recommendation_reaches_alerts(env, monkeypatch, capsys):
    """It used to be a bare print: ALERTS stayed empty, so the summary line and
    loop_history said nothing needed a human."""
    (env.grounding / "grounding_bleed_clf.joblib").write_text("m")
    install_fake(monkeypatch, "mlops.grounding.canary",
                 monitor_live=lambda **k: {"drift": 9, "rollback_recommended": True})
    out = loop._run_monitor("g", "o", False)
    assert out["rollback_recommended"] is True
    assert loop.ALERTS == ["monitor"]
    assert loop.FAILED_STEPS == []
    assert "ALERT: rollback recommended" in capsys.readouterr().out


def test_run_monitor_without_a_rollback_raises_no_alert(env, monkeypatch):
    (env.grounding / "grounding_bleed_clf.joblib").write_text("m")
    install_fake(monkeypatch, "mlops.grounding.canary",
                 monitor_live=lambda **k: {"drift": True, "rollback_recommended": False})
    loop._run_monitor("g", "o", False)
    assert loop.ALERTS == [] and loop.FAILED_STEPS == []


def test_main_monitor_rollback_is_in_the_history_alerts(mainenv, monkeypatch):
    e = mainenv
    (e.grounding / "grounding_bleed_clf.joblib").write_text("m")
    monkeypatch.setattr(loop, "_run_monitor", REAL_RUN_MONITOR)
    install_fake(monkeypatch, "mlops.grounding.canary",
                 monitor_live=lambda **k: {"drift": True, "rollback_recommended": True})
    assert e.main() == 0
    rec = read_history(e)[0]
    assert rec["alerts"] == ["monitor"]
    assert rec["failed_steps"] == []


def test_run_monitor_swallows_exception(env, monkeypatch, capsys):
    (env.grounding / "grounding_bleed_clf.joblib").write_text("m")

    def boom(**kw):
        raise RuntimeError("ollama down")

    install_fake(monkeypatch, "mlops.grounding.canary", monitor_live=boom)
    assert loop._run_monitor("g", "o", False) == {}
    assert "monitor step failed: ollama down" in capsys.readouterr().out


def test_run_monitor_import_error_degrades_to_empty(env, monkeypatch):
    (env.grounding / "grounding_bleed_clf.joblib").write_text("m")
    monkeypatch.setitem(sys.modules, "mlops.grounding.canary", None)
    assert loop._run_monitor("g", "o", False) == {}


def test_run_monitor_does_not_grow_sys_path(env, monkeypatch):
    """It prepended MLOPS/grounding on every call, which could never satisfy the
    absolute ``mlops.grounding.canary`` import it guarded."""
    install_fake(monkeypatch, "mlops.grounding.canary", monitor_live=lambda **k: {})
    (env.grounding / "grounding_bleed_clf.joblib").write_text("m")
    before = list(sys.path)
    loop._run_monitor("g", "o", False)
    assert sys.path == before


# ══════════════════════════════════════════════════════════════════════════════
# _run_embedding_drift
# ══════════════════════════════════════════════════════════════════════════════

def _drift_script(env):
    p = env.mlops / "embedding" / "drift.py"
    p.write_text("")
    return p


def test_embedding_drift_missing_script_returns_empty(env):
    assert loop._run_embedding_drift("o", False) == {}
    assert env.run.calls == []


def test_embedding_drift_builds_anchor_when_absent(env, capsys):
    _drift_script(env)
    assert loop._run_embedding_drift("http://o", False) == {"built_anchor": True}
    cmd = env.run.calls[0]
    assert cmd[2:] == ["--dataset", str(env.grounding / "grounding_dataset.jsonl"),
                       "--ollama", "http://o",
                       "--anchor", str(env.mlops / "embedding" / "anchor.npz"),
                       "--build-anchor"]
    assert "no anchor — building anchor set" in capsys.readouterr().out


def test_embedding_drift_a_failed_anchor_build_is_not_reported_as_built(env):
    """Was ..._reports_built_anchor_even_when_build_fails: the branch never
    inspected the return code, so a failed build claimed an anchor exists."""
    _drift_script(env)
    env.run.set("drift.py", FakeResult(1, stderr="boom"))
    assert loop._run_embedding_drift("o", False) == {"exit_code": 1}
    assert "embedding drift" in loop.FAILED_STEPS


def test_embedding_drift_clean_run_returns_exit_code(env):
    _drift_script(env)
    (env.mlops / "embedding" / "anchor.npz").write_text("a")
    assert loop._run_embedding_drift("o", False) == {"exit_code": 0}


def test_embedding_drift_prefers_result_json_over_exit_code(env):
    _drift_script(env)
    (env.mlops / "embedding" / "anchor.npz").write_text("a")
    env.run.on_call("drift.py", lambda: (env.mlops / "embedding" / "drift_result.json")
                    .write_text('{"drift": 0.4}'))
    assert loop._run_embedding_drift("o", False) == {"drift": 0.4}


def test_embedding_drift_does_not_return_a_stale_result_json(env):
    """Was ..._returns_stale_result_json. A file an earlier run left behind was
    returned verbatim as this run's answer -- last week's drift score reported
    as today's."""
    _drift_script(env)
    (env.mlops / "embedding" / "anchor.npz").write_text("a")
    (env.mlops / "embedding" / "drift_result.json").write_text('{"stale": true}')
    assert loop._run_embedding_drift("o", False) == {"exit_code": 0}


def test_embedding_drift_malformed_result_json_falls_back_to_exit_code(env):
    _drift_script(env)
    (env.mlops / "embedding" / "anchor.npz").write_text("a")
    (env.mlops / "embedding" / "drift_result.json").write_text("{broken")
    env.run.set("drift.py", FakeResult(1))
    assert loop._run_embedding_drift("o", False) == {"exit_code": 1}


def _measured_drift(env, exceeded=True):
    """drift.py writes --out only on the path where it actually measured."""
    def write(cmd):
        Path(cmd[cmd.index("--out") + 1]).write_text(
            '{"exceeded": %s}' % ("true" if exceeded else "false"))
    env.run.on_call_cmd("drift.py", write)


def test_embedding_drift_exit_1_with_a_measurement_emits_contrastive_script(env, capsys):
    _drift_script(env)
    (env.mlops / "embedding" / "anchor.npz").write_text("a")
    env.run.set("drift.py", FakeResult(1))
    _measured_drift(env)
    loop._run_embedding_drift("o", dry_run=False)
    assert (env.mlops / "run_contrastive.sh").exists()
    assert "embedding drift detected" in capsys.readouterr().out


def test_embedding_drift_exit_1_without_a_measurement_is_a_failure_not_drift(env, capsys):
    """drift.py exits 1 for BOTH 'exceeded' and 'could not measure'. It writes
    --out only when it measured, so a down Ollama used to read as drift and
    schedule a fine-tune."""
    _drift_script(env)
    (env.mlops / "embedding" / "anchor.npz").write_text("a")
    env.run.set("drift.py", FakeResult(1, stderr="[drift] ERROR: connection refused"))
    loop._run_embedding_drift("o", dry_run=False)
    out = capsys.readouterr().out
    assert not (env.mlops / "run_contrastive.sh").exists(), "scheduled a tune for an error"
    assert "embedding drift" in loop.FAILED_STEPS
    assert "drift detected" not in out


def test_embedding_drift_dry_run_measures_alerts_and_writes_nothing(env, capsys):
    """A dry run still measures, into a scratch file, and reads the result back.
    It used to write drift_result.json into the repo."""
    _drift_script(env)
    (env.mlops / "embedding" / "anchor.npz").write_text("a")
    env.run.set("drift.py", FakeResult(1))
    _measured_drift(env)
    before = sorted(p.name for p in env.mlops.rglob("*"))
    assert loop._run_embedding_drift("o", dry_run=True) == {"exceeded": True}
    out_arg = Path(env.run.calls[0][env.run.calls[0].index("--out") + 1])
    assert not out_arg.is_relative_to(env.tmp), "the dry run pointed --out into the repo"
    assert not out_arg.exists(), "the scratch result was left behind"
    assert sorted(p.name for p in env.mlops.rglob("*")) == before
    assert loop.ALERTS == ["embedding drift"]
    assert "embedding drift detected" in capsys.readouterr().out


def test_embedding_drift_live_run_writes_its_result_into_the_repo(env):
    _drift_script(env)
    (env.mlops / "embedding" / "anchor.npz").write_text("a")
    _measured_drift(env, exceeded=False)
    assert loop._run_embedding_drift("o", dry_run=False) == {"exceeded": False}
    assert json.loads((env.mlops / "embedding" / "drift_result.json").read_text()) == {
        "exceeded": False}


def test_embedding_drift_dry_run_does_not_build_an_anchor(env, capsys):
    _drift_script(env)
    assert loop._run_embedding_drift("o", dry_run=True) == {}
    assert env.run.calls == []
    assert "not built" in capsys.readouterr().out


def test_embedding_drift_exit_2_does_not_emit(env):
    _drift_script(env)
    (env.mlops / "embedding" / "anchor.npz").write_text("a")
    env.run.set("drift.py", FakeResult(2))
    assert loop._run_embedding_drift("o", False) == {"exit_code": 2}
    assert not (env.mlops / "run_contrastive.sh").exists()


# ══════════════════════════════════════════════════════════════════════════════
# _run_active_learn
# ══════════════════════════════════════════════════════════════════════════════

def test_active_learn_requires_live_model_dataset_and_script(env):
    script = env.mlops / "grounding" / "active_learn.py"
    live = env.grounding / "grounding_bleed_clf.joblib"
    assert loop._run_active_learn("o") == {}          # nothing present
    live.write_text("m")
    assert loop._run_active_learn("o") == {}          # no dataset
    write_dataset(env, 1)
    assert loop._run_active_learn("o") == {}          # no script
    assert env.run.calls == []
    script.write_text("")
    assert loop._run_active_learn("o") == {"exit_code": 0}


def test_active_learn_argv_and_nonzero_exit(env):
    (env.grounding / "grounding_bleed_clf.joblib").write_text("m")
    write_dataset(env, 1)
    (env.mlops / "grounding" / "active_learn.py").write_text("")
    env.run.set("active_learn.py", FakeResult(4))
    assert loop._run_active_learn("http://o") == {"exit_code": 4}
    assert env.run.calls[0][2:] == [
        "--model", str(env.grounding / "grounding_bleed_clf.joblib"),
        "--dataset", str(env.grounding / "grounding_dataset.jsonl"),
        "--out", str(env.mlops / "grounding" / "active_candidates.jsonl"),
        "--ollama", "http://o",
    ]


# ══════════════════════════════════════════════════════════════════════════════
# _emit_embedding_trigger
# ══════════════════════════════════════════════════════════════════════════════

def test_emit_embedding_trigger_content_and_mode(env, capsys):
    loop._emit_embedding_trigger()
    p = env.mlops / "run_contrastive.sh"
    body = p.read_text()
    assert body.startswith("#!/bin/bash\n")
    assert "set -e\n" in body
    assert f"cd {env.repo}\n" in body
    assert f"{sys.executable} mlops/embedding/contrastive.py" in body, (
        "the emitted command must use this interpreter: contrastive.py imports "
        "sentence_transformers, which the system python3 does not have")
    assert "--model-size small" in body
    assert stat.S_IMODE(p.stat().st_mode) == 0o755
    assert f"embedding trigger written to {p}" in capsys.readouterr().out


def test_emit_embedding_trigger_is_idempotent_overwrite(env):
    p = env.mlops / "run_contrastive.sh"
    p.write_text("junk")
    loop._emit_embedding_trigger()
    assert "junk" not in p.read_text()


def test_emit_embedding_trigger_leaves_mtime_alone_when_body_is_unchanged(env):
    """_embedding_tune_ran() dates a fine-tune against this file, so re-emitting
    an identical body must not touch it — otherwise every nightly tick would
    make a completed tune look older than its own trigger."""
    p = env.mlops / "run_contrastive.sh"
    loop._emit_embedding_trigger()
    os.utime(p, (1_000_000, 1_000_000))
    loop._emit_embedding_trigger()
    assert p.stat().st_mtime == 1_000_000
    assert stat.S_IMODE(p.stat().st_mode) == 0o755


# ══════════════════════════════════════════════════════════════════════════════
# _embedding_tune_ran
# ══════════════════════════════════════════════════════════════════════════════

def test_embedding_tune_ran_false_with_no_trigger_and_no_model(env):
    assert loop._embedding_tune_ran() is False


def test_embedding_tune_ran_false_when_model_predates_the_trigger(env):
    """The emitted script is not evidence of a tune: a loci-embed-* dir left over
    from an older run does not clear the cadence."""
    old_model = env.mlops / "embedding" / "loci-embed-small"
    old_model.mkdir()
    os.utime(old_model, (1_000_000, 1_000_000))
    loop._emit_embedding_trigger()
    assert loop._embedding_tune_ran() is False


def test_embedding_tune_ran_true_when_model_postdates_the_trigger(env):
    loop._emit_embedding_trigger()
    trigger = env.mlops / "run_contrastive.sh"
    os.utime(trigger, (1_000_000, 1_000_000))
    (env.mlops / "embedding" / "loci-embed-small").mkdir()
    assert loop._embedding_tune_ran() is True


def test_embedding_tune_ran_ignores_a_file_named_like_the_model_dir(env):
    loop._emit_embedding_trigger()
    os.utime(env.mlops / "run_contrastive.sh", (1_000_000, 1_000_000))
    (env.mlops / "embedding" / "loci-embed-small").write_text("not a model")
    assert loop._embedding_tune_ran() is False


# ══════════════════════════════════════════════════════════════════════════════
# main()  — orchestration
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mainenv(env, monkeypatch):
    """env + every side-effecting step stubbed, so main()'s *decisions* are what
    is under test."""
    calls = {k: [] for k in (
        "rebuild", "retrain", "canary", "decay", "monitor",
        "drift", "sft", "active_learn", "emit",
    )}
    rv = {
        "ollama_ok": True,
        "new_runs": [],
        "retrain": None,
        "canary": None,
        "sft": True,
        "active_learn": {},
        "rebuild": None,   # None → keep current dataset size
        "honesty": {"consistent": None, "unsupported_claims": [], "ok": False, "error": "stubbed"},
    }

    def rec(name, ret=None):
        def f(*a, **kw):
            calls[name].append((a, kw))
            return ret
        return f

    # Hermetic: without this the suite reads the developer's ~/.loci/backends.toml
    # and the asserted Ollama default becomes whatever that machine has configured.
    monkeypatch.setattr(loop, "_resolve_backends", lambda: {})
    monkeypatch.setenv("OLLAMA_BASE_URL", OLLAMA_URL + "/")
    monkeypatch.setattr(loop, "_ollama_ok", lambda base: rv["ollama_ok"])
    monkeypatch.setattr(loop, "_discover_runs", lambda g, seen: list(rv["new_runs"]))
    monkeypatch.setattr(loop, "_rebuild_dataset", lambda g, o: (
        calls["rebuild"].append((g, o)),
        rv["rebuild"] if rv["rebuild"] is not None else loop._current_dataset_size(),
    )[1])
    monkeypatch.setattr(loop, "_retrain", lambda *a, **k: (
        calls["retrain"].append((a, k)), rv["retrain"])[1])
    monkeypatch.setattr(loop, "_run_canary", lambda *a, **k: (
        calls["canary"].append((a, k)), rv["canary"])[1])
    monkeypatch.setattr(loop, "_run_decay", rec("decay", {}))
    monkeypatch.setattr(loop, "_run_monitor", rec("monitor", {}))
    monkeypatch.setattr(loop, "_run_embedding_drift", rec("drift", {}))
    monkeypatch.setattr(loop, "_run_sft_bake", lambda *a, **k: (
        calls["sft"].append((a, k)), rv["sft"])[1])
    monkeypatch.setattr(loop, "_run_active_learn", lambda *a, **k: (
        calls["active_learn"].append((a, k)), rv["active_learn"])[1])
    monkeypatch.setattr(loop, "_emit_embedding_trigger", rec("emit"))
    monkeypatch.setattr(loop, "check_summary_consistency",
                        lambda *a, **k: dict(rv["honesty"]))

    env.calls = calls
    env.rv = rv
    env.argv = ["loop.py"]

    def run(*extra):
        monkeypatch.setattr(sys, "argv", ["loop.py", *extra])
        return loop.main()

    env.main = run
    return env


def state_of(env):
    return json.loads((env.mlops / "loop_state.json").read_text())


def seed_state(env, **kw):
    s = dict(DEFAULT_STATE)
    s.update(kw)
    (env.mlops / "loop_state.json").write_text(json.dumps(s))
    return s


# --- gating -------------------------------------------------------------------

def test_main_first_run_no_ollama_does_not_retrain(mainenv):
    e = mainenv
    e.rv["ollama_ok"] = False
    e.main()
    assert e.calls["retrain"] == []
    assert e.calls["rebuild"] == []
    assert read_history(e)[0]["retrained"] is False


def test_main_retrain_requires_all_three_conditions(mainenv):
    e = mainenv
    e.rv["new_runs"] = ["r1", "r2"]
    write_dataset(e, 300)
    e.main("--min-new-runs", "2", "--min-new-pairs", "200")
    assert len(e.calls["retrain"]) == 1


def test_main_no_retrain_when_pairs_below_threshold(mainenv):
    e = mainenv
    e.rv["new_runs"] = ["r1", "r2", "r3"]
    write_dataset(e, 199)
    e.main("--min-new-pairs", "200")
    assert e.calls["retrain"] == []


def test_main_no_retrain_when_runs_below_threshold(mainenv):
    e = mainenv
    e.rv["new_runs"] = ["r1"]
    write_dataset(e, 5000)
    e.main("--min-new-runs", "2")
    assert e.calls["retrain"] == []


def test_main_no_retrain_when_ollama_down_even_with_plenty_of_data(mainenv):
    e = mainenv
    e.rv["ollama_ok"] = False
    e.rv["new_runs"] = ["a", "b", "c"]
    write_dataset(e, 9000)
    e.main()
    assert e.calls["retrain"] == []


def test_main_force_bypasses_ollama_and_thresholds(mainenv):
    """--force retrains even with Ollama unreachable and zero new data, which is
    exactly the path a cron operator reaches for when the loop looks stuck."""
    e = mainenv
    e.rv["ollama_ok"] = False
    e.main("--force")
    assert len(e.calls["retrain"]) == 1
    assert len(e.calls["rebuild"]) == 1
    assert read_history(e)[0]["retrained"] is True


def test_main_new_pairs_can_be_negative_and_blocks_retrain(mainenv):
    """A shrinking dataset yields a negative delta, which fails the threshold."""
    e = mainenv
    seed_state(e, last_dataset_size=1000)
    e.rv["new_runs"] = ["a", "b"]
    write_dataset(e, 10)
    e.main()
    assert e.calls["retrain"] == []
    assert read_history(e)[0]["dataset_size"] == 10


# --- promotion ----------------------------------------------------------------

def test_main_promotes_when_decision_promote_and_canary_exit_0(mainenv, capsys):
    e = mainenv
    e.rv["retrain"] = {"decision": "PROMOTE", "model": "lr", "cv_f1_mean": 0.9,
                       "cosine_baseline_cv_f1": 0.7}
    e.rv["canary"] = {"exit_code": 0}
    e.main("--force")
    assert read_history(e)[0]["promoted"] is True
    assert state_of(e)["total_promotions"] == 1
    assert "PROMOTED — total promotions: 1" in capsys.readouterr().out


def test_main_does_not_promote_on_canary_failure(mainenv, capsys):
    e = mainenv
    e.rv["retrain"] = {"decision": "PROMOTE"}
    e.rv["canary"] = {"exit_code": 1}
    e.main("--force")
    assert read_history(e)[0]["promoted"] is False
    assert state_of(e)["total_promotions"] == 0
    assert "canary held back or drift detected" in capsys.readouterr().out


def test_main_does_not_promote_when_canary_returns_none(mainenv):
    e = mainenv
    e.rv["retrain"] = {"decision": "PROMOTE"}
    e.rv["canary"] = None
    e.main("--force")
    assert read_history(e)[0]["promoted"] is False


def test_main_hold_decision_skips_canary_entirely(mainenv):
    e = mainenv
    e.rv["retrain"] = {"decision": "HOLD"}
    e.main("--force")
    assert e.calls["canary"] == []
    assert read_history(e)[0]["promoted"] is False


def test_main_missing_decision_key_defaults_to_hold(mainenv):
    e = mainenv
    e.rv["retrain"] = {"model": "lr"}
    e.main("--force")
    assert e.calls["canary"] == []


def test_main_canary_hold_is_not_a_promotion(mainenv, capsys):
    """canary.py exited 0 for HOLD as well as PROMOTE, and main() counts a 0 as a
    promotion — so every HOLD was recorded as promoted=True. HOLD now exits 3."""
    e = mainenv
    e.rv["retrain"] = {"decision": "PROMOTE"}
    e.rv["canary"] = {"exit_code": loop.CANARY_HOLD}
    assert e.main("--force") == 0
    assert read_history(e)[0]["promoted"] is False
    assert state_of(e)["total_promotions"] == 0
    assert "PROMOTED" not in capsys.readouterr().out


def test_main_dry_run_never_records_a_promotion(mainenv, capsys):
    """A dry-run canary copies nothing, so nothing was promoted. It used to bump
    the counter, print PROMOTED and write promoted=true to history."""
    e = mainenv
    e.rv["retrain"] = {"decision": "PROMOTE"}
    e.rv["canary"] = {"exit_code": 0}
    e.main("--force", "--dry-run")
    out = capsys.readouterr().out
    assert "PROMOTED" not in out
    assert "would promote; nothing promoted" in out
    assert not (e.mlops / "loop_state.json").exists()
    assert not (e.mlops / "loop_history.jsonl").exists()


def test_main_null_cv_f1_mean_does_not_crash_the_loop(mainenv, capsys):
    """train.py can write a JSON null for cv_f1_mean; formatting it with :.3f
    raised TypeError out of main() before decay, monitor and the state save."""
    e = mainenv
    e.rv["retrain"] = {"decision": "HOLD", "cv_f1_mean": None}
    assert e.main("--force") == 0
    assert "cv_f1=n/a" in capsys.readouterr().out
    assert state_of(e)["total_loop_runs"] == 1
    assert read_history(e)[0]["train_metrics"] == {"decision": "HOLD", "cv_f1_mean": None}
    assert len(e.calls["decay"]) == 1 and len(e.calls["monitor"]) == 1


# --- state bookkeeping --------------------------------------------------------

def test_main_state_updated_after_retrain(mainenv):
    e = mainenv
    seed_state(e, runs_seen=["old"], last_dataset_size=1)
    e.rv["new_runs"] = ["n1", "n2"]
    e.rv["rebuild"] = 500
    e.main("--force")
    s = state_of(e)
    assert s["last_dataset_size"] == 500
    assert s["runs_seen"] == ["old", "n1", "n2"]


def test_main_runs_seen_untouched_when_not_retraining(mainenv):
    """New runs are deliberately *not* marked seen unless a retrain happened, so
    they keep accumulating until the threshold trips."""
    e = mainenv
    seed_state(e, runs_seen=["old"])
    e.rv["new_runs"] = ["n1"]
    e.main()
    s = state_of(e)
    assert s["runs_seen"] == ["old"]
    assert s["last_dataset_size"] == 0


def test_main_runs_seen_records_each_run_once(mainenv):
    e = mainenv
    seed_state(e, runs_seen=["old", "dup"])
    e.rv["new_runs"] = ["dup", "new", "new"]
    e.main("--force")
    assert state_of(e)["runs_seen"] == ["old", "dup", "new"]


@pytest.mark.xfail(strict=True, reason=(
    "follow-up: a failed retrain must be retried. The rebuild marks the runs seen "
    "and advances last_dataset_size before train.py's result is known, so the "
    "next tick has neither new runs nor new pairs and never retrains."))
def test_a_failed_retrain_is_retried_on_the_next_tick(mainenv):
    e = mainenv
    seed_state(e, last_dataset_size=0)
    e.rv["new_runs"] = ["n1", "n2"]
    e.rv["rebuild"] = 900
    e.rv["retrain"] = None                 # train.py failed
    e.main()
    assert len(e.calls["retrain"]) == 1
    e.rv["new_runs"] = []                  # nothing new since
    e.rv["retrain"] = {"decision": "HOLD"}
    e.main()
    assert len(e.calls["retrain"]) == 2


# Captured at import, before the mainenv fixture stubs it out: the two tests below
# drive the REAL rebuild step so the failure they assert on is the one the nightly
# actually hits (builder exits non-zero), not a hand-written return value.
_REAL_REBUILD = loop._rebuild_dataset


def _failing_rebuild(e, monkeypatch):
    (e.repo / "deep_think_loci" / "grounding" / "build_grounding_dataset.py").write_text("")
    e.run.set("build_grounding_dataset.py",
              FakeResult(2, stderr="urllib.error.URLError: connection refused\n"))
    monkeypatch.setattr(loop, "_rebuild_dataset", _REAL_REBUILD)


def test_main_runs_seen_held_when_the_rebuild_failed(mainenv, monkeypatch):
    """_discover_runs excludes anything already in runs_seen, so a run marked seen
    by a build that never ingested it is never offered to a later tick: the loop
    then reports "new investigation runs: 0" and waits for --min-new-runs fresh
    investigations before it tries again."""
    e = mainenv
    write_dataset(e, 11)
    seed_state(e, runs_seen=["old"], last_dataset_size=11)
    e.rv["new_runs"] = ["n1", "n2"]
    _failing_rebuild(e, monkeypatch)
    e.main("--force")
    s = state_of(e)
    assert s["runs_seen"] == ["old"]
    assert s["last_dataset_size"] == 11


def test_main_says_which_runs_it_held(mainenv, monkeypatch, capsys):
    e = mainenv
    seed_state(e)
    e.rv["new_runs"] = ["n1", "n2"]
    _failing_rebuild(e, monkeypatch)
    e.main("--force")
    assert "holding 2 new run(s) unseen" in capsys.readouterr().out


def test_main_total_loop_runs_increments_and_persists(mainenv):
    e = mainenv
    e.main()
    assert state_of(e)["total_loop_runs"] == 1
    e.main()
    assert state_of(e)["total_loop_runs"] == 2


def test_main_last_run_is_iso_utc(mainenv):
    e = mainenv
    e.main()
    parsed = datetime.fromisoformat(state_of(e)["last_run"])
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)


def test_main_dry_run_never_writes_state(mainenv):
    e = mainenv
    e.main("--dry-run")
    assert not (e.mlops / "loop_state.json").exists()


@pytest.mark.parametrize("content", ["{}", "[]", '{"total_promotions": 4}'])
def test_main_survives_a_partial_or_non_object_state_file(mainenv, content):
    """A state file truncated to {} or written by an older schema raised
    KeyError (a list: TypeError) on main()'s first line."""
    e = mainenv
    (e.mlops / "loop_state.json").write_text(content)
    assert e.main() == 0
    s = state_of(e)
    assert s["total_loop_runs"] == 1
    assert s["total_promotions"] == (4 if "4" in content else 0)
    assert len(read_history(e)) == 1


# --- history ------------------------------------------------------------------

def test_main_history_record_shape(mainenv):
    e = mainenv
    e.rv["new_runs"] = ["a", "b"]
    write_dataset(e, 42)
    e.main()
    rec = read_history(e)[0]
    assert set(rec) == {"run_at", "new_runs", "dataset_size", "retrained",
                        "promoted", "train_metrics", "dry_run", "failed_steps",
                        "alerts"}
    assert rec["failed_steps"] == []
    assert rec["alerts"] == []
    assert rec["new_runs"] == 2
    assert rec["dataset_size"] == 42
    assert rec["retrained"] is False
    assert rec["promoted"] is False
    assert rec["train_metrics"] is None
    assert rec["dry_run"] is False


def test_main_dry_run_appends_no_history(mainenv, capsys):
    e = mainenv
    assert e.main("--dry-run") == 0
    assert not (e.mlops / "loop_history.jsonl").exists()
    assert "dry run: loop_history.jsonl not appended" in capsys.readouterr().out


def test_main_history_dataset_size_is_what_the_run_leaves_behind(mainenv):
    """History used to record the pre-rebuild count while state recorded the
    post-rebuild one, under-reporting the dataset on the nights that grew it."""
    e = mainenv
    write_dataset(e, 10)
    e.rv["rebuild"] = 999
    e.main("--force")
    assert read_history(e)[0]["dataset_size"] == 999
    assert state_of(e)["last_dataset_size"] == 999


def test_main_history_accumulates_across_runs(mainenv):
    e = mainenv
    e.main()
    e.main()
    assert len(read_history(e)) == 2


def test_main_history_carries_train_metrics_verbatim(mainenv):
    e = mainenv
    e.rv["retrain"] = {"decision": "HOLD", "model": "svm", "cv_f1_mean": 0.4}
    e.main("--force")
    assert read_history(e)[0]["train_metrics"] == {"decision": "HOLD", "model": "svm",
                                                  "cv_f1_mean": 0.4}


def test_redacted_honesty_payload_bounds_and_redacts_verbose_fields(env):
    payload = loop._redacted_honesty_payload(
        "S" * 2000,
        {
            "run_at": "2026-09-16T00:00:00+00:00",
            "new_runs": 2,
            "dataset_size": 42,
            "retrained": True,
            "promoted": False,
            "train_metrics": {
                "decision": "PROMOTE",
                "model": "lr",
                "cv_f1_mean": 0.9,
                "cosine_baseline_cv_f1": 0.7,
                "roc_auc": 0.99,
            },
            "dry_run": False,
            "failed_steps": [],
            "alerts": [],
        },
        {
            "ollama_reachable": True,
            "dataset_pairs_before_rebuild": 10,
            "dataset_pairs_after_run": 42,
            "step_results": {
                "monitor": {"rollback_recommended": False, "stdout": "X" * 1000},
                "dataset_rebuild": {"stderr": "secret", "completed": True},
            },
            "artifacts": {"dataset": {"exists": True, "mtime": 123}},
        },
    )
    assert len(payload["summary"]) == loop._HONESTY_SUMMARY_MAX_CHARS
    assert payload["history"]["train_metrics"] == {
        "decision": "PROMOTE",
        "model": "lr",
        "cv_f1_mean": 0.9,
        "cosine_baseline_cv_f1": 0.7,
    }
    dumped = json.dumps(payload)
    assert "roc_auc" not in dumped
    assert "stdout" not in dumped
    assert "stderr" not in dumped


def test_check_summary_consistency_uses_mocked_model_and_parses_json(env, monkeypatch):
    install_fake(monkeypatch, "backends", ollama_verify_model=lambda: "mock-verify")
    seen = {}

    def fake_gen(prompt, **kwargs):
        seen["prompt"] = prompt
        seen["kwargs"] = kwargs
        return {"text": '{"consistent": false, "unsupported_claims": ["claimed promotion"]}',
                "ok": True}

    out = loop.check_summary_consistency(
        "done. promoted=True dataset=42 total_promotions=1",
        {"run_at": "x", "new_runs": 0, "dataset_size": 42, "retrained": False,
         "promoted": True, "train_metrics": None, "dry_run": False,
         "failed_steps": [], "alerts": []},
        {"dataset_pairs_before_rebuild": 42, "dataset_pairs_after_run": 42,
         "step_results": {}, "artifacts": {}},
        gen_fn=fake_gen,
    )
    assert out == {"consistent": False, "unsupported_claims": ["claimed promotion"],
                   "ok": True, "error": None}
    assert seen["kwargs"]["model"] == "mock-verify"
    assert seen["kwargs"]["fmt"] == "json"
    assert seen["kwargs"]["temperature"] == 0.0
    assert '"summary": "done. promoted=True dataset=42 total_promotions=1"' in seen["prompt"]


def test_check_summary_consistency_fails_open_when_model_raises(env, monkeypatch):
    install_fake(monkeypatch, "backends", ollama_verify_model=lambda: "mock-verify")

    def boom(*a, **k):
        raise RuntimeError("ollama down")

    out = loop.check_summary_consistency("done", {}, {}, gen_fn=boom)
    assert out["ok"] is False
    assert out["consistent"] is None
    assert out["unsupported_claims"] == []
    assert "ollama down" in out["error"]


# --- decay cadence ------------------------------------------------------------

def test_main_decay_runs_every_tick_by_default(mainenv):
    e = mainenv
    e.main()
    assert len(e.calls["decay"]) == 1


def test_main_decay_cadence_uses_loop_count_modulo(mainenv, capsys):
    e = mainenv
    seed_state(e, total_loop_runs=0)
    e.main("--decay-every", "2")           # loop_count 1 → skipped
    assert e.calls["decay"] == []
    assert "decay skipped (run 1, cadence=2)" in capsys.readouterr().out
    e.main("--decay-every", "2")           # loop_count 2 → runs
    assert len(e.calls["decay"]) == 1


@pytest.mark.parametrize("value", ["0", "-3"])
def test_main_rejects_a_non_positive_decay_cadence_at_parse_time(mainenv, capsys, value):
    """--decay-every 0 used to reach ``loop_count % 0`` mid-run and raise
    ZeroDivisionError after the retrain, losing the night's state."""
    e = mainenv
    with pytest.raises(SystemExit) as exc:
        e.main("--decay-every", value)
    assert exc.value.code == 2
    assert "must be a positive integer" in capsys.readouterr().err
    assert e.calls["decay"] == [] and e.calls["rebuild"] == []
    assert read_history(e) == []


@pytest.mark.parametrize("flags,writes", [
    ((), False),                             # the nightly default: report only
    (("--decay-apply",), True),              # the one way to write
    (("--decay-apply", "--dry-run"), False), # dry run beats apply
    (("--dry-run",), False),
])
def test_main_decay_writes_only_with_decay_apply_and_no_dry_run(mainenv, flags, writes):
    """The live Mnemosyne DB is written only when asked. The old test passed
    --dry-run alone, so a loop that wrote unasked passed it too."""
    e = mainenv
    e.main("--db", "/tmp/x.db", *flags)
    assert e.calls["decay"] == [(("/tmp/x.db", not writes), {})]


# --- always-on steps ----------------------------------------------------------

def test_main_monitor_always_runs_regardless_of_ollama(mainenv):
    e = mainenv
    e.rv["ollama_ok"] = False
    e.main()
    assert len(e.calls["monitor"]) == 1


def test_main_embedding_drift_gated_on_ollama(mainenv):
    e = mainenv
    e.rv["ollama_ok"] = False
    e.main()
    assert e.calls["drift"] == []
    e.rv["ollama_ok"] = True
    e.main()
    assert len(e.calls["drift"]) == 1


# --- SFT cadence --------------------------------------------------------------

def test_main_sft_runs_on_first_ever_tick(mainenv, capsys):
    e = mainenv
    e.main()
    assert len(e.calls["sft"]) == 1
    assert "SFT bake (last was 999d ago)" in capsys.readouterr().out
    assert state_of(e)["last_sft_bake"] is not None


def test_main_sft_skipped_within_cadence(mainenv, capsys):
    e = mainenv
    recent = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    seed_state(e, last_sft_bake=recent)
    e.main("--sft-every", "7")
    assert e.calls["sft"] == []
    assert "SFT bake skipped — not due (3d ago, cadence 7d)" in capsys.readouterr().out
    assert state_of(e)["last_sft_bake"] == recent


def test_main_sft_day_delta_truncates_toward_zero(mainenv):
    """23h59m ago counts as 0 days, so a 1-day cadence still skips."""
    e = mainenv
    seed_state(e, last_sft_bake=(datetime.now(timezone.utc)
                                 - timedelta(hours=23, minutes=59)).isoformat())
    e.main("--sft-every", "1")
    assert e.calls["sft"] == []


def test_main_sft_runs_at_exact_cadence_boundary(mainenv):
    e = mainenv
    seed_state(e, last_sft_bake=(datetime.now(timezone.utc)
                                 - timedelta(days=7, minutes=1)).isoformat())
    e.main("--sft-every", "7")
    assert len(e.calls["sft"]) == 1


def test_main_sft_skipped_when_ollama_down(mainenv):
    e = mainenv
    e.rv["ollama_ok"] = False
    e.main()
    assert e.calls["sft"] == []


def test_main_sft_state_not_advanced_on_failure(mainenv):
    e = mainenv
    e.rv["sft"] = False
    e.main()
    assert state_of(e)["last_sft_bake"] is None


def test_main_sft_bake_gets_the_run_time_ollama_url(mainenv):
    """The URL is resolved when main() runs (env, trailing slash stripped), not
    frozen at import. The old test compared against the import-time
    loop.DEFAULT_OLLAMA and failed whenever the shell's env differed."""
    e = mainenv
    e.main()
    assert e.calls["sft"] == [((OLLAMA_URL, False), {})]


def test_main_sft_bake_is_not_run_in_dry_run(mainenv, capsys):
    """collect/format write mlops/finetune/data; a dry run writes nothing."""
    e = mainenv
    e.main("--dry-run")
    assert e.calls["sft"] == []
    assert "SFT bake skipped — dry run" in capsys.readouterr().out
    assert not (e.mlops / "loop_state.json").exists()


def test_main_naive_sft_timestamp_is_read_as_utc(mainenv):
    """A naive ISO string in loop_state.json (hand-edited, or written by a
    pre-timezone version) used to raise TypeError out of the cadence gate — which
    sits after the retrain and canary and before the state persist, so the tick
    did its work and threw the result away. It is now read as UTC."""
    e = mainenv
    seed_state(e, last_sft_bake="2024-01-01T00:00:00")
    e.main()
    assert len(e.calls["sft"]) == 1
    assert read_history(e) != []


def test_main_unparseable_sft_timestamp_is_treated_as_ancient(mainenv, capsys):
    """Same crash through ValueError. Unreadable age now falls back to the
    missing-key default — ancient, so the step runs — and says which value it
    could not read."""
    e = mainenv
    seed_state(e, last_sft_bake="never")
    e.main()
    assert len(e.calls["sft"]) == 1
    assert "unreadable timestamp 'never' in loop_state.json" in capsys.readouterr().out
    assert read_history(e) != []


def test_days_since_normalises_a_naive_stamp_instead_of_raising():
    now = datetime(2026, 1, 11, tzinfo=timezone.utc)
    assert loop._days_since(now, "2026-01-01T00:00:00") == 10
    assert loop._days_since(now, "2026-01-01T00:00:00+00:00") == 10


def test_days_since_falls_back_to_the_missing_key_default():
    now = datetime(2026, 1, 11, tzinfo=timezone.utc)
    assert loop._days_since(now, "never") == 999
    assert loop._days_since(now, None) == 999
    assert loop._days_since(now, "") == 999


# --- embedding trigger cadence ------------------------------------------------

def test_main_embedding_trigger_fires_on_first_tick_ignoring_ollama(mainenv, capsys):
    """The embedding trigger is the one cadence step with no Ollama gate."""
    e = mainenv
    e.rv["ollama_ok"] = False
    e.main()
    assert len(e.calls["emit"]) == 1
    assert "embedding trigger (last was 999d ago)" in capsys.readouterr().out


def test_main_embedding_trigger_does_not_stamp_the_cadence_on_emission(mainenv, capsys):
    """Nothing in the repo runs run_contrastive.sh. Stamping on emission bought
    30 days of silence for a fine-tune that had not happened, so with no
    loci-embed-* output the cadence stays unstamped and the run says so."""
    e = mainenv
    e.main()
    assert state_of(e)["last_embedding_tune"] is None
    assert ("embedding fine-tune pending — run: bash "
            f"{e.mlops / 'run_contrastive.sh'}") in capsys.readouterr().out


def test_main_embedding_trigger_stamps_once_a_fine_tune_exists(mainenv, capsys):
    e = mainenv
    (e.mlops / "embedding" / "loci-embed-small").mkdir()
    e.main()
    assert state_of(e)["last_embedding_tune"] is not None
    assert "embedding fine-tune pending" not in capsys.readouterr().out


def test_main_embedding_trigger_does_not_stamp_a_fine_tune_in_dry_run(mainenv):
    e = mainenv
    (e.mlops / "embedding" / "loci-embed-small").mkdir()
    e.main("--dry-run")
    assert not (e.mlops / "loop_state.json").exists()


def test_main_embedding_trigger_skipped_within_cadence(mainenv):
    e = mainenv
    seed_state(e, last_embedding_tune=(datetime.now(timezone.utc)
                                       - timedelta(days=10)).isoformat())
    e.main("--embedding-every", "30")
    assert e.calls["emit"] == []


def test_main_embedding_trigger_is_not_written_in_dry_run(mainenv, capsys):
    """_emit_embedding_trigger used to run before the dry-run check and write
    run_contrastive.sh on a dry run."""
    e = mainenv
    e.main("--dry-run")
    assert e.calls["emit"] == []
    assert "run_contrastive.sh not written" in capsys.readouterr().out
    assert not (e.mlops / "loop_state.json").exists()


# --- active learning cadence --------------------------------------------------

def test_main_active_learn_runs_first_tick_and_records_state(mainenv, capsys):
    e = mainenv
    e.rv["active_learn"] = {"exit_code": 0}
    e.main()
    assert len(e.calls["active_learn"]) == 1
    assert state_of(e)["last_active_learn"] is not None
    assert "active_learn (last was 999d ago)" in capsys.readouterr().out


def test_main_active_learn_state_not_advanced_on_nonzero_exit(mainenv):
    e = mainenv
    e.rv["active_learn"] = {"exit_code": 1}
    e.main()
    assert "last_active_learn" not in state_of(e)


def test_main_active_learn_empty_result_treated_as_failure(mainenv):
    """A degraded {} (missing model/dataset/script) defaults exit_code to 1."""
    e = mainenv
    e.rv["active_learn"] = {}
    e.main()
    assert "last_active_learn" not in state_of(e)


def test_main_active_learn_skipped_when_ollama_down(mainenv):
    e = mainenv
    e.rv["ollama_ok"] = False
    e.main()
    assert e.calls["active_learn"] == []


def test_main_active_learn_is_not_run_in_dry_run(mainenv, capsys):
    """active_learn.py rewrites active_candidates.jsonl; it used to be invoked
    for real on a --dry-run night."""
    e = mainenv
    e.rv["active_learn"] = {"exit_code": 0}
    e.main("--dry-run")
    assert e.calls["active_learn"] == []
    assert "active_learn skipped — dry run" in capsys.readouterr().out


def test_main_active_learn_gets_the_run_time_ollama_url(mainenv):
    e = mainenv
    e.rv["active_learn"] = {"exit_code": 0}
    e.main()
    assert e.calls["active_learn"] == [((OLLAMA_URL,), {})]


# --- argparse defaults --------------------------------------------------------

def test_main_default_thresholds(mainenv, capsys):
    """Pin the cron-visible defaults; changing any of them changes how often an
    unattended box retrains."""
    e = mainenv
    e.rv["new_runs"] = ["a"]
    write_dataset(e, 100)
    e.main()
    out = capsys.readouterr().out
    # One new run is under min_new_runs (2), so nothing is rebuilt, and the line
    # says which threshold decided rather than only reporting the outcome.
    assert "rebuild=False" in out
    assert "1 new runs < 2" in out


def test_main_argparse_defaults(mainenv, monkeypatch):
    captured = {}
    namespaces = []
    real_parse = loop.argparse.ArgumentParser.parse_args

    def spy(self, *a, **k):
        ns = real_parse(self, *a, **k)
        captured.update(vars(ns))
        namespaces.append(ns)
        return ns

    monkeypatch.setattr(loop.argparse.ArgumentParser, "parse_args", spy)
    mainenv.main()
    assert captured["min_new_runs"] == 2
    assert captured["min_new_pairs"] == 200
    assert captured["sft_every"] == 7
    assert captured["embedding_every"] == 30
    assert captured["decay_every"] == 1
    assert captured["active_learn_every"] == 7
    assert captured["dry_run"] is False
    assert captured["force"] is False
    assert captured["findings"] == loop.DEFAULT_FINDINGS
    # --ollama parses as None and is resolved after, so the config file gets a say;
    # an import-time default would have been fixed before backends.toml was read.
    assert captured["ollama"] is None
    assert namespaces[0].ollama, "main() must fill in an Ollama URL"
    assert captured["db"] == loop.DEFAULT_DB


def test_module_level_defaults_are_expanded_paths():
    assert loop.DEFAULT_FINDINGS.endswith("/dt-loci-*/findings.jsonl")
    assert "~" not in loop.DEFAULT_DB
    assert not loop.DEFAULT_OLLAMA.endswith("/")


def test_main_forwards_findings_glob_to_every_consumer(mainenv):
    e = mainenv
    e.rv["retrain"] = {"decision": "PROMOTE"}
    e.rv["canary"] = {"exit_code": 0}
    e.main("--force", "--findings", "GLOB", "--ollama", "http://o")
    assert e.calls["rebuild"][0] == ("GLOB", "http://o")
    assert e.calls["retrain"][0][0] == ("GLOB", "http://o", False)
    assert e.calls["canary"][0][0] == ("GLOB", "http://o", False)
    assert e.calls["monitor"][0][0] == ("GLOB", "http://o", False)


# --- the retrain gate decides from inputs -------------------------------------

def test_new_runs_alone_trigger_a_rebuild_even_when_the_file_shrank(mainenv, capsys):
    """The bug this split fixes. The cron wrapper resets its worktree to
    origin/main nightly, restoring the committed dataset while loop_state.json
    outside the worktree holds the previous run's larger count. The delta came out
    at -7,266 and vetoed everything on the night three new investigations carrying
    1,139 findings had just been discovered."""
    e = mainenv
    seed_state(e, last_dataset_size=12684)
    e.rv["new_runs"] = ["hunt-a", "hunt-b", "hunt-c"]
    write_dataset(e, 5418)                      # the reset artifact
    e.rv["rebuild"] = 12684                     # what the rebuild actually finds
    e.main()
    out = capsys.readouterr().out
    assert "rebuild=True" in out, out
    assert "3 new runs >= 2" in out
    assert len(e.calls["rebuild"]) == 1


def test_the_pair_threshold_is_applied_to_the_real_delta_not_the_stale_one(mainenv, capsys):
    """min_new_pairs still matters — it just has to be measured after the rebuild,
    which is the only point the number exists."""
    e = mainenv
    seed_state(e, last_dataset_size=1000)
    e.rv["new_runs"] = ["a", "b"]
    write_dataset(e, 1000)
    e.rv["rebuild"] = 1010                      # rebuild adds only 10 pairs
    e.main()
    out = capsys.readouterr().out
    assert len(e.calls["rebuild"]) == 1, "the rebuild should still run"
    assert e.calls["retrain"] == [], "training on +10 pairs is not justified"
    assert "under the 200 needed" in out


def test_a_shrinking_dataset_is_reported_rather_than_silently_vetoing(mainenv, capsys):
    e = mainenv
    seed_state(e, last_dataset_size=12684)
    e.rv["new_runs"] = ["a", "b"]
    write_dataset(e, 5418)
    e.rv["rebuild"] = 12684
    e.main()
    assert "-7266 vs last run, pre-rebuild" in capsys.readouterr().out


# ══════════════════════════════════════════════════════════════════════════════
# the embedding cadence, driven through main() with the REAL emit
#
# Every other test here stubs _emit_embedding_trigger, so the trigger file never
# exists and _embedding_tune_ran() takes its emitted=0.0 branch. That is what let
# a one-shot cadence look like a 30-day one: the script's body is stable, so it is
# written once and its mtime freezes, and a tune done once stays newer than that
# frozen mtime forever. These drive the real emit across three ticks.
# ══════════════════════════════════════════════════════════════════════════════

def test_a_single_tune_does_not_satisfy_the_cadence_forever(env):
    """The regression the review caught: after one real tune, every later cadence
    tick re-stamped last_embedding_tune for work that never happened."""
    loop._emit_embedding_trigger()                       # freezes the trigger mtime
    script = env.mlops / "run_contrastive.sh"
    assert script.exists()

    # Whole seconds throughout: a stamp is an ISO string truncated to
    # microseconds while an mtime is a float with more precision than that, and
    # a sub-microsecond difference is not what this test is about.
    emitted = float(int(script.stat().st_mtime))
    os.utime(script, (emitted, emitted))
    tuned = env.mlops / "embedding" / "loci-embed-small"
    tuned.mkdir(parents=True)
    tune_time = emitted + 10
    os.utime(tuned, (tune_time, tune_time))

    def iso(ts):
        return datetime.fromtimestamp(ts, timezone.utc).isoformat()

    # Tick 1: a real tune, newer than the trigger. It counts, and main() stamps.
    assert loop._embedding_tune_ran(None) is True
    assert loop._embedding_tune_ran(iso(emitted)) is True, (
        "a tune done after the last stamp must count"
    )

    # Tick 2, a cadence later, no new tune. The directory is unchanged, so it must
    # NOT count again -- the trigger's mtime is frozen at first write, which is
    # what made one tune satisfy the gate for the rest of time.
    assert loop._embedding_tune_ran(iso(tune_time)) is False, (
        "the same tune directory satisfied the gate a second time; the cadence "
        "must date against the last recorded tune, not only the frozen trigger"
    )
    assert loop._embedding_tune_ran(iso(tune_time + 86400 * 31)) is False


def test_the_emitted_command_can_actually_import_its_dependencies(env):
    """The nag says `bash run_contrastive.sh`. If that command dies on import the
    fine-tune never happens and the nag is permanent."""
    loop._emit_embedding_trigger()
    body = (env.mlops / "run_contrastive.sh").read_text()
    interp = body.split(" mlops/embedding/contrastive.py")[0].splitlines()[-1].strip()
    out = subprocess.run([interp, "-c", "import sentence_transformers"],
                         capture_output=True, text=True)
    assert out.returncode == 0, (
        f"the emitted script runs {interp}, which cannot import sentence_transformers:\n"
        + out.stderr.strip()
    )


# ══════════════════════════════════════════════════════════════════════════════
# _run must not be able to hang
#
# Measured 2026-09-01: a loop.py sat in futex_wait_queue for 19h07m with its two
# drain threads in pipe_read, holding /tmp/loci-mlops.lock. The join had no
# timeout on the assumption that reaping the child closes the pipes -- true only
# when the child had no grandchild. A grandchild inherits the pipe fds and holds
# the write end open after its parent dies, so `for line in pipe` never returns.
# ══════════════════════════════════════════════════════════════════════════════

def test_run_returns_even_when_a_grandchild_holds_the_pipe_open(monkeypatch):
    """The child exits immediately; a grandchild keeps stdout open past it."""
    monkeypatch.setattr(loop, "DRAIN_JOIN_SECONDS", 1.0, raising=False)
    started = time.monotonic()
    result = loop._run([
        sys.executable, "-c",
        # spawn a grandchild that inherits stdout and outlives us, then exit
        "import subprocess,sys;"
        "print('parent line', flush=True);"
        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']);"
        "sys.exit(0)",
    ])
    elapsed = time.monotonic() - started
    assert elapsed < 20, f"_run took {elapsed:.1f}s — it is waiting on the grandchild"
    assert result.returncode == 0
    assert "parent line" in result.stdout


def test_run_says_so_when_it_abandons_a_blocked_reader(monkeypatch, capsys):
    monkeypatch.setattr(loop, "DRAIN_JOIN_SECONDS", 0.5, raising=False)
    result = loop._run([
        sys.executable, "-c",
        "import subprocess,sys;"
        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']);"
        "sys.exit(0)",
    ])
    out = capsys.readouterr().out
    assert "still blocked" in out, out
    assert "still blocked" in result.stderr, result.stderr


# ══════════════════════════════════════════════════════════════════════════════
# the run has a wall clock
# ══════════════════════════════════════════════════════════════════════════════

def test_main_stops_at_the_deadline_and_still_persists(mainenv, monkeypatch, capsys):
    """A run that stops early must write its state and say why. An early exit
    that looks like a clean one is the whole problem."""
    e = mainenv
    write_dataset(e, 10)
    monkeypatch.setattr(loop, "DEADLINE_SECONDS", 0.0)
    rc = e.main()
    out = capsys.readouterr().out
    assert rc == 1, "a run cut short is not a clean run"
    assert "run deadline reached" in out, out
    assert "deadline" in loop.FAILED_STEPS
    rec = read_history(e)[0]
    assert rec["failed_steps"] == ["deadline"]
    assert (e.mlops / "loop_state.json").exists(), "state was not persisted on the early exit"


def test_main_does_not_stop_when_inside_the_deadline(mainenv, monkeypatch, capsys):
    e = mainenv
    write_dataset(e, 10)
    monkeypatch.setattr(loop, "DEADLINE_SECONDS", 10_000.0)
    assert e.main() == 0
    assert "run deadline reached" not in capsys.readouterr().out


def test_alerts_are_reported_separately_from_failures(mainenv, monkeypatch, capsys):
    """A canary rollback recommendation is not a broken step, and must not be
    filed as one -- nor swallowed. main() clears both lists at entry, so the
    alert has to be raised by a step during the run."""
    e = mainenv
    write_dataset(e, 10)
    monkeypatch.setattr(loop, "_run_monitor",
                        lambda *a, **k: (loop._alert("canary", "recommends ROLLBACK"), {})[1])
    rc = e.main()
    rec = read_history(e)[0]
    assert rec["alerts"] == ["canary"], rec
    assert rec["failed_steps"] == [], "an alert is not a failure"
    assert rc == 0, "an alert alone must not fail the run"
    assert "1 alert(s): canary" in capsys.readouterr().out


def test_summary_honesty_alert_does_not_change_a_clean_exit_or_promotion(mainenv):
    e = mainenv
    e.rv["retrain"] = {"decision": "PROMOTE", "model": "lr", "cv_f1_mean": 0.9,
                       "cosine_baseline_cv_f1": 0.7}
    e.rv["canary"] = {"exit_code": 0}
    e.rv["honesty"] = {
        "consistent": False,
        "unsupported_claims": ["claimed promotion without support"],
        "ok": True,
        "error": None,
    }
    rc = e.main("--force")
    rec = read_history(e)[0]
    assert rc == 0, "summary corroboration must not flip a clean run nonzero"
    assert rec["promoted"] is True
    assert state_of(e)["total_promotions"] == 1
    assert "summary honesty" in rec["alerts"]
    assert rec["failed_steps"] == []


def test_summary_honesty_alert_does_not_clear_an_existing_failure(mainenv):
    e = mainenv
    e.rv["retrain"] = None
    e.rv["rebuild"] = None
    e.rv["honesty"] = {
        "consistent": False,
        "unsupported_claims": ["claimed success"],
        "ok": True,
        "error": None,
    }
    loop._fail("train.py", "boom")
    rc = loop._finish(
        state=dict(DEFAULT_STATE),
        args=types.SimpleNamespace(dry_run=False),
        now_iso="2026-09-16T00:00:00+00:00",
        loop_count=1,
        new_runs=[],
        current_size=0,
        should_retrain=False,
        promoted=False,
        train_metrics=None,
        run_evidence={},
    )
    rec = read_history(e)[0]
    assert rc == 1, "summary corroboration must not mask pre-existing FAILED_STEPS"
    assert rec["failed_steps"] == ["train.py"]
    assert "summary honesty" in rec["alerts"]


# ══════════════════════════════════════════════════════════════════════════════
# a whole dry run, through the real steps
#
# The per-step tests above stub the steps; this drives every real step function
# through main() with fake children that write wherever they are pointed, as the
# real scripts do. The contract of --dry-run ("plan only") is that the tree is
# byte-for-byte unchanged afterwards and no promotion is recorded anywhere.
# ══════════════════════════════════════════════════════════════════════════════

def _snapshot(root):
    return {str(p.relative_to(root)): (p.read_bytes() if p.is_file() else None)
            for p in sorted(root.rglob("*"))}


def test_a_full_dry_run_writes_nothing(env, monkeypatch, capsys):
    e = env
    monkeypatch.setattr(loop, "_resolve_backends", lambda: {})
    monkeypatch.setenv("OLLAMA_BASE_URL", OLLAMA_URL)
    monkeypatch.setattr(loop, "_ollama_ok", lambda base: True)
    monkeypatch.setattr(loop, "check_summary_consistency",
                        lambda *a, **k: {"consistent": None, "unsupported_claims": [],
                                         "ok": False, "error": "stubbed"})
    decay_calls = []
    install_fake(monkeypatch, "memory.decay",
                 apply_decay=lambda **k: decay_calls.append(k) or {
                     "n_rows": 3, "n_decayed": 1, "mean_retention": 0.9})
    install_fake(monkeypatch, "mlops.grounding.canary",
                 monitor_live=lambda **k: {"drift": False, "rollback_recommended": False})

    # Everything a real nightly finds on disk, so every step has work to do.
    (e.grounding / "build_grounding_dataset.py").write_text("")
    (e.grounding / "grounding_bleed_clf.joblib").write_text("live")
    (e.mlops / "grounding" / "candidate.joblib").write_text("candidate")
    (e.mlops / "grounding" / "active_learn.py").write_text("")
    (e.mlops / "embedding" / "drift.py").write_text("")
    (e.mlops / "embedding" / "anchor.npz").write_text("anchor")
    write_dataset(e, 500)
    sessions = e.tmp / "sessions"
    for run in ("dt-loci-a", "dt-loci-b", "dt-loci-c"):
        (sessions / run).mkdir(parents=True)
        (sessions / run / "findings.jsonl").write_text("{}\n")

    def writes_out(cmd):
        Path(cmd[cmd.index("--out") + 1]).write_text("{}")

    e.run.on_call_cmd("build_grounding_dataset.py", lambda cmd: write_dataset(e, 900))
    e.run.on_call_cmd("train.py", lambda cmd: None if "--dry-run" in cmd else writes_out(cmd))
    e.run.on_call_cmd("drift.py", writes_out)
    e.run.on_call_cmd("active_learn.py", writes_out)
    e.run.on_call_cmd("collect.py", lambda cmd: (Path(cmd[cmd.index("--out") + 1])
                                                 / "raw_traces.jsonl").write_text("x"))

    before = _snapshot(e.repo)
    monkeypatch.setattr(sys, "argv", ["loop.py", "--dry-run", "--force",
                                      "--findings", str(sessions / "*" / "findings.jsonl"),
                                      "--db", str(e.tmp / "mnemosyne.db"), "--decay-apply"])
    rc = loop.main()
    out = capsys.readouterr().out

    assert rc == 0, out
    assert _snapshot(e.repo) == before, "a dry run changed the tree"
    assert "PROMOTED" not in out
    assert decay_calls == [{"db_path": str(e.tmp / "mnemosyne.db"), "dry_run": True}]
    # The read-only steps still ran; only the writers were held back.
    assert "train.py" in e.run.scripts() and "drift.py" in e.run.scripts()
    for writer in ("build_grounding_dataset.py", "active_learn.py", "collect.py",
                   "format_sft.py", "train_lora.py"):
        assert writer not in e.run.scripts(), f"{writer} ran on a dry run"
