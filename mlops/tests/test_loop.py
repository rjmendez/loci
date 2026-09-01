"""Characterization tests for mlops/loop.py — the unattended nightly MLOps loop.

These tests pin the CURRENT behaviour of the module, bugs included. They are a
safety net for a later refactor, not a specification of what the loop *should*
do. Where a test pins something that is arguably wrong, the docstring says so
and the finding is reported separately.

No external services are used: every subprocess call, every HTTP probe and
every dynamic import performed by the loop is replaced with an in-process fake.
"""

import json
import os
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

    def __call__(self, cmd, *a, **kw):
        self.calls.append(list(cmd))
        key = os.path.basename(cmd[1]) if len(cmd) > 1 else ""
        if key in self.side_effects:
            self.side_effects[key]()
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


def test_load_state_returns_partial_dict_verbatim_without_backfill(env):
    """Any dict that parses is returned as-is — missing keys are NOT filled in."""
    (env.mlops / "loop_state.json").write_text('{"total_promotions": 5}')
    assert loop._load_state() == {"total_promotions": 5}


def test_load_state_does_not_type_check_returns_list(env):
    """A JSON document that is not an object is returned unchanged."""
    (env.mlops / "loop_state.json").write_text("[1, 2, 3]")
    assert loop._load_state() == [1, 2, 3]


def test_save_state_writes_indent_2_json(env):
    loop._save_state({"a": 1, "b": [2]})
    text = (env.mlops / "loop_state.json").read_text()
    assert text == json.dumps({"a": 1, "b": [2]}, indent=2)
    assert "\n  " in text


def test_save_state_overwrites_previous_content(env):
    loop._save_state({"a": 1})
    loop._save_state({"b": 2})
    assert json.loads((env.mlops / "loop_state.json").read_text()) == {"b": 2}


def test_append_history_creates_file_and_appends_one_line_per_call(env):
    loop._append_history({"n": 1})
    loop._append_history({"n": 2})
    lines = (env.mlops / "loop_history.jsonl").read_text().splitlines()
    assert [json.loads(l)["n"] for l in lines] == [1, 2]


def test_append_history_raises_when_parent_dir_missing(env, monkeypatch):
    """History is opened in append mode with no mkdir — a missing mlops/ dir
    takes down the very last step of the nightly run."""
    monkeypatch.setattr(loop, "HISTORY_FILE", env.tmp / "nope" / "h.jsonl")
    with pytest.raises(FileNotFoundError):
        loop._append_history({"n": 1})


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


def test_ollama_ok_does_not_strip_trailing_slash(env, monkeypatch):
    """Only the DEFAULT_OLLAMA constant is rstrip()'d; a caller-supplied base
    with a trailing slash produces a double slash in the probe URL."""
    seen = {}
    monkeypatch.setattr(loop.urllib.request, "urlopen",
                        lambda url, timeout=None: seen.setdefault("url", url))
    loop._ollama_ok("http://h/")
    assert seen["url"] == "http://h//api/tags"


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


def test_discover_runs_does_not_dedupe_within_a_batch(env):
    """Run identity is only the parent directory *basename*, and the result list
    is not de-duplicated, so two same-named session dirs under different roots
    are reported twice (and both get appended to runs_seen)."""
    for root in ("a", "b"):
        d = env.tmp / root / "dt-loci-1"
        d.mkdir(parents=True)
        (d / "findings.jsonl").write_text("{}\n")
    g = str(env.tmp / "*" / "dt-loci-1" / "findings.jsonl")
    assert loop._discover_runs(g, []) == ["dt-loci-1", "dt-loci-1"]


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

def test_rebuild_dataset_missing_builder_returns_current_size_without_subprocess(env, capsys):
    write_dataset(env, 4)
    assert loop._rebuild_dataset("g", "http://o") == 4
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


def test_rebuild_dataset_failure_returns_current_size_and_prints_the_exception(env, capsys):
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
    assert loop._rebuild_dataset("g", "http://o") == 7
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


def test_retrain_returns_stale_metrics_when_train_writes_nothing(env):
    """train.py exiting 0 without writing train_metrics.json makes the loop
    re-read *last* night's metrics and act on them again."""
    _metrics_path(env).write_text('{"decision": "PROMOTE", "stale": true}')
    assert loop._retrain("g", "o", False) == {"decision": "PROMOTE", "stale": True}


def test_retrain_propagates_malformed_metrics_json(env):
    """Unlike _load_state, a corrupt metrics file is not tolerated."""
    _metrics_path(env).write_text("{oops")
    with pytest.raises(json.JSONDecodeError):
        loop._retrain("g", "o", False)


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


def test_run_canary_other_nonzero_exit_has_no_alert(env, capsys):
    (env.mlops / "grounding" / "candidate.joblib").write_text("m")
    env.run.set("canary.py", FakeResult(2, stdout="crash"))
    assert loop._run_canary("g", "o", False)["exit_code"] == 2
    assert "ALERT" not in capsys.readouterr().out


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


def test_sft_bake_creates_data_dir_and_runs_collect_then_format(env):
    env.run.on_call("format_sft.py", lambda: _good_sft(env))
    assert loop._run_sft_bake("http://o", dry_run=True) is True
    assert (env.mlops / "finetune" / "data").is_dir()
    assert env.run.scripts() == ["collect.py", "format_sft.py"]


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


def test_sft_bake_dry_run_returns_true_without_baking(env):
    env.run.on_call("format_sft.py", lambda: _good_sft(env))
    assert loop._run_sft_bake("o", dry_run=True) is True
    assert "train_lora.py" not in env.run.scripts()


def test_sft_bake_argv_of_real_bake(env):
    env.run.on_call("format_sft.py", lambda: _good_sft(env))
    loop._run_sft_bake("http://o", dry_run=False)
    data = env.mlops / "finetune" / "data"
    assert env.run.calls[0][2:] == ["--out", str(data), "--ollama", "http://o"]
    assert env.run.calls[1][2:] == ["--traces", str(data / "raw_traces.jsonl"),
                                    "--out", str(data / "sft_pairs.jsonl"),
                                    "--mode", "both"]
    assert env.run.calls[2][2:] == ["--sft", str(data / "sft_pairs.jsonl"),
                                    "--backend", "ollama-modelfile"]


def test_sft_bake_returns_false_when_bake_fails(env):
    env.run.on_call("format_sft.py", lambda: _good_sft(env))
    env.run.set("train_lora.py", FakeResult(1))
    assert loop._run_sft_bake("o", False) is False


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


def test_run_decay_missing_mean_retention_key_uses_zero_default(env, monkeypatch, capsys):
    install_fake(monkeypatch, "memory.decay",
                 apply_decay=lambda **k: {"n_rows": 1, "n_decayed": 0})
    assert loop._run_decay("/db", False) == {"n_rows": 1, "n_decayed": 0}
    assert "mean_retention=0.000" in capsys.readouterr().out


def test_run_decay_null_mean_retention_discards_successful_result(env, monkeypatch, capsys):
    """BUG pinned: the log line formats mean_retention with ``:.3f`` *inside* the
    try block. A None value (a real possibility for an empty table) raises
    TypeError, is swallowed by the bare ``except Exception``, and a decay that
    actually ran is reported to the caller as ``{}``."""
    install_fake(monkeypatch, "memory.decay",
                 apply_decay=lambda **k: {"n_rows": 0, "n_decayed": 0, "mean_retention": None})
    assert loop._run_decay("/db", False) == {}
    assert "decay step failed" in capsys.readouterr().out


def test_run_decay_non_dict_return_is_swallowed(env, monkeypatch):
    install_fake(monkeypatch, "memory.decay", apply_decay=lambda **k: None)
    assert loop._run_decay("/db", False) == {}


# ══════════════════════════════════════════════════════════════════════════════
# _run_live_evo
# ══════════════════════════════════════════════════════════════════════════════

def test_run_live_evo_returns_stats_and_forwards_kwargs(env, monkeypatch, capsys):
    seen = {}

    def adapt(db_path, hook_state_dir, dry_run):
        seen.update(db_path=db_path, hook_state_dir=hook_state_dir, dry_run=dry_run)
        return {"n_failures": 2, "n_correlated": 1, "n_penalized": 1}

    install_fake(monkeypatch, "memory.live_evo", adapt=adapt)
    out = loop._run_live_evo("/db", "/hooks", dry_run=False)
    assert out == {"n_failures": 2, "n_correlated": 1, "n_penalized": 1}
    assert seen == {"db_path": "/db", "hook_state_dir": "/hooks", "dry_run": False}
    assert "failures=2 correlated=1 penalized=1" in capsys.readouterr().out


def test_run_live_evo_swallows_adapt_exception(env, monkeypatch, capsys):
    def boom(**kw):
        raise ValueError("no hook state")

    install_fake(monkeypatch, "memory.live_evo", adapt=boom)
    assert loop._run_live_evo("/db", "/h", False) == {}
    assert "live_evo step failed: no hook state" in capsys.readouterr().out


def test_run_live_evo_does_not_extend_sys_path_itself(env, monkeypatch):
    """BUG pinned: _run_live_evo imports the top-level ``memory`` package but,
    unlike _run_decay, never puts MLOPS on sys.path. It only works as a side
    effect of _run_decay having run first; with --decay-every > 1 the import
    fails and Live-Evo silently degrades to {}."""
    monkeypatch.setitem(sys.modules, "memory", None)
    before = list(sys.path)
    assert loop._run_live_evo("/db", "/h", False) == {}
    assert sys.path == before


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


def test_run_monitor_alerts_on_rollback_recommendation(env, monkeypatch, capsys):
    (env.grounding / "grounding_bleed_clf.joblib").write_text("m")
    install_fake(monkeypatch, "mlops.grounding.canary",
                 monitor_live=lambda **k: {"drift": 9, "rollback_recommended": True})
    out = loop._run_monitor("g", "o", False)
    assert out["rollback_recommended"] is True
    assert "ALERT: rollback recommended" in capsys.readouterr().out


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


def test_run_monitor_sys_path_insert_is_a_no_op_for_the_import_it_guards(env, monkeypatch):
    """BUG pinned: it prepends MLOPS/grounding to sys.path but then imports the
    absolute path ``mlops.grounding.canary``, which needs REPO on sys.path
    instead. The inserted entry can never satisfy that import."""
    install_fake(monkeypatch, "mlops.grounding.canary", monitor_live=lambda **k: {})
    (env.grounding / "grounding_bleed_clf.joblib").write_text("m")
    loop._run_monitor("g", "o", False)
    assert sys.path[0] == str(env.mlops / "grounding")


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


def test_embedding_drift_reports_built_anchor_even_when_build_fails(env):
    """The anchor-build branch never inspects the return code."""
    _drift_script(env)
    env.run.set("drift.py", FakeResult(1, stderr="boom"))
    assert loop._run_embedding_drift("o", False) == {"built_anchor": True}


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


def test_embedding_drift_returns_stale_result_json(env):
    """A drift_result.json left by an earlier run is returned verbatim even when
    this run's drift.py wrote nothing."""
    _drift_script(env)
    (env.mlops / "embedding" / "anchor.npz").write_text("a")
    (env.mlops / "embedding" / "drift_result.json").write_text('{"stale": true}')
    assert loop._run_embedding_drift("o", False) == {"stale": True}


def test_embedding_drift_malformed_result_json_falls_back_to_exit_code(env):
    _drift_script(env)
    (env.mlops / "embedding" / "anchor.npz").write_text("a")
    (env.mlops / "embedding" / "drift_result.json").write_text("{broken")
    env.run.set("drift.py", FakeResult(1))
    assert loop._run_embedding_drift("o", False) == {"exit_code": 1}


def test_embedding_drift_exit_1_emits_contrastive_script(env, capsys):
    _drift_script(env)
    (env.mlops / "embedding" / "anchor.npz").write_text("a")
    env.run.set("drift.py", FakeResult(1))
    loop._run_embedding_drift("o", dry_run=False)
    assert (env.mlops / "run_contrastive.sh").exists()
    assert "ALERT: embedding drift detected" in capsys.readouterr().out


def test_embedding_drift_dry_run_alerts_but_emits_nothing(env, capsys):
    _drift_script(env)
    (env.mlops / "embedding" / "anchor.npz").write_text("a")
    env.run.set("drift.py", FakeResult(1))
    loop._run_embedding_drift("o", dry_run=True)
    assert not (env.mlops / "run_contrastive.sh").exists()
    assert "ALERT: embedding drift detected" in capsys.readouterr().out


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
        "rebuild", "retrain", "canary", "decay", "live_evo", "monitor",
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
    }

    def rec(name, ret=None):
        def f(*a, **kw):
            calls[name].append((a, kw))
            return ret
        return f

    # Hermetic: without this the suite reads the developer's ~/.loci/backends.toml
    # and the asserted Ollama default becomes whatever that machine has configured.
    monkeypatch.setattr(loop, "_resolve_backends", lambda: {})
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
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
    monkeypatch.setattr(loop, "_run_live_evo", rec("live_evo", {}))
    monkeypatch.setattr(loop, "_run_monitor", rec("monitor", {}))
    monkeypatch.setattr(loop, "_run_embedding_drift", rec("drift", {}))
    monkeypatch.setattr(loop, "_run_sft_bake", lambda *a, **k: (
        calls["sft"].append((a, k)), rv["sft"])[1])
    monkeypatch.setattr(loop, "_run_active_learn", lambda *a, **k: (
        calls["active_learn"].append((a, k)), rv["active_learn"])[1])
    monkeypatch.setattr(loop, "_emit_embedding_trigger", rec("emit"))

    env.calls = calls
    env.rv = rv
    env.argv = ["loop.py"]

    def run(*extra):
        monkeypatch.setattr(sys, "argv", ["loop.py", *extra])
        loop.main()

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


def test_main_promotion_counter_increments_in_dry_run_but_is_not_persisted(mainenv, capsys):
    e = mainenv
    e.rv["retrain"] = {"decision": "PROMOTE"}
    e.rv["canary"] = {"exit_code": 0}
    e.main("--force", "--dry-run")
    assert not (e.mlops / "loop_state.json").exists()
    assert "total promotions: 1" in capsys.readouterr().out
    assert read_history(e)[0]["promoted"] is True


def test_main_null_cv_f1_mean_in_metrics_crashes_the_loop(mainenv):
    """BUG pinned: the train-decision log line formats cv_f1_mean with ``:.3f``
    with only a *missing-key* default. A JSON null (which train.py can emit when
    CV is skipped) raises TypeError out of main() — the nightly dies before the
    decay / monitor / state-persist steps."""
    e = mainenv
    e.rv["retrain"] = {"decision": "HOLD", "cv_f1_mean": None}
    with pytest.raises(TypeError):
        e.main("--force")
    assert not (e.mlops / "loop_state.json").exists()
    assert read_history(e) == []


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


def test_main_runs_seen_is_appended_without_dedupe(mainenv):
    e = mainenv
    seed_state(e, runs_seen=["dup"])
    e.rv["new_runs"] = ["dup", "dup"]
    e.main("--force")
    assert state_of(e)["runs_seen"] == ["dup", "dup", "dup"]


def test_main_consumes_new_data_signal_even_when_training_failed(mainenv):
    """BUG pinned: last_dataset_size is advanced unconditionally inside the
    retrain branch. When train.py fails (metrics None) the accumulated
    new-pair delta is thrown away, so the next tick sees +0 pairs and will not
    retry until another --min-new-pairs arrive."""
    e = mainenv
    seed_state(e, last_dataset_size=0)
    e.rv["retrain"] = None
    e.rv["rebuild"] = 900
    e.main("--force")
    assert state_of(e)["last_dataset_size"] == 900


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


def test_main_partial_state_file_raises_key_error(mainenv):
    """BUG pinned: _load_state tolerates any JSON that parses, but main()
    immediately indexes state['last_run'] / ['last_dataset_size'] /
    ['runs_seen'] / ['total_promotions'] with []. A state file truncated to
    ``{}`` (or written by an older schema) kills the nightly on line one."""
    e = mainenv
    (e.mlops / "loop_state.json").write_text("{}")
    with pytest.raises(KeyError):
        e.main()


def test_main_list_state_file_raises_type_error(mainenv):
    e = mainenv
    (e.mlops / "loop_state.json").write_text("[]")
    with pytest.raises(TypeError):
        e.main()


# --- history ------------------------------------------------------------------

def test_main_history_record_shape(mainenv):
    e = mainenv
    e.rv["new_runs"] = ["a", "b"]
    write_dataset(e, 42)
    e.main()
    rec = read_history(e)[0]
    assert set(rec) == {"run_at", "new_runs", "dataset_size", "retrained",
                        "promoted", "train_metrics", "dry_run", "failed_steps"}
    assert rec["failed_steps"] == []
    assert rec["new_runs"] == 2
    assert rec["dataset_size"] == 42
    assert rec["retrained"] is False
    assert rec["promoted"] is False
    assert rec["train_metrics"] is None
    assert rec["dry_run"] is False


def test_main_history_is_written_even_in_dry_run(mainenv):
    e = mainenv
    e.main("--dry-run")
    assert read_history(e)[0]["dry_run"] is True


def test_main_history_dataset_size_is_the_pre_rebuild_count(mainenv):
    """BUG pinned: history records ``current_size`` (measured before the
    rebuild) while state records the post-rebuild size. loop_history.jsonl
    therefore under-reports the dataset on exactly the nights that grew it."""
    e = mainenv
    write_dataset(e, 10)
    e.rv["rebuild"] = 999
    e.main("--force")
    assert read_history(e)[0]["dataset_size"] == 10
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


def test_main_decay_every_zero_crashes_with_zero_division(mainenv):
    """BUG pinned: --decay-every 0 is accepted by argparse and reaches
    ``loop_count % 0``. An unattended run dies before decay, live-evo, monitor,
    drift, SFT, state persist and history append."""
    e = mainenv
    with pytest.raises(ZeroDivisionError):
        e.main("--decay-every", "0")
    assert read_history(e) == []


def test_main_decay_receives_db_and_dry_run(mainenv):
    e = mainenv
    e.main("--db", "/tmp/x.db", "--dry-run")
    assert e.calls["decay"][0][0] == ("/tmp/x.db", True)


# --- always-on steps ----------------------------------------------------------

def test_main_live_evo_monitor_always_run_regardless_of_ollama(mainenv):
    e = mainenv
    e.rv["ollama_ok"] = False
    e.main("--hook-state", "/hooks")
    assert e.calls["live_evo"][0][0] == (loop.DEFAULT_DB, "/hooks", False)
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


def test_main_sft_bake_is_still_attempted_in_dry_run(mainenv):
    """--dry-run is passed *down* into _run_sft_bake rather than gating it; the
    step runs, only the state write is suppressed (by not saving state at all)."""
    e = mainenv
    e.main("--dry-run")
    assert e.calls["sft"][0][0] == (loop.DEFAULT_OLLAMA, True)
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


def test_main_embedding_trigger_is_emitted_even_in_dry_run(mainenv):
    """Unlike every other write, _emit_embedding_trigger is called before the
    dry-run check; only the last_embedding_tune bookkeeping is suppressed."""
    e = mainenv
    e.main("--dry-run")
    assert len(e.calls["emit"]) == 1
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


def test_main_active_learn_runs_in_dry_run_without_dry_run_flag(mainenv):
    """_run_active_learn takes no dry_run parameter at all — active_learn.py is
    invoked for real (writing active_candidates.jsonl) on a --dry-run night."""
    e = mainenv
    e.rv["active_learn"] = {"exit_code": 0}
    e.main("--dry-run")
    assert e.calls["active_learn"][0][0] == (loop.DEFAULT_OLLAMA,)


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
    assert captured["hook_state"] == loop.DEFAULT_HOOK_STATE


def test_module_level_defaults_are_expanded_paths():
    assert loop.DEFAULT_FINDINGS.endswith("/dt-loci-*/findings.jsonl")
    assert "~" not in loop.DEFAULT_DB
    assert "~" not in loop.DEFAULT_HOOK_STATE
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
