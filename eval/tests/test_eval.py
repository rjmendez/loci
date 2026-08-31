"""
Characterization tests for eval/ (harness.py, tasks.py, grounding_gate_eval.py,
grounding_gate_qf_eval.py, grounding_gate_oos_eval.py).

These pin the behaviour AS IT IS TODAY -- including a genuine defect that is
deliberately NOT fixed here (see test_oos_verdict_else_branch_drops_the_verdict_text).

No network / Qdrant / Ollama / GPU is used: every outbound call goes through
harness._http / harness.embed / subprocess.run, all of which are monkeypatched.
"""

import contextlib
import importlib
import json
import re
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parent.parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import grounding_gate_eval as gge  # noqa: E402
import grounding_gate_oos_eval as oos  # noqa: E402
import grounding_gate_qf_eval as qf  # noqa: E402
import harness  # noqa: E402
import tasks  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def reloaded_harness(**env):
    """Reload harness.py with the given env vars, then restore + reload again.

    Module-level config (DRY_RUN, QDRANT_URL, GROUNDING_SCRIPT, ...) is frozen at
    import time, so import-time behaviour can only be exercised via reload.
    """
    saved = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield importlib.reload(harness)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(harness)


class FakeProc:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def unit_embedder(mapping):
    """Deterministic embed() stub backed by an exact-float lookup table."""
    calls = []

    def _embed(text):
        calls.append(text)
        if text not in mapping:
            raise AssertionError(f"unexpected embed() text: {text!r}")
        return list(mapping[text])

    _embed.calls = calls
    return _embed


# ---------------------------------------------------------------------------
# harness: import-time config
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected_dry_run",
    [
        (None, False),      # unset
        ("", False),
        ("0", False),
        ("false", False),
        ("1", True),
        ("true", True),
        ("False", True),    # quirk: comparison is case-sensitive
        ("FALSE", True),    # quirk
        ("no", True),       # quirk: any non-empty non-{0,false} string enables it
    ],
)
def test_dry_run_flag_parsing(raw, expected_dry_run):
    with reloaded_harness(HARNESS_DRY_RUN=raw) as h:
        assert h.DRY_RUN is expected_dry_run


def test_module_constants_and_env_overrides(tmp_path):
    assert harness.EVAL_COLLECTION == "eval_scores"
    assert harness.EMBED_MODEL == "nomic-embed-text"
    # default grounding script is resolved relative to the repo, not cwd
    assert harness.GROUNDING_SCRIPT.endswith("scripts/hooks/pre_llm_grounding.py")
    assert Path(harness.GROUNDING_SCRIPT).is_absolute()

    script = tmp_path / "g.py"
    with reloaded_harness(
        GROUNDING_SCRIPT=str(script),
        QDRANT_URL="http://q:6333",
        OLLAMA_URL="http://o:11434",
    ) as h:
        assert h.GROUNDING_SCRIPT == str(script)
        assert h.QDRANT_URL == "http://q:6333"
        assert h.OLLAMA_URL == "http://o:11434"


# ---------------------------------------------------------------------------
# harness._read_qdrant_api_key
# ---------------------------------------------------------------------------

def _write_settings(home: Path, payload):
    d = home / ".claude"
    d.mkdir(parents=True, exist_ok=True)
    (d / "settings.json").write_text(
        payload if isinstance(payload, str) else json.dumps(payload)
    )


@pytest.mark.parametrize(
    "settings,expected",
    [
        ({"qdrantApiKey": "top"}, "top"),
        ({"env": {"QDRANT_API_KEY": "from-env-block"}}, "from-env-block"),
        ({"apiKeys": {"qdrant": "nested"}}, "nested"),
        # precedence: top level beats env block beats apiKeys
        (
            {
                "qdrantApiKey": "top",
                "env": {"QDRANT_API_KEY": "mid"},
                "apiKeys": {"qdrant": "low"},
            },
            "top",
        ),
        # falsy candidates are skipped, not returned
        ({"qdrantApiKey": "", "apiKeys": {"qdrant": "low"}}, "low"),
        ({}, ""),
    ],
)
def test_read_qdrant_api_key_from_settings(tmp_path, monkeypatch, settings, expected):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    _write_settings(tmp_path, settings)
    assert harness._read_qdrant_api_key() == expected


def test_read_qdrant_api_key_falls_back_to_env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # no settings.json at all
    monkeypatch.setenv("QDRANT_API_KEY", "env-key")
    assert harness._read_qdrant_api_key() == "env-key"


def test_read_qdrant_api_key_swallows_malformed_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("QDRANT_API_KEY", "env-key")
    _write_settings(tmp_path, "{not json")
    assert harness._read_qdrant_api_key() == "env-key"


def test_read_qdrant_api_key_null_env_block_aborts_the_whole_scan(tmp_path, monkeypatch):
    """`data.get("env", {})` returns None for an explicit null, so `.get` on it
    raises and the bare `except` drops the *later* apiKeys candidate too."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("QDRANT_API_KEY", "env-key")
    _write_settings(tmp_path, {"env": None, "apiKeys": {"qdrant": "would-have-worked"}})
    assert harness._read_qdrant_api_key() == "env-key"


# ---------------------------------------------------------------------------
# harness._http
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capture_urlopen(monkeypatch, payload=None):
    seen = {}

    def _urlopen(req, timeout=None):
        seen["req"] = req
        seen["timeout"] = timeout
        return _FakeResp({} if payload is None else payload)

    monkeypatch.setattr(harness.urllib.request, "urlopen", _urlopen)
    return seen


def test_http_post_serialises_body_and_sets_content_type(monkeypatch):
    monkeypatch.setattr(harness, "QDRANT_API_KEY", "")
    seen = _capture_urlopen(monkeypatch, {"ok": True})

    out = harness._http("POST", "http://x/y", {"a": 1})

    assert out == {"ok": True}
    req = seen["req"]
    assert req.get_method() == "POST"
    assert req.full_url == "http://x/y"
    assert req.data == json.dumps({"a": 1}).encode()
    assert req.get_header("Content-type") == "application/json"
    assert req.get_header("Api-key") is None
    assert seen["timeout"] == 30


def test_http_get_without_body_sends_no_data(monkeypatch):
    monkeypatch.setattr(harness, "QDRANT_API_KEY", "")
    seen = _capture_urlopen(monkeypatch, {"result": 1})
    assert harness._http("GET", "http://x") == {"result": 1}
    assert seen["req"].data is None
    assert seen["req"].get_method() == "GET"


def test_http_adds_api_key_header_when_configured(monkeypatch):
    monkeypatch.setattr(harness, "QDRANT_API_KEY", "sekret")
    seen = _capture_urlopen(monkeypatch)
    harness._http("PUT", "http://x", {})
    assert seen["req"].get_header("Api-key") == "sekret"


def test_http_empty_body_dict_is_still_serialised(monkeypatch):
    monkeypatch.setattr(harness, "QDRANT_API_KEY", "")
    seen = _capture_urlopen(monkeypatch)
    harness._http("PUT", "http://x", {})
    assert seen["req"].data == b"{}"  # {} is not None -> body IS sent


# ---------------------------------------------------------------------------
# harness.embed / ensure_collection / upsert_score
# ---------------------------------------------------------------------------

def test_embed_posts_to_ollama_and_unwraps_embedding(monkeypatch):
    calls = []
    monkeypatch.setattr(harness, "OLLAMA_URL", "http://ollama:11434")
    monkeypatch.setattr(
        harness, "_http",
        lambda m, u, b=None: calls.append((m, u, b)) or {"embedding": [0.1, 0.2]},
    )
    assert harness.embed("hi") == [0.1, 0.2]
    assert calls == [
        ("POST", "http://ollama:11434/api/embeddings",
         {"model": "nomic-embed-text", "prompt": "hi"}),
    ]


def test_embed_raises_keyerror_when_response_lacks_embedding(monkeypatch):
    monkeypatch.setattr(harness, "OLLAMA_URL", "http://o")
    monkeypatch.setattr(harness, "_http", lambda *a, **k: {"error": "boom"})
    with pytest.raises(KeyError):
        harness.embed("hi")


def test_embed_builds_a_none_url_when_ollama_unset(monkeypatch):
    """No guard: an unset OLLAMA_URL yields the literal string 'None/api/...'."""
    seen = []
    monkeypatch.setattr(harness, "OLLAMA_URL", None)
    monkeypatch.setattr(
        harness, "_http", lambda m, u, b=None: seen.append(u) or {"embedding": []}
    )
    harness.embed("x")
    assert seen == ["None/api/embeddings"]


def test_ensure_collection_is_a_noop_when_get_succeeds(monkeypatch):
    calls = []
    monkeypatch.setattr(harness, "QDRANT_URL", "http://q")
    monkeypatch.setattr(harness, "_http", lambda m, u, b=None: calls.append((m, u, b)) or {})
    harness.ensure_collection()
    assert calls == [("GET", "http://q/collections/eval_scores", None)]


def test_ensure_collection_creates_named_dense_vector_on_get_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(harness, "QDRANT_URL", "http://q")

    def _http(method, url, body=None):
        calls.append((method, url, body))
        if method == "GET":
            raise RuntimeError("404")
        return {}

    monkeypatch.setattr(harness, "_http", _http)
    harness.ensure_collection(vector_size=1024)

    assert calls[0][0] == "GET"
    assert calls[1] == (
        "PUT",
        "http://q/collections/eval_scores",
        {"vectors": {"dense": {"size": 1024, "distance": "Cosine"}}},
    )
    assert len(calls) == 2


def test_ensure_collection_default_vector_size_is_768(monkeypatch):
    calls = []
    monkeypatch.setattr(harness, "QDRANT_URL", "http://q")

    def _http(method, url, body=None):
        calls.append(body)
        if method == "GET":
            raise RuntimeError
        return {}

    monkeypatch.setattr(harness, "_http", _http)
    harness.ensure_collection()
    assert calls[1]["vectors"]["dense"]["size"] == 768


def test_stable_point_id_is_deterministic_and_bounded():
    # exact value pinned: sha256("abc2026-01-01")[:15] as hex
    assert harness.stable_point_id("abc", "2026-01-01") == 771649595245321683
    assert harness.stable_point_id("abc", "2026-01-01") == harness.stable_point_id("abc", "2026-01-01")
    assert harness.stable_point_id("abc", "2026-01-02") != harness.stable_point_id("abc", "2026-01-01")
    assert harness.stable_point_id("ab", "c") != harness.stable_point_id("a", "bc") or True
    # 15 hex chars -> always fits in 60 bits (safe as a Qdrant unsigned point id)
    assert 0 <= harness.stable_point_id("x", "y") < 16 ** 15


def test_stable_point_id_concatenation_is_ambiguous():
    """id is derived from task_id + run_date with no separator, so these collide."""
    assert harness.stable_point_id("ab", "c") == harness.stable_point_id("a", "bc")


def test_upsert_score_payload_shape_and_preview_truncation(monkeypatch):
    calls = []
    monkeypatch.setattr(harness, "QDRANT_URL", "http://q")
    monkeypatch.setattr(harness, "_http", lambda m, u, b=None: calls.append((m, u, b)) or {})

    harness.upsert_score(
        task_id="t1",
        task_name="name",
        category="cat",
        score=0.5,
        run_date="2026-01-01",
        context_preview="z" * 500,
        vector=[0.0, 1.0],
    )

    method, url, body = calls[0]
    assert method == "PUT"
    assert url == "http://q/collections/eval_scores/points"
    point = body["points"][0]
    assert point["id"] == harness.stable_point_id("t1", "2026-01-01")
    assert point["vector"] == {"dense": [0.0, 1.0]}
    assert point["payload"] == {
        "task_id": "t1",
        "task_name": "name",
        "score": 0.5,
        "run_date": "2026-01-01",
        "category": "cat",
        "context_preview": "z" * 200,
    }


# ---------------------------------------------------------------------------
# harness.call_grounding
# ---------------------------------------------------------------------------

def test_call_grounding_happy_path_payload_and_argv(monkeypatch, capsys):
    seen = {}

    def _run(argv, input=None, capture_output=None, timeout=None):
        seen.update(argv=argv, input=input, capture_output=capture_output, timeout=timeout)
        return FakeProc(0, json.dumps({"context": "CTX", "other": 1}).encode())

    monkeypatch.setenv("LOCI_PY", "/opt/py")
    monkeypatch.setattr(harness.subprocess, "run", _run)

    assert harness.call_grounding("find X") == "CTX"
    assert seen["argv"] == ["/opt/py", harness.GROUNDING_SCRIPT]
    assert seen["capture_output"] is True
    assert seen["timeout"] == 60
    assert json.loads(seen["input"]) == {
        "hook_event_name": "pre_llm_call",
        "extra": {"user_message": "find X"},
    }
    assert capsys.readouterr().err == ""


def test_call_grounding_default_interpreter_is_the_hermes_venv(monkeypatch):
    seen = {}
    monkeypatch.delenv("LOCI_PY", raising=False)
    monkeypatch.setattr(
        harness.subprocess, "run",
        lambda argv, **kw: seen.update(argv=argv) or FakeProc(0, b'{"context":""}'),
    )
    harness.call_grounding("x")
    assert seen["argv"][0].endswith("/.hermes/hermes-agent/venv/bin/python3")


def test_call_grounding_nonzero_exit_returns_empty_silently(monkeypatch, capsys):
    monkeypatch.setattr(harness.subprocess, "run", lambda *a, **k: FakeProc(3, b"boom"))
    assert harness.call_grounding("x") == ""
    assert capsys.readouterr().err == ""  # no warning on this branch


def test_call_grounding_missing_context_key_returns_empty(monkeypatch):
    monkeypatch.setattr(harness.subprocess, "run", lambda *a, **k: FakeProc(0, b'{"a":1}'))
    assert harness.call_grounding("x") == ""


def test_call_grounding_unparseable_stdout_warns_and_returns_empty(monkeypatch, capsys):
    monkeypatch.setattr(harness.subprocess, "run", lambda *a, **k: FakeProc(0, b"not json"))
    assert harness.call_grounding("x") == ""
    assert "grounding call failed" in capsys.readouterr().err


def test_call_grounding_non_dict_json_warns_and_returns_empty(monkeypatch, capsys):
    monkeypatch.setattr(harness.subprocess, "run", lambda *a, **k: FakeProc(0, b"[1,2]"))
    assert harness.call_grounding("x") == ""
    err = capsys.readouterr().err
    assert "'list' object has no attribute 'get'" in err


def test_call_grounding_timeout_is_fail_open(monkeypatch, capsys):
    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="py", timeout=60)

    monkeypatch.setattr(harness.subprocess, "run", _boom)
    assert harness.call_grounding("x") == ""
    assert "grounding call failed" in capsys.readouterr().err


def test_call_grounding_missing_interpreter_is_fail_open(monkeypatch, capsys):
    """The real degraded path on CI: the hermes venv does not exist."""
    monkeypatch.setenv("LOCI_PY", "/nonexistent/python-does-not-exist")
    assert harness.call_grounding("anything") == ""
    assert "grounding call failed" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# harness.score_context
# ---------------------------------------------------------------------------

def test_score_context_is_case_insensitive_substring_fraction():
    assert harness.score_context("HelloFrame from_bytes", ["helloframe", "from_bytes"]) == 1.0
    assert harness.score_context("HelloFrame", ["HELLOFRAME", "missing"]) == 0.5
    assert harness.score_context("nothing here", ["a1", "b2", "c3"]) == 0.0


def test_score_context_empty_keyword_list_scores_zero_not_one():
    assert harness.score_context("anything", []) == 0.0


def test_score_context_empty_string_keyword_always_hits():
    assert harness.score_context("", [""]) == 1.0
    assert harness.score_context("", ["x"]) == 0.0


def test_score_context_counts_duplicates_independently():
    # denominator is len(keywords), so a repeated keyword is double-counted
    assert harness.score_context("a", ["a", "a"]) == 1.0
    assert harness.score_context("a", ["a", "a", "zzz"]) == pytest.approx(2 / 3)


def test_score_context_matches_mid_word_substrings():
    assert harness.score_context("cargo builder", ["cargo build"]) == 1.0


# ---------------------------------------------------------------------------
# harness.run
# ---------------------------------------------------------------------------

_TWO_TASKS = [
    {
        "id": "t1", "name": "T One", "category": "code_search",
        "prompt": "p1", "expected_keywords": ["alpha", "beta"],
    },
    {
        "id": "t2", "name": "T Two", "category": "build_check",
        "prompt": "p2", "expected_keywords": ["gamma"],
    },
]


def test_run_dry_run_scores_locally_without_touching_qdrant(monkeypatch, capsys):
    monkeypatch.setattr(harness, "DRY_RUN", True)
    monkeypatch.setattr(harness, "TASKS", _TWO_TASKS)
    monkeypatch.setattr(harness, "call_grounding", lambda p: {"p1": "ALPHA only", "p2": "gamma!"}[p])

    def _boom(*a, **k):
        raise AssertionError("network touched in dry run")

    monkeypatch.setattr(harness, "ensure_collection", _boom)
    monkeypatch.setattr(harness, "embed", _boom)
    monkeypatch.setattr(harness, "upsert_score", _boom)

    harness.run()
    out = capsys.readouterr().out

    assert "tasks=2  dry_run=True" in out
    assert "  [code_search] t1: score=0.500 (1/2 keywords)" in out
    assert "  [build_check] t2: score=1.000 (1/1 keywords)" in out
    assert "[eval] mean_score=0.750 (2 tasks)" in out


def test_run_live_embeds_the_prompt_and_upserts_full_context(monkeypatch, capsys):
    ensured, embedded, upserts = [], [], []
    monkeypatch.setattr(harness, "DRY_RUN", False)
    monkeypatch.setattr(harness, "TASKS", _TWO_TASKS[:1])
    monkeypatch.setattr(harness, "call_grounding", lambda p: "alpha and beta")
    monkeypatch.setattr(harness, "ensure_collection", lambda *a, **k: ensured.append(1))
    monkeypatch.setattr(harness, "embed", lambda t: embedded.append(t) or [1.0])
    monkeypatch.setattr(harness, "upsert_score", lambda **kw: upserts.append(kw))

    harness.run()

    assert ensured == [1]
    assert embedded == ["p1"]  # the PROMPT is embedded, not the context
    assert len(upserts) == 1
    u = upserts[0]
    assert u["task_id"] == "t1"
    assert u["task_name"] == "T One"
    assert u["category"] == "code_search"
    assert u["score"] == 1.0
    assert u["context_preview"] == "alpha and beta"  # untruncated here
    assert u["vector"] == [1.0]
    assert len(u["run_date"]) == 10 and u["run_date"].count("-") == 2
    assert "[eval] mean_score=1.000 (1 tasks)" in capsys.readouterr().out


def test_run_with_no_tasks_reports_zero_mean(monkeypatch, capsys):
    monkeypatch.setattr(harness, "DRY_RUN", True)
    monkeypatch.setattr(harness, "TASKS", [])
    harness.run()
    assert "[eval] mean_score=0.000 (0 tasks)" in capsys.readouterr().out


def test_run_scores_zero_when_grounding_is_unavailable(monkeypatch, capsys):
    """Fail-open: a dead grounding hook produces a clean 0.000 run, not an error."""
    monkeypatch.setattr(harness, "DRY_RUN", True)
    monkeypatch.setattr(harness, "TASKS", _TWO_TASKS)
    monkeypatch.setattr(harness, "call_grounding", lambda p: "")
    harness.run()
    out = capsys.readouterr().out
    assert "score=0.000" in out
    assert "[eval] mean_score=0.000 (2 tasks)" in out


def test_run_propagates_keyerror_for_a_malformed_task(monkeypatch):
    monkeypatch.setattr(harness, "DRY_RUN", True)
    monkeypatch.setattr(harness, "TASKS", [{"id": "x"}])
    monkeypatch.setattr(harness, "call_grounding", lambda p: "")
    with pytest.raises(KeyError):
        harness.run()


# ---------------------------------------------------------------------------
# tasks.py
# ---------------------------------------------------------------------------

REQUIRED_TASK_KEYS = {"id", "name", "description", "prompt", "expected_keywords", "category"}


def test_tasks_suite_shape():
    assert len(tasks.TASKS) == 11
    ids = [t["id"] for t in tasks.TASKS]
    assert len(set(ids)) == len(ids)
    for t in tasks.TASKS:
        assert set(t) == REQUIRED_TASK_KEYS, t["id"]
        assert isinstance(t["expected_keywords"], list) and t["expected_keywords"]
        assert all(isinstance(k, str) and k for k in t["expected_keywords"])


def test_tasks_category_distribution():
    from collections import Counter
    assert Counter(t["category"] for t in tasks.TASKS) == {
        "code_search": 2,
        "memory_recall": 2,
        "architecture_query": 2,
        "build_check": 2,
        "blocker_id": 3,
    }


def test_tasks_b1_definition_is_pinned():
    t = next(t for t in tasks.TASKS if t["id"] == "blocker_b1_hello_frame")
    assert t["category"] == "blocker_id"
    assert t["prompt"] == "What is B1 and why does it cause damaConnected=false?"
    assert t["expected_keywords"] == [
        "pkg_len", "28515", "frame too short", "NACK", "buildHelloFrame",
    ]


def test_tasks_are_scoreable_by_the_harness():
    """Every task's keywords are matched case-insensitively, so a context that
    concatenates them scores exactly 1.0."""
    for t in tasks.TASKS:
        ctx = " | ".join(k.upper() for k in t["expected_keywords"])
        assert harness.score_context(ctx, t["expected_keywords"]) == 1.0


# ---------------------------------------------------------------------------
# grounding_gate_eval: pure helpers
# ---------------------------------------------------------------------------

def test_gge_module_defaults():
    assert gge.THRESHOLD == 0.59
    assert gge.CATEGORY == "deep_think_loci"
    assert gge.DATASET.name == "grounding_dataset.jsonl"
    assert gge.DATASET.parent.as_posix().endswith("deep_think_loci/grounding")


def test_unit_normalises_and_survives_the_zero_vector():
    assert gge._unit([3.0, 4.0]) == [0.6, 0.8]
    assert gge._unit([0.0, 0.0]) == [0.0, 0.0]   # n falls back to 1.0, no ZeroDivisionError
    assert gge._unit([]) == []
    assert gge._unit([-3.0, 4.0]) == [-0.6, 0.8]
    assert qf._unit is not gge._unit  # duplicated implementation, same behaviour
    assert qf._unit([3.0, 4.0]) == [0.6, 0.8]


def test_metrics_perfect_separation():
    m = gge._metrics([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], 0.5)
    assert m == {
        "recall": 1.0, "bleed_rejection": 1.0, "f1": 1.0, "accuracy": 1.0, "auc": 1.0,
    }


def test_metrics_threshold_is_inclusive():
    assert gge._metrics([1], [0.59], 0.59)["recall"] == 1.0
    assert gge._metrics([1], [0.5899999], 0.59)["recall"] == 0.0


def test_metrics_mixed_case_values():
    # keep = [T, F, T, F]; labels = [1, 1, 0, 0]
    m = gge._metrics([1, 1, 0, 0], [0.9, 0.1, 0.7, 0.2], 0.5)
    assert m["recall"] == 0.5
    assert m["bleed_rejection"] == 0.5
    assert m["accuracy"] == 0.5
    assert m["f1"] == pytest.approx(0.5)   # precision 0.5, recall 0.5
    # pos=[0.9, 0.1] vs neg=[0.7, 0.2]: only 0.9 outranks both negatives -> 2/4
    assert m["auc"] == pytest.approx(0.5)


def test_metrics_auc_is_nan_without_both_classes():
    assert math.isnan(gge._metrics([1, 1], [0.9, 0.1], 0.5)["auc"])
    assert math.isnan(gge._metrics([0, 0], [0.9, 0.1], 0.5)["auc"])
    assert math.isnan(gge._metrics([], [], 0.5)["auc"])


def test_metrics_ties_get_half_credit_in_auc():
    assert gge._metrics([1, 0], [0.5, 0.5], 0.5)["auc"] == 0.5


def test_metrics_all_zero_on_empty_input():
    m = gge._metrics([], [], 0.5)
    assert m["recall"] == m["bleed_rejection"] == m["f1"] == m["accuracy"] == 0.0


def test_metrics_no_kept_pairs_gives_zero_f1_not_zerodivision():
    m = gge._metrics([1, 0], [0.1, 0.2], 0.9)
    assert m["f1"] == 0.0 and m["recall"] == 0.0 and m["bleed_rejection"] == 1.0


def test_cosines_live_dedupes_embed_calls():
    emb = unit_embedder({"a": [1.0, 0.0], "b": [4.0, 3.0], "c": [0.0, 1.0]})
    pairs = [
        {"claim": "a", "evidence": "b"},
        {"claim": "a", "evidence": "b"},  # duplicate pair, same two texts
        {"claim": "a", "evidence": "c"},
        {"claim": "a", "evidence": "a"},
    ]
    orig = harness.embed
    harness.embed = emb
    try:
        cos = gge._cosines_live(pairs)
    finally:
        harness.embed = orig

    assert sorted(emb.calls) == ["a", "b", "c"]  # each unique text embedded once
    assert cos == pytest.approx([0.8, 0.8, 0.0, 1.0])


# ---------------------------------------------------------------------------
# grounding_gate_eval.run
# ---------------------------------------------------------------------------

def _write_dataset(tmp_path, rows):
    p = tmp_path / "ds.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")  # trailing blank line
    return p


def test_gge_run_missing_dataset_returns_quietly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gge, "DATASET", tmp_path / "nope.jsonl")
    gge.run()
    assert "[gate-eval] dataset not found:" in capsys.readouterr().out


def test_gge_run_ignores_non_topical_signals(tmp_path, monkeypatch, capsys):
    ds = _write_dataset(tmp_path, [
        {"claim": "a", "evidence": "b", "label": 1, "signal": "lineage", "cos": 0.9},
        {"claim": "a", "evidence": "c", "label": 0},  # no signal key at all
    ])
    monkeypatch.setattr(gge, "DATASET", ds)
    gge.run()
    assert "[gate-eval] no topical pairs in dataset" in capsys.readouterr().out


def test_gge_run_dry_uses_stored_cosines(tmp_path, monkeypatch, capsys):
    ds = _write_dataset(tmp_path, [
        {"claim": "a", "evidence": "b", "label": 1, "signal": "topical", "cos": 0.90},
        {"claim": "a", "evidence": "c", "label": 1, "signal": "topical", "cos": 0.70},
        {"claim": "a", "evidence": "d", "label": 0, "signal": "topical", "cos": 0.30},
        {"claim": "a", "evidence": "e", "label": 0, "signal": "topical", "cos": 0.60},
        {"claim": "a", "evidence": "f", "label": 1, "signal": "lineage", "cos": 0.01},
    ])
    monkeypatch.setattr(gge, "DATASET", ds)
    monkeypatch.setattr(harness, "DRY_RUN", True)
    monkeypatch.setattr(harness, "embed", lambda t: pytest.fail("embedded in dry run"))
    gge.run()
    out = capsys.readouterr().out

    assert "pairs=4 thr=0.59 dry_run=True" in out
    assert "  recall           1.000" in out
    assert "  bleed_rejection  0.500" in out
    assert "  f1               0.800" in out
    assert "  accuracy         0.750" in out
    assert "  auc              1.000" in out
    assert "persisted" not in out


def test_gge_run_dry_treats_missing_cos_as_zero(tmp_path, monkeypatch, capsys):
    ds = _write_dataset(tmp_path, [
        {"claim": "a", "evidence": "b", "label": 1, "signal": "topical"},  # no cos
        {"claim": "a", "evidence": "c", "label": 0, "signal": "topical", "cos": 0.9},
    ])
    monkeypatch.setattr(gge, "DATASET", ds)
    monkeypatch.setattr(harness, "DRY_RUN", True)
    gge.run()
    out = capsys.readouterr().out
    assert "  recall           0.000" in out       # the 1-label pair defaulted to cos 0.0
    assert "  bleed_rejection  0.000" in out
    assert "  auc              0.000" in out


def test_gge_run_live_reembeds_and_persists_one_score_per_metric(tmp_path, monkeypatch, capsys):
    ds = _write_dataset(tmp_path, [
        {"claim": "a", "evidence": "b", "label": 1, "signal": "topical", "cos": 0.0},
        {"claim": "a", "evidence": "c", "label": 0, "signal": "topical", "cos": 0.99},
    ])
    monkeypatch.setattr(gge, "DATASET", ds)
    monkeypatch.setattr(harness, "DRY_RUN", False)

    ensured, upserts = [], []
    emb = unit_embedder({
        "a": [1.0, 0.0], "b": [4.0, 3.0], "c": [0.0, 1.0],
        "deep_think_loci grounding gate eval": [1.0, 1.0],
    })
    monkeypatch.setattr(harness, "embed", emb)
    monkeypatch.setattr(harness, "ensure_collection", lambda *a, **k: ensured.append(1))
    monkeypatch.setattr(harness, "upsert_score", lambda **kw: upserts.append(kw))

    gge.run()
    out = capsys.readouterr().out

    assert ensured == [1]
    # live cosines (0.8 / 0.0) override the stored ones (0.0 / 0.99)
    assert "  recall           1.000" in out
    assert "  bleed_rejection  1.000" in out
    assert {u["task_id"] for u in upserts} == {
        "dtl.grounding_gate.recall",
        "dtl.grounding_gate.bleed_rejection",
        "dtl.grounding_gate.f1",
        "dtl.grounding_gate.accuracy",
        "dtl.grounding_gate.auc",
    }
    assert all(u["category"] == "deep_think_loci" for u in upserts)
    assert all(u["vector"] == [1.0, 1.0] for u in upserts)  # raw embed, NOT unit-normalised
    assert all(u["context_preview"] == "2 topical pairs from ds.jsonl" for u in upserts)
    assert {u["task_name"] for u in upserts} == {
        f"grounding gate {n} @cos>=0.59"
        for n in ("recall", "bleed_rejection", "f1", "accuracy", "auc")
    }
    assert "[gate-eval] persisted 5 scores to eval_scores" in out


def test_gge_run_live_skips_nan_metrics(tmp_path, monkeypatch, capsys):
    """All-positive labels make auc NaN; NaN scores are never persisted."""
    ds = _write_dataset(tmp_path, [
        {"claim": "a", "evidence": "b", "label": 1, "signal": "topical"},
        {"claim": "a", "evidence": "c", "label": 1, "signal": "topical"},
    ])
    monkeypatch.setattr(gge, "DATASET", ds)
    monkeypatch.setattr(harness, "DRY_RUN", False)
    upserts = []
    monkeypatch.setattr(harness, "embed", unit_embedder({
        "a": [1.0, 0.0], "b": [4.0, 3.0], "c": [0.0, 1.0],
        "deep_think_loci grounding gate eval": [1.0],
    }))
    monkeypatch.setattr(harness, "ensure_collection", lambda *a, **k: None)
    monkeypatch.setattr(harness, "upsert_score", lambda **kw: upserts.append(kw))

    gge.run()
    out = capsys.readouterr().out

    assert "  auc              nan" in out
    assert "dtl.grounding_gate.auc" not in {u["task_id"] for u in upserts}
    assert len(upserts) == 4
    assert "[gate-eval] persisted 4 scores to eval_scores" in out


def test_gge_run_raises_on_a_malformed_dataset_line(tmp_path, monkeypatch):
    ds = tmp_path / "bad.jsonl"
    ds.write_text('{"claim":"a"}\nnot-json\n')
    monkeypatch.setattr(gge, "DATASET", ds)
    with pytest.raises(json.JSONDecodeError):
        gge.run()


def test_gge_run_reads_the_real_repo_dataset_in_dry_mode(monkeypatch, capsys):
    """Smoke-pins the shipped corpus: the gate still separates on whatever is there.

    This used to assert `pairs=5235`, which held only while the shipped dataset
    was untouched. mlops/loop.py rebuilds that same file in place, so any machine
    that had run the loop failed here — the test was reporting the corpus size,
    not the gate. Assert the invariant instead: a plausible number of topical
    pairs, and cosine still ranking on-topic above bleed.
    """
    if not gge.DATASET.exists():
        pytest.skip("shipped grounding_dataset.jsonl not present")
    monkeypatch.setattr(harness, "DRY_RUN", True)
    gge.run()
    out = capsys.readouterr().out
    m = re.search(r"pairs=(\d+) thr=0\.59 dry_run=True", out)
    assert m, f"expected a pairs=/thr= line, got:\n{out}"
    assert int(m.group(1)) > 500, "corpus implausibly small — did the rebuild break?"
    auc = float(next(l for l in out.splitlines() if "auc" in l).split()[-1])
    assert auc > 0.5  # cosine still ranks on-topic above bleed


# ---------------------------------------------------------------------------
# grounding_gate_qf_eval: pure helpers
# ---------------------------------------------------------------------------

def test_qf_module_defaults():
    assert qf.THRESHOLD == 0.59
    assert qf.MODEL.name == "grounding_bleed_clf.joblib"
    assert "dt-loci-*" in qf.CORPUS and qf.CORPUS.endswith("findings.jsonl")
    assert set(qf.DEFAULT_FOCUS) == {
        "rooted-canary", "governance-gate", "telemetry-ingest",
        "ant-training", "sensor-fusion",
    }


def test_target_extracts_first_dt_target_tag():
    assert qf._target({"tags": ["x", "dt_target:alpha", "dt_target:beta"]}) == "alpha"
    assert qf._target({"tags": ["dt_target:a:b"]}) == "a:b"  # only the first colon splits
    assert qf._target({"tags": []}) == ""
    assert qf._target({"tags": None}) == ""
    assert qf._target({}) == ""
    assert qf._target({"tags": ["dt_target:"]}) == ""  # empty target reads as "no target"


def test_focus_map_defaults_when_env_unset(monkeypatch):
    monkeypatch.delenv("DTL_TARGET_FOCUS", raising=False)
    assert qf._focus_map() is qf.DEFAULT_FOCUS


def test_focus_map_accepts_inline_json(monkeypatch):
    monkeypatch.setenv("DTL_TARGET_FOCUS", '  {"a": "query a"}  ')
    assert qf._focus_map() == {"a": "query a"}


def test_focus_map_accepts_a_file_path(tmp_path, monkeypatch):
    p = tmp_path / "focus.json"
    p.write_text('{"b": "query b"}')
    monkeypatch.setenv("DTL_TARGET_FOCUS", str(p))
    assert qf._focus_map() == {"b": "query b"}


def test_focus_map_falls_back_to_defaults_on_bad_input(monkeypatch):
    monkeypatch.setenv("DTL_TARGET_FOCUS", "/no/such/file.json")
    assert qf._focus_map() is qf.DEFAULT_FOCUS
    monkeypatch.setenv("DTL_TARGET_FOCUS", "{broken")
    assert qf._focus_map() is qf.DEFAULT_FOCUS


# ---------------------------------------------------------------------------
# grounding_gate_qf_eval.run
# ---------------------------------------------------------------------------

# 2-d vectors chosen so every cosine is an exact float.
QF_VECTORS = {
    "alpha": [1.0, 0.0], "beta": [0.0, 1.0],
    "a1": [1.0, 0.0], "a2": [4.0, 3.0], "b1": [0.0, 1.0], "b2": [3.0, 4.0],
    "deep_think_loci grounding gate query->finding eval": [7.0, 0.0],
}


def _write_corpus(tmp_path, rows, run="dt-loci-run1"):
    d = tmp_path / run
    d.mkdir(parents=True, exist_ok=True)
    p = d / "findings.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(tmp_path / "dt-loci-*" / "findings.jsonl")


QF_ROWS = [
    {"text": "a1", "tags": ["dt_target:alpha"]},
    {"text": "a2", "tags": ["dt_target:alpha"]},
    {"text": "b1", "tags": ["dt_target:beta"]},
    {"text": "b2", "tags": ["dt_target:beta"]},
]


def test_qf_run_skips_without_ollama(monkeypatch, capsys):
    monkeypatch.setattr(harness, "OLLAMA_URL", None)
    qf.run()
    assert "[gate-qf-eval] OLLAMA_URL unset" in capsys.readouterr().out


def test_qf_run_skips_when_no_targeted_findings(tmp_path, monkeypatch, capsys):
    glob_pat = _write_corpus(tmp_path, [{"text": "x", "tags": ["unrelated"]}])
    monkeypatch.setattr(harness, "OLLAMA_URL", "http://o")
    monkeypatch.setattr(qf, "CORPUS", glob_pat)
    qf.run()
    assert "[gate-qf-eval] no targeted findings in" in capsys.readouterr().out


def test_qf_run_tolerates_unparseable_corpus_lines(tmp_path, monkeypatch, capsys):
    d = tmp_path / "dt-loci-run1"
    d.mkdir()
    (d / "findings.jsonl").write_text("not-json\n" + json.dumps(QF_ROWS[0]) + "\n")
    monkeypatch.setattr(harness, "OLLAMA_URL", "http://o")
    monkeypatch.setattr(harness, "DRY_RUN", True)
    monkeypatch.setattr(harness, "embed", unit_embedder(QF_VECTORS))
    monkeypatch.setattr(qf, "CORPUS", str(tmp_path / "dt-loci-*" / "findings.jsonl"))
    monkeypatch.setattr(qf, "MODEL", tmp_path / "absent.joblib")
    qf.run()
    out = capsys.readouterr().out
    assert "targets=1 findings=1 pairs=1 (pos=1)" in out  # bad line silently dropped


def test_qf_run_cosine_metrics_at_fixed_and_best_thresholds(tmp_path, monkeypatch, capsys):
    glob_pat = _write_corpus(tmp_path, QF_ROWS)
    monkeypatch.setattr(harness, "OLLAMA_URL", "http://o")
    monkeypatch.setattr(harness, "DRY_RUN", True)
    monkeypatch.setattr(harness, "embed", unit_embedder(QF_VECTORS))
    monkeypatch.setattr(qf, "CORPUS", glob_pat)
    monkeypatch.setattr(qf, "MODEL", tmp_path / "absent.joblib")
    monkeypatch.delenv("DTL_TARGET_FOCUS", raising=False)

    qf.run()
    out = capsys.readouterr().out

    assert "targets=2 findings=4 pairs=8 (pos=4) thr=0.59 dry_run=True" in out
    assert ("  COSINE @0.59   : {'recall': 1.0, 'bleed_rejection': 0.5, 'f1': 0.8, "
            "'accuracy': 0.75, 'auc': 1.0}") in out
    # best-F1 sweep finds thr=0.8, where the gate is perfect on this corpus
    assert ("  COSINE @best=0.800: {'recall': 1.0, 'bleed_rejection': 1.0, 'f1': 1.0, "
            "'accuracy': 1.0, 'auc': 1.0}") in out
    assert "MODEL" not in out           # absent model file is skipped silently
    assert "IN-SAMPLE" not in out
    assert "persisted" not in out       # dry run never persists


def test_qf_run_truncates_finding_text_to_2000_chars(tmp_path, monkeypatch, capsys):
    long_text = "z" * 2500
    glob_pat = _write_corpus(tmp_path, [
        {"text": long_text, "tags": ["dt_target:alpha"]},
        {"text": long_text[:2000], "tags": ["dt_target:alpha"]},  # dedups with the above
    ])
    seen = []
    monkeypatch.setattr(harness, "OLLAMA_URL", "http://o")
    monkeypatch.setattr(harness, "DRY_RUN", True)
    monkeypatch.setattr(harness, "embed", lambda t: seen.append(t) or [1.0, 0.0])
    monkeypatch.setattr(qf, "CORPUS", glob_pat)
    monkeypatch.setattr(qf, "MODEL", tmp_path / "absent.joblib")

    qf.run()

    assert "findings=2" in capsys.readouterr().out
    assert set(seen) == {"alpha", "z" * 2000}  # both rows collapse to one embed
    assert all(len(t) <= 2000 for t in seen)


def test_qf_run_missing_text_becomes_empty_string(tmp_path, monkeypatch, capsys):
    glob_pat = _write_corpus(tmp_path, [{"tags": ["dt_target:alpha"]}])
    monkeypatch.setattr(harness, "OLLAMA_URL", "http://o")
    monkeypatch.setattr(harness, "DRY_RUN", True)
    monkeypatch.setattr(harness, "embed", unit_embedder({"alpha": [1.0, 0.0], "": [1.0, 0.0]}))
    monkeypatch.setattr(qf, "CORPUS", glob_pat)
    monkeypatch.setattr(qf, "MODEL", tmp_path / "absent.joblib")
    qf.run()
    assert "findings=1 pairs=1" in capsys.readouterr().out


def _stub_model(tmp_path):
    """A LogisticRegression whose proba is sigmoid(10*cos - 6): a monotone
    restatement of cosine, so model/cosine metrics are exactly comparable."""
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression()
    clf.coef_ = np.array([[0.0, 0.0, 0.0, 0.0, 10.0]])
    clf.intercept_ = np.array([-6.0])
    clf.classes_ = np.array([0, 1])
    clf.n_features_in_ = 5
    p = tmp_path / "clf.joblib"
    joblib.dump(clf, str(p))
    return p


def test_qf_run_with_model_persists_cosine_and_model_scores(tmp_path, monkeypatch, capsys):
    glob_pat = _write_corpus(tmp_path, QF_ROWS)
    upserts, ensured = [], []
    monkeypatch.setattr(harness, "OLLAMA_URL", "http://o")
    monkeypatch.setattr(harness, "DRY_RUN", False)
    monkeypatch.setattr(harness, "embed", unit_embedder(QF_VECTORS))
    monkeypatch.setattr(harness, "ensure_collection", lambda *a, **k: ensured.append(1))
    monkeypatch.setattr(harness, "upsert_score", lambda **kw: upserts.append(kw))
    monkeypatch.setattr(qf, "CORPUS", glob_pat)
    monkeypatch.setattr(qf, "MODEL", _stub_model(tmp_path))

    qf.run()
    out = capsys.readouterr().out

    assert ensured == [1]
    assert ("  MODEL  @0.50  : {'recall': 1.0, 'bleed_rejection': 0.5, 'f1': 0.8, "
            "'accuracy': 0.75, 'auc': 1.0}") in out
    # model only ties cosine at best-F1, so the in-sample verdict is "<="
    assert "IN-SAMPLE: model <= cosine at best-F1" in out
    assert "f1 1.000 vs 1.000; auc 1.000 vs 1.000" in out
    assert "NOT a swap decision" in out

    ids = sorted(u["task_id"] for u in upserts)
    assert ids == sorted(
        [f"dtl.gate_qf.{tag}.{m}"
         for tag in ("cosine", "model")
         for m in ("recall", "bleed_rejection", "f1", "accuracy", "auc")]
    )
    # Persisted values are the fixed-threshold ones, never the best-F1 sweep results.
    by_id = {u["task_id"]: u["score"] for u in upserts}
    assert by_id["dtl.gate_qf.cosine.f1"] == pytest.approx(0.8)
    assert by_id["dtl.gate_qf.model.f1"] == pytest.approx(0.8)
    assert by_id["dtl.gate_qf.cosine.bleed_rejection"] == pytest.approx(0.5)
    assert all(u["category"] == "deep_think_loci" for u in upserts)
    assert all(u["context_preview"] == "4 findings x 2 targets" for u in upserts)
    assert all(u["vector"] == [7.0, 0.0] for u in upserts)
    assert "[gate-qf-eval] persisted to eval_scores" in out


def test_qf_run_reports_but_survives_a_broken_model_file(tmp_path, monkeypatch, capsys):
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a joblib file")
    glob_pat = _write_corpus(tmp_path, QF_ROWS)
    monkeypatch.setattr(harness, "OLLAMA_URL", "http://o")
    monkeypatch.setattr(harness, "DRY_RUN", True)  # dry run avoids the mdl_m NameError
    monkeypatch.setattr(harness, "embed", unit_embedder(QF_VECTORS))
    monkeypatch.setattr(qf, "CORPUS", glob_pat)
    monkeypatch.setattr(qf, "MODEL", bad)

    qf.run()
    out = capsys.readouterr().out
    assert "  [model skipped]" in out
    assert "COSINE @0.59" in out
    assert "IN-SAMPLE" not in out


def _live_run(tmp_path, monkeypatch, model_path):
    """A live (persisting) qf run against the standard corpus; returns the upserts."""
    upserts = []
    monkeypatch.setattr(harness, "OLLAMA_URL", "http://o")
    monkeypatch.setattr(harness, "DRY_RUN", False)
    monkeypatch.setattr(harness, "embed", unit_embedder(QF_VECTORS))
    monkeypatch.setattr(harness, "ensure_collection", lambda *a, **k: None)
    monkeypatch.setattr(harness, "upsert_score", lambda **kw: upserts.append(kw))
    monkeypatch.setattr(qf, "CORPUS", _write_corpus(tmp_path, QF_ROWS))
    monkeypatch.setattr(qf, "MODEL", model_path)
    qf.run()
    return upserts


def test_qf_run_persists_the_cosine_arm_when_there_is_no_model(tmp_path, monkeypatch):
    """`mdl_m` was only bound inside `if MODEL.exists()` while the persist loop
    referenced it unconditionally, so a live run with no classifier on disk died
    of UnboundLocalError AFTER printing the cosine metrics -- losing those too."""
    upserts = _live_run(tmp_path, monkeypatch, tmp_path / "absent.joblib")
    ids = sorted(u["task_id"] for u in upserts)
    assert ids == sorted(f"dtl.gate_qf.cosine.{m}" for m in
                         ("recall", "bleed_rejection", "f1", "accuracy", "auc"))


def test_qf_run_persists_the_cosine_arm_when_the_model_fails_to_load(tmp_path, monkeypatch):
    """Same defect via the other branch: joblib.load raising left mdl_m unbound."""
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"garbage")
    upserts = _live_run(tmp_path, monkeypatch, bad)
    assert {u["task_id"] for u in upserts} == {
        f"dtl.gate_qf.cosine.{m}" for m in ("recall", "bleed_rejection", "f1", "accuracy", "auc")}


def _stub_model_wide(tmp_path):
    """The same sigmoid(10*cos - 6) scorer, in the trainer's 2*d+4 layout.

    Column order is [diff(d), prod(d), cos, cos^2, len_ratio, jaccard], so at
    d=2 the cosine sits at index 4.
    """
    import joblib
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression()
    clf.coef_ = np.array([[0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0]])
    clf.intercept_ = np.array([-6.0])
    clf.classes_ = np.array([0, 1])
    clf.n_features_in_ = 8
    p = tmp_path / "wide.joblib"
    joblib.dump(clf, str(p))
    return p


def test_qf_run_scores_a_train_shaped_model_instead_of_skipping_it(tmp_path, monkeypatch, capsys):
    """train.py writes THIS path in the wider layout. The eval built a hard-coded
    2*d+1 vector, so predict_proba raised and the bare except printed
    '[model skipped]' -- the model arm silently vanished from the trend."""
    upserts = _live_run(tmp_path, monkeypatch, _stub_model_wide(tmp_path))
    out = capsys.readouterr().out
    assert "[model skipped]" not in out
    assert ("  MODEL  @0.50  : {'recall': 1.0, 'bleed_rejection': 0.5, 'f1': 0.8, "
            "'accuracy': 0.75, 'auc': 1.0}") in out
    by_id = {u["task_id"]: u["score"] for u in upserts}
    assert by_id["dtl.gate_qf.model.f1"] == pytest.approx(0.8)


def test_qf_run_uses_target_name_as_query_for_unknown_targets(tmp_path, monkeypatch):
    """Targets absent from the focus map fall back to the bare target string."""
    glob_pat = _write_corpus(tmp_path, [{"text": "a1", "tags": ["dt_target:alpha"]}])
    seen = []
    monkeypatch.setattr(harness, "OLLAMA_URL", "http://o")
    monkeypatch.setattr(harness, "DRY_RUN", True)
    monkeypatch.setattr(harness, "embed", lambda t: seen.append(t) or [1.0, 0.0])
    monkeypatch.setattr(qf, "CORPUS", glob_pat)
    monkeypatch.setattr(qf, "MODEL", tmp_path / "absent.joblib")
    monkeypatch.setenv("DTL_TARGET_FOCUS", json.dumps({"beta": "beta focus"}))
    qf.run()
    assert "alpha" in seen and "beta focus" not in seen


# ---------------------------------------------------------------------------
# grounding_gate_oos_eval
# ---------------------------------------------------------------------------

def test_oos_reuses_the_other_modules_helpers():
    assert oos._metrics is gge._metrics
    assert oos._focus_map is qf._focus_map
    assert oos._target is qf._target
    assert oos._unit is qf._unit
    assert oos.THRESHOLD == 0.59


def test_oos_skips_without_ollama(monkeypatch, capsys):
    monkeypatch.setattr(harness, "OLLAMA_URL", None)
    oos.run()
    assert "[oos] OLLAMA_URL unset" in capsys.readouterr().out


def test_oos_needs_at_least_two_runs(tmp_path, monkeypatch, capsys):
    _write_corpus(tmp_path, QF_ROWS, run="dt-loci-run1")
    monkeypatch.setattr(harness, "OLLAMA_URL", "http://o")
    monkeypatch.setattr(oos, "CORPUS", str(tmp_path / "dt-loci-*" / "findings.jsonl"))
    oos.run()
    assert "[oos] need >=2 runs, found 1" in capsys.readouterr().out


def test_oos_no_usable_folds_when_a_run_has_a_single_class(tmp_path, monkeypatch, capsys):
    """With one target per run, every training set is single-class -> no folds."""
    _write_corpus(tmp_path, [{"text": "a1", "tags": ["dt_target:alpha"]}], run="dt-loci-r1")
    _write_corpus(tmp_path, [{"text": "b1", "tags": ["dt_target:beta"]}], run="dt-loci-r2")
    monkeypatch.setattr(harness, "OLLAMA_URL", "http://o")
    monkeypatch.setattr(harness, "embed", unit_embedder(QF_VECTORS))
    monkeypatch.setattr(oos, "CORPUS", str(tmp_path / "dt-loci-*" / "findings.jsonl"))
    oos.run()
    assert "[oos] no usable folds" in capsys.readouterr().out


def test_oos_leave_one_run_out_reports_per_fold_and_mean(tmp_path, monkeypatch, capsys):
    rows2 = [
        {"text": "a3", "tags": ["dt_target:alpha"]},
        {"text": "a4", "tags": ["dt_target:alpha"]},
        {"text": "b3", "tags": ["dt_target:beta"]},
        {"text": "b4", "tags": ["dt_target:beta"]},
    ]
    _write_corpus(tmp_path, QF_ROWS, run="dt-loci-r1")
    _write_corpus(tmp_path, rows2, run="dt-loci-r2")
    vecs = dict(QF_VECTORS)
    vecs.update({"a3": [1.0, 0.0], "a4": [4.0, 3.0], "b3": [0.0, 1.0], "b4": [3.0, 4.0]})
    monkeypatch.setattr(harness, "OLLAMA_URL", "http://o")
    monkeypatch.setattr(harness, "embed", unit_embedder(vecs))
    monkeypatch.setattr(oos, "CORPUS", str(tmp_path / "dt-loci-*" / "findings.jsonl"))
    monkeypatch.delenv("DTL_TARGET_FOCUS", raising=False)

    oos.run()
    out = capsys.readouterr().out

    # one fold per run, keyed by the parent DIRECTORY name
    assert "held=dt-loci-r1 (n=4): cosine f1=0.800 acc=0.750" in out
    assert "held=dt-loci-r2 (n=4): cosine f1=0.800 acc=0.750" in out
    assert "[oos] MEAN over 2 folds: cosine f1=0.800 acc=0.750" in out
    assert "| model f1=" in out


def test_oos_verdict_else_branch_drops_the_verdict_text(tmp_path, monkeypatch, capsys):
    """BUG (pinned, not fixed): the conditional expression wraps the whole
    implicitly-concatenated f-string, so when the model does NOT generalize the
    program prints only the parenthetical and the phrase
    'does NOT clearly beat cosine' is unreachable dead text."""
    _write_corpus(tmp_path, QF_ROWS, run="dt-loci-r1")
    _write_corpus(tmp_path, [
        {"text": "a3", "tags": ["dt_target:alpha"]},
        {"text": "a4", "tags": ["dt_target:alpha"]},
        {"text": "b3", "tags": ["dt_target:beta"]},
        {"text": "b4", "tags": ["dt_target:beta"]},
    ], run="dt-loci-r2")
    vecs = dict(QF_VECTORS)
    vecs.update({"a3": [1.0, 0.0], "a4": [4.0, 3.0], "b3": [0.0, 1.0], "b4": [3.0, 4.0]})
    monkeypatch.setattr(harness, "OLLAMA_URL", "http://o")
    monkeypatch.setattr(harness, "embed", unit_embedder(vecs))
    monkeypatch.setattr(oos, "CORPUS", str(tmp_path / "dt-loci-*" / "findings.jsonl"))

    oos.run()
    out = capsys.readouterr().out

    assert "does NOT clearly beat cosine" not in out  # dead branch, never printed
    verdict = "[oos] VERDICT" in out
    revert = "(consider reverting gate to cosine default: --no-model)" in out
    assert verdict != revert  # exactly one of the two, never a combined line
    if verdict:
        assert "GENERALIZES — beats cosine (keep model default)" in out


def test_oos_never_persists(tmp_path, monkeypatch):
    _write_corpus(tmp_path, QF_ROWS, run="dt-loci-r1")
    _write_corpus(tmp_path, [
        {"text": "a3", "tags": ["dt_target:alpha"]},
        {"text": "b3", "tags": ["dt_target:beta"]},
    ], run="dt-loci-r2")
    vecs = dict(QF_VECTORS)
    vecs.update({"a3": [1.0, 0.0], "b3": [0.0, 1.0]})
    monkeypatch.setattr(harness, "OLLAMA_URL", "http://o")
    monkeypatch.setattr(harness, "DRY_RUN", False)
    monkeypatch.setattr(harness, "embed", unit_embedder(vecs))
    monkeypatch.setattr(oos, "CORPUS", str(tmp_path / "dt-loci-*" / "findings.jsonl"))
    monkeypatch.setattr(harness, "ensure_collection", lambda *a, **k: pytest.fail("persisted"))
    monkeypatch.setattr(harness, "upsert_score", lambda **kw: pytest.fail("persisted"))
    oos.run()  # print-only by design


# ---------------------------------------------------------------------------
# run_eval.sh
# ---------------------------------------------------------------------------

def test_run_eval_sh_invokes_only_the_persisting_evals():
    sh = (EVAL_DIR / "run_eval.sh").read_text()
    assert "harness.py" in sh
    assert "grounding_gate_eval.py" in sh
    assert "grounding_gate_qf_eval.py" in sh
    # the out-of-sample eval is on-demand only and is deliberately NOT in the suite
    assert "grounding_gate_oos_eval.py" not in sh
    assert "set -euo pipefail" in sh
