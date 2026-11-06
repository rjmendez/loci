"""pool_grader oracle judging through the teacher core (role judge), and the unchanged path with teacher off.

The teacher core and the guard are real; only its transport (the LiteLLM call) and ``llm_local.generate`` are faked.
"""
import json
import os
import queue
import sys
from decimal import Decimal
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import backends  # noqa: E402
import llm_local  # noqa: E402
import model_pool as M  # noqa: E402
import pool_grader as G  # noqa: E402
import teacher as T  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    for k in list(os.environ):
        if k.startswith("LOCI_TEACHER_"):
            monkeypatch.delenv(k)
    monkeypatch.setattr(backends, "_CONFIG_PATH", str(tmp_path / "backends.toml"))
    backends._config_cache_clear()
    T.cache_clear()
    T._reservations.clear()
    T._settling.clear()
    T.overruns_reset()
    M.clear_cache()
    monkeypatch.setenv(M.SHADOW_ENV, "1")
    monkeypatch.setenv(G.GRADE_ENV, "1")
    monkeypatch.setenv(G.ORACLE_ENV, "1")
    monkeypatch.delenv(G.ROLE_ENV, raising=False)
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path / "memory-sessions"))
    M._LAST_DECISION.set(None)
    monkeypatch.setattr(G, "_queue", queue.Queue(maxsize=2))
    monkeypatch.setattr(G, "_ensure_worker", lambda: None)
    monkeypatch.setattr(G, "_oracle_bucket", None)
    monkeypatch.setattr(G, "_oracle_queue", None)
    yield
    T._reservations.clear()
    T._settling.clear()
    T.cache_clear()
    backends._config_cache_clear()
    M.clear_cache()


def _grades(tmp_path):
    path = tmp_path / "instrumentation" / M.GRADES_LOG_NAME
    return [json.loads(line) for line in path.read_text().splitlines()]


def _item(**kw):
    base = {"task": "classify", "prompt": "Labels: a, b\nText: t", "answer": "a", "model": "m1", "decision_id": "d1"}
    return {**base, **kw}


def _two_model_pool(monkeypatch):
    monkeypatch.setattr(M, "entries", lambda: [M.PoolEntry("m1", ("verify",), rank=1.0),
                                               M.PoolEntry("m2", ("verify",), rank=0.5)])
    monkeypatch.setattr(M, "inventory", lambda base_url=None: {"m1": 10 ** 9, "m2": 10 ** 9})
    monkeypatch.setattr(M, "resident_models", lambda base_url=None: {"m1", "m2"})


def _one_model_pool(monkeypatch):
    monkeypatch.setattr(M, "entries", lambda: [M.PoolEntry("m1", ("verify",), rank=1.0)])
    monkeypatch.setattr(M, "inventory", lambda base_url=None: {"m1": 10 ** 9})
    monkeypatch.setattr(M, "resident_models", lambda base_url=None: {"m1"})


class Local:
    """Recording stand-in for llm_local.generate; every judgment costs ``cost``."""

    def __init__(self, cost="0.3"):
        self.cost, self.calls = cost, []

    def __call__(self, prompt, **kw):
        self.calls.append((prompt, kw))
        return {"ok": True, "text": '{"correct": true}', "model": kw["model"], "cost_usd": self.cost}


class Cloud:
    """Recording stand-in for the teacher transport (the LiteLLM call)."""

    def __init__(self, text='{"correct": true}', cost=0.6, bad=False):
        self.text, self.cost, self.bad, self.calls = text, cost, bad, []

    def __call__(self, group, messages, max_tokens, metadata):
        self.calls.append({"group": group, "messages": messages, "max_tokens": max_tokens, "metadata": metadata})
        if self.bad:
            return None
        return {"choices": [{"message": {"content": self.text}}], "usage": {"cost": self.cost}}


def _enable_teacher(monkeypatch, tmp_path, daily="100", per_call="1"):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    for k, v in {"ENABLED": "1", "ROLE_JUDGE": "grp-judge", "MONTHLY_USD_CAP": "1000", "DAILY_USD_CAP": daily,
                 "MAX_USD_PER_CALL": per_call, "LEDGER_FILE": str(ledger), "SAFETY_MARGIN_PCT": "0",
                 "SANITIZE": "0", "PRICES": "grp-judge=1:1"}.items():
        monkeypatch.setenv("LOCI_TEACHER_" + k, v)
    T.cache_clear()


@pytest.fixture
def cloud(monkeypatch):
    c = Cloud()
    monkeypatch.setattr(T, "_default_transport", c)
    return c


# Outputs of the pre-teacher code (pool_grader at c1b9c440) on this exact scenario, with teacher off.
OLD_PATH_GOLDEN = {
    "verdicts": [True, True, None],
    "generate_models": ["m2", "m2"],
    "generate_kwargs": {"fmt": "json", "max_tokens": 24, "task": "judge", "timeout": 60},
    "rows": [("d0", True, None), ("d1", True, None), ("d2", None, "oracle_starved")],
    "bucket": {"reserved": 2, "starved": 1, "settled": 2, "spent": Decimal("0.6")},
}


class TestTeacherOffIsUnchanged:
    def _run(self, monkeypatch, tmp_path):
        _two_model_pool(monkeypatch)
        local, teacher_calls = Local("0.3"), []
        monkeypatch.setattr(llm_local, "generate", local)
        monkeypatch.setattr(T, "teacher_call", lambda *a, **k: teacher_calls.append((a, k)))
        verdicts = [G.grade_one(_item(decision_id=f"d{i}")) for i in range(3)]
        return verdicts, local, teacher_calls

    def test_the_old_path_runs_exactly_as_before(self, monkeypatch, tmp_path):
        verdicts, local, teacher_calls = self._run(monkeypatch, tmp_path)
        assert verdicts == OLD_PATH_GOLDEN["verdicts"]
        assert [kw["model"] for _, kw in local.calls] == OLD_PATH_GOLDEN["generate_models"]
        assert {k: v for k, v in local.calls[0][1].items() if k != "model"} == OLD_PATH_GOLDEN["generate_kwargs"]
        assert local.calls[0][0] == G.judge_prompt("classify", "Labels: a, b\nText: t", "a")
        assert [(r["decision_id"], r.get("correct"), r.get("skipped")) for r in _grades(tmp_path)] == \
            OLD_PATH_GOLDEN["rows"]
        s = G.oracle_stats()["bucket"]
        assert {"reserved": s["reserved"], "starved": s["starved"], "settled": s["settled"],
                "spent": G.oracle_bucket().spent()} == OLD_PATH_GOLDEN["bucket"]
        assert teacher_calls == []

    def test_a_configured_but_disabled_teacher_changes_nothing(self, monkeypatch, tmp_path, cloud):
        _enable_teacher(monkeypatch, tmp_path)
        monkeypatch.setenv("LOCI_TEACHER_ENABLED", "0")
        verdicts, local, _ = self._run(monkeypatch, tmp_path)
        assert verdicts == OLD_PATH_GOLDEN["verdicts"]
        assert len(local.calls) == 2
        assert cloud.calls == []

    def test_teacher_enabled_but_oracle_off_keeps_the_local_judge(self, monkeypatch, tmp_path, cloud):
        _enable_teacher(monkeypatch, tmp_path)
        monkeypatch.delenv(G.ORACLE_ENV)
        _two_model_pool(monkeypatch)
        local = Local()
        monkeypatch.setattr(llm_local, "generate", local)
        assert G.grade_one(_item()) is True
        assert [kw["model"] for _, kw in local.calls] == ["m2"]
        assert cloud.calls == []

    def test_unreadable_teacher_config_falls_back_to_the_old_path(self, monkeypatch, tmp_path, cloud):
        _two_model_pool(monkeypatch)
        local = Local()
        monkeypatch.setattr(llm_local, "generate", local)

        def broken():
            raise T.LedgerUnreadable("config boom")
        monkeypatch.setattr(T, "teacher_config", broken)
        assert G.grade_one(_item()) is True
        assert len(local.calls) == 1
        assert cloud.calls == []


class TestTeacherOn:
    def test_the_oracle_judgment_goes_to_the_judge_role_and_not_to_the_local_model(self, monkeypatch, tmp_path, cloud):
        _enable_teacher(monkeypatch, tmp_path)
        _two_model_pool(monkeypatch)
        local = Local()
        monkeypatch.setattr(llm_local, "generate", local)
        assert G.grade_one(_item()) is True
        assert local.calls == []
        assert len(cloud.calls) == 1
        call = cloud.calls[0]
        assert call["group"] == "grp-judge"
        assert call["max_tokens"] == 24
        assert call["messages"] == [{"role": "user", "content": G.judge_prompt("classify", "Labels: a, b\nText: t", "a")}]
        assert call["metadata"] == {"loci_internal_call": True, "loci_teacher_purpose": "pool_grader_oracle"}
        row = _grades(tmp_path)[0]
        assert (row["decision_id"], row["correct"], row["grader"], row["model"]) == ("d1", True, "judge", "m1")

    def test_a_cloud_verdict_of_false_is_recorded_false(self, monkeypatch, tmp_path):
        _enable_teacher(monkeypatch, tmp_path)
        monkeypatch.setattr(T, "_default_transport", Cloud(text='{"correct": false}'))
        assert G.grade_one(_item()) is False
        assert _grades(tmp_path)[0]["correct"] is False

    def test_the_half_dollar_bucket_is_replaced_by_the_teacher_caps(self, monkeypatch, tmp_path, cloud):
        _enable_teacher(monkeypatch, tmp_path, daily="100")
        verdicts = [G.grade_one(_item(decision_id=f"d{i}")) for i in range(5)]
        assert verdicts == [True] * 5                      # 5 x $0.60 reported: past the old $0.50 cap
        assert len(cloud.calls) == 5
        s = G.oracle_stats()["bucket"]
        assert (s["reserved"], s["settled"], s["starved"], G.oracle_bucket().spent()) == (0, 0, 0, Decimal(0))

    def test_a_teacher_cap_refusal_skips_the_item_and_sends_nothing(self, monkeypatch, tmp_path, cloud):
        _enable_teacher(monkeypatch, tmp_path, daily="0.0000001")
        assert G.grade_one(_item()) is None
        assert cloud.calls == []
        row = _grades(tmp_path)[0]
        assert (row["decision_id"], row["skipped"], row["model"], row["task"], "correct" in row) == (
            "d1", "teacher_refused", "m1", "classify", False)

    def test_a_bad_cloud_response_is_skipped_not_raised(self, monkeypatch, tmp_path):
        _enable_teacher(monkeypatch, tmp_path)
        bad = Cloud(bad=True)
        monkeypatch.setattr(T, "_default_transport", bad)
        assert G.grade_one(_item()) is None
        assert len(bad.calls) == 1
        assert _grades(tmp_path)[0]["skipped"] == "teacher_refused"

    def test_unparseable_cloud_text_is_skipped_as_unparseable(self, monkeypatch, tmp_path):
        _enable_teacher(monkeypatch, tmp_path)
        monkeypatch.setattr(T, "_default_transport", Cloud(text="yes"))
        assert G.grade_one(_item()) is None
        assert _grades(tmp_path)[0]["skipped"] == "judge_unparseable"

    def test_a_single_model_pool_is_not_same_model_when_the_cloud_judges(self, monkeypatch, tmp_path, cloud):
        _enable_teacher(monkeypatch, tmp_path)
        _one_model_pool(monkeypatch)
        monkeypatch.setattr(llm_local, "generate", Local())
        assert G.grade_one(_item(model="m1")) is True
        assert len(cloud.calls) == 1

    def test_the_same_single_model_pool_is_same_model_when_teacher_is_off(self, monkeypatch, tmp_path):
        _one_model_pool(monkeypatch)
        local = Local()
        monkeypatch.setattr(llm_local, "generate", local)
        assert G.grade_one(_item(model="m1")) is None
        assert _grades(tmp_path)[0]["skipped"] == "same_model"
        assert local.calls == []

    def test_the_spend_is_visible_to_the_teacher_guard(self, monkeypatch, tmp_path, cloud):
        _enable_teacher(monkeypatch, tmp_path)
        assert G.grade_one(_item()) is True
        assert T.status()["overruns"]["count"] == 1         # reported $0.60 vs the small estimate: counted by the core
