"""Live grading of a sample of classify/compress answers (pool_grader), and the task-scope fix it depends on."""
import json
import queue
import sys
import time
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import llm_local  # noqa: E402
import model_pool as M  # noqa: E402
import pool_grader as G  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    M.clear_cache()
    monkeypatch.setenv(M.SHADOW_ENV, "1")
    monkeypatch.setenv(G.GRADE_ENV, "1")
    monkeypatch.delenv(G.ROLE_ENV, raising=False)
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path / "memory-sessions"))
    M._LAST_DECISION.set(None)
    monkeypatch.setattr(G, "_queue", queue.Queue(maxsize=2))
    monkeypatch.setattr(G, "_ensure_worker", lambda: None)
    yield
    M.clear_cache()


def _grades(tmp_path):
    path = tmp_path / "instrumentation" / M.GRADES_LOG_NAME
    return [json.loads(line) for line in path.read_text().splitlines()]


def _item(**kw):
    base = {"task": "classify", "prompt": "Labels: a, b\nText: t", "answer": "a", "model": "m1", "decision_id": "d1"}
    return {**base, **kw}


def _judge(text, model="judge-model", ok=True):
    return lambda task, prompt, answer, answerer: {"text": text, "ok": ok, "model": model}


class TestRate:
    @pytest.mark.parametrize("raw, want", [("0.25", 0.25), ("1", 1.0), ("7", 1.0), ("-1", 0.0), ("junk", 0.0), ("", 0.0)])
    def test_rate_is_a_clamped_probability(self, monkeypatch, raw, want):
        monkeypatch.setenv(G.GRADE_ENV, raw)
        assert G.rate() == want

    def test_unset_means_off(self, monkeypatch):
        monkeypatch.delenv(G.GRADE_ENV)
        assert G.rate() == 0.0


class TestSubmit:
    def _submit(self, **kw):
        args = {"task": "classify", "prompt": "p", "answer": "a", "model": "m1", "decision_id": "d1"}
        return G.maybe_submit(**{**args, **kw})

    def test_a_sampled_call_is_queued_with_its_text(self):
        assert self._submit() is True
        assert G._queue.get_nowait() == {"task": "classify", "prompt": "p", "answer": "a", "model": "m1", "decision_id": "d1"}

    @pytest.mark.parametrize("kw", [{"task": "judge"}, {"task": ""}, {"task": "verify"}, {"ok": False}, {"decision_id": ""},
                                    {"model": ""}])
    def test_calls_that_are_not_gradable_are_not_queued(self, kw):
        assert self._submit(**kw) is False
        assert G._queue.empty()
        assert self._submit() is True                                            # positive twin

    def test_off_without_the_shadow_or_a_rate(self, monkeypatch):
        monkeypatch.delenv(M.SHADOW_ENV)
        assert self._submit() is False
        monkeypatch.setenv(M.SHADOW_ENV, "1")
        monkeypatch.setenv(G.GRADE_ENV, "0")
        assert self._submit() is False
        assert G._queue.empty()

    def test_the_rate_is_the_sampling_probability(self, monkeypatch):
        monkeypatch.setenv(G.GRADE_ENV, "0.3")
        monkeypatch.setattr(G.random, "random", lambda: 0.29)
        assert self._submit() is True
        monkeypatch.setattr(G.random, "random", lambda: 0.30)
        assert self._submit() is False

    def test_a_full_queue_skips_loudly(self, tmp_path):
        assert [self._submit(decision_id=f"d{i}") for i in range(3)] == [True, True, False]
        row = _grades(tmp_path)[0]
        assert (row["decision_id"], row["skipped"], row["model"], row["task"]) == ("d2", "queue_full", "m1", "classify")


class TestGradeOne:
    def test_a_verdict_is_recorded_with_the_judge_as_grader(self, tmp_path):
        assert G.grade_one(_item(), _judge('{"correct": true}')) is True
        assert G.grade_one(_item(decision_id="d2"), _judge('{"correct": false}')) is False
        rows = _grades(tmp_path)
        assert [(r["decision_id"], r["correct"], r["grader"], r["model"], r["task"]) for r in rows] == [
            ("d1", True, "judge", "m1", "classify"), ("d2", False, "judge", "m1", "classify")]

    def test_the_judge_sees_the_task_and_the_answer(self):
        seen = []

        def judge(task, prompt, answer, answerer):
            seen.append((task, prompt, answer, answerer))
            return {"text": '{"correct": true}', "ok": True, "model": "j"}
        G.grade_one(_item(task="compress", prompt="P", answer="A"), judge)
        assert seen == [("compress", "P", "A", "m1")]

    def test_the_prompt_names_the_rule_and_carries_both_texts(self):
        text = G.judge_prompt("classify", "PROMPT-TEXT", "ANSWER-TEXT")
        assert "best fit" in text and "PROMPT-TEXT" in text and "ANSWER-TEXT" in text and '{"correct": true}' in text
        assert "key facts" in G.judge_prompt("compress", "p", "a")

    def test_long_text_is_cut_before_it_reaches_the_judge(self):
        text = G.judge_prompt("classify", "p" * 5000, "a" * 3000)
        assert text.count("p") < 4100 and "a" * 2001 not in text and "a" * 2000 in text

    def test_a_judge_that_is_the_same_model_is_skipped_not_trusted(self, tmp_path):
        assert G.grade_one(_item(), _judge('{"correct": true}', model="m1")) is None
        row = _grades(tmp_path)[0]
        assert "correct" not in row and row["skipped"] == "same_model"

    @pytest.mark.parametrize("res, reason", [
        (_judge("", ok=False), "judge_failed"),
        (lambda *a: "not a dict", "judge_failed"),
        (_judge("not json"), "judge_unparseable"),
        (_judge('{"correct": "yes"}'), "judge_unparseable"),
        (_judge('{"correct": 1}'), "judge_unparseable"),
        (_judge('{"other": true}'), "judge_unparseable"),
        (_judge("[true]"), "judge_unparseable"),
    ])
    def test_a_bad_judge_answer_is_a_loud_skip_never_a_grade(self, tmp_path, res, reason):
        assert G.grade_one(_item(), res) is None
        row = _grades(tmp_path)[0]
        assert row["skipped"] == reason and "correct" not in row

    def test_a_judge_that_raises_is_a_skip(self, tmp_path):
        def boom(*a):
            raise RuntimeError("down")
        assert G.grade_one(_item(), boom) is None
        assert _grades(tmp_path)[0]["skipped"] == "judge_failed"

    def test_the_default_judge_is_a_pool_pick_other_than_the_answerer(self, monkeypatch):
        calls, asked = [], []
        monkeypatch.setattr(M, "pick_other", lambda role, exclude: asked.append((role, exclude)) or "pool-judge:4b")
        monkeypatch.setattr(llm_local, "generate", lambda prompt, **k: calls.append(k) or {"text": '{"correct": true}', "ok": True, "model": "pool-judge:4b"})
        assert G._judge("classify", "p", "a", "answerer:4b")["ok"] is True
        assert asked == [("verify", "answerer:4b")]
        assert (calls[0]["model"], calls[0]["fmt"], calls[0]["task"], "role" in calls[0]) == ("pool-judge:4b", "json", "judge", False)
        monkeypatch.setenv(G.ROLE_ENV, "gen_large")
        G._judge("classify", "p", "a", "answerer:4b")
        assert asked[1] == ("gen_large", "answerer:4b")

    def test_no_other_model_is_a_loud_skip_and_no_call(self, monkeypatch, tmp_path):
        monkeypatch.setattr(M, "pick_other", lambda role, exclude: "")
        monkeypatch.setattr(llm_local, "generate", lambda *a, **k: pytest.fail("the judge must not be called"))
        assert G.grade_one(_item()) is None
        row = _grades(tmp_path)[0]
        assert row["skipped"] == "no_other_model" and "correct" not in row


class TestReportSkips:
    def test_the_report_counts_skips_by_reason(self, tmp_path):
        G.grade_one(_item(decision_id="a"), _judge("", ok=False))
        G.grade_one(_item(decision_id="b"), _judge("", ok=False))
        G.grade_one(_item(decision_id="c"), _judge('{"correct": true}'))
        report = M.pool_report()
        assert report["grade_skips"] == {"judge_failed": 2} and report["grades"] == 1

    def test_a_skip_needs_a_reason_and_an_id(self, tmp_path):
        assert [M.record_grade_skip("", "x"), M.record_grade_skip("d", ""), M.record_grade_skip("d", "has space")] == [False] * 3
        assert M.record_grade_skip("d", "queue_full") is True


class TestGenerateWiring:
    def _fake(self, monkeypatch, model="m"):
        def fake(*a, **k):
            M._LAST_DECISION.set({"id": "dX", "model": model, "ts": time.time(), "role": "gen", "explored": False,
                                  "task": M.current_task()})
            return {"text": "ans", "ok": True, "model": model}
        monkeypatch.setattr(llm_local, "_generate", fake)

    def test_a_scope_set_by_the_caller_survives_a_generate_with_no_task(self, monkeypatch, tmp_path):
        """text_ops tags its calls with task_scope and never passes task=; generate must not reset the scope."""
        self._fake(monkeypatch)
        with M.task_scope("classify"):
            llm_local.generate("p")
        row = json.loads((tmp_path / "instrumentation" / M.OUTCOMES_LOG_NAME).read_text().splitlines()[0])
        assert row["task"] == "classify"

    def test_an_explicit_task_beats_the_scope(self, monkeypatch, tmp_path):
        self._fake(monkeypatch)
        with M.task_scope("classify"):
            llm_local.generate("p", task="compress")
        row = json.loads((tmp_path / "instrumentation" / M.OUTCOMES_LOG_NAME).read_text().splitlines()[0])
        assert row["task"] == "compress"

    def test_a_logged_call_is_offered_with_its_prompt_answer_and_id(self, monkeypatch):
        self._fake(monkeypatch)
        offered = []
        monkeypatch.setattr(G, "maybe_submit", lambda *a, **k: offered.append((a, k)))
        with M.task_scope("compress"):
            llm_local.generate("the prompt")
        assert offered == [(("compress", "the prompt", "ans", "m", "dX"), {"ok": True})]

    def test_a_call_with_no_decision_is_not_offered(self, monkeypatch):
        monkeypatch.setattr(llm_local, "_generate", lambda *a, **k: {"text": "ans", "ok": True, "model": "m"})
        offered = []
        monkeypatch.setattr(G, "maybe_submit", lambda *a, **k: offered.append(a))
        llm_local.generate("p", task="classify")
        assert offered == []

    def test_judge_calls_are_never_graded(self, monkeypatch):
        self._fake(monkeypatch)
        assert G.maybe_submit("judge", "p", "a", "m", "dX") is False
