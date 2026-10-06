"""Graded outcomes and the task tag: the pool's log said a call returned, never that the answer was right.

2026-10-06: a forced run over the exploration design had to grade answers outside the pool because the outcome log
holds only ``ok``. ``record_grade`` puts correctness beside the decision, ``task`` says what the call was for, and
the report and ``graded_rate`` selector use both.
"""
import json
import sys
import time
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import llm_local  # noqa: E402
import model_pool as M  # noqa: E402
import pool_selectors  # noqa: E402
import text_ops  # noqa: E402

GB = 1024 ** 3


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    M.clear_cache()
    for var in (M.SELECTOR_ENV, M.EXPLORE_ENV, M.EXPLORE_MODELS_ENV, M.EXPLORE_ROLES_ENV, M.EXPLORE_COLD_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(M.SHADOW_ENV, "1")
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path / "memory-sessions"))
    M._LAST_DECISION.set(None)
    pool_selectors._gcache.update(at=0.0, stats={})
    yield
    M.clear_cache()


def _rows(tmp_path, name):
    path = tmp_path / "instrumentation" / name
    return [json.loads(line) for line in path.read_text().splitlines()]


class TestTaskTag:
    @pytest.mark.parametrize("raw, clean", [("classify", "classify"), ("  Compress ", "compress"), ("a.b-c_1", "a.b-c_1"),
                                            ("has space", ""), ("", ""), (None, ""), ("x" * 40, ""), ("-lead", "")])
    def test_only_a_short_identifier_survives(self, raw, clean):
        assert M.clean_task(raw) == clean

    def test_scope_sets_and_restores_the_task(self):
        assert M.current_task() == ""
        with M.task_scope("classify"):
            assert M.current_task() == "classify"
            with M.task_scope("compress"):
                assert M.current_task() == "compress"
            assert M.current_task() == "classify"
        assert M.current_task() == ""

    def test_outcome_takes_the_explicit_task_over_the_scope(self, tmp_path):
        with M.task_scope("classify"):
            M.record_outcome("m", True, 5, task="compress")
            M.record_outcome("m", True, 5)
        assert [r.get("task") for r in _rows(tmp_path, M.OUTCOMES_LOG_NAME)] == ["compress", "classify"]

    def test_outcome_inherits_the_task_of_the_decision_it_links_to(self, monkeypatch, tmp_path):
        monkeypatch.setattr(M, "entries", lambda: [M.PoolEntry("a", ("gen",), rank=1.0)])
        monkeypatch.setattr(M, "inventory", lambda base_url=None: {"a": GB})
        monkeypatch.setattr(M, "resident_models", lambda base_url=None: set())
        with M.task_scope("classify"):
            M.pick("gen")
        assert M.current_task() == ""                                        # the outcome is logged outside the scope
        M.record_outcome("a", True, 5)
        assert _rows(tmp_path, M.OUTCOMES_LOG_NAME)[0]["task"] == "classify"

    def test_outcome_without_any_task_has_no_task_key(self, tmp_path):
        M.record_outcome("m", True, 5)
        assert "task" not in _rows(tmp_path, M.OUTCOMES_LOG_NAME)[0]

    def test_outcome_returns_the_decision_id_it_carries(self, tmp_path):
        assert M.record_outcome("m", True, 5, decision_id="d1") == "d1"
        assert M.record_outcome("m", True, 5) == ""

    def test_outcome_returns_empty_when_the_shadow_is_off(self, monkeypatch, tmp_path):
        monkeypatch.delenv(M.SHADOW_ENV)
        assert M.record_outcome("m", True, 5, decision_id="d1") == ""


class TestRecordGrade:
    def test_a_grade_row_holds_the_identifiers_only(self, tmp_path):
        assert M.record_grade("d1", True, model="m", task="classify", grader="gold") is True
        row = _rows(tmp_path, M.GRADES_LOG_NAME)[0]
        assert {k: row[k] for k in ("schema", "decision_id", "correct", "model", "task", "grader")} == {
            "schema": M.GRADE_SCHEMA, "decision_id": "d1", "correct": True, "model": "m", "task": "classify",
            "grader": "gold"}
        assert set(row) == {"schema", "ts", "decision_id", "correct", "model", "task", "grader"}

    def test_correct_must_be_a_real_bool_and_the_id_present(self, tmp_path):
        results = [M.record_grade("d1", 1), M.record_grade("d1", "yes"), M.record_grade("", True),
                   M.record_grade("d1", None)]
        assert results == [False] * 4
        assert not (tmp_path / "instrumentation" / M.GRADES_LOG_NAME).exists()
        assert M.record_grade("d1", False) is True                                    # positive twin

    def test_off_without_the_shadow(self, monkeypatch, tmp_path):
        monkeypatch.delenv(M.SHADOW_ENV)
        assert M.record_grade("d1", True) is False
        assert not (tmp_path / "instrumentation" / M.GRADES_LOG_NAME).exists()

    def test_summary_counts_the_latest_grade_per_decision(self, tmp_path):
        M.record_grade("d1", False, model="a")
        M.record_grade("d1", True, model="a")                                         # re-graded: counts once
        M.record_grade("d2", False, model="a")
        M.record_grade("d3", True, model="b")
        assert M.grades_summary() == {"a": {"graded": 2, "correct": 1, "correct_rate": 0.5},
                                      "b": {"graded": 1, "correct": 1, "correct_rate": 1.0}}


class TestReportJoin:
    def _run(self, monkeypatch, plan):
        """plan: [(model, task, correct or None)]; each is one decision + linked outcome (+ grade)."""
        names = sorted({m for m, _, _ in plan})
        monkeypatch.setattr(M, "entries", lambda: [M.PoolEntry(n, ("gen",), rank=float(i + 1)) for i, n in enumerate(names)])
        monkeypatch.setattr(M, "inventory", lambda base_url=None: {n: GB for n in names})
        monkeypatch.setattr(M, "resident_models", lambda base_url=None: set())
        for model, task, correct in plan:
            with M.task_scope(task):
                M.pick("gen")
                M._LAST_DECISION.get()["model"] = model
                did = M.record_outcome(model, True, 10)
            if correct is not None:
                assert M.record_grade(did, correct, model=model)

    def test_arms_report_graded_and_correct_rate(self, monkeypatch):
        self._run(monkeypatch, [("a", "classify", True), ("a", "classify", False), ("a", "classify", True),
                                ("b", "classify", None)])
        arms = M.pool_report()["roles"]["gen"]["arms"]
        assert (arms["a"]["graded"], arms["a"]["correct_rate"]) == (3, 0.667)
        assert (arms["b"]["graded"], arms["b"]["correct_rate"], arms["b"]["outcomes"]) == (0, None, 1)

    def test_by_task_splits_the_same_arm(self, monkeypatch):
        self._run(monkeypatch, [("a", "classify", True), ("a", "compress", False), ("a", "compress", False)])
        by_task = M.pool_report()["roles"]["gen"]["by_task"]
        assert by_task["classify"]["a"]["correct_rate"] == 1.0
        assert by_task["compress"]["a"]["correct_rate"] == 0.0 and by_task["compress"]["a"]["outcomes"] == 2

    def test_untagged_calls_are_grouped_not_dropped(self, monkeypatch):
        self._run(monkeypatch, [("a", "", None)])
        assert list(M.pool_report()["roles"]["gen"]["by_task"]) == ["(untagged)"]

    def test_quality_is_learnable_only_with_two_arms_graded_enough(self, monkeypatch):
        n = M.MIN_ARM_N
        self._run(monkeypatch, [("a", "classify", True)] * n + [("b", "classify", False)] * (n - 1))
        role = M.pool_report()["roles"]["gen"]
        assert role["quality_learnable"] is False
        assert role["quality_why"] == f"fewer than two arms have {n} graded answers yet ({2 * n - 1} graded in all)"

    def test_quality_learnable_names_the_arms(self, monkeypatch):
        n = M.MIN_ARM_N
        self._run(monkeypatch, [("a", "classify", True)] * n + [("b", "classify", False)] * n)
        role = M.pool_report()["roles"]["gen"]
        assert role["quality_learnable"] is True
        assert role["quality_why"] == f"2 arms have at least {n} graded answers: a, b"

    def test_ungraded_data_says_so(self, monkeypatch):
        self._run(monkeypatch, [("a", "classify", None)])
        report = M.pool_report()
        assert report["roles"]["gen"]["quality_why"] == "no answer has been graded, so correctness is unknown; see record_grade"
        assert report["grades"] == 0

    def test_cli_shows_correctness(self, monkeypatch, capsys):
        self._run(monkeypatch, [("a", "classify", True), ("a", "classify", False)])
        assert M._main(["report"]) == 0
        assert "arm a: n=2 ok=100% p50=10ms correct=50% (n=2 graded)" in capsys.readouterr().out


class TestGradedSelector:
    def _features(self, *names, eligible=True):
        return [{"name": n, "rank": 1, "effective_rank": 1, "resident": False, "eligible": eligible} for n in names]

    def _stats(self, monkeypatch, stats):
        monkeypatch.setattr(M, "grades_summary", lambda path=None: stats)

    def test_picks_the_most_correct_arm_with_enough_graded_answers(self, monkeypatch):
        self._stats(monkeypatch, {"a": {"graded": 40, "correct_rate": 0.7}, "b": {"graded": 30, "correct_rate": 0.9},
                                  "c": {"graded": 29, "correct_rate": 1.0}})
        assert pool_selectors.graded_rate("gen", self._features("a", "b", "c")) == "b"

    def test_ties_go_to_the_name_and_ineligible_arms_are_skipped(self, monkeypatch):
        self._stats(monkeypatch, {"a": {"graded": 40, "correct_rate": 0.9}, "b": {"graded": 40, "correct_rate": 0.9}})
        assert pool_selectors.graded_rate("gen", self._features("b", "a")) == "a"
        pool_selectors._gcache.update(at=0.0)
        assert pool_selectors.graded_rate("gen", self._features("a", eligible=False)) is None

    def test_no_graded_evidence_means_no_opinion(self, monkeypatch):
        self._stats(monkeypatch, {})
        assert pool_selectors.graded_rate("gen", self._features("a")) is None


class TestCallSites:
    def test_generate_returns_the_decision_id_and_tags_the_task(self, monkeypatch, tmp_path):
        def fake(*a, **k):
            M._LAST_DECISION.set({"id": "dX", "model": "m", "ts": time.time(), "role": "gen",
                                  "explored": False, "task": M.current_task()})
            return {"text": "hi", "ok": True, "model": "m"}
        monkeypatch.setattr(llm_local, "_generate", fake)
        out = llm_local.generate("p", task="classify")
        assert out == {"text": "hi", "ok": True, "model": "m", "decision_id": "dX"}
        row = _rows(tmp_path, M.OUTCOMES_LOG_NAME)[0]
        assert (row["decision_id"], row["task"]) == ("dX", "classify")

    def test_generate_without_a_decision_keeps_the_result_unchanged(self, monkeypatch):
        monkeypatch.setattr(llm_local, "_generate", lambda *a, **k: {"text": "hi", "ok": True, "model": "m"})
        assert llm_local.generate("p") == {"text": "hi", "ok": True, "model": "m"}

    def test_the_task_does_not_leak_out_of_generate(self, monkeypatch):
        monkeypatch.setattr(llm_local, "_generate", lambda *a, **k: {"text": "", "ok": False, "model": "m"})
        llm_local.generate("p", task="classify")
        assert M.current_task() == ""

    def test_classify_and_compress_tag_their_calls(self):
        seen = []

        def gen(prompt, *, fmt=None, max_tokens=256):
            seen.append(M.current_task())
            return {"text": "a", "ok": True}
        text_ops.classify("some text with no label in it", ["a", "b"], gen_fn=gen)
        text_ops.compress("x" * 700, max_chars=100, gen_fn=gen)
        assert seen == ["classify", "compress"]
        assert M.current_task() == ""
