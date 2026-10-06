"""The pool shadow as training and testing data for a selector.

2026-10-05: 2,686 decision rows existed with chosen_shadow empty in every one, and nothing tied a decision to the call
it led to (decisions are made when a model name is resolved, outcomes when generate() returns, under different role
names). Rows now carry a decision_id shared with their outcome, say why the shadow is empty, and an opt-in exploration
gives the arms the rule never picks real outcomes. The report says whether the data can support learning at all.
"""
import json
import sys
import threading
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import model_pool as M  # noqa: E402
import pool_selectors  # noqa: E402

GB = 1024 ** 3


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    M.clear_cache()
    for var in (M.SHADOW_ENV, M.SELECTOR_ENV, M.EXPLORE_ENV, M.EXPLORE_MODELS_ENV, M.EXPLORE_ROLES_ENV,
                M.EXPLORE_COLD_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path / "memory-sessions"))
    M._LAST_DECISION.set(None)
    pool_selectors._cache.update(at=0.0, stats={})
    yield
    M.clear_cache()


def _pool(monkeypatch, names=("a", "b", "c")):
    entries = [M.PoolEntry(n, ("gen",), rank=float(i + 1)) for i, n in enumerate(names)]
    monkeypatch.setattr(M, "entries", lambda: entries)
    monkeypatch.setattr(M, "inventory", lambda base_url=None: {n: GB for n in names})
    monkeypatch.setattr(M, "resident_models", lambda base_url=None: set())


def _decisions(tmp_path):
    path = tmp_path / "instrumentation" / M.SHADOW_LOG_NAME
    return [json.loads(line) for line in path.read_text().splitlines()]


def _outcomes(tmp_path):
    path = tmp_path / "instrumentation" / M.OUTCOMES_LOG_NAME
    return [json.loads(line) for line in path.read_text().splitlines()]


def _selector(monkeypatch, fn):
    monkeypatch.setattr(M, "_load_selector", lambda: fn)


class TestShadowStatus:
    def test_each_reason_the_shadow_is_empty_has_its_own_status(self, monkeypatch, tmp_path):
        _pool(monkeypatch)
        monkeypatch.setenv(M.SHADOW_ENV, "1")
        cases = {
            "no_selector": None,
            "chose": lambda role, f: "b",
            "abstained": lambda role, f: None,
            "invalid": lambda role, f: "not-a-candidate",
            "error": lambda role, f: (_ for _ in ()).throw(RuntimeError("selector bug")),
        }
        for expected, fn in cases.items():
            _selector(monkeypatch, fn)
            assert M.pick("gen") == "a"                     # whatever the selector says, the rule decides
            row = _decisions(tmp_path)[-1]
            assert row["shadow_status"] == expected, expected
            assert row["chosen_shadow"] == ("b" if expected == "chose" else None)
            assert row["agree"] == (False if expected == "chose" else None)

    def test_a_list_answer_uses_its_first_eligible_entry(self, monkeypatch, tmp_path):
        _pool(monkeypatch)
        monkeypatch.setenv(M.SHADOW_ENV, "1")
        _selector(monkeypatch, lambda role, f: ["c", "a"])
        M.pick("gen")
        assert _decisions(tmp_path)[-1]["chosen_shadow"] == "c"


class TestDecisionOutcomeJoin:
    def test_every_decision_has_its_own_id_and_the_returned_model(self, monkeypatch, tmp_path):
        _pool(monkeypatch)
        monkeypatch.setenv(M.SHADOW_ENV, "1")
        M.pick("gen")
        M.pick("gen")
        first, second = _decisions(tmp_path)
        assert len(first["decision_id"]) == 12 and first["decision_id"] != second["decision_id"]
        assert first["chosen_returned"] == first["chosen_rule"] == "a"

    def test_the_outcome_of_the_call_carries_the_decisions_id_and_role(self, monkeypatch, tmp_path):
        _pool(monkeypatch)
        monkeypatch.setenv(M.SHADOW_ENV, "1")
        model = M.pick("gen")
        M.record_outcome(model, True, 12.0, route_role="coding")
        decision = _decisions(tmp_path)[-1]
        outcome = _outcomes(tmp_path)[-1]
        assert outcome["decision_id"] == decision["decision_id"]
        assert (outcome["pool_role"], outcome["route_role"]) == ("gen", "coding")

    def test_one_decision_labels_one_call(self, monkeypatch, tmp_path):
        _pool(monkeypatch)
        monkeypatch.setenv(M.SHADOW_ENV, "1")
        model = M.pick("gen")
        M.record_outcome(model, True, 1.0)
        M.record_outcome(model, True, 1.0)
        first, second = _outcomes(tmp_path)
        assert "decision_id" in first and "decision_id" not in second

    def test_an_outcome_for_a_different_model_is_not_linked_but_the_matching_one_is(self, monkeypatch, tmp_path):
        _pool(monkeypatch)
        monkeypatch.setenv(M.SHADOW_ENV, "1")
        M.pick("gen")                                       # returned "a"
        M.record_outcome("b", True, 1.0)
        assert "decision_id" not in _outcomes(tmp_path)[-1]
        M.record_outcome("a", True, 1.0)                    # positive twin: the model the decision chose
        assert "decision_id" in _outcomes(tmp_path)[-1]

    def test_a_stale_decision_is_not_linked(self, monkeypatch, tmp_path):
        _pool(monkeypatch)
        monkeypatch.setenv(M.SHADOW_ENV, "1")
        M.pick("gen")
        monkeypatch.setattr(M, "_LINK_TTL_S", -1.0)
        M.record_outcome("a", True, 1.0)
        assert "decision_id" not in _outcomes(tmp_path)[-1]

    def test_a_decision_made_in_another_thread_is_never_linked(self, monkeypatch, tmp_path):
        _pool(monkeypatch)
        monkeypatch.setenv(M.SHADOW_ENV, "1")
        worker = threading.Thread(target=lambda: M.pick("gen"))
        worker.start()
        worker.join()
        M.record_outcome("a", True, 1.0)
        assert "decision_id" not in _outcomes(tmp_path)[-1]

    def test_an_explicit_decision_id_wins(self, monkeypatch, tmp_path):
        _pool(monkeypatch)
        monkeypatch.setenv(M.SHADOW_ENV, "1")
        M.record_outcome("a", True, 1.0, decision_id="abc123def456")
        assert _outcomes(tmp_path)[-1]["decision_id"] == "abc123def456"

    def test_generate_links_its_outcome_and_logs_a_size_class_never_the_prompt(self, monkeypatch, tmp_path):
        import llm_local
        _pool(monkeypatch)
        monkeypatch.setenv(M.SHADOW_ENV, "1")

        def fake_generate(prompt, **kw):
            return {"text": "SECRET OUTPUT", "ok": True, "model": M.pick("gen")}
        monkeypatch.setattr(llm_local, "_generate", fake_generate)
        llm_local.generate("SECRET PROMPT TEXT", role="coding")
        decision, outcome = _decisions(tmp_path)[-1], _outcomes(tmp_path)[-1]
        assert outcome["decision_id"] == decision["decision_id"]
        assert outcome["prompt_bucket"] == len("SECRET PROMPT TEXT").bit_length()
        assert "SECRET" not in json.dumps(outcome) + json.dumps(decision)

    def test_nothing_is_logged_or_linked_with_the_shadow_off(self, monkeypatch, tmp_path):
        _pool(monkeypatch)
        assert M.pick("gen") == "a"
        M.record_outcome("a", True, 1.0)
        assert not (tmp_path / "instrumentation").exists()


class TestExploration:
    def _on(self, monkeypatch, p="1", models="b,c"):
        _pool(monkeypatch)
        monkeypatch.setenv(M.SHADOW_ENV, "1")
        monkeypatch.setenv(M.EXPLORE_ENV, p)
        monkeypatch.setenv(M.EXPLORE_MODELS_ENV, models)
        monkeypatch.setenv(M.EXPLORE_COLD_ENV, "1")      # these tests are about which models, not about residency

    def test_it_is_off_by_default_and_the_rule_always_decides(self, monkeypatch, tmp_path):
        _pool(monkeypatch)
        monkeypatch.setenv(M.SHADOW_ENV, "1")
        monkeypatch.setenv(M.EXPLORE_MODELS_ENV, "b,c")      # an allow-list alone does nothing
        assert {M.pick("gen") for _ in range(50)} == {"a"}
        assert all("explored" not in row for row in _decisions(tmp_path))

    def test_it_needs_an_allow_list_a_probability_and_the_shadow(self, monkeypatch, tmp_path):
        self._on(monkeypatch, models="")
        assert {M.pick("gen") for _ in range(20)} == {"a"}
        self._on(monkeypatch, p="0")
        assert {M.pick("gen") for _ in range(20)} == {"a"}
        # Neither of those is exploration, so neither leaves exploration fields in the decision rows.
        assert all("explore_p" not in row and "explored" not in row for row in _decisions(tmp_path))
        self._on(monkeypatch)
        monkeypatch.delenv(M.SHADOW_ENV)
        assert {M.pick("gen") for _ in range(20)} == {"a"}

    def test_with_probability_one_the_call_goes_to_an_allow_listed_alternative_and_says_so(self, monkeypatch, tmp_path):
        self._on(monkeypatch, models="b")
        assert M.pick("gen") == "b"
        row = _decisions(tmp_path)[-1]
        assert (row["chosen_rule"], row["chosen_returned"], row["explored"]) == ("a", "b", True)
        assert (row["explore_p"], row["n_alternatives"], row["propensity"]) == (1.0, 1, 1.0)
        M.record_outcome("b", True, 5.0)
        assert _outcomes(tmp_path)[-1]["explored"] is True

    def _roles(self, monkeypatch, roles, models=""):
        entries = [M.PoolEntry(n, ("gen", "verify") if n != "c" else ("gen",), rank=float(i + 1))
                   for i, n in enumerate("abc")]
        monkeypatch.setattr(M, "entries", lambda: entries)
        monkeypatch.setattr(M, "inventory", lambda base_url=None: {n: GB for n in "abc"})
        monkeypatch.setattr(M, "resident_models", lambda base_url=None: set())
        monkeypatch.setenv(M.SHADOW_ENV, "1")
        monkeypatch.setenv(M.EXPLORE_ENV, "1")
        monkeypatch.setenv(M.EXPLORE_COLD_ENV, "1")
        monkeypatch.setenv(M.EXPLORE_ROLES_ENV, roles)
        if models:
            monkeypatch.setenv(M.EXPLORE_MODELS_ENV, models)

    def test_a_role_scope_needs_no_model_list_and_tries_every_eligible_alternative(self, monkeypatch, tmp_path):
        self._roles(monkeypatch, "gen")
        assert {M.pick("gen") for _ in range(200)} == {"b", "c"}
        row = _decisions(tmp_path)[-1]
        assert (row["n_alternatives"], row["propensity"]) == (2, 0.5)

    def test_a_role_scope_leaves_every_other_role_to_the_rule(self, monkeypatch, tmp_path):
        self._roles(monkeypatch, "gen")
        assert {M.pick("verify") for _ in range(100)} == {"a"}
        assert all("explored" not in row and "explore_p" not in row for row in _decisions(tmp_path))

    def test_the_role_names_ignore_case_and_spaces_and_may_be_several(self, monkeypatch):
        self._roles(monkeypatch, " GEN , verify ")
        assert {M.pick("gen") for _ in range(100)} == {"b", "c"}
        assert {M.pick("verify") for _ in range(100)} == {"b"}
        assert {M.pick(" Gen ") for _ in range(100)} == {"b", "c"}   # the caller's spelling does not matter either

    def test_a_role_scope_with_a_model_list_uses_only_the_models_in_both(self, monkeypatch):
        self._roles(monkeypatch, "gen", models="c,ghost")
        assert {M.pick("gen") for _ in range(100)} == {"c"}
        assert {M.pick("verify") for _ in range(50)} == {"a"}

    def test_a_role_scope_still_needs_a_probability_and_the_shadow(self, monkeypatch):
        self._roles(monkeypatch, "gen")
        monkeypatch.setenv(M.EXPLORE_ENV, "0")
        assert {M.pick("gen") for _ in range(50)} == {"a"}
        monkeypatch.setenv(M.EXPLORE_ENV, "1")
        monkeypatch.delenv(M.SHADOW_ENV)
        assert {M.pick("gen") for _ in range(50)} == {"a"}

    def _resident(self, monkeypatch, resident, **kw):
        self._on(monkeypatch, **kw)
        monkeypatch.delenv(M.EXPLORE_COLD_ENV)
        monkeypatch.setattr(M, "resident_models", lambda base_url=None: set(resident))

    def test_by_default_only_a_model_already_resident_is_tried(self, monkeypatch, tmp_path):
        self._resident(monkeypatch, {"a", "c"}, models="")
        monkeypatch.setenv(M.EXPLORE_ROLES_ENV, "gen")
        assert {M.pick("gen") for _ in range(100)} == {"c"}          # b is installed and eligible but cold
        row = _decisions(tmp_path)[-1]
        assert (row["n_alternatives"], row["propensity"]) == (1, 1.0)

    def test_with_no_resident_alternative_it_keeps_the_rules_choice_and_says_so(self, monkeypatch, tmp_path):
        self._resident(monkeypatch, {"a"}, models="")
        monkeypatch.setenv(M.EXPLORE_ROLES_ENV, "gen")
        assert {M.pick("gen") for _ in range(50)} == {"a"}
        row = _decisions(tmp_path)[-1]
        assert (row["n_alternatives"], row["explored"]) == (0, False)

    def test_cold_loads_are_allowed_only_when_asked_for(self, monkeypatch):
        """Positive twin: same setup, cold models allowed."""
        self._resident(monkeypatch, {"a"}, models="")
        monkeypatch.setenv(M.EXPLORE_ROLES_ENV, "gen")
        monkeypatch.setenv(M.EXPLORE_COLD_ENV, "1")
        assert {M.pick("gen") for _ in range(200)} == {"b", "c"}

    def test_each_truthy_value_allows_cold_loads_and_each_other_value_does_not(self, monkeypatch):
        for value in ("1", "true", "YES", "On"):
            self._resident(monkeypatch, {"a"}, models="")
            monkeypatch.setenv(M.EXPLORE_ROLES_ENV, "gen")
            monkeypatch.setenv(M.EXPLORE_COLD_ENV, value)
            assert {M.pick("gen") for _ in range(100)} == {"b", "c"}, value
        for value in ("", "0", "no", "off"):
            self._resident(monkeypatch, {"a"}, models="")
            monkeypatch.setenv(M.EXPLORE_ROLES_ENV, "gen")
            monkeypatch.setenv(M.EXPLORE_COLD_ENV, value)
            assert {M.pick("gen") for _ in range(50)} == {"a"}, value

    def test_a_resident_model_that_is_not_eligible_is_never_tried(self, monkeypatch):
        self._resident(monkeypatch, {"a", "b", "c"}, models="")
        monkeypatch.setenv(M.EXPLORE_ROLES_ENV, "gen")
        monkeypatch.setattr(M, "inventory", lambda base_url=None: {"a": GB, "b": GB})     # c is resident but gone
        assert {M.pick("gen") for _ in range(100)} == {"b"}

    def test_resident_only_still_respects_the_model_list(self, monkeypatch):
        self._resident(monkeypatch, {"a", "b", "c"}, models="b")
        assert {M.pick("gen") for _ in range(100)} == {"b"}

    def test_it_never_leaves_the_allow_list_or_the_eligible_set(self, monkeypatch):
        self._on(monkeypatch, models="b,ghost")
        monkeypatch.setattr(M, "inventory", lambda base_url=None: {"a": GB, "b": GB, "c": GB})
        assert {M.pick("gen") for _ in range(60)} == {"b"}      # c is eligible but not allowed; ghost is not installed

    def test_with_no_alternative_left_it_keeps_the_rules_choice(self, monkeypatch, tmp_path):
        self._on(monkeypatch, models="a")
        assert M.pick("gen") == "a"
        row = _decisions(tmp_path)[-1]
        assert row["n_alternatives"] == 0 and "explored" in row and row["explored"] is False

    def test_the_propensity_is_what_an_off_policy_estimate_needs(self, monkeypatch, tmp_path):
        self._on(monkeypatch, p="0.25", models="b,c")

        class Rng:
            def __init__(self, draws):
                self.draws = list(draws)

            def random(self):
                return self.draws.pop(0)

            def randrange(self, n):
                return 1
        monkeypatch.setattr(M, "_rng", Rng([0.9, 0.1]))
        assert M.pick("gen") == "a"                                 # 0.9 >= 0.25: the rule, probability 0.75
        assert M.pick("gen") == "c"                                 # 0.1 < 0.25: explore, uniform over b and c
        stay, explored = _decisions(tmp_path)
        assert (stay["explored"], stay["propensity"]) == (False, 0.75)
        assert (explored["explored"], explored["propensity"]) == (True, 0.125)

    def test_a_broken_environment_value_never_breaks_a_pick(self, monkeypatch):
        self._on(monkeypatch, p="not-a-number")
        assert M.pick("gen") == "a"


class TestReport:
    def _logs(self, tmp_path, decisions, outcomes):
        base = tmp_path / "instrumentation"
        base.mkdir(parents=True)
        (base / M.SHADOW_LOG_NAME).write_text("".join(json.dumps(r) + "\n" for r in decisions))
        (base / M.OUTCOMES_LOG_NAME).write_text("".join(json.dumps(r) + "\n" for r in outcomes))

    def _rows(self, arms):
        decisions, outcomes = [], []
        for model, n, ok in arms:
            for i in range(n):
                did = f"{model}{i:04d}"
                decisions.append({"role": "gen", "decision_id": did, "shadow_status": "no_selector",
                                  "chosen_returned": model, "explored": model != "a"})
                outcomes.append({"decision_id": did, "model": model, "ok": i < ok, "latency_ms": 100.0 + i})
        return decisions, outcomes

    def test_one_observed_arm_cannot_support_learning_and_says_so(self, tmp_path):
        self._logs(tmp_path, *self._rows([("a", 40, 40)]))
        rep = M.pool_report()
        gen = rep["roles"]["gen"]
        assert (gen["decisions"], gen["linked"], gen["learnable"]) == (40, 40, False)
        assert "only one arm has ever been observed" in gen["why"] and M.EXPLORE_ENV in gen["why"]

    def test_two_arms_with_enough_outcomes_can(self, tmp_path):
        self._logs(tmp_path, *self._rows([("a", 40, 40), ("b", 35, 30)]))
        gen = M.pool_report()["roles"]["gen"]
        assert gen["learnable"] is True and "2 arms" in gen["why"]
        assert gen["arms"]["b"] == {"outcomes": 35, "ok_rate": round(30 / 35, 3), "p50_ms": 117.0}
        assert gen["explored"] == 35

    def test_an_arm_below_the_minimum_does_not_count(self, tmp_path):
        self._logs(tmp_path, *self._rows([("a", 40, 40), ("b", 5, 5)]))
        gen = M.pool_report()["roles"]["gen"]
        assert gen["learnable"] is False and "fewer than two arms" in gen["why"]

    def test_rows_from_before_the_id_existed_are_counted_not_guessed_at(self, tmp_path):
        decisions, outcomes = self._rows([("a", 3, 3)])
        decisions += [{"role": "gen", "chosen_rule": "a", "chosen_shadow": None}] * 4
        self._logs(tmp_path, decisions, outcomes)
        rep = M.pool_report()
        assert rep["legacy_rows"] == 4 and rep["roles"]["gen"]["decisions"] == 3

    def test_status_counts_and_unlinked_decisions_are_reported(self, tmp_path):
        decisions, outcomes = self._rows([("a", 3, 3)])
        decisions.append({"role": "gen", "decision_id": "orphan000001", "shadow_status": "abstained"})
        self._logs(tmp_path, decisions, outcomes)
        gen = M.pool_report()["roles"]["gen"]
        assert (gen["decisions"], gen["linked"]) == (4, 3)
        assert gen["status"] == {"no_selector": 3, "abstained": 1}

    def test_no_logs_is_an_empty_report_not_an_error(self, tmp_path):
        assert M.pool_report() == {"roles": {}, "legacy_rows": 0, "min_arm_n": M.MIN_ARM_N, "decisions": 0, "outcomes": 0}

    def test_the_cli_prints_the_verdict(self, tmp_path, capsys):
        self._logs(tmp_path, *self._rows([("a", 40, 40)]))
        assert M._main(["report"]) == 0
        out = capsys.readouterr().out
        assert "decisions=40 outcomes=40" in out and "learnable=False" in out and "arm a: n=40 ok=100%" in out


class TestBaselineSelector:
    def _stats(self, monkeypatch, stats):
        monkeypatch.setattr(M, "outcomes_summary", lambda path=None: stats)

    def _features(self, *names, eligible=True):
        return [{"name": n, "rank": 1, "effective_rank": 1, "resident": False, "eligible": eligible} for n in names]

    def test_it_picks_the_best_success_rate_among_arms_with_enough_evidence(self, monkeypatch):
        self._stats(monkeypatch, {"a": {"calls": 90, "ok_rate": 0.90, "p50_ms": 100},
                                  "b": {"calls": 50, "ok_rate": 0.99, "p50_ms": 900}})
        assert pool_selectors.success_rate("gen", self._features("a", "b")) == "b"

    def test_ties_go_to_the_lower_latency_then_the_name(self, monkeypatch):
        self._stats(monkeypatch, {"a": {"calls": 40, "ok_rate": 1.0, "p50_ms": 300},
                                  "b": {"calls": 40, "ok_rate": 1.0, "p50_ms": 200},
                                  "c": {"calls": 40, "ok_rate": 1.0, "p50_ms": 200}})
        assert pool_selectors.success_rate("gen", self._features("a", "b", "c")) == "b"

    def test_it_abstains_without_evidence_and_ignores_ineligible_arms(self, monkeypatch):
        self._stats(monkeypatch, {"a": {"calls": 5, "ok_rate": 1.0, "p50_ms": 1}})
        assert pool_selectors.success_rate("gen", self._features("a", "b")) is None
        self._stats(monkeypatch, {"a": {"calls": 99, "ok_rate": 1.0, "p50_ms": 1}})
        pool_selectors._cache.update(at=0.0)
        assert pool_selectors.success_rate("gen", self._features("a", eligible=False)) is None
        assert pool_selectors.success_rate("gen", self._features("a")) == "a"      # positive twin

    def test_it_plugs_into_the_shadow_hook(self, monkeypatch, tmp_path):
        _pool(monkeypatch)
        monkeypatch.setenv(M.SHADOW_ENV, "1")
        monkeypatch.setenv(M.SELECTOR_ENV, "pool_selectors:success_rate")
        self._stats(monkeypatch, {"a": {"calls": 99, "ok_rate": 0.5, "p50_ms": 1}, "b": {"calls": 99, "ok_rate": 0.9, "p50_ms": 1}})
        assert M.pick("gen") == "a"
        row = _decisions(tmp_path)[-1]
        assert (row["shadow_status"], row["chosen_shadow"], row["agree"]) == ("chose", "b", False)
