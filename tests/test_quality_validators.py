"""Quality oracle (S12): deterministic validators, judge choice, the judge gate, judge-error correction, the gold script.

No model is called anywhere: the judge is a recording callable, the pool is a list of names, files live in tmp_path.

Mutation log (each applied to mcp/quality_validators.py or scripts/judge_gold_eval.py; the named tests went red; reverted):
- gate on agreement instead of false-accept (FA clause replaced by ``agreement < 0.6``): test_false_accept_alone_fails_the_gate
  (agreement 0.90 / kappa 0.8 with FA 0.20), test_accept_everything_judge_fails_with_false_accept_one
- hard-negative minimum 30 -> 29: test_one_hard_negative_short_of_the_minimum_fails
- gold minimum 100 -> 99: test_one_item_short_of_the_minimum_fails
- ``fa > FALSE_ACCEPT_MAX`` -> ``>=``: test_the_false_accept_boundary_is_inclusive
- kappa ``<`` -> ``<=`` KAPPA_MIN: test_kappa_exactly_at_the_minimum_passes
- the hard-negative false-accept clause deleted: test_a_judge_that_is_fooled_by_hard_negatives_fails_even_when_the_overall_rate_is_fine
- ``select_judge`` same_model branch removed (falls through to ``others[0]`` on an empty list -> IndexError, or returns the answerer): test_a_pool_of_one_model_is_same_model_not_a_self_grade
- empty-stream branch made a FAIL: test_an_empty_stream_is_excluded_from_reward_not_failed
- empty non-stream ``== "stop"`` check removed (survived the first run): test_only_a_normal_stop_makes_an_empty_answer_this_validators_failure
- tool-call arguments accepted when not a dict (``isinstance(..., dict)`` dropped): test_tool_call_arguments_must_be_a_json_object
- rogan_gladen denominator ``sens + spec - 1`` -> ``sens - spec``: test_rogan_gladen_matches_the_hand_computed_value
- result of the judge-model comparison removed from the gate (same_model_items): test_a_judge_that_answered_a_gold_item_fails_the_gate
- script writes ``gold_correct`` from the validator: test_candidates_carry_no_label
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "mcp") not in sys.path:
    sys.path.insert(0, str(_ROOT / "mcp"))

import quality_validators as qv  # noqa: E402


def _load_script():
    spec = importlib.util.spec_from_file_location("judge_gold_eval_under_test", _ROOT / "scripts" / "judge_gold_eval.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


script = _load_script()


# ------------------------------------------------------------------------------------------------- validators
def _call(args):
    return {"function": {"name": "read_file", "arguments": args}}


def _v(**kw):
    return qv.validate(qv.CallOutcome(**kw))


class TestToolCallJson:
    def test_a_truncated_arguments_string_fails(self):
        r = _v(text="", finish_reason="tool_calls", tool_calls=(_call('{"path": '),))
        assert (r.verdict, r.reasons, r.passed, r.reward_eligible) == (qv.FAIL, ("invalid_tool_call_json",), False, True)

    def test_the_valid_twin_passes(self):
        r = _v(text="", finish_reason="tool_calls", tool_calls=(_call('{"path": "a.txt"}'),))
        assert (r.verdict, r.reasons, r.passed) == (qv.PASS, (), True)

    @pytest.mark.parametrize("args", ["[]", '"x"', "3", "", "   ", None, '{"a": NaN}', 5, ["a"]])
    def test_tool_call_arguments_must_be_a_json_object(self, args):
        assert _v(tool_calls=(_call(args),)).reasons == ("invalid_tool_call_json",)

    @pytest.mark.parametrize("args", ['{"path": "a.txt"}', "{}", {"path": "a.txt"}, {}])
    def test_json_object_arguments_pass_as_text_or_as_a_mapping(self, args):
        assert _v(tool_calls=(_call(args),)).verdict == qv.PASS

    def test_the_flat_shape_and_a_missing_name(self):
        assert _v(tool_calls=({"name": "f", "arguments": "{}"},)).verdict == qv.PASS
        assert _v(tool_calls=({"name": "", "arguments": "{}"},)).reasons == ("invalid_tool_call_json",)
        assert _v(tool_calls=("not a mapping",)).reasons == ("invalid_tool_call_json",)

    def test_one_bad_call_among_good_ones_fails(self):
        assert _v(tool_calls=(_call("{}"), _call("{"), _call("{}"))).reasons == ("invalid_tool_call_json",)
        assert _v(tool_calls=(_call("{}"), _call("{}"))).verdict == qv.PASS


class TestEmpty:
    def test_an_empty_non_stream_answer_that_stopped_fails_and_has_valid_twins(self):
        r = _v(text="  \n", finish_reason="stop")
        assert (r.verdict, r.reasons, r.passed) == (qv.FAIL, ("empty_text_stop",), False)
        assert _v(text="the answer", finish_reason="stop").verdict == qv.PASS
        assert _v(text="", finish_reason="tool_calls", tool_calls=(_call("{}"),)).verdict == qv.PASS

    @pytest.mark.parametrize("reason", ["length", "", "content_filter"])
    def test_only_a_normal_stop_makes_an_empty_answer_this_validators_failure(self, reason):
        """An empty answer that did not stop normally is a completion problem the degeneracy judge counts, not a verdict here."""
        assert _v(text="", finish_reason=reason).verdict == qv.PASS
        assert _v(text="", finish_reason="stop").verdict == qv.FAIL

    def test_an_empty_stream_is_excluded_from_reward_not_failed(self):
        r = _v(text="", finish_reason="stop", stream=True)
        assert (r.verdict, r.reasons, r.passed, r.reward_eligible) == (qv.EXCLUDED, ("empty_suspect_stream",), None, False)
        assert _v(text="", finish_reason="", stream=True).verdict == qv.EXCLUDED
        twin = _v(text="the answer", finish_reason="stop", stream=True)
        assert (twin.verdict, twin.reward_eligible) == (qv.PASS, True)


class TestRefusal:
    @pytest.mark.parametrize("text", [
        "I'm sorry, but I can't help with that.",
        "I’m sorry, but I cannot assist with this request.",
        "Sorry, I am unable to comply.",
        "I cannot provide that information.",
        "As an AI language model, I cannot do that.",
        "  I apologize, but I can't continue with this.  ",
    ])
    def test_refusal_boilerplate_fails(self, text):
        r = _v(text=text, finish_reason="stop")
        assert (r.verdict, r.reasons) == (qv.FAIL, ("refusal_boilerplate",))

    @pytest.mark.parametrize("text", [
        "I can't believe how well this works: the label is B.",
        "Sorry for the delay, the label is B.",
        "As an engineer, I think the label is B.",
        "The model said: I'm sorry, but I can't help with that.",
    ])
    def test_the_valid_twins_pass(self, text):
        assert _v(text=text, finish_reason="stop").verdict == qv.PASS

    def test_a_long_answer_that_opens_with_an_apology_is_not_boilerplate(self):
        long_answer = "I'm sorry, but I can't tell from the first paragraph alone. " + "Reading the rest, the label is B. " * 20
        assert len(long_answer) > qv.REFUSAL_MAX_CHARS
        assert _v(text=long_answer, finish_reason="stop").verdict == qv.PASS
        assert _v(text=long_answer[:qv.REFUSAL_MAX_CHARS], finish_reason="stop").verdict == qv.FAIL

    def test_failures_accumulate_in_a_fixed_order(self):
        r = _v(text="I'm sorry, but I can't help with that.", tool_calls=(_call("{"),))
        assert r.reasons == ("invalid_tool_call_json", "refusal_boilerplate")


class TestFromMapping:
    def test_a_row_becomes_an_outcome(self):
        o = qv.CallOutcome.from_mapping({"text": "hi", "finish_reason": "stop", "tool_calls": [_call("{}")], "stream": 1})
        assert o == qv.CallOutcome(text="hi", finish_reason="stop", tool_calls=(_call("{}"),), stream=True)

    def test_missing_text_is_empty_and_a_non_list_tool_calls_value_is_invalid(self):
        assert qv.CallOutcome.from_mapping({"text": None, "finish_reason": "stop"}) == qv.CallOutcome(finish_reason="stop")
        assert qv.validate(qv.CallOutcome.from_mapping({"tool_calls": "oops"})).reasons == ("invalid_tool_call_json",)


# ------------------------------------------------------------------------------------------------- judge choice
class TestSelectJudge:
    def test_a_pool_of_one_model_is_same_model_not_a_self_grade(self):
        assert qv.select_judge("m1", ["m1"]) == qv.JudgeChoice("", "same_model")
        assert qv.select_judge("m1", ["m1:latest", "m1"]) == qv.JudgeChoice("", "same_model")
        assert qv.select_judge("m1:latest", ["m1"]) == qv.JudgeChoice("", "same_model")

    def test_a_second_model_is_the_judge(self):
        assert qv.select_judge("m1", ["m1", "m2", "m3"]) == qv.JudgeChoice("m2", "")

    def test_an_empty_pool_has_no_other_model(self):
        assert qv.select_judge("m1", []) == qv.JudgeChoice("", "no_other_model")
        assert qv.select_judge("m1", ["", None]) == qv.JudgeChoice("", "no_other_model")

    def test_the_highest_cost_arm_never_judges(self):
        costs = {"cheap": Decimal("0.001"), "mid": Decimal("0.01"), "big": Decimal("0.5")}
        asked = []

        def cost_of(name):
            asked.append(name)
            return costs[name]
        assert qv.select_judge("cheap", ["big", "mid"], cost_of) == qv.JudgeChoice("mid", "")
        assert sorted(set(asked)) == ["big", "cheap", "mid"]
        assert qv.select_judge("cheap", ["big"], cost_of) == qv.JudgeChoice("", "highest_cost_arm")

    def test_equal_costs_do_not_exclude_anyone(self):
        assert qv.select_judge("a", ["b"], lambda n: Decimal(0)) == qv.JudgeChoice("b", "")

    @pytest.mark.parametrize("bad", [lambda n: 0.5, lambda n: None, lambda n: Decimal("NaN")])
    def test_a_cost_that_cannot_be_trusted_fails_closed(self, bad):
        assert qv.select_judge("a", ["b"], bad) == qv.JudgeChoice("", "cost_unknown")

    def test_a_cost_reader_that_raises_fails_closed(self):
        def boom(name):
            raise KeyError(name)
        assert qv.select_judge("a", ["b"], boom) == qv.JudgeChoice("", "cost_unknown")


# ------------------------------------------------------------------------------------------------- judge gate
def _items(pos=100, neg_easy=70, neg_hard=30, fn=0, fa_easy=0, fa_hard=0):
    out = []
    for i in range(pos):
        out.append(qv.GoldItem(f"p{i}", True, i >= fn))
    for i in range(neg_easy):
        out.append(qv.GoldItem(f"e{i}", False, i < fa_easy, bucket="small"))
    for i in range(neg_hard):
        out.append(qv.GoldItem(f"h{i}", False, i < fa_hard, hard_negative=True, bucket="large"))
    return out


class TestJudgeGate:
    def test_accept_everything_judge_fails_with_false_accept_one(self):
        rep = qv.evaluate_gate(_items(pos=70, neg_easy=0, neg_hard=30, fa_hard=30))
        assert (rep.n, rep.n_hard_negative, rep.false_accept, rep.false_accept_hard, rep.kappa) == (100, 30, 1.0, 1.0, 0.0)
        assert rep.agreement == 0.7
        assert rep.passed is False
        assert rep.reasons == ("false_accept_too_high", "false_accept_hard_too_high", "kappa_too_low")

    def test_ten_false_accepts_of_one_hundred_negatives_is_exactly_point_one_and_passes(self):
        rep = qv.evaluate_gate(_items(fa_easy=10))
        assert (rep.tp, rep.fp, rep.fn, rep.tn) == (100, 10, 0, 90)
        assert rep.false_accept == 0.1
        assert rep.false_accept_hard == 0.0
        assert rep.kappa == 0.9                                  # po 0.95, pe 0.5 -> 0.9
        assert (rep.sensitivity, rep.specificity) == (1.0, 0.9)
        assert (rep.passed, rep.reasons) == (True, ())

    def test_kappa_is_reported_to_three_places(self):
        # tp 20, fp 5, fn 10, tn 15 -> po 0.7, pe 0.5, kappa 0.4 exactly
        assert qv.cohens_kappa(20, 5, 10, 15) == Fraction(2, 5)
        # tp 40 fp 10 fn 5 tn 45: po .85, judge pos .5, gold pos .45 -> pe .5*.45+.5*.55=.5 -> kappa .7
        assert qv.cohens_kappa(40, 10, 5, 45) == Fraction(7, 10)
        # 1/3 repeating: tp 1 fp 0 fn 1 tn 1 -> po 2/3, pe (1*2+2*1)/9 = 4/9 -> kappa (2/3-4/9)/(5/9) = 2/5
        assert qv.cohens_kappa(1, 0, 1, 1) == Fraction(2, 5)
        items = [qv.GoldItem(f"a{i}", True, True) for i in range(3)] + [qv.GoldItem(f"b{i}", False, False) for i in range(2)] \
            + [qv.GoldItem("c", True, False), qv.GoldItem("d", False, True)]
        # tp 3 fp 1 fn 1 tn 2: n 7, po 5/7, pe (4*4 + 3*3)/49 = 25/49, kappa (35-25)/(49-25) = 10/24 = 0.41666..
        assert qv.evaluate_gate(items).kappa == 0.417

    def test_false_accept_alone_fails_the_gate(self):
        rep = qv.evaluate_gate(_items(fa_easy=20))
        assert (rep.false_accept, rep.agreement, rep.kappa) == (0.2, 0.9, 0.8)
        assert (rep.passed, rep.reasons) == (False, ("false_accept_too_high",))

    def test_the_false_accept_boundary_is_inclusive(self):
        assert qv.evaluate_gate(_items(fa_easy=15)).passed is True
        sixteen = qv.evaluate_gate(_items(fa_easy=16))
        assert (sixteen.false_accept, sixteen.passed, sixteen.reasons) == (0.16, False, ("false_accept_too_high",))

    def test_a_judge_that_is_fooled_by_hard_negatives_fails_even_when_the_overall_rate_is_fine(self):
        rep = qv.evaluate_gate(_items(fa_hard=5))
        assert (rep.false_accept, rep.false_accept_hard) == (0.05, 5 / 30)
        assert (rep.passed, rep.reasons) == (False, ("false_accept_hard_too_high",))
        assert qv.evaluate_gate(_items(fa_hard=4)).passed is True      # 4/30 = 0.133 <= 0.15

    def test_kappa_alone_fails_the_gate(self):
        rep = qv.evaluate_gate(_items(fn=50, neg_easy=70, neg_hard=30))
        assert (rep.false_accept, rep.kappa) == (0.0, 0.5)
        assert (rep.passed, rep.reasons) == (False, ("kappa_too_low",))

    def test_kappa_exactly_at_the_minimum_passes(self):
        at = qv.evaluate_gate(_items(fn=40))                           # tp 60, fn 40, tn 100 -> kappa exactly 0.6
        assert (at.kappa, at.passed, at.reasons) == (0.6, True, ())
        below = qv.evaluate_gate(_items(fn=41))                        # kappa 0.59
        assert (below.kappa, below.passed, below.reasons) == (0.59, False, ("kappa_too_low",))

    def test_one_item_short_of_the_minimum_fails(self):
        fits = qv.evaluate_gate(_items(pos=50, neg_easy=20, neg_hard=30))
        assert (fits.n, fits.passed, fits.reasons) == (100, True, ())
        short = qv.evaluate_gate(_items(pos=49, neg_easy=20, neg_hard=30))
        assert (short.n, short.passed, short.reasons) == (99, False, ("insufficient_gold",))

    def test_one_hard_negative_short_of_the_minimum_fails(self):
        enough = qv.evaluate_gate(_items(neg_easy=70, neg_hard=30))
        assert (enough.n_hard_negative, enough.passed) == (30, True)
        short = qv.evaluate_gate(_items(neg_easy=71, neg_hard=29))
        assert (short.n_hard_negative, short.passed, short.reasons) == (29, False, ("insufficient_hard_negatives",))

    def test_items_without_a_judge_verdict_are_counted_not_graded(self):
        padded = _items() + [qv.GoldItem(f"u{i}", False, None, hard_negative=True) for i in range(10)]
        rep = qv.evaluate_gate(padded)
        assert (rep.n, rep.n_unusable, rep.n_hard_negative, rep.passed) == (200, 10, 30, True)
        only_unusable_hard = _items(neg_hard=0) + [qv.GoldItem(f"u{i}", False, None, hard_negative=True) for i in range(30)]
        thin = qv.evaluate_gate(only_unusable_hard)
        assert (thin.n_hard_negative, thin.n_unusable, thin.passed) == (0, 30, False)
        assert "insufficient_hard_negatives" in thin.reasons

    def test_malformed_input_fails_closed(self):
        assert qv.evaluate_gate(_items(), malformed=1).reasons == ("malformed_items",)
        assert qv.evaluate_gate(_items(), malformed=0).reasons == ()

    def test_a_judge_that_answered_a_gold_item_fails_the_gate(self):
        items = _items()
        items[0] = qv.GoldItem("p0", True, True, answerer="m1:latest", judge_model="m1")
        assert qv.evaluate_gate(items).reasons == ("same_model_items",)
        items[0] = qv.GoldItem("p0", True, True, answerer="m1", judge_model="m2")
        assert qv.evaluate_gate(items).reasons == ()

    def test_a_gold_set_with_nothing_wrong_in_it_cannot_pass(self):
        rep = qv.evaluate_gate([qv.GoldItem(f"p{i}", True, True) for i in range(100)])
        assert rep.false_accept is None and rep.kappa is None
        assert rep.reasons == ("insufficient_hard_negatives", "false_accept_too_high", "false_accept_hard_too_high",
                               "kappa_undefined")

    def test_an_empty_gold_set_fails(self):
        rep = qv.evaluate_gate([])
        assert (rep.n, rep.passed, rep.agreement) == (0, False, None)
        assert "insufficient_gold" in rep.reasons

    def test_false_accept_is_reported_per_bucket(self):
        rep = qv.evaluate_gate(_items(fa_easy=10, fa_hard=3))
        assert rep.by_bucket == {"large": 3 / 30, "small": 10 / 70}

    def test_report_to_dict_is_plain_json(self):
        d = qv.evaluate_gate(_items(fa_easy=10)).to_dict()
        assert json.loads(json.dumps(d))["reasons"] == []
        assert d["false_accept"] == 0.1


class TestGoldItemParsing:
    def test_a_good_row(self):
        row = {"id": "x", "gold_correct": False, "judge_accept": True, "hard_negative": True, "bucket": "b",
               "answerer": "m1", "judge_model": "m2"}
        assert qv.GoldItem.from_mapping(row) == qv.GoldItem("x", False, True, True, "b", "m1", "m2")
        assert qv.GoldItem.from_mapping({"id": "y", "gold_correct": True, "judge_accept": None}) == qv.GoldItem("y", True, None)

    @pytest.mark.parametrize("row, pattern", [
        ({"id": "x", "gold_correct": None}, "gold_correct must be a bool"),
        ({"id": "x", "gold_correct": "yes"}, "gold_correct must be a bool"),
        ({"id": "x", "gold_correct": True, "judge_accept": 1}, "judge_accept must be a bool or null"),
        ({"id": "x", "gold_correct": True, "hard_negative": True}, "only a gold-incorrect item"),
    ])
    def test_a_bad_row_is_refused(self, row, pattern):
        with pytest.raises(qv.GoldFormatError, match=pattern):
            qv.GoldItem.from_mapping(row)


# ------------------------------------------------------------------------------------------------- judge error
class TestRoganGladen:
    def test_rogan_gladen_matches_the_hand_computed_value(self):
        # sens .9, spec .8, observed accept .7 -> (0.7 + 0.8 - 1) / (0.9 + 0.8 - 1) = 0.5 / 0.7 = 5/7
        assert abs(qv.rogan_gladen(0.7, 0.9, 0.8) - 5 / 7) < 1e-6
        assert abs(qv.rogan_gladen(0.7, 0.9, 0.8) - 0.714285714) < 1e-6
        # a second table row: observed .55 -> 0.35 / 0.7 = 0.5 exactly
        assert abs(qv.rogan_gladen(0.55, 0.9, 0.8) - 0.5) < 1e-6

    def test_a_perfect_judge_needs_no_correction(self):
        assert abs(qv.rogan_gladen(0.37, 1.0, 1.0) - 0.37) < 1e-12

    def test_the_estimate_is_clamped_to_a_probability_unless_asked_not_to(self):
        assert qv.rogan_gladen(0.1, 0.9, 0.8) == 0.0
        assert abs(qv.rogan_gladen(0.1, 0.9, 0.8, clamp=False) - (-1 / 7)) < 1e-6
        assert qv.rogan_gladen(1.0, 0.9, 0.8) == 1.0
        assert abs(qv.rogan_gladen(1.0, 0.9, 0.8, clamp=False) - 0.8 / 0.7) < 1e-6

    @pytest.mark.parametrize("sens, spec", [(0.5, 0.5), (0.4, 0.5), (0.0, 1.0)])
    def test_an_uninformative_judge_cannot_be_corrected(self, sens, spec):
        with pytest.raises(ValueError, match="uninformative"):
            qv.rogan_gladen(0.5, sens, spec)

    @pytest.mark.parametrize("obs", [1.5, -0.1, float("nan")])
    def test_a_non_probability_is_refused(self, obs):
        with pytest.raises(ValueError, match="observed must be a probability"):
            qv.rogan_gladen(obs, 0.9, 0.8)

    def test_the_gate_report_supplies_the_confusion_matrix(self):
        rep = qv.evaluate_gate(_items(fa_easy=10, fn=20))
        assert (rep.sensitivity, rep.specificity) == (0.8, 0.9)
        assert abs(qv.rogan_gladen(0.7, rep.sensitivity, rep.specificity) - (0.7 + 0.9 - 1) / (0.8 + 0.9 - 1)) < 1e-9


# ------------------------------------------------------------------------------------------------- script
def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _gold_rows(**kw):
    return [{"id": i.item_id, "gold_correct": i.gold_correct, "judge_accept": i.judge_accept,
             "hard_negative": i.hard_negative, "bucket": i.bucket} for i in _items(**kw)]


class TestGoldScript:
    def test_a_passing_gold_file_exits_zero_and_says_pass(self, tmp_path):
        out = io.StringIO()
        code = script.main(["gate", "--gold", str(_write(tmp_path / "g.jsonl", _gold_rows(fa_easy=10)))], out)
        text = out.getvalue()
        assert code == 0
        assert text.splitlines()[0] == "judge gate: PASS"
        assert "false_accept=0.100" in text and "kappa=0.900" in text

    def test_a_failing_gold_file_exits_one_and_names_the_reasons(self, tmp_path):
        out = io.StringIO()
        code = script.main(["gate", "--gold", str(_write(tmp_path / "g.jsonl", _gold_rows(fa_easy=20)))], out)
        text = out.getvalue()
        assert code == 1
        assert text.splitlines()[0] == "judge gate: FAIL"
        assert "failed because: false_accept_too_high" in text
        assert "the judge gates nothing until this passes" in text

    def test_json_output_is_the_report_dict(self, tmp_path):
        out = io.StringIO()
        script.main(["gate", "--gold", str(_write(tmp_path / "g.jsonl", _gold_rows(fa_easy=10))), "--json"], out)
        assert json.loads(out.getvalue()) == qv.evaluate_gate(_items(fa_easy=10)).to_dict()

    def test_unreadable_and_unlabelled_rows_fail_the_gate_closed(self, tmp_path):
        rows = _gold_rows()
        path = _write(tmp_path / "g.jsonl", rows)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("not json\n")
        assert script.evaluate_file(path).reasons == ("malformed_items",)
        path2 = _write(tmp_path / "g2.jsonl", rows + [{"id": "u", "gold_correct": None, "judge_accept": True}])
        assert script.evaluate_file(path2).reasons == ("malformed_items",)
        assert script.evaluate_file(_write(tmp_path / "g3.jsonl", rows)).reasons == ()

    def test_an_injected_judge_replaces_the_recorded_verdict_and_nothing_else_is_called(self, tmp_path):
        seen = []

        def judge(row):
            seen.append(row["id"])
            return not row["gold_correct"] and row["id"].startswith("e") and int(row["id"][1:]) < 10 or row["gold_correct"]
        recorded_accept_all = [{**r, "judge_accept": True} for r in _gold_rows()]
        rep = script.evaluate_file(_write(tmp_path / "g.jsonl", recorded_accept_all), judge=judge)
        assert len(seen) == 200 and seen[0] == "p0"
        assert (rep.fp, rep.false_accept, rep.passed) == (10, 0.1, True)

    def test_a_judge_that_raises_or_returns_junk_is_unusable_never_an_accept(self, tmp_path):
        def judge(row):
            if row["id"] == "e0":
                raise ConnectionError("down")
            return "yes" if row["id"] == "e1" else row["gold_correct"]
        rep = script.evaluate_file(_write(tmp_path / "g.jsonl", _gold_rows()), judge=judge)
        assert (rep.n, rep.n_unusable, rep.fp, rep.passed) == (198, 2, 0, True)


def _outcome_rows():
    return [
        {"id": "ok-accepted", "text": "label B", "finish_reason": "stop", "judge_accept": True, "bucket": "small"},
        {"id": "bad-accepted", "text": "", "finish_reason": "stop", "judge_accept": True, "bucket": "small",
         "answerer": "m1", "existing_grade": False, "grader": "gold"},
        {"id": "ok-rejected", "text": "label A", "finish_reason": "stop", "judge_accept": False},
        {"id": "bad-rejected", "text": "I'm sorry, but I can't help with that.", "finish_reason": "stop", "judge_accept": False},
        {"id": "stream-empty", "text": "", "finish_reason": "stop", "stream": True, "judge_accept": True},
        {"id": "no-judge", "text": "label C", "finish_reason": "stop"},
        {"id": "bad-json", "tool_calls": [_call("{")], "judge_accept": True, "text": "x" * 400},
    ]


class TestCandidates:
    def test_candidates_carry_no_label(self):
        cands, _ = script.build_candidates(_outcome_rows())
        assert [c["gold_correct"] for c in cands] == [None] * 7
        by_id = {c["id"]: c for c in cands}
        assert by_id["bad-accepted"] == {
            "id": "bad-accepted", "bucket": "small", "answerer": "m1", "validator_verdict": "fail",
            "validator_reasons": ["empty_text_stop"], "judge_accept": True, "existing_grade": False, "grader": "gold",
            "candidate_hard_negative": True, "gold_correct": None}
        assert by_id["ok-accepted"]["validator_verdict"] == "pass" and by_id["ok-accepted"]["candidate_hard_negative"] is False
        assert by_id["stream-empty"]["validator_verdict"] == "excluded"
        assert by_id["stream-empty"]["candidate_hard_negative"] is False
        assert by_id["no-judge"]["judge_accept"] is None

    def test_disagreements_are_judge_versus_validator_only(self):
        _, dis = script.build_candidates(_outcome_rows())
        assert [(d["id"], d["kind"]) for d in dis] == [
            ("bad-accepted", "judge_accept_validator_fail"),
            ("ok-rejected", "judge_reject_validator_pass"),
            ("bad-json", "judge_accept_validator_fail")]
        assert dis[0] == {"id": "bad-accepted", "kind": "judge_accept_validator_fail",
                          "validator_reasons": ["empty_text_stop"], "judge_accept": True, "bucket": "small", "excerpt": ""}
        assert dis[1]["excerpt"] == "label A"
        assert len(dis[2]["excerpt"]) == 300

    def test_the_cli_writes_new_files_and_reports_zero_labelled(self, tmp_path):
        src = _write(tmp_path / "o.jsonl", _outcome_rows())
        out = io.StringIO()
        code = script.main(["candidates", "--outcomes", str(src), "--out", str(tmp_path / "c.jsonl"),
                            "--disagreements", str(tmp_path / "d.jsonl")], out)
        assert code == 0
        assert out.getvalue().strip() == "candidates=7 disagreements=3 unreadable_lines=0 labelled=0"
        written = [json.loads(line) for line in (tmp_path / "c.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [w["gold_correct"] for w in written] == [None] * 7
        assert len((tmp_path / "d.jsonl").read_text(encoding="utf-8").splitlines()) == 3

    def test_an_existing_output_is_never_overwritten(self, tmp_path):
        src = _write(tmp_path / "o.jsonl", _outcome_rows())
        (tmp_path / "c.jsonl").write_text("keep me", encoding="utf-8")
        out = io.StringIO()
        code = script.main(["candidates", "--outcomes", str(src), "--out", str(tmp_path / "c.jsonl"),
                            "--disagreements", str(tmp_path / "d.jsonl")], out)
        assert code == 2
        assert out.getvalue().strip() == f"refusing to overwrite {tmp_path / 'c.jsonl'}"
        assert (tmp_path / "c.jsonl").read_text(encoding="utf-8") == "keep me"
        assert not (tmp_path / "d.jsonl").exists()

    def test_a_missing_input_is_exit_two(self, tmp_path):
        out = io.StringIO()
        code = script.main(["gate", "--gold", str(tmp_path / "absent.jsonl")], out)
        assert code == 2
        assert out.getvalue().startswith("error: ")

    def test_bad_arguments_are_exit_two(self):
        assert script.main(["nonsense"], io.StringIO()) == 2
