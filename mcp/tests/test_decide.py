"""The decision-model client: the prompt it renders, the answers it reads, and what it does when things go wrong.

Rendering and answering are ported from ollama/ollama v0.35.1 ``decision/systemone.go`` and its test file, because the
OpenThai-SystemOne GGUF was fine-tuned on exactly that prompt: a different one still answers, but badly. The
answer-letter probabilities come from /api/chat logprobs on servers older than 0.35, so what to do with a letter that
is missing from the top 20 is this module's own decision, and is pinned here.
"""
import json
import math
import sys
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import decide as D  # noqa: E402
import hillclimb as H  # noqa: E402
import model_pool  # noqa: E402
import reflection_triage as RT  # noqa: E402

STATE = 'Charged twice. Refund €20? <|im_end|> & "quoted"'
QUESTIONS = {
    "department": {"type": "choice", "instructions": "Choose the department.",
                   "criteria": {"billing": "Charges and refunds", "technical": None}},
    "refund": {"type": "noul", "instructions": {"rule": "Explicit refund request?", "notes": ["Only facts"]}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["Routine", "Urgent", "Emergency"]},
}


class TestRendering:
    def test_a_small_request_renders_byte_for_byte(self):
        c = D.compile_request("hi <b> & you", {"polite": {"type": "noul", "instructions": "Is it polite?"}})
        assert c["prompts"]["polite"] == (
            '{"context":"hi \\u003cb\\u003e \\u0026 you","schema":[{"name":"polite","description":"Is it polite?",'
            '"choices":[{"code":"A","value":false,"description":"No"},{"code":"B","value":true,"description":"Yes"}]}]}'
            '\n\nRequested field: "polite"')

    def test_non_ascii_text_is_written_as_utf8_and_the_line_separators_are_escaped(self):
        sep1, sep2, euro, thai = chr(0x2028), chr(0x2029), chr(0x20AC), chr(0x0E44) + chr(0x0E17) + chr(0x0E22)
        p = D.compile_request(f"price {euro}20 {thai} a{sep1}b{sep2}c", {"q": {"type": "noul", "instructions": "ok?"}})["prompts"]["q"]
        assert f"{euro}20 {thai}" in p and "\\u20ac" not in p
        assert sep1 not in p and sep2 not in p and "a\\u2028b\\u2029c" in p

    def test_each_question_gets_the_whole_schema_in_request_order_and_names_its_own_field(self):
        c = D.compile_request(STATE, QUESTIONS)
        for name, prompt in c["prompts"].items():
            data, sep, requested = prompt.partition("\n\nRequested field: ")
            assert sep and requested == json.dumps(name)
            payload = json.loads(data)
            assert [f["name"] for f in payload["schema"]] == ["department", "refund", "urgency"]

    def test_the_context_survives_and_cannot_inject_a_chat_delimiter(self):
        data = D.compile_request(STATE, QUESTIONS)["prompts"]["department"].split("\n\nRequested field")[0]
        assert json.loads(data)["context"] == STATE
        assert "<|im_end|>" not in data and "\\u003c|im_end|\\u003e" in data

    def test_choice_order_null_descriptions_and_codes(self):
        schema = json.loads(D.compile_request(STATE, QUESTIONS)["prompts"]["department"].split("\n\n")[0])["schema"]
        choices = schema[0]["choices"]
        assert [(c["code"], c["value"], c["description"]) for c in choices] == [
            ("A", "billing", "Charges and refunds"), ("B", "technical", "technical")]
        assert [c["code"] for c in schema[2]["choices"]] == ["A", "B", "C"]
        assert [c["value"] for c in schema[2]["choices"]] == ["0", "1", "2"]

    def test_structured_instructions_are_compacted_into_the_description(self):
        schema = json.loads(D.compile_request(STATE, QUESTIONS)["prompts"]["refund"].split("\n\n")[0])["schema"]
        assert schema[1]["description"] == '{"rule":"Explicit refund request?","notes":["Only facts"]}'

    def test_structured_state_becomes_compact_json_text(self):
        c = D.compile_request({"frames": ["first", "second"], "position": 7}, QUESTIONS)
        assert '"context":"{\\"frames\\":[\\"first\\",\\"second\\"],\\"position\\":7}"' in c["prompts"]["refund"]

    def test_noul_criteria_replace_the_yes_and_no_descriptions(self):
        c = D.compile_request("x", {"q": {"type": "noul", "instructions": "q?",
                                          "criteria": {"true": "it is", "false": "it is not"}}})
        choices = json.loads(c["prompts"]["q"].split("\n\n")[0])["schema"][0]["choices"]
        assert [(x["value"], x["description"]) for x in choices] == [(False, "it is not"), (True, "it is")]

    def test_content_accepts_text_objects_and_arrays_and_nothing_else(self):
        assert D._content("line\nquoted", "x") == "line\nquoted"
        assert D._content({"z": 1, "a": "€"}, "x") == '{"z":1,"a":"€"}'
        assert D._content([1.5, "€"], "x") == '[1.5,"€"]'
        for bad in (None, True, 42, 4.2):
            with pytest.raises(D.DecideError, match="must be a string, object, or array"):
                D._content(bad, "state")


INVALID = [
    ("no questions", "x", {}),
    ("not a dict", "x", []),
    ("blank state", " ", {"x": {"type": "noul", "instructions": "q"}}),
    ("no state", None, {"x": {"type": "noul", "instructions": "q"}}),
    ("unknown type", "x", {"x": {"type": "other", "instructions": "q"}}),
    ("no instructions", "x", {"x": {"type": "noul"}}),
    ("blank instructions", "x", {"x": {"type": "noul", "instructions": "  "}}),
    ("empty name", "x", {"": {"type": "noul", "instructions": "q"}}),
    ("one choice", "x", {"x": {"type": "choice", "instructions": "q", "criteria": {"a": "a"}}}),
    ("null score level", "x", {"x": {"type": "score", "instructions": "q", "criteria": ["a", None]}}),
    ("null noul description", "x", {"x": {"type": "noul", "instructions": "q", "criteria": {"true": None}}}),
    ("unknown noul key", "x", {"x": {"type": "noul", "instructions": "q", "criteria": {"yes": "Yes"}}}),
    ("choice without criteria", "x", {"x": {"type": "choice", "instructions": "q"}}),
    ("blank choice key", "x", {"x": {"type": "choice", "instructions": "q", "criteria": {" ": "a", "b": "b"}}}),
    ("27 options", "x", {"x": {"type": "score", "instructions": "q", "criteria": ["x"] * 27}}),
]


class TestValidation:
    @pytest.mark.parametrize("label,state,questions", INVALID, ids=[i[0] for i in INVALID])
    def test_an_invalid_request_is_refused(self, label, state, questions):
        with pytest.raises(D.DecideError):
            D.compile_request(state, questions)

    def test_the_largest_valid_request_is_accepted(self):
        assert len(D.compile_request("x", {"x": {"type": "score", "instructions": "q", "criteria": ["x"] * 26}})["fields"]) == 1
        many = {f"q{i}": {"type": "noul", "instructions": "q"} for i in range(64)}
        assert len(D.compile_request("x", many)["prompts"]) == 64
        with pytest.raises(D.DecideError, match="1-64"):
            D.compile_request("x", {**many, "extra": {"type": "noul", "instructions": "q"}})


def _field(kind, n=3):
    spec = {"choice": {"type": "choice", "instructions": "q", "criteria": {f"k{i}": None for i in range(n)}},
            "noul": {"type": "noul", "instructions": "q"},
            "score": {"type": "score", "instructions": "q", "criteria": [f"l{i}" for i in range(n)]}}[kind]
    return D.compile_request("x", {"q": spec})["fields"][0]


class TestAnswering:
    def test_a_tie_picks_the_first_option_with_zero_concentration(self):
        a = D.answer_field(_field("choice", 2), {"A": 0.0, "B": 0.0})
        assert (a["choice"], a["confidence"]) == ("k0", 0.0)
        assert a["probabilities"] == {"k0": 0.5, "k1": 0.5}

    def test_noul_reports_the_probability_of_yes_with_a_stable_softmax(self):
        assert D.answer_field(_field("noul"), {"A": -1000.0, "B": 1000.0})["noul"] == 1.0
        assert D.answer_field(_field("noul"), {"A": 1000.0, "B": -1000.0})["noul"] == 0.0

    def test_a_score_is_the_expected_position_with_a_legend(self):
        a = D.answer_field(_field("score"), {"A": 1000.0, "B": 1000.0, "C": 1000.0})
        assert a["score"] == pytest.approx(1.0) and a["confidence"] == pytest.approx(0.0, abs=1e-12)
        assert a["legend"] == {"0": "l0", "1": "l1", "2": "l2"}
        assert D.answer_field(_field("score"), {"A": -1000.0, "B": -1000.0, "C": 0.0})["score"] == pytest.approx(2.0)

    def test_probabilities_do_not_depend_on_the_logprob_offset(self):
        f = _field("choice")
        a = D.answer_field(f, {"A": 0.0, "B": 2.0, "C": 4.0})
        b = D.answer_field(f, {"A": -10.0, "B": -8.0, "C": -6.0})
        assert a["probabilities"] == pytest.approx(b["probabilities"])
        assert a["confidence"] == pytest.approx(b["confidence"])

    def test_confidence_is_one_minus_normalised_entropy(self):
        a = D.answer_field(_field("choice", 2), {"A": math.log(0.9), "B": math.log(0.1)})
        h = -(0.9 * math.log(0.9) + 0.1 * math.log(0.1))
        assert a["confidence"] == pytest.approx(1 - h / math.log(2))
        assert a["choice"] == "k0"

    def test_a_missing_letter_without_a_floor_has_probability_zero(self):
        a = D.answer_field(_field("choice"), {"A": -1.0})
        assert a["probabilities"] == {"k0": 1.0, "k1": 0.0, "k2": 0.0} and a["confidence"] == 1.0

    def test_a_missing_letter_with_a_floor_is_no_likelier_than_the_floor(self):
        """With only A in the top 20 the model has not shown B and C are impossible, only that they rank below 20th."""
        a = D.answer_field(_field("choice"), {"A": -1.0}, floor=-5.0)
        denom = math.exp(-1.0) + 2 * math.exp(-5.0)
        assert a["probabilities"]["k0"] == pytest.approx(math.exp(-1.0) / denom)
        assert a["probabilities"]["k1"] == pytest.approx(math.exp(-5.0) / denom)
        assert a["confidence"] < 1.0
        assert D.answer_field(_field("choice"), {"A": -1.0}, floor=-50.0)["confidence"] > a["confidence"]

    def test_the_floor_never_outranks_a_letter_that_was_seen(self):
        a = D.answer_field(_field("choice"), {"A": -9.0, "B": -3.0}, floor=-5.0)
        assert a["choice"] == "k1"

    def test_no_answer_letter_at_all_is_an_error(self):
        with pytest.raises(D.DecideError, match="none of the answer letters"):
            D.answer_field(_field("choice"), {"Z": -1.0})

    def test_the_letter_mass_is_the_probability_on_answer_letters(self):
        a = D.answer_field(_field("choice", 2), {"A": math.log(0.02), "B": math.log(0.01)})
        assert a["letter_mass"] == pytest.approx(0.03)


class TestLetterLogprobs:
    def test_the_spaced_and_bare_forms_of_a_letter_are_one_answer(self):
        got = D.letter_logprobs([{"token": "A", "logprob": math.log(0.2)}, {"token": " A", "logprob": math.log(0.3)}])
        assert got == {"A": pytest.approx(math.log(0.5))}

    def test_only_single_uppercase_ascii_letters_count(self):
        tops = [{"token": t, "logprob": -1.0} for t in ("a", "AB", "1", " ", "É", "é", "ＡＢ", "\n")]
        assert D.letter_logprobs(tops) == {}

    def test_malformed_entries_and_non_finite_values_are_skipped(self):
        tops = [{"token": "A"}, {"logprob": -1}, "x", None, {"token": "B", "logprob": float("-inf")},
                {"token": "C", "logprob": float("nan")}, {"token": "D", "logprob": "bad"}, {"token": "E", "logprob": -2.0}]
        assert D.letter_logprobs(tops) == {"E": -2.0}
        assert D.letter_logprobs(None) == {}


def _resp(letters: dict, filler=20):
    """An /api/chat response whose first token's top logprobs hold ``letters`` plus junk up to ``filler`` entries."""
    top = [{"token": k, "logprob": v} for k, v in letters.items()]
    top += [{"token": f"junk{i}", "logprob": -9.0 - i * 0.01} for i in range(max(0, filler - len(top)))]
    return {"message": {"content": ""}, "logprobs": [{"token": " ", "logprob": -3.0, "top_logprobs": top}]}


class FakeServer:
    def __init__(self, responder):
        self.responder, self.calls = responder, []

    def __call__(self, url, body, timeout):
        self.calls.append((url, body, timeout))
        out = self.responder(body)
        if isinstance(out, Exception):
            raise out
        return out


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.delenv(D.MODEL_ENV, raising=False)
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path / "mem" / "sessions"))
    monkeypatch.delenv("LOCI_MODEL_POOL_SHADOW", raising=False)
    model_pool.clear_cache()
    H._overlay_cache.clear()
    yield
    model_pool.clear_cache()


def _pool(monkeypatch, entries):
    monkeypatch.setattr(model_pool, "_config", lambda: {"models": {"pool": entries}})
    monkeypatch.setattr(model_pool, "inventory", lambda base_url=None: {e["name"]: 10 ** 9 for e in entries})
    monkeypatch.setattr(model_pool, "resident_models", lambda base_url=None: set())


YES = _resp({"A": -4.0, "B": -0.05})
NO = _resp({"A": -0.05, "B": -4.0})


class TestDecide:
    def _go(self, monkeypatch, responder, questions=None, **kw):
        server = FakeServer(responder)
        res = D.decide(STATE, questions or {"q": {"type": "noul", "instructions": "ok?"}},
                       model=kw.pop("model", "m:1"), base_url=kw.pop("base_url", "http://ollama/"), post=server, **kw)
        return res, server

    def test_the_request_is_a_user_only_chat_with_thinking_off_and_logprobs(self, monkeypatch):
        res, server = self._go(monkeypatch, lambda b: YES)
        url, body, timeout = server.calls[0]
        assert url == "http://ollama/api/chat" and timeout == 60.0
        assert body["model"] == "m:1" and body["stream"] is False and body["think"] is False
        assert body["logprobs"] is True and body["top_logprobs"] == 20 and body["keep_alive"] == "30m"
        assert body["options"] == {"temperature": 0, "num_predict": 1}
        assert [m["role"] for m in body["messages"]] == ["user"]                 # the model's own system prompt is left alone
        assert body["messages"][0]["content"] == D.compile_request(STATE, {"q": {"type": "noul", "instructions": "ok?"}})["prompts"]["q"]
        assert res["ok"] is True and res["degraded"] is False and res["error"] is None and res["model"] == "m:1"

    def test_one_request_per_question_each_answered(self, monkeypatch):
        res, server = self._go(monkeypatch, lambda b: _resp({"A": -0.1, "B": -3.0, "C": -4.0}), QUESTIONS)
        assert len(server.calls) == 3
        assert set(res["answers"]) == {"department", "refund", "urgency"}
        assert res["answers"]["department"]["choice"] == "billing"
        assert res["answers"]["urgency"]["score"] < 0.5

    def test_the_answer_reads_the_letters_in_the_top_logprobs(self, monkeypatch):
        res, _ = self._go(monkeypatch, lambda b: YES)
        assert res["answers"]["q"]["noul"] > 0.95
        res, _ = self._go(monkeypatch, lambda b: NO)
        assert res["answers"]["q"]["noul"] < 0.05

    def test_a_full_top_20_with_one_letter_scores_the_others_at_the_floor(self, monkeypatch):
        res, _ = self._go(monkeypatch, lambda b: _resp({"A": -0.5}, filler=20),
                          {"q": {"type": "choice", "instructions": "q", "criteria": {"a": None, "b": None, "c": None}}})
        probs = res["answers"]["q"]["probabilities"]
        assert probs["a"] < 1.0 and probs["b"] == probs["c"] > 0.0

    def test_a_short_top_list_means_nothing_was_cut_off(self, monkeypatch):
        res, _ = self._go(monkeypatch, lambda b: _resp({"A": -0.5}, filler=1),
                          {"q": {"type": "choice", "instructions": "q", "criteria": {"a": None, "b": None, "c": None}}})
        assert res["answers"]["q"]["probabilities"] == {"a": 1.0, "b": 0.0, "c": 0.0}

    def test_the_card_comes_from_the_pool(self, monkeypatch):
        _pool(monkeypatch, [{"name": "m:1", "roles": ["decide"], "gpu": 1}])
        _, server = self._go(monkeypatch, lambda b: YES)
        assert server.calls[0][1]["options"] == {"temperature": 0, "num_predict": 1, "main_gpu": 1}
        _pool(monkeypatch, [{"name": "m:1", "roles": ["decide"]}])
        _, server = self._go(monkeypatch, lambda b: YES)
        assert "main_gpu" not in server.calls[0][1]["options"]

    def test_the_model_comes_from_the_pool_role_and_the_environment_overrides_it(self, monkeypatch):
        _pool(monkeypatch, [{"name": "pooled:1", "roles": ["decide"]}, {"name": "env:1", "roles": ["gen"]}])
        server = FakeServer(lambda b: YES)
        D.decide(STATE, {"q": {"type": "noul", "instructions": "q"}}, base_url="http://o", post=server)
        assert server.calls[0][1]["model"] == "pooled:1"
        monkeypatch.setenv(D.MODEL_ENV, "env:1")
        D.decide(STATE, {"q": {"type": "noul", "instructions": "q"}}, base_url="http://o", post=server)
        assert server.calls[1][1]["model"] == "env:1"
        D.decide(STATE, {"q": {"type": "noul", "instructions": "q"}}, base_url="http://o", post=server, model="arg:1")
        assert server.calls[2][1]["model"] == "arg:1"

    def test_no_model_configured_is_a_degraded_answer_not_an_exception(self, monkeypatch):
        _pool(monkeypatch, [{"name": "g:1", "roles": ["gen"]}])
        res = D.decide(STATE, {"q": {"type": "noul", "instructions": "q"}}, base_url="http://o", post=FakeServer(lambda b: YES))
        assert res["ok"] is False and res["degraded"] is True and "no decision model" in res["error"] and res["answers"] == {}

    def test_no_server_url_is_a_degraded_answer(self, monkeypatch):
        monkeypatch.setattr("backends.ollama_gen_url", lambda *a, **k: "")
        res = D.decide(STATE, {"q": {"type": "noul", "instructions": "q"}}, model="m:1", post=FakeServer(lambda b: YES))
        assert res["ok"] is False and res["error"] == "no Ollama URL configured"

    def test_a_bad_request_is_a_degraded_answer_naming_the_problem(self, monkeypatch):
        res, server = self._go(monkeypatch, lambda b: YES, {"q": {"type": "other", "instructions": "q"}})
        assert res["ok"] is False and "type must be choice, noul, or score" in res["error"] and server.calls == []

    def test_a_failing_server_degrades_that_question_and_the_rest_are_still_asked(self, monkeypatch):
        def responder(body):
            return RuntimeError("connection reset") if '"urgency"' in body["messages"][0]["content"].split("Requested field")[-1] else YES
        res, server = self._go(monkeypatch, responder, QUESTIONS)
        assert len(server.calls) == 3
        assert res["ok"] is False and res["degraded"] is True
        assert res["answers"]["urgency"] == {"error": "connection reset"}
        assert "noul" in res["answers"]["refund"] or "choice" in res["answers"]["refund"]
        assert res["error"].startswith("urgency: connection reset")

    def test_a_response_with_no_logprobs_or_no_letters_is_a_per_question_error(self, monkeypatch):
        for bad in ({}, {"logprobs": []}, {"logprobs": [{"top_logprobs": []}]}, _resp({"Z": -1.0})):
            res, _ = self._go(monkeypatch, lambda b, bad=bad: bad)
            assert res["ok"] is False and "none of the answer letters" in res["answers"]["q"]["error"]

    def test_yes_no_and_choose_return_the_answer_or_none(self, monkeypatch):
        monkeypatch.setattr(D, "_http_post", FakeServer(lambda b: YES))
        assert D.yes_no(STATE, "ok?", model="m:1", base_url="http://o") > 0.95
        monkeypatch.setattr(D, "_http_post", FakeServer(lambda b: _resp({"A": -0.1, "B": -4.0})))
        got = D.choose(STATE, "which?", {"x": "an x", "y": "a y"}, model="m:1", base_url="http://o")
        assert got["choice"] == "x" and set(got["probabilities"]) == {"x", "y"}
        monkeypatch.setattr(D, "_http_post", FakeServer(lambda b: RuntimeError("down")))
        assert D.yes_no(STATE, "ok?", model="m:1", base_url="http://o") is None
        assert D.choose(STATE, "which?", {"x": None, "y": None}, model="m:1", base_url="http://o") is None


class TestOutcomeLog:
    def _record(self, monkeypatch, responder):
        rows = []
        monkeypatch.setattr(model_pool, "record_outcome", lambda *a, **k: rows.append((a, k)))
        D.decide(STATE, QUESTIONS, model="m:1", base_url="http://o", post=FakeServer(responder))
        return rows

    def test_each_question_leaves_an_outcome_row_under_the_decide_role(self, monkeypatch):
        rows = self._record(monkeypatch, lambda b: _resp({"A": -0.1, "B": -3.0, "C": -4.0}))
        assert len(rows) == 3
        for args, kw in rows:
            assert args[0] == "m:1" and args[1] is True and args[2] >= 0
            assert kw["route_role"] == "decide" and kw["prompt_chars"] > 100

    def test_a_failed_question_is_logged_as_not_ok(self, monkeypatch):
        rows = self._record(monkeypatch, lambda b: RuntimeError("down"))
        assert [a[1] for a, _ in rows] == [False, False, False]


class TestCheck:
    def _letters_for(self, body):
        """Answer every built-in check correctly by reading the expected value back out of the prompt text."""
        user = body["messages"][0]["content"]
        for kind, state, question, criteria, expected in D.CHECKS:
            if question in user and state in json.loads(user.split("\n\nRequested field")[0])["context"]:
                if kind == "noul":
                    return _resp({"B": -0.05, "A": -4.0}) if expected else _resp({"A": -0.05, "B": -4.0})
                keys = list(criteria)
                letter = chr(65 + keys.index(expected))
                return _resp({letter: -0.05})
        raise AssertionError("unknown check prompt")

    def test_every_check_passing_is_ok(self, monkeypatch):
        monkeypatch.setattr(D, "_http_post", lambda url, body, timeout: self._letters_for(body))
        res = D.check(model="m:1", base_url="http://o")
        assert res == {"ok": True, "passed": len(D.CHECKS), "total": len(D.CHECKS), "failures": []}

    def test_a_wrong_answer_and_a_dead_server_are_both_failures(self, monkeypatch):
        monkeypatch.setattr(D, "_http_post", lambda url, body, timeout: _resp({"A": -0.05}))
        res = D.check(model="m:1", base_url="http://o")
        assert res["ok"] is False and 0 < res["passed"] < res["total"] and res["failures"]
        monkeypatch.setattr(D, "_http_post", lambda url, body, timeout: (_ for _ in ()).throw(RuntimeError("down")))
        dead = D.check(model="m:1", base_url="http://o")
        assert dead["passed"] == 0 and len(dead["failures"]) == len(D.CHECKS) and dead["failures"][0]["error"]

    def test_the_cli_exit_code_follows_the_check(self, monkeypatch, capsys):
        monkeypatch.setattr(D, "check", lambda **kw: {"ok": True, "passed": 8, "total": 8, "failures": []})
        assert D._main(["check"]) == 0
        monkeypatch.setattr(D, "check", lambda **kw: {"ok": False, "passed": 7, "total": 8, "failures": [{"x": 1}]})
        assert D._main(["check"]) == 1
        assert D._main(["nonsense"]) == 2


OBS = {"kind": "claude_code_event", "path": "C:/x/s.jsonl", "events": {"user": 3}, "tools": {"Bash": 2},
       "errors": {"claude tool_result error: exit code <n>": 3}, "warnings": {}}


def _decide_ok(category="config_or_environment", novelty="known_pattern", conf=0.8, nconf=0.4):
    return {"ok": True, "answers": {
        "category": {"type": "choice", "choice": category, "confidence": conf,
                     "probabilities": {category: 0.9, "unknown": 0.1}},
        "novelty": {"type": "choice", "choice": novelty, "confidence": nconf, "probabilities": {novelty: 1.0}}}}


class TestTriageClassifier:
    def test_it_returns_the_generating_classifiers_shape_plus_confidence(self):
        got = RT.classify_reflection_observation_decide(OBS["kind"], OBS["path"], decide_fn=lambda s, q: _decide_ok())
        assert got == {"category": "config_or_environment", "novelty": "known_pattern", "confidence": 0.8,
                       "novelty_confidence": 0.4, "probabilities": {"config_or_environment": 0.9, "unknown": 0.1},
                       "degraded": False, "ok": True, "error": None, "source": "decide"}

    def test_the_state_and_the_two_questions_it_sends(self):
        seen = {}

        def fake(state, questions):
            seen.update(state=state, questions=questions)
            return _decide_ok()
        RT.classify_reflection_observation_decide(
            "k", "p" * 500, sampling_mode="tail", events={f"e{i}": i for i in range(9)}, tools={"T": 1},
            errors={f"err{i}": i for i in range(7)}, warnings={"w": 1}, decide_fn=fake)
        st = seen["state"]
        assert st["kind"] == "k" and len(st["path"]) == 400 and st["sampling"] == "tail"
        assert len(st["top_events"]) == 6 and len(st["visible_errors"]) == 4 and st["visible_warnings"] == {"w": 1}
        assert list(seen["questions"]) == ["category", "novelty"]
        assert list(seen["questions"]["category"]["criteria"]) == [
            "real_regression", "flaky_or_nondeterministic", "config_or_environment", "noise_or_benign", "unknown"]
        assert list(seen["questions"]["novelty"]["criteria"]) == ["novel_signal", "known_pattern", "unclear"]

    def test_every_label_the_decision_model_can_pick_is_one_the_hillclimb_labels_know(self):
        assert set(RT._DECIDE_CATEGORY) == set(H.LABELS.values())

    def test_a_degraded_or_raising_or_malformed_decide_is_a_degraded_result(self):
        for fn in (lambda s, q: {"ok": False, "error": "no decision model"}, lambda s, q: None,
                   lambda s, q: {"ok": True, "answers": {}}):
            got = RT.classify_reflection_observation_decide("k", "p", decide_fn=fn)
            assert got["ok"] is False and got["degraded"] is True and got["category"] is None

        def boom(s, q):
            raise RuntimeError("kaput")
        got = RT.classify_reflection_observation_decide("k", "p", decide_fn=boom)
        assert got["degraded"] is True and "kaput" in got["error"]
        got = RT.classify_reflection_observation_decide("k", "p", decide_fn=lambda s, q: {"ok": False, "error": "no decision model"})
        assert got["error"] == "no decision model"


class TestHillclimbDecideHook:
    def _label(self, decide, answers):
        H.capture_observations([{"status": "processed", **OBS}])
        out = []
        it = iter(answers)
        tally = H.label(n=5, input_fn=lambda prompt: next(it), print_fn=out.append, decide=decide)
        return tally, out

    def test_the_decision_models_answer_is_shown_after_yours_and_tallied_separately(self):
        order = []

        def decide(o):
            order.append("decide")
            return ("config_or_environment", 0.83)

        def ask(prompt):
            order.append("ask")
            return "c" if prompt.startswith("[r]") else ""
        H.capture_observations([{"status": "processed", **OBS}])
        out = []
        tally = H.label(n=5, input_fn=ask, print_fn=out.append, decide=decide)
        assert order == ["ask", "ask", "decide"]
        assert tally["decide_compared"] == 1 and tally["decide_agreed"] == 1
        assert any("decision model said config_or_environment (confidence 0.83) (agrees)" in line for line in out)

    def test_a_disagreement_is_reported_and_counted(self):
        tally, out = self._label(lambda o: ("noise_or_benign", 0.5), ["c", ""])
        assert (tally["decide_compared"], tally["decide_agreed"]) == (1, 0)
        assert any("differs" in line for line in out)

    @staticmethod
    def _boom(o):
        raise RuntimeError("down")

    @pytest.mark.parametrize("fn", [lambda o: None, lambda o: (None, 0.0), _boom.__func__],
                             ids=["none", "no category", "raises"])
    def test_a_decision_model_that_cannot_answer_costs_nothing(self, fn):
        tally, out = self._label(fn, ["c", ""])
        assert tally["labelled"] == 1 and tally["decide_compared"] == 0 and tally["decide_agreed"] == 0
        assert not any("decision model said" in line for line in out)

    def test_without_the_hook_the_tally_has_no_decision_keys(self):
        tally, _ = self._label(None, ["c", ""])
        assert tally == {"labelled": 1, "skipped": 0, "rejected": 0, "compared": 0, "agreed": 0}
