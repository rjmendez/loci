"""Hillclimb teacher proposer: train-only cloud calls, allow_cloud gate, infra aborts, spend cap,
ledger privacy, and the query_expand gold-set suite. All offline: the teacher call is a recording fake."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hillclimb as H  # noqa: E402
import hillclimb_suites as S  # noqa: E402
import query_expand as QE  # noqa: E402

SECRET_TEXT = "ZEBRA-CASE-TEXT-7731"


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path / "mem" / "sessions"))
    monkeypatch.setenv("LOCI_TEACHER_GOLD_DIR", str(tmp_path / "gold"))
    H._overlay_cache.clear()
    return tmp_path


class TextToy:
    """Like the hillclimb Toy suite, but each failing trace carries case text and an email."""
    name = "toytext"
    surfaces = {"real": H.Surface("real", "choice", choices=(0, 1), default=0, desc="helps everything")}

    def __init__(self, n=30):
        self._cases = [H.Case(f"c{i:02d}", {"i": i}) for i in range(n)]
        self.train_ids = {c.id for c in H.split(self._cases)[0]}
        self.test_ids = {c.id for c in H.split(self._cases)[1]}
        self.runs = 0

    def cases(self):
        return self._cases

    def run(self, case, overlay):
        self.runs += 1
        i = case.data["i"]
        ok = (i % 10) < 2 or (overlay.get("real") == 1 and (i % 10) < 7)
        trace = "" if ok else f"{case.id} {SECRET_TEXT} mail bob@example.com"
        return H.CaseResult(case.id, 1.0 if ok else 0.0, trace)


class Calls:
    """Recording stand-in for teacher.teacher_call."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, role, prompt, max_tokens, purpose, **kw):
        self.calls.append({"role": role, "prompt": prompt, "max_tokens": max_tokens, "purpose": purpose})
        return self.results.pop(0)


def ok_patch(usd=0.0):
    return {"ok": True, "usd": usd, "reason": "",
            "text": json.dumps({"key": "real", "value": 1, "rationale": "fixes the failing group"})}


def refusal(reason="over_daily_cap"):
    return {"ok": False, "text": "", "usd": 0.0, "reason": reason}


# --------------------------------------------------------------------------- the proposer itself

def test_proposer_sends_role_propose_with_train_failures_only_and_scrubbed():
    suite = TextToy()
    train, test = H.split(suite.cases())
    tr_id, te_id = train[0].id, test[0].id
    calls = Calls(ok_patch())
    p = H.TeacherProposer(True, call_fn=calls)
    ctx = {"surfaces": suite.surfaces, "overlay": {}, "history": [], "train_ids": suite.train_ids,
           "failures": [{"id": tr_id, "score": 0.0, "trace": f"{tr_id} {SECRET_TEXT} mail bob@example.com"},
                        {"id": te_id, "score": 0.0, "trace": f"{te_id} TESTSIDE-LEAK"}]}
    patch = p.propose(ctx)
    assert patch == H.Patch("real", 1, "fixes the failing group")
    assert p.last_abort is None
    assert len(calls.calls) == 1
    sent = calls.calls[0]
    assert (sent["role"], sent["purpose"], sent["max_tokens"]) == ("propose", "hillclimb_propose", 700)
    assert f"case {tr_id} " in sent["prompt"] and SECRET_TEXT in sent["prompt"]
    assert "TESTSIDE-LEAK" not in sent["prompt"] and f"case {te_id} " not in sent["prompt"]
    assert "bob@example.com" not in sent["prompt"]          # sanitizer ran on the case text


def test_proposer_scrubs_credentials_in_case_text():
    calls = Calls(ok_patch())
    p = H.TeacherProposer(True, call_fn=calls)
    ctx = {"surfaces": TextToy.surfaces, "overlay": {}, "history": [],
           "failures": [{"id": "a", "score": 0.0, "trace": "boom password=hunter2xyz AKIAABCDEFGHIJKLMNOP"}]}
    p.propose(ctx)
    prompt = calls.calls[0]["prompt"]
    assert "hunter2xyz" not in prompt and "AKIAABCDEFGHIJKLMNOP" not in prompt
    assert "boom" in prompt


def test_allow_cloud_false_sends_nothing_and_true_sends():
    ctx = {"surfaces": TextToy.surfaces, "overlay": {}, "history": [],
           "failures": [{"id": "a", "score": 0.0, "trace": SECRET_TEXT}]}
    off = Calls(ok_patch())
    p_off = H.TeacherProposer(False, call_fn=off)
    assert p_off.propose(ctx) is None
    assert p_off.last_abort == "cloud_not_allowed"
    assert off.calls == []
    on = Calls(ok_patch())
    assert H.TeacherProposer(True, call_fn=on).propose(ctx) == H.Patch("real", 1, "fixes the failing group")
    assert len(on.calls) == 1


def test_refusal_reason_is_the_abort_reason():
    ctx = {"surfaces": TextToy.surfaces, "overlay": {}, "history": [], "failures": []}
    p = H.TeacherProposer(True, call_fn=Calls(refusal("over_monthly_cap")))
    assert p.propose(ctx) is None and p.last_abort == "over_monthly_cap"


def test_unparseable_reply_is_an_abort_not_a_patch():
    ctx = {"surfaces": TextToy.surfaces, "overlay": {}, "history": [], "failures": []}
    p = H.TeacherProposer(True, call_fn=Calls({"ok": True, "text": "no json here", "usd": 0.0}))
    assert p.propose(ctx) is None and p.last_abort == "bad_proposal"


def test_default_call_goes_through_the_real_teacher_and_a_disabled_guard_refuses():
    ctx = {"surfaces": TextToy.surfaces, "overlay": {}, "history": [], "failures": []}
    p = H.TeacherProposer(True)   # real teacher.teacher_call; teacher is off by default
    assert p.propose(ctx) is None
    assert p.last_abort == "disabled"


# ------------------------------------------------------------------------------------ in climb

def test_refused_rounds_are_aborted_counted_and_never_scored(_state):
    suite = TextToy()
    p = H.TeacherProposer(True, call_fn=Calls(refusal(), refusal(), refusal()))
    res = H.climb(suite, p, {}, H.ClimbConfig(rounds=6, patience=3))
    assert [r["verdict"] for r in res["rounds"]] == ["aborted_infra"] * 3
    assert [r["reason"] for r in res["rounds"]] == ["over_daily_cap"] * 3
    assert res["stop"] == "infra_aborted" and res["aborted_rounds"] == 3
    assert res["overlay"] == {}
    # runs: baseline train + test + diagnose re-score of 10 cases; no candidate was ever scored
    n_base = len(suite.cases()) + 10
    assert suite.runs == n_base


def test_good_proposal_is_scored_and_accepted_with_zero_aborts(_state):
    suite = TextToy()
    p = H.TeacherProposer(True, call_fn=Calls(ok_patch(usd=0.01)))
    res = H.climb(suite, p, {}, H.ClimbConfig(rounds=1, patience=3))
    assert [r["verdict"] for r in res["rounds"]] == ["accept"]
    assert res["aborted_rounds"] == 0 and res["overlay"] == {"real": 1}
    assert res["teacher_usd"] == 0.01
    assert suite.runs > len(suite.cases()) + 10


def test_abort_does_not_consume_patience_for_scored_rounds(_state):
    suite = TextToy()
    p = H.TeacherProposer(True, call_fn=Calls(refusal(), refusal(), ok_patch()))
    res = H.climb(suite, p, {}, H.ClimbConfig(rounds=3, patience=3))
    assert [r["verdict"] for r in res["rounds"]] == ["aborted_infra", "aborted_infra", "accept"]
    assert res["overlay"] == {"real": 1}


def test_ledger_holds_no_case_text_but_the_prompt_did(_state):
    suite = TextToy()
    calls = Calls(ok_patch())
    res = H.climb(suite, H.TeacherProposer(True, call_fn=calls), {}, H.ClimbConfig(rounds=1, patience=3))
    assert res["rounds"][0]["verdict"] == "accept"
    assert SECRET_TEXT in calls.calls[0]["prompt"]            # the text did reach the (fake) cloud
    ledger = (H.suite_dir("toytext") / "runs.jsonl").read_text(encoding="utf-8")
    assert SECRET_TEXT not in ledger and "bob@example.com" not in ledger
    rows = [json.loads(line) for line in ledger.splitlines()]
    rnd = next(r for r in rows if r["event"] == "round")
    assert (rnd["key"], rnd["value"], rnd["rationale"]) == ("real", 1, "fixes the failing group")


def test_failures_in_ctx_are_train_cases_only(_state):
    suite = TextToy()
    seen = []

    class Spy:
        last_abort = None

        def propose(self, ctx):
            seen.append(ctx)
            return None

    H.climb(suite, Spy(), {}, H.ClimbConfig(rounds=1, patience=3))
    assert seen[0]["train_ids"] == suite.train_ids
    assert {f["id"] for f in seen[0]["failures"]} <= suite.train_ids
    assert not ({f["id"] for f in seen[0]["failures"]} & suite.test_ids)


# ---------------------------------------------------------------------------------- spend cap

def test_max_usd_stops_the_run_before_the_next_call():
    ctx = {"surfaces": TextToy.surfaces, "overlay": {}, "history": [], "failures": []}
    calls = Calls(ok_patch(usd=0.6), ok_patch(usd=0.6))
    p = H.TeacherProposer(True, max_usd=1.0, call_fn=calls)
    assert p.propose(ctx) is not None          # 0.6 spent; next call could reach 1.2
    assert p.propose(ctx) is None and p.last_abort == "budget_exhausted"
    assert len(calls.calls) == 1 and p.spent_usd == 0.6


def test_within_budget_keeps_calling():
    ctx = {"surfaces": TextToy.surfaces, "overlay": {}, "history": [], "failures": []}
    calls = Calls(ok_patch(usd=0.3), ok_patch(usd=0.3))
    p = H.TeacherProposer(True, max_usd=1.0, call_fn=calls)
    assert p.propose(ctx) is not None and p.propose(ctx) is not None
    assert len(calls.calls) == 2 and p.spent_usd == 0.6


def test_climb_stops_with_budget_and_records_the_round(_state):
    suite = TextToy()
    p = H.TeacherProposer(True, max_usd=0.5, call_fn=Calls(
        {"ok": True, "usd": 0.6, "text": json.dumps({"key": "real", "value": 0, "rationale": "noop"})}))
    res = H.climb(suite, p, {}, H.ClimbConfig(rounds=5, patience=9))
    assert [r["verdict"] for r in res["rounds"]] == ["invalid", "aborted_infra"]
    assert res["rounds"][1]["reason"] == "budget_exhausted"
    assert res["stop"] == "budget" and res["teacher_usd"] == 0.6


# ---------------------------------------------------------------------------- proposer selection

def _write_cfg(name, cfg):
    d = H.suite_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")


def test_default_proposer_is_local_and_cloud_off():
    assert H.suite_config("nosuite") == {"allow_cloud": False, "proposer": "local"}
    prop, err = H.make_proposer("nosuite")
    assert isinstance(prop, H.LLMProposer) and err is None


def test_teacher_proposer_refused_without_allow_cloud():
    prop, err = H.make_proposer("nosuite", "teacher")
    assert prop is None and "allow_cloud" in err
    _write_cfg("nosuite", {"proposer": "teacher"})
    prop, err = H.make_proposer("nosuite")
    assert prop is None and "allow_cloud" in err


def test_suite_config_selects_teacher_and_cli_overrides_it():
    _write_cfg("cfgsuite", {"proposer": "teacher", "allow_cloud": True})
    prop, err = H.make_proposer("cfgsuite", None, 2.5)
    assert isinstance(prop, H.TeacherProposer) and prop.allow_cloud is True and prop.max_usd == 2.5 and err is None
    prop, err = H.make_proposer("cfgsuite", "local")
    assert isinstance(prop, H.LLMProposer)


def test_allow_cloud_must_be_literal_true():
    _write_cfg("strsuite", {"proposer": "teacher", "allow_cloud": "yes"})
    assert H.suite_config("strsuite")["allow_cloud"] is False


def test_cli_run_teacher_without_allow_cloud_exits_2(capsys, monkeypatch):
    monkeypatch.setattr(H, "load_suite", lambda spec: TextToy())
    assert H._main(["run", "--suite", "x", "--proposer", "teacher"]) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False and "allow_cloud" in out["why"]


# ------------------------------------------------------------------------- query_expand suite

def _gold(tmp_path, rows):
    d = tmp_path / "gold"
    d.mkdir(parents=True, exist_ok=True)
    (d / "query_expand.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_no_gold_set_reports_status_and_climb_stops(_state):
    suite = S.QueryExpandSuite()
    assert suite.unavailable() == "no gold set yet" and suite.cases() == []
    spy = Calls()
    res = H.climb(suite, H.TeacherProposer(True, call_fn=spy), {}, H.ClimbConfig())
    assert res["stop"] == "suite_unavailable" and res["status"] == "no gold set yet"
    assert spy.calls == []


def test_gold_set_scores_term_coverage_exactly(_state):
    _gold(_state, [{"id": "g1", "query": "disk full", "expect": ["storage", "vhdx"]}])

    def gen(prompt, **kw):
        return {"ok": True, "text": json.dumps({"queries": ["disk full storage"], "keywords": ["ntfs"]})}

    suite = S.QueryExpandSuite(gen_fn=gen)
    assert suite.unavailable() is None
    r = suite.run(suite.cases()[0], {})
    assert r.score == 0.5 and not r.infra_error and "['vhdx']" in r.trace

    def gen_full(prompt, **kw):
        return {"ok": True, "text": json.dumps({"queries": ["disk full storage"], "keywords": ["VHDX"]})}

    r2 = S.QueryExpandSuite(gen_fn=gen_full).run(suite.cases()[0], {})
    assert r2.score == 1.0 and r2.trace == ""


def test_gold_suite_model_down_is_infra_not_a_fail(_state):
    _gold(_state, [{"id": "g1", "query": "disk full", "expect": ["storage"]}])
    suite = S.QueryExpandSuite(gen_fn=lambda prompt, **kw: {"ok": False, "text": ""})
    r = suite.run(suite.cases()[0], {})
    assert r.infra_error is True


def test_guidance_surface_reaches_the_expand_prompt(_state):
    _gold(_state, [{"id": "g1", "query": "disk full", "expect": ["storage"]}])
    prompts = []

    def gen(prompt, **kw):
        prompts.append(prompt)
        return {"ok": True, "text": json.dumps({"queries": ["a"], "keywords": ["k"]})}

    suite = S.QueryExpandSuite(gen_fn=gen)
    suite.run(suite.cases()[0], {})
    suite.run(suite.cases()[0], {"guidance": "prefer storage vocabulary"})
    assert "Extra guidance" not in prompts[0]
    assert prompts[1].endswith("Extra guidance:\nprefer storage vocabulary\n")
    assert prompts[1].startswith(prompts[0])


def test_expand_without_guidance_prompt_is_unchanged():
    seen = []

    def gen(prompt, **kw):
        seen.append(prompt)
        return {"ok": True, "text": '{"queries": [], "keywords": []}'}

    QE.expand("hello", gen_fn=gen)
    assert seen[0] == QE._PROMPT_TMPL.format(query="hello", n_queries=3, n_keywords=6)


def test_gold_rows_without_expect_or_bad_json_are_skipped(_state):
    d = _state / "gold"
    d.mkdir(parents=True)
    (d / "query_expand.jsonl").write_text(
        '{"id": "a", "query": "q", "expect": ["x"]}\nnot json\n{"id": "b", "query": "q", "expect": []}\n'
        '{"id": "a", "query": "dup", "expect": ["y"]}\n', encoding="utf-8")
    assert [c.id for c in S.QueryExpandSuite().cases()] == ["a"]


def test_builtin_registers_query_expand_and_keeps_triage():
    assert set(S.BUILTIN) == {"triage", "query_expand"}
    assert S.BUILTIN["query_expand"]().name == "query_expand"
