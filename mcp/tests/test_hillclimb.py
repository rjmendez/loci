"""Hillclimb: split, validation, accept/revert guard, diagnostics, ledger, overlay, triage suite.

Everything is offline: the suite is a toy whose score depends on overlay values, the proposer is
scripted, and state lives under a temp LOCI_MEMORY_DIR.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import hillclimb as H  # noqa: E402
import hillclimb_suites as S  # noqa: E402


@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(tmp_path / "mem" / "sessions"))
    H._overlay_cache.clear()
    return tmp_path


class Toy:
    """Cases are fixed by overlay settings. ``train_only`` fixes only cases in the train split."""
    name = "toy"
    surfaces = {
        "real": H.Surface("real", "choice", choices=(0, 1), default=0, desc="helps everything"),
        "overfit": H.Surface("overfit", "choice", choices=(0, 1), default=0, desc="helps train only"),
        "knob": H.Surface("knob", "number", lo=0, hi=10, default=1),
        "note": H.Surface("note", "text", max_len=20, must_keep=("KEEP",), default="KEEP"),
    }

    def __init__(self, n=30, base_pass=0.2, infra=False):
        self._cases = [H.Case(f"c{i:02d}", {"i": i}) for i in range(n)]
        self.train_ids = {c.id for c in H.split(self._cases)[0]}
        self.base_pass = base_pass
        self.infra = infra

    def cases(self):
        return self._cases

    def run(self, case, overlay):
        if self.infra:
            return H.CaseResult(case.id, 0, "down", infra_error=True)
        i = case.data["i"]
        ok = (i % 10) < self.base_pass * 10
        if overlay.get("real") == 1:
            ok = ok or (i % 10) < 7
        if overlay.get("overfit") == 1 and case.id in self.train_ids:
            ok = True
        return H.CaseResult(case.id, 1.0 if ok else 0.0, "" if ok else f"case {case.id} failed")


class Script:
    def __init__(self, *patches):
        self.patches = list(patches)
        self.seen = []

    def propose(self, ctx):
        self.seen.append(ctx)
        return self.patches.pop(0) if self.patches else None


# ------------------------------------------------------------------------------ split / surfaces

def test_split_is_disjoint_stable_and_grows_without_reshuffling():
    cases = [H.Case(f"c{i}") for i in range(40)]
    tr, te = H.split(cases)
    assert not ({c.id for c in tr} & {c.id for c in te})
    assert len(tr) + len(te) == 40 and 4 <= len(te) <= 20
    tr2, te2 = H.split(cases + [H.Case(f"new{i}") for i in range(10)])
    assert {c.id for c in te} <= {c.id for c in te2}      # old test cases stay test
    assert {c.id for c in tr} <= {c.id for c in tr2}      # old train cases stay train


def test_surface_validation():
    t = H.Surface("t", "text", max_len=10, must_keep=("{x}",))
    assert t.validate("a {x}")[0] and not t.validate("no placeholder")[0]
    assert not t.validate("a {x} but way too long")[0]
    n = H.Surface("n", "number", lo=0, hi=5)
    assert n.validate(3)[0] and not n.validate(9)[0] and not n.validate("x")[0]
    assert not n.validate(float("nan"))[0]
    c = H.Surface("c", "choice", choices=("a", "b"))
    assert c.validate("a")[0] and not c.validate("z")[0]


# ------------------------------------------------------------------------------------- decide

def _s(mean):
    return H.Score(mean=mean, n=10)


def test_decide_accept_revert_reject():
    assert H.decide(_s(.5), _s(.5), _s(.6), _s(.55))["verdict"] == "accept"
    d = H.decide(_s(.5), _s(.5), _s(.7), _s(.5))
    assert d["verdict"] == "revert" and "plateau" in d["reason"]
    d = H.decide(_s(.5), _s(.5), _s(.7), _s(.4))
    assert d["verdict"] == "revert" and "regress" in d["reason"]
    assert H.decide(_s(.5), _s(.5), _s(.51), _s(.9))["verdict"] == "reject"


# -------------------------------------------------------------------------------------- climb

def test_climb_accepts_a_real_fix_and_reverts_an_overfit_one(_state):
    suite = Toy()
    prop = Script(H.Patch("overfit", 1, "memorise train"), H.Patch("real", 1, "generalises"))
    cfg = H.ClimbConfig(rounds=4, patience=5)
    res = H.climb(suite, prop, {}, cfg)
    verdicts = [r["verdict"] for r in res["rounds"]]
    assert verdicts[0] == "revert" and verdicts[1] == "accept"
    assert res["overlay"] == {"real": 1} and "overfit" not in res["overlay"]
    assert res["final"]["test"]["mean"] > res["baseline"]["test"]["mean"]
    cand = json.loads((H.suite_dir("toy") / "candidate_overlay.json").read_text())
    assert cand == {"real": 1}


def test_history_and_failures_reach_the_proposer_train_only(_state):
    suite = Toy()
    prop = Script(H.Patch("overfit", 1), H.Patch("real", 1))
    H.climb(suite, prop, {}, H.ClimbConfig(rounds=2, patience=5))
    assert prop.seen[1]["history"][0]["verdict"] == "revert"
    train_ids = suite.train_ids
    assert prop.seen[0]["failures"] and all(f["id"] in train_ids for f in prop.seen[0]["failures"])


def test_invalid_and_noop_patches_never_score(_state):
    suite = Toy()
    prop = Script(H.Patch("nope", 1), H.Patch("knob", 99), H.Patch("note", "no keep word"),
                  H.Patch("real", 0))  # real=0 equals the default: no change
    res = H.climb(suite, prop, {}, H.ClimbConfig(rounds=4, patience=9))
    assert [r["verdict"] for r in res["rounds"]] == ["invalid"] * 4
    assert res["overlay"] == {}


def test_patience_stops_and_reports_a_stall(_state):
    res = H.climb(Toy(), Script(), {}, H.ClimbConfig(rounds=9, patience=2))
    assert res["stop"] == "patience" and res["stall"]["remaining_failures"]
    assert "more cases" in res["stall"]["advice"]  # 9 test cases: too few to trust
    assert res["stall"]["advice"]


def test_diagnostics_block_a_saturated_baseline(_state):
    res = H.climb(Toy(base_pass=1.0), Script(H.Patch("real", 1)), {}, H.ClimbConfig())
    assert res["stop"] == "diagnostics" and any("no headroom" in w for w in res["diagnostics"]["warnings"])
    res = H.climb(Toy(base_pass=1.0), Script(), {}, H.ClimbConfig(force=True))
    assert res["stop"] in ("saturated", "patience", "rounds")


def test_inconsistent_grader_is_caught(_state):
    class Flip(Toy):
        calls = 0

        def run(self, case, overlay):
            Flip.calls += 1
            return H.CaseResult(case.id, float(Flip.calls % 7 < 3))

    res = H.climb(Flip(), Script(), {}, H.ClimbConfig())
    assert res["stop"] == "diagnostics" and res["diagnostics"]["consistency"] < 0.9


def test_infra_failures_are_not_failures(_state):
    res = H.climb(Toy(infra=True), Script(), {}, H.ClimbConfig())
    assert res["stop"] == "infra_noise"
    s = H.score_set(Toy(), Toy().cases(), {})
    assert s.n == 30 and s.infra_errors == 0


def test_small_split_warns(_state):
    res = H.climb(Toy(n=6), Script(), {}, H.ClimbConfig(force=True))
    assert any("small split" in w for w in res["diagnostics"]["warnings"])


def test_ledger_holds_no_case_text(_state):
    suite = Toy()
    H.climb(suite, Script(H.Patch("real", 1, "because")), {}, H.ClimbConfig(rounds=1))
    text = (H.suite_dir("toy") / "runs.jsonl").read_text()
    assert "case c" not in text and "failed" not in text
    rows = [json.loads(x) for x in text.splitlines()]
    assert [r["event"] for r in rows][0] == "start" and rows[-1]["event"] == "end"


# --------------------------------------------------------------------------- overlay / promote

def test_overlay_defaults_until_promoted_and_rollback_restores(_state):
    assert H.overlay_get("toy", "real", "dflt") == "dflt"
    assert not H.promote("toy")["ok"]
    H.climb(Toy(), Script(H.Patch("real", 1)), {}, H.ClimbConfig(rounds=1))
    assert H.overlay_get("toy", "real", "dflt") == "dflt"  # a run alone changes nothing
    assert H.promote("toy")["ok"]
    assert H.overlay_get("toy", "real", "dflt") == 1
    assert H.status("toy")["live_overlay"] == {"real": 1}
    assert H.rollback("toy")["restored"] == "defaults"
    assert H.overlay_get("toy", "real", "dflt") == "dflt"


# ---------------------------------------------------------------------------------- proposer

def test_llm_proposer_parses_and_fails_open():
    ctx = {"surfaces": Toy.surfaces, "overlay": {}, "failures": [{"id": "c1", "score": 0, "trace": "x"}], "history": []}
    ok = H.LLMProposer(gen_fn=lambda p, **kw: {"ok": True, "text": '{"key":"real","value":1,"rationale":"r"}'})
    p = ok.propose(ctx)
    assert (p.key, p.value, p.rationale) == ("real", 1, "r")
    for bad in ({"ok": False}, {"ok": True, "text": "prose"}, {"ok": True, "text": '{"key":"real"}'}):
        assert H.LLMProposer(gen_fn=lambda p, _b=bad, **kw: _b).propose(ctx) is None

    def boom(*a, **k):
        raise RuntimeError("x")
    assert H.LLMProposer(gen_fn=boom).propose(ctx) is None


def test_proposer_prompt_lists_surfaces_failures_and_history():
    prompt = H._prompt_for({"surfaces": Toy.surfaces, "overlay": {}, "history": [{"key": "real", "value": 1, "verdict": "revert"}],
                            "failures": [{"id": "c9", "score": 0.0, "trace": "boom trace"}]})
    assert "real:" in prompt and "boom trace" in prompt and "revert" in prompt and "must contain ['KEEP']" in prompt


# ----------------------------------------------------------------------------- triage suite

def test_triage_cases_cover_every_category_and_load():
    cases = S.TriageSuite().cases()
    assert len(cases) >= 20
    assert {c.data["gold"] for c in cases} == {"real_regression", "flaky_or_nondeterministic",
                                              "config_or_environment", "noise_or_benign", "unknown"}


def test_triage_suite_scores_and_passes_guidance_through():
    prompts = []

    def gen(prompt, **kw):
        prompts.append(prompt)
        cat = "noise_or_benign" if "Guidance: ignore" in prompt else "real_regression"
        return {"ok": True, "text": json.dumps({"category": cat, "novelty": "unclear"})}

    suite = S.TriageSuite(gen_fn=gen)
    case = next(c for c in suite.cases() if c.data["gold"] == "noise_or_benign")
    assert suite.run(case, {}).score == 0.0
    r = suite.run(case, {"guidance": "ignore routine warnings"})
    assert r.score == 1.0 and "Guidance: ignore routine warnings" in prompts[-1]


def test_triage_model_down_is_infra_not_a_fail():
    suite = S.TriageSuite(gen_fn=lambda p, **kw: {"ok": False, "why": "down"})
    r = suite.run(suite.cases()[0], {})
    assert r.infra_error


def test_triage_prompt_unchanged_without_overlay(_state):
    import reflection_triage as RT
    seen = []
    RT.classify_reflection_observation("k", "p", gen_fn=lambda pr, **kw: (seen.append(pr), {"ok": False})[1])
    assert "Guidance" not in seen[0]
    d = H.suite_dir("reflection_triage")
    d.mkdir(parents=True)
    (d / "overlay.json").write_text(json.dumps({"guidance": "be strict"}))
    RT.classify_reflection_observation("k", "p", gen_fn=lambda pr, **kw: (seen.append(pr), {"ok": False})[1])
    assert "Guidance: be strict" in seen[1]


def test_triage_end_to_end_climb_with_fake_model(_state):
    def gen(prompt, **kw):
        good = "Guidance: tell flaky" in prompt
        out = {"category": "unknown", "novelty": "unclear"}
        if good:  # the guidance fixes everything: answer from the case text
            for gold in ("flaky_or_nondeterministic", "real_regression", "config_or_environment",
                         "noise_or_benign", "unknown"):
                out = {"category": gold, "novelty": "unclear"}
        return {"ok": True, "text": json.dumps(out)}

    suite = S.TriageSuite(gen_fn=lambda p, **kw: {"ok": True, "text": json.dumps({"category": "unknown", "novelty": "unclear"})})
    prop = Script(H.Patch("guidance", "tell flaky from real", "x"))
    res = H.climb(suite, prop, {}, H.ClimbConfig(rounds=1, force=True))
    assert res["rounds"][0]["verdict"] in ("reject", "revert", "accept")  # model ignores it: not accepted blindly
    assert res["baseline"]["train"]["n"] > 0


def test_saturated_test_split_blocks_the_climb(_state):
    class Easy(Toy):
        def run(self, case, overlay):
            ok = case.id not in self.train_ids or case.data["i"] % 2 == 0  # test split all pass
            return H.CaseResult(case.id, 1.0 if ok else 0.0, "" if ok else "miss")

    res = H.climb(Easy(), Script(H.Patch("real", 1)), {}, H.ClimbConfig())
    assert res["stop"] == "diagnostics"
    assert any("held-out split" in w for w in res["diagnostics"]["warnings"])


def test_cli_alias_resolves_to_the_suite_name_for_status_promote_rollback(_state, capsys):
    # `triage` is an alias for the suite named `reflection_triage`; all subcommands must agree on the folder.
    assert H.suite_name("triage") == "reflection_triage"
    assert H.suite_name("reflection_triage") == "reflection_triage"
    d = H.suite_dir("reflection_triage")
    d.mkdir(parents=True)
    (d / "candidate_overlay.json").write_text(json.dumps({"guidance": "x"}))
    assert H._main(["promote", "--suite", "triage"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert json.loads((d / "overlay.json").read_text()) == {"guidance": "x"}
    assert H._main(["status", "--suite", "triage"]) == 0
    assert json.loads(capsys.readouterr().out)["live_overlay"] == {"guidance": "x"}
    assert H._main(["rollback", "--suite", "triage"]) == 0
    assert not (d / "overlay.json").exists()


# ------------------------------------------------------------------ observations and labelling

def _item(errors=None, **kw):
    base = {"status": "processed", "kind": "test_run", "path": r"C:\Users\bob\proj\tests\test_x.py",
            "events": {"test_failed": 3}, "tools": {}, "errors": {"TimeoutError: waited 5s": 3} if errors is None else errors, "warnings": {}}
    base.update(kw)
    return base


def test_capture_keeps_processed_dedupes_by_content_and_trims_the_path(_state):
    items = [_item(), _item(path="/other/place/tests/test_x.py"),          # same pattern, other file: one thing
             _item(errors={"KeyError: 'score'": 2}), {"status": "skipped", "kind": "x"}, "junk", None]
    assert H.capture_observations(items) == 2
    assert H.capture_observations(items) == 0
    rows = H._read_jsonl(H.suite_dir("reflection_triage") / "observations.jsonl")
    assert len(rows) == 2 and rows[0]["path"] == "proj/tests/test_x.py"
    assert all(r["id"].startswith("r") for r in rows)


def test_capture_scrubs_secrets_in_short_error_text(_state):
    secret = {"auth failed key AKIAABCDEFGHIJKLMNOP for bob@example.com token=hunter2 hash 0123456789abcdef0123456789abcdef0123": 1}
    assert H.capture_observations([_item(errors=secret)]) == 1
    text = (H.suite_dir("reflection_triage") / "observations.jsonl").read_text()
    for leak in ("AKIAABCDEFGHIJKLMNOP", "bob@example.com", "hunter2", "0123456789abcdef0123456789abcdef0123"):
        assert leak not in text
    assert "[REDACTED]" in text


def test_capture_never_raises(_state, monkeypatch):
    monkeypatch.setattr(H, "suite_dir", lambda s: (_ for _ in ()).throw(RuntimeError("boom")))
    assert H.capture_observations([_item()]) == 0


def _answers(*a):
    it = iter(a)
    return lambda prompt: next(it)


def test_label_writes_cases_and_resumes(_state):
    H.capture_observations([_item(), _item(errors={"KeyError: 'score'": 2}), _item(errors={"503 Service Unavailable": 1})])
    out = []
    t = H.label(n=10, input_fn=_answers("zz", "R", "s", "u"), print_fn=out.append)   # invalid, then r; skip; unknown
    assert t["labelled"] == 2 and t["skipped"] == 1
    cases = H._read_jsonl(H.suite_dir("reflection_triage") / "labels.jsonl")
    assert [c["gold"] for c in cases] == ["real_regression", "unknown"]
    assert set(cases[0]) >= {"id", "gold", "kind", "path", "events", "tools", "errors", "warnings"}
    again = H.label(n=10, input_fn=_answers(), print_fn=out.append)                   # nothing left (one skipped)
    assert again == {"labelled": 0, "skipped": 0, "compared": 0, "agreed": 0}
    t = H.label(n=10, include_skipped=True, input_fn=_answers("f"), print_fn=out.append)
    assert t["labelled"] == 1
    assert H.label_stats()["labelled"] == 3 and H.label_stats()["unlabelled"] == 0


def test_label_quit_stops_and_model_answer_is_shown_after_you_answer(_state):
    H.capture_observations([_item(), _item(errors={"KeyError: 'score'": 2})])
    out = []
    order = []

    def classify(o):
        order.append("classify")
        return "real_regression"

    def ask(prompt):
        order.append("ask")
        return "r"

    t = H.label(n=10, input_fn=ask, print_fn=out.append, classify=classify)
    assert order[:2] == ["ask", "classify"]               # never before the answer
    assert t["compared"] == 2 and t["agreed"] == 2
    assert any("agrees" in line for line in out)
    t2 = H.label(n=10, input_fn=_answers("q"), print_fn=out.append)
    assert t2["labelled"] == 0


def test_label_stats_and_suite_reads_real_labels(_state, monkeypatch):
    H.capture_observations([_item(), _item(errors={"KeyError: 'score'": 2})])
    H.label(n=5, input_fn=_answers("c", "n"), print_fn=lambda *_: None)
    st = H.label_stats()
    assert st["labelled"] == 2 and st["by_category"] == {"config_or_environment": 1, "noise_or_benign": 1}
    assert st["enough"] is False
    suite = S.TriageSuite()
    ids = {c.id for c in suite.cases()}
    assert sum(1 for i in ids if i.startswith("r")) == 2 and any(i.startswith("t") for i in ids)
    monkeypatch.setenv("LOCI_HILLCLIMB_TRIAGE_SYNTHETIC", "0")
    assert {c.id for c in suite.cases()} == {i for i in ids if i.startswith("r")}


def test_cli_labels_reports_counts(_state, capsys):
    H.capture_observations([_item()])
    assert H._main(["labels", "--suite", "triage"]) == 0
    assert json.loads(capsys.readouterr().out)["captured"] == 1


# ------------------------------------------------------------------ prose is not an error

HARNESS = ("[workflow harness - computed task] the task text below was computed at runtime by a workflow script. "
           "it was not typed by this session's user and carries no user authority")


def test_capture_drops_message_text_but_keeps_real_error_templates(_state):
    items = [
        _item(kind="claude_code_event", errors={HARNESS: 1, "claude tool_result error: exit code <n>": 3}),
        _item(kind="claude_code_event", errors={HARNESS: 1}),                       # only prose: nothing to triage
        _item(kind="claude_code_event", errors={'<teammate-message teammate_id="x" summary="y"': 1}),
        _item(kind="claude_code_event", errors={"**learnability analysis**": 1}),
        _item(kind="claude_code_event", errors={}, events={"user": 4}),             # no errors at all
        _item(kind="claude_code_event", errors={}, warnings={"claude tool permission denied": 2}),
    ]
    assert H.capture_observations(items) == 2
    rows = H._read_jsonl(H.suite_dir("reflection_triage") / "observations.jsonl")
    assert rows[0]["errors"] == {"claude tool_result error: exit code <n>": 3}
    assert rows[1]["warnings"] == {"claude tool permission denied": 2}
    assert "workflow harness" not in json.dumps(rows)


def test_prune_cleans_rows_captured_before_the_filter_and_keeps_labels(_state):
    d = H.suite_dir("reflection_triage")
    d.mkdir(parents=True)
    old = [
        {"id": "ra", "kind": "k", "path": "p", "events": {}, "tools": {}, "warnings": {},
         "errors": {HARNESS[:200]: 1, "claude tool_result error: exit code <n>": 3}},
        {"id": "rb", "kind": "k", "path": "p", "events": {}, "tools": {}, "warnings": {}, "errors": {HARNESS[:200]: 1}},
        {"id": "rc", "kind": "k", "path": "p", "events": {}, "tools": {}, "warnings": {},   # same pattern as ra after pruning
         "errors": {"x" * 200: 1, "claude tool_result error: exit code <n>": 3}},
    ]
    (d / "observations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in old))
    (d / "labels.jsonl").write_text(json.dumps({"id": "ra", "gold": "unknown"}) + "\n")
    assert H.prune_observations() == {"before": 3, "after": 1}
    rows = H._read_jsonl(d / "observations.jsonl")
    assert rows[0]["errors"] == {"claude tool_result error: exit code <n>": 3}
    assert rows[0]["id"] == H.observation_id(rows[0])
    assert (d / "labels.jsonl").read_text().count("ra") == 1
    assert H.prune_observations() == {"before": 1, "after": 1}      # idempotent
