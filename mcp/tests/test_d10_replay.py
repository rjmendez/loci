"""The offline D10 replay scores pairs with the live gates' own code, never writes into the
store, is polite to a busy host, and hands the labeller a sheet with no gate information.
The analysis script's estimators are checked on labels with a known answer.
"""
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import d10_gate
import grounding_gate
import memcheck.llm as llm
import server

REPO = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # dataclasses look their module up while it loads
    spec.loader.exec_module(module)
    return module


R = _load("d10_replay")
A = _load("d10_replay_analyze")
DIM = 768


# --------------------------------------------------------------------------- fixtures


def _vec(text):
    """Deterministic embedding: a shared topic direction per leading word, plus noise."""
    topic = text.split()[0].lower() if text.split() else ""
    base = np.random.default_rng(int(hashlib.sha256(topic.encode()).hexdigest()[:8], 16)).normal(size=DIM)
    noise = np.random.default_rng(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)).normal(size=DIM)
    return (base + 0.9 * noise).tolist()


class FakeEmbed:
    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [_vec(t) for t in texts]


def _write_inv(mem, name, *, n, manifest=None, retract=0, created="2026-07-01T00:00:00+00:00", topic="alpha"):
    d = mem / name
    d.mkdir(parents=True)
    tag = hashlib.sha256(name.encode()).hexdigest()[:6]      # unique text that does not name the directory
    m = {"id": name, "title": f"{topic} title {tag}", "hypothesis": f"{topic} hypothesis {tag}",
         "next_step": None, "created_at": created}
    if manifest is not None:
        m = manifest
    (d / "manifest.json").write_text(json.dumps(m))
    rows = []
    for i in range(n):
        t = f"{topic if i % 2 == 0 else 'omega'} finding {i} lot {tag} measured 3 {i}"
        rows.append({"id": f"{name}-f{i}", "type": "observed", "text": t})
    rows.append({"id": f"{name}-blank", "type": "observed", "text": "   "})
    (d / "findings.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if retract:
        (d / "retractions.jsonl").write_text("".join(
            json.dumps({"finding_id": f"{name}-f{i}", "active": True, "ts": "2026-07-02"}) + "\n" for i in range(retract)))
    return d


@pytest.fixture
def store(tmp_path):
    mem = tmp_path / "memory-sessions"
    mem.mkdir()
    for i, (n, created) in enumerate([(9, "2026-06-01"), (12, "2026-06-15"), (20, "2026-07-01"), (30, "2026-07-20"),
                                      (10, "2026-08-01"), (15, "2026-08-20"), (25, "2026-09-01"), (11, "2026-09-10"),
                                      (40, "2026-09-20")]):
        _write_inv(mem, f"inv-{i}", n=n, created=created + "T00:00:00+00:00", topic=["alpha", "beta", "gamma"][i % 3])
    _write_inv(mem, "_groom", n=10)
    (mem / "no-manifest").mkdir()
    _write_inv(mem, "flybrain-ops", n=10)
    _write_inv(mem, "nightly-smoke-1", n=10)
    _write_inv(mem, "probe-latency", n=10)
    _write_inv(mem, "retest-2026", n=10)
    _write_inv(mem, "recover-lost", n=10)
    _write_inv(mem, "tiny", n=7)
    _write_inv(mem, "huge", n=401)
    _write_inv(mem, "mostly-retracted", n=9, retract=2)
    _write_inv(mem, "no-questions", n=10, manifest={"id": "no-questions", "title": "  ", "hypothesis": "None"})
    (mem / "entities.jsonl").write_text("{}\n")
    return mem


def _snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(root.rglob("*")) if p.is_file()} | {
        str(p.relative_to(root)) + "/": None for p in sorted(root.rglob("*")) if p.is_dir()}


def _free_clock():
    """A fake clock that starts at a local minute 2, second 0 (outside the burst window)."""
    lt = time.localtime()
    start = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, lt.tm_hour, 2, 0, 0, 0, -1))
    state = {"t": float(start), "slept": []}

    def clock():
        return state["t"]

    def sleep(s):
        state["slept"].append(s)
        state["t"] += s

    return state, clock, sleep


def _pol(tmp_path, io="0.0", cpu="0.0", **kw):
    p = tmp_path / "pressure"
    p.mkdir(exist_ok=True)
    (p / "io").write_text(f"some avg10=0.00 avg60={io} avg300=0.00 total=1\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n")
    (p / "cpu").write_text(f"some avg10=0.00 avg60={cpu} avg300=0.00 total=1\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n")
    return R.Politeness(pressure_dir=p, **kw)


def _run(store, out, tmp_path, embed=None, **kw):
    state, clock, sleep = _free_clock()
    embed = embed or FakeEmbed()
    meta = R.run_replay(store, out, n_investigations=kw.pop("n", 80), seed=kw.pop("seed", 1), pol=_pol(tmp_path),
                        embed_fn=embed, model="nomic-embed-text", clock=clock, sleep=sleep, **kw)
    return meta, embed, state


# --------------------------------------------------------------------------- enumeration and questions


def test_enumeration_is_read_only_and_excludes_with_reasons(store):
    before = _snapshot(store)
    eligible, excluded = R.enumerate_investigations(store)
    assert _snapshot(store) == before
    assert sorted(e["investigation_id"] for e in eligible) == [f"inv-{i}" for i in range(9)]
    reasons = {e["investigation_id"]: e["reason"] for e in excluded}
    assert reasons == {
        "_groom": "internal directory",
        "no-manifest": "no readable manifest.json",
        "flybrain-ops": "synthetic / operations log",
        "nightly-smoke-1": "name matches smoke|probe|test|recover-",
        "probe-latency": "name matches smoke|probe|test|recover-",
        "retest-2026": "name matches smoke|probe|test|recover-",
        "recover-lost": "name matches smoke|probe|test|recover-",
        "tiny": "fewer than 8 active findings",
        "huge": "more than 400 active findings",
        "mostly-retracted": "fewer than 8 active findings",
        "no-questions": "no title, hypothesis or next_step text",
    }
    # Blank and retracted findings are not counted, as in investigation_reason.
    assert {e["investigation_id"]: e["n_findings"] for e in eligible}["inv-0"] == 9


def test_questions_are_deterministic_and_record_their_kind():
    m = {"title": "  Why   does X fail? ", "hypothesis": "Why does X fail?", "next_step": "Measure X at 3 V",
         "open_questions": ["ignored"]}
    assert R.questions_for(m) == [{"kind": "title", "text": "Why does X fail?"},
                                  {"kind": "next_step", "text": "Measure X at 3 V"}]
    assert R.questions_for({"title": None, "hypothesis": "None", "next_step": ["a"]}) == []
    assert R.questions_for(m) == R.questions_for(dict(m))


def test_investigation_sample_is_stratified_seeded_and_complete_when_small(store):
    eligible, _ = R.enumerate_investigations(store)
    a, design = R.stratified_sample(eligible, 5, seed=3)
    b, _ = R.stratified_sample(eligible, 5, seed=3)
    assert [x["investigation_id"] for x in a] == [x["investigation_id"] for x in b]
    assert len(a) == 5 and sum(v["sampled"] for v in design.values()) == 5
    assert all(v["sampled"] <= v["population"] for v in design.values())
    allx, design_all = R.stratified_sample(eligible, 80, seed=3)
    assert len(allx) == len(eligible)
    many, d9 = R.stratified_sample(eligible, 7, seed=3)
    # Every non-empty stratum gets at least one when n allows.
    assert len(many) == 7 and all(v["sampled"] >= 1 for v in d9.values() if v["population"])


# --------------------------------------------------------------------------- politeness


def test_burst_window_is_the_first_90_seconds_after_each_fifth_minute():
    lt = time.localtime()

    def at(minute, second):
        return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, lt.tm_hour, minute, second, 0, 0, -1))

    assert R.in_burst_window(at(5, 0)) and R.in_burst_window(at(5, 89)) and R.in_burst_window(at(0, 30))
    assert R.in_burst_window(at(6, 29))
    assert not R.in_burst_window(at(6, 30)) and not R.in_burst_window(at(7, 0)) and not R.in_burst_window(at(4, 59))


def test_pressure_gates_hold_and_missing_files_do_not(tmp_path):
    state, clock, sleep = _free_clock()
    cache = R.VectorCache(tmp_path / "v.jsonl")
    mk = lambda pol: R.PoliteEmbedder(FakeEmbed(), "m", cache, tmp_path / "p.jsonl", pol, clock=clock, sleep=sleep)  # noqa: E731
    assert mk(_pol(tmp_path, io="15.1")).hold_reason() == "io_pressure"
    assert mk(_pol(tmp_path, io="15.0")).hold_reason() is None
    assert mk(_pol(tmp_path, cpu="50.5")).hold_reason() == "cpu_pressure"
    assert mk(R.Politeness(pressure_dir=tmp_path / "nowhere")).hold_reason() is None
    assert R.read_pressure(tmp_path / "pressure" / "io") == 0.0


def test_embedder_waits_out_the_burst_window_and_counts_it(tmp_path):
    state, clock, sleep = _free_clock()
    state["t"] -= 120.0          # local minute 0, second 0: inside the window
    embed = FakeEmbed()
    e = R.PoliteEmbedder(embed, "m", R.VectorCache(tmp_path / "v.jsonl"), tmp_path / "p.jsonl", _pol(tmp_path),
                         clock=clock, sleep=sleep)
    e.run(["alpha one", "alpha two"])
    assert len(embed.calls) == 1
    assert e.stats.hold_s["burst_window"] == pytest.approx(90.0)
    assert state["t"] >= clock() - 1 and not R.in_burst_window(state["t"] - 0.0)


def test_embedder_gives_up_after_continuous_holding_and_checkpoints(tmp_path):
    state, clock, sleep = _free_clock()
    prog = tmp_path / "p.jsonl"
    e = R.PoliteEmbedder(FakeEmbed(), "m", R.VectorCache(tmp_path / "v.jsonl"), prog,
                         _pol(tmp_path, cpu="77.0", max_hold_s=600.0), clock=clock, sleep=sleep)
    with pytest.raises(R.HoldTimeout):
        e.run(["alpha one"])
    events = [json.loads(line)["event"] for line in prog.read_text().splitlines()]
    assert events[-1] == "gave_up_holding"
    # Continuous holding counts across reasons: the burst window at minute 5 does not reset it.
    assert sum(e.stats.hold_s.values()) == pytest.approx(600.0, abs=5.0)
    assert e.stats.hold_s["cpu_pressure"] >= 400 and e.stats.hold_s["burst_window"] > 0


def test_batches_are_small_single_and_paused(tmp_path):
    state, clock, sleep = _free_clock()
    embed = FakeEmbed()
    e = R.PoliteEmbedder(embed, "m", R.VectorCache(tmp_path / "v.jsonl"), tmp_path / "p.jsonl", _pol(tmp_path),
                         clock=clock, sleep=sleep)
    e.run([f"alpha {i}" for i in range(40)] + ["alpha 0"])
    assert [len(c) for c in embed.calls] == [16, 16, 8]
    assert state["slept"].count(2.0) >= 2


def test_consecutive_failures_stop_and_slow_requests_back_off(tmp_path):
    state, clock, sleep = _free_clock()
    e = R.PoliteEmbedder(lambda texts: [], "m", R.VectorCache(tmp_path / "v.jsonl"), tmp_path / "p.jsonl",
                         _pol(tmp_path), clock=clock, sleep=sleep)
    with pytest.raises(R.EmbedFailed):
        e.run(["alpha one"])
    assert e.stats.failures == 5 and e.stats.requests == 5

    def slow(texts):
        state["t"] += 25.0
        return [_vec(t) for t in texts]

    e2 = R.PoliteEmbedder(slow, "m", R.VectorCache(tmp_path / "v2.jsonl"), tmp_path / "p2.jsonl", _pol(tmp_path),
                          clock=clock, sleep=sleep)
    e2.run([f"alpha {i}" for i in range(20)])
    assert e2.stats.slow == 2 and e2.stats.backoff_s == pytest.approx(10.0 + 20.0)


def test_a_stopped_run_resumes_without_redoing_work(tmp_path):
    state, clock, sleep = _free_clock()
    texts = [f"alpha {i}" for i in range(40)]
    calls = {"n": 0}

    def flaky(chunk):
        calls["n"] += 1
        if calls["n"] >= 2:
            return []
        return [_vec(t) for t in chunk]

    e = R.PoliteEmbedder(flaky, "m", R.VectorCache(tmp_path / "v.jsonl"), tmp_path / "p.jsonl", _pol(tmp_path),
                         clock=clock, sleep=sleep)
    with pytest.raises(R.EmbedFailed):
        e.run(texts)
    embed = FakeEmbed()
    e2 = R.PoliteEmbedder(embed, "m", R.VectorCache(tmp_path / "v.jsonl"), tmp_path / "p.jsonl", _pol(tmp_path),
                          clock=clock, sleep=sleep)
    e2.run(texts)
    assert e2.stats.cached == 16 and sum(len(c) for c in embed.calls) == 24
    cache = R.VectorCache(tmp_path / "v.jsonl")
    assert all(R.cache_key("m", t) in cache for t in texts)
    # A torn last line (a kill mid-append) costs only that line.
    with open(tmp_path / "v.jsonl", "a") as fh:
        fh.write('{"key": "tor')
    assert len(R.VectorCache(tmp_path / "v.jsonl").vectors) == 40


# --------------------------------------------------------------------------- scoring with the production code


def test_cosine_gate_is_the_rule_investigation_reason_used():
    rng = np.random.default_rng(5)
    for _ in range(50):
        cos = [round(float(x), 2) for x in rng.uniform(0.4, 0.8, size=int(rng.integers(0, 30)))]
        scored = [(c, i) for i, c in enumerate(cos)]
        old = [f for _, f in sorted([(c, f) for c, f in scored if c >= 0.59], key=lambda x: x[0], reverse=True)[:12]]
        assert grounding_gate.cosine_gate(scored, 0.59) == old
    assert grounding_gate.cosine_gate([(None, "a"), (0.9, "b")]) == ["b"]
    assert d10_gate.CONTEXT_CAP == grounding_gate.CONTEXT_CAP == 12


def test_replay_decisions_equal_the_live_shadow_log(tmp_path, monkeypatch):
    """Same store, same embeddings: the replay's per-pair decisions are the live ones."""
    mem = tmp_path / "live-memory"
    mem.mkdir()
    original = server.MEMORY_DIR
    server.MEMORY_DIR = mem
    monkeypatch.setenv("LOCI_MEMORY_DIR", str(mem))
    monkeypatch.setenv("LOCI_D10_SHADOW", "1")
    monkeypatch.setattr(d10_gate, "_cached", None)
    try:
        inv = "replay-parity"
        server.investigation_start(investigation_id=inv, title="alpha parity")
        for i in range(30):
            server.investigation_store(investigation_id=inv, finding_type="observed",
                                       text=f"{'alpha' if i % 3 else 'omega'} parity finding {i} value {i}",
                                       source="unit-test", confidence="high")
        question = "alpha what drives the parity result"
        monkeypatch.setattr(llm, "llm_available", lambda: True)
        monkeypatch.setattr(llm, "embed_texts", lambda texts, **kw: [_vec(t) for t in texts])
        monkeypatch.setattr(llm, "call_llm", lambda prompt, **kw: (json.dumps(
            {"converged_claims": [], "contested_areas": [], "final_answer": "x", "confidence_score": 1})
            if kw.get("json_mode") else "analysis"))
        server.investigation_reason(inv, question, perspectives=1, persist=False)
        log = mem.parent / "instrumentation" / d10_gate.SHADOW_LOG_NAME
        live = {r["finding_id"]: r for r in map(json.loads, log.read_text().splitlines())}
        fs = R.active_findings(mem / inv)
        rows = R.score_question(question, _vec(question), fs, [_vec(str(f["text"])) for f in fs], d10_gate.load_gate())
    finally:
        server.MEMORY_DIR = original
    assert len(rows) == len(live) == 30
    kept = 0
    for r in rows:
        lv = live[r["finding_id"]]
        assert (r["cos_keep"], r["cos_in_context"], r["mlp_keep"], r["mlp_in_context"]) == \
               (lv["cos_keep"], lv["cos_in_context"], lv["mlp_keep"], lv["mlp_in_context"])
        assert r["mlp_score"] == pytest.approx(lv["mlp_score"], abs=1e-6)
        assert r["cos"] == pytest.approx(lv["cos"], abs=1e-4)
        kept += r["cos_keep"]
    assert 0 < kept < 30


def test_end_to_end_run_scores_every_pair_and_never_writes_into_the_store(store, tmp_path):
    before = _snapshot(store)
    out = tmp_path / "private"
    meta, embed, state = _run(store, out, tmp_path)
    assert _snapshot(store) == before
    assert meta["status"] == "done"
    pairs = [json.loads(line) for line in (out / R.PAIRS_NAME).read_text().splitlines()]
    eligible, _ = R.enumerate_investigations(store)
    assert len(pairs) == meta["pairs"] == sum(e["n_findings"] * len(e["questions"]) for e in eligible)
    for p in pairs:
        assert p["category"] == R.categorise(p["cos_keep"], p["mlp_keep"])
        assert p["cos_keep"] == (p["cos"] is not None and p["cos"] >= 0.59)
    assert sum(meta["categories"].values()) == len(pairs)
    assert set(meta["excluded_by_reason"]) and meta["seed"] == 1
    # Rerun: everything comes from the cache, nothing is re-embedded.
    meta2, embed2, _ = _run(store, out, tmp_path)
    assert embed2.calls == [] and meta2["pairs"] == meta["pairs"]
    assert _snapshot(store) == before


def test_output_dir_inside_the_store_is_refused(store, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for bad in (store / "inv-0" / "out", store, tmp_path / "home" / ".loci" / "x", tmp_path / "home" / ".hermes"):
        with pytest.raises(R.ReplayError):
            R.check_out_dir(bad, store)
    assert R.check_out_dir(tmp_path / "elsewhere", store) == (tmp_path / "elsewhere").resolve()


# --------------------------------------------------------------------------- labelling sample and sheet


def _rows(counts):
    rows = []
    for cat, n in counts.items():
        for i in range(n):
            rows.append({"investigation_id": f"inv-{i % 7}", "question_kind": "title", "finding_index": i,
                         "finding_id": f"{cat}-{i}", "category": cat})
    return rows


def test_draw_sample_takes_targets_or_all_and_places_duplicates_later():
    rows = _rows({"cos_only": 40, "mlp_only": 300, "both_keep": 50, "both_drop": 900})
    items, design = R.draw_sample(rows, R.SampleTargets(), seed=9)
    again, _ = R.draw_sample(rows, R.SampleTargets(), seed=9)
    assert [i["id"] for i in items] == [i["id"] for i in again]
    assert {c: design[c]["sampled"] for c in R.CATEGORIES} == {"cos_only": 40, "mlp_only": 100, "both_keep": 30,
                                                                "both_drop": 30}
    assert len(items) == 200 + 12 and len({i["id"] for i in items}) == len(items)
    pos = {it["id"]: k for k, it in enumerate(items)}
    dups = [it for it in items if it["duplicate_of"]]
    assert len(dups) == 12
    for d in dups:
        assert pos[d["id"]] - pos[d["duplicate_of"]] >= 30 or pos[d["id"]] == len(items) - 1 or \
            pos[d["duplicate_of"]] + 30 > len(items) - 12
        assert d["pair"] is items[pos[d["duplicate_of"]]]["pair"]
    # Shuffled: categories are interleaved, not in blocks.
    first = Counter(it["pair"]["category"] for it in items[:60])
    assert len(first) >= 3


def _sampled(store, tmp_path):
    out = tmp_path / "private"
    _run(store, out, tmp_path)
    plan = tmp_path / "plan.md"
    plan.write_text("plan\n")
    res = R.build_sample(out, store, plan, seed=4, targets=R.SampleTargets(cos_only=10, mlp_only=10, both_keep=5,
                                                                          both_drop=5, duplicates=3, min_dup_gap=5),
                         allow_uncommitted_plan=True)
    return out, res


def test_sheet_shows_no_gate_information_and_the_key_is_separate(store, tmp_path):
    before = _snapshot(store)
    out, res = _sampled(store, tmp_path)
    assert _snapshot(store) == before
    sheet = (out / R.SHEET_NAME).read_text()
    key = json.loads((out / R.KEY_NAME).read_text())
    items = [json.loads(line) for line in (out / R.ITEMS_NAME).read_text().splitlines()]
    assert all(set(i) == {"id", "question", "finding"} for i in items)
    assert set(key["items"]) == {i["id"] for i in items}
    payload = sheet.split('<script id="items" type="application/json">', 1)[1].split("</script>", 1)[0]
    parsed = json.loads(payload)
    assert parsed == items
    lowered = sheet.lower()
    for word in ("cos_only", "mlp_only", "both_keep", "both_drop", "mlp_score", "cosine", "category", "gate",
                 "investigation_id", "inv-", "question_kind", "tau", "0.59", "d10"):
        assert word not in lowered, word
    for v in key["items"].values():
        assert f"{v['mlp_score']}" not in payload
    # Self-contained: no external fetches of any kind.
    assert "src=" not in lowered and "http" not in lowered.replace('http-equiv', '') and "@import" not in lowered
    assert "default-src 'none'" in sheet
    assert key["sheet_id"] == res["sheet_id"] and f'"{res["sheet_id"]}"' in sheet
    with pytest.raises(R.ReplayError):     # drawn once
        R.build_sample(out, store, tmp_path / "plan.md", seed=4, targets=R.SampleTargets(), allow_uncommitted_plan=True)


def test_sheet_text_cannot_break_out_of_its_script_element():
    q = "q</script><script>alert(1)</script></SCRIPT x"
    html_text = R.render_sheet([{"id": "a", "question": q, "finding": "<!-- <script> & -->"}], "abc")
    payload = html_text.split('<script id="items" type="application/json">', 1)[1].split("</script>\n<script>", 1)[0]
    # The HTML tokenizer ends a script element at "</script" + space, "/" or ">", and
    # "<!--" + "<script" changes its state: no "<" may reach the payload at all.
    assert "<" not in payload and ">" not in payload
    assert json.loads(payload)[0]["question"] == q and json.loads(payload)[0]["finding"] == "<!-- <script> & -->"
    with pytest.raises(ValueError):
        R.render_sheet([{"id": "a", "question": "q", "finding": "f", "category": "cos_only"}], "abc")


def test_sampling_requires_a_committed_plan(tmp_path):
    plan = tmp_path / "plan.md"
    with pytest.raises(R.ReplayError):
        R.plan_provenance(plan, allow_uncommitted=False)
    plan.write_text("x")
    with pytest.raises(R.ReplayError):
        R.plan_provenance(plan, allow_uncommitted=False)
    prov = R.plan_provenance(plan, allow_uncommitted=True)
    assert prov["sha256"] == hashlib.sha256(b"x").hexdigest() and prov["committed_clean"] is False


def test_terminal_labeller_appends_resumes_and_undoes(store, tmp_path):
    out, _ = _sampled(store, tmp_path)
    answers = iter(["e", "1", "2", "u", "3", "q"])
    R.label_cli(out / R.ITEMS_NAME, out / R.CLI_LABELS_NAME, inp=lambda _p: next(answers), out=lambda _s: None)
    labels = R.read_cli_labels(out / R.CLI_LABELS_NAME)
    items = [json.loads(line) for line in (out / R.ITEMS_NAME).read_text().splitlines()]
    assert labels == {items[0]["id"]: "relevant", items[1]["id"]: "unsure"}
    rest = iter(["1"] * 200)
    R.label_cli(out / R.ITEMS_NAME, out / R.CLI_LABELS_NAME, inp=lambda _p: next(rest), out=lambda _s: None)
    labels = R.read_cli_labels(out / R.CLI_LABELS_NAME)
    assert len(labels) == len(items)
    sid, parsed = A.load_labels(out / R.CLI_LABELS_NAME)
    assert sid is None and parsed == labels
    res = A.analyse(json.loads((out / R.KEY_NAME).read_text()), parsed, reps=50)
    assert res["labelled_primary"] == res["primary_items"]


# --------------------------------------------------------------------------- analysis (known answers)


def _key(population, items):
    return {"sheet_id": "s", "plan": {"sha256": "p"}, "population": population,
            "items": {i: {"investigation_id": inv, "category": cat, "question_kind": "title", "duplicate_of": dup}
                      for i, (inv, cat, dup) in items.items()}}


def test_sign_test_is_exact_and_two_sided():
    assert A.sign_test_p(10, 10) == pytest.approx(2 / 1024)
    assert A.sign_test_p(0, 10) == pytest.approx(2 / 1024)
    assert A.sign_test_p(15, 20) == pytest.approx(2 * sum(math.comb(20, i) for i in range(6)) / 2 ** 20)
    assert A.sign_test_p(5, 10) == 1.0
    assert A.sign_test_p(0, 0) is None


def test_stratified_estimates_have_a_known_answer():
    # N: cos_only 100, mlp_only 300, both_keep 50, both_drop 550.
    # p: cos_only 0.2, mlp_only 0.8, both_keep 1.0, both_drop 0.1 relevant.
    N = {"cos_only": 100, "mlp_only": 300, "both_keep": 50, "both_drop": 550}
    rel = {"cos_only": 2, "mlp_only": 8, "both_keep": 10, "both_drop": 1}
    dec = {"cos_only": 10, "mlp_only": 10, "both_keep": 10, "both_drop": 10}
    est = A.stratified_rates(N, rel, dec)
    Rr = {"cos_only": 20, "mlp_only": 240, "both_keep": 50, "both_drop": 55}
    Bb = {"cos_only": 80, "mlp_only": 60, "both_keep": 0, "both_drop": 495}
    tr, tb = sum(Rr.values()), sum(Bb.values())
    assert est["cosine_grounded_recall"] == pytest.approx((20 + 50) / tr)
    assert est["mlp_grounded_recall"] == pytest.approx((240 + 50) / tr)
    assert est["cosine_bleed_rejection"] == pytest.approx((60 + 495) / tb)
    assert est["mlp_bleed_rejection"] == pytest.approx((80 + 495) / tb)
    assert est["mlp_correct_share_weighted"] == pytest.approx((80 + 240) / 400)
    assert est["mlp_recall_of_either_kept"] == pytest.approx(290 / 310)
    assert A.stratified_rates(N, rel, {**dec, "both_drop": 0}) is None


def test_analysis_on_synthetic_labels_known_answer_and_duplicates_excluded():
    items, labels = {}, {}
    k = 0
    plan = [("cos_only", 10, 2), ("mlp_only", 10, 8), ("both_keep", 10, 10), ("both_drop", 10, 1)]
    for cat, n, n_rel in plan:
        for j in range(n):
            iid = f"i{k}"
            items[iid] = (f"inv{k % 5}", cat, None)
            labels[iid] = "relevant" if j < n_rel else "not_relevant"
            k += 1
    items["u1"] = ("inv0", "cos_only", None)
    labels["u1"] = "unsure"
    items["d1"] = ("inv0", "cos_only", "i0")
    labels["d1"] = "not_relevant"              # disagrees with i0 (relevant)
    items["d2"] = ("inv1", "mlp_only", "i10")
    labels["d2"] = "relevant"                  # agrees with i10
    pop = {f"inv{j}": {"cos_only": 20, "mlp_only": 60, "both_keep": 10, "both_drop": 110} for j in range(5)}
    res = A.analyse(_key(pop, items), labels, reps=200, seed=1)
    d = res["disagreements"]
    # MLP correct: 8 cos_only not-relevant + 8 mlp_only relevant = 16 of 20.
    assert (d["decisive"], d["mlp_correct"], d["cosine_correct"]) == (20, 16, 4)
    assert d["sign_test_p_two_sided"] == pytest.approx(A.sign_test_p(16, 20))
    assert d["unsure_rate"] == pytest.approx(1 / 21)
    assert res["population_estimates"]["mlp_correct_share_weighted"] == pytest.approx((80 + 240) / 400)
    c = res["consistency"]
    assert (c["duplicates_labelled"], c["duplicates_agree"]) == (2, 1)
    assert c["control_relevant_share"] == {"both_keep": 1.0, "both_drop": 0.1} and c["controls_ordered"]
    assert res["decision"]["verdict"] == "cosine_remains_default"   # MLP recall 290/365 < 0.95
    assert not res["decision"]["checks"]["mlp_grounded_recall_at_least_0.95"]
    assert res["decision"]["checks"]["mlp_correct_share_at_least_0.60"]
    text = json.dumps(res)
    assert "inv0" not in text and '"i1' not in text               # aggregates only


def test_decision_passes_only_when_every_check_holds():
    items, labels = {}, {}
    for j in range(40):
        items[f"m{j}"] = (f"inv{j % 8}", "mlp_only", None)
        labels[f"m{j}"] = "relevant"
        items[f"c{j}"] = (f"inv{j % 8}", "cos_only", None)
        labels[f"c{j}"] = "not_relevant"
    for j in range(16):
        items[f"k{j}"] = (f"inv{j % 8}", "both_keep", None)
        labels[f"k{j}"] = "relevant"
        items[f"d{j}"] = (f"inv{j % 8}", "both_drop", None)
        labels[f"d{j}"] = "not_relevant"
    pop = {f"inv{j}": {"cos_only": 10, "mlp_only": 10, "both_keep": 5, "both_drop": 50} for j in range(8)}
    res = A.analyse(_key(pop, items), labels, reps=200, seed=2)
    assert all(res["decision"]["checks"].values()), res["decision"]
    assert res["decision"]["verdict"] == "mlp_candidate_for_enforcement_pending_live_shadow"
    labels["m0"] = labels["m1"] = labels["m2"] = labels["m3"] = labels["m4"] = "unsure"
    for j in range(5, 21):
        labels[f"m{j}"] = "unsure"
    res2 = A.analyse(_key(pop, items), labels, reps=200, seed=2)
    assert not res2["decision"]["checks"]["unsure_rate_below_0.25"]
    assert res2["decision"]["verdict"] == "cosine_remains_default"


def test_bootstrap_resamples_investigations_not_pairs():
    # inv A: 10 disagreements, all MLP-right; inv B: 10, all cosine-right.
    items, labels = {}, {}
    for j in range(10):
        items[f"a{j}"] = ("A", "mlp_only", None)
        labels[f"a{j}"] = "relevant"
        items[f"b{j}"] = ("B", "mlp_only", None)
        labels[f"b{j}"] = "not_relevant"
    pop = {"A": {"mlp_only": 10}, "B": {"mlp_only": 10}}
    triples = [(inv, cat, labels[i]) for i, (inv, cat, _d) in items.items()]
    boot = A.cluster_bootstrap(pop, triples, reps=400, seed=3)
    # Cluster resampling can only produce 0, 1/2 or 1; a pair bootstrap would give ~[0.3, 0.7].
    assert boot["ci95"]["mlp_correct_share"] == [0.0, 1.0]


def test_labels_for_another_sheet_are_refused(tmp_path):
    key = _key({"x": {"cos_only": 1}}, {"i": ("x", "cos_only", None)})
    with pytest.raises(ValueError):
        A.analyse(key, {"i": "relevant"}, sheet_id="other")
    with pytest.raises(ValueError):
        A.analyse(key, {"zz": "relevant"})
    p = tmp_path / "l.json"
    p.write_text(json.dumps({"sheet_id": "s", "labels": {"i": {"label": "relevant", "ts": "t"}, "j": {"label": "bogus"}}}))
    assert A.load_labels(p) == ("s", {"i": "relevant"})
