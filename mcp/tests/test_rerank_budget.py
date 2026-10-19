"""The rerank budget: a search must not spend minutes scoring pairs.

A CPU-only torch build ran the 568M-parameter cross-encoder at 0.1-0.3 s per (query, doc) pair, and one
rag_context_search scores hundreds of pairs, so searches took 60-130 s. These tests drive the budget with a
fake model and a fake clock, so every number below is exact: which pairs were scored, in what order the rows
come back, the degraded flag, and the counters.
"""
import json
import sys
import types
from pathlib import Path

import pytest

_MCP_DIR = Path(__file__).resolve().parent.parent
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

import qdrant_ops  # noqa: E402
import reranker as R  # noqa: E402
import server  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeModel:
    """Records every predict() call; each pair costs `per_pair` fake seconds. score = number after the dash/letter."""

    def __init__(self, clock, per_pair=1.0, fail_on_call=None):
        self.clock = clock
        self.per_pair = per_pair
        self.fail_on_call = fail_on_call
        self.calls = []

    def predict(self, pairs):
        self.calls.append([doc for _, doc in pairs])
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise RuntimeError("boom")
        self.clock.t += self.per_pair * len(pairs)
        return [float(doc[1:]) for _, doc in pairs]


def _pairs(n):
    return [("q", f"t{i}") for i in range(n)]


def _rows(n):
    return [{"id": f"t{i}", "text": f"t{i}"} for i in range(n)]


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch):
    for name in ("LOCI_RERANK_BUDGET_S", "LOCI_RERANK_MAX_PAIRS", "LOCI_RERANK_CHUNK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(R, "_stats", {
        "searches": 0, "degraded_searches": 0, "pairs_scored": 0, "pairs_skipped": 0,
        "last_seconds": None, "last_degraded": None, "last_reason": None, "last_at": None})


# --------------------------------------------------------------------------- score_pairs

def test_within_budget_every_pair_is_scored_in_chunks():
    clock = Clock()
    model = FakeModel(clock)
    b = R.RerankBudget(budget_s=100, max_pairs=0, chunk=2, clock=clock)
    scores = R.score_pairs(model, _pairs(5), b)
    assert scores == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert [len(c) for c in model.calls] == [2, 2, 1]
    assert (b.scored, b.skipped, b.degraded, b.reason, b.scoring_s) == (5, 0, False, None, 5.0)


def test_deadline_keeps_the_scores_obtained_and_leaves_the_tail_unscored():
    clock = Clock()
    model = FakeModel(clock)
    # candidate stage may use 10 * 0.75 = 7.5 s: chunks run at t=0 and t=4, the check at t=8 stops it.
    b = R.RerankBudget(budget_s=10, max_pairs=0, chunk=4, clock=clock)
    scores = R.score_pairs(model, _pairs(20), b)
    assert scores == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] + [None] * 12
    assert [len(c) for c in model.calls] == [4, 4]
    assert (b.scored, b.skipped, b.degraded, b.reason, b.scoring_s) == (8, 12, True, "deadline", 8.0)


def test_final_pass_may_use_the_whole_budget_where_the_candidate_stage_may_not():
    clock = Clock()
    model = FakeModel(clock)
    b = R.RerankBudget(budget_s=10, max_pairs=0, chunk=4, clock=clock)
    scores = R.score_pairs(model, _pairs(20), b, final=True)
    # full 10 s: chunks at t=0, 4, 8 (8 < 10), then 12 stops it.
    assert [s is not None for s in scores].count(True) == 12
    assert [len(c) for c in model.calls] == [4, 4, 4]
    assert (b.scored, b.skipped, b.reason) == (12, 8, "deadline")


def test_candidate_stage_leaves_the_last_quarter_for_the_final_pass():
    clock = Clock()
    model = FakeModel(clock)
    b = R.RerankBudget(budget_s=10, max_pairs=0, chunk=4, clock=clock)
    R.score_pairs(model, _pairs(20), b)           # stops at t=8
    assert clock.t == 8.0
    assert b.allowance(final=False) == (0, "deadline")
    allowed, why = b.allowance(final=True)
    assert allowed > 0 and why is None


def test_a_deadline_already_passed_calls_the_model_zero_times():
    clock = Clock()
    model = FakeModel(clock)
    b = R.RerankBudget(budget_s=5, max_pairs=0, chunk=4, clock=clock)
    clock.t = 100.0
    scores = R.score_pairs(model, _pairs(3), b)
    assert scores == [None, None, None]
    assert model.calls == []
    assert (b.scored, b.skipped, b.reason) == (0, 3, "deadline")


def test_max_pairs_caps_the_candidate_stage_but_not_the_final_pass():
    clock = Clock()
    model = FakeModel(clock)
    b = R.RerankBudget(budget_s=0, max_pairs=6, chunk=4, clock=clock)
    scores = R.score_pairs(model, _pairs(10), b)
    assert scores == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] + [None] * 4
    assert [len(c) for c in model.calls] == [4, 2]
    assert (b.scored, b.skipped, b.reason) == (6, 4, "max_pairs")
    final = R.score_pairs(model, _pairs(3), b, final=True)
    assert final == [0.0, 1.0, 2.0]
    assert b.scored == 9


def test_zero_disables_both_limits():
    clock = Clock()
    model = FakeModel(clock, per_pair=1000.0)
    b = R.RerankBudget(budget_s=0, max_pairs=0, chunk=3, clock=clock)
    scores = R.score_pairs(model, _pairs(7), b)
    assert scores == [float(i) for i in range(7)]
    assert (b.scored, b.skipped, b.degraded) == (7, 0, False)


def test_a_failing_model_propagates_and_records_nothing_for_the_failed_chunk():
    clock = Clock()
    model = FakeModel(clock, fail_on_call=2)
    b = R.RerankBudget(budget_s=0, max_pairs=0, chunk=2, clock=clock)
    with pytest.raises(RuntimeError, match="boom"):
        R.score_pairs(model, _pairs(6), b)
    assert [len(c) for c in model.calls] == [2, 2]
    assert (b.scored, b.scoring_s) == (2, 2.0)


def test_a_model_returning_the_wrong_number_of_scores_is_an_error_not_a_silent_truncation():
    class Short:
        def predict(self, pairs):
            return [1.0]
    b = R.RerankBudget(budget_s=0, max_pairs=0, chunk=3, clock=Clock())
    with pytest.raises(ValueError, match="1 scores for 3 pairs"):
        R.score_pairs(Short(), _pairs(3), b)


def test_environment_configures_the_budget_and_bad_values_fall_back(monkeypatch):
    monkeypatch.setenv("LOCI_RERANK_BUDGET_S", "7.5")
    monkeypatch.setenv("LOCI_RERANK_MAX_PAIRS", "9")
    monkeypatch.setenv("LOCI_RERANK_CHUNK", "3")
    b = R.RerankBudget(clock=Clock())
    assert (b.budget_s, b.max_pairs, b.chunk) == (7.5, 9, 3)
    monkeypatch.setenv("LOCI_RERANK_BUDGET_S", "soon")
    monkeypatch.delenv("LOCI_RERANK_MAX_PAIRS")
    monkeypatch.delenv("LOCI_RERANK_CHUNK")
    b = R.RerankBudget(clock=Clock())
    assert (b.budget_s, b.max_pairs, b.chunk) == (20.0, 0, 16)


# --------------------------------------------------------------------------- scopes and stats

def test_a_nested_scope_shares_the_outer_budget_and_records_one_search():
    clock = Clock()
    model = FakeModel(clock)
    with R.search_budget(budget_s=0, max_pairs=0, chunk=2, clock=clock) as outer:
        with R.search_budget() as inner:
            assert inner is outer
            R.score_pairs(model, _pairs(3), inner)
        assert R.stats()["searches"] == 0     # the inner exit must not record
    assert R.stats()["searches"] == 1
    assert R.stats()["pairs_scored"] == 3


def test_stats_count_degraded_searches_and_remember_the_last_one():
    clock = Clock()
    model = FakeModel(clock)
    with R.search_budget(budget_s=10, max_pairs=0, chunk=4, clock=clock) as b:
        R.score_pairs(model, _pairs(20), b)
    s = R.stats()
    assert (s["searches"], s["degraded_searches"], s["pairs_scored"], s["pairs_skipped"]) == (1, 1, 8, 12)
    assert (s["last_seconds"], s["last_degraded"], s["last_reason"]) == (8.0, True, "deadline")
    clock2 = Clock()
    with R.search_budget(budget_s=0, max_pairs=0, chunk=4, clock=clock2) as b2:
        R.score_pairs(FakeModel(clock2), _pairs(3), b2)
    s = R.stats()
    assert (s["searches"], s["degraded_searches"], s["pairs_scored"], s["pairs_skipped"]) == (2, 1, 11, 12)
    assert (s["last_seconds"], s["last_degraded"], s["last_reason"]) == (3.0, False, None)


def test_a_scope_that_never_reached_the_scorer_is_not_counted():
    with R.search_budget(budget_s=5, clock=Clock()):
        pass
    assert R.stats()["searches"] == 0


# --------------------------------------------------------------------------- health

def _set_state(monkeypatch, model, device, torch_build):
    monkeypatch.setattr(R, "_model", model)
    monkeypatch.setattr(R, "_device", device)
    monkeypatch.setattr(R, "_torch_build", torch_build)
    monkeypatch.setenv("RERANK_MODEL", "some/reranker")


def test_health_reports_device_model_and_budget_without_loading_anything(monkeypatch):
    loads = []
    monkeypatch.setattr(R, "_get_model", lambda: loads.append(1))
    _set_state(monkeypatch, object(), "cpu", "2.14.1+cpu")
    h = R.health()
    assert (h["model"], h["state"], h["device"], h["torch"]) == ("some/reranker", "loaded", "cpu", "2.14.1+cpu")
    assert (h["budget_s"], h["max_pairs"], h["searches"], h["degraded_searches"]) == (20.0, 0, 0, 0)
    _set_state(monkeypatch, None, "cpu", "2.14.1+cpu")
    h = R.health()
    assert (h["state"], h["device"]) == ("not_loaded", None)
    _set_state(monkeypatch, False, "cpu", "2.14.1+cpu")
    h = R.health()
    assert (h["state"], h["device"]) == ("unavailable", None)
    assert loads == [], "health must never load the model"


def test_health_never_raises(monkeypatch):
    def boom():
        raise RuntimeError("config exploded")
    monkeypatch.setattr(R, "_model_id", boom)
    h = R.health()
    assert h == {"state": "error", "error": "RuntimeError: config exploded"}


def _fake_torch(monkeypatch, cuda, version):
    torch = types.ModuleType("torch")
    torch.__version__ = version
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda, device_count=lambda: 1 if cuda else 0, get_device_name=lambda i: "Fake GPU %d" % i, mem_get_info=lambda i: (8 << 30, 11 << 30), empty_cache=lambda: None)
    st = types.ModuleType("sentence_transformers")
    created = []

    class CrossEncoder:
        def __init__(self, name, max_length, device):
            created.append((name, max_length, device))
    st.CrossEncoder = CrossEncoder
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)
    monkeypatch.setattr(R, "_model", None)
    monkeypatch.setattr(R, "_model_name", None)
    monkeypatch.setattr(R, "_device", None)
    monkeypatch.setattr(R, "_torch_build", None)
    monkeypatch.setattr(R, "_placement", {})
    monkeypatch.delenv("RERANK_HTTP_URL", raising=False)
    monkeypatch.delenv("LOCI_RERANK_DEVICE", raising=False)
    monkeypatch.delenv("LOCI_RERANK_MIN_FREE_MB", raising=False)
    monkeypatch.setenv("RERANK_MODEL", "some/reranker")
    return created


def test_loading_on_a_cpu_only_torch_records_cpu_and_the_build(monkeypatch):
    created = _fake_torch(monkeypatch, cuda=False, version="2.14.1+cpu")
    assert R._get_model() is not None
    assert created == [("some/reranker", 512, "cpu")]
    h = R.health()
    assert (h["state"], h["device"], h["torch"]) == ("loaded", "cpu", "2.14.1+cpu")


def test_loading_with_cuda_records_cuda(monkeypatch):
    created = _fake_torch(monkeypatch, cuda=True, version="2.11.0+cu128")
    assert R._get_model() is not None
    assert created == [("some/reranker", 512, "cuda:0")]
    h = R.health()
    assert (h["state"], h["device"], h["torch"]) == ("loaded", "cuda", "2.11.0+cu128")


def test_loci_health_exposes_the_reranker_and_warns_on_cpu(monkeypatch):
    _set_state(monkeypatch, object(), "cpu", "2.14.1+cpu")
    out = json.loads(server.loci_health())
    assert out["reranker"]["device"] == "cpu"
    assert out["reranker"]["model"] == "some/reranker"
    assert any("reranker some/reranker runs on CPU (torch 2.14.1+cpu)" in w for w in out["warnings"])


def test_loci_health_does_not_warn_when_the_reranker_runs_on_cuda(monkeypatch):
    _set_state(monkeypatch, object(), "cuda", "2.11.0+cu128")
    out = json.loads(server.loci_health())
    assert out["reranker"]["device"] == "cuda"
    assert not any("runs on CPU" in w for w in out.get("warnings", []))


def test_loci_health_warns_when_the_budget_cut_searches_short(monkeypatch):
    _set_state(monkeypatch, object(), "cuda", "2.11.0+cu128")
    clock = Clock()
    with R.search_budget(budget_s=10, max_pairs=0, chunk=4, clock=clock) as b:
        R.score_pairs(FakeModel(clock), _pairs(20), b)
    out = json.loads(server.loci_health())
    assert "reranker budget cut 1 of 1 searches short (last: deadline, 8.0s)" in out["warnings"]


# --------------------------------------------------------------------------- qdrant_ops._ce_rerank

def test_ce_rerank_under_budget_is_exactly_todays_output(monkeypatch):
    clock = Clock()
    model = FakeModel(clock)
    monkeypatch.setattr(qdrant_ops, "_get_cross_encoder", lambda: model)
    with R.search_budget(budget_s=100, max_pairs=0, chunk=2, clock=clock) as b:
        rows, reranked = qdrant_ops._ce_rerank("q", _rows(6), 4)
    assert reranked is True
    assert [r["id"] for r in rows] == ["t5", "t4", "t3", "t2"]
    assert [r["ce_score"] for r in rows] == [5.0, 4.0, 3.0, 2.0]
    assert b.degraded is False


def test_ce_rerank_out_of_budget_keeps_scored_rows_first_then_the_original_order(monkeypatch):
    clock = Clock()
    model = FakeModel(clock, per_pair=2.0)
    monkeypatch.setattr(qdrant_ops, "_get_cross_encoder", lambda: model)
    # stage deadline 7.5 s, 4 s per chunk of 2: t0 and t3 are scored, t4 and t5 are not.
    with R.search_budget(budget_s=10, max_pairs=0, chunk=2, clock=clock) as b:
        rows, reranked = qdrant_ops._ce_rerank("q", _rows(6), 5)
    assert reranked is True
    assert [r["id"] for r in rows] == ["t3", "t2", "t1", "t0", "t4"]
    assert [r.get("ce_score") for r in rows] == [3.0, 2.0, 1.0, 0.0, None]
    assert [len(c) for c in model.calls] == [2, 2]
    assert b.summary() == {"degraded": True, "reason": "deadline", "pairs_scored": 4, "pairs_skipped": 2,
                           "seconds": 8.0, "budget_s": 10.0, "max_pairs": 0}


def test_ce_rerank_with_nothing_scored_keeps_the_bi_encoder_order(monkeypatch):
    clock = Clock()
    model = FakeModel(clock)
    monkeypatch.setattr(qdrant_ops, "_get_cross_encoder", lambda: model)
    with R.search_budget(budget_s=5, max_pairs=0, chunk=2, clock=clock) as b:
        clock.t = 50.0
        rows, reranked = qdrant_ops._ce_rerank("q", _rows(4), 3)
    assert reranked is False
    assert [r["id"] for r in rows] == ["t0", "t1", "t2"]
    assert all("ce_score" not in r for r in rows)
    assert model.calls == []
    assert (b.degraded, b.reason, b.skipped) == (True, "deadline", 4)


def test_ce_rerank_model_error_fails_open_and_marks_the_search_degraded(monkeypatch):
    clock = Clock()
    model = FakeModel(clock, fail_on_call=2)
    monkeypatch.setattr(qdrant_ops, "_get_cross_encoder", lambda: model)
    with R.search_budget(budget_s=0, max_pairs=0, chunk=2, clock=clock) as b:
        rows, reranked = qdrant_ops._ce_rerank("q", _rows(6), 4)
    assert len(model.calls) == 2, "the failure must come from the second chunk, after one was scored"
    assert reranked is False
    assert [r["id"] for r in rows] == ["t0", "t1", "t2", "t3"]
    assert all("ce_score" not in r for r in rows), "partial scores from the first chunk must be discarded"
    assert (b.degraded, b.reason) == (True, "error:RuntimeError")


def test_ce_rerank_without_an_enclosing_search_gets_its_own_budget(monkeypatch):
    clock = Clock()
    model = FakeModel(clock, per_pair=2.0)
    monkeypatch.setattr(qdrant_ops, "_get_cross_encoder", lambda: model)
    monkeypatch.setattr(R, "_now", clock)
    monkeypatch.setenv("LOCI_RERANK_BUDGET_S", "10")
    monkeypatch.setenv("LOCI_RERANK_CHUNK", "2")
    rows, reranked = qdrant_ops._ce_rerank("q", _rows(6), 6)
    assert reranked is True
    assert [r["id"] for r in rows] == ["t3", "t2", "t1", "t0", "t4", "t5"]
    s = R.stats()
    assert (s["searches"], s["degraded_searches"], s["pairs_scored"], s["pairs_skipped"]) == (1, 1, 4, 2)


# --------------------------------------------------------------------------- server._rag_cross_encode

def _final_rows(n):
    return [{"id": f"t{i}", "text": f"t{i}"} for i in range(n)]


def test_final_pass_under_budget_matches_the_unbudgeted_ordering(monkeypatch):
    clock = Clock()
    model = FakeModel(clock)
    monkeypatch.setattr(server, "_get_cross_encoder", lambda: model)
    rows = _final_rows(23)
    with R.search_budget(budget_s=0, max_pairs=0, chunk=5, clock=clock):
        server._rag_cross_encode(rows, "q")
    ids = [r["id"] for r in rows]
    assert ids[:20] == [f"t{i}" for i in range(19, -1, -1)]
    assert ids[20:] == ["t20", "t21", "t22"]
    assert [r["final_ce_score"] for r in rows[:3]] == [19.0, 18.0, 17.0]


def test_final_pass_out_of_budget_orders_the_scored_head_and_keeps_the_rest_in_place(monkeypatch):
    clock = Clock()
    model = FakeModel(clock)
    monkeypatch.setattr(server, "_get_cross_encoder", lambda: model)
    rows = _final_rows(23)
    # full 10 s, 1 s per pair, chunks of 5: t0-t4 and t5-t9 are scored, then the deadline stops it.
    with R.search_budget(budget_s=10, max_pairs=0, chunk=5, clock=clock) as b:
        server._rag_cross_encode(rows, "q")
    ids = [r["id"] for r in rows]
    assert ids == [f"t{i}" for i in range(9, -1, -1)] + [f"t{i}" for i in range(10, 23)]
    assert [r.get("final_ce_score") for r in rows[:2]] == [9.0, 8.0]
    assert all("final_ce_score" not in r for r in rows[10:])
    assert (b.scored, b.skipped, b.reason) == (10, 10, "deadline")


def test_final_pass_model_error_leaves_results_untouched(monkeypatch):
    clock = Clock()
    model = FakeModel(clock, fail_on_call=2)
    monkeypatch.setattr(server, "_get_cross_encoder", lambda: model)
    rows = _final_rows(12)
    with R.search_budget(budget_s=0, max_pairs=0, chunk=5, clock=clock) as b:
        server._rag_cross_encode(rows, "q")
    assert len(model.calls) == 2
    assert [r["id"] for r in rows] == [f"t{i}" for i in range(12)]
    assert all("final_ce_score" not in r for r in rows)
    assert b.reason == "error:RuntimeError"


# --------------------------------------------------------------------------- rag_context_search end to end

def _search(monkeypatch, budget_s):
    """Two collections of six rows, real _ce_rerank and _rag_cross_encode, one fake model on a fake clock."""
    clock = Clock()
    model = FakeModel(clock)
    monkeypatch.setattr(R, "_now", clock)
    monkeypatch.setenv("LOCI_RERANK_BUDGET_S", str(budget_s))
    monkeypatch.setenv("LOCI_RERANK_CHUNK", "5")
    monkeypatch.setattr(qdrant_ops, "_get_cross_encoder", lambda: model)
    monkeypatch.setattr(server, "_get_cross_encoder", lambda: model)

    def per_collection(sq, collection_name=None, limit=None, query_filter=None):
        rows = [{"origin": collection_name, "id": f"{collection_name}{i}", "score": 0.9 - i / 100,
                 "text": f"t{i}"} for i in range(6)]
        return qdrant_ops._ce_rerank(sq, rows, limit)[0]

    monkeypatch.setattr(server, "_qdrant_search_collection", per_collection)
    monkeypatch.setattr(server, "_get_qdrant", lambda: (object(), "loci_memory"))
    monkeypatch.setattr(server, "_rag_record_access", lambda *a, **k: None)
    raw = server.rag_context_search("q", collections=["colA", "colB"], expand_query=False, decay=False)
    return json.loads(raw), model


def test_search_that_runs_out_of_budget_says_so_in_the_response(monkeypatch):
    out, model = _search(monkeypatch, 16)
    # candidate stage (deadline 12 s): colA 5+1 pairs to t=6, colB 5+1 to t=12; final pass: 5 pairs to t=17, 7 unscored.
    assert [len(c) for c in model.calls] == [5, 1, 5, 1, 5]
    assert out["rerank_degraded"] is True
    assert out["rerank"] == {"degraded": True, "reason": "deadline", "pairs_scored": 17, "pairs_skipped": 7,
                             "seconds": 17.0, "budget_s": 16.0, "max_pairs": 0}
    assert R.stats()["degraded_searches"] == 1


def test_search_with_the_budget_disabled_scores_everything_and_is_not_degraded(monkeypatch):
    out, model = _search(monkeypatch, 0)
    assert [len(c) for c in model.calls] == [5, 1, 5, 1, 5, 5, 2]
    assert "rerank_degraded" not in out
    assert out["rerank"] == {"degraded": False, "reason": None, "pairs_scored": 24, "pairs_skipped": 0,
                             "seconds": 24.0, "budget_s": 0.0, "max_pairs": 0}
    assert R.stats()["degraded_searches"] == 0


def test_search_without_a_cross_encoder_keeps_its_response_shape(monkeypatch):
    monkeypatch.setattr(qdrant_ops, "_get_cross_encoder", lambda: None)
    monkeypatch.setattr(server, "_get_cross_encoder", lambda: None)
    monkeypatch.setattr(server, "_qdrant_search_collection",
                        lambda sq, collection_name=None, limit=None, query_filter=None:
                        qdrant_ops._ce_rerank(sq, [{"origin": collection_name, "id": "a", "score": 0.5, "text": "t1"}], limit)[0])
    monkeypatch.setattr(server, "_get_qdrant", lambda: (object(), "loci_memory"))
    monkeypatch.setattr(server, "_rag_record_access", lambda *a, **k: None)
    out = json.loads(server.rag_context_search("q", collections=["colA"], expand_query=False, decay=False))
    assert out["result_count"] == 1
    assert "rerank" not in out and "rerank_degraded" not in out
    assert R.stats()["searches"] == 0
