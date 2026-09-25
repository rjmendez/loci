"""Sampled memory_route traces in the global audit log, replayable without text.

Before this, memory_route_counterfactual_simulate had nothing to replay: it
only read traces that a caller produced with include_trace=True and then
audited by hand, and the live audit log held none. A sampled trace keeps ids,
scores and the candidates' pairwise word overlap, never the query or the text.
"""
import json
from datetime import datetime, timezone

import pytest

import server

QUERY = "auth failure during the nightly batch"
TEXT_1 = "auth failure on cluster alpha during the nightly batch window"
TEXT_2 = "auth failure on cluster alpha during the nightly batch run"  # 9/11 = 0.818 with TEXT_1
TEXT_3 = "billing export stalled"
OVERLAP_12 = 9 / 11


def _hits():
    return [
        {"finding_id": "f-1", "id": "f-1", "investigation_id": "inv-1", "text": TEXT_1,
         "source": "scan of host-a", "authored_by": "agent-a", "score": 0.91},
        {"finding_id": "f-2", "id": "f-2", "investigation_id": "inv-1", "text": TEXT_2,
         "source": "scan of host-a", "authored_by": "agent-a", "score": 0.90},
        {"finding_id": "f-3", "id": "f-3", "investigation_id": "inv-2", "text": TEXT_3,
         "source": "billing job", "authored_by": "agent-b", "score": 0.5},
    ]


@pytest.fixture
def routed(tmp_path, monkeypatch):
    root = tmp_path / "memory-sessions"
    root.mkdir()
    monkeypatch.setattr(server, "MEMORY_DIR", root)
    monkeypatch.setattr(server, "_get_qdrant", lambda: (object(), "loci_memory"))
    monkeypatch.setattr(server, "_qdrant_search_collection", lambda *a, **k: [dict(h) for h in _hits()])
    monkeypatch.delenv("LOCI_ROUTE_TRACE_AUDIT_RATE", raising=False)
    return tmp_path / "audit"


def _route(**kwargs):
    return json.loads(server.memory_route(query=QUERY, top_k=3, deduplicate=True, **kwargs))


def _audit_entries(audit_dir):
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = audit_dir / f"{day}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_off_by_default(routed):
    out = _route()
    assert [r["finding_id"] for r in out["routed"]] == ["f-1", "f-3"]
    assert not routed.exists()


def test_tool_output_is_the_same_with_sampling_on(routed, monkeypatch):
    # Freeze the slow modulation state, which legitimately drifts between calls.
    monkeypatch.setattr(server, "observe", lambda *a, **k: None)
    off = _route()
    monkeypatch.setenv("LOCI_ROUTE_TRACE_AUDIT_RATE", "1")
    on = _route()
    assert on == off
    assert len(_audit_entries(routed)) == 1


def test_sampled_trace_is_ids_scores_and_overlaps_only(routed, monkeypatch):
    monkeypatch.setenv("LOCI_ROUTE_TRACE_AUDIT_RATE", "1")
    out = _route(agent_id=None)
    [entry] = _audit_entries(routed)
    assert entry["tool"] == "memory_route"
    assert entry["source"] == "route_trace_sampler"
    assert entry["investigation_id"] is None
    assert entry["evidence_provenance_tier"] == "deterministic_derived"
    assert json.loads(entry["inputs"]) == {
        "top_k": 3, "deduplicate": True, "agent_id": None,
        "drive_state_given": False, "sample_rate": 1.0,
    }
    output = json.loads(entry["output"])
    assert output["query"] == ""
    assert output["query_features"] == {"chars": len(QUERY), "tokens": 6}
    assert output["routed"] == [
        {"finding_id": "f-1", "investigation_id": "inv-1", "score": 0.91, "tier": "finding"},
        {"finding_id": "f-3", "investigation_id": "inv-2", "score": 0.5, "tier": "finding"},
    ]
    trace = output["routing_trace"]
    assert trace["version"] == 2
    assert trace["overlap_floor"] == 0.5
    assert trace["overlap_edges"] == [[0, 1, OVERLAP_12]]
    assert trace["candidate_hits"] == [
        {"trace_idx": 0, "finding_id": "f-1", "id": "f-1", "investigation_id": "inv-1",
         "authored_by": "agent-a", "score": 0.91, "tier": "finding"},
        {"trace_idx": 1, "finding_id": "f-2", "id": "f-2", "investigation_id": "inv-1",
         "authored_by": "agent-a", "score": 0.90, "tier": "finding"},
        {"trace_idx": 2, "finding_id": "f-3", "id": "f-3", "investigation_id": "inv-2",
         "authored_by": "agent-b", "score": 0.5, "tier": "finding"},
    ]
    assert trace["metrics"] == {
        "candidate_count": 3, "after_agent_filter": 3, "after_dedup": 2,
        "priority_slots": 3, "exploration_slots": 0, "after_top_k": 2,
    }
    assert trace["policy"]["drive_state"] == {"hunger": 0.0, "fatigue": 0.0, "urgency": 0.0}
    assert trace["policy"]["top_k"] == 3 and trace["policy"]["dedup_threshold"] == 0.80
    assert [r["finding_id"] for r in out["routed"]] == ["f-1", "f-3"]
    raw = json.dumps(entry)
    for forbidden in (QUERY, "cluster alpha", "billing export", "scan of host-a", "billing job"):
        assert forbidden not in raw


def test_include_trace_output_is_unchanged(routed):
    out = _route(include_trace=True)
    trace = out["routing_trace"]
    assert trace["version"] == 1
    assert [h["text"] for h in trace["candidate_hits"]] == [TEXT_1, TEXT_2, TEXT_3]
    assert trace["policy"]["dedup_threshold"] == 0.80
    assert "overlap_edges" not in trace


def _simulate(**kwargs):
    return json.loads(server.memory_route_counterfactual_simulate(**kwargs))


def test_replay_of_a_sampled_trace_reproduces_the_baseline(routed, monkeypatch):
    monkeypatch.setenv("LOCI_ROUTE_TRACE_AUDIT_RATE", "1")
    _route()
    sim = _simulate()
    assert sim["simulated"] == 1
    assert sim["changed"] == 0
    [result] = sim["results"]
    assert result["dedup_basis"] == "overlap_edges"
    assert result["dedup_exact"] is True
    assert result["counterfactual"]["added_finding_ids"] == []
    assert result["counterfactual"]["metrics"]["after_dedup"] == 2


def test_replay_uses_stored_overlaps_against_a_new_threshold(routed, monkeypatch):
    monkeypatch.setenv("LOCI_ROUTE_TRACE_AUDIT_RATE", "1")
    _route()
    looser = _simulate(dedup_threshold=0.85)["results"][0]
    assert looser["counterfactual"]["added_finding_ids"] == ["f-2"]
    assert looser["dedup_exact"] is True
    stricter = _simulate(dedup_threshold=0.80)["results"][0]
    assert stricter["counterfactual"]["added_finding_ids"] == []


def test_keyword_drives_survive_without_the_query(routed, monkeypatch):
    # "urgent" sets the urgency drive from the query text. The trace drops the
    # text, so the drive must be carried in policy.drive_state for replay.
    monkeypatch.setenv("LOCI_ROUTE_TRACE_AUDIT_RATE", "1")
    json.loads(server.memory_route(query="urgent auth failure", top_k=3, deduplicate=True))
    [entry] = _audit_entries(routed)
    output = json.loads(entry["output"])
    assert output["query"] == ""
    assert output["routing_trace"]["policy"]["drive_state"] == {
        "hunger": 0.0, "fatigue": 0.0, "urgency": 0.45}
    [result] = _simulate()["results"]
    assert result["counterfactual"]["policy"]["drive_state"]["urgency"] == 0.45
    assert result["counterfactual"]["policy"]["priority_ratio"] ==         output["routing_trace"]["policy"]["priority_ratio"]


def test_replay_below_the_floor_is_flagged_inexact(routed, monkeypatch):
    monkeypatch.setenv("LOCI_ROUTE_TRACE_AUDIT_RATE", "1")
    _route()
    below = _simulate(dedup_threshold=0.3)["results"][0]
    assert below["dedup_exact"] is False
    no_dedup = _simulate(dedup_threshold=0.3, deduplicate=False)["results"][0]
    assert no_dedup["dedup_exact"] is True


def test_sampled_and_text_traces_replay_identically(routed, monkeypatch):
    """The overlap edges stand in for the text exactly, at every threshold >= the floor."""
    monkeypatch.setenv("LOCI_ROUTE_TRACE_AUDIT_RATE", "1")
    text_trace = _route(include_trace=True)
    [entry] = _audit_entries(routed)
    sampled = json.loads(entry["output"])
    text_entry = {"tool": "memory_route", "ts": "t", "output": json.dumps(text_trace)}
    decisions_sampled, _ = server._route_counterfactual_decisions_from_audit([entry], 1)
    decisions_text, _ = server._route_counterfactual_decisions_from_audit([text_entry], 1)
    assert sampled["routing_trace"]["overlap_edges"]
    for threshold in (0.5, 0.6, 0.8, 0.81, 0.82, 0.9, 1.0):
        ids = []
        for decision in (decisions_sampled[0], decisions_text[0]):
            run = server._route_apply_policy(
                decision["candidate_hits"], top_k=3, deduplicate=True,
                dedup_threshold=threshold, overlap_fn=server._route_trace_overlap_fn(decision))
            ids.append([h["finding_id"] for h in run["hits"]])
        assert ids[0] == ids[1], threshold


def test_policy_optimize_evaluation_uses_stored_overlaps(routed, monkeypatch):
    monkeypatch.setenv("LOCI_ROUTE_TRACE_AUDIT_RATE", "1")
    _route()
    decisions, _ = server._route_counterfactual_decisions_from_audit(_audit_entries(routed), 5)
    same = server._route_eval_candidate(decisions, top_k=3, deduplicate=True,
                                        dedup_threshold=0.80, consolidation_flagged_rate=0.0)
    # Without the stored overlaps the text-less candidates would never dedup and
    # f-2 would come back as an added id.
    assert same["metrics"]["added_ids"] == 0
    assert same["metrics"]["removed_ids"] == 0
    assert same["metrics"]["stability"] == 1.0
    looser = server._route_eval_candidate(decisions, top_k=3, deduplicate=True,
                                          dedup_threshold=0.85, consolidation_flagged_rate=0.0)
    assert looser["metrics"]["added_ids"] == 1



def test_policy_optimize_evaluation_skips_decisions_below_the_floor(routed, monkeypatch):
    """Below the trace's overlap floor the stored edges cannot reproduce dedup: leave the decision out."""
    monkeypatch.setenv("LOCI_ROUTE_TRACE_AUDIT_RATE", "1")
    text_trace = _route(include_trace=True)
    [entry] = _audit_entries(routed)
    text_entry = {"tool": "memory_route", "ts": "t", "output": json.dumps(text_trace)}
    [sampled], _ = server._route_counterfactual_decisions_from_audit([entry], 1)
    [text], _ = server._route_counterfactual_decisions_from_audit([text_entry], 1)
    assert sampled["overlap_floor"] == 0.5

    def ev(decisions, threshold, dedup=True):
        return server._route_eval_candidate(decisions, top_k=3, deduplicate=dedup,
                                            dedup_threshold=threshold, consolidation_flagged_rate=0.0)

    below = ev([sampled], 0.3)
    assert below["dedup_exact"] is False
    assert below["metrics"]["decisions_skipped_inexact"] == 1
    assert below["metrics"]["decisions_compared"] == 0
    assert below["metrics"]["baseline_ids"] == 0
    at_floor = ev([sampled], 0.5)
    assert at_floor["dedup_exact"] is True
    assert at_floor["metrics"]["decisions_skipped_inexact"] == 0
    assert at_floor["metrics"]["decisions_compared"] == 1
    # no dedup, or a text trace, is exact at any threshold
    assert ev([sampled], 0.3, dedup=False)["dedup_exact"] is True
    mixed = ev([sampled, text], 0.3)
    assert mixed["dedup_exact"] is False
    assert mixed["metrics"]["decisions_skipped_inexact"] == 1
    assert mixed["metrics"]["decisions_compared"] == 1
    # only the text trace is replayed at 0.3: f-2 is still a duplicate of f-1 and f-3 is not,
    # so the routed set equals the baseline
    assert mixed["metrics"]["baseline_ids"] == 2 and mixed["metrics"]["removed_ids"] == 0

@pytest.mark.parametrize("raw,expected", [
    (None, 0.0), ("", 0.0), ("abc", 0.0), ("nan", 0.0), ("inf", 0.0),
    ("-1", 0.0), ("2", 1.0), ("0.25", 0.25),
])
def test_rate_parsing(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("LOCI_ROUTE_TRACE_AUDIT_RATE", raising=False)
    else:
        monkeypatch.setenv("LOCI_ROUTE_TRACE_AUDIT_RATE", raw)
    assert server._route_trace_audit_rate() == expected


@pytest.mark.parametrize("draw,written", [(0.49, 1), (0.5, 0), (0.9, 0)])
def test_sampling_is_a_bernoulli_draw_at_the_rate(routed, monkeypatch, draw, written):
    monkeypatch.setenv("LOCI_ROUTE_TRACE_AUDIT_RATE", "0.5")
    monkeypatch.setattr(server._route_trace_rng, "random", lambda: draw)
    _route()
    assert len(_audit_entries(routed)) == written


def test_audit_write_failure_never_breaks_routing(routed, monkeypatch):
    monkeypatch.setenv("LOCI_ROUTE_TRACE_AUDIT_RATE", "1")

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(server, "_append_jsonl", _boom)
    out = _route()
    assert [r["finding_id"] for r in out["routed"]] == ["f-1", "f-3"]


def test_overlap_edges_respect_the_floor():
    hits = [{"text": "a b c d"}, {"text": "a b c e"}, {"text": "a x y z"}, {"text": ""}]
    # (0,1): 3/5 = 0.6 kept; (0,2) and (1,2): 1/7 dropped; empty text never pairs.
    assert server._route_overlap_edges(hits) == [[0, 1, 0.6]]
    assert server._route_overlap_edges(hits, floor=0.1) == [[0, 1, 0.6], [0, 2, 1 / 7], [1, 2, 1 / 7]]


def test_overlap_at_exactly_the_floor_is_kept():
    # 2/4 = 0.5: the floor is inclusive, so a threshold of exactly 0.5 replays exactly.
    assert server._route_overlap_edges([{"text": "a b c"}, {"text": "a b d"}]) == [[0, 1, 0.5]]


def test_trace_overlap_lookup():
    fn = server._route_trace_overlap_fn(
        {"overlap_edges": [[0, 2, 0.9], ["bad"], [1, "x", 0.7], [3, 1, 0.6]]})
    a, b, c, d = ({"trace_idx": i} for i in range(4))
    assert fn(a, c) == 0.9 and fn(c, a) == 0.9  # symmetric
    assert fn(b, d) == 0.6 and fn(d, b) == 0.6  # stored high-low still found
    assert fn(a, b) == 0.0  # below the floor: not stored, so no overlap
    assert fn(a, {"text": "no index"}) is None  # not a sampled candidate: never dedups
    assert server._route_trace_overlap_fn({"overlap_edges": None}) is None
    assert server._route_trace_overlap_fn({}) is None
