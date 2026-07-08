"""Tests for grounding.py — the deterministic retrieval/assembly logic."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import grounding as G  # noqa: E402 — must follow the path setup above


def test_ground_fail_open_on_raising_source(monkeypatch):
    # A raising server tool (or malformed finding) must NOT propagate out of ground() —
    # the fail-open discipline the flagship dogfood found violated in sections 1 & 3.
    import types
    fake = types.ModuleType("server")

    def _boom(*a, **k):
        raise RuntimeError("dead DB")

    fake.investigation_load = _boom
    fake.investigation_entity_lookup = _boom
    fake.rag_context_search = _boom
    fake.investigation_search = _boom
    monkeypatch.setitem(sys.modules, "server", fake)
    r = G.ground({"title": "x", "focus": "y", "caseIds": ["c1"], "entities": ["1.2.3.4"]},
                 {"budgetChars": 500, "memoryDir": "/nonexistent", "allowKeyword": True})
    assert set(r) == {"block", "sources", "chars", "degraded"}   # well-formed, never raised


def test_ground_skips_malformed_findings(monkeypatch):
    # recent_findings with a non-dict element must be skipped, not raise AttributeError.
    import types
    fake = types.ModuleType("server")
    fake.investigation_load = lambda cid, **k: {
        "manifest": {"hypothesis": "h"}, "recent_findings": ["not-a-dict", {"text": "ok"}]}
    monkeypatch.setitem(sys.modules, "server", fake)
    r = G.ground({"title": "x", "caseIds": ["c1"]}, {"budgetChars": 500, "memoryDir": "/nonexistent"})
    assert "ok" in r["block"] and set(r) == {"block", "sources", "chars", "degraded"}


def test_filter_noise_drops_conversation_dumps():
    items = [
        {"text": "genuine finding: the flock patch prevents interleaved appends", "source": "graph-tools"},
        {"text": '[{"role": "user", "content": "do X"}, {"role": "assistant", ...}]' + " x" * 900,
         "source": "pre_compress"},
        {"text": "user: hi\nassistant: hello\n" * 60, "source": "turn"},
    ]
    kept = G.filter_noise(items)
    assert len(kept) == 1
    assert "flock patch" in kept[0]["text"]


def test_filter_noise_dedupes():
    items = [{"text": "same thing here", "source": "a"}, {"text": "same thing here", "source": "b"}]
    assert len(G.filter_noise(items)) == 1


def test_truncate_respects_budget_and_boundary():
    t = G._truncate("one sentence. two sentence. three sentence.", 20)
    assert len(t) <= 20 + len(" …[truncated]")
    assert t.endswith("…[truncated]")
    assert G._truncate("short", 100) == "short"


def test_select_memory_files_relevance_and_cap(tmp_path):
    (tmp_path / "MEMORY.md").write_text(
        "- [relevant](relevant.md) — dispatch-aware referenced flag dead-code detection registry rules\n"
        "- [irrelevant](irrelevant.md) — opnsense firewall perimeter routing shaper dhcp audit\n"
        "- [weak](weak.md) — dispatch something unrelated entirely\n"
    )
    (tmp_path / "relevant.md").write_text("BODY: the referenced flag powers dispatch-aware dead-code.")
    (tmp_path / "irrelevant.md").write_text("BODY: firewall stuff.")
    task = {"title": "dead-code detection", "focus": "dispatch-aware referenced flag registry rules"}
    picks = G._select_memory_files(task, str(tmp_path), limit=2, min_score=2)
    names = {p[0] for p in picks}
    assert "relevant.md" in names            # strong distinctive-token match
    assert "irrelevant.md" not in names      # no distinctive overlap
    assert len(picks) <= 2                    # capped
    assert any("BODY:" in p[2] for p in picks)  # body read


def test_select_memory_files_rejects_generic_token_match(tmp_path):
    # Two incidental English words ("event", "single") must NOT pull an off-topic memory;
    # a candidate needs a distinctive (len>=7) shared token, not just min_score generic ones.
    (tmp_path / "MEMORY.md").write_text(
        "- [offtopic](offtopic.md) — resilient rover single NEAREST-base early-Aug event mock-GPS\n")
    (tmp_path / "offtopic.md").write_text("BODY: rover stuff")
    task = {"title": "decompose investigation_store",
            "focus": "single-responsibility units, event-log, conflict-detect"}
    assert G._select_memory_files(task, str(tmp_path), min_score=2) == []


def test_select_memory_files_missing_index(tmp_path):
    assert G._select_memory_files({"title": "x"}, str(tmp_path)) == []


def test_ground_fail_open_no_server(monkeypatch):
    # No caseIds/entities and unreachable server -> well-formed, non-crashing result.
    r = G.ground({"title": "nothing", "focus": "nothing"}, {"budgetChars": 500, "memoryDir": "/nonexistent"})
    assert set(r) == {"block", "sources", "chars", "degraded"}
    assert r["chars"] <= 500 + 400  # header/footer overhead bounded
    assert isinstance(r["sources"], list)


def test_ground_emits_exclusion_block_for_resolved_findings(monkeypatch):
    # Findings marked fixed/intentional/wontfix on a grounded case must surface as a
    # compact "do NOT re-report" block so re-audits auto-exclude handled items.
    import types
    fake = types.ModuleType("server")
    fake.investigation_load = lambda cid, **k: {
        "manifest": {"hypothesis": "h", "next_step": "n"},
        "recent_findings": [
            {"text": "null-deref in parse()", "resolution": "fixed"},
            {"text": "wide CORS is by design", "resolution": "intentional"},
            {"text": "still an open bug in retry loop", "resolution": "open"},
            {"text": "legacy flag, leave it", "resolution": "wontfix"},
        ],
    }
    monkeypatch.setitem(sys.modules, "server", fake)
    r = G.ground({"title": "re-audit", "caseIds": ["c1"]},
                 {"budgetChars": 4000, "memoryDir": "/nonexistent"})
    assert "known — do NOT re-report" in r["block"]
    # Isolate the exclusion-block line and assert it lists the resolved findings only.
    known_line = next(ln for ln in r["block"].splitlines() if "known — do NOT re-report" in ln)
    assert "null-deref in parse()" in known_line
    assert "wide CORS is by design" in known_line
    assert "legacy flag, leave it" in known_line
    # An open finding must NOT be excluded (it stays re-reportable).
    assert "still an open bug" not in known_line


def test_ground_omits_exclusion_block_when_nothing_resolved(monkeypatch):
    # No resolved findings (all open / absent field) -> the block is omitted entirely.
    import types
    fake = types.ModuleType("server")
    fake.investigation_load = lambda cid, **k: {
        "manifest": {"hypothesis": "h"},
        "recent_findings": [
            {"text": "an open finding"},                       # no resolution -> open
            {"text": "another active item", "resolution": "open"},
        ],
    }
    monkeypatch.setitem(sys.modules, "server", fake)
    r = G.ground({"title": "re-audit", "caseIds": ["c1"]},
                 {"budgetChars": 4000, "memoryDir": "/nonexistent"})
    assert "known — do NOT re-report" not in r["block"]
    assert set(r) == {"block", "sources", "chars", "degraded"}


def test_ground_budget_respected(tmp_path, monkeypatch):
    (tmp_path / "MEMORY.md").write_text(
        "- [a](a.md) — dispatch referenced flag dead-code registry rules detection\n")
    (tmp_path / "a.md").write_text("X" * 20000)  # oversized body must be truncated
    r = G.ground({"title": "dead-code detection", "focus": "dispatch referenced flag registry rules"},
                 {"budgetChars": 800, "memoryDir": str(tmp_path)})
    # block stays within budget + bounded header/footer overhead
    assert r["chars"] <= 800 + 500
