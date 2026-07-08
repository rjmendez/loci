"""Tests for grounding.py — the deterministic retrieval/assembly logic."""
import grounding as G


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


def test_select_memory_files_missing_index(tmp_path):
    assert G._select_memory_files({"title": "x"}, str(tmp_path)) == []


def test_ground_fail_open_no_server(monkeypatch):
    # No caseIds/entities and unreachable server -> well-formed, non-crashing result.
    r = G.ground({"title": "nothing", "focus": "nothing"}, {"budgetChars": 500, "memoryDir": "/nonexistent"})
    assert set(r) == {"block", "sources", "chars", "degraded"}
    assert r["chars"] <= 500 + 400  # header/footer overhead bounded
    assert isinstance(r["sources"], list)


def test_ground_budget_respected(tmp_path, monkeypatch):
    (tmp_path / "MEMORY.md").write_text(
        "- [a](a.md) — dispatch referenced flag dead-code registry rules detection\n")
    (tmp_path / "a.md").write_text("X" * 20000)  # oversized body must be truncated
    r = G.ground({"title": "dead-code detection", "focus": "dispatch referenced flag registry rules"},
                 {"budgetChars": 800, "memoryDir": str(tmp_path)})
    # block stays within budget + bounded header/footer overhead
    assert r["chars"] <= 800 + 500
