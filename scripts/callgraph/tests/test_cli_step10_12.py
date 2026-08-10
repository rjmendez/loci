from .conftest import needs_corpus_deps, needs_git_history  # noqa: F401
"""CLI end-to-end tests for `cg writes-dead` / `cg literals` / `cg flags`
against the REAL corpus at the specific revisions BUG B/C/D were introduced
and fixed at — the regression tests the design's acceptance criteria ask
for, exercised through the actual command line, not just the underlying
analyze/* functions."""
import json

from ..cli import main


def _run(capsys, argv):
    code = main(argv)
    out = capsys.readouterr().out
    return code, out


# -- BUG C: writes-dead / dangling-global -------------------------------------


@needs_git_history
def test_writes_dead_finds_bug_c_at_broken_revision(capsys):
    code, out = _run(capsys, ["writes-dead", "--rev", "c1c40a9^", "--scope", "mcp/"])
    assert code == 0
    assert "DANGLING-GLOBAL" in out
    assert "_symbol_index_cache" in out
    assert "_symbol_index_count" in out
    assert "mcp/graph_tools.py" in out
    assert "mcp/ladybug_ops.py:106" in out


def test_writes_dead_clean_at_head(capsys):
    code, out = _run(capsys, ["writes-dead", "--rev", "HEAD", "--scope", "mcp/", "--format", "json"])
    assert code == 0
    payload = json.loads(out)
    slots = [f["slot"] for f in payload["dangling_global"]]
    assert not any("_symbol_index_cache" in s or "_symbol_index_count" in s for s in slots)


def test_writes_dead_include_tests_shrinks_weak_list(capsys):
    code, out_default = _run(capsys, ["writes-dead", "--rev", "HEAD", "--format", "json"])
    assert code == 0
    default_count = len(json.loads(out_default)["write_no_read"])

    code, out_tests = _run(capsys, ["writes-dead", "--rev", "HEAD", "--include-tests", "--format", "json"])
    assert code == 0
    payload = json.loads(out_tests)
    assert len(payload["write_no_read"]) <= default_count
    assert payload["include_tests"] is True


# -- BUG D: literals -----------------------------------------------------------


@needs_git_history
def test_literals_orphans_finds_bug_d_at_broken_revision(capsys):
    code, out = _run(capsys, ["literals", "--paths", "--orphans", "--rev", "08c9198^"])
    assert code == 0
    assert "graph.ladybug" in out
    assert "mcp/server.py:261" in out
    assert "CONSUMED BY NOBODY" in out
    assert "graph.kuzu" in out
    assert "scripts/graph_facts.py:29" in out
    assert "PRODUCED BY NOBODY" in out


@needs_git_history
def test_literals_near_miss_pairs_the_bug_d_orphans(capsys):
    code, out = _run(capsys, ["literals", "graph", "--paths", "--near-miss", "--rev", "08c9198^"])
    assert code == 0
    assert "'graph.ladybug'" in out
    assert "'graph.kuzu'" in out
    assert "stem=graph" in out
    assert "lead, not finding" in out


def test_literals_matched_at_head(capsys):
    code, out = _run(capsys, ["literals", "graph", "--paths", "--rev", "HEAD", "--format", "json"])
    assert code == 0
    rows = json.loads(out)
    matches = [r for r in rows if r["text"] == "graph.ladybug"]
    assert len(matches) == 1
    assert matches[0]["produced_by"]
    assert matches[0]["consumed_by"]


def test_literals_orphans_empty_at_head_for_graph_ladybug(capsys):
    code, out = _run(capsys, ["literals", "graph", "--paths", "--orphans", "--rev", "HEAD", "--format", "json"])
    assert code == 0
    payload = json.loads(out)
    texts = {g["text"] for g in payload["producer_only"]} | {g["text"] for g in payload["consumer_only"]}
    assert "graph.ladybug" not in texts
    assert "graph.kuzu" not in texts


# -- BUG B: flags ---------------------------------------------------------------


@needs_git_history
def test_flags_ranks_degraded_first_at_broken_revision(capsys):
    code, out = _run(capsys, ["flags", "mcp/grounding.py::ground", "--rev", "69adfa4^"])
    assert code == 0
    assert "ground::degraded" in out
    assert "init=" in out and "(constant)" in out
    assert "escapes as dict-value" in out
    assert out.count("guard exit") >= 4
    assert "does NOT assign degraded" in out


@needs_git_history
def test_flags_json_row_shape(capsys):
    code, out = _run(capsys, ["flags", "mcp/grounding.py::ground", "--rev", "69adfa4^", "--format", "json"])
    assert code == 0
    rows = json.loads(out)
    assert len(rows) == 1
    assert rows[0]["ident"] == "degraded"
    assert rows[0]["escape_form"] == "dict-value"
    assert rows[0]["score"] > 1.0
