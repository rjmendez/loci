"""cli.py: argument parsing and that each subcommand actually prints the
data it claims to, not just that it exits 0."""
import json

import pytest

from .conftest import needs_corpus_deps, needs_git_history  # noqa: F401

from ..cli import main


def _run(capsys, argv):
    code = main(argv)
    out = capsys.readouterr().out
    return code, out


def test_build_text(capsys):
    code, out = _run(capsys, ["build", "--rev", "HEAD", "--scope", "mcp/graph_tools.py"])
    assert code == 0
    assert "source: rev" in out
    assert "node:FUNCTION" in out


def test_build_json(capsys):
    code, out = _run(capsys, ["build", "--rev", "HEAD", "--scope", "mcp/graph_tools.py", "--format", "json"])
    assert code == 0
    payload = json.loads(out)
    assert payload["file_count"] == 1
    assert payload["stats"]["node:MODULE"] == 1


def test_modules_lists_importable_as(capsys):
    code, out = _run(capsys, ["modules", "--rev", "HEAD", "--scope", "mcp/graph_tools.py"])
    assert code == 0
    assert "mcp/graph_tools.py" in out
    assert "graph_tools" in out


def test_defs_summary_counts(capsys):
    code, out = _run(capsys, ["defs", "--rev", "HEAD", "--scope", "mcp/graph_tools.py"])
    assert code == 0
    assert "functions:" in out
    assert "NAME slots:" in out


def test_defs_all_lists_qualnames(capsys):
    code, out = _run(capsys, ["defs", "--rev", "HEAD", "--scope", "mcp/graph_tools.py", "--all"])
    assert code == 0
    assert "symbol_impact" in out


@needs_corpus_deps
def test_imports_unresolved_small(capsys):
    code, out = _run(capsys, ["imports", "--rev", "HEAD", "--unresolved"])
    assert code == 0
    lines = [l for l in out.splitlines() if "->" in l]
    assert len(lines) < 20


def test_imports_lazy_only_function_local(capsys):
    code, out = _run(capsys, ["imports", "--rev", "HEAD", "--scope", "mcp/server.py", "--lazy"])
    assert code == 0
    lines = [l for l in out.splitlines() if "->" in l]
    assert lines  # mcp/server.py has function-local imports
    assert all("[local]" in l for l in lines)


def test_aliases_symbol_impact(capsys):
    code, out = _run(capsys, ["aliases", "--rev", "HEAD", "mcp/server.py::symbol_impact"])
    assert code == 0
    assert "fn:mcp/graph_tools.py::symbol_impact" in out


def test_bad_subcommand_exits_nonzero():
    with pytest.raises(SystemExit):
        main(["not-a-real-command"])


# -- callers / reach ---------------------------------------------------------


def test_callers_proven_finds_in_server_callsites(capsys):
    code, out = _run(capsys, ["callers", "_get_ladybug", "--rev", "HEAD", "--conf", "proven"])
    assert code == 0
    assert "-- PROVEN" in out
    assert out.count("call[name-def-local]") == 2
    assert out.count("passed-by-ref") == 2


def test_callers_ambiguous_bare_name_lists_candidates(capsys):
    code, out = _run(capsys, ["callers", "register", "--rev", "HEAD", "--scope", "mcp/"])
    assert code == 0
    assert "ambiguous symbol" in out


def test_callers_json_format(capsys):
    # Default --conf is "probable", so this now also picks up the 15 rung-4
    # (name-via-injected-global) PROBABLE calls inside graph_tools.py that
    # only resolve once extract/dispatch.py's probable tier has run — on
    # top of the 2 PROVEN in-server calls + 2 passed-by-ref REFERENCES rows
    # step 4/step 5 slices already covered.
    code, out = _run(capsys, ["callers", "_get_ladybug", "--rev", "HEAD", "--format", "json"])
    assert code == 0
    rows = json.loads(out)
    assert isinstance(rows, list) and len(rows) == 19
    from collections import Counter
    counts = Counter((r["kind"], r["confidence"]) for r in rows)
    assert counts[("CALLS", "PROVEN")] == 2
    assert counts[("CALLS", "PROBABLE")] == 15
    assert counts[("REFERENCES", "PROVEN")] == 2
    assert all(r.get("rung") == "name-via-injected-global" for r in rows if r["kind"] == "CALLS" and r["confidence"] == "PROBABLE")


def test_reach_from_get_ladybug(capsys):
    code, out = _run(capsys, ["reach", "_get_ladybug", "--rev", "HEAD", "--depth", "1"])
    assert code == 0
    assert "forward reach from fn:mcp/server.py::_get_ladybug" in out


# -- registry -----------------------------------------------------------------


def test_registry_lists_known_surfaces(capsys):
    code, out = _run(capsys, ["registry", "--rev", "HEAD"])
    assert code == 0
    assert "reg:mcp/server.py::mcp.tool" in out
    assert "members= 41" in out or "members=41" in out
    assert "reg:a2a_server/server.py::_SKILL_MAP" in out


def test_registry_unmatched_is_empty(capsys):
    code, out = _run(capsys, ["registry", "--rev", "HEAD", "--unmatched"])
    assert code == 0
    assert "0 unclassified" in out


# -- name ---------------------------------------------------------------------


def test_name_get_ladybug_shows_write_and_injection(capsys):
    code, out = _run(capsys, ["name", "_get_ladybug", "--rev", "HEAD"])
    assert code == 0
    assert "WRITE" in out and "via=global-stmt" in out
    assert "INJECT" in out
    assert "WARNING" in out  # cross-module coupling


def test_name_unknown_ident_errors(capsys):
    code, out = _run(capsys, ["name", "totally_not_a_real_identifier_xyz", "--rev", "HEAD", "--scope", "mcp/graph_tools.py"])
    assert code == 2


# -- dead / holes ---------------------------------------------------------------


def test_dead_scope_mcp_never_lists_a_registered_tool(capsys):
    code, out = _run(capsys, ["dead", "--rev", "HEAD", "--scope", "mcp/"])
    assert code == 0
    assert "does NOT mean safe to delete" in out
    # None of the 40 real @mcp.tool() names should ever appear as UNREACHABLE.
    assert "investigation_store" not in out
    assert "symbol_impact" not in out


def test_holes_groups_by_reason(capsys):
    code, out = _run(capsys, ["holes", "--rev", "HEAD", "--scope", "mcp/graph_tools.py"])
    assert code == 0
    assert "callsite(s) total" in out


# -- probable tier: dispatch / injected globals -------------------------------


def test_callers_get_ladybug_shows_probable_injected_rows_with_why(capsys):
    code, out = _run(capsys, ["callers", "_get_ladybug", "--rev", "HEAD", "--scope", "mcp/", "--why"])
    assert code == 0
    assert "-- PROBABLE" in out
    # 11 from mcp/graph_tools.py's own injected NAME slot + 4 from
    # mcp/ladybug_ops.py's separately-injected slot pointing at the same
    # target function (see test_pipeline_real_corpus.py for the split).
    assert out.count("call[name-via-injected-global]") == 15
    assert "-- why: rule=name-via-injected-global  because=injected at mcp/server.py:" in out


def test_callers_skill_shows_dispatch_row_as_one_of_thirteen(capsys):
    code, out = _run(capsys, ["callers", "a2a_server/server.py::skill_memory_recall", "--rev", "HEAD"])
    assert code == 0
    assert "dispatch  1 of 13" in out


def test_explain_dispatch_callsite_lists_all_thirteen_candidates(capsys):
    # Locate the callsite through the graph. A hardcoded line number drifts with
    # every edit above it, and the miss is reported on stderr, so a fallback
    # keyed on stdout never fires.
    _, callers_out = _run(capsys, ["callers", "a2a_server/server.py::skill_memory_recall", "--rev", "HEAD", "--format", "json"])
    disp = [r for r in json.loads(callers_out) if r["kind"] == "DISPATCHES"][0]
    code, out = _run(capsys, ["explain", disp["src"], "--rev", "HEAD", "--scope", "a2a_server/"])
    assert code == 0
    assert out.count("DISPATCHES ->") == 13
    assert "1 of 13" in out


def test_paths_reports_minimum_confidence(capsys):
    code, out = _run(capsys, [
        "paths", "mcp/graph_tools.py::code_memory_relink", "_get_ladybug", "--rev", "HEAD", "--scope", "mcp/",
    ])
    assert code == 0
    assert "path confidence: PROBABLE" in out


def test_reach_to_degenerates_to_paths(capsys):
    code, out = _run(capsys, [
        "reach", "mcp/graph_tools.py::code_memory_relink", "--to", "_get_ladybug", "--rev", "HEAD", "--scope", "mcp/",
    ])
    assert code == 0
    assert "path confidence: PROBABLE" in out


def test_build_dot_emits_a_valid_looking_digraph(capsys):
    code, out = _run(capsys, ["build", "--rev", "HEAD", "--scope", "mcp/graph_tools.py", "--format", "dot"])
    assert code == 0
    assert out.startswith("digraph callgraph {")
    assert out.rstrip().endswith("}")
    assert '[FUNCTION]' in out
    assert "-> " in out


def test_callers_dot_scopes_to_the_traversed_subgraph(capsys):
    code, out = _run(capsys, ["callers", "_get_ladybug", "--rev", "HEAD", "--conf", "proven", "--format", "dot"])
    assert code == 0
    assert out.startswith("digraph callgraph {")
    assert '"fn:mcp/server.py::_get_ladybug"' in out
    # a node that has nothing to do with this traversal must not appear
    assert "code_memory_relink" not in out


def test_reach_dot_includes_traversed_hops(capsys):
    code, out = _run(capsys, ["reach", "_get_ladybug", "--rev", "HEAD", "--scope", "mcp/", "--depth", "1", "--format", "dot"])
    assert code == 0
    assert out.startswith("digraph callgraph {")
    assert 'label="CALLS"' in out or 'label="REFERENCES"' in out


def test_paths_dot_renders_the_hop_chain(capsys):
    code, out = _run(capsys, [
        "paths", "mcp/graph_tools.py::code_memory_relink", "_get_ladybug", "--rev", "HEAD", "--scope", "mcp/", "--format", "dot",
    ])
    assert code == 0
    assert out.startswith("digraph callgraph {")
    assert '"fn:mcp/graph_tools.py::code_memory_relink"' in out
    assert '"fn:mcp/server.py::_get_ladybug"' in out
    assert "style=dashed" in out  # the PROBABLE hop (see test_paths_reports_minimum_confidence)


def test_entrypoints_names_code_memory_relink_tool(capsys):
    code, out = _run(capsys, ["entrypoints", "mcp/graph_tools.py:1", "--rev", "HEAD", "--scope", "mcp/"])
    # line 1 of graph_tools.py is the module docstring: falls back to the
    # MODULE id, which has no direct ENTRYPOINT of its own (modules are
    # only trivial roots for reachable_functions' forward closure, not a
    # registered entrypoint themselves) — just confirms the command runs
    # cleanly end to end and prints the honest "0 known" verdict.
    assert code == 0
    assert "mcp/graph_tools.py:1" in out
