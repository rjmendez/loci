"""Integration tests against the real repo (read via --rev so a concurrent
edit to mcp/server.py can never make these flaky) — the actual acceptance
bar for steps 1-3, not just the synthetic fixtures.

Most of these share the `head_build` session fixture (see conftest.py) so
the corpus is only parsed once per test run; a few need a different
`--rev` or `--scope` and build independently."""
from .conftest import needs_corpus_deps, needs_git_history  # noqa: F401
from ..analyze.deadcode import registered_but_dead
from ..analyze.reach import (
    direct_callers, entrypoints_reaching, function_at_line, path_confidence, shortest_path,
)
from ..model import Confidence
from ..pipeline import build_graph


def test_build_is_clean_and_fast(head_build):
    assert head_build.meta.file_count == 122
    assert head_build.meta.error_count == 0
    # Loose sanity bound, not a benchmark: measured 4.2s standalone / 5.0s under suite load.
    assert head_build.meta.elapsed_s < 30, (
        f"cold build took {head_build.meta.elapsed_s:.1f}s -- that is a pathological "
        "regression, not load variance"
    )


def test_module_level_function_count_matches_census_within_tolerance(head_build):
    module_level = [
        n for n in head_build.store.nodes_of_kind("FUNCTION")
        if not n.attrs["is_nested"] and not n.attrs["is_method"]
    ]
    # Band is the 1002 measured count +-5%; the LOCI_* rename and the grounding
    # helpers moved it from ~951.
    assert 950 <= len(module_level) <= 1055, len(module_level)


def test_mcp_top_level_module_level_function_count(head_build):
    module_level = [
        n for n in head_build.store.nodes_of_kind("FUNCTION")
        if n.path is not None and n.path.startswith("mcp/") and n.path.count("/") == 1
        and not n.attrs["is_nested"] and not n.attrs["is_method"]
    ]
    # docs/census.txt estimate was 297; mcp/openrouter.py moved it to ~334.
    assert 300 <= len(module_level) <= 370, len(module_level)


def test_every_mcp_tool_decorator_is_classified_registering(head_build):
    registering = [
        e for e in head_build.store.edges_of_kind("DECORATED_BY")
        if e.attrs["classification"] == "registering" and e.attrs["raw"].startswith("mcp.tool")
        and e.src.startswith("fn:mcp/server.py::")
    ]
    assert len(registering) == 42


def test_no_unknown_decorators_in_real_corpus(head_build):
    unknown = [
        e for e in head_build.store.edges_of_kind("DECORATED_BY") if e.attrs["classification"] == "unknown"
    ]
    assert unknown == [], [(e.src, e.attrs["raw"]) for e in unknown]


def test_symbol_impact_reexport_alias_resolves_to_graph_tools(head_build):
    aliases = head_build.store.out_edges("name:mcp/server.py::symbol_impact", "ALIASES")
    assert len(aliases) == 1
    assert aliases[0].dst == "fn:mcp/graph_tools.py::symbol_impact"
    assert aliases[0].confidence.name == "PROVEN"


def test_callers_of_symbol_impact_do_not_double_count_the_alias(head_build):
    # A re-export is the SAME function node reached two ways, never a second function.
    fn_id = "fn:mcp/graph_tools.py::symbol_impact"
    definers = list(head_build.store.in_edges(fn_id, "DEFINES"))
    assert len(definers) == 1
    assert definers[0].src == "mod:mcp/graph_tools.py"
    aliasers = list(head_build.store.in_edges(fn_id, "ALIASES"))
    assert len(aliasers) == 1
    assert aliasers[0].src == "name:mcp/server.py::symbol_impact"


@needs_git_history
def test_bug_c_dangling_global_regression_before_the_fix():
    # c1c40a9~1 is the last rev where graph_tools.py declares these `global` with no local binding.
    result = build_graph(rev="c1c40a9~1", scope_prefixes=["mcp/graph_tools.py"])
    names = result.store.nodes_of_kind("NAME")
    dangling = {n.id.split("::")[-1] for n in names if "global-only" in n.attrs["binding_kind"]}
    assert {"_symbol_index_cache", "_symbol_index_count"} <= dangling


def test_bug_c_is_fixed_on_head(head_build):
    names = [n for n in head_build.store.nodes_of_kind("NAME") if n.path == "mcp/graph_tools.py"]
    dangling = {n.id.split("::")[-1] for n in names if "global-only" in n.attrs["binding_kind"]}
    assert "_symbol_index_cache" not in dangling
    assert "_symbol_index_count" not in dangling


@needs_corpus_deps
def test_unresolved_imports_are_all_genuinely_optional_third_party(head_build):
    unresolved = [e for e in head_build.store.edges_of_kind("IMPORTS") if e.attrs.get("resolved_via") == "unresolved"]
    modules = {e.attrs["module"] for e in unresolved}
    # All optional deps absent from the venv; a corpus-internal name appearing here IS a regression.
    assert modules <= {
        "cy_ioc_extract", "mnemosyne", "mnemosyne.core.memory", "mnemosyne.core.beam",
        "psycopg2", "psutil",
    }
    assert len(unresolved) < 20, "the unresolved list must fit on one screen"


# ---------------------------------------------------------------------------
# Step 4: CALLSITE / CALLS (rungs 1-3, proven tier)
# ---------------------------------------------------------------------------


def test_callsite_count_is_in_the_expected_ballpark(head_build):
    """Coarse sanity band: catch a pipeline that counts nothing or everything.

    docs/census.txt estimated ~11,600 (5,106 by name + 6,538 by attribute) when
    this was written. The band was 11,000-12,500 and the repo grew through the
    ceiling — 12,514 — which failed CI for adding code, not for miscounting.

    This tracks REPO SIZE, so it is deliberately loose. It exists to catch the
    pipeline returning 0, or an order of magnitude, not to pin a number that
    every feature branch moves. Re-widen it rather than trimming code to fit.
    """
    count = sum(1 for _ in head_build.store.nodes_of_kind("CALLSITE"))
    assert 8000 <= count <= 20000, count


def test_every_callsite_has_exactly_one_calls_edge(head_build):
    store = head_build.store
    for n in store.nodes_of_kind("CALLSITE"):
        edges = store.out_edges(n.id, "CALLS")
        assert len(edges) == 1, (n.id, len(edges))


def test_unresolved_callsites_go_to_the_sink_not_dropped(head_build):
    store = head_build.store
    total = sum(1 for _ in store.nodes_of_kind("CALLSITE"))
    unresolved = len(store.in_edges("?", "CALLS"))
    assert 0 < unresolved < total
    for e in store.in_edges("?", "CALLS"):
        assert "reason" in e.attrs


def test_callers_get_ladybug_proven_finds_the_in_server_callsites(head_build):
    # Proven tier sees only the direct in-server calls; the 15 injected-global ones are rung 4.
    store = head_build.store
    rows = direct_callers(store, "fn:mcp/server.py::_get_ladybug")
    calls = [r for r in rows if r.kind == "CALLS" and r.edge.confidence == Confidence.PROVEN]
    call_locs = {(r.site_node.line, r.site_node.col) for r in calls}
    assert len(calls) == 2, call_locs
    assert all(loc[0] > 0 for loc in call_locs)  # both callsites resolved to real lines in server.py
    refs = [r for r in rows if r.kind == "REFERENCES"]
    assert len(refs) == 2  # passed by reference at server.py:365 and the graph_tools.register() call


# ---------------------------------------------------------------------------
# Step 5: REGISTRY / REGISTERS / ENTERS / ENTRYPOINT — the hard gate
# ---------------------------------------------------------------------------


def test_hard_gate_no_registered_function_is_reported_dead(head_build):
    # The hard gate: a registered function must never appear in the dead list.
    bad = registered_but_dead(head_build.store)
    assert bad == [], [n.id for n in bad]


def test_registry_counts_match_the_real_corpus(head_build):
    from collections import Counter
    store = head_build.store
    by_rule = Counter(e.attrs["rule"] for e in store.edges_of_kind("REGISTERS"))
    # 42 @mcp.tool() + 1 @mcp.resource(): both make a function externally callable, so both are DEC-tool.
    assert by_rule["DEC-tool"] == 43
    assert by_rule["DEC-route"] == 6         # a2a_server's @app.get/@app.post
    assert by_rule["DEC-mcp-route"] == 1     # mcp/server.py's @mcp.custom_route("/health", ...)
    assert by_rule["MAN-LOOP"] == 31         # graph_tools(11) + investigation_tools(11) + llm_tools(9)
    assert by_rule["MAN-DICT"] == 13         # a2a_server's _SKILL_MAP


def test_registry_unmatched_is_empty(head_build):
    unmatched = [e for e in head_build.store.edges_of_kind("DECORATED_BY") if e.attrs["classification"] == "unknown"]
    assert unmatched == []


def test_decorated_by_retargeted_from_external_to_registry(head_build):
    # registry.py must retarget DECORATED_BY off defs.py's placeholder EXTERNAL sink.
    store = head_build.store
    _ = "fn:mcp/graph_tools.py::code_graph_ingest"
    # code_graph_ingest is MAN-LOOP registered, not decorated; find a decorated one.
    dec_fid = None
    for e in store.edges_of_kind("DECORATED_BY"):
        if e.attrs["raw"].startswith("mcp.tool") and e.src.startswith("fn:mcp/server.py::"):
            dec_fid = e.src
            edge = e
            break
    assert dec_fid is not None
    assert edge.dst == "reg:mcp/server.py::mcp.tool"
    assert store.get(edge.dst).kind == "REGISTRY"


def test_root_cli_entrypoints_cover_most_main_guard_modules(head_build):
    store = head_build.store
    modules_with_main = [n for n in store.nodes_of_kind("MODULE") if n.attrs.get("has_main")]
    cli_entries = [n for n in store.nodes_of_kind("ENTRYPOINT") if n.attrs["kind"] == "cli"]
    assert len(cli_entries) == len(modules_with_main)
    with_target = [n for n in cli_entries if store.out_edges(n.id, "ENTERS")]
    # argparse sub-dispatch guards do not resolve to a single main(); most others do.
    assert len(with_target) / len(cli_entries) > 0.5


# ---------------------------------------------------------------------------
# Step 6: READS_NAME / WRITES_NAME / INJECTS
# ---------------------------------------------------------------------------


def test_name_get_ladybug_write_and_injection_and_reads(head_build):
    store = head_build.store
    nid = "name:mcp/graph_tools.py::_get_ladybug"
    writes = store.in_edges(nid, "WRITES_NAME")
    global_stmt_writes = [w for w in writes if w.attrs["via"] == "global-stmt"]
    assert len(global_stmt_writes) == 1
    assert global_stmt_writes[0].src == "fn:mcp/graph_tools.py::register"
    assert global_stmt_writes[0].attrs["line"] == 396

    injects = store.in_edges(nid, "INJECTS")
    assert len(injects) == 1
    assert injects[0].attrs["value"] == "fn:mcp/server.py::_get_ladybug"
    assert injects[0].confidence == Confidence.PROBABLE

    reads = store.in_edges(nid, "READS_NAME")
    assert len(reads) == 11
    assert all(r.attrs["in_call_position"] for r in reads)


def test_investigation_tools_deps_dict_four_keys_injected(head_build):
    store = head_build.store
    expected = {"_apply_lifecycle", "_compute_self_check", "_event_log_append", "_qdrant_upsert"}
    found = set()
    for ident in expected:
        nid = f"name:mcp/investigation_tools.py::{ident}"
        injects = store.in_edges(nid, "INJECTS")
        assert len(injects) == 1, ident
        assert injects[0].attrs["param"] == "deps"
        assert injects[0].attrs["key"] == ident
        found.add(ident)
    assert found == expected


def test_qdrant_upsert_injection_resolves_through_the_reexport_alias(head_build):
    # A re-exported injected value must resolve to the real function, not a bare NAME.
    store = head_build.store
    nid = "name:mcp/investigation_tools.py::_qdrant_upsert"
    e = store.in_edges(nid, "INJECTS")[0]
    assert e.attrs["value"] == "fn:mcp/qdrant_ops.py::_qdrant_upsert"
    assert e.attrs["value_kind"] == "FUNCTION"


# ---------------------------------------------------------------------------
# Step 7: the probable tier (CALLS rungs 4-6) + DISPATCHES
# ---------------------------------------------------------------------------


def test_get_ladybug_inside_graph_tools_resolves_probable_via_injected_global(head_build):
    # The expected line is read off the real INJECTS edge, never hardcoded.
    store = head_build.store
    inject_edge = store.in_edges("name:mcp/graph_tools.py::_get_ladybug", "INJECTS")[0]
    inject_site = store.get(inject_edge.src)
    expected_because = f"injected at {inject_site.path}:{inject_site.line}"

    rows = direct_callers(store, "fn:mcp/server.py::_get_ladybug")
    probable_calls = [r for r in rows if r.kind == "CALLS" and r.edge.confidence == Confidence.PROBABLE]
    graph_tools_calls = [r for r in probable_calls if r.site_node is not None and r.site_node.path == "mcp/graph_tools.py"]
    # All 11 in_call_position READS_NAME edges on this slot must resolve through rung 4.
    assert len(graph_tools_calls) == 11
    assert all(r.edge.attrs["rung"] == "name-via-injected-global" for r in graph_tools_calls)
    assert all(r.edge.attrs["because"] == expected_because for r in graph_tools_calls)


def test_ladybug_ops_get_ladybug_calls_also_resolve_probable(head_build):
    # A second slot injected at a different callsite: rung 4 is not special-cased to one module.
    store = head_build.store
    rows = direct_callers(store, "fn:mcp/server.py::_get_ladybug")
    ladybug_ops_calls = [
        r for r in rows if r.kind == "CALLS" and r.edge.confidence == Confidence.PROBABLE
        and r.site_node is not None and r.site_node.path == "mcp/ladybug_ops.py"
    ]
    assert len(ladybug_ops_calls) == 4
    assert all(r.edge.attrs["rung"] == "name-via-injected-global" for r in ladybug_ops_calls)


def test_a2a_dispatch_fans_out_to_all_thirteen_skills(head_build):
    store = head_build.store
    site = next(
        (n for n in store.nodes_of_kind("CALLSITE")
         if n.path == "a2a_server/server.py" and n.attrs.get("form") == "name" and n.attrs.get("callee") == "handler"),
        None,
    )
    assert site is not None, "expected a2a_server's `handler(task)` dispatch callsite"
    call_edge = store.out_edges(site.id, "CALLS")[0]
    assert call_edge.dst == "?"
    assert call_edge.attrs["reason"] == "dict-dispatch-fanout"
    assert call_edge.attrs["fanout"] == 13
    assert call_edge.attrs["registry"] == "reg:a2a_server/server.py::_SKILL_MAP"

    disp = store.out_edges(site.id, "DISPATCHES")
    assert len(disp) == 13
    assert all(e.confidence == Confidence.PROBABLE for e in disp)
    assert all(e.attrs["fanout"] == 13 for e in disp)
    assert all(e.attrs["registry"] == "reg:a2a_server/server.py::_SKILL_MAP" for e in disp)
    targets = {e.dst for e in disp}
    registry_members = {e.dst for e in store.out_edges("reg:a2a_server/server.py::_SKILL_MAP", "REGISTERS")}
    assert targets == registry_members
    assert len(registry_members) == 13


def test_dispatches_edges_participate_in_backward_closure(head_build):
    # The backward closure spans DISPATCHES too, not just the proven in-corpus rows.
    store = head_build.store
    rows = direct_callers(store, "fn:a2a_server/server.py::skill_memory_recall")
    disp_rows = [r for r in rows if r.kind == "DISPATCHES"]
    assert len(disp_rows) == 1
    assert disp_rows[0].edge.attrs["selector"] == "skill_id"
    assert disp_rows[0].site_node is not None and disp_rows[0].site_node.path == "a2a_server/server.py"


def test_hard_gate_still_holds_after_the_probable_tier(head_build):
    # Retargeting `?` edges must never demote an already-resolved rung-1-3 edge.
    bad = registered_but_dead(head_build.store)
    assert bad == [], [n.id for n in bad]


# ---------------------------------------------------------------------------
# Step 9: traversal commands — cg paths / cg entrypoints
# ---------------------------------------------------------------------------


def test_entrypoints_names_the_mcp_tool_reaching_code_memory_relink(head_build):
    # The line is computed from the function's own span, never hardcoded.
    store = head_build.store
    fn = store.get("fn:mcp/graph_tools.py::code_memory_relink")
    assert fn is not None
    mid_line = (fn.line + fn.attrs["end_lineno"]) // 2
    start_id = function_at_line(store, "mcp/graph_tools.py", mid_line)
    assert start_id == fn.id

    entries = entrypoints_reaching(store, start_id)
    assert entries.get("entry:mcp-tool:code_memory_relink") == Confidence.PROVEN


def test_paths_from_code_memory_relink_to_get_ladybug_is_probable(head_build):
    # One PROBABLE rung-4 hop makes the whole path PROBABLE: confidence is min(), not average.
    store = head_build.store
    hops = shortest_path(
        store, "fn:mcp/graph_tools.py::code_memory_relink", "fn:mcp/server.py::_get_ladybug",
        conf_floor=Confidence.UNPROVEN,
    )
    assert hops is not None and len(hops) == 1
    assert hops[0].edge.confidence == Confidence.PROBABLE
    assert path_confidence(hops) == Confidence.PROBABLE
