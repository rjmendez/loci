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
    assert head_build.meta.file_count == 118
    assert head_build.meta.error_count == 0
    # The 2s figure in ingest.py's own docstring is about ast.parse'ing the
    # corpus (steps 1-3's budget). Steps 4-6 (CALLSITE/CALLS resolution over
    # ~11,600 callsites, the whole-module READS_NAME/WRITES_NAME walk, and
    # registry/injection extraction) are real additional work layered on
    # top of that parse; the design's own ceiling for the FULLY assembled
    # tool (all 13 build steps) is "under 5s" for a cold build plus every
    # fixture assertion (see build_steps step 13) — this asserts against
    # that end-state budget with headroom, not the steps-1-3-only figure.
    # Loose sanity bound, NOT a benchmark. Measured cold build is ~4.2s standalone
    # but ~5.0s when this test runs alongside the other 248 under load, so a tight
    # budget here is flaky by construction -- it failed on merge for exactly that
    # reason. A flaky test trains people to ignore failures, which costs more than
    # the regression it was meant to catch. This bound only trips on a pathological
    # blow-up (an accidental O(n^2) pass, or re-parsing per query); track real
    # performance with a benchmark, not a unit test.
    assert head_build.meta.elapsed_s < 30, (
        f"cold build took {head_build.meta.elapsed_s:.1f}s -- that is a pathological "
        "regression, not load variance"
    )


def test_module_level_function_count_matches_census_within_tolerance(head_build):
    module_level = [
        n for n in head_build.store.nodes_of_kind("FUNCTION")
        if not n.attrs["is_nested"] and not n.attrs["is_method"]
    ]
    # docs/census.txt estimate was ~890 corpus-wide; scripts/loci_groom.py and
    # mcp/openrouter.py moved it to ~951. Band re-centred, same +-5% tolerance.
    assert 900 <= len(module_level) <= 1000, len(module_level)


def test_mcp_top_level_module_level_function_count(head_build):
    # "mcp/ top-level" means files directly under mcp/ (server.py,
    # graph_tools.py, ...) as opposed to the mcp/graph/ and mcp/memcheck/
    # subpackages.
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
    assert len(registering) == 41


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
    # `cg callers symbol_impact` must not treat server.py's re-export as a
    # second, independent function — it is the SAME function node reached
    # two ways (DEFINES from graph_tools.py, ALIASES from server.py).
    fn_id = "fn:mcp/graph_tools.py::symbol_impact"
    definers = list(head_build.store.in_edges(fn_id, "DEFINES"))
    assert len(definers) == 1
    assert definers[0].src == "mod:mcp/graph_tools.py"
    aliasers = list(head_build.store.in_edges(fn_id, "ALIASES"))
    assert len(aliasers) == 1
    assert aliasers[0].src == "name:mcp/server.py::symbol_impact"


@needs_git_history
def test_bug_c_dangling_global_regression_before_the_fix():
    # At c1c40a9^ (the commit before "make code_memory_relink actually
    # invalidate the symbol-index cache"), graph_tools.py declares these
    # `global` with no module-level binding in that file — the real slot
    # lives in ladybug_ops.py:106-107 (a different module entirely). This
    # needs its own build: a different --rev than every other test here.
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
    # Every one of these is a real optional dependency guarded by
    # try/except ImportError or a lazy import, not present in the venv —
    # not a resolver bug. If this set grows to include a corpus-internal
    # module name, that IS a resolver regression.
    assert modules <= {
        "cy_ioc_extract", "mnemosyne", "mnemosyne.core.memory", "mnemosyne.core.beam",
        "psycopg2", "psutil",
    }
    assert len(unresolved) < 20, "the unresolved list must fit on one screen"


# ---------------------------------------------------------------------------
# Step 4: CALLSITE / CALLS (rungs 1-3, proven tier)
# ---------------------------------------------------------------------------


def test_callsite_count_is_in_the_expected_ballpark(head_build):
    # docs/census.txt estimate: ~11,600 (5,106 by name + 6,538 by attribute).
    count = sum(1 for _ in head_build.store.nodes_of_kind("CALLSITE"))
    assert 11000 <= count <= 12500, count


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
    # THE acceptance line for step 4: `cg callers _get_ladybug --conf
    # proven` must find the direct in-server calls. The 15 calls inside
    # graph_tools.py that only resolve through the injected global are
    # rung-4 (probable) territory — a later slice's job — and correctly do
    # NOT show up here.
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
    # "If any of the registered functions appears in the dead list, stop
    # and fix before writing another module" — the literal gate from
    # build_steps step 5, run against the WHOLE corpus.
    bad = registered_but_dead(head_build.store)
    assert bad == [], [n.id for n in bad]


def test_registry_counts_match_the_real_corpus(head_build):
    from collections import Counter
    store = head_build.store
    by_rule = Counter(e.attrs["rule"] for e in store.edges_of_kind("REGISTERS"))
    # 41 @mcp.tool() + 1 @mcp.resource(), both DEC-tool (both make a
    # function externally callable by its own Python name — the specific
    # decorator classification, tool vs resource, isn't the resolution's
    # concern here). Verified independently via `git grep '^@mcp\.tool'`.
    assert by_rule["DEC-tool"] == 42
    assert by_rule["DEC-route"] == 6         # a2a_server's @app.get/@app.post
    assert by_rule["DEC-mcp-route"] == 1     # mcp/server.py's @mcp.custom_route("/health", ...)
    assert by_rule["MAN-LOOP"] == 31         # graph_tools(11) + investigation_tools(11) + llm_tools(9)
    assert by_rule["MAN-DICT"] == 13         # a2a_server's _SKILL_MAP


def test_registry_unmatched_is_empty(head_build):
    unmatched = [e for e in head_build.store.edges_of_kind("DECORATED_BY") if e.attrs["classification"] == "unknown"]
    assert unmatched == []


def test_decorated_by_retargeted_from_external_to_registry(head_build):
    # The placeholder EXTERNAL sink defs.py creates for every decorator
    # must be gone from a REGISTERING function's DECORATED_BY edge once
    # registry.py has run — it now points at the REGISTRY node instead.
    store = head_build.store
    _ = "fn:mcp/graph_tools.py::code_graph_ingest"
    # code_graph_ingest is MAN-LOOP registered (not decorated) — check an
    # actually-decorated one instead:
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
    # Not every __main__ guard's first call resolves (some drive argparse
    # sub-dispatch instead of a single main()); most should, though.
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
    # _qdrant_upsert isn't DEFINED in server.py (it's re-exported from
    # qdrant_ops.py) — the injected value must still resolve to the real
    # function, not degrade to a bare NAME reference.
    store = head_build.store
    nid = "name:mcp/investigation_tools.py::_qdrant_upsert"
    e = store.in_edges(nid, "INJECTS")[0]
    assert e.attrs["value"] == "fn:mcp/qdrant_ops.py::_qdrant_upsert"
    assert e.attrs["value_kind"] == "FUNCTION"


# ---------------------------------------------------------------------------
# Step 7: the probable tier (CALLS rungs 4-6) + DISPATCHES
# ---------------------------------------------------------------------------


def test_get_ladybug_inside_graph_tools_resolves_probable_via_injected_global(head_build):
    # THE acceptance line for step 7's first half: `_get_ladybug()` inside
    # graph_tools.py resolves to server's definition as PROBABLE with
    # because="injected at <the graph_tools.register() call site>". The
    # exact line number is read off the real INJECTS edge (never hardcoded
    # here) so this test survives mcp/server.py growing or shrinking above
    # the register() call — only the RELATIONSHIP is asserted.
    store = head_build.store
    inject_edge = store.in_edges("name:mcp/graph_tools.py::_get_ladybug", "INJECTS")[0]
    inject_site = store.get(inject_edge.src)
    expected_because = f"injected at {inject_site.path}:{inject_site.line}"

    rows = direct_callers(store, "fn:mcp/server.py::_get_ladybug")
    probable_calls = [r for r in rows if r.kind == "CALLS" and r.edge.confidence == Confidence.PROBABLE]
    graph_tools_calls = [r for r in probable_calls if r.site_node is not None and r.site_node.path == "mcp/graph_tools.py"]
    # Matches step 6's own test: exactly 11 in_call_position READS_NAME
    # edges for THIS NAME slot (mcp/graph_tools.py::_get_ladybug) — every
    # one of them is a bare `_get_ladybug()` CALLSITE, and every one must
    # now resolve through rung 4. direct_callers(fn:...) additionally picks
    # up mcp/ladybug_ops.py's own separately-injected NAME slot pointing at
    # the same target function — see the next test for that one.
    assert len(graph_tools_calls) == 11
    assert all(r.edge.attrs["rung"] == "name-via-injected-global" for r in graph_tools_calls)
    assert all(r.edge.attrs["because"] == expected_because for r in graph_tools_calls)


def test_ladybug_ops_get_ladybug_calls_also_resolve_probable(head_build):
    # ladybug_ops.py's own `_get_ladybug` slot is injected at a DIFFERENT
    # callsite (mcp/server.py's ladybug_ops.register(...) call, not
    # graph_tools'), proving rung 4 isn't special-cased to one module.
    store = head_build.store
    rows = direct_callers(store, "fn:mcp/server.py::_get_ladybug")
    ladybug_ops_calls = [
        r for r in rows if r.kind == "CALLS" and r.edge.confidence == Confidence.PROBABLE
        and r.site_node is not None and r.site_node.path == "mcp/ladybug_ops.py"
    ]
    assert len(ladybug_ops_calls) == 4
    assert all(r.edge.attrs["rung"] == "name-via-injected-global" for r in ladybug_ops_calls)


def test_a2a_dispatch_fans_out_to_all_thirteen_skills(head_build):
    # THE acceptance line for step 7's second half: `await handler(task)` at
    # a2a_server/server.py fans out to all 13 skills, each printing as
    # "1 of 13" (see test_cli.py's dispatch test for the rendered form).
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
    # `cg callers <one skill>` must show the dispatch row (design: "Reverse
    # closure over CALLS u REFERENCES u DISPATCHES u ALIASES"), not just the
    # proven in-corpus rows earlier slices already covered.
    store = head_build.store
    rows = direct_callers(store, "fn:a2a_server/server.py::skill_memory_recall")
    disp_rows = [r for r in rows if r.kind == "DISPATCHES"]
    assert len(disp_rows) == 1
    assert disp_rows[0].edge.attrs["selector"] == "skill_id"
    assert disp_rows[0].site_node is not None and disp_rows[0].site_node.path == "a2a_server/server.py"


def test_hard_gate_still_holds_after_the_probable_tier(head_build):
    # The probable tier must never turn a registered function's status from
    # "already fine" into "now dead" — retargeting/attrs mutation on `?`
    # edges must never touch an already-PROVEN/PROBABLE rung-1-3 edge.
    bad = registered_but_dead(head_build.store)
    assert bad == [], [n.id for n in bad]


# ---------------------------------------------------------------------------
# Step 9: traversal commands — cg paths / cg entrypoints
# ---------------------------------------------------------------------------


def test_entrypoints_names_the_mcp_tool_reaching_code_memory_relink(head_build):
    # ACCEPTANCE for step 9: `cg entrypoints mcp/graph_tools.py:<a line
    # inside code_memory_relink>` names the MCP tool that reaches it. Uses a
    # line computed from the function's own span (never a hardcoded literal
    # line number) so this survives the file growing/shrinking elsewhere.
    store = head_build.store
    fn = store.get("fn:mcp/graph_tools.py::code_memory_relink")
    assert fn is not None
    mid_line = (fn.line + fn.attrs["end_lineno"]) // 2
    start_id = function_at_line(store, "mcp/graph_tools.py", mid_line)
    assert start_id == fn.id

    entries = entrypoints_reaching(store, start_id)
    assert entries.get("entry:mcp-tool:code_memory_relink") == Confidence.PROVEN


def test_paths_from_code_memory_relink_to_get_ladybug_is_probable(head_build):
    # ACCEPTANCE for step 9: path confidence equals the minimum hop
    # confidence — code_memory_relink's own `ks = _get_ladybug()` call is a
    # single PROBABLE (rung 4) hop, so the whole path reports PROBABLE, not
    # PROVEN. See test_analyze_reach.py for the hand-built mixed-hop fixture
    # chain that isolates the min-combinator behaviour itself.
    store = head_build.store
    hops = shortest_path(
        store, "fn:mcp/graph_tools.py::code_memory_relink", "fn:mcp/server.py::_get_ladybug",
        conf_floor=Confidence.UNPROVEN,
    )
    assert hops is not None and len(hops) == 1
    assert hops[0].edge.confidence == Confidence.PROBABLE
    assert path_confidence(hops) == Confidence.PROBABLE
