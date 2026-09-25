"""analyze/reach.py: shortest_path/path_confidence/entrypoints_reaching/
function_at_line, against real GraphStore output built from fixtures."""
from ..analyze.reach import (
    entrypoints_reaching, function_at_line, path_confidence, shortest_path,
)
from ..model import Confidence, Edge, GraphStore, Node
from ..tests.helpers import build_fixture_store


# -- shortest_path / path_confidence -----------------------------------------


def test_shortest_path_single_hop():
    store, _, _ = build_fixture_store(["calls_shapes.py", "calls_target.py"])
    hops = shortest_path(store, "fn:calls_shapes.py::caller_name_def_local", "fn:calls_shapes.py::helper")
    assert hops is not None and len(hops) == 1
    assert hops[0].edge.dst == "fn:calls_shapes.py::helper"
    assert hops[0].edge.confidence == Confidence.PROVEN
    assert path_confidence(hops) == Confidence.PROVEN


def test_shortest_path_src_equals_dst_is_empty_path():
    store, _, _ = build_fixture_store(["calls_shapes.py", "calls_target.py"])
    hops = shortest_path(store, "fn:calls_shapes.py::helper", "fn:calls_shapes.py::helper")
    assert hops == []


def test_shortest_path_unreachable_returns_none():
    store, _, _ = build_fixture_store(["calls_shapes.py", "calls_target.py"])
    hops = shortest_path(store, "fn:calls_shapes.py::helper", "fn:calls_target.py::target_fn")
    assert hops is None


def test_path_confidence_is_the_minimum_hop_not_the_average():
    # A mixed proven+probable chain reports min(), never a blended value.
    store, _, _ = build_fixture_store(["registry_injection.py", "registry_injection_caller.py"])
    hops = shortest_path(
        store, "fn:registry_injection.py::call_use_thing", "fn:registry_injection_caller.py::real_thing",
    )
    assert hops is not None and len(hops) == 2
    confidences = [h.edge.confidence for h in hops]
    assert confidences == [Confidence.PROVEN, Confidence.PROBABLE]
    assert path_confidence(hops) == Confidence.PROBABLE
    # A PROVEN-only prefix still reports PROVEN: not trivially always PROBABLE.
    assert path_confidence(hops[:1]) == Confidence.PROVEN


# -- function_at_line ----------------------------------------------------


def test_function_at_line_finds_enclosing_function():
    store, _, _ = build_fixture_store(["calls_shapes.py", "calls_target.py"])
    fn = store.get("fn:calls_shapes.py::caller_name_def_local")
    mid_line = (fn.line + fn.attrs["end_lineno"]) // 2
    assert function_at_line(store, "calls_shapes.py", mid_line) == fn.id


def test_function_at_line_picks_innermost_nested_function():
    store, _, _ = build_fixture_store(["calls_shapes.py", "calls_target.py"])
    nested = store.get("fn:calls_shapes.py::outer_with_nested.<locals>.helper_nested")
    assert function_at_line(store, "calls_shapes.py", nested.line) == nested.id


def test_function_at_line_falls_back_to_module_for_top_level_code():
    store, _, _ = build_fixture_store(["registry_injection_caller.py"])
    # line 1 is the module docstring, outside every function's span.
    assert function_at_line(store, "registry_injection_caller.py", 1) == "mod:registry_injection_caller.py"


# -- entrypoints_reaching -----------------------------------------------------


def test_entrypoints_reaching_direct_registration():
    store, _, _ = build_fixture_store(["decorator_registry.py"])
    fn_id = "fn:decorator_registry.py::registered_tool"
    entries = entrypoints_reaching(store, fn_id)
    assert entries == {"entry:mcp-tool:registered_tool": Confidence.PROVEN}


def test_entrypoints_reaching_injected_global_with_no_entry_is_empty():
    store, _, _ = build_fixture_store([
        "registry_injection.py", "registry_injection_caller.py", "registry_manloop.py",
    ])
    # No `register` here is an entrypoint target, so this pins the absence case.
    entries = entrypoints_reaching(store, "fn:registry_injection.py::use_thing")
    assert entries == {}


def _walk_graph() -> GraphStore:
    """A small graph with every edge kind entrypoints_reaching walks backwards.

        entry:e1 -ENTERS(PROVEN)->   fn:a
        entry:e2 -ENTERS(PROBABLE)-> reg:r -REGISTERS(PROVEN)-> fn:b
        entry:e3 -ENTERS(UNPROVEN)-> fn:a
        fn:a  --site:a1 CALLS(PROVEN)-->        fn:b
        fn:b  --site:b1 DISPATCHES(PROBABLE)--> fn:c
        mod:m -REFERENCES(PROVEN)-> fn:c   (mod:m is itself entered by entry:e4, PROVEN)

    No fixture in the corpus has a PROBABLE entry edge or a multi-hop walk, so the
    graph is built directly; the unit under test is the walk, not the extractor.
    """
    g = GraphStore()
    for nid, kind in (("entry:e1", "ENTRYPOINT"), ("entry:e2", "ENTRYPOINT"),
                      ("entry:e3", "ENTRYPOINT"), ("entry:e4", "ENTRYPOINT"),
                      ("reg:r", "REGISTRY"), ("fn:a", "FUNCTION"), ("fn:b", "FUNCTION"),
                      ("fn:c", "FUNCTION"), ("mod:m", "MODULE")):
        g.add_node(Node(nid, kind))
    g.add_node(Node("site:a1", "CALLSITE", attrs={"enclosing_fn": "fn:a"}))
    g.add_node(Node("site:b1", "CALLSITE", attrs={"enclosing_fn": "fn:b"}))
    for src, dst, kind, conf in (
        ("entry:e1", "fn:a", "ENTERS", Confidence.PROVEN),
        ("entry:e2", "reg:r", "ENTERS", Confidence.PROBABLE),
        ("entry:e3", "fn:a", "ENTERS", Confidence.UNPROVEN),
        ("entry:e4", "mod:m", "ENTERS", Confidence.PROVEN),
        ("reg:r", "fn:b", "REGISTERS", Confidence.PROVEN),
        ("site:a1", "fn:b", "CALLS", Confidence.PROVEN),
        ("site:b1", "fn:c", "DISPATCHES", Confidence.PROBABLE),
        ("mod:m", "fn:c", "REFERENCES", Confidence.PROVEN),
    ):
        g.add_edge(Edge(src, dst, kind, conf))
    return g


def test_entrypoints_reaching_walks_back_transitively_over_every_edge_kind():
    g = _walk_graph()
    # fn:c <- DISPATCHES from b <- CALLS from a <- e1; <- REGISTERS from r <- e2 (PROBABLE);
    # <- REFERENCES from mod:m <- e4. e3's UNPROVEN entry is below the default floor.
    assert entrypoints_reaching(g, "fn:c") == {
        "entry:e1": Confidence.PROVEN, "entry:e2": Confidence.PROBABLE, "entry:e4": Confidence.PROVEN,
    }
    assert entrypoints_reaching(g, "fn:b") == {
        "entry:e1": Confidence.PROVEN, "entry:e2": Confidence.PROBABLE,
    }


def test_entrypoints_reaching_respects_the_confidence_floor():
    g = _walk_graph()
    # PROVEN-only: the PROBABLE dispatch into c and e2's PROBABLE entry both drop out.
    assert entrypoints_reaching(g, "fn:c", conf_floor=Confidence.PROVEN) == {"entry:e4": Confidence.PROVEN}
    assert entrypoints_reaching(g, "fn:b", conf_floor=Confidence.PROVEN) == {"entry:e1": Confidence.PROVEN}
    # UNPROVEN floor admits e3 as well
    assert entrypoints_reaching(g, "fn:a", conf_floor=Confidence.UNPROVEN) == {
        "entry:e1": Confidence.PROVEN, "entry:e3": Confidence.UNPROVEN,
    }


def test_entrypoints_reaching_unregistered_function_is_empty():
    store, _, _ = build_fixture_store(["calls_shapes.py", "calls_target.py"])
    entries = entrypoints_reaching(store, "fn:calls_target.py::target_fn")
    assert entries == {}
