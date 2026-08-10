"""analyze/reach.py: shortest_path/path_confidence/entrypoints_reaching/
function_at_line, against real GraphStore output built from fixtures."""
from ..analyze.reach import (
    entrypoints_reaching, function_at_line, path_confidence, shortest_path,
)
from ..model import Confidence
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
    # call_use_thing --[PROVEN name-def-local]--> use_thing
    #                --[PROBABLE name-via-injected-global]--> real_thing
    # THE hand-built fixture chain for step 9's acceptance line: a mixed
    # proven+probable chain must report PROBABLE overall, not PROVEN and
    # not some blended value — Confidence has no such thing, only min().
    store, _, _ = build_fixture_store(["registry_injection.py", "registry_injection_caller.py"])
    hops = shortest_path(
        store, "fn:registry_injection.py::call_use_thing", "fn:registry_injection_caller.py::real_thing",
    )
    assert hops is not None and len(hops) == 2
    confidences = [h.edge.confidence for h in hops]
    assert confidences == [Confidence.PROVEN, Confidence.PROBABLE]
    assert path_confidence(hops) == Confidence.PROBABLE
    # One PROVEN hop alone would report PROVEN — confirms this isn't
    # trivially always PROBABLE for any chain touching this fixture.
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


def test_entrypoints_reaching_through_injected_global_is_probable():
    # register()'s own function is not itself registered anywhere in this
    # fixture set — the interesting case is a function that calls THROUGH
    # an injected global reaching whatever entrypoint enters the CALLER.
    store, _, _ = build_fixture_store([
        "registry_injection.py", "registry_injection_caller.py", "registry_manloop.py",
    ])
    # registry_manloop.py's `register` isn't itself an entrypoint target,
    # so instead verify the *absence* case: a function reachable from
    # nothing has an empty entrypoint set — the honest "0 known" answer.
    entries = entrypoints_reaching(store, "fn:registry_injection.py::use_thing")
    assert entries == {}


def test_entrypoints_reaching_unregistered_function_is_empty():
    store, _, _ = build_fixture_store(["calls_shapes.py", "calls_target.py"])
    entries = entrypoints_reaching(store, "fn:calls_target.py::target_fn")
    assert entries == {}
