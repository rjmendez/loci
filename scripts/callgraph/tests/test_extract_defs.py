"""extract/defs.py: MODULE/CLASS/FUNCTION/NAME nodes, DEFINES edges, and
DECORATED_BY classification, via a real GraphStore (not just "doesn't
raise")."""
from ..extract.defs import extract_module
from ..model import GraphStore
from ..tests.helpers import load_fixture


def test_module_node_created():
    store = GraphStore()
    sf = load_fixture("decorator_registry.py")
    extract_module(store, sf)
    mod = store.get("mod:decorator_registry.py")
    assert mod is not None and mod.kind == "MODULE"
    assert mod.attrs["origin"] == "fixture"


def test_function_nodes_and_defines_edges():
    store = GraphStore()
    sf = load_fixture("decorator_registry.py")
    extract_module(store, sf)
    fn = store.get("fn:decorator_registry.py::registered_tool")
    assert fn is not None
    assert fn.kind == "FUNCTION"
    assert fn.line == 19  # the `def registered_tool(...)` line

    defines = store.out_edges("mod:decorator_registry.py", "DEFINES")
    dst_ids = {e.dst for e in defines}
    assert "fn:decorator_registry.py::registered_tool" in dst_ids
    assert "fn:decorator_registry.py::cached_helper" in dst_ids


def test_decorated_by_classification():
    store = GraphStore()
    sf = load_fixture("decorator_registry.py")
    extract_module(store, sf)
    dbs = store.out_edges("fn:decorator_registry.py::registered_tool", "DECORATED_BY")
    assert len(dbs) == 1
    assert dbs[0].attrs["classification"] == "registering"
    assert dbs[0].attrs["raw"] == "mcp.tool()"

    dbs2 = store.out_edges("fn:decorator_registry.py::cached_helper", "DECORATED_BY")
    assert dbs2[0].attrs["classification"] == "wrapping"

    dbs3 = store.out_edges("fn:decorator_registry.py::mystery", "DECORATED_BY")
    assert dbs3[0].attrs["classification"] == "unknown"


def test_class_node_and_method_defines_edge():
    store = GraphStore()
    sf = load_fixture("nesting.py")
    extract_module(store, sf)
    cls = store.get("cls:nesting.py::Widget")
    assert cls is not None
    method_id = "fn:nesting.py::Widget.method"
    assert method_id in cls.attrs["method_ids"]
    defines = {e.dst for e in store.out_edges("cls:nesting.py::Widget", "DEFINES")}
    assert method_id in defines
    # the class itself is DEFINES'd by the module, not floating unattached
    mod_defines = {e.dst for e in store.out_edges("mod:nesting.py", "DEFINES")}
    assert "cls:nesting.py::Widget" in mod_defines


def test_lambda_gets_its_own_addressable_function_node():
    store = GraphStore()
    sf = load_fixture("nesting.py")
    extract_module(store, sf)
    lambdas = [n for n in store.nodes_of_kind("FUNCTION") if n.attrs.get("is_lambda")]
    assert len(lambdas) == 1
    assert lambdas[0].id.startswith("fn:nesting.py::<lambda>@")


def test_name_slots_created_for_module_globals():
    store = GraphStore()
    sf = load_fixture("dangling_global.py")
    extract_module(store, sf)
    dangling = store.get("name:dangling_global.py::_symbol_index_cache")
    assert dangling is not None
    assert dangling.attrs["binding_kind"] == ["global-only"]
    real = store.get("name:dangling_global.py::_real_counter")
    assert "assign" in real.attrs["binding_kind"]
    assert "global-only" not in real.attrs["binding_kind"]


def test_params_recorded_on_function_node():
    store = GraphStore()
    sf = load_fixture("decorator_registry.py")
    extract_module(store, sf)
    fn = store.get("fn:decorator_registry.py::registered_tool")
    names = [p["name"] for p in fn.attrs["params"]]
    assert names == ["x", "y"]
