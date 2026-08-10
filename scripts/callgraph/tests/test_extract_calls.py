"""extract/calls.py: CALLSITE nodes and the rungs-1-3 CALLS resolution
ladder, against real GraphStore output (not just "doesn't raise")."""
from ..model import Confidence
from ..tests.helpers import build_fixture_store


def _build():
    return build_fixture_store(["calls_shapes.py", "calls_target.py"])


def _calls_from(store, fn_qualname):
    fn_id = f"fn:calls_shapes.py::{fn_qualname}"
    sites = [n for n in store.nodes_of_kind("CALLSITE") if n.attrs.get("enclosing_fn") == fn_id]
    assert len(sites) == 1, f"expected exactly one callsite in {fn_qualname}, found {len(sites)}"
    site = sites[0]
    edges = store.out_edges(site.id, "CALLS")
    assert len(edges) == 1
    return site, edges[0]


def test_every_call_gets_a_callsite_node():
    store, _, _ = _build()
    sites = list(store.nodes_of_kind("CALLSITE"))
    assert len(sites) >= 10


def test_name_def_local_rung():
    store, _, _ = _build()
    site, edge = _calls_from(store, "caller_name_def_local")
    assert edge.dst == "fn:calls_shapes.py::helper"
    assert edge.confidence == Confidence.PROVEN
    assert edge.attrs["rung"] == "name-def-local"
    assert site.attrs["form"] == "name"


def test_nested_def_local_rung():
    store, _, _ = _build()
    site, edge = _calls_from(store, "outer_with_nested")
    assert edge.dst == "fn:calls_shapes.py::outer_with_nested.<locals>.helper_nested"
    assert edge.confidence == Confidence.PROVEN
    assert edge.attrs["rung"] == "nested-def-local"


def test_constructor_rung_targets_init():
    store, _, _ = _build()
    site, edge = _calls_from(store, "make_widget")
    assert edge.dst == "fn:calls_shapes.py::Widget.__init__"
    assert edge.confidence == Confidence.PROVEN
    assert edge.attrs["rung"] == "name-constructor"


def test_self_attribute_rung():
    store, _, _ = _build()
    fn_id = "fn:calls_shapes.py::Widget.bump"
    sites = [n for n in store.nodes_of_kind("CALLSITE") if n.attrs.get("enclosing_fn") == fn_id]
    assert len(sites) == 1
    edge = store.out_edges(sites[0].id, "CALLS")[0]
    assert edge.dst == "fn:calls_shapes.py::Widget.other"
    assert edge.attrs["rung"] == "self-attribute"
    assert edge.confidence == Confidence.PROVEN


def test_module_attribute_external_rung():
    store, _, _ = _build()
    site, edge = _calls_from(store, "caller_module_attribute_external")
    assert edge.dst == "ext:json.dumps"
    assert edge.confidence == Confidence.PROVEN
    assert edge.attrs["rung"] == "module-attribute-external"
    ext = store.get("ext:json.dumps")
    assert ext is not None and ext.attrs["distribution"] == "stdlib"


def test_module_attribute_in_corpus_rung():
    store, _, _ = _build()
    site, edge = _calls_from(store, "caller_module_attribute_corpus")
    assert edge.dst == "fn:calls_target.py::target_fn"
    assert edge.confidence == Confidence.PROVEN
    assert edge.attrs["rung"] == "module-attribute"


def test_builtin_rung():
    store, _, _ = _build()
    site, edge = _calls_from(store, "caller_builtin")
    assert edge.dst == "ext:builtins.len"
    assert edge.confidence == Confidence.PROVEN
    assert edge.attrs["rung"] == "builtin"


def test_function_local_import_rung_tagged_scope():
    store, _, _ = _build()
    site, edge = _calls_from(store, "caller_local_import")
    assert edge.dst == "ext:textwrap.dedent"
    assert edge.confidence == Confidence.PROVEN
    assert edge.attrs["scope"] == "function-local"
    assert "local import at calls_shapes.py" in edge.attrs["because"]


def test_param_call_is_unresolved_with_reason():
    store, _, _ = _build()
    site, edge = _calls_from(store, "caller_param_call")
    assert site.attrs["form"] == "param-call"
    assert edge.dst == "?"
    assert edge.confidence == Confidence.UNPROVEN
    assert edge.attrs["reason"] == "param-call"


def test_unknown_attribute_receiver_is_unresolved():
    store, _, _ = _build()
    site, edge = _calls_from(store, "caller_unknown_attribute")
    assert site.attrs["form"] == "attribute"
    assert edge.dst == "?"
    assert edge.attrs["reason"] == "attribute-unknown-receiver"


def test_unresolved_sink_node_exists_and_is_kind_unresolved():
    store, _, _ = _build()
    node = store.get("?")
    assert node is not None
    assert node.kind == "UNRESOLVED"


def test_chained_call_gets_two_distinct_callsite_ids():
    # `pyotp.TOTP(x).now()` shape: outer and inner Call share (line, col)
    # but must be two distinct CALLSITE nodes, not one merged/overwritten
    # node with two CALLS edges hanging off it.
    from ..tests.helpers import source_file
    from ..model import GraphStore
    from ..resolve import ResolutionTable
    from ..extract.defs import extract_module
    from ..extract.imports import extract_imports
    from ..extract.calls import build_top_level_index, extract_calls

    text = (
        "import pyotp\n\n"
        "def f(seed):\n"
        "    return pyotp.TOTP(seed).now()\n"
    )
    sf = source_file("chained.py", text)
    table = ResolutionTable([sf])
    store = GraphStore()
    scope = extract_module(store, sf, table)
    extract_imports(store, sf, scope, table, {"chained.py": scope})
    idx = build_top_level_index(store)
    extract_calls(store, sf, scope, table, idx)

    sites = [n for n in store.nodes_of_kind("CALLSITE") if n.path == "chained.py"]
    assert len(sites) == 2
    forms = {n.attrs["form"] for n in sites}
    assert forms == {"attribute"}
    ids = {n.id for n in sites}
    assert len(ids) == 2  # distinct node identities, not one node with 2 edges
