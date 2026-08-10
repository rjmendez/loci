"""extract/dispatch.py: the probable tier (CALLS rungs 4-6) and DISPATCHES,
against real GraphStore output — not just "doesn't raise"."""
from ..model import Confidence
from ..tests.helpers import build_fixture_store


def _calls_from(store, mod, fn_qualname, callee_ident=None):
    fn_id = f"fn:{mod}::{fn_qualname}"
    sites = [n for n in store.nodes_of_kind("CALLSITE") if n.attrs.get("enclosing_fn") == fn_id]
    if callee_ident is not None:
        sites = [n for n in sites if n.attrs.get("callee") == callee_ident]
    assert len(sites) == 1, (fn_qualname, callee_ident, len(sites))
    site = sites[0]
    edges = store.out_edges(site.id, "CALLS")
    assert len(edges) == 1
    return site, edges[0]


# -- rung 4: name-via-injected-global ---------------------------------------


def test_single_value_injection_resolves_probable_with_because():
    store, _, _ = build_fixture_store(["registry_injection.py", "registry_injection_caller.py"])
    site, edge = _calls_from(store, "registry_injection.py", "use_thing")
    assert edge.dst == "fn:registry_injection_caller.py::real_thing"
    assert edge.confidence == Confidence.PROBABLE
    assert edge.attrs["rung"] == "name-via-injected-global"
    assert edge.attrs["because"] == "injected at registry_injection_caller.py:16"


def test_multi_value_injection_leaves_calls_unresolved_and_dispatches_fan_out():
    store, _, _ = build_fixture_store([
        "registry_injection.py", "registry_injection_caller.py", "registry_injection_caller2.py",
    ])
    site, edge = _calls_from(store, "registry_injection.py", "use_thing")
    assert edge.dst == "?"
    assert edge.attrs["reason"] == "injected-global-multi-value"
    assert edge.attrs["alternatives"] == 2
    disp = {e.dst for e in store.out_edges(site.id, "DISPATCHES")}
    assert disp == {
        "fn:registry_injection_caller.py::real_thing",
        "fn:registry_injection_caller2.py::other_thing",
    }
    for e in store.out_edges(site.id, "DISPATCHES"):
        assert e.confidence == Confidence.PROBABLE
        assert e.attrs["fanout"] == 2


def test_proven_rungs_1_3_are_never_revisited_by_the_probable_tier():
    # register()'s own body calls _get_thing/_helper_a nowhere by bare name,
    # but the module DOES have proven-rung calls (real_thing() inside
    # registry_injection_caller.py's own module body is a def, not a call);
    # use the calls_shapes fixture instead, which is dense with proven rungs,
    # and assert none of them got touched by a second pass.
    store, _, _ = build_fixture_store(["calls_shapes.py", "calls_target.py"])
    fn_id = "fn:calls_shapes.py::caller_name_def_local"
    sites = [n for n in store.nodes_of_kind("CALLSITE") if n.attrs.get("enclosing_fn") == fn_id]
    edge = store.out_edges(sites[0].id, "CALLS")[0]
    assert edge.confidence == Confidence.PROVEN
    assert edge.attrs["rung"] == "name-def-local"


# -- rung 5: dict-dispatch ----------------------------------------------------


def test_dict_get_dispatch_fans_out_to_every_registry_member():
    store, _, _ = build_fixture_store(["registry_mandict.py"])
    site, edge = _calls_from(store, "registry_mandict.py", "dispatch", "handler")
    assert edge.dst == "?"
    assert edge.attrs["reason"] == "dict-dispatch-fanout"
    assert edge.attrs["registry"] == "reg:registry_mandict.py::_SKILL_MAP"
    assert edge.attrs["fanout"] == 2
    assert edge.attrs["selector"] == "skill_id"
    disp = {e.dst: e for e in store.out_edges(site.id, "DISPATCHES")}
    assert set(disp) == {"fn:registry_mandict.py::skill_one", "fn:registry_mandict.py::skill_two"}
    for e in disp.values():
        assert e.confidence == Confidence.PROBABLE
        assert e.attrs["fanout"] == 2
        assert e.attrs["selector"] == "skill_id"
        assert e.attrs["registry"] == "reg:registry_mandict.py::_SKILL_MAP"


def test_dict_subscript_dispatch_same_as_dict_get():
    store, _, _ = build_fixture_store(["registry_mandict.py"])
    site, edge = _calls_from(store, "registry_mandict.py", "dispatch_subscript", "handler")
    assert edge.attrs["reason"] == "dict-dispatch-fanout"
    disp = {e.dst for e in store.out_edges(site.id, "DISPATCHES")}
    assert disp == {"fn:registry_mandict.py::skill_one", "fn:registry_mandict.py::skill_two"}


def test_config_dict_is_not_a_dispatch_source():
    # _CONFIG_MAP's values are literals, not bare-callable Names, so
    # extract/registry.py never creates a REGISTRY for it — a local var
    # assigned from it must not spuriously get a DISPATCHES fan-out either.
    store, _, _ = build_fixture_store(["registry_mandict.py"])
    assert store.get("reg:registry_mandict.py::_CONFIG_MAP") is None


# -- rung 6: unique-method-name -----------------------------------------------


def test_unique_method_name_resolves_probable():
    store, _, _ = build_fixture_store(["dispatch_shapes.py", "dispatch_target.py"])
    site, edge = _calls_from(store, "dispatch_shapes.py", "call_unique_method")
    assert edge.dst == "fn:dispatch_shapes.py::OnlyOwner.unique_op"
    assert edge.confidence == Confidence.PROBABLE
    assert edge.attrs["rung"] == "unique-method-name"
    assert edge.attrs["alternatives"] == 1


def test_ambiguous_method_name_stays_unresolved_with_alternatives_count():
    store, _, _ = build_fixture_store(["dispatch_shapes.py", "dispatch_target.py"])
    site, edge = _calls_from(store, "dispatch_shapes.py", "call_ambiguous_method")
    assert edge.dst == "?"
    assert edge.attrs["reason"] == "ambiguous-method-name"
    assert edge.attrs["alternatives"] == 2


# -- getattr-literal -----------------------------------------------------


def test_getattr_with_literal_attribute_resolves_proven():
    store, _, _ = build_fixture_store(["dispatch_shapes.py", "dispatch_target.py"])
    fn_id = "fn:dispatch_shapes.py::call_getattr_literal"
    sites = [n for n in store.nodes_of_kind("CALLSITE")
             if n.attrs.get("enclosing_fn") == fn_id and n.attrs.get("form") == "getattr-result"]
    assert len(sites) == 1
    edge = store.out_edges(sites[0].id, "CALLS")[0]
    assert edge.dst == "fn:dispatch_target.py::target_fn"
    assert edge.confidence == Confidence.PROVEN
    assert edge.attrs["rung"] == "getattr-literal"


def test_getattr_with_variable_attribute_stays_a_hole():
    store, _, _ = build_fixture_store(["dispatch_shapes.py", "dispatch_target.py"])
    fn_id = "fn:dispatch_shapes.py::call_getattr_variable"
    sites = [n for n in store.nodes_of_kind("CALLSITE")
             if n.attrs.get("enclosing_fn") == fn_id and n.attrs.get("form") == "getattr-result"]
    assert len(sites) == 1
    edge = store.out_edges(sites[0].id, "CALLS")[0]
    assert edge.dst == "?"
    assert edge.attrs["reason"] == "computed-getattr"
