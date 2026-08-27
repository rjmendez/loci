"""extract/registry.py: DEC, MAN-LOOP, MAN-DICT, ROOT-CLI, and REG-FN/INJECTS,
against real GraphStore output."""
from ..model import Confidence
from ..tests.helpers import build_fixture_store


# -- DEC (reuses slice-1's decorator_registry.py fixture) ------------------


def test_dec_creates_registry_and_retargets_decorated_by():
    store, _, _ = build_fixture_store(["decorator_registry.py"])
    reg_id = "reg:decorator_registry.py::mcp.tool"
    reg = store.get(reg_id)
    assert reg is not None and reg.kind == "REGISTRY"
    assert reg.attrs["mechanism"] == "decorator"

    # DECORATED_BY must be retargeted off defs.py's placeholder EXTERNAL sink.
    fid = "fn:decorator_registry.py::registered_tool"
    dbs = store.out_edges(fid, "DECORATED_BY")
    assert len(dbs) == 1
    assert dbs[0].dst == reg_id
    # the classification/raw attrs survive the retarget untouched
    assert dbs[0].attrs["classification"] == "registering"
    assert dbs[0].attrs["raw"] == "mcp.tool()"

    # a non-registering decorator's DECORATED_BY edge is untouched (still EXTERNAL)
    cached_id = "fn:decorator_registry.py::cached_helper"
    cached_dbs = store.out_edges(cached_id, "DECORATED_BY")
    assert cached_dbs[0].dst.startswith("ext:")


def test_dec_registers_member_and_entrypoint():
    store, _, _ = build_fixture_store(["decorator_registry.py"])
    reg_id = "reg:decorator_registry.py::mcp.tool"
    fid = "fn:decorator_registry.py::registered_tool"
    members = store.out_edges(reg_id, "REGISTERS")
    assert len(members) == 1
    assert members[0].dst == fid
    assert members[0].attrs["key"] == "registered_tool"

    entry = store.get("entry:mcp-tool:registered_tool")
    assert entry is not None and entry.kind == "ENTRYPOINT"
    enters = store.out_edges(entry.id, "ENTERS")
    assert len(enters) == 1 and enters[0].dst == fid


def test_unrecognized_decorator_creates_no_registry():
    store, _, _ = build_fixture_store(["decorator_registry.py"])
    assert store.get("reg:decorator_registry.py::some_unrecognized_decorator") is None


# -- MAN-LOOP ----------------------------------------------------------


def test_manloop_registers_both_tuple_elements():
    store, _, _ = build_fixture_store(["registry_manloop.py"])
    reg_id = "reg:registry_manloop.py::register"
    reg = store.get(reg_id)
    assert reg is not None and reg.attrs["mechanism"] == "manifest-tuple"
    members = {e.dst for e in store.out_edges(reg_id, "REGISTERS")}
    assert members == {"fn:registry_manloop.py::tool_a", "fn:registry_manloop.py::tool_b"}
    for name in ("tool_a", "tool_b"):
        assert store.get(f"entry:mcp-tool:{name}") is not None


def test_manloop_emits_references_for_each_element():
    store, _, _ = build_fixture_store(["registry_manloop.py"])
    register_fn_id = "fn:registry_manloop.py::register"
    refs = {e.dst for e in store.out_edges(register_fn_id, "REFERENCES")}
    assert refs == {"fn:registry_manloop.py::tool_a", "fn:registry_manloop.py::tool_b"}


def test_unrelated_loop_over_bare_names_is_not_manloop():
    # `for fn in (_config,): fn.cache_clear()` calls a method ON the loop var; it registers nothing.
    store, _, _ = build_fixture_store(["registry_manloop.py"])
    assert store.get("reg:registry_manloop.py::_reset_cache") is None
    reset_fn_id = "fn:registry_manloop.py::_reset_cache"
    # A name-escape edge is correct here (it keeps `_config` off `cg dead`); a REGISTRATION form is not.
    forms = {e.attrs.get("form") for e in store.out_edges(reset_fn_id, "REFERENCES")}
    assert forms == {"name-escape"}, forms
    assert not store.edges_of_kind("REGISTERS") or all(
        e.src != "reg:registry_manloop.py::_reset_cache" for e in store.edges_of_kind("REGISTERS")
    )


# -- MAN-DICT ----------------------------------------------------------


def test_mandict_registers_skill_map_only():
    store, _, _ = build_fixture_store(["registry_mandict.py"])
    reg_id = "reg:registry_mandict.py::_SKILL_MAP"
    reg = store.get(reg_id)
    assert reg is not None and reg.attrs["mechanism"] == "manifest-dict"
    keys = {e.attrs["key"] for e in store.out_edges(reg_id, "REGISTERS")}
    assert keys == {"one", "two"}
    assert store.get("entry:skill:one") is not None
    assert store.get("entry:skill:two") is not None


def test_config_dict_with_literal_values_is_not_mandict():
    store, _, _ = build_fixture_store(["registry_mandict.py"])
    assert store.get("reg:registry_mandict.py::_CONFIG_MAP") is None


def test_mandict_emits_references_from_module():
    store, _, _ = build_fixture_store(["registry_mandict.py"])
    mod_refs = {e.dst for e in store.out_edges("mod:registry_mandict.py", "REFERENCES")}
    assert mod_refs == {"fn:registry_mandict.py::skill_one", "fn:registry_mandict.py::skill_two"}


# -- ROOT-CLI ------------------------------------------------------------


def test_rootcli_entrypoint_enters_main():
    store, _, _ = build_fixture_store(["registry_rootcli.py"])
    entry = store.get("entry:cli:registry_rootcli.py")
    assert entry is not None and entry.attrs["trust"] == "declared-in-source"
    enters = store.out_edges(entry.id, "ENTERS")
    assert len(enters) == 1
    assert enters[0].dst == "fn:registry_rootcli.py::main"


# -- REG-FN / INJECTS ------------------------------------------------------


def _injection_store():
    return build_fixture_store(["registry_injection.py", "registry_injection_caller.py"])


def test_direct_param_injection():
    store, _, _ = _injection_store()
    nid = "name:registry_injection.py::_get_thing"
    injects = store.in_edges(nid, "INJECTS")
    assert len(injects) == 1
    e = injects[0]
    assert e.confidence == Confidence.PROBABLE
    assert e.attrs["value"] == "fn:registry_injection_caller.py::real_thing"
    assert e.attrs["value_kind"] == "FUNCTION"
    assert e.attrs["param"] == "get_thing"


def test_deps_dict_key_injection():
    store, _, _ = _injection_store()
    nid = "name:registry_injection.py::_helper_a"
    injects = store.in_edges(nid, "INJECTS")
    assert len(injects) == 1
    e = injects[0]
    assert e.attrs["value"] == "fn:registry_injection_caller.py::real_helper_a"
    assert e.attrs["value_kind"] == "FUNCTION"
    assert e.attrs["param"] == "deps"
    assert e.attrs["key"] == "helper_a"


def test_injection_emits_references_for_function_valued_args():
    store, _, _ = _injection_store()
    mod_id = "mod:registry_injection_caller.py"
    refs = {e.dst for e in store.out_edges(mod_id, "REFERENCES") if e.attrs.get("form") == "injected-argument"}
    assert refs == {
        "fn:registry_injection_caller.py::real_thing",
        "fn:registry_injection_caller.py::real_helper_a",
    }


def test_writes_dead_style_fixture_still_bug_c_shaped():
    # extract/names.py and scopes.py must agree on what counts as global-only.
    store, _, _ = build_fixture_store(["dangling_global.py"])
    nid = "name:dangling_global.py::_symbol_index_cache"
    writes = store.in_edges(nid, "WRITES_NAME")
    assert len(writes) == 1
    assert writes[0].attrs["via"] == "global-stmt"
    assert writes[0].src == "fn:dangling_global.py::invalidate_cache"
