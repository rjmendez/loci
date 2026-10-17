"""scopes.py: qualnames (incl. nesting/lambdas), decorator source capture,
params, and NAME-slot binding kinds (incl. global-only, the raw signal
behind BUG C)."""
from ..scopes import ModuleScope
from ..tests.helpers import load_fixture


def _scope(rel_path: str) -> ModuleScope:
    return ModuleScope(load_fixture(rel_path))


def test_nesting_qualnames():
    s = _scope("nesting.py")
    quals = {f.qualname for f in s.functions}
    assert "outer" in quals
    assert "outer.<locals>.inner" in quals
    assert "register" in quals
    assert "Widget.method" in quals
    # A lambda argument is lexically at module scope, not inside the callee.
    lambdas = [f for f in s.functions if f.is_lambda]
    assert len(lambdas) == 1
    assert lambdas[0].qualname.startswith("<lambda>@")


def test_is_method_and_is_nested_flags():
    s = _scope("nesting.py")
    by_qual = {f.qualname: f for f in s.functions}
    assert by_qual["Widget.method"].is_method is True
    assert by_qual["Widget.method"].is_nested is False
    assert by_qual["outer.<locals>.inner"].is_nested is True
    assert by_qual["outer.<locals>.inner"].is_method is False
    assert by_qual["outer"].is_nested is False


def test_class_bases_and_qualname():
    s = _scope("nesting.py")
    assert len(s.classes) == 1
    assert s.classes[0].qualname == "Widget"
    assert s.classes[0].bases == []


def test_decorator_source_captured_verbatim():
    s = _scope("decorator_registry.py")
    by_qual = {f.qualname: f for f in s.functions}
    assert by_qual["registered_tool"].decorators == ["mcp.tool()"]
    assert by_qual["cached_helper"].decorators == ["functools.lru_cache"]
    assert by_qual["mystery"].decorators == ["some_unrecognized_decorator"]


def test_docstring_first_line():
    s = _scope("decorator_registry.py")
    by_qual = {f.qualname: f for f in s.functions}
    assert by_qual["registered_tool"].docstring_first_line == "First line of the docstring."
    assert by_qual["cached_helper"].docstring_first_line is None


def test_params_kinds_and_defaults():
    s = _scope("decorator_registry.py")
    by_qual = {f.qualname: f for f in s.functions}
    params = by_qual["registered_tool"].params
    assert [p.name for p in params] == ["x", "y"]
    assert params[0].kind == "POSITIONAL_OR_KEYWORD" and not params[0].has_default
    assert params[1].kind == "POSITIONAL_OR_KEYWORD" and params[1].has_default
    assert params[1].default_is_literal is True


def test_module_level_and_function_local_import_scope():
    s = _scope("lazy_import.py")
    module_level = [r for r in s.imports if r.scope == "module-level"]
    local = [r for r in s.imports if r.scope == "function-local"]
    assert len(module_level) == 1 and module_level[0].names == [("json", None)]
    assert {r.names[0][0] for r in local} == {"numpy", "textwrap"}
    numpy_rec = next(r for r in local if r.names[0][0] == "numpy")
    assert numpy_rec.enclosing_fn == "resolve_vllm"
    textwrap_rec = next(r for r in local if r.names[0][0] == "textwrap")
    assert textwrap_rec.enclosing_fn == "Widget.render"


def test_dangling_global_name_slot_detected():
    s = _scope("dangling_global.py")
    names = s.names
    assert "global-only" in names["_symbol_index_cache"].binding_kinds
    assert "global-only" in names["_symbol_index_count"].binding_kinds
    # Module-level-bound plus a `global` redeclaration is NOT global-only (BUG C's discriminator).
    assert "global-only" not in names["_real_counter"].binding_kinds
    assert "assign" in names["_real_counter"].binding_kinds


def test_name_dunder_and_private_flags():
    s = _scope("dangling_global.py")
    assert s.names["_symbol_index_cache"].is_private is True
    assert s.names["_symbol_index_cache"].is_dunder is False


def test_module_level_bare_alias_recorded():
    s = _scope("reexport.py")
    assert any(ma.target == "runner" and ma.value_kind == "name" and ma.value_name == "symbol_impact"
               for ma in s.module_assigns)
