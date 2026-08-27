from .conftest import needs_corpus_deps, needs_git_history  # noqa: F401
"""extract/imports.py: IMPORTS edges (scope, resolved_via) and ALIASES
edges (from-import re-exports, bare module-level A = B), across the
synthetic fixtures — this is the step-3 acceptance surface."""
from ..extract.defs import extract_module
from ..extract.imports import extract_imports
from ..model import GraphStore
from ..resolve import ResolutionTable
from ..tests.helpers import load_fixture


def _build(rel_paths: list[str]):
    sources = [load_fixture(p) for p in rel_paths]
    table = ResolutionTable(sources)
    store = GraphStore()
    scopes = {sf.rel_path: extract_module(store, sf, table) for sf in sources}
    for sf in sources:
        extract_imports(store, sf, scopes[sf.rel_path], table, scopes)
    return store, table


@needs_corpus_deps
def test_lazy_import_edge_carries_scope_and_enclosing_fn():
    store, _ = _build(["lazy_import.py"])
    edges = store.out_edges("mod:lazy_import.py", "IMPORTS")
    by_module = {e.attrs["module"]: e for e in edges}
    assert by_module["json"].attrs["scope"] == "module-level"
    assert by_module["json"].attrs["enclosing_fn"] is None
    assert by_module["numpy"].attrs["scope"] == "function-local"
    assert by_module["numpy"].attrs["enclosing_fn"] == "resolve_vllm"
    assert by_module["numpy"].confidence.name == "PROVEN"  # third-party numpy


def test_flat_sibling_import_resolves_to_module_node():
    store, _ = _build(["pkg/scripts/entry.py", "pkg/mcp/sibling.py"])
    edges = store.out_edges("mod:pkg/scripts/entry.py", "IMPORTS")
    sibling_edge = next(e for e in edges if e.attrs["module"] == "sibling")
    assert sibling_edge.dst == "mod:pkg/mcp/sibling.py"
    assert sibling_edge.confidence.name == "PROVEN"
    assert "sys.path.insert@pkg/scripts/entry.py" in sibling_edge.attrs["resolved_via"]


def test_reexport_aliases_to_function_node_not_a_disconnected_copy():
    store, _ = _build(["reexport.py", "reexport_source.py"])
    aliases = store.out_edges("name:reexport.py::symbol_impact", "ALIASES")
    assert len(aliases) == 1
    assert aliases[0].dst == "fn:reexport_source.py::symbol_impact"
    assert aliases[0].confidence.name == "PROVEN"
    assert aliases[0].attrs["form"] == "from-import"

    aliases2 = store.out_edges("name:reexport.py::impact_report", "ALIASES")
    assert aliases2[0].dst == "fn:reexport_source.py::impact_report"


def test_bare_module_level_assign_aliases_to_the_same_function():
    store, _ = _build(["reexport.py", "reexport_source.py"])
    # Two-hop chain, not a collapsed shortcut: `runner` -> the NAME slot -> the function.
    runner_aliases = store.out_edges("name:reexport.py::runner", "ALIASES")
    assert len(runner_aliases) == 1
    assert runner_aliases[0].dst == "name:reexport.py::symbol_impact"
    assert runner_aliases[0].attrs["form"] == "assign"

    imported_slot_aliases = store.out_edges("name:reexport.py::symbol_impact", "ALIASES")
    assert imported_slot_aliases[0].dst == "fn:reexport_source.py::symbol_impact"


def test_import_module_binding_aliases_to_module_node():
    store, _ = _build(["pkg/scripts/entry.py", "pkg/mcp/sibling.py"])
    alias = store.out_edges("name:pkg/scripts/entry.py::sibling", "ALIASES")
    assert len(alias) == 1
    assert alias[0].dst == "mod:pkg/mcp/sibling.py"
    assert alias[0].attrs["form"] == "import"


def test_stdlib_import_creates_external_node():
    store, _ = _build(["lazy_import.py"])
    ext = store.get("ext:json")
    assert ext is not None
    assert ext.attrs["distribution"] == "stdlib"
