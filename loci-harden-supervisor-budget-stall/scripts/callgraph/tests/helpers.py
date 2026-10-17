"""Shared test helpers: load a fixture .py file as a SourceFile without
going through ingest.load_corpus's repo-root file discovery (fixtures/ is
outside the analyzed corpus by design — config.SELF_PACKAGE_REL excludes
this whole package)."""
from __future__ import annotations

import ast
from pathlib import Path

from ..extract.calls import build_top_level_index, make_call_visitor
from ..extract.defs import extract_module
from ..extract.dispatch import extract_probable_calls
from ..extract.flow import extract_flow
from ..extract.funcrefs import make_funcrefs_visitor
from ..extract.imports import extract_imports
from ..extract.literals import make_literals_visitor
from ..extract.names import emit_non_walk_writes, make_names_visitor
from ..extract.registry import extract_external_roots, extract_injections, extract_registry_module
from ..extract.walk import walk_module
from ..ingest import SourceFile
from ..model import GraphStore
from ..resolve import ResolutionTable

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def load_fixture(rel_path: str) -> SourceFile:
    text = (FIXTURES_DIR / rel_path).read_text()
    return source_file(rel_path, text)


def source_file(rel_path: str, text: str) -> SourceFile:
    import hashlib
    sha1 = hashlib.sha1(text.encode()).hexdigest()
    try:
        tree = ast.parse(text, filename=rel_path)
        return SourceFile(rel_path, text, tree, None, "fixture", sha1)
    except SyntaxError as exc:
        return SourceFile(rel_path, text, None, f"SyntaxError: {exc.msg}", "fixture", sha1)


def build_fixture_store(rel_paths: list[str]):
    """Runs the FULL pipeline (defs -> imports -> calls/names -> registry ->
    injections/roots) over a small, curated set of fixture files — mirrors
    pipeline.build_graph's stage order exactly, without going through
    ingest.load_corpus's repo-root file discovery. Returns (store, table,
    scopes)."""
    sources = [load_fixture(p) for p in rel_paths]
    table = ResolutionTable(sources)
    store = GraphStore()
    scopes = {sf.rel_path: extract_module(store, sf, table) for sf in sources}
    for sf in sources:
        extract_imports(store, sf, scopes[sf.rel_path], table, scopes)
    top_level_index = build_top_level_index(store)
    funcref_flushes = []
    for sf in sources:
        scope = scopes[sf.rel_path]
        emit_non_walk_writes(store, sf, scope)
        call_visit = make_call_visitor(store, sf, scope, table, top_level_index)
        names_visit = make_names_visitor(store, sf, scope)
        literals_visit = make_literals_visitor(store, sf, scope)
        refs_visit, refs_flush = make_funcrefs_visitor(store, sf, scope, top_level_index)
        funcref_flushes.append(refs_flush)

        def combined(node, fi, _c=call_visit, _n=names_visit, _l=literals_visit, _r=refs_visit) -> None:
            _c(node, fi)
            _n(node, fi)
            _l(node, fi)
            _r(node, fi)

        walk_module(scope, combined)
        extract_flow(store, sf, scope)
    for sf in sources:
        extract_registry_module(store, sf, scopes[sf.rel_path], top_level_index)
    extract_injections(store, sources, scopes, top_level_index)
    extract_external_roots(store, sources, top_level_index)
    for flush in funcref_flushes:
        flush()
    extract_probable_calls(store, scopes, top_level_index)
    return store, table, scopes
