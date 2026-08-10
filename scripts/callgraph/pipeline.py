"""Orchestrates ingest -> resolve -> extract into one GraphStore. cli.py
calls this; nothing else should hand-roll the build sequence.

stdlib only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .extract.calls import build_top_level_index, make_call_visitor
from .extract.defs import extract_module
from .extract.dispatch import extract_probable_calls
from .extract.flow import extract_flow
from .extract.funcrefs import make_funcrefs_visitor
from .extract.imports import extract_imports
from .extract.literals import make_literals_visitor
from .extract.names import emit_non_walk_writes, make_names_visitor
from .extract.registry import extract_external_roots, extract_injections, extract_registry_module
from .extract.walk import walk_module
from .ingest import SourceFile, load_corpus
from .model import GraphStore
from .resolve import ResolutionTable
from .scopes import ModuleScope


@dataclass
class BuildMeta:
    source: str                 # "working tree" or "rev <sha>"
    file_count: int
    error_count: int
    errors: list[tuple[str, str]] = field(default_factory=list)
    elapsed_s: float = 0.0
    scope_prefixes: Optional[list[str]] = None


@dataclass
class BuildResult:
    store: GraphStore
    table: ResolutionTable
    scopes: dict[str, ModuleScope]
    sources: list[SourceFile]
    meta: BuildMeta


def _filter_scope(sources: list[SourceFile], scope_prefixes: Optional[list[str]]) -> list[SourceFile]:
    if not scope_prefixes:
        return sources
    return [sf for sf in sources if any(sf.rel_path.startswith(p) for p in scope_prefixes)]


def build_graph(rev: Optional[str] = None, scope_prefixes: Optional[list[str]] = None,
                 no_cache: bool = False) -> BuildResult:
    t0 = time.time()
    sources, origin = load_corpus(rev=rev, no_cache=no_cache)
    sources = _filter_scope(sources, scope_prefixes)

    table = ResolutionTable(sources)
    store = GraphStore()

    scopes: dict[str, ModuleScope] = {}
    errors: list[tuple[str, str]] = []
    for sf in sources:
        scopes[sf.rel_path] = extract_module(store, sf, table)
        if sf.error is not None:
            errors.append((sf.rel_path, sf.error))

    for sf in sources:
        extract_imports(store, sf, scopes[sf.rel_path], table, scopes)

    # Everything from here on needs every module's FUNCTION/CLASS/ALIASES
    # already in the store, so it runs as its own whole-corpus pass rather
    # than being folded into the per-file loops above.
    top_level_index = build_top_level_index(store)

    # extract/calls.py and extract/names.py each only need a callback
    # (they don't own the traversal) so both run in ONE whole-module walk
    # per file here, instead of two — see calls.py's make_call_visitor.
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

    # AFTER registry: the generic name-escape pass defers to the specific
    # registration forms above for any reference they already modelled.
    for flush in funcref_flushes:
        flush()

    # The probable tier (CALLS rungs 4-6 + DISPATCHES) needs REGISTRY/
    # REGISTERS/INJECTS already built, so it runs last, only ever touching
    # CALLS edges rungs 1-3 left on the `?` sink.
    extract_probable_calls(store, scopes, top_level_index)

    meta = BuildMeta(
        source=origin, file_count=len(sources), error_count=len(errors),
        errors=errors, elapsed_s=time.time() - t0, scope_prefixes=scope_prefixes,
    )
    return BuildResult(store=store, table=table, scopes=scopes, sources=sources, meta=meta)
