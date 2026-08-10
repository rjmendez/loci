"""READS_NAME / WRITES_NAME edges, and `global`-statement handling.

WRITES_NAME is emitted with a precise `via` per occurrence (module-level-
assign | global-stmt | import-binding | def-binding | augmented) — richer
than scopes.py's coarse per-ident `binding_kind` SET, because BUG C's
diagnosis and `cg name`'s narrative both need the LINE a given write
happened at, not just "this ident was assigned somewhere".

READS_NAME is emitted for every Load-context Name that resolves to a
module-global NAME slot — this module's own (a plain global read) or, for
`mod.X` where `mod` is a module-level-imported corpus module, that module's
NAME slot (a cross-module read).

stdlib only.
"""
from __future__ import annotations

import ast
from typing import Optional

from ..ingest import SourceFile
from ..model import Confidence, Edge, GraphStore
from ..scopes import FunctionInfo, ModuleScope
from .defs import function_node_id, module_node_id, name_node_id
from .walk import walk_module


def _write(store: GraphStore, src_id: str, rel_path: str, ident: str, via: str, line: int) -> None:
    nid = name_node_id(rel_path, ident)
    if nid not in store:
        return  # defensive: every module-level/global-declared ident should already have a NAME node from defs.py
    store.add_edge(Edge(src=src_id, dst=nid, kind="WRITES_NAME", confidence=Confidence.PROVEN,
                         attrs={"via": via, "line": line}))


def _add_write_target(store: GraphStore, src_id: str, rel_path: str, target: ast.expr, via: str, line: int) -> None:
    if isinstance(target, ast.Name):
        _write(store, src_id, rel_path, target.id, via, line)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            _add_write_target(store, src_id, rel_path, elt, via, line)
    elif isinstance(target, ast.Starred):
        _add_write_target(store, src_id, rel_path, target.value, via, line)


def _emit_module_level_writes(store: GraphStore, sf: SourceFile, mod_id: str) -> None:
    rel = sf.rel_path

    def scan(body: list[ast.stmt]) -> None:
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                _write(store, mod_id, rel, stmt.name, "def-binding", stmt.lineno)
                continue
            if isinstance(stmt, ast.Assign):
                for t in stmt.targets:
                    _add_write_target(store, mod_id, rel, t, "module-level-assign", stmt.lineno)
            elif isinstance(stmt, ast.AnnAssign) and stmt.target is not None:
                _add_write_target(store, mod_id, rel, stmt.target, "module-level-assign", stmt.lineno)
            elif isinstance(stmt, ast.AugAssign):
                _add_write_target(store, mod_id, rel, stmt.target, "augmented", stmt.lineno)
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    _write(store, mod_id, rel, bound, "import-binding", stmt.lineno)
            elif isinstance(stmt, ast.ImportFrom):
                for alias in stmt.names:
                    if alias.name == "*":
                        continue
                    bound = alias.asname or alias.name
                    _write(store, mod_id, rel, bound, "import-binding", stmt.lineno)
            elif isinstance(stmt, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                for _field, value in ast.iter_fields(stmt):
                    if isinstance(value, list):
                        nested = [v for v in value if isinstance(v, ast.stmt)]
                        if nested:
                            scan(nested)

    if sf.tree is not None:
        scan(sf.tree.body)


def _emit_global_stmt_writes(store: GraphStore, sf: SourceFile, fi: FunctionInfo) -> None:
    global_names: set[str] = set()
    for node in ast.walk(fi.node):
        if isinstance(node, ast.Global):
            global_names.update(node.names)
    if not global_names:
        return
    rel = sf.rel_path
    fn_id = function_node_id(rel, fi.qualname)

    def add_target(t: ast.expr, line: int) -> None:
        if isinstance(t, ast.Name):
            if t.id in global_names:
                _write(store, fn_id, rel, t.id, "global-stmt", line)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for elt in t.elts:
                add_target(elt, line)
        elif isinstance(t, ast.Starred):
            add_target(t.value, line)

    def scan(node: ast.AST) -> None:
        if node is not fi.node and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if isinstance(node, ast.Assign):
            for t in node.targets:
                add_target(t, node.lineno)
        elif isinstance(node, ast.AnnAssign) and node.target is not None:
            add_target(node.target, node.lineno)
        elif isinstance(node, ast.AugAssign):
            add_target(node.target, node.lineno)
        for child in ast.iter_child_nodes(node):
            scan(child)

    for stmt in fi.node.body:
        scan(stmt)


def _handle_name_load(store: GraphStore, sf: SourceFile, scope: ModuleScope, mod_id: str,
                       node: ast.Name, fi: Optional[FunctionInfo], call_func_ids: set[int]) -> None:
    ident = node.id
    if fi is not None:
        locs = scope.locals_of(fi)
        if ident in locs.nonlocal_declared:
            return
        if ident in locs.effective_locals:
            return
    nid = name_node_id(sf.rel_path, ident)
    if nid not in store:
        return
    src_id = function_node_id(sf.rel_path, fi.qualname) if fi is not None else mod_id
    store.add_edge(Edge(src=src_id, dst=nid, kind="READS_NAME", confidence=Confidence.PROVEN, attrs={
        "line": node.lineno, "col": node.col_offset,
        "in_call_position": id(node) in call_func_ids,
    }))


def _handle_attribute_load(store: GraphStore, sf: SourceFile, mod_id: str,
                            node: ast.Attribute, fi: Optional[FunctionInfo]) -> None:
    base = node.value
    if not isinstance(base, ast.Name):
        return
    alias_edges = store.out_edges(name_node_id(sf.rel_path, base.id), "ALIASES")
    if not alias_edges:
        return
    target = store.get(alias_edges[0].dst)
    if target is None or target.kind != "MODULE" or target.path is None:
        return
    target_nid = name_node_id(target.path, node.attr)
    if target_nid not in store:
        return
    src_id = function_node_id(sf.rel_path, fi.qualname) if fi is not None else mod_id
    store.add_edge(Edge(src=src_id, dst=target_nid, kind="READS_NAME", confidence=Confidence.PROVEN, attrs={
        "line": node.lineno, "col": node.col_offset, "in_call_position": False, "cross_module": True,
    }))


def emit_non_walk_writes(store: GraphStore, sf: SourceFile, scope: ModuleScope) -> None:
    """The WRITES_NAME edges that don't need the shared whole-module walk
    (module-level bindings, and each function's own `global`-declared
    assignments) — cheap, file-local scans, run separately from
    make_names_visitor's walk_module pass."""
    if sf.tree is None:
        return
    mod_id = module_node_id(sf.rel_path)
    _emit_module_level_writes(store, sf, mod_id)
    for fi in scope.functions:
        if fi.is_lambda:
            continue
        _emit_global_stmt_writes(store, sf, fi)


def make_names_visitor(store: GraphStore, sf: SourceFile, scope: ModuleScope):
    """Builds the walk_module(node, fi) callback for READS_NAME (both
    same-module and cross-module) without walking anything itself — see
    calls.py's make_call_visitor for why pipeline.py shares one traversal
    between the two."""
    mod_id = module_node_id(sf.rel_path)
    call_func_ids = {id(c.func) for c in ast.walk(sf.tree) if isinstance(c, ast.Call)} if sf.tree is not None else set()

    def on_node(node: ast.AST, fi: Optional[FunctionInfo]) -> None:
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            _handle_name_load(store, sf, scope, mod_id, node, fi, call_func_ids)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            _handle_attribute_load(store, sf, mod_id, node, fi)

    return on_node


def extract_names(store: GraphStore, sf: SourceFile, scope: ModuleScope) -> None:
    """Standalone entry point (unit tests, `cg` debug use) — pipeline.py
    calls emit_non_walk_writes + make_names_visitor directly instead, to
    share one walk_module pass with extract/calls.py."""
    if sf.tree is None:
        return
    emit_non_walk_writes(store, sf, scope)
    walk_module(scope, make_names_visitor(store, sf, scope))
