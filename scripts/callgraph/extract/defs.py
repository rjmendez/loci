"""MODULE / CLASS / FUNCTION / NAME nodes, DEFINES and DECORATED_BY edges,
and decorator classification (registering | wrapping | unknown), driven by
rules.toml.

stdlib only.
"""
from __future__ import annotations

import ast
from typing import Optional

from ..ingest import SourceFile
from ..model import Confidence, Edge, GraphStore, Node
from ..resolve import ResolutionTable
from ..rules import load_rules
from ..scopes import ModuleScope


def module_node_id(rel_path: str) -> str:
    return f"mod:{rel_path}"


def function_node_id(rel_path: str, qualname: str) -> str:
    return f"fn:{rel_path}::{qualname}"


def class_node_id(rel_path: str, qualname: str) -> str:
    return f"cls:{rel_path}::{qualname}"


def name_node_id(rel_path: str, ident: str) -> str:
    return f"name:{rel_path}::{ident}"


def _decorator_head(node: ast.AST) -> str:
    """Dotted "head" of a decorator expression with call parens stripped:
    @mcp.tool() -> "mcp.tool"; @app.get("/a2a") -> "app.get"; @staticmethod
    -> "staticmethod"."""
    target = node.func if isinstance(node, ast.Call) else node
    parts: list[str] = []
    cur = target
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return "<complex>"


def extract_module(store: GraphStore, sf: SourceFile, table: Optional[ResolutionTable] = None) -> ModuleScope:
    """Populate MODULE/CLASS/FUNCTION/NAME nodes and DEFINES/DECORATED_BY
    edges for one file. Returns the ModuleScope so extract/imports.py (and
    later slices) can reuse the same single AST pass instead of re-walking."""
    mod_id = module_node_id(sf.rel_path)
    info = table.module_for_path(sf.rel_path) if table is not None else None
    mod_attrs: dict = {
        "path": sf.rel_path,
        "kind": info.kind if info is not None else ("package-member" if sf.rel_path.endswith("__init__.py") else "flat-dir"),
        "importable_as": sorted(info.importable_as.keys()) if info is not None else [],
        "sys_path_roots": (
            sorted({p.evidence for provs in info.importable_as.values() for p in provs})
            if info is not None else []
        ),
        "has_main": info.has_main if info is not None else False,
        "origin": sf.origin,
    }
    if sf.error is not None:
        mod_attrs["parse_error"] = sf.error
    store.add_node(Node(id=mod_id, kind="MODULE", attrs=mod_attrs, path=sf.rel_path, line=1, col=0))

    scope = ModuleScope(sf)
    if sf.tree is None:
        return scope

    rules = load_rules()

    # -- classes -----------------------------------------------------------
    class_id_by_qualname: dict[str, str] = {}
    for ci in scope.classes:
        cid = class_node_id(sf.rel_path, ci.qualname)
        class_id_by_qualname[ci.qualname] = cid
        store.add_node(Node(
            id=cid, kind="CLASS",
            attrs={"qualname": ci.qualname, "bases": ci.bases, "method_ids": [], "ambiguous_duplicate": ci.ambiguous_duplicate},
            path=sf.rel_path, line=ci.lineno, col=ci.node.col_offset,
        ))
        # DEFINES source is the module for every class here: nested classes
        # are not exercised by this corpus (~3 real classes total), so a
        # module-level DEFINES edge is the honest, simple choice rather
        # than a precise-but-untested nested-class parent lookup.
        store.add_edge(Edge(src=mod_id, dst=cid, kind="DEFINES", confidence=Confidence.PROVEN,
                             attrs={"line": ci.lineno}))

    # -- functions -----------------------------------------------------------
    for fi in scope.functions:
        fid = function_node_id(sf.rel_path, fi.qualname)
        param_dicts = [
            {"name": p.name, "kind": p.kind, "has_default": p.has_default, "default_is_literal": p.default_is_literal}
            for p in fi.params
        ]
        store.add_node(Node(
            id=fid, kind="FUNCTION",
            attrs={
                "qualname": fi.qualname, "is_async": fi.is_async, "is_lambda": fi.is_lambda,
                "is_method": fi.is_method, "is_nested": fi.is_nested, "params": param_dicts,
                "decorators": fi.decorators, "docstring_first_line": fi.docstring_first_line,
                "end_lineno": fi.end_lineno, "ambiguous_duplicate": fi.ambiguous_duplicate,
            },
            path=sf.rel_path, line=fi.lineno, col=fi.col,
        ))
        if fi.is_method and fi.parent_class:
            parent_cid = class_id_by_qualname.get(fi.qualname.rsplit(".", 1)[0])
            if parent_cid is not None:
                store.add_edge(Edge(src=parent_cid, dst=fid, kind="DEFINES", confidence=Confidence.PROVEN,
                                     attrs={"line": fi.lineno}))
                cls_node = store.get(parent_cid)
                if cls_node is not None:
                    cls_node.attrs.setdefault("method_ids", []).append(fid)
            else:
                store.add_edge(Edge(src=mod_id, dst=fid, kind="DEFINES", confidence=Confidence.PROVEN,
                                     attrs={"line": fi.lineno}))
        else:
            store.add_edge(Edge(src=mod_id, dst=fid, kind="DEFINES", confidence=Confidence.PROVEN,
                                 attrs={"line": fi.lineno}))

        for dec_node, dec_src in zip(fi.node.decorator_list if not fi.is_lambda else [], fi.decorators):
            head = _decorator_head(dec_node)
            classification = rules.classify_decorator(head)
            # DECORATED_BY targets a REGISTRY node once extract/registry.py
            # (a later build step) creates one for this decorator site; for
            # now it targets an EXTERNAL sink named after the decorator's
            # dotted head so the classification is still recorded and
            # queryable via `cg defs`.
            ext_id = f"ext:{head}"
            store.add_node(Node(id=ext_id, kind="EXTERNAL", attrs={"dotted": head, "role": "decorator"}))
            store.add_edge(Edge(
                src=fid, dst=ext_id, kind="DECORATED_BY",
                confidence=Confidence.PROVEN,
                attrs={"raw": dec_src, "classification": classification, "line": dec_node.lineno},
            ))

    # -- module-global NAME slots -------------------------------------------
    for ident, ninfo in scope.names.items():
        nid = name_node_id(sf.rel_path, ident)
        store.add_node(Node(
            id=nid, kind="NAME",
            attrs={
                "binding_kind": sorted(ninfo.binding_kinds),
                "binding_lines": ninfo.binding_lines,
                "is_dunder": ninfo.is_dunder,
                "is_private": ninfo.is_private,
            },
            path=sf.rel_path, line=min((min(v) for v in ninfo.binding_lines.values()), default=None),
        ))

    return scope
