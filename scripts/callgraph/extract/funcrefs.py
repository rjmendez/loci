"""Bare function-name ESCAPES -> REFERENCES edges.

A function whose NAME appears anywhere in a value position -- passed as a
call argument, returned, stored in a container, bound to a variable -- has
escaped. Whoever holds the reference can call it, and this tool has no way
to prove they don't. For `cg dead` that is decisive: without this pass such
a function looks unreachable and gets reported dead, which is a FALSE
POSITIVE and the single fastest way to make people stop trusting the tool.

Found by validation against the real corpus at HEAD. Two shapes in
mcp/server.py's health block, both previously reported dead:

    checks.append(_health_check("embeddings_sparse", _health_probe_embeddings_sparse))
                                                     ^ bare name, call argument

    _collect_decls(root, lang, def_types, enclosing_symbol_src, in_method, add_decl)
                                                                           ^ nested
                                                       def, passed down and called
                                                       through a parameter

extract/registry.py already emitted REFERENCES for the two REGISTRATION
shapes it models (`manifest-tuple-element`, `manifest-dict-value`). This
generalizes the same edge to every escape, registration or not.

DELIBERATELY BROAD, and it costs recall: a function referenced only from
code that itself never runs still counts as referenced here, so genuinely
dead code can be hidden by a single stale mention. That is the intended
trade -- `cg dead` is a lead generator whose value is that its rows are
worth reading, and a tool people suppress is a tool that is not working.
See docs/LIMITS.md.

Only BARE NAMES are resolved. `mod.func` in a value position is not modelled
(it would need the same receiver-typing the CALLS ladder gives up on at
rung 4), so it remains a known blind spot rather than a guess.

stdlib only.
"""
from __future__ import annotations

import ast
from typing import Optional

from ..ingest import SourceFile
from ..model import Confidence, Edge, GraphStore
from ..scopes import FunctionInfo, ModuleScope
from .defs import function_node_id, module_node_id


def _resolve_local_nested(scope: ModuleScope, fi: Optional[FunctionInfo], name: str) -> Optional[str]:
    """A nested def visible by bare name from inside `fi` -- i.e. defined in
    fi itself or in one of its lexical ancestors. Walks outward through the
    `<parent>.<locals>.<child>` qualname chain the scope builder assigns."""
    if fi is None:
        return None
    qual = fi.qualname
    while True:
        cand = f"{qual}.<locals>.{name}"
        target = scope.function_by_qualname.get(cand)
        if target is not None and not target.is_lambda:
            return function_node_id(scope.sf.rel_path, cand)
        if ".<locals>." not in qual:
            return None
        qual = qual.rsplit(".<locals>.", 1)[0]


def make_funcrefs_visitor(store: GraphStore, sf: SourceFile, scope: ModuleScope, top_level_index):
    """Returns (visit, flush).

    `visit` rides along on the ONE shared whole-module walk that calls.py /
    names.py / literals.py already run (see extract/walk.py) — a separate
    traversal here measurably blew the build-time budget on this corpus.

    `flush` emits the edges, and must run AFTER extract/registry.py so the
    specific registration forms win the dedupe below.
    """
    mod_path = sf.rel_path
    module_level = top_level_index.get(mod_path, {})
    skip: set[int] = set()
    pending: list[tuple[str, str, int]] = []

    def visit(node: ast.AST, fi: Optional[FunctionInfo]) -> None:
        # walk_module visits a parent before its children, so a Call is always
        # seen before its own `.func`, and a def before its decorator list.
        # Recording the ids here is therefore enough to skip them below.
        if isinstance(node, ast.Call):
            skip.add(id(node.func))
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for d in node.decorator_list:
                skip.add(id(d))          # @foo / @foo(...) is DECORATED_BY, not an escape
            return
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            return
        if id(node) in skip:
            return
        target_id = _resolve_local_nested(scope, fi, node.id) or module_level.get(node.id)
        if target_id is None:
            return
        target = store.get(target_id)
        if target is None or target.kind != "FUNCTION":
            return
        src_id = (function_node_id(mod_path, fi.qualname) if fi is not None
                  else module_node_id(mod_path))
        if src_id == target_id:
            return  # a recursive self-mention is not an escape
        pending.append((src_id, target_id, node.lineno))

    def flush() -> None:
        for src_id, target_id, line in pending:
            # extract/registry.py models some of these with a more specific
            # form (`manifest-tuple-element`, `manifest-dict-value`,
            # `injected-argument`). Same source, same target, same LINE is the
            # same syntactic reference — re-emitting it as a generic
            # name-escape would double-count it in `cg callers`.
            dup = False
            for existing in store.out_edges(src_id, "REFERENCES"):
                if existing.dst == target_id and existing.attrs.get("line") == line:
                    dup = True
                    break
            if dup:
                continue
            store.add_edge(Edge(src=src_id, dst=target_id, kind="REFERENCES",
                                 confidence=Confidence.PROVEN,
                                 attrs={"form": "name-escape", "line": line}))

    return visit, flush
