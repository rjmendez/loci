"""One shared whole-module AST walk that tracks "am I at module level or
inside which FUNCTION" as it goes, used by extract/calls.py, extract/names.py
and extract/registry.py's REG-FN pass so all three attribute nodes to the
same enclosing scope consistently instead of each re-deriving it (which risks
disagreeing on ambiguous-duplicate qualnames).

stdlib only.
"""
from __future__ import annotations

import ast
from typing import Callable, Optional

from ..scopes import FunctionInfo, ModuleScope

OnNode = Callable[[ast.AST, Optional[FunctionInfo]], None]


def walk_module(scope: ModuleScope, on_node: OnNode) -> None:
    """Visit every node reachable from the module body exactly once,
    calling on_node(node, enclosing_fi) — enclosing_fi is None at module
    level, else the FunctionInfo (function, method, or lambda) the node
    lexically sits inside, per ModuleScope's own qualname assignment."""
    tree = scope.sf.tree
    if tree is None:
        return

    def visit(node: ast.AST, current: Optional[FunctionInfo]) -> None:
        on_node(node, current)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            qual = scope.qualname_for_node(node)
            fi = scope.function_by_qualname.get(qual) if qual else None
            nxt = fi if fi is not None else current
            for child in ast.iter_child_nodes(node):
                visit(child, nxt)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, current)

    for stmt in tree.body:
        visit(stmt, None)
