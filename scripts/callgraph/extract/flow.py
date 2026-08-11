"""Escaping-local detection and the lightweight per-function control-flow
summary `cg flags` ranks over.

Deliberately NOT a real CFG: a linear walk with a context stack (one tag
per statement — top-level-of-body | if-branch | loop-body | except-handler
| with-body; a `try:` body and its `else`/`finally` are pass-through, since
they don't branch), because a real CFG buys precision this corpus's bug
shapes do not need. THIS IS BUG B's data: `mcp/grounding.py:ground`'s
`degraded` is a LOCALBINDING — init'd `False` at the function's top level, a
dict-value escape at `return {..., "degraded": degraded}`, reassigned
`True` only inside a couple of `except`/`elif` branches, while several
`if not S: break`-shaped branches exit the function's enclosing loop
without ever touching it.

A LOCALBINDING is materialized ONLY for locals that escape (returned
directly, packed into a returned dict/tuple/list, or captured by a closure
via `nonlocal` or a Load inside a nested def/lambda that itself escapes) —
see the design's own framing: "an unassigned local that never escapes is
nobody's bug."

`guard_exits`: a `return`/`break`/`continue` is counted as a guard-that-
skips-the-flag when it sits DIRECTLY inside a non-top-level block (an
if/elif/else/for/while/except/with body) that does not ALSO directly
assign the ident anywhere in that same block — "directly" meaning at that
block's own statement list, not further nested inside a sub-branch of it
(a further-nested sub-branch is its own, separately-judged block). This is
exactly the shape of `for cid in ...: if not S: break` repeated across
several lanes in `ground()`.

stdlib only.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Optional

from ..ingest import SourceFile
from ..model import Confidence, Edge, GraphStore, Node
from ..scopes import FunctionInfo, ModuleScope
from .defs import function_node_id

TOP_LEVEL = "top-level-of-body"
_EXIT_TYPES = (ast.Return, ast.Break, ast.Continue)
_NESTED_SCOPE_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def local_binding_id(fn_id: str, ident: str) -> str:
    return f"local:{fn_id}::{ident}"


# -- block walk: assign/exit events, block-local -----------------------------


@dataclass
class _AssignEvent:
    ident: str
    line: int
    tag: str
    is_constant: bool
    block: int
    form: str = "assign"   # assign | ann-assign | augmented | for-target | with-target


@dataclass
class _ExitEvent:
    line: int
    kind: str          # "return" | "break" | "continue"
    tag: str
    block: int


def _assign_targets(stmt: ast.stmt) -> list[ast.expr]:
    targets: list[ast.expr] = []
    if isinstance(stmt, ast.Assign):
        targets = list(stmt.targets)
    elif isinstance(stmt, ast.AnnAssign) and stmt.target is not None:
        targets = [stmt.target]
    elif isinstance(stmt, ast.AugAssign):
        targets = [stmt.target]
    elif isinstance(stmt, (ast.For, ast.AsyncFor)):
        targets = [stmt.target]
    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
        targets = [w.optional_vars for w in stmt.items if w.optional_vars is not None]
    out: list[ast.expr] = []

    def flatten(t: ast.expr) -> None:
        if isinstance(t, ast.Name):
            out.append(t)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                flatten(e)
        elif isinstance(t, ast.Starred):
            flatten(t.value)

    for t in targets:
        flatten(t)
    return out


def _is_constant_value(stmt: ast.stmt) -> bool:
    value = getattr(stmt, "value", None)
    return isinstance(value, ast.Constant)


def _assign_form(stmt: ast.stmt) -> str:
    """How the binding was made. The distinction that matters downstream is
    REBINDING (`x = <const>`) vs ACCUMULATING (`x += ...`): a counter is
    *supposed* to be updated on only some paths, so scoring it like a
    conditionally-set flag is a false positive by construction. See
    analyze/flags.py."""
    if isinstance(stmt, ast.AugAssign):
        return "augmented"
    if isinstance(stmt, ast.AnnAssign):
        return "ann-assign"
    if isinstance(stmt, (ast.For, ast.AsyncFor)):
        return "for-target"
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return "with-target"
    return "assign"


def _walk_blocks(stmts: list[ast.stmt], tag: str, assigns: list[_AssignEvent], exits: list[_ExitEvent],
                  block_directly_assigns: dict[int, set[str]]) -> None:
    bid = id(stmts)
    directly: set[str] = set()
    for stmt in stmts:
        for name_node in _assign_targets(stmt):
            directly.add(name_node.id)
            assigns.append(_AssignEvent(ident=name_node.id, line=stmt.lineno, tag=tag,
                                         is_constant=_is_constant_value(stmt), block=bid,
                                         form=_assign_form(stmt)))
        if isinstance(stmt, _EXIT_TYPES):
            kind = {"Return": "return", "Break": "break", "Continue": "continue"}[type(stmt).__name__]
            exits.append(_ExitEvent(line=stmt.lineno, kind=kind, tag=tag, block=bid))
    block_directly_assigns[bid] = directly

    for stmt in stmts:
        if isinstance(stmt, ast.If):
            _walk_blocks(stmt.body, "if-branch", assigns, exits, block_directly_assigns)
            _walk_blocks(stmt.orelse, "if-branch", assigns, exits, block_directly_assigns)
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            _walk_blocks(stmt.body, "loop-body", assigns, exits, block_directly_assigns)
            _walk_blocks(stmt.orelse, "loop-body", assigns, exits, block_directly_assigns)
        elif isinstance(stmt, ast.While):
            _walk_blocks(stmt.body, "loop-body", assigns, exits, block_directly_assigns)
            _walk_blocks(stmt.orelse, "loop-body", assigns, exits, block_directly_assigns)
        elif isinstance(stmt, ast.Try):
            _walk_blocks(stmt.body, tag, assigns, exits, block_directly_assigns)
            for h in stmt.handlers:
                _walk_blocks(h.body, "except-handler", assigns, exits, block_directly_assigns)
            _walk_blocks(stmt.orelse, tag, assigns, exits, block_directly_assigns)
            _walk_blocks(stmt.finalbody, tag, assigns, exits, block_directly_assigns)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            _walk_blocks(stmt.body, "with-body", assigns, exits, block_directly_assigns)
        # nested def/class/lambda: own scope, not walked here (closure
        # escape detection below handles them separately and deliberately).


# -- escape detection ---------------------------------------------------------


def _direct_return_idents(value: ast.expr) -> set[str]:
    return {value.id} if isinstance(value, ast.Name) else set()


def _container_element_idents(value: ast.expr) -> tuple[set[str], set[str]]:
    """Returns (dict_value_idents, tuple_or_list_element_idents) for a
    return value shaped like a dict/call-with-kwargs literal, or a
    tuple/list literal — one level deep, matching the corpus's actual
    `return {"a": x, "b": y}` / `return (a, b)` shapes."""
    dict_idents: set[str] = set()
    seq_idents: set[str] = set()
    if isinstance(value, ast.Dict):
        for v in value.values:
            if isinstance(v, ast.Name):
                dict_idents.add(v.id)
    elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "dict":
        for kw in value.keywords:
            if kw.arg is not None and isinstance(kw.value, ast.Name):
                dict_idents.add(kw.value.id)
    elif isinstance(value, (ast.Tuple, ast.List)):
        for elt in value.elts:
            if isinstance(elt, ast.Name):
                seq_idents.add(elt.id)
    return dict_idents, seq_idents


def _find_returns(body: list[ast.stmt]) -> list[ast.Return]:
    out: list[ast.Return] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, _NESTED_SCOPE_TYPES):
            return
        if isinstance(node, ast.Return) and node.value is not None:
            out.append(node)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for stmt in body:
        visit(stmt)
    return out


def _find_nested_defs(body: list[ast.stmt]) -> list:
    out = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            out.append(node)
            return  # don't look inside nested-of-nested for THIS function's escape analysis
        if isinstance(node, ast.ClassDef):
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    for stmt in body:
        visit(stmt)
    return out


def _nested_def_bound_name(node) -> Optional[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name
    return None


def _escapes(fi: FunctionInfo) -> dict[str, tuple[int, str]]:
    """ident -> (escape_line, escape_form). form in {direct-return,
    dict-value, tuple-element, closure}. First escape found wins."""
    body = fi.node.body if not isinstance(fi.node, ast.Lambda) else [ast.Return(value=fi.node.body, lineno=fi.node.lineno)]
    out: dict[str, tuple[int, str]] = {}

    escaping_nested_names: set[str] = set()

    for ret in _find_returns(body):
        value = ret.value
        for ident in _direct_return_idents(value):
            out.setdefault(ident, (ret.lineno, "direct-return"))
        dict_idents, seq_idents = _container_element_idents(value)
        for ident in dict_idents:
            out.setdefault(ident, (ret.lineno, "dict-value"))
        for ident in seq_idents:
            out.setdefault(ident, (ret.lineno, "tuple-element"))
        # a nested callable escaping by name (`return inner`, or packed into
        # the same container shapes) — its free variables escape via closure.
        candidates = {value.id} if isinstance(value, ast.Name) else set()
        candidates |= dict_idents | seq_idents
        escaping_nested_names |= candidates

    # closure: a nested def/lambda whose bound name is in escaping_nested_names
    # (or a lambda that is itself the direct return value / container element —
    # matched by identity against the AST nodes found in the return values).
    nested_defs = _find_nested_defs(body)
    escaping_lambda_nodes: set[int] = set()
    for ret in _find_returns(body):
        value = ret.value
        if isinstance(value, ast.Lambda):
            escaping_lambda_nodes.add(id(value))
        dict_vals = value.values if isinstance(value, ast.Dict) else []
        seq_elts = value.elts if isinstance(value, (ast.Tuple, ast.List)) else []
        for v in list(dict_vals) + list(seq_elts):
            if isinstance(v, ast.Lambda):
                escaping_lambda_nodes.add(id(v))

    for nd in nested_defs:
        bound = _nested_def_bound_name(nd)
        is_escaping = (bound is not None and bound in escaping_nested_names) or id(nd) in escaping_lambda_nodes
        if not is_escaping:
            continue
        inner_locals: set[str] = {a.arg for a in nd.args.args} | {a.arg for a in nd.args.posonlyargs} \
            | {a.arg for a in nd.args.kwonlyargs}
        if nd.args.vararg:
            inner_locals.add(nd.args.vararg.arg)
        if nd.args.kwarg:
            inner_locals.add(nd.args.kwarg.arg)
        nonlocal_names: set[str] = set()
        for n in ast.walk(nd):
            if isinstance(n, ast.Nonlocal):
                nonlocal_names.update(n.names)
            elif isinstance(n, _NESTED_SCOPE_TYPES) and n is not nd:
                continue
        loads: set[str] = set()
        for n in ast.walk(nd if isinstance(nd, ast.Lambda) else nd):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                loads.add(n.id)
        for ident in loads - inner_locals:
            out.setdefault(ident, (nd.lineno, "closure"))
        for ident in nonlocal_names:
            out.setdefault(ident, (nd.lineno, "closure"))

    return out


# -- public entry point --------------------------------------------------


def analyze_function(fi: FunctionInfo) -> dict[str, dict]:
    """Returns ident -> {escape_line, escape_form, init_line, init_is_constant,
    assign_lines, assign_contexts, guard_exits} for every local that escapes
    fi. Non-escaping locals are never included — see module docstring."""
    if fi.is_lambda:
        body = [ast.Return(value=fi.node.body, lineno=fi.node.lineno)]
    else:
        body = fi.node.body

    escapes = _escapes(fi)
    if not escapes:
        return {}

    assigns: list[_AssignEvent] = []
    exits: list[_ExitEvent] = []
    block_directly_assigns: dict[int, set[str]] = {}
    _walk_blocks(body, TOP_LEVEL, assigns, exits, block_directly_assigns)

    out: dict[str, dict] = {}
    for ident, (escape_line, escape_form) in escapes.items():
        ident_assigns = sorted((a for a in assigns if a.ident == ident), key=lambda a: a.line)
        if not ident_assigns and escape_form != "closure":
            # A bare parameter, or a nested def's own name, returned as-is:
            # never locally (re)bound in THIS function's own body, so it
            # isn't a "local" in the sense LOCALBINDING models — see
            # module docstring. A `nonlocal`-captured ident can legitimately
            # have zero assigns in ITS OWN frame (the assignment lives in
            # the closure), so escape_form=="closure" is exempted.
            continue
        init_line = None
        init_is_constant = False
        reassigns: list[_AssignEvent] = ident_assigns
        if ident_assigns and ident_assigns[0].tag == TOP_LEVEL:
            init_line = ident_assigns[0].line
            init_is_constant = ident_assigns[0].is_constant
            reassigns = ident_assigns[1:]

        guard_exits: list[dict] = []
        for ex in exits:
            if ex.tag == TOP_LEVEL:
                continue
            if ident in block_directly_assigns.get(ex.block, set()):
                continue
            guard_exits.append({"line": ex.line, "kind": ex.kind, "context": ex.tag})

        # A REBIND is a plain `ident = <constant>` reassignment — the only
        # shape that actually flips a flag. `+=` on a counter, a for-target,
        # or a rebind to a non-constant expression are all excluded, because
        # each of them is normal code that the guard-exit ratio would
        # otherwise score exactly like a conditionally-set flag.
        constant_rebinds = [a for a in reassigns
                            if a.form in ("assign", "ann-assign") and a.is_constant]

        out[ident] = {
            "escape_line": escape_line, "escape_form": escape_form,
            "init_line": init_line, "init_is_constant": init_is_constant,
            "assign_lines": [a.line for a in reassigns],
            "assign_contexts": [{"line": a.line, "context": a.tag, "form": a.form,
                                  "is_constant": a.is_constant} for a in reassigns],
            "constant_rebind_lines": [a.line for a in constant_rebinds],
            "assign_forms": sorted({a.form for a in reassigns}),
            "guard_exits": guard_exits,
        }
    return out


def extract_flow(store: GraphStore, sf: SourceFile, scope: ModuleScope) -> None:
    if sf.tree is None:
        return
    for fi in scope.functions:
        if fi.is_method and fi.qualname.rsplit(".", 1)[-1] == "__init__":
            pass  # __init__ is analyzed like any other method; no special-case needed
        summary = analyze_function(fi)
        if not summary:
            continue
        fn_id = function_node_id(sf.rel_path, fi.qualname)
        for ident, info in summary.items():
            lid = local_binding_id(fn_id, ident)
            store.add_node(Node(
                id=lid, kind="LOCALBINDING",
                attrs={
                    "ident": ident, "init_line": info["init_line"], "init_is_constant": info["init_is_constant"],
                    "assign_lines": info["assign_lines"], "assign_contexts": info["assign_contexts"],
                    "constant_rebind_lines": info["constant_rebind_lines"],
                    "assign_forms": info["assign_forms"],
                    "escape_lines": [info["escape_line"]], "guard_exits": info["guard_exits"],
                },
                path=sf.rel_path, line=(info["init_line"] or info["escape_line"]),
            ))
            store.add_edge(Edge(src=fn_id, dst=lid, kind="ESCAPES", confidence=Confidence.PROVEN, attrs={
                "escape_line": info["escape_line"], "escape_form": info["escape_form"],
            }))
