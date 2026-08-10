"""LITERAL / PATHEXPR extraction, and PRODUCES_LITERAL / CONSUMES_LITERAL
edges — the data BUG D's diagnosis needs: does any consumer's path/key
literal agree with a producer's, across the whole corpus?

Two literal flavours (design's own split):
  PATH-LIKE — contains `/`, starts with `~`, or ends with a recognized file
              extension.
  KEY-LIKE  — an identifier-shaped string (env var name, dict lookup key,
              collection/table name) used in a lookup position.

Role (produce | consume) comes from the syntactic context the literal (or
the PATHEXPR it's the tail of) sits in — a small, explicit verb table
matched against the call's dotted name or method name, kept in this module
rather than rules.toml: unlike the decorator/registrar rules (which grow as
the corpus adds new frameworks), the verb set here is closed and small
(open/write_text/mkdir/glob/read_text/exists/environ.get/dict.get) and a
change to it is a code review of THIS logic, not a data tweak an unrelated
engineer should be able to make blind.

A composed path (`MEMORY_DIR / "graph.ladybug"`, `os.path.join(...)`, an
f-string, `+` concat) becomes a PATHEXPR node instead of a bare LITERAL —
PRODUCES_LITERAL/CONSUMES_LITERAL edges target whichever of the two applies
at that site. Its `tail_literal` (the last literal segment) is what a
cross-module orphan/near-miss comparison actually keys on: `graph.ladybug`
in `mcp/server.py`'s `MEMORY_DIR / "graph.ladybug"` PATHEXPR is compared
against the bare LITERAL `~/.hermes/**/graph.kuzu` in
`scripts/graph_facts.py` on exactly that basis (see analyze/literalaudit.py).

stdlib only.
"""
from __future__ import annotations

import ast
import hashlib
import re
from typing import Optional

from ..ingest import SourceFile
from ..model import Confidence, Edge, GraphStore, Node
from ..scopes import FunctionInfo, ModuleScope
from .calls import _dotted_call_name
from .defs import function_node_id, module_node_id
from .walk import walk_module

# -- ids ----------------------------------------------------------------


def literal_node_id(normalized: str) -> str:
    digest = hashlib.sha1(normalized.encode("utf-8", errors="surrogateescape")).hexdigest()
    return f"lit:{digest[:16]}"


def pathexpr_node_id(rel_path: str, line: int, col: int) -> str:
    return f"pathexpr:{rel_path}:{line}:{col}"


# -- flavour / normalization ---------------------------------------------

_PATH_EXT_SUFFIXES = (
    ".py", ".json", ".jsonl", ".toml", ".yaml", ".yml", ".db", ".kuzu", ".ladybug",
    ".md", ".txt", ".csv", ".service", ".sh", ".log", ".cfg", ".ini", ".pyc", ".lock",
    ".pid", ".sock",
)
_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]{0,63}$")


def looks_path_like(s: str) -> bool:
    if not s or "\n" in s or len(s) > 400:
        return False
    if "/" in s or s.startswith("~"):
        return True
    return s.endswith(_PATH_EXT_SUFFIXES)


def looks_key_like(s: str) -> bool:
    if not s or "\n" in s or len(s) > 128:
        return False
    if looks_path_like(s):
        return False
    return bool(_KEY_RE.match(s))


def classify_flavour(s: str) -> Optional[str]:
    if looks_path_like(s):
        return "path"
    if looks_key_like(s):
        return "key"
    return None


def normalize_literal(raw: str) -> str:
    """expanduser-strip's SPIRIT without touching the filesystem: `~` stays
    a marker (this tool never resolves a real home dir), and a recursive
    glob's `**` collapses to a single `*` so `~/.hermes/**/graph.kuzu` and a
    hypothetical single-star sibling normalize the same way. Case is
    preserved per design."""
    return raw.replace("**", "*")


def tail_segment(s: str) -> str:
    s2 = s.rstrip("/")
    if "/" in s2:
        return s2.rsplit("/", 1)[-1]
    return s2


def _stem(tail: str) -> str:
    base = tail.rsplit("/", 1)[-1]
    if "." in base and not base.startswith("."):
        return base.rsplit(".", 1)[0].lower()
    return base.lower()


# -- node helpers ----------------------------------------------------------


def _ensure_literal(store: GraphStore, raw: str, flavour: str) -> str:
    normalized = normalize_literal(raw)
    lid = literal_node_id(normalized)
    node = store.get(lid)
    if node is None:
        store.add_node(Node(id=lid, kind="LITERAL", attrs={
            "raw": raw, "normalized": normalized, "flavour": flavour,
            "tail": tail_segment(normalized), "occurrences": [],
        }))
        node = store.get(lid)
    return lid


def _record_occurrence(store: GraphStore, node_id: str, rel_path: str, line: int, col: int, role: str) -> None:
    node = store.get(node_id)
    if node is not None:
        node.attrs.setdefault("occurrences", []).append(
            {"path": rel_path, "line": line, "col": col, "role": role}
        )


def _emit_edge(store: GraphStore, src_id: str, dst_id: str, direction: str,
                rel_path: str, line: int, col: int, role: str) -> None:
    kind = "PRODUCES_LITERAL" if direction == "produce" else "CONSUMES_LITERAL"
    store.add_edge(Edge(src=src_id, dst=dst_id, kind=kind, confidence=Confidence.PROVEN,
                         attrs={"role": role, "line": line, "col": col, "path": rel_path}))
    _record_occurrence(store, dst_id, rel_path, line, col, role)


# -- PATHEXPR composition ---------------------------------------------------

_WRAPPER_TAILS = {"str", "path", "fspath", "repr", "expanduser", "abspath", "normpath", "realpath"}


def _unwrap(expr: ast.expr) -> ast.expr:
    while isinstance(expr, ast.Call) and len(expr.args) == 1 and not expr.keywords:
        dotted = _dotted_call_name(expr.func) or ""
        tail = dotted.rsplit(".", 1)[-1].lower()
        if tail in _WRAPPER_TAILS:
            expr = expr.args[0]
            continue
        break
    return expr


def _flatten_binop(expr: ast.expr, op_type) -> Optional[list[ast.expr]]:
    if not (isinstance(expr, ast.BinOp) and isinstance(expr.op, op_type)):
        return None
    left = _flatten_binop(expr.left, op_type)
    operands = left if left is not None else [expr.left]
    operands.append(expr.right)
    return operands


def _segment(expr: ast.expr, scope: ModuleScope) -> dict:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return {"kind": "literal", "value": expr.value}
    if isinstance(expr, ast.Name):
        return {"kind": "name", "value": expr.id}
    return {"kind": "?", "value": scope.source_of(expr)}


def _try_build_pathexpr(store: GraphStore, sf: SourceFile, scope: ModuleScope, expr: ast.expr) -> Optional[str]:
    segments: Optional[list[dict]] = None

    div_chain = _flatten_binop(expr, ast.Div)
    if div_chain is not None and len(div_chain) >= 2:
        segments = [_segment(o, scope) for o in div_chain]
    else:
        add_chain = _flatten_binop(expr, ast.Add)
        if add_chain is not None and len(add_chain) >= 2 and all(
            isinstance(o, ast.Constant) and isinstance(o.value, str) for o in add_chain
        ):
            segments = [_segment(o, scope) for o in add_chain]
        elif isinstance(expr, ast.Call):
            dotted = (_dotted_call_name(expr.func) or "").rsplit(".", 1)[-1]
            if dotted == "join" and expr.args:
                segments = [_segment(a, scope) for a in expr.args]
        elif isinstance(expr, ast.JoinedStr):
            segs = []
            for v in expr.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    segs.append({"kind": "literal", "value": v.value})
                elif isinstance(v, ast.FormattedValue):
                    segs.append(_segment(v.value, scope))
            if segs:
                segments = segs

    if segments is None:
        return None

    literal_segs = [s for s in segments if s["kind"] == "literal"]
    if not literal_segs:
        return None
    tail_literal = literal_segs[-1]["value"]

    pid = pathexpr_node_id(sf.rel_path, expr.lineno, expr.col_offset)
    if pid not in store:
        store.add_node(Node(
            id=pid, kind="PATHEXPR",
            attrs={"segments": segments, "tail_literal": tail_literal, "occurrences": []},
            path=sf.rel_path, line=expr.lineno, col=expr.col_offset,
        ))
    return pid


def _handle_composed_or_literal(store: GraphStore, sf: SourceFile, scope: ModuleScope,
                                 candidate: ast.expr, expected_flavour: str, role: str, direction: str,
                                 src_id: str, site_line: int, site_col: int) -> None:
    unwrapped = _unwrap(candidate)
    if isinstance(unwrapped, ast.Constant) and isinstance(unwrapped.value, str):
        flavour = classify_flavour(unwrapped.value)
        if flavour != expected_flavour:
            return
        lid = _ensure_literal(store, unwrapped.value, flavour)
        _emit_edge(store, src_id, lid, direction, sf.rel_path, site_line, site_col, role)
        return
    pid = _try_build_pathexpr(store, sf, scope, unwrapped)
    if pid is None:
        return
    node = store.get(pid)
    tail = node.attrs.get("tail_literal") if node is not None else None
    if tail is None:
        return
    flavour = classify_flavour(tail)
    if expected_flavour == "path" and not looks_path_like(tail):
        return
    if expected_flavour == "key" and flavour != "key":
        return
    # A PATHEXPR node is keyed by (file, line, col), not by text, so it
    # never collides across two different call sites — safe to stamp its
    # flavour from the FIRST (and, in practice, only) context it's used in.
    # Without this, analyze/literalaudit.py would have to guess a flavour
    # from node KIND alone and misclassify a KEY-LIKE f-string composition
    # (e.g. `f"_{name.upper()}"` used as a dict key) as PATH-LIKE.
    node.attrs.setdefault("flavour", expected_flavour)
    _emit_edge(store, src_id, pid, direction, sf.rel_path, site_line, site_col, role)


# -- role tables -------------------------------------------------------------

# method-on-receiver shapes: the literal/PATHEXPR sits in `call.func.value`
# (e.g. `(MEMORY_DIR / "notes.jsonl").write_text(...)`), keyed by attr name.
_RECEIVER_VERBS: dict[str, tuple[str, str]] = {
    "write_text": ("write_text", "produce"),
    "write_bytes": ("write_bytes", "produce"),
    "mkdir": ("mkdir", "produce"),
    "touch": ("touch", "produce"),
    "read_text": ("read_text", "consume"),
    "read_bytes": ("read_bytes", "consume"),
    "exists": ("exists", "consume"),
    "is_file": ("exists", "consume"),
    "is_dir": ("exists", "consume"),
    "unlink": ("unlink", "consume"),
}

# function-call-with-argument shapes: the literal/PATHEXPR sits in `args[0]`,
# keyed by the call's dotted name (or its final component as a fallback).
_ARG0_DOTTED_VERBS: dict[str, tuple[str, str]] = {
    "glob.glob": ("glob", "consume"),
    "os.path.exists": ("exists", "consume"),
    "os.path.isfile": ("exists", "consume"),
    "os.path.isdir": ("exists", "consume"),
    "os.remove": ("unlink", "consume"),
    "os.makedirs": ("mkdir", "produce"),
}

_STORE_CTOR_TAIL_RE = re.compile(r"Store$")


def _open_mode(call: ast.Call) -> str:
    if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant) and isinstance(call.args[1].value, str):
        return call.args[1].value
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return "r"


def _handle_call(store: GraphStore, sf: SourceFile, scope: ModuleScope, call: ast.Call, src_id: str) -> None:
    func = call.func
    dotted = _dotted_call_name(func) or ""
    tail = dotted.rsplit(".", 1)[-1] if dotted else None

    # -- receiver-based verbs: `<path-expr>.write_text(...)` etc.
    if isinstance(func, ast.Attribute) and func.attr in _RECEIVER_VERBS:
        role, direction = _RECEIVER_VERBS[func.attr]
        _handle_composed_or_literal(store, sf, scope, func.value, "path", role, direction,
                                     src_id, call.lineno, call.col_offset)
        return

    # -- open(path, mode): mode-dependent produce/consume, arg[0] is the path
    if tail == "open" and call.args:
        mode = _open_mode(call)
        if any(c in mode for c in "wax+"):
            role, direction = "open-w", "produce"
        else:
            role, direction = "open-r", "consume"
        _handle_composed_or_literal(store, sf, scope, call.args[0], "path", role, direction,
                                     src_id, call.lineno, call.col_offset)
        return

    # -- Store-ctor: `SomeStore(str(path_expr))` -> produce
    if isinstance(func, (ast.Name, ast.Attribute)) and dotted and _STORE_CTOR_TAIL_RE.search(dotted.rsplit(".", 1)[-1]):
        if call.args:
            _handle_composed_or_literal(store, sf, scope, call.args[0], "path", "store-ctor", "produce",
                                         src_id, call.lineno, call.col_offset)
        return

    # -- `.get("KEY")` — os.environ.get is a path-flavour-free key lookup;
    # any other receiver's `.get("KEY")` is treated as a dict-get lookup.
    if isinstance(func, ast.Attribute) and func.attr == "get" and call.args:
        base_dotted = _dotted_call_name(func.value) or ""
        if base_dotted.endswith("environ") or base_dotted.endswith("os.environ"):
            role = "environ-get"
        else:
            role = "dict-get"
        _handle_composed_or_literal(store, sf, scope, call.args[0], "key", role, "consume",
                                     src_id, call.lineno, call.col_offset)
        return

    # -- module-function arg0 verbs: glob.glob(...), os.path.exists(...), ...
    lookup_key = dotted if dotted in _ARG0_DOTTED_VERBS else (tail or "")
    if lookup_key in _ARG0_DOTTED_VERBS and call.args:
        role, direction = _ARG0_DOTTED_VERBS[lookup_key]
        _handle_composed_or_literal(store, sf, scope, call.args[0], "path", role, direction,
                                     src_id, call.lineno, call.col_offset)


def _handle_subscript(store: GraphStore, sf: SourceFile, scope: ModuleScope, node: ast.Subscript, src_id: str) -> None:
    sl = node.slice
    if not (isinstance(sl, ast.Constant) and isinstance(sl.value, str)):
        return
    if not looks_key_like(sl.value):
        return
    base_dotted = _dotted_call_name(node.value) or (node.value.id if isinstance(node.value, ast.Name) else "")
    is_environ = base_dotted.endswith("environ") or base_dotted.endswith("os.environ")
    if isinstance(node.ctx, ast.Load):
        role, direction = ("environ-get", "consume") if is_environ else ("dict-subscript", "consume")
    elif isinstance(node.ctx, ast.Store):
        role, direction = ("environ-set", "produce") if is_environ else ("dict-set", "produce")
    else:
        return
    lid = _ensure_literal(store, sl.value, "key")
    _emit_edge(store, src_id, lid, direction, sf.rel_path, node.lineno, node.col_offset, role)


def make_literals_visitor(store: GraphStore, sf: SourceFile, scope: ModuleScope):
    """Builds the walk_module(node, fi) callback without walking anything
    itself, so pipeline.py can fold it into the same shared whole-module
    traversal as calls.py/names.py instead of a fourth pass over the AST."""
    mod_id = module_node_id(sf.rel_path)

    def on_node(node: ast.AST, fi: Optional[FunctionInfo]) -> None:
        src_id = function_node_id(sf.rel_path, fi.qualname) if fi is not None else mod_id
        if isinstance(node, ast.Call):
            _handle_call(store, sf, scope, node, src_id)
        elif isinstance(node, ast.Subscript):
            _handle_subscript(store, sf, scope, node, src_id)

    return on_node


def extract_literals(store: GraphStore, sf: SourceFile, scope: ModuleScope) -> None:
    """Standalone entry point (unit tests, `cg` debug use) — pipeline.py
    folds make_literals_visitor into the shared whole-module walk instead."""
    if sf.tree is None:
        return
    walk_module(scope, make_literals_visitor(store, sf, scope))
