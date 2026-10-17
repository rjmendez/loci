"""CALLSITE nodes and the CALLS resolution ladder.

This slice implements rungs 1-3 of the design's seven-rung ladder (all
PROVEN-tier):

  1. callee is a plain Name bound in module scope (a def/class in this same
     module, a module-level import/alias reachable via one or more ALIASES
     hops, or a Python builtin).
  2. callee is an Attribute `m.f` where `m` is a module bound by a
     module-level import.
  3. callee is bound by a *function-local* import (either shape) — same
     resolution as 1/2 but scoped to the enclosing function and tagged
     scope=function-local.

Everything else (attribute calls on an unknown-type receiver, calls through
dict/getattr/subscript results, calls on a bare parameter, plain unresolved
names) is rung 7 territory for THIS slice: parked on the `?` UNRESOLVED sink
with a `reason`, never dropped. Rungs 4-6 (probable-tier: injected globals,
dict-dispatch fan-out, unique-method-name) are a later slice's job — see
extract/registry.py's INJECTS for the data those rungs will consume.

A CALLSITE node is created for every ast.Call in the corpus regardless of
whether it resolves, so `cg holes` can count what this tool does not (yet)
understand instead of that call vanishing silently.

stdlib only.
"""
from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
from typing import Optional

from ..ingest import SourceFile
from ..model import Confidence, Edge, GraphStore, Node
from ..resolve import ResolutionTable, ResolveResult
from ..scopes import FunctionInfo, ModuleScope
from .defs import function_node_id, module_node_id, name_node_id
from .walk import walk_module

BUILTIN_NAMES = frozenset(dir(builtins))
UNRESOLVED_ID = "?"


def callsite_node_id(rel_path: str, lineno: int, col: int, end_lineno: int, end_col: int) -> str:
    """Two distinct ast.Call nodes CAN share the same (lineno, col_offset) —
    a chained call `f(x).g()` has an outer Call node whose start position is
    identical to its own inner Call's start position. (lineno, col_offset)
    alone is therefore not a unique key; end position always differs
    between an outer and inner call sharing a start, and is an intrinsic
    property of the node (not traversal-order-dependent), so both
    extract/calls.py and extract/registry.py's REG-FN pass compute the same
    id for the same call independently, with no coordination needed. `cg`'s
    own output still displays just path:line:col — the extra precision is
    an internal disambiguator, not user-facing."""
    return f"call:{rel_path}:{lineno}:{col}:{end_lineno}:{end_col}"


def ensure_unresolved_sink(store: GraphStore) -> None:
    store.add_node(Node(id=UNRESOLVED_ID, kind="UNRESOLVED", attrs={}))


def external_node(store: GraphStore, dotted: str, distribution: str = "unknown") -> str:
    eid = f"ext:{dotted}"
    store.add_node(Node(id=eid, kind="EXTERNAL", attrs={"dotted": dotted, "distribution": distribution}))
    return eid


def build_top_level_index(store: GraphStore) -> dict[str, dict[str, str]]:
    """path -> {qualname: node_id} for every module-level (not nested, not
    method) FUNCTION and every CLASS. Built once per build and shared by
    calls.py/registry.py so a same-module or cross-module "does X define an
    ident called Y" check is an O(1) dict lookup instead of an O(defs in
    that module) re-scan per callsite — the difference between a build that
    stays under the 2s budget and one that does not once ~11,600 callsites
    are in play."""
    idx: dict[str, dict[str, str]] = {}
    for n in store.nodes_of_kind("FUNCTION"):
        if n.attrs.get("is_nested") or n.attrs.get("is_method") or n.path is None:
            continue
        idx.setdefault(n.path, {})[n.attrs["qualname"]] = n.id
    for n in store.nodes_of_kind("CLASS"):
        if n.path is None:
            continue
        idx.setdefault(n.path, {})[n.attrs["qualname"]] = n.id
    return idx


def init_method_id(store: GraphStore, class_id: str) -> Optional[str]:
    node = store.get(class_id)
    if node is None:
        return None
    for mid in node.attrs.get("method_ids", []):
        if mid.rsplit("::", 1)[-1].rsplit(".", 1)[-1] == "__init__":
            return mid
    return None


@dataclass
class _LocalImportBinding:
    kind: str            # "module" (`import X [as Y]`) | "from" (`from X import Y [as Z]`)
    res: ResolveResult
    display: str          # dotted module text as written
    imported: Optional[str]   # for kind="from": the original name imported
    line: int


def _build_local_import_index(scope: ModuleScope, table: ResolutionTable) -> dict[str, dict[str, _LocalImportBinding]]:
    idx: dict[str, dict[str, _LocalImportBinding]] = {}
    rel_path = scope.sf.rel_path
    for rec in scope.imports:
        if rec.scope != "function-local" or rec.enclosing_fn is None:
            continue
        bucket = idx.setdefault(rec.enclosing_fn, {})
        if not rec.is_from:
            for dotted, asname in rec.names:
                bound = asname or dotted.split(".")[0]
                res = table.resolve_dotted(dotted, rel_path)
                bucket[bound] = _LocalImportBinding("module", res, dotted, None, rec.lineno)
        else:
            if rec.level > 0:
                res = table.resolve_relative(rec.level, rec.module, rel_path)
                display = ("." * rec.level) + (rec.module or "")
            else:
                res = table.resolve_dotted(rec.module or "", rel_path)
                display = rec.module or ""
            for imported, asname in rec.names:
                if imported == "*":
                    continue
                bound = asname or imported
                bucket[bound] = _LocalImportBinding("from", res, display, imported, rec.lineno)
    return idx


def _distribution_for(res: ResolveResult) -> str:
    return {"stdlib": "stdlib", "third-party": "third-party"}.get(res.status, "unknown")


def _lookup_local_import(local_idx, qualname: Optional[str], ident: str):
    """A function-local `import X` binds a LOCAL name, and Python closures
    make that binding visible to every nested scope. So resolution has to
    walk OUTWARD through the `<parent>.<locals>.<child>` chain, not just
    check the immediately-enclosing function.

    Found by validation: `mcp/server.py::loci_health` does a function-local
    `import backends`, then calls it from inside a lambda --
    `lambda: backends.qdrant()[0]`. Keyed on the lambda's own qualname the
    binding is invisible, the call falls to the `?` sink, and
    `mcp/backends.py::qdrant` is reported dead despite a live call site.
    """
    if qualname is None:
        return None
    qual = qualname
    while True:
        binding = local_idx.get(qual, {}).get(ident)
        if binding is not None:
            return binding
        if ".<locals>." not in qual:
            return None
        qual = qual.rsplit(".<locals>.", 1)[0]


def _dotted_call_name(func: ast.AST) -> Optional[str]:
    parts: list[str] = []
    node = func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    else:
        return None
    return ".".join(reversed(parts))


def _classify_form(func: ast.expr, scope: ModuleScope, fi: Optional[FunctionInfo]) -> str:
    if isinstance(func, ast.Name):
        if fi is not None and func.id in scope.locals_of(fi).params:
            return "param-call"
        return "name"
    if isinstance(func, ast.Attribute):
        return "attribute"
    if isinstance(func, ast.Subscript):
        return "subscript-result"
    if isinstance(func, ast.Call):
        head = _dotted_call_name(func.func)
        if head == "getattr" or (head or "").endswith(".getattr"):
            return "getattr-result"
        if isinstance(func.func, ast.Attribute) and func.func.attr == "get":
            return "dict-get-result"
        return "call-result"
    return "other"


# -- resolution ---------------------------------------------------------


def _follow_alias_to_callable(store: GraphStore, name_id: str, depth: int = 0):
    if depth > 4:
        return None
    edges = store.out_edges(name_id, "ALIASES")
    if not edges:
        return None
    edge = edges[0]
    target = store.get(edge.dst)
    if target is None:
        return None
    if target.kind == "FUNCTION":
        return edge.dst, edge.confidence, {"rung": "module-alias"}
    if target.kind == "CLASS":
        init_id = init_method_id(store, edge.dst)
        if init_id is not None:
            return init_id, edge.confidence, {"rung": "module-alias-constructor"}
        return None
    if target.kind == "EXTERNAL":
        return edge.dst, edge.confidence, {"rung": "module-alias-external"}
    if target.kind == "NAME":
        return _follow_alias_to_callable(store, edge.dst, depth + 1)
    return None


def _resolve_name(store: GraphStore, sf: SourceFile, scope: ModuleScope, top_level_index, local_idx,
                   ident: str, fi: Optional[FunctionInfo]):
    rel = sf.rel_path

    # rung 1a': a nested function/lambda defined INSIDE the enclosing
    # function, called by its own bare name (shadows module scope, and is
    # excluded from top_level_index by design since it isn't a top-level
    # def) — e.g. a helper closure invoked directly by name.
    # Python resolves a bare name local -> ENCLOSING -> global, so this walks
    # outward through the `<locals>` chain: a nested def is visible to its
    # SIBLINGS, not just to itself. Found by validation:
    # mcp/graph/code_parse.py's `parse_source` defines `enclosing_def_node`
    # and `_in_method` side by side, and `_in_method` calls
    # `enclosing_def_node(node)` — checking only fi's own qualname missed it
    # and reported `enclosing_def_node` dead.
    if fi is not None:
        qual = fi.qualname
        while True:
            nested_qual = f"{qual}.<locals>.{ident}"
            nested_fi = scope.function_by_qualname.get(nested_qual)
            if nested_fi is not None:
                return function_node_id(rel, nested_qual), Confidence.PROVEN, {"rung": "nested-def-local"}
            if ".<locals>." not in qual:
                break
            qual = qual.rsplit(".<locals>.", 1)[0]

    # rung 1a: module-level def/class in THIS module
    target = top_level_index.get(rel, {}).get(ident)
    if target is not None:
        tnode = store.get(target)
        if tnode is not None and tnode.kind == "FUNCTION":
            return target, Confidence.PROVEN, {"rung": "name-def-local"}
        if tnode is not None and tnode.kind == "CLASS":
            init_id = init_method_id(store, target)
            if init_id is not None:
                return init_id, Confidence.PROVEN, {"rung": "name-constructor"}
            return UNRESOLVED_ID, Confidence.UNPROVEN, {"reason": "constructor-no-init"}

    # rung 1b: this NAME slot's own ALIASES edge (from-import or bare A=B),
    # chased through as many hops as it takes.
    resolved = _follow_alias_to_callable(store, name_node_id(rel, ident))
    if resolved is not None:
        return resolved

    # rung 3 (name form): bound by a function-local import in THIS function.
    if fi is not None:
        binding = _lookup_local_import(local_idx, fi.qualname, ident)
        if binding is not None and binding.kind == "from" and binding.imported:
            res = binding.res
            if res.status == "corpus" and res.target_path:
                found = top_level_index.get(res.target_path, {}).get(binding.imported)
                if found is not None:
                    fnode = store.get(found)
                    if fnode.kind == "FUNCTION":
                        return found, Confidence.PROVEN, {"rung": "name-local-import", "scope": "function-local",
                                                            "because": f"local import at {rel}:{binding.line}"}
                    if fnode.kind == "CLASS":
                        init_id = init_method_id(store, found)
                        if init_id is not None:
                            return init_id, Confidence.PROVEN, {"rung": "name-local-import-constructor",
                                                                 "scope": "function-local",
                                                                 "because": f"local import at {rel}:{binding.line}"}
            elif res.status in ("stdlib", "third-party"):
                dotted = f"{binding.display}.{binding.imported}" if binding.display else binding.imported
                eid = external_node(store, dotted, res.status)
                return eid, Confidence.PROVEN, {"rung": "name-local-import-external", "scope": "function-local",
                                                 "because": f"local import at {rel}:{binding.line}"}

    if ident in BUILTIN_NAMES:
        eid = external_node(store, f"builtins.{ident}", "stdlib")
        return eid, Confidence.PROVEN, {"rung": "builtin"}

    return UNRESOLVED_ID, Confidence.UNPROVEN, {"reason": "name-not-bound-in-module-scope"}


def _resolve_attr_via_module(store: GraphStore, top_level_index, target_module_path: str, attr: str):
    found = top_level_index.get(target_module_path, {}).get(attr)
    if found is None:
        return None
    fnode = store.get(found)
    if fnode.kind == "FUNCTION":
        return found, Confidence.PROVEN, {"rung": "module-attribute"}
    if fnode.kind == "CLASS":
        init_id = init_method_id(store, found)
        if init_id is not None:
            return init_id, Confidence.PROVEN, {"rung": "module-attribute-constructor"}
    return None


def _resolve_attribute(store: GraphStore, sf: SourceFile, scope: ModuleScope, top_level_index, local_idx,
                        func: ast.Attribute, fi: Optional[FunctionInfo], table: ResolutionTable):
    rel = sf.rel_path
    base = func.value
    attr = func.attr
    if not isinstance(base, ast.Name):
        return UNRESOLVED_ID, Confidence.UNPROVEN, {"reason": "attribute-unknown-receiver"}

    base_id = base.id

    if base_id == "self" and fi is not None and fi.is_method and fi.parent_class:
        cls_qual = fi.qualname.rsplit(".", 1)[0]
        method_id = function_node_id(rel, f"{cls_qual}.{attr}")
        if method_id in store:
            return method_id, Confidence.PROVEN, {"rung": "self-attribute"}
        return UNRESOLVED_ID, Confidence.UNPROVEN, {"reason": "self-attribute-not-found"}

    if base_id == "cls" and fi is not None and fi.is_method and fi.parent_class:
        cls_qual = fi.qualname.rsplit(".", 1)[0]
        method_id = function_node_id(rel, f"{cls_qual}.{attr}")
        if method_id in store:
            return method_id, Confidence.PROVEN, {"rung": "cls-attribute"}
        return UNRESOLVED_ID, Confidence.UNPROVEN, {"reason": "cls-attribute-not-found"}

    alias_edges = store.out_edges(name_node_id(rel, base_id), "ALIASES")
    if alias_edges:
        target = store.get(alias_edges[0].dst)
        if target is not None and target.kind == "MODULE":
            resolved = _resolve_attr_via_module(store, top_level_index, target.path, attr)
            if resolved is not None:
                return resolved
            return UNRESOLVED_ID, Confidence.UNPROVEN, {"reason": "attribute-not-found-in-module"}
        if target is not None and target.kind == "EXTERNAL":
            dotted = f"{target.attrs.get('dotted')}.{attr}"
            eid = external_node(store, dotted, target.attrs.get("distribution", "unknown"))
            return eid, Confidence.PROVEN, {"rung": "module-attribute-external"}

    if fi is not None:
        binding = _lookup_local_import(local_idx, fi.qualname, base_id)
        if binding is not None and binding.kind == "module":
            res = binding.res
            if res.status == "corpus" and res.target_path:
                resolved = _resolve_attr_via_module(store, top_level_index, res.target_path, attr)
                if resolved is not None:
                    target_id, conf, extra = resolved
                    extra = dict(extra)
                    extra["scope"] = "function-local"
                    extra["because"] = f"local import at {rel}:{binding.line}"
                    return target_id, conf, extra
            elif res.status in ("stdlib", "third-party"):
                dotted = f"{binding.display}.{attr}"
                eid = external_node(store, dotted, res.status)
                return eid, Confidence.PROVEN, {"rung": "module-attribute-external", "scope": "function-local",
                                                 "because": f"local import at {rel}:{binding.line}"}
        elif binding is not None and binding.kind == "from" and binding.imported:
            # `from pkg import submodule` (function-local), then
            # `submodule.fn()` — the imported name is itself a MODULE, not
            # a function/class in binding.res.target_path's own top level.
            # Mirrors extract/imports.py's _emit_alias_from_import submodule
            # fallback, but scoped to this function's own local import.
            submodule_res = table.resolve_dotted(f"{binding.display}.{binding.imported}", rel)
            if submodule_res.status == "corpus" and submodule_res.target_path:
                resolved = _resolve_attr_via_module(store, top_level_index, submodule_res.target_path, attr)
                if resolved is not None:
                    target_id, conf, extra = resolved
                    extra = dict(extra)
                    extra["scope"] = "function-local"
                    extra["because"] = f"local import at {rel}:{binding.line}"
                    return target_id, conf, extra

    return UNRESOLVED_ID, Confidence.UNPROVEN, {"reason": "attribute-unknown-receiver"}


_FORM_REASON = {
    "subscript-result": "computed-subscript-call",
    "getattr-result": "computed-getattr",
    "dict-get-result": "dict-with-computed-key",
    "call-result": "call-result-unresolved",
    "param-call": "param-call",
}


def _resolve_call(store, sf, scope, table, top_level_index, local_idx, node: ast.Call, fi, form: str):
    if form == "name":
        return _resolve_name(store, sf, scope, top_level_index, local_idx, node.func.id, fi)
    if form == "attribute":
        return _resolve_attribute(store, sf, scope, top_level_index, local_idx, node.func, fi, table)
    reason = _FORM_REASON.get(form, "unclassified")
    return UNRESOLVED_ID, Confidence.UNPROVEN, {"reason": reason}


def make_call_visitor(store: GraphStore, sf: SourceFile, scope: ModuleScope,
                       table: ResolutionTable, top_level_index: dict[str, dict[str, str]]):
    """Builds the walk_module(node, fi) callback without walking anything
    itself, so pipeline.py can run it in the SAME whole-module traversal as
    names.py's visitor — halving the generic-AST-walk overhead across the
    corpus's ~36k lines instead of paying it twice per file."""
    ensure_unresolved_sink(store)
    mod_id = module_node_id(sf.rel_path)
    local_idx = _build_local_import_index(scope, table)
    awaited_ids = {id(n.value) for n in ast.walk(sf.tree) if isinstance(n, ast.Await)} if sf.tree is not None else set()
    call_func_ids = {id(c.func) for c in ast.walk(sf.tree) if isinstance(c, ast.Call)} if sf.tree is not None else set()

    def on_node(node: ast.AST, fi: Optional[FunctionInfo]) -> None:
        if not isinstance(node, ast.Call):
            return
        form = _classify_form(node.func, scope, fi)
        callsite_id = callsite_node_id(sf.rel_path, node.lineno, node.col_offset,
                                        getattr(node, "end_lineno", node.lineno),
                                        getattr(node, "end_col_offset", node.col_offset))
        enclosing_id = function_node_id(sf.rel_path, fi.qualname) if fi is not None else mod_id
        receiver_src = None
        if isinstance(node.func, (ast.Attribute, ast.Subscript)):
            receiver_src = scope.source_of(node.func.value)
        attr_name = node.func.attr if isinstance(node.func, ast.Attribute) else None
        getattr_obj = getattr_attr_literal = None
        if form == "getattr-result" and isinstance(node.func, ast.Call) and len(node.func.args) >= 2:
            getattr_obj = scope.source_of(node.func.args[0])
            lit_arg = node.func.args[1]
            if isinstance(lit_arg, ast.Constant) and isinstance(lit_arg.value, str):
                getattr_attr_literal = lit_arg.value
        store.add_node(Node(
            id=callsite_id, kind="CALLSITE",
            attrs={
                "form": form,
                "callee": scope.source_of(node.func),
                "receiver": receiver_src,
                "attr": attr_name,
                "getattr_obj": getattr_obj,
                "getattr_attr_literal": getattr_attr_literal,
                "argc": len(node.args) + len(node.keywords),
                "enclosing_fn": enclosing_id,
                "awaited": id(node) in awaited_ids,
                "in_call_position": id(node.func) in call_func_ids,
            },
            path=sf.rel_path, line=node.lineno, col=node.col_offset,
        ))
        target_id, confidence, extra = _resolve_call(store, sf, scope, table, top_level_index, local_idx, node, fi, form)
        store.add_edge(Edge(src=callsite_id, dst=target_id, kind="CALLS", confidence=confidence, attrs=dict(extra)))

    return on_node


def extract_calls(store: GraphStore, sf: SourceFile, scope: ModuleScope,
                   table: ResolutionTable, top_level_index: dict[str, dict[str, str]]) -> None:
    """Standalone entry point (unit tests, `cg` debug use) — pipeline.py
    calls make_call_visitor directly instead, to share one walk_module pass
    with extract/names.py."""
    if sf.tree is None:
        return
    walk_module(scope, make_call_visitor(store, sf, scope, table, top_level_index))
