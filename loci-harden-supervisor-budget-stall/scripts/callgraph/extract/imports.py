"""IMPORTS (module-level vs function-local) and ALIASES/re-exports.

Records `importlib.import_module(<literal>)` as a proven import edge;
`import_module(<variable>)` is deliberately left alone here — an
unresolvable dynamic import is extract/calls.py's `?` sink territory (a
later build step), not this module's.

stdlib only.
"""
from __future__ import annotations

import ast
from typing import Optional

from ..ingest import SourceFile
from ..model import Confidence, Edge, GraphStore, Node
from ..resolve import ResolutionTable, ResolveResult
from ..scopes import ModuleScope
from .defs import class_node_id, function_node_id, module_node_id, name_node_id


def external_node_id(dotted: str) -> str:
    return f"ext:{dotted}"


def _distribution_for(res: ResolveResult) -> str:
    return {"stdlib": "stdlib", "third-party": "third-party"}.get(res.status, "unknown")


def _ensure_external(store: GraphStore, dotted: str, distribution: str) -> str:
    eid = external_node_id(dotted)
    store.add_node(Node(id=eid, kind="EXTERNAL", attrs={"dotted": dotted, "distribution": distribution}))
    return eid


def _resolved_via_confidence(res: ResolveResult) -> Confidence:
    if res.status in ("corpus", "stdlib", "third-party"):
        return Confidence.PROVEN
    return Confidence.UNPROVEN


def _emit_imports_edge(store: GraphStore, mod_id: str, display_module: str, res: ResolveResult,
                        scope_kind: str, enclosing_fn: Optional[str], bound_names: list[str], line: int) -> None:
    if res.status == "corpus" and res.target_path is not None:
        dst = module_node_id(res.target_path)
    else:
        dst = _ensure_external(store, display_module or "?", _distribution_for(res))
    attrs = {
        "scope": scope_kind, "enclosing_fn": enclosing_fn, "bound_names": bound_names,
        "resolved_via": res.resolved_via, "module": display_module, "line": line,
    }
    if res.ambiguous:
        attrs["ambiguous_candidates"] = res.candidates
    if res.cross_file_root:
        attrs["cross_file_root"] = True
    store.add_edge(Edge(src=mod_id, dst=dst, kind="IMPORTS",
                         confidence=_resolved_via_confidence(res), attrs=attrs))


def _lookup_in_module(all_scopes: dict[str, ModuleScope], target_path: str, ident: str) -> Optional[str]:
    """A FUNCTION/CLASS/NAME node id for `ident` defined at TOP LEVEL of
    the module at target_path, or None."""
    target_scope = all_scopes.get(target_path)
    if target_scope is None:
        return None
    for fi in target_scope.functions:
        if fi.qualname == ident and not fi.is_nested and not fi.is_method:
            return function_node_id(target_path, ident)
    for ci in target_scope.classes:
        if ci.qualname == ident:
            return class_node_id(target_path, ident)
    if ident in target_scope.names:
        return name_node_id(target_path, ident)
    return None


def extract_imports(store: GraphStore, sf: SourceFile, scope: ModuleScope,
                     table: ResolutionTable, all_scopes: dict[str, ModuleScope]) -> None:
    if sf.tree is None:
        return
    mod_id = module_node_id(sf.rel_path)

    for rec in scope.imports:
        if not rec.is_from:
            for dotted, asname in rec.names:
                res = table.resolve_dotted(dotted, sf.rel_path)
                bound = asname or dotted.split(".")[0]
                _emit_imports_edge(store, mod_id, dotted, res, rec.scope, rec.enclosing_fn, [bound], rec.lineno)
                if rec.scope == "module-level":
                    _emit_alias_to_module(store, sf, bound, res, rec.lineno)
        else:
            if rec.level > 0:
                res = table.resolve_relative(rec.level, rec.module, sf.rel_path)
                display_module = ("." * rec.level) + (rec.module or "")
            else:
                res = table.resolve_dotted(rec.module or "", sf.rel_path)
                display_module = rec.module or ""
            bound_all = [asname or imported for imported, asname in rec.names if imported != "*"]
            _emit_imports_edge(store, mod_id, display_module, res, rec.scope, rec.enclosing_fn, bound_all, rec.lineno)
            if rec.scope == "module-level":
                for imported, asname in rec.names:
                    if imported == "*":
                        continue  # star-import: no per-name alias (a hole, surfaced by later `cg holes`)
                    bound = asname or imported
                    _emit_alias_from_import(store, sf, bound, res, display_module, imported, rec.lineno, all_scopes, table)

    _emit_importlib_literal_imports(store, sf, mod_id, table)
    _emit_module_level_name_aliases(store, sf, scope, all_scopes)


def _emit_alias_to_module(store: GraphStore, sf: SourceFile, bound: str, res: ResolveResult, line: int) -> None:
    """`import X [as Y]` at module level: NAME Y aliases the MODULE object X."""
    nid = name_node_id(sf.rel_path, bound)
    if res.status == "corpus" and res.target_path is not None:
        dst = module_node_id(res.target_path)
        conf = Confidence.PROVEN
    else:
        dst = _ensure_external(store, bound, _distribution_for(res))
        conf = Confidence.PROVEN if res.status in ("stdlib", "third-party") else Confidence.UNPROVEN
    store.add_edge(Edge(src=nid, dst=dst, kind="ALIASES", confidence=conf, attrs={"form": "import", "line": line}))


def _emit_alias_from_import(store: GraphStore, sf: SourceFile, bound: str, res: ResolveResult,
                             display_module: str, imported: str, line: int,
                             all_scopes: dict[str, ModuleScope], table: ResolutionTable) -> None:
    """`from X import Y [as Z]` at module level: NAME Z aliases whatever Y
    is inside module X — a function/class/name if X is a corpus module and
    defines it, a submodule if Y is itself a module, else an external
    sink."""
    nid = name_node_id(sf.rel_path, bound)
    if res.status == "corpus" and res.target_path is not None:
        target_id = _lookup_in_module(all_scopes, res.target_path, imported)
        if target_id is not None:
            store.add_edge(Edge(src=nid, dst=target_id, kind="ALIASES", confidence=Confidence.PROVEN,
                                 attrs={"form": "from-import", "line": line}))
            return
        # `from . import Y` / `from .pkg import Y` where Y is a SIBLING
        # MODULE FILE. The dotted-name fallback below cannot see it: for a
        # relative import display_module is just "." (or ".."), so it probes
        # "..Y" and "Y" and never the package-qualified name. Resolve it
        # against the filesystem layout instead, which is what Python does.
        #
        # Found by validation: `mcp/graph/analytics.py` does
        # `from . import queries as Q` and then `Q.finding_symbols(...)`.
        # Without this, `Q` aliased an external sink instead of the MODULE,
        # so every `Q.*` call fell to the `?` sink and four live
        # mcp/graph/queries.py functions were reported dead.
        if res.target_path.endswith("/__init__.py"):
            pkg_dir = res.target_path[: -len("/__init__.py")]
            sibling = f"{pkg_dir}/{imported}.py"
            if sibling in all_scopes:
                store.add_edge(Edge(src=nid, dst=module_node_id(sibling), kind="ALIASES",
                                     confidence=Confidence.PROVEN,
                                     attrs={"form": "from-import-submodule", "line": line}))
                return

        # Not a name defined in X's top level — maybe X.Y is itself a
        # submodule (`from mcp import graph_tools` where graph_tools.py is
        # a sibling file reachable as "mcp.graph_tools" or "graph_tools").
        for cand in (f"{display_module}.{imported}", imported):
            matches = table.by_dotted.get(cand) or []
            if matches:
                mod_path = matches[0][0]
                store.add_edge(Edge(src=nid, dst=module_node_id(mod_path), kind="ALIASES",
                                     confidence=Confidence.PROVEN, attrs={"form": "from-import-submodule", "line": line}))
                return
        # Genuinely couldn't find it inside a resolved corpus module (e.g.
        # re-exported through `__init__.py` `from .x import *`, or a
        # dynamically-created attribute) — fall back to an external sink
        # named after the qualified import so it is still queryable.
        dst = _ensure_external(store, f"{display_module}.{imported}", "unknown")
        store.add_edge(Edge(src=nid, dst=dst, kind="ALIASES", confidence=Confidence.UNPROVEN,
                             attrs={"form": "from-import-unresolved-in-module", "line": line}))
        return
    dst = _ensure_external(store, f"{display_module}.{imported}" if display_module else imported,
                            _distribution_for(res))
    conf = Confidence.PROVEN if res.status in ("stdlib", "third-party") else Confidence.UNPROVEN
    store.add_edge(Edge(src=nid, dst=dst, kind="ALIASES", confidence=conf,
                         attrs={"form": "from-import", "line": line}))


def _emit_module_level_name_aliases(store: GraphStore, sf: SourceFile, scope: ModuleScope,
                                     all_scopes: dict[str, ModuleScope]) -> None:
    """Bare `A = B` at module level, where B is itself a name/function/class
    already bound in this same module."""
    for ma in scope.module_assigns:
        if ma.value_kind != "name" or ma.value_name is None:
            continue
        if ma.value_name == ma.target:
            continue
        target_id = _lookup_in_module(all_scopes, sf.rel_path, ma.value_name)
        if target_id is None:
            continue
        src_id = name_node_id(sf.rel_path, ma.target)
        store.add_edge(Edge(src=src_id, dst=target_id, kind="ALIASES", confidence=Confidence.PROVEN,
                             attrs={"form": "assign", "line": ma.lineno}))


def _emit_importlib_literal_imports(store: GraphStore, sf: SourceFile, mod_id: str, table: ResolutionTable) -> None:
    """`importlib.import_module("literal")` anywhere in the module, at any
    nesting depth, is treated as a proven import edge (the argument is a
    literal, so this is not a guess)."""
    if sf.tree is None:
        return
    for node in ast.walk(sf.tree):
        if not isinstance(node, ast.Call):
            continue
        fname = _dotted_call_name(node.func)
        if fname not in ("importlib.import_module", "import_module"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
            continue
        dotted = node.args[0].value
        res = table.resolve_dotted(dotted, sf.rel_path)
        enclosing_fn = _enclosing_function_qualname(sf.tree, node)
        _emit_imports_edge(store, mod_id, dotted, res,
                            "function-local" if enclosing_fn else "module-level",
                            enclosing_fn, [], node.lineno)


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


def _enclosing_function_qualname(tree: ast.Module, target: ast.AST) -> Optional[str]:
    """Best-effort: find the nearest enclosing FunctionDef/AsyncFunctionDef
    of `target` by line range containment. Good enough for attributing an
    importlib call to a function without a second full scope pass."""
    best: Optional[tuple[int, str]] = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start, end = node.lineno, getattr(node, "end_lineno", node.lineno)
            if start <= target.lineno <= end:
                width = end - start
                if best is None or width < best[0]:
                    best = (width, node.name)
    return best[1] if best else None
