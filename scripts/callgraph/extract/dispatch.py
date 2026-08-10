"""The probable tier: CALLS resolution rungs 4-6, plus DISPATCHES.

This is a whole-corpus pass that runs AFTER extract/registry.py has built
REGISTRY/REGISTERS/INJECTS (registration roots and dependency injection are
exactly the facts these rungs need), and after extract/calls.py's rungs 1-3
have already parked every callsite it couldn't resolve on the `?` sink with
a reason. This pass only ever touches CALLS edges whose dst is still `?` —
a proven rung-1-3 resolution is never revisited or downgraded.

  rung 4  name-via-injected-global   a bare-name callsite whose ident is a
          module NAME slot with exactly one distinct INJECTS value (a
          function) resolves to that function, PROBABLE, because=the
          register()-call site that performed the injection. Two or more
          distinct injected values (re-entrant registration) leaves CALLS
          alone and fans out via DISPATCHES instead (INJECTS is already
          PROBABLE by construction; this rung does not upgrade it further).

  rung 5  dict-dispatch              a bare-name callsite whose ident is a
          local variable assigned earlier in the SAME function from
          `<dict>.get(<key>)` or `<dict>[<key>]`, where `<dict>` is a
          MAN-DICT REGISTRY in this module (the `_SKILL_MAP` shape). CALLS
          stays on `?` (no single target is knowable) with its reason
          upgraded to "dict-dispatch-fanout"; one DISPATCHES edge per
          registry member is added, each carrying the registry id, the
          member count as `fanout`, and the key expression's source text
          as `selector`.

  rung 6  unique-method-name         an attribute callsite whose receiver's
          type this tool cannot know (rung-1-3 left it at `?` with reason
          "attribute-unknown-receiver"). If exactly one method anywhere in
          the corpus shares this call's attribute name, resolve to it,
          PROBABLE, rung=unique-method-name, alternatives=1. Two or more
          candidates leaves CALLS at `?` with reason "ambiguous-method-name"
          and records `alternatives=N` for `cg holes`/`cg explain` — this
          tool does not guess among same-named methods.

  getattr-literal   `getattr(<module-ish>, "<literal>")(...)` — calls.py
          already captured the getattr call's own two arguments on the
          CALLSITE (getattr_obj / getattr_attr_literal) so this pass never
          re-parses source. If the object resolves to an ALIASES-bound
          in-corpus MODULE, resolve directly (PROVEN — a literal attribute
          name on a known module is exactly as certain as any other
          module-attribute rung); a non-literal attribute argument is left
          on `?` with reason "computed-getattr" (rung 1-3 already put it
          there; this pass only revisits it when the literal case applies).

stdlib only.
"""
from __future__ import annotations

import ast

from ..model import Confidence, Edge, GraphStore, Node
from ..scopes import ModuleScope
from .calls import _resolve_attr_via_module
from .defs import name_node_id
from .registry import registry_node_id

UNRESOLVED_ID = "?"


# -- rung 6 helper: corpus-wide method name index ---------------------------


def _build_method_index(store: GraphStore) -> dict[str, list[str]]:
    """simple method name -> every FUNCTION id anywhere in the corpus whose
    qualname's last dotted component is that name AND is_method is true.
    Built once per probable-tier pass; O(functions), not O(callsites)."""
    idx: dict[str, list[str]] = {}
    for n in store.nodes_of_kind("FUNCTION"):
        if not n.attrs.get("is_method"):
            continue
        simple = n.attrs.get("qualname", "").rsplit(".", 1)[-1]
        if not simple:
            continue
        idx.setdefault(simple, []).append(n.id)
    return idx


# -- rung 4: name-via-injected-global ----------------------------------------


def _try_injected_global(store: GraphStore, site: Node, edge: Edge) -> bool:
    ident = site.attrs.get("callee")
    mod_path = site.path
    if not ident or mod_path is None:
        return False
    nid = name_node_id(mod_path, ident)
    if nid not in store:
        return False
    injects = store.in_edges(nid, "INJECTS")
    values: dict[str, list[Edge]] = {}
    for inj in injects:
        if inj.attrs.get("value_kind") == "FUNCTION" and inj.attrs.get("value"):
            values.setdefault(inj.attrs["value"], []).append(inj)
    if not values:
        return False
    if len(values) == 1:
        (target_id, inj_list), = values.items()
        inj_site = store.get(inj_list[0].src)
        because_loc = f"{inj_site.path}:{inj_site.line}" if inj_site is not None else "unknown"
        store.retarget_edge(edge, target_id)
        edge.confidence = Confidence.PROBABLE
        edge.attrs = {"rung": "name-via-injected-global", "because": f"injected at {because_loc}"}
        return True
    # Re-entrant registration: two+ distinct functions have been injected
    # into the same global — CALLS cannot pick one, but every candidate is
    # still a real, nameable possibility, so DISPATCHES fans out to all of
    # them instead of silently guessing the first.
    edge.attrs = {"reason": "injected-global-multi-value", "alternatives": len(values)}
    for target_id in values:
        store.add_edge(Edge(src=site.id, dst=target_id, kind="DISPATCHES", confidence=Confidence.PROBABLE,
                             attrs={"registry": None, "fanout": len(values), "selector": ident, "via": "injected-global"}))
    return False


# -- rung 5: dict-dispatch ----------------------------------------------------


def _find_dict_dispatch_binding(fi_node: ast.AST, ident: str):
    """Scan `fi_node`'s own body (not descending into nested def/class/
    lambda scopes) for the LAST `ident = <dict>.get(<key>)` or
    `ident = <dict>[<key>]` assignment. Returns (dict_name, key_node, line)
    or None. A linear-walk, last-wins heuristic — matches the rest of this
    tool's "no CFG" philosophy."""
    found = None

    def walk(node: ast.AST) -> None:
        nonlocal found
        if node is not fi_node and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == ident):
            value = node.value
            dict_name = key_node = None
            if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "get" and isinstance(value.func.value, ast.Name)
                    and len(value.args) >= 1):
                dict_name, key_node = value.func.value.id, value.args[0]
            elif isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
                dict_name, key_node = value.value.id, value.slice
            if dict_name is not None:
                found = (dict_name, key_node, node.lineno)
        for child in ast.iter_child_nodes(node):
            walk(child)

    body = [fi_node.body] if isinstance(fi_node, ast.Lambda) else fi_node.body
    for stmt in body:
        walk(stmt)
    return found


def _fn_id_to_scope_lookup(fn_id: str) -> tuple[str, str]:
    rest = fn_id[len("fn:"):]
    mod_path, qualname = rest.split("::", 1)
    return mod_path, qualname


def _try_dict_dispatch(store: GraphStore, site: Node, edge: Edge, scopes: dict[str, ModuleScope]) -> bool:
    ident = site.attrs.get("callee")
    mod_path = site.path
    enclosing_fn = site.attrs.get("enclosing_fn")
    if not ident or mod_path is None or enclosing_fn is None or not enclosing_fn.startswith("fn:"):
        return False
    fn_mod, qualname = _fn_id_to_scope_lookup(enclosing_fn)
    scope = scopes.get(fn_mod)
    if scope is None:
        return False
    fi = scope.function_by_qualname.get(qualname)
    if fi is None:
        return False
    binding = _find_dict_dispatch_binding(fi.node, ident)
    if binding is None:
        return False
    dict_name, key_node, _assign_line = binding
    registry_id = registry_node_id(mod_path, dict_name)
    reg_node = store.get(registry_id)
    if reg_node is None or reg_node.attrs.get("mechanism") != "manifest-dict":
        return False
    members = store.out_edges(registry_id, "REGISTERS")
    if not members:
        return False
    selector = scope.source_of(key_node) if key_node is not None else dict_name
    edge.attrs = {"reason": "dict-dispatch-fanout", "registry": registry_id, "fanout": len(members), "selector": selector}
    for m in members:
        store.add_edge(Edge(src=site.id, dst=m.dst, kind="DISPATCHES", confidence=Confidence.PROBABLE,
                             attrs={"registry": registry_id, "fanout": len(members), "selector": selector, "via": "dict-dispatch"}))
    return True


# -- rung 6: unique-method-name ----------------------------------------------


def _try_unique_method_name(store: GraphStore, site: Node, edge: Edge, method_index: dict[str, list[str]]) -> bool:
    if edge.attrs.get("reason") != "attribute-unknown-receiver":
        return False
    attr = site.attrs.get("attr")
    if not attr:
        return False
    candidates = method_index.get(attr, [])
    if len(candidates) == 1:
        store.retarget_edge(edge, candidates[0])
        edge.confidence = Confidence.PROBABLE
        edge.attrs = {"rung": "unique-method-name", "alternatives": 1}
        return True
    if len(candidates) > 1:
        edge.attrs = {"reason": "ambiguous-method-name", "alternatives": len(candidates)}
    return False


# -- getattr-literal ----------------------------------------------------------


def _try_getattr_literal(store: GraphStore, site: Node, edge: Edge, top_level_index) -> bool:
    obj_src = site.attrs.get("getattr_obj")
    attr_lit = site.attrs.get("getattr_attr_literal")
    mod_path = site.path
    if not obj_src or attr_lit is None or mod_path is None:
        return False
    if not obj_src.isidentifier():
        return False  # only a bare Name receiver is modelled; anything else stays a hole
    alias_edges = store.out_edges(name_node_id(mod_path, obj_src), "ALIASES")
    if not alias_edges:
        return False
    target = store.get(alias_edges[0].dst)
    if target is None or target.kind != "MODULE" or target.path is None:
        return False
    resolved = _resolve_attr_via_module(store, top_level_index, target.path, attr_lit)
    if resolved is None:
        return False
    target_id, _conf, _extra = resolved
    store.retarget_edge(edge, target_id)
    edge.confidence = Confidence.PROVEN
    edge.attrs = {"rung": "getattr-literal", "attr": attr_lit}
    return True


# -- entry point --------------------------------------------------------------


def extract_probable_calls(store: GraphStore, scopes: dict[str, ModuleScope], top_level_index) -> None:
    """Call once per build, after extract_registry_module + extract_injections
    + extract_external_roots have all run for the whole corpus."""
    method_index = _build_method_index(store)
    unresolved = [e for e in store.in_edges(UNRESOLVED_ID, "CALLS")]
    for edge in unresolved:
        site = store.get(edge.src)
        if site is None or site.kind != "CALLSITE":
            continue
        form = site.attrs.get("form")
        if form == "name":
            if _try_injected_global(store, site, edge):
                continue
            if edge.dst == UNRESOLVED_ID:
                _try_dict_dispatch(store, site, edge, scopes)
        elif form == "attribute":
            _try_unique_method_name(store, site, edge, method_index)
        elif form == "getattr-result":
            _try_getattr_literal(store, site, edge, top_level_index)
