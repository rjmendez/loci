"""Registration roots: REGISTRY / REGISTERS / ENTERS / ENTRYPOINT / INJECTS.

Five rules, matching the design's naming exactly:

  DEC       a decorator defs.py already classified "registering" (rules.toml)
            -> one REGISTRY per (module, decorator dotted head); the
            function's DECORATED_BY edge is RETARGETED from the placeholder
            EXTERNAL sink defs.py created onto this REGISTRY node.
  MAN-LOOP  `for fn in (a, b, c): <registering-call>(fn)` — a manifest tuple
            whose loop body registers each element (e.g. `mcp.tool()(fn)`);
            distinguished from an unrelated `for fn in (...): fn.method()`
            utility loop by requiring the body call itself be decorator-
            classified "registering".
  MAN-DICT  a module-level `NAME = {"key": bare_name, ...}` (Assign or
            AnnAssign) whose values are ALL bare Names resolving to
            in-corpus functions — the `_SKILL_MAP` shape. A dict whose
            values are literals (config, not dispatch) never matches.
  ROOT-CLI  `if __name__ == "__main__":` — enters whatever bare-name call
            appears first inside the guard, if any.
  ROOT-EXT  rules.toml's `[[roots]]` table — hand-maintained roots living
            outside the corpus (systemd/cron/shell), trust=declared-in-
            roots.toml, reusing ROOT-CLI's own guard-scan to find the
            entered function.

REG-FN (register()'s parameter/deps-dict -> `global` binding, emitting
INJECTS) is a whole-corpus pass (extract_injections) because it has to
correlate a function's OWN definition (which globals does it inject into)
with every CALL SITE of that function anywhere in the corpus.

stdlib only.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional

from ..ingest import SourceFile
from ..model import Confidence, Edge, GraphStore, Node
from ..rules import Rules, load_rules
from ..scopes import FunctionInfo, ModuleScope
from .calls import _follow_alias_to_callable, callsite_node_id
from .defs import function_node_id, module_node_id, name_node_id
from .walk import walk_module


def registry_node_id(mod_path: str, label: str) -> str:
    return f"reg:{mod_path}::{label}"


def _ensure_registry(store: GraphStore, mod_path: str, label: str, mechanism: str, line: Optional[int]) -> str:
    rid = registry_node_id(mod_path, label)
    node = store.get(rid)
    if node is None:
        store.add_node(Node(id=rid, kind="REGISTRY", attrs={
            "mechanism": mechanism, "keys": [], "member_count": 0, "declaration_line": line,
        }, path=mod_path, line=line))
    return rid


_ENTRY_KIND_BY_RULE = {
    "DEC-tool": "mcp-tool", "DEC-route": "fastapi", "DEC-mcp-route": "mcp-route",
    "MAN-LOOP": "mcp-tool", "MAN-DICT": "skill",
}


def _add_entrypoint(store: GraphStore, key: str, fn_id: str, rule: str, evidence: str, trust: str = "declared-in-source") -> None:
    kind = _ENTRY_KIND_BY_RULE.get(rule, "mcp-tool")
    entry_id = f"entry:{kind}:{key}"
    node = store.get(entry_id)
    if node is None:
        store.add_node(Node(id=entry_id, kind="ENTRYPOINT", attrs={
            "kind": kind, "key": key, "evidence": evidence, "trust": trust,
        }))
    store.add_edge(Edge(src=entry_id, dst=fn_id, kind="ENTERS", confidence=Confidence.PROVEN,
                         attrs={"trust": trust, "evidence": evidence}))


def _register_member(store: GraphStore, registry_id: str, fn_id: str, key: str, line: int, rule: str, evidence: str) -> None:
    reg_node = store.get(registry_id)
    if reg_node is not None:
        keys = reg_node.attrs.setdefault("keys", [])
        if key not in keys:
            keys.append(key)
        reg_node.attrs["member_count"] = reg_node.attrs.get("member_count", 0) + 1
        if reg_node.attrs.get("declaration_line") is None:
            reg_node.attrs["declaration_line"] = line
    store.add_edge(Edge(src=registry_id, dst=fn_id, kind="REGISTERS", confidence=Confidence.PROVEN,
                         attrs={"key": key, "rule": rule, "line": line}))
    _add_entrypoint(store, key, fn_id, rule, evidence)


# -- DEC ---------------------------------------------------------------


def _parse_decorator_call(raw: str) -> Optional[ast.Call]:
    try:
        expr = ast.parse(raw, mode="eval").body
    except SyntaxError:
        return None
    return expr if isinstance(expr, ast.Call) else None


def _first_string_arg(call: Optional[ast.Call]) -> Optional[str]:
    if call is None or not call.args:
        return None
    a0 = call.args[0]
    return a0.value if isinstance(a0, ast.Constant) and isinstance(a0.value, str) else None


def _methods_kwarg(call: Optional[ast.Call]) -> Optional[str]:
    if call is None:
        return None
    for kw in call.keywords:
        if kw.arg == "methods" and isinstance(kw.value, (ast.List, ast.Tuple)):
            names = [e.value for e in kw.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if names:
                return "/".join(names)
    return None


def _dec_key_and_rule(fi: FunctionInfo, dec_head: str, raw: str) -> tuple[str, str]:
    last = dec_head.rsplit(".", 1)[-1]
    call = _parse_decorator_call(raw)
    if last in ("get", "post", "put", "delete", "patch"):
        path = _first_string_arg(call) or "?"
        return f"{last.upper()} {path}", "DEC-route"
    if last == "route":
        path = _first_string_arg(call) or "?"
        methods = _methods_kwarg(call) or "ROUTE"
        return f"{methods} {path}", "DEC-route"
    if last == "custom_route":
        path = _first_string_arg(call) or "?"
        methods = _methods_kwarg(call) or "ROUTE"
        return f"{methods} {path}", "DEC-mcp-route"
    return fi.qualname, "DEC-tool"


def _extract_dec(store: GraphStore, sf: SourceFile, scope: ModuleScope) -> None:
    mod_path = sf.rel_path
    for fi in scope.functions:
        if fi.is_lambda or fi.is_nested:
            continue
        fid = function_node_id(mod_path, fi.qualname)
        for edge in list(store.out_edges(fid, "DECORATED_BY")):
            if edge.attrs.get("classification") != "registering":
                continue
            ext_node = store.get(edge.dst)
            dec_head = ext_node.attrs.get("dotted") if ext_node is not None else edge.attrs.get("raw", "").split("(")[0]
            key, rule = _dec_key_and_rule(fi, dec_head, edge.attrs.get("raw", ""))
            line = edge.attrs.get("line") or fi.lineno
            registry_id = _ensure_registry(store, mod_path, dec_head, "decorator", line)
            store.retarget_edge(edge, registry_id)
            _register_member(store, registry_id, fid, key, line, rule, f"{mod_path}:{line}")


# -- MAN-LOOP ------------------------------------------------------------


def _dotted_head(target: ast.expr) -> Optional[str]:
    parts: list[str] = []
    cur = target
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _loop_body_registers_var(body: list[ast.stmt], varname: str, rules: Rules) -> bool:
    for stmt in body:
        for node in ast.walk(stmt):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Call):
                continue
            if len(node.args) != 1 or not isinstance(node.args[0], ast.Name) or node.args[0].id != varname:
                continue
            head = _dotted_head(node.func.func)
            if head is not None and rules.classify_decorator(head) == "registering":
                return True
    return False


def _extract_manloop(store: GraphStore, sf: SourceFile, scope: ModuleScope, top_level_index) -> None:
    if sf.tree is None:
        return
    rules = load_rules()
    mod_path = sf.rel_path
    for fi in scope.functions:
        if fi.is_lambda:
            continue
        for node in ast.walk(fi.node):
            if not isinstance(node, ast.For) or not isinstance(node.target, ast.Name):
                continue
            if not isinstance(node.iter, (ast.Tuple, ast.List)):
                continue
            elts = node.iter.elts
            if not elts or not all(isinstance(e, ast.Name) for e in elts):
                continue
            if not _loop_body_registers_var(node.body, node.target.id, rules):
                continue
            registry_id = _ensure_registry(store, mod_path, fi.qualname, "manifest-tuple", node.lineno)
            enclosing_id = function_node_id(mod_path, fi.qualname)
            for e in elts:
                target_id = top_level_index.get(mod_path, {}).get(e.id)
                dst = target_id if target_id is not None else f"ext:{mod_path}::{e.id}"
                store.add_edge(Edge(src=enclosing_id, dst=dst, kind="REFERENCES", confidence=Confidence.PROVEN,
                                     attrs={"form": "manifest-tuple-element", "line": e.lineno}))
                if target_id is not None and store.get(target_id) is not None and store.get(target_id).kind == "FUNCTION":
                    _register_member(store, registry_id, target_id, e.id, e.lineno, "MAN-LOOP", f"{mod_path}:{e.lineno}")


# -- MAN-DICT --------------------------------------------------------------


def _extract_mandict(store: GraphStore, sf: SourceFile, top_level_index) -> None:
    if sf.tree is None:
        return
    mod_path = sf.rel_path
    for stmt in sf.tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target, value = stmt.targets[0], stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            target, value = stmt.target, stmt.value
        else:
            continue
        if not isinstance(target, ast.Name) or not isinstance(value, ast.Dict):
            continue
        if not value.keys or any(k is None for k in value.keys):
            continue
        if not all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in value.keys):
            continue
        if not all(isinstance(v, ast.Name) for v in value.values):
            continue
        resolved = []
        ok = True
        for k, v in zip(value.keys, value.values):
            fid = top_level_index.get(mod_path, {}).get(v.id)
            if fid is None or store.get(fid) is None or store.get(fid).kind != "FUNCTION":
                ok = False
                break
            resolved.append((k.value, fid, v.lineno))
        if not ok or not resolved:
            continue
        registry_id = _ensure_registry(store, mod_path, target.id, "manifest-dict", stmt.lineno)
        mod_id = module_node_id(mod_path)
        for key, fid, line in resolved:
            store.add_edge(Edge(src=mod_id, dst=fid, kind="REFERENCES", confidence=Confidence.PROVEN,
                                 attrs={"form": "manifest-dict-value", "line": line}))
            _register_member(store, registry_id, fid, key, line, "MAN-DICT", f"{mod_path}:{line}")


# -- ROOT-CLI / ROOT-EXT ------------------------------------------------


def _find_main_guard(tree: ast.Module) -> Optional[ast.If]:
    for stmt in tree.body:
        if isinstance(stmt, ast.If):
            test = stmt.test
            if isinstance(test, ast.Compare) and len(test.comparators) == 1:
                left, right = test.left, test.comparators[0]
                names = {n for n in (left, right) if isinstance(n, ast.Name)}
                lits = {n.value for n in (left, right) if isinstance(n, ast.Constant)}
                if any(n.id == "__name__" for n in names) and "__main__" in lits:
                    return stmt
    return None


def _main_guard_entry_function(sf: SourceFile, top_level_index) -> tuple[Optional[str], Optional[int]]:
    if sf.tree is None:
        return None, None
    guard = _find_main_guard(sf.tree)
    if guard is None:
        return None, None
    for stmt in guard.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                target_id = top_level_index.get(sf.rel_path, {}).get(node.func.id)
                if target_id is not None and store_kind_is_function(target_id, top_level_index):
                    return target_id, node.lineno
    return None, guard.lineno


def store_kind_is_function(node_id: str, top_level_index) -> bool:
    # top_level_index only ever holds FUNCTION/CLASS ids; a CLASS id here
    # would mean `if __name__ == "__main__": SomeClass()` — not a function
    # entry, deliberately not modelled (rare, and not worth guessing an
    # __init__ target for what's usually direct script driver code).
    return not node_id.startswith("cls:")


def _extract_rootcli(store: GraphStore, sf: SourceFile, top_level_index) -> None:
    if sf.tree is None or _find_main_guard(sf.tree) is None:
        return
    mod_path = sf.rel_path
    guard = _find_main_guard(sf.tree)
    entry_fn_id, line = _main_guard_entry_function(sf, top_level_index)
    entry_id = f"entry:cli:{mod_path}"
    evidence = f"{mod_path}:{guard.lineno}"
    store.add_node(Node(id=entry_id, kind="ENTRYPOINT", attrs={
        "kind": "cli", "key": mod_path, "evidence": evidence, "trust": "declared-in-source",
    }))
    if entry_fn_id is not None:
        store.add_edge(Edge(src=entry_id, dst=entry_fn_id, kind="ENTERS", confidence=Confidence.PROVEN,
                             attrs={"trust": "declared-in-source", "line": line}))


def extract_registry_module(store: GraphStore, sf: SourceFile, scope: ModuleScope, top_level_index) -> None:
    """The four per-file rules (DEC, MAN-LOOP, MAN-DICT, ROOT-CLI). Call once
    per source file, after defs.py/imports.py have both run for every file
    in the corpus (top_level_index needs every module's FUNCTION/CLASS
    nodes already created)."""
    _extract_dec(store, sf, scope)
    _extract_manloop(store, sf, scope, top_level_index)
    _extract_mandict(store, sf, top_level_index)
    _extract_rootcli(store, sf, top_level_index)


def extract_external_roots(store: GraphStore, sources: list[SourceFile], top_level_index) -> None:
    """ROOT-EXT: rules.toml's `[[roots]]` table. Call once for the whole
    corpus, after extract_registry_module has run for every file (so the
    ROOT-CLI ENTERS edge it may reuse already exists)."""
    rules = load_rules()
    by_path = {sf.rel_path: sf for sf in sources}
    for root in rules.roots:
        mod_path = root.get("module")
        evidence = root.get("evidence", mod_path or "roots.toml")
        sf = by_path.get(mod_path) if mod_path else None
        entry_id = f"entry:external:{evidence}"
        store.add_node(Node(id=entry_id, kind="ENTRYPOINT", attrs={
            "kind": "external", "key": mod_path, "evidence": evidence, "trust": "declared-in-roots.toml",
        }))
        if sf is None:
            continue
        entry_fn_id, line = _main_guard_entry_function(sf, top_level_index)
        if entry_fn_id is not None:
            store.add_edge(Edge(src=entry_id, dst=entry_fn_id, kind="ENTERS", confidence=Confidence.PROVEN,
                                 attrs={"trust": "declared-in-roots.toml", "line": line}))


# -- REG-FN / INJECTS ----------------------------------------------------


@dataclass
class InjectorSpec:
    fn_id: str
    owning_module: str
    direct: dict[str, str] = field(default_factory=dict)          # param_name -> global ident
    dict_keys: dict[tuple[str, str], str] = field(default_factory=dict)  # (dict_param, key) -> global ident


def _build_injector_specs(scopes: dict[str, ModuleScope], rules: Rules) -> dict[str, InjectorSpec]:
    specs: dict[str, InjectorSpec] = {}
    for mod_path, scope in scopes.items():
        for fi in scope.functions:
            if fi.is_nested or fi.is_method or fi.is_lambda:
                continue
            simple_name = fi.qualname.rsplit(".", 1)[-1]
            if not rules.is_register_fn_name(simple_name):
                continue
            global_names: set[str] = set()
            for node in ast.walk(fi.node):
                if isinstance(node, ast.Global):
                    global_names.update(node.names)
            if not global_names:
                continue
            param_names = {p.name for p in fi.params}
            direct: dict[str, str] = {}
            dict_keys: dict[tuple[str, str], str] = {}
            for node in ast.walk(fi.node):
                if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                    continue
                tgt = node.targets[0]
                if not isinstance(tgt, ast.Name) or tgt.id not in global_names:
                    continue
                val = node.value
                if isinstance(val, ast.Name) and val.id in param_names:
                    direct[val.id] = tgt.id
                elif (isinstance(val, ast.Subscript) and isinstance(val.value, ast.Name)
                      and val.value.id in param_names):
                    key_node = val.slice
                    if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                        dict_keys[(val.value.id, key_node.value)] = tgt.id
            if direct or dict_keys:
                fid = function_node_id(mod_path, fi.qualname)
                specs[fid] = InjectorSpec(fid, mod_path, direct, dict_keys)
    return specs


def _resolve_register_call_target(store: GraphStore, sf: SourceFile, top_level_index, node: ast.Call) -> Optional[str]:
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        alias_edges = store.out_edges(name_node_id(sf.rel_path, func.value.id), "ALIASES")
        if alias_edges:
            target = store.get(alias_edges[0].dst)
            if target is not None and target.kind == "MODULE" and target.path is not None:
                return top_level_index.get(target.path, {}).get(func.attr)
        return None
    if isinstance(func, ast.Name):
        return top_level_index.get(sf.rel_path, {}).get(func.id)
    return None


def _resolve_injected_value(store: GraphStore, sf: SourceFile, scope: ModuleScope, top_level_index, expr: ast.expr):
    if isinstance(expr, ast.Lambda):
        qual = scope.qualname_for_node(expr)
        if qual:
            fid = function_node_id(sf.rel_path, qual)
            if fid in store:
                return fid, "FUNCTION"
        return None, "unknown"
    if isinstance(expr, ast.Name):
        fid = top_level_index.get(sf.rel_path, {}).get(expr.id)
        if fid is not None:
            node = store.get(fid)
            return fid, (node.kind if node is not None else "unknown")
        # Not defined in THIS module — maybe it's imported here (e.g.
        # `from qdrant_ops import _qdrant_upsert` re-exported and then
        # passed straight through to register()); chase the NAME slot's
        # own ALIASES edge(s) to the real definition before giving up.
        nid = name_node_id(sf.rel_path, expr.id)
        resolved = _follow_alias_to_callable(store, nid)
        if resolved is not None:
            target_id, _conf, _extra = resolved
            tnode = store.get(target_id)
            return target_id, (tnode.kind if tnode is not None else "unknown")
        if nid in store:
            return nid, "NAME"
        return None, "unknown"
    if isinstance(expr, ast.Constant):
        return None, "LITERAL"
    return None, "unknown"


def _bind_call_args(store: GraphStore, spec: InjectorSpec, call_node: ast.Call) -> dict[str, ast.expr]:
    fn_node = store.get(spec.fn_id)
    param_order = [p["name"] for p in fn_node.attrs.get("params", [])] if fn_node is not None else []
    bound: dict[str, ast.expr] = {}
    for i, arg in enumerate(call_node.args):
        if i < len(param_order):
            bound[param_order[i]] = arg
    for kw in call_node.keywords:
        if kw.arg is not None:
            bound[kw.arg] = kw.value
    return bound


def _emit_inject(store: GraphStore, callsite_id: str, enclosing_id: str, owning_module: str,
                  global_ident: str, value_id, value_kind: str, param_name: str, dict_key: Optional[str],
                  line: int) -> None:
    dst = name_node_id(owning_module, global_ident)
    attrs = {"value": value_id, "value_kind": value_kind, "param": param_name, "line": line}
    if dict_key is not None:
        attrs["key"] = dict_key
    store.add_edge(Edge(src=callsite_id, dst=dst, kind="INJECTS", confidence=Confidence.PROBABLE, attrs=attrs))
    if value_id is not None and value_kind == "FUNCTION":
        store.add_edge(Edge(src=enclosing_id, dst=value_id, kind="REFERENCES", confidence=Confidence.PROVEN,
                             attrs={"form": "injected-argument", "param": param_name, "key": dict_key, "line": line}))


def extract_injections(store: GraphStore, sources: list[SourceFile], scopes: dict[str, ModuleScope],
                        top_level_index) -> None:
    """REG-FN: whole-corpus pass. Finds every `<module>.register(...)`-shaped
    call site targeting a function with an InjectorSpec, matches call
    arguments to injected params/deps-dict keys, and emits INJECTS
    (CALLSITE -> NAME, value=resolved arg) plus a REFERENCES edge for any
    argument that is itself a function/lambda reference."""
    rules = load_rules()
    specs = _build_injector_specs(scopes, rules)
    if not specs:
        return
    for sf in sources:
        scope = scopes.get(sf.rel_path)
        if scope is None or sf.tree is None:
            continue

        def on_node(node: ast.AST, fi: Optional[FunctionInfo], _sf=sf, _scope=scope) -> None:
            if not isinstance(node, ast.Call):
                return
            target_fid = _resolve_register_call_target(store, _sf, top_level_index, node)
            if target_fid is None or target_fid not in specs:
                return
            spec = specs[target_fid]
            bound = _bind_call_args(store, spec, node)
            callsite_id = callsite_node_id(_sf.rel_path, node.lineno, node.col_offset,
                                            getattr(node, "end_lineno", node.lineno),
                                            getattr(node, "end_col_offset", node.col_offset))
            enclosing_id = function_node_id(_sf.rel_path, fi.qualname) if fi is not None else module_node_id(_sf.rel_path)

            for param_name, global_ident in spec.direct.items():
                expr = bound.get(param_name)
                if expr is None:
                    continue
                value_id, value_kind = _resolve_injected_value(store, _sf, _scope, top_level_index, expr)
                _emit_inject(store, callsite_id, enclosing_id, spec.owning_module, global_ident,
                             value_id, value_kind, param_name, None, node.lineno)

            dict_params = {p for (p, _k) in spec.dict_keys}
            for dict_param in dict_params:
                dict_expr = bound.get(dict_param)
                if not isinstance(dict_expr, ast.Dict):
                    continue
                literal_map = {
                    k.value: v for k, v in zip(dict_expr.keys, dict_expr.values)
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                }
                for (p, key), global_ident in spec.dict_keys.items():
                    if p != dict_param:
                        continue
                    vexpr = literal_map.get(key)
                    if vexpr is None:
                        continue
                    value_id, value_kind = _resolve_injected_value(store, _sf, _scope, top_level_index, vexpr)
                    _emit_inject(store, callsite_id, enclosing_id, spec.owning_module, global_ident,
                                 value_id, value_kind, dict_param, key, node.lineno)

        walk_module(scope, on_node)
