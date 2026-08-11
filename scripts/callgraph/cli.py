"""argparse front end for `cg`.

Slice 1 implements: build, modules, defs, imports, aliases.
Slice 2 adds: callers, reach, registry, name, dead, holes — on the CALLSITE/
CALLS resolution ladder (rungs 1-3), REGISTRY/REGISTERS/ENTERS/ENTRYPOINT,
and READS_NAME/WRITES_NAME/INJECTS extract/* modules built alongside it.
Slice 3 (build steps 7-9) adds: the probable tier (extract/dispatch.py's
CALLS rungs 4-6 + DISPATCHES), `--why` on callers/reach rows, and the
traversal commands `reach --to`/`paths`/`entrypoints`/`explain`, all riding
the same pipeline.build_graph() — no new node/edge kind needed a schema
change, this slice only adds resolution power and ways to see it.
Slice 4/step 13 (hardening) adds: `selftest` (build_steps step 13's
standalone health check, see selftest.py) and `limits` (prints
docs/LIMITS.md), plus `--format dot` on build/reach/callers/paths.
`whatchanged` (diff -> blast radius) is the one query from the design not
yet built.

Exit codes: 0 clean, 1 findings present (`selftest` uses this for a failed
check; the design's other analysis commands are lead generators and are
deliberately never wired to a nonzero exit — see docs/LIMITS.md), 2 tool
error (bad args, git failure, etc).

stdlib only.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from . import config
from .analyze.deadcode import dead_functions
from .analyze.flags import rank_flags
from .analyze.literalaudit import literal_table, materialize_near_miss_edges, near_miss_pairs, orphans
from .analyze.nameaudit import dangling_globals, read_by_tests, write_no_read
from .analyze.reach import (
    Confidence, direct_callers, entrypoints_reaching, forward_calls, function_at_line,
    path_confidence, resolve_symbol, shortest_path,
)
from .ingest import load_test_sources
from .model import GraphStore
from .pipeline import BuildResult, build_graph
from .selftest import run_selftest


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--rev", default=None, help="git rev to read instead of the working tree (e.g. HEAD)")
    p.add_argument("--scope", action="append", default=None, metavar="PATH_PREFIX",
                    help="restrict to files whose repo-relative path starts with this prefix; repeatable")
    p.add_argument("--format", choices=["text", "json", "dot"], default="text")
    p.add_argument("--no-cache", action="store_true", help="currently a no-op (see ingest.py)")
    p.add_argument("--conf", choices=["proven", "probable", "all"], default="probable",
                    help="confidence floor for edges considered (default: probable)")
    p.add_argument("--depth", type=int, default=None, help="traversal depth limit (command-specific default if omitted)")
    p.add_argument("--why", action="store_true", help="print the rule/because behind each row")


def _conf_floor(args: argparse.Namespace) -> Optional[Confidence]:
    return None if args.conf == "all" else Confidence.parse(args.conf)


def _build(args: argparse.Namespace) -> BuildResult:
    try:
        return build_graph(rev=args.rev, scope_prefixes=args.scope, no_cache=args.no_cache)
    except Exception as exc:  # tool error, not a finding
        print(f"callgraph: build failed: {exc}", file=sys.stderr)
        raise SystemExit(2)


def _footer(result: BuildResult) -> str:
    m = result.meta
    scope_note = f" scope={'+'.join(m.scope_prefixes)}" if m.scope_prefixes else ""
    return (f"# source: {m.source}{scope_note}  files: {m.file_count}  "
            f"parse errors: {m.error_count}  built in {m.elapsed_s:.3f}s")


# -- build ------------------------------------------------------------------

def cmd_build(args: argparse.Namespace) -> int:
    result = _build(args)
    if args.format == "dot":
        print(result.store.to_dot())
        return 0
    if args.format == "json":
        print(json.dumps({
            "source": result.meta.source, "file_count": result.meta.file_count,
            "error_count": result.meta.error_count, "errors": result.meta.errors,
            "elapsed_s": result.meta.elapsed_s, "stats": result.store.stats(),
        }, indent=2, sort_keys=True))
        return 0
    print(_footer(result))
    print()
    stats = result.store.stats()
    for key in sorted(stats):
        if key in ("nodes", "edges"):
            continue
        print(f"  {key:28s} {stats[key]:6d}")
    print(f"  {'TOTAL nodes':28s} {stats['nodes']:6d}")
    print(f"  {'TOTAL edges':28s} {stats['edges']:6d}")
    if result.meta.errors:
        print("\nparse errors:")
        for path, err in result.meta.errors:
            print(f"  {path}: {err}")
    return 0


# -- modules ------------------------------------------------------------------

def cmd_modules(args: argparse.Namespace) -> int:
    result = _build(args)
    rows = []
    for node in sorted(result.store.nodes_of_kind("MODULE"), key=lambda n: n.path or ""):
        rows.append({
            "path": node.path, "kind": node.attrs.get("kind"),
            "importable_as": node.attrs.get("importable_as"),
            "has_main": node.attrs.get("has_main"),
        })
    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0
    print(_footer(result))
    print()
    for r in rows:
        names = ", ".join(r["importable_as"])
        main_flag = " [__main__]" if r["has_main"] else ""
        print(f"  {r['path']:45s} {r['kind']:15s} {{{names}}}{main_flag}")
    print(f"\n  {len(rows)} modules")
    return 0


# -- defs ------------------------------------------------------------------

def cmd_defs(args: argparse.Namespace) -> int:
    result = _build(args)
    store = result.store
    functions = sorted(store.nodes_of_kind("FUNCTION"), key=lambda n: (n.path or "", n.line or 0))
    classes = sorted(store.nodes_of_kind("CLASS"), key=lambda n: (n.path or "", n.line or 0))
    names = list(store.nodes_of_kind("NAME"))

    if args.filter:
        functions = [n for n in functions if args.filter in (n.path or "")]
        classes = [n for n in classes if args.filter in (n.path or "")]

    module_level_fns = [n for n in functions if not n.attrs.get("is_nested") and not n.attrs.get("is_method")]
    function_local_imports = sum(
        1 for e in store.edges_of_kind("IMPORTS") if e.attrs.get("scope") == "function-local"
    )
    dangling_globals = sum(
        1 for n in names if "global-only" in n.attrs.get("binding_kind", [])
    )

    if args.format == "json":
        payload = {
            "functions": [
                {"id": n.id, "path": n.path, "line": n.line, "qualname": n.attrs.get("qualname"),
                 "is_method": n.attrs.get("is_method"), "is_nested": n.attrs.get("is_nested"),
                 "is_lambda": n.attrs.get("is_lambda"), "decorators": n.attrs.get("decorators")}
                for n in functions
            ],
            "classes": [
                {"id": n.id, "path": n.path, "line": n.line, "qualname": n.attrs.get("qualname"),
                 "bases": n.attrs.get("bases")}
                for n in classes
            ],
            "summary": {
                "functions_total": len(functions), "functions_module_level": len(module_level_fns),
                "classes_total": len(classes), "names_total": len(names),
                "function_local_imports": function_local_imports, "dangling_globals": dangling_globals,
            },
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(_footer(result))
    print()
    if args.filter or args.all:
        for n in functions:
            deco = f"  @{', @'.join(_decorator_heads(n))}" if n.attrs.get("decorators") else ""
            print(f"  {n.path}:{n.line}  fn  {n.attrs.get('qualname')}{deco}")
        for n in classes:
            print(f"  {n.path}:{n.line}  cls {n.attrs.get('qualname')}  bases=({', '.join(n.attrs.get('bases', []))})")
    print()
    print(f"  functions: {len(functions)} total, {len(module_level_fns)} module-level (not nested/method)")
    print(f"  classes:   {len(classes)}")
    print(f"  NAME slots: {len(names)}  (dangling-global candidates: {dangling_globals})")
    print(f"  function-local imports: {function_local_imports}")
    unmatched = [
        e for n in functions for e in store.out_edges(n.id, "DECORATED_BY")
        if e.attrs.get("classification") == "unknown"
    ]
    if unmatched:
        print(f"\n  {len(unmatched)} decorator(s) not classified by rules.toml:")
        for e in unmatched:
            print(f"    {e.attrs.get('raw')}  (line {e.attrs.get('line')})")
    return 0


def _decorator_heads(fn_node) -> list[str]:
    return [d.split("(")[0] for d in fn_node.attrs.get("decorators", [])]


# -- imports ------------------------------------------------------------------

def cmd_imports(args: argparse.Namespace) -> int:
    result = _build(args)
    edges = list(result.store.edges_of_kind("IMPORTS"))
    if args.lazy:
        edges = [e for e in edges if e.attrs.get("scope") == "function-local"]
    if args.unresolved:
        edges = [e for e in edges if e.attrs.get("resolved_via") == "unresolved"]

    edges.sort(key=lambda e: (e.src, e.attrs.get("line") or 0))

    if args.format == "json":
        print(json.dumps([
            {"src": e.src, "dst": e.dst, "confidence": e.confidence.name, **e.attrs}
            for e in edges
        ], indent=2, sort_keys=True))
        return 0

    print(_footer(result))
    print()
    for e in edges:
        src_path = e.src.split(":", 1)[1] if e.src.startswith("mod:") else e.src
        loc = f"{src_path}:{e.attrs.get('line')}"
        scope_tag = "[local]" if e.attrs.get("scope") == "function-local" else "        "
        fn_tag = f" in {e.attrs.get('enclosing_fn')}" if e.attrs.get("enclosing_fn") else ""
        print(f"  {loc:38s} {scope_tag} {e.attrs.get('module') or '(bare)':30s} "
              f"-> {e.dst:45s} {e.confidence.name:9s} via={e.attrs.get('resolved_via')}{fn_tag}")
    print(f"\n  {len(edges)} import edges")
    if args.unresolved and not edges:
        print("  (none — every import resolves to a corpus module, stdlib, or a third-party package in the venv)")
    return 0


# -- aliases (debug/verification command for step 3) ------------------------

def cmd_aliases(args: argparse.Namespace) -> int:
    result = _build(args)
    edges = list(result.store.edges_of_kind("ALIASES"))
    if args.filter:
        edges = [e for e in edges if args.filter in e.src]
    edges.sort(key=lambda e: (e.src, e.attrs.get("line") or 0))
    if args.format == "json":
        print(json.dumps([
            {"src": e.src, "dst": e.dst, "confidence": e.confidence.name, **e.attrs} for e in edges
        ], indent=2, sort_keys=True))
        return 0
    print(_footer(result))
    print()
    for e in edges:
        print(f"  {e.src}  --[{e.attrs.get('form')}]-->  {e.dst}  ({e.confidence.name})")
    print(f"\n  {len(edges)} alias edges")
    return 0


# -- callers ------------------------------------------------------------------


def cmd_callers(args: argparse.Namespace) -> int:
    result = _build(args)
    store = result.store
    targets = resolve_symbol(store, args.symbol)
    if not targets:
        print(f"# no FUNCTION found matching '{args.symbol}'", file=sys.stderr)
        return 2
    if len(targets) > 1:
        print(f"# ambiguous symbol '{args.symbol}' — {len(targets)} candidates (pass one of these ids to disambiguate):")
        for t in sorted(targets):
            print(f"  {t}")
        return 0

    target = targets[0]
    conf_floor = _conf_floor(args)
    rows = direct_callers(store, target)
    if conf_floor is not None:
        rows = [r for r in rows if r.edge.confidence >= conf_floor]

    if args.format == "dot":
        node_ids = {target} | {r.edge.src for r in rows} | {r.edge.dst for r in rows}
        print(store.to_dot(node_ids, [r.edge for r in rows]))
        return 0
    if args.format == "json":
        print(json.dumps([
            {"kind": r.kind, "src": r.edge.src, "confidence": r.edge.confidence.name, **r.edge.attrs}
            for r in rows
        ], indent=2, sort_keys=True))
        return 0

    print(_footer(result))
    print(f"# callers of {target}")
    print()
    for tier in (Confidence.PROVEN, Confidence.PROBABLE, Confidence.UNPROVEN):
        tier_rows = [r for r in rows if r.edge.confidence == tier]
        if not tier_rows:
            continue
        print(f"-- {tier.name} " + "-" * (60 - len(tier.name)))
        for r in tier_rows:
            _print_caller_row(store, r, tier, args.why)
        print()
    print(f"  {len(rows)} caller row(s)")
    return 0


def _print_caller_row(store: GraphStore, r, tier: Confidence, why: bool) -> None:
    if r.kind == "CALLS":
        site = r.site_node
        loc = f"{site.path}:{site.line}:{site.col}" if site is not None else r.edge.src
        extra = f"  because={r.edge.attrs['because']}" if "because" in r.edge.attrs else ""
        rung = r.edge.attrs.get("rung", "")
        callee = site.attrs.get("callee", "") if site is not None else ""
        print(f"  {loc:42s} {tier.name:9s} call[{rung}]  {callee}(){extra}")
    elif r.kind == "DISPATCHES":
        site = r.site_node
        loc = f"{site.path}:{site.line}:{site.col}" if site is not None else r.edge.src
        fanout = r.edge.attrs.get("fanout", "?")
        selector = r.edge.attrs.get("selector", "")
        callee = site.attrs.get("callee", "") if site is not None else ""
        print(f"  {loc:42s} {tier.name:9s} dispatch  1 of {fanout}  {callee}(){f' on {selector}' if selector else ''}")
    else:  # REFERENCES
        src_node = store.get(r.edge.src)
        loc = f"{src_node.path}:{r.edge.attrs.get('line', '?')}" if src_node is not None else r.edge.src
        print(f"  {loc:42s} {tier.name:9s} passed-by-ref  form={r.edge.attrs.get('form')}")
    if why:
        rule = r.edge.attrs.get("rung") or r.edge.attrs.get("reason") or r.edge.attrs.get("form") or r.kind
        because = r.edge.attrs.get("because")
        registry = r.edge.attrs.get("registry")
        extra_bits = [f"rule={rule}"]
        if because:
            extra_bits.append(f"because={because}")
        if registry:
            extra_bits.append(f"registry={registry}")
        print(f"      -- why: {'  '.join(extra_bits)}")


# -- reach --------------------------------------------------------------------


def cmd_reach(args: argparse.Namespace) -> int:
    result = _build(args)
    store = result.store
    targets = resolve_symbol(store, args.symbol)
    if not targets:
        print(f"# no FUNCTION found matching '{args.symbol}'", file=sys.stderr)
        return 2
    if len(targets) > 1:
        print(f"# ambiguous symbol '{args.symbol}' — {len(targets)} candidates:")
        for t in sorted(targets):
            print(f"  {t}")
        return 0
    target = targets[0]
    conf_floor = _conf_floor(args) or Confidence.UNPROVEN

    if args.to:
        return _print_path(args, result, target, args.to, conf_floor)

    depth = args.depth or 3
    hops = forward_calls(store, target, depth=depth, conf_floor=conf_floor)

    if args.format == "dot":
        node_ids = {target} | {e.src for _d, e, _s in hops} | {e.dst for _d, e, _s in hops}
        print(store.to_dot(node_ids, [e for _d, e, _s in hops]))
        return 0
    if args.format == "json":
        print(json.dumps([
            {"depth": d, "dst": e.dst, "confidence": e.confidence.name, **e.attrs} for d, e, _ in hops
        ], indent=2, sort_keys=True))
        return 0

    print(_footer(result))
    print(f"# forward reach from {target} (depth<={depth})")
    print()
    for d, e, site in hops:
        loc = f"{site.path}:{site.line}:{site.col}" if site is not None else "(passed-by-ref)"
        tag = "dispatch" if e.kind == "DISPATCHES" else e.kind.lower()
        print(f"  depth={d}  {loc:38s} {e.confidence.name:9s} {tag:10s} -> {e.dst}")
    print(f"\n  {len(hops)} node(s) reached")
    return 0


# -- paths ----------------------------------------------------------------


def _print_path(args: argparse.Namespace, result: BuildResult, src: str, to_symbol: str, conf_floor: Confidence) -> int:
    store = result.store
    dst_targets = resolve_symbol(store, to_symbol)
    if not dst_targets:
        print(f"# no FUNCTION found matching '{to_symbol}'", file=sys.stderr)
        return 2
    if len(dst_targets) > 1:
        print(f"# ambiguous symbol '{to_symbol}' — {len(dst_targets)} candidates:")
        for t in sorted(dst_targets):
            print(f"  {t}")
        return 0
    dst = dst_targets[0]
    hops = shortest_path(store, src, dst, conf_floor=conf_floor, max_depth=args.depth or 20)

    if args.format == "dot":
        if not hops:
            print(store.to_dot({src}, []))
            return 0
        node_ids = {src} | {h.edge.src for h in hops} | {h.edge.dst for h in hops}
        print(store.to_dot(node_ids, [h.edge for h in hops]))
        return 0
    if args.format == "json":
        if hops is None:
            print(json.dumps({"reachable": False}))
            return 0
        print(json.dumps({
            "reachable": True, "confidence": path_confidence(hops).name,
            "hops": [
                {"dst": h.edge.dst, "confidence": h.edge.confidence.name,
                 "site": (f"{h.site.path}:{h.site.line}:{h.site.col}" if h.site is not None else None), **h.edge.attrs}
                for h in hops
            ],
        }, indent=2, sort_keys=True))
        return 0

    print(_footer(result))
    print(f"# path {src} -> {dst}")
    print()
    if hops is None:
        print(f"  UNREACHABLE within depth={args.depth or 20} at --conf {args.conf}")
        return 0
    if not hops:
        print("  (src == dst)")
        return 0
    for i, h in enumerate(hops, start=1):
        loc = f"{h.site.path}:{h.site.line}:{h.site.col}" if h.site is not None else "(passed-by-ref)"
        tag = "dispatch" if h.edge.kind == "DISPATCHES" else h.edge.kind.lower()
        because = f"  because={h.edge.attrs['because']}" if "because" in h.edge.attrs else ""
        print(f"  hop {i}: {loc:38s} {h.edge.confidence.name:9s} {tag:10s} -> {h.edge.dst}{because}")
    print(f"\n  path confidence: {path_confidence(hops).name}  ({len(hops)} hop(s))")
    return 0


def cmd_paths(args: argparse.Namespace) -> int:
    result = _build(args)
    store = result.store
    src_targets = resolve_symbol(store, args.src)
    if not src_targets:
        print(f"# no FUNCTION found matching '{args.src}'", file=sys.stderr)
        return 2
    if len(src_targets) > 1:
        print(f"# ambiguous symbol '{args.src}' — {len(src_targets)} candidates:")
        for t in sorted(src_targets):
            print(f"  {t}")
        return 0
    conf_floor = _conf_floor(args) or Confidence.UNPROVEN
    return _print_path(args, result, src_targets[0], args.dst, conf_floor)


# -- registry -----------------------------------------------------------------


def cmd_registry(args: argparse.Namespace) -> int:
    result = _build(args)
    store = result.store

    if args.unmatched:
        unmatched = [e for e in store.edges_of_kind("DECORATED_BY") if e.attrs.get("classification") == "unknown"]
        if args.format == "json":
            print(json.dumps([{"src": e.src, "raw": e.attrs.get("raw"), "line": e.attrs.get("line")} for e in unmatched],
                              indent=2, sort_keys=True))
            return 0
        print(_footer(result))
        print()
        for e in unmatched:
            print(f"  {e.src}  @{e.attrs.get('raw')}  (line {e.attrs.get('line')})")
        print(f"\n  {len(unmatched)} unclassified registering-looking decorator(s)")
        return 0

    regs = sorted(store.nodes_of_kind("REGISTRY"), key=lambda n: (n.path or "", n.id))
    if args.tool:
        regs = [r for r in regs if args.tool in r.attrs.get("keys", [])]

    if args.format == "json":
        print(json.dumps([
            {
                "id": r.id, "path": r.path, "mechanism": r.attrs.get("mechanism"),
                "member_count": r.attrs.get("member_count"), "keys": r.attrs.get("keys"),
                "declaration_line": r.attrs.get("declaration_line"),
            } for r in regs
        ], indent=2, sort_keys=True))
        return 0

    print(_footer(result))
    print()
    total_members = 0
    for r in regs:
        members = store.out_edges(r.id, "REGISTERS")
        total_members += len(members)
        print(f"  {r.id:45s} mechanism={r.attrs.get('mechanism'):15s} members={len(members):3d}  declared={r.path}:{r.attrs.get('declaration_line')}")
        if args.tool:
            for m in sorted(members, key=lambda e: e.attrs.get("key", "")):
                print(f"      {m.attrs.get('key'):40s} -> {m.dst}")
    print(f"\n  {len(regs)} registries, {total_members} total REGISTERS edges")
    return 0


# -- name ---------------------------------------------------------------------


def cmd_name(args: argparse.Namespace) -> int:
    result = _build(args)
    store = result.store
    ident = args.ident
    name_nodes = [n for n in store.nodes_of_kind("NAME") if n.id.endswith(f"::{ident}")]
    if args.module:
        name_nodes = [n for n in name_nodes if n.path == args.module]
    if not name_nodes:
        print(f"# no NAME slot found for '{ident}'" + (f" in {args.module}" if args.module else ""), file=sys.stderr)
        return 2

    rows = []
    for n in name_nodes:
        for e in store.in_edges(n.id, "WRITES_NAME"):
            rows.append(("WRITE", n, e, e.src))
        for e in store.in_edges(n.id, "READS_NAME"):
            rows.append(("READ", n, e, e.src))
        for e in store.in_edges(n.id, "INJECTS"):
            rows.append(("INJECT", n, e, e.src))

    if args.format == "json":
        print(json.dumps([
            {"slot": n.id, "kind": k, "src": src, **e.attrs} for (k, n, e, src) in rows
        ], indent=2, sort_keys=True))
        return 0

    print(_footer(result))
    print(f"# NAME slots matching '::{ident}'" + (f" (module={args.module})" if args.module else ""))
    print()
    slot_modules = {n.path for n in name_nodes}
    touching_modules = {store.get(src).path if store.get(src) is not None else src for (_k, _n, _e, src) in rows}
    for n in sorted(name_nodes, key=lambda n: n.path or ""):
        print(f"  slot {n.id}  binding_kind={n.attrs.get('binding_kind')}")
        for (kind, slot_n, e, src) in sorted(rows, key=lambda r: (r[1].id != n.id, r[3])):
            if slot_n.id != n.id:
                continue
            src_node = store.get(src)
            loc = f"{src_node.path}:{e.attrs.get('line', src_node.line)}" if src_node is not None else src
            if kind == "WRITE":
                print(f"    WRITE  {loc:42s} via={e.attrs.get('via')}")
            elif kind == "READ":
                cross = "  (cross-module)" if e.attrs.get("cross_module") else ""
                print(f"    READ   {loc:42s} in_call={e.attrs.get('in_call_position')}{cross}")
            else:
                print(f"    INJECT {loc:42s} value={e.attrs.get('value')} param={e.attrs.get('param')} key={e.attrs.get('key')}")
        print()
    if len(touching_modules | slot_modules) > 1:
        print(f"  WARNING: this global is written in one module and touched from {len(touching_modules)} module(s) total"
              f" — cross-module coupling via {'global-stmt/INJECTS' if any(r[0]=='INJECT' for r in rows) else 'global-stmt'}.")
    print(f"  {len(rows)} row(s) across {len(name_nodes)} slot(s)")
    return 0


# -- dead ---------------------------------------------------------------------


def cmd_dead(args: argparse.Namespace) -> int:
    result = _build(args)
    store = result.store
    dead = dead_functions(store, args.scope)
    dead.sort(key=lambda n: (n.path or "", n.line or 0))

    if args.format == "json":
        print(json.dumps([{"id": n.id, "path": n.path, "line": n.line} for n in dead], indent=2, sort_keys=True))
        return 0

    print(_footer(result))
    roots_count = sum(1 for _ in store.nodes_of_kind("ENTRYPOINT"))
    print(f"# dead here means no path this tool can model; it does NOT mean safe to delete —")
    print(f"# check roots.toml, shell scripts, systemd units and cron first. ({roots_count} entrypoint(s) known)")
    print()
    for n in dead:
        print(f"  UNREACHABLE  {n.path}:{n.line}  {n.attrs.get('qualname')}")
    print(f"\n  {len(dead)} unreachable function(s)")
    return 0


# -- holes --------------------------------------------------------------------


def cmd_holes(args: argparse.Namespace) -> int:
    result = _build(args)
    store = result.store
    edges = list(store.in_edges("?", "CALLS"))
    if args.scope:
        edges = [e for e in edges if any(e.src.split(":", 2)[1].startswith(p) for p in args.scope if e.src.startswith("call:"))]
    by_reason: dict[str, int] = {}
    for e in edges:
        by_reason[e.attrs.get("reason", "unknown")] = by_reason.get(e.attrs.get("reason", "unknown"), 0) + 1

    if args.format == "json":
        print(json.dumps({"total": len(edges), "by_reason": by_reason}, indent=2, sort_keys=True))
        return 0

    print(_footer(result))
    print()
    for reason, count in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"  {count:6d}  {reason}")
    print(f"\n  {len(edges)} unresolved callsite(s) total / {len(list(store.nodes_of_kind('CALLSITE')))} callsites")
    return 0


# -- writes-dead -------------------------------------------------------------


def _ident_of(name_id: str) -> str:
    return name_id.rsplit("::", 1)[-1]


def cmd_writes_dead(args: argparse.Namespace) -> int:
    result = _build(args)
    store = result.store
    dangling = dangling_globals(store, args.scope)
    weak = write_no_read(store, args.scope)

    tests_hits: dict[str, list[str]] = {}
    if args.include_tests:
        idents = {_ident_of(f.slot.id) for f in weak}
        test_sources = load_test_sources(rev=args.rev)
        tests_hits = read_by_tests(idents, test_sources)
        weak = [f for f in weak if _ident_of(f.slot.id) not in tests_hits]

    if args.format == "json":
        print(json.dumps({
            "dangling_global": [
                {
                    "slot": f.slot.id, "path": f.slot.path,
                    "write_sites": [{"src": e.src, "line": e.attrs.get("line")} for e in f.global_writes],
                    "real_slots": [{"id": rn.id, "path": rn.path, "line": rn.line} for rn in f.real_slots],
                } for f in dangling
            ],
            "write_no_read": [
                {"slot": f.slot.id, "path": f.slot.path,
                 "write_sites": [{"src": e.src, "line": e.attrs.get("line"), "via": e.attrs.get("via")} for e in f.writes]}
                for f in weak
            ],
            "include_tests": args.include_tests,
            "tests_excluded_idents": sorted(tests_hits) if args.include_tests else None,
        }, indent=2, sort_keys=True))
        return 0

    print(_footer(result))
    print()
    print("-- DANGLING-GLOBAL (near-zero false positives) " + "-" * 30)
    for f in dangling:
        ident = _ident_of(f.slot.id)
        write_lines = ", ".join(str(e.attrs.get("line")) for e in f.global_writes) or "?"
        print(f"  {f.slot.path}:{write_lines}  DANGLING-GLOBAL  {ident}"
              f" — declared global here, never bound at module level in this module")
        if f.real_slots:
            real = f.real_slots[0]
            others = f" (+{len(f.real_slots) - 1} more)" if len(f.real_slots) > 1 else ""
            print(f"      real slot: {real.path}:{real.line}{others}  (different slot — same ident, different module)")
        else:
            print("      no module-level binding of this ident found anywhere else in the corpus either")
    print(f"\n  {len(dangling)} dangling-global finding(s)")

    print()
    print("-- WRITE-WITH-NO-READ (weak check, see docs/LIMITS.md) " + "-" * 24)
    for f in weak:
        ident = _ident_of(f.slot.id)
        write_lines = ", ".join(str(e.attrs.get("line")) for e in f.writes)
        print(f"  {f.slot.path}:{write_lines}  WRITE-NO-READ  {ident}")
    print(f"\n  {len(weak)} write-with-no-read finding(s)"
          + (f"  ({len(tests_hits)} ident(s) excluded via --include-tests)" if args.include_tests else ""))
    return 0


# -- entrypoints ----------------------------------------------------------


def cmd_entrypoints(args: argparse.Namespace) -> int:
    result = _build(args)
    store = result.store
    try:
        path, line_str = args.location.rsplit(":", 1)
        line = int(line_str)
    except ValueError:
        print(f"# expected <file>:<line>, got '{args.location}'", file=sys.stderr)
        return 2
    start_id = function_at_line(store, path, line)
    if start_id is None:
        print(f"# no module or function found at {path}:{line}", file=sys.stderr)
        return 2
    conf_floor = _conf_floor(args) or Confidence.UNPROVEN
    entries = entrypoints_reaching(store, start_id, conf_floor=conf_floor)

    if args.format == "json":
        print(json.dumps({
            "location": f"{path}:{line}", "start": start_id,
            "entrypoints": [{"id": eid, "confidence": c.name} for eid, c in sorted(entries.items())],
        }, indent=2, sort_keys=True))
        return 0

    print(_footer(result))
    node = store.get(start_id)
    label = node.attrs.get("qualname", start_id) if (node is not None and start_id.startswith("fn:")) else start_id
    print(f"# {path}:{line}  ->  {label}")
    print()
    if not entries:
        print("  reachable from 0 known entry points")
        print("  # dead here means no path this tool can model, not proven unreachable — see docs/LIMITS.md")
        return 0
    print(f"  reachable from {len(entries)} entry point(s):")
    for eid, conf in sorted(entries.items(), key=lambda kv: (-kv[1], kv[0])):
        enode = store.get(eid)
        trust = enode.attrs.get("trust") if enode is not None else "?"
        print(f"    {eid:55s} {conf.name:9s} trust={trust}")
    return 0


# -- explain ----------------------------------------------------------------


def _explain_target_sites(store: GraphStore, target: str):
    node = store.get(target)
    if node is not None and node.kind == "CALLSITE":
        return [node]
    parts = target.rsplit(":", 2)
    path = line = col = None
    if len(parts) == 3 and parts[1].lstrip("-").isdigit() and parts[2].lstrip("-").isdigit():
        path, line, col = parts[0], int(parts[1]), int(parts[2])
    elif len(parts) >= 2 and parts[-1].lstrip("-").isdigit():
        path, line = target.rsplit(":", 1)
        line = int(line)
    if path is None:
        return []
    return [
        n for n in store.nodes_of_kind("CALLSITE")
        if n.path == path and n.line == line and (col is None or n.col == col)
    ]


def cmd_explain(args: argparse.Namespace) -> int:
    result = _build(args)
    store = result.store
    sites = _explain_target_sites(store, args.target)
    if not sites:
        print(f"# no CALLSITE found for '{args.target}' (pass a CALLSITE id, or <file>:<line>[:<col>])", file=sys.stderr)
        return 2

    if args.format == "json":
        out = []
        for site in sites:
            out.append({
                "callsite": site.id, "form": site.attrs.get("form"), "callee": site.attrs.get("callee"),
                "enclosing_fn": site.attrs.get("enclosing_fn"),
                "calls": [{"dst": e.dst, "confidence": e.confidence.name, **e.attrs} for e in store.out_edges(site.id, "CALLS")],
                "dispatches": [{"dst": e.dst, "confidence": e.confidence.name, **e.attrs} for e in store.out_edges(site.id, "DISPATCHES")],
            })
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0

    print(_footer(result))
    for site in sites:
        print(f"# {site.id}")
        print(f"  {site.path}:{site.line}:{site.col}  form={site.attrs.get('form')}  callee={site.attrs.get('callee')}()")
        print(f"  enclosing: {site.attrs.get('enclosing_fn')}")
        for e in store.out_edges(site.id, "CALLS"):
            bits = [f"CALLS -> {e.dst}", e.confidence.name]
            for key in ("rung", "reason", "because", "alternatives"):
                if key in e.attrs:
                    bits.append(f"{key}={e.attrs[key]}")
            print("    " + "  ".join(bits))
        for e in store.out_edges(site.id, "DISPATCHES"):
            print(f"    DISPATCHES -> {e.dst}  {e.confidence.name}  1 of {e.attrs.get('fanout')}"
                  f"  selector={e.attrs.get('selector')}  registry={e.attrs.get('registry')}")
        print()
    return 0


# -- literals -----------------------------------------------------------------


def cmd_literals(args: argparse.Namespace) -> int:
    result = _build(args)
    store = result.store
    flavour = "path" if args.paths else ("key" if args.keys else None)

    if args.near_miss:
        producer_only, consumer_only = orphans(store, flavour)
        pairs = near_miss_pairs(producer_only, consumer_only)
        if args.glob:
            pairs = [p for p in pairs if args.glob in p.producer.match_text or args.glob in p.consumer.match_text]
        materialize_near_miss_edges(store, pairs)
        if args.format == "json":
            print(json.dumps([
                {"producer": nm.producer.match_text, "consumer": nm.consumer.match_text,
                 "shared_stem": nm.shared_stem, "distance": nm.distance,
                 "producer_sites": [f"{s.path}:{s.line}" for s in nm.producer.produce_sites],
                 "consumer_sites": [f"{s.path}:{s.line}" for s in nm.consumer.consume_sites]}
                for nm in pairs
            ], indent=2, sort_keys=True))
            return 0
        print(_footer(result))
        print("# NEAR_MISS pairs — lead, not finding; never influences reachability/dead-code/exit code")
        print()
        for nm in pairs:
            print(f"  {nm.producer.match_text!r}  <~>  {nm.consumer.match_text!r}"
                  f"  stem={nm.shared_stem}  distance={nm.distance}")
            for s in nm.producer.produce_sites:
                print(f"      PRODUCED {s.path}:{s.line}  role={s.role}")
            for s in nm.consumer.consume_sites:
                print(f"      CONSUMED {s.path}:{s.line}  role={s.role}")
        print(f"\n  {len(pairs)} near-miss pair(s)")
        return 0

    if args.orphans:
        producer_only, consumer_only = orphans(store, flavour)
        if args.glob:
            producer_only = [g for g in producer_only if args.glob in g.match_text]
            consumer_only = [g for g in consumer_only if args.glob in g.match_text]
        if args.format == "json":
            print(json.dumps({
                "producer_only": [{"text": g.match_text, "sites": [f"{s.path}:{s.line}" for s in g.produce_sites]} for g in producer_only],
                "consumer_only": [{"text": g.match_text, "sites": [f"{s.path}:{s.line}" for s in g.consume_sites]} for g in consumer_only],
            }, indent=2, sort_keys=True))
            return 0
        print(_footer(result))
        print()
        for g in producer_only:
            sites = ", ".join(f"{s.path}:{s.line}" for s in g.produce_sites)
            print(f"  {g.match_text}  —  PRODUCED {sites}  —  CONSUMED BY NOBODY")
        for g in consumer_only:
            sites = ", ".join(f"{s.path}:{s.line}" for s in g.consume_sites)
            print(f"  {g.match_text}  —  CONSUMED {sites}  —  PRODUCED BY NOBODY")
        print(f"\n  {len(producer_only)} producer-only, {len(consumer_only)} consumer-only orphan(s)")
        return 0

    groups = literal_table(store, flavour)
    if args.glob:
        groups = [g for g in groups if args.glob in g.match_text]
    if args.format == "json":
        print(json.dumps([
            {"text": g.match_text, "flavour": g.flavour,
             "produced_by": [f"{s.path}:{s.line}({s.role})" for s in g.produce_sites],
             "consumed_by": [f"{s.path}:{s.line}({s.role})" for s in g.consume_sites]}
            for g in groups
        ], indent=2, sort_keys=True))
        return 0
    print(_footer(result))
    print()
    for g in groups:
        prod = ", ".join(f"{s.path}:{s.line}" for s in g.produce_sites) or "(none)"
        cons = ", ".join(f"{s.path}:{s.line}" for s in g.consume_sites) or "(none)"
        print(f"  {g.match_text:40s} [{g.flavour}]  produced: {prod}  consumed: {cons}")
    print(f"\n  {len(groups)} literal(s)")
    return 0


# -- flags ----------------------------------------------------------------


def cmd_flags(args: argparse.Namespace) -> int:
    result = _build(args)
    store = result.store
    fn_filter = None
    scope = args.scope
    target = args.target
    if target:
        if "::" in target:
            path, qual = target.split("::", 1)
            fn_filter = f"fn:{path}::{qual}"
            scope = None
        else:
            scope = [target]

    rows = rank_flags(store, scope, fn_filter,
                       include_accumulators=getattr(args, "include_accumulators", False))

    if args.format == "json":
        print(json.dumps([
            {
                "ident": r.ident, "fn": r.fn_id, "path": r.binding.path, "score": r.score,
                "escape_form": r.escape_form, "pattern": r.pattern, **r.binding.attrs,
            } for r in rows
        ], indent=2, sort_keys=True))
        return 0

    print(_footer(result))
    print("# ranker, not a detector — scope to one file/function; never a CI gate (see docs/LIMITS.md)")
    print()
    for r in rows:
        b = r.binding
        print(f"  {b.path}  {r.fn_id.split('::', 1)[-1]}::{r.ident}  score={r.score:.2f}  [{r.pattern}]")
        print(f"      init={b.attrs.get('init_line')} (constant)  escapes as {r.escape_form} at "
              f"{b.attrs.get('escape_lines')}")
        for a in b.attrs.get("assign_contexts", []):
            print(f"      assigned True-ish at :{a['line']}  ({a['context']}, {a.get('form', 'assign')})")
        for g in b.attrs.get("guard_exits", []):
            print(f"      guard exit :{g['line']}  {g['kind']}  ({g['context']}) — does NOT assign {r.ident}")
        print()
    print(f"  {len(rows)} candidate row(s)")
    return 0


# -- selftest ----------------------------------------------------------------


def cmd_selftest(args: argparse.Namespace) -> int:
    report = run_selftest()
    if args.format == "json":
        print(json.dumps({
            "ok": report.ok, "elapsed_s": report.elapsed_s,
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in report.checks],
        }, indent=2, sort_keys=True))
        return 0 if report.ok else 1

    print(f"# callgraph selftest — fixtures/ (no git) + one real-corpus build --rev HEAD")
    print()
    for c in report.checks:
        status = "PASS" if c.ok else "FAIL"
        print(f"  [{status}] {c.name}")
        if not c.ok:
            print(f"         {c.detail}")
    print()
    passed = sum(1 for c in report.checks if c.ok)
    print(f"  {passed}/{len(report.checks)} checks passed  ({report.elapsed_s:.3f}s)")
    if not report.ok:
        print("  SELFTEST FAILED")
        return 1
    print("  selftest green")
    return 0


# -- limits -------------------------------------------------------------------


def cmd_limits(args: argparse.Namespace) -> int:
    limits_path = config.PACKAGE_ROOT / "docs" / "LIMITS.md"
    try:
        text = limits_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"callgraph: could not read {limits_path}: {exc}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps({"path": str(limits_path), "text": text}, indent=2))
        return 0
    print(text, end="" if text.endswith("\n") else "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cg", description="Dependency-light static call-graph tool for the loci corpus.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="parse the corpus and print graph stats")
    _add_common_args(p)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("modules", help="list MODULE nodes and their importable names")
    _add_common_args(p)
    p.set_defaults(func=cmd_modules)

    p = sub.add_parser("defs", help="list/summarize FUNCTION, CLASS, NAME nodes")
    _add_common_args(p)
    p.add_argument("filter", nargs="?", default=None, help="only show defs whose path contains this substring")
    p.add_argument("--all", action="store_true", help="list every def, not just the summary")
    p.set_defaults(func=cmd_defs)

    p = sub.add_parser("imports", help="list IMPORTS edges")
    _add_common_args(p)
    p.add_argument("--lazy", action="store_true", help="only function-local imports")
    p.add_argument("--unresolved", action="store_true", help="only imports that resolve to neither corpus, stdlib, nor a third-party package")
    p.set_defaults(func=cmd_imports)

    p = sub.add_parser("aliases", help="list ALIASES edges (debug/verification)")
    _add_common_args(p)
    p.add_argument("filter", nargs="?", default=None, help="only show aliases whose src id contains this substring")
    p.set_defaults(func=cmd_aliases)

    p = sub.add_parser("callers", help="who calls this symbol (direct CALLS + REFERENCES)")
    _add_common_args(p)
    p.add_argument("symbol", help="a FUNCTION id, module::qualname, or bare name")
    p.set_defaults(func=cmd_callers)

    p = sub.add_parser("reach", help="forward closure: what does this symbol call, transitively")
    _add_common_args(p)
    p.add_argument("symbol", help="a FUNCTION id, module::qualname, or bare name")
    p.add_argument("--to", default=None, metavar="SYMBOL",
                    help="degenerate to `cg paths`: print the shortest evidence chain from symbol to this target")
    p.set_defaults(func=cmd_reach)

    p = sub.add_parser("paths", help="shortest evidence chain between two symbols, hop by hop")
    _add_common_args(p)
    p.add_argument("src", help="a FUNCTION id, module::qualname, or bare name")
    p.add_argument("dst", help="a FUNCTION id, module::qualname, or bare name")
    p.set_defaults(func=cmd_paths)

    p = sub.add_parser("entrypoints", help="which ENTRYPOINTs can reach a given file:line")
    _add_common_args(p)
    p.add_argument("location", help="<repo-relative-path>:<line>")
    p.set_defaults(func=cmd_entrypoints)

    p = sub.add_parser("registry", help="list REGISTRY surfaces and their REGISTERS members")
    _add_common_args(p)
    p.add_argument("--tool", default=None, help="only the registry containing this externally-visible key")
    p.add_argument("--unmatched", action="store_true", help="list registering-classified decorators rules.toml failed to key")
    p.set_defaults(func=cmd_registry)

    p = sub.add_parser("name", help="who reads/writes/injects this module-global ident, across the corpus")
    _add_common_args(p)
    p.add_argument("ident", help="the bare identifier, e.g. _get_ladybug")
    p.add_argument("--module", default=None, help="restrict to one module's NAME slot (repo-relative path)")
    p.set_defaults(func=cmd_name)

    p = sub.add_parser("writes-dead", help="DANGLING-GLOBAL and WRITE-WITH-NO-READ NAME-slot audits")
    _add_common_args(p)
    p.add_argument("--include-tests", action="store_true",
                    help="re-check WRITE-WITH-NO-READ findings against a textual scan of tests/ source; "
                         "drop any whose ident appears there")
    p.set_defaults(func=cmd_writes_dead)

    p = sub.add_parser("literals", help="path/key literal producer-consumer table, orphans, and near-miss leads")
    _add_common_args(p)
    p.add_argument("glob", nargs="?", default=None, help="only literals whose tail text contains this substring")
    p.add_argument("--paths", action="store_true", help="restrict to path-like literals")
    p.add_argument("--keys", action="store_true", help="restrict to key-like literals")
    p.add_argument("--orphans", action="store_true", help="only literals with occurrences in exactly one direction")
    p.add_argument("--near-miss", action="store_true", help="pair producer-only/consumer-only orphans by shared stem")
    p.set_defaults(func=cmd_literals)

    p = sub.add_parser("flags", help="partial-assignment ranker over escaping constant-initialized locals")
    _add_common_args(p)
    p.add_argument("target", nargs="?", default=None,
                    help="a repo-relative path (file scope) or path::qualname (single function)")
    p.add_argument("--include-accumulators", action="store_true",
                    help="also list counters/running totals (locals reassigned only by += or "
                         "to a computed expression). Off by default: they match the flag shape "
                         "but are correct by construction — see docs/LIMITS.md")
    p.set_defaults(func=cmd_flags)

    p = sub.add_parser("dead", help="functions unreachable from any known ENTRYPOINT (lead generator, not proof)")
    _add_common_args(p)
    p.set_defaults(func=cmd_dead)

    p = sub.add_parser("holes", help="inbound edges to the `?` UNRESOLVED sink, grouped by reason")
    _add_common_args(p)
    p.set_defaults(func=cmd_holes)

    p = sub.add_parser("explain", help="the rule/evidence behind one callsite's CALLS/DISPATCHES edges")
    _add_common_args(p)
    p.add_argument("target", help="a CALLSITE id (call:<path>:<line>:<col>:...), or <path>:<line>[:<col>]")
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("selftest", help="build over fixtures/ + one real HEAD build, assert the full expected finding set")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("limits", help="print docs/LIMITS.md — the honest failure-mode catalogue")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.set_defaults(func=cmd_limits)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
