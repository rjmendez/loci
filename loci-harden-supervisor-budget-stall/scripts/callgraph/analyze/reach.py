"""Forward/backward closures over the graph, at whatever confidence tier the
caller asks for. This is deliberately small: it backs `cg callers`, `cg
reach`, `cg entrypoints`, and `cg dead`'s "what does an ENTRYPOINT actually
reach" question, without inventing a second traversal per command.

A MODULE-level statement executes automatically the instant the module is
imported — Python doesn't wait for anyone to "call" module top level. Since
this corpus is one monolithic MCP server where every module IS imported
(directly or transitively) by server.py, every MODULE node is treated as a
trivially-reachable root for `reachable_functions()`'s closure (NOT for
`callers`/`reach`, which answer a narrower question about one symbol). This
is what keeps `register()` helper functions — never externally callable by
name, but executed unconditionally at import time — off the dead list
without requiring a full import-order simulation. It is also exactly the
kind of approximation `cg dead`'s own footer says out loud: this tool
reports "no path it can model", not "safe to delete".
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional

from ..model import Confidence, Edge, GraphStore, Node

# DISPATCHES joined CALLS/REFERENCES once extract/dispatch.py's probable
# tier existed: a callsite this tool cannot pin to one target (dict-of-
# callables fan-out, a widened injected global) still names every real
# candidate, and those candidates are exactly as "reachable" as a proven
# call for dead-code/reach purposes — just at PROBABLE confidence, same as
# any other rung-4+ edge.
CALL_EDGE_KINDS = ("CALLS", "REFERENCES", "DISPATCHES")
CLOSURE_EDGE_KINDS = ("ENTERS", "REGISTERS")


def _lambda_host(store: GraphStore, fn_id: str) -> Optional[str]:
    """The nearest enclosing NON-lambda FUNCTION of a lambda, or its MODULE.

    A lambda has no name to be called by: it is always consumed in place --
    passed, returned, or stored -- at the point it is written. So its body
    is reachable exactly when the scope that CONSTRUCTS it is reachable, and
    attributing its callsites to that scope is precise, not a widening.

    Without this, everything called only from inside a lambda is reported
    dead. Measured on the real corpus at HEAD: mcp/server.py's health block
    dispatches nine `_health_probe_*` functions through
    `lambda: _health_probe_qdrant_reachable(qdrant_url, qd)` and every one of
    them was a false positive on `cg dead`.
    """
    node = store.get(fn_id)
    if node is None or not node.attrs.get("is_lambda"):
        return None
    qual = node.attrs.get("qualname", "")
    path = node.path or ""
    while ".<locals>." in qual:
        qual = qual.rsplit(".<locals>.", 1)[0]
        host_id = f"fn:{path}::{qual}"
        host = store.get(host_id)
        if host is not None and host.kind == "FUNCTION" and not host.attrs.get("is_lambda"):
            return host_id
    mod_id = f"mod:{path}"
    return mod_id if mod_id in store else None


def _callsites_by_enclosing(store: GraphStore) -> dict[Optional[str], list[Node]]:
    idx: dict[Optional[str], list[Node]] = {}
    for n in store.nodes_of_kind("CALLSITE"):
        encl = n.attrs.get("enclosing_fn")
        idx.setdefault(encl, []).append(n)
        # A callsite inside a lambda ALSO belongs to the scope that builds
        # the lambda — see _lambda_host.
        if encl is not None:
            host = _lambda_host(store, encl)
            if host is not None and host != encl:
                idx.setdefault(host, []).append(n)
    return idx


def reachable_functions(store: GraphStore, conf_floor: Confidence = Confidence.PROBABLE,
                         extra_roots: Iterable[str] = ()) -> set[str]:
    """Every FUNCTION id reachable from any ENTRYPOINT, or trivially via
    module-execution, following CALLS/REFERENCES (through CALLSITEs) and
    REGISTERS/ENTERS edges at or above conf_floor."""
    callsites = _callsites_by_enclosing(store)
    visited: set[str] = set()
    frontier: list[str] = []

    def seed(node_id: str) -> None:
        if node_id not in visited:
            visited.add(node_id)
            frontier.append(node_id)

    for n in store.nodes_of_kind("ENTRYPOINT"):
        seed(n.id)
    for n in store.nodes_of_kind("MODULE"):
        seed(n.id)
    for rid in extra_roots:
        seed(rid)

    while frontier:
        nxt: list[str] = []
        for nid in frontier:
            for kind in CLOSURE_EDGE_KINDS:
                for e in store.out_edges(nid, kind):
                    if e.confidence >= conf_floor and e.dst not in visited:
                        visited.add(e.dst)
                        nxt.append(e.dst)
            for cs in callsites.get(nid, []):
                for kind in CALL_EDGE_KINDS:
                    for e in store.out_edges(cs.id, kind):
                        if e.confidence >= conf_floor and e.dst not in visited:
                            visited.add(e.dst)
                            nxt.append(e.dst)
            # REFERENCES whose src is a FUNCTION/MODULE directly (MAN-LOOP's
            # register()-body tuple, MAN-DICT's module-level dict literal),
            # not routed via a CALLSITE.
            for e in store.out_edges(nid, "REFERENCES"):
                if e.confidence >= conf_floor and e.dst not in visited:
                    visited.add(e.dst)
                    nxt.append(e.dst)
        frontier = nxt

    return {nid for nid in visited if (nd := store.get(nid)) is not None and nd.kind == "FUNCTION"}


@dataclass
class CallerRow:
    kind: str          # "CALLS" | "REFERENCES" | "DISPATCHES"
    edge: Edge
    site_node: Optional[Node]   # the CALLSITE (for CALLS/DISPATCHES) or None (REFERENCES)


def direct_callers(store: GraphStore, target_id: str) -> list[CallerRow]:
    rows: list[CallerRow] = []
    for e in store.in_edges(target_id, "CALLS"):
        rows.append(CallerRow("CALLS", e, store.get(e.src)))
    for e in store.in_edges(target_id, "DISPATCHES"):
        rows.append(CallerRow("DISPATCHES", e, store.get(e.src)))
    for e in store.in_edges(target_id, "REFERENCES"):
        rows.append(CallerRow("REFERENCES", e, None))
    return rows


def resolve_symbol(store: GraphStore, symbol: str) -> list[str]:
    """A FUNCTION id, a `module::qualname` shorthand, or a bare name (which
    may be ambiguous — every module-level FUNCTION with that exact
    qualname is returned so the caller can print the candidate set rather
    than silently pick one)."""
    if symbol in store and store.get(symbol).kind == "FUNCTION":
        return [symbol]
    if "::" in symbol and not symbol.startswith("fn:"):
        cand = f"fn:{symbol}"
        if cand in store:
            return [cand]
    exact = sorted({n.id for n in store.nodes_of_kind("FUNCTION") if n.attrs.get("qualname") == symbol})
    if exact:
        return exact
    suffix = sorted({
        n.id for n in store.nodes_of_kind("FUNCTION")
        if n.id.endswith(f"::{symbol}") or n.id.endswith(f".{symbol}")
    })
    return suffix


def function_at_line(store: GraphStore, path: str, line: int) -> Optional[str]:
    """The innermost FUNCTION at `path` whose [lineno, end_lineno] span
    contains `line`, or the MODULE id if no function's span does (the line
    is top-level module code). Ties among nested functions are broken by
    picking the one with the LARGEST start line — the most specific,
    innermost span containing the target line."""
    best: Optional[Node] = None
    for n in store.nodes_of_kind("FUNCTION"):
        if n.path != path or n.line is None:
            continue
        end = n.attrs.get("end_lineno", n.line)
        if n.line <= line <= end:
            if best is None or n.line > best.line:
                best = n
    if best is not None:
        return best.id
    mod_id = f"mod:{path}"
    return mod_id if mod_id in store else None


def forward_calls(store: GraphStore, fn_id: str, depth: int = 3,
                   conf_floor: Confidence = Confidence.PROBABLE) -> list[tuple[int, Edge, Node]]:
    """BFS out from fn_id's own CALLSITEs, depth-limited. Returns
    (hop_depth, CALLS-or-REFERENCES edge, callsite-or-None) tuples in
    discovery order; a target visited at a shallower depth is not repeated
    at a deeper one."""
    callsites = _callsites_by_enclosing(store)
    out: list[tuple[int, Edge, Node]] = []
    seen_targets: set[str] = {fn_id}
    frontier = [fn_id]
    d = 0
    while frontier and d < depth:
        d += 1
        nxt: list[str] = []
        for nid in frontier:
            for cs in callsites.get(nid, []):
                for kind in CALL_EDGE_KINDS:
                    for e in store.out_edges(cs.id, kind):
                        if e.confidence < conf_floor or e.dst in seen_targets:
                            continue
                        seen_targets.add(e.dst)
                        out.append((d, e, cs))
                        tnode = store.get(e.dst)
                        if tnode is not None and tnode.kind == "FUNCTION":
                            nxt.append(e.dst)
            for kind in CALL_EDGE_KINDS:
                for e in store.out_edges(nid, kind):
                    if e.confidence < conf_floor or e.dst in seen_targets:
                        continue
                    seen_targets.add(e.dst)
                    out.append((d, e, None))
                    tnode = store.get(e.dst)
                    if tnode is not None and tnode.kind == "FUNCTION":
                        nxt.append(e.dst)
        frontier = nxt
    return out


# ---------------------------------------------------------------------------
# `cg paths` / `cg reach --to` / `cg entrypoints`
# ---------------------------------------------------------------------------


@dataclass
class PathHop:
    edge: Edge
    site: Optional[Node]   # the CALLSITE this hop travelled through, or None (a REFERENCES/DISPATCHES-off-a-FUNCTION hop)


def path_confidence(hops: list["PathHop"]) -> Confidence:
    """A path's confidence is the MINIMUM over its hops, never an average —
    one PROBABLE link in an otherwise-PROVEN chain demotes the whole
    answer, because the chain is only as trustworthy as its weakest edge."""
    return Confidence.combine(h.edge.confidence for h in hops)


def shortest_path(store: GraphStore, src_id: str, dst_id: str,
                   conf_floor: Confidence = Confidence.UNPROVEN, max_depth: int = 20) -> Optional[list[PathHop]]:
    """BFS shortest hop-chain from src_id to dst_id, over the same edge set
    forward_calls/reachable_functions use (CALLS/REFERENCES/DISPATCHES,
    through each source function's own CALLSITEs). Returns the hop list in
    call order (empty list if src_id == dst_id), or None if dst_id is not
    reachable within max_depth hops at conf_floor or better. Breadth-first
    so the FIRST path found is shortest by hop count — ties are broken by
    store iteration order, not by confidence (callers wanting the highest-
    confidence path should raise conf_floor and re-run, not sort after)."""
    if src_id == dst_id:
        return []
    callsites = _callsites_by_enclosing(store)
    visited = {src_id}
    queue: deque = deque([(src_id, [])])
    while queue:
        nid, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        candidates: list[tuple[Edge, Optional[Node]]] = []
        for cs in callsites.get(nid, []):
            for kind in CALL_EDGE_KINDS:
                for e in store.out_edges(cs.id, kind):
                    candidates.append((e, cs))
        for kind in CALL_EDGE_KINDS:
            for e in store.out_edges(nid, kind):
                candidates.append((e, None))
        for e, cs in candidates:
            if e.confidence < conf_floor or e.dst in visited:
                continue
            new_path = path + [PathHop(e, cs)]
            if e.dst == dst_id:
                return new_path
            visited.add(e.dst)
            tnode = store.get(e.dst)
            if tnode is not None and tnode.kind == "FUNCTION":
                queue.append((e.dst, new_path))
    return None


def entrypoints_reaching(store: GraphStore, start_id: str,
                          conf_floor: Confidence = Confidence.PROBABLE) -> dict[str, Confidence]:
    """Reverse closure from a FUNCTION (or MODULE) id back to every
    ENTRYPOINT that can reach it — the backward mirror of
    reachable_functions' forward walk, over the identical edge set (ENTERS/
    REGISTERS/REFERENCES walked directly; CALLS/DISPATCHES walked back via
    each CALLSITE's own `enclosing_fn`, since those edges originate at a
    CALLSITE, not a FUNCTION). Returns {entrypoint_id: best confidence seen
    on any path to it} — `cg entrypoints` prints this as "reachable from N
    entry points", one line per key."""
    visited = {start_id}
    frontier = [start_id]
    entrypoints: dict[str, Confidence] = {}
    while frontier:
        nxt: list[str] = []
        for nid in frontier:
            for e in store.in_edges(nid, "ENTERS"):
                if e.confidence < conf_floor:
                    continue
                best = entrypoints.get(e.src)
                if best is None or e.confidence > best:
                    entrypoints[e.src] = e.confidence
            for e in store.in_edges(nid, "REGISTERS"):
                if e.confidence >= conf_floor and e.src not in visited:
                    visited.add(e.src)
                    nxt.append(e.src)
            for e in store.in_edges(nid, "REFERENCES"):
                if e.confidence >= conf_floor and e.src not in visited:
                    visited.add(e.src)
                    nxt.append(e.src)
            for kind in ("CALLS", "DISPATCHES"):
                for e in store.in_edges(nid, kind):
                    if e.confidence < conf_floor:
                        continue
                    site = store.get(e.src)
                    encl = site.attrs.get("enclosing_fn") if site is not None else None
                    if encl is not None and encl not in visited:
                        visited.add(encl)
                        nxt.append(encl)
        frontier = nxt
    return entrypoints
