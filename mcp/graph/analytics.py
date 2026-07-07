"""Higher-level code<->memory analytics — composed from the graph.queries primitives.

These are the "tools built with the tooling": each function layers the composable
primitives in :mod:`graph.queries` (plus a focused Cypher read where a primitive does
not fit exactly) into an answer to a concrete question. All are fail-open: on an
unavailable store or any error they return an empty structure of the documented shape.

- impact_report(ks, symbol)            — change blast radius: callers + findings +
                                          investigations affected if you touch a symbol/class.
- finding_code_context(ks, finding_id) — the code a finding references, each symbol
                                          wrapped with its callers/callees (review context).
- related_investigations_via_code(ks, investigation_id) — other investigations that
                                          reference the SAME code (code-mediated case links).
"""
from __future__ import annotations

from typing import Any

from . import queries as Q


def _ok(ks: Any) -> bool:
    try:
        return bool(ks) and ks.available()
    except Exception:
        return False


def _q(ks: Any, cypher: str, params: dict | None = None) -> list:
    """Fail-open read via the store's read-only code_query."""
    try:
        return ks.code_query(cypher, params) or []
    except Exception:
        return []


def _enclosing_class_id(symbol_id: str) -> str | None:
    """"file::A.B.m" -> "file::A.B" (the member's enclosing symbol), else None."""
    if "::" not in symbol_id:
        return None
    file, _, qual = symbol_id.partition("::")
    if "." not in qual:
        return None
    return f"{file}::{qual.rpartition('.')[0]}"


def impact_report(ks: Any, symbol: str, hops: int = 3, limit: int = 200) -> dict:
    """Change blast radius for a symbol (method) or class name.

    Resolves ``symbol`` to matching CodeSymbols; for a class target, also folds in
    that class's own methods (so "change this class" includes its members' callers).
    Returns the transitive CALLS callers plus every Finding/Investigation that
    REFERENCES any target or caller — the code AND memory that a change touches.

    Shape: ``{symbol, resolved:[{id,name,kind}], direct_callers:[names],
    transitive_caller_count, referencing_finding_count,
    investigations:[{id, finding_count, sample}], co_referenced:[{name,count}]}``.
    """
    empty = {"symbol": symbol, "resolved": [], "direct_callers": [],
             "transitive_caller_count": 0, "referencing_finding_count": 0,
             "investigations": [], "co_referenced": []}
    if not _ok(ks) or not symbol:
        return empty
    hops = max(1, min(int(hops or 1), 6))

    # 1) Resolve the name to target symbol ids (by id first, else by name).
    rows = _q(ks, "MATCH (s:CodeSymbol) WHERE s.id = $s OR s.name = $s "
                  "RETURN s.id, s.name, s.kind, s.file", {"s": symbol})
    if not rows:
        return empty
    targets = {r[0]: {"id": r[0], "name": r[1], "kind": r[2], "file": r[3]} for r in rows}

    # 2) For class-like targets, fold in the class's own methods (prefix match on id).
    type_kinds = {"class", "interface", "enum", "struct", "trait"}
    for t in list(targets.values()):
        if t["kind"] in type_kinds:
            prefix = f"{t['file']}::{t['name']}."
            for r in _q(ks, "MATCH (s:CodeSymbol) WHERE s.file = $f AND s.kind = 'method' "
                            "RETURN s.id, s.name, s.kind, s.file", {"f": t["file"]}):
                if r[0].startswith(prefix):
                    targets.setdefault(r[0], {"id": r[0], "name": r[1], "kind": r[2], "file": r[3]})
    target_ids = list(targets)

    # 3) Transitive CALLS callers of any target (up to hops).
    caller_rows = _q(ks,
        f"MATCH (c:CodeSymbol)-[:CALLS*1..{hops}]->(t:CodeSymbol) "
        "WHERE t.id IN $ids AND NOT c.id IN $ids "
        "RETURN DISTINCT c.id, c.name LIMIT $lim",
        {"ids": target_ids, "lim": int(limit)})
    caller_ids = [r[0] for r in caller_rows]
    direct = _q(ks,
        "MATCH (c:CodeSymbol)-[:CALLS]->(t:CodeSymbol) WHERE t.id IN $ids AND NOT c.id IN $ids "
        "RETURN DISTINCT c.name LIMIT 40", {"ids": target_ids})

    # 4) Findings referencing any target OR caller; grouped by investigation.
    affected = target_ids + caller_ids
    fi = _q(ks,
        "MATCH (f:Finding)-[:REFERENCES]->(s:CodeSymbol) WHERE s.id IN $ids "
        "RETURN DISTINCT f.id, f.investigation, f.text", {"ids": affected})
    by_inv: dict[str, dict] = {}
    seen_f: set = set()
    for fid, inv, text in fi:
        seen_f.add(fid)
        inv = inv or "(none)"
        b = by_inv.setdefault(inv, {"id": inv, "finding_count": 0, "sample": []})
        b["finding_count"] += 1
        if len(b["sample"]) < 2:
            b["sample"].append((text or "")[:140])
    investigations = sorted(by_inv.values(), key=lambda x: -x["finding_count"])

    # 5) Symbols most co-referenced with the target (subsystem neighbours in memory).
    co = _q(ks,
        "MATCH (f:Finding)-[:REFERENCES]->(t:CodeSymbol) WHERE t.id IN $ids "
        "MATCH (f)-[:REFERENCES]->(o:CodeSymbol) WHERE NOT o.id IN $ids "
        "RETURN o.name, count(DISTINCT f) AS c ORDER BY c DESC LIMIT 8", {"ids": target_ids})

    return {
        "symbol": symbol,
        "resolved": [{"id": t["id"], "name": t["name"], "kind": t["kind"]} for t in targets.values()],
        "direct_callers": [r[0] for r in direct],
        "transitive_caller_count": len(caller_ids),
        "referencing_finding_count": len(seen_f),
        "investigations": investigations,
        "co_referenced": [{"name": r[0], "count": r[1]} for r in co],
    }


def finding_code_context(ks: Any, finding_id: str, neighbours: int = 6) -> dict:
    """The code a finding references, each symbol wrapped with its callers/callees.

    Shape: ``{finding_id, text, symbols:[{id,name,kind,file,line,
    callers:[names], callees:[names]}]}``. Composes queries.finding_symbols +
    a CALLS neighbourhood read per symbol.
    """
    empty = {"finding_id": finding_id, "text": "", "symbols": []}
    if not _ok(ks) or not finding_id:
        return empty
    trow = _q(ks, "MATCH (f:Finding {id:$i}) RETURN f.text", {"i": finding_id})
    text = trow[0][0] if trow else ""
    out = []
    for s in Q.finding_symbols(ks, finding_id):
        sid = s.get("id")
        callers = [r[0] for r in _q(ks,
            "MATCH (c:CodeSymbol)-[:CALLS]->(s:CodeSymbol {id:$i}) RETURN DISTINCT c.name LIMIT $n",
            {"i": sid, "n": int(neighbours)})]
        callees = [r[0] for r in _q(ks,
            "MATCH (s:CodeSymbol {id:$i})-[:CALLS]->(c:CodeSymbol) RETURN DISTINCT c.name LIMIT $n",
            {"i": sid, "n": int(neighbours)})]
        out.append({**s, "callers": callers, "callees": callees})
    return {"finding_id": finding_id, "text": (text or "")[:400], "symbols": out}


def subsystem_report(ks: Any, anchor: str, limit: int = 15) -> dict:
    """Full picture of a subsystem = the CodeSymbols under a file or path prefix.

    ``anchor`` matches a CodeFile whose path equals it OR starts with it (a single
    file, a directory, or a package path). Unions the code boundary (who calls in /
    what it calls out) with the memory footprint (hotspot symbols, investigations
    analysing it) — the composed answer to "tell me everything about this subsystem".

    Shape: ``{anchor, files:[...], symbol_count, kinds:{kind:n},
    inbound_callers:[{name,file}], outbound_callees:[{name,file}],
    hotspot_symbols:[{name,findings}], investigations:[{id,finding_count}]}``.
    """
    empty = {"anchor": anchor, "files": [], "symbol_count": 0, "kinds": {},
             "inbound_callers": [], "outbound_callees": [], "hotspot_symbols": [],
             "investigations": []}
    if not _ok(ks) or not anchor:
        return empty

    # Resolve the subsystem's files (exact path or path-prefix) in Python — robust
    # across Kuzu string-fn differences.
    all_files = [r[0] for r in _q(ks, "MATCH (c:CodeFile) RETURN c.path")]
    files = [f for f in all_files if f == anchor or f.startswith(anchor)]
    if not files:
        return empty

    kinds: dict[str, int] = {}
    symbol_count = 0
    for name, cnt in _q(ks, "MATCH (s:CodeSymbol) WHERE s.file IN $f RETURN s.kind, count(*) AS c",
                        {"f": files}):
        kinds[name] = int(cnt)
        symbol_count += int(cnt)

    inbound = [{"name": r[0], "file": r[1]} for r in _q(ks,
        "MATCH (c:CodeSymbol)-[:CALLS]->(t:CodeSymbol) WHERE t.file IN $f AND NOT c.file IN $f "
        "RETURN DISTINCT c.name, c.file LIMIT $lim", {"f": files, "lim": int(limit)})]
    outbound = [{"name": r[0], "file": r[1]} for r in _q(ks,
        "MATCH (s:CodeSymbol)-[:CALLS]->(o:CodeSymbol) WHERE s.file IN $f AND NOT o.file IN $f "
        "RETURN DISTINCT o.name, o.file LIMIT $lim", {"f": files, "lim": int(limit)})]
    hotspots = [{"name": r[0], "findings": int(r[1])} for r in _q(ks,
        "MATCH (fd:Finding)-[:REFERENCES]->(s:CodeSymbol) WHERE s.file IN $f "
        "RETURN s.name, count(DISTINCT fd) AS c ORDER BY c DESC LIMIT $lim", {"f": files, "lim": int(limit)})]
    invs = [{"id": r[0], "finding_count": int(r[1])} for r in _q(ks,
        "MATCH (fd:Finding)-[:REFERENCES]->(s:CodeSymbol) WHERE s.file IN $f "
        "MATCH (fd)-[:IN_INVESTIGATION]->(i:Investigation) "
        "RETURN i.id, count(DISTINCT fd) AS c ORDER BY c DESC LIMIT $lim", {"f": files, "lim": int(limit)})]

    return {"anchor": anchor, "files": files, "symbol_count": symbol_count, "kinds": kinds,
            "inbound_callers": inbound, "outbound_callees": outbound,
            "hotspot_symbols": hotspots, "investigations": invs}


def related_investigations_via_code(ks: Any, investigation_id: str, limit: int = 15) -> list:
    """Other investigations whose findings reference the SAME CodeSymbols as this one.

    Code-mediated case linkage: ranked by count of shared referenced symbols.
    Returns ``[{investigation, shared_symbols, sample_symbols:[names]}]``.
    """
    if not _ok(ks) or not investigation_id:
        return []
    rows = _q(ks,
        "MATCH (f:Finding)-[:IN_INVESTIGATION]->(a:Investigation {id:$id}) "
        "MATCH (f)-[:REFERENCES]->(s:CodeSymbol)<-[:REFERENCES]-(f2:Finding) "
        "MATCH (f2)-[:IN_INVESTIGATION]->(b:Investigation) WHERE b.id <> $id "
        "RETURN b.id, count(DISTINCT s.id) AS shared, collect(DISTINCT s.name) AS names "
        "ORDER BY shared DESC LIMIT $lim",
        {"id": investigation_id, "lim": int(limit)})
    out = []
    for iid, shared, names in rows:
        sample = list(names)[:6] if isinstance(names, list) else []
        out.append({"investigation": iid, "shared_symbols": int(shared), "sample_symbols": sample})
    return out
