"""Composable code<->memory query primitives over :class:`KuzuStore`.

These are the low-level building blocks higher-level MCP tools compose from.
Every function:

* takes a :class:`~graph.kuzu_store.KuzuStore` as its first argument,
* issues exactly **one** focused read query via ``ks.code_query`` (the
  read-only, write-guarded entry point) and shapes the result in Python,
* is **fail-open**: on an unavailable store, a bad argument, or any raised
  exception it returns an empty structure of the right shape (never raises),
* returns only JSON-serializable data (dicts / lists / str / int / bool).

The graph carries two overlaid subgraphs (code + investigation); see
``graph.kuzu_store`` for the schema. Symbol ids are ``"file::Qualname"`` and the
``REFERENCES`` edge has no properties — its existence *is* the finding<->symbol
link.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("loci-mcp.queries")

__all__ = [
    "subgraph",
    "symbol_findings",
    "finding_symbols",
    "investigation_footprint",
    "symbol_impact",
    "related_findings_via_code",
]

# Primary-key property per node label. Also the whitelist of anchor labels the
# generic ``subgraph`` primitive accepts — values are injected into a query
# string, so this MUST stay a closed, code-controlled set (never user input).
_PK = {
    "CodeFile": "path",
    "CodeSymbol": "id",
    "Finding": "id",
    "Entity": "name",
    "Investigation": "id",
}


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _available(ks: Any) -> bool:
    """True only when the store exists and reports itself usable. Fail-open."""
    try:
        return bool(ks) and bool(ks.available())
    except Exception:
        return False


def _finding_row(row: list) -> dict:
    """Shape the canonical 7-column finding projection into a dict."""
    return {
        "id": row[0], "investigation": row[1], "ftype": row[2],
        "text": row[3], "confidence": row[4], "source": row[5], "ts": row[6],
    }


def _symbol_row(row: list) -> dict:
    """Shape the canonical 6-column CodeSymbol projection into a dict."""
    return {
        "id": row[0], "name": row[1], "kind": row[2],
        "file": row[3], "line": row[4], "lang": row[5],
    }


def _props(node: dict) -> dict:
    """Public, non-null properties of a returned Kuzu node dict.

    Kuzu returns a node as a dict of every node table's property (irrelevant
    ones ``None``) plus internal ``_id`` / ``_label`` keys. Drop the internals
    and the nulls to leave a clean, JSON-serializable prop bag.
    """
    return {
        k: v for k, v in node.items()
        if not str(k).startswith("_") and v is not None
    }


def _internal_id(id_dict: Any) -> Optional[tuple]:
    """Hashable identity for a Kuzu internal node id (``{table, offset}``)."""
    if isinstance(id_dict, dict):
        return (id_dict.get("table"), id_dict.get("offset"))
    return None


def _rel_pattern(rels: Any) -> str:
    """Build the type filter for a var-length pattern: ``""`` or ``":A|B"``.

    Rel-type names are injected into the query string, so keep this to plain
    identifiers; anything falsy yields an untyped (all-rels) pattern.
    """
    if not rels:
        return ""
    if isinstance(rels, str):
        rels = [rels]
    types = [str(t).strip() for t in rels if str(t).strip()]
    return (":" + "|".join(types)) if types else ""


def _int(value: Any, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(value))
    except Exception:
        return default


# --------------------------------------------------------------------------- #
# THE generic building block
# --------------------------------------------------------------------------- #
def subgraph(
    ks: Any,
    anchor_label: str,
    anchor_key: str,
    hops: int = 1,
    rels: Any = None,
    limit: int = 200,
) -> dict:
    """BFS neighbourhood around one node.

    Walks up to ``hops`` undirected steps from the anchor node
    (``anchor_label`` in the :data:`_PK` whitelist, ``anchor_key`` its primary
    key) over the given ``rels`` types (default: all rel types). The anchor is
    always included even when it has no neighbours.

    Returns ``{"nodes": [{"label", "key", "props"}],
    "edges": [{"from", "to", "rel"}]}`` capped at ``limit`` nodes; only edges
    whose both endpoints survive the cap are emitted.
    """
    empty: dict = {"nodes": [], "edges": []}
    if not _available(ks):
        return empty
    pk = _PK.get(anchor_label)
    if not pk or not anchor_key:
        return empty

    h = _int(hops, 1)
    lim = _int(limit, 200)
    rel_types = _rel_pattern(rels)
    # Bound how much the DB materialises; nodes are still hard-capped below.
    row_limit = lim * 25

    cy = (
        f"MATCH (a:{anchor_label} {{{pk}:$k}}) "
        f"OPTIONAL MATCH p = (a)-[{rel_types}*1..{h}]-(b) "
        "RETURN a, nodes(p), rels(p) "
        "LIMIT $rlim"
    )
    try:
        rows = ks.code_query(cy, {"k": str(anchor_key), "rlim": row_limit})
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("subgraph query failed: %s", exc)
        return empty

    # Pass 1: ordered-unique node collection (anchor first — it is row[0] of
    # every row) truncated to the node cap.
    ordered: list[tuple] = []
    seen: set = set()
    for row in rows:
        anchor_node = row[0]
        path_nodes = row[1] or []
        for node in [anchor_node, *path_nodes]:
            if not isinstance(node, dict):
                continue
            nid = _internal_id(node.get("_id"))
            if nid is None or nid in seen:
                continue
            seen.add(nid)
            ordered.append((nid, node))

    kept = ordered[:lim]
    kept_ids = {nid for nid, _ in kept}
    id_to_key: dict = {}
    nodes_out: list = []
    for nid, node in kept:
        label = node.get("_label")
        key = node.get(_PK.get(label, ""))
        id_to_key[nid] = key
        nodes_out.append({"label": label, "key": key, "props": _props(node)})

    # Pass 2: edges among surviving nodes, deduped.
    edges_out: list = []
    edge_seen: set = set()
    for row in rows:
        for rel in (row[2] or []):
            if not isinstance(rel, dict):
                continue
            src = _internal_id(rel.get("_src"))
            dst = _internal_id(rel.get("_dst"))
            if src not in kept_ids or dst not in kept_ids:
                continue
            rel_label = rel.get("_label")
            sig = (src, dst, rel_label)
            if sig in edge_seen:
                continue
            edge_seen.add(sig)
            edges_out.append({
                "from": id_to_key.get(src),
                "to": id_to_key.get(dst),
                "rel": rel_label,
            })

    return {"nodes": nodes_out, "edges": edges_out}


# --------------------------------------------------------------------------- #
# Finding <-> symbol primitives
# --------------------------------------------------------------------------- #
def symbol_findings(ks: Any, symbol: str, limit: int = 50) -> list:
    """Findings that REFERENCE ``symbol`` (matched by symbol id **or** name)."""
    if not _available(ks) or not symbol:
        return []
    lim = _int(limit, 50)
    cy = (
        "MATCH (f:Finding)-[:REFERENCES]->(s:CodeSymbol) "
        "WHERE s.id = $q OR s.name = $q "
        "RETURN DISTINCT f.id, f.investigation, f.ftype, f.text, "
        "f.confidence, f.source, f.ts "
        "LIMIT $lim"
    )
    try:
        rows = ks.code_query(cy, {"q": str(symbol), "lim": lim})
        return [_finding_row(r) for r in rows]
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("symbol_findings failed: %s", exc)
        return []


def finding_symbols(ks: Any, finding_id: str) -> list:
    """CodeSymbols the finding REFERENCES."""
    if not _available(ks) or not finding_id:
        return []
    cy = (
        "MATCH (f:Finding {id:$fid})-[:REFERENCES]->(s:CodeSymbol) "
        "RETURN DISTINCT s.id, s.name, s.kind, s.file, s.line, s.lang"
    )
    try:
        rows = ks.code_query(cy, {"fid": str(finding_id)})
        return [_symbol_row(r) for r in rows]
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("finding_symbols failed: %s", exc)
        return []


def investigation_footprint(ks: Any, investigation_id: str) -> dict:
    """Code an investigation's findings touch, via REFERENCES.

    Returns ``{"symbols": [...], "files": [...], "finding_count": int}`` where
    ``symbols`` are the referenced CodeSymbols, ``files`` their distinct source
    paths, and ``finding_count`` the number of findings in the investigation.
    """
    empty: dict = {"symbols": [], "files": [], "finding_count": 0}
    if not _available(ks) or not investigation_id:
        return empty
    cy = (
        "MATCH (i:Investigation {id:$id})<-[:IN_INVESTIGATION]-(f:Finding) "
        "OPTIONAL MATCH (f)-[:REFERENCES]->(s:CodeSymbol) "
        "RETURN f.id, s.id, s.name, s.kind, s.file, s.line, s.lang"
    )
    try:
        rows = ks.code_query(cy, {"id": str(investigation_id)})
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("investigation_footprint failed: %s", exc)
        return empty

    findings: set = set()
    symbols: dict = {}
    files: list = []
    files_seen: set = set()
    for r in rows:
        if r[0] is not None:
            findings.add(r[0])
        sid = r[1]
        if sid is not None and sid not in symbols:
            symbols[sid] = _symbol_row([r[1], r[2], r[3], r[4], r[5], r[6]])
            fpath = r[4]
            if fpath and fpath not in files_seen:
                files_seen.add(fpath)
                files.append(fpath)
    return {
        "symbols": list(symbols.values()),
        "files": files,
        "finding_count": len(findings),
    }


def symbol_impact(ks: Any, symbol: str, hops: int = 3, limit: int = 200) -> dict:
    """Blast radius of ``symbol`` (matched by id or name).

    Combines the transitive CALLS callers reaching the symbol (up to ``hops``)
    with the Findings — and their Investigations — that REFERENCE the symbol or
    any of those callers.

    Returns ``{"callers": [...symbols...], "findings": [...], "investigations":
    [...]}``.
    """
    empty: dict = {"callers": [], "findings": [], "investigations": []}
    if not _available(ks) or not symbol:
        return empty
    h = _int(hops, 3)
    lim = _int(limit, 200)
    # One query: gather target + its transitive callers, then fan out to the
    # findings/investigations referencing any of them. is_target flags the
    # anchor so callers can be separated in Python.
    # Carry symbol *ids* (not node values) through the collect/unwind — a Kuzu
    # node bound out of a list literal cannot be re-used as a pattern node, so
    # we re-MATCH each symbol by id before fanning out to its findings.
    cy = (
        "MATCH (target:CodeSymbol) WHERE target.id = $q OR target.name = $q "
        f"OPTIONAL MATCH (caller:CodeSymbol)-[:CALLS*1..{h}]->(target) "
        "WHERE caller.id <> target.id "
        "WITH target.id AS tid, collect(DISTINCT caller.id) AS caller_ids "
        # collect() yields NULL (not []) when there are no callers; coalesce so
        # the target's own findings are never dropped for a caller-less symbol.
        "WITH tid, coalesce(caller_ids, []) + [tid] AS sym_ids "
        "UNWIND sym_ids AS sid "
        "WITH DISTINCT sid, tid "
        "WHERE sid IS NOT NULL "
        "MATCH (sym:CodeSymbol {id: sid}) "
        "OPTIONAL MATCH (f:Finding)-[:REFERENCES]->(sym) "
        "OPTIONAL MATCH (f)-[:IN_INVESTIGATION]->(inv:Investigation) "
        "RETURN sym.id, sym.name, sym.kind, sym.file, sym.line, sym.lang, "
        "(sym.id = tid) AS is_target, "
        "f.id, f.investigation, f.ftype, f.text, f.confidence, f.source, f.ts, "
        "inv.id, inv.title "
        "LIMIT $lim"
    )
    try:
        rows = ks.code_query(cy, {"q": str(symbol), "lim": lim * 10})
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("symbol_impact failed: %s", exc)
        return empty

    callers: dict = {}
    findings: dict = {}
    investigations: dict = {}
    for r in rows:
        sid, sname, skind, sfile, sline, slang = r[0], r[1], r[2], r[3], r[4], r[5]
        is_target = bool(r[6])
        if sid is not None and not is_target and sid not in callers:
            callers[sid] = _symbol_row([sid, sname, skind, sfile, sline, slang])
        fid = r[7]
        if fid is not None and fid not in findings:
            findings[fid] = _finding_row([r[7], r[8], r[9], r[10], r[11], r[12], r[13]])
        inv_id = r[14]
        if inv_id is not None and inv_id not in investigations:
            investigations[inv_id] = {"id": inv_id, "title": r[15]}
        if len(callers) >= lim:
            break
    return {
        "callers": list(callers.values())[:lim],
        "findings": list(findings.values()),
        "investigations": list(investigations.values()),
    }


def related_findings_via_code(ks: Any, finding_id: str, limit: int = 25) -> list:
    """Findings that share a REFERENCED symbol with ``finding_id``.

    Code-mediated similarity: other findings pointing at any CodeSymbol this
    finding also references, ranked by the number of shared symbols. Each result
    carries a ``shared`` count alongside the finding fields.
    """
    if not _available(ks) or not finding_id:
        return []
    lim = _int(limit, 25)
    cy = (
        "MATCH (f:Finding {id:$fid})-[:REFERENCES]->(s:CodeSymbol)"
        "<-[:REFERENCES]-(other:Finding) "
        "WHERE other.id <> $fid "
        "RETURN other.id, other.investigation, other.ftype, other.text, "
        "other.confidence, other.source, other.ts, count(DISTINCT s) AS shared "
        "ORDER BY shared DESC, other.id "
        "LIMIT $lim"
    )
    try:
        rows = ks.code_query(cy, {"fid": str(finding_id), "lim": lim})
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("related_findings_via_code failed: %s", exc)
        return []
    out = []
    for r in rows:
        rec = _finding_row(r)
        rec["shared"] = r[7]
        out.append(rec)
    return out
