"""Code<->memory graph MCP tools — split out of server.py (P1 of the Loci self-review).

Thin wrappers over the graph.* layer; they need only a _get_kuzu accessor, injected by
register(). server.py calls register(mcp, _get_kuzu) after the FastMCP instance exists,
so the tools register identically (FastMCP reads each function signature + docstring).
"""
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("loci-mcp")

_get_kuzu = None  # injected by register()


def code_graph_ingest(path: str, max_files: Optional[int] = None, replace: bool = False) -> str:
    """
    Parse a source file or directory with tree-sitter and ingest its symbol graph
    into the Kuzu code graph (CodeFile / CodeSymbol nodes; DEFINES / CALLS / IMPORTS
    edges). Enables ``code_graph_query`` and the AST-backed code<->memory linkage.

    Supports python, java, kotlin, rust, javascript, typescript/tsx, go (c/c++
    parse but currently yield file-level nodes only). Binary/oversized files and
    common vendor dirs (.git, node_modules, .venv, build, dist, target) are skipped.

    Ingest is additive (MERGE) by default. When ``replace=True`` the existing
    CodeFile/CodeSymbol nodes under this ``path`` (and their DEFINES/CALLS/IMPORTS
    edges + inbound REFERENCES) are deleted BEFORE parsing, so re-ingesting a
    moved or updated checkout stays idempotent and drops stale-path nodes. Only
    code nodes are pruned — Findings / Investigations / Entities are untouched.

    Args:
        path: A source file or a directory root to walk.
        max_files: Optional cap on files parsed (directory mode).
        replace: If True, prune existing code nodes under ``path`` before ingest
            (idempotent re-ingest). Default False preserves additive behaviour.

    Returns:
        JSON with per-run counts {files, symbols, defines, calls, imports} (and a
        ``pruned`` block when ``replace`` is set) or an error.
    """
    ks = _get_kuzu()
    if not ks:
        return json.dumps({"error": "Kuzu graph store unavailable — cannot ingest code graph."})
    try:
        from graph.code_parse import parse_path, parse_source, detect_lang
        p = Path(path).expanduser()
        if not p.exists():
            return json.dumps({"error": f"Path not found: {path}"})
        if p.is_file():
            lang = detect_lang(str(p))
            if not lang:
                return json.dumps({"error": f"Unsupported/undetected language for file: {path}"})
            try:
                src = p.read_bytes()
            except Exception as exc:
                return json.dumps({"error": f"Could not read {path}: {exc!r}"})
            parsed = [parse_source(str(p), src, lang)]
        else:
            parsed = parse_path(str(p), max_files=max_files)
        # Prune stale/duplicate code nodes under this path so re-ingest is clean.
        pruned = ks.delete_code_under(str(p)) if replace else None
        counts = ks.ingest_code(parsed)
        out = {"path": str(p), "parsed_files": len(parsed), "ingested": counts}
        if pruned is not None:
            out["pruned"] = pruned
        return json.dumps(out, indent=2)
    except Exception as exc:
        logger.debug("code_graph_ingest failed: %r", exc)
        return json.dumps({"error": f"code_graph_ingest failed: {exc!r}"})


def code_graph_query(cypher: str, params: Optional[dict] = None) -> str:
    """
    Run a READ-ONLY Cypher query against the Kuzu graph (code symbols + findings +
    entities + investigations) and return the rows.

    Write-shaped queries (CREATE/DELETE/SET/MERGE/DROP/COPY/ALTER) are rejected —
    this tool never mutates the graph. Use it for impact analysis and traversal, e.g.
    finding callers of a symbol, symbols a file defines, or findings that reference a
    given CodeSymbol.

    Node tables: CodeFile(path,lang), CodeSymbol(id,name,kind,file,line,lang),
      Finding(id,investigation,ftype,text,confidence,source,ts), Entity(name,etype,distinctive),
      Investigation(id,title).
    Rel tables: DEFINES(CodeFile->CodeSymbol), CALLS(CodeSymbol->CodeSymbol),
      IMPORTS(CodeFile->CodeFile), REFERENCES(Finding->CodeSymbol), MENTIONS(Finding->Entity),
      DERIVED_FROM(Finding->Finding), IN_INVESTIGATION(Finding->Investigation),
      RELATED(Investigation->Investigation).

    Example: MATCH (c:CodeSymbol)-[:CALLS]->(t:CodeSymbol {name:'helper'}) RETURN c.id, c.file

    Args:
        cypher: A read-only Cypher query.
        params: Optional parameter dict for ``$name`` placeholders.

    Returns:
        JSON with {rows: [...], row_count} or an error (including a write-guard rejection).
    """
    ks = _get_kuzu()
    if not ks:
        return json.dumps({"error": "Kuzu graph store unavailable."})
    try:
        rows = ks.code_query(cypher, params or None)
        return json.dumps({"row_count": len(rows), "rows": rows}, indent=2, default=str)
    except ValueError as exc:  # write-guard rejection
        return json.dumps({"error": f"rejected (read-only tool): {exc}"})
    except Exception as exc:
        logger.debug("code_graph_query failed: %r", exc)
        return json.dumps({"error": f"code_graph_query failed: {exc!r}"})


def code_memory_relink() -> str:
    """
    (Re)build all Finding -> CodeSymbol REFERENCES edges across the whole graph.

    Scans every Finding's text and links it to the CodeSymbols it distinctively
    names (precision-focused: distinctive types / explicit ``symbol:`` markers /
    ``File.ext`` mentions / unique long identifiers — never bare common words).
    Idempotent (MERGE). Run this after ingesting code with ``code_graph_ingest``
    so already-stored findings get connected; new findings auto-link on write.

    Returns:
        JSON ``{"findings_scanned": int, "links_created": int}`` (zeros if the
        graph store is unavailable).
    """
    ks = _get_kuzu()
    if not ks:
        return json.dumps({"error": "Kuzu graph store unavailable."})
    try:
        from graph import linker
        result = linker.relink_all(ks)
        # A relink can change the symbol set relationship; drop the cached index
        # so subsequent auto-links rebuild against the current graph.
        global _symbol_index_cache, _symbol_index_count
        _symbol_index_cache, _symbol_index_count = None, -1
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.debug("code_memory_relink failed: %r", exc)
        return json.dumps({"error": f"code_memory_relink failed: {exc!r}"})


def code_memory_map(anchor: str, anchor_type: str = "auto", hops: int = 1) -> str:
    """
    Map the code<->memory neighbourhood around an anchor node as a subgraph.

    Walks up to ``hops`` steps from the anchor over all edge types (CALLS,
    DEFINES, REFERENCES, IN_INVESTIGATION, MENTIONS, ...) and returns the nodes
    and edges around it — the concrete bridge between a code symbol and the
    findings/investigations that touch it (and vice-versa).

    Args:
        anchor: The anchor's key — a CodeSymbol name, Finding id, Investigation
            id, or CodeFile path, depending on ``anchor_type``.
        anchor_type: One of ``CodeSymbol`` / ``Finding`` / ``Investigation`` /
            ``CodeFile``, or ``auto`` (default) to try each in that order and use
            the first that exists in the graph.
        hops: BFS radius (default 1).

    Returns:
        JSON ``{"matched": {"label", "key"} | None, "nodes": [...],
        "edges": [...]}``. ``matched`` reports which anchor label resolved.
    """
    ks = _get_kuzu()
    if not ks:
        return json.dumps({"error": "Kuzu graph store unavailable."})
    try:
        from graph import queries
        candidates = (
            [anchor_type] if anchor_type and anchor_type != "auto"
            else ["CodeSymbol", "Finding", "Investigation", "CodeFile"]
        )
        for label in candidates:
            key = anchor
            if label == "CodeSymbol":
                # CodeSymbol's primary key is its id ("file::Qual"); callers usually
                # pass a bare NAME. If the anchor isn't a direct id, resolve name->id.
                if not ks.code_query("MATCH (s:CodeSymbol {id:$k}) RETURN s.id LIMIT 1", {"k": anchor}):
                    rows = ks.code_query("MATCH (s:CodeSymbol {name:$n}) RETURN s.id LIMIT 1", {"n": anchor})
                    if rows:
                        key = rows[0][0]
            sub = queries.subgraph(ks, label, key, hops=hops)
            # A resolved anchor yields at least the anchor node itself.
            if sub.get("nodes"):
                return json.dumps(
                    {"matched": {"label": label, "key": key}, **sub},
                    indent=2, default=str,
                )
        return json.dumps({"matched": None, "nodes": [], "edges": []}, indent=2)
    except Exception as exc:
        logger.debug("code_memory_map failed: %r", exc)
        return json.dumps({"error": f"code_memory_map failed: {exc!r}"})


def symbol_impact(symbol: str, hops: int = 3) -> str:
    """
    Blast radius of a code symbol across code and memory.

    Combines the transitive CALLS callers that reach ``symbol`` (up to ``hops``)
    with the Findings — and their Investigations — that REFERENCE the symbol or
    any of those callers. Answers "if this symbol is broken, which prior
    findings/investigations are implicated, and what calls into it?"

    Args:
        symbol: A CodeSymbol id (``file::Qualname``) or a bare symbol name.
        hops: Max transitive CALLS depth (default 3).

    Returns:
        JSON ``{"callers": [...symbols...], "findings": [...],
        "investigations": [...]}``.
    """
    ks = _get_kuzu()
    if not ks:
        return json.dumps({"error": "Kuzu graph store unavailable."})
    try:
        from graph import queries
        return json.dumps(queries.symbol_impact(ks, symbol, hops=hops), indent=2, default=str)
    except Exception as exc:
        logger.debug("symbol_impact failed: %r", exc)
        return json.dumps({"error": f"symbol_impact failed: {exc!r}"})


def impact_report(symbol: str, hops: int = 3) -> str:
    """
    Change blast radius for a code symbol or class: who calls it (transitively) and
    which findings / investigations reference it or its callers.

    Answers "if I change this, what code and what prior analysis is affected?".
    For a class name it also folds in the class's own methods. Composes the
    graph.analytics.impact_report primitive over the code<->memory graph.

    Args:
        symbol: A CodeSymbol name (method or class), e.g. "EskfFusion" or "updateGps".
        hops: Transitive CALLS depth for callers (default 3, max 6).

    Returns:
        JSON with resolved symbols, direct + transitive caller counts, the count of
        referencing findings, affected investigations (by finding count, with samples),
        and the symbols most co-referenced with it. Empty structure if unavailable.
    """
    ks = _get_kuzu()
    if not ks:
        return json.dumps({"error": "Kuzu graph store unavailable."})
    try:
        from graph.analytics import impact_report as _impact
        return json.dumps(_impact(ks, symbol, hops=hops), indent=2, default=str)
    except Exception as exc:
        logger.debug("impact_report failed: %r", exc)
        return json.dumps({"error": f"impact_report failed: {exc!r}"})


def finding_code_context(finding_id: str) -> str:
    """
    The code a finding references, each symbol wrapped with its callers/callees.

    Attach concrete code context when reviewing a finding — the CodeSymbols it
    mentions plus each one's immediate call neighbourhood — so you can judge the
    claim against the actual code. Composes graph.analytics.finding_code_context.

    Args:
        finding_id: The Finding id to contextualize.

    Returns:
        JSON {finding_id, text, symbols:[{id,name,kind,file,line,callers,callees}]}.
        Empty symbols if the finding references no code (or is unavailable).
    """
    ks = _get_kuzu()
    if not ks:
        return json.dumps({"error": "Kuzu graph store unavailable."})
    try:
        from graph.analytics import finding_code_context as _ctx
        return json.dumps(_ctx(ks, finding_id), indent=2, default=str)
    except Exception as exc:
        logger.debug("finding_code_context failed: %r", exc)
        return json.dumps({"error": f"finding_code_context failed: {exc!r}"})


def investigation_code_briefing(investigation_id: str, top: int = 3) -> str:
    """
    The code story of an investigation: what code it touches, the blast radius of its
    most-analysed symbols, and which other investigations share that code.

    A one-call briefing composed from investigation_footprint + impact_report (over the
    investigation's hotspot symbols) + related_investigations_via_code. Use it to onboard
    onto a case or see how it connects to the codebase and to other cases.

    Args:
        investigation_id: The investigation to brief.
        top: How many hotspot symbols to expand impact for (default 3).

    Returns:
        JSON {investigation, finding_count, symbols_touched, files_touched,
        top_symbols:[{symbol, in_investigation_findings, transitive_callers,
        total_referencing_findings, other_investigations}], related_investigations}.
    """
    ks = _get_kuzu()
    if not ks:
        return json.dumps({"error": "Kuzu graph store unavailable."})
    try:
        from graph.analytics import investigation_code_briefing as _brief
        return json.dumps(_brief(ks, investigation_id, top=top), indent=2, default=str)
    except Exception as exc:
        logger.debug("investigation_code_briefing failed: %r", exc)
        return json.dumps({"error": f"investigation_code_briefing failed: {exc!r}"})


def subsystem_report(anchor: str, limit: int = 15) -> str:
    """
    Full picture of a subsystem: the code under a file path or path/package prefix,
    its call boundary (who calls in / what it calls out), the symbols most analysed
    in memory, and which investigations touch it.

    Give a file (".../EskfFusion.java"), a directory, or a package prefix. Composes
    graph.analytics.subsystem_report over the code<->memory graph.

    Args:
        anchor: A CodeFile path, or a path/package prefix matching several files.
        limit: Max rows per section (default 15).

    Returns:
        JSON {anchor, files, symbol_count, kinds, inbound_callers, outbound_callees,
        hotspot_symbols, investigations}.
    """
    ks = _get_kuzu()
    if not ks:
        return json.dumps({"error": "Kuzu graph store unavailable."})
    try:
        from graph.analytics import subsystem_report as _sub
        return json.dumps(_sub(ks, anchor, limit=limit), indent=2, default=str)
    except Exception as exc:
        logger.debug("subsystem_report failed: %r", exc)
        return json.dumps({"error": f"subsystem_report failed: {exc!r}"})


def related_investigations_via_code(investigation_id: str, limit: int = 15) -> str:
    """
    Other investigations that reference the SAME code symbols as this one.

    Code-mediated case linkage: surfaces prior investigations that touched the same
    subsystem (shared CodeSymbols), ranked by overlap — even when they share no
    entities or text. Composes graph.analytics.related_investigations_via_code.

    Args:
        investigation_id: The investigation to find code-neighbours for.
        limit: Max related investigations (default 15).

    Returns:
        JSON list of {investigation, shared_symbols, sample_symbols}, ranked desc.
    """
    ks = _get_kuzu()
    if not ks:
        return json.dumps({"error": "Kuzu graph store unavailable."})
    try:
        from graph.analytics import related_investigations_via_code as _rel
        return json.dumps({"investigation_id": investigation_id,
                           "related": _rel(ks, investigation_id, limit=limit)}, indent=2, default=str)
    except Exception as exc:
        logger.debug("related_investigations_via_code failed: %r", exc)
        return json.dumps({"error": f"related_investigations_via_code failed: {exc!r}"})


def dead_code_candidates(lang: Optional[str] = None, limit: int = 50) -> str:
    """
    Functions/methods with no code caller and no finding reference — a reliable
    dead-code list with framework entry points (@mcp.tool, @app.route, @property…),
    private (_), test, and dunder symbols excluded.

    Requires an ingested code graph (code_graph_ingest). Call-graph recall varies by
    language (Python is sparser than Java/Kotlin), so treat results as candidates to
    verify. Composes graph.analytics.dead_code_candidates.

    Args:
        lang: Optional filter — "python" / "java" / "kotlin" / etc.
        limit: Max candidates (default 50).

    Returns:
        JSON {candidates:[{name,kind,file,line,decorators}], count, note}.
    """
    ks = _get_kuzu()
    if not ks:
        return json.dumps({"error": "Kuzu graph store unavailable."})
    try:
        from graph.analytics import dead_code_candidates as _dead
        return json.dumps(_dead(ks, lang=lang, limit=limit), indent=2, default=str)
    except Exception as exc:
        logger.debug("dead_code_candidates failed: %r", exc)
        return json.dumps({"error": f"dead_code_candidates failed: {exc!r}"})


def register(mcp, get_kuzu):
    """Inject deps and register every graph tool on the shared FastMCP instance."""
    global _get_kuzu
    _get_kuzu = get_kuzu
    for fn in (
        code_graph_ingest,
        code_graph_query,
        code_memory_relink,
        code_memory_map,
        symbol_impact,
        impact_report,
        finding_code_context,
        investigation_code_briefing,
        subsystem_report,
        related_investigations_via_code,
        dead_code_candidates,
    ):
        mcp.tool()(fn)
