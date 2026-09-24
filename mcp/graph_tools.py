"""Code↔memory graph MCP wrappers, split from server.py.

These thin wrappers sit on ``graph.*`` and only need the injected
``_get_ladybug`` accessor from register().
"""
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("loci-mcp")

_get_ladybug = None  # injected by register()


def code_graph_ingest(path: str, max_files: Optional[int] = None, replace: bool = False) -> str:
    """
    Parse a source file or directory with tree-sitter and ingest its symbol
    graph into LadybugDB. Enables ``code_graph_query`` and code↔memory linkage.

    Supports python, java, kotlin, rust, javascript, typescript/tsx, and go;
    c/c++ currently produce file-level nodes only. Binary or oversized files and
    common vendor dirs (``.git``, ``node_modules``, ``.venv``, ``build``,
    ``dist``, ``target``) are skipped.

    Ingest is additive by default. With ``replace=True``, existing
    ``CodeFile``/``CodeSymbol`` nodes under ``path`` plus their
    ``DEFINES``/``CALLS``/``IMPORTS`` edges and inbound ``REFERENCES`` are
    deleted before parsing, keeping re-ingest idempotent and removing stale
    paths. Only code nodes are pruned; findings, investigations, and entities
    are untouched.

    Args:
        path: A source file or a directory root to walk.
        max_files: Optional cap on files parsed (directory mode).
        replace: If True, prune existing code nodes under ``path`` before ingest
            (idempotent re-ingest). Default False preserves additive behaviour.

    Returns:
        JSON with per-run counts ``{files, symbols, defines, calls, imports}``
        plus ``pruned`` when ``replace`` is set, or an error.
    """
    ks = _get_ladybug()
    if not ks:
        return json.dumps({"error": "LadybugDB graph store unavailable — cannot ingest code graph."})
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
    Run a read-only Cypher query against the LadybugDB graph and return rows.

    Write-shaped queries (``CREATE/DELETE/SET/MERGE/DROP/COPY/ALTER``) are
    rejected: this tool never mutates the graph. Use it for traversal or impact
    analysis, e.g. callers of a symbol, symbols defined by a file, or findings
    that reference a ``CodeSymbol``.

    Node tables: CodeFile(path,lang), CodeSymbol(id,name,kind,file,line,lang),
      Finding(id,investigation,ftype,text,confidence,source,ts), Entity(name,etype,distinctive),
      Investigation(id,title).
    Rel tables: DEFINES(CodeFile->CodeSymbol), CALLS(CodeSymbol->CodeSymbol),
      IMPORTS(CodeFile->CodeFile), REFERENCES(Finding->CodeSymbol), MENTIONS(Finding->Entity),
      DERIVED_FROM(Finding->Finding), IN_INVESTIGATION(Finding->Investigation),
      RELATED(Investigation->Investigation).

    Example: ``MATCH (c:CodeSymbol)-[:CALLS]->(t:CodeSymbol {name:'helper'}) RETURN c.id, c.file``

    Args:
        cypher: A read-only Cypher query.
        params: Optional parameter dict for ``$name`` placeholders.

    Returns:
        JSON ``{rows: [...], row_count}`` or an error, including write-guard
        rejection.
    """
    ks = _get_ladybug()
    if not ks:
        return json.dumps({"error": "LadybugDB graph store unavailable."})
    try:
        # code_query fails open to [] (engine errors, lease timeouts, a closed store),
        # which is indistinguishable from "no matches"; only its failure counter can
        # tell them apart, and an impact query read as "no callers" is a false negative.
        if not getattr(ks, "ok", True):
            return json.dumps({"error": "code_graph_query failed: graph store is not open"})
        before = getattr(ks, "code_query_failures", 0)
        rows = ks.code_query(cypher, params or None)
        if getattr(ks, "code_query_failures", 0) != before:
            detail = getattr(ks, "code_query_last_error", "") or "see server log"
            return json.dumps({"error": f"code_graph_query failed: the query did not run: {detail}"})
        return json.dumps({"row_count": len(rows), "rows": rows}, indent=2, default=str)
    except ValueError as exc:  # write-guard rejection
        return json.dumps({"error": f"rejected (read-only tool): {exc}"})
    except Exception as exc:
        logger.debug("code_graph_query failed: %r", exc)
        return json.dumps({"error": f"code_graph_query failed: {exc!r}"})


def code_memory_relink() -> str:
    """
    Rebuild all ``Finding -> CodeSymbol`` ``REFERENCES`` edges.

    Scans every finding's text and links distinctive symbol mentions: typed
    entities, explicit ``symbol:`` markers, ``File.ext`` mentions, and unique
    long identifiers, but never bare common words. Idempotent. Run it after
    ``code_graph_ingest`` so old findings gain links; new findings auto-link on
    write.

    Returns:
        JSON ``{"findings_scanned": int, "links_created": int}``. Returns an
        error if the graph store is unavailable.
    """
    ks = _get_ladybug()
    if not ks:
        return json.dumps({"error": "LadybugDB graph store unavailable."})
    try:
        from graph import linker
        result = linker.relink_all(ks)
        # A relink can change the symbol set relationship; drop the cached index
        # so subsequent auto-links rebuild against the current graph. Must go
        # through ladybug_ops -- assigning the names here would bind graph_tools'
        # own globals and leave the real cache stale.
        import ladybug_ops
        ladybug_ops.invalidate_symbol_index()
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.debug("code_memory_relink failed: %r", exc)
        return json.dumps({"error": f"code_memory_relink failed: {exc!r}"})


def code_memory_map(anchor: str, anchor_type: str = "auto", hops: int = 1) -> str:
    """
    Return the code↔memory neighborhood around an anchor as a subgraph.

    Walks up to ``hops`` steps across all edge types (``CALLS``, ``DEFINES``,
    ``REFERENCES``, ``IN_INVESTIGATION``, ``MENTIONS``, ...), showing which code
    symbols, findings, and investigations touch the anchor.

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
    ks = _get_ladybug()
    if not ks:
        return json.dumps({"error": "LadybugDB graph store unavailable."})
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
    Return the blast radius of a code symbol across code and memory.

    Combines transitive callers of ``symbol`` (up to ``hops``) with findings and
    investigations that reference the symbol or those callers.

    Args:
        symbol: A CodeSymbol id (``file::Qualname``) or a bare symbol name.
        hops: Max transitive CALLS depth (default 3).

    Returns:
        JSON ``{"callers": [...symbols...], "findings": [...],
        "investigations": [...]}``.
    """
    ks = _get_ladybug()
    if not ks:
        return json.dumps({"error": "LadybugDB graph store unavailable."})
    try:
        from graph import queries
        return json.dumps(queries.symbol_impact(ks, symbol, hops=hops), indent=2, default=str)
    except Exception as exc:
        logger.debug("symbol_impact failed: %r", exc)
        return json.dumps({"error": f"symbol_impact failed: {exc!r}"})


def impact_report(symbol: str, hops: int = 3) -> str:
    """
    Report the change blast radius of a code symbol or class.

    Shows transitive callers plus findings and investigations that reference the
    symbol or its callers. For a class name, folds in the class's own methods.

    Args:
        symbol: A CodeSymbol name (method or class), e.g. "EskfFusion" or "updateGps".
        hops: Transitive CALLS depth for callers (default 3, max 6).

    Returns:
        JSON with resolved symbols, direct and transitive caller counts,
        referencing-finding count, affected investigations with samples, and the
        most co-referenced symbols. Empty structure if unavailable.
    """
    ks = _get_ladybug()
    if not ks:
        return json.dumps({"error": "LadybugDB graph store unavailable."})
    try:
        from graph.analytics import impact_report as _impact
        return json.dumps(_impact(ks, symbol, hops=hops), indent=2, default=str)
    except Exception as exc:
        logger.debug("impact_report failed: %r", exc)
        return json.dumps({"error": f"impact_report failed: {exc!r}"})


def finding_code_context(finding_id: str) -> str:
    """
    Return the code a finding references, with each symbol's immediate
    caller/callee neighborhood. Use it to review a claim against actual code.

    Args:
        finding_id: The Finding id to contextualize.

    Returns:
        JSON ``{finding_id, text, symbols:[{id,name,kind,file,line,callers,callees}]}``.
        ``symbols`` is empty when the finding references no code or is unavailable.
    """
    ks = _get_ladybug()
    if not ks:
        return json.dumps({"error": "LadybugDB graph store unavailable."})
    try:
        from graph.analytics import finding_code_context as _ctx
        return json.dumps(_ctx(ks, finding_id), indent=2, default=str)
    except Exception as exc:
        logger.debug("finding_code_context failed: %r", exc)
        return json.dumps({"error": f"finding_code_context failed: {exc!r}"})


def investigation_code_briefing(investigation_id: str, top: int = 3) -> str:
    """
    Summarize an investigation's code footprint, its hottest-symbol blast
    radius, and other investigations that touch the same code.

    Composes ``investigation_footprint``, ``impact_report`` on hotspot symbols,
    and ``related_investigations_via_code``. Useful for onboarding or tracing
    how a case connects to the codebase.

    Args:
        investigation_id: The investigation to brief.
        top: How many hotspot symbols to expand impact for (default 3).

    Returns:
        JSON {investigation, finding_count, symbols_touched, files_touched,
        top_symbols:[{symbol, in_investigation_findings, transitive_callers,
        total_referencing_findings, other_investigations}], related_investigations}.
    """
    ks = _get_ladybug()
    if not ks:
        return json.dumps({"error": "LadybugDB graph store unavailable."})
    try:
        from graph.analytics import investigation_code_briefing as _brief
        return json.dumps(_brief(ks, investigation_id, top=top), indent=2, default=str)
    except Exception as exc:
        logger.debug("investigation_code_briefing failed: %r", exc)
        return json.dumps({"error": f"investigation_code_briefing failed: {exc!r}"})


def subsystem_report(anchor: str, limit: int = 15) -> str:
    """
    Report a subsystem's code boundary, memory hotspots, and linked
    investigations.

    ``anchor`` may be a file, directory, or path/package prefix. Composes
    ``graph.analytics.subsystem_report`` over the code↔memory graph.

    Args:
        anchor: A CodeFile path, or a path/package prefix matching several files.
        limit: Max rows per section (default 15).

    Returns:
        JSON {anchor, files, symbol_count, kinds, inbound_callers, outbound_callees,
        hotspot_symbols, investigations}.
    """
    ks = _get_ladybug()
    if not ks:
        return json.dumps({"error": "LadybugDB graph store unavailable."})
    try:
        from graph.analytics import subsystem_report as _sub
        return json.dumps(_sub(ks, anchor, limit=limit), indent=2, default=str)
    except Exception as exc:
        logger.debug("subsystem_report failed: %r", exc)
        return json.dumps({"error": f"subsystem_report failed: {exc!r}"})


def related_investigations_via_code(investigation_id: str, limit: int = 15) -> str:
    """
    List investigations that reference the same code symbols as this one.

    This is code-mediated case linkage: it finds overlapping subsystems even
    when investigations share no entities or text.

    Args:
        investigation_id: The investigation to find code-neighbours for.
        limit: Max related investigations (default 15).

    Returns:
        JSON list of ``{investigation, shared_symbols, sample_symbols}``, ranked
        by overlap descending.
    """
    ks = _get_ladybug()
    if not ks:
        return json.dumps({"error": "LadybugDB graph store unavailable."})
    try:
        from graph.analytics import related_investigations_via_code as _rel
        return json.dumps({"investigation_id": investigation_id,
                           "related": _rel(ks, investigation_id, limit=limit)}, indent=2, default=str)
    except Exception as exc:
        logger.debug("related_investigations_via_code failed: %r", exc)
        return json.dumps({"error": f"related_investigations_via_code failed: {exc!r}"})


def dead_code_candidates(lang: Optional[str] = None, limit: int = 50) -> str:
    """
    List dead-code candidates: functions or methods with no code caller and no
    finding reference.

    Framework entry points (``@mcp.tool``, ``@app.route``, ``@property``...),
    private names, tests, and dunder symbols are excluded. Requires an ingested
    code graph. Call-graph recall varies by language, so verify results.

    Args:
        lang: Optional filter — "python" / "java" / "kotlin" / etc.
        limit: Max candidates (default 50).

    Returns:
        JSON {candidates:[{name,kind,file,line,decorators}], count, note}.
    """
    ks = _get_ladybug()
    if not ks:
        return json.dumps({"error": "LadybugDB graph store unavailable."})
    try:
        from graph.analytics import dead_code_candidates as _dead
        return json.dumps(_dead(ks, lang=lang, limit=limit), indent=2, default=str)
    except Exception as exc:
        logger.debug("dead_code_candidates failed: %r", exc)
        return json.dumps({"error": f"dead_code_candidates failed: {exc!r}"})


def register(mcp, get_ladybug):
    """Inject deps and register every graph tool on the shared FastMCP instance."""
    global _get_ladybug
    _get_ladybug = get_ladybug
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
