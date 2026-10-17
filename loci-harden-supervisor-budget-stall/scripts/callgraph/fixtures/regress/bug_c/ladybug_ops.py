# REGRESSION FIXTURE -- DO NOT IMPORT, DO NOT RUN, DO NOT COPY BACK INTO THE TREE.
#
# BUG C (real slot): the module that ACTUALLY owns _symbol_index_cache / _symbol_index_count at module level. Present so the audit has a real cross-module slot to name in its diagnosis.
#
# Source : mcp/ladybug_ops.py @ 30a2dc22656898d58fbdd4da4e3e7a3bdac39953  (whole file)
#          `git show 30a2dc226568:mcp/ladybug_ops.py`
# Status : THIS BUG IS FIXED IN THE REAL TREE. This file preserves the
#          pre-fix code purely so the callgraph tool can be re-pointed at it
#          and proven to still surface the defect. It is parsed by ast only
#          and is excluded from the analyzed corpus (config.SELF_PACKAGE_REL).
# ---- VERBATIM BODY BELOW (do not edit; see test_regression_real_bugs.py) ----
"""Leaf LadybugDB helpers — split out of server.py (P2c of the Loci self-review split).

Only the *stateless leaves* live here: the finding/investigation mirror, the
entity-lookup projection, the symbol-index cache, and the one-time backfill. The
LadybugStore singleton itself — ``_ladybug_store`` / ``_ladybug_failed`` /
``_ladybug_last_attempt`` / ``_LADYBUG_RETRY_SECONDS`` / ``_ladybug_lock`` and the
``_get_ladybug`` accessor built on them, plus ``_ladybug_health_state`` /
``_ladybug_writer_pid`` which feed the frozen ``kuzu`` / ``kuzu_writer_pid`` wire
keys — DELIBERATELY stay in server.py: several tests monkeypatch those latches on
the server module and assert on ``server._get_ladybug()``, so moving them here
would make every one of those patches a silent no-op.

Dependencies follow the same injection contract as inv_store / graph_tools: the
store accessor and the memory root are pushed in through ``register()`` rather than
imported, so this module never imports server and no cycle exists. The memory root
arrives as a LAMBDA (not a Path) because ``_ladybug_backfill_if_empty`` iterates it
and ~10 tests rebind ``server.MEMORY_DIR`` to a tmpdir.
"""
from __future__ import annotations

import logging

from inv_store import _distinctive_entity_set, _read_jsonl

logger = logging.getLogger("loci-mcp")

# --- injected by register() -------------------------------------------------
_get_ladybug = None
_get_memory_dir = None


def _root():
    """The investigation memory root, resolved through the injected accessor."""
    return _get_memory_dir()


def _ladybug_upsert_investigation(investigation_id: str, title: str = "") -> None:
    ks = _get_ladybug()
    if not ks:
        return
    try:
        ks.upsert_investigation(str(investigation_id), str(title or ""))
    except Exception as exc:
        logger.debug("LadybugDB investigation upsert failed (fail-open): %r", exc)


def _coerce_ts(v) -> int:
    """Coerce a finding timestamp (int, epoch-string, or ISO-8601) to an int epoch. 0 on failure."""
    if isinstance(v, bool):
        return 0
    if isinstance(v, (int, float)):
        return int(v)
    if isinstance(v, str) and v.strip():
        s = v.strip()
        if s.isdigit():
            return int(s)
        try:
            from datetime import datetime
            return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
        except Exception:
            return 0
    return 0


def _mirror_finding_to_ladybug(finding: dict, investigation_id: str, ks=None) -> None:
    """Mirror one finding (node + MENTIONS + DERIVED_FROM) into the graph. Fail-open."""
    ks = ks or _get_ladybug()
    if not ks or not isinstance(finding, dict):
        return
    fid = finding.get("id")
    if not fid:
        return
    try:
        ks.upsert_finding({
            "id": fid,
            "investigation": investigation_id or finding.get("investigation_id") or "",
            "type": (finding.get("finding_type") or finding.get("type")
                     or finding.get("ftype") or ""),
            "text": finding.get("text", "") or "",
            "confidence": finding.get("confidence", "") or "",
            "source": finding.get("source", "") or "",
            "ts": _coerce_ts(finding.get("ts") or finding.get("created_at") or finding.get("timestamp")),
        })
        ents = finding.get("entities")
        if isinstance(ents, dict):
            distinctive = _distinctive_entity_set(ents)
            triples = []
            for etype, vals in ents.items():
                for v in vals or []:
                    name = str(v).strip()
                    if name:
                        triples.append((name, str(etype), name.lower() in distinctive))
            if triples:
                ks.link_mentions(fid, triples)
        df = finding.get("derived_from")
        if df:
            ks.link_derived_from(fid, list(df) if isinstance(df, (list, tuple, set)) else [df])
    except Exception as exc:
        logger.debug("LadybugDB finding mirror failed (fail-open): %r", exc)


# --- Finding -> CodeSymbol auto-linker (REFERENCES) --------------------------
# A tiny cache so the per-write auto-link doesn't rebuild the symbol index on
# every finding. Invalidated when the CodeSymbol count changes (e.g. after a new
# code_graph_ingest / code_memory_relink). Fail-open throughout.
_symbol_index_cache = None             # built graph.linker index
_symbol_index_count = -1               # CodeSymbol count the cache was built at


def _get_symbol_index(ks):
    """Return a cached graph.linker symbol index, rebuilding it when the graph's
    CodeSymbol count changes. Returns None (fail-open) if unavailable/empty."""
    global _symbol_index_cache, _symbol_index_count
    if not ks:
        return None
    try:
        rows = ks.code_query("MATCH (s:CodeSymbol) RETURN count(s)")
        count = int(rows[0][0]) if rows and rows[0] else 0
        if count == 0:
            _symbol_index_cache, _symbol_index_count = None, 0
            return None
        if _symbol_index_cache is None or count != _symbol_index_count:
            from graph import linker
            srows = ks._rows("MATCH (s:CodeSymbol) RETURN s.id, s.name, s.kind, s.file")
            symbols = [{"id": r[0], "name": r[1], "kind": r[2], "file": r[3]} for r in srows]
            _symbol_index_cache = linker.build_symbol_index(symbols)
            _symbol_index_count = count
        return _symbol_index_cache
    except Exception as exc:
        logger.debug("symbol index build failed (fail-open): %r", exc)
        return None


def _autolink_finding_to_ladybug(finding: dict, ks=None) -> None:
    """Auto-create REFERENCES edges from one just-mirrored finding to CodeSymbols.
    Cheap single-finding link over a cached index. Fail-open — never raises."""
    ks = ks or _get_ladybug()
    if not ks or not isinstance(finding, dict):
        return
    fid = finding.get("id")
    text = finding.get("text")
    if not fid or not text:
        return
    try:
        index = _get_symbol_index(ks)
        if not index:
            return  # no code graph ingested yet — nothing to link against
        from graph import linker
        linker.link_findings(ks, [{"id": fid, "text": text}], index)
    except Exception as exc:
        logger.debug("LadybugDB finding auto-link failed (fail-open): %r", exc)


def _ladybug_backfill_if_empty(ks) -> None:
    """Backfill existing on-disk findings into a freshly-created graph (once)."""
    try:
        rows = ks.code_query("MATCH (f:Finding) RETURN count(f)")
        existing = int(rows[0][0]) if rows and rows[0] else 0
    except Exception:
        existing = 0
    if existing > 0:
        return
    finding_rows: list[dict] = []
    mention_rows: list[dict] = []
    derived_rows: list[dict] = []
    invs: list[str] = []
    try:
        for inv_dir in sorted(_root().iterdir()):
            if not inv_dir.is_dir() or inv_dir.name.startswith("_"):
                continue
            fjsonl = inv_dir / "findings.jsonl"
            if not fjsonl.exists():
                continue
            invs.append(inv_dir.name)
            for f in _read_jsonl(fjsonl):
                fid = f.get("id")
                if not fid:
                    continue
                inv = str(f.get("investigation_id") or inv_dir.name)
                finding_rows.append({
                    "id": fid, "investigation": inv,
                    "type": f.get("finding_type") or f.get("type") or f.get("ftype") or "",
                    "text": f.get("text", "") or "", "confidence": f.get("confidence", "") or "",
                    "source": f.get("source", "") or "",
                    "ts": _coerce_ts(f.get("ts") or f.get("created_at") or f.get("timestamp")),
                })
                ents = f.get("entities")
                if isinstance(ents, dict):
                    distinctive = _distinctive_entity_set(ents)
                    for etype, vals in ents.items():
                        for v in vals or []:
                            name = str(v).strip()
                            if name:
                                mention_rows.append({"f": fid, "name": name, "etype": str(etype),
                                                     "distinctive": name.lower() in distinctive})
                df = f.get("derived_from")
                if df:
                    for p in (df if isinstance(df, (list, tuple, set)) else [df]):
                        if p:
                            derived_rows.append({"f": fid, "p": str(p)})
    except Exception as exc:
        logger.debug("LadybugDB backfill scan failed (fail-open): %r", exc)
    if not finding_rows:
        return
    try:
        for iv in invs:
            ks.upsert_investigation(iv, "")
        n = ks.upsert_findings_batch(finding_rows)
        ks.link_mentions_batch(mention_rows)
        ks.link_derived_from_batch(derived_rows)
        logger.info("LadybugDB backfill: mirrored %d findings, %d mentions, %d derivations (batched).",
                    n, len(mention_rows), len(derived_rows))
    except Exception as exc:
        logger.debug("LadybugDB backfill batch failed (fail-open): %r", exc)


def _entity_lookup_ladybug(entity: str, investigation_id, limit: int) -> list[dict]:
    """Graph-primary entity lookup. Normalizes to the finding shape the tools use."""
    ks = _get_ladybug()
    if not ks:
        return []
    try:
        rows = ks.entity_findings(entity, limit=max(limit * 2, limit))
    except Exception as exc:
        logger.debug("Kuzu entity_findings failed (fail-open): %r", exc)
        return []
    out: list[dict] = []
    for r in rows or []:
        inv = r.get("investigation") or ""
        if investigation_id and inv != investigation_id:
            continue
        out.append({
            "id": r.get("id"),
            "investigation_id": inv,
            "finding_type": r.get("ftype", "") or "",
            "text": r.get("text", "") or "",
            "confidence": r.get("confidence", "") or "",
            "source": r.get("source", "") or "",
        })
        if len(out) >= limit:
            break
    return out


def register(get_ladybug, get_memory_dir):
    """Inject the store accessor and the memory-root accessor.

    Both are callables, never values: ``get_ladybug`` is server's ``_get_ladybug``
    (so these helpers reach the singleton through the same patched server globals
    the resilience tests drive), and ``get_memory_dir`` is a lambda over
    ``server.MEMORY_DIR`` so tmpdir rebinds keep steering the backfill scan.
    """
    global _get_ladybug, _get_memory_dir
    _get_ladybug = get_ladybug
    _get_memory_dir = get_memory_dir
