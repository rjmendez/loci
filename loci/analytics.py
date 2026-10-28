"""
DuckDB-powered analytics over Loci's existing JSONL investigation files.

Zero migration cost: queries existing JSONL files directly using
DuckDB's read_ndjson_auto() function.

Install optional dep: pip install duckdb  (or pip install loci[analytics])
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import duckdb
    _DUCKDB_AVAILABLE = True
except ImportError:
    _DUCKDB_AVAILABLE = False


def _get_memory_dir() -> Path:
    """Get Loci memory directory from environment or default."""
    env = os.getenv("LOCI_MEMORY_DIR") or os.getenv("MEMORY_DIR")
    if env:
        return Path(env)
    return Path.home() / ".loci" / "memory-sessions"


def _findings_glob(memory_dir: Path) -> str:
    """Return glob pattern for all findings.jsonl files under memory_dir."""
    return str(memory_dir / "*" / "findings.jsonl")


def is_available() -> bool:
    return _DUCKDB_AVAILABLE


def _source(glob: str) -> str:
    return f"read_ndjson_auto('{glob}', union_by_name=true, filename=true)"


def _is_no_files(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "no files found" in msg or "no such file" in msg or "could not open" in msg


def _columns(con: Any, glob: str) -> set[str]:
    """Columns present in at least one findings.jsonl (union_by_name).

    Rows differ in schema: real findings carry ``created_at_ts`` (epoch seconds) and no
    ``created_at``, and a memory dir may hold files without ``resolution`` or ``source``.
    Naming an absent column is a Binder Error, so the SQL is built from what exists.
    """
    return {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {_source(glob)}").fetchall()}


def _created_expr(cols: set[str]) -> str | None:
    """SQL for a finding's creation time as a naive UTC TIMESTAMP, from whichever field exists."""
    parts = []
    if "created_at" in cols:
        parts.append("TRY_CAST(created_at AS TIMESTAMP)")
    if "created_at_ts" in cols:
        parts.append("epoch_ms(TRY_CAST(created_at_ts AS BIGINT) * 1000)")
    if "ts" in cols:
        parts.append("TRY_CAST(ts AS TIMESTAMP)")
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else f"COALESCE({', '.join(parts)})"


def _not_access_row() -> str:
    # findings.jsonl also holds record_type='access' rows (a read log, not findings)
    return "COALESCE(record_type, '') <> 'access'"


def cross_investigation_search(
    query_text: str | None = None,
    min_confidence: float | None = None,
    finding_type: str | None = None,
    days: int | None = None,
    limit: int = 50,
    memory_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Search findings across ALL investigations using DuckDB.

    Much faster than Python JSONL iteration for cross-investigation queries:
    DuckDB applies columnar pushdown instead of full-file Python scans.

    Falls back gracefully if duckdb not installed (raises ImportError so
    callers can surface a clear install message).

    Returns list of finding dicts with ``investigation_id`` injected.
    """
    if not _DUCKDB_AVAILABLE:
        raise ImportError("duckdb not installed: pip install duckdb")

    mdir = memory_dir or _get_memory_dir()
    glob = _findings_glob(mdir)

    con = duckdb.connect(":memory:")
    try:
        have = _columns(con, glob)
        created = _created_expr(have)

        def col(name: str) -> str:
            # VARCHAR: DuckDB auto-detects UUID-looking ids (and similar) as non-JSON types.
            return f'CAST("{name}" AS VARCHAR) AS "{name}"' if name in have else f'CAST(NULL AS VARCHAR) AS "{name}"'

        conditions: list[str] = []
        params: list[Any] = []
        if "record_type" in have:
            conditions.append(_not_access_row())

        # A filter on a column no file has matches nothing, rather than raising.
        if min_confidence is not None:
            if "numeric_confidence" in have:
                conditions.append("TRY_CAST(numeric_confidence AS DOUBLE) >= ?")
                params.append(min_confidence)
            else:
                conditions.append("FALSE")

        if finding_type is not None:
            if "type" in have:
                conditions.append('"type" = ?')
                params.append(finding_type)
            else:
                conditions.append("FALSE")

        if days is not None:
            if created:
                # days is validated as int before interpolation — no injection risk.
                conditions.append(
                    f"{created} >= (CURRENT_TIMESTAMP AT TIME ZONE 'UTC') - INTERVAL {int(days)} DAY"
                )
            else:
                conditions.append("FALSE")

        if query_text is not None:
            if "text" in have:
                conditions.append("text ILIKE ?")
                params.append(f"%{query_text}%")
            else:
                conditions.append("FALSE")

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)

        numeric = ("TRY_CAST(numeric_confidence AS DOUBLE)" if "numeric_confidence" in have
                   else "CAST(NULL AS DOUBLE)")
        created_out = f"strftime({created}, '%Y-%m-%dT%H:%M:%SZ')" if created else "CAST(NULL AS VARCHAR)"
        sql = f"""
            SELECT
                filename,
                {col("id")},
                {col("type")},
                {col("text")},
                {col("confidence")},
                {numeric} AS numeric_confidence,
                {created_out} AS created_at,
                {col("resolution")},
                {col("source")}
            FROM {_source(glob)}
            {where_clause}
            ORDER BY {numeric} DESC NULLS LAST{f", {created} DESC NULLS LAST" if created else ""}
            LIMIT ?
        """
        rows = con.execute(sql, params).fetchall()
        cols = [d[0] for d in con.description]
    except Exception as exc:  # noqa: BLE001
        # Glob matched no files (empty memory_dir) or DuckDB I/O error.
        if _is_no_files(exc):
            return []
        raise
    finally:
        con.close()

    results = []
    for row in rows:
        d = dict(zip(cols, row))
        fname = d.pop("filename", "")
        try:
            d["investigation_id"] = Path(fname).parent.name
        except Exception:  # noqa: BLE001
            d["investigation_id"] = None
        results.append(d)

    return results


def cross_investigation_stats(
    memory_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Aggregate statistics across all investigations.

    Returns total findings count, distinct investigation count,
    average confidence, and counts by resolution state.
    """
    if not _DUCKDB_AVAILABLE:
        raise ImportError("duckdb not installed: pip install duckdb")

    mdir = memory_dir or _get_memory_dir()
    glob = _findings_glob(mdir)

    con = duckdb.connect(":memory:")
    try:
        have = _columns(con, glob)
        avg = ("AVG(TRY_CAST(numeric_confidence AS DOUBLE))" if "numeric_confidence" in have
               else "CAST(NULL AS DOUBLE)")
        open_ = "COUNT(*) FILTER (WHERE resolution = 'open')" if "resolution" in have else "CAST(0 AS BIGINT)"
        resolved = ("COUNT(*) FILTER (WHERE resolution = 'resolved')" if "resolution" in have
                    else "CAST(0 AS BIGINT)")
        where = f"WHERE {_not_access_row()}" if "record_type" in have else ""
        # COUNT(DISTINCT filename) counts distinct investigation directories
        # because each investigation contributes exactly one findings.jsonl.
        sql = f"""
            SELECT
                COUNT(*) AS total_findings,
                COUNT(DISTINCT filename) AS investigation_count,
                {avg} AS avg_confidence,
                {open_} AS open_count,
                {resolved} AS resolved_count
            FROM {_source(glob)}
            {where}
        """
        row = con.execute(sql).fetchone()
        cols = [d[0] for d in con.description]
        return dict(zip(cols, row))
    except Exception as exc:  # noqa: BLE001
        if _is_no_files(exc):
            return {"total_findings": 0, "investigation_count": 0}
        return {"error": str(exc), "available": False}
    finally:
        con.close()
