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

    conditions: list[str] = []
    params: list[Any] = []

    if min_confidence is not None:
        conditions.append("TRY_CAST(numeric_confidence AS DOUBLE) >= ?")
        params.append(min_confidence)

    if finding_type is not None:
        conditions.append("type = ?")
        params.append(finding_type)

    if days is not None:
        # days is validated as int before interpolation — no injection risk.
        conditions.append(
            f"TRY_CAST(created_at AS TIMESTAMP) >= CURRENT_TIMESTAMP - INTERVAL {int(days)} DAY"
        )

    if query_text is not None:
        conditions.append("text ILIKE ?")
        params.append(f"%{query_text}%")

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    params.append(limit)

    sql = f"""
        SELECT
            filename,
            id,
            type,
            text,
            confidence,
            TRY_CAST(numeric_confidence AS DOUBLE) AS numeric_confidence,
            created_at,
            resolution,
            source
        FROM read_ndjson_auto('{glob}', union_by_name=true, filename=true)
        {where_clause}
        ORDER BY TRY_CAST(numeric_confidence AS DOUBLE) DESC NULLS LAST
        LIMIT ?
    """

    con = duckdb.connect(":memory:")
    try:
        rows = con.execute(sql, params).fetchall()
        cols = [d[0] for d in con.description]
    except Exception as exc:  # noqa: BLE001
        # Glob matched no files (empty memory_dir) or DuckDB I/O error.
        _no_files = (
            "no files found" in str(exc).lower()
            or "no such file" in str(exc).lower()
            or "could not open" in str(exc).lower()
        )
        if _no_files:
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

    # COUNT(DISTINCT filename) counts distinct investigation directories
    # because each investigation contributes exactly one findings.jsonl.
    sql = f"""
        SELECT
            COUNT(*) AS total_findings,
            COUNT(DISTINCT filename) AS investigation_count,
            AVG(TRY_CAST(numeric_confidence AS DOUBLE)) AS avg_confidence,
            COUNT(*) FILTER (WHERE resolution = 'open') AS open_count,
            COUNT(*) FILTER (WHERE resolution = 'resolved') AS resolved_count
        FROM read_ndjson_auto('{glob}', union_by_name=true, filename=true)
    """

    con = duckdb.connect(":memory:")
    try:
        row = con.execute(sql).fetchone()
        cols = [d[0] for d in con.description]
        return dict(zip(cols, row))
    except Exception as exc:  # noqa: BLE001
        _no_files = (
            "no files found" in str(exc).lower()
            or "no such file" in str(exc).lower()
            or "could not open" in str(exc).lower()
        )
        if _no_files:
            return {"total_findings": 0, "investigation_count": 0}
        return {"error": str(exc), "available": False}
    finally:
        con.close()
