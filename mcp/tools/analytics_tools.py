"""MCP tools for DuckDB-powered cross-investigation analytics.

Registers:
  mcp__loci__investigation_search_fast  — cross-investigation JSONL search via DuckDB
  mcp__loci__investigation_stats_fast   — aggregate statistics across all investigations
"""
from __future__ import annotations

import asyncio
import json
from typing import Optional


def register(mcp_server):
    """Register DuckDB analytics tools on the given FastMCP server."""

    @mcp_server.tool()
    async def investigation_search_fast(
        query: str = "",
        min_confidence: Optional[float] = None,
        finding_type: Optional[str] = None,
        days: Optional[int] = None,
        limit: int = 50,
    ) -> str:
        """
        Fast cross-investigation finding search powered by DuckDB.

        Queries all findings.jsonl files directly — no schema migration needed.
        10-100× faster than the Python JSONL iteration path for cross-investigation
        queries because DuckDB applies columnar pushdown instead of full-file scans.

        Requires: pip install duckdb  (or pip install loci[analytics])
        """
        try:
            from loci.analytics import cross_investigation_search
        except ImportError:
            return json.dumps({"error": "duckdb_not_installed", "install": "pip install duckdb"})

        try:
            results = await asyncio.to_thread(
                cross_investigation_search,
                query_text=query or None,
                min_confidence=min_confidence,
                finding_type=finding_type,
                days=days,
                limit=limit,
            )
        except ImportError:
            return json.dumps({"error": "duckdb_not_installed", "install": "pip install duckdb"})

        # Serialise any non-JSON-native values (e.g. Decimal, datetime).
        safe = []
        for r in results:
            safe.append({k: (float(v) if hasattr(v, "__float__") and not isinstance(v, (int, float)) else v)
                         for k, v in r.items()})

        return json.dumps({"results": safe, "count": len(safe), "backend": "duckdb"})

    @mcp_server.tool()
    async def investigation_stats_fast() -> str:
        """
        Aggregate statistics across all investigations, powered by DuckDB.

        Returns total finding count, number of investigations, average confidence,
        and counts by resolution state (open / resolved).

        Requires: pip install duckdb  (or pip install loci[analytics])
        """
        try:
            from loci.analytics import cross_investigation_stats
        except ImportError:
            return json.dumps({"error": "duckdb_not_installed", "install": "pip install duckdb"})

        try:
            stats = await asyncio.to_thread(cross_investigation_stats)
        except ImportError:
            return json.dumps({"error": "duckdb_not_installed", "install": "pip install duckdb"})

        safe = {}
        for k, v in stats.items():
            if v is None or isinstance(v, (int, float, str, bool)):
                safe[k] = v
            else:
                try:
                    safe[k] = float(v)
                except (TypeError, ValueError):
                    safe[k] = str(v)

        return json.dumps(safe)
