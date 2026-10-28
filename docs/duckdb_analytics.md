# DuckDB Zero-Migration Analytics

Loci ships an optional DuckDB analytics layer that queries **existing JSONL files directly** — no schema migration, no index rebuild, no data copy.

## Why DuckDB

The baseline cross-investigation search in `server.py` iterates every investigation directory, opens each `findings.jsonl`, and scans the full file in Python — **O(K × N)** where K is the number of investigations and N is the average findings per investigation.

DuckDB on the same files reads only the rows and columns the query needs (**columnar pushdown**). For cross-investigation queries this is typically **10–100× faster**.

| Approach | Complexity | Notes |
|---|---|---|
| Python JSONL scan | O(K × N) | Opens every file, reads every row |
| DuckDB `read_ndjson_auto()` | sub-linear in practice | Parallel glob, columnar I/O, predicate pushdown |

## Installation

```bash
pip install duckdb            # standalone
pip install "loci[analytics]" # with the loci package
```

DuckDB is an **optional** dependency. All existing tools continue to work without it.

## How It Works

DuckDB's `read_ndjson_auto()` with `union_by_name=true` reads JSONL files matching a glob pattern and unions their schemas automatically. Heterogeneous schemas (older findings missing newer fields) are handled transparently — missing columns become NULL.

```python
from loci.analytics import cross_investigation_search, cross_investigation_stats

# Search across every investigation
results = cross_investigation_search(
    query_text="buffer overflow",
    min_confidence=0.7,
    finding_type="observed",
    days=30,
    limit=50,
)

# Aggregate statistics
stats = cross_investigation_stats()
# {"total_findings": 1240, "investigation_count": 42, "avg_confidence": 0.71, ...}
```

## Memory Directory

The analytics layer reads from the same directory as the rest of Loci:

| Source | Path |
|---|---|
| `LOCI_MEMORY_DIR` env var | used if set |
| `MEMORY_DIR` env var | fallback |
| Default | `~/.loci/memory-sessions/` |

Each investigation is a subdirectory; `findings.jsonl` is the append-only finding log:

```
~/.loci/memory-sessions/
  inv-<id>/
    findings.jsonl   ← queried by DuckDB
    manifest.json
    audit.jsonl
```

## MCP Tools

### `mcp__loci__investigation_search_fast`

Fast cross-investigation finding search.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | `""` | Full-text substring filter (ILIKE on `text` / `content`) |
| `min_confidence` | `float` | `null` | Minimum `numeric_confidence` |
| `finding_type` | `str` | `null` | Filter by `type` field (e.g. `"observed"`, `"inferred"`) |
| `days` | `int` | `null` | Recency filter — findings newer than N days |
| `limit` | `int` | `50` | Maximum results returned |

Response:

```json
{
  "results": [...],
  "count": 12,
  "backend": "duckdb"
}
```

### `mcp__loci__investigation_stats_fast`

No parameters. Returns:

```json
{
  "total_findings": 1240,
  "investigation_count": 42,
  "avg_confidence": 0.71,
  "open_count": 980,
  "resolved_count": 260
}
```

## Graceful Degradation

If `duckdb` is not installed:

- `loci.analytics.is_available()` returns `False`
- `cross_investigation_search()` / `cross_investigation_stats()` raise `ImportError("duckdb not installed: pip install duckdb")`
- The MCP tools return `{"error": "duckdb_not_installed", "install": "pip install duckdb"}` without crashing the server

## Schema Heterogeneity

DuckDB's `union_by_name=true` handles investigations with different finding schemas (e.g. older records missing `content` or `source`). Missing columns are returned as `NULL` and filtered safely with `TRY_CAST`.

## Zero Migration

No database files are written. DuckDB operates in `:memory:` mode and reads the existing JSONL files. Adding or removing the `duckdb` package has no effect on stored data.
