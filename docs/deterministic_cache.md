# Deterministic Graph Tool Cache

## Overview

Several Loci tools are **fully deterministic** — they make zero model calls and always
produce the same output for the same input. These are ideal cache candidates: a repeated
call with identical parameters should skip computation entirely.

`mcp/graph_cache.py` provides a lightweight JSONL-backed result cache for these tools.

---

## Deterministic Tool Catalogue

| Tool | Model calls | Notes |
|---|---|---|
| `code_graph_query` | 0 | Pure graph traversal |
| `symbol_impact` | 0 | Structural reachability only |
| `impact_report` | 0 | Aggregates graph edges |
| `subsystem_report` | 0 | Static subgraph summary |
| `related_investigations_via_code` | 0 | Graph + index lookup |
| `causal_infer` (heuristic path) | 0–1 | Heuristic runs first; LLM only on low-confidence fallback |

`causal_infer` is a special case: the heuristic path is deterministic and cacheable.
When the cache returns a hit, the LLM slow path is bypassed entirely.

---

## Cache Key Design

```
key = sha256(tool_name | sorted_params_json | graph_revision)[:16]
```

- **`tool_name`** — the MCP tool identifier string.
- **`sorted_params_json`** — `json.dumps(params, sort_keys=True)`, ensuring parameter
  order does not affect the key.
- **`graph_revision`** — optional string (e.g. a git SHA or ingest timestamp) that can
  be used to partition cache entries by graph version. Defaults to `""`.

The 16-character hex prefix gives 64 bits of collision resistance — sufficient for the
number of distinct tool calls in any single session.

---

## Storage

Results are stored as **JSONL files, one file per tool**:

```
$LOCI_GRAPH_CACHE_DIR/
    code_graph_query.jsonl
    symbol_impact.jsonl
    impact_report.jsonl
    ...
```

Each line is a JSON object (`GraphCacheEntry.to_dict()`). Reads scan the file for a
matching key; writes append. The file is deleted (compacted) on invalidation.

---

## Invalidation Triggers

The cache must be invalidated whenever the underlying graph changes:

| Event | Action |
|---|---|
| `code_graph_ingest` completes | `GraphCache().invalidate()` — clear all |
| `code_memory_relink` completes | `GraphCache().invalidate()` — clear all |
| TTL expires (`LOCI_GRAPH_CACHE_TTL_MIN`) | Entry skipped on read (lazy expiry) |
| Explicit per-tool reset | `GraphCache().invalidate("tool_name")` |

Recommended: call `GraphCache().invalidate()` from the `code_graph_ingest` and
`code_memory_relink` server tool wrappers immediately after the underlying operation
succeeds.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LOCI_GRAPH_CACHE_DIR` | `~/.loci/graph_cache` | Directory for JSONL cache files |
| `LOCI_GRAPH_CACHE_TTL_MIN` | `30` | TTL in minutes; entries older than this are treated as misses |

---

## Expected Hit Rates

Hit rates depend on query diversity within a session.

| Tool | Expected hit rate | Rationale |
|---|---|---|
| `code_graph_query` | 40–70% | Agents re-query the same symbols during investigation |
| `symbol_impact` | 30–60% | Impact queries cluster around recently changed modules |
| `impact_report` | 25–50% | Often re-run after minor context changes |
| `subsystem_report` | 50–80% | Subsystem queries are highly repetitive |
| `related_investigations_via_code` | 20–40% | More unique per session |
| `causal_infer` (heuristic) | 30–60% | Same causal chains revisited during reasoning loops |

---

## Integration Pattern (server.py tool wrappers)

```python
from graph_cache import GraphCache

_graph_cache = GraphCache()

async def handle_code_graph_query(params: dict) -> dict:
    hit = _graph_cache.get("code_graph_query", params)
    if hit:
        return hit.result  # cache hit — zero computation

    result = await _run_code_graph_query(params)  # actual tool
    _graph_cache.put("code_graph_query", params, result)
    return result
```

For tools that accept a `graph_revision` parameter, thread it through:

```python
hit = _graph_cache.get("code_graph_query", params, graph_revision=current_revision())
```

---

## `causal_infer` Integration

`causal_infer` has two internal paths:

1. **Heuristic path** — deterministic rule-based inference (fast, zero LLM calls).
2. **LLM fallback** — invoked when heuristic confidence is below threshold.

Cache the heuristic result; on a cache hit, return it directly without entering the
LLM fallback:

```python
async def handle_causal_infer(params: dict) -> dict:
    hit = _graph_cache.get("causal_infer", params)
    if hit:
        return hit.result  # skip LLM entirely

    heuristic_result = run_heuristic_infer(params)
    if heuristic_result.confidence >= HEURISTIC_THRESHOLD:
        _graph_cache.put("causal_infer", params, heuristic_result.to_dict())
        return heuristic_result.to_dict()

    # Low confidence — fall through to LLM (result not cached)
    return await run_llm_infer(params)
```

Only cache high-confidence heuristic results. Do **not** cache LLM fallback results
(those belong to the Qdrant-backed memory cache in `cache.py`).
