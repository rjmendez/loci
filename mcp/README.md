# Loci — MCP Server

Hybrid RAG + pre-answer claim validation + provenance tracing MCP server.

Persistent investigation memory for AI agents: tracks findings, observations,
inferences, assumptions, and tool audit logs across sessions. Uses Qdrant for
hybrid dense+sparse search with cross-encoder reranking, and optionally
Mnemosyne as a shared memory substrate.

## Architecture

```
query
  → nomic-embed-text (768-dim, Ollama)    [dense]
  → fastembed Qdrant/bm25                 [sparse]
  → Qdrant RRF fusion (dense + sparse prefetch)
  → candidate pool (limit×5 overfetch)
  → cross-encoder ms-marco-MiniLM-L-6-v2 rerank
  → top-K results
  + Mnemosyne recall merged + deduped (optional)
  + JSONL fallback when Qdrant unavailable
```

Collections created on first run:
- `hermes_memory` — findings (named vectors: dense=768 cosine + sparse=BM25 IDF)
- `hermes_verdicts` — pre-answer claim check verdicts

## Requirements

- Python 3.11+
- [Qdrant](https://qdrant.tech/) — local or remote instance
- [Ollama](https://ollama.com/) with `nomic-embed-text` pulled — for 768-dim embeddings
- [Mnemosyne](https://github.com/loci-project/mnemosyne) — optional, for cross-session shared memory

## Setup

```bash
git clone https://github.com/<your-org>/loci
cd loci/mcp
python3 -m venv .venv
.venv/bin/pip install -e .
cp .env.example .env  # edit QDRANT_URL, QDRANT_API_KEY, OLLAMA_BASE_URL
```

With optional Mnemosyne:
```bash
.venv/bin/pip install -e ".[mnemosyne]"
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant instance URL |
| `QDRANT_API_KEY` | _(none)_ | Qdrant API key if required |
| `QDRANT_COLLECTION_PREFIX` | `hermes_memory` | Prefix for Qdrant collections |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama instance URL |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `HERMES_MEMORY_DIR` | `~/.hermes/memory-sessions` | Local JSONL storage root |
| `HERMES_MNEMO_BANK` | `default` | Mnemosyne bank name (optional) |
| `HERMES_REFLECTION_INVESTIGATION` | `copilot-self-reflection-loop` | Default investigation for reflection loop |

## Claude Code / MCP wiring

Add to `~/.claude/settings.json` under `mcpServers`:

```json
{
  "mcpServers": {
    "hermes_memory": {
      "command": "/path/to/loci-mcp/.venv/bin/python",
      "args": ["/path/to/loci-mcp/server.py"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_API_KEY": "",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "HERMES_MEMORY_DIR": "~/.hermes/memory-sessions"
      }
    }
  }
}
```

## Tools (24)

**Session management:**
- `investigation_start(investigation_id, title, context?)` — create or resume a session
- `investigation_load(investigation_id, last_n_findings?, include_retracted?)` — load manifest + recent findings
- `investigation_note(investigation_id, field, value)` — update hypothesis/next_step/open_questions/checked_source
- `investigation_reflect(investigation_id)` — synthesize current investigation state
- `investigation_list(limit?, offset?, summary?)` — list sessions, most-recent first; paginated (`limit=30`, `offset=0`) and compact (`summary=True`) by default. Pass `summary=False` for full records, `limit=0` for all. Returns `{investigations, total, limit, offset}`

**Finding storage:**
- `investigation_store(investigation_id, finding_type, text, source, confidence?, tags?, derived_from?)` — store a finding
  - `finding_type`: one of `observed | inferred | assumed | gap`
  - Returns: `{"stored": true, "finding_id": "<uuid>", "type": "<finding_type>", "mnemo_stored": true}`
- `memory_retract(investigation_id, finding_id, reason?)` — soft-delete a finding
- `memory_restore(investigation_id, finding_id, reason?)` — undo a retraction

**Search & retrieval:**
- `investigation_search(query, investigation_id?, ...)` — hybrid RAG search
- `investigation_related_cases(investigation_id)` — find related past investigations
- `rag_context_search(query, ...)` — cross-collection RAG search

**Claim validation:**
- `investigation_pre_answer_check(investigation_id, claims, ...)` — validate claims against evidence before answering
- `investigation_evidence_precheck(investigation_id, claim)` — lightweight duplicate/evidence check

**Entity & provenance:**
- `investigation_entity_lookup(entity, entity_type?, investigation_id?, limit?)` — find findings by entity
- `investigation_finding_provenance(investigation_id, finding_id)` — trace finding lineage

**Audit & health:**
- `audit_log(investigation_id, event_type, detail)` — log an event
- `memory_self_check()` — provenance + contradiction self-check over stored findings
- `memory_health()` — Qdrant + Mnemosyne + embedder status
- `code_memory_correlate(investigation_id, code_snippet)` — correlate code with findings
- `memory_consolidate()` — run Mnemosyne sleep consolidation over all sessions

**Reflection loop:**
- `reflection_loop_seed(paths, kind?)` — enqueue process logs / session events for bounded self-reflection
- `reflection_loop_tick(batch_size?, max_lines?, max_bytes?)` — process a batch: tail-read, classify, dedupe, store findings
- `reflection_loop_status()` — inspect queue depth and processing stats

### investigation_start return shape

```json
{
  "status": "created",
  "manifest": {
    "id": "<investigation_id>",
    "title": "<title>",
    "context": "",
    "status": "active",
    "created_at": "<iso8601>",
    "updated_at": "<iso8601>",
    "hypothesis": null,
    "open_questions": [],
    "next_step": null,
    "checked_sources": {},
    "finding_counts": {"observed": 0, "inferred": 0, "assumed": 0, "gap": 0}
  }
}
```

**Note:** Investigation ID is at `result["manifest"]["id"]`, not `result["investigation_id"]`.

## Performance notes

- `EMBED_BATCH_SIZE=32` hard ceiling — Ollama stalls silently above this
- Named vectors `dense`+`sparse` in same Qdrant point — hybrid RRF without extra collections
- Payload indexes on `investigation_id`, `confidence`, `record_type`, `tags` — O(log N) filter at scale
- BM25 kept in RAM (`SparseIndexParams(on_disk=False, modifier=Modifier.IDF)`)
- 30-day TTL purge on `hermes_memory` applied at startup

## memcheck module

The `memcheck/` package provides standalone claim validation and code hallucination
detection that can be used independently of the MCP server:

```bash
# CLI usage
.venv/bin/python -m memcheck.cli check-action < pretooluse_payload.json

# Warm daemon (holds Qdrant connection; hook_client talks to it via Unix socket)
.venv/bin/python -m memcheck.cli daemon
```

Code hallucination rules vendored from
[llm-code-hallucination-patterns](https://github.com/loci-project/llm-code-hallucination-patterns)
(MIT).
