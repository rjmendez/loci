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
  → cross-encoder BAAI/bge-reranker-v2-m3 rerank (pin MiniLM via `RERANK_MODEL`)
  → top-K results
  + Mnemosyne recall merged + deduped (optional)
  + JSONL fallback when Qdrant unavailable
```

Collections:
- `hermes_memory` — findings (named vectors: dense=768 cosine + sparse=BM25 IDF); created on first Qdrant connection
- `hermes_verdicts` — pre-answer claim check verdicts; created lazily on the first verdict write

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
| `QDRANT_URL` | _(unset — Qdrant search disabled)_ | Qdrant instance URL, e.g. `http://localhost:6333` |
| `QDRANT_API_KEY` | _(none)_ | Qdrant API key if required |
| `QDRANT_COLLECTION_PREFIX` | `hermes_memory` | Name of the shared findings collection (used verbatim, nothing is appended) |
| `OLLAMA_BASE_URL` | _(unset — falls back to 384-dim fastembed, which mismatches the 768-dim collection unless `EMBED_DIM=384`)_ | Ollama instance URL, e.g. `http://localhost:11434` |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `LOCI_MEMORY_DIR` | `~/.hermes/memory-sessions` | Local JSONL storage root |
| `LOCI_MNEMO_BANK` | `default` | Mnemosyne bank name (optional) |
| `LOCI_REFLECTION_INVESTIGATION` | `copilot-self-reflection-loop` | Default investigation for reflection loop |

## Claude Code / MCP wiring

The repo checks in a `.mcp.json` at its root, so a Claude Code session started
**inside this checkout** registers the server as `loci` with no per-machine
setup. Its paths are relative — Claude Code launches stdio servers with the
project directory as cwd, so they resolve in any clone:

```json
{
  "mcpServers": {
    "loci": {
      "type": "stdio",
      "command": "mcp/.venv/bin/python",
      "args": ["mcp/server.py"]
    }
  }
}
```

It carries no `env` block: `server.py` loads the repo-root `.env` and then
`mcp/.env` (override) at import, so credentials and host-specific endpoints stay
in gitignored files rather than a committed config. Complete [Setup](#setup)
first — the interpreter it names does not exist until you create `mcp/.venv`,
and Claude Code will report the server as failed to start until you do.

To use the server from **outside** this checkout, register it by absolute path
in `~/.claude/settings.json` instead. Keep the name `loci`; registering the same
server twice under two names publishes every tool twice:

```json
{
  "mcpServers": {
    "loci": {
      "type": "stdio",
      "command": "/path/to/loci/mcp/.venv/bin/python",
      "args": ["/path/to/loci/mcp/server.py"],
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_API_KEY": "",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "LOCI_MEMORY_DIR": "~/.hermes/memory-sessions"
      }
    }
  }
}
```

## Tools (71)

71 tools are registered at runtime: 60 defined in `server.py`, plus 11 code-graph
tools registered from `graph_tools.py` via `graph_tools.register(mcp, _get_ladybug)` at
server startup — which is why they do not appear as `@mcp.tool()` in `server.py`.
The same inventory is listed in the [top-level README](../README.md#mcp-tools-71).

**Session management:**
- `investigation_start(investigation_id, title, context?)` — create or resume a session
- `investigation_load(investigation_id, last_n_findings?, include_retracted?)` — load manifest + recent findings
- `investigation_note(investigation_id, field, value)` — update context/hypothesis/next_step/open_question_add/open_question_remove/checked_source/closed_summary
- `investigation_reflect(investigation_id)` — synthesize current investigation state
- `investigation_list(limit?, offset?, summary?)` — list sessions, most-recent first; paginated (`limit=30`, `offset=0`) and compact (`summary=True`) by default. Pass `summary=False` for full records, `limit=0` (or any value `<=0`) for all — `offset` is still honored in no-limit mode, so it returns all *remaining* records starting at `offset`. Returns `{investigations, total, limit, offset}`
- `investigation_as_of(investigation_id, as_of_timestamp)` — findings as they were believed at a point in time
- `investigation_share(investigation_id, agent_ids)` — grant access to one or more agents
- `investigation_unshare(investigation_id, agent_ids)` — revoke access

**Finding storage:**
- `investigation_store(investigation_id, finding_type, text, source, confidence?, tags?, derived_from?)` — store a finding
  - `finding_type`: one of `observed | inferred | assumed | gap`
  - Returns: `{"stored": true, "finding_id": "<uuid>", "type": "<finding_type>", "mnemo_stored": true}`
- `memory_retract(investigation_id, target, reason?, dry_run?, scope_semantic?)` — soft-delete findings matching `target`; `dry_run=True` by default, pass `dry_run=False` to actually retract
- `memory_restore(investigation_id, finding_id?, retraction_id?, reason?)` — undo a retraction
- `finding_resolve(investigation_id, finding_id, resolution, note?)` — mark a finding fixed/intentional/wontfix
- `procedure_attempt(investigation_id, finding_id, success)` — record a pass/fail attempt against a procedure-type finding
- `procedure_search(query, investigation_id?, limit?)` — search procedure-type findings

**Search & retrieval:**
- `investigation_search(query, investigation_id?, ...)` — hybrid RAG search
- `investigation_related_cases(entities, entity_type?, limit_per_entity?)` — find past investigations sharing the given entities
- `rag_context_search(query, ...)` — cross-collection RAG search
- `memory_surface(context, investigation_id?, top_k?)` — proactively surface prior findings for the current working context
- `memory_route(query, agent_id?, top_k?, deduplicate?)` — agent-mesh search across all investigations
- `memory_hints(investigation_id, limit?, since_ts?)` — recent findings as lightweight hints
- `memory_confidence(query, top_k?)` — metamemory: how reliably memory knows a topic
- `ground(title, focus?, case_ids?, entities?, code_refs?, budget_chars?, allow_keyword?, graph_available?)` — assemble a char-budgeted, provenance-tagged grounding block

**Claim validation:**
- `investigation_pre_answer_check(investigation_id, claims, ...)` — validate claims against evidence before answering
- `investigation_evidence_precheck(investigation_id, proposed_query, min_similarity?)` — lightweight duplicate/evidence check
- `verify_finding(claim, context?, investigation_id?)` — adversarially verify a claim with the local model
- `investigation_verify_all(investigation_id, limit?)` — batch adversarial-verify the open findings
- `investigation_reason(investigation_id, question, perspectives?, ground_threshold?, persist?)` — grounded multi-perspective reasoning over findings
- `conflict_list(investigation_id)` — list detected conflicts
- `conflict_resolve(investigation_id, conflict_id, verdict)` — record a verdict on a conflict

**Entity & provenance:**
- `investigation_entity_lookup(entity, entity_type?, investigation_id?, limit?)` — find findings by entity
- `investigation_finding_provenance(finding_id, investigation_id)` — trace finding lineage
- `entity_list(investigation_id, entity_type?)` — list entities extracted from findings
- `entity_timeline(investigation_id, entity_id)` — chronological findings mentioning an entity
- `causal_edges_list(investigation_id)` — list causal edges inferred for an investigation

**Memory tiering:**
- `memory_promote(investigation_id, finding_id, tier)` — promote a finding to a higher tier
- `memory_demote(investigation_id, finding_id, tier)` — demote a finding to a lower tier

**Contracts & wiring obligations:**
- `contract_declare(investigation_id, entity, role, fields, protocol?)` — store a cross-boundary contract declaration
- `contract_query(investigation_id, entity, role?)` — query stored contract declarations
- `contract_check(investigation_id, field_name, entity?)` — check a field name against stored contracts
- `wiring_obligation_declare(investigation_id, class_name, method_name, expected_effect)` — declare an unverified integration point
- `wiring_obligation_list(investigation_id, resolved?)` — list wiring obligations
- `wiring_obligation_resolve(investigation_id, finding_id, evidence)` — resolve one with evidence of fulfillment

**Local model offload:**
- `llm_local(prompt, model?, fmt?, max_tokens?, temperature?, keep_alive?)` — generate with the local GPU model
- `generate_batch(prompts, model?, max_tokens?, fmt?)` — generate for many prompts at once
- `query_expand(query, n_queries?, n_keywords?)` — HyDE-lite query expansion
- `classify_text(text, labels)` — pick the best label for a text
- `compress_text(text, max_chars?)` — semantically condense text to a char budget
- `semantic_dedup(items, threshold?, text_key?)` — cluster near-duplicate items by embedding similarity
- `semantic_relevance(texts, topic)` — cosine relevance of each text to a topic

**Audit & health:**
- `audit_log(tool_name, inputs_json, output, investigation_id?, embedding_text?)` — record a tool call in the audit trail
- `memory_self_check()` — provenance + contradiction self-check over stored findings
- `memory_health()` — Qdrant + Mnemosyne + embedder status
- `loci_health()` — read-only self-diagnosis snapshot of the server
- `code_memory_correlate(investigation_id, target_file?, entity?)` — correlate code files/entities with findings
- `memory_consolidate()` — run Mnemosyne sleep consolidation over all sessions

**Portability:**
- `investigation_export(investigation_id, include_embeddings?)` — export an investigation as a portable JSON bundle
- `investigation_import(bundle_json, new_title?)` — import a bundle produced by `investigation_export`

**Reflection loop:**
- `reflection_loop_seed(investigation_id?, session_events_limit?, process_logs_limit?, reset_queue?)` — enqueue process logs / session events for bounded self-reflection
- `reflection_loop_tick(max_items?, max_lines_per_file?, store_item_findings?)` — process a batch: tail-read, classify, dedupe, store findings
- `reflection_loop_status()` — inspect queue depth and processing stats

**Code graph** (registered from `graph_tools.py`):
- `code_graph_ingest(path, max_files?, replace?)` — parse with tree-sitter and ingest the symbol graph
- `code_graph_query(cypher, params?)` — read-only Cypher over the code + findings graph
- `code_memory_relink()` — rebuild all Finding → CodeSymbol edges
- `code_memory_map(anchor, anchor_type?, hops?)` — code↔memory neighbourhood around an anchor
- `symbol_impact(symbol, hops?)` — blast radius of a symbol across code and memory
- `impact_report(symbol, hops?)` — change blast radius: transitive callers and callees
- `finding_code_context(finding_id)` — the code a finding references, with callers/callees
- `investigation_code_briefing(investigation_id, top?)` — the code story of an investigation
- `subsystem_report(anchor, limit?)` — full picture of the code under a path or package prefix
- `related_investigations_via_code(investigation_id, limit?)` — other investigations touching the same symbols
- `dead_code_candidates(lang?, limit?)` — functions with no caller and no finding reference

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

- Embed batching hard-capped at 32 (`_EMBED_BATCH_SIZE` in `server.py`, not an env var) — Ollama stalls silently above this
- Named vectors `dense`+`sparse` in same Qdrant point — hybrid RRF without extra collections
- Payload indexes on `investigation_id`, `confidence`, `record_type`, `tags` — O(log N) filter at scale
- BM25 kept in RAM (`SparseVectorParams(index=SparseIndexParams(on_disk=False), modifier=Modifier.IDF)`)
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
