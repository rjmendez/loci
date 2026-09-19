# Loci Architecture

_Last reviewed: 2026-09-17 — rolling document, maintained as the system changes. When a store, component, or data flow moves, update this map in the same change._

This page is the subsystem map: the data stores that hold state, the components that act on them, and how a single turn flows through the system.

## Core data stores

Loci keeps state in four places, each with a distinct job.

### 1. Investigation JSONL — source of truth for findings
`~/.hermes/memory-sessions/<inv>/`

Each investigation is a directory of append-only logs:

- `findings.jsonl` — the append log, with access-tracking rows
- `manifest.json` — investigation metadata (context, hypothesis, next step, questions, checked sources, finding counts)
- `entities.jsonl` — extracted entities (IPs, emails, hostnames, hashes, CVEs, domains)
- `retractions.jsonl` — soft tombstones

Findings are append-only. A retraction is a soft delete (tombstone), never a hard erase — it can be restored.

### 2. Qdrant — vector search
The required vector store. Collections (auto-created on first touch):

- `loci_memory` — findings (dense 768-dim cosine + sparse BM25/IDF)
- `loci_sessions` — session-history embeddings
- `loci_verdicts` — claim-validation history
- `mnemosyne` — synced Mnemosyne SQLite vectors

First call auto-creates collections and applies INT8 quantization + HNSW config.

### 3. LadybugDB graph — `graph.ladybug`
Two overlaid graphs in one store:

- **Code graph** — tree-sitter symbols with `DEFINES` / `CALLS` / `IMPORTS` edges (≈11,273 symbols on the live host)
- **Investigation graph** — `Finding` and `Entity` nodes linked by `MENTIONS` / `DERIVED_FROM` / `REFERENCES` edges (≈718 finding→code links on the live host)

This is what lets Loci answer "which findings touch this function?" and "what is the blast radius of this symbol?"

### 4. Mnemosyne SQLite — `~/.hermes/mnemosyne/data/mnemosyne.db`
Optional structured memory with a three-level hierarchy:

- **L1** — `working_memory` / utterances
- **L2** — `episodic_memory` / summaries
- **L3** — `consolidated_facts` / topics

## Major components

1. **Memory / Investigations** — the fundamental unit of persistent work. Findings are append-only, searchable, retractable (soft tombstone), and carry metadata (hypothesis, open questions, checked sources). Key tools: `investigation_start/load/store/note/reflect/search/list/share/export/import`.

2. **Code Graph** — LadybugDB stores code symbols and links findings to them. Tools: `code_graph_ingest`, `code_graph_query`, `code_memory_relink`, `code_memory_map`, `symbol_impact`, `impact_report`, `dead_code_candidates`.

3. **RAG / Grounding pipeline** — the per-turn hook (`pre_llm_grounding.py`) embeds user intent via Ollama (`nomic-embed-text`, 768-dim), fans out to Qdrant in parallel, fuses dense + sparse rankings (RRF), reranks with `BAAI/bge-reranker-v2-m3`, applies a four-term weighted score (relevance / recency / trust / type), and selects via MMR with pheromone reinforcement. Latency budget: <100ms. Output: a context block injected into the model's window.

4. **Swarm / Deep-Think** — the multi-tier reasoning engine in `deep_think_loci/`. Three versions exist: `deep-think.js`, `deep-think-loci.js` (v3.2, current default), and `deep-think-v4.js`. Shape: haiku ideation → dedicated writer (persists findings to the investigation) → verify → opus synthesis (red-team + integrity check). It ships 22 workflows for domain audits (wiring-gap, silent-failure, dead-code, schema-drift, security-boundary, performance, memory-leak, test-coverage-gap, and more).

5. **Investigation tools** — `investigation_tools.py` registers the CRUD operations. `investigation_pre_answer_check` validates claims via lexical + semantic lanes (gates at overlap ≥ 0.15 and margin ≥ 0.05 over the pool median). Verdicts land in the `loci_verdicts` Qdrant collection.

6. **Consolidation / Grooming** — `loci_groom.py` (idempotent, fail-open, shadow-first). Passes include `index --apply` (reconcile Qdrant against `findings.jsonl`, restoring lost vectors), `tags` / `knn_tags` / `codelink` (propose via a local model), and `summaries` (drive the investigation summary ladder). Scheduled: index every 6h, knn_tags daily 3am, codelink daily 3:40am, summaries daily 4:50am.

7. **Decay & Refresh** — `ebbinghaus_consolidation.py` triggers on R < 0.3 (70%+ decay) using the FSRS DSR model. `amem_consolidation.py` discovers cross-links and detects conflicts. Both run on demand, not on cron.

8. **A2A Mesh Server** — `a2a_server/server.py` exposes 13 JSON-RPC skills over HTTP for agent-to-agent memory. Key skills: `memory_recall`, `memory_remember`, `rag_search`, `context_broadcast` (fan to peers), `mnemosyne_triple_add/query`, `gpu_inference`, `docker_status`. Enables mesh-wide context sharing.

9. **Supply-chain security** — `pre_tool_grounding.py` scans writes for IOCs and injection patterns. Audit-only by default (`HOOK_BLOCK_MODE=0`), with an unconditional block on injection into agent config files (`AGENTS.md`, `CLAUDE.md`, `.cursorrules`).

10. **Session sync** — `session_end_sync.py` embeds the current session and upserts it to the `loci_sessions` collection. It falls back to Claude Code's `transcript_path` (Hermes `state.db` is Hermes-only), and an mtime cache skips unchanged sessions.

## How a turn flows

```
User message
  → pre_llm_grounding.py hook  (<100ms embed + Qdrant search)
  → MEMORY MATCH context injected into the window
  → model answers, writes findings via investigation_store
      → findings go to JSONL + Qdrant simultaneously
  → deep-think-loci can wrap multi-agent reasoning over the same investigation
  → grooming (cron): reconcile index + propose tags + link findings to code + summaries
  → code-graph edges land in LadybugDB
  → A2A server broadcasts to peers
  → ebbinghaus decay triggers optional refresh for fading memories
```

## Design principles

- **Fail-open** — a backend down means degrade, never crash.
- **Shadow-first** — model output goes to `proposals.jsonl` and is never merged until verified.
- **Idempotent** — every pass can re-run without inventing.
- **Cognitive foundations** — FSRS spacing intervals, Tulving episodic/semantic tiers, Cowan's 4-chunk limit (`HOOK_RECALL_TOP_K=3` default).
- **Multi-modal retrieval** — dense + sparse + reranking; no single threshold is sufficient alone.

## Key files

- `mcp/server.py` — FastMCP entry point
- `mcp/investigation_tools.py`, `mcp/graph_tools.py`, `mcp/llm_tools.py`, `mcp/qdrant_ops.py`, `mcp/inv_store.py`
- `mcp/graph/ladybug_store.py`
- `scripts/hooks/pre_llm_grounding.py`, `pre_tool_grounding.py`, `session_end_sync.py`
- `scripts/loci_groom.py`, `scripts/ebbinghaus_consolidation.py`, `scripts/amem_consolidation.py`
- `a2a_server/server.py`
- `deep_think_loci/workflows/deep-think-loci.js`
- Reference docs: `docs/ARCHITECTURE.md`, `docs/COMPONENTS.md`, `docs/CONCEPTS.md`, `docs/COGNITIVE_FOUNDATIONS.md`
