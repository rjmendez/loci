# Memory System Architecture

## Overview

hermes_memory provides multi-layer persistent memory for LLM agents running in
Claude Code and compatible agent frameworks. It handles three time horizons:

| Horizon | Mechanism | Location |
|---|---|---|
| **Per-turn** | Qdrant grounding fan-out (< 100ms) | `scripts/hooks/pre_llm_grounding.py` |
| **Per-session** | Hook-driven event logging, Mnemosyne writes, Qdrant session sync | `scripts/hooks/` + Mnemosyne DB + Qdrant |
| **Long-term** | Cron consolidation, decay-triggered refresh | `scripts/` + Hermes cron |

---

## Data stores

### Mnemosyne SQLite (`~/.hermes/mnemosyne/data/mnemosyne.db`)

Primary structured store. Tables queried by this codebase:

| Table | Purpose | TTL model |
|---|---|---|
| `memories` | Raw memory writes (append-only working store) | Pruned by Mnemosyne sleep |
| `working_memory` | Episodic traces from current/recent sessions | Decays via Ebbinghaus (R < 0.3 triggers refresh) |
| `episodic_memory` | Consolidated session summaries and distilled events | Longer half-life; promoted from working_memory |
| `consolidated_facts` | Context-free semantic facts | High stability; survives indefinitely |
| `triples` | Subject-predicate-object knowledge graph | Permanent unless superseded |
| `scratchpad` | Mutable per-session working scratchpad | Overwritten each session |
| `facts` | Structured fact store | Updated by fact-extraction pipeline |
| `gists` | Compressed memory gists | Produced by consolidation |

> Note: `graph_edges`, `conflicts`, and `annotations` may exist in the Mnemosyne
> schema but are not directly queried by the scripts in this repository.

### Qdrant (configured via `QDRANT_URL`)

Vector search layer. 768-dim Cosine, `nomic-embed-text` embeddings throughout.
**Qdrant is disabled when `QDRANT_URL` is unset** — there is no automatic
localhost fallback in `mcp/server.py` or the grounding hooks. Only
`a2a_server/server.py` documents `http://localhost:6333` as a conventional
default in its inline comments.

| Collection | Contents | Named vector |
|---|---|---|
| `mnemosyne` | Mirror of Mnemosyne working+episodic memory | `{"dense": ...}` |
| `hermes_sessions` | Session-level traces synced by `session_end_sync.py` | `{"dense": ...}` |
| `hermes_memory` | Investigation notes, research findings (MCP server primary collection) | `{"dense": ...}` |
| `hermes_verdicts` | Claim-check verdicts for `investigation_pre_answer_check` | `{"dense": ...}` |
| `<custom>` | Domain-specific collections configured via `GROUNDING_EXTRA_COLLECTIONS` | flat or named |
| `memgas_l1` | MemGAS L1 utterance layer | `{"dense": ...}` |
| `memgas_l2` | MemGAS L2 summary layer | `{"dense": ...}` |
| `memgas_l3` | MemGAS L3 topic/semantic layer | `{"dense": ...}` |
| `eval_scores` | Longitudinal eval harness scores | `{"dense": ...}` |
| `score_traces` | SCoRe correction pairs for fine-tuning | `{"dense": ...}` |

The MCP server (`mcp/server.py`) uses the collection name set by
`QDRANT_COLLECTION_PREFIX` (default: `hermes_memory`) as its primary
investigation findings store.

---

## Per-turn grounding pipeline

```
User message
    │
    ▼
UserPromptSubmit hook (user_prompt_grounding.sh)
    │
    ▼
pre_llm_grounding.py (v3)
    │
    ├── 1. Extract intent from user message
    ├── 2. Embed via OLLAMA_BASE_URL + nomic-embed-text (~70ms warm)
    │       Falls back to BeamMemory (v2/SQLite path) when Ollama is unreachable
    ├── 3. Fan-out parallel Qdrant search — built-in + custom collections (~30ms)
    │       mnemosyne, hermes_sessions, hermes_memory
    │       + GROUNDING_EXTRA_COLLECTIONS (grounding hook env var)
    ├── 4. Score fusion: Qdrant cosine * importance weight
    ├── 5. _keyword_rerank(): keyword overlap boost (skill shadowing mitigation)
    ├── 6. Multi-signal ranking on four axes:
    │       relevance (cosine × importance) · recency (exponential decay)
    │       · trust (confidence tier) · record type (observed > inferred > assumed > gap)
    │       Weights: RANKER_W_RELEVANCE, RANKER_W_RECENCY, RANKER_W_TRUST, RANKER_W_TYPE
    ├── 7. Stigmergic pheromone deposit on retrieved hermes_memory points
    │       (pheromone stored in Qdrant payload; decays by PHERO_HALFLIFE_H)
    ├── 8. MMR diversity selection with ε-exploration
    │       (MMR_LAMBDA controls relevance-vs-diversity trade-off;
    │        PHERO_EPSILON controls random exploration probability)
    ├── 9. Optional spreading activation enrichment (SA-RAG, arxiv 2512.15922)
    │       Seeds from mnemosyne hits with mnemosyne_id payload;
    │       skipped if SA takes > HOOK_SA_TIMEOUT_MS or SA module is absent
    ├── 10. Filter: MIN_SCORE, MIN_IMPORTANCE, keep top HOOK_RECALL_TOP_K results
    └── 11. Inject MEMORY MATCH block into Claude's context window
```

**Total latency target:** < 100ms (70ms embed + 30ms parallel Qdrant)

A companion hook, `scripts/hooks/pre_tool_grounding.py`, provides the same
grounding injection before tool calls.

### Key grounding env vars

| Variable | Default | Purpose |
|---|---|---|
| `QDRANT_URL` | _(none — Qdrant disabled if unset)_ | Qdrant instance URL |
| `QDRANT_API_KEY` | `""` | Qdrant API key |
| `OLLAMA_BASE_URL` | _(none)_ | Embedding base URL (no `/v1` suffix) |
| `MNEMOSYNE_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model name |
| `EMBED_API_KEY` | `""` | Cloud embedding provider API key |
| `HOOK_RECALL_TOP_K` | `3` | Maximum results injected per turn |
| `HOOK_RECALL_MIN_SCORE` | `0.55` | Minimum cosine score threshold |
| `HOOK_RECALL_MIN_IMPORTANCE` | `0.2` | Minimum importance threshold |
| `GROUNDING_EXTRA_COLLECTIONS` | `""` | Extra Qdrant collections for the grounding hook only |
| `HOOK_SA_ENABLED` | `true` | Enable spreading activation enrichment |
| `HOOK_SA_TIMEOUT_MS` | `25` | SA budget in ms; skipped if exceeded |
| `MMR_LAMBDA` | `0.75` | MMR relevance weight (1.0 = pure relevance) |
| `PHERO_BETA` | `0.08` | Pheromone score boost coefficient |
| `PHERO_HALFLIFE_H` | `24` | Pheromone evaporation half-life (hours) |

> `GROUNDING_EXTRA_COLLECTIONS` is read independently by the grounding hook.
> `EXTRA_RAG_COLLECTIONS` is a separate variable read by `a2a_server/server.py`
> for its `rag_search` skill. The `.env.example` documents both; if you want
> the same extra collections in both paths, set both variables.

---

## Per-session event capture

Claude Code hooks write structured events at each lifecycle point:

```
SessionStart      → session_start_guard.sh
                    Logs CWD, project, MCP status, Qdrant probe, rules summary

UserPromptSubmit  → user_prompt_grounding.sh
                    Grounding fan-out per turn (see above)

PreToolUse        → pre_bash_guard.sh             (Bash only: guard protocol)
                  → hermes_pre_tool_grounding.sh   (all tools: Hermes audit)

PostToolUse       → post_bash_success_memory.sh   (Bash success logging)

PostToolUseFailure → post_bash_failure_memory.sh   (Bash: repeated failure → Mnemosyne)
                   → post_tool_failure_reflection.sh  (all tools: Reflexion trace)

PreCompact        → pre_compact_guard.sh
                    Checkpoint before context compaction

Stop / SessionEnd → session_end_evaluate_guard.sh
                    Writes session evaluation to Mnemosyne and local log files
                  → scripts/hooks/session_end_sync.py
                    Reads session messages from state.db, embeds, and upserts
                    a single point into hermes_sessions Qdrant collection
```

Hook state directory: `~/.claude/hook-state/`

A shared `env.sh` in `~/.claude/hooks/` loads common environment variables for
all hooks.

### session_end_sync.py env vars

| Variable | Default | Purpose |
|---|---|---|
| `QDRANT_URL` | _(none — sync skipped if unset)_ | Qdrant instance URL |
| `OLLAMA_BASE_URL` | _(none)_ | Embedding base URL |
| `MNEMOSYNE_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `MNEMOSYNE_EMBEDDING_DIM` | `768` | Vector dimension |
| `HERMES_STATE_DB` | `~/.hermes/state.db` | Claude Code session database |
| `HERMES_SYNC_CACHE` | `~/.hermes/.session_sync_cache` | Per-session mtime cache |
| `HERMES_AGENT_ID` | `""` | Agent identity tag stamped on payloads |

---

## MCP server (loci-mcp)

`mcp/server.py` is registered as `FastMCP('loci')` and exposes 18 tools under
the `loci-mcp` server name. Key env vars:

| Variable | Default | Purpose |
|---|---|---|
| `QDRANT_URL` | _(none — Qdrant disabled if unset)_ | Qdrant instance URL |
| `QDRANT_API_KEY` | `""` | Qdrant API key |
| `QDRANT_COLLECTION_PREFIX` | `hermes_memory` | Primary investigation findings collection |
| `OLLAMA_BASE_URL` | _(none)_ | Embedding base URL |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `EMBED_API_KEY` | `""` | Cloud embedding provider key |
| `HERMES_MEMORY_DIR` | `~/.hermes/memory-sessions` | JSONL session storage root |
| `HERMES_MNEMO_BANK` | `default` | Active Mnemosyne bank |

> `mcp/server.py` uses `OLLAMA_BASE_URL` and `EMBED_MODEL` for embedding.
> `a2a_server/server.py` uses `MNEMOSYNE_EMBEDDING_API_URL` and `MNEMOSYNE_EMBEDDING_MODEL`.
> `session_end_sync.py` uses `OLLAMA_BASE_URL` and `MNEMOSYNE_EMBEDDING_MODEL`.
> These are three distinct env var names for the embedding base URL. Set all
> relevant variables in your `.env` when running multiple components together.

---

## Consolidation and self-improvement pipeline

Cron jobs (live in `~/.hermes/cron/jobs.json`):

| Job | Script / Command | Interval | Purpose |
|---|---|---|---|
| mnemosyne-consolidation | `mnemosyne_activity_check.py` (agent decides whether to call `mnemosyne_sleep`) | 20m | Activity-gated Mnemosyne consolidation |
| mnemosyne-session-summarizer | `mnemosyne_activity_check.py` (agent writes memories, triples, scratchpad) | 20m | Full session archival: memories, triples, scratchpad, sleep |
| mnemosyne-sleep-cli | `mnemosyne_sleep_all.sh` | 30m | Prune expired working memory (no-agent mode) |
| ebbinghaus-memory-decay | `ebbinghaus_consolidation.py` | 47m | Decay-triggered Qdrant refresh |
| amem-consolidation | `amem_consolidation.py` | 61m | Cross-link graph + conflict detection |
| skill-annotation-updater | `skill_annotation_updater.py` | 120m | SKILL.md learned constraints |
| agentHER-relabeler | `agentHER_relabeler.py` | 720m | Failure → positive trace relabeling |
| eval-harness-weekly | `eval/harness.py` | 10080m | Longitudinal grounding quality score |

> The `mnemosyne-consolidation` and `mnemosyne-session-summarizer` jobs both use
> `mnemosyne_activity_check.py` as the script. The cron runner provides that
> script's output as context; the agent prompt then decides whether to invoke
> `mnemosyne_sleep` based on the output. The script `mnemosyne_consolidate.sh`
> exists in `scripts/` but is not what these cron jobs execute.

On-demand scripts (not croned):
- `memgas_hierarchy.py --index` — rebuild MemGAS 3-level Qdrant collections
- `memgas_hierarchy.py --search <query>` — entropy-weighted 3-level search
- `exif_skill_discovery.py` — EXIF skill gap analysis → candidate SKILL.md
- `score_trace_collector.py` — build SCoRe fine-tuning dataset from logs
- `skillops_maintenance.py` — skill shadow detection + last_validated update
- `state_db_qdrant_sync.py` — sync Hermes state.db sessions → hermes_sessions Qdrant
- `mnemosyne_qdrant_sync.py` — sync Mnemosyne → mnemosyne Qdrant collection

---

## Memory hierarchy (MemGAS 3 levels)

Inspired by MemGAS (arxiv 2505.19549). Three levels map to cognitive memory types:

```
L1 — Utterances (working_memory)          ← ephemeral, session-scoped
L2 — Summaries  (episodic_memory)         ← consolidated, medium-term
L3 — Topics     (consolidated_facts)      ← semantic, long-term
```

`memgas_hierarchy.py` searches all 3 levels in parallel and uses entropy weighting:
- **Low entropy** at a level = confident, focused results = higher weight
- **High entropy** at a level = scattered results = lower weight

This prevents a noisy level from dominating the final ranking.

---

## Self-improvement loop

```
PostToolUseFailure hook
    │
    ▼
post_tool_failure_reflection.sh
    │  Categorizes failure type, writes typed trace to:
    │  - Mnemosyne (importance=7)
    │  - guard_tool_reflections.log (JSONL)
    │
    ├──► ebbinghaus_consolidation.py (every 47m)
    │    Re-embeds decayed memories → Qdrant refresh
    │
    ├──► amem_consolidation.py (every 61m)
    │    Builds semantic cross-links, flags near-duplicate conflicts
    │
    ├──► skill_annotation_updater.py (every 120m)
    │    Reads guard_tool_reflections.log → updates SKILL.md "Learned constraints"
    │
    ├──► agentHER_relabeler.py (every 720m)
    │    Relabels failure memories as positive examples via Ollama
    │    Writes synthetic positives back to Mnemosyne + Qdrant
    │
    ├──► score_trace_collector.py (on demand)
    │    Aggregates negatives/positives/corrections → SFT dataset
    │
    └──► exif_skill_discovery.py (on demand)
         Detects skill gaps from failure patterns → candidate SKILL.md
```

See [COGNITIVE_FOUNDATIONS.md](COGNITIVE_FOUNDATIONS.md) for the research basis
of each step in this loop.

---

## A2A mesh integration

`a2a_server/server.py` (Loci A2A Server) provides an HTTP A2A-protocol server
using FastAPI + uvicorn. It exposes memory operations as JSON-RPC 2.0 skills
so peer agents can recall, store, and search memories without direct Qdrant
credentials.

### A2A skills

| Skill | Description |
|---|---|
| `memory_recall` | FTS5 (fts_working + fts_episodes) + optional Qdrant semantic search |
| `memory_remember` | Write to the SQLite `memories` table with cross-agent author tagging |
| `memory_stats` | Row counts for all monitored SQLite tables + Qdrant collection sizes |
| `session_search` | Semantic search over hermes_sessions Qdrant collection |
| `memory_sleep` | Trigger Mnemosyne sleep consolidation via dashboard API |
| `rag_search` | Fan-out semantic search across all configured Qdrant collections |
| `context_broadcast` | Store locally and push to all peer A2A endpoints (PEER_A2A_URLS) |
| `mnemosyne_triple_add` | Store a knowledge triple in the SQLite `triples` table |
| `mnemosyne_triple_query` | Query the knowledge graph by subject/predicate/object |
| `gpu_inference` | Run a prompt through local Ollama |
| `docker_status` | List running Docker containers and k3s pods |
| `ua_search` | Semantic search over understand-anything knowledge graphs |

### A2A env vars

| Variable | Default | Purpose |
|---|---|---|
| `HERMES_A2A_TOKEN` | _(required — server exits if unset)_ | Bearer token for callers |
| `HERMES_A2A_HOST` | `0.0.0.0` | Bind address |
| `HERMES_A2A_PORT` | `8201` | Bind port |
| `HERMES_A2A_URL` | `http://127.0.0.1:8201` | Public base URL in agent card |
| `HERMES_A2A_TOTP_SEED` | `""` | Base32 TOTP seed (RFC 6238); disabled when empty |
| `HERMES_AGENT_ID` | `hermes-agent` | Agent identity tag |
| `QDRANT_URL` | _(none)_ | Qdrant instance URL |
| `QDRANT_API_KEY` | `""` | Qdrant API key |
| `MNEMOSYNE_EMBEDDING_API_URL` | _(none)_ | Embedding base URL (with `/v1`) |
| `MNEMOSYNE_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `MNEMOSYNE_EMBEDDING_DIM` | `768` | Vector dimension |
| `MNEMOSYNE_DATA_DIR` | `~/.hermes/mnemosyne/data` | Directory containing mnemosyne.db |
| `EXTRA_RAG_COLLECTIONS` | `""` | Extra collections for the A2A rag_search skill |
| `PEER_A2A_URLS` | `""` | Comma-separated peer A2A endpoints for context_broadcast |
| `PEER_A2A_TOKEN` | `""` | Shared bearer token for all peers |

`scripts/a2a_context_bridge.py` subscribes to Hermes events and routes context
updates to the A2A `context_broadcast` endpoint, propagating discoveries across
the mesh.
