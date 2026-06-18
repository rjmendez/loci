# hermes_memory — Persistent Memory for AI Agent Meshes

Persistent memory and knowledge layer for AI agent meshes. Provides grounding,
consolidation, self-improvement, and longitudinal evaluation for Claude Code and
other agent sessions.

## Quick orientation

| What you want | Where to look |
|---|---|
| How the system works | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Why it's designed this way | [docs/COGNITIVE_FOUNDATIONS.md](docs/COGNITIVE_FOUNDATIONS.md) |
| What each script does | [docs/COMPONENTS.md](docs/COMPONENTS.md) |
| How to run / configure (scripts) | [docs/OPERATIONS.md](docs/OPERATIONS.md) |
| How to deploy (Docker / systemd) | [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) |

## Feature highlights

- **Qdrant vector store** — hybrid dense (768-dim nomic-embed-text) + sparse BM25 search
  with cross-encoder reranking over investigation findings
- **Mnemosyne SQLite substrate** — FTS5 full-text search, cross-session shared memory,
  bank-scoped storage for multi-agent meshes
- **MCP server (25 tools)** — investigation sessions, RAG retrieval, claim validation,
  entity lookup, provenance tracing, audit log, reflection loop, context search,
  memory consolidation, confidence estimation, and routing query
- **A2A server** — JSON-RPC 2.0 over HTTP; exposes 14 memory skills to peer agents
  without requiring the MCP stack
- **Bio-inspired memory** — slow-wave/REM consolidation, stigmergic recall scoring,
  glymphatic sweep (TTL purge), metamemory confidence tracking

## Repo layout

```
hermes_memory/
├── mcp/                   MCP server — 25 tools: investigation memory, RAG, claim validation
│   ├── server.py          FastMCP server entry point
│   ├── memcheck/          Standalone claim-validation + code-hallucination module
│   ├── pyproject.toml     Package definition (pip install -e .)
│   └── README.md          MCP setup and tool reference
├── docs/                  Architecture, theory, component reference, ops guide
├── scripts/               Python scripts (run standalone or via cron)
│   └── hooks/             Claude Code / agent hook adapters
├── eval/                  Longitudinal grounding quality evaluation harness
├── a2a_server/            A2A RAG broadcast server (mesh-wide context sharing)
├── rules/                 Agent behavioral rules (loaded at session start)
├── cron/jobs.json         (reference copy — live file in ~/.hermes/cron/)
└── .env.example           Full environment variable reference for all components
```

## Infrastructure

| Resource | Default | Env var(s) |
|---|---|---|
| Qdrant | `http://localhost:6333` | `QDRANT_URL`, `QDRANT_API_KEY` |
| Ollama (MCP server) | `http://localhost:11434` | `OLLAMA_BASE_URL` |
| Ollama (A2A server) | `http://localhost:11434/v1` | `MNEMOSYNE_EMBEDDING_API_URL` |
| Embedding model (MCP) | `nomic-embed-text` | `EMBED_MODEL` |
| Embedding model (A2A) | `nomic-embed-text` | `MNEMOSYNE_EMBEDDING_MODEL` |
| Embedding dimensions | `768` | `MNEMOSYNE_EMBEDDING_DIM` |
| Mnemosyne DB | `~/.hermes/mnemosyne/data/mnemosyne.db` | `MNEMOSYNE_DATA_DIR` |
| Memory session dir (MCP) | `~/.hermes/memory-sessions` | `HERMES_MEMORY_DIR` |
| Qdrant collection prefix | `hermes_memory` | `QDRANT_COLLECTION_PREFIX` |
| Hook state | `~/.claude/hook-state/` | — |

The MCP server reads `OLLAMA_BASE_URL` for its embedding calls. The A2A server reads
`MNEMOSYNE_EMBEDDING_API_URL` (an OpenAI-compatible endpoint, typically
`OLLAMA_BASE_URL + /v1`). The two variables serve different components — set both
when running the full stack.

Two `.env.example` files are provided:

- **`.env.example`** (repo root) — complete reference covering all components (MCP
  server, A2A server, hooks, cron scripts).
- **`mcp/.env.example`** — minimal file for deployments that run only the MCP server.

Override any default with env vars — see [docs/OPERATIONS.md](docs/OPERATIONS.md).

## Quick start

```bash
git clone https://github.com/<your-org>/hermes_memory
cd hermes_memory/mcp
python3 -m venv .venv
# Base install (Qdrant-only mode):
.venv/bin/pip install -e "."
# With Mnemosyne SQLite substrate (optional):
.venv/bin/pip install -e ".[mnemosyne]"
cp .env.example .env   # fill in QDRANT_URL, QDRANT_API_KEY, OLLAMA_BASE_URL
.venv/bin/python server.py
```

See [mcp/README.md](mcp/README.md) for full tool reference and Claude Code wiring.
See [a2a_server/README.md](a2a_server/README.md) for the A2A mesh endpoint.

## MCP tools (26)

| Tool | Purpose |
|---|---|
| `investigation_start` | Open a new investigation session |
| `investigation_load` | Load an existing investigation by ID |
| `investigation_store` | Persist findings to the investigation |
| `investigation_note` | Append a free-form note |
| `investigation_reflect` | Run reflection over current findings |
| `investigation_search` | Search within an investigation |
| `investigation_pre_answer_check` | Validate a claim against stored evidence before answering |
| `investigation_evidence_precheck` | Pre-screen evidence before ingestion |
| `investigation_entity_lookup` | Look up an entity by name across stored findings |
| `investigation_related_cases` | Find related prior investigations |
| `investigation_finding_provenance` | Trace source provenance for a finding |
| `investigation_list` | List all investigations |
| `audit_log` | Append to the audit trail |
| `memory_self_check` | Cross-check stored memories for consistency |
| `code_memory_correlate` | Correlate a code change with stored memory context |
| `memory_health` | Report memory system health |
| `memory_retract` | Soft-retract an incorrect or stale memory |
| `memory_restore` | Restore a previously retracted memory |
| `reflection_loop_seed` | Seed the reflection loop with new material |
| `reflection_loop_tick` | Advance the reflection loop one step |
| `reflection_loop_status` | Report reflection loop queue status |
| `rag_context_search` | Fan-out semantic search across all configured Qdrant collections |
| `memory_consolidate` | Trigger memory consolidation (dedup + merge) |
| `dama_routing_query` | Query domain routing decisions from the configured collection |
| `memory_confidence` | Estimate confidence in a memory-derived claim before asserting it |

## A2A skills (14)

The A2A server exposes 14 skills via `_SKILL_MAP`. Thirteen are advertised in the
agent card; `memory_prime` is callable but not listed in the agent card discovery
response.

| Skill | Advertised in agent card |
|---|---|
| `memory_recall` | Yes |
| `memory_remember` | Yes |
| `memory_stats` | Yes |
| `session_search` | Yes |
| `memory_sleep` | Yes |
| `rag_search` | Yes |
| `context_broadcast` | Yes |
| `mnemosyne_triple_add` | Yes |
| `mnemosyne_triple_query` | Yes |
| `gpu_inference` | Yes |
| `docker_status` | Yes |
| `ua_search` | Yes |
| `dama_telemetry` | Yes |
| `memory_prime` | No |

## Key env vars

### MCP server

| Variable | Default | Purpose |
|---|---|---|
| `QDRANT_URL` | _(required)_ | Qdrant instance URL |
| `QDRANT_API_KEY` | `""` | Qdrant auth key (blank for no-auth) |
| `QDRANT_COLLECTION_PREFIX` | `hermes_memory` | Main Qdrant collection name |
| `OLLAMA_BASE_URL` | _(required for embeddings)_ | Ollama base URL (no trailing `/v1`) |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model for MCP server |
| `EMBED_API_KEY` | `""` | Cloud embedding provider API key |
| `EMBED_API_KEY_HEADER` | `Authorization` | Auth header name for cloud embeddings |
| `HERMES_MEMORY_DIR` | `~/.hermes/memory-sessions` | Investigation session storage root |
| `MNEMOSYNE_EMBEDDING_DIM` | `768` | Vector dimension — must match your model |
| `CODE_CHUNKS_COLLECTION` | _(unset)_ | Qdrant collection for `code_memory_correlate` |
| `ROUTING_DECISIONS_COLLECTION` | _(unset)_ | Qdrant collection for `dama_routing_query` |
| `HERMES_MCP_TRANSPORT` | `stdio` | Transport mode: `stdio`, `sse`, or `streamable-http` |
| `HERMES_MCP_HOST` | `0.0.0.0` | Bind host for SSE/HTTP transport |
| `HERMES_MCP_PORT` | `8000` | Bind port for SSE/HTTP transport |

### A2A server

| Variable | Default | Purpose |
|---|---|---|
| `MNEMOSYNE_EMBEDDING_API_URL` | _(required)_ | OpenAI-compat embedding endpoint (e.g. `http://localhost:11434/v1`) |
| `MNEMOSYNE_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model for A2A server |
| `MNEMOSYNE_DATA_DIR` | `~/.hermes/mnemosyne/data` | Mnemosyne SQLite data directory |
| `HERMES_A2A_TOKEN` | _(required)_ | Bearer token callers must present |
| `HERMES_A2A_TOTP_SEED` | `""` | TOTP base32 seed (blank to disable) |
| `HERMES_A2A_URL` | `http://127.0.0.1:8201` | Public URL injected into the agent card |
| `HERMES_AGENT_ID` | `hermes-agent` | Agent identity stamped on all writes |
| `EXTRA_RAG_COLLECTIONS` | `""` | Comma-separated extra Qdrant collections for fan-out RAG |
| `PEER_A2A_URLS` | `""` | Comma-separated peer A2A endpoints for context broadcast |
| `DAMA_TELEMETRY_COLLECTION` | `""` | Qdrant collection for `dama_telemetry` skill |

## Qdrant collections

| Collection | Purpose |
|---|---|
| `hermes_memory` (configurable via `QDRANT_COLLECTION_PREFIX`) | Primary long-term memory store |
| `hermes_sessions` | Session history embeddings |
| `hermes_verdicts` | Claim verdict history for `investigation_pre_answer_check` and `memory_self_check` |
| `mnemosyne` | Synced Mnemosyne SQLite vectors |

## MCP transport modes

By default the MCP server runs over `stdio` for use as a Claude Code subprocess.
For Docker or remote deployments set `HERMES_MCP_TRANSPORT=sse` (or
`streamable-http`) and configure `HERMES_MCP_HOST` / `HERMES_MCP_PORT`.

```bash
HERMES_MCP_TRANSPORT=sse HERMES_MCP_HOST=0.0.0.0 HERMES_MCP_PORT=8000 \
  .venv/bin/python server.py
```
