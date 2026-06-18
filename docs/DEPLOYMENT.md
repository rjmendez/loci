# Deployment Guide

## Architecture

hermes_memory has two custom-built services and two external dependencies.

```
┌─────────────────────────────────────────┐
│           custom-built services         │
│                                         │
│  hermes-mcp  (port 8000, SSE/stdio)     │  ← investigation memory, RAG, claim check
│  hermes-a2a  (port 8201, HTTP)          │  ← cross-mesh context sharing            
└──────────────┬──────────────────────────┘
               │ env vars (QDRANT_URL, OLLAMA_BASE_URL)
               ▼
┌──────────────────────────────────────────┐
│           external dependencies          │
│                                          │
│  Qdrant   — vector store                 │
│  Embedding API — /v1/embeddings compat   │
└──────────────────────────────────────────┘
```

The custom services are fully contained in this repo and portable.
The external services are swappable — see options below.

---

## External service requirements

### Qdrant

| Property | Requirement |
|---|---|
| API | REST (HTTP) or gRPC |
| Version | ≥ 1.7 (named-vector support required) |
| Collection format | 768-dim Cosine, named vectors `{"dense": [...]}` |
| Env var | `QDRANT_URL` (default: `http://localhost:6333`) |
| Auth | `QDRANT_API_KEY` (optional — leave blank if no auth) |

**Options:**
- [Qdrant OSS](https://qdrant.tech/documentation/quick-start/) — self-hosted Docker or binary
- [Qdrant Cloud](https://cloud.qdrant.io/) — managed, free tier available
- Any Qdrant-compatible endpoint

Collections are created automatically on first run. If you change `QDRANT_COLLECTION_PREFIX`, new collections are created — existing data in the old names is unaffected.

### Embedding API

The embedding model defines the vector space. All data must be embedded with the **same model** — mixing models across writes/queries corrupts search results.

| Property | Requirement |
|---|---|
| API | OpenAI-compatible `/v1/embeddings` endpoint |
| Default model | `nomic-embed-text` (768-dim) |
| Env vars | `OLLAMA_BASE_URL`, `EMBED_MODEL` |

**Options:**

| Provider | Model | Dimension | Notes |
|---|---|---|---|
| [Ollama](https://ollama.com/) (default) | `nomic-embed-text` | 768 | Self-hosted; free; CPU or GPU |
| [OpenAI](https://platform.openai.com/docs/api-reference/embeddings) | `text-embedding-3-small` | 1536 | Set `MNEMOSYNE_EMBEDDING_DIM=1536`; different vector space from default |
| [Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/) | `text-embedding-ada-002` | 1536 | Set `MNEMOSYNE_EMBEDDING_DIM=1536`; set `EMBED_API_KEY_HEADER=api-key` |
| [Cohere](https://docs.cohere.com/reference/embed) | `embed-english-v3.0` | 1024 | Requires adapter — not directly OpenAI-compat |
| [Voyage AI](https://docs.voyageai.com/) | `voyage-3-lite` | 512 | OpenAI-compat endpoint available |

**Switching embedding models:** collections must be rebuilt from scratch. Wipe the Qdrant collections and re-index. The vector dimension is not backward-compatible.

`MNEMOSYNE_EMBEDDING_DIM` controls the Qdrant collection vector size (default: `768`). Set it to match your model's output dimension **before** first run. Changing it on an existing collection requires wiping and re-indexing.

For cloud embedding providers that require authentication, set `EMBED_API_KEY` to your API key. The `EMBED_API_KEY_HEADER` variable controls which HTTP header carries the key (default: `Authorization`, which sends `Bearer <key>`; set to `api-key` for Azure OpenAI).

---

## Running the custom services

### Option A — Docker Compose (recommended)

**Production (bring your own Qdrant + embedding API):**

```bash
cp .env.example .env
# Edit .env: set QDRANT_URL, QDRANT_API_KEY, OLLAMA_BASE_URL, HERMES_A2A_TOKEN
docker compose up -d
```

`HERMES_A2A_TOKEN` is the bearer token required by the A2A service. Generate one with:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

**Local development (with Qdrant + Ollama sidecars):**

```bash
cp .env.example .env
# Set HERMES_A2A_TOKEN in .env (the only required secret)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# First run only: pull the embedding model.
# The container name is derived from the project directory name.
# If cloned as hermes_memory (default), the container is hermes_memory-ollama-1.
docker exec -it hermes_memory-ollama-1 ollama pull nomic-embed-text
```

### Option B — Bare metal / venv

**MCP server (Claude Code subprocess, stdio):**
```bash
cd mcp/
python3 -m venv .venv && .venv/bin/pip install -e .
cp .env.example .env   # edit
.venv/bin/python server.py   # HERMES_MCP_TRANSPORT defaults to stdio
```

Configure Claude Code (`~/.claude/settings.json`):
```json
{
  "mcpServers": {
    "hermes_memory": {
      "command": "/path/to/mcp/.venv/bin/python",
      "args": ["/path/to/mcp/server.py"],
      "env": {
        "QDRANT_URL": "http://your-qdrant:6333",
        "OLLAMA_BASE_URL": "http://your-ollama:11434"
      }
    }
  }
}
```

**MCP server (HTTP/SSE, for remote or containerized access):**
```bash
HERMES_MCP_TRANSPORT=sse HERMES_MCP_PORT=8000 .venv/bin/python server.py
```

Configure Claude Code to use HTTP transport:
```json
{
  "mcpServers": {
    "hermes_memory": {
      "transport": "sse",
      "url": "http://your-host:8000/sse"
    }
  }
}
```

**A2A server:**
```bash
cd a2a_server/
pip install -e .
export HERMES_A2A_TOKEN=$(python3 -c "import secrets;print(secrets.token_hex(32))")
python server.py
```

The A2A server reads `HERMES_A2A_TOKEN`, `HERMES_A2A_HOST`, `HERMES_A2A_PORT`, and `HERMES_A2A_URL` from the environment.

### Option C — systemd (user service)

```bash
# Edit a2a_server/loci-a2a.service — adjust WorkingDirectory and ExecStart paths
cp a2a_server/loci-a2a.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now loci-a2a
journalctl --user -u loci-a2a -f
```

---

## Persistent data

| Service | What's stored | Volume path | Env var |
|---|---|---|---|
| hermes-mcp | JSONL session fallback files | `/data/memory-sessions` | `HERMES_MEMORY_DIR` |
| hermes-mcp | fastembed ONNX model cache | `/data/fastembed-cache` | `FASTEMBED_CACHE_PATH` |
| hermes-a2a | Mnemosyne SQLite (memory recall/write) | `/data/mnemosyne` | `MNEMOSYNE_DATA_DIR` |

Qdrant data persists in your external Qdrant instance and is not managed by these services.

---

## First-run initialization

On first startup the MCP server creates two Qdrant collections:
- `hermes_memory` — investigation findings (30-day TTL sweep at startup)
- `hermes_verdicts` — claim validation history

The A2A server creates the Mnemosyne SQLite schema on first connect if the DB file does not exist.

To verify both services are healthy:
```bash
# MCP server (HTTP mode)
curl http://localhost:8000/health

# A2A server
curl http://localhost:8201/health
```

---

## Environment variable reference

See `.env.example` (root) and `mcp/.env.example` for the full variable list with descriptions.

### Required variables

| Variable | Service | Purpose |
|---|---|---|
| `HERMES_A2A_TOKEN` | hermes-a2a | Bearer token read directly by `server.py`; server exits at startup if unset |

Set `HERMES_A2A_TOKEN` in your `.env` file (or export it in the environment) regardless of deployment method.

### Variables with localhost defaults (override for production)

| Variable | Default | Used by | Purpose |
|---|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | both | Qdrant REST endpoint |
| `QDRANT_API_KEY` | _(empty)_ | both | Qdrant auth key; omit for unauthenticated instances |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | hermes-mcp | Base URL of the embedding/LLM API |
| `MNEMOSYNE_EMBEDDING_API_URL` | `http://localhost:11434/v1` | hermes-a2a | OpenAI-compat embedding endpoint for Mnemosyne |
| `EMBED_MODEL` | `nomic-embed-text` | both | Embedding model name |
| `MNEMOSYNE_EMBEDDING_MODEL` | `nomic-embed-text` | hermes-a2a | Embedding model for Mnemosyne |
| `MNEMOSYNE_EMBEDDING_DIM` | `768` | hermes-mcp | Qdrant vector dimension — must match the model's output |
| `HERMES_A2A_URL` | `http://127.0.0.1:8201` | hermes-a2a | Public base URL injected into the A2A agent card |
| `HERMES_A2A_HOST` | `0.0.0.0` | hermes-a2a | Bind address |
| `HERMES_A2A_PORT` | `8201` | hermes-a2a | Bind port |

### Authentication variables

| Variable | Default | Purpose |
|---|---|---|
| `HERMES_A2A_TOTP_SEED` | _(empty)_ | Base32 TOTP seed for second-factor auth; leave blank to disable |
| `EMBED_API_KEY` | _(empty)_ | API key for cloud embedding providers (OpenAI, Azure, Bedrock) |
| `EMBED_API_KEY_HEADER` | `Authorization` | HTTP header that carries `EMBED_API_KEY`; set to `api-key` for Azure OpenAI |

### Optional collection and namespace variables

| Variable | Default | Purpose |
|---|---|---|
| `LOCI_NAMESPACE` | _(empty)_ | Namespace tag stamped on all Qdrant writes; use to partition a shared instance |
| `EXTRA_RAG_COLLECTIONS` | _(empty)_ | Comma-separated Qdrant collection names included in fan-out RAG search |
| `GROUNDING_EXTRA_COLLECTIONS` | _(empty)_ | Same as above, scoped to the grounding hook only; defaults to `EXTRA_RAG_COLLECTIONS` |
| `CODE_CHUNKS_COLLECTION` | _(empty)_ | Collection of code-chunk embeddings for the `code_memory_correlate` tool |
| `ROUTING_DECISIONS_COLLECTION` | _(empty)_ | Collection of routing/decision records for the routing query tool |
