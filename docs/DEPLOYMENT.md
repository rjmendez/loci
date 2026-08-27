# Deployment Guide

## Architecture

Loci has two custom-built services and two external dependencies. The service,
image and volume names are still `hermes-mcp` / `hermes-a2a`.

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
| Version | `docker-compose.dev.yml` pins `qdrant/qdrant:v1.13.6`; client pin is `qdrant-client>=1.17.0,<1.19.0`. Collection creation asks for sparse named vectors with the IDF modifier and INT8 scalar quantization, and no lower server version has been tested |
| Collection format | dense named vector `dense`, `MNEMOSYNE_EMBEDDING_DIM`-dim (768) Cosine, plus a sparse named vector `sparse` (IDF), HNSW `m=32`/`ef_construct=200`, INT8 scalar quantization at quantile 0.99, `always_ram` |
| Env var | `QDRANT_URL` (no code default — unset disables Qdrant; the Docker image sets `http://localhost:6333`) |
| Auth | `QDRANT_API_KEY` (optional — leave blank if no auth) |

The two named-vector wire shapes are not interchangeable, and mixing them fails
in opposite directions. **Upsert** takes `{"dense": [...]}`. **Search** takes
`{"name": "dense", "vector": [...]}` — the upsert shape returns HTTP 400 there
(#228).

**Options:**
- [Qdrant OSS](https://qdrant.tech/documentation/quick-start/) — self-hosted Docker or binary
- [Qdrant Cloud](https://cloud.qdrant.io/) — managed, free tier available
- Any Qdrant-compatible endpoint

The findings collection is created automatically on first use. Despite the name, `QDRANT_COLLECTION_PREFIX` is the *whole* collection name, not a prefix — all findings live in one collection and `investigation_id` is a payload field. Changing it creates a new collection; data under the old name is untouched.

### Embedding API

The embedding model defines the vector space. All data must be embedded with the **same model** — mixing models across writes/queries corrupts search results.

| Property | Requirement |
|---|---|
| API | OpenAI-compatible `/v1/embeddings` endpoint |
| Default model | `nomic-embed-text` (768-dim) |
| Env vars | `OLLAMA_BASE_URL`, `EMBED_MODEL` |

`OLLAMA_BASE_URL` is required for **every** provider, cloud ones included:
`qdrant_ops._embed` returns `None` when it is unset, whatever `EMBED_API_KEY`
says. Point it at the provider's base URL and set `EMBED_API_KEY` alongside.
With no embedding endpoint the server does not fall back to another embedder — it
drops to keyword-only recall, `_qdrant_upsert` returns without writing, and
`memory_health` reports the dense probe as `fail`.

Generation resolves separately from embeddings and does **not** read
`OLLAMA_BASE_URL` (#222/#224/#225). Set `LOCI_OLLAMA_GEN_URL` (or
`OLLAMA_GEN_URL`), else `mcp/backends.py:ollama_gen_url()` resolves it. The vLLM
fallback is opt-in: `LOCI_VLLM_FALLBACK=1`, default off.

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
# Edit .env: set QDRANT_URL, QDRANT_API_KEY, OLLAMA_BASE_URL, LOCI_A2A_TOKEN
docker compose up -d
```

`LOCI_A2A_TOKEN` is the bearer token required by the A2A service. Generate one with:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

**Local development (with Qdrant + Ollama sidecars):**

```bash
cp .env.example .env
# Set LOCI_A2A_TOKEN in .env (the only required secret)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# First run only: pull the embedding model. The container name is
# <compose project>-ollama-1, and the project name is the directory you cloned
# into — `loci-ollama-1` for a default clone. `docker compose ps` prints it.
docker exec -it "$(docker compose ps -q ollama | head -1)" ollama pull nomic-embed-text
```

Both compose files publish on `127.0.0.1` only. `hermes-mcp` sets
`LOCI_MCP_HOST=0.0.0.0` on purpose — a container has to bind all interfaces to
receive published traffic. Do not "fix" that to `127.0.0.1`: it binds the
server to the container's own loopback, the port mapping stops working, and the
in-container healthcheck keeps passing, so the service reports healthy while
dead.

### Option B — Bare metal / venv

**MCP server (Claude Code subprocess, stdio):**
```bash
cd mcp/
python3 -m venv .venv && .venv/bin/pip install -e .
cp .env.example .env   # edit
.venv/bin/python server.py   # LOCI_MCP_TRANSPORT defaults to stdio
```

Configure Claude Code (`~/.claude/settings.json`):
```json
{
  "mcpServers": {
    "loci": {
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
LOCI_MCP_TRANSPORT=sse LOCI_MCP_PORT=8000 .venv/bin/python server.py
```

`LOCI_MCP_HOST` defaults to `127.0.0.1` (#206). This server exposes the whole
tool surface, so a wide bind has to be a decision you made rather than one you
got by not making it. Two rules follow (#211):

- Binding anything other than `127.0.0.1` / `::1` / `localhost` **without**
  `LOCI_MCP_TOKEN` is refused at startup with a `SystemExit`, not served.
- With `LOCI_MCP_TOKEN` set, every HTTP request needs
  `Authorization: Bearer <token>`; `/health` stays open for liveness probes and
  returns a fixed `{"status": "ok"}`. Comparison is `hmac.compare_digest`.

```bash
export LOCI_MCP_TOKEN=$(python3 -c "import secrets;print(secrets.token_hex(32))")
LOCI_MCP_TRANSPORT=sse LOCI_MCP_HOST=0.0.0.0 .venv/bin/python server.py
```

This is a shared secret, not OAuth — the server publishes no
`/.well-known/oauth-*` discovery, because there is no authorization server to
describe.

Configure Claude Code to use HTTP transport:
```json
{
  "mcpServers": {
    "loci": {
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
export LOCI_A2A_TOKEN=$(python3 -c "import secrets;print(secrets.token_hex(32))")
python server.py
```

The A2A server reads `LOCI_A2A_TOKEN`, `LOCI_A2A_HOST`, `LOCI_A2A_PORT`, and `LOCI_A2A_URL` from the environment.

### Option C — systemd (user service)

```bash
# Edit a2a_server/loci-a2a.service — adjust WorkingDirectory and ExecStart paths
cp a2a_server/loci-a2a.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now loci-a2a
journalctl --user -u loci-a2a -f
```

`a2a_server/loci-a2a.service` is a template: its paths are `%h`-relative, so it
is user-scope only, and its `WorkingDirectory` / `ExecStart` assume
`~/.hermes/loci`. `scripts/systemd/` additionally ships the context-bridge
service and timer units.

---

## Persistent data

| Service | What's stored | Volume path | Env var |
|---|---|---|---|
| hermes-mcp | JSONL session fallback files | `/data/memory-sessions` | `LOCI_MEMORY_DIR` |
| hermes-mcp | fastembed ONNX model cache | `/data/fastembed-cache` | `FASTEMBED_CACHE_PATH` |
| hermes-a2a | Mnemosyne SQLite (memory recall/write) | `/data/mnemosyne` | `MNEMOSYNE_DATA_DIR` |

Qdrant data persists in your external Qdrant instance and is not managed by these services.

---

## First-run initialization

The MCP server creates its Qdrant collections lazily, not at process start:
- `loci_memory` (or whatever `QDRANT_COLLECTION_PREFIX` is set to) — investigation findings; created on the first Qdrant-touching tool call, along with its payload indexes. If the collection already exists, that same call issues an idempotent `update_collection` to apply the HNSW and INT8 quantization config.
- `loci_verdicts` — claim validation history. Created by `mcp/memcheck/cli.py` alone, as a 384-dim Cosine `dense` vector filled by a deterministic `hash_embed` (no model, no network). `mcp/verdict_ops.py` — the path `memory_self_check(record=True)` and `investigation_pre_answer_check` go through — writes to the collection but never creates it, and embeds with the server's 768-dim `_embed`. Until the collection exists `memory_health` reports it as "not yet created", which is benign.

That first call also runs the retention purge, and **the purge is disabled by
default**: `LOCI_QDRANT_RETENTION_DAYS` resolves to `0` (#204). Any positive
value makes the first Qdrant call of *every* process delete findings past the
window, silently. Under the previous default of 30 that was measurable on the
live store — 912 findings indexed against 2,831 on disk, the split falling
exactly on the 30-day boundary, re-indexing restoring coverage and the next
server start removing it again.

Resolution order is environment → `[qdrant].retention_days` in
`~/.loci/backends.toml` → `0`. A non-integer value disables the purge rather than
guessing a window. If you do set it, note that `scripts/loci_groom.py` refuses to
run and exits 3 while it is non-zero: re-indexing findings a purge then deletes
is a loop that burns embedding compute and reports success.

The A2A server does **not** create the Mnemosyne schema — there is no
`CREATE TABLE` in `a2a_server/`. It opens `MNEMOSYNE_DB` with `sqlite3.connect`,
which produces an empty file, and every memory query against that file then
fails. Point `MNEMOSYNE_DATA_DIR` at a database Mnemosyne itself created.
`/health` only reports `mnemosyne_db_found`, which is a file-existence test and
says nothing about the schema.

To verify both services are healthy:
```bash
# MCP server — HTTP transports only; the stdio transport serves no routes
curl http://localhost:8000/health

# A2A server
curl http://localhost:8201/health
```

Both `/health` routes are unauthenticated by design.

---

## Environment variable reference

See `.env.example` (root) and `mcp/.env.example` for the full variable list with
descriptions, and `backends.toml.example` for the file-based alternative.

`mcp/backends.py` resolves each backend as: environment variable → local probe
(`localhost:11434` for Ollama, `:8000` for vLLM) → `~/.loci/backends.toml` (or
`$LOCI_CONFIG`) → safe default. The TOML file is the durable channel — it is
parsed with stdlib `tomllib`, so nothing about it depends on a launcher
remembering to export anything, or on `python-dotenv` being importable. Machine-
specific endpoints and keys belong there, not in this repo.

### Required variables

| Variable | Service | Purpose |
|---|---|---|
| `LOCI_A2A_TOKEN` | hermes-a2a | Bearer token read directly by `server.py` |
| `LOCI_MCP_TOKEN` | hermes-mcp | Bearer token for the HTTP transports. Required — startup refuses — when `LOCI_MCP_HOST` is not loopback. Unused by the stdio transport |

Set `LOCI_A2A_TOKEN` in your `.env` file (or export it in the environment) regardless of deployment method.

`server.py` does **not** exit when `LOCI_A2A_TOKEN` is unset: it prints a
warning and keeps serving, and every authenticated route then returns 401. The
failure is at request time, not startup. `docker-compose.yml` is the layer that
fails fast — it declares `${LOCI_A2A_TOKEN:?...}`, so `docker compose up`
aborts. A bare-metal or systemd launch gets the warning instead; grep the log for
it. `LOCI_MCP_TOKEN` is the opposite: a non-loopback bind without it is a hard
`SystemExit`.

### Variables with localhost defaults (override for production)

| Variable | Default | Used by | Purpose |
|---|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` (Docker/compose only) | both | Qdrant REST endpoint; **no code default** — unset silently disables Qdrant and drops to keyword-only recall |
| `QDRANT_API_KEY` | _(empty)_ | both | Qdrant auth key; omit for unauthenticated instances |
| `OLLAMA_BASE_URL` | `http://localhost:11434` (Docker/compose; `mcp/backends.py` also probes it locally) | hermes-mcp | Base URL of the embedding API. Unset and unprobeable means **no embeddings at all** — not a smaller embedder: `_embed` returns `None`, writes skip Qdrant, and recall falls back to keyword |
| `MNEMOSYNE_EMBEDDING_API_URL` | `http://localhost:11434/v1` | hermes-a2a | OpenAI-compat embedding endpoint for Mnemosyne |
| `EMBED_MODEL` | `nomic-embed-text` | both | Embedding model name |
| `MNEMOSYNE_EMBEDDING_MODEL` | `nomic-embed-text` | hermes-a2a | Embedding model for Mnemosyne |
| `MNEMOSYNE_EMBEDDING_DIM` | `768` | hermes-mcp | Qdrant vector dimension — must match the model's output |
| `LOCI_A2A_URL` | `http://127.0.0.1:8201` | hermes-a2a | Public base URL injected into the A2A agent card |
| `LOCI_A2A_HOST` | `0.0.0.0` | hermes-a2a | Bind address. Note the asymmetry with `LOCI_MCP_HOST`, which defaults to `127.0.0.1`; the A2A server binds wide by default and relies on its bearer token |
| `LOCI_A2A_PORT` | `8201` | hermes-a2a | Bind port |
| `LOCI_MCP_HOST` | `127.0.0.1` | hermes-mcp | Bind address for the HTTP transports. Non-loopback requires `LOCI_MCP_TOKEN` |
| `LOCI_MCP_PORT` | `8000` | hermes-mcp | Bind port for the HTTP transports |
| `LOCI_MCP_TRANSPORT` | `stdio` | hermes-mcp | `stdio`, `sse`, or `streamable-http` |

### Authentication variables

| Variable | Default | Purpose |
|---|---|---|
| `LOCI_A2A_TOTP_SEED` | _(empty)_ | Base32 TOTP seed for second-factor auth; leave blank to disable |
| `EMBED_API_KEY` | _(empty)_ | API key for cloud embedding providers (OpenAI, Azure, Bedrock) |
| `EMBED_API_KEY_HEADER` | `Authorization` | HTTP header that carries `EMBED_API_KEY`; set to `api-key` for Azure OpenAI |

### Optional collection and namespace variables

| Variable | Default | Purpose |
|---|---|---|
| `LOCI_NAMESPACE` | _(empty)_ | Namespace tag stamped on all Qdrant writes; use to partition a shared instance |
| `EXTRA_RAG_COLLECTIONS` | _(empty)_ | Comma-separated Qdrant collection names included in fan-out RAG search |
| `GROUNDING_EXTRA_COLLECTIONS` | _(empty)_ | Extra Qdrant collections for the grounding hook only. Read independently — `scripts/hooks/pre_llm_grounding.py` has no fallback to `EXTRA_RAG_COLLECTIONS`, whatever `.env.example` implies. Set both if you want the same collections in both paths. The hook's base set is `mnemosyne`, `loci_sessions`, `loci_memory`, and it knows the content-field mapping for `ecc_skills`, `agent_core_chunks`, `dama_gotchi_code`; anything else is assumed to be `{"text": ..., named vector}` |
| `CODE_CHUNKS_COLLECTION` | _(empty)_ | Collection of code-chunk embeddings for the `code_memory_correlate` tool |
| `ROUTING_DECISIONS_COLLECTION` | _(empty)_ | **Unused** — appears only in `.env.example` and `mcp/.env.example`, read by no code; `memory_route` searches `QDRANT_COLLECTION_PREFIX` |
| `LOCI_QDRANT_RETENTION_DAYS` | `0` (purge disabled) | Days after which the first Qdrant call of a process **deletes** findings. See First-run initialization |
| `LOCI_CONFIG` | `~/.loci/backends.toml` | Path to the TOML backend config |
