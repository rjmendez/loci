# Running Loci

_Last reviewed: 2026-09-17 — rolling document. Setup, backends, and gotchas drift with the environment; update this page whenever a step or default changes._

This page is for standing up and operating Loci locally: the MCP server, backends and local models, infrastructure, testing, conventions, and the gotchas that bite people.

## Quick start (local development)

```bash
cd /home/rjmendez/development/loci/mcp
python3 -m venv .venv
.venv/bin/pip install -e .
cp .env.example .env        # fill QDRANT_URL and OLLAMA_BASE_URL at minimum
.venv/bin/python server.py
```

**Claude Code integration is automatic in-checkout.** The repo checks in `.mcp.json` at the root with relative paths, so launching Claude Code inside this checkout auto-registers the `loci` server — no per-machine setup.

**From outside the checkout:** register the server in `~/.claude/settings.json` with absolute paths pointing at `.venv/bin/python` and `mcp/server.py`.

## Backends and local models

- **Config file:** `~/.loci/backends.toml` (gitignored, machine-specific)
- **Template:** `backends.toml.example` at the repo root
- **Resolution chain, per endpoint:** environment variable → localhost probe → `~/.loci/backends.toml` → code default

Key `[ollama]` endpoints:

| Key | Role |
|-----|------|
| `url` | Embedding endpoint (`nomic-embed-text`, 768-dim). Probes `http://localhost:11434` by default. |
| `gen_url` | Generation endpoint — resolved **separately** from embeddings. Often a different host (shared cluster embeds, local GPU generates). |
| `gen_model` | Generation model tag (e.g. `qwen2.5:3b`). Must be a tag `ollama list` actually shows. |
| `verify_model` | Optional adversarial model for claim validation; falls back to `gen_model`. |
| `compress_model` | Optional summarization model; falls back to `gen_model`. |
| `guardian_model` | Prompt-injection classifier; defaults to `granite3-guardian:2b`. |
| `redteam_model` | Optional heretic/abliterated model for adversarial red-team analysis. |

### Embedding API options

- **Ollama (default):** `http://localhost:11434`, model `nomic-embed-text`, 768-dim.
- **OpenAI:** set `EMBED_API_KEY`, point at the OpenAI URL, model `text-embedding-3-small` (1536-dim → also set `MNEMOSYNE_EMBEDDING_DIM=1536`).
- **Azure OpenAI:** as OpenAI, but set `EMBED_API_KEY_HEADER=api-key`.
- **vLLM:** optional fallback from Ollama generation (`LOCI_VLLM_FALLBACK=1`, `LOCI_VLLM_URL`).

## Infrastructure (Qdrant)

Qdrant is the required vector store — local Docker or Qdrant Cloud. Collections are auto-created on first touch and get INT8 quantization + HNSW config:

- `loci_memory` — findings (dense 768-dim cosine + sparse BM25/IDF)
- `loci_sessions` — session-history embeddings
- `loci_verdicts` — claim-validation history
- `mnemosyne` — synced Mnemosyne SQLite vectors

Retention is disabled by default (`LOCI_QDRANT_RETENTION_DAYS=0`). **A non-zero value deletes findings at process startup** — keep it at 0 unless you truly want a retention window.

## A2A server (optional, cross-agent memory)

- Runs on port **8201**, exposes 13 JSON-RPC 2.0 skills.
- Requires `LOCI_A2A_TOKEN` (bearer). Generate one: `python3 -c "import secrets;print(secrets.token_hex(32))"`.
- Reads `MNEMOSYNE_DATA_DIR` — it **does not create the schema**; point it at a Mnemosyne-created DB.
- Broadcasts to peers via `PEER_A2A_URLS` (comma-separated).
- systemd user-service template: `a2a_server/loci-a2a.service`.

## MCP transport modes

- Default: `stdio` (Claude Code subprocess).
- Production: `LOCI_MCP_TRANSPORT=sse` or `streamable-http`.
- Loopback bind (127.0.0.1): no auth required.
- Wide bind (0.0.0.0): requires `LOCI_MCP_TOKEN`. `/health` stays open for liveness; every other route needs `Authorization: Bearer <token>`. Binding wide with no token is a hard `SystemExit` — the server refuses to start.

## Deployment options

- **Docker Compose (prod):** `docker compose up -d` (bring your own Qdrant/Ollama).
- **Dev with sidecars:** `docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d` (Qdrant + Ollama sidecars).
- **Bare-metal venv:** the quick-start above.
- **systemd user service:** for the A2A server.

## Testing & verification

- Tests live in `tests/` (MCP unit tests) and `a2a_server/tests/`.
- Run all: `pytest tests/ a2a_server/tests/ -v`.
- Schema consistency (validates Mnemosyne column names): `python3 tests/test_schema_consistency.py`.
- Grounding-gate eval (RAG quality): `eval/run_eval.sh`.
- Verify-skeptic eval (claim-validation false-refutation rate): `mcp/.venv/bin/python eval/verify_skeptic_eval.py [trials] [votes]`.

**Caveat:** the `verify` groom pass has a ~22% false-refutation rate — do not schedule it.

### Health checks

- MCP HTTP routes: `curl http://localhost:8000/health` → `{"status": "ok"}`.
- A2A server: `curl http://localhost:8201/health` → `{"mnemosyne_db_found": true, ...}` (checks file existence only).
- Memory system: the `loci_health` MCP tool (checks Qdrant, embeddings, code graph, Mnemosyne).

## Repository conventions

### Environment variable fallback

- The MCP server loads repo-root `.env`, then `mcp/.env` (override), at import time.
- Standalone scripts use `scripts/loci_groom.py:load_env()` for cron portability.
- **Two Ollama variables, on purpose:** `OLLAMA_BASE_URL` (embeddings, MCP) and `OLLAMA_URL` (standalone scripts); generation uses `OLLAMA_GEN_URL` / `LOCI_OLLAMA_GEN_URL`. `OLLAMA_BASE_URL` does **not** feed generation.

### Storage locations

- Findings: `~/.hermes/memory-sessions/<investigation_id>/` (JSONL, audit logs, `manifest.json`).
- Code graph: `$LOCI_MEMORY_DIR/graph.ladybug` (Ladybug tree-sitter index).
- Mnemosyne DB: `~/.hermes/mnemosyne/data/mnemosyne.db` (SQLite, optional).
- Hook state: `~/.claude/hook-state/` (grounding cache, skill annotations).

### Grooming tier (`scripts/loci_groom.py`)

Eight passes: `index`, `tags`, `recall`, `knn_tags`, `codelink`, `verify`, `reflect`, `summaries`. Only `index` is destructive/applier; the rest write proposals to `$LOCI_MEMORY_DIR/_groom/`. Scheduled cron:

- `index --apply` — upsert disk findings missing from Qdrant (every 6h)
- `knn_tags` — propose tags from semantic neighbors (daily 3am)
- `codelink` — propose symbol→finding links (daily 3:40am)
- `summaries` — fill the investigation summary ladder (daily 4:50am)

Grooming **refuses to run** if `LOCI_QDRANT_RETENTION_DAYS` is non-zero (it would create an index-then-delete loop).

### Cron infrastructure

- Jobs defined in `cron/jobs.json` (5 enabled, 1 disabled: `dtl_harvest`), driven by `scripts/hermes_cron_runner.py` (1-minute user timer).
- The grooming tier is separate, driven by `scripts/loci_groom_cron.sh` (user crontab).
- The MLOps loop rebuilds the grounding dataset, trains an ensemble, and canary-evaluates — it runs in an isolated worktree to avoid a dirty checkout.

### Claude Code hooks (opt-in: `scripts/hooks/install.sh`)

- `pre_llm_grounding.py` — injects per-turn Qdrant results (events: `UserPromptSubmit`, `SubagentStart`).
- `pre_tool_grounding.py` — tool-call audit + injection scanning (event: `PreToolUse`).
- `session_end_sync.py` — syncs the session transcript to `loci_sessions` (event: `Stop`).
- Drift check: `scripts/hooks/install.sh --check` (exits 1 on drift — deployed and repo copies have diverged before).

## Gotchas & known issues

1. **Qdrant named vectors are directional.** Upsert shape `{"dense": [...]}` fails *search* (HTTP 400); the search shape is `{"name": "dense", "vector": [...]}`. Sparse vectors mix in the opposite direction and require the IDF modifier.
2. **Generation endpoint is resolved separately from embeddings.** `OLLAMA_BASE_URL` does not feed generation; set `LOCI_OLLAMA_GEN_URL` / `OLLAMA_GEN_URL` independently.
3. **Unpulled/misspelled Ollama tags fail silently.** Every `generate()` returns `degraded=True` with the reason only in the `why` field.
4. **Ebbinghaus consolidation needs retention disabled.** `LOCI_QDRANT_RETENTION_DAYS=0` is the safe default; non-zero makes the first Qdrant call of every process delete findings.
5. **A2A does not create the Mnemosyne schema.** `/health` only checks file existence — point `MNEMOSYNE_DATA_DIR` at a Mnemosyne-created DB.
6. **Hooks drift silently.** Run `install.sh --check`.
7. **MCP HTTP on a non-loopback bind without a token is a `SystemExit`** — no fallback.
8. **Embedding dimension must match the model.** Moving 768-dim → 1536-dim (OpenAI) requires wiping and re-indexing Qdrant.
9. **Windows + WSL2 NAT networking.** WSL cannot reach a Windows-bound Ollama via `127.0.0.1` or the gateway IP. Use the Windows host's Tailscale IP, and verify with `curl http://<ip>:11434/api/tags` before trusting an IP guess.

## Manual operations cheat sheet

```bash
LOCI=~/development/loci
LOCI_PY=~/.hermes/hermes-agent/venv/bin/python3   # or any interpreter with deps

# Rebuild MemGAS 3-level hierarchy / search it
$LOCI_PY $LOCI/scripts/memgas_hierarchy.py --index
$LOCI_PY $LOCI/scripts/memgas_hierarchy.py --search "your query"

# Run a groom pass by hand
$LOCI/scripts/loci_groom_cron.sh index          # dry run
$LOCI/scripts/loci_groom_cron.sh index --apply  # apply

# Bench local generation
$LOCI_PY $LOCI/scripts/bench_local_models.py --model qwen2.5:3b --concurrency 1 4 8

# Check adversarial-skeptic accuracy
$LOCI/mcp/.venv/bin/python $LOCI/eval/verify_skeptic_eval.py 20 5

# Force Mnemosyne → Qdrant sync
QDRANT_API_KEY=... $LOCI_PY $LOCI/scripts/mnemosyne_qdrant_sync.py
```

## Repo structure quick reference

```
loci/
├── mcp/                   — MCP server (investigation, RAG, claim validation, code graph)
│   ├── server.py          — FastMCP entry point
│   ├── backends.py        — endpoint resolution: env → probe → ~/.loci/backends.toml
│   ├── memcheck/          — claim validation + code-hallucination detection
│   └── pyproject.toml     — deps (mcp[cli], qdrant-client, fastembed, sentence-transformers, ...)
├── a2a_server/            — A2A RAG broadcast (cross-agent memory sharing)
├── docs/                  — CONCEPTS.md (start here), ARCHITECTURE.md, CALLGRAPH.md, OPERATIONS.md
├── scripts/               — standalone Python + shell (grooming, consolidation, hooks)
├── tests/                 — pytest (schema consistency, guard validation, ...)
├── eval/                  — grounding-gate + skeptic verification benchmarks
├── deep_think_loci/       — multi-tier reasoning engine (beta, workflow-based)
├── mlops/                 — embedding/routing/grounding experiments + loop.py
├── cron/jobs.json         — Hermes cron job definitions
├── .mcp.json              — Claude Code auto-registration (relative paths)
├── backends.toml.example  — template for ~/.loci/backends.toml (gitignored)
├── .env.example           — full env-var reference
├── docker-compose.yml     — production (bring your own Qdrant/Ollama)
└── docker-compose.dev.yml — dev (Qdrant + Ollama sidecars)
```
