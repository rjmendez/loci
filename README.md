# Loci — Persistent Memory for AI Agents

**Loci gives AI agents like Claude a long-term memory that survives across sessions.**

Without it, every conversation starts from zero: no history, accumulated knowledge, or
memory of what you tried and why. With Loci, sessions build on each other. Decisions,
findings, and context persist in a searchable memory store Claude can read and write like notes.

![The memory problem Loci solves](docs/img/loci-problem.svg)

---

## What it unlocks

- **Continuity** — Session 7 knows what happened in Sessions 1–6, without you re-explaining
- **Grounded answers** — Claude checks stored history before answering, reducing hallucinations
- **Claim validation** — Verify a proposed answer against stored evidence before asserting it
- **Accumulated knowledge** — Your project context grows richer with every session
- **Multi-agent memory** — Multiple AI agents can share a common memory pool via the A2A server
- **Supply chain security** — An opt-in Claude Code hook scans file-writing tool calls
  for prompt-injection and supply-chain IOC patterns

---

## How it works

![Loci architecture](docs/img/loci-overview.svg)

Loci runs as an MCP server alongside Claude Code. When Claude needs to remember or recall
something, it calls one of Loci's 76 tools — the same way it calls any other tool. Your
data stays on your own infrastructure: Qdrant and Ollama run locally or on your own server.

> **New to terms like "vector search", "RAG", or "MCP"?**
> Start with **[docs/CONCEPTS.md](docs/CONCEPTS.md)** — a plain-English guide that explains
> everything from scratch with no assumed knowledge.

---

## Why FlyBrain matters to Loci

FlyBrain is the practical test for one of Loci's core promises: when evidence comes from different datasets, stages, or interpretations, the system should keep each claim scoped instead of flattening it into a vague "fact."

Start with the short user-facing path in [docs/FLYBRAIN_GUIDE.md](docs/FLYBRAIN_GUIDE.md). Then use [docs/FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md](docs/FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md), [docs/FLYBRAIN_DATASET_PROVENANCE_MATRIX.md](docs/FLYBRAIN_DATASET_PROVENANCE_MATRIX.md), and [docs/FLYBRAIN_IO_TO_LOCI_MAPPING.md](docs/FLYBRAIN_IO_TO_LOCI_MAPPING.md) as the canonical follow-ups. Keep [docs/FLYBRAIN_REASONING_GLOSSARY.md](docs/FLYBRAIN_REASONING_GLOSSARY.md) open for quick term lookup, and use [docs/FLYBRAIN_ARCHIVE.md](docs/FLYBRAIN_ARCHIVE.md) only when you need the deeper historical or technical notes.

For filesystem safety in FlyBrain harness jobs, enforce the allowlist-only write boundary documented in [docs/FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md](docs/FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md).

The current VFB/FlyBrain tool surface supports this workflow in practice: `search_terms` finds the relevant neuron or anatomy, `get_term_info` and `get_hierarchy` confirm scope, `query_connectivity` and `run_query` compare evidence, and the provenance docs capture the dataset and version trail. If you are comparing results across datasets, the guide is the right starting point.

Important caveat: FlyBrain results remain dataset-, version-, and query-scoped. Empty results, `count_status` warnings, or a stale cached response are not evidence of a biological absence. When a claim depends on VFB data, record the dataset symbols, version label, and query settings, and use `force_refresh` or explicit scope notes when the underlying tool output is ambiguous.


## Quick start

![Three-step setup](docs/img/loci-quickstart.svg)

```bash
git clone https://github.com/rjmendez/loci
cd loci/mcp
python3 -m venv .venv && .venv/bin/pip install -e "."
cp ../.env.example .env   # fill in QDRANT_URL and OLLAMA_BASE_URL at minimum
.venv/bin/python server.py
```

Inside this checkout, Claude Code picks the server up from the checked-in `.mcp.json`
once the venv above exists. From anywhere else, add absolute paths to `~/.claude/settings.json`:

```json
"loci": {
  "type": "stdio",
  "command": "/path/to/.venv/bin/python3",
  "args": ["/path/to/loci/mcp/server.py"],
  "env": {
    "QDRANT_URL": "http://localhost:6333",
    "OLLAMA_BASE_URL": "http://localhost:11434"
  }
}
```

Grounding and injection-scanning hooks are a separate, opt-in install:
`scripts/hooks/install.sh` copies the three hooks into `~/.claude/hooks`, and
`install.sh --check` reports drift between the repo copy and the deployed one.

See [mcp/README.md](mcp/README.md) for the full tool reference and wiring guide, and
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for Docker and systemd deployment.

---

## Quick orientation

| What you want | Where to look |
|---|---|
| New to all of this — start here | [docs/CONCEPTS.md](docs/CONCEPTS.md) |
| FlyBrain overview and recommended reading path | [docs/FLYBRAIN_GUIDE.md](docs/FLYBRAIN_GUIDE.md) |
| FlyBrain provenance and replay rules | [docs/FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md](docs/FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md) |
| FlyBrain pilot queue / rollback / go-no-go controls | [docs/FLYBRAIN_QUEUE_CONTRACT_FREEZE_V1.md](docs/FLYBRAIN_QUEUE_CONTRACT_FREEZE_V1.md), [docs/FLYBRAIN_FAIL_BEHAVIOR_MATRIX.md](docs/FLYBRAIN_FAIL_BEHAVIOR_MATRIX.md), [docs/FLYBRAIN_QUEUE_STATE_MACHINE_SPEC.md](docs/FLYBRAIN_QUEUE_STATE_MACHINE_SPEC.md), [docs/FLYBRAIN_REPLAY_ACCEPTANCE_TEST_SPEC.md](docs/FLYBRAIN_REPLAY_ACCEPTANCE_TEST_SPEC.md), [docs/FLYBRAIN_PILOT_SLOS_AND_TRIPWIRES.md](docs/FLYBRAIN_PILOT_SLOS_AND_TRIPWIRES.md), [docs/FLYBRAIN_QUEUE_AND_PROMOTION_ROLLBACK_PLAYBOOK.md](docs/FLYBRAIN_QUEUE_AND_PROMOTION_ROLLBACK_PLAYBOOK.md), [docs/FLYBRAIN_PR_SLICING_GUARDRAILS.md](docs/FLYBRAIN_PR_SLICING_GUARDRAILS.md), [docs/FLYBRAIN_PILOT_GO_NO_GO_CHECKLIST.md](docs/FLYBRAIN_PILOT_GO_NO_GO_CHECKLIST.md) |
| FlyBrain dataset boundary matrix | [docs/FLYBRAIN_DATASET_PROVENANCE_MATRIX.md](docs/FLYBRAIN_DATASET_PROVENANCE_MATRIX.md) |
| FlyBrain harness write-path safety policy | [docs/FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md](docs/FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md) |
| FlyBrain harness storage-root layout | [docs/FLYBRAIN_HARNESS_STORAGE_LAYOUT.md](docs/FLYBRAIN_HARNESS_STORAGE_LAYOUT.md) |
| Repo-wide docs retention / archive policy | [docs/DOCS_RETENTION_POLICY.md](docs/DOCS_RETENTION_POLICY.md) |
| FlyBrain architecture / implementation mapping | [docs/FLYBRAIN_IO_TO_LOCI_MAPPING.md](docs/FLYBRAIN_IO_TO_LOCI_MAPPING.md) |
| FlyBrain glossary / quick term lookup | [docs/FLYBRAIN_REASONING_GLOSSARY.md](docs/FLYBRAIN_REASONING_GLOSSARY.md) |
| FlyBrain archive and redirect index | [docs/FLYBRAIN_ARCHIVE.md](docs/FLYBRAIN_ARCHIVE.md) |
| FlyBrain open-source tooling map | [docs/FLYBRAIN_OPEN_SOURCE_TOOLING_MAP.md](docs/FLYBRAIN_OPEN_SOURCE_TOOLING_MAP.md) |
| How the system works (technical) | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| API overview by tool family | [docs/API.md](docs/API.md) |
| Runtime registration and call paths | [docs/CALLGRAPH.md](docs/CALLGRAPH.md) |
| MCP reference for code graph / contracts / audit | [docs/API_CODE_GRAPH.md](docs/API_CODE_GRAPH.md) |
| Why it's designed this way | [docs/COGNITIVE_FOUNDATIONS.md](docs/COGNITIVE_FOUNDATIONS.md) |
| What each script does | [docs/COMPONENTS.md](docs/COMPONENTS.md) |
| How to run / configure (scripts) | [docs/OPERATIONS.md](docs/OPERATIONS.md) |
| How to deploy (Docker / systemd) | [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) |
| What the grounding corpus is worth | [docs/grounding-corpus-limits.md](docs/grounding-corpus-limits.md) |
| GPU offload architecture map | [docs/GPU_OFFLOAD_ARCHITECTURE.md](docs/GPU_OFFLOAD_ARCHITECTURE.md) |

---

## MCP tools (77)

![Tool groups](docs/img/loci-tools.svg)

*This diagram groups an earlier 24-tool snapshot by purpose. The grouping still holds.
The canonical source-checked composition is 77 tools: 43 server-local, 11
investigation, 11 code-graph, and 12 local-model tools, all registered onto the
shared FastMCP instance at import time. See [docs/CALLGRAPH.md](docs/CALLGRAPH.md)
for registration anchors and call paths.*

**By category** (full signatures, param docs, and response shapes: [mcp/README.md](mcp/README.md); curated deep-dive on the 24 most-used: [docs/memory-and-code-review-tools.md](docs/memory-and-code-review-tools.md#2d-loci-mcp--loci_memory--investigation-and-memory-layer)):

- **Session & findings** — `investigation_start`, `investigation_load`, `investigation_as_of`, `investigation_store`, `investigation_note`, `investigation_reflect`, `investigation_reason`, `investigation_list`, `investigation_share`, `investigation_unshare`, `investigation_export`, `investigation_import`, `finding_resolve`, `procedure_attempt`, `procedure_search`
- **Search & entities** — `investigation_search`, `investigation_pre_answer_check`, `investigation_evidence_precheck`, `investigation_entity_lookup`, `entity_list`, `entity_timeline`, `investigation_related_cases`, `investigation_finding_provenance`, `causal_infer`, `causal_edges_list`, `conflict_list`, `conflict_resolve`, `rag_context_search`, `ground`
- **Memory lifecycle & health** — `audit_log`, `memory_self_check`, `code_memory_correlate`, `memory_health`, `loci_health`, `retrieval_selftest`, `memory_retract`, `memory_restore`, `memory_promote`, `memory_demote`, `memory_hints`, `memory_surface`, `memory_route`, `memory_consolidate`, `memory_confidence`
- **Reflection loop** — `reflection_loop_seed`, `reflection_loop_tick`, `reflection_loop_status`
- **Contracts & wiring obligations** — `contract_declare`, `contract_query`, `contract_check`, `wiring_obligation_scan`, `wiring_obligation_declare`, `wiring_obligation_list`, `wiring_obligation_resolve`
- **Local-model offload** — `llm_local`, `generate_batch`, `query_expand`, `verify_finding`, `adversarial_review`, `investigation_verify_all`, `classify_text`, `compress_text`, `semantic_dedup`, `semantic_relevance`, `swarm_reason`
- **Code graph** — `code_graph_ingest`, `code_graph_query`, `code_memory_relink`, `code_memory_map`, `symbol_impact`, `impact_report`, `finding_code_context`, `investigation_code_briefing`, `subsystem_report`, `related_investigations_via_code`, `dead_code_candidates`

---

## A2A skills (13)

The A2A server exposes 13 skills via JSON-RPC 2.0 over HTTP, letting peer agents share
memory without requiring the MCP stack. Twelve are advertised in the agent card;
`memory_prime` is callable but not listed in discovery.

| Skill | Advertised |
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
| `memory_prime` | No |

---

## Infrastructure

| Resource | Default | Env var(s) |
|---|---|---|
| Qdrant | `http://localhost:6333` | `QDRANT_URL`, `QDRANT_API_KEY` |
| Ollama — embeddings (MCP server) | `http://localhost:11434` | `OLLAMA_BASE_URL` |
| Ollama — generation (MCP server) | resolved by `mcp/backends.py` | `LOCI_OLLAMA_GEN_URL`, `OLLAMA_GEN_URL` |
| Ollama (standalone scripts) | `http://localhost:11434` | `OLLAMA_URL` |
| Ollama (A2A server) | `http://localhost:11434/v1` | `MNEMOSYNE_EMBEDDING_API_URL` |
| Embedding model (MCP) | `nomic-embed-text` | `EMBED_MODEL` |
| Embedding model (A2A) | `nomic-embed-text` | `MNEMOSYNE_EMBEDDING_MODEL` |
| Embedding dimensions | `768` | `MNEMOSYNE_EMBEDDING_DIM` |
| Mnemosyne DB | `~/.hermes/mnemosyne/data/mnemosyne.db` | `MNEMOSYNE_DATA_DIR` |
| Memory session dir (MCP) | `~/.hermes/memory-sessions` | `LOCI_MEMORY_DIR` |
| Code graph store | `$LOCI_MEMORY_DIR/graph.ladybug` | — |
| Qdrant collection prefix | `loci_memory` | `QDRANT_COLLECTION_PREFIX` |
| Backend config file | `~/.loci/backends.toml` | `LOCI_CONFIG` |
| Hook state | `~/.claude/hook-state/` | — |

`OLLAMA_BASE_URL` is the **embedding** endpoint only. Generation resolves separately
(`backends.ollama_gen_url()`, `mcp/backends.py:101`), so an Ollama that serves only
`nomic-embed-text` cannot absorb generation calls. `OLLAMA_URL` is the standalone-script
fallback. Set all three when the components run against different hosts.

Unset values fall through `mcp/backends.py`: explicit env var, localhost probe,
`~/.loci/backends.toml`, then empty; the offload tiers fail open on empty.
`backends.toml.example` is the template. See [docs/OPERATIONS.md](docs/OPERATIONS.md)
for the full env var reference.

Two `.env.example` files are provided:

- **`.env.example`** (repo root) — complete reference covering all components
- **`mcp/.env.example`** — minimal file for MCP-server-only deployments

---

## Key env vars

### MCP server

| Variable | Default | Purpose |
|---|---|---|
| `QDRANT_URL` | _(required)_ | Qdrant instance URL |
| `QDRANT_API_KEY` | `""` | Qdrant auth key (blank for no-auth) |
| `QDRANT_COLLECTION_PREFIX` | `loci_memory` | Main Qdrant collection name |
| `OLLAMA_BASE_URL` | _(required for embeddings)_ | Ollama base URL (no trailing `/v1`) |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model for MCP server |
| `EMBED_API_KEY` | `""` | Cloud embedding provider API key |
| `EMBED_API_KEY_HEADER` | `Authorization` | Auth header name for cloud embeddings |
| `LOCI_MEMORY_DIR` | `~/.hermes/memory-sessions` | Investigation session storage root |
| `MNEMOSYNE_EMBEDDING_DIM` | `768` | Vector dimension — must match your model |
| `CODE_CHUNKS_COLLECTION` | _(unset)_ | Qdrant collection for `code_memory_correlate` |
| `LOCI_MCP_TRANSPORT` | `stdio` | Transport mode: `stdio`, `sse`, or `streamable-http` |
| `LOCI_MCP_HOST` | `127.0.0.1` | Bind host for SSE/HTTP transport |
| `LOCI_MCP_PORT` | `8000` | Bind port for SSE/HTTP transport |
| `LOCI_MCP_TOKEN` | `""` | Bearer token for SSE/HTTP. Required for any non-loopback bind — the server exits rather than serve unauthenticated |
| `LOCI_OLLAMA_GEN_URL` | _(resolved by `backends.py`)_ | Generation endpoint override (`OLLAMA_GEN_URL` also read) |
| `LOCI_VLLM_FALLBACK` | `0` | Allow generation to fall back to vLLM (`VLLM_BASE_URL`) when Ollama fails |
| `LOCI_CONFIG` | `~/.loci/backends.toml` | Backend resolution config file |

### A2A server

| Variable | Default | Purpose |
|---|---|---|
| `MNEMOSYNE_EMBEDDING_API_URL` | `http://localhost:11434/v1` | OpenAI-compat embedding endpoint |
| `MNEMOSYNE_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model for A2A server |
| `MNEMOSYNE_DATA_DIR` | `~/.hermes/mnemosyne/data` | Mnemosyne SQLite data directory |
| `LOCI_A2A_TOKEN` | _(required)_ | Bearer token callers must present |
| `LOCI_A2A_TOTP_SEED` | `""` | TOTP base32 seed (blank to disable) |
| `LOCI_A2A_URL` | `http://127.0.0.1:8201` | Public URL injected into the agent card |
| `HERMES_AGENT_ID` | `hermes-agent` | Agent identity stamped on all writes |
| `EXTRA_RAG_COLLECTIONS` | `""` | Comma-separated extra Qdrant collections for fan-out RAG |
| `PEER_A2A_URLS` | `""` | Comma-separated peer A2A endpoints for context broadcast |

---

## Qdrant collections

| Collection | Purpose |
|---|---|
| `loci_memory` (configurable via `QDRANT_COLLECTION_PREFIX`) | Primary long-term memory store |
| `loci_sessions` | Session history embeddings |
| `loci_verdicts` | Claim verdict history for `investigation_pre_answer_check` and `memory_self_check` |
| `mnemosyne` | Synced Mnemosyne SQLite vectors |

---

## MCP transport modes

Default transport is `stdio` for Claude Code subprocess use. For Docker or remote
deployments set `LOCI_MCP_TRANSPORT=sse` (or `streamable-http`) and configure
`LOCI_MCP_HOST` / `LOCI_MCP_PORT`.

Loopback is default. Wider binds require `LOCI_MCP_TOKEN`; without one the server
exits (`mcp/server.py:8144`). With a token set, callers present
`Authorization: ******; `/health` stays open for liveness probes. `docker-compose.yml`
sets `LOCI_MCP_HOST=0.0.0.0`, so `.env` must also set a token before
`docker compose up`.

```bash
# loopback — no token needed
LOCI_MCP_TRANSPORT=sse LOCI_MCP_PORT=8000 \
  .venv/bin/python server.py

# wide bind — token required, or the server refuses to start
LOCI_MCP_TRANSPORT=sse LOCI_MCP_HOST=0.0.0.0 LOCI_MCP_PORT=8000 \
  LOCI_MCP_TOKEN="$(python3 -c 'import secrets;print(secrets.token_hex(32))')" \
  .venv/bin/python server.py
```

---

## Repo layout

```
loci/
├── mcp/                   MCP server — 76 tools: investigation memory, RAG, claim validation, code graph
│   ├── server.py          FastMCP server entry point
│   ├── backends.py        Endpoint resolution: env var -> localhost probe -> ~/.loci/backends.toml
│   ├── memcheck/          Standalone claim-validation + code-hallucination module
│   ├── pyproject.toml     Package definition (pip install -e .)
│   └── README.md          MCP setup and tool reference
├── docs/                  Architecture, theory, component reference, ops guide
│   ├── CONCEPTS.md        Plain-English guide for beginners
│   └── img/               SVG diagrams
├── scripts/               Python scripts (run standalone or via cron)
│   ├── loci_groom.py      Passive grooming passes: index, knn_tags, codelink, summaries, ...
│   └── hooks/             Claude Code / agent hook adapters + install.sh
├── mlops/                 Embedding, routing, grounding and finetune experiments
├── eval/                  Grounding-gate and skeptic-verification benchmarks
├── deep_think_loci/       Multi-tier reasoning engine — Workflow over the Loci corpus (beta)
├── a2a_server/            A2A RAG broadcast server (mesh-wide context sharing)
├── rules/                 Agent behavioral rules (loaded at session start)
├── cron/jobs.json         Hermes cron job list — tick with scripts/hermes_cron_runner.py
│                          on a 1-minute user timer/crontab; grooming stays separate
├── backends.toml.example  Template for ~/.loci/backends.toml
└── .env.example           Full environment variable reference for all components
```
