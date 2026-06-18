# Memory & Code Review Tools — How Everything Ties Together
## Hermes Agent Stack

This document maps every memory, knowledge, and code review tool in the
Hermes stack, explains what each one does, and shows how they connect.
Written from the actual running config and source code, not assumed behavior.

> **Security note:** This document uses `<your-host>` for the Qdrant/Ollama
> host. Real IPs, API keys, and collection names are stored in config.yaml
> and the .env file — not here.

---

## Table of Contents

1. [The big picture](#1-the-big-picture)
2. [Memory layers — what lives where](#2-memory-layers)
3. [The grounding pipeline — what runs before every LLM reply](#3-the-grounding-pipeline)
4. [The grounding daemon](#4-the-grounding-daemon)
5. [Background automation — the full cron stack](#5-background-automation)
6. [Code review and navigation tools](#6-code-review-tools)
7. [Knowledge stores in Qdrant](#7-qdrant-knowledge-stores)
8. [The session sync pipeline](#8-session-sync-pipeline)
9. [Rules files — always-on constraints](#9-rules-files)
10. [How it all connects — a single turn traced end to end](#10-a-turn-traced-end-to-end)
11. [When to use what — decision guide](#11-decision-guide)
12. [Health checks and recovery](#12-health-checks-and-recovery)
13. [Configuration reference](#13-configuration-reference)
14. [Pitfalls](#14-pitfalls)

---

## 1. The big picture

```
                        ┌─────────────────────────────────────────────────┐
                        │                  EVERY TURN                     │
                        │                                                 │
  User message ──────►  │  pre_llm_call hook                             │
                        │    └─ grounding_client.py → daemon socket      │
                        │    └─ embed message (nomic-embed-text, 768d)   │
                        │    └─ parallel fan-out to 7 Qdrant collections │
                        │    └─ inject MEMORY MATCH block (~100ms)       │
                        │    └─ inject rules files (≤1200 chars)         │
                        │                                                 │
                        │  LLM sees: message + memory matches + rules     │
                        │                                                 │
                        │  pre_tool_call hook (every tool use)           │
                        │    └─ grounding enforcement (read before write) │
                        │    └─ dangerous command detection (log; BLOCK  │
                        │       if BLOCK_MODE=1 in config)               │
                        │    └─ audit log write                          │
                        │                                                 │
  Session ends ──────►  │  on_session_end hook                           │
                        │    └─ embed session → hermes_sessions Qdrant   │
                        │    └─ ~140ms, Ollama nomic-embed-text          │
                        └─────────────────────────────────────────────────┘

  Persistent background (cron jobs — 5 total):
    state_db_qdrant_sync.py     — every 5 min  — catch-all session sync
    mnemosyne_activity_check.py — every 20 min — consolidation gate
    mnemosyne_activity_check.py — every 20 min — AI memory archivist
    mnemosyne_sleep_all.sh      — every 30 min — CLI sleep/consolidation
    mnemosyne_qdrant_sync.py    — every 30 min — Mnemosyne → Qdrant sync

  Persistent service:
    hermes-grounding            — systemd --user — grounding daemon (UNIX socket)

  Interactive tools (in-session):
    loci-mcp (hermes_memory)  — investigation/memory layer, 25 tools (see Section 2d)
    Mnemosyne MCP             — episodic facts, store/recall with semantic search
    Serena MCP                — LSP-powered code navigation (symbols, refs, diagnostics)
    CocoIndex MCP             — semantic code search (ccc) across indexed repos
    Open Design MCP           — artifact and project retrieval (Docker-based)
    deep_think MCP            — parallel perspective reasoning via HTTP
    Qdrant MCP                — direct vector search across all collections
    memory tool               — MEMORY.md + USER.md flat-file facts (injected every turn)
    session_search            — FTS5 search over local state.db session history
    skills                    — procedural docs loaded on demand
```

The key insight: **most context injection happens automatically before you type
anything.** By the time the LLM sees your message, it already has relevant
memories, past session snippets, and always-on rules injected into context via
the pre_llm_call hook.

---

## 2. Memory layers

There are five distinct memory layers. They serve different purposes and have
different query patterns.

### 2a. MEMORY.md + USER.md (flat-file, injected every turn)

**Where:** `~/.hermes/memories/MEMORY.md` and `USER.md`

**What it is:** Two small markdown files injected verbatim into the system
prompt on every single turn. No query needed — always present.

**MEMORY.md** — notes about the environment, tools, and codebase.
  - Stack facts: Qdrant host/port, Ollama host, setup details
  - Tool quirks and workarounds
  - hermes_memory repo layout, session sync config
  - Stable conventions that would otherwise need re-explaining every session

**USER.md** — facts about the user.
  - Name, timezone, working style preferences
  - Fleet context and project-specific notes
  - Pet peeves and corrections

**Budget:** ~2200 chars for MEMORY.md, ~1375 for USER.md (from config.yaml).
When files hit 96-99%, prune before adding new entries.

**Updated via:** `memory` tool in-session.
  ```
  memory(action='add', target='memory', content='...')
  memory(action='replace', target='user', old_text='...', content='...')
  memory(action='remove', target='memory', old_text='...')
  ```

**Rule:** Write facts, not instructions. "User prefers concise responses" ✓
"Always respond concisely" ✗ — imperative phrasing re-reads as a directive.
Procedures belong in skills; constraints belong in rules files.

### 2b. Mnemosyne (episodic SQLite + semantic vector search)

**Where:** `~/.hermes/mnemosyne/data/` (SQLite — local only)
  Embeddings via Ollama nomic-embed-text on `<your-host>`.

**What it is:** A semantic memory store — store arbitrary text with importance
scores, query by similarity. Used for episodic facts, research findings,
investigation notes, architectural decisions.

**Backend:** Mnemosyne runs on local SQLite only. It is NOT backed by
PostgreSQL. Any Postgres service is a separate operational database for
project data — not the Mnemosyne store.

**MCP tools available (Mnemosyne server):**
  - `mcp_mnemosyne_mnemosyne_remember(content, importance, bank)` — store
  - `mcp_mnemosyne_mnemosyne_recall(query, top_k, vec_weight, fts_weight, bank)` — search
  - `mcp_mnemosyne_mnemosyne_get_stats(bank)` — collection stats
  - `mcp_mnemosyne_mnemosyne_scratchpad_read/write(bank)` — temp workspace
  - `mcp_mnemosyne_mnemosyne_sleep(bank)` — consolidation pass (merge working memories into episodic)

These tools belong to the **Mnemosyne MCP server**, which is distinct from the
hermes_memory server (loci-mcp) described in Section 2d. The loci-mcp server
also writes to Mnemosyne internally via `_mnemo_remember()`, but is a separate
server providing different tools.

**Scoring:** Hybrid — vector similarity (0.5) + full-text search (0.3) +
importance weight (0.2) + optional temporal boost. Tune with vec_weight,
fts_weight params.

**Also synced to Qdrant** `mnemosyne` collection (by the 30-min cron) so the
pre_llm_call hook can include it in the fan-out without going through MCP.

### 2c. Qdrant hermes_sessions (session history, vector search)

**Where:** Qdrant on `<your-host>`, collection `hermes_sessions`

**What it is:** Every completed session embedded as a single 768-dim vector
(nomic-embed-text). Used by the grounding hook to surface relevant past sessions
automatically, and directly queryable.

**How it gets populated:** `on_session_end` hook runs `session_end_sync.py`
(~140ms target), embeds the session content, upserts into Qdrant. Background
cron `state_db_qdrant_sync.py` runs every 5 min as catch-all for any missed.

**Also queryable via:** `session_search` tool — FTS5 search over `state.db`
(the local SQLite session store). session_search is faster for keyword/phrase
queries; Qdrant hermes_sessions is better for semantic/topical recall.

### 2d. loci-mcp / hermes_memory — investigation and memory layer

**Server name:** `loci` (FastMCP name in `mcp/server.py`)
**Storage:** `$HERMES_MEMORY_DIR/<investigation_id>/` (default: `~/.hermes/memory-sessions/`)
**Qdrant collection:** `hermes_memory` (or value of `QDRANT_COLLECTION_PREFIX`)

**What it is:** A manifest-first investigation and memory layer. Tracks
findings, observations, inferences, audit logs, and hallucination retractions
across investigation sessions. This is the primary writer to the
`hermes_memory` Qdrant collection and the Mnemosyne `default` bank via
`investigation_store`, `audit_log`, and `reflection_loop_tick` — not the
Qdrant MCP server writing directly.

**Storage layout:**
```
$HERMES_MEMORY_DIR/
  <investigation_id>/
    manifest.json        — structured investigation state
    findings.jsonl       — append-only finding log
    audit.jsonl          — tool call/response audit log (investigation-scoped)
    retractions.jsonl    — soft-tombstone log (append-only; active:false = restore)
    retraction_audit.jsonl — retract/restore action audit trail

  ../audit/
    YYYY-MM-DD.jsonl     — global cross-investigation audit log
```

**finding_type enum:** `observed` | `inferred` | `assumed` | `gap`
- `observed` — from a direct tool response; cite source and key values
- `inferred` — reasoned from observations but not directly stated
- `assumed` — working hypothesis with no current evidence
- `gap` — something that should be checked but hasn't been

**25 MCP tools provided by loci-mcp:**

#### Investigation CRUD

| Tool | Purpose |
|------|---------|
| `investigation_start(investigation_id, title, context?)` | Create or resume an investigation. Idempotent — resuming returns current manifest without overwriting. |
| `investigation_load(investigation_id, last_n_findings?, include_retracted?)` | Retrieve manifest + recent findings for context recovery at session start. Retracted findings excluded by default. |
| `investigation_store(investigation_id, finding_type, text, source, confidence?, tags?, derived_from?)` | Record a finding. Writes to JSONL + Mnemosyne + Qdrant. `derived_from` links lineage for retraction cleanup. |
| `investigation_note(investigation_id, field, value)` | Update manifest fields: `context`, `hypothesis`, `next_step`, `open_question_add/remove`, `checked_source`, `closed_summary`. |
| `investigation_reflect(investigation_id)` | Synthesize current state — finding breakdown, open questions, gaps, key entities, advisory self-check. |
| `investigation_list()` | List all investigations with status and finding counts, most-recently-updated first. |

#### Investigation Search and Evidence

| Tool | Purpose |
|------|---------|
| `investigation_search(query, investigation_id?, limit?, include_retracted?, min_confidence?)` | Hybrid Mnemosyne+Qdrant search. Resolution order: Mnemosyne recall → Qdrant RRF (dense+sparse) → cross-encoder reranking. Retracted findings excluded by default. |
| `investigation_pre_answer_check(investigation_id, claims, min_confidence?, record?)` | Validate proposed response claims against stored findings and audit receipts. Implements CIBER dual-retrieval: also retrieves benign baseline context from other investigations. Verdicts recorded to `hermes_verdicts` Qdrant collection when `record=True`. |
| `investigation_evidence_precheck(investigation_id, proposed_query, min_similarity?)` | Lightweight duplicate-call avoidance. Checks if similar evidence already exists before running a new tool call. |
| `investigation_entity_lookup(entity, entity_type?, investigation_id?, limit?)` | Find every finding mentioning a specific IP, email, hostname, hash, or CVE. Uses Qdrant payload indexes for O(1) lookup; falls back to JSONL scan. `entity_type="auto"` infers from value. |
| `investigation_related_cases(entities, entity_type?, limit_per_entity?)` | Find prior investigations that dealt with the same entities. Call before opening a new investigation to check for prior resolutions. |
| `investigation_finding_provenance(finding_id, investigation_id)` | Trace a finding back through `derived_from` links to root observed evidence. Identifies hypothesis chains vs evidence-grounded conclusions. |

#### Memory Quality and Retraction

| Tool | Purpose |
|------|---------|
| `memory_self_check(investigation_id?, checks?, record?)` | Advisory provenance + contradiction checks over stored findings. Surfaces `hallucination_candidates` — findings that are both unsupported (no audit receipt) and contradicted by a receipted finding. Never auto-retracts. |
| `memory_retract(investigation_id, target, reason?, dry_run?, scope_semantic?)` | Soft-tombstone a hallucinated finding and its contaminated lineage. `dry_run=True` (default) previews the cluster without changing anything. Re-run with `dry_run=False` to apply. Reversible via `memory_restore`. |
| `memory_restore(investigation_id, finding_id?, retraction_id?, reason?)` | Reverse a retraction — un-tombstone a finding so it returns to recall/search/reflect. Appends `active:false` to retractions log. |
| `memory_health(investigation_id?)` | Substrate self-check: 8 probes — `qdrant_reachable`, `qdrant_collections`, `embeddings_dense`, `embeddings_sparse`, `mnemo_mirror`, `dimension_consistency`, `retraction_integrity`, `store_counts`. Read-only and fail-open. |

#### Audit

| Tool | Purpose |
|------|---------|
| `audit_log(tool_name, inputs_json, output, investigation_id?, embedding_text?)` | Record a tool call and full output. Writes to global daily JSONL (`../audit/YYYY-MM-DD.jsonl`), investigation-scoped JSONL, Mnemosyne, and Qdrant. Use `embedding_text` to store a crafted semantic summary instead of raw output for better recall. |

#### RAG and Broader Search

| Tool | Purpose |
|------|---------|
| `rag_context_search(query, limit?, collections?, budget_chars?, exclude_types?)` | Hybrid Qdrant RAG with cross-encoder reranking. Searches `hermes_memory` and any code-chunks collection by default. Returns prompt-ready context with `[SOURCE N]` citations. Requires Qdrant — no keyword fallback. |
| `memory_confidence(query, top_k?)` | Metamemory: 5-cue calibrated confidence score (fluency, accessibility, source_diversity, corroboration, trust). Use before asserting a memory-derived claim. |
| `dama_routing_query(device_id?, action?, reason_label?, min_confidence?, limit?)` | Query routing decisions by structured payload filters. Requires `ROUTING_DECISIONS_COLLECTION` env var. Returns matching decisions sorted by timestamp. |

#### Memory Maintenance

| Tool | Purpose |
|------|---------|
| `memory_consolidate(dry_run?)` | Run Mnemosyne sleep/consolidation cycle — merges working_memory into episodic memory. Wraps `Mnemosyne.sleep_all_sessions()`. |
| `code_memory_correlate(investigation_id, target_file?, entity?)` | Link a code hallucination to contaminated memories. Runs `run_code_checks` on a `.py` file (LH000/LH001/LH003/LH007/LH009) and finds memories contaminated by the same entities. Advisory and read-only. |

#### Self-Reflection Loop

| Tool | Purpose |
|------|---------|
| `reflection_loop_seed(investigation_id?, session_events_limit?, process_logs_limit?, reset_queue?)` | Enqueue local Copilot artifacts (session events, process logs, temp_ingest) into the reflection queue. Does not parse files — use `reflection_loop_tick` to process. |
| `reflection_loop_tick(max_items?, max_lines_per_file?, store_item_findings?)` | Process a small queue batch. Deterministic parsing only — no LLM pass. Bounded by `max_items` and `max_lines_per_file`. Writes findings via `investigation_store`. |
| `reflection_loop_status(queue_preview?)` | Inspect reflection queue size, processed count, and aggregate stats. |

**Embedding and storage flow for `investigation_store`:**
```
investigation_store(...)
  → append to findings.jsonl (durable, always)
  → _mnemo_remember(text, importance, metadata) → Mnemosyne SQLite
  → _qdrant_upsert(finding_id, text, payload) → hermes_memory collection
    (dense + sparse RRF, INT8 quantized, HNSW m=32)
```

**`hermes_verdicts` Qdrant collection:** Used by `investigation_pre_answer_check`
to record per-claim verdict history and detect conflicts. Verdict types:
`claim_supported`, `claim_contradicted`, `claim_unsupported`, `claim_ambiguous`.
Implements PE-gated reconsolidation: high prediction-error verdict changes are
marked provisional until a second independent observation confirms them.

### 2e. skills (procedural docs, loaded on demand)

**Where:** `~/.hermes/skills/` (186+ skills across 26 categories)

**What it is:** Markdown files with YAML frontmatter describing how to do
specific things. NOT injected automatically — loaded explicitly when triggered
by task type.

**The system prompt includes** a list of all skill names and descriptions so
the LLM can decide when to load them. The actual SKILL.md content only loads
when `skill_view()` is called.

**Rule:** If any skill matches your task, load it. Skills contain API endpoints,
exact commands, proven workflows, and pitfalls that outperform general knowledge.

---

## 3. The grounding pipeline

This is the most important automation in the stack. It runs before every LLM
reply, transparently.

### 3a. pre_llm_call hook

**Entry point:** `~/.hermes/scripts/grounding_client.py`
  → connects to the grounding daemon via UNIX socket at `/tmp/hermes-grounding.sock`
  → falls back to spawning `hooks/pre_llm_grounding.py` directly if daemon is down

The daemon must be running for the fast path. See Section 4.

**Timing:**
  - Daemon running (warm): ~100ms total (70ms embed + 30ms parallel Qdrant)
  - Daemon down (fallback subprocess): ~140ms cold start
  - Previous version (BeamMemory/SQLite): ~1500ms

**What it does, step by step:**

```
1. Receive the user's message text
2. Embed via nomic-embed-text on <your-host> Ollama → 768-dim vector
3. Fan out to 7 Qdrant collections in parallel (ThreadPoolExecutor, 8 workers):
   - mnemosyne          — personal facts, preferences
   - hermes_sessions    — past conversation history
   - hermes_memory      — investigation notes, findings (written by loci-mcp)
   - ecc_skills         — skill library
   - agent_core_chunks  — great-library KB (DAMA, infra, code)
   - dama_gotchi_code   — DAMA codebase
   - prometheus_dama_code — prometheus codebase
4. Score fusion: Qdrant cosine_score × importance_weight (where available)
5. Filter: score ≥ 0.55, importance ≥ 0.2, deduplicate
6. Take top 5 results, truncate each to 200 chars
7. Inject MEMORY MATCH block into the system prompt
8. Inject rules files (≤1200 chars from ~/.hermes/rules/)
```

**Guard conditions — hook skips if:**
- Message is < 15 chars (too short to embed meaningfully)
- Session is a subagent (avoid circular context injection)
- Ollama is unreachable (falls back to BeamMemory FTS path — best-effort,
  requires mnemosyne library importable)

**Output injected into context looks like:**
```
[MEMORY MATCH — top 5 results across 7 collections]
[mnemosyne 0.87] User prefers direct commands over explanations...
[hermes_sessions 0.82] ...investigated proxy.ts dead middleware June 13...
[agent_core_chunks 0.79] ...FedAvg implementation requires matching...
```

**Also reads:** `~/.hermes/.env` at startup for env overrides.
The hook's env vars are set both in config.yaml hooks stanza AND this .env file.

### 3b. pre_tool_call hook

**Script:** `~/.hermes/scripts/hooks/pre_tool_grounding.py`

**What it does:**

1. **Grounding enforcement** — before write tools (edit, terminal, file create),
   checks that the LLM has used at least one read/search tool on the target
   first. Prevents "edit without reading" patterns.

2. **Dangerous command detection** — scans terminal commands for:
   `rm -rf`, `DROP TABLE`, `force-push`, `kubectl delete`, `chmod +x + execute`
   Logs a warning to the audit log. Set `BLOCK_MODE=1` to also enforce blocking.
   **Default is `BLOCK_MODE=0` — log only, no blocking.** This is an advisory
   control in current config. Treat it as a record, not a guarantee.

3. **Audit log** — writes every tool call to a local audit log (capped at 5MB).

**GROUNDING_TOOLS** (always allowed, no audit noise):
  All Mnemosyne MCP tools, Serena read tools, session_search, web_search,
  web_extract, read_file, search_files, Open Design read tools.

---

## 4. The grounding daemon

**Service name:** `hermes-grounding` (systemd --user unit)
**UNIX socket:** `/tmp/hermes-grounding.sock`
**Script:** `~/.hermes/scripts/grounding_daemon.py`

The daemon is a persistent Python process that keeps Ollama and Qdrant
connections warm. Without it, every turn spawns a cold subprocess (~140ms).
With it, the round-trip is ~50ms socket overhead + ~70ms Ollama + ~30ms Qdrant.

**Lifecycle:**
```bash
# Check status
systemctl --user status hermes-grounding

# Start
systemctl --user start hermes-grounding

# Enable on login
systemctl --user enable hermes-grounding

# View logs
journalctl --user -u hermes-grounding -n 50 -f

# Restart after config change
systemctl --user restart hermes-grounding
```

**If the daemon is down:** the grounding_client.py falls back to spawning the
hook script directly as a subprocess. Sessions still work — just slower.
Check `journalctl` output if the grounding hook seems slow or stale.

---

## 5. Background automation — the full cron stack

Five cron jobs run continuously. All defined in `cron/jobs.json`.

### Job 1: state-db-qdrant-sync (every 5 min)

**Script:** `scripts/state_db_qdrant_sync.py` | **Type:** no_agent (script only)

Reads sessions from `state.db`, embeds via Ollama, upserts into Qdrant
`hermes_sessions`. Incremental — tracks synced sessions in a cache file.

This is the catch-all safety net for sessions that the on_session_end hook
missed (Ollama down, hook timeout, sessions from before hook was wired).

**Check if working:**
```bash
# View last run output
ls ~/.hermes/cron/output/ | grep state-db | tail -5

# Manually run
python3 ~/.hermes/scripts/state_db_qdrant_sync.py

# Compare state.db session count vs Qdrant hermes_sessions
sqlite3 ~/.hermes/state.db "SELECT COUNT(*) FROM sessions"
```

### Job 2: mnemosyne-consolidation (every 20 min)

**Script:** `scripts/mnemosyne_activity_check.py` | **Type:** LLM agent

Gate check: runs the script first. If working_memory grew since last check,
calls `mnemosyne_sleep` for both `default` and project-specific banks. This
merges working memories into episodic memory (the consolidation pass).

If the script output is empty/blank — job is silent, no tokens spent.

### Job 3: mnemosyne-session-summarizer (every 20 min)

**Script:** `scripts/mnemosyne_activity_check.py` | **Type:** LLM agent

Same gate check as Job 2. If activity detected, runs a full AI memory
archival cycle:
1. Reads bank scratchpads
2. Recalls recent related memories (deduplication)
3. Writes structured narrative memories (decisions, discoveries, state)
4. Writes knowledge triples (entity relationships)
5. Adds temporal annotations to stored memories
6. Updates running scratchpad state
7. Runs consolidation sleep

This is an AI agent writing memories autonomously every 20 minutes when there
is active session activity.

### Job 4: mnemosyne-sleep-cli (every 30 min)

**Script:** `scripts/mnemosyne_sleep_all.sh` | **Type:** no_agent (shell only)

CLI-level sleep/consolidation for all Mnemosyne banks. Lighter than Job 2/3 —
no LLM involved. Just calls the Mnemosyne CLI sleep command.

### Job 5: mnemosyne-qdrant-sync (every 30 min)

**Script:** `scripts/mnemosyne_qdrant_sync.py` | **Type:** no_agent (script only)

Syncs Mnemosyne SQLite → Qdrant `mnemosyne` collection so the grounding hook
can search Mnemosyne content without going through the MCP server on every turn.

---

## 6. Code review and navigation tools

### 6a. Serena MCP (LSP-powered symbol navigation)

**Command:** `~/.local/bin/serena start-mcp-server`

**What it is:** Language Server Protocol client that understands your code
structurally. Works on the active project (set via `mcp_serena_activate_project`).

**Tools for code review:**

| Tool | What it does |
|------|-------------|
| `mcp_serena_get_symbols_overview` | Top-level class/function map of a file |
| `mcp_serena_find_symbol` | Find any symbol by name path (e.g. `MyClass/my_method`) |
| `mcp_serena_find_referencing_symbols` | All callers of a symbol — dead code detection |
| `mcp_serena_find_declaration` | Jump to declaration from a call site |
| `mcp_serena_find_implementations` | All implementations of an interface |
| `mcp_serena_get_diagnostics_for_file` | TS/language errors on a file |
| `mcp_serena_search_for_pattern` | Regex search across codebase |
| `mcp_serena_read_file` | Read file with line numbers |
| `mcp_serena_list_dir` | Recursive directory listing |

**Write tools:**

| Tool | What it does |
|------|-------------|
| `mcp_serena_replace_content` | Regex or literal find-replace in a file |
| `mcp_serena_replace_symbol_body` | Replace an entire function/class body |
| `mcp_serena_insert_before/after_symbol` | Insert code around a symbol |
| `mcp_serena_create_text_file` | Write a new file |
| `mcp_serena_rename_symbol` | Rename across entire codebase |
| `mcp_serena_safe_delete_symbol` | Delete only if no references exist |

**Activate project first:**
```python
mcp_serena_activate_project("~/development/myproject")
```
Without activation, all symbol tools return empty or navigate the wrong repo.

**Project-scoped memory:** `mcp_serena_write_memory` / `mcp_serena_read_memory`
for notes that should persist across sessions for a specific project.

### 6b. CocoIndex MCP (semantic code search)

**Command:** `~/.local/bin/ccc mcp`

**What it is:** Semantic code indexer. Indexes a codebase into Qdrant, then
lets you search by meaning ("where is authentication handled") not just text.

**Difference from Serena:** Serena = structural (knows symbols, types, call
graphs). CocoIndex = semantic (finds code by what it does, not what it's named).
Use together: CocoIndex to find the general area, Serena to navigate structure.

**If index is stale:** `ccc index <project_path>` to rebuild. Check stale state:
```bash
ccc status  # or check if search results reference old code paths
```

**Skill:** `cocoindex-serena` — install, init, and combined workflow.

### 6c. Open Design MCP (artifact and project retrieval)

**Command:** `~/.hermes/scripts/od-mcp.sh`
  → runs `node /app/apps/daemon/dist/cli.js mcp` inside the `open-design` Docker container

**What it is:** Retrieves design artifacts, project files, and agent context
from the Open Design orchestration platform running at port 7456.

**Requirements:** Docker container `open-design` must be running.
**Check:**
```bash
docker ps | grep open-design
# If not running:
cd ~/open-design/deploy && docker compose up -d
```

**Tools available:** `mcp_open_design_get_artifact`, `mcp_open_design_get_project`,
`mcp_open_design_list_projects`, `mcp_open_design_search_files`, etc.
These are GROUNDING_TOOLS — always allowed, no audit noise.

### 6d. deep_think MCP (parallel perspective reasoning)

**URL:** `http://127.0.0.1:30852/mcp` (systemd --user service)

**What it is:** Spawns multiple LLM instances with different analytical
perspectives in parallel. Used for adversarial code review — each "perspective"
attacks the code from a different angle (security, correctness, architecture,
performance).

**Use for:** Security reviews, complex debugging, architecture decisions.

**Check service:**
```bash
systemctl --user status deep-think-mcp
curl http://127.0.0.1:30852/health
```

**Skill:** `deep-think-fan-out` — exact job firing, polling, result retrieval.

**Note on small models:** Models <4B params fabricate function names, line
numbers, and code blocks. Use finding classes as pointers; verify specifics
with search_files/read_file before acting. See the skill for full guidance.

### 6e. session_search (FTS5 over state.db)

**What it is:** Full-text search over the local session database (`state.db`).
Every conversation is stored. FTS5-backed with BM25 ranking.

**Four calling shapes:**
1. **Discovery** — `session_search(query="topic")` — finds sessions, returns
   bookend_start (first 3 messages), FTS5 hit in context, bookend_end (last 3).
2. **Scroll** — `session_search(session_id, around_message_id)` — get ±N
   messages around a point in a known session.
3. **Read** — `session_search(session_id)` — dump entire session.
4. **Browse** — `session_search()` — recent sessions by time.

**vs Qdrant hermes_sessions:** session_search = keyword/phrase/boolean (fast,
exact, has live in-progress sessions). Qdrant = semantic/topical (finds what
sessions are *about*, synced after session ends). Use session_search first.

---

## 7. Qdrant knowledge stores

All collections are on `<your-host>`, 768-dim nomic-embed-text vectors (except
where noted), cosine similarity. Point counts below are approximate — treat
them as orders of magnitude.

| Collection | Approx size | Content | Updated by |
|------------|-------------|---------|------------|
| `mnemosyne` | hundreds | Personal facts, preferences, project notes | Mnemosyne MCP + 30-min cron |
| `hermes_sessions` | hundreds | Past conversation sessions | on_session_end hook + 5-min cron |
| `hermes_memory` | tens–hundreds | Investigation notes, findings (dense+sparse, INT8 quant) | loci-mcp via investigation_store, audit_log, reflection_loop_tick |
| `hermes_verdicts` | varies | Claim check verdict history (pre_answer_check results) | loci-mcp via investigation_pre_answer_check |
| `ecc_skills` | hundreds | Skill library knowledge | ECC skill indexing pipeline |
| `agent_core_chunks` | hundreds of thousands | Knowledge base: DAMA, infra, code, telemetry | ingestion pipeline |
| `dama_gotchi_code` | tens of thousands | DAMA codebase (source indexed) | CocoIndex / ccc |
| `prometheus_dama_code` | thousands | Prometheus DAMA source | CocoIndex / ccc |

**Check live collection sizes:**
```bash
curl -s http://<your-host>:<qdrant-port>/collections \
  -H "api-key: $(grep QDRANT_API_KEY ~/.hermes/.env | cut -d= -f2)" \
  | python3 -m json.tool | grep -A3 '"name"'
```

---

## 8. Session sync pipeline

This is how what you do in a session becomes findable later.

```
Session ends
    │
    ▼
on_session_end hook fires
    script: scripts/hooks/session_end_sync.py
    target latency: <500ms
    │
    ├─ Reads session from state.db (local SQLite)
    ├─ Concatenates user+assistant messages, cap at 4000 chars
    ├─ Fast path: skip if no new messages since last upsert (cache file check)
    ├─ Embeds via nomic-embed-text on <your-host> Ollama
    ├─ Upserts point into Qdrant hermes_sessions collection
    └─ Updates sync cache to track synced sessions
    │
    ▼
state_db_qdrant_sync.py (background cron, every 5 min)
    │
    └─ Finds sessions in state.db NOT in sync cache
       Embeds and upserts them into hermes_sessions
       Safety net for: Ollama down, hook timeout, pre-hook sessions
```

**Cache location:** set via `HERMES_SYNC_CACHE` env var in the on_session_end
hook config stanza.

**Debugging sync issues:**
```bash
# Check Qdrant point count vs local session count
sqlite3 ~/.hermes/state.db "SELECT COUNT(*) FROM sessions"
# Compare to Qdrant hermes_sessions collection count (use curl above)

# Clear sync cache to force full re-sync
rm -rf ~/.hermes/.session_sync_cache  # or wherever HERMES_SYNC_CACHE points
python3 ~/.hermes/scripts/state_db_qdrant_sync.py
```

---

## 9. Rules files

**Where:** `~/.hermes/rules/`
  - `agent-ops.md` — orchestration, parallelism, autonomy consent
  - `quality.md` — investigate before acting, baseline before change, no silent errors
  - `infra.md` — secrets management, change management, least privilege
  - `knowledge.md` — audit first, search before build, context budget

**What they are:** Always-on constraints distilled from skills. Unlike skills
(loaded on demand), rules are injected into every turn by the grounding hook.
Budget: ≤1200 chars total across all rules files.

**Injection order:** agent-ops → quality → infra → knowledge. The hook reads
files in this order and truncates at 1200 chars total. The combined files are
~3200 chars — only the first ~1200 chars (agent-ops + most of quality) are
reliably injected. infra.md and knowledge.md rules may be partially or fully
dropped.

**To ensure security/infra rules are always injected:** either compress
agent-ops.md and quality.md, or reorder files so infra.md comes first.

**Updated via:** Direct file edit. No restart needed — hook reads fresh each turn.

---

## 10. A turn traced end to end

You type: "audit this codebase for dead code"

```
0ms    User message arrives at Hermes gateway

1ms    pre_llm_call hook fires (grounding_client.py via UNIX socket to daemon)
         → payload: {"role":"user","content":"audit this codebase..."}

~5ms   Daemon routes to grounding logic

~75ms  nomic-embed-text on <your-host> returns 768-dim vector

~105ms Parallel Qdrant fan-out (8 workers) completes:
         mnemosyne       → "project is at ~/development/..."
         hermes_sessions → "prior dead code session..."
         ecc_skills      → "dead-code-audit skill"
         agent_core_chunks → "rg --glob '*.ts' orphan file pattern"
         (others below score threshold, filtered)

~110ms Rules files read: agent-ops.md → quality.md → (infra.md if budget allows)

~115ms MEMORY MATCH block assembled, injected into system prompt

~115ms LLM receives:
         [system prompt with rules + persona]
         [MEMORY MATCH block with top results]
         [skills list with 186 skill descriptions]
         [user message: "audit this codebase for dead code"]

~200ms LLM decides: load dead-code-audit skill
         → skill_view("dead-code-audit")
         → pre_tool_call hook fires, logs tool use, allows (read-only)

~300ms LLM has skill content, begins audit using rg commands
         → each tool call fires pre_tool_call hook: log, grounding check

[Session ends]
         → on_session_end hook fires (~140ms)
         → session embedded → hermes_sessions Qdrant
         → available for future session_search queries within minutes
         → mnemosyne-session-summarizer (next 20-min tick) may archive findings
```

---

## 11. Decision guide — when to use what

```
"I need to remember a stable fact across sessions"
  → memory(action='add', target='memory', content='...')
  → Declarative, under 80 chars, no imperatives, no task progress

"I need to start a structured investigation or research session"
  → investigation_start(investigation_id, title, context)
  → Then use investigation_store/investigation_note to track findings
  → Use investigation_search to query what was found

"I need to store a research finding or investigation result"
  → investigation_store(investigation_id, 'observed'|'inferred', text, source, confidence)
  → Automatically writes to JSONL + Mnemosyne + Qdrant hermes_memory

"I need to check if a proposed claim is supported by stored evidence"
  → investigation_pre_answer_check(investigation_id, claims)
  → Validates claims against findings + audit receipts; records verdicts

"I need to clean up a hallucinated fact from memory"
  → investigation_pre_answer_check to identify it, then:
  → memory_retract(investigation_id, target, dry_run=True)  ← review first
  → memory_retract(investigation_id, target, dry_run=False) ← apply
  → memory_restore(investigation_id, finding_id) to undo if needed

"I want to find what I worked on / what I know about X"
  → session_search(query="X")  ← start here (fast, has live sessions)
  → investigation_search(query="X") ← for stored investigation findings
  → mcp_mnemosyne_mnemosyne_recall(query="X") ← for stored episodic facts
  → (grounding hook already did a Qdrant pass automatically)

"I need to check if an IP/hash/CVE appears in any investigation"
  → investigation_entity_lookup(entity="198.51.100.5")
  → investigation_related_cases(entities=["198.51.100.5"])

"I need broad RAG context for a query (not scoped to one investigation)"
  → rag_context_search(query, collections=["hermes_memory", "agent_core_chunks"])

"I want to know how confident hermes_memory is about a topic"
  → memory_confidence(query)

"I need to check the memory substrate health"
  → memory_health()

"I need to navigate code structure — find a symbol, its callers, its type"
  → mcp_serena_activate_project first, then find_symbol / find_referencing_symbols
  → Best for: dead export detection, refactoring, type navigation

"I need to find code by what it does (not what it's named)"
  → CocoIndex MCP semantic search
  → Best for: "where is auth handled", "what validates user input"

"I need to review code from multiple independent angles"
  → deep_think MCP via deep-think-fan-out skill
  → Best for: security reviews, complex bugs, architecture decisions
  → Check service is running first: curl http://127.0.0.1:30852/health

"I need to audit the whole codebase for a pattern"
  → terminal: rg with --glob flags (fast, scriptable)
  → mcp_serena_search_for_pattern (LSP-aware)

"I want to check what's in Qdrant directly"
  → Qdrant MCP tools, or use the curl command from Section 7
  → Or check the pre_llm_call output — if relevant, it appeared already

"I need a constraint to apply on every turn"
  → Edit ~/.hermes/rules/<category>.md
  → Compress other files first to stay within 1200-char budget
  → Prioritize: put high-importance rules in files earlier in injection order
```

---

## 12. Health checks and recovery

### Is the grounding hook working?

```bash
# Check daemon status
systemctl --user status hermes-grounding

# Check daemon logs
journalctl --user -u hermes-grounding -n 30

# Check Ollama is reachable
curl http://<your-host>:11434/api/tags

# Verify Qdrant is reachable
curl -s http://<your-host>:<qdrant-port>/health
```

Signs the hook is working: MEMORY MATCH blocks appear at the top of sessions.
Signs it's not: no memory block in context, or hook takes >5s (timeout).

### Is the loci-mcp (hermes_memory) server healthy?

```
investigation_id=None → memory_health()
```

Returns an 8-probe substrate check. Look for `status: "ok"` in the response.
A `"fail"` on `mnemo_mirror` means the mnemosyne library is not importable in
the server's venv — `investigation_store` will still write JSONL and Qdrant
but won't mirror to Mnemosyne.

### Is session sync working?

```bash
# Check recent sync cron output
ls -lt ~/.hermes/cron/output/ | head -10

# Manual run to check for errors
python3 ~/.hermes/scripts/state_db_qdrant_sync.py

# Verify sessions are in Qdrant
# (point count should grow after sessions complete)
```

### Is Mnemosyne synced to Qdrant?

```bash
# Check the 30-min sync cron last ran
ls -lt ~/.hermes/cron/output/ | grep mnemosyne-qdrant

# Manual sync
python3 ~/.hermes/scripts/mnemosyne_qdrant_sync.py

# Check Mnemosyne stats
# In-session: mcp_mnemosyne_mnemosyne_get_stats()
```

### Qdrant collection empty or missing — rebuild

```bash
# Check all collections
curl -s http://<your-host>:<qdrant-port>/collections \
  -H "api-key: <key>" | python3 -m json.tool

# Rebuild hermes_sessions: clear cache and re-run sync
rm -rf ~/.hermes/.session_sync_cache
python3 ~/.hermes/scripts/state_db_qdrant_sync.py

# Rebuild mnemosyne collection:
python3 ~/.hermes/scripts/mnemosyne_qdrant_sync.py

# Rebuild hermes_memory: re-run loci backfill if available
# (hermes_memory is created lazily on first investigation_store call)
```

### Open Design MCP not working

```bash
# Check container
docker ps | grep open-design

# Start if needed
cd ~/open-design/deploy && docker compose up -d

# Check OD_API_TOKEN is set (from .env or config.yaml)
grep OD_API_TOKEN ~/.hermes/.env
```

### Check what rules are actually being injected

The hook truncates at 1200 chars reading files in order. To see what gets in:

```bash
cat ~/.hermes/rules/agent-ops.md | wc -c
cat ~/.hermes/rules/quality.md | wc -c
cat ~/.hermes/rules/infra.md | wc -c
cat ~/.hermes/rules/knowledge.md | wc -c
# Sum until you hit 1200 — everything after that is cut
```

---

## 13. Configuration reference

**Profile root:** `~/.hermes/`

**Key paths:**
```
config.yaml                    — main config (model, hooks, MCP servers)
.env                           — env var overrides for hooks (QDRANT_URL,
                                 QDRANT_API_KEY, OLLAMA_BASE_URL, etc.)
memories/MEMORY.md             — environment facts, always injected
memories/USER.md               — user profile facts, always injected
rules/agent-ops.md             — orchestration rules
rules/quality.md               — verification and correctness rules
rules/infra.md                 — infrastructure and security rules
rules/knowledge.md             — skill/knowledge management rules
skills/                        — 186 skill files across 26 categories
mcp/server.py                  — loci-mcp server (hermes_memory MCP, 25 tools)
scripts/grounding_client.py    — pre_llm_call hook entry (UNIX socket client)
scripts/grounding_daemon.py    — persistent daemon (systemd --user)
scripts/hooks/pre_llm_grounding.py  — grounding logic v3
scripts/hooks/pre_tool_grounding.py — write-gate + audit log hook
scripts/hooks/session_end_sync.py   — on_session_end Qdrant sync
scripts/state_db_qdrant_sync.py     — 5-min catch-all cron sync
scripts/mnemosyne_qdrant_sync.py    — 30-min Mnemosyne → Qdrant sync
scripts/mnemosyne_activity_check.py — gate script for consolidation crons
scripts/mnemosyne_sleep_all.sh      — CLI-level sleep script
cron/jobs.json                 — all 5 cron job definitions
state.db                       — SQLite session store (session_search queries this)
mnemosyne/data/                — Mnemosyne SQLite + embeddings (local)
hermes.db                      — Hermes internal state
SOUL.md                        — profile identity/persona definition
```

**MCP servers wired (from config.yaml):**
```
hermes_memory (loci): stdio via mcp/server.py              (this repo)
  env: QDRANT_URL, OLLAMA_BASE_URL, QDRANT_API_KEY, HERMES_AGENT_ID, LOCI_NAMESPACE
deep_think:           http://127.0.0.1:30852/mcp           (systemd --user)
mnemosyne:            stdio via mnemosyne mcp               (local venv)
qdrant:               stdio via mcp-server-qdrant           (local venv → <your-host>)
cocoindex_code:       stdio via ccc mcp                     (~/.local/bin/ccc)
serena:               stdio via serena start-mcp-server     (~/.local/bin/serena)
open_design:          stdio via od-mcp.sh                   (Docker container)
```

**Embedding model:** `nomic-embed-text` on `<your-host>` Ollama, 768-dim.
All collections use this same model. Never mix embedding models across
collections — cosine similarity breaks.

**loci-mcp environment variables:**

| Variable | Purpose | Default |
|----------|---------|---------|
| `HERMES_MEMORY_DIR` | Root for investigation sessions | `~/.hermes/memory-sessions` |
| `QDRANT_COLLECTION_PREFIX` | Main Qdrant collection name | `hermes_memory` |
| `MNEMOSYNE_EMBEDDING_DIM` | Dense vector dimension | `768` |
| `EMBED_MODEL` | Ollama model name for embeddings | `nomic-embed-text` |
| `OLLAMA_BASE_URL` | Ollama endpoint (OpenAI-compat `/v1/embeddings`) | unset |
| `EMBED_API_KEY` | API key for cloud embedding providers | unset |
| `EMBED_API_KEY_HEADER` | Header name for the above key | `Authorization` |
| `QDRANT_URL` | Qdrant server URL | unset (disables vector search) |
| `QDRANT_API_KEY` | Qdrant API key | unset |
| `HERMES_MNEMO_BANK` | Mnemosyne bank for mirroring | `default` |
| `CODE_CHUNKS_COLLECTION` | Qdrant collection for code-chunk correlation | unset |
| `ROUTING_DECISIONS_COLLECTION` | Qdrant collection for `dama_routing_query` | unset |
| `HERMES_AGENT_ID` | Agent identity stamp on Qdrant points | unset |
| `LOCI_NAMESPACE` | Namespace stamp on Qdrant points | unset |
| `FASTEMBED_MODEL` | Fallback local embedder (used if Ollama is unavailable) | `BAAI/bge-small-en-v1.5` |
| `HERMES_REFLECTION_INVESTIGATION` | Default investigation id for reflection loop | `copilot-self-reflection-loop` |
| `HERMES_MCP_TRANSPORT` | Server transport: `stdio`, `sse`, `streamable-http` | `stdio` |
| `HERMES_MCP_HOST` | Bind host for HTTP transports | `0.0.0.0` |
| `HERMES_MCP_PORT` | Port for HTTP transports | `8000` |

**Warning:** The fastembed fallback (`BAAI/bge-small-en-v1.5`) produces 384-dim
vectors. If `MNEMOSYNE_EMBEDDING_DIM` is set to 768 (the default) and Ollama
becomes unavailable, fastembed upserts will fail silently with a dimension
mismatch. Either set `EMBED_DIM=384` when using fastembed, or restore Ollama.

**loci-mcp tool index (`mcp/server.py` — 25 `@mcp.tool()` functions):**

| Tool | Brief description |
|------|------------------|
| `investigation_start` | Create or resume an investigation by ID. Idempotent — resuming returns the current manifest without overwriting. |
| `investigation_load` | Retrieve manifest and recent findings for context recovery at session start. Retracted findings excluded by default. |
| `investigation_store` | Record a finding (observed/inferred/assumed/gap). Writes to JSONL, Mnemosyne, and Qdrant `hermes_memory`. |
| `investigation_note` | Update manifest fields: context, hypothesis, next_step, open questions, checked sources, or close the investigation. |
| `investigation_reflect` | Synthesize current investigation state — finding breakdown, open questions, gaps, and advisory self-check. |
| `investigation_search` | Hybrid Mnemosyne+Qdrant search over findings with cross-encoder reranking. Retracted findings excluded by default. |
| `investigation_pre_answer_check` | Validate proposed response claims against stored findings and audit receipts. Records verdicts to `hermes_verdicts`. |
| `investigation_evidence_precheck` | Lightweight duplicate-call avoidance — checks if similar evidence already exists before running a new tool call. |
| `investigation_entity_lookup` | Find every finding mentioning a specific IP, email, hostname, hash, or CVE. O(1) via Qdrant payload indexes. |
| `investigation_related_cases` | Find prior investigations that dealt with the same entities. Call before opening a new investigation. |
| `investigation_finding_provenance` | Trace a finding back through `derived_from` links to root observed evidence. |
| `investigation_list` | List all investigations with status and finding counts, most-recently-updated first. |
| `audit_log` | Record a tool call and full output to global daily JSONL, investigation JSONL, Mnemosyne, and Qdrant. |
| `memory_self_check` | Advisory provenance and contradiction checks over stored findings. Surfaces hallucination candidates. Never auto-retracts. |
| `memory_retract` | Soft-tombstone a hallucinated finding and its contaminated lineage. `dry_run=True` (default) previews without changing anything. |
| `memory_restore` | Reverse a retraction — un-tombstone a finding so it returns to recall/search/reflect. |
| `memory_health` | 8-probe substrate self-check: Qdrant reachability, collections, embedders, dimension consistency, retraction integrity, store counts. |
| `memory_consolidate` | Run Mnemosyne sleep/consolidation cycle — merges working_memory into episodic memory. |
| `memory_confidence` | Metamemory: 5-cue calibrated confidence score (fluency, accessibility, source_diversity, corroboration, trust). |
| `code_memory_correlate` | Link a code hallucination to contaminated memories. Advisory and read-only. |
| `rag_context_search` | Hybrid Qdrant RAG with cross-encoder reranking. Returns prompt-ready context with `[SOURCE N]` citations. Requires Qdrant. |
| `dama_routing_query` | Query DAMA routing decisions by structured payload filters. Requires `ROUTING_DECISIONS_COLLECTION` env var. |
| `reflection_loop_seed` | Enqueue Copilot artifacts (session events, process logs, temp_ingest) into the reflection queue. |
| `reflection_loop_tick` | Process a small queue batch. Deterministic parsing only — no LLM pass. Writes findings via `investigation_store`. |
| `reflection_loop_status` | Inspect reflection queue size, processed count, and aggregate stats. |

**Grounding hook parameters (tunable in config.yaml hooks stanza or .env):**
```
HOOK_RECALL_TOP_K=5           max results injected per turn
HOOK_RECALL_MIN_SCORE=0.55    cosine similarity threshold
HOOK_RECALL_MIN_IMPORTANCE=0.2 importance weight threshold
HOOK_RECALL_MAX_CHARS=200     max chars per memory match result
HOOK_RULES_MAX_CHARS=1200     total chars budget for rules injection
HOOK_EMBED_TIMEOUT=3.0        Ollama embed call timeout (seconds)
HOOK_QDRANT_TIMEOUT=2.0       per-collection Qdrant query timeout
HOOK_QDRANT_WORKERS=8         parallel workers for fan-out
BLOCK_MODE=0                  0=log only, 1=block dangerous commands
```

---

## 14. Pitfalls

### 1. Memory.md written as instructions, not facts
Writing "Always check middleware.ts before auditing auth" causes it to re-read
as a permanent directive overriding the current task. Write the fact instead:
"Next.js only runs middleware.ts; proxy.ts named wrong was dead code."
Instructions → rules files. Procedures → skills. Facts → MEMORY.md.

### 2. Ollama fallback is best-effort, not guaranteed
If `<your-host>` is unreachable, the hook falls back to BeamMemory FTS.
This requires the mnemosyne library to be importable in the hook's Python env.
If mnemosyne isn't installed or fails to import, the fallback itself fails.
"You won't notice" is optimistic — check journalctl if grounding seems absent.

### 3. Grounding daemon requires explicit start
`systemctl --user start hermes-grounding` — it does not auto-start
without `systemctl --user enable`. On a new machine, enable it first.
Without it, grounding still works via the fallback subprocess path (~140ms).

### 4. deep_think MCP is a --user service, not --system
`systemctl --user status deep-think-mcp`. Not running after reboot unless
enabled: `systemctl --user enable deep-think-mcp`. Check the actual unit name:
`systemctl --user list-units | grep -i think` — name may vary.

### 5. Serena needs project activation every session
`mcp_serena_activate_project("~/development/project")` before any symbol
navigation. Without it, all Serena find/search tools return empty results.

### 6. Writing task progress to MEMORY.md
"Fixed PR #28, merged June 14" does not belong in MEMORY.md. Stale in a week,
pollutes always-injected context. Use session_search or investigation_store for
episodic. MEMORY.md is for stable, durable facts that are still relevant in 6 months.

### 7. CocoIndex index becomes stale after code changes
After major refactors, run `ccc index <project_path>` to rebuild. No automatic
re-indexing. Stale results look like searches referencing deleted functions.

### 8. Rules budget silently truncates infra/knowledge rules
The 1200-char budget fills up reading agent-ops.md + quality.md first.
infra.md (secrets management, change management, least privilege) is often
partially or fully cut. High-importance rule not getting injected? Move it
to an earlier file or compress the files ahead of it.

### 9. Collection point counts in this doc are stale snapshots
Point counts change continuously. Use the health check curl command in
Section 12 for live counts.

### 10. hermes_memory is not populated by direct Qdrant MCP writes
The `hermes_memory` Qdrant collection is written by loci-mcp (`mcp/server.py`)
via `investigation_store`, `audit_log`, and `reflection_loop_tick`. Do not
use Qdrant MCP tools to write directly to it — the payload schema and payload
indexes (entities, investigation_id, confidence, etc.) will be missing.
Query it via `investigation_search` or `rag_context_search`, not raw Qdrant.

### 11. Open Design MCP silently fails when container is down
No error is surfaced in the session — tools just return empty. If you expect
OD tools to work, verify `docker ps | grep open-design` first.

### 12. .env file path confusion
The hook reads `~/.hermes/.env` but the shell environment may
have `MNEMOSYNE_DATA_DIR` set to the default profile path. Scripts called
outside the hook (manual runs, cron) use shell env, not the hook's .env.
Always pass env vars explicitly when running hook scripts manually.

### 13. Dangerous command detection is advisory by default
`BLOCK_MODE=0` means the pre_tool_call hook logs dangerous commands but does
NOT block them. An `rm -rf` or `kubectl delete` will proceed. This is intentional
for development flexibility but means the "dangerous command detection" is a
record, not a safety net. Set `BLOCK_MODE=1` if you want enforcement.

### 14. fastembed fallback produces 384-dim vectors
When Ollama is unavailable, the loci-mcp server falls back to
`BAAI/bge-small-en-v1.5` (384-dim). If the `hermes_memory` collection was
created with 768-dim vectors, upserts will fail and new findings won't be
indexed to Qdrant. The JSONL and Mnemosyne writes still succeed — no data is
lost, but semantic search will degrade. Restore Ollama or align dimensions.

### 15. memory_retract dry_run=True is the default — nothing changes until False
`memory_retract` defaults to `dry_run=True` intentionally. The first call
always returns a proposed cluster for review. You must explicitly re-run with
`dry_run=False` to apply the soft tombstone. Nothing is irreversible —
`memory_restore` undoes any retraction by appending `active:false` to the log.
