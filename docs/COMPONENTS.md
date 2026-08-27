# Component Reference

Components live in four trees: `scripts/` (cron- and hook-driven jobs), `mcp/`
(the MCP server and its libraries), `eval/`, and `deep_think_loci/`. Each entry
covers purpose, inputs, outputs, config env vars, and schedule (if any).

Three schedulers are declared in this repo and only two of them fire:

- **`cron/jobs.json`** — six agent jobs, five marked `"enabled": true`. All six
  carry `"last_run_at": null` and `"last_status": null` (read 2026-08-27).
  Nothing on this host reads the file (issue #205). Treat an entry here as a
  declaration, not a running job.
- **the user crontab** — four `loci_groom_cron.sh` passes. Verified running:
  `~/.loci/groom/runs.jsonl` has an `rc:0` line for each within the last 24h.
- **a systemd user timer** — `mrpink-context-bridge.timer`, every 10m, last
  fired 2026-08-27 12:39 EDT.

---

## Grounding pipeline

### `scripts/hooks/pre_llm_grounding.py`
**Purpose:** Per-turn grounding. Embeds user message intent, fans out to 3 base Qdrant
collections (`mnemosyne`, `loci_sessions`, `loci_memory`) plus any named in
`GROUNDING_EXTRA_COLLECTIONS`, in parallel; ranks by a four-term weighted score
(relevance / recency / trust / type) with MMR diversification and stigmergic
pheromone reinforcement, then injects MEMORY MATCH context.

**Accepted events:** `pre_llm_call`, `PreLlmCall` (legacy Hermes) and
`UserPromptSubmit`, `SubagentStart` (Claude Code). Anything else exits 0.

**Message field:** top-level `prompt`, falling back to `extra.user_message`.
Before #228 only the nested form was read, so under Claude Code every turn
grounded on an empty string and exited early.

**Installed by:** `scripts/hooks/install.sh` (copies into `~/.claude/hooks`;
`--check` reports drift between repo and deployed copy). It can also be reached
through `grounding_client.py`, which proxies to the warm daemon.

**Key env vars:**
- `QDRANT_URL` (no default — Qdrant fan-out is skipped if unset)
- `OLLAMA_BASE_URL` (bare host; `/v1` is appended) or `MNEMOSYNE_EMBEDDING_API_URL`
  (already a full `/v1` endpoint). With neither set, embedding is disabled and
  recall falls back to BeamMemory FTS.
- `QDRANT_API_KEY`
- `MNEMOSYNE_EMBEDDING_MODEL` (default: `nomic-embed-text`)
- `GROUNDING_EXTRA_COLLECTIONS` (default: empty) — comma-separated extra collections
  to search alongside the 3 base ones
- `HOOK_RECALL_TOP_K` (default: `3`; the shipped `.env.example` sets `5`)
- `HOOK_RECALL_MIN_SCORE` (default: `0.55`)
- `HOOK_RECALL_MIN_IMPORTANCE` (default: `0.2`)
- `HOOK_QDRANT_WORKERS` (default: `8`)
- `HOOK_RANKER_W_RELEVANCE` / `_RECENCY` / `_TRUST` / `_TYPE`
  (defaults: `0.50` / `0.20` / `0.15` / `0.15`; must sum to 1.0)
- `HOOK_MMR_LAMBDA` (default: `0.75`), `HOOK_RECENCY_HALFLIFE_DAYS` (default: `7`)

**Output:** `{"context": "MEMORY MATCH (...) for ...:\n..."}` on stdout

---

### `scripts/grounding_client.py`
**Purpose:** Thin UDS socket client. Connects to the `grounding_daemon.py` socket.
Falls back to direct subprocess on socket failure.

**Socket:** `$GROUNDING_SOCK`, default `/tmp/hermes-grounding-$HERMES_AGENT_ID.sock`
(`$HERMES_AGENT_ID` itself defaults to `hermes`). The path is per-agent, not shared.

**Input:** JSON payload on stdin from whatever the hook is wired to
**Output:** Proxied output from `pre_llm_grounding.py`

---

### `scripts/grounding_daemon.py`
**Purpose:** Long-running daemon that keeps `pre_llm_grounding.py` warm in memory,
eliminating Python startup cost (~80ms) from grounding latency.

**Socket:** same path as the client, chmod 0600, with a PID file alongside.

---

### `scripts/hooks/pre_tool_grounding.py`
**Purpose:** PreToolUse guard, not just an audit sink. Three tiers:

1. Read-only / grounding tools (`GROUNDING_TOOLS` — Read, ToolSearch, WebFetch,
   the Mnemosyne and Serena read tools, …) pass silently with no audit row.
2. Mutation tools (`MUTATION_TOOLS` — Edit, Write, MultiEdit, `write_file`, the
   Serena editing tools) get their write targets and content scanned for
   supply-chain path IOCs and prompt-injection patterns. Writes to agent config
   paths (`AGENTS.md`, `CLAUDE.md`, `.cursorrules`, …) block on any injection hit
   regardless of `HOOK_BLOCK_MODE`.
3. Everything else is audited; dangerous terminal patterns (`rm -rf`, `DROP TABLE`,
   `git push --force` without `--force-with-lease`, `mkfs`, `dd if=`, pipe-to-
   interpreter, …) are always logged and blocked when `HOOK_BLOCK_MODE=1`.

**Accepted events:** `pre_tool_call`, `PreToolUse`.
**Key env vars:** `HOOK_BLOCK_MODE` (default: `0` — audit only)
**Wire out:** `{}` to allow, `{"action":"block","message":"..."}` to block.

---

### `scripts/hooks/session_end_sync.py`
**Purpose:** Live-syncs the current session **into Qdrant** `loci_sessions` — it
reads Hermes state, it does not write it. Fires at the end of every turn, not only
at session end. Resolves session content from `state.db` first and falls back to
the `transcript_path` the Claude Code Stop payload carries; before #228 only the
`state.db` lookup existed, and Claude Code session UUIDs are not in that database,
so the hook had never synced a session.

Fast path: a session with no new messages since the last upsert (mtime cache under
`$LOCI_SYNC_CACHE`, default `~/.hermes/.session_sync_cache`) exits 0 immediately.

**Key env vars:** `LOCI_STATE_DB` (default `~/.hermes/state.db`), `QDRANT_URL`,
`QDRANT_API_KEY`, `OLLAMA_BASE_URL` or `MNEMOSYNE_EMBEDDING_API_URL`,
`MNEMOSYNE_EMBEDDING_MODEL` (default `nomic-embed-text`),
`MNEMOSYNE_EMBEDDING_DIM` (default `768`)

**Latency budget:** <500ms.

---

## MCP server (`mcp/`)

### `mcp/server.py`
**Purpose:** the FastMCP server. 8,433 lines; 42 tools defined here and 31 more
registered by the submodules below, for **73 tools total** (counted by calling
`server.mcp.list_tools()`).

**Transport:** `LOCI_MCP_TRANSPORT` — `stdio` (default), `sse`, or
`streamable-http`. For the HTTP transports:
- `LOCI_MCP_HOST` defaults to `127.0.0.1` (#206). A non-loopback bind without
  `LOCI_MCP_TOKEN` is refused with a `SystemExit`, because it would expose every
  tool unauthenticated. `docker-compose.yml` sets `0.0.0.0` deliberately and
  publishes on loopback.
- `LOCI_MCP_TOKEN` enables `_BearerAuthMiddleware` (#211).
- `LOCI_MCP_PORT` defaults to `8000`.

**Startup:** fires a non-blocking `embed_ops.warm()` ping so the first RAG call
does not eat the ~9s nomic cold-load.

**Retention:** `LOCI_QDRANT_RETENTION_DAYS` defaults to **0 — purge disabled**
(#204). It used to default to 30, which silently deleted every indexed finding
older than a month on each server start; measured on the live store at the time,
the index held 912 findings against a corpus of 2,831 and the split was exactly
the 30-day boundary. Resolution order: env, then `~/.loci/backends.toml`, then 0.
A non-integer value disables the purge rather than guessing a window.

**Tool modules** (each exposes `register(mcp, ...)`, injected rather than importing
`server`, which is what keeps the imports acyclic):
- `investigation_tools.py` — 11 tools: start/load/as_of/note/reflect/provenance/
  list/share/unshare/export/import
- `graph_tools.py` — 11 tools: `code_graph_ingest`, `code_graph_query`,
  `code_memory_relink`, `code_memory_map`, `symbol_impact`, `impact_report`,
  `finding_code_context`, `investigation_code_briefing`, `subsystem_report`,
  `related_investigations_via_code`, `dead_code_candidates`
- `llm_tools.py` — 9 tools: `llm_local`, `generate_batch`, `query_expand`,
  `verify_finding`, `classify_text`, `compress_text`, `semantic_dedup`,
  `semantic_relevance`, `ground`

**Supporting libraries:**

| Module | Role |
|---|---|
| `backends.py` | portable backend resolution (`~/.loci/backends.toml`) so an install works unchanged on another machine |
| `qdrant_ops.py` | embedding + vector-store helpers, retention window, quantized-search params (#220) |
| `embed_ops.py` | the embedding tier (Ollama nomic, 768d) — the offload that works today |
| `llm_local.py` | local-GPU generation primitive; vLLM fallback is opt-in via `LOCI_VLLM_FALLBACK`, default OFF (#222/#224/#225) |
| `batched_gen.py` | concurrent fan-out generation with Ollama fallback |
| `openrouter.py` | generation tier whose availability is not this node's |
| `grounding.py` | `ground(task)` — run ONCE in the main loop before a fan-out; the gate corroborates the semantic lane rather than trusting a bare 0.55 (#221) |
| `query_expand.py` | HyDE-lite retrieval query expansion |
| `reranker.py` | pluggable two-stage GPU reranker |
| `retrieval_eval.py` | retrieval shadow-eval, so gated upgrades flip on evidence |
| `inv_store.py` | on-disk investigation store |
| `ladybug_ops.py` | stateless LadybugDB leaves |
| `verify.py` | adversarial candidate → skeptic → keep-if-survives loop |
| `verdict_ops.py` | claim-verdict recording |
| `mnemo_ops.py` | fail-open wrappers around the optional `mnemosyne` package |
| `text_ops.py`, `model_json.py` | cheap generation-backed text ops; defensive JSON extraction from noisy model output |

**`mcp/graph/`** — the code↔memory graph: `code_parse.py` (tree-sitter symbol and
reference extraction), `ladybug_store.py` (embedded Kuzu, two overlaid graphs),
`queries.py` (composable primitives), `analytics.py` (higher-level reports),
`linker.py` (precision-focused Finding→CodeSymbol `REFERENCES` edges).
`causal_infer` and `derived_from` lineage edges landed in #218/#212.

**`mcp/memcheck/`** — the PreToolUse audit lane. `hook_client.py` is the wired
entrypoint (ultra-thin); `cli.py` is the fallback and the human-facing
`tail`/`stats`; `daemon.py` is the warm half. `check-action` is structurally
audit-only: it can never exit non-zero and never emits `{"decision":"approve"}`,
so it observes without bypassing the permission prompt or blocking a tool.
`memcheck` reports the audit lane's state rather than a bare count (#213).

---

## Consolidation and decay

### `scripts/ebbinghaus_consolidation.py`
**Purpose:** Forgetting-curve-triggered memory refresh. Reads `working_memory` and
`episodic_memory`, computes retention R = exp(-t/S) for each entry using an
FSRS-inspired stability model, re-embeds and upserts decayed entries (R < threshold)
to the Qdrant `mnemosyne` collection, then updates SQLite recall metadata so the
next forgetting window resets.

The stability S is computed from FSRS DSR parameters:
- Initialized as `S = base * (1 + recall_count)^(1/D)` where D is difficulty derived from importance.
- Success update: `S' = S * exp(w1 * (11-D) * (exp(w2*(1-R))-1) + 1)`
- Failure update: `S' = S * FSRS_DECAY_FACTOR`

**Cron:** No configured schedule — run on demand.

**Key env vars:**
- `MNEMOSYNE_DB` (default: `~/.hermes/mnemosyne/data/mnemosyne.db`)
- `QDRANT_URL`
- `OLLAMA_URL`
- `EMBED_MODEL` (default: `nomic-embed-text`)
- `FORGET_THRESH` (default: `0.3`) — entries with R below this are refreshed
- `EBBINGHAUS_MAX_PER_RUN`, falling back to `MAX_PER_RUN` (default: `50`)
- `FSRS_W1` (default: `0.4`) — success stability growth rate
- `FSRS_W2` (default: `0.6`) — retrievability factor in success update
- `FSRS_DECAY_FACTOR` (default: `0.5`) — failure stability penalty multiplier
- `FSRS_DIFF_INIT` (default: `5.0`) — neutral starting difficulty

**Research basis:** Ebbinghaus (1885), FOREVER technique (ACL 2026, arxiv 2601.03938),
FSRS (open-spaced-repetition/free-spaced-repetition-scheduler)

---

### `scripts/amem_consolidation.py`
**Purpose:** A-MEM cross-link discovery and conflict detection. Loads recent
`working_memory` entries, embeds each via Ollama, computes pairwise cosine similarity.
Writes three kinds of output, all to SQLite only — no Qdrant dependency:
semantic links to `graph_edges` (sim > link threshold), `updated_by` edges for
near-copies with a temporal ordering (sim > update threshold), and conflict flags
to `conflicts` (sim > conflict threshold and opposing keyword pairs).

Supports an optional QuorumGate mechanism: set `QUORUM_AMEM_THRESHOLD > 0` to require
a minimum accumulated signal count before the expensive embedding pass runs.

**Cron:** No configured schedule — run on demand.

**Key env vars:**
- `MNEMOSYNE_DB` (default: `~/.hermes/mnemosyne/data/mnemosyne.db`)
- `OLLAMA_URL` (default: `http://localhost:11434`)
- `EMBED_MODEL` (default: `nomic-embed-text`)
- `AMEM_LINK_THRESHOLD` (default: `0.88`) — cosine similarity threshold for cross-links
- `AMEM_UPDATE_THRESHOLD` (default: `0.92`) — near-copy threshold for `updated_by`
- `AMEM_CONFLICT_THRESHOLD` (default: `0.96`) — threshold for conflict detection
- `AMEM_MAX_PER_RUN`, falling back to `MAX_PER_RUN` (default: `100`)
- `QUORUM_AMEM_THRESHOLD` (default: `0`) — minimum accumulated signal before running; 0 disables

**Research basis:** A-MEM (arxiv 2502.12110, Feb 2025)

---

### `scripts/mnemosyne_qdrant_sync.py`
**Purpose:** Syncs all Mnemosyne memories to the Qdrant `mnemosyne` collection.
Embeds via Ollama `/v1/embeddings` (`MNEMOSYNE_EMBEDDING_API_URL`) and upserts directly
into the `mnemosyne` collection. Runs incrementally to avoid re-uploading already-synced
entries.

**Cron:** declared as `mnemosyne-qdrant-sync` (every 30m) in `cron/jobs.json`, which
has never run. No live schedule.

**Key env vars:** `QDRANT_URL`, `MNEMOSYNE_EMBEDDING_API_URL`, `MNEMOSYNE_EMBEDDING_MODEL`
(default: `nomic-embed-text`), `MNEMOSYNE_DATA_DIR` (default: `~/.hermes/mnemosyne/data`),
`EMBED_WORKER_URL` (no default; read but never referenced again — this script embeds
via Ollama only)

---

### `scripts/state_db_qdrant_sync.py`
**Purpose:** Syncs Hermes `state.db` sessions to the Qdrant `loci_sessions` collection.
Incremental (tracks already-synced session_ids in payload). Chunks per-session messages
to 4000 chars before embedding. Unlike the Mnemosyne sync, this one does use
`EMBED_WORKER_URL` — it POSTs to `$EMBED_WORKER/embed`.

**Cron:** declared as `state-db-qdrant-sync` (every 5m) in `cron/jobs.json`, which
has never run. No live schedule.

**Key env vars:** `LOCI_STATE_DB` (default: `~/.hermes/state.db`),
`QDRANT_URL`, `QDRANT_API_KEY`, `EMBED_WORKER_URL`, `MNEMOSYNE_EMBEDDING_DIM`
(default `768`)

---

## Self-improvement

### `scripts/agentHER_relabeler.py`
**Purpose:** AgentHER hindsight relabeling. Reads failure memories from `working_memory`
(last 7 days, importance ≥ 5), relabels each via Ollama generate ("This trace shows
how to..."), stores synthetic positives back to both Mnemosyne SQLite and the Qdrant
`mnemosyne` collection.

**Cron:** No configured schedule — run on demand.

**Key env vars:**
- `MNEMOSYNE_DB` (default: `~/.hermes/mnemosyne/data/mnemosyne.db`)
- `QDRANT_URL`
- `OLLAMA_URL`
- `EMBED_MODEL` (default: `nomic-embed-text`)
- `AGENTHER_GEN_MODEL` (default: `llama3.2:latest`)
- `AGENTHER_MAX_PER_RUN`, falling back to `MAX_PER_RUN` (default: `20`)

**Research basis:** AgentHER (arxiv 2603.21357, Apr 2026)

---

### `scripts/skill_annotation_updater.py`
**Purpose:** DRAFT self-annotation. Reads `guard_tool_reflections.log` (and optionally
`guard_bash_failures.log`) from hook state, aggregates failures by tool_name, finds
matching SKILL.md files, writes or updates "## Learned constraints" sections with
top-3 failure patterns.

**Cron:** No configured schedule — run on demand.

**Key env vars:**
- `STATE_DIR` (default: `~/.claude/hook-state`)
- `SKILLS_DIR` (default: `~/.claude/skills`)
- `MIN_USES` (default: `3`)

**Research basis:** DRAFT technique (arxiv 2410.08197, ICLR 2025 Oral)

---

### `scripts/exif_skill_discovery.py`
**Purpose:** EXIF closed-loop skill discovery. Alice (Ollama) analyzes recent failure
memories and existing skill names to identify a gap. Bob generates a candidate SKILL.md
written to `STATE_DIR/candidate_skills/{skill_name}/`. Never auto-promotes — all
candidates require human review before promotion.

**Run on demand** (not croned — human review required before promoting candidates)

**Key env vars:**
- `EXIF_GEN_MODEL` (default: `llama3.2:latest`)
- `STATE_DIR` (default: `~/.claude/hook-state`)
- `SKILLS_DIR` (default: `~/.claude/skills`)
- `MNEMOSYNE_DB` (default: `~/.hermes/mnemosyne/data/mnemosyne.db`)
- `OLLAMA_URL` — Ollama base URL (required; no default)
- `DISCOVERY_LOG` (default: `STATE_DIR/exif_discoveries.jsonl`)

**Research basis:** EXIF (arxiv 2506.04287, Jun 2025)

---

### `scripts/score_trace_collector.py`
**Purpose:** SCoRe data pipeline. Reads bash failure/success logs and AgentHER
positives from Mnemosyne. Builds `negatives.jsonl`, `positives.jsonl`, `corrections.jsonl`
in `~/.hermes/mnemosyne/data/score_traces/`. Upserts correction pairs to Qdrant
`score_traces` collection. When `n_corrections >= 10`, sets `ready_for_sft: true` in
`manifest.json`; the manifest is rewritten afterwards with the count of corrections
that actually upserted, so dataset and manifest cannot disagree.

**Run on demand**

**Key env vars:** `STATE_DIR`, `OUTPUT_DIR`, `MNEMOSYNE_DB`, `QDRANT_URL`,
`OLLAMA_URL`, `EMBED_MODEL`

**Research basis:** SCoRe (arxiv 2409.12917, Google DeepMind, ICLR 2025)

---

### `scripts/skillops_maintenance.py`
**Purpose:** SkillOps library maintenance. Scans all SKILL.md files under `~/.claude/skills`
and `~/.hermes/skills`. Embeds each description via Ollama. Computes pairwise cosine
similarity. Reports SHADOW_RISK pairs above threshold. Updates `last_validated` date in
all SKILL.md frontmatter.

**Run on demand** (or weekly; no cron currently)

**Key env vars:**
- `SHADOW_THRESHOLD` (default: `0.92`)
- `OLLAMA_URL`, `EMBED_MODEL`

**Research basis:** Skill Shadowing (arxiv 2605.24050, May 2026)

---

## Multi-level search

### `scripts/memgas_hierarchy.py`
**Purpose:** MemGAS 3-level memory search. Indexes L1 (working_memory), L2
(episodic_memory), L3 (consolidated_facts) into Qdrant collections `memgas_l1/l2/l3`.
Search query is embedded, all 3 levels searched in parallel, entropy-weighted fusion
applied: `weight = 1 / (1 + entropy(softmax(scores)))`.

**Commands:**
- `python3 memgas_hierarchy.py --index` — build/refresh all 3 Qdrant collections
- `python3 memgas_hierarchy.py --search <query>` — entropy-weighted 3-level search

**Key env vars:**
- `MNEMOSYNE_DB` (default: `~/.hermes/mnemosyne/data/mnemosyne.db`)
- `QDRANT_URL`
- `OLLAMA_URL`
- `EMBED_MODEL` (default: `nomic-embed-text`)
- `TOP_K_PER_LEVEL` (default: `3`) — results returned per memory level

**Research basis:** MemGAS (arxiv 2505.19549, May 2025)

---

## Passive grooming tier

The only scheduled work on this host that demonstrably fires.

### `scripts/loci_groom.py`
**Purpose:** unattended maintenance of the Loci corpus. Every pass is idempotent
(a second run over unchanged input proposes nothing new), fail-open (a dead backend
degrades the pass to a report rather than raising), and shadow-first: model-derived
output goes to `_groom/proposals.jsonl` with its provenance and is **never** merged
into `findings.jsonl`.

`--apply` promotes only for passes declaring `applyable`, which today is `index`
alone — its write is a re-upsert of the record already on disk, so it can restore
but cannot invent.

**Passes** (`PASSES` in the module):

| Pass | Applyable | What it does |
|---|---|---|
| `index` | yes | reconcile the vector index against `findings.jsonl`; upsert-only, never deletes |
| `tags` | no | propose canonical tags via the local model |
| `recall` | no | sampled retrieval scoring |
| `knn_tags` | no | propose tags by kNN vote over neighbours |
| `codelink` | no | propose finding→symbol links |
| `verify` | no | adversarial re-verification of findings |
| `reflect` | no | seed the reflection loop from session artifacts |
| `summaries` | no | drive the investigation summary ladder |

`tags` is deliberately not scheduled: kNN tagging measured better at a fraction of
the cost, so running both would spend a model to do worse.

`load_env()` resolves backend config the way the server does — existing env, then
the repo `.env` files, then `~/.loci/backends.toml` — because a cron job does not
inherit the MCP launcher's environment. A failed `python-dotenv` import is logged
loudly, not swallowed — per the code's own note, `LOCI_QDRANT_RETENTION_DAYS=0` is
set in the (gitignored) `.env`, so swallowing the import is how a scheduled run
turns destructive while reporting success. It is not in `.env.example` or
`backends.toml.example`; the safe value comes from `_retention_days()`'s default.

### `scripts/loci_groom_cron.sh`
**Purpose:** the cron entrypoint. Appends one line per run to
`$LOCI_GROOM_STATE/runs.jsonl` (default `~/.loci/groom/runs.jsonl`) so "has this
ever actually run, and what did it say" has an answer.

**Exit codes:** `0` ok, `1` a pass errored, `3` a pass refused or degraded. A `3`
prints the pass's own reason on stderr so cron mails it.

**Live schedule and last measured result (2026-08-27):**

| Pass | Schedule | Result |
|---|---|---|
| `index --apply` | `17 */6 * * *` | ok, `on_disk=2922 indexed=2769 missing=153 coverage=0.9476` |
| `knn_tags` | `20 3 * * *` | ok, `candidates=360 generated=18 proposed=0` |
| `codelink` | `40 3 * * *` | ok, `symbols=11273 generated=718 proposed=0` |
| `summaries` | `50 4 * * *` | ok, `already_had=137 nothing_to_say=5 errors=0` |

`verify` exists as a pass but is **not** scheduled and is not fit to schedule:
`eval/verify_skeptic_eval.py` measured 22% false refutation on main, and five
prompt/guard variants were all neutral or worse (#231).

The summary ladder reached 137 of 142 investigations only after #229, which
stopped counting access rows as findings — 3,681 of 6,610 records in
`findings.jsonl` (55.7%) are text-less access rows, and `_only_findings()` drops
them. #230 made an empty or fully-retracted investigation `nothing_to_say`
rather than an error.

---

## Static analysis

### `scripts/callgraph/`
**Purpose:** a dependency-light, stdlib-only static call-graph tool over `mcp/`,
`scripts/`, `a2a_server/`, `mlops/`, `eval/`. No LadybugDB, no network, no
third-party imports, so it runs during debugging when nothing else in the stack is
up, and it can read a specific git revision rather than a possibly mid-edit
working tree.

It complements `mcp/graph_tools.py` / `mcp/graph/*` (which need a live LadybugDB)
by modelling what this codebase actually does for dispatch and a symbol graph does
not capture: `@mcp.tool()` registration, `register(mcp, deps)` injection,
dict-of-callables dispatch, module-global reads/writes across files, and path/key
literal agreement between producers and consumers.

**Invocation:** `PYTHONPATH=scripts python3 -m callgraph.cli <command> [--rev HEAD]`

**Commands:** `build`, `modules`, `defs`, `imports`, `aliases`, `callers`, `reach`,
`paths`, `entrypoints`, `registry`, `name`, `writes-dead`, `literals`, `flags`,
`dead`, `holes`, `explain`, `selftest`, `limits`.

**Docs:** `scripts/callgraph/docs/DESIGN.md` (node/edge model, `CALLS` resolution
ladder) and `docs/LIMITS.md` / `cg limits` (the failure-mode catalogue).

---

## Evaluation

### `eval/harness.py`
**Purpose:** Longitudinal grounding quality evaluation. Runs 11 tasks through
`pre_llm_grounding.py`, scores keyword hits, upserts to Qdrant `eval_scores`.

**Run:** `eval/run_eval.sh`, which invokes `harness.py`, `grounding_gate_eval.py`
and `grounding_gate_qf_eval.py` in turn with the same arguments. There is **no
cron entry for it** — the 7-day (`10080m`) job in `cron/jobs.json` is
`deep-think-loci-harvest`, which runs `dtl_harvest.sh` and is marked
`"enabled": false`.

**Tasks defined in:** `eval/tasks.py` (11 tasks: 2 code_search, 2 memory_recall,
2 architecture_query, 2 build_check, 3 blocker_id)

**What it measures:** the harness sends the *legacy* payload shape
(`{"hook_event_name": "pre_llm_call", "extra": {"user_message": ...}}`), which the
hook still accepts. It therefore does not exercise the Claude Code
`UserPromptSubmit` / top-level `prompt` path that #228 repaired.

**Baseline (2026-06-17):** `mean_score=0.167` — low expected; keyword matching is strict.
Track the trend, not the absolute value.

**Environment:** `LOCI_PY` selects the interpreter (default
`~/.hermes/hermes-agent/venv/bin/python3`); `HARNESS_DRY_RUN=1` makes the gate
evals CI-safe by using stored cosines instead of Qdrant/Ollama.

### `eval/grounding_gate_eval.py`
Longitudinal eval of the deep-think-loci grounding gate on the labeled
finding↔finding corpus (`deep_think_loci/grounding/grounding_dataset.jsonl`).
Persists recall / bleed_rejection / f1 / accuracy / auc into `eval_scores`
alongside the grounding-pipeline scores. Threshold from `$DTL_GROUND_THRESHOLD`
(default `0.59`).

### `eval/grounding_gate_qf_eval.py`
Query→finding eval — the gate's *actual* deployed operation (a target's focus query
vs each candidate finding), as opposed to the classifier's finding↔finding training
task. In-sample diagnostic only: the trained model is fit on this same corpus, so
its AUC can read ~1.0 from memorization. It does not decide the gate default.

### `eval/grounding_gate_oos_eval.py`
Leave-one-run-out validation — the honest check, and the one that governs the gate
default. For each held-out run, trains the bleed-detector on the other runs and
evaluates cosine vs model on data it never saw. On-demand, live-only (needs
`OLLAMA_URL`), print-only.

### `eval/verify_skeptic_eval.py`
Fixed benchmark for the adversarial skeptic (#231). Every case carries its own
context and nothing touches the loci corpus, so no commit can falsify a label —
an earlier probe was invalidated exactly that way. The headline number is
**false refutation**, not accuracy: an "uncertain" verdict is harmless because a
verdict never changes a finding's lifecycle, while a "refuted" on a true claim is
what a later reader acts on. Measured 22% false refutation on main.

**Run:** `./mcp/.venv/bin/python eval/verify_skeptic_eval.py [trials] [votes]`

---

## Sync utilities

### `scripts/mnemosyne_sleep_all.sh`
Runs `mnemosyne sleep` on all banks to prune expired working memory.

**Cron:** declared as `mnemosyne-sleep-cli` (every 30m) in `cron/jobs.json`, which
has never run.

### `scripts/mnemosyne_activity_check.py`
Checks if Mnemosyne is active and responding. Used as the script input for two declared
agent jobs (`mnemosyne-consolidation` and `mnemosyne-session-summarizer`), both at
every 20m. The agents gate on whether working_memory grew before doing further work.
Neither job has run.

### `scripts/a2a_context_bridge.py`
Pushes this node's recent Mnemosyne memories to all mesh peers through the local
A2A server's `context_broadcast` skill, so local storage and peer fanout happen
atomically server-side. It reads Mnemosyne SQLite, not the Hermes event stream,
and keeps its own watermark in `BRIDGE_STATE_FILE` so a skipped tick loses nothing.

**Key env vars:** `LOCI_A2A_URL` (default `http://127.0.0.1:8201`),
`LOCI_A2A_TOKEN`, `BRIDGE_LOOKBACK_MIN` (default `30`), `BRIDGE_MIN_IMP`
(default `0.5`), `BRIDGE_MAX_ITEMS` (default `20`), `MNEMOSYNE_DATA_DIR`,
`BRIDGE_STATE_FILE` (default `~/.hermes/bridge_state.json`), `PEER_A2A_URLS`,
`PEER_A2A_TOKEN`. Flags: `--dry-run`, `--verbose`.

**Schedule:** systemd, not cron. `scripts/systemd/` ships two pairs —
`loci-context-bridge.{service,timer}` (a template for a system unit, needs the
marked block adjusted) and `mrpink-context-bridge.{service,timer}`, which is
installed as a **user** unit on this host and fires every 10m
(`OnUnitActiveSec=10min`, `OnBootSec=3min`).

**Before enabling on a second node:** the local A2A server must support
`store_local` (#144). An older server silently ignores the flag, stores before it
fans out, and every run re-inserts a copy of the memory it just read with a fresh
id and `created_at` — unbounded growth on a single node, no peer required. See
`scripts/systemd/README.md` for the check.

### `scripts/install_hooks.sh`
Symlinks `scripts/hooks/post-commit` into `.git/hooks/`. Two post-commit bodies
ship alongside it: `post_commit_ingest.sh` (re-ingests changed `.py` files into the
`loci-codebase` investigation) and `post-commit-contract-extract.sh` (extracts
contracts from changed `.py/.ts/.go/.rs/.java` files into the active investigation;
non-blocking, every path ends in `|| true`).

---

## Deep-think-loci reasoning engine

Lives in `deep_think_loci/` (not `scripts/`). A multi-tier reasoning engine that runs
as a Claude Code **Workflow** over a Loci investigation — the supported replacement for
the `deep_think` MCP server's reasoning surface. See `deep_think_loci/README.md` for the
full reference and `deep_think_loci/CHANGELOG.md` for the v1→v3.2 evolution.

**Not to be confused with the Grounding pipeline above.** That pipeline grounds *every
user turn* (pre-LLM/pre-tool hooks). This engine grounds *its own multi-agent reasoning*:
tiered models (haiku ideation → opus synthesis) fan out, persist findings to the
investigation with lineage (`derived_from`), and an opus tier produces a grounded answer.

`deep_think_loci/workflows/` holds 22 workflows: three engines (`deep-think.js`,
`deep-think-loci.js`, `deep-think-v4.js`) and 19 single-purpose audits built on
them — `wiring-gap-audit`, `silent-failure-audit`, `dead-code-audit`,
`schema-drift-audit`, `security-boundary-audit`, `performance-audit`,
`config-drift-audit`, `memory-leak-audit`, `async-ordering-audit`,
`boundary-blindness-audit`, `llm-antipatterns-audit`, `llm-context-collapse-audit`,
`dependency-contract-audit`, `test-coverage-gap-audit`, `ci-quality-gates`,
`contract-sync`, `investigation-review`, `loci-fix-sweep`, `loci-self-audit`.

### `deep_think_loci/workflows/deep-think-v4.js`
**Purpose:** the current generic engine. Its own `meta.description` states it
"Replaces v3.2 for dama-gotchi and loci-self-audit". Runs in RAG mode or
direct-read mode with parameterized targets, and can auto-file ranked actions as
GitHub issues when `github_repo` is set.
**Shape:** init → N haiku generators per target → dedicated haiku writer → verify
→ 3 opus domain reviewers → opus final → file.
**Key args:** `run_id`, `title`, `targets[]` (`{name, files, focus, read_range?}`),
`rag_collections` (set → RAG mode, null → direct-read mode).
Hardened in #199/#202: the writer stays single-purpose and the integrity check is
enforceable.

### `deep_think_loci/workflows/deep-think-loci.js`
**Purpose:** the v3.2 engine, still present and still what `install.sh` deploys.
init → 5 haiku ideation generators (RAG-grounded) → dedicated writer (persists all
findings) → verify → 2 opus half-syntheses → opus final (red-team + integrity check).
All-Claude as of v3.2 (external uncensored tier removed).
**Run:** `Workflow({ scriptPath: "deep_think_loci/workflows/deep-think-loci.js" })`.
**Key args:** `run_id`, `targets[]`, `rag_collections`, `ground_threshold` (default 0.59).
`VERSION` still reads `3.2.0-beta`.

### `deep_think_loci/grounding/ground_gate.py`
**Purpose:** cosine RAG-bleed gate — embeds query + candidates (nomic) and keeps only
those clearing a per-target threshold (0.59), dropping cross-target bleed before any model
reasons. Local, ~$0. `--model grounding_bleed_clf.joblib` swaps in the trained classifier.
**Key env:** nomic-embed endpoint (Ollama `/v1/embeddings`).

### `deep_think_loci/grounding/build_grounding_dataset.py`
**Purpose:** reproducible — harvest labeled (claim, evidence) pairs from a Loci corpus,
train + eval the bleed-detector, write `grounding_dataset.jsonl` / `grounding_bleed_clf.joblib`
/ `metrics.json`. Re-run as the corpus grows (each engine run mints more labeled pairs).
`grounding/harvest.sh` is the harvest driver; `scripts/dtl_harvest.sh` is the cron
wrapper referenced by the disabled `deep-think-loci-harvest` job.

### `deep_think_loci/install.sh`
**Purpose:** deploy to the `~/.hermes` runtime locations the workflow defaults to.
Idempotent. It copies **only** `deep-think-loci.js` (landing as
`deep-think-loci-v3.js`) plus the six `grounding/` files — the other 21 workflows,
`deep-think-v4.js` included, are not installed by it.

---

## Schedule summary

### Live — the user crontab

| Entrypoint | Pass | Schedule |
|---|---|---|
| `scripts/loci_groom_cron.sh` | `index --apply` | `17 */6 * * *` |
| `scripts/loci_groom_cron.sh` | `knn_tags` | `20 3 * * *` |
| `scripts/loci_groom_cron.sh` | `codelink` | `40 3 * * *` |
| `scripts/loci_groom_cron.sh` | `summaries` | `50 4 * * *` |

### Live — systemd user timers

| Unit | Runs | Schedule |
|---|---|---|
| `mrpink-context-bridge.timer` | `a2a_context_bridge.py` | every 10m |

### Declared in `cron/jobs.json` — none of these have ever run

| Script | Job name | Interval | Enabled |
|---|---|---|---|
| `mnemosyne_activity_check.py` | `mnemosyne-consolidation` | every 20m | yes |
| `mnemosyne_activity_check.py` | `mnemosyne-session-summarizer` | every 20m | yes |
| `mnemosyne_sleep_all.sh` | `mnemosyne-sleep-cli` | every 30m | yes |
| `mnemosyne_qdrant_sync.py` | `mnemosyne-qdrant-sync` | every 30m | yes |
| `state_db_qdrant_sync.py` | `state-db-qdrant-sync` | every 5m | yes |
| `dtl_harvest.sh` | `deep-think-loci-harvest` | every 7d | **no** |

All six carry `"last_run_at": null` and `"last_status": null`.

### On demand

`ebbinghaus_consolidation.py`, `amem_consolidation.py`, `agentHER_relabeler.py`,
`skill_annotation_updater.py`, `exif_skill_discovery.py`, `score_trace_collector.py`,
`skillops_maintenance.py`, `memgas_hierarchy.py`, the `eval/` harnesses, the
`loci_groom.py` passes not listed above (`tags`, `recall`, `verify`, `reflect`),
and the `callgraph` CLI.
