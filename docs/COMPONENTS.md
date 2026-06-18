# Component Reference

All scripts live in `scripts/` unless noted. Each entry covers: purpose, inputs,
outputs, config env vars, and cron schedule (if any).

Cron schedules are defined in `cron/jobs.json`. Scripts not listed there have no
configured schedule and must be run on demand.

---

## Grounding pipeline

### `scripts/hooks/pre_llm_grounding.py`
**Purpose:** Per-turn grounding. Embeds user message intent, fans out to 7 Qdrant
collections in parallel, fuses scores, keyword-reranks, injects MEMORY MATCH context.

**Invoked by:** `grounding_client.py` (via UDS or subprocess) on every UserPromptSubmit.

**Key env vars:**
- `QDRANT_URL` (default: `http://localhost:6333`)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `QDRANT_API_KEY`
- `MNEMOSYNE_EMBEDDING_MODEL` (default: `nomic-embed-text`)
- `HOOK_RECALL_TOP_K` (default: `5`)
- `HOOK_RECALL_MIN_SCORE` (default: `0.55`)
- `HOOK_QDRANT_WORKERS` (default: `8`)

**Output:** `{"context": "MEMORY MATCH (...) for ...:\n..."}` on stdout

---

### `scripts/grounding_client.py`
**Purpose:** Thin UDS socket client. Connects to `grounding_daemon.py` socket at
`/tmp/hermes-grounding.sock`. Falls back to direct subprocess on socket failure.

**Input:** JSON payload from `user_prompt_grounding.sh` hook
**Output:** Proxied output from `pre_llm_grounding.py`

---

### `scripts/grounding_daemon.py`
**Purpose:** Long-running daemon that keeps `pre_llm_grounding.py` warm in memory,
eliminating Python startup cost (~80ms) from grounding latency.

**Socket:** `/tmp/hermes-grounding.sock`

---

### `scripts/hooks/pre_tool_grounding.py`
**Purpose:** PreToolUse hook handler. Logs tool calls to Hermes audit trail.
Accepts missing `hook_event_name` (defaults to `pre_tool_call`).

**Invoked by:** `hermes_pre_tool_grounding.sh` (fast-path: skips Read/Write/Edit/ToolSearch)

---

### `scripts/hooks/session_end_sync.py`
**Purpose:** SessionEnd hook. Syncs final session state to Hermes state store.

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
- `MAX_PER_RUN` (default: `50`) — maximum entries processed per invocation
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
Writes semantic links to the `graph_edges` table (sim > link threshold) and conflict
flags to the `conflicts` table (sim > conflict threshold and opposing keyword pairs).
All output goes to SQLite only — no Qdrant dependency.

Supports an optional QuorumGate mechanism: set `QUORUM_AMEM_THRESHOLD > 0` to require
a minimum accumulated signal count before the expensive embedding pass runs.

**Cron:** No configured schedule — run on demand.

**Key env vars:**
- `MNEMOSYNE_DB` (default: `~/.hermes/mnemosyne/data/mnemosyne.db`)
- `OLLAMA_URL`
- `EMBED_MODEL` (default: `nomic-embed-text`)
- `AMEM_LINK_THRESHOLD` (default: `0.88`) — cosine similarity threshold for cross-links
- `AMEM_CONFLICT_THRESHOLD` (default: `0.96`) — threshold for conflict detection
- `MAX_PER_RUN` (default: `100`) — maximum entries loaded per invocation
- `QUORUM_AMEM_THRESHOLD` (default: `0`) — minimum accumulated signal before running; 0 disables

**Research basis:** A-MEM (arxiv 2502.12110, Feb 2025)

---

### `scripts/mnemosyne_qdrant_sync.py`
**Purpose:** Syncs all Mnemosyne memories to the Qdrant `mnemosyne` collection.
Uses embed-worker (`:30888`) and copies vectors from `agent_core_chunks` into `mnemosyne`.
Runs incrementally to avoid re-uploading already-synced entries.

**Cron:** every 30m (`mnemosyne-qdrant-sync` in `cron/jobs.json`)

**Key env vars:** `QDRANT_URL`, `EMBED_WORKER_URL` (default: `http://localhost:30888`)

---

### `scripts/state_db_qdrant_sync.py`
**Purpose:** Syncs Hermes `state.db` sessions to the Qdrant `hermes_sessions` collection.
Incremental (tracks already-synced session_ids in payload). Chunks per-session messages
to 4000 chars before embedding.

**Cron:** every 5m (`state-db-qdrant-sync` in `cron/jobs.json`)

**Key env vars:** `HERMES_STATE_DB` (default: `~/.hermes/state.db`),
`QDRANT_URL`, `EMBED_WORKER_URL`

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
- `MAX_PER_RUN` (default: `20`) — maximum failure entries processed per invocation

**Research basis:** AgentHER (arxiv 2603.21357, Apr 2026)

---

### `scripts/skill_annotation_updater.py`
**Purpose:** DRAFT self-annotation. Reads `guard_tool_reflections.log` from hook state,
aggregates failures by tool_name, finds matching SKILL.md files, writes or updates
"## Learned constraints" sections with top-3 failure patterns.

**Cron:** every 120m

**Key env vars:**
- `STATE_DIR` (default: `~/.claude/hook-state`)
- `SKILLS_DIR` (default: `~/.claude/skills`)
- `SKILL_ANNOTATE_MIN_USES` (default: `3`)

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
`manifest.json`.

**Run on demand**

**Key env vars:** `STATE_DIR`, `OUTPUT_DIR`, `QDRANT_URL`, `OLLAMA_URL`, `EMBED_MODEL`

**Research basis:** SCoRe (arxiv 2409.12917, Google DeepMind, ICLR 2025)

---

### `scripts/skillops_maintenance.py`
**Purpose:** SkillOps library maintenance. Scans all SKILL.md files under `~/.claude/skills`
and `~/.hermes/skills`. Embeds each description via Ollama. Computes pairwise cosine
similarity. Reports SHADOW_RISK pairs above threshold. Updates `last_validated` date in
all SKILL.md frontmatter.

**Run on demand** (or weekly; no cron currently)

**Key env vars:**
- `SKILL_SHADOW_THRESHOLD` (default: `0.92`)
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

## Evaluation

### `eval/harness.py`
**Purpose:** Longitudinal grounding quality evaluation. Runs 11 tasks through
`pre_llm_grounding.py`, scores keyword hits, upserts to Qdrant `eval_scores`. Run weekly
to track grounding quality over time as memory evolves.

**Run:** `eval/run_eval.sh` or via cron (every 10080m)

**Tasks defined in:** `eval/tasks.py` (11 tasks: code_search, memory_recall,
architecture_query, build_check, blocker_id categories)

**Baseline (2026-06-17):** `mean_score=0.167` — low expected; keyword matching is strict.
Track the trend, not the absolute value.

---

## Sync utilities

### `scripts/mnemosyne_sleep_all.sh`
Runs `mnemosyne sleep` on all banks to prune expired working memory.

**Cron:** every 30m (`mnemosyne-sleep-cli` in `cron/jobs.json`)

### `scripts/mnemosyne_activity_check.py`
Checks if Mnemosyne is active and responding. Used as the script input for two scheduled
agent jobs (`mnemosyne-consolidation` and `mnemosyne-session-summarizer`), both running
every 20m. The agents gate on whether working_memory grew before doing further work.

**Cron:** every 20m (two jobs: `mnemosyne-consolidation` and `mnemosyne-session-summarizer`)

### `scripts/a2a_context_bridge.py`
Bridges Hermes event stream to the A2A broadcast server for mesh-wide context sharing.

---

## Cron schedule summary

| Script | Job name | Interval |
|---|---|---|
| `mnemosyne_activity_check.py` | `mnemosyne-consolidation` | every 20m |
| `mnemosyne_activity_check.py` | `mnemosyne-session-summarizer` | every 20m |
| `mnemosyne_sleep_all.sh` | `mnemosyne-sleep-cli` | every 30m |
| `mnemosyne_qdrant_sync.py` | `mnemosyne-qdrant-sync` | every 30m |
| `state_db_qdrant_sync.py` | `state-db-qdrant-sync` | every 5m |

Scripts not in this table (`ebbinghaus_consolidation.py`, `amem_consolidation.py`,
`agentHER_relabeler.py`, `exif_skill_discovery.py`, `score_trace_collector.py`,
`skillops_maintenance.py`, `memgas_hierarchy.py`) have no configured cron schedule
and must be run on demand.
