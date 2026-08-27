# Memory System Architecture

## Overview

hermes_memory provides multi-layer persistent memory for LLM agents running in
Claude Code and compatible agent frameworks. It handles three time horizons:

| Horizon | Mechanism | Location |
|---|---|---|
| **Per-turn** | Qdrant grounding fan-out (< 100ms) | `scripts/hooks/pre_llm_grounding.py` |
| **Per-session** | Tool-call audit, Qdrant session sync | `scripts/hooks/` + Qdrant |
| **Long-term** | Grooming passes on the user crontab | `scripts/loci_groom.py` |

---

## Data stores

Four of them, and the split matters:

| Store | Role | Path |
|---|---|---|
| Investigation JSONL | Source of truth for findings | `$LOCI_MEMORY_DIR/<investigation>/` |
| Qdrant | Search index over those findings, plus session and Mnemosyne mirrors | `QDRANT_URL` |
| LadybugDB graph | Code graph overlaid on the investigation graph | `$LOCI_MEMORY_DIR/graph.ladybug` |
| Mnemosyne SQLite | Structured/FTS substrate for the consolidation scripts and A2A | `~/.hermes/mnemosyne/data/mnemosyne.db` |

A finding reaches Qdrant only on write (`_qdrant_upsert` at store time), so anything
that leaves the index — the startup TTL purge below is the usual reason — is still on
disk and simply invisible to search. `loci_groom.py index` is the reconciliation:
on the live corpus `on_disk=2922`, `indexed=2769`, `missing=153`, `coverage=0.9476`.
`index --apply` re-embeds and re-upserts the missing ones.

### Investigation store (`LOCI_MEMORY_DIR`, default `~/.hermes/memory-sessions`)

One directory per investigation — 146 on the live host, including the two
underscore-prefixed internal ones (`_groom`, `_reflection-loop`). Each holds:

| File | Contents |
|---|---|
| `manifest.json` | `hypothesis`, `open_questions`, `checked_sources`, `next_step`, `context`, `status`, `acl`, `owner`, `finding_counts`, `summary_l1` / `summary_l2`, `closed_summary` (143 dirs) |
| `findings.jsonl` | Append log — see below (139) |
| `entities.jsonl` | Entity records for `investigation_entity_lookup` (75) |
| `retractions.jsonl` | Soft tombstones; a retracted finding stays in the log (3) |
| `finding_updates.jsonl`, `conflicts.jsonl`, `causal_edges.jsonl`, `retraction_audit.jsonl` | Written only where the corresponding tool has run |

`summary_l1` / `summary_l2` are the summary ladder `loci_groom.py summaries` fills;
137 of the corpus's investigations already carry one, 5 report `nothing_to_say`.

`findings.jsonl` is a **mixed** append log, not a findings list. Alongside real
findings it carries access-tracking rows written on every read: they hold no text,
and because one is appended per access they are also the newest rows, so any
`findings[-N:]` slice fills with them. Measured across the corpus, 3,681 of 6,610
records (55.7%) are access rows. `investigation_tools._only_findings()`
(`mcp/investigation_tools.py:100-114`) drops them, and `investigation_load`,
`investigation_as_of`, and `investigation_reflect` all read through it. A reader
that opens the file directly gets the mixed log; `loci_groom._summarisable()`
re-derives the same filter for that reason.

### Code graph (`$LOCI_MEMORY_DIR/graph.ladybug`)

`mcp/graph/ladybug_store.py` — `LadybugStore`, an embedded LadybugDB (Kuzu engine)
database holding two overlaid graphs in one file (40 MB on the live host):

- **code graph** from tree-sitter (`graph/code_parse.py`) — `CodeFile` / `CodeSymbol`
  nodes joined by `DEFINES` / `CALLS` / `IMPORTS`;
- **investigation graph** — `Finding` / `Entity` / `Investigation` nodes joined by
  `MENTIONS` / `DERIVED_FROM` / `IN_INVESTIGATION` / `REFERENCES` / `RELATED`.

The `derived_from` lineage edges and the `causal_infer` lane land here (#218, #212).
The store is **fail-open everywhere**: if `import ladybug` fails, the db cannot be
opened, or any query raises, public methods return `False` / `[]` / `{}` and
`available()` stays `False`. `server._get_ladybug()` distinguishes an unrecoverable
failure (ladybug not importable — latches permanently) from a transient one
(another process holds the single-writer lock — retried after a backoff), so the
graph self-heals rather than staying dark for the process lifetime. Writes take a
bounded cross-process lease (`_LEASE_TIMEOUT_S = 6.0`); a wedged holder can never
hang the server.

`loci_groom.py codelink` links findings to symbols on cron: last run indexed 11,273
symbols and generated 718 links.

### Mnemosyne SQLite (`~/.hermes/mnemosyne/data/mnemosyne.db`)

Structured/FTS substrate. Not where Loci's own findings live — those are the
investigation JSONL above. Tables queried by this codebase:

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

> Note: `graph_edges` and `conflicts` *are* queried — written by
> `scripts/amem_consolidation.py` and read/updated by `scripts/glymphatic_sweep.py`,
> `scripts/spreading_activation.py`, and `scripts/mnem_fix.py`. `annotations` may
> exist in the Mnemosyne schema but is not directly queried by this repository.

### Qdrant (configured via `QDRANT_URL`)

Search index. Dense vectors are `MNEMOSYNE_EMBEDDING_DIM`-wide (default 768),
Cosine, `nomic-embed-text`. **Qdrant is disabled when `QDRANT_URL` is unset** —
there is no automatic localhost fallback in `mcp/server.py` or the grounding hooks.
Only `a2a_server/server.py` documents `http://localhost:6333` as a conventional
default in its inline comments.

| Collection | Contents | Vector layout |
|---|---|---|
| `mnemosyne` | Mirror of Mnemosyne working+episodic memory | named `dense` |
| `hermes_sessions` | Session-level traces synced by `session_end_sync.py` | named `dense` |
| `hermes_memory` | Investigation findings (MCP server primary collection) | named `dense` + `sparse` |
| `hermes_verdicts` | Claim-check verdicts for `investigation_pre_answer_check` | named `dense` |
| `<custom>` | Domain-specific collections via `GROUNDING_EXTRA_COLLECTIONS` | per-collection, see below |
| `memgas_l1` | MemGAS L1 utterance layer | named `dense` |
| `memgas_l2` | MemGAS L2 summary layer | named `dense` |
| `memgas_l3` | MemGAS L3 topic/semantic layer | named `dense` |
| `eval_scores` | Longitudinal eval harness scores (`eval/harness.py`) | named `dense` |
| `score_traces` | SCoRe correction pairs for fine-tuning | named `dense` |

The MCP server (`mcp/server.py`) uses the collection name set by
`QDRANT_COLLECTION_PREFIX` (default: `hermes_memory`) as its primary
investigation findings store.

**Named vs unnamed is not uniform, and getting it wrong fails silently.** A
named-vector search must send `{"name": "dense", "vector": [...]}`; `{"dense": [...]}`
is the *upsert* shape and a search with it returns HTTP 400. Sending a name to an
unnamed collection returns `400 Not existing vector name error`. Both come back as
zero hits, which is indistinguishable from a genuinely unmatched query — that is
exactly how the grounding hook returned nothing from all three base collections for
months (#228). Two mitigations exist:

- `qdrant_ops._dense_vector_name()` (`mcp/qdrant_ops.py:644`) probes the layout once
  per collection and caches it, so cross-collection search adapts instead of guessing.
- The hook keeps an explicit per-collection map
  (`scripts/hooks/pre_llm_grounding.py:142-162`): `ecc_skills` and `dama_gotchi_code`
  are named, **`agent_core_chunks` is unnamed**, and anything unlisted defaults to
  named `dense`. `_SearchFailed` is raised per collection so the fan-out can tell
  "nothing matched" from "every request failed".

`hermes_memory` is created (`mcp/qdrant_ops.py:293-310`) with a BM25 sparse vector
(IDF modifier), HNSW `m=32` / `ef_construct=200`, and INT8 scalar quantization
(`quantile=0.99`, `always_ram`). `rag_context_search` is therefore not plain cosine:
stage 1 fuses dense and sparse prefetches with RRF, stage 2 reranks the candidates
with a cross-encoder (`RERANK_MODEL`, default `BAAI/bge-reranker-v2-m3`). The
returned `reason` field says which path ran — `hybrid+reranked`, `hybrid`,
`semantic`, or a failure reason.

### Startup retention purge

`_get_qdrant()` calls `_purge_old_records()` on first connect. **The window defaults
to 0, which disables the purge**, and it has to stay that way unless somebody chooses
otherwise: this deletes findings and the deletion is silent. It used to default to
30 days, so a process that simply forgot to export `LOCI_QDRANT_RETENTION_DAYS`
destroyed every indexed finding older than a month on its next start. Measured on the
live store before #204: the index held 912 findings against a corpus of 2,831, and
the split was exact — all 912 younger than 30 days indexed, zero of the 1,919 older
ones. The index boundary *was* the retention window; re-indexing restored coverage
and the next server start removed it again.

Resolution order is environment, then `~/.loci/backends.toml`, then 0. The
backends.toml floor exists because the env var only reaches a process whose launcher
remembers to set it, and the four live MCP servers did not. A non-integer value
disables the purge rather than guessing a window (`mcp/qdrant_ops.py:154-194`).
`loci_groom_cron.sh` exits 3 when a groom pass refuses because the window is
non-zero — an unattended groomer against a purging server is an infinite
index-then-delete loop that burns embedding compute and reports success.

---

## Per-turn grounding pipeline

```
User message
    │
    ▼
UserPromptSubmit / SubagentStart hook
    │
    ▼
pre_llm_grounding.py (v3)
    │
    ├── 1. Read the message from payload["prompt"] (Claude Code) or
    │       payload["extra"]["user_message"] (Hermes); extract intent
    ├── 2. Embed via OLLAMA_BASE_URL + nomic-embed-text (~70ms warm)
    │       Ollama down → BeamMemory (v2/SQLite path)
    │       Both down → inject an explicit "grounding UNAVAILABLE" warning, stop
    ├── 3. Fan-out parallel Qdrant search over HOOK_QDRANT_WORKERS threads (~30ms)
    │       mnemosyne, hermes_sessions, hermes_memory
    │       + GROUNDING_EXTRA_COLLECTIONS (grounding hook env var)
    │       Per-collection request shape from the named/unnamed map; a raised
    │       _SearchFailed is counted, so "all collections failed" falls back to
    │       BeamMemory instead of passing as "nothing matched"
    │       MIN_SCORE is applied here, per hit, as results are normalised
    ├── 4. Score fusion: Qdrant cosine * importance weight
    ├── 5. _keyword_rerank(): keyword overlap boost (skill shadowing mitigation)
    ├── 6. Drop hits below MIN_IMPORTANCE
    ├── 7. Multi-signal ranking on four axes (weights must sum to 1.0):
    │       relevance (cosine × importance) · recency (exponential decay,
    │       HOOK_RECENCY_HALFLIFE_DAYS) · trust (confidence tier)
    │       · record type (observed > inferred > assumed > gap)
    ├── 8. MMR diversity selection down to HOOK_RECALL_TOP_K, with ε-exploration
    │       (HOOK_MMR_LAMBDA controls relevance-vs-diversity trade-off;
    │        HOOK_PHERO_EPSILON controls random exploration probability)
    ├── 9. Stigmergic pheromone deposit on the selected hermes_memory points
    │       (fire-and-forget; only that collection has payloads we own)
    ├── 10. Optional spreading activation enrichment (SA-RAG, arxiv 2512.15922)
    │       Seeds from mnemosyne hits carrying mnemosyne_id;
    │       result DISCARDED if SA took > HOOK_SA_TIMEOUT_MS, or skipped when
    │       the SA module is absent
    └── 11. Inject the MEMORY MATCH block (plus the rules summary, when
            HOOK_RULES_DIR has one) into Claude's context window
```

**Total latency target:** < 100ms (70ms embed + 30ms parallel Qdrant)

With no hits, the hook still injects `GROUNDING_DIRECTIVE` — the turn is never
left with an empty context block.

A companion hook, `scripts/hooks/pre_tool_grounding.py`, runs on `PreToolUse`. It
does **not** inject grounding: it scans for supply-chain IOCs and prompt injection,
and writes the tool audit log.

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
| `HOOK_RECALL_VECTOR_NAME` | `dense` | Named-vector name used on named collections |
| `GROUNDING_EXTRA_COLLECTIONS` | `""` | Extra Qdrant collections for the grounding hook only |
| `HOOK_QDRANT_WORKERS` | `8` | Fan-out thread count |
| `HOOK_EMBED_TIMEOUT` | `3.0` | Embedding request timeout (s) |
| `HOOK_QDRANT_TIMEOUT` | `2.0` | Per-collection search timeout (s) |
| `HOOK_RANKER_W_RELEVANCE` | `0.50` | Multi-signal weight — relevance |
| `HOOK_RANKER_W_RECENCY` | `0.20` | Multi-signal weight — recency |
| `HOOK_RANKER_W_TRUST` | `0.15` | Multi-signal weight — confidence tier |
| `HOOK_RANKER_W_TYPE` | `0.15` | Multi-signal weight — record type |
| `HOOK_RECENCY_HALFLIFE_DAYS` | `7` | Recency decay half-life |
| `HOOK_SA_ENABLED` | `true` | Enable spreading activation enrichment |
| `HOOK_SA_TIMEOUT_MS` | `25` | SA budget in ms; result discarded if exceeded |
| `HOOK_MMR_LAMBDA` | `0.75` | MMR relevance weight (1.0 = pure relevance) |
| `HOOK_PHERO_BETA` | `0.08` | Pheromone score boost coefficient |
| `HOOK_PHERO_HALFLIFE_H` | `24` | Pheromone evaporation half-life (hours) |
| `HOOK_PHERO_DEPOSIT` | `1.0` | Pheromone deposited per retrieval |
| `HOOK_PHERO_EPSILON` | `0.05` | ε-exploration probability |
| `HOOK_RULES_DIR` | `<profile>/rules` | Rules summary appended to the injected block |
| `HOOK_BLOCK_MODE` | `0` | `pre_tool_grounding.py`: block on detection instead of audit |

> `GROUNDING_EXTRA_COLLECTIONS` is read independently by the grounding hook.
> `EXTRA_RAG_COLLECTIONS` is a separate variable read by `a2a_server/server.py`
> for its `rag_search` skill. **Set both** if you want the same extra collections in
> both paths — `.env.example:90` claims the grounding variable "defaults to
> `EXTRA_RAG_COLLECTIONS` if not set independently", and it does not:
> `pre_llm_grounding.py:156` reads only its own name and falls back to `""`.

---

## Per-session event capture

This repo ships **three** hooks. `scripts/hooks/install.sh` copies them into
`~/.claude/hooks/` (backing up what it replaces) and `install.sh --check` reports
drift without changing anything — the two copies had already diverged once, with
the deployed `pre_tool_grounding.py` hand-edited to accept Claude Code's
`PreToolUse` event name while the repo copy still only accepted the Hermes name,
so a fresh install would have silently disabled the hook.

```
UserPromptSubmit  → pre_llm_grounding.py
                    Grounding fan-out per turn (see above)

SubagentStart     → pre_llm_grounding.py  (HERMES_SUBAGENT=1)
                    Same fan-out, so a subagent starts grounded too

PreToolUse        → pre_tool_grounding.py
                    Supply-chain IOC + prompt-injection scan, dangerous-pattern
                    detection, audit log

Stop              → session_end_sync.py
                    Embeds the session and upserts one point into hermes_sessions
```

`pre_llm_grounding.py` reads the user message from the payload's **top-level
`prompt`** field, falling back to Hermes' `extra.user_message`. Reading only the
Hermes shape is what made it inert under Claude Code: the hook ran, exited 0, and
injected nothing (#228, `pre_llm_grounding.py:550-551`). It accepts both event
names — Hermes' `pre_llm_call` and Claude Code's `UserPromptSubmit` /
`SubagentStart` — and treats a run as subagent context when the task id says so or
`HERMES_SUBAGENT` is set.

`session_end_sync.py` prefers the session's messages from `LOCI_STATE_DB`
(`~/.hermes/state.db`) and falls back to the Stop payload's `transcript_path`
(`session_end_sync.py:186-204`). The fallback is the one that fires under Claude
Code: Claude Code session UUIDs are not rows in a Hermes `state.db`, so the
state.db-only version had never synced a session. A per-session mtime cache
(`LOCI_SYNC_CACHE`) makes an unchanged session exit immediately.

`pre_tool_grounding.py` **audits by default**. `HOOK_BLOCK_MODE` is `0` unless set,
so detections are logged, not refused. The one unconditional block is an injection
pattern in an agent-config write target (`AGENTS.md`, `CLAUDE.md`, `.cursorrules`),
which blocks regardless of `HOOK_BLOCK_MODE`. Read-only and grounding tools are
allowlisted out of the audit entirely to keep the log signal-bearing.

### session_end_sync.py env vars

| Variable | Default | Purpose |
|---|---|---|
| `QDRANT_URL` | _(none — sync skipped if unset)_ | Qdrant instance URL |
| `OLLAMA_BASE_URL` | _(none)_ | Embedding base URL |
| `MNEMOSYNE_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `MNEMOSYNE_EMBEDDING_DIM` | `768` | Vector dimension |
| `LOCI_STATE_DB` | `~/.hermes/state.db` | Hermes session database; unused under Claude Code, which takes the `transcript_path` fallback |
| `LOCI_SYNC_CACHE` | `~/.hermes/.session_sync_cache` | Per-session mtime cache |
| `HERMES_AGENT_ID` | `""` | Agent identity tag stamped on payloads |

---

## MCP server (loci-mcp)

`mcp/server.py` is registered as `FastMCP('loci')` and exposes **73** tools under
the `loci-mcp` server name. Only 42 are declared with `@mcp.tool()` in `server.py`;
the rest come from modules that are handed the shared instance at import time and
register their own list:

| Source | Tools | Registration |
|---|---|---|
| `server.py` | 42 | `@mcp.tool()` |
| `graph_tools.py` | 11 | `graph_tools.register(mcp, _get_ladybug)` |
| `investigation_tools.py` | 11 | `investigation_tools.register(mcp, …, deps)` |
| `llm_tools.py` | 9 | `llm_tools.register(mcp)` |

Count them with `grep -c '@mcp.tool()' mcp/server.py` plus the `for fn in (...)`
tuple in each module's `register()`. `inv_store.register()` and
`ladybug_ops.register()` register **no** tools — they only inject accessors.

Modules are given callables, never values: `_get_ladybug` by reference and the
memory root as a lambda over `MEMORY_DIR`, so tests that rebind those globals still
steer the helpers. Key env vars:

| Variable | Default | Purpose |
|---|---|---|
| `QDRANT_URL` | _(none — Qdrant disabled if unset)_ | Qdrant instance URL |
| `QDRANT_API_KEY` | `""` | Qdrant API key |
| `QDRANT_COLLECTION_PREFIX` | `hermes_memory` | Primary investigation findings collection |
| `OLLAMA_BASE_URL` | _(none)_ | Embedding base URL |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `EMBED_API_KEY` | `""` | Cloud embedding provider key |
| `LOCI_MEMORY_DIR` | `~/.hermes/memory-sessions` | JSONL session storage root |
| `LOCI_MNEMO_BANK` | `default` | Active Mnemosyne bank |

> `mcp/server.py` uses `OLLAMA_BASE_URL` and `EMBED_MODEL` for embedding.
> `a2a_server/server.py` uses `MNEMOSYNE_EMBEDDING_API_URL` and `MNEMOSYNE_EMBEDDING_MODEL`.
> `session_end_sync.py` uses `OLLAMA_BASE_URL` and `MNEMOSYNE_EMBEDDING_MODEL`.
> These are three distinct env var names for the embedding base URL. Set all
> relevant variables in your `.env` when running multiple components together.

Generation resolves **separately** from embeddings (#222/#224). `llm_local.py` reads
`LOCI_OLLAMA_GEN_URL` / `OLLAMA_GEN_URL` at call time (not import time — `loci_groom`'s
`load_env()` runs after this module is imported, so an import-time capture reads the
environment from before the process configured itself). Reachability is not capability:
an Ollama that answers `/api/tags` may carry no generation model, which is what sent
100% of generation to a host that could not serve it. The two also want opposite hosts —
measured here, embeddings are 93 ms in-cluster vs 5,595 ms over the tailnet, while the
generation model exists only over the tailnet (35.7 s cold, 247 ms warm with
`keep_alive`). One URL cannot serve both.

The vLLM fallback is opt-in: `LOCI_VLLM_FALLBACK` must be set to something other than
`""` / `0` (`mcp/llm_local.py:170-180`).

### Transports and auth

`LOCI_MCP_TRANSPORT` defaults to `stdio`. For `sse` / `streamable-http`:

| Variable | Default | Purpose |
|---|---|---|
| `LOCI_MCP_HOST` | `127.0.0.1` | Bind address |
| `LOCI_MCP_PORT` | `8000` | Bind port |
| `LOCI_MCP_TOKEN` | _(none)_ | Bearer token; **required** for a non-loopback bind |

The bind is loopback by default (#206) and a non-loopback bind without a token is a
`SystemExit`, not a warning (#211) — this server exposes the whole tool surface, so
a wide open bind has to be a decision somebody made rather than one they got by not
making it. `docker-compose.yml` sets `LOCI_MCP_HOST=0.0.0.0` explicitly, because a
container must bind all interfaces to receive published traffic, and publishes on
127.0.0.1.

The token check is a small ASGI middleware, deliberately not the SDK's
`token_verifier` path: that requires `AuthSettings` with an `issuer_url` and would
publish `/.well-known/oauth-*` describing an authorization server that does not
exist. `/health` stays exempt so liveness probes work without the secret, and
returns a fixed `{"status": "ok"}`. Comparison is `hmac.compare_digest` on both
branches — a plain `==` leaks token length and prefix through timing, and an early
return on empty leaks whether a token was presented at all.

---

## Claim checking (`investigation_pre_answer_check`)

A proposed answer is split into claims and each claim is checked against stored
evidence on two lanes: a lexical lane and a semantic (dense) lane. Verdicts are
written to the `hermes_verdicts` collection (`mcp/verdict_ops.py`).

**A cosine score alone does not establish support.** Measured on the live corpus
with 300 claims copied verbatim *out of a different investigation*, a bare
`score >= 0.55` gate reported 264 of 300 (88.0%) as supported, and in all 264 the
lexical lane had returned nothing. 96.7% of those negatives scored inside the
positive range, so no choice of threshold fixes it — cosine here is similarity
inside a small pool, not a probability of support.

`_semantic_ref_corroborated()` (`mcp/server.py:1408`) therefore requires a
conjunction of two independent conditions, both fitted on 600 probes:

| Condition | Constant | Meaning |
|---|---|---|
| `lexical_overlap` | `>= 0.15` | The hit mentions what the claim mentions, computed against the full payload text, not the 260-char snippet (which biases long findings) |
| `score - pool_median` | `>= 0.05` | The hit is distinctive inside its own 25-point neighbourhood, not one of a pool equally close to the claim |
| `pool_size` | `>= 8` | Below this a margin over the pool median is undefined, so the check fails closed |

Operating point of the conjunction, replayed offline in
`mcp/tests/fixtures/semantic_gate_probe.json`:

| | before | after |
|---|---|---|
| false support (n=300 negatives) | 88.0% | 1.7% |
| true support (n=300 positives) | 100.0% | 80.3% |

Neither half separates the classes on its own — best-ref lexical overlap is NEG p95
0.162 vs POS p05 0.070, and best-ref margin is NEG p95 0.1013 vs POS p05 0.0528.
The conjunction is what carries the measurement. The `pool_size` rule is close to a
no-op on the headline numbers (removing it moves false support 1.67% → 2.00%) and
is kept because it fails closed.

The 0.55 score survives as a pre-filter, not as the decision: a neighbour above it
enters `support_refs` only when `_semantic_ref_corroborated` holds, and everything
else is surfaced — labelled, not dropped — under `semantic_candidates`. Every ref
carries `pool_size`, and each claim reports `support_basis` as `lexical` /
`semantic_corroborated` / `semantic_candidate_only` / `none`, with `supported` true
only for the first two. So a caller can always see which lane decided and why a
candidate was not promoted.

---

## Consolidation and self-improvement pipeline

### What actually runs on a schedule

The **user crontab**, four grooming passes through `scripts/loci_groom_cron.sh`.
Verified through that real entrypoint, with the last line each one logged:

| Pass | Schedule | Last measured result |
|---|---|---|
| `index --apply` | `17 */6 * * *` | ok, on_disk=2922 indexed=2769 coverage=0.9476 |
| `knn_tags` | `20 3 * * *` | ok, vocabulary=60 candidates=360 generated=18 proposed=0 |
| `codelink` | `40 3 * * *` | ok, symbols=11273 generated=718 proposed=0 |
| `summaries` | `50 4 * * *` | ok, already_had=137 nothing_to_say=5 errors=0 |

Every pass holds three rules: **idempotent** (a second run over unchanged input
proposes nothing new), **fail-open** (a dead backend degrades the pass to a report),
and **shadow-first** (model-derived output goes to `_groom/proposals.jsonl` with its
provenance and is *never* merged into `findings.jsonl`). `--apply` is honoured only
by passes declaring `applyable` — today just `index`, whose write re-upserts a record
already on disk, so it can restore but cannot invent.

The wrapper exists for its exit codes: `0` ok, `1` a pass errored, `3` a pass refused
or degraded. It appends one line per run to `~/.loci/groom/runs.jsonl`, so "has this
ever actually run, and what did it say" is a question with an answer.

`loci_groom.py` also defines `tags`, `recall`, `verify`, and `reflect`. None are
scheduled. **`verify` is not fit to schedule**: measured at 22% false-refutation
against the fixed benchmark in `eval/verify_skeptic_eval.py`, and five prompt/guard
variants were all neutral or worse, so #231 changed no behaviour.

`pass_summaries` drives the summary ladder over `_only_findings()` output rather than
the raw log; an investigation that is empty or fully retracted reports
`nothing_to_say`, not an error (#229/#230). Before that, the ladder was handed twenty
text-less access rows and returned an invented summary.

### cron/jobs.json — present, not running

`cron/jobs.json` describes six Hermes cron jobs (mnemosyne-consolidation,
mnemosyne-session-summarizer, mnemosyne-sleep-cli, deep-think-loci-harvest,
mnemosyne-qdrant-sync, state-db-qdrant-sync). Do not read it as a description of
live behaviour: all six carry `last_run_at: null`, `~/.hermes/cron/` is empty, and
`crontab -l` contains no Hermes runner (issue #205). Nothing on this host reads the
file.

On-demand scripts (not croned):
- `memgas_hierarchy.py --index` — rebuild MemGAS 3-level Qdrant collections
- `memgas_hierarchy.py --search <query>` — entropy-weighted 3-level search
- `exif_skill_discovery.py` — EXIF skill gap analysis → candidate SKILL.md
- `score_trace_collector.py` — build SCoRe fine-tuning dataset from logs
- `skillops_maintenance.py` — skill shadow detection + last_validated update
- `ebbinghaus_consolidation.py` — FSRS/Ebbinghaus decay-triggered Qdrant refresh
- `swr_replay.py` — replays top-K recent findings (recency × salience × reward) into one
  compressed `record_type=consolidated` abstraction
- `glymphatic_sweep.py` — prunes superseded verdicts, orphaned sessions, dangling
  `graph_edges`, and near-duplicate Qdrant points. Must not run concurrently with
  `swr_replay.py` or `amem_consolidation.py` — it takes a mutex flag for that reason
- `amem_consolidation.py` — cross-link graph + conflict detection
- `skill_annotation_updater.py` — SKILL.md learned constraints
- `agentHER_relabeler.py` — failure → positive trace relabeling
- `eval/harness.py` — longitudinal grounding quality score
- `eval/verify_skeptic_eval.py` — fixed benchmark for the adversarial skeptic

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

**Its input is not being produced.** Every consumer below reads
`guard_tool_reflections.log` out of `STATE_DIR` (default `~/.claude/hook-state`,
`scripts/skill_annotation_updater.py:17`). That directory does not exist on this
host, no hook in `scripts/hooks/` writes the file, and none of the consumers are
scheduled. The loop below is the design; treat it as unrun until a reflection
writer is wired.

The one tool-call record that *is* being written is `pre_tool_grounding.py`'s audit
log at `~/.hermes/logs/tool-audit.log` (5 MB, self-rotating to the last 2 MB) — a
different shape from a typed reflection trace, and nothing downstream reads it.

```
PostToolUseFailure hook          ← NOT WIRED (no such Claude Code event; no writer)
    │
    ▼
reflection writer
    │  Would categorize failure type and write a typed trace to:
    │  - Mnemosyne (importance=7)
    │  - guard_tool_reflections.log (JSONL)
    │
    ├──► ebbinghaus_consolidation.py (on demand)
    │    Re-embeds decayed memories → Qdrant refresh
    │
    ├──► amem_consolidation.py (on demand)
    │    Builds semantic cross-links, flags near-duplicate conflicts
    │
    ├──► skill_annotation_updater.py (on demand)
    │    Reads guard_tool_reflections.log → updates SKILL.md "Learned constraints"
    │
    ├──► agentHER_relabeler.py (on demand)
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
| `memory_prime` | SAR-style priming broadcast — decaying skepticism boost for a topic cluster, written to `~/.hermes/sar-priming.json` and optionally pushed to peers |
| `mnemosyne_triple_add` | Store a knowledge triple in the SQLite `triples` table |
| `mnemosyne_triple_query` | Query the knowledge graph by subject/predicate/object |
| `gpu_inference` | Run a prompt through local Ollama |
| `docker_status` | List running Docker containers and k3s pods |
| `ua_search` | Semantic search over understand-anything knowledge graphs |

### A2A env vars

| Variable | Default | Purpose |
|---|---|---|
| `LOCI_A2A_TOKEN` | `""` | Bearer token for callers |
| `LOCI_A2A_BOOTSTRAP_KEY` | `""` | Enrollment key for the bootstrap path |
| `LOCI_A2A_HOST` | `0.0.0.0` | Bind address |
| `LOCI_A2A_PORT` | `8201` | Bind port |
| `LOCI_A2A_URL` | `http://127.0.0.1:8201` | Public base URL in agent card |
| `LOCI_A2A_TOTP_SEED` | `""` | Base32 TOTP seed (RFC 6238); disabled when empty |
| `HERMES_AGENT_ID` | `hermes-agent` | Agent identity tag |
| `QDRANT_URL` | _(none)_ | Qdrant instance URL |
| `QDRANT_API_KEY` | `""` | Qdrant API key |
| `MNEMOSYNE_EMBEDDING_API_URL` | `http://localhost:11434/v1` | Embedding base URL (with `/v1`) |
| `MNEMOSYNE_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `MNEMOSYNE_EMBEDDING_DIM` | `768` | Vector dimension |
| `MNEMOSYNE_DATA_DIR` | `~/.hermes/mnemosyne/data` | Directory containing mnemosyne.db |
| `EXTRA_RAG_COLLECTIONS` | `""` | Extra collections for the A2A rag_search skill |
| `PEER_A2A_URLS` | `""` | Comma-separated peer A2A endpoints for context_broadcast |
| `PEER_A2A_TOKEN` | `""` | Shared bearer token for all peers |

Unlike the MCP server, the A2A server **binds `0.0.0.0` by default and only warns**
when `LOCI_A2A_TOKEN` is unset (`a2a_server/server.py:210-216`) — every
bearer check then fails, so callers are refused, but the port is still open on all
interfaces. Set a token, or bind loopback, before running it anywhere reachable.

`scripts/a2a_context_bridge.py` subscribes to Hermes events and routes context
updates to the A2A `context_broadcast` endpoint, propagating discoveries across
the mesh.
