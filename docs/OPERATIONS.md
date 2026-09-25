# Operations Guide

## Environment variables

Settings resolve through a chain (`mcp/backends.py`):

1. the environment variable
2. a local probe — `http://localhost:11434` for Ollama, `:8000` for vLLM
3. `~/.loci/backends.toml`, or `$LOCI_CONFIG` — gitignored, machine-specific
4. the code default

`scripts/loci_groom.py:load_env()` inserts the repo `.env` and `mcp/.env` between
steps 1 and 3 because cron does not inherit the MCP launcher's environment. Copy
`backends.toml.example` to `~/.loci/backends.toml` for endpoints and keys that must
not live in the repo.

### Core infrastructure

| Variable | Default | Used by |
|---|---|---|
| `QDRANT_URL` | _(none — required; unset means Qdrant steps are skipped or the script errors out, never a localhost fallback)_ | all Qdrant-touching scripts |
| `QDRANT_API_KEY` | _(none)_ | all Qdrant-touching scripts |
| `OLLAMA_URL` | _(none, required)_ | most embedding + generation scripts (memgas_hierarchy.py, ebbinghaus_consolidation.py, amem_consolidation.py, agentHER_relabeler.py, skillops_maintenance.py, exif_skill_discovery.py, score_trace_collector.py, eval/harness.py) |
| `OLLAMA_BASE_URL` | _(none in code; `backends.ollama_url()` probes `http://localhost:11434`)_ | the **embedding** endpoint for 16 non-test files: `mcp/{qdrant_ops,embed_ops,backends,memcheck/llm}.py`, `scripts/hooks/{pre_llm_grounding,session_end_sync}.py`, `scripts/{loci_groom,glymphatic_sweep,gpu_warm}.py`, all of `mlops/`, `deep_think_loci/grounding/` |
| `LOCI_OLLAMA_GEN_URL` / `OLLAMA_GEN_URL` | _(none; falls back to `backends.ollama_gen_url()` → `ollama_url()`)_ | the **generation** endpoint, resolved separately from embeddings (`mcp/llm_local.py`). `OLLAMA_BASE_URL` deliberately does **not** feed it |
| `LOCI_OLLAMA_GEN_MODEL` | auto (`qwen2.5:3b` if present, else first local non-embedding tag, else `qwen2.5:3b`) | the generation model tag on `gen_url` (`mcp/backends.py:ollama_gen_model`). Explicit env/config still wins. This auto-fallback prevents hardcoded defaults from silently pointing at missing local tags. |
| `LOCI_OLLAMA_REDTEAM_MODEL` | auto (preferred: `hf.co/slevinw/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF:Q4_K_M`, then local heretic/abliterated tag, else that default string) | explicit adversarial model for `scripts/local_deep_think.py --red-team`. This intentionally biases toward heretic/abliterated models because aligned models often refuse or soften adversarial critique prompts; only the opt-in red-team tier uses it. The same script now also auto-promotes confirmed high-confidence action-shaped findings into procedure memory unless you pass `--no-learn-procedures`. |
| `LOCI_VLLM_FALLBACK` | `0` (off) | opt-in fallback from Ollama generation to a batched vLLM endpoint (`mcp/llm_local.py`, `mcp/batched_gen.py`). Worth enabling whenever the Ollama generation tier is anything other than fully verified working — it is a real, independent tier, not just a stub |
| `LOCI_TMUX_COMPANION_REQUIRED` | `0` (off) | if set truthy (`1/true/yes/on`), `loci_health` fails loud when required tmux companion sessions are missing. Use this when Copilot/Claude tmux loops are part of required runtime posture |
| `LOCI_TMUX_COMPANION_SESSIONS` | `claude,copilot` | comma-separated tmux session names checked by `loci_health` when companion monitoring is enabled |
| `LOCI_TMUX_ROLE_SESSION_MAP` | _(empty)_ | optional `role=session` mappings for offload telemetry attribution (example: `triage=copilot,code=claude`) |
| `LOCI_TMUX_STALE_SECONDS` | `300` | deterministic stale threshold for tmux workers. A mapped session with latest live-pane activity older than this is marked `stale_worker` |
| `LOCI_CLOUD_TIER_ENABLED` | `0` (off) | enables third-tier cloud fallback orchestration in `mcp/llm_local.py` after local tiers fail. When enabled, unspecified-model calls can be role-routed to OpenRouter or Abliteration |
| `LOCI_CLOUD_TIER_SUPERVISOR_MODEL` | `backends.ollama_verify_model()` (the verify-tier model) | local supervisor model used to triage/dispatch unspecified-model calls by role (`triage`, `coding`, `reasoning`, `synthesis`, `redteam`) before cloud escalation |
| `LOCI_CLOUD_TIER_ALLOW_PROMPT_EXPORT` | `0` (off) | hard gate for third-party prompt egress. Cloud fallback will not send prompts to OpenRouter/Abliteration unless this is explicitly set to `1/true/on` |
| `LOCI_CLOUD_TIER_MAX_TOKENS_PER_CALL` | `0` (disabled) | hard cloud-fallback per-call cap. When set, cloud routing refuses requests with larger `max_tokens` and returns `tier=cloud-refused` with reason |
| `LOCI_CLOUD_TIER_DAILY_CALL_BUDGET` | `0` (disabled) | max cloud fallback calls per UTC day. Exceeding it causes explicit refusal (no silent fallback) |
| `LOCI_CLOUD_TIER_DAILY_TOKEN_BUDGET` | `0` (disabled) | max requested cloud fallback tokens per UTC day. Exceeding it causes explicit refusal |
| `LOCI_CLOUD_TIER_DENY_PROVIDERS` | _(none)_ | comma-list deny gate (`openrouter`, `abliteration`) applied before provider attempts |
| `LOCI_CLOUD_TIER_DENY_ROLES` | _(none)_ | comma-list deny gate for routed roles (`triage`, `coding`, `reasoning`, `synthesis`, `redteam`) |
| `LOCI_CLOUD_TIER_ALLOWED_ROLES` | _(none)_ | optional allow-list for routed roles. If set, non-listed roles are explicitly refused |
| `LOCI_CLOUD_TIER_BUDGET_STATE_PATH` | `~/.loci/cloud_tier_budget.json` | JSON ledger path for daily cloud call/token accounting |
| `LOCI_TMUX_OFFLOAD_ENABLED` | `0` (off) | enables tmux-based offload lane policy reads from `mcp/backends.py`; off by default to preserve current behavior |
| `LOCI_TMUX_OFFLOAD_ROLE_SESSIONS` | _(none)_ | role→session map for tmux lanes. Accepts comma pairs (`triage=loci-fast,reasoning=loci-deep`) or JSON object |
| `LOCI_TMUX_OFFLOAD_EXPENSIVE_ROLES` | _(none)_ | comma-list roles treated as expensive lanes (for priority/routing policy), e.g. `reasoning,synthesis,redteam` |
| `LOCI_TMUX_OFFLOAD_REQUIRE_MAPPED_SESSION` | `0` (off) | strict mode: if `1/true/on`, callers should fail closed when a mapped tmux session is unavailable |
| `OPENROUTER_BASE_URL` / `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` / _(none)_ | OpenRouter cloud offload tier (`mcp/openrouter.py`) |
| `OPENROUTER_MODEL` / `OPENROUTER_MODEL_<ROLE>` | `qwen/qwen3.8-27b:free` / _(none)_ | shared and per-role OpenRouter model mapping used by cloud-tier fallback; role keys include `TRIAGE`, `CODING`, `REASONING`, `SYNTHESIS`, `REDTEAM` |
| `LOCI_OPENROUTER_MAX_TOKENS_PER_CALL` | `0` (disabled) | provider-level hard cap in `mcp/openrouter.py`; requests above this max are explicitly rejected before HTTP calls |
| `ABLITERATION_BASE_URL` / `ABLITERATION_API_KEY` | `https://api.abliteration.ai/v1` / _(none)_ | Abliteration cloud escalation tier (`mcp/abliteration.py`) |
| `ABLITERATION_MODEL` / `ABLITERATION_MODEL_<ROLE>` | `abliterated-model` / _(none)_ | shared and per-role Abliteration model mapping; red-team/offensive escalation should usually set `ABLITERATION_MODEL_REDTEAM` explicitly |
| `LOCI_ABLITERATION_MAX_TOKENS_PER_CALL` | `0` (disabled) | provider-level hard cap in `mcp/abliteration.py`; requests above this max are explicitly rejected before HTTP calls |

**Diagnosed 2026-09-15, corrected in `~/.loci/backends.toml` (machine-specific,
gitignored — not shown here):** this box's `[ollama].gen_url` had been pointed at a
tailnet host that only ever carried the embedding model, and `gen_model` named a tag
that was never actually pulled anywhere — so every single-shot generation call
(`compress_text`, `classify_text`, `query_expand`, `verify_finding`) had been silently
degraded, invisibly, for some time. Nothing raised or logged loudly because every layer
here is deliberately fail-open. **If any of these tools report `degraded: true` more
than occasionally, first re-verify `gen_url`/`gen_model` directly** — pick a real
`ollama list` tag on a host you can `curl .../api/generate` successfully — before
assuming the model itself is the problem. On Windows+WSL2 hosts specifically, see the
NAT-networking gotcha in `backends.toml.example`.
| `LOCI_QDRANT_RETENTION_DAYS` | `0` — purge **disabled** | `mcp/qdrant_ops.py:_retention_days`. Any value > 0 makes the first Qdrant call of every process delete findings older than that window. Also readable as `[qdrant].retention_days` in `~/.loci/backends.toml` |
| `MNEMOSYNE_EMBEDDING_MODEL` | `nomic-embed-text` | all embedding operations |

Note: `OLLAMA_URL` and `OLLAMA_BASE_URL` are distinct variables read by different
files; the split is not hooks-vs-standalone. The MCP server, the grooming tier, and
both Claude Code hooks read `OLLAMA_BASE_URL` (a bare host) and append `/v1`
themselves. Older standalone scripts read `OLLAMA_URL`: `memgas_hierarchy.py`,
`ebbinghaus_consolidation.py`, `amem_consolidation.py`, `agentHER_relabeler.py`,
`skillops_maintenance.py`, `exif_skill_discovery.py`, `score_trace_collector.py`,
`swr_replay.py`, `ua-ingest.py`, `gpu_warm.py`, and `eval/harness.py`.
`mcp/embed_ops.py` and `mcp/backends.py` read both.

### tmux offload observability + remediation

`loci_health` now reports deterministic tmux companion health in:

- `tmux_companion_health` rows (`status`: `healthy|missing_session|no_live_worker|stale_worker`)
- rollups: `tmux_companion_missing`, `tmux_companion_no_live_worker`, `tmux_companion_stale`
- threshold: `tmux_stale_threshold_s`

If `LOCI_TMUX_COMPANION_REQUIRED=1`, any missing/no-live/stale required session
marks health `unhealthy` (no silent fallback).

Remediation flow:

1. Run `loci_health` and inspect the tmux fields above.
2. For `missing_session`, recreate/start the named session.
3. For `no_live_worker`, restart the worker pane/process in that session.
4. For `stale_worker`, inspect pane activity and restart the stalled worker.
5. Re-run `loci_health` and verify all required sessions are `healthy`.

### Cloud tier routing policy (OpenRouter + Abliteration)

The cloud tier is intentionally **opt-in** and only activates when local generation
fails and `LOCI_CLOUD_TIER_ENABLED=1`.

- **OpenRouter role**: low-cost/free offload and burst absorption.
- **Abliteration role**: heavy uncensored red-team escalation.
- **Supervisor role**: a stronger local model (default: the verify-tier model, `backends.ollama_verify_model()`) triages
  unspecified-model calls into role buckets so provider choice is explicit and auditable.
- **Egress guard**: set `LOCI_CLOUD_TIER_ALLOW_PROMPT_EXPORT=1` or cloud fallback will refuse external prompt export by default.
- **Budget guardrails**: optional per-call and daily budgets now fail closed with explicit
  refusal reasons (`tier=cloud-refused`) instead of silently trying more expensive routes.

Recommended starting assignments (cost-aware defaults from 2026-09 research):

| Role | OpenRouter default | Abliteration default |
|---|---|---|
| `triage` | `qwen/qwen3.8-27b:free` | `abliterated-model` |
| `coding` | `prism-ml/ternary-bonsai-2-27b` | `abliterated-model` |
| `reasoning` | `z-ai/glm-5.3-flash` | `abliterated-model` |
| `synthesis` | `deepseek/deepseek-pro-latest` | `abliterated-model-large-v2` |
| `redteam` | `qwen/qwen3.8-flash` | `abliterated-model-large-v2` |

Benchmark gate before promoting defaults:
1. Run role benchmarks across local/openrouter/abliteration lanes.
2. Promote only after threshold pass and zero critical misroutes.
3. Record the decision + model map in ops notes so assignment is reproducible.

### tmux offload lane policy

`mcp/backends.py` now exposes tmux lane policy readers under `[tmux_offload]` in
`~/.loci/backends.toml` (or matching env vars). Safe defaults keep the feature disabled:

- `enabled=false`
- `role_sessions={...}` optional role→session map
- `expensive_roles=[]` optional expensive-role set
- `require_mapped_session=false` (strict fail-closed toggle)

### Opt-in heavier Ollama generation tags

The checked-in default stays `qwen2.5:3b`. If you want a local abliterated tier,
deploy one explicitly:

```bash
scripts/deploy_abliterated_model.py 14b-coder
scripts/deploy_abliterated_model.py 24b-mistral
scripts/deploy_abliterated_model.py 27b-qwen38
scripts/deploy_abliterated_model.py 26b-gemma4
```

The script downloads one researched GGUF quant into `~/.loci/models/ollama/`,
rewrites the matching `ollama/Modelfile.*` to that local path, runs `ollama create`,
then does a tiny `/api/generate` smoke test and prints `/api/ps` / GPU state.
It now also verifies each downloaded GGUF against a pinned SHA256 manifest before
`ollama create`; if you change repos/quants, update the manifest in
`scripts/deploy_abliterated_model.py` first or the deploy will fail closed.

The new Qwen3.8-27B option is still opt-in and keeps the code default unchanged, but
it rides llama.cpp's `qwen3_5` hybrid linear-attention path, where
`ggml-org/llama.cpp#28879` tracks unresolved non-monotonic quant-precision behavior,
and the still-open draft `ggml-org/llama.cpp#27132` covers a distinct qwen3_5
tensor-layout conversion-correctness bug. Treat Q5_K_M or higher as the safe
starting point when you can, expect roughly 16-20GB VRAM at Q4/Q5, and verify
outputs empirically before precision-sensitive use.
The Gemma4 A4B option is an MoE with only ~4B active params; Google also publishes
official QAT GGUFs of the non-abliterated base as a more-trusted comparison baseline.

For the abliterated deployments above:

- The GGUF artifacts come from third-party/community Hugging Face repos, not
  first-party model publishers or other verified upstream release channels.
- Abliteration removes built-in safety training, so outputs may be more permissive
  or less filtered than the original instruct models; restrict access accordingly.
- Local hosting does not make prompt contents private by default: treat the Ollama
  endpoint like any other inference service and apply appropriate host, network,
  and access controls for sensitive data.
- The deploy smoke test and current `scripts/ab_eval_local_model.py` checks only
  cover endpoint reachability plus JSON/schema-format conformance, not adversarial
  robustness, policy compliance, or broader security properties.

After that, switch Loci yourself with `LOCI_OLLAMA_GEN_MODEL=<the-created-tag>`.
See the resolution chain in `mcp/backends.py` (`LOCI_OLLAMA_GEN_MODEL` →
`~/.loci/backends.toml` `[ollama].gen_model` → code default) rather than changing code.

### Benchmark-driven role assignment (required)

Do **not** hand-pick role defaults from memory or one-off host state. Assignments must
come from a benchmark artifact plus local tag availability.

1. Run the quality benchmark against the target generation endpoint:

```bash
python3 scripts/bench_model_catalog_quality.py \
  --base-url http://<ollama-host>:11434 \
  --timeout-s 45 \
  --max-tokens 96 \
  --output artifacts/model_catalog/quality_<date>.json
```

2. Derive role assignments from that JSON and currently installed local tags:

```bash
python3 scripts/assign_models_from_benchmark.py \
  --benchmark-json artifacts/model_catalog/quality_<date>.json
```

This prints an `[ollama]` TOML block mapping:
- `synthesis` winner -> `gen_model`
- `escalation` winner -> `verify_model`
- `cheap_fanout` winner -> `compress_model`
- `guardian` winner -> `guardian_model`
- best available local heretic/abliterated -> `redteam_model`

If the script exits non-zero, at least one selected winner is not installed locally.
Fix the local model inventory first, then rerun.

3. Copy the emitted block into `~/.loci/backends.toml` (or the file pointed to by
`$LOCI_CONFIG`) and validate every assigned tag resolves:

```bash
ollama show <gen_model>
ollama show <verify_model>
ollama show <compress_model>
ollama show <guardian_model>
ollama show <redteam_model>
```

4. If you update additive catalog recommendations in `scripts/model_catalog.py`, keep
the benchmark and catalog tests green:

```bash
python3 -m pytest scripts/tests/test_bench_model_catalog_quality.py scripts/tests/test_model_catalog.py -q
```


### Implementation verification and phased rollout

See [docs/IMPLEMENTATION_VERIFICATION_AND_ROLLOUT.md](./IMPLEMENTATION_VERIFICATION_AND_ROLLOUT.md) for the end-to-end verification path that ties the benchmark harness, queue guardrails, acceptance gates, and audit-trace design together. The operational summary is:

1. Run the benchmark harness and persist the JSON summary from `scripts/bench_model_catalog_quality.py`.
2. Derive local role assignments with `scripts/assign_models_from_benchmark.py` and fail closed if a selected winner is not installed.
3. Gate rollout with the phases below: dry-run -> shadow -> canary -> default-on -> rollback-ready.
4. Verify each phase with the relevant pytest targets plus `ollama show` checks for the selected tags.
5. Keep an immutable decision record in the audit lane: router inputs, model choice, verification outcome, memory writes, and any rollback reason.

This is the implementation-ready path for the `benchmark-harness-spec`, `rollout-plan-spec`, `loci-implementation-verification-pass`, and `loci-audit-trace-architecture` work items.

### Memory store paths

| Variable | Default | Used by |
|---|---|---|
| `MNEMOSYNE_DB` | `~/.hermes/mnemosyne/data/mnemosyne.db` | ebbinghaus, amem, agentHER, memgas, score_trace |
| `STATE_DIR` | `~/.claude/hook-state` | all hooks, skill_annotation_updater, score_trace, exif |
| `SKILLS_DIR` | `~/.claude/skills` | skill_annotation_updater, skillops_maintenance, exif |
| `LOCI_STATE_DB` | `~/.hermes/state.db` | state_db_qdrant_sync |
| `LOCI_DOCS_ROOTS` | `LOCI_CODE_ROOT`, else the service's cwd | docs_ingest_indexer: `os.pathsep`-separated roots it may read under (symlink targets must stay inside). Set it to ingest docs from any other repo; relative paths resolve against the code root. A directory ingest reads at most 500 files and reports `truncated: true` with `files_found` when it hits that cap. |

### FlyBrain

FlyBrain (harness storage, brain-cluster training, promotion and rollback) moved to
the private repo `rjmendez/flybrain`; its operator runbooks live there. Loci no
longer reads `LOCI_FLYBRAIN_STORAGE_ROOT`, `HARNESS_*` or
`FLYBRAIN_CLUSTER_STATE_PATH`. See [FLYBRAIN.md](./FLYBRAIN.md) for what Loci still
owns (claim-scope and provenance validation for FlyBrain-derived findings).

### Tuning parameters

| Variable | Default | Effect |
|---|---|---|
| `HOOK_RECALL_TOP_K` | `3` (code default; `.env.example` sets `5`) | Max grounding results injected per turn |
| `HOOK_RECALL_MIN_SCORE` | `0.55` | Qdrant cosine threshold; lower = more noise |
| `FORGET_THRESH` | `0.3` | Entries with retention probability below this are re-embedded (0 = never, 1 = always); used by ebbinghaus_consolidation.py |
| `EBBINGHAUS_MAX_PER_RUN` / `AGENTHER_MAX_PER_RUN` / `AMEM_MAX_PER_RUN` | `50` / `20` / `100` | Max entries per tick, one name per script. `MAX_PER_RUN` is still honoured as a fallback for all three |
| `AMEM_LINK_THRESHOLD` | `0.88` | Cosine threshold for cross-link creation |
| `AMEM_CONFLICT_THRESHOLD` | `0.96` | Cosine threshold for conflict flagging |
| `SHADOW_THRESHOLD` | `0.92` | Cosine threshold for SHADOW_RISK pairs; used by skillops_maintenance.py |
| `AGENTHER_GEN_MODEL` | `llama3.2:latest` | Ollama model for failure relabeling |
| `EXIF_GEN_MODEL` | `llama3.2:latest` | Ollama model for skill gap analysis |
| `TOP_K_PER_LEVEL` | `3` | Results per level in MemGAS search; used by memgas_hierarchy.py |
| `LOCI_TOOL_WORKERS` | `1` | Worker threads that run sync MCP tools off the event loop (`mcp/tool_offload.py`). `1` keeps tools serial on one thread, as they were on the loop; raise only after checking the tools you call are thread-safe. The loop itself always stays free for `/health` and handshakes |
| `LOCI_LLM_DEADLINE_S` | `150` | Total budget for one `llm_local.generate()` call across the configured model, the discovered-model retry, the supervisor route, vLLM and cloud. Each attempt gets `min(OLLAMA_GEN_TIMEOUT, remaining)`; a tier starts only with >=5 s left. Exhausted calls return `ok: false, deadline_exceeded: true` |

---

### Coordination queue / cross-session handoff

Loci now supports a durable investigation-scoped coordination queue so parallel sessions can reserve and complete non-overlapping work without direct runtime interaction.

**Why this queue exists**
- Prevents silent overlap across active Copilot/Claude sessions.
- Captures ownership and lease expiry in machine-readable state.
- Keeps work pickup deterministic: discover, claim, complete, then verify queue state.

**Workflow**
1. Discover: `investigation_queue_status(investigation_id=...)` for current queue state.
2. Enqueue: `investigation_queue_enqueue(...)` with a stable item id, scope, and targets.
3. Claim lease: `investigation_queue_claim(...)` with `owner_session` and bounded `lease_seconds`.
4. Heartbeat/renew: re-run `investigation_queue_claim(...)` with the same owner before expiry.
5. Complete: `investigation_queue_complete(...)` with `state=done|blocked|cancelled`. To hand work back instead, `investigation_queue_release(...)` returns the item to `queued` for any session to claim.

**Conflict-avoidance rules**
- One owner per item while lease is active.
- Claims from other sessions fail unless the lease is expired.
- Completion by non-owners is rejected while another owner’s lease is still valid.
- Use explicit `dependencies` to serialize truly dependent work only: a claim is refused until every dependency exists and is `done`.
- A claim always carries a lease. A lease-less claim (legacy or imported) counts as expired; `investigation_queue_status` flags expired leases with `lease_expired: true` and claimable items with `available: true`.

**Example MCP calls**

```json
// enqueue a work item
{
  "investigation_id": "flock-re-hardening",
  "item_id": "wiki-correction-pass",
  "scope_kind": "file",
  "scope_targets": ["docs/wiki/backend-protocol.md"],
  "dependencies": [],
  "notes": "Add sendHello #8 and remove stale overclaims"
}
```

```json
// claim lease
{
  "investigation_id": "flock-re-hardening",
  "item_id": "wiki-correction-pass",
  "owner_session": "800cdfd9-0fd7-46e0-b292-71b6f556b025",
  "lease_seconds": 300
}
```

```json
// complete
{
  "investigation_id": "flock-re-hardening",
  "item_id": "wiki-correction-pass",
  "owner_session": "800cdfd9-0fd7-46e0-b292-71b6f556b025",
  "state": "done",
  "notes": "Updated docs + verified wording against findings"
}
```

Queue state is persisted on the investigation manifest under `coordination.items`; treat that manifest as the source of truth for item state, ownership, and lease expiry.

---
## Cron jobs

`cron/jobs.json` defines seven jobs; the six enabled ones are below.
`deep-think-loci-harvest` (`dtl_harvest.sh`, every 7d, `no_agent`) ships disabled
and is omitted from the table.

For orchestration policy (triggered consolidation + grooming + maintenance),
see [sleep-consolidation-scheduler-spec.md](./sleep-consolidation-scheduler-spec.md).

These jobs are driven by `scripts/hermes_cron_runner.py`, intended for a 1-minute
user timer or crontab entry. Issue #205 was a stale `next_run_at` loop in the live
gateway scheduler: it fast-forwarded overdue runs in memory, never persisted the new
time, and never executed the job. The runner here collapses backlog to one catch-up
run, then writes the next run back to `jobs.json` on the same tick. The grooming
tier below still runs from the user crontab separately.

Reference crontab line for the live profile copy:

```cron
* * * * * /home/rjmendez/development/loci/scripts/hermes_cron_runner.py --jobs-file ~/.hermes/profiles/edge/cron/jobs.json
```

| ID | Name | Interval | Script |
|---|---|---|---|
| `5872853d8b28` | mnemosyne-consolidation | 20m | `mnemosyne_activity_check.py` |
| `65355f0c518f` | mnemosyne-session-summarizer | 20m | `mnemosyne_activity_check.py` |
| `b40ae8101c2a` | mnemosyne-sleep-cli | 30m | `mnemosyne_sleep_all.sh` |
| `c857cd706f67` | mnemosyne-qdrant-sync | 30m | `mnemosyne_qdrant_sync.py` |
| `a9fc1ea0886a` | state-db-qdrant-sync | 5m | `state_db_qdrant_sync.py` |
| `f3a4d7c9b8e1` | proactive-self-model-loop | 5m | `self_model_trigger_eval.py` |

`scripts/hermes_cron_runner.py` now rejects absolute script paths, parent-traversal
segments, and any resolved script outside `<root>/scripts`; invalid job entries are
persisted as `last_status=error` with reason instead of being executed.

**mnemosyne-consolidation** and **mnemosyne-session-summarizer** both use
`mnemosyne_activity_check.py` as the pre-flight gate script. They differ in the
agent prompt: consolidation runs a lightweight `mnemosyne_sleep` pass, while the
session-summarizer archives structured session facts, triples, and scratchpad state.

**mnemosyne-sleep-cli** (`no_agent: true`) invokes the Mnemosyne CLI directly via
shell and runs consolidation across all configured banks without spawning an LLM
agent.

**proactive-self-model-loop** refreshes the durable self-model, writes the
introspection snapshot, and emits T1/T2/T3 alerts for queue floods, stale
investigations, blocked work, and overdue daily summaries. It is intentionally
fail-open and silent when idle.

---

## Passive grooming tier

`scripts/loci_groom.py` holds eight passes: `index`, `tags`, `recall`, `knn_tags`,
`codelink`, `verify`, `reflect`, `summaries`. Only `index` is `applyable` — every
other pass writes proposals under `$LOCI_MEMORY_DIR/_groom/` and never touches
`findings.jsonl`.

Four are scheduled, from the user crontab, through `scripts/loci_groom_cron.sh`:

| When | Pass | What it does | Last measured |
|---|---|---|---|
| `17 */6 * * *` | `index --apply` | upserts findings present on disk but missing from Qdrant; upsert-only, never deletes | on_disk 2922, indexed 2769, coverage 0.9476 |
| `20 3 * * *` | `knn_tags` | proposes tags from k-NN neighbours | vocabulary 60, generated 18, proposed 0 |
| `40 3 * * *` | `codelink` | proposes finding→symbol links | 11,273 symbols, 718 links generated |
| `50 4 * * *` | `summaries` | fills the investigation summary ladder | already_had 137, nothing_to_say 5, errors 0 |

`tags` is deliberately unscheduled: `knn_tags` measured better at a fraction of
the cost. `verify` is **not scheduled and is not fit to schedule** — the fixed
benchmark in `eval/verify_skeptic_eval.py` measured 22% false refutation on main,
and five prompt/guard variants were all neutral or worse (#231). A false
"refuted" on a true finding is what later readers act on.

The wrapper's exit codes:

| Code | Meaning |
|---|---|
| 0 | ok |
| 1 | a pass raised |
| 3 | a pass **refused** or **degraded** |

The refusal that matters: `connect()` reads `_retention_days()` before touching
Qdrant and refuses when it is not 0, because `_get_qdrant()` runs the startup
purge on its first call in a process. Grooming re-indexes findings; with a non-zero
retention window that becomes an index-then-delete loop that reports success. The
wrapper prints the pass's own reason on stderr and appends one line per run to
`$LOCI_GROOM_STATE/runs.jsonl` (default `~/.loci/groom/runs.jsonl`), so "did this
ever actually run" has an answer.

Per-run ceilings: `LOCI_GROOM_VERIFY_INVESTIGATIONS` (5),
`LOCI_GROOM_VERIFY_FINDINGS` (10), `LOCI_GROOM_SUMMARY_INVESTIGATIONS` (12),
`LOCI_GROOM_REFLECT_ITEMS` (3), `LOCI_GROOM_BATCH` (16). `LOCI_GROOM_MODEL` is
unset on purpose — the vLLM and Ollama tiers name the same model differently, so
each tier resolves its own.

---

## MLOps loop

`mlops/loop.py` rebuilds the grounding dataset, retrains the classifier
ensemble, canary-evaluates the candidate, and promotes it if it beats the
baseline. It ran for the first time on 2026-08-29; before that it had queued
against a self-hosted runner that did not exist for 67 consecutive nights, and
every child process was unbounded (#238).

Scheduled from the user crontab through `scripts/mlops_loop_cron.sh`:

| When | What |
|---|---|
| `0 2 * * *` | full loop, into an isolated worktree |

**The loop promotes by writing into the repo.** `grounding_bleed_clf.joblib`,
`grounding_dataset.jsonl`, and `metrics.json` under `deep_think_loci/grounding/`
are all tracked, so running the loop in a working checkout leaves that checkout
dirty — how a 12,684-row rebuilt dataset ends up in an unrelated commit. The
wrapper runs it in `~/.loci/mlops/worktree`, reset to `origin/main` each night,
and prints what changed. **Nothing is applied automatically**; adopting a
promoted model is a deliberate commit.

Roughly twenty minutes end to end, the bulk of it GradientBoosting under 10-fold
CV. Each step is bounded at `LOCI_MLOPS_STEP_TIMEOUT` (3600s), and a step that
times out comes back as returncode 124 rather than hanging the run. Child output
is streamed as it arrives, prefixed with the script that produced it.

| Code | Meaning |
|---|---|
| 0 | ok — including steps deliberately skipped (Ollama unreachable, nothing new) |
| 1 | one or more steps **failed**, named in the summary line |

A skip is not a failure, and the log says which it was: a cadence-gated step that
could not run because the backend was unreachable says so rather than reciting
the cadence.

See [grounding-corpus-limits.md](grounding-corpus-limits.md) before reading much
into the scores — the CV figure it prints is a pair-level split and optimistic by
about two thirds of its margin over cosine.

---

## Manual operations

The commands below assume this repo is `loci` (github.com/rjmendez/loci):

```bash
LOCI=~/development/loci                              # your checkout
LOCI_PY=~/.hermes/hermes-agent/venv/bin/python3    # interpreter with the deps
```

`scripts/loci_groom_cron.sh` uses `$LOCI/mcp/.venv/bin/python` instead, and
`eval/run_eval.sh` reads `$LOCI_PY` with the same default as above.

### Rebuild MemGAS 3-level index

```bash
$LOCI_PY $LOCI/scripts/memgas_hierarchy.py --index
```

Run after major Mnemosyne consolidation, or when memgas_l1/l2/l3 collections get stale.

### Run MemGAS search

```bash
$LOCI_PY $LOCI/scripts/memgas_hierarchy.py --search "your query here"
```

### Detect skill shadows

```bash
OLLAMA_URL=http://localhost:11434 \
$LOCI_PY $LOCI/scripts/skillops_maintenance.py
```

Review SHADOW_RISK pairs. For sim=1.000 pairs: one is usually a duplicate install or
has an empty description — populate a distinctive description.

### Discover skill gaps (EXIF)

```bash
STATE_DIR=~/.claude/hook-state \
$LOCI_PY $LOCI/scripts/exif_skill_discovery.py
```

Review `~/.claude/hook-state/exif_discoveries.jsonl` for candidates. Promote manually:
```bash
cp -r ~/.claude/hook-state/candidate_skills/SKILLNAME ~/.claude/skills/SKILLNAME/
```

### Build SCoRe fine-tuning dataset

```bash
$LOCI_PY $LOCI/scripts/score_trace_collector.py
cat ~/.hermes/mnemosyne/data/score_traces/manifest.json
```

When `ready_for_sft: true` (≥ 10 correction pairs), the dataset is usable for SFT.

### Run eval harness

```bash
$LOCI/eval/run_eval.sh
```

Runs three scorers in sequence — `harness.py`, `grounding_gate_eval.py`, and
`grounding_gate_qf_eval.py` — and passes its arguments through to each.

> The CV figure `mlops/grounding/train.py` prints is a pair-level split and is
> optimistic by roughly 2/3 of its margin over cosine. See
> [grounding-corpus-limits.md](grounding-corpus-limits.md) for the leak-free
> numbers and what more findings are actually worth.

### Run executable chaos/adversarial hardening gates (pre-merge / pre-deploy)

Generate fresh artifacts first (for example a chaos run log plus the red-team report),
then gate them with one machine-readable check:

```bash
$LOCI_PY $LOCI/scripts/chaos_hardening_gate.py \
  --chaos-events artifacts/chaos/latest-events.jsonl \
  --adversarial-report scripts/redteam/reports/latest-sandbox-report.json \
  --max-timeout-rate 0.05 \
  --min-retry-recovery-rate 0.80 \
  --max-duplicate-effect-rate 0.0 \
  --min-provenance-completeness 0.99 \
  --max-candidate-bypass 0
```

The script prints JSON with explicit per-gate `pass`/`fail`/`skipped` states and
returns:

- `0` when all evaluated gates pass
- `1` when any gate fails (or when `--fail-on-skipped` is set and a gate is skipped)
- `2` on invocation/config errors

This makes it suitable for CI or release pipelines:

```bash
$LOCI_PY $LOCI/scripts/chaos_hardening_gate.py ... > hardening-gate.json
```

### Benchmark local Ollama generation honestly

`scripts/bench_local_models.py` is the latency/throughput harness for the local
generation tier. It does explicit discarded warmups, reports mean/stddev plus
`p50`/`p90`/`p99`, writes stable JSON, and only reports TTFT when it is actually
measuring streamed chunks instead of guessing.

```bash
$LOCI_PY $LOCI/scripts/bench_local_models.py \
  --model qwen2.5:3b \
  --concurrency 1 4 8 \
  --warmup 3 \
  --trials 20 \
  --output bench-local-models-qwen25.json
```

Use this when you need to check whether swarm fan-out is really parallel on the
current Ollama host or only appears parallel at the caller.

### Prefilter bulky agent output before cloud synthesis

`scripts/agent_output_prefilter.py` is an **opt-in** helper for workflows that
fan out to local/background agents, collect long result blobs, then hand those
results to an expensive cloud model for the final synthesis.

It composes the existing local-model tier:

- `semantic_relevance` keeps the chunks that best match the task context
- `semantic_dedup` collapses repeated chunks when embeddings are available
- `compress_text` condenses the kept span under a character budget

It always preserves the original text in the returned JSON. If the local tier
degrades, errors, times out, or produces a confidence score below the threshold,
the script fails open and returns the original text unchanged with
`fallback_used: true`.

```bash
cat agent-output.txt | \
$LOCI_PY $LOCI/scripts/agent_output_prefilter.py \
  --context "summarize the build blockers for PR review" \
  --threshold 0.6
```

### Swarm reasoning tool

`scripts/swarm_escalate.py` is now also exposed directly as the MCP tool
`swarm_reason`, so downstream MCP clients can invoke the 4-stage fan-out →
triage → selective escalation → synthesis chain without shelling out. The tool
returns one structured JSON object with `schema_version`, `topic`, `findings`,
`summary`, `stats`, plus diagnostic fields (`decomposition`, `triage`, `tiers`,
`lineage`). Like the script, it is fail-open: import/runtime errors degrade into
well-formed JSON instead of raising across the MCP boundary.

#### Production setup (swarm-first)

1. Ensure local generation lane is reachable (`LOCI_OLLAMA_GEN_URL`/`OLLAMA_GEN_URL`)
   and all configured tags resolve (`ollama show <tag>` for cheap/escalate/synthesize).
2. Optional batched lane: set `VLLM_BASE_URL` (and role-specific `VLLM_MODEL_*` if used).
3. Set explicit swarm defaults in env/backends config as needed:
   - `LOCI_SWARM_CHEAP_MODEL`
   - `LOCI_SWARM_ESCALATE_MODEL`
   - `LOCI_SWARM_SYNTHESIZE_MODEL`
   - `LOCI_SWARM_DECOMPOSE_MODEL`
4. For MCP capacity guardrails, tune:
   - `LOCI_SWARM_MAX_INFLIGHT`
   - `LOCI_SWARM_MAX_FANOUT`
   - `LOCI_SWARM_MAX_SEEDS`
   - `LOCI_SWARM_MAX_SELF_CONSISTENCY_SAMPLES`
   - `LOCI_SWARM_MAX_REDUCE_GROUP_SIZE`

#### Reproducible run flow

```json
{
  "schema_version": 1,
  "topic": "Explain auth cache miss spikes after deploys",
  "findings": [
    {
      "subtask": "Check invalidation timing",
      "answer": "...",
      "confidence": "medium",
      "tier_reached": "escalated",
      "model": "...",
      "ok": true
    }
  ],
  "summary": "...",
  "stats": {
    "fanout_count": 12,
    "escalated_count": 3,
    "escalation_rate": 0.25
  }
}
```

CLI baseline (single seed, deterministic rollout baseline):

```bash
python3 scripts/swarm_escalate.py \
  "Explain auth cache miss spikes after deploys" \
  --fanout-count 12 \
  --pretty > artifacts/swarm-auth-cache-baseline.json
```

Canary expansion (explicitly opt in to stronger gates):

```bash
python3 scripts/swarm_escalate.py \
  "Explain auth cache miss spikes after deploys" \
  --fanout-count 12 \
  --seeds 3 \
  --self-consistency-samples 3 \
  --escalate-with-prior-context \
  --pretty > artifacts/swarm-auth-cache-canary.json
```

The script exits non-zero when `validate_swarm_result(...)` fails. Persist the JSON
artifact from each run and compare `stats`, `triage`, and `lineage` before promoting.

#### Validation gates before promotion

```bash
python3 -m pytest scripts/tests/test_swarm_escalate.py -q
python3 -m pytest mcp/tests/test_swarm_reason_tool.py -q
python3 -m pytest mcp/tests/test_backends.py -q
```

If you changed tier routing/defaults, also run:

```bash
python3 -m pytest scripts/tests/test_model_catalog.py -q
```

#### Failure handling and rollback

| Symptom | Likely cause | Operator action |
|---|---|---|
| `degraded: true` with summary "Swarm reasoning degraded..." | wrapper/import/runtime failure in `swarm_reason` | Keep artifact, treat as non-authoritative, rerun CLI `swarm_escalate.py` directly to isolate |
| `swarm_reason busy: global inflight limit ... reached` | `LOCI_SWARM_MAX_INFLIGHT` saturated | Retry after queue drains, or temporarily raise `LOCI_SWARM_MAX_INFLIGHT` |
| Multi-seed run becomes slower than baseline | vLLM probe inconclusive or batched lane missing requested model | Pin `--seeds 1` or set `LOCI_SWARM_AUTO_PARALLEL=0`; verify `/v1/models` serves requested tags |
| Many escalations with low confidence | cheap tier underpowered for workload | raise `self_consistency_samples`, enable `--escalate-with-prior-context`, or temporarily pin stronger `--cheap-model` |
| Empty/unparseable synthesis in think mode | reasoning consumed synthesis budget | keep `--synthesize-think` optional; built-in fallback already retries with normal synthesis |

#### Operational procedures

1. **Daily production mode:** run single-seed baseline for routine jobs.
2. **Canary mode:** enable one tier knob at a time (`seeds`, `self-consistency`, then `synthesize-think`).
3. **Incident containment:** force deterministic low-risk mode (`--seeds 1`, no `--synthesize-think`, no consensus), archive failing JSON, open follow-up.
4. **Rollback defaults:** pin legacy models with `LOCI_SWARM_ESCALATE_MODEL=qwen3.8:latest` and `LOCI_SWARM_SYNTHESIZE_MODEL=qwen3.8:latest`, then re-run validation gates.

Query longitudinal scores:

```bash
curl -s -X POST $QDRANT_URL/collections/eval_scores/points/scroll \
  -H "api-key: $QDRANT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"limit": 50, "with_payload": true, "with_vector": false}' \
  | python3 -c "import json,sys; pts=json.load(sys.stdin)['result']['points']; [print(p['payload']['run_date'], p['payload']['task_name'], p['payload']['score']) for p in sorted(pts, key=lambda x: x['payload']['run_date'])]"
```

### Force Mnemosyne → Qdrant sync

```bash
QDRANT_API_KEY=$QDRANT_API_KEY \
$LOCI_PY $LOCI/scripts/mnemosyne_qdrant_sync.py
```

This only adds and re-embeds. To also delete points whose memory is gone from SQLite, pass
`--prune`; it requires `HERMES_AGENT_ID` and `HERMES_PROFILE` to be set and refuses to delete
more than half of this host's points in one run unless `--force-prune` is also given.

### Run a groom pass by hand

```bash
$LOCI/scripts/loci_groom_cron.sh index          # dry run: reports coverage, writes nothing
$LOCI/scripts/loci_groom_cron.sh index --apply  # upsert the missing findings
```

Exit 3 means the pass refused or degraded; the reason is on stderr. See
[Passive grooming tier](#passive-grooming-tier).

### Check the adversarial skeptic

```bash
$LOCI/mcp/.venv/bin/python $LOCI/eval/verify_skeptic_eval.py [trials] [votes]
```

A fixed, self-contained case set — every case carries its own context, so no
commit to this repo can falsify a label. The headline number is FALSE REFUTATION,
not accuracy: "uncertain" leaves a finding unverified and is harmless, "refuted"
on a true claim is the damage.

### Offload tool loop

`offload_tool_loop` lets the local Ollama model work a multi-step, read-only tool
workflow so the calling cloud model does not pay for the intermediate steps
(issue #376, MVP). The local model emits one JSON intent per turn; every intent is
validated against a closed registry in `mcp/offload_loop.py` (`TOOL_SPECS`), executed
under budgets, and fed back as untrusted data. The registry is code: env vars and
arguments can only remove tools, never add them. No cloud model is called inside the
loop; a run that cannot finish returns `status="fallback"` plus a compact `handoff`
and the caller continues from that.

Read-only tools: `investigation_search`, `investigation_entity_lookup`,
`investigation_list`, `investigation_load`, `memory_health`, `code_graph_query`
(stricter Cypher guard: MATCH/WITH/UNWIND/RETURN only, no `;`, no CALL/LOAD/etc.).
Model-supplied `investigation_id`s must already exist. `investigation_id=` on the tool
pins the run to one investigation: it overwrites the `investigation_id` argument of every
tool that has one, and removes the tools that cannot be scoped (`investigation_list`,
`code_graph_query`) from the allowlist for that run (they are also denied as
`pinned_unscoped`).

| Env var | Effect |
|---|---|
| `LOCI_OFFLOAD_DISABLE=1` | tool returns fallback `disabled` |
| `LOCI_OFFLOAD_TOOLS=a,b` | narrows the allowlist (never widens) |
| `LOCI_OFFLOAD_AUDIT_DIR` | audit directory (default `<memory dir>/../audit/offload`) |
| `LOCI_OFFLOAD_AUDIT_FULL_PROMPTS=1` | also store every full prompt (default: sha256 + length) |

Budgets default to 8 steps / 8 tool calls / 120 s / 32 KiB fed back, and are clamped
to hard ceilings of 20 / 20 / 300 s / 256 KiB. Per-tool output is capped (4 KiB) and
marked `[truncated N bytes]`; a prompt over 16 KB stops the run rather than being
silently truncated by Ollama.

Stop reasons (`reason`; only `finished` is `status="done"`): `finished`, `gave_up`,
`max_steps`, `max_tool_calls`, `timeout`, `output_budget`, `prompt_budget`, `bad_turns`
(3 consecutive unparseable replies), `denied_streak` (3), `repeat_call`, `no_progress`
(3 identical results), `tool_error_streak` (2), `tool_timeout` (2 abandoned calls),
`model_unavailable` (transport failure or empty reply; a reply cut off mid-JSON counts as a bad turn instead), `approval_required` (a non-read-only
spec was requested; never executed), `audit_unavailable`, `no_tools_allowed`,
`disabled`, `tools_unbound`, `unknown_investigation`, `bad_task`, `wrapper_exception`.

Audit: one file per run, `offload-<YYYY-MM-DD>-<run_id>.jsonl`, one JSON record per
event (`run_start`, `model_call`, `model_call_result`, `intent`, `decision`,
`tool_result`, `run_end`) with `v`, `run_id`, `seq`, `ts`. The `decision` record is
written before the tool executes; if it cannot be written the tool is not run. Purge
by deleting old files. `offload_loop.aggregate_metrics(dir, days=7)` summarises
`run_end` records (no MCP tool for it yet).

Warm the model first (`scripts/gpu_warm.py`): a ~70 s cold load otherwise consumes the
elapsed budget. `python scripts/offload_demo.py` runs the loop offline with a scripted model (mechanics only).

Token figures in `metrics` are ESTIMATES, not billed tokens: `est_tokens_local =
(prompt + completion bytes) // 4` is the local model's traffic and `est_tokens_returned =
returned_bytes // 4` is what the caller pays to read the result. No cloud saving or
baseline is reported: a number derived from the local model's own traffic says nothing
about what a cloud loop would have cost, and a fallback run may cost the cloud more.
Measuring that needs a real cloud-driven run of the same task.

Status of the acceptance criteria: the loop mechanics are tested offline with a scripted
model and fake tools (`mcp/tests/test_offload_loop.py`, `scripts/offload_demo.py`).
Criterion 1 (a real local lane completing a multi-step workflow) and criterion 5 (a
demonstrated reduction in cloud token spend) are NOT demonstrated: no warmed-lane run
against a real investigation is recorded yet.

Residual risks: the `answer` is the local model's unverified claim
(`answer_provenance: local_model_unverified`); Mnemosyne recall inside
`investigation_search` is an external package and may bump access counters; a tool that
times out leaves an abandoned worker thread (capped at 2 per run); the audit has no
secret redaction or hash chain; small local models may fall back often (safe, but less
saving). Out of scope for this MVP: planner/executor split, swarm intent voting, write
tools, approval tokens, cloud-vs-offload dashboards. Issue #376 is only partly done.

### Brain-cluster operations (moved)

The brain-cluster runbooks that used to be here (trainlog privacy scrub, feedback
retraining gate, preflight/deploy/rollback/incident checklists, and the
queue/drift/threshold/promotion alert playbooks) moved to `rjmendez/flybrain` with
the code they operate. The in-server `flybrain_cluster_*` / `flybrain_expert_inspect`
MCP tools were removed from Loci; see [FLYBRAIN.md](./FLYBRAIN.md).

---

## Claude Code hooks

This repo ships three hooks in `scripts/hooks/`. Each reads a JSON payload on
stdin and exits 0 on any event name it does not recognise.

| Script | Events accepted | Purpose |
|---|---|---|
| `pre_llm_grounding.py` | `UserPromptSubmit`, `SubagentStart` (and legacy `pre_llm_call` / `PreLlmCall`) | per-turn Qdrant grounding injected into the prompt |
| `pre_tool_grounding.py` | `PreToolUse` (and legacy `pre_tool_call`) | tool-call audit |
| `session_end_sync.py` | `Stop` — reads `transcript_path` from the payload | session → `loci_sessions` |

Install and drift-check them with `scripts/hooks/install.sh`:

```bash
scripts/hooks/install.sh          # copy repo -> ~/.claude/hooks, backing up what is there
scripts/hooks/install.sh --check  # report drift, exit 1 if any, change nothing
```

Run `--check` before trusting a hook. The deployed and repo copies have diverged
silently before — a hand-edited `pre_tool_grounding.py` accepted `PreToolUse`
while the repo copy accepted only the Hermes name, so a fresh install would have
disabled the hook.

`--check` also reads `~/.claude/settings.json` (`$CLAUDE_SETTINGS`) and prints
`UNMANAGED` for any hook an event invokes out of the hooks dir that this repo does
not ship — on the reference host, the `Stop` wrapper `session_end_sync.sh`, which
supplies `QDRANT_URL` and the embedding endpoint and exists in no commit. Those
lines do not affect the exit status; only file drift does. Grading only the
shipped files meant `--check` said "hooks in sync" without looking at the file the
`Stop` hook actually runs.

Two payload shapes to get right when writing a new hook; both were wrong until
#228, and all three hooks ran, exited 0, and did nothing:

- Claude Code puts the prompt text at the **top level** as `prompt`.
  `extra.user_message` is the Hermes shape and is empty under Claude Code.
- A named-vector Qdrant **search** takes `{"name": "dense", "vector": [...]}`.
  `{"dense": [...]}` is the **upsert** shape and returns HTTP 400.

`scripts/install_hooks.sh` is unrelated — it symlinks the git `post-commit` hook
into `.git/hooks`.

---

## Portability (new machine setup)

No infra address or path is hardcoded. To stand up on a new machine:

1. `cp backends.toml.example ~/.loci/backends.toml` and fill in the endpoints and
   keys for this machine — Ollama, vLLM, Qdrant, embed/rerank models, memory dir.
   This is the durable channel: it needs no third-party import and no launcher
   that remembers to export anything. Leave a section blank on a laptop that
   runs its own Ollama; the local probe finds it.
   For cloud/offload setup, use `scripts/loci_setup_verify.py`:

   ```bash
   # Apply values from local key files into ~/.loci/backends.toml then verify.
   python scripts/loci_setup_verify.py \
     --apply \
     --cloud-tier on \
     --openrouter-key-file ~/.openrouter \
     --abliteration-key-file ~/.abliteration \
     --openrouter-url https://openrouter.ai/api/v1 \
     --abliteration-url https://api.abliteration.ai/v1

   # Verify only (no writes).
   python scripts/loci_setup_verify.py --config ~/.loci/backends.toml --check-tmux-sessions
   ```

   Safety notes:
   - API keys are read from local key files and written to local user config only
     (`~/.loci/backends.toml` by default).
   - The tool refuses `--apply` to repo-tracked paths.
   - The tool reports key presence/missing status only; it does not print key values.
2. `scripts/hooks/install.sh` to place the three hooks in `~/.claude/hooks`.
3. Register them in `~/.claude/settings.json`. Hook paths there are absolute —
   JSON does no `$HOME` expansion.
4. `mkdir -p ~/.claude/hook-state`
5. Qdrant credentials go in `~/.loci/backends.toml` under `[qdrant]`, or in
   `~/.claude/settings.json` at `mcpServers.loci.env.QDRANT_API_KEY`.

The one setting worth checking by hand on a new machine is
`LOCI_QDRANT_RETENTION_DAYS`. The code default is 0, and 0 is safe, so a fresh
install needs nothing — but a stray non-zero value anywhere in the chain makes
the first Qdrant call of every process delete findings.

---

## #383 deploy runbook

Deploys the honesty-audit fixes (#383) and their follow-ups together. Run it
top to bottom on the host that runs `loci-mcp`. Paths below are the defaults:
`LOCI_MEMORY_DIR=~/.loci/memory-sessions`, the user unit
`~/.config/systemd/user/loci-mcp.service`. Substitute yours.

What changes behaviour on deploy (read before starting):

- **ACL identity.** `requesting_agent_id` (and `memory_route`'s `agent_id`) can
  only *narrow* access now. The identity the ACL checks is the transport-bound
  one: a per-agent bearer token from `LOCI_MCP_AGENT_TOKENS` on the HTTP
  transports, or a `/bootstrap` session token on A2A. With stdio, a shared
  `LOCI_MCP_TOKEN`, or unauthenticated loopback, there is nothing to bind, and
  the caller is the process identity `HERMES_AGENT_ID`. **Limitation:** on
  those transports every client is the same agent, so ACLs separate
  investigations between *deployments* (processes), not between clients of one
  process. To separate clients, give each one its own token (see step 6).
  `grounding`, `memory_route`, `investigation_as_of`, `rag_context_search` and
  `memory_surface` are gated now, and each reports `excluded_acl`.
- **Retraction propagation.** `memory_retract` (applied) sets `retracted: true`
  on the finding's own Qdrant point payload and stamps `valid_until` on
  matching Mnemosyne `working_memory` / `episodic_memory` rows. `memory_restore`
  clears exactly what retract wrote. Nothing is deleted. The legacy Mnemosyne
  `memories` table has no lifecycle column: its rows are counted in the reply
  (`legacy_unflagged`) and left alone, so `mnemosyne_qdrant_sync.py` can still
  copy those texts into the `mnemosyne` collection. `LOCI_RETRACT_PROPAGATE=0`
  turns propagation off. The reply's `propagation` block reports per-store
  status. `failed` means the tombstone applied but that store was not updated.
- **memory_promote** returns `ok:false, retryable:true` when the Qdrant upsert
  did not land. Re-running it with the same tier retries the index write.
- **docs_recall / docs_search** return the real lexical score (1.0 for a phrase
  match, else the fraction of query tokens matched) with `score_kind:
  "lexical"`. The fixed 0.95 is gone, and hits are ranked by that score.
- **Coordination queue** writes hold `<investigation>/.lock`. A contended call
  returns `{"error": "busy", "retryable": true}` instead of silently losing the
  write.

### 1. Stop the service

```bash
systemctl --user stop loci-mcp
systemctl --user is-active loci-mcp   # expect: inactive
```

Stop anything else that writes the store too: the A2A server, cron grooming
(`loci_groom_cron.sh`), `mnemosyne_qdrant_sync.py` and reflection-loop ticks.

### 2. Back up `~/.loci`

```bash
TS=$(date -u +%Y%m%dT%H%M%SZ)
tar -C ~ -czf ~/loci-backup-$TS.tgz .loci
tar -tzf ~/loci-backup-$TS.tgz | head      # sanity check it is readable
```

If the Mnemosyne database will take retraction stamps (it will, whenever
`MNEMOSYNE_DATA_DIR` holds a `mnemosyne.db`), back it up as well:
`cp "$MNEMOSYNE_DATA_DIR/mnemosyne.db" ~/mnemosyne-backup-$TS.db`.

### 3. Migrate legacy access rows (dry run, then apply)

```bash
PY=~/development/loci/mcp/.venv/bin/python
$PY scripts/migrate_access_rows.py --memory-dir ~/.loci/memory-sessions          # dry run
$PY scripts/migrate_access_rows.py --memory-dir ~/.loci/memory-sessions --apply  # writes findings.jsonl.bak-<ts> first
```

Check that the dry-run counts match what `--apply` reports.

### 4. Inspect, then move aside, the legacy `undefined` investigation dir

Older builds wrote findings for callers that passed the literal string
`undefined` as an investigation id. That directory fails id validation, is
listed as `malformed` by the recall filter, and keeps `memory_health`'s
`retraction_integrity` at `warn`.

```bash
D=~/.loci/memory-sessions/undefined
ls -la "$D"; wc -l "$D"/*.jsonl
head -c 2000 "$D/findings.jsonl"      # decide whether anything in it is worth re-filing
mkdir -p ~/.loci/quarantine
mv "$D" ~/.loci/quarantine/undefined-$TS
```

Move it outside `memory-sessions`: global scans walk every directory under
that root. Do not delete it. Re-file anything worth keeping through
`investigation_store` under a real id once the service is back.

### 5. Set `LOCI_DOCS_ROOTS` if docs ingest reads outside the code root

`docs_ingest_indexer` only reads under `LOCI_DOCS_ROOTS` (`:`-separated), or under
the code root when that is unset. If you ingest docs from anywhere else, add
the variable to the unit:

```bash
systemctl --user edit loci-mcp
# [Service]
# Environment=LOCI_DOCS_ROOTS=/home/<you>/development/loci/docs:/home/<you>/notes
```

### 6. (Optional) Bind MCP client identity

If more than one agent talks to this server and ACLs should tell them apart,
give each agent its own bearer token. Keep the tokens out of the unit file:

```bash
install -m 600 /dev/null ~/.loci/agent-tokens.json
# {"agent-a": "<python3 -c 'import secrets;print(secrets.token_hex(32))'>", "agent-b": "..."}
systemctl --user edit loci-mcp
# [Service]
# Environment=LOCI_MCP_AGENT_TOKENS_FILE=%h/.loci/agent-tokens.json
```

Each client then sends `Authorization: Bearer <its token>`. The server refuses
to start if the file does not parse, or if `LOCI_MCP_TOKEN` is also used as an
agent token.

### 7. Backfill provenance tiers (dry run, then apply)

```bash
$PY scripts/backfill_provenance_tiers.py --memory-dir ~/.loci/memory-sessions          # per-rule counts, writes nothing
$PY scripts/backfill_provenance_tiers.py --memory-dir ~/.loci/memory-sessions --apply  # backs up, then appends
```

The dry run prints how many findings each rule would tag and how many stay
untagged (`no_unambiguous_rule`, `conflict`, `explicit_tier`). `--apply` writes
only `<inv>/provenance_updates.jsonl` (append-only). It first copies every log
it will append to, plus the planned records and a `ROLLBACK.txt`, into
`~/.loci/backups/provenance-backfill-<ts>/`. A second run tags nothing
(`already_backfilled`).

### 8. Restart

```bash
systemctl --user daemon-reload     # only if you edited the unit
systemctl --user start loci-mcp
systemctl --user is-active loci-mcp
journalctl --user -u loci-mcp -n 50 --no-pager   # no tracebacks; note the token log line if step 6 ran
```

Restart the A2A server and re-enable any cron jobs you stopped in step 1.

### 9. Post-deploy checks

Run these through an MCP client connected to the service:

1. `loci_health` returns `status: ok` (or `degraded` only for a backend you
   know is down). `code_version` matches the deployed commit.
2. `memory_health` returns `retraction_integrity: ok`. `warn` here usually means
   the `undefined` dir is still under `memory-sessions` (step 4).
3. A `pre_answer_check` sanity probe. Use `record=false` so the probe leaves no
   trace. Pick an investigation with a known tool-verified finding:
   `investigation_pre_answer_check(investigation_id=<id>, claims=[<that finding's text>], record=false)`
   should support the claim. A made-up claim should come back in
   `unsupported_claims`, and a claim that only a `[reasoned]` finding supports
   should come back `provenance_blocked`.
4. `docs_recall("<a phrase you know is indexed>")` returns `score: 1.0` and
   `score_kind: "lexical"`.
5. If step 6 ran: from agent-a's token, `investigation_load` on an investigation
   whose ACL excludes agent-a returns `permission_denied`, even with
   `requesting_agent_id` set to a member.

### 10. Rollback

1. `systemctl --user stop loci-mcp`
2. Check out the previous release in the service's checkout, or `git revert`
   the merge.
3. Restore the store: `rm -rf ~/.loci && tar -C ~ -xzf ~/loci-backup-$TS.tgz`.
   This undoes steps 3, 4 and 7 together. To undo only the backfill, follow
   `ROLLBACK.txt` in the backfill backup dir. The appended log is the only file
   it touched.
4. Qdrant points flagged by retractions made *after* the deploy keep
   `retracted: true` in their payload. The previous release ignores that key,
   so no action is needed. To clear it anyway, `memory_restore` each finding
   before rolling back.
5. If you restored the Mnemosyne backup, stop Mnemosyne writers first. The
   stamps are plain `valid_until` values, so the previous release reads them as
   Mnemosyne's own expiry.
6. Remove any `LOCI_MCP_AGENT_TOKENS*`, `LOCI_DOCS_ROOTS` or
   `LOCI_RETRACT_PROPAGATE` lines you added to the unit, then run
   `systemctl --user daemon-reload && systemctl --user start loci-mcp`.

---

## Known issues and limitations

| Issue | Severity | Workaround |
|---|---|---|
| `settings.json` hook paths are hardcoded (no `$HOME` expansion in JSON) | LOW | Manual edit on new machine |
| Ebbinghaus timestamp format warnings for microsecond ISO strings | LOW | Non-fatal; entries fall back to 30-day default decay |
| `eval/harness.py` mean_score=0.167 baseline is low | INFO | Keyword matching is strict; trend matters more than absolute value |
| MemGAS index takes ~5min for 500+ entries (sequential embed) | MED | Run `--index` in off-hours; add batch embedding |
| agentHER requires a generative Ollama model to be available | MED | Set `AGENTHER_GEN_MODEL` to your installed model (default: `llama3.2:latest`) |
| The `verify` groom pass false-refutes 22% of true claims | HIGH | Do not schedule it. Measured by `eval/verify_skeptic_eval.py`; five prompt/guard variants were neutral or worse (#231) |
| Hermes cron jobs can fast-forward forever if a stale `next_run_at` is never persisted | MED | Use `scripts/hermes_cron_runner.py`, which executes one catch-up run and writes the future `next_run_at` on the same tick (#205) |
| `backends.toml.example` has no `[qdrant] retention_days` key | LOW | The key is read (`qdrant_ops._retention_days`) but not shown in the example; the code default of 0 applies |
| SCoRe `corrections=0` until sessions accumulate overlap | INFO | Corrections require same-session failure→success pairs; grow naturally |
