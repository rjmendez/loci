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

### FlyBrain harness write-path safety

FlyBrain harness jobs must follow the allowlist-only path boundary in
[FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md](./FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md).

Operationally:

- Keep the effective write root at `LOCI_FLYBRAIN_STORAGE_ROOT` (or stricter approved override).
- Block symlink/junction (reparse-point) escapes before any write/delete/move.
- Never run destructive operations outside the allowlisted root.
- Use the durable root structure in [FLYBRAIN_HARNESS_STORAGE_LAYOUT.md](./FLYBRAIN_HARNESS_STORAGE_LAYOUT.md): `graph\`, `snapshots\`, `cache\`, `backups\`, and `logs\` under `LOCI_FLYBRAIN_STORAGE_ROOT`.
- Sequence operational gates using [FLYBRAIN_HARNESS_ROLLOUT_MILESTONES.md](./FLYBRAIN_HARNESS_ROLLOUT_MILESTONES.md), starting with `dry-run` inventory and ending at the first reproducible local query harness run.

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
5. Complete/release: `investigation_queue_complete(...)` with `state=done|blocked|cancelled` (or `investigation_queue_release(...)` alias).

**Conflict-avoidance rules**
- One owner per item while lease is active.
- Claims from other sessions fail unless the lease is expired.
- Completion by non-owners is rejected while another owner’s lease is still valid.
- Use explicit `dependencies` to serialize truly dependent work only.

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

Output shape:

```json
{
  "compressed_text": "...candidate condensed view...",
  "original_text": "...full agent output...",
  "confidence": 0.83,
  "coverage": 0.58,
  "fallback_used": false,
  "fallback_reason": ""
}
```

Callers decide which field to forward. The helper does **not** change any
default fleet-dispatch path on its own.
>
> Scores are upserted to the Qdrant `eval_scores` collection with `run_date` in
> the payload.

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
