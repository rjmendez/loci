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

### Braincluster trainlog privacy scrub (contract + usage)

`scripts/braincluster_trainlog_privacy_scrub.py` is the required fail-closed
redaction boundary before trainlog payloads are written or exported.

Scrub rules:

1. Field-name redaction wrappers: any key matching sensitive names (`token`,
   `session_id`, `authorization`, `secret`, etc.) is replaced with a wrapper
   object, not masked inline.
2. Wrapper metadata is contractual: `__scrubbed__`, `redaction_kind`,
   `field_name`, `path`, `fingerprint`, and `length` (for sized values) must be
   present so downstream checks can audit redaction integrity deterministically.
3. Inline string redaction is tokenized when the field name is not itself
   sensitive: emails -> `[EMAIL_REDACTED]`, secret URLs -> `[URL_REDACTED]`,
   and credential-like key/value fragments -> `[TOKEN_REDACTED]`. Sensitive
   field-name wrappers take precedence over inline masking.
4. Blob-like payloads (`raw_payload`, large multiline bodies, bytes) are wrapped
   as `redaction_kind=raw_blob` with fingerprint metadata.
5. Unsupported shapes fail closed: non-object roots, unsupported value types
   (for example `set`), parse errors, or scrubber import failures return exit
   `2` with a machine-readable error payload; no unsafe partial output is
   emitted.

Output metadata contract:

- `privacy.schema_version=braincluster-trainlog-privacy-scrub/v1`
- `privacy.scrubbed=true`
- `privacy.redaction_count` equals `len(privacy.redactions)`
- `privacy.redaction_kinds` is a deterministic sorted set of observed kinds
- `privacy.redactions[*]` rows include `path`, `field_name`, `redaction_kind`,
  and `fingerprint`

Usage:

```powershell
python scripts\braincluster_trainlog_privacy_scrub.py `
  --input artifacts\trainlog-raw.json `
  --output artifacts\trainlog-scrubbed.json
```

Or stdin/stdout mode for pipeline composition:

```powershell
Get-Content artifacts\trainlog-raw.json -Raw | `
python scripts\braincluster_trainlog_privacy_scrub.py > artifacts\trainlog-scrubbed.json
```

`braincluster_trainlog_schema_normalizer.py` invokes this scrubber before
emitting canonical rows and fails closed if `privacy.scrubbed` is not true.

### Braincluster feedback learning loop (bounded retraining gate)

`scripts/braincluster_feedback_learning_loop.py` builds retraining samples from
canonical trainlog rows while fail-closing on contamination risk.

Deterministic inclusion defaults:

- `source_name=qdrant_findings`
- `confidence_tier >= high`
- resolution in `fixed|intentional|wontfix|superseded`
- `evidence_provenance_tier` in
  `human_authored|tool_verified|deterministic_derived`
- `provenance_refs` must contain both `investigation_id:*` and `text_hash:*`

Deterministic exclusions:

- uncertain labels (`confidence_tier` below threshold, missing/unknown primary label)
- unresolved outcomes (`open`/unknown/missing resolution unless explicitly allowed)
- retracted/tombstoned outcomes (`retract*` markers)
- weak provenance tier/traceability (`model_asserted`, missing refs)

Bounded-loop controls:

- `--max-samples-total`: hard cap over admitted rows
- `--max-samples-per-label`: per-label cap
- stable ordering `(event_ts, row_id)` before bounds are applied

Usage:

```powershell
python scripts\braincluster_feedback_learning_loop.py `
  --canonical-rows-json artifacts\braincluster-trainlog-canonical-rows.json `
  --max-samples-total 2000 `
  --max-samples-per-label 300 `
  --output artifacts\braincluster-feedback-training-samples.json
```

The output contains `samples[*]` in `TrainingSample`-compatible shape for
downstream brain-cluster dry-run/training stages. If contamination guards reject
all rows, the tool exits `2` and emits an explicit JSON error payload.

### Brain-cluster operator checklists (preflight / deploy / rollback / incident)

Use this sequence when promoting brain-cluster artifacts. Do not bypass any gate.

#### 1) Preflight checklist

- [ ] Run deterministic holdout campaign:

  ```powershell
  python scripts\braincluster_holdout_campaign.py `
    --output-root F:\.flybrain\cache\braincluster-holdout `
    --base-storage-root F:\.flybrain `
    --holdout-storage-root F:\.flybrain-holdout `
    --split-seed holdout-seed-20260923
  ```

- [ ] Confirm `holdout-campaign-report.json` contract:
  - `schema_version=braincluster-holdout-campaign/v1`
  - `status=pass`
  - `pass=true`
- [ ] Confirm `holdout-artifact-checklist.json` contract:
  - `schema_version=braincluster-holdout-artifact-checklist/v1`
  - `status=ok`
  - `missing_count=0`
- [ ] Confirm `release-prep-report.json` gate output:
  - `schema_version=braincluster-release-prep/v1`
  - `status=ok`

#### 2) Deploy checklist

- [ ] Generate readiness decision from holdout + release-prep outputs:

  ```powershell
  python mcp\flybrain_brain_cluster_promotion_readiness.py `
    --holdout-report F:\.flybrain\cache\braincluster-holdout\holdout-campaign-report.json `
    --release-prep-report F:\.flybrain\cache\braincluster-holdout\release-prep-report.json `
    --output F:\.flybrain\cache\braincluster-holdout\promotion-readiness.json
  ```

- [ ] Confirm readiness contract:
  - `schema_version=braincluster-promotion-readiness/v1`
  - `status=ready`
  - `pass=true`
  - `decision=go`
  - `summary.failed_checks=0`
- [ ] Run canary promotion drill before production pointer changes:

  ```powershell
  python scripts\braincluster_canary_promotion_drill.py `
    --state-path F:\.flybrain\cache\braincluster-holdout\promotion-state.json `
    --candidate-manifest F:\.flybrain\cache\braincluster-holdout\candidate\artifact-bundle\manifest.json `
    --readiness-report F:\.flybrain\cache\braincluster-holdout\promotion-readiness.json `
    --report F:\.flybrain\cache\braincluster-holdout\canary-promotion-drill.json
  ```

- [ ] Confirm drill contract:
  - `schema_version=braincluster-canary-promotion-drill/v1`
  - `status=pass`
  - `pass=true`
  - `rollback_attempted=true`
  - `rollback_succeeded=true`

#### 3) Rollback checklist

Trigger rollback when any deploy gate fails, when post-promotion audit fails, or when runtime behavior regresses versus the approved report.

- [ ] Capture evidence first (no mutations):
  - `promotion-readiness.json`
  - `canary-promotion-drill.json`
  - current `promotion-state.json`
  - candidate `artifact-bundle\manifest.json`
- [ ] Run promotion audit and require pass before/after rollback:

  ```powershell
  python -c "import json,sys; from pathlib import Path; sys.path.insert(0, str((Path.cwd() / 'mcp').resolve())); import flybrain_brain_cluster as fbc; print(json.dumps(fbc.check_brain_cluster_promotion_audit_trail(r'F:\.flybrain\cache\braincluster-holdout\promotion-state.json'), indent=2, sort_keys=True))"
  ```

- [ ] If active pointer is unsafe, execute rollback with the same state file:

  ```powershell
  python -c "import sys; from pathlib import Path; sys.path.insert(0, str((Path.cwd() / 'mcp').resolve())); import flybrain_brain_cluster as fbc; fbc.rollback_brain_cluster_promoted(r'F:\.flybrain\cache\braincluster-holdout\promotion-state.json')"
  ```

- [ ] Re-run audit; require `schema_version=braincluster-promotion-audit/v1`, `pass=true`, `failure_count=0`.

#### 4) Incident response checklist

- [ ] Stop new promotions immediately (`decision=no-go` until closed).
- [ ] Preserve immutable artifacts for triage:
  - `holdout-campaign-report.json`
  - `holdout-artifact-checklist.json`
  - `release-prep-report.json`
  - `promotion-readiness.json`
  - `canary-promotion-drill.json`
  - `promotion-state.json`
- [ ] Classify incident by contract break:
  - artifact contract mismatch (`braincluster-artifact-manifest/v1` / missing files),
  - gate breach (`gate_report.pass=false` or `shadow_report.pass=false`),
  - pointer/audit breach (`braincluster-promotion-audit/v1 pass=false`).
- [ ] Keep last known-good promoted pointer active until audit passes and a fresh readiness report returns `decision=go`.
- [ ] Log closure with root cause, corrected artifact version, and exact report paths used for re-approval.

### Brain-cluster alert response playbooks (queue / drift / thresholds / promotion)

Use these when alerts fire; execute top-to-bottom and do not skip evidence capture.

#### A) Queue alert playbook (`T1-QUEUE-FLOOD`, blocked-work surge)

**Signals (source of truth)**
- `scripts\self_model_trigger_eval.py` output:
  - `_self-model\alerts.jsonl`: `trigger_type`, `tier`, `status`, `reason`, `value`, `cooldown_until`
  - `_self-model\introspection_report.json`: `blockers[*].type=queue_overflow`, `metrics.reflection_queue_backlog`, `metrics.blocked_todos`
- Queue ownership state from `investigation_queue_status`: per-item `state`, `owner_session`, `lease_expires_at`, `dependencies`.

**Triage**
1. Snapshot current trigger payload + introspection report (copy files before any queue mutation).
2. Identify pressure type:
   - reflection backlog (`reflection_queue_size >= LOCI_REFLECTION_QUEUE_THRESHOLD`), or
   - coordination lock contention (high `claimed`/`blocked`, stale leases).
3. List blocked/claimed queue items and sort by oldest `lease_expires_at`.

**Mitigation**
- Drain stale claims first: complete or release items that exceeded lease or are owner-abandoned.
- Enforce bounded intake: pause new enqueues until backlog is below threshold.
- For persistent blocked work, mark explicit `state=blocked` with notes and assign next owner before resuming intake.

**Rollback / escalation**
- Escalate when `T1-QUEUE-FLOOD` repeats after one drain cycle or blocked count keeps rising.
- Roll back any scheduler/config change that widened queue intake (for example threshold/cooldown edits) and re-run `self_model_trigger_eval.py`.

**Evidence to retain**
- `_self-model\alerts.jsonl` slice covering the incident window.
- `_self-model\introspection_report.json`.
- Queue snapshot before/after mitigation (items with `state`, `owner_session`, `lease_expires_at`).

#### B) Drift alert playbook (`routing_drift_*`, `routing_drift_alerts`)

**Signals (source of truth)**
- `braincluster-p0-dry-run/v1` report fields:
  - `routing_drift_alerts[*].code|severity|signal|threshold|observed|delta_from_threshold`
  - `shadow_report.pass`, `shadow_report.status`, `shadow_report.alerts`
- Shadow metrics in `braincluster-shadow-replay-report/v1`:
  - `decision_match_rate`
  - `confidence_drift.mean_abs_delta`
  - `routing_entropy.mean_abs_delta`
  - `expert_collapse.candidate.concentration`
  - `expert_collapse.concentration_delta`

**Triage**
1. Run/inspect latest holdout outputs (`scripts\braincluster_holdout_campaign.py`) and locate failing objective/slice rows.
2. Classify drift:
   - contract fail (`pass=false`) vs warning-only (`alerts` present but pass),
   - concentration collapse vs confidence/entropy drift.
3. Check `context.objective_task_type_deltas` on alert rows to localize drifted task types.

**Mitigation**
- Recalibrate thresholds from current holdout reports (`mcp\flybrain_brain_cluster_thresholds.py`) when drift is legitimate but bounded.
- If collapse/concentration critical alerts fire, stop promotion and retrain candidate artifacts from balanced samples before rerun.
- If `routing_drift_not_evaluable`, treat as critical data-quality fault: rebuild fixtures and rerun shadow replay before any promotion decision.

**Rollback / escalation**
- Escalate immediately on critical drift codes:
  - `routing_drift_decision_match_rate`
  - `routing_drift_expert_concentration`
  - `routing_drift_not_evaluable`
- Keep current promoted pointer; do not stage or promote candidate until shadow replay returns `pass=true` with no critical alerts.

**Evidence to retain**
- `holdout-campaign-report.json` (`runs[*]`, `errors`).
- Failing p0 report(s) under `runs\<objective>\<slice>\p0-report.json`.
- Threshold bundle(s) used for the run (`thresholds\braincluster-thresholds-*.json`).

#### C) Threshold alert playbook (stale/invalid threshold bundles)

**Signals (source of truth)**
- Validation failures raised by `mcp\flybrain_brain_cluster_pipeline.py`:
  - stale bundle: `generated_at ... older than 7 days`
  - schema/objective/fingerprint mismatch
  - missing required gate/shadow threshold fields
- Readiness degradation where checks fail against threshold refs in `promotion-readiness.json`.

**Triage**
1. Inspect threshold metadata:
   - `schema_version=braincluster-threshold-calibration/v1`
   - `objective`, `generated_at`, `calibration_id`, `input_fingerprint`.
2. Verify fingerprint continuity:
   - `calibration_id == braincluster-thresholds-<input_fingerprint[:16]>`.
3. Confirm release-prep points to the expected `threshold_file` per objective.

**Mitigation**
- Regenerate thresholds from latest holdout runs (`mcp\flybrain_brain_cluster_thresholds.py`) and update release-prep outputs.
- Re-run readiness evaluation (`mcp\flybrain_brain_cluster_promotion_readiness.py`) after replacing stale/invalid bundles.
- Reject ad hoc manual threshold edits; only accept generated bundles with matching fingerprints.

**Rollback / escalation**
- Escalate when thresholds cannot be regenerated from valid holdout artifacts (missing/corrupt reports).
- Roll back to last known-good threshold bundle set and hold `decision=no-go` until fresh calibration succeeds.

**Evidence to retain**
- Old and replacement threshold bundles.
- `release-prep-report.json` (`objectives_calibrated[*].threshold_file`).
- `promotion-readiness.json` before/after recalibration.

#### D) Promotion alert playbook (readiness/canary/audit failures)

**Signals (source of truth)**
- `promotion-readiness.json` (`status`, `pass`, `decision`, `summary.failed_checks`, `reasons`).
- `canary-promotion-drill.json` (`status`, `pass`, `rollback_attempted`, `rollback_succeeded`, `reasons`).
- `braincluster-promotion-audit/v1` from `check_brain_cluster_promotion_audit_trail(...)`:
  - `pass`, `failure_count`, `failures[*].code`.

**Triage**
1. Stop promotions and snapshot:
   - `promotion-state.json`
   - candidate `artifact-bundle\manifest.json`
   - readiness + canary reports.
2. Run promotion audit; classify failure family:
   - pointer continuity (`BROKEN_PROMOTED_POINTER_CONTINUITY`, `STATE_*_MISMATCH`)
   - manifest integrity (`TARGET_MANIFEST_UNREADABLE`, `MANIFEST_IDENTITY_MISMATCH`)
   - trail integrity (`MISSING_AUDIT_TRAIL`, `IMMUTABLE_RECORD_MISMATCH`).
3. Confirm whether the active promoted pointer is still known-good.

**Mitigation**
- If pointer unsafe, execute `rollback_brain_cluster_promoted(...)`, then re-run audit and require `pass=true`, `failure_count=0`.
- Restage candidate manifest and re-run canary drill only after readiness returns `decision=go`.
- If audit failure is manifest-related, rebuild artifact bundle and restamp manifest identity before restaging.

**Rollback / escalation**
- Hard escalation when rollback cannot restore audited-good state.
- Freeze deploy lane until:
  - readiness `status=ready`, `pass=true`, `decision=go`
  - canary drill `pass=true`, `rollback_succeeded=true`
  - promotion audit `pass=true`.

**Evidence to retain**
- `promotion-readiness.json`, `canary-promotion-drill.json`, `promotion-state.json`.
- Audit JSON before/after rollback.
- Candidate + promoted manifest identity fields (`manifest_path`, `manifest_sha256`, `artifact_id`, `artifact_version`).

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
