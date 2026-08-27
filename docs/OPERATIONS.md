# Operations Guide

## Environment variables

Nothing is hardcoded. Settings resolve through a chain (`mcp/backends.py`):

1. the environment variable
2. a local probe — `http://localhost:11434` for Ollama, `:8000` for vLLM
3. `~/.loci/backends.toml`, or `$LOCI_CONFIG` — gitignored, machine-specific
4. the code default

`scripts/loci_groom.py:load_env()` inserts the repo `.env` and `mcp/.env` between
steps 1 and 3, because a cron job does not inherit the MCP launcher's environment.
Copy `backends.toml.example` to `~/.loci/backends.toml` for endpoints and keys
that must not live in the repo.

### Core infrastructure

| Variable | Default | Used by |
|---|---|---|
| `QDRANT_URL` | _(none — required; unset means Qdrant steps are skipped or the script errors out, never a localhost fallback)_ | all Qdrant-touching scripts |
| `QDRANT_API_KEY` | _(none)_ | all Qdrant-touching scripts |
| `OLLAMA_URL` | _(none, required)_ | most embedding + generation scripts (memgas_hierarchy.py, ebbinghaus_consolidation.py, amem_consolidation.py, agentHER_relabeler.py, skillops_maintenance.py, exif_skill_discovery.py, score_trace_collector.py, eval/harness.py) |
| `OLLAMA_BASE_URL` | _(none in code; `backends.ollama_url()` probes `http://localhost:11434`)_ | the **embedding** endpoint for 16 non-test files: `mcp/{qdrant_ops,embed_ops,backends,memcheck/llm}.py`, `scripts/hooks/{pre_llm_grounding,session_end_sync}.py`, `scripts/{loci_groom,glymphatic_sweep,gpu_warm}.py`, all of `mlops/`, `deep_think_loci/grounding/` |
| `LOCI_OLLAMA_GEN_URL` / `OLLAMA_GEN_URL` | _(none; falls back to `backends.ollama_gen_url()` → `ollama_url()`)_ | the **generation** endpoint, resolved separately from embeddings (`mcp/llm_local.py`). `OLLAMA_BASE_URL` deliberately does **not** feed it |
| `LOCI_VLLM_FALLBACK` | `0` (off) | opt-in fallback from Ollama generation to a batched vLLM endpoint (`mcp/llm_local.py`, `mcp/batched_gen.py`) |
| `LOCI_QDRANT_RETENTION_DAYS` | `0` — purge **disabled** | `mcp/qdrant_ops.py:_retention_days`. Any value > 0 makes the first Qdrant call of every process delete findings older than that window. Also readable as `[qdrant].retention_days` in `~/.loci/backends.toml` |
| `MNEMOSYNE_EMBEDDING_MODEL` | `nomic-embed-text` | all embedding operations |

Note: `OLLAMA_URL` and `OLLAMA_BASE_URL` are distinct variables read by different
files, and the split is not hooks-vs-standalone. The MCP server, the grooming tier
and both Claude Code hooks read `OLLAMA_BASE_URL` (a bare host) and append `/v1`
themselves. The older standalone scripts read `OLLAMA_URL`: `memgas_hierarchy.py`,
`ebbinghaus_consolidation.py`, `amem_consolidation.py`, `agentHER_relabeler.py`,
`skillops_maintenance.py`, `exif_skill_discovery.py`, `score_trace_collector.py`,
`swr_replay.py`, `ua-ingest.py`, `gpu_warm.py`, and `eval/harness.py`.
`mcp/embed_ops.py` and `mcp/backends.py` read both.

### Memory store paths

| Variable | Default | Used by |
|---|---|---|
| `MNEMOSYNE_DB` | `~/.hermes/mnemosyne/data/mnemosyne.db` | ebbinghaus, amem, agentHER, memgas, score_trace |
| `STATE_DIR` | `~/.claude/hook-state` | all hooks, skill_annotation_updater, score_trace, exif |
| `SKILLS_DIR` | `~/.claude/skills` | skill_annotation_updater, skillops_maintenance, exif |
| `LOCI_STATE_DB` | `~/.hermes/state.db` | state_db_qdrant_sync |

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

## Cron jobs

`cron/jobs.json` defines six jobs; the five enabled ones are below.
`deep-think-loci-harvest` (`dtl_harvest.sh`, every 7d, `no_agent`) ships disabled
and is omitted from the table.

These jobs need an agent runner that reads `cron/jobs.json`. Nothing in this repo
does, and on the reference host every one of the five has `last_run_at: null`
(issue #205) — check that before assuming they fire. The grooming tier below runs
from the user crontab instead, and does fire.

| ID | Name | Interval | Script |
|---|---|---|---|
| `5872853d8b28` | mnemosyne-consolidation | 20m | `mnemosyne_activity_check.py` |
| `65355f0c518f` | mnemosyne-session-summarizer | 20m | `mnemosyne_activity_check.py` |
| `b40ae8101c2a` | mnemosyne-sleep-cli | 30m | `mnemosyne_sleep_all.sh` |
| `c857cd706f67` | mnemosyne-qdrant-sync | 30m | `mnemosyne_qdrant_sync.py` |
| `a9fc1ea0886a` | state-db-qdrant-sync | 5m | `state_db_qdrant_sync.py` |

**mnemosyne-consolidation** and **mnemosyne-session-summarizer** both use `mnemosyne_activity_check.py`
as the pre-flight gate script. They differ in their agent prompt: consolidation runs a lightweight
`mnemosyne_sleep` pass while the session-summarizer archives structured session facts, triples,
and scratchpad state.

**mnemosyne-sleep-cli** (`no_agent: true`) invokes the Mnemosyne CLI directly via shell and
runs consolidation across all configured banks without spawning an LLM agent.

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
"refuted" on a true finding is what a later reader acts on.

The wrapper's exit code is the point of it:

| Code | Meaning |
|---|---|
| 0 | ok |
| 1 | a pass raised |
| 3 | a pass **refused** or **degraded** |

The refusal that matters: `connect()` reads `_retention_days()` before touching
Qdrant and refuses when it is not 0, because `_get_qdrant()` runs the startup
purge on its first call in a process. Grooming re-indexes findings; against a
non-zero retention window that is an index-then-delete loop that reports success.
The wrapper prints the pass's own reason on stderr and appends one line per run
to `$LOCI_GROOM_STATE/runs.jsonl` (default `~/.loci/groom/runs.jsonl`), so
"did this ever actually run" has an answer.

Per-run ceilings: `LOCI_GROOM_VERIFY_INVESTIGATIONS` (5),
`LOCI_GROOM_VERIFY_FINDINGS` (10), `LOCI_GROOM_SUMMARY_INVESTIGATIONS` (12),
`LOCI_GROOM_REFLECT_ITEMS` (3), `LOCI_GROOM_BATCH` (16). `LOCI_GROOM_MODEL` is
unset on purpose — the vLLM and Ollama tiers name the same model differently, so
each tier resolves its own.

---

## Manual operations

The repo is `loci` (github.com/rjmendez/loci). The commands below assume:

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

Runs three scorers in sequence — `harness.py`, `grounding_gate_eval.py`,
`grounding_gate_qf_eval.py` — and passes its arguments through to each. Scores are
upserted to the Qdrant `eval_scores` collection with `run_date` in the payload.
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

This repo ships three hooks, in `scripts/hooks/`. Each reads a JSON payload on
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
silently before — a hand-edited `pre_tool_grounding.py` accepting `PreToolUse`
against a repo copy that only accepted the Hermes name, where a fresh install
would have disabled the hook.

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
`LOCI_QDRANT_RETENTION_DAYS`. The code default is 0 and 0 is safe, so a fresh
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
| `cron/jobs.json` is not read by anything in this repo | MED | The five enabled jobs there have never run on the reference host (#205). Schedule from the user crontab, as the grooming tier does |
| `backends.toml.example` has no `[qdrant] retention_days` key | LOW | The key is read (`qdrant_ops._retention_days`) but not shown in the example; the code default of 0 applies |
| SCoRe `corrections=0` until sessions accumulate overlap | INFO | Corrections require same-session failure→success pairs; grow naturally |
