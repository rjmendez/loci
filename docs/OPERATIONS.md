# Operations Guide

## Environment variables

All scripts use `os.environ.get(VAR, default)` — no value is hardcoded.
Override any default by exporting the variable before running, or by editing
`~/.claude/hooks/env.sh` for persistent overrides.

### Core infrastructure

| Variable | Default | Used by |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | all Qdrant-touching scripts |
| `QDRANT_API_KEY` | _(none)_ | all Qdrant-touching scripts |
| `OLLAMA_URL` | _(none, required)_ | most embedding + generation scripts (memgas_hierarchy.py, ebbinghaus_consolidation.py, amem_consolidation.py, agentHER_relabeler.py, skillops_maintenance.py, exif_skill_discovery.py, score_trace_collector.py, eval/harness.py) |
| `OLLAMA_BASE_URL` | _(none)_ | hook scripts only: hooks/pre_llm_grounding.py, hooks/session_end_sync.py, swr_replay.py |
| `MNEMOSYNE_EMBEDDING_MODEL` | `nomic-embed-text` | all embedding operations |

Note: `OLLAMA_URL` and `OLLAMA_BASE_URL` are distinct variables used by different script groups.
Most standalone scripts read `OLLAMA_URL` (full base URL, e.g. `http://localhost:11434`).
Hook scripts read `OLLAMA_BASE_URL` and append `/v1` internally.

### Memory store paths

| Variable | Default | Used by |
|---|---|---|
| `MNEMOSYNE_DB` | `~/.hermes/mnemosyne/data/mnemosyne.db` | ebbinghaus, amem, agentHER, memgas, score_trace |
| `STATE_DIR` | `~/.claude/hook-state` | all hooks, skill_annotation_updater, score_trace, exif |
| `SKILLS_DIR` | `~/.claude/skills` | skill_annotation_updater, skillops_maintenance, exif |
| `HERMES_STATE_DB` | `~/.hermes/state.db` | state_db_qdrant_sync |

### Tuning parameters

| Variable | Default | Effect |
|---|---|---|
| `HOOK_RECALL_TOP_K` | `3` (code default; `.env.example` sets `5`) | Max grounding results injected per turn |
| `HOOK_RECALL_MIN_SCORE` | `0.55` | Qdrant cosine threshold; lower = more noise |
| `FORGET_THRESH` | `0.3` | Entries with retention probability below this are re-embedded (0 = never, 1 = always); used by ebbinghaus_consolidation.py |
| `MAX_PER_RUN` | `50` (ebbinghaus), `20` (agentHER) | Max entries processed per cron tick; read by both ebbinghaus_consolidation.py and agentHER_relabeler.py |
| `AMEM_LINK_THRESHOLD` | `0.88` | Cosine threshold for cross-link creation |
| `AMEM_CONFLICT_THRESHOLD` | `0.96` | Cosine threshold for conflict flagging |
| `SHADOW_THRESHOLD` | `0.92` | Cosine threshold for SHADOW_RISK pairs; used by skillops_maintenance.py |
| `AGENTHER_GEN_MODEL` | `llama3.2:latest` | Ollama model for failure relabeling |
| `EXIF_GEN_MODEL` | `llama3.2:latest` | Ollama model for skill gap analysis |
| `TOP_K_PER_LEVEL` | `3` | Results per level in MemGAS search; used by memgas_hierarchy.py |

---

## Cron jobs

Live cron config: `cron/jobs.json` (in repo root; deployed path depends on your cron runner setup).

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

## Manual operations

### Rebuild MemGAS 3-level index

```bash
HERMES_PY=~/.hermes/hermes-agent/venv/bin/python3
$HERMES_PY ~/development/hermes_memory/scripts/memgas_hierarchy.py --index
```

Run after major Mnemosyne consolidation, or when memgas_l1/l2/l3 collections get stale.

### Run MemGAS search

```bash
$HERMES_PY ~/development/hermes_memory/scripts/memgas_hierarchy.py --search "your query here"
```

### Detect skill shadows

```bash
OLLAMA_URL=http://localhost:11434 \
$HERMES_PY ~/development/hermes_memory/scripts/skillops_maintenance.py
```

Review SHADOW_RISK pairs. For sim=1.000 pairs: one is usually a duplicate install or
has an empty description — populate a distinctive description.

### Discover skill gaps (EXIF)

```bash
STATE_DIR=~/.claude/hook-state \
$HERMES_PY ~/development/hermes_memory/scripts/exif_skill_discovery.py
```

Review `~/.claude/hook-state/exif_discoveries.jsonl` for candidates. Promote manually:
```bash
cp -r ~/.claude/hook-state/candidate_skills/SKILLNAME ~/.claude/skills/SKILLNAME/
```

### Build SCoRe fine-tuning dataset

```bash
$HERMES_PY ~/development/hermes_memory/scripts/score_trace_collector.py
cat ~/.hermes/mnemosyne/data/score_traces/manifest.json
```

When `ready_for_sft: true` (≥ 10 correction pairs), the dataset is usable for SFT.

### Run eval harness

```bash
~/development/hermes_memory/eval/run_eval.sh
```

Scores are upserted to Qdrant `eval_scores` collection with run_date in payload.
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
$HERMES_PY ~/development/hermes_memory/scripts/mnemosyne_qdrant_sync.py
```

---

## Claude Code hook registry

Hooks registered in `~/.claude/settings.json`:

| Event | Matcher | Script | Purpose |
|---|---|---|---|
| SessionStart | * | `session_start_guard.sh` | Project detect, MCP probe, rules inject |
| UserPromptSubmit | * | `user_prompt_grounding.sh` | Per-turn Qdrant grounding |
| PreToolUse | Bash | `pre_bash_guard.sh` | Guard protocol (ACK_REMOTE_WRITE etc.) |
| PreToolUse | * | `hermes_pre_tool_grounding.sh` | Hermes tool audit |
| PostToolUse | Bash | `post_bash_success_memory.sh` | Success event logging |
| PostToolUseFailure | Bash | `post_bash_failure_memory.sh` | Repeated failure → Mnemosyne |
| PostToolUseFailure | * | `post_tool_failure_reflection.sh` | Reflexion trace for all tools |
| PreCompact | — | `pre_compact_guard.sh` | Checkpoint before compaction |
| Stop | — | `session_end_evaluate_guard.sh` | AgentRR trace → hermes_sessions |
| SessionEnd | * | `session_end_evaluate_guard.sh` | Same as Stop |

---

## Portability (new machine setup)

All hook paths and infra addresses are driven by env vars. To stand up on a new machine:

1. Copy `~/.claude/hooks/` directory
2. Set overrides in `~/.claude/hooks/env.sh`:
   ```bash
   export OLLAMA_URL=<ollama-base-url>
   export OLLAMA_BASE_URL=<ollama-base-url>
   export MNEMOSYNE_AUTHOR_ID=<your-id>
   export HERMES_PY=/path/to/python3
   # etc.
   ```
3. Update `~/.claude/settings.json` hook paths (currently hardcoded — no env expansion in JSON)
4. Create `~/.claude/hook-state/` directory
5. Ensure Qdrant API key is in `~/.claude/settings.json` at
   `mcpServers.hermes_memory.env.QDRANT_API_KEY`

---

## Known issues and limitations

| Issue | Severity | Workaround |
|---|---|---|
| `settings.json` hook paths are hardcoded (no `$HOME` expansion in JSON) | LOW | Manual edit on new machine |
| Ebbinghaus timestamp format warnings for microsecond ISO strings | LOW | Non-fatal; entries fall back to 30-day default decay |
| `eval/harness.py` mean_score=0.167 baseline is low | INFO | Keyword matching is strict; trend matters more than absolute value |
| MemGAS index takes ~5min for 500+ entries (sequential embed) | MED | Run `--index` in off-hours; add batch embedding |
| agentHER requires a generative Ollama model to be available | MED | Set `AGENTHER_GEN_MODEL` to your installed model (default: `llama3.2:latest`) |
| SCoRe `corrections=0` until sessions accumulate overlap | INFO | Corrections require same-session failure→success pairs; grow naturally |
