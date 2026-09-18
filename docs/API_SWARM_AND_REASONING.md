# Swarm and Reasoning API Reference

This reference covers the local reasoning MCP tools plus the two related CLI entry points:

- MCP tools: `swarm_reason`, `llm_local`, `reflection_loop_seed`, `reflection_loop_status`, `reflection_loop_tick`, `memory_confidence`
- CLI scripts: `scripts/swarm_escalate.py`, `scripts/local_deep_think.py`

The two most-misunderstood ideas are:

1. A **seed** is a full independent swarm pipeline run, not sampling noise on one prompt.
2. `local_deep_think.py` and `swarm_escalate.py` use different kinds of parallelism.

---

## 1) MCP tools

## `swarm_reason`

**Defined in:** `mcp/llm_tools.py`

**Signature**

```python
swarm_reason(
    topic: str,
    fanout_count: int = 20,
    seeds: int = 1,
    cheap_model: str = "",
    escalate_model: str = "",
    synthesize_model: str = "",
    decompose_model: str = "",
    subtasks: Optional[list] = None,
    escalate_confidences: Optional[list] = None,
    synthesize_think: Optional[bool] = None,
    safety_check: Optional[bool] = None,
    self_consistency_samples: int = 1,
    escalate_with_prior_context: bool = False,
    reduce_group_size: int = 0,
    stigmergic_consensus: bool = False,
    stigmergic_ttl_minutes: float = 60.0,
) -> str
```

**Purpose**

Runs the local swarm reasoner and returns one structured JSON result. The wrapper loads
`scripts/swarm_escalate.py`, constructs `SwarmConfig`, runs the pipeline, validates the
result, and fails open to degraded JSON on wrapper/import/runtime errors.

**Important behavior**

- `seeds > 1` means multiple independent full pipeline runs before one merged synthesis.
- Empty `escalate_model` / `synthesize_model` means “use the resolved default for the
  current tier mode,” not “explicit override to empty”.
- The MCP wrapper sets `auto_parallel=False`, so CLI auto-seed expansion does **not**
  happen implicitly through the tool.

**Parameters**

| Param | Meaning |
|---|---|
| `topic` | Top-level question/task. |
| `fanout_count` | Target number of atomic subtasks from decomposition. |
| `seeds` | Number of independent swarm seeds to run. Default `1`. |
| `cheap_model` | Cheap fan-out model. Empty means use swarm default. |
| `escalate_model` | Escalation model override. Empty means use tier-resolved default. |
| `synthesize_model` | Synthesis model override. Empty means use tier-resolved default. |
| `decompose_model` | Optional dedicated decomposition model; otherwise cheap model is reused. |
| `subtasks` | Pre-supplied subtasks; skips model decomposition if present. |
| `escalate_confidences` | Confidence labels that force escalation; default is effectively `low`. |
| `synthesize_think` | Opt into one reasoning-enabled synthesis call. |
| `safety_check` | Add advisory Granite Guardian `safety_flag` to the final summary. |
| `self_consistency_samples` | Cheap-tier majority sampling count for low-confidence findings before escalation. |
| `escalate_with_prior_context` | Include the cheap-tier answer in escalation prompts for critique-and-improve. |
| `reduce_group_size` | Hierarchical reduce size before final synthesis; `0` keeps flat synthesis. |
| `stigmergic_consensus` | Enable deterministic consensus gating before escalation. |
| `stigmergic_ttl_minutes` | TTL for stigmergic trail entries when consensus is enabled. |

**Return shape**

Returns JSON text with this core shape:

```json
{
  "schema_version": 1,
  "topic": "...",
  "findings": [
    {
      "subtask": "...",
      "answer": "...",
      "confidence": "low|medium|high",
      "tier_reached": "cheap|escalated|synthesized",
      "model": "...",
      "ok": true,
      "escalation_attempted": true,
      "escalation_reasons": ["confidence:low"]
    }
  ],
  "summary": "...",
  "stats": {
    "fanout_count": 12,
    "escalated_count": 3,
    "escalation_rate": 0.25
  },
  "decomposition": {"...": "..."},
  "triage": {"...": "..."},
  "tiers": {
    "cheap": {"...": "..."},
    "escalate": {"...": "..."}
  },
  "lineage": [
    {
      "subtask": "...",
      "answer": "...",
      "confidence": "...",
      "tier_reached": "...",
      "model": "...",
      "ok": true,
      "parse_ok": true,
      "why": "",
      "escalation_attempted": false,
      "escalation_reasons": []
    }
  ],
  "multi_seed": {"...": "..."}
}
```

`multi_seed` appears only on multi-seed runs.

**Example**

```json
{
  "topic": "Explain why auth cache misses spike after deploys",
  "fanout_count": 10,
  "seeds": 3,
  "self_consistency_samples": 3,
  "escalate_with_prior_context": true,
  "stigmergic_consensus": true
}
```

---

## `llm_local`

**Defined in:** `mcp/llm_tools.py` (wrapper over `mcp/llm_local.py`)

**Signature**

```python
llm_local(
    prompt: str,
    model: str = "",
    fmt: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.2,
    keep_alive: str = "30m",
) -> str
```

**Purpose**

Single local-model generation call for cheap, high-volume work that should avoid cloud
tokens. If `model` is unset, it uses the configured generation model.

**Parameters**

| Param | Meaning |
|---|---|
| `prompt` | Prompt text. |
| `model` | Optional Ollama model tag. Empty means use configured generation model. |
| `fmt` | `"json"` requests and validates JSON output; otherwise free text. |
| `max_tokens` | Max generated tokens. |
| `temperature` | Sampling temperature. |
| `keep_alive` | Ollama residency window; default `30m`. |

**Return shape**

```json
{
  "text": "...",
  "ok": true,
  "model": "qwen2.5:3b"
}
```

Failures stay fail-open and usually include `why`:

```json
{
  "text": "",
  "ok": false,
  "model": "qwen2.5:3b",
  "why": "..."
}
```

**Example**

```json
{
  "prompt": "Return only JSON: {\"status\":\"ok\"}",
  "fmt": "json",
  "max_tokens": 80
}
```

---

## `reflection_loop_seed`

**Defined in:** `mcp/server.py`

**Signature**

```python
reflection_loop_seed(
    investigation_id: str = "copilot-self-reflection-loop",
    session_events_limit: int = 250,
    process_logs_limit: int = 120,
    reset_queue: bool = False,
) -> str
```

**Purpose**

Seeds the persistent self-reflection queue from local Copilot and Claude artifacts. This
tool only enqueues targets; it does not parse files or store findings.

Queue state lives at `LOCI_MEMORY_DIR/_reflection-loop/state.json`.

Artifact sources scanned:

- `~/.copilot/temp_ingest/payload.json`
- `~/.copilot/session-state/*/events.jsonl`
- `~/.copilot/logs/process-*.log`
- `~/.claude/projects/**/*.jsonl`

**Parameters**

| Param | Meaning |
|---|---|
| `investigation_id` | Investigation where later tick findings will be written. Default comes from `LOCI_REFLECTION_INVESTIGATION`, else `copilot-self-reflection-loop`. |
| `session_events_limit` | Max recent event files to consider, clamped to `1..2000`. |
| `process_logs_limit` | Max recent process logs to consider, clamped to `1..2000`. |
| `reset_queue` | If true, clears queue and processed map before reseeding. |

**Return shape**

```json
{
  "queued_added": 37,
  "queue_size": 37,
  "investigation_id": "copilot-self-reflection-loop",
  "sources": {
    "temp_ingest": 1,
    "session_events_candidates": 250,
    "process_logs_candidates": 120,
    "claude_code_events_candidates": 250
  },
  "state_file": "/.../_reflection-loop/state.json"
}
```

**Example**

```json
{
  "investigation_id": "copilot-self-reflection-loop",
  "session_events_limit": 100,
  "process_logs_limit": 50,
  "reset_queue": true
}
```

---

## `reflection_loop_status`

**Defined in:** `mcp/server.py`

**Signature**

```python
reflection_loop_status(queue_preview: int = 8) -> str
```

**Purpose**

Reports current queue state and aggregate stats for the persistent reflection loop.

**Parameters**

| Param | Meaning |
|---|---|
| `queue_preview` | Number of queued items to preview, clamped to `0..50`. |

**Return shape**

```json
{
  "investigation_id": "copilot-self-reflection-loop",
  "queue_size": 22,
  "processed_count": 145,
  "stats": {
    "files_processed": 145,
    "lines_scanned": 184230,
    "errors_seen": 98,
    "warnings_seen": 211
  },
  "last_tick": {
    "ts": "...",
    "processed_items": 3,
    "findings_written": 4,
    "remaining_queue": 22
  },
  "updated_at": "...",
  "queue_preview": [
    {"kind": "session_event", "path": "..."}
  ],
  "state_file": "/.../_reflection-loop/state.json"
}
```

**Example**

```json
{
  "queue_preview": 5
}
```

---

## `reflection_loop_tick`

**Defined in:** `mcp/server.py`

**Signature**

```python
reflection_loop_tick(
    max_items: int = 3,
    max_lines_per_file: int = 4000,
    store_item_findings: bool = True,
    enable_llm_triage: bool = False,
    max_llm_items: int = 3,
) -> str
```

**Purpose**

Processes a small bounded batch from the reflection queue and optionally stores findings
through `investigation_store`.

**What it does**

- pops up to `max_items` highest-priority queued artifacts
- parses them deterministically by default
- optionally does bounded local-model semantic triage for error/warning-heavy items
- stores per-item findings plus batch summaries
- requeues dropped/unprocessable items and records gap findings when storing is enabled

**Parameters**

| Param | Meaning |
|---|---|
| `max_items` | Batch size, clamped to `1..20`. |
| `max_lines_per_file` | Per-file parsing bound, clamped to `50..20000`. |
| `store_item_findings` | If false, acts like a preview pass and does not retire items into `processed`. |
| `enable_llm_triage` | Enables bounded advisory semantic triage on items with visible errors/warnings. |
| `max_llm_items` | Budget for advisory LLM triage calls, clamped to `0..20`. |

**Return shape**

Queue empty:

```json
{
  "processed_items": 0,
  "queue_size": 0,
  "message": "Queue is empty. Run reflection_loop_seed first.",
  "state_file": "/.../_reflection-loop/state.json"
}
```

Normal batch:

```json
{
  "investigation_id": "copilot-self-reflection-loop",
  "processed_items": 3,
  "findings_written": 4,
  "remaining_queue": 19,
  "batch": [
    {
      "status": "processed",
      "kind": "session_event",
      "path": "...",
      "errors": {},
      "warnings": {}
    }
  ],
  "stats": {
    "files_processed": 148,
    "lines_scanned": 187900,
    "errors_seen": 103,
    "warnings_seen": 214,
    "last_error_signatures": [{"signature": "...", "count": 3}],
    "last_warning_signatures": [{"signature": "...", "count": 7}]
  }
}
```

**Example**

```json
{
  "max_items": 5,
  "max_lines_per_file": 2000,
  "enable_llm_triage": true,
  "max_llm_items": 2
}
```

---

## `memory_confidence`

**Defined in:** `mcp/server.py`

**Signature**

```python
memory_confidence(
    query: str,
    top_k: int = 8,
) -> str
```

**Purpose**

Estimates how reliably `loci_memory` knows about a claim or topic. This is a metamemory
tool: not “what is true?”, but “how much should I trust this memory retrieval?”

It computes five cues and maps them to a calibrated confidence score:

- `fluency` — top-hit similarity
- `accessibility` — mean top-4 similarity
- `source_diversity` — distinct sources or investigations
- `corroboration` — repeated occurrence saturation
- `trust` — mean stored confidence tier

It may also add `llm_entailment_note`, an advisory local-model check asking whether the
top hit actually supports the exact claim. That note never changes the numeric formula
output.

**Parameters**

| Param | Meaning |
|---|---|
| `query` | Claim or topic to evaluate. |
| `top_k` | Number of retrieval results to inspect. |

**Return shape**

Typical result:

```json
{
  "confidence": 0.742,
  "basis": "recollection",
  "cues": {
    "fluency": 0.812,
    "accessibility": 0.731,
    "source_diversity": 3,
    "corroboration": 0.463,
    "trust": 0.775
  },
  "top_hit_preview": "...",
  "recommendation": "moderate — use with source citation",
  "llm_entailment_note": {
    "available": true,
    "verdict": "confirmed",
    "rationale": "...",
    "confidence": 0.84,
    "degraded": false,
    "error": ""
  }
}
```

No-trace and hard-stop results remain structured low-confidence payloads.

**Example**

```json
{
  "query": "The auth cache invalidation bug was fixed by moving key derivation to the write path",
  "top_k": 8
}
```

---

## 2) The seed concept in `scripts/swarm_escalate.py`

A **seed** is one independent, complete run of the full swarm pipeline:

1. decompose
2. cheap-tier answer
3. triage
4. consensus gate
5. self-consistency
6. escalate

It is **not** temperature jitter on one call.

In code:

- `_run_single_seed(...)` executes one full run.
- `decompose_subtasks(...)` adds a seed-specific diversity hint when `seed_count > 1`:
  `Independent seed: {seed_index + 1}/{seed_count}. Prefer a slightly different decomposition angle...`
- `_run_multi_seed(...)` runs all seeds concurrently with `ThreadPoolExecutor(max_workers=effective_seeds)`.
- After all seeds finish, `_merge_findings(...)` merges and deduplicates all seeds' `final_findings`.
- The merged findings are fed into one final synthesis pass.

So `--seeds 3` means **three complete decompose -> cheap -> triage -> consensus -> self-consistency -> escalate runs in parallel**, then one merged synthesis.

### What varies across seeds?

The intended variation is the **subtask breakdown / approach**. Each seed gets a distinct
`seed_index` and a prompt nudge toward a slightly different decomposition angle.

### Merge behavior

`_merge_findings(...)` deduplicates by normalized `(subtask, answer)` pairs and keeps the
strongest version, preferring:

1. `ok=True`
2. `parse_ok=True`
3. higher confidence
4. higher tier reached
5. longer answer

### Cost / benefit tradeoff

Benefits:

- broader subtask coverage
- cross-validation across independent decompositions
- reduced dependence on one bad decomposition
- better recall on multi-part questions

Costs:

- approximately **N x** decomposition + cheap-tier work
- approximately **N x** possible escalation work too
- one final synthesis still happens only once after merge

The right mental model is **ensemble robustness**, not “sample the same answer several
times.”

---

## 3) Auto-parallel default (PR #359)

The CLI script has explicit logic to auto-expand seed count when a batched vLLM backend is
available.

Relevant fields and functions:

- `SwarmConfig.seeds`
- `SwarmConfig.seeds_explicit`
- `SwarmConfig.auto_parallel`
- `_batched_backend_available()`
- `_effective_seed_count(config)`
- env vars `LOCI_SWARM_AUTO_PARALLEL` and `LOCI_SWARM_AUTO_VLLM_SEEDS`

### Actual defaults

`SwarmConfig.auto_parallel` defaults from:

```python
os.environ.get("LOCI_SWARM_AUTO_PARALLEL", "1")
```

So auto-parallel is **on by default**.

`_AUTO_VLLM_SEEDS` is:

```python
int(os.environ.get("LOCI_SWARM_AUTO_VLLM_SEEDS", "3"))
```

So the literal default is **`3` seeds**.

### Resolution logic

`_effective_seed_count(config)` does this:

1. Start with `requested = max(1, int(config.seeds or 1))`.
2. If any of the following is true, return `requested` unchanged:
   - `requested > 1`
   - `config.seeds_explicit`
   - `not config.auto_parallel`
3. Otherwise, if `_batched_backend_available()` is true, return `max(1, _AUTO_VLLM_SEEDS)`.
4. Else return `requested` (normally `1`).

`_batched_backend_available()` returns `bool(str(backends.vllm_url(probe_timeout=0.2) or "").strip())`.
That means it uses the resolved shared vLLM URL as the heuristic. On the shared path,
that includes the localhost probe; if env or config already names a URL, the function
trusts that configuration.

### Meaning

When `--seeds` is omitted:

- resolved batched vLLM backend present -> auto-resolve to `3` seeds by default
- no batched backend (for example plain Ollama) -> stay at `1`

Why: vLLM continuous batching makes concurrent short requests relatively cheap, so
multi-seed robustness is attractive there. On non-batched backends, multi-seed is a much
more expensive latency or cost choice, so the script keeps historic single-seed behavior.

### Important MCP vs CLI distinction

The MCP tool `swarm_reason(...)` explicitly constructs `SwarmConfig(..., auto_parallel=False)`.
So the auto-parallel behavior applies to **CLI `scripts/swarm_escalate.py` when `--seeds`
is omitted**, not to default MCP tool calls.

---

## 4) Tiered escalation model

The swarm is a bounded gate stack, not merely “cheap model then strong model”.

## Stage A: decompose

`decompose_subtasks(...)`

- one model call to produce `fanout_count` atomic subtasks
- uses `decompose_model` if set, else the cheap model
- if decomposition output is bad or unparseable, fails open to one fallback subtask equal to the original topic

**Cost:** one call per seed.

## Stage B: cheap tier

`answer_subtasks(...)`

- one cheap-model answer pass over every subtask
- emits `SwarmFinding` rows with `answer`, `confidence`, `tier_reached="cheap"`, `ok`, `parse_ok`, `why`

**Cost:** roughly `fanout_count` responses per seed.

## Stage C: triage

`triage_findings(...)`

This is the first escalation gate. It flags findings when any of these are true:

- generation failed (`not finding.ok`)
- parse failure (`not finding.parse_ok`)
- confidence is in `config.escalate_confidences` (default low confidence)
- two subtasks are lexically similar enough and their answers look:
  - duplicate (`same normalized answer`)
  - duplicate-like (`answer similarity >= answer_similarity_threshold`)
  - contradictory (`similar subtasks with opposite detected polarity`)

So escalation is triggered by **low confidence, failures, duplicates on overlapping
subtasks, or contradictions on overlapping subtasks**.

**Cost:** deterministic or local only.

## Stage D: consensus gate

`apply_consensus_gate(...)`

- runs only when `stigmergic_consensus=True`
- calls `stigmergic_consensus.apply_stigmergic_consensus(...)` with a TTL config
- returns updated triage plus a consensus report
- on errors, degrades and leaves triage unchanged

This is a deterministic pre-escalation gate, not a separate LLM tier.

**Cost:** deterministic or local only.

## Stage E: self-consistency

`self_consistency_findings(...)`

- runs only when `self_consistency_samples > 1`
- does **not** resample every subtask
- only revisits triage-flagged items whose reasons include `confidence:low`
- asks the cheap tier the same subtask `samples` times
- if a strict majority agrees on the same normalized answer (`> samples / 2`), the winner replaces the original finding
- if the winning finding no longer matches an escalation-trigger confidence, the confidence trigger can be removed
- if there is no strict majority, it adds `self_consistency_disagreement`

So self-consistency is a cheap-tier rescue attempt before paying escalation-tier cost.

**Cost:** extra cheap-tier calls only for low-confidence flagged findings.

## Stage F: escalate tier

`escalate_findings(...)`

- escalates every remaining `triage.flagged_indices` item
- prompt is either:
  - plain re-answer, or
  - critique-and-improve if `escalate_with_prior_context=True`
- replacement is fail-open:
  - replace if escalated result is `ok`, or
  - replace if escalated answer is non-empty and current answer is empty, or
  - otherwise keep the current answer and annotate the failure

**Cost:** one stronger-model call per still-flagged finding.

## Stage G: synthesis

`synthesize_swarm(...)`

- synthesizes final findings into one structured JSON answer
- optionally runs hierarchical reduce first when `reduce_group_size > 0`
- optionally runs one `think=True` synthesis call when `synthesize_think=True`, with automatic fallback to normal synthesis if empty or unparseable
- optionally adds advisory `safety_flag` via Granite Guardian

In multi-seed mode, synthesis happens **after** merge and dedup across seeds.

**Cost:**

- flat mode: one synthesis call
- reduce mode: one synthesis call per reduce group, then one final synthesis call
- `safety_check`: one extra guardian check on the summary

### Practical summary

- cheap tier handles most work
- triage decides what looks weak, conflicting, or overlapping
- consensus gate optionally filters via deterministic stigmergic consensus
- self-consistency retries only low-confidence cheap answers
- escalate tier spends stronger-model budget only on the residue
- synthesis turns the final set into one consumable answer

---

## 5) Per-role vLLM endpoint routing (PR #360)

`mcp/backends.py` exposes:

```python
vllm_url(role: str | None = None, probe_timeout: float = 1.0) -> str
vllm_model(role: str | None = None) -> str
```

Role-specific env var names are built as:

- `VLLM_BASE_URL_<ROLE>`
- `VLLM_MODEL_<ROLE>`

with uppercasing and `-` converted to `_`.

Example: role `tool-calling` maps to `VLLM_BASE_URL_TOOL_CALLING` and
`VLLM_MODEL_TOOL_CALLING`.

## `vllm_url(role=...)` resolution order

### When `role` is provided

1. `VLLM_BASE_URL_<ROLE>` env var
2. `[vllm.<role>].url` in `~/.loci/backends.toml` (or `LOCI_CONFIG`)
3. fall back to shared `vllm_url()` resolution

There is intentionally **no role-specific localhost probe**. Specialist routing must be
explicit, then it falls back to the shared resolver.

### Shared `vllm_url()` resolution

1. `VLLM_BASE_URL`
2. localhost probe of `http://localhost:8000`
3. `[vllm].url` from config
4. empty string

## `vllm_model(role=...)` resolution order

### When `role` is provided

1. `VLLM_MODEL_<ROLE>` env var
2. `[vllm.<role>].model` in config
3. fall back to shared `vllm_model()`

### Shared `vllm_model()` resolution

1. `VLLM_MODEL`
2. `[vllm].model`
3. default literal `Qwen2.5-3B-Instruct`

## Supported roles

The resolver accepts any role string, but the repository's canonical role sets are:

### Specialist roles from `scripts/model_catalog.py`

- `code`
- `math`
- `safety`
- `tool_calling`

### Swarm role names from `scripts/model_catalog.py`

- `cheap_fanout`
- `guardian`
- `escalation`
- `synthesis`

`mcp/batched_gen.py` already supports `endpoint_role=...`, and the tests explicitly cover
role-based resolution, for example `code`. Use the catalog role names when you want
stable, semantically named routing.

---

## 6) `local_deep_think.py` vs `swarm_escalate.py`

These scripts solve different problems.

## `scripts/local_deep_think.py`

Mental model: **one reasoning chain that seeks diversity from multiple different models**.

Core pattern:

1. retrieve and gate evidence
2. ideate across `--ideate-models`
3. persist ideas
4. verify claims
5. optional red-team
6. synthesize
7. optional bounded self-reflection

The key parallelism is in `ideate(...)`: if there are multiple ideation models, it runs
one ideation task per model concurrently via `ThreadPoolExecutor`.

So deep-think diversity comes from **different models producing ideas**.

## `scripts/swarm_escalate.py`

Mental model: **one swarm pipeline that seeks diversity from multiple independent full runs**.

Core pattern:

1. decompose into subtasks
2. cheap parallel answer stage
3. triage / consensus / self-consistency
4. escalate uncertain or conflicting slice
5. synthesize

Multi-seed diversity comes from **different decompositions or approaches across seeds**,
typically using the same configured model tiers.

## When to use which

Use **`local_deep_think.py`** when you want:

- multiple model perspectives on an open-ended question
- grounded claim verification and persistence into an investigation
- optional red-team critique and self-reflection
- a slower, investigation-centric chain

Use **`swarm_escalate.py`** when you want:

- decomposition of a complex multi-part task
- cheap wide fan-out with selective escalation
- robustness from multiple independent decompositions (`--seeds`)
- one structured synthesized result without the investigation-writing chain

Short version:

- **deep-think = cross-model idea diversity**
- **swarm = cross-seed pipeline diversity**

---

## 7) Real CLI examples

These use the actual current flags from `--help`.

## `scripts/swarm_escalate.py`

Basic single-seed run:

```bash
python3 scripts/swarm_escalate.py \
  "Explain why auth cache misses spike after deploys" \
  --pretty
```

Explicit multi-seed run:

```bash
python3 scripts/swarm_escalate.py \
  "Explain why auth cache misses spike after deploys" \
  --fanout-count 12 \
  --seeds 4 \
  --self-consistency-samples 3 \
  --escalate-with-prior-context \
  --pretty
```

Use deterministic consensus before escalation:

```bash
python3 scripts/swarm_escalate.py \
  "Audit the retry/backoff behavior across the queue workers" \
  --stigmergic-consensus \
  --stigmergic-ttl-minutes 90 \
  --pretty
```

Pre-supply subtasks instead of decomposition:

```bash
python3 scripts/swarm_escalate.py \
  "Check rollout safety" \
  --subtask "Check feature-flag default path" \
  --subtask "Check cache invalidation on rollback" \
  --subtask "Check migration idempotency" \
  --pretty
```

## `scripts/local_deep_think.py`

Basic run:

```bash
python3 scripts/local_deep_think.py \
  "What explains the recent auth cache misses?" \
  --pretty
```

Multi-model ideation with strict grounding:

```bash
python3 scripts/local_deep_think.py \
  "What explains the recent auth cache misses?" \
  --ideate-models llama3.1-agent:latest,qwen3.8:latest \
  --ideas-per-model 2 \
  --strict-grounding \
  --pretty
```

Add red-team review and disable self-reflection:

```bash
python3 scripts/local_deep_think.py \
  "What are the highest-risk bypasses in the deployment approval flow?" \
  --red-team \
  --no-self-reflect \
  --pretty
```

---

## 8) Quick guidance

If you remember only three things:

1. `--seeds N` means **N concurrent full swarm runs**, not N samples of one answer.
2. Auto-parallel-to-3 is a **CLI optimization** for batched vLLM backends when `--seeds`
   is omitted; it is not the MCP tool default.
3. `local_deep_think.py` gets diversity from multiple models; `swarm_escalate.py` gets
   diversity from multiple independent decompositions and seeds.
