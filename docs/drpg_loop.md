# DRPG Loop — Trajectory Persistence & Policy Evolution

## Overview

The **DRPG (Dynamic Retrieval-based Policy Generation) loop** closes the reward feedback cycle in Loci: reward signals computed during retrieval episodes are persisted to disk, scored, and used to evolve policy documents back into Loci memory for curriculum learning.

---

## Architecture

```
  LiteLLM eval_hook_stub
         │
         │ StepFeedback.to_dict()
         ▼
  ┌─────────────────┐   push(policy_id, step)
  │ TrajectoryStore  │◄──────────────────────────── recorder only (not an MCP tool)
  │  ~/.loci/traj/  │
  │  {policy_id}.   │
  │  jsonl          │
  └────────┬────────┘
           │ read(policy_id)
           ▼
  ┌──────────────────┐
  │ TrajectoryScorer │   score = Σ(r_ret - |r_ovp| - |r_bkp| + r_ref - |r_step| + r_cor) / N
  └────────┬─────────┘
           │ score: float
           ▼
  ┌─────────────────────┐
  │   PolicyPromoter    │   score > 0.7  → promote
  │                     │   score < 0.2  → demote
  │                     │   else         → keep
  └────────┬────────────┘
           │ action
           ▼
  ┌──────────────────────────┐
  │ DrpgSuggestionGenerator  │   inspects step patterns → suggestions
  └────────┬─────────────────┘
           │
           ▼
  ┌────────────────────┐
  │   PolicyBackend    │   if promote: bump version, set source="drpg-evolved"
  │  ~/.loci/policies/ │   re-store policy document
  └────────────────────┘
```

---

## Trajectory Storage

Each step is appended as a JSONL line to `$LOCI_TRAJECTORY_DIR/{policy_id}.jsonl` (default: `~/.loci/trajectories/`).

**Step record format:**
```json
{
  "step": 3,
  "action": "SEARCH",
  "reward": {
    "r_ret": 1.0,
    "r_ovp": -0.1,
    "r_bkp": 0.0,
    "r_ref": 0.0,
    "r_step": -0.05,
    "r_cor": 0.5
  },
  "policy_id": "abc-123",
  "timestamp": "2026-10-17T18:00:00Z",
  "_ts": "2026-10-17T18:00:00Z"
}
```

Steps are written by `eval_hook_stub` in `mcp/hooks.py` automatically whenever a LiteLLM call carries `policy_id`, `step`, `action`, and `reward_vector` in its kwargs.

There is no MCP tool for pushing steps. `mcp__loci__policy_trajectory_push` was removed: it had no
caller identity, so any agent could write the reward vector that decides promotion. Only the recorder
writes trajectories: `eval_hook_stub`, or `record_trajectory_step()` in `mcp/tools/drpg_tools.py`, which
requires all seven keys. `TrajectoryStore.push` itself refuses any reward value outside [-1, 1] (and,
with `strict=True`, a vector missing a key) with `ValueError("invalid reward: ...")`.

Each policy's trajectory file is bounded to 5000 rows or 1 MiB. The next push past either limit renames
the file to `{policy_id}.jsonl.1` (then `.2`, ...; plain JSONL, still readable) and starts a fresh one;
reads take only the tail of the current file. Policy ids must match `^[a-z0-9][a-z0-9_.-]{0,63}$`.

---

## Triggering Evolution

Call the MCP tool:
```
mcp__loci__policy_evolve(policy_id="abc-123")
```

Returns:
```json
{
  "policy_id": "abc-123",
  "action": "promote",
  "score": 0.82,
  "steps_analyzed": 47,
  "suggestions": [
    {
      "suggestion": "Enable semantic dedup filter: high subquery overlap detected",
      "confidence": 0.72,
      "affects": "dedup_filter"
    }
  ]
}
```

---

## Reward Vector Weights

The 7D reward vector (from EVO-RAG research) has different weights per phase:

| Component | Key     | Discovery weight | Refinement weight | Description                        |
|-----------|---------|-----------------|-------------------|------------------------------------|
| Retrieval Bonus      | `r_ret`  | +1.0  | +0.5  | Reward for retrieving relevant docs   |
| Retrieval Penalty    | `r_rem`  | −0.05 | −0.2  | Penalty for excess search calls       |
| Overlap Penalty      | `r_ovp`  | −0.1  | −0.15 | Penalty for duplicate subqueries      |
| Backtrack Penalty    | `r_bkp`  | −0.2  | −0.3  | Penalty for backtracking              |
| Refusal Reward       | `r_ref`  | −0.5  | +1.0  | Reward/penalty for refusal decisions  |
| Step Penalty         | `r_step` | −0.02 | −0.1  | Per-step cost                         |
| Answer Correctness   | `r_cor`  | +0.5  | +1.0  | Reward for correct answer             |

The **scoring formula** used by `TrajectoryScorer` is a weighted mean over all steps. `evolve_policy`
builds the scorer from the policy itself (`TrajectoryScorer.from_policy`), so the weights are the ones in
the table above for the policy's tier and phase (`Policy.get_reward_vector_weights()`):
```
per_step = w_ret*r_ret + w_mem*r_mem + w_ovp*|r_ovp| + w_bkp*|r_bkp| + w_ref*r_ref + w_step*|r_step| + w_cor*r_cor
score    = Σ(per_step) / N
```
The table's `r_rem` column is the `r_mem` term: a weight mapping that still says `r_rem` is mapped to
`r_mem` with one logged warning. `TrajectoryScorer()` with no argument keeps the fixed default weights.

`mcp__loci__policy_evolve_all` evolves its policies one after another in this process (no process
pool), at most 64 per call, and returns `status` 1 with `error: "rows_read=0"` when no trajectory rows
were read.

---

## Connection to EVO-RAG Curriculum Scheduling

The DRPG loop is the feedback mechanism for EVO-RAG curriculum learning:

1. **Discovery phase** (`discovery_mode: true`): broad exploration, high `r_ret` weight, low `r_ref` penalty — encourages the agent to retrieve widely.
2. **Refinement phase** (`discovery_mode: false`): efficiency focus, high `r_ref` and `r_cor` weights — encourages the agent to refuse when evidence is insufficient and to be precise.

`DrpgSuggestionGenerator` detects when to transition:
- High backtrack rate (>30%) → reduce `max_retrieval_hops`
- High overlap (avg `|r_ovp|` > 0.15) → enable dedup filter
- High refusal rate (>40%) → transition from discovery to refinement phase

Promoted policies (score > 0.7) have their `metadata.source` set to `"drpg-evolved"` and `metadata.version` incremented, making them traceable in Loci memory as curriculum-evolved artifacts.

---

## Environment Variables

| Variable                   | Default                        | Description                            |
|----------------------------|--------------------------------|----------------------------------------|
| `LOCI_TRAJECTORY_DIR`      | `~/.loci/trajectories`         | Directory for JSONL trajectory files   |
| `LOCI_POLICY_ALLOWED_TOOLS`| _(unrestricted)_               | Comma-separated allowed MCP tool names |
| `LOCI_POLICY_ENABLE`       | `0`                            | Enable policy enforcement hooks        |
