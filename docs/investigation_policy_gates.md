# Investigation Policy Gates

Policy enforcement hooks wired into the investigation reasoning chain.
Connects [`PolicyTier`](../mcp/policy.py) constraints and
[EVO-RAG](ARCHITECTURE.md) reward signals to three high-value call sites
in `server.py`.

---

## The Three Gates

### 1. `investigation_pre_answer_check` — r_ref refusal reward gate

**Location:** `server.py:4672-4875`

`investigation_pre_answer_check` already classifies findings as
`supported`, `unsupported`, or `ambiguous`. The gate reads that
classification and decides whether the model should refuse to synthesize
an answer.

**What it prevents:** hallucinated synthesis when the evidence base is
too weak. Without this gate, the model answers even when ≤30% of claims
are supported by retrieved evidence.

**Integration point:** `evaluate_pre_answer_for_refusal()` in
`mcp/investigation_policy.py`. When it returns `recommend_refusal=True`,
the caller should decline to produce an answer and instead surface the
evidence gap to the user. The `r_ref` value is pushed to the DRPG
`TrajectoryStore` so the policy can learn from the decision (see below).

---

### 2. `investigation_reason` — perspectives depth capping

**Location:** `server.py:9757-9905`

The `perspectives` parameter is clamped to 1–5 by the tool schema. The
gate further restricts it based on the active `PolicyTier`.

**What it prevents:** expensive multi-perspective reasoning on light-tier
policies where latency and cost budgets are tight.

**Tier table:**

| Tier   | Max perspectives |
|--------|-----------------|
| light  | 1               |
| medium | 3               |
| heavy  | 5 (no extra cap) |

**Integration point:** `cap_perspectives_to_tier(requested, tier)` in
`mcp/investigation_policy.py`. Call before passing `perspectives` to the
reasoning loop. The function logs when capping occurs.

---

### 3. `investigation_verify_all` — heuristic pre-pass

**Location:** `server.py:7371-7465`

`investigation_verify_all` calls `verify_finding` for every open finding.
For large investigations this is expensive. The heuristic pre-pass skips
findings whose `numeric_confidence` (or string `confidence`) already
exceeds the skip threshold.

**What it prevents:** redundant LLM verification calls for findings that
are already high-confidence.

**Integration point:** `filter_findings_for_verification(findings,
skip_threshold)` in `mcp/investigation_policy.py`. Only the returned
`to_verify` list should be passed to `verify_finding`; `skipped` findings
are accepted as-is.

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `LOCI_POLICY_ENABLE` | `0` | Set to `1`/`true`/`yes` to activate hooks |
| `LOCI_PRE_ANSWER_REFUSAL_THRESHOLD` | `0.3` | `supported_ratio ≤ threshold` triggers refusal |
| `LOCI_VERIFY_SKIP_THRESHOLD` | `0.85` | Findings with confidence ≥ threshold skip verification |
| `LOCI_TRAJECTORY_DIR` | `~/.loci/trajectories` | DRPG TrajectoryStore directory |

---

## r_ref Feedback Loop into DRPG Trajectory

`r_ref` is the EVO-RAG refusal reward signal:

- **`r_ref = 1.0`** — refusal is warranted (evidence insufficient). The
  policy is rewarded for not hallucinating.
- **`r_ref < 0`** — refusal would be unnecessary. Computed as
  `-0.5 × (1 - supported_ratio)`; the more evidence there is, the
  stronger the anti-refusal penalty.

`push_refusal_step_to_trajectory()` appends the decision as a step to the
DRPG `TrajectoryStore` JSONL file for the active `policy_id`. The
`DrpgSuggestionGenerator` in `drpg_evolution.py` reads this trajectory
during the next evolution cycle and may suggest tightening or relaxing the
refusal threshold.

### Example end-to-end

1. Investigation has 5 findings: 1 supported, 4 unsupported →
   `supported_ratio = 0.20`.
2. `evaluate_pre_answer_for_refusal` returns
   `recommend_refusal=True, r_ref=1.0`.
3. Caller refuses to synthesize an answer and surfaces the evidence gap.
4. `push_refusal_step_to_trajectory` writes:
   ```json
   {
     "step": 0,
     "action": "REFUSE",
     "reward": {"r_ref": 1.0, "r_step": -0.01, ...},
     "policy_id": "pol-default",
     "investigation_id": "inv-abc123"
   }
   ```
5. During the next DRPG evolution cycle, the trajectory scorer sees a
   positive `r_ref` and the policy learns that this class of investigation
   (many unsupported claims) needs better evidence before synthesis.

---

## Integration with litellm `async_pre_call_hook`

To wire the refusal gate into every LLM call made by an investigation
tool, add a hook in `hooks.py`:

```python
from investigation_policy import (
    get_investigation_policy_config,
    evaluate_pre_answer_for_refusal,
    push_refusal_step_to_trajectory,
)

async def async_pre_call_hook(data, user_api_key_dict, cache, call_type):
    cfg = get_investigation_policy_config()
    if not cfg["enabled"]:
        return data

    support_summary = data.get("_investigation_support_summary")
    if support_summary is None:
        return data

    result = evaluate_pre_answer_for_refusal(
        support_summary,
        refusal_threshold=cfg["pre_answer_refusal_threshold"],
    )
    if result["recommend_refusal"]:
        push_refusal_step_to_trajectory(
            policy_id=data.get("_policy_id", "pol-default"),
            investigation_id=data.get("_investigation_id", "unknown"),
            r_ref=result["r_ref"],
            trajectory_dir=cfg["trajectory_dir"],
        )
        raise Exception(f"[policy-gate] refusal: {result['reason']}")

    return data
```

The hook attaches to litellm via `litellm.callbacks = [MyHook()]` in
`server.py` startup.
