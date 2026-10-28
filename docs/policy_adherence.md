# Policy Adherence Confidence Tracking

## Why Adherence Tracking Matters

Loci can store policies and enforce tier routing — but without measurement, there is no way to know whether a policy actually *changed* model behavior. The question this module answers is: **"Did the policy work?"**

Adherence tracking closes the feedback loop:

```
Policy set → LiteLLM call → tier routing fires? → r_cor measured → adherence recorded
                                                                          ↓
                                                       compute_report → recommendation
                                                                          ↓
                                                       feed effectiveness into evolve_policy()
```

Without this, policies are aspirational. With it, they are measurable.

---

## `PolicyAdherenceRecord` Fields

| Field | Type | Description |
|---|---|---|
| `policy_id` | `str` | ID of the policy being tracked |
| `call_id` | `str` | Unique ID for this LiteLLM call |
| `tier_requested` | `str` | Tier from the policy (`light`/`medium`/`heavy`) |
| `model_requested` | `str` | Model the caller asked for |
| `model_used` | `str` | Model actually used after tier routing |
| `was_downgraded` | `bool` | Whether tier routing rewrote the model |
| `r_cor` | `float` | Correctness reward signal (0.0–1.0) |
| `r_step` | `float` | Step cost (typically negative, e.g. `-0.01`) |
| `timestamp` | `str` | UTC ISO-8601 timestamp (auto-filled) |
| `context` | `dict` | Optional extra metadata |

---

## How to Record: `mcp__loci__policy_adherence_record`

Call this tool after each LiteLLM call that is governed by a policy:

```python
# After your LiteLLM call completes:
mcp__loci__policy_adherence_record(
    policy_id="my-cost-policy",
    call_id="req-abc123",
    tier_requested="light",
    model_requested="claude-opus-4.8",   # what the caller wanted
    model_used="claude-haiku-4.5",        # what tier routing selected
    was_downgraded=True,
    r_cor=0.91,                           # from your eval signal
    r_step=-0.01,
)
# Returns: {"recorded": true, "policy_id": "my-cost-policy", "call_id": "req-abc123"}
```

Records are stored as JSONL at `$LOCI_ADHERENCE_DIR/{policy_id}.jsonl` (default: `~/.loci/adherence/`).

---

## How to Report: `mcp__loci__policy_adherence_report`

```python
mcp__loci__policy_adherence_report(
    policy_id="my-cost-policy",
    min_records=5,   # default; wait for at least this many records
)
```

Returns:

```json
{
  "policy_id": "my-cost-policy",
  "record_count": 20,
  "adherence_rate": 0.6,
  "downgrade_effectiveness": 0.978,
  "confidence": 0.978,
  "recommendation": "effective",
  "avg_r_cor_downgraded": 0.91,
  "avg_r_cor_baseline": 0.93,
  "avg_r_step": -0.01
}
```

### Recommendation values

| Value | Meaning |
|---|---|
| `effective` | Downgraded calls maintain ≥85% of baseline correctness and routing fires >10% of the time |
| `ineffective` | Downgraded calls fall below 70% of baseline correctness and routing fired >3 times |
| `monitoring` | Neither threshold met — continue collecting data |
| `insufficient_data` | Fewer than `min_records` recorded — no recommendation possible |

---

## Integration with DRPG: Feed `downgrade_effectiveness` into `evolve_policy()`

The `downgrade_effectiveness` field is a normalized correctness-preservation ratio. Feed it as an `r_cor` signal into the policy evolution loop:

```python
report = json.loads(mcp__loci__policy_adherence_report(policy_id="my-cost-policy"))

if report["recommendation"] != "insufficient_data":
    # mcp__loci__policy_trajectory_push no longer exists: the recorder writes this step
    # (loci/drpg.py TrajectoryStore.push; every reward value must be within [-1, 1]).
    record_trajectory_step(
        policy_id="my-cost-policy",
        step=current_step,
        action="tier_routing",
        reward_vector={  # all seven keys are required by the recorder
            "r_ret": 0.0, "r_mem": 0.0, "r_ovp": 0.0, "r_bkp": 0.0, "r_ref": 0.0,
            "r_cor": min(1.0, report["downgrade_effectiveness"]),
            "r_step": report["avg_r_step"],
        },
    )
    # Then run evolution:
    mcp__loci__policy_evolve(policy_id="my-cost-policy")
```

This closes the full loop: adherence data drives DRPG reward, which evolves the policy itself.

---

## Environment Variable

| Variable | Default | Description |
|---|---|---|
| `LOCI_ADHERENCE_DIR` | `~/.loci/adherence` | Directory for JSONL adherence files |

---

## Example: Policy Declared Effective

A policy has 20 recorded calls. 60% were downgraded (`adherence_rate=0.6`). The downgraded calls averaged `r_cor=0.91`; the baseline (non-downgraded) calls averaged `r_cor=0.93`.

```
effectiveness = 0.91 / 0.93 = 0.978
confidence    = min(1.0, (20/20) × 0.978) = 0.978
recommendation = "effective"   # 0.978 ≥ 0.85 and adherence_rate=0.6 > 0.1
```

The policy is working: tier routing fires reliably and downgraded calls preserve nearly all correctness.
