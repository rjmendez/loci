# Memory Route Policy Adapter

Bridges `memory_route_policy_optimize` (routing optimizer) with the Loci
[Policy backend](../mcp/policy.py) and DRPG evolution loop.

## How the pieces fit together

```
memory_route traces
       │
       ▼
memory_route_policy_optimize          ← existing MCP tool
       │  {top_k, deduplicate,
       │   dedup_threshold, score_delta, ...}
       ▼
memory_route_to_policy                ← NEW MCP tool (this adapter)
       │  adapt_route_optimization_to_policy()
       ▼
PolicyBackend.store(policy)           ← policy.md object persisted
       │
       ▼
DRPG evolution loop (drpg_evolution.py)
       │  evolve_policy(), PolicyPromoter
       ▼
Tier promotion / demotion decision
       │
       ▼  (before committing downgrade)
counterfactual_gate                   ← NEW MCP tool (pre-downgrade gate)
       │  memory_route_counterfactual_simulate
       │  + evaluate_counterfactual_gate()
       ▼
safe_to_downgrade: true / false
```

### Why these two tools are complementary

| Tool | Role |
|---|---|
| `memory_route_policy_optimize` | Proposes a better routing policy from historical traces |
| `policy_evolve` (DRPG loop) | Promotes/demotes policy tier based on trajectory rewards |
| `memory_route_to_policy` | **Connects** them — stores optimizer output as a Policy object |
| `counterfactual_gate` | **Guards** tier downgrades — validates impact before commit |

The optimizer uses statistics from prior routing decisions; the DRPG loop
uses live reward signals from retrieval trajectories. The adapter converts
the optimizer's recommendation dict into the Policy schema understood by
`drpg_evolution.py`, creating a closed feedback loop.

---

## `mcp__loci__memory_route_to_policy`

Runs `memory_route_policy_optimize` and stores the recommendation as a
`Policy` object in the PolicyBackend.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `investigate_id` | `str` | `""` | Restrict to a specific investigation's audit log |
| `limit` | `int` | `30` | Max number of routing decisions to analyse |
| `days` | `int` | `7` | Look-back window for global audit entries |
| `min_decisions` | `int` | `8` | Minimum decisions required to attempt optimization |

### Response

```json
{
  "adapted": true,
  "policy_id": "3f8a2c1d-...",
  "tier": "medium",
  "score_delta": 0.142,
  "reason": "stored_ok"
}
```

`adapted: false` is returned (without error) when the optimizer finds no safe
improvement (`optimized: false`) or when fewer than `min_decisions` traces are
available.

---

## `mcp__loci__counterfactual_gate`

Runs `memory_route_counterfactual_simulate` under the proposed policy
parameters and returns a go/no-go verdict for tier downgrades.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `top_k` | `int` | `3` | Candidate count for counterfactual policy |
| `deduplicate` | `bool` | `true` | Enable deduplication in counterfactual |
| `dedup_threshold` | `float` | `0.8` | Cosine similarity threshold for dedup |
| `max_changed_fraction` | `float` | `0.05` | Max fraction of decisions allowed to change |

### Response

```json
{
  "safe_to_downgrade": true,
  "changed_fraction": 0.0333,
  "changed": 1,
  "total": 30,
  "reason": "changed_fraction=3.33% <= 5.00%"
}
```

A downgrade is **safe** when `changed_fraction ≤ max_changed_fraction`.
The gate is conservative by design: when no historical traces are available,
`changed_fraction` defaults to `0.0` (no evidence of disruption → safe).

---

## Tier mapping

The adapter maps optimizer output to a policy tier using two signals:
`score_delta` (how much better the proposed policy is) and
`baseline_objective` (how good the current policy already is).

| Condition | Tier | Interpretation |
|---|---|---|
| `score_delta > 0.3` AND `baseline_objective < 0.5` | **light** | Large gain from a weak baseline → aggressive tightening appropriate |
| `score_delta > 0.1` | **medium** | Moderate gain → cautious refinement |
| `score_delta ≤ 0.1` | **heavy** | Small gain or high baseline → preserve existing routing |

The `heavy` tier corresponds to high trust in the current routing strategy;
`light` enables the DRPG loop to explore more aggressively.

---

## Example workflow

```python
# 1. Adapt latest routing optimization into a Policy
result = mcp__loci__memory_route_to_policy(days=7, limit=50)
# → {"adapted": true, "policy_id": "abc...", "tier": "medium", ...}

# 2. DRPG loop evolves the policy based on live trajectory rewards
#    (runs automatically via drpg_evolution.evolve_policy)

# 3. Before committing a tier downgrade, validate with counterfactual gate
gate = mcp__loci__counterfactual_gate(top_k=3, dedup_threshold=0.8)
# → {"safe_to_downgrade": true, "changed_fraction": 0.02, ...}

if gate["safe_to_downgrade"]:
    # Commit downgrade
    ...
```
