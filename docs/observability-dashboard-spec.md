# Observability dashboard specification

This spec defines the runtime observability artifacts and metrics for swarm execution. It is grounded in the output shapes already emitted by `scripts/swarm_escalate.py`, `scripts/swarm_supervisor.py`, and `mcp/llm_tools.py`.

## Goals

The dashboard must reveal whether the swarm is:

- decomposing the topic correctly
- staying within budget
- escalating only when justified
- degrading gracefully under failure
- preserving provenance through the final synthesis

## Core dashboard views

### Run summary

Show:

- topic
- seed count
- fanout count
- escalated count
- escalation rate
- degraded flag
- confidence distribution
- final summary

### Triage watch

Show:

- flagged indices
- per-finding supported / unsupported state
- source appropriateness
- duplicate-like / contradictory pairs
- escalation reasons

### Budget health

Show:

- calls
- prompt tokens
- completion tokens
- total tokens
- cost_usd
- max_calls / max_tokens / max_cost_usd

### Escalation flow

Show:

- cheap-tier outcomes
- model used for each tier
- count of succeed / fail / fail-open transitions
- whether escalation occurred after a low-confidence flag

## Metrics model

```python
OBSERVABILITY_DASHBOARD = {
  "metrics": {
    "fanout_count": "number of subtasks produced by decomposition",
    "escalated_count": "number of findings that needed escalation",
    "escalation_rate": "escalated_count / fanout_count",
    "confidence_distribution": "low/medium/high counts across findings",
    "degraded": "true when a stage must fail open or a parser drops a result",
    "parse_failures": "count of JSON parse failures or malformed model output",
    "latency_ms": "elapsed run time per stage",
    "budget_cost_usd": "SupervisorBudget.cost_usd with current token accounting",
    "seed_count": "number of independent swarm seeds merged for the output",
    "lineage_length": "how many provenance-bearing records are retained in the run"
  }
}
```

## Runtime artifact schema

The runtime already emits a useful structure that the dashboard can surface directly:

```json
{
  "topic": "...",
  "findings": [{ "subtask": "...", "answer": "...", "confidence": "high", "tier_reached": "cheap", "model": "...", "ok": true }],
  "summary": "...",
  "stats": {
    "fanout_count": 12,
    "escalated_count": 3,
    "escalation_rate": 0.25
  },
  "triage": {
    "flagged_indices": [1, 3],
    "flagged_count": 2,
    "escalation_reasons": { "1": ["confidence:low"] }
  },
  "tiers": {
    "cheap": { "count": 12, "ok": 10 },
    "escalate": { "attempted": 2, "succeeded": 1 }
  },
  "lineage": [{ "subtask": "...", "reason": "...", "confidence": "..." }]
}
```

## Operational semantics

- `degraded: true` is a visible operator signal, not a silent anomaly.
- `budget` and `triage` views are the first slices to inspect when a run looks wrong.
- `lineage` is the provenance layer; the dashboard should expose it alongside summary metrics.
- The dashboard must not hide unsupported evidence. The operator should see where the run weakened or failed open.

## Acceptance criteria

- The dashboard exposes both quality and cost signals.
- The runtime can derive the dashboard directly from the existing swarm output contract.
- Provenance is visible in the final run details and not only in the raw logs.
