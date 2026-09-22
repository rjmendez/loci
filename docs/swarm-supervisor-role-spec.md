# Swarm supervisor role specification

This spec is the runtime-facing contract for the swarm supervisor. It is grounded in the behavior already implemented in `scripts/swarm_supervisor.py`, `scripts/swarm_escalate.py`, and `mcp/llm_tools.py`, and it intentionally reuses those actual components rather than introducing a second orchestration model.

## Scope

The supervisor owns the operational loop for the swarm run:

- decompose the topic into tractable subtasks
- route each subtask to the best evidence source
- budget calls, tokens, and cost
- validate findings against the task and the source plan
- repair only the flagged findings
- escalate selectively when confidence or grounding is weak
- synthesize a final structured result without losing provenance

## Responsibilities

### 1. Plan and guardrails

The supervisor is responsible for building the routing plan and then enforcing the guardrails in that plan. The runtime already exposes this as `plan_source_routing()` and uses `SupervisorBudget` for the bounded budget envelope.

Concrete requirements:

- keep the root task visible throughout the run
- prefer evidence-rich sources when task-fit is clear
- fall back to broader sources only when the task requires it
- enforce max calls, max tokens, and max cost before a stage overspends

### 2. Triage and repair

The supervisor does not re-answer every finding. It only re-routes or repairs the ones the triage critic marks as unsupported, wrong-source, or low-confidence.

This matches the current implementation:

- `supervise_findings()` emits verdicts
- `supervise_and_correct()` redispatches only flagged findings
- a stall triggers re-plan rather than a broad re-think

### 3. Fail-open behavior

A supervisor-bound failure is not a hard crash. The existing runtime is intentionally fail-open:

- parse failures lead to degraded output rather than a thrown exception
- budget exhaustion aborts the expensive branch while preserving the partial plan
- failed escalations keep the last valid answer and annotate the result

### 4. Provenance preservation

Every stage must retain enough evidence for downstream review:

- the source of the evidence
- the task and subtask being evaluated
- the rationale for route selection or repair
- the confidence value and why the answer was considered supported or unsupported

This is the same design principle behind the runtime's `lineage` output and the policy in `docs/REASONING_POLICY_SPEC.md`.

## Runtime contract

```python
{
  "task": "Identify spider species near a location",
  "fail_open": False,
  "decision_tree": [{
    "sub_need": "GPS/photo-verified local observations",
    "preferred_sources": ["iNaturalist"],
    "fallback_sources": ["DuckDuckGo"],
    "rationale": "Structured local evidence is required."
  }],
  "assignments": [{
    "sub_need": "GPS/photo-verified local observations",
    "source": "iNaturalist",
    "rationale": "Best fit for local observations."
  }],
  "source_policy": "Use iNaturalist first; use taxonomy sources only as background."
}
```

This is the concrete contract the supervisor should enforce across the runtime and the docs layer.

## Required runtime hooks

- `scripts.swarm_supervisor.plan_source_routing`
- `scripts.swarm_supervisor.SupervisorBudget`
- `scripts.swarm_supervisor.supervise_findings`
- `scripts.swarm_supervisor.supervise_and_correct`
- `scripts.swarm_escalate.SwarmConfig`
- `mcp.llm_tools.swarm_reason`

## Acceptance criteria

- Supervisor owns orchestration and budget governance.
- Triage and re-plan are selective rather than blanket retries.
- Degraded output remains machine readable and provenance-aware.
- The supervisor behavior is consistent with the current swarm code path and policy docs.
