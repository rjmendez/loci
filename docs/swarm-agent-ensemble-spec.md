# Swarm agent ensemble specification

This spec defines the runtime-friendly agent ensemble for the swarm layer. It is intentionally a thin composition layer over the actual work already implemented in `scripts/swarm_escalate.py` and `scripts/swarm_supervisor.py`.

## Ensemble structure

| Role | Models / tier | Responsibility |
|---|---|---|
| Supervisor | orchestrator | maintains the plan, budgets, and triage gate |
| Decomposer | cheap tier or dedicated decomposition model | converts the task into bounded subtasks |
| Cheap worker | cheap model | answers most subtasks with low latency |
| Triage critic | local verifier / runtime guard | flags unsupported, wrong-source, or low-confidence findings |
| Escalator | stronger model | answers only the flagged slice |
| Synthesizer | final synthesis model | merges valid findings into a single JSON result |

## Composition rules

1. The supervisor is the single control point. It decides which evidence source is appropriate and when to escalate.
2. The ensemble is role-based, not free-form. A cheap worker should not silently bypass the triage stage.
3. Escalation is selective and bounded. Only findings flagged by triage are escalated.
4. Synthesis is downstream of validation. The final output must be grounded in retained findings, not the raw full run output.
5. The whole run remains fail-open. A broken tier degrades the result without aborting the run.

## Interaction flow

```text
topic
  -> decomposer
      -> cheap worker answers
      -> triage critic validates
          -> pass: keep
          -> fail: escalate only flagged slice
  -> synthesizer merges validated findings
  -> final summary + provenance trail
```

## Alignment with runtime code

This matches the present implementation:

- `scripts/swarm_escalate.py` defines the multi-stage pipeline and `SwarmConfig` fields
- `scripts/swarm_supervisor.py` defines the plan/triage/repair loop
- `mcp/llm_tools.py` wraps those into a single `swarm_reason` tool contract
- `docs/REASONING_POLICY_SPEC.md` defines the policy envelope for escalation and budget limits

## Required runtime interfaces

- `SwarmConfig.topic`, `fanout_count`, `seeds`, `cheap_model`, `escalate_model`, `synthesize_model`
- `SupervisorBudget.calls`, `prompt_tokens`, `completion_tokens`, `cost_usd`
- `lineage` results containing `subtask`, `answer`, `confidence`, `tier_reached`, `model`, `ok`, `why`

## Acceptance criteria

- Composition is explicit and small; no ad hoc free-for-all between models.
- The supervisor retains control over all escalation and repair decisions.
- The ensemble can operate with a cheap tier and a stronger tier without losing provenance.
