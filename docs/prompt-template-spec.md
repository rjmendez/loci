# Prompt template specification

This spec defines the reusable prompt templates for the swarm runtime. The templates are intentionally aligned with the behavior already implemented in `scripts/swarm_escalate.py`, `scripts/swarm_supervisor.py`, and `mcp/llm_tools.py`.

## Design principles

- preserve provenance: every template should request evidence, rationale, and scope
- prefer structured JSON: the runtime expects machine-readable output from the swarm pipeline
- keep output grounded: if evidence is weak, the prompt should force a conservative answer, not a guess
- fail-open: when a template is mis-scaled or a model fails, the runtime should degrade cleanly instead of crashing

## Template set

### Decompose template

```python
PROMPT_TEMPLATES["decompose"] = '''
You are the swarm decomposer.
Task: {topic}
Goal: break the task into atomic subtasks that can be answered independently.
Constraints:
- Use the current reasoning policy: fail-open, bounded fanout, no hidden assumptions.
- Prefer evidence-fit over broad-but-weak search.
- Return structured JSON with task, fail_open, decision_tree, assignments, and source_policy.
- Preserve provenance with each subtask's preferred source and rationale.
'''
```

### Triage template

```python
PROMPT_TEMPLATES["triage"] = '''
You are the swarm triage critic.
Review the findings against the plan and the current task.
Task: {topic}
Plan: {plan_json}
Findings:
{findings_json}
Return JSON with verdicts for every finding: finding_index, on_task, supported, source_appropriate, guidance, rationale.
Do not invent evidence; only judge whether the finding fits the task and the routing plan.
'''
```

### Escalation template

```python
PROMPT_TEMPLATES["escalate"] = '''
You are the escalation specialist.
Review only the flagged findings and produce a stronger answer while staying within the current budget.
Task: {topic}
Flagged findings: {flagged_findings_json}
Current rationale: {rationale_json}
Requirements:
- keep the answer grounded in the evidence actually provided
- explain why the original answer failed or was mis-routed
- if a repair is impossible, return a conservative answer and leave the finding degraded rather than fabricating certainty
'''
```

### Synthesis template

```python
PROMPT_TEMPLATES["synthesize"] = '''
You are the swarm synthesizer.
Combine validated findings into one structured result for downstream use.
Task: {topic}
Validated findings: {findings_json}
Constraints:
- summarize only supported, relevant, and provenance-preserving evidence
- cite confidence, model, and tier_reached for each retained finding
- mark degraded runs explicitly rather than silently smoothing over failures
- output JSON only
'''
```

## Mandatory response fields

Each template should preserve the fields already emitted by the current runtime:

- `subtask`
- `answer`
- `confidence`
- `tier_reached`
- `model`
- `ok`
- `why`
- `escalation_attempted`
- `escalation_reasons`

## Grounding contract

A template is considered valid only when its output:

1. stays bound to the user task and the evidence available to the run,
2. keeps the `lineage` trail for each retained finding, and
3. degrades explicitly when the evidence is weak or the model output is malformed.

This aligns with both the runtime's `SwarmFinding` data model and the policy rules in `docs/REASONING_POLICY_SPEC.md`.

## Acceptance criteria

- Templates are reusable across swarm runs and not hard-coded to one single task.
- Each template enforces provenance, scope, and output-shape constraints.
- Output remains machine-readable and compatible with the existing swarm JSON contract.
