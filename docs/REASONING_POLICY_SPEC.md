# Reasoning Policy Spec

This document defines the operational policy envelope for Loci's local reasoning stack:
deterministic escalation, budget limits, human-review triggers, and safe fallback
behavior when a reasoning tier fails.

It is intentionally written to mirror the behavior implemented in
`scripts/swarm_escalate.py` and the associated tests. Where the code already fixes a
threshold or decision rule, this spec records it as a hard rule. Where the correct
choice depends on deployment intent rather than mechanism, the spec calls that out as
policy-only.

## 1) Deterministic escalation policy

### Hard escalation triggers

A finding enters escalation when any of these are true:

- generation failed
- JSON parsing failed
- confidence is `low` by default
- another finding on a similar subtask is a duplicate, duplicate-like, or contradictory

The current hard thresholds are:

| Rule | Threshold |
|---|---|
| Similar subtasks | Jaccard similarity `>= 0.50` |
| Duplicate-like answers | Answer-token similarity `>= 0.82` |
| Confidence trigger | `low` by default |

The triage gate is deterministic and local. It does not depend on a secondary model call.

### Self-consistency before escalation

If `self_consistency_samples > 1`, the system retries only the triaged findings whose
reason set includes `confidence:low`.

The retry rule is:

- sample the cheap tier `N` times
- accept a strict majority only when one normalized answer appears more than `N / 2`
  times
- if there is no strict majority, keep the original finding and add
  `self_consistency_disagreement`

If the majority winner is no longer low-confidence, the confidence trigger is removed.
If it is still low-confidence, the trigger remains.

### Escalation replacement rule

Escalation is fail-open:

- replace the finding if the escalated result is `ok`
- replace the finding if the escalated answer is non-empty and the current answer is empty
- otherwise keep the current finding and annotate the failure

Escalation never requires a successful strong-model response before the overall run can
complete.

### Consensus gate

When stigmergic consensus is enabled, it is a deterministic pre-escalation gate only.
It may suppress escalation for corroborated duplicates, but it does not introduce a new
LLM tier.

## 2) Budget governor

The budget governor is the set of hard bounds that keep the swarm bounded even when the
inputs are noisy or the models fail to comply.

### Default budgets

| Stage | Default budget |
|---|---|
| Decompose | 1200 tokens |
| Cheap answer | 320 tokens |
| Synthesize | 1400 tokens |

### Opt-in tier budgets

If any opt-in tier knob is enabled, the code switches to a larger budget envelope unless
the caller explicitly pins a model:

- escalate model: `heretic-llama31-8b-instruct:latest`
- synthesize model:
  `hf.co/slevinw/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF:Q4_K_M`
- decompose budget: 2200 tokens
- synthesize budget: 2200 tokens
- synthesize-think budget: 4000 tokens

The `think` synthesis path is allowed to fail open: if the reasoning-heavy attempt comes
back empty or unparseable, the system retries with a normal synthesis call at the usual
budget.

### Clamp rules

The following values are clamped before execution:

- `fanout_count >= 1`
- `seeds >= 1`
- `self_consistency_samples >= 1`
- `reduce_group_size >= 0`
- `stigmergic_ttl_minutes >= 0.0`

Seed parallelism is bounded by the caller's explicit choice when provided. If seeds are
omitted, the CLI may auto-parallelize only when a batched vLLM endpoint is detected and
the deployment has not disabled that path.

### Budget policy boundaries

The code does **not** dynamically rebalance budgets across stages. If a stage exhausts
its budget, the fallback behavior is to degrade that stage, not to silently expand the
budget envelope.

## 3) Human-review triggers

Human review is required before any irreversible downstream action when one or more of
the following are true:

- the reasoning result is still degraded after fallback
- a finding remains flagged after triage and self-consistency
- a contradiction survives the escalation path
- the final result carries a safety annotation
- the downstream consumer is about to make a destructive, externally-visible, or
  security-sensitive change based on the reasoning output

### Review-advisory vs review-blocking

The current code only annotates some conditions, such as the safety flag. Whether that
annotation is blocking is deployment policy, not a mechanism guarantee.

Use this rule of thumb:

- **advisory** for exploratory analysis, summarization, and internal notes
- **blocking** for actions that mutate state, touch users, publish externally, or commit
  to a course of action that is hard to reverse

### Review threshold summary

If the output has any of these markers, route it to review before action:

- `degraded: true`
- `confidence: low` after all retries
- `self_consistency_disagreement`
- `duplicate_like_similar_subtask`
- `contradiction_similar_subtask`
- `safety_flag`

## 4) Failure modes and safe fallback behavior

The system is designed to fail open and remain machine-readable.

### Decomposition failure

If decomposition fails or produces unusable JSON, the fallback is a single subtask equal
to the original topic.

### Cheap-tier failure

If the cheap tier fails to generate or parse an answer, the finding is marked failed and
the triage gate may still escalate it.

### Escalation failure

If escalation fails, the system keeps the current finding, marks the failure in the
report, and completes the run.

### Synthesis failure

If the `think=True` synthesis attempt fails, the system retries with a normal synthesis
call. If the final synthesis is still empty or invalid, the output falls back to a
mechanical summary and marks the run degraded.

### Multi-seed failure

If one seed fails, the completed seeds are still merged and synthesized. A partial multi-
seed run is therefore still useful.

### Boundary condition

The reasoning system should not crash across the MCP boundary merely because one tier is
down, one parse failed, or one seed was incomplete. The acceptable failure mode is a
well-formed degraded result.

## 5) Deliberately policy-only areas

These choices are intentionally left to deployment policy rather than hard-coded
mechanism:

- whether a `safety_flag` is blocking or advisory
- whether degraded outputs may be shown to users without manual review
- the maximum fan-out allowed for a particular deployment
- whether a failed run should be retried automatically or handed to a human
- whether a particular downstream action is “irreversible” enough to require review

Those decisions should be documented by the deployment owner, but the code should keep
its current fail-open behavior regardless.
