# Loci Validated Knowledge Promotion

This spec defines the lifecycle rule for promoting a claim from tentative or episodic memory into durable, trusted knowledge.

## Goal

Validated knowledge promotion is not the same as storage-tier promotion.

`memory_promote()` only changes a finding's `hot` / `warm` / `cold` storage tier. The policy below defines when a claim is eligible to be considered durable knowledge and therefore safe for higher-trust recall.

## Required policy stack

Promotion requires a complete chain of checks:

1. direct claim or finding exists in an investigation
2. claim is verified by `verify_finding()` or equivalent adversarial verification
3. the evidence set passes the provenance firewall
4. no contradiction, stale reference, or retraction pattern invalidates the claim
5. the claim is stable across repeated reinforcement or successful replay
6. only then may `memory_promote` be used to move the finding to a higher-tier storage state

## Decision logic

A claim is eligible for promotion only when:

- `verdict == "confirmed"`
- `confidence > 0` and not degraded
- `provenance_firewall["allowed"]` is `True`
- supporting evidence includes at least one independent provenance tier
- the claim is not contradicted by stronger, newer evidence
- the claim has not been identified as stale or contaminated by `memory_retract`

A claim is blocked when:

- the verifier returned `uncertain` or `refuted`
- the evidence is only `model_asserted`
- the claim is contradicted, stale, or quarantined
- the model call failed or the result was malformed

## Promotion path

The promotion sequence must be:

1. record the finding with the correct provenance tag
2. collect independent evidence rows for support
3. run verification
4. apply the firewall check
5. run contradiction/staleness review (`memory_self_check`, `memory_retract`, or equivalent)
6. if all pass, promote the finding to the appropriate storage tier

This keeps promotion as an enforcement step, not a passive index mutation.

## Repeated verified findings

Repeated confirmation should strengthen confidence but must not bypass the gate. When the same claim is independently verified again, it becomes a stronger candidate for a higher-tier memory state.

The policy is:

- repeated confirmation increases stability and reduces volatility
- repeated contradiction or unsupported evidence triggers demotion or quarantine
- a single unverified claim never becomes durable trusted memory by accumulation alone

## Decay and demotion interaction

Promotion and decay are complementary:

- validated and repeatedly reinforced claims may be promoted
- weak, stale, or contradicted claims decay and may be demoted to `cold` or hidden below retrieval floors
- decay should not erase the original evidence provenance or a validated record's baseline importance

## Required fail-open and fail-closed semantics

- on verification/model failure: do not promote; return a degraded, non-promoted result
- on provenance failure: reject promotion
- on contradiction or retraction discovery: block or demote the candidate
- on any optional subsystem outage: do not crash the promotion flow; return a safe degraded status

## Acceptance criteria

A validated knowledge promotion flow is acceptable when all are true:

- no `model_asserted`-only claim is promoted without independent evidence
- verification gate failures block promotion even when the claim appears repeated or popular
- stale or contradicted claims are demoted or retracted before they re-enter hot memory
- `memory_promote` is used only after the validation gate passes

## Relevance to existing code

This policy is built on the current repo implementation:

- `mcp/verify.py` — adversarial verification loop
- `mcp/provenance_firewall.py` — evidence gate
- `mcp/server.py` — `memory_promote`, `memory_demote`, `memory_self_check`, `memory_retract`
- `mlops/memory/decay.py` — temporal decay and importance floor
