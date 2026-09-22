# Verification Gate Spec

This spec defines the acceptance gates that must pass before a claim is treated as confirmed, durable memory, or ready for promotion.

## Scope

The authoritative implementation is in:

- `mcp/verify.py`
- `mcp/provenance_firewall.py`
- `mcp/server.py` (`memory_self_check`, `verify_finding`, and evidence-backed retrieval)

## Core rule

A claim is only `confirmed` when a skeptical verifier fails to refute it and the provenance firewall passes.

The current verifier default is adversarial and skeptical:

- it actively tries to refute the claim
- it returns `refuted` if a concrete failure is found
- it returns `confirmed` only when the claim survives the attack
- it returns `uncertain` when the evidence is insufficient or the backend fails

## Provenance gate

The separation of memory tier and evidence provenance is mandatory.

Evidence provenance tiers are:

- `human_authored`
- `tool_verified`
- `deterministic_derived`
- `model_asserted`

The firewall rule is:

- a `model_asserted` claim is not considered verified by other `model_asserted` findings alone
- it requires at least one `human_authored`, `tool_verified`, or `deterministic_derived` evidence row
- legacy untagged rows default to `tool_verified` for compatibility

This is enforced by `assert_evidence_firewall()` in `mcp/provenance_firewall.py`.

## Fail-open / fail-closed behavior

The current behavior intentionally combines robust fallback with strict gating:

- backend/model failure or parse failure -> `uncertain`, not `confirmed`
- malformed output -> degraded result with `uncertain`
- a candidate that fails provenance -> `uncertain` and the firewall records `allowed=False`
- a fail-open firewall malfunction may return `allowed=True` with `degraded=True`, but this is a fallback for robustness and must be treated as degraded evidence, not a clean verification pass

This is a safety property: the system fails open for service resilience while preserving the skeptical default.

## Required gate sequence

Before a claim can be promoted into more durable memory, the following gates must pass:

1. Claim is non-empty and parseable.
2. Verification backend responds with a valid JSON verdict.
3. Verdict is `confirmed` or equivalent strong evidence, not `uncertain` or `refuted`.
4. Provenance firewall allows the claim before the evidence set is used as support.
5. No contradiction or retraction signal invalidates the claim.

If any required gate fails, promotion is blocked.

## Special cases

### model_asserted candidate with only model_asserted evidence

This is explicitly rejected. The verifier must not run the model on circular model-only support. It returns `uncertain` and `provenance_firewall["allowed"]` is `False`.

### tool_verified evidence

A `model_asserted` candidate is accepted when it is backed by tool-verified evidence, such as a matching test or audit receipt.

### malformed output

The verifier coerces unknown verdicts to `uncertain` and clamps confidence to `[0, 1]`.

## Acceptance criteria

A claim is accepted as verified only when:

- the skeptic cannot refute it
- the evidence passes provenance gating
- the output is parseable and valid
- confidence never exceeds the model's valid range
- any failure or ambiguity produces `uncertain`, not a silent pass

## Relevance to existing code

This spec matches the current implementation in:

- `mcp/verify.py`
- `mcp/provenance_firewall.py`
- `mcp/tests/test_verify.py`
