# Reasoning Loop Provenance and Adjudication

This spec defines how Loci should move evidence, assumptions, derived claims, contradictions, and confidence through the reasoning loop without conflating them.

It extends the existing patterns already present in the repo:

- `mcp/provenance_firewall.py` separates provenance authority from storage tier.
- `mcp/memcheck/checks/provenance.py` enforces “observed must cite a receipt”.
- `mcp/memcheck/checks/contradiction.py` treats contradictions as advisory signals, not automatic overwrites.
- `mcp/inv_store.py` defines the numeric confidence, lifecycle resolution, and `derived_from` chain model.
- `docs/API_MEMORY.md` and `docs/API_INVESTIGATION.md` already distinguish evidence provenance, memory tier, and lifecycle.

The rule is simple: evidence provenance, memory storage tier, and belief confidence are three different axes and must remain separate.

## 1) Core model: do not conflate the axes

Every finding must carry enough metadata to answer three different questions independently:

1. Where did this claim come from?
   - `finding_type`: `observed` / `inferred` / `assumed` / `gap` / `procedure`
   - `evidence_provenance_tier`: `human_authored` / `tool_verified` / `deterministic_derived` / `model_asserted`
2. Where is it stored and how searchable is it?
   - storage tier: `hot` / `warm` / `cold`
3. How strongly do we believe it right now?
   - `confidence`: `low` / `medium` / `high`
   - `numeric_confidence`: `0.0..1.0`
   - `resolution`: `open` / `fixed` / `intentional` / `wontfix` / `superseded`

These are not interchangeable:

- `evidence_provenance_tier` answers “what kind of evidence supports this?”
- `numeric_confidence` answers “how strongly do we presently believe it?”
- `resolution` answers “is this still active or has it been handled?”
- `storage tier` answers “how accessible is this record in memory?”

A `model_asserted` finding is still a low-trust assertion unless independent evidence exists. A `cold` record may still be highly credible. A `high` confidence claim may still be a weakly supported assumption if it lacks a receipt.

## 2) Evidence-provenance loop

The reasoning loop must preserve the chain from source to conclusion.

### 2.1 Finding categories

Observed
- Direct claim from a human, tool, or dataset snapshot.
- Must include source details and, when practical, a receipt or direct evidence block.
- Must be traceable to a tool response or explicit human record.
- Rule: `observed` without a receipt is a `gap` or an unsupported claim, not a verified observation.

Inferred
- A conclusion derived from one or more upstream findings.
- Must list `derived_from` references to the source findings.
- Must carry its own provenance tier, usually `deterministic_derived` or `model_asserted` depending on method.

Assumed
- Working hypothesis used to direct the next probe.
- Must be explicitly marked as `assumed` and should never be treated as evidence.
- It may be promoted to `inferred` only after independent support appears.

Gap
- Missing fact or required check.
- Indicates an open question, not a claim.

Procedure
- Runbook or reusable action step.
- May reference evidence and observations, but remains a procedural artifact rather than a claim about the world.

### 2.2 Required metadata for all non-gap claims

A claim that is intended to influence downstream reasoning should carry these fields:

- `finding_type`
- `text`
- `source`
- `confidence` and `numeric_confidence`
- `evidence_provenance_tier`
- `derived_from` if it is not root evidence
- `valid_from` / `valid_until` when temporal bounds matter
- `code_refs` when a code or file claim is involved

### 2.3 Provenance flow

1. Root evidence is captured as `observed`.
   - Human-authored evidence: field notes, reproduced outcomes, direct citations.
   - Tool-verified evidence: receipts from tool output, log lines, tests, file hashes, or API responses.
   - Deterministic-derived evidence: results produced by parsing, diffing, or deterministic transforms over observed data.

2. Intermediate reasoning creates `inferred` findings.
   - Each derived claim must point back to upstream findings via `derived_from`.
   - The derived claim’s provenance tier is not “more trusted” merely because it is new; it depends on the source evidence.

3. Hypotheses are kept as `assumed` until evidence accumulates.
   - They may steer search but they must not satisfy the evidence firewall for a final claim.

4. Final conclusions are validated against the chain.
   - `investigation_finding_provenance()` walks `derived_from` backward to root evidence.
   - `memory_retract()` follows contamination through the lineage when a root fact is invalidated.

5. Evidence firewall enforcement
   - `model_asserted` claims are allowed only when at least one independent evidence item exists in the support set.
   - Independent evidence means `human_authored`, `tool_verified`, or `deterministic_derived` — not another `model_asserted` claim.
   - This mirrors `mcp/provenance_firewall.py` and keeps the model from “verifying itself with itself.”

### 2.4 Operational invariant

The reasoning loop must preserve provenance in a way that allows later reconstruction:

- root evidence is not replaced by summaries
- derived claims cite their sources
- assumptions remain labeled as assumptions
- contradiction verdicts are stored as separate evidence objects, not hidden in the source text

## 3) Contradiction adjudication

Contradictions are not silently “resolved” by whichever output arrives last. They are adjudicated through an explicit ruleset.

### 3.1 Contradiction types

- Semantic contradiction: same subject and incompatible claims.
- Negation mismatch: “A is true” vs “A is not true”.
- Temporal contradiction: same entity behaves differently across time windows without a valid `valid_until`/`valid_from` boundary.
- Context contradiction: both claims are true in different contexts but not mutually exclusive.
- Value contradiction: one value displaces another without a qualifying explanation.

### 3.2 Detection rules

The contradiction pass should only trigger when all of the following are true:

1. same subject or same feature cluster is involved
2. overlap in semantics or tokens is high enough to suggest same claim
3. polarity or value is incompatible, or one explicitly negates the other
4. neither record is clearly marked as a stale assumption or unresolved gap

This follows the repo’s existing heuristics in `mcp/memcheck/checks/contradiction.py` and the broader `memory_self_check` advisory flow.

### 3.3 Adjudication order

When two findings conflict, the system should rank them in this order:

1. Direct observation and tool-verified evidence outrank model assertions.
2. More specific and better-scoped evidence outranks broader, weaker evidence.
3. Claims with explicit time bounds or valid-range metadata outrank unbounded claims in the same window.
4. Later records do not automatically win; they only win if they are better supported or explicitly supersede older ones.
5. Assumptions without support are provisional and never adjudicate final truth.

### 3.4 Resolution policy

A contradiction should result in one of the following:

- `a_wins` / `b_wins`: one record is chosen as the current truth
- `both_valid`: the findings are true under different contexts or scopes
- `false_positive`: the apparent contradiction was an artifact of weak matching
- `unresolved`: both remain open pending new evidence

The adjudication outcome must be stored as a separate verdict or update record. It must not silently rewrite the original fact. The original evidence remains in the log; the latest decision is appended as an overlay.

### 3.5 Protection rule for established findings

If a finding is already high-confidence and observed or tool-verified, a contradictory later claim should be treated as provisional rather than immediately authoritative. The repo’s current EWC-style handling in `mcp/memcheck/checks/contradiction.py` is the correct pattern: established facts are resistant to weak contradiction and require corroboration before they are overturned.

### 3.6 Provenance under contradiction

Contradiction adjudication must never erase the lineage chain. The result of the conflict should still answer:

- which findings contradicted each other
- which one was more reliable at the time
- what evidence was used to decide
- whether the winner was backed by independent evidence or only a newer assumption

This prevents “last word wins” from quietly destroying the evidence trail.

## 4) Confidence calibration

Confidence is not the same as provenance and not the same as factual certainty. It is a current belief estimate after observing support, contradictions, and chain quality.

### 4.1 Representation

Every claim should carry:

- `confidence`: qualitative label (`low`, `medium`, `high`)
- `numeric_confidence`: scalar in `[0, 1]`
- optional `evidence_provenance_tier`
- optional `valid_from`, `valid_until`

The label is for human readability; the numeric value is used for downstream aggregation and derived chains.

### 4.2 Base calibration by evidence quality

A recommended default policy:

| Provenance tier | Base weight |
|---|---:|
| `human_authored` | 1.00 |
| `tool_verified` | 0.90 |
| `deterministic_derived` | 0.80 |
| `model_asserted` | 0.55 |

Type weighting should be separate:

| Finding type | Base weight |
|---|---:|
| `observed` | 1.00 |
| `inferred` | 0.80 |
| `assumed` | 0.40 |
| `gap` | 0.20 |

These are not truth values; they are support weights for credibility. They should be combined with contradiction and support-depth penalties.

### 4.3 Derived chains

For a derived claim, confidence should be calculated from its upstream chain rather than treated as an independent flat score.

Recommended rule:

- compute the product of upstream `numeric_confidence` values
- apply a modest chain penalty for each derived step
- cap the result with a maximum ceiling for unsupported or assumption-heavy chains

Pseudo-policy:

```
base = product(upstream_confidences)
derived_penalty = 0.9 ** (num_derived_steps - 1)
confidence = min(0.99, base * derived_penalty)
```

If a derived chain is built primarily on assumptions, the result should be treated as provisional and capped lower.

### 4.4 Contradiction and support penalties

Confidence should decrease when evidence is weak or contradicted.

Recommended adjustments:

- unresolved contradiction: subtract `0.15` to `0.30`
- one independent supporting source: add `+0.05`
- multiple independent supporting sources across different sources: add `+0.10` up to a ceiling
- assumption-only chain: clamp to `<= 0.60` until verified
- model-only support without independent evidence: clamp to `<= 0.45`
- expired or stale valid window: confidence drops to 0 for future reasoning

This policy matches the repo’s guarded reasoning decisions in `mcp/provenance_firewall.py`, `verify.py`, and the contradiction checks and is compatible with `investigation_pre_answer_check`.

### 4.5 Final decision policy

The system should not declare a final answer when confidence is based only on model assertion or unresolved contradiction.

A claim is ready to act upon when:

- it is supported by at least one independent evidence source, or
- it is explicitly marked as `assumed` and not used as final evidence, or
- it has been explicitly adjudicated and remains `open` while under investigation

A claim is not ready when:

- it is `model_asserted` and its support is only other `model_asserted` findings
- it is contradicted by better-supported findings without a resolution
- it depends on a stale or unbounded assumption chain

## 5) Reasoning-loop state machine

The reasoning loop should pass through these states cleanly:

1. `candidate` — raw proposition under consideration
2. `observed` / `assumed` / `gap` — first classification
3. `supported` — has direct or independent evidence
4. `derived` — connected through `derived_from`
5. `contradicted` — flagged by contradiction check
6. `adjudicated` — resolved as `a_wins`, `b_wins`, `both_valid`, `false_positive`, or `unresolved`
7. `final` — accepted conclusion ready for downstream use

The important property is that a claim cannot move from `candidate` to `final` without staying traceable to its evidence and contradiction history.

## 6) Implementation-specific rules for Loci

The current implementation already has the right primitives; the reasoning loop should enforce these rules:

- `investigation_store()` stores `finding_type`, `source`, `confidence`, `derived_from`, and `evidence_provenance_tier`.
- `memory_self_check()` runs provenance + contradiction audits as advisory checks.
- `memory_retract()` follows contaminated lineage and removes contaminated conclusions from recall.
- `investigation_finding_provenance()` reconstructs the chain back to root evidence.
- `verify_finding()` and `investigation_pre_answer_check()` keep model assertions from being accepted as self-validated evidence.

In short: evidence is stored, assumptions are labeled, derived claims cite ancestors, contradictions create a verdict, and confidence is a calibrated estimate computed after those steps — never before.

## 7) Summary

The safe reasoning model is:

- observation first
- assumptions explicit
- derived claims traceable
- contradictions adjudicated, not overwritten
- confidence calibrated only after the above

This prevents the common failure mode where a model’s latest guess, a memory hit, or a derived inference gets mistaken for evidence. Loci should never confuse “what we think now,” “what we derived,” and “what we actually observed.”
