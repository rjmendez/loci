# Memory Consolidation Spec

This spec defines how Loci turns a burst of working-memory fragments into durable, recallable memory without silently upgrading weak or unverified claims.

## Scope

This is the policy for the Mnemosyne sleep/consolidation pass surfaced by `memory_consolidate()` in `mcp/server.py` and the SQLite-backed memory logic under `mlops/memory/`.

The consolidation pass is intentionally separate from:

- storage-tier promotion (`memory_promote` / `memory_demote`)
- evidence provenance (`evidence_provenance_tier`)
- verification gating (`verify_finding`)
- retraction and quarantine (`memory_retract` / `memory_restore`)

Consolidation is a retention and summarization step. It is not a trust gate.

## Canonical behavior

The current code path does the following:

1. Loads the Mnemosyne memory bank state.
2. Merges old `working_memory` rows into `episodic_memory`.
3. Runs a best-effort causal inference pass when the most recent investigation has enough findings.
4. Emits a fail-open summary even if the optional causal or quality-audit steps fail.

The tool contract is intentionally tolerant: `memory_consolidate(dry_run=False)` may still return a result even when the optional summarization or inference layer cannot run.

## Lifecycle invariants

### 1. Consolidation is not promotion

A finding may be merged into episodic memory without becoming a trusted or hot memory object. The storage tier and the mnemonic bank are intentionally distinct.

- `memory_consolidate` manages Mnemosyne consolidation only.
- `memory_promote` changes `hot` / `warm` / `cold` storage tier only.
- Verified claims are promoted only via the validated-promotion policy; they are not auto-upgraded by consolidation alone.

### 2. High-confidence verified claims can be re-encountered and stabilized

When a claim is repeatedly verified, the system should keep the strongest version rather than the latest weak assertion. Repeated successful verification contributes to stabilization but does not bypass provenance and contradiction checks.

Expected behavior:

- repeated `confirmed` verification raises confidence and reduces volatility
- contradictory or stale findings remain visible as flagged, not silently merged into durable memory
- a stable, verified finding may be eligible for warm/hot promotion once the validated promotion gate passes

### 3. Consolidation is append-only and fail-open

The current implementation does not block the full memory run when optional quality audits fail. All non-critical steps are best-effort. This preserves service availability while still recording what was done.

## Required policy

Any implementation that reuses this path must follow these rules:

- do not treat consolidation as proof of truth
- do not elevate a finding to hot storage solely because it was seen multiple times
- do not rewrite provenance or degrade a finding silently during consolidation
- if a post-consolidation audit finds contradictions or unsupported evidence, trigger decay, demotion, or retraction instead of merge-acceptance

## Required outputs

`memory_consolidate` must return a JSON object including at least:

- row counts and merge statistics
- whether the run was `dry_run`
- the causal-edge count when available
- optional `sleep_like_consolidation` or `consolidation_quality_audit` reports when those steps ran

## Failure handling

The implementation must fail open:

- if quality audit fails, retain the consolidation result
- if causal inference fails, keep the merge result and record the failure in metadata
- if a bank is missing or the DB is unavailable, return a well-formed degraded status instead of aborting the tool call

## Acceptance criteria

A consolidation pass is accepted when all of the following hold:

- old working-memory entries are merged into episodic memory without mutating claim provenance
- repeated verified findings are preserved preferentially, not overwritten by noisy weaker copies
- contradiction or retraction signals are not ignored
- tool failures are non-blocking and reported as degraded rather than fatal

## Relevance to existing code

This policy matches the implementation in:

- `mcp/server.py` — `memory_consolidate()`
- `mcp/consolidation_quality_audit.py`
- `mlops/memory/decay.py` — temporal importance decay
- `mcp/provenance_firewall.py` — evidence provenance separation
