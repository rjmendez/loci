# Memory Decay Spec

This spec defines how Loci decays weak, stale, or low-support memory items so old noise does not crowd out newer verified knowledge.

## Scope

The authoritative implementation is `mlops/memory/decay.py`.

This module applies Weibull temporal decay to Mnemosyne `working_memory` rows using a base importance snapshot and the row's `created_at` timestamp.

## Core rule

The retention factor is:

`retention = exp(-((age_days / lambda_days) ** k))`

with default values:

- `lambda_days = 30.0`
- `k = 0.8`

The output importance is computed from the row's baseline value, never from the already-decayed live value:

`decayed_importance = max(min_importance, base_importance * retention)`

## Why the baseline matters

The implementation explicitly stores a `base_importance` column and seeds it once if absent. It is critical because re-decaying a row from its already-decayed value compounds the effect and destroys the original authored signal.

This is the primary invariant:

- `importance` may decay over time
- `base_importance` stays fixed at the first-authoring value
- re-running decay must not progressively erase the memory by compounding decay over itself

## Required invariants

### 1. No self-compounding decay

The code must always derive decay from `base_importance`, not from the current `importance` field. If a row has no baseline, a single seeding step creates it before decay is applied.

### 2. Null-safe and fail-open behavior

- rows with `importance IS NULL` are ignored
- bad timestamps are skipped rather than crashing the run
- `created_at` values with no timezone are treated as UTC
- missing DB or missing table returns a structured error result instead of raising

### 3. Grounding visibility floor

The module keeps a floor at `DEFAULT_MIN_IMPORTANCE` and also hides rows below `GROUNDING_MIN_IMPORTANCE` from recall-by-grounding. This ensures a memory system can age out weak facts without making retrieval brittle.

## Trigger conditions

A row should decay when any of these are true:

- it has aged beyond the retention horizon
- it is weakly supported or only weakly weighted
- it has not been reinforced by repeated successful verification
- it has become stale relative to newer, stronger evidence

Decay is a memory hygiene step. It is not an evidence invalidation step. Contradictions and retractions still need separate review and retraction flows.

## Required outputs

`apply_decay()` returns a dict with at least:

- `n_rows`
- `n_decayed`
- `mean_retention`
- `min_retention`
- `n_grounding_visible_before`
- `n_grounding_visible_after`
- `lambda_days`
- `k`
- `dry_run`

## Failure behavior

The implementation is intentionally fail-open and deterministic:

- if the DB is missing, return an error descriptor
- if the table is missing, return an error descriptor
- if timestamps are malformed, skip the row
- if a dry run is requested, do not mutate the DB

## Acceptance criteria

A decay cycle passes when:

- retention is monotonic with age
- retention is never applied against already-decayed values
- `base_importance` is seeded exactly once
- weak memories age below the grounding visibility floor without crashing the system
- `dry_run` leaves the dataset unchanged

## Relevance to existing code

This spec matches the current implementation in:

- `mlops/memory/decay.py`
- `mlops/memory/tests/test_memory.py`
- `mcp/server.py` (`memory_health`, `memory_surface`, and retrieval filtering behavior)
