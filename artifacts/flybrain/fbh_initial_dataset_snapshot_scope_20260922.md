# FlyBrain harness initial dataset/snapshot scope (2026-09-22)

## Decision

Adopt a phase-1 local scope with:

1. Hemibrain local graph snapshot (`hb`, `neuprint_JRC_Hemibrain_1point2point1`)
2. FlyWire metadata snapshot (`fw`, `flywire783`) without full local adjacency graph

This is the smallest scope that supports deterministic local graph-first execution plus cross-dataset provenance checks.

## Version pin strategy

- Store each dataset under a version-bearing folder key.
- Record per-file checksums and source URI in a snapshot manifest.
- Treat version changes as explicit PR-reviewed pin bumps, not background drift.

## Recommended scope table

| dataset | pinned version | refresh cadence | size class | intended use | path contract |
|---|---|---|---|---|---|
| Hemibrain graph (`hb`) | `neuprint_JRC_Hemibrain_1point2point1` | Quarterly integrity re-check | M (tens of GB) | Primary local graph queries and reproducible harness replay | `$LOCI_FLYBRAIN_STORAGE_ROOT\datasets\hb\neuprint_JRC_Hemibrain_1point2point1\graph\` |
| FlyWire metadata (`fw`) | `flywire783` | Monthly metadata check; bump only by explicit pin change | S-M (single-digit to low tens of GB) | Entity normalization, annotation lookup, provenance envelopes | `$LOCI_FLYBRAIN_STORAGE_ROOT\datasets\fw\flywire783\metadata\` |

## Explicit phase-1 exclusions

- No full local FlyWire adjacency/chunkedgraph mirror.
- No additional large dataset mirrors (`mc`, `mv`, `BANC`) until storage telemetry remains below soft cap for at least one full refresh cycle.

## Guardrail alignment

- Respect `LOCI_FLYBRAIN_STORAGE_ROOT` as the only writable root for these snapshots.
- Keep total FlyBrain storage below soft cap targets from `artifacts/flybrain/fbh_storage_budget_guardrails_20260922.md`.
- If threshold pressure is reached, prefer pruning ephemeral/staging data over expanding snapshot scope.

## Provenance contract required at ingest

Each snapshot ingest must persist:

- `dataset_symbol`
- `version_id`
- `source_uri`
- `retrieved_at`
- `file_checksums`
- `refresh_decision` (no-change, patch, major-bump)

This keeps all claims replayable and dataset-scoped.
