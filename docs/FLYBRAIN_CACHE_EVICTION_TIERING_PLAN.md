# FlyBrain cache eviction and tiering plan

## Purpose

Define a retention-safe cache strategy for local FlyBrain harness data so the workspace stays under budget while preserving the provenance and replayability needed for Loci work.

This plan complements:

- `docs/FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md`
- `docs/FLYBRAIN_HARNESS_STORAGE_LAYOUT.md`
- `docs/FLYBRAIN_FETCH_SYNC_PIPELINES_PLAN.md`
- `docs/FLYBRAIN_ARTIFACT_INTEGRITY_QUARANTINE_POLICY.md`

## Core rule

The storage root remains the only writable boundary:

- `LOCI_FLYBRAIN_STORAGE_ROOT`

No operation may escape this root. No hardcoded machine-specific drive path is allowed in code, docs, or operational text.

## Storage tiers

### Hot tier

Files that are currently active and must remain immediately queryable.

Examples:

- active graph promotion paths under `graph\`
- pinned dataset snapshot manifests
- current query-index metadata

Hot-tier retention rule:

- keep only the current promoted version and the immediately required metadata set
- never keep multiple conflicting promoted versions at once unless explicitly pinned for comparison

### Warm tier

Files that are valid but not actively in use.

Examples:

- older snapshot manifests
- prior integrity checks
- pre-promotion candidate graph stores awaiting verification

Warm-tier retention rule:

- retain for replay and forensic validation within a fixed retention window
- prune only after the corresponding manifest is superseded or archived

### Cold tier

Large or rebuildable intermediates that can be re-created from source provenance.

Examples:

- partial downloads
- temporary import caches
- unpacked stage artifacts no longer used in the active policy cycle

Cold-tier retention rule:

- the first cleanup target
- safe to evict after successful manifest sealing and a fresh integrity check

## Eviction order

When the harness crosses its soft budget or enters a pressure state, eviction must run in this order:

1. `cache\downloads\` partial files
2. `cache\imports\` staging data not sealed into a manifest
3. old candidate graph directories not promoted
4. stale warm-tier manifests older than the current pin
5. only then, if still needed, prune older archived snapshots according to policy

Never evict:

- the currently active pinned manifest
- the active promoted graph path
- any dataset whose checksum manifest is still referenced by a current query plan

## Budget policy

Use configurable policy values instead of fixed thresholds:

- `LOCI_FLYBRAIN_BUDGET_SOFT_CAP_GB`
- `LOCI_FLYBRAIN_BUDGET_HARD_CAP_GB`

Recommended behavior:

- soft cap: warn and begin cleanup
- hard cap: block write operations and require operator action

The system must refuse to write outside the configured root even if the budget would otherwise allow it.

## Tiering and manifest coupling

Before any artifact is evicted, the harness must confirm:

- whether the file is referenced by a current manifest
- whether it is necessary for deterministic replay
- whether a replacement artifact already exists in a newer or safer tier

If a file is still needed for a manifest or replay, it remains in place even if it is large.

## Cleanup safety boundaries

Allowed cleanup actions:

- prune descendants under `cache\`
- remove temporary/partial files from preflight or failed acquisitions
- delete only unreferenced candidate writes with a corresponding manifest trace

Blocked cleanup actions:

- destructive cleanup of active `graph\` trees
- deletion of pinned snapshots without explicit version bump or retention approval
- cleanup of logs or manifests that are still referenced by a valid run journal

## Query impact and integrity guardrails

Every tier change should update the corresponding manifest metadata:

- tier assignment
- retention timestamp
- eviction reason
- integrity hash or state pointer
- provenance of the replacement or retention decision

This keeps the system honest: a file is not silently removed just because it is old.

## Operational recommendations

1. Keep the active graph and active snapshots in the hot/warm tier for the pinned dataset.
2. Treat `cache\` as the eviction buffer and never the durable source of truth.
3. Use the manifest as the authority on whether an artifact is still needed.
4. Keep a small retention log in `logs\` for every eviction or promotion decision.

## Implementation checklist

- define budget pressure states (`watch`, `soft`, `critical`, `hard-stop`)
- add eviction planner keyed by tier and manifest reference count
- enforce dry-run before execute mode for any deletion
- validate `cache\` cleanup against the manifest index
- record a retention journal entry for every cleanup event

## Recommended next implementation step

Add the budget and eviction planner behind the shared storage helper, with manifest-driven cleanup decisions before any graph or snapshot deletion.

---

Status: ready for implementation as the second remaining FlyBrain harness hardening pass.
