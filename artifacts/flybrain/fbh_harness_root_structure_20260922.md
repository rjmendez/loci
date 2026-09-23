# FlyBrain harness root structure artifact (2026-09-22)

## Decision

Define a durable, config-driven harness root layout under:

- `LOCI_FLYBRAIN_STORAGE_ROOT`

with fixed top-level subtrees:

- `graph\`
- `snapshots\`
- `cache\`
- `backups\`
- `logs\`

## Ownership and lifecycle summary

- `graph\`: active durable graph/index state, owned by graph writers.
- `snapshots\`: point-in-time exports, append-mostly, retention-pruned.
- `cache\`: rebuildable ephemeral intermediates, first cleanup target.
- `backups\`: versioned restore bundles, immutable after write, retention-managed.
- `logs\`: run/guard diagnostics, rotated but retained for replay.

## Safety and guardrail alignment

- Canonical root is variable-driven (`LOCI_FLYBRAIN_STORAGE_ROOT`), never drive-hardcoded.
- Layout stays inside allowlisted root from `docs/FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md`.
- Destructive operations remain descendant-scoped and root-protected.
- Storage-budget pressure should prune `cache\` before durable state trees.

## Implementation anchors

- `mcp/flybrain_harness_storage.py` (path resolution + validation + layout mapping)
- `mcp/tests/test_flybrain_harness_storage.py` (layout and validation coverage)
- `docs/FLYBRAIN_HARNESS_STORAGE_LAYOUT.md` (operator-facing structure/lifecycle doc)
