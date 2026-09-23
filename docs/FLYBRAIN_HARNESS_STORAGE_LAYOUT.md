# FlyBrain Harness Storage Layout

## Purpose

Define a durable, config-driven storage layout for FlyBrain harness state under one canonical root, with no hardcoded machine paths.

## Canonical root

- Root env/config input: `LOCI_FLYBRAIN_STORAGE_ROOT`
- All FlyBrain harness filesystem writes must stay at this root or below it.
- Root validation and write boundaries are governed by [FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md](./FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md).

## Durable root structure

```text
<LOCI_FLYBRAIN_STORAGE_ROOT>\
  graph\
  snapshots\
  cache\
  backups\
  logs\
```

## Ownership and lifecycle per subtree

### `graph\`
- **Owner:** graph/index writers in the FlyBrain harness pipeline.
- **Content:** active graph databases, materialized indexes, and current run graph artifacts.
- **Lifecycle:** long-lived, mutable working state; replace only through validated graph jobs.

### `snapshots\`
- **Owner:** snapshot/export tasks.
- **Content:** point-in-time graph snapshots for replay/debug/comparison.
- **Lifecycle:** append-mostly; prune by retention policy when superseded.

### `cache\`
- **Owner:** fetch/enrichment stages and query acceleration paths.
- **Content:** rebuildable intermediates, response caches, temporary acceleration indexes.
- **Lifecycle:** disposable; first target for cleanup when budget thresholds are reached.

### `backups\`
- **Owner:** backup/restore lane.
- **Content:** versioned backup bundles with metadata/manifests.
- **Lifecycle:** retention-managed, immutable after write; purged by backup policy windows.
- **Policy anchor:** [FLYBRAIN_HARNESS_BACKUP_RECOVERY_STRATEGY.md](./FLYBRAIN_HARNESS_BACKUP_RECOVERY_STRATEGY.md).

### `logs\`
- **Owner:** harness runtime, job orchestrators, path safety guard logging.
- **Content:** structured run logs, guard decisions, and operational diagnostics.
- **Lifecycle:** rotated and retained by log policy; must remain available for incident replay.

## Safety and guardrail alignment

- Use `LOCI_FLYBRAIN_STORAGE_ROOT` as the single source of truth for all layout paths.
- Apply path validation before mkdir/write/delete/move operations.
- Keep destructive operations scoped to explicit descendants only (never root).
- Respect storage budget thresholds from [FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md](./FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md) and prioritize pruning `cache\` before durable stores.

## Reference implementation

`mcp/flybrain_harness_storage.py` provides:
- env-driven root resolution (`LOCI_FLYBRAIN_STORAGE_ROOT`)
- root validation (absolute path, local-drive on Windows, no drive-root target, no symlink/reparse traversal)
- deterministic subtree mapping for `graph`, `snapshots`, `cache`, `backups`, `logs`
- optional idempotent directory creation for harness preflight/bootstrap
