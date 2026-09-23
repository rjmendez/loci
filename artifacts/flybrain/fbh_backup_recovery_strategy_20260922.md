# FlyBrain harness backup/recovery strategy artifact (2026-09-22)

## Decision

Adopt a metadata-only backup strategy for FlyBrain harness recovery:

- back up manifests, config snapshots, and replay state;
- exclude bulk source datasets and rebuildable large blobs;
- enforce variable-driven roots only;
- run event-driven + daily backups with quarterly restore drills.

## Path contract

- Required root: `LOCI_FLYBRAIN_STORAGE_ROOT`
- Backup root: `LOCI_FLYBRAIN_BACKUP_ROOT` (defaults to `<LOCI_FLYBRAIN_STORAGE_ROOT>\backups`)
- Drill log root: `LOCI_FLYBRAIN_BACKUP_DRILL_LOG_ROOT` (defaults to `<LOCI_FLYBRAIN_STORAGE_ROOT>\logs\restore-drills`)

No hardcoded absolute machine paths are permitted.

## Backup scope

Back up:

- `fbh-manifest/v1` manifests and lineage links
- resolved config snapshots used by harness runs
- run/replay ledger state and backup catalog
- integrity metadata (bundle-level and file-level SHA-256)

Do not back up:

- bulk source dataset blobs
- full graph DB payloads and snapshot content blobs
- cache/scratch/tmp artifacts

## Cadence and retention

- Event-driven backup after each manifest publish/refresh decision
- Daily consolidated metadata backup
- Weekly checksum verification
- Quarterly full restore drill

Retention windows:

- 14 daily
- 8 weekly
- 6 monthly

## Recovery targets

- Metadata RPO: ≤ 24h
- Metadata RTO: ≤ 2h

## Operator constraints

- Validate and contain all paths under allowlisted roots.
- Ignore incomplete bundles (no completion marker); restore from latest complete valid bundle.
- Freeze new manifest publishes if latest two bundles fail integrity verification.

## Normative doc

- `docs/FLYBRAIN_HARNESS_BACKUP_RECOVERY_STRATEGY.md`
