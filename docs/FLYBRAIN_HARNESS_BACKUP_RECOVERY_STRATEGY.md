# FlyBrain Harness Backup and Recovery Strategy

## Purpose

Define the minimal backup/recovery plan for FlyBrain harness metadata and manifests so operators can recover provenance and run state without duplicating large source datasets.

## Scope and non-goals

- **In scope:** manifest/provenance metadata, harness config snapshots, replay/ledger state, and restore instructions.
- **Out of scope:** bulk source blobs, raw connectome dumps, imported graph databases, and other rebuildable large artifacts.

This strategy is metadata-first and intentionally small-footprint to stay within the storage guardrails in [FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md](./FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md).

## Path contract (variable-driven only)

All paths must be derived from configured variables:

- `LOCI_FLYBRAIN_STORAGE_ROOT` (required canonical root)
- `LOCI_FLYBRAIN_BACKUP_ROOT` (optional; default `<LOCI_FLYBRAIN_STORAGE_ROOT>\backups`)
- `LOCI_FLYBRAIN_BACKUP_RETENTION_DAYS` (optional retention tuning)
- `LOCI_FLYBRAIN_BACKUP_DRILL_LOG_ROOT` (optional drill evidence root; default `<LOCI_FLYBRAIN_STORAGE_ROOT>\logs\restore-drills`)

No hardcoded drive letters or absolute machine-specific paths are allowed.

## What to back up

Backup set is limited to metadata/state required to replay harness decisions:

1. **Manifest records**
   - `fbh-manifest/v1` dataset manifests and integrity envelopes.
   - Manifest lineage links (including `refresh.supersedes_manifest_id`).
2. **Harness config snapshots**
   - Effective resolved harness configuration used per run (env-derived settings, refresh cadence, dataset selection scope).
3. **Run/replay state**
   - Run ledger/state index needed to replay or resume orchestration safely.
   - Backup catalog/index for previously created metadata bundles.
4. **Verification metadata**
   - SHA-256 checksum file for every backed-up metadata object.
   - Backup bundle manifest with creation timestamp and source references.

## What not to back up

Do **not** include:

- bulk source dataset files,
- full graph DB stores and snapshot payload blobs,
- temporary caches/scratch/intermediate blobs,
- any artifact class that can be deterministically rebuilt from source + manifests.

Under storage pressure, retain metadata backups and purge rebuildable bulk/caches first.

## Backup cadence

Minimal cadence:

- **Event-driven:** create a metadata backup bundle after each successful manifest publish or refresh decision.
- **Daily checkpoint:** one daily consolidated metadata bundle.
- **Weekly verification:** verify checksum integrity for retained bundles.
- **Quarterly restore drill:** execute full drill and log evidence.

## Retention policy (metadata bundles)

Retention should be compact and guardrail-aligned:

- keep last **14 daily** bundles,
- keep last **8 weekly** bundles,
- keep last **6 monthly** bundles.

Apply retention pruning only inside `LOCI_FLYBRAIN_BACKUP_ROOT` and never on active run outputs. Metadata backups should remain a small fraction of the effective harness budget; if thresholds in the write-path policy are hit, prune oldest metadata bundles only after cache/scratch cleanup.

## Target RPO/RTO

- **RPO target:** ≤ 24 hours for manifest/config/state metadata.
- **RTO target:** ≤ 2 hours to restore metadata services and replay capability from the most recent valid bundle.

These targets are for harness metadata continuity, not bulk dataset re-download/import timelines.

## Restore drill steps

1. Resolve and validate roots from `LOCI_FLYBRAIN_STORAGE_ROOT` and `LOCI_FLYBRAIN_BACKUP_ROOT`.
2. Select latest valid backup bundle (completed marker present, checksums file present).
3. Verify bundle integrity (all SHA-256 checks pass).
4. Restore manifest records and config/state metadata to staging under the allowlisted root.
5. Run schema/provenance validation (`fbh-manifest/v1` required fields and lineage links).
6. Atomically promote restored metadata into active metadata locations.
7. Execute replay smoke test (load manifest chain, run no-write replay).
8. Record drill result, duration, gaps, and corrective actions under `LOCI_FLYBRAIN_BACKUP_DRILL_LOG_ROOT`.

## Corruption and replay scenarios

### Scenario A: Manifest corruption/hash mismatch

- Quarantine the corrupt manifest.
- Restore the newest prior valid manifest chain from backup.
- Re-run manifest validation and mark the incident in operator logs.

### Scenario B: Partial/incomplete backup write

- Ignore bundles without completion marker.
- Fall back to previous completed bundle.
- Open operator action item to repair bundle writer atomicity.

### Scenario C: Accidental metadata deletion

- Restore only missing metadata subtree from latest valid backup.
- Verify lineage continuity and replay smoke test before resuming writes.

### Scenario D: Replay divergence after restore

- Compare restored run ledger/config snapshot with expected run fingerprint.
- If mismatch persists, pin to last known-good bundle and escalate for manual replay review.

## Operator guidance

- Treat metadata backup failures as **high priority** because they break provenance and reproducibility guarantees.
- Never recover by copying unknown files from outside allowlisted roots.
- Do not restore bulk blobs through metadata backup lane.
- If both latest and previous bundle fail integrity, freeze new manifest publishes and escalate to incident handling.

## Cross-reference

- [FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md](./FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md)
- [FLYBRAIN_HARNESS_STORAGE_LAYOUT.md](./FLYBRAIN_HARNESS_STORAGE_LAYOUT.md)
- [FLYBRAIN_HARNESS_MANIFEST_PROVENANCE_SCHEMA.md](./FLYBRAIN_HARNESS_MANIFEST_PROVENANCE_SCHEMA.md)
