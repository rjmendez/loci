# FlyBrain index and query adapters artifact (2026-09-22)

## Decision

Use a registry-based adapter layer that routes FlyBrain reads through dataset-pinned local snapshots before any remote fallback, with explicit provenance and dataset-boundary checks.

## Scope alignment

- Canonical storage root: `LOCI_FLYBRAIN_STORAGE_ROOT`
- Primary local dataset pins: `hb` (`neuprint_JRC_Hemibrain_1point2point1`), `fw` (`flywire783`)
- Local-first routing with explicit remote-fallback metadata when required
- No hardcoded machine-specific paths or `F:\` assumptions

## Required adapters

- `HbGraphLocalAdapter`
- `FwMetadataLocalAdapter`
- `RemoteFallbackAdapter`

## Routing rules

1. Evaluate local adapter match by dataset symbol + version pin.
2. Confirm query support against a capability manifest.
3. Require provenance for any fallback to remote/live source.
4. Block scope violations or unsupported cross-dataset generalization.

## Provenance and failure policy

- record dataset symbol, version, query kind, source, and scope tags
- stop on malformed manifests, invalid paths, or unsupported dataset/version combinations
- allow retry only for transient remote or staging failures

## References

- `docs/FLYBRAIN_INDEX_QUERY_ADAPTERS_PLAN.md`
- `docs/FLYBRAIN_FETCH_SYNC_PIPELINES_PLAN.md`
- `docs/FLYBRAIN_DATASET_PROVENANCE_MATRIX.md`
