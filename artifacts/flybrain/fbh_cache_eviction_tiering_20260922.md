# FlyBrain cache eviction and tiering artifact (2026-09-22)

## Decision

Adopt a tiered local cache policy that keeps the active graph and pinned snapshots in hot/warm tiers while treating `cache\` as the first cleanup target.

## Tier policy

- **Hot**: active graph + current manifest + live query index
- **Warm**: retained snapshot artifacts and prior manifests
- **Cold**: rebuildable partials and temporary import state

## Cleanup sequence

1. evict partial download/cache files
2. evict unreferenced import staging
3. remove stale candidate graph directories
4. retire old warm-tier manifests only after explicit supersession
5. never touch active graph or pinned datasets without a manifest reference check

## Budget behavior

- soft cap -> warn + begin cleanup
- hard cap -> block writes
- all cleanup remains under `LOCI_FLYBRAIN_STORAGE_ROOT`

## References

- `docs/FLYBRAIN_CACHE_EVICTION_TIERING_PLAN.md`
- `docs/FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md`
- `docs/FLYBRAIN_HARNESS_STORAGE_LAYOUT.md`
