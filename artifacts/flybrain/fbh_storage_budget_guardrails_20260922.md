# FlyBrain configured storage root guardrail inventory

## Summary

This harness should treat the configured shared storage root as a safety-sensitive working drive, not a fully available scratch partition. The repo already enforces a canonical allowlist under a configurable data root (for example `$HARNESS_DATA_ROOT` or a workstation-specific path such as `F:\loci-data\flybrain`), and the storage policy should apply the same safety-first posture to budget enforcement.

## Baseline assumption

- Available free space on the harness storage root is approximately 300 GB in the example shared-drive layout.
- The storage root is shared with unbacked user photo/media trees in the example layout.
- The harness must keep at least 100 GB as reserved headroom for photo growth, OS/app churn, and other non-FlyBrain activity.

## Effective FlyBrain budget

- Reserved headroom: 100 GB
- Effective harness budget: 200 GB
- Hard cap: 200 GB absolute maximum under the configured data root, for example `F:\loci-data\flybrain`
- Soft cap: 160 GB steady-state target for active runs
- Warning thresholds: 150 GB watch, 180 GB critical, 200 GB hard stop

## Pressure responses

1. At 150 GB: review retention, prune stale caches, and stop new large downloads.
2. At 160 GB: switch the harness to read-only mode for nonessential jobs; keep only active datasets and current run metadata.
3. At 180 GB: freeze nonessential writes, purge `tmp`, `staging`, and scratch directories, and require explicit override to continue.
4. At 200 GB or above: block new writes, keep only the current job's minimal working set, and clean up ephemeral artifacts before resuming.

## Safety-first rule

The configured storage root is not a disposable bulk-storage drive for FlyBrain. The harness should prefer under-allocation and aggressive cleanup over long-lived growth on a shared volume that may be reclaimed by unbacked photo content.


