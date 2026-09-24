# FlyBrain harness write-path safety artifact (2026-09-22)

## Decision

Adopt an allowlist-only filesystem policy for FlyBrain harness jobs with:

- canonical root: `F:\loci-data\flybrain`
- explicit denylist guards for known photo/media trees
- canonical normalization + containment checks
- symlink/junction (reparse-point) escape prevention
- destructive-operation blocking outside allowlist

## Source of truth

- Policy doc: `docs/FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md`

## Operational intent

- Prevent any harness write/delete/move on unbacked `F:\` photo directories.
- Require shared path guard validation before all filesystem side effects.
- Keep cleanup jobs dry-run capable and root-protected.
