# FlyBrain Harness Write-Path Safety Policy

## Purpose

Prevent FlyBrain harness scripts/jobs from reading, writing, moving, or deleting data outside a configured storage root by enforcing an allowlist-only filesystem boundary.

## Policy scope

This policy applies to every FlyBrain harness path input, including:

- CLI arguments
- environment variables
- config files
- generated/staged output paths
- cleanup/delete targets

No script may bypass this policy with direct `os`, `pathlib`, `shutil`, or shell filesystem calls.

## Canonical allowlisted root

- **Configured root variable (canonical):** `LOCI_FLYBRAIN_STORAGE_ROOT`
- **Compatibility aliases accepted by runtime:** `HARNESS_DATA_ROOT`, `HARNESS_STORAGE_ROOT`, `LOCI_FLYBRAIN_ALLOWED_ROOT`
- **Example workstation value:** `<configured-storage-root>\loci-data\flybrain`
- Root must be absolute, local-drive, normalized, non-symlink/junction, and configured via an explicit env/config value.

All effective paths must resolve to this root or a descendant of this root.

## Explicit denylist safeguards

Even when a path appears syntactically valid, reject if it matches any deny rule:

- Root-level user media trees:
  - `<storage-drive>\photos`
  - `<storage-drive>\photo`
  - `<storage-drive>\pictures`
  - `<storage-drive>\dcim`
  - `<storage-drive>\camera`
- Common unbacked/photo naming patterns anywhere outside allowlist:
  - `*\photo*`
  - `*\picture*`
  - `*\dcim*`
  - `*\camera*`
- Root-only and drive-only targets:
  - `<storage-drive>\`
  - `<storage-drive>:`

These are **defense-in-depth** checks; allowlist containment remains the primary control.

## Required path validation rules

Every candidate path must pass all checks in this order:

1. **Normalize input**
   - Reject empty, whitespace-only, or non-string paths.
   - Expand environment variables.
   - Convert to absolute path.
   - Canonicalize case, separators, and dot-segments (`.`/`..`).
2. **Resolve canonical paths**
   - Resolve allowlist root and candidate with `Path.resolve(strict=False)`.
   - Compare canonical forms only.
3. **Containment check**
   - Candidate must satisfy `candidate == root` or `candidate.is_relative_to(root)`.
   - Reject if canonical path escapes root through traversal tricks.
4. **Symlink/junction escape prevention**
   - Reject candidate if any existing ancestor from `root` to candidate is:
     - symlink, junction, or other reparse-point hop.
   - On Windows, treat reparse points as untrusted for harness writes.
5. **Denylist/pattern check**
   - Apply explicit denylist rules after normalization.
6. **Operation guard check**
   - Reject destructive ops unless target is strictly inside allowlisted root.
   - Never allow deletion/rename of the root itself.

Any failed check is a hard failure (no fail-open behavior). The root must be configured from an explicit env/config value rather than hardcoded to a single drive letter.

## Manifest integrity-path guardrails (execution gate)

Path safety also applies to manifest-driven integrity checks before any local
query execution:

- `artifact.relative_root` and `integrity.files[*].relative_path` must be safe
  relative paths (no absolute path, drive prefix, or traversal).
- Each integrity file path must resolve under the configured artifact root.
- Duplicate paths after case-folding are rejected.
- Partial artifact names (`.partial`, `.tmp`, `.inprogress`) are rejected.
- Any path escape or missing file is fail-closed (`MANIFEST_INVALID`,
  `MANIFEST_MISSING`, or `PATH_ESCAPE` depending on adapter).

This is enforced in adapter manifest validation, not only in write-path
preflight.

## Destructive-operation restrictions

Destructive operations include delete, recursive delete, move/rename-overwrite, truncate, and cleanup sweeps.

Rules:

- Must pass full validation against allowlisted root.
- Must never target:
  - storage drive root (for example `<configured-storage-root-drive>\`)
  - bare drive selectors (for example `<configured-storage-root-drive>:`)
  - allowlisted root itself (for example `<configured-storage-root>\loci-data\flybrain`)
  - any path outside allowlisted root
- Cleanup jobs must operate on explicit subdirectories (for example, `staging\`, `tmp\`, `runs\`) under the allowlisted root.

## Operational guidance for scripts/jobs

1. **Single gate function**
   - Route all file targets through one shared path-guard utility before filesystem calls.
2. **Validate before side effects**
   - Validate all paths in preflight; abort job if any path fails.
3. **Log every decision**
   - Log normalized path, operation type, and allow/deny result (without sensitive content).
4. **Default dry-run for cleanup**
   - Cleanup/deletion jobs should support dry-run and require explicit execute mode.
5. **No shell escape hatches**
   - Do not call `del`, `rmdir`, `Remove-Item`, or equivalent on unvalidated paths.
6. **Review requirement**
   - Any change to path guard logic or allowed root must be reviewed as a safety-sensitive change.

## Minimal implementation contract

Implement and use a shared guard with this contract:

- `validate_flybrain_path(path, operation) -> normalized_path | error`
- Inputs: raw path + intended operation (`read`, `write`, `mkdir`, `delete`, `move`)
- Output: canonical normalized path only if policy passes
- Error: explicit denial reason (containment, reparse-point, denylist, destructive-op guard)

No harness script/job may perform path-based filesystem I/O without this guard.

## Compliance checklist

- [ ] All FlyBrain harness entrypoints call the shared guard.
- [ ] Allowed root is configured via `LOCI_FLYBRAIN_STORAGE_ROOT` (for example `<configured-storage-root>\loci-data\flybrain`) or stricter approved override.
- [ ] Reparse-point/symlink checks are enabled.
- [ ] Cleanup tasks are dry-run capable and root-protected.
- [ ] Job logs include path safety allow/deny events.

## Required harness root layout

Under `LOCI_FLYBRAIN_STORAGE_ROOT`, keep a stable top-level layout:

- `graph\` — durable active graph/index state
- `snapshots\` — point-in-time exports and replay snapshots
- `cache\` — rebuildable intermediates and acceleration cache
- `backups\` — retention-managed backup bundles
- `logs\` — harness runtime and path-safety logs

See [FLYBRAIN_HARNESS_STORAGE_LAYOUT.md](./FLYBRAIN_HARNESS_STORAGE_LAYOUT.md) for ownership and lifecycle details, and `mcp/flybrain_harness_storage.py` for the env-driven path mapping implementation.

## Storage budget guardrails for the configured storage root

The configured storage root is a shared, safety-sensitive volume. The harness may not assume that all visible free space is disposable work space, because the drive may also contain user photo/media content and other unbacked data. Treat the volume as a constrained shared resource rather than a high-capacity scratch partition.

### Working budget assumptions

- Approximate free space on the configured storage root: 300 GB in the example shared-drive layout
- Reserved headroom for photo/media growth and other non-harness activity: 100 GB
- Effective FlyBrain working budget: 200 GB
- Soft cap for active FlyBrain work: 160 GB
- Hard cap for FlyBrain writes on the configured storage root: 200 GB

### Guardrail thresholds

| Level | Threshold | Action |
|---|---:|---|
| Watch | 150 GB used or above | Pause large downloads; archive stale outputs; review retention. |
| Soft cap | 160 GB used or above | Stop creating new long-lived datasets; keep only active run metadata and current outputs. |
| Critical | 180 GB used or above | Freeze nonessential writes; purge `tmp`, `staging`, and scratch; require explicit override for any continuing work. |
| Hard stop | 200 GB used or above | Block all nonessential FlyBrain writes; keep only the minimal active run set; clean up ephemeral artifacts before resuming. |

### Lifecycle and retention pressure responses

1. At 150 GB, trigger a retention review and remove stale caches, archived checkpoints, and duplicate intermediate artifacts.
2. At 160 GB, switch the harness to read-only mode for noncritical jobs and keep only the current active dataset plus the latest validated outputs.
3. At 180 GB, prune scratch data aggressively (`tmp`, `staging`, `runs`, completed-but-unused downloads) and require a human override for further dataset creation.
4. At 200 GB or above, stop all nonessential writes immediately and keep only the minimal working set required for the active job.

### Safety-first operating rule

The harness should prefer under-allocation and regular cleanup over long-lived growth in the configured storage root. If the drive is near or beyond the budget thresholds, the default response is to free space first, pause expansion, and reduce retention risk rather than keep writing until the drive fills.

This policy is intentionally conservative because the same storage root can host unbacked photo/media content outside the FlyBrain allowlist and outside the harness's operational control; config-driven roots keep the rule portable across machines and shared-drive layouts.

## Operator remediation for fail-closed path/integrity errors

When path or integrity validation fails:

1. stop execution (do not bypass guards),
2. correct the path configuration and manifest file list,
3. restore or re-fetch missing/corrupt files under the allowed root,
4. regenerate manifest hash/checksum metadata,
5. rerun validation and proceed only after all checks pass.

## Targeted test coverage references

- `mcp/tests/test_flybrain_hb_adapter.py` (manifest + integrity fail-closed)
- `mcp/tests/test_flybrain_fw_metadata_adapter.py` (manifest hash mismatch and path escape rejection)
