# FlyBrain local Neo4j/neuPrint stack plan (hemibrain scale)

## Purpose

Define a reproducible, variable-driven local stack for hemibrain-scale Neo4j/neuPrint use in the FlyBrain harness, with explicit resource profiles, import stages, and operational guardrails.

This plan is intentionally constrained by:

- [FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md](./FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md)
- [FLYBRAIN_HARNESS_STORAGE_LAYOUT.md](./FLYBRAIN_HARNESS_STORAGE_LAYOUT.md)
- [FLYBRAIN_ARTIFACT_INTEGRITY_QUARANTINE_POLICY.md](./FLYBRAIN_ARTIFACT_INTEGRITY_QUARANTINE_POLICY.md)
- [FLYBRAIN_DATASET_PROVENANCE_MATRIX.md](./FLYBRAIN_DATASET_PROVENANCE_MATRIX.md)
- [FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md](./FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md)

## Deployment shape (local harness)

Run the local stack as three lanes with one canonical storage root:

1. **Neo4j runtime lane (stateful)**
   - Hosts the active graph store for a pinned dataset version.
   - Reads/writes only under `$LOCI_FLYBRAIN_STORAGE_ROOT\graph\...`.
2. **neuPrint API lane (read-mostly)**
   - Serves query endpoints backed by the local Neo4j graph.
   - No direct filesystem writes outside runtime logs/cache under the allowlisted root.
3. **Import lane (ephemeral/batch)**
   - Performs staged data import and validation into a *new* graph store path.
   - Never mutates the currently promoted graph in place.

## Variable-driven path contract

All filesystem references must be rooted under `LOCI_FLYBRAIN_STORAGE_ROOT`.

```text
$LOCI_FLYBRAIN_STORAGE_ROOT\
  graph\
    hb\
      neuprint_JRC_Hemibrain_1point2point1\
        neo4j-store\
        promotion\
  snapshots\
    hb\
      neuprint_JRC_Hemibrain_1point2point1\
        source\
        manifest\
  cache\
    imports\
      hb\
        neuprint_JRC_Hemibrain_1point2point1\
  logs\
    neo4j\
    neuprint\
    imports\
```

No hardcoded drive letters or machine-specific absolute paths are allowed.

## Dataset/version pinning strategy

For phase-1 local graph support:

- `dataset_symbol`: `hb`
- `version_id`: `neuprint_JRC_Hemibrain_1point2point1`
- pin changes only through explicit PR updates

Every import must include a manifest with:

- `dataset_symbol`
- `version_id`
- `source_uri`
- `retrieved_at`
- `file_checksums`
- `import_tool_version`
- `schema/constraint version`
- `promotion_decision`

Imports without a complete manifest are non-promotable.

## Resource profiles (hemibrain scale)

Profiles assume dataset scope pinned to hemibrain v1.2.1 and storage constrained by guardrails.

| Profile | CPU | RAM | Graph+index storage | Scratch/import overhead | Intended use |
|---|---:|---:|---:|---:|---|
| **Minimal** | 8 vCPU | 32 GB | 40-60 GB | 25-40 GB | Single-user local analysis; serial imports; no concurrent heavy query load |
| **Recommended** | 16 vCPU | 64 GB | 60-90 GB | 40-70 GB | Regular harness use; concurrent query + import validation; faster rebuild/replay |

Guardrail alignment:

- Keep steady-state FlyBrain usage below soft cap in the write-path policy.
- Treat scratch growth as burst-only and prune after promotion.
- Block nonessential import work when policy thresholds require write freeze.

## Import workflow (staged, non-destructive)

### Stage 0 — preflight safety gate

1. Resolve all candidate paths via shared guard (`mcp/flybrain_harness_storage.py`).
2. Validate allowlist containment and reparse-point protections.
3. Verify available space against budget thresholds before side effects.

### Stage 1 — acquire pinned source snapshot

1. Pull/export source artifacts for pinned `dataset_symbol` + `version_id`.
2. Store immutable snapshot files under:
   - `$LOCI_FLYBRAIN_STORAGE_ROOT\snapshots\hb\<version_id>\source\`
3. Write manifest + checksums under `...\manifest\`.

### Stage 2 — verify and prepare import inputs

1. Validate checksums and manifest completeness.
2. Normalize import files to deterministic, replayable layout under:
   - `$LOCI_FLYBRAIN_STORAGE_ROOT\cache\imports\hb\<version_id>\`
3. Abort on any mismatch (no fail-open).

### Stage 3 — offline graph build

1. Build into a new target store:
   - `$LOCI_FLYBRAIN_STORAGE_ROOT\graph\hb\<version_id>\neo4j-store\candidate-<build_id>\`
2. Apply required constraints/indexes.
3. Run structural integrity checks and basic neuPrint-compatible smoke queries.

### Stage 4 — promotion (blue/green style)

1. Keep current promoted store untouched.
2. Promote by updating a promotion pointer/manifest atomically under:
   - `$LOCI_FLYBRAIN_STORAGE_ROOT\graph\hb\<version_id>\promotion\`
3. Restart/rebind runtime lane to new promoted store.

### Stage 5 — post-promotion cleanup

1. Retain previous promoted store for rollback window.
2. Prune only ephemeral scratch (`cache\imports\...`) per retention rules.
3. Never delete root or dataset version root; prune explicit descendants only.

## Write-path safety and operational boundaries

1. **No out-of-root writes:** all paths must validate under `LOCI_FLYBRAIN_STORAGE_ROOT`.
2. **No in-place mutation of active store:** imports always build into a candidate path first.
3. **No destructive ops on roots:** root, drive root, and dataset-version anchors are protected.
4. **No unpinned imports:** dataset symbol/version must match approved pin.
5. **No fail-open on safety checks:** path, checksum, and manifest failures hard-stop promotion.
6. **Read-only freeze behavior:** when storage policy enters critical/hard-stop bands, allow only essential read and cleanup operations until below threshold.

## Recommended rollout order

1. Land path-guard and storage preflight enforcement.
2. Land pinned hemibrain snapshot manifest flow.
3. Land offline import + promotion pointer flow.
4. Enable neuPrint API lane against promoted local store.
5. Add telemetry for import duration, storage deltas, and promotion success/failure.

## Out-of-scope for this phase

- Full local FlyWire adjacency mirror
- Multi-dataset concurrent local Neo4j instances beyond pinned hemibrain baseline
- Any storage layout that bypasses `LOCI_FLYBRAIN_STORAGE_ROOT` controls

---

Status: planned and ready for implementation sequencing in the FlyBrain harness.
