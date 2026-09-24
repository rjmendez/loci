# FlyBrain hb/fw Local Index + Query Adapter and Import Strategy

## Purpose

Define a realistic, phase-1 local architecture for `hb` and `fw` that is operationally safe, reproducible, and explicit about scope limits.

This strategy consolidates the existing harness documents into one implementation-oriented contract for:

- local graph/query layering,
- metadata enrichment flow,
- adapter query contracts,
- import promotion + smoke-test gates,
- explicit stop boundaries to avoid over-claiming.

## Grounded assumptions (from current docs)

1. Local storage is rooted at `LOCI_FLYBRAIN_STORAGE_ROOT` with guardrail enforcement.
2. Phase-1 dataset pins are:
   - `hb=neuprint_JRC_Hemibrain_1point2point1`
   - `fw=flywire783`
3. `hb` is the local graph backbone; `fw` is metadata/provenance enrichment in phase 1 (not a full local adjacency mirror).
4. Promotion is pointer-based (blue/green style), never in-place mutation of active graph state.

## Layered architecture (local graph -> enrichment -> query contract)

### Layer 0: Storage and manifest substrate

**Responsibilities**

- Enforce path safety and root containment.
- Persist source snapshots/manifests/checksums.
- Persist candidate/promoted graph pointers.

**Canonical paths**

- `snapshots\<dataset>\<version>\source\`
- `snapshots\<dataset>\<version>\manifest\`
- `graph\hb\<version>\neo4j-store\candidate-<build_id>\`
- `graph\hb\<version>\promotion\`
- `snapshots\fw\<version>\metadata\`

**Hard rule**: if manifest or integrity checks fail, data is non-promotable.

### Layer 1: Local graph execution (`HbGraphLocalAdapter`)

**Responsibilities**

- Execute structural connectivity queries over promoted hemibrain graph.
- Enforce scope tags (`sex=female`, `stage=adult`, `anatomy=hemibrain_region`).
- Return deterministic query results + local provenance envelope.

**Query classes supported**

- Connectivity lookup (bounded partner and weight filters)
- Local neighborhood traversal (bounded depth)
- Deterministic replay/smoke query pack

**Not supported in phase 1**

- Global whole-brain claims
- Cross-dataset graph traversal (`hb` nodes mixed with `fw` IDs)

### Layer 2: Metadata enrichment (`FwMetadataLocalAdapter`)

**Responsibilities**

- Normalize entity IDs/names and annotation labels against pinned `fw` metadata snapshot.
- Add scope/context metadata (dataset/version/access-method/completeness warnings).
- Supply auxiliary annotations for reporting and route checks.

**Query classes supported**

- Entity normalization (input label -> canonical ID/name)
- Annotation and class lookups
- Metadata-level cross-checks and provenance hydration

**Not supported in phase 1**

- Local full-FlyWire graph analytics
- Full `fw` adjacency/synapse traversal from local graph

### Layer 3: Query contract and routing (`LocalFlyBrainQueryRouter`)

**Responsibilities**

- Resolve request intent and scope.
- Route to `hb` graph adapter, `fw` metadata adapter, or explicit remote fallback.
- Assemble response contract with boundary warnings.

**Routing precedence**

1. Local exact match (`dataset/version/query_kind` supported)
2. Local metadata enrichment (when graph not needed)
3. Explicit remote fallback only with provenance source switch

## Adapter query contracts

Use explicit request/response contracts to prevent ambiguous behavior.

### Request contract

Required fields:

- `dataset_symbol` (`hb` or `fw`)
- `version_id`
- `query_kind`
- `scope_tags` (`sex`, `stage`, `anatomy`, `access_layer`)
- `query_payload`
- `allow_remote_fallback` (boolean; default `false`)

### Response contract

Required fields:

- `count_status` (`exact` or `unavailable`)
- `result_rows`
- `warnings[]`
- `source` (`local` or `remote_fallback`)
- `provenance` envelope:
  - `dataset_symbol`, `version_id`, `query_kind`
  - `scope_tags`
  - `manifest_id` / `manifest_sha256`
  - `replay_fingerprint`

### Error contract

Explicit non-fail-open codes:

- `SCOPE_VIOLATION`
- `OUT_OF_SCOPE`
- `UNSUPPORTED_QUERY_KIND`
- `MANIFEST_INVALID`
- `INTEGRITY_MISMATCH`
- `PROMOTION_STATE_INVALID`

## Import strategy (hb graph + fw metadata)

### A. `hb` graph import pipeline

1. **Preflight gate**
   - Path guard pass, budget pass, pin approval pass.
2. **Snapshot acquisition**
   - Materialize source artifacts under snapshot source path.
3. **Integrity + manifest gate**
   - Validate checksums; emit complete `fbh-manifest/v1`.
4. **Offline candidate build**
   - Build Neo4j/neuPrint store in candidate path only.
5. **Smoke-test gate**
   - Run structural and semantic smoke pack (below).
6. **Promotion gate**
   - Atomic update of promotion pointer to candidate.
7. **Post-promotion verify**
   - Re-run small query pack on active pointer and compare hashes.

### B. `fw` metadata import pipeline

1. **Preflight + pin gate** (`fw=flywire783`)
2. **Metadata acquisition** to snapshot/cache
3. **Integrity + manifest gate**
4. **Normalization indexing** (entity/annotation indexes)
5. **Metadata smoke tests** (ID resolution, class lookup)
6. **Ready-state mark** (`metadata_ready=true`)

No graph promotion step is allowed for `fw` in phase 1.

## Recommended smoke-test gates before promotion

Promotion should be blocked unless all gates pass.

### Gate set 1: integrity and manifests

- `manifest.json` validates required schema fields.
- `manifest.sha256` and all file hashes match.
- `dataset_symbol + version_id` match configured pins.

### Gate set 2: graph/store liveness (`hb`)

- Neo4j/neuPrint startup probe succeeds.
- Required indexes/constraints present.
- Candidate store opens without warnings flagged as corruption.

### Gate set 3: deterministic query pack

Run fixed query pack and compare expected contract shape + bounded values:

1. Known neuron class connectivity lookup returns `count_status=exact`.
2. Bounded traversal returns non-empty rows within expected depth limits.
3. Scope-violation test intentionally fails with `SCOPE_VIOLATION`.
4. Replay fingerprint for each smoke query is emitted and stable.

### Gate set 4: promotion safety

- Candidate build ID is immutable.
- Promotion pointer update is atomic and reversible.
- Previous promoted pointer retained for rollback window.

## Recommended promotion criteria

Promote only when all of the following are true:

- Integrity gates pass (no mismatches).
- Smoke query pack passes with stable contracts.
- Storage pressure is below critical thresholds.
- Scope metadata is complete and validated.
- No unresolved `MANIFEST_INVALID`/`INTEGRITY_MISMATCH`/`PROMOTION_STATE_INVALID` errors.

## Where the stack should stop (anti-over-claim boundary)

The harness should explicitly stop short of these claims in phase 1:

1. **No full local FlyWire graph claims** from `fw` metadata-only ingestion.
2. **No cross-sex or cross-stage generalization** without direct evidence from matching datasets.
3. **No `hb` -> whole-brain extrapolation** beyond hemibrain coverage.
4. **No silent curated/raw/derived blending**; access layer must stay explicit.
5. **No remote fallback masquerading as local evidence**; source switch must be visible in provenance.
6. **No promotion after degraded smoke tests** even if import technically succeeded.

## Minimal implementation sequence

1. Implement adapter registry + capability manifests for pinned `hb`/`fw`.
2. Implement strict request/response/error contracts.
3. Implement `hb` candidate-build -> smoke-pack -> promotion pipeline.
4. Implement `fw` metadata ingest + normalization index + smoke checks.
5. Wire router with explicit fallback policy and provenance envelopes.

## Done criteria for this strategy

- A single pinned `hb` local graph can be promoted and queried reproducibly.
- `fw` metadata enrichment is available for normalization/provenance only.
- Query contracts are deterministic and fail closed on boundary violations.
- Promotion requires integrity + smoke-test pass, with rollback preserved.
- Documentation and routing language make non-supported claims impossible to mistake as supported.

## Next scoped phase adapter layer (`hb` + `fw`, local-root only)

This section is the concrete contract for the next execution phase and is intentionally constrained to:

- dataset scope: `hb`, `fw` only
- version scope: `hb=neuprint_JRC_Hemibrain_1point2point1`, `fw=flywire783`
- storage scope: descendants of `LOCI_FLYBRAIN_STORAGE_ROOT` only

### Adapter responsibilities (required)

1. `HbGraphLocalAdapter`
   - Read-only structural query execution on promoted `hb` graph state.
   - Enforce query bounds (depth/row/weight filters) and scope tags.
   - Emit deterministic provenance + replay fingerprint per query.
2. `FwMetadataLocalAdapter`
   - Read-only metadata normalization/annotation lookup over pinned `fw` snapshot.
   - Return entity map confidence and completeness warnings.
   - Refuse any graph-adjacency operation in this phase (`UNSUPPORTED_QUERY_KIND`).
3. `LocalFlyBrainQueryRouter`
   - Perform local capability resolution before any fallback decision.
   - Route to `hb` or `fw` adapter only when version pin + scope checks pass.
   - Surface source and downgrade reasons explicitly in response warnings.
4. `AdapterCompatibilityGuard`
   - Validate root containment, dataset/version pin match, and manifest integrity before execution.
   - Block requests that imply cross-dataset identity joins without allowed mapping class.

### Local-first routing policy (normative)

1. Resolve storage root from `LOCI_FLYBRAIN_STORAGE_ROOT`; reject unresolved or non-canonical root.
2. Resolve dataset/version against the pinned local manifest index.
3. Execute locally when adapter capability and scope are compatible.
4. Reject incompatible local requests with explicit fail-closed errors.
5. Use remote fallback only when `allow_remote_fallback=true`; mark `source=remote_fallback` and attach provenance source-switch warning.

No adapter may read from or write to paths outside `$LOCI_FLYBRAIN_STORAGE_ROOT\...` for this phase.

### Compatibility checks (must pass before operation dispatch)

- **Root containment check:** all resolved artifact paths are descendants of `LOCI_FLYBRAIN_STORAGE_ROOT`.
- **Dataset pin check:** request dataset/version exactly matches configured phase pins.
- **Capability check:** requested `query_kind` is declared by the selected adapter capability manifest.
- **Scope compatibility check:** requested `sex`, `stage`, `anatomy`, and `access_layer` are valid for the selected snapshot.
- **Manifest integrity check:** manifest schema + checksums valid (`MANIFEST_INVALID`/`INTEGRITY_MISMATCH` if not).
- **Promotion state check (`hb`):** active pointer points to an integrity-passed candidate build.
- **Mapping class check (`hb`<->`fw`):** only allowed join classes from comparison model; otherwise `OUT_OF_SCOPE`.

### Explicit non-goals for this scoped phase

1. Building or serving a full local FlyWire adjacency graph.
2. Supporting datasets beyond `hb` and `fw` (including auto-discovery expansion).
3. Auto-upgrading dataset versions outside explicit pin updates.
4. Performing write operations outside import/promotion pipeline roots.
5. Returning remote-derived data as if it were local evidence.
6. Producing definitive cross-dataset identity claims from medium/low-confidence mappings.

### Minimum operation contract required before next execution gate

The next gate must not open until all adapters and router implement this minimum contract and pass a deterministic smoke suite:

**Required operations**

- `capabilities(dataset_symbol, version_id) -> CapabilityManifest`
- `resolve_snapshot(dataset_symbol, version_id) -> SnapshotHandle`
- `validate_compatibility(request) -> ok | error_code`
- `execute(request) -> ResponseContract`
- `health(dataset_symbol, version_id) -> HealthStatus`

**Required response fields**

- `status`, `count_status`, `source`, `warnings[]`
- `dataset_symbol`, `version_id`, `query_kind`, `scope_tags`
- `manifest_id`, `manifest_sha256`, `replay_fingerprint`
- `compatibility_checks[]` (per-check pass/fail with reason)

**Execution-gate acceptance (all required)**

1. `hb` connectivity + traversal smoke queries return deterministic fingerprints on two consecutive runs.
2. `fw` normalization + annotation smoke queries return expected contract shapes.
3. Scope-violation and unsupported-operation tests fail closed with canonical error codes.
4. Root containment test proves no read/write path escapes configured local root.
5. Fallback test proves `allow_remote_fallback=false` blocks fallback and `true` annotates source switch.

---

Status: implementation-ready strategy for phase-1 and next scoped adapter execution gate (`hb` graph + `fw` metadata under local-root constraints).
