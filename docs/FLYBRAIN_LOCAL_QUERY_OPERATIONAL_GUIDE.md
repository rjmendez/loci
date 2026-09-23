# FlyBrain Local Query Operationalization Guide (hb/fw scope)

## Executive Summary

This guide explains how to operationalize the FlyBrain harness for local reproducible queries using Hemibrain (`hb`) and FlyWire (`fw`) datasets. It is a practical, step-by-step manual for users who want to set up the harness, populate local data, and execute queries against a local graph.

**Target scope:** Adult Drosophila hemibrain connectome snapshot + FlyWire metadata for provenance queries.

**Key constraint:** All storage is config-driven via `LOCI_FLYBRAIN_STORAGE_ROOT`; no hardcoded drive letters.

---

## 1. Prerequisites and Environment Setup

### 1.1 Environment Variables

Set the following in your shell or `.env` file:

```bash
# Canonical storage root for all FlyBrain harness data (no hardcoded F:\ or similar)
export LOCI_FLYBRAIN_STORAGE_ROOT="/mnt/data/loci-flybrain"
# or on Windows:
# set LOCI_FLYBRAIN_STORAGE_ROOT=D:\loci-flybrain

# Optional convenience aliases
export HARNESS_DATA_ROOT="$LOCI_FLYBRAIN_STORAGE_ROOT"
export HARNESS_STORAGE_ROOT="$LOCI_FLYBRAIN_STORAGE_ROOT"

# Pipeline run mode (dry-run | plan | execute)
export LOCI_FLYBRAIN_RUN_MODE="dry-run"

# Dataset scope for phase 1
export LOCI_FLYBRAIN_DATASET_SCOPE="hb,fw"

# Version pins (immutable reference identifiers)
export LOCI_FLYBRAIN_DATASET_VERSION_PINS="hb=neuprint_JRC_Hemibrain_1point2point1;fw=flywire783"

# Storage budget caps (in GB)
export LOCI_FLYBRAIN_BUDGET_SOFT_CAP_GB=160
export LOCI_FLYBRAIN_BUDGET_HARD_CAP_GB=200

# Retry settings
export LOCI_FLYBRAIN_MAX_RETRIES=3
export LOCI_FLYBRAIN_RETRY_BACKOFF_SECONDS=10

# Continue on dataset-level failure (strict mode by default)
export LOCI_FLYBRAIN_CONTINUE_ON_DATASET_FAILURE="false"
```

### 1.2 Quick Bootstrap: Windows and WSL2

Use the exact sequence below to bootstrap the local FlyBrain harness in a way that stays inside the configured root and does not rely on a hardcoded drive letter.

#### PowerShell (Windows)

```powershell
$ErrorActionPreference = 'Stop'
$env:LOCI_FLYBRAIN_STORAGE_ROOT = 'F:\loci-data\flybrain'
$env:LOCI_FLYBRAIN_RUN_MODE = 'dry-run'
$env:LOCI_FLYBRAIN_DATASET_SCOPE = 'hb,fw'
$env:LOCI_FLYBRAIN_DATASET_VERSION_PINS = 'hb=neuprint_JRC_Hemibrain_1point2point1;fw=flywire783'
$env:LOCI_FLYBRAIN_BUDGET_SOFT_CAP_GB = '160'
$env:LOCI_FLYBRAIN_BUDGET_HARD_CAP_GB = '200'

python -c "import os; from pathlib import Path; root = Path(os.environ['LOCI_FLYBRAIN_STORAGE_ROOT']); root.mkdir(parents=True, exist_ok=True); for d in ['graph','snapshots','cache','backups','logs']: (root / d).mkdir(parents=True, exist_ok=True); print(root); print(sorted((p.name for p in root.iterdir())))"

python -c "import os; from mcp.flybrain_harness_storage import build_flybrain_harness_layout; layout = build_flybrain_harness_layout(create=True); print(layout.as_dict())"
```

#### Bash (WSL2)

```bash
export LOCI_FLYBRAIN_STORAGE_ROOT="/mnt/f/loci-data/flybrain"
export LOCI_FLYBRAIN_RUN_MODE="dry-run"
export LOCI_FLYBRAIN_DATASET_SCOPE="hb,fw"
export LOCI_FLYBRAIN_DATASET_VERSION_PINS="hb=neuprint_JRC_Hemibrain_1point2point1;fw=flywire783"
export LOCI_FLYBRAIN_BUDGET_SOFT_CAP_GB=160
export LOCI_FLYBRAIN_BUDGET_HARD_CAP_GB=200

mkdir -p "$LOCI_FLYBRAIN_STORAGE_ROOT"/{graph,snapshots,cache,backups,logs}
python3 -c "import os; from pathlib import Path; root = Path(os.environ['LOCI_FLYBRAIN_STORAGE_ROOT']); print(root); print(sorted([p.name for p in root.iterdir()]))"
python3 -c "import os; from mcp.flybrain_harness_storage import build_flybrain_harness_layout; layout = build_flybrain_harness_layout(create=True); print(layout.as_dict())"
```

#### Dry-run / plan / execute sequence

```bash
# Dry run: validation only; no bulk download starts
export LOCI_FLYBRAIN_RUN_MODE="dry-run"
python3 scripts/flybrain_fetch_sync_pipeline.py \
  --run-mode dry-run \
  --dataset-scope "hb,fw" \
  --version-pins "hb=neuprint_JRC_Hemibrain_1point2point1;fw=flywire783" \
  --log-dir "$LOCI_FLYBRAIN_STORAGE_ROOT/logs/pipelines"

# Plan: materialize the action plan and idempotency state
export LOCI_FLYBRAIN_RUN_MODE="plan"
python3 scripts/flybrain_fetch_sync_pipeline.py \
  --run-mode plan \
  --dataset-scope "hb,fw" \
  --version-pins "hb=neuprint_JRC_Hemibrain_1point2point1;fw=flywire783" \
  --log-dir "$LOCI_FLYBRAIN_STORAGE_ROOT/logs/pipelines"

# Execute: only after dry-run and plan pass
export LOCI_FLYBRAIN_RUN_MODE="execute"
python3 scripts/flybrain_fetch_sync_pipeline.py \
  --run-mode execute \
  --dataset-scope "hb,fw" \
  --version-pins "hb=neuprint_JRC_Hemibrain_1point2point1;fw=flywire783" \
  --log-dir "$LOCI_FLYBRAIN_STORAGE_ROOT/logs/pipelines"
```

### 1.3 Storage Root Validation

Before any data work, validate the storage root exists and is writable:

```bash
# Preflight check
python3 -c "
from mcp.flybrain_harness_storage import build_flybrain_harness_layout
import os

root = os.getenv('LOCI_FLYBRAIN_STORAGE_ROOT')
if not root:
    raise ValueError('LOCI_FLYBRAIN_STORAGE_ROOT not set')

layout = build_flybrain_harness_layout(create=True)
print(f'Storage root validated: {layout.root}')
print(layout.as_dict())
"
```

---

## 2. Data Provenance Contract (hb/fw scope)

### 2.1 Dataset Pins and Versions

| Dataset | Symbol | Version ID | Storage Path | Scope |
|---|---|---|---|---|
| Hemibrain | `hb` | `neuprint_JRC_Hemibrain_1point2point1` | `$LOCI_FLYBRAIN_STORAGE_ROOT\graph\hb\neuprint_JRC_Hemibrain_1point2point1\` | Adult female hemibrain region (~1/3 brain, excludes full optic lobes, VNC) |
| FlyWire Metadata | `fw` | `flywire783` | `$LOCI_FLYBRAIN_STORAGE_ROOT\snapshots\fw\flywire783\metadata\` | Adult female whole-brain metadata and entity annotations |

### 2.2 Scope Boundaries (Hard Limits)

**Do not apply these results to claims outside the stated scope:**

- Hemibrain results are **hemibrain-region only**; do not generalize to whole-brain or optic-lobe claims.
- FlyWire metadata is **female adult only**; do not treat as representative of male or larval data.
- Both datasets are **adult-stage data**; do not apply to larval/developmental claims.
- Sex is **explicitly female**; cross-sex claims require separate male dataset evidence.

### 2.3 Provenance Requirements for Claims

Every local query result must include:

```json
{
  "flybrain_provenance": {
    "dataset_symbol": "hb",
    "version_id": "neuprint_JRC_Hemibrain_1point2point1",
    "access_method": "local_graph_snapshot",
    "query_kind": "connectivity_lookup",
    "scope": {
      "sex": "female",
      "stage": "adult",
      "anatomy": "hemibrain_region",
      "evidence_family": "connectome_structural"
    },
    "local_snapshot_dir": "$LOCI_FLYBRAIN_STORAGE_ROOT/graph/hb/neuprint_JRC_Hemibrain_1point2point1/",
    "manifest_version": "fbh-manifest/v1",
    "retrieved_at": "2026-09-22T23:30:00Z"
  }
}
```

---

## 3. Fetch and Sync Workflow

### 3.1 Stage 0 — Preflight and Policy Gate

**Goal:** Validate configuration and budget before any data writes.

```bash
# Set run mode to dry-run (safe, no writes)
export LOCI_FLYBRAIN_RUN_MODE="dry-run"

# Run preflight validation
python3 scripts/flybrain_fetch_sync_pipeline.py \
  --run-mode dry-run \
  --dataset-scope "hb,fw" \
  --version-pins "hb=neuprint_JRC_Hemibrain_1point2point1;fw=flywire783" \
  --log-dir "$LOCI_FLYBRAIN_STORAGE_ROOT/logs/pipelines"
```

**Expected output:**
- Path validation results
- Storage budget analysis
- Dataset scope compliance check
- Dry-run action plan (no writes performed)

**Go/no-go gates:**
- Root path must pass containment check (under configured root, no symlink escapes)
- Storage used + planned data must fit soft cap (160 GB)
- Dataset scope must be ["hb", "fw"] only

### 3.2 Stage 1 — Planning

**Goal:** Generate concrete plan and verify before execution.

```bash
export LOCI_FLYBRAIN_RUN_MODE="plan"

python3 scripts/flybrain_fetch_sync_pipeline.py \
  --run-mode plan \
  --dataset-scope "hb,fw" \
  --version-pins "hb=neuprint_JRC_Hemibrain_1point2point1;fw=flywire783" \
  --log-dir "$LOCI_FLYBRAIN_STORAGE_ROOT/logs/pipelines"
```

This is the current phase-1 gate: a plan-only run must produce `logs/pipelines/<run_id>/plan.json` and stage-state metadata without progressing to the execute stage.

**Expected output:**
- Concrete plan artifact: `$LOCI_FLYBRAIN_STORAGE_ROOT/logs/pipelines/<run_id>/plan.json`
- Expected downloads: hemibrain graph (~15 GB) + FlyWire metadata (~5 GB)
- Expected writes: snapshots, manifests, checksums

**Review before proceeding:**
- Verify dataset versions match pins
- Verify storage paths are within root
- Confirm total size will fit budget

### 3.3 Stage 2–5 — Resumable Acquisition and Integrity

**Goal:** Download, verify, and promote data to active graph.

```bash
export LOCI_FLYBRAIN_RUN_MODE="execute"

python3 scripts/flybrain_fetch_sync_pipeline.py \
  --run-mode execute \
  --dataset-scope "hb,fw" \
  --version-pins "hb=neuprint_JRC_Hemibrain_1point2point1;fw=flywire783" \
  --log-dir "$LOCI_FLYBRAIN_STORAGE_ROOT/logs/pipelines"
```

**Per-dataset handling:**

1. **Hemibrain (`hb`):**
   - Download or acquire graph export from Neuprint `neuprint_JRC_Hemibrain_1point2point1`.
   - Build candidate graph in `$LOCI_FLYBRAIN_STORAGE_ROOT/graph/hb/neuprint_JRC_Hemibrain_1point2point1/neo4j-store/candidate-<build_id>/`.
   - Run smoke tests (connectivity counts, traversal checks).
   - Require active promotion pointer at `$LOCI_FLYBRAIN_STORAGE_ROOT/graph/hb/neuprint_JRC_Hemibrain_1point2point1/promotion/active_pointer.json` with `{"active": true}` before query execution.

2. **FlyWire Metadata (`fw`):**
   - Download metadata from VFB/FlyBrain dataset list for `flywire783`.
   - Extract and materialize in `$LOCI_FLYBRAIN_STORAGE_ROOT/snapshots/fw/flywire783/metadata/`.
   - Write manifest with entity inventory and checksums.
   - No graph promotion; marked as metadata-ready.

**Resumption on failure:**

If the pipeline fails partway through:

```bash
# Re-run with the same environment and dataset pins
# The pipeline will check idempotency keys and resume from the checkpoint
python3 scripts/flybrain_fetch_sync_pipeline.py \
  --run-mode execute \
  --dataset-scope "hb,fw" \
  --version-pins "hb=neuprint_JRC_Hemibrain_1point2point1;fw=flywire783" \
  --log-dir "$LOCI_FLYBRAIN_STORAGE_ROOT/logs/pipelines"
```

The pipeline will skip completed stages and resume partial downloads via byte-range requests.

### Stage 2–5 hard fail-closed gates (before local query use)

Do not route local queries to a snapshot unless all of these gates pass:

1. `integrity.manifest_sha256` canonical self-hash verification passes.
2. Every `integrity.files` entry resolves safely under artifact root and matches `sha256` (and `size_bytes` when present).
3. `integrity.verification.status == "verified"` and `verified_at` is present.
4. `refresh.decision != "rollback"` and `refresh.next_check_due` is not expired.
5. `hb` active promotion pointer exists and is active (`active_pointer.json`).

---

## 4. Local Storage Layout After Population

Once the fetch/sync completes, the directory tree will look like:

```
$LOCI_FLYBRAIN_STORAGE_ROOT/
├── graph/
│   └── hb/
│       └── neuprint_JRC_Hemibrain_1point2point1/
│           ├── neo4j-store/
│           │   ├── promotion/
│           │   │   └── active_pointer.json
│           │   └── candidate-<hash>/
│           │       ├── store/
│           │       ├── index/
│           │       └── smoke-test-results.json
│           └── schema.json
├── snapshots/
│   ├── hb/
│   │   └── neuprint_JRC_Hemibrain_1point2point1/
│   │       ├── source/
│   │       └── manifest/
│   │           ├── manifest.json
│   │           ├── manifest.sha256
│   │           └── source_provenance.json
│   └── fw/
│       └── flywire783/
│           ├── source/
│           ├── metadata/
│           │   ├── entities.json
│           │   ├── annotations.json
│           │   └── index.json
│           └── manifest/
│               └── manifest.json
├── cache/
│   └── downloads/
│       ├── hb/
│       ├── fw/
│       └── (partial/chunk state files)
├── backups/
│   └── (retention-managed backup bundles)
└── logs/
    └── pipelines/
        └── <run_id>/
            ├── preflight.json
            ├── plan.json
            ├── stage_state/
            └── summary.json
```

---

## 5. Query Patterns and Local Adapter Usage

### 5.1 Local Hemibrain Graph Queries

Use the `HbGraphLocalAdapter` for reproducible, offline queries.

```python
from mcp.flybrain_harness_adapters import HbGraphLocalAdapter
import os

# Initialize adapter from environment
storage_root = os.getenv('LOCI_FLYBRAIN_STORAGE_ROOT')
hb_adapter = HbGraphLocalAdapter(storage_root=storage_root)

# Query 1: Connectivity lookup
# "Which neurons connect to the Kenyon cell class in the hemibrain?"
result = hb_adapter.query(
    query_type='connectivity_lookup',
    target_class='FBbt_00003686',  # Kenyon cell
    upstream=True,
    limit=100
)
# Returns: {
#   'edges': [{'source': '...', 'target': '...', 'weight': 50}, ...],
#   'count': 1858,
#   'provenance': {'dataset_symbol': 'hb', 'version_id': '...'}
# }

# Query 2: Class-neighbor traversal
# "What are the direct presynaptic inputs to a Kenyon cell?"
neighbors = hb_adapter.query(
    query_type='neighborhood_traversal',
    target_class='FBbt_00003686',
    direction='presynaptic',
    threshold_weight=10
)

# Query 3: Validate connectivity deterministically
# (Smoke test: verify connectivity counts remain stable across runs)
smoke_result = hb_adapter.smoke_test()
# Returns: {'status': 'pass', 'connectivity_count': 12345, 'graph_health': 'ok'}
```

### 5.2 FlyWire Metadata Queries

Use the `FwMetadataLocalAdapter` for entity normalization and annotation lookup.

```python
from mcp.flybrain_harness_adapters import FwMetadataLocalAdapter

fw_adapter = FwMetadataLocalAdapter(storage_root=storage_root)

# Query 1: Entity normalization
# "Map a FlyWire cell ID to a canonical FlyBrain term"
entity = fw_adapter.query(
    query_type='entity_lookup',
    fw_cell_id='720575940625281'
)
# Returns: {
#   'fw_id': '720575940625281',
#   'flybase_term': 'FBbt_00003686',
#   'anatomical_region': 'mushroom_body',
#   'cell_type': 'Kenyon cell'
# }

# Query 2: Annotation lookup
# "Find all annotations tagged with 'descending neuron'"
annotations = fw_adapter.query(
    query_type='annotation_lookup',
    tag='descending_neuron',
    limit=50
)

# Query 3: Provenance check
# "Verify the metadata snapshot is current"
provenance = fw_adapter.provenance_envelope()
# Returns: {
#   'version_id': 'flywire783',
#   'generated_at': '2026-09-22T...',
#   'source_uri': 'https://flywire.ai/...',
#   'entity_count': 150000,
#   'manifest_sha256': '...'
# }
```

### 5.3 Cross-Dataset Queries (Careful Scope Boundaries)

Use the `RemoteFallbackAdapter` only when local data is insufficient **and** you explicitly note the scope change.

```python
from mcp.flybrain_harness_adapters import RemoteFallbackAdapter

remote_adapter = RemoteFallbackAdapter()

# ONLY if you need male CNS data not in the local hb snapshot:
male_data = remote_adapter.query(
    dataset='male_cns_v1_0',
    query_type='connectivity_lookup',
    target_class='FBbt_00003686',
    note='Male CNS fallback; not hemibrain scope'
)

# **Ensure the result is tagged with remote scope and cannot be
#    confused with the local hemibrain result.**
```

---

## 6. Query Result Handling and Scope

### 6.1 Recommended Query Pattern

```python
def query_connectivity_local(
    target_class: str,
    upstream: bool = True,
    dataset_scope: str = "hb",
    storage_root: str = None
) -> dict:
    """
    Query local hemibrain connectivity with strict scope binding.
    
    Args:
        target_class: FlyBrain term ID (e.g., 'FBbt_00003686')
        upstream: direction of query
        dataset_scope: must be 'hb' (hemibrain only in phase 1)
        storage_root: $LOCI_FLYBRAIN_STORAGE_ROOT
    
    Returns:
        Result dict with strict provenance envelope.
    """
    if dataset_scope != "hb":
        raise ValueError("Phase 1 supports 'hb' only. Use RemoteFallbackAdapter for others.")
    
    storage_root = storage_root or os.getenv('LOCI_FLYBRAIN_STORAGE_ROOT')
    hb_adapter = HbGraphLocalAdapter(storage_root=storage_root)
    
    raw_result = hb_adapter.query(
        query_type='connectivity_lookup',
        target_class=target_class,
        upstream=upstream
    )
    
    # Wrap with strict scope boundary
    return {
        'result': raw_result,
        'scope_boundary': {
            'dataset': 'hemibrain',
            'version': 'neuprint_JRC_Hemibrain_1point2point1',
            'sex': 'female',
            'stage': 'adult',
            'anatomy': 'hemibrain_region_only',
            'warning': 'Do not apply to whole-brain, optic-lobe, or male/larval claims'
        },
        'flybrain_provenance': {
            'tool_name': 'query_connectivity',
            'dataset_symbol': 'hb',
            'version_id': 'neuprint_JRC_Hemibrain_1point2point1',
            'access_method': 'local_graph_snapshot',
            'retrieved_at': datetime.now().isoformat()
        }
    }
```

### 6.2 Scope Language Guidelines

| Evidence | Safe Language | Unsafe Language |
|---|---|---|
| Local hb query result | "In the hemibrain connectome..." | "In the fly brain..." |
| Local hb + fw metadata | "Observed in hemibrain with FlyWire metadata..." | "In all Drosophila..." |
| Multiple adult female datasets (future) | "Across adult female datasets..." | "In the brain..." |

---

## 7. Storage Limits and Budget Management

### 7.1 Soft and Hard Caps

```
Watch threshold     (150 GB): Pause downloads; review retention
Soft cap            (160 GB): Stop new long-lived datasets
Critical threshold  (180 GB): Freeze non-essential writes
Hard stop           (200 GB): Block all non-essential writes
```

### 7.2 Checking Current Disk Usage

```bash
du -sh "$LOCI_FLYBRAIN_STORAGE_ROOT"

# Expected breakdown after full population:
# - graph/hb: ~15 GB (active graph store)
# - snapshots/fw/metadata: ~5 GB (metadata)
# - snapshots: ~20 GB (snapshot bundles with manifests)
# - cache: ~2 GB (resumable partials, index caches)
# - backups: ~1 GB (backup metadata)
# - logs: <1 GB (run logs)
# Total: ~43 GB (well under soft cap)
```

### 7.3 Cleanup and Retention

If disk pressure occurs:

**Step 1 — Check cache (safe to reclaim):**
```bash
rm -rf "$LOCI_FLYBRAIN_STORAGE_ROOT/cache/downloads"
# Rebuilds on next sync run
```

**Step 2 — Archive old snapshots (retain one):**
```bash
ls -lh "$LOCI_FLYBRAIN_STORAGE_ROOT/snapshots/hb/" | tail -n +2 | awk '{print $NF}' | xargs -I {} \
  tar czf "$LOCI_FLYBRAIN_STORAGE_ROOT/backups/snapshot_archive_{}.tar.gz" \
  "$LOCI_FLYBRAIN_STORAGE_ROOT/snapshots/hb/{}" && \
  rm -rf "$LOCI_FLYBRAIN_STORAGE_ROOT/snapshots/hb/"*
```

**Step 3 — Never delete active graph or logs without explicit approval.**

---

## 8. Integration with Loci Memory and Investigation

Once local data is ready, use it in Loci queries:

```python
# In loci/mcp/investigation_start.py or query routing logic:

from mcp.flybrain_harness_adapters import HbGraphLocalAdapter, FwMetadataLocalAdapter

def resolve_flybrain_entity(entity_input: str, dataset_preference: str = "local_hb"):
    """Route entity resolution through local adapter if available."""
    
    storage_root = os.getenv('LOCI_FLYBRAIN_STORAGE_ROOT')
    
    if dataset_preference == "local_hb" and storage_root:
        try:
            hb_adapter = HbGraphLocalAdapter(storage_root=storage_root)
            result = hb_adapter.resolve_entity(entity_input)
            if result:
                return result  # Local hit; use it with provenance
        except Exception as e:
            log.warning(f"Local adapter failed, falling back to remote: {e}")
    
    # Fallback: remote resolution with scope tagging
    # ...
```

---

## 9. Operational Checklist

- [ ] Environment variables set and validated
- [ ] Storage root created and passes write test
- [ ] Preflight check passes (dry-run mode)
- [ ] Plan review completed (plan mode)
- [ ] Data fetch executed (execute mode)
- [ ] Smoke tests pass (connectivity counts stable)
- [ ] Manifests written with correct checksums
- [ ] Log artifacts present and readable
- [ ] Query adapters initialized and tested
- [ ] Loci integration routes queries to local adapters
- [ ] Scope boundaries documented in every claim

---

## 10. Troubleshooting

### Problem: `LOCI_FLYBRAIN_STORAGE_ROOT not set`
**Solution:** Export the environment variable in `.env`, `.bashrc`, or run it inline:
```bash
LOCI_FLYBRAIN_STORAGE_ROOT=/path/to/storage python3 scripts/...
```

### Problem: `Path outside allowlist root` error
**Solution:** Check that all paths resolve under the configured root. Use the validator:
```bash
python3 -c "from mcp.flybrain_harness_storage import validate_flybrain_path; print(validate_flybrain_path('$LOCI_FLYBRAIN_STORAGE_ROOT/graph', 'read'))"
```

### Problem: `Storage budget exceeded` error
**Solution:** Clean cache or old snapshots (see section 7.3) and re-run.

### Problem: Download interrupted; how do I resume?
**Solution:** Re-run the execute command with the same dataset pins. The pipeline checks idempotency keys and resumes from the checkpoint.

### Problem: Checksum mismatch on acquired files
**Solution:** Hard failure; do not auto-retry. Check source data availability and clear `cache/downloads/<dataset>` before retry.

### Problem: `integrity.verification.status` is not `verified`
**Solution:** Treat as non-promotable and non-queryable. Re-run integrity verification, populate `verified_at`, and retry only after status is `verified`.

### Problem: Missing or inactive hb `active_pointer.json`
**Solution:** Do not query local hb data. Restore a valid active promotion pointer or rerun promotion for the pinned hb snapshot.

---

## 11. Next Steps and Future Scope

**Phase 1 (current):** Local hemibrain graph + FlyWire metadata snapshot.

**Phase 2 (future, requires policy update):**
- Add male CNS dataset (`mc`)
- Implement cross-dataset entity normalization
- Build comparative routing logic

**Phase 3 (future):**
- Multi-dataset graph federation
- Query optimization and caching across datasets
- Automated data refresh cadence

---

**Status:** Ready for local-first operational use.  
**Document version:** 1.1 (2026-09-23)  
**Last reviewed:** 2026-09-23

## 11.1 Targeted test coverage for the guardrails

- `mcp/tests/test_flybrain_hb_adapter.py`
  - scope/fallback rejection
  - manifest verification rollback enforcement
- `mcp/tests/test_flybrain_fw_metadata_adapter.py`
  - manifest self-hash mismatch rejection
  - path-escape rejection for `integrity.files`

## 12. Bias and Failure-Mode Handling (hb/fw local stack)

This section distills practical failure handling from:

- `docs/FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md`
- `docs/FLYBRAIN_DATASET_PROVENANCE_MATRIX.md`
- `docs/FLYBRAIN_HEMIBRAIN_VS_FLYWIRE.md`
- local evidence matrix `FLYBRAIN_HB_FW_LOCAL_GRAPH_FAILURE_MATRIX.md`

Treat these as hard operational rules, not soft recommendations.

| Failure mode | Typical trigger | Detection rule | Required mitigation | Claim impact |
|---|---|---|---|---|
| Overgeneralization | A result from one dataset/scope is phrased as species-wide fact | Provenance has one dataset/version and no matched corroboration | Keep wording dataset-local (`In hb...` or `In fw...`), or run matched corroboration query before broadening | Downgrade to dataset-local tier |
| Stage mixing | Adult (`hb`, `fw`) result used for larval claim | `scope.stage` mismatch (`adult` vs `larval`) | Block automatic merge; require explicit larval evidence (`l1em`) | Mark unsupported for cross-stage claims |
| Annotation drift | New upstream snapshot or changed class labels | Manifest version differs from active version pin, or replay fingerprint drift | Re-run pinned query, compare `result_contract`; if drift >5%, flag and re-baseline | Confidence decay until revalidated |
| Query-induced false certainty | Narrow query outputs look complete (`count_status` ignored, high weight filters, over-tight `exclude_dbs`) | `count_status != exact`, high `weight` without relaxation, or scoped exclusions omitted in summary | Include `count_status`, `weight`, `exclude_dbs`, and caveat language in output contract | Prevent promotion to high-confidence |
| Neuron-class connectivity misread | Grouped totals interpreted as raw edge counts | `group_by_class=true` but totals read as pairwise exact edges | State grouped semantics explicitly; rerun ungrouped sample for sanity check | Keep claim as rollup, not exact pair count |
| Anatomy query boundary leak | Region hierarchy terms reused as neuron-class evidence | `part_of` anatomy query used without class resolution | Resolve neuron classes (`search_terms` + filter) before connectivity claims | Restrict to structural/anatomy statement only |

### 12.1 Practical penalties and stop conditions

Use this default policy when writing memory findings or final answers:

1. Cross-sex or cross-stage extrapolation without direct evidence: **stop** (no confidence score).
2. Version mismatch or replay drift >5%: apply confidence decay and schedule replay check.
3. `count_status=unavailable` or failed upstream query: report as **query failed**, never as zero.
4. Predicted neurotransmitter-only chains: cap at low/medium and label as predicted.

### 12.2 Operator checks before storing a claim

- Confirm dataset symbol + version pin are present (`hb`/`fw` + version-bearing ID).
- Confirm `scope` fields (`sex`, `stage`, `anatomy`) are populated.
- Confirm query semantics are retained (`group_by_class`, `weight`, paging, dataset filters).
- Confirm result contract includes `count`, `count_status`, and returned row count.
- Confirm replay fingerprint exists for deterministic re-checks.

## 13. Reusable Query Templates (evidence-bound contracts)

These templates are designed for local hb/fw operations with explicit limits and replayability.

### Template A — Neuron-class connectivity within one dataset

**Use when:** you need inputs/outputs for a neuron class (for example Kenyon cell family) without cross-dataset claims.

1. Resolve neuron class ID with `search_terms` (`filter_types=["neuron","class"]`).
2. Run connectivity query in one dataset lane (`hb` or `fw`) with explicit `weight`, `limit`, and `group_by_class`.
3. If zero rows, relax in order: lower `weight` -> remove exclusions -> optional grouped mode.
4. Store full provenance and result contract.

Contract skeleton:

```json
{
  "question_type": "neuron_class_connectivity",
  "dataset_scope": {
    "symbol": "hb",
    "version_id": "neuprint_JRC_Hemibrain_1point2point1",
    "sex": "female",
    "stage": "adult"
  },
  "query_semantics": {
    "group_by_class": true,
    "weight": 20,
    "limit": 50,
    "exclude_dbs": ["fw"]
  },
  "result_contract": {
    "count": 1858,
    "count_status": "exact",
    "returned_rows": 50
  },
  "safe_wording": "In the hemibrain dataset and this query configuration..."
}
```

### Template B — hb/fw matched comparison (same class, same semantics)

**Use when:** you need a comparison, not a merge, across hb and fw.

1. Use the same resolved class IDs and the same query semantics in both runs.
2. Run `hb` and `fw` separately; do not mix row sets.
3. Compare at normalized level (rank/order/relative proportion), not raw absolute totals alone.
4. Report where scopes diverge (coverage, annotation model, region boundaries).

Contract skeleton:

```json
{
  "question_type": "cross_dataset_comparison",
  "runs": [
    {"symbol": "hb", "version_id": "neuprint_JRC_Hemibrain_1point2point1"},
    {"symbol": "fw", "version_id": "flywire783"}
  ],
  "matched_semantics": {
    "group_by_class": true,
    "weight": 20,
    "limit": 50
  },
  "comparison_basis": "ranked partner classes and relative weight distribution",
  "safe_wording": "Compared across hb and fw under matched settings; differences may reflect dataset scope and annotation model."
}
```

### Template C — Anatomy hierarchy then scoped class extraction

**Use when:** user asks an anatomy question first (region/neuropil), then connectivity.

1. Run hierarchy (`part_of`) to establish structural context.
2. Resolve neuron classes inside that anatomy scope.
3. Run class connectivity query only after class resolution.
4. Keep anatomy statements and connectivity statements separate in output.

Contract skeleton:

```json
{
  "question_type": "anatomy_to_class_connectivity",
  "anatomy_query": {
    "relationship": "part_of",
    "max_depth": 1
  },
  "class_resolution": {
    "filter_types": ["neuron", "class"],
    "scope_bound": true
  },
  "connectivity_query": {
    "dataset_symbol": "fw",
    "group_by_class": true,
    "weight": 50
  },
  "safe_wording": "Within the specified anatomy scope and resolved class set..."
}
```

### Template D — Evidence-bound answer contract

Use this shape for final stored findings and user-facing summaries:

```json
{
  "claim": "...",
  "claim_tier": "dataset_run_specific",
  "supports": [
    {"dataset": "hb", "version_id": "neuprint_JRC_Hemibrain_1point2point1", "count_status": "exact"}
  ],
  "boundaries": {
    "sex": "female",
    "stage": "adult",
    "anatomy_scope": "hemibrain region",
    "not_generalizable_to": ["male", "larval", "full VNC"]
  },
  "replay_fingerprint": "sha256:..."
}
```

If this contract cannot be filled with evidence from the executed query path, the answer should be downgraded to an explicit gap/assumption rather than presented as established fact.
