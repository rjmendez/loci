# FlyBrain

FlyBrain, the connectome research harness (dataset adapters, sample builders, wiring
features, brain-cluster training/promotion, evaluation, and its docs and design
artifacts), lives in the private repo
[rjmendez/flybrain](https://github.com/rjmendez/flybrain). It moved out of Loci on
2026-09-25 with its git history preserved there (its `main` carries Loci's FlyBrain
history through #392; its `feat/flybrain-real-models` branch carries the work that
merged here as #393). Loci's own history still contains the old files
(`mcp/flybrain_*.py`, `docs/FLYBRAIN_*.md`, `docs/flybrain-research/`,
`artifacts/flybrain/`, `scripts/flybrain-slot`) up to `main` at `e49ad940`.

## How Loci talks to FlyBrain

Only through an MCP server. Loci carries no FlyBrain code and imports no FlyBrain module.

- **Now:** the external Virtual Fly Brain MCP. Its tools are named
  `virtual-fly-brain-*` / `virtual_fly_brain_*` (for example `search_terms`,
  `get_term_info`, `get_hierarchy`, `query_connectivity`, `run_query`). Findings built
  from those results are stored in Loci with `metadata.flybrain_provenance` and a
  `claim_scope`.
- **Planned:** the FlyBrain Research MCP, served from the flybrain repo (see its
  roadmap, `docs/FLYBRAIN_ROADMAP.md`). First read-only reference tools over
  datasets, provenance, literature, results, findings and errata; model predictions
  later.

## What Loci keeps, and why

These are Loci's own honesty guardrails for FlyBrain-derived claims arriving over MCP.
They validate what gets stored; they do not run FlyBrain.

| File | Role |
|---|---|
| `mcp/server.py` | `_FLYBRAIN_*` scope constants, `_normalize_and_validate_flybrain_claim_scope` (fail-closed `claim_scope` check on `investigation_store`, cross-sex / cross-stage stability ceiling), and the claim-scope audit emission |
| `mcp/mnemo_ops.py` | carries `metadata.flybrain_provenance` through the Mnemosyne round trip |
| `mcp/replay_fingerprint.py` | deterministic replay / access-path fingerprints for `virtual-fly-brain-*` provenance blocks |
| `mcp/tests/test_claim_scope_validator.py` | server-side claim-scope validation tests |
| `mcp/tests/test_flybrain_provenance_envelope.py` | provenance envelope and replay-fingerprint stamping through findings, memory and audit |
| `mcp/tests/test_replay_fingerprint.py` | fingerprint determinism tests |

The field contract is documented in [API_INVESTIGATION.md](API_INVESTIGATION.md)
(`metadata.flybrain_provenance`).

**Byte compatibility:** the flybrain repo vendors `mcp/replay_fingerprint.py` as
`flybrain/replay_fingerprint.py`. The two copies must stay byte-compatible so that
fingerprints computed on either side match Loci's audit records. Change them together.

## Removed from Loci

- The in-server brain-cluster MCP tools `flybrain_cluster_describe`,
  `flybrain_cluster_classify`, `flybrain_cluster_route` and `flybrain_expert_inspect`
  (added in #392), and the `FLYBRAIN_CLUSTER_STATE_PATH` setting they read. Their
  replacement is the planned FlyBrain Research MCP.
- The `braincluster-*` console scripts, the `flybrain` extra in `mcp/pyproject.toml`,
  the `test-flybrain-braincluster-gates` CI job, and the `LOCI_FLYBRAIN_STORAGE_ROOT` /
  `HARNESS_*` settings.
- FlyBrain docs, research notes, runbooks (brain-cluster promotion, rollback, alert
  playbooks), design artifacts and the `flybrain-slot` job limiter. They are in the
  flybrain repo under `docs/`, `artifacts/` and `scripts/`.
