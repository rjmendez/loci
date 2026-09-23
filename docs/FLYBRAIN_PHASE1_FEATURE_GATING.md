# FlyBrain feature gating and performance constraints (phase 1/2/3)

## Purpose

Define explicit stop/go gates for enabling FlyBrain feature tiers under local-first harness constraints, grounded in the current storage, rollout, and import strategy docs.

Primary evidence inputs:

- [FLYBRAIN_HARNESS_STORAGE_LAYOUT.md](./FLYBRAIN_HARNESS_STORAGE_LAYOUT.md)
- [FLYBRAIN_HARNESS_ROLLOUT_MILESTONES.md](./FLYBRAIN_HARNESS_ROLLOUT_MILESTONES.md)
- [FLYBRAIN_NEO4J_NEUPRINT_LOCAL_STACK_PLAN.md](./FLYBRAIN_NEO4J_NEUPRINT_LOCAL_STACK_PLAN.md)
- [FLYBRAIN_HB_FW_LOCAL_INDEX_QUERY_IMPORT_STRATEGY.md](./FLYBRAIN_HB_FW_LOCAL_INDEX_QUERY_IMPORT_STRATEGY.md)
- [FLYBRAIN_HB_FW_PHASE1_LOCAL_VALUE_EXPERIMENT.md](./FLYBRAIN_HB_FW_PHASE1_LOCAL_VALUE_EXPERIMENT.md)
- [FLYBRAIN_HARNESS_OBSERVABILITY_ALERTING.md](./FLYBRAIN_HARNESS_OBSERVABILITY_ALERTING.md)
- [../artifacts/flybrain/fbh_storage_budget_guardrails_20260922.md](../artifacts/flybrain/fbh_storage_budget_guardrails_20260922.md)

## Local-first baseline constraints (applies to every phase)

1. All writes remain under `LOCI_FLYBRAIN_STORAGE_ROOT` (`graph`, `snapshots`, `cache`, `backups`, `logs` only).
2. Promotion is candidate-first + atomic pointer update (no in-place mutation of active graph state).
3. Fail-closed on `MANIFEST_INVALID`, `INTEGRITY_MISMATCH`, `PROMOTION_STATE_INVALID`, path-policy violations, and unapproved dataset/version pins.
4. No success-shaped fallback: if local capability is unavailable, return explicit contract error or explicit `remote_fallback` source switch.

## Storage pressure thresholds and required actions

Default thresholds come from the current harness guardrail/observability docs and should be treated as the shared-drive default policy until explicitly overridden in config.

| Pressure level | Threshold (used) | Required action | Feature gate impact |
|---|---:|---|---|
| `watch` | 150 GB | Prune stale `cache` artifacts; stop new large downloads. | No phase promotion while above watch for >24h. |
| `soft_cap` | 160 GB | Keep noncritical jobs read-only; retain only active artifacts. | Block enabling new feature tier. |
| `critical` | 180 GB | Freeze nonessential writes; purge `tmp`/staging/scratch; require operator override. | Immediate stop for tier enablement and import promotion. |
| `hard_stop` | 200 GB | Block new writes except minimal safe job closure. | Hard no-go for all tier progression. |

Operational requirement: sustain `< soft_cap` before any tier enablement decision; if pressure is `critical` or `hard_stop`, only recovery/cleanup work is allowed.

## Concrete performance constraints by phase

Numbers below are operator sizing targets for reliable local-first execution. Phase 1 values are direct from the local Neo4j/neuPrint stack plan; higher-phase values are bounded expansion targets that preserve the same safety posture.

| Phase | Feature tier intent | Compute target | Storage target (active + scratch) | Concurrency expectation |
|---|---|---|---|---|
| **Phase 1** | `hb` local graph + `fw` metadata enrichment only | **Recommended:** 16 vCPU / 64 GB RAM (minimum 8 vCPU / 32 GB) | Graph+index 60-90 GB; scratch/import 40-70 GB | 1 active import lane + local query traffic |
| **Phase 2** | Expanded pinned dataset set and broader local adapter coverage (still no full local FlyWire adjacency mirror) | 24 vCPU / 96 GB RAM recommended | 120-180 GB active datasets; 60-120 GB burst scratch (requires aggressive cache eviction windows) | 1 import lane + concurrent query + metadata refresh |
| **Phase 3** | Full local multi-dataset graph posture including FlyWire-scale local graph analytics | 32 vCPU / 128 GB RAM recommended | 250-450 GB active graph/index + 120-250 GB scratch (dedicated volume strongly recommended) | Concurrent imports/promotions with strict staged gates |

If host resources cannot meet the phase target, remain at lower phase and keep higher-tier features disabled.

## Phase stop/go gates

### Gate A: enable Phase 1 feature tier

Phase-1 features become available only after rollout milestones M0-M3 are all green:

- M0-M1: preflight/path guard + deterministic plan/idempotency keys complete.
- M2: snapshot + manifest integrity pass for pinned `hb`/`fw` artifacts.
- M3: first reproducible local query harness run passes with stable replay hashes.

Additional go criteria:

1. Storage pressure below `soft_cap` for enablement window.
2. Promotion smoke pack passes (including intentional negative scope test returning `SCOPE_VIOLATION`).
3. Phase-1 value experiment acceptance passes (`delta_scope_complete_pp >= +20`, no over-claim regression, no integrity/promotion safety breach).

Stop signals (any one = no-go):

- Any hard-fail contract error (`MANIFEST_INVALID`, `INTEGRITY_MISMATCH`, `PROMOTION_STATE_INVALID`).
- Pressure at `critical` or `hard_stop`.
- Non-deterministic replay hash drift for same prompt/config pin.

### Gate B: enable Phase 2 feature tier

Phase 2 may be enabled only when Phase 1 has stable operations and all of these hold:

1. 14-day stability window with no unresolved promotion integrity failures.
2. Storage remains below `watch` after normal retention/cleanup cycles.
3. Expanded dataset pins are explicitly approved and manifested (no implicit scope creep).
4. Query router still emits complete provenance envelope and explicit source switching.

Immediate rollback-to-Phase-1 triggers:

- Two consecutive failed promotions in any newly enabled dataset lane.
- Over-claim or fail-open regression in contract tests.
- Sustained `soft_cap`+ pressure that cannot be relieved by cache/candidate cleanup.

### Gate C: enable Phase 3 feature tier

Phase 3 (full local multi-dataset graph, including FlyWire-scale local analytics) is go only when:

1. Dedicated storage capacity is available (shared-volume 200 GB hard cap is insufficient for durable Phase-3 posture).
2. Cross-dataset promotion and rollback drills are reproducibly passing.
3. Query and provenance contracts remain deterministic under concurrent load.
4. Safety policy still blocks out-of-root writes and unpinned imports under stress.

No-go / hold conditions:

- Any dependency on shared-drive overflow or repeated manual override to bypass budget pressure.
- Inability to keep fail-closed semantics under load (for safety or scope violations).

## Feature-tier switch policy (implementation-facing)

Use a single config/state selector (for example `LOCI_FLYBRAIN_FEATURE_TIER=phase1|phase2|phase3`) backed by automated gate checks:

1. Evaluate milestone state + storage pressure + integrity + contract tests.
2. If all required checks pass, allow tier change.
3. If any stop signal appears, freeze tier change and (if already elevated) auto-downgrade to last known-good tier.

Tier changes must be logged under `${LOCI_FLYBRAIN_STORAGE_ROOT}\logs` with gate evidence references (run id, manifest id, promotion build id, pressure level).

---

Status: concrete phase-gating policy ready to drive implementation and operational go/no-go decisions for local FlyBrain harness rollout.
