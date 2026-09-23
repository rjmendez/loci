# Sleep Consolidation Scheduler Spec

This defines a practical scheduler capability that turns the current Loci memory jobs into one coordinated loop for periodic consolidation, memory grooming, and background maintenance.

The design is FlyBrain-derived: **fast reflex loop** (cheap probes and immediate cleanup) + **slow consolidation loop** (deeper synthesis and quality checks) + **homeostatic guardrails** (health checks, decay, and backpressure).

## Existing integration hooks (already present)

Scheduler runtime and state:

- `scripts/hermes_cron_runner.py` — 1-minute tick runner, lock-protected, persists `next_run_at` and collapses backlog to one catch-up run.
- `cron/jobs.json` — current interval jobs and script/agent dispatch contract.

Consolidation hooks:

- `scripts/mnemosyne_activity_check.py` — trigger probe (silent when idle, emits when `working_memory` grows, warns when a bank is unmonitored).
- `scripts/mnemosyne_sleep_all.sh` and `scripts/mnemosyne_consolidate.sh` — bank-level sleep/consolidate executors.
- `mcp/server.py::memory_consolidate()` — canonical sleep pass plus optional causal inference and advisory quality audit.

Memory grooming hooks:

- `scripts/loci_groom.py` passes:
  - applyable: `index`
  - proposal-only: `tags`, `recall`, `knn_tags`, `codelink`, `verify`, `reflect`, `summaries`
- `scripts/loci_groom_cron.sh` — unattended wrapper, preserves refusal/degraded semantics (exit `3`) and run logs.

Background maintenance hooks:

- `scripts/mnemosyne_qdrant_sync.py` — sqlite → Qdrant reconciliation.
- `mcp/server.py::memory_health()` — substrate health.
- `mcp/server.py::memory_self_check()` — advisory provenance/contradiction checks.
- `mcp/server.py::reflection_loop_seed()` + `reflection_loop_tick()` — bounded self-reflection queue processing.

## Scheduler capability

Add one orchestration policy layer (can be implemented as a new script, e.g. `scripts/loci_sleep_scheduler.py`, or folded into the existing cron runner):

1. **Sense (every 5m):** read trigger signals.
2. **Reflex (0-5m response):** run low-cost cleanup when trigger thresholds trip.
3. **Consolidate (20-60m):** run deeper memory consolidation and selected grooming.
4. **Homeostasis (hourly/daily):** run integrity and maintenance checks; throttle expensive passes.

### Trigger signals

Use these concrete signals:

- `working_memory_grew`: from `mnemosyne_activity_check.py` output.
- `reflection_backlog`: from `reflection_loop_status.queue_size`.
- `substrate_degraded`: from `memory_health.status in {degraded, unhealthy}`.
- `contradiction_pressure`: count from `memory_self_check` contradiction verdicts.
- `queue_pressure`: active claimed/blocked ratio from `investigation_queue_status`.
- `sync_lag`: ratio of missing indexed findings from `loci_groom.py index` dry run report.

### Cadence and actions

| Lane | Cadence | Trigger | Action |
|---|---:|---|---|
| Reflex-sleep | every 5m | `working_memory_grew=true` | Run `memory_consolidate(dry_run=false)` once; skip when false |
| Reflection-drain | every 5m | `reflection_backlog > 0` | `reflection_loop_tick(max_items=3..10)` with backlog-scaled bound |
| Consolidation-heavy | every 20m | `working_memory_grew=true` OR `reflection_backlog >= 25` | Run `memory_consolidate`; then groom `index --apply`; then proposal passes `summaries`, `codelink`, `knn_tags` |
| Sync-reconcile | every 30m | always | Run `mnemosyne_qdrant_sync.py` |
| Health-audit | hourly | always | Run `memory_health`; if degraded run `retrieval_selftest` and emit alert finding |
| Contradiction-watch | every 6h | always | Run `memory_self_check(checks=provenance,contradiction)` advisory only |
| Daily-homeostasis | daily | always | Run `loci_groom.py reflect` and `summaries` (bounded), plus decay/consolidation audit review |

## Guardrails (required)

1. **No replay storms:** preserve `hermes_cron_runner` behavior of one catch-up run + persisted `next_run_at`.
2. **Single-writer tick lock:** keep lock-file semantics for scheduler tick execution.
3. **Bounded work per tick:** respect existing bounds (`max_items`, per-run pass budgets, timeout ceilings).
4. **Fail-open, explicit degraded state:** if optional audit/inference fails, return degraded metadata, not silent success.
5. **Safe indexing precondition:** honor `loci_groom.connect()` refusal when retention is non-zero; never bypass.
6. **Do not auto-schedule `verify` pass:** keep unscheduled due measured false-refutation risk in operations docs.
7. **Quiet idle contract:** no-op lanes stay silent when no trigger to prevent token and log burn.
8. **State persistence:** write per-run status (`last_status`, `last_error`, run summaries) on the same tick that executes.

## Practical implementation notes

- Keep `cron/jobs.json` as the execution backend but add a scheduler policy phase before dispatch that:
  - computes trigger signals;
  - decides whether each job should fire now, defer, or downshift;
  - records decision rationale in run metadata.
- For minimal disruption, start with a policy wrapper that invokes existing scripts/tools; avoid changing each pass implementation.
- Treat this as an orchestration layer only; do not change tool-level trust semantics (consolidation is not promotion, advisory checks stay advisory).

## Acceptance criteria

- Consolidation runs only when activity/backlog justifies it, not purely on fixed interval.
- Grooming and reflection drain operate under bounded budgets and never starve consolidation.
- Health and contradiction checks run periodically and can trigger maintenance escalation.
- Idle periods remain silent and cheap; busy periods increase throughput without replay storms.
- Every scheduler decision is inspectable from persisted run metadata.
