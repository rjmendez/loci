# Hybrid Lane Router

This document defines the first mergeable implementation slice for local-lane parallelism: a deterministic router that chooses between Copilot lanes and local model lanes using a shared dispatch contract.

## Scope

Implemented by `scripts/hybrid_lane_router.py`:

- validates a `dispatch_request` envelope,
- chooses a lane deterministically (`copilot-critical`, `copilot-general`, `local-batch`, `local-speculative`, `local-recovery`),
- emits machine-readable routing receipts from CLI mode.

## Routing Rules

1. Critical-path tasks always route to `copilot-critical`.
2. Integration/tooling tasks route to `copilot-general`.
3. Reasoning/fanout tasks route to `local-batch`.
4. Speculative tasks route to `local-speculative` only when upstream confidence is `>= 0.80`; otherwise fallback to `local-batch`.
5. Brownout L3/L4 degrades to local-only behavior for reasoning classes; unhealthy lanes route to `local-recovery`.

## Dispatch Contract

Required request fields:

- `task_id`, `session_id`, `priority`, `critical_path`
- `task_class`, `idempotency_key`
- `max_latency_ms`, `max_cost_usd`, `quality_floor`
- `evidence_required`

## CLI Example

```bash
python scripts/hybrid_lane_router.py --request-json req.json --confidence 0.85 --brownout-level L1
```

Output:

```json
{"ok": true, "lane_id": "local-speculative", "reason": "speculation_ok", "degraded": false}
```
