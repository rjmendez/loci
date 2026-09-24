# FLYBRAIN_QUEUE_STATE_MACHINE_SPEC

This document defines the canonical coordination-queue state machine used by Loci investigation work items.
It is intentionally aligned with the semantics implemented in `mcp/investigation_tools.py` and the queue tests in `mcp/tests/test_coordination_queue.py`.

## 1. Scope and source of truth

- Queue items are stored in `manifest["coordination"]["items"]`.
- `investigation_queue_enqueue(...)` creates a queue item.
- `investigation_queue_claim(...)` claims or renews a lease.
- `investigation_queue_complete(...)` finalizes a queue item as `done`, `blocked`, or `cancelled`.
- `investigation_queue_release(...)` is an alias for `state="blocked"`.
- The manifest is the source of truth for `id`, `scope_kind`, `scope_targets`, `state`, `owner_session`, `lease_expires_at`, `dependencies`, `notes`, `created_at`, and `updated_at`.

Important: the queue API does not currently enforce dependency resolution automatically. `dependencies` is persisted as item metadata, but the actual unlock rule is a scheduler policy decision. This spec therefore defines the canonical policy model while preserving the implementation semantics: the queue stores the dependency list and enforces ownership/lease rules, but it does not auto-mutate an item's state based on dependency completion.

## 2. Canonical state set

The queue lives in exactly these states:

- `queued`: available to be claimed.
- `claimed`: reserved by a session with a live lease.
- `done`: terminal success state.
- `blocked`: terminal blocked/unavailable state.
- `cancelled`: terminal cancelled state.

The implementation normalizes unknown or invalid states to `queued` during enqueue, and final states are only entered through `investigation_queue_complete(...)`.

## 3. State machine

### Transition table

| From | Event | To | Allowed? | Conditions / notes |
|---|---|---|---|---|
| -- | `enqueue` | `queued` | Yes | Item is created with stable `id`, valid `scope_kind`, non-empty `scope_targets`, and optional `dependencies`. Duplicate IDs are rejected. |
| `queued` | `claim(owner_session, lease_seconds)` | `claimed` | Yes | `owner_session` must be non-empty; no other owner may hold a live lease. |
| `queued` | `complete(state=done|blocked|cancelled)` | `done` / `blocked` / `cancelled` | Yes | No current owner required if none exists. |
| `claimed` | `claim(owner_session=same_owner, lease_seconds)` | `claimed` | Yes | Same-owner renewal/heartbeat. Lease expiry is reset to `now + ttl`. |
| `claimed` | `claim(owner_session=other_owner, lease_seconds)` | `claimed` | Conditional | Only if the current lease is expired; otherwise rejected. This is the reclaim path. |
| `claimed` | `complete(state=done|blocked|cancelled, owner_session=same_owner)` | final | Yes | Completion is allowed only by the current owner. |
| `claimed` | `complete(..., owner_session=other_owner)` | unchanged | No | Rejected with `"owned by session ... and cannot be completed"` semantics. |
| `claimed` | `complete(..., owner_session=None)` | unchanged | No | Rejected when current owner exists and no owner is supplied. |
| `done` | `claim(...)` | unchanged | No | Final states are not claimable. |
| `blocked` | `claim(...)` | unchanged | No | Final states are not claimable. |
| `cancelled` | `claim(...)` | unchanged | No | Final states are not claimable. |
| `done` | `complete(state=done)` | `done` | Yes | Idempotent completion of the same final state is allowed. |
| `blocked` | `complete(state=blocked)` | `blocked` | Yes | Idempotent completion of the same final state is allowed. |
| `cancelled` | `complete(state=cancelled)` | `cancelled` | Yes | Idempotent completion of the same final state is allowed. |
| `done` | `complete(state=blocked|cancelled)` | unchanged | No | Changing final state after completion is rejected. |
| `blocked` | `complete(state=done|cancelled)` | unchanged | No | Same rule: final state is sticky. |
| `cancelled` | `complete(state=done|blocked)` | unchanged | No | Same rule: final state is sticky. |

### Transition diagram

```text
queued
  ├─ claim(owner) --------------------> claimed
  ├─ complete(done|blocked|cancelled) -> done|blocked|cancelled
  └─ invalid or duplicate input ----> error

claimed
  ├─ claim(same owner) -------------> claimed (renew/heartbeat)
  ├─ claim(other owner, expired) --> claimed (reclaim)
  ├─ complete(same owner) ----------> done|blocked|cancelled
  └─ complete(other owner/none) --> error

final
  ├─ complete(same final state) ---> same final state (idempotent)
  └─ claim / state change ---------> error
```

## 4. Lease semantics

### Lease lifecycle

- Each claim sets `owner_session` and `lease_expires_at = now + lease_seconds`.
- Lease expiry is computed against the current UTC time via `_coordination_lease_expired(item)`.
- A lease is expired when `lease_expires_at <= now`.
- Renewal is the same as a claim by the same owner: it extends the lease without changing the logical state from `claimed` to anything else.
- `lease_expires_at` is cleared to `null` when the item reaches a terminal state.

### Reclaim semantics

A different session may reclaim an item only when all of the following are true:

1. the item is not already in a terminal state,
2. the current owner differs from the new owner,
3. the current lease is expired,
4. the new owner is valid and non-empty.

If the lease is still active, the claim is rejected with the existing-owner error. This is the exact behavior implemented in `investigation_queue_claim(...)`.

### Reclaim examples

- Example A: `session-a` holds a claim for 60s. `session-b` attempts a claim while `lease_expires_at` is still in the future. Result: reject; item remains claimed by `session-a`.
- Example B: `session-a` holds a claim for 1s, then the manifest is manually or automatically set to a timestamp in the past. `session-b` claims it. Result: success; `owner_session` becomes `session-b`, `state` remains `claimed`, and a fresh lease is created.

## 5. Retry and backoff behavior

The queue itself does not implement a scheduler-level retry loop. It provides the primitives required for callers to implement one.

Canonical retry semantics are:

- `claim` failure due to an active lease is a soft conflict, not a queue corruption event.
- The caller must back off and retry after the current owner’s lease expires or the owner yields the item.
- The queue makes the retry decision deterministic by using lease expiry as the state transition boundary.
- Retrying a `complete` call with the same final state is idempotent and should be treated as a no-op.
- Retrying a `claim` by the same owner is safe and renews the lease.

Recommended backoff policy:

- First conflict: retry in ~1x lease granularity or immediate short jitter.
- Repeated conflict: exponential backoff with cap, e.g., 1s -> 2s -> 4s -> 8s; do not exceed the current lease horizon unless the item is known to be stale.
- Reclaim after expiry: immediate re-attempt once the lease has expired and the item is visible as `claimed` with an expired timestamp.
- Never use backoff to bypass a valid owner check. Ownership enforcement is authoritative until expiry.

## 6. Dependency unlock behavior

### Storage behavior

`dependencies` is a list of item IDs stored on the item. It is accepted as a list or comma-separated string and validated as such. The implementation does not auto-resolve or auto-unlock the item based on dependency completion.

### Canonical scheduler rule

The canonical queue semantics define dependency unlock as a policy layer outside the raw queue API:

- `all`: the dependent item is eligible to be claimed only after every dependency item reaches a terminal state, typically `done`.
- `any`: the dependent item is eligible once at least one dependency item reaches a terminal state.

For compatibility with the queue implementation, the item record simply stores the dependency list; the scheduler or orchestration layer decides whether to interpret it as `all` or `any`.

Recommended default:

- Default to `all` for pipeline stages where correctness is more important than throughput.
- Use `any` only for opportunistic or fan-out work where a partially successful dependency graph is sufficient.

Operationally, the queue state machine should treat dependency completion as a gating condition for transitioning from `queued` to `claimed`, not as an internal queue state. The implementation does not currently enforce that gate automatically, so a caller must check dependency completion before calling `claim` or before promoting an item out of `queued`.

### Example dependency graph

```text
A -> B
A -> C
B -> D
C -> D
```

- With `all`: `D` remains `queued` until `B` and `C` are both terminal (`done` or otherwise accepted by scheduler policy).
- With `any`: `D` may become claimable as soon as either `B` or `C` is terminal.

## 7. Invalid transitions and validation rules

The queue rejects malformed inputs and invalid transitions explicitly.

### Input validation enforced by the implementation

- `id` must exist and be non-empty.
- `scope_kind` must be a short stable identifier (regex: `[A-Za-z0-9_.:-]+`).
- `scope_targets` must be a non-empty list of target IDs or a comma-separated string that decodes to a list.
- `dependencies` must be a list of item IDs or parseable string representation.
- `state` must be one of `queued`, `claimed`, `done`, `blocked`, `cancelled`.
- `lease_expires_at` must be an ISO 8601 datetime string when provided.
- `lease_seconds` must be a finite positive number.
- Duplicate item IDs are rejected on enqueue.

### Invalid state transitions

Invalid transitions are rejected with JSON error results containing an `"error"` field. Representative cases include:

- claim while another owner owns an active lease,
- claim a final item (`done`, `blocked`, or `cancelled`),
- complete a claimed item without providing the current owner,
- complete with the wrong owner while another lease is still valid,
- move a final item into a new final state,
- final-state completion with a different state than the one already held.

### Enqueue normalization

`investigation_queue_enqueue(...)` intentionally normalizes a supplied non-queued/claimed state into `queued`:

```python
if item['state'] not in {'queued', 'claimed'}:
    item['state'] = 'queued'
```

This means the canonical enqueue state is always `queued` unless a caller explicitly sets `claimed` before enqueue, which is not the normal path. Final states are reached only via completion.

## 8. Idempotency rules

The queue is idempotent where the state and ownership semantics are stable:

- `enqueue` of the same `id` twice is rejected; the first item wins.
- `claim` by the same owner is idempotent for lease renewal; it updates `lease_expires_at` and `updated_at` and keeps the item in `claimed`.
- `complete` with the same final state (`done`/`blocked`/`cancelled`) is idempotent and should be treated as a no-op to callers. The implementation allows the same-state re-completion if ownership checks pass.
- `complete` with a different final state from the current terminal state is rejected; once final, the item is sticky.
- `claim` of a final item is rejected outright.

## 9. Example flows

### Happy path

```text
enqueue item-42 (queued)
  -> claim item-42 by session-a (claimed, lease_expires_at = now + 300s)
  -> complete item-42 by session-a with state=done (done, lease_expires_at = null)
```

### Lease renewal

```text
queue item-42 (queued)
  -> claim by session-a with 300s
  -> claim by session-a with 450s (renew lease; still claimed)
  -> complete by session-a with state=done
```

### Rejected takeover while lease is active

```text
queue item-42 (queued)
  -> claim by session-a with 60s
  -> claim by session-b with 10s while lease is active
  -> rejected: another session already owns the active lease
```

### Reclaim after expiry

```text
queue item-42 (queued)
  -> claim by session-a with 1s
  -> manually set lease_expires_at to a past timestamp
  -> claim by session-b with 10s
  -> accepted: session-b becomes owner, item remains claimed
```

### Final-state idempotency

```text
queue item-42 (queued)
  -> complete by session-a with state=blocked
  -> complete by session-a with state=blocked again
  -> accepted as idempotent no-op

  -> complete with state=done
  -> rejected: final state is sticky and state cannot change after completion
```

## 10. Canonical operational guidance

1. Enqueue items with a stable ID and explicit `dependencies` metadata.
2. Claim only when the item is `queued` or you are renewing an existing lease owned by the same session.
3. Treat `lease_expires_at` as the authoritative ownership boundary.
4. Use a backoff policy on claim conflicts while the current lease remains valid.
5. Treat final states (`done`, `blocked`, `cancelled`) as sticky and idempotent.
6. Do not rely on automatic dependency unlocks unless a scheduler layer implements and enforces `all` or `any` semantics outside of the raw queue engine.

This spec is the canonical contract for queue state transitions and should be used for implementation, review, and debugging. Any future queue feature that changes this behavior must be reflected in both the code and this document.
