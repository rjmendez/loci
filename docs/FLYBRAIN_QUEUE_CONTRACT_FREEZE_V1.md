# FlyBrain Queue Contract Freeze v1

Status: frozen for implementation compatibility review.

This contract defines the exact queue item schema and readiness semantics already supported by the current Loci investigation queue implementation. It is intentionally concrete and compatibility-oriented: it freezes the manifest shape and validation semantics already enforced in `mcp/investigation_tools.py`, while documenting the deferred and retry extension fields needed for FlyBrain task execution without redefining the existing JSON response contract in `mcp/server.py`.

## Source-of-truth behavior in the repo

This freeze is grounded in the current implementation and existing docs:

- `mcp/investigation_tools.py::_coordination_item_from_payload()` validates the core queue item payload and serializes manifold queue-state checks.
- `mcp/investigation_tools.py::_coordination_manifest()` normalizes legacy manifests into `manifest["coordination"]["items"]` and keeps `coordination.version = 1`.
- `mcp/investigation_tools.py::investigation_queue_enqueue()` accepts `item_json` or explicit args and rejects duplicate IDs.
- `mcp/investigation_tools.py::investigation_queue_claim()` enforces lease ownership and expiry semantics.
- `mcp/investigation_tools.py::investigation_queue_complete()` enforces final states and owner checks.
- `mcp/server.py` re-exports these functions and expects machine-friendly JSON strings as return values.
- `mcp/replay_fingerprint.py` canonicalizes FlyBrain provenance metadata using stable key ordering and JSON serialization; this defines the deterministic normalization pattern for replay safety.
- `docs/FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md` specifies the minimum provenance metadata required for FlyBrain-derived facts.

## 1. Canonical queue item envelope

A queue item is a JSON object stored in `manifest["coordination"]["items"]`.

Required fields:

- `id`: string, non-empty, stable unique item identifier within one investigation manifest
- `scope_kind`: string, short stable identifier matching `[A-Za-z0-9_.:-]+`
- `scope_targets`: array of strings with at least one non-empty element
- `state`: one of `queued`, `claimed`, `done`, `blocked`, `cancelled`
- `dependencies`: array of item IDs; may be empty
- `created_at`: RFC3339 timestamp string
- `updated_at`: RFC3339 timestamp string

Optional fields:

- `owner_session`: string or `null`; default `null`
- `lease_expires_at`: RFC3339 timestamp string or `null`; default `null`
- `notes`: string; default `""`
- `dependency_mode`: `all`, `any`, or `none`; default `all`
- `defer_until`: RFC3339 timestamp string or `null`; default `null`
- `retry_policy`: object or `null`; default `null`

Exact v1 schema:

```json
{
  "id": "string",
  "scope_kind": "string",
  "scope_targets": ["string"],
  "state": "queued|claimed|done|blocked|cancelled",
  "owner_session": "string|null",
  "lease_expires_at": "RFC3339|null",
  "dependencies": ["string"],
  "dependency_mode": "all|any|none",
  "defer_until": "RFC3339|null",
  "retry_policy": {
    "enabled": true,
    "max_attempts": 3,
    "initial_backoff_seconds": 30,
    "max_backoff_seconds": 600,
    "backoff_multiplier": 2.0,
    "retryable_states": ["transient", "dependency_blocked", "lease_conflict"],
    "retry_after_seconds": 0
  },
  "notes": "string",
  "created_at": "RFC3339",
  "updated_at": "RFC3339"
}
```

The persisted item shape is the canonical v1 contract. Extra keys may be tolerated for forward compatibility, but they are not part of the v1 readiness contract.

## 2. Deferred semantics

### 2.1 `dependency_mode`

- Type: enum string
- Allowed values: `all`, `any`, `none`
- Default: `all`
- Semantics:
  - `all`: item is eligible only when every dependency is in `done`
  - `any`: item is eligible when at least one dependency is in `done`
  - `none`: dependency gating is disabled; readiness depends on state and `defer_until`
- Validation:
  - must be a lower-case string and one of the enum values
  - if absent, normalize to `all`

This matches the current queue state semantics in `mcp/investigation_tools.py` where dependencies are validated as string lists and must not include the item itself.

### 2.2 `defer_until`

- Type: RFC3339 timestamp string or `null`
- Default: `null`
- Semantics:
  - if set and `now < defer_until`, the item is not ready for execution even if dependencies are satisfied
  - it is an absolute time gate, not a relative delay field
- Validation:
  - if present, it must parse via ISO 8601/RFC3339 with `Z` normalized to `+00:00`
  - invalid values are rejected during enqueue/claim/update

This is the same style of validation already used for `lease_expires_at` in `_coordination_item_from_payload()`.

### 2.3 Retry policy

`retry_policy` is a v1 optional object with these exact fields:

- `enabled`: boolean; default `false`
- `max_attempts`: integer >= 1; default `1`
- `initial_backoff_seconds`: integer >= 0; default `30`
- `max_backoff_seconds`: integer >= `initial_backoff_seconds`; default `600`
- `backoff_multiplier`: number >= 1.0; default `2.0`
- `retryable_states`: array of strings; default `["transient", "dependency_blocked", "lease_conflict"]`
- `retry_after_seconds`: integer >= 0; default `0`

Semantics:

- Retry handling is declarative. The current queue implementation in `mcp/investigation_tools.py` implements only state progression, lease ownership, and finalization; it does not auto-requeue items. This freeze therefore treats `retry_policy` as an executor contract to be evaluated by external code, not as a new mutation method in the existing queue API.
- `retry_after_seconds` is the next eligible delay override. A runner may translate it into a `defer_until` timestamp when re-enqueuing the item.

## 3. Stage I/O schemas

The queue lifecycle is defined as: query -> normalize -> tier -> store -> verify.

### 3.1 Query stage input

This is the external input accepted by `mcp/server.py::investigation_queue_enqueue()` through either `item_json` or explicit function arguments.

```json
{
  "id": "required:string",
  "scope_kind": "required:string",
  "scope_targets": ["required:string"],
  "dependencies": ["string"],
  "dependency_mode": "all|any|none",
  "defer_until": "RFC3339|null",
  "retry_policy": {
    "enabled": "boolean",
    "max_attempts": "integer",
    "initial_backoff_seconds": "integer",
    "max_backoff_seconds": "integer",
    "backoff_multiplier": "number",
    "retryable_states": ["string"],
    "retry_after_seconds": "integer"
  },
  "notes": "string",
  "owner_session": "string|null",
  "state": "queued|claimed|done|blocked|cancelled"
}
```

### 3.2 Normalize stage output

Normalization is performed by `mcp/investigation_tools.py::_coordination_item_from_payload()` and `_coordination_manifest()`.

The canonical normalized item is:

```json
{
  "id": "string",
  "scope_kind": "string",
  "scope_targets": ["string"],
  "state": "queued",
  "owner_session": null,
  "lease_expires_at": null,
  "dependencies": ["string"],
  "dependency_mode": "all",
  "defer_until": null,
  "retry_policy": null,
  "notes": "string",
  "created_at": "RFC3339",
  "updated_at": "RFC3339"
}
```

Normalization rules:

- `id` must be present and non-empty after trimming
- `scope_kind` must be a valid short identifier; invalid values are rejected by `_coordination_item_from_payload()`
- `scope_targets` must contain at least one non-empty string
- `state` must be one of `queued`, `claimed`, `done`, `blocked`, `cancelled`
- `dependencies` must be a list of strings and cannot contain the item itself
- `notes` defaults to `""`
- `owner_session` must be `null` or a trimmed string
- `lease_expires_at` must be `null` or RFC3339
- `dependency_mode`, `defer_until`, and `retry_policy` default to the v1 freeze defaults when absent
- `created_at` and `updated_at` are set by runtime logic, not by caller input

### 3.3 Tier stage output

The tier stage is the readiness classifier derived from the normalized item and the current time.

```json
{
  "eligible": true,
  "ready": true,
  "deferred": false,
  "blocked_by_dependency": false,
  "blocked_by_timer": false,
  "lease_expired": false,
  "state": "queued|claimed|done|blocked|cancelled",
  "owner_session": "string|null",
  "dependency_mode": "all|any|none",
  "defer_until": "RFC3339|null"
}
```

Readiness semantics:

- `eligible`: item is not terminal (`done`, `blocked`, `cancelled`)
- `ready`: item is eligible and not gated by timer or dependency resolution
- `deferred`: true if `defer_until` is in the future or dependency resolution is incomplete under `dependency_mode`
- `lease_expired`: true when `lease_expires_at` is set and that timestamp is in the past; this is the condition already computed by `_coordination_lease_expired()`

### 3.4 Store stage output

Persisted manifest state is the current implementation contract:

```json
{
  "coordination": {
    "version": 1,
    "items": [
      { "<canonical queue item>": "..." }
    ]
  }
}
```

This is created by `_coordination_manifest()` and initialized by `investigation_start()`. The queue must be stored only under `manifest["coordination"]["items"]`.

### 3.5 Verify stage output

The verify stage is the validation response returned by queue claim/status/complete operations and any external executor performing contract validation before execution starts.

```json
{
  "valid": true,
  "errors": [],
  "warnings": [],
  "item": { "<canonical queue item>": "..." },
  "ready": true,
  "owner_session": "string|null",
  "lease_expires_at": "RFC3339|null"
}
```

The validation step must reject malformed JSON, invalid ISO timestamps, invalid states, invalid dependencies, empty IDs, self-dependencies, and malformed retry/defer fields.

## 4. Required invariants and validation rules

The following invariants are mandatory for v1.

1. Queue identity invariance
   - `id` is unique within one investigation manifest
   - duplicate IDs are rejected during enqueue, matching `investigation_queue_enqueue()`

2. Terminal-state invariance
   - only `queued`, `claimed`, `done`, `blocked`, `cancelled` are allowed
   - terminal states are final unless explicit re-enqueue logic is implemented outside the current helper set

3. Session ownership invariance
   - when `owner_session` is set, claim and completion actions must match the same session unless the lease has expired
   - this matches the ownership checks in `investigation_queue_claim()` and `investigation_queue_complete()`

4. Lease expiry invariance
   - if `lease_expires_at` is in the past, the lease is expired and may be reclaimed by a new owner session
   - current logic is in `_coordination_lease_expired()`

5. Dependency invariance
   - an item may not depend on itself
   - dependencies must be arrays of strings, not nested objects
   - `dependency_mode` must be applied before marking an item ready

6. Defer invariance
   - if `defer_until` is set in the future, the item is not ready even when dependencies are satisfied
   - `defer_until` must be absolute and RFC3339-compliant

7. Retry invariance
   - `retry_policy.enabled` implies `max_attempts >= 1`
   - retry policy is declarative and does not mutate the queue state machine
   - `retry_after_seconds` and backoff values must be non-negative

8. Manifest invariance
   - `manifest` must remain JSON-serializable
   - `coordination.version` must remain `1` for v1
   - `coordination.items` must remain a list, not a dict

9. String normalization invariance
   - all string fields must be trimmed before validation
   - empty strings are invalid except for optional `notes`, which defaults to `""`

10. Replay safety invariance
    - queue metadata is execution metadata, not evidence text
    - it must not alter the canonical replay fingerprinting in `mcp/replay_fingerprint.py`; if included in a run record, it must be stored in canonical JSON-stable form

## 5. Compatibility notes with current code

### 5.1 `mcp/investigation_tools.py`

Current queue helpers are the compatibility baseline:

- `_coordination_item_from_payload()` validates the core item fields and rejects malformed JSON or invalid states.
- `_coordination_manifest()` normalizes legacy manifests and injects `coordination.version = 1` plus `coordination.items = [...]`.
- `investigation_queue_enqueue()` accepts a JSON object or string and then calls `_coordination_item_from_payload()`.
- `investigation_queue_claim()` validates `lease_seconds` and reclaims expired leases.
- `investigation_queue_complete()` enforces final states and owner checks.
- `investigation_queue_status()` and `investigation_queue_list()` return `{"investigation_id": ..., "queue": [...], "item_count": ...}`.

Compatibility rule: v1 must accept both the legacy fields and deferred/retry extensions without breaking existing runtime behavior or tests.

### 5.2 `mcp/server.py`

The exported queue functions are expected to remain machine-friendly JSON strings, not Python objects. The exact response forms are:

- `investigation_queue_enqueue(...) -> json.dumps({"queued": true, "item": item}, indent=2)`
- `investigation_queue_claim(...) -> json.dumps({"claimed": true, "item": item}, indent=2)`
- `investigation_queue_complete(...) -> json.dumps({"updated": true, "item": item}, indent=2)`
- `investigation_queue_status(...) -> json.dumps({"investigation_id": ..., "queue": [...], "item_count": ...}, indent=2)`

Compatibility rule: v1 deferred fields must be preserved inside the `item` object without changing the top-level response keys.

### 5.3 `mcp/replay_fingerprint.py`

The FlyBrain provenance helpers canonicalize request and dataset scope and produce deterministic signatures via `normalize_flybrain_provenance()` and `flybrain_replay_fingerprint()`. This is the canonical pattern for stable, replay-safe metadata.

Queue contract compatibility requirements:

- queue item normalization must be deterministic and trimmed before any hashing or persistence
- deferred fields must not be treated as arbitrary free text when generating replay metadata
- if queue metadata is included in a run record, it must be stored in canonical JSON-stable form using the same ordering and canonicalization logic as `_canonicalize()` in `mcp/replay_fingerprint.py`
- a queue item must not silently change identity or dependency semantics between query and verify stages

## 6. Implementation checklist

A v1 implementation is compliant only if all of the following are true:

- enqueue accepts a JSON object or JSON string and normalizes missing fields to the freeze defaults
- `id` remains unique within the investigation manifest
- `state` remains one of the allowed legacy values
- `dependency_mode` is recognized and defaults to `all`
- `defer_until` is validated as RFC3339 when present
- `retry_policy` is validated and preserved once present
- lease ownership and expiry semantics remain as implemented in `mcp/investigation_tools.py`
- `manifest["coordination"]["items"]` remains the durable queue store
- `mcp/server.py` response keys remain stable for callers and tests

## 7. Operational summary

This freeze is intentionally narrow: it defines the exact v1 item envelope, deferred execution semantics, normalization and verification rules, and the compatibility boundaries with the existing queue, server, and FlyBrain provenance behavior already in the repository.

It is not an aspirational redesign. It is the implementation boundary that matches the current Loci code paths today.