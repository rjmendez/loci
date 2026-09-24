# FlyBrain Replay Acceptance Test Spec

This document defines the golden acceptance scenarios for FlyBrain replay/provenance integrity. It is intentionally small, deterministic, and aligned with the current helper/tests in `mcp/replay_fingerprint.py` and `mcp/tests/test_replay_fingerprint.py`.

## Scope

The acceptance surface covers four invariants:

1. claim_scope round-trip integrity
2. flybrain_provenance normalization and persistence
3. result_contract consistency (`count`, `count_status`, `warnings`)
4. replay_fingerprint stability and drift detection

These scenarios are the minimum required before a FlyBrain query result is considered replay-safe.

## Golden scenario 1: claim_scope round-trip integrity

### Preconditions
- A FlyBrain result is stored as a finding with `metadata.claim_scope`.
- The payload contains the required tuple fields: `dataset`, `dataset_version`, `sex`, `life_stage`, `annotation_completeness`, `circuit_class`, and `experience_window`.
- The source tool is a FlyBrain tool (`virtual-fly-brain-*`).

### Acceptance criteria
- The exact `claim_scope` object survives a store/load cycle without mutation.
- The same `claim_scope` is also present under `metadata.flybrain_provenance.claim_scope` when the provenance envelope is normalized by the investigation store path.
- Semantically equivalent payloads remain equal after canonicalization (sorting, Unicode NFC normalization, deduplicating set-like fields).

### Canonical example

```python
claim_scope = {
    "dataset": "fw",
    "dataset_version": "flywire783",
    "sex": "female",
    "life_stage": "adult",
    "annotation_completeness": 0.95,
    "circuit_class": "Kenyon cell",
    "experience_window": "naive",
}
```

### Expected result
- `investigation_store(..., metadata={"claim_scope": claim_scope, "flybrain_provenance": {...}})` succeeds without error.
- `investigation_load(...)` returns the same `claim_scope` values.
- The persisted `metadata.flybrain_provenance.claim_scope` equals the original tuple.

## Golden scenario 2: flybrain_provenance normalization and persistence

### Preconditions
- A FlyBrain provenance block is supplied with equivalent-but-noncanonical request and dataset ordering (e.g. list order reversed, set-like values in different orders, Unicode normalization differences).

### Acceptance criteria
- `normalize_flybrain_provenance()` canonicalizes the payload deterministically.
- Fields are sorted and normalized for stable JSON hashing.
- `excluded_symbols` and `version_ids_seen` are deduplicated and ordered deterministically.
- The normalized envelope persists through both the finding store and mnemo recall path.

### Canonical example

```python
flybrain_provenance = {
    "tool_name": "query_connectivity",
    "tool_variant": "virtual-fly-brain-query_connectivity",
    "request": {
        "upstream_type_input": "FBbt_00003686",
        "exclude_dbs": ["hb", "fafb"],
        "group_by_class": True,
    },
    "dataset_scope": {
        "included_symbols": ["fw", "mc"],
        "excluded_symbols": ["hb", "fafb"],
        "version_ids_seen": ["flywire783", "male_cns_v1_0"],
    },
    "result_contract": {
        "count": 1858,
        "count_status": "exact",
        "returned_rows": 5,
        "warnings": [],
    },
}
```

### Expected result
- `dataset_scope["excluded_symbols"]` normalizes to a canonical sorted order used by the helper.
- `replay_fingerprint` is present and matches the persisted `finding["flybrain_replay_fingerprint"]`.
- The normalization is stable across re-reads and mnemo recall.

## Golden scenario 3: result_contract consistency

### Preconditions
- A FlyBrain output payload is parsed by `flybrain_audit_fingerprint()`.
- The raw output contains `count`, `count_status`, and warnings as returned by the tool.

### Acceptance criteria
- `result_contract.count` and `result_contract.count_status` are retained exactly.
- `warnings` is normalized to a canonical list, never left as a raw non-list object.
- The returned `count_status` value matches the result contract in the normalized provenance envelope.

### Canonical example

```json
{
  "count": 14,
  "count_status": "exact",
  "rows": [{"db": "fw", "short_form": "flywire783"}],
  "warnings": ["dataset skewed toward fw"]
}
```

### Expected result
- `flybrain_audit_fingerprint(...)["flybrain_provenance"]["result_contract"]["count_status"] == "exact"`
- `warnings` is a list, even when a caller provides an equivalent list-like value.
- The persisted provenance is consistent with the raw output contract.

## Golden scenario 4: replay_fingerprint stability and drift detection

### Preconditions
- Two FlyBrain requests are semantically identical but supplied in different key orders or with equivalent list/set ordering.
- A third request intentionally changes a materially relevant field (e.g. dataset version, upstream type, or exclusion list) or mutates a stability-critical flag.

### Acceptance criteria
- Equivalent payloads produce the same `flybrain_replay_fingerprint`.
- Drift in request or dataset scope produces a different fingerprint.
- The fingerprint is deterministic and hash-based rather than timestamp-based.

### Canonical example

```python
request_a = {
    "upstream_type_input": "FBbt_00003686",
    "exclude_dbs": ["hb", "fafb"],
    "group_by_class": True,
}
request_b = {
    "group_by_class": True,
    "exclude_dbs": ["fafb", "hb"],
    "upstream_type_input": "FBbt_00003686",
}
```

### Expected result
- `flybrain_replay_fingerprint(..., request_a, scope) == flybrain_replay_fingerprint(..., request_b, scope)`
- A changed dataset version or scope yields a different fingerprint.
- The result is stable across equivalent round-trips and sensitive to real drift.

## Acceptance gate

A FlyBrain replay scenario passes when all four golden scenarios are satisfied in the current test harness without changing the API contract of the existing helpers.

The recommended minimal validation command is:

```bash
pytest mcp/tests/test_replay_fingerprint.py -q
```

If a FlyBrain provenance persistence scenario is also being exercised via the investigation store path, the broader but still targeted selection is:

```bash
pytest mcp/tests/test_replay_fingerprint.py mcp/tests/test_flybrain_provenance_envelope.py -q
```
