# FlyBrain hb/fw phase-1 local value experiment (final focused validation)

## Objective

Run one narrow, evidence-bound experiment that answers a single practical question:

**Does phase-1 local hb graph + fw metadata enrichment provide materially better local answer quality for real user-style connectivity questions than hb-only local querying, without violating safety/provenance constraints?**

This experiment is intentionally constrained to the realistic phase-1 stack already defined in:

- [FLYBRAIN_HB_FW_LOCAL_INDEX_QUERY_IMPORT_STRATEGY.md](./FLYBRAIN_HB_FW_LOCAL_INDEX_QUERY_IMPORT_STRATEGY.md)
- [FLYBRAIN_HARNESS_ROLLOUT_MILESTONES.md](./FLYBRAIN_HARNESS_ROLLOUT_MILESTONES.md)
- [FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md](./FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md)
- [FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md](./FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md)

## Narrow hypothesis

For a fixed set of local, reproducible connectivity tasks, enabling `fw` metadata enrichment on top of promoted `hb` local graph will:

1. Increase **scope-complete responses** (responses that include required dataset/version/sex/stage/anatomy/provenance fields) by at least **20 percentage points** over hb-only local responses.
2. Keep **boundary safety guarantees unchanged** (no increase in out-of-scope claims, no fail-open success-shaped fallbacks, no promotion/integrity violations).

This is a value hypothesis about answer utility and trustworthiness under local use, not a throughput benchmark and not a full FlyWire graph test.

## Fixed data scope (phase-1 only)

### Dataset/version pins

- `hb=neuprint_JRC_Hemibrain_1point2point1` (local graph backbone)
- `fw=flywire783` (metadata enrichment only)

### Allowed capabilities

- `hb`: local structural connectivity/traversal queries against promoted graph pointer.
- `fw`: local entity normalization + annotation/class metadata hydration.

### Explicitly disallowed

- Full local FlyWire adjacency/synapse graph claims.
- Cross-sex/cross-stage generalization without direct matching evidence.
- Any writes outside `LOCI_FLYBRAIN_STORAGE_ROOT`.
- Remote fallback presented as local evidence.

## Experiment design

### Task set

Use exactly **20 deterministic user-style prompts**, each mapped to a fixed query contract and replay fingerprint. Keep prompts in-scope for hemibrain structural questions where metadata disambiguation can matter (ID normalization, class labeling, scope wording).

Each prompt is executed in two modes:

- **Mode A (control):** hb-only local adapter.
- **Mode B (treatment):** hb local adapter + fw metadata enrichment.

All non-adapter variables remain fixed: same pinned versions, same query payload, same run environment, same response schema validator.

### Response quality rubric (deterministic)

Score each response on binary checks:

1. `has_result_contract` (count_status, row_count, warnings)
2. `has_scope_tags` (sex, stage, anatomy_scope)
3. `has_dataset_version` (dataset symbol + version id)
4. `has_provenance` (manifest ref + replay_fingerprint)
5. `no_overclaim` (no unsupported global/full-brain/cross-stage language)

`scope_complete = 1` only if checks 1-4 pass.

## Runbook (single-operator, local)

1. **Preflight and policy gate (must pass)**
   - Validate `LOCI_FLYBRAIN_STORAGE_ROOT` and path guards.
   - Validate phase-1 scope pins (`hb`,`fw`) and versions.
   - Confirm promoted hb pointer exists and fw metadata snapshot is manifest-valid.
2. **Freeze test pack**
   - Persist `prompt_pack.json` and expected contract schema under run log directory.
   - Assign immutable `experiment_run_id` and config fingerprint.
3. **Execute control (Mode A)**
   - Run all 20 prompts with hb-only path.
   - Persist per-prompt response, structured rubric results, and replay fingerprint.
4. **Execute treatment (Mode B)**
   - Run same 20 prompts with hb+fw enrichment.
   - Persist same telemetry fields.
5. **Compute metrics and deltas**
   - `scope_complete_rate_A`, `scope_complete_rate_B`
   - `delta_scope_complete_pp = B - A` (percentage points)
   - `overclaim_rate_A/B`
   - `boundary_error_count_A/B` for explicit hard-stop codes.
6. **Acceptance decision**
   - Evaluate against acceptance criteria below.
   - Emit decision artifact: `experiment_decision.json` with pass/fail reasons.

All logs/artifacts must be written under:

- `$LOCI_FLYBRAIN_STORAGE_ROOT\logs\harness\experiments\hb-fw-value\<experiment_run_id>\`

## Acceptance criteria

Experiment is **accepted** only if all conditions are met:

1. `delta_scope_complete_pp >= +20`
2. `overclaim_rate_B <= overclaim_rate_A`
3. `boundary_error_count_B == 0` for:
   - `MANIFEST_INVALID`
   - `INTEGRITY_MISMATCH`
   - `PROMOTION_STATE_INVALID`
4. `SCOPE_VIOLATION` checks remain deterministic (negative tests fail closed, never success-shaped).
5. 100% of treatment responses include provenance envelope fields required by phase-1 contract.

If any condition fails, keep hb-only behavior as default and file targeted remediation before rerun.

## Measurable failure modes (must be reported, not masked)

| Failure mode | Measurement | Fail threshold | Expected handling |
|---|---|---:|---|
| Scope metadata regression | `scope_complete_rate_B - scope_complete_rate_A` | `< +20pp` | Mark experiment failed; keep hb-only default. |
| Over-claim regression | `overclaim_rate_B - overclaim_rate_A` | `> 0` | Fail; require router/response boundary fix. |
| Provenance omission | `% responses missing manifest or replay_fingerprint` | `> 0%` | Fail; block adoption. |
| Integrity/promotion safety breach | count of `MANIFEST_INVALID`, `INTEGRITY_MISMATCH`, `PROMOTION_STATE_INVALID` | `> 0` | Hard-stop experiment; investigate substrate. |
| Fail-open behavior | count of success-shaped responses where error code required | `> 0` | Hard-stop; contract violation. |
| Non-deterministic replay | hash mismatch for same prompt/config | `> 0` | Fail; stabilize query or serialization path. |

## Evidence-bound operational constraints enforced

- Fail closed on all contract/safety violations.
- No implicit dataset broadening beyond pinned hb/fw versions.
- No cross-stage/sex generalization language without direct evidence.
- No writes outside `LOCI_FLYBRAIN_STORAGE_ROOT`.
- All decisions backed by persisted artifacts (`prompt_pack`, per-prompt outputs, rubric scores, decision JSON).

## Deliverables

At experiment completion, persist:

1. `prompt_pack.json`
2. `mode_a_results.jsonl`
3. `mode_b_results.jsonl`
4. `metrics_summary.json`
5. `experiment_decision.json`

## Decision rule for phase-1 stack

- **Pass:** enable hb+fw enrichment for phase-1 local query path where supported.
- **Fail:** keep hb-only default, open explicit remediation issue(s), rerun with same prompt pack after fixes.

This keeps the decision practical, bounded, and reproducible under current local constraints.

---

Status: final focused phase-1 experiment definition for validating hb/fw local value under real local use.
