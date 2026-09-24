# Loci implementation verification and rollout

This document ties the implemented Loci changes into one verification and rollout path. It is meant to be read alongside `scripts/bench_model_catalog_quality.py`, `scripts/assign_models_from_benchmark.py`, `docs/REASONING_POLICY_SPEC.md`, `mcp/tests/test_openrouter.py`, and the audit-lane tests in `mcp/tests/test_audit_lane_status.py` / `mcp/tests/test_audit_log_is_isolated.py`.

## 1) Benchmark harness spec

The benchmark harness already exists in `scripts/bench_model_catalog_quality.py`. It is intentionally model- and role-aware:

- `cheap_fanout` cases validate extraction, classification, and routing-style short outputs
- `guardian` cases validate allow/block decisions and harmful-content gating
- `escalation` cases validate evidence-based reasoning and contradiction handling
- `synthesis` cases validate final answer quality and determinism

Run it against a live local endpoint:

```bash
python3 scripts/bench_model_catalog_quality.py \
  --base-url http://<ollama-host>:11434 \
  --timeout-s 45 \
  --max-tokens 96 \
  --output artifacts/model_catalog/quality_<date>.json
```

This emits a JSON summary with `summary[*].quality_winners` per role. The winner list is used as the source of truth for role assignment, not a human guess.

`scripts/assign_models_from_benchmark.py` turns the winning models into a TOML fragment for `~/.loci/backends.toml`:

```bash
python3 scripts/assign_models_from_benchmark.py \
  --benchmark-json artifacts/model_catalog/quality_<date>.json
```

Required acceptance gates:

- `bench_model_catalog_quality.py` returns the expected role coverage and no duplicate IDs
- `assign_models_from_benchmark.py` exits `0` only when each selected winner is installed locally
- `ollama show <selected-tag>` resolves for all model roles before rollout
- `python3 -m pytest scripts/tests/test_bench_model_catalog_quality.py scripts/tests/test_assign_models_from_benchmark.py -q` passes

## 2) Phased rollout plan

The implementation-ready rollout for local and OpenRouter-backed reasoning is:

### Phase 0: dry-run and benchmark-only

- Validate the benchmark JSON is reproducible and role winners are stable across three runs
- Confirm the selected tags exist locally; do not change `~/.loci/backends.toml` yet
- Keep all routing in the current default model while logging the benchmark result only
- Guardrail: if any role has no winner, stop before enabling anything

### Phase 1: shadow routing

- Put the new generation/verification pair in shadow mode only
- Use the same input, but do not let the new route mutate state or ship external output
- Log router decision, chosen model, response status, and whether verification passed
- Guardrail: blocked or rate-limited responses are ignored unless the model passes the same JSON/cost checks on a second attempt

### Phase 2: canary by workload

- Enable only for a single low-risk workflow or a low-traffic MCP call path
- Keep `max_tokens` cap, json-only outputs where required, and `OpenRouter` ladder retries for rate-limited upstreams only
- Adopt a rollout limit (for example: 5-10% of traffic or one queue/tenant at a time)
- Guardrail: if `degraded: true` or `status` indicates upstream rate limit or malformed JSON, the request falls back to the previous stable tier with audit logging

### Phase 3: default-on with monthly review

- Promote the benchmark winner to the default role mapping after canary success
- Keep `mcp/openrouter.py` in a fail-open ladder with `available()` and `generate_batch()` behavior unchanged
- Continue to record latency, error status, and cost per model
- Guardrail: do not silently weaken the policy gate; any failure that breaks the benchmark minimum reverts to the last known-good configuration

### Phase 4: rollback readiness

- Keep the previous `[ollama]` settings in a dated backup
- The rollback condition is any of: benchmark winner drops below the minimum for a role, JSON parser fails at a sustained rate, or verification outcomes trend worse than the previous route
- Rollback command pattern:

```bash
cp ~/.loci/backends.toml ~/.loci/backends.toml.bak.$(date +%Y%m%d-%H%M%S)
# restore the prior TOML block and restart the MCP/service process
```

## 3) Acceptance and verification checks

The implementation is ready to ship only when all of the following checks are green. These are the current repo gates that should be run before enabling a new model assignment or routing change:

```bash
python3 -m pytest scripts/tests/test_bench_model_catalog_quality.py scripts/tests/test_assign_models_from_benchmark.py -q
python3 -m pytest mcp/tests/test_openrouter.py -q
python3 -m pytest mcp/tests/test_audit_lane_status.py mcp/tests/test_audit_log_is_isolated.py -q
python3 -m pytest deep_think_loci/tests/test_deep_think_loci_workflow_guards.py -q
```

Operational acceptance checklist:

- benchmark summary includes all required roles
- no selected winner is missing from the local `ollama list` output
- OpenRouter free/paid ladder retries on 429 without raising or losing prompt alignment
- JSON-mode outputs are extracted from reasoning prose and rejected when no object is present
- audit lane reports the real state (`empty`, `present`, or `degraded`) rather than silently hiding it
- memory writes and verification verdicts remain isolated to the active run

## 4) Audit-trace architecture

The audit trace is a first-class decision record, not an afterthought. The repo already encodes the pattern in `mcp/tests/test_audit_lane_status.py` and `mcp/tests/test_audit_log_is_isolated.py`: the lane reports whether evidence is present, and the file path is isolated from the operator's home directory.

The architecture should include the following trace points:

1. `routing_request`: input task, selected tier, prompt digest, feature flags, and caller context
2. `model_selection`: benchmark role, local tag, OpenRouter model, latency, and token budget
3. `completion_result`: `status` / `ok` / `json` parse outcome, and whether fallback was used
4. `verification`: claim check result, confidence, contradiction or provenance verdict, and route to review if applicable
5. `memory_write`: what was written to investigation or mnemosyne memory, plus the seed/derivation lineage
6. `rollback`: reason for revert, previous config hash, and time when the revert occurred

Minimal log contract:

```json
{
  "ts": "2026-09-22T18:09:24Z",
  "request_id": "...",
  "route": "local|openrouter|shadow",
  "model": "qwen2.5:3b",
  "role": "cheap_fanout",
  "status": "ok|timeout|rate_limited|degraded",
  "verification": "passed|failed|skipped",
  "memory": "written|not_written",
  "rollback_reason": null
}
```

This makes a single decision reproducible in a later incident review: which prompt was routed, which model was chosen, which guardrail fired, and which follow-up action (recovery, fallback, or rollback) occurred.

## 5) Completion criteria

The work is complete when:

- the benchmark harness produces reproducible role winners
- the rollout plan is backed by an explicit phase gate and rollback condition
- every verification command above passes
- the audit trace records key routing and model decisions without leaking into the operator's home directory

Those gates are the concrete implementation path for the in-progress Loci tasks that were tracking benchmark harness design, rollout policy, verification, and audit trace architecture.
