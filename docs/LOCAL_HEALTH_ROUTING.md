# Local health routing

`llm_local.generate()` can move a request off the local model when the local model is measurably failing.
Off by default: `LOCI_LOCAL_HEALTH_ROUTING=1`.

## Signal

```
LiteLLM proxy -> cost ledger (one row per attempt)
              -> ops_logs collector (loci_litellm.ledger_summary, every ship run)
              -> ~/.loci/litellm_ledger_health.json   (also written to the WSL ~/.loci)
              -> mcp/local_health.py -> llm_local
```

The snapshot holds, per model group, for LOCAL deployments over the last hour: attempts, organic attempts,
good rate (succeeded AND returned an answer), empty rate, p95. It says whether the numbers come from
`organic` traffic or `all` traffic (which includes eval and probe calls).

## Decision (`local_health.check`)

Degraded only if all hold: snapshot version 1, not older than `LOCI_LOCAL_HEALTH_MAX_AGE_S` (1800), the group
with the most organic attempts has `basis == "organic"` and at least `LOCI_LOCAL_HEALTH_MIN_N` (20) attempts,
and either good rate is below `LOCI_LOCAL_HEALTH_MIN_GOOD` (0.9) or empty rate is above
`LOCI_LOCAL_HEALTH_MAX_EMPTY` (0.15).

A missing, corrupt, stale, thin or synthetic-only snapshot is **not** evidence of degradation.

## Behavior

When degraded, a request with no explicit model and `think=False` tries the cloud tier first
(`_try_cloud_tier`: prompt-export opt-in, role lists, per-call cap, daily budget and redaction all apply).
If that does not answer, local is tried as before. Health alone never refuses a request. A result served
this way carries `health_routed: true`; a WARNING names the group and the numbers.

## Limits

- The snapshot measures LiteLLM's local deployments, which are the same Ollama/resident model Loci calls
  directly; it is a correlated signal, not a measurement of Loci's own calls.
- Attempts are per attempt, not per request.
- `local_health.check` mirrors `loci_litellm.ledger_summary.local_degraded` in the LiteLLM plugin
  (rjmendez/litellm). Keep the thresholds and the conservative rules equal.
- No snapshot is produced until the ledger has organic traffic; until then this does nothing.
