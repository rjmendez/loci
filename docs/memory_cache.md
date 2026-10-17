# Memory-Augmented Cache Hook

## Overview

The memory-augmented cache hook short-circuits model calls when Loci's Qdrant
memory already contains a high-confidence answer to the current query. Instead
of injecting context and invoking the model, `grounding_hook_augmented` returns
the stored finding directly as `cache_response`, eliminating token spend and
latency for near-duplicate queries (~31% of production traffic).

## How it works

```
                        grounding_hook_augmented()
                                    │
                          load grounding_context
                                    │
                     ┌──── findings present? ────┐
                     │ yes                        │ no / policy disabled
                     ▼                            ▼
           check_memory_cache()          build augmented system msg
                     │                   inject into messages → model
         ┌── hit? ──┤
         │ yes       │ no
         ▼           ▼
    return           build augmented system msg
    cache_response   inject into messages → model
    (no model call)
```

A `CacheHit` is returned when a finding satisfies **both** conditions:

| Condition | Default | Env var |
|-----------|---------|---------|
| `score >= confidence_threshold` | 0.85 | `LOCI_CACHE_CONFIDENCE` |
| `age_hours <= ttl_hours` | 24 h | `LOCI_CACHE_TTL_HOURS` |

If a finding has no timestamp it is treated as always-fresh (age = 0).

## Return shape on a cache hit

```json
{
  "messages": "<original messages, unchanged>",
  "policy_id": "<policy_id or null>",
  "augmented": false,
  "cache_hit": true,
  "cache_response": "[LOCI-CACHE hit=0.92 age=1.0h src=loci_memory id=f1]\n<content>",
  "r_mem": 0.92
}
```

`cache_hit: false` (or absent) means the hook fell through to normal augmentation.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOCI_POLICY_ENABLE` | `0` | Master switch — cache is only active when `1`/`true`/`yes` |
| `LOCI_CACHE_CONFIDENCE` | `0.85` | Minimum Qdrant score to accept as a cache hit |
| `LOCI_CACHE_TTL_HOURS` | `24.0` | Maximum age of a cached finding in hours |

## `r_mem` reward signal and DRPG trajectories

`r_mem` is set to the cache hit's embedding score (e.g. `0.92`). Callers
that record DRPG trajectories via `eval_hook_stub` should include `r_mem`
in the reward vector for the `ANSWER` action:

```python
reward_vector = {
    "r_ret": 0.0,   # no retrieval step taken
    "r_mem": result["r_mem"],  # cache hit reward
    "r_step": -0.02,
    "r_cor": ...,
}
```

A cache hit delivers `r_mem > 0` with **zero model cost**, which is strictly
better than any tier outcome for a well-matched query. The DRPG evolution loop
will naturally up-weight cache-friendly retrieval strategies over time.

## Expected hit rate

Internal profiling (2026-10) shows ~31% of production queries are
near-duplicates with cosine similarity ≥ 0.85 against findings stored in the
last 24 hours. At the default thresholds these queries are fully served from
memory, saving roughly one-third of model calls on busy investigations.

## Composition with tier-model-binding

Cache hits are evaluated **before** tier routing. A cache hit therefore has
zero model cost regardless of which tier would have been selected:

```
cache hit → r_mem reward, zero token spend   (beats light/medium/heavy)
cache miss → normal tier routing applies
```

This means `LOCI_CACHE_CONFIDENCE` and `LOCI_CACHE_TTL_HOURS` are the primary
levers for cost vs. freshness trade-offs, independent of tier policy.
