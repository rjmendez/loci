# 7-Tier PolicyTier Ladder

Loci's policy tier system controls how much compute budget (context window, retrieval hops,
refinement steps, and model cost) a given tool call may consume.

---

## Tier Table

| Tier       | Value        | Model Examples                                        | `max_context_chars` | Cost/1M tokens | Use Case                                                   |
|------------|--------------|-------------------------------------------------------|---------------------|----------------|------------------------------------------------------------|
| `ZERO`     | `"zero"`     | *(none)*                                              | 0                   | $0             | Graph queries, cache hits — no LLM call whatsoever         |
| `NANO`     | `"nano"`     | `ollama/llama3.2:3b`, `ollama/mistral:7b`             | 512                 | $0 (local)     | Local Ollama inference; reflection stubs, verify ping      |
| `LIGHT`    | `"light"`    | `claude-haiku-4.5`, `gpt-5-mini`                      | 2 048               | ~$0.25         | Fast classification, short-context grounding               |
| `STANDARD` | `"standard"` | `claude-sonnet-4.6`, `gpt-5.4-mini`                   | 8 192               | ~$0.60         | General reasoning, investigation steps *(replaces MEDIUM)* |
| `TURBO`    | `"turbo"`    | `claude-sonnet-4.6-extended`, `o3-mini`               | 32 768              | ~$3            | Extended thinking, reasoning-heavy tasks                   |
| `HEAVY`    | `"heavy"`    | `claude-opus-4.8`, `gpt-5.4`, `gpt-5.5`              | 65 536              | ~$15           | Full document synthesis, deep analysis                     |
| `RESEARCH` | `"research"` | `claude-opus-4.8`, `gpt-5.5`                          | 524 288             | ~$50+/run      | Full swarm pipeline; `session_limit=1` safety gate         |

> **MEDIUM → STANDARD migration** — `PolicyTier.MEDIUM` (`"medium"`) is preserved as a
> backward-compatible alias. It resolves to the same constraints and model list as `STANDARD`.
> Existing code using `PolicyTier.MEDIUM` continues to work without changes.

---

## ZERO Tier — graph/cache path

`ZERO` allows no LLM calls (`max_context_chars=0`, `max_retrieval_hops=0`).
Use it when the answer is deterministic (graph look-up, cache hit, structured index query)
and you want to assert that no model API cost is incurred.

Pairs with `GraphCache` and `code_graph_query` to give zero-cost pre-answer checks.

```python
sp = StagePolicy.for_investigation()
# pre_answer_check stage always uses ZERO — check cache before calling a model
assert sp.tier_for("pre_answer_check") == PolicyTier.ZERO
```

---

## NANO Tier — local Ollama path

`NANO` maps to the existing Ollama routing already used by `verify_finding` and
`investigation_reflect`. It adds no API cost and does not require network access beyond
the local Ollama daemon.

Constraint highlight: `max_context_chars=512`, `max_retrieval_hops=1` — suitable for
yes/no reflection pings and lightweight token scoring.

```python
# verify_finding already selects an ollama model at NANO tier:
model = resolve_model_for_tier(PolicyTier.NANO)
# → "ollama/llama3.2:3b"
```

---

## TURBO Tier — reasoning sweet spot

`TURBO` is the new sweet spot for tasks that need extended-thinking or chain-of-thought
without paying HEAVY rates:

- `max_context_chars=32 768` — fits most investigation contexts
- `max_refinement_steps=6` — enough for multi-step reasoning
- Models: `claude-sonnet-4.6-extended`, `o3-mini`

Use TURBO for `swarm_reason` escalation stages or any tool where reasoning depth matters
but corpus size is bounded.

---

## RESEARCH Tier — safety gate via `session_limit=1`

`RESEARCH` is reserved for full swarm pipelines and runs with a hard `session_limit=1`
constraint, preventing accidental runaway multi-session spend. Always assign RESEARCH
explicitly; it is never an automatic fallback.

```python
sp = StagePolicy.for_swarm_reason(outer=PolicyTier.RESEARCH)
# synthesis stage reaches HEAVY (≤ RESEARCH outer), escalation reaches STANDARD
```

---

## StagePolicy — per-stage tier assignment

`StagePolicy` lets multi-stage tools run different stages at different tiers while
respecting an `outer_tier` ceiling.

### `StagePolicy.for_swarm_reason`

```python
sp = StagePolicy.for_swarm_reason(outer=PolicyTier.STANDARD)
# decompose        → NANO    (struct breakdown, cheap)
# fanout           → LIGHT   (parallel short queries)
# self_consistency → LIGHT   (comparison, short)
# escalation       → STANDARD (reasoning with context)
# synthesis        → STANDARD (clamped from HEAVY by outer=STANDARD)
```

### `StagePolicy.for_investigation`

```python
sp = StagePolicy.for_investigation(outer=PolicyTier.STANDARD)
# pre_answer_check → ZERO     (cache/graph check, no model)
# reason           → STANDARD (main reasoning step)
# reflect          → NANO     (lightweight self-critique)
# verify           → NANO     (boolean verification ping)
# adversarial      → STANDARD (challenge the answer)
```

### Custom stage policies

```python
sp = StagePolicy(
    stages={"my_stage": PolicyTier.TURBO},
    outer_tier=PolicyTier.HEAVY,
)
tier = sp.tier_for("my_stage")  # → TURBO (below HEAVY ceiling)
```

---

## Cost Projection: StagePolicy saves ~58% vs all-HEAVY

| Scenario                           | Effective avg tier | Relative cost |
|------------------------------------|-------------------|---------------|
| All stages at HEAVY                | HEAVY (~$15/1M)   | 100%          |
| `StagePolicy.for_swarm_reason`     | LIGHT–STANDARD    | ~42%          |
| `StagePolicy.for_investigation`    | ZERO–STANDARD     | ~35%          |

The savings come from running high-frequency cheap stages (decompose, reflect, verify)
at NANO/LIGHT while reserving HEAVY/RESEARCH only for the synthesis step where quality
matters most.

---

## Reward vector weights per tier

Each tier emits a 7-D reward vector used by the EVO-RAG DRPG optimizer:

| Dimension | ZERO | NANO | LIGHT/STANDARD (discovery) | TURBO | RESEARCH |
|-----------|------|------|----------------------------|-------|----------|
| `r_ret`   | 0.0  | 0.8  | 1.0                        | 0.6   | 1.0      |
| `r_bkp`   | 0.0  | -0.1 | -0.2                       | -0.25 | -0.6     |
| `r_step`  | 0.0  | -0.01| -0.02                      | -0.06 | -0.02    |

RESEARCH carries the highest `r_bkp` penalty (-0.6) because backtracking at that tier
is extremely expensive. ZERO zeroes all dimensions since no model call is made.
