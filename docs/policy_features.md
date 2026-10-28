# Policy.md backend

Tier-aware policy documents that gate retrieval/context/routing behavior for
LociMCP and the LiteLLM Loci plugin.

## Components

- `mcp/policy.py` — `Policy`, `PolicyTier` (light/medium/heavy), `PolicyRegion`
  (memory/context/routing), `PolicyBackend` (JSON file storage under
  `~/.loci/policies/<policy_id>.json`), `validate_policy()`, `get_policy_config()`.
- `mcp/hooks.py` — LiteLLM-style async hooks: `loci_policy_hook` (pre-call
  apply/skip decision), `grounding_hook_augmented` (injects policy-aware
  memory-match context into messages), `tier_router_surface` (surfaces/enforces
  tier constraints on routing), `eval_hook_stub` (logs step-level DRPG
  feedback using the 7D EVO-RAG reward vector).
- `mcp/server.py` — exposes two MCP tools: `policy_store(title, tier, region,
  labels, config)` and `policy_load(policy_id)`.
- `litellm-loci/loci_litellm/policy_hook.py` — the LiteLLM-proxy-side mirror of
  the policy hook (env-driven enable/max-chars/allowed-models), with
  `EvoRagRewardTensor` and `PolicyViolation` tracking.

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `LOCI_POLICY_ENABLE` | `0` (off) | `1`/`true`/`yes` enables policy evaluation |
| `LOCI_POLICY_MAX_CHARS` | `8192` | Max context chars a policy may inject |
| `LOCI_POLICY_ALLOWED_MODELS` | unset (all allowed) | Comma-separated allowlist; hook returns `apply: false, reason: "model_not_allowed"` for others |

## Policy tiers

`PolicyTier.constraints()` returns per-tier caps used by `tier_router_surface`
and `validate_policy`:

| Tier | max_context_chars | max_retrieval_hops | max_refinement_steps |
|---|---|---|---|
| light | 2048 | 3 | 2 |
| medium | 8192 | 7 | 4 |
| heavy | 65536 | 15 | 10 |

## 7D reward vector (DRPG / EVO-RAG)

`Policy.get_reward_vector_weights()` returns phase-dependent weights
(discovery vs. refinement) over `r_ret, r_rem, r_ovp, r_bkp, r_ref, r_step,
r_cor`, consumed by `eval_hook_stub` for step-level feedback logging.

## Usage

```python
from policy import Policy, PolicyBackend, PolicyTier, PolicyRegion

backend = PolicyBackend()
policy = Policy(title="heavy-research", tier=PolicyTier.HEAVY,
                 region=PolicyRegion.ROUTING, config={"discovery_mode": True})
result = backend.store(policy)   # {"policy_id": ..., "status": "stored"}
loaded = backend.load(result["policy_id"])
```

Via MCP tools: `mcp__loci__policy_store(title=..., tier=..., region=...)` and
`mcp__loci__policy_load(policy_id=...)`.

## Status

Backend (policy.py, hooks.py, MCP tools) and LiteLLM-side hook
(policy_hook.py) are implemented with unit test coverage (28 tests in
`mcp/tests/test_policy_backend.py`, 25 tests in
`litellm-loci/tests/test_policy_hook.py`). Not yet wired into a live
LiteLLM proxy `config.yaml` callback list — that integration step is
tracked separately.
