# Tier→Model Binding and Cost-Driven Downgrade Enforcement

## Overview

Loci's policy system assigns each policy a **tier** (`light`, `medium`, or `heavy`).
The `TIER_MODEL_MAP` wires each tier to a concrete list of permitted models.  At call
time the `tier_router_surface` hook inspects the requested model, compares it against
the policy tier's allowed list, and rewrites the `model` field before the request
reaches the LLM API.  This is the single biggest cost lever in the system — a properly
tuned policy can reduce per-call spend by **50–80 %**.

---

## TIER_MODEL_MAP

Defined in `mcp/policy.py`:

```python
TIER_MODEL_MAP = {
    PolicyTier.LIGHT: [
        "claude-haiku-4.5",
        "gpt-5-mini",
        "mistral-7b-instruct",
        "llama-3-8b-instruct",
    ],
    PolicyTier.MEDIUM: [
        "claude-sonnet-4.6",
        "gpt-5.4-mini",
        "llama-3-70b-instruct",
        "mistral-large",
    ],
    PolicyTier.HEAVY: [
        "claude-opus-4.8",
        "gpt-5.4",
        "gpt-5.5",
        "claude-sonnet-4.6",  # fallback if opus unavailable
    ],
}
```

Resolution rules (see `resolve_model_for_tier`):

| Requested model tier | Policy tier | Outcome |
|---|---|---|
| Same tier | Any | Returned unchanged |
| Lower tier (cheaper) | Any | Returned unchanged — within budget |
| Higher tier (pricier) | Any | Downgraded to first model in policy tier |
| Unknown | Any | Downgraded to first model in policy tier |

---

## Environment Variable Override — `LOCI_TIER_MODEL_MAP`

Set `LOCI_TIER_MODEL_MAP` to a JSON string to replace the built-in map at runtime
without touching source code.  Useful for A/B testing, fleet-wide rollouts, or
temporarily pinning a model while an upstream provider is degraded.

```bash
export LOCI_TIER_MODEL_MAP='{
  "light":  ["my-haiku-clone", "another-cheap-model"],
  "medium": ["my-sonnet-clone"],
  "heavy":  ["my-opus-clone"]
}'
```

Rules:
- Only keys `light`, `medium`, and `heavy` are accepted; others are silently ignored.
- Invalid JSON falls back to the built-in `TIER_MODEL_MAP` without raising an error.
- A partial override (e.g. only `"light"`) replaces only that tier; other tiers return
  an empty list, so unknown and higher-tier models are not rewritten for those tiers.
  Use a complete three-tier JSON to avoid surprises.

---

## Cost Comparison (rough $/1 M input tokens)

| Tier | Representative models | Approx. cost |
|---|---|---|
| **light** | claude-haiku-4.5, gpt-5-mini | ~$0.25 |
| **medium** | claude-sonnet-4.6, gpt-5.4-mini | ~$0.60 |
| **heavy** | claude-opus-4.8, gpt-5.4, gpt-5.5 | ~$15.00 |

Routing a `heavy`-tier call through a `light` policy saves roughly **60×** on that
call.  In a typical Loci session that issues dozens of retrieval calls, this compounds
quickly.

---

## How `tier_router_surface` Enforces the Binding

The hook is called by LiteLLM's pre-call machinery.  Flow:

```
caller → LiteLLM → tier_router_surface
                        │
                        ├─ no policy_id? → pass through (model unchanged)
                        │
                        ├─ load policy from PolicyBackend
                        │
                        └─ resolve_model_for_tier(requested_model, policy.tier)
                                │
                                ├─ model in tier or cheaper? → (model, False)
                                │
                                └─ model pricier or unknown? → (first_tier_model, True)
                                        │
                                        └─ append "model_downgraded" violation
                                        └─ return resolved model in "model" field
```

Return dict shape:

```python
{
    "tier_decision": "light",          # policy.tier.value
    "downgraded": True,                # whether the model was rewritten
    "model": "claude-haiku-4.5",       # model to use (may differ from requested)
    "original_model": "claude-opus-4.8",  # set only when downgraded=True
    "violations": [
        {
            "type": "model_downgraded",
            "message": "Model 'claude-opus-4.8' downgraded to 'claude-haiku-4.5' per light tier policy",
            "severity": "warning",
        }
    ],
}
```

LiteLLM should replace `model` in the outbound request with `result["model"]`.

---

## DRPG Auto-Demotion

The DRPG evolution loop (`mcp/drpg_evolution.py`) tracks per-policy reward signals
over time.  When a policy's rolling reward indicates that a lighter-tier model is
performing equally well (high `r_cor`, low `r_rem`), the loop can auto-demote the
policy tier (e.g. `heavy → medium` or `medium → light`).

Combined with `tier_router_surface`, this creates a closed-loop cost-optimisation
cycle:

```
DRPG reward signal
      │
      ▼
PolicyBackend.update_tier(policy_id, new_tier)
      │
      ▼
Next call: tier_router_surface sees lighter tier → cheaper model selected
```

Demotion is conservative: the loop requires several consecutive episodes above the
reward threshold before promoting a downgrade, preventing oscillation.
