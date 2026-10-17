# Loci Policy.md Backend Adoption Guide

## Overview

This document describes the new `policy.md` object type and tier-aware policy tracking system integrated into Loci. Policies drive retrieval strategy, tier constraints, and DRPG evolution through LiteLLM hooks.

## Quick Start

### Enable Policy Tracking

Set environment variables:

```bash
export LOCI_POLICY_ENABLE=1                              # Enable policy tracking
export LOCI_POLICY_MAX_CHARS=8192                        # Max policy document size (bytes)
export LOCI_POLICY_ALLOWED_MODELS=gpt-4,claude-3,llama   # Allowed models (comma-separated)
```

### Create & Store a Policy

```python
from policy import Policy, PolicyTier, PolicyRegion, PolicyBackend

# Create a discovery-phase policy
policy = Policy(
    title="EVO-RAG Discovery Phase",
    tier=PolicyTier.LIGHT,           # light, medium, or heavy
    region=PolicyRegion.MEMORY,      # memory, context, or routing
    labels=["drpg", "evo-rag"],
    config={
        "discovery_mode": True,      # Enable discovery phase
        "reward_vector_enabled": True
    }
)

# Store it
backend = PolicyBackend()
result = backend.store(policy)
print(f"Stored policy: {result['policy_id']}")
```

### Load a Policy

```python
backend = PolicyBackend()
policy = backend.load("policy-id-here")

# Get tier constraints
constraints = policy.get_constraints()
print(f"Max context: {constraints['max_context_chars']} chars")
print(f"Max hops: {constraints['max_retrieval_hops']}")

# Get reward vector weights for DRPG
weights = policy.get_reward_vector_weights()
print(f"Discovery incentive: r_ret={weights['r_ret']}, r_rem={weights['r_rem']}")
```

## Policy Tiers

### Light Tier
- **Max context:** 2,048 characters
- **Max retrieval hops:** 3
- **Max refinement steps:** 2
- **Use case:** Resource-constrained environments, fast inference

### Medium Tier
- **Max context:** 8,192 characters
- **Max retrieval hops:** 7
- **Max refinement steps:** 4
- **Use case:** Balanced retrieval and efficiency (default)

### Heavy Tier
- **Max context:** 65,536 characters
- **Max retrieval hops:** 15
- **Max refinement steps:** 10
- **Use case:** Deep exploration, complex reasoning

## Policy Regions

### Memory Region
- **Scope:** Affects `grounding_hook` retrieval strategy
- **Effect:** Controls discovery mode, reward vector weighting
- **Example:** Enable broad search with high r_ret weight

### Context Region
- **Scope:** Affects system message injection and evidence backing
- **Effect:** Surfaces policy constraints in prompt
- **Example:** Include policy tier and constraints in system prompt

### Routing Region
- **Scope:** Affects `tier_router` circuit-breaker decisions
- **Effect:** Enforces tier downgrades when limits exceeded
- **Example:** Prevent heavy-tier models if policy tier=light

## LiteLLM Hooks Integration

### loci_policy_hook (pre-call)

Determines whether to apply policy based on model and environment.

```python
from hooks import loci_policy_hook

result = await loci_policy_hook(
    model="gpt-4",
    messages=[...],
    kwargs={"policy_id": "..."},
    litellm_params={...}
)

# Result:
# {
#     "apply": True,
#     "policy_id": "...",
#     "reason": "policy_enabled",
#     "violations": []
# }
```

### grounding_hook_augmented (retrieval)

Injects policy-backed evidence into system message.

```python
from hooks import grounding_hook_augmented

result = await grounding_hook_augmented(
    model="gpt-4",
    messages=[...],
    kwargs={"policy_id": "...", "grounding_context": {...}},
    litellm_params={...}
)

# Augmented system message includes:
# POLICY-AWARE MEMORY MATCH:
# - Tier: light
# - Phase: discovery
# - Reward Vector: {r_ret: 1.0, r_rem: -0.05, ...}
# - Max Context: 2048 chars
# - Max Hops: 3
# - Findings: [...]
```

### tier_router_surface (routing)

Surfaces policy constraints and enforces tier limits.

```python
from hooks import tier_router_surface

result = await tier_router_surface(
    model="gpt-4",
    messages=[...],
    kwargs={"policy_id": "...", "tier": "medium"},  # Exceeds policy limit if policy=light
    litellm_params={...}
)

# Result:
# {
#     "tier_decision": "light",         # Downgraded
#     "downgraded": True,
#     "violations": [
#         {
#             "type": "tier_exceeded",
#             "message": "Routing tier 'medium' exceeds policy tier 'light'"
#         }
#     ]
# }
```

### eval_hook_stub (DRPG feedback)

Captures step-level feedback for offline policy optimization via DRPG.

```python
from hooks import eval_hook_stub

result = await eval_hook_stub(
    model="gpt-4",
    messages=[...],
    kwargs={
        "policy_id": "...",
        "step": 1,
        "action": "SEARCH",
        "reward_vector": {
            "r_ret": 1.0,        # Retrieval bonus
            "r_rem": -0.05,      # Search penalty
            "r_ovp": -0.1,       # Overlap penalty
            "r_bkp": -0.2,       # Backtrack penalty
            "r_ref": -0.5,       # Refusal penalty (discovery)
            "r_step": -0.02,     # Step penalty
            "r_cor": 0.5         # Correctness reward
        }
    },
    litellm_params={...}
)

# Logs feedback for DRPG to consume:
# {
#     "step": 1,
#     "action": "SEARCH",
#     "reward": {...},
#     "policy_id": "...",
#     "timestamp": "2026-10-17T..."
# }
```

## 7D Reward Vector (EVO-RAG Research)

The reward vector drives DRPG policy evolution across discovery and refinement phases.

### Discovery Phase (`discovery_mode: True`)
Maximizes breadth of retrieval.

```python
policy = Policy(
    title="Discovery",
    config={"discovery_mode": True}
)

weights = policy.get_reward_vector_weights()
# {
#     "r_ret": 1.0,      # HIGH: incentivize retrieving new evidence
#     "r_rem": -0.05,    # LOW: tolerate search overhead
#     "r_ovp": -0.1,     # MODERATE: penalize overlap lightly
#     "r_bkp": -0.2,     # MODERATE: penalize backtracking
#     "r_ref": -0.5,     # HIGH PENALTY: discourage refusal
#     "r_step": -0.02,   # LOW: minimize step penalty
#     "r_cor": 0.5       # MODERATE: correctness bonus
# }
```

### Refinement Phase (`discovery_mode: False`)
Optimizes for efficiency and correctness.

```python
policy = Policy(
    title="Refinement",
    config={"discovery_mode": False}
)

weights = policy.get_reward_vector_weights()
# {
#     "r_ret": 0.5,      # MODERATE: controlled retrieval
#     "r_rem": -0.2,     # HIGH PENALTY: minimize searches
#     "r_ovp": -0.15,    # HIGHER: penalize redundancy
#     "r_bkp": -0.3,     # HIGH PENALTY: avoid backtracking
#     "r_ref": 1.0,      # HIGH: reward refusal when insufficient evidence
#     "r_step": -0.1,    # HIGHER: step efficiency
#     "r_cor": 1.0       # HIGH: correctness is paramount
# }
```

## Validation & Constraints

### Policy Validation

```python
from policy import validate_policy

policy_dict = {
    "title": "Test",
    "tier": "invalid",     # Will fail
    "region": "memory",
    "config": {}
}

errors = validate_policy(policy_dict)
# [
#     {
#         "code": "invalid_tier",
#         "message": "Tier must be one of: light, medium, heavy. Got: invalid"
#     }
# ]
```

### Tier-Config Consistency

Policies are validated to ensure config constraints match tier limits:

```python
policy_dict = {
    "tier": "light",
    "config": {
        "max_context_chars": 100000  # Exceeds light tier limit (2048)
    }
}

errors = validate_policy(policy_dict)
# [
#     {
#         "code": "tier_config_mismatch",
#         "message": "max_context_chars (100000) exceeds light tier limit (2048)"
#     }
# ]
```

## Testing

Run the comprehensive test suite:

```bash
cd loci/mcp
python -m pytest tests/test_policy_backend.py -v
```

Tests cover:
- Policy storage and retrieval
- Environment variable defaults
- Tier-aware constraint tracking
- Hook invocation (policy, grounding, routing, eval)
- Validation against tier/region constraints
- DRPG discovery vs refinement reward vectors

**Results:** 28 tests, all passing.

## File Structure

```
loci/mcp/
├── policy.py                    # Core policy backend
├── hooks.py                     # LiteLLM hook integrations
├── tests/
│   └── test_policy_backend.py   # Comprehensive test suite
└── docs/
    └── adoption.md               # This file
```

## Environment Variables

| Variable | Default | Example | Purpose |
|----------|---------|---------|---------|
| `LOCI_POLICY_ENABLE` | `0` | `1` | Enable/disable policy tracking |
| `LOCI_POLICY_MAX_CHARS` | `8192` | `16384` | Max policy document size (bytes) |
| `LOCI_POLICY_ALLOWED_MODELS` | (none) | `gpt-4,claude-3,llama` | Comma-separated allowed models |

## Integration with Loci MCP Server

To register policy tools in the MCP server, add (already done in `mcp/server.py`):

```python
@mcp.tool()
def policy_store(
    title: str,
    tier: str = "medium",
    region: str = "memory",
    labels: Optional[list[str]] = None,
    config: Optional[dict] = None,
) -> str:
    """Store a policy document."""
    backend = PolicyBackend()
    policy = Policy(
        title=title,
        tier=PolicyTier(tier),
        region=PolicyRegion(region),
        labels=labels or [],
        config=config or {},
    )
    result = backend.store(policy)
    return json.dumps(result)

@mcp.tool()
def policy_load(policy_id: str) -> str:
    """Load a policy by ID."""
    backend = PolicyBackend()
    policy = backend.load(policy_id)
    if policy:
        return json.dumps(policy.to_dict())
    return json.dumps({"error": "not_found", "policy_id": policy_id})
```

Exposed to MCP clients as `mcp__loci__policy_store` and `mcp__loci__policy_load`.

## References

- **DRPG:** Discovery↔Refinement Policy Gradient
- **EVO-RAG:** Evolution-based Retrieval Augmented Generation
- **Grounding:** Injecting retrieved evidence into system prompts
- **Tier Router:** Circuit-breaker enforcing resource constraints
- **Reward Vector:** 7D feedback signal (r_ret, r_rem, r_ovp, r_bkp, r_ref, r_step, r_cor)

## Known Limitations

1. Policy changes do not automatically retrain existing models
2. Reward vector logging is synchronous; consider async buffering for high throughput
3. Policy tier downgrades are enforced at routing time only (no pre-check)
4. No inter-policy dependencies or inheritance (each policy is independent)

## Future Enhancements

- Policy composition (combine multiple policies)
- Automatic tier selection based on query complexity
- Reward vector optimization via DRPG gradient descent
- Policy versioning and rollback
- A/B testing framework for policy variants
