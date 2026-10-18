# Tier -> model routing (local pick from the model pool)

`loci/model_selector.py` names no model. For a `PolicyTier` it returns an ordered candidate
list: the local model the **model pool** ranks first for that tier's role, then the entries of
the policy map (`loci.policy.get_tier_model_map()`: the compiled `TIER_MODEL_MAP`, or the
`LOCI_TIER_MODEL_MAP` JSON override). To change which local model serves a tier, edit the pool
(`[[models.pool]]` in `backends.toml`), never this repo.

## Which pool role serves a tier

The lookup is `model_pool.rank_role(role).chosen` (installed, within the VRAM cap, best
`rank - resident_bonus`). It is pure: it logs no routing decision and never explores, because
rewriting a tier is not a model call. Roles are tried in order; the first that yields a model wins.

| Tier | Roles tried |
|------|-------------|
| zero | none (no LLM call is allowed) |
| nano | `tier_nano`, `decide` |
| light | `tier_light`, `gen` |
| standard, medium | `tier_standard`, `gen` |
| turbo | `tier_turbo`, `gen_large` |
| heavy | `tier_heavy`, `gen_large` |
| research | `tier_research`, `gen_large` |

A `task_type` goes first: `security_review` -> `security`, `rtl_verify` -> `rtl`,
`code` -> `code`, `triage` -> `decide`. An unknown task type is ignored.

To pin a model to one tier, give a pool entry that role:

```toml
[[models.pool]]
name = "<an installed tag>"
roles = ["gen", "tier_standard"]
```

The pick is returned as `ollama/<tag>`.

## Ordering

* `prefer_local` (default on; `LOCI_TIER_PREFER_LOCAL=0` turns it off): `[local pick, *policy map]`.
* Off: the map's non-`ollama/` entries first, then the local pick, then the map's own `ollama/`
  entries, so the cloud fallback is reachable.
* Duplicates are dropped. `select_model(tier)` is the first candidate, or `""`.
* `select_model("zero")` and an unknown tier name raise `ValueError`.

## What uses it

* `resolve_model_for_tier(tier, available=None)` walks the candidates.
* `downgrade_model_to_tier(requested, tier)`: a model the map lists at or below the tier, or that
  the pool serves for such a tier, is kept. Anything else (including a model no tier lists) is
  replaced by the first candidate for the tier.

## Failing open

No pool configured, a role nobody serves, `model_pool` not importable (the LiteLLM proxy
process has no `mcp/` on its path) or an error in the lookup all leave the local slot empty.
The candidates are then exactly the policy map's, which is what the code did before the pool
existed.
