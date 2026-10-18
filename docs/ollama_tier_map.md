# Ollama Tier Map -- Local Model Routing

This document describes how Loci routes LLM requests to local Ollama models
based on the `PolicyTier` assigned to each task.

## Tier -> Model Mapping (This Machine)

| Tier | Primary Models | Size | Notes |
|------|---------------|------|-------|
| **zero** | *(none)* | -- | No model; used for pass-through / skip |
| **nano** | `openthai-systemone:0.8b-q8_0` -> `antares-1b:Q8_0` -> `qwen2.5:3b` | 0.8-2 GB | Triage, classification, fast checks |
| **light** | `qwen2.5:3b` -> `huihui_ai/qwen3.5-abliterated:4b` -> `granite3-guardian:2b` | 1.9-3.3 GB | Low-cost reasoning, quick answers |
| **standard** | `ministral3-8b-reasoning-uncensored:Q4_K_M` -> `qwen3.5-9b-defiant-fable-heretic:Q4_K_M` | 5-7 GB | General-purpose STANDARD quality |
| **medium** | Same as standard (alias) | 5-7 GB | -- |
| **turbo** | `gemma4-12b-heretic:Q4_K_M` -> `claude-sonnet-4.6` | 7.4 GB | Heavy local or API extended thinking |
| **heavy** | `qwen3.8-27b-heretic:64k` -> `claude-opus-4.8` -> `gpt-5.4` | 17 GB | Splits across both GPUs |
| **research** | `qwen3.8-27b-heretic:64k` -> `claude-opus-4.8` | 17 GB | Deep research mode |

GPU allocation: RTX 4070 Ti (12 GB) + RTX 2080 Ti (11 GB). The 27b model requires both GPUs.

## Expected Latency Per Tier

| Tier | Typical first-token latency | Tokens/sec (est.) |
|------|----------------------------|-------------------|
| nano | 0.5-2 s | 80-150 tok/s |
| light | 2-5 s | 40-80 tok/s |
| standard | 5-15 s | 15-40 tok/s |
| turbo | 8-20 s | 10-25 tok/s |
| heavy | 15-40 s | 5-12 tok/s |
| research | 30-120 s | 3-8 tok/s |

## Task-Type Overrides

| Task type | Tier | Override model |
|-----------|------|---------------|
| `security_review` | standard | `foundation-sec-8b-reasoning:Q4_K_M` |
| `security_review` | light | `vulnllm-r-7b:Q4_K_M` |
| `code` | standard | `qwen2.5-coder-agent:latest` |
| `code` | light | `qwen2.5-coder:7b` |
| `triage` | nano | `antares-1b:Q8_0` |
| `triage` | light | `qwen2.5:3b` |
| `rtl_verify` | standard | `verireason-rtl-7b:Q4_K_M` |

Pass `task_type=` to `select_model()` to activate overrides.

## `LOCI_TIER_MODEL_MAP` Environment Override

Set this env var to a JSON object to override all defaults at runtime:

```bash
export LOCI_TIER_MODEL_MAP='{"nano":["ollama/my-custom-1b"],"standard":["ollama/my-8b","claude-sonnet-4.6"]}'
```

Keys are tier names (`zero`, `nano`, `light`, `standard`, `medium`, `turbo`, `heavy`, `research`).
Values are ordered lists -- the first available model wins. Loaded on every call (no restart needed).

The `.loci/.env` file on this machine sets `LOCI_TIER_MODEL_MAP` to the full deployment map.

## LiteLLM Routing: `ollama/` Prefix

Models prefixed with `ollama/` are routed to the local Ollama instance via LiteLLM.
The proxy URL is configured by `LOCI_LITELLM_PROXY_URL` (default: `http://localhost:4000`).

Cloud API models (`claude-*`, `gpt-*`) are used as fallbacks when `prefer_local=False`.

## Adding a New Model

1. Pull the model: `ollama pull my-new-model:tag`
2. Add to `LOCI_TIER_MODEL_MAP` in `~/.loci/.env`
3. No restart needed -- the map is read on each call.

## API Reference

```python
from loci.model_selector import select_model, resolve_for_loci, get_tier_model_map

model = select_model("standard")
model = select_model("standard", task_type="security_review")

resolved, was_downgraded = resolve_for_loci("ollama/qwen3.8-27b-heretic:64k", "nano")

tier_map = get_tier_model_map()
```