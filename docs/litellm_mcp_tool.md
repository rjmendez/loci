# LiteLLM MCP Tool — Governed Proxy Calls from Copilot CLI

## Purpose

`mcp__loci__litellm_call` and `mcp__loci__litellm_health` let Copilot CLI and other
MCP clients route cheap sub-tasks (summarisation, classification, extraction) to
**light-tier models** via a LiteLLM proxy, while Copilot continues using its primary
model for complex reasoning. Model selection is automatically constrained by a stored
Loci policy, preventing accidental use of heavy (expensive) models for simple work.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LOCI_LITELLM_PROXY_URL` | `http://localhost:4000` | Base URL of your LiteLLM proxy |
| `LOCI_LITELLM_API_KEY` | *(empty)* | Optional Bearer token sent as `Authorization: Bearer <key>` |
| `LOCI_LITELLM_TIMEOUT_S` | `30` | Per-request timeout in seconds |

Export these before starting Loci, or add them to `mcp/.env`.

## Example MCP Call

### Basic call (no policy)

```json
{
  "tool": "mcp__loci__litellm_call",
  "arguments": {
    "model": "claude-haiku-4.5",
    "messages": [
      {"role": "user", "content": "Summarise this log line: ERROR 500 /api/search"}
    ],
    "max_tokens": 256
  }
}
```

### Policy-governed call

```json
{
  "tool": "mcp__loci__litellm_call",
  "arguments": {
    "model": "claude-opus-4.8",
    "messages": [
      {"role": "user", "content": "Classify this finding as critical/high/medium/low"}
    ],
    "policy_id": "pol_abc123",
    "max_tokens": 128
  }
}
```

If `pol_abc123` has `tier = "light"`, the response will show `"downgraded": true` and
`"model": "claude-haiku-4.5"` — the heavy model was silently swapped.

## Tier Binding and Auto-Downgrade

The `_TIER_MODEL_MAP` (mirrored inline to avoid circular imports) defines three tiers:

| Tier | Models (first = fallback default) |
|---|---|
| `light` | `claude-haiku-4.5`, `gpt-5-mini`, `mistral-7b-instruct` |
| `medium` | `claude-sonnet-4.6`, `gpt-5.4-mini`, `llama-3-70b-instruct` |
| `heavy` | `claude-opus-4.8`, `gpt-5.4`, `gpt-5.5` |

**Downgrade example:**

1. Caller requests `claude-opus-4.8` with `policy_id = "pol_abc123"`.
2. Policy `pol_abc123` has `tier = "light"`.
3. `claude-opus-4.8` is a heavy-tier model — heavier than "light" allows.
4. The tool rewrites the model to `claude-haiku-4.5` (first model in "light").
5. Response: `"model": "claude-haiku-4.5"`, `"original_model": "claude-opus-4.8"`, `"downgraded": true`.

No downgrade happens when the requested model is already within the policy tier.
When no `policy_id` is given, tier defaults to `"heavy"` (no restriction).

## Fail-Open on Proxy Error

If the LiteLLM proxy is unreachable or returns an error, `mcp__loci__litellm_call`
**never raises** — it returns:

```json
{
  "error": "proxy_unavailable",
  "detail": "<exception message>",
  "model": "<originally requested model>"
}
```

Copilot CLI should check for `"error"` in the response and fall back to its own
primary model when the proxy is down.

## Trajectory Feedback Flow

When `policy_id` is provided and the DRPG evolution module (`drpg_evolution`) is
available, a successful call automatically pushes a step to `TrajectoryStore`:

```json
{
  "step": 1718123456,
  "action": "ANSWER",
  "reward": {
    "r_ret": 1.0,
    "r_cor": 1.0,
    "r_step": -0.0003
  },
  "policy_id": "pol_abc123"
}
```

- `r_ret = 1.0` if the proxy returned non-empty content, else `0.0`.
- `r_cor = 1.0` (correctness assumed; callers may push explicit corrections via
  `policy_trajectory_push`).
- `r_step = -0.01 * latency_ms / 100` — penalises slow responses.

These trajectories feed `policy_evolve`, which promotes or demotes the policy tier
based on accumulated reward scores.

## Checking Proxy Health

```json
{
  "tool": "mcp__loci__litellm_health",
  "arguments": {}
}
```

Response (healthy):
```json
{"healthy": true, "proxy_url": "http://localhost:4000", "detail": "..."}
```

Response (unhealthy):
```json
{"healthy": false, "proxy_url": "http://localhost:4000", "detail": "Connection refused"}
```

`mcp__loci__litellm_health` never raises; it is safe to call from health-check loops.

## Recommended LiteLLM Proxy `config.yaml`

Match the models in `_TIER_MODEL_MAP` so tier routing resolves without 404s:

```yaml
model_list:
  # light tier
  - model_name: claude-haiku-4.5
    litellm_params:
      model: anthropic/claude-haiku-4-5
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: gpt-5-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

  - model_name: mistral-7b-instruct
    litellm_params:
      model: ollama/mistral:7b-instruct
      api_base: http://localhost:11434

  # medium tier
  - model_name: claude-sonnet-4.6
    litellm_params:
      model: anthropic/claude-sonnet-4-5
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: gpt-5.4-mini
    litellm_params:
      model: openai/gpt-4.1-mini
      api_key: os.environ/OPENAI_API_KEY

  - model_name: llama-3-70b-instruct
    litellm_params:
      model: ollama/llama3:70b-instruct
      api_base: http://localhost:11434

  # heavy tier
  - model_name: claude-opus-4.8
    litellm_params:
      model: anthropic/claude-opus-4
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: gpt-5.4
    litellm_params:
      model: openai/gpt-4.5
      api_key: os.environ/OPENAI_API_KEY

  - model_name: gpt-5.5
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

Start the proxy with:

```bash
litellm --config config.yaml --port 4000
```

Then set `LOCI_LITELLM_PROXY_URL=http://localhost:4000` in your Loci environment.
