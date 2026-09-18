# Second vLLM pod plan: specialist model on GPU 1

This is a ready-to-execute deployment plan for a **second** OpenAI-compatible vLLM pod,
pinned to **GPU index 1**, while the existing/default vLLM pod keeps serving the generic
batched generation tier.

This document is a **plan only**. It was prepared from the current single-endpoint docs and
code paths in `scripts/vllm_serve.md`, `scripts/gpu_placement.md`, `scripts/model_catalog.py`,
`mcp/batched_gen.py`, and `mcp/backends.py`. It was **not** validated against real GPU hardware
in this sandbox, so the hardware checks in [Real-hardware verification](#real-hardware-verification)
must be run before trusting it.

## Why a second pod

The current batched path is single-endpoint only:

- `mcp/backends.py:191-208` resolves one `vllm_url()` and one `vllm_model()`.
- `mcp/batched_gen.py:58-75` snapshots one `VLLM_BASE_URL` / `VLLM_MODEL` pair and resolves a
  single batched endpoint.
- `mcp/batched_gen.py:220-245` sends every batched request to that one endpoint.
- `scripts/swarm_escalate.py:212-218` passes only `model=...` into `batched_gen.generate_batch(...)`.

That works for one generic batched server, but it cannot distinguish:

- the default/general batched model, from
- a dedicated **specialist** pod such as the code specialist in
  `scripts/model_catalog.py:10,29-33`.

A second pod is therefore most useful when a specialist role has enough traffic or latency
sensitivity that it should not queue behind the generic tier.

## Concrete deployment: code specialist on GPU 1

### Chosen specialist

Use the existing code specialist alias from `scripts/model_catalog.py:10,29-33`:

- catalog role: `code`
- current catalog model tag: `qwen2.5-coder:7b`

For vLLM, the practical Hugging Face backend is the Transformers/AWQ equivalent, while the
**served model name stays the catalog tag** so callers do not need a separate public-facing
model string:

- HF weights: `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`
- served model name: `qwen2.5-coder:7b`

That AWQ choice is deliberate: `scripts/gpu_placement.md` discusses 11-12 GB-class cards, and a
full fp16 7B model is unlikely to fit there with useful KV cache headroom. A quantized 7B model
with a 4K context cap is the safer starting point.

### Docker command

This mirrors the style of `scripts/vllm_serve.md`, but binds a **second** port and pins the
container to **GPU 1**.

```bash
docker run --rm --name vllm-code-specialist \
  --gpus '"device=1"' \
  -p 8001:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-Coder-7B-Instruct-AWQ \
  --served-model-name qwen2.5-coder:7b \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.92 \
  --max-model-len 4096 \
  --port 8000
```

Recommended starting assumptions for this pod:

- `--gpus '"device=1"'`: explicitly pins the specialist pod to GPU index 1.
- `-p 8001:8000`: leaves the primary/default vLLM server free to keep using host port 8000.
- `--served-model-name qwen2.5-coder:7b`: matches the existing code specialist tag from
  `scripts/model_catalog.py:10`.
- `--gpu-memory-utilization 0.92`: aggressive but still leaves a little VRAM headroom for CUDA
  overhead on an otherwise dedicated card.
- `--max-model-len 4096`: safer than 8192 on 11-12 GB cards for a quantized 7B model.

If GPU 1 turns out to be a 16 GB+ card with plenty of headroom, the first tuning knobs to revisit
are:

1. raise `--max-model-len` from `4096` to `8192`, then
2. only if needed, raise `--gpu-memory-utilization` slightly.

If the operator wants a math specialist instead, use the same shape but swap the weights to the
non-GGUF HF checkpoint (for example `Qwen/Qwen2.5-Math-7B-Instruct`) and set
`--served-model-name` to whatever routed model name the follow-up implementation standardizes on.

## Proposed minimal-diff wiring into the codebase

Do **not** implement this in this doc-only change. This is the API shape the follow-up task
`implement-multi-vllm-endpoint-routing` should add.

### Recommendation: keep the existing shared endpoint, add optional per-role endpoints

The lowest-risk extension is:

- keep the current shared/default path untouched
- add optional **role-specific** vLLM URL/model resolvers
- only route to a specialist endpoint when a caller opts in explicitly

That preserves today's behavior for every existing caller that knows only about the shared batched
server.

### Proposed environment variables

Add optional role-specific overrides following the current env-first style in `mcp/backends.py`:

- `VLLM_BASE_URL_CODE`
- `VLLM_MODEL_CODE`
- `VLLM_BASE_URL_MATH`
- `VLLM_MODEL_MATH`
- `VLLM_BASE_URL_SAFETY`
- `VLLM_MODEL_SAFETY`
- `VLLM_BASE_URL_TOOL_CALLING`
- `VLLM_MODEL_TOOL_CALLING`

Keep the existing shared vars unchanged:

- `VLLM_BASE_URL`
- `VLLM_MODEL`

### Proposed `~/.loci/backends.toml` shape

Use nested TOML tables under `[vllm]`, which fits the existing config file and keeps the default
endpoint intact:

```toml
[vllm]
url = "http://gpu-host:8000"
model = "Qwen2.5-3B-Instruct"

[vllm.code]
url = "http://gpu-host:8001"
model = "qwen2.5-coder:7b"

[vllm.math]
url = "http://gpu-host:8002"
model = "qwen2.5-math:7b"

[vllm.safety]
url = "http://gpu-host:8003"
model = "llama-guard3:8b"
```

Notes:

- `[vllm]` remains the default/generic batched endpoint.
- `[vllm.code]` is the dedicated second pod from this plan.
- The `model` field should be the **served model name** the caller sends, not necessarily the raw
  HF repo ID.
- One vLLM process generally serves one model well; `mcp/batched_gen.py:103-107` already treats
  Ollama as the multi-model diversity fallback because vLLM is single-model-per-process here.

### Proposed resolver API in `mcp/backends.py`

Hook points:

- current shared resolver: `mcp/backends.py:191-208`
- existing per-task precedent on the Ollama side: `mcp/backends.py:134-187`

Minimal-diff extension:

```python
def vllm_url(role: str | None = None, probe_timeout: float = 1.0) -> str:
    ...

def vllm_model(role: str | None = None) -> str:
    ...
```

Behavior proposal:

- `role is None`: preserve current behavior exactly
  - `VLLM_BASE_URL` -> local `localhost:8000` probe -> `[vllm].url` -> `""`
  - `VLLM_MODEL` -> `[vllm].model` -> default model
- `role == "code"` (same pattern for `math`, `safety`, `tool_calling`):
  - `VLLM_BASE_URL_CODE` -> `[vllm.code].url` -> shared `vllm_url()`
  - `VLLM_MODEL_CODE` -> `[vllm.code].model` -> shared `vllm_model()`

Important design choice: **do not add a role-specific localhost probe**.

Reason: a silent `localhost:8000` fallback for `role="code"` would make specialist routing look
configured when it is actually collapsing back to the generic pod. For specialist roles, explicit
env/config should win; only then should the resolver fall back to the already-existing shared
endpoint.

### Proposed batched client extension in `mcp/batched_gen.py`

Hook points:

- import-time single-endpoint snapshots: `mcp/batched_gen.py:56-59`
- shared endpoint resolvers: `mcp/batched_gen.py:62-75`
- request-time URL/model selection: `mcp/batched_gen.py:220-245`

Minimal-diff API shape:

```python
def generate_batch(
    prompts: list[str],
    model: Optional[str] = None,
    max_tokens: int = 256,
    fmt: Optional[str] = None,
    client_fn: Optional[Callable[[], object]] = None,
    think: bool = False,
    endpoint_role: Optional[str] = None,
) -> list[dict]:
    ...
```

Resolution proposal inside `generate_batch(...)`:

- `url = backends.vllm_url(endpoint_role)`
- `served_model = model or backends.vllm_model(endpoint_role)`

One implementation detail the follow-up should handle carefully:

- `_VLLM` and `_DEFAULT_MODEL` at `mcp/batched_gen.py:58-59` currently snapshot only the shared
  env vars at import time.
- Role-aware routing should move URL/model selection to request time (or a small role-aware helper)
  so `VLLM_BASE_URL_CODE` / `VLLM_MODEL_CODE` can be honored.

### Proposed swarm hook

Hook point:

- `scripts/swarm_escalate.py:212-218`

Minimal-diff extension:

```python
def _batched_generate(..., model: str, ..., endpoint_role: Optional[str] = None) -> list[dict]:
    return batched_gen.generate_batch(..., model=model, ..., endpoint_role=endpoint_role)
```

Then the follow-up specialist-routing task can opt specific swarm lanes into distinct pods without
changing the default swarm path. Example future usage:

```python
_batched_generate(prompts, model="qwen2.5-coder:7b", max_tokens=..., endpoint_role="code")
```

The same pass-through is also useful at the MCP tool boundary in `mcp/llm_tools.py:140-153` if the
team wants tool callers to opt into a role-specific endpoint explicitly.

## Example operator configuration

If the primary generic pod stays on port 8000 and the new code pod runs on port 8001, a minimal
operator config would look like this:

```bash
export VLLM_BASE_URL="http://gpu-host:8000"
export VLLM_MODEL="Qwen2.5-3B-Instruct"

export VLLM_BASE_URL_CODE="http://gpu-host:8001"
export VLLM_MODEL_CODE="qwen2.5-coder:7b"
```

Or in `~/.loci/backends.toml`:

```toml
[vllm]
url = "http://gpu-host:8000"
model = "Qwen2.5-3B-Instruct"

[vllm.code]
url = "http://gpu-host:8001"
model = "qwen2.5-coder:7b"
```

## Curl smoke test for the new pod

After the container starts, verify both discovery and generation.

```bash
curl -sS http://gpu-host:8001/v1/models | jq .

curl -sS http://gpu-host:8001/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2.5-coder:7b",
    "prompt": "Write a Python function that returns the first 5 Fibonacci numbers.",
    "max_tokens": 64,
    "temperature": 0.0,
    "stream": false
  }' | jq .
```

Expected success criteria:

- `/v1/models` lists `qwen2.5-coder:7b`
- `/v1/completions` returns HTTP 200
- the JSON contains `choices[0].text`

If `/v1/models` works but `/v1/completions` fails, the most likely causes are:

- `--served-model-name` does not match the request `model`
- the model barely loaded but left insufficient KV-cache VRAM
- the chosen quantization/backend flags do not match the selected checkpoint

## Real-hardware verification

This plan was **not validated in this sandbox against a live NVIDIA host**. Before relying on it,
someone on the actual GPU box should run all of the following.

### 1. Verify the host really has a GPU 1

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
```

Confirm:

- there are at least two GPUs
- the intended second card is really index `1`
- GPU 1 has enough free VRAM for a quantized 7B model plus KV cache

### 2. Start only the specialist pod

Run the exact `docker run` command from this document.

Watch for startup failures such as:

- CUDA device not found
- out-of-memory during weight load
- unsupported quantization/backend combination
- Hugging Face auth/download failures

### 3. Run the curl smoke tests

Use the exact `/v1/models` and `/v1/completions` curls above.

Do not consider the pod ready until both pass.

### 4. Check steady-state VRAM headroom

While the pod is idle and while it serves a few concurrent requests:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
```

Verify:

- the process is really on GPU 1
- VRAM usage is stable enough that modest request bursts do not OOM
- GPU 0 still has room for the rerank + embedding tier described in `scripts/gpu_placement.md`

### 5. Only then wire Loci to it

After the hardware smoke test passes, add the proposed env/config entries and do one end-to-end
Loci run that intentionally targets the code specialist route.

## When a second dedicated pod is worth it

A second specialist pod is usually worth the extra operational complexity when:

- specialist traffic is frequent enough that queueing behind the generic batched model is visible
- the specialist prompts are latency-sensitive or long enough to poison the generic queue
- GPU 1 has enough VRAM to keep the specialist model resident without starving retrieval on GPU 0
- the role really benefits from a different model family than the default 3B generic pod

It is usually **not** worth it when:

- specialist traffic is rare
- one shared vLLM queue is already fast enough
- GPU 1 does not exist or has marginal VRAM headroom
- the team does not want the extra operational burden of another container, another port, and
  another health check

Rule of thumb:

- **time-share one pod** when the main problem is simplicity
- **split into two pods** when the main problem is queueing latency or model specialization and
  you can afford the VRAM
