# Deploying vLLM for batched OpenAI-compatible generation

The primary path of `mcp/batched_gen.py`. Stand up a vLLM OpenAI-compatible server on a GPU host
so high-concurrency workflow fan-out gets true continuous batching instead of Ollama's serialized
one-request-at-a-time behavior. The endpoint is resolved via `backends.vllm_url()` — nothing about
your host lives in the code.

> `batched_gen.generate_batch` degrades to the sequential Ollama tier (`mcp/llm_local.generate`)
> until a vLLM base URL is configured (env `VLLM_BASE_URL`, a local `:8000`, or `[vllm].url` in
> `~/.loci/backends.toml`).

## Why a batched server at all

Ollama serves requests essentially serially. When a workflow fans out N planning/research agents
(or runs a per-item map/gate over dozens of short prompts), that serialization is the bottleneck.
vLLM's **continuous batching** interleaves many in-flight sequences on one GPU, so throughput
scales with concurrency instead of collapsing to it.

If the same GPU also hosts the latency-sensitive reranker/embeddings, keep the tiers from
contending — see `gpu_placement.md`.

## Pick a model that fits your GPU

Use a small instruct model sized to your VRAM. `Qwen2.5-3B-Instruct` is a good default — same
family as the Ollama `qwen2.5:3b` tier, so the fallback path stays behaviorally consistent. In
fp16, 3B weights are ~6 GB, leaving room for the KV cache. On older cards without bf16 tensor
cores, force `--dtype float16`. If VRAM is tight, `--max-model-len` is the main knob; a 1.5B model
is the smaller fallback.

## Docker command

Run on the host that owns the target GPU. Pin the GPU explicitly (verify its index with
`nvidia-smi`) so vLLM never grabs a card reserved for another tier.

```bash
docker run --rm --name vllm-qwen \
  --gpus '"device=1"' \                       # pin the target GPU; verify index with nvidia-smi
  -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \                    # only if the model repo requires auth
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-3B-Instruct \
  --served-model-name Qwen2.5-3B-Instruct \   # this is the `model` field clients send
  --dtype float16 \                           # force fp16 on cards without bf16
  --gpu-memory-utilization 0.90 \             # fraction of VRAM vLLM may claim
  --max-model-len 8192 \                      # cap context to bound KV-cache VRAM
  --port 8000
```

Notes:
- `--gpu-memory-utilization` pre-reserves that fraction of VRAM for weights + KV cache. Lower it
  (e.g. `0.85`) if anything else shares the card; raise cautiously.
- `--served-model-name` MUST match what clients put in the request `model` field. `batched_gen`
  defaults to `VLLM_MODEL` (default `"Qwen2.5-3B-Instruct"`), so leaving both at the default "just
  works".
- `--max-model-len` bounds the KV cache; if vLLM logs "No available memory for the cache", lower it
  or lower `--gpu-memory-utilization`.

### TGI alternative

`batched_gen` posts OpenAI-style `/v1/completions`, which HuggingFace **TGI** also serves
(`ghcr.io/huggingface/text-generation-inference`). vLLM is the default here because its OpenAI
surface and continuous batching are the closest fit; TGI is a drop-in if preferred. Either way,
point the base URL at its base (no `/v1` suffix).

## Wire it up

`batched_gen` treats the resolved vLLM base URL as the sole on/off switch: **unset → Ollama
fallback; set → try batched first, fall back on failure.** Configure it one of three ways
(env > local probe > config):

```bash
# 1. Env override (host + port, no /v1):
export VLLM_BASE_URL="http://localhost:8000"          # same box as the MCP server
export VLLM_BASE_URL="http://gpu-host:8000"           # a GPU host reachable over your network

# 2. Or leave it unset and let backends probe localhost:8000 automatically.

# 3. Or put it in ~/.loci/backends.toml:  [vllm]\n url = "http://gpu-host:8000"

# Optional overrides (defaults shown):
export VLLM_MODEL="Qwen2.5-3B-Instruct"   # must equal --served-model-name
export VLLM_TIMEOUT="120"                 # seconds
```

`batched_gen` appends `/v1/completions` itself — pass only the base.

## Smoke test (no client code)

```bash
curl -s "$VLLM_BASE_URL/v1/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen2.5-3B-Instruct","prompt":"ping","max_tokens":8}' | jq .
```

A JSON body with `choices[0].text` means the primary path is live and `generate_batch` will stop
falling back to Ollama.

## When vLLM beats Ollama (and when it doesn't)

**Use vLLM (batched path) when:**
- A workflow fans out **many concurrent short generations** — per-agent prompts, map/gate/classify
  over a list, N-way planning. Continuous batching is the whole point.
- You want stable throughput under bursty concurrency instead of a serialized Ollama queue.

**Stick with Ollama (fallback) when:**
- **Low concurrency / single prompts** — one-off `llm_local.generate` calls. Ollama's warm
  `keep_alive` model has no batching overhead and no second server to babysit.
- The target GPU is busy or the model won't fit — the fallback keeps generation working.
- You need a model already pulled in Ollama and don't want to manage a HF download.

Because the fallback is automatic and fail-open, you can deploy vLLM for the throughput win and
lose nothing if it's down: `generate_batch` silently reverts to the sequential Ollama tier.
