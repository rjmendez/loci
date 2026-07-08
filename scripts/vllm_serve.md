# Deploying vLLM on oxalis's RTX 2080 Ti — batched OpenAI-compatible generation

This is the concrete deploy plan for the primary path of `mcp/batched_gen.py`. It stands up a
vLLM OpenAI-compatible server on oxalis's **second** GPU so high-concurrency workflow fan-out
gets true continuous batching instead of Ollama's serialized one-request-at-a-time behavior.

> Status: **plan, not yet executed.** `batched_gen.generate_batch` degrades to the sequential
> Ollama tier (`mcp/llm_local.generate`) until `VLLM_BASE_URL` points at a running server.

## Why a 2nd server at all

Grounding recap:
- `[hardware]` oxalis has two GPUs: the **RTX 4070 Ti** (torch `cuda:0` inside `mcp/.venv`) and a
  **RTX 2080 Ti, 11 GB** on the Windows/Ollama side (`OLLAMA_BASE_URL=http://100.73.200.19:11434`).
- `[rerank]` the 4070 Ti is already committed to the CrossEncoder reranker in `server.py`.
- `[gen]` the current local-generation tier is Ollama `qwen2.5:3b`, pinned warm via `keep_alive`.

Ollama serves requests essentially serially. When a workflow fans out N planning/research
agents (or runs a per-item map/gate over dozens of short prompts), that serialization is the
bottleneck. vLLM's **continuous batching** interleaves many in-flight sequences on one GPU, so
throughput scales with concurrency instead of collapsing to it.

Putting vLLM on the **2080 Ti** keeps it off the 4070 Ti reranker and off Ollama's queue —
the two generation tiers run on different GPUs and don't contend.

## Pick a model that fits 11 GB

The 2080 Ti has 11 GB and no bf16 tensor cores, so **use fp16** and a small instruct model.
`Qwen2.5-3B-Instruct` is the natural match — same family as the Ollama `qwen2.5:3b` tier, so the
fallback path stays behaviorally consistent. In fp16, 3B weights are ~6 GB, leaving room for the
KV cache. If VRAM is tight, `--max-model-len` (below) is the main knob; a 1.5B model is the
smaller fallback.

## Docker command (recommended)

Run on the host that owns the 2080 Ti. If that GPU is index `1` on the Windows/WSL box, pin it
explicitly so vLLM never grabs the 4070 Ti.

```bash
docker run --rm --name vllm-qwen \
  --gpus '"device=1"' \                       # pin the 2080 Ti; verify index with nvidia-smi
  -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \                    # only if the model repo requires auth
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-3B-Instruct \
  --served-model-name Qwen2.5-3B-Instruct \   # this is the `model` field clients send
  --dtype float16 \                           # 2080 Ti has no bf16; force fp16
  --gpu-memory-utilization 0.90 \             # fraction of the 11 GB vLLM may claim
  --max-model-len 8192 \                      # cap context to bound KV-cache VRAM
  --port 8000
```

Notes:
- `--gpu-memory-utilization 0.90` tells vLLM to pre-reserve ~90% of the 11 GB for weights +
  KV cache. Lower it (e.g. `0.85`) if anything else shares the card; raise cautiously.
- `--served-model-name Qwen2.5-3B-Instruct` MUST match what clients put in the request `model`
  field. `batched_gen` defaults to exactly this string (`VLLM_MODEL`, default
  `"Qwen2.5-3B-Instruct"`), so leaving both at the default "just works".
- `--max-model-len` bounds the KV cache; if vLLM logs "No available memory for the cache",
  lower it or lower `--gpu-memory-utilization`.

### TGI alternative

`batched_gen` posts OpenAI-style `/v1/completions`, which HuggingFace **TGI** also serves
(`ghcr.io/huggingface/text-generation-inference`, message/completions routes). vLLM is the
default here because its OpenAI surface and continuous batching are the closest fit; TGI is a
drop-in if preferred. Either way, point `VLLM_BASE_URL` at its base (no `/v1` suffix).

## Wire it up

`batched_gen` treats `VLLM_BASE_URL` as the sole on/off switch: **unset → Ollama fallback; set →
try batched first, fall back on failure.** Set the base URL (host + port, no `/v1`):

```bash
# Same box as the MCP server:
export VLLM_BASE_URL="http://localhost:8000"

# Reaching the Windows/WSL GPU host over tailscale (mirrors OLLAMA_BASE_URL's host):
export VLLM_BASE_URL="http://100.73.200.19:8000"

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

A JSON body with `choices[0].text` means the primary path is live and
`generate_batch` will stop falling back to Ollama.

## When vLLM beats Ollama (and when it doesn't)

**Use vLLM (batched path) when:**
- A workflow fans out **many concurrent short generations** — per-agent prompts, map/gate/
  classify over a list, N-way planning. Continuous batching is the whole point; this is the
  win the module exists for.
- You want stable throughput under bursty concurrency instead of a serialized Ollama queue.

**Stick with Ollama (fallback) when:**
- **Low concurrency / single prompts** — one-off `llm_local.generate` calls. Ollama's warm
  `keep_alive` model has no batching overhead and no second server to babysit.
- The 2080 Ti is busy or the model won't fit — the fallback keeps generation working.
- You need a model already pulled in Ollama and don't want to manage a HF download.

Because the fallback is automatic and fail-open (`[pattern:fail-open]`), you can deploy vLLM for
the throughput win and lose nothing if it's down: `generate_batch` silently reverts to the
sequential Ollama tier, one prompt at a time.

## Open items grounding is silent on
- Exact GPU **device index** of the 2080 Ti on the WSL/Windows host — verify with `nvidia-smi`
  before pinning `--gpus`.
- Whether the 2080 Ti is reachable to the vLLM container under WSL2 GPU passthrough, or whether
  vLLM must run on the Windows side and be exposed over the tailscale IP.
- Real measured tokens/s and max concurrency on this specific card — benchmark before relying on
  it for latency-sensitive paths.
