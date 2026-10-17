# Dual-GPU placement for the Loci inference tiers

When a machine has more than one GPU, place the Loci tiers so the **latency-sensitive retrieval
tier** (reranker + embeddings) never queues behind the **throughput/heavy generation tier**. All
endpoints are resolved through `backends` (`ollama_url()` / `vllm_url()`), so this is purely about
which physical card each backend lands on — no host is hardcoded.

## The tiers

- **[rerank]** `mcp/server.py` reranks on GPU via a lazy `sentence_transformers.CrossEncoder`
  (default `BAAI/bge-reranker-v2-m3`), loaded on torch `cuda:0` when available and used by
  `rag_context_search`. It's a small model on the critical path of **every** retrieval call.
- **[embed]** Embeddings via Ollama (`nomic-embed-text`), warm on GPU.
- **[gen]** Local generation via Ollama (`qwen2.5:3b`, pinned via `keep_alive`), with an optional
  vLLM batched-gen server as the higher-throughput primary path (`mcp/batched_gen.py`).

## The contention problem

If a heavy/batched generation model and the interactive reranker land on the **same** GPU, a long
generation occupies the SMs and VRAM and stalls rerank latency (and vice-versa). The goal of
placement is to keep the latency-sensitive retrieval tier off the heavy generation tier.

## Recommended layout

| Tier | Workload | Target GPU |
|------|----------|-----------|
| Retrieval (latency-sensitive) | CrossEncoder rerank (torch) + warm Ollama embeddings | the **inference GPU** (`cuda:0`) |
| Generation (throughput / batched) | vLLM/TGI batched-gen + heavy Ollama gen | a **separate GPU**, if available |

Rerank stays on the inference GPU and must never queue behind a multi-second generation. A
batched-gen server is heavier and belongs on a second card when one exists. If a card is shared
with another tenant (e.g. a training job), treat generation there as **opportunistic and
preemptible** — read live GPU telemetry (`nvidia-smi` headroom) and yield rather than squatting.
With only one GPU, batched-gen shares it and accepts the contention.

## Levers

### torch side (the reranker)
The reranker is pinned to `cuda:0` in code. To force the whole venv process onto a specific
physical GPU, launch it with `CUDA_VISIBLE_DEVICES` so `cuda:0` **is** the card you want:

```bash
CUDA_VISIBLE_DEVICES=0 python -m mcp.server
```

`CUDA_VISIBLE_DEVICES` remaps indices: the first listed device becomes `cuda:0`. It is read once
at CUDA init, so set it at process launch.

> Caveat: GPU enumeration order is not guaranteed to match `nvidia-smi` index order (notably under
> WSL2 ↔ Windows passthrough). **Verify** which physical card `cuda:0` is with
> `torch.cuda.get_device_name(0)` before trusting the mapping.

### Ollama side (gen / embed models)
Ollama is a single server that schedules across the GPUs it can see. Set these **on the Ollama
host**, not in this venv:

- **`CUDA_VISIBLE_DEVICES`** — restrict which physical GPUs Ollama uses. To dedicate one card to a
  heavy gen model, run that Ollama instance with `CUDA_VISIBLE_DEVICES` pointing at just that GPU.
- **`OLLAMA_SCHED_SPREAD=1`** — spread a model across all visible GPUs instead of packing onto one.
  Use it when you *want* a big model to shard; **leave it unset** when you want each model pinned to
  one GPU so gen and rerank/embeddings don't collide.
- **`OLLAMA_MAX_LOADED_MODELS`** / **`OLLAMA_NUM_PARALLEL`** — cap concurrent resident models /
  parallel requests so a batched gen model doesn't evict the warm `nomic-embed-text`.
- **`OLLAMA_KEEP_ALIVE`** (or per-request `keep_alive`) — keep hot models resident so a warm stream
  never re-pays the cold load.

A clean split is **two Ollama endpoints**, each with its own `CUDA_VISIBLE_DEVICES`: one on the
inference GPU for embeddings, one on a second card for the heavy gen model. A single Ollama over
both cards works too — keep `OLLAMA_SCHED_SPREAD` off and rely on `keep_alive` pinning.

## Interaction with `gpu_warm.py`

`scripts/gpu_warm.py` pins the gen + embed models resident via `keep_alive=-1` and reports
`/api/ps` residency. It pins **whatever GPU(s) the Ollama it talks to can see** — it does not itself
choose a card. Placement is decided by the env vars above on the Ollama host; `gpu_warm.py` then
keeps the chosen models warm so the cold load is paid once. Run the keeper (`--loop`) pointed at
each Ollama endpoint you stand up.

## Design principles

- **Telemetry-driven, not hardcoded** — place work by live `nvidia-smi` headroom. Any batched-gen
  launcher sharing a card with another tenant should read telemetry and yield rather than pinning.
- **Gate quality changes** — a reranker/expansion change is a retrieval-QUALITY change; A/B it on a
  held-out query set before flipping the default (see `scripts/judge_eval.py`).
- **Fail-open, warm, cached** — every tier degrades gracefully when a backend is missing, and keeps
  models warm (`keep_alive` / the cross-encoder singleton). Keep that symmetry.

## Verify before assuming
- GPU **device index** ↔ physical card mapping (see the torch caveat) — confirm with `nvidia-smi`
  and `torch.cuda.get_device_name`.
- Whether your intended gen model fits the target card's VRAM — check before committing.
- `OLLAMA_SCHED_SPREAD` / `CUDA_VISIBLE_DEVICES` semantics against your installed Ollama / CUDA
  versions.
