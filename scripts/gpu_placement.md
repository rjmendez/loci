# Dual-GPU placement for the Loci inference tiers

> **This box is shared with the DAMA ant-trainer.** GPU placement is governed by ONE
> authoritative policy — `dama-gotchi/training/GPU_PACKING_POLICY.md` — because it is a single
> physical machine (WSL2 on `oxalis`). That policy assigns **GPU 0 = RTX 2080 Ti = ant-trainer
> primary** and **GPU 1 = RTX 4070 Ti = persistent inference**. This doc EXTENDS it for the Loci
> retrieval/generation tiers; it does not define a competing scheme. Where the two ever disagree,
> the DAMA packing policy wins for the training GPU.

## The hardware (session grounding)

- **[hardware]** This box is WSL2 on the `oxalis` host.
  - **RTX 4070 Ti** — visible to torch as `cuda:0` inside `mcp/.venv`
    (`torch 2.12+cu130`, `cuda.is_available()=True`). Local compute.
  - **RTX 2080 Ti (11 GB)** — the *second* GPU, exposed on the Windows/Ollama
    side. Ollama is reached over the network at
    `OLLAMA_BASE_URL=http://100.73.200.19:11434`.
- **[rerank]** `mcp/server.py` already reranks on GPU: a lazy
  `sentence_transformers.CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')`
  (server.py ~588–601), loaded on **`cuda:0` (the 4070 Ti)**, used by
  `rag_context_search`.
- **[retrieval]** Qdrant is a **separate CPU-only k3s host** — no GPU there.
  Embeddings are produced via **Ollama `nomic-embed-text` (768-dim)**, warm on GPU.
- **[gen]** Local generation shipped: `mcp/llm_local.py` → **`qwen2.5:3b`**,
  pinned via `keep_alive`.

## The contention problem

Two hot paths currently want GPU at the same time:

1. **Retrieval-side, `cuda:0` / 4070 Ti** — the CrossEncoder reranker (torch, in-venv)
   **and** whatever embedding calls resolve locally.
2. **Generation-side, Ollama** — `qwen2.5:3b` gen + `nomic-embed-text` embeddings, both
   served by the Ollama process (`100.73.200.19`).

If a large / batched generation model and the interactive reranker land on the *same*
GPU, a long gen occupies the SMs and VRAM and stalls rerank latency (and vice-versa).
The goal of placement is to keep the **latency-sensitive retrieval tier** (embeddings +
rerank) off the **throughput/heavy gen tier**.

## Recommended layout (reconciled with the DAMA packing policy)

| Tier | Workload | Target GPU | Priority rule |
|------|----------|-----------|-----|
| Training (bursty, DAMA) | `ant-trainer-*` k3s jobs | **2080 Ti (GPU 0)** | **owns this GPU** — highest priority per the packing policy |
| Retrieval (latency-sensitive, Loci) | CrossEncoder rerank (torch) + warm Ollama embed/gen | **4070 Ti (`cuda:0`)** | already there; keep it there |
| Generation (throughput / batched, Loci) | vLLM/TGI batched-gen server | **2080 Ti, opportunistic** | tenant only — **must yield to training**, never a persistent squatter |

Rationale, corrected: rerank is a small model hit on the critical path of every
`rag_context_search`; it stays on the 4070 Ti and must never queue behind a multi-second
generation. A batched-gen server is heavier and *can* use the 2080 Ti's headroom — **but the
2080 Ti is the ant-trainer's GPU**, so batched-gen runs there only opportunistically and
preemptibly (telemetry-gated: don't launch/keep it resident while an `ant-trainer` job holds
the card). Persistently squatting the 2080 Ti with a gen server — the earlier draft's plan —
would collide with every training burst. When in doubt, batched-gen shares the 4070 Ti and
accepts the contention rather than starving training.

## Levers

### torch side (the 4070 Ti reranker)
- The reranker is pinned to `cuda:0` in code. To force the *whole venv process* onto a
  specific physical GPU, launch it with `CUDA_VISIBLE_DEVICES` so `cuda:0` **is** the
  4070 Ti:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python -m mcp.server   # 4070 Ti only
  ```
  `CUDA_VISIBLE_DEVICES` remaps indices: the first listed device becomes `cuda:0`. Set it
  at process launch (it is read once at CUDA init).
  > Caveat: WSL2 ↔ Windows GPU enumeration ordering is not guaranteed to match
  > `nvidia-smi` index order. **Verify** which physical card `cuda:0` is
  > (`torch.cuda.get_device_name(0)`) before trusting the mapping — grounding only
  > confirms the 4070 Ti is *currently* `cuda:0`, not that ordering is stable across hosts.

### Ollama side (the gen / embed models)
Ollama is a single server that schedules across the GPUs *it* can see. Relevant env vars,
set **on the Ollama host** (`100.73.200.19`), not in this venv:

- **`CUDA_VISIBLE_DEVICES`** — restrict which physical GPUs Ollama uses. To dedicate the
  2080 Ti to a heavy/batched gen model, run that Ollama instance with
  `CUDA_VISIBLE_DEVICES` pointing at just the 2080 Ti.
- **`OLLAMA_SCHED_SPREAD=1`** — tells Ollama to **spread** a model across *all* visible
  GPUs rather than packing onto one. Use it when you *want* a big model to shard across
  both cards; **leave it unset (0)** when you want each model to stay pinned to one GPU so
  gen and rerank/embeddings don't collide.
- **`OLLAMA_MAX_LOADED_MODELS`** / **`OLLAMA_NUM_PARALLEL`** — cap concurrent resident
  models / parallel requests so a batched gen server doesn't evict the warm
  `nomic-embed-text`.
- **`OLLAMA_KEEP_ALIVE`** (or per-request `keep_alive`, what `gpu_warm.py` sends) — keep
  the hot models resident so a warm single-stream never re-eats the **~70s cold load**
  (`mcp/llm_local.py` [substrate]).

A clean split is **two Ollama endpoints**, each with its own
`CUDA_VISIBLE_DEVICES`: one on the 4070 Ti for embeddings, one on the 2080 Ti for the
heavy gen model. If you instead run a single Ollama over both cards, keep
`OLLAMA_SCHED_SPREAD` **off** and rely on `keep_alive` pinning so the small
embed/gen models stay put and the big model isn't forced to share.

## Interaction with `gpu_warm.py`

`scripts/gpu_warm.py` pins `qwen2.5:3b` and `nomic-embed-text` resident via
`keep_alive=-1` and reports `/api/ps` residency. It pins **whatever GPU(s) the Ollama it
talks to can see** — it does not itself choose a card. Placement is decided by the env
vars above on the Ollama host; `gpu_warm.py` then keeps the chosen models warm so the cold
load is paid once. Run the keeper (`--loop`) pointed at each Ollama endpoint you stand up.

## Combined design philosophy (shared with the DAMA ant-trainer)

The Loci GPU tiers and the DAMA training pipeline are the same box and should share one
discipline, not two. The DAMA training system already codifies patterns the Loci offload
tiers benefit from adopting:

- **One telemetry-driven packing policy** — `GPU_PACKING_POLICY.md` places work by live
  `nvidia-smi` headroom, not by hardcoded assumption. `gpu_warm.py` and any batched-gen
  launcher should read GPU telemetry and **yield to training** rather than blindly pinning.
- **Gated rollout for quality changes** — DAMA never ships a model to the fleet without
  shadow-eval + an NIS canary. The reranker upgrade (`RERANK_MODEL` → `bge-reranker-v2-m3`)
  is the retrieval analogue: it stays **default-OFF** until it's A/B'd on a held-out query
  set. Same for `query_expand` (prove it lifts recall before wiring it into the live path).
- **Fail-open, warm, cached** — the ant models degrade gracefully when a signal is missing;
  the Loci tiers already mirror this (every op fail-open; models kept warm via `keep_alive`
  / the cross-encoder singleton). Keep that symmetry.

## What the grounding does NOT settle (do not assume)

- Whether a **batched gen server** on the 2080 Ti actually exists yet — this doc proposes the
  layout; grounding only states the 2080 Ti is *available* and is the ant-trainer's GPU, so
  any gen server there must be preemptible w.r.t. training.
- Whether the 2080 Ti's 11 GB fits your intended "larger gen model" — verify VRAM against
  the specific model before committing.
- Exact WSL2 GPU index ↔ physical card mapping stability (see the torch caveat above).
- `OLLAMA_SCHED_SPREAD` / `CUDA_VISIBLE_DEVICES` semantics are Ollama/CUDA conventions, not
  facts asserted by the session grounding — confirm against your installed Ollama version.
