# GPU-aware batch orchestrator (Ollama)

An Airflow-lite scheduler that keeps GPUs busy running local LLM (Ollama) jobs without
oversubscribing VRAM. It reads per-GPU free VRAM via `nvidia-smi`, admits a queued job
only when its target endpoint's GPU pool has marginal headroom (a model already resident
costs zero marginal VRAM), dispatches concurrently, honors a dependency DAG with retries,
proactively unloads a finished model when queued jobs need the VRAM, and writes each
job's result to JSON.

## Install
Python 3 (stdlib only) and a reachable Ollama endpoint. `nvidia-smi` must be on PATH or
pointed to via `NVIDIA_SMI_PATH`.

## Configuration (environment variables)
Nothing environment-specific is hardcoded. Defaults are local/generic.

| Var | Meaning | Default |
|-----|---------|---------|
| `GPU_ENDPOINT_MULTI` | Ollama endpoint that may span several cards (auto-splits large models) | `http://127.0.0.1:11434` |
| `GPU_ENDPOINT_GPU0` | Ollama endpoint pinned to GPU0 (started with `CUDA_VISIBLE_DEVICES=...`) | `http://127.0.0.1:11435` |
| `NVIDIA_SMI_PATH` | path to the `nvidia-smi` binary | `nvidia-smi` |
| `VRAM_GUARD_GB` | VRAM kept free per GPU as slack | `0.8` |
| `GPU_MULTI_MAX_INFLIGHT` / `GPU_GPU0_MAX_INFLIGHT` | per-endpoint concurrency caps | `3` / `2` |

```bash
export GPU_ENDPOINT_MULTI=http://<your-ollama-host>:11434
export NVIDIA_SMI_PATH=/usr/lib/wsl/lib/nvidia-smi   # e.g. on WSL2
```

Note: CUDA device order can differ from `nvidia-smi` order; confirm which physical card
`CUDA_VISIBLE_DEVICES=N` selects (the ollama serve log prints the GPU name), and make each
endpoint's `gpus` indices match `nvidia-smi`.

## Usage
```bash
python3 scripts/orchestrator.py --jobs scripts/orchestrator_jobs_example.jsonl --out out/ --once
```
Flags: `--jobs` (required, jsonl), `--out` (default `out`), `--poll` seconds (default 4),
`--heartbeat` file (default `orchestrator.heartbeat`), `--once` (exit when the queue drains).

## Job schema (`jobs.jsonl`, one JSON object per line)
| Field | Required | Notes |
|-------|----------|-------|
| `id` | yes | unique; names the output file |
| `model` | yes | Ollama model tag |
| `endpoint` | no | endpoint key; omit to let any fitting endpoint take it |
| `est_vram_gb` | no | marginal VRAM if not already resident (default 5.0) |
| `prompt` / `prompt_file` | yes | inline prompt or a path to read |
| `system` | no | system prompt |
| `options` | no | Ollama options (`temperature`, `num_predict`, `num_ctx`, ...) |
| `depends_on` | no | job ids that must be `done` first |
| `priority` | no | lower = scheduled sooner (default 0) |

## Output
`out/<job_id>.json` — the job record (minus the expanded prompt) plus the result:
`status` (`done`/`failed`), timing, eval counts, or error. A heartbeat file is written
each tick for cron/watchdog integration. Exit 0 on clean drain; exit 1 if no endpoint is
reachable at startup.
