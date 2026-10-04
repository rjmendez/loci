# GPU offload architecture map

This is the implementation map for where Loci should run work in simple code, specialist local models, and general LLMs. It is grounded in the current repo: `mcp/backends.py`, `mcp/llm_tools.py`, `mcp/batched_gen.py`, `scripts/gpu_placement.md`, and `scripts/model_catalog.py`.

## 1) What stays as code

Keep the following in deterministic code and not in model inference layers:

- Retrieval orchestration: Qdrant fan-out, collection selection, score fusion, keyword rerank, MMR diversification, and recency/trust type weighting. This is the logic in `docs/ARCHITECTURE.md` and the grounding hook contract.
- Backend resolution and environment plumbing: `mcp/backends.py` must remain the one source of truth for env → local probe → config → fail-open fallback for Ollama endpoints. This is the stable deployment seam.
- Routing and batching control: `mcp/batched_gen.py` fans prompts out concurrently through the Ollama tier and keeps the output aligned 1:1 with the input prompts.
- Validation and safety gates: schema enforcement, JSON validation, tool-call guardrails, and evidence provenance checks should remain deterministic checks instead of being pushed into a model.
- Cheap local summation/triage logic: `query_expand`, `semantic_dedup`, `semantic_relevance`, `classify_text`, and `compress_text` should stay in code wrappers over local models, but the actual “main decision” logic remains explicit and bounded.

This is the right place for rules, timeouts, fail-open behavior, and scoring because those are the expensive-to-test parts that can be audited with code reviews and small unit tests.

## 2) What moves to specialist local models

The repo already points to a clear specialist-model pattern:

- code specialist: `qwen2.5-coder:7b` (`scripts/model_catalog.py`)
- math specialist: `Qwen2.5-Math-7B-Instruct` (GGUF variant, same catalog)
- safety specialist: `llama-guard3:8b` (catalog)
- tool-calling specialist: `Watt-Tool-8B` (catalog)

These belong on dedicated local endpoints, not the general batched generation lane:

- code-analysis tasks: patch review, code-graph grounding, refactor safety, bug triage, symbol-level reasoning
- math / logic tasks: proofs, numerical reasoning, constrained arithmetic, consistency checks
- safety / policy tasks: prompt-injection gating, harmful-content classification, policy checks
- tool-calling / structured action planning: JSON tool invocations and function-call planning

Keep specialist workloads out of the generic fan-out path unless they are small and broad; otherwise they queue behind the general tier and add latency to every other task.

## 3) What stays on general LLMs

General assistant / frontier LLMs should keep the following:

- end-user narrative synthesis and final answer writing
- broad, open-ended reasoning that requires semantic flexibility and a wide world model
- final case summaries that must read naturally and integrate many sources
- tasks where the model is the primary quality bottleneck and local specialists are not meaningfully better

In other words: keep the expensive “main brain” on the general LLM, but offload the high-volume structured work to local specialist models and deterministic orchestration.

## 4) GPU tier placement

The repo already has the correct physical placement model:

| Tier | Workload | GPU placement |
|---|---|---|
| Retrieval / latency-critical | reranker + warm embeddings | `cuda:0` (inference GPU) |
| Generation | batched/heavy Ollama gen | separate GPU if available |
| Specialist models | code/math/safety/tool-calling specialists | a dedicated Ollama model per role, ideally on a second card |
| Shared fallback | Ollama single prompts / emergency generation | same GPU as whichever model is loaded, but not at the expense of rerank |

This is derived from `scripts/gpu_placement.md`.

Concrete placement rules:

- Keep the reranker and embeddings on the inference GPU (`cuda:0`). They are on the critical path of every retrieval call and must not queue behind a long generation.
- Put generic generation on a second GPU when available. If only one GPU exists, run generic generation there but accept the contention; do not let the reranker be starved.
- For specialist roles, reserve a dedicated Ollama model per role or per one small cluster of roles. Code-heavy sweeps should not sit behind the generic batch queue.
- Use `CUDA_VISIBLE_DEVICES` to pin the physical card and verify with `nvidia-smi` + `torch.cuda.get_device_name(0)` before trusting the mapping.
- Keep `keep_alive` on warm models; use worker limits so heavy generation does not evict the embedding model.

## 5) Practical Loci map

This is the intended split for the repo as it exists today:

- Stay as code:
  - backend resolution (`mcp/backends.py`)
  - `rag_context_search` orchestration and weighting
  - Qdrant fan-out and dedup logic
  - `llm_tools` wrappers and timeout/fail-open behavior
  - route selection, prompts, and invocation contracts
- Specialist local models:
  - `verify_finding`, `query_expand`, `compress_text`, `classify_text`, `semantic_dedup`, `semantic_relevance`
  - swarm fan-out and specialist role decomposition
  - code, math, safety, and tool-calling dedicated Ollama models
- General LLMs:
  - final synthesis, user-facing explanation, broad reasoning chains, and integration narrative

## 6) Implementation order

1. Keep the shared Ollama endpoint as the generic generation tier.
2. Add optional per-role model resolvers, following the existing `backends.py` pattern.
3. Farm code, math, safety, and tool-calling tasks to those role-specific models when configured.
4. Keep the retrieval tier pinned to the inference GPU and never let batched generation steal that lane.
5. Leave the general LLM for end-stage synthesis only.

This keeps the architecture simple, testable, and consistent with the repo’s current docs and code, while still giving a real GPU path for the highest-volume local inference work.
