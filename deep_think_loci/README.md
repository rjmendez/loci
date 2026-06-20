# deep-think-loci

**A multi-tier reasoning engine that runs as a Claude Code Workflow over a shared Loci/qdrant memory corpus** — the supported replacement for the `deep_think` MCP server's reasoning surface. Tiered models fan out across a problem, persist their findings to a Loci investigation with position + lineage tagging, and an opus tier synthesizes a grounded final answer.

> Status: **v3.2.0-beta**. The pure-Claude path is validated and reliable; see [CHANGELOG.md](./CHANGELOG.md) for the v1→v3.2 evolution and the empirical findings behind each change.

## Why

`deep_think` reasoning maps cleanly onto a Workflow (multi-pass → `pipeline`, fan-out → `parallel`, verify → judge panel) and runs on the session's Claude auth — no Anthropic API key in a configmap. Findings persist to a Loci investigation, so every run leaves an auditable, queryable corpus with cross-model influence lineage (`derived_from`).

## Shape (v3.2 — all-Claude)

```
Init        open the Loci investigation
Ideate      5 haiku generators (RAG-grounded), return ideas only — no store
Write       1 dedicated WRITER agent persists all ideas (returns real finding_ids)
VerifyIdeate load investigation for ground truth (per-target persistence check)
Final       2 opus half-syntheses (per-target grounding-gated, own the red-team)
            → 1 opus final (cross-target nightmares + integrity check)
```

Every tier reasons **only over grounding-gated evidence** (see below). The opus tiers own the adversarial red-team — the external uncensored tier was removed in v3.2 (it never persisted across 4 runs and opus covers it; see CHANGELOG).

## The two load-bearing patterns

1. **Dedicated-writer persistence.** Workflow subagents unreliably execute an `investigation_store` MCP call when it's one step in a multi-step task (RAG → generate → store) — they silently skip it or *fabricate* a confirmation. Fix: generation agents return schema-validated output **only**; a single-purpose **writer** agent does all the stores and returns the real `finding_id`s. Took persistence from ~40% to 100%.

2. **Cosine grounding gate (`grounding/ground_gate.py`).** RAG retrieval bleeds cross-target findings at cosine 0.35–0.59 — moderately similar, plausibly relevant, but *wrong topic* — and a non-verifying model will hallucinate over them. The gate embeds the query + each candidate (nomic, local) and keeps only those clearing a per-target threshold (**0.59**), dropping the bleed before any model reasons. Local, ~$0 marginal.
   - **Query per-target, never blended** — a blended multi-target query dilutes cosines and false-drops whole genuine targets (the v3 bug).
   - Drop-in upgrade: a trained classifier (`grounding/grounding_bleed_clf.joblib`) once it beats the cosine threshold on a larger corpus.

## Usage

The workflow runs via the Claude Code `Workflow` tool:

```
Workflow({ scriptPath: "deep_think_loci/workflows/deep-think-loci.js" })
```

Parameterize via `args` (all optional):

| arg | default | meaning |
|---|---|---|
| `run_id` | `dt-loci-005` | Loci investigation id (use a fresh one per run) |
| `targets` | 5 dama-gotchi targets | `[{name, focus}]` — the subsystems to reason about |
| `rag_collections` | `['dama_gotchi_code']` | qdrant collections for ideation grounding |
| `ideas_per_agent` | `10` | ideas per ideation generator |
| `ground_gate` | `~/.hermes/specialists/grounding/ground_gate.py` | gate script path (installed location) |
| `ground_threshold` | `0.59` | per-target cosine keep threshold |

The gate also runs standalone:

```sh
echo '[{"id":"x","text":"..."}]' | python3 deep_think_loci/grounding/ground_gate.py --query "<topic focus>" --threshold 0.59
```

## Install (deploy source → runtime)

The repo is the source of truth; `install.sh` deploys the workflow + gate to the `~/.hermes` runtime locations the workflow defaults to:

```sh
deep_think_loci/install.sh
```

Requires: the Loci MCP server, a reachable embeddings endpoint, and (for the gate's trained-model mode + the dataset builder) `pip install -r deep_think_loci/requirements.txt`.

**Config** follows Loci's `.env` conventions — `ground_gate.py` and `build_grounding_dataset.py` read the embeddings endpoint from `OLLAMA_BASE_URL` (no `/v1` suffix; `/v1/embeddings` is appended) and the model from `EMBED_MODEL` (default `nomic-embed-text`), falling back to a built-in default. No new env vars are introduced.

## Cost (measured, API-rate equivalent; runs are subscription-metered)

A full run ≈ **$5–8**, dominated by **cached context** (cache read/write), not output. There is a **~$0.15-per-agent floor** from the context each agent carries → cost scales with agent *count*; smaller shapes mean fewer agents. v3.2 (all-Claude, adversarial phase removed) ≈ $5 and spends **zero** external-provider tokens.

## The grounding specialist (`grounding/`)

| file | what |
|---|---|
| `ground_gate.py` | the cosine grounding gate (v0) + `--model` hook for the trained classifier |
| `build_grounding_dataset.py` | reproducible: harvest labeled pairs from a Loci corpus → train + eval |
| `grounding_dataset.jsonl` | 519 labeled examples (450 topical, 68 lineage, 1 hallucination) |
| `grounding_bleed_clf.joblib` | trained bleed-detector (LR on nomic pair-features) |
| `metrics.json` | CV AUC 0.957 / acc 0.847 (vs tuned-cosine 0.858 — data-starved; harvest-as-you-run) |

Each run mints more labeled pairs as a byproduct; rebuild with `build_grounding_dataset.py` as the corpus grows.

## Known limits

- `mnemo_mirror` is down in Loci's venv → findings persist to qdrant `hermes_memory` (RAG works) but don't mirror to mnemosyne; `pip install mnemosyne-memory` into Loci's venv to enable.
- The trained classifier doesn't yet beat a tuned cosine threshold (small corpus); the cosine gate is the shipped default.
- Entailment-grounding (lineage/hallucination signals) is too sparse to train — accumulate deliberately.
