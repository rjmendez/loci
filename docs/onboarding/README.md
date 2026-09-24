# Loci Onboarding

_Last reviewed: 2026-09-17 — this is a rolling document. Keep it current: when a subsystem, tool, or setup step changes, update the relevant page in the same change._

Welcome. This doc set orients **both new people and new agents** to Loci: what it is, how it is built, the tools it exposes, how an agent should use it, and how to run it locally.

## What Loci is

Loci is a **persistent memory system for AI agents** that survives across sessions. It gives agents long-term memory through:

- **RAG** (Retrieval-Augmented Generation) — semantic recall of prior findings
- **Investigation tracking** — append-only, searchable units of work
- **Code-graph linking** — findings tied to the code symbols they touch
- **Multi-agent reasoning** — bounded local-model swarms and verification

Loci runs as an **MCP (Model Context Protocol) server** alongside Claude Code and exposes roughly 75 tools. It stores findings in JSONL files, searches them via Qdrant (a vector database), and links them to code symbols via an embedded graph database (LadybugDB).

The design is grounded in cognitive science:

- **Ebbinghaus spacing** (FSRS decay/refresh intervals)
- **Tulving's episodic/semantic distinction** (memory tiers)
- **Miller/Cowan working-memory limits** (small default recall top-k)

## Who this is for

- **New people** — engineers and operators standing up, running, and maintaining Loci. Start with [running.md](running.md) and [architecture.md](architecture.md).
- **New agents** — models that will *use* Loci as their memory and reasoning substrate. Start with [for-agents.md](for-agents.md) and [tools.md](tools.md).
- **FlyBrain readers** — start with [../FLYBRAIN_GUIDE.md](../FLYBRAIN_GUIDE.md), then [../FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md](../FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md), [../FLYBRAIN_DATASET_PROVENANCE_MATRIX.md](../FLYBRAIN_DATASET_PROVENANCE_MATRIX.md), and [../FLYBRAIN_IO_TO_LOCI_MAPPING.md](../FLYBRAIN_IO_TO_LOCI_MAPPING.md). Use [../FLYBRAIN_ARCHIVE.md](../FLYBRAIN_ARCHIVE.md) for deeper technical or historical notes.
- **FlyBrain harness operators** — enforce write-path controls from [../FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md](../FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md) before running filesystem-writing jobs.

## The doc set

| Page | What it covers |
|------|----------------|
| [architecture.md](architecture.md) | Subsystem map: the four data stores, the ten major components, and how they fit together. |
| [tools.md](tools.md) | The MCP tool surface, grouped by function, with usage notes for the critical tools. |
| [for-agents.md](for-agents.md) | How an agent should *use* Loci: ground first, reason with swarm/local models, store and verify memory. Concrete patterns. |
| [running.md](running.md) | Setup: backends and local models, infrastructure, testing, conventions, and known gotchas. |

## One-paragraph orientation

A user message triggers a per-turn grounding hook that embeds intent, searches Qdrant, and injects a context block into the model's window. The model answers and records findings via `investigation_store`; findings go to JSONL and Qdrant simultaneously. Multi-agent reasoning (`swarm_reason`, deep-think workflows) can wrap the same investigation. Background grooming reconciles the index, proposes tags, links findings to code symbols, and generates summaries. The code graph lives in LadybugDB; the A2A mesh server broadcasts context to peers; and Ebbinghaus decay can trigger optional refresh for fading memories.

## Reference docs in-repo

The canonical deep-dive docs live under `docs/`:

- `docs/CONCEPTS.md` — start here for the mental model
- `docs/ARCHITECTURE.md`, `docs/COMPONENTS.md`, `docs/COGNITIVE_FOUNDATIONS.md`
- `docs/OPERATIONS.md`, `docs/DEPLOYMENT.md`, `docs/CALLGRAPH.md`
- `docs/FLYBRAIN_GUIDE.md` — short FlyBrain entry point and canonical reading path
- `docs/FLYBRAIN_DATA_PROVENANCE_GUIDANCE.md`, `docs/FLYBRAIN_DATASET_PROVENANCE_MATRIX.md`, `docs/FLYBRAIN_IO_TO_LOCI_MAPPING.md`
- `docs/FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md` — allowlist-only FlyBrain harness path policy for configured storage-root safety
- `docs/FLYBRAIN_REASONING_GLOSSARY.md`, `docs/FLYBRAIN_ARCHIVE.md`
- `mcp/README.md` — the MCP server and tool surface

This onboarding set is the front door; those docs are the detail.
