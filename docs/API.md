# Loci MCP API Overview

Loci is the repository's FastMCP server for persistent investigation memory, code-aware retrieval, code-graph analysis, and local reasoning workflows. The current source-checked API surface spans roughly 75 MCP tools; this overview groups that surface into four practical categories and points you to the detailed reference for each one.

## API reference map

| Category | Detail doc | What it covers |
|---|---|---|
| Code graph, contracts, and audit | [docs/API_CODE_GRAPH.md](./API_CODE_GRAPH.md) | Graph ingest/query, code↔memory linking, impact analysis, contracts, wiring obligations, conflicts, causal inference, audit logs, subsystem reporting, and server health. |
| Investigations and findings | [docs/API_INVESTIGATION.md](./API_INVESTIGATION.md) | Investigation lifecycle, finding storage and resolution, provenance, search, entity lookup, verification, export/import, sharing, reflection, and investigation-scoped reasoning. |
| Memory lifecycle and retrieval | [docs/API_MEMORY.md](./API_MEMORY.md) | Hot/warm/cold memory tiers, retraction and restore, health checks, surfacing and routing prior findings, consolidation, procedure recall, and retrieval/helper utilities. |
| Swarm and local reasoning | [docs/API_SWARM_AND_REASONING.md](./API_SWARM_AND_REASONING.md) | `swarm_reason`, `llm_local`, reflection-loop tools, `memory_confidence`, the seed model, escalation stages, and the related CLI reasoning entry points. |

## Quickstart

Loci's default MCP transport is **stdio**. Inside this repository, Claude Code can discover the checked-in server config from the repo-root `.mcp.json`, which launches:

```json
{
  "mcpServers": {
    "loci": {
      "type": "stdio",
      "command": "mcp/.venv/bin/python",
      "args": ["mcp/server.py"]
    }
  }
}
```

That matches the server entrypoint in `mcp/server.py`, where `main()` calls `mcp.run(transport="stdio")` unless `LOCI_MCP_TRANSPORT` is explicitly set to `sse` or `streamable-http`.

Typical local setup from this checkout:

```bash
git clone https://github.com/rjmendez/loci
cd loci/mcp
python3 -m venv .venv && .venv/bin/pip install -e "."
cp ../.env.example .env   # fill in QDRANT_URL and OLLAMA_BASE_URL at minimum
.venv/bin/python server.py
```

To register the same server from outside the checkout, point your MCP client at the same stdio command with absolute paths, as shown in [mcp/README.md](../mcp/README.md#claude-code--mcp-wiring).

## FAQ

### What is a seed in the swarm?

A seed is one **independent full swarm pipeline run**, not random jitter on a single prompt. In `scripts/swarm_escalate.py`, each seed performs its own decompose → cheap-tier answer → triage → consensus gate → self-consistency → escalate flow, the runs fan out concurrently via `ThreadPoolExecutor(max_workers=effective_seeds)`, and only then are findings merged, deduplicated, and synthesized once. See [docs/API_SWARM_AND_REASONING.md](./API_SWARM_AND_REASONING.md).

### What's the difference between `swarm_reason` and `investigation_reason`?

`swarm_reason` is the general local swarm wrapper over `scripts/swarm_escalate.py`: it decomposes a topic into subtasks, runs cheap-tier answers, escalates selected items, and optionally uses multiple independent seeds before a final synthesis. `investigation_reason` is narrower and evidence-bound: it starts from one existing investigation, gates stored findings against the question, fans out up to five named analytical perspectives, then synthesizes converged and contested claims and can optionally persist the result back into the investigation. See [docs/API_SWARM_AND_REASONING.md](./API_SWARM_AND_REASONING.md) and [docs/API_INVESTIGATION.md](./API_INVESTIGATION.md).

### How do memory tiers work?

Each finding has a storage tier: `hot`, `warm`, or `cold`. `hot` findings stay searchable in Qdrant and also append a snippet into manifest notes for immediate in-context use; `warm` findings are the default searchable Qdrant-backed tier; `cold` findings remain in `findings.jsonl` only and are not vector-searchable. These storage tiers are separate from Mnemosyne's internal `working_memory` and `episodic_memory` substrate states. See [docs/API_MEMORY.md](./API_MEMORY.md).

### Do these tools rewrite investigations in place?

Usually no: Loci leans heavily on append-only state. Findings are appended to `findings.jsonl`, lifecycle changes such as `finding_resolve` are recorded as overlay updates, retractions are soft tombstones that `memory_restore` can reverse, and investigation history can also be reconstructed with `investigation_as_of`. See [docs/API_INVESTIGATION.md](./API_INVESTIGATION.md) and [docs/API_MEMORY.md](./API_MEMORY.md).

### Which document should I read first?

Start with the family closest to your task. Use [docs/API_INVESTIGATION.md](./API_INVESTIGATION.md) for case lifecycle and evidence handling, [docs/API_MEMORY.md](./API_MEMORY.md) for storage tiers and recall behavior, [docs/API_CODE_GRAPH.md](./API_CODE_GRAPH.md) for code-graph and contract tooling, and [docs/API_SWARM_AND_REASONING.md](./API_SWARM_AND_REASONING.md) for swarm execution, seeds, and local reasoning flows.
