# The "hermes" name in Loci

Audited and migrated 2026-08-27. Loci was configured and stored under `HERMES_*`
names because it started life inside a Hermes installation. The names are now
`LOCI_*`, with the old spellings still accepted so an existing deployment keeps
working.

Hermes itself is not gone and is not a former name for Loci: `~/.hermes/hermes-agent`
is a 6.8 GB codebase with processes running, and Loci runs inside it. Names that
refer to Hermes are unchanged.

## What moved

| | from | to | how an existing install keeps working |
|---|---|---|---|
| Environment (29 vars, 322 sites) | `HERMES_*` | `LOCI_*` | `mcp/legacy_env.py` maps the old spelling onto the new one at startup; the new name wins if both are set |
| Qdrant collections | `hermes_memory`, `hermes_sessions`, `hermes_verdicts` | `loci_*` | Qdrant aliases point the new names at the existing collections |
| Memory directory | `~/.hermes/memory-sessions` | `~/.loci/memory-sessions` | resolved at runtime: the new path if it exists, else the old one. Nothing is moved implicitly |

## What deliberately did not move

**Names that refer to the Hermes installation Loci runs inside.** Renaming these
breaks that integration:

`HERMES_PROFILE`, `HERMES_HOME`, `HERMES_VENV_SITE`, `HERMES_SUBAGENT`,
`HERMES_AGENT_ID`, `~/.hermes/profiles/`, `hermes-agent`, the `pre_llm_call`
event name, and every description of "the Hermes shape" of a hook payload.

**Docker image, volume and container names** — `hermes-mcp`, `hermes-a2a`,
`hermes-memory-ollama-1`. Renaming orphans existing volumes and containers.
`docs/DEPLOYMENT.md` records that they kept the old name.

**The legacy MCP registration name.** `scripts/ebbinghaus_consolidation.py`,
`scripts/reembed_daemon.py` and `scripts/qdrant_payload_indexes.py` read the
Qdrant key from `~/.claude/settings.json` under `loci`, falling back to
`hermes_memory`. That fallback is a fact about settings files that already exist
and must keep the old spelling.

**Verbatim regression fixtures** under `scripts/callgraph/fixtures/regress/`.
They are snapshots of historical code and their tests assert they are unchanged.

## Migrating a deployment

Nothing is required — the compatibility paths cover an in-place upgrade. To
finish the move:

1. **Collections.** Create the aliases, or rename the collections and drop them:

       POST /collections/aliases
       {"actions":[{"create_alias":{"collection_name":"hermes_memory","alias_name":"loci_memory"}}]}

   Done on the reference host for `loci_memory` (2,771 points) and
   `loci_sessions` (318 points).

2. **Environment.** Rename `HERMES_*` to `LOCI_*` in `.env`, systemd units and
   any hook wrappers. `mcp/legacy_env.py` lists every pair.

3. **Memory directory.** `mv ~/.hermes/memory-sessions ~/.loci/memory-sessions`.
   The resolver prefers the new path once it exists. 182 MB on the reference
   host, including a 40 MB code graph — stop the MCP server first.

4. **Hooks.** `scripts/hooks/install.sh` deploys `legacy_env.py` alongside them,
   so a hook wrapper exporting the old names still works. `--check` reports drift.

## Still open

`loci_verdicts` does not exist on the live instance, and `mcp/memcheck/cli.py`
creates it 384-dim via `hash_embed` while `mcp/verdict_ops.py` writes 768-dim
vectors and never creates it. That is a defect, not a naming problem.
