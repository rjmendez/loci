# The "hermes" name in Loci

Audited 2026-08-27 at `400c3e2`. 1,013 occurrences across 138 tracked files.

The short version: **most of it is not stale branding.** It is either the name of
a live system Loci runs alongside, or a storage and configuration contract that
renaming would break. Six occurrences were genuinely stale and are fixed; the
rest are catalogued here so a future rename is a decision rather than a sweep.

## What is actually there

| what | hits | files | what renaming costs |
|---|---:|---:|---|
| Qdrant collection names | 332 | 49 | live data migration — 3,086 points |
| Environment variable names | 284 | 64 | redeploying every consumer |
| Filesystem paths under `~/.hermes/` | 222 | 81 | moving 182 MB + 1.6 GB of live data |
| References to the real Hermes | 71 | 22 | nothing — they are correct |
| Prose and other | 70 | 25 | nothing |
| Docker image / volume / entrypoint | 34 | 5 | breaks existing deployments |

## Hermes is a running system, not a former name

`~/.hermes/hermes-agent` is a 6.8 GB codebase with its own README, Dockerfile and
licence, and three processes were running during this audit:

    hermes_cli.main --profile mrpink gateway run
    profiles/mrpink/scripts/grounding_daemon.py
    profiles/mrpink/a2a_server/server.py

The third is Loci's own code (`Loci A2A Server v0.1.0`) deployed into a Hermes
profile directory. That is the shape of the whole relationship: Loci is a tenant
of a Hermes installation, sharing its home directory, its profile `.env`, and its
Mnemosyne database.

So every mention of `HERMES_PROFILE`, `~/.hermes/profiles/`, `hermes-agent`, the
`pre_llm_call` event name or "the Hermes shape" of a hook payload is **correct**
and must not be renamed. 71 occurrences are in this category.

## What is Loci's, wearing the old name

The Qdrant collections and the memory directory are Loci's own data. The
hermes-agent codebase references none of `hermes_memory`, `hermes_sessions`,
`hermes_verdicts` or `memory-sessions` — only `mnemosyne`, which is genuinely
shared. These are stale names on live storage:

| name | holds | rename path |
|---|---|---|
| `hermes_memory` | 2,769 points | Qdrant alias, then dual-read, then migrate |
| `hermes_sessions` | 317 points | same |
| `hermes_verdicts` | does not exist on the live instance | rename freely |
| `~/.hermes/memory-sessions` | 142 investigations + a 40 MB code graph | move with a compatibility symlink |

`hermes_verdicts` is worth noting separately: `memcheck/cli.py` creates it 384-dim
via `hash_embed` while `verdict_ops.py` writes 768-dim vectors and never creates
it. It is absent from the live instance. That is a defect, not just a name.

The 23 `HERMES_*` environment variables are read at roughly 62 sites. They can be
renamed behind a resolver that reads `LOCI_*` first and falls back, but the
consumers include the deployed Claude Code hooks in `~/.claude/hooks`, so the
rename is a redeploy as well as an edit — see `scripts/hooks/install.sh --check`.

## Fixed in this change

Six occurrences described Loci's own components as Hermes':

- `a2a_server/README.md` — "the Hermes MCP stack" is Loci's MCP stack.
- `a2a_server/client.py` — client for "the Hermes Memory server", which is
  `Loci A2A Server v0.1.0`.
- `scripts/a2a_watchdog.py` — docstring, Windows firewall rule name, and rule
  description, all naming the Loci A2A server as Hermes'.
- `mcp/server.py` — a user-facing health message reading "hermes runs on keyword
  fallback only".

The firewall rule needed more than a rename. The watchdog deletes and re-adds the
rule by name each run, so changing the constant alone would have left the old
`Hermes A2A Server (port)` rule in place forever while adding a second one. The
generated script now deletes the legacy name too.

## Deliberately not renamed

`hermes-mcp` and `hermes-a2a` are Docker image, volume and console-entrypoint
names. `docs/DEPLOYMENT.md` already records that they kept the old name.
Renaming them orphans existing volumes.
