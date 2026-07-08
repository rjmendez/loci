# reembed_daemon — GPU batch re-embed / index refresh (SAFE, dry-run default)

Incremental, idempotent batch job that (re)embeds Qdrant point vectors on the local GPU
via the Ollama nomic path. Use it after an embed model/version change, or to backfill
points that are missing a vector.

## Safety model (read first)

- **Default is DRY-RUN.** Without `--apply` the job only *scans* and reports how many
  points *would* be re-embedded. It writes nothing and does not even call the GPU.
- **`--apply` is the only mutating path.** There is no destructive default.
- **Fail-open, per batch.** An embed or upsert error (or a Qdrant hiccup) is logged and
  the run continues; it never raises. A total scan failure returns `degraded:true` and
  exit code 1 without having written anything.

## What counts as "stale" (incremental targeting)

A point is targeted only if:
- it has **no vector** (missing / backfill), OR
- its payload `embed_model` differs from the current target, OR
- its payload `embed_version` differs from the current target.

On re-embed we stamp `embed_model` + `embed_version` into the payload, so a second run
targets nothing — **safe to re-run**.

## Usage

```bash
# 1) dry-run: how many points WOULD be re-embedded? (writes nothing)
python3 scripts/reembed_daemon.py --collection hermes_memory

# 2) apply: actually re-embed the stale/missing points
python3 scripts/reembed_daemon.py --collection hermes_memory --apply

# named-vector collection, custom batch, new version marker
python3 scripts/reembed_daemon.py --collection agent_core_chunks \
    --vector-name dense --batch-size 128 --embed-version 2 --apply
```

Always run the dry-run first, eyeball `targeted`, then re-run with `--apply`.

## Config (env)

| var | purpose | default |
|-----|---------|---------|
| `QDRANT_URL` | Qdrant endpoint (separate k3s host, CPU) | — (required) |
| `QDRANT_API_KEY` | Qdrant key; falls back to `~/.claude/settings.json` `mcpServers.hermes_memory.env` | "" |
| `OLLAMA_BASE_URL` / `OLLAMA_URL` | Ollama for the warm-GPU nomic embed path | — |
| `EMBED_MODEL` | current embed model tag | `nomic-embed-text` |
| `EMBED_VERSION` | current embed version marker | `1` |

The embed step reuses `mcp/embed_ops.py::embed_texts` (768-dim nomic, warm on GPU). The
GPU work here is **embedding, not generation**.

## Report fields

`scanned`, `missing_vector`, `stale_meta`, `targeted` (what dry-run counts), `reembedded`
(what apply actually wrote), `embed_batches` / `upserted_batches`, `errors[]`, `degraded`.

## Flags

`--collection` (required) · `--apply` · `--batch-size` (default 64) ·
`--page-limit` (default 256) · `--embed-model` · `--embed-version` · `--vector-name`.

## Notes / caveats

- Source text is read from the first present payload field of:
  `document, text, content, summary, title`. A targeted point with no text in payload is
  skipped and noted in `errors` (nothing to embed).
- `qdrant_client` and `embed_fn` are **injectable** (both default `None` → lazy-resolve
  from env), so `import reembed_daemon` hard-requires nothing and tests stub both.
- **Grounding gap:** the exact payload key names (`embed_model`, `embed_version`) and the
  document text field are conventions chosen here, not verified against the live
  hermes_memory / agent_core_chunks schema. Confirm the real payload keys before
  `--apply` on production data, and override `--vector-name` if those collections use
  named vectors. Central integration wires this into the server later.
```
