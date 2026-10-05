# Loci — A2A Memory Server

Default port: **8201**

## What this does

Exposes Mnemosyne memory operations over the A2A JSON-RPC protocol so other
mesh agents can read and write memory without going through the Loci MCP stack.

Protocol: JSON-RPC 2.0 over HTTP POST to `/a2a`
Auth: Bearer token (`LOCI_A2A_TOKEN`) + optional TOTP (`LOCI_A2A_TOTP_SEED`)
Agent card: `GET /.well-known/agent.json`

## Trust-boundary controls (b2b lanes)

- `tasks/send` rejects malformed envelopes (`params` object, `skill_id` non-empty
  string, `input` object, `message` string).
- Command routing is explicit allowlist-only:
  - inbound dispatch only from the local skill map
  - outbound peer fan-out only to `memory_remember` / `memory_prime`
- Peer lane envelopes include `_boundary` metadata with
  `lane`, `idempotency_key`, `artifact_sha256`, `origin_agent_id`, and `ts`.
- Inbound `_boundary` metadata is validated for lane/skill allowlist, artifact
  SHA-256 integrity, and idempotency replay.
- Accepted/rejected boundary operations emit structured `boundary_receipt` logs.

Optional env:
- `LOCI_A2A_IDEMPOTENCY_TTL_S` (default `3600`) sets replay window TTL for
  `(sender, skill_id, idempotency_key)`.

## Agent card and node profile

`GET /.well-known/agent.json` (public) says what this node is: `name`, `version`, `url`, the **skills it
actually serves**, how to authenticate, and a short summary of what it has. Nothing in it is hard-coded to a
device, so each node advertises only its own hardware, sensors and data.

- **`url`** is `LOCI_A2A_URL`. Set it to an address peers can reach (the node's tailnet or LAN address). A
  loopback URL on a node that listens beyond loopback is reported at startup and as
  `advertised_url_is_loopback` in `/health`, because a peer that follows the card cannot reach it.
- **`skills`** is the served set, from `LOCI_A2A_SKILLS` (comma-separated; default all). A skill that is not
  listed is refused as unknown, not just hidden. `privileged: true` marks skills that are refused unless the
  caller is a privileged sender (`LOCI_A2A_PRIVILEGED_SENDERS`) and has proved who it is.
- **`resources`** is the public summary of the node profile: hardware names, sensor names and kinds, data
  names and kinds. Locations and details are not in it.
- **`GET /a2a/extended-card`** needs the same credentials as `/a2a`. It adds the full profile and a live
  inventory: host, GPUs, serial/USB/video/audio devices, storage, Ollama models, Qdrant collections with point
  counts, and the memory store's size and row count. A probe that does not apply reports
  `{"available": false, "reason": ...}`. The same inventory is the `device_inventory` skill. It is cached for
  30 seconds. Tokens, seeds and keys are never part of it.

`LOCI_A2A_PROFILE` points at a JSON file the operator writes. Only these fields are read; strings are clipped to
300 characters, lists to 50 rows, and a file over 64 KB is ignored (with a log line):

```json
{
  "description": "Field laptop: DAMA stack, local models, no GPU.",
  "summary": "i7 laptop, 15 GB, no GPU",
  "hardware": [{"name": "Intel i7-8550U", "detail": "4 cores / 8 threads"}],
  "sensors": [{"name": "CubeCell radiation counter", "kind": "radiation", "status": "intermittent"}],
  "data": [{"name": "DAMA telemetry", "kind": "timeseries", "description": "InfluxDB", "where": "influxdb:8086"}],
  "notes": ["Free disk is tight."]
}
```

Describe only what the node really has and can reach. The live inventory is the check on the profile: a
profile that claims a GPU on a node whose inventory shows none is a profile bug.

`LOCI_A2A_INIT_DB=1` creates the Mnemosyne SQLite file and its `memories` table when missing, for a fresh node.
It never alters an existing database.

## Signed requests (Ed25519)

A caller can prove who it is with a key instead of a shared token. It sends `X-Agent-ID` and
`X-Signature`; the server checks the signature against the public key registered for that agent.
A signed caller is bound to its agent id (the `sender` field cannot name anyone else), skips TOTP,
and may call a destructive skill only if listed in `LOCI_A2A_PRIVILEGED_SENDERS`. A signature that
does not verify is refused outright; it never falls back to a bearer token.

Register keys with `PEER_PUBKEYS_DIR` (a directory of `<agent_id>.pub` PEM files) and/or
`PEER_PUBKEYS_JSON` (`{"agent_id": "<PEM>"}`); a malformed entry is skipped and named in the log.
`/health` lists `registered_peers`, `signature_required` and `signature_min_version`, never keys.

| Setting | Meaning | Default |
|---|---|---|
| `A2A_REQUIRE_SIGNATURE=1` | refuse everything that is not validly signed (the server refuses to start without the `cryptography` package) | off |
| `A2A_SIGNATURE_MIN_VERSION` | `2` refuses v1 | `1` |
| `A2A_SIGNATURE_MAX_SKEW_S` | accepted clock skew of `X-Timestamp` | `60` |
| `A2A_SIGNING_KEY_FILE`, `PEER_A2A_SIGNED_URLS`, `A2A_SIGNING_VERSION` | sign this node's outbound peer calls (instead of bearer + TOTP) for the listed peers | off, version `2` |

Two wire versions:

- **v1** signs the raw body only. This is what the agent-mesh clients (`agent-mesh/a2a/auth.py`)
  send. Its `X-Timestamp` and `X-Request-ID` are not covered by the signature, so a captured request
  can be replayed with fresh headers; the server only refuses a byte-identical replay. Kept so the
  existing fleet keeps working.
- **v2** (`X-Signature-Version: 2`) signs `a2a-sig-v2`, agent id, timestamp, nonce, method, path and
  the SHA-256 of the body, joined by newlines. Nothing in it can be changed or replayed. `client.py`
  signs v2 when `A2A_SIGNING_KEY_FILE` is set. Behind a reverse proxy that rewrites the path, sign the
  path the server sees.

Move a fleet over by registering keys, switching callers to v2, then setting
`A2A_SIGNATURE_MIN_VERSION=2` and, when every caller signs, `A2A_REQUIRE_SIGNATURE=1`.

## Docker

The image needs files from `mcp/` (the investigation ACL helpers), so build from the repository root:

```
docker build -f a2a_server/Dockerfile -t loci-a2a .
```

`docker compose` does this already (`context: .`). Building with `a2a_server/` as the context fails
at the `COPY mcp/...` step on purpose: an image without those helpers would withhold every
investigation hit from `rag_search`.

## Skills

| skill_id            | What it does |
|---------------------|--------------|
| `memory_recall`     | FTS5 + Qdrant semantic search across working_memory, episodic_memory, mnemosyne collection |
| `memory_remember`   | Write a memory tagged with caller's sender/agent_id |
| `memory_stats`      | SQLite row counts + Qdrant collection sizes |
| `session_search`    | Semantic search over `loci_sessions` Qdrant collection |
| `memory_sleep`      | Trigger Mnemosyne consolidation via dashboard API |
| `rag_search`        | RAG-style retrieval: hybrid search + context assembly for grounding LLM prompts |
| `context_broadcast` | Broadcast a context update to all subscribed mesh agents |

## Quick start

### 1. Add secrets to .env

```
LOCI_A2A_TOKEN=<generate a strong token>
LOCI_A2A_TOTP_SEED=<base32 seed — optional, omit to disable TOTP>
LOCI_A2A_URL=http://<your-host>:8201
HERMES_AGENT_ID=<your-agent-id>
```

### 2. Start manually (dev / test)

```bash
cd /path/to/loci/a2a_server
python3 server.py
```

### 3. Install as user systemd service (persistent)

```bash
mkdir -p ~/.config/systemd/user
cp loci-a2a.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now loci-a2a
systemctl --user status loci-a2a
journalctl --user -u loci-a2a -f
```

### 4. Smoke test

```bash
# Health (no auth needed)
curl -s http://localhost:8201/health | python3 -m json.tool

# Agent card
curl -s http://localhost:8201/.well-known/agent.json | python3 -m json.tool

# Call a skill (requires LOCI_A2A_TOKEN)
TOKEN="<your-token>"
curl -s -X POST http://localhost:8201/a2a \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":"1","method":"tasks/send",
       "params":{"skill_id":"memory_stats","message":"","input":{},"sender":"test"}}' \
  | python3 -m json.tool

# Or use the client CLI
python3 client.py health
python3 client.py stats
python3 client.py recall "my query"
python3 client.py sessions "search term"
python3 client.py remember "Test memory from CLI" --sender my-agent
```

## Calling from a peer agent (Python async pattern)

```python
import aiohttp, pyotp, uuid

LOCI_ENDPOINT = "http://<your-host>:8201/a2a"
LOCI_TOKEN    = os.environ["LOCI_A2A_TOKEN"]
LOCI_TOTP     = pyotp.TOTP(os.environ["LOCI_A2A_TOTP_SEED"])  # if TOTP enabled

payload = {
    "jsonrpc": "2.0",
    "id": str(uuid.uuid4()),
    "method": "tasks/send",
    "params": {
        "skill_id": "memory_recall",
        "message": "search query",
        "input": {"query": "search query", "top_k": 5},
        "sender": "my-agent"
    }
}
headers = {
    "Authorization": f"Bearer {LOCI_TOKEN}",
    "X-TOTP": LOCI_TOTP.now(),   # omit if TOTP disabled
    "Content-Type": "application/json"
}
async with aiohttp.ClientSession() as sess:
    async with sess.post(LOCI_ENDPOINT, json=payload, headers=headers) as r:
        result = await r.json()
# result["result"]["output"]["memories"] -> list of matching memories
```

Or use the client helper directly:

```python
sys.path.insert(0, '/path/to/loci/a2a_server')
from client import LociMemoryClient
c = LociMemoryClient(sender="my-agent")
memories = await c.memory_recall("search query")
```

## Design notes

- Auth: Bearer token + optional TOTP. Set `LOCI_A2A_TOKEN` in your `.env`.
  Token is read at startup; restart the server after rotating.
- Memory writes are tagged with `sender` (caller's agent_id) in `metadata_json`
  so cross-agent provenance is preserved.
- Qdrant search uses the `dense` named vector (matches the upsert format used by
  `session_end_sync.py` and `state_db_qdrant_sync.py`).
- SQLite FTS uses `fts_working` (has content column) and `fts_episodes` (external
  content via rowid join) — both from the Mnemosyne schema.
- Phase 2 additions (not yet implemented): Redis inbox fallback,
  streaming SSE, Qdrant write-back on `memory_remember`, TOTP onboarding seeds.
