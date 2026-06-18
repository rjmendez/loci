# Loci — A2A Memory Server

Default port: **8201**

## What this does

Exposes Mnemosyne memory operations over the A2A JSON-RPC protocol so other
mesh agents can read and write memory without going through the Hermes MCP stack.

Protocol: JSON-RPC 2.0 over HTTP POST to `/a2a`
Auth: Bearer token (`HERMES_A2A_TOKEN`) + optional TOTP (`HERMES_A2A_TOTP_SEED`)
Agent card: `GET /.well-known/agent.json`

## Skills

| skill_id            | What it does |
|---------------------|--------------|
| `memory_recall`     | FTS5 + Qdrant semantic search across working_memory, episodic_memory, mnemosyne collection |
| `memory_remember`   | Write a memory tagged with caller's sender/agent_id |
| `memory_stats`      | SQLite row counts + Qdrant collection sizes |
| `session_search`    | Semantic search over `hermes_sessions` Qdrant collection |
| `memory_sleep`      | Trigger Mnemosyne consolidation via dashboard API |
| `rag_search`        | RAG-style retrieval: hybrid search + context assembly for grounding LLM prompts |
| `context_broadcast` | Broadcast a context update to all subscribed mesh agents |

## Quick start

### 1. Add secrets to .env

```
HERMES_A2A_TOKEN=<generate a strong token>
HERMES_A2A_TOTP_SEED=<base32 seed — optional, omit to disable TOTP>
HERMES_A2A_URL=http://<your-host>:8201
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

# Call a skill (requires HERMES_A2A_TOKEN)
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
LOCI_TOKEN    = os.environ["HERMES_A2A_TOKEN"]
LOCI_TOTP     = pyotp.TOTP(os.environ["HERMES_A2A_TOTP_SEED"])  # if TOTP enabled

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

- Auth: Bearer token + optional TOTP. Set `HERMES_A2A_TOKEN` in your `.env`.
  Token is read at startup; restart the server after rotating.
- Memory writes are tagged with `sender` (caller's agent_id) in `metadata_json`
  so cross-agent provenance is preserved.
- Qdrant search uses the `dense` named vector (matches the upsert format used by
  `session_end_sync.py` and `state_db_qdrant_sync.py`).
- SQLite FTS uses `fts_working` (has content column) and `fts_episodes` (external
  content via rowid join) — both from the Mnemosyne schema.
- Phase 2 additions (not yet implemented): Redis inbox fallback,
  streaming SSE, Qdrant write-back on `memory_remember`, TOTP onboarding seeds.
