#!/usr/bin/env python3
"""
Loci A2A Server v0.1.0
Exposes Mnemosyne memory operations over the A2A JSON-RPC protocol.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RUNTIME REQUIREMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Python: 3.11  (venv: ~/.hermes/hermes-agent/venv/bin/python3)

Pip packages (see requirements.txt for pinned versions):
  fastapi==0.133.1     HTTP server, dependency injection, Bearer auth
  uvicorn==0.41.0      ASGI runner
  starlette==1.0.1     fastapi dep — Request, JSONResponse
  pydantic==2.13.4     fastapi dep — validation
  aiohttp==3.13.4      async HTTP client for Qdrant + Ollama
  pyotp==2.9.0         TOTP (RFC 6238) for X-TOTP header auth

stdlib (no install needed):
  os, uuid, json, sqlite3, logging, datetime, typing, asyncio, sys

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENVIRONMENT VARIABLES  (loaded from ~/.hermes/.env at startup)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Required:
  HERMES_A2A_TOKEN          Bearer token callers must supply.
                            REQUIRED — server exits at startup if unset.
                            Generate: python3 -c "import secrets;print(secrets.token_hex(32))"

Optional / tunable:
  HERMES_A2A_HOST           Bind address.  Default: 0.0.0.0
  HERMES_A2A_PORT           Bind port.     Default: 8201
  HERMES_A2A_URL            Public base URL injected into the agent card.
                            Default: http://127.0.0.1:8201
  HERMES_A2A_TOTP_SEED      Base32 TOTP seed (RFC 6238).  If set, callers
                            must include a valid X-TOTP header.
                            Default: '' (TOTP disabled)
  HERMES_AGENT_ID           Agent identity tag written into memory metadata.
                            Default: 'hermes-agent'

Qdrant (shared with session_end_sync.py + state_db_qdrant_sync.py):

  QDRANT_URL                Default: http://localhost:6333
  QDRANT_API_KEY            Qdrant API key.  No default — set in .env.

Ollama embedding (shared with sync scripts):
  MNEMOSYNE_EMBEDDING_API_URL  Default: http://localhost:11434/v1
  MNEMOSYNE_EMBEDDING_MODEL    Default: nomic-embed-text  (768-dim)
  MNEMOSYNE_EMBEDDING_DIM      Default: 768

Mnemosyne SQLite:
  MNEMOSYNE_DATA_DIR        Directory containing mnemosyne.db.
                            Default: ~/.hermes/mnemosyne/data

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXTERNAL SERVICE DEPENDENCIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


Qdrant  http://localhost:6333
  Collections used:
    mnemosyne        198 pts  768d/Cosine  named-vector "dense"
    hermes_sessions  151 pts  768d/Cosine  named-vector "dense"
    hermes_memory     89 pts  768d/Cosine  named-vector "dense"
  Auth: api-key header from QDRANT_API_KEY
  Used by: memory_recall (semantic), session_search, memory_stats
  NOTE: search payload must include "vector": {"name": "dense", "vector": [...]}
        (named-vector format — plain vector array will 400)

Ollama  http://localhost:11434/v1
  Model:    nomic-embed-text  (768-dim, Cosine)
  Endpoint: POST /v1/embeddings  {"model": "nomic-embed-text", "input": "<text>"}
  Response: data.data[0].embedding  OR  data.embedding  (both shapes handled)
  Timeout:  10s — embedding failures degrade gracefully (FTS results still returned)
  Used by: memory_recall (semantic leg), session_search

Mnemosyne SQLite  ~/.hermes/mnemosyne/data/mnemosyne.db
  Tables read:    fts_working (FTS5 full-text), fts_episodes (FTS5 external content),
                  memories, episodic_memory (rowid join via fts_episodes)
  Tables written: memories  (memory_remember skill)
  Used by: memory_recall, memory_remember, memory_stats

Mnemosyne Dashboard  http://127.0.0.1:8765  (optional — local only)
  Endpoint: POST /api/sleep  {"dry_run": bool}
  Used by:  memory_sleep skill
  Failure:  returns status="deferred" — non-fatal, server continues running

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ENDPOINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  GET  /.well-known/agent.json      Agent card (RFC-002) — no auth required
  GET  /.well-known/agent-card.json Agent card alias — no auth
  GET  /health                       Liveness + config check — no auth required
  POST /a2a                          JSON-RPC 2.0 dispatch — Bearer required
  GET  /a2a/tasks/{task_id}          Task status — Bearer required

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SKILLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  memory_recall    FTS5 (fts_working + fts_episodes) + optional Qdrant semantic
                   Input:  {query: str, top_k?: int=5, semantic?: bool=true}
                   Output: {memories: [...], total: int, query: str}

  memory_remember  Write a memory to SQLite memories table.
                   Input:  {content: str, source?: str, importance?: float=0.5,
                            bank?: str="default"}
                   Output: {id: str, status: "stored", bank: str, importance: float}
                   Note:   metadata_json records sender + HERMES_AGENT_ID for
                           cross-agent provenance (per PR #1 pattern).

  memory_stats     Point counts across all monitored tables and Qdrant collections.
                   Input:  {}
                   Output: {sqlite: {...}, qdrant: {...}, db_path: str}

  session_search   Semantic search over hermes_sessions Qdrant collection.
                   Input:  {query: str, top_k?: int=5, agent_id?: str}
                   Output: {sessions: [...], total: int, query: str}

  memory_sleep     Trigger Mnemosyne consolidation (working → episodic).
                   Input:  {dry_run?: bool=false}
                   Output: {status: "consolidated"|"deferred", ...}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JSON-RPC CALL SHAPE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  POST /a2a
  Authorization: Bearer <HERMES_A2A_TOKEN>
  X-TOTP: <6-digit code>   (only if HERMES_A2A_TOTP_SEED is set)

  {
    "jsonrpc": "2.0",
    "id": "<caller-uuid>",
    "method": "tasks/send",
    "params": {
      "skill_id": "memory_recall",
      "message":  "DAMA ant colony",
      "input":    {"query": "DAMA ant colony", "top_k": 5},
      "sender":   "hermes-agent"
    }
  }

  Response:
  {
    "jsonrpc": "2.0",
    "id": "<caller-uuid>",
    "result": {
      "task_id": "<uuid>",
      "status":  "completed",
      "output":  { <skill-specific output> }
    }
  }
"""

import os, uuid, json, sqlite3, logging, datetime
from typing import Optional, Any

# ── load .env before anything else ─────────────────────────────────────────────
# Override with HERMES_ENV_FILE env var. Default searches ~/.hermes/.env then
# the legacy per-profile path for backward compatibility.
_ENV_FILE = os.path.expanduser(
    os.environ.get('HERMES_ENV_FILE', '~/.hermes/.env')
)
if os.path.exists(_ENV_FILE):
    for _line in open(_ENV_FILE):
        _line = _line.strip()
        if _line and not _line.startswith('#') and '=' in _line:
            _k, _v = _line.split('=', 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import pyotp
import aiohttp
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

# ── config ──────────────────────────────────────────────────────────────────────
A2A_HOST  = os.environ.get('HERMES_A2A_HOST', '0.0.0.0')
A2A_PORT  = int(os.environ.get('HERMES_A2A_PORT', '8201'))
A2A_TOKEN = os.environ.get('HERMES_A2A_TOKEN', '')
if not A2A_TOKEN:
    print('ERROR: HERMES_A2A_TOKEN is not set. Generate one with: '
          'python3 -c "import secrets;print(secrets.token_hex(32))"', flush=True)
    sys.exit(1)
TOTP_SEED = os.environ.get('HERMES_A2A_TOTP_SEED', '')
AGENT_ID  = os.environ.get('HERMES_AGENT_ID', 'hermes-agent')
AGENT_URL = os.environ.get('HERMES_A2A_URL', 'http://127.0.0.1:8201')

# Mnemosyne SQLite — path built from MNEMOSYNE_DATA_DIR (set in .env)
_mnem_data_dir = os.path.expanduser(os.environ.get('MNEMOSYNE_DATA_DIR', '~/.hermes/mnemosyne/data'))
MNEMOSYNE_DB   = os.path.join(_mnem_data_dir, 'mnemosyne.db')

# Qdrant

QDRANT_URL = os.environ.get('QDRANT_URL')
QDRANT_KEY = os.environ.get('QDRANT_API_KEY', '')

# Ollama embedding (same config as hooks/pre_llm_grounding.py + session_end_sync.py)
OLLAMA_BASE           = os.environ.get('MNEMOSYNE_EMBEDDING_API_URL')
EMBED_MODEL           = os.environ.get('MNEMOSYNE_EMBEDDING_MODEL',   'nomic-embed-text')
EMBED_DIM             = int(os.environ.get('MNEMOSYNE_EMBEDDING_DIM', '768'))
_EMBED_API_KEY        = os.environ.get('EMBED_API_KEY', '')
_EMBED_API_KEY_HEADER = os.environ.get('EMBED_API_KEY_HEADER', 'Authorization')

# Core collections always available; extra collections are project-specific and opt-in.
_CORE_RAG_COLLECTIONS = ['mnemosyne', 'hermes_sessions', 'hermes_memory']
_EXTRA_RAG_COLLECTIONS = [
    c.strip() for c in os.environ.get('EXTRA_RAG_COLLECTIONS', '').split(',')
    if c.strip()
]

# Optional named collection for domain-specific skills. Skill returns "not configured"
# if the env var is unset, so the server works without the backing collection.
_DAMA_TELEMETRY_COLLECTION = os.environ.get('DAMA_TELEMETRY_COLLECTION', '')

_LOG_AGENT_ID = os.environ.get('HERMES_AGENT_ID', 'hermes-agent')
logging.basicConfig(level=logging.INFO,
                    format=f'%(asctime)s [{_LOG_AGENT_ID}] %(message)s')
log = logging.getLogger(__name__)

# ── agent card (RFC-002 schema) ─────────────────────────────────────────────────
AGENT_CARD = {
    'name': _LOG_AGENT_ID,
    'description': (
        'Persistent memory and knowledge node for the Hermes agent mesh. '
        'FTS + semantic search over session history, episodic memory, and working memory. '
        'Write new memories with cross-agent author tagging. '
        'Backend: Mnemosyne SQLite + Qdrant hermes_sessions/mnemosyne/hermes_memory collections.'
    ),
    'url': AGENT_URL,
    'protocol_version': '0.3.0',
    'agent_id': AGENT_ID,
    'skills': [
        {
            'id': 'memory_recall',
            'name': 'Memory Recall',
            'description': (
                'Search memories via SQLite FTS5 + Qdrant semantic search. '
                'Input: {query: str, top_k?: int=5, bank?: str, semantic?: bool=true}'
            )
        },
        {
            'id': 'memory_remember',
            'name': 'Memory Remember',
            'description': (
                'Store a new memory tagged with caller agent_id. '
                'Input: {content: str, source?: str, importance?: float=0.5, bank?: str="default"}'
            )
        },
        {
            'id': 'memory_stats',
            'name': 'Memory Stats',
            'description': (
                'SQLite row counts per table + Qdrant collection sizes. '
                'Input: {} (no parameters required)'
            )
        },
        {
            'id': 'session_search',
            'name': 'Session Search',
            'description': (
                'Semantic search over hermes session history in Qdrant. '
                'Input: {query: str, top_k?: int=5, agent_id?: str}'
            )
        },
        {
            'id': 'memory_sleep',
            'name': 'Memory Sleep / Consolidation',
            'description': (
                'Trigger Mnemosyne sleep consolidation cycle via dashboard API. '
                'Input: {dry_run?: bool=false}'
            )
        },
        {
            'id': 'rag_search',
            'name': 'Shared RAG Search',
            'description': (
                'Fan-out semantic search across Qdrant collections without requiring '
                'direct Qdrant credentials. Core: hermes_memory, hermes_sessions, mnemosyne. '
                'Additional collections: set EXTRA_RAG_COLLECTIONS env var (comma-separated). '
                'Input: {query: str, top_k?: int=5, collections?: [str]}'
            )
        },
        {
            'id': 'context_broadcast',
            'name': 'Context Broadcast',
            'description': (
                'Store a memory locally AND push it to all peer A2A endpoints (PEER_A2A_URLS). '
                'Used by the context bridge cron to propagate discoveries across the mesh. '
                'Input: {content: str, source?: str, importance?: float=0.5, bank?: str}'
            )
        },
        {
            'id': 'mnemosyne_triple_add',
            'name': 'Triple Add',
            'description': (
                'Store a knowledge triple (subject, predicate, object) in the SQLite triples table. '
                'Input: {subject: str, predicate: str, object: str, valid_from?: str, '
                'valid_until?: str, source?: str, confidence?: float=1.0, bank?: str}'
            )
        },
        {
            'id': 'gpu_inference',
            'name': 'GPU Inference',
            'description': (
                'Run a prompt through local Ollama. '
                'Input: {prompt: str, model?: str="llama3.1:8b", max_tokens?: int=512, system?: str}'
            )
        },
        {
            'id': 'docker_status',
            'name': 'Docker / k3s Status',
            'description': (
                'List running Docker containers and k3s pods. '
                'Input: {namespace?: str="all", filter?: str}'
            )
        },
        {
            'id': 'ua_search',
            'name': 'Codebase Semantic Search',
            'description': (
                'Semantic search over understand-anything knowledge graphs in Qdrant. '
                'Input: {query: str, repo?: str, type?: str, layer?: str, limit?: int=10}'
            )
        },
        {
            'id': 'dama_telemetry',
            'name': 'DAMA Telemetry Query',
            'description': (
                'Query domain telemetry from the configured Qdrant collection. '
                'Requires DAMA_TELEMETRY_COLLECTION to be set. '
                'Input: {query?: str, device?: str, limit?: int=5}'
            )
        },
        {
            'id': 'mnemosyne_triple_query',
            'name': 'Triple Query',
            'description': (
                'Query the knowledge graph triples table by subject, predicate, or object. '
                'Input: {subject?: str, predicate?: str, object?: str, limit?: int=20, bank?: str}'
            )
        },
    ],
    'capabilities': {
        'streaming': False,
        'push_notifications': False
    },
    'authentication': {
        'schemes': ['bearer'],
        'totp_enabled': bool(TOTP_SEED),
        'totp_header': 'X-TOTP'
    }
}

# ── in-memory task store (Phase 1 — no persistence) ────────────────────────────
# Bounded at 1000 entries; oldest evicted first to prevent memory growth.
_TASK_CAP = 1000
_tasks: dict[str, dict] = {}


def _store_task(task_id: str, task: dict) -> None:
    if len(_tasks) >= _TASK_CAP:
        oldest = next(iter(_tasks))
        del _tasks[oldest]
    _tasks[task_id] = task

# ── FastAPI app + auth ──────────────────────────────────────────────────────────
app = FastAPI(title=f'{_LOG_AGENT_ID} A2A', version='0.1.0')
_bearer_scheme = HTTPBearer(auto_error=False)


def _verify_bearer(creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme)):
    if not creds or creds.credentials != A2A_TOKEN:
        raise HTTPException(status_code=401, detail='Unauthorized — invalid or missing bearer token')


def _verify_totp(x_totp: Optional[str] = Header(default=None)):
    if TOTP_SEED:
        if x_totp is None:
            raise HTTPException(status_code=401, detail='X-TOTP header required (TOTP is enabled)')
        if not pyotp.TOTP(TOTP_SEED).verify(x_totp, valid_window=1):
            raise HTTPException(status_code=401, detail='Invalid TOTP code')


# ── helpers ─────────────────────────────────────────────────────────────────────
def _db() -> sqlite3.Connection:
    """Open Mnemosyne SQLite with row_factory."""
    conn = sqlite3.connect(MNEMOSYNE_DB, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _embed_auth_headers() -> dict:
    h = {'Content-Type': 'application/json'}
    if _EMBED_API_KEY:
        if _EMBED_API_KEY_HEADER.lower() == 'authorization':
            h['Authorization'] = f'Bearer {_EMBED_API_KEY}'
        else:
            h[_EMBED_API_KEY_HEADER] = _EMBED_API_KEY
    return h


async def _embed(text: str) -> Optional[list]:
    """Embed via OpenAI-compat /v1/embeddings. Works with Ollama and cloud providers."""
    url = OLLAMA_BASE.rstrip('/') + '/embeddings'
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as sess:
            async with sess.post(url, json={'model': EMBED_MODEL, 'input': text[:2000]},
                                 headers=_embed_auth_headers()) as r:
                if r.status == 200:
                    data = await r.json()
                    vec = (data.get('data') or [{}])[0].get('embedding') or data.get('embedding')
                    if vec and len(vec) == EMBED_DIM:
                        return vec
                    log.warning(f'embed: unexpected shape: {list(data.keys())}')
    except Exception as e:
        log.warning(f'embed failed: {e}')
    return None


async def _qdrant_search(
    collection: str,
    vector: list,
    top_k: int = 5,
    qdrant_filter: Optional[dict] = None
) -> list:
    """Named-vector semantic search in Qdrant. Uses 'dense' vector name (matches sync scripts)."""
    url = f'{QDRANT_URL}/collections/{collection}/points/search'
    body: dict = {
        'vector': {'name': 'dense', 'vector': vector},
        'limit': top_k,
        'with_payload': True
    }
    if qdrant_filter:
        body['filter'] = qdrant_filter
    headers = {'Content-Type': 'application/json', 'api-key': QDRANT_KEY}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as sess:
            async with sess.post(url, json=body, headers=headers) as r:
                if r.status == 200:
                    return (await r.json()).get('result', [])
                log.warning(f'qdrant search {collection}: HTTP {r.status}')
    except Exception as e:
        log.warning(f'qdrant search {collection}: {e}')
    return []


# ── skill: memory_recall ────────────────────────────────────────────────────────
async def skill_memory_recall(task: dict) -> dict:
    inp      = task.get('input', {})
    query    = (inp.get('query') or task.get('message', '')).strip()
    top_k    = int(inp.get('top_k', 5))
    do_sem   = inp.get('semantic', True)

    if not query:
        return {'error': 'query is required'}

    results = []

    # 1. FTS on fts_working (stores id + content directly — not external content)
    try:
        with _db() as conn:
            rows = conn.execute(
                "SELECT id, content FROM fts_working WHERE fts_working MATCH ? LIMIT ?",
                (query, top_k)
            ).fetchall()
            for r in rows:
                results.append({
                    'id': r['id'], 'content': r['content'],
                    'tier': 'working', 'source': 'fts_working', 'score': 0.8,
                    'importance': 0.5, 'created_at': ''
                })
    except Exception as e:
        log.warning(f'fts_working search: {e}')

    # 2. FTS on fts_episodes (external content table — join via rowid)
    try:
        with _db() as conn:
            rows = conn.execute(
                "SELECT em.id, em.content, em.importance, em.created_at "
                "FROM episodic_memory em "
                "WHERE em.rowid IN (SELECT rowid FROM fts_episodes WHERE fts_episodes MATCH ?) "
                "LIMIT ?",
                (query, top_k)
            ).fetchall()
            for r in rows:
                results.append({
                    'id': r['id'], 'content': r['content'],
                    'importance': float(r['importance'] or 0.5),
                    'created_at': r['created_at'] or '',
                    'tier': 'episodic', 'source': 'fts_episodic', 'score': 0.75
                })
    except Exception as e:
        log.warning(f'fts_episodes search: {e}')

    # 3. LIKE fallback on plain memories table (no FTS index on this table)
    try:
        with _db() as conn:
            rows = conn.execute(
                "SELECT id, content, importance, created_at FROM memories "
                "WHERE content LIKE ? LIMIT ?",
                (f'%{query}%', top_k)
            ).fetchall()
            seen = {r['id'] for r in results}
            for r in rows:
                if r['id'] not in seen:
                    results.append({
                        'id': r['id'], 'content': r['content'],
                        'importance': float(r['importance'] or 0.5),
                        'created_at': r['created_at'] or '',
                        'tier': 'memory', 'source': 'memories_like', 'score': 0.6
                    })
    except Exception as e:
        log.warning(f'memories LIKE search: {e}')

    # 4. Semantic via Qdrant mnemosyne collection
    if do_sem:
        vec = await _embed(query)
        if vec:
            hits = await _qdrant_search('mnemosyne', vec, top_k=top_k)
            seen_ids = {r['id'] for r in results}
            for h in hits:
                pl = h.get('payload', {})
                mid = pl.get('memory_id', str(h.get('id', '')))
                if mid not in seen_ids:
                    results.append({
                        'id': mid, 'content': pl.get('content', ''),
                        'importance': float(pl.get('importance', 0.5)),
                        'created_at': pl.get('created_at', ''),
                        'tier': 'episodic', 'source': 'qdrant_mnemosyne',
                        'score': round(float(h.get('score', 0)), 4)
                    })

    # Sort by score desc, deduplicate, cap at top_k
    results.sort(key=lambda x: x.get('score', 0), reverse=True)
    seen, deduped = set(), []
    for r in results:
        key = r.get('id') or r['content'][:40]
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    return {'memories': deduped[:top_k], 'total': len(deduped), 'query': query}


# ── skill: memory_remember ──────────────────────────────────────────────────────
async def skill_memory_remember(task: dict) -> dict:
    inp        = task.get('input', {})
    content    = (inp.get('content') or task.get('message', '')).strip()
    source     = inp.get('source', 'a2a')
    importance = float(inp.get('importance', 0.5))
    bank       = inp.get('bank', 'default')
    sender     = task.get('sender', AGENT_ID)

    if not content:
        return {'error': 'content is required'}

    mem_id = str(uuid.uuid4())
    now    = datetime.datetime.utcnow().isoformat()
    meta   = json.dumps({'author_id': sender, 'via': 'a2a', 'stored_by': AGENT_ID})

    try:
        with _db() as conn:
            conn.execute(
                'INSERT INTO memories '
                '(id, content, source, timestamp, session_id, importance, metadata_json, created_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (mem_id, content, source, now, bank, importance, meta, now)
            )
            conn.commit()
    except Exception as e:
        log.error(f'memory_remember write failed: {e}')
        return {'error': f'db write failed: {e}'}

    log.info(f'Stored memory {mem_id} from sender={sender} importance={importance}')
    return {'id': mem_id, 'status': 'stored', 'bank': bank, 'importance': importance}


# ── skill: memory_stats ─────────────────────────────────────────────────────────
async def skill_memory_stats(task: dict) -> dict:
    sqlite_stats: dict = {}
    tables = ['memories', 'working_memory', 'episodic_memory', 'scratchpad',
              'facts', 'consolidated_facts', 'gists', 'triples']
    try:
        with _db() as conn:
            for tbl in tables:
                try:
                    n = conn.execute(f'SELECT COUNT(*) FROM {tbl}').fetchone()[0]
                    sqlite_stats[tbl] = n
                except Exception:
                    sqlite_stats[tbl] = -1
    except Exception as e:
        sqlite_stats['error'] = str(e)

    # Qdrant collection point counts
    qdrant_stats: dict = {}
    headers = {'Content-Type': 'application/json', 'api-key': QDRANT_KEY}
    collections = _CORE_RAG_COLLECTIONS + _EXTRA_RAG_COLLECTIONS
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as sess:
            for col in collections:
                try:
                    async with sess.get(f'{QDRANT_URL}/collections/{col}',
                                        headers=headers) as r:
                        if r.status == 200:
                            data = await r.json()
                            qdrant_stats[col] = data.get('result', {}).get('points_count', '?')
                        else:
                            qdrant_stats[col] = f'HTTP {r.status}'
                except Exception as e:
                    qdrant_stats[col] = str(e)
    except Exception as e:
        qdrant_stats['error'] = str(e)

    return {
        'sqlite': sqlite_stats,
        'qdrant': qdrant_stats,
        'agent_id': AGENT_ID,
    }


# ── skill: session_search ───────────────────────────────────────────────────────
async def skill_session_search(task: dict) -> dict:
    inp          = task.get('input', {})
    query        = (inp.get('query') or task.get('message', '')).strip()
    top_k        = int(inp.get('top_k', 5))
    agent_filter = inp.get('agent_id')

    if not query:
        return {'error': 'query is required'}

    vec = await _embed(query)
    if not vec:
        return {
            'error': 'embedding failed — Ollama may be unreachable',
            'sessions': [],
        }

    qdrant_filter = None
    if agent_filter:
        qdrant_filter = {'must': [{'key': 'agent_id', 'match': {'value': agent_filter}}]}

    hits = await _qdrant_search('hermes_sessions', vec, top_k=top_k,
                                 qdrant_filter=qdrant_filter)
    sessions = []
    for h in hits:
        pl = h.get('payload', {})
        sessions.append({
            'session_id':      pl.get('session_id', ''),
            'title':           pl.get('title', ''),
            'content_preview': (pl.get('content_preview') or '')[:300],
            'agent_id':        pl.get('agent_id', ''),
            'profile':         pl.get('profile', ''),
            'msg_count':       pl.get('msg_count', 0),
            'last_synced':     pl.get('last_synced', ''),
            'score':           round(float(h.get('score', 0)), 4),
        })

    return {'sessions': sessions, 'total': len(sessions), 'query': query}


# ── skill: memory_sleep ─────────────────────────────────────────────────────────
async def skill_memory_sleep(task: dict) -> dict:
    inp     = task.get('input', {})
    dry_run = bool(inp.get('dry_run', False))

    # Try Mnemosyne dashboard HTTP API (port 8765)
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as sess:
            async with sess.post(
                'http://127.0.0.1:8765/api/sleep',
                json={'dry_run': dry_run},
                headers={'Content-Type': 'application/json'}
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    return {'status': 'consolidated', 'result': data, 'via': 'dashboard_api'}
                log.warning(f'dashboard /api/sleep returned HTTP {r.status}')
    except Exception as e:
        log.warning(f'dashboard sleep call failed: {e}')

    return {
        'status': 'deferred',
        'message': (
            'Mnemosyne dashboard not running or /api/sleep not available. '
            'Start the dashboard first (mnemosyne_dashboard_start tool) then retry.'
        ),
        'dry_run': dry_run
    }




# ── skill: rag_search ────────────────────────────────────────────────────────────
async def skill_rag_search(task: dict) -> dict:
    """
    Fan-out semantic search across ALL 768-dim Qdrant collections.
    Lets any mesh agent query the full shared corpus
    without needing Qdrant credentials or knowing collection names.

    Input:  {query: str, top_k?: int=5, collections?: [str]}
    Output: {results: [...merged ranked hits], query: str, collections_searched: [str]}
    """
    inp         = task.get('input', {})
    query       = (inp.get('query') or task.get('message', '')).strip()
    top_k       = int(inp.get('top_k', 5))
    req_cols    = inp.get('collections')

    if not query:
        return {'error': 'query is required'}

    # Default collection list — core plus any extra configured via env var
    ALL_COLLECTIONS = _CORE_RAG_COLLECTIONS + _EXTRA_RAG_COLLECTIONS
    collections = req_cols if (req_cols and isinstance(req_cols, list)) else ALL_COLLECTIONS

    vec = await _embed(query)
    if not vec:
        return {'error': 'embedding failed — Ollama may be unreachable', 'results': []}

    import asyncio

    async def _search_one(col: str) -> list:
        hits = await _qdrant_search(col, vec, top_k=top_k)
        results = []
        for h in hits:
            pl = h.get('payload', {})
            content = (
                pl.get('content') or pl.get('content_preview') or
                pl.get('text') or pl.get('chunk_text') or ''
            )[:500]
            results.append({
                'collection':  col,
                'score':       round(float(h.get('score', 0)), 4),
                'content':     content,
                'id':          str(h.get('id', '')),
                'payload':     {k: v for k, v in pl.items()
                                if k not in ('content', 'chunk_text', 'text', 'content_preview')
                                and not isinstance(v, (list, dict))},
            })
        return results

    all_hits = []
    tasks_q = [_search_one(c) for c in collections]
    results_per_col = await asyncio.gather(*tasks_q, return_exceptions=True)
    for col, res in zip(collections, results_per_col):
        if isinstance(res, Exception):
            log.warning(f'rag_search {col}: {res}')
        else:
            all_hits.extend(res)

    all_hits.sort(key=lambda x: x['score'], reverse=True)

    # Deduplicate by (collection, id)
    seen, deduped = set(), []
    for h in all_hits:
        key = f"{h['collection']}:{h['id']}"
        if key not in seen:
            seen.add(key)
            deduped.append(h)

    return {
        'results':              deduped[:top_k * 2],   # return more than top_k since multi-collection
        'query':                query,
        'collections_searched': collections,
        'total_hits':           len(deduped),
    }


# ── skill: context_broadcast ─────────────────────────────────────────────────────
async def skill_context_broadcast(task: dict) -> dict:
    """
    Push a memory to all configured peer A2A endpoints (PEER_A2A_URLS env var).
    Used by the a2a_context_bridge cron to propagate discoveries to the mesh.

    Input:  {content: str, source?: str, importance?: float=0.5, bank?: str="default"}
    Output: {broadcast: [{peer, status, error?}], stored_locally: bool}

    PEER_A2A_URLS:  comma-separated list, e.g.
        http://peer-a:8201/a2a,http://peer-b:8201/a2a
    PEER_A2A_TOKEN: shared Bearer token for all peers (or set per-peer in PEER_A2A_TOKENS_JSON)
    PEER_A2A_TOKENS_JSON: JSON dict mapping base_url -> token (overrides PEER_A2A_TOKEN per peer)
    """
    inp        = task.get('input', {})
    content    = (inp.get('content') or task.get('message', '')).strip()
    source     = inp.get('source', 'context_broadcast')
    importance = float(inp.get('importance', 0.5))
    bank       = inp.get('bank', 'default')
    sender     = task.get('sender', AGENT_ID)

    if not content:
        return {'error': 'content is required'}

    # 1. Store locally first
    local_result = await skill_memory_remember({
        'input': {'content': content, 'source': source,
                  'importance': importance, 'bank': bank},
        'sender': sender
    })
    stored_locally = 'error' not in local_result

    # 2. Fan-out to peers
    peer_urls_raw = os.environ.get('PEER_A2A_URLS', '')
    default_token = os.environ.get('PEER_A2A_TOKEN', '')
    try:
        token_map = json.loads(os.environ.get('PEER_A2A_TOKENS_JSON', '{}'))
    except Exception:
        token_map = {}

    peer_urls = [u.strip() for u in peer_urls_raw.split(',') if u.strip()]
    broadcast_results = []

    import uuid as _uuid

    async def _push_to_peer(peer_url: str):
        # Derive base URL for token lookup (strip /a2a suffix)
        base = peer_url.rstrip('/a2a').rstrip('/')
        token = token_map.get(peer_url) or token_map.get(base) or default_token
        if not token:
            return {'peer': peer_url, 'status': 'skipped', 'error': 'no token configured'}

        payload = {
            'jsonrpc': '2.0',
            'id': str(_uuid.uuid4()),
            'method': 'tasks/send',
            'params': {
                'skill_id': 'memory_remember',
                'message':  content,
                'input':    {'content': content, 'source': f'broadcast:{AGENT_ID}',
                             'importance': importance, 'bank': bank},
                'sender':   AGENT_ID,
            }
        }
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type':  'application/json',
        }
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as sess:
                async with sess.post(peer_url, json=payload, headers=headers) as r:
                    if r.status == 200:
                        data = await r.json()
                        return {'peer': peer_url, 'status': 'ok',
                                'output': data.get('result', {}).get('output', {})}
                    else:
                        body = await r.text()
                        return {'peer': peer_url, 'status': f'http_{r.status}',
                                'error': body[:200]}
        except Exception as e:
            return {'peer': peer_url, 'status': 'error', 'error': str(e)}

    if peer_urls:
        import asyncio as _asyncio
        results = await _asyncio.gather(*[_push_to_peer(u) for u in peer_urls])
        broadcast_results = list(results)
        log.info(f'context_broadcast: pushed to {len(peer_urls)} peers — '
                 f'{sum(1 for r in broadcast_results if r.get("status") == "ok")} ok')
    else:
        broadcast_results = [{'peer': 'none', 'status': 'skipped',
                               'error': 'PEER_A2A_URLS not set'}]

    return {
        'stored_locally': stored_locally,
        'broadcast':      broadcast_results,
        'content_len':    len(content),
        'peers_count':    len(peer_urls),
    }

# ── skill: mnemosyne_triple_add ──────────────────────────────────────────────────
async def skill_mnemosyne_triple_add(task: dict) -> dict:
    """
    Store a knowledge triple (subject, predicate, object) in the SQLite triples table.
    Input: {subject: str, predicate: str, object: str, valid_from?: str,
            valid_until?: str, source?: str, confidence?: float=1.0}
    """
    inp        = task.get('input', {})
    subject    = (inp.get('subject') or '').strip()
    predicate  = (inp.get('predicate') or '').strip()
    obj        = (inp.get('object') or '').strip()
    valid_from = inp.get('valid_from') or datetime.datetime.utcnow().strftime('%Y-%m-%d')
    valid_until = inp.get('valid_until')
    source     = inp.get('source', 'a2a')
    confidence = float(inp.get('confidence', 1.0))

    if not subject or not predicate or not obj:
        return {'error': 'subject, predicate, and object are required'}

    try:
        with _db() as conn:
            conn.execute(
                'INSERT INTO triples (subject, predicate, object, valid_from, valid_until, source, confidence) '
                'VALUES (?, ?, ?, ?, ?, ?, ?)',
                (subject, predicate, obj, valid_from, valid_until, source, confidence)
            )
            conn.commit()
            triple_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
    except Exception as e:
        log.error(f'triple_add write failed: {e}')
        return {'error': f'db write failed: {e}'}

    log.info(f'Stored triple [{triple_id}] {subject} -{predicate}-> {obj}')
    return {
        'status': 'stored',
        'triple_id': triple_id,
        'subject': subject,
        'predicate': predicate,
        'object': obj,
        'valid_from': valid_from,
    }


# ── skill: mnemosyne_triple_query ────────────────────────────────────────────────
async def skill_mnemosyne_triple_query(task: dict) -> dict:
    """
    Query the knowledge graph triples table.
    Input: {subject?: str, predicate?: str, object?: str, limit?: int=20}
    """
    inp       = task.get('input', {})
    subject   = (inp.get('subject') or '').strip() or None
    predicate = (inp.get('predicate') or '').strip() or None
    obj       = (inp.get('object') or '').strip() or None
    limit     = int(inp.get('limit', 20))

    conditions = []
    params: list = []
    if subject:
        conditions.append('subject = ?')
        params.append(subject)
    if predicate:
        conditions.append('predicate = ?')
        params.append(predicate)
    if obj:
        conditions.append('object = ?')
        params.append(obj)

    where_clause = ('WHERE ' + ' AND '.join(conditions)) if conditions else ''
    sql = f'SELECT id, subject, predicate, object, valid_from, valid_until, source, confidence, created_at FROM triples {where_clause} ORDER BY created_at DESC LIMIT ?'
    params.append(limit)

    try:
        with _db() as conn:
            rows = conn.execute(sql, params).fetchall()
            triples = [
                {
                    'id': r['id'],
                    'subject': r['subject'],
                    'predicate': r['predicate'],
                    'object': r['object'],
                    'valid_from': r['valid_from'],
                    'valid_until': r['valid_until'],
                    'source': r['source'],
                    'confidence': float(r['confidence'] or 1.0),
                    'created_at': r['created_at'],
                }
                for r in rows
            ]
    except Exception as e:
        log.error(f'triple_query failed: {e}')
        return {'error': f'db query failed: {e}', 'triples': []}

    return {'triples': triples, 'total': len(triples)}




# ── real skill handlers ──────────────────────────────────────────────────────────

async def skill_gpu_inference(task: dict) -> dict:
    """
    Run a prompt through the local Ollama LLM stack.
    Input: {prompt: str, model?: str='llama3.1:8b', max_tokens?: int=512, system?: str}
    """
    inp    = task.get('input', {})
    prompt = inp.get('prompt') or task.get('message', '')
    model  = inp.get('model', 'llama3.1:8b')
    max_t  = int(inp.get('max_tokens', 512))
    system = inp.get('system', '')

    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': prompt})

    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                f"{OLLAMA_BASE.rstrip('/v1')}/v1/chat/completions",
                json={'model': model, 'messages': messages, 'max_tokens': max_t},
                timeout=aiohttp.ClientTimeout(total=120)
            ) as r:
                d = await r.json()
                content = d['choices'][0]['message']['content']
                return {'response': content, 'model': model, 'status': 'ok'}
    except Exception as e:
        return {'error': str(e), 'status': 'error'}


async def skill_docker_status(task: dict) -> dict:
    """
    List running Docker containers and k3s pods.
    Input: {namespace?: str='all', filter?: str}
    """
    inp = task.get('input', {})
    ns  = inp.get('namespace', 'all')
    flt = inp.get('filter', '')

    import subprocess
    results: dict = {}

    # Docker
    try:
        r = subprocess.run(
            ['docker', 'ps', '--format', '{{.Names}}\t{{.Status}}\t{{.Image}}'],
            capture_output=True, text=True, timeout=10
        )
        containers = []
        for line in r.stdout.strip().splitlines():
            parts = line.split('\t')
            if len(parts) >= 3:
                entry = {'name': parts[0], 'status': parts[1], 'image': parts[2]}
                if not flt or flt.lower() in parts[0].lower():
                    containers.append(entry)
        results['docker'] = containers
    except Exception as e:
        results['docker_error'] = str(e)

    # k3s pods
    try:
        ns_flag = ['--all-namespaces'] if ns == 'all' else ['-n', ns]
        r = subprocess.run(
            ['kubectl', 'get', 'pods'] + ns_flag + ['--no-headers',
             '-o', 'custom-columns=NS:.metadata.namespace,NAME:.metadata.name,STATUS:.status.phase,READY:.status.containerStatuses[0].ready'],
            capture_output=True, text=True, timeout=15
        )
        pods = []
        for line in r.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                entry = {'namespace': parts[0], 'name': parts[1], 'status': parts[2], 'ready': parts[3] if len(parts) > 3 else '?'}
                if not flt or flt.lower() in parts[1].lower():
                    pods.append(entry)
        results['k3s_pods'] = pods
    except Exception as e:
        results['k3s_error'] = str(e)

    return results


async def skill_ua_search(task: dict) -> dict:
    """
    Semantic search over understand-anything knowledge graphs in Qdrant.
    Input: {query: str, repo?: str, type?: str, layer?: str, limit?: int=10}
    """
    inp   = task.get('input', {})
    query = inp.get('query') or task.get('message', '')
    repo  = inp.get('repo')
    ntype = inp.get('type')
    layer = inp.get('layer')
    limit = int(inp.get('limit', 10))

    if not query:
        return {'error': 'query required'}

    import subprocess, sys
    ua_script = os.environ.get('UA_SEARCH_SCRIPT', '')
    if not ua_script or not os.path.exists(ua_script):
        return {'error': 'UA_SEARCH_SCRIPT env var not set or script not found', 'query': query}
    args = [sys.executable, ua_script, query, '--json', '-n', str(limit)]
    if repo:  args += ['--repo', repo]
    if ntype: args += ['--type', ntype]
    if layer: args += ['--layer', layer]

    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=30)
        results = json.loads(r.stdout) if r.stdout.strip() else []
        return {'results': results, 'count': len(results), 'query': query}
    except Exception as e:
        return {'error': str(e), 'query': query}


async def skill_dama_telemetry(task: dict) -> dict:
    """
    Query a telemetry Qdrant collection by semantic similarity.
    Requires DAMA_TELEMETRY_COLLECTION env var to be set.
    Input: {query?: str, device?: str, limit?: int=5}
    """
    if not _DAMA_TELEMETRY_COLLECTION:
        return {
            'error': 'DAMA_TELEMETRY_COLLECTION env var not configured',
            'hint': 'Set DAMA_TELEMETRY_COLLECTION=<your-collection-name> to enable this skill',
        }

    inp    = task.get('input', {})
    query  = inp.get('query', 'telemetry')
    device = inp.get('device')
    limit  = int(inp.get('limit', 5))

    must = []
    if device:
        must.append({'key': 'device_id', 'match': {'value': device}})

    payload = {'vector': None, 'limit': limit, 'with_payload': True}
    if must:
        payload['filter'] = {'must': must}

    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                f"{OLLAMA_BASE.rstrip('/v1')}/api/embed",
                json={'model': EMBED_MODEL, 'input': query},
                timeout=aiohttp.ClientTimeout(total=15)
            ) as r:
                d = await r.json()
                vec = d['embeddings'][0]

        payload['vector'] = vec
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                f"{QDRANT_URL}/collections/{_DAMA_TELEMETRY_COLLECTION}/points/search",
                headers={'api-key': QDRANT_KEY, 'Content-Type': 'application/json'},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as r:
                d = await r.json()
                hits = d.get('result', [])
                return {
                    'results': [{'score': h['score'], 'payload': h['payload']} for h in hits],
                    'count': len(hits),
                    'query': query,
                    'collection': _DAMA_TELEMETRY_COLLECTION,
                }
    except Exception as e:
        return {'error': str(e)}


# ── skill dispatcher ─────────────────────────────────────────────────────────────
async def skill_memory_prime(task: dict) -> dict:
    """
    SAR-style priming broadcast (plant systemic acquired resistance analog).

    Unlike context_broadcast (which ships content), this ships a defensive posture:
    a decaying skepticism boost for a topic cluster that lowers memcheck thresholds
    on all nodes without propagating the hallucinated content itself.

    Input:  {topic: str, skepticism_delta: float=0.2, ttl_seconds: int=3600,
             broadcast?: bool=true}
    Output: {primed_topics: [topic], ttl_seconds: int, broadcast: [{peer, status}]}

    The priming state is written to SAR_PRIMING_STATE_PATH
    (~/.hermes/sar-priming.json) and read by the grounding hook / memcheck
    to lower block thresholds for the primed topic window.
    """
    import time as _time
    inp              = task.get('input', {})
    topic            = str(inp.get('topic', '') or task.get('message', '')).strip()
    skepticism_delta = float(inp.get('skepticism_delta', 0.2))
    ttl_seconds      = int(inp.get('ttl_seconds', 3600))
    do_broadcast     = bool(inp.get('broadcast', True))

    if not topic:
        return {'error': 'topic is required'}

    state_path = os.environ.get(
        'SAR_PRIMING_STATE_PATH',
        os.path.expanduser('~/.hermes/sar-priming.json'),
    )

    # Load existing state, write updated priming entry.
    try:
        with open(state_path) as f:
            state: dict = json.load(f)
    except Exception:
        state = {}

    expires_at = int(_time.time()) + ttl_seconds
    state[topic] = {
        'skepticism_delta': skepticism_delta,
        'expires_at': expires_at,
        'source': task.get('sender', AGENT_ID),
    }
    # Prune expired entries while we have the file open.
    now_ts = int(_time.time())
    state  = {t: v for t, v in state.items() if v.get('expires_at', 0) > now_ts}

    try:
        tmp = state_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(state, f)
        os.replace(tmp, state_path)
    except Exception as e:
        return {'error': f'Failed to write priming state: {e}'}

    broadcast_results = []
    if do_broadcast:
        peer_urls_raw  = os.environ.get('PEER_A2A_URLS', '')
        default_token  = os.environ.get('PEER_A2A_TOKEN', '')
        try:
            token_map = json.loads(os.environ.get('PEER_A2A_TOKENS_JSON', '{}'))
        except Exception:
            token_map = {}
        peer_urls = [u.strip() for u in peer_urls_raw.split(',') if u.strip()]

        import uuid as _uuid
        import aiohttp

        async def _prime_peer(peer_url: str):
            base  = peer_url.rstrip('/a2a').rstrip('/')
            token = token_map.get(peer_url) or token_map.get(base) or default_token
            if not token:
                return {'peer': peer_url, 'status': 'skipped', 'error': 'no token'}
            payload = {
                'jsonrpc': '2.0', 'id': str(_uuid.uuid4()), 'method': 'tasks/send',
                'params': {
                    'skill_id': 'memory_prime',
                    'message': topic,
                    'input': {'topic': topic, 'skepticism_delta': skepticism_delta,
                              'ttl_seconds': ttl_seconds, 'broadcast': False},
                    'sender': AGENT_ID,
                },
            }
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        peer_url, json=payload,
                        headers={'Authorization': f'Bearer {token}'},
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        return {'peer': peer_url, 'status': resp.status}
            except Exception as e:
                return {'peer': peer_url, 'status': 'error', 'error': str(e)}

        broadcast_results = await asyncio.gather(
            *[_prime_peer(u) for u in peer_urls], return_exceptions=False
        )

    return {
        'primed_topics': [topic],
        'skepticism_delta': skepticism_delta,
        'ttl_seconds': ttl_seconds,
        'expires_at': expires_at,
        'broadcast': broadcast_results,
    }


_SKILL_MAP: dict[str, Any] = {
    'memory_recall':           skill_memory_recall,
    'memory_remember':         skill_memory_remember,
    'memory_stats':            skill_memory_stats,
    'session_search':          skill_session_search,
    'memory_sleep':            skill_memory_sleep,
    'rag_search':              skill_rag_search,
    'context_broadcast':       skill_context_broadcast,
    'memory_prime':            skill_memory_prime,
    'mnemosyne_triple_add':    skill_mnemosyne_triple_add,
    'mnemosyne_triple_query':  skill_mnemosyne_triple_query,
    'gpu_inference':           skill_gpu_inference,
    'docker_status':           skill_docker_status,
    'ua_search':               skill_ua_search,
    'dama_telemetry':          skill_dama_telemetry,
}

async def _dispatch(skill_id: str, task: dict) -> Any:
    handler = _SKILL_MAP.get(skill_id)
    if not handler:
        return {
            'error': f"Unknown skill '{skill_id}'.",
            'available': list(_SKILL_MAP.keys())
        }
    return await handler(task)


# ── routes ───────────────────────────────────────────────────────────────────────
@app.get('/.well-known/agent.json')
async def agent_card_rfc002():
    """Agent card — RFC-002 spec location."""
    return JSONResponse(AGENT_CARD)


@app.get('/.well-known/agent-card.json')
async def agent_card_legacy_alias():
    """Agent card — legacy alias."""
    return JSONResponse(AGENT_CARD)


@app.get('/health')
async def health():
    db_ok = os.path.exists(MNEMOSYNE_DB)
    return JSONResponse({
        'status': 'ok',
        'agent': AGENT_ID,
        'skills': list(_SKILL_MAP.keys()),
        'mnemosyne_db_found': db_ok,
        'qdrant_configured': bool(QDRANT_URL),
        'ollama_configured': bool(OLLAMA_BASE),
        'totp_enabled': bool(TOTP_SEED),
    })


@app.post('/a2a', dependencies=[Depends(_verify_bearer), Depends(_verify_totp)])
async def a2a_endpoint(request: Request):
    """Main JSON-RPC 2.0 dispatch."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({'jsonrpc': '2.0', 'id': None,
                             'error': {'code': -32700, 'message': 'Parse error'}})

    rpc_id = body.get('id', str(uuid.uuid4()))
    method = body.get('method', '')
    params = body.get('params', {})

    if method == 'tasks/send':
        return await _handle_task_send(rpc_id, params)
    elif method == 'tasks/get':
        return await _handle_task_get(rpc_id, params)
    elif method == 'tasks/list':
        return JSONResponse({
            'jsonrpc': '2.0', 'id': rpc_id,
            'result': {'tasks': list(_tasks.values())}
        })
    else:
        return JSONResponse({
            'jsonrpc': '2.0', 'id': rpc_id,
            'error': {'code': -32601, 'message': f"Method not found: '{method}'"}
        })


@app.get('/a2a/tasks/{task_id}', dependencies=[Depends(_verify_bearer)])
async def get_task(task_id: str):
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail='Task not found')
    return JSONResponse(_tasks[task_id])


async def _handle_task_send(rpc_id: str, params: dict) -> JSONResponse:
    task_id  = str(uuid.uuid4())
    skill_id = params.get('skill_id', '')
    message  = params.get('message', '')
    sender   = params.get('sender', 'unknown')

    task = {
        'id':         task_id,
        'skill_id':   skill_id,
        'message':    message,
        'input':      params.get('input', {}),
        'sender':     sender,
        'status':     'working',
        'created_at': datetime.datetime.utcnow().isoformat(),
        'result':     None
    }
    _store_task(task_id, task)
    log.info(f'Task [{task_id}] skill={skill_id} sender={sender}')

    try:
        result = await _dispatch(skill_id, task)
    except Exception as e:
        log.exception(f'Skill {skill_id} raised')
        result = {'error': str(e)}

    task['status'] = 'completed'
    task['result'] = result

    return JSONResponse({
        'jsonrpc': '2.0', 'id': rpc_id,
        'result': {'task_id': task_id, 'status': 'completed', 'output': result}
    })


async def _handle_task_get(rpc_id: str, params: dict) -> JSONResponse:
    task_id = params.get('task_id')
    if not task_id or task_id not in _tasks:
        return JSONResponse({
            'jsonrpc': '2.0', 'id': rpc_id,
            'error': {'code': -32602, 'message': 'Task not found'}
        })
    return JSONResponse({'jsonrpc': '2.0', 'id': rpc_id, 'result': _tasks[task_id]})


# ── entrypoint ───────────────────────────────────────────────────────────────────
def main() -> None:
    log.info(f'{_LOG_AGENT_ID} A2A v0.1.0  {A2A_HOST}:{A2A_PORT}')
    log.info(f'Agent card:    {AGENT_URL}/.well-known/agent.json')
    log.info(f'Mnemosyne DB:  {MNEMOSYNE_DB}  ({"found" if os.path.exists(MNEMOSYNE_DB) else "MISSING"})')
    log.info(f'Qdrant:        {QDRANT_URL}')
    log.info(f'Ollama embed:  {OLLAMA_BASE}  model={EMBED_MODEL}')
    log.info(f'TOTP:          {"enabled" if TOTP_SEED else "disabled (set HERMES_A2A_TOTP_SEED to enable)"}')
    log.info(f'Skills:        {", ".join(_SKILL_MAP.keys())}')
    uvicorn.run(app, host=A2A_HOST, port=A2A_PORT, log_level='info', timeout_graceful_shutdown=5)


if __name__ == '__main__':
    main()

