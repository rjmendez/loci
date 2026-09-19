#!/usr/bin/env python3
"""Loci A2A Server v0.1.0 — Mnemosyne memory over A2A JSON-RPC.

Runtime requirements
- Python 3.11 (venv: ~/.hermes/hermes-agent/venv/bin/python3)
- Pip packages from requirements.txt:
  fastapi==0.133.1 HTTP server/dependency injection/******
  uvicorn==0.41.0 ASGI runner
  starlette==1.3.1 fastapi dep for Request/JSONResponse
  pydantic==2.13.4 fastapi dep for validation
  aiohttp==3.14.3 async HTTP client for Qdrant + Ollama
  pyotp==2.9.0 TOTP (RFC 6238) for X-TOTP auth
- Stdlib: os, uuid, json, sqlite3, logging, datetime, typing, asyncio, sys

Env vars (loaded from ~/.hermes/.env at startup)
Required:
  LOCI_A2A_TOKEN ****** callers must supply.
    REQUIRED — server exits at startup if unset.
    Generate: python3 -c "import secrets;print(secrets.token_hex(32))"
Optional / tunable:
  LOCI_A2A_BOOTSTRAP_KEY pre-shared key for POST /bootstrap.
    Returns a 24h session token usable as bearer without TOTP.
    If unset, /bootstrap 501s.
    Generate: python3 -c "import secrets;print(secrets.token_hex(24))"
  LOCI_A2A_HOST bind address. Default: 0.0.0.0
  LOCI_A2A_PORT bind port. Default: 8201
  LOCI_A2A_URL public base URL injected into the agent card.
    Default: http://127.0.0.1:8201
  LOCI_A2A_TOTP_SEED base32 TOTP seed (RFC 6238).
    If set, callers must send X-TOTP. Default: '' (disabled)
  HERMES_AGENT_ID agent identity tag written into memory metadata.
    Default: 'hermes-agent'
  RERANK_HTTP_URL llama.cpp `llama-server --rerank --pooling rank` endpoint.
    Example: http://127.0.0.1:8082/rerank
    When set, rag_search reranks merged cross-collection hits with a cross-encoder
    instead of trusting raw cosine scores across collections. Adds no torch
    dependency here. Unset = cosine order. Fails open on error.
  RERANK_TIMEOUT_S rerank request timeout. Default: 10
  LOCI_ENV_FILE path to the .env file loaded at import time.
    Default: ~/.hermes/.env
  EXTRA_RAG_COLLECTIONS comma-separated extra Qdrant collections appended to the
    core three for rag_search / memory_stats. Default: ''
  EMBED_API_KEY API key for the embedding endpoint. Default: '' (no auth header)
  EMBED_API_KEY_HEADER header carrying EMBED_API_KEY.
    'Authorization' sends '******'; any other name sends the raw key.
    Default: Authorization
  LOCI_A2A_PRIVILEGED_SENDERS comma-separated sender IDs allowed to call
    DESTRUCTIVE_SKILLS (memory_remember, memory_sleep, context_broadcast,
    mnemosyne_triple_add). Default: '' so destructive skills are effectively disabled.
  PEER_A2A_URLS comma-separated peer A2A base URLs for fan-out skills
    (memory_broadcast, memory_prime). Default: ''
  PEER_A2A_TOKEN shared ****** for every peer. Default: ''
  PEER_A2A_TOKENS_JSON JSON dict base_url -> token; overrides PEER_A2A_TOKEN.
    Default: '{}'
  PEER_A2A_TOTP_SEED shared base32 TOTP seed for peers requiring X-TOTP.
    Default: ''
  PEER_A2A_TOTP_SEEDS_JSON JSON dict base_url -> seed; overrides
    PEER_A2A_TOTP_SEED. Default: '{}'
  SAR_PRIMING_STATE_PATH where memory_prime persists per-peer priming state.
    Default: ~/.hermes/sar-priming.json
  LOCI_A2A_IDEMPOTENCY_TTL_S replay-protection window for `_boundary.idempotency_key`
    reservations (`sender + skill + key`). Default: 3600
  UA_SEARCH_SCRIPT path to the external UA search helper script.
    Default: '' (skill returns "not configured")

Qdrant (shared with session_end_sync.py + state_db_qdrant_sync.py)
  QDRANT_URL unset disables Qdrant-backed legs; they fail open to [].
  QDRANT_API_KEY Qdrant API key; set in .env.

Ollama embedding (shared with sync scripts)
  MNEMOSYNE_EMBEDDING_API_URL default http://localhost:11434/v1
  MNEMOSYNE_EMBEDDING_MODEL default nomic-embed-text (768-dim)
  MNEMOSYNE_EMBEDDING_DIM default 768

Mnemosyne SQLite
  MNEMOSYNE_DATA_DIR directory containing mnemosyne.db.
    Default: ~/.hermes/mnemosyne/data

External services
- Qdrant http://localhost:6333
  Collections: mnemosyne, loci_sessions, loci_memory (all 768d/Cosine, named vector "dense")
  Auth: api-key header from QDRANT_API_KEY
  Used by: memory_recall (semantic), session_search, memory_stats
  Search payload must include
  "vector": {"name": "dense", "vector": [...]} (plain arrays 400)
- Ollama http://localhost:11434/v1
  Model: nomic-embed-text (768-dim, Cosine)
  Endpoint: POST /v1/embeddings {"model": "nomic-embed-text", "input": "<text>"}
  Response: data.data[0].embedding or data.embedding
  Timeout: 10s; failures degrade gracefully so FTS results still return
  Used by: memory_recall (semantic leg), session_search
- Mnemosyne SQLite ~/.hermes/mnemosyne/data/mnemosyne.db
  Reads: fts_working, fts_episodes, memories, episodic_memory
  Writes: memories (memory_remember)
  Used by: memory_recall, memory_remember, memory_stats
- Mnemosyne Dashboard http://127.0.0.1:8765 (optional, local only)
  Endpoint: POST /api/sleep {"dry_run": bool}
  Used by: memory_sleep
  Failure: returns status="deferred"; server keeps running

Endpoints
  GET  /.well-known/agent.json      Agent card (RFC-002) — no auth
  GET  /.well-known/agent-card.json Agent card alias — no auth
  GET  /health                      Liveness + config check — no auth
  POST /bootstrap                   Exchange bootstrap key -> 24h session token — no auth
  POST /a2a                         JSON-RPC 2.0 dispatch — ******
  GET  /a2a/tasks/{task_id}         Task status — ******

Skills
  _SKILL_MAP is the source of truth and is also exposed by GET /health.
  Public descriptions live in AGENT_CARD. See README.md "A2A skills (13)".

JSON-RPC call shape
  POST /a2a
  Authorization: ******
  X-TOTP: <6-digit code>  (only if LOCI_A2A_TOTP_SEED is set)

  {
    "jsonrpc": "2.0",
    "id": "<caller-uuid>",
    "method": "tasks/send",
    "params": {
      "skill_id": "memory_recall",
      "message": "recent authentication decisions",
      "input": {"query": "recent authentication decisions", "top_k": 5},
      "sender": "hermes-agent"
    }
  }

  Response:
  {
    "jsonrpc": "2.0",
    "id": "<caller-uuid>",
    "result": {
      "task_id": "<uuid>",
      "status": "completed",
      "output": { <skill-specific output> }
    }
  }
"""

import os, sys, asyncio, uuid, json, sqlite3, logging, datetime, hmac, time, collections, secrets, threading, hashlib, re
from typing import Optional, Any
from contextlib import contextmanager

# Accept legacy HERMES_* spellings. This server runs standalone and reaches
# the map by path, not by package.
try:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "mcp"))
    from legacy_env import apply as _apply_legacy_env
    _apply_legacy_env()
except Exception:
    pass

# ── load .env before anything else ─────────────────────────────────────────────
# Override with LOCI_ENV_FILE. Default: ~/.hermes/.env, then the legacy
# per-profile path for backward compatibility.
_ENV_FILE = os.path.expanduser(
    os.environ.get('LOCI_ENV_FILE', '~/.hermes/.env')
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
A2A_HOST  = os.environ.get('LOCI_A2A_HOST', '0.0.0.0')
A2A_PORT  = int(os.environ.get('LOCI_A2A_PORT', '8201'))
A2A_TOKEN = os.environ.get('LOCI_A2A_TOKEN', '')
if not A2A_TOKEN:
    print('WARNING: LOCI_A2A_TOKEN is not set. All bearer-token checks will fail. '
          'Generate one with: python3 -c "import secrets;print(secrets.token_hex(32))"',
          flush=True)
TOTP_SEED      = os.environ.get('LOCI_A2A_TOTP_SEED', '')
BOOTSTRAP_KEY  = os.environ.get('LOCI_A2A_BOOTSTRAP_KEY', '')
AGENT_ID  = os.environ.get('HERMES_AGENT_ID', 'hermes-agent')
AGENT_URL = os.environ.get('LOCI_A2A_URL', 'http://127.0.0.1:8201')

# Mnemosyne SQLite — path built from MNEMOSYNE_DATA_DIR (set in .env)
_mnem_data_dir = os.path.expanduser(os.environ.get('MNEMOSYNE_DATA_DIR', '~/.hermes/mnemosyne/data'))
MNEMOSYNE_DB   = os.path.join(_mnem_data_dir, 'mnemosyne.db')

# Qdrant

QDRANT_URL = os.environ.get('QDRANT_URL')
QDRANT_KEY = os.environ.get('QDRANT_API_KEY', '')

# Ollama embedding (same config as hooks/pre_llm_grounding.py + session_end_sync.py)
OLLAMA_BASE           = os.environ.get('MNEMOSYNE_EMBEDDING_API_URL', 'http://localhost:11434/v1')
EMBED_MODEL           = os.environ.get('MNEMOSYNE_EMBEDDING_MODEL',   'nomic-embed-text')
EMBED_DIM             = int(os.environ.get('MNEMOSYNE_EMBEDDING_DIM', '768'))
_EMBED_API_KEY        = os.environ.get('EMBED_API_KEY', '')
_EMBED_API_KEY_HEADER = os.environ.get('EMBED_API_KEY_HEADER', 'Authorization')

# Core collections always available; extra collections are project-specific and opt-in.
_CORE_RAG_COLLECTIONS = ['mnemosyne', 'loci_sessions', 'loci_memory']
_EXTRA_RAG_COLLECTIONS = [
    c.strip() for c in os.environ.get('EXTRA_RAG_COLLECTIONS', '').split(',')
    if c.strip()
]

logging.basicConfig(level=logging.INFO,
                    format=f'%(asctime)s [{AGENT_ID}] %(message)s')
log = logging.getLogger(__name__)

# ── per-skill privilege tiers ───────────────────────────────────────────────────
# Skills that mutate/delete data — restricted to explicitly allowlisted senders.
DESTRUCTIVE_SKILLS: frozenset[str] = frozenset({
    'memory_remember',
    'memory_sleep',
    'context_broadcast',
    'mnemosyne_triple_add',
})

# Explicitly allowed skills for peer fan-out envelopes.
_PEER_FANOUT_ALLOWLIST: frozenset[str] = frozenset({
    'memory_remember',
    'memory_prime',
})

# Boundary lanes accepted via tasks/send metadata.
_BOUNDARY_LANE_SKILL_ALLOWLIST: dict[str, frozenset[str]] = {
    'context_broadcast': frozenset({'memory_remember'}),
    'memory_prime': frozenset({'memory_prime'}),
}

_IDEMPOTENCY_KEY_RE = re.compile(r'^[A-Za-z0-9._:-]{8,128}$')
_SHA256_HEX_RE = re.compile(r'^[a-f0-9]{64}$')
_IDEMPOTENCY_TTL_S = max(60, int(os.environ.get('LOCI_A2A_IDEMPOTENCY_TTL_S', '3600')))

# Senders allowed to call destructive skills. Configure via
# LOCI_A2A_PRIVILEGED_SENDERS=agent1,agent2.
_PRIVILEGED_SENDERS: frozenset[str] = frozenset(
    s.strip() for s in os.getenv('LOCI_A2A_PRIVILEGED_SENDERS', '').split(',') if s.strip()
)

# ── agent card (RFC-002 schema) ─────────────────────────────────────────────────
AGENT_CARD = {
    'name': AGENT_ID,
    'description': (
        'Persistent memory and knowledge node for the Hermes agent mesh. '
        'FTS + semantic search over session history, episodic memory, and working memory. '
        'Write new memories with cross-agent author tagging. '
        'Backend: Mnemosyne SQLite + Qdrant loci_sessions/mnemosyne/loci_memory collections.'
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
                'direct Qdrant credentials. Core: loci_memory, loci_sessions, mnemosyne. '
                'Additional collections: set EXTRA_RAG_COLLECTIONS env var (comma-separated). '
                'Cross-encoder reranking when RERANK_HTTP_URL is set (fails open to cosine order). '
                'Input: {query: str, top_k?: int=5, collections?: [str]}'
            )
        },
        {
            'id': 'context_broadcast',
            'name': 'Context Broadcast',
            'description': (
                'Store a memory locally AND push it to all peer A2A endpoints (PEER_A2A_URLS). '
                'Used by the context bridge cron to propagate discoveries across the mesh. '
                'Set store_local=false when relaying something this node already holds. '
                'Input: {content: str, source?: str, importance?: float=0.5, bank?: str, '
                'store_local?: bool=true}'
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
# Keyed by caller_id (sender) then task_id for per-caller isolation.
# Bounded at _TASK_CAP entries per caller; oldest evicted first.
_TASK_CAP = 1000
_tasks: dict[str, dict[str, dict]] = {}
_tasks_lock = threading.Lock()
_idempotency_seen: dict[str, float] = {}
_idempotency_lock = threading.Lock()


def _store_task(task_id: str, task: dict) -> None:
    caller_id = task.get('sender', 'unknown')
    with _tasks_lock:
        bucket = _tasks.setdefault(caller_id, {})
        if len(bucket) >= _TASK_CAP:
            oldest = next(iter(bucket))
            del bucket[oldest]
        bucket[task_id] = task

# session tokens issued by /bootstrap — token → expiry (UTC)
_session_tokens: dict[str, datetime.datetime] = {}
_session_token_agents: dict[str, str] = {}


def _json_dumps_stable(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _artifact_payload_for_skill(skill_id: str, params: dict) -> Optional[dict]:
    inp = params.get('input', {})
    if not isinstance(inp, dict):
        return None

    if skill_id == 'memory_remember':
        content = str(inp.get('content') or params.get('message', '')).strip()
        if not content:
            return None
        return {
            'content': content,
            'source': str(inp.get('source', 'a2a')),
            'bank': str(inp.get('bank', 'default')),
        }
    if skill_id == 'memory_prime':
        topic = str(inp.get('topic', '') or params.get('message', '')).strip()
        if not topic:
            return None
        return {
            'topic': topic,
            'skepticism_delta': float(inp.get('skepticism_delta', 0.2)),
            'ttl_seconds': int(inp.get('ttl_seconds', 3600)),
            'broadcast': bool(inp.get('broadcast', True)),
        }
    return None


def _build_boundary_receipt(*, accepted: bool, reason: str, skill_id: str,
                            sender: str, lane: str, metadata: Optional[dict] = None) -> dict:
    return {
        'receipt_id': str(uuid.uuid4()),
        'ts': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'accepted': accepted,
        'reason': reason,
        'skill_id': skill_id,
        'sender': sender,
        'lane': lane,
        'metadata': metadata or {},
    }


def _log_boundary_receipt(receipt: dict) -> None:
    level = log.info if receipt.get('accepted') else log.warning
    level(f"boundary_receipt={_json_dumps_stable(receipt)}")


def _validate_and_reserve_idempotency(sender: str, skill_id: str,
                                      idempotency_key: str) -> tuple[bool, str]:
    now = time.time()
    compound = f'{sender}:{skill_id}:{idempotency_key}'
    with _idempotency_lock:
        expired = [k for k, exp in _idempotency_seen.items() if exp <= now]
        for k in expired:
            _idempotency_seen.pop(k, None)
        if compound in _idempotency_seen:
            return False, 'idempotency key replay'
        _idempotency_seen[compound] = now + _IDEMPOTENCY_TTL_S
    return True, 'accepted'


def _validate_boundary_metadata(skill_id: str, sender: str, params: dict) -> Optional[dict]:
    inp = params.get('input', {})
    if not isinstance(inp, dict):
        return _build_boundary_receipt(
            accepted=False,
            reason='input must be an object',
            skill_id=skill_id,
            sender=sender,
            lane='unknown',
        )
    boundary = inp.get('_boundary')
    if boundary is None:
        return None
    if not isinstance(boundary, dict):
        return _build_boundary_receipt(
            accepted=False,
            reason='_boundary must be an object',
            skill_id=skill_id,
            sender=sender,
            lane='unknown',
        )

    lane = str(boundary.get('lane', '')).strip()
    if lane not in _BOUNDARY_LANE_SKILL_ALLOWLIST:
        return _build_boundary_receipt(
            accepted=False,
            reason='lane not allowlisted',
            skill_id=skill_id,
            sender=sender,
            lane=lane or 'unknown',
            metadata={'lane': lane},
        )
    if skill_id not in _BOUNDARY_LANE_SKILL_ALLOWLIST[lane]:
        return _build_boundary_receipt(
            accepted=False,
            reason='lane/skill mismatch',
            skill_id=skill_id,
            sender=sender,
            lane=lane,
            metadata={'lane': lane},
        )

    idempotency_key = str(boundary.get('idempotency_key', '')).strip()
    if not _IDEMPOTENCY_KEY_RE.fullmatch(idempotency_key):
        return _build_boundary_receipt(
            accepted=False,
            reason='invalid idempotency_key format',
            skill_id=skill_id,
            sender=sender,
            lane=lane,
        )

    artifact_sha256 = str(boundary.get('artifact_sha256', '')).strip().lower()
    if not _SHA256_HEX_RE.fullmatch(artifact_sha256):
        return _build_boundary_receipt(
            accepted=False,
            reason='invalid artifact_sha256 format',
            skill_id=skill_id,
            sender=sender,
            lane=lane,
        )

    artifact_payload = _artifact_payload_for_skill(skill_id, params)
    if artifact_payload is None:
        return _build_boundary_receipt(
            accepted=False,
            reason='artifact payload missing or invalid',
            skill_id=skill_id,
            sender=sender,
            lane=lane,
        )

    computed_hash = hashlib.sha256(_json_dumps_stable(artifact_payload).encode('utf-8')).hexdigest()
    if not hmac.compare_digest(artifact_sha256, computed_hash):
        return _build_boundary_receipt(
            accepted=False,
            reason='artifact_sha256 mismatch',
            skill_id=skill_id,
            sender=sender,
            lane=lane,
            metadata={'expected': computed_hash, 'provided': artifact_sha256},
        )

    ok, reason = _validate_and_reserve_idempotency(sender, skill_id, idempotency_key)
    if not ok:
        return _build_boundary_receipt(
            accepted=False,
            reason=reason,
            skill_id=skill_id,
            sender=sender,
            lane=lane,
            metadata={'idempotency_key': idempotency_key},
        )

    return _build_boundary_receipt(
        accepted=True,
        reason='accepted',
        skill_id=skill_id,
        sender=sender,
        lane=lane,
        metadata={
            'idempotency_key': idempotency_key,
            'artifact_sha256': artifact_sha256,
        },
    )

# ── FastAPI app + auth ──────────────────────────────────────────────────────────
app = FastAPI(title=f'{AGENT_ID} A2A', version='0.1.0')
_bearer_scheme = HTTPBearer(auto_error=False)


def _is_live_session_token(tok: str) -> bool:
    """True if tok is a /bootstrap-issued session token that has not expired."""
    exp = _session_tokens.get(tok)
    return bool(exp and exp > datetime.datetime.now(datetime.timezone.utc))


def _verify_bearer(creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme)):
    if not creds:
        raise HTTPException(status_code=401, detail='Unauthorized — missing bearer token')
    tok = creds.credentials
    if hmac.compare_digest(tok, A2A_TOKEN):
        return {'token_type': 'primary', 'sender': None}
    if _is_live_session_token(tok):
        return {'token_type': 'session', 'sender': _session_token_agents.get(tok, 'unknown')}
    raise HTTPException(status_code=401, detail='Unauthorized — invalid or expired token')


# TOTP rate limiter: max 5 attempts per 60 seconds per client IP
_TOTP_WINDOW = 60
_TOTP_MAX_ATTEMPTS = 5
_totp_attempts: dict = collections.defaultdict(list)
# Sweep idle client-IP keys once the dict exceeds this. Chosen well above any
# plausible concurrent-client count so the sweep is rare and its cost amortised.
_TOTP_SWEEP_AFTER = 1024
_totp_attempts_lock = threading.Lock()

# Bootstrap rate limiter: max 5 failed attempts per 5 minutes per client IP.
_BOOTSTRAP_WINDOW = 300
_BOOTSTRAP_MAX_ATTEMPTS = 5
_bootstrap_attempts: dict = collections.defaultdict(list)
_BOOTSTRAP_SWEEP_AFTER = 1024
_bootstrap_attempts_lock = threading.Lock()


def _bound_sender(requested_sender: Optional[str], auth: dict) -> str:
    """Bind bootstrap-issued session tokens to their issuing agent_id."""
    if auth.get('token_type') == 'session':
        sender = auth.get('sender') or 'unknown'
        if requested_sender and requested_sender != sender:
            raise HTTPException(
                status_code=403,
                detail='Bootstrap session tokens are bound to their issuing agent_id',
            )
        return sender
    return requested_sender or 'unknown'


def _verify_totp(request: Request,
                 x_totp: Optional[str] = Header(default=None),
                 creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme)):
    # A /bootstrap session token already proves possession of the pre-shared
    # bootstrap key and carries its own expiry, so it stands alone as a bearer —
    # that is the entire point of issuing one. Without this, enabling TOTP makes
    # session tokens useless (they'd still need X-TOTP) and /bootstrap's
    # 'totp_required': False response would be a lie.
    if creds and _is_live_session_token(creds.credentials):
        return
    if TOTP_SEED:
        if x_totp is None:
            raise HTTPException(status_code=401, detail='X-TOTP header required (TOTP is enabled)')
        client_ip = request.client.host if request.client else 'unknown'
        now = time.monotonic()
        with _totp_attempts_lock:
            # Evict timestamps older than the window, and drop the key when it
            # empties. Pruning only the lists left one dict entry per client IP
            # forever — an unauthenticated caller could grow it by rotating
            # source addresses, and a busy server grew it just by being used.
            fresh = [t for t in _totp_attempts[client_ip] if now - t < _TOTP_WINDOW]
            if len(fresh) >= _TOTP_MAX_ATTEMPTS:
                _totp_attempts[client_ip] = fresh
                raise HTTPException(status_code=429, detail='Too many TOTP attempts — try again later')
            fresh.append(now)
            _totp_attempts[client_ip] = fresh
            # Opportunistic sweep: bounded work, keeps idle keys from accumulating
            # between requests without needing a background task.
            if len(_totp_attempts) > _TOTP_SWEEP_AFTER:
                for ip in [k for k, v in _totp_attempts.items()
                           if not v or now - v[-1] >= _TOTP_WINDOW]:
                    del _totp_attempts[ip]
        if not pyotp.TOTP(TOTP_SEED).verify(x_totp, valid_window=1):
            raise HTTPException(status_code=401, detail='Invalid TOTP code')


# ── helpers ─────────────────────────────────────────────────────────────────────

def _public_error(public_message: str, *, log_message: str,
                  exc: Optional[BaseException] = None, **extra) -> dict:
    if exc is None:
        log.error(log_message)
    else:
        log.exception(log_message)
    payload = {'error': public_message}
    payload.update(extra)
    return payload

_http_session: aiohttp.ClientSession | None = None

def _get_http_session() -> aiohttp.ClientSession:
    global _http_session
    if _http_session is None or _http_session.closed:
        _http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
    return _http_session


@contextmanager
def _db():
    """Open Mnemosyne SQLite with row_factory, commit on success, always close."""
    conn = sqlite3.connect(MNEMOSYNE_DB, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _embed_auth_headers() -> dict:
    h = {'Content-Type': 'application/json'}
    if _EMBED_API_KEY:
        if _EMBED_API_KEY_HEADER.lower() == 'authorization':
            h['Authorization'] = f'Bearer {_EMBED_API_KEY}'
        else:
            h[_EMBED_API_KEY_HEADER] = _EMBED_API_KEY
    return h


async def _embed(text: str) -> Optional[list]:
    """Embed via OpenAI-compatible /v1/embeddings for Ollama or cloud providers."""
    url = OLLAMA_BASE.rstrip('/').removesuffix('/v1') + '/v1/embeddings'
    try:
        sess = _get_http_session()
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
    """Named-vector semantic search in Qdrant using the `dense` vector name."""
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
        sess = _get_http_session()
        async with sess.post(url, json=body, headers=headers) as r:
            if r.status == 200:
                return (await r.json()).get('result', [])
            log.warning(f'qdrant search {collection}: HTTP {r.status}')
    except Exception as e:
        log.warning(f'qdrant search {collection}: {e}')
    return []


# ── skill: memory_recall ────────────────────────────────────────────────────────
def _recall_fts_working(query: str, top_k: int) -> list:
    """FTS on fts_working (stores id + content directly — not external content)."""
    out = []
    try:
        with _db() as conn:
            rows = conn.execute(
                "SELECT id, content FROM fts_working WHERE fts_working MATCH ? LIMIT ?",
                (query, top_k)
            ).fetchall()
            for r in rows:
                out.append({
                    'id': r['id'], 'content': r['content'],
                    'tier': 'working', 'source': 'fts_working', 'score': 0.8,
                    'importance': 0.5, 'created_at': ''
                })
    except Exception as e:
        log.warning(f'fts_working search: {e}')
    return out


def _recall_fts_episodes(query: str, top_k: int) -> list:
    """FTS on fts_episodes (external content table — join via rowid)."""
    out = []
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
                out.append({
                    'id': r['id'], 'content': r['content'],
                    'importance': float(r['importance'] or 0.5),
                    'created_at': r['created_at'] or '',
                    'tier': 'episodic', 'source': 'fts_episodic', 'score': 0.75
                })
    except Exception as e:
        log.warning(f'fts_episodes search: {e}')
    return out


def _recall_memories_like(query: str, top_k: int, seen: set) -> list:
    """LIKE fallback on plain memories table (no FTS index on this table)."""
    out = []
    try:
        with _db() as conn:
            rows = conn.execute(
                "SELECT id, content, importance, created_at FROM memories "
                "WHERE content LIKE ? LIMIT ?",
                (f'%{query}%', top_k)
            ).fetchall()
            for r in rows:
                if r['id'] not in seen:
                    out.append({
                        'id': r['id'], 'content': r['content'],
                        'importance': float(r['importance'] or 0.5),
                        'created_at': r['created_at'] or '',
                        'tier': 'memory', 'source': 'memories_like', 'score': 0.6
                    })
    except Exception as e:
        log.warning(f'memories LIKE search: {e}')
    return out


async def _recall_semantic(query: str, top_k: int, seen_ids: set) -> list:
    """Semantic via Qdrant mnemosyne collection."""
    out = []
    vec = await _embed(query)
    if vec:
        hits = await _qdrant_search('mnemosyne', vec, top_k=top_k)
        for h in hits:
            pl = h.get('payload', {})
            mid = pl.get('memory_id', str(h.get('id', '')))
            if float(h.get('score', 0)) < 0.59:
                continue
            if mid not in seen_ids:
                out.append({
                    'id': mid, 'content': pl.get('content', ''),
                    'importance': float(pl.get('importance', 0.5)),
                    'created_at': pl.get('created_at', ''),
                    'tier': 'episodic', 'source': 'qdrant_mnemosyne',
                    'score': round(float(h.get('score', 0)), 4)
                })
    return out


async def skill_memory_recall(task: dict) -> dict:
    inp      = task.get('input', {})
    query    = (inp.get('query') or task.get('message', '')).strip()
    top_k    = int(inp.get('top_k', 5))
    do_sem   = inp.get('semantic', True)

    if not query:
        return {'error': 'query is required'}

    results = []
    results += _recall_fts_working(query, top_k)          # 1. FTS on fts_working
    results += _recall_fts_episodes(query, top_k)         # 2. FTS on fts_episodes
    results += _recall_memories_like(                     # 3. LIKE on memories
        query, top_k, {r['id'] for r in results})
    if do_sem:                                            # 4. Semantic via Qdrant
        results += await _recall_semantic(
            query, top_k, {r['id'] for r in results})

    # Sort by score desc, dedupe, cap at top_k.
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
    now    = datetime.datetime.now(datetime.timezone.utc).isoformat()
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
        return _public_error('db write failed', log_message='memory_remember write failed', exc=e)

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
                except Exception as e:
                    log.debug(f'memory_stats: table {tbl}: {e}')
                    sqlite_stats[tbl] = -1
    except Exception:
        log.exception('memory_stats: sqlite unavailable')
        sqlite_stats['error'] = 'sqlite unavailable'

    # Qdrant collection point counts
    qdrant_stats: dict = {}
    headers = {'Content-Type': 'application/json', 'api-key': QDRANT_KEY}
    collections = _CORE_RAG_COLLECTIONS + _EXTRA_RAG_COLLECTIONS
    try:
        sess = _get_http_session()
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
                log.warning(f'memory_stats: qdrant collection {col} unavailable: {e!r}')
                qdrant_stats[col] = 'unavailable'
    except Exception:
        log.exception('memory_stats: qdrant unavailable')
        qdrant_stats['error'] = 'unavailable'

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

    hits = await _qdrant_search('loci_sessions', vec, top_k=top_k,
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
        sess = _get_http_session()
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
async def _rerank(query: str, hits: list) -> bool:
    """Reorder `hits` in place with a cross-encoder when `RERANK_HTTP_URL` is set.

    rag_search merges raw cosine scores from different collections, and those
    scores are not cross-collection comparable. A llama.cpp
    `llama-server --rerank --pooling rank` endpoint fixes that weakest link
    without adding a second model or a torch dependency here.

    Fails open: unset URL, unreachable server, malformed response, or a short /
    oversized result list leave cosine order untouched and return `False`.
    Return `True` only when the order changed.
    """
    url = os.environ.get('RERANK_HTTP_URL', '').strip()
    if not url or len(hits) < 2:
        return False

    docs = [(h.get('content') or '') for h in hits]
    if not any(d.strip() for d in docs):
        return False

    try:
        sess = _get_http_session()
        async with sess.post(url, json={'query': query, 'documents': docs}) as r:
            if r.status != 200:
                log.warning(f'rerank: HTTP {r.status} — keeping cosine order')
                return False
            data = await r.json()
    except Exception as e:
        log.warning(f'rerank: {e!r} — keeping cosine order')
        return False

    scored = []
    for item in (data.get('results') or []):
        try:
            idx = int(item['index'])
            if 0 <= idx < len(hits):
                scored.append((idx, float(item['relevance_score'])))
        except (KeyError, TypeError, ValueError):
            continue

    # Partial coverage would drop hits, so accept only a complete permutation.
    if len(scored) != len(hits) or len({i for i, _ in scored}) != len(hits):
        log.warning(f'rerank: got {len(scored)} scores for {len(hits)} hits — keeping cosine order')
        return False

    scored.sort(key=lambda t: t[1], reverse=True)
    reordered = []
    for idx, score in scored:
        h = hits[idx]
        h['cosine_score'] = h.get('score')      # keep the original so callers can compare
        h['rerank_score'] = round(score, 4)
        reordered.append(h)
    hits[:] = reordered
    return True


async def skill_rag_search(task: dict) -> dict:
    """Fan out semantic search across all 768-dim Qdrant collections.

    Lets mesh agents query the shared corpus without direct Qdrant
    credentials or hardcoded collection names.

    Input:  {query: str, top_k?: int=5, collections?: [str]}
    Output: {results: [...merged ranked hits], query: str, collections_searched: [str]}
    """
    inp         = task.get('input', {})
    query       = (inp.get('query') or task.get('message', '')).strip()
    top_k       = int(inp.get('top_k', 5))
    req_cols    = inp.get('collections')

    if not query:
        return {'error': 'query is required'}

    # Default collection list: core plus any env-configured extras.
    ALL_COLLECTIONS = _CORE_RAG_COLLECTIONS + _EXTRA_RAG_COLLECTIONS
    collections = req_cols if (req_cols and isinstance(req_cols, list)) else ALL_COLLECTIONS

    vec = await _embed(query)
    if not vec:
        return {'error': 'embedding failed — Ollama may be unreachable', 'results': []}

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

    # Deduplicate by (collection, id).
    seen, deduped = set(), []
    for h in all_hits:
        key = f"{h['collection']}:{h['id']}"
        if key not in seen:
            seen.add(key)
            deduped.append(h)

    reranked = await _rerank(query, deduped)

    return {
        'results':              deduped[:top_k * 2],   # return more than top_k since multi-collection
        'query':                query,
        'collections_searched': collections,
        'reranked':             reranked,
        'total_hits':           len(deduped),
    }


# ── skill: context_broadcast ─────────────────────────────────────────────────────
def _peer_base_url(peer_url: str) -> str:
    """Strip one trailing `/a2a` path segment to get a peer base URL.

    Use an explicit suffix check, not `str.rstrip('/a2a')`, because `rstrip`
    removes any trailing run of `/`, `a`, and `2` and can mangle ports.
    """
    url = peer_url.rstrip('/')
    if url.endswith('/a2a'):
        url = url[:-len('/a2a')]
    return url.rstrip('/')


def _peer_credentials() -> tuple[dict, str, dict, str]:
    """Read peer auth config from env.

    Returns `(token_map, default_token, seed_map, default_seed)`.
    `PEER_A2A_TOKEN` / `PEER_A2A_TOKENS_JSON` supply shared or per-peer
    ******; `PEER_A2A_TOTP_SEED` / `PEER_A2A_TOTP_SEEDS_JSON` do the same for
    base32 TOTP seeds.
    """
    default_token = os.environ.get('PEER_A2A_TOKEN', '')
    default_seed  = os.environ.get('PEER_A2A_TOTP_SEED', '')
    try:
        token_map = json.loads(os.environ.get('PEER_A2A_TOKENS_JSON', '{}'))
    except Exception as e:
        log.debug(f'_peer_credentials: PEER_A2A_TOKENS_JSON unparseable ({e!r}) '
                  f'— falling back to per-peer map {{}}')
        token_map = {}
    try:
        seed_map = json.loads(os.environ.get('PEER_A2A_TOTP_SEEDS_JSON', '{}'))
    except Exception as e:
        log.debug(f'_peer_credentials: PEER_A2A_TOTP_SEEDS_JSON unparseable ({e!r}) '
                  f'— falling back to per-peer map {{}}')
        seed_map = {}
    return token_map, default_token, seed_map, default_seed


def _peer_headers(peer_url: str, token_map: dict, default_token: str,
                  seed_map: dict, default_seed: str):
    """Build outbound headers for one peer.

    Returns `(headers, None)` on success or `(None, reason)` if the peer is
    not callable. Peers with `LOCI_A2A_TOTP_SEED` reject bearer-only requests,
    so attach `X-TOTP` whenever a seed is configured.
    """
    base  = _peer_base_url(peer_url)
    token = token_map.get(peer_url) or token_map.get(base) or default_token
    if not token:
        return None, 'no token configured'

    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type':  'application/json',
    }
    seed = seed_map.get(peer_url) or seed_map.get(base) or default_seed
    if seed:
        try:
            headers['X-TOTP'] = pyotp.TOTP(seed).now()
        except Exception as e:
            log.warning(f'_peer_headers: invalid TOTP seed for {peer_url}: {e!r}')
            return None, 'invalid TOTP seed'
    return headers, None


def _peer_targets() -> list[tuple[str, Optional[dict], Optional[str]]]:
    """Resolve `PEER_A2A_URLS` into `(peer_url, headers, skip_reason)`.

    Preserve env order and read credentials once. `headers is None` means the
    peer is not callable and `skip_reason` explains why.
    """
    token_map, default_token, seed_map, default_seed = _peer_credentials()
    targets: list[tuple[str, Optional[dict], Optional[str]]] = []
    for raw in os.environ.get('PEER_A2A_URLS', '').split(','):
        peer_url = raw.strip()
        if not peer_url:
            continue
        headers, reason = _peer_headers(
            peer_url, token_map, default_token, seed_map, default_seed)
        targets.append((peer_url, headers, reason))
    return targets


def _peer_task_payload(skill_id: str, message: str, inp: dict) -> dict:
    """Build the JSON-RPC tasks/send envelope used for every peer fan-out."""
    if skill_id not in _PEER_FANOUT_ALLOWLIST:
        raise ValueError(f'peer skill {skill_id!r} is not allowlisted')

    lane = 'context_broadcast' if skill_id == 'memory_remember' else 'memory_prime'
    artifact_source = _json_dumps_stable(_artifact_payload_for_skill(
        skill_id,
        {'message': message, 'input': inp},
    ) or {})
    boundary = {
        'lane': lane,
        'idempotency_key': str(uuid.uuid4()),
        'artifact_sha256': hashlib.sha256(artifact_source.encode('utf-8')).hexdigest(),
        'origin_agent_id': AGENT_ID,
        'ts': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    payload_input = dict(inp)
    payload_input['_boundary'] = boundary
    return {
        'jsonrpc': '2.0',
        'id': str(uuid.uuid4()),
        'method': 'tasks/send',
        'params': {
            'skill_id': skill_id,
            'message':  message,
            'input':    payload_input,
            'sender':   AGENT_ID,
        },
    }


async def skill_context_broadcast(task: dict) -> dict:
    """Push one memory to all configured peer A2A endpoints.

    Used by the `a2a_context_bridge` cron to propagate discoveries through the
    mesh.

    Input:  {content: str, source?: str, importance?: float=0.5, bank?: str="default"}
    Output: {broadcast: [{peer, status, error?}], stored_locally: bool}

    Peer auth config:
      - `PEER_A2A_URLS`: comma-separated list such as
        `http://peer-a:8201/a2a,http://peer-b:8201/a2a`
      - `PEER_A2A_TOKEN`: shared ******, overrideable per peer via
        `PEER_A2A_TOKENS_JSON`
      - `PEER_A2A_TOTP_SEED`: shared base32 TOTP seed, overrideable per peer
        via `PEER_A2A_TOTP_SEEDS_JSON`
    """
    inp        = task.get('input', {})
    content    = (inp.get('content') or task.get('message', '')).strip()
    source     = inp.get('source', 'context_broadcast')
    importance = float(inp.get('importance', 0.5))
    bank       = inp.get('bank', 'default')
    sender     = task.get('sender', AGENT_ID)

    if not content:
        return {'error': 'content is required'}

    # 1. Store locally first unless the caller is relaying something this node
    # already has. Re-storing a bridged memory gives it a new id/created_at, makes
    # it look new to the next bridge run, and causes unbounded local duplication.
    store_local = bool(inp.get('store_local', True))
    if store_local:
        local_result = await skill_memory_remember({
            'input': {'content': content, 'source': source,
                      'importance': importance, 'bank': bank},
            'sender': sender
        })
        stored_locally = 'error' not in local_result
    else:
        stored_locally = False

    # 2. Fan out to peers.
    targets   = _peer_targets()
    peer_urls = [u for u, _, _ in targets]
    broadcast_results = []

    async def _push_to_peer(peer_url: str, headers: Optional[dict], reason: Optional[str]):
        if headers is None:
            return {'peer': peer_url, 'status': 'skipped', 'error': reason}

        payload = _peer_task_payload(
            'memory_remember', content,
            {'content': content, 'source': f'broadcast:{AGENT_ID}',
             'importance': importance, 'bank': bank},
        )
        try:
            sess = _get_http_session()
            async with sess.post(peer_url, json=payload, headers=headers) as r:
                if r.status == 200:
                    data = await r.json()
                    return {'peer': peer_url, 'status': 'ok',
                            'output': data.get('result', {}).get('output', {})}
                body = await r.text()
                log.warning(f'context_broadcast peer {peer_url} HTTP {r.status}: {body[:200]!r}')
                return {'peer': peer_url, 'status': f'http_{r.status}',
                        'error': 'peer request failed'}
        except Exception as e:
            log.warning(f'context_broadcast peer {peer_url} failed: {e!r}')
            return {'peer': peer_url, 'status': 'error', 'error': 'peer request failed'}

    if peer_urls:
        results = await asyncio.gather(*[_push_to_peer(*t) for t in targets])
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
    valid_from = inp.get('valid_from') or datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')
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
        return _public_error('db write failed', log_message='triple_add write failed', exc=e)

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
        return _public_error('db query failed', log_message='triple_query failed',
                             exc=e, triples=[])

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
        sess = _get_http_session()
        async with sess.post(
            f"{OLLAMA_BASE.rstrip('/').removesuffix('/v1')}/v1/chat/completions",
            json={'model': model, 'messages': messages, 'max_tokens': max_t},
        ) as r:
            d = await r.json()
            content = d['choices'][0]['message']['content']
            return {'response': content, 'model': model, 'status': 'ok'}
    except Exception as e:
        return _public_error('inference failed', log_message='gpu_inference failed',
                             exc=e, status='error')


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
        if r.returncode != 0:
            log.warning(f'docker_status: docker ps exited {r.returncode}: {r.stderr.strip()!r}')
            results['docker_error'] = 'docker unavailable'
        else:
            containers = []
            for line in r.stdout.strip().splitlines():
                parts = line.split('\t')
                if len(parts) >= 3:
                    entry = {'name': parts[0], 'status': parts[1], 'image': parts[2]}
                    if not flt or flt.lower() in parts[0].lower():
                        containers.append(entry)
            results['docker'] = containers
    except Exception as e:
        log.warning(f'docker_status: docker probe failed: {e!r}')
        results['docker_error'] = 'docker unavailable'

    # k3s pods
    try:
        ns_flag = ['--all-namespaces'] if ns == 'all' else ['-n', ns]
        r = subprocess.run(
            ['kubectl', 'get', 'pods'] + ns_flag + ['--no-headers',
             '-o', 'custom-columns=NS:.metadata.namespace,NAME:.metadata.name,STATUS:.status.phase,READY:.status.containerStatuses[0].ready'],
            capture_output=True, text=True, timeout=15
        )
        if r.returncode != 0:
            log.warning(f'docker_status: kubectl get pods exited {r.returncode}: {r.stderr.strip()!r}')
            results['k3s_error'] = 'kubectl unavailable'
        else:
            pods = []
            for line in r.stdout.strip().splitlines():
                parts = line.split()
                if len(parts) >= 3:
                    entry = {'namespace': parts[0], 'name': parts[1], 'status': parts[2], 'ready': parts[3] if len(parts) > 3 else '?'}
                    if not flt or flt.lower() in parts[1].lower():
                        pods.append(entry)
            results['k3s_pods'] = pods
    except Exception as e:
        log.warning(f'docker_status: kubectl probe failed: {e!r}')
        results['k3s_error'] = 'kubectl unavailable'

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

    import subprocess
    ua_script = os.environ.get('UA_SEARCH_SCRIPT', '')
    if not ua_script or not os.path.exists(ua_script):
        return {'error': 'UA_SEARCH_SCRIPT env var not set or script not found', 'query': query}
    args = [sys.executable, ua_script, query, '--json', '-n', str(limit)]
    if repo:  args += ['--repo', repo]
    if ntype: args += ['--type', ntype]
    if layer: args += ['--layer', layer]

    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            log.warning(f'ua_search exited {r.returncode}: {r.stderr.strip()[-500:]!r}')
            return {'error': 'ua_search failed', 'query': query}
        results = json.loads(r.stdout) if r.stdout.strip() else []
        return {'results': results, 'count': len(results), 'query': query}
    except Exception as e:
        return _public_error('ua_search failed', log_message='ua_search crashed',
                             exc=e, query=query)


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
    except Exception as e:
        log.debug(f'memory_prime: priming state at {state_path} unreadable '
                  f'({e!r}) — starting fresh')
        state = {}

    expires_at = int(time.time()) + ttl_seconds
    state[topic] = {
        'skepticism_delta': skepticism_delta,
        'expires_at': expires_at,
        'source': task.get('sender', AGENT_ID),
    }
    # Prune expired entries while we have the file open.
    now_ts = int(time.time())
    state  = {t: v for t, v in state.items() if v.get('expires_at', 0) > now_ts}

    try:
        # The state dir may not exist — notably in the Docker image, where HOME
        # is /root and nothing has ever created /root/.hermes. Without this the
        # skill fails every call with ENOENT on the .tmp write.
        parent = os.path.dirname(state_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = state_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(state, f)
        os.replace(tmp, state_path)
    except Exception as e:
        return _public_error('Failed to write priming state',
                             log_message='memory_prime failed to write priming state',
                             exc=e)

    broadcast_results = []
    if do_broadcast:
        targets = _peer_targets()

        async def _prime_peer(peer_url: str, headers: Optional[dict], reason: Optional[str]):
            if headers is None:
                return {'peer': peer_url, 'status': 'skipped', 'error': reason}
            payload = _peer_task_payload(
                'memory_prime', topic,
                {'topic': topic, 'skepticism_delta': skepticism_delta,
                 'ttl_seconds': ttl_seconds, 'broadcast': False},
            )
            try:
                sess = _get_http_session()
                async with sess.post(
                    peer_url, json=payload,
                    headers=headers,
                ) as resp:
                    return {'peer': peer_url, 'status': resp.status}
            except Exception as e:
                log.warning(f'memory_prime peer {peer_url} failed: {e!r}')
                return {'peer': peer_url, 'status': 'error', 'error': 'peer request failed'}

        broadcast_results = await asyncio.gather(
            *[_prime_peer(*t) for t in targets], return_exceptions=False
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
        'bootstrap_configured': bool(BOOTSTRAP_KEY),
        'active_sessions': len(_session_tokens),
    })


@app.post('/bootstrap')
async def bootstrap(request: Request):
    """Exchange a bootstrap key for a 24h session token (no TOTP required on session token)."""
    if not BOOTSTRAP_KEY:
        raise HTTPException(status_code=501, detail='Bootstrap not configured — set LOCI_A2A_BOOTSTRAP_KEY')
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid JSON')
    presented = body.get('bootstrap_key', '')
    client_ip = request.client.host if request.client else 'unknown'
    now_mono = time.monotonic()
    with _bootstrap_attempts_lock:
        fresh = [t for t in _bootstrap_attempts[client_ip] if now_mono - t < _BOOTSTRAP_WINDOW]
        if len(fresh) >= _BOOTSTRAP_MAX_ATTEMPTS:
            _bootstrap_attempts[client_ip] = fresh
            raise HTTPException(status_code=429, detail='Too many bootstrap attempts — try again later')
    if not secrets.compare_digest(presented, BOOTSTRAP_KEY):
        with _bootstrap_attempts_lock:
            fresh.append(now_mono)
            _bootstrap_attempts[client_ip] = fresh
            if len(_bootstrap_attempts) > _BOOTSTRAP_SWEEP_AFTER:
                for ip in [k for k, v in _bootstrap_attempts.items()
                           if not v or now_mono - v[-1] >= _BOOTSTRAP_WINDOW]:
                    del _bootstrap_attempts[ip]
        raise HTTPException(status_code=401, detail='Invalid bootstrap key')
    with _bootstrap_attempts_lock:
        _bootstrap_attempts.pop(client_ip, None)

    ttl_hours = min(int(body.get('ttl_hours', 24)), 168)  # cap at 7 days
    now = datetime.datetime.now(datetime.timezone.utc)
    expires_at = now + datetime.timedelta(hours=ttl_hours)
    session_token = secrets.token_hex(32)
    _session_tokens[session_token] = expires_at
    agent_id = body.get('agent_id', 'unknown')
    _session_token_agents[session_token] = agent_id

    # Prune expired sessions
    expired = [t for t, exp in list(_session_tokens.items()) if exp <= now]
    for t in expired:
        _session_tokens.pop(t, None)
        _session_token_agents.pop(t, None)
    log.info(f'Bootstrap: issued session token for agent_id={agent_id} ttl={ttl_hours}h expires={expires_at.isoformat()}')
    return JSONResponse({
        'session_token': session_token,
        'expires_at': expires_at.isoformat(),
        'totp_required': False,
    })


@app.post('/a2a')
async def a2a_endpoint(request: Request,
                       auth: dict = Depends(_verify_bearer),
                       _: None = Depends(_verify_totp)):
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
        params = dict(params)
        params['sender'] = _bound_sender(params.get('sender'), auth)
        return await _handle_task_send(rpc_id, params)
    if method == 'tasks/get':
        params = dict(params)
        params['sender'] = _bound_sender(params.get('sender'), auth)
        return await _handle_task_get(rpc_id, params)
    if method == 'tasks/list':
        caller_id = _bound_sender(params.get('sender'), auth)
        return JSONResponse({
            'jsonrpc': '2.0', 'id': rpc_id,
            'result': {'tasks': list(_tasks.get(caller_id, {}).values())}
        })
    return JSONResponse({
        'jsonrpc': '2.0', 'id': rpc_id,
        'error': {'code': -32601, 'message': f"Method not found: '{method}'"}
    })


@app.get('/a2a/tasks/{task_id}')
async def get_task(task_id: str, sender: Optional[str] = None,
                   auth: dict = Depends(_verify_bearer),
                   _: None = Depends(_verify_totp)):
    if auth.get('token_type') == 'primary' and not sender:
        raise HTTPException(status_code=400, detail='sender query parameter required')
    task = _tasks.get(_bound_sender(sender, auth), {}).get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail='Task not found')
    return JSONResponse(task)


async def _handle_task_send(rpc_id: str, params: dict) -> JSONResponse:
    if not isinstance(params, dict):
        return JSONResponse({
            'jsonrpc': '2.0', 'id': rpc_id,
            'error': {'code': -32602, 'message': 'params must be an object'}
        }, status_code=400)

    skill_id = params.get('skill_id', '')
    if not isinstance(skill_id, str) or not skill_id.strip():
        return JSONResponse({
            'jsonrpc': '2.0', 'id': rpc_id,
            'error': {'code': -32602, 'message': 'skill_id must be a non-empty string'}
        }, status_code=400)
    skill_id = skill_id.strip()

    if skill_id not in _SKILL_MAP:
        return JSONResponse({
            'jsonrpc': '2.0', 'id': rpc_id,
            'error': {'code': -32601, 'message': f"Unknown skill '{skill_id}'."}
        }, status_code=404)

    input_payload = params.get('input', {})
    if input_payload is None:
        input_payload = {}
    if not isinstance(input_payload, dict):
        return JSONResponse({
            'jsonrpc': '2.0', 'id': rpc_id,
            'error': {'code': -32602, 'message': 'input must be an object'}
        }, status_code=400)

    message = params.get('message', '')
    if not isinstance(message, str):
        return JSONResponse({
            'jsonrpc': '2.0', 'id': rpc_id,
            'error': {'code': -32602, 'message': 'message must be a string'}
        }, status_code=400)

    task_id  = str(uuid.uuid4())
    sender_raw = params.get('sender', 'unknown')
    sender = sender_raw if isinstance(sender_raw, str) else str(sender_raw)
    boundary_receipt = _validate_boundary_metadata(
        skill_id,
        sender,
        {'message': message, 'input': input_payload},
    )
    if boundary_receipt is not None:
        _log_boundary_receipt(boundary_receipt)
        if not boundary_receipt.get('accepted'):
            return JSONResponse({
                'jsonrpc': '2.0', 'id': rpc_id,
                'error': {
                    'code': -32600,
                    'message': 'boundary validation failed',
                    'data': boundary_receipt,
                }
            }, status_code=403)

    task = {
        'id':         task_id,
        'skill_id':   skill_id,
        'message':    message,
        'input':      input_payload,
        'sender':     sender,
        'status':     'working',
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'result':     None,
        'boundary_receipt': boundary_receipt,
    }
    _store_task(task_id, task)
    log.info(f'Task [{task_id}] skill={skill_id} sender={sender}')

    if skill_id in DESTRUCTIVE_SKILLS:
        if sender not in _PRIVILEGED_SENDERS:
            log.warning(f'Blocked destructive skill {skill_id!r} from unprivileged sender {sender!r}')
            return JSONResponse({
                'jsonrpc': '2.0', 'id': rpc_id,
                'error': {'code': -32600, 'message': f'skill {skill_id!r} requires elevated privilege'}
            }, status_code=403)

    try:
        result = await _dispatch(skill_id, task)
    except Exception:
        log.exception(f'Skill {skill_id} raised')
        result = {'error': 'Internal task error'}

    task['status'] = 'completed'
    task['result'] = result

    return JSONResponse({
        'jsonrpc': '2.0', 'id': rpc_id,
        'result': {'task_id': task_id, 'status': 'completed', 'output': result}
    })


async def _handle_task_get(rpc_id: str, params: dict) -> JSONResponse:
    task_id = params.get('task_id')
    caller_id = params.get('sender', 'unknown')
    task = _tasks.get(caller_id, {}).get(task_id) if task_id else None
    if not task:
        return JSONResponse({
            'jsonrpc': '2.0', 'id': rpc_id,
            'error': {'code': -32602, 'message': 'Task not found'}
        })
    return JSONResponse({'jsonrpc': '2.0', 'id': rpc_id, 'result': task})


# ── entrypoint ───────────────────────────────────────────────────────────────────
def main() -> None:
    log.info(f'{AGENT_ID} A2A v0.1.0  {A2A_HOST}:{A2A_PORT}')
    log.info(f'Agent card:    {AGENT_URL}/.well-known/agent.json')
    log.info(f'Mnemosyne DB:  {MNEMOSYNE_DB}  ({"found" if os.path.exists(MNEMOSYNE_DB) else "MISSING"})')
    log.info(f'Qdrant:        {QDRANT_URL}')
    log.info(f'Ollama embed:  {OLLAMA_BASE}  model={EMBED_MODEL}')
    log.info(f'TOTP:          {"enabled" if TOTP_SEED else "disabled (set LOCI_A2A_TOTP_SEED to enable)"}')
    log.info(f'Bootstrap:     {"enabled (POST /bootstrap)" if BOOTSTRAP_KEY else "disabled (set LOCI_A2A_BOOTSTRAP_KEY to enable)"}')
    log.info(f'Skills:        {", ".join(_SKILL_MAP.keys())}')
    uvicorn.run(app, host=A2A_HOST, port=A2A_PORT, log_level='info', timeout_graceful_shutdown=5)


if __name__ == '__main__':
    main()
