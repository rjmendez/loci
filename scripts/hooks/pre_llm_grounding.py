#!/usr/bin/env python3
"""
pre_llm_call hook — multi-collection Qdrant grounding (v3)

Architecture:
  1. Extract intent from user message
  2. Embed via nomic-embed-text on Ollama (70ms warm)
  3. Fan-out parallel Qdrant searches across all 768-dim collections (~30ms)
  4. Merge + rank by score * importance, deduplicate, truncate
  5. Inject MEMORY MATCH block into context

Collections searched per turn:
  mnemosyne         (198 pts)   — personal facts, preferences, project notes
  hermes_sessions   (126 pts)   — past conversation history
  hermes_memory     (89 pts)    — investigation notes, research findings
  ecc_skills        (262 pts)   — skill library knowledge
  agent_core_chunks (3.09M pts) — great-library KB (DAMA, infra, code, telemetry)
  dama_gotchi_code  (24.9k pts) — DAMA codebase search
  prometheus_dama_code (2,605 pts) — prometheus DAMA codebase

Benchmarks (vs v2 BeamMemory):
  v2 (BeamMemory / SQLite):  ~1500ms per turn
  v3 (this hook):            ~100ms per turn (70ms embed + 30ms parallel Qdrant)

v3 changes vs v2:
  - Removes BeamMemory / SQLite dependency entirely — pure Qdrant path
  - Embeds via Ollama directly (nomic-embed-text, same model used everywhere)
  - Fan-out to all 768-dim collections in a single parallel sweep
  - Score fusion: Qdrant cosine score * importance weight (where available)
  - FTS fallback: if Ollama is down, falls back to BeamMemory (v2 path)
  - Rules index injection preserved from v2
  - Subagent skip, min-length guard preserved from v2
"""
# v4 hardening:
#   - Subagent skip replaced with lightweight single-collection path (hermes_memory only)
#   - Slash-command skip narrowed to navigation-only commands; code-affecting commands grounded
#   - Short-message length guard removed entirely
#   - Ollama + BeamMemory dual failure now injects explicit WARNING instead of silently proceeding
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Optional spreading activation enrichment (SA-RAG, arxiv 2512.15922).
# Only loaded when mnemosyne collection results carry mnemosyne_id payloads.
_SA_MODULE = None
try:
    import importlib.util as _ilu
    _sa_path = os.path.join(os.path.dirname(__file__), "..", "spreading_activation.py")
    _sa_path = os.path.normpath(_sa_path)
    if os.path.exists(_sa_path):
        _spec = _ilu.spec_from_file_location("spreading_activation", _sa_path)
        _SA_MODULE = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_SA_MODULE)
except Exception:
    pass

SA_ENABLED    = os.environ.get("HOOK_SA_ENABLED", "true").lower() not in ("false", "0", "no")
SA_TIMEOUT_MS = float(os.environ.get("HOOK_SA_TIMEOUT_MS", "25"))  # skip if SA takes > 25ms

# ── env ──────────────────────────────────────────────────────────────────────
_HERMES_HOME    = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
_HERMES_PROFILE = os.environ.get("HERMES_PROFILE", "")
_PROFILE_DIR    = (os.path.join(_HERMES_HOME, "profiles", _HERMES_PROFILE)
                   if _HERMES_PROFILE else _HERMES_HOME)
_ENV_FILE = os.path.join(_PROFILE_DIR, ".env")
if os.path.exists(_ENV_FILE):
    with open(_ENV_FILE) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ── constants ─────────────────────────────────────────────────────────────────
QDRANT_URL   = os.environ.get("QDRANT_URL")
QDRANT_KEY   = os.environ.get("QDRANT_API_KEY", "")
_OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL")
OLLAMA_URL   = f"{_OLLAMA_BASE}/v1" if _OLLAMA_BASE else None
EMBED_MODEL  = os.environ.get("MNEMOSYNE_EMBEDDING_MODEL",   "nomic-embed-text")
_EMBED_API_KEY        = os.environ.get("EMBED_API_KEY", "")
_EMBED_API_KEY_HEADER = os.environ.get("EMBED_API_KEY_HEADER", "Authorization")

RULES_DIR    = os.environ.get("HOOK_RULES_DIR",
                              os.path.join(_PROFILE_DIR, "rules"))

RECALL_TOP_K      = int(os.environ.get("HOOK_RECALL_TOP_K",          "3"))
MIN_IMPORTANCE    = float(os.environ.get("HOOK_RECALL_MIN_IMPORTANCE", "0.2"))
MIN_SCORE         = float(os.environ.get("HOOK_RECALL_MIN_SCORE",      "0.55"))
MIN_PROMPT_LEN    = int(os.environ.get("HOOK_RECALL_MIN_PROMPT",       "15"))
MAX_CONTENT_CHARS = int(os.environ.get("HOOK_RECALL_MAX_CHARS",        "200"))
RULES_MAX_CHARS   = int(os.environ.get("HOOK_RULES_MAX_CHARS",         "1200"))
EMBED_TIMEOUT     = float(os.environ.get("HOOK_EMBED_TIMEOUT",         "3.0"))
QDRANT_TIMEOUT    = float(os.environ.get("HOOK_QDRANT_TIMEOUT",        "2.0"))
WORKERS           = int(os.environ.get("HOOK_QDRANT_WORKERS",          "8"))

# Multi-signal ranker weights (must sum to 1.0).
# relevance = semantic cosine × importance; recency = exponential decay by age;
# trust = confidence-tier proxy; type_w = observed > inferred > assumed > gap.
RANKER_W_RELEVANCE = float(os.environ.get("HOOK_RANKER_W_RELEVANCE", "0.50"))
RANKER_W_RECENCY   = float(os.environ.get("HOOK_RANKER_W_RECENCY",   "0.20"))
RANKER_W_TRUST     = float(os.environ.get("HOOK_RANKER_W_TRUST",     "0.15"))
RANKER_W_TYPE      = float(os.environ.get("HOOK_RANKER_W_TYPE",      "0.15"))
# Half-life for recency decay (days). 7 = a week-old finding scores 0.5 on recency.
RECENCY_HALFLIFE_DAYS = float(os.environ.get("HOOK_RECENCY_HALFLIFE_DAYS", "7"))
# MMR lambda: 1.0 = pure relevance, 0.0 = pure diversity.
MMR_LAMBDA = float(os.environ.get("HOOK_MMR_LAMBDA", "0.75"))

# Stigmergic recall (ant colony / Physarum — pheromone reinforcement + evaporation).
# Retrieved memories deposit pheromone; subsequent queries boost them proportionally.
# Evaporation via timestamp decay prevents monoculture lock-in.
PHERO_BETA         = float(os.environ.get("HOOK_PHERO_BETA",        "0.08"))  # score boost coefficient
PHERO_HALFLIFE_H   = float(os.environ.get("HOOK_PHERO_HALFLIFE_H",  "24"))    # pheromone half-life in hours
PHERO_DEPOSIT      = float(os.environ.get("HOOK_PHERO_DEPOSIT",     "1.0"))   # amount deposited per retrieval
PHERO_EPSILON      = float(os.environ.get("HOOK_PHERO_EPSILON",     "0.05"))  # ε-exploration probability

# Collections to search, with their field names for content and optional importance.
# Core collections always searched. Extra collections are project-specific:
# set GROUNDING_EXTRA_COLLECTIONS=name1,name2 to add them.
#
# Known extra-collection field mappings (name -> (content_field, importance_field, named_vec)):
_EXTRA_FIELD_MAP: dict[str, tuple] = {
    "ecc_skills":        ("content_preview", None, True),
    "agent_core_chunks": ("text",            None, False),
    "dama_gotchi_code":  ("text",            None, True),
}
_DEFAULT_EXTRA_FIELDS = ("text", None, True)

# fmt: (collection, content_field, importance_field_or_None, use_named_vector)
_BASE_COLLECTIONS = [
    ("mnemosyne",       "content",         "importance", True),
    ("hermes_sessions", "content_preview", None,         True),
    ("hermes_memory",   "text",            "confidence", True),
]
_extra_names = [
    c.strip() for c in os.environ.get("GROUNDING_EXTRA_COLLECTIONS", "").split(",")
    if c.strip()
]
COLLECTIONS = _BASE_COLLECTIONS + [
    (name,) + _EXTRA_FIELD_MAP.get(name, _DEFAULT_EXTRA_FIELDS)
    for name in _extra_names
]

# Override the full directive via env var, or use the generic default.
GROUNDING_DIRECTIVE = os.environ.get("GROUNDING_DIRECTIVE") or (
    "[GROUNDING DIRECTIVE — active every turn]\n"
    "Before answering from parametric knowledge:\n"
    "1. RECALL from memory: use the mnemosyne recall tool for the topic\n"
    "2. SEARCH: use available code search and artifact tools\n"
    "3. WEB SEARCH: for current events, versions, external documentation\n"
    "4. SESSION SEARCH: for 'what did we do about X' questions\n"
    "Skip recall only when the answer comes directly from tool output in this turn."
)

# Slash commands that are pure navigation — no code or memory work involved.
# Every other slash command (including /fix, /review, /code-review, /remember,
# /ultrareview, /schedule) gets full grounding because it may touch code.
NAVIGATION_SLASH_COMMANDS: frozenset[str] = frozenset({
    "/help", "/clear", "/compact", "/cost", "/status",
    "/history", "/ide", "/doctor", "/login", "/logout",
})


# ── embedding ─────────────────────────────────────────────────────────────────

def _embed_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if _EMBED_API_KEY:
        h["Authorization" if _EMBED_API_KEY_HEADER.lower() == "authorization"
          else _EMBED_API_KEY_HEADER] = (
            f"Bearer {_EMBED_API_KEY}"
            if _EMBED_API_KEY_HEADER.lower() == "authorization"
            else _EMBED_API_KEY
        )
    return h


def _embed(text: str) -> Optional[list[float]]:
    """Get embedding vector via OpenAI-compat /v1/embeddings endpoint.
    Works with Ollama (no auth) and cloud providers (set EMBED_API_KEY)."""
    if not OLLAMA_URL:
        return None
    url = f"{OLLAMA_URL.rstrip('/')}/embeddings"
    body = json.dumps({"model": EMBED_MODEL, "input": [text]}).encode()
    req = urllib.request.Request(url, data=body, headers=_embed_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as resp:
            d = json.loads(resp.read())
        return d["data"][0]["embedding"]
    except Exception:
        return None


# ── Qdrant search ─────────────────────────────────────────────────────────────

def _search_collection(
    collection: str,
    vector: list[float],
    content_field: str,
    importance_field: Optional[str],
    named_vec: bool,
    top_k: int = 5,
) -> list[dict]:
    """Search one Qdrant collection, return normalised hit dicts."""
    url = f"{QDRANT_URL}/collections/{collection}/points/search"
    vec_payload = {"dense": vector} if named_vec else vector
    body = json.dumps({
        "vector": vec_payload,
        "limit": top_k,
        "with_payload": True,
        "with_vector": False,
    }).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={
            "Content-Type": "application/json",
            "api-key": QDRANT_KEY,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=QDRANT_TIMEOUT) as resp:
            d = json.loads(resp.read())
    except Exception:
        return []

    hits = []
    for pt in d.get("result", []):
        score = pt.get("score", 0.0)
        if score < MIN_SCORE:
            continue
        payload = pt.get("payload") or {}
        content = payload.get(content_field, "")
        if not content:
            # fallback to any text-ish field
            for f in ("text", "content", "content_preview", "description"):
                content = payload.get(f, "")
                if content:
                    break
        if not content:
            continue
        importance = float(payload.get(importance_field, 0.5) or 0.5) \
            if importance_field else 0.5
        hits.append({
            "collection": collection,
            "point_id": pt.get("id"),   # needed for pheromone deposit
            "score": score,
            "importance": importance,
            "fused": score * importance,
            "content": str(content),
            "payload": payload,
        })
    return hits


# ── BeamMemory fallback ───────────────────────────────────────────────────────

def _beam_fallback(query: str) -> list[dict]:
    """v2 BeamMemory path — used only when Ollama is unreachable."""
    MNEMOSYNE_ROOT = os.path.expanduser("~/.hermes/mnemosyne")
    HERMES_VENV_SITE = os.path.expanduser(
        "~/.hermes/hermes-agent/venv/lib/python3.11/site-packages"
    )
    for p in (HERMES_VENV_SITE, MNEMOSYNE_ROOT):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from mnemosyne.core.beam import BeamMemory  # type: ignore
    except ImportError:
        return []
    try:
        beam = BeamMemory(session_id=os.environ.get("HERMES_AGENT_ID", "hermes"))
        results = beam.recall(
            query, top_k=RECALL_TOP_K,
            vec_weight=0.5, fts_weight=0.3, importance_weight=0.2,
        )
        return [
            {
                "collection": "mnemosyne_beam",
                "score": r.get("score", 0.5),
                "importance": r.get("importance", 0.5),
                "fused": r.get("score", 0.5) * r.get("importance", 0.5),
                "content": r.get("content", ""),
            }
            for r in results
            if r.get("importance", 0) >= MIN_IMPORTANCE
        ]
    except Exception:
        return []


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_rules_summary() -> str:
    rules_dir = Path(RULES_DIR)
    if not rules_dir.exists():
        return ""
    parts, total = [], 0
    for f in sorted(rules_dir.glob("*.md")):
        content = f.read_text(encoding="utf-8", errors="replace")
        headers = [ln.lstrip("# ").strip() for ln in content.splitlines()
                   if ln.startswith("## ")]
        if headers:
            entry = f"[rules/{f.name}]: {', '.join(headers)}"
            if total + len(entry) < RULES_MAX_CHARS:
                parts.append(entry)
                total += len(entry)
    return ("ACTIVE RULES FILES:\n" + "\n".join(parts)) if parts else ""


def _keyword_rerank(hits, query):
    query_tokens = set(re.sub(r"[^a-z0-9]", " ", query.lower()).split())
    if len(query_tokens) < 2: return hits
    for h in hits:
        content_lower = (h.get("content","") or "").lower()
        overlap = sum(1 for t in query_tokens if len(t) > 3 and t in content_lower)
        h["fused"] = h.get("fused", 0.0) + min(0.25, overlap * 0.05)
    return sorted(hits, key=lambda h: h["fused"], reverse=True)


_CONF_TRUST  = {"high": 1.0, "medium": 0.7, "low": 0.4}
_TYPE_WEIGHT = {"observed": 1.0, "inferred": 0.8, "assumed": 0.6, "gap": 0.4}
_LN2 = math.log(2)


def _multi_signal_score(hit: dict, now_ts: float) -> float:
    """Score a hit on four axes plus stigmergic pheromone boost."""
    payload = hit.get("payload") or {}

    relevance = hit.get("fused", hit["score"] * hit.get("importance", 0.5))

    created = payload.get("created_at_ts") or payload.get("ts_epoch")
    if created:
        age_days = max(0.0, (now_ts - float(created)) / 86400.0)
        recency = math.exp(-_LN2 * age_days / RECENCY_HALFLIFE_DAYS)
    else:
        recency = 0.5  # unknown age → neutral

    conf  = str(payload.get("confidence", "") or "").lower()
    trust = _CONF_TRUST.get(conf, 0.6)

    rtype  = str(payload.get("record_type") or payload.get("type", "") or "").lower()
    type_w = _TYPE_WEIGHT.get(rtype, 0.7)

    base = (RANKER_W_RELEVANCE * relevance
            + RANKER_W_RECENCY * recency
            + RANKER_W_TRUST   * trust
            + RANKER_W_TYPE    * type_w)

    # Stigmergic boost: paths used before get a mild reinforcement advantage.
    # Evaporation ensures cold paths are not permanently suppressed.
    phero = _effective_pheromone(payload, now_ts)
    return base + PHERO_BETA * math.log1p(phero)


def _mmr_select(hits: list[dict], top_k: int) -> list[dict]:
    """Maximal Marginal Relevance selection with ε-exploration.

    Uses token-overlap as a cheap text-similarity proxy (no re-embedding).
    ε-exploration: with probability PHERO_EPSILON, the final slot is filled
    with a random non-selected hit rather than the MMR winner — prevents
    pheromone-reinforced recall from locking into the same top-K forever.
    """
    import random as _random

    if len(hits) <= top_k:
        return hits

    def _tok_sim(a: dict, b: dict) -> float:
        ta = set((a.get("content") or "").lower().split())
        tb = set((b.get("content") or "").lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(len(ta), len(tb))

    selected: list[dict] = []
    remaining = list(hits)
    while remaining and len(selected) < top_k:
        # ε-exploration on the last slot: occasionally surface a cold hit.
        if (len(selected) == top_k - 1
                and len(remaining) > 1
                and _random.random() < PHERO_EPSILON):
            # Pick a random hit NOT by score — diversify the recall pool.
            best = _random.choice(remaining)
        elif not selected:
            best = max(remaining, key=lambda h: h["_ms_score"])
        else:
            def _mmr(h):
                rel = h["_ms_score"]
                max_sim = max(_tok_sim(h, s) for s in selected)
                return MMR_LAMBDA * rel - (1.0 - MMR_LAMBDA) * max_sim
            best = max(remaining, key=_mmr)
        selected.append(best)
        remaining.remove(best)
    return selected


def _effective_pheromone(payload: dict, now_ts: float) -> float:
    """Pheromone after exponential evaporation since last reinforcement."""
    stored = float(payload.get("pheromone", 0.0) or 0.0)
    if stored <= 0.0:
        return 0.0
    reinforced = payload.get("pheromone_reinforced_ts")
    if not reinforced:
        return stored
    age_h = max(0.0, (now_ts - float(reinforced)) / 3600.0)
    return stored * math.exp(-_LN2 * age_h / PHERO_HALFLIFE_H)


def _pheromone_deposit(collection: str, point_id, current_phero: float, now_ts: float) -> None:
    """Bump pheromone on a retrieved point via Qdrant set_payload. Fire-and-forget."""
    if not point_id:
        return
    new_phero = current_phero + PHERO_DEPOSIT
    url = f"{QDRANT_URL}/collections/{collection}/points/payload"
    body = json.dumps({
        "points": [point_id],
        "payload": {
            "pheromone": new_phero,
            "pheromone_reinforced_ts": now_ts,
        },
    }).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", "api-key": QDRANT_KEY},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=0.5).close()
    except Exception:
        pass  # pheromone deposit is always best-effort


def _extract_intent(prompt: str) -> str:
    meta = re.compile(
        r"^(can you|please|help me|could you|I need you to|I want you to"
        r"|I'd like you to|would you)\s+",
        re.IGNORECASE,
    )
    return meta.sub("", prompt.strip())[:200].strip()


def _clean_content(content: str) -> str:
    stripped = content.strip()
    if not (stripped.startswith("[{") or stripped.startswith('["')):
        return stripped
    try:
        obj = json.loads(stripped)
        if isinstance(obj, list):
            parts = []
            for msg in obj:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "")
                c = msg.get("content") or ""
                if role == "user" and isinstance(c, str) and c.strip():
                    parts.append(f"user: {c.strip()[:150]}")
                elif role == "assistant" and isinstance(c, str) and c.strip():
                    parts.append(f"assistant: {c.strip()[:150]}")
            return " | ".join(parts[:2]) if parts else ""
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r'"role"\s*:\s*"user"[^}]*"content"\s*:\s*"([^"]{10,})"', stripped)
    return f"user: {m.group(1)[:200]}" if m else ""


def _format_results(hits: list[dict], query: str, used_fallback: bool) -> str:
    lines = []
    seen = set()
    for h in hits:
        content = _clean_content(h["content"])
        if not content:
            continue
        key = content[:80]
        if key in seen:
            continue
        seen.add(key)
        if len(content) > MAX_CONTENT_CHARS:
            content = content[:MAX_CONTENT_CHARS].rstrip() + "..."
        col = h["collection"].replace("_", "-")
        score = h["score"]
        lines.append(f"[{col}|{score:.2f}] {content}")
        if len(lines) >= RECALL_TOP_K:
            break

    if not lines:
        return GROUNDING_DIRECTIVE

    src = "BeamMemory fallback" if used_fallback else f"{len(COLLECTIONS)} Qdrant collections"
    header = f'MEMORY MATCH ({len(lines)} results from {src}) for "{query}":'
    body = "\n".join(lines)
    footer = (
        "Use mcp_mnemosyne_mnemosyne_recall for full content if relevant. "
        "Disclose recalled context to the user if it changes your answer."
    )
    return f"{header}\n{body}\n{footer}"


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        sys.exit(0)

    if payload.get("hook_event_name", "") not in (
        "pre_llm_call", "PreLlmCall",       # legacy Hermes names
        "UserPromptSubmit", "SubagentStart", # Claude Code native names
    ):
        sys.exit(0)

    extra = payload.get("extra") or {}

    # Detect subagent context — triggers lightweight path, NOT a skip.
    # Previous behaviour (sys.exit) left all workflow subagents ungrounded;
    # that is where cross-file mistakes (wrong import names, renamed fields) occur.
    task_id = extra.get("task_id") or payload.get("session_id") or ""
    is_subagent = "subagent" in task_id.lower() or bool(os.environ.get("HERMES_SUBAGENT"))

    user_message = extra.get("user_message") or ""
    if not isinstance(user_message, str):
        user_message = ""
    msg_stripped = user_message.strip()

    # Skip only for truly empty messages.
    if not msg_stripped:
        sys.exit(0)

    # Narrow slash-command skip: navigation-only commands need no grounding.
    # Code-affecting commands (/fix, /review, /remember, /code-review, etc.) get grounded.
    if msg_stripped.startswith("/"):
        cmd = msg_stripped.split()[0].lower()
        if cmd in NAVIGATION_SLASH_COMMANDS:
            sys.exit(0)
        # Fall through: non-navigation slash command → continue to grounding.

    # Short-message length guard removed: "fix the bug" is 13 chars but needs grounding.

    intent = _extract_intent(msg_stripped)

    # ── Lightweight subagent path ─────────────────────────────────────────────
    # Skips the 7-collection fan-out, session history, SA enrichment.
    # Still grounds on hermes_memory (investigation findings) — the collection
    # most relevant to code-generation tasks in an active investigation.
    if is_subagent:
        sub_hits: list[dict] = []
        if QDRANT_URL:
            sub_vec = _embed(intent)
            if sub_vec is not None:
                sub_hits = _search_collection(
                    "hermes_memory", sub_vec, "text", "confidence", True,
                    top_k=min(RECALL_TOP_K, 2),
                )
                sub_hits = [h for h in sub_hits if h["importance"] >= MIN_IMPORTANCE]
                _sub_ts = time.time()
                for h in sub_hits:
                    h["_ms_score"] = _multi_signal_score(h, _sub_ts)
                sub_hits = sorted(sub_hits, key=lambda h: h["_ms_score"], reverse=True)[:2]
            else:
                sub_hits = _beam_fallback(intent)

        if sub_hits:
            recall_block = _format_results(sub_hits, intent, False)
        else:
            recall_block = (
                "[WARNING: memory grounding unavailable in subagent context — "
                "Qdrant unreachable. Verify cross-file references by reading "
                "the actual files before using any name, field, or import path.]\n\n"
                + GROUNDING_DIRECTIVE
            )
        rules_summary = _load_rules_summary()
        context = (recall_block + "\n\n" + rules_summary) if rules_summary else recall_block
        print(json.dumps({"context": context}))
        return

    # ── v3: embed → parallel Qdrant fan-out (main session) ───────────────────
    vector = _embed(intent)
    used_fallback = False

    if vector is None:
        # Ollama unavailable — fall back to BeamMemory (v2 path)
        hits = _beam_fallback(intent)
        used_fallback = True
        if not hits:
            # Both Ollama and BeamMemory failed — memory is dark this turn.
            # Inject a visible warning rather than silently proceeding.
            warning = (
                "[WARNING: memory grounding UNAVAILABLE this turn — Ollama unreachable "
                "and BeamMemory fallback also failed. Do NOT rely on parametric memory "
                "for cross-file references, API names, field names, or import paths. "
                "Read the relevant files before using any name.]\n\n"
                + GROUNDING_DIRECTIVE
            )
            rules_summary = _load_rules_summary()
            context = (warning + "\n\n" + rules_summary) if rules_summary else warning
            print(json.dumps({"context": context}))
            return
    else:
        all_hits: list[dict] = []
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {
                ex.submit(
                    _search_collection,
                    col, vector, cf, impf, named, RECALL_TOP_K
                ): col
                for col, cf, impf, named in COLLECTIONS
            }
            for f in as_completed(futures):
                try:
                    all_hits.extend(f.result())
                except Exception:
                    pass

        # Keyword rerank adds overlap bonus to fused before multi-signal scoring.
        hits = _keyword_rerank(all_hits, intent)
        # Filter by min importance before scoring.
        hits = [h for h in hits if h["importance"] >= MIN_IMPORTANCE]
        # Multi-signal score: relevance + recency + source trust + record type.
        _now_ts = time.time()
        for h in hits:
            h["_ms_score"] = _multi_signal_score(h, _now_ts)
        # MMR selection: top RECALL_TOP_K with diversity balancing.
        hits = sorted(hits, key=lambda h: h["_ms_score"], reverse=True)
        hits = _mmr_select(hits, RECALL_TOP_K)

        # Pheromone deposit on selected hits (fire-and-forget, 0.5s timeout).
        # Only hermes_memory collection points have mutable payloads we own.
        for _h in hits:
            if _h.get("collection") in ("hermes_memory",) and _h.get("point_id"):
                _phero_now = _now_ts
                _current   = _effective_pheromone(_h.get("payload") or {}, _phero_now)
                _pheromone_deposit(_h["collection"], _h["point_id"], _current, _phero_now)

        # ── Optional spreading activation enrichment ──────────────────────
        # Seeds: mnemosyne collection hits that carry mnemosyne_id in payload.
        # SA discovers associatively-linked memories that vector search missed.
        if SA_ENABLED and _SA_MODULE is not None:
            try:
                import time as _time
                _sa_t0 = _time.monotonic()
                _sa_seeds = {}
                for _h in hits:
                    _mid = (_h.get("payload") or {}).get("mnemosyne_id")
                    if _mid and _h["collection"] == "mnemosyne":
                        _sa_seeds[str(_mid)] = _h["score"]
                if _sa_seeds:
                    _db = os.environ.get("MNEMOSYNE_DB",
                                         os.path.expanduser("~/.hermes/mnemosyne/data/mnemosyne.db"))
                    _sa_results = _SA_MODULE.run_spreading_activation(
                        db_path=_db,
                        seed_ids=list(_sa_seeds.keys()),
                        seed_scores=_sa_seeds,
                        max_results=2,
                    )
                    _sa_ms = (_time.monotonic() - _sa_t0) * 1000
                    if _sa_ms <= SA_TIMEOUT_MS:
                        for _r in _sa_results:
                            if _r.get("content"):
                                hits.append({
                                    "collection": "mnemosyne_sa",
                                    "score": _r["activation"],
                                    "importance": _r.get("importance", 0.5),
                                    "fused": _r["activation"] * _r.get("importance", 0.5),
                                    "content": _r["content"],
                                    "payload": {},
                                })
            except Exception:
                pass

    if hits:
        recall_block = _format_results(hits, intent, used_fallback)
    else:
        recall_block = GROUNDING_DIRECTIVE

    rules_summary = _load_rules_summary()
    context = (recall_block + "\n\n" + rules_summary) if rules_summary else recall_block

    print(json.dumps({"context": context}))


if __name__ == "__main__":
    main()
