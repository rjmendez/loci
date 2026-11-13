#!/usr/bin/env python3
"""pre_llm_call / UserPromptSubmit / SubagentStart hook: Qdrant recall grounding.

Embeds the prompt, searches the configured Qdrant collections in parallel,
ranks and injects a MEMORY MATCH block plus a rules index.
Env: QDRANT_URL, QDRANT_API_KEY, OLLAMA_BASE_URL / MNEMOSYNE_EMBEDDING_API_URL,
GROUNDING_EXTRA_COLLECTIONS, GROUNDING_DIRECTIVE, HOOK_* tuning knobs.
Fail-open: embed/Qdrant outages degrade to BeamMemory, then to a warning + directive.
"""
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

# Accept legacy HERMES_* env names (legacy_env.py is a sibling in both layouts).
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from legacy_env import apply as _apply_legacy_env
    _apply_legacy_env()
except Exception:
    pass

# Spreading activation enrichment (SA-RAG, arxiv 2512.15922).
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

_HERMES_HOME   = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
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

QDRANT_URL  = os.environ.get("QDRANT_URL")
QDRANT_KEY   = os.environ.get("QDRANT_API_KEY", "")
def _embed_base_url() -> Optional[str]:
    """OLLAMA_BASE_URL is a bare host (needs /v1); MNEMOSYNE_EMBEDDING_API_URL is already a full /v1 endpoint."""
    base = os.environ.get("OLLAMA_BASE_URL")
    if base:
        return f"{base.rstrip('/')}/v1"
    api = os.environ.get("MNEMOSYNE_EMBEDDING_API_URL")
    return api.rstrip("/") if api else None


OLLAMA_URL   = _embed_base_url()
EMBED_MODEL  = os.environ.get("MNEMOSYNE_EMBEDDING_MODEL",   "nomic-embed-text")
_EMBED_API_KEY        = os.environ.get("EMBED_API_KEY", "")
_EMBED_API_KEY_HEADER = os.environ.get("EMBED_API_KEY_HEADER", "Authorization")

RULES_DIR    = os.environ.get("HOOK_RULES_DIR",
                              os.path.join(_PROFILE_DIR, "rules"))

RECALL_TOP_K      = int(os.environ.get("HOOK_RECALL_TOP_K",          "3"))
MIN_IMPORTANCE    = float(os.environ.get("HOOK_RECALL_MIN_IMPORTANCE", "0.2"))
MIN_SCORE         = float(os.environ.get("HOOK_RECALL_MIN_SCORE",      "0.55"))
# Every named collection here uses "dense".
VECTOR_NAME       = os.environ.get("HOOK_RECALL_VECTOR_NAME",    "dense")
MAX_CONTENT_CHARS = int(os.environ.get("HOOK_RECALL_MAX_CHARS",        "200"))
RULES_MAX_CHARS   = int(os.environ.get("HOOK_RULES_MAX_CHARS",         "1200"))
EMBED_TIMEOUT     = float(os.environ.get("HOOK_EMBED_TIMEOUT",         "3.0"))
QDRANT_TIMEOUT    = float(os.environ.get("HOOK_QDRANT_TIMEOUT",        "2.0"))
WORKERS           = int(os.environ.get("HOOK_QDRANT_WORKERS",          "8"))

# Must sum to 1.0.
RANKER_W_RELEVANCE = float(os.environ.get("HOOK_RANKER_W_RELEVANCE", "0.50"))
RANKER_W_RECENCY   = float(os.environ.get("HOOK_RANKER_W_RECENCY",   "0.20"))
RANKER_W_TRUST     = float(os.environ.get("HOOK_RANKER_W_TRUST",     "0.15"))
RANKER_W_TYPE      = float(os.environ.get("HOOK_RANKER_W_TYPE",      "0.15"))
# Recency half-life in days.
RECENCY_HALFLIFE_DAYS = float(os.environ.get("HOOK_RECENCY_HALFLIFE_DAYS", "7"))
# MMR lambda: 1.0 = pure relevance, 0.0 = pure diversity.
MMR_LAMBDA = float(os.environ.get("HOOK_MMR_LAMBDA", "0.75"))

# Stigmergic recall: retrieval deposits pheromone; decay prevents lock-in.
PHERO_BETA         = float(os.environ.get("HOOK_PHERO_BETA",        "0.08"))  # score boost coefficient
PHERO_HALFLIFE_H   = float(os.environ.get("HOOK_PHERO_HALFLIFE_H",  "24"))    # pheromone half-life in hours
PHERO_DEPOSIT      = float(os.environ.get("HOOK_PHERO_DEPOSIT",     "1.0"))   # amount deposited per retrieval
PHERO_EPSILON      = float(os.environ.get("HOOK_PHERO_EPSILON",     "0.05"))  # ε-exploration probability

# Project-specific collections are added via GROUNDING_EXTRA_COLLECTIONS=name1,name2.
_EXTRA_FIELD_MAP: dict[str, tuple] = {
    "ecc_skills":        ("content_preview", None, True),
    "agent_core_chunks": ("text",            None, False),
    "dama_gotchi_code":  ("text",            None, True),
}
_DEFAULT_EXTRA_FIELDS = ("text", None, True)

# fmt: (collection, content_field, importance_field_or_None, use_named_vector)
_BASE_COLLECTIONS = [
    ("mnemosyne",       "content",         "importance", True),
    ("loci_sessions", "content_preview", None,         True),
    # numeric_confidence, not confidence: the latter is a 'high'/'medium'/'low'
    # label (read by _multi_signal_score); float()ing it raised on every prompt.
    ("loci_memory",   "text",            "numeric_confidence", True),
]
_extra_names = [
    c.strip() for c in os.environ.get("GROUNDING_EXTRA_COLLECTIONS", "").split(",")
    if c.strip()
]
COLLECTIONS = _BASE_COLLECTIONS + [
    (name,) + _EXTRA_FIELD_MAP.get(name, _DEFAULT_EXTRA_FIELDS)
    for name in _extra_names
]

GROUNDING_DIRECTIVE = os.environ.get("GROUNDING_DIRECTIVE") or (
    "[GROUNDING DIRECTIVE — active every turn]\n"
    "Before answering from parametric knowledge:\n"
    "1. RECALL from memory: use the mnemosyne recall tool for the topic\n"
    "2. SEARCH: use available code search and artifact tools\n"
    "3. WEB SEARCH: for current events, versions, external documentation\n"
    "4. SESSION SEARCH: for 'what did we do about X' questions\n"
    "Skip recall only when the answer comes directly from tool output in this turn."
)

# Navigation only; every other slash command is grounded because it may touch code.
NAVIGATION_SLASH_COMMANDS: frozenset[str] = frozenset({
    "/help", "/clear", "/compact", "/cost", "/status",
    "/history", "/ide", "/doctor", "/login", "/logout",
})


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
    """Embed via OpenAI-compat /v1/embeddings; None on any failure."""
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


class _SearchFailed(Exception):
    """A collection search failed. Raised (not []) so an outage is distinguishable
    from 'nothing matched': a wrong named-vector shape once returned [] silently for months."""


def _search_collection(
    collection: str,
    vector: list[float],
    content_field: str,
    importance_field: Optional[str],
    named_vec: bool,
    top_k: int = 5,
) -> list[dict]:
    """Search one collection; raises _SearchFailed on transport/HTTP error."""
    url = f"{QDRANT_URL}/collections/{collection}/points/search"
    vec_payload = {"name": VECTOR_NAME, "vector": vector} if named_vec else vector
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
        raise _SearchFailed(collection)

    hits = []
    for pt in d.get("result") or []:   # error envelopes carry "result": null
        score = pt.get("score", 0.0)
        if score < MIN_SCORE:
            continue
        payload = pt.get("payload") or {}
        content = payload.get(content_field, "")
        if not content:
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
            "point_id": pt.get("id"),
            "score": score,
            "importance": importance,
            "fused": score * importance,
            "content": str(content),
            "payload": payload,
        })
    return hits


def _beam_fallback(query: str) -> list[dict]:
    """BeamMemory recall, used only when embedding or every Qdrant search failed."""
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
    """Relevance, recency, trust and type, plus a pheromone boost."""
    payload = hit.get("payload") or {}

    relevance = (hit["fused"] if "fused" in hit
                 else hit["score"] * hit.get("importance", 0.5))

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

    phero = _effective_pheromone(payload, now_ts)
    return base + PHERO_BETA * math.log1p(phero)


def _mmr_select(hits: list[dict], top_k: int) -> list[dict]:
    """MMR selection (token-overlap similarity, no re-embedding). With probability
    PHERO_EPSILON the last slot is a random hit, so pheromone reinforcement
    cannot lock recall into one top-K."""
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
        if (len(selected) == top_k - 1
                and len(remaining) > 1
                and _random.random() < PHERO_EPSILON):
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
    """Pheromone after exponential decay since last reinforcement."""
    stored = float(payload.get("pheromone", 0.0) or 0.0)
    if stored <= 0.0:
        return 0.0
    reinforced = payload.get("pheromone_reinforced_ts")
    if not reinforced:
        return stored
    age_h = max(0.0, (now_ts - float(reinforced)) / 3600.0)
    return stored * math.exp(-_LN2 * age_h / PHERO_HALFLIFE_H)


def _pheromone_deposit(collection: str, point_id, current_phero: float, now_ts: float) -> None:
    """Bump pheromone on a retrieved point via set_payload; best-effort."""
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
        pass


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
            if not any(isinstance(msg, dict) and "role" in msg for msg in obj):
                return stripped     # structured content, not a chat transcript
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
    if '"role"' not in stripped:
        return stripped             # broken JSON that is not a transcript either
    m = re.search(r'"role"\s*:\s*"user"[^}]*"content"\s*:\s*"([^"]{10,})"', stripped)
    return f"user: {m.group(1)[:200]}" if m else ""


def _format_results(hits: list[dict], query: str, used_fallback: bool,
                    unreachable: int = 0, searched: int | None = None) -> str:
    """Render the recall block. The header states real coverage (``unreachable``
    failed collections) because the model trusts it as "what memory holds".
    ``searched`` overrides len(COLLECTIONS) for subset callers (the subagent path)."""
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

    total = len(COLLECTIONS) if searched is None else searched
    if used_fallback:
        src = "BeamMemory fallback"
    elif unreachable:
        reached = max(total - unreachable, 0)
        src = (f"{reached} of {total} Qdrant collections "
               f"({unreachable} unreachable)")
    else:
        src = f"{total} Qdrant collection" + ("" if total == 1 else "s")
    header = f'MEMORY MATCH ({len(lines)} results from {src}) for "{query}":'
    body = "\n".join(lines)
    footer = (
        "Use mcp_mnemosyne_mnemosyne_recall for full content if relevant. "
        "Disclose recalled context to the user if it changes your answer."
    )
    return f"{header}\n{body}\n{footer}"


# Wire format depends on the calling host. Claude Code reads only
# hookSpecificOutput.additionalContext for UserPromptSubmit/SubagentStart (top-level
# "context" and plain stdout are ignored; 10,000-char cap). Hermes reads {"context": ...}.
_HOST_EVENT = ""
_CLAUDE_CONTEXT_EVENTS = ("UserPromptSubmit", "SubagentStart")
_CLAUDE_CONTEXT_MAX = 9500


def _emit_context(context: str) -> None:
    """Print the context block in the invoking host's format."""
    if _HOST_EVENT in _CLAUDE_CONTEXT_EVENTS:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": _HOST_EVENT,
            "additionalContext": context[:_CLAUDE_CONTEXT_MAX],
        }}))
    else:
        print(json.dumps({"context": context}))


def main() -> None:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        sys.exit(0)
    if not isinstance(payload, dict):
        sys.exit(0)

    if payload.get("hook_event_name", "") not in (
        "pre_llm_call", "PreLlmCall",       # legacy Hermes names
        "UserPromptSubmit", "SubagentStart", # Claude Code native names
    ):
        sys.exit(0)
    global _HOST_EVENT
    _HOST_EVENT = payload.get("hook_event_name", "")

    extra = payload.get("extra") or {}

    # Lightweight path, not a skip: skipping would leave workflow subagents ungrounded.
    task_id = extra.get("task_id") or payload.get("session_id") or ""
    is_subagent = "subagent" in task_id.lower() or bool(os.environ.get("HERMES_SUBAGENT"))

    # Claude Code puts the text at the top level; Hermes nested it under extra.user_message.
    user_message = payload.get("prompt") or extra.get("user_message") or ""
    if not isinstance(user_message, str):
        user_message = ""
    msg_stripped = user_message.strip()

    if not msg_stripped:
        sys.exit(0)

    if msg_stripped.startswith("/"):
        cmd = msg_stripped.split()[0].lower()
        if cmd in NAVIGATION_SLASH_COMMANDS:
            sys.exit(0)

    # No length guard: "fix the bug" is 13 chars and still needs grounding.
    intent = _extract_intent(msg_stripped)

    # Subagents search loci_memory only (most relevant mid-investigation).
    if is_subagent:
        sub_hits: list[dict] = []
        sub_used_fallback = False
        if QDRANT_URL:
            sub_vec = _embed(intent)
            if sub_vec is not None:
                try:
                    sub_hits = _search_collection(
                        "loci_memory", sub_vec, "text", "numeric_confidence", True,
                        top_k=min(RECALL_TOP_K, 2),
                    )
                except _SearchFailed:
                    sub_hits = _beam_fallback(intent)
                    sub_used_fallback = True
                sub_hits = [h for h in sub_hits if h["importance"] >= MIN_IMPORTANCE]
                _sub_ts = time.time()
                for h in sub_hits:
                    h["_ms_score"] = _multi_signal_score(h, _sub_ts)
                sub_hits = sorted(sub_hits, key=lambda h: h["_ms_score"], reverse=True)[:2]
            else:
                sub_hits = _beam_fallback(intent)
                sub_used_fallback = True

        if sub_hits:
            recall_block = _format_results(sub_hits, intent, sub_used_fallback,
                                           searched=1)
        elif QDRANT_URL and not sub_used_fallback:
            # The search answered and nothing matched: a genuine miss, not an outage.
            recall_block = GROUNDING_DIRECTIVE
        else:
            recall_block = (
                "[WARNING: memory grounding unavailable in subagent context — "
                "Qdrant unreachable. Verify cross-file references by reading "
                "the actual files before using any name, field, or import path.]\n\n"
                + GROUNDING_DIRECTIVE
            )
        rules_summary = _load_rules_summary()
        context = (recall_block + "\n\n" + rules_summary) if rules_summary else recall_block
        _emit_context(context)
        return

    vector = _embed(intent)
    used_fallback = False
    failed = 0

    if vector is None:
        hits = _beam_fallback(intent)
        used_fallback = True
        if not hits:
            # Memory is dark this turn: warn visibly rather than proceeding ungrounded.
            warning = (
                "[WARNING: memory grounding UNAVAILABLE this turn — Ollama unreachable "
                "and BeamMemory fallback also failed. Do NOT rely on parametric memory "
                "for cross-file references, API names, field names, or import paths. "
                "Read the relevant files before using any name.]\n\n"
                + GROUNDING_DIRECTIVE
            )
            rules_summary = _load_rules_summary()
            context = (warning + "\n\n" + rules_summary) if rules_summary else warning
            _emit_context(context)
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
            searched = 0
            for f in as_completed(futures):
                searched += 1
                try:
                    all_hits.extend(f.result())
                except _SearchFailed:
                    failed += 1
                except Exception:
                    failed += 1
        if searched and failed == searched:
            used_fallback = True
            all_hits = _beam_fallback(intent)

        hits = _keyword_rerank(all_hits, intent)
        hits = [h for h in hits if h["importance"] >= MIN_IMPORTANCE]
        _now_ts = time.time()
        for h in hits:
            h["_ms_score"] = _multi_signal_score(h, _now_ts)
        hits = sorted(hits, key=lambda h: h["_ms_score"], reverse=True)

        # Spreading activation adds associatively-linked memories. It must join the
        # pool BEFORE top-K selection, or a full vector result set cuts it off.
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
                        _sa_hits = []
                        for _r in _sa_results:
                            if _r.get("content"):
                                _sa_hit = {
                                    "collection": "mnemosyne_sa",
                                    "score": _r["activation"],
                                    "importance": _r.get("importance", 0.5),
                                    "fused": _r["activation"] * _r.get("importance", 0.5),
                                    "content": _r["content"],
                                    "payload": {},
                                }
                                _sa_hit["_ms_score"] = _multi_signal_score(_sa_hit, _now_ts)
                                _sa_hits.append(_sa_hit)
                        if _sa_hits:
                            hits = sorted(hits + _sa_hits,
                                          key=lambda h: h["_ms_score"], reverse=True)
            except Exception:
                pass

        hits = _mmr_select(hits, RECALL_TOP_K)

        # Only loci_memory payloads are ours to mutate.
        for _h in hits:
            if _h.get("collection") in ("loci_memory",) and _h.get("point_id"):
                _phero_now = _now_ts
                _current   = _effective_pheromone(_h.get("payload") or {}, _phero_now)
                _pheromone_deposit(_h["collection"], _h["point_id"], _current, _phero_now)

    if hits:
        recall_block = _format_results(hits, intent, used_fallback, failed)
    elif failed:
        # Empty because part of memory was dark, not a genuine miss: say so.
        recall_block = (
            f"[WARNING: memory grounding INCOMPLETE this turn — {failed} of "
            f"{len(COLLECTIONS)} Qdrant collections were unreachable and no result "
            "came back. Treat this as 'memory was not searched', not as 'memory "
            "holds nothing'. Read the relevant files before using any name.]\n\n"
            + GROUNDING_DIRECTIVE
        )
    else:
        recall_block = GROUNDING_DIRECTIVE

    rules_summary = _load_rules_summary()
    context = (recall_block + "\n\n" + rules_summary) if rules_summary else recall_block

    _emit_context(context)


if __name__ == "__main__":
    main()
