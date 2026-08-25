#!/usr/bin/env python3
"""
loci-mcp — persistent memory and knowledge layer for AI agent investigations.

Provides a manifest-first memory layer for tracking findings, observations,
inferences, assumptions, and tool audit logs across investigation sessions.
Mnemosyne is optional and treated as the primary shared memory substrate when
installed. Qdrant is optional and used as a secondary semantic/hybrid index.
JSONL files remain the durable local storage and final fallback path.

Storage layout:
    $HERMES_MEMORY_DIR/<investigation_id>/
        manifest.json     — structured investigation state
        findings.jsonl    — append-only finding log
        audit.jsonl       — tool call/response audit log

    $HERMES_MEMORY_DIR/../audit/
        YYYY-MM-DD.jsonl  — global cross-investigation audit log

Tools:
    investigation_start          — create or resume an investigation
    investigation_load           — retrieve manifest + recent findings (context recovery)
    investigation_store          — record an observation, inference, assumption, or gap
    investigation_note           — update manifest fields (hypothesis, next_step, questions)
    investigation_reflect        — synthesize current investigation state (+ entity frequency)
    investigation_search         — search findings by keyword or semantics (hybrid + reranked)
    investigation_pre_answer_check — validate response claims against stored evidence
    investigation_evidence_precheck — detect likely duplicate queries/claims
    investigation_entity_lookup  — find all findings mentioning a specific IP/email/hostname/hash/CVE
    investigation_related_cases  — find prior investigations that dealt with the same entities
    investigation_finding_provenance — trace a finding back to its root observed evidence
    investigation_list           — list all investigations
    audit_log                    — record a tool call/response pair (post-call hook)
    memory_self_check            — provenance + contradiction self-check on investigation findings
    memory_retract               — soft-tombstone a hallucinated finding + its derived lineage
    memory_restore               — undo a retraction
    memory_health                — substrate self-check (qdrant / embedders / mirror / integrity)
    code_memory_correlate        — link code-hallucination flags to contaminated investigation findings
    reflection_loop_seed         — enqueue Copilot artifacts for bounded self-reflection
    reflection_loop_tick         — process small queued batches and store findings
    reflection_loop_status       — inspect reflection queue and aggregate loop stats
"""

from __future__ import annotations

import asyncio  # noqa: F401  (kept on the module namespace; live users import it function-locally)
import fcntl
import hashlib
import hmac
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
import uuid
import weakref
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# memcheck (the shared verdict core) lives alongside this file. Ensure its
# parent dir is importable whether server.py is run as __main__ (spawned by
# path) or loaded via importlib in tests under a synthetic module name.
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from memcheck.checks import (  # noqa: E402
    find_contamination,
    run_code_checks,
    run_contradiction,
    run_provenance,
)
from memcheck.verdict import make_signature, new_verdict, redact_excerpt  # noqa: E402

# ---------------------------------------------------------------------------
# Optional IOC extraction helpers
# ---------------------------------------------------------------------------

_HAS_IOCEXTRACT = False
_HAS_CY_IOC = False
try:
    import iocextract as _iocextract
    _HAS_IOCEXTRACT = True
except ImportError:
    pass
try:
    import cy_ioc_extract as _cy_ioc
    _HAS_CY_IOC = True
except ImportError:
    pass

_EMAIL_RE = re.compile(r'\b[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}\b', re.I)

# ── Immutable event log (fail-open) ───────────────────────────────────────────
# Appends a record before every memory mutation so the full operation history
# is preserved independent of the live SQLite/Qdrant store (SSGM + AgentCore pattern).
def _event_log_append(event: dict) -> None:
    """Write one event to the immutable append-only event log. Fail-open."""
    try:
        import sys as _sys
        _scripts = str(Path(__file__).resolve().parent.parent / "scripts")
        if _scripts not in _sys.path:
            _sys.path.insert(0, _scripts)
        from event_log import append as _el_append
        _el_append(event)
    except Exception as exc:
        logger.debug("event log append failed (fail-open): %r", exc)
        # Never let event log failures block memory operations
_HOST_RE = re.compile(
    r'\b[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?'
    r'(?:\.[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?)*'
    r'(?:\.(?:local|corp|internal|lan|dev|test|net|com|io|org))\b', re.I
)
_URL_RE = re.compile(r'https?://[^\s"\' <>;]+', re.I)




def _extract_entities(text: str) -> dict:
    out: dict = {"ips": [], "hashes": [], "cves": [], "emails": [], "hostnames": [], "urls": []}
    if _HAS_CY_IOC:
        try:
            ioc = _cy_ioc.IOCEXtract(text).extract_ioc()
            out["ips"] = ioc.get("IP", [])
            # Normalise to lowercase so Qdrant keyword index lookups (which query
            # with .lower()) never miss a mixed-case value from the IOC extractor.
            out["hashes"] = [h.lower() for h in (ioc.get("SHA256") or [])]
            out["cves"]   = [c.lower() for c in (ioc.get("CVE") or [])]
        except Exception as exc:
            logger.debug("cy_ioc extract failed (fail-open): %r", exc)
    if _HAS_IOCEXTRACT and not out["ips"]:
        # fallback: iocextract handles defanged IPs (e.g. 198[.]51[.]100[.]1)
        try:
            out["ips"] = list(_iocextract.extract_ips(text))
        except Exception as exc:
            logger.debug("iocextract extract_ips failed (fail-open): %r", exc)
    out["emails"]    = [e.lower() for e in _EMAIL_RE.findall(text)]
    out["hostnames"] = list({m.lower() for m in _HOST_RE.findall(text)})
    out["urls"] = [u for u in _URL_RE.findall(text)]
    return out

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("loci-mcp")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MEMORY_DIR = Path(os.environ.get(
    "HERMES_MEMORY_DIR",
    Path.home() / ".hermes" / "memory-sessions",
))
# Embedding + Qdrant cluster lives in qdrant_ops.py. Imported here (after
# load_dotenv above) so its import-time os.environ reads see the same env.
# Re-exported so `server.<helper>()` keeps working for the rest of this module,
# for in-process callers, and for tests (which patch e.g. server._get_qdrant).
import qdrant_ops  # noqa: E402,F401
from qdrant_ops import (  # noqa: E402,F401
    QDRANT_COLLECTION_PREFIX, VECTOR_DIM,
    _get_sparse_embedder, _get_cross_encoder, _embed_sparse, _create_payload_indexes,
    _purge_old_records, _get_qdrant, _embed_auth_headers, _embed, _qdrant_upsert,
    _qdrant_degraded_mode, _ce_rerank, _qdrant_similarity_search, _qdrant_search_collection,
    probe_collection,
)
from qdrant_ops import RERANK_MAX_CHARS as _RERANK_MAX_CHARS  # noqa: E402
REFLECTION_STATE_DIR = MEMORY_DIR / "_reflection-loop"
REFLECTION_STATE_FILE = REFLECTION_STATE_DIR / "state.json"
REFLECTION_DEFAULT_INVESTIGATION = os.environ.get(
    "HERMES_REFLECTION_INVESTIGATION",
    "copilot-self-reflection-loop",
)
REFLECTION_LOG_TAIL_MIN_FILE_BYTES = 1_000_000
REFLECTION_LOG_TAIL_READ_BYTES = 512_000
REFLECTION_SIGNATURE_OBSERVE_LIMIT = 3
REFLECTION_SIGNATURE_MAP_LIMIT = 500
AGENT_ID = os.environ.get("HERMES_AGENT_ID", "")

# ---------------------------------------------------------------------------
# Investigation storage layer (P2b of the Loci self-review split)
#
# The store lives in inv_store.py. Its memory root is injected as a LAMBDA closing
# over MEMORY_DIR above — not the Path value — so that rebinding server.MEMORY_DIR
# (which ~10 tests do, to a tmpdir) still redirects every store write. Registered
# here, immediately after MEMORY_DIR, so the helpers are bound before first use.
# ---------------------------------------------------------------------------

import inv_store  # noqa: E402
inv_store.register(lambda: MEMORY_DIR)
# Re-exported so `server.<helper>()` keeps working for the rest of this module,
# for in-process callers, and for tests (which patch e.g. server._append_jsonl).
from inv_store import (  # noqa: E402,F401
    _now, _inv_dir, _manifest_cache, _load_manifest, _save_manifest,
    _atomic_write_text, _append_jsonl, _read_jsonl, _finding_updates_path,
    _load_resolution_overrides, _load_retracted_ids, _make_ref, _tag_finding_ids,
    _summarise_finding, _safe_float, _CONFIDENCE_RANK, _RESOLUTION_STATES,
    _distinctive_entity_set,
)


# ---------------------------------------------------------------------------
# Optional Qdrant + fastembed
# ---------------------------------------------------------------------------

_investigation_locks: weakref.WeakValueDictionary = weakref.WeakValueDictionary()  # per-investigation lock; auto-evicted when caller releases
_investigation_locks_lock = threading.Lock()          # guards _investigation_locks dict itself


def _investigation_lock(investigation_id: str) -> threading.Lock:
    """Return the per-investigation lock, creating it under the dict-guard lock if absent.

    Returns the lock UNACQUIRED — callers are responsible for `with` on the result.
    Uses WeakValueDictionary so entries are evicted automatically once no caller
    holds a strong reference, preventing unbounded accumulation (#101).
    """
    with _investigation_locks_lock:
        lock = _investigation_locks.get(investigation_id)
        if lock is None:
            lock = threading.Lock()
            _investigation_locks[investigation_id] = lock
        return lock

# ---------------------------------------------------------------------------
# LadybugDB graph store (primary relationship/graph backend) — fail-open like Qdrant.
# Mirrors findings/entities/derivation into an embedded graph and backs the
# entity-lookup / related-cases / contamination / code-symbol paths. If ladybug or
# the module is unavailable, every consumer degrades to the pre-existing path.
# ---------------------------------------------------------------------------
_ladybug_store = None                     # LadybugStore singleton once initialized
_ladybug_failed = False                   # PERMANENT-failure latch (ladybug unimportable) — don't retry
_ladybug_last_attempt = 0.0               # monotonic ts of last TRANSIENT init failure
_LADYBUG_RETRY_SECONDS = 30               # backoff before retrying after a transient failure
_ladybug_lock = threading.Lock()


def _get_ladybug():
    """Lazy, fail-open LadybugStore singleton. Returns None if unavailable.

    Distinguishes a genuinely UNRECOVERABLE failure (ladybug is not importable) — which
    latches permanently so we stop retrying — from a TRANSIENT one (another process
    holds Kuzu's single-writer lock, or a transient IO error at open time), which does
    NOT latch: a later call retries after a short backoff so the code graph self-heals
    once the other writer releases the lock. Never raises.
    """
    global _ladybug_store, _ladybug_failed, _ladybug_last_attempt
    if _ladybug_store is not None:
        return _ladybug_store
    if _ladybug_failed:
        return None
    with _ladybug_lock:
        if _ladybug_store is not None:
            return _ladybug_store
        if _ladybug_failed:
            return None
        # Back off between transient-failure retries so we don't hammer a held lock.
        if _ladybug_last_attempt and (time.monotonic() - _ladybug_last_attempt) < _LADYBUG_RETRY_SECONDS:
            return None
        try:
            from graph import ladybug_store as _kz
            if not getattr(_kz, "_HAS_LADYBUG", True):
                # ladybug itself isn't importable — unrecoverable, latch permanently.
                _ladybug_failed = True
                logger.warning("LadybugDB not importable — graph features disabled (permanent).")
                return None
            MEMORY_DIR.mkdir(parents=True, exist_ok=True)  # ladybug won't create parents
            ks = _kz.LadybugStore(str(MEMORY_DIR / "graph.ladybug"))
            if not ks.available():
                # Import worked but open failed — almost always another process holds
                # the single-writer lock. Treat as TRANSIENT: retry after the backoff.
                _ladybug_last_attempt = time.monotonic()
                logger.warning("LadybugDB store unavailable (lock contention or transient IO?) "
                               "— will retry after %ss.", _LADYBUG_RETRY_SECONDS)
                return None
            _ladybug_store = ks
            _ladybug_last_attempt = 0.0
        except ImportError as exc:
            # graph module / ladybug genuinely missing — unrecoverable, latch permanently.
            _ladybug_failed = True
            logger.warning("LadybugDB graph module missing (%r) — graph features disabled (permanent).", exc)
            return None
        except Exception as exc:  # fail-open — never break the server on graph init
            # Unknown/transient error (e.g. IO on mkdir/open) — do NOT latch; retry later.
            _ladybug_last_attempt = time.monotonic()
            logger.warning("LadybugDB graph init failed (%r) — will retry after %ss.", exc, _LADYBUG_RETRY_SECONDS)
            return None
    # One-time backfill of pre-existing findings, guarded by an empty-graph check.
    try:
        _ladybug_backfill_if_empty(_ladybug_store)
    except Exception as exc:
        logger.debug("LadybugDB backfill skipped (fail-open): %r", exc)
    return _ladybug_store


def _ladybug_health_state() -> str:
    """Read-only view of the LadybugDB store state for loci_health, WITHOUT grabbing the
    single-writer lock (a RO probe never steals a writer's lock): 'available' (open +
    readable now) | 'contended' (store up but another process holds Kuzu's writer lock
    right now — a RO open fails) | 'latched' (permanent failure, won't retry) |
    'backoff' (transient failure, retrying after the window) | 'unavailable' (not yet
    initialized / unknown). With per-op leasing the store no longer holds the lock
    between ops, so 'contended' is transient and self-heals. Never raises."""
    try:
        if _ladybug_store is not None:
            probe = getattr(_ladybug_store, "readable_probe", None)
            if probe is not None and not probe():
                return "contended"
            return "available"
        if _ladybug_failed:
            return "latched"
        if _ladybug_last_attempt and (time.monotonic() - _ladybug_last_attempt) < _LADYBUG_RETRY_SECONDS:
            return "backoff"
        return "unavailable"
    except Exception:
        return "unavailable"


def _ladybug_writer_pid() -> Optional[int]:
    """Best-effort PID currently stamped as the Kuzu write-lease holder (diagnostics;
    only populated once holders run the per-op-lease code). None if unknown."""
    try:
        if _ladybug_store is not None:
            fn = getattr(_ladybug_store, "lock_holder_pid", None)
            return fn() if fn is not None else None
    except Exception as exc:
        logger.debug("_ladybug_writer_pid: fail-open swallow: %r", exc)
    return None


_code_version_cache: Optional[str] = None  # computed once per process (incl. '' failure)
_code_version_lock = threading.Lock()


def _code_version() -> str:
    """Best-effort short git SHA of the running code. '' when not a git checkout / on
    any error. Never raises. Cached for the process lifetime (incl. the failure/empty
    result) so the polled health tool never re-forks git on every call.

    First compute is guarded by a lock with double-checked locking so concurrent
    callers before the cache is filled do not each spawn a `git rev-parse`."""
    global _code_version_cache
    if _code_version_cache is not None:  # fast path: no lock once cached
        return _code_version_cache
    with _code_version_lock:
        if _code_version_cache is not None:  # re-check under the lock
            return _code_version_cache
        result = ""
        try:
            import subprocess
            out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(Path(__file__).resolve().parent),
                capture_output=True, text=True, timeout=2,
            )
            if out.returncode == 0:
                result = out.stdout.strip()
        except Exception:
            result = ""
        _code_version_cache = result
        return result


# Leaf LadybugDB helpers live in ladybug_ops.py (P2c of the split). Only the leaves
# moved: the singleton above (_ladybug_store / _ladybug_failed / _ladybug_last_attempt
# / _LADYBUG_RETRY_SECONDS) and _get_ladybug stay HERE, because tests monkeypatch those
# latches on this module and assert on server._get_ladybug(). Deps are injected on the
# same contract as inv_store/graph_tools: _get_ladybug by reference (so the helpers
# reach the store through these same patchable globals) and the memory root as a lambda
# over MEMORY_DIR (not the Path) so tmpdir rebinds still steer the backfill scan.
import ladybug_ops  # noqa: E402
ladybug_ops.register(_get_ladybug, lambda: MEMORY_DIR)
# Re-exported so `server.<helper>()` keeps working for the rest of this module, for
# in-process callers, and for tests.
from ladybug_ops import (  # noqa: E402,F401
    _ladybug_upsert_investigation, _coerce_ts, _mirror_finding_to_ladybug,
    _get_symbol_index,
    _autolink_finding_to_ladybug, _ladybug_backfill_if_empty, _entity_lookup_ladybug,
)
# NOTE: ladybug_ops._symbol_index_cache / _symbol_index_count are deliberately NOT
# re-exported. `from x import y` binds by VALUE, so re-exporting mutable module state
# would pin server.<name> to a None/-1 snapshot while the real cache moves on --
# a name that looks live and is permanently stale. Read them via ladybug_ops.


import mnemo_ops  # noqa: E402,F401
from mnemo_ops import (  # noqa: E402,F401
    _mnemo_bank, _get_mnemo_funcs, _mnemo_remember, _coerce_mnemo_results, _mnemo_recall,
)


_MEMORY_DECAY_LAMBDA  = float(os.environ.get("MEMORY_DECAY_LAMBDA", "0.007"))  # Ebbinghaus decay; half-life ~100 days


# Optional extra collection for code-chunk correlation (set CODE_CHUNKS_COLLECTION
# to the name of a Qdrant collection that holds code embeddings).
_CODE_CHUNKS_COLLECTION = os.environ.get("CODE_CHUNKS_COLLECTION", "")


from verdict_ops import _get_verdict_backend, _record_claim_verdicts  # noqa: E402,F401


# ---------------------------------------------------------------------------
# RAG context assembly helpers
# ---------------------------------------------------------------------------

def context_assemble(
    results: list[dict],
    query: str,
    budget_chars: int = 6000,
    include_metadata: bool = True,
) -> dict:
    """
    Assemble a RAG context block from search result dicts.

    Returns a dict with:
      context  - formatted prompt-ready string with [SOURCE N] citations
      sources  - list of {n, id, title, origin, score}
      query, total_chars, truncated, result_count
    """
    lines = [f"## Retrieved Context\nQuery: {query}\n"]
    sources = []
    total = 0

    for i, r in enumerate(results, 1):
        text = str(r.get("text") or r.get("content") or r.get("finding") or "")
        title = (
            r.get("title")
            or r.get("source")
            or r.get("investigation_id")
            or f"result-{i}"
        )
        score = float(r.get("score") or r.get("relevance_score") or 0.0)
        origin = r.get("origin") or r.get("collection") or "hermes_memory"
        mem_id = str(r.get("memory_id") or r.get("finding_id") or r.get("id") or "")

        meta = f"  [score={score:.3f}, origin={origin}]" if include_metadata else ""
        block = f"[SOURCE {i}]{meta}\nTitle: {title}\n{text}\n---\n"

        if total + len(block) > budget_chars and i > 1:
            lines.append(
                f"\n[{len(results) - i + 1} more results omitted — budget {budget_chars} chars]\n"
            )
            return {
                "query": query,
                "context": "\n".join(lines),
                "sources": sources,
                "total_chars": total,
                "truncated": True,
                "result_count": len(results),
            }

        lines.append(block)
        total += len(block)
        sources.append({
            "n": i,
            "id": mem_id,
            "title": title[:80],
            "origin": origin,
            "score": round(score, 4),
        })

    return {
        "query": query,
        "context": "\n".join(lines),
        "sources": sources,
        "total_chars": total,
        "truncated": False,
        "result_count": len(results),
    }


# ---------------------------------------------------------------------------
# Dual retrieval — benign-context search (CIBER / CHR pattern)
# ---------------------------------------------------------------------------

def _search_benign_context_qdrant(
    claim_text: str,
    current_investigation_id: str,
    limit: int = 5,
) -> list[dict]:
    """Retrieve cross-investigation baseline findings for entities in a claim.

    Implements the CIBER dual-retrieval pattern (arXiv:2503.07937): for each
    claim, alongside supporting evidence from the current investigation, also
    retrieve findings from OTHER investigations that describe the same entities
    behaving normally or as expected.  These are surfaced as ``benign_context_refs``
    so the analyst can ask: "does this differ meaningfully from the baseline?"

    If benign context exists alongside supporting evidence the claim is flagged
    as ``ambiguous`` — the LLM must not assert malicious intent without
    explaining why the current activity differs from the baseline.

    Uses entity payload indexes (O(1) per entity lookup), excluding the current
    investigation to avoid circular reasoning. Fail-open.
    """
    client, col = _get_qdrant()
    if client is None:
        return []

    entities = _extract_entities(claim_text)
    entity_count = sum(len(v) for v in entities.values())
    if entity_count == 0:
        return []

    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchExcept
    except ImportError:
        return []

    seen: dict[str, dict] = {}

    entity_field_map = {p: f"entities.{p}" for _, p in _ENTITY_TYPES}

    for entity_type, field in entity_field_map.items():
        for val in list(entities.get(entity_type, []))[:2]:
            if not val:
                continue
            try:
                # Query across ALL investigations for this entity, excluding current.
                # MatchExcept filters out the current investigation_id so we only
                # get cross-investigation baseline findings.
                hits, _ = client.scroll(
                    col,
                    scroll_filter=Filter(must=[
                        FieldCondition(key=field, match=MatchValue(value=val.lower())),
                        FieldCondition(
                            key="investigation_id",
                            match=MatchExcept(**{"except": [current_investigation_id]}),
                        ),
                    ]),
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )
                for p in hits:
                    pid = str(p.id)
                    if pid not in seen:
                        seen[pid] = dict(p.payload or {})
            except Exception as exc:
                logger.debug("benign_context lookup failed (%s=%r): %s", field, val, exc)

    findings = sorted(seen.values(), key=lambda f: f.get("created_at_ts", 0), reverse=True)[:limit]
    return [_summarise_finding(f, include_tags=False) for f in findings]


# ---------------------------------------------------------------------------
# Entity lookup helpers
# ---------------------------------------------------------------------------

# Explicit map avoids "hash" + "s" = "hashs" (should be "hashes").
# Order is observable: rendered into the entity_type error message below.
_ENTITY_TYPES = (
    ("ip", "ips"),
    ("email", "emails"),
    ("hostname", "hostnames"),
    ("hash", "hashes"),
    ("cve", "cves"),
)

_ENTITY_FIELD_MAP = {s: f"entities.{p}" for s, p in _ENTITY_TYPES}
_PLURAL = dict(_ENTITY_TYPES)

_IP_RE   = re.compile(r'^\d{1,3}(?:\.\d{1,3}){3}$')
# IPv6: all chars are hex digits or colons, at least two colons (covers ::1,
# fe80::1, 2001:db8::8a2e:370:7334, etc.).  Colons distinguish IPv6 from hashes.
_IPV6_RE = re.compile(r'^[0-9a-f:]+$', re.I)
_HASH_RE = re.compile(r'^[0-9a-f]{32,64}$', re.I)
_CVE_RE  = re.compile(r'^CVE-\d{4}-\d+$', re.I)


def _detect_entity_type(entity: str) -> str:
    """Infer entity type from value pattern."""
    e = entity.strip()
    if _IP_RE.match(e):
        return "ip"
    if _IPV6_RE.match(e) and e.count(":") >= 2:
        # IPv6 address — stored under entities.ips by the IOC extractor
        return "ip"
    if _HASH_RE.match(e):
        return "hash"
    if _CVE_RE.match(e):
        return "cve"
    if "@" in e and "." in e.split("@", 1)[-1]:
        return "email"
    return "hostname"


def _entity_lookup_qdrant(
    entity: str,
    entity_type: str,
    investigation_id: Optional[str],
    limit: int,
) -> list[dict]:
    """Filtered scroll on the entity payload index — O(1) vs. a full collection scan."""
    client, col = _get_qdrant()
    if client is None:
        return []
    field = _ENTITY_FIELD_MAP.get(entity_type)
    if not field:
        return []
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        must = [FieldCondition(key=field, match=MatchValue(value=entity.lower()))]
        if investigation_id:
            must.append(FieldCondition(
                key="investigation_id", match=MatchValue(value=investigation_id)
            ))
        results, _ = client.scroll(
            col,
            scroll_filter=Filter(must=must),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [dict(p.payload or {}) for p in results]
    except Exception as exc:
        logger.debug("entity_lookup_qdrant failed for %r: %s", entity, exc)
        return []


def _entity_lookup_jsonl(
    entity: str,
    entity_type: str,
    investigation_id: Optional[str],
    limit: int,
) -> list[dict]:
    """JSONL fallback: scan findings files checking stored entities, then raw text."""
    entity_lower = entity.lower()
    field_plural = _PLURAL.get(entity_type, entity_type + "s")
    results: list[dict] = []

    inv_dirs: list[Path] = []
    if investigation_id:
        d = MEMORY_DIR / investigation_id
        if d.is_dir():
            inv_dirs = [d]
    elif MEMORY_DIR.exists():
        inv_dirs = [d for d in MEMORY_DIR.iterdir() if d.is_dir()]

    for inv_dir in inv_dirs:
        findings_file = inv_dir / "findings.jsonl"
        if not findings_file.exists():
            continue
        for finding in _read_jsonl(findings_file):
            if len(results) >= limit:
                break
            entities = finding.get("entities") or {}
            stored_vals = [str(v).lower() for v in entities.get(field_plural, [])]
            if entity_lower in stored_vals:
                results.append(finding)
                continue
            # Fallback for older findings without stored entities.
            # Use word-boundary search so "10.0.0.1" doesn't match "10.0.0.10",
            # and short hashes don't match every occurrence of their hex prefix.
            if re.search(r'(?<![.\w])' + re.escape(entity_lower) + r'(?![.\w])',
                         str(finding.get("text", "")).lower()):
                results.append(finding)
        if len(results) >= limit:
            break
    return results


def _entity_lookup_cascade(
    entity: str,
    entity_type: str,
    investigation_id: Optional[str],
    limit: int,
) -> tuple[list[dict], str]:
    """Prefer the LadybugDB graph (primary), then Qdrant (indexed), then JSONL scan.

    Returns ``(findings, method)`` where ``method`` names the tier that produced
    the findings.  A total miss reports the last tier tried (``jsonl_fallback``).
    """
    findings = _entity_lookup_ladybug(entity, investigation_id, limit)
    method = "ladybug"
    if not findings:
        findings = _entity_lookup_qdrant(entity, entity_type, investigation_id, limit)
        method = "qdrant"
    if not findings:
        findings = _entity_lookup_jsonl(entity, entity_type, investigation_id, limit)
        method = "jsonl_fallback"
    return findings, method




# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

















def _rewrite_jsonl_set_field(path: Path, target_ids: set, field: str, value) -> int:
    """
    Atomically rewrite a JSONL file, setting ``field`` = ``value`` on every
    entry whose "id" is in ``target_ids``.

    Returns the count of entries that were modified.  Fails open — if any I/O
    or JSON error occurs the original file is left untouched.
    """
    if not path.exists():
        return 0
    try:
        lines = path.read_text().splitlines()
        new_lines = []
        modified = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                new_lines.append(line)
                continue
            try:
                entry = json.loads(stripped)
            except Exception:
                new_lines.append(line)
                continue
            if str(entry.get("id", "")) in target_ids:
                entry[field] = value
                new_lines.append(json.dumps(entry))
                modified += 1
            else:
                new_lines.append(line)
        # Atomic replace via temp file in the same directory
        _atomic_write_text(path, "\n".join(new_lines) + ("\n" if new_lines else ""))
        return modified
    except Exception as exc:
        logger.debug("_rewrite_jsonl_set_field failed (fail-open): %r", exc)
        return 0


# ---------------------------------------------------------------------------
# Finding lifecycle: append-only updates log + staleness (code-ref hashing).
#
# findings.jsonl stays pure (only real finding records) so every existing reader
# is untouched. In-place resolutions are recorded as APPEND-ONLY update records in
# a sibling ``finding_updates.jsonl`` and folded in last-write-wins by the read
# path. verify-all verdicts are written to a SEPARATE ``finding_verifications.jsonl``
# (round-3 scaling fix) so the resolution read path never scans the high-churn
# verification log. Both features are additive + fail-open.
# ---------------------------------------------------------------------------

# Injectable generation fn for investigation_verify_all — None means "use the
# shipped verify.py default" (lazy llm_local). Tests set this to a stub.
_verify_gen_fn = None

# Known source/config/doc file extensions a code ref may end in. Requiring the
# extension to be from THIS set (not merely "some letters after a dot") is what
# keeps ordinary prose out: "e.g." would otherwise parse as file "e" ext "g".
_CODE_REF_EXTS = (
    "pyi|pyx|py|ipynb|"
    "tsx|ts|jsx|js|mjs|cjs|vue|svelte|"
    "kts|kt|java|scala|sc|groovy|gradle|"
    "cpp|cxx|cc|hpp|hxx|hh|c|h|"
    "rs|go|rb|php|swift|mm|m|cs|dart|lua|"
    "sh|bash|zsh|fish|ps1|"
    "sql|proto|graphql|"
    "html|htm|css|scss|sass|less|"
    "json|jsonl|yaml|yml|toml|ini|cfg|conf|xml|"
    "md|rst|txt|"
    "tf|hcl|dockerfile|mk|cmake|bzl"
)
# A file-path-with-optional-line reference in finding text, e.g. "mcp/server.py:3204"
# or "verify.py". The path segment ends in a KNOWN code/config/doc extension (see
# _CODE_REF_EXTS, case-insensitive), so ordinary prose such as "e.g." — which would
# parse as file "e" with extension "g" — does not match. A trailing lookahead keeps
# "file.pyc"-style longer suffixes from matching only the "py" prefix.
_FILE_REF_RE = re.compile(
    r"\b((?:[\w.\-]+/)*[\w.\-]+\.(?:" + _CODE_REF_EXTS + r"))(?![\w.\-])(?::(\d+))?",
    re.IGNORECASE,
)




def _finding_verifications_path(investigation_id: str) -> Path:
    """Verify-all verdict notes — kept in a SEPARATE, high-churn log so the
    resolution read path (_load_resolution_overrides) never has to scan the
    accumulating per-finding verification records. Write-only for now."""
    return _inv_dir(investigation_id) / "finding_verifications.jsonl"




# Files larger than this are never hashed — a code ref points at source, not
# blobs, and this caps how much a rogue ref can make the server read. Overridable.
_HASH_FILE_MAX_BYTES_DEFAULT = 8 * 1024 * 1024


def _parse_hash_file_max_bytes() -> int:
    """Parse the file-hash size cap from the environment, fail-open to the default.

    A non-integer (or non-positive) LOCI_HASH_FILE_MAX_BYTES must never crash
    server startup — the whole code-ref hashing path is best-effort, so a bad
    override falls back to the default cap rather than raising at import time.
    """
    raw = os.environ.get("LOCI_HASH_FILE_MAX_BYTES")
    if raw is None:
        return _HASH_FILE_MAX_BYTES_DEFAULT
    try:
        val = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "invalid LOCI_HASH_FILE_MAX_BYTES=%r; falling back to default %d",
            raw,
            _HASH_FILE_MAX_BYTES_DEFAULT,
        )
        return _HASH_FILE_MAX_BYTES_DEFAULT
    if val <= 0:
        logger.warning(
            "non-positive LOCI_HASH_FILE_MAX_BYTES=%r; falling back to default %d",
            raw,
            _HASH_FILE_MAX_BYTES_DEFAULT,
        )
        return _HASH_FILE_MAX_BYTES_DEFAULT
    return val


_HASH_FILE_MAX_BYTES = _parse_hash_file_max_bytes()


def _code_root() -> Path:
    """Root that user-supplied code refs must stay under. Defaults to the process
    working directory (the repo being investigated); overridable via LOCI_CODE_ROOT.
    Re-read each call so tests / re-rooted runs are honored."""
    root = os.environ.get("LOCI_CODE_ROOT") or os.getcwd()
    return Path(root).resolve()


def _safe_repo_path(path_str: str) -> Optional[Path]:
    """Resolve a user-controlled ref to a real path UNDER the code root, else None.

    Security boundary for _hash_file_bytes: file refs come from finding text /
    caller-supplied code_refs, so an absolute path or ``..`` traversal could probe
    any readable file on the host. This rejects absolute paths and any ``..``
    component, resolves relative to the code root, and requires the resolved path
    (symlinks included) to remain inside that root. Fail-open — returns None on
    anything rejected or unresolvable so the caller simply skips the ref."""
    try:
        raw = (path_str or "").strip()
        if not raw:
            return None
        p = Path(raw)
        if p.is_absolute() or any(part == ".." for part in p.parts):
            return None
        root = _code_root()
        resolved = (root / p).resolve()
        if not resolved.is_relative_to(root):
            return None
        return resolved
    except Exception:  # noqa: BLE001
        return None


def _hash_file_bytes(path_str: str) -> Optional[str]:
    """sha256 of a referenced file's current bytes, or None if unreadable (fail-open).

    The path is scoped to the code root via _safe_repo_path (absolute paths, ``..``
    traversal, and anything escaping the root are rejected) and files larger than
    _HASH_FILE_MAX_BYTES are skipped — so this only ever hashes in-repo source,
    never arbitrary or oversized files on the host."""
    try:
        p = _safe_repo_path(path_str)
        if p is None or not p.is_file():
            return None
        if p.stat().st_size > _HASH_FILE_MAX_BYTES:
            return None
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:  # noqa: BLE001
        return None


def _compute_code_refs(text: str, explicit=None) -> list[dict]:
    """Best-effort: resolve file paths to [{"path": .., "hash": ..}] by hashing each
    file that currently exists.

    ``explicit`` refs are AUTHORITATIVE whenever they are *provided* — including when
    provided but empty. The distinction is NOT-PROVIDED vs PROVIDED:
      * ``explicit is None``       -> not provided: paths are best-effort parsed from
                                      ``text`` (tokens like "path/file.py:12").
      * ``explicit`` == '' or []   -> provided but empty: authoritative "no refs", so
                                      text-parsing is skipped and [] is returned.
      * ``explicit`` non-empty     -> the caller named exactly the files this finding
                                      refers to; text-parsing is skipped entirely.

    Fully optional + fail-open: unreadable/nonexistent paths are simply omitted, and
    any error returns []. Line suffixes ("file.py:12") are stripped for hashing.
    """
    candidates: list[str] = []
    try:
        if explicit is not None:
            # Provided (possibly empty) -> authoritative, never fall back to text.
            items = explicit if isinstance(explicit, list) else str(explicit).split(",")
            candidates.extend(str(i) for i in items)
        else:
            for m in _FILE_REF_RE.finditer(text or ""):
                candidates.append(m.group(1))
    except Exception:  # noqa: BLE001
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for c in candidates:
        path = (c or "").strip().split(":")[0].strip()
        if not path or path in seen:
            continue
        seen.add(path)
        h = _hash_file_bytes(path)
        if h is not None:
            out.append({"path": path, "hash": h})
    return out


def _finding_is_stale(finding: dict) -> Optional[bool]:
    """Re-hash a finding's stamped code_refs. Returns True if any referenced file's
    content changed, False if refs exist and none changed, None if the finding has
    no usable refs (caller omits the ``stale`` key in that case). Fail-open."""
    refs = finding.get("code_refs") if isinstance(finding, dict) else None
    if not isinstance(refs, list) or not refs:
        return None
    checked = False
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        path = ref.get("path")
        old = ref.get("hash")
        if not path or not old:
            continue
        cur = _hash_file_bytes(str(path))
        if cur is None:
            continue  # unreadable now -> don't flag (fail-open)
        checked = True
        if cur != old:
            return True
    return False if checked else None


def _apply_lifecycle(findings: list[dict], investigation_id: str) -> None:
    """In-place: stamp effective ``resolution`` (append-log override else stored/'open')
    and ``stale`` (only when the finding carries usable code_refs). Fail-open."""
    overrides = _load_resolution_overrides(investigation_id)
    for f in findings:
        if not isinstance(f, dict):
            continue
        fid = str(f.get("id", ""))
        if fid and fid in overrides:
            f["resolution"] = overrides[fid]
        elif not f.get("resolution"):
            f["resolution"] = "open"
        stale = _finding_is_stale(f)
        if stale is not None:
            f["stale"] = stale


# ---------------------------------------------------------------------------
# Session hints — module-level ring buffer updated by investigation_store so
# that memory_hints (and the MCP resource) can surface "what changed recently"
# without re-scanning JSONL.  Each key is an investigation_id; the value is a
# list of hint dicts (most-recent-last) capped at _SESSION_HINTS_MAX_PER_INV.
# ---------------------------------------------------------------------------
_session_hints: dict[str, list[dict]] = {}
_SESSION_HINTS_MAX_PER_INV = 20  # ring-buffer cap per investigation


def _session_hints_push(investigation_id: str, hint: dict) -> None:
    """Append a hint to the in-process ring buffer (fail-open)."""
    try:
        buf = _session_hints.setdefault(investigation_id, [])
        buf.append(hint)
        if len(buf) > _SESSION_HINTS_MAX_PER_INV:
            del buf[0]
    except Exception as exc:  # noqa: BLE001
        logger.debug("_session_hints_push: fail-open swallow: %r", exc)


_REFLECTION_ERROR_RE = re.compile(r"\b(error|exception|traceback|failed|failure|timeout|conflict)\b", re.I)
_REFLECTION_WARN_RE = re.compile(r"\b(warn|warning|degraded|fallback|retry)\b", re.I)
_REFLECTION_HEX_RE = re.compile(r"\b[0-9a-f]{7,64}\b", re.I)
_REFLECTION_NUM_RE = re.compile(r"\b\d+\b")
_REFLECTION_TS_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:z|[+-]\d{2}:\d{2})?\b", re.I)


def _reflection_default_state() -> dict:
    now = _now()
    return {
        "version": 1,
        "investigation_id": REFLECTION_DEFAULT_INVESTIGATION,
        "queue": [],
        "processed": {},
        "stats": {
            "files_processed": 0,
            "lines_scanned": 0,
            "errors_seen": 0,
            "warnings_seen": 0,
            "bytes_scanned": 0,
            "error_signatures_suppressed": 0,
            "warning_signatures_suppressed": 0,
            "error_signature_observations": {},
            "warning_signature_observations": {},
            "last_error_signatures": [],
            "last_warning_signatures": [],
        },
        "created_at": now,
        "updated_at": now,
        "last_tick": None,
    }


def _load_reflection_state() -> dict:
    if not REFLECTION_STATE_FILE.exists():
        return _reflection_default_state()
    try:
        state = json.loads(REFLECTION_STATE_FILE.read_text())
    except Exception:
        logger.warning("reflection_loop: state file unreadable, resetting: %s", REFLECTION_STATE_FILE)
        return _reflection_default_state()
    if not isinstance(state, dict):
        return _reflection_default_state()
    merged = _reflection_default_state()
    merged.update(state)
    merged["stats"].update(state.get("stats") or {})
    merged["queue"] = list(state.get("queue") or [])
    merged["processed"] = dict(state.get("processed") or {})
    return merged


def _save_reflection_state(state: dict) -> None:
    state["updated_at"] = _now()
    _atomic_write_text(REFLECTION_STATE_FILE, json.dumps(state, indent=2))


def _canonicalize_reflection_signature(text: str) -> str:
    line = str(text or "").strip().lower()
    line = _REFLECTION_TS_RE.sub("<ts>", line)
    line = _REFLECTION_HEX_RE.sub("<hex>", line)
    line = _REFLECTION_NUM_RE.sub("<n>", line)
    line = re.sub(r"\s+", " ", line)
    return line[:220]


def _hash_path(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]


def _reflection_queue_priority(kind: str) -> int:
    # Lower = higher priority.
    return {
        "process_log": 0,
        "temp_ingest": 1,
        "session_event": 2,
    }.get(str(kind or ""), 3)


def _read_tail_lines(path: Path, *, max_lines: int, max_bytes: int) -> list[str]:
    try:
        size = int(path.stat().st_size)
    except Exception:
        return []
    if size <= 0:
        return []
    read_size = min(size, max_bytes)
    with path.open("rb") as fh:
        if size > read_size:
            fh.seek(-read_size, os.SEEK_END)
        chunk = fh.read(read_size)
    text = chunk.decode("utf-8", errors="ignore")
    return text.splitlines()[-max_lines:]


def _prune_signature_observations(observations: dict) -> dict:
    if not isinstance(observations, dict):
        return {}
    if len(observations) <= REFLECTION_SIGNATURE_MAP_LIMIT:
        return {str(k): int(v) for k, v in observations.items()}
    ranked = sorted(
        ((str(k), int(v)) for k, v in observations.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:REFLECTION_SIGNATURE_MAP_LIMIT]
    return dict(ranked)


def _ensure_investigation_exists(investigation_id: str, *, title: str, context: str) -> None:
    if _load_manifest(investigation_id):
        return
    manifest = {
        "id": investigation_id,
        "title": title,
        "context": context,
        "status": "active",
        "created_at": _now(),
        "updated_at": _now(),
        "hypothesis": None,
        "open_questions": [],
        "next_step": None,
        "checked_sources": {},
        "finding_counts": {"observed": 0, "inferred": 0, "assumed": 0, "gap": 0},
        "closed_at": None,
        "closed_summary": None,
    }
    _save_manifest(manifest)


@dataclass
class _ReflectionScan:
    event_counts: Counter = field(default_factory=Counter)
    tool_counts: Counter = field(default_factory=Counter)
    error_counts: Counter = field(default_factory=Counter)
    warning_counts: Counter = field(default_factory=Counter)
    lines_scanned: int = 0
    bytes_scanned: int = 0
    sampling_mode: str = "full"

    def scan_line(self, line: str) -> None:
        canon = _canonicalize_reflection_signature(line)
        if _REFLECTION_ERROR_RE.search(line):
            self.error_counts[canon] += 1
        elif _REFLECTION_WARN_RE.search(line):
            self.warning_counts[canon] += 1


def _scan_temp_ingest(file_path: Path, scan: "_ReflectionScan") -> None:
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    scan.bytes_scanned += len(content.encode("utf-8", errors="ignore"))
    scan.lines_scanned = 1
    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            for key in list(payload.keys())[:30]:
                scan.event_counts[f"payload_key:{key}"] += 1
        scan.scan_line(json.dumps(payload)[:20000])
    except Exception:
        scan.scan_line(content[:20000])


def _scan_session_event(file_path: Path, scan: "_ReflectionScan", max_lines: int) -> None:
    with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            if scan.lines_scanned >= max_lines:
                break
            scan.lines_scanned += 1
            scan.bytes_scanned += len(raw.encode("utf-8", errors="ignore"))
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                scan.scan_line(line)
                continue
            event_type = str(event.get("type") or event.get("event") or "unknown")
            scan.event_counts[event_type] += 1
            tool_name = event.get("tool_start_name") or event.get("tool_complete_name") or event.get("tool_name")
            if tool_name:
                scan.tool_counts[str(tool_name)] += 1
            joined = " ".join([
                str(event.get("user_content") or ""),
                str(event.get("assistant_content") or ""),
                str(event.get("tool_complete_result_content") or ""),
            ])
            scan.scan_line(joined)


def _scan_claude_code_event(file_path: Path, scan: "_ReflectionScan", max_lines: int) -> None:
    with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            if scan.lines_scanned >= max_lines:
                break
            scan.lines_scanned += 1
            scan.bytes_scanned += len(raw.encode("utf-8", errors="ignore"))
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                scan.scan_line(line)
                continue
            # Claude Code event schema: {"type": "...", "message": {...}, "attachments": [...]}
            event_type = str(event.get("type") or "unknown")
            scan.event_counts[event_type] += 1
            # Tool names appear in attachments list as {"toolName": "..."} entries
            for attachment in (event.get("attachments") or []):
                if isinstance(attachment, dict):
                    tool_name = attachment.get("toolName") or attachment.get("tool_name")
                    if tool_name:
                        scan.tool_counts[str(tool_name)] += 1
            # Message content is nested under "message" with role/content structure
            message = event.get("message") or {}
            if isinstance(message, dict):
                content = message.get("content") or ""
                if isinstance(content, list):
                    # content may be a list of blocks: [{"type": "text", "text": "..."}]
                    content = " ".join(
                        str(block.get("text") or "") for block in content
                        if isinstance(block, dict)
                    )
                joined = str(content)
            else:
                joined = str(message)
            scan.scan_line(joined)


def _scan_process_log(file_path: Path, scan: "_ReflectionScan", max_lines: int) -> None:
    file_bytes = int(file_path.stat().st_size)
    if file_bytes > REFLECTION_LOG_TAIL_MIN_FILE_BYTES:
        scan.sampling_mode = "tail"
        for raw in _read_tail_lines(
            file_path,
            max_lines=max_lines,
            max_bytes=REFLECTION_LOG_TAIL_READ_BYTES,
        ):
            scan.lines_scanned += 1
            scan.bytes_scanned += len(raw.encode("utf-8", errors="ignore"))
            scan.scan_line(raw)
            m = re.search(r"\btool(?:Name)?[=:\"]+([a-zA-Z0-9_.:-]+)", raw)
            if m:
                scan.tool_counts[m.group(1)] += 1
    else:
        with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                if scan.lines_scanned >= max_lines:
                    break
                scan.lines_scanned += 1
                scan.bytes_scanned += len(raw.encode("utf-8", errors="ignore"))
                scan.scan_line(raw)
                m = re.search(r"\btool(?:Name)?[=:\"]+([a-zA-Z0-9_.:-]+)", raw)
                if m:
                    scan.tool_counts[m.group(1)] += 1


def _process_reflection_item(kind: str, path: str, max_lines: int) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {
            "status": "missing",
            "kind": kind,
            "path": path,
            "lines_scanned": 0,
            "bytes_scanned": 0,
            "events": {},
            "tools": {},
            "errors": {},
            "warnings": {},
        }

    scan = _ReflectionScan()

    if kind == "temp_ingest":
        _scan_temp_ingest(file_path, scan)
    elif kind == "session_event":
        _scan_session_event(file_path, scan, max_lines)
    elif kind == "claude_code_event":
        _scan_claude_code_event(file_path, scan, max_lines)
    elif kind == "process_log":
        _scan_process_log(file_path, scan, max_lines)
    else:
        return {
            "status": "unsupported_kind",
            "kind": kind,
            "path": path,
            "lines_scanned": 0,
            "bytes_scanned": 0,
            "events": {},
            "tools": {},
            "errors": {},
            "warnings": {},
        }

    return {
        "status": "processed",
        "kind": kind,
        "path": path,
        "lines_scanned": scan.lines_scanned,
        "bytes_scanned": scan.bytes_scanned,
        "sampling_mode": scan.sampling_mode,
        "events": dict(scan.event_counts.most_common(8)),
        "tools": dict(scan.tool_counts.most_common(8)),
        "errors": dict(scan.error_counts.most_common(8)),
        "warnings": dict(scan.warning_counts.most_common(8)),
    }


def _normalize_derived_from(derived_from: str | list[str] | None) -> list[str]:
    """Coerce a ``derived_from`` arg to a clean, deduped list of strings.

    Accepts a single id/claim string or a list; returns ``[]`` for empty/None so
    callers omit the field. Backward-compatible: existing callers pass nothing.
    """
    if derived_from is None:
        return []
    items = derived_from if isinstance(derived_from, (list, tuple, set)) else [derived_from]
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        s = str(item).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out




_NEGATION_RE = re.compile(r"\b(?:no|not|never|none|without|cannot|can't|didn't|isn't|aren't|won't)\b", re.I)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._:/-]{2,}", re.I)
# Words that carry no evidentiary weight in a lexical match. Two groups, kept
# separate because they are dropped for different reasons.
#
# Domain-generic: true of almost every finding in this corpus, so sharing one
# says nothing.
_GENERIC_MATCH_TOKENS = {
    "host", "user", "device", "query", "result", "results", "output", "input", "tool",
    "found", "seen", "shows", "reported", "detected", "contacted", "event", "events",
    "record", "records", "row", "rows",
}

# Function words. These were NOT dropped, and _lexical_match_score normalises by
# the claim side only, so a short claim could clear the 0.45 gate on stopword
# overlap alone. Measured on 400 real findings before this change: 68.8% of
# generic short claims were reported as supported, e.g. "the server is down"
# declared supported by a document on cultural-noise characterisation, on the
# overlap {"down", "the"}. investigation_pre_answer_check is the tool an agent
# calls BEFORE asserting a claim, so this was a rubber stamp on the one gate
# meant to catch unsupported assertions.
_STOPWORD_MATCH_TOKENS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "may", "might", "must", "no",
    "not", "of", "on", "or", "our", "out", "over", "own", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "them", "then", "there", "these",
    "they", "this", "those", "to", "too", "under", "up", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "why", "will", "with", "would", "you",
    "your", "am", "being", "both", "each", "few", "more", "most", "only", "other",
    "same", "very", "just", "also", "any", "all", "about", "after", "before",
}

_NON_EVIDENCE_TOKENS = _GENERIC_MATCH_TOKENS | _STOPWORD_MATCH_TOKENS
_QDRANT_SUPPORT_MIN_SCORE = 0.55
_QDRANT_PRECHECK_MIN_SCORE = 0.5


def _normalize_claims(claims: str | list[str]) -> list[str]:
    if isinstance(claims, list):
        return [str(c).strip() for c in claims if str(c).strip()]

    if not isinstance(claims, str):
        return [str(claims).strip()] if str(claims).strip() else []

    raw = claims.strip()
    if not raw:
        return []

    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(c).strip() for c in parsed if str(c).strip()]
        except Exception as exc:
            logger.debug("_normalize_claims: fail-open swallow: %r", exc)

    lines = [line.strip(" -\t") for line in raw.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    return [raw]


def _confidence_allowed(confidence: str, min_confidence: str) -> bool:
    return _CONFIDENCE_RANK.get(confidence, -1) >= _CONFIDENCE_RANK.get(min_confidence, -1)


def tokenize(text: str) -> set[str]:
    """Content tokens only — the units a lexical match is allowed to count.

    Shared by every lexical lane (pre-answer check, recall scoring, target match,
    provenance). None of them want a function word to count as evidence, so the
    drop set lives here rather than at one call site.
    """
    return {
        token for token in (m.group(0).lower() for m in _TOKEN_RE.finditer(text or ""))
        if token not in _NON_EVIDENCE_TOKENS
    }


def _evidence_id(entry: dict, fallback_prefix: str, idx: int) -> str:
    if entry.get("id"):
        return str(entry["id"])
    if entry.get("ts") and entry.get("tool"):
        return f"audit:{entry.get('tool')}:{entry.get('ts')}"
    return f"{fallback_prefix}:{idx}"


def _entry_snippet(entry: dict) -> str:
    text = str(entry.get("text") or entry.get("output") or entry.get("inputs") or "")
    return text.replace("\n", " ").strip()[:260]


def _collect_recent_global_audit(limit: int = 200, days: int = 3) -> list[dict]:
    audit_dir = MEMORY_DIR.parent / "audit"
    if not audit_dir.exists():
        return []
    files = sorted(audit_dir.glob("*.jsonl"), reverse=True)[: max(days, 1)]
    entries: list[dict] = []
    for path in files:
        for entry in reversed(_read_jsonl(path)):
            entries.append(entry)
            if len(entries) >= limit:
                return entries
    return entries


def build_validation_evidence(
    investigation_id: str,
    min_confidence: str,
) -> list[dict]:
    findings = _read_jsonl(_inv_dir(investigation_id) / "findings.jsonl")
    scoped_audit = _read_jsonl(_inv_dir(investigation_id) / "audit.jsonl")
    global_recent_audit = _collect_recent_global_audit(limit=150, days=2)

    evidence: list[dict] = []
    for idx, finding in enumerate(findings):
        conf = str(finding.get("confidence", "low")).lower()
        if not _confidence_allowed(conf, min_confidence):
            continue
        evidence.append({
            "evidence_id": _evidence_id(finding, "finding", idx),
            "record_type": str(finding.get("record_type") or finding.get("type") or "observed"),
            "source": str(finding.get("source", "")),
            "ts": finding.get("ts"),
            "text": str(finding.get("text", "")),
            "snippet": _entry_snippet(finding),
            "tokens": tokenize(str(finding.get("text", ""))),
            "origin": "findings_jsonl",
        })

    for idx, entry in enumerate(scoped_audit):
        output = str(entry.get("output", ""))
        evidence_text = f"{entry.get('tool', '')} {entry.get('inputs', '')} {output[:3000]}"
        evidence.append({
            "evidence_id": _evidence_id(entry, "audit_scoped", idx),
            "record_type": "audit",
            "source": str(entry.get("tool", "")),
            "ts": entry.get("ts"),
            "text": evidence_text,
            "snippet": _entry_snippet(entry),
            "tokens": tokenize(evidence_text),
            "origin": "audit_jsonl",
        })

    for idx, entry in enumerate(global_recent_audit):
        if entry.get("investigation_id") != investigation_id:
            continue
        output = str(entry.get("output", ""))
        evidence_text = f"{entry.get('tool', '')} {entry.get('inputs', '')} {output[:2000]}"
        evidence.append({
            "evidence_id": _evidence_id(entry, "audit_global", idx),
            "record_type": "audit",
            "source": str(entry.get("tool", "")),
            "ts": entry.get("ts"),
            "text": evidence_text,
            "snippet": _entry_snippet(entry),
            "tokens": tokenize(evidence_text),
            "origin": "global_audit_jsonl",
        })
    return evidence


def _search_qdrant_claim_evidence(
    claim: str,
    investigation_id: str,
    limit: int = 5,
) -> tuple[list[dict], dict]:
    client, col = _get_qdrant()
    qdrant_url = os.environ.get("QDRANT_URL", "")
    status = {
        "enabled": bool(qdrant_url),
        "available": client is not None,
        "query_attempted": False,
        "error": None,
    }
    if client is None:
        return [], status

    vector = _embed(claim)
    if vector is None:
        status["error"] = "embedding_unavailable"
        return [], status

    status["query_attempted"] = True
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        search_filter = Filter(must=[
            FieldCondition(key="investigation_id", match=MatchValue(value=investigation_id))
        ])
        result = client.query_points(
            collection_name=col,
            query=vector,
            using="dense",
            query_filter=search_filter,
            limit=max(1, min(limit, 20)),
            with_payload=True,
        )
        matches = []
        for point in result.points:
            payload = point.payload or {}
            text = str(payload.get("text") or payload.get("output") or "")
            matches.append({
                "evidence_id": str(payload.get("id") or point.id),
                "record_type": str(payload.get("record_type") or payload.get("type") or "unknown"),
                "source": str(payload.get("source") or payload.get("tool") or ""),
                "ts": payload.get("ts"),
                "origin": "qdrant",
                "score": round(float(point.score), 4),
                "snippet": text.replace("\n", " ").strip()[:260],
            })
        return matches, status
    except Exception as exc:
        status["error"] = str(exc)
        return [], status


def _lexical_match_score(claim_tokens: set[str], evidence_tokens: set[str]) -> float:
    if not claim_tokens or not evidence_tokens:
        return 0.0
    overlap = claim_tokens.intersection(evidence_tokens)
    return len(overlap) / max(1, len(claim_tokens))




def _compute_aggregate_confidence(
    finding_id: str,
    findings_by_id: dict,
    max_depth: int = 5,
) -> float:
    """Walk the derived_from chain for a finding and return the product of
    numeric_confidence values along the chain (up to max_depth nodes).

    Findings without numeric_confidence are treated as 1.0 (backward compat).
    Returns 1.0 if the finding_id is not found or the chain is empty.
    """
    try:
        product = 1.0
        visited: set[str] = set()
        current_id = finding_id
        depth = 0
        while current_id and current_id not in visited and depth < max_depth:
            visited.add(current_id)
            node = findings_by_id.get(current_id)
            if not node:
                break
            nc = node.get("numeric_confidence", 1.0)
            try:
                product *= float(nc)
            except (TypeError, ValueError) as exc:
                logger.debug("_compute_aggregate_confidence: fail-open swallow: %r", exc)
            parents = node.get("derived_from") or []
            current_id = str(parents[0]) if parents else None
            depth += 1
        return round(product, 6)
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Memory self-check helpers (advisory-only; never mutate/delete findings)
# ---------------------------------------------------------------------------



def _compute_self_check(investigation_id: str, llm_verify: bool = False) -> dict:
    """Run provenance + contradiction inline over an investigation's JSONL.

    Pure over the JSONL — does NOT require qdrant. Returns the raw verdict lists
    keyed by check, plus a derived ``hallucination_candidates`` list. Already-
    retracted findings are excluded so a cleaned-up hallucination stops being
    re-surfaced. Fail-open: a check error degrades to an empty list.

    When ``llm_verify`` is set (the deep_think -> loci merge path), the lexical
    contradiction verdicts are run through an embedding subject gate + LLM
    polarity judge (``contradiction_llm.verify_and_merge``): same-subject pairs
    are confirmed by a model that ignores wording, which drops the bag-of-words
    false positives and adds the semantic negations token overlap misses. Stays
    fail-open — if embeddings/LLM are unreachable the lexical verdicts pass
    through unchanged.
    """
    raw_findings = _read_jsonl(_inv_dir(investigation_id) / "findings.jsonl")
    retracted = _load_retracted_ids(investigation_id)
    if retracted:
        raw_findings = [f for f in raw_findings if str(f.get("id", "")) not in retracted]
    findings = _tag_finding_ids(raw_findings, investigation_id)
    audit_entries = _read_jsonl(_inv_dir(investigation_id) / "audit.jsonl")
    audit_lane = _audit_lane_status(findings, audit_entries)

    try:
        unsupported = run_provenance(
            findings,
            audit_entries,
            tokenizer=tokenize,
            lexical_score=_lexical_match_score,
        )
    except Exception as exc:  # fail-open — advisory check must never break the caller
        logger.debug("provenance check failed, degrading to none: %r", exc)
        unsupported = []

    try:
        contradictions = run_contradiction(
            findings,
            negation_re=_NEGATION_RE,
            tokenizer=tokenize,
        )
    except Exception as exc:  # fail-open
        logger.debug("contradiction check failed, degrading to none: %r", exc)
        contradictions = []

    if llm_verify:
        try:
            from memcheck.checks.contradiction_llm import verify_and_merge

            contradictions = verify_and_merge(findings, contradictions)
        except Exception as exc:  # fail-open — keep lexical verdicts on any error
            logger.debug("llm contradiction verify failed, keeping lexical: %r", exc)

    try:
        candidates = _hallucination_candidates(
            findings, audit_entries, unsupported, contradictions
        )
    except Exception as exc:  # fail-open
        logger.debug("hallucination-candidate surfacing failed, degrading: %r", exc)
        candidates = []

    return {
        "unsupported_observed": unsupported,
        "contradictions": contradictions,
        "hallucination_candidates": candidates,
        "audit_lane": audit_lane,
    }


def _audit_lane_status(findings: list[dict], audit_entries: list[dict]) -> dict:
    """Whether provenance verdicts on this investigation mean anything.

    run_provenance flags every observed finding without a matching receipt, which
    is the documented per-finding contract and is correct. But with NO receipts
    at all it flags every observed finding it is given, and a verdict that fires
    on 100% of its inputs carries no information.

    Measured on the live corpus: 1 of 140 investigations has an audit.jsonl, and
    nothing has written a receipt since 2026-06-20. The cause is structural —
    audit_log is an MCP tool an agent has to remember to call after every
    invocation, described as a "post-call hook" but not hooked to anything. When
    whatever workflow was calling it stopped, the lane went silent and nothing
    said so.

    This does not change any verdict. It reports the lane's state so a caller can
    say "no receipts exist" instead of "every observed finding is unsupported" —
    two very different claims that were indistinguishable in the output.
    """
    observed = sum(1 for f in findings or []
                   if (f.get("type") or f.get("record_type")) == "observed"
                   and str(f.get("text", "") or "").strip())
    receipts = sum(1 for e in audit_entries or [] if isinstance(e, dict))
    if receipts:
        return {"status": "present", "receipts": receipts, "observed_findings": observed}
    return {
        "status": "empty",
        "receipts": 0,
        "observed_findings": observed,
        "verdicts_informative": False,
        "detail": (
            f"no audit receipts for this investigation, so all {observed} observed "
            "finding(s) are reported unsupported by construction. This is the "
            "absence of evidence about provenance, not evidence of bad provenance."
        ),
    }


def _hallucination_candidates(
    findings: list[dict],
    audit_entries: list[dict],
    unsupported,
    contradictions,
) -> list[dict]:
    """Surface findings likely to be self-generated hallucinations.

    A candidate is a finding that is BOTH:
      - ``unsupported_observed`` (no audit receipt, from the provenance check), AND
      - on the contradicted side of a contradiction whose OTHER finding DOES have
        an audit receipt — i.e. an unsupported positive contradicted by a
        receipted negative.

    These are the strongest "stored a fact that testing later disproved" signals.
    Advisory only: each candidate carries a hint to run ``memory_retract``; this
    NEVER auto-retracts. Pure over the in-memory findings/verdicts.
    """
    unsupported_ids = {r for v in (unsupported or []) for r in (v.refs or [])}

    # An "unsupported" verdict means two very different things depending on
    # whether an audit lane exists at all. run_provenance flags every observed
    # finding that lacks a matching receipt — which, on an investigation with NO
    # audit.jsonl, is every observed finding it is given. Measured: 1 of 140
    # investigations on the live corpus has one. The `other in unsupported_ids`
    # skip below then discards every observed-vs-observed contradiction, so the
    # strongest hallucination signal available can never produce a candidate.
    #
    # When EVERY observed finding is unsupported the signal is uninformative
    # rather than damning, so it must not gate the contradiction pass. The
    # provenance check keeps its documented per-finding semantics; this is the
    # consumer declining to read a blanket verdict as evidence about any one
    # finding.
    _observed_ids = {str(f.get("id", "")) for f in (findings or [])
                     if str(f.get("record_type") or f.get("type") or "") == "observed"
                     and f.get("id")}
    _blanket = bool(_observed_ids) and _observed_ids <= unsupported_ids
    if _blanket:
        logger.debug("_hallucination_candidates: every observed finding is unsupported "
                     "(%d) — treating provenance as uninformative rather than gating on it",
                     len(_observed_ids))

    if not unsupported_ids or not contradictions:
        return []

    # Which findings have an audit receipt? A finding is "receipted" iff it is
    # NOT flagged unsupported by provenance (provenance only flags observed
    # findings; treat inferred/assumed as receipted-by-default only when they
    # are not themselves unsupported). We approximate "has a receipt" as
    # "not in unsupported_ids".
    findings_by_id = {str(f.get("id", "")): f for f in findings}

    candidates: list[dict] = []
    seen: set[str] = set()
    for v in contradictions:
        refs = list(v.refs or [])
        if len(refs) != 2:
            continue
        a, b = str(refs[0]), str(refs[1])
        # Identify which side is the unsupported positive and which is the
        # receipted counter-finding.
        pairs = [(a, b), (b, a)]
        for unsup, other in pairs:
            if unsup not in unsupported_ids:
                continue
            if other in unsupported_ids and not _blanket:
                continue  # other also unsupported — no receipted counter
            if other not in findings_by_id:
                continue
            if unsup in seen:
                continue
            seen.add(unsup)
            f = findings_by_id.get(unsup, {})
            candidates.append({
                "finding_id": unsup,
                "contradicted_by": other,
                "excerpt": redact_excerpt(str(f.get("text", "") or "")),
                "rationale": (
                    "unsupported observed finding (no receipt) contradicted by a "
                    "receipted finding — likely a self-generated hallucination"
                ),
                "hint": "review and run memory_retract(target=<finding_id>) to clean the lineage",
            })
    return candidates


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("loci")


@mcp.custom_route("/health", methods=["GET"], include_in_schema=False)
async def health(request):  # noqa: ARG001
    from starlette.responses import JSONResponse
    return JSONResponse({"status": "ok", "server": "loci"})


# ---- Tool: investigation_start ----

# ---- Tool: investigation_load ----

# ---------------------------------------------------------------------------
# Conflict detection helpers
# ---------------------------------------------------------------------------

_NEGATION_MARKERS = frozenset(["not ", "no ", "never", "false"])


def _has_negation(text: str) -> bool:
    """Return True if text contains any negation marker (case-insensitive)."""
    lower = text.lower()
    return any(marker in lower for marker in _NEGATION_MARKERS)


_CONFLICT_NEGATION_HEURISTIC = os.environ.get("LOCI_CONFLICT_NEGATION_HEURISTIC", "") == "1"


def _query_points_compat(
    client,
    collection: str,
    dense_vec,
    *,
    limit: int,
    query_filter=None,
    score_threshold: Optional[float] = None,
    search_params=None,
) -> list:
    """Dense vector search over ``collection``, returning the raw point list.

    ``QdrantClient.search()`` was removed in qdrant-client 1.16 — inside this
    project's own ``>=1.17.0,<1.19.0`` pin — so on every supported client the
    attribute lookup raised before any request was made. Both callers wrapped
    that in a bare except, so the removal read as "found nothing" rather than as
    a fault. ``query_points()`` is the replacement.

    Named-vector collections need ``using="dense"`` and flat ones reject it, so
    try named first and fall back rather than probing the collection config.
    """
    kwargs = {
        "collection_name": collection,
        "query": dense_vec,
        "limit": limit,
        "with_payload": True,
    }
    if query_filter is not None:
        kwargs["query_filter"] = query_filter
    if score_threshold is not None:
        kwargs["score_threshold"] = score_threshold
    if search_params is not None:
        kwargs["search_params"] = search_params

    try:
        return client.query_points(using="dense", **kwargs).points
    except Exception as exc:
        logger.debug("_query_points_compat: named-vector query failed (%r); retrying flat", exc)
        return client.query_points(**kwargs).points


def _detect_conflicts(investigation_id: str, new_finding: dict) -> list[dict]:
    """
    Search Qdrant for near-neighbors of new_finding (same investigation, cosine
    > 0.82, excluding the new finding itself) and apply simple conflict heuristics.

    Returns a list of conflict dicts (may be empty). Fail-open — any exception
    returns an empty list so investigation_store is never blocked.
    """
    try:
        client, col = _get_qdrant()
        if client is None:
            return []

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
        except ImportError:
            return []

        dense_vec = _embed(new_finding.get("text", ""))
        if dense_vec is None:
            return []

        search_filter = Filter(must=[
            FieldCondition(
                key="investigation_id",
                match=MatchValue(value=investigation_id),
            )
        ])

        try:
            from qdrant_client.models import SearchParams, QuantizationSearchParams
            _sp = SearchParams(quantization=QuantizationSearchParams(rescore=True, oversampling=2.0))
            result = _query_points_compat(
                client, col, dense_vec,
                query_filter=search_filter,
                limit=10,
                score_threshold=0.82,
                search_params=_sp,
            )
        except Exception as exc:
            logger.warning("_detect_conflicts: neighbour search failed: %r", exc)
            return []

        new_id = new_finding.get("id", "")
        new_type = new_finding.get("record_type", "")
        new_text = new_finding.get("text", "")
        new_neg = _has_negation(new_text)

        conflicts = []
        for hit in result:
            payload = dict(hit.payload or {})
            neighbor_id = str(payload.get("id", hit.id))
            # Skip the finding itself (shouldn't appear since it may not yet be
            # in the index, but guard anyway).
            if neighbor_id == new_id:
                continue

            neighbor_type = payload.get("record_type") or payload.get("type", "")
            neighbor_text = str(payload.get("text", ""))
            neighbor_neg = _has_negation(neighbor_text)

            is_conflict = False

            # Heuristic 1: gap now filled by an observed finding
            if neighbor_type == "gap" and new_type == "observed":
                is_conflict = True

            # Heuristic 2: assumption overridden by a non-assumed finding
            elif neighbor_type == "assumed" and new_type != "assumed":
                is_conflict = True

            # Heuristic 3: opposing negation markers. Off by default — this is a
            # bare token-presence scan, not a polarity comparison, and phrases like
            # "no adverse records found" or "did not appear" are common enough that
            # it manufactures conflicts from incidental wording. Measured in the
            # 2026-08-19 upgrade pack against live hermes_memory: feeding a finding
            # back verbatim flagged two `observed` neighbours at 0.892 and 0.8679
            # through this heuristic alone, and an exact duplicate is agreement.
            elif _CONFLICT_NEGATION_HEURISTIC and new_neg != neighbor_neg:
                is_conflict = True

            if is_conflict:
                conflicts.append({
                    "neighbor_id": neighbor_id,
                    "neighbor_type": neighbor_type,
                    "score": round(float(hit.score), 4),
                })

        return conflicts
    except Exception as exc:
        logger.debug("_detect_conflicts: fail-open on exception: %s", exc)
        return []


def _write_conflict(investigation_id: str, finding_id_a: str, neighbor_id: str) -> str:
    """Append a conflict record to conflicts.jsonl and return its id."""
    conflict = {
        "id": str(uuid.uuid4()),
        "investigation_id": investigation_id,
        "finding_id_a": finding_id_a,
        "finding_id_b": neighbor_id,
        "detected_at": _now(),
        "status": "open",
        "resolution": None,
    }
    path = _inv_dir(investigation_id) / "conflicts.jsonl"
    _append_jsonl(path, conflict)
    return conflict["id"]
# Entity node helpers (object permanence across findings)
# ---------------------------------------------------------------------------

# Regex patterns for named-entity extraction from finding text
_CAPITALIZED_PHRASE_RE = re.compile(r'\b([A-Z][a-zA-Z0-9]*(?:[ \t][A-Z][a-zA-Z0-9]*)+)\b')
_QUOTED_PHRASE_RE = re.compile(r'"([^"]{2,80})"')
_IP_ADDR_RE = re.compile(r'\b\d{1,3}(?:\.\d{1,3}){3}\b')
_HOSTNAME_ENTITY_RE = re.compile(
    r'\b[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?'
    r'(?:\.[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?)*'
    r'(?:\.(?:local|corp|internal|lan|dev|test|net|com|io|org))\b', re.I
)
_URL_ENTITY_RE = re.compile(r'https?://[^\s"\'<>;]+', re.I)

_ENTITY_STOP_WORDS = frozenset({
    "The", "This", "That", "These", "Those", "There", "They", "Then",
    "When", "What", "With", "From", "Into", "True", "False", "None",
    "HTTP", "JSON", "SQL", "API", "URL", "ID",
})


def _classify_named_entity(name: str) -> str:
    """Heuristic type classifier for a named entity string."""
    if _IP_ADDR_RE.match(name):
        return "location"
    if _URL_ENTITY_RE.match(name):
        return "system"
    low = name.lower()
    if any(kw in low for kw in ("server", "service", "system", "db", "database",
                                 "cluster", "host", "node", "api", "gateway",
                                 "azure", "aws", "gcp", "cloud")):
        return "system"
    # Two-word capitalized names that look like people
    parts = name.split()
    if len(parts) == 2 and all(p[0].isupper() and p[1:].islower() for p in parts):
        return "person"
    return "concept"


def _extract_named_entities(text: str) -> list:
    """
    Heuristic extraction of named entities from finding text.
    Returns list of {name, type} dicts.  No LLM call — fail-open.
    """
    try:
        candidates = []

        # Capitalized multi-word phrases
        for m in _CAPITALIZED_PHRASE_RE.finditer(text):
            phrase = m.group(1).strip()
            if phrase and phrase not in _ENTITY_STOP_WORDS and len(phrase) >= 3:
                candidates.append(phrase)

        # Things in double quotes
        for m in _QUOTED_PHRASE_RE.finditer(text):
            phrase = m.group(1).strip()
            if phrase and len(phrase) >= 2:
                candidates.append(phrase)

        # IP addresses
        for ip in _IP_ADDR_RE.findall(text):
            candidates.append(ip)

        # Hostnames / FQDNs
        for host in _HOSTNAME_ENTITY_RE.findall(text):
            candidates.append(host)

        # URLs
        for url in _URL_ENTITY_RE.findall(text):
            candidates.append(url)

        # Deduplicate preserving order
        seen = set()
        result = []
        for name in candidates:
            key = name.lower()
            if key not in seen:
                seen.add(key)
                result.append({"name": name, "type": _classify_named_entity(name)})
        return result
    except Exception:
        return []


def _update_entities_jsonl(investigation_id: str, finding_id: str, text: str) -> None:
    """
    Merge extracted named entities into entities.jsonl for the investigation.
    Creates new entity records or updates existing ones (fuzzy name match).
    Fail-open — all exceptions are silently swallowed.
    """
    try:
        inv_path = MEMORY_DIR / investigation_id
        if not inv_path.exists():
            return

        entities_path = inv_path / "entities.jsonl"
        extracted = _extract_named_entities(text)
        if not extracted:
            return

        now = _now()

        # Read existing entities
        existing = _read_jsonl(entities_path)

        # Build a lookup: lowercase name → index in existing list
        name_index = {}
        for i, ent in enumerate(existing):
            name_index[ent.get("name", "").lower()] = i

        changed = False
        for item in extracted:
            name = item["name"]
            etype = item["type"]
            name_lower = name.lower()

            # Fuzzy match: substring in either direction
            match_idx = None
            if name_lower in name_index:
                match_idx = name_index[name_lower]
            else:
                for existing_name_lower, idx in name_index.items():
                    if name_lower in existing_name_lower or existing_name_lower in name_lower:
                        match_idx = idx
                        break

            if match_idx is not None:
                ent = existing[match_idx]
                if finding_id not in ent.get("finding_refs", []):
                    ent.setdefault("finding_refs", []).append(finding_id)
                    ent["last_seen"] = now
                    changed = True
            else:
                new_ent = {
                    "entity_id": str(uuid.uuid4()),
                    "name": name,
                    "type": etype,
                    "aliases": [],
                    "first_seen": now,
                    "last_seen": now,
                    "finding_refs": [finding_id],
                }
                existing.append(new_ent)
                name_index[name_lower] = len(existing) - 1
                changed = True

        if not changed:
            return

        # Atomic rewrite: temp file + rename
        tmp_path = entities_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            for ent in existing:
                f.write(json.dumps(ent) + "\n")
        tmp_path.replace(entities_path)
    except Exception as exc:
        logger.debug("_update_entities_jsonl: entity extraction/merge/write failed (fail-open): %r", exc)
        pass  # fail-open — never crash investigation_store


# ---- Tool: investigation_store ----

# --------------------------------------------------------------------------- #
# investigation_store internals
#
# Split out of the tool body so each stage is separately readable and testable.
# Every helper preserves the tool's original behaviour exactly, including the
# verbatim error strings, which are part of the tool's response contract.
# --------------------------------------------------------------------------- #
_STORE_FINDING_TYPES = {"observed", "inferred", "assumed", "gap", "procedure"}
_STORE_CONFIDENCES = {"high", "medium", "low"}
_STORE_TIERS = {"hot", "warm", "cold"}
_CONFIDENCE_TO_NUMERIC = {"high": 0.9, "medium": 0.6, "low": 0.3}


def _store_validate(finding_type: str, confidence: str, tier: str, resolution: str) -> Optional[str]:
    """Return the tool's error JSON for the first invalid argument, else None."""
    if finding_type not in _STORE_FINDING_TYPES:
        return json.dumps({"error": "finding_type must be one of: observed, inferred, assumed, gap, procedure"})
    if confidence not in _STORE_CONFIDENCES:
        return json.dumps({"error": "confidence must be one of: high, medium, low"})
    if tier not in _STORE_TIERS:
        return json.dumps({"error": "tier must be one of: hot, warm, cold"})
    if resolution not in _RESOLUTION_STATES:
        return json.dumps({"error": "resolution must be one of: open, fixed, intentional, wontfix, superseded"})
    return None


def _store_numeric_confidence(confidence: str, numeric_confidence: float | None) -> float:
    """Caller-supplied confidence (clamped to 0..1), else derived from the label."""
    if numeric_confidence is None:
        return _CONFIDENCE_TO_NUMERIC.get(confidence, 0.6)
    try:
        return max(0.0, min(1.0, float(numeric_confidence)))
    except (TypeError, ValueError):
        return _CONFIDENCE_TO_NUMERIC.get(confidence, 0.6)


def _store_build_finding(investigation_id, finding_type, text, source, confidence, tags,
                         derived_from, numeric_confidence, procedure_preconditions,
                         procedure_steps, procedure_postconditions, valid_from,
                         valid_until, authored_by, tier, resolution, code_refs):
    """Build the finding record. Returns (finding, error_json); one is always None.

    The only failure mode is a derived_from id with no matching parent, which the
    tool reports rather than storing a dangling lineage link.
    """
    ts_now = _now()
    finding = {
        "id": str(uuid.uuid4()),
        "investigation_id": investigation_id,
        "ts": ts_now,
        "created_at_ts": int(datetime.now(timezone.utc).timestamp()),
        "record_type": finding_type,   # "observed" | "inferred" | "assumed" | "gap"
        "type": finding_type,          # kept for backwards compat with existing JSONL
        "text": text,
        "source": source,
        "confidence": confidence,
        "numeric_confidence": _store_numeric_confidence(confidence, numeric_confidence),
        "tags": [t.strip() for t in (
            ",".join(tags) if isinstance(tags, list) else (tags or "")
        ).split(",") if t.strip()],
        "valid_from": valid_from if valid_from is not None else ts_now,
        "valid_until": valid_until,
        "authored_by": authored_by or "",
        "tier": tier,
        "resolution": resolution,
    }

    derived = _normalize_derived_from(derived_from)
    if derived:
        existing_ids = {f["id"] for f in _read_jsonl(_inv_dir(investigation_id) / "findings.jsonl") if "id" in f}
        unknown = [pid for pid in derived if pid not in existing_ids]
        if unknown:
            return None, json.dumps({"error": f"derived_from contains unknown parent id(s): {unknown}. Verify the parent findings exist before linking."})
        finding["derived_from"] = derived

    finding["entities"] = _extract_entities(text)

    # Staleness: stamp sha256 of any referenced code file so a later re-hash can flag
    # the finding stale once that file changes. Best-effort + fail-open (no refs / an
    # unreadable file -> omit; never error).
    try:
        refs = _compute_code_refs(text, code_refs)
        if refs:
            finding["code_refs"] = refs
    except Exception as exc:  # noqa: BLE001
        logger.debug("investigation_store: code-ref hashing failed (fail-open): %r", exc)

    if finding_type == "procedure":
        finding["procedure_meta"] = {
            "preconditions": procedure_preconditions or "",
            "steps": procedure_steps or "",
            "postconditions": procedure_postconditions or "",
            "success_count": 0,
            "attempt_count": 0,
        }
    return finding, None


def _store_commit(investigation_id: str, manifest: dict, finding: dict,
                  finding_type: str, text: str, tier: str) -> None:
    """Append the finding and update the manifest under the per-investigation lock."""
    lock_path = _inv_dir(investigation_id) / ".lock"
    with open(lock_path, "w") as lock_fh:
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
        try:
            _append_jsonl(_inv_dir(investigation_id) / "findings.jsonl", finding)
            manifest["finding_counts"][finding_type] = manifest["finding_counts"].get(finding_type, 0) + 1
            # Update hot-tier manifest notes
            if tier == "hot":
                snippet = text[:200]
                notes = manifest.get("notes") or ""
                manifest["notes"] = (notes + "; " + snippet) if notes else snippet
            _save_manifest(manifest)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)


def _store_index(investigation_id: str, finding: dict, finding_type: str,
                 text: str, source: str, confidence: str, tier: str) -> bool:
    """Fan the finding out to Mnemosyne, Qdrant, the event log and the graph.

    Returns whether Mnemosyne accepted it. Cold tier skips Qdrant indexing; the
    graph mirror is tier-agnostic, since the relationship graph carries findings
    regardless of index tier.
    """
    mnemo_stored = _mnemo_remember(
        text,
        importance={"high": 0.9, "medium": 0.7, "low": 0.5}.get(confidence, 0.6),
        metadata={
            "investigation_id": investigation_id,
            "record_type": finding_type,
            "source": source,
            "confidence": confidence,
            "tags": finding["tags"],
            "finding_id": finding["id"],
        },
    )
    if tier != "cold":
        _qdrant_upsert(finding["id"], text, finding)
    _event_log_append({
        "op": "store",
        "investigation_id": investigation_id,
        "finding_id": finding["id"],
        "finding_type": finding_type,
        "confidence": confidence,
        "tier": tier,
    })
    _mirror_finding_to_ladybug(finding, investigation_id)
    _autolink_finding_to_ladybug(finding)
    return mnemo_stored


def _store_conflicts(investigation_id: str, finding: dict) -> tuple:
    """Detect conflicts with existing findings. Fail-open: never blocks a store.

    Returns (detected, conflicting_finding_id, conflict_id).
    """
    try:
        conflicts = _detect_conflicts(investigation_id, finding)
        if conflicts:
            first = conflicts[0]
            conflict_id = _write_conflict(investigation_id, finding["id"], first["neighbor_id"])
            return True, first["neighbor_id"], conflict_id
    except Exception as exc:
        logger.debug("investigation_store: conflict detection failed (fail-open): %s", exc)
    return False, None, None


@mcp.tool()
def investigation_store(
    investigation_id: str,
    finding_type: str,
    text: str,
    source: str,
    confidence: str = "medium",
    tags: Optional[str] = None,
    derived_from: str | list[str] | None = None,
    numeric_confidence: float | None = None,
    procedure_preconditions: Optional[str] = None,
    procedure_steps: Optional[str] = None,
    procedure_postconditions: Optional[str] = None,
    valid_from: Optional[str] = None,
    valid_until: Optional[str] = None,
    authored_by: Optional[str] = None,
    tier: str = "warm",
    resolution: str = "open",
    code_refs: str | list[str] | None = None,
) -> str:
    """
    Record a finding in the investigation.

    Args:
        investigation_id: Investigation identifier.
        finding_type: One of: observed, inferred, assumed, gap, procedure.
                      observed  — from a direct tool response; cite source and key values.
                      inferred  — reasoned from observations but not directly stated.
                      assumed   — working hypothesis with no current evidence.
                      gap       — something that should be checked but hasn't been.
                      procedure — a reusable step-by-step procedure (runbook/playbook entry).
        text: The finding. For observed, include enough detail to reproduce the
              query (table name, time range, key field values).
        source: Tool or data source this came from (e.g. sentinel__run_kql_query).
        confidence: high / medium / low.
        tags: Optional comma-separated tags, or a list of tag strings.
              Both forms are accepted (e.g. "lateral_movement,phishing" or
              ["lateral_movement", "phishing"]).
        derived_from: Optional finding id(s) (or claim strings) this finding
                      builds on — a single id or a list. Recorded as a
                      forward-derivation link so a later hallucination retraction
                      (memory_retract) can follow the lineage and clean up
                      everything built on a false fact. Omit when the finding
                      stands alone.
        numeric_confidence: Optional float in [0.0, 1.0]. When omitted, auto-derived
                            from string confidence: high→0.9, medium→0.6, low→0.3.
                            Values outside [0,1] are clamped.
        procedure_preconditions: (procedure only) Comma-separated or natural-language
                                 preconditions that must be true before running the procedure.
        procedure_steps: (procedure only) Numbered steps as a string.
        procedure_postconditions: (procedure only) Expected outcomes after the procedure.
        valid_from: ISO8601 timestamp from which this finding is valid. Defaults
                    to the current time when the finding is stored. Use to record
                    a finding that was true at an earlier point in time.
        valid_until: ISO8601 timestamp at which this finding ceased to be valid,
                     or null (default) meaning it is currently believed to be true.
                     Set when a finding has a known expiry or is superseded.
        authored_by: Optional agent_id of the agent storing this finding.
                     Used with investigation ACL to filter findings per agent.
        tier: Memory tier — "hot", "warm", or "cold". Default "warm".
              hot  — indexed in Qdrant AND summarized in manifest notes (instantly in-context).
              warm — indexed in Qdrant only (default, searchable).
              cold — stored in JSONL only, NOT indexed in Qdrant (archived).
        resolution: Lifecycle state of the finding. One of:
              open (default) — active/unresolved; the normal state for a new finding.
              fixed          — the issue has been addressed/remediated.
              intentional    — reviewed and deemed acceptable/by-design.
              wontfix        — acknowledged but will not be actioned.
              superseded     — replaced by a newer finding.
              The three resolved states (fixed/intentional/wontfix) let
              exclusion-aware grounding auto-skip handled items on re-audit.
              Findings stored before this field existed read as "open".
        code_refs: Optional comma-separated (or list of) file paths this finding
                   refers to, e.g. "mcp/server.py,mcp/verify.py". Authoritative when
                   provided: pass ``[]`` / ``""`` to assert "no refs" (text is NOT
                   parsed). Only when omitted entirely (None) are refs best-effort
                   parsed from ``text`` (tokens like "path/file.py:12").
                   Each readable file's current sha256 is stamped so investigation_load
                   / investigation_search can flag the finding ``stale`` once that file
                   changes. Fully optional + fail-open: unreadable paths are skipped.

    Returns:
        JSON: {"stored": true, "finding_id": "<uuid>", "type": "<finding_type>",
               "mnemo_stored": true|false, "tier": "<tier>"}
        On error: {"error": "<message>"}

    Note on arg name: the second positional parameter is ``finding_type``, NOT
    ``record_type``.  Call as:
        investigation_store(inv_id, "observed", "text", "source", "high")
    """
    manifest = _load_manifest(investigation_id)
    if not manifest:
        return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

    resolution = str(resolution or "open").lower()
    invalid = _store_validate(finding_type, confidence, tier, resolution)
    if invalid:
        return invalid

    finding, invalid = _store_build_finding(
        investigation_id, finding_type, text, source, confidence, tags, derived_from,
        numeric_confidence, procedure_preconditions, procedure_steps,
        procedure_postconditions, valid_from, valid_until, authored_by, tier,
        resolution, code_refs,
    )
    if invalid:
        return invalid

    _store_commit(investigation_id, manifest, finding, finding_type, text, tier)
    mnemo_stored = _store_index(investigation_id, finding, finding_type, text, source, confidence, tier)
    conflict_detected, conflicting_finding_id, conflict_id = _store_conflicts(investigation_id, finding)

    # Update the in-process session hints ring buffer so memory_hints can
    # surface this finding immediately without re-scanning JSONL.
    _session_hints_push(investigation_id, {
        "finding_id": finding["id"],
        "text": text,
        "source": source,
        "record_type": finding_type,
        "ts": finding["ts"],
        "created_at_ts": finding["created_at_ts"],
    })
    # Background entity extraction — fail-open, never blocks the response
    _update_entities_jsonl(investigation_id, finding["id"], text)

    result = {
        "stored": True,
        "finding_id": finding["id"],
        "type": finding_type,
        "mnemo_stored": mnemo_stored,
        "conflict_detected": conflict_detected,
        "tier": tier,
    }
    if conflict_detected:
        result["conflicting_finding_id"] = conflicting_finding_id
        result["conflict_id"] = conflict_id

    return json.dumps(result, indent=2)


@mcp.tool()
def finding_resolve(
    investigation_id: str,
    finding_id: str,
    resolution: str,
    note: Optional[str] = None,
) -> str:
    """
    Mark a finding's lifecycle state in place (open -> fixed / intentional / wontfix /
    superseded, or back to open). Completes the lifecycle gap: findings could only get
    a resolution at store time, never afterwards.

    findings.jsonl is append-only, so this records an append-only update record in a
    sibling ``finding_updates.jsonl``; investigation_load / investigation_search fold it
    in last-write-wins. The original finding is never rewritten and no data is lost.

    Args:
        investigation_id: Investigation that owns the finding.
        finding_id: ID of the finding to resolve.
        resolution: One of open, fixed, intentional, wontfix, superseded.
        note: Optional free-text note explaining the resolution.

    Returns:
        JSON: {"resolved": true, "finding_id": ..., "resolution": ...}
        On error: {"error": "<message>"}
    """
    manifest = _load_manifest(investigation_id)
    if not manifest:
        return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

    res = str(resolution or "").lower()
    if res not in _RESOLUTION_STATES:
        return json.dumps({"error": "resolution must be one of: open, fixed, intentional, wontfix, superseded"})

    # Confirm the finding exists so we don't record an orphan resolution.
    # Uses the codebase-wide _read_jsonl (whole-file read, tolerant of a
    # partial last line) — same semantics as every other findings.jsonl reader.
    findings = _read_jsonl(_inv_dir(investigation_id) / "findings.jsonl")
    if not any(isinstance(f, dict) and str(f.get("id", "")) == str(finding_id) for f in findings):
        return json.dumps({"error": f"Finding '{finding_id}' not found in investigation '{investigation_id}'."})

    record = {
        "record_type": "resolution",
        "investigation_id": investigation_id,
        "finding_id": str(finding_id),
        "resolution": res,
        "note": note or "",
        "ts": _now(),
    }
    try:
        _append_jsonl(_finding_updates_path(investigation_id), record)
    except Exception as exc:  # noqa: BLE001 — fail-open: never raise out of a tool
        logger.warning("finding_resolve append failed (fail-open): %r", exc)
        return json.dumps({"error": f"Could not record resolution: {exc}"})
    _event_log_append({
        "op": "finding_resolve",
        "investigation_id": investigation_id,
        "finding_id": str(finding_id),
        "resolution": res,
    })
    return json.dumps({
        "resolved": True,
        "finding_id": str(finding_id),
        "resolution": res,
    }, indent=2)


# ---- Tool: procedure_attempt ----

@mcp.tool()
def procedure_attempt(
    investigation_id: str,
    finding_id: str,
    success: bool,
) -> str:
    """
    Record an attempt (pass or fail) against a procedure-type finding.

    Increments attempt_count on the finding's procedure_meta, and if success is
    True also increments success_count.  The findings.jsonl file is rewritten
    atomically (write to a temp file then rename) so no data is lost on crash.

    Args:
        investigation_id: Investigation that owns the finding.
        finding_id: ID of the procedure finding to update.
        success: True if the procedure succeeded on this attempt, False otherwise.

    Returns:
        JSON: {"finding_id": "<id>", "success_count": int, "attempt_count": int,
               "success_rate": float}
        On error: {"error": "<message>"}
    """
    try:
        findings_path = _inv_dir(investigation_id) / "findings.jsonl"
        if not findings_path.exists():
            return json.dumps({"error": f"No findings file for investigation '{investigation_id}'."})

        findings = _read_jsonl(findings_path)
        target = None
        for f in findings:
            if f.get("id") == finding_id:
                target = f
                break

        if target is None:
            return json.dumps({"error": f"Finding '{finding_id}' not found in investigation '{investigation_id}'."})

        if target.get("record_type") != "procedure" and target.get("type") != "procedure":
            return json.dumps({"error": f"Finding '{finding_id}' is not a procedure-type finding."})

        if "procedure_meta" not in target:
            target["procedure_meta"] = {
                "preconditions": "",
                "steps": "",
                "postconditions": "",
                "success_count": 0,
                "attempt_count": 0,
            }

        target["procedure_meta"]["attempt_count"] = target["procedure_meta"].get("attempt_count", 0) + 1
        if success:
            target["procedure_meta"]["success_count"] = target["procedure_meta"].get("success_count", 0) + 1

        attempt_count = target["procedure_meta"]["attempt_count"]
        success_count = target["procedure_meta"]["success_count"]
        success_rate = round(success_count / attempt_count, 4) if attempt_count > 0 else 0.0

        # Atomic rewrite: write to temp file then rename
        import tempfile as _tempfile
        tmp_fd, tmp_path = _tempfile.mkstemp(dir=str(findings_path.parent), suffix=".jsonl.tmp")
        try:
            with os.fdopen(tmp_fd, "w") as tmp_fh:
                for f in findings:
                    tmp_fh.write(json.dumps(f) + "\n")
            os.replace(tmp_path, str(findings_path))
        except Exception:
            try:
                os.unlink(tmp_path)
            except Exception as exc:
                logger.debug("procedure_attempt: fail-open swallow: %r", exc)
            raise

        # Update Qdrant payload for the finding
        try:
            _qdrant_upsert(finding_id, target.get("text", ""), target)
        except Exception as exc:
            logger.debug("procedure_attempt: qdrant upsert failed: %s", exc)

        return json.dumps({
            "finding_id": finding_id,
            "success_count": success_count,
            "attempt_count": attempt_count,
            "success_rate": success_rate,
        }, indent=2)

    except Exception as exc:
        logger.warning("procedure_attempt: unexpected error: %s", exc)
        return json.dumps({"error": str(exc)})


# ---- Tool: procedure_search ----

@mcp.tool()
def procedure_search(
    query: str,
    investigation_id: Optional[str] = None,
    limit: int = 5,
) -> str:
    """
    Search for procedure-type findings matching a query.

    Searches Qdrant for findings with record_type == "procedure".  When
    investigation_id is provided, results are filtered to that investigation.
    Falls back to a keyword scan of local JSONL files when Qdrant is unavailable.

    Args:
        query: Natural language query describing the procedure you need.
        investigation_id: Optional — limit search to a single investigation.
        limit: Max number of results to return (default 5).

    Returns:
        JSON: {"procedures": [{"finding_id", "text", "source", "success_rate",
               "procedure_meta", "investigation_id", "score"}], "count": int}
        On error: {"error": "<message>", "procedures": [], "count": 0}
    """
    try:
        client, _col = _get_qdrant()
        procedures = []

        if client is not None:
            try:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                must_conditions = [
                    FieldCondition(key="record_type", match=MatchValue(value="procedure")),
                ]
                if investigation_id:
                    must_conditions.append(
                        FieldCondition(key="investigation_id", match=MatchValue(value=investigation_id))
                    )
                qfilter = Filter(must=must_conditions)
                hits = _qdrant_search_collection(
                    query,
                    collection_name=QDRANT_COLLECTION_PREFIX,
                    limit=limit,
                    query_filter=qfilter,
                )
                for h in hits:
                    pm = h.get("procedure_meta", {})
                    attempt_count = pm.get("attempt_count", 0) if pm else 0
                    success_count = pm.get("success_count", 0) if pm else 0
                    success_rate = round(success_count / attempt_count, 4) if attempt_count > 0 else 0.0
                    procedures.append({
                        "finding_id": h.get("id", ""),
                        "text": h.get("text", ""),
                        "source": h.get("source", ""),
                        "investigation_id": h.get("investigation_id", ""),
                        "success_rate": success_rate,
                        "procedure_meta": pm,
                        "score": h.get("score", 0.0),
                    })
            except Exception as exc:
                logger.warning("procedure_search: qdrant search failed, falling back to JSONL scan: %s", exc)
                client = None  # trigger fallback below

        if client is None:
            # Fallback: scan JSONL files directly
            query_lower = query.lower()
            inv_dirs = []
            if investigation_id:
                d = MEMORY_DIR / investigation_id
                if d.is_dir():
                    inv_dirs = [d]
            else:
                try:
                    inv_dirs = [d for d in MEMORY_DIR.iterdir() if d.is_dir()]
                except Exception:
                    inv_dirs = []

            candidates = []
            for inv_dir_path in inv_dirs:
                findings_path = inv_dir_path / "findings.jsonl"
                if not findings_path.exists():
                    continue
                try:
                    for f in _read_jsonl(findings_path):
                        if f.get("record_type") == "procedure" or f.get("type") == "procedure":
                            text = f.get("text", "")
                            if query_lower in text.lower():
                                pm = f.get("procedure_meta", {})
                                attempt_count = pm.get("attempt_count", 0) if pm else 0
                                success_count = pm.get("success_count", 0) if pm else 0
                                success_rate = round(success_count / attempt_count, 4) if attempt_count > 0 else 0.0
                                candidates.append({
                                    "finding_id": f.get("id", ""),
                                    "text": text,
                                    "source": f.get("source", ""),
                                    "investigation_id": f.get("investigation_id", ""),
                                    "success_rate": success_rate,
                                    "procedure_meta": pm,
                                    "score": 0.0,
                                })
                except Exception as exc:
                    logger.debug("procedure_search: error reading %s: %s", findings_path, exc)
            procedures = candidates[:limit]

        return json.dumps({"procedures": procedures, "count": len(procedures)}, indent=2)

    except Exception as exc:
        logger.warning("procedure_search: unexpected error: %s", exc)
        return json.dumps({"error": str(exc), "procedures": [], "count": 0})


# ---- Tool: investigation_as_of ----

# ---- Tool: investigation_note ----

# ---- Tool: investigation_reflect ----

# ---- Tool: reflection_loop_seed ----

@mcp.tool()
def reflection_loop_seed(
    investigation_id: str = REFLECTION_DEFAULT_INVESTIGATION,
    session_events_limit: int = 250,
    process_logs_limit: int = 120,
    reset_queue: bool = False,
) -> str:
    """
    Seed the bounded self-reflection queue from Copilot local artifacts.

    The queue is persisted under ``$HERMES_MEMORY_DIR/_reflection-loop/state.json``.
    This call only enqueues file targets — it does not parse files or write findings.
    Use ``reflection_loop_tick`` to process queued items in small batches.
    """
    session_events_limit = max(1, min(int(session_events_limit), 2000))
    process_logs_limit = max(1, min(int(process_logs_limit), 2000))

    _ensure_investigation_exists(
        investigation_id,
        title="Copilot self-reflection loop",
        context=(
            "Continuous bounded mining of ~/.copilot/temp_ingest, "
            "~/.copilot/session-state/*/events.jsonl, ~/.copilot/logs/process-*.log, "
            "and ~/.claude/projects/**/*.jsonl (Claude Code)"
        ),
    )

    state = _load_reflection_state()
    state["investigation_id"] = investigation_id
    if reset_queue:
        state["queue"] = []
        state["processed"] = {}

    queue: list[dict] = list(state.get("queue") or [])
    processed: dict = dict(state.get("processed") or {})
    existing_keys = {
        f"{item.get('kind','')}|{item.get('path','')}"
        for item in queue
    }
    existing_keys.update(processed.keys())

    candidates: list[dict] = []
    temp_ingest = Path.home() / ".copilot" / "temp_ingest" / "payload.json"
    if temp_ingest.exists():
        candidates.append({"kind": "temp_ingest", "path": str(temp_ingest)})

    session_files = sorted(
        (Path.home() / ".copilot" / "session-state").glob("*/events.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:session_events_limit]
    candidates.extend({"kind": "session_event", "path": str(p)} for p in session_files)

    process_logs = sorted(
        (Path.home() / ".copilot" / "logs").glob("process-*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:process_logs_limit]
    candidates.extend({"kind": "process_log", "path": str(p)} for p in process_logs)

    # Claude Code source paths: ~/.claude/projects/**/*.jsonl
    claude_code_files = sorted(
        (Path.home() / ".claude" / "projects").glob("**/*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:session_events_limit]
    candidates.extend({"kind": "claude_code_event", "path": str(p)} for p in claude_code_files)

    candidates.sort(key=lambda item: _reflection_queue_priority(item.get("kind")))

    added = 0
    for item in candidates:
        key = f"{item['kind']}|{item['path']}"
        if key in existing_keys:
            continue
        queue.append(item)
        existing_keys.add(key)
        added += 1

    state["queue"] = queue
    _save_reflection_state(state)
    return json.dumps({
        "queued_added": added,
        "queue_size": len(queue),
        "investigation_id": investigation_id,
        "sources": {
            "temp_ingest": int(temp_ingest.exists()),
            "session_events_candidates": len(session_files),
            "process_logs_candidates": len(process_logs),
            "claude_code_events_candidates": len(claude_code_files),
        },
        "state_file": str(REFLECTION_STATE_FILE),
    }, indent=2)


# ---- Tool: reflection_loop_status ----

@mcp.tool()
def reflection_loop_status(queue_preview: int = 8) -> str:
    """
    Return current queue and aggregate stats for the self-reflection loop.
    """
    queue_preview = max(0, min(int(queue_preview), 50))
    state = _load_reflection_state()
    queue = list(state.get("queue") or [])
    processed = dict(state.get("processed") or {})
    stats = dict(state.get("stats") or {})
    preview = queue[:queue_preview]
    return json.dumps({
        "investigation_id": state.get("investigation_id"),
        "queue_size": len(queue),
        "processed_count": len(processed),
        "stats": stats,
        "last_tick": state.get("last_tick"),
        "updated_at": state.get("updated_at"),
        "queue_preview": preview,
        "state_file": str(REFLECTION_STATE_FILE),
    }, indent=2)


def _reflection_filter_signatures(
    raw: dict[str, int],
    observations: dict[str, int],
) -> tuple[dict[str, int], int, int]:
    """Drop signatures already observed ``REFLECTION_SIGNATURE_OBSERVE_LIMIT`` times.

    Returns ``(visible, suppressed_hits, suppressed_signatures)``. ``observations``
    is mutated in place: every signature that stays visible has its observation
    count bumped by one.
    """
    visible: dict[str, int] = {}
    suppressed_hits = 0
    suppressed_signatures = 0
    for sig, count in raw.items():
        observed_count = int(observations.get(sig) or 0)
        if observed_count >= REFLECTION_SIGNATURE_OBSERVE_LIMIT:
            suppressed_hits += count
            suppressed_signatures += 1
            continue
        visible[sig] = count
        observations[sig] = observed_count + 1
    return visible, suppressed_hits, suppressed_signatures


def _reflection_signature_summary(
    visible: dict[str, int],
    suppressed_signatures: int,
    suppressed_hits: int,
) -> str:
    """Render the top-3 visible signatures, with a saturation suffix when suppressing."""
    summary = ", ".join(
        f"{sig} ({count})" for sig, count in list(visible.items())[:3]
    ) or "none"
    if suppressed_signatures:
        summary += (
            f"; saturated={suppressed_signatures} signatures "
            f"({suppressed_hits} hits)"
        )
    return summary


def _reflection_store_finding(
    investigation_id: str,
    finding_type: str,
    text: str,
    confidence: str,
    tags: str,
) -> bool:
    """Store one reflection-loop finding; True when it was actually written.

    ``source`` is always ``"reflection_loop_tick"``. No try/except on purpose —
    a failing store must propagate exactly as it does inline.
    """
    return bool(json.loads(investigation_store(
        investigation_id=investigation_id,
        finding_type=finding_type,
        text=text,
        source="reflection_loop_tick",
        confidence=confidence,
        tags=tags,
    )).get("stored"))


# ---- Tool: reflection_loop_tick ----

def _reflection_item_finding(kind, path, summary, raw_errors, raw_warnings,
                             error_observations, warning_observations, stats,
                             investigation_id, low_signal_session_events) -> bool:
    """Store one reflection finding for a processed queue item.

    Returns True when a finding was written. Session events with no errors or
    warnings are recorded as low-signal for the batched roll-up instead, and
    return False -- this was a `continue` at the tail of the caller's loop, so
    returning early is equivalent.

    Mutates `stats` (suppression counters) and `low_signal_session_events` in
    place, matching the original inline behaviour.
    """
    if kind == "session_event" and not raw_errors and not raw_warnings:
        low_signal_session_events.append({
            "path": path,
            "lines_scanned": int(summary.get("lines_scanned") or 0),
            "bytes_scanned": int(summary.get("bytes_scanned") or 0),
            "events": summary.get("events") or {},
            "tools": summary.get("tools") or {},
        })
        return False

    visible_errors, suppressed_error_hits, suppressed_error_signatures = (
        _reflection_filter_signatures(raw_errors, error_observations)
    )
    visible_warnings, suppressed_warning_hits, suppressed_warning_signatures = (
        _reflection_filter_signatures(raw_warnings, warning_observations)
    )
    stats["error_signatures_suppressed"] += suppressed_error_hits
    stats["warning_signatures_suppressed"] += suppressed_warning_hits
    top_error = _reflection_signature_summary(
        visible_errors, suppressed_error_signatures, suppressed_error_hits
    )
    top_warning = _reflection_signature_summary(
        visible_warnings, suppressed_warning_signatures, suppressed_warning_hits
    )
    finding_text = (
        f"reflection_loop_tick processed {kind} target={path}; "
        f"lines={summary.get('lines_scanned', 0)} bytes={summary.get('bytes_scanned', 0)}; "
        f"sampling={summary.get('sampling_mode', 'full')}; "
        f"top_events={summary.get('events', {})}; top_tools={summary.get('tools', {})}; "
        f"errors={top_error}; warnings={top_warning}."
    )
    return bool(_reflection_store_finding(
        investigation_id=investigation_id,
        finding_type="observed",
        text=finding_text,
        confidence="low",
        tags="self-reflection,loop-tick,artifact-mining,unreceipted-observed",
    ))


def _reflection_batch_low_signal(investigation_id: str, low_signal_session_events: list[dict]) -> int:
    """Store the batched low-signal session_event roll-up finding.

    Returns the number to add to ``findings_written`` (0 or 1). Caller must
    only invoke this when ``store_item_findings and low_signal_session_events``.
    """
    event_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    for entry in low_signal_session_events:
        event_counts.update(entry.get("events") or {})
        tool_counts.update(entry.get("tools") or {})
    sample_paths = [e["path"] for e in low_signal_session_events[:3]]
    low_signal_text = (
        f"reflection_loop_tick batched low-signal session_event files count={len(low_signal_session_events)}; "
        f"total_lines={sum(e['lines_scanned'] for e in low_signal_session_events)} "
        f"total_bytes={sum(e['bytes_scanned'] for e in low_signal_session_events)}; "
        f"top_events={dict(event_counts.most_common(8))}; top_tools={dict(tool_counts.most_common(8))}; "
        f"sample_paths={sample_paths}."
    )
    if _reflection_store_finding(
        investigation_id=investigation_id,
        finding_type="observed",
        text=low_signal_text,
        confidence="low",
        tags="self-reflection,loop-tick,artifact-mining,unreceipted-observed,batched-low-signal",
    ):
        return 1
    return 0


def _reflection_batch_error_signature(investigation_id: str, batch_error_signatures: Counter) -> int:
    """Store the batch dominant-error-signature inference finding.

    Returns the number to add to ``findings_written`` (0 or 1). Caller must
    only invoke this when ``store_item_findings and batch_error_signatures``.
    """
    sig, count = batch_error_signatures.most_common(1)[0]
    infer_text = (
        "Batch dominant error signature suggests reliability hotspot: "
        f"{sig} (count={count}) in latest processed artifacts."
    )
    if _reflection_store_finding(
        investigation_id=investigation_id,
        finding_type="inferred",
        text=infer_text,
        confidence="medium",
        tags="self-reflection,error-cluster,inference",
    ):
        return 1
    return 0


def _reflection_requeue_dropped(
    queue: list, dropped_items: list[dict], investigation_id: str, store_item_findings: bool
) -> int:
    """Re-queue dropped items and record a gap finding for each, in place.

    Mutates ``queue`` via ``.append`` -- never rebinds it, since the caller
    persists the same list object via ``state["queue"] = queue``. Returns the
    number to add to ``findings_written``.
    """
    added = 0
    for dropped in dropped_items:
        queue.append(dropped)
        if store_item_findings:
            d_kind = str(dropped.get("kind") or "")
            d_path = str(dropped.get("path") or "")
            gap_text = (
                f"reflection_loop_tick could not process item kind={d_kind} path={d_path}; "
                "item re-queued for future processing."
            )
            if _reflection_store_finding(
                investigation_id=investigation_id,
                finding_type="gap",
                text=gap_text,
                confidence="low",
                tags="self-reflection,loop-tick,dropped-item,re-queued",
            ):
                added += 1
    return added


@mcp.tool()
def reflection_loop_tick(
    max_items: int = 3,
    max_lines_per_file: int = 4000,
    store_item_findings: bool = True,
) -> str:
    """
    Process a small queue batch for self-reflection and store findings.

    Designed to avoid passive burn:
    - bounded by ``max_items`` and ``max_lines_per_file``
    - deterministic parsing only (no LLM pass)
    - writes findings through ``investigation_store`` (JSONL + Mnemosyne + Qdrant)
    """
    max_items = max(1, min(int(max_items), 20))
    max_lines_per_file = max(50, min(int(max_lines_per_file), 20000))
    state = _load_reflection_state()
    queue = list(state.get("queue") or [])
    if not queue:
        return json.dumps({
            "processed_items": 0,
            "queue_size": 0,
            "message": "Queue is empty. Run reflection_loop_seed first.",
            "state_file": str(REFLECTION_STATE_FILE),
        }, indent=2)

    investigation_id = str(state.get("investigation_id") or REFLECTION_DEFAULT_INVESTIGATION)
    _ensure_investigation_exists(
        investigation_id,
        title="Copilot self-reflection loop",
        context="Bounded deterministic queue-based Copilot artifact reflection.",
    )

    processed = dict(state.get("processed") or {})
    stats = dict(state.get("stats") or {})
    stats.setdefault("files_processed", 0)
    stats.setdefault("lines_scanned", 0)
    stats.setdefault("errors_seen", 0)
    stats.setdefault("warnings_seen", 0)
    stats.setdefault("bytes_scanned", 0)
    stats.setdefault("error_signatures_suppressed", 0)
    stats.setdefault("warning_signatures_suppressed", 0)
    stats.setdefault("error_signature_observations", {})
    stats.setdefault("warning_signature_observations", {})
    error_observations = _prune_signature_observations(stats.get("error_signature_observations") or {})
    warning_observations = _prune_signature_observations(stats.get("warning_signature_observations") or {})
    findings_written = 0
    batch_error_signatures: Counter[str] = Counter()
    batch_warning_signatures: Counter[str] = Counter()
    item_reports: list[dict] = []
    low_signal_session_events: list[dict[str, Any]] = []
    dropped_items: list[dict] = []

    for _ in range(min(max_items, len(queue))):
        next_index = min(
            range(len(queue)),
            key=lambda idx: _reflection_queue_priority(queue[idx].get("kind")),
        )
        item = queue.pop(next_index)
        kind = str(item.get("kind") or "")
        path = str(item.get("path") or "")
        summary = _process_reflection_item(kind, path, max_lines=max_lines_per_file)
        item_reports.append(summary)
        if summary.get("status") != "processed":
            dropped_items.append(item)
            continue

        key = f"{kind}|{path}"
        stats["files_processed"] += 1
        stats["lines_scanned"] += int(summary.get("lines_scanned") or 0)
        stats["bytes_scanned"] += int(summary.get("bytes_scanned") or 0)
        raw_errors = {str(k): int(v) for k, v in (summary.get("errors") or {}).items()}
        raw_warnings = {str(k): int(v) for k, v in (summary.get("warnings") or {}).items()}
        stats["errors_seen"] += sum(raw_errors.values())
        stats["warnings_seen"] += sum(raw_warnings.values())
        processed[key] = {
            "kind": kind,
            "path_hash": _hash_path(path),
            "processed_at": _now(),
            "lines_scanned": summary.get("lines_scanned"),
        }
        batch_error_signatures.update(raw_errors)
        batch_warning_signatures.update(raw_warnings)

        if store_item_findings and _reflection_item_finding(
            kind, path, summary, raw_errors, raw_warnings, error_observations,
            warning_observations, stats, investigation_id, low_signal_session_events,
        ):
            findings_written += 1

    if store_item_findings and low_signal_session_events:
        findings_written += _reflection_batch_low_signal(investigation_id, low_signal_session_events)

    if store_item_findings and batch_error_signatures:
        findings_written += _reflection_batch_error_signature(investigation_id, batch_error_signatures)

    stats["last_error_signatures"] = [
        {"signature": sig, "count": count}
        for sig, count in batch_error_signatures.most_common(5)
    ]
    stats["last_warning_signatures"] = [
        {"signature": sig, "count": count}
        for sig, count in batch_warning_signatures.most_common(5)
    ]
    stats["error_signature_observations"] = _prune_signature_observations(error_observations)
    stats["warning_signature_observations"] = _prune_signature_observations(warning_observations)

    findings_written += _reflection_requeue_dropped(
        queue, dropped_items, investigation_id, store_item_findings
    )

    state["stats"] = stats
    state["processed"] = processed
    state["queue"] = queue
    state["last_tick"] = {
        "ts": _now(),
        "processed_items": len(item_reports),
        "findings_written": findings_written,
        "remaining_queue": len(queue),
    }
    _save_reflection_state(state)

    return json.dumps({
        "investigation_id": investigation_id,
        "processed_items": len(item_reports),
        "findings_written": findings_written,
        "remaining_queue": len(queue),
        "batch": item_reports,
        "stats": stats,
    }, indent=2)


# ---- Tool: investigation_search ----

def _search_resolution_maps(rows: list[dict]) -> tuple[dict[str, str], dict[str, list]]:
    """Build authoritative ``finding_id -> resolution`` / ``-> code_refs`` maps.

    Scoped to only the investigations that actually produced ``rows``, so cost is
    bounded regardless of how many investigations exist. Fail-open: a failed build
    returns two empty dicts, and the caller falls back to per-row payload values.
    """
    resolution_map: dict[str, str] = {}
    coderefs_map: dict[str, list] = {}
    try:
        _invs_in_results = {str(r.get("investigation_id", "")) for r in rows}
        _invs_in_results.discard("")
        for _inv in _invs_in_results:
            for _f in _read_jsonl(MEMORY_DIR / _inv / "findings.jsonl"):
                _fid = str(_f.get("id", ""))
                if _fid:
                    resolution_map[_fid] = str(_f.get("resolution") or "open").lower()
                    _crefs = _f.get("code_refs")
                    if isinstance(_crefs, list) and _crefs:
                        coderefs_map[_fid] = _crefs
            # Fold in append-only resolution overrides (finding_resolve) last-write-wins.
            for _ofid, _ores in _load_resolution_overrides(_inv).items():
                resolution_map[_ofid] = _ores
    except Exception as exc:  # fail-open — never block search on the map build
        logger.debug("resolution map build failed, using per-row values: %r", exc)
        return {}, {}
    return resolution_map, coderefs_map


def _search_row_resolution(row: dict, resolution_map: dict[str, str]) -> str:
    """Resolve one search row's resolution, preferring the JSONL-derived map."""
    fid = str(row.get("finding_id") or row.get("id") or "")
    if fid and fid in resolution_map:
        return resolution_map[fid]
    return str(row.get("resolution") or "open").lower()


def _search_retraction_scope(
    investigation_id: Optional[str],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Precompute retracted finding ids (and their texts) per investigation in scope.

    Fail-safe: on any error both maps come back empty, which means nothing is
    filtered rather than everything.
    """
    retracted_by_inv: dict[str, set[str]] = {}
    retracted_text_by_inv: dict[str, set[str]] = {}
    try:
        scope_invs = (
            [investigation_id] if investigation_id
            else ([p.name for p in MEMORY_DIR.iterdir() if p.is_dir()] if MEMORY_DIR.exists() else [])
        )
        for _inv in scope_invs:
            rids = _load_retracted_ids(_inv)
            if not rids:
                continue
            retracted_by_inv[_inv] = rids
            texts: set[str] = set()
            for f in _read_jsonl(MEMORY_DIR / _inv / "findings.jsonl"):
                if str(f.get("id", "")) in rids:
                    t = str(f.get("text", "") or "").strip()
                    if t:
                        texts.add(t)
            retracted_text_by_inv[_inv] = texts
    except Exception as exc:  # fail-safe — never block search on filtering
        logger.debug("retraction scope precompute failed, not filtering: %r", exc)
        return {}, {}
    return retracted_by_inv, retracted_text_by_inv


def _search_row_is_retracted(
    row: dict,
    rids_by_inv: dict[str, set[str]],
    texts_by_inv: dict[str, set[str]],
) -> bool:
    """True when a search row names — or repeats the text of — a retracted finding.

    An empty ``rids_by_inv`` means no filtering is in effect (either nothing is
    retracted in scope, the precompute failed, or the caller asked for retracted
    rows to be included).
    """
    if not rids_by_inv:
        return False
    inv = str(row.get("investigation_id", ""))
    rids = rids_by_inv.get(inv)
    rtexts = texts_by_inv.get(inv)
    if not rids and not rtexts:
        return False
    rid = row.get("finding_id") or row.get("id")
    if rid is not None and str(rid) in (rids or set()):
        return True
    text = str(row.get("text", "") or "").strip()
    if text and rtexts and text in rtexts:
        return True
    return False


def _search_normalize_filters(min_confidence: str, resolution: Optional[str]) -> tuple[str, Optional[str]]:
    """Fail-open normalisation of investigation_search's resolution and
    min_confidence filters. An unknown value logs a warning and falls back
    (resolution -> None, min_confidence -> "low") rather than raising.
    """
    # Normalise resolution filter (fail-open: an unknown value disables the filter).
    resolution = str(resolution).lower() if resolution else None
    if resolution is not None and resolution not in _RESOLUTION_STATES:
        logger.warning(
            "investigation_search: unknown resolution %r — ignoring filter; valid "
            "values: %s", resolution, ", ".join(sorted(_RESOLUTION_STATES))
        )
        resolution = None

    # Normalise confidence floor so callers passing "High" or "MEDIUM" are not
    # silently ignored by the lowercase dict lookup downstream.
    min_confidence = str(min_confidence or "low").lower()
    if min_confidence not in _CONFIDENCE_RANK:
        logger.warning(
            "investigation_search: unknown min_confidence %r — ignoring filter; "
            "valid values: low, medium, high", min_confidence
        )
        min_confidence = "low"

    return min_confidence, resolution


def _search_apply_confidence_floor(rows: list[dict], min_confidence: str) -> list[dict]:
    """Apply the confidence floor post-hoc so mnemosyne rows — which carry no
    "confidence" field — don't silently undermine the caller's filter when
    mnemosyne satisfies the result limit before Qdrant is queried.
    """
    if min_confidence and min_confidence in _CONFIDENCE_RANK and min_confidence != "low":
        floor = _CONFIDENCE_RANK[min_confidence]
        return [
            r for r in rows
            # Normalise stored confidence to lowercase — mnemosyne-sourced rows may
            # carry mixed-case values; without .lower() "High" maps to rank 0.
            if _CONFIDENCE_RANK.get(str(r.get("confidence", "low")).lower(), 0) >= floor
        ]
    return rows


def _search_annotate_rows(rows: list[dict]) -> None:
    """Surface each row's resolution and staleness in place.

    Rows sourced from mnemosyne carry no resolution, so we build an authoritative
    finding_id -> resolution map from JSONL — scoped to only the investigations
    that actually produced rows, so cost is bounded regardless of how many
    investigations exist. Absent record -> "open". Fail-open (a failed map build
    just falls back to the per-row payload value, defaulting to "open").
    """
    _resolution_map, _coderefs_map = _search_resolution_maps(rows)

    for r in rows:
        r["resolution"] = _search_row_resolution(r, _resolution_map)
        # Staleness: rows from mnemo/qdrant don't carry code_refs, so consult the
        # JSONL-derived map. Only set ``stale`` when the finding has usable refs.
        _rfid = str(r.get("finding_id") or r.get("id") or "")
        _refs = _coderefs_map.get(_rfid)
        if _refs:
            _st = _finding_is_stale({"code_refs": _refs})
            if _st is not None:
                r["stale"] = _st


def _search_empty_response(qdrant: dict, mnemo_enabled: bool) -> str:
    """Build the JSON payload for an empty investigation_search result set."""
    qdrant_avail = bool(os.environ.get("QDRANT_URL", ""))
    # Only use rag_required mode when Qdrant itself is unavailable.
    # Empty results with Qdrant available is a normal no-match response.
    if not qdrant_avail or (qdrant.get("reason") == "qdrant_unavailable"):
        return json.dumps({
            "mode": "rag_required",
            "reason": "qdrant_unavailable",
            "results": [],
            "qdrant_enabled": qdrant_avail,
            "error": "RAG_REQUIRED: Qdrant unavailable. Check QDRANT_URL and QDRANT_API_KEY.",
        }, indent=2)
    return json.dumps({
        "mode": "no_matches",
        "reason": str(qdrant.get("reason") or "no_matches"),
        "results": [],
        "qdrant_enabled": qdrant_avail,
        "mnemo_status": {
            "enabled": mnemo_enabled,
            "bank": _mnemo_bank() if mnemo_enabled else None,
            "match_count": 0,
        },
    }, indent=2)


@mcp.tool()
def investigation_search(
    query: str,
    investigation_id: Optional[str] = None,
    limit: int = 10,
    include_retracted: bool = False,
    min_confidence: str = "low",
    resolution: Optional[str] = None,
) -> str:
    """
    Search findings by similarity.
    Resolution order: Mnemosyne recall (primary) → Qdrant semantic/hybrid
    enrichment (secondary, when needed) → local keyword scoring fallback.

    Soft-retracted findings (a known hallucination + its contaminated lineage)
    are excluded from results by default and counted under
    ``excluded_retracted``. Pass ``include_retracted=True`` to surface them on
    demand — the data is never lost, only filtered.

    Args:
        query: Search query string.
        investigation_id: Limit to one investigation, or omit to search all.
        limit: Maximum results to return (default 10).
        include_retracted: Include soft-retracted findings (default False).
        resolution: Optional lifecycle filter. When set to one of
                    open/fixed/intentional/wontfix/superseded, only findings in that
                    resolution state are returned. Omit (default) to return all.
                    Each result row surfaces its ``resolution`` (absent -> "open").

    Returns:
        JSON list of matching findings with investigation context.
    """
    min_confidence, resolution = _search_normalize_filters(min_confidence, resolution)

    # Precompute retracted finding ids per investigation in scope. A row is
    # filtered when it names a finding id (or matches text of one) that is
    # retracted. Fail-safe: an empty map means nothing is filtered.
    _retracted_by_inv, _retracted_text_by_inv = (
        _search_retraction_scope(investigation_id) if not include_retracted else ({}, {})
    )

    _excluded_retracted = {"n": 0}

    _, recall_fn = _get_mnemo_funcs()
    mnemo_enabled = recall_fn is not None
    mnemo_rows = _mnemo_recall(query, top_k=max(limit * 4, 20), investigation_id=investigation_id)

    deduped: list[dict] = []
    seen: set[str] = set()

    def _add_row(row: dict) -> None:
        if _search_row_is_retracted(row, _retracted_by_inv, _retracted_text_by_inv):
            _excluded_retracted["n"] += 1
            return
        key = "|".join([
            str(row.get("investigation_id", "")),
            str(row.get("record_type", "")),
            str(row.get("source", "")),
            str(row.get("text", ""))[:220],
        ])
        if key in seen:
            return
        seen.add(key)
        deduped.append(row)

    for row in sorted(mnemo_rows, key=lambda r: _safe_float(r.get("score", 0.0), default=0.0), reverse=True):
        _add_row(row)

    qdrant = {"ok": False, "reason": "not_attempted", "results": []}
    if len(deduped) < limit:
        qdrant = _qdrant_similarity_search(
            query,
            investigation_id=investigation_id,
            limit=max(limit * 3, 20),     # overfetch to absorb dedup losses
            rerank_top_k=limit,           # but CE only ranks the original limit
            min_confidence=min_confidence if min_confidence != "low" else None,
        )
        if qdrant.get("ok"):
            for row in qdrant.get("results", []):
                _add_row(row)
                if len(deduped) >= limit:
                    break

    deduped = _search_apply_confidence_floor(deduped, min_confidence)

    # Surface each row's resolution and optionally filter by it.
    _search_annotate_rows(deduped)
    if resolution is not None:
        deduped = [r for r in deduped if r.get("resolution") == resolution]

    if not deduped:
        return _search_empty_response(qdrant, mnemo_enabled)

    mode = "mnemo_primary"
    if qdrant.get("ok"):
        mode = f"mnemo+{qdrant.get('reason')}" if mnemo_rows else str(qdrant.get("reason"))

    return json.dumps({
        "mode": mode,
        "results": deduped[: max(1, min(limit, 200))],
        "excluded_retracted": _excluded_retracted["n"],
        "include_retracted": include_retracted,
        "resolution_filter": resolution,
        "mnemo_status": {
            "enabled": mnemo_enabled,
            "bank": _mnemo_bank() if mnemo_enabled else None,
            "match_count": len(mnemo_rows),
        },
        "qdrant_status": {
            "enabled": bool(os.environ.get("QDRANT_URL", "")),
            "available": bool(_get_qdrant()[0] is not None),
            "queried": bool(qdrant.get("reason") != "not_attempted"),
            "reason": qdrant.get("reason"),
            "match_count": len(qdrant.get("results", [])),
        },
    }, indent=2)


# ---- Tool: investigation_pre_answer_check ----

def _pre_answer_lexical_refs(
    claim_tokens, claim_negated: bool, evidence_pool: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Score a claim's tokens against the evidence pool via lexical matching.

    Returns ``(support_refs, contradiction_refs)`` in the append order produced
    by iterating ``evidence_pool`` -- callers rely on this order for the
    downstream ``[:8]`` slice and for reference deduplication.
    """
    claim_support_refs: list[dict] = []
    claim_contradiction_refs: list[dict] = []
    for evid in evidence_pool:
        score = _lexical_match_score(claim_tokens, evid.get("tokens", set()))
        if score < 0.45:
            continue
        evidence_negated = bool(_NEGATION_RE.search(str(evid.get("text", ""))))
        if claim_negated != evidence_negated and score >= 0.5:
            ref = _make_ref(evid, "contradiction", score=score)
            claim_contradiction_refs.append(ref)
        else:
            ref = _make_ref(evid, "support", score=score)
            claim_support_refs.append(ref)
    return claim_support_refs, claim_contradiction_refs


def _pre_answer_chain_confidence(
    investigation_id: str, matched_ids: set[str]
) -> tuple[float | None, str | None]:
    """Compute ``(min_chain_confidence, confidence_summary)`` for the matched evidence.

    Fail-open: any error yields ``(None, None)`` so the pre-answer check itself
    never breaks on a confidence computation.
    """
    min_chain_confidence = None
    confidence_summary = None
    try:
        findings_by_id_for_conf: dict[str, dict] = {
            str(f.get("id", "")): f
            for f in _read_jsonl(_inv_dir(investigation_id) / "findings.jsonl")
            if f.get("id")
        }
        chain_confidences: list[float] = []
        for ev_id in matched_ids:
            if ev_id and ev_id in findings_by_id_for_conf:
                agg = _compute_aggregate_confidence(ev_id, findings_by_id_for_conf)
                chain_confidences.append(agg)
        if chain_confidences:
            min_chain_confidence = round(min(chain_confidences), 6)
            if min_chain_confidence >= 0.8:
                confidence_summary = "high (≥0.8)"
            elif min_chain_confidence >= 0.5:
                confidence_summary = "medium (0.5-0.8)"
            else:
                confidence_summary = "low (<0.5)"
    except Exception as exc:
        logger.debug(
            "investigation_pre_answer_check: chain-confidence computation failed (fail-open): %r", exc
        )
        return (None, None)
    return (min_chain_confidence, confidence_summary)


@mcp.tool()
def investigation_pre_answer_check(
    investigation_id: str,
    claims: str | list[str],
    min_confidence: str = "medium",
    record: bool = True,
) -> str:
    """
    Validate proposed response claims against investigation findings plus recent
    audit receipts. Works in degraded JSONL-only mode when Qdrant is disabled.

    When ``record=True`` (default) each claim verdict is persisted to the
    ``hermes_verdicts`` Qdrant collection so prior-check history and conflict
    detection are available on subsequent calls. Each claim result gains three
    fields: ``verdict_type`` (claim_supported / claim_contradicted /
    claim_unsupported), ``prior_occurrences``, and ``verdict_conflict``.
    """
    manifest = _load_manifest(investigation_id)
    if not manifest:
        return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

    min_confidence = str(min_confidence or "medium").lower()
    if min_confidence not in _CONFIDENCE_RANK:
        return json.dumps({"error": "min_confidence must be one of: low, medium, high"})

    normalized_claims = _normalize_claims(claims)
    if not normalized_claims:
        return json.dumps({"error": "claims must contain at least one non-empty claim"})

    evidence_pool = build_validation_evidence(investigation_id, min_confidence=min_confidence)
    claim_results: list[dict] = []
    matched_refs: list[dict] = []
    matched_ids: set[str] = set()
    unsupported_claims: list[str] = []
    support_count = 0
    contradiction_count = 0

    qdrant_matches_total = 0
    qdrant_errors: list[str] = []
    qdrant_enabled = bool(os.environ.get("QDRANT_URL", ""))
    qdrant_available = False
    qdrant_query_success = False

    for claim in normalized_claims:
        claim_tokens = tokenize(claim)
        claim_negated = bool(_NEGATION_RE.search(claim))
        claim_support_refs, claim_contradiction_refs = _pre_answer_lexical_refs(
            claim_tokens, claim_negated, evidence_pool
        )

        qdrant_refs, qdrant_status = _search_qdrant_claim_evidence(claim, investigation_id, limit=5)
        qdrant_available = qdrant_available or bool(qdrant_status.get("available"))
        if qdrant_status.get("error"):
            qdrant_errors.append(str(qdrant_status["error"]))
        elif qdrant_status.get("query_attempted"):
            qdrant_query_success = True
        qdrant_matches_total += len(qdrant_refs)
        for ref in qdrant_refs:
            score = float(ref.get("score", 0.0))
            if score < _QDRANT_SUPPORT_MIN_SCORE:
                continue
            claim_support_refs.append(_make_ref(ref, "support", score=score))

        if claim_support_refs:
            support_count += 1
        else:
            unsupported_claims.append(claim)
        if claim_contradiction_refs:
            contradiction_count += 1

        seen_in_claim: set[str] = set()
        for ref in claim_support_refs + claim_contradiction_refs:
            ev_id = str(ref.get("evidence_id"))
            if not ev_id or ev_id in seen_in_claim:
                continue
            seen_in_claim.add(ev_id)
            matched_ids.add(ev_id)
            matched_refs.append(ref)

        # Dual retrieval — benign baseline search (CIBER / CHR pattern).
        # Only run when there IS supporting evidence; benign context can never
        # make an already-unsupported claim ambiguous, so the Qdrant queries
        # would be pure waste for those claims.
        benign_context_refs = (
            _search_benign_context_qdrant(claim, investigation_id)
            if claim_support_refs else []
        )

        claim_results.append({
            "claim": claim,
            "supported": bool(claim_support_refs),
            "contradicted": bool(claim_contradiction_refs),
            "ambiguous": bool(claim_support_refs and benign_context_refs),
            "support_refs": claim_support_refs[:8],
            "contradiction_refs": claim_contradiction_refs[:8],
            "benign_context_refs": benign_context_refs,
        })

    unique_errors = sorted(set(qdrant_errors))
    degraded_active, degraded_reason = _qdrant_degraded_mode(
        qdrant_enabled, qdrant_available, unique_errors, qdrant_query_success
    )

    verdict_summary = _record_claim_verdicts(investigation_id, claim_results, record=record)

    # Compute min_chain_confidence and confidence_summary from supporting findings.
    min_chain_confidence, confidence_summary = _pre_answer_chain_confidence(
        investigation_id, matched_ids
    )

    response = {
        "investigation_id": investigation_id,
        "checked_at": _now(),
        "claims_checked": len(normalized_claims),
        "support_count": support_count,
        "contradiction_count": contradiction_count,
        "unsupported_claims": unsupported_claims,
        "matched_evidence_ids": sorted(matched_ids),
        "matched_evidence": matched_refs[:50],
        "claim_results": claim_results,
        "verdict_recording": verdict_summary,
        "min_chain_confidence": min_chain_confidence,
        "confidence_summary": confidence_summary,
        "qdrant_status": {
            "enabled": qdrant_enabled,
            "available": qdrant_available,
            "match_count": qdrant_matches_total,
            "errors": unique_errors,
        },
        "degraded_mode": {
            "active": degraded_active,
            "reason": degraded_reason,
            "fallback": "findings_jsonl+audit_jsonl",
        },
    }
    return json.dumps(response, indent=2)


# ---- Tool: investigation_entity_lookup ----

@mcp.tool()
def investigation_entity_lookup(
    entity: str,
    entity_type: str = "auto",
    investigation_id: Optional[str] = None,
    limit: int = 30,
) -> str:
    """
    Find every finding that mentions a specific observable — IP, email, hostname,
    file hash, or CVE — across one investigation or the entire memory store.

    Uses Qdrant payload indexes for O(1) lookup when available, with a full
    JSONL scan as a fallback. Results are grouped by investigation so you can
    immediately see whether the entity has appeared in prior cases.

    This is the primary tool for keeping conclusions evidence-bound: before
    asserting "this IP is malicious" or "this user is compromised", call this
    to see what the memory actually contains about them.

    Args:
        entity: The observable to search for (e.g. "198.51.100.5",
                "user@example.com", "workstation-01.corp",
                "d41d8cd98f00b204e9800998ecf8427e", "CVE-2024-1234").
        entity_type: One of "ip", "email", "hostname", "hash", "cve", or
                     "auto" (default). "auto" infers the type from the value.
        investigation_id: Scope to a single investigation. Omit to search
                          all investigations (cross-case entity graph).
        limit: Max findings to return (default 30).

    Returns:
        JSON with the entity, detected type, total match count, results grouped
        by investigation_id, and the retrieval method used (qdrant | jsonl_fallback).
    """
    entity = entity.strip()
    if not entity:
        return json.dumps({"error": "entity must not be empty"})

    if entity_type == "auto":
        entity_type = _detect_entity_type(entity)
    if entity_type not in _ENTITY_FIELD_MAP:
        return json.dumps({
            "error": f"entity_type must be one of: {', '.join(_ENTITY_FIELD_MAP)} or 'auto'"
        })

    findings, method = _entity_lookup_cascade(entity, entity_type, investigation_id, limit)

    # Group by investigation and build compact summaries.
    by_inv: dict[str, list[dict]] = {}
    for f in findings:
        inv = str(f.get("investigation_id") or "unknown")
        by_inv.setdefault(inv, []).append(_summarise_finding(f))

    return json.dumps({
        "entity": entity,
        "entity_type": entity_type,
        "scope": investigation_id or "all_investigations",
        "total_findings": len(findings),
        "investigations_count": len(by_inv),
        "retrieval": method,
        "by_investigation": by_inv,
    }, indent=2, default=str)


# ---- Tool: entity_list ----

@mcp.tool()
def entity_list(
    investigation_id: str,
    entity_type: Optional[str] = None,
) -> str:
    """
    List all named entities extracted from findings in an investigation.

    Entities are extracted automatically during investigation_store from
    capitalized phrases, quoted strings, IP addresses, hostnames, and URLs.
    This gives a quick overview of all actors, systems, and concepts that
    have appeared across findings.

    Args:
        investigation_id: The investigation to list entities for.
        entity_type: Optional filter — one of "person", "system", "concept",
                     "location", "other". Omit to return all types.

    Returns:
        JSON: {"entities": [{entity_id, name, type, finding_count}], "count": int}
        On error: {"error": "<message>"}
    """
    try:
        inv_path = MEMORY_DIR / investigation_id
        if not inv_path.exists():
            return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

        entities_path = inv_path / "entities.jsonl"
        raw_entities = _read_jsonl(entities_path)

        results = []
        for ent in raw_entities:
            if entity_type and ent.get("type") != entity_type:
                continue
            results.append({
                "entity_id": ent.get("entity_id"),
                "name": ent.get("name"),
                "type": ent.get("type"),
                "finding_count": len(ent.get("finding_refs", [])),
            })

        # Sort by finding_count descending for relevance
        results.sort(key=lambda e: e["finding_count"], reverse=True)

        return json.dumps({"entities": results, "count": len(results)}, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---- Tool: entity_timeline ----

@mcp.tool()
def entity_timeline(
    investigation_id: str,
    entity_id: str,
) -> str:
    """
    Show a chronological timeline of all findings that mention a specific entity.

    Use this to reconstruct the narrative arc of how an actor, system, or
    concept evolved across the investigation — from first mention through
    latest observation.

    Args:
        investigation_id: The investigation containing the entity.
        entity_id: The entity_id returned by entity_list.

    Returns:
        JSON: {
            "entity": {entity_id, name, type, aliases, first_seen, last_seen},
            "timeline": [{finding_id, ts, text, record_type}],
            "count": int
        }
        On error: {"error": "<message>"}
    """
    try:
        inv_path = MEMORY_DIR / investigation_id
        if not inv_path.exists():
            return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

        entities_path = inv_path / "entities.jsonl"
        raw_entities = _read_jsonl(entities_path)

        target_entity = None
        for ent in raw_entities:
            if ent.get("entity_id") == entity_id:
                target_entity = ent
                break

        if target_entity is None:
            return json.dumps({"error": f"Entity '{entity_id}' not found in investigation '{investigation_id}'."})

        finding_refs = set(target_entity.get("finding_refs", []))

        findings_path = inv_path / "findings.jsonl"
        all_findings = _read_jsonl(findings_path)

        timeline = []
        for f in all_findings:
            if f.get("id") in finding_refs:
                timeline.append({
                    "finding_id": f.get("id"),
                    "ts": f.get("ts"),
                    "text": f.get("text", ""),
                    "record_type": f.get("record_type") or f.get("type"),
                })

        # Sort chronologically by ts
        timeline.sort(key=lambda x: x.get("ts") or "")

        entity_summary = {
            "entity_id": target_entity.get("entity_id"),
            "name": target_entity.get("name"),
            "type": target_entity.get("type"),
            "aliases": target_entity.get("aliases", []),
            "first_seen": target_entity.get("first_seen"),
            "last_seen": target_entity.get("last_seen"),
        }

        return json.dumps({
            "entity": entity_summary,
            "timeline": timeline,
            "count": len(timeline),
        }, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---- Tool: investigation_related_cases ----

@mcp.tool()
def investigation_related_cases(
    entities: str | list[str],
    entity_type: str = "auto",
    limit_per_entity: int = 5,
) -> str:
    """
    Find prior investigations that dealt with the same entities as a new alert.

    Call this before opening a new investigation to check whether the entities
    involved have appeared in past cases. A prior resolution as benign changes
    the triage posture; a prior escalation adds urgency. Prevents investigators
    from treating known-good or known-bad entities as novel unknowns.

    Uses entity payload indexes for O(1) lookup — no embedding required.

    Args:
        entities: One or more observables to search for. May be a single string
                  or a list. Each is looked up independently.
        entity_type: "ip", "email", "hostname", "hash", "cve", or "auto"
                     (default). "auto" infers type from each value independently.
        limit_per_entity: Max findings to fetch per entity (default 5).

    Returns:
        JSON with each entity, its detected type, and related investigations
        grouped by case — including finding counts and sample texts.
    """
    if isinstance(entities, str):
        entities = [entities]
    entities = [e.strip() for e in entities if e and e.strip()]
    if not entities:
        return json.dumps({"error": "entities must not be empty"})

    results: list[dict] = []
    for entity in entities[:10]:  # cap total entities to avoid runaway queries
        etype = entity_type if entity_type != "auto" else _detect_entity_type(entity)
        findings, method = _entity_lookup_cascade(entity, etype, None, limit_per_entity * 4)

        # Group by investigation, exclude findings with no investigation context
        by_inv: dict[str, list[dict]] = {}
        for f in findings:
            inv = str(f.get("investigation_id") or "").strip()
            if inv:
                by_inv.setdefault(inv, []).append(_summarise_finding(f))

        results.append({
            "entity": entity,
            "entity_type": etype,
            "related_investigation_count": len(by_inv),
            "retrieval": method,
            "related_investigations": {
                inv_id: {
                    "finding_count": len(flist),
                    "sample": flist[:2],
                }
                for inv_id, flist in sorted(
                    by_inv.items(),
                    key=lambda kv: len(kv[1]),
                    reverse=True,
                )
            },
        })

    return json.dumps({
        "entities_queried": len(results),
        "results": results,
    }, indent=2, default=str)


# ---- Tool: investigation_finding_provenance ----

# ---- Tool: investigation_evidence_precheck ----

@mcp.tool()
def investigation_evidence_precheck(
    investigation_id: str,
    proposed_query: str,
    min_similarity: float = 0.4,
) -> str:
    """
    Lightweight duplicate-call avoidance helper. Checks if similar evidence
    already exists in findings/audit logs (and Qdrant when available).
    """
    manifest = _load_manifest(investigation_id)
    if not manifest:
        return json.dumps({"error": f"Investigation '{investigation_id}' not found."})
    if not proposed_query or not str(proposed_query).strip():
        return json.dumps({"error": "proposed_query is required"})

    try:
        min_similarity = float(min_similarity)
    except (TypeError, ValueError):
        return json.dumps({"error": "min_similarity must be a numeric value between 0.1 and 1.0"})
    min_similarity = max(0.1, min(min_similarity, 1.0))
    query = str(proposed_query).strip()
    query_tokens = tokenize(query)

    evidence_pool = build_validation_evidence(investigation_id, min_confidence="low")
    lexical_matches: list[dict] = []
    for record in evidence_pool:
        score = _lexical_match_score(query_tokens, record.get("tokens", set()))
        if score < min_similarity:
            continue
        lexical_matches.append(_make_ref(record, "similar", score=score))

    lexical_matches.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    qdrant_refs, qdrant_status = _search_qdrant_claim_evidence(query, investigation_id, limit=5)
    qdrant_enabled = bool(os.environ.get("QDRANT_URL", ""))
    qdrant_available = bool(qdrant_status.get("available"))
    qdrant_errors = [qdrant_status["error"]] if qdrant_status.get("error") else []
    qdrant_query_success = bool(qdrant_status.get("query_attempted")) and not qdrant_errors
    degraded_active, degraded_reason = _qdrant_degraded_mode(
        qdrant_enabled, qdrant_available, qdrant_errors, qdrant_query_success
    )

    qdrant_matches = []
    for ref in qdrant_refs:
        score = float(ref.get("score", 0.0))
        if score < max(min_similarity, _QDRANT_PRECHECK_MIN_SCORE):
            continue
        qdrant_matches.append(_make_ref(ref, "similar", score=score))
    combined = lexical_matches[:10] + qdrant_matches
    seen: set[str] = set()
    deduped: list[dict] = []
    for item in combined:
        key = str(item.get("evidence_id"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return json.dumps({
        "investigation_id": investigation_id,
        "proposed_query": query,
        "has_similar_evidence": bool(deduped),
        "similar_evidence_count": len(deduped),
        "similar_evidence": deduped[:10],
        "qdrant_status": {
            "enabled": qdrant_enabled,
            "available": qdrant_available,
            "match_count": len(qdrant_matches),
            "errors": qdrant_errors,
        },
        "degraded_mode": {
            "active": degraded_active,
            "reason": degraded_reason,
            "fallback": "findings_jsonl+audit_jsonl",
        },
    }, indent=2)


# ---- Tool: investigation_list ----

# ---- Tool: investigation_share ----

# ---- Tool: investigation_unshare ----

# ---- Tool: audit_log ----

@mcp.tool()
def audit_log(
    tool_name: str,
    inputs_json: str,
    output: str,
    investigation_id: Optional[str] = None,
    embedding_text: Optional[str] = None,
) -> str:
    """
    Record a tool call and its full output to the audit log. Called as a
    post-call hook after any MCP tool invocation to maintain a complete
    record of what was queried and what was returned — sufficient to
    reconstruct the investigation without re-calling the API.

    Writes to the global daily audit log and, if investigation_id is
    provided, also to the investigation-specific audit log. Mirrors into
    Mnemosyne (primary memory sink) when available, and indexes into Qdrant
    (secondary semantic index) when available.

    Args:
        tool_name: Name of the tool called (e.g. sentinel__run_kql_query).
        inputs_json: JSON-encoded inputs passed to the tool.
        output: Full tool output — not truncated. Include the complete
                response so the investigation can be reconstructed from
                memory alone.
        investigation_id: Associate with a specific investigation if known.
        embedding_text: Optional natural-language summary to vectorize
                        instead of the raw output. Construct this as an
                        investigation note: entity names, key values,
                        and what the result means. When omitted, the first
                        3000 chars of output are used. Providing a
                        hand-crafted embedding_text dramatically improves
                        semantic search recall — see CLAUDE.md for
                        per-tool templates.

    Returns:
        JSON confirmation.
    """
    entry = {
        "ts": _now(),
        "created_at_ts": int(datetime.now(timezone.utc).timestamp()),
        "tool": tool_name,
        "investigation_id": investigation_id,
        "inputs": inputs_json,
        "output": output,
    }
    entry["entities"] = _extract_entities(embedding_text or output[:2000])

    # Global daily audit log
    audit_dir = MEMORY_DIR.parent / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _append_jsonl(audit_dir / f"{date_str}.jsonl", entry)

    # Investigation-scoped audit log
    if investigation_id and _load_manifest(investigation_id):
        _append_jsonl(_inv_dir(investigation_id) / "audit.jsonl", entry)

    embed_text = embedding_text or f"{tool_name}: {output[:2000]}"
    mnemo_stored = _mnemo_remember(
        embed_text,
        importance=0.65,
        metadata={
            "record_type": "audit",
            "tool": tool_name,
            "investigation_id": investigation_id,
            "source": "audit_log",
        },
    )

    # Qdrant (best-effort secondary index)
    qdrant_indexed = False
    if _get_qdrant()[0] is not None:
        _qdrant_upsert(
            str(uuid.uuid4()),
            embed_text,
            {**entry, "record_type": "audit"},
        )
        qdrant_indexed = True

    return json.dumps({
        "logged": True,
        "tool": tool_name,
        "ts": entry["ts"],
        "mnemo_stored": mnemo_stored,
        "qdrant_indexed": qdrant_indexed,
    })


# ---- Tool: memory_self_check ----

def _verdict_view(v) -> dict:
    """Compact JSON view of a Verdict for tool output."""
    return {
        "verdict_type": v.verdict_type,
        "decision": v.decision,
        "confidence": v.confidence,
        "rationale": v.rationale,
        "refs": v.refs,
        "excerpt": v.subject_excerpt,
        "subject_signature": v.subject_signature,
    }


def _record_verdicts(verdicts: list) -> bool:
    """Record verdicts into the hermes_verdicts qdrant collection. Fail-open.

    Builds a ``VerdictEngine`` over a ``QdrantBackend`` wired to the server's real
    ``_embed`` (so memory verdicts get genuine dense vectors). Any failure —
    qdrant unavailable, embedding down, backend error — degrades to ``False``
    and never raises out of the tool.
    """
    if not verdicts:
        return False
    client, _col = _get_qdrant()
    if client is None:
        return False
    try:
        import asyncio

        from memcheck import EmlConfig, QdrantBackend, VerdictEngine

        backend = QdrantBackend(
            client=client,
            collection="hermes_verdicts",
            embed=_embed,
            vector_name="dense",
        )
        engine = VerdictEngine(backend, EmlConfig())

        async def _record_all() -> None:
            for v in verdicts:
                # Engine.record is itself fail-open; embed here so a real dense
                # vector is stored rather than the backend's fallback.
                await engine.record(v, _embed(v.subject_excerpt))

        # FastMCP dispatches sync tools inline on a running event loop — asyncio.run()
        # raises RuntimeError in that context. Delegate to a daemon thread when needed.
        try:
            asyncio.get_running_loop()
            _exc: list[Exception] = []

            def _run_thread() -> None:
                try:
                    asyncio.run(_record_all())
                except Exception as e:
                    _exc.append(e)

            t = threading.Thread(target=_run_thread, daemon=True)
            t.start()
            t.join()
            if _exc:
                raise _exc[0]
        except RuntimeError:
            asyncio.run(_record_all())
        return True
    except Exception as exc:  # fail-open — recording must not fail the tool
        logger.debug("memcheck verdict recording failed, degrading: %r", exc)
        return False


@mcp.tool()
def memory_self_check(
    investigation_id: Optional[str] = None,
    checks: str = "provenance,contradiction",
    record: bool = True,
    llm_verify: bool = False,
) -> str:
    """
    Run the server's advisory memory self-check over stored findings.

    Checks the investigation's own memory against the rules the server's reasoning
    discipline states but cannot otherwise verify:
      - provenance     — every ``observed`` finding should trace to an audit
                         receipt; flags ones that don't (``unsupported_observed``).
      - contradiction  — surfaces pairs of findings that appear to disagree
                         (negation-polarity mismatch on overlapping text).

    It also surfaces ``hallucination_candidates``: findings that are BOTH
    unsupported (no receipt) AND contradicted by a receipted finding — the
    strongest "stored a fact testing later disproved" signal. These carry a hint
    to run ``memory_retract``; they are NEVER auto-retracted.

    This is **advisory only**: verdicts annotate and surface. Nothing is hidden,
    deleted, or mutated — findings.jsonl stays append-only. Verdicts are computed
    purely over the JSONL, so the check still works when qdrant is unavailable;
    recording into the shared ``hermes_verdicts`` collection is best-effort.

    Args:
        investigation_id: Investigation to check, or omit to check all.
        checks: Comma-separated subset of: provenance, contradiction.
        record: When true and qdrant is reachable, record verdicts for recall.
        llm_verify: Opt into the deep_think -> loci semantic contradiction path —
            an embedding subject gate + LLM polarity judge that supersedes the
            lexical token-overlap heuristic (drops its false positives, catches
            the semantic negations it misses). Default False keeps the check
            pure/offline. Also enabled by env MEMCHECK_LLM_CONTRADICTION=1.
            Fail-open: lexical verdicts pass through if embeddings/LLM are down.

    Returns:
        JSON with per-investigation counts and the advisory verdicts.
    """
    llm_verify = llm_verify or os.environ.get(
        "MEMCHECK_LLM_CONTRADICTION", ""
    ).strip().lower() in ("1", "true", "yes", "on")
    requested = {c.strip().lower() for c in (checks or "").split(",") if c.strip()}
    if not requested:
        requested = {"provenance", "contradiction"}
    unknown = requested - {"provenance", "contradiction"}
    if unknown:
        return json.dumps({
            "error": f"unknown checks: {sorted(unknown)}. Valid: provenance, contradiction"
        })

    if investigation_id is not None:
        if not (MEMORY_DIR / investigation_id).exists():
            return json.dumps({"error": f"Investigation '{investigation_id}' not found."})
        targets = [investigation_id]
    elif MEMORY_DIR.exists():
        targets = [p.name for p in MEMORY_DIR.iterdir() if p.is_dir()]
    else:
        targets = []

    qdrant_available = _get_qdrant()[0] is not None
    all_verdicts: list = []
    all_candidates: list[dict] = []
    per_investigation: list[dict] = []

    for inv_id in targets:
        computed = _compute_self_check(inv_id, llm_verify=llm_verify)
        inv_verdicts: list = []
        if "provenance" in requested:
            inv_verdicts.extend(computed["unsupported_observed"])
        if "contradiction" in requested:
            inv_verdicts.extend(computed["contradictions"])
        all_verdicts.extend(inv_verdicts)
        # Hallucination candidates require both provenance + contradiction
        # signals; only surface when both checks ran.
        inv_candidates = (
            computed.get("hallucination_candidates", [])
            if {"provenance", "contradiction"} <= requested
            else []
        )
        all_candidates.extend(inv_candidates)
        entry = {
            "investigation_id": inv_id,
            "counts": {
                "unsupported_observed": sum(
                    1 for v in inv_verdicts if v.verdict_type == "unsupported_observed"
                ),
                "contradiction": sum(
                    1 for v in inv_verdicts if v.verdict_type == "contradiction"
                ),
                "hallucination_candidates": len(inv_candidates),
            },
            "verdicts": [_verdict_view(v) for v in inv_verdicts],
            "hallucination_candidates": inv_candidates,
        }
        # Carry the lane's state next to the count it explains. An
        # unsupported_observed count computed against zero receipts is the
        # investigation's observed-finding count wearing a different name, and
        # reporting it bare invites reading absence of evidence as evidence.
        lane = computed.get("audit_lane") or {}
        if "provenance" in requested and lane.get("status") == "empty":
            entry["audit_lane"] = lane
        per_investigation.append(entry)

    recorded = False
    if record and qdrant_available:
        recorded = _record_verdicts(all_verdicts)

    result = {
        "checked_at": _now(),
        "advisory": (
            "Verdicts annotate and surface only — no finding is hidden, "
            "deleted, or modified."
        ),
        "checks": sorted(requested),
        "counts": {
            "unsupported_observed": sum(
                1 for v in all_verdicts if v.verdict_type == "unsupported_observed"
            ),
            "contradiction": sum(
                1 for v in all_verdicts if v.verdict_type == "contradiction"
            ),
            "hallucination_candidates": len(all_candidates),
        },
        "hallucination_candidates": all_candidates,
        "recorded": recorded,
        "qdrant": "ok" if qdrant_available else "unavailable",
    }
    if investigation_id is not None:
        result["investigation_id"] = investigation_id
        result["verdicts"] = per_investigation[0]["verdicts"] if per_investigation else []
    else:
        result["investigation_ids"] = targets
        result["investigations"] = per_investigation

    return json.dumps(result, indent=2)


# ---- Tool: code_memory_correlate (code -> memory loop) ----

# Code-hallucination rule codes that count as a "real" LH issue on a file. A
# parse failure surfaces as LH000; the LH001/LH003/LH007/LH009 family are the
# AST-detected smells.
_CODE_HALLUCINATION_CODES = {"LH000", "LH001", "LH003", "LH007", "LH009"}


def _suspected_entities_from_text(text: str) -> set[str]:
    """Distinctive entities of ``text`` (typed buckets) as a lowercased set.

    Reuses the server's ``_extract_entities`` + ``_distinctive_entity_set`` so the
    correlation anchor matches exactly what ``memory_retract`` / contagion use.
    """
    try:
        return _distinctive_entity_set(_extract_entities(text or ""))
    except Exception as exc:  # fail-open — extraction must never break correlate
        logger.debug("entity extraction failed for correlate, degrading: %r", exc)
        return set()


def _verdict_symbols(verdicts: list) -> set[str]:
    """Pull flagged ``symbol:<name>`` refs out of code verdicts as a token set.

    ``run_code_checks`` stashes the flagged identifier (when a rule names one)
    as a ``"symbol:<name>"`` ref. These are loop-fuel for exactly this pass —
    correlate them as suspected entities alongside the file's extracted ones.
    """
    out: set[str] = set()
    for v in verdicts or []:
        for ref in getattr(v, "refs", None) or []:
            s = str(ref)
            if s.startswith("symbol:"):
                token = s[len("symbol:"):].strip().lower()
                if token:
                    out.add(token)
    return out


def _code_findings_from_verdicts(verdicts: list) -> list[dict]:
    """Group code verdicts into ``[{file, codes}]`` for the report. Fail-open."""
    by_file: dict[str, list[str]] = {}
    for v in verdicts or []:
        rel = None
        for ref in getattr(v, "refs", None) or []:
            s = str(ref)
            if s.startswith("path:"):
                rel = s[len("path:"):]
                break
        rel = rel or "?"
        code = str(getattr(v, "verdict_type", "") or "")
        bucket = by_file.setdefault(rel, [])
        if code and code not in bucket:
            bucket.append(code)
    return [{"file": f, "codes": sorted(codes)} for f, codes in sorted(by_file.items())]


def _finding_distinctive_set(finding: dict) -> set[str]:
    """Distinctive entities for a finding — its stored ``entities`` or extracted."""
    stored = finding.get("entities")
    if isinstance(stored, dict):
        ents = _distinctive_entity_set(stored)
        if ents:
            return ents
    return _suspected_entities_from_text(str(finding.get("text", "") or ""))


def _correlate_memories(
    investigation_id: str,
    suspected: set[str],
    findings: list[dict],
) -> dict:
    """Resolve which findings overlap ``suspected`` entities (the contamination).

    Reuses the inc3 machinery: findings carrying any suspected entity are seeds,
    qdrant supplies semantic neighbors (fail-open), and ``find_contamination``
    expands the cluster over entity-anchor + semantic + ``derived_from`` links.
    Returns ``{contaminated_ids, reasons, seed_ids, semantic_neighbors}``.
    """
    # Seeds: findings whose distinctive entities intersect the suspected set, or
    # whose lowercased text literally contains a suspected token (catches entities
    # the typed extractor doesn't bucket, e.g. fabricated module names / localhost).
    seeds: list[dict] = []
    seed_ids: list[str] = []
    for f in findings:
        f_ents = _finding_distinctive_set(f)
        ftext = str(f.get("text", "") or "").lower()
        if (suspected & f_ents) or any(tok in ftext for tok in suspected):
            seeds.append(f)
            fid = str(f.get("id", ""))
            if fid and fid not in seed_ids:
                seed_ids.append(fid)

    if not seed_ids:
        return {"contaminated_ids": [], "reasons": {}, "seed_ids": [], "semantic_neighbors": 0}

    semantic_ids: list[str] = []
    try:
        semantic_ids = _semantic_neighbor_ids(seeds, investigation_id)
    except Exception as exc:  # fail-open — semantic scope is best-effort
        logger.debug("semantic neighbor lookup failed in correlate, skipping: %r", exc)
        semantic_ids = []

    # Primary: the LadybugDB graph traversal (semantics identical to find_contamination).
    # Fallback: the in-memory traversal if the graph is unavailable or errors.
    cluster = None
    ks = _get_ladybug()
    if ks:
        try:
            graph_cluster = ks.contamination(
                list(seed_ids),
                min_shared_entities=1,
                semantic_neighbor_ids=semantic_ids,
            )
            if isinstance(graph_cluster, dict) and "contaminated_ids" in graph_cluster:
                cluster = graph_cluster
        except Exception as exc:
            logger.debug("LadybugDB contamination failed, falling back to in-memory: %r", exc)
    if cluster is None:
        try:
            cluster = find_contamination(
                seed_ids,
                findings,
                entities_of=_extract_entities,
                semantic_neighbor_ids=semantic_ids,
                min_shared_entities=1,
            )
        except Exception as exc:  # fail-open — degrade to seeds only
            logger.debug("find_contamination failed in correlate, degrading to seeds: %r", exc)
            cluster = {"contaminated_ids": list(seed_ids), "reasons": {sid: ["seed"] for sid in seed_ids}}

    return {
        "contaminated_ids": cluster.get("contaminated_ids", []),
        "reasons": cluster.get("reasons", {}),
        "seed_ids": seed_ids,
        "semantic_neighbors": len(semantic_ids),
    }


def _correlate_scan_target_file(target_file: str):
    """Scan a target .py file for suspected entities and code-hallucination verdicts.

    Returns (suspected_entities, code_findings, code_note). Every stage is
    fail-open: a missing, non-Python, unreadable, or checker-failing file yields
    an advisory note rather than an error, and correlation proceeds on whatever
    entities were extracted.
    """
    tf = str(target_file).strip()
    path = Path(tf)
    if not path.exists() or not path.is_file():
        return set(), None, f"target_file {tf!r} does not exist or is not a file — skipped"
    if path.suffix != ".py":
        return set(), None, f"target_file {tf!r} is not a .py file — skipped"

    code_note = None
    try:
        content = path.read_text()
    except Exception as exc:  # fail-open — unreadable file is advisory
        logger.debug("could not read target_file %s: %r", tf, exc)
        return set(), None, f"target_file {tf!r} could not be read — skipped"

    suspected = _suspected_entities_from_text(content)

    # tree-sitter: ingest the file's AST into the code graph (fail-open) so its
    # symbols/calls are queryable via code_graph_query and linkable to the memory
    # graph. Targeted symbol suspicion still comes from the code checker's flagged
    # identifiers below, not every symbol.
    try:
        from graph.code_parse import parse_source, detect_lang
        lang = detect_lang(tf)
        ks = _get_ladybug()
        if lang and ks:
            ks.ingest_code([parse_source(tf, content.encode("utf-8", "replace"), lang)])
    except Exception as exc:
        logger.debug("AST ingest for %s failed (fail-open): %r", tf, exc)

    try:
        verdicts = run_code_checks(path)
    except Exception as exc:  # fail-open — checker errors are advisory
        logger.debug("run_code_checks failed for %s, degrading: %r", tf, exc)
        verdicts = []

    lh_verdicts = [
        v for v in verdicts
        if str(getattr(v, "verdict_type", "")) in _CODE_HALLUCINATION_CODES
    ]
    suspected |= _verdict_symbols(lh_verdicts)
    if not lh_verdicts:
        code_note = (
            f"no code-hallucination issues found in {path.name} — "
            "correlating on its extracted entities only (advisory)"
        )
    return suspected, _code_findings_from_verdicts(lh_verdicts), code_note


@mcp.tool()
def code_memory_correlate(
    investigation_id: str,
    target_file: Optional[str] = None,
    entity: Optional[str] = None,
) -> str:
    """
    Link a detected code hallucination to the memories it contaminated.

    The code->memory loop. When generated CODE references a fabricated entity (a
    fake ``localhost`` endpoint, a hallucinated module/symbol) and that same
    entity also seeded or contaminated stored MEMORIES, this surfaces the
    contaminated memory lineage so it can be cleaned up. It bridges part-1 code
    detection (``run_code_checks``) into the memory contagion/retraction machinery
    (``find_contamination`` + the ``memory_retract`` resolution path).

    Resolve the suspected-hallucinated entities one of two ways:
      - ``entity`` given — that string (plus its distinctive entities) is the anchor.
      - ``target_file`` given (an existing ``.py``) — run ``run_code_checks`` to
        confirm code-hallucination verdicts, then anchor on the file's distinctive
        entities (URLs/hosts/identifiers from its content) PLUS any flagged symbol
        from the verdicts. A file with no LH issues is reported as such but still
        correlated on its entities (advisory).

    At least one of ``entity`` / ``target_file`` is required.

    This is strictly **advisory and read-only**: it suggests running
    ``memory_retract`` for review and NEVER mutates, retracts, or creates
    anything on disk or in qdrant. Fail-open: qdrant/file/parse errors degrade to
    a well-formed report rather than raising.

    Args:
        investigation_id: Investigation whose memories to correlate against.
        target_file: Optional path to a generated ``.py`` file to check + anchor on.
        entity: Optional suspected-hallucinated entity string (e.g. a fake host
                or endpoint) to anchor on directly.

    Returns:
        JSON: {investigation_id, suspected_entities, source, code_findings?,
        contaminated_memories: [{finding_id, excerpt, reasons}], already_retracted,
        count, suggestion, advisory:true}.
    """
    if not str(entity or "").strip() and not str(target_file or "").strip():
        return json.dumps({
            "error": "provide at least one of entity or target_file",
        })

    manifest = _load_manifest(investigation_id)
    if not manifest:
        return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

    suspected: set[str] = set()
    source: dict = {}
    code_findings: list[dict] | None = None
    code_note: str | None = None

    # --- Resolve suspected entities from the entity arg. ---
    if str(entity or "").strip():
        ent = entity.strip()
        source["entity"] = ent
        suspected.add(ent.lower())
        suspected |= _suspected_entities_from_text(ent)

    # --- Resolve suspected entities from the target file (+ confirm LH issues). ---
    if str(target_file or "").strip():
        source["target_file"] = str(target_file).strip()
        _tf_suspected, code_findings, code_note = _correlate_scan_target_file(target_file)
        suspected |= _tf_suspected

    suspected = {s for s in suspected if s}
    if not suspected:
        return json.dumps({
            "investigation_id": investigation_id,
            "suspected_entities": [],
            "source": source,
            "code_findings": code_findings,
            "contaminated_memories": [],
            "already_retracted": [],
            "count": 0,
            "note": code_note or "no suspected entities resolved from inputs",
            "suggestion": None,
            "advisory": True,
        }, indent=2)

    findings = _tag_finding_ids(
        _read_jsonl(_inv_dir(investigation_id) / "findings.jsonl"), investigation_id
    )
    retracted = _load_retracted_ids(investigation_id)

    correlation = _correlate_memories(investigation_id, suspected, findings)
    contaminated_ids = correlation["contaminated_ids"]
    reasons = correlation["reasons"]
    by_id = {str(f.get("id", "")): f for f in findings}

    contaminated_memories: list[dict] = []
    already_retracted: list[dict] = []
    for fid in contaminated_ids:
        item = {
            "finding_id": fid,
            "excerpt": redact_excerpt(str(by_id.get(fid, {}).get("text", "") or "")),
            "reasons": reasons.get(fid, []),
        }
        if fid in retracted:
            already_retracted.append(item)
        else:
            contaminated_memories.append(item)

    # Suggest the most-targeted retraction handle: the entity arg if given,
    # else a seed finding id (so the analyst can review the exact lineage).
    if source.get("entity"):
        retract_target = source["entity"]
    elif correlation["seed_ids"]:
        retract_target = correlation["seed_ids"][0]
    elif sorted(suspected):
        retract_target = sorted(suspected)[0]
    else:
        retract_target = None

    suggestion = (
        f"run memory_retract(investigation_id={investigation_id!r}, "
        f"target={retract_target!r}, dry_run=true) to review cleanup"
        if retract_target and contaminated_memories else None
    )

    result = {
        "investigation_id": investigation_id,
        "suspected_entities": sorted(suspected),
        "source": source,
        "code_findings": code_findings,
        "contaminated_memories": contaminated_memories,
        "already_retracted": already_retracted,
        "count": len(contaminated_memories),
        "suggestion": suggestion,
        "advisory": True,
    }
    if code_note:
        result["note"] = code_note
    return json.dumps(result, indent=2)


# ---- Tool: memory_health (substrate self-check) ----

# Worst-status precedence used to roll per-check results up into the overall
# status. "fail" -> unhealthy/degraded, "warn" -> degraded, all "ok" -> ok.
_HEALTH_SEVERITY = {"ok": 0, "warn": 1, "fail": 2}


def _health_check(name: str, fn) -> dict:
    """Run a single probe ``fn`` fail-open, returning a check entry dict.

    ``fn`` returns ``(status, detail, remediation_or_None)`` or, on any
    exception, this wrapper synthesizes a ``fail`` entry. A failing probe must
    never crash ``memory_health`` — that's the whole point of self-check.
    """
    try:
        status, detail, remediation = fn()
    except Exception as exc:  # fail-open: a broken probe is itself a finding
        entry = {
            "name": name,
            "status": "fail",
            "detail": f"probe raised: {exc!r}",
            "remediation": "this probe crashed — inspect server logs; the check itself is buggy",
        }
        return entry
    entry = {"name": name, "status": status, "detail": detail}
    if remediation:
        entry["remediation"] = remediation
    return entry


def _health_qdrant_collection_info(client, collection: str) -> dict:
    """Best-effort introspection of one qdrant collection's count + vector layout."""
    info: dict = {"collection": collection}
    try:
        info["points"] = int(client.count(collection, exact=False).count)
    except Exception as exc:
        info["points"] = None
        info["count_error"] = repr(exc)
    try:
        ci = client.get_collection(collection)
        params = ci.config.params
        vectors = getattr(params, "vectors", None)
        named: dict = {}
        if isinstance(vectors, dict):
            for vname, vparams in vectors.items():
                named[vname] = getattr(vparams, "size", None)
        elif vectors is not None:
            # unnamed/default single vector
            named["(default)"] = getattr(vectors, "size", None)
        info["dense_vectors"] = named
        sparse = getattr(params, "sparse_vectors", None)
        info["sparse_vectors"] = sorted(sparse.keys()) if isinstance(sparse, dict) else []
    except Exception as exc:
        info["config_error"] = repr(exc)
    return info


def _health_collection_dim(info: dict) -> int | None:
    """Pull the dense vector dimension out of a collection-info dict, if known."""
    dense = info.get("dense_vectors")
    if isinstance(dense, dict):
        for vname in ("dense", "(default)"):
            dim = dense.get(vname)
            if isinstance(dim, int):
                return dim
        for dim in dense.values():
            if isinstance(dim, int):
                return dim
    return None


# The two memory_health probes below are STATELESS — they read no enclosing local,
# only module globals (_embed_sparse; _get_mnemo_funcs/_mnemo_bank), which is why they
# live out here rather than as closures inside memory_health. Each returns the same
# (status, detail, remediation) 3-tuple _health_check expects, and _health_check still
# wraps them, so a raising probe still degrades to a "fail" entry (fail-open).
def _health_probe_embeddings_sparse():
    """memory_health probe 4: optional sparse (fastembed BM25) embedder."""
    sparse = _embed_sparse("memory_health probe")  # transient, never stored
    if sparse is None:
        return (
            "warn",
            "sparse embedder (fastembed BM25) unavailable — hybrid search "
            "degrades to dense-only; this is optional.",
            "install the fastembed sparse model if hybrid scoring is wanted",
        )
    n = len(getattr(sparse, "indices", []) or [])
    return ("ok", f"sparse embedder active; {n} non-zero terms on probe", None)


def _health_probe_mnemo_mirror():
    """memory_health probe 5: mnemosyne importable in THIS venv (the silent-flag bug)."""
    remember, recall = _get_mnemo_funcs()
    bank = _mnemo_bank()
    resolved = {"remember": remember is not None, "recall": recall is not None}
    if remember is None and recall is None:
        return (
            "fail",
            {
                "target_bank": bank,
                "resolved": resolved,
                "note": "mnemosyne is not importable in the server's venv — the "
                        "loci->mnemo mirror is silently inert.",
            },
            "install mnemosyne into the server's venv: "
            "pip install 'mnemosyne-memory[embeddings]' sqlite-vec",
        )
    if remember is None or recall is None:
        return (
            "warn",
            {"target_bank": bank, "resolved": resolved,
             "note": "mnemosyne partially resolved — one of remember/recall is missing."},
            "reinstall 'mnemosyne-memory[embeddings]' to restore both entry points",
        )
    return ("ok", {"target_bank": bank, "resolved": resolved}, None)


# The next three probes are the *stateful* trio: probe 1 discovers the qdrant
# client and main collection, probe 2 reads them and records per-collection
# dims, and probe 3 records the embedder dimension probe 6 compares against.
# What used to be ``nonlocal`` writes into memory_health's frame are now writes
# into an explicit ``sink`` dict the tool body owns, which is what lets these
# live at module level. Their call sites wrap them in lambdas, so the reads are
# still deferred to probe-execution time exactly as the closures were.
def _health_probe_qdrant_reachable(qdrant_url: str, sink: dict) -> tuple:
    """memory_health probe 1: is QDRANT_URL set and the server answering?

    On a successful ``_get_qdrant()`` this publishes ``sink["client"]`` and
    ``sink["main_col"]`` for probe 2. The assignment happens only *after*
    ``_get_qdrant()`` returns, so if it raises the sink keeps its ``None``
    initialisers and ``_health_check`` synthesizes the ``fail`` entry — the same
    fail-open behaviour the ``nonlocal`` version had.
    """
    if not qdrant_url:
        return (
            "warn",
            "QDRANT_URL is not set — qdrant indexing disabled; loci falls back"
            "to mnemosyne/keyword search.",
            "set QDRANT_URL if vector search is expected; otherwise this is benign",
        )
    client, main_col = _get_qdrant()
    sink["client"], sink["main_col"] = client, main_col
    if client is None:
        return (
            "fail",
            f"QDRANT_URL={qdrant_url} is set but the server is unreachable.",
            "confirm the qdrant container is up and reachable at QDRANT_URL "
            "(docker ps; curl $QDRANT_URL/healthz)",
        )
    return ("ok", f"connected to qdrant at {qdrant_url}", None)


def _health_probe_qdrant_collections(client, main_col, collection_dims: dict) -> tuple:
    """memory_health probe 2: expected collections present + their layout.

    ``collection_dims`` is mutated in place, so probe 6 observes whatever dims
    are discovered here.
    """
    if client is None:
        return ("warn", "skipped — qdrant not reachable", None)
    existing = {c.name for c in client.get_collections().collections}
    report: dict = {}
    main = main_col or QDRANT_COLLECTION_PREFIX
    main_present = main in existing
    verdicts_present = "hermes_verdicts" in existing
    if main_present:
        info = _health_qdrant_collection_info(client, main)
        report[main] = info
        collection_dims[main] = _health_collection_dim(info)
    if verdicts_present:
        info = _health_qdrant_collection_info(client, "hermes_verdicts")
        report["hermes_verdicts"] = info
        collection_dims["hermes_verdicts"] = _health_collection_dim(info)
    report["expected_main_collection"] = main
    report["main_present"] = main_present
    report["hermes_verdicts_present"] = verdicts_present
    if not main_present:
        return (
            "fail",
            report,
            f"main memory collection '{main}' is missing while qdrant is up — "
            "it is created lazily on first store; run a store/search or "
            "backfill (scripts/backfill_qdrant.py)",
        )
    if not verdicts_present:
        return (
            "warn",
            report,
            "'hermes_verdicts' not yet created — it appears on first "
            "memory_self_check(record=True); benign until then",
        )
    return ("ok", report, None)


def _health_probe_embeddings_dense(sink: dict) -> tuple:
    """memory_health probe 3: fastembed dense embedder loads and embeds.

    Publishes ``sink["embed_dim"]`` for the dimension-consistency probe.
    """
    vec = _embed("memory_health probe")  # transient throwaway, never stored
    if not vec:
        return (
            "fail",
            "dense embedder (Ollama) unavailable — semantic/hybrid search "
            "is disabled; hermes runs on keyword fallback only.",
            "ensure Ollama is running and the nomic-embed-text model is available "
            "(OLLAMA_BASE_URL and EMBED_MODEL env vars can override defaults).",
        )
    sink["embed_dim"] = len(vec)
    return ("ok", f"dense embedder active; dimension={sink['embed_dim']}", None)


# The next three probes are pure functions of their arguments: what used to be
# read from memory_health's enclosing scope is now passed in explicitly.
# ``collection_dims`` is handed over by reference, so the dimension probe still
# observes whatever the qdrant_collections probe wrote into that same dict.
def _health_probe_dimension_consistency(
    embed_dim: int | None, collection_dims: dict[str, int | None]
) -> tuple:
    """memory_health probe 6: embedder dim vs configured collection dim(s)."""
    known = {c: d for c, d in collection_dims.items() if isinstance(d, int)}
    if embed_dim is None and not known:
        return ("warn", "neither embedder dim nor any collection dim known — cannot compare", None)
    detail = {"embedder_dim": embed_dim, "configured_expected": VECTOR_DIM,
              "collection_dims": known}
    mismatches = []
    if embed_dim is not None:
        if embed_dim != VECTOR_DIM:
            mismatches.append(f"embedder dim {embed_dim} != configured VECTOR_DIM {VECTOR_DIM}")
        for col, dim in known.items():
            if dim != embed_dim:
                mismatches.append(f"collection '{col}' dim {dim} != embedder dim {embed_dim}")
    else:
        for col, dim in known.items():
            if dim != VECTOR_DIM:
                mismatches.append(f"collection '{col}' dim {dim} != configured VECTOR_DIM {VECTOR_DIM}")
    if mismatches:
        detail["mismatches"] = mismatches
        return (
            "fail",
            detail,
            "dimension mismatch causes silent search corruption — recreate the "
            "collection or align the embedding model so all dims match",
        )
    return ("ok", detail, None)


def _health_resolve_inv_scope(investigation_id: Optional[str]) -> tuple[list[str], str | None]:
    """Resolve the target investigations for memory_health's store-scoped checks.

    Returns ``(inv_targets, inv_missing)``. Read-only: never use ``_inv_dir``
    here, which would mkdir.
    """
    if investigation_id is not None:
        inv_targets = [investigation_id] if (MEMORY_DIR / investigation_id).exists() else []
        inv_missing = investigation_id if not inv_targets else None
    elif MEMORY_DIR.exists():
        inv_targets = sorted(p.name for p in MEMORY_DIR.iterdir() if p.is_dir())
        inv_missing = None
    else:
        inv_targets = []
        inv_missing = None
    return inv_targets, inv_missing


def _health_probe_retraction_integrity(inv_targets: list[str], inv_missing: str | None) -> tuple:
    """memory_health probe 7: retraction logs parse cleanly + no orphans."""
    if inv_missing is not None:
        return ("warn", f"investigation '{inv_missing}' not found", None)
    per_inv: list[dict] = []
    total_active = 0
    total_orphans = 0
    parse_errors: list[str] = []
    for inv in inv_targets:
        inv_path = MEMORY_DIR / inv
        findings = _read_jsonl(inv_path / "findings.jsonl")
        valid_ids = {
            str(f.get("id") or f.get("finding_id") or f"{inv}:{i}")
            for i, f in enumerate(findings) if isinstance(f, dict)
        }
        ret_path = inv_path / "retractions.jsonl"
        audit_path = inv_path / "retraction_audit.jsonl"
        # parse-cleanliness: count raw non-empty lines vs parsed rows
        for label, path in (("retractions.jsonl", ret_path),
                            ("retraction_audit.jsonl", audit_path)):
            if path.exists():
                raw = [ln for ln in path.read_text().splitlines() if ln.strip()]
                parsed = _read_jsonl(path)
                if len(parsed) != len(raw):
                    parse_errors.append(f"{inv}/{label}: {len(raw) - len(parsed)} unparseable line(s)")
        active = _load_retracted_ids(inv) if ret_path.exists() else set()
        orphans = sorted(fid for fid in active if fid not in valid_ids)
        total_active += len(active)
        total_orphans += len(orphans)
        if active or orphans:
            per_inv.append({
                "investigation_id": inv,
                "active_retractions": len(active),
                "orphaned_retractions": orphans,
            })
    detail = {
        "investigations_scanned": len(inv_targets),
        "active_retractions": total_active,
        "orphaned_retractions": total_orphans,
        "per_investigation": per_inv,
    }
    if parse_errors:
        detail["parse_errors"] = parse_errors
        return (
            "fail",
            detail,
            "retraction log has unparseable lines — inspect the JSONL; "
            "append-only integrity may be compromised",
        )
    if total_orphans:
        return (
            "warn",
            detail,
            "orphaned retraction(s): a retraction references a finding id "
            "not present in findings.jsonl — verify the finding wasn't lost",
        )
    return ("ok", detail, None)


def _health_probe_store_counts(inv_targets: list[str], inv_missing: str | None) -> tuple:
    """memory_health probe 8: inventory of findings/audit/retraction records."""
    if inv_missing is not None:
        return ("warn", f"investigation '{inv_missing}' not found", None)
    per_inv: list[dict] = []
    totals = {"findings": 0, "audit": 0, "retractions": 0}
    for inv in inv_targets:
        inv_path = MEMORY_DIR / inv
        f_n = len(_read_jsonl(inv_path / "findings.jsonl"))
        a_n = len(_read_jsonl(inv_path / "audit.jsonl"))
        r_n = len(_read_jsonl(inv_path / "retractions.jsonl"))
        totals["findings"] += f_n
        totals["audit"] += a_n
        totals["retractions"] += r_n
        per_inv.append({"investigation_id": inv, "findings": f_n,
                        "audit": a_n, "retractions": r_n})
    detail = {
        "investigations": len(inv_targets),
        "totals": totals,
        "per_investigation": per_inv,
    }
    return ("ok", detail, None)


def _health_rollup(checks: list[dict]) -> tuple[str, str]:
    """Roll per-check results up into ``(status, summary)`` for memory_health."""
    worst = max((_HEALTH_SEVERITY.get(c["status"], 0) for c in checks), default=0)
    if worst >= 2:
        status = "unhealthy"
    elif worst == 1:
        status = "degraded"
    else:
        status = "ok"

    n_fail = sum(1 for c in checks if c["status"] == "fail")
    n_warn = sum(1 for c in checks if c["status"] == "warn")
    summary = (
        f"{len(checks)} checks: {n_fail} fail, {n_warn} warn, "
        f"{len(checks) - n_fail - n_warn} ok -> {status}"
    )
    return status, summary


def _selftest_rollup(rows: list[dict]) -> tuple[str, str]:
    """ok | degraded | unhealthy, plus a one-line summary.

    An empty collection is not a fault — nothing to retrieve is not the same as
    unable to retrieve.
    """
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    broken = counts.get("width_mismatch", 0) + counts.get("error", 0)
    if broken and broken >= len(rows):
        status = "unhealthy"
    elif broken:
        status = "degraded"
    else:
        status = "ok"
    parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    return status, f"{len(rows)} collection(s): {parts} -> {status}"


@mcp.tool()
def retrieval_selftest(query: str = "system architecture", limit: int = 3,
                       collections: Optional[list] = None, scope: str = "queried") -> str:
    """
    Prove every Qdrant collection can actually be retrieved from.

    memory_health inspects the substrate — is Qdrant up, do the embedders load —
    but only for the collections it knows by name. A store usually holds more,
    written by other ingests at other embedding widths, and those fail in a way
    nothing reports: the query raises on a width mismatch, the caller catches it
    per-collection, and the collection becomes indistinguishable from one that
    simply had no match.

    This sweeps the whole store and answers the operational question directly:
    if I ask this collection something, do rows come back? Per collection it
    reports points, dense width, sparse presence and hit count, classified as
    ok | empty | no_results | width_mismatch | error, rolling up to
    ok | degraded | unhealthy. Remediations are deduplicated — one unmapped
    width usually explains many collections at once.

    Read-only. Bypasses the cross-encoder and its relevance floor on purpose:
    the floor is a relevance judgement and this is a question about wiring.

    Scope matters more than it looks. A store accumulates collections this server
    never queries — feature vectors at their own widths, corpora left over from
    other tools — and sweeping them rolls a dozen irrelevant width mismatches into
    the health verdict. A diagnostic that reports `degraded` because of things
    nobody asks about teaches you to ignore it, which is the exact failure this
    tool exists to prevent. So the default scope is what the server would actually
    retrieve from.

    Args:
        query: Probe text. Any in-domain phrase works; the point is retrievability.
        limit: Rows to request per collection (default 3).
        collections: Probe exactly these. Overrides ``scope``.
        scope: ``queried`` (default) — the collections this server retrieves from:
               the findings collection plus the configured code-chunks collection.
               ``all`` — every collection in the store, for an inventory; width
               mismatches outside the queried set are reported but do NOT count
               against health, since nothing asks them anything.

    Returns:
        JSON with {status, summary, scope, collections, remediations}.
    """
    client, _col = _get_qdrant()
    if client is None:
        return json.dumps({
            "status": "unhealthy",
            "summary": "Qdrant unavailable — nothing could be probed",
            "collections": [],
            "remediations": ["Check QDRANT_URL and QDRANT_API_KEY."],
        }, indent=2)

    try:
        present = sorted(c.name for c in client.get_collections().collections)
        queried = [QDRANT_COLLECTION_PREFIX] + (
            [_CODE_CHUNKS_COLLECTION] if _CODE_CHUNKS_COLLECTION else [])
        if collections:
            names, in_scope = list(collections), set(collections)
        elif scope == "all":
            names, in_scope = present, set(queried)
        else:
            names, in_scope = [n for n in queried if n in present], set(queried)
    except Exception as exc:
        return json.dumps({
            "status": "unhealthy",
            "summary": f"could not list collections: {exc}",
            "collections": [],
            "remediations": ["Check Qdrant connectivity and API key scope."],
        }, indent=2)

    if not names:
        return json.dumps({
            "status": "ok", "summary": "no collections exist",
            "collections": [], "remediations": [],
        }, indent=2)

    try:
        query_vec = _embed(query)
    except Exception as exc:
        logger.warning("retrieval_selftest: embed failed: %r", exc)
        query_vec = None

    rows = [probe_collection(query_vec, client, name, limit=limit) for name in names]
    for r in rows:
        r["queried_by_server"] = r["collection"] in in_scope
    # Only the collections this server actually retrieves from can make it unhealthy.
    status, summary = _selftest_rollup([r for r in rows if r["queried_by_server"]] or rows)

    remediations: list[str] = []
    for r in (r for r in rows if r["queried_by_server"]):
        rem = r.get("remediation")
        if rem and rem not in remediations:
            remediations.append(rem)

    out_of_scope = [r["collection"] for r in rows if not r["queried_by_server"]]
    return json.dumps({
        "status": status,
        "summary": summary + (
            f" (+{len(out_of_scope)} not queried by this server, excluded from health)"
            if out_of_scope else ""),
        "scope": "explicit" if collections else scope,
        "embedder_dim": len(query_vec) if query_vec else None,
        "collections": rows,
        "remediations": remediations,
    }, indent=2)


@mcp.tool()
def memory_health(investigation_id: Optional[str] = None) -> str:
    """
    Check the server's own memory substrate — a read-only self-diagnosis.

    Where ``memory_self_check`` inspects what the server *remembered*, this inspects
    the machinery that does the remembering: qdrant reachability and collection
    layout, the dense/sparse embedders, the mnemosyne mirror wired into this
    venv, vector-dimension consistency, retraction-log integrity, and a store
    inventory. It's the server's equivalent of ``mnemosyne diagnose`` and would have
    surfaced the silently-broken loci->mnemo mirror as a failed probe.

    This tool is strictly **read-only**: it writes, mutates, retracts, and
    creates nothing on disk or in qdrant. The single side effect it may incur is
    a transient throwaway embed of a tiny string to confirm the embedder loads.
    Every probe is wrapped fail-open — one failing check never crashes the tool;
    it becomes a ``fail`` entry in the report instead.

    Args:
        investigation_id: Scope retraction/store checks to one investigation.
            When omitted, substrate checks run globally and retraction/store
            checks summarize across all investigations.

    Returns:
        JSON: {checked_at, status: ok|degraded|unhealthy, checks: [{name,
        status, detail, remediation?}], summary}. ``status`` is rolled up from
        the worst check — any fail -> unhealthy (degraded if also otherwise
        usable), any warn -> degraded, all ok -> ok.
    """
    checks: list[dict] = []
    # State shared between the first three probes and the dimension check. The
    # probes are module-level functions now, so what they used to reach via
    # ``nonlocal`` they write into this explicit holder instead.
    qd: dict = {"client": None, "main_col": None, "embed_dim": None}

    # 1. qdrant_reachable — is QDRANT_URL set and the server answering?
    qdrant_url = os.environ.get("QDRANT_URL", "")
    checks.append(_health_check(
        "qdrant_reachable",
        lambda: _health_probe_qdrant_reachable(qdrant_url, qd),
    ))

    # 2. qdrant_collections — expected collections present + their layout.
    # The lambda defers reading ``qd`` until after probe 1 has populated it.
    collection_dims: dict[str, int | None] = {}
    checks.append(_health_check(
        "qdrant_collections",
        lambda: _health_probe_qdrant_collections(
            qd["client"], qd["main_col"], collection_dims
        ),
    ))

    # 3. embeddings_dense — fastembed dense embedder loads and embeds.
    checks.append(_health_check(
        "embeddings_dense",
        lambda: _health_probe_embeddings_dense(qd),
    ))

    # 4. embeddings_sparse — optional sparse embedder. (module-level probe)
    checks.append(_health_check("embeddings_sparse", _health_probe_embeddings_sparse))

    # 5. mnemo_mirror — mnemosyne importable in THIS venv (the silent-flag bug).
    checks.append(_health_check("mnemo_mirror", _health_probe_mnemo_mirror))

    # 6. dimension_consistency — embedder dim vs configured collection dim(s).
    # The lambda defers the read of ``qd["embed_dim"]`` to probe-execution time,
    # exactly as the closure this replaced did — by then probe 3 has run.
    checks.append(_health_check(
        "dimension_consistency",
        lambda: _health_probe_dimension_consistency(qd["embed_dim"], collection_dims),
    ))

    # Resolve target investigations for the store-scoped checks (read-only:
    # never use _inv_dir here, which would mkdir).
    inv_targets, inv_missing = _health_resolve_inv_scope(investigation_id)

    # 7. retraction_integrity — parse cleanly + flag orphaned retractions.
    checks.append(_health_check(
        "retraction_integrity",
        lambda: _health_probe_retraction_integrity(inv_targets, inv_missing),
    ))

    # 8. store_counts — inventory of findings/audit/retraction records.
    checks.append(_health_check(
        "store_counts",
        lambda: _health_probe_store_counts(inv_targets, inv_missing),
    ))

    # Roll up overall status from the worst check.
    status, summary = _health_rollup(checks)

    return json.dumps({
        "checked_at": _now(),
        "status": status,
        "scope": investigation_id if investigation_id is not None else "all",
        "checks": checks,
        "summary": summary,
    }, indent=2)


# ---- Tools: memory_retract / memory_restore ----

# Similarity floor for treating a qdrant neighbor as semantically contaminated.
_RETRACT_SEMANTIC_MIN_SCORE = 0.62


def _resolve_seed_findings(investigation_id: str, target: str, findings: list[dict]) -> list[dict]:
    """Resolve a ``target`` (finding id, or a claim/entity string) to seed findings.

    - If ``target`` exactly matches a finding id, that finding is the seed.
    - Otherwise treat ``target`` as a claim/entity string: rank findings by
      entity overlap (distinctive entities of the target text) then lexical
      overlap, and return the best match(es). Findings sharing a distinctive
      entity with the target all qualify as seeds (the hallucinated entity is
      the anchor). Falls back to the single best lexical match.
    """
    by_id = {str(f.get("id", "")): f for f in findings}
    if target in by_id:
        return [by_id[target]]

    target_entities = _distinctive_entity_set(_extract_entities(target))
    target_tokens = tokenize(target)

    entity_hits: list[dict] = []
    scored: list[tuple[float, dict]] = []
    for f in findings:
        ftext = str(f.get("text", "") or "")
        f_entities = _distinctive_entity_set(f.get("entities") or _extract_entities(ftext))
        if target_entities and (target_entities & f_entities):
            entity_hits.append(f)
            continue
        score = _lexical_match_score(target_tokens, tokenize(ftext)) if target_tokens else 0.0
        if score > 0:
            scored.append((score, f))

    if entity_hits:
        return entity_hits
    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        return [scored[0][1]]
    return []


def _semantic_neighbor_ids(seed_findings: list[dict], investigation_id: str) -> list[str]:
    """Find finding ids qdrant places near the seed text. Fail-open -> []."""
    client, _col = _get_qdrant()
    if client is None:
        return []
    ids: set[str] = set()
    try:
        for seed in seed_findings:
            text = str(seed.get("text", "") or "")
            if not text.strip():
                continue
            res = _qdrant_similarity_search(text, investigation_id=investigation_id, limit=25)
            if not res.get("ok"):
                continue
            for row in res.get("results", []):
                if _safe_float(row.get("score", 0.0)) < _RETRACT_SEMANTIC_MIN_SCORE:
                    continue
                rid = row.get("id") or row.get("finding_id")
                if rid:
                    ids.add(str(rid))
    except Exception as exc:  # fail-open — semantic scope is best-effort
        logger.debug("semantic neighbor lookup failed, skipping: %r", exc)
        return []
    return sorted(ids)


def _forget_finding_verdicts(finding: dict) -> int:
    """Delete the qdrant verdict points keyed on this finding's text. Fail-open."""
    text = str(finding.get("text", "") or "")
    if not text.strip():
        return 0
    client, _col = _get_qdrant()
    if client is None:
        return 0
    try:
        import asyncio

        from memcheck import QdrantBackend, VerdictEngine

        backend = QdrantBackend(
            client=client, collection="hermes_verdicts", embed=_embed, vector_name="dense",
        )
        engine = VerdictEngine(backend)

        async def _forget_all() -> int:
            removed = 0
            removed += await engine.forget(redact_excerpt(text), "memory")
            return removed

        # Same loop-detection guard as _record_claim_verdicts — asyncio.run() raises
        # RuntimeError when FastMCP is dispatching inline on a running event loop.
        try:
            asyncio.get_running_loop()
            _result: list[int] = []
            _exc2: list[Exception] = []

            def _run_forget() -> None:
                try:
                    _result.append(asyncio.run(_forget_all()))
                except Exception as e:
                    _exc2.append(e)

            t2 = threading.Thread(target=_run_forget, daemon=True)
            t2.start()
            t2.join()
            if _exc2:
                raise _exc2[0]
            return _result[0] if _result else 0
        except RuntimeError:
            return asyncio.run(_forget_all())
    except Exception as exc:  # fail-open — verdict cleanup is best-effort
        logger.debug("verdict forget failed, degrading to 0: %r", exc)
        return 0


def _retract_cluster(
    seed_ids: list[str], findings: list[dict], semantic_ids: list[str]
) -> tuple[list[str], dict, dict, list[dict]]:
    """Compute the contamination cluster for a retraction and its dry-run items."""
    cluster = find_contamination(
        seed_ids,
        findings,
        entities_of=_extract_entities,
        semantic_neighbor_ids=semantic_ids,
        min_shared_entities=1,
    )
    contaminated_ids = cluster["contaminated_ids"]
    reasons = cluster["reasons"]
    by_id = {str(f.get("id", "")): f for f in findings}

    items = [
        {
            "finding_id": fid,
            "text_excerpt": redact_excerpt(str(by_id.get(fid, {}).get("text", "") or "")),
            "reasons": reasons.get(fid, []),
        }
        for fid in contaminated_ids
    ]
    return contaminated_ids, reasons, by_id, items


def _retract_write_tombstones(
    retractions_path,
    contaminated_ids: list[str],
    by_id: dict,
    reasons: dict,
    seed_anchor: str,
    reason: str,
    ts: str,
) -> tuple[list[dict], int]:
    """Append one retraction tombstone per contaminated finding, forgetting verdicts."""
    retracted_records: list[dict] = []
    verdicts_forgotten = 0
    for fid in contaminated_ids:
        retraction_id = str(uuid.uuid4())
        entry = {
            "retraction_id": retraction_id,
            "finding_id": fid,
            "seed_id": seed_anchor,
            "reason": reason or "hallucination retraction",
            "ts": ts,
            "active": True,
        }
        _append_jsonl(retractions_path, entry)
        verdicts_forgotten += _forget_finding_verdicts(by_id.get(fid, {}))
        retracted_records.append({
            "retraction_id": retraction_id,
            "finding_id": fid,
            "reasons": reasons.get(fid, []),
        })
    return retracted_records, verdicts_forgotten


def _retract_stamp_valid_until(investigation_id: str, contaminated_ids: list[str], ts: str) -> None:
    """Bi-temporal hook: stamp valid_until on retracted findings so that
    investigation_as_of queries exclude them from any future as-of view.
    Fails open — never propagates."""
    try:
        findings_path = _inv_dir(investigation_id) / "findings.jsonl"
        _rewrite_jsonl_set_field(
            findings_path,
            set(contaminated_ids),
            "valid_until",
            ts,
        )
    except Exception as exc:  # fail-open
        logger.debug("bi-temporal valid_until stamp failed, degrading: %r", exc)


def _retract_quarantine_verdict(
    seeds: list[dict], seed_anchor: str, reason: str, contaminated_ids: list[str]
) -> bool:
    """Record a quarantine verdict to the store (fail-open) for recall."""
    try:
        seed_text = str(seeds[0].get("text", "") or "")
        quarantine_verdict = new_verdict(
            subject_kind="memory",
            subject_signature=make_signature("memory", seed_anchor),
            subject_excerpt=redact_excerpt(seed_text),
            verdict_type="retracted",
            decision="quarantine",
            confidence=0.9,
            rationale=reason or "hallucination retracted with contaminated lineage",
            source="human",
            refs=list(contaminated_ids),
        )
        return _record_verdicts([quarantine_verdict])
    except Exception as exc:  # fail-open
        logger.debug("quarantine verdict record failed, degrading: %r", exc)
        return False


@mcp.tool()
def memory_retract(
    investigation_id: str,
    target: str,
    reason: str = "",
    dry_run: bool = True,
    scope_semantic: bool = True,
) -> str:
    """
    Retract a hallucinated finding and its contaminated lineage — reversibly.

    When testing reveals a stored fact never existed (e.g. a fabricated
    ``http://localhost:8080/v1/foo`` endpoint), everything built on it is
    contaminated. This finds that lineage — by shared distinctive entities,
    semantic proximity (qdrant), and forward ``derived_from`` links — and
    soft-tombstones it so it drops out of recall/search/reflect. Nothing is
    hard-deleted: ``findings.jsonl`` stays append-only and ``memory_restore``
    reverses any retraction.

    **Advisory-first**: ``dry_run`` defaults to True and changes NOTHING — it
    returns the proposed cluster for review. Re-run with ``dry_run=False`` to
    apply the soft tombstone.

    Args:
        investigation_id: Investigation identifier.
        target: A finding id to retract, OR a claim/entity string (e.g. the
                hallucinated URL) — its distinctive entities become the anchor.
        reason: Why this is being retracted (e.g. "endpoint never existed —
                confirmed by testing"). Recorded in the retraction + audit trail.
        dry_run: When True (default), return the proposed cluster and change
                 nothing. When False, apply the soft tombstone.
        scope_semantic: Include qdrant semantic neighbors in the cluster
                        (default True). Fails open — if qdrant is down, the
                        entity + derivation scope still applies.

    Returns:
        dry_run=True  -> {seed_ids, would_retract:[{finding_id, text_excerpt,
                          reasons}], count, applied:false, advisory:...}
        dry_run=False -> {seed_ids, retracted:[...], count, applied:true, ...}
    """
    manifest = _load_manifest(investigation_id)
    if not manifest:
        return json.dumps({"error": f"Investigation '{investigation_id}' not found."})
    _owner = manifest.get("owner", "")
    if _owner and _owner != AGENT_ID:
        return json.dumps({"error": "permission_denied", "detail": f"investigation {investigation_id!r} is owned by {_owner!r}"})
    if not str(target or "").strip():
        return json.dumps({"error": "target must be a non-empty finding id or claim/entity string"})

    findings = _tag_finding_ids(
        _read_jsonl(_inv_dir(investigation_id) / "findings.jsonl"), investigation_id
    )
    seeds = _resolve_seed_findings(investigation_id, str(target).strip(), findings)
    if not seeds:
        return json.dumps({
            "error": f"could not resolve target {target!r} to any finding",
            "seed_ids": [],
            "count": 0,
            "applied": False,
        })
    seed_ids = [str(s.get("id")) for s in seeds]

    semantic_ids: list[str] = []
    if scope_semantic:
        semantic_ids = _semantic_neighbor_ids(seeds, investigation_id)

    contaminated_ids, reasons, by_id, items = _retract_cluster(seed_ids, findings, semantic_ids)

    if dry_run:
        return json.dumps({
            "seed_ids": seed_ids,
            "would_retract": items,
            "count": len(items),
            "applied": False,
            "scope_semantic": scope_semantic,
            "semantic_neighbors": len(semantic_ids),
            "advisory": "review then re-run with dry_run=false to apply the soft tombstone (reversible via memory_restore)",
        }, indent=2)

    # --- Apply: append soft tombstones, forget qdrant verdicts, audit. ---
    ts = _now()
    retractions_path = _inv_dir(investigation_id) / "retractions.jsonl"
    audit_path = _inv_dir(investigation_id) / "retraction_audit.jsonl"
    seed_anchor = seed_ids[0]

    # Acquire a per-investigation lock so that the retractions.jsonl appends
    # and the matching retraction_audit.jsonl append are observed atomically
    # by concurrent readers (no window where a retraction exists without its
    # audit record).
    inv_lock = _investigation_lock(investigation_id)

    with inv_lock:
        retracted_records, verdicts_forgotten = _retract_write_tombstones(
            retractions_path, contaminated_ids, by_id, reasons, seed_anchor, reason, ts
        )

        _retract_stamp_valid_until(investigation_id, contaminated_ids, ts)

        _append_jsonl(audit_path, {
            "action": "retract",
            "ts": ts,
            "target": target,
            "seed_ids": seed_ids,
            "reason": reason or "hallucination retraction",
            "retracted_finding_ids": list(contaminated_ids),
            "count": len(contaminated_ids),
            "verdicts_forgotten": verdicts_forgotten,
            "scope_semantic": scope_semantic,
        })

    quarantine_recorded = _retract_quarantine_verdict(seeds, seed_anchor, reason, contaminated_ids)

    _event_log_append({
        "op": "retract",
        "investigation_id": investigation_id,
        "seed_ids": seed_ids,
        "count": len(retracted_records),
        "reason": reason or "hallucination retraction",
    })


    return json.dumps({
        "seed_ids": seed_ids,
        "retracted": retracted_records,
        "count": len(retracted_records),
        "verdicts_forgotten": verdicts_forgotten,
        "applied": True,
        "quarantine_verdict_recorded": quarantine_recorded,
        "reversible": "findings.jsonl is untouched (append-only); reverse with memory_restore",
    }, indent=2)


@mcp.tool()
def memory_restore(
    investigation_id: str,
    finding_id: Optional[str] = None,
    retraction_id: Optional[str] = None,
    reason: str = "",
) -> str:
    """
    Reverse a retraction — un-tombstone a finding so it returns to recall.

    Appends an ``active:false`` entry to the investigation's retractions log,
    which the read-path fold treats as a restore (the finding stops being
    filtered). Identify the finding by ``finding_id`` or by a specific
    ``retraction_id``. Fully reversible and audited; no data is mutated.

    Args:
        investigation_id: Investigation identifier.
        finding_id: The finding to restore. Either this or retraction_id required.
        retraction_id: A specific retraction entry to reverse (resolves its
                       finding_id). Used when finding_id is omitted.
        reason: Optional note on why it's being restored.

    Returns:
        JSON confirming the restore.
    """
    manifest = _load_manifest(investigation_id)
    if not manifest:
        return json.dumps({"error": f"Investigation '{investigation_id}' not found."})
    _owner = manifest.get("owner", "")
    if _owner and _owner != AGENT_ID:
        return json.dumps({"error": "permission_denied", "detail": f"investigation {investigation_id!r} is owned by {_owner!r}"})

    retractions_path = _inv_dir(investigation_id) / "retractions.jsonl"
    existing = _read_jsonl(retractions_path)

    target_fid = str(finding_id).strip() if finding_id else ""
    if not target_fid and retraction_id:
        for e in existing:
            if str(e.get("retraction_id", "")) == str(retraction_id):
                target_fid = str(e.get("finding_id", ""))
                break
    if not target_fid:
        return json.dumps({
            "error": "provide finding_id, or a retraction_id that resolves to a finding"
        })

    if target_fid not in _load_retracted_ids(investigation_id):
        return json.dumps({
            "finding_id": target_fid,
            "restored": False,
            "note": "finding is not currently retracted — nothing to restore",
        })

    ts = _now()
    _append_jsonl(retractions_path, {
        "retraction_id": str(uuid.uuid4()),
        "finding_id": target_fid,
        "seed_id": None,
        "reason": reason or "restore",
        "ts": ts,
        "active": False,
    })
    _append_jsonl(_inv_dir(investigation_id) / "retraction_audit.jsonl", {
        "action": "restore",
        "ts": ts,
        "finding_id": target_fid,
        "reason": reason or "restore",
    })

    return json.dumps({
        "finding_id": target_fid,
        "restored": True,
        "reason": reason or "restore",
    }, indent=2)


# ── Contract Declaration Store ─────────────────────────────────────────────────


@mcp.tool()
def contract_declare(
    investigation_id: str,
    entity: str,
    role: str,
    fields: str,
    protocol: str = "",
) -> str:
    """Store a cross-boundary contract declaration for an entity.

    Records what a producer outputs or what a consumer expects at a serialization
    boundary (HTTP API, message queue, DB schema, file format). Stored as a gap
    finding (unverified integration) tagged ``contract_declaration``. Surfaces in
    grounding so future agents can query before generating consumer code.

    Args:
        investigation_id: Investigation to store the contract in.
        entity: The entity this contract describes, e.g. ``"UserSerializer"``,
            ``"POST /api/users"``, ``"sensor/+/data topic"``.
        role: Either ``"producer"`` (what it outputs) or ``"consumer"``
            (what it expects as input).
        fields: JSON object mapping field names to type descriptions,
            e.g. ``'{"user_id": "int", "created_at": "ISO8601 string"}'``.
        protocol: Optional wire protocol, e.g. ``"JSON-HTTP"``, ``"MQTT"``,
            ``"gRPC"``, ``"Parquet"``. Stored for reference.

    Returns:
        JSON with ``{"stored": true, "finding_id": "<uuid>", "entity": ..., "role": ...}``
    """
    if role not in ("producer", "consumer"):
        return json.dumps({"error": "role must be 'producer' or 'consumer'"})
    try:
        json.loads(fields)
    except (json.JSONDecodeError, TypeError):
        return json.dumps({"error": "fields must be a valid JSON object string"})

    inv_dir = _inv_dir(investigation_id)
    manifest = _load_manifest(investigation_id)
    if manifest is None:
        return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

    protocol_note = f" protocol={protocol}" if protocol else ""
    text = (
        f"Contract {entity!r} as {role}: fields={fields}{protocol_note}"
    )
    tags = ["contract_declaration", f"entity:{entity}", f"role:{role}"]
    if protocol:
        tags.append(f"protocol:{protocol}")

    fid = str(uuid.uuid4())
    finding: dict = {
        "id": fid,
        "investigation_id": investigation_id,
        "ts": _now(),
        "created_at_ts": int(__import__("time").time()),
        "record_type": "gap",
        "type": "gap",
        "text": text,
        "source": "contract_declare",
        "confidence": "medium",
        "tags": tags,
        "derived_from": [],
        "entities": {},
    }
    _append_jsonl(inv_dir / "findings.jsonl", finding)
    manifest.setdefault("finding_counts", {})
    manifest["finding_counts"]["gap"] = manifest["finding_counts"].get("gap", 0) + 1
    _save_manifest(manifest)

    _mnemo_remember(
        f"Contract declaration — {entity} ({role}): {fields}",
        importance=0.8,
        metadata={"investigation_id": investigation_id, "finding_id": fid, "entity": entity, "role": role},
    )
    _qdrant_upsert(fid, text, {**finding, "tags": ",".join(tags)})

    _event_log_append({
        "event": "contract_declare", "investigation_id": investigation_id,
        "finding_id": fid, "entity": entity, "role": role,
    })
    return json.dumps({"stored": True, "finding_id": fid, "entity": entity, "role": role}, indent=2)


@mcp.tool()
def contract_query(
    investigation_id: str,
    entity: str,
    role: str = "",
) -> str:
    """Query stored contract declarations for an entity.

    Searches the investigation's findings for ``contract_declaration`` findings
    matching ``entity``. Optionally filters by role (``"producer"`` or
    ``"consumer"``). Falls back to Qdrant semantic search if local JSONL is empty.

    Args:
        investigation_id: Investigation to search.
        entity: Entity name to look up (exact match on the ``entity:<name>`` tag).
        role: Optional filter — ``"producer"``, ``"consumer"``, or ``""`` for both.

    Returns:
        JSON with ``{"contracts": [...findings], "count": N}``.
    """
    inv_dir = _inv_dir(investigation_id)
    jsonl_path = inv_dir / "findings.jsonl"
    findings = _read_jsonl(jsonl_path) if jsonl_path.exists() else []

    entity_tag = f"entity:{entity}"
    results = []
    for f in findings:
        tags = f.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]
        if "contract_declaration" not in tags:
            continue
        if entity_tag not in tags:
            continue
        if role and f"role:{role}" not in tags:
            continue
        results.append(f)

    if not results:
        hits = _qdrant_similarity_search(
            f"Contract {entity} {role}".strip(), limit=5, investigation_id=investigation_id
        )
        results = [
            h.get("payload", {}) for h in (hits or [])
            if "contract_declaration" in str(h.get("payload", {}).get("tags", ""))
            and f"entity:{entity}" in str(h.get("payload", {}).get("tags", ""))
        ]

    return json.dumps({"contracts": results, "count": len(results)}, indent=2)


@mcp.tool()
def contract_check(
    investigation_id: str,
    field_name: str,
    entity: str = "",
) -> str:
    """Check whether a field name conflicts with stored contract declarations.

    Loads contract declarations from the investigation and checks whether
    ``field_name`` appears as a near-miss for a declared field on the same entity
    (suggesting rename drift across a boundary). Uses the same prefix-overlap
    heuristic as the ``contract_contradiction`` memcheck rule.

    Args:
        investigation_id: Investigation to search.
        field_name: Field name to check (the name used in the new code).
        entity: Optional — scope the check to a specific entity's contracts.

    Returns:
        JSON with ``{"conflicts": [...], "consistent": bool}``.
    """
    inv_dir = _inv_dir(investigation_id)
    jsonl_path = inv_dir / "findings.jsonl"
    findings = _read_jsonl(jsonl_path) if jsonl_path.exists() else []

    contracts = []
    entity_filter = f"entity:{entity}" if entity else None
    for f in findings:
        tags = f.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]
        if "contract_declaration" not in tags:
            continue
        if entity_filter and entity_filter not in tags:
            continue
        contracts.append(f)

    if not contracts:
        return json.dumps({"conflicts": [], "consistent": True, "note": "no contracts stored"})

    new_tok = field_name.lower()
    conflicts = []
    for contract in contracts:
        text = str(contract.get("text", ""))
        import re as _re
        m = _re.search(r"fields=(\{[^}]+\})", text)
        if not m:
            continue
        try:
            declared = json.loads(m.group(1))
        except Exception as exc:
            logger.debug("contract_check: fail-open swallow: %r", exc)
            continue
        for stored_field, stored_type in declared.items():
            sf = stored_field.lower()
            if sf == new_tok:
                continue
            shorter, longer = sorted([new_tok, sf], key=len)
            if not longer:
                continue
            is_suffix = longer.endswith(shorter) or longer.startswith(shorter)
            if is_suffix and len(shorter) >= 4:
                contracts_entity = next(
                    (t[len("entity:"):] for t in (contract.get("tags") or []) if str(t).startswith("entity:")),
                    "unknown"
                )
                conflicts.append({
                    "entity": contracts_entity,
                    "your_field": field_name,
                    "declared_field": stored_field,
                    "declared_type": stored_type,
                    "finding_id": contract.get("id", ""),
                })

    return json.dumps({"conflicts": conflicts, "consistent": len(conflicts) == 0}, indent=2)


# ── Wiring Obligation Tracker ──────────────────────────────────────────────────


@mcp.tool()
def wiring_obligation_declare(
    investigation_id: str,
    class_name: str,
    method_name: str,
    expected_effect: str,
) -> str:
    """Declare a wiring obligation — a method that SHOULD perform an integration but is unverified.

    Stores a ``gap`` finding tagged ``wiring_obligation``. The obligation is open
    until ``wiring_obligation_resolve`` is called with evidence of fulfillment.

    Use this when generating a class that is named for an integration
    (Publisher, Sender, Exporter, Notifier) — declare the obligation immediately
    so it can be tracked across sessions and verified before shipping.

    Args:
        investigation_id: Investigation to store the obligation in.
        class_name: Class that bears the integration responsibility.
        method_name: Method that should perform the integration effect.
        expected_effect: What the method SHOULD do (the integration promise).

    Returns:
        JSON with ``{"stored": true, "finding_id": "<uuid>", "obligation": {...}}``.
    """
    inv_dir = _inv_dir(investigation_id)
    manifest = _load_manifest(investigation_id)
    if manifest is None:
        return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

    text = (
        f"Wiring obligation: {class_name}.{method_name}() must {expected_effect} "
        f"[UNVERIFIED — call wiring_obligation_resolve with evidence when confirmed]"
    )
    tags = [
        "wiring_obligation",
        f"class:{class_name}",
        f"method:{method_name}",
    ]
    fid = str(uuid.uuid4())
    finding: dict = {
        "id": fid,
        "investigation_id": investigation_id,
        "ts": _now(),
        "created_at_ts": int(__import__("time").time()),
        "record_type": "gap",
        "type": "gap",
        "text": text,
        "source": "wiring_obligation_declare",
        "confidence": "medium",
        "tags": tags,
        "derived_from": [],
        "entities": {},
    }
    _append_jsonl(inv_dir / "findings.jsonl", finding)
    manifest.setdefault("finding_counts", {})
    manifest["finding_counts"]["gap"] = manifest["finding_counts"].get("gap", 0) + 1
    _save_manifest(manifest)
    _event_log_append({
        "event": "wiring_obligation_declare", "investigation_id": investigation_id,
        "finding_id": fid, "class": class_name, "method": method_name,
    })
    return json.dumps({
        "stored": True,
        "finding_id": fid,
        "obligation": {"class": class_name, "method": method_name, "expected_effect": expected_effect},
    }, indent=2)


@mcp.tool()
def wiring_obligation_list(
    investigation_id: str,
    resolved: bool = False,
) -> str:
    """List wiring obligations for an investigation.

    By default returns only unresolved obligations (``gap`` findings tagged
    ``wiring_obligation``). Pass ``resolved=True`` to include all obligations
    including those resolved via ``wiring_obligation_resolve``.

    Args:
        investigation_id: Investigation to query.
        resolved: When ``True``, include obligations that have already been resolved.

    Returns:
        JSON with ``{"obligations": [...], "unresolved_count": N}``.
    """
    inv_dir = _inv_dir(investigation_id)
    jsonl_path = inv_dir / "findings.jsonl"
    findings = _read_jsonl(jsonl_path) if jsonl_path.exists() else []

    obligations = []
    unresolved = 0
    seen_ids: set = set()
    for f in reversed(findings):
        fid = f.get("id", "")
        if fid in seen_ids:
            continue
        seen_ids.add(fid)
        tags = f.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]
        if "wiring_obligation" not in tags:
            continue
        is_gap = f.get("record_type", f.get("type", "")) == "gap"
        if not resolved and not is_gap:
            continue
        obligations.append(f)
        if is_gap:
            unresolved += 1

    obligations.reverse()
    return json.dumps({"obligations": obligations, "unresolved_count": unresolved}, indent=2)


@mcp.tool()
def wiring_obligation_resolve(
    investigation_id: str,
    finding_id: str,
    evidence: str,
) -> str:
    """Resolve a wiring obligation by providing evidence of fulfillment.

    Changes the obligation from a ``gap`` finding to an ``observed`` finding,
    appends the evidence to the text, and sets confidence to ``"high"``.

    Args:
        investigation_id: Investigation containing the obligation.
        finding_id: ID of the ``wiring_obligation`` gap finding to resolve.
        evidence: What was verified — cite the file and line where the integration
            call was confirmed to exist.

    Returns:
        JSON with ``{"resolved": true, "finding_id": "<uuid>"}``.
    """
    inv_dir = _inv_dir(investigation_id)
    manifest = _load_manifest(investigation_id)
    if manifest is None:
        return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

    jsonl_path = inv_dir / "findings.jsonl"
    findings = _read_jsonl(jsonl_path) if jsonl_path.exists() else []

    target = next((f for f in reversed(findings) if f.get("id") == finding_id), None)
    if target is None:
        return json.dumps({"error": f"Finding '{finding_id}' not found in '{investigation_id}'."})

    tags = target.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]
    if "wiring_obligation" not in tags:
        return json.dumps({"error": f"Finding '{finding_id}' is not a wiring_obligation."})
    if target.get("record_type", "gap") != "gap":
        return json.dumps({"error": f"Finding '{finding_id}' is already resolved."})

    resolved_finding = {
        **target,
        "record_type": "observed",
        "type": "observed",
        "text": target["text"] + f" | RESOLVED: {evidence}",
        "confidence": "high",
        "ts": _now(),
        "tags": [t for t in tags if t != "wiring_obligation"] + ["wiring_obligation", "wiring_obligation_resolved"],
    }
    _append_jsonl(jsonl_path, resolved_finding)

    counts = manifest.setdefault("finding_counts", {})
    counts["gap"] = max(0, counts.get("gap", 1) - 1)
    counts["observed"] = counts.get("observed", 0) + 1
    _save_manifest(manifest)

    _event_log_append({
        "event": "wiring_obligation_resolve", "investigation_id": investigation_id,
        "finding_id": finding_id, "evidence": evidence[:200],
    })
    return json.dumps({"resolved": True, "finding_id": finding_id}, indent=2)


@mcp.tool()
def investigation_verify_all(investigation_id: str, limit: int = 20) -> str:
    """
    Batch adversarial-verify the OPEN findings of an investigation — run the skeptic
    (verify.verify_finding) over each and return per-finding {finding_id, verdict,
    confidence}. A triage aid, NOT a lifecycle change: the verdict is recorded as an
    append-only verification note and does NOT overwrite the finding's resolution
    (a verdict != a lifecycle state). Use finding_resolve to actually resolve.

    Skips resolved (fixed/intentional/wontfix/superseded) and soft-retracted findings.
    Fail-open: an unavailable local model yields verdict='uncertain' per finding.

    Args:
        investigation_id: Investigation whose open findings to verify.
        limit: Max number of open findings to verify (default 20).

    Returns:
        JSON: {"investigation_id": ..., "verified": N,
               "results": [{"finding_id", "verdict", "confidence"}, ...]}
        On error: {"error": "<message>"}
    """
    manifest = _load_manifest(investigation_id)
    if not manifest:
        return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

    import verify as _v

    findings = _read_jsonl(_inv_dir(investigation_id) / "findings.jsonl")
    overrides = _load_resolution_overrides(investigation_id)
    retracted = _load_retracted_ids(investigation_id)

    def _effective_res(f: dict) -> str:
        fid = str(f.get("id", ""))
        if fid in overrides:
            return overrides[fid]
        return str(f.get("resolution") or "open").lower()

    try:
        _limit = max(0, int(limit))
    except (TypeError, ValueError):
        _limit = 20

    # Apply the limit WHILE iterating so we stop after collecting `_limit` open
    # findings instead of materializing every open finding and then slicing
    # (Copilot round 7): cheaper on large findings sets.
    open_findings = []
    for f in findings:
        if len(open_findings) >= _limit:
            break
        if (
            isinstance(f, dict)
            and _effective_res(f) == "open"
            and str(f.get("id", "")) not in retracted
            and str(f.get("text") or "").strip()
        ):
            open_findings.append(f)

    results = []
    for f in open_findings:
        fid = str(f.get("id", ""))
        res = _v.verify_finding(
            str(f.get("text") or ""),
            investigation_id=investigation_id,
            gen_fn=_verify_gen_fn,
        )
        verdict = res.get("verdict", "uncertain")
        confidence = res.get("confidence", 0.0)
        # Record as an append-only verification note in a SEPARATE log so these
        # accumulating records never slow the resolution read path — does NOT
        # touch resolution. Guarded per-finding: a single write failure (disk
        # full, permission) is logged and skipped so the rest of the run
        # continues rather than aborting the whole verification batch.
        try:
            _append_jsonl(_finding_verifications_path(investigation_id), {
                "record_type": "verification",
                "investigation_id": investigation_id,
                "finding_id": fid,
                "verdict": verdict,
                "confidence": confidence,
                "ts": _now(),
            })
        except Exception as exc:  # noqa: BLE001 — never abort the batch on one write
            logger.warning("verification note append failed for %s (fail-open, skipped): %r",
                           fid, exc)
        results.append({
            "finding_id": fid,
            "verdict": verdict,
            "confidence": confidence,
        })

    return json.dumps({
        "investigation_id": investigation_id,
        "verified": len(results),
        "results": results,
    }, indent=2)


@mcp.tool()
def loci_health() -> str:
    """
    Read-only self-diagnosis of the Loci MCP server. Returns a JSON snapshot of the
    running code version, the graph-store state, and per-backend reachability so a
    'reconnect needed / backend down / graph lock held' condition is self-evident
    instead of hand-diagnosed. Every probe is independent, cheap (short timeout), and
    fail-open — a dead backend reports False/unavailable, never an exception.

    Returns JSON:
      code_version:      short git SHA of the running code ('' if not a git checkout)
      ladybug:           'available' | 'contended' | 'unavailable' | 'latched' | 'backoff'
                         — reflects the graph store state via a READ-ONLY probe (never
                         grabs the writer lock). 'contended' = another process holds the
                         writer lock right now (transient with per-op leasing).
      ladybug_writer_pid: (optional) PID stamped as the current write-lease holder
      ollama_reachable:  TCP reachability of the resolved Ollama endpoint
      vllm_reachable:    TCP reachability of the resolved vLLM endpoint
      qdrant_reachable:  TCP reachability of the resolved Qdrant endpoint
      embed_model:       configured embedding model
      rerank_model:      configured cross-encoder rerank model
      warm:              whether the embed warm-ping has been fired this process
    """
    out: dict = {
        "code_version": "",
        "ladybug": "unavailable",
        "ollama_reachable": False,
        "vllm_reachable": False,
        "qdrant_reachable": False,
        "embed_model": "",
        "rerank_model": "",
        "warm": False,
    }
    try:
        out["code_version"] = _code_version()
    except Exception as exc:
        logger.debug("loci_health: code_version probe failed: %r", exc)
        pass
    try:
        out["ladybug"] = _ladybug_health_state()
        pid = _ladybug_writer_pid()
        if pid is not None:
            out["ladybug_writer_pid"] = pid
    except Exception as exc:
        logger.debug("loci_health: ladybug health-state probe failed: %r", exc)
        pass
    try:
        import backends
        # Bounded, fast even on the FIRST call with backends down: resolve each
        # endpoint URL ONCE (the resolvers are memoized) and pass a SHORT probe
        # timeout so the resolvers' own internal local probe cannot block for the
        # 1.0s default. Then run loci_health's own reachability check with the same
        # short timeout. Each probe is independent + fail-open: one raising /
        # unreachable backend must not mask the others.
        _PROBE_T = 0.5
        for key, resolver in (
            ("ollama_reachable", lambda: backends.ollama_url(_PROBE_T)),
            ("vllm_reachable", lambda: backends.vllm_url(_PROBE_T)),
            ("qdrant_reachable", lambda: backends.qdrant()[0]),
        ):
            try:
                out[key] = bool(backends._alive(resolver(), timeout=_PROBE_T))
            except Exception as exc:
                logger.debug("loci_health: reachability probe %s failed: %r", key, exc)
                pass
        try:
            out["embed_model"] = backends.embed_model()
        except Exception as exc:
            logger.debug("loci_health: embed_model probe failed: %r", exc)
            pass
        try:
            out["rerank_model"] = backends.rerank_model()
        except Exception as exc:
            logger.debug("loci_health: rerank_model probe failed: %r", exc)
            pass
    except Exception as exc:
        logger.debug("loci_health: backends import/probe block failed: %r", exc)
        pass
    try:
        import embed_ops
        out["warm"] = embed_ops.warmed()
    except Exception as exc:
        logger.debug("loci_health: embed warm-state probe failed: %r", exc)
        pass

    # Corpus durability. A 30-day default purge window silently deleted every
    # indexed finding older than a month on each process start, and nothing
    # reported it: coverage looked fine right after a re-index and collapsed at
    # the next restart. These two fields make that condition answerable from the
    # tool people already call, instead of from a set-difference against disk.
    try:
        import qdrant_ops
        days = qdrant_ops._retention_days()
        out["retention_days"] = days
        out["purge_active"] = days > 0
        if days > 0:
            out["purge_warning"] = (
                f"findings older than {days} days are deleted on every server start"
            )
    except Exception as exc:
        logger.debug("loci_health: retention probe failed: %r", exc)
        pass
    return json.dumps(out, indent=2)


# --------------------------------------------------------------------------- #
# rag_context_search internals
#
# Each stage is independently fail-open, matching the tool's contract: a dead
# expander, collection, decay pass or access-log write never aborts the search.
# --------------------------------------------------------------------------- #
def _rag_expand_queries(query: str) -> tuple[list[str], dict]:
    """Widen the recall pool with HyDE-lite expansion.

    Returns (search_queries, expansion_info). The original query always leads, so
    a degraded expander costs nothing. The cross-encoder re-pass ranks against the
    ORIGINAL query, so expansion widens recall without diluting precision.
    """
    try:
        import query_expand as _qe
        exp = _qe.expand(query, n_queries=3, n_keywords=6)
        sq = list(exp.get("queries") or [query])
        kw = exp.get("keywords") or []
        if kw:
            sq.append(" ".join(kw))
        seen: set = set()
        queries = [q for q in sq if q and not (q in seen or seen.add(q))] or [query]
        return queries, {"enabled": True, "degraded": bool(exp.get("degraded")),
                         "n_queries": len(queries), "keywords": kw}
    except Exception as exc:
        logger.debug("rag_context_search: query expansion failed (fail-open): %s", exc)
        return [query], {"enabled": True, "degraded": True, "error": str(exc)}


def _rag_search_collections(collections, search_queries, limit, agent_filter, errors) -> list[dict]:
    """Union hits across (collection x expanded query), deduped by (origin, id).

    Keeps the best bi-encoder score per key. A failing collection is recorded in
    `errors` and skipped rather than aborting the search.
    """
    best: dict = {}
    for col in collections:
        qf = agent_filter if col == "agent_core_chunks" else None
        for sq in search_queries:
            try:
                hits = _qdrant_search_collection(sq, collection_name=col, limit=limit, query_filter=qf)
            except Exception as exc:
                errors.append(f"{col}: {exc}")
                logger.warning("rag_context_search: collection %s failed: %s", col, exc)
                continue
            for h in hits:
                key = (h.get("origin"), h.get("id"))
                prev = best.get(key)
                if prev is None or float(h.get("score") or 0.0) > float(prev.get("score") or 0.0):
                    best[key] = h
    return list(best.values())


def _rag_apply_decay(results: list[dict]) -> None:
    """Rescore findings in place with Ebbinghaus exponential time decay.

    Only touches rows from the main findings collection; rows without a usable
    timestamp keep their raw score.
    """
    try:
        now_ts = time.time()
        for r in results:
            if r.get("origin") != QDRANT_COLLECTION_PREFIX:
                continue
            ts_val = r.get("created_at_ts") or r.get("ts")
            if ts_val is None:
                continue
            # ts may be an ISO string; created_at_ts is always an int epoch
            try:
                age_days = (now_ts - float(ts_val)) / 86400.0
                if age_days < 0:
                    age_days = 0.0
                raw_score = float(r.get("score") or 0.0)
                r["score"] = round(raw_score * math.exp(-_MEMORY_DECAY_LAMBDA * age_days), 4)
                r["decay_applied"] = True
            except Exception as exc:
                logger.debug("rag_context_search: recency decay scoring failed (fail-open): %r", exc)
    except Exception as exc:
        logger.debug("rag_context_search: decay rescoring failed: %s", exc)


def _rag_cross_encode(results: list[dict], query: str) -> None:
    """Re-rank the top 20 in place against the ORIGINAL query, for consistent
    cross-collection ordering. No-op when the cross-encoder is unavailable."""
    ce = _get_cross_encoder()
    if ce is None or len(results) <= 1:
        return
    try:
        # Same budget as qdrant_ops._ce_rerank — this is the SECOND cross-encoder,
        # on rag_context_search's final cross-collection re-pass, and it was missed
        # when the first was fixed. 512 chars against a 1,048-char median finding
        # scored below using no reranker at all.
        pairs = [(query, str(r.get('text', r.get('content', '')))[:_RERANK_MAX_CHARS])
                 for r in results[:20]]
        for r, sc in zip(results[:20], ce.predict(pairs)):
            r['final_ce_score'] = round(float(sc), 4)
        results[:20] = sorted(results[:20], key=lambda r: r.get('final_ce_score', -999), reverse=True)
    except Exception as exc:
        logger.debug('Final CE re-pass failed: %s', exc)


def _rag_record_access(results: list[dict], query: str) -> None:
    """Append a last_accessed marker to each returned finding's JSONL.

    Best-effort per row: this must never block the response.
    """
    access_ts = int(time.time())
    for r in results:
        try:
            if r.get("origin") != QDRANT_COLLECTION_PREFIX:
                continue
            finding_id, inv_id = r.get("id"), r.get("investigation_id")
            if not finding_id or not inv_id:
                continue
            findings_path = _inv_dir(inv_id) / "findings.jsonl"
            if not findings_path.exists():
                continue
            _append_jsonl(findings_path, {
                "id": finding_id,
                "investigation_id": inv_id,
                "record_type": "access",
                "last_accessed": access_ts,
                "query": query[:200],
            })
        except Exception as exc:
            logger.debug("rag access-log write-back failed for %s: %r", r.get("id"), exc)


@mcp.tool()
def rag_context_search(
    query: str,
    limit: int = 10,
    collections: Optional[list] = None,
    budget_chars: int = 6000,
    exclude_types: Optional[list] = None,
    decay: bool = True,
    expand_query: Optional[bool] = None,
) -> str:
    """
    Hybrid RAG search over Qdrant corpus. Returns prompt-ready context with cited sources.
    ALWAYS uses Qdrant — no keyword fallback. Raises rag_required error if Qdrant unavailable.

    Searches hermes_memory (task findings) and agent_core_chunks (1.84M knowledge base)
    by default, merges results, reranks with cross-encoder, assembles cited context.

    Args:
        query: Natural language search query.
        limit: Results per collection (default 10).
        collections: Override collections list (default: ["hermes_memory", "agent_core_chunks"]).
        budget_chars: Max characters in assembled context (default 6000).
        exclude_types: Payload 'type' values to exclude from agent_core_chunks results.
                       Default None → ['gps_trajectory'] to suppress high-volume GPS pings.
                       Pass [] to disable filtering.
        decay: If True (default), apply Ebbinghaus exponential time-decay rescoring to
               findings from the main investigation collection before ranking.
               Set False to disable decay and use raw similarity scores.
        expand_query: Fan retrieval out over LLM-generated query paraphrases + keywords
               (query_expand) before the cross-encoder re-pass, lifting recall. None
               (default) reads env LOCI_RAG_EXPAND (default ON); True/False force it.
               Fail-open: if the local generator is unavailable, falls back to the bare
               query with no error. Gated on judge-eval evidence: +4% nDCG@10, no regression.

    Returns:
        JSON with {query, context, sources, total_chars, truncated, result_count, mode,
                   collections_searched, qdrant_available}
    """
    if not query or not query.strip():
        return json.dumps({"error": "query must not be empty", "results": [], "query": query})

    if exclude_types is None:
        exclude_types = ["gps_trajectory"]

    _default_cols = [QDRANT_COLLECTION_PREFIX] + ([_CODE_CHUNKS_COLLECTION] if _CODE_CHUNKS_COLLECTION else [])
    _collections = list(collections) if collections else _default_cols
    client, _col = _get_qdrant()
    qdrant_available = client is not None

    if not qdrant_available:
        return json.dumps({
            "mode": "rag_required",
            "error": "RAG_REQUIRED: Qdrant unavailable. Check QDRANT_URL and QDRANT_API_KEY.",
            "query": query,
            "results": [],
            "qdrant_available": False,
        }, indent=2)

    all_results: list[dict] = []
    errors: list[str] = []
    expansion_info: Optional[dict] = None

    # Optional query expansion: fan retrieval out over LLM paraphrases + keywords for recall.
    # Gate: expand_query arg wins; else env LOCI_RAG_EXPAND (default ON). Fail-open — if the
    # generator is down, `expand` returns just the original query, so retrieval is unaffected.
    _do_expand = expand_query
    if _do_expand is None:
        _do_expand = os.environ.get("LOCI_RAG_EXPAND", "1").strip().lower() not in ("0", "false", "no", "off", "")
    search_queries = [query]
    if _do_expand:
        search_queries, expansion_info = _rag_expand_queries(query)

    # Build the GPS-exclusion filter for agent_core_chunks (only).
    _agent_filter = None
    if exclude_types:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        _agent_filter = Filter(must_not=[FieldCondition(key="type", match=MatchAny(any=exclude_types))])

    # Retrieve per (collection x expanded query), unioning hits deduped by (origin, id) keeping
    # the best bi-encoder score. The cross-encoder re-pass below re-ranks against the ORIGINAL
    # query, so expansion only widens the candidate pool (recall) without diluting precision.
    all_results.extend(_rag_search_collections(_collections, search_queries, limit, _agent_filter, errors))

    # Apply Ebbinghaus exponential time-decay rescoring to findings from the main
    # investigation collection only (agent_core_chunks is static knowledge, not time-sensitive).
    if decay:
        _rag_apply_decay(all_results)

    # Merge-sort by score descending across all collections
    all_results.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)

    # Final cross-encoder re-pass for consistent cross-collection ranking
    _rag_cross_encode(all_results, query)

    # Best-effort access tracking: append last_accessed timestamp to each returned finding's JSONL.
    # Failures are silently ignored — this must never block the response.
    _rag_record_access(all_results, query)

    ctx = context_assemble(all_results, query, budget_chars=budget_chars)
    # Report retrieval honestly. This used to claim mode="rag_hybrid" no matter
    # what, so a search where EVERY collection raised was indistinguishable from
    # a genuine zero-hit search: a misconfigured embedder or a dimension mismatch
    # read back to the caller as "nothing is known about this topic".
    # Count distinct collections, not error entries — one broken collection
    # raises once per expanded query.
    _failed_cols = {e.split(":", 1)[0] for e in errors}
    _failed = len(_failed_cols)
    if _failed and _failed >= len(_collections):
        ctx["mode"] = "rag_failed"
    elif _failed:
        ctx["mode"] = "rag_degraded"
    else:
        ctx["mode"] = "rag_hybrid"
    ctx["collections_searched"] = _collections
    ctx["collections_failed"] = sorted(_failed_cols)
    ctx["qdrant_available"] = True
    if expansion_info is not None:
        ctx["query_expansion"] = expansion_info
    if errors:
        ctx["collection_errors"] = errors
    return json.dumps(ctx, indent=2)


# ---------------------------------------------------------------------------
# Tool: memory_surface — proactive context surfacing
# ---------------------------------------------------------------------------

def _surface_query_filter(investigation_id: Optional[str]) -> Optional[object]:
    """Build an investigation-scoped Qdrant filter for memory_surface.

    Fail-open: on any construction error, logs a warning and returns None so
    the caller falls back to an unscoped (wider) search rather than erroring.
    """
    if not investigation_id:
        return None
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        return Filter(
            must=[FieldCondition(key="investigation_id", match=MatchValue(value=investigation_id))]
        )
    except Exception as exc:
        logger.warning(
            "memory_surface: investigation filter construction failed for %s "
            "- search will NOT be scoped: %r", investigation_id, exc,
        )
        return None


def _surface_apply_decay(rows: list[dict]) -> None:
    """Rescore memory_surface rows in place with Ebbinghaus exponential time decay.

    Applies decay only if _MEMORY_DECAY_LAMBDA is defined on this module. Unlike
    _rag_apply_decay, this defaults created_at_ts to now (no-op decay) rather than
    skipping rows without a timestamp, and does not restrict to the findings collection.
    Fail-open — decay is an optional enhancement and never breaks the tool.
    """
    _decay_lambda = globals().get("_MEMORY_DECAY_LAMBDA")
    now_ts = time.time()
    if _decay_lambda is not None:
        try:
            for r in rows:
                created_ts = float(r.get("created_at_ts") or now_ts)
                age_days = (now_ts - created_ts) / 86400.0
                decay = math.exp(-float(_decay_lambda) * age_days)
                r["score"] = round(float(r.get("score") or 0.0) * decay, 4)
        except Exception as exc:
            logger.debug("memory_surface: recency decay scoring failed (fail-open): %r", exc)
            pass  # decay is optional enhancement; never break the tool


def _surface_rows(top_results: list[dict], ctx_prefix: str, investigation_id: Optional[str]) -> list[dict]:
    """Build the memory_surface 'surfaced' response rows from top_results.

    ctx_prefix is the pre-computed first-8-whitespace-split-words prefix of the
    original context, used verbatim in the relevance_note fallback label.
    """
    surfaced = []
    for r in top_results:
        finding_id = r.get("id") or r.get("finding_id") or ""
        text = str(r.get("text") or r.get("content") or "")
        source = str(r.get("source") or r.get("origin") or "")
        inv_id = r.get("investigation_id") or investigation_id or ""
        score = round(float(r.get("score") or 0.0), 4)

        # Generate relevance_note: simple label based on context prefix
        relevance_note = f"Related to: {ctx_prefix}"

        surfaced.append({
            "finding_id": finding_id,
            "text": text[:300] if len(text) > 300 else text,
            "source": source,
            "relevance_note": relevance_note,
            "score": score,
            "investigation_id": inv_id,
        })
    return surfaced


@mcp.tool()
def memory_surface(
    context: str,
    investigation_id: Optional[str] = None,
    top_k: int = 5,
) -> str:
    """
    Proactively surface prior findings most relevant to the agent's current working context.

    Unlike rag_context_search (which requires a precise query), memory_surface accepts
    a free-form paragraph describing what the agent is doing and returns the most
    tangentially-relevant prior findings using a lower similarity threshold (0.25 vs 0.5+).
    This is designed for passive context injection — call it at the start of a task to
    surface related memory without knowing exactly what to search for.

    Args:
        context:          A paragraph describing the agent's current working context.
        investigation_id: Optional — if provided, restrict results to this investigation.
        top_k:            How many surfaced results to return (default 5).

    Returns:
        JSON with {surfaced: [{finding_id, text, source, relevance_note, score,
                               investigation_id}], context_used, count}
    """
    if not context or not context.strip():
        return json.dumps({
            "error": "context must not be empty",
            "surfaced": [],
            "context_used": context,
            "count": 0,
        })

    try:
        client, _col = _get_qdrant()
        if client is None:
            return json.dumps({
                "error": "memory_surface requires Qdrant",
                "surfaced": [],
                "context_used": context,
                "count": 0,
            })

        # Build investigation filter if requested
        query_filter = _surface_query_filter(investigation_id)

        # Fetch top_k * 3 candidates with a lower threshold via _qdrant_search_collection
        fetch_limit = top_k * 3
        try:
            candidates = _qdrant_search_collection(
                context,
                collection_name=QDRANT_COLLECTION_PREFIX,
                limit=fetch_limit,
                query_filter=query_filter,
            )
        except RuntimeError as _rte:
            _rte_msg = str(_rte)
            if "qdrant_unavailable" in _rte_msg or "embedding_unavailable" in _rte_msg:
                return json.dumps({
                    "error": "memory_surface requires Qdrant",
                    "surfaced": [],
                    "context_used": context,
                    "count": 0,
                })
            raise
        except Exception as _exc:
            return json.dumps({
                "error": f"memory_surface search failed: {_exc}",
                "surfaced": [],
                "context_used": context,
                "count": 0,
            })

        # Apply lower score threshold (0.25) to allow tangentially relevant findings
        _SURFACE_SCORE_THRESHOLD = 0.25
        filtered = [r for r in candidates if float(r.get("score") or 0.0) >= _SURFACE_SCORE_THRESHOLD]

        # Apply Ebbinghaus decay if _MEMORY_DECAY_LAMBDA is defined on this module
        _surface_apply_decay(filtered)

        # Re-sort after possible decay adjustment, then take top_k
        filtered.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
        top_results = filtered[:top_k]

        # Build short context prefix for relevance note fallback
        _ctx_words = context.strip().split()
        _ctx_prefix = " ".join(_ctx_words[:8])

        surfaced = _surface_rows(top_results, _ctx_prefix, investigation_id)

        return json.dumps({
            "surfaced": surfaced,
            "context_used": context[:200] if len(context) > 200 else context,
            "count": len(surfaced),
        }, indent=2)

    except Exception as _top_exc:
        return json.dumps({
            "error": f"memory_surface error: {_top_exc}",
            "surfaced": [],
            "context_used": context[:200] if len(context) > 200 else context,
            "count": 0,
        })


# ---------------------------------------------------------------------------
# Mnemosyne sleep / consolidation
# ---------------------------------------------------------------------------

def _find_most_recent_investigation() -> tuple[str, list[dict]] | tuple[None, None]:
    """Find the most recently updated investigation and its last 10 findings.

    Returns (investigation_id, findings) or (None, None) on any error.
    Fail-open — never raises.
    """
    try:
        if not MEMORY_DIR.exists():
            return None, None
        best_id = None
        best_ts = ""
        for d in MEMORY_DIR.iterdir():
            if not d.is_dir():
                continue
            manifest = _load_manifest(d.name)
            if manifest is None:
                continue
            ts = str(manifest.get("updated_at") or "")
            if ts > best_ts:
                best_ts = ts
                best_id = d.name
        if best_id is None:
            return None, None
        findings_path = MEMORY_DIR / best_id / "findings.jsonl"
        findings = _read_jsonl(findings_path)
        return best_id, findings[-10:] if len(findings) > 10 else findings
    except Exception:
        return None, None


def _declared_causal_edges(findings: list[dict]) -> list[dict]:
    """Causal edges from author-declared lineage — the derived_from links.

    investigation_store records derived_from so a retraction can follow what was
    built on a false fact. That is the same relation causal inference tries to
    guess from text, except stated by whoever wrote the finding rather than
    inferred from keywords, so it is strictly better evidence and does not need
    to be re-derived.

    Measured on the live corpus, 136 investigations:

        heuristic         62 edges
        declared lineage 524 edges   (234 findings carrying 565 links)

    causal_edges.jsonl nevertheless held zero records everywhere, because the
    producer is only reached from memory_consolidate behind `not dry_run` and
    `len(findings) >= 3`. The census reading of "zero output" was a statement
    about invocation, not about either producer — the heuristic is not inert, it
    had simply never run.

    524 < 565 because derived_from also accepts free-text claim strings, and ids
    pointing outside the investigation have no node to attach to.

    Direction matches the heuristic's convention: source is the antecedent (the
    finding depended on), target is the finding that declared the dependency.
    """
    known = {str(f.get("id")) for f in findings if f.get("id")}
    edges: list[dict] = []
    for f in findings:
        target = str(f.get("id") or "")
        if not target:
            continue
        raw = f.get("derived_from")
        if not raw:
            continue
        parents = raw if isinstance(raw, (list, tuple)) else [raw]
        for parent in parents:
            source = str(parent or "").strip()
            # derived_from also accepts free-text claim strings, not just ids.
            # Only ids resolvable within this investigation become edges; a
            # claim string has no node to point at.
            if not source or source == target or source not in known:
                continue
            edges.append({
                "id": str(uuid.uuid4()),
                "source_id": source,
                "target_id": target,
                "edge_type": "caused_by",
                # Declared, not inferred. Higher than the heuristic's 0.5 because
                # an author saying "this builds on that" is testimony, not a guess.
                "confidence": 0.9,
                "inferred_at": _now(),
                "method": "declared_lineage",
            })
    return edges


def _heuristic_causal_edges(findings: list[dict]) -> list[dict]:
    """Infer causal edges heuristically from finding texts.

    If finding B's text references finding A's id or contains causal keywords
    alongside A's text snippet, emit a caused_by edge with confidence 0.5.
    Returns a (possibly empty) list of edge dicts.
    """
    _CAUSAL_KEYWORDS = re.compile(
        r"\b(because|caused|led to|resulted in|after|following|due to|triggered)\b",
        re.I,
    )
    edges = []
    for i, b in enumerate(findings):
        b_text = str(b.get("text") or "")
        b_id = str(b.get("id") or "")
        if not b_text:
            continue
        for j, a in enumerate(findings):
            if j >= i:
                break
            a_id = str(a.get("id") or "")
            a_text = str(a.get("text") or "")
            if not a_id or not a_text:
                continue
            # Cross-reference: B mentions A's id, or B contains causal phrasing
            # and A's text appears as a substring in B's text.
            id_ref = a_id in b_text
            snippet = a_text[:60].strip().lower()
            snippet_ref = len(snippet) > 10 and snippet in b_text.lower()
            keyword_match = bool(_CAUSAL_KEYWORDS.search(b_text))
            if id_ref or (snippet_ref and keyword_match):
                edges.append({
                    "id": str(uuid.uuid4()),
                    "source_id": a_id,
                    "target_id": b_id,
                    "edge_type": "caused_by",
                    "confidence": 0.5,
                    "inferred_at": _now(),
                    "method": "heuristic",
                })
    return edges


def _merge_causal_edges(inferred: list[dict], declared: list[dict]) -> list[dict]:
    """Union two edge sets; declared lineage wins any (source, target) collision.

    An author's derived_from is testimony about their own reasoning. Anything the
    text heuristic or the LLM pass concluded about the same pair is a guess at
    the same fact, so it is replaced rather than kept alongside — two edges for
    one relation would double-count it downstream.
    """
    by_pair: dict[tuple, dict] = {}
    for edge in inferred:
        by_pair[(edge.get("source_id"), edge.get("target_id"))] = edge
    for edge in declared:
        by_pair[(edge.get("source_id"), edge.get("target_id"))] = edge
    return list(by_pair.values())


def _run_causal_inference(investigation_id: str, findings: list[dict]) -> int:
    """Run causal inference on the last findings of an investigation.

    Attempts an LLM slow path first; falls back to heuristic if LLM unavailable.
    Writes inferred edges to {MEMORY_DIR}/{investigation_id}/causal_edges.jsonl.
    Returns the number of edges written.  Fail-open — never raises.
    """
    if not findings:
        return 0
    edges: list[dict] = []
    try:
        from memcheck import llm as _llm  # type: ignore
        if _llm.llm_available():
            numbered = "\n".join(
                f"{idx + 1}. [{f.get('id', '?')}] {str(f.get('text', ''))[:300]}"
                for idx, f in enumerate(findings)
            )
            prompt = (
                "Given these investigation findings in chronological order:\n"
                f"{numbered}\n\n"
                "Identify causal relationships. For each pair where A caused or "
                "enabled B, output a JSON line: "
                '{"source_id": "<id of A>", "target_id": "<id of B>", '
                '"edge_type": "<caused_by|enabled_by|correlates_with>", '
                '"confidence": <0.0-1.0>}. '
                "edge_type must be 'caused_by', 'enabled_by', or 'correlates_with'. "
                "Only output confident relationships (confidence >= 0.6). "
                "If none, output empty list []."
            )
            raw = _llm.call_llm(prompt, timeout=60.0)
            if raw:
                # Parse JSON lines or a JSON array from the response.
                valid_types = {"caused_by", "enabled_by", "correlates_with"}
                for line in raw.splitlines():
                    line = line.strip().lstrip("- ")
                    if not line:
                        continue
                    # strip trailing comma for array-style output
                    line = line.rstrip(",")
                    try:
                        obj = json.loads(line)
                    except Exception:
                        # try to find an embedded JSON object
                        m = re.search(r'\{[^}]+\}', line)
                        if m:
                            try:
                                obj = json.loads(m.group(0))
                            except Exception as exc:
                                logger.debug("_run_causal_inference: fail-open swallow: %r", exc)
                                continue
                        else:
                            continue
                    if not isinstance(obj, dict):
                        continue
                    src = str(obj.get("source_id") or "")
                    tgt = str(obj.get("target_id") or "")
                    etype = str(obj.get("edge_type") or "")
                    conf = float(obj.get("confidence") or 0.0)
                    if src and tgt and etype in valid_types and conf >= 0.6:
                        edges.append({
                            "id": str(uuid.uuid4()),
                            "source_id": src,
                            "target_id": tgt,
                            "edge_type": etype,
                            "confidence": conf,
                            "inferred_at": _now(),
                            "method": "llm_slow_path",
                        })
    except Exception as exc:
        logger.debug("_run_causal_inference: LLM causal-edge slow path failed (fail-open): %r", exc)
        pass  # LLM path failed; fall through to heuristic

    if not edges:
        try:
            edges = _heuristic_causal_edges(findings)
        except Exception:
            edges = []

    try:
        edges = _merge_causal_edges(edges, _declared_causal_edges(findings))
    except Exception as exc:
        logger.debug("_run_causal_inference: declared-lineage pass failed (fail-open): %r", exc)

    if not edges:
        return 0

    try:
        edges_path = MEMORY_DIR / investigation_id / "causal_edges.jsonl"
        for edge in edges:
            _append_jsonl(edges_path, edge)
    except Exception:
        return 0

    return len(edges)


@mcp.tool()
def memory_consolidate(dry_run: bool = False) -> str:
    """
    Run Mnemosyne sleep/consolidation cycle.
    Merges old working_memory entries into episodic memory, reducing DB size.
    Safe to run periodically (daily or after large investigation sessions).

    After the Mnemosyne pass, runs a causal inference slow path on the most
    recently active investigation (if it has ≥3 findings), writing inferred
    edges to causal_edges.jsonl.  The causal step is fail-open and never
    blocks the consolidation result.

    Args:
        dry_run: If True, preview consolidation without executing.

    Returns JSON with consolidation stats and causal_edges_inferred count.
    """
    import json as _json
    causal_edges_inferred = 0
    try:
        from mnemosyne.core.memory import Mnemosyne
        m = Mnemosyne()
        result = m.sleep_all_sessions(dry_run=dry_run)
        _event_log_append({"op": "consolidate", "dry_run": dry_run,
                           "result_summary": str(result)[:200] if result else ""})

        # Causal inference slow path (fail-open).
        try:
            if not dry_run:
                inv_id, findings = _find_most_recent_investigation()
                if inv_id and findings and len(findings) >= 3:
                    causal_edges_inferred = _run_causal_inference(inv_id, findings)
        except Exception:
            causal_edges_inferred = 0

        return _json.dumps({
            "status": "ok",
            "dry_run": dry_run,
            "result": result if isinstance(result, dict) else str(result),
            "causal_edges_inferred": causal_edges_inferred,
        })
    except Exception as e:
        return _json.dumps({
            "status": "error",
            "error": "internal error",
            "type": type(e).__name__,
            "causal_edges_inferred": causal_edges_inferred,
        })


@mcp.tool()
def causal_edges_list(investigation_id: str) -> str:
    """
    List causal edges inferred for an investigation.

    Edges are written by the causal inference slow path in memory_consolidate.
    Each edge describes a directional relationship between two findings.

    Args:
        investigation_id: The investigation whose causal_edges.jsonl to read.

    Returns JSON with {edges: [{id, source_id, target_id, edge_type,
                                confidence, inferred_at}], count}.
    """
    if not investigation_id:
        return json.dumps({"error": "investigation_id is required"})
    inv_path = MEMORY_DIR / investigation_id
    if not inv_path.exists():
        return json.dumps({"error": f"Investigation '{investigation_id}' not found"})
    try:
        edges_path = inv_path / "causal_edges.jsonl"
        raw = _read_jsonl(edges_path)
        edges = [
            {
                "id": str(e.get("id") or ""),
                "source_id": str(e.get("source_id") or ""),
                "target_id": str(e.get("target_id") or ""),
                "edge_type": str(e.get("edge_type") or ""),
                "confidence": float(e.get("confidence") or 0.0),
                "inferred_at": str(e.get("inferred_at") or ""),
            }
            for e in raw
            if isinstance(e, dict)
        ]
        return json.dumps({"edges": edges, "count": len(edges)})
    except Exception as exc:
        return json.dumps({"error": str(exc), "edges": [], "count": 0})

# ---------------------------------------------------------------------------
# Metamemory
# ---------------------------------------------------------------------------


def _confidence_retrieve(query: str, top_k: int) -> tuple[list, Optional[str]]:
    """
    Run the retrieval steps for memory_confidence: empty-query guard, Qdrant
    availability check, embedding, and vector search.

    Returns (results, hard_stop_basis). When hard_stop_basis is not None the
    caller should short-circuit with a hard-stop payload using that basis
    string ("empty_query" / "qdrant_unavailable" / "embed_failed") and
    results is always []. When hard_stop_basis is None, results holds the
    (possibly empty) search hits — an empty list there means "no_trace",
    which the caller handles separately.
    """
    if not query:
        return [], "empty_query"

    client, col = _get_qdrant()
    if client is None:
        return [], "qdrant_unavailable"

    try:
        emb = _embed(query)
    except Exception as exc:
        logger.debug("memory_confidence embed failed: %r", exc)
        emb = None
    if not emb:
        return [], "embed_failed"

    try:
        results = _query_points_compat(client, col, emb, limit=top_k)
    except Exception as exc:
        # NOT "no_trace": an empty list from a broken search is a different claim
        # from an empty list from a healthy one, and the caller has to be able to
        # tell them apart.
        logger.warning("memory_confidence search failed: %r", exc)
        return [], "search_failed"

    return results, None


def _confidence_cues(results: list) -> dict:
    """
    Compute the five metamemory cues for memory_confidence from a list of
    Qdrant search results.

    Returns {fluency, accessibility, source_div, corroboration, trust,
    top_text}. top_text is the truncated text/content of the first result
    that has any (not necessarily results[0]) — "" if none do.
    """
    scores = [float(getattr(r, "score", 0.0) or 0.0) for r in results]

    # Cue 1: Fluency — cosine of top hit (retrieval ease proxy)
    fluency = scores[0] if scores else 0.0

    # Cue 2: Accessibility — mean of top-4 scores (Koriat amount-retrieved)
    accessibility = sum(scores[:4]) / max(1, len(scores[:4]))

    # Cue 3: Source diversity — distinct investigation_id or source values
    sources: set = set()
    conf_tiers = {"high": 1.0, "medium": 0.7, "low": 0.4}
    conf_vals = []
    max_occurrences = 0
    top_text = ""
    for r in results:
        pl = dict(getattr(r, "payload", None) or {})
        src = pl.get("investigation_id") or pl.get("source") or pl.get("record_type") or ""
        if src:
            sources.add(src)
        conf = str(pl.get("confidence", "") or "").lower()
        conf_vals.append(conf_tiers.get(conf, 0.5))
        occ = int(pl.get("occurrences", 0) or 0)
        if occ > max_occurrences:
            max_occurrences = occ
        if not top_text:
            top_text = str(pl.get("text") or pl.get("content") or "")[:200]

    source_div = len(sources)

    # Cue 4: Corroboration — log-saturated occurrence count
    corroboration = math.log1p(max_occurrences) / math.log1p(20)  # saturates ~20

    # Cue 5: Trust — mean confidence tier of top hits
    trust = sum(conf_vals) / max(1, len(conf_vals))

    return {
        "fluency": fluency,
        "accessibility": accessibility,
        "source_div": source_div,
        "corroboration": corroboration,
        "trust": trust,
        "top_text": top_text,
    }


def _confidence_verdict(cues: dict) -> tuple[float, str, str]:
    """
    Combine the five metamemory cues into a calibrated (confidence, basis,
    recommendation) verdict for memory_confidence. Pure function of `cues`.
    """
    fluency = cues["fluency"]
    accessibility = cues["accessibility"]
    source_div = cues["source_div"]
    corroboration = cues["corroboration"]
    trust = cues["trust"]

    # Weighted combination (weights: source_div and trust dominate fluency;
    # this ordering follows Fleming 2010 — source/recollection > familiarity/fluency).
    W = {
        "fluency":       0.10,
        "accessibility": 0.20,
        "source_div":    0.30,
        "corroboration": 0.15,
        "trust":         0.25,
    }
    # Normalise source_div to [0,1] (capped at 5 distinct sources = max)
    source_div_norm = min(1.0, source_div / 5.0)

    raw = (W["fluency"]       * fluency
           + W["accessibility"] * accessibility
           + W["source_div"]    * source_div_norm
           + W["corroboration"] * corroboration
           + W["trust"]         * trust)
    # Sigmoid to keep in (0,1); shift so 0.5 raw → ~0.5 output.
    confidence = 1.0 / (1.0 + math.exp(-8 * (raw - 0.5)))

    # Basis: prefer recollection (source_div) over familiarity (fluency).
    if source_div >= 2:
        basis = "recollection"     # multiple independent sources corroborate
    elif corroboration > 0.3:
        basis = "corroboration"    # same source seen many times
    elif trust > 0.7:
        basis = "trust"            # high-confidence single source
    else:
        basis = "familiarity"      # only cosine match, low corroboration

    if confidence >= 0.75:
        recommendation = "confident — cite memory"
    elif confidence >= 0.50:
        recommendation = "moderate — use with source citation"
    elif confidence >= 0.30:
        recommendation = "low — verify with investigation tools before asserting"
    else:
        recommendation = "unreliable — investigate fresh before asserting"

    return confidence, basis, recommendation


@mcp.tool()
def memory_confidence(
    query: str,
    top_k: int = 8,
) -> str:
    """
    Estimate how reliably hermes_memory knows about a topic (metamemory).

    Combines five evidence cues into a calibrated confidence score:
      fluency        — cosine similarity of top hit to query (retrieval ease)
      accessibility  — mean score of top-4 hits (amount of partial info recalled)
      source_div     — number of distinct sources/investigations in top results
      corroboration  — max occurrences across top hits (repeated evidence)
      trust          — mean confidence tier (high/medium/low) of top hits

    Fluency is down-weighted relative to source_div and trust because it tracks
    retrieval ease, not correctness (Koriat 1993 over-confidence mechanism).

    Use this before asserting a memory-derived claim to get a calibrated estimate
    of reliability. Low confidence → verify with investigation tools first.

    Args:
        query: The claim or topic to estimate confidence for.
        top_k: Number of results to base the estimate on (default 8).

    Returns:
        JSON with {confidence, basis, cues, top_hit_preview, recommendation}.
    """
    results, hard_stop_basis = _confidence_retrieve(query, top_k)
    if hard_stop_basis is not None:
        return json.dumps({
            "confidence": 0.0, "basis": hard_stop_basis,
            "cues": {}, "top_hit_preview": "", "recommendation": "verify",
        })

    if not results:
        return json.dumps({
            "confidence": 0.0, "basis": "no_trace",
            "cues": {"fluency": 0.0, "accessibility": 0.0, "source_div": 0,
                     "corroboration": 0, "trust": 0.0},
            "top_hit_preview": "",
            "recommendation": "no memory found — investigate before asserting",
        })

    cues = _confidence_cues(results)
    fluency = cues["fluency"]
    accessibility = cues["accessibility"]
    source_div = cues["source_div"]
    corroboration = cues["corroboration"]
    trust = cues["trust"]
    top_text = cues["top_text"]

    confidence, basis, recommendation = _confidence_verdict(cues)

    return json.dumps({
        "confidence": round(confidence, 3),
        "basis": basis,
        "cues": {
            "fluency":         round(fluency, 3),
            "accessibility":   round(accessibility, 3),
            "source_diversity": source_div,
            "corroboration":   round(corroboration, 3),
            "trust":           round(trust, 3),
        },
        "top_hit_preview": top_text,
        "recommendation": recommendation,
    }, indent=2)


# ---------------------------------------------------------------------------
# Memory tier management helpers
# ---------------------------------------------------------------------------

def _change_finding_tier(investigation_id: str, finding_id: str, new_tier: str) -> dict:
    """
    Core logic for memory_promote / memory_demote.

    Rewrites findings.jsonl atomically, updating the tier field of the target
    finding. Returns a dict with {finding_id, old_tier, new_tier, ok} or {error}.
    All Qdrant operations are fail-open.
    """
    if new_tier not in {"hot", "warm", "cold"}:
        return {"error": "tier must be one of: hot, warm, cold"}

    manifest = _load_manifest(investigation_id)
    if not manifest:
        return {"error": f"Investigation '{investigation_id}' not found."}

    findings_path = _inv_dir(investigation_id) / "findings.jsonl"
    findings = _read_jsonl(findings_path)

    target = None
    for f in findings:
        if str(f.get("id", "")) == finding_id:
            target = f
            break
    if target is None:
        return {"error": f"Finding '{finding_id}' not found in investigation '{investigation_id}'."}

    old_tier = target.get("tier", "warm")
    if old_tier == new_tier:
        return {"finding_id": finding_id, "old_tier": old_tier, "new_tier": new_tier, "ok": True}

    # Update the finding in-memory
    for f in findings:
        if str(f.get("id", "")) == finding_id:
            f["tier"] = new_tier

    # Atomically rewrite the JSONL file
    _lock_path = _inv_dir(investigation_id) / ".lock"
    try:
        with open(_lock_path, "w") as _lock_fh:
            fcntl.flock(_lock_fh, fcntl.LOCK_EX)
            try:
                dir_ = findings_path.parent
                with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as tf:
                    for f in findings:
                        tf.write(json.dumps(f) + "\n")
                    tmp_path = Path(tf.name)
                tmp_path.replace(findings_path)
            finally:
                fcntl.flock(_lock_fh, fcntl.LOCK_UN)
    except Exception as exc:
        return {"error": f"Failed to rewrite findings.jsonl: {exc}"}

    text = str(target.get("text", "") or "")

    # Handle Qdrant changes based on tier transition
    try:
        if new_tier == "cold":
            # Remove from Qdrant vector index
            client, col = _get_qdrant()
            if client is not None:
                try:
                    from qdrant_client.models import PointIdsList
                    client.delete(col, points_selector=PointIdsList(points=[finding_id]))
                except Exception as exc:
                    logger.warning("Qdrant delete failed (demote to cold) — JSONL updated: %s", exc)
        elif new_tier in ("warm", "hot") and old_tier == "cold":
            # Re-index in Qdrant (was cold, now searchable)
            _qdrant_upsert(finding_id, text, target)
        elif new_tier == "hot" and old_tier == "warm":
            # Already in Qdrant; just ensure it stays indexed (upsert is idempotent)
            _qdrant_upsert(finding_id, text, target)
    except Exception as exc:
        logger.warning("Qdrant tier-change operation failed (fail-open): %s", exc)

    # If promoting to hot: append text snippet to manifest notes
    if new_tier == "hot":
        try:
            snippet = text[:200]
            notes = manifest.get("notes") or ""
            manifest["notes"] = (notes + "; " + snippet) if notes else snippet
            _save_manifest(manifest)
        except Exception as exc:
            logger.warning("manifest notes update failed (fail-open): %s", exc)

    return {"finding_id": finding_id, "old_tier": old_tier, "new_tier": new_tier, "ok": True}


# ---------------------------------------------------------------------------
# Tool: memory_promote
# ---------------------------------------------------------------------------

@mcp.tool()
def memory_promote(investigation_id: str, finding_id: str, tier: str) -> str:
    """
    Promote a finding to a higher memory tier.

    Memory tiers:
      hot  — text snippet added to manifest notes (instantly in-context) + Qdrant indexed.
      warm — Qdrant indexed (default searchable tier).
      cold — JSONL only; NOT in Qdrant (archived, not vector-searchable).

    Typical promotions: cold → warm, warm → hot.

    Args:
        investigation_id: Investigation identifier.
        finding_id: UUID of the finding to promote.
        tier: Target tier — "hot", "warm", or "cold".

    Returns:
        JSON: {finding_id, old_tier, new_tier, ok: true}
        On error: {error: "<message>"}
    """
    try:
        result = _change_finding_tier(investigation_id, finding_id, tier)
        return json.dumps(result, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: memory_demote
# ---------------------------------------------------------------------------

@mcp.tool()
def memory_demote(investigation_id: str, finding_id: str, tier: str) -> str:
    """
    Demote a finding to a lower memory tier.

    Memory tiers:
      hot  — text snippet added to manifest notes (instantly in-context) + Qdrant indexed.
      warm — Qdrant indexed (default searchable tier).
      cold — JSONL only; NOT in Qdrant (archived, not vector-searchable).

    Typical demotions: hot → warm, warm → cold.
    Demoting to cold removes the vector from Qdrant so it no longer appears in
    semantic searches.

    Args:
        investigation_id: Investigation identifier.
        finding_id: UUID of the finding to demote.
        tier: Target tier — "hot", "warm", or "cold".

    Returns:
        JSON: {finding_id, old_tier, new_tier, ok: true}
        On error: {error: "<message>"}
    """
    try:
        result = _change_finding_tier(investigation_id, finding_id, tier)
        return json.dumps(result, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool: investigation_reason  (deep_think -> loci in-server reasoning surface)
# ---------------------------------------------------------------------------

# General-purpose adversarial mandates (ported from deep_think PERSPECTIVE_MANDATES).
# Width clips from the front, so the first N are the most load-bearing.
_REASON_MANDATES: list[tuple[str, str]] = [
    ("primary", "Provide a thorough, balanced analysis from first principles. Cover the major angles."),
    ("adversarial", "Challenge the primary framing. Find every assumption, logical gap, and place the obvious conclusion is overstated or wrong."),
    ("alternative", "Propose different interpretations and underexplored angles a standard analysis would miss."),
    ("risk", "Identify failure modes and second-order consequences. What happens if the key assumptions are wrong?"),
    ("devils_advocate", "Steelman the strongest case AGAINST the likely conclusion."),
]

_REASON_SYNTHESIS_PROMPT = """You are the synthesis analyst integrating {n} independent perspective analyses of the question below.

QUESTION:
{question}

PERSPECTIVES:
{perspectives}

Identify where perspectives independently converged (high confidence) and where they explicitly conflict (contested). Respond with ONLY this JSON, no other text:
{{"confidence_score": <integer 0-100>, "converged_claims": ["..."], "contested_areas": ["..."], "final_answer": "<integrated answer: lead with convergence, mark contested areas, note gaps>"}}"""


@mcp.tool()
def investigation_reason(
    investigation_id: str,
    question: str,
    perspectives: int = 3,
    ground_threshold: float = 0.59,
    persist: bool = False,
) -> str:
    """Reason over an investigation's findings with grounded, multi-perspective analysis.

    The in-server complement to the deep_think_loci Workflow: it fuses the merge's
    two pieces of tech in-process — the **grounding gate** (embed the question +
    each finding, keep only on-topic findings, dropping cross-target RAG-bleed)
    and **deep_think fan-out** (N adversarial perspectives + a synthesis that
    extracts converged vs contested claims). Runs N+1 LLM calls inline; intended
    as an explicit, user-invoked "reason now" call, not a hot path.

    Requires an LLM endpoint (Ollama by default; anthropic/copilot if a key is
    set). Fail-soft: returns an ``error`` field if the LLM is unreachable rather
    than raising.

    Args:
        investigation_id: Investigation whose findings ground the reasoning.
        question:         The question/problem to reason about.
        perspectives:     Number of adversarial perspectives (1-5). Default 3.
        ground_threshold: Per-finding cosine keep threshold for the grounding
                          gate (deep_think_loci convention: 0.59). Findings below
                          it are dropped as off-topic before any model reasons.
        persist:          If True, store each converged claim as an ``inferred``
                          finding (source=investigation_reason). Default False.

    Returns:
        JSON: {investigation_id, question, perspectives_used, grounded_findings,
               gate_applied, confidence_score, converged_claims, contested_areas,
               final_answer, persisted_finding_ids}.
    """
    from memcheck import llm as _llm
    from memcheck.checks.contradiction_llm import extract_json as _extract_json

    if not (investigation_id and (MEMORY_DIR / investigation_id).exists()):
        return json.dumps({"error": f"Investigation '{investigation_id}' not found."})
    if not (question or "").strip():
        return json.dumps({"error": "question is required."})
    if not _llm.llm_available():
        return json.dumps({"error": "No LLM endpoint available. Set OLLAMA_BASE_URL "
                                    "or a provider key (ANTHROPIC_API_KEY / GITHUB_COPILOT_OAUTH_TOKEN)."})

    n = max(1, min(int(perspectives or 3), len(_REASON_MANDATES)))

    # Load active (non-retracted) findings.
    raw = _read_jsonl(_inv_dir(investigation_id) / "findings.jsonl")
    retracted = _load_retracted_ids(investigation_id)
    findings = [f for f in raw if isinstance(f, dict) and str(f.get("id", "")) not in retracted
                and str(f.get("text", "") or "").strip()]

    # Grounding gate: keep only findings on-topic to the question (fail-open).
    gate_applied = False
    gated = findings
    if findings:
        try:
            vecs = _llm.embed_texts([question] + [str(f["text"]) for f in findings])
        except Exception:
            vecs = []
        if vecs and len(vecs) == len(findings) + 1:
            qv = vecs[0]
            scored = [(_llm.cosine(qv, vecs[i + 1]), f) for i, f in enumerate(findings)]
            kept = [(c, f) for c, f in scored if c >= ground_threshold]
            gated = [f for _, f in sorted(kept, key=lambda x: x[0], reverse=True)[:12]]
            gate_applied = True

    evidence = "\n".join(
        f"- [{f.get('type', f.get('record_type', '?'))}] {str(f['text'])[:300]}"
        for f in gated[:12]
    ) or "(no on-topic findings in this investigation — reason from the question alone)"

    # Fan out N perspectives.
    perspective_outputs: list[dict] = []
    for name, mandate in _REASON_MANDATES[:n]:
        prompt = (
            f"You are the {name} analyst.\n{mandate}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"GROUNDED EVIDENCE (investigation {investigation_id}):\n{evidence}\n\n"
            "Give your analysis in <=200 words. Ground every claim in the evidence; "
            "if the evidence is insufficient, say so rather than inventing facts."
        )
        out = _llm.call_llm(prompt, timeout=90.0)
        if out:
            perspective_outputs.append({"name": name, "analysis": out})

    if not perspective_outputs:
        return json.dumps({"error": "All perspective LLM calls failed (endpoint unreachable or timed out)."})

    # Synthesize.
    persp_text = "\n\n".join(f"=== {p['name'].upper()} ===\n{p['analysis']}" for p in perspective_outputs)
    synth_raw = _llm.call_llm(
        _REASON_SYNTHESIS_PROMPT.format(n=len(perspective_outputs), question=question, perspectives=persp_text),
        json_mode=True, timeout=120.0,
    )
    synth = _extract_json(synth_raw or "") or {}
    converged = [str(c) for c in (synth.get("converged_claims") or []) if str(c).strip()]
    contested = [str(c) for c in (synth.get("contested_areas") or []) if str(c).strip()]
    final_answer = str(synth.get("final_answer") or synth_raw or "").strip()

    # Optionally persist converged claims as inferred findings.
    persisted: list[str] = []
    if persist and converged:
        for claim in converged:
            try:
                res = json.loads(investigation_store(
                    investigation_id=investigation_id,
                    finding_type="inferred",
                    text=f"[reasoned] {claim}",
                    source="investigation_reason",
                    confidence="medium",
                    tags="reasoned,investigation_reason",
                ))
                if res.get("finding_id"):
                    persisted.append(res["finding_id"])
            except Exception as exc:  # noqa: BLE001 — persistence is best-effort
                logger.debug("investigation_reason persist failed: %r", exc)

    return json.dumps({
        "investigation_id": investigation_id,
        "question": question,
        "perspectives_used": [p["name"] for p in perspective_outputs],
        "grounded_findings": len(gated),
        "gate_applied": gate_applied,
        "confidence_score": synth.get("confidence_score"),
        "converged_claims": converged,
        "contested_areas": contested,
        "final_answer": final_answer,
        "persisted_finding_ids": persisted,
    }, indent=2)


# ---------------------------------------------------------------------------
# Conflict management tools
# ---------------------------------------------------------------------------

@mcp.tool()
def conflict_list(investigation_id: str) -> str:
    """
    List all detected conflicts for an investigation.

    Conflicts are recorded automatically by investigation_store when a new
    finding appears to contradict an existing one (e.g. an observed finding
    that contradicts an assumed or gap finding, or opposing negation markers).

    Args:
        investigation_id: Investigation identifier.

    Returns:
        JSON: {"conflicts": [{id, finding_id_a, finding_id_b, detected_at,
               status, resolution}], "count": <int>}
        On error: {"error": "<message>"}
    """
    try:
        manifest = _load_manifest(investigation_id)
        if not manifest:
            return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

        path = _inv_dir(investigation_id) / "conflicts.jsonl"
        raw_conflicts = _read_jsonl(path)

        conflicts = [
            {
                "id": c.get("id"),
                "finding_id_a": c.get("finding_id_a"),
                "finding_id_b": c.get("finding_id_b"),
                "detected_at": c.get("detected_at"),
                "status": c.get("status", "open"),
                "resolution": c.get("resolution"),
            }
            for c in raw_conflicts
        ]

        return json.dumps({"conflicts": conflicts, "count": len(conflicts)}, indent=2)
    except Exception as exc:
        logger.exception("conflict_list failed: %s", exc)
        return json.dumps({"error": str(exc)})


_VALID_VERDICTS = frozenset(["a_wins", "b_wins", "both_valid", "false_positive"])


@mcp.tool()
def conflict_resolve(investigation_id: str, conflict_id: str, verdict: str) -> str:
    """
    Resolve a detected conflict by recording a verdict.

    Args:
        investigation_id: Investigation identifier.
        conflict_id: The conflict id to resolve (from conflict_list or
                     conflict_detected in investigation_store response).
        verdict: One of: a_wins | b_wins | both_valid | false_positive
                 a_wins        — finding_id_a (the newer finding) is correct.
                 b_wins        — finding_id_b (the older finding) is correct.
                 both_valid    — both findings are valid in different contexts.
                 false_positive — the conflict detector was wrong; no real conflict.

    Returns:
        JSON: {"resolved": true, "conflict_id": "...", "verdict": "..."}
        On error: {"error": "<message>"}
    """
    try:
        if verdict not in _VALID_VERDICTS:
            return json.dumps({
                "error": f"verdict must be one of: {', '.join(sorted(_VALID_VERDICTS))}"
            })

        manifest = _load_manifest(investigation_id)
        if not manifest:
            return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

        path = _inv_dir(investigation_id) / "conflicts.jsonl"
        conflicts = _read_jsonl(path)

        updated = False
        new_rows = []
        for c in conflicts:
            if c.get("id") == conflict_id:
                c = dict(c)
                c["status"] = "resolved"
                c["resolution"] = verdict
                updated = True
            new_rows.append(c)

        if not updated:
            return json.dumps({"error": f"Conflict '{conflict_id}' not found in investigation '{investigation_id}'."})

        # Atomic rewrite using a temp file
        import tempfile as _tmpmod
        dir_ = _inv_dir(investigation_id)
        with _tmpmod.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as tf:
            for row in new_rows:
                tf.write(json.dumps(row) + "\n")
            tmp_path = Path(tf.name)
        tmp_path.replace(path)

        return json.dumps({"resolved": True, "conflict_id": conflict_id, "verdict": verdict}, indent=2)
    except Exception as exc:
        logger.exception("conflict_resolve failed: %s", exc)
        return json.dumps({"error": str(exc)})


# Memory hints — polling tool + MCP resource
# ---------------------------------------------------------------------------


def _compute_hints(investigation_id: str, limit: int, since_ts: Optional[str]) -> dict:
    """
    Core logic for memory_hints — shared by the tool and the MCP resource.

    Strategy:
      1. If _session_hints has entries for this investigation, use those
         (fast path — no disk I/O).
      2. Otherwise fall back to reading the tail of findings.jsonl.

    Returns the hints payload dict (not yet JSON-serialised).
    """
    now_ts = int(datetime.now(timezone.utc).timestamp())

    def _recency(created_at_ts) -> float:
        try:
            age_hours = max(0.0, (now_ts - int(created_at_ts)) / 3600.0)
            return round(1.0 / (1.0 + age_hours), 6)
        except Exception:  # noqa: BLE001
            return 0.0

    # --- source: session ring buffer (fast path) ---
    buf = _session_hints.get(investigation_id, [])
    if buf:
        candidates = list(buf)  # copy; most-recent-last already
    else:
        # --- source: JSONL tail (cold path) ---
        findings_path = _inv_dir(investigation_id) / "findings.jsonl"
        raw = _read_jsonl(findings_path)
        candidates = [
            {
                "finding_id": f.get("id", ""),
                "text": f.get("text", ""),
                "source": f.get("source", ""),
                "record_type": f.get("record_type", f.get("type", "observed")),
                "ts": f.get("ts", ""),
                "created_at_ts": f.get("created_at_ts", 0),
            }
            for f in raw
            if isinstance(f, dict)
        ]

    # Apply since_ts filter if requested
    if since_ts:
        candidates = [h for h in candidates if str(h.get("ts", "")) > since_ts]

    # Take the most recent `limit` entries (tail of the list)
    recent = candidates[-limit:] if len(candidates) > limit else candidates

    # Attach recency score
    hints = []
    for h in reversed(recent):  # most-recent first in output
        hints.append({
            "finding_id": h.get("finding_id", ""),
            "text": h.get("text", ""),
            "source": h.get("source", ""),
            "record_type": h.get("record_type", "observed"),
            "recency_score": _recency(h.get("created_at_ts", 0)),
            "ts": h.get("ts", ""),
        })

    return {
        "investigation_id": investigation_id,
        "hints": hints,
        "count": len(hints),
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


@mcp.tool()
def memory_hints(
    investigation_id: str,
    limit: int = 3,
    since_ts: Optional[str] = None,
) -> str:
    """
    Return the most recent findings for an investigation as lightweight hints.

    Suitable for polling after investigation_store to surface "what changed
    recently" without loading the full investigation context.  Assigns a
    recency_score (1.0 / (1 + age_hours)) to each hint so callers can rank
    them by freshness.

    The in-process session ring buffer (populated by investigation_store) is
    preferred for speed; the tool falls back to reading findings.jsonl when
    the ring buffer is empty (e.g. after a server restart).

    Args:
        investigation_id: Investigation identifier.
        limit: Maximum number of hints to return (default 3, max 20).
        since_ts: Optional ISO-8601 timestamp.  When provided, only findings
                  with ts > since_ts are returned.  Use the ``as_of`` field
                  from the previous response as the next ``since_ts`` to poll
                  for changes incrementally.

    Returns:
        JSON: {investigation_id, hints: [{finding_id, text, source,
               record_type, recency_score, ts}], count, as_of}
        On error: {"error": "<message>"}
    """
    try:
        manifest = _load_manifest(investigation_id)
        if not manifest:
            return json.dumps({"error": f"Investigation '{investigation_id}' not found."})
        limit = max(1, min(int(limit or 3), 20))
        payload = _compute_hints(investigation_id, limit, since_ts)
        return json.dumps(payload, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.debug("memory_hints error: %r", exc)
        return json.dumps({"error": str(exc)})


# ---- MCP resource: memory://hints/{investigation_id} ----
# FastMCP supports @mcp.resource() with URI template parameters.
# The resource exposes the same hint payload as the polling tool so MCP
# clients that support resource subscriptions can receive proactive push
# updates when the resource changes.
#
# Note: FastMCP resource subscriptions (real-time push) require the client to
# support MCP resource change notifications over SSE/streamable-http transport.
# The resource is readable over all transports; push is transport-dependent.

@mcp.resource(
    "memory://hints/{investigation_id}",
    name="memory_hints_resource",
    title="Investigation Memory Hints",
    description=(
        "Top recent findings for the given investigation, ranked by recency. "
        "Poll or subscribe to surface what changed since the last context load."
    ),
    mime_type="application/json",
)
def memory_hints_resource(investigation_id: str) -> str:
    """
    MCP resource handler for memory://hints/{investigation_id}.

    Returns the same JSON payload as the memory_hints tool.  FastMCP registers
    this as a URI-template resource so clients can request it as:
        memory://hints/<investigation_id>

    Fail-open: returns a JSON error payload on any exception rather than
    raising, so resource reads never crash the server.
    """
    try:
        manifest = _load_manifest(investigation_id)
        if not manifest:
            return json.dumps({"error": f"Investigation '{investigation_id}' not found."})
        payload = _compute_hints(investigation_id, limit=3, since_ts=None)
        return json.dumps(payload, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.debug("memory_hints_resource error: %r", exc)
        return json.dumps({"error": str(exc)})


# Tool: investigation_export
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tool: investigation_import
# ---------------------------------------------------------------------------


# Memory-as-a-Service: cross-investigation routing
# ---------------------------------------------------------------------------


def _route_filter_by_agent(hits: list[dict], agent_id: str) -> list[dict]:
    """
    Filter memory_route hits down to those authored by `agent_id` or whose
    investigation ACL includes `agent_id`. Fail-open: ACL lookup errors are
    logged and treated as "no match" for that hit.
    """
    filtered = []
    for hit in hits:
        authored_by = hit.get("authored_by", "")
        if authored_by == agent_id:
            filtered.append(hit)
            continue
        # Check if the investigation ACL includes this agent
        inv_id = hit.get("investigation_id", "")
        if inv_id:
            try:
                manifest = _load_manifest(inv_id)
                if manifest:
                    acl = manifest.get("acl", [])
                    if isinstance(acl, list) and agent_id in acl:
                        filtered.append(hit)
                        continue
            except Exception as exc:
                logger.debug("memory_route: ACL check failed for %s: %r", inv_id, exc)
                pass  # graceful skip
    return filtered


def _route_dedup_by_overlap(hits: list[dict]) -> list[dict]:
    """
    Deduplicate memory_route hits by word-overlap (>80% overlap → keep the
    highest-scoring hit). Assumes `hits` is already sorted by score
    (descending), so among an overlapping pair the later one in iteration
    order is always the lower-scoring one to suppress.
    """
    kept = []
    suppressed = set()
    for i, hit_a in enumerate(hits):
        if i in suppressed:
            continue
        words_a = set(str(hit_a.get("text", "")).lower().split())
        for j, hit_b in enumerate(hits):
            if j <= i or j in suppressed:
                continue
            words_b = set(str(hit_b.get("text", "")).lower().split())
            union = words_a | words_b
            if not union:
                continue
            overlap = len(words_a & words_b) / max(len(union), 1)
            if overlap > 0.80:
                # Suppress the lower-scoring one (raw_hits already sorted by score)
                suppressed.add(j)
        kept.append(hit_a)
    return kept


def _route_rows(hits: list[dict]) -> list[dict]:
    """
    Build memory_route response rows from raw Qdrant hits, resolving each
    hit's investigation title via the manifest. Fail-open: manifest lookup
    errors are logged and the row keeps an empty title.
    """
    routed = []
    for hit in hits:
        inv_id = hit.get("investigation_id", "")
        inv_title = ""
        if inv_id:
            try:
                manifest = _load_manifest(inv_id)
                if manifest:
                    inv_title = manifest.get("title", "")
            except Exception as exc:
                logger.debug("memory_route: manifest title lookup failed for %s: %r", inv_id, exc)
                pass

        routed.append({
            "finding_id": hit.get("finding_id") or hit.get("id", ""),
            "investigation_id": inv_id,
            "investigation_title": inv_title,
            "text": hit.get("text", ""),
            "source": hit.get("source", ""),
            "authored_by": hit.get("authored_by", ""),
            "score": hit.get("score", 0.0),
            "tier": hit.get("tier") or hit.get("record_type", "finding"),
        })
    return routed


@mcp.tool()
def memory_route(
    query: str,
    agent_id: Optional[str] = None,
    top_k: int = 10,
    deduplicate: bool = True,
) -> str:
    """
    Agent-mesh-aware search across ALL investigations — no investigation_id filter.

    Searches the main Qdrant collection and returns findings with full provenance,
    optionally filtered by agent_id and deduplicated by content similarity.

    Args:
        query:       Natural language query to search across all investigations.
        agent_id:    If provided, filter to findings authored by this agent or in
                     investigations whose ACL includes this agent_id.
        top_k:       Maximum results to return after deduplication (default 10).
        deduplicate: If True, remove near-duplicate findings (>80% word overlap).
                     Default True.

    Returns JSON with:
        {routed: [{finding_id, investigation_id, investigation_title, text, source,
                   authored_by, score, tier}],
         query, agent_id, total_before_dedup, total_after_dedup, count}
    """
    if not query or not query.strip():
        return json.dumps({
            "error": "query must not be empty",
            "routed": [],
            "query": query,
        })

    try:
        client, _col = _get_qdrant()
        if client is None:
            return json.dumps({
                "error": "memory_route requires Qdrant",
                "routed": [],
            })

        # Step 1+2: Search across the whole collection (no investigation_id filter)
        try:
            raw_hits = _qdrant_search_collection(
                query,
                collection_name=QDRANT_COLLECTION_PREFIX,
                limit=top_k * 3,
            )
        except RuntimeError as exc:
            return json.dumps({
                "error": f"memory_route requires Qdrant: {exc}",
                "routed": [],
            })
        except Exception as exc:
            logger.warning("memory_route: Qdrant search failed: %s", exc)
            return json.dumps({
                "error": f"memory_route search failed: {exc}",
                "routed": [],
            })

        total_before_dedup = len(raw_hits)

        # Step 3: Filter by agent_id if provided
        if agent_id:
            raw_hits = _route_filter_by_agent(raw_hits, agent_id)

        # Step 4: Deduplicate by word-overlap (>80% overlap → keep highest score)
        if deduplicate and len(raw_hits) > 1:
            raw_hits = _route_dedup_by_overlap(raw_hits)

        # Step 5: Trim to top_k
        raw_hits = raw_hits[:top_k]
        total_after_dedup = len(raw_hits)

        # Step 6: Build response with provenance
        routed = _route_rows(raw_hits)

        return json.dumps({
            "routed": routed,
            "query": query,
            "agent_id": agent_id,
            "total_before_dedup": total_before_dedup,
            "total_after_dedup": total_after_dedup,
            "count": len(routed),
        }, indent=2)

    except Exception as exc:
        logger.exception("memory_route: unexpected error: %s", exc)
        return json.dumps({
            "error": f"memory_route failed: {exc}",
            "routed": [],
        })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


# Code<->memory graph tools are defined in graph_tools.py; register them on the
# shared FastMCP instance here (P1 of the Loci self-review split).
import graph_tools  # noqa: E402
graph_tools.register(mcp, _get_ladybug)
# Re-export the graph tool callables so `server.<tool>()` keeps working for
# in-process callers and tests (they use graph_tools' injected _get_ladybug).
from graph_tools import (  # noqa: E402,F401
    code_graph_ingest, code_graph_query, code_memory_relink, code_memory_map,
    symbol_impact, impact_report, finding_code_context, investigation_code_briefing,
    subsystem_report, related_investigations_via_code, dead_code_candidates,
)

# Local-model / embedding passthrough tools live in llm_tools.py (P2a of the split).
import llm_tools  # noqa: E402
llm_tools.register(mcp)
# Re-exported so `server.<tool>()` keeps working for in-process callers and tests
# (e.g. tests/test_ground_tool.py calls server.ground and patches grounding.ground).
from llm_tools import (  # noqa: E402,F401
    llm_local, generate_batch, query_expand, verify_finding, classify_text,
    compress_text, semantic_dedup, semantic_relevance, ground,
)

# Investigation lifecycle tools live in investigation_tools.py (P2b of the split).
# The memory root is injected as a lambda over MEMORY_DIR (same contract as
# inv_store); the collaborators below stayed in server.py and are passed in
# explicitly so investigation_tools never has to import server.
import investigation_tools  # noqa: E402
investigation_tools.register(mcp, lambda: MEMORY_DIR, {
    "_apply_lifecycle": _apply_lifecycle,
    "_compute_self_check": _compute_self_check,
    "_event_log_append": _event_log_append,
    "_qdrant_upsert": _qdrant_upsert,
})
# Re-exported so `server.<tool>()` keeps working for in-process callers and tests.
# NOTE: these tools now resolve their storage helpers in investigation_tools' own
# namespace. A test that needs to intercept one (e.g. force _append_jsonl to fail)
# must patch `investigation_tools.<helper>` — patching `server.<helper>` no longer
# reaches them. No current test does; the two mock.patch.object(server,
# "_append_jsonl") sites target finding_resolve and investigation_verify_all,
# both of which stayed in this module.
from investigation_tools import (  # noqa: E402,F401
    investigation_start, investigation_load, investigation_as_of,
    investigation_note, investigation_reflect, investigation_finding_provenance,
    investigation_list, investigation_share, investigation_unshare,
    investigation_export, investigation_import,
)


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "0:0:0:0:0:0:0:1"})


def _is_loopback(host: str) -> bool:
    return (host or "").strip().lower() in _LOOPBACK_HOSTS


class _BearerAuthMiddleware:
    """Shared-secret gate for the HTTP transports.

    Deliberately not the SDK's token_verifier path: that requires AuthSettings
    with an issuer_url, i.e. the full OAuth resource-server model, and would
    publish /.well-known/oauth-* discovery describing an authorization server
    that does not exist. A single secret shared between an operator and their own
    server is not OAuth, and modelling it as OAuth advertises a fiction.

    /health stays open so liveness probes work without the secret; it returns a
    fixed {"status": "ok"} and discloses nothing.
    """

    __slots__ = ("_app", "_token", "_exempt")

    def __init__(self, app, token: str, exempt_paths=frozenset({"/health"})):
        self._app = app
        self._token = token
        self._exempt = exempt_paths

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("path") in self._exempt:
            await self._app(scope, receive, send)
            return
        raw = dict(scope.get("headers") or {}).get(b"authorization", b"")
        value = raw.decode("latin-1") if isinstance(raw, bytes) else str(raw)
        presented = value[7:] if value[:7].lower() == "bearer " else ""
        # compare_digest on both branches: a plain == would leak the token length
        # and prefix through timing, and an early return on empty would leak
        # whether a token was presented at all.
        if not (presented and hmac.compare_digest(presented, self._token)):
            body = b'{"error":"unauthorized"}'
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode()),
                                    (b"www-authenticate", b"Bearer")]})
            await send({"type": "http.response.body", "body": body})
            return
        await self._app(scope, receive, send)


def main() -> None:
    # Fire-and-forget embed warm-ping so the FIRST real RAG/dedup call doesn't eat the
    # ~9s nomic cold-load. Non-blocking (daemon thread) and fail-open — never blocks or
    # breaks startup.
    try:
        import embed_ops
        embed_ops.warm()
    except Exception as exc:
        logger.debug("main: fail-open swallow: %r", exc)
    transport = os.environ.get("HERMES_MCP_TRANSPORT", "stdio")
    if transport in ("sse", "streamable-http"):
        # FastMCP.run() takes only transport and mount_path; the bind address
        # lives on settings.
        # Loopback by default. This server has no authentication, so a wide bind
        # is a decision, not something you should get by not making one — the
        # same reasoning that fixed the retention default in #204. Every
        # legitimate wide bind already sets this explicitly: docker-compose.yml
        # passes HERMES_MCP_HOST=0.0.0.0 because a container must bind all
        # interfaces to receive published traffic, and publishes on 127.0.0.1.
        # Getting this wrong now fails loudly (cannot connect) instead of
        # silently exposing an unauthenticated tool server.
        mcp.settings.host = os.environ.get("HERMES_MCP_HOST", "127.0.0.1")
        mcp.settings.port = int(os.environ.get("HERMES_MCP_PORT", "8000"))

        # This server exposes the whole tool surface. #206 made the bind loopback
        # by default; a token is what makes a NON-loopback bind defensible, so
        # without one we refuse rather than serve. Loud beats exposed — the same
        # trade as the bind default and the retention default.
        token = os.environ.get("HERMES_MCP_TOKEN", "").strip()
        if not token and not _is_loopback(mcp.settings.host):
            raise SystemExit(
                f"refusing to serve {transport} on {mcp.settings.host} without "
                "HERMES_MCP_TOKEN: this would expose every tool unauthenticated. "
                'Generate one with: python3 -c "import secrets;print(secrets.token_hex(32))" '
                "— or bind 127.0.0.1."
            )

        app = (mcp.streamable_http_app() if transport == "streamable-http"
               else mcp.sse_app())
        if token:
            app = _BearerAuthMiddleware(app, token)
        else:
            logger.warning("HERMES_MCP_TOKEN is not set — serving %s on %s with no "
                           "authentication. Safe only because the bind is loopback.",
                           transport, mcp.settings.host)

        import uvicorn
        uvicorn.run(app, host=mcp.settings.host, port=mcp.settings.port,
                    log_level=str(mcp.settings.log_level).lower())
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
