#!/usr/bin/env python3
"""
loci-mcp — persistent memory and knowledge layer for AI agent investigations.

Provides a manifest-first memory layer for tracking findings, observations,
inferences, assumptions, and tool audit logs across investigation sessions.
Mnemosyne is optional and treated as the primary shared memory substrate when
installed. Qdrant is optional and used as a secondary semantic/hybrid index.
JSONL files remain the durable local storage and final fallback path.

Storage layout:
    $LOCI_MEMORY_DIR/<investigation_id>/
        manifest.json     — structured investigation state
        findings.jsonl    — append-only finding log
        audit.jsonl       — tool call/response audit log

    $LOCI_MEMORY_DIR/../audit/
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
    memory_route_counterfactual_simulate — replay audited memory-route traces under alternate policies
    memory_route_policy_optimize — derive conservative route-policy recommendations from audited outcomes
    memory_self_check            — provenance + contradiction self-check on investigation findings
    memory_retract               — soft-tombstone a hallucinated finding + its derived lineage
    memory_restore               — undo a retraction
    memory_health                — substrate self-check (qdrant / embedders / mirror / integrity)
    code_memory_correlate        — link code-hallucination flags to contaminated investigation findings
    wiring_obligation_scan       — advisory-only scan for implicit obligations that may merit manual declaration
    reflection_loop_seed         — enqueue Copilot artifacts for bounded self-reflection
    reflection_loop_tick         — process small queued batches and store findings
    reflection_loop_status       — inspect reflection queue and aggregate loop stats
"""

from __future__ import annotations

import asyncio  # noqa: F401  (kept on the module namespace; live users import it function-locally)
import hashlib
import hmac
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
import uuid
import weakref
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# memcheck sits alongside this file; importable as __main__ (spawned by path) or under a test's synthetic module name.
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
from compact import compact_context_rows, compact_finding_row, compact_sources  # noqa: E402
from model_json import extract_json_object  # noqa: E402
from slow_neuromod import (
    assert_confidence_policy_invariants,
    assert_consolidation_policy_invariants,
    assert_routing_policy_invariants,
    consolidation_policy,
    confidence_policy,
    load_state,
    observe,
    routing_policy,
)  # noqa: E402
from untrusted_memory import wrap_untrusted_memory_text  # noqa: E402

# Accept the legacy HERMES_* spelling of Loci's own variables.
try:
    from legacy_env import apply as _apply_legacy_env, memory_dir as _legacy_memory_dir
except ImportError:  # not on sys.path in every entrypoint
    try:
        from mcp.legacy_env import apply as _apply_legacy_env, memory_dir as _legacy_memory_dir
    except ImportError:
        _apply_legacy_env = _legacy_memory_dir = None
if _apply_legacy_env is not None:
    _apply_legacy_env()

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
            # Lowercase: Qdrant keyword-index lookups query with .lower().
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

# The .env files above are two of the three places a backend endpoint lives; the third
# is ~/.loci/backends.toml, kept outside the tree so the Qdrant key is not committable.
# Nothing here consulted it, so a server started without an env block (.mcp.json
# registers loci as bare stdio) had no QDRANT_URL, while every RAG gate below reads
# os.environ directly. That fails inverted: loci_health resolves through
# backends.qdrant(), which DOES read backends.toml, so health reports reachable while
# investigation_search returns qdrant_enabled=false with no results — and a caller
# reading empty as "nothing matched" appends a duplicate instead of superseding.
#
# load_env() only fills what is still unset, so it ranks below both .env files. Note the
# mcp/.env call above passes override=True and so beats even an exported variable:
# measured effective order is mcp/.env > exported env > ../.env > backends.toml. Must run
# before qdrant_ops is imported — that module reads os.environ at module level.
import backends  # noqa: E402
backends.load_env()

import logging  # noqa: E402
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("loci-mcp")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MEMORY_DIR = Path(_legacy_memory_dir()) if _legacy_memory_dir else Path(os.environ.get(
    "LOCI_MEMORY_DIR",
    Path.home() / ".loci" / "memory-sessions",
))
# Imported after load_dotenv so its import-time os.environ reads see the same env; re-exported so tests can patch server._get_qdrant.
import qdrant_ops  # noqa: E402,F401
from qdrant_ops import (  # noqa: E402,F401
    QDRANT_COLLECTION_PREFIX, VECTOR_DIM,
    _get_sparse_embedder, _get_cross_encoder, _embed_sparse, _create_payload_indexes,
    _purge_old_records, _get_qdrant, _embed_auth_headers, _embed, _qdrant_upsert,
    _qdrant_degraded_mode, _ce_rerank, _qdrant_similarity_search, _qdrant_search_collection,
    probe_collection, _embed_uncached, _qdrant_client_readonly,
)
from qdrant_ops import RERANK_MAX_CHARS as _RERANK_MAX_CHARS  # noqa: E402
REFLECTION_STATE_DIR = MEMORY_DIR / "_reflection-loop"
REFLECTION_STATE_FILE = REFLECTION_STATE_DIR / "state.json"
REFLECTION_DEFAULT_INVESTIGATION = os.environ.get(
    "LOCI_REFLECTION_INVESTIGATION",
    "copilot-self-reflection-loop",
)
REFLECTION_LOG_TAIL_MIN_FILE_BYTES = 1_000_000
REFLECTION_LOG_TAIL_READ_BYTES = 512_000
REFLECTION_SIGNATURE_OBSERVE_LIMIT = 3
REFLECTION_SIGNATURE_MAP_LIMIT = 500
AGENT_ID = os.environ.get("HERMES_AGENT_ID", "")

# ---------------------------------------------------------------------------
# Investigation storage layer
# ---------------------------------------------------------------------------

import inv_store  # noqa: E402
import caller_identity  # noqa: E402
# Lambda, not the Path value: tests rebind server.MEMORY_DIR to a tmpdir.
inv_store.register(lambda: MEMORY_DIR)
# Re-exported so server.<helper>() keeps resolving for callers and test patches.
from inv_store import (  # noqa: E402,F401
    _now, _inv_dir, _manifest_cache, _load_manifest, _save_manifest,
    _load_manifest_fresh, _atomic_write_text, _append_jsonl, _read_jsonl, _finding_updates_path,
    _load_resolution_overrides, _load_retracted_ids, _make_ref, _tag_finding_ids,
    _summarise_finding, _safe_float, _CONFIDENCE_RANK, _RESOLUTION_STATES,
    _distinctive_entity_set, _CONFIDENCE_TO_NUMERIC, _node_numeric_confidence,
    StoreBusyError, _locked_file,
)
from recall_filter import build_recall_filter  # noqa: E402
from provenance_firewall import (  # noqa: E402
    DETERMINISTIC_DERIVED,
    MODEL_ASSERTED,
    assert_evidence_firewall,
    audit_provenance_fields,
    firewall_candidate_tier,
    normalize_provenance_tier,
    provenance_fields,
)
from replay_fingerprint import (  # noqa: E402
    apply_finding_fingerprints,
    flybrain_audit_fingerprint,
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


def _busy_payload(exc: StoreBusyError, *, investigation_id: Optional[str] = None, **extra) -> dict:
    payload = {
        "error": "busy",
        "detail": str(exc),
        "retryable": True,
    }
    if investigation_id is not None:
        payload["investigation_id"] = investigation_id
    for key, value in extra.items():
        if value is not None:
            payload[key] = value
    return payload


def _busy_result(exc: StoreBusyError, *, investigation_id: Optional[str] = None, **extra) -> str:
    return json.dumps(_busy_payload(exc, investigation_id=investigation_id, **extra))

# ---------------------------------------------------------------------------
# LadybugDB graph store (primary relationship/graph backend) — fail-open like Qdrant.
# ---------------------------------------------------------------------------
_ladybug_store = None                     # LadybugStore singleton once initialized
_ladybug_failed = False                   # PERMANENT-failure latch (ladybug unimportable) — don't retry
_ladybug_last_attempt = 0.0               # monotonic ts of last TRANSIENT init failure
_ladybug_backfilled = False               # one-time findings backfill attempted (deferred past health)
_ladybug_backfill_lock = threading.Lock()  # own lock: _ladybug_lock is non-reentrant and already held on one path
_LADYBUG_RETRY_SECONDS = 30               # backoff before retrying after a transient failure
_ladybug_lock = threading.Lock()


def _get_ladybug(backfill: bool = True):
    """Lazy, fail-open LadybugStore singleton. Returns None if unavailable.

    Distinguishes a genuinely UNRECOVERABLE failure (ladybug is not importable) — which
    latches permanently so we stop retrying — from a TRANSIENT one (another process
    holds Kuzu's single-writer lock, or a transient IO error at open time), which does
    NOT latch: a later call retries after a short backoff so the code graph self-heals
    once the other writer releases the lock. Never raises.

    backfill=False opens the store WITHOUT the one-time findings backfill, which writes
    and therefore takes the writer lease. loci_health passes False so a health check
    stays a read: the backfill still runs, on the first caller that actually wants to
    use the graph.
    """
    global _ladybug_store, _ladybug_failed, _ladybug_last_attempt, _ladybug_backfilled
    if _ladybug_store is not None:
        return _ladybug_backfill_once(backfill)
    if _ladybug_failed:
        return None
    with _ladybug_lock:
        if _ladybug_store is not None:
            return _ladybug_backfill_once(backfill)
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
                # Import OK but open failed = single-writer lock contention: transient, retry after the backoff.
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
    return _ladybug_backfill_once(backfill)


def _ladybug_backfill_once(backfill: bool):
    """Run the one-time findings backfill on first use, then return the store.

    Deferred rather than done at init so a health check can open the store without
    triggering a write. Attempted at most once per process either way — a failed
    attempt is not retried, matching the previous behaviour.

    Claims the attempt under its own lock: this runs on a path where the caller may
    already hold the non-reentrant _ladybug_lock, and two threads racing the flag would
    both take the writer lease."""
    global _ladybug_backfilled
    if not backfill or _ladybug_backfilled:
        return _ladybug_store
    with _ladybug_backfill_lock:
        if _ladybug_backfilled:
            return _ladybug_store
        _ladybug_backfilled = True
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
    between ops, so 'contended' is transient and self-heals. Never raises.

    The store is lazily created on first graph op, so health MUST attempt that
    initialization itself — otherwise the first loci_health of every process reports
    'unavailable' for a perfectly healthy graph, and callers switch the code-graph lane
    off on a false negative. _get_ladybug() is idempotent and fail-open, and still
    honours the permanent latch and the transient-failure backoff, so this cannot turn
    a real failure into a retry storm."""
    try:
        store = _ladybug_store
        if store is None:
            # Predetermined answers cost nothing to settle — do that before opening anything.
            closed = _ladybug_closed_state()
            if closed is not None:
                return closed
            store = _get_ladybug(backfill=False)
        if store is None:
            # The open just failed; it will have latched or armed the backoff, so re-read
            # rather than reporting the attempt as an unknown.
            return _ladybug_closed_state() or "unavailable"
        probe = getattr(store, "readable_probe", None)
        if probe is not None and not probe():
            return "contended"
        return "available"
    except Exception:
        return "unavailable"


def _ladybug_closed_state() -> Optional[str]:
    """'latched' / 'backoff' if the store is known to be unopenable right now, else None."""
    if _ladybug_failed:
        return "latched"
    if _ladybug_last_attempt and (time.monotonic() - _ladybug_last_attempt) < _LADYBUG_RETRY_SECONDS:
        return "backoff"
    return None


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


# The singleton latches and _get_ladybug stay here: tests monkeypatch them on this module.
import ladybug_ops  # noqa: E402
ladybug_ops.register(_get_ladybug, lambda: MEMORY_DIR)
# Re-exported so server.<helper>() keeps resolving for callers and test patches.
from ladybug_ops import (  # noqa: E402,F401
    _ladybug_upsert_investigation, _coerce_ts, _mirror_finding_to_ladybug,
    _get_symbol_index,
    _autolink_finding_to_ladybug, _ladybug_backfill_if_empty, _entity_lookup_ladybug,
)
# _symbol_index_cache/_count are NOT re-exported: `from x import y` binds by value, pinning a stale snapshot.


import mnemo_ops  # noqa: E402,F401
from mnemo_ops import (  # noqa: E402,F401
    _mnemo_bank, _get_mnemo_funcs, _mnemo_remember, _coerce_mnemo_results, _mnemo_recall,
    _mnemo_set_retracted,
)


_MEMORY_DECAY_LAMBDA  = float(os.environ.get("MEMORY_DECAY_LAMBDA", "0.007"))  # Ebbinghaus decay; half-life ~100 days


# Qdrant collection holding code embeddings; empty disables code-chunk correlation.
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
    mode: str = "normal",
) -> dict:
    """
    Assemble a RAG context block from search result dicts.

    Returns a dict with:
      context  - formatted prompt-ready string with [SOURCE N] citations
      sources  - list of {n, id, title, origin, score}
      query, total_chars, truncated, result_count
    """
    def _wrap_untrusted_memory_text(text: str, row: dict) -> str:
        attrs = {k: row.get(k) for k in
                 ("origin", "investigation_id", "finding_id", "memory_id", "id", "source")}
        return wrap_untrusted_memory_text(text, **attrs)

    if mode == "compact":
        compacted = compact_context_rows(
            results,
            budget_chars,
            keep_scores=include_metadata,
            wrap_text=_wrap_untrusted_memory_text,
        )
        return {
            "query": query,
            "context": compacted["context"],
            "sources": compact_sources(results),
            "total_chars": compacted["total_chars"],
            "truncated": compacted["truncated"],
            "result_count": compacted["result_count"],
        }


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
        origin = r.get("origin") or r.get("collection") or "loci_memory"
        mem_id = str(r.get("memory_id") or r.get("finding_id") or r.get("id") or "")
        if text and (
            origin == QDRANT_COLLECTION_PREFIX
            or r.get("investigation_id")
            or r.get("finding_id")
            or r.get("memory_id")
        ):
            text = wrap_untrusted_memory_text(
                text,
                origin=origin,
                investigation_id=r.get("investigation_id"),
                finding_id=r.get("finding_id") or mem_id,
                memory_id=r.get("memory_id"),
                id=r.get("id"),
                source=r.get("source"),
            )

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

    for entity_type, entity_field in entity_field_map.items():
        for val in list(entities.get(entity_type, []))[:2]:
            if not val:
                continue
            try:
                # MatchExcept: cross-investigation baseline only, current investigation excluded.
                hits, _ = client.scroll(
                    col,
                    scroll_filter=Filter(must=[
                        FieldCondition(key=entity_field, match=MatchValue(value=val.lower())),
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

# Explicit plurals (not name + "s"); order is observable — rendered into the entity_type error.
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
# Colons are what distinguish IPv6 from a hex hash.
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
            # Word boundaries: "10.0.0.1" must not match "10.0.0.10", nor a short hash its own hex prefix.
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


# ---------------------------------------------------------------------------
# Finding lifecycle: append-only updates log + staleness (code-ref hashing).
# ---------------------------------------------------------------------------

# None = use the shipped verify.py default; tests set a stub.
_verify_gen_fn = None

# Closed extension set, not "letters after a dot": keeps prose like "e.g." from parsing as a file.
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
# Trailing lookahead stops "file.pyc" from matching on the "py" prefix.
_FILE_REF_RE = re.compile(
    r"\b((?:[\w.\-]+/)*[\w.\-]+\.(?:" + _CODE_REF_EXTS + r"))(?![\w.\-])(?::(\d+))?",
    re.IGNORECASE,
)




def _finding_verifications_path(investigation_id: str) -> Path:
    """Verify-all verdict notes — kept in a SEPARATE, high-churn log so the
    resolution read path (_load_resolution_overrides) never has to scan the
    accumulating per-finding verification records. Write-only for now."""
    return _inv_dir(investigation_id) / "finding_verifications.jsonl"




# Caps how much a rogue code ref can make the server read.
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
    tiers = inv_store._load_provenance_overrides(investigation_id)
    for f in findings:
        if not isinstance(f, dict):
            continue
        fid = str(f.get("id", ""))
        if fid and fid in tiers:
            f.update(inv_store._apply_provenance_override(f, tiers[fid]))
        if fid and fid in overrides:
            f["resolution"] = overrides[fid]
        elif not f.get("resolution"):
            f["resolution"] = "open"
        stale = _finding_is_stale(f)
        if stale is not None:
            f["stale"] = stale


# ---------------------------------------------------------------------------
# Session hints — ring buffer so memory_hints can answer without re-scanning JSONL.
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
_REFLECTION_WARN_LABEL_RE = re.compile(r"\[(?:warn|warning)\]", re.I)
_REFLECTION_ERROR_LABEL_RE = re.compile(r"\[(?:err|error)\]", re.I)
_REFLECTION_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.I,
)
_REFLECTION_HEXISH_ID_RE = re.compile(r"\b(?:[0-9a-f]{4,}-){2,}[0-9a-f]{4,}\b", re.I)
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
    line = re.sub(r"^(?:<ts>\s+)+", "", line)
    line = _REFLECTION_UUID_RE.sub("<uuid>", line)
    line = _REFLECTION_HEXISH_ID_RE.sub("<hexid>", line)
    line = _REFLECTION_HEX_RE.sub("<hex>", line)
    line = _REFLECTION_NUM_RE.sub("<n>", line)
    line = re.sub(r"\s+", " ", line)
    return line[:220]


def _reflection_line_severity(line: str) -> str | None:
    text = str(line or "")
    if not text.strip():
        return None
    if _REFLECTION_WARN_LABEL_RE.search(text) or _REFLECTION_WARN_RE.search(text):
        return "warning"
    if _REFLECTION_ERROR_LABEL_RE.search(text) or _REFLECTION_ERROR_RE.search(text):
        return "error"
    return None


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
    lock_path = _inv_dir(investigation_id) / ".lock"
    with _locked_file(lock_path, "a+", exclusive=True):
        if _load_manifest_fresh(investigation_id):
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
        severity = _reflection_line_severity(line)
        if severity == "error":
            self.error_counts[canon] += 1
        elif severity == "warning":
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
# Domain-generic: true of almost every finding in this corpus, so sharing one says nothing.
_GENERIC_MATCH_TOKENS = {
    "host", "user", "device", "query", "result", "results", "output", "input", "tool",
    "found", "seen", "shows", "reported", "detected", "contacted", "event", "events",
    "record", "records", "row", "rows",
}

# Stopwords, dropped for a different reason: with them in, 68.8% of generic short claims cleared the 0.45 gate.
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
    # High-frequency function words that still tokenize under _TOKEN_RE in the
    # multilingual reports this lane sees; without them, stopword-only claims
    # like "el y de la que en" or "les des avec pour sur dans" can clear the
    # lexical gate on a single shared function word.
    "avec", "como", "con", "dans", "del", "des", "dos", "elle", "elles", "ellos",
    "esta", "este", "esto", "las", "les", "los", "para", "pas", "plus", "por", "pour",
    "que", "qui", "sans", "ses", "son", "sur", "una", "uno", "unos", "vous",
}

_NON_EVIDENCE_TOKENS = _GENERIC_MATCH_TOKENS | _STOPWORD_MATCH_TOKENS
# Cheap PRE-FILTER, never the adjudicator: a filtered top-k search cannot return "no match".
_QDRANT_SUPPORT_MIN_SCORE = 0.55
_QDRANT_PRECHECK_MIN_SCORE = 0.5

# The wider pool is what makes a hit's margin measurable; only _SEMANTIC_REF_LIMIT are surfaced, so the response shape is unchanged.
_SEMANTIC_NEIGHBOURHOOD_K = 25
_SEMANTIC_REF_LIMIT = 5

# Fitted on 600 live probes: false support 88.0% -> 1.7%, true support 100% -> 80.3%; neither half separates the classes alone.
_SEMANTIC_SUPPORT_MIN_OVERLAP = 0.15
_SEMANTIC_SUPPORT_MIN_MARGIN = 0.05
# Near no-op on the headline numbers; kept because it fails CLOSED and a margin against a 3-point pool is undefined.
_SEMANTIC_SUPPORT_MIN_POOL = 8


def _semantic_ref_corroborated(ref: dict) -> bool:
    """Is a dense-similarity hit distinctive enough, and on-topic enough, to count
    as support on its own?

    Two independent conditions, both measured above:
      * ``lexical_overlap`` -- does the hit actually mention what the claim
        mentions, computed against the FULL payload text (not the 260-char
        snippet, which biases long findings).
      * ``score - pool_median`` -- is this hit distinctive inside its own
        neighbourhood, or is the whole pool equally close to the claim?

    A pool smaller than ``_SEMANTIC_SUPPORT_MIN_POOL`` returns False: the margin
    is undefined there and the tool must not assert support it cannot back.
    Callers that only want candidates (dedup advice, contamination scope) can
    reuse this once they have their own measurement.
    """
    try:
        pool_size = int(ref.get("pool_size") or 0)
        if pool_size < _SEMANTIC_SUPPORT_MIN_POOL:
            return False
        overlap = float(ref.get("lexical_overlap") or 0.0)
        if overlap < _SEMANTIC_SUPPORT_MIN_OVERLAP:
            return False
        margin = ref.get("margin")
        if margin is None:
            margin = float(ref.get("score") or 0.0) - float(ref.get("pool_median") or 0.0)
        return float(margin) >= _SEMANTIC_SUPPORT_MIN_MARGIN
    except (TypeError, ValueError):
        return False


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


_AUDIT_LANE_STALE_AFTER = timedelta(days=7)


def _parse_utc_ts(value) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _audit_lane_state(
    investigation_id: str,
    findings: list[dict],
    scoped_audit: list[dict],
    global_recent_audit: list[dict],
) -> tuple[dict, list[dict], list[dict]]:
    scoped_entries = [e for e in scoped_audit if isinstance(e, dict)]
    global_entries = [
        e for e in global_recent_audit
        if isinstance(e, dict) and str(e.get("investigation_id") or "") == investigation_id
    ]
    all_entries = scoped_entries + global_entries
    audit_ts = [ts for ts in (_parse_utc_ts(e.get("ts")) for e in all_entries) if ts is not None]
    finding_ts = [
        ts for ts in (_parse_utc_ts(f.get("ts")) for f in findings if isinstance(f, dict))
        if ts is not None
    ]
    latest_audit = max(audit_ts, default=None)
    latest_finding = max(finding_ts, default=None)

    base = {
        "status": "empty",
        "usable": False,
        "reason": "no_audit_entries",
        "threshold_seconds": int(_AUDIT_LANE_STALE_AFTER.total_seconds()),
        "scoped_receipts": len(scoped_entries),
        "global_receipts": len(global_entries),
        "latest_audit_ts": latest_audit.isoformat() if latest_audit else None,
        "latest_finding_ts": latest_finding.isoformat() if latest_finding else None,
    }
    if not all_entries:
        return base, [], []
    if latest_audit is None:
        return {
            **base,
            "status": "stale",
            "reason": "audit_timestamps_unparseable",
        }, [], []
    if latest_finding is not None and latest_finding - latest_audit > _AUDIT_LANE_STALE_AFTER:
        return {
            **base,
            "status": "stale",
            "reason": "audit_older_than_investigation",
            "lag_seconds": int((latest_finding - latest_audit).total_seconds()),
        }, [], []
    return {
        **base,
        "status": "fresh",
        "usable": True,
        "reason": None,
    }, scoped_entries, global_entries


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
) -> tuple[list[dict], dict]:
    # Backfilled provenance tiers (provenance_updates.jsonl) overlay untagged rows.
    findings = inv_store._fold_provenance_overrides(
        _read_jsonl(_inv_dir(investigation_id) / "findings.jsonl"), investigation_id)
    scoped_audit = _read_jsonl(_inv_dir(investigation_id) / "audit.jsonl")
    global_recent_audit = _collect_recent_global_audit(limit=150, days=2)
    audit_lane, scoped_audit, global_recent_audit = _audit_lane_state(
        investigation_id, findings, scoped_audit, global_recent_audit
    )
    # A retracted finding is not evidence for anything (pre_answer_check support,
    # the provenance firewall's linked evidence, verify_all, promotion).
    _rf = build_recall_filter(MEMORY_DIR, [investigation_id], with_texts=False)
    _retracted = _rf.all_retracted
    excluded_retracted = 0

    evidence: list[dict] = []
    for idx, finding in enumerate(findings):
        if str(finding.get("id") or "") in _retracted:
            excluded_retracted += 1
            continue
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
            **provenance_fields(finding),
        })

    if audit_lane["usable"]:
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
                # Caller-written receipt: tool_verified only for a non-model tool.
                **audit_provenance_fields(entry.get("tool")),
            })

        for idx, entry in enumerate(global_recent_audit):
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
                **audit_provenance_fields(entry.get("tool")),
            })
    return evidence, {"audit": audit_lane,
                      "retraction": {"excluded_retracted": excluded_retracted, **_rf.status()}}


def _search_qdrant_claim_evidence(
    claim: str,
    investigation_id: str,
    limit: int = _SEMANTIC_REF_LIMIT,
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
        surfaced = max(1, min(limit, _SEMANTIC_NEIGHBOURHOOD_K))
        result = client.query_points(
            collection_name=col,
            query=vector,
            using="dense",
            query_filter=search_filter,
            limit=max(surfaced, _SEMANTIC_NEIGHBOURHOOD_K),
            with_payload=True,
        )
        points = list(result.points)
        pool_scores = [float(point.score) for point in points]
        pool_median = _median(pool_scores)
        claim_tokens = tokenize(claim)
        # memory_retract keeps the Qdrant point: a retracted finding is not evidence.
        _retracted = build_recall_filter(MEMORY_DIR, [investigation_id], with_texts=False).all_retracted
        matches = []
        for point in points[:surfaced]:
            payload = point.payload or {}
            if str(payload.get("id") or point.id) in _retracted:
                status["excluded_retracted"] = status.get("excluded_retracted", 0) + 1
                continue
            text = str(payload.get("text") or payload.get("output") or "")
            score = round(float(point.score), 4)
            matches.append({
                "evidence_id": str(payload.get("id") or point.id),
                "record_type": str(payload.get("record_type") or payload.get("type") or "unknown"),
                "source": str(payload.get("source") or payload.get("tool") or ""),
                "ts": payload.get("ts"),
                "origin": "qdrant",
                "score": score,
                # Computed here while the FULL payload text is in hand; the ref carries only a 260-char snippet.
                "text": text,
                "lexical_overlap": round(_lexical_match_score(claim_tokens, tokenize(text)), 4),
                "pool_median": pool_median,
                "pool_size": len(pool_scores),
                "margin": round(score - pool_median, 4),
                "snippet": text.replace("\n", " ").strip()[:260],
                **provenance_fields(payload),
            })
        return matches, status
    except Exception as exc:
        status["error"] = str(exc)
        return [], status


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[mid], 4)
    return round((ordered[mid - 1] + ordered[mid]) / 2.0, 4)


def _lexical_match_score(claim_tokens: set[str], evidence_tokens: set[str]) -> float:
    claim_content = {
        token for token in (claim_tokens or set()) if token not in _NON_EVIDENCE_TOKENS
    }
    evidence_content = {
        token for token in (evidence_tokens or set()) if token not in _NON_EVIDENCE_TOKENS
    }
    if not claim_content or not evidence_content:
        return 0.0
    overlap = claim_content.intersection(evidence_content)
    return len(overlap) / max(1, len(claim_content))




def _compute_aggregate_confidence(
    finding_id: str,
    findings_by_id: dict,
    max_depth: int = 5,
) -> float | None:
    """Walk the derived_from chain for a finding and return the product of
    numeric_confidence values along the chain (up to max_depth nodes).

    A finding with no numeric_confidence is resolved from its own confidence label
    (_node_numeric_confidence), not scored 1.0 — absence is not perfect certainty,
    and in a product a 1.0 node silently stops constraining the aggregate.
    Returns 1.0 if the finding_id is not found or the chain is empty.
    Returns None if the walk itself failed: for the same reason a node is not
    worth 1.0, a crash is not worth 1.0 either — it is the top of the scale AND
    the identity element of the product, so it would report perfect certainty
    over a chain that was never read. investigation_finding_provenance already
    returns None here (investigation_tools.py).
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
            nc = _node_numeric_confidence(node)
            try:
                product *= float(nc)
            except (TypeError, ValueError) as exc:
                logger.debug("_compute_aggregate_confidence: fail-open swallow: %r", exc)
            parents = node.get("derived_from") or []
            current_id = str(parents[0]) if parents else None
            depth += 1
        return round(product, 6)
    except Exception as exc:
        logger.debug("_compute_aggregate_confidence: chain walk failed: %r", exc)
        return None


# ---------------------------------------------------------------------------
# Memory self-check helpers (advisory-only; never mutate/delete findings)
# ---------------------------------------------------------------------------



def _compute_self_check(investigation_id: str, llm_verify: bool = False) -> dict:
    """Run provenance + contradiction inline over an investigation's JSONL.

    Pure over the JSONL — does NOT require qdrant. Returns the raw verdict lists
    keyed by check, plus a derived ``hallucination_candidates`` list. Already-
    retracted findings are excluded so a cleaned-up hallucination stops being
    re-surfaced. Fail-open: a check error degrades to an empty list, and
    ``check_status`` records which checks ran ok, failed, or fell back.

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

    # Per-check outcome: a crashed check yields [] exactly like a clean store, so the
    # caller must be told which lists are real results and which are failures.
    check_status: dict = {}

    try:
        unsupported = run_provenance(
            findings,
            audit_entries,
            tokenizer=tokenize,
            lexical_score=_lexical_match_score,
        )
        check_status["provenance"] = "ok"
    except Exception as exc:  # fail-open — advisory check must never break the caller
        logger.warning("provenance check failed, degrading to none: %r", exc)
        unsupported = []
        check_status["provenance"] = f"failed: {exc!r}"

    try:
        contradictions = run_contradiction(
            findings,
            negation_re=_NEGATION_RE,
            tokenizer=tokenize,
        )
        check_status["contradiction"] = "ok"
    except Exception as exc:  # fail-open
        logger.warning("contradiction check failed, degrading to none: %r", exc)
        contradictions = []
        check_status["contradiction"] = f"failed: {exc!r}"

    if llm_verify:
        try:
            from memcheck.checks.contradiction_llm import verify_and_merge

            contradictions = verify_and_merge(findings, contradictions)
            check_status["llm_verify"] = "applied"
        except Exception as exc:  # fail-open — keep lexical verdicts on any error
            logger.warning("llm contradiction verify failed, keeping lexical: %r", exc)
            check_status["llm_verify"] = f"fallback_lexical: {exc!r}"

    try:
        candidates = _hallucination_candidates(
            findings, audit_entries, unsupported, contradictions
        )
        # Candidates need both inputs; a failed input makes an empty list meaningless.
        inputs_ok = check_status["provenance"] == "ok" and check_status["contradiction"] == "ok"
        check_status["hallucination_candidates"] = "ok" if inputs_ok else "incomplete"
    except Exception as exc:  # fail-open
        logger.warning("hallucination-candidate surfacing failed, degrading: %r", exc)
        candidates = []
        check_status["hallucination_candidates"] = f"failed: {exc!r}"

    return {
        "unsupported_observed": unsupported,
        "contradictions": contradictions,
        "hallucination_candidates": candidates,
        "audit_lane": audit_lane,
        "check_status": check_status,
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

    # A blanket unsupported verdict means no audit lane exists (1 of 140 investigations has one), so it must not gate this pass.
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

    # Receipted = an observed finding the provenance check did not flag. Inferred/gap
    # rows are never checked for receipts, so they are not receipted counter-evidence.
    receipted_ids = _observed_ids - unsupported_ids
    findings_by_id = {str(f.get("id", "")): f for f in findings}

    candidates: list[dict] = []
    seen: set[str] = set()
    for v in contradictions:
        refs = list(v.refs or [])
        if len(refs) != 2:
            continue
        a, b = str(refs[0]), str(refs[1])
        pairs = [(a, b), (b, a)]
        for unsup, other in pairs:
            if unsup not in unsupported_ids:
                continue
            counter_receipted = other in receipted_ids
            if not counter_receipted and not _blanket:
                continue  # no receipted counter
            if other not in findings_by_id:
                continue
            if unsup in seen:
                continue
            seen.add(unsup)
            f = findings_by_id.get(unsup, {})
            candidates.append({
                "finding_id": unsup,
                "contradicted_by": other,
                "counter_receipted": counter_receipted,
                "excerpt": redact_excerpt(str(f.get("text", "") or "")),
                "rationale": (
                    "unsupported observed finding (no receipt) contradicted by a "
                    "receipted finding — likely a self-generated hallucination"
                    if counter_receipted else
                    "unsupported observed finding contradicted by another finding; no "
                    "audit receipt backs either side, so this cannot tell which is wrong"
                ),
                "hint": (
                    "review and run memory_retract(target=<finding_id>) to clean the lineage"
                    if counter_receipted else
                    "verify both findings before retracting either; neither has a receipt"
                ),
            })
    return candidates


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("loci")

# Sync tools run on a worker thread, not inline on the event loop: one slow
# backend call (Ollama, Qdrant) used to freeze /health and every client's
# handshake. Must precede the first @mcp.tool() below.
import tool_offload  # noqa: E402

tool_offload.install(mcp)


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


def _judge_conflict_pair(new_finding: dict, neighbor_finding: dict, *, gen_fn=None) -> dict:
    """LLM-adjudicate whether a high-cosine neighbour directly contradicts the new finding.

    Qdrant's cosine gate is the cheap candidate generator: two findings in the same
    investigation are near each other, so they may concern the same subject. The
    heuristics below only recognize a few structural cases and cannot tell "same topic"
    from "same fact with opposite polarity". This asks the local reasoning model ONLY
    after the cheap gate has produced a candidate pair, and it stays fail-open:
    any import/backend/model issue returns ``verdict=None`` so the caller preserves the
    prior heuristic-only behaviour exactly.
    """
    try:
        from conflict_verify import judge_conflict

        return judge_conflict(
            str((new_finding or {}).get("text", "") or ""),
            str((neighbor_finding or {}).get("text", "") or ""),
            type_a=str((new_finding or {}).get("record_type")
                       or (new_finding or {}).get("type", "") or ""),
            type_b=str((neighbor_finding or {}).get("record_type")
                       or (neighbor_finding or {}).get("type", "") or ""),
            gen_fn=gen_fn,
        )
    except Exception as exc:
        logger.debug("_judge_conflict_pair: fail-open on exception: %r", exc)
        return {"verdict": None, "reason": "", "ok": False, "error": str(exc)[:200]}


def _detect_conflicts(investigation_id: str, new_finding: dict) -> list[dict]:
    """
    Search Qdrant for near-neighbors of new_finding (same investigation, cosine
    > 0.82, excluding the new finding itself), then add an LLM contradiction judge
    on top of the existing heuristics.

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
            if neighbor_id == new_id:
                continue

            neighbor_type = payload.get("record_type") or payload.get("type", "")
            neighbor_text = str(payload.get("text", ""))
            neighbor_neg = _has_negation(neighbor_text)

            heuristic_conflict = False

            # Heuristic 1: gap now filled by an observed finding
            if neighbor_type == "gap" and new_type == "observed":
                heuristic_conflict = True

            # Heuristic 2: assumption overridden by a non-assumed finding
            elif neighbor_type == "assumed" and new_type != "assumed":
                heuristic_conflict = True

            # Off by default: bare token presence, not polarity — it manufactures conflicts from incidental wording.
            elif _CONFLICT_NEGATION_HEURISTIC and new_neg != neighbor_neg:
                heuristic_conflict = True

            llm = _judge_conflict_pair(new_finding, payload)
            llm_contradict = llm.get("verdict") == "contradict"

            if heuristic_conflict or llm_contradict:
                conflicts.append({
                    "neighbor_id": neighbor_id,
                    "neighbor_type": neighbor_type,
                    "score": round(float(hit.score), 4),
                    "heuristic_conflict": heuristic_conflict,
                    "llm_verdict": llm.get("verdict"),
                    "llm_reason": llm.get("reason", ""),
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
# investigation_store internals — error strings are verbatim, part of the tool's response contract.
# --------------------------------------------------------------------------- #
_STORE_FINDING_TYPES = {"observed", "inferred", "assumed", "gap", "procedure"}
_STORE_CONFIDENCES = {"high", "medium", "low"}
_STORE_TIERS = {"hot", "warm", "cold"}
_FLYBRAIN_CLAIM_SCOPE_KEYS = (
    "dataset",
    "dataset_version",
    "sex",
    "life_stage",
    "annotation_completeness",
    "circuit_class",
    "experience_window",
)
_FLYBRAIN_CROSS_SEX_TOKENS = (
    "cross-sex",
    "across sex",
    "both",
    "male+female",
    "male/female",
    "all sexes",
)
_FLYBRAIN_CROSS_STAGE_TOKENS = (
    "cross-stage",
    "across stage",
    "all stages",
    "larval+adult",
    "developmental",
)
_FLYBRAIN_DEVELOPMENT_PLASTICITY_TOKENS = (
    "development",
    "develop",
    "larva",
    "pupa",
    "plastic",
    "learning",
    "trained",
    "experience-dependent",
)


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


def _normalize_finding_metadata(metadata: Any) -> Optional[dict]:
    """Best-effort normalize optional finding metadata to a dict; fail-open."""
    if metadata in (None, "", {}):
        return None
    if isinstance(metadata, dict):
        return metadata or None
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
            return parsed if isinstance(parsed, dict) and parsed else None
        except Exception:
            return {"value": metadata}
    return {"value": metadata}


def _is_valid_flybrain_scope_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return False
        return True
    return False


def _contains_any_scope_token(value: Any, tokens: tuple[str, ...]) -> bool:
    text = str(value or "").strip().lower()
    return any(token in text for token in tokens)


def _validate_flybrain_stability_ceiling(metadata: dict, normalized_scope: dict) -> Optional[str]:
    flybrain_prov = metadata.get("flybrain_provenance")
    support = {}
    if isinstance(flybrain_prov, dict):
        gs = flybrain_prov.get("generalization_support")
        if isinstance(gs, dict):
            support = gs
    cross_sex = _contains_any_scope_token(normalized_scope.get("sex"), _FLYBRAIN_CROSS_SEX_TOKENS)
    cross_stage = _contains_any_scope_token(normalized_scope.get("life_stage"), _FLYBRAIN_CROSS_STAGE_TOKENS)
    edge_case = _contains_any_scope_token(normalized_scope.get("life_stage"), _FLYBRAIN_DEVELOPMENT_PLASTICITY_TOKENS) or _contains_any_scope_token(
        normalized_scope.get("experience_window"), _FLYBRAIN_DEVELOPMENT_PLASTICITY_TOKENS
    )
    if edge_case and cross_sex and support.get("cross_sex_validated") is not True:
        return (
            "flybrain claim_scope cross-sex generalization in development/plasticity scope "
            "requires flybrain_provenance.generalization_support.cross_sex_validated=true"
        )
    if edge_case and cross_stage and support.get("cross_stage_validated") is not True:
        return (
            "flybrain claim_scope cross-stage generalization in development/plasticity scope "
            "requires flybrain_provenance.generalization_support.cross_stage_validated=true"
        )
    return None


def _flybrain_scope_hash(scope: Optional[dict]) -> Optional[str]:
    """Return a deterministic hash for a FlyBrain scope payload or None."""
    if not isinstance(scope, dict):
        return None
    stable = json.dumps(scope, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _emit_flybrain_claim_audit(investigation_id: Optional[str], *, decision: str, claim_scope: Optional[dict], reason: Optional[str], metadata: Optional[dict], tool_name: str = "flybrain_claim_scope_validator", tier: Optional[str] = None) -> Optional[dict]:
    """Emit a single, deterministic audit record for FlyBrain claim scope decisions."""
    if not isinstance(claim_scope, dict) and not isinstance(metadata, dict):
        return None
    scope = dict(claim_scope) if isinstance(claim_scope, dict) else {}
    prov = metadata.get("flybrain_provenance") if isinstance(metadata, dict) and isinstance(metadata.get("flybrain_provenance"), dict) else {}
    if isinstance(prov, dict):
        nested_scope = prov.get("claim_scope")
        if isinstance(nested_scope, dict) and not scope:
            scope = dict(nested_scope)
        prov = {k: v for k, v in prov.items() if k != "audit_hook" and k != "provenance"}
    if not scope and not prov:
        return None
    payload = {
        "audit_type": "flybrain_claim_scope_decision",
        "decision": decision,
        "tool_name": tool_name,
        "reason": reason,
        "tier": tier,
        "claim_scope": scope,
        "scope_hash": _flybrain_scope_hash(scope),
        "provenance": prov,
        "ts": _now(),
    }
    try:
        audit_log(
            tool_name=tool_name,
            inputs_json=json.dumps({"audit_type": "flybrain_claim_scope_decision", "decision": decision, "claim_scope": scope}, sort_keys=True),
            output=json.dumps(payload, sort_keys=True),
            investigation_id=investigation_id,
            embedding_text=(
                f"FlyBrain claim decision={decision} "
                f"reason={reason or 'scope_validated'} "
                f"scope_hash={payload['scope_hash']}"
            ),
        )
    except Exception as exc:  # fail-open: audit logging is advisory
        logger.debug("flybrain claim audit hook failed (fail-open): %r", exc)
    return payload


def _normalize_and_validate_flybrain_claim_scope(metadata: Optional[dict]) -> tuple[Optional[dict], Optional[str]]:
    """Validate flybrain claim scope metadata fail-closed; keep non-flybrain fail-open."""
    if not isinstance(metadata, dict) or not metadata:
        return metadata, None

    out = dict(metadata)
    flybrain_prov = out.get("flybrain_provenance")
    claim_scope = out.get("claim_scope")

    # Accept either top-level metadata.claim_scope or the nested
    # metadata.flybrain_provenance.claim_scope form, but normalize to both.
    if claim_scope is None and isinstance(flybrain_prov, dict):
        nested_scope = flybrain_prov.get("claim_scope")
        if nested_scope is not None:
            claim_scope = nested_scope

    should_validate = ("claim_scope" in out) or ("flybrain_provenance" in out)
    if not should_validate:
        return out, None

    if not isinstance(claim_scope, dict):
        error = (
            "flybrain claim_scope must be a JSON object containing keys: "
            + ", ".join(_FLYBRAIN_CLAIM_SCOPE_KEYS)
        )
        _emit_flybrain_claim_audit(None, decision="rejected", claim_scope=None, reason=error, metadata=out, tool_name="flybrain_claim_scope_validator", tier="T0")
        return None, error

    missing = [k for k in _FLYBRAIN_CLAIM_SCOPE_KEYS if k not in claim_scope]
    if missing:
        error = (
            "flybrain claim_scope is missing required key(s): "
            + ", ".join(missing)
        )
        _emit_flybrain_claim_audit(None, decision="rejected", claim_scope=claim_scope, reason=error, metadata=out, tool_name="flybrain_claim_scope_validator", tier="T0")
        return None, error

    invalid = [k for k in _FLYBRAIN_CLAIM_SCOPE_KEYS if not _is_valid_flybrain_scope_value(claim_scope.get(k))]
    if invalid:
        error = (
            "flybrain claim_scope has invalid value(s) for key(s): "
            + ", ".join(invalid)
            + ". Values must be non-empty strings or scalar numbers/booleans."
        )
        _emit_flybrain_claim_audit(None, decision="rejected", claim_scope=claim_scope, reason=error, metadata=out, tool_name="flybrain_claim_scope_validator", tier="T0")
        return None, error

    normalized_scope = {
        k: (claim_scope[k].strip() if isinstance(claim_scope[k], str) else claim_scope[k])
        for k in _FLYBRAIN_CLAIM_SCOPE_KEYS
    }
    out["claim_scope"] = normalized_scope

    if isinstance(flybrain_prov, dict):
        flybrain_out = dict(flybrain_prov)
        flybrain_out["claim_scope"] = dict(normalized_scope)
        out["flybrain_provenance"] = flybrain_out

    stability_ceiling_error = _validate_flybrain_stability_ceiling(out, normalized_scope)
    if stability_ceiling_error:
        audit = _emit_flybrain_claim_audit(
            None,
            decision="rejected",
            claim_scope=normalized_scope,
            reason=stability_ceiling_error,
            metadata=out,
            tool_name="flybrain_claim_scope_validator",
            tier="T0",
        )
        if isinstance(out.get("flybrain_provenance"), dict):
            out["flybrain_provenance"]["decision"] = "rejected"
            out["flybrain_provenance"]["audit_hook"] = audit or {"decision": "rejected", "reason": stability_ceiling_error}
        return None, stability_ceiling_error

    audit = _emit_flybrain_claim_audit(
        None,
        decision="accepted",
        claim_scope=normalized_scope,
        reason="scope_validated",
        metadata=out,
        tool_name="flybrain_claim_scope_validator",
        tier="T1",
    )
    if isinstance(out.get("flybrain_provenance"), dict):
        out["flybrain_provenance"]["decision"] = "accepted"
        out["flybrain_provenance"]["audit_hook"] = audit or {"decision": "accepted", "reason": "scope_validated"}
    return out, None


def _docs_ingest_summary_from_markdown(raw_text: str, *, title: str | None = None, max_chars: int = 600) -> str:
    """Return a compact summary for a markdown doc using its headings and opening paragraphs."""
    text = (raw_text or "").strip()
    if not text:
        return "Empty markdown document."
    headings = re.findall(r"^#{1,6}\s+(.+)$", text, flags=re.M)
    para_candidates = []
    for block in re.split(r"\n\s*\n", text):
        compact = re.sub(r"\s+", " ", block).strip()
        if len(compact) > 40:
            para_candidates.append(compact)
    lead = para_candidates[0] if para_candidates else ""
    if headings:
        label = headings[0].strip()
        summary = f"{label}. {lead}" if lead else label
        if len(summary) > max_chars:
            summary = summary[:max_chars - 1].rsplit(" ", 1)[0] + "…"
        return summary
    if lead:
        summary = lead
        if len(summary) > max_chars:
            summary = summary[:max_chars - 1].rsplit(" ", 1)[0] + "…"
        return summary
    base = title or "Documentation"
    return f"{base}: no useful body text was found."


def _docs_ingest_src_hash(raw_text: str) -> str:
    """Return the deterministic content hash used to decide whether a doc changed."""
    return hashlib.sha256((raw_text or "").encode("utf-8")).hexdigest()


def _docs_ingest_state_for_path(investigation_id: str, document_path: Path, content_hash: str) -> str:
    """Return 'new', 'changed', or 'unchanged' based on prior index metadata for the same path."""
    try:
        loaded = json.loads(investigation_load(investigation_id=investigation_id, last_n_findings=2000))
    except Exception:
        return "new"

    for finding in loaded.get("recent_findings", []) or []:
        metadata = finding.get("metadata") if isinstance(finding, dict) else None
        if not isinstance(metadata, dict):
            continue
        if metadata.get("source_path") != str(document_path):
            continue
        previous_hash = metadata.get("source_sha256") or metadata.get("provenance", {}).get("document_sha256")
        if previous_hash == content_hash:
            return "unchanged"
        return "changed"
    return "new"


_DOCS_INGEST_MAX_FILES = 500
_DOCS_INGEST_EXTS = frozenset({".md", ".markdown", ".txt"})


def _docs_ingest_roots() -> list[Path]:
    """Roots docs_ingest_indexer may read under: ``LOCI_DOCS_ROOTS``
    (``os.pathsep``-separated), else the code root. Re-read each call."""
    raw = os.environ.get("LOCI_DOCS_ROOTS", "")
    roots = []
    for part in raw.split(os.pathsep):
        if part.strip():
            try:
                roots.append(Path(part.strip()).expanduser().resolve())
            except Exception:  # noqa: BLE001
                continue
    return roots or [_code_root()]


def _docs_ingest_confined(p: Path, roots: list[Path]) -> Optional[Path]:
    """``p`` resolved (symlinks included) if it stays under a docs root, else None."""
    try:
        resolved = p.resolve(strict=True)
    except Exception:  # noqa: BLE001
        return None
    return resolved if any(resolved.is_relative_to(r) for r in roots) else None


def _docs_ingest_all_targets(document_path: str) -> list[Path]:
    """Resolve a file or directory to markdown/text targets; fail-open to [] if the path is invalid.

    Only paths under a docs root (``_docs_ingest_roots``) are read, and each file
    is checked after symlink resolution: a ``notes.md`` link to a secret outside
    the roots, or to a non-doc file, is skipped. Every match, before the caller
    applies ``_DOCS_INGEST_MAX_FILES``."""
    roots = _docs_ingest_roots()
    p = Path(document_path).expanduser()
    if not p.is_absolute():
        p = _code_root() / p
    if _docs_ingest_confined(p, roots) is None:
        return []

    def _ok(x: Path) -> bool:
        target = _docs_ingest_confined(x, roots)
        return (
            target is not None and target.is_file()
            and x.suffix.lower() in _DOCS_INGEST_EXTS
            and target.suffix.lower() in _DOCS_INGEST_EXTS
        )

    if p.is_file():
        return [p] if _ok(p) else []
    if p.is_dir():
        return sorted(
            {x for x in p.rglob("*") if x.is_file() and _ok(x)},
            key=lambda item: str(item),
        )
    return []


@mcp.tool()
def docs_ingest_indexer(
    document_path: str,
    investigation_id: str = "loci-docs-index",
    summary_only: bool = False,
    source: str = "docs_ingest_indexer",
    confidence: str = "medium",
) -> str:
    """Index markdown or text docs into the standard Loci investigation store with provenance.

    Only documents under the docs roots (``LOCI_DOCS_ROOTS``, ``os.pathsep``-separated;
    default: the code root) are read, symlink targets included. Indexed findings are
    tagged ``model_asserted``: the tool verifies which bytes it read (sha256), not
    what the document claims, so its content is not independent evidence.
    """
    all_targets = _docs_ingest_all_targets(document_path)
    targets = all_targets[:_DOCS_INGEST_MAX_FILES]
    if not targets:
        return json.dumps({
            "error": (
                f"No readable markdown/text documents found under: {document_path} "
                f"(only paths under the docs roots {[str(r) for r in _docs_ingest_roots()]} are read; "
                "set LOCI_DOCS_ROOTS to widen them)"
            ),
            "stored": 0,
            "investigation_id": investigation_id,
        })

    _ensure_investigation_exists(
        investigation_id,
        title=f"Docs index for {Path(document_path).name or 'markdown'}",
        context="Markdown docs are indexed here through the standard Loci investigation-store provenance path.",
    )

    records: list[dict] = []
    for doc_path in targets:
        text = doc_path.read_text(encoding="utf-8", errors="replace")
        raw = text.strip()
        doc_title = doc_path.stem.replace("-", " ").replace("_", " ").strip() or doc_path.name
        summary = _docs_ingest_summary_from_markdown(raw, title=doc_title)
        content_hash = _docs_ingest_src_hash(raw)
        change_state = _docs_ingest_state_for_path(investigation_id, doc_path, content_hash)
        metadata = {
            "title": doc_title,
            "source_path": str(doc_path),
            "doc_kind": "markdown",
            "source_sha256": content_hash,
            "source_mtime_ns": getattr(doc_path.stat(), "st_mtime_ns", None),
            "content_length": len(raw),
            "change_state": change_state,
            "doc_summary": summary,
            "provenance": {
                "tool_name": "docs_ingest_indexer",
                "tool_variant": "docs_ingest_indexer",
                "source_path": str(doc_path),
                "document_sha256": content_hash,
                "content_length": len(raw),
                "ingested_at": _now(),
                # The hash proves which bytes were read, not that their claims hold.
                "content_verified": False,
            },
            # Document text is an unverified claim of unknown authorship, so it takes
            # the non-independent tier rather than tool_verified.
            "evidence_provenance_tier": MODEL_ASSERTED,
        }
        if not summary_only:
            metadata["content_excerpt"] = raw[:1200]

        if change_state == "unchanged":
            records.append({
                "path": str(doc_path),
                "stored": False,
                "changed": False,
                "change_state": "unchanged",
                "finding_id": None,
                "summary": summary,
            })
            continue

        finding_text = f"{doc_title}: {summary}"
        store_result = json.loads(investigation_store(
            investigation_id=investigation_id,
            finding_type="observed",
            text=finding_text,
            source=source,
            confidence=confidence,
            tags=["docs", "markdown", "loci-index"],
            metadata=metadata,
            evidence_provenance_tier=MODEL_ASSERTED,
        ))
        records.append({
            "path": str(doc_path),
            "stored": bool(store_result.get("stored")),
            "changed": change_state in {"new", "changed"},
            "change_state": change_state,
            "finding_id": store_result.get("finding_id"),
            "summary": summary,
        })

    out = {
        "stored": sum(1 for r in records if r["stored"]),
        "unmodified": sum(1 for r in records if r.get("change_state") == "unchanged"),
        "changed": sum(1 for r in records if r.get("changed") is True),
        "investigation_id": investigation_id,
        "records": records,
    }
    if len(all_targets) > len(targets):
        # The file cap cut the tree short: say so rather than read as complete.
        out.update(truncated=True, files_found=len(all_targets),
                   files_ingested=len(targets), max_files=_DOCS_INGEST_MAX_FILES)
    return json.dumps(out)


def _docs_search_score(text: str, query: str) -> float:
    """Lexical match score of ``query`` against an indexed document string.

    1.0 when the whole query occurs as a phrase; otherwise the fraction of
    distinct query tokens that occur in the text (0.0 = no match). This is the
    only relevance signal docs_search has. It is lexical, not semantic, and is
    reported as such (``score_kind``) rather than presented as a similarity.
    """
    if not text or not query:
        return 0.0
    haystack = text.lower()
    needle = query.lower().strip()
    if not needle:
        return 0.0
    if needle in haystack:
        return 1.0
    tokens = sorted({token for token in re.findall(r"[A-Za-z0-9]+", needle) if token})
    if not tokens:
        return 0.0
    return round(sum(1 for token in tokens if token in haystack) / len(tokens), 4)


@mcp.tool()
def docs_search(
    query: str,
    investigation_id: str = "loci-docs-index",
    limit: int = 5,
    include_excerpt: bool = False,
) -> str:
    """Search stored markdown/text guidance by query and return concise hits."""
    q = (query or "").strip()
    if not q:
        return json.dumps({
            "error": "Query must be a non-empty string.",
            "count": 0,
            "results": [],
            "investigation_id": investigation_id,
        })

    try:
        n = int(limit)
    except (TypeError, ValueError):
        return json.dumps({
            "error": "limit must be an integer.",
            "count": 0,
            "results": [],
            "q": q,
            "investigation_id": investigation_id,
        })
    if n <= 0:
        return json.dumps({
            "error": "limit must be positive.",
            "count": 0,
            "results": [],
            "q": q,
            "investigation_id": investigation_id,
        })

    findings_path = _inv_dir(investigation_id) / "findings.jsonl"
    if not findings_path.exists():
        return json.dumps({
            "error": f"No indexed docs found for investigation_id: {investigation_id}",
            "count": 0,
            "results": [],
            "query": q,
            "investigation_id": investigation_id,
        })

    results: list[dict] = []
    for finding in _read_jsonl(findings_path):
        tags = {str(tag).lower() for tag in finding.get("tags", [])}
        metadata = finding.get("metadata") or {}
        if "docs" not in tags and not metadata.get("source_path"):
            continue

        doc_title = metadata.get("title") or (Path(str(metadata.get("source_path") or "")).stem or "Indexed document")
        summary = metadata.get("doc_summary") or str(finding.get("text") or "").strip() or "No summary available."
        search_text = " ".join([
            str(finding.get("text") or ""),
            str(metadata.get("doc_summary") or ""),
            str(metadata.get("content_excerpt") or ""),
        ])
        score = _docs_search_score(search_text, q)
        if score <= 0.0:
            continue

        hit = {
            "title": doc_title,
            "path": str(metadata.get("source_path") or ""),
            "summary": summary,
            "finding_id": finding.get("id"),
            "score": score,
            "score_kind": "lexical",
            "origin": "docs_search",
        }
        if include_excerpt and metadata.get("content_excerpt"):
            hit["excerpt"] = str(metadata["content_excerpt"])[:1000]
        results.append(hit)

    # Best lexical match first; ties keep file order (the sort is stable).
    results.sort(key=lambda h: -h["score"])
    results = results[:n]

    if not results:
        return json.dumps({
            "error": f"No matching docs found for query: {q}",
            "count": 0,
            "results": [],
            "query": q,
            "investigation_id": investigation_id,
        })

    return json.dumps({
        "query": q,
        "count": len(results),
        "results": results,
        "investigation_id": investigation_id,
    })


@mcp.tool()
def docs_recall(
    query: str,
    investigation_id: str = "loci-docs-index",
    limit: int = 5,
    include_excerpt: bool = False,
) -> str:
    """Recall indexed docs guidance by query using the existing docs index/search path."""
    q = (query or "").strip()
    if not q:
        return json.dumps({
            "error": "Query must be a non-empty string.",
            "count": 0,
            "results": [],
            "query": q,
            "investigation_id": investigation_id,
            "source": "docs_search",
        })

    try:
        result = json.loads(docs_search(q, investigation_id=investigation_id, limit=limit, include_excerpt=include_excerpt))
    except Exception as exc:
        return json.dumps({
            "error": f"docs recall failed: {exc}",
            "count": 0,
            "results": [],
            "query": q,
            "investigation_id": investigation_id,
            "source": "docs_search",
        })

    docs_results = []
    for item in result.get("results", []):
        docs_results.append({
            "title": item.get("title"),
            "path": item.get("path"),
            "summary": item.get("summary"),
            "finding_id": item.get("finding_id"),
            "excerpt": item.get("excerpt"),
            "source": "docs_search",
            # docs_search's own match score (lexical phrase/token overlap), never
            # a constant: a fixed 0.95 read as a strong semantic hit even when
            # one shared token was the only match.
            "score": item.get("score"),
            "score_kind": item.get("score_kind", "lexical"),
            "origin": item.get("origin", "docs_search"),
        })

    if result.get("error") and not docs_results:
        return json.dumps({
            "error": result["error"],
            "count": 0,
            "results": [],
            "query": q,
            "investigation_id": investigation_id,
            "source": "docs_search",
        })

    return json.dumps({
        "query": q,
        "count": len(docs_results),
        "results": docs_results,
        "investigation_id": investigation_id,
        "source": "docs_search",
    })


def _store_build_finding(investigation_id, finding_type, text, source, confidence, tags,
                         derived_from, numeric_confidence, procedure_preconditions,
                         procedure_steps, procedure_postconditions, valid_from,
                         valid_until, authored_by, tier, resolution, code_refs, metadata,
                         evidence_provenance_tier):
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
    normalized_metadata = _normalize_finding_metadata(metadata)
    normalized_metadata, flybrain_scope_error = _normalize_and_validate_flybrain_claim_scope(normalized_metadata)
    if flybrain_scope_error:
        return None, json.dumps({"error": flybrain_scope_error})
    if isinstance(normalized_metadata, dict) and isinstance(normalized_metadata.get("flybrain_provenance"), dict):
        flybrain_prov = normalized_metadata["flybrain_provenance"]
        if isinstance(flybrain_prov.get("audit_hook"), dict):
            flybrain_prov["audit_hook"]["investigation_id"] = investigation_id
            flybrain_prov["decision"] = flybrain_prov["audit_hook"].get("decision", flybrain_prov.get("decision", "accepted"))
            _emit_flybrain_claim_audit(
                investigation_id,
                decision=str(flybrain_prov["audit_hook"].get("decision") or flybrain_prov.get("decision") or "accepted"),
                claim_scope=flybrain_prov.get("claim_scope") or normalized_metadata.get("claim_scope"),
                reason=flybrain_prov["audit_hook"].get("reason") or flybrain_prov.get("reason") or "scope_validated",
                metadata=normalized_metadata,
                tool_name="flybrain_claim_scope_validator",
                tier=str(flybrain_prov["audit_hook"].get("tier") or flybrain_prov.get("tier") or "T1"),
            )
    # An assumption or a gap is by definition not observed evidence; unless the
    # caller asserted a tier, it must not inherit the legacy tool_verified default.
    if (not evidence_provenance_tier and finding_type in ("assumed", "gap")
            and provenance_fields({"metadata": normalized_metadata})["provenance_defaulted"]):
        evidence_provenance_tier = MODEL_ASSERTED
    if evidence_provenance_tier:
        normalized_metadata = normalized_metadata or {}
        normalized_metadata["evidence_provenance_tier"] = normalize_provenance_tier(
            {"evidence_provenance_tier": evidence_provenance_tier}
        )
    if normalized_metadata is not None:
        finding["metadata"] = normalized_metadata
        if normalized_metadata.get("evidence_provenance_tier"):
            finding["evidence_provenance_tier"] = normalized_metadata["evidence_provenance_tier"]
        finding = apply_finding_fingerprints(
            finding,
            source=source,
            explicit_provenance_tier=normalized_metadata.get("evidence_provenance_tier"),
        )

    finding["entities"] = _extract_entities(text)

    # Stamp sha256 of any referenced code file so a later re-hash can flag the finding stale; fail-open.
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
    with _locked_file(lock_path, "a+", exclusive=True):
        manifest = _load_manifest_fresh(investigation_id)
        if manifest is None:
            logger.debug("_store_commit: manifest vanished before commit for %s", investigation_id)
            return
        _append_jsonl(_inv_dir(investigation_id) / "findings.jsonl", finding)
        counts = manifest.setdefault("finding_counts", {})
        counts[finding_type] = counts.get(finding_type, 0) + 1
        # Update hot-tier manifest notes
        if tier == "hot":
            snippet = text[:200]
            notes = manifest.get("notes") or ""
            manifest["notes"] = (notes + "; " + snippet) if notes else snippet
        _save_manifest(manifest)


def _store_index(investigation_id: str, finding: dict, finding_type: str,
                 text: str, source: str, confidence: str, tier: str) -> bool:
    """Fan the finding out to Mnemosyne, Qdrant, the event log and the graph.

    Returns whether Mnemosyne accepted it. Cold tier skips Qdrant indexing; the
    graph mirror is tier-agnostic, since the relationship graph carries findings
    regardless of index tier.
    """
    finding_metadata = finding.get("metadata")
    claim_scope = (
        finding_metadata.get("claim_scope")
        if isinstance(finding_metadata, dict) and isinstance(finding_metadata.get("claim_scope"), dict)
        else None
    )
    flybrain_provenance = (
        finding_metadata.get("flybrain_provenance")
        if isinstance(finding_metadata, dict) and isinstance(finding_metadata.get("flybrain_provenance"), dict)
        else None
    )
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
            **({"claim_scope": dict(claim_scope)} if claim_scope is not None else {}),
            **({"flybrain_provenance": dict(flybrain_provenance)} if flybrain_provenance is not None else {}),
            # Carry the finding's own provenance tier so a later Mnemosyne recall
            # doesn't lose e.g. model_asserted and silently normalize it to the
            # legacy tool_verified default (see _mnemo_recall).
            **provenance_fields(finding),
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
    metadata: Any | None = None,
    evidence_provenance_tier: Optional[str] = None,
) -> str:
    """
    Record a finding in an investigation.

    Args:
        investigation_id: Investigation identifier.
        finding_type: One of ``observed``, ``inferred``, ``assumed``, ``gap``,
            ``procedure``.
            ``observed`` comes from a direct tool response and should cite source
            plus key values. ``inferred`` is reasoned from observations but not
            directly stated. ``assumed`` is a working hypothesis with no current
            evidence. ``gap`` is something that should be checked but has not
            been. ``procedure`` is a reusable step-by-step runbook entry.
        text: The finding text. For ``observed``, include enough detail to
            reproduce the query, such as table name, time range, and key values.
        source: Tool or data source this came from (e.g. sentinel__run_kql_query).
        confidence: high / medium / low.
        tags: Optional comma-separated tags or a list of tag strings; both forms
            are accepted.
        derived_from: Optional finding IDs or claim strings this finding builds
            on, as one ID or a list. Stored as forward derivation so
            ``memory_retract`` can follow lineage and clean up everything built
            on a false fact. Omit when the finding stands alone.
        numeric_confidence: Optional float in ``[0.0, 1.0]``. When omitted it is
            derived from string confidence: ``high→0.9``, ``medium→0.6``,
            ``low→0.3``. Out-of-range values are clamped.
        procedure_preconditions: ``procedure`` only. Preconditions as comma-
            separated or natural-language text.
        procedure_steps: (procedure only) Numbered steps as a string.
        procedure_postconditions: (procedure only) Expected outcomes after the procedure.
        valid_from: ISO8601 timestamp from which the finding is valid. Defaults
            to store time; use it for facts that were true earlier.
        valid_until: ISO8601 timestamp when the finding stopped being valid, or
            null/default for "currently believed true". Use it for known expiry
            or supersession.
        authored_by: Optional storing agent ID. Used with investigation ACLs to
            filter findings per agent.
        tier: ``"hot"``, ``"warm"``, or ``"cold"``; default ``"warm"``.
            ``hot`` is Qdrant-indexed and summarized in manifest notes.
            ``warm`` is Qdrant-indexed only. ``cold`` is JSONL-only and not
            indexed in Qdrant.
        resolution: Lifecycle state: ``open`` (default), ``fixed``,
            ``intentional``, ``wontfix``, or ``superseded``. The resolved states
            ``fixed``/``intentional``/``wontfix`` let exclusion-aware grounding
            skip handled items on re-audit. Older findings without the field read
            as ``open``.
        code_refs: Optional comma-separated paths or a list of paths this
            finding refers to, e.g. ``"mcp/server.py,mcp/verify.py"``.
            Provided refs are authoritative: pass ``[]`` or ``""`` to assert
            "no refs", and the text will not be parsed. Only ``None`` triggers
            best-effort parsing from ``text`` (tokens like ``"path/file.py:12"``).
            Each readable file's current sha256 is stamped so
            ``investigation_load`` and ``investigation_search`` can later flag
            the finding ``stale`` if the file changes. Fully optional and
            fail-open: unreadable paths are skipped.
        metadata: Optional advisory metadata dict (or JSON string encoding one)
            stored verbatim under ``finding["metadata"]``. Additive only; ignored
            when empty/unparseable-to-dict.
        evidence_provenance_tier: Optional provenance authority for this finding:
            ``human_authored``, ``tool_verified``, ``deterministic_derived``, or
            ``model_asserted``. Untagged legacy findings default to tool-verified
            only when used as evidence, preserving old stores and callers.

    Returns:
        JSON ``{"stored": true, "finding_id": "<uuid>", "type": "<finding_type>",
        "mnemo_stored": true|false, "tier": "<tier>"}``, or ``{"error": ...}``.

    Note: the second positional parameter is ``finding_type``, not
    ``record_type``. Call as
    ``investigation_store(inv_id, "observed", "text", "source", "high")``.
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
        resolution, code_refs, metadata, evidence_provenance_tier,
    )
    if invalid:
        return invalid

    try:
        _store_commit(investigation_id, manifest, finding, finding_type, text, tier)
    except StoreBusyError as exc:
        return _busy_result(exc, investigation_id=investigation_id)
    mnemo_stored = _store_index(investigation_id, finding, finding_type, text, source, confidence, tier)
    conflict_detected, conflicting_finding_id, conflict_id = _store_conflicts(investigation_id, finding)

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
    except StoreBusyError as exc:
        logger.info("finding_resolve busy for %s/%s: %s", investigation_id, finding_id, exc)
        return _busy_result(
            exc,
            investigation_id=investigation_id,
            finding_id=str(finding_id),
            resolution=res,
        )
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


def _procedure_success_rate(success_count: int, attempt_count: int) -> float | None:
    """Successes per attempt, or None when the procedure has never been attempted.

    Untried and always-failed are the two states a caller most needs to tell apart
    when picking a procedure to follow, and 0.0 spells both. A brand-new procedure
    — exactly the one whose author wants it exercised — read as a 0% success rate
    and lost to anything that had ever succeeded once. Never measured has no number.
    """
    if attempt_count <= 0:
        return None
    return round(success_count / attempt_count, 4)


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
               "success_rate": float|null}
        ``success_rate`` is null for a procedure that has never been attempted —
        that is not the same as a 0.0 success rate.
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
        success_rate = _procedure_success_rate(success_count, attempt_count)

        # Atomic, line-preserving rewrite of the first row for this id (the target).
        # Unparseable lines and legacy access rows are kept verbatim.
        _first = [True]

        def _replace_target(f):
            if _first[0] and f.get("id") == finding_id:
                _first[0] = False
                return target
            return None

        inv_store._rewrite_jsonl_preserving(findings_path, _replace_target)

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
        ``success_rate`` is null for a procedure that has never been attempted —
        that is not the same as a 0.0 success rate. ``procedure_meta`` carries
        ``attempt_count`` if you need to rank the untried ones yourself.
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
                    success_rate = _procedure_success_rate(success_count, attempt_count)
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
                                success_rate = _procedure_success_rate(success_count, attempt_count)
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

    The queue is persisted under ``$LOCI_MEMORY_DIR/_reflection-loop/state.json``.
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
    metadata: Optional[dict] = None,
) -> bool:
    """Store one reflection-loop finding; True when it was actually written.

    ``source`` is always ``"reflection_loop_tick"``. No try/except on purpose —
    a failing store must propagate exactly as it does inline.
    """
    payload: dict[str, Any] = {
        "investigation_id": investigation_id,
        "finding_type": finding_type,
        "text": text,
        "source": "reflection_loop_tick",
        "confidence": confidence,
        "tags": tags,
        # Heuristic self-reflection, no receipt: never the legacy tool_verified default.
        "evidence_provenance_tier": MODEL_ASSERTED,
    }
    if metadata is not None:
        payload["metadata"] = metadata
    return bool(json.loads(investigation_store(**payload)).get("stored"))


def _reflection_llm_triage_metadata(
    kind: str,
    path: str,
    summary: dict,
    visible_errors: dict[str, int],
    visible_warnings: dict[str, int],
) -> Optional[dict]:
    """Best-effort advisory semantic triage for one reflection finding."""
    try:
        import reflection_triage as _rt
        triage = _rt.classify_reflection_observation(
            kind,
            path,
            sampling_mode=str(summary.get("sampling_mode") or "full"),
            events=summary.get("events") or {},
            tools=summary.get("tools") or {},
            errors=visible_errors,
            warnings=visible_warnings,
        )
    except Exception:
        triage = {"degraded": True}
    llm_triage: dict[str, Any] = {}
    if triage.get("category"):
        llm_triage["category"] = triage["category"]
    if triage.get("novelty"):
        llm_triage["novelty"] = triage["novelty"]
    if triage.get("degraded"):
        llm_triage["degraded"] = True
    return {"llm_triage": llm_triage} if llm_triage else None


# ---- Tool: reflection_loop_tick ----

def _reflection_item_finding(kind, path, summary, raw_errors, raw_warnings,
                             error_observations, warning_observations, stats,
                             investigation_id, low_signal_session_events,
                             enable_llm_triage=False, llm_triage_budget=None) -> bool:
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
    finding_metadata = None
    remaining = int((llm_triage_budget or {}).get("remaining") or 0)
    if enable_llm_triage and remaining > 0 and (visible_errors or visible_warnings):
        if llm_triage_budget is not None:
            llm_triage_budget["remaining"] = max(0, remaining - 1)
        finding_metadata = _reflection_llm_triage_metadata(
            kind, path, summary, visible_errors, visible_warnings
        )
    return bool(_reflection_store_finding(
        investigation_id=investigation_id,
        finding_type="observed",
        text=finding_text,
        confidence="low",
        tags="self-reflection,loop-tick,artifact-mining,unreceipted-observed",
        metadata=finding_metadata,
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
    enable_llm_triage: bool = False,
    max_llm_items: int = 3,
) -> str:
    """
    Process a small queue batch for self-reflection and store findings.

    Designed to avoid passive burn:
    - bounded by ``max_items`` and ``max_lines_per_file``
    - deterministic parsing only by default (no LLM pass)
    - optional bounded local-model advisory triage when ``enable_llm_triage=True``
    - writes findings through ``investigation_store`` (JSONL + Mnemosyne + Qdrant)
    """
    max_items = max(1, min(int(max_items), 20))
    max_lines_per_file = max(50, min(int(max_lines_per_file), 20000))
    max_llm_items = max(0, min(int(max_llm_items), 20))
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
    llm_triage_budget = {"remaining": max_llm_items} if enable_llm_triage else {"remaining": 0}

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
        if store_item_findings:
            # Only a tick that could store findings may mark an item done:
            # reflection_loop_seed permanently excludes every key in processed, so a
            # preview tick (store_item_findings=False writes nothing) would otherwise
            # retire the artifacts it only looked at, and no reseed brings them back.
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
            enable_llm_triage=enable_llm_triage,
            llm_triage_budget=llm_triage_budget,
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

    Thin view over the shared recall filter. A malformed or unreadable
    investigation dir is handled per dir, so it never empties the whole map.
    """
    rf = build_recall_filter(MEMORY_DIR, [investigation_id] if investigation_id else None)
    return rf.retracted, rf.retracted_texts


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

    # Lowercase so "High"/"MEDIUM" are not silently ignored by the lookup below.
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
            # mnemosyne rows may be mixed-case; without .lower() "High" maps to rank 0.
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
        # mnemo/qdrant rows carry no code_refs; consult the JSONL-derived map.
        _rfid = str(r.get("finding_id") or r.get("id") or "")
        _refs = _coderefs_map.get(_rfid)
        if _refs:
            _st = _finding_is_stale({"code_refs": _refs})
            if _st is not None:
                r["stale"] = _st


def _search_empty_response(
    qdrant: dict,
    mnemo_enabled: bool,
    excluded_retracted: int = 0,
    include_retracted: bool = False,
    retraction_filter: Optional[dict] = None,
) -> str:
    """Build the JSON payload for an empty investigation_search result set.

    An empty result is only ``no_matches`` when every lane answered and
    nothing matched. Rows filtered as retracted report ``all_retracted`` and a
    failed Qdrant lane (e.g. ``embedding_unavailable``) reports ``rag_degraded``.
    """
    qdrant_avail = bool(os.environ.get("QDRANT_URL", ""))
    honesty = {
        "excluded_retracted": excluded_retracted,
        "include_retracted": include_retracted,
        "retraction_filter": retraction_filter or {"status": "ok"},
    }
    # rag_required only when Qdrant is down; empty results with Qdrant up is a normal no-match.
    if not qdrant_avail or (qdrant.get("reason") == "qdrant_unavailable"):
        return json.dumps({
            "mode": "rag_required",
            "reason": "qdrant_unavailable",
            "results": [],
            "qdrant_enabled": qdrant_avail,
            "error": "RAG_REQUIRED: Qdrant unavailable. Check QDRANT_URL and QDRANT_API_KEY.",
            **honesty,
        }, indent=2)
    _reason = str(qdrant.get("reason") or "no_matches")
    _qdrant_failed = not qdrant.get("ok") and _reason != "not_attempted"
    if _qdrant_failed:
        mode = "rag_degraded"
    elif excluded_retracted:
        mode = "all_retracted"
    else:
        mode = "no_matches"
    payload = {
        "mode": mode,
        "reason": _reason,
        "results": [],
        "qdrant_enabled": qdrant_avail,
        **honesty,
        "mnemo_status": {
            "enabled": mnemo_enabled,
            "bank": _mnemo_bank() if mnemo_enabled else None,
            "match_count": 0,
        },
    }
    if _qdrant_failed:
        payload["error"] = f"Qdrant search failed ({_reason}); an empty result is not evidence of no matches."
    return json.dumps(payload, indent=2)


@mcp.tool()
def investigation_search(
    query: str,
    investigation_id: Optional[str] = None,
    limit: int = 10,
    include_retracted: bool = False,
    min_confidence: str = "low",
    resolution: Optional[str] = None,
    requesting_agent_id: Optional[str] = None,
) -> str:
    """
    Search findings by similarity.
    Resolution order: Mnemosyne recall (primary) → Qdrant semantic/hybrid
    enrichment (secondary, when needed) → local keyword scoring fallback.
    A slow cross-session neuromodulation layer may gently bias recall/query
    overfetch limits while preserving the same fail-open retrieval semantics.

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
        requesting_agent_id: Optional agent_id of the caller. Rows from an
                    investigation with a non-empty ACL that the caller is neither
                    owner nor member of are dropped and counted under
                    ``excluded_acl``.

    Returns:
        JSON list of matching findings with investigation context.
    """
    min_confidence, resolution = _search_normalize_filters(min_confidence, resolution)

    _acl_denied_by_inv: dict[str, Optional[str]] = {}

    def _acl_denied(inv_id: str) -> Optional[str]:
        if inv_id not in _acl_denied_by_inv:
            try:
                manifest = _load_manifest(inv_id) if inv_id else None
            except Exception:  # noqa: BLE001 — a malformed row id has no manifest, hence no ACL
                manifest = None
            _acl_denied_by_inv[inv_id] = (
                inv_store._acl_access_denied(manifest, requesting_agent_id) if manifest else None
            )
        return _acl_denied_by_inv[inv_id]

    if investigation_id and _acl_denied(str(investigation_id)):
        return json.dumps({"error": "permission_denied", "detail": _acl_denied(str(investigation_id))})
    _excluded_acl = {"n": 0}

    # Per-dir fail-safe: a malformed dir is reported, never disables filtering.
    _rfilter = (
        build_recall_filter(MEMORY_DIR, [investigation_id] if investigation_id else None)
        if not include_retracted else None
    )

    _excluded_retracted: set[str] = set()  # distinct findings, not duplicate rows

    _, recall_fn = _get_mnemo_funcs()
    mnemo_enabled = recall_fn is not None
    _base_mnemo_top_k = max(limit * 4, 20)
    _base_qdrant_limit = max(limit * 3, 20)
    try:
        _route_mod = routing_policy(
            load_state(MEMORY_DIR),
            mnemo_top_k=_base_mnemo_top_k,
            qdrant_limit=_base_qdrant_limit,
        )
        assert_routing_policy_invariants(
            _route_mod,
            minimum_top_k=1,
        )
    except Exception as exc:
        logger.warning("investigation_search slow-neuromod invariant failed; fail-closed baseline policy: %r", exc)
        _route_mod = {
            "routing_tone": 0.0,
            "mnemo_top_k": _base_mnemo_top_k,
            "qdrant_limit": _base_qdrant_limit,
        }
    mnemo_rows = _mnemo_recall(
        query,
        top_k=int(_route_mod.get("mnemo_top_k", _base_mnemo_top_k)),
        investigation_id=investigation_id,
    )

    deduped: list[dict] = []
    seen: set[str] = set()

    def _add_row(row: dict) -> None:
        if _rfilter is not None and _rfilter.is_retracted(row):
            _excluded_retracted.add(_rfilter.finding_key(row))
            return
        if _acl_denied(str(row.get("investigation_id") or "")):
            _excluded_acl["n"] += 1
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
    _pre_qdrant_count = len(deduped)
    if len(deduped) < limit:
        qdrant = _qdrant_similarity_search(
            query,
            investigation_id=investigation_id,
            limit=int(_route_mod.get("qdrant_limit", _base_qdrant_limit)),  # overfetch to absorb dedup losses
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
    for row in deduped:
        text = str(row.get("text") or "")
        if not text:
            continue
        row["text"] = wrap_untrusted_memory_text(
            text,
            origin=row.get("origin") or "loci_memory",
            investigation_id=row.get("investigation_id"),
            finding_id=row.get("finding_id") or row.get("id"),
            kind=row.get("record_type") or row.get("type") or "finding",
            source=row.get("source"),
        )

    _rfilter_status = _rfilter.status() if _rfilter is not None else {"status": "ok"}
    if not deduped:
        empty = _search_empty_response(
            qdrant, mnemo_enabled, len(_excluded_retracted), include_retracted, _rfilter_status
        )
        if _excluded_acl["n"]:
            # Everything matched was withheld by an ACL; say so rather than "no matches".
            empty = json.dumps({**json.loads(empty), "excluded_acl": _excluded_acl["n"]}, indent=2)
        return empty

    mode = "mnemo_primary"
    if qdrant.get("ok"):
        mode = f"mnemo+{qdrant.get('reason')}" if mnemo_rows else str(qdrant.get("reason"))
    try:
        _qdrant_gain = max(0, len(deduped) - _pre_qdrant_count)
        _routing_signal = 0.0
        if _qdrant_gain > 0:
            _routing_signal = min(0.35, 0.08 * _qdrant_gain)
        elif mnemo_rows:
            _routing_signal = -0.08
        observe(MEMORY_DIR, event="investigation_search", routing_signal=_routing_signal)
    except Exception as exc:
        logger.debug("investigation_search slow-neuromod update failed (fail-open): %r", exc)

    return json.dumps({
        "mode": mode,
        "results": deduped[: max(1, min(limit, 200))],
        "excluded_retracted": len(_excluded_retracted),
        **({"excluded_acl": _excluded_acl["n"]} if _excluded_acl["n"] else {}),
        "include_retracted": include_retracted,
        "retraction_filter": _rfilter_status,
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


def _firewall_linked_evidence(investigation_id: str, finding: dict,
                              pool: Optional[list] = None) -> list[dict]:
    """Evidence rows actually linked to ``finding``'s claim, for the provenance firewall.

    Its ``derived_from`` parents plus the rows the pre_answer_check lexical lane
    counts as support (findings and audit receipts). Never "every other finding":
    an unrelated row says nothing about this claim. No link -> [] (fails closed).
    ``pool`` lets a batch caller build the validation evidence once.
    """
    if pool is None:
        pool, _ = build_validation_evidence(investigation_id, min_confidence="low")
    fid = str(finding.get("id") or "")
    parents = set(_normalize_derived_from(finding.get("derived_from")))
    others = [e for e in pool if str(e.get("evidence_id") or "") != fid]
    claim = str(finding.get("text") or "")
    support, _ = _pre_answer_lexical_refs(tokenize(claim), bool(_NEGATION_RE.search(claim)), others)
    linked_ids = parents | {str(ref.get("evidence_id") or "") for ref in support}
    return [e for e in others if str(e.get("evidence_id") or "") in linked_ids]


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
                if agg is None:
                    # The chain walk failed: no measurement. It must not enter the
                    # min() as 1.0, which cannot lower the bucket and so renders an
                    # unread chain as 'high'.
                    continue
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


def _should_run_pre_answer_entailment_check(
    support_refs: list[dict],
    support_basis: str,
    contradiction_refs: list[dict],
    benign_context_refs: list[dict],
) -> bool:
    """When should the additive LLM entailment lane run?

    The gap this lane closes is lexical-only or otherwise borderline support:
    a finding can share enough words to look supportive while actually talking
    about the wrong subject, timeframe, or certainty level. Stronger semantic
    corroboration stays sufficient on its own unless the overall result is still
    mixed (contradictions / benign baseline alongside support).
    """
    return bool(support_refs) and (
        support_basis == "lexical"
        or bool(contradiction_refs)
        or bool(benign_context_refs)
    )


def _pre_answer_entailment_evidence(
    refs: list[dict],
    role: str,
    evidence_by_id: dict[str, dict],
    seen_ids: set[str],
) -> list[dict]:
    """Hydrate surfaced refs back to fuller evidence text for the advisory prompt.

    The user-facing refs intentionally stay compact (snippet + metadata). The
    entailment checker, however, should reason over the actual cited evidence
    when we still have it locally (findings/audit text). Dense-only matches may
    only have a snippet; that still degrades safely to a shorter prompt.
    """
    rows: list[dict] = []
    for ref in refs or []:
        if not isinstance(ref, dict):
            continue
        evidence_id = str(ref.get("evidence_id") or "").strip()
        dedupe_key = evidence_id or f"{role}:{len(rows)}"
        if dedupe_key in seen_ids:
            continue
        seen_ids.add(dedupe_key)
        source = evidence_by_id.get(evidence_id, {}) if evidence_id else {}
        text = str(source.get("text") or ref.get("text") or ref.get("snippet") or "").strip()
        if not text:
            continue
        rows.append({
            "role": role,
            "evidence_id": evidence_id,
            "record_type": ref.get("record_type") or source.get("record_type"),
            "source": ref.get("source") or source.get("source"),
            "ts": ref.get("ts") or source.get("ts"),
            "text": text,
            "snippet": ref.get("snippet", ""),
        })
    return rows


def _run_pre_answer_llm_entailment_check(
    claim: str,
    support_refs: list[dict],
    contradiction_refs: list[dict],
    benign_context_refs: list[dict],
    evidence_by_id: dict[str, dict],
) -> dict:
    """Best-effort advisory entailment corroboration for one claim. Never raises."""
    try:
        import pre_answer_entailment as _entailment
    except Exception as exc:
        return {
            "available": False,
            "verdict": None,
            "rationale": "",
            "confidence": 0.0,
            "degraded": True,
            "error": f"pre_answer_entailment import failed: {exc}"[:200],
        }

    seen_ids: set[str] = set()
    evidence = []
    evidence.extend(_pre_answer_entailment_evidence(support_refs, "support", evidence_by_id, seen_ids))
    evidence.extend(
        _pre_answer_entailment_evidence(
            contradiction_refs, "contradiction", evidence_by_id, seen_ids
        )
    )
    evidence.extend(
        _pre_answer_entailment_evidence(
            benign_context_refs, "benign_context", evidence_by_id, seen_ids
        )
    )
    return _entailment.check_claim_entailment(claim, evidence)


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
    ``loci_verdicts`` Qdrant collection so prior-check history and conflict
    detection are available on subsequent calls. Each claim result gains three
    fields: ``verdict_type`` (claim_supported / claim_contradicted /
    claim_unsupported), ``prior_occurrences``, and ``verdict_conflict``.

    The dense-similarity lane is a candidate generator, not an adjudicator. A
    neighbour only enters ``support_refs`` (and the confidence chain) when
    ``_semantic_ref_corroborated`` holds; the rest are surfaced, labelled, under
    ``semantic_candidates``. ``support_basis`` reports which lane decided:
    lexical / semantic_corroborated / semantic_candidate_only / none, and
    ``supported`` is True only for the first two.

    For supported-but-lexical or otherwise borderline claims, ``claim_results``
    may also include ``llm_entailment_check``: an advisory local-model verdict
    (confirmed / refuted / uncertain) on whether the cited evidence actually
    supports the EXACT claim, considering subject, scope, time, modality, and
    negation. This NEVER flips ``supported``; it is additive corroboration only.
    If the local verifier is unavailable the field stays fail-open as
    ``available=False`` and deterministic results are unchanged.
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

    evidence_pool, evidence_lanes = build_validation_evidence(
        investigation_id, min_confidence=min_confidence
    )
    evidence_by_id: dict[str, dict] = {
        str(entry.get("evidence_id") or ""): entry
        for entry in evidence_pool
        if isinstance(entry, dict) and entry.get("evidence_id")
    }
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
            ev_id = str(ref.get("evidence_id") or "")
            if ev_id:
                evidence_by_id[ev_id] = ref
        lexical_support = bool(claim_support_refs)
        semantic_candidates: list[dict] = []
        semantic_corroborated = False
        for ref in qdrant_refs:
            score = float(ref.get("score", 0.0))
            if score < _QDRANT_SUPPORT_MIN_SCORE:
                continue
            made = _make_ref(ref, "support", score=score)
            if _semantic_ref_corroborated(made):
                claim_support_refs.append(made)
                semantic_corroborated = True
            else:
                made["match_type"] = "semantic_candidate"
                semantic_candidates.append(made)

        if lexical_support:
            support_basis = "lexical"
        elif semantic_corroborated:
            support_basis = "semantic_corroborated"
        elif semantic_candidates:
            support_basis = "semantic_candidate_only"
        else:
            support_basis = "none"

        support_evidence_rows = []
        provenance_firewall = {"allowed": True, "phase": "investigation_pre_answer_check",
                               "reason": "no_support_refs", "degraded": False}
        if claim_support_refs:
            for ref in claim_support_refs:
                ev_id = str(ref.get("evidence_id") or "")
                support_evidence_rows.append(evidence_by_id.get(ev_id) or ref)
            provenance_firewall = assert_evidence_firewall(
                {"evidence_provenance_tier": MODEL_ASSERTED},
                support_evidence_rows,
                phase="investigation_pre_answer_check",
            )
            if not provenance_firewall.get("allowed"):
                claim_support_refs = []
                support_basis = "provenance_blocked"

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

        # Benign baseline only where there IS support: it cannot make an already-unsupported claim ambiguous.
        benign_context_refs = (
            _search_benign_context_qdrant(claim, investigation_id)
            if claim_support_refs else []
        )
        claim_result = {
            "claim": claim,
            "supported": bool(claim_support_refs),
            "contradicted": bool(claim_contradiction_refs),
            "ambiguous": bool(claim_support_refs and benign_context_refs),
            "support_basis": support_basis,
            "support_refs": claim_support_refs[:8],
            "semantic_candidates": semantic_candidates[:8],
            "contradiction_refs": claim_contradiction_refs[:8],
            "benign_context_refs": benign_context_refs,
            "provenance_firewall": provenance_firewall,
        }
        if _should_run_pre_answer_entailment_check(
            claim_support_refs, support_basis, claim_contradiction_refs, benign_context_refs
        ):
            claim_result["llm_entailment_check"] = _run_pre_answer_llm_entailment_check(
                claim,
                claim_support_refs[:8],
                claim_contradiction_refs[:8],
                benign_context_refs[:8],
                evidence_by_id,
            )

        claim_results.append(claim_result)

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
        "evidence_lanes": evidence_lanes,
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

    evidence_pool, evidence_lanes = build_validation_evidence(
        investigation_id, min_confidence="low"
    )
    lexical_matches: list[dict] = []
    for record in evidence_pool:
        score = _lexical_match_score(query_tokens, record.get("tokens", set()))
        if score < min_similarity:
            continue
        lexical_matches.append(_make_ref(record, "similar", score=score))

    lexical_matches.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    # Advisory tool, so the ref fields are informational: _semantic_ref_corroborated is NOT applied — its thresholds were fitted on other probes.
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

    provenance_firewall = {"allowed": True, "phase": "investigation_evidence_precheck",
                           "reason": "no_similar_evidence", "degraded": False}
    if deduped:
        provenance_firewall = assert_evidence_firewall(
            {"evidence_provenance_tier": MODEL_ASSERTED},
            deduped,
            phase="investigation_evidence_precheck",
        )
        if not provenance_firewall.get("allowed"):
            deduped = []

    return json.dumps({
        "investigation_id": investigation_id,
        "proposed_query": query,
        "has_similar_evidence": bool(deduped),
        "similar_evidence_count": len(deduped),
        "similar_evidence": deduped[:10],
        "provenance_firewall": provenance_firewall,
        "qdrant_status": {
            "enabled": qdrant_enabled,
            "available": qdrant_available,
            "match_count": len(qdrant_matches),
            "errors": qdrant_errors,
        },
        "evidence_lanes": evidence_lanes,
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
        # The caller wrote this receipt; a model tool's output is model_asserted.
        **audit_provenance_fields(tool_name),
    }
    fb_audit_fp = flybrain_audit_fingerprint(tool_name, inputs_json, output)
    if isinstance(fb_audit_fp, dict):
        entry.update(fb_audit_fp)
    entry["entities"] = _extract_entities(embedding_text or output[:2000])

    # Global daily audit log
    audit_dir = MEMORY_DIR.parent / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        _append_jsonl(audit_dir / f"{date_str}.jsonl", entry)
    except StoreBusyError as exc:
        logger.info("audit_log busy for global audit entry: %s", exc)
        return _busy_result(exc, investigation_id=investigation_id, tool=tool_name)

    # Investigation-scoped audit log
    investigation_logged = False
    if investigation_id and _load_manifest(investigation_id):
        try:
            _append_jsonl(_inv_dir(investigation_id) / "audit.jsonl", entry)
            investigation_logged = True
        except StoreBusyError as exc:
            logger.info("audit_log busy for investigation %s: %s", investigation_id, exc)

    embed_text = embedding_text or f"{tool_name}: {output[:2000]}"
    mnemo_stored = _mnemo_remember(
        embed_text,
        importance=0.65,
        metadata={
            "record_type": "audit",
            "tool": tool_name,
            "investigation_id": investigation_id,
            "source": "audit_log",
            **audit_provenance_fields(tool_name),
            **({"flybrain_provenance": entry.get("flybrain_provenance")}
               if isinstance(entry.get("flybrain_provenance"), dict) else {}),
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
        "investigation_logged": investigation_logged if investigation_id else None,
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
    """Record verdicts into the loci_verdicts qdrant collection. Fail-open.

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
        from memcheck.vectors import COLLECTION, VECTOR_NAME, ensure_collection, hash_embed

        ensure_collection(client, COLLECTION)
        backend = QdrantBackend(
            client=client,
            collection=COLLECTION,
            embed=hash_embed,
            vector_name=VECTOR_NAME,
        )
        engine = VerdictEngine(backend, EmlConfig())

        async def _record_all() -> None:
            for v in verdicts:
                await engine.record(v, hash_embed(v.subject_excerpt))

        # FastMCP dispatches sync tools inline on a running loop, where asyncio.run() raises.
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

    It verifies two rules the server otherwise cannot prove:
    - ``provenance``: every ``observed`` finding should trace to an audit
      receipt; missing ones become ``unsupported_observed`` verdicts.
    - ``contradiction``: surfaces finding pairs that appear to disagree.

    It also surfaces ``hallucination_candidates``: findings that are both
    unsupported and contradicted by a receipted finding. These are the strongest
    "stored a fact that later failed testing" signal and include a hint to run
    ``memory_retract``. They are never auto-retracted.

    Advisory only: verdicts annotate and surface. Nothing is hidden, deleted, or
    mutated; ``findings.jsonl`` stays append-only. Computation uses JSONL only,
    so the check still works without Qdrant. Recording into
    ``loci_verdicts`` is best-effort.

    Args:
        investigation_id: Investigation to check, or omit to check all.
        checks: Comma-separated subset of: provenance, contradiction.
        record: When true and qdrant is reachable, record verdicts for recall.
        llm_verify: Opt into the deep_think->loci semantic contradiction path:
            embedding subject gate plus LLM polarity judge. It replaces the
            lexical token-overlap heuristic, reducing its false positives and
            catching semantic negations it misses. Default ``False`` keeps the
            check pure/offline. Also enabled by env
            ``MEMCHECK_LLM_CONTRADICTION=1``. Fail-open: lexical verdicts pass
            through if embeddings or the LLM are unavailable.

    Returns:
        JSON with advisory verdicts and per-investigation counts. ``degraded`` is
        true and ``degraded_checks`` names each check that crashed (its count is
        then 0 because it did not run, not because the store is clean);
        ``llm_verify_applied`` says whether the LLM path actually ran.
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
    skipped_investigations: list[dict] = []
    degraded_checks: list[str] = []

    for inv_id in targets:
        try:
            computed = _compute_self_check(inv_id, llm_verify=llm_verify)
        except Exception as exc:
            if investigation_id is not None:
                raise
            # Per-dir: a malformed dir (e.g. a legacy 'undefined') is reported, not fatal.
            skipped_investigations.append({"investigation_id": inv_id, "reason": str(exc)})
            continue
        status_map = computed.get("check_status") or {}
        relevant = set(requested)
        if {"provenance", "contradiction"} <= requested:
            relevant.add("hallucination_candidates")
        if llm_verify and "contradiction" in requested:
            relevant.add("llm_verify")
        for name in sorted(relevant):
            state = status_map.get(name, "ok")
            if state not in ("ok", "applied"):
                degraded_checks.append(f"{inv_id}:{name}: {state}")
        inv_verdicts: list = []
        if "provenance" in requested:
            inv_verdicts.extend(computed["unsupported_observed"])
        if "contradiction" in requested:
            inv_verdicts.extend(computed["contradictions"])
        all_verdicts.extend(inv_verdicts)
        # Only surface when both the provenance and contradiction checks ran.
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
        # Carry the lane's state next to the count: an unsupported count computed against zero receipts is not evidence.
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
        # A crashed check reports zero verdicts; degraded says those zeros are not clean.
        "degraded": bool(degraded_checks),
        "degraded_checks": degraded_checks,
    }
    if llm_verify:
        result["llm_verify_applied"] = not any(":llm_verify:" in d for d in degraded_checks)
    if investigation_id is not None:
        result["investigation_id"] = investigation_id
        result["verdicts"] = per_investigation[0]["verdicts"] if per_investigation else []
    else:
        result["investigation_ids"] = [e["investigation_id"] for e in per_investigation]
        result["investigations"] = per_investigation
        result["skipped_investigations"] = skipped_investigations

    return json.dumps(result, indent=2)


# ---- Tool: code_memory_correlate (code -> memory loop) ----

# LH000 is a parse failure; LH001/LH003/LH007/LH009 are the AST-detected smells.
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
    # Raw-text match too: catches entities the typed extractor doesn't bucket (fabricated module names, localhost).
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

    # Graph traversal is primary; the in-memory traversal is the fail-open fallback.
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

    # Ingest the AST (fail-open) so symbols/calls are queryable; suspicion still comes from the checker's flagged identifiers.
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
    Link a suspected code hallucination to contaminated memories.

    This is the code→memory loop: if generated code references a fabricated
    entity such as a fake ``localhost`` endpoint or hallucinated module/symbol,
    and that entity also seeded stored memories, the tool surfaces the
    contaminated lineage so it can be reviewed for cleanup.

    Resolve suspected entities either from ``entity`` directly, or from
    ``target_file`` by running ``run_code_checks`` on an existing ``.py`` file
    and anchoring on its distinctive entities plus any flagged symbol. A file
    with no LH issues is still correlated on its entities and reported as
    advisory-only. At least one of ``entity`` or ``target_file`` is required.

    The tool is strictly advisory and read-only: it only suggests a follow-up
    ``memory_retract`` call. Qdrant, file, and parse errors fail open into a
    well-formed report instead of raising.

    Args:
        investigation_id: Investigation whose memories to correlate against.
        target_file: Optional path to a generated ``.py`` file to check + anchor on.
        entity: Optional suspected-hallucinated entity string (e.g. a fake host
                or endpoint) to anchor on directly.

    Returns:
        JSON ``{investigation_id, suspected_entities, source, code_findings?,
        contaminated_memories:[{finding_id, excerpt, reasons}],
        already_retracted, count, suggestion, advisory:true}``.
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

    # Most-targeted handle: the entity if given, else a seed finding id.
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

# Worst status wins when rolling per-check results up.
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


# Stateless: these read only module globals, which is why they are not closures inside memory_health.
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


# Stateful trio: probe 1 discovers client + collection, 2 records per-collection dims, 3 records the embedder dim, all through the caller's `sink`.
def _health_probe_qdrant_reachable(qdrant_url: str, sink: dict) -> tuple:
    """memory_health probe 1: is QDRANT_URL set and the server answering?

    Uses ``_qdrant_client_readonly()`` (never creates the main collection, unlike
    ``_get_qdrant()``) and makes a live ``get_collections`` call, so a cached client
    cannot report "connected" while Qdrant is down. Publishes ``sink["client"]``
    and ``sink["main_col"]`` for probe 2 only after that call succeeds.
    """
    if not qdrant_url:
        return (
            "warn",
            "QDRANT_URL is not set — qdrant indexing disabled; loci falls back"
            "to mnemosyne/keyword search.",
            "set QDRANT_URL if vector search is expected; otherwise this is benign",
        )
    unreachable_hint = ("confirm the qdrant container is up and reachable at QDRANT_URL "
                        "(docker ps; curl $QDRANT_URL/healthz)")
    client, main_col = _qdrant_client_readonly()
    if client is None:
        return ("fail", f"QDRANT_URL={qdrant_url} is set but no client could be built.",
                unreachable_hint)
    try:
        client.get_collections()  # live round trip, not the cached client object
    except Exception as exc:
        return ("fail", f"QDRANT_URL={qdrant_url} is set but the server did not answer: {exc!r}",
                unreachable_hint)
    sink["client"], sink["main_col"] = client, main_col
    return ("ok", f"connected to qdrant at {qdrant_url}", None)


def _health_probe_qdrant_collections(client, main_col, collection_dims: dict) -> tuple:
    """memory_health probe 2: expected collections present + their layout.

    ``collection_dims`` is mutated in place, so probe 6 observes whatever dims
    are discovered here.
    """
    if client is None:
        return ("warn", "skipped — qdrant not reachable", None)
    from memcheck.vectors import COLLECTION as VERDICTS_COLLECTION
    from memcheck.vectors import EMBED_DIM as VERDICTS_DIM

    existing = {c.name for c in client.get_collections().collections}
    report: dict = {}
    main = main_col or QDRANT_COLLECTION_PREFIX
    main_present = main in existing
    verdicts_present = VERDICTS_COLLECTION in existing
    verdicts_dim = None
    if main_present:
        info = _health_qdrant_collection_info(client, main)
        report[main] = info
        collection_dims[main] = _health_collection_dim(info)
    if verdicts_present:
        info = _health_qdrant_collection_info(client, VERDICTS_COLLECTION)
        report[VERDICTS_COLLECTION] = info
        # Deliberately NOT added to collection_dims: probe 6 compares that dict
        # against the 768-dim embedder, and this collection is 384-dim by design.
        verdicts_dim = _health_collection_dim(info)
        report["expected_verdicts_dim"] = VERDICTS_DIM
    report["expected_main_collection"] = main
    report["main_present"] = main_present
    report["loci_verdicts_present"] = verdicts_present
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
            f"'{VERDICTS_COLLECTION}' not yet created — it appears on first "
            "memory_self_check(record=True); benign until then",
        )
    if verdicts_dim is not None and verdicts_dim != VERDICTS_DIM:
        return (
            "fail",
            report,
            f"'{VERDICTS_COLLECTION}' has dim {verdicts_dim} but every writer "
            f"produces {VERDICTS_DIM}-dim hash vectors — every verdict upsert "
            "fails; recreate the collection at the expected dim",
        )
    return ("ok", report, None)


def _health_probe_embeddings_dense(sink: dict) -> tuple:
    """memory_health probe 3: fastembed dense embedder loads and embeds.

    Publishes ``sink["embed_dim"]`` for the dimension-consistency probe.
    """
    # Uncached: through the embed cache this fixed string hit the embedder once per
    # process and then reported "ok" from memory for the rest of any outage.
    vec = _embed_uncached("memory_health probe")  # transient throwaway, never stored
    if not vec:
        brownout = ""
        try:
            open_now, left = qdrant_ops._breaker_is_open("embed")
            if open_now:
                brownout = f" (embed brownout breaker open for another {left:.0f}s)"
        except Exception as exc:
            logger.debug("embeddings_dense probe: breaker state unreadable: %r", exc)
        return (
            "fail",
            "dense embedder (Ollama) unavailable — semantic/hybrid search "
            "is disabled; the server runs on keyword fallback only." + brownout,
            "ensure Ollama is running and the nomic-embed-text model is available "
            "(OLLAMA_BASE_URL and EMBED_MODEL env vars can override defaults).",
        )
    sink["embed_dim"] = len(vec)
    return ("ok", f"dense embedder active; dimension={sink['embed_dim']}", None)


# collection_dims is passed by reference, so the dimension probe sees what the collections probe wrote.
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
    malformed: list[dict] = []
    skipped: list[dict] = []
    scanned = 0
    for inv in inv_targets:
        try:
            inv_store._validated_investigation_id(inv)
        except ValueError as exc:
            # A legacy dir (e.g. "undefined") is still scanned by path, and reported:
            # its retractions are real and other global tools must tolerate it.
            malformed.append({"investigation_id": inv, "reason": str(exc)})
        try:
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
            active = inv_store._fold_retracted_ids(ret_path) if ret_path.exists() else set()
            orphans = sorted(fid for fid in active if fid not in valid_ids)
            total_active += len(active)
            total_orphans += len(orphans)
            scanned += 1
            if active or orphans:
                per_inv.append({
                    "investigation_id": inv,
                    "active_retractions": len(active),
                    "orphaned_retractions": orphans,
                })
        except Exception as exc:
            # Per-dir: one unreadable dir is reported, never fails the rest.
            skipped.append({"investigation_id": inv, "reason": repr(exc)})
            continue
    detail = {
        "investigations_scanned": scanned,
        "active_retractions": total_active,
        "orphaned_retractions": total_orphans,
        "per_investigation": per_inv,
    }
    if malformed:
        detail["malformed_investigations"] = malformed
    if skipped:
        detail["skipped_investigations"] = skipped
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
    if skipped or malformed:
        return (
            "warn",
            detail,
            "investigation dir(s) with an invalid name or unreadable logs — see "
            "malformed_investigations / skipped_investigations; tools that scan "
            "every investigation must tolerate them",
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
        # Real findings only: _read_jsonl drops legacy access rows. Rows sharing
        # an id count once, and each row with no id counts as its own finding.
        f_rows = [f for f in _read_jsonl(inv_path / "findings.jsonl") if isinstance(f, dict)]
        f_n = (len({str(f["id"]) for f in f_rows if f.get("id")})
               + sum(1 for f in f_rows if not f.get("id")))
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
    broken = counts.get("width_mismatch", 0) + counts.get("error", 0) + counts.get("missing", 0)
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
    Prove Qdrant collections are actually retrievable.

    ``memory_health`` checks substrate wiring for named collections, but stores
    often contain more collections from other ingests or embedding widths. Those
    can fail with width mismatches that otherwise look like simple no-hit
    searches. This tool answers the operational question directly: if queried,
    does the collection return rows?

    For each collection it reports points, dense width, sparse presence, and hit
    count, classifying results as ``ok`` | ``empty`` | ``no_results`` |
    ``width_mismatch`` | ``error`` | ``missing`` and rolling them up to ``ok`` |
    ``degraded`` | ``unhealthy``. Remediations are deduplicated because one
    missing width mapping often explains many collections.

    Read-only. It intentionally bypasses the cross-encoder and relevance floor:
    those judge relevance, while this tool is checking retrieval wiring.

    Scope matters. Stores accumulate collections this server never queries, and
    counting their width mismatches against overall health teaches operators to
    ignore the diagnostic. The default scope is therefore the collections this
    server would actually query.

    Args:
        query: Probe text. Any in-domain phrase works; the point is
            retrievability, not semantic quality.
        limit: Rows to request per collection (default 3).
        collections: Probe exactly these. Overrides ``scope``.
        scope: ``queried`` (default) checks the findings collection plus the
            configured code-chunks collection. ``all`` inventories the whole
            store; width mismatches outside the queried set are still reported
            but do not count against health because this server never asks them
            anything.

    Returns:
        JSON ``{status, summary, scope, collections, remediations}``.
    """
    # Read-only client: _get_qdrant() would recreate a lost main collection empty.
    client, _col = _qdrant_client_readonly()
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
        # A collection this server queries but that does not exist is a fault, not "nothing to probe".
        missing = [] if collections else [n for n in queried if n not in present]
    except Exception as exc:
        return json.dumps({
            "status": "unhealthy",
            "summary": f"could not list collections: {exc}",
            "collections": [],
            "remediations": ["Check Qdrant connectivity and API key scope."],
        }, indent=2)

    if not names and not missing:
        return json.dumps({
            "status": "ok", "summary": "no collections exist",
            "collections": [], "remediations": [],
        }, indent=2)

    query_vec = None
    if names:
        try:
            query_vec = _embed_uncached(query)  # live: a cached default query hides an outage
        except Exception as exc:
            logger.warning("retrieval_selftest: embed failed: %r", exc)

    rows = [probe_collection(query_vec, client, name, limit=limit) for name in names]
    rows += [{"collection": n, "hits": 0, "status": "missing",
              "detail": "queried by this server but does not exist in Qdrant",
              "remediation": (f"'{n}' is missing; if it existed before, its index was lost. "
                              "Restore it from backup or re-run the backfill.")}
             for n in missing]
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
    Check the memory substrate itself, read-only.

    Where ``memory_self_check`` inspects what the server remembered, this checks
    the machinery that remembers: Qdrant reachability and collection layout,
    dense and sparse embedders, the Mnemosyne mirror in this venv,
    vector-dimension consistency, retraction-log integrity, and store counts.
    It is the server-side equivalent of ``mnemosyne diagnose``.

    Strictly read-only: it does not write, mutate, retract, or create anything
    on disk or in Qdrant. The only possible side effect is a transient tiny
    embed to confirm the embedder loads. Every probe is fail-open, so one broken
    probe becomes a ``fail`` entry instead of crashing the tool.

    Args:
        investigation_id: Scope retraction/store checks to one investigation.
            When omitted, substrate checks run globally and retraction/store
            checks summarize across all investigations.

    Returns:
        JSON ``{checked_at, status: ok|degraded|unhealthy,
        checks:[{name,status,detail,remediation?}], summary}``. ``status`` is
        the worst-check rollup: any fail -> unhealthy, any warn -> degraded,
        otherwise ok.
    """
    checks: list[dict] = []
    # Explicit holder shared by the first three probes and the dimension check.
    qd: dict = {"client": None, "main_col": None, "embed_dim": None}

    # 1. qdrant_reachable — is QDRANT_URL set and the server answering?
    qdrant_url = os.environ.get("QDRANT_URL", "")
    checks.append(_health_check(
        "qdrant_reachable",
        lambda: _health_probe_qdrant_reachable(qdrant_url, qd),
    ))

    # 2. qdrant_collections — the lambda defers reading `qd` until probe 1 has populated it.
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

    # 6. dimension_consistency — the lambda defers reading qd["embed_dim"] until probe 3 has run.
    checks.append(_health_check(
        "dimension_consistency",
        lambda: _health_probe_dimension_consistency(qd["embed_dim"], collection_dims),
    ))

    # Read-only: _inv_dir would mkdir.
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
        from memcheck.vectors import COLLECTION, VECTOR_NAME, hash_embed

        # No ensure_collection: forgetting must never create the collection.
        backend = QdrantBackend(
            client=client, collection=COLLECTION, embed=hash_embed, vector_name=VECTOR_NAME,
        )
        engine = VerdictEngine(backend)

        async def _forget_all() -> int:
            removed = 0
            removed += await engine.forget(redact_excerpt(text), "memory")
            return removed

        # asyncio.run() raises when FastMCP dispatches inline on a running event loop.
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


def _retract_propagation_enabled() -> bool:
    return os.environ.get("LOCI_RETRACT_PROPAGATE", "1").strip().lower() not in ("0", "false", "no", "off")


def _qdrant_set_retracted(finding_ids: list[str], *, retracted: bool, ts: str) -> dict:
    """Flag (or unflag) the findings' own Qdrant points as retracted. Soft.

    Sets ``retracted`` (and ``retracted_at`` / ``restored_at``) on each point's
    payload; the vector and the rest of the payload stay, so memory_restore
    flips it back and nothing is deleted. Points are updated one at a time: a
    finding with no point (cold tier, never indexed) is counted as ``missing``
    rather than failing the batch.

    Returns ``{status, updated, missing, failed}``; status is ``ok``,
    ``unavailable`` (no client), ``partial`` or ``failed``.
    """
    ids = sorted({str(f) for f in finding_ids or [] if f})
    out = {"status": "ok", "updated": 0, "missing": 0, "failed": 0}
    if not ids:
        return out
    client, col = _get_qdrant()
    if client is None:
        return {**out, "status": "unavailable"}
    payload = ({"retracted": True, "retracted_at": ts} if retracted
               else {"retracted": False, "restored_at": ts})
    errors: list[str] = []
    for n, fid in enumerate(ids):
        try:
            client.set_payload(collection_name=col, payload=payload, points=[fid], wait=True)
            out["updated"] += 1
        except Exception as exc:
            text = str(exc).lower()
            if "not found" in text or "no point" in text or "404" in text:
                out["missing"] += 1
                continue
            # Anything else (unreachable, timeout, auth) will fail for the rest
            # too; this runs under the investigation lock, so stop here rather
            # than wait out one client timeout per remaining id.
            out["failed"] += len(ids) - n
            errors.append(f"{fid}: {str(exc)[:120]}")
            break
    if out["failed"]:
        out["status"] = "failed" if not out["updated"] else "partial"
        out["errors"] = errors[:5]
        logger.warning("retraction flag not propagated to %d Qdrant point(s): %s", out["failed"], errors[:3])
    return out


def _propagate_retraction(finding_ids: list[str], *, retracted: bool, ts: str,
                          mnemo_stamps: Optional[list[str]] = None) -> dict:
    """Push a retract/restore to the Qdrant point payload and Mnemosyne rows.

    JSONL (retractions.jsonl) stays the source of truth and every read path
    filters by it; this makes the stores that recall reads directly agree with
    it. Fail-open per store, but never silent: each store reports its status.
    """
    if not _retract_propagation_enabled():
        return {"qdrant": {"status": "disabled"}, "mnemosyne": {"status": "disabled"}}
    try:
        q = _qdrant_set_retracted(finding_ids, retracted=retracted, ts=ts)
    except Exception as exc:  # never let propagation undo an applied tombstone
        q = {"status": "failed", "error": str(exc)[:200]}
    try:
        m = _mnemo_set_retracted(finding_ids, retracted=retracted, stamps=mnemo_stamps)
    except Exception as exc:
        m = {"status": "failed", "error": str(exc)[:200]}
    return {"qdrant": q, "mnemosyne": m}


def _retraction_mnemo_stamps(investigation_id: str, finding_id: str) -> list[str]:
    """Every Mnemosyne valid_until stamp memory_retract wrote for this finding."""
    stamps: list[str] = []
    for rec in _read_jsonl(_inv_dir(investigation_id) / "retraction_audit.jsonl"):
        if not isinstance(rec, dict) or rec.get("action") != "retract":
            continue
        if finding_id not in [str(x) for x in rec.get("retracted_finding_ids") or []]:
            continue
        stamp = ((rec.get("propagation") or {}).get("mnemosyne") or {}).get("stamp")
        if stamp:
            stamps.append(str(stamp))
    return stamps


def _owner_only_denied(manifest: dict, investigation_id: str) -> Optional[str]:
    """Owner-only tools (retract/restore): a permission_denied reply, or None.

    The caller is the transport-bound identity (per-agent MCP token, A2A
    session) when there is one, else this process's AGENT_ID. No tool argument
    can name a different caller.
    """
    _owner = manifest.get("owner", "")
    _who = caller_identity.bound_agent_id() or AGENT_ID
    if _owner and _owner != _who:
        return json.dumps({"error": "permission_denied",
                           "detail": f"investigation {investigation_id!r} is owned by {_owner!r}"})
    return None


@mcp.tool()
def memory_retract(
    investigation_id: str,
    target: str,
    reason: str = "",
    dry_run: bool = True,
    scope_semantic: bool = True,
) -> str:
    """
    Retract a hallucinated finding and its contaminated lineage, reversibly.

    If testing proves a stored fact never existed, everything derived from it is
    contaminated. This tool finds that lineage through shared distinctive
    entities, semantic proximity in Qdrant, and forward ``derived_from`` links,
    then soft-tombstones it so it drops out of recall, search, and reflect.
    Nothing is hard-deleted or rewritten: only ``retractions.jsonl`` is
    appended, ``findings.jsonl`` is left as-is, and ``memory_restore``
    reverses retractions exactly (``investigation_as_of`` reads the
    retraction intervals from the log).

    Advisory-first: ``dry_run`` defaults to ``True`` and changes nothing. It
    returns the proposed cluster for review; re-run with ``dry_run=False`` to
    apply the soft tombstone.

    Args:
        investigation_id: Investigation identifier.
        target: A finding ID to retract, or a claim/entity string such as the
            hallucinated URL. Distinctive entities from that text become the
            anchor.
        reason: Why this is being retracted (e.g. "endpoint never existed —
                confirmed by testing"). Recorded in the retraction + audit trail.
        dry_run: When True (default), return the proposed cluster and change
                 nothing. When False, apply the soft tombstone.
        scope_semantic: Include Qdrant semantic neighbors in the cluster
            (default ``True``). Fail-open: if Qdrant is down, entity and
            derivation scope still apply.

    Returns:
        ``dry_run=True`` returns
        ``{seed_ids, would_retract:[{finding_id, text_excerpt, reasons}], count,
        applied:false, advisory:...}``.
        ``dry_run=False`` returns
        ``{seed_ids, retracted:[...], count, applied:true, ...}``.
    """
    manifest = _load_manifest(investigation_id)
    if not manifest:
        return json.dumps({"error": f"Investigation '{investigation_id}' not found."})
    _denied = _owner_only_denied(manifest, investigation_id)
    if _denied:
        return _denied
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

    # Per-investigation lock: no window where a retraction exists without its audit record.
    inv_lock = _investigation_lock(investigation_id)

    try:
        with inv_lock:
            with _locked_file(_inv_dir(investigation_id) / ".lock", "a+", exclusive=True):
                retracted_records, verdicts_forgotten = _retract_write_tombstones(
                    retractions_path, contaminated_ids, by_id, reasons, seed_anchor, reason, ts
                )
                # findings.jsonl is not rewritten: investigation_as_of reads the
                # retraction intervals from retractions.jsonl, so restore is an exact inverse.

                # Flag the findings' own index entries too, so a reader that goes
                # to Qdrant or Mnemosyne directly (not through Loci's filters) also
                # sees them as retracted. Soft: restore reverses both.
                propagation = _propagate_retraction(contaminated_ids, retracted=True, ts=ts)

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
                    "propagation": propagation,
                })
    except StoreBusyError as exc:
        logger.info("memory_retract busy for %s/%s: %s", investigation_id, target, exc)
        return _busy_result(exc, investigation_id=investigation_id, target=str(target).strip())

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
        "propagation": propagation,
        "reversible": ("retractions.jsonl was appended and findings.jsonl is untouched; the Qdrant "
                       "point payload is flagged retracted=true and matching Mnemosyne rows get "
                       "valid_until (see propagation). Nothing is deleted; memory_restore reverses all of it."),
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
    _denied = _owner_only_denied(manifest, investigation_id)
    if _denied:
        return _denied

    retractions_path = _inv_dir(investigation_id) / "retractions.jsonl"

    ts = _now()
    target_fid = str(finding_id or retraction_id or "")
    try:
        with _investigation_lock(investigation_id):
            # Serialise on the per-investigation .lock like memory_retract does:
            # holding a flock on retractions.jsonl itself made the _append_jsonl
            # below wait on this thread's own lock and always return "busy".
            with _locked_file(_inv_dir(investigation_id) / ".lock", "a+", exclusive=True):
                try:
                    existing = _read_jsonl(retractions_path)
                except PermissionError as exc:
                    logger.info("memory_restore busy for %s/%s: %s", investigation_id, finding_id or retraction_id or "", exc)
                    return _busy_result(exc, investigation_id=investigation_id, finding_id=str(finding_id or retraction_id or ""))
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

                _append_jsonl(retractions_path, {
                    "retraction_id": str(uuid.uuid4()),
                    "finding_id": target_fid,
                    "seed_id": None,
                    "reason": reason or "restore",
                    "ts": ts,
                    "active": False,
                })
                propagation = _propagate_retraction(
                    [target_fid], retracted=False, ts=ts,
                    mnemo_stamps=_retraction_mnemo_stamps(investigation_id, target_fid),
                )
                _append_jsonl(_inv_dir(investigation_id) / "retraction_audit.jsonl", {
                    "action": "restore",
                    "ts": ts,
                    "finding_id": target_fid,
                    "reason": reason or "restore",
                    "propagation": propagation,
                })
    except StoreBusyError as exc:
        logger.info("memory_restore busy for %s/%s: %s", investigation_id, target_fid, exc)
        return _busy_result(exc, investigation_id=investigation_id, finding_id=target_fid)

    return json.dumps({
        "finding_id": target_fid,
        "restored": True,
        "reason": reason or "restore",
        "propagation": propagation,
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
        "numeric_confidence": _store_numeric_confidence("medium", None),
        "evidence_provenance_tier": MODEL_ASSERTED,  # a declaration, not evidence
        "tags": tags,
        "derived_from": [],
        "entities": {},
    }
    try:
        with _locked_file(inv_dir / ".lock", "a+", exclusive=True):
            manifest = _load_manifest_fresh(investigation_id)
            if manifest is None:
                return json.dumps({"error": f"Investigation '{investigation_id}' not found."})
            _append_jsonl(inv_dir / "findings.jsonl", finding)
            manifest.setdefault("finding_counts", {})
            manifest["finding_counts"]["gap"] = manifest["finding_counts"].get("gap", 0) + 1
            _save_manifest(manifest)
    except StoreBusyError as exc:
        logger.info("contract_declare busy for %s/%s: %s", investigation_id, entity, exc)
        return _busy_result(exc, investigation_id=investigation_id, entity=entity, role=role)

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
def wiring_obligation_scan(
    content: str,
    path: str = "",
    context: str = "",
) -> str:
    """Advisory-only scan for undeclared implicit wiring obligations.

    This is a suggestion generator only: it inspects a snippet, diff hunk, or
    full file body and returns candidate obligations that MAY merit an explicit
    ``wiring_obligation_declare`` later. It never writes investigation state,
    never auto-declares, and never auto-resolves.

    Args:
        content: Code snippet, diff hunk, or full file content to scan.
        path: Optional path label for the scanned text.
        context: Optional extra operator context (for example "PR diff" or
            "new helper extracted from notifier.py").

    Returns:
        JSON with ``{"candidates": [...], "degraded": bool, "error": str|None}``.
        Fail-open: model/backend errors return an empty candidate list with
        ``degraded=True`` instead of raising.
    """
    import wiring_obligation_scan as _scan

    return json.dumps(_scan.scan(content, path=path, context=context), indent=2)


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
        "numeric_confidence": _store_numeric_confidence("medium", None),
        "evidence_provenance_tier": MODEL_ASSERTED,  # UNVERIFIED obligation, not evidence
        "tags": tags,
        "derived_from": [],
        "entities": {},
    }
    try:
        with _locked_file(inv_dir / ".lock", "a+", exclusive=True):
            manifest = _load_manifest_fresh(investigation_id)
            if manifest is None:
                return json.dumps({"error": f"Investigation '{investigation_id}' not found."})
            _append_jsonl(inv_dir / "findings.jsonl", finding)
            manifest.setdefault("finding_counts", {})
            manifest["finding_counts"]["gap"] = manifest["finding_counts"].get("gap", 0) + 1
            _save_manifest(manifest)
    except StoreBusyError as exc:
        logger.info("wiring_obligation_declare busy for %s/%s.%s: %s",
                    investigation_id, class_name, method_name, exc)
        return _busy_result(
            exc,
            investigation_id=investigation_id,
            class_name=class_name,
            method_name=method_name,
        )
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
    try:
        # Serialise on inv_dir/.lock and let _append_jsonl take the findings.jsonl
        # flock: holding that flock here made the append wait on this thread's own lock.
        with _locked_file(inv_dir / ".lock", "a+", exclusive=True):
            try:
                findings = _read_jsonl(jsonl_path) if jsonl_path.exists() else []
            except PermissionError as exc:
                logger.info("wiring_obligation_resolve busy for %s/%s: %s", investigation_id, finding_id, exc)
                return _busy_result(exc, investigation_id=investigation_id, finding_id=finding_id)

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
            manifest = _load_manifest_fresh(investigation_id)
            if manifest is None:
                return json.dumps({"error": f"Investigation '{investigation_id}' not found."})
            _append_jsonl(jsonl_path, resolved_finding)

            counts = manifest.setdefault("finding_counts", {})
            counts["gap"] = max(0, counts.get("gap", 1) - 1)
            counts["observed"] = counts.get("observed", 0) + 1
            _save_manifest(manifest)
    except StoreBusyError as exc:
        logger.info("wiring_obligation_resolve busy for %s/%s: %s", investigation_id, finding_id, exc)
        return _busy_result(exc, investigation_id=investigation_id, finding_id=finding_id)

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
               "results": [{"finding_id", "verdict", "confidence", "degraded"}, ...]}
               ``degraded`` is True when no model could be reached, so a caller
               can tell "the skeptic was uncertain" from "the skeptic never ran".
        On error: {"error": "<message>"}
    """
    manifest = _load_manifest(investigation_id)
    if not manifest:
        return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

    import verify as _v

    findings = inv_store._fold_provenance_overrides(
        _read_jsonl(_inv_dir(investigation_id) / "findings.jsonl"), investigation_id)
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

    # Limit while iterating instead of materializing every open finding and slicing.
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
    # One evidence pool for the batch; verify_all writes only to the verifications log,
    # so the pool stays current across the loop.
    _linked_pool = (build_validation_evidence(investigation_id, min_confidence="low")[0]
                    if open_findings else [])
    for f in open_findings:
        fid = str(f.get("id", ""))
        # Thread this finding's own stored provenance tier plus the evidence actually
        # linked to it (parents, lexical support), so a model_asserted finding with no
        # independent (human/tool/deterministic) support is gated 'uncertain' by the
        # provenance firewall instead of reaching the model verifier unchecked.
        res = _v.verify_finding(
            str(f.get("text") or ""),
            investigation_id=investigation_id,
            gen_fn=_verify_gen_fn,
            # The finding's stamped {path, hash} refs are the only thing that says
            # WHICH checkout its source lives in; without them the skeptic reasons
            # over prose while the file sits on disk.
            code_refs=f.get("code_refs"),
            candidate_provenance_tier=firewall_candidate_tier(f),
            evidence_rows=_firewall_linked_evidence(investigation_id, f, pool=_linked_pool),
        )
        verdict = res.get("verdict", "uncertain")
        confidence = res.get("confidence", 0.0)
        # Carry degraded: without it a considered "uncertain" is indistinguishable from "nothing ran".
        degraded = bool(res.get("degraded"))
        # Separate log so these never slow the resolution read path; per-finding guard keeps one failed write from aborting the batch.
        try:
            _append_jsonl(_finding_verifications_path(investigation_id), {
                "record_type": "verification",
                "investigation_id": investigation_id,
                "finding_id": fid,
                "verdict": verdict,
                "confidence": confidence,
                "degraded": degraded,
                "ts": _now(),
            })
        except Exception as exc:  # noqa: BLE001 — never abort the batch on one write
            logger.warning("verification note append failed for %s (fail-open, skipped): %r",
                           fid, exc)
        results.append({
            "finding_id": fid,
            "verdict": verdict,
            "confidence": confidence,
            "degraded": degraded,
        })

    return json.dumps({
        "investigation_id": investigation_id,
        "verified": len(results),
        "results": results,
    }, indent=2)


def _embed_probe_headers() -> dict:
    """Auth headers the embed client sends (EMBED_API_KEY), for health probes. Never raises."""
    try:
        import qdrant_ops
        return {k: v for k, v in qdrant_ops._embed_auth_headers().items() if k != "Content-Type"}
    except Exception:
        return {}


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
      ladybug_writer_pid: (optional) PID stamped as the write-lease holder, only while
                         that process is alive
      ollama_reachable:  the resolved Ollama (embed) endpoint answers GET /api/tags; a
                         non-Ollama OpenAI-compatible embeddings host (no /api/tags)
                         counts when it answers that GET with a 4xx
      ollama_gen_reachable: the generation endpoint (ollama_gen_url), same rule
      ollama_gen_model_present: (optional) the configured gen model is listed there
      vllm_reachable:    the resolved vLLM endpoint answers GET /health
      qdrant_reachable:  the resolved Qdrant endpoint answers GET /readyz
                         (each is a short TCP gate followed by a bounded HTTP request)
      embed_model:       configured embedding model
      rerank_model:      configured cross-encoder rerank model
      warm:              whether the embed warm-ping has been fired this process
    """
    out: dict = {
        "status": "ok",
        "code_version": "",
        "ladybug": "unavailable",
        "ollama_reachable": False,
        "ollama_gen_reachable": False,
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
        # Resolve each endpoint once and probe with a SHORT timeout so one dead backend cannot block or mask the others.
        _PROBE_T = 0.5
        explicit_backend = {
            "ollama": bool(os.environ.get("OLLAMA_BASE_URL")
                           or os.environ.get("OLLAMA_URL")
                           or backends._cfg("ollama", "url", "")),
            "vllm": bool(os.environ.get("VLLM_BASE_URL")
                         or backends._cfg("vllm", "url", "")),
            "qdrant": bool(os.environ.get("QDRANT_URL")
                           or backends._cfg("qdrant", "url", "")),
        }
        # Generation has its own endpoint (backends.ollama_gen_url); it falls back to the embed one.
        _gen_explicit = bool(os.environ.get("LOCI_OLLAMA_GEN_URL") or os.environ.get("OLLAMA_GEN_URL")
                             or backends._cfg("ollama", "gen_url", ""))
        explicit_backend["ollama_gen"] = _gen_explicit or explicit_backend["ollama"]
        # TCP accept is not an answer: a hung server or a relay with a dead upstream passes
        # it. After a short TCP gate, each endpoint must answer a bounded HTTP GET.
        _HTTP_T = 1.0
        _qdrant_key = ""
        try:
            _qdrant_key = backends.qdrant()[1]
        except Exception as exc:
            logger.debug("loci_health: qdrant key resolve failed: %r", exc)
        http_answers: dict = {}
        for key, resolver, path, headers in (
            ("ollama_reachable", lambda: backends.ollama_url(_PROBE_T), "/api/tags", None),
            ("ollama_gen_reachable", lambda: backends.ollama_gen_url(_PROBE_T), "/api/tags", None),
            ("vllm_reachable", lambda: backends.vllm_url(probe_timeout=_PROBE_T), "/health", None),
            ("qdrant_reachable", lambda: backends.qdrant()[0], "/readyz",
             {"api-key": _qdrant_key} if _qdrant_key else None),
        ):
            try:
                url = resolver()
                out[key] = False
                if backends._alive(url, timeout=_PROBE_T):
                    ok, body = backends._http_probe(url, path, timeout=_HTTP_T, headers=headers)
                    if not ok and key in ("ollama_reachable", "ollama_gen_reachable"):
                        # OLLAMA_BASE_URL may name any OpenAI-compatible embeddings host, which
                        # has no /api/tags: a 4xx still proves it answers HTTP. A 5xx (e.g. a
                        # relay whose upstream is dead) or no answer does not.
                        _st = backends._http_status(url, path, timeout=_HTTP_T,
                                                    headers=_embed_probe_headers())
                        ok = _st is not None and 400 <= _st < 500
                    out[key] = bool(ok)
                    http_answers[key] = body
            except Exception as exc:
                logger.debug("loci_health: reachability probe %s failed: %r", key, exc)
                pass
        # When a generation model is configured, the gen endpoint must actually carry it.
        _gen_model = os.environ.get("LOCI_OLLAMA_GEN_MODEL") or backends._cfg("ollama", "gen_model", "")
        _tags = http_answers.get("ollama_gen_reachable")
        if (out.get("ollama_gen_reachable") and _gen_model and isinstance(_tags, dict)
                and isinstance(_tags.get("models"), list)):
            names = {str(m.get("name") or m.get("model") or "")
                     for m in _tags["models"] if isinstance(m, dict)}
            out["ollama_gen_model_present"] = bool(
                _gen_model in names or f"{_gen_model}:latest" in names)
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

        failures = []
        optional_down = []
        for label, key in (("ollama", "ollama_reachable"),
                           ("ollama_gen", "ollama_gen_reachable"),
                           ("vllm", "vllm_reachable"),
                           ("qdrant", "qdrant_reachable")):
            if out.get(key):
                continue
            if explicit_backend.get(label, False):
                failures.append(f"{label}: configured/enabled but unreachable")
            else:
                optional_down.append(label)
        if out.get("ollama_gen_model_present") is False:
            failures.append(f"ollama_gen: model {_gen_model!r} not installed at the generation endpoint")
        if failures:
            out["status"] = "unhealthy"
            out["failures"] = failures
            out["remediation"] = (
                "Restore reachability for configured backends or remove explicit enablement."
            )
        elif optional_down:
            out["optional_down"] = optional_down
    except Exception as exc:
        logger.debug("loci_health: backends import/probe block failed: %r", exc)
        pass
    try:
        import embed_ops
        out["warm"] = embed_ops.warmed()
    except Exception as exc:
        logger.debug("loci_health: embed warm-state probe failed: %r", exc)
        pass

    # A 30-day purge default silently deleted older indexed findings on each start and nothing reported it; these fields make that answerable.
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

    tmux_required = os.environ.get("LOCI_TMUX_COMPANION_REQUIRED", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    expected_raw = os.environ.get("LOCI_TMUX_COMPANION_SESSIONS", "claude,copilot")
    expected = [s.strip() for s in expected_raw.split(",") if s.strip()]
    out["tmux_companion_required"] = tmux_required
    out["tmux_companion_expected"] = expected
    try:
        proc = subprocess.run(
            ["tmux", "ls"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        out["tmux_available"] = True
        listed = proc.stdout or ""
        running = []
        for ln in listed.splitlines():
            name = ln.split(":", 1)[0].strip()
            if name:
                running.append(name)
        running = sorted(set(running))
        missing = [s for s in expected if s not in running]
        out["tmux_companion_running"] = running
        if missing:
            out["tmux_companion_missing"] = missing
            if tmux_required:
                out["status"] = "unhealthy"
                out.setdefault("failures", [])
                out["failures"].append(
                    "tmux companions required but missing: " + ", ".join(missing)
                )
                out["remediation"] = (
                    "Start required tmux sessions or unset LOCI_TMUX_COMPANION_REQUIRED."
                )
            else:
                out["tmux_companion_optional_down"] = missing
    except FileNotFoundError:
        out["tmux_available"] = False
        out["tmux_companion_error"] = "tmux not installed"
        if tmux_required:
            out["status"] = "unhealthy"
            out.setdefault("failures", [])
            out["failures"].append("tmux required but not installed")
            out["remediation"] = (
                "Install tmux and start required sessions, or unset LOCI_TMUX_COMPANION_REQUIRED."
            )
    except Exception as exc:
        out["tmux_available"] = False
        out["tmux_companion_error"] = f"{type(exc).__name__}: {exc}"
        if tmux_required:
            out["status"] = "unhealthy"
            out.setdefault("failures", [])
            out["failures"].append("tmux companion probe failed while required")
            out["remediation"] = (
                "Resolve tmux probe failure or unset LOCI_TMUX_COMPANION_REQUIRED."
            )
    return json.dumps(out, indent=2)


# --------------------------------------------------------------------------- #
# rag_context_search internals — each stage independently fail-open, matching the tool's contract.
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
            # ts may be an ISO string (inv_store._now); created_at_ts is always an
            # int epoch. float() alone raised on every ISO fallback, so any point
            # carrying only ts — the corpus predating created_at_ts — escaped decay.
            try:
                ts_num = float(ts_val)
            except (TypeError, ValueError):
                ts_num = float(_coerce_ts(ts_val))
            if ts_num <= 0:
                continue
            try:
                age_days = (now_ts - ts_num) / 86400.0
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
        # Same budget as qdrant_ops._ce_rerank: 512 chars scored below using no reranker at all on a 1,048-char median finding.
        pairs = [(query, str(r.get('text', r.get('content', '')))[:_RERANK_MAX_CHARS])
                 for r in results[:20]]
        for r, sc in zip(results[:20], ce.predict(pairs)):
            r['final_ce_score'] = round(float(sc), 4)
        results[:20] = sorted(results[:20], key=lambda r: r.get('final_ce_score', -999), reverse=True)
    except Exception as exc:
        logger.debug('Final CE re-pass failed: %s', exc)


_SUPERSEDED_MARK = "[superseded: a later finding replaced this; do not rely on it] "


def _rag_mark_superseded(rows: list[dict], rfilter) -> None:
    """Tag superseded findings in place so they never read as current context."""
    for r in rows:
        if rfilter.is_superseded(r):
            r["resolution"] = "superseded"
            key = "text" if r.get("text") else "content"
            if not str(r.get(key) or "").startswith(_SUPERSEDED_MARK):
                r[key] = _SUPERSEDED_MARK + str(r.get(key) or "")


def _rag_record_access(results: list[dict], query: str) -> None:
    """Append a last_accessed marker for each returned finding to access.jsonl.

    The marker goes to the investigation's access log and never to
    findings.jsonl, where it would reuse the finding's id and shadow it.
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
            inv_path = _inv_dir(inv_id)
            if not (inv_path / "findings.jsonl").exists():
                continue
            _append_jsonl(inv_path / inv_store.ACCESS_LOG_NAME, {
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
    mode: Literal["normal", "compact"] = "normal",
    requesting_agent_id: Optional[str] = None,
) -> str:
    """
    Run hybrid RAG over Qdrant and return prompt-ready cited context.

    This tool always uses Qdrant: there is no keyword fallback, and missing
    Qdrant returns ``rag_required``. By default it searches
    ``QDRANT_COLLECTION_PREFIX`` (usually ``"loci_memory"``) plus
    ``CODE_CHUNKS_COLLECTION`` when configured, then merges results, reranks
    them with a cross-encoder, and assembles cited context.

    Important default note: the default collections come only from those two env
    vars. The docstring used to claim an ``agent_core_chunks`` default; it never
    existed. On this deployment that collection is a large DAMA telemetry lake,
    mostly GPS points and barely any code, so overriding into it is usually slow
    and irrelevant.

    Args:
        query: Natural language search query.
        limit: Results per collection (default 10).
        collections: Override the collection list. The default is
            ``[QDRANT_COLLECTION_PREFIX] + [CODE_CHUNKS_COLLECTION if set]``,
            not ``agent_core_chunks``.
        budget_chars: Max characters in assembled context (default 6000).
        exclude_types: Payload ``type`` values to exclude from
            ``agent_core_chunks``-style results. ``None`` defaults to
            ``['gps_trajectory']`` to suppress high-volume GPS pings. Pass
            ``[]`` to disable filtering.
        decay: If ``True`` (default), apply Ebbinghaus exponential time-decay
            rescoring to findings from the main investigation collection before
            ranking. Set ``False`` to keep raw similarity scores.
        expand_query: Expand retrieval with local-model paraphrases and keywords
            from ``query_expand`` before the cross-encoder repass. ``None``
            reads env ``LOCI_RAG_EXPAND`` (default on); ``True``/``False``
            force it. Fail-open: if the local generator is unavailable, the tool
            falls back to the original query. Enabled because judge evals showed
            ``+4% nDCG@10`` with no regression.
        mode: "normal" (default) for the legacy markdown block, or "compact" for
            deterministic cited one-liners plus slim source metadata.
        requesting_agent_id: Optional agent_id to narrow ACL visibility. Hits
            from an investigation whose ACL the caller cannot read are dropped
            and counted in ``excluded_acl``. The caller is the transport-bound
            identity (or this process); this argument can only narrow it.

    Returns:
        JSON ``{query, context, sources, total_chars, truncated, result_count,
        mode, collections_searched, qdrant_available, excluded_retracted,
        excluded_acl}``.
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

    # Gate: expand_query arg wins, else LOCI_RAG_EXPAND (default ON); fail-open to the original query.
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

    # Expansion only widens the candidate pool; the cross-encoder re-ranks against the ORIGINAL query.
    all_results.extend(_rag_search_collections(_collections, search_queries, limit, _agent_filter, errors))

    # memory_retract keeps the finding's Qdrant point: drop retracted hits by id
    # before ranking, access tracking and assembly; mark superseded ones.
    _rfilter = build_recall_filter(
        MEMORY_DIR,
        {str(r.get("investigation_id")) for r in all_results if r.get("investigation_id")},
        with_superseded=True,
    )
    all_results, _rag_excluded = _rfilter.split(all_results)
    # ACL: hits from investigations the caller may not read never reach the context.
    all_results, _rag_acl_excluded = inv_store._acl_filter_rows(all_results, requesting_agent_id)

    # Findings only — agent_core_chunks is static knowledge, not time-sensitive.
    if decay:
        _rag_apply_decay(all_results)

    # Merge-sort by score descending across all collections
    all_results.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)

    # Final cross-encoder re-pass for consistent cross-collection ranking
    _rag_cross_encode(all_results, query)

    # Best-effort: access-tracking failures must never block the response.
    _rag_record_access(all_results, query)
    _rag_mark_superseded(all_results, _rfilter)

    ctx = context_assemble(
        all_results,
        query,
        budget_chars=budget_chars,
        mode="compact" if mode == "compact" else "normal",
    )
    # Count distinct failed collections: an all-errors search must not read back as a genuine zero-hit search.
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
    ctx["excluded_retracted"] = len(_rag_excluded)
    ctx["excluded_acl"] = _rag_acl_excluded
    ctx["retraction_filter"] = _rfilter.status()
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
            "text": wrap_untrusted_memory_text(
                text[:300] if len(text) > 300 else text,
                origin="loci_memory",
                investigation_id=inv_id,
                finding_id=finding_id,
                source=source,
            ),
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
            docs_hits = []
            try:
                docs_response = json.loads(docs_search(context, investigation_id=investigation_id or "loci-docs-index", limit=max(1, top_k), include_excerpt=True))
                if isinstance(docs_response, dict) and docs_response.get("results"):
                    for item in docs_response["results"]:
                        docs_hits.append({
                            "finding_id": item.get("finding_id") or item.get("title") or "doc-guidance",
                            "text": str(item.get("summary") or item.get("title") or "").strip(),
                            "source": "docs_search",
                            "relevance_note": f"Related to: {context.strip().split()[:8]} (docs guidance)",
                            # docs_search is lexical and carries no similarity score; never invent one.
                            "score": item.get("score"),
                            "origin": "docs_search",
                            "investigation_id": investigation_id or "loci-docs-index",
                        })
            except Exception as _docs_exc:
                logger.debug("memory_surface docs recall failed (fail-open): %r", _docs_exc)
            if docs_hits:
                return json.dumps({
                    "surfaced": docs_hits,
                    "context_used": context[:200] if len(context) > 200 else context,
                    "count": len(docs_hits),
                    # Memory surfacing did not run; these are docs guidance only.
                    "degraded": True,
                    "fallback": "docs_search",
                    "reason": "qdrant unavailable",
                }, indent=2)
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

        # Retracted findings keep their Qdrant point: drop them by id here.
        _rfilter = build_recall_filter(
            MEMORY_DIR,
            {str(r.get("investigation_id")) for r in candidates if r.get("investigation_id")}
            | ({investigation_id} if investigation_id else set()),
            with_superseded=True,
        )
        candidates, _surface_excluded = _rfilter.split(candidates)
        # ACL: never surface findings from an investigation the caller cannot read.
        candidates, _surface_acl_excluded = inv_store._acl_filter_rows(candidates)

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
        for _row in surfaced:
            if _rfilter.is_superseded(_row):
                _row["resolution"] = "superseded"
                _row["text"] = _SUPERSEDED_MARK + _row["text"]

        docs_hits = []
        try:
            docs_response = json.loads(docs_search(context, investigation_id=investigation_id or "loci-docs-index", limit=max(1, top_k), include_excerpt=True))
            if isinstance(docs_response, dict) and docs_response.get("results"):
                for item in docs_response["results"]:
                    docs_hits.append({
                        "finding_id": item.get("finding_id") or item.get("title") or "doc-guidance",
                        "text": str(item.get("summary") or item.get("title") or "").strip(),
                        "source": "docs_search",
                        "relevance_note": f"Related to: {_ctx_prefix} (docs guidance)",
                        "score": item.get("score"),  # lexical hit: no similarity score to report
                        "origin": "docs_search",
                        "investigation_id": investigation_id or "loci-docs-index",
                    })
        except Exception as _docs_exc:
            logger.debug("memory_surface docs recall failed (fail-open): %r", _docs_exc)

        if docs_hits:
            # Docs guidance only fills slots the memory hits left free: an unscored lexical
            # hit must not outrank (and so displace) a finding with a real similarity score.
            surfaced = (surfaced + docs_hits)[:top_k]

        return json.dumps({
            "surfaced": surfaced,
            "context_used": context[:200] if len(context) > 200 else context,
            "count": len(surfaced),
            "excluded_retracted": len(_surface_excluded),
            "excluded_acl": _surface_acl_excluded,
            "retraction_filter": _rfilter.status(),
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
            try:
                manifest = _load_manifest(d.name)
            except Exception as exc:  # per-dir: a malformed dir (e.g. 'undefined') is skipped
                logger.debug("consolidate: skipping investigation dir %r: %r", d.name, exc)
                continue
            if manifest is None:
                continue
            ts = str(manifest.get("updated_at") or "")
            if ts > best_ts:
                best_ts = ts
                best_id = d.name
        if best_id is None:
            return None, None
        findings_path = MEMORY_DIR / best_id / "findings.jsonl"
        # Findings only, minus retracted ones: causal edges must not be inferred from them.
        retracted = build_recall_filter(MEMORY_DIR, [best_id], with_texts=False).all_retracted
        findings = [
            f for f in investigation_tools._only_findings(_read_jsonl(findings_path))
            if str(f.get("id", "")) not in retracted
        ]
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
            # derived_from also accepts free-text claims; only ids resolvable here become edges.
            if not source or source == target or source not in known:
                continue
            edges.append({
                "id": str(uuid.uuid4()),
                "source_id": source,
                "target_id": target,
                "edge_type": "caused_by",
                # Declared, not inferred — above the heuristic's 0.5.
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
    known_ids = {str(f.get("id")) for f in findings if f.get("id")}
    try:
        from memcheck import llm as _llm  # type: ignore
        if _llm.llm_available():
            numbered = "\n".join(
                f"{idx + 1}. [{f.get('id', '?')}] "
                f"{wrap_untrusted_memory_text(str(f.get('text', ''))[:300], investigation_id=investigation_id, finding_id=str(f.get('id') or ''), kind=str(f.get('record_type') or f.get('type') or 'finding'), source=str(f.get('source') or 'causal_infer'))}"
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
                    # The model sometimes returns the prompt ORDINAL instead of the id; a non-empty check let those dangling edges through.
                    if src not in known_ids or tgt not in known_ids:
                        logger.warning("causal LLM edge names unknown finding(s) "
                                       "%r -> %r; dropping", src[:40], tgt[:40])
                        continue
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


_consolidation_quality_audit_gen_fn = None
_CONSOLIDATION_QUALITY_AUDIT_SAMPLE_LIMIT = 5


def _load_mnemosyne_class():
    from mnemosyne.core.memory import Mnemosyne
    return Mnemosyne


def _row_value(row, key: str, index: int = 0):
    if row is None:
        return None
    if hasattr(row, "keys"):
        return row[key]
    return row[index]


def _snapshot_sleep_consolidation_rowid(m) -> int | None:
    """Best-effort baseline so the advisory audit can inspect only new summaries."""
    try:
        cursor = m.beam.conn.cursor()
        cursor.execute(
            "SELECT COALESCE(MAX(rowid), 0) AS max_rowid "
            "FROM episodic_memory WHERE source = ?",
            ("sleep_consolidation",),
        )
        row = cursor.fetchone()
        return int(_row_value(row, "max_rowid") or 0)
    except Exception as exc:
        logger.debug("_snapshot_sleep_consolidation_rowid failed (fail-open): %r", exc)
        return None


def _session_consolidated_id_map(result: dict) -> dict[str, set[str]]:
    session_map: dict[str, set[str]] = {}
    if not isinstance(result, dict):
        return session_map
    for row in result.get("session_results", []) or []:
        if not isinstance(row, dict):
            continue
        session_id = str(row.get("session_id") or "").strip()
        consolidated_ids = {
            str(mid).strip()
            for mid in (row.get("consolidated_ids") or [])
            if str(mid).strip()
        }
        if session_id and consolidated_ids:
            session_map[session_id] = consolidated_ids
    return session_map


def _fetch_consolidation_quality_samples(
    m,
    result: dict,
    baseline_rowid: int | None,
    limit: int = _CONSOLIDATION_QUALITY_AUDIT_SAMPLE_LIMIT,
) -> tuple[list[dict] | None, bool]:
    """Return bounded merge samples plus whether merge details looked unavailable."""
    if baseline_rowid is None:
        return None, True

    session_map = _session_consolidated_id_map(result)
    if not session_map:
        return [], False

    session_ids = list(session_map)
    placeholders = ",".join("?" * len(session_ids))
    scan_limit = max(limit * 6, limit)
    samples: list[dict] = []
    undetermined = False

    try:
        cursor = m.beam.conn.cursor()
        cursor.execute(
            f"""
            SELECT rowid, id, content, session_id, summary_of
            FROM episodic_memory
            WHERE source = ?
              AND rowid > ?
              AND session_id IN ({placeholders})
            ORDER BY rowid DESC
            LIMIT ?
            """,
            ("sleep_consolidation", baseline_rowid, *session_ids, scan_limit),
        )
        rows = cursor.fetchall()
        if not rows:
            return None, True

        for row in rows:
            session_id = str(_row_value(row, "session_id", 3) or "").strip()
            raw_summary_of = str(_row_value(row, "summary_of", 4) or "")
            source_ids = [part.strip() for part in raw_summary_of.split(",") if part.strip()]

            if not source_ids:
                undetermined = True
                continue
            if len(source_ids) < 2:
                continue
            if not set(source_ids).issubset(session_map.get(session_id, set())):
                undetermined = True
                continue

            src_placeholders = ",".join("?" * len(source_ids))
            cursor.execute(
                f"""
                SELECT id, content, source, timestamp
                FROM working_memory
                WHERE id IN ({src_placeholders})
                """,
                tuple(source_ids),
            )
            source_rows = cursor.fetchall()
            by_id = {
                str(_row_value(src, "id", 0)): {
                    "id": str(_row_value(src, "id", 0) or ""),
                    "content": str(_row_value(src, "content", 1) or ""),
                    "source": str(_row_value(src, "source", 2) or ""),
                    "timestamp": str(_row_value(src, "timestamp", 3) or ""),
                }
                for src in source_rows
            }
            if any(mid not in by_id or not by_id[mid]["content"].strip() for mid in source_ids):
                undetermined = True
                continue

            samples.append({
                "session_id": session_id,
                "merged_summary": str(_row_value(row, "content", 2) or ""),
                "source_entries": [by_id[mid] for mid in source_ids],
            })
            if len(samples) >= limit:
                break
    except Exception as exc:
        logger.debug("_fetch_consolidation_quality_samples failed (fail-open): %r", exc)
        return None, True

    return samples, undetermined


def _audit_summary_preview(text: str, limit: int = 240) -> str:
    return " ".join(str(text or "").split())[:limit]


def _audit_source_entry_ids(source_entries: list[dict], limit: int = 8) -> list[str]:
    ids: list[str] = []
    for entry in source_entries or []:
        if not isinstance(entry, dict):
            continue
        raw = str(entry.get("id") or "").strip()
        if not raw or raw in ids:
            continue
        ids.append(raw)
        if len(ids) >= max(1, int(limit)):
            break
    return ids


def _dedupe_quality_flags(flagged: list[dict]) -> list[dict]:
    """Deterministically de-duplicate near-identical advisory findings."""
    kept: list[dict] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    for row in flagged or []:
        if not isinstance(row, dict):
            continue
        session_id = str(row.get("session_id") or "")
        source_ids = tuple(str(x).strip() for x in (row.get("source_entry_ids") or []) if str(x).strip())
        concern = " ".join(str(row.get("concern") or "").lower().split())
        key = (session_id, source_ids, concern)
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def _run_consolidation_quality_audit(
    m,
    result: dict,
    baseline_rowid: int | None,
    *,
    gen_fn=None,
) -> dict | None:
    """Best-effort advisory audit over a bounded sample of just-created merges.

    Flagged entries include source/session provenance and are de-duplicated to
    reduce repeated noise across equivalent merge concerns.
    """
    if not isinstance(result, dict):
        return None
    if int(result.get("items_consolidated", 0) or 0) <= 0:
        return None

    try:
        import consolidation_quality_audit as _audit
    except Exception as exc:
        logger.debug("_run_consolidation_quality_audit import failed (fail-open): %r", exc)
        return {"sampled": 0, "flagged": [], "degraded": True}

    samples, undetermined = _fetch_consolidation_quality_samples(m, result, baseline_rowid)
    if samples is None:
        return {"sampled": 0, "flagged": [], "degraded": True}
    if not samples:
        return {"sampled": 0, "flagged": [], "degraded": bool(undetermined)}

    flagged: list[dict] = []
    sampled = 0
    degraded = bool(undetermined)
    for sample in samples:
        verdict = _audit.audit_merge_quality(
            sample.get("merged_summary", ""),
            sample.get("source_entries", []),
            gen_fn=gen_fn,
        )
        if not verdict.get("available"):
            degraded = True
            continue
        sampled += 1
        if verdict.get("verdict") == "lost_or_conflated":
            source_ids = _audit_source_entry_ids(sample.get("source_entries") or [])
            flagged.append({
                "summary": _audit_summary_preview(sample.get("merged_summary", "")),
                "concern": str(verdict.get("concern") or "possible information loss"),
                "session_id": str(sample.get("session_id") or ""),
                "source_entry_ids": source_ids,
                "source_entry_count": len(source_ids),
            })

    flagged = _dedupe_quality_flags(flagged)
    return {"sampled": sampled, "flagged": flagged, "degraded": degraded}


_SLEEP_CONSOLIDATION_BURST_WINDOW_SECONDS = max(
    300,
    int(os.environ.get("LOCI_SLEEP_BURST_WINDOW_SECONDS", "5400") or "5400"),
)
_SLEEP_CONSOLIDATION_PROVENANCE_REF_LIMIT = max(
    3,
    int(os.environ.get("LOCI_SLEEP_PROVENANCE_REF_LIMIT", "10") or "10"),
)


def _parse_iso_to_epoch(raw_ts: str | None) -> float | None:
    """Best-effort ISO timestamp parse, returning epoch seconds or None."""
    text = str(raw_ts or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return float(dt.timestamp())


def _sleep_like_burst_summary(
    investigation_id: str | None,
    findings: list[dict] | None,
    *,
    now_ts: float | None = None,
) -> dict:
    """Summarize whether consolidation follows an active recent reasoning burst."""
    now_epoch = float(now_ts if now_ts is not None else time.time())
    window_seconds = _SLEEP_CONSOLIDATION_BURST_WINDOW_SECONDS

    findings = findings or []
    type_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    reasoning_recent = 0
    findings_recent = 0
    burst_sources = {"investigation_reason", "swarm_reason", "reflection_loop_tick"}
    finding_refs: list[dict] = []

    for finding in findings:
        finding_id = str(finding.get("id") or "")
        record_type = str(finding.get("record_type") or finding.get("type") or "").lower()
        source = str(finding.get("source") or "").lower()
        ts_epoch = _parse_iso_to_epoch(str(finding.get("ts") or ""))
        if ts_epoch is None or now_epoch - ts_epoch > window_seconds:
            continue

        findings_recent += 1
        if record_type:
            type_counts[record_type] += 1
        if source:
            source_counts[source] += 1
        if record_type in {"inferred", "assumed", "procedure"} or source in burst_sources:
            reasoning_recent += 1
        finding_refs.append({
            "finding_id": finding_id,
            "record_type": record_type,
            "source": source,
            "ts": str(finding.get("ts") or ""),
            **provenance_fields(finding),
        })

    reflection_recent = False
    reflection_last_tick = None
    try:
        reflection_state = _load_reflection_state()
        raw_last_tick = reflection_state.get("last_tick")
        tick = raw_last_tick if isinstance(raw_last_tick, dict) else None
        if tick is not None:
            reflection_last_tick = {
                "ts": str(tick.get("ts") or ""),
                "processed_items": int(tick.get("processed_items") or 0),
                "findings_written": int(tick.get("findings_written") or 0),
            }
            inv = str(tick.get("investigation_id") or "").strip()
            if inv:
                reflection_last_tick["investigation_id"] = inv
        tick_epoch = _parse_iso_to_epoch(str((tick or {}).get("ts") or ""))
        if tick_epoch is not None and now_epoch - tick_epoch <= window_seconds:
            reflection_recent = bool(
                int((tick or {}).get("processed_items") or 0) > 0
                or int((tick or {}).get("findings_written") or 0) > 0
            )
    except Exception as exc:
        logger.debug("_sleep_like_burst_summary reflection read failed (fail-open): %r", exc)

    finding_refs = finding_refs[:_SLEEP_CONSOLIDATION_PROVENANCE_REF_LIMIT]
    provenance = {
        "aggregation_method": "recent_finding_window_with_provenance_fields",
        "refs_considered": findings_recent,
        "refs_emitted": len(finding_refs),
        "finding_refs": finding_refs,
        "coverage_ratio": (len(finding_refs) / findings_recent) if findings_recent > 0 else 0.0,
    }

    return {
        "window_seconds": window_seconds,
        "active_burst": bool(reasoning_recent > 0 or reflection_recent),
        "investigation_id": investigation_id,
        "signals": {
            "reasoning_findings_recent": reasoning_recent,
            "reflection_tick_recent": reflection_recent,
            "findings_recent_total": findings_recent,
        },
        "top_types": [{"type": kind, "count": count} for kind, count in type_counts.most_common(3)],
        "top_sources": [{"source": src, "count": count} for src, count in source_counts.most_common(3)],
        "provenance": provenance,
        "reflection_last_tick": reflection_last_tick,
    }


def _assert_sleep_like_consolidation_invariants(report: dict) -> None:
    """Fail-closed contract gate for sleep-like consolidation report invariants."""
    if not isinstance(report, dict):
        raise ValueError("sleep-like invariant failed: report must be a dict")
    if not isinstance(report.get("active_burst"), bool):
        raise ValueError("sleep-like invariant failed: active_burst must be bool")

    phases = report.get("phases")
    if not isinstance(phases, list) or len(phases) != 3:
        raise ValueError("sleep-like invariant failed: phases must contain summarize/replay/stabilize")

    expected_names = ("summarize", "replay", "stabilize")
    observed_names = tuple(str((phase or {}).get("name") or "") for phase in phases)
    if observed_names != expected_names:
        raise ValueError(f"sleep-like invariant failed: phase order mismatch {observed_names!r}")

    allowed_status = {
        "summarize": {"ok"},
        "replay": {"ok", "skipped"},
        "stabilize": {"ok", "skipped"},
    }
    for phase in phases:
        name = str((phase or {}).get("name") or "")
        status = str((phase or {}).get("status") or "")
        details = (phase or {}).get("details")
        if status not in allowed_status.get(name, set()):
            raise ValueError(f"sleep-like invariant failed: invalid status for {name}: {status!r}")
        if not isinstance(details, dict):
            raise ValueError(f"sleep-like invariant failed: {name} details must be dict")

    summarize_details = phases[0]["details"]
    provenance = summarize_details.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("sleep-like invariant failed: summarize.provenance missing")
    refs = provenance.get("finding_refs")
    if not isinstance(refs, list):
        raise ValueError("sleep-like invariant failed: summarize.provenance.finding_refs must be list")
    for ref in refs:
        if not isinstance(ref, dict):
            raise ValueError("sleep-like invariant failed: finding ref must be object")
        if "evidence_provenance_tier" not in ref or "provenance_defaulted" not in ref:
            raise ValueError("sleep-like invariant failed: finding ref missing provenance fields")


def _sleep_like_consolidation_report(
    *,
    investigation_id: str | None,
    findings: list[dict] | None,
    min_findings_for_causal: int,
    causal_edges_inferred: int,
    quality_audit: dict | None,
) -> dict:
    """Structured summarize/replay/stabilize report for offline consolidation."""
    burst_summary = _sleep_like_burst_summary(investigation_id, findings)
    findings_count = len(findings or [])
    replay_status = "ok" if investigation_id and findings_count >= int(min_findings_for_causal) else "skipped"

    if quality_audit is None:
        stabilize_status = "skipped"
        stabilize_details: dict[str, Any] = {"reason": "quality audit unavailable"}
    else:
        stabilize_status = "ok"
        stabilize_details = {
            "sampled": int(quality_audit.get("sampled", 0) or 0),
            "flagged_count": len(quality_audit.get("flagged") or []),
            "degraded": bool(quality_audit.get("degraded")),
        }

    report = {
        "active_burst": bool(burst_summary.get("active_burst")),
        "phases": [
            {"name": "summarize", "status": "ok", "details": burst_summary},
            {
                "name": "replay",
                "status": replay_status,
                "details": {
                    "investigation_id": investigation_id,
                    "findings_considered": findings_count,
                    "causal_edges_inferred": int(causal_edges_inferred or 0),
                    "min_findings_required": int(min_findings_for_causal),
                },
            },
            {"name": "stabilize", "status": stabilize_status, "details": stabilize_details},
        ],
    }
    _assert_sleep_like_consolidation_invariants(report)
    return report


@mcp.tool()
def memory_consolidate(dry_run: bool = False) -> str:
    """
    Run Mnemosyne sleep/consolidation cycle.
    Merges old working_memory entries into episodic memory, reducing DB size.
    Safe to run periodically (daily or after large investigation sessions).

    After the Mnemosyne pass, runs a causal inference slow path on the most
    recently active investigation (if it has enough findings), writing inferred
    edges to causal_edges.jsonl.  The causal step is fail-open and never
    blocks the consolidation result.

    Args:
        dry_run: If True, preview consolidation without executing.

    Returns JSON with consolidation stats and causal_edges_inferred count.
    On real (non-dry-run) consolidations, may also include an advisory-only
    consolidation_quality_audit field summarizing a bounded local-model spot-check
    of just-created multi-entry merges, plus a ``sleep_like_consolidation`` phase
    report (summarize/replay/stabilize) for offline post-burst consolidation.
    These never change what Mnemosyne writes.
    """
    import json as _json
    causal_edges_inferred = 0
    min_findings_for_causal = 3
    try:
        Mnemosyne = _load_mnemosyne_class()
        m = Mnemosyne()
        inv_id: str | None = None
        findings: list[dict] | None = None
        mod_state = load_state(MEMORY_DIR)
        try:
            mod_consolidation = consolidation_policy(
                mod_state,
                default_min_findings=3,
            )
            assert_consolidation_policy_invariants(mod_consolidation)
        except Exception as exc:
            logger.warning("memory_consolidate slow-neuromod invariant failed; fail-closed baseline threshold: %r", exc)
            mod_consolidation = {
                "consolidation_tone": 0.0,
                "min_findings_for_causal": 3,
            }
        min_findings_for_causal = int(mod_consolidation.get("min_findings_for_causal", 3))
        audit_baseline_rowid = None
        quality_audit: dict | None = None
        if not dry_run:
            audit_baseline_rowid = _snapshot_sleep_consolidation_rowid(m)
        result = m.sleep_all_sessions(dry_run=dry_run)
        _event_log_append({"op": "consolidate", "dry_run": dry_run,
                           "result_summary": str(result)[:200] if result else ""})

        # Causal inference slow path (fail-open).
        try:
            if not dry_run:
                inv_id, findings = _find_most_recent_investigation()
                if inv_id and findings and len(findings) >= int(min_findings_for_causal):
                    causal_edges_inferred = _run_causal_inference(inv_id, findings)
        except Exception as exc:
            # A feature that produces nothing should be able to say why.
            logger.warning("causal inference failed during consolidate "
                           "(fail-open, 0 edges): %r", exc)
            causal_edges_inferred = 0

        payload = {
            "status": "ok",
            "dry_run": dry_run,
            "result": result if isinstance(result, dict) else str(result),
            "causal_edges_inferred": causal_edges_inferred,
            "consolidation_aggregation": {
                "method": "deterministic_sleep_summary_plus_slow_threshold",
                "default_min_findings_for_causal": 3,
                "applied_min_findings_for_causal": int(min_findings_for_causal),
                "modulation": {
                    "consolidation_tone": round(float(mod_consolidation.get("consolidation_tone", 0.0) or 0.0), 3),
                    "provenance": "deterministic_derived",
                },
            },
        }
        if not dry_run:
            try:
                quality_audit = _run_consolidation_quality_audit(
                    m,
                    result,
                    audit_baseline_rowid,
                    gen_fn=_consolidation_quality_audit_gen_fn,
                )
                if quality_audit is not None:
                    payload["consolidation_quality_audit"] = quality_audit
            except Exception as exc:
                logger.debug("memory_consolidate advisory audit failed (fail-open): %r", exc)
                payload["consolidation_quality_audit"] = {
                    "sampled": 0,
                    "flagged": [],
                    "degraded": True,
                }
                quality_audit = payload["consolidation_quality_audit"]

            payload["sleep_like_consolidation"] = _sleep_like_consolidation_report(
                investigation_id=inv_id,
                findings=findings,
                min_findings_for_causal=min_findings_for_causal,
                causal_edges_inferred=causal_edges_inferred,
                quality_audit=quality_audit,
            )
            try:
                result_items = int((result or {}).get("items_consolidated", 0) or 0)
                consolidation_signal = 0.0
                if result_items > 0:
                    consolidation_signal += min(0.35, 0.05 * result_items)
                if causal_edges_inferred > 0:
                    consolidation_signal += 0.1
                observe(
                    MEMORY_DIR,
                    event="memory_consolidate",
                    consolidation_signal=max(-1.0, min(1.0, consolidation_signal)),
                )
            except Exception as exc:
                logger.debug("memory_consolidate slow-neuromod update failed (fail-open): %r", exc)

        return _json.dumps(payload)
    except Exception as e:
        try:
            if not dry_run:
                observe(
                    MEMORY_DIR,
                    event="memory_consolidate_error",
                    consolidation_signal=-0.6,
                )
        except Exception:
            pass
        return _json.dumps({
            "status": "error",
            "error": "internal error",
            "type": type(e).__name__,
            "causal_edges_inferred": causal_edges_inferred,
        })


@mcp.tool()
def causal_infer(investigation_id: str, limit: int = 200) -> str:
    """
    Infer causal edges for an investigation and write them to causal_edges.jsonl.

    Producing edges is now something you can do, rather than a side effect you
    might trigger. Previously the only caller was memory_consolidate, behind
    `not dry_run` and `len(findings) >= 3`, on whichever investigation happened
    to be most recent — so causal_edges_list answered "nothing is known" and
    "this never ran" with the same empty shape.

    Two lanes run and are merged, declared winning any collision:
      - declared lineage from derived_from links (author-stated, confidence 0.9)
      - text/keyword heuristic, or the LLM slow path when a model is reachable

    Args:
        investigation_id: Investigation to infer over.
        limit: Max findings to consider, newest first (default 200).

    Returns JSON: {investigation_id, findings_considered, edges_written,
                   status} — or {error} if the investigation does not exist.
    """
    # _load_manifest, not _inv_dir: _inv_dir mkdirs, so a typo'd id would silently create an empty investigation.
    if not _load_manifest(investigation_id):
        return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

    findings = _read_jsonl(_inv_dir(investigation_id) / "findings.jsonl")
    retracted = _load_retracted_ids(investigation_id)
    if retracted:
        findings = [f for f in findings if str(f.get("id", "")) not in retracted]
    findings = [f for f in findings if str(f.get("text", "") or "").strip()]
    if limit and limit > 0:
        findings = findings[-int(limit):]

    if len(findings) < 2:
        return json.dumps({
            "investigation_id": investigation_id,
            "findings_considered": len(findings),
            "edges_written": 0,
            "status": "too_few_findings",
            "detail": "causal inference needs at least two findings to relate.",
        }, indent=2)

    written = _run_causal_inference(investigation_id, findings)
    return json.dumps({
        "investigation_id": investigation_id,
        "findings_considered": len(findings),
        "edges_written": written,
        "status": "ok",
    }, indent=2)


@mcp.tool()
def causal_edges_list(investigation_id: str) -> str:
    """
    List causal edges inferred for an investigation.

    Edges are written by causal_infer, or by the slow path in memory_consolidate.
    An empty list means no edges have been INFERRED yet — run causal_infer.
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
        # An edge touching a retracted finding is dropped (and counted), not served.
        _retracted = build_recall_filter(MEMORY_DIR, [investigation_id], with_texts=False).all_retracted
        raw = [e for e in _read_jsonl(edges_path) if isinstance(e, dict)]
        _kept = [
            e for e in raw
            if str(e.get("source_id") or "") not in _retracted
            and str(e.get("target_id") or "") not in _retracted
        ]
        _excluded_edges = len(raw) - len(_kept)
        raw = _kept
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
        return json.dumps({"edges": edges, "count": len(edges), "excluded_retracted_edges": _excluded_edges})
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
        # Not "no_trace": an empty list from a broken search is a different claim from an empty one from a healthy search.
        logger.warning("memory_confidence search failed: %r", exc)
        return [], "search_failed"

    return results, None


def _confidence_evidence_ref(r) -> dict:
    """One evidence_refs row for memory_confidence from a Qdrant search result.

    Results are ScoredPoint-shaped (``.id``/``.score``/``.payload``), exactly as
    _confidence_cues reads them -- not dicts. Plain dicts are still accepted so a
    caller that hands in pre-flattened rows keeps working.
    """
    if isinstance(r, dict):
        pl, point_id, score = r, r.get("id"), r.get("score")
    else:
        pl = dict(getattr(r, "payload", None) or {})
        point_id, score = getattr(r, "id", None), getattr(r, "score", None)
    return {
        "finding_id": str(pl.get("finding_id") or pl.get("id") or point_id or ""),
        "source": str(pl.get("source") or ""),
        "investigation_id": str(pl.get("investigation_id") or ""),
        "score": round(_safe_float(score, 0.0), 4),
    }


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

    # Weights follow Fleming 2010: source/recollection > familiarity/fluency.
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


_CONFIDENCE_ENTAILMENT_VALID_VERDICTS = ("confirmed", "refuted", "uncertain")
_CONFIDENCE_ENTAILMENT_PROMPT_TMPL = (
    "You are checking whether retrieved memory evidence REALLY supports an EXACT claim.\n"
    "Judge only from the evidence shown. Consider subject identity, scope, time, modality,\n"
    "uncertainty, and negation. Evidence about a related topic, weaker possibility, or\n"
    "different subject does NOT confirm the claim.\n\n"
    "Return ONLY a JSON object of this exact shape, with no prose outside it:\n"
    '{{"verdict": "confirmed|refuted|uncertain", "rationale": "brief why", "confidence": 0.0}}\n\n'
    'Use "confirmed" only when the evidence itself supports the exact claim.\n'
    'Use "refuted" when the evidence points the other way or clearly mismatches scope.\n'
    'Use "uncertain" when the evidence is relevant but insufficient or ambiguous.\n\n'
    "CLAIM:\n{claim}\n\n"
    "TOP MEMORY HIT:\n{evidence}\n"
)


def _confidence_llm_entailment(
    query: str,
    top_text: str,
    *,
    gen_fn: Optional[Callable[..., dict]] = None,
) -> dict:
    """Advisory exact-claim support check for memory_confidence. Never raises."""
    claim = (query or "").strip()
    evidence = (top_text or "").strip()
    unavailable = {
        "available": False,
        "verdict": None,
        "rationale": "",
        "confidence": 0.0,
        "degraded": True,
        "error": "",
    }
    if not claim or not evidence:
        return dict(unavailable)

    if gen_fn is None:
        try:
            import llm_local
            gen_fn = llm_local.generate
        except Exception as exc:
            out = dict(unavailable)
            out["error"] = f"llm_local import failed: {exc}"[:200]
            return out

    try:
        import backends
        model = backends.ollama_verify_model()
    except Exception:
        model = ""

    prompt = _CONFIDENCE_ENTAILMENT_PROMPT_TMPL.format(claim=claim, evidence=evidence[:1200])

    try:
        result = gen_fn(
            prompt,
            model=model,
            fmt="json",
            max_tokens=220,
            temperature=0.0,
        )
    except Exception as exc:
        out = dict(unavailable)
        out["error"] = f"generate() raised: {exc}"[:200]
        return out

    if not isinstance(result, dict) or not result.get("ok"):
        out = dict(unavailable)
        out["rationale"] = str((result or {}).get("text", "") or "") if isinstance(result, dict) else ""
        out["error"] = (
            str((result or {}).get("why", "") or "model unavailable")[:200]
            if isinstance(result, dict)
            else "model unavailable"
        )
        return out

    obj = extract_json_object(str(result.get("text", "")) or "")
    if obj is None:
        out = dict(unavailable)
        out["error"] = f"unparseable response: {str(result.get('text', ''))[:120]!r}"
        return out

    verdict = str(obj.get("verdict", "") or "").strip().lower()
    if verdict not in _CONFIDENCE_ENTAILMENT_VALID_VERDICTS:
        verdict = "uncertain"
    rationale = obj.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        rationale = obj.get("reasoning")
    if not isinstance(rationale, str) or not rationale.strip():
        rationale = obj.get("refutation")
    if not isinstance(rationale, str):
        rationale = "" if rationale is None else str(rationale)
    try:
        llm_confidence = float(obj.get("confidence"))
        if llm_confidence != llm_confidence:
            raise ValueError("nan")
    except (TypeError, ValueError):
        llm_confidence = 0.0

    return {
        "available": True,
        "verdict": verdict,
        "rationale": rationale.strip(),
        "confidence": max(0.0, min(1.0, llm_confidence)),
        "degraded": False,
        "error": "",
    }


@mcp.tool()
def memory_confidence(
    query: str,
    top_k: int = 8,
) -> str:
    """
    Estimate how reliably loci_memory knows about a topic (metamemory).

    Combines five evidence cues into a calibrated confidence score:
      fluency        — cosine similarity of top hit to query (retrieval ease)
      accessibility  — mean score of top-4 hits (amount of partial info recalled)
      source_div     — number of distinct sources/investigations in top results
      corroboration  — max occurrences across top hits (repeated evidence)
      trust          — mean confidence tier (high/medium/low) of top hits

    A slow cross-session neuromodulation tone can apply a small bounded bias
    to the final numeric confidence after cue aggregation.

    It also MAY include ``llm_entailment_note``: an advisory local-model verdict
    on whether the top memory hit actually supports the exact claim/query,
    considering subject, scope, negation, and modality. This is additive only:
    it NEVER changes the formula-based ``confidence``, ``basis``, or
    ``recommendation`` fields in the default call path.

    Fluency is down-weighted relative to source_div and trust because it tracks
    retrieval ease, not correctness (Koriat 1993 over-confidence mechanism).

    Use this before asserting a memory-derived claim to get a calibrated estimate
    of reliability. Low confidence → verify with investigation tools first.

    Args:
        query: The claim or topic to estimate confidence for.
        top_k: Number of results to base the estimate on (default 8).

    Returns:
        JSON with {confidence, basis, cues, top_hit_preview, recommendation,
        llm_entailment_note?}. When the model is unavailable/errors, the advisory
        field is returned degraded or omitted; the numeric verdict is unchanged.
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
    base_confidence = float(confidence)
    try:
        _confidence_bias = confidence_policy(load_state(MEMORY_DIR), confidence=base_confidence)
        assert_confidence_policy_invariants(_confidence_bias)
    except Exception as exc:
        logger.warning("memory_confidence slow-neuromod invariant failed; fail-closed base confidence: %r", exc)
        _confidence_bias = {
            "confidence_tone": 0.0,
            "delta": 0.0,
            "confidence": base_confidence,
        }
    confidence = float(_confidence_bias.get("confidence", base_confidence))
    llm_entailment = _confidence_llm_entailment(query, top_text) if top_text else None

    payload = {
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
        "confidence_aggregation": {
            "method": "cue_verdict_plus_slow_modulation",
            "base_confidence": round(base_confidence, 3),
            "adjusted_confidence": round(confidence, 3),
            "modulation": {
                "confidence_tone": round(float(_confidence_bias.get("confidence_tone", 0.0) or 0.0), 3),
                "delta": round(float(_confidence_bias.get("delta", 0.0) or 0.0), 3),
                "provenance": "deterministic_derived",
            },
            "evidence_refs": [_confidence_evidence_ref(r) for r in results[:5]],
        },
    }
    if isinstance(llm_entailment, dict):
        payload["llm_entailment_note"] = {
            "available": bool(llm_entailment.get("available")),
            "verdict": llm_entailment.get("verdict"),
            "rationale": llm_entailment.get("rationale", ""),
            "confidence": round(float(llm_entailment.get("confidence", 0.0) or 0.0), 3),
            "degraded": bool(llm_entailment.get("degraded")),
            "error": llm_entailment.get("error", ""),
        }

    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Memory tier management helpers
# ---------------------------------------------------------------------------

def _change_finding_tier(investigation_id: str, finding_id: str, new_tier: str) -> dict:
    """
    Core logic for memory_promote / memory_demote.

    Rewrites findings.jsonl atomically, updating the tier field of the target
    finding. Returns a dict with {finding_id, old_tier, new_tier, ok} or {error}.
    Qdrant failures never undo the JSONL change, but they do set ok:false:
    ``ok`` means the index matches the recorded tier.
    """
    if new_tier not in {"hot", "warm", "cold"}:
        return {"error": "tier must be one of: hot, warm, cold"}

    findings_path = _inv_dir(investigation_id) / "findings.jsonl"

    # Atomically rewrite the JSONL file
    _lock_path = _inv_dir(investigation_id) / ".lock"
    try:
        with _locked_file(_lock_path, "a+", exclusive=True):
            findings = _read_jsonl(findings_path)
            target = None
            for f in findings:
                if str(f.get("id", "")) == finding_id:
                    target = f
                    break
            if target is None:
                return {"error": f"Finding '{finding_id}' not found in investigation '{investigation_id}'."}

            old_tier = target.get("tier", "warm")
            text = str(target.get("text", "") or "")
            # Same tier: nothing to rewrite, but the index step below still runs.
            # A promote whose upsert failed reports ok:false and must be
            # retryable; returning ok:true here would skip the upsert on retry.
            if old_tier != new_tier:
                # Line-preserving rewrite: a torn or unparseable line stays as it is.
                inv_store._rewrite_jsonl_preserving(
                    findings_path,
                    lambda f: {**f, "tier": new_tier} if str(f.get("id", "")) == finding_id else None,
                )

            # If promoting to hot, update manifest notes before releasing the lock.
            if new_tier == "hot" and old_tier != "hot":
                manifest = _load_manifest_fresh(investigation_id)
                if manifest is None:
                    raise FileNotFoundError(f"Investigation '{investigation_id}' not found.")
                snippet = text[:200]
                notes = manifest.get("notes") or ""
                manifest["notes"] = (notes + "; " + snippet) if notes else snippet
                _save_manifest(manifest)
    except StoreBusyError as exc:
        return _busy_payload(exc, investigation_id=investigation_id, finding_id=finding_id)
    except Exception as exc:
        return {"error": f"Failed to rewrite findings.jsonl: {exc}"}

    # Handle Qdrant changes based on tier transition. hot and warm are documented
    # as "Qdrant indexed", so a tier change into them succeeds only when the
    # upsert actually landed; _qdrant_upsert returns False on a swallowed failure
    # (no client, embed failure, upsert exception).
    result = {"finding_id": finding_id, "old_tier": old_tier, "new_tier": new_tier, "ok": True}
    try:
        if new_tier == "cold":
            # Remove from Qdrant vector index
            client, col = _get_qdrant()
            if client is None:
                # Unknown, not "removed": the point may still be in the index.
                result["qdrant_removed"] = None
                result["degraded"] = True
            else:
                try:
                    from qdrant_client.models import PointIdsList
                    client.delete(col, points_selector=PointIdsList(points=[finding_id]))
                    result["qdrant_removed"] = True
                except Exception as exc:
                    logger.warning("Qdrant delete failed (demote to cold) — JSONL updated: %s", exc)
                    result.update(ok=False, qdrant_removed=False, degraded=True,
                                  error=f"tier recorded as cold but the Qdrant point was not removed: {exc}")
        elif old_tier == "hot":
            # hot -> hot/warm: the point is already indexed; nothing to re-assert.
            pass
        else:
            # cold -> warm/hot re-indexes; warm -> hot and a same-tier retry
            # re-assert the point (upsert is idempotent).
            indexed = bool(_qdrant_upsert(finding_id, text, target))
            result["qdrant_indexed"] = indexed
            if not indexed:
                result.update(
                    ok=False, degraded=True, retryable=True,
                    error=(f"tier recorded as {new_tier} but the finding was not indexed in "
                           "Qdrant (unavailable, embedding failed or upsert failed); "
                           "retry memory_promote once Qdrant is reachable"),
                )
    except Exception as exc:
        logger.warning("Qdrant tier-change operation failed: %s", exc)
        result.update(ok=False, degraded=True, error=f"Qdrant tier-change operation failed: {exc}")

    return result


# ---------------------------------------------------------------------------
# Tool: loci_validated_knowledge_promotion
# ---------------------------------------------------------------------------


def _finding_text_key(text: str) -> str:
    """Normalize a finding string so repeated claims can be detected deterministically."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


@mcp.tool()
def loci_validated_knowledge_promotion(
    investigation_id: str,
    finding_id: str,
    target_tier: str = "warm",
    *,
    require_repeated_verification: bool = False,
) -> str:
    """
    Enforce the verification gate before promoting a finding into trusted memory.

    The promotion gate is intentionally conservative and reuses the repo's
    existing investigation + provenance + verification code paths:
      - find the target finding in the investigation
      - reject resolved / retracted / empty findings up front
      - verify the claim with verify_finding(), which applies the provenance firewall
      - only permit promotion when verification is confirmed and the evidence passes
        the model_asserted guard
      - upgrade repeated verified claims to hot memory when they are seen more than once
        or when the caller opts into repeated verification as a hard gate

    This helper preserves the repo's fail-open/fail-closed contract:
      - backend/model failures become ``uncertain`` and therefore block promotion
      - model_asserted findings must be supported by independent evidence before a
        verified promotion is allowed
    """
    manifest = _load_manifest(investigation_id)
    if not manifest:
        return json.dumps({"status": "blocked", "error": f"Investigation '{investigation_id}' not found."})

    if not investigation_id or not finding_id:
        return json.dumps({"status": "blocked", "error": "investigation_id and finding_id are required."})

    findings = inv_store._fold_provenance_overrides(
        _read_jsonl(_inv_dir(investigation_id) / "findings.jsonl"), investigation_id)
    finding = next((f for f in findings if str(f.get("id") or "") == str(finding_id)), None)
    if not finding:
        return json.dumps({"status": "blocked", "error": f"Finding '{finding_id}' not found in investigation '{investigation_id}'."})

    if str(finding.get("resolution") or "open").lower() not in {"open", ""}:
        return json.dumps({
            "status": "blocked",
            "reason": "resolved_finding",
            "resolution": str(finding.get("resolution") or "open"),
            "finding_id": finding_id,
        })

    if str(finding.get("text") or "").strip() == "":
        return json.dumps({"status": "blocked", "reason": "empty_finding", "finding_id": finding_id})

    retracted = _load_retracted_ids(investigation_id)
    if str(finding_id) in retracted:
        return json.dumps({"status": "blocked", "reason": "retracted_finding", "finding_id": finding_id})

    text = str(finding.get("text") or "")
    # Same reader as verify_all: honours metadata.provenance_tier / evidence_kind aliases;
    # an untagged (defaulted) finding is gated like model_asserted.
    evidence_tier = firewall_candidate_tier(finding)
    import verify as _v
    verify_result = _v.verify_finding(
        text,
        investigation_id=investigation_id,
        finding_id=finding_id,
        candidate_provenance_tier=evidence_tier,
        evidence_rows=_firewall_linked_evidence(investigation_id, finding),
    )

    # verify_finding attaches provenance_firewall only when it blocked; absent means it passed.
    firewall = verify_result.get("provenance_firewall")
    if isinstance(firewall, dict) and not firewall.get("allowed", True):
        return json.dumps({
            "status": "blocked",
            "reason": "provenance_firewall",
            "finding_id": finding_id,
            "verification": verify_result,
            "promotion": None,
        })

    if verify_result.get("verdict") != "confirmed":
        return json.dumps({
            "status": "blocked",
            "reason": "verification_gate_failed",
            "finding_id": finding_id,
            "verification": verify_result,
            "promotion": None,
        })

    normalized_text = _finding_text_key(text)
    repeat_count = 0
    for other in findings:
        if str(other.get("id") or "") == str(finding_id):
            continue
        if str(other.get("resolution") or "open").lower() not in {"open", ""}:
            continue
        other_text = str(other.get("text") or "")
        if _finding_text_key(other_text) == normalized_text:
            repeat_count += 1

    if require_repeated_verification and repeat_count == 0:
        return json.dumps({
            "status": "blocked",
            "reason": "missing_repeat_verification",
            "finding_id": finding_id,
            "repeat_count": 0,
            "verification": verify_result,
            "promotion": None,
        })

    promoted_tier = "hot" if repeat_count > 0 else str(target_tier or "warm")
    if promoted_tier not in {"hot", "warm", "cold"}:
        promoted_tier = "warm"

    promotion = json.loads(memory_promote(investigation_id, finding_id, promoted_tier))
    return json.dumps({
        "status": "promoted" if promotion.get("ok") else "blocked",
        "reason": ("promotion_failed" if not promotion.get("ok")
                   else "verified_repeat_promoted" if repeat_count > 0 else "verified_promoted"),
        "finding_id": finding_id,
        "target_tier": promoted_tier,
        "repeat_count": repeat_count,
        "verification": verify_result,
        "promotion": promotion,
    }, indent=2)


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
        JSON: {finding_id, old_tier, new_tier, ok, qdrant_indexed}
        ``ok`` is true only when the finding reached the Qdrant index. When the
        upsert fails (Qdrant down, embedding failed) the tier is still recorded
        but the reply is ``ok:false, degraded:true, retryable:true``; calling
        memory_promote again with the same tier retries the index write.
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

# Width clips from the front: the first N are the most load-bearing.
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
    """Reason over an investigation with grounded, multi-perspective analysis.

    This is the in-server counterpart to the deep_think_loci workflow. It runs
    an on-topic grounding gate over findings, then fans out to N adversarial
    perspectives plus one synthesis step that extracts converged and contested
    claims. It makes ``N+1`` inline LLM calls, so it is for explicit "reason
    now" use, not hot paths.

    Requires an LLM endpoint (Ollama by default; Anthropic/Copilot when keys are
    set). Fail-soft: returns an ``error`` field instead of raising if the LLM is
    unreachable.

    Args:
        investigation_id: Investigation whose findings ground the reasoning.
        question:         The question/problem to reason about.
        perspectives: Number of adversarial perspectives (1-5). Default 3.
        ground_threshold: Per-finding cosine threshold for the grounding gate.
            Deep-think convention is ``0.59``; lower-scoring findings are
            dropped as off-topic before any model reasons.
        persist: If ``True``, store each converged claim as an ``inferred``
            finding with ``source=investigation_reason``. Default ``False``.

    Returns:
        JSON ``{investigation_id, question, perspectives_used,
        grounded_findings, gate_applied, confidence_score, converged_claims,
        contested_areas, final_answer, persisted_finding_ids}``.
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
        f"- [{f.get('type', f.get('record_type', '?'))}] "
        f"{wrap_untrusted_memory_text(str(f['text'])[:300], investigation_id=investigation_id, finding_id=str(f.get('id') or ''), kind=str(f.get('type', f.get('record_type', 'finding'))), source=str(f.get('source') or 'investigation_reason'))}"
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
                    evidence_provenance_tier=MODEL_ASSERTED,
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
            "text": wrap_untrusted_memory_text(
                h.get("text", ""),
                origin="loci_memory",
                investigation_id=investigation_id,
                finding_id=h.get("finding_id", ""),
                kind=h.get("record_type", "observed"),
                source=h.get("source", ""),
            ),
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
    mode: Literal["normal", "compact"] = "normal",
) -> str:
    """
    Return recent findings for an investigation as lightweight hints.

    This is suited for polling after ``investigation_store`` when callers want
    "what changed recently" without reloading full context. Each hint carries
    ``recency_score = 1.0 / (1 + age_hours)`` so callers can rank freshness.

    The in-process session ring buffer is preferred for speed; if empty, the
    tool falls back to ``findings.jsonl`` (for example after a server restart).

    Args:
        investigation_id: Investigation identifier.
        limit: Maximum number of hints to return (default 3, max 20).
        since_ts: Optional ISO-8601 timestamp. When provided, only findings
            with ``ts > since_ts`` are returned. Use the previous response's
            ``as_of`` as the next ``since_ts`` for incremental polling.
        mode: "normal" (default) for the legacy payload, or "compact" to clip
            each hint's text while preserving all other hint fields.

    Returns:
        JSON ``{investigation_id, hints:[{finding_id, text, source,
        record_type, recency_score, ts}], count, as_of}``, or ``{"error": ...}``.
    """
    try:
        manifest = _load_manifest(investigation_id)
        if not manifest:
            return json.dumps({"error": f"Investigation '{investigation_id}' not found."})
        limit = max(1, min(int(limit or 3), 20))
        payload = _compute_hints(investigation_id, limit, since_ts)
        if mode == "compact":
            payload = dict(payload)
            payload["hints"] = [compact_finding_row(h) for h in payload.get("hints", [])]
        return json.dumps(payload, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.debug("memory_hints error: %r", exc)
        return json.dumps({"error": str(exc)})


# ---- MCP resource: memory://hints/{investigation_id} ----
# Push updates are transport-dependent; the resource is readable over all transports.

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


def _route_word_overlap(hit_a: dict, hit_b: dict) -> Optional[float]:
    """Word-set Jaccard of two hits' text; None when both texts are empty."""
    words_a = set(str(hit_a.get("text", "")).lower().split())
    words_b = set(str(hit_b.get("text", "")).lower().split())
    union = words_a | words_b
    if not union:
        return None
    return len(words_a & words_b) / max(len(union), 1)


def _route_dedup_by_overlap(
    hits: list[dict],
    threshold: float = 0.80,
    overlap_fn: Optional[Callable[[dict, dict], Optional[float]]] = None,
) -> list[dict]:
    """
    Deduplicate memory_route hits by word-overlap (>threshold overlap → keep
    the highest-scoring hit). Assumes `hits` is already sorted by score
    (descending), so among an overlapping pair the later one in iteration
    order is always the lower-scoring one to suppress.

    ``overlap_fn`` replaces the text comparison. Replay of an audited trace that
    carries precomputed overlaps instead of text uses it (see
    ``_route_trace_overlap_fn``).
    """
    overlap_of = overlap_fn or _route_word_overlap
    kept = []
    suppressed = set()
    for i, hit_a in enumerate(hits):
        if i in suppressed:
            continue
        for j, hit_b in enumerate(hits):
            if j <= i or j in suppressed:
                continue
            overlap = overlap_of(hit_a, hit_b)
            if overlap is None:
                continue
            if overlap > threshold:
                # Suppress the lower-scoring one (raw_hits already sorted by score)
                suppressed.add(j)
        kept.append(hit_a)
    return kept


def _route_policy_trace_hit(hit: dict) -> dict:
    """Compact replay-safe snapshot for route simulation and audit tracing."""
    return {
        "finding_id": hit.get("finding_id") or hit.get("id", ""),
        "id": hit.get("id") or hit.get("finding_id", ""),
        "investigation_id": hit.get("investigation_id", ""),
        "text": hit.get("text", ""),
        "source": hit.get("source", ""),
        "authored_by": hit.get("authored_by", ""),
        "score": _safe_float(hit.get("score"), 0.0),
        "tier": hit.get("tier") or hit.get("record_type", "finding"),
    }


def _homeostatic_drive_value(value: object, *, name: str) -> float:
    try:
        numeric = float(value)
    except Exception as exc:
        raise ValueError(f"drive '{name}' must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"drive '{name}' must be finite")
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"drive '{name}' must be between 0.0 and 1.0")
    return numeric


def _derive_homeostatic_drives(query: str, drive_state: Optional[dict]) -> dict:
    if drive_state is not None and not isinstance(drive_state, dict):
        raise ValueError("drive_state must be an object when provided")
    drives = {"hunger": 0.0, "fatigue": 0.0, "urgency": 0.0}
    if isinstance(drive_state, dict):
        for key in tuple(drives.keys()):
            if key in drive_state:
                drives[key] = _homeostatic_drive_value(drive_state[key], name=key)

    q = str(query or "").lower()
    if q:
        hunger_hits = (
            "explore",
            "unknown",
            "novel",
            "hypothesis",
            "alternative",
            "broaden",
        )
        fatigue_hits = ("quick", "triage", "summarize", "brief", "tldr")
        urgency_hits = ("urgent", "critical", "sev1", "incident", "asap", "now")

        drives["hunger"] = max(drives["hunger"], 0.35 if any(t in q for t in hunger_hits) else 0.0)
        drives["fatigue"] = max(drives["fatigue"], 0.30 if any(t in q for t in fatigue_hits) else 0.0)
        drives["urgency"] = max(drives["urgency"], 0.45 if any(t in q for t in urgency_hits) else 0.0)
    return drives


def _homeostatic_route_policy(drives: dict, top_k: int) -> dict:
    hunger = _homeostatic_drive_value(drives.get("hunger", 0.0), name="hunger")
    fatigue = _homeostatic_drive_value(drives.get("fatigue", 0.0), name="fatigue")
    urgency = _homeostatic_drive_value(drives.get("urgency", 0.0), name="urgency")

    candidate_multiplier = max(1.5, min(5.0, 3.0 + (1.5 * hunger) - (1.0 * fatigue) - (0.8 * urgency)))
    priority_ratio = max(0.40, min(1.0, 1.0 - (0.45 * hunger) + (0.30 * fatigue) + (0.35 * urgency)))
    cap = max(0, int(top_k))
    candidate_limit = max(cap, int(round(max(1, cap) * candidate_multiplier)))
    priority_slots = max(0, min(cap, int(math.floor(cap * priority_ratio))))
    exploration_slots = max(0, cap - priority_slots)
    return {
        "drives": {"hunger": hunger, "fatigue": fatigue, "urgency": urgency},
        "candidate_multiplier": candidate_multiplier,
        "candidate_limit": candidate_limit,
        "priority_ratio": priority_ratio,
        "priority_slots": priority_slots,
        "exploration_slots": exploration_slots,
    }


def _assert_route_policy_invariants(policy: dict, *, top_k: int) -> None:
    required = ("candidate_limit", "priority_slots", "exploration_slots", "candidate_multiplier", "priority_ratio")
    for key in required:
        if key not in policy:
            raise ValueError(f"routing policy invariant failed: missing {key}")
    cap = max(0, int(top_k))
    candidate_limit = int(policy["candidate_limit"])
    priority_slots = int(policy["priority_slots"])
    exploration_slots = int(policy["exploration_slots"])
    if candidate_limit < cap:
        raise ValueError("routing policy invariant failed: candidate_limit below top_k")
    if priority_slots < 0 or priority_slots > cap:
        raise ValueError("routing policy invariant failed: priority_slots out of range")
    if exploration_slots < 0:
        raise ValueError("routing policy invariant failed: exploration_slots negative")
    if priority_slots + exploration_slots != cap:
        raise ValueError("routing policy invariant failed: slot partition mismatch")
    candidate_multiplier = float(policy["candidate_multiplier"])
    if not math.isfinite(candidate_multiplier):
        raise ValueError("routing policy invariant failed: candidate_multiplier not finite")
    priority_ratio = float(policy["priority_ratio"])
    if not math.isfinite(priority_ratio) or priority_ratio < 0.0 or priority_ratio > 1.0:
        raise ValueError("routing policy invariant failed: priority_ratio out of range")


def _route_aggregation_with_provenance(selected_hits: list[dict], candidate_hits: list[dict]) -> dict:
    selected_ids: set[str] = set()
    selected_refs: list[dict] = []
    source_counts: dict[str, int] = {}
    investigation_counts: dict[str, int] = {}

    for rank, hit in enumerate(selected_hits, start=1):
        finding_id = str(hit.get("finding_id") or hit.get("id") or "")
        source = str(hit.get("source") or "unknown")
        investigation_id = str(hit.get("investigation_id") or "")
        score = _safe_float(hit.get("score"), 0.0)
        if finding_id:
            selected_ids.add(finding_id)
        selected_refs.append({
            "rank": rank,
            "finding_id": finding_id,
            "source": source,
            "investigation_id": investigation_id,
            "score": score,
        })
        source_counts[source] = source_counts.get(source, 0) + 1
        if investigation_id:
            investigation_counts[investigation_id] = investigation_counts.get(investigation_id, 0) + 1

    candidate_ids = {
        str(hit.get("finding_id") or hit.get("id") or "")
        for hit in (candidate_hits or [])
        if str(hit.get("finding_id") or hit.get("id") or "")
    }
    return {
        "selected_refs": selected_refs,
        "source_counts": source_counts,
        "investigation_counts": investigation_counts,
        "coverage_ratio": (len(selected_ids & candidate_ids) / len(candidate_ids)) if candidate_ids else 0.0,
    }


def _route_select_priority_and_exploration(hits: list[dict], *, top_k: int, priority_slots: int) -> list[dict]:
    cap = max(0, int(top_k))
    if cap <= 0:
        return []
    if len(hits) <= cap:
        return list(hits)

    p_slots = max(0, min(cap, int(priority_slots)))
    selected = list(hits[:p_slots])
    remaining = cap - len(selected)
    if remaining <= 0:
        return selected

    pool = list(hits[p_slots:])
    if not pool:
        return selected
    if len(pool) <= remaining:
        return selected + pool

    if remaining == 1:
        return selected + [pool[-1]]

    max_idx = len(pool) - 1
    step = max_idx / float(remaining - 1)
    idxs = [int(round(i * step)) for i in range(remaining)]
    deduped: list[int] = []
    seen = set()
    for idx in idxs:
        idx = max(0, min(max_idx, idx))
        if idx in seen:
            continue
        seen.add(idx)
        deduped.append(idx)
    cursor = 0
    while len(deduped) < remaining:
        if cursor not in seen:
            deduped.append(cursor)
            seen.add(cursor)
        cursor += 1
    selected.extend(pool[idx] for idx in deduped[:remaining])
    return selected


def _route_apply_policy(
    candidate_hits: list[dict],
    *,
    top_k: int,
    deduplicate: bool,
    dedup_threshold: float = 0.80,
    agent_id: Optional[str] = None,
    priority_slots: Optional[int] = None,
    overlap_fn: Optional[Callable[[dict, dict], Optional[float]]] = None,
) -> dict:
    """Apply a routing policy deterministically over a captured candidate set."""
    hits = list(candidate_hits or [])
    filtered = _route_filter_by_agent(hits, agent_id) if agent_id else hits
    deduped = filtered
    if deduplicate and len(filtered) > 1:
        threshold = _safe_float(dedup_threshold, 0.80)
        threshold = max(0.0, min(1.0, threshold))
        deduped = _route_dedup_by_overlap(filtered, threshold=threshold, overlap_fn=overlap_fn)
    cap = max(0, int(top_k))
    if priority_slots is None:
        trimmed = deduped[:cap]
        applied_priority_slots = cap
    else:
        trimmed = _route_select_priority_and_exploration(
            deduped,
            top_k=cap,
            priority_slots=int(priority_slots),
        )
        applied_priority_slots = max(0, min(cap, int(priority_slots)))
    return {
        "hits": trimmed,
        "metrics": {
            "candidate_count": len(hits),
            "after_agent_filter": len(filtered),
            "after_dedup": len(deduped),
            "priority_slots": applied_priority_slots,
            "exploration_slots": max(0, cap - applied_priority_slots),
            "after_top_k": len(trimmed),
        },
    }


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
    include_trace: bool = False,
    drive_state: Optional[dict] = None,
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
        include_trace: If True, include candidate hits + active policy in
                       `routing_trace` so audit logs can replay this decision.
        drive_state: Optional homeostatic routing drives (`hunger`, `fatigue`,
                     `urgency`, each 0..1) that steer exploration vs priority.

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
        drives = _derive_homeostatic_drives(query, drive_state)
        homeostasis = _homeostatic_route_policy(drives, top_k=top_k)
        _assert_route_policy_invariants(homeostasis, top_k=top_k)
    except ValueError as exc:
        return json.dumps({
            "error": str(exc),
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
        try:
            slow_mod = routing_policy(
                load_state(MEMORY_DIR),
                mnemo_top_k=max(1, top_k),
                qdrant_limit=max(1, int(homeostasis["candidate_limit"])),
            )
            assert_routing_policy_invariants(slow_mod, minimum_top_k=max(1, top_k))
        except Exception as exc:
            logger.warning("memory_route slow-neuromod invariant failed; fail-closed baseline policy: %r", exc)
            slow_mod = {
                "routing_tone": 0.0,
                "mnemo_top_k": max(1, top_k),
                "qdrant_limit": max(1, int(homeostasis["candidate_limit"])),
            }

        # Step 1+2: Search across the whole collection (no investigation_id filter)
        try:
            raw_hits = _qdrant_search_collection(
                query,
                collection_name=QDRANT_COLLECTION_PREFIX,
                limit=int(slow_mod.get("qdrant_limit", homeostasis["candidate_limit"])),
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

        # Retracted findings keep their Qdrant point (flagged retracted=true);
        # drop them, and every hit from an investigation the caller cannot read,
        # before ranking, so neither they nor the include_trace candidate list
        # leak. agent_id is caller-supplied and can only narrow the ACL identity.
        _route_rfilter = build_recall_filter(
            MEMORY_DIR,
            {str(r.get("investigation_id")) for r in raw_hits if r.get("investigation_id")},
            with_texts=False,
        )
        raw_hits, _route_retracted = _route_rfilter.split(raw_hits)
        raw_hits, _route_acl_excluded = inv_store._acl_filter_rows(raw_hits, agent_id)

        policy_run = _route_apply_policy(
            raw_hits,
            top_k=top_k,
            deduplicate=deduplicate,
            dedup_threshold=0.80,
            agent_id=agent_id,
            priority_slots=homeostasis["priority_slots"],
        )
        final_hits = policy_run["hits"]
        metrics = policy_run["metrics"]
        aggregation = _route_aggregation_with_provenance(final_hits, raw_hits)

        # Step 6: Build response with provenance
        routed = _route_rows(final_hits)
        payload = {
            "routed": routed,
            "query": query,
            "agent_id": agent_id,
            "total_before_dedup": metrics["candidate_count"],
            "total_after_dedup": metrics["after_top_k"],
            "count": len(routed),
            "excluded_retracted": len(_route_retracted),
            "excluded_acl": _route_acl_excluded,
            "retraction_filter": _route_rfilter.status(),
            "routing_aggregation": aggregation,
            "slow_modulation": {
                "routing_tone": round(float(slow_mod.get("routing_tone", 0.0) or 0.0), 3),
                "qdrant_limit": int(slow_mod.get("qdrant_limit", homeostasis["candidate_limit"])),
                "provenance": "deterministic_derived",
            },
        }
        trace_policy = _route_trace_policy(agent_id, top_k, deduplicate, homeostasis, slow_mod)
        if include_trace:
            payload["routing_trace"] = {
                "version": 1,
                "captured_at": _now(),
                "policy": trace_policy,
                "candidate_hits": [_route_policy_trace_hit(hit) for hit in raw_hits],
                "metrics": metrics,
                "aggregation": aggregation,
            }
        _audit_route_trace_sample(
            query=query,
            agent_id=agent_id,
            top_k=top_k,
            deduplicate=deduplicate,
            drive_state_given=drive_state is not None,
            policy=trace_policy,
            candidate_hits=raw_hits,
            routed_hits=final_hits,
            metrics=metrics,
            excluded_retracted=len(_route_retracted),
            excluded_acl=_route_acl_excluded,
        )
        try:
            route_signal = 0.0
            if int(metrics.get("after_top_k", 0)) > 0:
                route_signal = min(0.3, 0.06 * int(metrics.get("after_top_k", 0)))
                if deduplicate and int(metrics.get("after_dedup", 0)) < int(metrics.get("candidate_count", 0)):
                    route_signal -= 0.05
            observe(MEMORY_DIR, event="memory_route", routing_signal=route_signal)
        except Exception as exc:
            logger.debug("memory_route slow-neuromod update failed (fail-open): %r", exc)
        return json.dumps(payload, indent=2)

    except ValueError as exc:
        return json.dumps({
            "error": str(exc),
            "routed": [],
            "query": query,
        })
    except Exception as exc:
        logger.exception("memory_route: unexpected error: %s", exc)
        return json.dumps({
            "error": f"memory_route failed: {exc}",
            "routed": [],
        })


def _route_trace_policy(agent_id, top_k, deduplicate, homeostasis: dict, slow_mod: dict) -> dict:
    """The policy block of a routing trace: everything replay needs besides the candidates."""
    return {
        "agent_id": agent_id,
        "top_k": int(top_k),
        "deduplicate": bool(deduplicate),
        "dedup_threshold": 0.80,
        "drive_state": homeostasis["drives"],
        "candidate_multiplier": homeostasis["candidate_multiplier"],
        "priority_ratio": homeostasis["priority_ratio"],
        "slow_modulation": {
            "routing_tone": round(float(slow_mod.get("routing_tone", 0.0) or 0.0), 3),
            "qdrant_limit": int(slow_mod.get("qdrant_limit", homeostasis["candidate_limit"])),
            "provenance": "deterministic_derived",
        },
    }


# Sampled routing traces in the global audit log, so the counterfactual replay
# and policy-optimize tools have decisions to read without anyone remembering to
# call memory_route(include_trace=True) and audit_log by hand.
#
# LOCI_ROUTE_TRACE_AUDIT_RATE is the fraction of memory_route calls traced, 0..1.
# Default 0 (off): the audit-lane readers take the newest global receipts, so
# traces written at a high rate can push other receipts out of that window.
#
# A sampled trace holds ids, scores, tiers and counts only. The query is reduced
# to its length; the candidate texts are replaced by their pairwise word-overlap
# at or above ROUTE_TRACE_OVERLAP_FLOOR, which is all dedup replay needs for any
# threshold at or above that floor.
ROUTE_TRACE_AUDIT_SOURCE = "route_trace_sampler"
ROUTE_TRACE_VERSION = 2
ROUTE_TRACE_OVERLAP_FLOOR = 0.5
_route_trace_rng = random.Random()


def _route_trace_audit_rate() -> float:
    raw = os.environ.get("LOCI_ROUTE_TRACE_AUDIT_RATE", "")
    try:
        rate = float(raw) if raw.strip() else 0.0
    except ValueError:
        return 0.0
    if not math.isfinite(rate):
        return 0.0
    return max(0.0, min(1.0, rate))


def _route_overlap_edges(hits: list[dict], floor: float = ROUTE_TRACE_OVERLAP_FLOOR) -> list[list]:
    """``[i, j, overlap]`` for every candidate pair i < j whose word overlap is >= floor."""
    edges: list[list] = []
    for i in range(len(hits)):
        for j in range(i + 1, len(hits)):
            overlap = _route_word_overlap(hits[i], hits[j])
            if overlap is not None and overlap >= floor:
                edges.append([i, j, overlap])
    return edges


def _route_compact_trace_hit(idx: int, hit: dict) -> dict:
    """A candidate as a sampled trace stores it: ids, score and tier, no text or source."""
    return {
        "trace_idx": idx,
        "finding_id": hit.get("finding_id") or hit.get("id", ""),
        "id": hit.get("id") or hit.get("finding_id", ""),
        "investigation_id": hit.get("investigation_id", ""),
        "authored_by": hit.get("authored_by", ""),
        "score": _safe_float(hit.get("score"), 0.0),
        "tier": hit.get("tier") or hit.get("record_type", "finding"),
    }


def _audit_route_trace_sample(
    *,
    query: str,
    agent_id: Optional[str],
    top_k: int,
    deduplicate: bool,
    drive_state_given: bool,
    policy: dict,
    candidate_hits: list[dict],
    routed_hits: list[dict],
    metrics: dict,
    excluded_retracted: int,
    excluded_acl: int,
) -> bool:
    """Write a sampled memory_route decision to the global audit log. Fail-open."""
    try:
        rate = _route_trace_audit_rate()
        if rate <= 0.0 or _route_trace_rng.random() >= rate:
            return False
        routed_ids = [
            {
                "finding_id": h.get("finding_id") or h.get("id", ""),
                "investigation_id": h.get("investigation_id", ""),
                "score": _safe_float(h.get("score"), 0.0),
                "tier": h.get("tier") or h.get("record_type", "finding"),
            }
            for h in routed_hits
        ]
        output = {
            # Never the query text. Replay derives the drives from policy.drive_state,
            # which already includes the keyword-derived values.
            "query": "",
            "query_features": {"chars": len(query or ""), "tokens": len(str(query or "").split())},
            "routed": routed_ids,
            "count": len(routed_ids),
            "excluded_retracted": int(excluded_retracted),
            "excluded_acl": int(excluded_acl),
            "routing_trace": {
                "version": ROUTE_TRACE_VERSION,
                "captured_at": _now(),
                "policy": policy,
                "candidate_hits": [_route_compact_trace_hit(i, h) for i, h in enumerate(candidate_hits)],
                "overlap_floor": ROUTE_TRACE_OVERLAP_FLOOR,
                "overlap_edges": _route_overlap_edges(candidate_hits),
                "metrics": metrics,
            },
        }
        entry = {
            "ts": _now(),
            "created_at_ts": int(datetime.now(timezone.utc).timestamp()),
            "tool": "memory_route",
            "source": ROUTE_TRACE_AUDIT_SOURCE,
            "investigation_id": None,
            "inputs": json.dumps({
                "top_k": int(top_k),
                "deduplicate": bool(deduplicate),
                "agent_id": agent_id,
                "drive_state_given": bool(drive_state_given),
                "sample_rate": rate,
            }),
            "output": json.dumps(output),
            "evidence_provenance_tier": DETERMINISTIC_DERIVED,
            "provenance_defaulted": False,
        }
        audit_dir = MEMORY_DIR.parent / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        _append_jsonl(audit_dir / f"{date_str}.jsonl", entry)
        return True
    except Exception as exc:
        logger.debug("memory_route trace sampling failed (fail-open): %r", exc)
        return False


def _route_trace_overlap_fn(decision: dict) -> Optional[Callable[[dict, dict], Optional[float]]]:
    """Overlap lookup for a replayed decision whose trace stored overlap edges instead of text.

    None for a trace that carries candidate text (include_trace, version 1): replay
    then compares the text, exactly as before.
    """
    edges = decision.get("overlap_edges")
    if not isinstance(edges, list):
        return None
    table: dict[tuple[int, int], float] = {}
    for edge in edges:
        try:
            a, b, value = int(edge[0]), int(edge[1]), float(edge[2])
        except (TypeError, ValueError, IndexError):
            continue
        table[(min(a, b), max(a, b))] = value

    def _lookup(hit_a: dict, hit_b: dict) -> Optional[float]:
        ia, ib = hit_a.get("trace_idx"), hit_b.get("trace_idx")
        if not isinstance(ia, int) or not isinstance(ib, int):
            return None
        return table.get((min(ia, ib), max(ia, ib)), 0.0)

    return _lookup


def _route_counterfactual_decisions_from_audit(entries: list[dict], limit: int) -> tuple[list[dict], int]:
    """Extract replayable memory_route decisions from audit entries."""
    decisions: list[dict] = []
    skipped_missing_trace = 0
    for idx, entry in enumerate(entries or []):
        if len(decisions) >= max(0, int(limit)):
            break
        if not isinstance(entry, dict) or entry.get("tool") != "memory_route":
            continue
        raw_output = entry.get("output", "")
        output_obj = None
        if isinstance(raw_output, str):
            try:
                output_obj = json.loads(raw_output)
            except Exception:
                output_obj = None
        elif isinstance(raw_output, dict):
            output_obj = raw_output
        if not isinstance(output_obj, dict):
            continue
        trace = output_obj.get("routing_trace")
        if not isinstance(trace, dict):
            skipped_missing_trace += 1
            continue
        candidates = trace.get("candidate_hits")
        if not isinstance(candidates, list) or not candidates:
            skipped_missing_trace += 1
            continue
        baseline_routed = output_obj.get("routed")
        if not isinstance(baseline_routed, list):
            baseline_routed = []
        baseline_policy = trace.get("policy") if isinstance(trace.get("policy"), dict) else {}
        query = output_obj.get("query", "")
        decision_id = f"{entry.get('ts', 'unknown')}::{idx}"
        decisions.append({
            "decision_id": decision_id,
            "ts": entry.get("ts"),
            "query": query,
            "baseline_policy": baseline_policy,
            "baseline_routed": baseline_routed,
            "candidate_hits": candidates,
            "overlap_edges": trace.get("overlap_edges"),
            "overlap_floor": trace.get("overlap_floor"),
        })
    return decisions, skipped_missing_trace


@mcp.tool()
def memory_route_counterfactual_simulate(
    investigation_id: Optional[str] = None,
    limit: int = 20,
    deduplicate: Optional[bool] = None,
    dedup_threshold: float = 0.80,
    top_k: Optional[int] = None,
    agent_id_override: Optional[str] = None,
    routing_tone_override: Optional[float] = None,
) -> str:
    """
    Replay audited memory_route decisions under an alternative policy.

    This is a non-destructive simulation path: it never mutates findings,
    manifests, or production routing behavior. It reads prior audited route
    traces and compares baseline outputs vs a counterfactual policy, including
    optional slow-neuromodulation routing-tone replay overrides.
    """
    try:
        if investigation_id:
            entries = list(reversed(_read_jsonl(_inv_dir(investigation_id) / "audit.jsonl")))
            source = f"investigation:{investigation_id}"
        else:
            entries = _collect_recent_global_audit(limit=max(int(limit) * 10, 100), days=7)
            source = "global_recent_audit"

        decisions, skipped_missing_trace = _route_counterfactual_decisions_from_audit(entries, limit)
        if not decisions:
            return json.dumps({
                "source": source,
                "simulated": 0,
                "skipped_missing_trace": skipped_missing_trace,
                "results": [],
                "note": (
                    "No replayable memory_route traces found. Capture traces by calling "
                    "memory_route(..., include_trace=True) and auditing that output."
                ),
            }, indent=2)

        comparisons: list[dict] = []
        changed = 0
        for decision in decisions:
            baseline_policy = decision.get("baseline_policy") or {}
            if deduplicate is None:
                baseline_dedup = baseline_policy.get("deduplicate", True)
                if isinstance(baseline_dedup, str):
                    use_deduplicate = baseline_dedup.strip().lower() not in ("0", "false", "no")
                else:
                    use_deduplicate = bool(baseline_dedup)
            else:
                use_deduplicate = bool(deduplicate)
            try:
                use_top_k = int(baseline_policy.get("top_k", 10)) if top_k is None else int(top_k)
            except Exception:
                use_top_k = 10
            use_agent_id = agent_id_override if agent_id_override is not None else baseline_policy.get("agent_id")
            baseline_drive_state = baseline_policy.get("drive_state")
            drives = _derive_homeostatic_drives(decision.get("query", ""), baseline_drive_state if isinstance(baseline_drive_state, dict) else None)
            homeostasis = _homeostatic_route_policy(drives, top_k=use_top_k)
            _assert_route_policy_invariants(homeostasis, top_k=use_top_k)
            baseline_slow_mod = baseline_policy.get("slow_modulation")
            baseline_tone = (
                baseline_slow_mod.get("routing_tone", 0.0)
                if isinstance(baseline_slow_mod, dict)
                else 0.0
            )
            if routing_tone_override is not None:
                baseline_tone = routing_tone_override
            try:
                counter_slow_mod = routing_policy(
                    {"routing_tone": baseline_tone},
                    mnemo_top_k=max(1, use_top_k),
                    qdrant_limit=max(1, int(homeostasis["candidate_limit"])),
                )
                assert_routing_policy_invariants(counter_slow_mod, minimum_top_k=max(1, use_top_k))
            except Exception as exc:
                logger.warning("memory_route_counterfactual_simulate slow-neuromod invariant failed; fail-closed baseline policy: %r", exc)
                counter_slow_mod = {
                    "routing_tone": 0.0,
                    "mnemo_top_k": max(1, use_top_k),
                    "qdrant_limit": max(1, int(homeostasis["candidate_limit"])),
                }
            candidates_for_policy = list(decision.get("candidate_hits") or [])[
                : max(1, int(counter_slow_mod.get("qdrant_limit", homeostasis["candidate_limit"])))
            ]
            overlap_fn = _route_trace_overlap_fn(decision)
            counter = _route_apply_policy(
                candidates_for_policy,
                top_k=use_top_k,
                deduplicate=use_deduplicate,
                dedup_threshold=dedup_threshold,
                agent_id=use_agent_id,
                priority_slots=homeostasis["priority_slots"],
                overlap_fn=overlap_fn,
            )
            counter_rows = _route_rows(counter["hits"])
            counter_aggregation = _route_aggregation_with_provenance(counter["hits"], candidates_for_policy)
            baseline_rows = decision.get("baseline_routed") or []
            baseline_ids = [str(r.get("finding_id") or r.get("id") or "") for r in baseline_rows]
            counter_ids = [str(r.get("finding_id") or r.get("id") or "") for r in counter_rows]
            baseline_set = {x for x in baseline_ids if x}
            counter_set = {x for x in counter_ids if x}
            added = sorted(counter_set - baseline_set)
            removed = sorted(baseline_set - counter_set)
            if added or removed:
                changed += 1
            floor = _safe_float(decision.get("overlap_floor"), 0.0)
            comparisons.append({
                "decision_id": decision.get("decision_id"),
                "ts": decision.get("ts"),
                "query": decision.get("query", ""),
                "baseline_count": len(baseline_rows),
                # "overlap_edges": the trace stored overlaps >= its floor, not text.
                # Dedup replay is then exact only for thresholds at or above that floor.
                "dedup_basis": "overlap_edges" if overlap_fn is not None else "text",
                "dedup_exact": (
                    overlap_fn is None
                    or not use_deduplicate
                    or _safe_float(dedup_threshold, 0.80) >= floor
                ),
                "counterfactual": {
                    "policy": {
                        "agent_id": use_agent_id,
                        "top_k": use_top_k,
                        "deduplicate": use_deduplicate,
                        "dedup_threshold": _safe_float(dedup_threshold, 0.80),
                        "drive_state": homeostasis["drives"],
                        "candidate_multiplier": homeostasis["candidate_multiplier"],
                        "priority_ratio": homeostasis["priority_ratio"],
                        "slow_modulation": {
                            "routing_tone": round(float(counter_slow_mod.get("routing_tone", 0.0) or 0.0), 3),
                            "qdrant_limit": int(counter_slow_mod.get("qdrant_limit", homeostasis["candidate_limit"])),
                            "provenance": "deterministic_derived",
                        },
                    },
                    "count": len(counter_rows),
                    "added_finding_ids": added[:25],
                    "removed_finding_ids": removed[:25],
                    "overlap_count": len(baseline_set & counter_set),
                    "metrics": counter["metrics"],
                    "aggregation": counter_aggregation,
                },
            })

        return json.dumps({
            "source": source,
            "simulated": len(comparisons),
            "changed": changed,
            "skipped_missing_trace": skipped_missing_trace,
            "counterfactual_overrides": {
                "top_k": top_k,
                "deduplicate": deduplicate,
                "dedup_threshold": _safe_float(dedup_threshold, 0.80),
                "agent_id_override": agent_id_override,
                "routing_tone_override": routing_tone_override,
            },
            "results": comparisons,
        }, indent=2)
    except Exception as exc:
        logger.exception("memory_route_counterfactual_simulate failed: %s", exc)
        return json.dumps({
            "error": f"memory_route_counterfactual_simulate failed: {exc}",
            "results": [],
        })


def _route_baseline_policy_summary(decisions: list[dict]) -> dict:
    top_k_values: list[int] = []
    dedup_true = 0
    dedup_false = 0
    threshold_values: list[float] = []
    for decision in decisions or []:
        policy = decision.get("baseline_policy") if isinstance(decision, dict) else {}
        if not isinstance(policy, dict):
            continue
        try:
            top_k_values.append(max(0, int(policy.get("top_k", 10))))
        except Exception:
            top_k_values.append(10)
        dedup_raw = policy.get("deduplicate", True)
        dedup_value = (
            dedup_raw.strip().lower() not in ("0", "false", "no")
            if isinstance(dedup_raw, str)
            else bool(dedup_raw)
        )
        if dedup_value:
            dedup_true += 1
        else:
            dedup_false += 1
        threshold_values.append(_safe_float(policy.get("dedup_threshold", 0.80), 0.80))
    if not top_k_values:
        top_k_values = [10]
    if not threshold_values:
        threshold_values = [0.80]
    top_k_values = sorted(top_k_values)
    threshold_values = sorted(threshold_values)
    median_top_k = top_k_values[len(top_k_values) // 2]
    median_threshold = threshold_values[len(threshold_values) // 2]
    return {
        "observed_decisions": len(decisions or []),
        "top_k_median": int(median_top_k),
        "deduplicate_majority": dedup_true >= dedup_false,
        "dedup_threshold_median": max(0.0, min(1.0, _safe_float(median_threshold, 0.80))),
    }


def _route_collect_consolidation_outcomes(entries: list[dict]) -> dict:
    sampled = 0
    flagged = 0
    degraded = 0
    seen = 0
    for entry in entries or []:
        if not isinstance(entry, dict) or entry.get("tool") != "memory_consolidate":
            continue
        raw_output = entry.get("output", "")
        output_obj = None
        if isinstance(raw_output, str):
            try:
                output_obj = json.loads(raw_output)
            except Exception:
                output_obj = None
        elif isinstance(raw_output, dict):
            output_obj = raw_output
        if not isinstance(output_obj, dict):
            continue
        seen += 1
        quality = output_obj.get("consolidation_quality_audit")
        if not isinstance(quality, dict):
            continue
        sampled += max(0, int(quality.get("sampled", 0) or 0))
        flagged += len(quality.get("flagged") or [])
        degraded += 1 if bool(quality.get("degraded")) else 0
    rate = (flagged / sampled) if sampled > 0 else 0.0
    return {
        "seen": seen,
        "sampled": sampled,
        "flagged": flagged,
        "flagged_rate": max(0.0, min(1.0, _safe_float(rate, 0.0))),
        "degraded_runs": degraded,
    }


def _route_policy_optimization_invariants(decisions: list[dict], consolidation: dict) -> dict:
    """Fail-closed invariant gate for policy optimization inputs."""
    violations: list[str] = []
    checked = 0
    for decision in decisions or []:
        if not isinstance(decision, dict):
            continue
        checked += 1
        decision_id = str(decision.get("decision_id") or f"decision-{checked}")
        policy = decision.get("baseline_policy") if isinstance(decision.get("baseline_policy"), dict) else {}
        top_k_raw = policy.get("top_k", 10)
        top_k = int(_safe_float(top_k_raw, 10))
        if top_k < 0:
            violations.append(f"{decision_id}: baseline top_k is negative")
        dedup_threshold = _safe_float(policy.get("dedup_threshold", 0.80), 0.80)
        if dedup_threshold < 0.0 or dedup_threshold > 1.0:
            violations.append(f"{decision_id}: baseline dedup_threshold out of range")
        candidates = decision.get("candidate_hits")
        if not isinstance(candidates, list) or not candidates:
            violations.append(f"{decision_id}: candidate_hits missing or empty")
            continue
        candidate_ids = {
            str(hit.get("finding_id") or hit.get("id") or "")
            for hit in candidates
            if isinstance(hit, dict)
        }
        candidate_ids = {x for x in candidate_ids if x}
        if not candidate_ids:
            violations.append(f"{decision_id}: candidate_hits contain no valid finding IDs")
        baseline_rows = decision.get("baseline_routed")
        if not isinstance(baseline_rows, list):
            violations.append(f"{decision_id}: baseline_routed is not a list")
            continue
        for row in baseline_rows:
            if not isinstance(row, dict):
                continue
            fid = str(row.get("finding_id") or row.get("id") or "")
            if fid and fid not in candidate_ids:
                violations.append(f"{decision_id}: baseline finding_id {fid} absent from candidate_hits")

    sampled = max(0, int(consolidation.get("sampled", 0) or 0))
    flagged = max(0, int(consolidation.get("flagged", 0) or 0))
    if flagged > sampled:
        violations.append("consolidation outcomes invalid: flagged exceeds sampled")

    return {
        "ok": len(violations) == 0,
        "checked_decisions": checked,
        "violations": violations[:25],
    }


def _route_policy_provenance_aggregation(decisions: list[dict], consolidation: dict, source: str) -> dict:
    decision_refs: list[dict] = []
    candidate_total = 0
    baseline_total = 0
    for decision in decisions or []:
        if not isinstance(decision, dict):
            continue
        decision_refs.append({
            "decision_id": decision.get("decision_id"),
            "ts": decision.get("ts"),
        })
        candidate_hits = decision.get("candidate_hits") if isinstance(decision.get("candidate_hits"), list) else []
        baseline_rows = decision.get("baseline_routed") if isinstance(decision.get("baseline_routed"), list) else []
        candidate_total += len(candidate_hits)
        baseline_total += len(baseline_rows)
    return {
        "source": source,
        "decision_count": len(decision_refs),
        "decision_refs": decision_refs[:25],
        "candidate_hits_total": candidate_total,
        "baseline_routed_total": baseline_total,
        "consolidation_seen": max(0, int(consolidation.get("seen", 0) or 0)),
        "consolidation_sampled": max(0, int(consolidation.get("sampled", 0) or 0)),
    }


def _route_candidate_grid(baseline: dict) -> list[dict]:
    base_top_k = max(1, int(baseline.get("top_k_median", 10) or 10))
    base_dedup = bool(baseline.get("deduplicate_majority", True))
    base_threshold = max(0.0, min(1.0, _safe_float(baseline.get("dedup_threshold_median", 0.80), 0.80)))
    top_k_values = sorted({max(1, base_top_k - 1), base_top_k, base_top_k + 1})
    threshold_values = sorted({
        max(0.0, min(1.0, round(base_threshold - 0.05, 2))),
        base_threshold,
        max(0.0, min(1.0, round(base_threshold + 0.05, 2))),
    })
    grid: list[dict] = []
    for tk in top_k_values:
        for dd in (base_dedup, not base_dedup):
            for th in threshold_values:
                # Keep a compact conservative grid: threshold only matters if dedup is on.
                if not dd and th != base_threshold:
                    continue
                grid.append({
                    "top_k": int(tk),
                    "deduplicate": bool(dd),
                    "dedup_threshold": max(0.0, min(1.0, _safe_float(th, 0.80))),
                })
    # Deterministic ordering and stable de-dup.
    unique: list[dict] = []
    seen: set[tuple[int, bool, float]] = set()
    for cand in grid:
        key = (cand["top_k"], cand["deduplicate"], cand["dedup_threshold"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(cand)
    return unique


def _route_eval_candidate(
    decisions: list[dict],
    *,
    top_k: int,
    deduplicate: bool,
    dedup_threshold: float,
    consolidation_flagged_rate: float,
) -> dict:
    compared = 0
    total_baseline = 0
    total_counter = 0
    removed = 0
    added = 0
    overlap = 0
    for decision in decisions or []:
        if not isinstance(decision, dict):
            continue
        counter = _route_apply_policy(
            decision.get("candidate_hits") or [],
            top_k=top_k,
            deduplicate=deduplicate,
            dedup_threshold=dedup_threshold,
            agent_id=decision.get("baseline_policy", {}).get("agent_id"),
            overlap_fn=_route_trace_overlap_fn(decision),
        )
        baseline_rows = decision.get("baseline_routed") or []
        counter_rows = _route_rows(counter["hits"])
        baseline_ids = {str(r.get("finding_id") or r.get("id") or "") for r in baseline_rows if isinstance(r, dict)}
        counter_ids = {str(r.get("finding_id") or r.get("id") or "") for r in counter_rows if isinstance(r, dict)}
        compared += 1
        total_baseline += len(baseline_ids)
        total_counter += len(counter_ids)
        removed += len([x for x in (baseline_ids - counter_ids) if x])
        added += len([x for x in (counter_ids - baseline_ids) if x])
        overlap += len(baseline_ids & counter_ids)
    compared = max(compared, 1)
    # We optimize for stability first; modest compression is good, evidence loss is bad.
    baseline_nonzero = max(total_baseline, 1)
    stability = overlap / baseline_nonzero
    compression = max(0.0, (total_baseline - total_counter) / baseline_nonzero)
    removal_rate = removed / baseline_nonzero
    addition_rate = added / baseline_nonzero
    penalty_factor = 0.7 + max(0.0, min(1.0, _safe_float(consolidation_flagged_rate, 0.0)))
    objective = (
        stability
        + (0.20 * compression)
        - (penalty_factor * removal_rate)
        - (0.20 * addition_rate)
    )
    return {
        "policy": {
            "top_k": int(top_k),
            "deduplicate": bool(deduplicate),
            "dedup_threshold": max(0.0, min(1.0, _safe_float(dedup_threshold, 0.80))),
        },
        "metrics": {
            "decisions_compared": compared,
            "baseline_ids": total_baseline,
            "counter_ids": total_counter,
            "overlap_ids": overlap,
            "removed_ids": removed,
            "added_ids": added,
            "stability": round(stability, 6),
            "compression": round(compression, 6),
            "removal_rate": round(removal_rate, 6),
            "addition_rate": round(addition_rate, 6),
        },
        "objective": round(float(objective), 6),
    }


def _append_route_policy_report(record: dict) -> None:
    try:
        path = MEMORY_DIR / "_policy" / "route_policy_optimization.jsonl"
        _append_jsonl(path, record)
    except Exception as exc:
        logger.debug("_append_route_policy_report failed (fail-open): %r", exc)


@mcp.tool()
def memory_route_policy_optimize(
    investigation_id: Optional[str] = None,
    limit: int = 30,
    days: int = 7,
    min_decisions: int = 8,
    persist: bool = False,
) -> str:
    """
    Derive conservative memory_route policy recommendations from audited outcomes.

    Reads audited `memory_route` traces (captured via `include_trace=True`) and
    `memory_consolidate` advisory quality outcomes, evaluates a bounded policy
    grid, and proposes at most one recommendation. Safe by default: no routing
    behavior changes are applied automatically.
    """
    try:
        if investigation_id:
            entries = list(reversed(_read_jsonl(_inv_dir(investigation_id) / "audit.jsonl")))
            source = f"investigation:{investigation_id}"
        else:
            entries = _collect_recent_global_audit(limit=max(int(limit) * 20, 200), days=max(1, int(days)))
            source = f"global_recent_audit_{max(1, int(days))}d"

        decisions, skipped_missing_trace = _route_counterfactual_decisions_from_audit(entries, limit)
        consolidation = _route_collect_consolidation_outcomes(entries)
        if len(decisions) < max(1, int(min_decisions)):
            return json.dumps({
                "source": source,
                "optimized": False,
                "reason": "insufficient_decisions",
                "decision_count": len(decisions),
                "min_decisions": max(1, int(min_decisions)),
                "skipped_missing_trace": skipped_missing_trace,
                "consolidation_outcomes": consolidation,
                "recommendation": None,
            }, indent=2)

        invariant_gate = _route_policy_optimization_invariants(decisions, consolidation)
        if not invariant_gate.get("ok"):
            return json.dumps({
                "source": source,
                "optimized": False,
                "reason": "invariant_gate_failed",
                "decision_count": len(decisions),
                "min_decisions": max(1, int(min_decisions)),
                "skipped_missing_trace": skipped_missing_trace,
                "consolidation_outcomes": consolidation,
                "invariant_gate": invariant_gate,
                "recommendation": None,
            }, indent=2)

        provenance_aggregation = _route_policy_provenance_aggregation(decisions, consolidation, source)
        baseline = _route_baseline_policy_summary(decisions)
        candidates = _route_candidate_grid(baseline)
        evaluations = [
            _route_eval_candidate(
                decisions,
                top_k=c["top_k"],
                deduplicate=c["deduplicate"],
                dedup_threshold=c["dedup_threshold"],
                consolidation_flagged_rate=consolidation["flagged_rate"],
            )
            for c in candidates
        ]
        evaluations.sort(
            key=lambda row: (
                _safe_float(row.get("objective"), -999.0),
                _safe_float((row.get("metrics") or {}).get("stability"), 0.0),
            ),
            reverse=True,
        )
        best = evaluations[0] if evaluations else None
        baseline_eval = next(
            (
                row for row in evaluations
                if row.get("policy", {}).get("top_k") == baseline["top_k_median"]
                and row.get("policy", {}).get("deduplicate") == baseline["deduplicate_majority"]
                and abs(_safe_float(row.get("policy", {}).get("dedup_threshold"), 0.80) - baseline["dedup_threshold_median"]) < 1e-9
            ),
            None,
        )
        if baseline_eval is None:
            baseline_eval = _route_eval_candidate(
                decisions,
                top_k=baseline["top_k_median"],
                deduplicate=baseline["deduplicate_majority"],
                dedup_threshold=baseline["dedup_threshold_median"],
                consolidation_flagged_rate=consolidation["flagged_rate"],
            )

        baseline_score = _safe_float(baseline_eval.get("objective"), -999.0)
        best_score = _safe_float(best.get("objective"), -999.0) if isinstance(best, dict) else -999.0
        score_delta = best_score - baseline_score
        best_metrics = (best or {}).get("metrics", {}) if isinstance(best, dict) else {}
        safe_recommendation = (
            isinstance(best, dict)
            and score_delta >= 0.02
            and _safe_float(best_metrics.get("stability"), 0.0) >= 0.85
            and _safe_float(best_metrics.get("removal_rate"), 1.0) <= 0.15
        )
        recommendation = (best or {}).get("policy") if safe_recommendation else None
        note = (
            "recommended policy is conservative and simulation-backed"
            if recommendation is not None
            else "no safe improvement over baseline policy"
        )
        payload = {
            "source": source,
            "optimized": recommendation is not None,
            "note": note,
            "decision_count": len(decisions),
            "min_decisions": max(1, int(min_decisions)),
            "skipped_missing_trace": skipped_missing_trace,
            "consolidation_outcomes": consolidation,
            "baseline_policy_summary": baseline,
            "baseline_objective": round(baseline_score, 6),
            "best_objective": round(best_score, 6),
            "score_delta": round(score_delta, 6),
            "recommendation": recommendation,
            "top_candidates": evaluations[:5],
            "provenance_aggregation": provenance_aggregation,
        }
        if persist:
            # Persist aggregate outcomes only (policy + metrics), never raw hit texts.
            _append_route_policy_report({
                "ts": _now(),
                "source": source,
                "decision_count": len(decisions),
                "skipped_missing_trace": skipped_missing_trace,
                "consolidation_outcomes": consolidation,
                "baseline_policy_summary": baseline,
                "baseline_objective": payload["baseline_objective"],
                "best_objective": payload["best_objective"],
                "score_delta": payload["score_delta"],
                "recommendation": recommendation,
                "top_candidates": payload["top_candidates"],
                "provenance_aggregation": provenance_aggregation,
            })
            _event_log_append({
                "op": "route_policy_optimize",
                "source": source,
                "decision_count": len(decisions),
                "optimized": recommendation is not None,
                "score_delta": payload["score_delta"],
                "persisted": True,
            })
            payload["persisted"] = True
        return json.dumps(payload, indent=2)
    except Exception as exc:
        logger.exception("memory_route_policy_optimize failed: %s", exc)
        return json.dumps({
            "error": f"memory_route_policy_optimize failed: {exc}",
            "optimized": False,
        })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


import graph_tools  # noqa: E402
graph_tools.register(mcp, _get_ladybug)
# Re-exported so server.<tool>() keeps resolving for in-process callers and tests.
from graph_tools import (  # noqa: E402,F401
    code_graph_ingest, code_graph_query, code_memory_relink, code_memory_map,
    symbol_impact, impact_report, finding_code_context, investigation_code_briefing,
    subsystem_report, related_investigations_via_code, dead_code_candidates,
)

# Local-model / embedding passthrough tools live in llm_tools.py (P2a of the split).
import llm_tools  # noqa: E402
llm_tools.register(mcp)
llm_tools.linked_evidence_fn = _firewall_linked_evidence
# Re-exported so server.<tool>() keeps resolving for in-process callers and tests.
from llm_tools import (  # noqa: E402,F401
    llm_local, generate_batch, query_expand, verify_finding, adversarial_review,
    classify_text, compress_text, semantic_dedup, semantic_relevance, ground,
    swarm_reason, offload_tool_loop,
)

# Memory root injected as a lambda over MEMORY_DIR; collaborators are passed in so investigation_tools never imports server.
import investigation_tools  # noqa: E402
investigation_tools.register(mcp, lambda: MEMORY_DIR, {
    "_apply_lifecycle": _apply_lifecycle,
    "_compute_self_check": _compute_self_check,
    "_event_log_append": _event_log_append,
    "_qdrant_upsert": _qdrant_upsert,
})
# These resolve their helpers in investigation_tools' namespace: patch investigation_tools.<helper>, not server.<helper>.
from investigation_tools import (  # noqa: E402,F401
    investigation_start, investigation_load, investigation_as_of,
    investigation_note, investigation_reflect, investigation_finding_provenance,
    investigation_list, investigation_share, investigation_unshare,
    investigation_export, investigation_import,
    investigation_queue_enqueue, investigation_queue_claim,
    investigation_queue_complete, investigation_queue_release,
    investigation_queue_status, investigation_queue_list,
)

# Real tools are injected into the offload loop here (server.py may run as __main__, so it cannot resolve them itself).
import offload_loop  # noqa: E402
offload_loop.bind_tools(
    {n: globals()[n] for n in offload_loop.TOOL_SPECS if n in globals()},
    lambda: MEMORY_DIR,
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

    Per-agent tokens (``LOCI_MCP_AGENT_TOKENS``, ``{token: agent_id}`` here)
    are what bind a caller's identity to the transport: a request presenting one
    has that agent id written into the ASGI scope under
    ``caller_identity.SCOPE_KEY``, which the ACL checks read in place of the
    self-declared ``requesting_agent_id``. The shared token authenticates but
    cannot tell callers apart, so it binds nothing.
    """

    __slots__ = ("_app", "_token", "_exempt", "_agent_tokens")

    def __init__(self, app, token: str, exempt_paths=frozenset({"/health"}), agent_tokens=None):
        self._app = app
        self._token = token or ""
        self._exempt = exempt_paths
        self._agent_tokens = dict(agent_tokens or {})

    def _match(self, presented: str):
        """(authenticated, bound agent id or None), checking every token."""
        ok = False
        agent = None
        if self._token and hmac.compare_digest(presented, self._token):
            ok = True
        # No early exit: the loop's timing does not depend on which token matched.
        for tok, agent_id in self._agent_tokens.items():
            if hmac.compare_digest(presented, tok):
                ok, agent = True, agent_id
        return ok, agent

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("path") in self._exempt:
            await self._app(scope, receive, send)
            return
        raw = dict(scope.get("headers") or {}).get(b"authorization", b"")
        value = raw.decode("latin-1") if isinstance(raw, bytes) else str(raw)
        presented = value[7:] if value[:7].lower() == "bearer " else ""
        # compare_digest on both branches: == leaks token length/prefix, and an early return leaks whether a token was presented.
        ok, agent = self._match(presented) if presented else (False, None)
        if not ok:
            body = b'{"error":"unauthorized"}'
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(body)).encode()),
                                    (b"www-authenticate", b"Bearer")]})
            await send({"type": "http.response.body", "body": body})
            return
        scope = dict(scope)
        if agent:
            scope[caller_identity.SCOPE_KEY] = agent
        else:
            scope.pop(caller_identity.SCOPE_KEY, None)
        await self._app(scope, receive, send)


def main() -> None:
    # Warm-ping so the first RAG/dedup call doesn't eat the ~9s nomic cold-load; non-blocking, fail-open.
    try:
        import embed_ops
        embed_ops.warm()
    except Exception as exc:
        logger.debug("main: fail-open swallow: %r", exc)
    transport = os.environ.get("LOCI_MCP_TRANSPORT", "stdio")
    if transport in ("sse", "streamable-http"):
        # Loopback default: no authentication here, so a wide bind must be an explicit decision (docker-compose sets LOCI_MCP_HOST=0.0.0.0).
        mcp.settings.host = os.environ.get("LOCI_MCP_HOST", "127.0.0.1")
        mcp.settings.port = int(os.environ.get("LOCI_MCP_PORT", "8000"))

        # A token is what makes a NON-loopback bind defensible; without one we refuse rather than serve.
        token = os.environ.get("LOCI_MCP_TOKEN", "").strip()
        try:
            agent_tokens = caller_identity.load_agent_tokens()
        except (OSError, ValueError) as exc:
            raise SystemExit(f"refusing to start: LOCI_MCP_AGENT_TOKENS is unusable: {exc}")
        if token and token in agent_tokens:
            raise SystemExit("refusing to start: LOCI_MCP_TOKEN is also a per-agent token; "
                             "a shared secret cannot also be one agent's identity")
        if not token and not agent_tokens and not _is_loopback(mcp.settings.host):
            raise SystemExit(
                f"refusing to serve {transport} on {mcp.settings.host} without "
                "LOCI_MCP_TOKEN: this would expose every tool unauthenticated. "
                'Generate one with: python3 -c "import secrets;print(secrets.token_hex(32))" '
                "— or bind 127.0.0.1."
            )

        app = (mcp.streamable_http_app() if transport == "streamable-http"
               else mcp.sse_app())
        if token or agent_tokens:
            app = _BearerAuthMiddleware(app, token, agent_tokens=agent_tokens)
            if agent_tokens:
                logger.info("per-agent MCP tokens configured for %d agent(s); ACL identity "
                            "is bound from the bearer token", len(set(agent_tokens.values())))
        else:
            logger.warning("LOCI_MCP_TOKEN is not set — serving %s on %s with no "
                           "authentication. Safe only because the bind is loopback.",
                           transport, mcp.settings.host)

        import uvicorn
        uvicorn.run(app, host=mcp.settings.host, port=mcp.settings.port,
                    log_level=str(mcp.settings.log_level).lower())
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
