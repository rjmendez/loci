"""Embedding + Qdrant vector-store helpers (extracted from server.py).

Every ``from qdrant_client ...`` / ``import requests`` import in here is kept
FUNCTION-LOCAL on purpose: that laziness is what makes the whole cluster
fail-open when Qdrant / fastembed / the embedding endpoint are unavailable.

The singletons below (`_qdrant_client`, `_sparse_model`, `_embed_cache`,
`_embed_sparse_cache`) live here and only here — server.py re-exports the
functions, not the state, so there is exactly one latch per process.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

from inv_store import _CONFIDENCE_RANK

logger = logging.getLogger("loci-mcp")


# --- Configuration (public: re-exported by server.py) ---
QDRANT_COLLECTION_PREFIX = os.environ.get("QDRANT_COLLECTION_PREFIX", "loci_memory")
VECTOR_DIM = int(os.environ.get("MNEMOSYNE_EMBEDDING_DIM", 768))

# server.py still reads CODE_CHUNKS_COLLECTION from os.environ while it imports this
# module. Seed that env var from the same resolver used by unattended entry points
# without touching Qdrant or any network backend.
try:
    if not os.environ.get("CODE_CHUNKS_COLLECTION"):
        import backends as _loci_backends
        _code_col = _loci_backends.code_chunks_collection()
        if _code_col:
            os.environ["CODE_CHUNKS_COLLECTION"] = _code_col
    if "LOCI_RAG_EXPAND" not in os.environ:
        # Profiled 2026-08-29 on this deployment: query expansion took 2.82s via
        # gen_url and multiplied retrieval from 2 collection searches to 8. The
        # duplicated per-search reranker then pushed a normal call to 37-140s.
        # Keep hybrid dense+sparse retrieval responsive by default; operators can
        # re-enable expansion with LOCI_RAG_EXPAND=1 after sizing the deadline.
        os.environ["LOCI_RAG_EXPAND"] = "0"
except Exception as exc:
    logger.debug("code chunks collection resolution failed (fail-open): %r", exc)


# --- Lazy singletons / fail-open latches ---
_sparse_model = None
_sparse_model_lock = threading.Lock()          # guards _sparse_model lazy-init (#106)
_qdrant_client: tuple | None = None    # (QdrantClient, collection_name) singleton
_qdrant_failed_at: float | None = None  # monotonic timestamp of last connection failure
_QDRANT_RETRY_SECONDS = 60             # backoff before retrying after a transient failure


def _get_sparse_embedder():
    global _sparse_model
    if _sparse_model is None:
        with _sparse_model_lock:
            if _sparse_model is None:
                try:
                    from fastembed import SparseTextEmbedding
                    _sparse_model = SparseTextEmbedding("Qdrant/bm25", language="english", avg_len=200, disable_stemmer=True)
                except Exception as exc:
                    logger.warning("SparseTextEmbedding unavailable: %s", exc)
                    _sparse_model = False
    return _sparse_model if _sparse_model is not False else None


def _get_cross_encoder():
    """The two-stage reranker's CrossEncoder, or None when unavailable (fail-open).

    Delegates to reranker.get_model() so the backend is env-pluggable via RERANK_MODEL (and the
    backends config): default 'BAAI/bge-reranker-v2-m3' (flipped in on judge-eval evidence,
    +14% nDCG@10); pin the lighter 'cross-encoder/ms-marco-MiniLM-L-6-v2' back on constrained
    hosts. Lazy-init, globally cached, loaded on GPU (cuda:0) when available. Call sites keep
    calling `.predict(pairs)` unchanged.

    NOTE: changing RERANK_MODEL is a retrieval-QUALITY change — A/B it on a held-out query set
    first (scripts/judge_eval.py is the judge-based harness that gated the bge flip).
    """
    try:
        import reranker
        return reranker.get_model()
    except Exception as exc:
        logger.warning("Reranker unavailable — reranking disabled: %s", exc)
        return None


def _embed_sparse(text: str):
    """Returns a SparseVector or None."""
    cached = _embed_sparse_cache.get(text)
    if cached is not None:
        try:
            from qdrant_client.models import SparseVector
            return SparseVector(indices=list(cached[0]), values=list(cached[1]))
        except Exception as exc:
            logger.debug("_embed_sparse: fail-open swallow: %r", exc)
    model = _get_sparse_embedder()
    if model is None:
        return None
    try:
        from qdrant_client.models import SparseVector
        result = list(model.embed([text]))[0]
        indices = result.indices.tolist()
        values = result.values.tolist()
        with _embed_sparse_cache_lock:
            if len(_embed_sparse_cache) >= _EMBED_CACHE_MAXSIZE:
                _embed_sparse_cache.pop(next(iter(_embed_sparse_cache)))
            _embed_sparse_cache[text] = (tuple(indices), tuple(values))
        return SparseVector(indices=indices, values=values)
    except Exception as exc:
        logger.debug("sparse embed failed: %s", exc)
        return None


def _create_payload_indexes(client, col: str) -> None:
    """Create payload indexes for filtered search. Idempotent."""
    from qdrant_client.models import (
        KeywordIndexParams, KeywordIndexType,
        IntegerIndexParams, IntegerIndexType,
    )
    indexes = [
        # Core investigation fields
        ("investigation_id", KeywordIndexParams(
            type=KeywordIndexType.KEYWORD, is_tenant=True, on_disk=False)),
        ("record_type", KeywordIndexParams(
            type=KeywordIndexType.KEYWORD, on_disk=False)),
        ("server", KeywordIndexParams(
            type=KeywordIndexType.KEYWORD, on_disk=False)),
        ("tool", KeywordIndexParams(
            type=KeywordIndexType.KEYWORD, on_disk=False)),
        ("created_at_ts", IntegerIndexParams(
            type=IntegerIndexType.INTEGER, lookup=False, range=True, on_disk=False)),
        # Evidence quality fields — enable confidence-filtered retrieval
        ("confidence", KeywordIndexParams(
            type=KeywordIndexType.KEYWORD, on_disk=False)),
        ("tags", KeywordIndexParams(
            type=KeywordIndexType.KEYWORD, on_disk=False)),
        # Multi-tenancy fields — is_tenant=True gives HNSW partition hints.
        ("agent_id",    KeywordIndexParams(
            type=KeywordIndexType.KEYWORD, is_tenant=True, on_disk=False)),
        ("operator_id", KeywordIndexParams(
            type=KeywordIndexType.KEYWORD, is_tenant=True, on_disk=False)),
        ("namespace",   KeywordIndexParams(
            type=KeywordIndexType.KEYWORD, on_disk=False)),
        ("promoted_at_ts", IntegerIndexParams(
            type=IntegerIndexType.INTEGER, lookup=False, range=True, on_disk=False)),
        ("promoted_from", KeywordIndexParams(
            type=KeywordIndexType.KEYWORD, on_disk=False)),
        # Entity fields — nested-array dot-notation indexing is undocumented qdrant-client 1.17.x behaviour; re-verify on upgrade.
        ("entities.ips",       KeywordIndexParams(type=KeywordIndexType.KEYWORD, on_disk=False)),
        ("entities.emails",    KeywordIndexParams(type=KeywordIndexType.KEYWORD, on_disk=False)),
        ("entities.hostnames", KeywordIndexParams(type=KeywordIndexType.KEYWORD, on_disk=False)),
        ("entities.hashes",    KeywordIndexParams(type=KeywordIndexType.KEYWORD, on_disk=False)),
        ("entities.cves",      KeywordIndexParams(type=KeywordIndexType.KEYWORD, on_disk=False)),
    ]
    for field_name, schema in indexes:
        try:
            client.create_payload_index(col, field_name=field_name,
                                        field_schema=schema, wait=False)
        except Exception as exc:
            logger.debug("payload index %r creation skipped: %s", field_name, exc)


def _retention_days() -> int:
    """Startup purge window, in days. 0 (the default) disables the purge entirely.

    The default is 0 because this window DELETES FINDINGS and the deletion is
    silent. It used to default to 30, which meant a process that simply forgot to
    export LOCI_QDRANT_RETENTION_DAYS destroyed every indexed finding older than a
    month on its next start. Measured on the live store before this change: the
    index held 912 findings and the corpus held 2,831, and the split was exact —
    all 912 findings younger than 30 days were indexed, and zero of the 1,919
    older ones were. The index boundary WAS the retention window. Re-indexing
    restored coverage and the next server start removed it again.

    Purging is a real option, but it has to be one somebody chose. Nothing in this
    repo opts in.

    Resolution order: environment, then ~/.loci/backends.toml, then 0. The
    backends floor exists because the env var only reaches a process whose
    launcher remembers to set it — and the four live MCP servers did not. A
    setting that protects the corpus must not depend on being remembered.

    Read at call time, not import time, so a test or a caller can set it."""
    raw = os.environ.get("LOCI_QDRANT_RETENTION_DAYS", "").strip()
    if not raw:
        try:
            import backends
            days = backends._cfg("qdrant", "retention_days", None)
            if days is not None:
                raw = str(days).strip()
        except Exception as exc:
            logger.debug("retention: backends.toml unreadable (%r); using the safe default", exc)
    if not raw:
        return 0
    try:
        days = int(raw)
    except ValueError:
        # Falls back to DISABLED, not to a window: a guessed number guesses how much corpus to delete.
        logger.warning("LOCI_QDRANT_RETENTION_DAYS=%r is not an integer; "
                       "disabling the purge rather than guessing a window", raw)
        return 0
    return max(0, days)


def _purge_old_records(client, col: str, retention_days: Optional[int] = None) -> None:
    """Delete records older than retention_days. Requires created_at_ts payload index.

    This DESTROYS data: findings carry created_at_ts (server._store_finding), so
    anything past the window goes on the next process start. Set
    LOCI_QDRANT_RETENTION_DAYS=0 to keep the store durable.
    """
    if retention_days is None:
        retention_days = _retention_days()
    if retention_days <= 0:
        logger.info("Qdrant TTL purge disabled for %r (retention=0) — no findings deleted", col)
        return

    from qdrant_client.models import Filter, FieldCondition, Range, FilterSelector

    cutoff = int(time.time()) - (retention_days * 86400)
    stale = Filter(must=[FieldCondition(key="created_at_ts", range=Range(lt=cutoff))])
    try:
        # Count first: the delete is wait=False, so the log otherwise cannot say what it removed.
        try:
            doomed = int(client.count(collection_name=col, count_filter=stale, exact=True).count)
        except Exception as exc:
            logger.debug("Qdrant TTL purge: count failed: %s", exc)
            doomed = -1

        if doomed == 0:
            logger.debug("Qdrant TTL purge: nothing older than %d days", retention_days)
            return

        client.delete(
            collection_name=col,
            points_selector=FilterSelector(filter=stale),
            wait=False,
        )
        logger.warning(
            "Qdrant TTL purge: deleting %s point(s) older than %d days from %r "
            "(set LOCI_QDRANT_RETENTION_DAYS=0 to disable)",
            doomed if doomed >= 0 else "an unknown number of", retention_days, col,
        )
    except Exception as exc:
        logger.debug("Qdrant TTL purge failed (non-fatal): %s", exc)


def _get_qdrant():
    """Return (QdrantClient, collection_name) or (None, None) if unavailable.
    All findings share one collection; investigation_id is a payload field.
    Reads QDRANT_URL lazily so the env var is picked up even if set after import.

    Connection failures are cached for _QDRANT_RETRY_SECONDS so a transient
    startup race (container not yet ready) doesn't permanently disable Qdrant
    for the process lifetime.
    """
    import time as _time
    global _qdrant_client, _qdrant_failed_at
    qdrant_url = os.environ.get("QDRANT_URL", "")
    if not qdrant_url:
        return None, None
    # Return cached failure if still within the backoff window
    if _qdrant_client == (None, None) and _qdrant_failed_at is not None:
        if _time.monotonic() - _qdrant_failed_at < _QDRANT_RETRY_SECONDS:
            return None, None
        _qdrant_client = None
        _qdrant_failed_at = None
    if _qdrant_client is None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                Distance, VectorParams, SparseVectorParams,
                SparseIndexParams, Modifier,
                HnswConfigDiff,
                ScalarQuantization, ScalarQuantizationConfig, ScalarType,
            )

            qdrant_api_key = os.environ.get("QDRANT_API_KEY", "") or None
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key,
                                  timeout=_QDRANT_TIMEOUT)
            col = QDRANT_COLLECTION_PREFIX
            existing = {c.name for c in client.get_collections().collections}
            # An alias is an addressable name: creating a collection over one
            # fails with 400, so a name that resolves must count as existing.
            # The rename migration points loci_* at the old collections this way.
            try:
                existing |= {a.alias_name for a in client.get_aliases().aliases}
            except Exception:  # older server, or aliases unsupported — ignore
                pass

            _hnsw   = HnswConfigDiff(m=32, ef_construct=200, on_disk=False)
            _quant  = ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            )

            if col not in existing:
                client.create_collection(
                    col,
                    vectors_config={"dense": VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)},
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            index=SparseIndexParams(on_disk=False),
                            modifier=Modifier.IDF,
                        )
                    },
                    # m=32 doubles recall at high similarity thresholds; ef_construct=200 buys index quality at build time.
                    hnsw_config=_hnsw,
                    # INT8: ~4x less memory for <1% recall; always_ram keeps them hot, originals rescored on search.
                    quantization_config=_quant,
                )
                logger.info(
                    "Created Qdrant collection '%s' (named-vector + sparse + INT8 quant)", col
                )
            else:
                # update_collection is idempotent; the optimizer applies changes in the background.
                try:
                    client.update_collection(
                        col,
                        hnsw_config=_hnsw,
                        quantization_config=_quant,
                    )
                    logger.debug("Applied INT8 quant + HNSW config to existing collection '%s'", col)
                except Exception as upd_exc:
                    logger.debug("Collection config update skipped: %s", upd_exc)

            _create_payload_indexes(client, col)

            # No literal window here: pinning one at the call site is what made the default unreachable.
            _purge_old_records(client, col)

            _qdrant_client = (client, col)
        except Exception as exc:
            logger.warning("Qdrant connection failed — using Mnemo/keyword fallback: %s", exc)
            _qdrant_client = (None, None)
            _qdrant_failed_at = _time.monotonic()
    return _qdrant_client


_OLLAMA_BASE          = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_URL") or ""
_EMBED_MODEL          = os.environ.get("EMBED_MODEL", "nomic-embed-text")
_EMBED_API_KEY        = os.environ.get("EMBED_API_KEY", "")
_EMBED_API_KEY_HEADER = os.environ.get("EMBED_API_KEY_HEADER", "Authorization")
_EMBED_CACHE_MAXSIZE = 512
# Host-observed Ollama embedding latency over the tailnet is 5.595s; allow one
# full retry's worth of server-side jitter without making normal RAG calls hang
# for a round, arbitrary 30/60s window.
_EMBED_TIMEOUT = float(os.environ.get("LOCI_EMBED_TIMEOUT", "11.19"))
_embed_cache: dict[str, list[float]] = {}         # text → dense vector (bounded, FIFO eviction)
_embed_sparse_cache: dict[str, tuple] = {}        # text → (indices_tuple, values_tuple)
_embed_cache_lock = threading.Lock()              # guards _embed_cache check-evict-insert (#86)
_embed_sparse_cache_lock = threading.Lock()       # guards _embed_sparse_cache check-evict-insert (#86)
_embed_last_error = ""

# Degraded mode (keyword-only) rather than refusing to start.
if not os.environ.get("QDRANT_URL"):
    logger.warning(
        "QDRANT_URL is not set — Qdrant semantic search disabled. "
        "Set QDRANT_URL in your .env to enable vector search."
    )
if not _OLLAMA_BASE and not _EMBED_API_KEY:
    logger.warning(
        "OLLAMA_BASE_URL and EMBED_API_KEY are both unset — embedding disabled. "
        "Set OLLAMA_BASE_URL for local Ollama or EMBED_API_KEY for a cloud provider. "
        "The query path will still try backends.toml at call time."
    )


def _embed_auth_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if _EMBED_API_KEY:
        if _EMBED_API_KEY_HEADER.lower() == "authorization":
            h["Authorization"] = f"Bearer {_EMBED_API_KEY}"
        else:
            h[_EMBED_API_KEY_HEADER] = _EMBED_API_KEY
    return h


def _resolve_embed_backend() -> tuple[str, str]:
    """Resolve the embedding endpoint at call time, not import time.

    MCP launches can have QDRANT_URL in the environment while OLLAMA_BASE_URL is
    supplied only by ~/.loci/backends.toml. Capturing _OLLAMA_BASE during import
    latched that absence forever, so memory_health/backends probes could be green
    while RAG returned embedding_unavailable.
    """
    base = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_URL") or _OLLAMA_BASE
    model = os.environ.get("EMBED_MODEL") or _EMBED_MODEL or "nomic-embed-text"
    if base and model:
        return base, model
    try:
        import backends
        return base or backends.ollama_url(), model or backends.embed_model()
    except Exception:
        return base, model


def _embedding_unavailable_reason() -> str:
    return _embed_last_error or "embedding_unavailable: no dense vector returned"


def _embed(text: str) -> list[float] | None:
    """Single-text embed via OpenAI-compat /v1/embeddings.
    Works with Ollama (EMBED_API_KEY unset) and cloud providers (set EMBED_API_KEY)."""
    global _embed_last_error
    cached = _embed_cache.get(text)
    if cached is not None:
        return cached
    base, model = _resolve_embed_backend()
    if not base:
        _embed_last_error = (
            "embedding_unavailable: no embedding endpoint configured "
            "(OLLAMA_BASE_URL/OLLAMA_URL unset and backends.ollama_url() resolved empty)"
        )
        return None
    try:
        import requests as _req
        r = _req.post(
            f"{base.rstrip('/')}/v1/embeddings",
            json={"model": model, "input": [text]},
            headers=_embed_auth_headers(),
            timeout=_EMBED_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        result = list(data[0]["embedding"]) if data else None
        if result is not None:
            with _embed_cache_lock:
                if len(_embed_cache) >= _EMBED_CACHE_MAXSIZE:
                    _embed_cache.pop(next(iter(_embed_cache)))
                _embed_cache[text] = result
            _embed_last_error = ""
        else:
            _embed_last_error = (
                f"embedding_unavailable: {base.rstrip('/')}/v1/embeddings "
                f"model={model} returned no embedding data"
            )
        return result
    except Exception as exc:
        _embed_last_error = (
            f"embedding_unavailable: {base.rstrip('/')}/v1/embeddings "
            f"model={model} failed after {_EMBED_TIMEOUT}s: {exc}"
        )
        logger.warning("embed failed: %s", _embed_last_error)
        return None


def _qdrant_upsert(point_id: str, text: str, payload: dict) -> None:
    """Store a point with dense + sparse vectors. Fails silently."""
    client, col = _get_qdrant()
    if client is None:
        return
    dense_vec = _embed(text)
    if dense_vec is None:
        return
    sparse_vec = _embed_sparse(text)
    # Stamp multi-tenancy fields if not already set by the caller.
    _agent_id  = os.environ.get("HERMES_AGENT_ID", "")
    _namespace = os.environ.get("LOCI_NAMESPACE", "")
    if _agent_id and "agent_id" not in payload:
        payload = {**payload, "agent_id": _agent_id}
    if _namespace and "namespace" not in payload:
        payload = {**payload, "namespace": _namespace}
    try:
        from qdrant_client.models import PointStruct
        vector_dict: dict = {"dense": dense_vec}
        if sparse_vec is not None:
            vector_dict["sparse"] = sparse_vec
        client.upsert(
            col,
            points=[PointStruct(id=point_id, vector=vector_dict, payload=payload)],
        )
    except Exception as exc:
        logger.warning("Qdrant upsert failed — finding stored in JSONL but not indexed: %s", exc)


def _qdrant_degraded_mode(enabled: bool, available: bool, errors, query_success: bool) -> tuple[bool, str | None]:
    """Classify Qdrant degraded mode for the ``degraded_mode`` tool payload.

    Returns ``(active, reason)``.  ``errors`` is only tested for truthiness, so any
    container (or None) is accepted.
    """
    degraded_active = (not enabled) or (not available) or bool(errors) or (enabled and available and not query_success)
    degraded_reason = None
    if not enabled:
        degraded_reason = "qdrant_disabled"
    elif not available:
        degraded_reason = "qdrant_unavailable"
    elif errors:
        degraded_reason = "qdrant_semantic_error"
    elif not query_success:
        degraded_reason = "qdrant_semantic_not_executed"
    return degraded_active, degraded_reason


# Passage chars the cross-encoder sees: 1024 swept against identity recall over three seeds; 512 scored worse than no CE at all.
RERANK_MAX_CHARS = int(os.environ.get("LOCI_RERANK_MAX_CHARS", "1024"))
# Client-wide, not per-collection: sized for the slowest collection searched (5s timed out agent_core_chunks entirely).
_QDRANT_TIMEOUT = float(os.environ.get("LOCI_QDRANT_TIMEOUT", "20"))
# Overall RAG wall-clock budget. The MCP client timed out on 37-140s calls after
# expansion fanned out to 8 searches and each search paid a BGE rerank. With
# expansion off and only one final batched rerank, measured calls are a few
# seconds warm and <25s cold, so 25s leaves room for model load but bounds damage.
_RAG_DEADLINE_SECONDS = float(os.environ.get("LOCI_RAG_DEADLINE_SECONDS", "25"))
_RAG_DEADLINE_AT: float | None = None
_RAG_LAST_COLLECTION: str | None = None
_RAG_DEADLINE_LOCK = threading.Lock()

# Collection-local reranking is redundant for rag_context_search, which performs a
# single final cross-collection rerank. Profiled duplicate cost: 33.98s of a
# 37.79s call (8 BGE predictions over 50 candidates each). Leave it opt-in for
# callers that use _qdrant_search_collection directly and need pre-ranked rows.
_RAG_COLLECTION_RERANK = (
    os.environ.get("LOCI_RAG_COLLECTION_RERANK", "0").strip().lower()
    not in ("0", "false", "no", "off", "")
)
_RAG_RERANK_MIN_REMAINING = float(os.environ.get("LOCI_RAG_RERANK_MIN_REMAINING", "4"))

# rescore=True recovers the ~0.5-1% recall INT8 costs, oversampling=2.0 feeds it 2x candidates; lazy because qdrant_client is optional.
_QUANT_SEARCH_PARAMS = None


def _quant_search_params():
    """Shared SearchParams. Was hand-built at each call site, and the site that
    forgot it (retrieval_selftest) was then diagnosing a different search path
    from the one production runs."""
    global _QUANT_SEARCH_PARAMS
    if _QUANT_SEARCH_PARAMS is None:
        from qdrant_client.models import SearchParams, QuantizationSearchParams
        _QUANT_SEARCH_PARAMS = SearchParams(
            quantization=QuantizationSearchParams(rescore=True, oversampling=2.0)
        )
    return _QUANT_SEARCH_PARAMS


def _rag_deadline_start(collection_name: str | None = None) -> float:
    global _RAG_DEADLINE_AT, _RAG_LAST_COLLECTION
    now = time.monotonic()
    if _RAG_DEADLINE_SECONDS <= 0:
        return float("inf")
    with _RAG_DEADLINE_LOCK:
        # Start a new budget on the first collection search of a call, or after a
        # previous call's budget has expired. server.py owns the public tool frame,
        # so qdrant_ops cannot install a cleaner request-scoped context without
        # editing that file.
        new_default_collection = (
            collection_name == QDRANT_COLLECTION_PREFIX
            and _RAG_LAST_COLLECTION not in (None, QDRANT_COLLECTION_PREFIX)
        )
        if (
            _RAG_DEADLINE_AT is None
            or new_default_collection
            or now > _RAG_DEADLINE_AT + max(1.0, _RAG_DEADLINE_SECONDS)
        ):
            _RAG_DEADLINE_AT = now + _RAG_DEADLINE_SECONDS
        if collection_name is not None:
            _RAG_LAST_COLLECTION = collection_name
        return _RAG_DEADLINE_AT


def _rag_time_remaining() -> float:
    deadline = _rag_deadline_start()
    if deadline == float("inf"):
        return float("inf")
    return max(0.0, deadline - time.monotonic())


def _rag_deadline_error(stage: str) -> RuntimeError:
    return RuntimeError(
        f"deadline_exceeded: rag_context_search budget {_RAG_DEADLINE_SECONDS:.2f}s "
        f"exhausted before {stage}"
    )


def _rag_should_skip_rerank() -> bool:
    return (
        _RAG_DEADLINE_SECONDS > 0
        and _RAG_DEADLINE_AT is not None
        and _rag_time_remaining() < _RAG_RERANK_MIN_REMAINING
    )



def _ce_rerank(query: str, rows: list[dict], top_k: int) -> tuple[list[dict], bool]:
    """Cross-encoder rerank of ``rows`` against ``query``, truncated to ``top_k``.

    Returns ``(rows, True)`` only when the cross-encoder scored every row and the
    sort completed.  Returns ``(rows, False)`` — original bi-encoder order, still
    truncated to ``top_k`` — when the cross-encoder is unavailable or scoring
    raised, so the return count is consistent whether CE is installed or not.
    """
    if _rag_should_skip_rerank():
        logger.warning(
            "Cross-encoder reranking skipped: %.2fs remains in %.2fs RAG deadline",
            _rag_time_remaining(), _RAG_DEADLINE_SECONDS,
        )
        return rows[:top_k], False
    ce = _get_cross_encoder()
    if ce is not None and rows:
        try:
            pairs = [(query, str(r.get("text", ""))[:RERANK_MAX_CHARS]) for r in rows]
            ce_scores = ce.predict(pairs)
            for row, ce_score in zip(rows, ce_scores):
                row["ce_score"] = round(float(ce_score), 4)
            return sorted(rows, key=lambda r: r.get("ce_score", 0.0), reverse=True)[:top_k], True
        except Exception as exc:
            logger.debug("Cross-encoder reranking failed, using bi-encoder order: %s", exc)
            # All rows scored or none: strip partial ce_score annotations written before the exception.
            for row in rows:
                row.pop("ce_score", None)
    return rows[:top_k], False


def _qdrant_similarity_search(
    query: str,
    *,
    investigation_id: Optional[str] = None,
    limit: int = 10,
    rerank: bool = True,
    min_confidence: Optional[str] = None,
    rerank_top_k: Optional[int] = None,
) -> dict:
    """Hybrid dense + sparse retrieval with optional cross-encoder reranking.

    Two-stage pipeline (per arXiv production recommendations):
    Stage 1 — bi-encoder: retrieve ``limit * 5`` candidates fast via RRF fusion.
    Stage 2 — cross-encoder: if sentence-transformers is available, rerank the
    candidates by full query-passage cross-attention and return the top ``limit``.

    The cross-encoder dramatically improves precision on noisy finding sets (10-40%
    accuracy improvement in the literature) by evaluating query and passage jointly
    instead of as independent vectors. Falls back to bi-encoder scores if the
    cross-encoder is unavailable.
    """
    client, col = _get_qdrant()
    if client is None:
        return {"ok": False, "reason": "qdrant_unavailable", "results": []}

    from qdrant_client.models import (
        Filter, FieldCondition, MatchValue,
        Prefetch, FusionQuery, Fusion,
    )

    # Normalise confidence floor; callers may pass mixed-case ("High", "MEDIUM").
    if min_confidence:
        min_confidence = min_confidence.lower()
    must_conditions = []
    if investigation_id:
        must_conditions.append(FieldCondition(key="investigation_id", match=MatchValue(value=investigation_id)))
    if min_confidence and min_confidence in _CONFIDENCE_RANK:
        from qdrant_client.models import MatchAny
        allowed = [c for c, rank in _CONFIDENCE_RANK.items() if rank >= _CONFIDENCE_RANK[min_confidence]]
        must_conditions.append(FieldCondition(key="confidence", match=MatchAny(any=allowed)))
    search_filter = Filter(must=must_conditions) if must_conditions else None

    dense_vec = _embed(query)
    if dense_vec is None:
        return {"ok": False, "reason": _embedding_unavailable_reason(), "results": []}

    # rerank_top_k caps the CE batch: investigation_search inflates limit for dedup, which would otherwise mean limit*15 CE pairs.
    output_k = min(rerank_top_k, limit) if rerank_top_k is not None else limit
    fetch_limit = output_k * 5 if rerank else limit * 4

    _search_params = _quant_search_params()

    sparse_vec = _embed_sparse(query)
    if sparse_vec is not None:
        result = client.query_points(
            collection_name=col,
            prefetch=[
                Prefetch(query=dense_vec, using="dense", limit=fetch_limit * 2, filter=search_filter),
                Prefetch(query=sparse_vec, using="sparse", limit=fetch_limit * 2, filter=search_filter),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=fetch_limit,
            with_payload=True,
            search_params=_search_params,
        )
        mode = "hybrid"
    else:
        result = client.query_points(
            collection_name=col,
            query=dense_vec,
            using="dense",
            query_filter=search_filter,
            limit=fetch_limit,
            with_payload=True,
            search_params=_search_params,
        )
        mode = "semantic"

    rows = []
    for p in result.points:
        payload = dict(p.payload or {})
        rows.append({"score": round(float(p.score), 4), **payload, "origin": payload.get("origin", "qdrant")})

    if rerank:
        rows, reranked = _ce_rerank(query, rows, output_k)
        if reranked:
            mode = mode + "+reranked"
    else:
        # Honour output_k so the return count matches whether or not CE ran; mnemosyne still feeds the dedup loop.
        rows = rows[:output_k]

    return {"ok": True, "reason": mode, "results": rows}


# Vector layout is fixed under a running process; the alternative is a get_collection per query.
_dense_name_cache: dict = {}
_collection_shape_cache: dict = {}


def _dense_vector_name(client, collection: str) -> Optional[str]:
    """Name of the dense vector on ``collection``, or None when it is unnamed.

    ``_qdrant_search_collection`` and ``probe_collection`` take a collection name
    from the caller, so they run against collections this server did not create and
    whose vector is not necessarily called ``dense``. Asking for a name that is not
    there returns ``400 Not existing vector name error`` — which the retrieval path
    catches per-collection and reports as "no results", indistinguishable from a
    collection that genuinely had no match.

    Cheap insurance rather than a live fault: every collection currently in the
    store either uses ``dense`` or is unnamed. Where we create the collection
    ourselves the name is ours and the call sites keep passing it directly.

    When a collection carries several dense vectors, prefer one whose width matches
    this server's embedder; a vector we cannot produce is not a candidate.
    """
    cache_key = (id(client), collection)
    if cache_key in _dense_name_cache:
        return _dense_name_cache[cache_key]
    name = None
    try:
        vectors = client.get_collection(collection).config.params.vectors
        if isinstance(vectors, dict) and vectors:
            if "dense" in vectors:
                name = "dense"
            else:
                same_width = [k for k, v in vectors.items()
                              if getattr(v, "size", None) == VECTOR_DIM]
                name = same_width[0] if same_width else sorted(vectors)[0]
    except Exception as exc:
        logger.debug("_dense_vector_name(%s): %r", collection, exc)
    _dense_name_cache[cache_key] = name
    return name


def _collection_shape(client, name: str) -> dict:
    """Vector layout of one collection: dense width(s), sparse presence, points.

    Returns {"points", "dense_dims", "named", "sparse"} — dense_dims is a list
    because a named-vector collection can carry several dense vectors.
    """
    cache_key = (id(client), name)
    if cache_key in _collection_shape_cache:
        return dict(_collection_shape_cache[cache_key])
    info = client.get_collection(name)
    params = info.config.params
    vectors = params.vectors
    named = isinstance(vectors, dict)
    if named:
        dims = sorted({int(v.size) for k, v in vectors.items() if getattr(v, "size", None)})
    elif vectors is not None:
        dims = [int(vectors.size)]
    else:
        dims = []
    sparse = bool(getattr(params, "sparse_vectors", None))
    points = int(getattr(info, "points_count", 0) or 0)
    shape = {"points": points, "dense_dims": dims, "named": named, "sparse": sparse}
    _collection_shape_cache[cache_key] = dict(shape)
    return shape


def probe_collection(query_vec, client, name: str, limit: int = 3) -> dict:
    """Can this collection be retrieved from at all? Never raises.

    Deliberately NOT _qdrant_search_collection: that path cross-encodes and
    applies a relevance floor, so a collection of terse rows legitimately
    returns nothing and would look identical to a broken one. This asks about
    wiring, not relevance.

    status is one of:
      ok             rows came back
      empty          the collection holds no points — not a fault
      width_mismatch this server's embedder is the wrong width for it, so every
                     query it makes against this collection will fail
      no_results     queryable, nothing matched
      error          the probe raised
    """
    out = {"collection": name, "hits": 0, "status": "error", "detail": ""}
    try:
        shape = _collection_shape(client, name)
    except Exception as exc:
        out["detail"] = f"get_collection failed: {exc}"
        return out

    out.update({
        "points": shape["points"],
        "dense_dims": shape["dense_dims"],
        "sparse": shape["sparse"],
    })

    if shape["points"] == 0:
        out["status"] = "empty"
        out["detail"] = "no points stored"
        return out

    our_dim = len(query_vec) if query_vec is not None else None
    if our_dim is None:
        out["status"] = "error"
        out["detail"] = "no embedder — dense vector unavailable"
        out["remediation"] = "Set OLLAMA_BASE_URL or EMBED_API_KEY so the server can embed."
        return out

    if shape["dense_dims"] and our_dim not in shape["dense_dims"]:
        out["status"] = "width_mismatch"
        out["detail"] = (
            f"collection is {'/'.join(str(d) for d in shape['dense_dims'])}-dim, "
            f"this server embeds at {our_dim}"
        )
        out["remediation"] = (
            f"Nothing this server asks can match a {shape['dense_dims'][0]}-dim collection "
            f"while it embeds at {our_dim}. Re-embed the collection, or query it with an "
            f"embedder of its own width."
        )
        return out

    # search_params: without it the probe queries a different path from production, so green would mean nothing.
    kwargs = {"collection_name": name, "query": query_vec, "limit": limit,
              "with_payload": False, "search_params": _quant_search_params()}
    try:
        dense_name = _dense_vector_name(client, name) if shape["named"] else None
        if dense_name:
            points = client.query_points(using=dense_name, **kwargs).points
        else:
            points = client.query_points(**kwargs).points
    except Exception as exc:
        out["status"] = "error"
        out["detail"] = f"query failed: {exc}"
        return out

    out["hits"] = len(points)
    out["status"] = "ok" if points else "no_results"
    out["detail"] = f"{len(points)} row(s)" if points else "queryable, nothing matched"
    return out


def _qdrant_search_collection(
    query: str,
    collection_name: str,
    limit: int = 10,
    query_filter=None,
) -> list[dict]:
    """
    Dense + sparse (RRF) search against any named Qdrant collection.
    Falls back to dense-only when sparse vectors are unavailable.
    Returns a flat list of payload dicts with an added 'score' key.
    Raises on Qdrant errors so callers can catch per-collection failures.
    """
    _rag_deadline_start(collection_name)
    if _rag_time_remaining() <= 0:
        raise _rag_deadline_error(f"collection {collection_name}")

    client, _default_col = _get_qdrant()
    if client is None:
        raise RuntimeError("qdrant_unavailable")

    dense_vec = _embed(query)
    if dense_vec is None:
        raise RuntimeError(_embedding_unavailable_reason())
    if _rag_time_remaining() <= 0:
        raise _rag_deadline_error(f"qdrant search for {collection_name}")

    fetch_limit = limit * 5
    sparse_vec = _embed_sparse(query)

    from qdrant_client.models import Prefetch, FusionQuery, Fusion

    # Detect whether this collection uses named vectors (dense/sparse) or a flat vector.
    try:
        col_info = client.get_collection(collection_name)
        vectors_config = col_info.config.params.vectors
        has_named_vectors = isinstance(vectors_config, dict)
        has_sparse_index = has_named_vectors and "sparse" in (vectors_config or {})
        shape = _collection_shape(client, collection_name)
        dense_dims = shape.get("dense_dims") or []
        if dense_dims and len(dense_vec) not in dense_dims:
            raise RuntimeError(
                "dimension_mismatch: "
                f"collection {collection_name} dense dim(s)={dense_dims}; "
                f"embedder produced {len(dense_vec)}"
            )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"collection_unavailable: get_collection failed: {exc}") from exc

    # The dense vector is not always called "dense" — see _dense_vector_name.
    dense_name = _dense_vector_name(client, collection_name) if has_named_vectors else None
    if has_named_vectors and dense_name is None:
        has_named_vectors = False        # nothing nameable to query; fall through to flat

    _qsp = _quant_search_params()

    if has_named_vectors and has_sparse_index and sparse_vec is not None:
        result = client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(query=dense_vec, using=dense_name, limit=fetch_limit * 2),
                Prefetch(query=sparse_vec, using="sparse", limit=fetch_limit * 2),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=fetch_limit,
            with_payload=True,
            query_filter=query_filter,
            search_params=_qsp,
        )
    elif has_named_vectors:
        result = client.query_points(
            collection_name=collection_name,
            query=dense_vec,
            using=dense_name,
            limit=fetch_limit,
            with_payload=True,
            query_filter=query_filter,
            search_params=_qsp,
        )
    else:
        # Flat/unnamed vector collection (agent_core_chunks, gl_decision_library, etc.)
        result = client.query_points(
            collection_name=collection_name,
            query=dense_vec,
            limit=fetch_limit,
            with_payload=True,
            query_filter=query_filter,
            search_params=_qsp,
        )
    if _rag_time_remaining() <= 0:
        raise _rag_deadline_error(f"rerank for {collection_name}")

    rows = []
    for p in result.points:
        payload = dict(p.payload or {})
        rows.append({
            "score": round(float(p.score), 4),
            # Before the payload spread so a payload's own id wins; callers dedup on (origin, id).
            "id": str(p.id),
            **payload,
            "origin": collection_name,
        })

    if _RAG_COLLECTION_RERANK:
        rows, _ = _ce_rerank(query, rows, limit)
    else:
        rows = rows[:limit]

    return rows
