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

import asyncio
import hashlib
import json
import os
import re
import sys
import threading
import time
import uuid
from collections import Counter
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

import re as _re
_EMAIL_RE = _re.compile(r'\b[\w._%+\-]+@[\w.\-]+\.[a-zA-Z]{2,}\b', _re.I)
_HOST_RE = _re.compile(
    r'\b[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?'
    r'(?:\.[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?)*'
    r'(?:\.(?:local|corp|internal|lan|dev|test|net|com|io|org))\b', _re.I
)
_URL_RE = _re.compile(r'https?://[^\s"\' <>;]+', _re.I)


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
        except Exception:
            pass
    if _HAS_IOCEXTRACT and not out["ips"]:
        # fallback: iocextract handles defanged IPs (e.g. 198[.]51[.]100[.]1)
        try:
            out["ips"] = list(_iocextract.extract_ips(text))
        except Exception:
            pass
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
QDRANT_COLLECTION_PREFIX = os.environ.get("QDRANT_COLLECTION_PREFIX", "hermes_memory")
VECTOR_DIM = int(os.environ.get("MNEMOSYNE_EMBEDDING_DIM", 768))
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

# Confidence tier ranking — used by investigation_search and _qdrant_similarity_search.
# Defined once here to avoid the same dict appearing inline in multiple functions.
_CONFIDENCE_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2}

# ---------------------------------------------------------------------------
# Optional Qdrant + fastembed
# ---------------------------------------------------------------------------

_sparse_model = None
_cross_encoder = None          # sentence_transformers.CrossEncoder | False (permanent failure)
_cross_encoder_lock = threading.Lock()   # guards lazy init against concurrent first-callers
_qdrant_client: tuple | None = None    # (QdrantClient, collection_name) singleton
_qdrant_failed_at: float | None = None  # monotonic timestamp of last connection failure
_QDRANT_RETRY_SECONDS = 60             # backoff before retrying after a transient failure
_mnemo_remember_fn = None
_mnemo_recall_fn = None
_verdict_backend = None                # QdrantBackend for hermes_verdicts (pre_answer_check)
_verdict_backend_failed = False        # permanent-failure sentinel — don't retry


def _mnemo_bank() -> str:
    return os.environ.get("HERMES_MNEMO_BANK", "default")


def _get_mnemo_funcs() -> tuple[Any | None, Any | None]:
    global _mnemo_remember_fn, _mnemo_recall_fn
    if _mnemo_remember_fn is None or _mnemo_recall_fn is None:
        try:
            import mnemosyne as _mnemo
            _mnemo_remember_fn = getattr(_mnemo, "remember", False)
            _mnemo_recall_fn = getattr(_mnemo, "recall", False)
        except Exception as exc:
            logger.info("Mnemosyne unavailable — using JSONL/Qdrant paths: %s", exc)
            _mnemo_remember_fn = False
            _mnemo_recall_fn = False
    remember = _mnemo_remember_fn if _mnemo_remember_fn is not False else None
    recall = _mnemo_recall_fn if _mnemo_recall_fn is not False else None
    return remember, recall


def _mnemo_remember(content: str, *, importance: float = 0.6, metadata: Optional[dict] = None) -> bool:
    remember, _ = _get_mnemo_funcs()
    if remember is None or not content.strip():
        return False
    try:
        remember(
            content=content,
            source="loci-mcp",
            importance=float(max(0.0, min(importance, 1.0))),
            metadata=metadata or {},
            # Disable entity/fact extraction — Qdrant handles embedding/search.
            # These flags trigger fastembed model downloads and block for 30-60s
            # on first call in the venv, causing MCP timeouts.
            extract_entities=False,
            extract=False,
            bank=_mnemo_bank(),
        )
        return True
    except TypeError:
        # Older Mnemosyne signatures may not support bank/extract flags.
        try:
            remember(content=content, source="loci-mcp", importance=importance, metadata=metadata or {})
            return True
        except Exception as exc:
            logger.debug("Mnemo remember fallback failed: %s", exc)
            return False
    except Exception as exc:
        logger.debug("Mnemo remember failed: %s", exc)
        return False


def _coerce_mnemo_results(raw: Any) -> list[dict]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return [{"content": raw}]
    if isinstance(raw, dict):
        for key in ("results", "memories", "items", "data"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
        else:
            raw = [raw]
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if isinstance(item, str):
            out.append({"content": item})
        elif isinstance(item, dict):
            out.append(item)
    return out


def _mnemo_recall(query: str, *, top_k: int = 10, investigation_id: Optional[str] = None) -> list[dict]:
    _, recall = _get_mnemo_funcs()
    if recall is None or not query.strip():
        return []
    try:
        result = recall(query=query, top_k=max(1, min(top_k, 100)), bank=_mnemo_bank())
    except TypeError:
        try:
            result = recall(query=query, top_k=max(1, min(top_k, 100)))
        except Exception as exc:
            logger.debug("Mnemo recall fallback failed: %s", exc)
            return []
    except Exception as exc:
        logger.debug("Mnemo recall failed: %s", exc)
        return []

    rows = []
    for item in _coerce_mnemo_results(result):
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        inv_from_meta = metadata.get("investigation_id") or metadata.get("investigation")
        if investigation_id and str(inv_from_meta or "") != investigation_id:
            continue
        text = str(item.get("content") or item.get("text") or item.get("memory") or "")
        if not text:
            continue
        score = _safe_float(item.get("score", item.get("similarity", 0.0)), default=0.0)
        rows.append({
            "score": round(score, 4),
            "investigation_id": str(inv_from_meta or investigation_id or metadata.get("investigation_id") or ""),
            "record_type": str(metadata.get("record_type") or metadata.get("type") or "memory"),
            "source": str(metadata.get("source") or item.get("source") or "mnemosyne"),
            "ts": item.get("ts") or item.get("created_at"),
            "text": text,
            "origin": "mnemosyne",
        })
    return rows



def _get_sparse_embedder():
    global _sparse_model
    if _sparse_model is None:
        try:
            from fastembed import SparseTextEmbedding
            _sparse_model = SparseTextEmbedding("Qdrant/bm25", language="english", avg_len=200, disable_stemmer=True)
        except Exception as exc:
            logger.warning("SparseTextEmbedding unavailable: %s", exc)
            _sparse_model = False
    return _sparse_model if _sparse_model is not False else None


def _get_cross_encoder():
    """Lazy-init a sentence-transformers CrossEncoder for two-stage reranking.

    Uses ms-marco-MiniLM-L-6-v2 — a 22 M-param model trained on passage
    ranking, CPU-capable at <100 ms per batch of 50. Gracefully disabled when
    sentence-transformers is not installed so the rest of retrieval keeps
    working with bi-encoder scores alone.
    """
    global _cross_encoder
    if _cross_encoder is None:                        # fast path — no lock needed post-init
        with _cross_encoder_lock:                     # slow path — guard concurrent first callers
            if _cross_encoder is None:                # re-check after acquiring the lock
                try:
                    from sentence_transformers import CrossEncoder as _CE
                    _cross_encoder = _CE("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
                    logger.info("Cross-encoder reranking enabled (ms-marco-MiniLM-L-6-v2)")
                except ImportError:
                    logger.debug("sentence-transformers not installed — reranking disabled")
                    _cross_encoder = False
                except Exception as exc:
                    logger.warning("Cross-encoder init failed — reranking disabled: %s", exc)
                    _cross_encoder = False
    return _cross_encoder if _cross_encoder is not False else None


def _embed_sparse(text: str):
    """Returns a SparseVector or None."""
    model = _get_sparse_embedder()
    if model is None:
        return None
    try:
        from qdrant_client.models import SparseVector
        result = list(model.embed([text]))[0]
        return SparseVector(indices=result.indices.tolist(), values=result.values.tolist())
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
        # Multi-tenancy fields (agent_id + operator_id use is_tenant=True for
        # HNSW partition hints — same pattern as investigation_id)
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
        # Entity fields — enable O(log N) indexed lookup vs. full collection scan.
        # Qdrant indexes array elements individually so MatchValue on a list field
        # matches any element in the array (standard inverted-index behaviour).
        # Note: dot-notation indexing of arrays inside nested JSON objects
        # (entities.ips is an array inside the "entities" dict) is an implicit
        # behaviour of qdrant-client 1.17.x — not formally documented in the API
        # spec.  Works at this version but should be re-verified on upgrade.
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


def _purge_old_records(client, col: str, retention_days: int = 30) -> None:
    """Delete records older than retention_days. Requires created_at_ts payload index."""
    from qdrant_client.models import Filter, FieldCondition, Range, FilterSelector
    cutoff = int(time.time()) - (retention_days * 86400)
    try:
        client.delete(
            collection_name=col,
            points_selector=FilterSelector(
                filter=Filter(must=[
                    FieldCondition(key="created_at_ts", range=Range(lt=cutoff))
                ])
            ),
            wait=False,
        )
        logger.info("Qdrant TTL purge: deleted records older than %d days", retention_days)
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
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=5)
            col = QDRANT_COLLECTION_PREFIX
            existing = {c.name for c in client.get_collections().collections}

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
                    # Stronger HNSW graph: m=32 doubles recall at high similarity
                    # thresholds; ef_construct=200 improves index quality at build time.
                    hnsw_config=_hnsw,
                    # INT8 scalar quantization: ~4x memory reduction, <1% recall loss.
                    # always_ram keeps quantized vectors hot; originals rescored on search.
                    quantization_config=_quant,
                )
                logger.info(
                    "Created Qdrant collection '%s' (named-vector + sparse + INT8 quant)", col
                )
            else:
                # Upgrade existing collection: apply quantization + HNSW if not configured.
                # update_collection is idempotent; the optimizer applies changes in the
                # background without interrupting reads or writes.
                try:
                    client.update_collection(
                        col,
                        hnsw_config=_hnsw,
                        quantization_config=_quant,
                    )
                    logger.debug("Applied INT8 quant + HNSW config to existing collection '%s'", col)
                except Exception as upd_exc:
                    logger.debug("Collection config update skipped: %s", upd_exc)

            # Create payload indexes — idempotent, safe on existing collection
            _create_payload_indexes(client, col)

            # Purge records older than 30 days on startup
            _purge_old_records(client, col, retention_days=30)

            _qdrant_client = (client, col)
        except Exception as exc:
            logger.warning("Qdrant connection failed — using Mnemo/keyword fallback: %s", exc)
            _qdrant_client = (None, None)
            _qdrant_failed_at = _time.monotonic()
    return _qdrant_client


_OLLAMA_BASE          = os.environ.get("OLLAMA_BASE_URL")
_EMBED_MODEL          = os.environ.get("EMBED_MODEL", "nomic-embed-text")
_EMBED_API_KEY        = os.environ.get("EMBED_API_KEY", "")
_EMBED_API_KEY_HEADER = os.environ.get("EMBED_API_KEY_HEADER", "Authorization")
_EMBED_BATCH_SIZE = 32  # Ollama stalls on >32

# Startup validation — warn clearly when required backends are not configured.
# Server runs in degraded mode (keyword-only) rather than refusing to start.
if not os.environ.get("QDRANT_URL"):
    logger.warning(
        "QDRANT_URL is not set — Qdrant semantic search disabled. "
        "Set QDRANT_URL in your .env to enable vector search."
    )
if not _OLLAMA_BASE and not _EMBED_API_KEY:
    logger.warning(
        "OLLAMA_BASE_URL and EMBED_API_KEY are both unset — embedding disabled. "
        "Set OLLAMA_BASE_URL for local Ollama or EMBED_API_KEY for a cloud provider."
    )


def _embed_auth_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if _EMBED_API_KEY:
        if _EMBED_API_KEY_HEADER.lower() == "authorization":
            h["Authorization"] = f"Bearer {_EMBED_API_KEY}"
        else:
            h[_EMBED_API_KEY_HEADER] = _EMBED_API_KEY
    return h

# Optional extra collection for code-chunk correlation (set CODE_CHUNKS_COLLECTION
# to the name of a Qdrant collection that holds code embeddings).
_CODE_CHUNKS_COLLECTION = os.environ.get("CODE_CHUNKS_COLLECTION", "")



def _embed(text: str) -> list[float] | None:
    """Single-text embed via OpenAI-compat /v1/embeddings.
    Works with Ollama (EMBED_API_KEY unset) and cloud providers (set EMBED_API_KEY)."""
    try:
        import requests as _req
        r = _req.post(
            f"{_OLLAMA_BASE.rstrip('/')}/v1/embeddings",
            json={"model": _EMBED_MODEL, "input": [text]},
            headers=_embed_auth_headers(),
            timeout=10,
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        return list(data[0]["embedding"]) if data else None
    except Exception as exc:
        logger.warning("embed failed: %s", exc)
        return None


_EMBED_BACKEND: str | None = None   # "ollama" | "fastembed" | None
_FE_MODEL = None                    # lazy fastembed TextEmbedding instance


def _embed_batch(texts: list[str]) -> list[list[float] | None]:
    """Batch embed. Tries Ollama /v1/embeddings first; falls back to fastembed.
    Hard ceiling: _EMBED_BATCH_SIZE (32) items per call.
    Returns one vector (or None) per input text, preserving order.
    """
    global _EMBED_BACKEND, _FE_MODEL
    if not texts:
        return []
    import requests as _req
    results: list[list[float] | None] = []
    for i in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[i:i + _EMBED_BATCH_SIZE]
        vecs = None

        # --- Try Ollama /v1/embeddings (OpenAI-compatible) ---
        if _EMBED_BACKEND != "fastembed" and _OLLAMA_BASE:
            try:
                r = _req.post(
                    f"{_OLLAMA_BASE.rstrip('/')}/v1/embeddings",
                    json={"model": _EMBED_MODEL, "input": batch},
                    headers=_embed_auth_headers(),
                    timeout=30,
                )
                r.raise_for_status()
                data = r.json()
                vecs = [item["embedding"] for item in data.get("data", [])]
                if vecs:
                    _EMBED_BACKEND = "ollama"
            except Exception as exc:
                logger.warning("Ollama /v1/embeddings failed, falling back to fastembed: %s", exc)
                _EMBED_BACKEND = "fastembed"

        # --- fastembed fallback (CPU, BAAI/bge-small-en-v1.5, 384-dim) ---
        if vecs is None:
            fe_model_name = os.environ.get("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
            if int(os.environ.get("EMBED_DIM", VECTOR_DIM)) != 384:
                logger.warning(
                    "fastembed fallback produces 384-dim but VECTOR_DIM=%d"
                    " — upserts may fail. Set EMBED_DIM=384 or restore Ollama.",
                    VECTOR_DIM,
                )
            try:
                from fastembed import TextEmbedding
                if _FE_MODEL is None:
                    _FE_MODEL = TextEmbedding(model_name=fe_model_name)
                vecs = [v.tolist() for v in _FE_MODEL.embed(batch)]
                _EMBED_BACKEND = "fastembed"
                logger.info("Using fastembed fallback (%s)", fe_model_name)
            except Exception as exc:
                logger.error("fastembed fallback also failed: %s", exc)
                vecs = []

        for j in range(len(batch)):
            results.append(list(vecs[j]) if vecs and j < len(vecs) else None)
    return results


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


def _get_verdict_backend():
    """Lazy QdrantBackend for the hermes_verdicts collection (pre_answer_check verdicts).

    Reuses the same Qdrant instance as investigations but in a separate collection
    so claim-check history never pollutes finding storage. Fail-open: returns None
    when Qdrant is unavailable or memcheck is not importable.
    """
    global _verdict_backend, _verdict_backend_failed
    if _verdict_backend_failed:
        return None
    if _verdict_backend is not None:
        return _verdict_backend
    qdrant_url = os.environ.get("QDRANT_URL", "")
    if not qdrant_url:
        return None
    try:
        from memcheck.qdrant import QdrantBackend
        from qdrant_client import QdrantClient
        _vb_api_key = os.environ.get("QDRANT_API_KEY", "") or None
        client = QdrantClient(url=qdrant_url, api_key=_vb_api_key, timeout=5)
        _verdict_backend = QdrantBackend(
            client,
            collection="hermes_verdicts",
            embed=_embed,
            vector_name="dense",
        )
        return _verdict_backend
    except Exception as exc:
        logger.debug("Verdict backend unavailable: %s", exc)
        _verdict_backend_failed = True
        return None


def _record_claim_verdicts(
    investigation_id: str,
    claim_results: list[dict],
    *,
    record: bool,
) -> dict:
    """Record a verdict per claim to hermes_verdicts and annotate claim_results in-place.

    Each claim result gains three fields: ``verdict_type`` (claim_supported /
    claim_contradicted / claim_unsupported), ``prior_occurrences`` (how many
    times this exact claim was checked before in this investigation), and
    ``verdict_conflict`` (True when the current verdict contradicts the most
    recent prior verdict — e.g. was supported before, now contradicted).
    All steps are fail-open. Returns a summary dict.
    """
    if not record:
        for cr in claim_results:
            cr.update({"verdict_type": None, "prior_occurrences": 0, "verdict_conflict": False})
        return {"recorded": 0, "qdrant": "disabled"}

    backend = _get_verdict_backend()
    if backend is None:
        for cr in claim_results:
            cr.update({"verdict_type": None, "prior_occurrences": 0, "verdict_conflict": False})
        return {"recorded": 0, "qdrant": "unavailable"}

    from memcheck.verdict import Verdict, make_signature, new_verdict, redact_excerpt

    _VERDICT_MAP = {
        "claim_ambiguous":    ("warn",  0.75, "claim has supporting evidence but cross-investigation benign baseline also exists — disambiguation required"),
        "claim_contradicted": ("flag",  0.90, "claim contradicted by negation-mismatch evidence"),
        "claim_supported":    ("allow", 0.85, "claim supported by investigation evidence"),
        "claim_unsupported":  ("warn",  0.70, "no supporting evidence found in investigation"),
    }

    # PE-gated reconsolidation (Nader 2000 / Sevenster 2013).
    # Verdict severity order: supported(1) < ambiguous(2) < unsupported(3) < contradicted(4).
    # Prediction error = |new_severity - prior_severity| / 3. High PE on an
    # established prior → verdict is provisional (recorded, not enforced) until
    # a second independent observation confirms the direction change.
    _VERDICT_SEVERITY = {
        "claim_supported": 1, "claim_ambiguous": 2,
        "claim_unsupported": 3, "claim_contradicted": 4,
    }
    _PE_HIGH_THRESH = float(os.environ.get("HERMES_PE_HIGH_THRESH", "0.5"))
    _PE_PROTECTION_MIN_OCC = int(os.environ.get("HERMES_PE_PROTECTION_MIN_OCC", "3"))

    recorded = 0
    qdrant_ok = True

    async def _process() -> None:
        nonlocal recorded, qdrant_ok
        for cr in claim_results:
            claim = str(cr.get("claim", ""))
            if cr.get("contradicted"):
                vtype = "claim_contradicted"
            elif cr.get("ambiguous"):
                # Supported but cross-investigation benign baseline also present.
                # Record as ambiguous so the signal survives in verdict history.
                vtype = "claim_ambiguous"
            elif cr.get("supported"):
                vtype = "claim_supported"
            else:
                vtype = "claim_unsupported"

            sig = make_signature("claim_check", f"{investigation_id}:{claim}")
            decision, confidence, rationale = _VERDICT_MAP[vtype]

            # Recall prior verdict by exact point-id to detect conflicts and count history.
            prior_vtype: Optional[str] = None
            prior_count: int = 0
            try:
                pid_fn = getattr(backend, "point_id", None)
                retrieve_fn = getattr(getattr(backend, "_client", None), "retrieve", None)
                if callable(pid_fn) and callable(retrieve_fn):
                    pid = pid_fn(sig)
                    hits = await asyncio.to_thread(
                        retrieve_fn,
                        collection_name="hermes_verdicts",
                        ids=[pid],
                        with_payload=True,
                    )
                    if hits:
                        pl = getattr(hits[0], "payload", None)
                        if pl:
                            prior = Verdict.from_payload(dict(pl))
                            prior_vtype = prior.verdict_type
                            prior_count = prior.occurrences
            except Exception as exc:
                logger.debug("Verdict recall failed for claim %r: %s", claim[:60], exc)

            cr["verdict_type"] = vtype
            cr["prior_occurrences"] = prior_count
            # Flag a conflict whenever the verdict transitions into or out of a
            # "warning" state (contradicted or ambiguous).  Transitions between
            # supported↔ambiguous matter: a claim that was clean and now has a
            # benign baseline (or had one and now appears clean) deserves scrutiny.
            cr["verdict_conflict"] = bool(
                prior_vtype and prior_vtype != vtype and (
                    vtype in ("claim_contradicted", "claim_ambiguous") or
                    prior_vtype in ("claim_contradicted", "claim_ambiguous")
                )
            )

            refs = [
                str(r.get("evidence_id", ""))
                for r in cr.get("support_refs", [])
                if r.get("evidence_id")
            ][:5]

            # PE-gated reconsolidation: measure direction change against prior.
            provisional = False
            if prior_vtype and prior_vtype != vtype:
                prior_sev = _VERDICT_SEVERITY.get(prior_vtype, 2)
                new_sev   = _VERDICT_SEVERITY.get(vtype, 2)
                pe = abs(new_sev - prior_sev) / 3.0
                if pe >= _PE_HIGH_THRESH and prior_count >= _PE_PROTECTION_MIN_OCC:
                    provisional = True
                    rationale = (
                        rationale
                        + f" [PROVISIONAL: PE={pe:.2f}, prior={prior_vtype}×{prior_count},"
                        " requires second confirmation before enforcement]"
                    )
                    logger.debug(
                        "PE-provisional verdict for claim %r: PE=%.2f prior=%s×%d",
                        claim[:60], pe, prior_vtype, prior_count,
                    )

            v = new_verdict(
                subject_kind="memory",
                subject_signature=sig,
                subject_excerpt=redact_excerpt(f"{investigation_id}: {claim}"),
                verdict_type=vtype,
                decision=decision,
                confidence=confidence,
                rationale=rationale,
                source="rule",
                refs=refs,
                provisional=provisional,
            )
            try:
                await backend.record(v)
                recorded += 1
            except Exception as exc:
                logger.debug("Verdict record failed for claim %r: %s", claim[:60], exc)
                qdrant_ok = False

    # FastMCP dispatches sync @mcp.tool() functions inline on the running event
    # loop (fn(**kwargs), no executor).  asyncio.run() requires *no* running loop
    # and raises RuntimeError when one already exists.  Detect the situation and
    # delegate to a fresh thread that owns its own event loop instead.
    try:
        asyncio.get_running_loop()
        # A loop IS running — we are being called from a sync tool on the loop
        # thread (FastMCP inline dispatch).  Delegate to a daemon thread that
        # owns its own event loop; threading.Thread is lighter than a full
        # ThreadPoolExecutor for a one-shot fire-and-join.
        _exc: list[Exception] = []

        def _run() -> None:
            try:
                asyncio.run(_process())
            except Exception as e:
                _exc.append(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join()
        if _exc:
            raise _exc[0]
    except RuntimeError:
        # No running loop — safe to call asyncio.run() directly.
        try:
            asyncio.run(_process())
        except Exception as exc:
            logger.debug("_record_claim_verdicts failed: %s", exc)
            qdrant_ok = False
    except Exception as exc:
        logger.debug("_record_claim_verdicts failed: %s", exc)
        qdrant_ok = False

    return {"recorded": recorded, "qdrant": "ok" if qdrant_ok else "partial"}


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
    # Build filter: investigation scope + optional confidence floor.
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
        return {"ok": False, "reason": "embedding_unavailable", "results": []}

    # rerank_top_k separates "how many CE-ranked results to return" from "how
    # many candidates to fetch for dedup".  investigation_search inflates limit
    # to compensate for dedup losses; without rerank_top_k that inflation would
    # multiply the CE batch size (limit*3 caller → limit*15 CE pairs).
    # Clamp to limit so the function never returns more rows than requested.
    output_k = min(rerank_top_k, limit) if rerank_top_k is not None else limit
    fetch_limit = output_k * 5 if rerank else limit * 4

    from qdrant_client.models import SearchParams, QuantizationSearchParams
    # rescore=True: after ANN candidate selection from quantized index, re-score
    # with original full-precision vectors. Recovers ~0.5-1% recall lost to INT8.
    # oversampling fetches 2x candidates to give rescore more to work with.
    _search_params = SearchParams(
        quantization=QuantizationSearchParams(rescore=True, oversampling=2.0)
    )

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

    # Stage 2: cross-encoder reranking — full query-passage joint scoring.
    ce = _get_cross_encoder() if rerank else None
    if ce is not None and rows:
        try:
            pairs = [(query, str(r.get("text", ""))[:512]) for r in rows]
            ce_scores = ce.predict(pairs)
            for row, ce_score in zip(rows, ce_scores):
                row["ce_score"] = round(float(ce_score), 4)
            rows = sorted(rows, key=lambda r: r.get("ce_score", 0.0), reverse=True)[:output_k]
            mode = mode + "+reranked"
        except Exception as exc:
            logger.debug("Cross-encoder reranking failed, using bi-encoder order: %s", exc)
            # Strip any partial ce_score annotations written before the exception
            # so downstream consumers see a consistent payload (all rows scored,
            # or none — never a mix).
            for row in rows:
                row.pop("ce_score", None)
            rows = rows[:output_k]
    else:
        # CE not available — honour output_k so the function's return count is
        # consistent whether CE is installed or not.  investigation_search still
        # gets enough dedup candidates from mnemosyne (up to limit*4 rows) so
        # capping here at output_k does not starve the dedup loop.
        rows = rows[:output_k]

    return {"ok": True, "reason": mode, "results": rows}


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
    client, _default_col = _get_qdrant()
    if client is None:
        raise RuntimeError("qdrant_unavailable")

    dense_vec = _embed(query)
    if dense_vec is None:
        raise RuntimeError("embedding_unavailable")

    fetch_limit = limit * 5
    sparse_vec = _embed_sparse(query)

    from qdrant_client.models import Prefetch, FusionQuery, Fusion, SearchParams, QuantizationSearchParams

    # Detect whether this collection uses named vectors (dense/sparse) or a flat vector.
    try:
        col_info = client.get_collection(collection_name)
        vectors_config = col_info.config.params.vectors
        has_named_vectors = isinstance(vectors_config, dict)
    except Exception:
        has_named_vectors = False

    _qsp = SearchParams(quantization=QuantizationSearchParams(rescore=True, oversampling=2.0))

    if has_named_vectors and sparse_vec is not None:
        result = client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(query=dense_vec, using="dense", limit=fetch_limit * 2),
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
            using="dense",
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

    rows = []
    for p in result.points:
        payload = dict(p.payload or {})
        rows.append({
            "score": round(float(p.score), 4),
            **payload,
            "origin": collection_name,
        })

    # Cross-encoder rerank if available
    ce = _get_cross_encoder()
    if ce is not None and rows:
        try:
            pairs = [(query, str(r.get("text", ""))[:512]) for r in rows]
            ce_scores = ce.predict(pairs)
            for row, ce_score in zip(rows, ce_scores):
                row["ce_score"] = round(float(ce_score), 4)
            rows = sorted(rows, key=lambda r: r.get("ce_score", 0.0), reverse=True)[:limit]
        except Exception as exc:
            logger.debug("_qdrant_search_collection: CE rerank failed: %s", exc)
            for row in rows:
                row.pop("ce_score", None)
            rows = rows[:limit]
    else:
        rows = rows[:limit]

    return rows


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

    entity_field_map = {
        "ips":       "entities.ips",
        "emails":    "entities.emails",
        "hostnames": "entities.hostnames",
        "hashes":    "entities.hashes",
        "cves":      "entities.cves",
    }

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
    return [
        {
            "finding_id": f.get("id"),
            "investigation_id": f.get("investigation_id"),
            "ts": f.get("ts"),
            "record_type": f.get("record_type") or f.get("type"),
            "confidence": f.get("confidence"),
            "source": f.get("source"),
            "text": str(f.get("text", ""))[:300],
        }
        for f in findings
    ]


# ---------------------------------------------------------------------------
# Entity lookup helpers
# ---------------------------------------------------------------------------

_ENTITY_FIELD_MAP = {
    "ip":       "entities.ips",
    "email":    "entities.emails",
    "hostname": "entities.hostnames",
    "hash":     "entities.hashes",
    "cve":      "entities.cves",
}

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
    # Explicit map avoids "hash" + "s" = "hashs" (should be "hashes").
    _PLURAL = {"ip": "ips", "email": "emails", "hostname": "hostnames",
               "hash": "hashes", "cve": "cves"}
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


def _summarise_finding(f: dict) -> dict:
    """Compact finding summary for entity-lookup results."""
    return {
        "finding_id": f.get("id"),
        "investigation_id": f.get("investigation_id"),
        "ts": f.get("ts"),
        "record_type": f.get("record_type") or f.get("type"),
        "confidence": f.get("confidence"),
        "source": f.get("source"),
        "text": str(f.get("text", ""))[:300],
        "tags": f.get("tags", []),
    }


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _inv_dir(investigation_id: str) -> Path:
    d = MEMORY_DIR / investigation_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_manifest(investigation_id: str) -> dict | None:
    p = MEMORY_DIR / investigation_id / "manifest.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _save_manifest(manifest: dict) -> None:
    manifest["updated_at"] = _now()
    p = _inv_dir(manifest["id"]) / "manifest.json"
    p.write_text(json.dumps(manifest, indent=2))


def _append_jsonl(path: Path, entry: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


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
    import tempfile
    REFLECTION_STATE_DIR.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _now()
    tmp_fd, tmp_path = tempfile.mkstemp(dir=REFLECTION_STATE_DIR, prefix=".reflection_tmp_")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(json.dumps(state, indent=2))
        os.replace(tmp_path, REFLECTION_STATE_FILE)
    except Exception:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise


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

    event_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    lines_scanned = 0
    bytes_scanned = 0
    sampling_mode = "full"

    def _scan_line(line: str) -> None:
        canon = _canonicalize_reflection_signature(line)
        if _REFLECTION_ERROR_RE.search(line):
            error_counts[canon] += 1
        elif _REFLECTION_WARN_RE.search(line):
            warning_counts[canon] += 1

    if kind == "temp_ingest":
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        bytes_scanned += len(content.encode("utf-8", errors="ignore"))
        lines_scanned = 1
        try:
            payload = json.loads(content)
            if isinstance(payload, dict):
                for key in list(payload.keys())[:30]:
                    event_counts[f"payload_key:{key}"] += 1
            _scan_line(json.dumps(payload)[:20000])
        except Exception:
            _scan_line(content[:20000])
    elif kind == "session_event":
        with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                if lines_scanned >= max_lines:
                    break
                lines_scanned += 1
                bytes_scanned += len(raw.encode("utf-8", errors="ignore"))
                line = raw.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    _scan_line(line)
                    continue
                event_type = str(event.get("type") or event.get("event") or "unknown")
                event_counts[event_type] += 1
                tool_name = event.get("tool_start_name") or event.get("tool_complete_name") or event.get("tool_name")
                if tool_name:
                    tool_counts[str(tool_name)] += 1
                joined = " ".join([
                    str(event.get("user_content") or ""),
                    str(event.get("assistant_content") or ""),
                    str(event.get("tool_complete_result_content") or ""),
                ])
                _scan_line(joined)
    elif kind == "process_log":
        file_bytes = int(file_path.stat().st_size)
        if file_bytes > REFLECTION_LOG_TAIL_MIN_FILE_BYTES:
            sampling_mode = "tail"
            for raw in _read_tail_lines(
                file_path,
                max_lines=max_lines,
                max_bytes=REFLECTION_LOG_TAIL_READ_BYTES,
            ):
                lines_scanned += 1
                bytes_scanned += len(raw.encode("utf-8", errors="ignore"))
                _scan_line(raw)
                m = re.search(r"\btool(?:Name)?[=:\"]+([a-zA-Z0-9_.:-]+)", raw)
                if m:
                    tool_counts[m.group(1)] += 1
        else:
            with file_path.open("r", encoding="utf-8", errors="ignore") as fh:
                for raw in fh:
                    if lines_scanned >= max_lines:
                        break
                    lines_scanned += 1
                    bytes_scanned += len(raw.encode("utf-8", errors="ignore"))
                    _scan_line(raw)
                    m = re.search(r"\btool(?:Name)?[=:\"]+([a-zA-Z0-9_.:-]+)", raw)
                    if m:
                        tool_counts[m.group(1)] += 1
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
        "lines_scanned": lines_scanned,
        "bytes_scanned": bytes_scanned,
        "sampling_mode": sampling_mode,
        "events": dict(event_counts.most_common(8)),
        "tools": dict(tool_counts.most_common(8)),
        "errors": dict(error_counts.most_common(8)),
        "warnings": dict(warning_counts.most_common(8)),
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


def _load_retracted_ids(investigation_id: str) -> set[str]:
    """Fold ``retractions.jsonl`` into the set of currently-retracted finding ids.

    A finding is retracted iff its id has an ``active:true`` retraction with no
    later ``active:false`` (restore) entry. The log is append-only, so we replay
    it in order and the last entry per finding id wins. Fail-safe: a missing or
    malformed log yields an empty set, never raises.
    """
    path = _inv_dir(investigation_id) / "retractions.jsonl"
    state: dict[str, bool] = {}
    for entry in _read_jsonl(path):
        if not isinstance(entry, dict):
            continue
        fid = entry.get("finding_id")
        if not fid:
            continue
        state[str(fid)] = bool(entry.get("active", True))
    return {fid for fid, active in state.items() if active}


_NEGATION_RE = re.compile(r"\b(?:no|not|never|none|without|cannot|can't|didn't|isn't|aren't|won't)\b", re.I)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9._:/-]{2,}", re.I)
_GENERIC_MATCH_TOKENS = {
    "host", "user", "device", "query", "result", "results", "output", "input", "tool",
    "found", "seen", "shows", "reported", "detected", "contacted", "event", "events",
    "record", "records", "row", "rows",
}
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
        except Exception:
            pass

    lines = [line.strip(" -\t") for line in raw.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    return [raw]


def _confidence_allowed(confidence: str, min_confidence: str) -> bool:
    return _CONFIDENCE_RANK.get(confidence, -1) >= _CONFIDENCE_RANK.get(min_confidence, -1)


def tokenize(text: str) -> set[str]:
    return {
        token for token in (m.group(0).lower() for m in _TOKEN_RE.finditer(text or ""))
        if token not in _GENERIC_MATCH_TOKENS
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


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _make_ref(record: dict, match_type: str, score: float | None = None) -> dict:
    ref = {
        "evidence_id": record.get("evidence_id"),
        "record_type": record.get("record_type"),
        "source": record.get("source"),
        "ts": record.get("ts"),
        "origin": record.get("origin"),
        "snippet": record.get("snippet", ""),
        "match_type": match_type,
    }
    if score is not None:
        ref["score"] = round(score, 4)
    return ref


# ---------------------------------------------------------------------------
# Memory self-check helpers (advisory-only; never mutate/delete findings)
# ---------------------------------------------------------------------------

def _tag_finding_ids(findings: list[dict], investigation_id: str) -> list[dict]:
    """Return shallow copies of findings each carrying a stable ``id``.

    Reuses an existing id field when present, else derives
    ``f"{investigation_id}:{index}"``. Findings on disk are never mutated — this
    only annotates the in-memory copies the checks operate on.
    """
    tagged: list[dict] = []
    for index, f in enumerate(findings or []):
        if not isinstance(f, dict):
            continue
        fid = f.get("id") or f.get("finding_id") or f"{investigation_id}:{index}"
        tagged.append({**f, "id": str(fid)})
    return tagged


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
            if other in unsupported_ids:
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

@mcp.tool()
def investigation_start(
    investigation_id: str,
    title: str,
    context: Optional[str] = None,
) -> str:
    """
    Create or resume an investigation. Call at the start of any session to
    initialize the manifest. Idempotent — resuming an existing ID returns
    the current manifest without overwriting it.

    Args:
        investigation_id: Short identifier — ticket number, case ID, or a
                          descriptive slug (e.g. "RQ41919026", "pww-actor-2026").
        title: One-line description of the investigation.
        context: Optional background to record on first creation only.

    Returns:
        JSON: {"status": "created"|"resumed", "manifest": {id, title, context, status,
               created_at, updated_at, hypothesis, open_questions, next_step,
               checked_sources, finding_counts, closed_at, closed_summary}}

        The investigation ID is at result["manifest"]["id"], NOT result["investigation_id"].
        Example extraction: inv_id = json.loads(result)["manifest"]["id"]
    """
    existing = _load_manifest(investigation_id)
    if existing:
        return json.dumps({"status": "resumed", "manifest": existing}, indent=2)

    manifest = {
        "id": investigation_id,
        "title": title,
        "context": context or "",
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
    logger.info("Created investigation %s", investigation_id)
    return json.dumps({"status": "created", "manifest": manifest}, indent=2)


# ---- Tool: investigation_load ----

@mcp.tool()
def investigation_load(
    investigation_id: str,
    last_n_findings: int = 20,
    include_retracted: bool = False,
) -> str:
    """
    Retrieve manifest and recent findings for an investigation.
    Use at session start to recover context without re-running all previous
    tool calls. The manifest contains hypothesis, open questions, checked
    sources, and next step — everything needed to resume cleanly.

    Soft-tombstoned (retracted) findings are excluded by default so a known
    hallucination and its contaminated lineage don't re-enter recall. The data
    is never lost — pass ``include_retracted=True`` to see them, and the count
    of excluded findings is always reported as ``excluded_retracted``.

    Args:
        investigation_id: Investigation identifier.
        last_n_findings: How many recent findings to include (default 20).
        include_retracted: Include soft-retracted findings (default False).

    Returns:
        JSON with manifest, total finding count, recent findings, and
        ``excluded_retracted`` (count of findings filtered out).
    """
    manifest = _load_manifest(investigation_id)
    if not manifest:
        return json.dumps({
            "error": f"Investigation '{investigation_id}' not found. Call investigation_start first."
        })

    findings = _read_jsonl(_inv_dir(investigation_id) / "findings.jsonl")
    all_retracted = _load_retracted_ids(investigation_id)
    total_retracted = len(all_retracted)
    retracted = set() if include_retracted else all_retracted
    excluded_retracted = 0
    if retracted:
        kept = [f for f in findings if str(f.get("id", "")) not in retracted]
        excluded_retracted = len(findings) - len(kept)
        findings = kept
    recent = findings[-last_n_findings:]

    return json.dumps({
        "manifest": manifest,
        "total_findings": len(findings),
        "recent_findings": recent,
        "excluded_retracted": excluded_retracted,
        "total_retracted": total_retracted,
        "include_retracted": include_retracted,
    }, indent=2)


# ---- Tool: investigation_store ----

@mcp.tool()
def investigation_store(
    investigation_id: str,
    finding_type: str,
    text: str,
    source: str,
    confidence: str = "medium",
    tags: Optional[str] = None,
    derived_from: str | list[str] | None = None,
) -> str:
    """
    Record a finding in the investigation.

    Args:
        investigation_id: Investigation identifier.
        finding_type: One of: observed, inferred, assumed, gap.
                      observed  — from a direct tool response; cite source and key values.
                      inferred  — reasoned from observations but not directly stated.
                      assumed   — working hypothesis with no current evidence.
                      gap       — something that should be checked but hasn't been.
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

    Returns:
        JSON: {"stored": true, "finding_id": "<uuid>", "type": "<finding_type>",
               "mnemo_stored": true|false}
        On error: {"error": "<message>"}

    Note on arg name: the second positional parameter is ``finding_type``, NOT
    ``record_type``.  Call as:
        investigation_store(inv_id, "observed", "text", "source", "high")
    """
    manifest = _load_manifest(investigation_id)
    if not manifest:
        return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

    if finding_type not in {"observed", "inferred", "assumed", "gap"}:
        return json.dumps({"error": "finding_type must be one of: observed, inferred, assumed, gap"})
    if confidence not in {"high", "medium", "low"}:
        return json.dumps({"error": "confidence must be one of: high, medium, low"})

    finding = {
        "id": str(uuid.uuid4()),
        "investigation_id": investigation_id,
        "ts": _now(),
        "created_at_ts": int(datetime.now(timezone.utc).timestamp()),
        "record_type": finding_type,   # "observed" | "inferred" | "assumed" | "gap"
        "type": finding_type,          # kept for backwards compat with existing JSONL
        "text": text,
        "source": source,
        "confidence": confidence,
        "tags": [t.strip() for t in (
            ",".join(tags) if isinstance(tags, list) else (tags or "")
        ).split(",") if t.strip()],
    }
    derived = _normalize_derived_from(derived_from)
    if derived:
        existing_ids = {f["id"] for f in _read_jsonl(_inv_dir(investigation_id) / "findings.jsonl") if "id" in f}
        unknown = [pid for pid in derived if pid not in existing_ids]
        if unknown:
            return json.dumps({"error": f"derived_from contains unknown parent id(s): {unknown}. Verify the parent findings exist before linking."})
        finding["derived_from"] = derived
    finding["entities"] = _extract_entities(text)

    _append_jsonl(_inv_dir(investigation_id) / "findings.jsonl", finding)

    manifest["finding_counts"][finding_type] = manifest["finding_counts"].get(finding_type, 0) + 1
    _save_manifest(manifest)

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

    _qdrant_upsert(finding["id"], text, finding)

    return json.dumps({
        "stored": True,
        "finding_id": finding["id"],
        "type": finding_type,
        "mnemo_stored": mnemo_stored,
    }, indent=2)


# ---- Tool: investigation_note ----

@mcp.tool()
def investigation_note(
    investigation_id: str,
    field: str,
    value: str,
) -> str:
    """
    Update a manifest field for the investigation. Use to track the working
    hypothesis, next action, open questions, and which sources have been checked.

    Args:
        investigation_id: Investigation identifier.
        field: One of:
               context             — overwrite the investigation context (corrects stale
                                     framing set at creation time).
               hypothesis          — current working hypothesis (overwrite).
               next_step           — recommended next action (overwrite).
               open_question_add   — append a question to the open list.
               open_question_remove — remove a question from the open list.
               checked_source      — mark a source as checked; format as
                                     "tool_name: one-line summary of what was found".
               closed_summary      — close the investigation with a final summary.

    Returns:
        JSON with the updated manifest.
    """
    manifest = _load_manifest(investigation_id)
    if not manifest:
        return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

    if field == "context":
        manifest["context"] = value
    elif field == "hypothesis":
        manifest["hypothesis"] = value
    elif field == "next_step":
        manifest["next_step"] = value
    elif field == "open_question_add":
        if value not in manifest["open_questions"]:
            manifest["open_questions"].append(value)
    elif field == "open_question_remove":
        manifest["open_questions"] = [q for q in manifest["open_questions"] if q != value]
    elif field == "checked_source":
        parts = value.split(":", 1)
        tool = parts[0].strip()
        summary = parts[1].strip() if len(parts) > 1 else ""
        manifest["checked_sources"][tool] = {"summary": summary, "ts": _now()}
    elif field == "closed_summary":
        manifest["closed_summary"] = value
        manifest["status"] = "closed"
        manifest["closed_at"] = _now()
    else:
        return json.dumps({
            "error": (
                f"Unknown field '{field}'. Valid: context, hypothesis, next_step, "
                "open_question_add, open_question_remove, checked_source, closed_summary"
            )
        })

    _save_manifest(manifest)
    return json.dumps({"updated": field, "manifest": manifest}, indent=2)


# ---- Tool: investigation_reflect ----

@mcp.tool()
def investigation_reflect(investigation_id: str) -> str:
    """
    Synthesize the current state of an investigation. Returns a structured
    summary of what has been established, what is still open, and what has
    not been checked. Call before write actions, at handoff points, or when
    context is growing long.

    Args:
        investigation_id: Investigation identifier.

    Returns:
        JSON reflection: finding breakdown, open questions, gaps, hypothesis,
        checked vs unchecked sources, and most recent findings per type.
    """
    manifest = _load_manifest(investigation_id)
    if not manifest:
        return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

    findings = _read_jsonl(_inv_dir(investigation_id) / "findings.jsonl")
    retracted = _load_retracted_ids(investigation_id)
    excluded_retracted = 0
    if retracted:
        kept = [f for f in findings if str(f.get("id", "")) not in retracted]
        excluded_retracted = len(findings) - len(kept)
        findings = kept

    by_type: dict[str, list] = {"observed": [], "inferred": [], "assumed": [], "gap": []}
    for f in findings:
        by_type.setdefault(f.get("type", "observed"), []).append(f)

    # Advisory self-check (additive): surface observed findings with no audit
    # receipt and findings that appear to contradict. Pure over the JSONL — no
    # qdrant required. Fail-open: degrades to empty lists, never blocks reflect.
    checks = _compute_self_check(investigation_id)
    self_check = {
        "unsupported_observed": [
            {
                "refs": v.refs,
                "excerpt": v.subject_excerpt,
                "rationale": v.rationale,
                "decision": v.decision,
                "confidence": v.confidence,
            }
            for v in checks["unsupported_observed"]
        ],
        "contradictions": [
            {
                "refs": v.refs,
                "excerpt": v.subject_excerpt,
                "rationale": v.rationale,
                "decision": v.decision,
                "confidence": v.confidence,
            }
            for v in checks["contradictions"]
        ],
        "hallucination_candidates": checks.get("hallucination_candidates", []),
    }

    # Entity frequency — count entity mentions across all non-retracted findings.
    # Surfacing the most-discussed observables gives analysts instant pivot points
    # and feeds investigation_entity_lookup calls.
    entity_counts: dict[str, dict[str, int]] = {
        "ips": {}, "emails": {}, "hostnames": {}, "hashes": {}, "cves": {}
    }
    for f in findings:
        for etype, vals in (f.get("entities") or {}).items():
            if etype in entity_counts:
                for v in (vals or []):
                    v = str(v).lower()
                    if v:
                        entity_counts[etype][v] = entity_counts[etype].get(v, 0) + 1
    key_entities = {
        etype: sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
        for etype, freq in entity_counts.items()
        if freq
    }

    return json.dumps({
        "investigation_id": investigation_id,
        "title": manifest["title"],
        "status": manifest["status"],
        "hypothesis": manifest["hypothesis"],
        "next_step": manifest["next_step"],
        "finding_counts": {t: len(v) for t, v in by_type.items()},
        "open_questions": manifest["open_questions"],
        "checked_sources": manifest["checked_sources"],
        "gaps": [f["text"] for f in by_type.get("gap", [])],
        "recent_per_type": {t: entries[-3:] for t, entries in by_type.items() if entries},
        "key_entities": key_entities,
        "excluded_retracted": excluded_retracted,
        "self_check": self_check,
    }, indent=2)


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
            "~/.copilot/session-state/*/events.jsonl, and ~/.copilot/logs/process-*.log"
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


# ---- Tool: reflection_loop_tick ----

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

        if store_item_findings:
            if kind == "session_event" and not raw_errors and not raw_warnings:
                low_signal_session_events.append({
                    "path": path,
                    "lines_scanned": int(summary.get("lines_scanned") or 0),
                    "bytes_scanned": int(summary.get("bytes_scanned") or 0),
                    "events": summary.get("events") or {},
                    "tools": summary.get("tools") or {},
                })
                continue
            visible_errors: dict[str, int] = {}
            visible_warnings: dict[str, int] = {}
            suppressed_error_hits = 0
            suppressed_warning_hits = 0
            suppressed_error_signatures = 0
            suppressed_warning_signatures = 0
            for sig, count in raw_errors.items():
                observed_count = int(error_observations.get(sig) or 0)
                if observed_count >= REFLECTION_SIGNATURE_OBSERVE_LIMIT:
                    suppressed_error_hits += count
                    suppressed_error_signatures += 1
                    continue
                visible_errors[sig] = count
                error_observations[sig] = observed_count + 1
            for sig, count in raw_warnings.items():
                observed_count = int(warning_observations.get(sig) or 0)
                if observed_count >= REFLECTION_SIGNATURE_OBSERVE_LIMIT:
                    suppressed_warning_hits += count
                    suppressed_warning_signatures += 1
                    continue
                visible_warnings[sig] = count
                warning_observations[sig] = observed_count + 1
            stats["error_signatures_suppressed"] += suppressed_error_hits
            stats["warning_signatures_suppressed"] += suppressed_warning_hits
            top_error = ", ".join(
                f"{sig} ({count})" for sig, count in list(visible_errors.items())[:3]
            ) or "none"
            top_warning = ", ".join(
                f"{sig} ({count})" for sig, count in list(visible_warnings.items())[:3]
            ) or "none"
            if suppressed_error_signatures:
                top_error += (
                    f"; saturated={suppressed_error_signatures} signatures "
                    f"({suppressed_error_hits} hits)"
                )
            if suppressed_warning_signatures:
                top_warning += (
                    f"; saturated={suppressed_warning_signatures} signatures "
                    f"({suppressed_warning_hits} hits)"
                )
            finding_text = (
                f"reflection_loop_tick processed {kind} target={path}; "
                f"lines={summary.get('lines_scanned', 0)} bytes={summary.get('bytes_scanned', 0)}; "
                f"sampling={summary.get('sampling_mode', 'full')}; "
                f"top_events={summary.get('events', {})}; top_tools={summary.get('tools', {})}; "
                f"errors={top_error}; warnings={top_warning}."
            )
            store_res = json.loads(investigation_store(
                investigation_id=investigation_id,
                finding_type="observed",
                text=finding_text,
                source="reflection_loop_tick",
                confidence="low",
                tags="self-reflection,loop-tick,artifact-mining,unreceipted-observed",
            ))
            if bool(store_res.get("stored")):
                findings_written += 1

    if store_item_findings and low_signal_session_events:
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
        low_signal_res = json.loads(investigation_store(
            investigation_id=investigation_id,
            finding_type="observed",
            text=low_signal_text,
            source="reflection_loop_tick",
            confidence="low",
            tags="self-reflection,loop-tick,artifact-mining,unreceipted-observed,batched-low-signal",
        ))
        if bool(low_signal_res.get("stored")):
            findings_written += 1

    if store_item_findings and batch_error_signatures:
        sig, count = batch_error_signatures.most_common(1)[0]
        infer_text = (
            "Batch dominant error signature suggests reliability hotspot: "
            f"{sig} (count={count}) in latest processed artifacts."
        )
        infer_res = json.loads(investigation_store(
            investigation_id=investigation_id,
            finding_type="inferred",
            text=infer_text,
            source="reflection_loop_tick",
            confidence="medium",
            tags="self-reflection,error-cluster,inference",
        ))
        if bool(infer_res.get("stored")):
            findings_written += 1

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

@mcp.tool()
def investigation_search(
    query: str,
    investigation_id: Optional[str] = None,
    limit: int = 10,
    include_retracted: bool = False,
    min_confidence: str = "low",
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

    Returns:
        JSON list of matching findings with investigation context.
    """
    # Normalise confidence floor so callers passing "High" or "MEDIUM" are not
    # silently ignored by the lowercase dict lookup downstream.
    min_confidence = str(min_confidence or "low").lower()
    if min_confidence not in _CONFIDENCE_RANK:
        logger.warning(
            "investigation_search: unknown min_confidence %r — ignoring filter; "
            "valid values: low, medium, high", min_confidence
        )
        min_confidence = "low"

    # Precompute retracted finding ids per investigation in scope. A row is
    # filtered when it names a finding id (or matches text of one) that is
    # retracted. Fail-safe: an empty map means nothing is filtered.
    _retracted_by_inv: dict[str, set[str]] = {}
    _retracted_text_by_inv: dict[str, set[str]] = {}
    if not include_retracted:
        try:
            scope_invs = (
                [investigation_id] if investigation_id
                else ([p.name for p in MEMORY_DIR.iterdir() if p.is_dir()] if MEMORY_DIR.exists() else [])
            )
            for _inv in scope_invs:
                rids = _load_retracted_ids(_inv)
                if not rids:
                    continue
                _retracted_by_inv[_inv] = rids
                texts: set[str] = set()
                for f in _read_jsonl(MEMORY_DIR / _inv / "findings.jsonl"):
                    if str(f.get("id", "")) in rids:
                        t = str(f.get("text", "") or "").strip()
                        if t:
                            texts.add(t)
                _retracted_text_by_inv[_inv] = texts
        except Exception as exc:  # fail-safe — never block search on filtering
            logger.debug("retraction scope precompute failed, not filtering: %r", exc)
            _retracted_by_inv = {}
            _retracted_text_by_inv = {}

    _excluded_retracted = {"n": 0}

    def _is_retracted_row(row: dict) -> bool:
        if include_retracted or not _retracted_by_inv:
            return False
        inv = str(row.get("investigation_id", ""))
        rids = _retracted_by_inv.get(inv)
        rtexts = _retracted_text_by_inv.get(inv)
        if not rids and not rtexts:
            return False
        rid = row.get("finding_id") or row.get("id")
        if rid is not None and str(rid) in (rids or set()):
            return True
        text = str(row.get("text", "") or "").strip()
        if text and rtexts and text in rtexts:
            return True
        return False
    def _iter_target_investigations() -> list[str]:
        if investigation_id:
            if not (MEMORY_DIR / investigation_id).exists():
                return []
            return [investigation_id]
        if not MEMORY_DIR.exists():
            return []
        return [p.name for p in MEMORY_DIR.iterdir() if p.is_dir()]

    _, recall_fn = _get_mnemo_funcs()
    mnemo_enabled = recall_fn is not None
    mnemo_rows = _mnemo_recall(query, top_k=max(limit * 4, 20), investigation_id=investigation_id)

    deduped: list[dict] = []
    seen: set[str] = set()

    def _add_row(row: dict) -> None:
        if _is_retracted_row(row):
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

    # Apply confidence floor post-hoc so mnemosyne rows — which carry no
    # "confidence" field — don't silently undermine the caller's filter when
    # mnemosyne satisfies the result limit before Qdrant is queried.
    if min_confidence and min_confidence in _CONFIDENCE_RANK and min_confidence != "low":
        floor = _CONFIDENCE_RANK[min_confidence]
        deduped = [
            r for r in deduped
            # Normalise stored confidence to lowercase — mnemosyne-sourced rows may
            # carry mixed-case values; without .lower() "High" maps to rank 0.
            if _CONFIDENCE_RANK.get(str(r.get("confidence", "low")).lower(), 0) >= floor
        ]

    if not deduped:
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

    mode = "mnemo_primary"
    if qdrant.get("ok"):
        mode = f"mnemo+{qdrant.get('reason')}" if mnemo_rows else str(qdrant.get("reason"))

    return json.dumps({
        "mode": mode,
        "results": deduped[: max(1, min(limit, 200))],
        "excluded_retracted": _excluded_retracted["n"],
        "include_retracted": include_retracted,
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
        claim_support_refs: list[dict] = []
        claim_contradiction_refs: list[dict] = []

        for record in evidence_pool:
            score = _lexical_match_score(claim_tokens, record.get("tokens", set()))
            if score < 0.45:
                continue
            evidence_negated = bool(_NEGATION_RE.search(str(record.get("text", ""))))
            if claim_negated != evidence_negated and score >= 0.5:
                ref = _make_ref(record, "contradiction", score=score)
                claim_contradiction_refs.append(ref)
            else:
                ref = _make_ref(record, "support", score=score)
                claim_support_refs.append(ref)

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
    degraded_active = (not qdrant_enabled) or (not qdrant_available) or bool(unique_errors) or (qdrant_enabled and qdrant_available and not qdrant_query_success)
    degraded_reason = None
    if not qdrant_enabled:
        degraded_reason = "qdrant_disabled"
    elif not qdrant_available:
        degraded_reason = "qdrant_unavailable"
    elif unique_errors:
        degraded_reason = "qdrant_semantic_error"
    elif not qdrant_query_success:
        degraded_reason = "qdrant_semantic_not_executed"

    verdict_summary = _record_claim_verdicts(investigation_id, claim_results, record=record)

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

    # Prefer Qdrant (indexed, fast); fall back to JSONL scan.
    findings = _entity_lookup_qdrant(entity, entity_type, investigation_id, limit)
    method = "qdrant"
    if not findings:
        findings = _entity_lookup_jsonl(entity, entity_type, investigation_id, limit)
        method = "jsonl_fallback"

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
        findings = _entity_lookup_qdrant(entity, etype, None, limit_per_entity * 4)
        if not findings:
            findings = _entity_lookup_jsonl(entity, etype, None, limit_per_entity * 4)
            method = "jsonl_fallback"
        else:
            method = "qdrant"

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

@mcp.tool()
def investigation_finding_provenance(
    finding_id: str,
    investigation_id: str,
) -> str:
    """
    Trace a finding back through its derivation chain to root observed evidence.

    Follows the ``derived_from`` links stored on each finding, walking up the
    chain until it reaches findings with no parent (root observations) or a
    cycle is detected. Returns the full chain so the analyst can verify that
    an inference is actually grounded in observed data and not built on top of
    another inference or assumption.

    A chain that terminates in an ``assumed`` finding — rather than ``observed``
    data — means the conclusion is a hypothesis chain, not an evidence chain.

    Args:
        finding_id: The ID of the finding to trace.
        investigation_id: The investigation containing the finding.

    Returns:
        JSON with the chain from the target finding to its root evidence,
        each node annotated with its type, confidence, source, and text.
    """
    # Use MEMORY_DIR directly — _inv_dir() creates the directory, which would
    # silently create an empty investigation dir for any non-existent id.
    inv_path = MEMORY_DIR / investigation_id
    if not inv_path.is_dir():
        return json.dumps({"error": f"Investigation '{investigation_id}' not found."})

    findings_by_id: dict[str, dict] = {
        str(f.get("id", "")): f
        for f in _read_jsonl(inv_path / "findings.jsonl")
        if f.get("id")
    }

    target = findings_by_id.get(finding_id)
    if not target:
        return json.dumps({"error": f"Finding '{finding_id}' not found in '{investigation_id}'"})

    chain: list[dict] = []
    visited: set[str] = set()
    current_id = finding_id

    while current_id and current_id not in visited:
        visited.add(current_id)
        node = findings_by_id.get(current_id)
        if not node:
            chain.append({"finding_id": current_id, "error": "not_found"})
            break
        chain.append({
            "finding_id": current_id,
            "ts": node.get("ts"),
            "record_type": node.get("record_type") or node.get("type"),
            "confidence": node.get("confidence"),
            "source": node.get("source"),
            "text": str(node.get("text", ""))[:400],
            "derived_from": node.get("derived_from", []),
        })
        # Only the first parent is followed.  Multi-parent derived_from chains
        # (a finding citing two or more independent observations) are not fully
        # traversed — grounded_in_observed reflects the first-listed branch only.
        parents = node.get("derived_from") or []
        if not parents:
            break
        current_id = str(parents[0]) if parents else None

    root = chain[-1] if chain else None
    root_type = root.get("record_type") if root else None
    grounded = root_type == "observed"

    return json.dumps({
        "investigation_id": investigation_id,
        "chain_length": len(chain),
        "grounded_in_observed": grounded,
        "grounding_assessment": (
            "fully grounded" if grounded
            else f"chain terminates in '{root_type}' — not directly observed evidence"
        ),
        "chain": chain,
    }, indent=2, default=str)


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
    degraded_active = (not qdrant_enabled) or (not qdrant_available) or bool(qdrant_errors) or (qdrant_enabled and qdrant_available and not qdrant_query_success)
    degraded_reason = None
    if not qdrant_enabled:
        degraded_reason = "qdrant_disabled"
    elif not qdrant_available:
        degraded_reason = "qdrant_unavailable"
    elif qdrant_errors:
        degraded_reason = "qdrant_semantic_error"
    elif not qdrant_query_success:
        degraded_reason = "qdrant_semantic_not_executed"

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

@mcp.tool()
def investigation_list() -> str:
    """
    List all investigations with status and finding counts, most recently
    updated first.

    Returns:
        JSON list of investigation summaries.
    """
    if not MEMORY_DIR.exists():
        return json.dumps({"investigations": []})

    investigations = []
    for d in sorted(MEMORY_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        manifest = _load_manifest(d.name)
        if manifest:
            investigations.append({
                "id": manifest["id"],
                "title": manifest["title"],
                "status": manifest["status"],
                "created_at": manifest["created_at"],
                "updated_at": manifest["updated_at"],
                "finding_counts": manifest["finding_counts"],
                "open_questions_count": len(manifest["open_questions"]),
                "hypothesis": manifest["hypothesis"],
            })

    return json.dumps({"investigations": investigations}, indent=2)


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
        per_investigation.append({
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
        })

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
        tf = str(target_file).strip()
        source["target_file"] = tf
        p = Path(tf)
        if not p.exists() or not p.is_file():
            code_note = f"target_file {tf!r} does not exist or is not a file — skipped"
        elif p.suffix != ".py":
            code_note = f"target_file {tf!r} is not a .py file — skipped"
        else:
            try:
                content = p.read_text()
            except Exception as exc:  # fail-open — unreadable file is advisory note
                logger.debug("could not read target_file %s: %r", tf, exc)
                content = ""
                code_note = f"target_file {tf!r} could not be read — skipped"
            if content or code_note is None:
                suspected |= _suspected_entities_from_text(content)
                try:
                    verdicts = run_code_checks(p)
                except Exception as exc:  # fail-open — checker errors are advisory
                    logger.debug("run_code_checks failed for %s, degrading: %r", tf, exc)
                    verdicts = []
                lh_verdicts = [
                    v for v in verdicts
                    if str(getattr(v, "verdict_type", "")) in _CODE_HALLUCINATION_CODES
                ]
                suspected |= _verdict_symbols(lh_verdicts)
                code_findings = _code_findings_from_verdicts(lh_verdicts)
                if not lh_verdicts:
                    code_note = (
                        f"no code-hallucination issues found in {p.name} — "
                        "correlating on its extracted entities only (advisory)"
                    )

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

    # 1. qdrant_reachable — is QDRANT_URL set and the server answering?
    qdrant_url = os.environ.get("QDRANT_URL", "")
    client = None
    main_col = None

    def _probe_qdrant_reachable():
        nonlocal client, main_col
        if not qdrant_url:
            return (
                "warn",
                "QDRANT_URL is not set — qdrant indexing disabled; loci falls back"
                "to mnemosyne/keyword search.",
                "set QDRANT_URL if vector search is expected; otherwise this is benign",
            )
        client, main_col = _get_qdrant()
        if client is None:
            return (
                "fail",
                f"QDRANT_URL={qdrant_url} is set but the server is unreachable.",
                "confirm the qdrant container is up and reachable at QDRANT_URL "
                "(docker ps; curl $QDRANT_URL/healthz)",
            )
        return ("ok", f"connected to qdrant at {qdrant_url}", None)

    checks.append(_health_check("qdrant_reachable", _probe_qdrant_reachable))

    # 2. qdrant_collections — expected collections present + their layout.
    collection_dims: dict[str, int | None] = {}

    def _probe_qdrant_collections():
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

    checks.append(_health_check("qdrant_collections", _probe_qdrant_collections))

    # 3. embeddings_dense — fastembed dense embedder loads and embeds.
    embed_dim: int | None = None

    def _probe_embeddings_dense():
        nonlocal embed_dim
        vec = _embed("memory_health probe")  # transient throwaway, never stored
        if not vec:
            return (
                "fail",
                "dense embedder (Ollama) unavailable — semantic/hybrid search "
                "is disabled; hermes runs on keyword fallback only.",
                "ensure Ollama is running and the nomic-embed-text model is available "
                "(OLLAMA_BASE_URL and EMBED_MODEL env vars can override defaults).",
            )
        embed_dim = len(vec)
        return ("ok", f"dense embedder active; dimension={embed_dim}", None)

    checks.append(_health_check("embeddings_dense", _probe_embeddings_dense))

    # 4. embeddings_sparse — optional sparse embedder.
    def _probe_embeddings_sparse():
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

    checks.append(_health_check("embeddings_sparse", _probe_embeddings_sparse))

    # 5. mnemo_mirror — mnemosyne importable in THIS venv (the silent-flag bug).
    def _probe_mnemo_mirror():
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

    checks.append(_health_check("mnemo_mirror", _probe_mnemo_mirror))

    # 6. dimension_consistency — embedder dim vs configured collection dim(s).
    def _probe_dimension_consistency():
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

    checks.append(_health_check("dimension_consistency", _probe_dimension_consistency))

    # Resolve target investigations for the store-scoped checks (read-only:
    # never use _inv_dir here, which would mkdir).
    if investigation_id is not None:
        inv_targets = [investigation_id] if (MEMORY_DIR / investigation_id).exists() else []
        inv_missing = investigation_id if not inv_targets else None
    elif MEMORY_DIR.exists():
        inv_targets = sorted(p.name for p in MEMORY_DIR.iterdir() if p.is_dir())
        inv_missing = None
    else:
        inv_targets = []
        inv_missing = None

    # 7. retraction_integrity — parse cleanly + flag orphaned retractions.
    def _probe_retraction_integrity():
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

    checks.append(_health_check("retraction_integrity", _probe_retraction_integrity))

    # 8. store_counts — inventory of findings/audit/retraction records.
    def _probe_store_counts():
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

    checks.append(_health_check("store_counts", _probe_store_counts))

    # Roll up overall status from the worst check.
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


def _distinctive_entity_set(entities: dict | None) -> set[str]:
    """Flatten the server's typed-entity dict to a set of distinctive entity tokens."""
    out: set[str] = set()
    if not isinstance(entities, dict):
        return out
    for bucket in ("ips", "hashes", "cves", "emails", "hostnames"):
        for v in entities.get(bucket, []) or []:
            s = str(v).strip().lower()
            if s:
                out.add(s)
    return out


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

        return asyncio.run(_forget_all())
    except Exception as exc:  # fail-open — verdict cleanup is best-effort
        logger.debug("verdict forget failed, degrading to 0: %r", exc)
        return 0


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

    # Record a quarantine verdict to the store (fail-open) for recall.
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
        _record_verdicts([quarantine_verdict])
    except Exception as exc:  # fail-open
        logger.debug("quarantine verdict record failed, degrading: %r", exc)

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

    return json.dumps({
        "seed_ids": seed_ids,
        "retracted": retracted_records,
        "count": len(retracted_records),
        "verdicts_forgotten": verdicts_forgotten,
        "applied": True,
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


@mcp.tool()
def rag_context_search(
    query: str,
    limit: int = 10,
    collections: Optional[list] = None,
    budget_chars: int = 6000,
    exclude_types: Optional[list] = None,
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

    # Build the GPS-exclusion filter for agent_core_chunks (only).
    _agent_filter = None
    if exclude_types:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        _agent_filter = Filter(must_not=[FieldCondition(key="type", match=MatchAny(any=exclude_types))])

    for col in _collections:
        try:
            qf = _agent_filter if col == "agent_core_chunks" else None
            hits = _qdrant_search_collection(query, collection_name=col, limit=limit, query_filter=qf)
            all_results.extend(hits)
        except Exception as exc:
            errors.append(f"{col}: {exc}")
            logger.warning("rag_context_search: collection %s failed: %s", col, exc)

    # Merge-sort by score descending across all collections
    all_results.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)

    # Final cross-encoder re-pass for consistent cross-collection ranking
    ce = _get_cross_encoder()
    if ce is not None and len(all_results) > 1:
        try:
            pairs = [(query, str(r.get('text', r.get('content', '')))[:512]) for r in all_results[:20]]
            ce_scores = ce.predict(pairs)
            for r, s in zip(all_results[:20], ce_scores):
                r['final_ce_score'] = round(float(s), 4)
            all_results[:20] = sorted(all_results[:20], key=lambda r: r.get('final_ce_score', -999), reverse=True)
        except Exception as _ce_exc:
            logger.debug('Final CE re-pass failed: %s', _ce_exc)

    ctx = context_assemble(all_results, query, budget_chars=budget_chars)
    ctx["mode"] = "rag_hybrid"
    ctx["collections_searched"] = _collections
    ctx["qdrant_available"] = True
    if errors:
        ctx["collection_errors"] = errors
    return json.dumps(ctx, indent=2)


# ---------------------------------------------------------------------------
# Mnemosyne sleep / consolidation
# ---------------------------------------------------------------------------

@mcp.tool()
def memory_consolidate(dry_run: bool = False) -> str:
    """
    Run Mnemosyne sleep/consolidation cycle.
    Merges old working_memory entries into episodic memory, reducing DB size.
    Safe to run periodically (daily or after large investigation sessions).

    Args:
        dry_run: If True, preview consolidation without executing.

    Returns JSON with consolidation stats.
    """
    import json as _json
    try:
        from mnemosyne.core.memory import Mnemosyne
        m = Mnemosyne()
        result = m.sleep_all_sessions(dry_run=dry_run)
        return _json.dumps({
            "status": "ok",
            "dry_run": dry_run,
            "result": result if isinstance(result, dict) else str(result),
        })
    except Exception as e:
        return _json.dumps({"status": "error", "error": str(e)})

# ---------------------------------------------------------------------------
# Metamemory
# ---------------------------------------------------------------------------


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
    import math as _math

    client, col = _get_qdrant()
    if client is None:
        return json.dumps({
            "confidence": 0.0, "basis": "qdrant_unavailable",
            "cues": {}, "top_hit_preview": "", "recommendation": "verify",
        })

    try:
        emb = _embed(query)
    except Exception:
        emb = None
    if not emb:
        return json.dumps({
            "confidence": 0.0, "basis": "embed_failed",
            "cues": {}, "top_hit_preview": "", "recommendation": "verify",
        })

    try:
        results = client.search(
            collection_name="hermes_memory",
            query_vector={"dense": emb} if isinstance(emb, list) else emb,
            limit=top_k,
            with_payload=True,
        )
    except Exception:
        results = []

    if not results:
        return json.dumps({
            "confidence": 0.0, "basis": "no_trace",
            "cues": {"fluency": 0.0, "accessibility": 0.0, "source_div": 0,
                     "corroboration": 0, "trust": 0.0},
            "top_hit_preview": "",
            "recommendation": "no memory found — investigate before asserting",
        })

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
    corroboration = _math.log1p(max_occurrences) / _math.log1p(20)  # saturates ~20

    # Cue 5: Trust — mean confidence tier of top hits
    trust = sum(conf_vals) / max(1, len(conf_vals))

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
    confidence = 1.0 / (1.0 + _math.exp(-8 * (raw - 0.5)))

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
    from memcheck.checks.contradiction_llm import _extract_json

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
            kept = [f for c, f in scored if c >= ground_threshold]
            gated = sorted(kept, key=lambda f: 0)[:12] if kept else []
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
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    transport = os.environ.get("HERMES_MCP_TRANSPORT", "stdio")
    if transport in ("sse", "streamable-http"):
        host = os.environ.get("HERMES_MCP_HOST", "0.0.0.0")
        port = int(os.environ.get("HERMES_MCP_PORT", "8000"))
        mcp.run(transport=transport, host=host, port=port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
