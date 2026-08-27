"""Mnemosyne memory-bank helpers (extracted from server.py).

Fail-open wrappers around the optional `mnemosyne` package. Every consumer of
these helpers still lives in server.py and resolves them through server's
globals (they are re-exported there), so tests that rebind e.g.
`server._mnemo_recall` keep working unchanged.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from inv_store import _safe_float

logger = logging.getLogger("loci-mcp")

_mnemo_remember_fn = None
_mnemo_recall_fn = None


def _mnemo_bank() -> str:
    return os.environ.get("LOCI_MNEMO_BANK", "default")


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
