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
from provenance_firewall import provenance_fields

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
        row = {
            "score": round(score, 4),
            "investigation_id": str(inv_from_meta or investigation_id or metadata.get("investigation_id") or ""),
            "record_type": str(metadata.get("record_type") or metadata.get("type") or "memory"),
            "source": str(metadata.get("source") or item.get("source") or "mnemosyne"),
            "ts": item.get("ts") or item.get("created_at"),
            "text": text,
            "origin": "mnemosyne",
            # Surface the provenance tier the finding was stored with (see
            # _store_index) instead of leaving it out — an absent field here
            # would otherwise let a model-asserted finding read back as the
            # legacy tool_verified default without any way to tell the two
            # apart. provenance_fields() still applies that default for truly
            # untagged (pre-fix) rows, but flags it via provenance_defaulted.
            **provenance_fields(metadata),
        }
        # _store_index stores the finding id; without it, resolution and
        # retraction lookups fall back to "open" and exact-text matching.
        if metadata.get("finding_id"):
            row["finding_id"] = str(metadata["finding_id"])
        claim_scope = metadata.get("claim_scope")
        if isinstance(claim_scope, dict):
            row["claim_scope"] = dict(claim_scope)
        flybrain_provenance = metadata.get("flybrain_provenance")
        if isinstance(flybrain_provenance, dict):
            row["flybrain_provenance"] = dict(flybrain_provenance)
        rows.append(row)
    return rows


# --- retraction propagation --------------------------------------------------
#
# Mnemosyne's own lifecycle mark is ``valid_until`` on working_memory and
# episodic_memory (its invalidate() sets it to now, and its recall skips rows
# whose valid_until has passed). It has no public "un-invalidate", and Loci does
# not keep Mnemosyne's memory ids, so this writes the column directly, matching
# rows by the ``finding_id`` Loci stored in their metadata. Soft and reversible:
# restore clears only the exact stamp this module wrote, so an invalidation
# Mnemosyne made on its own is never undone. The legacy ``memories`` table has
# no lifecycle column; its rows are counted and reported, never modified.

_MNEMO_LIFECYCLE_TABLES = ("working_memory", "episodic_memory")


def _mnemo_db_path() -> str:
    data_dir = os.path.expanduser(os.environ.get("MNEMOSYNE_DATA_DIR", "~/.hermes/mnemosyne/data"))
    return os.path.join(data_dir, "mnemosyne.db")


def _mnemo_set_retracted(finding_ids: list[str], *, retracted: bool, stamps: Optional[list[str]] = None) -> dict:
    """Mark (or unmark) Mnemosyne rows for ``finding_ids`` as expired.

    ``retracted=True`` stamps ``valid_until`` with a fresh timestamp on rows
    that have none, and returns it as ``stamp`` so the caller can record it.
    ``retracted=False`` clears ``valid_until`` only where it equals one of
    ``stamps``. Never creates the database and never deletes a row.

    Returns ``{status, updated, legacy_unflagged, stamp?, error?}`` where status
    is ``ok``, ``unavailable`` (no database) or ``failed``.
    """
    import sqlite3
    from datetime import datetime

    ids = sorted({str(f) for f in finding_ids or [] if f})
    path = _mnemo_db_path()
    if not ids:
        return {"status": "ok", "updated": 0, "legacy_unflagged": 0}
    if not os.path.exists(path):
        return {"status": "unavailable", "updated": 0, "legacy_unflagged": 0,
                "reason": "no Mnemosyne database at MNEMOSYNE_DATA_DIR"}
    # Mnemosyne compares valid_until against datetime.now().isoformat() (naive
    # local time); stamping the same form makes the row expire immediately.
    stamp = datetime.now().isoformat()
    updated = 0
    legacy = 0
    marks = ",".join("?" for _ in ids)
    # CASE guards json_extract: SQLite does not promise AND short-circuits.
    match = ("CASE WHEN json_valid(metadata_json) THEN json_extract(metadata_json, '$.finding_id') END "
             f"IN ({marks})")
    try:
        conn = sqlite3.connect(f"file:{path}?mode=rw", uri=True, timeout=2.0)
    except sqlite3.Error as exc:
        return {"status": "failed", "updated": 0, "legacy_unflagged": 0, "error": str(exc)[:200]}
    try:
        with conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table in _MNEMO_LIFECYCLE_TABLES:
                if table not in tables:
                    continue
                cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
                if not {"metadata_json", "valid_until"} <= cols:
                    continue
                if retracted:
                    cur = conn.execute(
                        f"UPDATE {table} SET valid_until = ? WHERE valid_until IS NULL AND {match}",
                        [stamp, *ids],
                    )
                    updated += max(cur.rowcount, 0)
                else:
                    for old in sorted({s for s in stamps or [] if s}):
                        cur = conn.execute(
                            f"UPDATE {table} SET valid_until = NULL WHERE valid_until = ? AND {match}",
                            [old, *ids],
                        )
                        updated += max(cur.rowcount, 0)
            if "memories" in tables:
                cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)")}
                if "metadata_json" in cols:
                    legacy = conn.execute(
                        f"SELECT COUNT(*) FROM memories WHERE {match}",
                        ids,
                    ).fetchone()[0]
    except sqlite3.Error as exc:
        logger.warning("Mnemosyne retraction propagation failed: %r", exc)
        return {"status": "failed", "updated": 0, "legacy_unflagged": 0, "error": str(exc)[:200]}
    finally:
        conn.close()
    out = {"status": "ok", "updated": updated, "legacy_unflagged": int(legacy)}
    if retracted:
        out["stamp"] = stamp
    return out
