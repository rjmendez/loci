#!/usr/bin/env python3
"""
SWR (Sharp-Wave Ripple) biased replay consolidation.

Biological analog: during NREM sleep, hippocampal sharp-wave ripples replay
recent episodes in time-compressed form, biased toward salient/reward-associated
memories. Replay drives hippocampus-to-neocortex systems consolidation.

This script:
  1. Fetches recent hermes_memory Qdrant points (configurable look-back window).
  2. Scores each by replay priority: recency × salience × reward.
     - recency: exponential decay from created_at_ts (bias toward recent)
     - salience: importance field (novel + high-confidence findings score higher)
     - reward: recall_count proxy (often-retrieved = useful = reinforced)
  3. Selects the top-K by priority (K=7, theta-gamma working memory bound).
  4. Interleaves with existing consolidated findings (prevents catastrophic
     interference — CLS McClelland 1995).
  5. Calls Ollama to generate a compressed abstraction from the batch.
  6. Stores the abstraction back to hermes_memory with record_type=consolidated.

Run from cron or the hermes sleep scheduler. Idempotent and fail-safe.

Usage:
    python3 swr_replay.py [--dry-run] [--lookback-hours N] [--replay-k K]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
import urllib.error
import uuid
from datetime import datetime, timezone

# ── config ────────────────────────────────────────────────────────────────────

QDRANT_URL            = os.environ.get("QDRANT_URL")
QDRANT_KEY            = os.environ.get("QDRANT_API_KEY",   "")
OLLAMA_URL            = os.environ.get("OLLAMA_URL")
EMBED_MODEL           = os.environ.get("MNEMOSYNE_EMBEDDING_MODEL", "nomic-embed-text")
_EMBED_API_KEY        = os.environ.get("EMBED_API_KEY", "")
_EMBED_API_KEY_HEADER = os.environ.get("EMBED_API_KEY_HEADER", "Authorization")
LLM_MODEL    = os.environ.get("SWR_LLM_MODEL",   "llama3.2:latest")
COLLECTION   = os.environ.get("SWR_COLLECTION",  "hermes_memory")

LOOKBACK_HOURS = float(os.environ.get("SWR_LOOKBACK_HOURS", "6"))
REPLAY_K       = int(os.environ.get("SWR_REPLAY_K",        "7"))    # theta-gamma bound
INTERLEAVE_K   = int(os.environ.get("SWR_INTERLEAVE_K",    "3"))    # existing consolidated
FETCH_LIMIT    = int(os.environ.get("SWR_FETCH_LIMIT",     "200"))  # candidate pool

# Replay priority weights (must sum to 1.0).
W_RECENCY  = float(os.environ.get("SWR_W_RECENCY",  "0.40"))
W_SALIENCE = float(os.environ.get("SWR_W_SALIENCE", "0.35"))
W_REWARD   = float(os.environ.get("SWR_W_REWARD",   "0.25"))
RECENCY_HALFLIFE_HOURS = float(os.environ.get("SWR_RECENCY_HALFLIFE_HOURS", "3"))

_LN2 = math.log(2)


# ── Qdrant helpers ────────────────────────────────────────────────────────────

def _qdrant_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if QDRANT_KEY:
        h["api-key"] = QDRANT_KEY
    return h


def _qdrant_scroll(collection: str, limit: int, min_created_ts: float) -> list[dict]:
    """Scroll points with created_at_ts >= min_created_ts, return payload dicts."""
    url = f"{QDRANT_URL}/collections/{collection}/points/scroll"
    body = json.dumps({
        "limit": limit,
        "with_payload": True,
        "with_vector": False,
        "filter": {
            "must": [{
                "key": "created_at_ts",
                "range": {"gte": min_created_ts},
            }],
        },
    }).encode()
    req = urllib.request.Request(url, data=body, headers=_qdrant_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            d = json.loads(resp.read())
        return [pt.get("payload") or {} for pt in d.get("result", {}).get("points", [])]
    except Exception as e:
        print(f"[swr] scroll failed: {e}", file=sys.stderr)
        return []


def _qdrant_search_consolidated(collection: str, embedding: list[float], k: int) -> list[dict]:
    """Fetch existing consolidated points for interleaving."""
    url = f"{QDRANT_URL}/collections/{collection}/points/search"
    body = json.dumps({
        "vector": {"dense": embedding},
        "limit": k,
        "with_payload": True,
        "with_vector": False,
        "filter": {
            "must": [{
                "key": "record_type",
                "match": {"value": "consolidated"},
            }],
        },
    }).encode()
    req = urllib.request.Request(url, data=body, headers=_qdrant_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            d = json.loads(resp.read())
        return [pt.get("payload") or {} for pt in d.get("result", [])]
    except Exception:
        return []


def _qdrant_upsert(collection: str, point_id: str, vector: list[float], payload: dict) -> bool:
    url = f"{QDRANT_URL}/collections/{collection}/points"
    body = json.dumps({
        "points": [{
            "id": point_id,
            "vector": {"dense": vector},
            "payload": payload,
        }],
    }).encode()
    req = urllib.request.Request(url, data=body, headers=_qdrant_headers(), method="PUT")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        return True
    except Exception as e:
        print(f"[swr] upsert failed: {e}", file=sys.stderr)
        return False


# ── Ollama helpers ────────────────────────────────────────────────────────────

def _embed_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if _EMBED_API_KEY:
        if _EMBED_API_KEY_HEADER.lower() == "authorization":
            h["Authorization"] = f"Bearer {_EMBED_API_KEY}"
        else:
            h[_EMBED_API_KEY_HEADER] = _EMBED_API_KEY
    return h


def _embed(text: str) -> list[float] | None:
    url = f"{OLLAMA_URL}/v1/embeddings"
    body = json.dumps({"model": EMBED_MODEL, "input": [text]}).encode()
    req = urllib.request.Request(url, data=body, headers=_embed_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            d = json.loads(resp.read())
        return d["data"][0]["embedding"]
    except Exception as e:
        print(f"[swr] embed failed: {e}", file=sys.stderr)
        return None


def _generate(prompt: str) -> str | None:
    url = f"{OLLAMA_URL}/api/generate"
    body = json.dumps({
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 400},
    }).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            d = json.loads(resp.read())
        return d.get("response", "").strip()
    except Exception as e:
        print(f"[swr] generate failed: {e}", file=sys.stderr)
        return None


# ── priority scoring ──────────────────────────────────────────────────────────

def _replay_priority(payload: dict, now_ts: float) -> float:
    """Priority score for replay selection: recency × salience × reward."""
    created = payload.get("created_at_ts")
    if created:
        age_hours = max(0.0, (now_ts - float(created)) / 3600.0)
        recency = math.exp(-_LN2 * age_hours / RECENCY_HALFLIFE_HOURS)
    else:
        recency = 0.3  # unknown age → deprioritized

    importance = float(payload.get("importance", 0.5) or 0.5)
    conf = str(payload.get("confidence", "") or "").lower()
    conf_boost = {"high": 1.2, "medium": 1.0, "low": 0.7}.get(conf, 1.0)
    salience = min(1.0, importance * conf_boost)

    recall_count = int(payload.get("recall_count", 0) or 0)
    reward = math.log1p(recall_count) / math.log1p(10)  # saturates at ~10 recalls

    return W_RECENCY * recency + W_SALIENCE * salience + W_REWARD * reward


# ── distillation prompt ───────────────────────────────────────────────────────

def _build_prompt(replay_batch: list[dict], interleaved: list[dict]) -> str:
    def _fmt(payloads):
        lines = []
        for p in payloads:
            text = p.get("text") or p.get("content") or p.get("content_preview") or ""
            if text.strip():
                lines.append(f"- {text.strip()[:300]}")
        return "\n".join(lines) or "(none)"

    return (
        "You are a memory consolidation agent. Below are recent episodic memories "
        "(replay batch) and existing consolidated knowledge (prior schema). "
        "Generate a single concise abstraction (2-4 sentences) that captures the "
        "durable, schema-level insight shared across the replay batch, reconciled "
        "with the prior knowledge. Focus on what is generalizable and stable — "
        "not individual observations. Be specific and factual.\n\n"
        f"REPLAY BATCH (recent, high-salience):\n{_fmt(replay_batch)}\n\n"
        f"PRIOR CONSOLIDATED KNOWLEDGE (for reconciliation):\n{_fmt(interleaved)}\n\n"
        "CONSOLIDATED ABSTRACTION:"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False, lookback_hours: float = LOOKBACK_HOURS,
         replay_k: int = REPLAY_K) -> None:
    now_ts = time.time()
    min_ts = now_ts - lookback_hours * 3600

    print(f"[swr] lookback={lookback_hours}h  k={replay_k}  dry_run={dry_run}")

    # ── 1. Fetch candidates ───────────────────────────────────────────────────
    candidates = _qdrant_scroll(COLLECTION, FETCH_LIMIT, min_ts)
    # Exclude already-consolidated points from the candidate pool.
    candidates = [p for p in candidates if p.get("record_type") != "consolidated"]
    print(f"[swr] {len(candidates)} candidates in window")

    if not candidates:
        print("[swr] nothing to replay — exiting")
        return

    # ── 2. Score and select top-K ─────────────────────────────────────────────
    scored = sorted(candidates, key=lambda p: _replay_priority(p, now_ts), reverse=True)
    replay_batch = scored[:replay_k]

    print(f"[swr] replay batch ({len(replay_batch)} items):")
    for p in replay_batch:
        prio = _replay_priority(p, now_ts)
        text_preview = (p.get("text") or p.get("content") or "")[:80]
        print(f"  [{prio:.3f}] {text_preview!r}")

    if dry_run:
        print("[swr] dry-run — stopping before embed/generate/upsert")
        return

    # ── 3. Embed centroid of batch for interleave search ─────────────────────
    batch_texts = [
        p.get("text") or p.get("content") or "" for p in replay_batch
    ]
    centroid_text = " ".join(t[:200] for t in batch_texts if t.strip())
    centroid_emb = _embed(centroid_text)
    if centroid_emb is None:
        print("[swr] embedding failed — exiting", file=sys.stderr)
        return

    # ── 4. Fetch existing consolidated for interleaving ───────────────────────
    interleaved = _qdrant_search_consolidated(COLLECTION, centroid_emb, INTERLEAVE_K)
    print(f"[swr] {len(interleaved)} consolidated points fetched for interleave")

    # ── 5. Generate compressed abstraction ───────────────────────────────────
    prompt = _build_prompt(replay_batch, interleaved)
    print("[swr] generating abstraction via Ollama …")
    abstraction = _generate(prompt)
    if not abstraction:
        print("[swr] generation failed — exiting", file=sys.stderr)
        return
    print(f"[swr] abstraction: {abstraction[:200]!r}")

    # ── 6. Embed and store abstraction ────────────────────────────────────────
    abs_emb = _embed(abstraction)
    if abs_emb is None:
        print("[swr] abstraction embedding failed — exiting", file=sys.stderr)
        return

    point_id = uuid.uuid4().hex
    source_ids = [
        str(p.get("id") or p.get("finding_id") or "")
        for p in replay_batch if p.get("id") or p.get("finding_id")
    ]
    payload = {
        "text": abstraction,
        "record_type": "consolidated",
        "type": "consolidated",
        "confidence": "medium",
        "importance": 0.75,
        "source": "swr_replay",
        "derived_from": source_ids,
        "created_at_ts": int(now_ts),
        "ts": datetime.now(timezone.utc).isoformat(),
        "replay_batch_size": len(replay_batch),
        "interleave_count": len(interleaved),
    }

    ok = _qdrant_upsert(COLLECTION, point_id, abs_emb, payload)
    if ok:
        print(f"[swr] consolidated point stored: {point_id}")
    else:
        print("[swr] store failed", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWR biased replay consolidation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Score and select but do not embed, generate, or store")
    parser.add_argument("--lookback-hours", type=float, default=LOOKBACK_HOURS,
                        help=f"Look-back window in hours (default {LOOKBACK_HOURS})")
    parser.add_argument("--replay-k", type=int, default=REPLAY_K,
                        help=f"Replay batch size (default {REPLAY_K})")
    args = parser.parse_args()
    main(dry_run=args.dry_run,
         lookback_hours=args.lookback_hours,
         replay_k=args.replay_k)
