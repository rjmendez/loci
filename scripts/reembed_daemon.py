#!/usr/bin/env python3
"""Safe, incremental GPU re-embed / index refresh for Qdrant.

Re-embeds or backfills dense vectors through the Ollama nomic embedding path.
Use it for model/version changes or points missing vectors. It is the batch
counterpart to the online embed tier.

Grounding assumptions
- Qdrant is a separate k3s CPU host storing vectors for collections such as
  `loci_memory` and `agent_core_chunks` (~1.84M points). We only read/rewrite
  vectors. `QDRANT_URL` / `QDRANT_API_KEY` follow the same env convention as
  `scripts/qdrant_payload_indexes.py` and `scripts/mnemosyne_qdrant_sync.py`.
- GPU work here is embeddings, not generation. Embeddings come from Ollama
  `nomic-embed-text` (768-dim) via the exact `mcp/embed_ops.py:embed_texts`
  path used online.
- Every batch fails open: embed or upsert errors are logged and the run
  continues, matching `embed_ops` / `llm_local`.
- `qdrant_client` and `embed_fn` are injectable (`None` => lazy env resolve),
  so importing the module hard-requires nothing and tests can stub both.

Safety
- Default is dry-run. Without `--apply`, the script scans, reports stale/missing
  points, writes nothing, and does not even call the GPU.
- `--apply` is the only mutating mode.

Incremental / idempotent
- A point is targeted only when it is missing a vector or its payload
  `embed_model` / `embed_version` differ from the current target.
- Freshly re-embedded points are stamped with those payload fields, so a second
  run targets nothing.

Usage:
    python3 scripts/reembed_daemon.py --collection loci_memory
    python3 scripts/reembed_daemon.py --collection loci_memory --apply

Env:
    QDRANT_URL, QDRANT_API_KEY   Qdrant endpoint (key also falls back to ~/.claude settings)
    OLLAMA_BASE_URL / OLLAMA_URL Ollama for the nomic embed path (via embed_ops)
    EMBED_MODEL                  current embed model tag (default `nomic-embed-text`)
    EMBED_VERSION                current embed version marker (default `"1"`)
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Callable, Optional

# Current embedding target. Missing vectors or mismatched payload stamps are stale.
_EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
_EMBED_VERSION = os.environ.get("EMBED_VERSION", "1")

# Payload keys stamped on re-embed and checked for staleness.
_MODEL_KEY = "embed_model"
_VERSION_KEY = "embed_version"

# Candidate payload fields for source text, in priority order.
_TEXT_KEYS = ("document", "text", "content", "summary", "title")


# ── injectable resolvers (lazy; import stays dependency-light) ──────────

def _resolve_qdrant():
    """Build a `QdrantClient` from env when no client was injected."""
    from qdrant_client import QdrantClient  # lazy: not needed for import or tests
    url = os.environ.get("QDRANT_URL")
    key = os.environ.get("QDRANT_API_KEY", "")
    if not key:
        try:
            import json
            home = os.path.expanduser("~")
            cfg = json.load(open(os.path.join(home, ".claude", "settings.json")))
            servers = cfg["mcpServers"]
            # Accept both the current "loci" registration name and older
            # "hermes_memory" settings.json entries.
            entry = servers.get("loci") or servers["hermes_memory"]
            key = entry["env"]["QDRANT_API_KEY"]
        except Exception:
            key = ""
    if not url:
        raise RuntimeError("QDRANT_URL is not set")
    return QdrantClient(url=url, api_key=key or None)


def _resolve_embed_fn() -> Callable[[list[str]], list[list[float]]]:
    """Default embed function: the warm-GPU nomic path from `mcp/embed_ops.py`."""
    # Add mcp/ so we reuse the exact online embed implementation.
    here = os.path.dirname(os.path.abspath(__file__))
    mcp_dir = os.path.join(os.path.dirname(here), "mcp")
    if mcp_dir not in sys.path:
        sys.path.insert(0, mcp_dir)
    import embed_ops  # noqa: E402
    return embed_ops.embed_texts


# ── small structure helpers (points may be objects or plain dicts) ──────────────

def _attr(point: Any, name: str, default=None):
    if isinstance(point, dict):
        return point.get(name, default)
    return getattr(point, name, default)


def _point_text(payload: Optional[dict], text_keys) -> str:
    if not isinstance(payload, dict):
        return ""
    for k in text_keys:
        v = payload.get(k)
        if v:
            return v if isinstance(v, str) else str(v)
    return ""


def _has_vector(point: Any, vector_name: Optional[str]) -> bool:
    vec = _attr(point, "vector")
    if vector_name:
        return isinstance(vec, dict) and bool(vec.get(vector_name))
    if isinstance(vec, dict):
        # Named-vector point when no specific name was requested.
        return any(bool(v) for v in vec.values())
    return bool(vec)


def _is_stale(point: Any, vector_name: Optional[str],
              embed_model: str, embed_version: str) -> bool:
    """A point is stale when it lacks a vector or its model/version stamp differs."""
    if not _has_vector(point, vector_name):
        return True
    payload = _attr(point, "payload") or {}
    if not isinstance(payload, dict):
        return True
    if payload.get(_MODEL_KEY) != embed_model:
        return True
    if payload.get(_VERSION_KEY) != embed_version:
        return True
    return False


def _make_point(pid, vector, payload: dict, vector_name: Optional[str]):
    """Build an upsert point, preferring qdrant's `PointStruct` and falling back to a dict."""
    vec_payload = {vector_name: vector} if vector_name else vector
    try:
        from qdrant_client.models import PointStruct  # lazy
        return PointStruct(id=pid, vector=vec_payload, payload=payload)
    except Exception:
        return {"id": pid, "vector": vec_payload, "payload": payload}


# ── core ────────────────────────────────────────────────────────────────────────

def reembed(collection: str,
            *,
            qdrant_client=None,
            embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
            apply: bool = False,
            batch_size: int = 64,
            page_limit: int = 256,
            embed_model: str = _EMBED_MODEL,
            embed_version: str = _EMBED_VERSION,
            vector_name: Optional[str] = None,
            text_keys=_TEXT_KEYS,
            logger: Optional[Callable[[str], None]] = None) -> dict:
    """Scan `collection` and, only with `apply=True`, re-embed stale points.

    Injected deps default to `None` and are resolved lazily from env:
      `qdrant_client` needs `.scroll(...) -> (points, next_offset)` and `.upsert(...)`
      `embed_fn` maps `list[str] -> list[list[float]]` and may fail open with `[]`

    Returns a fail-open report dict and never raises:
      {applied, dry_run, collection, scanned, missing_vector, stale_meta, targeted,
       reembedded, upserted_batches, embed_batches, errors:[...], degraded}
    `reembedded` counts actual writes (0 on dry-run). On dry-run, `targeted` is
    the would-write count and nothing, including the GPU, is touched.
    """
    log = logger or (lambda m: print(m, file=sys.stderr))
    report = {
        "applied": bool(apply),
        "dry_run": not apply,
        "collection": collection,
        "scanned": 0,
        "missing_vector": 0,
        "stale_meta": 0,
        "targeted": 0,        # points that qualify for re-embed (dry-run stops here)
        "reembedded": 0,      # points actually re-embedded + upserted (apply only)
        "embed_batches": 0,
        "upserted_batches": 0,
        "errors": [],
        "degraded": False,
        "embed_model": embed_model,
        "embed_version": embed_version,
    }

    # Resolve injected deps lazily. Resolution failure yields a fail-open degraded report.
    try:
        client = qdrant_client if qdrant_client is not None else _resolve_qdrant()
    except Exception as e:
        report["degraded"] = True
        report["errors"].append(f"qdrant-resolve: {e}")
        log(f"[reembed] cannot resolve qdrant client, aborting: {e}")
        return report
    ef = embed_fn if embed_fn is not None else None  # resolve only if we actually embed

    # Buffer of (point_id, text, base_payload) awaiting a batch flush.
    pending: list[tuple] = []

    def _flush(batch: list[tuple]) -> None:
        """Embed and upsert one batch. Log and return on any error."""
        if not batch:
            return
        nonlocal ef
        report["embed_batches"] += 1
        texts = [t for (_pid, t, _pl) in batch]
        try:
            if ef is None:
                ef = embed_fn if embed_fn is not None else _resolve_embed_fn()
            vecs = ef(texts)
        except Exception as e:
            report["errors"].append(f"embed: {e}")
            log(f"[reembed] embed error on batch of {len(batch)}, skipping: {e}")
            return
        if not vecs or len(vecs) != len(batch):
            report["errors"].append(
                f"embed-count: got {len(vecs) if vecs else 0} for {len(batch)}")
            log(f"[reembed] embed returned {len(vecs) if vecs else 0} vectors for "
                f"{len(batch)} texts, skipping batch (fail-open)")
            return
        points = []
        for (pid, _t, base_payload), vec in zip(batch, vecs):
            payload = dict(base_payload or {})
            payload[_MODEL_KEY] = embed_model
            payload[_VERSION_KEY] = embed_version
            points.append(_make_point(pid, vec, payload, vector_name))
        try:
            client.upsert(collection_name=collection, points=points)
        except Exception as e:
            report["errors"].append(f"upsert: {e}")
            log(f"[reembed] upsert error on batch of {len(points)}, skipping: {e}")
            return
        report["upserted_batches"] += 1
        report["reembedded"] += len(points)

    # ── scroll the collection, paging ───────────────────────────────────
    offset = None
    while True:
        try:
            points, offset = client.scroll(
                collection_name=collection,
                limit=page_limit,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
        except Exception as e:
            report["degraded"] = True
            report["errors"].append(f"scroll: {e}")
            log(f"[reembed] scroll error, stopping scan (fail-open): {e}")
            break

        for point in (points or []):
            report["scanned"] += 1
            missing = not _has_vector(point, vector_name)
            stale = _is_stale(point, vector_name, embed_model, embed_version)
            if missing:
                report["missing_vector"] += 1
            elif stale:
                report["stale_meta"] += 1
            if not stale:
                continue
            report["targeted"] += 1
            if not apply:
                continue  # DRY-RUN: count only, never buffer/embed/write
            payload = _attr(point, "payload") or {}
            text = _point_text(payload, text_keys)
            if not text:
                # No source text: note and skip (fail-open).
                report["errors"].append(f"no-text: {_attr(point, 'id')}")
                continue
            pending.append((_attr(point, "id"), text, payload))
            if len(pending) >= batch_size:
                _flush(pending)
                pending = []

        if not offset or not points:
            break

    if apply and pending:
        _flush(pending)
        pending = []

    verb = "would re-embed" if not apply else "re-embedded"
    log(f"[reembed] {collection}: scanned={report['scanned']} "
        f"missing={report['missing_vector']} stale_meta={report['stale_meta']} "
        f"targeted={report['targeted']} {verb}={report['reembedded'] if apply else report['targeted']} "
        f"errors={len(report['errors'])} dry_run={report['dry_run']}")
    return report


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--collection", required=True, help="Qdrant collection to scan")
    p.add_argument("--apply", action="store_true",
                   help="MUTATE: actually re-embed + upsert (default is dry-run)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="embed/upsert batch size (default 64, GPU-efficient)")
    p.add_argument("--page-limit", type=int, default=256,
                   help="scroll page size (default 256)")
    p.add_argument("--embed-model", default=_EMBED_MODEL)
    p.add_argument("--embed-version", default=_EMBED_VERSION)
    p.add_argument("--vector-name", default=None,
                   help="named vector to target (default: unnamed/default vector)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if not args.apply:
        print("[reembed] DRY-RUN (default): scanning only, nothing will be written. "
              "Pass --apply to mutate.", file=sys.stderr)
    report = reembed(
        args.collection,
        apply=args.apply,
        batch_size=args.batch_size,
        page_limit=args.page_limit,
        embed_model=args.embed_model,
        embed_version=args.embed_version,
        vector_name=args.vector_name,
    )
    import json
    print(json.dumps(report, indent=2, default=str))
    # Non-zero exit only on hard degrade (could not scan at all), never on skipped batches.
    return 1 if report.get("degraded") else 0


if __name__ == "__main__":
    raise SystemExit(main())
