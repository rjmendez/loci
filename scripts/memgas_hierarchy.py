#!/usr/bin/env python3
"""
MemGAS: Multi-Granularity Adaptive Search (arxiv 2505.19549)

Three-level memory hierarchy:
  L1 = utterances      (working_memory)
  L2 = summaries       (episodic_memory)
  L3 = topics          (consolidated_facts)

Entropy-weighted score fusion across levels.
"""

import concurrent.futures
import hashlib
import json
import math
import os
import sqlite3
import sys
import urllib.request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = os.environ.get(
    "MNEMOSYNE_DB",
    os.path.expanduser("~/.hermes/mnemosyne/data/mnemosyne.db"),
)
QDRANT_URL = os.environ.get("QDRANT_URL")
OLLAMA_URL = os.environ.get("OLLAMA_URL")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
TOP_K_PER_LEVEL = int(os.environ.get("TOP_K_PER_LEVEL", "3"))

COLLECTION_NAMES = {
    1: "memgas_l1",
    2: "memgas_l2",
    3: "memgas_l3",
}

VECTOR_SIZE = 768
VECTOR_DISTANCE = "Cosine"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_qdrant_api_key() -> str:
    settings_path = os.path.expanduser("~/.claude/settings.json")
    try:
        with open(settings_path) as f:
            data = json.load(f)
        # Support both top-level and nested env/apiKey patterns
        key = (
            os.environ.get("QDRANT_API_KEY")
            or data.get("qdrantApiKey")
            or data.get("env", {}).get("QDRANT_API_KEY")
            or data.get("mcpServers", {})
                   .get("hermes_memory", {})
                   .get("env", {})
                   .get("QDRANT_API_KEY")
            or data.get("mcpServers", {})
                   .get("qdrant", {})
                   .get("env", {})
                   .get("QDRANT_API_KEY")
            or ""
        )
        return key
    except Exception:
        return ""


QDRANT_API_KEY = _read_qdrant_api_key()


def _qdrant_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY
    return headers


def _http_request(method: str, url: str, body=None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=_qdrant_headers(), method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        raise RuntimeError(f"HTTP {e.code} {method} {url}: {error_body}") from e


def stable_id(s: str) -> int:
    """Deterministic integer ID from string content."""
    return int(hashlib.sha256(s.encode()).hexdigest()[:15], 16)


def embed(text: str) -> list:
    """Embed text via Ollama, return list[float]."""
    url = f"{OLLAMA_URL}/api/embeddings"
    payload = {"model": EMBED_MODEL, "prompt": text}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode())
    return result["embedding"]


def softmax(scores: list) -> list:
    m = max(scores)
    e = [math.exp(s - m) for s in scores]
    total = sum(e)
    return [x / total for x in e]


def entropy(probs: list) -> float:
    return -sum(p * math.log(p + 1e-9) for p in probs)


# ---------------------------------------------------------------------------
# Qdrant collection management
# ---------------------------------------------------------------------------

def _ensure_collection(level: int) -> None:
    name = COLLECTION_NAMES[level]
    url = f"{QDRANT_URL}/collections/{name}"
    try:
        _http_request("GET", url)
        return  # already exists
    except RuntimeError as e:
        if "404" not in str(e) and "doesn't exist" not in str(e):
            raise

    # Create collection with named vector
    body = {
        "vectors": {
            "dense": {
                "size": VECTOR_SIZE,
                "distance": VECTOR_DISTANCE,
            }
        }
    }
    _http_request("PUT", url, body)
    print(f"  Created collection {name}")


def _upsert_points(level: int, points: list) -> None:
    name = COLLECTION_NAMES[level]
    url = f"{QDRANT_URL}/collections/{name}/points"
    body = {"points": points}
    _http_request("PUT", url, body)


def _search_collection(level: int, vec: list, top_k: int) -> list:
    name = COLLECTION_NAMES[level]
    url = f"{QDRANT_URL}/collections/{name}/points/search"
    body = {
        "vector": {"name": "dense", "vector": vec},
        "limit": top_k,
        "with_payload": True,
    }
    result = _http_request("POST", url, body)
    return result.get("result", [])


# ---------------------------------------------------------------------------
# SQL queries per level
# ---------------------------------------------------------------------------

LEVEL_QUERIES = {
    1: "SELECT id, content, importance FROM working_memory WHERE content IS NOT NULL LIMIT 500",
    2: "SELECT id, content, importance FROM episodic_memory WHERE content IS NOT NULL LIMIT 200",
    3: (
        "SELECT id, (subject || ' ' || predicate || ' ' || object) AS content, "
        "confidence AS importance FROM consolidated_facts LIMIT 200"
    ),
}


def _fetch_rows(conn: sqlite3.Connection, level: int) -> list:
    cur = conn.cursor()
    try:
        cur.execute(LEVEL_QUERIES[level])
        return cur.fetchall()
    except sqlite3.OperationalError:
        return []


# ---------------------------------------------------------------------------
# memgas_index
# ---------------------------------------------------------------------------

def memgas_index(conn: sqlite3.Connection) -> None:
    """Sync all 3 memory levels to Qdrant collections."""
    for level in (1, 2, 3):
        print(f"\n[L{level}] Syncing {COLLECTION_NAMES[level]} ...")
        _ensure_collection(level)

        rows = _fetch_rows(conn, level)
        print(f"  Fetched {len(rows)} rows from SQLite")
        if not rows:
            continue

        points = []
        for row_id, content, importance in rows:
            if not content or not content.strip():
                continue
            try:
                vec = embed(content)
            except Exception as exc:
                print(f"  WARNING: embed failed for row {row_id}: {exc}")
                continue

            sid = stable_id(content)
            points.append({
                "id": sid,
                "vector": {"dense": vec},
                "payload": {
                    "source_id": row_id,
                    "content": content,
                    "importance": importance,
                    "level": level,
                },
            })

        if points:
            _upsert_points(level, points)
            print(f"  Upserted {len(points)} points")
        else:
            print("  No embeddable content found")

    print("\n[memgas_index] Done.")


# ---------------------------------------------------------------------------
# memgas_search
# ---------------------------------------------------------------------------

def _search_level(level: int, vec: list, top_k: int) -> tuple:
    """Return (level, hits_list). Swallows errors to allow partial results."""
    try:
        hits = _search_collection(level, vec, top_k)
        return level, hits
    except Exception as exc:
        print(f"  WARNING: search L{level} failed: {exc}", file=sys.stderr)
        return level, []


def memgas_search(query: str) -> list:
    """
    Entropy-weighted multi-level search.

    Returns list of dicts sorted by weighted_score desc.
    """
    vec = embed(query)

    # Parallel search across all 3 levels
    level_hits = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_search_level, level, vec, TOP_K_PER_LEVEL): level
            for level in (1, 2, 3)
        }
        for future in concurrent.futures.as_completed(futures):
            level, hits = future.result()
            level_hits[level] = hits

    # Compute entropy-based weight per level
    level_weights = {}
    for level in (1, 2, 3):
        hits = level_hits.get(level, [])
        scores = [h.get("score", 0.0) for h in hits]
        if len(scores) < 2:
            # Single or zero hits: assign neutral weight (entropy = 0 means certain -> weight=1)
            weight = 1.0
        else:
            probs = softmax(scores)
            ent = entropy(probs)
            weight = 1.0 / (1.0 + ent)
        level_weights[level] = weight

    # Normalize weights across levels
    total_weight = sum(level_weights.values())
    if total_weight == 0:
        total_weight = 1.0
    norm_weights = {lvl: w / total_weight for lvl, w in level_weights.items()}

    # Merge hits with weighted scores
    merged = []
    for level in (1, 2, 3):
        hits = level_hits.get(level, [])
        w = norm_weights[level]
        for hit in hits:
            raw_score = hit.get("score", 0.0)
            weighted = raw_score * w
            payload = hit.get("payload", {})
            merged.append({
                "level": level,
                "id": hit.get("id"),
                "score": raw_score,
                "weighted_score": weighted,
                "level_weight": w,
                "content": payload.get("content", ""),
                "importance": payload.get("importance"),
                "source_id": payload.get("source_id"),
            })

    merged.sort(key=lambda x: x["weighted_score"], reverse=True)
    return merged[: TOP_K_PER_LEVEL * 3]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    conn = sqlite3.connect(DB_PATH)

    try:
        if "--index" in sys.argv:
            memgas_index(conn)

        elif "--search" in sys.argv:
            idx = sys.argv.index("--search")
            query_parts = sys.argv[idx + 1 :]
            if not query_parts:
                print("Usage: memgas_hierarchy.py --search <query terms>", file=sys.stderr)
                sys.exit(1)
            query = " ".join(query_parts)
            results = memgas_search(query)
            for i, r in enumerate(results, 1):
                print(
                    f"[{i}] L{r['level']} score={r['score']:.4f} "
                    f"weighted={r['weighted_score']:.4f} "
                    f"w={r['level_weight']:.4f} | {r['content'][:120]}"
                )

        else:
            print(
                "Usage:\n"
                "  memgas_hierarchy.py --index\n"
                "  memgas_hierarchy.py --search <query>",
                file=sys.stderr,
            )
            sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
