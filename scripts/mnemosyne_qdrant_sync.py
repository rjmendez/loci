#!/usr/bin/env python3
"""
Sync Mnemosyne SQLite memories -> Qdrant `mnemosyne` collection.
Embedding path: Ollama /v1/embeddings (primary, direct) or embed-worker (fallback).
Run standalone or from cron.
"""
import sqlite3, json, hashlib, sys, time, subprocess, os, base64, urllib.request

# Load .env before anything else — override path with HERMES_ENV_FILE
_ENV_FILE = os.path.expanduser(os.environ.get("HERMES_ENV_FILE", "~/.hermes/.env"))
if os.path.exists(_ENV_FILE):
    with open(_ENV_FILE) as _env_fh:
        for _line in _env_fh:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

QDRANT       = os.environ.get("QDRANT_URL")
EMBED_WORKER = os.environ.get("EMBED_WORKER_URL")
OLLAMA_BASE  = os.environ.get("MNEMOSYNE_EMBEDDING_API_URL")
EMBED_MODEL  = os.environ.get("MNEMOSYNE_EMBEDDING_MODEL", "nomic-embed-text")
MNEMOSYNE_DB = os.path.expanduser(os.environ.get("MNEMOSYNE_DATA_DIR", "~/.hermes/mnemosyne/data") + "/mnemosyne.db")
COLLECTION   = "mnemosyne"
BATCH        = 8
AGENT_ID     = os.environ.get("HERMES_AGENT_ID", "")
PROFILE      = os.environ.get("HERMES_PROFILE", "")

def get_key():
    env_key = os.environ.get("QDRANT_API_KEY", "")
    if env_key:
        return env_key
    try:
        r = subprocess.run(
            ["kubectl", "get", "secret", "qdrant-secret", "-n", "default",
             "-o", "jsonpath={.data.qdrant-api-key}"],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0 and r.stdout.strip():
            return base64.b64decode(r.stdout.strip()).decode()
    except Exception:
        pass
    return ""

KEY = get_key()

def curl(method, url, data=None, extra_headers=None):
    headers = {"Content-Type": "application/json", "api-key": KEY}
    if extra_headers:
        headers.update(extra_headers)
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read()) if r.length != 0 else {}
    except Exception as e:
        print(f"  [curl] {method} {url} error: {e}", file=sys.stderr)
        return {}

def embed_via_ollama(texts):
    """Embed a list of texts via Ollama /v1/embeddings. Returns list of vectors."""
    results = []
    for text in texts:
        data = json.dumps({"model": EMBED_MODEL, "input": text[:2048]}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/embeddings",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                resp = json.loads(r.read())
            vec = (resp.get("data") or [{}])[0].get("embedding") or resp.get("embedding")
            results.append(vec)
        except Exception as e:
            print(f"  [embed] ollama error: {e}", file=sys.stderr)
            results.append(None)
    return results

def embed_and_get_vector(chunks):
    """Embed chunks. Returns {chunk_id: vector}."""
    texts = [c["text"] for c in chunks]
    vecs  = embed_via_ollama(texts)
    return {c["id"]: v for c, v in zip(chunks, vecs) if v}

def stable_num_id(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:15], 16)

def ensure_collection(name=None, dim=None):
    """Create the collection if it does not exist.

    Nothing in this repo ever created one -- every sync path only upserts points, and
    a2a_server's _qdrant_search swallows a 404 and returns [], so a missing collection
    surfaces as "0 results" rather than an error. That is how `mnemosyne` and
    `hermes_sessions` sat absent on a host while every consumer reported success.

    Schema matches what the consumers require: a NAMED "dense" vector, Cosine.
    """
    name = name or COLLECTION
    dim  = int(dim or os.environ.get("MNEMOSYNE_EMBEDDING_DIM", 768))
    # curl() swallows every exception and returns {} -- the same fail-quiet habit that let this
    # bug hide -- so probe by response shape rather than by catching HTTPError, which never
    # reaches us. A present collection answers {"result": {...}, "status": "ok"}.
    if curl("GET", f"{QDRANT}/collections/{name}").get("result"):
        return False
    curl("PUT", f"{QDRANT}/collections/{name}",
         {"vectors": {"dense": {"size": dim, "distance": "Cosine"}}})
    # Re-read: PUT on an existing collection conflicts and curl() would hide that too, so
    # confirm the collection is actually there before the caller starts upserting into it.
    if not curl("GET", f"{QDRANT}/collections/{name}").get("result"):
        print(f"[mnemosyne->qdrant] WARNING: {name} still missing after create attempt",
              file=sys.stderr)
        return False
    print(f"[mnemosyne->qdrant] created missing collection {name} (dense {dim}d Cosine)")
    return True

def load_memories(conn, tables=(("memories", "memory"), ("working_memory", "working"))):
    """Read every memory tier out of the Mnemosyne DB, newest tier precedence first.

    `conn` must have row_factory = sqlite3.Row.

    This used to be an inline SELECT against working_memory alone -- the short-lived staging
    tier. On a real node that is a couple of rows while the corpus sits in `memories` (137 vs 2
    on hugbot5000-jetson), so the sync ran clean, printed a success line, and left the semantic
    index essentially empty.

    Rows with empty content are dropped (nothing to embed). An id present in more than one tier
    keeps its FIRST occurrence, so the durable `memories` copy wins over a staging duplicate.
    A missing table is skipped rather than fatal -- schemas differ across hosts.
    """
    out, seen = [], set()
    _ALLOWED_TABLES = {"memories", "working_memory"}
    for table, tier in tables:
        if table not in _ALLOWED_TABLES:
            print(f"[mnemosyne->qdrant] skipping unknown table {table}")
            continue
        try:
            rows = conn.execute(
                f"SELECT id, content, source, importance, session_id, created_at FROM {table} "
                "WHERE content IS NOT NULL AND TRIM(content) != '' ORDER BY created_at ASC"
            ).fetchall()
        except sqlite3.OperationalError as e:
            print(f"[mnemosyne->qdrant] skipping {table}: {e}")
            continue
        kept = 0
        for r in rows:
            d = dict(r)
            if d["id"] in seen:
                continue
            seen.add(d["id"])
            d["tier"] = tier
            out.append(d)
            kept += 1
        print(f"[mnemosyne->qdrant]   {table}: {len(rows)} rows, {kept} new")
    return out

def get_synced_ids():
    """Return set of memory_ids already in the mnemosyne Qdrant collection.
    Uses paginated scroll to handle collections with >10000 points."""
    synced = set()
    offset = None
    while True:
        body = {"limit": 1000, "with_payload": ["memory_id"], "with_vector": False}
        if offset:
            body["offset"] = offset
        data = curl("POST", f"{QDRANT}/collections/{COLLECTION}/points/scroll", body)
        pts = data.get("result", {}).get("points", [])
        for p in pts:
            mid = (p.get("payload") or {}).get("memory_id")
            if mid:
                synced.add(mid)
        offset = data.get("result", {}).get("next_page_offset")
        if not offset:
            break
    return synced

def main():
    ensure_collection()

    conn = sqlite3.connect(MNEMOSYNE_DB)
    conn.row_factory = sqlite3.Row
    memories = load_memories(conn)
    conn.close()
    print(f"[mnemosyne->qdrant] Found {len(memories)} memories")

    # Check what's already in mnemosyne collection (paginated)
    existing_ids = get_synced_ids()
    to_sync = [m for m in memories if m["id"] not in existing_ids]
    # Count the actual delta, not len(memories) - len(existing_ids): existing_ids is every
    # point in the collection, so that subtraction goes negative once the index outgrows
    # whatever this run happens to read.
    print(f"[mnemosyne->qdrant] {len(existing_ids)} already synced, {len(to_sync)} to add")
    if not to_sync:
        print("[mnemosyne->qdrant] All up to date.")
        return

    total = 0
    for i in range(0, len(to_sync), BATCH):
        batch = to_sync[i:i+BATCH]
        chunks = [{"id": m["id"], "text": m["content"][:2048]} for m in batch]
        id_to_vec = embed_and_get_vector(chunks)
        if not id_to_vec:
            print(f"  Batch {i//BATCH}: embed returned nothing, skipping")
            continue

        points = []
        for m in batch:
            vec = id_to_vec.get(m["id"])
            if not vec:
                continue
            points.append({
                "id": stable_num_id(m["id"]),
                "vector": {"dense": vec},
                "payload": {
                    "memory_id": m["id"],
                    "content": m["content"][:2048],
                    "source": m["source"] or "conversation",
                    "importance": float(m["importance"] or 0.5),
                    "bank": m["session_id"] or "default",
                    "created_at": m["created_at"] or "",
                    "agent_id": AGENT_ID,
                    "profile": PROFILE,
                }
            })

        if not points:
            continue
        res = curl("PUT", f"{QDRANT}/collections/{COLLECTION}/points", {"points": points})
        if res.get("status") == "ok":
            total += len(points)
            print(f"  Batch {i//BATCH}: upserted {len(points)} OK (total={total})")
        else:
            print(f"  Batch {i//BATCH}: FAILED {res}")
        time.sleep(0.2)  # mild rate limiting

    print(f"[mnemosyne->qdrant] Sync complete. {total} new points added.")

if __name__ == "__main__":
    main()
