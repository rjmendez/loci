#!/usr/bin/env python3
"""
Sync Mnemosyne SQLite memories -> Qdrant `mnemosyne` collection.
Embedding path: Ollama /v1/embeddings (primary, direct) or embed-worker (fallback).
Run standalone or from cron.

New and edited memories are (re-)embedded. Deleting points this host wrote for memories since
removed from SQLite is opt-in (--prune): it needs a non-empty HERMES_AGENT_ID and HERMES_PROFILE
and refuses to remove more than PRUNE_MAX_FRACTION of this host's points unless --force-prune
is also passed. Exits non-zero when the DB, Qdrant or the embedder fails, so cron does not
record success.
"""
import sqlite3, json, hashlib, sys, time, subprocess, os, base64, urllib.request

# Load .env before anything else — override path with LOCI_ENV_FILE
_ENV_FILE = os.path.expanduser(os.environ.get("LOCI_ENV_FILE")
                             or os.environ.get("HERMES_ENV_FILE") or "~/.hermes/.env")
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
# A single --prune run may delete at most this fraction of this host's mirrored points.
PRUNE_MAX_FRACTION = 0.5

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
    `loci_sessions` sat absent on a host while every consumer reported success.

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
    A missing table is skipped rather than fatal -- schemas differ across hosts. Any other
    error reading a table that exists (a missing column, a locked or corrupt DB) raises
    SyncError: reading it as "no memories" would make every mirrored point look orphaned.
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
            if not str(e).startswith("no such table"):
                raise SyncError(f"cannot read {table}: {e}") from e
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

class SyncError(RuntimeError):
    """A backend call failed in a way that makes this run's result untrustworthy."""


def get_synced_points():
    """Return {memory_id: point} for every point this script wrote into the collection.

    Uses paginated scroll to handle collections with >10000 points. Only points carrying a
    ``memory_id`` payload are returned -- other writers (ebbinghaus, agentHER) use their own
    payload keys and are never touched by the mirror. Raises SyncError when a scroll page
    fails: curl() turns every HTTP/network error into {}, and reading that as "nothing synced
    yet" is how an unreachable Qdrant used to end in "Sync complete" and exit 0.
    """
    synced = {}
    offset = None
    while True:
        body = {"limit": 1000, "with_vector": False,
                "with_payload": ["memory_id", "content", "agent_id", "profile"]}
        if offset:
            body["offset"] = offset
        data = curl("POST", f"{QDRANT}/collections/{COLLECTION}/points/scroll", body)
        if data.get("status") != "ok" or not isinstance(data.get("result"), dict):
            raise SyncError(f"scroll of {COLLECTION} failed: {data or 'no response'}")
        for p in data["result"].get("points", []):
            payload = p.get("payload") or {}
            mid = payload.get("memory_id")
            if mid:
                synced[mid] = {**payload, "id": p.get("id")}
        offset = data["result"].get("next_page_offset")
        if not offset:
            break
    return synced


def plan_sync(memories, synced):
    """Split the work into (to_upsert, orphan_point_ids).

    to_upsert: memories missing from Qdrant, or whose stored content no longer matches SQLite
    (an edited memory used to be skipped by memory_id and keep its stale vector forever).
    orphan_point_ids: points this script wrote under this host's agent_id/profile whose memory
    no longer exists in SQLite (forgotten or retracted). Points written under a different
    agent_id/profile belong to another host sharing the collection and are left alone.
    """
    current = {m["id"] for m in memories}
    to_upsert = [
        m for m in memories
        if m["id"] not in synced
        or (synced[m["id"]].get("content") or "") != m["content"][:2048]
    ]
    orphans = [
        pt["id"] for mid, pt in synced.items()
        if mid not in current
        and pt.get("id") is not None
        and (pt.get("agent_id") or "") == AGENT_ID
        and (pt.get("profile") or "") == PROFILE
    ]
    return to_upsert, orphans


def main(argv=None):
    """Mirror SQLite into Qdrant. Returns the exit code: 0 only if every step succeeded.

    Pass --prune to delete points whose memory is gone from SQLite (off by default; --no-prune
    is still accepted and means the default). --force-prune lifts the PRUNE_MAX_FRACTION cap.
    """
    argv = sys.argv[1:] if argv is None else argv
    prune = "--prune" in argv and "--no-prune" not in argv
    force_prune = "--force-prune" in argv

    if not os.path.exists(MNEMOSYNE_DB):
        # sqlite3.connect() would silently create an empty DB here, and an empty source would
        # then read as "every memory was deleted".
        print(f"[mnemosyne->qdrant] ERROR: Mnemosyne DB not found at {MNEMOSYNE_DB}",
              file=sys.stderr)
        return 1

    ensure_collection()

    conn = sqlite3.connect(f"file:{MNEMOSYNE_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        tables_present = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('memories', 'working_memory')")]
        memories = load_memories(conn)
    except (SyncError, sqlite3.Error) as e:
        print(f"[mnemosyne->qdrant] ERROR: {e}; refusing to sync", file=sys.stderr)
        return 1
    finally:
        conn.close()
    print(f"[mnemosyne->qdrant] Found {len(memories)} memories")
    if not tables_present:
        # Without a source table an empty result would prune the whole mirror.
        print("[mnemosyne->qdrant] ERROR: no memory tables in the Mnemosyne DB; refusing to sync",
              file=sys.stderr)
        return 1

    try:
        synced = get_synced_points()
    except SyncError as e:
        print(f"[mnemosyne->qdrant] ERROR: {e}", file=sys.stderr)
        return 1
    to_sync, orphans = plan_sync(memories, synced)
    # Count the actual delta, not len(memories) - len(synced): synced is every point in the
    # collection, so that subtraction goes negative once the index outgrows whatever this run
    # happens to read.
    print(f"[mnemosyne->qdrant] {len(synced)} already synced, {len(to_sync)} to add/update, "
          f"{len(orphans)} orphaned")

    failures = 0
    host_points = sum(1 for pt in synced.values()
                      if (pt.get("agent_id") or "") == AGENT_ID
                      and (pt.get("profile") or "") == PROFILE)
    if orphans and prune and not (AGENT_ID and PROFILE):
        # Ownership is decided by agent_id/profile; with either unset, every host running on
        # defaults matches every other such host's points and would prune them.
        failures += 1
        print(f"[mnemosyne->qdrant] ERROR: --prune needs HERMES_AGENT_ID and HERMES_PROFILE set; "
              f"leaving {len(orphans)} orphaned points", file=sys.stderr)
    elif orphans and prune and not force_prune and len(orphans) > PRUNE_MAX_FRACTION * host_points:
        failures += 1
        print(f"[mnemosyne->qdrant] ERROR: refusing to delete {len(orphans)} of {host_points} "
              f"points this host mirrored (cap {PRUNE_MAX_FRACTION:.0%}); rerun with "
              "--force-prune if that is intended", file=sys.stderr)
    elif orphans and prune:
        res = curl("POST", f"{QDRANT}/collections/{COLLECTION}/points/delete", {"points": orphans})
        if res.get("status") == "ok":
            print(f"[mnemosyne->qdrant] deleted {len(orphans)} points whose memory is gone")
        else:
            failures += 1
            print(f"[mnemosyne->qdrant] deleting {len(orphans)} orphaned points FAILED {res}",
                  file=sys.stderr)
    elif orphans:
        print(f"[mnemosyne->qdrant] pruning off (pass --prune): leaving {len(orphans)} "
              "orphaned points")

    total = 0
    for i in range(0, len(to_sync), BATCH):
        batch = to_sync[i:i+BATCH]
        chunks = [{"id": m["id"], "text": m["content"][:2048]} for m in batch]
        id_to_vec = embed_and_get_vector(chunks)

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

        if len(points) < len(batch):
            failures += 1
            print(f"  Batch {i//BATCH}: embedding failed for {len(batch) - len(points)} "
                  f"of {len(batch)}", file=sys.stderr)
        if not points:
            continue
        res = curl("PUT", f"{QDRANT}/collections/{COLLECTION}/points", {"points": points})
        if res.get("status") == "ok":
            total += len(points)
            print(f"  Batch {i//BATCH}: upserted {len(points)} OK (total={total})")
        else:
            failures += 1
            print(f"  Batch {i//BATCH}: FAILED {res}", file=sys.stderr)
        time.sleep(0.2)  # mild rate limiting

    if failures:
        print(f"[mnemosyne->qdrant] Sync INCOMPLETE: {total} points added/updated, "
              f"{failures} step(s) failed.", file=sys.stderr)
        return 1
    print(f"[mnemosyne->qdrant] Sync complete. {total} points added/updated.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
