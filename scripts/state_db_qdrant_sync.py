#!/usr/bin/env python3
"""
Sync Hermes state.db sessions -> Qdrant `hermes_sessions` collection.

Each Qdrant point = one session (unit of search), containing a
concatenation of its user+assistant messages as the embedded text.
Chunked at 4000 chars to keep embedding quality high and avoid
Ollama stall on huge inputs.

Uses embedding-worker (NodePort 30888) -> agent_core_chunks -> copy
vector -> hermes_sessions (named vector "dense", 768-dim Cosine).

Incremental: tracks synced sessions by session_id in payload.
Run standalone or from cron (no_agent=True).
"""
import sqlite3, json, hashlib, sys, os, time, subprocess, datetime

STATE_DB     = os.environ.get("HERMES_STATE_DB", os.path.expanduser("~/.hermes/state.db"))
QDRANT       = os.environ.get("QDRANT_URL")
EMBED_WORKER = os.environ.get("EMBED_WORKER_URL")
COLLECTION   = "hermes_sessions"
SRC_COLL     = "agent_core_chunks"
MAX_CHARS    = 4000   # per-session content cap before embedding
BATCH_SIZE   = 4      # sessions per embed round-trip (keep under 32-chunk Ollama limit)

AGENT_ID = os.environ.get("HERMES_AGENT_ID", "")
PROFILE  = os.environ.get("HERMES_PROFILE", "")

def get_key():
    # Primary: QDRANT_API_KEY environment variable.
    # k8s deployments: set this via a secretKeyRef in your pod spec.
    key = os.environ.get("QDRANT_API_KEY") or os.environ.get("QDRANT_KEY", "")
    return key

def curl_json(method, url, data=None, key=""):
    cmd = ["curl", "-s", "-X", method, url,
           "-H", f"api-key: {key}",
           "-H", "Content-Type: application/json"]
    if data is not None:
        cmd += ["-d", json.dumps(data)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if not r.stdout.strip():
        return {}
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"_raw": r.stdout[:200]}

def stable_id(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:15], 16)

def load_sessions_with_messages(conn):
    """Return list of dicts: session metadata + concatenated message text."""
    sessions = conn.execute(
        "SELECT id, title, started_at, source, model FROM sessions ORDER BY started_at ASC"
    ).fetchall()
    results = []
    for s in sessions:
        msgs = conn.execute(
            """SELECT role, content FROM messages
               WHERE session_id=? AND role IN ('user','assistant')
                 AND content IS NOT NULL AND length(content) > 20
               ORDER BY timestamp ASC""",
            (s["id"],)
        ).fetchall()
        if not msgs:
            continue
        # Build embedded text: role-prefixed lines, capped
        parts = []
        total = 0
        for m in msgs:
            line = f"{m['role'].upper()}: {(m['content'] or '').strip()}"
            if total + len(line) > MAX_CHARS:
                parts.append(line[:MAX_CHARS - total])
                break
            parts.append(line)
            total += len(line)
        text = "\n\n".join(parts)
        if not text.strip():
            continue
        # started_at is a float (unix epoch) in state.db
        try:
            dt = datetime.datetime.utcfromtimestamp(float(s["started_at"])).isoformat()
        except Exception:
            dt = str(s["started_at"])
        results.append({
            "session_id": s["id"],
            "title": s["title"] or "",
            "started_at": dt,
            "source": s["source"] or "cli",
            "model": s["model"] or "",
            "msg_count": len(msgs),
            "text": text,
        })
    return results

def get_synced_ids(key):
    """Return set of session_ids already in hermes_sessions."""
    synced = set()
    offset = None
    while True:
        body = {"limit": 1000, "with_payload": ["session_id"], "with_vector": False}
        if offset:
            body["offset"] = offset
        data = curl_json("POST", f"{QDRANT}/collections/{COLLECTION}/points/scroll",
                         body, key)
        pts = data.get("result", {}).get("points", [])
        for p in pts:
            sid = (p.get("payload") or {}).get("session_id")
            if sid:
                synced.add(sid)
        offset = data.get("result", {}).get("next_page_offset")
        if not offset:
            break
    return synced

def embed_batch(chunks, key):
    """Embed via worker, return {chunk_id: vector} for successfully embedded chunks."""
    r = subprocess.run(
        ["curl", "-s", "-X", "POST", f"{EMBED_WORKER}/embed",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"chunks": chunks})],
        capture_output=True, text=True, timeout=90
    )
    if not r.stdout.strip():
        return {}
    try:
        edata = json.loads(r.stdout)
    except json.JSONDecodeError:
        return {}
    if edata.get("status") != "ok":
        print(f"  [embed] error: {edata}", file=sys.stderr)
        return {}
    vectors_map = edata.get("vectors", {})   # {chunk_id: qdrant_uuid in agent_core_chunks}
    if not vectors_map:
        return {}
    # Fetch vectors back from agent_core_chunks
    uuids = list(vectors_map.values())
    fr = curl_json("POST", f"{QDRANT}/collections/{SRC_COLL}/points",
                   {"ids": uuids, "with_vector": True}, key)
    uuid_to_vec = {p["id"]: p.get("vector")
                   for p in fr.get("result", []) if p.get("vector")}
    id_to_vec = {}
    for chunk_id, uuid in vectors_map.items():
        vec = uuid_to_vec.get(uuid)
        if vec is None:
            continue
        # Handle both unnamed (list) and named (dict) vector shapes
        if isinstance(vec, list):
            id_to_vec[chunk_id] = vec
        elif isinstance(vec, dict):
            id_to_vec[chunk_id] = vec.get("dense") or next(iter(vec.values()), None)
    return id_to_vec

def upsert_points(points, key):
    """Upsert a batch of points into hermes_sessions."""
    r = curl_json("PUT", f"{QDRANT}/collections/{COLLECTION}/points",
                  {"points": points}, key)
    return r.get("status") == "ok"

def main():
    key = get_key()
    if not key:
        print("ERROR: could not retrieve Qdrant API key", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(STATE_DB)
    conn.row_factory = sqlite3.Row
    sessions = load_sessions_with_messages(conn)
    conn.close()
    print(f"[state_db->qdrant] {len(sessions)} sessions with messages in state.db")

    synced = get_synced_ids(key)
    print(f"[state_db->qdrant] {len(synced)} already synced in Qdrant")

    to_sync = [s for s in sessions if s["session_id"] not in synced]
    print(f"[state_db->qdrant] {len(to_sync)} sessions to embed and upsert")

    if not to_sync:
        print("[state_db->qdrant] nothing to do, exiting")
        return

    ok = 0
    err = 0
    for i in range(0, len(to_sync), BATCH_SIZE):
        batch = to_sync[i : i + BATCH_SIZE]
        chunks = [{"id": s["session_id"], "text": s["text"]} for s in batch]
        id_to_vec = embed_batch(chunks, key)

        points = []
        for s in batch:
            vec = id_to_vec.get(s["session_id"])
            if vec is None:
                print(f"  [skip] no vector for {s['session_id'][:20]}", file=sys.stderr)
                err += 1
                continue
            points.append({
                "id": stable_id(s["session_id"]),
                "vector": {"dense": vec},
                "payload": {
                    "session_id": s["session_id"],
                    "title":      s["title"],
                    "started_at": s["started_at"],
                    "source":     s["source"],
                    "model":      s["model"],
                    "agent_id":   AGENT_ID,
                    "profile":    PROFILE,
                    "msg_count":  s.get("msg_count", 0),
                    "last_synced": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                    "content_preview": s["text"][:500],
                }
            })

        if points and upsert_points(points, key):
            ok += len(points)
            print(f"  upserted {ok}/{len(to_sync)} ...", end="\r", flush=True)
        else:
            err += len(points)

        time.sleep(0.1)  # gentle rate limit

    print(f"\n[state_db->qdrant] done — {ok} upserted, {err} errors")

if __name__ == "__main__":
    main()
