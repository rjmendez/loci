#!/usr/bin/env python3
"""on_session_end hook: upsert the current session into Qdrant loci_sessions.

Reads session_id/transcript_path from stdin JSON, takes messages from state.db
(or the Claude Code transcript when the id isn't there), embeds them, upserts one point.
Env: QDRANT_URL, QDRANT_API_KEY, OLLAMA_BASE_URL / MNEMOSYNE_EMBEDDING_API_URL, LOCI_STATE_DB.
Fail-open: every error exits 0; an unchanged message count exits before any network call.
"""
import json, sys, os, sqlite3, hashlib, datetime, time
import urllib.request, urllib.error

# Accept the legacy HERMES_* spelling (legacy_env.py is a sibling in both layouts).
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from legacy_env import apply as _apply_legacy_env
    _apply_legacy_env()
except Exception:
    pass

STATE_DB   = os.path.expanduser(os.environ.get("LOCI_STATE_DB", "~/.hermes/state.db"))
QDRANT      = os.environ.get("QDRANT_URL")
QDRANT_KEY  = os.environ.get("QDRANT_API_KEY", "")
def _embeddings_url() -> "str | None":
    """OLLAMA_BASE_URL is a bare host; MNEMOSYNE_EMBEDDING_API_URL is already a /v1 endpoint."""
    base = os.environ.get("OLLAMA_BASE_URL")
    if base:
        return f"{base.rstrip('/')}/v1/embeddings"
    api = os.environ.get("MNEMOSYNE_EMBEDDING_API_URL")
    return f"{api.rstrip('/')}/embeddings" if api else None


OLLAMA      = _embeddings_url()
EMBED_MODEL           = os.environ.get("MNEMOSYNE_EMBEDDING_MODEL", "nomic-embed-text")
_EMBED_API_KEY        = os.environ.get("EMBED_API_KEY", "")
_EMBED_API_KEY_HEADER = os.environ.get("EMBED_API_KEY_HEADER", "Authorization")
COLLECTION  = "loci_sessions"
EMBED_DIM   = int(os.environ.get("MNEMOSYNE_EMBEDDING_DIM", "768"))
MAX_CHARS   = 4000   # chars of session content to embed
CACHE_DIR   = os.path.expanduser(os.environ.get("LOCI_SYNC_CACHE", "~/.hermes/.session_sync_cache"))
AGENT_ID    = os.environ.get("HERMES_AGENT_ID", "")
PROFILE     = os.environ.get("HERMES_PROFILE", "")
ACTIVE_INV  = os.environ.get("LOCI_ACTIVE_INVESTIGATION", "")

def ensure_collection():
    """Create loci_sessions (or re-tune it) via raw REST, keeping the hook zero-dep. Non-fatal."""
    if not QDRANT:
        return
    url = f"{QDRANT}/collections/{COLLECTION}"
    headers = {"Content-Type": "application/json", "api-key": QDRANT_KEY}

    exists = False
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=5):
            exists = True
    except urllib.error.HTTPError as e:
        exists = e.code != 404
    except Exception:
        return  # Qdrant unreachable — let upsert fail with its own error

    _hnsw = {"m": 32, "ef_construct": 200, "on_disk": False}
    _quant = {"scalar": {"type": "int8", "quantile": 0.99, "always_ram": True}}

    if not exists:
        body = json.dumps({
            "vectors": {"dense": {"size": EMBED_DIM, "distance": "Cosine"}},
            "hnsw_config": _hnsw,
            "quantization_config": _quant,
        }).encode()
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="PUT")
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception as e:
            print(f"[session_end_sync] create collection failed: {e}", file=sys.stderr)
    else:
        # Idempotent; applied on the next optimizer pass without pausing writes.
        body = json.dumps({
            "hnsw_config": {"m": 32, "ef_construct": 200},
            "quantization_config": _quant,
        }).encode()
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="PATCH")
            with urllib.request.urlopen(req, timeout=10):
                pass
        except Exception:
            pass  # Non-fatal — older Qdrant may not support all PATCH fields


def stable_id(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:15], 16)

def read_stdin_payload() -> dict:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return {}
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def read_stdin_session_id() -> str:
    # session_id is top-level in both the Hermes and Claude Code payloads.
    sid = read_stdin_payload().get("session_id")
    return sid if isinstance(sid, str) else ""

def get_session_content(session_id: str):
    """Return the session as a dict (title, started_at, source, model, content, msg_count), or None."""
    if not os.path.exists(STATE_DB):
        return None
    try:
        conn = sqlite3.connect(f"file:{STATE_DB}?mode=ro", uri=True,
                               timeout=3.0)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT title, started_at, source, model FROM sessions WHERE id=?",
            (session_id,)
        ).fetchone()
        if not row:
            conn.close()
            return None
        msgs = conn.execute(
            """SELECT role, content FROM messages
               WHERE session_id=? AND role IN ('user','assistant')
                 AND content IS NOT NULL AND length(content) > 20
               ORDER BY timestamp ASC""",
            (session_id,)
        ).fetchall()
        msg_count = len(msgs)
        conn.close()
        if not msgs:
            return None

        # Taken from the end so recent context drives the embedding.
        lines = [f"{m['role'].upper()}: {(m['content'] or '').strip()}" for m in msgs]
        sep = "\n\n"
        buf = []
        total = 0
        for line in reversed(lines):
            joiner = len(sep) if buf else 0     # the separators count against the budget too
            if total + joiner + len(line) > MAX_CHARS:
                room = MAX_CHARS - total - joiner
                if room > 0:                    # never join in an empty line
                    buf.append(line[:room])
                break
            buf.append(line)
            total += joiner + len(line)
        content = sep.join(reversed(buf))

        try:
            dt = datetime.datetime.fromtimestamp(float(row["started_at"]), tz=datetime.timezone.utc).isoformat()
        except Exception:
            dt = str(row["started_at"])

        return {
            "title":      row["title"] or "",
            "started_at": dt,
            "source":     row["source"] or "cli",
            "model":      row["model"] or "",
            "content":    content,
            "msg_count":  msg_count,
        }
    except Exception:
        return None


def _transcript_text(message) -> str:
    """User content is a string; assistant content is a block list (keep text blocks only)."""
    content = (message or {}).get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content
                 if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(x for x in parts if x).strip()
    return ""


def transcript_session_content(transcript_path: str, session_id: str):
    """Like get_session_content, from a Claude Code transcript (STATE_DB only knows Hermes ids).

    Streams the file: transcripts reach tens of MB and only the last MAX_CHARS is embedded.
    """
    if not transcript_path or not os.path.exists(transcript_path):
        return None
    title = model = ""
    started_at = ""
    msg_count = 0
    tail: list[str] = []
    tail_chars = 0
    try:
        with open(transcript_path, "r", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(rec, dict):
                    continue
                kind = rec.get("type")
                if kind == "ai-title" and not title:
                    title = str(rec.get("aiTitle") or "")
                    continue
                if kind not in ("user", "assistant"):
                    continue
                msg = rec.get("message")
                if not isinstance(msg, dict):
                    continue
                if kind == "assistant" and not model:
                    model = str(msg.get("model") or "")
                if not started_at and rec.get("timestamp"):
                    started_at = str(rec["timestamp"])
                text = _transcript_text(msg)
                if len(text) <= 20:
                    continue
                msg_count += 1
                entry = f"{kind.upper()}: {text}"
                tail.append(entry)
                tail_chars += len(entry)
                # bounded tail: recent context drives the embedding
                while tail_chars > MAX_CHARS * 2 and len(tail) > 1:
                    tail_chars -= len(tail.pop(0))
    except OSError:
        return None

    if not msg_count:
        return None

    buf: list[str] = []
    total = 0
    for entry in reversed(tail):
        if total + len(entry) > MAX_CHARS:
            buf.append(entry[:MAX_CHARS - total])
            break
        buf.append(entry)
        total += len(entry)

    return {
        "title":      title,
        "started_at": started_at,
        "source":     "claude-code",
        "model":      model,
        "content":    "\n\n".join(reversed(buf)),
        "msg_count":  msg_count,
    }

# One cache file per session id; the TTL stops the directory growing forever.
CACHE_TTL_DAYS = int(os.environ.get("LOCI_SYNC_CACHE_TTL_DAYS", "7"))


def prune_cache(now: float | None = None) -> int:
    """Delete cache entries older than CACHE_TTL_DAYS; return the count. Fail-open per file."""
    if CACHE_TTL_DAYS <= 0:
        return 0
    cutoff = (now if now is not None else time.time()) - CACHE_TTL_DAYS * 86400
    removed = 0
    try:
        entries = os.listdir(CACHE_DIR)
    except OSError:
        return 0
    for name in entries:
        fp = os.path.join(CACHE_DIR, name)
        try:
            if os.path.isfile(fp) and os.path.getmtime(fp) < cutoff:
                os.unlink(fp)
                removed += 1
        except OSError:
            continue
    return removed


def cache_path(session_id: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, hashlib.md5(session_id.encode()).hexdigest()[:12])

def cached_msg_count(session_id: str) -> int:
    try:
        # inside the try: an unusable cache dir only loses the fast path
        p = cache_path(session_id)
        with open(p) as f:
            return int(f.read().strip())
    except Exception:
        return -1

def write_cache(session_id: str, count: int):
    try:
        with open(cache_path(session_id), "w") as f:
            f.write(str(count))
    except Exception:
        pass

def _embed_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if _EMBED_API_KEY:
        if _EMBED_API_KEY_HEADER.lower() == "authorization":
            h["Authorization"] = f"Bearer {_EMBED_API_KEY}"
        else:
            h[_EMBED_API_KEY_HEADER] = _EMBED_API_KEY
    return h


def embed(text: str) -> list:
    if not OLLAMA:
        raise RuntimeError("no embeddings endpoint configured "
                           "(set OLLAMA_BASE_URL or MNEMOSYNE_EMBEDDING_API_URL)")
    body = json.dumps({"model": EMBED_MODEL, "input": text}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers=_embed_headers(), method="POST")
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read())
    return data["data"][0]["embedding"]

def qdrant_upsert(point_id: int, vector: list, payload: dict):
    body = json.dumps({
        "points": [{
            "id": point_id,
            "vector": {"dense": vector},
            "payload": payload,
        }]
    }).encode()
    req = urllib.request.Request(
        f"{QDRANT}/collections/{COLLECTION}/points",
        data=body,
        headers={
            "Content-Type": "application/json",
            "api-key": QDRANT_KEY,
        },
        method="PUT"
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        result = json.loads(resp.read())
    return result.get("status") == "ok"

def _check_wiring_obligations(investigation_id: str, payload: dict) -> "str | None":
    """Count unresolved wiring_obligation gaps in findings.jsonl and add them to `payload`.

    Returns "" when nothing is outstanding and None when the check could not run
    (a failure must not read as a clean line). Non-fatal.
    """
    loci_dir = os.path.expanduser(
        os.environ.get("LOCI_INVESTIGATIONS_DIR", "~/.loci/investigations")
    )
    findings_file = os.path.join(loci_dir, investigation_id, "findings.jsonl")
    if not os.path.exists(findings_file):
        return ""

    unresolved = []
    seen_ids: set = set()
    try:
        with open(findings_file) as f:
            lines = f.readlines()
        for line in reversed(lines):
            try:
                rec = json.loads(line.strip())
            except Exception:
                continue
            # A legacy access row reuses the finding's id; it must not shadow it.
            if (rec.get("record_type") or rec.get("type") or "") == "access":
                continue
            fid = rec.get("id", "")
            # Only a real id can be deduplicated; id-less records are distinct.
            if fid:
                if fid in seen_ids:
                    continue
                seen_ids.add(fid)
            tags = rec.get("tags", [])
            if "wiring_obligation" in tags and rec.get("record_type") == "gap":
                unresolved.append(str(rec.get("text") or fid)[:120])
    except Exception as e:
        print(f"[session_end_sync] wiring-obligation check failed: {e}", file=sys.stderr)
        return None

    count = len(unresolved)
    if count == 0:
        return ""

    payload["unresolved_wiring_obligations"] = count
    payload["unresolved_wiring_obligation_samples"] = unresolved[:3]
    return f" | ⚠ UNRESOLVED WIRING OBLIGATIONS: {count}"


def main():
    t0 = time.monotonic()
    payload = read_stdin_payload()
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        sys.exit(0)

    # This hook is the cache's only writer, so it must prune it.
    try:
        gone = prune_cache()
        if gone:
            print(f"[session_end_sync] pruned {gone} cache entr"
                  f"{'y' if gone == 1 else 'ies'} older than {CACHE_TTL_DAYS}d",
                  file=sys.stderr)
    except Exception as e:
        print(f"[session_end_sync] cache prune skipped: {e}", file=sys.stderr)

    sess = get_session_content(session_id) or transcript_session_content(
        payload.get("transcript_path") or "", session_id)
    if not sess:
        sys.exit(0)

    prev_count = cached_msg_count(session_id)
    if sess["msg_count"] == prev_count:
        sys.exit(0)     # the fast path: nothing changed, so no Qdrant round-trip at all

    ensure_collection()

    try:
        vector = embed(sess["content"])
    except Exception as e:
        print(f"[session_end_sync] embed error: {e}", file=sys.stderr)
        sys.exit(0)   # non-fatal — cron catches it later

    if len(vector) != EMBED_DIM:
        print(f"[session_end_sync] unexpected vector dim {len(vector)}, expected {EMBED_DIM}", file=sys.stderr)
        sys.exit(0)

    point_id = stable_id(session_id)
    payload = {
        "session_id":      session_id,
        "title":           sess["title"],
        "started_at":      sess["started_at"],
        "source":          sess["source"],
        "model":           sess["model"],
        "msg_count":       sess["msg_count"],
        "content_preview": sess["content"][:500],
        "last_synced":     datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "agent_id":        AGENT_ID,
        "profile":         PROFILE,
    }

    # Must run before the upsert: the check adds fields to `payload`.
    unresolved_note = ""
    if ACTIVE_INV:
        try:
            unresolved_note = _check_wiring_obligations(ACTIVE_INV, payload)
        except Exception as e:
            print(f"[session_end_sync] wiring-obligation check failed: {e}", file=sys.stderr)
            unresolved_note = None
        if unresolved_note is None:
            unresolved_note = " | wiring-obligation check FAILED"

    try:
        ok = qdrant_upsert(point_id, vector, payload)
    except Exception as e:
        print(f"[session_end_sync] upsert error: {e}", file=sys.stderr)
        sys.exit(0)

    if ok:
        write_cache(session_id, sess["msg_count"])
        elapsed = time.monotonic() - t0

        print(f"[session_end_sync] synced {session_id[:20]} ({sess['msg_count']} msgs) in {elapsed:.2f}s{unresolved_note}")

if __name__ == "__main__":
    main()
