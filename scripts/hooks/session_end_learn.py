#!/usr/bin/env python3
"""
on_session_end hook — continuous learning via RAG-backed diff.

Fires after session_end_sync.py has synced the session into Qdrant.

Algorithm
---------
1. Read session payload from stdin (session_id).
2. Load the current session's content from state.db.
3. Query /search for the top 3 most semantically similar past sessions.
4. Call Ollama (granite8b-heretic-agent) with the current session summary
   + retrieved past sessions and ask: "What did I do differently? What's new?"
5. Write the learning note to Mnemosyne working_memory
   (author_id=mrpink-learn, importance=0.6, scope=global).
   The sleep consolidation cycle will merge it into episodic memory overnight.

Design choices
--------------
- Uses /search endpoint (no Qdrant client, no special toolset) — same path
  a worker would use, so this validates Option C end-to-end every session.
- Non-fatal throughout: every network error exits 0 (no noise in hook runner).
- Timeout budget: 30s total (search 5s + Ollama 20s + Mnemosyne write 5s).
- Skips sessions with <4 messages or <400 chars (not enough signal).
- Skips if no similar sessions found (score < 0.6 threshold).
- De-dupes: writes a cache entry keyed on session_id so the note is only
  written once even if the hook fires multiple times for the same session.

Environment variables (set in config.yaml hooks block)
-------------------------------------------------------
  HERMES_STATE_DB          path to state.db
  MRPINK_SEARCH_TOKEN      read-only token for /search
  SEARCH_BASE_URL          base URL for /search (default http://127.0.0.1:8201)
  OLLAMA_BASE_URL          Ollama base (no /v1 suffix)
  LEARN_MODEL              Ollama model to use (default granite8b-heretic-agent)
  MNEMOSYNE_DB             path to mnemosyne.db
  LEARN_CACHE_DIR          path to dedup cache (default ~/.hermes/.learn_cache)
  LEARN_MIN_MSGS           skip sessions with fewer messages (default 4)
  LEARN_MIN_CHARS          skip sessions with fewer chars (default 400)
  LEARN_MIN_SCORE          only use past sessions with score >= this (default 0.6)
  LEARN_TOP_K              how many past sessions to retrieve (default 3)
"""

import json, sys, os, sqlite3, hashlib, datetime, time, textwrap
import urllib.request, urllib.error

# ── Config ──────────────────────────────────────────────────────────────────
STATE_DB     = os.path.expanduser(os.environ.get("LOCI_STATE_DB") or
                   os.environ.get("HERMES_STATE_DB",
                   "~/.hermes/profiles/mrpink/state.db"))
SEARCH_TOKEN = os.environ.get("MRPINK_SEARCH_TOKEN", "")
SEARCH_BASE  = os.environ.get("SEARCH_BASE_URL", "http://127.0.0.1:8201")
OLLAMA       = os.environ.get("OLLAMA_BASE_URL", "http://100.73.200.19:11434")
LEARN_MODEL  = os.environ.get("LEARN_MODEL", "qwen3-4b-instruct-heretic-agent")
MNEM_DB      = os.path.expanduser(os.environ.get("MNEMOSYNE_DB",
                   "~/.hermes/mnemosyne/data/mnemosyne.db"))
CACHE_DIR    = os.path.expanduser(os.environ.get("LEARN_CACHE_DIR",
                   "~/.hermes/profiles/mrpink/.learn_cache"))
MIN_MSGS     = int(os.environ.get("LEARN_MIN_MSGS", "4"))
MIN_CHARS    = int(os.environ.get("LEARN_MIN_CHARS", "400"))
MIN_SCORE    = float(os.environ.get("LEARN_MIN_SCORE", "0.6"))
TOP_K        = int(os.environ.get("LEARN_TOP_K", "3"))
MAX_CONTENT  = 3000   # chars of current session to pass to Ollama
MAX_PAST     = 800    # chars per past session snippet

LOG_PREFIX = "[session_end_learn]"


# ── Helpers ──────────────────────────────────────────────────────────────────

def log(msg: str):
    print(f"{LOG_PREFIX} {msg}", file=sys.stderr)


def http_post(url: str, body: dict, timeout: int = 8) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def http_get(url: str, timeout: int = 8) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ── Dedup cache ───────────────────────────────────────────────────────────────

def _cache_path(session_id: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR,
                        "learn_" + hashlib.md5(session_id.encode()).hexdigest()[:12])


def already_processed(session_id: str) -> bool:
    return os.path.exists(_cache_path(session_id))


def mark_processed(session_id: str):
    try:
        open(_cache_path(session_id), "w").write(
            datetime.datetime.utcnow().isoformat()
        )
    except Exception:
        pass


# ── Session content ───────────────────────────────────────────────────────────

def read_stdin_session_id() -> str:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return ""
        data = json.loads(raw)
        return data.get("session_id") or ""
    except Exception:
        return ""


def get_session_content(session_id: str) -> dict | None:
    """Return {title, started_at, content, msg_count} or None."""
    if not os.path.exists(STATE_DB):
        return None
    try:
        conn = sqlite3.connect(f"file:{STATE_DB}?mode=ro", uri=True, timeout=3.0)
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
        conn.close()
        msg_count = len(msgs)
        if msg_count < MIN_MSGS:
            return None

        lines = [f"{m['role'].upper()}: {(m['content'] or '').strip()}" for m in msgs]
        buf, total = [], 0
        for line in reversed(lines):
            if total + len(line) > MAX_CONTENT:
                buf.append(line[:(MAX_CONTENT - total)])
                break
            buf.append(line)
            total += len(line)
        content = "\n\n".join(reversed(buf))

        if len(content) < MIN_CHARS:
            return None

        try:
            dt = datetime.datetime.fromtimestamp(
                float(row["started_at"]), tz=datetime.timezone.utc
            ).isoformat()
        except Exception:
            dt = str(row["started_at"])

        return {
            "title":      row["title"] or "(untitled)",
            "started_at": dt,
            "source":     row["source"] or "cli",
            "content":    content,
            "msg_count":  msg_count,
        }
    except Exception as e:
        log(f"state.db read error: {e}")
        return None


# ── /search ───────────────────────────────────────────────────────────────────

def search_similar_sessions(query: str, exclude_session_id: str) -> list[dict]:
    """
    Returns list of {title, score, content_preview, session_id}
    sorted by score desc, filtered to score >= MIN_SCORE,
    excluding the current session.
    """
    import urllib.parse
    q = urllib.parse.quote(query)
    url = (f"{SEARCH_BASE}/search"
           f"?q={q}&top_k={TOP_K + 2}"
           f"&collections=hermes_sessions,mnemosyne"
           f"&token={SEARCH_TOKEN}")
    try:
        data = http_get(url, timeout=8)
    except urllib.error.HTTPError as e:
        log(f"/search HTTP error {e.code}")
        return []
    except Exception as e:
        log(f"/search error: {e}")
        return []

    results = []
    for hit in data.get("results", []):
        sid = (hit.get("payload") or {}).get("session_id", "")
        if sid == exclude_session_id:
            continue
        score = hit.get("score", 0)
        if score < MIN_SCORE:
            continue
        preview = hit.get("content", "")[:MAX_PAST]
        title   = (hit.get("payload") or {}).get("title", hit.get("collection", "?"))
        results.append({"title": title, "score": score, "preview": preview, "session_id": sid})
        if len(results) >= TOP_K:
            break
    return results


# ── Ollama inference ──────────────────────────────────────────────────────────

def call_ollama(prompt: str) -> str:
    body = {
        "model": LEARN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "num_ctx": 8192,
            "temperature": 0.3,
        },
    }
    url = f"{OLLAMA}/v1/chat/completions"
    try:
        resp = http_post(url, body, timeout=45)
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log(f"Ollama error: {e}")
        return ""


# ── Mnemosyne write ───────────────────────────────────────────────────────────

def write_learning_note(session_id: str, session_title: str, note: str,
                        similar_count: int):
    """Insert into working_memory directly (A2A memory_remember would also work)."""
    if not os.path.exists(MNEM_DB):
        log("mnemosyne.db not found — skipping write")
        return False
    note_id = "learn_" + hashlib.sha256(
        (session_id + note[:50]).encode()
    ).hexdigest()[:20]
    ts = datetime.datetime.utcnow().isoformat()
    meta = json.dumps({
        "source_session_id": session_id,
        "similar_sessions":  similar_count,
        "learn_model":       LEARN_MODEL,
        "hook":              "session_end_learn",
    })
    content = (
        f"LEARNING NOTE — session: {session_title!r}\n"
        f"Generated: {ts}\n\n"
        f"{note}"
    )
    try:
        conn = sqlite3.connect(MNEM_DB, timeout=5.0)
        conn.execute(
            """INSERT OR IGNORE INTO working_memory
               (id, content, source, timestamp, session_id, importance,
                metadata_json, author_id, author_type, scope, memory_type)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                note_id, content, "session_end_learn", ts,
                session_id, 0.6, meta,
                "mrpink-learn", "agent", "global", "episodic",
            )
        )
        conn.commit()
        conn.close()
        log(f"wrote learning note {note_id[:16]} to working_memory")
        return True
    except Exception as e:
        log(f"Mnemosyne write error: {e}")
        return False


# ── Prompt builder ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
You are a learning-synthesis agent. Your job is to extract durable, reusable
insights by comparing a CURRENT session against similar PAST sessions.

Rules:
- Be concise (200-350 words total).
- Focus on DIFFERENCES and NEW PATTERNS — not on what stayed the same.
- If the current session repeats exactly the same approach as a past session,
  note it in one sentence and skip to genuine novelty.
- Identify: new tools used, new error types encountered, changed approaches,
  resolved blockers, and any patterns that appear across multiple sessions.
- Write in a declarative style useful for future context injection.
  Example: "Session introduced /search REST endpoint on mrpink A2A server ..."
- Do NOT repeat the session title or date in the body.
- Output format:
    WHAT'S NEW: <1-3 bullet points>
    APPROACH DELTA: <compared to past sessions, what changed>
    DURABLE PATTERNS: <anything stable that should persist to memory>
""")


def build_prompt(current: dict, similar: list[dict]) -> str:
    past_block = ""
    for i, s in enumerate(similar, 1):
        past_block += (
            f"\n--- PAST SESSION {i} (score={s['score']:.2f}) ---\n"
            f"Title: {s['title']}\n"
            f"{s['preview']}\n"
        )
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== CURRENT SESSION ===\n"
        f"Title: {current['title']}\n"
        f"Messages: {current['msg_count']}\n\n"
        f"{current['content'][:MAX_CONTENT]}\n\n"
        f"=== SIMILAR PAST SESSIONS ===\n"
        f"{past_block.strip() or '(none found)'}\n\n"
        f"Synthesize the learning note now:"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.monotonic()

    session_id = read_stdin_session_id()
    if not session_id:
        sys.exit(0)

    if already_processed(session_id):
        log(f"already processed {session_id[:20]}, skipping")
        sys.exit(0)

    sess = get_session_content(session_id)
    if not sess:
        sys.exit(0)   # too short or not found

    # Use title + first user message as search query (more specific than full content)
    query = sess["title"]
    similar = search_similar_sessions(query, exclude_session_id=session_id)
    if not similar:
        log(f"no similar sessions found above threshold {MIN_SCORE} — skipping note")
        mark_processed(session_id)
        sys.exit(0)

    log(f"found {len(similar)} similar sessions, calling {LEARN_MODEL}...")
    prompt  = build_prompt(sess, similar)
    note    = call_ollama(prompt)
    if not note or len(note) < 30:
        log("Ollama returned empty or too-short note, skipping write")
        sys.exit(0)

    ok = write_learning_note(session_id, sess["title"], note, len(similar))
    elapsed = time.monotonic() - t0
    if ok:
        mark_processed(session_id)
        log(f"done in {elapsed:.1f}s — note written for {sess['title']!r}")
    else:
        log(f"done in {elapsed:.1f}s — note NOT written (error above)")


if __name__ == "__main__":
    main()
