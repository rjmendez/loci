#!/usr/bin/env python3
"""
AgentHER: Hindsight Experience Replay for LLM Agents (arxiv 2603.21357)

Reads failed session traces from Mnemosyne SQLite, relabels them as
positive examples via Ollama, stores synthetic positives back into both
the Mnemosyne DB and the Qdrant mnemosyne collection for future retrieval.
"""
import sqlite3
import json
import hashlib
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MNEMOSYNE_DB = os.path.expanduser(
    os.environ.get("MNEMOSYNE_DB", "~/.hermes/mnemosyne/data/mnemosyne.db")
)
QDRANT_URL   = os.environ.get("QDRANT_URL")
OLLAMA_URL   = os.environ.get("OLLAMA_URL")
EMBED_MODEL  = os.environ.get("EMBED_MODEL", "nomic-embed-text")
GEN_MODEL    = os.environ.get("AGENTHER_GEN_MODEL", "llama3.2:latest")
MAX_PER_RUN  = int(os.environ.get("MAX_PER_RUN", "20"))
COLLECTION   = "mnemosyne"

# ---------------------------------------------------------------------------
# Helpers: Qdrant API key
# ---------------------------------------------------------------------------

def _load_env_file(path):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def get_qdrant_key():
    key = os.environ.get("QDRANT_API_KEY", "")
    if key:
        return key
    # Try loading from known env files
    for env_path in [
        os.path.expanduser(os.environ.get("HERMES_ENV_FILE", "~/.hermes/.env")),
        os.path.expanduser("~/.claude/settings.json"),
    ]:
        if env_path.endswith(".json") and os.path.exists(env_path):
            try:
                with open(env_path) as f:
                    data = json.load(f)
                # Flatten nested env dicts
                def _find(obj, target):
                    if isinstance(obj, dict):
                        if target in obj:
                            return obj[target]
                        for v in obj.values():
                            result = _find(v, target)
                            if result:
                                return result
                    return None
                found = _find(data, "QDRANT_API_KEY")
                if found:
                    return found
            except Exception:
                pass
        else:
            _load_env_file(env_path)
            key = os.environ.get("QDRANT_API_KEY", "")
            if key:
                return key
    return ""


QDRANT_KEY = get_qdrant_key()

# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------

def _http(method, url, data=None, extra_headers=None, timeout=30):
    headers = {"Content-Type": "application/json"}
    if QDRANT_KEY and "30633" in url:
        headers["api-key"] = QDRANT_KEY
    if extra_headers:
        headers.update(extra_headers)
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        print(f"[agentHER] HTTP {e.code} {method} {url}: {e.read()[:200]}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[agentHER] {method} {url} error: {e}", file=sys.stderr)
        return None

# ---------------------------------------------------------------------------
# Stable deterministic ID
# ---------------------------------------------------------------------------

def stable_id(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:15], 16)

# ---------------------------------------------------------------------------
# Embedding via Ollama
# ---------------------------------------------------------------------------

def embed(text: str):
    payload = {"model": EMBED_MODEL, "prompt": text}
    resp = _http("POST", f"{OLLAMA_URL}/api/embeddings", payload, timeout=30)
    if resp and "embedding" in resp:
        return resp["embedding"]
    # Fallback: /v1/embeddings OpenAI-compat endpoint
    payload2 = {"model": EMBED_MODEL, "input": text}
    resp2 = _http("POST", f"{OLLAMA_URL}/v1/embeddings", payload2, timeout=30)
    if resp2:
        try:
            return resp2["data"][0]["embedding"]
        except (KeyError, IndexError):
            pass
    return None

# ---------------------------------------------------------------------------
# Ollama generation
# ---------------------------------------------------------------------------

RELABEL_PROMPT_TEMPLATE = (
    "Given this failed agent trace, what task does it actually accomplish "
    "or what useful info does it reveal? Reply in one sentence starting with "
    "'This trace shows how to' or 'This trace reveals that'. "
    "Trace: {content}"
)


def generate_relabel(content: str) -> str | None:
    prompt = RELABEL_PROMPT_TEMPLATE.format(content=content[:500])
    payload = {"model": GEN_MODEL, "prompt": prompt, "stream": False}
    resp = _http("POST", f"{OLLAMA_URL}/api/generate", payload, timeout=60)
    if resp is None:
        return None
    text = resp.get("response", "").strip()
    return text if text else None

# ---------------------------------------------------------------------------
# Qdrant upsert
# ---------------------------------------------------------------------------

def qdrant_upsert(point_id: int, vector, payload: dict) -> bool:
    body = {
        "points": [
            {
                "id": point_id,
                "vector": {"dense": vector},
                "payload": payload,
            }
        ]
    }
    resp = _http("PUT", f"{QDRANT_URL}/collections/{COLLECTION}/points", body, timeout=30)
    if resp and resp.get("status") == "ok":
        return True
    return False

# ---------------------------------------------------------------------------
# Mnemosyne DB helpers
# ---------------------------------------------------------------------------

FAILURE_QUERY = """
SELECT id, content, importance, created_at, session_id
FROM working_memory
WHERE (
    content LIKE '%FAILURE%'
    OR content LIKE '%FAILED%'
    OR content LIKE '%ERROR%'
    OR content LIKE '%REPEATED%'
)
AND importance >= 5
AND created_at > datetime('now', '-7 days')
ORDER BY created_at DESC
LIMIT ?
"""

INSERT_SYNTHETIC = """
INSERT INTO working_memory
    (content, source, importance, created_at, session_id, author_id, author_type)
VALUES
    (?, 'agentHER', 6, ?, 'agentHER_relabel', 'agent', 'agent')
"""


def load_failures(conn: sqlite3.Connection):
    cur = conn.execute(FAILURE_QUERY, (MAX_PER_RUN,))
    return cur.fetchall()


def store_synthetic(conn: sqlite3.Connection, synthetic: str):
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    conn.execute(INSERT_SYNTHETIC, (synthetic, now_iso))
    conn.commit()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(MNEMOSYNE_DB):
        print(f"[agentHER] DB not found: {MNEMOSYNE_DB}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(MNEMOSYNE_DB)
    rows = load_failures(conn)
    n_processed = 0
    n_relabeled = 0
    n_stored = 0

    for row in rows:
        row_id, content, importance, created_at, session_id = row
        n_processed += 1

        # Step a: generate relabel via Ollama
        try:
            relabeled = generate_relabel(content)
        except Exception as e:
            print(f"[agentHER] gen error row {row_id}: {e}", file=sys.stderr)
            continue

        if relabeled is None:
            print(f"[agentHER] no response for row {row_id}", file=sys.stderr)
            continue

        # Step b: validate response
        if not (relabeled.startswith("This trace") and len(relabeled) > 20):
            continue

        n_relabeled += 1

        synthetic = f"AGENTHER_POSITIVE: {relabeled}\nOriginal: {content[:200]}"

        # Step c: embed
        vec = embed(synthetic)
        if vec is None:
            print(f"[agentHER] embed failed for row {row_id}", file=sys.stderr)
            continue

        # Step d: upsert to Qdrant
        point_id = stable_id(synthetic)
        qdrant_payload = {
            "content": synthetic,
            "importance": 6,
            "source": "agentHER",
            "original_id": str(row_id),
        }
        upserted = qdrant_upsert(point_id, vec, qdrant_payload)
        if not upserted:
            print(f"[agentHER] qdrant upsert failed for row {row_id}", file=sys.stderr)

        # Step e: store in Mnemosyne DB
        try:
            store_synthetic(conn, synthetic)
            n_stored += 1
        except Exception as e:
            print(f"[agentHER] db insert error row {row_id}: {e}", file=sys.stderr)

    conn.close()
    print(f"[agentHER] processed {n_processed}, relabeled {n_relabeled}, stored {n_stored}")


if __name__ == "__main__":
    main()
