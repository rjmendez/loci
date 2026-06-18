"""
Ebbinghaus forgetting-curve-triggered memory consolidation.

Implements the FOREVER algorithm (ACL 2026): entries whose retention
probability falls below FORGET_THRESH are re-embedded and upserted into
Qdrant to refresh their representation, then their SQLite recall metadata
is updated so the next forgetting window resets.

Retention formula: R = exp(-t / S)
  t = days since last_recalled (or created_at if never recalled)
  S (stability) is now computed using an FSRS-inspired DSR model:
  - Initialized: S = (1 + recall_count)^(1/D) * (D/2), where D = difficulty
  - D (difficulty) initialized from importance: D = 11 - importance
  - Success update: S' = S * exp(w1 * (11-D) * (exp(w2*(1-R))-1) + 1)
  - Failure update: S' = S * 0.5
  Reference: FSRS (open-spaced-repetition/free-spaced-repetition-scheduler)
"""

import hashlib
import json
import math
import os
import sqlite3
import sys
import urllib.request
from datetime import datetime, timezone


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
FORGET_THRESH = float(os.environ.get("FORGET_THRESH", "0.3"))
MAX_PER_RUN = int(os.environ.get("MAX_PER_RUN", "50"))
QDRANT_COLLECTION = "mnemosyne"

# FSRS-inspired stability model parameters
FSRS_W1 = float(os.environ.get("FSRS_W1", "0.4"))   # success stability growth rate
FSRS_W2 = float(os.environ.get("FSRS_W2", "0.6"))   # retrievability factor in success
FSRS_DECAY_FACTOR = float(os.environ.get("FSRS_DECAY_FACTOR", "0.5"))  # failure penalty
FSRS_DIFF_INIT = float(os.environ.get("FSRS_DIFF_INIT", "5.0"))  # neutral difficulty
FSRS_DIFF_MAX = 10.0
FSRS_DIFF_MIN = 1.0


def _load_qdrant_api_key() -> str:
    settings_path = os.path.expanduser("~/.claude/settings.json")
    try:
        with open(settings_path) as fh:
            cfg = json.load(fh)
        return cfg["mcpServers"]["hermes_memory"]["env"]["QDRANT_API_KEY"]
    except (KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"[warn] could not read QDRANT_API_KEY from settings.json: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stable_id(s: str) -> int:
    """Deterministic integer point-id from content string."""
    return int(hashlib.sha256(s.encode()).hexdigest()[:15], 16)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def days_since(ts_str: str) -> float:
    """Return fractional days between ts_str (ISO-8601 or SQLite datetime) and now."""
    if not ts_str:
        return 0.0
    ts_str = ts_str.strip()
    # Normalise SQLite datetimes that lack timezone info.
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(ts_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - dt
            return delta.total_seconds() / 86400.0
        except ValueError:
            continue
    print(f"[warn] unrecognised timestamp format: {ts_str!r}")
    return 0.0


def _init_difficulty(importance: float) -> float:
    """Initialize difficulty D from importance. High importance → lower D (easier to retain)."""
    return max(FSRS_DIFF_MIN, min(FSRS_DIFF_MAX, 11.0 - float(importance or 5.0)))


def _stability_from_count(recall_count: int, difficulty: float) -> float:
    """FSRS-inspired stability: S = (1 + recall_count)^(1/difficulty) * base."""
    base = max(1.0, difficulty * 0.5)
    return base * ((1 + (recall_count or 0)) ** (1.0 / max(1.0, difficulty)))


def _update_stability_success(s: float, r: float, d: float) -> float:
    """FSRS success stability update: S' = S * exp(w1 * (11-D) * (exp(w2*(1-R))-1) + 1)"""
    exponent = FSRS_W1 * (11.0 - d) * (math.exp(FSRS_W2 * (1.0 - r)) - 1.0) + 1.0
    return s * math.exp(exponent)


def _update_stability_failure(s: float) -> float:
    """FSRS failure: stability halved (simplified from FSRS forgotten-card formula)."""
    return max(0.5, s * FSRS_DECAY_FACTOR)


def _update_difficulty(d: float, grade: float) -> float:
    """Update difficulty: 4 is target grade. Above 4 → easier; below → harder."""
    delta = 0.15 * (4.0 - grade)
    return max(FSRS_DIFF_MIN, min(FSRS_DIFF_MAX, d + delta))


def _grade_from_retention(r: float) -> float:
    """Map current retention to a 1–5 grade for FSRS difficulty update."""
    if r >= 0.90: return 5.0
    if r >= 0.75: return 4.0
    if r >= 0.55: return 3.0
    if r >= 0.35: return 2.0
    return 1.0


def retention(recall_count: int, last_recalled: str, created_at: str,
              fsrs_stability: float | None = None) -> float:
    """R = exp(-t / S) where S comes from FSRS stability if available, else computed."""
    t = days_since(last_recalled) if last_recalled else days_since(created_at)
    if fsrs_stability and fsrs_stability > 0:
        s = fsrs_stability
    else:
        s = _stability_from_count(recall_count or 0, FSRS_DIFF_INIT)
    return math.exp(-t / s)


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------

def http_post(url: str, payload: dict, headers: dict | None = None) -> dict:
    body = json.dumps(payload).encode()
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=body, headers=req_headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def http_put(url: str, payload: dict, headers: dict | None = None) -> dict:
    body = json.dumps(payload).encode()
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=body, headers=req_headers, method="PUT")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed(content: str) -> list[float]:
    url = f"{OLLAMA_URL}/v1/embeddings"
    resp = http_post(url, {"model": EMBED_MODEL, "input": content})
    # OpenAI-compatible response shape
    return resp["data"][0]["embedding"]


# ---------------------------------------------------------------------------
# Qdrant upsert
# ---------------------------------------------------------------------------

def qdrant_upsert(point_id: int, vector: list[float], payload: dict, api_key: str) -> None:
    url = f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points"
    headers = {}
    if api_key:
        headers["api-key"] = api_key
    points_payload = {
        "points": [
            {
                "id": point_id,
                "vector": {"dense": vector},
                "payload": payload,
            }
        ]
    }
    http_put(url, points_payload, headers=headers)


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

TABLES = ("working_memory", "episodic_memory")

FETCH_SQL = """
    SELECT id, content, recall_count, last_recalled, created_at, importance
    FROM {table}
    WHERE length(content) > 20
"""

UPDATE_SQL = """
    UPDATE {table}
    SET last_recalled = ?, recall_count = ?
    WHERE id = ?
"""


def fetch_candidates(conn: sqlite3.Connection) -> list[tuple]:
    """Return rows from both tables as (table, id, content, recall_count, last_recalled, created_at, importance)."""
    candidates = []
    for table in TABLES:
        try:
            cur = conn.execute(FETCH_SQL.format(table=table))
            for row in cur.fetchall():
                candidates.append((table,) + row)
        except sqlite3.OperationalError as exc:
            print(f"[warn] could not query {table}: {exc}")
    return candidates


def update_recall(conn: sqlite3.Connection, table: str, row_id: int, new_count: int, ts: str) -> None:
    conn.execute(UPDATE_SQL.format(table=table), (ts, new_count, row_id))
    conn.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = _load_qdrant_api_key()

    if not os.path.exists(DB_PATH):
        print(f"[error] SQLite DB not found at {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)

    print(f"[info] scanning working_memory + episodic_memory in {DB_PATH}")
    all_rows = fetch_candidates(conn)
    print(f"[info] {len(all_rows)} rows with content > 20 chars")

    decayed = []
    for (table, row_id, content, recall_count, last_recalled, created_at, importance) in all_rows:
        r = retention(recall_count or 0, last_recalled, created_at)
        if r < FORGET_THRESH:
            decayed.append((r, table, row_id, content, recall_count or 0, last_recalled, created_at, importance))

    # Lowest retention first (most-forgotten first)
    decayed.sort(key=lambda x: x[0])
    batch = decayed[:MAX_PER_RUN]

    print(f"[info] {len(decayed)} entries below forget threshold {FORGET_THRESH}, processing {len(batch)}")

    processed = 0
    errors = 0

    for (r, table, row_id, content, recall_count, last_recalled, created_at, importance) in batch:
        try:
            print(f"  [{table}:{row_id}] R={r:.4f}  refreshing …", end=" ", flush=True)

            # FSRS: read existing stability from Qdrant payload if available
            # (first run: None → computed from recall_count + importance)
            existing_stability = None  # will be overridden once payloads carry fsrs_stability
            d_init = _init_difficulty(importance or 5.0)
            s_current = existing_stability or _stability_from_count(recall_count or 0, d_init)

            if r < FORGET_THRESH:
                # Forgotten: penalise stability, increase difficulty
                new_stability = _update_stability_failure(s_current)
                grade = _grade_from_retention(r)
                new_difficulty = _update_difficulty(d_init, grade)
            else:
                # Scheduled refresh (not forgotten): stability update on successful recall
                grade = _grade_from_retention(r)
                new_stability = _update_stability_success(s_current, r, d_init)
                new_difficulty = _update_difficulty(d_init, grade)

            # (a) Embed
            vec = embed(content)

            # (b) Upsert to Qdrant
            point_id = stable_id(content)
            ts = now_iso()
            qdrant_upsert(
                point_id=point_id,
                vector=vec,
                payload={
                    "content": content,
                    "importance": importance,
                    "last_refreshed": ts,
                    "decay_score": r,
                    "fsrs_stability": round(new_stability, 4),
                    "fsrs_difficulty": round(new_difficulty, 4),
                    "mnemosyne_id": str(row_id),
                    "mnemosyne_table": table,
                },
                api_key=api_key,
            )

            # (c) Update SQLite
            update_recall(conn, table, row_id, recall_count + 1, ts)

            processed += 1
            print(f"ok  S={new_stability:.3f} D={new_difficulty:.3f}")

        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"ERROR: {exc}")

    conn.close()
    print(f"[done] processed={processed} errors={errors}")


if __name__ == "__main__":
    main()
