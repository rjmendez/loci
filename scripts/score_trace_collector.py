#!/usr/bin/env python3
"""
SCoRe data collection pipeline (arxiv 2409.12917).

Collects corrective trace data from guard logs and working_memory to build
a dataset for future fine-tuning of local Ollama models.

Sources:
  NEGATIVE      - guard_bash_failures.log (count >= 2)
  POSITIVE      - guard_bash_successes.log
  AGENTHR_POS   - working_memory WHERE source='agentHER'

Correction pairs: negative session_id matched to later positive from same session.

Output: OUTPUT_DIR/{negatives,positives,corrections}.jsonl + manifest.json
Qdrant: score_traces collection, named vector {"dense": vec}, 768-dim Cosine.
"""

import json
import os
import sqlite3
import hashlib
import datetime
import urllib.request


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MNEMOSYNE_DB = os.environ.get(
    "MNEMOSYNE_DB",
    os.path.expanduser("~/.hermes/mnemosyne/data/mnemosyne.db"),
)
STATE_DIR = os.environ.get(
    "STATE_DIR",
    os.path.expanduser("~/.claude/hook-state"),
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.expanduser("~/.hermes/mnemosyne/data/score_traces"),
)
QDRANT_URL = os.environ.get("QDRANT_URL")
OLLAMA_URL = os.environ.get("OLLAMA_URL")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

SCORE_COLLECTION = "score_traces"
VECTOR_DIM = 768


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stable_id(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:15], 16)


def read_qdrant_key() -> str:
    settings_path = os.path.expanduser("~/.claude/settings.json")
    try:
        with open(settings_path) as f:
            data = json.load(f)
        # Common key locations in settings.json
        key = (
            data.get("qdrantApiKey")
            or data.get("QDRANT_API_KEY")
            or data.get("env", {}).get("QDRANT_API_KEY")
            or data.get("env", {}).get("QDRANT_KEY")
            or ""
        )
        return key
    except Exception:
        return os.environ.get("QDRANT_API_KEY", "")


def http_post(url: str, payload: dict, headers: dict | None = None) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        return {"error": str(exc)}


def http_put(url: str, payload: dict, headers: dict | None = None) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method="PUT")
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        return {"error": str(exc)}


def embed_text(text: str) -> list[float] | None:
    payload = {"model": EMBED_MODEL, "prompt": text}
    result = http_post(f"{OLLAMA_URL}/api/embeddings", payload)
    return result.get("embedding")


def ensure_score_traces_collection(api_key: str) -> None:
    headers = {"api-key": api_key} if api_key else {}
    # Check if collection exists
    check_url = f"{QDRANT_URL}/collections/{SCORE_COLLECTION}"
    req = urllib.request.Request(check_url)
    if api_key:
        req.add_header("api-key", api_key)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            if data.get("result"):
                return  # already exists
    except urllib.error.HTTPError as e:
        if e.code != 404:
            return  # unexpected error, skip creation

    # Create collection with named vector
    create_payload = {
        "vectors": {
            "dense": {
                "size": VECTOR_DIM,
                "distance": "Cosine",
            }
        }
    }
    http_put(
        f"{QDRANT_URL}/collections/{SCORE_COLLECTION}",
        create_payload,
        headers=headers,
    )


def upsert_to_qdrant(point_id: int, vector: list[float], payload: dict, api_key: str) -> None:
    headers = {"api-key": api_key} if api_key else {}
    body = {
        "points": [
            {
                "id": point_id,
                "vector": {"dense": vector},
                "payload": payload,
            }
        ]
    }
    http_put(
        f"{QDRANT_URL}/collections/{SCORE_COLLECTION}/points",
        body,
        headers=headers,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_negatives() -> list[dict]:
    """Load guard_bash_failures.log, filter count >= 2."""
    path = os.path.join(STATE_DIR, "guard_bash_failures.log")
    results = []
    if not os.path.exists(path):
        return results
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            count = rec.get("count", 1)
            if count < 2:
                continue
            results.append({
                "trace_type": "negative",
                "content": rec.get("canonical_command", rec.get("command", "")),
                "count": count,
                "session_id": rec.get("session_id", ""),
            })
    return results


def load_positives() -> list[dict]:
    """Load guard_bash_successes.log if it exists."""
    path = os.path.join(STATE_DIR, "guard_bash_successes.log")
    results = []
    if not os.path.exists(path):
        return results
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            results.append({
                "trace_type": "positive",
                "content": rec.get("command", ""),
                "session_id": rec.get("session_id", ""),
            })
    return results


def load_agenthr_positives() -> list[dict]:
    """Query mnemosyne.db working_memory WHERE source='agentHER'."""
    results = []
    if not os.path.exists(MNEMOSYNE_DB):
        return results
    try:
        conn = sqlite3.connect(MNEMOSYNE_DB)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT content, session_id FROM working_memory WHERE source = 'agentHER'"
        )
        for row in cur.fetchall():
            results.append({
                "trace_type": "positive_relabeled",
                "content": row["content"] or "",
                "session_id": row["session_id"] or "",
            })
        conn.close()
    except sqlite3.Error:
        pass
    return results


# ---------------------------------------------------------------------------
# Correction pair matching
# ---------------------------------------------------------------------------

def build_correction_pairs(
    negatives: list[dict], positives: list[dict]
) -> list[dict]:
    """
    For each negative session_id, find a later positive from the same session.
    'Later' is inferred by list order (log order = temporal order).
    """
    # Build index: session_id -> list of positive records (in order)
    pos_by_session: dict[str, list[dict]] = {}
    for pos in positives:
        sid = pos.get("session_id", "")
        pos_by_session.setdefault(sid, []).append(pos)

    corrections = []
    # Track which positives have been consumed to avoid duplicates
    pos_consumed: dict[str, int] = {}

    for neg in negatives:
        sid = neg.get("session_id", "")
        if not sid:
            continue
        candidates = pos_by_session.get(sid, [])
        used = pos_consumed.get(sid, 0)
        if used >= len(candidates):
            continue
        matched_pos = candidates[used]
        pos_consumed[sid] = used + 1
        corrections.append({
            "trace_type": "correction",
            "failed_content": neg["content"],
            "corrected_content": matched_pos["content"],
            "session_id": sid,
        })

    return corrections


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def write_manifest(n_neg: int, n_pos: int, n_corr: int) -> None:
    manifest = {
        "n_neg": n_neg,
        "n_pos": n_pos,
        "n_corr": n_corr,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "ready_for_sft": n_corr >= 10,
    }
    path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    api_key = read_qdrant_key()

    # Load data
    negatives = load_negatives()
    positives_raw = load_positives()
    agenthr = load_agenthr_positives()
    all_positives = positives_raw + agenthr

    # Build correction pairs using raw positives (agentHER are relabeled, not direct corrections)
    corrections = build_correction_pairs(negatives, positives_raw)

    # Write JSONL files
    write_jsonl(os.path.join(OUTPUT_DIR, "negatives.jsonl"), negatives)
    write_jsonl(os.path.join(OUTPUT_DIR, "positives.jsonl"), all_positives)
    write_jsonl(os.path.join(OUTPUT_DIR, "corrections.jsonl"), corrections)

    # Write manifest
    write_manifest(len(negatives), len(all_positives), len(corrections))

    # Embed corrections and upsert to Qdrant
    if corrections:
        ensure_score_traces_collection(api_key)
        for corr in corrections:
            vec = embed_text(corr["failed_content"])
            if vec is None:
                continue
            point_id = stable_id(corr["session_id"] + corr["failed_content"])
            payload = {
                "trace_type": corr["trace_type"],
                "failed_content": corr["failed_content"],
                "corrected_content": corr["corrected_content"],
                "session_id": corr["session_id"],
            }
            upsert_to_qdrant(point_id, vec, payload, api_key)

    sft_ready = len(corrections) >= 10
    print(
        f"[score] negatives={len(negatives)}, positives={len(all_positives)}, "
        f"corrections={len(corrections)}, sft_ready={sft_ready}"
    )


if __name__ == "__main__":
    main()
