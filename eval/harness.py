"""
Longitudinal eval harness for dama-gotchi Claude Code agent.

For each task in tasks.py:
  1. Call pre_llm_grounding.py with the task prompt.
  2. Parse the returned context string.
  3. Score = fraction of expected_keywords present in context.
  4. Upsert result to Qdrant eval_scores collection.
Print per-task scores and final mean.
"""

import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from tasks import TASKS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
QDRANT_URL = os.environ.get("QDRANT_URL")
OLLAMA_URL = os.environ.get("OLLAMA_URL")
GROUNDING_SCRIPT = os.environ.get(
    "GROUNDING_SCRIPT",
    str(Path(__file__).resolve().parent.parent / "scripts/hooks/pre_llm_grounding.py"),
)
EVAL_COLLECTION = "eval_scores"
EMBED_MODEL = "nomic-embed-text"
# Set HARNESS_DRY_RUN=1 to score locally without Qdrant/Ollama (CI-safe).
DRY_RUN = os.environ.get("HARNESS_DRY_RUN", "") not in ("", "0", "false")


def _read_qdrant_api_key() -> str:
    settings_path = Path.home() / ".claude" / "settings.json"
    try:
        data = json.loads(settings_path.read_text())
        # Key may live at top level or nested under env/apiKeys
        for candidate in [
            data.get("qdrantApiKey"),
            data.get("env", {}).get("QDRANT_API_KEY"),
            data.get("apiKeys", {}).get("qdrant"),
        ]:
            if candidate:
                return candidate
    except Exception:
        pass
    return os.environ.get("QDRANT_API_KEY", "")


QDRANT_API_KEY = _read_qdrant_api_key()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _http(method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def embed(text: str) -> list[float]:
    url = f"{OLLAMA_URL}/api/embeddings"
    result = _http("POST", url, {"model": EMBED_MODEL, "prompt": text})
    return result["embedding"]


def ensure_collection(vector_size: int = 768) -> None:
    url = f"{QDRANT_URL}/collections/{EVAL_COLLECTION}"
    try:
        _http("GET", url)
        return  # already exists
    except Exception:
        pass
    _http(
        "PUT",
        url,
        {
            "vectors": {
                "dense": {
                    "size": vector_size,
                    "distance": "Cosine",
                }
            }
        },
    )


def stable_point_id(task_id: str, run_date: str) -> int:
    raw = (task_id + run_date).encode()
    return int(hashlib.sha256(raw).hexdigest()[:15], 16)


def upsert_score(
    task_id: str,
    task_name: str,
    category: str,
    score: float,
    run_date: str,
    context_preview: str,
    vector: list[float],
) -> None:
    point_id = stable_point_id(task_id, run_date)
    url = f"{QDRANT_URL}/collections/{EVAL_COLLECTION}/points"
    _http(
        "PUT",
        url,
        {
            "points": [
                {
                    "id": point_id,
                    "vector": {"dense": vector},
                    "payload": {
                        "task_id": task_id,
                        "task_name": task_name,
                        "score": score,
                        "run_date": run_date,
                        "category": category,
                        "context_preview": context_preview[:200],
                    },
                }
            ]
        },
    )


# ---------------------------------------------------------------------------
# Grounding call
# ---------------------------------------------------------------------------

def call_grounding(prompt: str) -> str:
    """
    Invoke pre_llm_grounding.py, sending the prompt as a JSON payload on stdin.
    Returns the context string, or empty string on failure.
    """
    payload = json.dumps(
        {"hook_event_name": "pre_llm_call", "extra": {"user_message": prompt}}
    ).encode()

    python_exe = os.environ.get(
        "HERMES_PY",
        str(Path.home() / ".hermes/hermes-agent/venv/bin/python3"),
    )

    try:
        result = subprocess.run(
            [python_exe, GROUNDING_SCRIPT],
            input=payload,
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            return ""
        parsed = json.loads(result.stdout)
        return parsed.get("context", "")
    except Exception as exc:
        print(f"  [warn] grounding call failed: {exc}", file=sys.stderr)
        return ""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_context(context: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 0.0
    hits = sum(1 for kw in expected_keywords if kw.lower() in context.lower())
    return hits / len(expected_keywords)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"[eval] run_date={run_date}  tasks={len(TASKS)}  dry_run={DRY_RUN}")

    if not DRY_RUN:
        ensure_collection()

    scores: list[float] = []
    for task in TASKS:
        task_id: str = task["id"]
        task_name: str = task["name"]
        category: str = task["category"]
        prompt: str = task["prompt"]
        expected_keywords: list[str] = task["expected_keywords"]

        context = call_grounding(prompt)
        score = score_context(context, expected_keywords)
        scores.append(score)

        if not DRY_RUN:
            vec = embed(prompt)
            upsert_score(
                task_id=task_id,
                task_name=task_name,
                category=category,
                score=score,
                run_date=run_date,
                context_preview=context,
                vector=vec,
            )

        hits = sum(
            1 for kw in expected_keywords if kw.lower() in context.lower()
        )
        print(
            f"  [{category}] {task_id}: score={score:.3f} "
            f"({hits}/{len(expected_keywords)} keywords)"
        )

    mean_score = sum(scores) / len(scores) if scores else 0.0
    print(f"[eval] mean_score={mean_score:.3f} ({len(scores)} tasks)")


if __name__ == "__main__":
    run()
