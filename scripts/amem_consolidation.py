"""
A-MEM Consolidation — Associative Memory (Zettelkasten-style) over Mnemosyne SQLite.

Phase 1: Cross-link discovery via cosine similarity on Ollama embeddings.
Phase 2: Conflict detection for near-duplicate but divergent entries.
"""

import importlib.util
import json
import math
import os
import sqlite3
import urllib.request
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH = os.environ.get(
    "MNEMOSYNE_DB",
    os.path.expanduser("~/.hermes/mnemosyne/data/mnemosyne.db"),
)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
AMEM_LINK_THRESHOLD = float(os.environ.get("AMEM_LINK_THRESHOLD", "0.88"))
AMEM_CONFLICT_THRESHOLD = float(os.environ.get("AMEM_CONFLICT_THRESHOLD", "0.96"))
AMEM_UPDATE_THRESHOLD = float(os.environ.get("AMEM_UPDATE_THRESHOLD", "0.92"))  # near-copy with temporal ordering → updated_by
MAX_PER_RUN = int(os.environ.get("MAX_PER_RUN", "100"))

# Keyword pairs whose co-presence signals a potential conflict.
CONFLICT_KEYWORD_PAIRS = [
    ("true", "false"),
    ("enabled", "disabled"),
    ("success", "failure"),
]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed(text: str) -> list[float]:
    """Return the embedding vector for *text* via Ollama."""
    payload = json.dumps({"model": EMBED_MODEL, "input": text}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/v1/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        body = json.loads(resp.read())
    # Ollama returns {"data": [{"embedding": [...]}]}
    return body["data"][0]["embedding"]


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Conflict heuristic
# ---------------------------------------------------------------------------

def has_contradiction(content_a: str, content_b: str) -> bool:
    """Return True when the two contents contain opposing keywords."""
    combined = (content_a + " " + content_b).lower()
    tokens = set(combined.split())
    for kw_pos, kw_neg in CONFLICT_KEYWORD_PAIRS:
        if kw_pos in tokens and kw_neg in tokens:
            return True
    return False


# Phrases that signal one entry supersedes or follows from another.
_UPDATE_SIGNALS = [
    "updated", "revised", "corrected", "supersedes", "replaces", "now", "actually",
    "new finding", "correction:", "update:", "revision:",
]

def _has_update_signal(content: str) -> bool:
    lower = content.lower()
    return any(sig in lower for sig in _UPDATE_SIGNALS)


def _has_causal_signal(content_b: str, id_a: str) -> bool:
    """Return True if content_b references entry A by id or common causal phrasing."""
    causal_phrases = ["because", "caused by", "due to", "as a result", "triggered by",
                      "led to", "resulting from", "following", "after"]
    lower = content_b.lower()
    return id_a[:8] in content_b or any(p in lower for p in causal_phrases)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_quorum_gate():
    """Import QuorumGate from the scripts directory (sibling of this file)."""
    try:
        _qg_path = os.path.join(os.path.dirname(__file__), "quorum_gate.py")
        spec = importlib.util.spec_from_file_location("quorum_gate", _qg_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.QuorumGate
    except Exception:
        return None


# One new edge or conflict found → deposit 1.0 to the amem_consolidation topic.
# Set QUORUM_AMEM_THRESHOLD to require N accumulated signals before running the
# expensive embedding pass again. 0 (default) disables the gate.
QUORUM_AMEM_THRESHOLD = float(os.environ.get("QUORUM_AMEM_THRESHOLD", "0"))


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()

    # Quorum gate: skip if not enough signal has accumulated since last run.
    # Deposit happens at the end when we actually did work.
    if QUORUM_AMEM_THRESHOLD > 0:
        QuorumGate = _load_quorum_gate()
        if QuorumGate is not None:
            gate = QuorumGate()
            if not gate.check_quorum("amem_consolidation", QUORUM_AMEM_THRESHOLD):
                eff = gate.effective("amem_consolidation")
                print(f"[amem] quorum not reached ({eff:.2f}/{QUORUM_AMEM_THRESHOLD}) — skipping")
                return
        else:
            QuorumGate = None
            gate = None
    else:
        QuorumGate = None
        gate = None

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # ------------------------------------------------------------------
    # Load recent entries
    # ------------------------------------------------------------------
    cur.execute(
        "SELECT id, content, created_at FROM working_memory ORDER BY created_at DESC LIMIT ?",
        (MAX_PER_RUN,),
    )
    rows = cur.fetchall()

    if len(rows) < 2:
        print("[amem] not enough entries to compare — skipping")
        conn.close()
        return

    # ------------------------------------------------------------------
    # Phase 1 — Embed all entries
    # ------------------------------------------------------------------
    entries: list[tuple[str, str, list[float], str]] = []
    for row in rows:
        vec = embed(row["content"])
        entries.append((row["id"], row["content"], vec, row["created_at"] or ""))

    # ------------------------------------------------------------------
    # Phase 1 — Pairwise cosine → cross-links
    # ------------------------------------------------------------------
    new_links = 0
    flagged_conflicts = 0

    n = len(entries)
    causal_links = 0
    update_links = 0

    for i in range(n):
        id_a, content_a, vec_a, ts_a = entries[i]
        for j in range(i + 1, n):
            id_b, content_b, vec_b, ts_b = entries[j]

            sim = cosine(vec_a, vec_b)

            if sim > AMEM_LINK_THRESHOLD:
                cur.execute(
                    """
                    INSERT OR IGNORE INTO graph_edges
                        (source, target, edge_type, weight, timestamp, created_at)
                    VALUES (?, ?, 'semantic_link', ?, ?, ?)
                    """,
                    (id_a, id_b, sim, now, now),
                )
                if cur.rowcount:
                    new_links += 1

                # ------------------------------------------------------
                # Phase 2 — Conflict detection (subset of linked pairs)
                # ------------------------------------------------------
                if sim > AMEM_CONFLICT_THRESHOLD and has_contradiction(content_a, content_b):
                    cur.execute(
                        """
                        INSERT OR IGNORE INTO conflicts
                            (fact_a_id, fact_b_id, conflict_type, created_at)
                        VALUES (?, ?, 'near_duplicate_divergent', ?)
                        """,
                        (id_a, id_b, now),
                    )
                    if cur.rowcount:
                        flagged_conflicts += 1
                    # Also add contradicts edge for causal traversal
                    cur.execute(
                        """
                        INSERT OR IGNORE INTO graph_edges
                            (source, target, edge_type, weight, timestamp, created_at)
                        VALUES (?, ?, 'contradicts', ?, ?, ?)
                        """,
                        (id_a, id_b, sim, now, now),
                    )

                # ------------------------------------------------------
                # Phase 3 — Causal edge detection (ActMem/GAM pattern)
                # updated_by: high similarity + B is newer + B has update signal
                # caused_by: B explicitly references A or uses causal phrasing
                # ------------------------------------------------------
                if sim >= AMEM_UPDATE_THRESHOLD:
                    newer, older = (id_b, id_a) if ts_b > ts_a else (id_a, id_b)
                    newer_content = content_b if ts_b > ts_a else content_a
                    if _has_update_signal(newer_content):
                        cur.execute(
                            """
                            INSERT OR IGNORE INTO graph_edges
                                (source, target, edge_type, weight, timestamp, created_at)
                            VALUES (?, ?, 'updated_by', ?, ?, ?)
                            """,
                            (older, newer, sim, now, now),
                        )
                        if cur.rowcount:
                            update_links += 1

                if _has_causal_signal(content_b, id_a):
                    cur.execute(
                        """
                        INSERT OR IGNORE INTO graph_edges
                            (source, target, edge_type, weight, timestamp, created_at)
                        VALUES (?, ?, 'caused_by', ?, ?, ?)
                        """,
                        (id_b, id_a, sim, now, now),
                    )
                    if cur.rowcount:
                        causal_links += 1
                elif _has_causal_signal(content_a, id_b):
                    cur.execute(
                        """
                        INSERT OR IGNORE INTO graph_edges
                            (source, target, edge_type, weight, timestamp, created_at)
                        VALUES (?, ?, 'caused_by', ?, ?, ?)
                        """,
                        (id_a, id_b, sim, now, now),
                    )
                    if cur.rowcount:
                        causal_links += 1

    conn.commit()
    conn.close()

    print(f"[amem] {new_links} new cross-links created")
    print(f"[amem] {flagged_conflicts} conflicts flagged")
    print(f"[amem] {update_links} updated_by edges | {causal_links} caused_by edges")

    # Reset quorum accumulator now that we ran (so next quorum cycle starts fresh).
    if gate is not None:
        gate.reset("amem_consolidation")
        print("[amem] quorum accumulator reset")


if __name__ == "__main__":
    main()
