#!/usr/bin/env python3
"""
Orchestrates SCoRe + AgentHER data collection into a unified training JSONL.

Usage:
    python3 mlops/finetune/collect.py \
        --out mlops/finetune/data/ \
        [--ollama http://localhost:11434] \
        [--db ~/.hermes/mnemosyne/data/mnemosyne.db] \
        [--hook-state ~/.claude/hook-state]
"""

import argparse
import datetime
import hashlib
import json
import os
import sqlite3
import sys

# ---------------------------------------------------------------------------
# Resolve project root so we can import from scripts/
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.score_trace_collector import (
    load_negatives,
    load_positives,
    load_agenthr_positives,
    build_correction_pairs,
    STATE_DIR as _DEFAULT_STATE_DIR,
    MNEMOSYNE_DB as _DEFAULT_DB,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# AgentHER reader — pulls already-relabeled rows from Mnemosyne
# ---------------------------------------------------------------------------

def load_agentHER_from_db(db_path: str) -> list[dict]:
    results = []
    if not os.path.exists(db_path):
        return results
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT content, session_id FROM working_memory WHERE source = 'agentHER'"
        )
        for row in cur.fetchall():
            content = row["content"] or ""
            session_id = row["session_id"] or ""
            results.append({
                "type": "agentHER",
                "content": content,
                "source": "mnemosyne",
                "session_id": session_id,
            })
        conn.close()
    except sqlite3.Error as exc:
        print(f"[collect] sqlite error reading agentHER rows: {exc}", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# Unified record builder
# ---------------------------------------------------------------------------

def build_unified_records(
    negatives: list[dict],
    positives: list[dict],
    corrections: list[dict],
    agentHER: list[dict],
    collected_at: str,
) -> list[dict]:
    records = []

    for rec in negatives:
        content = rec.get("content", "")
        records.append({
            "id": _sha256(content),
            "type": "negative",
            "content": content,
            "source": "guard_log",
            "session_id": rec.get("session_id", ""),
            "collected_at": collected_at,
        })

    for rec in positives:
        content = rec.get("content", "")
        records.append({
            "id": _sha256(content),
            "type": "positive",
            "content": content,
            "source": "guard_log",
            "session_id": rec.get("session_id", ""),
            "collected_at": collected_at,
        })

    for rec in corrections:
        # Corrections carry both sides; content is the failed side for dedup key.
        failed = rec.get("failed_content", "")
        corrected = rec.get("corrected_content", "")
        content = json.dumps({"failed": failed, "corrected": corrected})
        records.append({
            "id": _sha256(content),
            "type": "correction",
            "content": content,
            "source": "guard_log",
            "session_id": rec.get("session_id", ""),
            "collected_at": collected_at,
        })

    for rec in agentHER:
        content = rec.get("content", "")
        records.append({
            "id": _sha256(content),
            "type": "agentHER",
            "content": content,
            "source": "mnemosyne",
            "session_id": rec.get("session_id", ""),
            "collected_at": collected_at,
        })

    return records


def deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    unique: list[dict] = []
    for rec in records:
        rid = rec["id"]
        if rid not in seen:
            seen.add(rid)
            unique.append(rec)
    return unique, len(records) - len(unique)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect SCoRe + AgentHER traces")
    p.add_argument("--out", default="mlops/finetune/data/", help="Output directory")
    p.add_argument("--ollama", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    p.add_argument("--db", default=_DEFAULT_DB, help="Path to mnemosyne.db")
    p.add_argument("--hook-state", dest="hook_state", default=_DEFAULT_STATE_DIR,
                   help="Directory containing guard_bash_*.log files")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Patch the module-level globals that score_trace_collector uses so our
    # --hook-state and --db flags take effect when calling its functions.
    import scripts.score_trace_collector as _stc
    _stc.STATE_DIR = args.hook_state
    _stc.MNEMOSYNE_DB = args.db
    _stc.OLLAMA_URL = args.ollama

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    collected_at = _now_iso()

    negatives = load_negatives()
    positives_raw = load_positives()
    corrections = build_correction_pairs(negatives, positives_raw)

    # load_agenthr_positives() in score_trace_collector already reads
    # working_memory WHERE source='agentHER', but we also want the canonical
    # pull with session_id intact for the unified schema.
    agentHER = load_agentHER_from_db(args.db)

    all_records = build_unified_records(
        negatives, positives_raw, corrections, agentHER, collected_at
    )
    unique_records, n_deduped = deduplicate(all_records)

    out_path = os.path.join(out_dir, "raw_traces.jsonl")
    with open(out_path, "w") as f:
        for rec in unique_records:
            f.write(json.dumps(rec) + "\n")

    n_neg = sum(1 for r in unique_records if r["type"] == "negative")
    n_pos = sum(1 for r in unique_records if r["type"] == "positive")
    n_corr = sum(1 for r in unique_records if r["type"] == "correction")
    n_her = sum(1 for r in unique_records if r["type"] == "agentHER")

    print(
        f"[collect] negatives={n_neg}  positives={n_pos}  "
        f"corrections={n_corr}  agentHER={n_her}  deduped={n_deduped}"
    )
    print(f"[collect] wrote {len(unique_records)} records → {out_path}")


if __name__ == "__main__":
    main()
