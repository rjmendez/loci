#!/usr/bin/env python3
"""
Collect explicit fine-tuning traces into a unified training JSONL.

The fixture-only Bash guard-log inputs were retired; production collection now
exports the AgentHER rows that really exist in Mnemosyne.
"""

import argparse
import datetime
import hashlib
import json
import os
import sqlite3
import sys


_DEFAULT_DB = os.path.expanduser(
    os.environ.get("MNEMOSYNE_DB", "~/.hermes/mnemosyne/data/mnemosyne.db")
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
        try:
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
        finally:
            conn.close()
    except sqlite3.Error as exc:
        print(f"[collect] sqlite error reading agentHER rows: {exc}", file=sys.stderr)
    return results


# ---------------------------------------------------------------------------
# Unified record builder
# ---------------------------------------------------------------------------

def build_unified_records(agentHER: list[dict], collected_at: str) -> list[dict]:
    records = []
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
    p = argparse.ArgumentParser(description="Collect AgentHER traces")
    p.add_argument("--out", default="mlops/finetune/data/", help="Output directory")
    p.add_argument("--db", default=_DEFAULT_DB, help="Path to mnemosyne.db")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    collected_at = _now_iso()
    agentHER = load_agentHER_from_db(args.db)
    unique_records, n_deduped = deduplicate(build_unified_records(agentHER, collected_at))

    out_path = os.path.join(out_dir, "raw_traces.jsonl")
    with open(out_path, "w") as f:
        for rec in unique_records:
            f.write(json.dumps(rec) + "\n")

    print(
        f"[collect] negatives=0  positives=0  "
        f"corrections=0  agentHER={len(unique_records)}  deduped={n_deduped}"
    )
    print(f"[collect] wrote {len(unique_records)} records → {out_path}")


if __name__ == "__main__":
    main()
