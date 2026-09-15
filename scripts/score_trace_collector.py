#!/usr/bin/env python3
"""
SCoRe data collection pipeline.

The original design expected Bash guard logs in ``STATE_DIR``, but no production
hook writes them. Rather than keep shipping readers for fixture-only files, the
remaining production source is AgentHER-positive rows from Mnemosyne. We keep
writing the historical output filenames so downstream manual inspection still
works, but negatives/corrections stay empty until a real producer is
intentionally introduced.
"""

import datetime
import json
import os
import sqlite3


MNEMOSYNE_DB = os.environ.get(
    "MNEMOSYNE_DB",
    os.path.expanduser("~/.hermes/mnemosyne/data/mnemosyne.db"),
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.expanduser("~/.hermes/mnemosyne/data/score_traces"),
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_agenthr_positives() -> list[dict]:
    """Query mnemosyne.db working_memory WHERE source='agentHER'."""
    results = []
    if not os.path.exists(MNEMOSYNE_DB):
        return results
    try:
        conn = sqlite3.connect(MNEMOSYNE_DB)
        try:
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
        finally:
            conn.close()
    except sqlite3.Error:
        pass
    return results


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
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "ready_for_sft": False,
    }
    path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    positives = load_agenthr_positives()
    negatives: list[dict] = []
    corrections: list[dict] = []

    write_jsonl(os.path.join(OUTPUT_DIR, "negatives.jsonl"), negatives)
    write_jsonl(os.path.join(OUTPUT_DIR, "positives.jsonl"), positives)
    write_jsonl(os.path.join(OUTPUT_DIR, "corrections.jsonl"), corrections)
    write_manifest(len(negatives), len(positives), len(corrections))

    print(
        f"[score] negatives={len(negatives)}, positives={len(positives)}, "
        f"corrections={len(corrections)}, sft_ready=False"
    )


if __name__ == "__main__":
    main()
