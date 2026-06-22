#!/usr/bin/env python3
"""
Formats raw_traces.jsonl into SFT instruction pairs for Ollama/Unsloth.

Usage:
    python3 mlops/finetune/format_sft.py \
        --traces mlops/finetune/data/raw_traces.jsonl \
        --out mlops/finetune/data/sft_pairs.jsonl \
        [--min-pairs 50]
"""

import argparse
import hashlib
import json
import sys


MIN_CONTENT_LEN = 20


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Pair builders
# ---------------------------------------------------------------------------

def pairs_from_corrections(records: list[dict]) -> list[dict]:
    pairs = []
    for rec in records:
        if rec.get("type") != "correction":
            continue
        try:
            sides = json.loads(rec["content"])
        except (json.JSONDecodeError, KeyError):
            continue
        failed = sides.get("failed", "")
        corrected = sides.get("corrected", "")
        if len(failed) < MIN_CONTENT_LEN or len(corrected) < MIN_CONTENT_LEN:
            continue
        pairs.append({
            "messages": [
                {"role": "user", "content": failed},
                {"role": "assistant", "content": corrected},
            ],
            "source": "score_correction",
            "session_id": rec.get("session_id", ""),
        })
    return pairs


def pairs_from_agentHER(records: list[dict]) -> list[dict]:
    pairs = []
    for rec in records:
        if rec.get("type") != "agentHER":
            continue
        content = rec.get("content", "")
        if len(content) < MIN_CONTENT_LEN:
            continue
        pairs.append({
            "messages": [
                {"role": "user", "content": f"Recall relevant memory for: {content}"},
                {"role": "assistant", "content": content},
            ],
            "source": "agentHER",
            "session_id": rec.get("session_id", ""),
        })
    return pairs


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------

def deduplicate_pairs(pairs: list[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    unique: list[dict] = []
    for pair in pairs:
        key = _sha256(json.dumps(pair["messages"], sort_keys=True))
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    return unique, len(pairs) - len(unique)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Format raw traces into SFT pairs")
    p.add_argument("--traces", required=True, help="Path to raw_traces.jsonl")
    p.add_argument("--out", required=True, help="Output path for sft_pairs.jsonl")
    p.add_argument("--min-pairs", type=int, default=50,
                   help="Warn (but don't fail) if fewer pairs than this are produced")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    records: list[dict] = []
    try:
        with open(args.traces) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"[format_sft] traces file not found: {args.traces}", file=sys.stderr)
        sys.exit(1)

    correction_pairs = pairs_from_corrections(records)
    agentHER_pairs = pairs_from_agentHER(records)
    all_pairs = correction_pairs + agentHER_pairs

    unique_pairs, n_deduped = deduplicate_pairs(all_pairs)

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for pair in unique_pairs:
            f.write(json.dumps(pair) + "\n")

    n_corr = sum(1 for p in unique_pairs if p["source"] == "score_correction")
    n_her = sum(1 for p in unique_pairs if p["source"] == "agentHER")
    total = len(unique_pairs)

    print(
        f"[format_sft] correction_pairs={n_corr}  agentHER_pairs={n_her}  "
        f"deduped={n_deduped}  total={total}"
    )

    if total < args.min_pairs:
        print(
            f"[format_sft] WARNING: only {total} pairs produced "
            f"(min-pairs threshold is {args.min_pairs}). "
            "Collect more traces before fine-tuning.",
            file=sys.stderr,
        )

    print(f"[format_sft] wrote {total} pairs → {args.out}")


if __name__ == "__main__":
    main()
