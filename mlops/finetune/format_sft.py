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
# DPO pair builder (SCoRe: SFT-only causes identity collapse; use DPO instead)
# ---------------------------------------------------------------------------

def pairs_from_corrections_dpo(records: list[dict]) -> list[dict]:
    """Emit DPO-format contrastive pairs: chosen=corrected, rejected=failed."""
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
        if failed == corrected:
            continue  # identity pair — exactly what SCoRe warns against
        pairs.append({
            "prompt": "You are a helpful assistant.",
            "chosen": [
                {"role": "user", "content": failed},
                {"role": "assistant", "content": corrected},
            ],
            "rejected": [
                {"role": "user", "content": failed},
                {"role": "assistant", "content": failed},
            ],
            "source": "score_correction_dpo",
            "session_id": rec.get("session_id", ""),
        })
    return pairs


def deduplicate_dpo(pairs: list[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    unique: list[dict] = []
    for pair in pairs:
        key = _sha256(json.dumps(pair["chosen"], sort_keys=True))
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    return unique, len(pairs) - len(unique)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Format raw traces into SFT/DPO pairs")
    p.add_argument("--traces", required=True, help="Path to raw_traces.jsonl")
    p.add_argument("--out", required=True, help="Output path for sft_pairs.jsonl")
    p.add_argument("--min-pairs", type=int, default=50,
                   help="Warn (but don't fail) if fewer pairs than this are produced")
    p.add_argument("--mode", choices=["sft", "dpo", "both"], default="sft",
                   help="sft: chat-format; dpo: chosen/rejected contrastive (prevents identity collapse per SCoRe); both: emit both")
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

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    total_written = 0

    if args.mode in ("sft", "both"):
        correction_pairs = pairs_from_corrections(records)
        agentHER_pairs = pairs_from_agentHER(records)
        all_pairs = correction_pairs + agentHER_pairs
        unique_pairs, n_deduped = deduplicate_pairs(all_pairs)
        with open(args.out, "w") as f:
            for pair in unique_pairs:
                f.write(json.dumps(pair) + "\n")
        n_corr = sum(1 for p in unique_pairs if p["source"] == "score_correction")
        n_her = sum(1 for p in unique_pairs if p["source"] == "agentHER")
        total = len(unique_pairs)
        total_written += total
        print(f"[format_sft] SFT: correction={n_corr} agentHER={n_her} deduped={n_deduped} total={total} → {args.out}")

    if args.mode in ("dpo", "both"):
        dpo_pairs = pairs_from_corrections_dpo(records)
        unique_dpo, n_deduped_dpo = deduplicate_dpo(dpo_pairs)
        dpo_path = args.out.replace(".jsonl", "_dpo.jsonl")
        if dpo_path == args.out:
            dpo_path = args.out + ".dpo.jsonl"
        with open(dpo_path, "w") as f:
            for pair in unique_dpo:
                f.write(json.dumps(pair) + "\n")
        total_written += len(unique_dpo)
        print(f"[format_sft] DPO: pairs={len(unique_dpo)} deduped={n_deduped_dpo} identity_filtered → {dpo_path}")
        if len(unique_dpo) == 0:
            print("[format_sft] WARNING: 0 DPO pairs — no correction traces collected yet.", file=sys.stderr)

    if total_written < args.min_pairs:
        print(f"[format_sft] WARNING: only {total_written} pairs (min={args.min_pairs}). Collect more traces.", file=sys.stderr)


if __name__ == "__main__":
    main()
