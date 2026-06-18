#!/usr/bin/env python3
"""
skillops_maintenance.py — detect skill shadowing and update last_validated dates.

Algorithm:
  1. Find all SKILL.md files under SKILLS_ROOTS
  2. Read name + description from YAML frontmatter
  3. Embed each description via Ollama nomic-embed-text
  4. Compute pairwise cosine similarity
  5. Print SHADOW_RISK lines for pairs above SHADOW_THRESHOLD
  6. Update last_validated date in each file's frontmatter
"""
from __future__ import annotations

import json
import math
import os
import re
import urllib.request
import urllib.error
from datetime import date
from pathlib import Path
from typing import Optional

# ── config ───────────────────────────────────────────────────────────────────
SKILLS_ROOTS = [
    os.path.expanduser("~/.claude/skills"),
    os.path.expanduser("~/.hermes/skills"),
]
SHADOW_THRESHOLD = float(os.environ.get("SHADOW_THRESHOLD", "0.92"))
OLLAMA_URL = os.environ.get("OLLAMA_URL")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
EMBED_TIMEOUT = 5.0
TODAY = date.today().isoformat()


# ── frontmatter helpers ───────────────────────────────────────────────────────

def _parse_frontmatter(text: str) -> dict:
    """Extract key: value pairs from YAML frontmatter block."""
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end == -1:
        return {}
    fm_block = text[3:end]
    result = {}
    for line in fm_block.splitlines():
        m = re.match(r"^(\w[\w-]*):\s*(.*)", line)
        if m:
            key = m.group(1).strip()
            val = m.group(2).strip().strip('"').strip("'")
            result[key] = val
    return result


def _update_last_validated(file_path: Path, today: str) -> None:
    """Set or insert last_validated: <today> in the frontmatter."""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    if re.search(r"^last_validated:", text, re.MULTILINE):
        updated = re.sub(
            r"^(last_validated:\s*).*$",
            rf"\g<1>{today}",
            text,
            flags=re.MULTILINE,
        )
    else:
        # Insert after the opening ---
        updated = re.sub(
            r"^(---\n)",
            rf"\1last_validated: {today}\n",
            text,
            count=1,
        )
    if updated != text:
        file_path.write_text(updated, encoding="utf-8")


# ── embedding ─────────────────────────────────────────────────────────────────

def _embed(text: str) -> Optional[list[float]]:
    url = f"{OLLAMA_URL.rstrip('/')}/api/embeddings"
    body = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as resp:
            d = json.loads(resp.read())
        return d.get("embedding")
    except Exception as exc:
        print(f"  WARN: embed failed for text snippet — {exc}")
        return None


# ── cosine similarity ─────────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Collect SKILL.md files
    skill_files: list[Path] = []
    for root in SKILLS_ROOTS:
        p = Path(root)
        if p.exists():
            skill_files.extend(p.rglob("SKILL.md"))

    if not skill_files:
        print("No SKILL.md files found under configured roots.")
        return

    print(f"Found {len(skill_files)} SKILL.md files.\n")

    # 2. Parse name + description from frontmatter
    records: list[dict] = []
    for fp in skill_files:
        text = fp.read_text(encoding="utf-8", errors="replace")
        fm = _parse_frontmatter(text)
        name = fm.get("name") or fp.parent.name
        description = fm.get("description") or ""
        records.append({"path": fp, "name": name, "description": description})

    # 3. Embed each description
    print("Embedding descriptions via Ollama...")
    for rec in records:
        desc = rec["description"]
        if desc:
            rec["vector"] = _embed(desc)
        else:
            rec["vector"] = None
        if rec["vector"] is None:
            print(f"  SKIP embed: {rec['name']} (no description or embed failed)")

    # 4. Compute pairwise cosine similarity
    print("\nChecking pairwise similarity...")
    shadow_pairs = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            va = records[i].get("vector")
            vb = records[j].get("vector")
            if va is None or vb is None:
                continue
            sim = _cosine(va, vb)
            if sim >= SHADOW_THRESHOLD:
                shadow_pairs.append((records[i]["name"], records[j]["name"], sim))

    # 5. Print shadow risks
    if shadow_pairs:
        print(f"\nSHADOW_RISK pairs (threshold={SHADOW_THRESHOLD}):")
        for name_a, name_b, score in sorted(shadow_pairs, key=lambda x: -x[2]):
            print(f"  SHADOW_RISK: {name_a} <-> {name_b} sim={score:.3f}")
    else:
        print(f"No shadow pairs found above threshold {SHADOW_THRESHOLD}.")

    # 6. Update last_validated in each file
    print(f"\nUpdating last_validated to {TODAY} in all SKILL.md files...")
    for rec in records:
        _update_last_validated(rec["path"], TODAY)
        print(f"  updated: {rec['path']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
