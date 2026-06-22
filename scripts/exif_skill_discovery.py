#!/usr/bin/env python3
"""
EXIF closed-loop skill discovery (arxiv 2506.04287).

Alice (Ollama gap-finder) analyzes failures -> Bob generates candidate SKILL.md
-> validator logs for human review (NO auto-promote).
"""

import datetime
import json
import os
import re
import sqlite3
import sys
import urllib.request


STATE_DIR = os.environ.get("STATE_DIR", os.path.expanduser("~/.claude/hook-state"))
SKILLS_DIR = os.environ.get("SKILLS_DIR", os.path.expanduser("~/.claude/skills"))
OLLAMA_URL = os.environ.get("OLLAMA_URL")
EXIF_GEN_MODEL = os.environ.get("EXIF_GEN_MODEL", "llama3.2:latest")
MNEMOSYNE_DB = os.environ.get(
    "MNEMOSYNE_DB",
    os.path.expanduser("~/.hermes/mnemosyne/data/mnemosyne.db"),
)
DISCOVERY_LOG = os.environ.get(
    "DISCOVERY_LOG", os.path.join(STATE_DIR, "exif_discoveries.jsonl")
)


def read_recent_failures():
    """Read recent failure memories from working_memory (last 7 days, importance>=5, LIMIT 20)."""
    if not os.path.exists(MNEMOSYNE_DB):
        print(f"[exif] WARNING: mnemosyne db not found at {MNEMOSYNE_DB}", file=sys.stderr)
        return []

    cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7)).isoformat()
    query = """
        SELECT content, importance
        FROM memories
        WHERE importance >= 5
          AND created_at >= ?
          AND (bank = 'working_memory' OR bank IS NULL)
        ORDER BY importance DESC, created_at DESC
        LIMIT 20
    """
    try:
        conn = sqlite3.connect(MNEMOSYNE_DB)
        try:
            rows = conn.execute(query, (cutoff,)).fetchall()
            return [row[0] for row in rows]
        finally:
            conn.close()
    except sqlite3.OperationalError:
        # Table or column may differ; try a fallback without bank filter
        try:
            conn = sqlite3.connect(MNEMOSYNE_DB)
            fallback_query = """
                SELECT content, importance
                FROM memories
                WHERE importance >= 5
                  AND created_at >= ?
                ORDER BY importance DESC, created_at DESC
                LIMIT 20
            """
            rows = conn.execute(fallback_query, (cutoff,)).fetchall()
            conn.close()
            return [row[0] for row in rows]
        except Exception as exc2:
            print(f"[exif] db read error: {exc2}", file=sys.stderr)
            return []


def read_existing_skill_names():
    """Read existing skill names from SKILLS_DIR/*/SKILL.md frontmatter 'name:' fields."""
    skill_names = []
    if not os.path.isdir(SKILLS_DIR):
        return skill_names

    for entry in os.scandir(SKILLS_DIR):
        if not entry.is_dir():
            continue
        skill_md = os.path.join(entry.path, "SKILL.md")
        if not os.path.exists(skill_md):
            continue
        try:
            with open(skill_md, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("name:"):
                        name_val = line[len("name:"):].strip().strip('"').strip("'")
                        if name_val:
                            skill_names.append(name_val)
                        break
        except OSError:
            continue

    return skill_names


def call_ollama(prompt: str) -> str:
    """POST to Ollama /api/generate and return the response text."""
    payload = json.dumps(
        {"model": EXIF_GEN_MODEL, "prompt": prompt, "stream": False}
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8")

    data = json.loads(body)
    return data.get("response", "")


def extract_json_from_text(text: str) -> dict:
    """Extract the first JSON object from a text block."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Find JSON block via regex
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON object found in response: {text[:200]}")


def phase_alice(skill_names: list, failure_summaries: list) -> dict:
    """Alice: gap-finder. Returns parsed suggestion dict."""
    skills_str = ", ".join(skill_names) if skill_names else "(none yet)"
    failures_str = "\n".join(failure_summaries)[:1500]

    prompt = (
        f'I have these skills: {skills_str}. '
        f'Recent failures:\n{failures_str}\n'
        'What ONE new skill would most reduce failures? '
        'Reply as JSON: '
        '{"skill_name":"...","description":"...","when_to_use":"...","solves_failure":"...","confidence":0.0}'
    )

    response_text = call_ollama(prompt)
    return extract_json_from_text(response_text)


def phase_bob(suggestion: dict) -> str:
    """Bob: generator. Writes candidate SKILL.md, returns candidate_dir path."""
    skill_name = suggestion["skill_name"]
    description = suggestion.get("description", "")
    when_to_use = suggestion.get("when_to_use", "TBD")
    solves_failure = suggestion.get("solves_failure", "TBD")

    candidate_dir = os.path.join(STATE_DIR, "candidate_skills", skill_name)
    os.makedirs(candidate_dir, exist_ok=True)

    skill_md_content = f"""---
name: {skill_name}
description: {description}
version: 0.1.0-candidate
author: exif-discovery
---

# {skill_name}

{description}

## When to use

{when_to_use}

## Solves

{solves_failure}

## NOT for

TBD

## Status

CANDIDATE - not yet promoted
"""

    skill_md_path = os.path.join(candidate_dir, "SKILL.md")
    with open(skill_md_path, "w", encoding="utf-8") as fh:
        fh.write(skill_md_content)

    return candidate_dir


def phase_validator(suggestion: dict, candidate_dir: str) -> None:
    """Validator: log for human review, NO auto-promote."""
    skill_name = suggestion["skill_name"]
    confidence = suggestion.get("confidence", 0.0)

    record = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "skill_name": skill_name,
        "confidence": confidence,
        "path": candidate_dir,
        "status": "candidate",
    }

    os.makedirs(os.path.dirname(DISCOVERY_LOG), exist_ok=True)
    with open(DISCOVERY_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    print(f"[exif] candidate skill: {skill_name} (confidence={confidence:.2f})")
    print(f"[exif] review and promote: cp -r {candidate_dir} {SKILLS_DIR}/{skill_name}/")


def main():
    # Phase 1 data gathering
    failures = read_recent_failures()
    if not failures:
        print("[exif] no recent failures found; nothing to analyze.", file=sys.stderr)
        return

    skill_names = read_existing_skill_names()

    # Phase 1 (Alice)
    try:
        suggestion = phase_alice(skill_names, failures)
    except urllib.error.URLError as exc:
        print(f"[exif] Ollama request failed: {exc}", file=sys.stderr)
        return
    except ValueError as exc:
        print(f"[exif] JSON parse failure: {exc}", file=sys.stderr)
        return
    except Exception as exc:
        print(f"[exif] unexpected error in Alice phase: {exc}", file=sys.stderr)
        return

    confidence = float(suggestion.get("confidence", 0.0))

    if confidence <= 0.6:
        print(
            f"[exif] confidence {confidence:.2f} <= 0.6; skipping candidate generation.",
            file=sys.stderr,
        )
        return

    # Phase 2 (Bob)
    try:
        candidate_dir = phase_bob(suggestion)
    except Exception as exc:
        print(f"[exif] error generating candidate: {exc}", file=sys.stderr)
        return

    # Phase 3 (validator)
    phase_validator(suggestion, candidate_dir)


if __name__ == "__main__":
    main()
