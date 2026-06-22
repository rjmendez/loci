"""
skill_annotation_updater.py — Nightly SKILL.md self-annotation from PostToolUse failure logs.

Reads guard_tool_reflections.log (and optionally guard_bash_failures.log) from STATE_DIR,
then appends/replaces "## Learned constraints" sections in matching SKILL.md files.
"""

import collections
import datetime
import glob
import json
import os
import re
import sys


STATE_DIR_DEFAULT = os.path.expanduser("~/.claude/hook-state")
SKILLS_DIR_DEFAULT = os.path.expanduser("~/.claude/skills")
MIN_USES_DEFAULT = 3

LEARNED_CONSTRAINTS_HEADER = "## Learned constraints"
LEARNED_CONSTRAINTS_RE = re.compile(
    r"^## Learned constraints\b.*?(?=^##|\Z)", re.MULTILINE | re.DOTALL
)


def load_jsonl(path):
    records = []
    try:
        with open(path, encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    records.append(json.loads(raw))
                except json.JSONDecodeError as exc:
                    print(f"  WARN: {path}:{lineno}: JSON parse error — {exc}", file=sys.stderr)
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(f"  WARN: could not read {path}: {exc}", file=sys.stderr)
    return records


def aggregate_failures(records, failures_by_tool):
    for rec in records:
        tool_name = rec.get("tool_name", "").strip()
        category = rec.get("category", "unknown").strip()
        note = rec.get("note", "").strip()
        if not tool_name:
            continue
        pattern = f"{category}: {note[:80]}"
        failures_by_tool[tool_name].append(pattern)


def find_skill_md(skills_dir, tool_name):
    """Return path to first SKILL.md whose frontmatter 'name:' contains tool_name (case-insensitive)."""
    pattern = os.path.join(skills_dir, "*", "SKILL.md")
    for skill_path in sorted(glob.glob(pattern)):
        try:
            with open(skill_path, encoding="utf-8") as fh:
                content = fh.read()
        except OSError:
            continue
        # Check frontmatter name field (between leading --- delimiters or first few lines)
        name_match = re.search(r"^name:\s*(.+)$", content, re.MULTILINE | re.IGNORECASE)
        if name_match:
            declared_name = name_match.group(1).strip()
            if tool_name.lower() in declared_name.lower():
                return skill_path, content
    return None, None


def build_constraints_block(failures, today_str):
    n = len(failures)
    # Deduplicate while preserving order, take up to 3 unique patterns
    seen = set()
    bullets = []
    for pattern in failures:
        key = pattern[:90]
        if key not in seen:
            seen.add(key)
            bullets.append(f"- {key}")
        if len(bullets) >= 3:
            break

    bullet_text = "\n".join(bullets)
    block = (
        f"{LEARNED_CONSTRAINTS_HEADER}\n"
        f"_Auto-updated from {n} failures. {today_str}_\n"
        f"{bullet_text}\n"
    )
    return block


def update_skill_md(skill_path, existing_content, constraints_block):
    if LEARNED_CONSTRAINTS_HEADER in existing_content:
        # Replace existing section
        new_content = LEARNED_CONSTRAINTS_RE.sub(
            lambda _: constraints_block,
            existing_content,
            count=1,
        )
        # If regex didn't match (edge case), fall back to append
        if new_content == existing_content:
            new_content = existing_content.rstrip("\n") + "\n\n" + constraints_block
    else:
        new_content = existing_content.rstrip("\n") + "\n\n" + constraints_block

    try:
        with open(skill_path, "w", encoding="utf-8") as fh:
            fh.write(new_content)
        return True
    except OSError as exc:
        print(f"  ERROR: could not write {skill_path}: {exc}", file=sys.stderr)
        return False


def main():
    state_dir = os.environ.get("STATE_DIR", STATE_DIR_DEFAULT)
    skills_dir = os.environ.get("SKILLS_DIR", SKILLS_DIR_DEFAULT)
    min_uses = int(os.environ.get("MIN_USES", MIN_USES_DEFAULT))
    today_str = datetime.date.today().isoformat()

    reflections_log = os.path.join(state_dir, "guard_tool_reflections.log")
    bash_failures_log = os.path.join(state_dir, "guard_bash_failures.log")

    # Primary log must exist
    if not os.path.exists(reflections_log):
        print("No log yet")
        sys.exit(0)

    failures_by_tool = collections.defaultdict(list)

    tool_records = load_jsonl(reflections_log)
    aggregate_failures(tool_records, failures_by_tool)

    bash_records = load_jsonl(bash_failures_log)
    aggregate_failures(bash_records, failures_by_tool)

    total_events = sum(len(v) for v in failures_by_tool.values())

    updated_count = 0
    for tool_name, failures in failures_by_tool.items():
        if tool_name == "Bash":
            continue
        if len(failures) < min_uses:
            continue

        skill_path, existing_content = find_skill_md(skills_dir, tool_name)
        if skill_path is None:
            continue

        constraints_block = build_constraints_block(failures, today_str)
        if update_skill_md(skill_path, existing_content, constraints_block):
            updated_count += 1

    print(f"Updated {updated_count} SKILL.md files from {total_events} failure events")


if __name__ == "__main__":
    main()
