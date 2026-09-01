#!/usr/bin/env python3
"""
generate_memory_index.py — build the curated MEMORY.md index from memory files.

Replaces hand-maintenance of MEMORY.md: hand-trimming does not hold (the index
overflowed its context-load budget twice within days of a manual trim), and a
flat hand-edited list silently drops files it forgets to link (orphans).

This script derives the index from the memory files themselves and enforces
the budget structurally — by construction the output can never exceed it.

Usage:
    scripts/generate_memory_index.py [memory_dir] [--budget-chars N] [--source PATH]
                                      [--headroom-pct N] [--dry-run] [--check]

memory_dir defaults to backends.memory_dir(), which resolves in order:
LOCI_MEMORY_MD_DIR env -> gitignored config [memory].dir -> LOCI_MEMORY_DIR env ->
HERMES_MEMORY_DIR env -> '' (unconfigured) — the same resolution the grounding
lane uses. No host-specific path is hardcoded here.

--source defaults to <memory_dir>/MEMORY.md when that file exists, so the
default run preserves its existing section grouping instead of collapsing it
to metadata.type buckets (the destructive failure mode this script must not
have on its primary documented invocation).

Output: <memory_dir>/MEMORY.md. An existing file is backed up first, as
MEMORY.md.bak-YYYYMMDD-HHMMSS-ffffff (matching the operator's manual-backup
convention, with microseconds so same-second reruns don't clobber a backup),
unless --dry-run. --headroom-pct (default 5) targets generation below
--budget-chars so a manual edit between runs doesn't immediately re-overflow.

Wiring: nothing currently invokes this script on a schedule or in CI — the
budget guarantee is structural only when something calls it. --check exits
non-zero when the on-disk index has drifted from what regeneration would
produce, without writing; wire that into a cron job or pre-commit hook to make
the guarantee real rather than aspirational.
"""
from __future__ import annotations

import argparse
import datetime
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))

DEFAULT_BUDGET_CHARS = 24986  # 24.4 KiB, measured in characters — see [m1]
DEFAULT_HEADROOM_PCT = 5.0
ELLIPSIS = "…"
_FRONTMATTER_KEY_RE = re.compile(r'^([A-Za-z_][\w-]*):\s*(.*)$')
_SECTION_RE = re.compile(r'^##\s+(.+?)\s*$')
_LINK_RE = re.compile(r'\(([\w.\-]+\.md)\)')
_WORKING_AGREEMENTS_SECTION = "Working agreements"
_MARKER_WARNING = "⚠"  # ⚠ — entries carrying this keep more of their hook under truncation
_MARKER_STATUS = "✅"   # ✅


def resolve_memory_dir(cli_arg: str | None) -> str | None:
    if cli_arg:
        return cli_arg
    try:
        import backends
        return backends.memory_dir() or None
    except Exception as exc:
        print(f"[generate_memory_index] backends.memory_dir() unavailable: {exc}", file=sys.stderr)
        return None


def _unquote(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    return v


def parse_frontmatter(text: str) -> tuple[dict, str | None]:
    """Pull name/description/metadata.type out of a '---' YAML-ish block.

    Never raises. Returns (fields, error) — error is set (but fields still
    populated with whatever was recoverable) for anything malformed, so a
    broken file is reported rather than silently skipped."""
    fields: dict = {"name": None, "description": None, "type": None}
    start = text.find("---")
    if start == -1 or text[:start].strip():
        return fields, "no leading frontmatter delimiter"
    end = text.find("\n---", start + 3)
    if end == -1:
        return fields, "unterminated frontmatter block (no closing ---)"
    block = text[start + 3:end]

    in_metadata = False
    for raw in block.splitlines():
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        m = _FRONTMATTER_KEY_RE.match(line)
        if indent == 0:
            if not m:
                continue
            key, val = m.group(1), _unquote(m.group(2))
            if key == "metadata":
                in_metadata = True
                continue
            in_metadata = False
            if key in fields:
                fields[key] = val or None
        elif in_metadata and m and m.group(1) == "type":
            fields["type"] = _unquote(m.group(2)) or None

    errors = [f"missing '{k}'" for k in ("name", "description") if not fields.get(k)]
    return fields, "; ".join(errors) if errors else None


class Entry:
    __slots__ = ("filename", "title", "hook", "mtype", "error")

    def __init__(self, filename: str, title: str, hook: str, mtype: str | None, error: str | None):
        self.filename = filename
        self.title = title
        self.hook = hook
        self.mtype = mtype
        self.error = error


def _title_from(name: str | None, path: Path) -> str:
    raw = name or path.stem
    return raw.replace("-", " ").replace("_", " ").strip()


def load_entries(memory_dir: Path) -> tuple[list[Entry], list[tuple[str, str]]]:
    files = sorted(p for p in memory_dir.glob("*.md") if p.name != "MEMORY.md")
    entries: list[Entry] = []
    malformed: list[tuple[str, str]] = []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            err = str(exc)
            malformed.append((p.name, err))
            print(f"[generate_memory_index] WARNING: {p.name}: unreadable: {err}", file=sys.stderr)
            entries.append(Entry(p.name, _title_from(None, p), f"(unreadable: {err})", None, err))
            continue
        fields, err = parse_frontmatter(text)
        if err:
            malformed.append((p.name, err))
            print(f"[generate_memory_index] WARNING: {p.name}: {err}", file=sys.stderr)
        title = _title_from(fields.get("name"), p)
        hook = fields.get("description") or ""
        if err:
            note = f"(frontmatter: {err})"
            hook = f"{hook} {note}".strip() if hook else note
        entries.append(Entry(p.name, title, hook, fields.get("type"), err))
    return entries, malformed


def parse_source_sections(text: str) -> tuple[list[str], dict[str, list[str]]]:
    """Recover '## Section' -> [filenames] from an existing index, in order."""
    order: list[str] = []
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        h = _SECTION_RE.match(line)
        if h:
            current = h.group(1).strip()
            if current not in sections:
                sections[current] = []
                order.append(current)
            continue
        if current is None:
            continue
        m = _LINK_RE.search(line)
        if m:
            sections[current].append(m.group(1))
    return order, sections


def _type_section_name(mtype: str | None) -> str:
    if not mtype:
        return "Other"
    return mtype.replace("_", " ").replace("-", " ").title()


def build_sections(entries: list[Entry], source_order: list[str],
                    source_sections: dict[str, list[str]]) -> dict[str, list[Entry]]:
    """Preserve the source index's grouping where a file is still listed there;
    group everything else (new/orphaned files) by metadata.type."""
    by_file = {e.filename: e for e in entries}
    used: set[str] = set()
    result: dict[str, list[Entry]] = {}

    for section in source_order:
        bucket = [by_file[f] for f in source_sections[section] if f in by_file and f not in used]
        used.update(e.filename for e in bucket)
        if bucket:
            result[section] = bucket

    remaining = [e for e in entries if e.filename not in used]
    by_type: dict[str, list[Entry]] = {}
    for e in remaining:
        by_type.setdefault(_type_section_name(e.mtype), []).append(e)
    for section in sorted(by_type):
        by_type[section].sort(key=lambda e: e.filename)
        result[section] = by_type[section]

    return result


def _render_line(e: Entry) -> str:
    if e.hook:
        return f"- [{e.title}]({e.filename}) — {e.hook}"
    return f"- [{e.title}]({e.filename})"


def render(sections: dict[str, list[Entry]]) -> str:
    lines = ["# Memory index", ""]
    for section, ents in sections.items():
        lines.append(f"## {section}")
        for e in ents:
            lines.append(_render_line(e))
        lines.append("")
    text = "\n".join(lines)
    while text.endswith("\n\n"):
        text = text[:-1]
    return text if text.endswith("\n") else text + "\n"


def _boundary_cut(hook: str, limit: int) -> str:
    """Cut `hook` to at most `limit` chars, preferring a word/clause boundary
    and never leaving an unbalanced backtick span, so an identifier/path/PR
    number already inside the limit isn't split in half."""
    candidate = hook[:limit]
    if candidate.count("`") % 2 == 1:
        last_tick = candidate.rfind("`")
        if last_tick > 0:
            candidate = candidate[:last_tick]
    if candidate and not candidate[-1].isspace():
        window = max(0, len(candidate) - 40)
        for i in range(len(candidate) - 1, window - 1, -1):
            if candidate[i] in " ,;:":
                candidate = candidate[:i]
                break
    return candidate


def _truncate_hook(hook: str, allowance: int) -> str:
    if len(hook) <= allowance:
        return hook
    if allowance <= 0:
        return ""
    if allowance == 1:
        return hook[:1]
    return _boundary_cut(hook, allowance - 1).rstrip() + ELLIPSIS


def _entry_weight(section: str, hook: str) -> float:
    """Salience: warning/status markers and the operator's curated agreements
    section earn more of the truncation budget than a routine entry."""
    weight = 1.0
    if _MARKER_WARNING in hook:
        weight += 2.0
    if _MARKER_STATUS in hook:
        weight += 1.0
    if section == _WORKING_AGREEMENTS_SECTION:
        weight += 1.0
    return weight


def fit_to_budget(sections: dict[str, list[Entry]], budget_chars: int) -> bool:
    """Shorten hooks to fit, weighting the per-entry allowance by salience
    (warning/status markers, the Working agreements section) so the highest-
    value entries are truncated last and least — never touching a title or
    link. Binary-searches a scale factor over a real render, so the hard
    budget is still guaranteed. Returns False only when the budget can't be
    met even with every hook emptied out."""
    entries = [e for ents in sections.values() for e in ents]
    original = {id(e): e.hook for e in entries}
    weight = {id(e): _entry_weight(section, original[id(e)])
              for section, ents in sections.items() for e in ents}

    def apply(scale: int) -> None:
        for e in entries:
            allowance = min(len(original[id(e)]), max(0, round(scale * weight[id(e)])))
            e.hook = _truncate_hook(original[id(e)], allowance)

    def fits(scale: int) -> bool:
        apply(scale)
        return len(render(sections)) <= budget_chars

    if not fits(0):
        return False

    hi = max((len(h) for h in original.values()), default=0)
    lo, best = 0, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if fits(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    apply(best)
    return True


def _apply_headroom(budget_chars: int, headroom_pct: float) -> int:
    if headroom_pct <= 0:
        return budget_chars
    return max(0, int(budget_chars * (1 - headroom_pct / 100.0)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("memory_dir", nargs="?", default=None,
                     help="directory of memory files (default: backends.memory_dir())")
    ap.add_argument("--budget-chars", type=int, default=DEFAULT_BUDGET_CHARS,
                     help=f"hard char budget for the generated index (default {DEFAULT_BUDGET_CHARS})")
    ap.add_argument("--source", default=None,
                     help="existing index to read '## Section' grouping from "
                          "(default: <memory_dir>/MEMORY.md, if it exists)")
    ap.add_argument("--headroom-pct", type=float, default=DEFAULT_HEADROOM_PCT,
                     help=f"target this %% below --budget-chars, leaving room for a manual "
                          f"edit before the next run (default {DEFAULT_HEADROOM_PCT})")
    ap.add_argument("--dry-run", action="store_true", help="print to stdout, write nothing")
    ap.add_argument("--check", action="store_true",
                     help="exit non-zero if the on-disk index has drifted from what "
                          "regeneration would produce; writes nothing")
    args = ap.parse_args(argv)

    memory_dir_str = resolve_memory_dir(args.memory_dir)
    if not memory_dir_str:
        print("error: no memory directory given and none configured "
              "(pass one, or set LOCI_MEMORY_MD_DIR / [memory].dir / "
              "LOCI_MEMORY_DIR / HERMES_MEMORY_DIR)",
              file=sys.stderr)
        return 2
    d = Path(memory_dir_str)
    if not d.is_dir():
        print(f"error: memory dir not found: {d}", file=sys.stderr)
        return 2

    entries, malformed = load_entries(d)
    if not entries:
        print(f"error: no memory files found in {d}", file=sys.stderr)
        return 2

    out_path = d / "MEMORY.md"
    source_path = args.source
    if source_path is None and out_path.exists():
        source_path = str(out_path)

    source_order: list[str] = []
    source_sections: dict[str, list[str]] = {}
    if source_path:
        sp = Path(source_path)
        if sp.exists():
            source_order, source_sections = parse_source_sections(
                sp.read_text(encoding="utf-8", errors="replace"))
        else:
            print(f"[generate_memory_index] WARNING: --source {sp} not found; "
                  f"grouping by metadata.type instead", file=sys.stderr)

    sections = build_sections(entries, source_order, source_sections)

    target_budget = _apply_headroom(args.budget_chars, args.headroom_pct)
    if not fit_to_budget(sections, target_budget):
        print(f"error: cannot fit {len(entries)} entries within --budget-chars={args.budget_chars} "
              f"(target {target_budget} after --headroom-pct={args.headroom_pct}) "
              f"even with every hook emptied — raise the budget or reduce the memory count",
              file=sys.stderr)
        return 3

    doc = render(sections)
    if len(doc) > args.budget_chars:
        # fit_to_budget's own render() calls already guarantee this; a mismatch here
        # would mean render() is non-deterministic, which is a bug, not a soft failure.
        print(f"error: generated index is {len(doc)} chars, over budget {args.budget_chars}",
              file=sys.stderr)
        return 3

    print(f"[generate_memory_index] {len(entries)} memory files indexed "
          f"({len(malformed)} flagged: malformed/missing frontmatter), "
          f"{len(doc)} chars (budget {args.budget_chars}, target {target_budget})", file=sys.stderr)

    if args.check:
        current = out_path.read_text(encoding="utf-8") if out_path.exists() else None
        if current != doc:
            print(f"error: {out_path} is stale relative to its source memory files "
                  f"(run without --check to regenerate)", file=sys.stderr)
            return 4
        print(f"OK: {out_path} is up to date", file=sys.stderr)
        return 0

    if args.dry_run:
        print(doc)
        return 0

    if out_path.exists():
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        backup = d / f"MEMORY.md.bak-{stamp}"
        shutil.copy2(out_path, backup)
        print(f"[generate_memory_index] backed up {out_path} -> {backup}", file=sys.stderr)
    out_path.write_text(doc, encoding="utf-8")
    print(f"OK: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
