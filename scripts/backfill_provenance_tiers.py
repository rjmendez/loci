#!/usr/bin/env python3
"""Backfill provenance tiers for findings stored without one, where the source is unambiguous.

Findings written before the provenance tier existed, or by writers that never set
it, read back as ``tool_verified`` with ``provenance_defaulted=True``. The
firewall already refuses to count a defaulted row as independent evidence, so
these rows are safe but uninformative. This script asserts a tier for a row
only when its own stored fields make the answer unambiguous, and leaves every
other row untagged:

  finding_type_assumed_or_gap  record_type/type is ``assumed`` or ``gap``
                               -> model_asserted
  model_writer_source          source is one of Loci's model writers
                               (investigation_reason, swarm_reason, reflect,
                               reflection_loop_tick, llm, ...) -> model_asserted
  deep_think_ideate_source     source is a ``dt://.../ideate/...`` deep-think
                               ideation writer -> model_asserted
  reasoned_text_marker         text starts with ``[reasoned]`` (the marker
                               investigation_reason writes) -> model_asserted
  audit_receipt_tool           an audit receipt row (record_type ``audit`` or
                               source ``audit_log``) naming its tool; the tier is
                               provenance_firewall.audit_provenance_fields(tool):
                               tool_verified for a non-model tool, model_asserted
                               for a model tool
  explicit_human_author        authored_by is ``human`` / ``human:<name>`` or
                               author_type is ``human`` -> human_authored

A row that already carries a known tier is never touched. A row that matches
rules pointing at different tiers is skipped as ``conflict``. Nothing is
inferred from free text beyond the one fixed marker.

Storage: nothing is rewritten. Each assertion is one appended record in the
investigation's ``provenance_updates.jsonl``::

    {"record_type": "provenance_tier", "finding_id": ..., "evidence_provenance_tier": ...,
     "rule": ..., "source": "backfill_provenance_tiers", "ts": ...}

Loci's firewall readers (build_validation_evidence, verify_all, verify_finding,
knowledge promotion) and investigation_load/search overlay it on the untagged
row. An explicit tier on the row always wins over a backfill record.

Safety:
  * Dry run is the default: it only reports per-rule and per-investigation counts.
  * --apply first copies every provenance_updates.jsonl it will append to (and a
    plan of the records) into a timestamped backup directory, then appends under
    the investigation's ``.lock`` and the log file's own flock.
  * Rerunning is safe: findings that already have a backfill record are skipped.
  * Rollback: restore the backed-up provenance_updates.jsonl files, or delete the
    ones the backup lists as ``created`` (see ROLLBACK.txt in the backup dir).
  * Stop loci-mcp before --apply. Never point this at a live store in tests.

Usage:
    backfill_provenance_tiers.py                          # dry run over $LOCI_MEMORY_DIR
    backfill_provenance_tiers.py --memory-dir DIR --json  # machine-readable report
    backfill_provenance_tiers.py --memory-dir DIR --apply [--backup-dir DIR]
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_MCP_DIR = Path(__file__).resolve().parent.parent / "mcp"
if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

from provenance_firewall import (  # noqa: E402
    HUMAN_AUTHORED,
    MODEL_ASSERTED,
    audit_provenance_fields,
    provenance_fields,
)

FINDINGS = "findings.jsonl"
UPDATES = "provenance_updates.jsonl"
SOURCE_TAG = "backfill_provenance_tiers"
_VALID_ID = re.compile(r"^[A-Za-z0-9_\-]+$")
_SENTINEL_IDS = frozenset({"undefined", "null", "none", "nan"})

#: Sources that Loci's own model-backed writers stamp on the findings they persist.
MODEL_WRITER_SOURCES = frozenset({
    "investigation_reason",
    "investigation_reflect",
    "reflect",
    "reflection_loop_tick",
    "swarm",
    "swarm_reason",
    "llm",
    "llm_local",
    "adversarial_review",
})
MODEL_FINDING_TYPES = frozenset({"assumed", "gap"})
REASONED_MARKER = "[reasoned]"

RULES = (
    "finding_type_assumed_or_gap",
    "model_writer_source",
    "deep_think_ideate_source",
    "reasoned_text_marker",
    "audit_receipt_tool",
    "explicit_human_author",
)


def default_memory_dir() -> Path:
    return Path(os.environ.get("LOCI_MEMORY_DIR", Path.home() / ".loci" / "memory-sessions"))


def _meta(row: dict) -> dict:
    meta = row.get("metadata")
    return meta if isinstance(meta, dict) else {}


def _is_human_author(value) -> bool:
    text = str(value or "").strip().lower()
    return text == "human" or text.startswith("human:")


def rule_matches(row: dict) -> list[tuple[str, str]]:
    """Every (rule, tier) the row's stored fields support. Pure; no I/O."""
    out: list[tuple[str, str]] = []
    meta = _meta(row)
    ftype = str(row.get("record_type") or row.get("type") or "").strip().lower()
    source = str(row.get("source") or "").strip()
    source_l = source.lower()
    text = str(row.get("text") or "").lstrip()

    if ftype in MODEL_FINDING_TYPES or str(row.get("type") or "").strip().lower() in MODEL_FINDING_TYPES:
        out.append(("finding_type_assumed_or_gap", MODEL_ASSERTED))
    if source_l in MODEL_WRITER_SOURCES:
        out.append(("model_writer_source", MODEL_ASSERTED))
    if source_l.startswith("dt://") and "ideate" in source_l.split("://", 1)[1].split("/"):
        out.append(("deep_think_ideate_source", MODEL_ASSERTED))
    if text.lower().startswith(REASONED_MARKER):
        out.append(("reasoned_text_marker", MODEL_ASSERTED))
    if ftype == "audit" or source_l == "audit_log":
        tool = row.get("tool") or meta.get("tool")
        if str(tool or "").strip():
            out.append(("audit_receipt_tool", audit_provenance_fields(tool)["evidence_provenance_tier"]))
    if (_is_human_author(row.get("authored_by"))
            or str(row.get("author_type") or meta.get("author_type") or "").strip().lower() == "human"):
        out.append(("explicit_human_author", HUMAN_AUTHORED))
    return out


def classify(row: dict) -> dict:
    """``{"action": "tag", "tier", "rules"}`` or ``{"action": "skip", "reason"}``."""
    if not provenance_fields(row)["provenance_defaulted"]:
        return {"action": "skip", "reason": "explicit_tier"}
    matches = rule_matches(row)
    if not matches:
        return {"action": "skip", "reason": "no_unambiguous_rule"}
    tiers = {tier for _, tier in matches}
    if len(tiers) > 1:
        return {"action": "skip", "reason": "conflict", "rules": [r for r, _ in matches]}
    return {"action": "tag", "tier": tiers.pop(), "rules": [r for r, _ in matches]}


def _read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, "rb") as fh:
        for raw in fh:
            try:
                row = json.loads(raw.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _existing_backfill_ids(inv_dir: Path) -> set[str]:
    return {
        str(r.get("finding_id"))
        for r in _read_rows(inv_dir / UPDATES)
        if r.get("record_type") == "provenance_tier" and r.get("finding_id")
    }


def plan(memory_dir: Path, investigations: list[str] | None = None) -> dict:
    """Scan the store and return the report plus the records --apply would append."""
    now = datetime.now(timezone.utc).isoformat()
    per_rule = collections.Counter()
    per_tier = collections.Counter()
    skipped = collections.Counter()
    per_inv: dict[str, dict] = {}
    records: dict[str, list[dict]] = {}
    ignored_dirs: list[dict] = []
    if investigations:
        names = list(investigations)
    elif memory_dir.exists():
        names = sorted(p.name for p in memory_dir.iterdir() if p.is_dir())
    else:
        names = []
    for name in names:
        inv_dir = memory_dir / name
        if not (inv_dir / FINDINGS).exists():
            continue
        if not _VALID_ID.match(name) or name.lower() in _SENTINEL_IDS:
            ignored_dirs.append({"investigation": name, "reason": "not a valid investigation id; not written"})
            continue
        already = _existing_backfill_ids(inv_dir)
        seen: set[str] = set()
        counts = collections.Counter()
        for row in _read_rows(inv_dir / FINDINGS):
            if row.get("record_type") == "access":
                continue
            fid = str(row.get("id") or "")
            if not fid or fid in seen:
                continue
            seen.add(fid)
            if fid in already:
                skipped["already_backfilled"] += 1
                counts["already_backfilled"] += 1
                continue
            verdict = classify(row)
            if verdict["action"] == "skip":
                skipped[verdict["reason"]] += 1
                counts[verdict["reason"]] += 1
                continue
            for rule in verdict["rules"]:
                per_rule[rule] += 1
            per_tier[verdict["tier"]] += 1
            counts[f"tag:{verdict['tier']}"] += 1
            records.setdefault(name, []).append({
                "record_type": "provenance_tier",
                "finding_id": fid,
                "evidence_provenance_tier": verdict["tier"],
                "rule": "+".join(verdict["rules"]),
                "source": SOURCE_TAG,
                "ts": now,
            })
        if counts:
            per_inv[name] = dict(counts)
    return {
        "memory_dir": str(memory_dir),
        "would_tag": sum(per_tier.values()),
        "per_rule": {r: per_rule.get(r, 0) for r in RULES},
        "per_tier": dict(per_tier),
        "skipped": dict(skipped),
        "per_investigation": per_inv,
        "ignored_dirs": ignored_dirs,
        "records": records,
    }


def _append_locked(inv_dir: Path, recs: list[dict]) -> None:
    with open(inv_dir / ".lock", "a+") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            with open(inv_dir / UPDATES, "a", encoding="utf-8") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    for rec in recs:
                        fh.write(json.dumps(rec, sort_keys=True) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def apply(memory_dir: Path, report: dict, backup_dir: Path) -> dict:
    """Back up every log that will change, then append the planned records."""
    backup_dir.mkdir(parents=True, exist_ok=False)
    created, backed_up = [], []
    for name, recs in report["records"].items():
        src = memory_dir / name / UPDATES
        dst = backup_dir / name
        dst.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.copy2(src, dst / UPDATES)
            backed_up.append(name)
        else:
            created.append(name)
        with open(dst / "planned.jsonl", "w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
    (backup_dir / "ROLLBACK.txt").write_text(
        "Rollback for backfill_provenance_tiers --apply\n"
        f"memory dir: {memory_dir}\n\n"
        "Stop loci-mcp, then for each investigation below:\n"
        "  backed_up: cp <backup>/<inv>/provenance_updates.jsonl <memory>/<inv>/provenance_updates.jsonl\n"
        "  created:   rm <memory>/<inv>/provenance_updates.jsonl\n\n"
        f"backed_up: {json.dumps(sorted(backed_up))}\n"
        f"created: {json.dumps(sorted(created))}\n",
        encoding="utf-8",
    )
    appended = 0
    for name, recs in report["records"].items():
        _append_locked(memory_dir / name, recs)
        appended += len(recs)
    return {"appended": appended, "backup_dir": str(backup_dir),
            "backed_up": sorted(backed_up), "created": sorted(created)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--memory-dir", type=Path, default=None)
    ap.add_argument("--investigation", action="append", default=None,
                    help="limit to this investigation id (repeatable)")
    ap.add_argument("--apply", action="store_true", help="append the records (default: dry run)")
    ap.add_argument("--backup-dir", type=Path, default=None,
                    help="where --apply backs up logs (default: <memory-dir>/../backups/provenance-backfill-<ts>)")
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = ap.parse_args(argv)

    memory_dir = (args.memory_dir or default_memory_dir()).expanduser()
    if not memory_dir.is_dir():
        print(f"memory dir not found: {memory_dir}", file=sys.stderr)
        return 2
    report = plan(memory_dir, args.investigation)
    summary = {k: v for k, v in report.items() if k != "records"}
    summary["mode"] = "apply" if args.apply else "dry-run"
    if args.apply and report["would_tag"]:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        backup = args.backup_dir or (memory_dir.parent / "backups" / f"provenance-backfill-{stamp}")
        summary["applied"] = apply(memory_dir, report, backup)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"{summary['mode']}: {report['would_tag']} finding(s) "
              f"{'tagged' if args.apply else 'would be tagged'} in {memory_dir}")
        print("per rule:")
        for rule in RULES:
            print(f"  {rule:32s} {report['per_rule'][rule]}")
        print("per tier: " + (", ".join(f"{t}={n}" for t, n in sorted(report["per_tier"].items())) or "none"))
        print("left untagged: " + (", ".join(f"{r}={n}" for r, n in sorted(report["skipped"].items())) or "none"))
        for d in report["ignored_dirs"]:
            print(f"ignored dir {d['investigation']!r}: {d['reason']}")
        if "applied" in summary:
            print(f"appended {summary['applied']['appended']} record(s); backup + ROLLBACK.txt in "
                  f"{summary['applied']['backup_dir']}")
        elif not args.apply:
            print("dry run: nothing written. Re-run with --apply (service stopped) to append.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
