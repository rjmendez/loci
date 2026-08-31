#!/usr/bin/env python3
"""Turn hunt output into investigation findings the grounding builder can see.

Two days of hunting dama-gotchi produced 1,021 adjudicated records with real
text. None of it reached the model: the builder globs
``~/.hermes/memory-sessions/dt-loci-*/findings.jsonl`` and the hunts wrote plain
JSON into ``~/.loci/scratch/``, where prune.sh deletes it after 30 days. The
corpus stayed at 303 findings across both days, and the retrained model came out
bit-identical.

This converts a hunt directory into one investigation per source file, under a
``dt-loci-hunt-*`` id so the existing glob picks it up with no other change.

Idempotent: a finding's id is a hash of (investigation, file, line, text), so
re-running overwrites rather than duplicates. That matters because the builder
keeps one row per id and a duplicate would silently reweight the corpus.

Only records carrying real text are emitted. A record whose text is empty is the
exact defect the builder was fixed for in #241, and re-introducing it here would
undo that fix from the other end.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
from datetime import datetime, timezone

DEFAULT_CORPUS = os.path.expanduser(
    os.environ.get("LOCI_MEMORY_DIR", "~/.hermes/memory-sessions")
)

# file -> (text field, label field, topic field). Each hunt writes its own shape;
# rather than guess, name them, and skip anything unrecognised out loud.
SHAPES = {
    "FINAL-verdicts.json": {"text": "claim", "detail": "failure_scenario",
                            "label": "verdict", "topic": "file", "sev": "severity"},
    "ALL-trace.json":      {"text": "missing_effect", "detail": "dead_ends_at",
                            "label": "verdict", "topic": "dim", "sev": "severity"},
    "ALL-explained.json":  {"text": "why", "detail": "what_would_settle_it",
                            "label": "classification", "topic": "shard", "sev": "severity"},
}


def _topic(raw: str) -> str:
    """A dt_target tag: the builder groups pairs by it, so it must be stable and
    coarse enough that a topic has more than one member."""
    if not raw:
        return "unknown"
    stem = pathlib.PurePosixPath(raw).stem or raw
    stem = re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-").lower()
    return stem or "unknown"


def _fid(inv: str, rec: dict, text: str) -> str:
    h = hashlib.sha256(
        f"{inv}|{rec.get('file') or rec.get('name','')}|{rec.get('line','')}|{text}".encode()
    ).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def convert(path: pathlib.Path, shape: dict, inv: str, ts: str, now: int) -> list[dict]:
    try:
        rows = json.load(path.open())
    except Exception as exc:
        print(f"  {path.name}: unreadable ({exc}) — skipped", file=sys.stderr)
        return []
    if not isinstance(rows, list):
        rows = rows.get("findings") or rows.get("verdicts") or []

    out, empty = [], 0
    for r in rows:
        text = (r.get(shape["text"]) or "").strip()
        if not text:
            empty += 1
            continue
        detail = (r.get(shape["detail"]) or "").strip()
        body = f"{_topic(r.get(shape['topic'], ''))}: {text}"
        if detail:
            body += f" — {detail}"
        label = (r.get(shape["label"]) or "unlabelled").lower()
        out.append({
            "id": _fid(inv, r, text),
            "investigation_id": inv,
            "ts": ts,
            "created_at_ts": now,
            "record_type": "inferred",
            "type": "inferred",
            "text": body[:2000],
            "source": f"hunt://{inv}/{path.name}",
            "confidence": {"confirmed": "high", "dark": "high",
                           "plausible": "medium", "wired": "medium"}.get(label, "low"),
            "tags": [
                f"dt_run:{inv}",
                "dt_phase:hunt",
                "dt_agent:hunt",
                f"dt_target:{_topic(r.get(shape['topic'], ''))}",
                f"hunt_verdict:{label}",
                f"hunt_severity:{(r.get(shape['sev']) or 'unknown').lower()}",
            ],
            "entities": {"ips": [], "hashes": [], "cves": [],
                         "emails": [], "hostnames": [], "urls": []},
        })
    if empty:
        print(f"  {path.name}: {empty} records had no text and were skipped")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("hunt_dirs", nargs="+", help="scratch directories to convert")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--prefix", default="dt-loci-hunt")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ts = datetime.now(timezone.utc).isoformat()
    now = int(datetime.now(timezone.utc).timestamp())
    total = 0
    for d in a.hunt_dirs:
        p = pathlib.Path(os.path.expanduser(d))
        if not p.is_dir():
            print(f"{p}: not a directory — skipped", file=sys.stderr)
            continue
        known = [f for f in sorted(p.iterdir()) if f.name in SHAPES]
        if not known:
            print(f"{p.name}: no recognised hunt output "
                  f"(looked for {', '.join(SHAPES)}) — skipped", file=sys.stderr)
            continue
        for f in known:
            inv = f"{a.prefix}-{_topic(p.name)}-{f.stem.lower()}"
            recs = convert(f, SHAPES[f.name], inv, ts, now)
            if not recs:
                continue
            dest = pathlib.Path(a.corpus) / inv
            print(f"{inv}: {len(recs)} findings <- {p.name}/{f.name}"
                  + (" (dry run)" if a.dry_run else ""))
            if not a.dry_run:
                dest.mkdir(parents=True, exist_ok=True)
                with (dest / "findings.jsonl").open("w") as fh:
                    for r in recs:
                        fh.write(json.dumps(r) + "\n")
            total += len(recs)
    print(f"{total} findings {'would be' if a.dry_run else ''} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
