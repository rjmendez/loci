"""Supersession check — "a replaced finding is stale, say so".

A newer finding that declares it supersedes an older one (``metadata.supersedes``
or a leading ``SUPERSEDES <id>`` in its text) leaves the old one open in the
append-only store. This surfaces the old finding as an informational
``superseded`` verdict (decision ``info``). It is never a retraction.

Pure: takes plain dicts, returns ``list[Verdict]``. Fail-safe.
"""

from __future__ import annotations

import logging

from ..verdict import Verdict, make_signature, new_verdict, redact_excerpt
from ._common import _finding_id, _supersedes

__all__ = ["run_supersession"]

_log = logging.getLogger("memcheck.supersession")


def run_supersession(findings: list[dict]) -> list[Verdict]:
    """One info verdict per known finding that a newer finding supersedes."""
    by_id: dict[str, dict] = {}
    items: list[tuple[str, dict]] = []
    for index, f in enumerate(findings or []):
        if isinstance(f, dict):
            fid = _finding_id(f, index)
            by_id[fid] = f
            items.append((fid, f))

    verdicts: list[Verdict] = []
    seen: set[str] = set()
    for new_id, f in items:
        try:
            for old_id in sorted(_supersedes(f)):
                if old_id == new_id or old_id not in by_id or old_id in seen:
                    continue
                seen.add(old_id)
                verdicts.append(
                    new_verdict(
                        subject_kind="memory",
                        subject_signature=make_signature("memory", old_id),
                        subject_excerpt=redact_excerpt(str(by_id[old_id].get("text", "") or "")),
                        verdict_type="superseded",
                        decision="info",
                        confidence=0.9,
                        rationale=f"superseded by newer finding {new_id}; informational, not a retraction",
                        source="rule",
                        refs=[old_id, new_id],
                    )
                )
        except Exception as exc:
            _log.debug("supersession: skipping finding %s: %s", new_id, exc)
    return verdicts
