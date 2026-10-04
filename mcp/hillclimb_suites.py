"""Built-in hillclimb suites.

``triage`` grades ``reflection_triage.classify_reflection_observation`` against hand-labelled
observations in ``eval/hillclimb/triage_cases.jsonl`` (override with ``LOCI_HILLCLIMB_TRIAGE_CASES``).
Its one surface is ``guidance``, the extra instruction block the classifier prompt appends.
The cases are synthetic; replace or extend them with labelled real findings as they accumulate
(the split is keyed on case id, so adding cases does not reshuffle the old ones).

A suite elsewhere (for example a log-recall self-test) plugs in as ``module:factory``: an object
with ``name``, ``surfaces`` ({key: Surface}), ``cases()`` and ``run(case, overlay) -> CaseResult``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Optional

from hillclimb import Case, CaseResult, Surface

_DEFAULT_CASES = Path(__file__).resolve().parent.parent / "eval" / "hillclimb" / "triage_cases.jsonl"


class TriageSuite:
    name = "reflection_triage"
    surfaces = {
        "guidance": Surface(
            "guidance", "text", default="", max_len=1200,
            desc=("Extra instructions appended to the classifier prompt. Use it to teach the model how to "
                  "tell the categories apart (what makes a failure a regression versus flaky versus an "
                  "environment problem versus noise). Plain text, no placeholders."),
        ),
    }

    def __init__(self, cases_path: Optional[Path] = None, gen_fn: Optional[Callable[..., dict]] = None):
        self._path = Path(cases_path or os.environ.get("LOCI_HILLCLIMB_TRIAGE_CASES") or _DEFAULT_CASES)
        self._gen = gen_fn

    def cases(self) -> list[Case]:
        out = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                out.append(Case(str(d["id"]), d))
        return out

    def run(self, case: Case, overlay: dict) -> CaseResult:
        from reflection_triage import _lazy_generate, classify_reflection_observation

        d = case.data
        seen = {"ok": True}
        real = self._gen or _lazy_generate

        def gen(prompt, **kw):
            r = real(prompt, **kw)
            seen["ok"] = bool(isinstance(r, dict) and r.get("ok"))
            return r

        res = classify_reflection_observation(
            d["kind"], d["path"], events=d.get("events"), tools=d.get("tools"),
            errors=d.get("errors"), warnings=d.get("warnings"), gen_fn=gen,
            guidance=str(overlay.get("guidance", "")),
        )
        if not seen["ok"]:
            return CaseResult(case.id, 0.0, "model unavailable", infra_error=True)
        got = res.get("category")
        hit = got == d["gold"]
        trace = (f"expected {d['gold']}, got {got}. kind={d['kind']} errors={d.get('errors')} "
                 f"warnings={d.get('warnings')} events={d.get('events')}")
        return CaseResult(case.id, 1.0 if hit else 0.0, "" if hit else trace)


BUILTIN = {"triage": TriageSuite}
