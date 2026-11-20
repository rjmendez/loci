"""Built-in hillclimb suites.

``triage`` grades ``reflection_triage.classify_reflection_observation`` against hand-labelled
observations in ``eval/hillclimb/triage_cases.jsonl`` (override with ``LOCI_HILLCLIMB_TRIAGE_CASES``).
Real labels (``hillclimb.py label``) are read from the suite folder next to them. Its one surface is ``guidance``, the extra instruction block the classifier prompt appends.
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

from hillclimb import Case, CaseResult, Surface, suite_dir

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
        """Synthetic cases (unless LOCI_HILLCLIMB_TRIAGE_SYNTHETIC=0) plus the person's real labels
        from ``<suite folder>/labels.jsonl`` (see ``hillclimb.py label``)."""
        paths = []
        if os.environ.get("LOCI_HILLCLIMB_TRIAGE_SYNTHETIC", "1") != "0":
            paths.append(self._path)
        paths.append(suite_dir(self.name) / "labels.jsonl")
        out, seen = [], set()
        for path in paths:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                if line.strip():
                    d = json.loads(line)
                    if str(d["id"]) not in seen:
                        seen.add(str(d["id"]))
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
        # A case a person rated for novelty is scored on both outputs (mean of the two); "unclear" is
        # not graded because it is a valid answer to a thin observation. Cases without a rating
        # (the synthetic ones, older labels) are scored on category alone, as before.
        gold_nov = d.get("novelty")
        graded_nov = gold_nov in ("novel_signal", "known_pattern")
        nov_hit = graded_nov and res.get("novelty") == gold_nov
        score = ((1.0 if hit else 0.0) + (1.0 if nov_hit else 0.0)) / 2 if graded_nov else (1.0 if hit else 0.0)
        trace = (f"expected {d['gold']}, got {got}. kind={d['kind']} errors={d.get('errors')} "
                 f"warnings={d.get('warnings')} events={d.get('events')}")
        if graded_nov and not nov_hit:
            trace += f". novelty expected {gold_nov}, got {res.get('novelty')}"
        if d.get("note"):
            trace += f". reviewer note: {d['note']}"
        return CaseResult(case.id, score, "" if score >= 1.0 else trace)


class QueryExpandSuite:
    """Grades ``query_expand.expand`` against the teacher gold set.

    Gold rows (``<memory parent>/teacher/gold/query_expand.jsonl``, override ``LOCI_TEACHER_GOLD_DIR``):
    ``{"id", "query", "expect": [terms]}``. Score = share of ``expect`` terms that appear (case-insensitive)
    in the expanded queries or keywords. One surface, ``guidance``. No gold file -> ``unavailable()``."""
    name = "query_expand"
    surfaces = {
        "guidance": Surface(
            "guidance", "text", default="", max_len=800,
            desc=("Extra instructions appended to the query-expansion prompt: how to phrase alternates and "
                  "which domain terms to surface so relevant documents are reached. Plain text."),
        ),
    }

    def __init__(self, gold_path: Optional[Path] = None, gen_fn: Optional[Callable[..., dict]] = None):
        if gold_path is None:
            base = os.environ.get("LOCI_TEACHER_GOLD_DIR")
            gold_path = (Path(base) if base else suite_dir(self.name).parent.parent / "teacher" / "gold") / "query_expand.jsonl"
        self._path = Path(gold_path)
        self._gen = gen_fn

    def _rows(self) -> list[dict]:
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        rows = []
        for line in lines:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict) and d.get("id") and d.get("query") and isinstance(d.get("expect"), list) and d["expect"]:
                rows.append(d)
        return rows

    def unavailable(self) -> Optional[str]:
        return None if self._rows() else "no gold set yet"

    def cases(self) -> list[Case]:
        seen, out = set(), []
        for d in self._rows():
            if str(d["id"]) not in seen:
                seen.add(str(d["id"]))
                out.append(Case(str(d["id"]), d))
        return out

    def run(self, case: Case, overlay: dict) -> CaseResult:
        from query_expand import expand, _lazy_generate

        d = case.data
        seen = {"ok": True}
        real = self._gen or _lazy_generate

        def gen(prompt, **kw):
            r = real(prompt, **kw)
            seen["ok"] = bool(isinstance(r, dict) and r.get("ok"))
            return r

        res = expand(d["query"], gen_fn=gen, guidance=str(overlay.get("guidance", "")))
        if not seen["ok"]:
            return CaseResult(case.id, 0.0, "model unavailable", infra_error=True)
        blob = " ".join(list(res.get("queries", [])) + list(res.get("keywords", []))).lower()
        expect = [str(t) for t in d["expect"]]
        missing = [t for t in expect if t.lower() not in blob]
        score = (len(expect) - len(missing)) / len(expect)
        return CaseResult(case.id, score, "" if not missing else f"query {d['query']!r}: expansion lacks {missing}")


BUILTIN = {"triage": TriageSuite, "query_expand": QueryExpandSuite}
