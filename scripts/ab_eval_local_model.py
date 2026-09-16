#!/usr/bin/env python3
"""A/B eval for the local Ollama generation tier on JSON-constrained ops.

Compares a candidate local model against the current baseline on the three cheap,
high-volume call sites that rely on the generation tier:

  - classify: choose one label from an allowed set
  - compress: condense text under a hard character budget
  - verify: adversarially judge a claim as confirmed/refuted/uncertain

The script talks directly to the Ollama /api/generate endpoint, using the same backend
resolution path as production (backends.ollama_gen_url()). It is fail-open: an
unreachable endpoint yields N/A rows rather than a crash.
"""
from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))

import backends  # noqa: E402
from model_json import extract_json_object  # noqa: E402

BASELINE_MODEL = "qwen2.5:3b"
TASKS = ("classify", "compress", "verify")
VALID_VERDICTS = {"confirmed", "refuted", "uncertain"}
TIMEOUT_S = float(os.environ.get("OLLAMA_GEN_TIMEOUT", "120"))

CLASSIFY_CASES = [
    {
        "text": "The deployment keeps failing with a 500 after the latest config change.",
        "labels": ["bug", "feature", "question"],
        "expected": "bug",
    },
    {
        "text": "Can you explain how to restore a memory snapshot from the CLI?",
        "labels": ["bug", "feature", "question"],
        "expected": "question",
    },
    {
        "text": "Add a dry-run flag so operators can preview memory demotions safely.",
        "labels": ["bug", "feature", "question"],
        "expected": "feature",
    },
    {
        "text": "The server works, but agents want a shorter alias for investigation export.",
        "labels": ["bug", "feature", "question"],
        "expected": "feature",
    },
]

COMPRESS_CASES = [
    {
        "text": (
            "Loci stores investigation findings, provenance, and code references so agents can "
            "reload prior work without re-deriving everything. The memory lane is useful, but "
            "some findings are too bulky to forward to an expensive remote model. Compress the "
            "important facts: a finding has a claim, supporting context, timestamps, and often "
            "file:line references that must stay recognizable."
        ),
        "max_chars": 160,
        "expected": (
            "Keep findings reloadable by retaining the claim, supporting context, timestamps, "
            "and recognizable file:line references."
        ),
    },
    {
        "text": (
            "A retrieval shadow eval samples findings from the Qdrant corpus, builds query sets "
            "grouped by investigation, retrieves dense candidates, reranks them, and compares "
            "recall, MRR, and nDCG before a flip is recommended. The summary should retain the "
            "gate purpose, the retrieval path, and the fact that regressions should block a flip."
        ),
        "max_chars": 180,
        "expected": (
            "The shadow eval gates flips by retrieving and reranking Qdrant findings, comparing "
            "recall, MRR, and nDCG, and blocking regressions."
        ),
    },
    {
        "text": (
            "The local generation tier must fail open. If Ollama is down, classify should return "
            "a degraded label=None result, compress should hard-truncate to budget, and verify "
            "should return an uncertain verdict. Operators need graceful degradation rather than "
            "exceptions, because scheduled maintenance jobs should keep running even when the GPU "
            "host is offline."
        ),
        "max_chars": 170,
        "expected": (
            "Fail open when Ollama is down: classify returns degraded label=None, compress "
            "truncates to budget, verify returns uncertain, and scheduled jobs keep running."
        ),
    },
]

VERIFY_CASES = [
    {
        "claim": "The migration is reversible because the old column is left untouched.",
        "context": (
            "The rollout adds a new nullable column and backfills it. Reads still consult the "
            "old column. The old column is only dropped in a later manual cleanup step."
        ),
        "expected": "confirmed",
    },
    {
        "claim": "The script never writes to disk.",
        "context": (
            "It opens output.json with write mode, dumps the response payload, and appends a log "
            "entry to state/history.log after each run."
        ),
        "expected": "refuted",
    },
    {
        "claim": "This bug only affects Linux hosts.",
        "context": (
            "A user reported the issue on Linux. No reproduction from macOS or Windows is "
            "included, and the stack trace does not mention any platform-specific path."
        ),
        "expected": "uncertain",
    },
]


@dataclass
class CallResult:
    text: str
    latency_ms: Optional[float]
    transport_ok: bool
    error: str = ""


@dataclass
class CaseScore:
    latency_ms: Optional[float]
    json_ok: bool
    schema_ok: bool
    correct: bool
    transport_ok: bool


@dataclass
class SummaryRow:
    task: str
    arm: str
    model: str
    total: int
    json_ok: int
    schema_ok: int
    correct: int
    latencies_ms: list[float]
    available: bool
    note: str = ""

    def json_rate(self) -> Optional[float]:
        return None if not self.available or not self.total else self.json_ok / self.total

    def schema_rate(self) -> Optional[float]:
        return None if not self.available or not self.total else self.schema_ok / self.total

    def correct_rate(self) -> Optional[float]:
        return None if not self.available or not self.total else self.correct / self.total

    def avg_latency_ms(self) -> Optional[float]:
        return None if not self.available or not self.latencies_ms else sum(self.latencies_ms) / len(self.latencies_ms)

    def p50_latency_ms(self) -> Optional[float]:
        return None if not self.available or not self.latencies_ms else statistics.median(self.latencies_ms)


def _task_cases(task: str) -> list[dict]:
    return {
        "classify": CLASSIFY_CASES,
        "compress": COMPRESS_CASES,
        "verify": VERIFY_CASES,
    }[task]


def _task_prompt(task: str, case: dict) -> tuple[str, int]:
    if task == "classify":
        labels = ", ".join(case["labels"])
        return (
            "You are a strict classifier. Choose the single best label from the allowed set.\n"
            "Reply with ONLY a JSON object of this exact shape:\n"
            '{"label":"one of the allowed labels"}\n'
            f"Allowed labels: {labels}\n"
            f"Text: {case['text']}\n"
        ), 64
    if task == "compress":
        return (
            "Condense the text while preserving the main facts.\n"
            "Reply with ONLY a JSON object of this exact shape:\n"
            '{"text":"summary no longer than the requested character budget"}\n'
            f"Character budget: {case['max_chars']}\n"
            f"Text: {case['text']}\n"
        ), max(64, (case["max_chars"] // 3) + 24)
    if task == "verify":
        return (
            "You are a skeptical verifier. Judge whether the claim is supported by the context.\n"
            "Reply with ONLY a JSON object of this exact shape:\n"
            '{"verdict":"confirmed|refuted|uncertain","reasoning":"brief rationale"}\n'
            "Prefer uncertain when the context is incomplete.\n"
            f"Claim: {case['claim']}\n"
            f"Context: {case['context']}\n"
        ), 160
    raise ValueError(f"unknown task: {task}")


def _format_pct(part: Optional[float], num: Optional[int] = None, den: Optional[int] = None) -> str:
    if part is None:
        return "N/A"
    if num is None or den is None:
        return f"{part * 100:.0f}%"
    return f"{num}/{den} ({part * 100:.0f}%)"


def _format_ms(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.1f}"


_COMPRESS_STOPWORDS = {
    "and",
    "are",
    "but",
    "for",
    "from",
    "into",
    "just",
    "keep",
    "must",
    "that",
    "the",
    "their",
    "them",
    "then",
    "they",
    "this",
    "when",
    "with",
}


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[:=][a-z0-9]+)?", text.lower())


def _compress_retains_key_facts(obj: Optional[dict], case: dict) -> bool:
    """Proxy for compress correctness based on keyword retention, not semantic equivalence."""
    if not isinstance(obj, dict):
        return False
    text = obj.get("text")
    expected = case.get("expected")
    if not isinstance(text, str) or not isinstance(expected, str):
        return False
    expected_terms = []
    for word in _normalized_words(expected):
        if len(word) < 4 or word in _COMPRESS_STOPWORDS or word in expected_terms:
            continue
        expected_terms.append(word)
    if not expected_terms:
        return False
    summary_terms = set(_normalized_words(text))
    matched = sum(1 for term in expected_terms if term in summary_terms)
    required = max(1, (len(expected_terms) + 1) // 2)
    return matched >= required


def _validate_classify(obj: Optional[dict], case: dict) -> bool:
    if not isinstance(obj, dict):
        return False
    label = obj.get("label")
    return isinstance(label, str) and label in set(case["labels"])


def _validate_compress(obj: Optional[dict], case: dict) -> bool:
    if not isinstance(obj, dict):
        return False
    text = obj.get("text")
    return isinstance(text, str) and len(text) <= int(case["max_chars"])


def _validate_verify(obj: Optional[dict], case: dict) -> bool:
    del case
    if not isinstance(obj, dict):
        return False
    verdict = obj.get("verdict")
    return isinstance(verdict, str) and verdict in VALID_VERDICTS


def validate_schema(task: str, obj: Optional[dict], case: dict) -> bool:
    if task == "classify":
        return _validate_classify(obj, case)
    if task == "compress":
        return _validate_compress(obj, case)
    if task == "verify":
        return _validate_verify(obj, case)
    raise ValueError(f"unknown task: {task}")


def score_correctness(task: str, obj: Optional[dict], case: dict) -> bool:
    if task == "classify":
        return isinstance(obj, dict) and obj.get("label") == case.get("expected")
    if task == "compress":
        return _compress_retains_key_facts(obj, case)
    if task == "verify":
        return isinstance(obj, dict) and obj.get("verdict") == case.get("expected")
    raise ValueError(f"unknown task: {task}")


def call_ollama(base_url: str, model: str, prompt: str, *, max_tokens: int) -> CallResult:
    if not base_url:
        return CallResult(text="", latency_ms=None, transport_ok=False, error="no Ollama endpoint resolved")

    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "30m",
        "format": "json",
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.2,
        },
    }

    started = time.perf_counter()
    try:
        import requests

        response = requests.post(f"{base_url}/api/generate", json=body, timeout=TIMEOUT_S)
        response.raise_for_status()
        payload = response.json()
        latency_ms = (time.perf_counter() - started) * 1000.0
        return CallResult(text=str(payload.get("response") or ""), latency_ms=latency_ms, transport_ok=True)
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return CallResult(text="", latency_ms=latency_ms, transport_ok=False, error=str(exc)[:200])


def score_case(task: str, case: dict, call: CallResult) -> CaseScore:
    obj = extract_json_object(call.text) if call.text else None
    json_ok = obj is not None
    schema_ok = validate_schema(task, obj, case) if json_ok else False
    correct = score_correctness(task, obj, case) if json_ok else False
    return CaseScore(
        latency_ms=call.latency_ms,
        json_ok=json_ok,
        schema_ok=schema_ok,
        correct=correct,
        transport_ok=call.transport_ok,
    )


def summarize_scores(task: str, arm: str, model: str, scores: list[CaseScore], *, note: str = "") -> SummaryRow:
    if not scores:
        return SummaryRow(
            task=task,
            arm=arm,
            model=model,
            total=0,
            json_ok=0,
            schema_ok=0,
            correct=0,
            latencies_ms=[],
            available=False,
            note=note or "no cases",
        )
    available = any(score.transport_ok for score in scores)
    if not available:
        return SummaryRow(
            task=task,
            arm=arm,
            model=model,
            total=len(scores),
            json_ok=0,
            schema_ok=0,
            correct=0,
            latencies_ms=[],
            available=False,
            note=note or "endpoint unavailable",
        )
    latencies = [score.latency_ms for score in scores if score.latency_ms is not None]
    return SummaryRow(
        task=task,
        arm=arm,
        model=model,
        total=len(scores),
        json_ok=sum(1 for score in scores if score.json_ok),
        schema_ok=sum(1 for score in scores if score.schema_ok),
        correct=sum(1 for score in scores if score.correct),
        latencies_ms=[float(ms) for ms in latencies],
        available=True,
        note=note,
    )


def aggregate_rows(rows: list[SummaryRow], *, arm: str, model: str) -> SummaryRow:
    available_rows = [row for row in rows if row.available]
    if not available_rows:
        note = next((row.note for row in rows if row.note), "endpoint unavailable")
        return SummaryRow(
            task="all",
            arm=arm,
            model=model,
            total=sum(row.total for row in rows),
            json_ok=0,
            schema_ok=0,
            correct=0,
            latencies_ms=[],
            available=False,
            note=note,
        )
    return SummaryRow(
        task="all",
        arm=arm,
        model=model,
        total=sum(row.total for row in available_rows),
        json_ok=sum(row.json_ok for row in available_rows),
        schema_ok=sum(row.schema_ok for row in available_rows),
        correct=sum(row.correct for row in available_rows),
        latencies_ms=[ms for row in available_rows for ms in row.latencies_ms],
        available=True,
        note="",
    )


def evaluate_task(
    task: str,
    arm: str,
    model: str,
    *,
    base_url: str,
    call_fn: Callable[[str, str, str, int], CallResult],
) -> SummaryRow:
    scores = []
    note = ""
    for case in _task_cases(task):
        prompt, max_tokens = _task_prompt(task, case)
        call = call_fn(base_url, model, prompt, max_tokens)
        if not call.transport_ok and call.error and not note:
            note = call.error
        scores.append(score_case(task, case, call))
    return summarize_scores(task, arm, model, scores, note=note)


def format_table(rows: list[SummaryRow]) -> str:
    headers = ["task", "arm", "model", "calls", "json_ok", "schema_ok", "correct", "avg_ms", "p50_ms", "note"]
    formatted = []
    for row in rows:
        formatted.append([
            row.task,
            row.arm,
            row.model,
            str(row.total) if row.available else "N/A",
            _format_pct(row.json_rate(), row.json_ok, row.total if row.available else None),
            _format_pct(row.schema_rate(), row.schema_ok, row.total if row.available else None),
            _format_pct(row.correct_rate(), row.correct, row.total if row.available else None),
            _format_ms(row.avg_latency_ms()),
            _format_ms(row.p50_latency_ms()),
            row.note,
        ])
    widths = [len(h) for h in headers]
    for row in formatted:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def line(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells))

    divider = "  ".join("-" * width for width in widths)
    out = [line(headers), divider]
    out.extend(line(row) for row in formatted)
    return "\n".join(out)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-model", required=True, help="candidate Ollama model tag")
    ap.add_argument("--baseline-model", default=BASELINE_MODEL, help=f"baseline Ollama model tag (default: {BASELINE_MODEL})")
    ap.add_argument("--task", choices=(*TASKS, "all"), default="all", help="task to evaluate")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    selected_tasks = list(TASKS) if args.task == "all" else [args.task]
    base_url = backends.ollama_gen_url()
    rows: list[SummaryRow] = []

    for arm, model in (("baseline", args.baseline_model), ("candidate", args.candidate_model)):
        task_rows = [
            evaluate_task(
                task,
                arm,
                model,
                base_url=base_url,
                call_fn=lambda base, mdl, prompt, max_tokens: call_ollama(base, mdl, prompt, max_tokens=max_tokens),
            )
            for task in selected_tasks
        ]
        rows.extend(task_rows)
        rows.append(aggregate_rows(task_rows, arm=arm, model=model))

    print(format_table(rows))
    if not base_url:
        print("note: no Ollama generation endpoint resolved; rows reported as N/A", file=sys.stderr)
    else:
        print(f"note: endpoint {base_url}", file=sys.stderr)
    print("note: compress correctness uses a keyword-retention proxy, not full semantic equivalence.", file=sys.stderr)
    print("note: live GPU-backed execution was not validated here; run against a real Ollama host to assess model quality.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
