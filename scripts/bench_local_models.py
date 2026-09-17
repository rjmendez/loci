#!/usr/bin/env python3
"""Benchmark Ollama local-model calls with warmup, percentiles, and concurrency sweeps.

This harness is intentionally honest:
- warmup iterations are executed and discarded
- percentiles are reported from measured wall-clock timings
- TTFT is only reported when streaming is actually used
- the output records the endpoint, model, prompt shape, and model-residency snapshot
- concurrency runs report aggregate throughput versus the concurrency=1 baseline

Example:
    python3 scripts/bench_local_models.py \
      --model qwen2.5:3b \
      --concurrency 1 4 8 \
      --warmup 3 \
      --trials 20 \
      --output bench-qwen25.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

try:
    import requests
except Exception as exc:  # pragma: no cover - exercised only when runtime deps are broken
    raise SystemExit(f"error: requests is required for this harness: {exc}")

REPO_ROOT = Path(__file__).resolve().parent.parent
MCP_DIR = REPO_ROOT / "mcp"
if str(MCP_DIR) not in sys.path:
    sys.path.insert(0, str(MCP_DIR))

try:
    import backends
except Exception:
    backends = None

DEFAULT_PROMPT = (
    "Reply in one short paragraph: why does rigorous benchmark warmup matter when "
    "measuring local LLM latency and throughput?"
)
DEFAULT_TIMEOUT_S = float(os.environ.get("OLLAMA_BENCH_TIMEOUT", "180"))


@dataclass(frozen=True)
class RequestMetrics:
    total_ms: Optional[float]
    ttft_ms: Optional[float]
    output_tokens: Optional[int]
    prompt_tokens: Optional[int]
    output_chars: int
    wall_tokens_per_sec: Optional[float]
    server_eval_tokens_per_sec: Optional[float]
    transport_ok: bool
    stream_used: bool
    error: str = ""


@dataclass(frozen=True)
class TrialMetrics:
    trial_index: int
    concurrency: int
    wall_ms: float
    requests: list[RequestMetrics]


@dataclass(frozen=True)
class ResidencySnapshot:
    resident: Optional[bool]
    loaded_model_names: list[str]
    raw: Optional[dict[str, Any]]
    error: str = ""


def percentile(values: list[float], pct: float) -> Optional[float]:
    vals = sorted(float(v) for v in values)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    rank = (pct / 100.0) * (len(vals) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return vals[lo]
    frac = rank - lo
    return vals[lo] + (vals[hi] - vals[lo]) * frac


def summarize_numeric(values: list[float]) -> dict[str, Optional[float]]:
    vals = [float(v) for v in values]
    if not vals:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "stddev": None,
            "p50": None,
            "p90": None,
            "p99": None,
        }
    return {
        "count": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": statistics.fmean(vals),
        "stddev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "p50": percentile(vals, 50),
        "p90": percentile(vals, 90),
        "p99": percentile(vals, 99),
    }


def tokens_per_second(output_tokens: Optional[int], elapsed_ms: Optional[float]) -> Optional[float]:
    if output_tokens is None or elapsed_ms is None or output_tokens < 0 or elapsed_ms <= 0:
        return None
    return float(output_tokens) / (float(elapsed_ms) / 1000.0)


def _iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _default_base_url() -> str:
    if os.environ.get("LOCI_OLLAMA_GEN_URL"):
        return os.environ["LOCI_OLLAMA_GEN_URL"]
    if os.environ.get("OLLAMA_GEN_URL"):
        return os.environ["OLLAMA_GEN_URL"]
    if os.environ.get("OLLAMA_URL"):
        return os.environ["OLLAMA_URL"]
    if backends is not None:
        try:
            return backends.ollama_gen_url()
        except Exception:
            pass
    return "http://localhost:11434"


def _default_model() -> str:
    if os.environ.get("LOCI_OLLAMA_GEN_MODEL"):
        return os.environ["LOCI_OLLAMA_GEN_MODEL"]
    if backends is not None:
        try:
            return backends.ollama_gen_model()
        except Exception:
            pass
    return "qwen2.5:3b"


def _load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text().strip()
    return args.prompt.strip()


def _request_body(*, model: str, prompt: str, max_tokens: int, temperature: float, keep_alive: str, stream: bool) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "keep_alive": keep_alive,
        "think": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        },
    }


def _sample_residency(base_url: str, model: str, timeout_s: float) -> ResidencySnapshot:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/ps", timeout=timeout_s)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        return ResidencySnapshot(resident=None, loaded_model_names=[], raw=None, error=str(exc)[:300])
    models = payload.get("models") if isinstance(payload, dict) else []
    names = [str(entry.get("name") or "") for entry in models if isinstance(entry, dict)]
    resident = model in names
    return ResidencySnapshot(resident=resident, loaded_model_names=[n for n in names if n], raw=payload)


def _measure_streaming_request(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    keep_alive: str,
    timeout_s: float,
) -> RequestMetrics:
    started = time.perf_counter()
    ttft_ms: Optional[float] = None
    output_chars = 0
    output_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    server_eval_tokens_per_sec: Optional[float] = None
    try:
        body = _request_body(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            keep_alive=keep_alive,
            stream=True,
        )
        with requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json=body,
            stream=True,
            timeout=(10, timeout_s),
        ) as response:
            response.raise_for_status()
            final_payload: Optional[dict[str, Any]] = None
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                payload = json.loads(raw_line)
                piece = payload.get("response")
                if isinstance(piece, str) and piece:
                    output_chars += len(piece)
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - started) * 1000.0
                if payload.get("done"):
                    final_payload = payload
            if final_payload is None:
                raise RuntimeError("stream ended without a done frame")
        total_ms = (time.perf_counter() - started) * 1000.0
        output_tokens = final_payload.get("eval_count") if isinstance(final_payload.get("eval_count"), int) else None
        prompt_tokens = final_payload.get("prompt_eval_count") if isinstance(final_payload.get("prompt_eval_count"), int) else None
        eval_duration = final_payload.get("eval_duration")
        if isinstance(eval_duration, int) and eval_duration > 0 and output_tokens is not None:
            server_eval_tokens_per_sec = output_tokens / (eval_duration / 1_000_000_000.0)
        return RequestMetrics(
            total_ms=total_ms,
            ttft_ms=ttft_ms,
            output_tokens=output_tokens,
            prompt_tokens=prompt_tokens,
            output_chars=output_chars,
            wall_tokens_per_sec=tokens_per_second(output_tokens, total_ms),
            server_eval_tokens_per_sec=server_eval_tokens_per_sec,
            transport_ok=True,
            stream_used=True,
        )
    except Exception as exc:
        total_ms = (time.perf_counter() - started) * 1000.0
        return RequestMetrics(
            total_ms=total_ms,
            ttft_ms=ttft_ms,
            output_tokens=output_tokens,
            prompt_tokens=prompt_tokens,
            output_chars=output_chars,
            wall_tokens_per_sec=tokens_per_second(output_tokens, total_ms),
            server_eval_tokens_per_sec=server_eval_tokens_per_sec,
            transport_ok=False,
            stream_used=True,
            error=str(exc)[:300],
        )


def _measure_blocking_request(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    keep_alive: str,
    timeout_s: float,
) -> RequestMetrics:
    started = time.perf_counter()
    try:
        body = _request_body(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            keep_alive=keep_alive,
            stream=False,
        )
        response = requests.post(
            f"{base_url.rstrip('/')}/api/generate",
            json=body,
            timeout=timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        total_ms = (time.perf_counter() - started) * 1000.0
        text = str(payload.get("response") or "")
        output_tokens = payload.get("eval_count") if isinstance(payload.get("eval_count"), int) else None
        prompt_tokens = payload.get("prompt_eval_count") if isinstance(payload.get("prompt_eval_count"), int) else None
        eval_duration = payload.get("eval_duration")
        server_eval_tokens_per_sec = None
        if isinstance(eval_duration, int) and eval_duration > 0 and output_tokens is not None:
            server_eval_tokens_per_sec = output_tokens / (eval_duration / 1_000_000_000.0)
        return RequestMetrics(
            total_ms=total_ms,
            ttft_ms=None,
            output_tokens=output_tokens,
            prompt_tokens=prompt_tokens,
            output_chars=len(text),
            wall_tokens_per_sec=tokens_per_second(output_tokens, total_ms),
            server_eval_tokens_per_sec=server_eval_tokens_per_sec,
            transport_ok=True,
            stream_used=False,
        )
    except Exception as exc:
        total_ms = (time.perf_counter() - started) * 1000.0
        return RequestMetrics(
            total_ms=total_ms,
            ttft_ms=None,
            output_tokens=None,
            prompt_tokens=None,
            output_chars=0,
            wall_tokens_per_sec=None,
            server_eval_tokens_per_sec=None,
            transport_ok=False,
            stream_used=False,
            error=str(exc)[:300],
        )


def _run_trial(
    *,
    trial_index: int,
    concurrency: int,
    request_fn,
) -> TrialMetrics:
    barrier = threading.Barrier(concurrency)

    def call_one() -> RequestMetrics:
        barrier.wait()
        return request_fn()

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(call_one) for _ in range(concurrency)]
        results = [future.result() for future in futures]
    wall_ms = (time.perf_counter() - started) * 1000.0
    return TrialMetrics(trial_index=trial_index, concurrency=concurrency, wall_ms=wall_ms, requests=results)


def summarize_suite(
    *,
    concurrency: int,
    warmup_iterations: int,
    trials: list[TrialMetrics],
    resident_before: ResidencySnapshot,
    resident_after: ResidencySnapshot,
    stream_requested: bool,
) -> dict[str, Any]:
    flat_requests = [req for trial in trials for req in trial.requests]
    total_ms = [req.total_ms for req in flat_requests if req.total_ms is not None]
    ttft_ms = [req.ttft_ms for req in flat_requests if req.ttft_ms is not None]
    wall_tps = [req.wall_tokens_per_sec for req in flat_requests if req.wall_tokens_per_sec is not None]
    server_tps = [req.server_eval_tokens_per_sec for req in flat_requests if req.server_eval_tokens_per_sec is not None]
    prompt_tokens = [float(req.prompt_tokens) for req in flat_requests if req.prompt_tokens is not None]
    output_tokens = [float(req.output_tokens) for req in flat_requests if req.output_tokens is not None]
    wall_ms = [trial.wall_ms for trial in trials]
    aggregate_req_per_s = [trial.concurrency / (trial.wall_ms / 1000.0) for trial in trials if trial.wall_ms > 0]
    aggregate_tok_per_s = [
        (sum(req.output_tokens or 0 for req in trial.requests) / (trial.wall_ms / 1000.0))
        for trial in trials
        if trial.wall_ms > 0
    ]
    errors = sorted({req.error for req in flat_requests if req.error})
    if stream_requested:
        ttft_note = (
            "measured from first non-empty streamed response chunk"
            if ttft_ms
            else "streaming requested, but no first-token chunk was observed"
        )
    else:
        ttft_note = "streaming disabled, so TTFT is unavailable by design"

    return {
        "concurrency": concurrency,
        "warmup_iterations": warmup_iterations,
        "trial_count": len(trials),
        "request_count": len(flat_requests),
        "transport_failures": sum(1 for req in flat_requests if not req.transport_ok),
        "streaming": {
            "requested": stream_requested,
            "ttft_measured": bool(ttft_ms),
            "ttft_note": ttft_note,
        },
        "resident_before": resident_before.resident,
        "resident_after": resident_after.resident,
        "loaded_models_before": resident_before.loaded_model_names,
        "loaded_models_after": resident_after.loaded_model_names,
        "residency_probe_error_before": resident_before.error,
        "residency_probe_error_after": resident_after.error,
        "latency_ms": {
            "ttft": summarize_numeric(ttft_ms),
            "total": summarize_numeric(total_ms),
            "trial_wall": summarize_numeric(wall_ms),
        },
        "tokens": {
            "prompt_eval_count": summarize_numeric(prompt_tokens),
            "output_eval_count": summarize_numeric(output_tokens),
            "per_request_wall_tokens_per_sec": summarize_numeric(wall_tps),
            "per_request_server_eval_tokens_per_sec": summarize_numeric(server_tps),
            "aggregate_tokens_per_sec": summarize_numeric(aggregate_tok_per_s),
        },
        "aggregate_requests_per_sec": summarize_numeric(aggregate_req_per_s),
        "notes": errors,
        "trial_details": [
            {
                "trial_index": trial.trial_index,
                "concurrency": trial.concurrency,
                "wall_ms": trial.wall_ms,
                "aggregate_requests_per_sec": (trial.concurrency / (trial.wall_ms / 1000.0)) if trial.wall_ms > 0 else None,
                "aggregate_tokens_per_sec": (
                    sum(req.output_tokens or 0 for req in trial.requests) / (trial.wall_ms / 1000.0)
                ) if trial.wall_ms > 0 else None,
                "requests": [asdict(req) for req in trial.requests],
            }
            for trial in trials
        ],
    }


def attach_speedups(results: list[dict[str, Any]]) -> None:
    baseline = next(
        (
            row["tokens"]["aggregate_tokens_per_sec"]["mean"]
            for row in results
            if row.get("concurrency") == 1
        ),
        None,
    )
    for row in results:
        mean_tps = row["tokens"]["aggregate_tokens_per_sec"]["mean"]
        row["tokens"]["aggregate_tokens_per_sec"]["speedup_vs_concurrency_1"] = (
            (mean_tps / baseline) if baseline and mean_tps is not None else None
        )


def format_summary_table(results: list[dict[str, Any]]) -> str:
    headers = [
        "conc",
        "reqs",
        "resident_before",
        "ttft_p50_ms",
        "total_p50_ms",
        "agg_tok_s_mean",
        "speedup_vs_1",
        "agg_req_s_mean",
        "failures",
        "notes",
    ]
    rows: list[list[str]] = []
    for row in results:
        ttft = row["latency_ms"]["ttft"]["p50"]
        total = row["latency_ms"]["total"]["p50"]
        agg_tps = row["tokens"]["aggregate_tokens_per_sec"]["mean"]
        speedup = row["tokens"]["aggregate_tokens_per_sec"].get("speedup_vs_concurrency_1")
        agg_rps = row["aggregate_requests_per_sec"]["mean"]
        rows.append([
            str(row["concurrency"]),
            str(row["request_count"]),
            "yes" if row["resident_before"] else "no" if row["resident_before"] is False else "unknown",
            f"{ttft:.1f}" if ttft is not None else "N/A",
            f"{total:.1f}" if total is not None else "N/A",
            f"{agg_tps:.2f}" if agg_tps is not None else "N/A",
            f"{speedup:.2f}x" if speedup is not None else "N/A",
            f"{agg_rps:.2f}" if agg_rps is not None else "N/A",
            str(row["transport_failures"]),
            row["streaming"]["ttft_note"] if not row["streaming"]["ttft_measured"] else "; ".join(row["notes"][:1]),
        ])
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(cells))

    divider = "  ".join("-" * width for width in widths)
    out = [fmt(headers), divider]
    out.extend(fmt(row) for row in rows)
    return "\n".join(out)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=_default_base_url(), help="Ollama base URL (default: resolved from env/backends)")
    ap.add_argument("--model", default=_default_model(), help="Ollama model tag")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="prompt text to send")
    ap.add_argument("--prompt-file", help="read prompt text from a file instead of --prompt")
    ap.add_argument("--max-tokens", type=int, default=64, help="num_predict per request")
    ap.add_argument("--temperature", type=float, default=0.0, help="sampling temperature")
    ap.add_argument("--keep-alive", default="30m", help="Ollama keep_alive request field")
    ap.add_argument("--warmup", type=int, default=5, help="discarded warmup iterations per concurrency level")
    ap.add_argument("--trials", type=int, default=300, help="measured trials per concurrency level")
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1], help="one or more concurrency levels to benchmark")
    ap.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S, help="request timeout in seconds")
    ap.add_argument("--no-stream", action="store_true", help="disable streaming; TTFT will be reported unavailable")
    ap.add_argument("--output", required=True, help="write stable JSON results to this file")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if any(c <= 0 for c in args.concurrency):
        raise SystemExit("error: all --concurrency values must be >= 1")
    if args.warmup < 0 or args.trials <= 0:
        raise SystemExit("error: --warmup must be >= 0 and --trials must be > 0")

    prompt = _load_prompt(args)
    if not prompt:
        raise SystemExit("error: prompt is empty")
    stream = not args.no_stream

    started_at = _iso_now()
    overall_residency = _sample_residency(args.base_url, args.model, args.timeout_s)

    def request_fn() -> RequestMetrics:
        measure = _measure_streaming_request if stream else _measure_blocking_request
        return measure(
            base_url=args.base_url,
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            keep_alive=args.keep_alive,
            timeout_s=args.timeout_s,
        )

    results: list[dict[str, Any]] = []
    for concurrency in args.concurrency:
        resident_before = _sample_residency(args.base_url, args.model, args.timeout_s)
        for _ in range(args.warmup):
            _run_trial(trial_index=-1, concurrency=concurrency, request_fn=request_fn)
        trials = [
            _run_trial(trial_index=idx, concurrency=concurrency, request_fn=request_fn)
            for idx in range(args.trials)
        ]
        resident_after = _sample_residency(args.base_url, args.model, args.timeout_s)
        results.append(
            summarize_suite(
                concurrency=concurrency,
                warmup_iterations=args.warmup,
                trials=trials,
                resident_before=resident_before,
                resident_after=resident_after,
                stream_requested=stream,
            )
        )

    attach_speedups(results)

    payload = {
        "schema_version": 1,
        "started_at": started_at,
        "completed_at": _iso_now(),
        "environment": {
            "base_url": args.base_url,
            "host": urlparse(args.base_url).netloc or args.base_url,
            "model": args.model,
            "prompt": prompt,
            "prompt_chars": len(prompt),
            "prompt_words": len(prompt.split()),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "keep_alive": args.keep_alive,
            "timeout_s": args.timeout_s,
            "stream_requested": stream,
            "warmup_iterations": args.warmup,
            "trials": args.trials,
            "concurrency": args.concurrency,
            "resident_before_run": overall_residency.resident,
            "loaded_models_before_run": overall_residency.loaded_model_names,
            "residency_probe_error_before_run": overall_residency.error,
        },
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"Benchmark results written to {output_path}")
    print(format_summary_table(results))
    print(f"endpoint={args.base_url} model={args.model} prompt_chars={len(prompt)} resident_before_run={overall_residency.resident}")
    if overall_residency.loaded_model_names:
        print("loaded_models_before_run=" + ", ".join(overall_residency.loaded_model_names))
    if overall_residency.error:
        print(f"note: residency probe before run failed: {overall_residency.error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
