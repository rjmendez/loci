#!/usr/bin/env python3
"""Prefilter bulky agent output with the local-model tier before cloud synthesis.

This is opt-in tooling for workflows that fan out to research/explore/build agents,
collect large result blobs, then hand them to an expensive cloud model for the final
answer. The prefilter keeps the decision with the caller:

  - it ALWAYS preserves the original text unchanged in the returned payload
  - it composes existing local-model primitives from mcp/llm_tools.py
  - it fails open: any local-tier error or low-confidence route returns the original

Usage:
  agent_output_prefilter.py --context "summarize the build failures" [path]

If `path` is omitted or `-`, input is read from stdin. Output is JSON:
  {compressed_text, original_text, confidence, coverage, fallback_used, ...}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from importlib import util as importlib_util
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp"))

import llm_tools  # noqa: E402
from model_json import extract_json_object  # noqa: E402

logger = logging.getLogger("loci-mcp.agent_output_prefilter")
_MCP_TEXT_OPS_PATH = Path(__file__).resolve().parents[1] / "mcp" / "text_ops.py"


def _load_mcp_text_ops():
    spec = importlib_util.spec_from_file_location("_loci_mcp_text_ops", _MCP_TEXT_OPS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {_MCP_TEXT_OPS_PATH}")
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mcp_text_ops = _load_mcp_text_ops()


@dataclass
class PrefilterResult:
    compressed_text: str
    original_text: str
    confidence: float
    coverage: float
    fallback_used: bool
    fallback_reason: str = ""
    selected_chunks: int = 0
    total_chunks: int = 0


def _clamp_01(value: object, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return default


def _chunk_text(text: str, *, chunk_chars: int) -> list[str]:
    text = text if isinstance(text, str) else ("" if text is None else str(text))
    try:
        chunk_chars = int(chunk_chars)
    except Exception:
        chunk_chars = 1200
    chunk_chars = max(200, chunk_chars)
    if not text:
        return []

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        if len(paragraph) > chunk_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            for start in range(0, len(paragraph), chunk_chars):
                piece = paragraph[start:start + chunk_chars].strip()
                if piece:
                    chunks.append(piece)
            continue

        extra = len(paragraph) + (2 if current else 0)
        if current and current_len + extra > chunk_chars:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_len = len(paragraph)
            continue
        current.append(paragraph)
        current_len += extra

    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text]


def _tool_json(raw: str, tool_name: str) -> Optional[dict]:
    obj = extract_json_object(raw or "")
    if isinstance(obj, dict):
        return obj
    logger.warning("%s returned an unparseable payload — returning original text unchanged", tool_name)
    return None


def _fallback(text: str, *, confidence: float, coverage: float, reason: str,
              selected_chunks: int, total_chunks: int) -> PrefilterResult:
    logger.warning("agent_output_prefilter degraded: %s — returning original text unchanged", reason)
    return PrefilterResult(
        compressed_text=text,
        original_text=text,
        confidence=_clamp_01(confidence),
        coverage=_clamp_01(coverage),
        fallback_used=True,
        fallback_reason=reason,
        selected_chunks=selected_chunks,
        total_chunks=total_chunks,
    )


def prefilter_agent_output(
    text: str,
    task_context: str,
    *,
    threshold: float = 0.6,
    chunk_chars: int = 1200,
    max_chars: int = 1600,
    selection_floor: float = 0.35,
    selection_ratio: float = 0.75,
    dedup_threshold: float = 0.9,
) -> PrefilterResult:
    """Return a compressed, relevance-gated view of `text`, preserving the original.

    Pipeline:
      1. split the agent output into chunks
      2. use semantic_relevance(topic=task_context) to keep the chunks that best match
      3. use semantic_dedup() to collapse repeated chunks when embeddings are available
      4. use compress_text() on the kept span

    Any local-tier error/degradation, or a confidence score below `threshold`, fails
    open to the original text with `fallback_used=True`.
    """
    text = text if isinstance(text, str) else ("" if text is None else str(text))
    task_context = task_context if isinstance(task_context, str) else ("" if task_context is None else str(task_context))
    threshold = _clamp_01(threshold, default=0.6)
    selection_floor = _clamp_01(selection_floor, default=0.35)
    selection_ratio = _clamp_01(selection_ratio, default=0.75)
    dedup_threshold = _clamp_01(dedup_threshold, default=0.9)
    try:
        max_chars = max(1, int(max_chars))
    except Exception:
        max_chars = 1600

    if not text:
        return PrefilterResult(
            compressed_text="",
            original_text="",
            confidence=1.0,
            coverage=1.0,
            fallback_used=False,
            selected_chunks=0,
            total_chunks=0,
        )
    if len(text) <= max_chars:
        return PrefilterResult(
            compressed_text=text,
            original_text=text,
            confidence=1.0,
            coverage=1.0,
            fallback_used=False,
            selected_chunks=1,
            total_chunks=1,
        )
    if not task_context.strip():
        return _fallback(
            text,
            confidence=0.0,
            coverage=1.0,
            reason="task context must not be empty",
            selected_chunks=0,
            total_chunks=0,
        )

    chunks = _chunk_text(text, chunk_chars=chunk_chars)
    total_chunks = len(chunks)
    if not chunks:
        return _fallback(
            text,
            confidence=0.0,
            coverage=0.0,
            reason="no chunks produced from input text",
            selected_chunks=0,
            total_chunks=0,
        )

    try:
        relevance_obj = _tool_json(
            llm_tools.semantic_relevance(chunks, task_context),
            "semantic_relevance",
        )
    except Exception as exc:
        return _fallback(
            text,
            confidence=0.0,
            coverage=1.0,
            reason=f"semantic_relevance raised {type(exc).__name__}: {exc}",
            selected_chunks=0,
            total_chunks=total_chunks,
        )
    if not relevance_obj or relevance_obj.get("degraded"):
        return _fallback(
            text,
            confidence=0.0,
            coverage=1.0,
            reason="semantic_relevance degraded or returned no usable scores",
            selected_chunks=0,
            total_chunks=total_chunks,
        )

    raw_scores = relevance_obj.get("scores") or []
    if len(raw_scores) != total_chunks:
        return _fallback(
            text,
            confidence=0.0,
            coverage=1.0,
            reason="semantic_relevance score count did not match chunk count",
            selected_chunks=0,
            total_chunks=total_chunks,
        )

    numeric_scores = [
        _clamp_01(score) if isinstance(score, (int, float)) else None for score in raw_scores
    ]
    usable_scores = [score for score in numeric_scores if score is not None]
    if not usable_scores:
        return _fallback(
            text,
            confidence=0.0,
            coverage=1.0,
            reason="semantic_relevance returned no numeric scores",
            selected_chunks=0,
            total_chunks=total_chunks,
        )

    best_score = max(usable_scores)
    cutoff = max(selection_floor, best_score * selection_ratio)
    selected_pairs = [
        (chunk, score) for chunk, score in zip(chunks, numeric_scores)
        if score is not None and score >= cutoff
    ]
    if not selected_pairs:
        return _fallback(
            text,
            confidence=best_score,
            coverage=0.0,
            reason="no chunks cleared the semantic relevance cutoff",
            selected_chunks=0,
            total_chunks=total_chunks,
        )

    selected_chunks = len(selected_pairs)
    selected_texts = [chunk for chunk, _ in selected_pairs]
    selected_scores = [float(score) for _, score in selected_pairs]
    confidence = sum(selected_scores) / len(selected_scores)
    coverage = sum(len(chunk) for chunk in selected_texts) / max(1, len(text))
    if confidence < threshold:
        return _fallback(
            text,
            confidence=confidence,
            coverage=coverage,
            reason=f"confidence {confidence:.3f} below threshold {threshold:.3f}",
            selected_chunks=selected_chunks,
            total_chunks=total_chunks,
        )

    deduped_texts = selected_texts
    if len(selected_texts) > 1:
        try:
            dedup_obj = _tool_json(
                llm_tools.semantic_dedup(selected_texts, threshold=dedup_threshold),
                "semantic_dedup",
            )
        except Exception as exc:
            logger.warning(
                "semantic_dedup raised %s: %s — keeping selected chunks unchanged",
                type(exc).__name__,
                exc,
            )
            dedup_obj = None
        kept = dedup_obj.get("kept") if isinstance(dedup_obj, dict) else None
        if isinstance(kept, list) and kept:
            deduped_texts = [item if isinstance(item, str) else str(item) for item in kept]

    try:
        compress_obj = _mcp_text_ops.compress("\n\n".join(deduped_texts), max_chars=max_chars)
    except Exception as exc:
        return _fallback(
            text,
            confidence=confidence,
            coverage=coverage,
            reason=f"compress_text raised {type(exc).__name__}: {exc}",
            selected_chunks=selected_chunks,
            total_chunks=total_chunks,
        )
    if not compress_obj or compress_obj.get("degraded"):
        return _fallback(
            text,
            confidence=confidence,
            coverage=coverage,
            reason="compress_text degraded or returned no usable text",
            selected_chunks=selected_chunks,
            total_chunks=total_chunks,
        )

    compressed_text = compress_obj.get("text")
    if not isinstance(compressed_text, str) or not compressed_text.strip():
        return _fallback(
            text,
            confidence=confidence,
            coverage=coverage,
            reason="compress_text returned empty text",
            selected_chunks=selected_chunks,
            total_chunks=total_chunks,
        )

    return PrefilterResult(
        compressed_text=compressed_text,
        original_text=text,
        confidence=_clamp_01(confidence),
        coverage=_clamp_01(coverage, default=1.0),
        fallback_used=False,
        selected_chunks=selected_chunks,
        total_chunks=total_chunks,
    )


def _read_text(path: Optional[str]) -> str:
    if not path or path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", default="-", help="input file path, or '-' / omitted for stdin")
    parser.add_argument("--context", required=True, help="short task/query context used to score relevant chunks")
    parser.add_argument("--threshold", type=float, default=0.6, help="minimum confidence required to trust compressed output")
    parser.add_argument("--chunk-chars", type=int, default=1200, help="chunk size for semantic relevance scoring")
    parser.add_argument("--max-chars", type=int, default=1600, help="target character budget for compressed output")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        text = _read_text(args.path)
    except OSError as exc:
        print(f"agent_output_prefilter: could not read input: {exc}", file=sys.stderr)
        return 2

    result = prefilter_agent_output(
        text,
        args.context,
        threshold=args.threshold,
        chunk_chars=args.chunk_chars,
        max_chars=args.max_chars,
    )
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
