"""Deterministic, schema-aware compact renderers for hot-path MCP read tools."""
from __future__ import annotations

import re
from typing import Callable, Optional

from untrusted_memory import _FRAME_TAG_RE, neutralize_frame_tags

TRUNCATION_MARKER = " …[truncated]"
_WS_RE = re.compile(r"\s+")
_UNTRUSTED_RE = re.compile(
    r"(?s)^(?P<open><untrusted_memory_content\b[^>]*>)\s*(?P<body>.*?)\s*(?P<close></untrusted_memory_content>)$"
)
# One well-formed frame whose body holds no frame tag, found anywhere in a string.
# Only used for text Loci composed itself from wrap_untrusted_memory_text output.
_CLEAN_FRAME_RE = re.compile(
    r"(?s)<untrusted_memory_content(?:\s[^<>]*)?>"
    r"(?:(?!<\s*/?\s*(?i:untrusted_memory_content)).)*?"
    r"</untrusted_memory_content>"
)


def _single_line(text: str) -> str:
    return _WS_RE.sub(" ", str(text or "").strip())


def _clip_plain_text(
    text: str,
    max_chars: int,
    *,
    preserve_sentence_boundary: bool = True,
) -> str:
    body = str(text or "").strip()
    if max_chars <= 0:
        return ""
    if len(body) <= max_chars:
        return body
    budget = max(1, max_chars - len(TRUNCATION_MARKER))
    cut = body[:budget]
    if preserve_sentence_boundary:
        for sep in ("\n", ". ", "; ", ", ", " "):
            idx = cut.rfind(sep)
            if idx > budget * 0.5:
                cut = cut[: idx + (1 if sep == "\n" else 0)]
                break
    cut = cut.rstrip(" \n\t,;:-")
    return (cut or body[:budget].rstrip()) + TRUNCATION_MARKER


def _compact_composed(body: str, max_chars: int, preserve_sentence_boundary: bool) -> str:
    """Clip Loci-composed text made of clean frames and Loci's own plain text.

    Clean frames are kept whole (or clipped inside their tags); every frame tag in
    the text between them is escaped. The result never ends inside an open frame.
    """
    pieces: list[tuple[bool, str]] = []
    pos = 0
    for fm in _CLEAN_FRAME_RE.finditer(body):
        if fm.start() > pos:
            pieces.append((False, neutralize_frame_tags(body[pos:fm.start()])))
        pieces.append((True, fm.group(0)))
        pos = fm.end()
    if pos < len(body):
        pieces.append((False, neutralize_frame_tags(body[pos:])))
    full = "".join(t for _, t in pieces)
    if len(full) <= max_chars:
        return full
    out = ""
    for is_frame, piece in pieces:
        if len(out) + len(piece) + len(TRUNCATION_MARKER) <= max_chars:
            out += piece
            continue
        room = max_chars - len(out)
        if is_frame:
            m = _UNTRUSTED_RE.match(piece)
            shell = (len(m.group("open")) + len(m.group("close")) + 2 + len(TRUNCATION_MARKER)) if m else 0
            if m and room >= shell + 16:
                return out + compact_text(piece, room, preserve_sentence_boundary)
        elif room > len(TRUNCATION_MARKER) + 8:
            return out + _clip_plain_text(piece, room, preserve_sentence_boundary=preserve_sentence_boundary)
        break
    return out.rstrip() + TRUNCATION_MARKER


def compact_text(
    text: str,
    max_chars: int,
    preserve_sentence_boundary: bool = True,
    *,
    keep_frames: bool = False,
) -> str:
    """Clip text deterministically and preserve untrusted-memory framing when present.

    ``keep_frames=True`` is for text Loci composed itself out of
    wrap_untrusted_memory_text frames (a multi-row RAG context, a
    "[superseded: ...] <frame>" line): every clean frame in it stays a real frame.
    Without it, only a text that is exactly one clean frame keeps its tags; any
    other frame tag is treated as stored content and escaped.
    """
    body = str(text or "").strip()
    if max_chars <= 0:
        return ""
    m = _UNTRUSTED_RE.match(body)
    if m and _FRAME_TAG_RE.search(m.group("body")):
        # A frame with raw frame tags inside is not one Loci emitted (bodies are
        # always neutralised): keeping its open tag would keep forged attributes
        # and let the inner close tag end the frame early.
        m = None
    if not m and keep_frames:
        return _compact_composed(body, max_chars, preserve_sentence_boundary)
    if not m:
        return _clip_plain_text(
            neutralize_frame_tags(body), max_chars, preserve_sentence_boundary=preserve_sentence_boundary
        )
    open_tag = m.group("open")
    close_tag = m.group("close")
    shell = len(open_tag) + len(close_tag) + 1
    inner_budget = max(1, max_chars - shell)
    inner = _clip_plain_text(
        m.group("body"),
        inner_budget,
        preserve_sentence_boundary=preserve_sentence_boundary,
    )
    return f"{open_tag} {inner} {close_tag}"


def compact_finding_row(row: dict, text_chars: int = 140) -> dict:
    """Return a shallow-copied finding row with only its text compacted."""
    out = dict(row or {})
    out["text"] = compact_text(str(out.get("text") or ""), text_chars)
    return out


def compact_sources(sources: list[dict]) -> list[dict]:
    """Reduce sources to the must-keep fields for compact-mode reloads."""
    out: list[dict] = []
    for i, src in enumerate(sources or [], 1):
        row = dict(src or {})
        out.append({
            "n": row.get("n", i),
            "id": str(row.get("id") or row.get("finding_id") or row.get("memory_id") or ""),
            "investigation_id": str(row.get("investigation_id") or ""),
            "origin": str(row.get("origin") or row.get("collection") or row.get("source") or ""),
            "score": round(float(row.get("score") or row.get("relevance_score") or 0.0), 4),
        })
    return out


def _row_line(
    row: dict,
    index: int,
    *,
    max_chars: int,
    keep_scores: bool = True,
    keep_ids: bool = True,
    wrap_text: Optional[Callable[[str, dict], str]] = None,
) -> str:
    payload = dict(row or {})
    text = str(payload.get("text") or payload.get("content") or payload.get("finding") or "")
    if wrap_text is not None and text:
        text = wrap_text(text, payload)
    parts = [str(index)]
    row_id = str(payload.get("finding_id") or payload.get("memory_id") or payload.get("id") or "")
    if keep_ids and row_id:
        parts.append(f"id={row_id}")
    inv = str(payload.get("investigation_id") or "")
    if keep_ids and inv:
        parts.append(f"inv={inv}")
    if keep_scores:
        score = payload.get("score", payload.get("relevance_score"))
        if score is not None:
            try:
                parts.append(f"score={float(score):.3f}")
            except Exception:
                pass
    source = str(payload.get("source") or payload.get("origin") or payload.get("collection") or "")
    if source:
        parts.append(f"src={source}")
    origin = str(payload.get("origin") or payload.get("collection") or "")
    if origin and origin != source:
        parts.append(f"origin={origin}")
    path = str(payload.get("path") or payload.get("file") or "")
    if path:
        parts.append(f"path={path}")
    line = payload.get("line")
    if line not in (None, ""):
        parts.append(f"line={line}")
    else:
        start = payload.get("start_line", payload.get("line_start"))
        end = payload.get("end_line", payload.get("line_end"))
        if start not in (None, "") and end not in (None, ""):
            parts.append(f"line={start}-{end}")
        elif start not in (None, ""):
            parts.append(f"line={start}")
    symbol = str(payload.get("symbol") or payload.get("symbol_name") or payload.get("name") or "")
    if symbol:
        parts.append(f"symbol={symbol}")
    kind = str(payload.get("kind") or "")
    if kind:
        parts.append(f"kind={kind}")
    confidence = payload.get("confidence")
    if confidence not in (None, ""):
        parts.append(f"confidence={confidence}")
    resolution = str(payload.get("resolution") or "")
    if resolution:
        parts.append(f"resolution={resolution}")
    if payload.get("contested") is True:
        parts.append("state=contested")
    elif payload.get("converged") is True:
        parts.append("state=converged")
    prefix = "[" + " ".join(parts) + "]"
    if not text:
        return prefix
    text_budget = max(24, max_chars - len(prefix) - 1)
    compacted = compact_text(_single_line(text), text_budget, preserve_sentence_boundary=False)
    return f"{prefix} {compacted}"


def compact_context_rows(
    rows: list[dict],
    budget_chars: int,
    keep_scores: bool = True,
    keep_ids: bool = True,
    wrap_text: Optional[Callable[[str, dict], str]] = None,
) -> dict:
    """Flatten context rows to deterministic cited one-liners within a char budget."""
    lines: list[str] = []
    total = 0
    for i, row in enumerate(rows or [], 1):
        remaining = max(1, budget_chars - total)
        line = _row_line(
            row,
            i,
            max_chars=remaining,
            keep_scores=keep_scores,
            keep_ids=keep_ids,
            wrap_text=wrap_text,
        )
        extra = len(line) + (1 if lines else 0)
        if total + extra > budget_chars and i > 1:
            omitted = f"[{len(rows) - i + 1} more results omitted — budget {budget_chars} chars]"
            if total + len(omitted) + 1 <= budget_chars:
                lines.append(omitted)
            return {
                "context": "\n".join(lines),
                "total_chars": total,
                "truncated": True,
                "result_count": len(rows),
            }
        lines.append(line)
        total += extra
    return {
        "context": "\n".join(lines),
        "total_chars": total,
        "truncated": False,
        "result_count": len(rows),
    }
