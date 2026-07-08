"""Grounding-injection for Loci-native workflows.

`ground(task)` runs ONCE in the main loop before a fan-out and returns a compact,
provenance-tagged, char-budgeted context block to inject into every agent prompt —
so agents start with relevant prior knowledge instead of rediscovering it, and never
each hit Loci themselves (cost).

Design (from the Loci self-review plan):
- STRUCTURED-FIRST, embedding-independent retrieval is the quality path: curated
  memory files -> investigation_load(named case) -> investigation_entity_lookup(exact
  IDs) -> code graph (when available). Semantic RAG and filtered keyword recall are
  optional enhancements that no-op when down (degraded=True) rather than blocking.
- filter_noise() drops pre_compress/session_end conversation-dump blobs that otherwise
  pollute keyword/FTS recall.
- Budget cascade: unused budget from an empty/skipped source rolls to the next.
- Fail-open everywhere: a dead source never aborts grounding.

All Loci calls are lazy-imported from `server` so importing this module is cheap and
does not create an import cycle.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

_MEMORY_DIR_DEFAULT = os.environ.get(
    "LOCI_MEMORY_MD_DIR",
    str(Path.home() / ".claude" / "projects" / "-home-rjmendez" / "memory"),
)

# Sources/types whose text is a raw conversation dump — never inject these.
_DUMP_MARKERS = ("pre_compress", "session_end", "session_dump", "conversation", "transcript", "turn")
_ROLE_RE = re.compile(r'\b(user|assistant|system)\b\s*[:"]', re.IGNORECASE)
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def _looks_like_dump(text: str) -> bool:
    """Heuristic: long, low-diversity text with role markers = a conversation dump."""
    if not text:
        return False
    if len(text) > 1500:
        toks = _TOKEN_RE.findall(text)
        if toks:
            uniq_ratio = len(set(t.lower() for t in toks)) / len(toks)
            if _ROLE_RE.search(text) or uniq_ratio < 0.35:
                return True
    return bool(_ROLE_RE.search(text)) and len(text) > 800


def filter_noise(items: list[dict]) -> list[dict]:
    """Drop conversation-dump blobs + duplicates from fuzzy (keyword/FTS) results.

    Each item: {text, source?, type?, tags?, score?}. Curated/structured sources
    should NOT be passed through this — it is only for the noisy recall lanes.
    """
    out: list[dict] = []
    seen: set = set()
    for it in items or []:
        text = str(it.get("text") or it.get("content") or "")
        if not text.strip():
            continue
        meta = " ".join(str(it.get(k) or "") for k in ("source", "type", "tags", "origin")).lower()
        if any(m in meta for m in _DUMP_MARKERS):
            continue
        if _looks_like_dump(text):
            continue
        h = hash(re.sub(r"\s+", " ", text.strip().lower())[:200])
        if h in seen:
            continue
        seen.add(h)
        out.append(it)
    return out


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    # back off to a sentence/line boundary
    for sep in ("\n", ". ", "; ", ", ", " "):
        i = cut.rfind(sep)
        if i > limit * 0.5:
            cut = cut[: i + 1]
            break
    return cut.rstrip() + " …[truncated]"


# Common tokens that inflate overlap without indicating relevance.
_MEM_STOP = {"the", "and", "for", "with", "not", "via", "per", "any", "all", "new",
             "code", "node", "map", "data", "file", "list", "task", "type", "into",
             "loci", "dama", "mcp", "fix", "add"}


_MEM_DISTINCTIVE = 7  # a shared token this long is topical, not incidental English


def _select_memory_files(task: dict, memory_dir: str, limit: int = 2,
                         min_score: int = 2) -> list[tuple[str, str, str]]:
    """Top-`limit` MEMORY.md entries that share enough DISTINCTIVE tokens with the task.
    Returns [(slug, index_line, body)] with bodies read. Fail-open.

    A candidate must share >= `min_score` tokens AND >= 1 *distinctive* token (len >=
    _MEM_DISTINCTIVE) — otherwise two incidental English words ("event", "single") from
    an unrelated memory pass the bar and inject off-topic noise. Ranked by a weight that
    counts distinctive tokens double. Capped so a fuzzy match never crowds out the precise
    (case/RAG) lanes."""
    out: list[tuple[str, str, str]] = []
    try:
        d = Path(memory_dir)
        idx = d / "MEMORY.md"
        if not idx.exists():
            return out
        want = {t.lower() for t in _TOKEN_RE.findall(
            f"{task.get('title','')} {task.get('focus','')} {' '.join(task.get('codeRefs', []) or [])}")
            if len(t) >= 4 and t.lower() not in _MEM_STOP}
        if not want:
            return out
        scored = []
        for line in idx.read_text(errors="ignore").splitlines():
            m = re.search(r"\((?P<file>[\w.-]+\.md)\)", line)
            if not m:
                continue
            toks = {t.lower() for t in _TOKEN_RE.findall(line) if len(t) >= 4}
            shared = want & toks
            if len(shared) < min_score or not any(len(t) >= _MEM_DISTINCTIVE for t in shared):
                continue
            weight = sum(2 if len(t) >= _MEM_DISTINCTIVE else 1 for t in shared)
            scored.append((weight, m.group("file"), line.strip()))
        scored.sort(reverse=True)
        for score, fname, line in scored[:limit]:
            body = ""
            fp = d / fname
            if fp.exists():
                body = fp.read_text(errors="ignore")
            out.append((fname, line, body))
    except Exception:
        return out
    return out


def _jload(s: Any) -> dict | list | None:
    try:
        return json.loads(s) if isinstance(s, str) else s
    except Exception:
        return None


def ground(task: dict, opts: Optional[dict] = None) -> dict:
    """Assemble a grounding block for a task. Returns
    {block: str, sources: [str], chars: int, degraded: bool}. Fail-open.

    task: {title, focus?, caseIds?:[str], entities?:[str], codeRefs?:[str]}
    opts: {budgetChars=4000, memoryDir=..., allowKeyword=False, graphAvailable=False}
    """
    opts = opts or {}
    budget = int(opts.get("budgetChars", 4000))
    memory_dir = opts.get("memoryDir", _MEMORY_DIR_DEFAULT)
    parts: list[str] = []
    sources: list[str] = []
    degraded = False
    remaining = [budget]

    def add(tag: str, text: str, slice_frac: float) -> None:
        if remaining[0] <= 0 or not text:
            return
        cap = min(remaining[0], max(200, int(budget * slice_frac)))
        chunk = _truncate(text, cap)
        block = f"[{tag}] {chunk}"
        parts.append(block)
        sources.append(tag)
        remaining[0] -= len(block)

    import importlib
    try:
        S = importlib.import_module("server")
    except Exception:
        S = None

    # 1. Named cases -> investigation_load (structured, retracted excluded). Fail-open per
    # case: a raising server tool (or malformed finding) skips that case, never aborts ground().
    for cid in (task.get("caseIds") or [])[:3]:
        if not S:
            break
        try:
            data = _jload(S.investigation_load(cid, last_n_findings=6))
            if isinstance(data, dict) and not data.get("error"):
                man = data.get("manifest", {})
                summary = f"{cid} :: hypothesis={man.get('hypothesis')} | next={man.get('next_step')}"
                add(f"case:{cid}", summary, 0.12)
                for f in (data.get("recent_findings") or [])[:3]:
                    if isinstance(f, dict):
                        add(f"case:{cid}:finding", str(f.get("text", "")), 0.08)
        except Exception:
            continue

    # 3. Exact entities -> entity_lookup (O(1), no embedding). Fail-open per entity.
    for ent in (task.get("entities") or [])[:5]:
        if not S:
            break
        try:
            data = _jload(S.investigation_entity_lookup(ent, limit=3))
            if isinstance(data, dict) and data.get("total_findings"):
                add(f"entity:{ent}", f"seen in {data.get('investigations_count')} case(s), "
                                     f"{data.get('total_findings')} finding(s)", 0.06)
        except Exception:
            continue

    # 4. Code graph (only if reconnected / available).
    if opts.get("graphAvailable") and task.get("codeRefs") and S:
        for ref in (task.get("codeRefs") or [])[:2]:
            try:
                rep = _jload(S.impact_report(ref))
                if isinstance(rep, dict) and rep.get("resolved"):
                    add(f"code:{ref}", f"callers={rep.get('transitive_caller_count')} "
                                       f"findings={rep.get('referencing_finding_count')} "
                                       f"co={[c.get('name') for c in rep.get('co_referenced', [])[:4]]}", 0.10)
            except Exception:
                pass
    elif task.get("codeRefs"):
        degraded = True  # code-graph grounding wanted but unavailable

    # 5. Semantic RAG (enhancement; live now that embeddings are up, degraded when down).
    if S and remaining[0] > 400:
        try:
            q = f"{task.get('title','')} {task.get('focus','')}".strip()
            res = _jload(S.rag_context_search(q, budget_chars=min(remaining[0], 2000), limit=6))
            ctx = (res or {}).get("context", "") if isinstance(res, dict) else ""
            if ctx and (res.get("result_count") or 0) > 0:
                add("rag", ctx, 0.35)
            elif isinstance(res, dict) and not res.get("qdrant_available", True):
                degraded = True
        except Exception:
            degraded = True

    # Curated memory files — a SUPPLEMENT after the precise (case/RAG) lanes, so a
    # fuzzy-matched memory can never crowd them out. Capped + thresholded selection.
    for fname, line, body in _select_memory_files(task, memory_dir):
        add(f"memory:{fname[:-3]}", (body or line), 0.15)

    # 6. Keyword/FTS fallback (off by default; filtered) — only if structured yield was thin.
    if opts.get("allowKeyword") and S and sum(len(p) for p in parts) < 500:
        try:
            res = _jload(S.investigation_search(f"{task.get('title','')} {task.get('focus','')}", limit=8))
            items = (res or {}).get("results", []) if isinstance(res, dict) else []
            for it in filter_noise([{"text": r.get("text"), "source": r.get("source")} for r in items])[:3]:
                add("recall", str(it.get("text", "")), 0.10)
        except Exception:
            pass

    header = ("## GROUNDING — prior context (read-only reference, NOT ground truth; verify "
              "against live code/data before asserting; cite the [tag] if you rely on it)")
    if degraded:
        header += "\n(NOTE: some grounding lanes were unavailable this run — coverage is partial.)"
    footer = ("Do not present facts absent above as remembered; if grounding is silent on a "
              "point, say so.")
    block = header + "\n" + "\n".join(parts) + "\n" + footer if parts else ""
    return {"block": block, "sources": sources, "chars": len(block), "degraded": degraded}
