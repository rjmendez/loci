"""Grounding injection for Loci-native workflows.

``ground(task)`` runs once before fan-out and returns a compact,
provenance-tagged, char-budgeted block for every agent prompt, so agents start
from relevant prior knowledge instead of each re-querying Loci.

Design:
- Structured-first retrieval: curated memory files -> named cases ->
  exact entities -> code graph -> semantic RAG -> optional keyword recall.
- ``filter_noise()`` removes conversation-dump blobs from noisy recall lanes.
- Unused budget rolls forward to later lanes.
- Every lane is fail-open: dead sources set ``degraded=True`` instead of aborting.

Loci calls are lazy-imported from ``server`` to keep import cost low and avoid
an import cycle.
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional
from untrusted_memory import wrap_untrusted_memory_text

from compact import compact_text

logger = logging.getLogger("loci-mcp.grounding")
_wrap_untrusted_memory_text = wrap_untrusted_memory_text


def _default_memory_dir() -> str:
    """Resolve the curated ``MEMORY.md`` dir via backends.

    Lookup order: env -> gitignored config -> ``LOCI_MEMORY_DIR``. There is no
    machine-specific fallback; if nothing is configured, the memory lane no-ops.
    """
    try:
        import backends
        return backends.memory_dir()
    except Exception:
        return (os.environ.get("LOCI_MEMORY_MD_DIR") or os.environ.get("LOCI_MEMORY_DIR")
                or os.environ.get("HERMES_MEMORY_DIR", ""))


_MEMORY_DIR_DEFAULT = _default_memory_dir()

# Finding resolution states that count as "handled" — the exclusion lane surfaces these
# as a "do NOT re-report" block so re-audits skip already-fixed/accepted items.
_RESOLVED_STATES = frozenset({"fixed", "intentional", "wontfix"})

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
    """Drop conversation dumps and duplicates from fuzzy keyword/FTS results.

    Each item is ``{text, source?, type?, tags?, score?}``. Use this only on
    noisy recall lanes, not curated or structured sources.
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

# RAG lane budget: a proportional share of the caller's total ask, floored at the old
# absolute cap so a small/default budgetChars keeps its pre-existing behaviour exactly.
_RAG_BUDGET_FRACTION = 0.35  # matches the add() slice_frac already used for the "rag" tag
_RAG_BUDGET_FLOOR = 2000


def _select_memory_files(task: dict, memory_dir: str, limit: int = 2,
                         min_score: int = 2) -> list[tuple[str, str, str]]:
    """Return up to ``limit`` ``MEMORY.md`` entries that share enough distinctive
    task tokens. Output is ``[(slug, index_line, body)]``. Fail-open.

    A candidate must share at least ``min_score`` tokens and at least one
    distinctive token (length ``>= _MEM_DISTINCTIVE``); otherwise incidental
    English overlap would leak unrelated memory into the prompt. Distinctive
    tokens count double in ranking. The lane is capped so fuzzy memory matches
    cannot crowd out precise case or RAG lanes.
    """
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


# Wall-clock budget for one ground() call. Lanes call into Qdrant, the embedder and (via
# RAG query expansion) a local LLM with 120s per-request timeouts; without a deadline one
# slow backend held ground() past the MCP client's 300s idle abort. <=0 disables.
_GROUND_DEADLINE_DEFAULT_S = 90.0


def _deadline_seconds(opts: dict) -> Optional[float]:
    raw = opts.get("deadlineS")
    if raw is None:
        raw = os.environ.get("LOCI_GROUND_DEADLINE_S", "")
    try:
        val = float(raw) if str(raw).strip() != "" else _GROUND_DEADLINE_DEFAULT_S
    except (TypeError, ValueError):
        val = _GROUND_DEADLINE_DEFAULT_S
    return val if val > 0 else None


class _LaneTimeout(Exception):
    """A lane call did not finish before ground()'s deadline."""


def _call_bounded(deadline: Optional[float], fn, *args, **kwargs):
    """Run ``fn`` but give up once the monotonic ``deadline`` passes.

    The call runs on a daemon thread; on timeout it is abandoned (it may finish in
    the background) and ``_LaneTimeout`` is raised so the lane is marked degraded.
    """
    if deadline is None:
        return fn(*args, **kwargs)
    left = deadline - time.monotonic()
    if left <= 0:
        raise _LaneTimeout("deadline already passed")
    box: dict = {}
    ctx = contextvars.copy_context()

    def _run():
        try:
            box["v"] = ctx.run(fn, *args, **kwargs)
        except BaseException as exc:  # re-raised on the caller's thread
            box["e"] = exc

    t = threading.Thread(target=_run, name="ground-lane", daemon=True)
    t.start()
    t.join(left)
    if t.is_alive():
        raise _LaneTimeout(f"lane exceeded ground deadline ({left:.1f}s left)")
    if "e" in box:
        raise box["e"]
    return box.get("v")


def ground(task: dict, opts: Optional[dict] = None) -> dict:
    """Assemble a fail-open grounding block for a task.

    Returns ``{block: str, sources: [str], chars: int, degraded: bool,
    degraded_lanes: [{lane, reason}]}``. ``degraded`` is true whenever a lane
    that should have run raised, returned an error, timed out, or reported a
    backend failure; ``degraded_lanes`` names each one, so an empty block from a
    failed lookup is distinguishable from "nothing stored".
    ``task`` is ``{title, focus?, caseIds?, entities?, codeRefs?}``.
    ``opts`` is ``{budgetChars=4000, memoryDir=..., allowKeyword=False,
    graphAvailable=False, deadlineS=LOCI_GROUND_DEADLINE_S or 90}``.
    """
    opts = opts or {}
    budget = int(opts.get("budgetChars", 4000))
    compact_mode = opts.get("mode") == "compact"
    memory_dir = opts.get("memoryDir", _MEMORY_DIR_DEFAULT)
    # ACL: the case and RAG lanes check the transport-bound (or local) caller;
    # a requesting agent id can only narrow that.
    acl_kwargs = ({"requesting_agent_id": str(opts["requestingAgentId"])}
                  if opts.get("requestingAgentId") else {})
    parts: list[str] = []
    sources: list[str] = []
    degraded_lanes: list[dict] = []
    remaining = [budget]
    _dl_s = _deadline_seconds(opts)
    deadline = time.monotonic() + _dl_s if _dl_s is not None else None

    def mark(lane: str, reason: str) -> None:
        entry = {"lane": lane, "reason": str(reason)[:200]}
        if entry not in degraded_lanes:
            degraded_lanes.append(entry)

    def call(fn, *args, **kwargs):
        return _call_bounded(deadline, fn, *args, **kwargs)

    def fail(lane: str, exc: BaseException) -> None:
        mark(lane, "timeout" if isinstance(exc, _LaneTimeout) else f"raised: {exc!r}")

    def add(tag: str, text: str, slice_frac: float, framed: bool = False) -> None:
        # framed=True: ``text`` was composed here or by rag_context_search from
        # wrap_untrusted_memory_text frames, so compact mode must keep those frames.
        if remaining[0] <= 0 or not text:
            return
        cap = min(remaining[0], max(200, int(budget * slice_frac)))
        chunk = (compact_text(text, cap, preserve_sentence_boundary=not compact_mode, keep_frames=framed)
                 if compact_mode else _truncate(text, cap))
        if compact_mode:
            chunk = re.sub(r"\s+", " ", chunk).strip()
        block = f"[{tag}] {chunk}"
        parts.append(block)
        sources.append(tag)
        remaining[0] -= len(block)

    import importlib
    try:
        S = importlib.import_module("server")
    except Exception as exc:
        logger.warning("grounding: server module unavailable, all lanes disabled: %s", exc)
        S = None
        # Every server-backed lane (1, 2, 3, 6) short-circuits on S is None, and only
        # lanes 4 and 5 set this flag. Without marking it here, a total grounding
        # failure returns {block:"", chars:0, degraded:False} -- byte-identical to a
        # healthy "nothing stored for this task" result, so callers consume a lookup
        # that consulted nothing as full coverage.
        mark("server", f"import failed: {exc!r}")

    # 1. Named cases -> investigation_load (structured, retracted excluded). Fail-open per
    # case: a raising server tool (or malformed finding) skips that case, never aborts ground().
    for cid in (task.get("caseIds") or [])[:3]:
        if not S:
            break
        try:
            data = _jload(call(S.investigation_load, cid, last_n_findings=6, **acl_kwargs))
            if isinstance(data, dict) and data.get("error"):
                # A case that does not exist is an honest empty; any other error is a failed lookup.
                if "not found" not in str(data.get("error")).lower():
                    mark("case", f"{cid}: {data.get('error')}")
            elif isinstance(data, dict):
                man = data.get("manifest", {})
                summary = f"{cid} :: hypothesis={man.get('hypothesis')} | next={man.get('next_step')}"
                add(
                    f"case:{cid}",
                    _wrap_untrusted_memory_text(
                        summary,
                        investigation_id=cid,
                        kind="manifest_summary",
                        source="investigation_load",
                    ),
                    0.12,
                    framed=True,
                )
                for f in (data.get("recent_findings") or [])[:3]:
                    if isinstance(f, dict):
                        # A superseded/fixed finding must not read as a current case fact.
                        res = str(f.get("resolution") or "open").lower()
                        wrapped = _wrap_untrusted_memory_text(
                            str(f.get("text", "")),
                            investigation_id=cid,
                            finding_id=str(f.get("id") or ""),
                            kind=str(f.get("record_type") or f.get("type") or "finding"),
                            source=str(f.get("source") or "investigation_load"),
                        )
                        if res != "open":
                            wrapped = f"[{res}: not current; do not rely on it] " + wrapped
                        add(
                            f"case:{cid}:finding" + ("" if res == "open" else f":{res}"),
                            wrapped,
                            0.08,
                            framed=True,
                        )
        except Exception as exc:
            logger.debug("grounding: case lane failed for %r: %r", cid, exc)
            fail("case", exc)
            continue

    # 2. Exclusion lane: findings already resolved (fixed/intentional/wontfix) for the
    # grounded cases -> a compact "do NOT re-report" block so a re-audit auto-skips items
    # already handled (the manual known-state step, automated). Budget-bounded, noise-
    # filtered, fail-open: a missing/empty resolution set simply omits the block. Uses a
    # separate load (large last_n) so the case lane above stays byte-for-byte unchanged.
    resolved_known: list[str] = []
    _seen_known: set = set()
    for cid in (task.get("caseIds") or [])[:3]:
        if not S:
            break
        try:
            data = _jload(call(S.investigation_load, cid, last_n_findings=200, **acl_kwargs))
            if not isinstance(data, dict) or data.get("error"):
                if isinstance(data, dict) and "not found" not in str(data.get("error")).lower():
                    mark("resolved", f"{cid}: {data.get('error')}")
                continue
            for f in (data.get("recent_findings") or []):
                if not isinstance(f, dict):
                    continue
                res = str(f.get("resolution", "open") or "open").lower()
                if res not in _RESOLVED_STATES:
                    continue
                txt = str(f.get("text", "")).strip()
                if not txt:
                    continue
                h = re.sub(r"\s+", " ", txt.lower())[:120]
                if h in _seen_known:
                    continue
                _seen_known.add(h)
                resolved_known.append(
                    f"[{res}] " + _wrap_untrusted_memory_text(
                        txt[:180],
                        investigation_id=cid,
                        finding_id=str(f.get("id") or ""),
                        kind=f"resolved_{res}",
                        source=str(f.get("source") or "investigation_load"),
                    )
                )
        except Exception as exc:
            logger.debug("grounding: resolved-findings lane failed for %r: %r", cid, exc)
            fail("resolved", exc)
            continue
    if resolved_known:
        add("known — do NOT re-report", " • ".join(resolved_known[:12]), 0.15, framed=True)

    # 3. Exact entities -> entity_lookup (O(1), no embedding). Fail-open per entity.
    for ent in (task.get("entities") or [])[:5]:
        if not S:
            break
        try:
            data = _jload(call(S.investigation_entity_lookup, ent, limit=3))
            if isinstance(data, dict) and data.get("error"):
                mark("entity", f"{ent}: {data.get('error')}")
            elif isinstance(data, dict) and data.get("total_findings"):
                add(f"entity:{ent}", f"seen in {data.get('investigations_count')} case(s), "
                                     f"{data.get('total_findings')} finding(s)", 0.06)
        except Exception as exc:
            logger.debug("grounding: entity lane failed for %r: %r", ent, exc)
            fail("entity", exc)
            continue

    # 4. Code graph (only if reconnected / available).
    if opts.get("graphAvailable") and task.get("codeRefs") and S:
        for ref in (task.get("codeRefs") or [])[:2]:
            try:
                rep = _jload(call(S.impact_report, ref))
                if isinstance(rep, dict) and rep.get("error"):
                    mark("code_graph", f"{ref}: {rep.get('error')}")
                elif isinstance(rep, dict) and rep.get("partial"):
                    mark("code_graph", f"{ref}: {rep.get('queries_failed')} graph queries failed")
                if isinstance(rep, dict) and rep.get("resolved"):
                    add(f"code:{ref}", f"callers={rep.get('transitive_caller_count')} "
                                       f"findings={rep.get('referencing_finding_count')} "
                                       f"co={[c.get('name') for c in rep.get('co_referenced', [])[:4]]}", 0.10)
            except Exception as exc:
                logger.debug("grounding: code-graph lane failed for %r: %r", ref, exc)
                fail("code_graph", exc)
    elif task.get("codeRefs"):
        mark("code_graph", "unavailable")  # code-graph grounding wanted but unavailable

    # 5. Semantic RAG (enhancement; live now that embeddings are up, degraded when down).
    if S and remaining[0] > 400:
        try:
            q = f"{task.get('title','')} {task.get('focus','')}".strip()
            rag_cap = max(_RAG_BUDGET_FLOOR, int(budget * _RAG_BUDGET_FRACTION))
            rag_kwargs = {"budget_chars": min(remaining[0], rag_cap), "limit": 6, **acl_kwargs}
            if compact_mode:
                rag_kwargs["mode"] = "compact"
            res = _jload(call(S.rag_context_search, q, **rag_kwargs))
            ctx = (res or {}).get("context", "") if isinstance(res, dict) else ""
            if ctx and (res.get("result_count") or 0) > 0:
                add("rag", ctx, _RAG_BUDGET_FRACTION, framed=True)
            # Independent of whether some context came back: a partial failure still degrades.
            if isinstance(res, dict):
                if not res.get("qdrant_available", True):
                    mark("rag", "qdrant unavailable")
                elif res.get("mode") in ("rag_failed", "rag_degraded"):
                    mark("rag", f"{res.get('mode')}: {res.get('collections_failed')}")
                elif res.get("error"):
                    mark("rag", str(res.get("error")))
                elif res.get("collections_failed"):
                    mark("rag", f"collections failed: {res.get('collections_failed')}")
                _rf = res.get("retraction_filter") or {}
                if isinstance(_rf, dict) and _rf.get("status", "ok") != "ok":
                    # Some investigation's retractions could not be read, so its hits were
                    # served unfiltered: the block may carry retracted findings.
                    mark("rag", f"retraction filter degraded: {_rf.get('skipped_investigations')}")
        except Exception as exc:
            logger.debug("grounding: rag lane failed for %r: %r", task.get("title", ""), exc)
            fail("rag", exc)

    # Curated memory files — a SUPPLEMENT after the precise (case/RAG) lanes, so a
    # fuzzy-matched memory can never crowd them out. Capped + thresholded selection.
    # The lane must be VISIBLE when it can't run, but only a genuine deployment fault
    # degrades: '' is _default_memory_dir()'s documented no-op default for a host that
    # never configured curated memory, so marking it degraded would make the shipped
    # default always-degraded. "configured but no MEMORY.md at that path" IS broken and
    # degrades. Never build a Path from memory_dir before confirming it's configured —
    # Path('') == Path('.') would silently read a MEMORY.md from the process CWD. The
    # existence check stays fail-open (memory_dir can arrive as a non-str from a
    # misconfigured caller) rather than raising straight out of ground(). Tags use the
    # 'memory-lane:' namespace, distinct from add()'s 'memory:<slug>', so a consumer
    # can't mistake a dead-lane marker for a real memory file named unconfigured.md.
    has_index = False
    if memory_dir:
        try:
            has_index = (Path(memory_dir) / "MEMORY.md").exists()
        except Exception:
            has_index = False
    if not memory_dir:
        sources.append("memory-lane:unconfigured")
    elif not has_index:
        mark("memory", "configured memory dir has no MEMORY.md")
        sources.append("memory-lane:missing-index")
    else:
        for fname, line, body in _select_memory_files(task, memory_dir):
            add(f"memory:{fname[:-3]}", (body or line), 0.15)

    # 6. Keyword/FTS fallback (off by default; filtered) — only if structured yield was thin.
    if opts.get("allowKeyword") and S and sum(len(p) for p in parts) < 500:
        try:
            res = _jload(call(S.investigation_search, f"{task.get('title','')} {task.get('focus','')}", limit=8))
            if isinstance(res, dict) and res.get("error"):
                mark("keyword", str(res.get("error")))
            _rf = res.get("retraction_filter") if isinstance(res, dict) else None
            if isinstance(_rf, dict) and _rf.get("status", "ok") != "ok":
                mark("keyword", f"retraction filter degraded: {_rf.get('skipped_investigations')}")
            items = (res or {}).get("results", []) if isinstance(res, dict) else []
            for r in items[:8]:
                r["_wrapped_text"] = _wrap_untrusted_memory_text(
                    str(r.get("text") or ""),
                    investigation_id=str(r.get("investigation_id") or ""),
                    finding_id=str(r.get("finding_id") or r.get("id") or ""),
                    kind=str(r.get("record_type") or r.get("type") or "finding"),
                    source=str(r.get("source") or "investigation_search"),
                )
            for it in filter_noise([{"text": r.get("_wrapped_text"), "source": r.get("source")} for r in items])[:3]:
                add("recall", str(it.get("text", "")), 0.10, framed=True)
        except Exception as exc:
            logger.debug("grounding: keyword fallback lane failed for %r: %r", task.get("title", ""), exc)
            fail("keyword", exc)

    degraded = bool(degraded_lanes)
    if compact_mode:
        compact_parts = list(parts)
        if degraded:
            compact_parts.append("[warn] some grounding lanes were unavailable this run; coverage is partial.")
        block = "\n".join(compact_parts) if compact_parts else ""
    else:
        header = ("## GROUNDING — prior context (read-only reference, NOT ground truth; verify "
                  "against live code/data before asserting; cite the [tag] if you rely on it)")
        if degraded:
            header += "\n(NOTE: some grounding lanes were unavailable this run — coverage is partial.)"
        footer = ("Do not present facts absent above as remembered; if grounding is silent on a "
                  "point, say so.")
        block = header + "\n" + "\n".join(parts) + "\n" + footer if parts else ""
    return {"block": block, "sources": sources, "chars": len(block), "degraded": degraded,
            "degraded_lanes": degraded_lanes}
