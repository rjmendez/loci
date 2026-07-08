"""Adversarial finding-verification — the candidate->skeptic->keep-if-survives loop as a tool.

Workflows run a per-finding "try to refute this" pass before keeping a claim: a skeptic
reads the claim (plus any grounding context) and actively tries to break it. If it survives
the attack, we keep it (confirmed); if the attack lands, we drop it (refuted); if the skeptic
can't tell, we stay cautious (uncertain). This module makes that loop a reusable Loci
primitive so any caller gets the same discipline without re-implementing the prompt.

Design mirrors mcp/query_expand.py:

- Reasoning runs on the *generation* tier (Ollama qwen2.5:3b), injectable so a warm client
  can be reused and tests can stub it. `gen_fn` defaults to None; when None we LAZILY import
  ``llm_local.generate`` at call time, so importing this module never hard-requires llm_local.

  gen_fn contract (shared): gen_fn(prompt, *, fmt=None, max_tokens=256) -> {"text": str,
  "ok": bool}. ok=False signals the caller should fall back — we treat it as degraded.

- Grounding is optional. When an investigation_id is given and no explicit context is passed,
  we LAZILY + fail-open pull a little RAG context (rag_context_search) to help the skeptic.
  This is injectable via `rag_fn` for tests; a dead/absent RAG lane just means no extra context.

- Code grounding is optional. For CODE claims a prose summary is a poor thing to reason over
  (the live smoke saw a clearly-true code claim come back 'uncertain'), so when the claim /
  context / an explicit `code_refs` arg carry ``file:line`` (or ``file:start-end``) references
  we FETCH the actual source lines and put them in the prompt so the skeptic reasons over real
  code. The file reader is injectable via `reader` for tests; it is fail-open (an unreadable
  path just contributes no code). We also surface the model's RAW `reasoning` alongside the
  verdict so a caller can still judge when the verdict is the cautious 'uncertain'.

- Fail-open + skeptical default: on not-ok / timeout / parse failure / any error we return a
  well-formed {"verdict": "uncertain"} result rather than raising. We also default to the
  cautious verdict when the model is unsure — a claim is only 'confirmed' when the skeptic
  explicitly fails to refute it. Never raises.

Grounding: [pattern:injectable] lazy llm_local import + injectable gen_fn; [interface] the
gen_fn contract above; [pattern:fail-open] degraded 'uncertain' fallback. The skeptic prompt
copy and the JSON schema {verdict,refutation,confidence} are this task's design.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Callable, Optional

# Injectable function types: match the shared [interface] contracts.
GenFn = Callable[..., dict]
RagFn = Callable[..., dict]
# reader(path) -> full file text; fail-open to "" on any error.
ReaderFn = Callable[[str], str]

_VALID_VERDICTS = ("confirmed", "refuted", "uncertain")

# Caps so a stray/huge ref can't blow up the prompt. Fail-open, additive.
_MAX_REFS = 8
_MAX_LINES_PER_REF = 60
# Size cap on a single file read so an oversized/binary file can't blow up memory/prompt.
_MAX_FILE_BYTES = 1_000_000

# "file:line" or "file:start-end". Require the path to contain a '.' or '/' so bare
# "10:30"-style tokens don't get mistaken for refs; anything that still slips through
# just fails-open at read time (unreadable path -> no code).
_REF_RE = re.compile(r"([\w./\-]*[./][\w./\-]*):(\d+)(?:-(\d+))?")

_PROMPT_TMPL = (
    "You are a rigorous SKEPTIC performing adversarial verification of a claim.\n"
    "Your job is to TRY TO REFUTE the claim — actively look for a counterexample, a logical\n"
    "flaw, a missing precondition, or evidence in the context that contradicts it.\n"
    "Do NOT try to confirm it; assume it is wrong until it survives your attack.\n\n"
    "Decide a verdict:\n"
    "  - \"refuted\": you found a concrete reason the claim is false or unsupported.\n"
    "  - \"confirmed\": you genuinely tried and CANNOT refute it; the claim holds.\n"
    "  - \"uncertain\": you cannot tell from the claim and context (default when unsure).\n"
    "Prefer \"refuted\" or \"uncertain\" over \"confirmed\" when in doubt.\n\n"
    "Respond with ONLY a JSON object of this exact shape, no prose:\n"
    '{{"verdict": "confirmed|refuted|uncertain", "reasoning": "your step-by-step skeptical '
    'analysis", "refutation": "your strongest attack or why it survives", "confidence": 0.0}}\n\n'
    "CLAIM:\n{claim}\n\n"
    "REFERENCED CODE (actual source at the cited locations — trust this over any summary):\n"
    "{code}\n\n"
    "CONTEXT (may be empty — do not assume it is complete):\n{context}\n"
)


def _lazy_generate(prompt: str, *, fmt: Optional[str] = None, max_tokens: int = 256) -> dict:
    """Default gen_fn: import llm_local.generate only when actually called (fail-open)."""
    try:
        from llm_local import generate  # imported lazily so module import never needs it
        return generate(prompt, fmt=fmt, max_tokens=max_tokens)
    except Exception:
        return {"text": "", "ok": False}


def _lazy_rag(query: str, *, limit: int = 5) -> dict:
    """Default rag_fn: best-effort grounding via rag_context_search. Fail-open to {}.

    server.rag_context_search returns a JSON string; we parse it defensively. Any failure
    (Qdrant down, tool absent, bad JSON) yields {} so verification proceeds ungrounded.
    """
    try:
        import server
        raw = server.rag_context_search(query, limit=limit)
        obj = json.loads(raw) if isinstance(raw, str) else raw
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


@lru_cache(maxsize=1)
def _repo_root() -> str:
    """Best-effort repo root used to sandbox file refs. Walk up from this module for a
    ``.git`` marker; fall back to the package parent. Cached; never raises."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        d = here
        while True:
            if os.path.exists(os.path.join(d, ".git")):
                return os.path.realpath(d)
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
        return os.path.realpath(os.path.dirname(here))  # mcp/ -> repo root
    except Exception:
        return os.path.realpath(os.getcwd())


def _safe_resolve(path: str) -> Optional[str]:
    """Resolve a repo-relative ref path to an absolute path UNDER the repo root.

    SECURITY: file refs are parsed from free-form (attacker-influenceable) claim/context
    text, so the default reader must never read arbitrary files. Rejects absolute paths and
    any ``..`` traversal segment, resolves relative to the repo root, and returns None for
    anything that still lands outside the root (e.g. via a symlink). Returns None on reject
    (caller fails open by contributing no code). Never raises.
    """
    if not isinstance(path, str) or not path:
        return None
    if os.path.isabs(path):
        return None
    if ".." in path.replace("\\", "/").split("/"):
        return None
    try:
        root = _repo_root()
        full = os.path.realpath(os.path.join(root, path))
        # realpath collapses symlinks/traversal; require the result to stay under the root.
        if full == root or full.startswith(root + os.sep):
            return full
    except Exception:
        return None
    return None


def _lazy_read_file(path: str) -> str:
    """Default reader: read a repo-relative source file, SANDBOXED under the repo root.

    Rejects absolute paths, ``..`` traversal, and anything resolving outside the repo, and
    caps the read at ``_MAX_FILE_BYTES``. Fail-open to "" on any rejection or error.
    """
    try:
        full = _safe_resolve(path)
        if not full or not os.path.isfile(full):
            return ""
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            return f.read(_MAX_FILE_BYTES)
    except Exception:
        return ""


def _parse_refs(*texts: str):
    """Pull unique (path, start, end) file:line refs out of the given strings, in order.

    Deduplicated and capped at _MAX_REFS. Non-string inputs are ignored (fail-open).
    SECURITY: absolute paths and ``..`` traversal are dropped here (defense-in-depth; the
    default reader also sandboxes reads under the repo root) so a ref like ``/etc/passwd:1``
    or ``../../secret.txt:1`` parsed from free-form text never becomes a fetched ref.
    """
    refs = []
    seen = set()
    for t in texts:
        if not isinstance(t, str):
            continue
        for m in _REF_RE.finditer(t):
            path = m.group(1)
            if os.path.isabs(path) or ".." in path.replace("\\", "/").split("/"):
                continue
            start = int(m.group(2))
            end = int(m.group(3)) if m.group(3) else start
            key = (path, start, end)
            if key in seen:
                continue
            seen.add(key)
            refs.append(key)
            if len(refs) >= _MAX_REFS:
                return refs
    return refs


def _fetch_code(refs, reader: ReaderFn) -> str:
    """Read the cited line ranges via `reader` and format them (line-numbered) for the prompt.

    Fail-open per ref: an unreadable/missing file or out-of-range span just contributes nothing.
    """
    blocks = []
    for path, start, end in refs:
        try:
            text = reader(path)
        except Exception:
            text = ""
        if not isinstance(text, str) or not text:
            continue
        lines = text.splitlines()
        n = len(lines)
        s = max(1, start)
        if s > n:
            continue
        e = min(n, end if end >= start else start)
        if e - s + 1 > _MAX_LINES_PER_REF:
            e = s + _MAX_LINES_PER_REF - 1
        numbered = "\n".join(f"{i}: {lines[i - 1]}" for i in range(s, e + 1))
        header = f"--- {path}:{start}" + (f"-{end}" if end != start else "") + " ---"
        blocks.append(header + "\n" + numbered)
    return "\n\n".join(blocks)


def _coerce_code_refs(code_refs) -> list:
    """Normalize the ``code_refs`` arg to a list of ref strings; ignore other types.

    Documented as a list, but a caller may pass a single ``file:line`` string; ``list(str)``
    would split it into characters. Accept a list/tuple (keeping only its string items) or a
    lone string; anything else yields []. Never raises.
    """
    if code_refs is None:
        return []
    if isinstance(code_refs, str):
        return [code_refs]
    if isinstance(code_refs, (list, tuple)):
        return [x for x in code_refs if isinstance(x, str)]
    return []


def _extract_json_object(text: str) -> Optional[dict]:
    """Defensively pull a JSON object out of possibly-noisy model text.

    Handles: clean JSON, JSON wrapped in ```json fences, and JSON embedded in stray prose.
    Returns the parsed dict, or None if nothing parseable is found.
    """
    if not text or not isinstance(text, str):
        return None
    # 1) straight parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # 2) strip code fences and retry
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # 3) brace-match scan: first '{' whose balanced span parses as an object
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        start = -1  # keep scanning for a later valid object
    return None


def _coerce_verdict(raw) -> str:
    """Map model output to one of _VALID_VERDICTS, skeptically. Unknown -> 'uncertain'."""
    if not isinstance(raw, str):
        return "uncertain"
    v = raw.strip().lower()
    return v if v in _VALID_VERDICTS else "uncertain"


def _coerce_confidence(raw) -> float:
    """Coerce confidence to a float in [0,1]; unparseable -> 0.0 (cautious)."""
    try:
        c = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if c != c:  # NaN
        return 0.0
    return max(0.0, min(1.0, c))


def _degraded(refutation: str = "", reasoning: str = "") -> dict:
    """Well-formed skeptical fallback: uncertain, low confidence, degraded=True.

    `reasoning` carries any raw model text we did manage to get, so a caller can still judge.
    """
    return {"verdict": "uncertain", "refutation": refutation, "reasoning": reasoning,
            "confidence": 0.0, "degraded": True}


def verify_finding(claim: str,
                   context: str = "",
                   investigation_id: Optional[str] = None,
                   gen_fn: Optional[GenFn] = None,
                   rag_fn: Optional[RagFn] = None,
                   code_refs: Optional[list] = None,
                   reader: Optional[ReaderFn] = None) -> dict:
    """Adversarially verify a `claim`: run a skeptic that tries to refute it.

    Args:
        claim: the finding/claim to stress-test.
        context: optional grounding (code snippet, file refs, prior evidence).
        investigation_id: if given and `context` is empty, best-effort pull RAG grounding
            (fail-open) to give the skeptic something to attack with.
        gen_fn: injectable generation fn (shared contract). None -> lazy llm_local.generate.
        rag_fn: injectable grounding fn. None -> lazy rag_context_search.
        code_refs: optional list of ``file:line`` / ``file:start-end`` strings whose source
            should be fetched into the prompt. ``file:line`` refs found in the claim/context
            are also picked up automatically. Fail-open: unreadable refs contribute nothing.
        reader: injectable file reader ``reader(path) -> text``. None -> lazy FS read.

    Returns:
        {"verdict": "confirmed"|"refuted"|"uncertain", "refutation": str, "reasoning": str,
         "confidence": float, "degraded": bool}. `reasoning` surfaces the model's raw analysis
         so a caller can judge even on 'uncertain'. Fail-open: on any failure returns a
         skeptical uncertain result. Never raises.
    """
    c = (claim or "").strip()
    if not c:
        return _degraded()

    ctx = (context or "").strip()
    # Optional, fail-open grounding: only when we have an investigation and no explicit context.
    if not ctx and investigation_id:
        rag = rag_fn or _lazy_rag
        try:
            res = rag(c)
            if isinstance(res, dict):
                ctx = str(res.get("context") or "").strip()
        except Exception:
            ctx = ""

    # Optional, fail-open code grounding: fetch real source for any cited file:line refs so
    # the skeptic reasons over code, not a prose summary. Explicit code_refs are authoritative.
    code_block = ""
    try:
        refs = _parse_refs(*_coerce_code_refs(code_refs), c, ctx)
        if refs:
            code_block = _fetch_code(refs, reader or _lazy_read_file)
    except Exception:
        code_block = ""

    prompt = _PROMPT_TMPL.format(claim=c, code=code_block or "(none)", context=ctx or "(none)")
    fn = gen_fn or _lazy_generate
    try:
        res = fn(prompt, fmt="json", max_tokens=384)
    except Exception:
        return _degraded()

    if not isinstance(res, dict) or not res.get("ok"):
        return _degraded(reasoning=(res.get("text", "") if isinstance(res, dict) else ""))

    raw = res.get("text", "")
    obj = _extract_json_object(raw)
    if obj is None:
        return _degraded(reasoning=raw if isinstance(raw, str) else "")

    verdict = _coerce_verdict(obj.get("verdict"))
    refutation = obj.get("refutation")
    if not isinstance(refutation, str):
        refutation = "" if refutation is None else str(refutation)
    confidence = _coerce_confidence(obj.get("confidence"))
    # Surface raw reasoning: prefer the model's own field, fall back to its raw text.
    reasoning = obj.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        reasoning = raw if isinstance(raw, str) else ""
    return {"verdict": verdict, "refutation": refutation.strip(),
            "reasoning": reasoning.strip(), "confidence": confidence, "degraded": False}
