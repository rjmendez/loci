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
import re
from typing import Callable, Optional

# Injectable function types: match the shared [interface] contracts.
GenFn = Callable[..., dict]
RagFn = Callable[..., dict]

_VALID_VERDICTS = ("confirmed", "refuted", "uncertain")

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
    '{{"verdict": "confirmed|refuted|uncertain", "refutation": "your strongest attack or '
    'why it survives", "confidence": 0.0}}\n\n'
    "CLAIM:\n{claim}\n\n"
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


def _degraded(refutation: str = "") -> dict:
    """Well-formed skeptical fallback: uncertain, low confidence, degraded=True."""
    return {"verdict": "uncertain", "refutation": refutation, "confidence": 0.0,
            "degraded": True}


def verify_finding(claim: str,
                   context: str = "",
                   investigation_id: Optional[str] = None,
                   gen_fn: Optional[GenFn] = None,
                   rag_fn: Optional[RagFn] = None) -> dict:
    """Adversarially verify a `claim`: run a skeptic that tries to refute it.

    Args:
        claim: the finding/claim to stress-test.
        context: optional grounding (code snippet, file refs, prior evidence).
        investigation_id: if given and `context` is empty, best-effort pull RAG grounding
            (fail-open) to give the skeptic something to attack with.
        gen_fn: injectable generation fn (shared contract). None -> lazy llm_local.generate.
        rag_fn: injectable grounding fn. None -> lazy rag_context_search.

    Returns:
        {"verdict": "confirmed"|"refuted"|"uncertain", "refutation": str,
         "confidence": float, "degraded": bool}. Fail-open: on any failure returns a
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

    prompt = _PROMPT_TMPL.format(claim=c, context=ctx or "(none)")
    fn = gen_fn or _lazy_generate
    try:
        res = fn(prompt, fmt="json", max_tokens=384)
    except Exception:
        return _degraded()

    if not isinstance(res, dict) or not res.get("ok"):
        return _degraded()

    obj = _extract_json_object(res.get("text", ""))
    if obj is None:
        return _degraded()

    verdict = _coerce_verdict(obj.get("verdict"))
    refutation = obj.get("refutation")
    if not isinstance(refutation, str):
        refutation = "" if refutation is None else str(refutation)
    confidence = _coerce_confidence(obj.get("confidence"))
    return {"verdict": verdict, "refutation": refutation.strip(),
            "confidence": confidence, "degraded": False}
