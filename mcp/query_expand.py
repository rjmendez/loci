"""RAG query expansion (HyDE-lite) — widen a retrieval query for better recall.

A single retrieval query often misses relevant chunks because the corpus phrases the
same idea differently (synonyms, domain jargon, alternate framings). This module asks
the local-GPU generation tier for a few alternate phrasings plus a handful of domain
keywords, so the caller can fan the search out over all of them and union the hits.

Design mirrors mcp/embed_ops.py:

- Generation is on the *generation* tier (Ollama qwen2.5:3b), which is injectable so a
  warm client can be reused and tests can stub it. `gen_fn` defaults to None; when None we
  LAZILY import ``llm_local.generate`` at call time, so importing this module never hard-
  requires llm_local (a sibling agent is writing it in parallel).

  gen_fn contract (shared): gen_fn(prompt, *, fmt=None, max_tokens=256) -> {"text": str,
  "ok": bool}. ok=False signals the caller should fall back — we treat it as degraded.

- Fail-open: on not-ok / timeout / parse failure / empty output, return a well-formed
  degraded result that still includes the ORIGINAL query, so retrieval always has at least
  one term to run. Never raises.

Grounding: [pattern:injectable] lazy llm_local import + injectable gen_fn; [interface]
the gen_fn contract above; [pattern:fail-open] degraded=True no-op fallback. The prompt
copy/wording and the JSON schema {queries,keywords} are this task's design — the grounding
is silent on them.
"""
from __future__ import annotations

from typing import Callable, Optional

from model_json import extract_json_object

# Private alias kept so existing call sites and test monkeypatches of the private
# name keep working after the shared extraction moved to mcp/model_json.py.
_extract_json_object = extract_json_object

# Injectable generation function type: matches the shared [interface] contract.
GenFn = Callable[..., dict]

_PROMPT_TMPL = (
    "You expand a search query to improve document retrieval recall.\n"
    "Given the user's query, produce:\n"
    "  - queries: {n_queries} alternate phrasings of the SAME information need "
    "(synonyms, rephrasings, more/less specific variants). Do NOT answer the query.\n"
    "  - keywords: up to {n_keywords} salient domain terms / entities likely to appear "
    "in relevant documents.\n"
    "Respond with ONLY a JSON object of this exact shape, no prose:\n"
    '{{"queries": ["..."], "keywords": ["..."]}}\n\n'
    "Query: {query}\n"
)


def _lazy_generate(prompt: str, *, fmt: Optional[str] = None, max_tokens: int = 256) -> dict:
    """Default gen_fn: import llm_local.generate only when actually called (fail-open)."""
    try:
        from llm_local import generate  # imported lazily so module import never needs it
        return generate(prompt, fmt=fmt, max_tokens=max_tokens)
    except Exception:
        return {"text": "", "ok": False}


def _clean_list(raw, cap: int, seed: Optional[str] = None) -> list[str]:
    """Coerce to a de-duplicated, order-stable list of non-empty strings, capped at `cap`.

    If `seed` is given it is placed first (so the original query always leads `queries`).
    De-dup is case-insensitive on the trimmed value.
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(v) -> None:
        if not isinstance(v, str):
            v = str(v)
        v = v.strip()
        if not v:
            return
        k = v.lower()
        if k in seen:
            return
        seen.add(k)
        out.append(v)

    if seed is not None:
        _add(seed)
    if isinstance(raw, (list, tuple)):
        for item in raw:
            _add(item)
    elif raw:
        _add(raw)
    return out[:cap]


def _degraded(query: str) -> dict:
    """Well-formed no-op fallback: original query only, no keywords, degraded=True."""
    q = (query or "").strip()
    return {"queries": [q] if q else [], "keywords": [], "degraded": True}


def expand(query: str, gen_fn: Optional[GenFn] = None,
           n_queries: int = 3, n_keywords: int = 6) -> dict:
    """Expand `query` into alternate phrasings + domain keywords for retrieval fan-out.

    Returns {"queries": [...], "keywords": [...], "degraded": bool}. `queries` always
    leads with the original query. `n_queries` caps the total queries (original + alts);
    `n_keywords` caps keywords. Fail-open: on any failure returns {queries:[query],
    keywords:[], degraded:True}. Never raises.
    """
    q = (query or "").strip()
    if not q:
        return {"queries": [], "keywords": [], "degraded": True}
    n_queries = max(1, int(n_queries))
    n_keywords = max(0, int(n_keywords))

    prompt = _PROMPT_TMPL.format(query=q, n_queries=n_queries, n_keywords=n_keywords)
    fn = gen_fn or _lazy_generate
    try:
        res = fn(prompt, fmt="json", max_tokens=220)
    except Exception:
        return _degraded(q)

    if not isinstance(res, dict) or not res.get("ok"):
        return _degraded(q)

    obj = _extract_json_object(res.get("text", ""))
    if obj is None:
        return _degraded(q)

    queries = _clean_list(obj.get("queries"), cap=n_queries, seed=q)
    keywords = _clean_list(obj.get("keywords"), cap=n_keywords)
    # If the model gave us nothing usable beyond the seed and no keywords, mark degraded so
    # the caller knows expansion added no signal (still a valid, runnable result).
    degraded = len(queries) <= 1 and not keywords
    return {"queries": queries, "keywords": keywords, "degraded": degraded}
