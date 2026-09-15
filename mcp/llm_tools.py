"""Local-model and embedding MCP wrappers, split from server.py.

These tools hold no server state: register() only needs the shared FastMCP
instance, and server.py re-exports them for in-process callers and tests.
Sibling imports stay inside function bodies because the tool ``llm_local``
would shadow the sibling module at module scope.
"""
import json
import logging
from typing import Optional

logger = logging.getLogger("loci-mcp")


def _coerce_labels(labels) -> list:
    """Normalize malformed tool input to a safe list without raising."""
    if labels is None:
        return []
    if isinstance(labels, list):
        return labels
    if isinstance(labels, (tuple, set)):
        return list(labels)
    return [labels]


def llm_local(prompt: str, model: str = "qwen2.5:3b", fmt: Optional[str] = None,
              max_tokens: int = 256, temperature: float = 0.2, keep_alive: str = "30m") -> str:
    """
    Generate with a local Ollama model for cheap high-volume work that should
    avoid Claude tokens. Verified-good model: ``qwen2.5:3b``.

    ``keep_alive`` keeps the model resident (default ``'30m'``) to avoid the
    ~70s cold load; keep it long on hot paths. ``fmt='json'`` constrains and
    validates JSON output. Fail-open: errors, timeouts, or invalid JSON return
    ``ok=False`` so callers can fall back upstream.

    Returns JSON ``{text, ok, model}``.
    """
    import llm_local as _llm
    return json.dumps(_llm.generate(prompt, model=model, fmt=fmt, max_tokens=max_tokens,
                                    temperature=temperature, keep_alive=keep_alive), indent=2)


def generate_batch(prompts: list, model: Optional[str] = None, max_tokens: int = 256,
                   fmt: Optional[str] = None) -> str:
    """
    Generate many prompts at once for fan-out stages.

    Uses a batched OpenAI-compatible server (vLLM/TGI at ``VLLM_BASE_URL``)
    when configured; otherwise fails open to sequential Ollama via
    ``llm_local``. Returns a JSON list of ``{text, ok}`` aligned 1:1 to
    ``prompts``. Failed prompts return ``{text:'', ok:False}``; the tool does
    not raise.
    """
    import batched_gen
    return json.dumps(batched_gen.generate_batch(list(prompts or []), model=model,
                                                 max_tokens=max_tokens, fmt=fmt), indent=2)


def query_expand(query: str, n_queries: int = 3, n_keywords: int = 6) -> str:
    """
    Expand a search query with local-model paraphrases and domain keywords
    before embedding retrieval. Runs on the GPU with near-zero Claude-token
    cost. Fail-open: if the local model is unavailable, returns the original
    query with ``degraded=True``.

    Returns JSON ``{queries, keywords, degraded}``.
    """
    import query_expand as _qe
    return json.dumps(_qe.expand(query, n_queries=n_queries, n_keywords=n_keywords), indent=2)


def verify_finding(claim: str, context: str = "", investigation_id: Optional[str] = None) -> str:
    """
    Adversarially verify a claim with a local-model skeptic: keep it only if the
    skeptic cannot refute it. Optional ``context`` can hold code, file refs, or
    other evidence. If ``context`` is empty and ``investigation_id`` is given,
    best-effort RAG grounding is pulled fail-open.

    Skeptical by default: returns ``confirmed`` only when the skeptic fails to
    refute the claim, otherwise ``refuted`` or ``uncertain``. If the model is
    unavailable or output is unparseable, returns ``verdict='uncertain'`` with
    ``degraded=True``.

    Returns JSON ``{verdict, refutation, confidence, degraded}``.
    """
    import verify as _v
    return json.dumps(_v.verify_finding(claim, context=context,
                                        investigation_id=investigation_id), indent=2)


def classify_text(text: str, labels: list) -> str:
    """
    Pick the best label from ``labels`` for ``text`` with the local model. This
    is a cheap gate/router in place of a classifier agent. Fail-open: returns
    ``label=None`` and ``degraded=True`` if the model is unavailable or emits an
    out-of-set label.

    Returns JSON ``{label, degraded}``.
    """
    import text_ops as _to
    return json.dumps(_to.classify(text, _coerce_labels(labels)), indent=2)


def compress_text(text: str, max_chars: int = 600) -> str:
    """
    Condense ``text`` to ``<= max_chars`` with the local model, e.g. before a
    Claude synthesis stage. Fail-open: if the model is unavailable, returns a
    character truncation with ``degraded=True``.

    Returns JSON ``{text, degraded}``.
    """
    import text_ops as _to
    return json.dumps(_to.compress(text, max_chars=max_chars), indent=2)


def semantic_dedup(items: list, threshold: float = 0.88, text_key: Optional[str] = None) -> str:
    """
    Cluster near-duplicate items by local embedding cosine similarity. No
    generation model is used, so token cost stays near zero. Use it after fanout
    so downstream synthesis sees one representative per cluster.

    ``items`` may be strings or dicts; dict text is pulled from ``text_key`` or
    from ``text/content/summary/title``. ``threshold`` is the duplicate floor
    (default ``0.88``; raise it to be stricter). Fail-open: if embeddings are
    unavailable, nothing is dropped and ``degraded=True``.

    Returns JSON ``{clusters:[{rep_index, member_indices, text}], kept:[...],
    dropped:int, degraded}``.
    """
    import embed_ops
    result = embed_ops.dedup(items or [], threshold=threshold, key=text_key)
    # Fail-open is preserved (nothing dropped), but make the silent degradation
    # observable: without embeddings, semantic_dedup returns every item unchanged.
    if result.get("degraded") and len(items or []) > 1:
        logger.warning("semantic_dedup degraded (embeddings unavailable) — %d items "
                       "returned unchanged, nothing deduped.", len(items or []))
    return json.dumps(result, indent=2)


def semantic_relevance(texts: list, topic: str) -> str:
    """
    Score each text's cosine relevance to ``topic`` on the local embedding path.
    Use it as a cheap gate/router before Claude. No generation model is used.

    Returns JSON ``{scores:[float|None], degraded}``; scores align with
    ``texts``. ``None`` means embeddings were unavailable and ``degraded=True``.
    """
    if not topic or not str(topic).strip():
        return json.dumps({"scores": [None] * len(texts or []), "degraded": True,
                           "error": "topic must not be empty"})
    import embed_ops
    return json.dumps(embed_ops.relevance(list(texts or []), topic), indent=2)


def ground(
    title: str,
    focus: str = "",
    case_ids: Optional[list] = None,
    entities: Optional[list] = None,
    code_refs: Optional[list] = None,
    budget_chars: int = 4000,
    allow_keyword: bool = False,
    graph_available: bool = False,
) -> str:
    """
    Build a compact, provenance-tagged grounding block for a task. Call it once
    before fan-out, then inject the block into every agent prompt so agents do
    not each re-query Loci.

    Retrieval is structured-first and embedding-independent where possible:
    named cases (``investigation_load``), exact entities
    (``investigation_entity_lookup``), code graph when ``graph_available``,
    semantic RAG, curated ``MEMORY.md``, then optional keyword FTS. Every lane
    is fail-open: dead sources set ``degraded=True`` instead of aborting.

    Prefer this to piecing together individual ``investigation_*`` or RAG calls
    when preparing a workflow. The block is read-only reference, not ground
    truth: consumers must still verify against live code/data and cite the
    ``[tag]`` they rely on.

    Args:
        title: Short task title (drives retrieval).
        focus: Optional longer task description.
        case_ids: Named investigation IDs to load.
        entities: Exact entity IDs to look up (O(1), no embedding).
        code_refs: Symbol names for code-graph grounding (used only if graph_available).
        budget_chars: Max characters in the assembled block (default 4000).
        allow_keyword: Enable the noisy keyword/FTS fallback lane (default off).
        graph_available: Enable the code-graph lane (default off; requires the
            LadybugDB graph).

    Returns:
        JSON ``{block, sources, chars, degraded}``.
    """
    if not title or not title.strip():
        return json.dumps({"error": "title must not be empty",
                           "block": "", "sources": [], "chars": 0, "degraded": True})
    import grounding
    task = {
        "title": title, "focus": focus or "",
        "caseIds": case_ids or [], "entities": entities or [],
        "codeRefs": code_refs or [],
    }
    opts = {
        "budgetChars": budget_chars,
        "allowKeyword": allow_keyword,
        "graphAvailable": graph_available,
    }
    return json.dumps(grounding.ground(task, opts), indent=2)


def register(mcp):
    """Register every local-model passthrough tool on the shared FastMCP instance."""
    for fn in (
        llm_local,
        generate_batch,
        query_expand,
        verify_finding,
        classify_text,
        compress_text,
        semantic_dedup,
        semantic_relevance,
        ground,
    ):
        mcp.tool()(fn)
