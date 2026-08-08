"""Local-model / embedding passthrough MCP tools — split out of server.py.

Thin wrappers over the llm_local / batched_gen / query_expand / verify / text_ops /
embed_ops / grounding modules; they hold no server state, so register() only needs
the shared FastMCP instance. server.py re-exports the callables so `server.<tool>()`
keeps working for in-process callers and tests.

Each `import <sibling>` stays INSIDE its function body on purpose: the tool
`llm_local` shadows the sibling module of the same name at module scope, so
hoisting these imports to the top would break it.
"""
import json
import logging
from typing import Optional

logger = logging.getLogger("loci-mcp")


def llm_local(prompt: str, model: str = "qwen2.5:3b", fmt: Optional[str] = None,
              max_tokens: int = 256, temperature: float = 0.2, keep_alive: str = "30m") -> str:
    """
    Generate with a LOCAL model on the GPU (Ollama) — the generation tier of the offload
    hierarchy, for cheap high-volume ops (classify/expand/compress) that shouldn't spend
    Claude tokens. Verified-good model: qwen2.5:3b (sub-second warm, ~111 tok/s).

    keep_alive pins the model resident (default '30m') to avoid the ~70s cold load — keep it
    long for hot paths. Set fmt='json' to constrain + validate JSON output. Fail-open: on any
    error/timeout, or invalid JSON when fmt='json', returns ok=False (the caller should then
    fall back to a Claude model). Returns JSON {text, ok, model}.
    """
    import llm_local as _llm
    return json.dumps(_llm.generate(prompt, model=model, fmt=fmt, max_tokens=max_tokens,
                                    temperature=temperature, keep_alive=keep_alive), indent=2)


def generate_batch(prompts: list, model: Optional[str] = None, max_tokens: int = 256,
                   fmt: Optional[str] = None) -> str:
    """
    Generate for MANY prompts at once — for high-concurrency fan-out (per-item classify/expand
    gates, map stages). Uses a batched OpenAI-compatible server (vLLM/TGI at VLLM_BASE_URL,
    dispatched concurrently so continuous batching engages) when configured, else fails open to
    the sequential Ollama tier (llm_local). Returns JSON: a list of {text, ok} aligned 1:1 to
    `prompts` (a failed prompt is {text:'', ok:False}; never raises).
    """
    import batched_gen
    return json.dumps(batched_gen.generate_batch(list(prompts or []), model=model,
                                                 max_tokens=max_tokens, fmt=fmt), indent=2)


def query_expand(query: str, n_queries: int = 3, n_keywords: int = 6) -> str:
    """
    Expand a search query (HyDE-lite) using the LOCAL model — alternative phrasings + domain
    keywords to improve retrieval recall before an embedding search. Runs on the GPU, ~zero
    Claude tokens. Fail-open: if the local model is down, returns the original query with
    degraded=True. Returns JSON {queries, keywords, degraded}.
    """
    import query_expand as _qe
    return json.dumps(_qe.expand(query, n_queries=n_queries, n_keywords=n_keywords), indent=2)


def verify_finding(claim: str, context: str = "", investigation_id: Optional[str] = None) -> str:
    """
    Adversarially VERIFY a claim/finding using the LOCAL model — a skeptic actively tries to
    REFUTE it (candidate->skeptic->keep-if-survives), the same discipline workflows run per
    finding. Pass optional `context` (code snippet / file refs / evidence); if omitted and an
    `investigation_id` is given, best-effort RAG grounding is pulled (fail-open). Skeptical by
    default: only 'confirmed' when the skeptic cannot refute it, else 'refuted'/'uncertain'.
    Fail-open: if the local model is down or output is unparseable, returns verdict='uncertain'
    with degraded=True. Returns JSON {verdict, refutation, confidence, degraded}.
    """
    import verify as _v
    return json.dumps(_v.verify_finding(claim, context=context,
                                        investigation_id=investigation_id), indent=2)


def classify_text(text: str, labels: list) -> str:
    """
    Pick the single best label from `labels` for `text` using the LOCAL model — a cheap
    gate/router that replaces a classifier agent. Fail-open: label=None + degraded=True if the
    model is down or returns an out-of-set label. Returns JSON {label, degraded}.
    """
    import text_ops as _to
    return json.dumps(_to.classify(text, list(labels or [])), indent=2)


def compress_text(text: str, max_chars: int = 600) -> str:
    """
    Semantically condense `text` to <= max_chars using the LOCAL model — e.g. shrink a long
    agent output before a Claude synthesis stage (saves Claude input tokens). Fail-open:
    returns a char-truncation + degraded=True if the model is down. Returns JSON {text, degraded}.
    """
    import text_ops as _to
    return json.dumps(_to.compress(text, max_chars=max_chars), indent=2)


def semantic_dedup(items: list, threshold: float = 0.88, text_key: Optional[str] = None) -> str:
    """
    Cluster near-duplicate items by embedding cosine similarity on the local-GPU path —
    no generation model, ~zero token cost. Use in a fan-out's synthesis step so an N-way
    search doesn't triple-report the same finding: pass the aggregated items, feed the
    returned `kept` (one representative per cluster) downstream.

    items: list of strings OR dicts (text pulled from text_key, else text/content/summary/title).
    threshold: cosine >= this counts as a duplicate (default 0.88; raise to be stricter).
    Fail-open: if embeddings are unavailable, nothing is dropped and degraded=True.

    Returns JSON {clusters:[{rep_index, member_indices, text}], kept:[...], dropped:int, degraded}.
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
    Cosine relevance of each text to `topic` on the local-GPU embedding path — a cheap
    gate/router (keep texts above a score) that trims what reaches Claude, replacing a
    classifier agent. No generation model.

    Returns JSON {scores:[float|None], degraded}; scores align with `texts` (None when
    embeddings are unavailable, degraded=True).
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
    Assemble a compact, provenance-tagged, char-budgeted GROUNDING block for a task —
    run ONCE in an orchestrator before a fan-out and inject the block into every agent
    prompt, so agents start with relevant prior context instead of each re-querying Loci
    (the cost win). Structured-first, embedding-independent retrieval order: named cases
    (investigation_load) -> exact entities (investigation_entity_lookup) -> code graph
    (when graph_available) -> semantic RAG -> curated MEMORY.md -> keyword FTS (opt-in).
    Every lane is fail-open: a dead source sets degraded=True rather than aborting.

    Prefer this over calling the individual investigation_*/rag tools when preparing a
    workflow — one warm call here beats N cold ones (and keeps the cross-encoder loaded,
    which the ground.py CLI cannot). The block is tagged read-only reference, NOT ground
    truth: consumers must verify against live code/data and cite the [tag] they rely on.

    Args:
        title: Short task title (drives retrieval).
        focus: Optional longer task description.
        case_ids: Named investigation IDs to load.
        entities: Exact entity IDs to look up (O(1), no embedding).
        code_refs: Symbol names for code-graph grounding (used only if graph_available).
        budget_chars: Max characters of the assembled block (default 4000).
        allow_keyword: Enable the noisy keyword/FTS fallback lane (default off).
        graph_available: Enable the code-graph lane (default off; needs the Kuzu graph).

    Returns:
        JSON with {block, sources, chars, degraded}.
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
