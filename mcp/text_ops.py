"""Generation-tier basic text ops for Loci-native workflows — the local-GPU offload.

Two cheap generation-backed ops that trim what has to reach Claude, running on the
local Ollama GPU (default qwen2.5:3b, or backends.ollama_gen_model()) at ~zero token cost:

- classify(text, labels): pick the single best label from a caller-supplied set —
  replaces a one-shot classifier/router agent. Always uses the shared gen_model:
  high call volume + low per-call stakes, so the fast default is the right tradeoff.
- compress(text, max_chars): semantically condense text under a hard char budget —
  shrink a bulky finding/context before it is forwarded. Independently routable via
  backends.ollama_compress_model() (LOCI_OLLAMA_COMPRESS_MODEL / [ollama].compress_model)
  since live adversarial benchmarking showed compress correctness varies far more across
  models than classify does — an operator can opt this one call site into a
  stronger/slower model without touching the shared default.

Both fail-open (NEVER raise): on timeout / HTTP error / bad-JSON / not-ok generation,
they return a well-formed degraded result. classify -> label=None, degraded=True;
compress -> text truncated to max_chars, degraded=True. This mirrors embed_ops.py.

The generation function is INJECTABLE (gen_fn). When None it is lazily imported at call
time (llm_local.generate) so importing this module never hard-requires llm_local — a
sibling module still being written in parallel. Tests pass a stub gen_fn and never touch
live Ollama. The shared gen_fn contract is:

    gen_fn(prompt:str, *, fmt:str|None=None, max_tokens:int=256) -> {"text":str, "ok":bool}

ok=False signals the caller (here: classify/compress) to fall back to the degraded path.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

GenFn = Callable[..., dict]


def _resolve_gen_fn(gen_fn: Optional[GenFn], model: str = "") -> Optional[GenFn]:
    """Return the injected gen_fn, else lazily import llm_local.generate.

    Lazy so that importing text_ops never requires llm_local to exist yet
    (it is written by a sibling agent in parallel). Returns None if it cannot
    be imported — callers treat that as the degraded path, never an exception.

    `model`, when set, is bound into the returned callable so a specific op (e.g.
    compress) can route to a model different from the shared gen_model without
    changing the gen_fn contract callers/tests rely on.
    """
    if gen_fn is not None:
        return gen_fn
    try:
        from llm_local import generate  # type: ignore
    except Exception:
        return None
    if not model:
        return generate

    def _bound(prompt: str, *, fmt: Optional[str] = None, max_tokens: int = 256) -> dict:
        return generate(prompt, model=model, fmt=fmt, max_tokens=max_tokens)

    return _bound


def _call_gen(gen_fn: GenFn, prompt: str, *, fmt: Optional[str] = None,
              max_tokens: int = 256) -> dict:
    """Invoke a gen_fn and normalize its result to {'text':str,'ok':bool}.

    Any exception / malformed return collapses to {'text':'','ok':False} so the
    op above can take its degraded path. Never raises.
    """
    try:
        out = gen_fn(prompt, fmt=fmt, max_tokens=max_tokens)
    except Exception:
        return {"text": "", "ok": False}
    if not isinstance(out, dict):
        return {"text": "", "ok": False}
    text = out.get("text")
    return {"text": text if isinstance(text, str) else "",
            "ok": bool(out.get("ok"))}


def classify(text: str, labels: list, gen_fn: Optional[GenFn] = None) -> dict:
    """Pick the single best label from `labels` for `text`.

    Returns {'label': str|None, 'degraded': bool}. The prompt asks for the label
    only; the returned string is validated to be one of `labels` (case-insensitive,
    mapped back to the canonical spelling). Fast path: if exactly one label is
    explicitly present in the input text, return it directly (no model call). This
    keeps short semantic routing tasks off the general generation path. Otherwise,
    classify uses a dedicated classify model override when configured. If generation
    is unavailable / not-ok / returns an out-of-set label, ->
    {'label': None, 'degraded': True}. Never raises.
    """
    text = text if isinstance(text, str) else ("" if text is None else str(text))
    labels = [str(l) for l in (labels or [])]
    if not text.strip() or not labels:
        return {"label": None, "degraded": True}

    mentioned = _single_explicit_label_mention(text, labels)
    if mentioned is not None:
        return {"label": mentioned, "degraded": False}

    gf = _resolve_gen_fn(gen_fn, model=_classify_model())
    if gf is None:
        return {"label": None, "degraded": True}

    label_list = ", ".join(labels)
    prompt = (
        "You are a strict single-label classifier. Choose the ONE label from this "
        "list that best fits the text. Reply with the label EXACTLY as written and "
        "NOTHING else.\n"
        f"Labels: {label_list}\n"
        f"Text: {text}\n"
        "Label:"
    )
    res = _call_gen(gf, prompt, fmt=None, max_tokens=32)
    if not res["ok"]:
        return {"label": None, "degraded": True}

    raw = res["text"].strip().strip("\"'`.").strip()
    # Exact match first, then case-insensitive map to canonical spelling.
    if raw in labels:
        return {"label": raw, "degraded": False}
    lowered = {l.lower(): l for l in labels}
    if raw.lower() in lowered:
        return {"label": lowered[raw.lower()], "degraded": False}
    # Out-of-set / noisy answer -> degraded.
    return {"label": None, "degraded": True}


def compress(text: str, max_chars: int = 600, gen_fn: Optional[GenFn] = None) -> dict:
    """Semantically condense `text` to <= max_chars.

    Returns {'text': str, 'degraded': bool}. If text is already within budget it is
    returned unchanged (degraded=False). Otherwise generation is asked to condense it;
    the result is hard-clamped to max_chars regardless. Fail-open: on unavailable /
    not-ok generation, or if the condensed text still exceeds budget, returns
    text[:max_chars] with degraded=True. Never raises.
    """
    text = text if isinstance(text, str) else ("" if text is None else str(text))
    try:
        max_chars = int(max_chars)
    except Exception:
        max_chars = 600
    if max_chars <= 0:
        return {"text": "", "degraded": True}
    if len(text) <= max_chars:
        return {"text": text, "degraded": False}

    gf = _resolve_gen_fn(gen_fn, model=_compress_model())
    if gf is None:
        return {"text": text[:max_chars], "degraded": True}

    prompt = (
        "Condense the following text so it preserves the key facts and meaning but "
        f"fits in at most {max_chars} characters. Reply with ONLY the condensed text.\n"
        f"Text: {text}\n"
        "Condensed:"
    )
    # Rough token budget: ~4 chars/token, with headroom, floored so tiny budgets still work.
    max_tokens = max(32, (max_chars // 3) + 16)
    res = _call_gen(gf, prompt, fmt=None, max_tokens=max_tokens)
    if not res["ok"]:
        return {"text": text[:max_chars], "degraded": True}

    condensed = res["text"].strip()
    if not condensed:
        return {"text": text[:max_chars], "degraded": True}
    if len(condensed) <= max_chars:
        return {"text": condensed, "degraded": False}
    # Model overran the budget -> clamp and flag degraded.
    return {"text": condensed[:max_chars], "degraded": True}


def _compress_model() -> str:
    """Best-effort model override for compress; "" means "use the shared gen_model".

    Live adversarial benchmarking (ab_eval_local_model.py --difficulty hard) showed
    compress correctness varies a lot more across models than classify does, so this is
    independently configurable via LOCI_OLLAMA_COMPRESS_MODEL / [ollama].compress_model.
    Never raises — any resolution failure just keeps the caller on the shared default.
    """
    try:
        import backends
        return backends.ollama_compress_model()
    except Exception:
        return ""


def _classify_model() -> str:
    """Best-effort model override for classify; "" means "use the shared gen_model".

    Classify is intentionally cheap/high-volume, but this knob allows operators to pin
    short semantic-routing tasks to a tiny specialist model without changing the shared
    generation tier used by unrelated call sites.
    """
    try:
        import backends
        return backends.ollama_classify_model()
    except Exception:
        return ""


def _single_explicit_label_mention(text: str, labels: list[str]) -> Optional[str]:
    """Return label when exactly one candidate is explicitly mentioned in text.

    Word-boundary matching is case-insensitive and supports multi-word labels.
    If none or multiple labels are present, returns None so the model classifier
    handles ambiguity.
    """
    low = text.lower()
    hits: list[str] = []
    for label in labels:
        item = str(label or "").strip()
        if not item:
            continue
        if re.search(r"(?<!\w)" + re.escape(item.lower()) + r"(?!\w)", low):
            hits.append(label)  # canonical spelling, as classify() documents
    if len(hits) == 1:
        return hits[0]
    return None
