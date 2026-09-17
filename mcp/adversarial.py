"""Adversarial review of a finding set — red-team and gap-analysis as a Loci primitive.

Where verify_finding takes ONE claim and tries to refute it, this module takes a SET of
findings and does the two adversarial passes that a refutation loop does not:

- ``redteam``: per finding, a blunt offensive-security critique — is it a real weakness,
  and if so the concrete attack path, preconditions, impact, and how to confirm it. This
  deliberately routes to backends.ollama_redteam_model() (an uncensored/abliterated local
  model by default) because aligned instruct models soften or refuse "attack this" prompts,
  which is exactly the grunt work we want off the hosted model and onto a local one.
- ``gaps``: one pass over the whole set — a completeness critic that names attack surface
  not examined, findings asserted but never verified, and the highest-value next probes.

Refutation is intentionally NOT reimplemented here: for "is this single claim true?" use
verify_finding, which already runs the skeptic loop with grounding.

Design mirrors mcp/verify.py:
- Reasoning runs on the generation tier (Ollama), injectable so a warm client can be reused
  and tests can stub it. ``gen_fn`` defaults to None; when None we LAZILY import
  ``llm_local.generate`` at call time so importing this module never hard-requires it. The
  default routes through backends.ollama_redteam_model(), independently configurable via
  LOCI_OLLAMA_REDTEAM_MODEL / [ollama].redteam_model.
  gen_fn contract (shared): gen_fn(prompt, *, fmt=None, max_tokens=256) -> {"text": str,
  "ok": bool}. ok=False signals the caller should fall back — we treat it as degraded.
- Fail-open: on not-ok / timeout / parse failure / any error we return a well-formed
  degraded result rather than raising. A single finding's failure never sinks the batch.

Returns a dict; the MCP wrapper in llm_tools.py serializes it to JSON.
"""
import logging
from typing import Callable, Optional

from model_json import extract_json_object

logger = logging.getLogger("loci-mcp")

# Injectable generation fn: match the shared contract used across the local-model tools.
GenFn = Callable[..., dict]

# Caps so an oversized/hostile input can't blow up the prompt or fan out unboundedly.
_MAX_FINDINGS = 50
_MAX_FINDING_LEN = 2000
_MAX_CONTEXT_LEN = 6000
_VALID_MODES = ("redteam", "gaps")

_REDTEAM_TMPL = (
    "You are a blunt offensive-security engineer red-teaming a system. Be concrete and\n"
    "technical. Do NOT moralize, hedge, or add disclaimers.{domain}\n\n"
    "Given ONE finding about the target, decide whether it is a real, exploitable weakness.\n"
    "If it is, give the attack. If it is not, say so plainly.\n\n"
    "Respond with ONLY a JSON object of this exact shape, no prose:\n"
    '{{"exploitable": true, "attack": "the concrete attack path", "preconditions": "access '
    'or state an attacker needs", "impact": "realistic impact", "confirm": "how to confirm '
    'this against the dump/artifact"}}\n\n'
    "FINDING:\n{finding}\n\n"
    "CONTEXT (may be empty — do not assume it is complete):\n{context}\n"
)

_GAPS_TMPL = (
    "You are an adversarial reviewer auditing the COMPLETENESS of a security analysis. Be\n"
    "specific and technical. Do NOT moralize or hedge.{domain}\n\n"
    "Here is the full list of findings the primary analysis produced. Identify what is\n"
    "MISSING: attack surface never examined, findings asserted but never verified, likely\n"
    "secrets/endpoints/logic the primary pass would skip or self-censor, and the\n"
    "highest-value next probes.\n\n"
    "Respond with ONLY a JSON object of this exact shape, no prose:\n"
    '{{"gaps": ["specific gap", "..."], "next_probes": ["specific probe", "..."], '
    '"summary": "one-paragraph adversarial assessment"}}\n\n'
    "FINDINGS:\n{findings}\n\n"
    "CONTEXT (may be empty):\n{context}\n"
)


def _coerce_findings(findings) -> list:
    """Normalize tool input to a bounded list of non-empty strings without raising."""
    if findings is None:
        return []
    if isinstance(findings, str):
        findings = [findings]
    elif isinstance(findings, (tuple, set)):
        findings = list(findings)
    elif not isinstance(findings, list):
        findings = [findings]
    out = []
    for f in findings:
        if isinstance(f, dict):
            f = "; ".join(f"{k}={v}" for k, v in f.items()
                          if isinstance(v, (str, int, float)))
        s = str(f).strip()
        if s:
            out.append(s[:_MAX_FINDING_LEN])
        if len(out) >= _MAX_FINDINGS:
            break
    return out


def _domain_clause(domain: str) -> str:
    d = (domain or "").strip()
    return f" Target domain: {d[:200]}." if d else ""


def _lazy_generate(prompt: str, *, fmt: Optional[str] = None, max_tokens: int = 256) -> dict:
    """Default gen_fn: import llm_local.generate only when called (fail-open).

    Routes through backends.ollama_redteam_model() — an uncensored/abliterated model by
    default — because the red-team framing is where aligned models soften or refuse.
    Independently configurable via LOCI_OLLAMA_REDTEAM_MODEL / [ollama].redteam_model.
    """
    try:
        from llm_local import generate  # lazy so module import never requires llm_local
        try:
            import backends
            model = backends.ollama_redteam_model()
        except Exception:
            model = ""
        return generate(prompt, model=model, fmt=fmt, max_tokens=max_tokens)
    except Exception:
        return {"text": "", "ok": False}


def _redteam_model_name() -> str:
    try:
        import backends
        return backends.ollama_redteam_model()
    except Exception:
        return ""


def _redteam(findings: list, context: str, domain: str, gen_fn: GenFn) -> dict:
    ctx = (context or "").strip()[:_MAX_CONTEXT_LEN]
    dom = _domain_clause(domain)
    results = []
    any_ok = False
    for f in findings:
        prompt = _REDTEAM_TMPL.format(domain=dom, finding=f, context=ctx or "(none)")
        try:
            res = gen_fn(prompt, fmt="json", max_tokens=400)
        except Exception:
            res = {"text": "", "ok": False}
        if not isinstance(res, dict) or not res.get("ok"):
            results.append({"finding": f, "exploitable": None, "attack": "",
                            "preconditions": "", "impact": "", "confirm": "",
                            "degraded": True})
            continue
        obj = extract_json_object(res.get("text", ""))
        if obj is None:
            results.append({"finding": f, "exploitable": None, "attack": "",
                            "preconditions": "", "impact": "", "confirm": "",
                            "degraded": True})
            continue
        any_ok = True
        results.append({
            "finding": f,
            "exploitable": obj.get("exploitable"),
            "attack": str(obj.get("attack", "")).strip(),
            "preconditions": str(obj.get("preconditions", "")).strip(),
            "impact": str(obj.get("impact", "")).strip(),
            "confirm": str(obj.get("confirm", "")).strip(),
            "degraded": False,
        })
    return {"mode": "redteam", "model": _redteam_model_name(), "results": results,
            "degraded": not any_ok}


def _gaps(findings: list, context: str, domain: str, gen_fn: GenFn) -> dict:
    ctx = (context or "").strip()[:_MAX_CONTEXT_LEN]
    dom = _domain_clause(domain)
    listing = "\n".join(f"- {f}" for f in findings)
    prompt = _GAPS_TMPL.format(domain=dom, findings=listing, context=ctx or "(none)")
    try:
        res = gen_fn(prompt, fmt="json", max_tokens=900)
    except Exception:
        res = {"text": "", "ok": False}
    if not isinstance(res, dict) or not res.get("ok"):
        return {"mode": "gaps", "model": _redteam_model_name(), "gaps": [],
                "next_probes": [], "summary": "", "degraded": True}
    obj = extract_json_object(res.get("text", ""))
    if obj is None:
        return {"mode": "gaps", "model": _redteam_model_name(), "gaps": [],
                "next_probes": [], "summary": "", "degraded": True}

    def _strlist(v):
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str) and v.strip():
            return [v.strip()]
        return []

    return {"mode": "gaps", "model": _redteam_model_name(),
            "gaps": _strlist(obj.get("gaps")),
            "next_probes": _strlist(obj.get("next_probes")),
            "summary": str(obj.get("summary", "")).strip(),
            "degraded": False}


def adversarial_review(findings, mode: str = "redteam", context: str = "",
                       domain: str = "", gen_fn: Optional[GenFn] = None) -> dict:
    """Run an adversarial pass over a set of findings. Never raises.

    Args:
        findings: a finding string or list of them (dicts are flattened to k=v text).
        mode: ``redteam`` (per-finding attack critique) or ``gaps`` (completeness critic
            over the whole set). Unknown values fall back to ``redteam``.
        context: optional shared grounding for every finding (evidence, device profile).
        domain: optional short target descriptor to sharpen the critique.
        gen_fn: injectable generation fn (shared contract). None -> lazy
            llm_local.generate on the red-team model tier.

    Returns:
        redteam -> {"mode","model","results":[{finding,exploitable,attack,preconditions,
                    impact,confirm,degraded}],"degraded"}
        gaps    -> {"mode","model","gaps":[...],"next_probes":[...],"summary","degraded"}
        Fail-open: an empty finding set or a dead backend returns a well-formed degraded
        result. ``degraded`` is True when the model produced nothing usable.
    """
    items = _coerce_findings(findings)
    m = mode if mode in _VALID_MODES else "redteam"
    if not items:
        if m == "gaps":
            return {"mode": "gaps", "model": _redteam_model_name(), "gaps": [],
                    "next_probes": [], "summary": "", "degraded": True}
        return {"mode": "redteam", "model": _redteam_model_name(), "results": [],
                "degraded": True}
    fn = gen_fn or _lazy_generate
    if m == "gaps":
        return _gaps(items, context, domain, fn)
    return _redteam(items, context, domain, fn)
