"""Decision-model client: typed questions in, calibrated probabilities out, no generation.

A decision model (OpenThai-SystemOne, 0.8B) does not write text. Given a context and typed questions
(``choice``: pick an option, ``noul``: yes/no, ``score``: a position on an ordered scale) it answers with
probabilities over the options in one forward pass. Ollama 0.35+ serves that at ``/v1/systemone``; older
servers can run the same GGUF through ``/api/chat``, which is what this module does. It reproduces
ollama/ollama v0.35.1 ``decision/systemone.go``: one chat prompt per question holding the compact JSON of
``{"context", "schema"}`` (the schema of EVERY question, in order) plus ``Requested field: "<name>"``, scored
on the next-token probabilities of the answer letters ``A``..``Z`` that label the options.

Measured on 2026-10-06 (docs/decision_model.md): 12/12 on easy cases; 69% on telling a harness refusal from a
real tool failure in 36 real error strings (about 86% once six disputed labels went to the model), and
over-confident when wrong. Treat ``confidence`` as a ranking signal and route low or contested answers to a
person or a bigger model; do not treat it as a probability of being right.

Fail-open like ``llm_local``: nothing here raises; a problem returns ``{"ok": False, "degraded": True, ...}``.
The model comes from the pool role ``decide`` (``LOCI_DECIDE_MODEL`` overrides), the card from the pool's
``gpu`` key, the server from ``backends.ollama_gen_url()``. Calls are logged for the pool's outcome log.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from typing import Any, Callable, Optional

_LOG = logging.getLogger("loci-mcp.decide")

ROLE = "decide"
MODEL_ENV = "LOCI_DECIDE_MODEL"
MAX_QUESTIONS = 64
MIN_OPTIONS, MAX_OPTIONS = 2, 26
TOP_LOGPROBS = 20
_KINDS = ("choice", "noul", "score")


class DecideError(ValueError):
    """A request the decision model cannot be asked (shape, size or type)."""


# ---------------------------------------------------------------------------------- rendering

def _go_json(value: Any) -> str:
    """``json.Marshal`` as Go writes it: compact, UTF-8, and <, >, & and U+2028/9 escaped."""
    text = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    return (text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def _content(value: Any, what: str) -> str:
    """A string stays text; an object or array becomes compact JSON; anything else is not content."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    raise DecideError(f"{what} must be a string, object, or array")


def _options(name: str, kind: str, criteria: Any) -> list[tuple[Any, Any]]:
    """The ordered ``(value, description)`` options of one question."""
    if kind == "noul":
        c = criteria if criteria is not None else {}
        if not isinstance(c, dict):
            raise DecideError(f"question {name!r}: noul criteria must be an object of true/false descriptions")
        unknown = [k for k in c if k not in ("true", "false")]
        if unknown:
            raise DecideError(f"question {name!r}: unknown noul criterion {unknown[0]!r}")
        if any(not isinstance(v, str) for v in c.values()):
            raise DecideError(f"question {name!r}: noul descriptions must be strings")
        return [(False, c.get("false", "No")), (True, c.get("true", "Yes"))]
    if kind == "choice":
        if not isinstance(criteria, dict):
            raise DecideError(f"question {name!r}: choice criteria must map option keys to descriptions or null")
        out = []
        for key, desc in criteria.items():
            if not str(key).strip():
                raise DecideError(f"question {name!r}: choice keys must not be empty")
            out.append((str(key), key if desc is None else str(desc)))
        return out
    if kind == "score":
        if not isinstance(criteria, list) or any(not isinstance(d, str) for d in criteria):
            raise DecideError(f"question {name!r}: score criteria must be an array of descriptions")
        return [(str(i), d) for i, d in enumerate(criteria)]
    raise DecideError(f"question {name!r}: type must be choice, noul, or score")


def compile_request(state: Any, questions: dict) -> dict:
    """Validate a request and render its prompts.

    ``state`` is text, an object or an array. ``questions`` maps a name to
    ``{"type": "choice"|"noul"|"score", "instructions": ..., "criteria": ...}``. Returns
    ``{"fields": [...], "prompts": {name: user message}}``; raises ``DecideError`` on a bad request.
    """
    if not isinstance(questions, dict) or not 1 <= len(questions) <= MAX_QUESTIONS:
        raise DecideError(f"questions must contain 1-{MAX_QUESTIONS} fields")
    context = _content(state, "state")
    if not context.strip():
        raise DecideError("state must not be empty")
    fields = []
    for name, q in questions.items():
        if not isinstance(name, str) or not name.strip():
            raise DecideError("field name must not be empty")
        if not isinstance(q, dict):
            raise DecideError(f"question {name!r}: must be an object")
        kind = q.get("type")
        if kind not in _KINDS:
            raise DecideError(f"question {name!r}: type must be choice, noul, or score")
        if q.get("instructions") is None:
            raise DecideError(f"question {name!r}: instructions must be a nonempty string, object, or array")
        description = _content(q["instructions"], f"question {name!r}: instructions")
        if not description.strip():
            raise DecideError(f"question {name!r}: instructions must be a nonempty string, object, or array")
        opts = _options(name, kind, q.get("criteria"))
        if not MIN_OPTIONS <= len(opts) <= MAX_OPTIONS:
            raise DecideError(f"question {name!r}: criteria must contain {MIN_OPTIONS}-{MAX_OPTIONS} candidates")
        fields.append({"name": name, "kind": kind, "description": description,
                       "choices": [{"code": chr(65 + i), "value": v, "description": d} for i, (v, d) in enumerate(opts)]})
    wire = [{"name": f["name"], "description": f["description"], "choices": f["choices"]} for f in fields]
    data = _go_json({"context": context, "schema": wire})
    prompts = {f["name"]: data + "\n\nRequested field: " + _go_json(f["name"]) for f in fields}
    return {"fields": fields, "prompts": prompts}


# ---------------------------------------------------------------------------------- answering

def _logsumexp(values: list[float]) -> float:
    peak = max(values)
    return peak + math.log(sum(math.exp(v - peak) for v in values))


def letter_logprobs(top_logprobs: list) -> dict[str, float]:
    """``{"A": logprob, ...}`` from a token's top-logprobs list; ``"A"`` and ``" A"`` are one answer."""
    seen: dict[str, list[float]] = {}
    for entry in top_logprobs or []:
        try:
            tok = str(entry["token"]).strip()
            lp = float(entry["logprob"])
        except (KeyError, TypeError, ValueError):
            continue
        if len(tok) == 1 and tok.isalpha() and tok.isupper() and tok.isascii() and math.isfinite(lp):
            seen.setdefault(tok, []).append(lp)
    return {k: _logsumexp(v) for k, v in seen.items()}


def answer_field(field: dict, letters: dict[str, float], floor: Optional[float] = None) -> dict:
    """The answer to one compiled question from the answer-letter logprobs.

    A letter missing from the top logprobs is not impossible, only no likelier than the 20th token: with
    ``floor`` (that token's logprob) it is scored at the floor, an upper bound, so confidence is not overstated
    when the model's top 20 held a single letter. Without ``floor`` a missing letter has probability 0, which is
    what Ollama's own scorer reads when it has every candidate's logit. Raises ``DecideError`` if the model put
    none of the question's letters in its top logprobs (it was not answering with a letter).
    """
    codes = [c["code"] for c in field["choices"]]
    present = [c for c in codes if c in letters]
    if not present:
        raise DecideError("none of the answer letters is in the model's top logprobs")
    absent = -math.inf if floor is None else floor
    scores = [letters.get(c, absent) for c in codes]
    peak = max(scores)
    p = [math.exp(s - peak) for s in scores]
    total = sum(p)
    p = [x / total for x in p]
    entropy = -sum(x * math.log(x) for x in p if x > 0)
    confidence = max(0.0, min(1.0, 1 - entropy / math.log(len(p))))
    # Probability the model put on valid answer letters. Low is normal for this model: it was trained on a loss
    # over the candidate letters only, so it never learned to put mass on them among the whole vocabulary.
    mass = sum(math.exp(min(letters[c], 0.0)) for c in present)       # logprobs are <= 0; raw logits are clamped
    values = [c["value"] for c in field["choices"]]
    kind = field["kind"]
    if kind == "noul":
        return {"type": "noul", "noul": p[values.index(True)], "letter_mass": mass}
    probs = {str(v): p[i] for i, v in enumerate(values)}
    if kind == "choice":
        return {"type": "choice", "choice": str(values[p.index(max(p))]), "probabilities": probs,
                "confidence": confidence, "letter_mass": mass}
    legend = {str(v): field["choices"][i]["description"] for i, v in enumerate(values)}
    return {"type": "score", "score": sum(i * x for i, x in enumerate(p)), "legend": legend,
            "probabilities": probs, "confidence": confidence, "letter_mass": mass}


# ---------------------------------------------------------------------------------- the call

def _resolve_model(model: str) -> str:
    if model:
        return model
    env = (os.environ.get(MODEL_ENV) or "").strip()
    if env:
        return env
    try:
        import model_pool
        return model_pool.pick(ROLE)
    except Exception as exc:
        _LOG.debug("decide: pool pick skipped: %r", exc)
        return ""


def _resolve_base(base_url: str) -> str:
    if base_url:
        return base_url.rstrip("/")
    try:
        import backends
        return (backends.ollama_gen_url() or "").rstrip("/")
    except Exception as exc:
        _LOG.debug("decide: base url skipped: %r", exc)
        return ""


def _placement(model: str) -> dict:
    try:
        import model_pool
        return model_pool.options_for(model)
    except Exception:
        return {}


def _http_post(url: str, body: dict, timeout: float) -> dict:
    import requests
    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _record(model: str, ok: bool, started: float, prompt_chars: int) -> None:
    try:
        import model_pool
        model_pool.record_outcome(model, ok, (time.monotonic() - started) * 1000.0, route_role=ROLE,
                                  prompt_chars=prompt_chars)
    except Exception as exc:
        _LOG.debug("decide: outcome log skipped: %r", exc)


def decide(state: Any, questions: dict, *, model: str = "", base_url: str = "", timeout: float = 60.0,
           keep_alive: str = "30m", post: Optional[Callable[[str, dict, float], dict]] = None) -> dict:
    """Ask every question. Returns ``{"ok", "degraded", "model", "answers", "error", "ms"}``.

    ``answers`` maps each question name to its answer (see ``answer_field``) or ``{"error": ...}``;
    ``ok`` is True only when every question was answered. ``post(url, body, timeout)`` is injectable for tests.
    """
    started = time.monotonic()
    out: dict = {"ok": False, "degraded": True, "model": "", "answers": {}, "error": None, "ms": 0.0}

    def done(error: Optional[str] = None) -> dict:
        out["error"] = error or out["error"]
        out["ms"] = round((time.monotonic() - started) * 1000.0, 1)
        return out
    try:
        compiled = compile_request(state, questions)
    except DecideError as exc:
        return done(str(exc))
    except Exception as exc:  # noqa: BLE001 - fail-open
        return done(f"request could not be compiled: {exc}")
    mdl = _resolve_model(model)
    base = _resolve_base(base_url)
    if not mdl:
        return done(f"no decision model: add a pool entry with roles = [\"{ROLE}\"] or set {MODEL_ENV}")
    if not base:
        return done("no Ollama URL configured")
    out["model"] = mdl
    sender = post or _http_post
    options = {"temperature": 0, "num_predict": 1, **_placement(mdl)}
    errors = []
    for f in compiled["fields"]:
        user = compiled["prompts"][f["name"]]
        # think=False: without it this server renders a thinking-mode prompt (an open <think> tag), not the closed
        # empty think block the model was trained on, and the letter probabilities fall five-fold.
        body = {"model": mdl, "stream": False, "think": False, "keep_alive": keep_alive, "logprobs": True,
                "top_logprobs": TOP_LOGPROBS, "options": options,
                "messages": [{"role": "user", "content": user}]}
        t0 = time.monotonic()
        try:
            resp = sender(f"{base}/api/chat", body, timeout)
            first = (resp.get("logprobs") or [{}])[0]
            tops = first.get("top_logprobs") or []
            lps = [float(e["logprob"]) for e in tops if isinstance(e, dict) and "logprob" in e]
            floor = min(lps) if len(tops) >= TOP_LOGPROBS and lps else None   # a short list means nothing was cut off
            out["answers"][f["name"]] = answer_field(f, letter_logprobs(tops), floor)
            _record(mdl, True, t0, len(user))
        except Exception as exc:  # noqa: BLE001 - fail-open: one bad question must not sink the rest
            out["answers"][f["name"]] = {"error": str(exc)[:200]}
            errors.append(f"{f['name']}: {str(exc)[:120]}")
            _record(mdl, False, t0, len(user))
    out["ok"] = not errors
    out["degraded"] = bool(errors)
    return done("; ".join(errors) if errors else None)


def yes_no(state: Any, question: str, *, true_means: str = "", false_means: str = "", **kw) -> Optional[float]:
    """P(yes) for one yes/no question, or None when the model could not answer."""
    criteria = {k: v for k, v in (("true", true_means), ("false", false_means)) if v}
    res = decide(state, {"q": {"type": "noul", "instructions": question, "criteria": criteria or None}}, **kw)
    ans = res["answers"].get("q") or {}
    return ans.get("noul") if res["ok"] else None


def choose(state: Any, question: str, options: dict, **kw) -> Optional[dict]:
    """The ``choice`` answer for one question (``choice``, ``probabilities``, ``confidence``), or None."""
    res = decide(state, {"q": {"type": "choice", "instructions": question, "criteria": options}}, **kw)
    return res["answers"]["q"] if res["ok"] else None


# ---------------------------------------------------------------------------------- self-check

CHECKS = (
    ("choice", "The app charged me twice for the same order, refund me.", "Which team should handle this?",
     {"billing": "payments", "technical": "the app is broken", "sales": "buying"}, "billing"),
    ("choice", "The checkout page crashes with error 500 every time I press pay.", "Which team should handle this?",
     {"billing": "payments", "technical": "the app is broken", "sales": "buying"}, "technical"),
    ("noul", "You are a complete idiot and everyone here hates you.", "Is this message toxic or insulting?", None, True),
    ("noul", "Thanks so much, that fixed my problem!", "Is this message toxic or insulting?", None, False),
    ("noul", "Passage: The bridge opened in 2002 and is 475 m long.", "Does the passage state the bridge's length?", None, True),
    ("noul", "Passage: The bridge opened in 2002 and is 475 m long.", "Does the passage state the bridge's construction budget?", None, False),
    ("choice", "Premise: A man is playing a guitar on stage. Hypothesis: Nobody is on the stage.",
     "How does the hypothesis relate to the premise?",
     {"entailment": "follows from the premise", "contradiction": "conflicts with it", "neutral": "cannot tell"}, "contradiction"),
    ("choice", "The unit tests failed after my change to the parser; the diff only touched parser.py.",
     "What kind of problem is this?",
     {"regression": "a change broke working behaviour", "flaky": "intermittent for no clear reason",
      "config": "environment or setup problem"}, "regression"),
)


def check(**kw) -> dict:
    """Run the built-in easy questions against the live model. ``{"ok", "passed", "total", "failures"}``."""
    failures = []
    for kind, state, question, criteria, expected in CHECKS:
        spec = {"type": kind, "instructions": question}
        if criteria:
            spec["criteria"] = criteria
        res = decide(state, {"q": spec}, **kw)
        ans = res["answers"].get("q") or {}
        got = ans.get("choice") if kind == "choice" else (None if "noul" not in ans else ans["noul"] >= 0.5)
        if got != expected:
            failures.append({"question": question[:60], "expected": expected, "got": got, "error": res["error"]})
    return {"ok": not failures, "passed": len(CHECKS) - len(failures), "total": len(CHECKS), "failures": failures}


def _main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "check"
    if cmd == "check":
        res = check()
        print(json.dumps(res, indent=1))
        return 0 if res["ok"] else 1
    if cmd == "ask" and len(argv) >= 4:
        state, question = argv[1], argv[2]
        options = dict(a.split("=", 1) if "=" in a else (a, None) for a in argv[3:])
        print(json.dumps(choose(state, question, options) or {"error": "no answer"}, indent=1))
        return 0
    print("usage: decide.py check | decide.py ask <state> <question> <key[=description]>...")
    return 2


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(_main(sys.argv[1:]))
