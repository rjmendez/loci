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

  The corpus is MULTI-REPO, so a ref names a file in some checkout on the host, not
  necessarily this one — see _search_roots for the allow-list and _ground_ref for the rule
  that decides WHICH checkout, refusing rather than guessing when worktrees disagree.

- Fail-open + skeptical default: on not-ok / timeout / parse failure / any error we return a
  well-formed {"verdict": "uncertain"} result rather than raising. We also default to the
  cautious verdict when the model is unsure — a claim is only 'confirmed' when the skeptic
  explicitly fails to refute it. Never raises.

Grounding: [pattern:injectable] lazy llm_local import + injectable gen_fn; [interface] the
gen_fn contract above; [pattern:fail-open] degraded 'uncertain' fallback. The skeptic prompt
copy and the JSON schema {verdict,refutation,confidence} are this task's design.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from typing import Callable, Optional

from model_json import extract_json_object

# Private alias kept so existing call sites and test monkeypatches of the private
# name keep working after the shared extraction moved to mcp/model_json.py.
_extract_json_object = extract_json_object

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
# Size cap on a file we will HASH for revision matching (must cover whole-file bytes, so it
# is larger than the display cap). Mirrors server._hash_file_bytes' own cap.
_MAX_HASH_BYTES = 8 * 1024 * 1024
# Bound on how many checkouts we will search, so a home directory full of repos can't turn
# one ref into thousands of stat calls.
_MAX_SEARCH_ROOTS = 256

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


def _safe_resolve(path: str, root: Optional[str] = None) -> Optional[str]:
    """Resolve a relative ref path to an absolute path UNDER ``root`` (default: repo root).

    SECURITY: file refs are parsed from free-form (attacker-influenceable) claim/context
    text, so the default reader must never read arbitrary files. Rejects absolute paths,
    any ``..`` traversal segment, and any DOT-PREFIXED path component (``.ssh/id_rsa``,
    ``.env``, ``.git/config`` — credential stores that are not the source we ground on),
    resolves relative to the root, and returns None for anything that still lands outside
    that root (e.g. via a symlink). Returns None on reject (caller fails open by
    contributing no code). Never raises.

    ``root`` is chosen by the caller from the vetted set in _search_roots(); it is never
    taken from the ref itself, so widening the search does not widen what a ref may name.
    """
    if not isinstance(path, str) or not path:
        return None
    if os.path.isabs(path):
        return None
    parts = path.replace("\\", "/").split("/")
    if ".." in parts:
        return None
    if any(p.startswith(".") for p in parts if p):
        return None
    try:
        base = os.path.realpath(root) if root else _repo_root()
        full = os.path.realpath(os.path.join(base, path))
        # realpath collapses symlinks/traversal; require the result to stay under the root.
        if full == base or full.startswith(base + os.sep):
            return full
    except Exception:
        return None
    return None


def _root_parents() -> list:
    """Directories scanned for sibling checkouts. LOCI_CODE_ROOT_PARENTS (os.pathsep-
    separated) overrides; default is $HOME plus the repo root's parent. Never raises."""
    try:
        raw = os.environ.get("LOCI_CODE_ROOT_PARENTS")
        if raw:
            return [p for p in raw.split(os.pathsep) if p]
        out = []
        for cand in (os.path.expanduser("~"), os.path.dirname(_repo_root())):
            if cand and cand not in out:
                out.append(cand)
        return out
    except Exception:
        return []


def _search_roots() -> tuple:
    """The vetted checkout roots a ref may resolve under, primary repo root FIRST.

    loci_memory is a MULTI-REPO corpus: findings cite ``perception/depth.py`` recorded
    against a hugbot5000 checkout, not against this repo, so sandboxing every ref to
    _repo_root() rejected 100% of them (measured: 62 stored code_refs, 0 resolvable).

    The set is the repo root, any root named explicitly in LOCI_CODE_ROOTS, and the git
    WORKING TREES that are immediate children of _root_parents(). Requiring a ``.git``
    marker is what keeps this an allow-list rather than "$HOME": ``~/.ssh`` and friends are
    never roots, and _safe_resolve still refuses absolute paths, ``..`` and dot-components
    under every one of them. Cached; call ``_search_roots.cache_clear()`` after changing
    the env. Never raises.
    """
    roots = []
    seen = set()

    def _add(d):
        try:
            real = os.path.realpath(d)
        except Exception:
            return
        if real in seen or not os.path.isdir(real):
            return
        seen.add(real)
        roots.append(real)

    try:
        _add(_repo_root())
        for d in (os.environ.get("LOCI_CODE_ROOTS") or "").split(os.pathsep):
            if d:
                _add(d)
        for parent in _root_parents():
            try:
                entries = sorted(os.scandir(parent), key=lambda e: e.name)
            except Exception:
                continue
            for e in entries:
                if len(roots) >= _MAX_SEARCH_ROOTS:
                    break
                try:
                    if e.name.startswith(".") or not e.is_dir(follow_symlinks=False):
                        continue
                    if os.path.exists(os.path.join(e.path, ".git")):
                        _add(e.path)
                except Exception:
                    continue
    except Exception:
        pass
    return tuple(roots[:_MAX_SEARCH_ROOTS])


_search_roots = lru_cache(maxsize=1)(_search_roots)


def _file_sha256(full: str) -> Optional[str]:
    """sha256 of a file's whole bytes, or None (too large / unreadable). Never raises.

    Must hash the WHOLE file to be comparable with the ``hash`` server.py stamped into a
    finding's code_refs; the display cap (_MAX_FILE_BYTES) is a separate, smaller budget.
    """
    try:
        if os.path.getsize(full) > _MAX_HASH_BYTES:
            return None
        h = hashlib.sha256()
        with open(full, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _ground_ref(path: str, want_hash: Optional[str] = None):
    """Pick the ONE checkout a ref may be read from. Returns (abs_path, label) or (None, "").

    The host carries many worktrees of the same repo, so "first root that has the file"
    would ground the skeptic on an arbitrary revision — a confident answer from the wrong
    source, worse than no grounding. This refuses rather than guesses:

      1. a candidate whose sha256 equals the finding's stamped ``hash`` wins outright — that
         is the exact revision the finding was recorded against;
      2. otherwise the primary repo root wins if it has the file (this is the checkout we
         are running in, and it preserves the previous in-repo behaviour unchanged);
      3. otherwise, only if EVERY candidate holds byte-identical content is it read — the
         revision question has a single answer, so there is nothing to guess;
      4. otherwise -> refused (None). Ambiguity is reported, never resolved by position.

    ``label`` names the checkout and the basis, so the fetched block can say which revision
    it is and whether the finding has drifted from it. Never raises.
    """
    try:
        roots = _search_roots()
        cands = []
        chosen = set()
        for root in roots:
            full = _safe_resolve(path, root)
            if full and full not in chosen and os.path.isfile(full):
                chosen.add(full)
                cands.append((root, full))
        if not cands:
            return None, ""
        digests = {}
        for root, full in cands:
            d = _file_sha256(full)
            if d is not None:
                digests.setdefault(d, (root, full))
        if not digests:
            return None, ""
        if want_hash and want_hash in digests:
            root, full = digests[want_hash]
            return full, f"{os.path.basename(root)}, exact revision"
        primary = _safe_resolve(path)
        if primary and any(full == primary for _r, full in cands):
            return primary, f"{os.path.basename(roots[0]) if roots else 'repo'}, this checkout"
        if len(digests) == 1:
            root, full = next(iter(digests.values()))
            note = f"{os.path.basename(root)}, identical in {len(cands)} checkouts"
            if want_hash:
                note += "; CHANGED since the finding was recorded"
            return full, note
        return None, ""
    except Exception:
        return None, ""


def _make_reader(hashes: Optional[dict] = None, provenance: Optional[dict] = None) -> ReaderFn:
    """Build the default reader, closing over a ``path -> stamped sha256`` map.

    The reader contract stays ``reader(path) -> text`` so injected test readers are
    unaffected; the hash a finding stamped on the ref rides along in the closure instead of
    the signature. ``provenance``, when given, collects ``path -> label`` for the chosen
    checkout so _fetch_code can name the revision in the block header.
    """
    h = hashes or {}

    def _read(path: str) -> str:
        try:
            full, label = _ground_ref(path, h.get(path))
            if not full:
                return ""
            # Read raw BYTES so the cap is byte-accurate (text-mode f.read(n) caps CHARACTERS
            # and can pull in more bytes for multibyte text). Read the cap, then decode.
            with open(full, "rb") as f:
                raw = f.read(_MAX_FILE_BYTES)
            if provenance is not None and label:
                provenance[path] = label
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return ""

    return _read


def _lazy_read_file(path: str) -> str:
    """Default reader with no stamped hashes: read a ref's source from the one checkout
    _ground_ref will commit to. Fail-open to "" on any rejection or error."""
    return _make_reader()(path)


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


def _fetch_code(refs, reader: ReaderFn, provenance: Optional[dict] = None) -> str:
    """Read the cited line ranges via `reader` and format them (line-numbered) for the prompt.

    Fail-open per ref: an unreadable/missing file or out-of-range span just contributes nothing.
    ``provenance`` (populated by the default reader) names the checkout each block came from,
    so a multi-repo ref never presents itself as if it were this repo's source.
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
        if n == 0:
            continue
        # Clamp the START into the file (a ref like file.py:0 becomes line 1).
        s = max(1, start)
        if s > n:
            continue
        # Clamp the END to the ACTUAL last displayable line: at least `s`, no past EOF,
        # and no more than _MAX_LINES_PER_REF lines. The header is then formatted from the
        # real s..e span so a 0/oversized/truncated ref never shows a misleading range or
        # an empty block.
        e = end if end >= start else start
        e = min(n, max(s, e))
        if e - s + 1 > _MAX_LINES_PER_REF:
            e = s + _MAX_LINES_PER_REF - 1
        numbered = "\n".join(f"{i}: {lines[i - 1]}" for i in range(s, e + 1))
        label = (provenance or {}).get(path)
        header = f"--- {path}:{s}" + (f"-{e}" if e != s else "")
        header += (f" [{label}]" if label else "") + " ---"
        blocks.append(header + "\n" + numbered)
    return "\n\n".join(blocks)


def _coerce_code_refs(code_refs) -> list:
    """Normalize the ``code_refs`` arg to a list of ref strings; ignore other types.

    Documented as a list, but a caller may pass a single ``file:line`` string; ``list(str)``
    would split it into characters. Accept a list/tuple (keeping only its string items) or a
    lone string; anything else yields []. Never raises.

    A stored finding's code_refs are ``{"path": .., "hash": ..}`` dicts with no line number,
    so a dict becomes a whole-file ref (``path:1-<_MAX_LINES_PER_REF>``); its hash is picked
    up separately by _coerce_code_ref_hashes.
    """
    if code_refs is None:
        return []
    if isinstance(code_refs, str):
        return [code_refs]
    if not isinstance(code_refs, (list, tuple)):
        return []
    out = []
    for x in code_refs:
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            p = x.get("path")
            if isinstance(p, str) and p.strip():
                p = p.strip()
                out.append(p if ":" in p else f"{p}:1-{_MAX_LINES_PER_REF}")
    return out


def _coerce_code_ref_hashes(code_refs) -> dict:
    """``path -> sha256`` for any ``{"path": .., "hash": ..}`` refs. {} otherwise.

    This is what makes multi-repo grounding safe: the stamped hash identifies WHICH checkout
    holds the revision the finding was recorded against, so _ground_ref can pick that one
    instead of guessing between worktrees. Never raises.
    """
    out: dict = {}
    if not isinstance(code_refs, (list, tuple)):
        return out
    for x in code_refs:
        if not isinstance(x, dict):
            continue
        p, h = x.get("path"), x.get("hash")
        if isinstance(p, str) and p.strip() and isinstance(h, str) and h.strip():
            out[p.strip().split(":")[0]] = h.strip()
    return out


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
    Callers may pass through non-string values (e.g. ``res.get('text')`` is None), so coerce
    both text fields to a stripped str here — the single normalization point — to keep the
    documented all-strings return shape.
    """
    ref = refutation if isinstance(refutation, str) else ("" if refutation is None else str(refutation))
    rea = reasoning if isinstance(reasoning, str) else ("" if reasoning is None else str(reasoning))
    return {"verdict": "uncertain", "refutation": ref.strip(), "reasoning": rea.strip(),
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
        code_refs: optional list of ``file:line`` / ``file:start-end`` strings, or stored
            ``{"path": .., "hash": ..}`` refs, whose source should be fetched into the
            prompt. ``file:line`` refs found in the claim/context are also picked up
            automatically. A stamped hash selects the exact checkout to read from.
            Fail-open: unreadable and AMBIGUOUS refs alike contribute nothing.
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
            provenance: dict = {}
            rd = reader
            if rd is None:
                # Default reader: carry the stamped per-path hashes so a ref that exists in
                # several checkouts is resolved to the exact revision, not the first hit.
                rd = _make_reader(_coerce_code_ref_hashes(code_refs), provenance)
            code_block = _fetch_code(refs, rd, provenance)
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
