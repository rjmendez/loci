#!/usr/bin/env python3
"""Passive grooming tier for the Loci corpus — runs unattended, on cron.

Three rules hold for every pass:

  IDEMPOTENT   a second run over unchanged input proposes nothing new.
  FAIL-OPEN    a dead backend degrades the pass to a report; it never raises and
               never leaves the corpus half-written.
  SHADOW-FIRST model-derived output is written to _groom/proposals.jsonl with its
               provenance (pass, model, score) and is NEVER merged into
               findings.jsonl. A proposal is a claim awaiting adjudication, and it
               has to stay distinguishable from an author-written field.

``--apply`` promotes only for passes that declare ``applyable`` — today just
``index``, whose write is a re-upsert of the record already on disk, so it can
restore but cannot invent.

Usage:
    loci_groom.py index                    # report Qdrant/disk drift
    loci_groom.py index --apply            # re-embed and re-upsert what is missing
    loci_groom.py tags --limit 200         # propose canonical tags via the local model
    loci_groom.py all --json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

_REPO = Path(os.path.dirname(os.path.abspath(__file__))).parent
sys.path.insert(0, str(_REPO / "mcp"))


def load_env() -> dict:
    """Resolve backend config the way the server does, then from the config file.

    A cron job does not inherit the MCP launcher's environment, and the running
    server's QDRANT_URL currently lives only in its own process env — so without
    this a scheduled pass silently reports 'qdrant unreachable' forever.
    Precedence: existing env -> repo .env files -> ~/.loci/backends.toml.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(_REPO / ".env")
        load_dotenv(_REPO / "mcp" / ".env", override=True)
    except Exception:
        pass

    resolved = {}
    if not os.environ.get("QDRANT_URL"):
        try:
            import backends
            url, key = backends.qdrant()
            if url:
                os.environ["QDRANT_URL"] = url
                resolved["QDRANT_URL"] = url
                if key and not os.environ.get("QDRANT_API_KEY"):
                    os.environ["QDRANT_API_KEY"] = key
        except Exception:
            pass
    if not os.environ.get("OLLAMA_BASE_URL"):
        try:
            import backends
            url = backends.ollama_url()
            if url:
                os.environ["OLLAMA_BASE_URL"] = url
                resolved["OLLAMA_BASE_URL"] = url
        except Exception:
            pass
    return resolved


MEMORY_DIR = Path(os.environ.get(
    "HERMES_MEMORY_DIR",
    os.path.expanduser("~/.hermes/memory-sessions"),
))
GROOM_DIR = MEMORY_DIR / "_groom"

# Left unset on purpose: the batched (vLLM) and Ollama tiers identify the same
# model by different names — "Qwen/Qwen2.5-3B-Instruct" vs "qwen2.5:3b" — and a
# name the serving tier does not recognise is rejected outright. None lets each
# tier resolve its own. Set LOCI_GROOM_MODEL only to pin one deliberately.
GROOM_MODEL = os.environ.get("LOCI_GROOM_MODEL") or None
GROOM_BATCH = int(os.environ.get("LOCI_GROOM_BATCH", "16"))


# --- corpus access ----------------------------------------------------------

def iter_findings(memory_dir: Optional[Path] = None) -> Iterable[dict]:
    """Every finding on disk. The JSONL files are the system of record."""
    root = memory_dir or MEMORY_DIR
    for path in sorted(root.glob("*/findings.jsonl")):
        try:
            with open(path, errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue
        except OSError:
            continue


def _proposal(pass_name: str, subject_id: str, kind: str, value, **extra) -> dict:
    return {
        "id": f"{pass_name}:{kind}:{subject_id}",
        "pass": pass_name,
        "subject_id": subject_id,
        "kind": kind,
        "value": value,
        "proposed_by": f"loci_groom/{pass_name}",
        "model": extra.pop("model", None),
        "score": extra.pop("score", None),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        **extra,
    }


def write_proposals(rows: list[dict], groom_dir: Optional[Path] = None) -> int:
    """Append proposals, skipping ids already on file. Returns the number written."""
    if not rows:
        return 0
    root = groom_dir or GROOM_DIR
    root.mkdir(parents=True, exist_ok=True)
    path = root / "proposals.jsonl"

    seen = set()
    if path.exists():
        with open(path, errors="ignore") as fh:
            for line in fh:
                try:
                    seen.add(json.loads(line).get("id"))
                except ValueError:
                    continue

    fresh = [r for r in rows if r.get("id") not in seen]
    if not fresh:
        return 0
    with open(path, "a") as fh:
        for r in fresh:
            fh.write(json.dumps(r) + "\n")
    return len(fresh)


# --- pass: index ------------------------------------------------------------

def indexed_ids(client, col: str) -> set:
    """Every point id in the collection, paged."""
    out: set = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=col, limit=1024, offset=offset,
            with_payload=False, with_vectors=False,
        )
        out.update(str(p.id) for p in points)
        if offset is None:
            return out


def pass_index(apply: bool = False, limit: Optional[int] = None, **_) -> dict:
    """Reconcile the JSONL corpus against the Qdrant index.

    Findings reach Qdrant only on write (``_qdrant_upsert`` at store time) and the
    startup TTL purge deletes by ``created_at_ts``, so anything past the retention
    window leaves the index and nothing ever puts it back. Disk keeps it; search
    does not see it. This pass is the missing reconciliation.
    """
    on_disk = {}
    for f in iter_findings():
        fid = f.get("id")
        if fid:
            on_disk[str(fid)] = f

    report = {
        "pass": "index",
        "on_disk": len(on_disk),
        "indexed": 0,
        "missing": 0,
        "applied": 0,
        "status": "ok",
    }

    try:
        import qdrant_ops
        client, col = qdrant_ops._get_qdrant()
    except Exception as exc:
        report.update(status="degraded", detail=f"qdrant_ops unavailable: {exc!r}")
        return report
    if client is None:
        report.update(status="degraded", detail="qdrant unreachable")
        return report

    try:
        indexed = indexed_ids(client, col)
    except Exception as exc:
        report.update(status="degraded", detail=f"scroll failed: {exc!r}")
        return report

    # _qdrant_upsert stores the finding id verbatim as the point id.
    missing = [f for fid, f in on_disk.items() if fid not in indexed]

    report["indexed"] = len(indexed)
    report["missing"] = len(missing)
    report["coverage"] = round(1.0 - len(missing) / max(len(on_disk), 1), 4)

    if not apply:
        report["sample_missing"] = [f.get("id") for f in missing[:5]]
        return report

    for finding in missing[: (limit or len(missing))]:
        text = str(finding.get("text") or "")
        if not text:
            continue
        try:
            qdrant_ops._qdrant_upsert(str(finding["id"]), text, finding)
            report["applied"] += 1
        except Exception:
            continue
    return report


# --- pass: tags -------------------------------------------------------------

_TAG_PROMPT = (
    "You are labelling an engineering finding with tags from a fixed vocabulary.\n\n"
    "VOCABULARY (the only permitted values):\n{vocab}\n\n"
    "FINDING:\n{text}\n\n"
    "Pick the 1-4 vocabulary terms that genuinely describe this finding. If none fit, "
    "return an empty list. Do not invent terms. Do not explain.\n"
    'Output exactly one JSON object of the form {{"tags": ["term", "term"]}} and nothing else.\n'
    'Example output: {{"tags": ["mqtt", "wiring-gap"]}}'
)


def _parse_tags(raw: str) -> Optional[list]:
    """Pull a tag list out of a small model's JSON. None = unparseable.

    A 3B model asked for {"tags": [...]} will sometimes answer with a bare list or
    put the list under a key of its own choosing. Accepting those shapes is not the
    same as accepting invented content — every term still has to survive the
    vocabulary filter downstream.
    """
    try:
        obj = json.loads(raw or "")
    except ValueError:
        return None
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if isinstance(obj.get("tags"), list):
            return obj["tags"]
        lists = [v for v in obj.values() if isinstance(v, list)]
        if len(lists) == 1:
            return lists[0]
    return None


def build_vocabulary(findings: Iterable[dict], min_uses: int = 5, top_n: int = 60) -> list[str]:
    """The tags that carry signal — ones reused across the corpus.

    4,267 distinct tags exist on this corpus and 2,809 of them are used exactly
    once, so the raw tag set is closer to free text than to a vocabulary. Terms
    that recur are the ones a reader could actually filter on.
    """
    counts: collections.Counter = collections.Counter()
    for f in findings:
        for t in (f.get("tags") or []):
            t = str(t).strip().lower()
            # dt_* are per-run provenance stamps, not subject tags.
            if t and not t.startswith("dt_"):
                counts[t] += 1
    return [t for t, n in counts.most_common(top_n) if n >= min_uses]


def pass_tags(limit: Optional[int] = None, gen_fn: Optional[Callable] = None,
              memory_dir: Optional[Path] = None, groom_dir: Optional[Path] = None,
              **_) -> dict:
    """Propose vocabulary tags for findings that carry none.

    Proposals only — an author's tags are evidence about what they meant, and a 3B
    model's guess must not become indistinguishable from them.
    """
    findings = list(iter_findings(memory_dir))
    vocab = build_vocabulary(findings)
    report = {
        "pass": "tags", "corpus": len(findings), "vocabulary": len(vocab),
        "candidates": 0, "proposed": 0, "status": "ok",
    }
    if not vocab:
        report.update(status="degraded", detail="no vocabulary could be derived")
        return report

    candidates = [f for f in findings if not (f.get("tags") or []) and f.get("text")]
    report["candidates"] = len(candidates)
    if limit:
        candidates = candidates[:limit]
    if not candidates:
        return report

    if gen_fn is None:
        try:
            import batched_gen
            gen_fn = lambda prompts: batched_gen.generate_batch(  # noqa: E731
                prompts, model=GROOM_MODEL, max_tokens=96, fmt="json")
        except Exception as exc:
            report.update(status="degraded", detail=f"no generation tier: {exc!r}")
            return report

    vocab_block = ", ".join(vocab)
    vocab_set = set(vocab)
    proposals = []
    # A zero here has several causes and they are not interchangeable, so each is
    # counted rather than folded into one silent 0.
    rej = collections.Counter()

    for i in range(0, len(candidates), GROOM_BATCH):
        chunk = candidates[i:i + GROOM_BATCH]
        prompts = [
            _TAG_PROMPT.format(vocab=vocab_block, text=str(f.get("text"))[:1200])
            for f in chunk
        ]
        try:
            results = gen_fn(prompts)
        except Exception:
            rej["generation_error"] += len(chunk)
            continue
        for finding, res in zip(chunk, results or []):
            if not (res or {}).get("ok"):
                rej["generation_failed"] += 1
                continue
            tags = _parse_tags(res.get("text") or "")
            if tags is None:
                rej["unparseable"] += 1
                continue
            # A term outside the vocabulary is the model inventing one; drop it
            # rather than letting the pass widen the vocabulary it was given.
            if not tags:
                rej["declined"] += 1        # the model saw no fitting term — a real answer
                continue
            kept = [t for t in (str(x).strip().lower() for x in tags) if t in vocab_set][:4]
            if not kept:
                rej["out_of_vocabulary"] += 1
                continue
            proposals.append(_proposal(
                "tags", str(finding["id"]), "tags", kept,
                model=GROOM_MODEL,
                investigation_id=finding.get("investigation_id"),
            ))

    report["generated"] = len(proposals)
    report["proposed"] = write_proposals(proposals, groom_dir)
    report["rejected"] = dict(rej)
    return report


# --- pass: recall -----------------------------------------------------------

_QUESTION_PROMPT = (
    "Below is a finding from an engineering investigation. Write ONE short question "
    "that this finding answers — the question someone would type if they wanted this "
    "finding back. Use the finding's own vocabulary. No preamble, no quotes.\n\n"
    "FINDING:\n{text}\n\nQUESTION:"
)


def _rank_of(finding_id: str, results: list) -> Optional[int]:
    for i, row in enumerate(results, start=1):
        if str(row.get("id") or "") == finding_id:
            return i
    return None


def _score(ranks: list, k: int, attempted: int) -> dict:
    hits1 = sum(1 for r in ranks if r == 1)
    hitsk = sum(1 for r in ranks if r is not None and r <= k)
    mrr = sum(1.0 / r for r in ranks if r) / max(attempted, 1)
    return {
        "attempted": attempted,
        "recall_at_1": round(hits1 / max(attempted, 1), 4),
        f"recall_at_{k}": round(hitsk / max(attempted, 1), 4),
        "mrr": round(mrr, 4),
    }


def pass_recall(sample: int = 40, k: int = 5, paraphrase: bool = True,
                gen_fn: Optional[Callable] = None, memory_dir: Optional[Path] = None,
                groom_dir: Optional[Path] = None, search_fn: Optional[Callable] = None,
                seed: int = 0, limit: Optional[int] = None, rerank: bool = True,
                **_) -> dict:
    """Ask the retriever for findings it already holds, and see if it returns them.

    Two probes, kept apart on purpose, because today they fail identically:

      identity   query = the finding's own text. A miss here is WIRING — wrong
                 embedder, wrong width, a collection that cannot be queried at all.
                 No model is involved, so a regression is unambiguous.
      paraphrase query = a question the local model writes from the finding. A miss
                 here with identity intact is SEMANTIC reach, not breakage.

    Only findings that are actually indexed are sampled — otherwise this measures
    the index gap that ``index`` already reports, and reads as a retrieval failure.
    """
    sample = limit or sample          # --limit is the CLI's name for the sample size
    report = {"pass": "recall", "k": k, "sample": sample, "rerank": rerank, "status": "ok"}

    try:
        import qdrant_ops
        client, col = qdrant_ops._get_qdrant()
    except Exception as exc:
        report.update(status="degraded", detail=f"qdrant_ops unavailable: {exc!r}")
        return report
    if client is None:
        report.update(status="degraded", detail="qdrant unreachable")
        return report

    by_id = {}
    for f in iter_findings(memory_dir):
        fid = f.get("id")
        if fid and (f.get("text") or "").strip():
            by_id[str(fid)] = f
    try:
        live = sorted(indexed_ids(client, col) & set(by_id))
    except Exception as exc:
        report.update(status="degraded", detail=f"scroll failed: {exc!r}")
        return report

    report["indexed_with_text"] = len(live)
    if not live:
        report.update(status="degraded", detail="nothing indexed to probe")
        return report

    rnd = random.Random(seed)
    chosen = rnd.sample(live, min(sample, len(live)))
    search = search_fn or (
        lambda q: qdrant_ops._qdrant_similarity_search(q, limit=k, rerank=rerank))

    ident_ranks, errors = [], 0
    for fid in chosen:
        try:
            res = search(str(by_id[fid]["text"])[:1000])
        except Exception:
            errors += 1
            continue
        if not (res or {}).get("ok"):
            errors += 1
            continue
        ident_ranks.append(_rank_of(fid, res.get("results") or []))
    report["identity"] = _score(ident_ranks, k, len(ident_ranks))
    report["search_errors"] = errors

    if not paraphrase:
        _append_recall(report, groom_dir)
        return report

    if gen_fn is None:
        try:
            import batched_gen
            gen_fn = lambda prompts: batched_gen.generate_batch(  # noqa: E731
                prompts, model=GROOM_MODEL, max_tokens=64)
        except Exception as exc:
            report["paraphrase"] = {"status": "degraded", "detail": f"no generation tier: {exc!r}"}
            _append_recall(report, groom_dir)
            return report

    para_ranks, unusable = [], 0
    for i in range(0, len(chosen), GROOM_BATCH):
        chunk = chosen[i:i + GROOM_BATCH]
        prompts = [_QUESTION_PROMPT.format(text=str(by_id[f]["text"])[:1000]) for f in chunk]
        try:
            gen = gen_fn(prompts)
        except Exception:
            unusable += len(chunk)
            continue
        for fid, g in zip(chunk, gen or []):
            question = (g or {}).get("text", "").strip()
            if not (g or {}).get("ok") or len(question) < 8:
                unusable += 1
                continue
            try:
                res = search(question)
            except Exception:
                unusable += 1
                continue
            if not (res or {}).get("ok"):
                unusable += 1
                continue
            para_ranks.append(_rank_of(fid, res.get("results") or []))
    report["paraphrase"] = {**_score(para_ranks, k, len(para_ranks)), "unusable": unusable}

    _append_recall(report, groom_dir)
    return report


def _append_recall(report: dict, groom_dir: Optional[Path] = None) -> None:
    """One row per run. The number matters far less than its trend."""
    root = groom_dir or GROOM_DIR
    try:
        root.mkdir(parents=True, exist_ok=True)
        with open(root / "recall.jsonl", "a") as fh:
            fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
                                 **report}) + "\n")
    except OSError:
        pass


# --- pass: knn_tags ---------------------------------------------------------

def _tags_of(row: dict) -> list:
    """Payload tags, normalised. Some write paths store a comma-joined string."""
    raw = row.get("tags")
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, list):
        return []
    return [t for t in (str(x).strip().lower() for x in raw) if t and not t.startswith("dt_")]


def _knn_vote(neighbours: list, self_id: str, vocab: set, min_weight: float) -> list:
    """Similarity-weighted tag vote over retrieved neighbours.

    Weighting by score rather than counting means one very close neighbour can
    carry a tag that five loose ones cannot — which is the whole reason to use
    the embedding rather than a bag of words.
    """
    weights: dict = {}
    for row in neighbours:
        if str(row.get("id") or "") == self_id:
            continue
        score = float(row.get("score") or 0.0)
        if score <= 0:
            continue
        for tag in _tags_of(row):
            if vocab and tag not in vocab:
                continue
            weights[tag] = weights.get(tag, 0.0) + score
    ranked = sorted(weights.items(), key=lambda kv: -kv[1])
    return [(t, round(w, 4)) for t, w in ranked if w >= min_weight]


def pass_knn_tags(limit: Optional[int] = None, k: int = 8, min_weight: float = 1.0,
                  max_tags: int = 3, calibrate: bool = False, seed: int = 0,
                  memory_dir: Optional[Path] = None, groom_dir: Optional[Path] = None,
                  search_fn: Optional[Callable] = None, **_) -> dict:
    """Transfer tags from a finding's nearest already-tagged neighbours.

    No generation involved. The embeddings are already in the index, the author's
    own tags are the labels, and the whole thing is deterministic — so unlike a
    model's guess it can be scored against held-out truth before anyone promotes
    it. ``calibrate=True`` does exactly that: it hides the tags of findings that
    have them, re-derives them from neighbours, and reports how often it agrees.
    """
    report = {"pass": "knn_tags", "k": k, "min_weight": min_weight,
              "calibrate": calibrate, "status": "ok"}

    findings = list(iter_findings(memory_dir))
    vocab = set(build_vocabulary(findings))
    report["vocabulary"] = len(vocab)
    if not vocab:
        report.update(status="degraded", detail="no vocabulary could be derived")
        return report

    if search_fn is None:
        try:
            import qdrant_ops
            if qdrant_ops._get_qdrant()[0] is None:
                report.update(status="degraded", detail="qdrant unreachable")
                return report
            search_fn = lambda q: qdrant_ops._qdrant_similarity_search(  # noqa: E731
                q, limit=k + 1, rerank=False)
        except Exception as exc:
            report.update(status="degraded", detail=f"qdrant_ops unavailable: {exc!r}")
            return report

    if calibrate:
        subjects = [f for f in findings
                    if f.get("text") and set(_tags_of(f)) & vocab]
        rnd = random.Random(seed)
        rnd.shuffle(subjects)
    else:
        subjects = [f for f in findings if f.get("text") and not _tags_of(f)]
    report["candidates"] = len(subjects)
    if limit:
        subjects = subjects[:limit]

    proposals, agree, checked = [], [], 0
    for f in subjects:
        try:
            res = search_fn(str(f["text"])[:1000])
        except Exception:
            continue
        if not (res or {}).get("ok"):
            continue
        voted = _knn_vote(res.get("results") or [], str(f.get("id") or ""), vocab, min_weight)
        picked = [t for t, _w in voted[:max_tags]]

        if calibrate:
            truth = set(_tags_of(f)) & vocab
            if not truth:
                continue
            checked += 1
            agree.append(len(set(picked) & truth) / max(len(picked), 1) if picked else 0.0)
            continue

        if picked:
            proposals.append(_proposal(
                "knn_tags", str(f["id"]), "tags", picked,
                model=None, score=voted[0][1],
                method=f"knn/{k}", investigation_id=f.get("investigation_id"),
            ))

    if calibrate:
        report["checked"] = checked
        report["mean_precision"] = round(sum(agree) / max(checked, 1), 4)
        report["exact_or_partial_hit_rate"] = round(
            sum(1 for a in agree if a > 0) / max(checked, 1), 4)
        return report

    report["generated"] = len(proposals)
    report["proposed"] = write_proposals(proposals, groom_dir)
    return report


# --- pass: codelink ---------------------------------------------------------

# Identifier-shaped tokens: snake_case, CamelCase, dotted paths, File.ext.
_IDENT_RE = re.compile(r"\b(?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]{3,}\b")

# Words that look like identifiers to a regex and like noise to a reader.
_IDENT_STOP = frozenset({
    "which", "there", "these", "those", "where", "while", "would", "could",
    "should", "because", "before", "after", "every", "never", "always", "still",
    "value", "values", "return", "returns", "error", "errors", "false", "true",
    "none", "null", "class", "function", "method", "module", "import", "input",
    "output", "result", "results", "state", "check", "checks", "count", "counts",
    "under", "above", "below", "about", "against", "without", "within", "into",
    "test", "tests", "code", "path", "paths", "file", "files", "line", "lines",
    "data", "type", "types", "name", "names", "call", "calls", "read", "write",
})


def _is_distinctive(tok: str) -> bool:
    """Does this token name a symbol, or is it just a word that happens to be one?

    Measured, not guessed: the first live calibration linked `device`, `roll`,
    `train`, `report`, `baseline` and `confirmed` to real symbols, because each is
    genuinely a symbol name *somewhere* in an 11k-symbol graph across three
    languages. A bare lowercase word carries no evidence that the author meant the
    code. Structure does — an underscore, an internal case change, or real length.
    This is the same judgement `code_memory_relink` encodes as "distinctive", and
    it is why its coverage is conservative rather than broken.
    """
    if "_" in tok:
        return True
    if any(c.isupper() for c in tok[1:]) and any(c.islower() for c in tok):
        return True
    return len(tok) >= 12


def _candidate_symbols(text: str, index: dict) -> tuple:
    """(unique_hits, ambiguous_hits) for the identifier-shaped tokens in `text`.

    A token that names exactly one symbol in the graph is evidence. A token that
    names five is a question, and questions go to the model rather than being
    guessed at — that asymmetry is the whole design.
    """
    unique, ambiguous = {}, {}
    for raw in set(_IDENT_RE.findall(text or "")):
        tok = raw.split(".")[-1]
        if len(tok) < 4 or tok.lower() in _IDENT_STOP or not _is_distinctive(tok):
            continue
        hits = index.get(tok) or []
        if len(hits) == 1:
            unique[tok] = hits[0]
        elif 1 < len(hits) <= 8:
            ambiguous[tok] = hits
    return unique, ambiguous


_DISAMBIGUATE_PROMPT = (
    "A finding mentions the symbol `{token}`. Several symbols share that name.\n\n"
    "FINDING:\n{text}\n\nCANDIDATES:\n{options}\n\n"
    "Which candidate does the finding mean? Answer with the number alone, or 0 if "
    "the finding is not about any of them."
)


def pass_codelink(limit: Optional[int] = None, calibrate: bool = False,
                  gen_fn: Optional[Callable] = None, symbols_fn: Optional[Callable] = None,
                  linked_fn: Optional[Callable] = None, memory_dir: Optional[Path] = None,
                  groom_dir: Optional[Path] = None, **_) -> dict:
    """Propose Finding -> CodeSymbol links the strict lexical linker will not make.

    ``code_memory_relink`` is precision-first by design and reaches 180 of 2,634
    findings. This adds the two things it refuses to do: it treats a token that
    resolves to exactly one symbol as evidence even when that token is not
    'distinctive' by its rules, and it hands genuinely ambiguous tokens to the
    local model with the candidate file paths attached. Proposals only — a wrong
    code link is worse than no code link, because it survives as provenance.
    """
    report = {"pass": "codelink", "calibrate": calibrate, "status": "ok"}

    if symbols_fn is None or linked_fn is None:
        try:
            from graph.ladybug_store import LadybugStore
            ks = LadybugStore(str((memory_dir or MEMORY_DIR) / "graph.ladybug"))
            if not ks.readable_probe():
                report.update(status="degraded", detail="graph store unreadable")
                return report
            symbols_fn = symbols_fn or (lambda: ks._rows(
                "MATCH (s:CodeSymbol) RETURN s.name, s.id, s.file"))
            linked_fn = linked_fn or (lambda: ks._rows(
                "MATCH (f:Finding)-[:REFERENCES]->(s:CodeSymbol) RETURN f.id, s.id"))
        except Exception as exc:
            report.update(status="degraded", detail=f"graph unavailable: {exc!r}")
            return report

    index: dict = {}
    where: dict = {}
    try:
        for name, sid, sfile in symbols_fn():
            index.setdefault(str(name), []).append(str(sid))
            where[str(sid)] = str(sfile or "")
    except Exception as exc:
        report.update(status="degraded", detail=f"symbol read failed: {exc!r}")
        return report
    report["symbols"] = len(where)

    already: dict = {}
    try:
        for fid, sid in linked_fn():
            already.setdefault(str(fid), set()).add(str(sid))
    except Exception as exc:
        report.update(status="degraded", detail=f"link read failed: {exc!r}")
        return report
    report["already_linked"] = len(already)

    findings = [f for f in iter_findings(memory_dir) if f.get("text") and f.get("id")]
    if calibrate:
        subjects = [f for f in findings if str(f["id"]) in already]
    else:
        subjects = [f for f in findings if str(f["id"]) not in already]
    report["candidates"] = len(subjects)
    if limit:
        subjects = subjects[:limit]

    proposals, hits, checked, ambiguous_total = [], [], 0, 0
    for f in subjects:
        unique, ambiguous = _candidate_symbols(str(f["text"]), index)
        ambiguous_total += len(ambiguous)
        picked = dict(unique)

        if ambiguous and gen_fn is not None:
            toks = list(ambiguous)[:4]
            prompts = []
            for tok in toks:
                opts = "\n".join(
                    f"{i}. {where.get(sid, '?')}" for i, sid in enumerate(ambiguous[tok], 1))
                prompts.append(_DISAMBIGUATE_PROMPT.format(
                    token=tok, text=str(f["text"])[:600], options=opts))
            try:
                answers = gen_fn(prompts)
            except Exception:
                answers = []
            for tok, ans in zip(toks, answers or []):
                if not (ans or {}).get("ok"):
                    continue
                digits = "".join(ch for ch in (ans.get("text") or "") if ch.isdigit())
                if not digits:
                    continue
                choice = int(digits[:2])
                if 1 <= choice <= len(ambiguous[tok]):
                    picked[tok] = ambiguous[tok][choice - 1]

        if calibrate:
            truth = already.get(str(f["id"]), set())
            if not truth:
                continue
            checked += 1
            got = set(picked.values())
            hits.append(len(got & truth) / max(len(got), 1) if got else 0.0)
            continue

        if picked:
            proposals.append(_proposal(
                "codelink", str(f["id"]), "code_refs",
                [{"token": t, "symbol_id": sid, "file": where.get(sid, "")}
                 for t, sid in picked.items()],
                model=GROOM_MODEL if ambiguous and gen_fn else None,
                investigation_id=f.get("investigation_id"),
            ))

    report["ambiguous_tokens"] = ambiguous_total
    if calibrate:
        report["checked"] = checked
        report["mean_precision"] = round(sum(hits) / max(checked, 1), 4)
        report["any_correct_rate"] = round(
            sum(1 for h in hits if h > 0) / max(checked, 1), 4)
        return report

    report["generated"] = len(proposals)
    report["proposed"] = write_proposals(proposals, groom_dir)
    return report


PASSES = {
    "index": {"fn": pass_index, "applyable": True},
    "tags": {"fn": pass_tags, "applyable": False},
    "recall": {"fn": pass_recall, "applyable": False},
    "knn_tags": {"fn": pass_knn_tags, "applyable": False},
    "codelink": {"fn": pass_codelink, "applyable": False},
}


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("passes", nargs="*", default=["all"],
                    help=f"one or more of: {', '.join(PASSES)}, all")
    ap.add_argument("--apply", action="store_true",
                    help="promote results for passes that allow it (index only)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-rerank", action="store_true",
                    help="recall: probe the bi-encoder stage alone, without the cross-encoder")
    args = ap.parse_args(argv)
    load_env()

    names = list(PASSES) if (not args.passes or "all" in args.passes) else args.passes
    unknown = [n for n in names if n not in PASSES]
    if unknown:
        print(f"unknown pass(es): {', '.join(unknown)}", file=sys.stderr)
        return 2

    reports = []
    for name in names:
        spec = PASSES[name]
        apply = args.apply and spec["applyable"]
        if args.apply and not spec["applyable"]:
            print(f"[groom] {name}: --apply ignored (proposals only)", file=sys.stderr)
        try:
            reports.append(spec["fn"](apply=apply, limit=args.limit,
                                      rerank=not args.no_rerank))
        except Exception as exc:      # a pass must never take the cron job down
            reports.append({"pass": name, "status": "error", "detail": repr(exc)})

    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        for r in reports:
            head = f"[groom] {r['pass']}: {r.get('status')}"
            rest = " ".join(f"{k}={v}" for k, v in r.items()
                            if k not in ("pass", "status") and not isinstance(v, (list, dict)))
            print(f"{head} {rest}".rstrip())
    return 0 if all(r.get("status") != "error" for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
