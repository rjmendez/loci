#!/usr/bin/env python3
"""Offline replay of the D10 pair-MLP gate against the live cosine gate, and a blind
labelling sheet for the pairs on which they disagree (docs/d10_replay_plan.md).

The live shadow log (docs/d10_shadow_gate.md) fills slowly: one row set per
investigation_reason call. This replays historical investigations instead. For each
sampled investigation it asks the questions the manifest already states (title,
hypothesis, next_step; no language model writes them), embeds them and the
investigation's active findings with the embedder the live gate uses, and scores every
(question, finding) pair with both production gates:

* cosine: ``grounding_gate.cosine_gate``, the function investigation_reason calls;
* D10 MLP: ``d10_gate.load_gate`` (hash-pinned artifact) and ``d10_gate.mlp_decisions``,
  the functions the shadow logger calls, on the same features.

Then ``sample`` draws the labelling sample (only after the analysis plan is committed)
and writes a self-contained HTML sheet that shows a question and a finding and nothing
else: no score, no gate, no category. The key that maps sheet items back to gates is a
separate file the labeller never needs to open. scripts/d10_replay_analyze.py joins the
two.

Read-only towards the store. It opens <memory dir>/<investigation>/{manifest.json,
findings.jsonl, retractions.jsonl} for reading and never writes under the memory dir,
~/.loci or ~/.hermes; it refuses an output directory there. It never touches Qdrant or
Mnemosyne. Everything it writes goes to ``--out-dir``, which holds investigation text
and so must stay private: never commit it or upload it.

Polite by construction (the host runs bursty CronJobs every 5 minutes): one request at a
time, 16 texts per request, a pause between requests, no request started in the first
90 s after a clock minute divisible by 5 or while /proc/pressure io/cpu ``some avg60``
exceeds 15 % / 50 %. Every vector is appended to an on-disk cache keyed by the sha256 of
model and text as soon as it arrives, so a stopped run resumes where it stopped. After 30
minutes of continuous holding it exits 75 (try again later); after 5 consecutive failed
requests it exits 1. It only calls ``/v1/embeddings`` (and ``/api/tags`` once, to record
the model digest); it never calls a generation endpoint and never loads another model.

Usage:
  d10_replay.py run    --out-dir DIR [--memory-dir DIR] [--investigations 80] [--seed N]
  d10_replay.py sample --out-dir DIR --plan docs/d10_replay_plan.md [--seed N]
  d10_replay.py label  --out-dir DIR            # terminal fallback for the HTML sheet
Exit codes: 0 done, 1 failed, 2 usage, 75 gave up holding (rerun to resume).
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
MCP_DIR = REPO / "mcp"
if str(MCP_DIR) not in sys.path:
    sys.path.insert(0, str(MCP_DIR))

import grounding_gate  # noqa: E402
from inv_store import _fold_retracted_ids, _read_jsonl  # noqa: E402  (path-level, read-only)

SCHEMA = "loci-d10-replay/v1"
QUESTION_KINDS = ("title", "hypothesis", "next_step")
MIN_FINDINGS, MAX_FINDINGS = 8, 400
EXCLUDE_NAMES = {"flybrain-ops"}
EXCLUDE_PATTERN = re.compile(r"smoke|probe|test|recover-", re.IGNORECASE)
CATEGORIES = ("cos_only", "mlp_only", "both_keep", "both_drop")
LABELS = ("relevant", "not_relevant", "unsure")
MAX_TEXT_CHARS = 2000        # memcheck.llm.embed_texts truncates here; so does the cache key
SHEET_TRUNCATE = 700
EXIT_HOLD = 75

PAIRS_NAME, META_NAME, PROGRESS_NAME = "pairs.jsonl", "run_meta.json", "progress.jsonl"
CACHE_NAME = "vectors.jsonl"
KEY_NAME, ITEMS_NAME, SHEET_NAME, CLI_LABELS_NAME = "key.json", "sheet_items.jsonl", "label_sheet.html", "labels.jsonl"


class ReplayError(RuntimeError):
    """A condition the operator must fix; the message says what."""


class HoldTimeout(RuntimeError):
    """Held continuously for longer than the limit; progress is saved, rerun later."""


class EmbedFailed(RuntimeError):
    """Too many consecutive failed embedding requests."""


# --------------------------------------------------------------------------- paths


def default_memory_dir() -> Path:
    """The server's resolution (mcp/legacy_env.py), as in scripts/d10_shadow_report.py."""
    explicit = os.environ.get("LOCI_MEMORY_DIR") or os.environ.get("HERMES_MEMORY_DIR")
    if explicit:
        return Path(explicit).expanduser()
    new = Path.home() / ".loci" / "memory-sessions"
    legacy = Path.home() / ".hermes" / "memory-sessions"
    return new if new.is_dir() or not legacy.is_dir() else legacy


def check_out_dir(out_dir: Path, memory_dir: Path) -> Path:
    """Refuse an output directory inside the store, ~/.loci or ~/.hermes."""
    out = out_dir.expanduser().resolve()
    for root in (memory_dir, Path.home() / ".loci", Path.home() / ".hermes"):
        r = root.expanduser().resolve()
        if out == r or r in out.parents:
            raise ReplayError(f"refusing to write under {r}: the replay never writes into the store")
    return out


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# --------------------------------------------------------------------------- enumeration


def _manifest(inv_dir: Path) -> Optional[dict]:
    try:
        m = json.loads((inv_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return m if isinstance(m, dict) else None


def active_findings(inv_dir: Path) -> list[dict]:
    """investigation_reason's filter: dict rows, not retracted, non-blank text; file order."""
    raw = _read_jsonl(inv_dir / "findings.jsonl")
    retracted = _fold_retracted_ids(inv_dir / "retractions.jsonl")
    return [f for f in raw if isinstance(f, dict) and str(f.get("id", "")) not in retracted
            and str(f.get("text", "") or "").strip()]


def questions_for(manifest: dict) -> list[dict]:
    """One question per non-empty title / hypothesis / next_step; a repeated text is asked once."""
    out: list[dict] = []
    seen: set[str] = set()
    for kind in QUESTION_KINDS:
        v = manifest.get(kind)
        if not isinstance(v, str):
            continue
        text = " ".join(v.split())
        if not text or text.lower() in ("none", "null", "n/a") or text in seen:
            continue
        seen.add(text)
        out.append({"kind": kind, "text": text})
    return out


def exclusion_reason(name: str, manifest: Optional[dict], n_active: int, questions: list) -> Optional[str]:
    if name.startswith("_") or name.startswith("."):
        return "internal directory"
    if manifest is None:
        return "no readable manifest.json"
    if name in EXCLUDE_NAMES:
        return "synthetic / operations log"
    if EXCLUDE_PATTERN.search(name):
        return "name matches smoke|probe|test|recover-"
    if n_active < MIN_FINDINGS:
        return f"fewer than {MIN_FINDINGS} active findings"
    if n_active > MAX_FINDINGS:
        return f"more than {MAX_FINDINGS} active findings"
    if not questions:
        return "no title, hypothesis or next_step text"
    return None


def enumerate_investigations(memory_dir: Path) -> tuple[list[dict], list[dict]]:
    """(eligible, excluded) over the store's investigation directories. Reads only."""
    eligible, excluded = [], []
    if not memory_dir.is_dir():
        raise ReplayError(f"no memory dir at {memory_dir}")
    for d in sorted(memory_dir.iterdir()):
        if not d.is_dir():
            continue
        manifest = _manifest(d)
        n_active = len(active_findings(d)) if manifest is not None else 0
        qs = questions_for(manifest) if manifest is not None else []
        reason = exclusion_reason(d.name, manifest, n_active, qs)
        if reason:
            excluded.append({"investigation_id": d.name, "reason": reason, "n_findings": n_active})
            continue
        eligible.append({"investigation_id": d.name, "n_findings": n_active,
                         "created_at": str((manifest or {}).get("created_at") or ""),
                         "questions": qs})
    return eligible, excluded


def _tertile(ranked_ids: list[str]) -> dict[str, int]:
    n = len(ranked_ids)
    return {i: min(2, (r * 3) // max(n, 1)) for r, i in enumerate(ranked_ids)}


def stratified_sample(eligible: list[dict], n: int, seed: int) -> tuple[list[dict], dict]:
    """Up to n investigations, stratified 3 x 3 by size tertile and age tertile.

    Allocation is proportional to stratum size (largest remainder), at least one per
    non-empty stratum when n allows; within a stratum a seeded simple random sample.
    """
    by_id = {e["investigation_id"]: e for e in eligible}
    ids = sorted(by_id)
    size_t = _tertile(sorted(ids, key=lambda i: (by_id[i]["n_findings"], i)))
    age_t = _tertile(sorted(ids, key=lambda i: (by_id[i]["created_at"], i)))
    strata: dict[str, list[str]] = defaultdict(list)
    for i in ids:
        strata[f"size{size_t[i]}-age{age_t[i]}"].append(i)
    total = len(ids)
    n = min(n, total)
    keys = sorted(strata)
    if n >= total:
        alloc = {k: len(strata[k]) for k in keys}
    else:
        quota = {k: n * len(strata[k]) / total for k in keys}
        alloc = {k: int(quota[k]) for k in keys}
        if n >= len(keys):
            for k in keys:
                alloc[k] = max(alloc[k], 1)
        # Hand out (or take back) the remainder by the largest fractional quota.
        order = sorted(keys, key=lambda k: (-(quota[k] - int(quota[k])), k))
        j = 0
        while sum(alloc.values()) < n:
            k = order[j % len(order)]
            if alloc[k] < len(strata[k]):
                alloc[k] += 1
            j += 1
        for k in sorted(keys, key=lambda k: (quota[k] - alloc[k], k)):
            while sum(alloc.values()) > n and alloc[k] > 1:
                alloc[k] -= 1
    rng = random.Random(seed)
    chosen: list[dict] = []
    design = {}
    for k in keys:
        pick = rng.sample(strata[k], alloc[k])
        design[k] = {"population": len(strata[k]), "sampled": len(pick)}
        chosen.extend(dict(by_id[i], stratum=k) for i in pick)
    chosen.sort(key=lambda e: e["investigation_id"])
    return chosen, design


# --------------------------------------------------------------------------- polite embedding


def cache_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\n{text[:MAX_TEXT_CHARS]}".encode("utf-8")).hexdigest()


class VectorCache:
    """Append-only JSONL of {key, model, dim, vec}. A torn last line is ignored."""

    def __init__(self, path: Path):
        self.path = path
        self.vectors: dict[str, list[float]] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    row = json.loads(line)
                    self.vectors[row["key"]] = row["vec"]
                except (ValueError, KeyError, TypeError):
                    continue

    def __contains__(self, key: str) -> bool:
        return key in self.vectors

    def get(self, key: str) -> list[float]:
        return self.vectors[key]

    def put_many(self, model: str, keys: Sequence[str], vecs: Sequence[Sequence[float]]) -> None:
        rows = [{"key": k, "model": model, "dim": len(v), "vec": [float(x) for x in v]} for k, v in zip(keys, vecs)]
        _append_jsonl(self.path, rows)
        for r in rows:
            self.vectors[r["key"]] = r["vec"]


def read_pressure(path: Path) -> Optional[float]:
    """``some avg60`` from a /proc/pressure file, or None when unreadable."""
    try:
        for line in path.read_text().splitlines():
            if line.startswith("some "):
                for tok in line.split():
                    if tok.startswith("avg60="):
                        return float(tok.split("=", 1)[1])
    except (OSError, ValueError):
        return None
    return None


def in_burst_window(now: float, window_s: float = 90.0) -> bool:
    """True in the first ``window_s`` seconds after a local clock minute divisible by 5."""
    lt = time.localtime(now)
    return (lt.tm_min % 5) * 60 + lt.tm_sec < window_s


@dataclass
class Politeness:
    batch: int = 16
    pause_s: float = 2.0
    burst_window_s: float = 90.0
    io_limit: float = 15.0
    cpu_limit: float = 50.0
    max_hold_s: float = 1800.0
    poll_s: float = 5.0
    slow_s: float = 20.0
    backoff_base_s: float = 10.0
    backoff_max_s: float = 300.0
    max_failures: int = 5
    pressure_dir: Path = Path("/proc/pressure")


@dataclass
class EmbedStats:
    requests: int = 0
    texts: int = 0
    cached: int = 0
    failures: int = 0
    slow: int = 0
    hold_s: dict = field(default_factory=lambda: {"burst_window": 0.0, "io_pressure": 0.0, "cpu_pressure": 0.0})
    backoff_s: float = 0.0
    request_s: list = field(default_factory=list)


class PoliteEmbedder:
    """Embeds missing texts one request at a time, holding for cron bursts and pressure."""

    def __init__(self, embed_fn: Callable[[list[str]], list], model: str, cache: VectorCache, progress: Path,
                 pol: Politeness, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep):
        self.embed_fn, self.model, self.cache, self.progress = embed_fn, model, cache, progress
        self.pol, self.clock, self.sleep = pol, clock, sleep
        self.stats = EmbedStats()

    def hold_reason(self) -> Optional[str]:
        if in_burst_window(self.clock(), self.pol.burst_window_s):
            return "burst_window"
        io = read_pressure(self.pol.pressure_dir / "io")
        if io is not None and io > self.pol.io_limit:
            return "io_pressure"
        cpu = read_pressure(self.pol.pressure_dir / "cpu")
        if cpu is not None and cpu > self.pol.cpu_limit:
            return "cpu_pressure"
        return None

    def wait_turn(self) -> None:
        start = None
        while True:
            reason = self.hold_reason()
            if reason is None:
                return
            now = self.clock()
            start = now if start is None else start
            if now - start >= self.pol.max_hold_s:
                _append_jsonl(self.progress, [{"event": "gave_up_holding", "ts": now, "held_s": now - start,
                                               "reason": reason}])
                raise HoldTimeout(f"held {now - start:.0f}s continuously (last: {reason}); rerun to resume")
            self.sleep(self.pol.poll_s)
            self.stats.hold_s[reason] += self.clock() - now

    def _backoff(self, streak: int) -> None:
        s = min(self.pol.backoff_max_s, self.pol.backoff_base_s * (2 ** max(0, streak - 1)))
        self.stats.backoff_s += s
        self.sleep(s)

    def run(self, texts: Iterable[str]) -> None:
        todo, seen = [], set()
        for t in texts:
            k = cache_key(self.model, t)
            if k in seen:
                continue
            seen.add(k)
            if k in self.cache:
                self.stats.cached += 1
            else:
                todo.append((k, t[:MAX_TEXT_CHARS]))
        fails = slow_streak = 0
        i = 0
        while i < len(todo):
            chunk = todo[i:i + self.pol.batch]
            self.wait_turn()
            t0 = self.clock()
            try:
                vecs = self.embed_fn([t for _, t in chunk])
            except Exception:  # noqa: BLE001 - counted as a failed request
                vecs = []
            dt = self.clock() - t0
            self.stats.requests += 1
            self.stats.request_s.append(dt)
            ok = (isinstance(vecs, list) and len(vecs) == len(chunk)
                  and all(isinstance(v, (list, tuple)) and len(v) > 0 for v in vecs))
            if not ok:
                fails += 1
                self.stats.failures += 1
                _append_jsonl(self.progress, [{"event": "request_failed", "ts": self.clock(), "n": len(chunk),
                                               "consecutive": fails, "s": round(dt, 3)}])
                if fails >= self.pol.max_failures:
                    raise EmbedFailed(f"{fails} consecutive embedding requests failed; stopped")
                self._backoff(fails)
                continue
            fails = 0
            self.cache.put_many(self.model, [k for k, _ in chunk], vecs)
            self.stats.texts += len(chunk)
            _append_jsonl(self.progress, [{"event": "batch", "ts": self.clock(), "n": len(chunk),
                                           "s": round(dt, 3), "done": i + len(chunk), "todo": len(todo)}])
            i += len(chunk)
            if dt > self.pol.slow_s:
                slow_streak += 1
                self.stats.slow += 1
                self._backoff(slow_streak)
            else:
                slow_streak = 0
                if i < len(todo):
                    self.sleep(self.pol.pause_s)


# --------------------------------------------------------------------------- scoring


def categorise(cos_keep: bool, mlp_keep: bool) -> str:
    if cos_keep and mlp_keep:
        return "both_keep"
    if cos_keep:
        return "cos_only"
    if mlp_keep:
        return "mlp_only"
    return "both_drop"


def score_question(question: str, q_vec: Sequence[float], findings: list[dict], f_vecs: list, gate,
                   threshold: float = grounding_gate.GROUND_THRESHOLD) -> list[dict]:
    """Both gates on one question's pairs, with the production code paths."""
    import numpy as np

    import d10_gate
    from memcheck import llm as _llm

    n = len(findings)
    if n == 0:
        return []
    texts = [str(f["text"]) for f in findings]
    scored = [(_llm.cosine(q_vec, f_vecs[i]), i) for i in range(n)]
    in_ctx = set(grounding_gate.cosine_gate(scored, threshold))
    emb = np.asarray([list(q_vec)] + [list(v) for v in f_vecs], dtype=np.float64)
    scores = gate.score_pairs(np.repeat(emb[:1], n, axis=0), emb[1:], [question] * n, texts)
    mlp_keep, mlp_ctx = d10_gate.mlp_decisions([float(s) for s in scores], gate.tau)
    rows = []
    for i, f in enumerate(findings):
        c = scored[i][0]
        ck = grounding_gate.passes(c, threshold)
        rows.append({
            "finding_index": i,
            "finding_id": str(f.get("id") or ""),
            "cos": None if c is None else round(float(c), 6),
            "cos_keep": ck,
            "cos_in_context": i in in_ctx,
            "mlp_score": round(float(scores[i]), 6),
            "mlp_keep": bool(mlp_keep[i]),
            "mlp_in_context": bool(mlp_ctx[i]),
            "category": categorise(ck, bool(mlp_keep[i])),
            "category_in_context": categorise(i in in_ctx, bool(mlp_ctx[i])),
        })
    return rows


def ollama_model_digest(base: str, model: str) -> Optional[str]:
    """The served model's digest from /api/tags (metadata only; loads nothing)."""
    try:
        with urllib.request.urlopen(f"{base.rstrip('/')}/api/tags", timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
        for m in data.get("models", []):
            if m.get("name") in (model, f"{model}:latest") or m.get("model") in (model, f"{model}:latest"):
                return m.get("digest")
    except Exception:  # noqa: BLE001 - provenance only
        return None
    return None


def run_replay(memory_dir: Path, out_dir: Path, *, n_investigations: int, seed: int, pol: Politeness,
               embed_fn: Optional[Callable[[list[str]], list]] = None, model: Optional[str] = None,
               gate=None, clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
               digest: Optional[str] = None) -> dict:
    """Enumerate, sample, embed (politely, cached) and score. Returns the run metadata."""
    from memcheck import llm as _llm

    import d10_gate

    out_dir = check_out_dir(out_dir, memory_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wall0, cpu0 = time.monotonic(), time.process_time()
    model = model or _llm._embed_model()
    if embed_fn is None:
        def embed_fn(chunk: list[str]) -> list:
            return _llm.embed_texts(chunk, timeout=60.0, batch=len(chunk))
        digest = ollama_model_digest(_llm._ollama_base(), model)
    gate = gate or d10_gate.load_gate()

    eligible, excluded = enumerate_investigations(memory_dir)
    chosen, design = stratified_sample(eligible, n_investigations, seed)
    progress = out_dir / PROGRESS_NAME
    _append_jsonl(progress, [{"event": "start", "ts": clock(), "seed": seed, "investigations": len(chosen)}])

    loaded = {}
    texts: list[str] = []
    for inv in chosen:
        fs = active_findings(memory_dir / inv["investigation_id"])
        loaded[inv["investigation_id"]] = fs
        texts.extend(q["text"] for q in inv["questions"])
        texts.extend(str(f["text"]) for f in fs)

    cache = VectorCache(out_dir / CACHE_NAME)
    emb = PoliteEmbedder(embed_fn, model, cache, progress, pol, clock=clock, sleep=sleep)
    status = "done"
    try:
        emb.run(texts)
    except HoldTimeout as exc:
        status = f"held: {exc}"
    except EmbedFailed as exc:
        status = f"failed: {exc}"

    meta = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "seed": seed,
        "memory_dir_investigations": len(eligible) + len(excluded),
        "eligible": len(eligible),
        "excluded": excluded,
        "excluded_by_reason": dict(Counter(e["reason"] for e in excluded)),
        "investigation_design": design,
        "investigations": [{"investigation_id": c["investigation_id"], "n_findings": c["n_findings"],
                            "created_at": c["created_at"], "stratum": c["stratum"],
                            "question_kinds": [q["kind"] for q in c["questions"]]} for c in chosen],
        "embedder": {"model": model, "served_digest": digest, "max_chars": MAX_TEXT_CHARS,
                     "client": "memcheck.llm.embed_texts"},
        "cosine_gate": {"threshold": grounding_gate.GROUND_THRESHOLD, "context_cap": grounding_gate.CONTEXT_CAP,
                        "code": "grounding_gate.cosine_gate"},
        "mlp_gate": {"version": gate.version, "tau": gate.tau, "manifest_sha256": gate.manifest_sha256,
                     "code": "d10_gate.mlp_decisions"},
        "embedding": {"requests": emb.stats.requests, "texts_embedded": emb.stats.texts,
                      "texts_cached": emb.stats.cached, "failures": emb.stats.failures,
                      "slow_requests": emb.stats.slow, "hold_s": {k: round(v, 1) for k, v in emb.stats.hold_s.items()},
                      "backoff_s": round(emb.stats.backoff_s, 1),
                      "request_s_p50": _pct(emb.stats.request_s, 50), "request_s_max": _pct(emb.stats.request_s, 100)},
    }
    if status != "done":
        meta["wall_s"] = round(time.monotonic() - wall0, 1)
        meta["cpu_s"] = round(time.process_time() - cpu0, 1)
        _atomic_write(out_dir / META_NAME, json.dumps(meta, indent=2, sort_keys=True))
        return meta

    rows = []
    for inv in chosen:
        iid = inv["investigation_id"]
        fs = loaded[iid]
        f_vecs = [cache.get(cache_key(model, str(f["text"]))) for f in fs]
        for q in inv["questions"]:
            q_vec = cache.get(cache_key(model, q["text"]))
            for r in score_question(q["text"], q_vec, fs, f_vecs, gate):
                rows.append({"investigation_id": iid, "question_kind": q["kind"], **r})
    _atomic_write(out_dir / PAIRS_NAME, "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    meta["pairs"] = len(rows)
    meta["questions"] = sum(len(c["questions"]) for c in chosen)
    meta["questions_by_kind"] = dict(Counter(q["kind"] for c in chosen for q in c["questions"]))
    meta["categories"] = dict(Counter(r["category"] for r in rows))
    meta["categories_in_context"] = dict(Counter(r["category_in_context"] for r in rows))
    meta["categories_by_kind"] = {k: dict(Counter(r["category"] for r in rows if r["question_kind"] == k))
                                  for k in QUESTION_KINDS}
    meta["unanswerable_cosines"] = sum(1 for r in rows if r["cos"] is None)
    meta["wall_s"] = round(time.monotonic() - wall0, 1)
    meta["cpu_s"] = round(time.process_time() - cpu0, 1)
    _atomic_write(out_dir / META_NAME, json.dumps(meta, indent=2, sort_keys=True))
    _append_jsonl(progress, [{"event": "done", "ts": clock(), "pairs": len(rows)}])
    return meta


def _pct(xs: list, q: float) -> Optional[float]:
    """Nearest-rank percentile."""
    if not xs:
        return None
    s = sorted(xs)
    return round(s[max(0, min(len(s), math.ceil(q / 100 * len(s))) - 1)], 3)


# --------------------------------------------------------------------------- labelling sample


@dataclass
class SampleTargets:
    cos_only: int = 100
    mlp_only: int = 100
    both_keep: int = 30
    both_drop: int = 30
    duplicates: int = 12
    min_dup_gap: int = 30


def plan_provenance(plan: Path, allow_uncommitted: bool) -> dict:
    """sha256 of the plan, and the commit that last changed it. Refuses an uncommitted plan."""
    if not plan.is_file():
        raise ReplayError(f"no analysis plan at {plan}; commit docs/d10_replay_plan.md before sampling")
    sha = hashlib.sha256(plan.read_bytes()).hexdigest()
    commit, clean = None, False
    try:
        cwd = str(plan.resolve().parent)
        commit = subprocess.run(["git", "log", "-1", "--format=%H", "--", plan.name], cwd=cwd,
                                capture_output=True, text=True, timeout=30).stdout.strip() or None
        dirty = subprocess.run(["git", "status", "--porcelain", "--", plan.name], cwd=cwd,
                               capture_output=True, text=True, timeout=30).stdout.strip()
        clean = bool(commit) and not dirty
    except (OSError, subprocess.SubprocessError):
        pass
    if not clean and not allow_uncommitted:
        raise ReplayError(f"{plan} is not committed (or has local changes): pre-register the plan first")
    return {"path": str(plan), "sha256": sha, "commit": commit, "committed_clean": clean}


def draw_sample(rows: list[dict], targets: SampleTargets, seed: int) -> tuple[list[dict], dict]:
    """Seeded simple random sample per category, then duplicates, then a shuffle.

    Returns (items in sheet order, design). Each item carries its opaque id, its pair
    and, for a duplicate, ``duplicate_of``; the sheet shows none of that.
    """
    rng = random.Random(seed)
    by_cat: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for r in rows:
        by_cat[r["category"]].append(r)
    want = {"cos_only": targets.cos_only, "mlp_only": targets.mlp_only,
            "both_keep": targets.both_keep, "both_drop": targets.both_drop}
    design = {}
    picked: list[dict] = []
    for c in CATEGORIES:
        pool = sorted(by_cat[c], key=lambda r: (r["investigation_id"], r["question_kind"], r["finding_index"]))
        k = min(want[c], len(pool))
        take = rng.sample(pool, k)
        design[c] = {"population": len(pool), "sampled": k}
        picked.extend(take)
    rng.shuffle(picked)
    used: set[str] = set()

    def opaque() -> str:
        while True:
            h = "%016x" % rng.getrandbits(64)
            if h not in used:
                used.add(h)
                return h

    items = [dict(pair=r, id=opaque(), duplicate_of=None) for r in picked]
    n_dup = min(targets.duplicates, len(items))
    for src_pos in sorted(rng.sample(range(len(items)), n_dup), reverse=True):
        src = items[src_pos]
        lo = min(len(items), src_pos + targets.min_dup_gap)
        pos = rng.randint(lo, len(items))
        items.insert(pos, dict(pair=src["pair"], id=opaque(), duplicate_of=src["id"]))
    design["duplicates"] = n_dup
    return items, design


def population_by_investigation(rows: list[dict]) -> dict[str, dict[str, int]]:
    pop: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        pop[r["investigation_id"]][r["category"]] += 1
    return {k: dict(v) for k, v in sorted(pop.items())}


def sheet_id_for(sheet_items: list[dict]) -> str:
    return hashlib.sha256(json.dumps(sheet_items, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def build_sample(out_dir: Path, memory_dir: Path, plan: Path, *, seed: int, targets: SampleTargets,
                 allow_uncommitted_plan: bool = False, force: bool = False) -> dict:
    out_dir = check_out_dir(out_dir, memory_dir)
    if (out_dir / KEY_NAME).exists() and not force:
        raise ReplayError(f"{out_dir / KEY_NAME} exists: the sample is drawn once (use --force only before labelling)")
    prov = plan_provenance(plan, allow_uncommitted_plan)
    meta = json.loads((out_dir / META_NAME).read_text(encoding="utf-8"))
    if meta.get("status") != "done":
        raise ReplayError("the replay did not finish; rerun `run` first")
    rows = [json.loads(line) for line in (out_dir / PAIRS_NAME).read_text(encoding="utf-8").splitlines() if line]
    items, design = draw_sample(rows, targets, seed)

    # Text for the sheet, re-read (read-only) from the store.
    qtext: dict[tuple[str, str], str] = {}
    ftext: dict[tuple[str, int], str] = {}
    for iid in sorted({it["pair"]["investigation_id"] for it in items}):
        d = memory_dir / iid
        for q in questions_for(_manifest(d) or {}):
            qtext[(iid, q["kind"])] = q["text"]
        for i, f in enumerate(active_findings(d)):
            ftext[(iid, i)] = str(f["text"])
    sheet_items, key_items = [], {}
    for it in items:
        p = it["pair"]
        f = ftext.get((p["investigation_id"], p["finding_index"]))
        q = qtext.get((p["investigation_id"], p["question_kind"]))
        if f is None or q is None:
            raise ReplayError(f"store changed since the replay: {p['investigation_id']} no longer matches")
        sheet_items.append({"id": it["id"], "question": q, "finding": f})
        key_items[it["id"]] = {"investigation_id": p["investigation_id"], "question_kind": p["question_kind"],
                               "finding_index": p["finding_index"], "finding_id": p["finding_id"],
                               "category": p["category"], "category_in_context": p["category_in_context"],
                               "cos": p["cos"], "mlp_score": p["mlp_score"], "duplicate_of": it["duplicate_of"]}
    sid = sheet_id_for(sheet_items)
    key = {"schema": SCHEMA + "/key", "sheet_id": sid, "seed": seed, "plan": prov, "design": design,
           "targets": targets.__dict__, "population": population_by_investigation(rows),
           "run": {"created_at": meta["created_at"], "pairs": len(rows), "mlp_gate": meta["mlp_gate"],
                   "cosine_gate": meta["cosine_gate"]},
           "items": key_items}
    _atomic_write(out_dir / KEY_NAME, json.dumps(key, indent=1, sort_keys=True))
    _atomic_write(out_dir / ITEMS_NAME, "".join(json.dumps(s, sort_keys=True) + "\n" for s in sheet_items))
    _atomic_write(out_dir / SHEET_NAME, render_sheet(sheet_items, sid))
    return {"sheet_id": sid, "items": len(sheet_items), "design": design}


# --------------------------------------------------------------------------- the sheet

_SHEET = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'none'; connect-src 'none'; form-action 'none'">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>Relevance labelling</title>
<style>
:root { --bg:#fbfaf7; --fg:#1d1d1b; --muted:#6b6a64; --card:#fff; --line:#dedbd2; --accent:#2f5d8a; --warn:#8a4b2f; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --fg:#e9e7e1; --muted:#a19f97; --card:#20201e; --line:#3a3935; --accent:#8db4dc; --warn:#dca38d; } }
body { background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; margin:0; }
main { max-width: 860px; margin: 0 auto; padding: 16px; }
header { display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; justify-content:space-between; }
.meta { color:var(--muted); font-size:14px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:16px; margin:12px 0; }
.label { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin-bottom:4px; }
.q { font-size:18px; font-weight:600; white-space:pre-wrap; overflow-wrap:anywhere; }
.f { white-space:pre-wrap; overflow-wrap:anywhere; }
.choices { display:flex; flex-wrap:wrap; gap:8px; }
button { font:inherit; border:1px solid var(--line); background:var(--card); color:var(--fg); border-radius:6px; padding:8px 14px; cursor:pointer; }
button.sel { outline:3px solid var(--accent); }
button.link { border:none; background:none; color:var(--accent); padding:0; }
kbd { border:1px solid var(--line); border-radius:3px; padding:0 4px; font-size:13px; }
.bar { height:6px; background:var(--line); border-radius:3px; overflow:hidden; }
.bar > div { height:100%; background:var(--accent); width:0; }
.done { color:var(--warn); font-weight:600; }
details { color:var(--muted); font-size:14px; }
</style>
</head>
<body>
<main>
<header>
  <div><strong>Is this finding relevant to the question?</strong> <span class="meta" id="progress"></span></div>
  <div class="choices">
    <button id="export" type="button">Export labels</button>
    <label class="meta"><input id="import" type="file" accept="application/json,.json"> import</label>
  </div>
</header>
<div class="bar"><div id="bar"></div></div>
<details>
<summary>How to label</summary>
<p><b>Relevant</b>: a careful analyst answering this question would want this finding in front of them;
it is evidence for, against or about the question. <b>Not relevant</b>: it is about something else, or too
generic to bear on the question. <b>Unsure</b>: you cannot tell from the text shown. Judge only the text;
the order is random. Keys: <kbd>1</kbd> relevant, <kbd>2</kbd> not relevant, <kbd>3</kbd> unsure,
<kbd>e</kbd> expand, <kbd>&larr;</kbd>/<kbd>Backspace</kbd> back, <kbd>&rarr;</kbd> forward, <kbd>u</kbd> undo the last label.
Labels save in this browser as you go; export when done (or any time) and keep the file private.</p>
</details>
<div class="card"><div class="label">Question</div><div class="q" id="q"></div></div>
<div class="card"><div class="label">Finding <span id="pos"></span></div><div class="f" id="f"></div>
  <button class="link" id="expand" type="button">show all (e)</button></div>
<div class="choices">
  <button data-l="relevant" type="button"><kbd>1</kbd> Relevant</button>
  <button data-l="not_relevant" type="button"><kbd>2</kbd> Not relevant</button>
  <button data-l="unsure" type="button"><kbd>3</kbd> Unsure</button>
  <button id="back" type="button">&larr; Back</button>
  <button id="undo" type="button">Undo (u)</button>
</div>
<p class="meta" id="status"></p>
</main>
<script id="items" type="application/json">__ITEMS__</script>
<script>
(function () {
  "use strict";
  var SHEET = "__SHEET_ID__", TRUNC = __TRUNC__;
  var items = JSON.parse(document.getElementById("items").textContent);
  var KEY = "relevance-labels-" + SHEET;
  var labels = {}, history = [], idx = 0, expanded = false, shownAt = Date.now();
  function load() { try { var s = localStorage.getItem(KEY); if (s) { var o = JSON.parse(s); labels = o.labels || {}; history = o.history || []; } } catch (e) { setStatus("Browser storage unavailable: export often."); } }
  function save() { try { localStorage.setItem(KEY, JSON.stringify({labels: labels, history: history})); } catch (e) { setStatus("Could not save in this browser: export now."); } }
  function setStatus(t) { document.getElementById("status").textContent = t; }
  function nDone() { var n = 0; for (var i = 0; i < items.length; i++) if (labels[items[i].id]) n++; return n; }
  function firstOpen(from) { for (var k = 0; k < items.length; k++) { var j = (from + k) % items.length; if (!labels[items[j].id]) return j; } return -1; }
  function eta(left) {
    var ms = []; for (var id in labels) if (labels[id].ms > 0 && labels[id].ms < 300000) ms.push(labels[id].ms);
    ms.sort(function (a, b) { return a - b; });
    var per = ms.length ? ms[Math.floor(ms.length / 2)] : 15000;
    var min = Math.round(left * per / 60000);
    return min < 1 ? "under a minute left" : "about " + min + " min left";
  }
  function render() {
    var it = items[idx], done = nDone(), left = items.length - done;
    document.getElementById("q").textContent = it.question;
    var long = it.finding.length > TRUNC;
    document.getElementById("f").textContent = (long && !expanded) ? it.finding.slice(0, TRUNC) + " …" : it.finding;
    var ex = document.getElementById("expand"); ex.style.display = long ? "" : "none"; ex.textContent = expanded ? "show less (e)" : "show all (e)";
    document.getElementById("pos").textContent = "(item " + (idx + 1) + " of " + items.length + ")";
    document.getElementById("progress").textContent = done + " / " + items.length + " labelled, " + (left ? eta(left) : "done");
    document.getElementById("bar").style.width = (100 * done / items.length) + "%";
    var cur = labels[it.id] ? labels[it.id].label : null;
    var bs = document.querySelectorAll("button[data-l]"); for (var i = 0; i < bs.length; i++) bs[i].className = bs[i].getAttribute("data-l") === cur ? "sel" : "";
    if (!left) setStatus("All items labelled. Export the labels file now.");
    shownAt = Date.now();
  }
  function go(i) { idx = (i + items.length) % items.length; expanded = false; render(); }
  function choose(l) {
    var id = items[idx].id;
    labels[id] = {label: l, ts: new Date().toISOString(), ms: Date.now() - shownAt};
    history.push(id); save();
    var nxt = firstOpen(idx + 1); if (nxt < 0) { render(); } else { go(nxt); }
  }
  function undo() { var id = history.pop(); if (!id) return; delete labels[id]; save(); for (var i = 0; i < items.length; i++) if (items[i].id === id) { go(i); return; } }
  function exportLabels() {
    var out = {schema: "relevance-labels/v1", sheet_id: SHEET, exported_at: new Date().toISOString(), n_items: items.length, labels: labels};
    var blob = new Blob([JSON.stringify(out, null, 1)], {type: "application/json"});
    var a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "relevance_labels_" + SHEET + ".json";
    document.body.appendChild(a); a.click(); setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
    setStatus("Exported " + nDone() + " labels.");
  }
  function importLabels(file) {
    var r = new FileReader();
    r.onload = function () { try { var o = JSON.parse(r.result); if (o.sheet_id !== SHEET) { setStatus("That file belongs to another sheet."); return; }
      for (var id in o.labels) labels[id] = o.labels[id]; save(); go(Math.max(0, firstOpen(0))); setStatus("Imported."); } catch (e) { setStatus("Not a labels file."); } };
    r.readAsText(file);
  }
  document.addEventListener("keydown", function (e) {
    if (e.ctrlKey || e.metaKey || e.altKey || e.target.tagName === "INPUT") return;
    var k = e.key;
    if (k === "1") choose("relevant"); else if (k === "2") choose("not_relevant"); else if (k === "3") choose("unsure");
    else if (k === "ArrowLeft" || k === "Backspace" || k === "b") { e.preventDefault(); go(idx - 1); }
    else if (k === "ArrowRight") go(idx + 1);
    else if (k === "e") { expanded = !expanded; render(); }
    else if (k === "u") undo();
  });
  var bs = document.querySelectorAll("button[data-l]");
  for (var i = 0; i < bs.length; i++) bs[i].addEventListener("click", function () { choose(this.getAttribute("data-l")); });
  document.getElementById("back").addEventListener("click", function () { go(idx - 1); });
  document.getElementById("undo").addEventListener("click", undo);
  document.getElementById("expand").addEventListener("click", function () { expanded = !expanded; render(); });
  document.getElementById("export").addEventListener("click", exportLabels);
  document.getElementById("import").addEventListener("change", function () { if (this.files[0]) importLabels(this.files[0]); });
  load(); go(Math.max(0, firstOpen(0)));
})();
</script>
</body>
</html>
"""


def render_sheet(sheet_items: list[dict], sheet_id: str) -> str:
    """The self-contained sheet. Items carry only id, question and finding text."""
    for s in sheet_items:
        if set(s) != {"id", "question", "finding"}:
            raise ValueError("sheet items may carry only id, question and finding")
    # json.dumps then escape '<', '>' and '&' so no text can close the script element.
    payload = json.dumps(sheet_items, ensure_ascii=False)
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return (_SHEET.replace("__ITEMS__", payload)
            .replace("__SHEET_ID__", html.escape(sheet_id))
            .replace("__TRUNC__", str(SHEET_TRUNCATE)))


# --------------------------------------------------------------------------- terminal labelling


def read_cli_labels(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("label") in LABELS:
                out[str(row["id"])] = row["label"]
            elif row.get("label") is None and "id" in row:
                out.pop(str(row["id"]), None)
    return out


def label_cli(items_path: Path, labels_path: Path, *, inp: Callable[[str], str] = input,
              out: Callable[[str], None] = print) -> int:
    """1/2/3 label, e expand, b back, u undo, q quit. Appends to a JSONL; resumes."""
    items = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines() if line]
    done = read_cli_labels(labels_path)
    history: list[str] = []
    choice = {"1": "relevant", "2": "not_relevant", "3": "unsure"}
    idx = next((i for i, it in enumerate(items) if it["id"] not in done), len(items))
    while idx < len(items):
        it = items[idx]
        expanded = False
        while True:
            text = it["finding"] if expanded or len(it["finding"]) <= SHEET_TRUNCATE else it["finding"][:SHEET_TRUNCATE] + " ..."
            out(f"\n[{len(done)}/{len(items)} labelled] item {idx + 1}\nQUESTION: {it['question']}\nFINDING: {text}")
            a = inp("1 relevant, 2 not relevant, 3 unsure, e expand, b back, u undo, q quit > ").strip().lower()
            if a in choice:
                done[it["id"]] = choice[a]
                history.append(it["id"])
                _append_jsonl(labels_path, [{"id": it["id"], "label": choice[a], "ts": datetime.now(timezone.utc).isoformat()}])
                idx += 1
                break
            if a == "e":
                expanded = not expanded
            elif a == "b":
                idx = max(0, idx - 1)
                break
            elif a == "u" and history:
                last = history.pop()
                done.pop(last, None)
                _append_jsonl(labels_path, [{"id": last, "label": None, "ts": datetime.now(timezone.utc).isoformat()}])
                idx = next(i for i, x in enumerate(items) if x["id"] == last)
                break
            elif a == "q":
                return 0
        if idx >= len(items):
            idx = next((i for i, x in enumerate(items) if x["id"] not in done), len(items))
    out(f"All {len(items)} items labelled; labels are in {labels_path}.")
    return 0


# --------------------------------------------------------------------------- CLI


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="enumerate, embed politely (cached, resumable) and score both gates")
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--memory-dir", type=Path, default=None)
    r.add_argument("--investigations", type=int, default=80)
    r.add_argument("--seed", type=int, default=20260926)
    r.add_argument("--pause", type=float, default=2.0)
    r.add_argument("--max-hold", type=float, default=1800.0)
    s = sub.add_parser("sample", help="draw the labelling sample and write the sheet (after the plan is committed)")
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--memory-dir", type=Path, default=None)
    s.add_argument("--plan", type=Path, default=REPO / "docs" / "d10_replay_plan.md")
    s.add_argument("--seed", type=int, default=20260927)
    s.add_argument("--force", action="store_true")
    s.add_argument("--allow-uncommitted-plan", action="store_true", help="tests only")
    lab = sub.add_parser("label", help="terminal labelling (fallback for the HTML sheet)")
    lab.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "run":
            mem = args.memory_dir or default_memory_dir()
            meta = run_replay(mem, args.out_dir, n_investigations=args.investigations, seed=args.seed,
                              pol=Politeness(pause_s=args.pause, max_hold_s=args.max_hold))
            summary = {k: meta.get(k) for k in ("status", "eligible", "excluded_by_reason", "pairs", "questions",
                                                 "questions_by_kind", "categories", "categories_in_context",
                                                 "embedding", "wall_s", "cpu_s")}
            print(json.dumps(summary, indent=2, sort_keys=True))
            if meta["status"].startswith("held"):
                return EXIT_HOLD
            return 0 if meta["status"] == "done" else 1
        if args.cmd == "sample":
            mem = args.memory_dir or default_memory_dir()
            res = build_sample(args.out_dir, mem, args.plan, seed=args.seed, targets=SampleTargets(),
                               allow_uncommitted_plan=args.allow_uncommitted_plan, force=args.force)
            print(json.dumps(res, indent=2, sort_keys=True))
            return 0
        if args.cmd == "label":
            return label_cli(args.out_dir / ITEMS_NAME, args.out_dir / CLI_LABELS_NAME)
    except ReplayError as exc:
        print(f"d10_replay: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
