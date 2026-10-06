"""Hillclimb: improve a graded Loci surface one patch at a time, with an overfitting guard.

A *suite* is a set of graded cases plus an allow-list of *surfaces* (overlay keys the climber
may change: a prompt guidance block, a numeric knob, a choice). Each round the proposer sees
the failing TRAIN cases and proposes exactly one change to one surface. The change is scored
on train and on a held-out TEST split:

* accepted only if train improves by ``margin`` AND test improves by ``test_gain``;
* otherwise reverted. Train up with test flat is memorising the eval, so it goes back.

Nothing is applied to production by a run. A run writes ``candidate_overlay.json`` and a
ledger; ``promote`` copies the candidate to ``overlay.json``, which consumers read through
``overlay_get``. With no overlay file every consumer keeps its built-in default, so shipping
this module changes no behaviour.

Before climbing it checks the grader itself: re-scoring the same cases must agree
(consistency), the case set must be big enough to mean something, and a baseline at or above
``saturated`` has no headroom. Cases that fail for infrastructure reasons (the model was down,
not wrong) are excluded from the mean and counted; too many aborts the run.

Privacy: the ledger holds surface names, patch values, rationales, scores and case ids. Case
text and model traces stay in memory and go only to the local proposer model.

Pure and injectable: suites and proposers are plain objects, so tests run offline.

    python mcp/hillclimb.py run --suite triage --rounds 6
    python mcp/hillclimb.py status --suite triage
    python mcp/hillclimb.py promote --suite triage
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

_MAX_TEXT = 1200
_TRACE_HEAD = 500
_MIN_CASES_PER_SPLIT = 5
_MAX_INFRA_FRACTION = 0.2


# ------------------------------------------------------------------------------ data types

@dataclass
class Case:
    id: str
    data: dict = field(default_factory=dict)


@dataclass
class CaseResult:
    id: str
    score: float = 0.0
    trace: str = ""
    infra_error: bool = False  # the run failed for infrastructure reasons: not the overlay's fault


@dataclass
class Surface:
    """One overlay key the climber may change."""
    key: str
    kind: str                     # "text" | "number" | "choice"
    desc: str = ""
    default: Any = None
    lo: float = 0.0
    hi: float = 0.0
    choices: tuple = ()
    max_len: int = _MAX_TEXT
    must_keep: tuple = ()         # substrings a text value must still contain

    def validate(self, value: Any) -> tuple[bool, Any, str]:
        """(ok, cleaned value, reason)."""
        if self.kind == "text":
            if not isinstance(value, str):
                return False, None, "text surface needs a string"
            v = value.strip()
            if len(v) > self.max_len:
                return False, None, f"longer than {self.max_len} chars"
            for need in self.must_keep:
                if need not in v:
                    return False, None, f"must keep {need!r}"
            return True, v, ""
        if self.kind == "number":
            try:
                v = float(value)
            except (TypeError, ValueError):
                return False, None, "not a number"
            if math.isnan(v) or not (self.lo <= v <= self.hi):
                return False, None, f"outside [{self.lo}, {self.hi}]"
            return True, v, ""
        if self.kind == "choice":
            if value not in self.choices:
                return False, None, f"not one of {list(self.choices)}"
            return True, value, ""
        return False, None, f"unknown surface kind {self.kind!r}"


# ------------------------------------------------------------------------------------ paths

def _root() -> Path:
    import legacy_env
    mem = str(legacy_env.memory_dir())
    return Path(mem).parent / "hillclimb"


def suite_dir(suite: str) -> Path:
    safe = "".join(c for c in suite if c.isalnum() or c in "-_") or "suite"
    return _root() / safe


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


# ----------------------------------------------------------------------- overlay (consumers)

_overlay_cache: dict[str, tuple[float, dict]] = {}


def overlay_get(suite: str, key: str, default: Any = None) -> Any:
    """Promoted overlay value for ``key``, else ``default``. Never raises; cached by mtime."""
    try:
        path = suite_dir(suite) / "overlay.json"
        mtime = path.stat().st_mtime
        hit = _overlay_cache.get(str(path))
        if hit is None or hit[0] != mtime:
            hit = (mtime, _read_json(path))
            _overlay_cache[str(path)] = hit
        return hit[1].get(key, default)
    except Exception:
        return default


# ----------------------------------------------------------------------------------- split

def split(cases: list[Case], test_frac: float = 0.3, seed: str = "") -> tuple[list[Case], list[Case]]:
    """Stable train/test split: each case lands on a side by its own id hash, so adding cases
    never moves an old one (barring the tiny-set fallback below)."""
    def bucket(c: Case) -> float:
        h = hashlib.sha256(f"{seed}:{c.id}".encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64

    test = [c for c in cases if bucket(c) < test_frac]
    if not test and len(cases) >= 2:  # tiny set: still hold one case out
        test = [min(cases, key=bucket)]
    ids = {c.id for c in test}
    return [c for c in cases if c.id not in ids], test


# --------------------------------------------------------------------------------- scoring

@dataclass
class Score:
    mean: float = 0.0
    n: int = 0
    infra_errors: int = 0
    per_case: dict = field(default_factory=dict)   # id -> mean score over repeats
    failures: list = field(default_factory=list)   # [{id, score, trace}] below 1.0, worst first

    def summary(self) -> dict:
        return {"mean": round(self.mean, 4), "n": self.n, "infra_errors": self.infra_errors}


def score_set(suite, cases: list[Case], overlay: dict, repeats: int = 1) -> Score:
    """Score ``overlay`` over ``cases``. An exception from the suite counts as infra, not a fail."""
    per: dict[str, list[float]] = {}
    traces: dict[str, str] = {}
    infra = 0
    for case in cases:
        for _ in range(max(1, repeats)):
            try:
                r = suite.run(case, overlay)
            except Exception as exc:
                r = CaseResult(case.id, 0.0, f"suite raised: {exc}", infra_error=True)
            if r.infra_error:
                infra += 1
                continue
            per.setdefault(case.id, []).append(max(0.0, min(1.0, float(r.score))))
            if r.score < 1.0:
                traces[case.id] = r.trace
    means = {cid: sum(v) / len(v) for cid, v in per.items()}
    mean = sum(means.values()) / len(means) if means else 0.0
    failures = sorted(({"id": cid, "score": round(s, 3), "trace": traces.get(cid, "")}
                       for cid, s in means.items() if s < 1.0), key=lambda f: f["score"])
    return Score(mean, len(means), infra, means, failures)


def infra_too_high(s: Score, attempted: int) -> bool:
    return attempted > 0 and (s.infra_errors / max(1, attempted)) > _MAX_INFRA_FRACTION


# -------------------------------------------------------------------------------- decision

def decide(base_train: Score, base_test: Score, cand_train: Score, cand_test: Score,
           margin: float = 0.02, test_gain: float = 0.01) -> dict:
    """Accept only when train gains ``margin`` and the held-out test gains ``test_gain``."""
    dtr = cand_train.mean - base_train.mean
    dte = cand_test.mean - base_test.mean
    if dtr < margin:
        verdict, why = "reject", f"train gain {dtr:+.3f} < margin {margin}"
    elif dte < test_gain:
        kind = "regressed" if dte < 0 else "plateaued"
        verdict, why = "revert", f"train {dtr:+.3f} but test {kind} ({dte:+.3f} < {test_gain}): overfit"
    else:
        verdict, why = "accept", f"train {dtr:+.3f}, test {dte:+.3f}"
    return {"verdict": verdict, "reason": why,
            "train_delta": round(dtr, 4), "test_delta": round(dte, 4)}


# ------------------------------------------------------------------------------- proposers

@dataclass
class Patch:
    key: str
    value: Any
    rationale: str = ""


def _prompt_for(ctx: dict) -> str:
    lines = [
        "You tune ONE setting of a system to fix its failing examples. Propose exactly one change.",
        "Settings you may change (key: kind, description, current value):",
    ]
    for s in ctx["surfaces"].values():
        cur = ctx["overlay"].get(s.key, s.default)
        extra = ""
        if s.kind == "number":
            extra = f" range [{s.lo}, {s.hi}]"
        elif s.kind == "choice":
            extra = f" one of {list(s.choices)}"
        elif s.kind == "text":
            extra = f" max {s.max_len} chars" + (f", must contain {list(s.must_keep)}" if s.must_keep else "")
        lines.append(f"- {s.key}: {s.kind}{extra}. {s.desc} Current: {json.dumps(cur)[:300]}")
    if ctx["history"]:
        lines.append("Already tried (do not repeat a rejected change):")
        for h in ctx["history"][-6:]:
            lines.append(f"- {h['key']}={json.dumps(h['value'])[:160]} -> {h['verdict']}")
    lines.append("Failing examples (score below 1):")
    for f in ctx["failures"][:5]:
        lines.append(f"- case {f['id']} score {f['score']}: {f['trace'][:_TRACE_HEAD]}")
    lines.append('Reply with ONLY JSON: {"key": "<setting>", "value": <new value>, "rationale": "<one sentence>"}')
    return "\n".join(lines)


class LLMProposer:
    """Asks a local model for one change. Fail-open: returns None when it cannot produce one."""

    def __init__(self, gen_fn: Optional[Callable[..., dict]] = None, model: str = ""):
        self._gen = gen_fn
        self._model = model or os.environ.get("LOCI_HILLCLIMB_MODEL", "")

    def _generate(self, prompt: str) -> dict:
        if self._gen is not None:
            return self._gen(prompt, fmt="json", max_tokens=700)
        try:
            from llm_local import generate
            model = self._model
            if not model:
                import backends
                model = backends.ollama_gen_model()
            return generate(prompt, model=model, fmt="json", max_tokens=700, temperature=0.4)
        except Exception as exc:
            return {"ok": False, "why": str(exc)}

    def propose(self, ctx: dict) -> Optional[Patch]:
        try:
            res = self._generate(_prompt_for(ctx))
            if not isinstance(res, dict) or not res.get("ok"):
                return None
            from model_json import extract_json_object
            obj = extract_json_object(str(res.get("text", "")))
            if not obj or "key" not in obj or "value" not in obj:
                return None
            return Patch(str(obj["key"]), obj["value"], str(obj.get("rationale", ""))[:300])
        except Exception:
            return None


# ------------------------------------------------------------------------------ diagnostics

def diagnose(suite, train: list[Case], test: list[Case], overlay: dict,
             base_train: Score, base_test: Optional[Score] = None, saturated: float = 0.95) -> dict:
    """Is the grader fit to climb on? Re-scores a sample and checks size and headroom."""
    warnings: list[str] = []
    ok = True
    if len(train) < _MIN_CASES_PER_SPLIT or len(test) < _MIN_CASES_PER_SPLIT:
        warnings.append(f"small split (train {len(train)}, test {len(test)}; want >= {_MIN_CASES_PER_SPLIT} each): "
                        "gains will be noisy, gather more cases")
    sample = train[:10]
    again = score_set(suite, sample, overlay)
    drift = [abs(again.per_case[c.id] - base_train.per_case[c.id])
             for c in sample if c.id in again.per_case and c.id in base_train.per_case]
    consistency = round(1.0 - (sum(drift) / len(drift)), 4) if drift else 1.0
    if consistency < 0.9:
        ok = False
        warnings.append(f"grader inconsistent (re-score agreement {consistency}): fix noise before climbing")
    if base_train.mean >= saturated:
        ok = False
        warnings.append(f"baseline {base_train.mean:.3f} >= {saturated}: no headroom, cases are too easy")
    if base_test is not None and base_test.n and base_test.mean >= saturated:
        ok = False
        warnings.append(f"test baseline {base_test.mean:.3f} >= {saturated}: no gain can be confirmed on the "
                        "held-out split, so no patch could pass the guard; add harder or more test cases")
    if base_train.n == 0:
        ok = False
        warnings.append("no case produced a score")
    return {"ok": ok, "consistency": consistency, "warnings": warnings}


# ------------------------------------------------------------------------------------ climb

@dataclass
class ClimbConfig:
    rounds: int = 6
    margin: float = 0.02
    test_gain: float = 0.01
    repeats: int = 1
    patience: int = 3
    target: float = 0.98
    test_frac: float = 0.3
    seed: str = ""
    force: bool = False  # climb even when the diagnostics object


def _log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _stall_report(failures: list[dict], n_test: int, history: list[dict]) -> dict:
    rejected = sum(1 for h in history if h["verdict"] != "accept")
    advice = ("gather more cases: the test split is too small to confirm any gain" if n_test < 30
              else "accept the plateau, or add a surface the failures point to")
    return {"remaining_failures": [{"id": f["id"], "score": f["score"]} for f in failures[:10]],
            "rejected_patches": rejected, "advice": advice}


def climb(suite, proposer, overlay0: Optional[dict] = None, cfg: Optional[ClimbConfig] = None,
          log_path: Optional[Path] = None) -> dict:
    """Run the loop. Returns {run_id, stop, overlay, baseline, final, rounds, diagnostics, stall}."""
    cfg = cfg or ClimbConfig()
    surfaces: dict[str, Surface] = dict(suite.surfaces)
    overlay = dict(overlay0 or {})
    run_id = uuid.uuid4().hex[:10]
    log = log_path or (suite_dir(suite.name) / "runs.jsonl")

    train, test = split(list(suite.cases()), cfg.test_frac, cfg.seed)
    cur_tr = score_set(suite, train, overlay, cfg.repeats)
    cur_te = score_set(suite, test, overlay, cfg.repeats)
    out: dict = {"run_id": run_id, "suite": suite.name, "overlay": dict(overlay), "rounds": [],
                 "baseline": {"train": cur_tr.summary(), "test": cur_te.summary()}}

    if infra_too_high(cur_tr, len(train) * cfg.repeats) or infra_too_high(cur_te, len(test) * cfg.repeats):
        out.update(stop="infra_noise", diagnostics={"ok": False, "warnings": ["too many infrastructure failures at baseline"]})
        _log(log, {"run": run_id, "event": "end", **{k: out[k] for k in ("stop", "baseline")}})
        return out

    diag = diagnose(suite, train, test, overlay, cur_tr, cur_te)
    out["diagnostics"] = diag
    _log(log, {"run": run_id, "event": "start", "suite": suite.name, "baseline": out["baseline"],
               "diagnostics": diag, "n_train": len(train), "n_test": len(test), "ts": int(time.time())})
    if not diag["ok"] and not cfg.force:
        out["stop"] = "diagnostics"
        _log(log, {"run": run_id, "event": "end", "stop": "diagnostics"})
        return out

    history: list[dict] = []
    misses = 0
    stop = "rounds"
    for rnd in range(1, cfg.rounds + 1):
        if cur_tr.mean >= cfg.target:
            stop = "saturated"
            break
        ctx = {"surfaces": surfaces, "overlay": dict(overlay), "failures": cur_tr.failures, "history": history}
        patch = proposer.propose(ctx)
        row: dict = {"run": run_id, "event": "round", "round": rnd}
        if patch is None:
            row.update(verdict="no_proposal")
            _log(log, row)
            out["rounds"].append(row)
            misses += 1
            if misses >= cfg.patience:
                stop = "patience"
                break
            continue
        surface = surfaces.get(patch.key)
        ok, value, why = (surface.validate(patch.value) if surface else (False, None, "not an allowed surface"))
        row.update(key=patch.key, value=patch.value if ok else str(patch.value)[:200], rationale=patch.rationale)
        if not ok:
            row.update(verdict="invalid", reason=why)
        elif overlay.get(patch.key, surface.default) == value:
            row.update(verdict="invalid", reason="no change from current value")
        else:
            cand = dict(overlay)
            cand[patch.key] = value
            cand_tr = score_set(suite, train, cand, cfg.repeats)
            if cand_tr.mean - cur_tr.mean < cfg.margin:
                # No train gain: skip the test split, it can only cost time.
                d = {"verdict": "reject", "reason": f"train gain {cand_tr.mean - cur_tr.mean:+.3f} < margin {cfg.margin}",
                     "train_delta": round(cand_tr.mean - cur_tr.mean, 4), "test_delta": None}
                cand_te = None
            else:
                cand_te = score_set(suite, test, cand, cfg.repeats)
                d = decide(cur_tr, cur_te, cand_tr, cand_te, cfg.margin, cfg.test_gain)
            row.update(d, train=cand_tr.summary(), test=cand_te.summary() if cand_te else None)
            if d["verdict"] == "accept":
                overlay, cur_tr, cur_te = cand, cand_tr, cand_te
                misses = 0
        history.append({"key": patch.key, "value": row.get("value"), "verdict": row["verdict"]})
        _log(log, row)
        out["rounds"].append(row)
        if row["verdict"] != "accept":
            misses += 1
            if misses >= cfg.patience:
                stop = "patience"
                break
    out.update(stop=stop, overlay=dict(overlay), final={"train": cur_tr.summary(), "test": cur_te.summary()})
    if stop in ("patience", "rounds"):
        out["stall"] = _stall_report(cur_tr.failures, len(test), history)
    _log(log, {"run": run_id, "event": "end", "stop": stop, "final": out["final"], "overlay_keys": sorted(overlay)})
    d = suite_dir(suite.name)
    _atomic_write(d / "candidate_overlay.json", overlay)
    return out


# ------------------------------------------------------------------------------ promotion

def promote(suite_name: str) -> dict:
    """Copy the candidate overlay to the live ``overlay.json`` (a previous one is kept as .prev)."""
    d = suite_dir(suite_name)
    cand = _read_json(d / "candidate_overlay.json")
    if not cand:
        return {"ok": False, "why": "no candidate overlay (run first, or the run improved nothing)"}
    live = d / "overlay.json"
    if live.exists():
        os.replace(live, d / "overlay.prev.json")
    _atomic_write(live, cand)
    _log(d / "runs.jsonl", {"event": "promote", "overlay_keys": sorted(cand), "ts": int(time.time())})
    return {"ok": True, "keys": sorted(cand)}


def rollback(suite_name: str) -> dict:
    """Restore the previous overlay, or remove the live one (back to built-in defaults)."""
    d = suite_dir(suite_name)
    prev, live = d / "overlay.prev.json", d / "overlay.json"
    if prev.exists():
        os.replace(prev, live)
        res = {"ok": True, "restored": "previous"}
    elif live.exists():
        live.unlink()
        res = {"ok": True, "restored": "defaults"}
    else:
        return {"ok": False, "why": "nothing promoted"}
    _log(d / "runs.jsonl", {"event": "rollback", "ts": int(time.time())})
    return res


def status(suite_name: str) -> dict:
    d = suite_dir(suite_name)
    runs = []
    try:
        for line in (d / "runs.jsonl").read_text(encoding="utf-8").splitlines()[-200:]:
            runs.append(json.loads(line))
    except Exception:
        pass
    ends = [r for r in runs if r.get("event") == "end"]
    return {"suite": suite_name, "live_overlay": _read_json(d / "overlay.json"),
            "candidate_overlay": _read_json(d / "candidate_overlay.json"),
            "last_run": ends[-1] if ends else None, "runs": len(ends)}


# ----------------------------------------------------------------- observations and labels
#
# Real cases for the triage suite. The reflection tick sees each observation's inputs (kind, path,
# event/tool counts, errors, warnings) and then drops them, so ``capture_observations`` keeps a
# scrubbed, de-duplicated copy, and ``label`` lets a person give each one a category. Labels become
# cases the suite reads next to the synthetic ones. Files, under the suite folder:
#   observations.jsonl  captured inputs (id, kind, path tail, counts, short scrubbed error text)
#   labels.jsonl        case records {id, gold, kind, path, events, tools, errors, warnings}
#   skipped.json        ids the person passed on
#   rejected.jsonl      observations marked "not a failure" (ids + error keys); capture drops matching ones

import re as _re

LABELS = {"r": "real_regression", "f": "flaky_or_nondeterministic", "c": "config_or_environment",
          "n": "noise_or_benign", "u": "unknown"}
_OBS_TEXT = 200
_OBS_KEYS = 6
_EXTRA_SECRETS = [
    _re.compile(r"AKIA[0-9A-Z]{12,}"),
    _re.compile(r"\b[0-9a-fA-F]{32,}\b"),
    _re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    _re.compile(r"(?i)(password|passwd|secret|token)\s*[:=]\s*\S+"),
]


_NOTE_TEXT = 300


def _scrub_text(s: str, cap: int = _OBS_TEXT) -> str:
    text = str(s)
    try:
        from llm_local import _sanitize_for_cloud
        text = _sanitize_for_cloud(text, max_len=4000)
    except Exception:
        pass
    for rx in _EXTRA_SECRETS:
        text = rx.sub("[REDACTED]", text)
    return text.replace("\n", " ")[:cap]


# The tick also reports session message text as "errors" (workflow-harness task prompts, teammate
# messages, pasted analysis). That is conversation content, not a failure to triage, and labelling it
# teaches the classifier nothing. Real tool errors arrive as short normalised templates
# ("claude tool_result error: exit code <n>"), so anything long or shaped like a message is dropped.
_PROSE_LEN = 120
_PROSE_PREFIXES = ("[workflow harness", "<teammate-message", "**")


def _is_prose(key: Any) -> bool:
    k = str(key).lstrip()
    return len(k) > _PROSE_LEN or k.startswith(_PROSE_PREFIXES)


def _clean_counts(d: Any, text_keys: bool = False, drop_prose: bool = False) -> dict:
    out: dict = {}
    for k, v in (d or {}).items():
        if drop_prose and _is_prose(k):
            continue
        if len(out) >= _OBS_KEYS:
            break
        key = _scrub_text(k) if text_keys else str(k)[:60]
        out[key] = v if isinstance(v, (int, float)) else _scrub_text(v)
    return out


def _path_tail(p: Any) -> str:
    return "/".join(str(p or "").replace("\\", "/").split("/")[-3:])[:120]


def observation_id(item: dict) -> str:
    """Stable id from what the classifier sees, not where it came from: the same pattern in a
    hundred files is one thing to label."""
    sig = json.dumps([item.get("kind"), item.get("events"), item.get("tools"),
                      item.get("errors"), item.get("warnings")], sort_keys=True, default=str)
    return "r" + hashlib.sha1(sig.encode("utf-8", "replace")).hexdigest()[:10]


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return rows


def _obs_keys(obs: dict) -> set:
    return set(obs.get("errors") or {}) | set(obs.get("warnings") or {})


def _rejected(suite: str) -> tuple[set, set]:
    """(ids, keys) the person marked 'not a failure'. A later observation is dropped when its id is
    rejected, or when every error/warning key it has was rejected before."""
    rows = _read_jsonl(suite_dir(suite) / "rejected.jsonl")
    return {r.get("id") for r in rows}, {k for r in rows for k in r.get("keys", [])}


def _is_rejected(obs: dict, ids: set, keys: set) -> bool:
    ks = _obs_keys(obs)
    return obs.get("id") in ids or (bool(ks) and ks <= keys)


def capture_observations(items: list, suite: str = "reflection_triage") -> int:
    """Append new, scrubbed observations from a tick's ``batch``; returns how many were new.
    Only processed items are kept. Never raises (a capture problem must not break the loop)."""
    try:
        path = suite_dir(suite) / "observations.jsonl"
        seen = {r.get("id") for r in _read_jsonl(path)}
        rej_ids, rej_keys = _rejected(suite)
        new = []
        for it in items or []:
            if not isinstance(it, dict) or it.get("status") != "processed":
                continue
            obs = {"kind": str(it.get("kind") or "")[:40], "path": _path_tail(it.get("path")),
                   "events": _clean_counts(it.get("events")), "tools": _clean_counts(it.get("tools")),
                   "errors": _clean_counts(it.get("errors"), text_keys=True, drop_prose=True),
                   "warnings": _clean_counts(it.get("warnings"), text_keys=True, drop_prose=True)}
            if not (obs["errors"] or obs["warnings"]):   # nothing left that looks like a failure
                continue
            obs["id"] = observation_id(obs)
            if obs["id"] in seen or _is_rejected(obs, rej_ids, rej_keys):
                continue
            seen.add(obs["id"])
            obs["ts"] = int(time.time())
            new.append(obs)
        if new:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                for o in new:
                    fh.write(json.dumps(o, sort_keys=True) + "\n")
        return len(new)
    except Exception:
        return 0


def prune_observations(suite: str = "reflection_triage") -> dict:
    """Re-apply the prose filter to observations captured before it existed: drop message-text keys
    (including ones truncated at the cap), drop rows with no error or warning left, re-derive ids and
    merge duplicates. Labels are untouched. Returns {before, after}."""
    path = suite_dir(suite) / "observations.jsonl"
    rows = _read_jsonl(path)
    rej_ids, rej_keys = _rejected(suite)
    keep, seen = [], set()
    for r in rows:
        for field in ("errors", "warnings"):
            r[field] = {k: v for k, v in (r.get(field) or {}).items() if not _is_prose(k) and len(k) < _OBS_TEXT}
        if not (r["errors"] or r["warnings"]):
            continue
        r["id"] = observation_id(r)
        if r["id"] in seen or _is_rejected(r, rej_ids, rej_keys):
            continue
        seen.add(r["id"])
        keep.append(r)
    if rows:   # rows were edited in place, so a before/after comparison would see no change: always rewrite
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(o, sort_keys=True) + "\n" for o in keep), encoding="utf-8")
        os.replace(tmp, path)
    return {"before": len(rows), "after": len(keep)}


def _show(obs: dict) -> str:
    def fmt(d: dict) -> str:
        return ", ".join(f"{k} x{v}" if isinstance(v, int) else f"{k}: {v}" for k, v in d.items()) or "-"
    return (f"  kind:     {obs.get('kind')}\n  path:     {obs.get('path')}\n  events:   {fmt(obs.get('events') or {})}\n"
            f"  tools:    {fmt(obs.get('tools') or {})}\n  errors:   {fmt(obs.get('errors') or {})}\n"
            f"  warnings: {fmt(obs.get('warnings') or {})}")


NOVELTY = {"k": "known_pattern", "w": "novel_signal", "": "unclear", "?": "unclear"}

PROMPT = ("[r]egression  [f]laky  [c]onfig/env  [n]oise  [u]nknown  "
          "[x] not a failure  [s]kip  [q]uit > ")
DETAIL_PROMPT = "  novelty [k]nown / [w] new / Enter=unclear, optional note after a colon (k: why) > "
REJECT_PROMPT = "  why is it not a failure? (optional) > "


def parse_detail(text: str) -> tuple[str, str]:
    """``'k: from the wf script'`` -> ("known_pattern", note). The part before the colon is the novelty
    key (k, w, ? or empty); if it is anything else the whole text is the note and novelty is unclear."""
    head, sep, tail = str(text).partition(":")
    key = head.strip().lower()
    if key in NOVELTY:
        return NOVELTY[key], _scrub_text(tail.strip(), _NOTE_TEXT)
    return "unclear", _scrub_text(str(text).strip(), _NOTE_TEXT)


def label(suite: str = "reflection_triage", n: int = 30, include_skipped: bool = False,
          input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print,
          classify: Optional[Callable[[dict], Optional[str]]] = None,
          decide: Optional[Callable[[dict], Optional[tuple]]] = None) -> dict:
    """Label up to ``n`` unlabelled observations. Per item: a category (or ``x`` = not a failure, which
    also teaches capture to drop that kind of item), then one line for novelty and an optional note.
    ``classify(obs)`` (optional) returns the model's category, shown only AFTER you answer so it
    cannot bias you; agreement is tallied. ``decide(obs)`` (optional) is the same for the decision model and
    returns ``(category, confidence)``; it is tallied separately, so both can be compared with your answers."""
    d = suite_dir(suite)
    obs = _read_jsonl(d / "observations.jsonl")
    done = {r.get("id") for r in _read_jsonl(d / "labels.jsonl")}
    rej_ids, _ = _rejected(suite)
    done |= rej_ids
    skipped = set(_read_json(d / "skipped.json").get("ids", []))
    todo = [o for o in obs if o["id"] not in done and (include_skipped or o["id"] not in skipped)][:n]
    tally = {"labelled": 0, "skipped": 0, "rejected": 0, "compared": 0, "agreed": 0}
    if decide is not None:
        tally.update(decide_compared=0, decide_agreed=0)
    print_fn(f"{len(todo)} to label ({len(obs)} captured, {len(done)} labelled or rejected, {len(skipped)} skipped)")
    for i, o in enumerate(todo, 1):
        print_fn(f"\n--- {i}/{len(todo)}  {o['id']}\n{_show(o)}")
        while True:
            ans = input_fn(PROMPT).strip().lower()[:1]
            if ans in LABELS or ans in ("s", "q", "x"):
                break
            print_fn("  r f c n u x s q")
        if ans == "q":
            break
        if ans == "s":
            skipped.add(o["id"])
            tally["skipped"] += 1
            continue
        if ans == "x":
            keys = sorted(_obs_keys(o))
            row = {"id": o["id"], "keys": keys, "note": _scrub_text(input_fn(REJECT_PROMPT).strip(), _NOTE_TEXT),
                   "ts": int(time.time())}
            with (d / "rejected.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
            skipped.discard(o["id"])
            tally["rejected"] += 1
            print_fn(f"  will no longer capture items whose only errors/warnings are: {keys}")
            continue
        novelty, note = parse_detail(input_fn(DETAIL_PROMPT))
        case = {"id": o["id"], "gold": LABELS[ans], "novelty": novelty, "note": note, "kind": o["kind"],
                "path": o["path"], "events": o["events"], "tools": o["tools"], "errors": o["errors"],
                "warnings": o["warnings"]}
        with (d / "labels.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(case, sort_keys=True) + "\n")
        skipped.discard(o["id"])
        tally["labelled"] += 1
        if classify is not None:
            try:
                got = classify(o)
            except Exception:
                got = None
            if got:
                tally["compared"] += 1
                tally["agreed"] += int(got == case["gold"])
                print_fn(f"  model said {got} ({'agrees' if got == case['gold'] else 'differs'})")
        if decide is not None:
            try:
                got_d = decide(o)
            except Exception:
                got_d = None
            if got_d and got_d[0]:
                tally["decide_compared"] += 1
                tally["decide_agreed"] += int(got_d[0] == case["gold"])
                print_fn(f"  decision model said {got_d[0]} (confidence {got_d[1]:.2f}) "
                         f"({'agrees' if got_d[0] == case['gold'] else 'differs'})")
    _atomic_write(d / "skipped.json", {"ids": sorted(skipped)})
    return tally


def label_stats(suite: str = "reflection_triage") -> dict:
    """How far the real label set is from being useful: counts, per-category, and split sizes."""
    d = suite_dir(suite)
    obs = _read_jsonl(d / "observations.jsonl")
    labels = _read_jsonl(d / "labels.jsonl")
    skipped = _read_json(d / "skipped.json").get("ids", [])
    rej_ids, rej_keys = _rejected(suite)
    by_cat: dict[str, int] = {}
    by_nov: dict[str, int] = {}
    for r in labels:
        by_cat[r["gold"]] = by_cat.get(r["gold"], 0) + 1
        nov = r.get("novelty") or "unrated"
        by_nov[nov] = by_nov.get(nov, 0) + 1
    tr, te = split([Case(r["id"]) for r in labels])
    handled = {r["id"] for r in labels} | rej_ids
    return {"captured": len(obs), "labelled": len(labels), "skipped": len(skipped),
            "rejected": len(rej_ids), "rejected_keys": len(rej_keys),
            "with_notes": sum(1 for r in labels if r.get("note")),
            "unlabelled": len([o for o in obs if o["id"] not in handled]),
            "by_category": by_cat, "by_novelty": by_nov, "real_train": len(tr), "real_test": len(te),
            "enough": len(tr) >= 30 and len(te) >= 30}


# ---------------------------------------------------------------------------------- CLI

def load_suite(spec: str):
    """``triage`` (built-in) or ``module:factory`` for a suite that lives elsewhere."""
    if ":" in spec:
        mod, _, attr = spec.partition(":")
        return getattr(importlib.import_module(mod), attr)()
    suites = importlib.import_module("hillclimb_suites")
    return suites.BUILTIN[spec]()


def suite_name(spec: str) -> str:
    """The on-disk suite name for a CLI spec: a built-in alias or ``module:factory`` resolves to the
    suite's own ``name``; anything else is taken as the name itself."""
    suites = importlib.import_module("hillclimb_suites")
    if ":" in spec or spec in suites.BUILTIN:
        return load_suite(spec).name
    return spec


def _main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="hillclimb")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--suite", required=True)
    r.add_argument("--rounds", type=int, default=6)
    r.add_argument("--margin", type=float, default=0.02)
    r.add_argument("--test-gain", type=float, default=0.01)
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--patience", type=int, default=3)
    r.add_argument("--force", action="store_true")
    for name in ("status", "promote", "rollback"):
        s = sub.add_parser(name)
        s.add_argument("--suite", required=True)
    lb = sub.add_parser("label", help="label captured observations (becomes real test cases)")
    lb.add_argument("--suite", default="triage")
    lb.add_argument("--n", type=int, default=30)
    lb.add_argument("--skipped", action="store_true", help="show observations you skipped before")
    lb.add_argument("--model", action="store_true", help="show what the classifier says AFTER you answer")
    lb.add_argument("--decide", action="store_true",
                    help="show what the decision model (decide.py) says AFTER you answer, with its confidence")
    sub.add_parser("prune", help="drop message-text observations captured before the prose filter").add_argument("--suite", default="triage")
    ls = sub.add_parser("labels", help="how many real labels exist and whether they are enough")
    ls.add_argument("--suite", default="triage")
    a = p.parse_args(argv)
    if a.cmd == "label":
        classify = None
        if a.model:
            def classify(o):
                from reflection_triage import classify_reflection_observation
                return classify_reflection_observation(o["kind"], o["path"], events=o["events"], tools=o["tools"],
                                                       errors=o["errors"], warnings=o["warnings"]).get("category")
        decide_cb = None
        if a.decide:
            def decide_cb(o):
                from reflection_triage import classify_reflection_observation_decide
                r = classify_reflection_observation_decide(o["kind"], o["path"], events=o["events"], tools=o["tools"],
                                                           errors=o["errors"], warnings=o["warnings"])
                return (r.get("category"), r.get("confidence") or 0.0) if r.get("ok") else None
        res = label(suite_name(a.suite), a.n, a.skipped, classify=classify, decide=decide_cb)
        print(json.dumps(res))
        return 0
    if a.cmd == "prune":
        print(json.dumps(prune_observations(suite_name(a.suite))))
        return 0
    if a.cmd == "labels":
        print(json.dumps(label_stats(suite_name(a.suite)), indent=1))
        return 0
    if a.cmd == "run":
        suite = load_suite(a.suite)
        start = _read_json(suite_dir(suite.name) / "overlay.json")
        cfg = ClimbConfig(rounds=a.rounds, margin=a.margin, test_gain=a.test_gain,
                          repeats=a.repeats, patience=a.patience, force=a.force)
        res = climb(suite, LLMProposer(), start, cfg)
    elif a.cmd == "status":
        res = status(suite_name(a.suite))
    elif a.cmd == "promote":
        res = promote(suite_name(a.suite))
    else:
        res = rollback(suite_name(a.suite))
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
