"""Model pool: declared candidates per role, ranked, residency- and size-aware.

Each role resolver in ``backends`` used to carry its own hardcoded preference tuple and
take the first installed tag. A configured tag that was not installed silently broke
generation (the tag was returned anyway), and nothing said which model should serve a
role instead. The pool is one declared list the resolvers consult:

    [models]
    resident_bonus = 0.5     # rank credit for a model Ollama already holds in memory
    max_vram_gb = 10         # optional: prefer entries that fit one GPU
    over_cap_fallback = true # when NOTHING fits, relax the cap and use the best-ranked model anyway

    [[models.pool]]
    name = "gemma4-e4b-hermes:64k"
    roles = ["gen", "verify", "compress"]
    rank = 1                 # lower is preferred
    vram_gb = 5.0            # optional; otherwise the size /api/tags reports
    pinned = false           # never evicted by a model lease (model_lease.py)
    evictable = true         # false behaves like pinned for leases
    role_rank = { verify = 2 }   # optional: rank differently for one role

Resolution for one role: take the entries listing the role, keep those installed at the
generation endpoint (and under ``max_vram_gb``), and order by ``rank - resident_bonus``
(ties break on rank, then name). With ``over_cap_fallback`` the cap only binds while some
installed candidate fits: when none does, the over-cap candidates become eligible and the
best-ranked one (the strongest, by the operator's ranking) is used. An operator-configured tag (``[ollama].gen_model`` etc.)
joins as an implicit rank 0 so existing configs keep their behaviour while the tag is
installed and fall through to the pool when it is not.

Off unless ``[[models.pool]]`` exists: with no pool every ``pick`` returns "" and the
legacy resolvers behave exactly as before.

Shadow selector (the FlyBrain integration shape, docs/flybrain_brains_eval.md): the
rule above always decides. With ``LOCI_MODEL_POOL_SHADOW=1`` each decision is logged next
to what an optional learned selector (``LOCI_MODEL_POOL_SELECTOR=module:callable``) would
have chosen. Rows carry model names, ranks and enums only; no prompt or output text. A
selector that raises, hangs the import or returns junk is ignored. Rollback: unset the
variables.

Training and testing a selector (a FlyBrain-style brain, or the baseline in ``pool_selectors``):

* Every decision row has a ``decision_id``; the outcome of the call it led to carries the same id
  (plus ``pool_role`` and a ``prompt_bucket`` size class), so the join is exact, not "the next
  outcome for that model".
* ``shadow_status`` says why ``chosen_shadow`` is empty: ``no_selector``, ``abstained``, ``invalid``
  or ``error``; ``chose`` when it is not.
* Shadow alone cannot teach anything about an arm the rule never picks: no outcome exists for it.
  ``LOCI_MODEL_POOL_EXPLORE=<p>`` with ``LOCI_MODEL_POOL_EXPLORE_ROLES=gen`` is the opt-in way to
  get that data: with probability ``p`` a call for one of those roles goes to one eligible
  alternative (uniformly; eligible already means installed and within the VRAM cap), and
  ``propensity`` records how likely the arm actually used was. No model list is needed;
  ``LOCI_MODEL_POOL_EXPLORE_MODELS=a,b`` optionally narrows the alternatives, and on its own it
  applies to every role. It is off unless ``p``, roles or models are set and the shadow is on. It
  changes live routing for that fraction of calls, and a non-resident model has to be loaded.
* ``python model_pool.py report`` joins the logs and says, per role, whether the data can support
  learning (two or more arms with enough outcomes) or only one arm has ever been observed.
"""
from __future__ import annotations

import contextvars
import importlib
import json
import logging
import os
import random
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Optional

logger = logging.getLogger("loci-mcp.model_pool")

SHADOW_ENV = "LOCI_MODEL_POOL_SHADOW"
SELECTOR_ENV = "LOCI_MODEL_POOL_SELECTOR"
EXPLORE_ENV = "LOCI_MODEL_POOL_EXPLORE"
EXPLORE_MODELS_ENV = "LOCI_MODEL_POOL_EXPLORE_MODELS"
EXPLORE_ROLES_ENV = "LOCI_MODEL_POOL_EXPLORE_ROLES"
SHADOW_LOG_NAME = "model_pool_shadow.jsonl"
SHADOW_SCHEMA = 1
OUTCOMES_LOG_NAME = "model_pool_outcomes.jsonl"
OUTCOME_SCHEMA = 1

_LINK_TTL_S = 300.0           # an outcome links to the decision made this recently, in the same thread
MIN_ARM_N = 30                # outcomes an arm needs before the report counts it as observed
_LAST_DECISION: contextvars.ContextVar = contextvars.ContextVar("loci_model_pool_last_decision", default=None)
_rng = random.Random()

DEFAULT_RESIDENT_BONUS = 0.5
_TAGS_TTL_S = 30.0
_PS_TTL_S = 5.0
_STALE_MAX_S = 600.0          # a failed refresh serves the last good answer for up to this long
_HTTP_TIMEOUT_S = 3.0

# Substring -> roles, first match wins. Order matters: guard/embed before the generic chat families.
_ROLE_HINTS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("-cpu",), ("gen_cpu",)),                 # CPU-pinned variants must not compete for the GPU gen slot
    (("embed",), ("embed",)),
    (("guard",), ("guardian", "safety")),
    (("coder", "codestral", "starcoder", "deepseek-coder"), ("code",)),
    (("math",), ("math",)),
    (("watt-tool", "tool-calling", "functionary", "gorilla"), ("tool",)),
    (("minicpm-v", "llava", "vision", "-vl", ":vl", "moondream", "bakllava"), ("vision",)),
    (("heretic", "abliterat", "uncensored"), ("redteam",)),   # add "gen" by hand if you want one as a general model
)


@dataclass(frozen=True)
class PoolEntry:
    name: str
    roles: tuple[str, ...]
    rank: float = 100.0
    vram_gb: Optional[float] = None
    pinned: bool = False
    role_rank: tuple[tuple[str, float], ...] = ()
    evictable: bool = True        # False (or pinned) keeps a model resident through a lease (model_lease)

    def rank_for(self, role: str) -> float:
        for r, value in self.role_rank:
            if r == role:
                return value
        return self.rank


@dataclass
class Candidate:
    name: str
    rank: float
    effective_rank: float
    installed: bool
    resident: bool
    fits: bool
    eligible: bool
    reason: str = ""
    over_cap: bool = False        # eligible only because nothing fit the cap (over_cap_fallback)
    source: str = "pool"          # "pool" or "configured" (the operator's own tag)


@dataclass
class Decision:
    role: str
    chosen: str
    candidates: list[Candidate] = field(default_factory=list)
    resident_bonus: float = DEFAULT_RESIDENT_BONUS
    selector: str = "rules"

    def ordered(self) -> list[str]:
        return [c.name for c in self.candidates if c.eligible]


# ---------------------------------------------------------------------------------- config

def _config() -> dict:
    try:
        import backends
        return backends._config() or {}
    except Exception as exc:
        logger.debug("model_pool: config unavailable: %r", exc)
        return {}


def _models_cfg() -> dict:
    models = _config().get("models")
    return models if isinstance(models, dict) else {}


def _num(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def entries() -> list[PoolEntry]:
    """Parse ``[[models.pool]]``. Malformed rows are skipped, never raised."""
    raw = _models_cfg().get("pool")
    out: list[PoolEntry] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        roles_raw = item.get("roles")
        if isinstance(roles_raw, str):
            roles_raw = [roles_raw]
        roles = tuple(str(r).strip().lower() for r in (roles_raw or []) if str(r).strip())
        if not name or not roles:
            continue
        vram = item.get("vram_gb")
        rr = item.get("role_rank")
        role_rank = tuple(sorted(
            (str(k).strip().lower(), _num(v, 100.0)) for k, v in rr.items()
        )) if isinstance(rr, dict) else ()
        out.append(PoolEntry(
            name=name, roles=roles, rank=_num(item.get("rank"), 100.0),
            vram_gb=_num(vram, None) if vram is not None else None,
            pinned=bool(item.get("pinned", False)), role_rank=role_rank,
            evictable=bool(item.get("evictable", True)),
        ))
    return out


def configured() -> bool:
    return bool(entries())


def resident_bonus() -> float:
    return max(0.0, _num(_models_cfg().get("resident_bonus"), DEFAULT_RESIDENT_BONUS))


def max_vram_gb() -> Optional[float]:
    value = _models_cfg().get("max_vram_gb")
    return _num(value, None) if value is not None else None


def over_cap_fallback() -> bool:
    return bool(_models_cfg().get("over_cap_fallback", False))


# ------------------------------------------------------------------------------- inventory

_lock = threading.Lock()
_cache: dict[tuple[str, str], tuple[float, object]] = {}
_good: dict[tuple[str, str], tuple[float, object]] = {}     # last successful load, for stale-if-error


def _get_json(url: str) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT_S) as resp:  # noqa: S310 - operator-configured URL
            data = json.loads(resp.read().decode("utf-8", "replace"))
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.debug("model_pool: GET %s failed: %r", url, exc)
        return None


def _cached(kind: str, base_url: str, ttl: float, loader: Callable[[], object], empty):
    """TTL cache with stale-if-error. ``loader`` returns None when the fetch failed.

    One slow or refused request must not blank the inventory: with every candidate
    "not installed" the pool resolves to nothing and the legacy resolver returns a
    possibly missing tag. A failed refresh keeps serving the last good value (up to
    _STALE_MAX_S) and is itself cached for ``ttl`` so a down endpoint is not hammered.
    """
    key = (kind, base_url)
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
        good = _good.get(key)
    value = loader()
    with _lock:
        if value is None:
            stale = good if good and time.monotonic() - good[0] < _STALE_MAX_S else None
            value = stale[1] if stale else empty
        else:
            _good[key] = (time.monotonic(), value)
        _cache[key] = (time.monotonic(), value)
    return value


def clear_cache() -> None:
    with _lock:
        _cache.clear()
        _good.clear()


def _gen_url() -> str:
    try:
        import backends
        return backends.ollama_gen_url(0.5) or ""
    except Exception:
        return ""


def inventory(base_url: Optional[str] = None) -> dict[str, Optional[int]]:
    """tag -> size in bytes for what the generation endpoint has installed. {} when unreachable."""
    url = (base_url if base_url is not None else _gen_url()).rstrip("/")
    if not url:
        return {}

    def load():
        data = _get_json(url + "/api/tags")
        if data is None:
            return None
        return {str(m.get("name")): m.get("size") for m in (data.get("models") or []) if m.get("name")}

    return dict(_cached("tags", url, _TAGS_TTL_S, load, {}))


def resident_models(base_url: Optional[str] = None) -> set[str]:
    """Tags Ollama currently holds in memory (``/api/ps``). Empty when unreachable."""
    url = (base_url if base_url is not None else _gen_url()).rstrip("/")
    if not url:
        return set()

    def load():
        data = _get_json(url + "/api/ps")
        if data is None:
            return None
        return {str(m.get("name") or m.get("model")) for m in (data.get("models") or [])}

    return set(_cached("ps", url, _PS_TTL_S, load, set()))


def _installed(name: str, inv: dict) -> bool:
    return name in inv or (":" not in name and f"{name}:latest" in inv)


# ----------------------------------------------------------------------------- resolution

def rank_role(role: str, *, hint: str = "", pool: Optional[Iterable[PoolEntry]] = None,
              inv: Optional[dict] = None, resident: Optional[set] = None,
              bonus: Optional[float] = None, cap_gb: Optional[float] = None,
              fallback: Optional[bool] = None, held_out: Optional[set] = None) -> Decision:
    """Order the candidates for ``role``. Pure when ``pool``/``inv``/``resident`` are passed."""
    role = (role or "").strip().lower()
    pool_entries = list(entries() if pool is None else pool)
    inventory_ = inventory() if inv is None else inv
    resident_ = resident_models() if resident is None else resident
    bonus_ = resident_bonus() if bonus is None else bonus
    cap = max_vram_gb() if cap_gb is None else cap_gb
    relax = over_cap_fallback() if fallback is None else fallback
    if held_out is None:      # models an active lease evicted: do not reload one into the headroom it reserved
        try:
            import model_lease
            held_out = model_lease.held_out()
        except Exception:
            held_out = set()

    rows: list[tuple[PoolEntry, str]] = [(e, "pool") for e in pool_entries if role in e.roles]
    hint = (hint or "").strip()
    if hint:
        # The operator's own tag outranks the pool: rank 0, keeping any size/pin the pool declares for it.
        rows = [((replace(e, rank=0.0, role_rank=()), "configured") if e.name == hint else (e, src))
                for e, src in rows]
        if not any(e.name == hint for e, _ in rows):
            rows.append((PoolEntry(name=hint, roles=(role,), rank=0.0), "configured"))

    cands: list[Candidate] = []
    for entry, source in rows:
        installed = _installed(entry.name, inventory_)
        res = entry.name in resident_ or f"{entry.name}:latest" in resident_
        size = entry.vram_gb
        if size is None:
            raw = inventory_.get(entry.name) or inventory_.get(f"{entry.name}:latest")
            size = (raw / 1e9) if isinstance(raw, (int, float)) else None
        fits = cap is None or size is None or size <= cap
        evicted = entry.name in held_out or f"{entry.name}:latest" in held_out
        eligible = installed and fits and not evicted
        reason = ("" if eligible else "not installed" if not installed
                  else "evicted by an active lease" if evicted and fits
                  else f"{size:.1f} GB over the {cap:g} GB cap")
        rank = entry.rank_for(role)
        cands.append(Candidate(
            name=entry.name, rank=rank,
            effective_rank=rank - (bonus_ if res else 0.0),
            installed=installed, resident=res, fits=fits, eligible=eligible,
            reason=reason, source=source,
        ))
    if relax and cands and not any(c.eligible for c in cands):
        # Nothing installed fits the cap: use the strongest (best-ranked) over-cap model rather than none.
        for c in cands:
            if c.installed and not c.fits and not (c.name in held_out or f"{c.name}:latest" in held_out):
                c.eligible, c.over_cap = True, True
                c.reason = f"{c.reason}; nothing fits, used as the strongest available"
    cands.sort(key=lambda c: (not c.eligible, c.effective_rank, c.rank, c.name))
    chosen = next((c.name for c in cands if c.eligible), "")
    return Decision(role=role, chosen=chosen, candidates=cands, resident_bonus=bonus_)


def pick(role: str, hint: str = "") -> str:
    """Best eligible model for ``role``, or "" (no pool configured, role not pooled, or none installed)."""
    try:
        pool_entries = entries()
        if not pool_entries or not any((role or "").strip().lower() in e.roles for e in pool_entries):
            return ""
        decision = rank_role(role, hint=hint, pool=pool_entries)
        returned, explore = _explore(decision)
        _shadow(decision, returned=returned, explore=explore)
        return returned
    except Exception as exc:  # fail-open: the legacy resolver takes over
        logger.warning("model_pool.pick(%r) failed, using the legacy resolver: %r", role, exc)
        return ""


def summary() -> dict:
    """Health view: per role the chosen model, its rank, and whether the top rank was available."""
    pool_entries = entries()
    if not pool_entries:
        return {"configured": False}
    inv, res = inventory(), resident_models()
    roles = sorted({r for e in pool_entries for r in e.roles})
    out: dict = {}
    for role in roles:
        d = rank_role(role, pool=pool_entries, inv=inv, resident=res)
        top = min((c for c in d.candidates), key=lambda c: (c.rank, c.name), default=None)
        chosen = next((c for c in d.candidates if c.eligible), None)
        out[role] = {
            "chosen": d.chosen or None,
            "rank": chosen.rank if chosen else None,
            "degraded": bool(top and chosen and chosen.name != top.name) or not chosen,
            "over_cap": bool(chosen and chosen.over_cap),
        }
    return {"configured": True, "inventory_reachable": bool(inv), "roles": out}


# ------------------------------------------------------------------------- shadow selector

def _shadow_enabled() -> bool:
    raw = os.environ.get(SHADOW_ENV)
    return bool(raw and raw.strip().lower() not in ("", "0", "false", "no", "off"))


def _load_selector() -> Optional[Callable]:
    spec = (os.environ.get(SELECTOR_ENV) or "").strip()
    if not spec or ":" not in spec:
        return None
    module, _, attr = spec.partition(":")
    try:
        fn = getattr(importlib.import_module(module), attr)
        return fn if callable(fn) else None
    except Exception as exc:
        logger.debug("model_pool: selector %s not loadable: %r", spec, exc)
        return None


def _log_path() -> Path:
    import legacy_env
    mem = str(legacy_env.memory_dir())
    return Path(mem).parent / "instrumentation" / SHADOW_LOG_NAME


def _explore(decision: Decision) -> tuple:
    """Opt-in exploration: ``(model to use, info)``. Off unless the shadow is on, a probability in (0, 1] is set and
    roles or models are named. Only an eligible alternative to the rule's choice can be picked, from the model list
    when there is one, from every eligible candidate when there is not."""
    off = (decision.chosen, {"p": 0.0})
    try:
        if not _shadow_enabled() or not decision.chosen:
            return off
        p = float(os.environ.get(EXPLORE_ENV) or 0.0)
        allowed = {n.strip() for n in (os.environ.get(EXPLORE_MODELS_ENV) or "").split(",") if n.strip()}
        roles = {n.strip().lower() for n in (os.environ.get(EXPLORE_ROLES_ENV) or "").split(",") if n.strip()}
        if not (0.0 < p <= 1.0) or not (allowed or roles):
            return off
        if roles and decision.role not in roles:   # rank_role has already normalized the role
            return off
        alts = [n for n in decision.ordered() if n != decision.chosen and (not allowed or n in allowed)]
        info: dict = {"p": p, "n_alternatives": len(alts)}
        if not alts:
            return decision.chosen, info
        if _rng.random() < p:
            info.update(explored=True, propensity=round(p / len(alts), 6))
            return alts[_rng.randrange(len(alts))], info
        info.update(explored=False, propensity=round(1.0 - p, 6))
        return decision.chosen, info
    except Exception as exc:
        logger.debug("model_pool: exploration skipped: %r", exc)
        return off


def _shadow(decision: Decision, returned: Optional[str] = None, explore: Optional[dict] = None) -> None:
    """Log the rule decision beside the optional selector's. Never raises, never changes the pick."""
    if not _shadow_enabled():
        return
    try:
        started = time.perf_counter()
        returned = returned or decision.chosen
        decision_id = uuid.uuid4().hex[:12]
        alt = ""
        status = "no_selector"
        selector = _load_selector()
        if selector is not None:
            status = "abstained"
            features = [{"name": c.name, "rank": c.rank, "effective_rank": c.effective_rank,
                         "resident": c.resident, "eligible": c.eligible} for c in decision.candidates]
            try:
                result = selector(decision.role, features)
                names = decision.ordered()
                if isinstance(result, str) and result in names:
                    alt = result
                elif isinstance(result, (list, tuple)) and result and result[0] in names:
                    alt = str(result[0])
                if alt:
                    status = "chose"
                elif result not in (None, "", [], ()):
                    status = "invalid"          # it answered, but not with an eligible candidate
            except Exception as exc:
                status = "error"
                logger.debug("model_pool: shadow selector raised: %r", exc)
        row = {
            "schema": SHADOW_SCHEMA, "ts": time.time(), "role": decision.role,
            "decision_id": decision_id, "shadow_status": status, "chosen_returned": returned,
            "chosen_rule": decision.chosen, "chosen_shadow": alt or None,
            "agree": (alt == decision.chosen) if alt else None,
            "n_candidates": len(decision.candidates),
            "n_eligible": sum(1 for c in decision.candidates if c.eligible),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "candidates": [{"name": c.name, "rank": c.rank, "resident": c.resident,
                            "eligible": c.eligible} for c in decision.candidates],
        }
        if explore and explore.get("p"):
            row.update(explore_p=explore["p"], n_alternatives=explore.get("n_alternatives", 0),
                       explored=bool(explore.get("explored")), propensity=explore.get("propensity"))
        from instrumentation_log import append_rows
        append_rows(_log_path(), [row])
        _LAST_DECISION.set({"id": decision_id, "role": decision.role, "model": returned, "ts": time.time(),
                            "explored": bool(explore and explore.get("explored"))})
    except Exception as exc:
        logger.debug("model_pool: shadow log skipped: %r", exc)


def record_outcome(model: str, ok: bool, latency_ms: float, *, route_role: str = "",
                   deadline_exceeded: bool = False, tier: str = "", fmt: str = "",
                   decision_id: str = "", prompt_chars: int = 0) -> None:
    """Log how one generate() call went, so a pool decision has a label (D23 in docs/flybrain_brains_eval.md).

    Rows hold the model tag, ok, latency and enums only: no prompt, output or error text. Written
    only under ``LOCI_MODEL_POOL_SHADOW=1``. The row carries the ``decision_id`` of the pool decision that
    chose this model in the same thread within the last few minutes (each decision labels one call), so the
    join is exact; with none, the id is absent. ``prompt_bucket`` is the bit length of the prompt size, a
    size class and never any text. Never raises.
    """
    if not _shadow_enabled():
        return
    try:
        row = {
            "schema": OUTCOME_SCHEMA, "ts": time.time(), "model": str(model or ""),
            "ok": bool(ok), "latency_ms": round(float(latency_ms), 1),
            "deadline_exceeded": bool(deadline_exceeded), "route_role": str(route_role or ""),
            "tier": str(tier or "ollama"), "fmt": "json" if fmt == "json" else "",
        }
        if decision_id:
            row["decision_id"] = str(decision_id)
        else:
            cur = _LAST_DECISION.get()
            if cur and cur["model"] == str(model or "") and time.time() - cur["ts"] <= _LINK_TTL_S:
                row.update(decision_id=cur["id"], pool_role=cur["role"], explored=cur["explored"])
                _LAST_DECISION.set(None)
        if prompt_chars and int(prompt_chars) > 0:
            row["prompt_bucket"] = int(prompt_chars).bit_length()
        from instrumentation_log import append_rows
        append_rows(_log_path().with_name(OUTCOMES_LOG_NAME), [row])
    except Exception as exc:
        logger.debug("model_pool: outcome log skipped: %r", exc)


def outcomes_summary(path: Optional[Path] = None) -> dict:
    """Per model: calls, success rate, p50/p95 latency and deadline hits, from the outcomes log."""
    base = path or _log_path().with_name(OUTCOMES_LOG_NAME)
    files = [base] + [base.with_name(f"{base.name}.{i}") for i in range(1, 6)]
    by: dict[str, list[dict]] = {}
    for f in files:
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and row.get("model"):
                    by.setdefault(row["model"], []).append(row)
        except OSError:
            continue

    def pct(values: list[float], q: float) -> float:
        values = sorted(values)
        return values[min(len(values) - 1, int(q * len(values)))] if values else 0.0

    out = {}
    for model, rows in sorted(by.items()):
        lat = [float(r.get("latency_ms") or 0.0) for r in rows]
        out[model] = {
            "calls": len(rows),
            "ok_rate": round(sum(1 for r in rows if r.get("ok")) / len(rows), 3),
            "p50_ms": round(pct(lat, 0.5), 1), "p95_ms": round(pct(lat, 0.95), 1),
            "deadline_exceeded": sum(1 for r in rows if r.get("deadline_exceeded")),
        }
    return out


def _read_rows(base: Path) -> list[dict]:
    rows: list[dict] = []
    for f in [base] + [base.with_name(f"{base.name}.{i}") for i in range(1, 6)]:
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
        except OSError:
            continue
    return rows


def pool_report(decisions_path: Optional[Path] = None, outcomes_path: Optional[Path] = None,
                min_arm_n: int = MIN_ARM_N) -> dict:
    """Join decisions to outcomes by ``decision_id`` and say, per role, whether a selector could be trained or
    tested on this data. Rows from before the id existed are counted, not guessed at."""
    dpath = decisions_path or _log_path()
    opath = outcomes_path or _log_path().with_name(OUTCOMES_LOG_NAME)
    decisions, outcomes = _read_rows(dpath), _read_rows(opath)
    by_id = {r["decision_id"]: r for r in outcomes if r.get("decision_id")}
    roles: dict[str, dict] = {}
    legacy = 0
    for d in decisions:
        if not d.get("decision_id"):
            legacy += 1
            continue
        info = roles.setdefault(str(d.get("role")), {"decisions": 0, "linked": 0, "status": {}, "explored": 0, "_arms": {}})
        info["decisions"] += 1
        status = str(d.get("shadow_status") or "unknown")
        info["status"][status] = info["status"].get(status, 0) + 1
        out = by_id.get(d["decision_id"])
        if out is None:
            continue
        info["linked"] += 1
        info["explored"] += 1 if d.get("explored") else 0
        arm = info["_arms"].setdefault(str(out.get("model") or d.get("chosen_returned")), [])
        arm.append(out)
    for role, info in roles.items():
        arms = {}
        for model, rows in info.pop("_arms").items():
            lat = sorted(float(r.get("latency_ms") or 0.0) for r in rows)
            arms[model] = {"outcomes": len(rows), "ok_rate": round(sum(1 for r in rows if r.get("ok")) / len(rows), 3),
                           "p50_ms": round(lat[len(lat) // 2], 1)}
        info["arms"] = arms
        enough = sorted(m for m, a in arms.items() if a["outcomes"] >= min_arm_n)
        info["learnable"] = len(enough) >= 2
        if info["learnable"]:
            info["why"] = f"{len(enough)} arms have at least {min_arm_n} outcomes: {', '.join(enough)}"
        elif len(arms) <= 1:
            info["why"] = ("only one arm has ever been observed, so no alternative has an outcome to learn from or "
                           f"to test against; set {EXPLORE_ENV} and {EXPLORE_ROLES_ENV} to collect some")
        else:
            info["why"] = f"fewer than two arms have {min_arm_n} outcomes yet"
    return {"roles": roles, "legacy_rows": legacy, "min_arm_n": min_arm_n, "decisions": len(decisions), "outcomes": len(outcomes)}


# ------------------------------------------------------------------ discovery / suggestions

def classify_model(name: str) -> tuple[str, ...]:
    """Roles a tag plausibly serves, from its name. Specialists first, then generic generation."""
    low = (name or "").lower()
    for needles, roles in _ROLE_HINTS:
        if any(n in low for n in needles):
            return roles
    return ("gen",)


def suggest(inv: dict, cap_gb: float = 10.0) -> list[PoolEntry]:
    """A draft pool from what is installed: specialists by name, ranked big-to-small within a role.

    A starting point to edit, not a policy: capability vs latency is the operator's call.
    Models above ``cap_gb`` are left out (a load that spans both cards stalls Ollama).
    """
    by_role: dict[str, list[tuple[str, float]]] = {}
    for tag, size in inv.items():
        gb = (size / 1e9) if isinstance(size, (int, float)) else 0.0
        if gb > cap_gb:
            continue
        for role in classify_model(tag):
            by_role.setdefault(role, []).append((tag, gb))
    ranks: dict[str, dict[str, int]] = {}
    for role, items in by_role.items():
        for i, (tag, _) in enumerate(sorted(items, key=lambda t: (-t[1], t[0])), start=1):
            ranks.setdefault(tag, {})[role] = i
    out = []
    for tag in sorted(ranks):
        roles = tuple(sorted(ranks[tag]))
        size = inv.get(tag)
        out.append(PoolEntry(
            name=tag, roles=roles, rank=float(min(ranks[tag].values())),
            vram_gb=round(size / 1e9, 1) if isinstance(size, (int, float)) else None,
            pinned="embed" in roles,
        ))
    return out


def render_toml(pool_entries: Iterable[PoolEntry]) -> str:
    lines = ["[models]", f"resident_bonus = {DEFAULT_RESIDENT_BONUS}", "# max_vram_gb = 12", ""]
    for e in pool_entries:
        lines += ["[[models.pool]]", f'name = "{e.name}"',
                  "roles = [" + ", ".join(f'"{r}"' for r in e.roles) + "]", f"rank = {e.rank:g}"]
        if e.vram_gb is not None:
            lines.append(f"vram_gb = {e.vram_gb:g}")
        if e.pinned:
            lines.append("pinned = true")
        if not e.evictable:
            lines.append("evictable = false")
        if e.role_rank:
            lines.append("role_rank = { " + ", ".join(f"{r} = {v:g}" for r, v in e.role_rank) + " }")
        lines.append("")
    return "\n".join(lines)


def _main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "show"
    if cmd == "init":
        print(render_toml(suggest(inventory())))
        return 0
    if cmd == "outcomes":
        summary_ = outcomes_summary()
        if not summary_:
            print("no outcomes logged; set LOCI_MODEL_POOL_SHADOW=1 and use Loci for a while")
        for model, s_ in summary_.items():
            print(f"{model:55s} n={s_['calls']:5d} ok={s_['ok_rate']:.0%} "
                  f"p50={s_['p50_ms']:.0f}ms p95={s_['p95_ms']:.0f}ms deadline={s_['deadline_exceeded']}")
        return 0
    if cmd == "report":
        rep_ = pool_report()
        print(f"decisions={rep_['decisions']} outcomes={rep_['outcomes']} rows without a decision_id={rep_['legacy_rows']}")
        for role, info in sorted(rep_["roles"].items()):
            print(f"{role:10s} decisions={info['decisions']} linked={info['linked']} explored={info['explored']} "
                  f"status={info['status']} learnable={info['learnable']}")
            print(f"           {info['why']}")
            for model, a in sorted(info["arms"].items()):
                print(f"           arm {model}: n={a['outcomes']} ok={a['ok_rate']:.0%} p50={a['p50_ms']:.0f}ms")
        return 0
    if cmd == "pick" and len(argv) > 1:
        print(pick(argv[1]) or "(none)")
        return 0
    s = summary()
    if not s.get("configured"):
        print("no [[models.pool]] configured; run `python model_pool.py init` for a draft")
        return 0
    for role, info in s["roles"].items():
        flag = ("  DEGRADED" if info["degraded"] else "") + ("  OVER-CAP" if info.get("over_cap") else "")
        print(f"{role:10s} -> {info['chosen']} (rank {info['rank']}){flag}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
