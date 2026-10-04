"""Model pool: declared candidates per role, ranked, residency- and size-aware.

Each role resolver in ``backends`` used to carry its own hardcoded preference tuple and
take the first installed tag. A configured tag that was not installed silently broke
generation (the tag was returned anyway), and nothing said which model should serve a
role instead. The pool is one declared list the resolvers consult:

    [models]
    resident_bonus = 0.5     # rank credit for a model Ollama already holds in memory
    max_vram_gb = 12         # optional: skip entries bigger than this

    [[models.pool]]
    name = "gemma4-e4b-hermes:64k"
    roles = ["gen", "verify", "compress"]
    rank = 1                 # lower is preferred
    vram_gb = 5.0            # optional; otherwise the size /api/tags reports
    pinned = false           # reserved for the lease/eviction layer: never evict

Resolution for one role: take the entries listing the role, keep those installed at the
generation endpoint (and under ``max_vram_gb``), and order by ``rank - resident_bonus``
(ties break on rank, then name). An operator-configured tag (``[ollama].gen_model`` etc.)
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
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Optional

logger = logging.getLogger("loci-mcp.model_pool")

SHADOW_ENV = "LOCI_MODEL_POOL_SHADOW"
SELECTOR_ENV = "LOCI_MODEL_POOL_SELECTOR"
SHADOW_LOG_NAME = "model_pool_shadow.jsonl"
SHADOW_SCHEMA = 1

DEFAULT_RESIDENT_BONUS = 0.5
_TAGS_TTL_S = 30.0
_PS_TTL_S = 5.0
_HTTP_TIMEOUT_S = 1.5

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
        out.append(PoolEntry(
            name=name, roles=roles, rank=_num(item.get("rank"), 100.0),
            vram_gb=_num(vram, None) if vram is not None else None,
            pinned=bool(item.get("pinned", False)),
        ))
    return out


def configured() -> bool:
    return bool(entries())


def resident_bonus() -> float:
    return max(0.0, _num(_models_cfg().get("resident_bonus"), DEFAULT_RESIDENT_BONUS))


def max_vram_gb() -> Optional[float]:
    value = _models_cfg().get("max_vram_gb")
    return _num(value, None) if value is not None else None


# ------------------------------------------------------------------------------- inventory

_lock = threading.Lock()
_cache: dict[tuple[str, str], tuple[float, object]] = {}


def _get_json(url: str) -> Optional[dict]:
    try:
        with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT_S) as resp:  # noqa: S310 - operator-configured URL
            data = json.loads(resp.read().decode("utf-8", "replace"))
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.debug("model_pool: GET %s failed: %r", url, exc)
        return None


def _cached(kind: str, base_url: str, ttl: float, loader: Callable[[], object]):
    key = (kind, base_url)
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    value = loader()
    with _lock:
        _cache[key] = (time.monotonic(), value)
    return value


def clear_cache() -> None:
    with _lock:
        _cache.clear()


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
        data = _get_json(url + "/api/tags") or {}
        return {str(m.get("name")): m.get("size") for m in (data.get("models") or []) if m.get("name")}

    return dict(_cached("tags", url, _TAGS_TTL_S, load))


def resident_models(base_url: Optional[str] = None) -> set[str]:
    """Tags Ollama currently holds in memory (``/api/ps``). Empty when unreachable."""
    url = (base_url if base_url is not None else _gen_url()).rstrip("/")
    if not url:
        return set()

    def load():
        data = _get_json(url + "/api/ps") or {}
        return {str(m.get("name") or m.get("model")) for m in (data.get("models") or [])}

    return set(_cached("ps", url, _PS_TTL_S, load))


def _installed(name: str, inv: dict) -> bool:
    return name in inv or (":" not in name and f"{name}:latest" in inv)


# ----------------------------------------------------------------------------- resolution

def rank_role(role: str, *, hint: str = "", pool: Optional[Iterable[PoolEntry]] = None,
              inv: Optional[dict] = None, resident: Optional[set] = None,
              bonus: Optional[float] = None, cap_gb: Optional[float] = None) -> Decision:
    """Order the candidates for ``role``. Pure when ``pool``/``inv``/``resident`` are passed."""
    role = (role or "").strip().lower()
    pool_entries = list(entries() if pool is None else pool)
    inventory_ = inventory() if inv is None else inv
    resident_ = resident_models() if resident is None else resident
    bonus_ = resident_bonus() if bonus is None else bonus
    cap = max_vram_gb() if cap_gb is None else cap_gb

    rows: list[tuple[PoolEntry, str]] = [(e, "pool") for e in pool_entries if role in e.roles]
    hint = (hint or "").strip()
    if hint:
        # The operator's own tag outranks the pool: rank 0, keeping any size/pin the pool declares for it.
        rows = [((replace(e, rank=0.0), "configured") if e.name == hint else (e, src)) for e, src in rows]
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
        eligible = installed and fits
        reason = ("" if eligible else "not installed" if not installed
                  else f"{size:.1f} GB over the {cap:g} GB cap")
        cands.append(Candidate(
            name=entry.name, rank=entry.rank,
            effective_rank=entry.rank - (bonus_ if res else 0.0),
            installed=installed, resident=res, fits=fits, eligible=eligible,
            reason=reason, source=source,
        ))
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
        _shadow(decision)
        return decision.chosen
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
    mem = os.environ.get("LOCI_MEMORY_DIR") or str(Path.home() / ".loci" / "memory-sessions")
    return Path(mem).parent / "instrumentation" / SHADOW_LOG_NAME


def _shadow(decision: Decision) -> None:
    """Log the rule decision beside the optional selector's. Never raises, never changes the pick."""
    if not _shadow_enabled():
        return
    try:
        started = time.perf_counter()
        alt = ""
        selector = _load_selector()
        if selector is not None:
            features = [{"name": c.name, "rank": c.rank, "effective_rank": c.effective_rank,
                         "resident": c.resident, "eligible": c.eligible} for c in decision.candidates]
            try:
                result = selector(decision.role, features)
                names = decision.ordered()
                if isinstance(result, str) and result in names:
                    alt = result
                elif isinstance(result, (list, tuple)) and result and result[0] in names:
                    alt = str(result[0])
            except Exception as exc:
                logger.debug("model_pool: shadow selector raised: %r", exc)
        row = {
            "schema": SHADOW_SCHEMA, "ts": time.time(), "role": decision.role,
            "chosen_rule": decision.chosen, "chosen_shadow": alt or None,
            "agree": (alt == decision.chosen) if alt else None,
            "n_candidates": len(decision.candidates),
            "n_eligible": sum(1 for c in decision.candidates if c.eligible),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "candidates": [{"name": c.name, "rank": c.rank, "resident": c.resident,
                            "eligible": c.eligible} for c in decision.candidates],
        }
        from instrumentation_log import append_rows
        append_rows(_log_path(), [row])
    except Exception as exc:
        logger.debug("model_pool: shadow log skipped: %r", exc)


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
        lines.append("")
    return "\n".join(lines)


def _main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "show"
    if cmd == "init":
        print(render_toml(suggest(inventory())))
        return 0
    if cmd == "pick" and len(argv) > 1:
        print(pick(argv[1]) or "(none)")
        return 0
    s = summary()
    if not s.get("configured"):
        print("no [[models.pool]] configured; run `python model_pool.py init` for a draft")
        return 0
    for role, info in s["roles"].items():
        flag = "  DEGRADED" if info["degraded"] else ""
        print(f"{role:10s} -> {info['chosen']} (rank {info['rank']}){flag}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
