"""Portable backend resolution for the Loci offload tiers.

Lets a Loci install work UNCHANGED on any machine without hardcoding infra — a laptop uses
its own local GPU, a headless box falls back to shared infra over tailscale — and keeps every
machine-specific endpoint/key OUT of this (public) code. Each backend resolves via a chain:

  1. explicit env var (OLLAMA_BASE_URL, EMBED_MODEL, ...) — power-user override
  2. a LOCAL probe (e.g. localhost:11434 for Ollama) — a laptop auto-uses its own hardware
  3. a gitignored config file — remote infra (e.g. a GPU host over the network + Qdrant key)
  4. a safe default / empty — the tiers already fail-open when a backend is empty

Config file: `$LOCI_CONFIG`, else `~/.loci/backends.toml`. TOML (stdlib `tomllib`). Hosts
below are PLACEHOLDERS — put your own local-GPU / remote-infra endpoints here, never in code:

    [ollama]
    url = "http://gpu-host:11434"        # a GPU host reachable over your network (omit -> local :11434)
    [embed]
    model = "nomic-embed-text"
    [rerank]
    model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    [qdrant]
    url = "http://qdrant-host:6333"
    api_key = "..."
    [memory]
    dir = "/path/to/curated/MEMORY.md/dir"

See backends.toml.example. Resolutions are memoized (the probe runs once per process).
"""
from __future__ import annotations

import logging
import functools
import os
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger("loci-mcp.backends")

_CONFIG_PATH = os.environ.get("LOCI_CONFIG") or str(Path.home() / ".loci" / "backends.toml")

# Local endpoints to probe (only reached when the env var is unset). Kept here, not in each
# module, and generic (localhost) — nothing machine-specific.
_LOCAL_OLLAMA = os.environ.get("LOCI_LOCAL_OLLAMA", "http://localhost:11434")
_OLLAMA_LIST_TIMEOUT = float(os.environ.get("LOCI_OLLAMA_LIST_TIMEOUT", "2.0"))


# The parsed config, re-read when the file changes. It used to be parsed once per process (lru_cache), so an edit
# reached only the processes started after it: the MCP server, the CLI tools and the scheduled tasks each held their
# own copy. Pinning a model to a GPU (``model_pool`` ``gpu = n``) makes every Ollama request carry ``main_gpu``, and
# Ollama reloads a loaded model whenever two callers disagree on it, so processes holding different copies of the
# config reloaded the same model back and forth (2026-10-06: ~15 s per call, in pairs).
_CONFIG_CACHE: dict = {"path": None, "stamp": None, "value": {}, "checked": 0.0}
_CONFIG_RECHECK_S = 2.0      # stat the file at most this often


def _config() -> dict:
    """Parse the gitignored TOML config; re-read it when its mtime or size changes (checked every 2 s).

    Fail-open: a missing or broken file is ``{}``, and a broken file is not re-parsed until it changes again."""
    c = _CONFIG_CACHE
    now = time.monotonic()
    path = _CONFIG_PATH
    if c["path"] == path and now - c["checked"] < _CONFIG_RECHECK_S:
        return c["value"]
    try:
        st = os.stat(path)
        stamp = (st.st_mtime_ns, st.st_size)
    except OSError:
        stamp = None
    c["checked"] = now
    if c["path"] == path and c["stamp"] == stamp:
        return c["value"]
    value: dict = {}
    if stamp is not None:
        try:
            import tomllib
            value = tomllib.loads(Path(path).read_text())
        except Exception as exc:
            logger.debug("_config: fail-open swallow: %r", exc)
            value = {}
    c.update(path=path, stamp=stamp, value=value)
    return value


def _config_cache_clear() -> None:
    _CONFIG_CACHE.update(path=None, stamp=None, value={})      # no path: the next call is a miss whatever "checked" says


_config.cache_clear = _config_cache_clear  # type: ignore[attr-defined]  # the hook _reset_cache() and tests use


def _cfg(section: str, key: str, default=None):
    return (_config().get(section) or {}).get(key, default)


def _cfg_nested(section: str, subsection: str, key: str, default=None):
    parent = _config().get(section) or {}
    child = parent.get(subsection) if isinstance(parent, dict) else None
    return child.get(key, default) if isinstance(child, dict) else default


def _alive(url: str, timeout: float = 1.0) -> bool:
    """Cheap TCP reachability probe of a URL's host:port. Never raises."""
    if not url:
        return False
    try:
        u = urlparse(url)
        host = u.hostname
        port = u.port or (443 if u.scheme == "https" else 80)
        if not host:
            return False
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _http_probe(url: str, path: str = "", timeout: float = 1.0,
                headers: "dict | None" = None) -> "tuple[bool, object]":
    """Bounded HTTP GET of ``url + path``. Returns ``(ok, parsed_json_or_None)``. Never raises.

    ``_alive`` only proves a socket accepts: a hung Qdrant, or a local relay whose
    upstream is dead, passes it. This proves the service answers a request.
    """
    if not url:
        return False, None
    try:
        import json as _json
        import urllib.request
        req = urllib.request.Request(url.rstrip("/") + path, headers=headers or {}, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — operator-configured URL
            ok = 200 <= resp.status < 400
            body = resp.read(1_000_000)
        try:
            return ok, _json.loads(body) if body else None
        except Exception:
            return ok, None
    except Exception as exc:
        logger.debug("_http_probe %s%s failed: %r", url, path, exc)
        return False, None


def _http_status(url: str, path: str = "", timeout: float = 1.0,
                 headers: "dict | None" = None) -> "int | None":
    """HTTP status of a bounded GET of ``url + path`` (4xx/5xx included), or None
    when nothing answered (refused, timeout, DNS). Never raises."""
    if not url:
        return None
    try:
        import urllib.error
        import urllib.request
        req = urllib.request.Request(url.rstrip("/") + path, headers=headers or {}, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — operator-configured URL
                return int(resp.status)
        except urllib.error.HTTPError as exc:
            return int(exc.code)
    except Exception as exc:
        logger.debug("_http_status %s%s failed: %r", url, path, exc)
        return None


_SIZE_UNITS = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12}


@functools.lru_cache(maxsize=1)
def _ollama_list() -> dict[str, "int | None"]:
    """Best-effort local Ollama inventory: tag -> size in bytes (None if unparsed). Never raises."""
    try:
        p = subprocess.run(["ollama", "list"], capture_output=True, text=True,
                           timeout=_OLLAMA_LIST_TIMEOUT, check=False)
    except Exception:
        return {}
    if p.returncode != 0:
        return {}
    models: dict[str, "int | None"] = {}
    for line in (p.stdout or "").splitlines():
        s = line.strip()
        if not s or s.startswith("NAME "):
            continue
        cols = s.split()
        size = None
        # NAME  ID  SIZE UNIT  MODIFIED...  e.g. "qwen2.5:3b  357c53fb659c  1.9 GB  3 weeks ago"
        if len(cols) >= 4 and cols[3].upper() in _SIZE_UNITS:
            try:
                size = int(round(float(cols[2]) * _SIZE_UNITS[cols[3].upper()]))
            except ValueError:
                size = None
        models[cols[0]] = size
    return models


@functools.lru_cache(maxsize=1)
def _ollama_local_tags() -> set[str]:
    """Best-effort local Ollama tag inventory. Never raises."""
    return set(_ollama_list())


# Built-in last resort when neither env nor config names a model: small enough to load
# on one consumer GPU. A model larger than one card is split across GPUs by Ollama, and
# on 11-12 GB cards those loads time out and block Ollama's scheduler for minutes at a
# time (2026-09-26: 75 of 77 gemma4:26b loads failed, starving the embedding model).
ONE_GPU_FALLBACK_MODEL = "qwen2.5:3b"
ONE_GPU_REDTEAM_FALLBACK_MODEL = "heretic-llama31-8b-instruct:latest"


def _auto_pick_max_bytes() -> int:
    """Largest installed model the resolvers may pick on their own (env LOCI_OLLAMA_AUTO_MAX_GB)."""
    try:
        return int(float(os.environ.get("LOCI_OLLAMA_AUTO_MAX_GB", "10")) * 10**9)
    except ValueError:
        return 10 * 10**9


def _fits_one_gpu(tag: str) -> bool:
    """True unless `ollama list` reports this tag above the auto-pick size cap.

    Only guards automatic picks from the local inventory. A tag named by env or
    config is the operator's choice and is never filtered.
    """
    size = _ollama_list().get(tag)
    return size is None or size <= _auto_pick_max_bytes()


def _first_non_embedding_local() -> str:
    tags = sorted(_ollama_local_tags())
    for t in tags:
        if "embed" not in t.lower() and _fits_one_gpu(t):
            return t
    return ""


def _first_matching_local(patterns: tuple[str, ...]) -> str:
    tags = sorted(_ollama_local_tags())
    for t in tags:
        lt = t.lower()
        if any(p in lt for p in patterns) and _fits_one_gpu(t):
            return t
    return ""


@functools.lru_cache(maxsize=8)
def ollama_url(probe_timeout: float = 1.0) -> str:
    """Ollama base URL: env -> local probe -> config -> ''. Memoized (probe runs once).

    `probe_timeout` bounds the local reachability probe. Callers that need a fast,
    bounded resolution (e.g. loci_health on a first call with backends down) pass a
    short value; the result is memoized per distinct timeout so the probe still runs
    at most once per process per timeout.
    """
    env = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_URL")
    if env:
        return env
    if _alive(_LOCAL_OLLAMA, timeout=probe_timeout):
        return _LOCAL_OLLAMA
    return _cfg("ollama", "url", "") or ""


def ollama_gen_url(probe_timeout: float = 1.0) -> str:
    """Ollama base URL for GENERATION, which is not always the embedding one.

    Reachability is not capability. An Ollama that answers /api/tags may carry no
    generation model at all: on this host the in-cluster instance serves only
    nomic-embed-text, so it satisfies _alive() and then fails every generate()
    call. ollama_url() resolved to it and llm_local inherited the choice.

    They also want opposite things. Measured here: embeddings 93 ms in-cluster vs
    5,595 ms over the tailnet, while the generation model only exists over the
    tailnet (35.7 s cold, 247 ms warm with keep_alive). One URL cannot serve both
    well, so generation gets its own.

    Resolution: LOCI_OLLAMA_GEN_URL / OLLAMA_GEN_URL -> [ollama].gen_url ->
    ollama_url() as a last resort, so a host with a single capable Ollama needs no
    extra config.
    """
    env = os.environ.get("LOCI_OLLAMA_GEN_URL") or os.environ.get("OLLAMA_GEN_URL")
    if env:
        return env
    configured = _cfg("ollama", "gen_url", "") or ""
    if configured:
        return configured
    return ollama_url(probe_timeout)


_PINNED_MODEL_KEYS = ("gen_model", "verify_model", "classify_model", "compress_model", "guardian_model",
                      "redteam_model", "swarm_escalate_model", "swarm_synthesize_model")
_warned_pinned = False


def _warn_pinned_models() -> None:
    """Models are chosen by the pool ([[models.pool]]), never named in config. A leftover ``[ollama].*_model`` key
    is ignored, loudly and once, so nobody believes it still pins anything."""
    global _warned_pinned
    if _warned_pinned:
        return
    _warned_pinned = True
    try:
        pinned = [k for k in _PINNED_MODEL_KEYS if _cfg("ollama", k, "")]
    except Exception:
        return
    if pinned:
        logger.warning("backends: [ollama].%s ignored: models come from the pool ([[models.pool]]), never from "
                       "named config; remove the key(s)", ", ".join(pinned))


def _pool_pick(role: str) -> str:
    """Ranked pick from the model pool ([[models.pool]]), or "" when no pool is configured,
    the role is not pooled, or nothing pooled is installed. Never raises."""
    try:
        import model_pool
        return model_pool.pick(role)
    except Exception as exc:
        logger.debug("_pool_pick(%r): fail-open swallow: %r", role, exc)
        return ""


def ollama_gen_model() -> str:
    """Generation model tag. Env LOCI_OLLAMA_GEN_MODEL (a per-process override, never config) -> the model pool
    ('gen') -> an installed local non-embedding model under the one-GPU cap -> a last-resort default."""
    _warn_pinned_models()
    env = os.environ.get("LOCI_OLLAMA_GEN_MODEL")
    if env:
        return env
    return _pool_pick("gen") or _first_non_embedding_local() or ONE_GPU_FALLBACK_MODEL


def ollama_verify_model() -> str:
    """Model for verify_finding's adversarial reasoning. Env -> the model pool ('verify') -> ollama_gen_model()."""
    env = os.environ.get("LOCI_OLLAMA_VERIFY_MODEL")
    return env or _pool_pick("verify") or ollama_gen_model()


def swarm_escalate_model() -> str:
    """swarm_escalate's escalation tier. Env LOCI_SWARM_ESCALATE_MODEL -> the verify tier."""
    return os.environ.get("LOCI_SWARM_ESCALATE_MODEL") or ollama_verify_model()


def swarm_synthesize_model() -> str:
    """swarm_escalate's synthesis tier. Env LOCI_SWARM_SYNTHESIZE_MODEL -> the verify tier."""
    return os.environ.get("LOCI_SWARM_SYNTHESIZE_MODEL") or ollama_verify_model()


def ollama_guardian_model() -> str:
    """Model for guardian.check_injection_risk's safety classification.

    Deliberately does NOT fall back to ollama_gen_model(): a safety classifier is a purpose-built model tuned on
    a risk-definition prompt, and an arbitrary general model would give meaningless Yes/No output. Env
    LOCI_OLLAMA_GUARDIAN_MODEL -> the model pool ('guardian') -> an installed tag with "guard" in its name ->
    a last-resort default.
    """
    _warn_pinned_models()
    env = os.environ.get("LOCI_OLLAMA_GUARDIAN_MODEL")
    if env:
        return env
    return _pool_pick("guardian") or _first_matching_local(("guard",)) or "granite3-guardian:2b"


def ollama_redteam_model() -> str:
    """Model for explicitly adversarial red-team critique, biased toward an uncensored or abliterated model so the
    attacks are phrased the way an adversary would. Env LOCI_OLLAMA_REDTEAM_MODEL -> the model pool ('redteam') ->
    an installed heretic/abliterated tag -> a last-resort default."""
    _warn_pinned_models()
    env = os.environ.get("LOCI_OLLAMA_REDTEAM_MODEL")
    if env:
        return env
    return _pool_pick("redteam") or _first_matching_local(("heretic", "abliterated")) or ONE_GPU_REDTEAM_FALLBACK_MODEL


def _role_env(prefix: str, role: str) -> str:
    return f"{prefix}_{role.strip().upper().replace('-', '_')}"


def embed_model() -> str:
    return os.environ.get("EMBED_MODEL") or _cfg("embed", "model", "nomic-embed-text")


def rerank_model() -> str:
    # Default flipped MiniLM -> bge on judge-eval evidence (+14% nDCG@10, no regression;
    # see scripts/judge_eval.py). bge is heavier (~600MB, slower/query); constrained hosts
    # pin the lighter model back via RERANK_MODEL / [rerank].model in the gitignored config.
    return os.environ.get("RERANK_MODEL") or _cfg(
        "rerank", "model", "BAAI/bge-reranker-v2-m3")


def qdrant() -> tuple[str, str]:
    return (os.environ.get("QDRANT_URL") or _cfg("qdrant", "url", "") or "",
            os.environ.get("QDRANT_API_KEY") or _cfg("qdrant", "api_key", "") or "")


def openrouter() -> tuple[str, str]:
    """(base_url, api_key) for the OpenRouter tier: env -> config -> ('', '').

    The key belongs in ~/.loci/backends.toml, which is outside any git tree —
    never in the repo. See docs/companion-service.md.
    """
    return (os.environ.get("OPENROUTER_BASE_URL")
            or _cfg("openrouter", "url", "") or "https://openrouter.ai/api/v1",
            os.environ.get("OPENROUTER_API_KEY") or _cfg("openrouter", "key", "") or "")


def openrouter_model(role: str | None = None) -> str:
    """OpenRouter model resolver for cloud-tier routing.

    Resolution order: role env -> role config -> shared env -> shared config -> default.
    """
    if role:
        env = os.environ.get(_role_env("OPENROUTER_MODEL", role))
        if env:
            return env
        configured = _cfg_nested("openrouter", role, "model", "") or ""
        if configured:
            return configured
    return (os.environ.get("OPENROUTER_MODEL")
            or _cfg("openrouter", "model", "")
            or "qwen/qwen3.8-27b:free")


def abliteration() -> tuple[str, str]:
    """(base_url, api_key) for Abliteration cloud escalation."""
    return (os.environ.get("ABLITERATION_BASE_URL")
            or _cfg("abliteration", "url", "") or "https://api.abliteration.ai/v1",
            os.environ.get("ABLITERATION_API_KEY") or _cfg("abliteration", "key", "") or "")


def abliteration_model(role: str | None = None) -> str:
    """Abliteration model resolver for cloud-tier routing."""
    if role:
        env = os.environ.get(_role_env("ABLITERATION_MODEL", role))
        if env:
            return env
        configured = _cfg_nested("abliteration", role, "model", "") or ""
        if configured:
            return configured
    return (os.environ.get("ABLITERATION_MODEL")
            or _cfg("abliteration", "model", "")
            or "abliterated-model")


def cloud_tier_enabled() -> bool:
    """Enable cloud third-tier fallback orchestration."""
    v = (os.environ.get("LOCI_CLOUD_TIER_ENABLED")
         or _cfg("cloud", "enabled", False))
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _int_nonneg(v, default: int = 0) -> int:
    try:
        n = int(v)
    except Exception:
        return default
    return max(0, n)


def _boolish(v, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _csvish_set(v) -> set[str]:
    if v is None:
        return set()
    if isinstance(v, (list, tuple, set)):
        raw = [str(x) for x in v]
    else:
        raw = str(v).split(",")
    out = {s.strip().lower() for s in raw if str(s).strip()}
    return out


def _role_session_map(v) -> dict[str, str]:
    if v is None:
        return {}
    if isinstance(v, dict):
        pairs = v.items()
    else:
        text = str(v).strip()
        if not text:
            return {}
        try:
            import json
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                pairs = parsed.items()
            else:
                pairs = []
        except Exception:
            pairs = []
        if not pairs:
            pairs = []
            for token in text.split(","):
                if "=" not in token:
                    continue
                role, session = token.split("=", 1)
                pairs.append((role, session))
    out: dict[str, str] = {}
    for role, session in pairs:
        rk = str(role).strip().lower()
        sv = str(session).strip()
        if rk and sv:
            out[rk] = sv
    return out


def cloud_max_tokens_per_call() -> int:
    """Per-call cloud tier token ceiling (0 disables guardrail)."""
    return _int_nonneg(
        os.environ.get("LOCI_CLOUD_TIER_MAX_TOKENS_PER_CALL",
                       _cfg("cloud", "max_tokens_per_call", 0)),
        default=0,
    )


def cloud_daily_call_budget() -> int:
    """Max cloud fallback calls per UTC day (0 disables guardrail)."""
    return _int_nonneg(
        os.environ.get("LOCI_CLOUD_TIER_DAILY_CALL_BUDGET",
                       _cfg("cloud", "daily_call_budget", 0)),
        default=0,
    )


def cloud_daily_token_budget() -> int:
    """Max requested cloud fallback tokens per UTC day (0 disables guardrail)."""
    return _int_nonneg(
        os.environ.get("LOCI_CLOUD_TIER_DAILY_TOKEN_BUDGET",
                       _cfg("cloud", "daily_token_budget", 0)),
        default=0,
    )


def cloud_deny_providers() -> set[str]:
    """Denied cloud providers (openrouter, abliteration)."""
    return _csvish_set(
        os.environ.get("LOCI_CLOUD_TIER_DENY_PROVIDERS",
                       _cfg("cloud", "deny_providers", ""))
    )


def cloud_deny_roles() -> set[str]:
    """Denied routing roles (triage/coding/reasoning/synthesis/redteam)."""
    return _csvish_set(
        os.environ.get("LOCI_CLOUD_TIER_DENY_ROLES",
                       _cfg("cloud", "deny_roles", ""))
    )


def cloud_allowed_roles() -> set[str]:
    """Allow-list for routing roles (empty means all roles allowed)."""
    return _csvish_set(
        os.environ.get("LOCI_CLOUD_TIER_ALLOWED_ROLES",
                       _cfg("cloud", "allowed_roles", ""))
    )


def cloud_budget_state_path() -> str:
    """State file used for daily cloud budget accounting."""
    return (os.environ.get("LOCI_CLOUD_TIER_BUDGET_STATE_PATH")
            or _cfg("cloud", "budget_state_path", "")
            or str(Path.home() / ".loci" / "cloud_tier_budget.json"))


def cloud_supervisor_model() -> str:
    """Local supervisor model used to dispatch unspecified model requests."""
    return (os.environ.get("LOCI_CLOUD_TIER_SUPERVISOR_MODEL")
            or _cfg("cloud", "supervisor_model", "")
            or ollama_verify_model())


def tmux_offload_enabled() -> bool:
    """Enable tmux-lane policy for offload execution."""
    return _boolish(
        os.environ.get("LOCI_TMUX_OFFLOAD_ENABLED",
                       _cfg("tmux_offload", "enabled", False)),
        default=False,
    )


def tmux_offload_role_sessions() -> dict[str, str]:
    """Role -> tmux session mapping for offload lanes."""
    return _role_session_map(
        os.environ.get("LOCI_TMUX_OFFLOAD_ROLE_SESSIONS",
                       _cfg("tmux_offload", "role_sessions", {}))
    )


def tmux_offload_expensive_roles() -> set[str]:
    """Roles considered expensive and lane-priority worthy."""
    return _csvish_set(
        os.environ.get("LOCI_TMUX_OFFLOAD_EXPENSIVE_ROLES",
                       _cfg("tmux_offload", "expensive_roles", ""))
    )


def tmux_offload_require_mapped_session() -> bool:
    """Fail closed when a mapped tmux session is unavailable."""
    return _boolish(
        os.environ.get("LOCI_TMUX_OFFLOAD_REQUIRE_MAPPED_SESSION",
                       _cfg("tmux_offload", "require_mapped_session", False)),
        default=False,
    )


def memory_dir() -> str:
    """Curated MEMORY.md dir for the grounding memory lane.

    Lookup order: LOCI_MEMORY_MD_DIR -> LOCI_MEMORY_DIR -> HERMES_MEMORY_DIR
    -> gitignored config [memory].dir -> ''. No machine/user-specific default
    (the old ~/.claude/.../-home-<user>/memory default is gone).
    """
    return (os.environ.get("LOCI_MEMORY_MD_DIR") or os.environ.get("LOCI_MEMORY_DIR")
            or os.environ.get("HERMES_MEMORY_DIR") or _cfg("memory", "dir", "") or "")


def load_env(repo: "Path | None" = None) -> dict:
    """Put resolved backends into os.environ so unattended entry points can reach them.

    A cron or CI run inherits none of the MCP launcher's environment, and on this
    host the only record of the Ollama and Qdrant endpoints lives inside that
    launcher's process. Without this, a scheduled job resolves the localhost
    default, finds nothing listening, and reports every backend step "skipped" —
    which reads as a schedule decision rather than a missing endpoint.

    Precedence is unchanged: anything already in the environment wins, then the
    repo .env files, then ~/.loci/backends.toml. Returns only what it set, so a
    caller can log what it had to fill in. Also propagates to child processes,
    which is what mlops/loop.py depends on.
    """
    root = Path(repo) if repo else Path(__file__).resolve().parent.parent
    preexisting = dict(os.environ)
    try:
        from dotenv import load_dotenv
        load_dotenv(root / ".env")
        # override=True lets mcp/.env win over the repo .env, which is the point
        # — but it also overwrites the caller's own environment, contradicting
        # the precedence this function documents. Anything that was already set
        # goes back afterwards, so an explicit export still wins.
        load_dotenv(root / "mcp" / ".env", override=True)
        for key, was in preexisting.items():
            if os.environ.get(key) != was:
                os.environ[key] = was
    except Exception as exc:
        logger.warning("load_env: .env unreadable (%r) — env-file settings, including "
                       "LOCI_QDRANT_RETENTION_DAYS, will NOT be applied", exc)

    resolved: dict = {}
    if not os.environ.get("QDRANT_URL"):
        try:
            url, key = qdrant()
            if url:
                os.environ["QDRANT_URL"] = resolved["QDRANT_URL"] = url
                if key and not os.environ.get("QDRANT_API_KEY"):
                    # Reported as well as set: mlops/loop.py restores its own
                    # environment from this dict, and a key it never hears about
                    # is a key it leaves behind.
                    os.environ["QDRANT_API_KEY"] = resolved["QDRANT_API_KEY"] = key
        except Exception as exc:
            logger.warning("load_env: could not resolve Qdrant: %r", exc)

    if not os.environ.get("OLLAMA_BASE_URL"):
        try:
            url = ollama_url()
            if url:
                os.environ["OLLAMA_BASE_URL"] = resolved["OLLAMA_BASE_URL"] = url
        except Exception as exc:
            logger.warning("load_env: could not resolve Ollama: %r", exc)

    if not os.environ.get("EMBED_MODEL"):
        try:
            model = embed_model()
            if model:
                os.environ["EMBED_MODEL"] = resolved["EMBED_MODEL"] = model
        except Exception as exc:
            logger.warning("load_env: could not resolve the embed model: %r", exc)

    # backends.toml is stdlib tomllib, so the setting that protects the corpus needs no import.
    if not os.environ.get("LOCI_QDRANT_RETENTION_DAYS"):
        try:
            days = _cfg("qdrant", "retention_days", None)
            if days is not None:
                os.environ["LOCI_QDRANT_RETENTION_DAYS"] = resolved["LOCI_QDRANT_RETENTION_DAYS"] = str(days)
        except Exception as exc:
            logger.warning("load_env: could not resolve retention: %r", exc)
    return resolved


def _reset_cache() -> None:
    """Test hook: clear memoized resolutions (env/config may have changed)."""
    for fn in (_config, _ollama_list, _ollama_local_tags, ollama_url):
        clear = getattr(fn, "cache_clear", None)
        if clear:
            clear()
