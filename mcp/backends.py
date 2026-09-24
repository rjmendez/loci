"""Portable backend resolution for the Loci offload tiers.

Lets a Loci install work UNCHANGED on any machine without hardcoding infra — a laptop uses
its own local GPU, a headless box falls back to shared infra over tailscale — and keeps every
machine-specific endpoint/key OUT of this (public) code. Each backend resolves via a chain:

  1. explicit env var (OLLAMA_BASE_URL, VLLM_BASE_URL, EMBED_MODEL, ...) — power-user override
  2. a LOCAL probe (e.g. localhost:11434 for Ollama) — a laptop auto-uses its own hardware
  3. a gitignored config file — remote infra (e.g. a GPU host over the network + Qdrant key)
  4. a safe default / empty — the tiers already fail-open when a backend is empty

Config file: `$LOCI_CONFIG`, else `~/.loci/backends.toml`. TOML (stdlib `tomllib`). Hosts
below are PLACEHOLDERS — put your own local-GPU / remote-infra endpoints here, never in code:

    [ollama]
    url = "http://gpu-host:11434"        # a GPU host reachable over your network (omit -> local :11434)
    [embed]
    model = "nomic-embed-text"
    [vllm]
    url = "http://gpu-host:8000"         # vLLM/OpenAI-compatible server (omit -> Ollama fallback)
    model = "Qwen/Qwen2.5-3B-Instruct"
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
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger("loci-mcp.backends")

_CONFIG_PATH = os.environ.get("LOCI_CONFIG") or str(Path.home() / ".loci" / "backends.toml")

# Local endpoints to probe (only reached when the env var is unset). Kept here, not in each
# module, and generic (localhost) — nothing machine-specific.
_LOCAL_OLLAMA = os.environ.get("LOCI_LOCAL_OLLAMA", "http://localhost:11434")
_LOCAL_VLLM = os.environ.get("LOCI_LOCAL_VLLM", "http://localhost:8000")
_OLLAMA_LIST_TIMEOUT = float(os.environ.get("LOCI_OLLAMA_LIST_TIMEOUT", "2.0"))


@functools.lru_cache(maxsize=1)
def _config() -> dict:
    """Parse the gitignored TOML config. Fail-open: missing/broken file -> {}."""
    try:
        import tomllib
        p = Path(_CONFIG_PATH)
        if p.exists():
            return tomllib.loads(p.read_text())
    except Exception as exc:
        logger.debug("_config: fail-open swallow: %r", exc)
    return {}


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


@functools.lru_cache(maxsize=1)
def _ollama_local_tags() -> set[str]:
    """Best-effort local Ollama tag inventory. Never raises."""
    try:
        p = subprocess.run(["ollama", "list"], capture_output=True, text=True,
                           timeout=_OLLAMA_LIST_TIMEOUT, check=False)
    except Exception:
        return set()
    if p.returncode != 0:
        return set()
    tags: set[str] = set()
    for line in (p.stdout or "").splitlines():
        s = line.strip()
        if not s or s.startswith("NAME "):
            continue
        tag = s.split()[0]
        if tag:
            tags.add(tag)
    return tags


def _first_installed(candidates: tuple[str, ...]) -> str:
    tags = _ollama_local_tags()
    for c in candidates:
        if c in tags:
            return c
    return ""


def _first_non_embedding_local() -> str:
    tags = sorted(_ollama_local_tags())
    for t in tags:
        if "embed" not in t.lower():
            return t
    return ""


def _first_matching_local(patterns: tuple[str, ...]) -> str:
    tags = sorted(_ollama_local_tags())
    for t in tags:
        lt = t.lower()
        if any(p in lt for p in patterns):
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


def ollama_gen_model() -> str:
    """Generation model tag. Env -> [ollama].gen_model -> installed local -> default."""
    env_or_cfg = (os.environ.get("LOCI_OLLAMA_GEN_MODEL")
                  or _cfg("ollama", "gen_model", ""))
    if env_or_cfg:
        return env_or_cfg
    preferred = ("qwen2.5:3b", "qwen3.8:latest", "heretic-llama31-8b-instruct:latest")
    return _first_installed(preferred) or _first_non_embedding_local() or "qwen2.5:3b"


def _task_model(env_var: str, cfg_key: str) -> str:
    """Per-task model override, falling back to the shared ollama_gen_model().

    Adversarial live benchmarking (ab_eval_local_model.py --difficulty hard) showed the
    single shared gen_model is not equally good at every task: classify is high-volume and
    low-stakes, but verify_finding and dense compress_text calls need real reasoning under
    a tighter budget, where a slower/stronger model measurably scores higher. This lets an
    operator opt specific call sites into a different model without changing the default
    that classify_text (and anything else unspecified) keeps using.
    """
    return (os.environ.get(env_var)
            or _cfg("ollama", cfg_key, "")
            or ollama_gen_model())


def ollama_verify_model() -> str:
    """Model for verify_finding's adversarial reasoning. Env -> [ollama].verify_model -> gen_model."""
    return _task_model("LOCI_OLLAMA_VERIFY_MODEL", "verify_model")


def ollama_classify_model() -> str:
    """Model for classify_text's short-label routing. Env -> [ollama].classify_model -> gen_model."""
    return _task_model("LOCI_OLLAMA_CLASSIFY_MODEL", "classify_model")


def ollama_compress_model() -> str:
    """Model for compress_text's summarization. Env -> [ollama].compress_model -> gen_model."""
    return _task_model("LOCI_OLLAMA_COMPRESS_MODEL", "compress_model")


def ollama_guardian_model() -> str:
    """Model for guardian.check_injection_risk's safety classification.

    Deliberately does NOT fall back to ollama_gen_model() like the other per-task
    resolvers: Granite Guardian is a purpose-built safety classifier tuned on a
    specific risk-definition prompt shape, not a general chat/reasoning model.
    Routing this check to an arbitrary general model would produce meaningless
    Yes/No output rather than a degraded-but-sane answer, so the default is the
    verified-good granite3-guardian tag itself.

    Env -> [ollama].guardian_model -> "granite3-guardian:2b".
    """
    env_or_cfg = (os.environ.get("LOCI_OLLAMA_GUARDIAN_MODEL")
                  or _cfg("ollama", "guardian_model", ""))
    if env_or_cfg:
        return env_or_cfg
    preferred = ("llama-guard3:8b", "granite3-guardian:2b", "qwen3.8:latest",
                 "heretic-llama31-8b-instruct:latest", "qwen2.5:3b")
    return _first_installed(preferred) or _first_non_embedding_local() or "granite3-guardian:2b"


def ollama_redteam_model() -> str:
    """Model for explicitly adversarial red-team critique.

    Unlike the normal generation/verify tiers, this one is deliberately biased toward an
    uncensored or abliterated local model. The point of the red-team phase is to phrase
    attacks the way an adversary would, without the softening/refusal behavior aligned
    instruct models often introduce on "attack this" prompts. Resolution stays portable:
    env -> [ollama].redteam_model -> a known-good heretic default.
    """
    env_or_cfg = (os.environ.get("LOCI_OLLAMA_REDTEAM_MODEL")
                  or _cfg("ollama", "redteam_model", ""))
    if env_or_cfg:
        return env_or_cfg
    preferred = ("hf.co/slevinw/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF:Q4_K_M",
                 "qwen3.8:latest")
    return (_first_installed(preferred)
            or _first_matching_local(("heretic", "abliterated"))
            or "hf.co/slevinw/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF:Q4_K_M")


def _vllm_role_env(prefix: str, role: str) -> str:
    return f"{prefix}_{role.strip().upper().replace('-', '_')}"


@functools.lru_cache(maxsize=32)
def vllm_url(role: str | None = None, probe_timeout: float = 1.0) -> str:
    """vLLM/OpenAI base URL: env -> config -> local probe -> '' (batched_gen falls back to Ollama).

    `probe_timeout` bounds the local reachability probe (see ollama_url)."""
    if role:
        env = os.environ.get(_vllm_role_env("VLLM_BASE_URL", role))
        if env:
            return env
        configured = _cfg_nested("vllm", role, "url", "") or ""
        if configured:
            return configured
        # Deliberately no role-specific localhost probe: specialist routing must be explicit,
        # then fall back to the shared/default resolver whose existing localhost probe remains.
        return vllm_url(probe_timeout=probe_timeout)
    env = os.environ.get("VLLM_BASE_URL")
    if env:
        return env
    configured = _cfg("vllm", "url", "") or ""
    if configured:
        return configured
    if _alive(_LOCAL_VLLM, timeout=probe_timeout):
        return _LOCAL_VLLM
    return ""


def embed_model() -> str:
    return os.environ.get("EMBED_MODEL") or _cfg("embed", "model", "nomic-embed-text")


def vllm_model(role: str | None = None) -> str:
    if role:
        env = os.environ.get(_vllm_role_env("VLLM_MODEL", role))
        if env:
            return env
        configured = _cfg_nested("vllm", role, "model", "") or ""
        if configured:
            return configured
        return vllm_model()
    return os.environ.get("VLLM_MODEL") or _cfg("vllm", "model", "Qwen2.5-3B-Instruct")


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
    for fn in (_config, _ollama_local_tags, ollama_url, vllm_url):
        clear = getattr(fn, "cache_clear", None)
        if clear:
            clear()
