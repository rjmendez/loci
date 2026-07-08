"""Portable backend resolution for the Loci offload tiers.

Lets a Loci install work UNCHANGED on any machine without hardcoding infra — a laptop uses
its own local GPU, a headless box falls back to shared infra over tailscale — and keeps every
machine-specific endpoint/key OUT of this (public) code. Each backend resolves via a chain:

  1. explicit env var (OLLAMA_BASE_URL, VLLM_BASE_URL, EMBED_MODEL, ...) — power-user override
  2. a LOCAL probe (e.g. localhost:11434 for Ollama) — a laptop auto-uses its own hardware
  3. a gitignored config file — remote infra (e.g. the oxalis tailscale endpoint + Qdrant key)
  4. a safe default / empty — the tiers already fail-open when a backend is empty

Config file: `$LOCI_CONFIG`, else `~/.loci/backends.toml`. TOML (stdlib `tomllib`), e.g.:

    [ollama]
    url = "http://100.73.200.19:11434"   # tailscale endpoint when there's no local GPU
    [embed]
    model = "nomic-embed-text"
    [vllm]
    url = "http://100.73.200.19:8000"
    model = "Qwen/Qwen2.5-3B-Instruct"
    [rerank]
    model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    [qdrant]
    url = "http://172.21.171.198:30633"
    api_key = "..."
    [memory]
    dir = "/path/to/curated/MEMORY.md/dir"

See backends.toml.example. Resolutions are memoized (the probe runs once per process).
"""
from __future__ import annotations

import functools
import os
import socket
from pathlib import Path
from urllib.parse import urlparse

_CONFIG_PATH = os.environ.get("LOCI_CONFIG") or str(Path.home() / ".loci" / "backends.toml")

# Local endpoints to probe (only reached when the env var is unset). Kept here, not in each
# module, and generic (localhost) — nothing machine-specific.
_LOCAL_OLLAMA = os.environ.get("LOCI_LOCAL_OLLAMA", "http://localhost:11434")
_LOCAL_VLLM = os.environ.get("LOCI_LOCAL_VLLM", "http://localhost:8000")


@functools.lru_cache(maxsize=1)
def _config() -> dict:
    """Parse the gitignored TOML config. Fail-open: missing/broken file -> {}."""
    try:
        import tomllib
        p = Path(_CONFIG_PATH)
        if p.exists():
            return tomllib.loads(p.read_text())
    except Exception:
        pass
    return {}


def _cfg(section: str, key: str, default=None):
    return (_config().get(section) or {}).get(key, default)


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


@functools.lru_cache(maxsize=1)
def ollama_url() -> str:
    """Ollama base URL: env -> local probe -> config -> ''. Memoized (probe runs once)."""
    env = os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_URL")
    if env:
        return env
    if _alive(_LOCAL_OLLAMA):
        return _LOCAL_OLLAMA
    return _cfg("ollama", "url", "") or ""


@functools.lru_cache(maxsize=1)
def vllm_url() -> str:
    """vLLM/OpenAI base URL: env -> local probe -> config -> '' (batched_gen falls back to Ollama)."""
    env = os.environ.get("VLLM_BASE_URL")
    if env:
        return env
    if _alive(_LOCAL_VLLM):
        return _LOCAL_VLLM
    return _cfg("vllm", "url", "") or ""


def embed_model() -> str:
    return os.environ.get("EMBED_MODEL") or _cfg("embed", "model", "nomic-embed-text")


def vllm_model() -> str:
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


def memory_dir() -> str:
    """Curated MEMORY.md dir for the grounding memory lane: env -> config -> HERMES_MEMORY_DIR -> ''.
    No machine/user-specific default (the old ~/.claude/.../-home-<user>/memory default is gone)."""
    return (os.environ.get("LOCI_MEMORY_MD_DIR") or _cfg("memory", "dir", "")
            or os.environ.get("HERMES_MEMORY_DIR", "") or "")


def _reset_cache() -> None:
    """Test hook: clear memoized resolutions (env/config may have changed)."""
    for fn in (_config, ollama_url, vllm_url):
        fn.cache_clear()
