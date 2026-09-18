"""Accept the legacy HERMES_* environment variables under their LOCI_* names.

Loci was configured through HERMES_* variables because it started life inside a
Hermes installation. The code now reads LOCI_*; this maps the old spelling onto
the new one at startup so an existing deployment keeps working without being
re-provisioned. Call apply() once, before anything reads config.

HERMES_PROFILE, HERMES_HOME, HERMES_VENV_SITE, HERMES_SUBAGENT and
HERMES_AGENT_ID are deliberately absent: they name the Hermes installation Loci
runs inside, not Loci itself, and renaming them would break that integration.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import urlparse

LOCALHOST_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}
LOCAL_SERVICE_KEYS = {
    "OLLAMA_BASE_URL",
    "QDRANT_URL",
    "MNEMOSYNE_LLM_BASE_URL",
    "MNEMOSYNE_EMBEDDING_API_URL",
    "HERMES_A2A_URL",
    "MRPINK_A2A_URL",
    "SEARXNG_URL",
    "FIRECRAWL_API_URL",
    "EMBED_WORKER_URL",
    "base_url",
}


def _normalize_host(host: str | None) -> str:
    if not host:
        return ""
    return host.strip().lower().strip("[]")


def _is_private_bridge(host: str) -> bool:
    if not host or host in LOCALHOST_HOSTS:
        return False
    parts = host.split(".")
    if len(parts) == 4:
        first, second = parts[0], parts[1]
        if first == "10":
            return True
        if first == "192" and second == "168":
            return True
        if first == "172" and 16 <= int(second) <= 31:
            return True
        if first == "100" and 64 <= int(second) <= 127:
            return True
    return False


def _iter_hermes_urls(environ: dict | None = None, files: list[Path] | None = None):
    env = os.environ if environ is None else environ
    paths = files or []
    if not paths:
        hermes_home = Path(env.get("HERMES_HOME", "~/.hermes")).expanduser()
        paths = [hermes_home / ".env", hermes_home / "config.yaml"]
        profiles = hermes_home / "profiles"
        if profiles.exists():
            paths.extend(sorted(profiles.glob("*/.env")))
            paths.extend(sorted(profiles.glob("*/config.yaml")))

    out = []
    for key in sorted(LOCAL_SERVICE_KEYS):
        value = env.get(key)
        if value and isinstance(value, str):
            out.append((None, key, value.strip()))

    for path in dict.fromkeys(paths):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        pattern = re.compile(r"(?P<key>[A-Za-z0-9_]+)\s*[:=]\s*(?P<value>https?://[^\s\"'\),\]}]+)")
        for match in pattern.finditer(text):
            key = match.group("key")
            if key not in LOCAL_SERVICE_KEYS:
                continue
            value = match.group("value").rstrip(") ,")
            out.append((path, key, value))
    return out


def validate_hermes_endpoint_policy(environ: dict | None = None, files: list[Path] | None = None) -> list[str]:
    """Reject bridge/private IP drift in Hermes local backend endpoints."""
    issues = []
    for path, key, value in _iter_hermes_urls(environ, files):
        try:
            host = _normalize_host(urlparse(value).hostname)
        except Exception:
            continue
        if not host:
            continue
        if host in LOCALHOST_HOSTS:
            continue
        if _is_private_bridge(host):
            issues.append(f"{path or 'env'}:{key} uses bridge/private hostname {host} instead of 127.0.0.1/localhost")
    return issues


RENAMED: dict[str, str] = {
    "HERMES_A2A_BOOTSTRAP_KEY":       "LOCI_A2A_BOOTSTRAP_KEY",
    "HERMES_A2A_HEALTH_URL":          "LOCI_A2A_HEALTH_URL",
    "HERMES_A2A_HOST":                "LOCI_A2A_HOST",
    "HERMES_A2A_PORT":                "LOCI_A2A_PORT",
    "HERMES_A2A_PRIVILEGED_SENDERS":  "LOCI_A2A_PRIVILEGED_SENDERS",
    "HERMES_A2A_SERVICE":             "LOCI_A2A_SERVICE",
    "HERMES_A2A_TOKEN":               "LOCI_A2A_TOKEN",
    "HERMES_A2A_TOTP_SEED":           "LOCI_A2A_TOTP_SEED",
    "HERMES_A2A_URL":                 "LOCI_A2A_URL",
    "HERMES_ACTIVE_INVESTIGATION":    "LOCI_ACTIVE_INVESTIGATION",
    "HERMES_ENV_FILE":                "LOCI_ENV_FILE",
    "HERMES_EVENT_ARCHIVE":           "LOCI_EVENT_ARCHIVE",
    "HERMES_EVENT_LOG":               "LOCI_EVENT_LOG",
    "HERMES_MCP_HOST":                "LOCI_MCP_HOST",
    "HERMES_MCP_PORT":                "LOCI_MCP_PORT",
    "HERMES_MCP_TOKEN":               "LOCI_MCP_TOKEN",
    "HERMES_MCP_TRANSPORT":           "LOCI_MCP_TRANSPORT",
    "HERMES_MEMORY_DIR":              "LOCI_MEMORY_DIR",
    "HERMES_MNEMO_BANK":              "LOCI_MNEMO_BANK",
    "HERMES_PE_HIGH_THRESH":          "LOCI_PE_HIGH_THRESH",
    "HERMES_PE_PROTECTION_MIN_OCC":   "LOCI_PE_PROTECTION_MIN_OCC",
    "HERMES_PORT":                    "LOCI_PORT",
    "HERMES_PORTPROXY_PS1":           "LOCI_PORTPROXY_PS1",
    "HERMES_PY":                      "LOCI_PY",
    "HERMES_REFLECTION_INVESTIGATION": "LOCI_REFLECTION_INVESTIGATION",
    "HERMES_STATE_DB":                "LOCI_STATE_DB",
    "HERMES_SYNC_CACHE":              "LOCI_SYNC_CACHE",
    "HERMES_SYNC_CACHE_TTL_DAYS":     "LOCI_SYNC_CACHE_TTL_DAYS",
    "HERMES_TAILSCALE_IP":            "LOCI_TAILSCALE_IP",
}


def apply(environ: dict | None = None) -> list[str]:
    """Copy any legacy variable onto its current name. Returns what was mapped.

    The current name always wins, so a deployment that sets both is not
    surprised by the old one.
    """
    env = os.environ if environ is None else environ
    mapped = []
    for old, new in RENAMED.items():
        if env.get(old) and not env.get(new):
            env[new] = env[old]
            mapped.append(old)
    errors = validate_hermes_endpoint_policy(env)
    if errors:
        raise RuntimeError("Hermes endpoint policy violation: " + " | ".join(errors))
    return mapped


def memory_dir() -> "os.PathLike | str":
    """Where investigations and the code graph live.

    Fresh installs get ~/.loci/memory-sessions. An existing deployment keeps
    using ~/.hermes/memory-sessions until someone moves it — 182 MB of
    investigations and a 40 MB code graph are not worth relocating implicitly
    on an upgrade. LOCI_MEMORY_DIR overrides both.
    """
    from pathlib import Path

    explicit = os.environ.get("LOCI_MEMORY_DIR") or os.environ.get("HERMES_MEMORY_DIR")
    if explicit:
        return Path(explicit).expanduser()
    new = Path.home() / ".loci" / "memory-sessions"
    if new.is_dir():
        return new
    legacy = Path.home() / ".hermes" / "memory-sessions"
    return legacy if legacy.is_dir() else new
