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

# Legacy name -> current name. Suffixes are identical; both are spelled out so
# the mapping is greppable from either direction.
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
