#!/usr/bin/env python3
"""
pre_tool_call hook — grounding enforcement, supply chain IOC detection,
prompt injection scanning, dangerous pattern detection, audit log.

v3 changes over v2:
  - Supply chain IOC patterns: Hades/Miasma worm file path and terminal IOCs
  - Prompt injection content scanner (HIGH + SUSPICIOUS tiers) applied to all
    content being written by mutation tools; always-blocks on agent config files
  - Agent config path detection: AGENTS.md, CLAUDE.md, .cursorrules etc. get
    extra scrutiny — any injection pattern triggers block regardless of BLOCK_MODE
  - Supply chain terminal patterns: Bun download/exec, pipe-to-interpreter,
    base64-encoded exec, site-packages manipulation
  - _extract_write_targets / _extract_write_content helpers cover all mutation tools

Wire in:  {"hook_event_name": "pre_tool_call", "tool_name": ..., "tool_input": ..., "extra": {...}}
Wire out: {} (allow) | {"action":"block","message":"..."} (block)
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

BLOCK_MODE: bool = os.environ.get("HOOK_BLOCK_MODE", "0").strip() in ("1", "true", "yes")
MAX_AUDIT_BYTES: int = 5 * 1024 * 1024  # 5 MB — matches Hermes logging.max_size_mb

# ---- Grounding / read-only tools — always allowed, no audit noise ----------------
GROUNDING_TOOLS: frozenset[str] = frozenset({
    # Mnemosyne MCP
    "mcp_mnemosyne_mnemosyne_recall",
    "mcp_mnemosyne_mnemosyne_remember",
    "mcp_mnemosyne_mnemosyne_get_stats",
    "mcp_mnemosyne_mnemosyne_scratchpad_read",
    "mcp_mnemosyne_mnemosyne_scratchpad_write",
    "mcp_mnemosyne_mnemosyne_sleep",
    # open_design MCP removed 2026-06-13
    # Web / session search
    "web_search",
    "web_extract",
    "session_search",
    # Serena code search (read-only)
    "mcp_serena_search_for_pattern",
    "mcp_serena_find_symbol",
    "mcp_serena_read_file",
    "mcp_serena_get_symbols_overview",
    "mcp_serena_find_referencing_symbols",
    "mcp_serena_get_diagnostics_for_file",
    "mcp_serena_list_dir",
    "mcp_serena_get_overview",
    # Claude Code read-only tools — suppress audit noise
    "Read",
    "ToolSearch",
    "WebFetch",
    "WebSearch",
    # Hermes/legacy read tools
    "read_file",
    "search_files",
    "skill_view",
    "skills_list",
    "memory",
    "vision_analyze",
    "browser_snapshot",
    "browser_vision",
    "browser_get_images",
    "browser_console",
    # Process read
    "process",
    "todo",
})

# ---- File-mutation tools — intercepted when BLOCK_MODE is on ------------------
MUTATION_TOOLS: frozenset[str] = frozenset({
    # Claude Code mutation tools
    "Edit",
    "Write",
    "MultiEdit",
    # Hermes/legacy mutation tools
    "write_file",
    "patch",
    "mcp_serena_create_text_file",
    "mcp_serena_replace_content",
    "mcp_serena_replace_symbol_body",
    "mcp_serena_insert_before_symbol",
    "mcp_serena_insert_after_symbol",
    "mcp_serena_rename_symbol",
    "mcp_serena_delete_lines",
    "mcp_serena_create_directory",
})

# ---- Dangerous terminal patterns — always audited, blocked in BLOCK_MODE ------
DANGEROUS_TERMINAL_PATTERNS: list[tuple[str, str]] = [
    (r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f\b", "rm -rf detected"),
    (r"\bDROP\s+(TABLE|DATABASE|SCHEMA)\b", "destructive SQL DDL"),
    (r"\bgit\s+push\s+.*--force(?!-with-lease)\b", "force push without --force-with-lease"),
    (r"\bgit\s+push\s+-f\b", "force push -f"),
    (r"\bkubectl\s+delete\s+(namespace|ns)\b", "kubectl delete namespace"),
    (r"\bdocker\s+(system\s+prune|volume\s+prune|image\s+prune)\b", "docker prune"),
    (r"\bdd\s+if=", "dd disk write"),
    (r"\bmkfs\b", "filesystem format"),
    (r"\bshred\b", "shred/wipe"),
    (r">\s*/dev/sd[a-z]", "raw device write"),
]

# ---- Hades/Miasma supply chain IOC — terminal patterns -----------------------
# Execution vectors documented in the June 2026 campaign.
SUPPLY_CHAIN_TERMINAL_PATTERNS: list[tuple[str, str]] = [
    (r"\bbun\b.{0,60}(install|run|x |exec|--)\b",
     "Bun runtime execution (Hades IOC)"),
    (r"(curl|wget)\s+.{0,120}\|\s*(bash|sh|bun|node|python[23]?)\b",
     "pipe-to-interpreter (supply chain IOC)"),
    (r"\bgithub\.com/.{0,60}/bun/releases\b",
     "Bun binary download from GitHub (Hades IOC)"),
    (r"pip\s+(install|download)\s+.{0,80}(/tmp/|--target\s+/|--prefix\s+/)",
     "pip install to system/temp path"),
    (r"python[23]?\s+-c\s+.{0,120}__import__.{0,60}base64",
     "base64-encoded Python exec"),
    (r"python[23]?\s+-c\s+.{0,120}exec\s*\(",
     "inline Python exec() call"),
    (r"\bnpm\s+install\s+-g\s+.{0,60}\bbun\b",
     "global bun install via npm"),
    (r"(site-packages|dist-packages).*__init__\.py",
     "direct site-packages __init__.py manipulation"),
    (r"\bgh-token-monitor\b",
     "gh-token-monitor (Hades persistence IOC)"),
]

# ---- Hades/Miasma supply chain IOC — file path patterns ----------------------
# Writing to these paths is the Hades import-hook / IDE-open-hook vector.
SUPPLY_CHAIN_PATH_PATTERNS: list[tuple[str, str]] = [
    (r"(^|[/\\])__init__\.py$",
     "__init__.py write (Hades import-hook vector)"),
    (r"(^|[/\\])\.claude[/\\]setup\.mjs$",
     ".claude/setup.mjs (Hades IDE-open vector)"),
    (r"(^|[/\\])\.vscode[/\\]tasks\.json$",
     ".vscode/tasks.json (Hades IDE-open vector)"),
    (r"[/\\]site-packages[/\\][^/\\]+\.pth$",
     ".pth in site-packages (Python import hook)"),
    (r"[/\\]dist-packages[/\\][^/\\]+\.pth$",
     ".pth in dist-packages (Python import hook)"),
    (r"(^|[/\\])binding\.gyp$",
     "binding.gyp (Phantom Gyp / Miasma install-hook)"),
]

# ---- Agent config file paths — extra scrutiny on writes ----------------------
# Writes to these paths containing ANY injection pattern are always blocked.
AGENT_CONFIG_PATTERNS: list[str] = [
    r"(^|[/\\])(AGENTS|CLAUDE|CLAUDE\.local)\.md$",
    r"(^|[/\\])\.cursorrules$",
    r"(^|[/\\])\.cursor[/\\]rules$",
    r"(^|[/\\])(system|agent)_prompt\.txt$",
    r"(^|[/\\])\.hermes[/\\].+\.(md|yaml|yml|json)$",
    r"(^|[/\\])\.openclaw[/\\].+\.(md|yaml|yml|json)$",
]

# ---- Prompt injection — HIGH confidence (explicit hijack attempts) ------------
# Patterns that clearly indicate an attempt to override agent instructions.
# Sources: OWASP LLM-01, Hades campaign analysis, known jailbreak corpora.
INJECTION_HIGH: list[tuple[str, str]] = [
    (r"(?i)\bIGNORE\s+(ALL\s+)?(PREVIOUS|PRIOR)\s+(INSTRUCTIONS?|RULES?|CONSTRAINTS?|DIRECTIVES?)\b",
     "explicit instruction override"),
    (r"(?i)\bDISREGARD\s+(ALL\s+)?(PREVIOUS|PRIOR|ABOVE|EARLIER)\b",
     "disregard-previous hijack"),
    (r"(?i)\bFORGET\s+(EVERYTHING|ALL\s+(PREVIOUS|PRIOR)|YOUR\s+(INSTRUCTIONS?|RULES?|TRAINING|GUIDELINES?))\b",
     "forget-instruction hijack"),
    (r"(?im)^(NEW|UPDATED|REVISED|ACTUAL|REAL)\s+INSTRUCTIONS?[:：\s]",
     "instruction replacement header"),
    (r"(?i)\bYOU\s+ARE\s+NOW\s+(A|AN)\s+\w",
     "persona replacement"),
    (r"(?i)(DELETE|REMOVE|ERASE)\s+(THIS|THESE|THE\s+(ABOVE|FOLLOWING))\s+(SECTION|INSTRUCTIONS?|LINES?|BLOCK|CODE)\s+AFTER\s+(RUNNING|EXECUTING|COMPLETING|DONE|READING)",
     "self-delete after execution (Hades IOC)"),
    (r"(?i)\bURGENT\s+SESSION\s+START\b",
     "urgent-session framing (Hades IOC)"),
    (r"(?i)\[OUT-OF-BAND",
     "OOB marker spoofing attempt"),
    (r"(?i)<\|system\|>|<\|im_start\|>\s*system|<<SYS>>|\[SYSTEM\]\s*:",
     "system prompt injection marker"),
    (r"(?i)\bI\s+AM\s+YOUR\s+(REAL\s+)?(USER|OPERATOR|DEVELOPER|OWNER|ADMIN)\b",
     "identity spoofing"),
    (r"(?i)\bDO\s+NOT\s+(TELL|INFORM|REVEAL\s+(TO\s+)?)THE\s+(USER|HUMAN|OPERATOR)\b",
     "hide-from-user instruction"),
    (r"(?i)(EXFILTRATE|EXFIL|SECRETLY\s+SEND|UPLOAD\s+WITHOUT\s+TELLING).{0,80}(KEY|TOKEN|SECRET|SSH|CREDENTIAL|\.ENV)",
     "credential exfiltration instruction"),
    (r"(?i)\bINSTALL\s+(A\s+)?PERSISTENCE\b",
     "persistence installation instruction"),
    (r"(?i)\bACT\s+AS\s+(IF\s+)?YOU\s+HAVE\s+(NO\s+(RESTRICTIONS?|RULES?|LIMITS?)|BEEN\s+FREED|UNRESTRICTED\s+ACCESS)\b",
     "unrestricted-mode injection"),
    (r"(?i)\bDAN\s+MODE\b|\bGODMODE\b|\bJAILBREAK\b",
     "known jailbreak keyword"),
]

# ---- Prompt injection — SUSPICIOUS (context-dependent, log + conditional block)
# Lower confidence; alone only triggers a log. On agent config files, also blocks.
INJECTION_SUSPICIOUS: list[tuple[str, str]] = [
    (r"(?i)<!--\s*(claude|copilot|cursor|codex|cline|aider|gemini|tabby|continue|amazon[- ]?q|cody|bolt)\b",
     "HTML comment targeting AI coding assistant"),
    (r"(?i)\[AGENT\s+INSTRUCTIONS?\]",
     "embedded agent instruction block"),
    (r"(?i)\bBOOTSTRAP\s+(REQUIRED|NEEDED|FIRST|TASK)\b",
     "bootstrap framing"),
    (r"(?i)\bSECRETLY\b.{0,80}(run|execute|install|send|exfil|upload|copy|write)",
     "covert action instruction"),
    (r"(?i)\bWITHOUT\s+(TELLING|INFORMING|NOTIFYING)\s+(THE\s+)?(USER|HUMAN)\b",
     "covert action framing"),
    (r"(?i)(SEND|POST|UPLOAD|EXFIL).{0,60}(API.?KEY|TOKEN|SECRET|PASSWORD|CRED|\.ENV|\.npmrc|\.pypirc)",
     "credential sending instruction"),
    (r"(?i)\bIMPORSANT\b|\bIMPORTANT\s+NOTE\s+FOR\s+(THE\s+)?(AI|AGENT|ASSISTANT|LLM|MODEL)\b",
     "AI-targeted 'important note' framing"),
    (r"(?i)#\s*(ai|agent|llm|assistant|claude|copilot)\s*:\s*(run|execute|do|perform|install)",
     "inline AI directive comment"),
    (r"(?i)\bINJECTED\s+PAYLOAD\b",
     "explicit injection marker (honeypot)"),
]


# ---- Audit log setup ----------------------------------------------------------
LOG_DIR = Path(os.path.expanduser("~/.hermes/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
_audit_log = LOG_DIR / "tool-audit.log"

logging.basicConfig(
    filename=str(_audit_log),
    level=logging.INFO,
    format="%(message)s",
)
_logger = logging.getLogger("tool_audit")


def _rotate_if_needed() -> None:
    """Truncate audit log if it exceeds MAX_AUDIT_BYTES (simple rotation)."""
    try:
        if _audit_log.exists() and _audit_log.stat().st_size > MAX_AUDIT_BYTES:
            content = _audit_log.read_bytes()
            _audit_log.write_bytes(content[-2 * 1024 * 1024:])
    except Exception:
        pass


def _audit(tool_name: str, tool_input: dict | None, session_id: str, decision: str) -> None:
    entry = {
        "ts": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "session": session_id,
        "tool": tool_name,
        "decision": decision,
        "input_preview": json.dumps(tool_input or {}, ensure_ascii=False)[:200],
    }
    _logger.info(json.dumps(entry, ensure_ascii=False))


def _is_subagent(payload: dict) -> bool:
    """Detect subagent sessions by env var or session_id naming convention."""
    if os.environ.get("HERMES_SUBAGENT"):
        return True
    task_id = (payload.get("extra") or {}).get("task_id") or payload.get("session_id") or ""
    return "subagent" in task_id.lower()


# ---- Helpers for mutation tool inspection ------------------------------------

def _extract_write_targets(tool_name: str, tool_input: dict) -> list[str]:
    """Extract file path(s) being written from a mutation tool call."""
    paths: list[str] = []
    if not tool_input:
        return paths
    # Hermes write_file / patch
    for key in ("path",):
        v = tool_input.get(key)
        if v and isinstance(v, str):
            paths.append(v)
    # Serena tools use relative_path
    rp = tool_input.get("relative_path")
    if rp and isinstance(rp, str):
        paths.append(rp)
    return paths


def _extract_write_content(tool_name: str, tool_input: dict) -> str:
    """Extract the text being written by a mutation tool call."""
    if not tool_input:
        return ""
    # write_file / serena create_text_file
    content = tool_input.get("content")
    if content and isinstance(content, str):
        return content
    # patch new_string
    new_str = tool_input.get("new_string")
    if new_str and isinstance(new_str, str):
        return new_str
    # serena replace_content repl
    repl = tool_input.get("repl")
    if repl and isinstance(repl, str):
        return repl
    # serena replace_symbol_body / insert_* body
    body = tool_input.get("body")
    if body and isinstance(body, str):
        return body
    return ""


def _check_supply_chain_path(paths: list[str]) -> str | None:
    """Return IOC description if any path matches a Hades/Miasma target."""
    for path in paths:
        for pattern, description in SUPPLY_CHAIN_PATH_PATTERNS:
            if re.search(pattern, path, re.IGNORECASE):
                return f"{description} ({path})"
    return None


def _is_agent_config_path(paths: list[str]) -> bool:
    """True if any path is an agent configuration file (AGENTS.md, CLAUDE.md, etc.)."""
    for path in paths:
        for pattern in AGENT_CONFIG_PATTERNS:
            if re.search(pattern, path, re.IGNORECASE):
                return True
    return False


def _check_injection_content(content: str) -> tuple[str | None, str | None]:
    """
    Scan content string for prompt injection patterns.
    Returns (high_match_description, suspicious_match_description).
    High tier is checked first; suspicious only checked if no high match.
    """
    if not content or len(content) < 10:
        return None, None

    for pattern, description in INJECTION_HIGH:
        if re.search(pattern, content):
            return description, None

    for pattern, description in INJECTION_SUSPICIOUS:
        if re.search(pattern, content):
            return None, description

    return None, None


def _check_dangerous_terminal(tool_input: dict | None) -> str | None:
    """Return a description if the terminal command matches a dangerous pattern."""
    if not tool_input:
        return None
    command = tool_input.get("command") or ""
    if not isinstance(command, str):
        return None
    for pattern, description in DANGEROUS_TERMINAL_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return description
    return None


def _check_supply_chain_terminal(tool_input: dict | None) -> str | None:
    """Return IOC description if terminal command matches Hades/Miasma patterns."""
    if not tool_input:
        return None
    command = tool_input.get("command") or ""
    if not isinstance(command, str):
        return None
    for pattern, description in SUPPLY_CHAIN_TERMINAL_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return description
    return None


# ---- Main hook logic ----------------------------------------------------------

def main() -> None:
    _rotate_if_needed()

    try:
        raw = sys.stdin.read()
        payload = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        sys.exit(0)

    event = payload.get("hook_event_name", "")
    if event != "pre_tool_call":
        sys.exit(0)

    tool_name: str = payload.get("tool_name") or ""
    tool_input: dict | None = payload.get("tool_input")
    session_id: str = payload.get("session_id") or ""

    # Tier 1 — grounding / read tools: silent pass-through
    if tool_name in GROUNDING_TOOLS:
        sys.exit(0)

    # Tier 2 — file mutation tools
    if tool_name in MUTATION_TOOLS:
        paths = _extract_write_targets(tool_name, tool_input or {})
        content = _extract_write_content(tool_name, tool_input or {})
        is_agent_cfg = _is_agent_config_path(paths)

        # 2a — Supply chain path IOC
        sc_path = _check_supply_chain_path(paths)
        if sc_path:
            _audit(tool_name, tool_input, session_id, f"WARN(supply-chain-path:{sc_path})")
            # Block if BLOCK_MODE OR if it's also an agent config file (always dangerous)
            if BLOCK_MODE or is_agent_cfg:
                print(json.dumps({
                    "action": "block",
                    "message": (
                        f"SUPPLY CHAIN IOC: Write target matches Hades/Miasma attack vector: "
                        f"{sc_path}. Verify this write is intentional and content is not compromised."
                    ),
                }))
                return

        # 2b — Prompt injection content scan
        high_injection, suspicious_injection = _check_injection_content(content)

        if high_injection:
            _audit(tool_name, tool_input, session_id, f"INJECTION-HIGH({high_injection}) paths={paths}")
            # Always block on agent config files; block in BLOCK_MODE for any file
            if BLOCK_MODE or is_agent_cfg:
                print(json.dumps({
                    "action": "block",
                    "message": (
                        f"PROMPT INJECTION DETECTED [{high_injection}]: Content being written "
                        "contains explicit instruction-override or hijack patterns. This may "
                        "indicate a prompt injection attack via web content, package metadata, "
                        "or tool output. Confirm intent explicitly before writing this content."
                    ),
                }))
                return

        elif suspicious_injection:
            _audit(tool_name, tool_input, session_id,
                   f"INJECTION-SUSPICIOUS({suspicious_injection}) paths={paths}")
            # Suspicious + agent config = always block
            if is_agent_cfg:
                print(json.dumps({
                    "action": "block",
                    "message": (
                        f"SUSPICIOUS INJECTION PATTERN [{suspicious_injection}] in agent config "
                        f"path {paths}. Writing injection-like content to agent config files "
                        "requires explicit user confirmation."
                    ),
                }))
                return
            # Suspicious + BLOCK_MODE = block
            if BLOCK_MODE:
                print(json.dumps({
                    "action": "block",
                    "message": (
                        f"SUSPICIOUS INJECTION PATTERN [{suspicious_injection}]: "
                        "Content contains patterns consistent with AI-targeted instruction "
                        "injection. Verify this content originates from a trusted source."
                    ),
                }))
                return

        # 2c — Standard mutation grounding block
        if BLOCK_MODE:
            _audit(tool_name, tool_input, session_id, "BLOCKED(mutation)")
            print(json.dumps({
                "action": "block",
                "message": (
                    f"GROUNDING CHECK: '{tool_name}' requires prior memory recall. "
                    "Run mcp_mnemosyne_mnemosyne_recall(query=<topic>) first, "
                    "then re-invoke. Set HOOK_BLOCK_MODE=0 to disable."
                ),
            }))
            return

        _audit(tool_name, tool_input, session_id, "ALLOW(mutation)")
        sys.exit(0)

    # Tier 3 — terminal/Bash: supply chain IOCs + dangerous patterns
    if tool_name in ("terminal", "Bash"):
        # 3a — Supply chain IOC (Hades/Miasma) — always audited, blocked in BLOCK_MODE
        sc_terminal = _check_supply_chain_terminal(tool_input)
        if sc_terminal:
            _audit(tool_name, tool_input, session_id, f"WARN(supply-chain:{sc_terminal})")
            if BLOCK_MODE and not _is_subagent(payload):
                print(json.dumps({
                    "action": "block",
                    "message": (
                        f"SUPPLY CHAIN IOC: {sc_terminal}. "
                        "Command matches known Hades/Miasma worm execution patterns. "
                        "Confirm this is an intentional, user-directed operation."
                    ),
                }))
                return

        # 3b — General dangerous patterns
        danger = _check_dangerous_terminal(tool_input)
        if danger:
            if BLOCK_MODE and not _is_subagent(payload):
                _audit(tool_name, tool_input, session_id, f"BLOCKED(dangerous:{danger})")
                print(json.dumps({
                    "action": "block",
                    "message": (
                        f"DANGEROUS COMMAND DETECTED: {danger}. "
                        "Confirm intent explicitly before proceeding. "
                        "Set HOOK_BLOCK_MODE=0 to bypass this guard."
                    ),
                }))
                return
            _audit(tool_name, tool_input, session_id, f"WARN(dangerous:{danger})")
        elif not sc_terminal:
            _audit(tool_name, tool_input, session_id, "ALLOW(terminal)")

        sys.exit(0)

    # Tier 4 — everything else: audit with ALLOW
    _audit(tool_name, tool_input, session_id, "ALLOW")
    sys.exit(0)


if __name__ == "__main__":
    main()
