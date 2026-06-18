#!/usr/bin/env python3
"""
pre_tool_call hook — grounding enforcement, dangerous pattern detection, audit log.

v2 changes over original:
  - read_file, search_files, read_file added to GROUNDING_TOOLS (no audit noise on reads)
  - terminal added to DANGEROUS_COMMANDS tier with pattern detection
  - Dangerous shell pattern detection: rm -rf, DROP TABLE, force-push, kubectl delete
  - Log rotation: audit log capped at 5MB (matches Hermes logging config)
  - Subagent detection via session_id pattern as well as env var
  - Added newer serena write tool names
  - BLOCK_MODE=1 also blocks dangerous terminal commands (not just file mutations)

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
    # Mnemosyne MCP (Claude Code uses double-underscore prefix)
    "mcp__mnemosyne__mnemosyne_recall",
    "mcp__mnemosyne__mnemosyne_remember",
    "mcp__mnemosyne__mnemosyne_get_stats",
    "mcp__mnemosyne__mnemosyne_scratchpad_read",
    "mcp__mnemosyne__mnemosyne_scratchpad_write",
    "mcp__mnemosyne__mnemosyne_sleep",
    "mcp__mnemosyne__mnemosyne_triple_add",
    "mcp__mnemosyne__mnemosyne_triple_query",
    # Legacy single-underscore format (keep for backward compat)
    "mcp_mnemosyne_mnemosyne_recall",
    "mcp_mnemosyne_mnemosyne_remember",
    "mcp_mnemosyne_mnemosyne_get_stats",
    "mcp_mnemosyne_mnemosyne_scratchpad_read",
    "mcp_mnemosyne_mnemosyne_scratchpad_write",
    "mcp_mnemosyne_mnemosyne_sleep",
    # Open Design MCP
    "mcp_open_design_get_artifact",
    "mcp_open_design_get_project",
    "mcp_open_design_get_file",
    "mcp_open_design_search_files",
    "mcp_open_design_list_files",
    "mcp_open_design_list_projects",
    "mcp_open_design_get_active_context",
    "mcp_open_design_list_agents",
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
    # Claude Code built-in read tools — early exit, no audit needed
    "Read",
    "ToolSearch",
    "Agent",
    "Skill",
    # Hermes read tools — not mutations, suppress audit noise
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
            # Keep last 2MB
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


def _resolve_tier(tool_name: str) -> str:
    """Classify a tool name into its processing tier."""
    if tool_name in GROUNDING_TOOLS:
        return "grounding"
    if tool_name in MUTATION_TOOLS:
        return "mutation"
    if tool_name == "terminal":
        return "terminal"
    return "unknown"


def _authorize(
    tier: str,
    tool_name: str,
    tool_input: dict | None,
    is_subagent: bool,
) -> tuple[str, str]:
    """Return (verdict, audit_reason). verdict is 'allow' or 'block'."""
    if tier == "grounding":
        return "allow", "grounding-pass"
    if tier == "mutation":
        if BLOCK_MODE:
            return "block", "BLOCKED(mutation)"
        return "allow", "ALLOW(mutation)"
    if tier == "terminal":
        danger = _check_dangerous_terminal(tool_input)
        if danger:
            if BLOCK_MODE and not is_subagent:
                return "block", f"BLOCKED(dangerous:{danger})"
            return "allow", f"WARN(dangerous:{danger})"
        return "allow", "ALLOW(terminal)"
    return "allow", "ALLOW"


def _block_response(tier: str, reason: str, tool_name: str) -> dict:
    """Build the block action payload for a denied tool call."""
    if tier == "mutation":
        return {
            "action": "block",
            "message": (
                f"GROUNDING CHECK: '{tool_name}' requires prior memory recall. "
                "Run mcp__mnemosyne__mnemosyne_recall(query=<topic>) first, "
                "then re-invoke. Set HOOK_BLOCK_MODE=0 to disable."
            ),
        }
    danger_desc = reason.split("dangerous:", 1)[-1] if "dangerous:" in reason else "dangerous command"
    return {
        "action": "block",
        "message": (
            f"DANGEROUS COMMAND DETECTED: {danger_desc}. "
            "Confirm intent explicitly before proceeding. "
            "Set HOOK_BLOCK_MODE=0 to bypass this guard."
        ),
    }


def main() -> None:
    _rotate_if_needed()

    try:
        raw = sys.stdin.read()
        payload = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        sys.exit(0)

    if payload.get("hook_event_name", "") != "pre_tool_call":
        sys.exit(0)

    tool_name: str = payload.get("tool_name") or ""
    tool_input: dict | None = payload.get("tool_input")
    session_id: str = payload.get("session_id") or ""

    tier = _resolve_tier(tool_name)
    verdict, reason = _authorize(tier, tool_name, tool_input, _is_subagent(payload))

    if tier == "grounding":
        sys.exit(0)

    _audit(tool_name, tool_input, session_id, reason)

    if verdict == "block":
        print(json.dumps(_block_response(tier, reason, tool_name)))
        return

    sys.exit(0)


if __name__ == "__main__":
    main()
