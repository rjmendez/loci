#!/usr/bin/env bash
# Copilot CLI -> Loci RAG grounding bridge.
# Wired by a Copilot CLI hook (PreToolUse / UserPromptSubmit). Reads the hook
# event JSON on stdin, runs Loci's grounding hook (the same pre_*_grounding.py
# the Claude/hermes agents use), and passes its allow/block decision through.
# FAIL-OPEN: any error emits a grounding reminder and allows, so a session is
# never broken.
#
#   $1 = "tool" -> pre_tool_grounding.py (default) | "llm" -> pre_llm_grounding.py
#
# Loci hooks are found next to this script's parent (scripts/hooks/) or via
# LOCI_HOOKS_DIR.
set -uo pipefail
kind="${1:-tool}"
here="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
LOCI_HOOKS="${LOCI_HOOKS_DIR:-$(cd "$here/.." && pwd)}"
export HERMES_AGENT_ID="${HERMES_AGENT_ID:-copilot-$(hostname 2>/dev/null || echo cli)}"

if [ "$kind" = "llm" ]; then hook="$LOCI_HOOKS/pre_llm_grounding.py"; else hook="$LOCI_HOOKS/pre_tool_grounding.py"; fi

payload="$(cat 2>/dev/null || true)"
reminder='{"additionalContext":"MANDATORY: ground this in Loci RAG first (loci-rag_context_search / loci-ground / loci-investigation_search) and cite the evidence; do not answer from memory. Respect exploratory/mock/relative-only labels - never present sim/mock output as validated."}'

if [ -f "$hook" ] && command -v python3 >/dev/null 2>&1; then
  out="$(printf '%s' "$payload" | python3 "$hook" 2>/dev/null || true)"
  if [ -n "$out" ] && printf '%s' "$out" | python3 -c 'import sys,json; json.load(sys.stdin)' >/dev/null 2>&1; then
    printf '%s' "$out"; exit 0
  fi
fi
printf '%s' "$reminder"; exit 0
