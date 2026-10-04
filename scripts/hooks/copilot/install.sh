#!/usr/bin/env bash
# Install the Loci RAG grounding hooks for GitHub Copilot CLI (user scope).
#   install.sh [REPO_DIR]     install; also write REPO_DIR/.github/hooks/ if given
#   install.sh --check        report drift and exit 1 if any; change nothing
# WSL -> Windows CLI: set COPILOT_CONFIG_DIR=/mnt/c/Users/<you>/.copilot
set -euo pipefail
SRC="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
COPILOT_DIR="${COPILOT_CONFIG_DIR:-$HOME/.copilot}"
HOOKS_DEST="$COPILOT_DIR/hooks"
WRAPPER_DEST="$HOOKS_DEST/loci_rag_grounding.sh"

# Raw command (unescaped quotes); Python json.dumps escapes it into the file.
raw_cmd() {
  case "$WRAPPER_DEST" in
    /mnt/[a-z]/*) printf 'wsl.exe -e bash -lc "%s %s"' "$WRAPPER_DEST" "$1" ;;
    *)            printf '%s %s' "$WRAPPER_DEST" "$1" ;;
  esac
}
write_json() { # $1 = dest path
  CMD_TOOL="$(raw_cmd tool)" CMD_LLM="$(raw_cmd llm)" DEST="$1" python3 - <<'PY'
import json, os
doc = {
  "$schema": "https://json.schemastore.org/github-copilot-cli-hooks.json",
  "description": "Enforce Loci RAG grounding before tool use and on each prompt (main agent + subagents). Fail-open.",
  "hooks": {
    "UserPromptSubmit": [{"matcher":"*","hooks":[{"type":"command","command":os.environ["CMD_LLM"],"timeout":30}]}],
    "PreToolUse":       [{"matcher":"*","hooks":[{"type":"command","command":os.environ["CMD_TOOL"],"timeout":30}]}],
  },
}
open(os.environ["DEST"],"w").write(json.dumps(doc, indent=2) + "\n")
PY
}

if [[ "${1:-}" == "--check" ]]; then
  drift=0
  diff -q "$SRC/loci_rag_grounding.sh" "$WRAPPER_DEST" >/dev/null 2>&1 || { echo "DRIFT wrapper"; drift=1; }
  [[ -f "$COPILOT_DIR/copilot-instructions.md" ]] || { echo "MISSING copilot-instructions.md"; drift=1; }
  [[ -f "$HOOKS_DEST/loci-rag-grounding.json" ]] || { echo "MISSING loci-rag-grounding.json"; drift=1; }
  [[ $drift -eq 0 ]] && echo "copilot hooks in sync"; exit $drift
fi

mkdir -p "$HOOKS_DEST"
install -m 755 "$SRC/loci_rag_grounding.sh" "$WRAPPER_DEST"; echo "installed $WRAPPER_DEST"
instr="$COPILOT_DIR/copilot-instructions.md"
if [[ -f "$instr" ]] && ! diff -q "$SRC/copilot-instructions.md" "$instr" >/dev/null 2>&1; then cp "$instr" "$instr.bak-$(date +%Y%m%d-%H%M%S)"; fi
install -m 644 "$SRC/copilot-instructions.md" "$instr"; echo "installed $instr"
write_json "$HOOKS_DEST/loci-rag-grounding.json"; echo "installed $HOOKS_DEST/loci-rag-grounding.json"
if [[ -n "${1:-}" && -d "${1:-}" ]]; then mkdir -p "$1/.github/hooks"; write_json "$1/.github/hooks/loci-rag-grounding.json"; echo "installed $1/.github/hooks/loci-rag-grounding.json"; fi
echo "Done. Confirm with /env inside a Copilot CLI session."
