#!/usr/bin/env bash
# Install the Claude Code hooks from this repo into ~/.claude/hooks.
#
#   install.sh          copy repo -> ~/.claude/hooks (backing up what is there)
#   install.sh --check  report drift and exit 1 if any; changes nothing
#
# Nothing kept the two copies in sync before this script existed, and they did
# diverge: the deployed pre_tool_grounding.py had been hand-edited to accept the
# Claude Code "PreToolUse" event while the repo copy still only accepted the
# Hermes name, so a fresh install would have silently disabled the hook.

set -uo pipefail

HOOKS=(pre_llm_grounding.py pre_tool_grounding.py session_end_sync.py session_end_sync.sh
       session_end_learn.py session_end_learn.sh legacy_env.py workflow_balanced_models.py)
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${CLAUDE_HOOKS_DIR:-$HOME/.claude/hooks}"

if [[ "${1:-}" == "--check" ]]; then
  drift=0
  for h in "${HOOKS[@]}"; do
    if [[ ! -f "$DEST/$h" ]]; then
      echo "MISSING  $DEST/$h"; drift=1
    elif ! diff -q "$SRC/$h" "$DEST/$h" >/dev/null; then
      echo "DRIFTED  $h"; diff -u "$SRC/$h" "$DEST/$h" | sed -n '3,$p' | head -20; drift=1
    elif [[ "$(stat -c '%a' "$DEST/$h")" != "755" ]]; then
      echo "MODE     $DEST/$h is $(stat -c '%a' "$DEST/$h"), expected 755"; drift=1
    fi
  done
  [[ $drift -eq 0 ]] && echo "hooks in sync"
  exit $drift
fi

mkdir -p "$DEST"
backup="$DEST/backup-$(date +%Y%m%d-%H%M%S)"
for h in "${HOOKS[@]}"; do
  [[ -f "$SRC/$h" ]] || { echo "no such hook in repo: $h" >&2; exit 1; }
  if [[ -f "$DEST/$h" ]] && ! diff -q "$SRC/$h" "$DEST/$h" >/dev/null; then
    mkdir -p "$backup"; cp "$DEST/$h" "$backup/"
  fi
  cp "$SRC/$h" "$DEST/$h"; chmod 755 "$DEST/$h"
  echo "installed $h"
done
[[ -d "$backup" ]] && echo "previous copies backed up to $backup"
exit 0
