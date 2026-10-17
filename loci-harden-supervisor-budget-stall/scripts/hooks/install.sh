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

HOOKS=(pre_llm_grounding.py pre_tool_grounding.py session_end_sync.py legacy_env.py
       workflow_balanced_models.py)
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${CLAUDE_HOOKS_DIR:-$HOME/.claude/hooks}"
SETTINGS="${CLAUDE_SETTINGS:-$HOME/.claude/settings.json}"

if [[ "${1:-}" == "--check" ]]; then
  drift=0
  for h in "${HOOKS[@]}"; do
    if [[ ! -f "$DEST/$h" ]]; then
      echo "MISSING  $DEST/$h"; drift=1
    elif ! diff -q "$SRC/$h" "$DEST/$h" >/dev/null; then
      echo "DRIFTED  $h"; diff -u "$SRC/$h" "$DEST/$h" | sed -n '3,$p' | head -20; drift=1
    fi
  done

  # HOOKS is what this repo ships, not what Claude Code runs. settings.json can
  # point an event at a file in the hooks dir that HOOKS does not name -- the Stop
  # hook on the reference host runs session_end_sync.sh, a wrapper that exports the
  # Qdrant and embedding endpoints and exists in no commit. Checking HOOKS alone
  # examined every file except the entry point and still said "hooks in sync".
  unmanaged=()
  if [[ -f "$SETTINGS" ]]; then
    while IFS= read -r path; do
      name="${path##*/}"
      known=0
      for h in "${HOOKS[@]}"; do [[ "$h" == "$name" ]] && { known=1; break; }; done
      [[ $known -eq 1 ]] || unmanaged+=("$name")
    done < <(tr -s '[:space:],"' '\n' < "$SETTINGS" | grep -F "$DEST/" | sort -u)
  fi
  if [[ ${#unmanaged[@]} -gt 0 ]]; then
    for u in "${unmanaged[@]}"; do
      echo "UNMANAGED $u  (invoked from $SETTINGS, not managed by this script)"
    done
  fi

  if [[ $drift -eq 0 ]]; then
    if [[ ${#unmanaged[@]} -gt 0 ]]; then
      echo "${#HOOKS[@]} hooks in sync; ${#unmanaged[@]} invoked but not managed here"
    else
      echo "hooks in sync"
    fi
  fi
  exit $drift
fi

mkdir -p "$DEST"
backup="$DEST/backup-$(date +%Y%m%d-%H%M%S)"
for h in "${HOOKS[@]}"; do
  [[ -f "$SRC/$h" ]] || { echo "no such hook in repo: $h" >&2; exit 1; }
  if [[ -f "$DEST/$h" ]] && ! diff -q "$SRC/$h" "$DEST/$h" >/dev/null; then
    mkdir -p "$backup"; cp "$DEST/$h" "$backup/"
  fi
  cp "$SRC/$h" "$DEST/$h"; chmod +x "$DEST/$h"
  echo "installed $h"
done
[[ -d "$backup" ]] && echo "previous copies backed up to $backup"
exit 0
