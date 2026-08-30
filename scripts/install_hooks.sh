#!/usr/bin/env bash
# Install/check the Git hooks used by this checkout.
#
# This is intentionally separate from scripts/hooks/install.sh: that installer
# manages Claude Code agent hooks in ~/.claude/hooks, while this one manages
# repository hooks in .git/hooks. The two entry points now share the same
# copy-not-symlink, backup-on-drift, --check behaviour.

set -uo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK_DIR="$REPO_ROOT/.git/hooks"
HOOKS=(post-commit)
DEPENDENCIES=(post-commit-contract-extract.sh)

mode="${1:-install}"
if [[ "$mode" != "install" && "$mode" != "--check" ]]; then
  echo "usage: $0 [--check]" >&2
  exit 2
fi

check_hook() {
  local name="$1"
  local src="$REPO_ROOT/scripts/hooks/$name"
  local dst="$HOOK_DIR/$name"
  local drift=0
  if [[ ! -f "$src" ]]; then
    echo "MISSING  $src"; return 1
  fi
  if [[ ! -f "$dst" ]]; then
    echo "MISSING  $dst"; drift=1
  elif ! diff -q "$src" "$dst" >/dev/null; then
    echo "DRIFTED  $dst"; drift=1
  elif [[ "$(stat -c '%a' "$dst")" != "755" ]]; then
    echo "MODE     $dst is $(stat -c '%a' "$dst"), expected 755"; drift=1
  fi
  return "$drift"
}

check_dependency() {
  local name="$1"
  local src="$REPO_ROOT/scripts/hooks/$name"
  [[ -f "$src" ]] || { echo "MISSING  $src"; return 1; }
  [[ -x "$src" ]] || { echo "MODE     $src is not executable"; return 1; }
}

if [[ "$mode" == "--check" ]]; then
  drift=0
  for h in "${HOOKS[@]}"; do
    check_hook "$h" || drift=1
  done
  for h in "${DEPENDENCIES[@]}"; do
    check_dependency "$h" || drift=1
  done
  [[ $drift -eq 0 ]] && echo "git hooks in sync"
  exit "$drift"
fi

mkdir -p "$HOOK_DIR"
backup="$HOOK_DIR/backup-$(date +%Y%m%d-%H%M%S)"
for h in "${HOOKS[@]}"; do
  src="$REPO_ROOT/scripts/hooks/$h"
  dst="$HOOK_DIR/$h"
  [[ -f "$src" ]] || { echo "no such hook in repo: $h" >&2; exit 1; }
  if [[ -f "$dst" ]] && ! diff -q "$src" "$dst" >/dev/null; then
    mkdir -p "$backup"; cp "$dst" "$backup/"
  fi
  cp "$src" "$dst"; chmod 755 "$dst"
  chmod 755 "$src"
  echo "installed git hook $h"
done
for h in "${DEPENDENCIES[@]}"; do
  src="$REPO_ROOT/scripts/hooks/$h"
  [[ -f "$src" ]] || { echo "no such hook dependency in repo: $h" >&2; exit 1; }
  chmod 755 "$src"
  echo "verified git hook dependency $h"
done

[[ -d "$backup" ]] && echo "previous git hooks backed up to $backup"
echo "Done. Git hooks installed from scripts/hooks/."
