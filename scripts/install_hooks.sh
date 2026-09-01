#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK_DIR="$REPO_ROOT/.git/hooks"

install_hook() {
  local name="$1"
  local src="$REPO_ROOT/scripts/hooks/$name"
  local dst="$HOOK_DIR/$name"
  if [ -f "$src" ]; then
    ln -sf "$src" "$dst"
    chmod +x "$src"
    echo "Installed: $name"
  fi
}

install_hook post-commit

echo "Done. Hooks installed from scripts/hooks/."
