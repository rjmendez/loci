#!/usr/bin/env bash
# Post-commit: extract contracts from changed files into the active Loci investigation.
# Non-blocking — all paths end with || true so a failure never aborts the commit.
# Fires only when .py, .ts, .go, .rs, or .java files changed.

set -euo pipefail

# Check for relevant file changes
if ! git diff-tree --no-commit-id -r --name-only HEAD 2>/dev/null \
     | grep -qE '\.(py|ts|go|rs|java)$'; then
  exit 0
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"

# Need both claude and an active investigation to proceed
command -v claude >/dev/null 2>&1 || exit 0

INV="${HERMES_ACTIVE_INVESTIGATION:-}"
if [ -z "$INV" ]; then
  # Try reading from the Loci active investigation file if present
  LOCI_ACTIVE="$HOME/.loci/active_investigation"
  if [ -f "$LOCI_ACTIVE" ]; then
    INV="$(cat "$LOCI_ACTIVE" | tr -d '[:space:]')"
  fi
fi

if [ -z "$INV" ]; then
  exit 0
fi

WORKFLOW="$REPO_ROOT/deep_think_loci/workflows/contract-sync.js"
if [ ! -f "$WORKFLOW" ]; then
  exit 0
fi

# Run contract-sync in the background (fire-and-forget, non-blocking)
claude --workflow "$WORKFLOW" \
       --args "{\"root\": \"$REPO_ROOT\", \"loci_investigation\": \"$INV\", \"since_commit\": \"HEAD~1\"}" \
       --dangerously-skip-permissions \
  >> "${TMPDIR:-/tmp}/loci-contract-extract.log" 2>&1 &

exit 0
