#!/usr/bin/env bash
# Git post-commit hook: re-ingests Loci codebase into loci-codebase Loci investigation
# when Python source files change.
# Install: ln -sf "$(pwd)/scripts/hooks/post_commit_ingest.sh" .git/hooks/post-commit

set -euo pipefail

# Check if any Python files changed in this commit
if ! git diff --name-only HEAD~1 HEAD 2>/dev/null | grep -q '\.py$'; then
  exit 0
fi

echo "[loci-ingest] Python files changed — re-ingesting into loci-codebase investigation..."

# Run the ingest workflow via Claude Code CLI (non-interactive)
if command -v claude &>/dev/null; then
  claude --workflow .claude/workflows/loci-codebase-ingest.js \
    --dangerously-skip-permissions 2>&1 | tail -5 || true
else
  echo "[loci-ingest] claude CLI not found — skipping auto-ingest. Run manually."
fi
