#!/usr/bin/env bash
# Git post-commit hook: re-ingests Loci codebase into loci-codebase Loci investigation
# when Python source files change.
# Install: ln -sf "$(pwd)/scripts/hooks/post_commit_ingest.sh" .git/hooks/post-commit

set -euo pipefail

# Only ingest commits on main (or LOCI_HOOK_INGEST_BRANCH) in the primary
# worktree; linked worktrees, feature branches and detached HEADs are skipped so
# unmerged code is never ingested. LOCI_HOOK_INGEST=0 disables it entirely.
[ "${LOCI_HOOK_INGEST:-1}" != "0" ] || exit 0
_git_dir="$(cd "$(git rev-parse --git-dir 2>/dev/null)" 2>/dev/null && pwd -P)" || exit 0
_common_dir="$(cd "$(git rev-parse --git-common-dir 2>/dev/null)" 2>/dev/null && pwd -P)" || exit 0
[ -n "$_git_dir" ] && [ "$_git_dir" = "$_common_dir" ] || exit 0
[ "$(git symbolic-ref --quiet --short HEAD 2>/dev/null || true)" = "${LOCI_HOOK_INGEST_BRANCH:-main}" ] || exit 0

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
