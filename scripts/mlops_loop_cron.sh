#!/usr/bin/env bash
# Cron entrypoint for the MLOps loop.
#
# The loop promotes a model by WRITING INTO THE REPO — grounding_bleed_clf.joblib,
# grounding_dataset.jsonl and metrics.json under deep_think_loci/grounding/ are
# all tracked. Running it in the working tree leaves that tree dirty every
# morning, in a checkout the operator also uses interactively, which is how a
# rebuilt dataset ends up in an unrelated commit. So this runs in a dedicated
# worktree pinned to origin/main and leaves the real checkout alone.
#
# A promotion is not applied automatically. If the loop promotes, the artifacts
# sit in the run worktree and this says so; adopting them is a deliberate commit.
#
# Exit codes are loop.py's own: 0 clean, 1 one or more steps failed. A step that
# was skipped (Ollama unreachable, nothing new to train on) is not a failure.
set -uo pipefail

REPO="${LOCI_REPO:-/home/rjmendez/development/loci}"
PY="$REPO/mcp/.venv/bin/python"
RUN_TREE="${LOCI_MLOPS_TREE:-$HOME/.loci/mlops/worktree}"
STATE="${LOCI_MLOPS_STATE:-$HOME/.loci/mlops}"
mkdir -p "$STATE"

[ -x "$PY" ] || { echo "mlops-loop: no interpreter at $PY" >&2; exit 1; }

# RUN_TREE reaches `git reset --hard`, `git clean -fd` and `rm -rf`, and it comes
# from the environment. A wrapper written to protect the working checkout must
# not be one LOCI_MLOPS_TREE=$REPO turns into the thing that destroys it.
case "$RUN_TREE" in
  /*) ;;
   *) echo "mlops-loop: LOCI_MLOPS_TREE must be an absolute path, got '$RUN_TREE'" >&2; exit 1 ;;
esac
RUN_TREE_RAW="$RUN_TREE"
RUN_TREE="${RUN_TREE%/}"
case "$RUN_TREE" in
  "" | "/" | "$HOME" | "$REPO" | "$REPO"/*)
    echo "mlops-loop: refusing to use '$RUN_TREE_RAW' as the run worktree — it is the" >&2
    echo "            repo, inside it, \$HOME, or /; this path gets reset --hard and rm -rf" >&2
    exit 1 ;;
esac
if [ -e "$RUN_TREE" ] && [ ! -e "$RUN_TREE/.git" ]; then
  echo "mlops-loop: '$RUN_TREE' exists and is not a git worktree — refusing to remove it" >&2
  exit 1
fi
mkdir -p "$(dirname "$RUN_TREE")"

git -C "$REPO" fetch -q origin main 2>/dev/null || \
  echo "mlops-loop: fetch failed, running against whatever the worktree has" >&2

if [ -e "$RUN_TREE/.git" ]; then
  # A failed reset means the loop would run against an unknown revision, which is
  # exactly the baseline guarantee this wrapper exists to provide. Fail instead.
  git -C "$RUN_TREE" reset -q --hard origin/main || {
    echo "mlops-loop: reset to origin/main failed in $RUN_TREE — not running" >&2; exit 1; }
  git -C "$RUN_TREE" clean -qfd || {
    echo "mlops-loop: clean failed in $RUN_TREE — not running" >&2; exit 1; }
else
  rm -rf "$RUN_TREE"
  git -C "$REPO" worktree add -q --detach "$RUN_TREE" origin/main || {
    echo "mlops-loop: could not create the run worktree at $RUN_TREE" >&2; exit 1; }
fi

# -u so the log is watchable while a step runs for twenty minutes.
"$PY" -u "$RUN_TREE/mlops/loop.py" "$@"
rc=$?

changed="$(git -C "$RUN_TREE" status --porcelain -- deep_think_loci/grounding/)"
if [ -n "$changed" ]; then
  echo "mlops-loop: the run produced new grounding artifacts in $RUN_TREE"
  printf '%s\n' "$changed" | sed 's/^/  /'
  echo "mlops-loop: review and commit them deliberately; nothing was applied to $REPO"
fi

[ $rc -ne 0 ] && echo "mlops-loop: exit $rc — one or more steps failed, see above" >&2
exit $rc
