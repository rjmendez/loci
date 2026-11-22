#!/usr/bin/env bash
# Removes the build/run artifacts the camera-targets work (Stages 1b/1c, Amendments 6/7) left inside WSL.
# Dry run by default: prints the size of every candidate. `--apply` deletes. Only paths created by that work.
#   bash camera_artifacts_cleanup.sh [--apply]
set -u
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1
H="${H:-$HOME}"   # when the Ubuntu disk is mounted under another distro: H=/mnt/ubuntu/home/rjmendez
sz() { du -sh "$1" 2>/dev/null | cut -f1; }
rm_path() { # path why
  [ -e "$1" ] || { printf '%-8s %s\n' absent "$1"; return; }
  printf '%-8s %s   (%s)\n' "$(sz "$1")" "$1" "$2"
  [ $APPLY = 1 ] && rm -rf -- "$1"
}
echo "free before: $(df -h "${ROOT:-/}" | tail -1)"
# 1. cargo targets of the camera work and the race replay (rebuildable)
rm_path "$H/.cache/flyengine-target-camtargets" "cargo target, camera work"
rm_path "$H/.cache/flyengine-target-racereplay" "cargo target, race replay"
# 2. per-job terraform and source copies of stage 1b / 1c: providers, bundles. Run evidence stays (generations.jsonl,
#    manifest.json, RESULT.txt, logs); S3 holds the full run dirs and the repo holds the Stage 1b copy.
for root in "$H/.local/state/flybrain-cloud/stage1b" "$H/.local/state/flybrain-cloud/stage1c"; do
  [ -d "$root" ] || continue
  for d in "$root"/*/; do
    [ -d "$d" ] || continue
    rm_path "${d}tf/.terraform" "terraform provider cache of one job"
    rm_path "${d}src" "source/bundle copy of one job"
    rm_path "${d}run/checkpoint.bin" "resume checkpoint, S3 holds it"
  done
done
# 3. scratch
for f in /tmp/a7gate /tmp/a7gateAE /tmp/s1c-list /tmp/fd_all.jsonl /tmp/fd_gate_1201.jsonl; do rm_path "$f" "scratch"; done
# 4. the detached worktree made for the race replay build, only if it has no local changes
W="$H/development/flybrain-worktrees/race-replay-e21bd5b9"
if [ -d "$W" ]; then
  if [ -z "$(git -C "$W" status --porcelain 2>/dev/null)" ]; then
    printf '%-8s %s   (detached worktree, clean)\n' "$(sz "$W")" "$W"
    [ $APPLY = 1 ] && git -C "$H/development/flybrain-worktrees/body-cameras" worktree remove --force "$W"
  else
    echo "KEPT     $W has local changes"
  fi
fi
echo "free after:  $(df -h "${ROOT:-/}" | tail -1)"
echo "(this frees space INSIDE the ext4; the VHDX on D: only shrinks after fstrim + the Sunday compaction)"
