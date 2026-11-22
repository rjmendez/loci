#!/bin/sh
# Permanent D: space guard, WSL side. Run as root inside Ubuntu (the d-free-watchdog calls it when D: free is low;
# also safe to run by hand). Removes ONLY regenerable build output and old logs. Never touches a protected pin.
#
#   wsl-cache-prune.sh [--dry-run] [--days N]
#
# Protected (never removed): every directory named in $PIN_FILE, plus the two that registered binaries/watchers use today.
# Written 2026-10-14 after D: hit 0 GB free and WSL failed to boot (VHDX 925 GB on D:, cargo target dirs ~98 GB in ~/.cache).
PATH=/usr/sbin:/sbin:/usr/bin:/bin
DRY=0; DAYS=7
while [ $# -gt 0 ]; do case "$1" in --dry-run) DRY=1;; --days) DAYS="$2"; shift;; esac; shift; done
H=/home/rjmendez
PIN_FILE=$H/.local/share/sitl-colony/cache-pins.txt
LOG=$H/.local/state/wsl-cache-prune.log
mkdir -p "$(dirname "$LOG")"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
act() { if [ "$DRY" = 1 ]; then say "DRY would run: $*"; else "$@" >>"$LOG" 2>&1; fi; }
protected() {
  case "$1" in flyengine-target-flare|flyengine-target-deathfloor) return 0;; esac
  [ -f "$PIN_FILE" ] && grep -qxF "$1" "$PIN_FILE" && return 0
  return 1
}
before=$(df -k / | awk 'NR==2{print $4}')
say "start dry=$DRY days=$DAYS free_k=$before"

# 1. per-build cargo target dirs under ~/.cache: remove unless protected or touched in the last N days
for d in "$H"/.cache/flyengine-target-*; do
  [ -d "$d" ] || continue
  n=$(basename "$d")
  protected "$n" && { say "keep (protected) $n"; continue; }
  if [ -n "$(find "$d" -maxdepth 2 -mtime -"$DAYS" -print -quit 2>/dev/null)" ]; then say "keep (recent) $n"; continue; fi
  say "remove $n ($(du -sk "$d" 2>/dev/null | cut -f1) k)"; act rm -rf -- "$d"
done

# 2. shared cargo targets: strip incremental + debug leftovers (regenerable), keep release outputs
for t in "$H"/cargo-targets/*; do
  [ -d "$t" ] || continue
  for sub in incremental debug/incremental; do [ -d "$t/$sub" ] && { say "clear $t/$sub"; act rm -rf -- "$t/$sub"; }; done
done

# 3. container images no longer referenced (k3s); safe: only unused
if command -v k3s >/dev/null 2>&1; then say "k3s unused images"; act k3s crictl rmi --prune; fi

# 4. journals and old rotated logs
command -v journalctl >/dev/null 2>&1 && act journalctl --vacuum-size=500M
find "$H/.local/state" -maxdepth 3 -name '*.log' -size +200M -mtime +7 -print 2>/dev/null | while read -r f; do say "truncate big old log $f"; act truncate -s 0 "$f"; done

# 5. hand freed blocks back to NTFS (sparse VHDX): without this D: does not grow.
# DISABLED 2026-10-14: a 298 GiB fstrim of this disk (after rm -rf of 83 GB of caches) was followed by a boot with I/O errors and
# new ext4 corruption (invalid extent nodes, 5 directories lost; e2fsck -fy repaired it). Cause unproven; do not re-enable
# until a trim is tested on a copy or in small steps. Set PRUNE_TRIM=1 to run it deliberately.
if [ "${PRUNE_TRIM:-0}" = 1 ]; then act fstrim -v /; else say "fstrim skipped (disabled, see comment)"; fi
after=$(df -k / | awk 'NR==2{print $4}')
say "done free_k=$after delta_k=$((after - before))"
