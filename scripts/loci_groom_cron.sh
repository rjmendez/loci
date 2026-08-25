#!/usr/bin/env bash
# Cron entrypoint for the passive grooming tier.
#
# Not scheduled into cron/jobs.json: that file lists five jobs marked
# "enabled": true, every one with last_run_at: None, and nothing on this host
# reads it (see issue #205). The live substrate is the user crontab.
#
# Exit codes come from loci_groom.py and are the point of this wrapper:
#   0  ok
#   1  a pass errored
#   3  a pass refused or degraded — most importantly, refused because the
#      retention window is non-zero and connecting would delete findings
#
# A 3 is not a warning to bury in a log. Grooming re-indexes findings; if the
# server is configured to purge them, an unattended groomer becomes an infinite
# index-then-delete loop that burns embedding compute and reports success.
set -uo pipefail

REPO="${LOCI_REPO:-/home/rjmendez/development/loci}"
PY="$REPO/mcp/.venv/bin/python"
STATE="${LOCI_GROOM_STATE:-$HOME/.loci/groom}"
mkdir -p "$STATE"

[ -x "$PY" ] || { echo "loci-groom: no interpreter at $PY" >&2; exit 1; }

pass="${1:?usage: loci_groom_cron.sh <pass> [args...]}"
shift || true

out="$("$PY" "$REPO/scripts/loci_groom.py" "$pass" "$@" 2>&1)"
rc=$?

printf '%s\n' "$out"
# One line per run, so "has this ever actually run, and what did it say" is a
# question with an answer. The absence of a recent entry is itself the signal.
printf '{"ts":%d,"pass":"%s","rc":%d,"summary":%s}\n' \
    "$(date +%s)" "$pass" "$rc" \
    "$(printf '%s' "$out" | tail -1 | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')" \
    >> "$STATE/runs.jsonl"

if [ "$rc" -eq 3 ]; then
    # Print the pass's OWN reason. This used to state the retention diagnosis
    # unconditionally, which is right for `index` and wrong for every other
    # pass — a fixed explanation for a variable failure is how you get told the
    # wrong cause with total confidence.
    echo "loci-groom: pass '$pass' REFUSED or DEGRADED — not groomed." >&2
    printf '%s\n' "$out" | grep -iE 'refused|degraded|detail=' | tail -2 >&2
fi
exit "$rc"
