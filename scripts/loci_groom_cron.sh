#!/usr/bin/env bash
# Cron entrypoint for the passive grooming tier.
#
# Not scheduled into cron/jobs.json: Hermes jobs use hermes_cron_runner.py on
# their own minute tick; the grooming tier still lives directly on the user
# crontab.
#
# Exit codes come from loci_groom.py and are the point of this wrapper:
#   0  ok
#   1  a pass errored
#   3  a pass refused or degraded — most importantly, refused because the
#      retention window is non-zero and connecting would delete findings
#
# A 3 means the pass refused or degraded rather than ran unsafely. Grooming
# re-indexes findings; if the server is configured to purge them, an
# unattended groomer would otherwise turn into an index-then-delete loop
# that burns embedding compute while reporting success.
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
# One line per run in runs.jsonl records whether this actually ran and what
# it said; absence of a recent entry is itself the signal.
printf '{"ts":%d,"pass":"%s","rc":%d,"summary":%s}\n' \
    "$(date +%s)" "$pass" "$rc" \
    "$(printf '%s' "$out" | tail -1 | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')" \
    >> "$STATE/runs.jsonl"

if [ "$rc" -eq 3 ]; then
    # Print the pass's OWN reason — the retention diagnosis only applies to
    # `index`, so each pass must report its own failure reason rather than a
    # fixed explanation for every pass.
    echo "loci-groom: pass '$pass' REFUSED or DEGRADED — not groomed." >&2
    printf '%s\n' "$out" | grep -iE 'refused|degraded|detail=' | tail -2 >&2
fi
exit "$rc"
