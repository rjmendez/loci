#!/usr/bin/env bash
# Cron entrypoint for the curated memory index.
#
# The index overflowed its context-load cap twice, the second time within four
# days of a hand trim. generate_memory_index.py enforces the budget, but a
# generator nobody runs is the same failure with extra steps — so this is the
# part that makes the guarantee continuous rather than per-invocation.
#
# Not scheduled via cron/jobs.json: those jobs have never fired (issue #205).
# The live substrate is the user crontab, same as the grooming tier.
#
# Exit codes:
#   0  index regenerated (and committed, if it changed)
#   1  the generator errored
#   3  refused or degraded — the generator or the memory dir is unreachable
set -uo pipefail

REPO="${LOCI_REPO:-/home/rjmendez/development/loci}"
STATE="${MEMORY_INDEX_STATE:-$HOME/.loci/memory-index}"
PY="$REPO/mcp/.venv/bin/python"
[ -x "$PY" ] || PY=python3
mkdir -p "$STATE"

log_run() {  # ts, rc, summary — absence of a recent line is itself the signal
    printf '{"ts":%d,"rc":%d,"summary":%s}\n' "$(date +%s)" "$1" \
        "$(printf '%s' "$2" | tail -1 | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')" \
        >> "$STATE/runs.jsonl"
    # There is no MTA on this host, so a non-zero exit cannot reach anyone by
    # mail the way the grooming tier assumes. Record failures where one cheap
    # check answers "has this ever broken".
    [ "$1" -ne 0 ] && printf '%s rc=%d %s\n' "$(date -Is)" "$1" "$2" >> "$STATE/failures.log"
    return 0
}

GEN="$REPO/scripts/generate_memory_index.py"
TMPGEN=""
if [ ! -f "$GEN" ]; then
    # Until the PR lands the script is only on its branch. Prefer main's copy
    # the moment it exists; this fallback then stops being used on its own.
    TMPGEN="$(mktemp)"
    if git -C "$REPO" show "feat/memory-index-generator:scripts/generate_memory_index.py" \
            > "$TMPGEN" 2>/dev/null && [ -s "$TMPGEN" ]; then
        GEN="$TMPGEN"
    else
        rm -f "$TMPGEN"
        echo "memory-index: no generator at $REPO/scripts/generate_memory_index.py and no branch copy" >&2
        log_run 3 "generator missing"
        exit 3
    fi
fi

MEMDIR="${LOCI_MEMORY_MD_DIR:-$("$PY" -c "
import sys; sys.path.insert(0, '$REPO/mcp')
try:
    import backends; print(backends.memory_dir() or '')
except Exception: print('')
" 2>/dev/null)}"

if [ -z "$MEMDIR" ] || [ ! -d "$MEMDIR" ]; then
    echo "memory-index: memory dir unresolved or missing (got '${MEMDIR:-}')." >&2
    echo "memory-index: set [memory].dir in ~/.loci/backends.toml or LOCI_MEMORY_MD_DIR." >&2
    log_run 3 "memory dir unresolved"
    [ -n "$TMPGEN" ] && rm -f "$TMPGEN"
    exit 3
fi

out="$("$PY" "$GEN" "$MEMDIR" 2>&1)"
rc=$?
printf '%s\n' "$out"
[ -n "$TMPGEN" ] && rm -f "$TMPGEN"

# git is the backup, so the generator's own .bak copy is pure accumulation.
if git -C "$MEMDIR" rev-parse --git-dir >/dev/null 2>&1; then
    rm -f "$MEMDIR"/MEMORY.md.bak-*
    if ! git -C "$MEMDIR" diff --quiet -- MEMORY.md 2>/dev/null; then
        git -C "$MEMDIR" add MEMORY.md \
            && git -C "$MEMDIR" commit -q -m "Regenerate the memory index" \
            && echo "memory-index: committed a changed index"
    fi
fi

log_run "$rc" "$out"
if [ "$rc" -ne 0 ]; then
    echo "memory-index: generator exited $rc — index NOT updated." >&2
    printf '%s\n' "$out" | tail -3 >&2
    [ "$rc" -eq 3 ] && exit 3
    exit 1
fi
exit 0
