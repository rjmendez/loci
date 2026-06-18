#!/bin/bash
# Mnemosyne consolidation — runs sleep cycle to promote working memory to episodic.
# Silent if nothing to consolidate (empty stdout = no cron notification).

VENV="$HOME/.hermes/hermes-agent/venv/bin/mnemosyne"
DB="$HOME/.hermes/mnemosyne/data/mnemosyne.db"

# Get working memory count before
before=$($VENV stats 2>/dev/null | grep -oP '"working_memory":\s*\K[0-9]+' | head -1)
[ -z "$before" ] && before=0

# Run sleep/consolidation
result=$($VENV sleep 2>/dev/null)

# Get count after
after=$($VENV stats 2>/dev/null | grep -oP '"working_memory":\s*\K[0-9]+' | head -1)
[ -z "$after" ] && after=0

consolidated=$(( before - after ))

# Only output if something actually happened (keeps cron silent when idle)
if [ "$consolidated" -gt 0 ]; then
  echo "mnemosyne: consolidated $consolidated working memories into episodic (${before} -> ${after})"
fi
