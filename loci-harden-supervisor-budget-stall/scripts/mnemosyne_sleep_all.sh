#!/bin/bash
# Mnemosyne sleep across all banks via CLI.
# Silent if nothing to consolidate. Outputs a line only when episodic memories are written.
# Uses MNEMOSYNE_DATA_DIR override to target each bank's separate DB.

export MNEMOSYNE_LLM_BASE_URL="http://localhost:11434/v1"
export MNEMOSYNE_LLM_MODEL="${MNEMOSYNE_LLM_MODEL:-llama3.2:latest}"
MNEM="$HOME/.hermes/hermes-agent/venv/bin/mnemosyne"
BASE="$HOME/.hermes/mnemosyne/data"

declare -A BANK_PATHS
BANK_PATHS["default"]="$BASE"
BANK_PATHS["dama-gotchi"]="$BASE/banks/dama-gotchi"
BANK_PATHS["deep_think"]="$BASE/banks/deep_think"

for bank in default dama-gotchi deep_think; do
    db_dir="${BANK_PATHS[$bank]}"
    if [ ! -f "$db_dir/mnemosyne.db" ]; then
        continue
    fi
    out=$(MNEMOSYNE_DATA_DIR="$db_dir" "$MNEM" sleep 2>&1)
    if echo "$out" | grep -qiE 'summaries_created.: [1-9]|episodic.*[1-9]'; then
        echo "[$(date +%H:%M)] sleep $bank: $out"
    fi
done
