#!/usr/bin/env bash
# Stop hook: synthesize durable learning from a completed session.
#
# Secrets and endpoint overrides are loaded from the active Hermes profile .env
# instead of being committed into this wrapper.
PROFILE_ENV="$HOME/.hermes/profiles/${HERMES_PROFILE:-mrpink}/.env"
if [[ -r "$PROFILE_ENV" ]]; then
  while IFS='=' read -r k v; do
    case "$k" in
      MRPINK_SEARCH_TOKEN|SEARCH_BASE_URL|OLLAMA_BASE_URL|LEARN_MODEL|MNEMOSYNE_DB|LEARN_CACHE_DIR|LEARN_MIN_MSGS|LEARN_MIN_CHARS|LEARN_MIN_SCORE|LEARN_TOP_K)
        export "$k=$v" ;;
    esac
  done < "$PROFILE_ENV"
fi

export LOCI_STATE_DB="${LOCI_STATE_DB:-$HOME/.hermes/profiles/${HERMES_PROFILE:-mrpink}/state.db}"
export HERMES_STATE_DB="${HERMES_STATE_DB:-$LOCI_STATE_DB}"
export SEARCH_BASE_URL="${SEARCH_BASE_URL:-http://127.0.0.1:8201}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://100.73.200.19:11434}"
export LEARN_MODEL="${LEARN_MODEL:-qwen3-4b-instruct-heretic-agent}"
export MNEMOSYNE_DB="${MNEMOSYNE_DB:-$HOME/.hermes/profiles/${HERMES_PROFILE:-mrpink}/mnemosyne/data/mnemosyne.db}"
export LEARN_CACHE_DIR="${LEARN_CACHE_DIR:-$HOME/.hermes/profiles/${HERMES_PROFILE:-mrpink}/.learn_cache}"
export LEARN_MIN_MSGS="${LEARN_MIN_MSGS:-4}"
export LEARN_MIN_CHARS="${LEARN_MIN_CHARS:-400}"
export LEARN_MIN_SCORE="${LEARN_MIN_SCORE:-0.6}"
export LEARN_TOP_K="${LEARN_TOP_K:-3}"
exec python3 "$(dirname "$0")/session_end_learn.py"
