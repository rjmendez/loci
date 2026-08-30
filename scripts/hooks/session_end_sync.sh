#!/usr/bin/env bash
# Stop hook: sync the finished session into Qdrant.
#
# Endpoints and credentials come from the Hermes profile .env rather than being
# duplicated here. Only the keys actually needed are read, and no eval is used,
# because one profile value is JSON.
PROFILE_ENV="$HOME/.hermes/profiles/${HERMES_PROFILE:-mrpink}/.env"
if [[ -r "$PROFILE_ENV" ]]; then
  while IFS='=' read -r k v; do
    case "$k" in
      QDRANT_URL|QDRANT_API_KEY|MNEMOSYNE_EMBEDDING_API_URL|MNEMOSYNE_EMBEDDING_MODEL|MNEMOSYNE_EMBEDDING_DIM)
        export "$k=$v" ;;
    esac
  done < "$PROFILE_ENV"
fi

export LOCI_STATE_DB="${LOCI_STATE_DB:-$HOME/.hermes/profiles/${HERMES_PROFILE:-mrpink}/state.db}"
export LOCI_SYNC_CACHE="${LOCI_SYNC_CACHE:-$HOME/.hermes/profiles/${HERMES_PROFILE:-mrpink}/.session_sync_cache}"
export HERMES_AGENT_ID="${HERMES_AGENT_ID:-mrpink}"
export HERMES_PROFILE="${HERMES_PROFILE:-mrpink}"
export MNEMOSYNE_EMBEDDING_MODEL="${MNEMOSYNE_EMBEDDING_MODEL:-nomic-embed-text}"
export MNEMOSYNE_EMBEDDING_DIM="${MNEMOSYNE_EMBEDDING_DIM:-768}"
exec python3 "$(dirname "$0")/session_end_sync.py"
