#!/usr/bin/env bash
# open-design MCP server wrapper
# Proxies stdio to the od CLI inside the running open-design Docker container.
# Requires OD_API_TOKEN to be set in the environment.
set -euo pipefail
: "${OD_API_TOKEN:?OD_API_TOKEN must be set in environment}"
exec docker exec -i \
  -e OD_DAEMON_URL=http://localhost:7456 \
  -e OD_API_TOKEN="${OD_API_TOKEN}" \
  open-design \
  node /app/apps/daemon/dist/cli.js mcp
