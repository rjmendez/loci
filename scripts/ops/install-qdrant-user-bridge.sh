#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
USER_SYSTEMD_DIR="${HOME}/.config/systemd/user"
DROPIN_DIR="${USER_SYSTEMD_DIR}/loci-mcp.service.d"
LOCAL_BIN_DIR="${HOME}/.local/bin"

install -d -m 700 "${LOCAL_BIN_DIR}" "${USER_SYSTEMD_DIR}" "${DROPIN_DIR}"

install -m 755 \
  "${ROOT_DIR}/scripts/ops/qdrant_local_bridge.py" \
  "${LOCAL_BIN_DIR}/qdrant_local_bridge.py"

install -m 644 \
  "${ROOT_DIR}/scripts/systemd/qdrant-pf.service" \
  "${USER_SYSTEMD_DIR}/qdrant-pf.service"

install -m 644 \
  "${ROOT_DIR}/scripts/systemd/loci-mcp-qdrant-pf.conf" \
  "${DROPIN_DIR}/d20-qdrant-pf.conf"

systemctl --user daemon-reload
systemctl --user enable --now qdrant-pf.service
systemctl --user restart loci-mcp.service

echo "qdrant-pf.service and loci-mcp dependency installed."
systemctl --user --no-pager --full status qdrant-pf.service | sed -n '1,20p'
echo "---"
systemctl --user --no-pager --full status loci-mcp.service | sed -n '1,20p'
