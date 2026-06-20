#!/usr/bin/env bash
# Cron entry point for the deep-think-loci harvest loop (rebuild dataset +
# retrain the bleed-detector + report the swap signal). Referenced by the
# disabled-by-default "deep-think-loci-harvest" job in cron/jobs.json — enable
# it deliberately, since it retrains a model from the accumulated corpus.
#
# Resolves the package harvest.sh from the runtime install first, then the repo.
set -euo pipefail
export HERMES_PY="${HERMES_PY:-$HOME/.hermes/hermes-agent/venv/bin/python3}"
for H in \
  "$HOME/.hermes/specialists/grounding/harvest.sh" \
  "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../deep_think_loci/grounding/harvest.sh" \
  "$HOME/development/loci/deep_think_loci/grounding/harvest.sh"; do
  if [ -f "$H" ]; then exec bash "$H"; fi
done
echo "[dtl-harvest] harvest.sh not found — run deep_think_loci/install.sh" >&2
exit 1
