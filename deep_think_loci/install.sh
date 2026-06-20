#!/usr/bin/env bash
# Deploy the deep-think-loci engine from this repo to the ~/.hermes runtime
# locations the workflow defaults to. Idempotent. Re-run after pulling updates.
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WF_DST="${HERMES_HOME:-$HOME/.hermes}/workflows"
GATE_DST="${HERMES_HOME:-$HOME/.hermes}/specialists/grounding"
mkdir -p "$WF_DST" "$GATE_DST"
cp "$SRC/workflows/deep-think-loci.js"            "$WF_DST/deep-think-loci-v3.js"
cp "$SRC/grounding/ground_gate.py"                "$GATE_DST/"
cp "$SRC/grounding/build_grounding_dataset.py"    "$GATE_DST/"
cp "$SRC/grounding/harvest.sh"                     "$GATE_DST/"
cp "$SRC/grounding/grounding_bleed_clf.joblib"    "$GATE_DST/"
cp "$SRC/grounding/grounding_dataset.jsonl"       "$GATE_DST/"
cp "$SRC/grounding/metrics.json"                  "$GATE_DST/"
chmod +x "$GATE_DST/ground_gate.py" "$GATE_DST/harvest.sh"
echo "deployed:"
echo "  workflow -> $WF_DST/deep-think-loci-v3.js"
echo "  gate     -> $GATE_DST/ground_gate.py"
echo "run via:  Workflow({ scriptPath: \"$WF_DST/deep-think-loci-v3.js\" })"
