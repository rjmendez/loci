#!/usr/bin/env bash
set -euo pipefail

LOCI_PY="${LOCI_PY:-${HOME}/.hermes/hermes-agent/venv/bin/python3}"
EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"

"$LOCI_PY" "$EVAL_DIR/harness.py" "$@"
"$LOCI_PY" "$EVAL_DIR/grounding_gate_eval.py" "$@"
"$LOCI_PY" "$EVAL_DIR/grounding_gate_qf_eval.py" "$@"
