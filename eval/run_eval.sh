#!/usr/bin/env bash
set -euo pipefail

HERMES_PY="${HERMES_PY:-${HOME}/.hermes/hermes-agent/venv/bin/python3}"
EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"

"$HERMES_PY" "$EVAL_DIR/harness.py" "$@"
"$HERMES_PY" "$EVAL_DIR/grounding_gate_eval.py" "$@"
"$HERMES_PY" "$EVAL_DIR/grounding_gate_qf_eval.py" "$@"
