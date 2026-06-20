#!/usr/bin/env bash
# Harvest loop — rebuild the grounding dataset + retrain the bleed-detector from
# ALL accumulated deep-think-loci runs, then report the swap-in signal.
#
# Each engine run mints more labeled (claim, evidence) pairs as a byproduct; this
# folds them back in so the specialist improves automatically. Run after a batch
# of runs, or on a schedule (cron). When the trained model's CV accuracy beats the
# cosine baseline, swap it in as the gate default (ground_gate.py --model ...).
#
# Env: HERMES_PY (python with numpy/scikit-learn), DTL_CORPUS_GLOB, OLLAMA_BASE_URL.
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # deep_think_loci/grounding
REPO="$(cd "$SRC/../.." && pwd)"
PY="${HERMES_PY:-python3}"
CORPUS="${DTL_CORPUS_GLOB:-$HOME/.hermes/memory-sessions/dt-loci-*/findings.jsonl}"

shopt -s nullglob
matches=( $CORPUS )
if [ ${#matches[@]} -eq 0 ]; then
  echo "[harvest] no corpus found at: $CORPUS — run the engine first." >&2
  exit 1
fi
echo "[harvest] rebuilding dataset from ${#matches[@]} run(s)"
"$PY" "$SRC/build_grounding_dataset.py" --findings $CORPUS --out "$SRC"

echo "[harvest] eval (dry-run, freshly built cosines):"
HARNESS_DRY_RUN=1 "$PY" "$REPO/eval/grounding_gate_eval.py" || true

echo "[harvest] swap-in check:"
"$PY" - "$SRC/metrics.json" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
acc, base = m.get("cv_acc"), m.get("cosine_baseline_acc")
print(f"  total={m.get('total')} cv_acc={acc} cv_auc={m.get('cv_auc')} cosine_baseline_acc={base}")
if acc and base and acc > base:
    print("  -> trained model beats cosine ON THE PAIR TASK (finding<->finding).")
    print("     CAVEAT: the live gate operates query->finding; validate on a query->finding")
    print("     eval before swapping the default. Enable explicitly: ground_gate.py --model grounding_bleed_clf.joblib")
else:
    print("  -> cosine threshold still competitive; keep cosine default (accumulate more runs)")
PYEOF
echo "[harvest] done — commit deep_think_loci/grounding/{grounding_dataset.jsonl,grounding_bleed_clf.joblib,metrics.json} if the dataset grew."
