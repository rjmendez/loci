"""mlops/loop.py — self-closing MLOps loop for Loci grounding gate and fine-tuning.

Cycle:
  1. Discover new investigation runs since last loop tick
  2. Rebuild grounding dataset with new findings (appends pairs, preserves old)
  3. Retrain classifier ensemble (mlops/grounding/train.py logic)
  4. Canary evaluation — auto-promote if candidate beats cosine baseline
  5. SFT data collection + Ollama model bake (on separate cadence)
  6. Embedding fine-tune trigger (weekly, emits run_contrastive.sh)

State persisted in mlops/loop_state.json. All decisions logged to mlops/loop_history.jsonl.
Designed to run as a cron/systemd timer locally (needs Ollama) or via GitHub Actions
with a self-hosted runner. Gracefully skips embedding steps when Ollama is unreachable.
"""

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).parent.parent
MLOPS = REPO / "mlops"
GROUNDING_DIR = REPO / "deep_think_loci" / "grounding"
STATE_FILE = MLOPS / "loop_state.json"
HISTORY_FILE = MLOPS / "loop_history.jsonl"
CANDIDATE_MODEL = MLOPS / "grounding" / "candidate.joblib"
LIVE_MODEL = GROUNDING_DIR / "grounding_bleed_clf.joblib"
DATASET = GROUNDING_DIR / "grounding_dataset.jsonl"

DEFAULT_OLLAMA = (os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
DEFAULT_FINDINGS = os.path.expanduser("~/.hermes/memory-sessions/dt-loci-*/findings.jsonl")


# ── State I/O ─────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {
        "last_run": None,
        "last_dataset_size": 0,
        "runs_seen": [],
        "last_sft_bake": None,
        "last_embedding_tune": None,
        "total_promotions": 0,
    }


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _append_history(record: dict) -> None:
    with HISTORY_FILE.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


# ── Ollama probe ───────────────────────────────────────────────────────────────

def _ollama_ok(base: str) -> bool:
    try:
        urllib.request.urlopen(f"{base}/api/tags", timeout=5)
        return True
    except Exception:
        return False


# ── New-run discovery ─────────────────────────────────────────────────────────

def _discover_runs(findings_glob: str, seen: list[str]) -> list[str]:
    seen_set = set(seen)
    new = []
    for path in glob.glob(findings_glob):
        run_id = Path(path).parent.name
        if run_id not in seen_set:
            new.append(run_id)
    return sorted(new)


# ── Dataset rebuild ───────────────────────────────────────────────────────────

def _rebuild_dataset(findings_glob: str, ollama: str) -> int:
    """Run build_grounding_dataset.py to refresh the dataset. Returns new pair count."""
    builder = REPO / "deep_think_loci" / "grounding" / "build_grounding_dataset.py"
    if not builder.exists():
        print("[loop] build_grounding_dataset.py not found — skipping rebuild")
        return _current_dataset_size()

    result = subprocess.run(
        [sys.executable, str(builder),
         "--findings", findings_glob,
         "--out", str(GROUNDING_DIR),
         "--ollama", f"{ollama}/v1/embeddings"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[loop] dataset rebuild failed:\n{result.stderr[-500:]}")
        return _current_dataset_size()

    size = _current_dataset_size()
    print(f"[loop] dataset rebuilt → {size} pairs")
    return size


def _current_dataset_size() -> int:
    if not DATASET.exists():
        return 0
    return sum(1 for _ in DATASET.open())


# ── Grounding gate retrain ────────────────────────────────────────────────────

def _retrain(findings_glob: str, ollama: str, dry_run: bool) -> dict | None:
    """Run mlops/grounding/train.py. Returns metrics dict or None on failure."""
    metrics_path = MLOPS / "grounding" / "train_metrics.json"
    cmd = [
        sys.executable, str(MLOPS / "grounding" / "train.py"),
        "--dataset", str(DATASET),
        "--out", str(metrics_path),
        "--ollama", ollama,
        "--candidate-out", str(CANDIDATE_MODEL),
    ]
    if findings_glob:
        cmd += ["--findings-glob", findings_glob]
    if dry_run:
        cmd.append("--dry-run")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-1000:])
    if result.returncode != 0:
        print(f"[loop] train.py failed:\n{result.stderr[-500:]}")
        return None

    if metrics_path.exists():
        return json.loads(metrics_path.read_text())
    return None


# ── Canary evaluation ─────────────────────────────────────────────────────────

def _run_canary(findings_glob: str, ollama: str, dry_run: bool) -> dict | None:
    if not CANDIDATE_MODEL.exists():
        print("[loop] no candidate model to evaluate")
        return None

    cmd = [
        sys.executable, str(MLOPS / "grounding" / "canary.py"),
        "--candidate", str(CANDIDATE_MODEL),
        "--target", str(LIVE_MODEL),
        "--findings", findings_glob,
        "--ollama", ollama,
    ]
    if dry_run:
        cmd.append("--dry-run")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-1000:])
    if result.returncode == 1:
        print("[loop] ALERT: canary drift detected")
    return {"exit_code": result.returncode, "stdout": result.stdout[-500:]}


# ── SFT bake ─────────────────────────────────────────────────────────────────

def _run_sft_bake(ollama: str, dry_run: bool) -> bool:
    collect_out = MLOPS / "finetune" / "data"
    collect_out.mkdir(parents=True, exist_ok=True)
    traces = collect_out / "raw_traces.jsonl"
    sft = collect_out / "sft_pairs.jsonl"

    for cmd in [
        [sys.executable, str(MLOPS / "finetune" / "collect.py"),
         "--out", str(collect_out), "--ollama", ollama],
        [sys.executable, str(MLOPS / "finetune" / "format_sft.py"),
         "--traces", str(traces), "--out", str(sft)],
    ]:
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout[-500:])
        if r.returncode != 0:
            print(f"[loop] SFT step failed: {r.stderr[-300:]}")
            return False

    if not sft.exists() or sft.stat().st_size < 100:
        print("[loop] SFT pairs file empty — skipping bake")
        return False

    if not dry_run:
        bake_cmd = [
            sys.executable, str(MLOPS / "finetune" / "train_lora.py"),
            "--sft", str(sft), "--backend", "ollama-modelfile",
        ]
        r = subprocess.run(bake_cmd, capture_output=True, text=True)
        print(r.stdout[-500:])
        return r.returncode == 0
    return True


# ── Embedding tune trigger ────────────────────────────────────────────────────

def _emit_embedding_trigger() -> None:
    """Emit a shell script to run contrastive fine-tuning. Doesn't run it inline
    since it may need GPU and takes ~20 min even on CPU."""
    script = MLOPS / "run_contrastive.sh"
    script.write_text(
        "#!/bin/bash\n"
        "# Auto-generated by mlops/loop.py — run when GPU is available\n"
        f"set -e\n"
        f"cd {REPO}\n"
        f"python3 mlops/embedding/contrastive.py \\\n"
        f"  --dataset deep_think_loci/grounding/grounding_dataset.jsonl \\\n"
        f"  --model-size small \\\n"
        f"  --out mlops/embedding/\n"
        f"echo 'Done. Load mlops/embedding/loci-embed-small/ as your embedding model.'\n"
    )
    script.chmod(0o755)
    print(f"[loop] embedding trigger written to {script}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Loci MLOps self-closing loop")
    ap.add_argument("--findings", default=DEFAULT_FINDINGS,
                    help="Glob for investigation findings.jsonl files")
    ap.add_argument("--ollama", default=DEFAULT_OLLAMA)
    ap.add_argument("--min-new-runs", type=int, default=2,
                    help="Minimum new investigation runs before retraining (default: 2)")
    ap.add_argument("--min-new-pairs", type=int, default=200,
                    help="Minimum new dataset pairs before retraining (default: 200)")
    ap.add_argument("--sft-every", type=int, default=7,
                    help="SFT bake cadence in days (default: 7)")
    ap.add_argument("--embedding-every", type=int, default=30,
                    help="Embedding fine-tune trigger cadence in days (default: 30)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan only — no model writes, no Ollama model creates")
    ap.add_argument("--force", action="store_true",
                    help="Skip new-data thresholds and retrain unconditionally")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    state = _load_state()

    print(f"[loop] starting at {now_iso}")
    print(f"[loop] state: last_run={state['last_run']} dataset={state['last_dataset_size']} promotions={state['total_promotions']}")

    # ── 1. Ollama probe ────────────────────────────────────────────────────────
    ollama_ok = _ollama_ok(args.ollama)
    if not ollama_ok:
        print(f"[loop] Ollama not reachable at {args.ollama} — embedding/training steps will be skipped")

    # ── 2. New-run discovery ───────────────────────────────────────────────────
    new_runs = _discover_runs(args.findings, state["runs_seen"])
    print(f"[loop] new investigation runs: {len(new_runs)} {new_runs}")

    # ── 3. Decide whether to retrain ──────────────────────────────────────────
    current_size = _current_dataset_size()
    new_pairs = current_size - state["last_dataset_size"]

    should_retrain = args.force or (
        ollama_ok and
        len(new_runs) >= args.min_new_runs and
        new_pairs >= args.min_new_pairs
    )
    print(f"[loop] dataset pairs: {current_size} (+{new_pairs} new) | retrain={should_retrain}")

    promoted = False
    train_metrics = None

    if should_retrain:
        # ── 4. Rebuild dataset ────────────────────────────────────────────────
        new_size = _rebuild_dataset(args.findings, args.ollama)
        new_pairs = new_size - state["last_dataset_size"]
        print(f"[loop] dataset after rebuild: {new_size} pairs (+{new_pairs})")

        # ── 5. Retrain ────────────────────────────────────────────────────────
        train_metrics = _retrain(args.findings, args.ollama, args.dry_run)
        if train_metrics:
            decision = train_metrics.get("decision", "HOLD")
            print(f"[loop] train decision: {decision}  model={train_metrics.get('model')}  "
                  f"cv_f1={train_metrics.get('cv_f1_mean', 0):.3f}  "
                  f"baseline_f1={train_metrics.get('cosine_baseline_cv_f1', 0):.3f}")

            # ── 6. Canary ─────────────────────────────────────────────────────
            if decision == "PROMOTE":
                canary = _run_canary(args.findings, args.ollama, args.dry_run)
                if canary and canary.get("exit_code", 1) == 0:
                    promoted = True
                    state["total_promotions"] = state.get("total_promotions", 0) + 1
                    print(f"[loop] PROMOTED — total promotions: {state['total_promotions']}")
                else:
                    print("[loop] canary held back or drift detected — keeping current model")

        state["last_dataset_size"] = new_size
        state["runs_seen"] = state["runs_seen"] + new_runs

    # ── 7. SFT bake (cadence-gated) ───────────────────────────────────────────
    last_sft = state.get("last_sft_bake")
    sft_days_ago = (
        (now - datetime.fromisoformat(last_sft)).days if last_sft else 999
    )
    if ollama_ok and sft_days_ago >= args.sft_every:
        print(f"[loop] SFT bake (last was {sft_days_ago}d ago)")
        ok = _run_sft_bake(args.ollama, args.dry_run)
        if ok and not args.dry_run:
            state["last_sft_bake"] = now_iso
    else:
        print(f"[loop] SFT bake skipped (last was {sft_days_ago}d ago, cadence={args.sft_every}d)")

    # ── 8. Embedding trigger (cadence-gated) ──────────────────────────────────
    last_emb = state.get("last_embedding_tune")
    emb_days_ago = (
        (now - datetime.fromisoformat(last_emb)).days if last_emb else 999
    )
    if emb_days_ago >= args.embedding_every:
        print(f"[loop] embedding trigger (last was {emb_days_ago}d ago)")
        _emit_embedding_trigger()
        if not args.dry_run:
            state["last_embedding_tune"] = now_iso
    else:
        print(f"[loop] embedding trigger skipped ({emb_days_ago}d ago, cadence={args.embedding_every}d)")

    # ── 9. Persist state + history ────────────────────────────────────────────
    state["last_run"] = now_iso
    if not args.dry_run:
        _save_state(state)

    _append_history({
        "run_at": now_iso,
        "new_runs": len(new_runs),
        "dataset_size": current_size,
        "retrained": should_retrain,
        "promoted": promoted,
        "train_metrics": train_metrics,
        "dry_run": args.dry_run,
    })

    print(f"[loop] done. promoted={promoted} dataset={state['last_dataset_size']} total_promotions={state['total_promotions']}")


if __name__ == "__main__":
    main()
