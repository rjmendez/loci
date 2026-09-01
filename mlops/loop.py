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
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    """Env-parsed int that refuses to crash the nightly at import. A typo in a
    cron line should not take the job down with a traceback and no context."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        print(f"[loop] {name}={raw!r} is not an integer — using {default}")
        return default
    if val <= 0:
        print(f"[loop] {name}={val} is not positive — using {default}")
        return default
    return val


STEP_TIMEOUT_S = _env_int("LOCI_MLOPS_STEP_TIMEOUT", 3600)


CHILD_ENV: dict[str, str] = {}


def _child_label(cmd) -> str:
    """The script being run, for prefixing its output in a merged log."""
    for part in cmd[1:]:
        if isinstance(part, str) and part.endswith(".py"):
            return Path(part).stem
    return Path(str(cmd[0])).stem


def _run(cmd, timeout: int | None = None, stream: bool = True):
    """Run a child with a bound, resolved backends, and its output echoed live.

    Three things a nightly needs and subprocess.run does not give together.

    A hung child stalls the whole run, so every call is bounded and a timeout
    comes back as returncode 124 rather than an exception.

    A child that cannot see OLLAMA_BASE_URL falls back to a localhost nothing
    listens on, so CHILD_ENV carries what backends resolved. It is handed over
    here rather than written into this process's own environment, so importing
    this module changes nothing for anyone else.

    And capture_output alone buffers everything until the child exits: train.py
    prints which model it is fitting, and none of that reached this log until the
    step was already over — 17 minutes of silence in the log cron actually mails
    you. Output is streamed as it arrives AND captured, so callers that parse
    result.stdout keep working. PYTHONUNBUFFERED is set because a child writing
    to a pipe block-buffers by default, which would make the streaming a no-op.
    """
    limit = timeout or STEP_TIMEOUT_S
    env = {**os.environ, **CHILD_ENV}
    env["PYTHONUNBUFFERED"] = "1"
    if not stream:
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=limit, env=env)
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout if isinstance(exc.stdout, str) else ""
            return subprocess.CompletedProcess(cmd, 124, out,
                                               f"timed out after {limit}s")

    label = _child_label(cmd)
    captured: dict[str, list[str]] = {"out": [], "err": []}

    def drain(pipe, key, echo):
        try:
            for line in pipe:
                captured[key].append(line)
                if echo:
                    print(f"  [{label}] {line.rstrip()}", flush=True)
        finally:
            pipe.close()

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, bufsize=1, env=env)
    threads = [
        threading.Thread(target=drain, args=(proc.stdout, "out", True), daemon=True),
        threading.Thread(target=drain, args=(proc.stderr, "err", False), daemon=True),
    ]
    for t in threads:
        t.start()
    try:
        code = proc.wait(timeout=limit)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        code = 124
    for t in threads:
        # No timeout. Returning while a drain thread still holds a pipe means log
        # lines land after the caller has moved on, interleaved into the next
        # step's output. The pipes are closed once the process is reaped, so
        # these threads end on their own.
        t.join()
    err = "".join(captured["err"])
    if code == 124:
        err = (err + f"\ntimed out after {limit}s").strip()
    return subprocess.CompletedProcess(cmd, code, "".join(captured["out"]), err)


REPO = Path(__file__).parent.parent
MLOPS = REPO / "mlops"

# Run as a script, sys.path[0] is mlops/, not the repo root, so `import mlops.*`
# fails. The monitor step did exactly that and had never once run.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Steps that failed outright, as opposed to steps deliberately skipped. A run
# that reports "done" after every step errored is the failure mode this loop
# spent 67 nights in.
FAILED_STEPS: list[str] = []


def _last_error_line(stderr: str) -> str:
    """The exception line out of a traceback. Slicing the last N chars of stderr
    lands mid-frame and prints a row of carets instead of the actual error."""
    lines = [ln.rstrip() for ln in (stderr or "").splitlines() if ln.strip()]
    if not lines:
        return "no stderr"
    for ln in reversed(lines):
        if not ln.lstrip().startswith(("File \"", "^", "~", "|")) and not ln.startswith("    "):
            return ln[:300]
    return lines[-1][:300]


def _fail(step: str, line: str) -> None:
    """Record a step as failed and print its message. The step name is the ledger
    key; the message stays whatever the call site already said."""
    FAILED_STEPS.append(step)
    print(f"[loop] {line}")


def _skip_reason(ollama_ok: bool, days_ago: int, cadence: int, ollama: str) -> str:
    """Why a cadence-gated step did not run. These gates are also Ollama-gated, so
    reciting only the cadence blames the schedule for an unreachable backend."""
    if not ollama_ok:
        return f"Ollama unreachable at {ollama}"
    return f"not due ({days_ago}d ago, cadence {cadence}d)"
GROUNDING_DIR = REPO / "deep_think_loci" / "grounding"
STATE_FILE = MLOPS / "loop_state.json"
HISTORY_FILE = MLOPS / "loop_history.jsonl"
CANDIDATE_MODEL = MLOPS / "grounding" / "candidate.joblib"
LIVE_MODEL = GROUNDING_DIR / "grounding_bleed_clf.joblib"
DATASET = GROUNDING_DIR / "grounding_dataset.jsonl"
ACTIVE_CANDIDATES = MLOPS / "grounding" / "active_candidates.jsonl"

def _resolve_backends() -> dict:
    """Fill OLLAMA_BASE_URL and friends from ~/.loci/backends.toml at run time.

    Nothing listens on localhost:11434 on a scheduled host — the endpoint is
    recorded in the config file that backends.py already reads, and until now
    only the MCP launcher's own process environment carried it. Without this the
    loop resolves the localhost default, every embedding step fails or skips, and
    the log blames the cadence.

    backends.load_env writes into os.environ, which is right for a real run and
    wrong for a test process, so the values are captured, this process is put
    back the way it was, and CHILD_ENV carries them to the children instead.
    """
    before = dict(os.environ)
    try:
        sys.path.insert(0, str(REPO / "mcp"))
        import backends
        resolved = backends.load_env(REPO)
    except Exception as exc:  # never block a run on config resolution
        print(f"[loop] backend resolution unavailable: {exc}")
        resolved = {}
    for key in resolved:
        if key in before:
            os.environ[key] = before[key]
        else:
            os.environ.pop(key, None)
    CHILD_ENV.clear()
    CHILD_ENV.update(resolved)
    return resolved


DEFAULT_OLLAMA = (os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
DEFAULT_FINDINGS = os.path.expanduser("~/.hermes/memory-sessions/dt-loci-*/findings.jsonl")
DEFAULT_DB = os.path.expanduser(
    os.environ.get("MNEMOSYNE_DB", "~/.hermes/mnemosyne/data/mnemosyne.db")
)
DEFAULT_HOOK_STATE = os.path.expanduser(
    os.environ.get("CLAUDE_HOOK_STATE", "~/.claude/hook-state")
)


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

    result = _run(
        [sys.executable, str(builder),
         "--findings", findings_glob,
         "--out", str(GROUNDING_DIR),
         "--ollama", f"{ollama}/v1/embeddings"],
    )
    if result.returncode != 0:
        _fail("dataset rebuild", f"dataset rebuild failed: {_last_error_line(result.stderr)}")
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

    result = _run(cmd)
    if result.returncode != 0:
        _fail("train.py", f"train.py failed: {_last_error_line(result.stderr)}")
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

    result = _run(cmd)
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
         "--traces", str(traces), "--out", str(sft), "--mode", "both"],
    ]:
        r = _run(cmd)
        if r.returncode != 0:
            _fail("SFT bake", f"SFT step failed: {_last_error_line(r.stderr)}")
            return False

    if not sft.exists() or sft.stat().st_size < 100:
        print("[loop] SFT pairs file empty — skipping bake")
        return False

    if not dry_run:
        bake_cmd = [
            sys.executable, str(MLOPS / "finetune" / "train_lora.py"),
            "--sft", str(sft), "--backend", "ollama-modelfile",
        ]
        r = _run(bake_cmd)
        return r.returncode == 0
    return True


# ── Weibull memory decay ──────────────────────────────────────────────────────

def _run_decay(db_path: str, dry_run: bool) -> dict:
    try:
        sys.path.insert(0, str(MLOPS))
        from memory.decay import apply_decay
        stats = apply_decay(db_path=db_path, dry_run=dry_run)
        print(f"[loop] decay: n_rows={stats.get('n_rows')} n_decayed={stats.get('n_decayed')} "
              f"mean_retention={stats.get('mean_retention', 0):.3f}"
              f"{' (dry run)' if dry_run else ''}")
        before = stats.get("n_grounding_visible_before")
        after = stats.get("n_grounding_visible_after")
        if before is not None:
            print(f"[loop] decay: rows visible to the grounding hook "
                  f"(importance >= {stats.get('grounding_min_importance')}): "
                  f"{before} -> {after}")
        return stats
    except Exception as exc:
        _fail("decay", f"decay step failed: {exc}")
        return {}


# ── Live-Evo memory adaptation ────────────────────────────────────────────────

def _run_live_evo(db_path: str, hook_state: str, dry_run: bool) -> dict:
    try:
        from memory.live_evo import adapt
        stats = adapt(db_path=db_path, hook_state_dir=hook_state, dry_run=dry_run)
        print(f"[loop] live_evo: failures={stats.get('n_failures')} "
              f"correlated={stats.get('n_correlated')} penalized={stats.get('n_penalized')}")
        return stats
    except Exception as exc:
        _fail("live_evo", f"live_evo step failed: {exc}")
        return {}


# ── Post-promotion monitoring ─────────────────────────────────────────────────

def _run_monitor(findings_glob: str, ollama: str, dry_run: bool) -> dict:
    if not LIVE_MODEL.exists():
        print("[loop] monitor skipped — no live model yet")
        return {}
    try:
        sys.path.insert(0, str(MLOPS / "grounding"))
        from mlops.grounding.canary import monitor_live
        result = monitor_live(
            live_model_path=str(LIVE_MODEL),
            findings_glob=findings_glob,
            ollama_url=ollama,
            dry_run=dry_run,
        )
        print(f"[loop] monitor: drift={result.get('drift')} "
              f"rollback_recommended={result.get('rollback_recommended')}")
        if result.get("rollback_recommended"):
            print("[loop] ALERT: rollback recommended — check monitor_history.jsonl")
        return result
    except Exception as exc:
        _fail("monitor", f"monitor step failed: {exc}")
        return {}


# ── Embedding drift detection ─────────────────────────────────────────────────

def _run_embedding_drift(ollama: str, dry_run: bool) -> dict:
    anchor = MLOPS / "embedding" / "anchor.npz"
    drift_script = MLOPS / "embedding" / "drift.py"
    if not drift_script.exists():
        return {}
    if not anchor.exists():
        print("[loop] embedding drift: no anchor — building anchor set ...")
        cmd = [sys.executable, str(drift_script),
               "--dataset", str(DATASET), "--ollama", ollama,
               "--anchor", str(anchor), "--build-anchor"]
        result = _run(cmd)
        return {"built_anchor": True}
    out_path = MLOPS / "embedding" / "drift_result.json"
    cmd = [sys.executable, str(drift_script),
           "--dataset", str(DATASET), "--ollama", ollama,
           "--anchor", str(anchor), "--out", str(out_path)]
    result = _run(cmd)
    if result.returncode == 1:
        print("[loop] ALERT: embedding drift detected — scheduling embedding fine-tune")
        if not dry_run:
            _emit_embedding_trigger()
    if out_path.exists():
        try:
            return json.loads(out_path.read_text())
        except Exception:
            pass
    return {"exit_code": result.returncode}


# ── Active learning ───────────────────────────────────────────────────────────

def _run_active_learn(ollama: str) -> dict:
    if not LIVE_MODEL.exists() or not DATASET.exists():
        return {}
    script = MLOPS / "grounding" / "active_learn.py"
    if not script.exists():
        return {}
    result = _run(
        [sys.executable, str(script),
         "--model", str(LIVE_MODEL),
         "--dataset", str(DATASET),
         "--out", str(ACTIVE_CANDIDATES),
         "--ollama", ollama],
    )
    if result.returncode != 0:
        # _run drains child stderr with echo=False, so without this a crash in
        # the sampler is a silent step that leaves the previous candidates file
        # in place. Same treatment the dataset-rebuild and train.py steps get.
        _fail("active_learn", f"active_learn failed: {_last_error_line(result.stderr)}")
    return {"exit_code": result.returncode}


# ── Embedding tune trigger ────────────────────────────────────────────────────

def _emit_embedding_trigger() -> None:
    """Emit a shell script to run contrastive fine-tuning. Doesn't run it inline
    since it may need GPU and takes ~20 min even on CPU.

    Rewritten only when the body differs. _embedding_tune_ran() dates a tune
    against this file's mtime, and touching it on every tick would make a
    finished fine-tune look stale forever.

    Runs the interpreter this loop is running under, not `python3`. contrastive.py
    imports sentence_transformers, which the system python3 does not have, so the
    emitted command died on import and the nag it produces was permanent.
    """
    script = MLOPS / "run_contrastive.sh"
    body = (
        "#!/bin/bash\n"
        "# Auto-generated by mlops/loop.py — run when GPU is available\n"
        f"set -e\n"
        f"cd {REPO}\n"
        f"{sys.executable} mlops/embedding/contrastive.py \\\n"
        f"  --dataset deep_think_loci/grounding/grounding_dataset.jsonl \\\n"
        f"  --model-size small \\\n"
        f"  --out mlops/embedding/\n"
        f"echo 'Done. Load mlops/embedding/loci-embed-small/ as your embedding model.'\n"
    )
    if not script.exists() or script.read_text() != body:
        script.write_text(body)
    script.chmod(0o755)
    print(f"[loop] embedding trigger written to {script}")


def _embedding_tune_ran(since_iso: str | None = None) -> bool:
    """True when a fine-tune newer than BOTH the trigger and the last recorded
    tune exists.

    The trigger is only ever a file: nothing in the repo runs
    run_contrastive.sh. Stamping last_embedding_tune on emission therefore
    bought 30 days of silence for work that had not happened, so the cadence is
    stamped against the fine-tune's own output directory instead.

    `since_iso` is the load-bearing half. The script's body is stable, so it is
    written once and its mtime freezes there; a tune done once is newer than that
    frozen mtime for the rest of time. Dating against the last recorded tune as
    well turns the 30-day cadence back into a cadence -- without it the second
    tick re-stamps last_embedding_tune for a tune that never happened, which is
    the same false stamp this function was written to remove.
    """
    script = MLOPS / "run_contrastive.sh"
    floor = script.stat().st_mtime if script.exists() else 0.0
    if since_iso:
        try:
            stamped = datetime.fromisoformat(since_iso)
        except ValueError:
            stamped = None
        if stamped is not None:
            # The loop writes an aware UTC stamp, but a hand-edited or older
            # state file may be naive, and .timestamp() would read that as local
            # time -- hours of skew against an mtime, silently.
            if stamped.tzinfo is None:
                stamped = stamped.replace(tzinfo=timezone.utc)
            floor = max(floor, stamped.timestamp())
    return any(d.is_dir() and d.stat().st_mtime > floor
               for d in (MLOPS / "embedding").glob("loci-embed-*"))


# ── Main loop ─────────────────────────────────────────────────────────────────

RUNS_SEEN_MAX = int(os.environ.get("LOCI_RUNS_SEEN_MAX", "1000"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Loci MLOps self-closing loop")
    ap.add_argument("--findings", default=DEFAULT_FINDINGS,
                    help="Glob for investigation findings.jsonl files")
    ap.add_argument("--ollama", default=None,
                    help="Ollama base URL. Default: $OLLAMA_BASE_URL, else ~/.loci/backends.toml, "
                         "else http://localhost:11434")
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
    ap.add_argument("--db", default=DEFAULT_DB, help="Path to Mnemosyne SQLite database")
    ap.add_argument("--hook-state", default=DEFAULT_HOOK_STATE,
                    help="Directory containing guard_bash_*.log files for Live-Evo")
    ap.add_argument("--decay-every", type=int, default=1,
                    help="Evaluate Weibull decay every N loop runs (default: every run)")
    ap.add_argument("--decay-apply", action="store_true",
                    help="Write the decayed importances back. Without this the decay step "
                         "reports what it would do and changes nothing -- decay has never "
                         "actually run against a real corpus, and the first write is large "
                         "and not something a nightly cron should do unasked.")
    ap.add_argument("--active-learn-every", type=int, default=7,
                    help="Generate active learning candidates every N days (default: 7)")
    args = ap.parse_args()

    FAILED_STEPS.clear()  # module-global; a second main() in one process must start clean
    resolved = _resolve_backends()
    if args.ollama is None:
        args.ollama = (os.environ.get("OLLAMA_BASE_URL")
                       or resolved.get("OLLAMA_BASE_URL")
                       or "http://localhost:11434").rstrip("/")

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    state = _load_state()

    print(f"[loop] starting at {now_iso}")
    if resolved:
        print("[loop] resolved from backends.toml: " + ", ".join(sorted(resolved)))
    print(f"[loop] state: last_run={state['last_run']} dataset={state['last_dataset_size']} promotions={state['total_promotions']}")

    # ── 1. Ollama probe ────────────────────────────────────────────────────────
    ollama_ok = _ollama_ok(args.ollama)
    if not ollama_ok:
        print(f"[loop] Ollama not reachable at {args.ollama} — embedding/training steps will be skipped")

    # ── 2. New-run discovery ───────────────────────────────────────────────────
    new_runs = _discover_runs(args.findings, state["runs_seen"])
    print(f"[loop] new investigation runs: {len(new_runs)} {new_runs}")

    # ── 3. Decide whether to rebuild ──────────────────────────────────────────
    #
    # Each threshold is applied where its input is actually knowable.
    # min_new_runs gates the REBUILD and is decidable now: discovery has just run.
    # min_new_pairs gates the TRAINING and is not — grounding_dataset.jsonl is an
    # OUTPUT of the rebuild, so its size here is last run's result. Under the cron
    # wrapper it is not even that: that wrapper resets its worktree to origin/main
    # nightly, restoring the committed 5,418-row file while loop_state.json outside
    # the worktree still holds the previous run's 12,684. The delta came out at
    # -7,266 and vetoed everything on the night three new investigations carrying
    # 1,139 findings had just been found.
    current_size = _current_dataset_size()
    stale_delta = current_size - state["last_dataset_size"]

    should_rebuild = args.force or (ollama_ok and len(new_runs) >= args.min_new_runs)
    why = ("forced" if args.force
           else f"Ollama unreachable at {args.ollama}" if not ollama_ok
           else f"{len(new_runs)} new runs >= {args.min_new_runs}" if should_rebuild
           else f"{len(new_runs)} new runs < {args.min_new_runs}")
    print(f"[loop] dataset on disk: {current_size} pairs "
          f"({stale_delta:+d} vs last run, pre-rebuild) | rebuild={should_rebuild} ({why})")

    promoted = False
    train_metrics = None

    should_retrain = should_rebuild
    if should_rebuild:
        # ── 4. Rebuild dataset ────────────────────────────────────────────────
        new_size = _rebuild_dataset(args.findings, args.ollama)
        new_pairs = new_size - state["last_dataset_size"]
        print(f"[loop] dataset after rebuild: {new_size} pairs ({new_pairs:+d})")

        # NOW the pair delta is real, so min_new_pairs can be applied to it.
        if not args.force and new_pairs < args.min_new_pairs:
            should_retrain = False
            print(f"[loop] rebuild produced {new_pairs:+d} pairs, under the "
                  f"{args.min_new_pairs} needed to justify training — skipping the "
                  "retrain. The dataset is still refreshed and the counts below "
                  "are current.")

        # ── 5. Retrain ────────────────────────────────────────────────────────
        train_metrics = _retrain(args.findings, args.ollama, args.dry_run) if should_retrain else None
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
        # Truncate: runs_seen only exists to skip runs already ingested, so the
        # tail is what matters. It appended forever and was rewritten in full on
        # every tick, so the state file grew without bound and got slower to
        # write as it went.
        state["runs_seen"] = (state["runs_seen"] + new_runs)[-RUNS_SEEN_MAX:]

    # ── 7a. Weibull memory decay (runs every loop tick) ──────────────────────
    loop_count = state.get("total_loop_runs", 0) + 1
    if loop_count % args.decay_every == 0:
        _run_decay(args.db, args.dry_run or not args.decay_apply)
    else:
        print(f"[loop] decay skipped (run {loop_count}, cadence={args.decay_every})")

    # ── 7b. Live-Evo memory adaptation ───────────────────────────────────────
    _run_live_evo(args.db, args.hook_state, args.dry_run)

    # ── 7c. Post-promotion online monitoring ──────────────────────────────────
    _run_monitor(args.findings, args.ollama, args.dry_run)

    # ── 7d. Embedding drift detection ─────────────────────────────────────────
    if ollama_ok:
        _run_embedding_drift(args.ollama, args.dry_run)

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
        print(f"[loop] SFT bake skipped — {_skip_reason(ollama_ok, sft_days_ago, args.sft_every, args.ollama)}")

    # ── 8. Embedding trigger (cadence-gated) ──────────────────────────────────
    last_emb = state.get("last_embedding_tune")
    emb_days_ago = (
        (now - datetime.fromisoformat(last_emb)).days if last_emb else 999
    )
    if emb_days_ago >= args.embedding_every:
        print(f"[loop] embedding trigger (last was {emb_days_ago}d ago)")
        _emit_embedding_trigger()
        if _embedding_tune_ran(last_emb):
            if not args.dry_run:
                state["last_embedding_tune"] = now_iso
        else:
            # An emitted script is not a completed tune, and nothing executes it
            # for us. Say so every run rather than sleeping on it for 30 days.
            print("[loop] embedding fine-tune pending — run: "
                  f"bash {MLOPS / 'run_contrastive.sh'}")
    else:
        print(f"[loop] embedding trigger skipped ({emb_days_ago}d ago, cadence={args.embedding_every}d)")

    # ── 8a. Active learning candidates (cadence-gated) ────────────────────────
    last_al = state.get("last_active_learn")
    al_days_ago = (now - datetime.fromisoformat(last_al)).days if last_al else 999
    if ollama_ok and al_days_ago >= args.active_learn_every:
        print(f"[loop] active_learn (last was {al_days_ago}d ago)")
        al_result = _run_active_learn(args.ollama)
        if al_result.get("exit_code", 1) == 0 and not args.dry_run:
            state["last_active_learn"] = now_iso
    else:
        print(f"[loop] active_learn skipped — {_skip_reason(ollama_ok, al_days_ago, args.active_learn_every, args.ollama)}")

    # ── 9. Persist state + history ────────────────────────────────────────────
    state["last_run"] = now_iso
    state["total_loop_runs"] = loop_count
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
        "failed_steps": list(FAILED_STEPS),
    })

    status = "done" if not FAILED_STEPS else f"done with {len(FAILED_STEPS)} failed step(s): " + ", ".join(FAILED_STEPS)
    print(f"[loop] {status}. promoted={promoted} dataset={state['last_dataset_size']} total_promotions={state['total_promotions']}")
    return 1 if FAILED_STEPS else 0


if __name__ == "__main__":
    sys.exit(main())
