"""Grow-to-spec v1: warm-start + progressive gate."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

MODULE_ROOT_CANDIDATES = [
    Path(__file__).parent,
    Path("/home/rjmendez/development/flybrain-worktrees/copilot-review/flybrain"),
    Path("/home/rjmendez/development/flybrain-worktrees/copilot-import/flybrain"),
]
for candidate in MODULE_ROOT_CANDIDATES:
    if (candidate / "flybrain_grower_optimizer.py").is_file():
        sys.path.insert(0, str(candidate))
        break
else:
    sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from flybrain_body_adapter import LociMockBodyAdapter
from flybrain_grower_optimizer import GrowerOptimizer, load_warm_start_genome
from flybrain_grower_v0 import genome_from_data

TYPE_STATS = Path("/home/rjmendez/.loci/flybrain-real-models/construction-rules/l1em_type_stats.json")
HGB_PATH = Path("/home/rjmendez/.loci/flybrain-real-models/reports/l1em/l1em_io_class/models/hgb/base/model.joblib")
OUT_DIR = Path("/home/rjmendez/.loci/flybrain-real-models/grow-to-spec/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GATE_SCHEDULE = [0.50, 0.60, 0.70, 0.80, 0.90]
GATE_STEPS = [0, 10, 20, 30, 40]
SIZES = [100, 300]
GENERATION_BUDGET = {100: 2, 300: 1}
POPSIZE = 2
N_SAMPLES = 1
SIGMA0 = 0.3


def ensure_cma() -> None:
    try:
        import cma  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cma"])


def build_warm_start() -> dict[str, object]:
    try:
        return load_warm_start_genome(TYPE_STATS)
    except Exception:
        return genome_from_data(TYPE_STATS)


def build_adapter(n_neurons: int):
    try:
        adapter = LociMockBodyAdapter(n_neurons=n_neurons)
    except TypeError:
        adapter = LociMockBodyAdapter()
    if not hasattr(adapter, "reward"):
        adapter.reward = lambda *, realism_scores, **kw: float(np.mean(realism_scores)) if len(realism_scores) else 0.0
    return adapter


def build_optimizer(*, n_neurons: int, warm_genome: dict[str, object]) -> GrowerOptimizer:
    return GrowerOptimizer(
        body_adapter=build_adapter(n_neurons),
        realism_critic_path=str(HGB_PATH),
        n_neurons=n_neurons,
        sigma0=SIGMA0,
        warm_start_genome=warm_genome,
        gate_schedule=GATE_SCHEDULE,
        gate_steps=GATE_STEPS,
    )


def run_budgeted_cma(optimizer: GrowerOptimizer, *, max_iter: int) -> dict[str, object]:
    import cma

    es = cma.CMAEvolutionStrategy(
        optimizer._initial_genome_mean(),
        optimizer.sigma0,
        {
            "seed": int(optimizer._stable_seed(np.asarray([optimizer.n_neurons, optimizer.sigma0], dtype=np.float64))),
            "popsize": POPSIZE,
            "verb_disp": 0,
            "verb_log": 0,
            "verbose": -9,
            "tolFun": 1e-11,
            "tolX": 1e-11,
        },
    )
    best_genome = None
    best_score = float("-inf")
    history = []
    gate_passing_genomes = []
    gate_passing_rewards = []
    optimizer._current_generation = 0

    for iteration in range(int(max_iter)):
        optimizer._current_generation = iteration
        candidates = [optimizer._coerce_genome(np.asarray(candidate, dtype=np.float64)) for candidate in es.ask()]
        scores = []
        for candidate in candidates:
            score = float(optimizer.evaluate_genome(candidate, n_samples=N_SAMPLES, current_generation=optimizer._current_generation))
            scores.append(score)
            evaluation = optimizer._last_evaluation if isinstance(optimizer._last_evaluation, dict) else None
            if evaluation and evaluation.get("passed_realism_gate"):
                gate_passing_rewards.append(float(evaluation.get("task_reward", 0.0)))
                if len(gate_passing_genomes) < 50:
                    gate_passing_genomes.append(np.asarray(evaluation.get("genome", candidate), dtype=np.float64).copy())
        es.tell(candidates, [-score for score in scores])
        iteration_best = max(scores)
        history.append(
            {
                "iteration": float(iteration + 1),
                "best_score": float(iteration_best),
                "mean_score": float(np.mean(scores)),
            }
        )
        best_idx = int(np.argmax(scores))
        if scores[best_idx] > best_score:
            best_score = float(scores[best_idx])
            best_genome = candidates[best_idx].copy()
        if es.stop():
            break

    if best_genome is None:
        raise RuntimeError("CMA-ES terminated before producing a candidate genome")
    return {
        "best_genome": best_genome,
        "best_score": float(best_score),
        "best_sbm": optimizer.genome_to_sbm(best_genome),
        "history": history,
        "degeneracy_count": int(len(gate_passing_rewards)),
        "behavioral_diversity": float(np.std(np.asarray(gate_passing_rewards, dtype=np.float64))) if len(gate_passing_rewards) >= 2 else 0.0,
        "gate_passing_genomes": gate_passing_genomes,
        "stop_reason": dict(es.stop()),
    }


def passed_gate_090(history: list[dict[str, float]]) -> bool:
    return any(float(entry.get("iteration", 0.0)) >= 41.0 and float(entry.get("best_score", 0.0)) > 0.0 for entry in history)


def main() -> None:
    ensure_cma()
    warm_genome = build_warm_start()

    results: dict[str, dict[str, object]] = {}
    for n_neurons in SIZES:
        print(f"\n=== n_neurons={n_neurons} ===", flush=True)
        t0 = time.time()
        optimizer = build_optimizer(n_neurons=n_neurons, warm_genome=warm_genome)
        result = run_budgeted_cma(optimizer, max_iter=GENERATION_BUDGET[n_neurons])
        elapsed = time.time() - t0
        history = result["history"]
        gate_090 = passed_gate_090(history)

        print(f"  best_score: {result['best_score']:.4f}", flush=True)
        print(f"  generations: {len(history)}", flush=True)
        print(f"  degeneracy: {result.get('degeneracy_count', 0)}", flush=True)
        print(f"  time: {elapsed:.1f}s", flush=True)
        print(f"  passed_0.90_gate: {gate_090}", flush=True)

        results[str(n_neurons)] = {
            "best_score": float(result["best_score"]),
            "history": history,
            "degeneracy_count": int(result.get("degeneracy_count", 0)),
            "behavioral_diversity": float(result.get("behavioral_diversity", 0.0)),
            "elapsed_s": float(elapsed),
            "generations": int(len(history)),
            "generation_budget": int(GENERATION_BUDGET[n_neurons]),
            "popsize": int(POPSIZE),
            "n_samples": int(N_SAMPLES),
            "passed_0_90_gate": bool(gate_090),
        }

    output = {
        "experiment": "grow-to-spec-v1",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "warm_start": True,
        "gate_schedule": GATE_SCHEDULE,
        "gate_steps": GATE_STEPS,
        "sizes_tested": SIZES,
        "results": results,
        "notes": (
            "v1: warm-start from l1em type_stats + progressive gate 0.50→0.90 + tolFun/tolX flat-landscape fix; "
            "run as a budgeted pilot with popsize=2 and n_samples=1 because sparse-checkout fallback imports make each realism eval ~100s"
        ),
    }

    out_path = OUT_DIR / "v1_experiment.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("\nSaved to", out_path, flush=True)


if __name__ == "__main__":
    main()
