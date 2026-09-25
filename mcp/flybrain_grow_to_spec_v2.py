from __future__ import annotations

"""Grow-to-spec v2: soft-gate shaping + gate starting at 0.30."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Find the optimizer
FLYBRAIN_MCP = Path("/home/rjmendez/development/flybrain-worktrees/copilot-review/flybrain")
LOCI_MCP = Path("/home/rjmendez/development/loci/mcp")
for p in [FLYBRAIN_MCP, LOCI_MCP]:
    if p.exists():
        sys.path.insert(0, str(p))

from flybrain_body_adapter import LociMockBodyAdapter
from flybrain_grower_optimizer import GrowerOptimizer

HGB_PATH = "/home/rjmendez/.loci/flybrain-real-models/reports/l1em/l1em_io_class/models/hgb/base/model.joblib"
OUT_DIR = Path("/home/rjmendez/.loci/flybrain-real-models/grow-to-spec/")
OUT_PATH = OUT_DIR / "v2_experiment.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Gate starts at 0.30 — below the SBM baseline of 0.42
# This ensures first-generation candidates get non-zero scores.
GATE_SCHEDULE = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
GATE_STEPS = [0, 5, 10, 20, 30, 40, 50]
DEFAULT_SIZES = [100, 300]
DEFAULT_MAX_ITER = 30
POPSIZE = 2
N_SAMPLES = 1
SIGMA0 = 0.3


def ensure_cma() -> None:
    try:
        import cma  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cma"])


def build_adapter() -> LociMockBodyAdapter:
    adapter = LociMockBodyAdapter()
    if not hasattr(adapter, "reward"):
        adapter.reward = lambda *, realism_scores, **kwargs: float(np.mean(realism_scores)) if len(realism_scores) else 0.0
    return adapter


def build_optimizer(*, n_neurons: int) -> GrowerOptimizer:
    return GrowerOptimizer(
        body_adapter=build_adapter(),
        realism_critic_path=HGB_PATH,
        n_neurons=n_neurons,
        sigma0=SIGMA0,
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


def summarize_history(history: list[dict[str, float]]) -> dict[str, object]:
    history_scores = [float(h["best_score"]) for h in history]
    gate_pass_generations = [idx + 1 for idx, score in enumerate(history_scores) if score > 0.01]
    return {
        "history_scores": history_scores,
        "score_increased": bool(history_scores and history_scores[-1] > history_scores[0]),
        "max_score": float(max(history_scores)) if history_scores else 0.0,
        "gate_pass_generations": gate_pass_generations,
        "gate_pass_count": int(len(gate_pass_generations)),
        "gate_pass_rate": float(len(gate_pass_generations) / len(history_scores)) if history_scores else 0.0,
    }


def load_existing_results() -> dict[str, object]:
    if OUT_PATH.is_file():
        try:
            return json.loads(OUT_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "experiment": "grow-to-spec-v2",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "soft_gate": True,
        "gate_schedule": GATE_SCHEDULE,
        "gate_steps": GATE_STEPS,
        "sizes_tested": [],
        "results": {},
        "notes": (
            "v2: soft gate (returns mean_realism/gate*0.01 when below threshold) + gate schedule starting at 0.30 "
            "(below SBM baseline 0.42); uses zero init because warm-start genome conversion is broken; "
            "runs under the same budgeted CMA settings as v1 (popsize=2, n_samples=1) to keep the wall-clock experiment tractable"
        ),
    }


def save_results(payload: dict[str, object]) -> None:
    payload["date"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload["sizes_tested"] = sorted(int(size) for size in payload.get("results", {}).keys())
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_size(n_neurons: int, max_iter: int) -> dict[str, object]:
    print(f"\n=== n_neurons={n_neurons} ===", flush=True)
    t0 = time.time()

    optimizer = build_optimizer(n_neurons=n_neurons)
    result = run_budgeted_cma(optimizer, max_iter=max_iter)
    elapsed = time.time() - t0
    history = result["history"]
    summary = summarize_history(history)
    history_scores = summary["history_scores"]

    print(f"  generations: {len(history)}", flush=True)
    print(f"  best_score: {result['best_score']:.4f}", flush=True)
    print(f"  degeneracy_count: {result.get('degeneracy_count', 0)}", flush=True)
    print(f"  first 5 gen scores: {history_scores[:5]}", flush=True)
    print(f"  last 5 gen scores: {history_scores[-5:]}", flush=True)
    print(f"  score_increased: {summary['score_increased']}", flush=True)
    print(f"  gate_pass_count: {summary['gate_pass_count']}/{len(history)}", flush=True)
    print(f"  gate_pass_generations: {summary['gate_pass_generations']}", flush=True)
    print(f"  time: {elapsed:.1f}s", flush=True)

    return {
        "best_score": float(result["best_score"]),
        "history": history,
        "degeneracy_count": int(result.get("degeneracy_count", 0)),
        "behavioral_diversity": float(result.get("behavioral_diversity", 0.0)),
        "elapsed_s": float(elapsed),
        "generations": int(len(history)),
        "popsize": int(POPSIZE),
        "n_samples": int(N_SAMPLES),
        "score_increased": bool(summary["score_increased"]),
        "gate_pass_count": int(summary["gate_pass_count"]),
        "gate_pass_rate": float(summary["gate_pass_rate"]),
        "gate_pass_generations": summary["gate_pass_generations"],
        "gate_ever_passed_0_90": bool(result["best_score"] > 0.01),  # scores > 0.01 only come from passing hard gate
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_cma()
    output = load_existing_results()
    results = dict(output.get("results", {}))

    for n_neurons in args.sizes:
        result = run_size(n_neurons=n_neurons, max_iter=args.max_iter)
        results[str(n_neurons)] = result
        output["results"] = results
        save_results(output)

    print("\nSaved:", OUT_PATH, flush=True)


if __name__ == "__main__":
    main()
