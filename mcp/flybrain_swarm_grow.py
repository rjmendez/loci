from __future__ import annotations

"""Run a multi-seed grow-to-spec swarm experiment."""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np

FLYBRAIN_MCP = Path("/home/rjmendez/development/flybrain-worktrees/copilot-review/flybrain")
LOCI_MCP = Path("/home/rjmendez/development/loci/mcp")
for p in (LOCI_MCP, FLYBRAIN_MCP):
    if p.exists():
        sys.path.insert(0, str(p))

from flybrain_body_adapter import LociMockBodyAdapter
from flybrain_grower_optimizer import GrowerOptimizer

HGB_PATH = Path("/home/rjmendez/.loci/flybrain-real-models/reports/l1em/l1em_io_class/models/hgb/base/model.joblib")
TYPE_STATS_PATH = Path("/home/rjmendez/.loci/flybrain-real-models/construction-rules/l1em_type_stats.json")
V2_EXPERIMENT_PATH = Path("/home/rjmendez/.loci/flybrain-real-models/grow-to-spec/v2_experiment.json")
OUT_PATH = Path("/home/rjmendez/.loci/flybrain-real-models/grow-to-spec/swarm_experiment.json")

DEFAULT_SEEDS = list(range(10))
DEFAULT_N_NEURONS = 100
DEFAULT_MAX_ITER = 8
DEFAULT_POPSIZE = 20
DEFAULT_N_SAMPLES = 3
DEFAULT_MAX_WORKERS = 4
DEFAULT_SIGMA0 = 0.3
DEFAULT_GATE_SCHEDULE = [0.30, 0.50, 0.70]
DEFAULT_GATE_STEPS = [0, 4, 8]


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


def build_optimizer(*, n_neurons: int, gate_schedule: list[float], gate_steps: list[int]) -> GrowerOptimizer:
    return GrowerOptimizer(
        body_adapter=build_adapter(),
        realism_critic_path=str(HGB_PATH),
        n_neurons=n_neurons,
        sigma0=DEFAULT_SIGMA0,
        gate_schedule=gate_schedule,
        gate_steps=gate_steps,
    )


def evaluate_candidate(
    optimizer: GrowerOptimizer,
    genome: np.ndarray,
    *,
    n_samples: int,
    current_generation: int,
) -> dict[str, Any]:
    genome = optimizer._coerce_genome(np.asarray(genome, dtype=np.float64))
    sbm = optimizer.genome_to_sbm(genome)
    base_seed = optimizer._stable_seed(optimizer.sbm_to_genome(sbm))
    sampled: list[dict[str, Any]] = []
    realism_scores: list[float] = []

    for sample_index in range(int(n_samples)):
        connectome = optimizer._sample_connectome(sbm, seed=base_seed + sample_index)
        critic_frame = optimizer._extract_critic_frame(connectome, sbm)
        realism_score = float(optimizer._realism_score(critic_frame, connectome["node_types"]))
        sampled.append(
            {
                "connectome": connectome,
                "critic_features": critic_frame,
                "realism_score": realism_score,
            }
        )
        realism_scores.append(realism_score)

    gate_threshold = float(optimizer.gate_at(current_generation))
    mean_realism = float(np.mean(realism_scores)) if realism_scores else 0.0
    if mean_realism < gate_threshold:
        task_reward = 0.0
        objective = float(mean_realism / max(gate_threshold, 1e-9)) * 0.01
        passed_gate = False
    else:
        task_reward = float(optimizer._task_reward(genome=optimizer.sbm_to_genome(sbm), sbm=sbm, samples=sampled))
        objective = float(mean_realism * task_reward)
        passed_gate = True

    optimizer._last_evaluation = {
        "passed_realism_gate": passed_gate,
        "task_reward": float(task_reward),
        "mean_realism": float(mean_realism),
        "realism_scores": [float(value) for value in realism_scores],
        "genome": genome.copy(),
        "realism_gate": gate_threshold,
    }
    return {
        "objective": float(objective),
        "mean_realism": float(mean_realism),
        "task_reward": float(task_reward),
        "passed_realism_gate": bool(passed_gate),
        "realism_scores": [float(value) for value in realism_scores],
        "gate_threshold": gate_threshold,
        "genome": genome,
        "sbm": sbm,
    }


def run_seed(seed: int, config: dict[str, Any]) -> dict[str, Any]:
    ensure_cma()
    import cma

    optimizer = build_optimizer(
        n_neurons=int(config["n_neurons"]),
        gate_schedule=[float(x) for x in config["gate_schedule"]],
        gate_steps=[int(x) for x in config["gate_steps"]],
    )
    initial_mean = optimizer._initial_genome_mean()
    es = cma.CMAEvolutionStrategy(
        initial_mean,
        optimizer.sigma0,
        {
            "seed": int(seed),
            "popsize": int(config["popsize"]),
            "verb_disp": 0,
            "verb_log": 0,
            "verbose": -9,
            "tolFun": 1e-11,
            "tolX": 1e-11,
        },
    )

    t0 = time.time()
    best_candidate: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    optimizer._current_generation = 0

    for iteration in range(int(config["max_iter"])):
        optimizer._current_generation = iteration
        candidates = [optimizer._coerce_genome(np.asarray(candidate, dtype=np.float64)) for candidate in es.ask()]
        evaluations = [
            evaluate_candidate(
                optimizer,
                candidate,
                n_samples=int(config["n_samples"]),
                current_generation=optimizer._current_generation,
            )
            for candidate in candidates
        ]
        objective_scores = [float(entry["objective"]) for entry in evaluations]
        realism_scores = [float(entry["mean_realism"]) for entry in evaluations]
        es.tell(candidates, [-score for score in objective_scores])

        gate_threshold = float(optimizer.gate_at(iteration))
        gate_pass_count = int(sum(1 for entry in evaluations if entry["passed_realism_gate"]))
        best_idx = int(np.argmax(objective_scores))
        current_best = evaluations[best_idx]
        if best_candidate is None or current_best["objective"] > best_candidate["objective"]:
            best_candidate = current_best

        history.append(
            {
                "iteration": int(iteration + 1),
                "gate_threshold": gate_threshold,
                "best_score": float(max(objective_scores)),
                "mean_score": float(np.mean(objective_scores)),
                "best_realism": float(max(realism_scores)),
                "mean_realism": float(np.mean(realism_scores)),
                "gate_pass_count": gate_pass_count,
            }
        )
        if es.stop():
            break

    if best_candidate is None:
        raise RuntimeError(f"seed {seed} terminated before producing a candidate")

    return {
        "seed": int(seed),
        "best_score": float(best_candidate["objective"]),
        "best_realism": float(best_candidate["mean_realism"]),
        "passed_realism_gate": bool(best_candidate["passed_realism_gate"]),
        "task_reward": float(best_candidate["task_reward"]),
        "best_genome": [float(value) for value in np.asarray(best_candidate["genome"], dtype=np.float64).tolist()],
        "best_sbm_type_counts": {str(k): int(v) for k, v in best_candidate["sbm"]["type_counts"].items()},
        "best_sample_realism_scores": [float(value) for value in best_candidate["realism_scores"]],
        "history": history,
        "stop_reason": dict(es.stop()),
        "elapsed_s": float(time.time() - t0),
    }


def pairwise_genome_distance(genomes: list[np.ndarray]) -> float:
    if len(genomes) < 2:
        return 0.0
    distances = [float(np.linalg.norm(left - right)) for left, right in combinations(genomes, 2)]
    return float(np.mean(distances)) if distances else 0.0


def classify_degeneracy(scores: list[float], pairwise_distance: float) -> tuple[bool, str]:
    if not scores:
        return False, "No successful seed results were produced."
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    score_cv = float(std_score / mean_score) if mean_score > 0 else float("inf")
    confirmed = mean_score > 0.50 and pairwise_distance > 1.0 and score_cv <= 0.25
    if confirmed:
        interpretation = (
            "Degeneracy confirmed: best genomes remain far apart in genome space while maintaining similar, high realism."
        )
    elif mean_score > 0.50 and pairwise_distance <= 1.0:
        interpretation = (
            "High-scoring seeds converged toward a relatively tight region of genome space, so degeneracy is weak."
        )
    else:
        interpretation = (
            "The swarm did not reach a strong high-score / high-distance regime, so degeneracy is not confirmed yet."
        )
    return confirmed, interpretation


def run_swarm(config: dict[str, Any]) -> dict[str, Any]:
    t0 = time.time()
    seed_results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(config["max_workers"])) as pool:
        futures = {pool.submit(run_seed, int(seed), config): int(seed) for seed in config["seeds"]}
        for future in as_completed(futures):
            seed_results.append(future.result())

    seed_results.sort(key=lambda item: int(item["seed"]))
    final_scores = [float(item["best_realism"]) for item in seed_results]
    genomes = [np.asarray(item["best_genome"], dtype=np.float64) for item in seed_results]
    gate_pass_count = int(sum(1 for item in seed_results if float(item["best_realism"]) > 0.50))
    pairwise_distance = pairwise_genome_distance(genomes)
    degeneracy_confirmed, interpretation = classify_degeneracy(final_scores, pairwise_distance)

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "script": str(Path(__file__).resolve()),
            "optimizer_path": str(FLYBRAIN_MCP / "flybrain_grower_optimizer.py"),
            "grower_path": str(LOCI_MCP / "flybrain_grower_v0.py"),
            "realism_critic_path": str(HGB_PATH),
            "type_stats_path": str(TYPE_STATS_PATH),
            "v2_experiment_path": str(V2_EXPERIMENT_PATH),
            "n_neurons": int(config["n_neurons"]),
            "max_iter": int(config["max_iter"]),
            "popsize": int(config["popsize"]),
            "n_samples": int(config["n_samples"]),
            "max_workers": int(config["max_workers"]),
            "seeds": [int(seed) for seed in config["seeds"]],
            "gate_schedule": [float(x) for x in config["gate_schedule"]],
            "gate_steps": [int(x) for x in config["gate_steps"]],
            "sigma0": float(DEFAULT_SIGMA0),
            "thread_caps": {
                key: os.environ.get(key) for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
            "warm_start_mode": "zero_init",
            "notes": "Thread caps keep HGB predict_proba fast; warm-start genome conversion remains disabled because the optimizer expects genome_vector/raw_parameters.",
        },
        "seed_results": seed_results,
        "metrics": {
            "mean_final_score": float(np.mean(final_scores)) if final_scores else 0.0,
            "std_final_score": float(np.std(final_scores)) if final_scores else 0.0,
            "pairwise_genome_distance": float(pairwise_distance),
            "gate_pass_count": gate_pass_count,
            "degeneracy_confirmed": bool(degeneracy_confirmed),
            "interpretation": interpretation,
        },
        "elapsed_s": float(time.time() - t0),
    }


def save_results(payload: dict[str, Any]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-neurons", type=int, default=DEFAULT_N_NEURONS)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--popsize", type=int, default=DEFAULT_POPSIZE)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--gate-schedule", nargs="+", type=float, default=DEFAULT_GATE_SCHEDULE)
    parser.add_argument("--gate-steps", nargs="+", type=int, default=DEFAULT_GATE_STEPS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.gate_schedule) != len(args.gate_steps):
        raise ValueError("gate_schedule and gate_steps must have the same length")
    if not args.seeds:
        raise ValueError("at least one seed is required")
    ensure_cma()
    config = {
        "n_neurons": int(args.n_neurons),
        "max_iter": int(args.max_iter),
        "popsize": int(args.popsize),
        "n_samples": int(args.n_samples),
        "max_workers": int(args.max_workers),
        "seeds": [int(seed) for seed in args.seeds],
        "gate_schedule": [float(x) for x in args.gate_schedule],
        "gate_steps": [int(x) for x in args.gate_steps],
    }
    payload = run_swarm(config)
    save_results(payload)
    print(json.dumps(payload["metrics"], indent=2), flush=True)
    print(f"Saved: {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
