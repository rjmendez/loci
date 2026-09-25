from __future__ import annotations

"""Size-vs-fidelity sweep for the grow-to-spec pipeline.

The requested CMA-ES configuration is extremely expensive:
``popsize=20 * n_samples=3 * max_iter=10 == 600`` sampled connectomes per size.
This script therefore performs a feasibility probe first. If the projected full
run exceeds ``--max-estimated-runtime-secs`` (20 minutes by default), the size is
recorded as skipped with probe telemetry instead of launching a multi-hour run.
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

FLYBRAIN_ROOT = Path("/home/rjmendez/development/flybrain-worktrees/copilot-review/flybrain")
LOCI_MCP = Path("/home/rjmendez/development/loci/mcp")
for candidate in (LOCI_MCP, FLYBRAIN_ROOT):
    if candidate.exists():
        sys.path.insert(0, str(candidate))

from flybrain_body_adapter import LociMockBodyAdapter
from flybrain_grower_optimizer import GrowerOptimizer

TYPE_STATS_PATH = Path("/home/rjmendez/.loci/flybrain-real-models/construction-rules/l1em_type_stats.json")
CRITIC_PATH = Path("/home/rjmendez/.loci/flybrain-real-models/reports/l1em/l1em_io_class/models/hgb/base/model.joblib")
OUT_PATH = Path("/home/rjmendez/.loci/flybrain-real-models/grow-to-spec/size_fidelity_curves.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SIZES = [50, 100, 200, 500, 1000]
MAX_ITER = 10
POPSIZE = 20
N_SAMPLES = 3
SIGMA0 = 0.3
RANDOM_SEED = 42
GATE_SCHEDULE = [0.30, 0.50, 0.70]
GATE_STEPS = [0, 5, 10]
PROBE_SIZES = {50, 100}
DEFAULT_MAX_ESTIMATED_RUNTIME_SECS = 20 * 60

SOURCE_TO_IO_CLASS = {
    "ascending": "ascending",
    "DN-SEZ": "dn_sez",
    "pre-DN-SEZ": "dn_sez",
    "DN-VNC": "dn_vnc",
    "pre-DN-VNC": "dn_vnc",
    "RGN": "rgn",
    "sensory": "sensory",
    "CN": "interneuron",
    "KC": "interneuron",
    "LHN": "interneuron",
    "LN": "interneuron",
    "MB-FBN": "interneuron",
    "MB-FFN": "interneuron",
    "MBIN": "interneuron",
    "MBON": "interneuron",
    "PN": "interneuron",
    "PN-somato": "interneuron",
    "interneuron": "interneuron",
}


class SweepGrowerOptimizer(GrowerOptimizer):
    """GrowerOptimizer with richer per-evaluation telemetry."""

    def evaluate_genome(
        self,
        genome: np.ndarray,
        n_samples: int = 10,
        current_generation: int | None = None,
    ) -> float:
        if int(n_samples) < 1:
            raise ValueError("n_samples must be >= 1")
        genome = self._coerce_genome(np.asarray(genome, dtype=np.float64))
        sbm = self.genome_to_sbm(genome)
        base_seed = self._stable_seed(self.sbm_to_genome(sbm))
        sampled: list[dict[str, Any]] = []
        realism_scores: list[float] = []
        self._last_evaluation = None

        for sample_index in range(int(n_samples)):
            connectome = self._sample_connectome(sbm, seed=base_seed + sample_index)
            critic_frame = self._extract_critic_frame(connectome, sbm)
            realism_score = self._realism_score(critic_frame, connectome["node_types"])
            sampled.append(
                {
                    "connectome": connectome,
                    "critic_features": critic_frame,
                    "realism_score": realism_score,
                }
            )
            realism_scores.append(realism_score)

        gate_threshold = self.gate_at(self._current_generation if current_generation is None else current_generation)
        mean_realism = float(np.mean(realism_scores))
        if mean_realism < gate_threshold:
            objective_score = float(mean_realism / max(gate_threshold, 1e-9)) * 0.01
            self._last_evaluation = {
                "passed_realism_gate": False,
                "task_reward": 0.0,
                "genome": genome.copy(),
                "realism_gate": gate_threshold,
                "mean_realism": mean_realism,
                "objective_score": objective_score,
                "sample_realism_scores": [float(score) for score in realism_scores],
            }
            return objective_score

        task_reward = self._task_reward(genome=self.sbm_to_genome(sbm), sbm=sbm, samples=sampled)
        objective_score = float(mean_realism * task_reward)
        self._last_evaluation = {
            "passed_realism_gate": True,
            "task_reward": float(task_reward),
            "genome": genome.copy(),
            "realism_gate": gate_threshold,
            "mean_realism": mean_realism,
            "objective_score": objective_score,
            "sample_realism_scores": [float(score) for score in realism_scores],
        }
        return objective_score


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


def _allocate_counts_from_probabilities(probs: Sequence[float], n_neurons: int) -> list[int]:
    values = np.asarray(probs, dtype=np.float64)
    values = np.clip(values, 1e-9, None)
    values = values / values.sum()
    raw = values * int(n_neurons)
    counts = np.floor(raw).astype(int)
    remainder = int(n_neurons) - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remainder]] += 1
    return [int(value) for value in counts]


def _logit(value: float) -> float:
    clipped = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return float(math.log(clipped / (1.0 - clipped)))


def _inverse_softplus_shifted(mean_value: float) -> float:
    shifted = max(float(mean_value) - 0.5, 1e-6)
    return float(np.log(np.expm1(shifted)))


def build_warm_start_payload(type_stats_path: Path, optimizer: GrowerOptimizer) -> dict[str, Any]:
    stats = json.loads(type_stats_path.read_text(encoding="utf-8"))
    target_cell_types = list(optimizer.cell_types)
    source_cell_types = [str(value) for value in stats["cell_types"]]
    source_counts = {str(key): int(value) for key, value in stats["neuron_counts"].items()}
    connection_probability = stats["connection_probability"]
    mean_synapse_count = stats["mean_synapse_count"]
    std_synapse_count = stats["std_synapse_count"]

    aggregated_source_counts = {target: 0 for target in target_cell_types}
    for source in source_cell_types:
        target = SOURCE_TO_IO_CLASS.get(source, "interneuron")
        if target in aggregated_source_counts:
            aggregated_source_counts[target] += int(source_counts[source])

    total_source = sum(aggregated_source_counts.values())
    if total_source <= 0:
        raise ValueError("warm-start type statistics produced no aggregate counts")

    target_probs = [aggregated_source_counts.get(cell_type, 0) / total_source for cell_type in target_cell_types]
    target_counts = _allocate_counts_from_probabilities(target_probs, optimizer.n_neurons)
    count_logits = np.log(np.clip(np.asarray(target_counts, dtype=np.float64) / max(optimizer.n_neurons, 1), 1e-9, None))

    conn_logits = np.zeros((optimizer.n_cell_types, optimizer.n_cell_types), dtype=np.float64)
    syn_logits = np.zeros((optimizer.n_cell_types, optimizer.n_cell_types), dtype=np.float64)

    for row, pre_target in enumerate(target_cell_types):
        pre_sources = [source for source in source_cell_types if SOURCE_TO_IO_CLASS.get(source, "interneuron") == pre_target]
        for col, post_target in enumerate(target_cell_types):
            post_sources = [source for source in source_cell_types if SOURCE_TO_IO_CLASS.get(source, "interneuron") == post_target]
            pair_weight = 0.0
            prob_weighted = 0.0
            syn_mean_weight = 0.0
            syn_std_weight = 0.0
            expected_edge_weight = 0.0
            for pre_source in pre_sources:
                for post_source in post_sources:
                    pre_count = int(source_counts.get(pre_source, 0))
                    post_count = int(source_counts.get(post_source, 0))
                    if pre_count <= 0 or post_count <= 0:
                        continue
                    possible_pairs = float(pre_count * post_count)
                    if pre_source == post_source:
                        possible_pairs = float(max(pre_count * max(pre_count - 1, 0), 0))
                    if possible_pairs <= 0.0:
                        continue
                    block_prob = float(connection_probability[pre_source][post_source])
                    block_mean = float(mean_synapse_count[pre_source][post_source])
                    block_std = float(std_synapse_count[pre_source][post_source])
                    pair_weight += possible_pairs
                    prob_weighted += possible_pairs * block_prob
                    expected_edges = possible_pairs * max(block_prob, 0.0)
                    if expected_edges > 0.0:
                        expected_edge_weight += expected_edges
                        syn_mean_weight += expected_edges * max(block_mean, 1.0)
                        syn_std_weight += expected_edges * max(block_std, math.sqrt(max(block_mean, 1.0)))
            block_probability = (prob_weighted / pair_weight) if pair_weight > 0.0 else 0.05
            block_syn_mean = (syn_mean_weight / expected_edge_weight) if expected_edge_weight > 0.0 else 3.0
            conn_logits[row, col] = _logit(block_probability)
            syn_logits[row, col] = _inverse_softplus_shifted(block_syn_mean)

    genome_vector = np.concatenate(
        [count_logits.reshape(-1), conn_logits.reshape(-1), syn_logits.reshape(-1)],
        axis=0,
    ).astype(np.float64)
    return {
        "cell_types": target_cell_types,
        "type_counts": {cell_type: count for cell_type, count in zip(target_cell_types, target_counts)},
        "metadata": {
            "source_path": str(type_stats_path),
            "aggregate_map": "l1em-17-type -> l1em_io_class-6-type",
            "n_neurons": int(optimizer.n_neurons),
        },
        "genome_vector": genome_vector,
        "raw_parameters": {
            "type_count_logits": count_logits,
            "connection_prob_logits": conn_logits.reshape(-1),
            "syn_mean_logits": syn_logits.reshape(-1),
        },
    }


def build_optimizer(n_neurons: int) -> SweepGrowerOptimizer:
    optimizer = SweepGrowerOptimizer(
        body_adapter=build_adapter(),
        realism_critic_path=str(CRITIC_PATH),
        n_neurons=int(n_neurons),
        sigma0=SIGMA0,
        warm_start_genome=None,
        gate_schedule=GATE_SCHEDULE,
        gate_steps=GATE_STEPS,
    )
    optimizer.warm_start_genome = build_warm_start_payload(TYPE_STATS_PATH, optimizer)
    return optimizer


def estimate_from_probe(
    *,
    probe_eval_secs: float,
    max_iter: int,
    popsize: int,
    n_samples: int,
) -> float:
    return float(probe_eval_secs) * int(max_iter) * int(popsize) * int(n_samples)


def historical_skip_estimate_for_1000() -> dict[str, Any]:
    per_generation_secs = 1172.0 / 5.0
    estimated_total_secs = per_generation_secs * MAX_ITER
    return {
        "estimated_from": "user-provided n=100 timing",
        "historical_n100_wall_time_secs": 1172.0,
        "historical_n100_generations": 5,
        "estimated_per_generation_secs": float(per_generation_secs),
        "estimated_total_secs": float(estimated_total_secs),
        "skip_threshold_secs": DEFAULT_MAX_ESTIMATED_RUNTIME_SECS,
        "skip_reason": "user requested skipping n=1000 if the historical n=100 estimate exceeds 20 minutes",
    }


def run_full_cma(optimizer: SweepGrowerOptimizer, *, max_iter: int, popsize: int, n_samples: int, seed: int) -> dict[str, Any]:
    import cma

    es = cma.CMAEvolutionStrategy(
        optimizer._initial_genome_mean(),
        optimizer.sigma0,
        {
            "seed": int(seed),
            "popsize": int(popsize),
            "verb_disp": 0,
            "verb_log": 0,
            "verbose": -9,
            "tolFun": 1e-11,
            "tolX": 1e-11,
        },
    )
    history: list[dict[str, Any]] = []
    best_objective = float("-inf")
    best_realism = float("-inf")
    best_genome: np.ndarray | None = None

    for generation in range(int(max_iter)):
        optimizer._current_generation = generation
        candidates = [optimizer._coerce_genome(np.asarray(candidate, dtype=np.float64)) for candidate in es.ask()]
        objective_scores: list[float] = []
        realism_scores: list[float] = []
        passed_flags: list[bool] = []
        for candidate in candidates:
            objective_score = float(
                optimizer.evaluate_genome(
                    candidate,
                    n_samples=int(n_samples),
                    current_generation=optimizer._current_generation,
                )
            )
            evaluation = dict(optimizer._last_evaluation or {})
            objective_scores.append(objective_score)
            realism_scores.append(float(evaluation.get("mean_realism", 0.0)))
            passed_flags.append(bool(evaluation.get("passed_realism_gate", False)))
        es.tell(candidates, [-score for score in objective_scores])

        best_idx = int(np.argmax(objective_scores))
        generation_best_objective = float(objective_scores[best_idx])
        generation_best_realism = float(realism_scores[best_idx])
        generation_gate = float(optimizer.gate_at(generation))
        generation_passed_gate = bool(passed_flags[best_idx])
        history.append(
            {
                "generation": int(generation + 1),
                "best_objective_score": generation_best_objective,
                "best_realism_score": generation_best_realism,
                "mean_objective_score": float(np.mean(objective_scores)),
                "mean_realism_score": float(np.mean(realism_scores)),
                "gate_threshold": generation_gate,
                "passed_gate": generation_passed_gate,
            }
        )
        if generation_best_objective > best_objective:
            best_objective = generation_best_objective
            best_realism = generation_best_realism
            best_genome = candidates[best_idx].copy()
        if es.stop():
            break

    if best_genome is None:
        raise RuntimeError("CMA-ES terminated before producing a candidate genome")
    return {
        "history": history,
        "best_genome": best_genome,
        "best_objective_score": float(best_objective),
        "best_realism_score": float(best_realism),
        "stop_reason": dict(es.stop()),
    }


def build_skip_result(
    *,
    n_neurons: int,
    probe_realism: float | None,
    probe_objective: float | None,
    probe_elapsed_secs: float | None,
    estimated_full_run_secs: float | None,
    skip_reason: str,
    estimate_details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "skipped_estimate",
        "n_neurons": int(n_neurons),
        "final_score": None,
        "gate_pass_rate": None,
        "wall_time_secs": float(probe_elapsed_secs) if probe_elapsed_secs is not None else 0.0,
        "score_trajectory": [],
        "probe_realism_score": float(probe_realism) if probe_realism is not None else None,
        "probe_objective_score": float(probe_objective) if probe_objective is not None else None,
        "estimated_full_run_secs": float(estimated_full_run_secs) if estimated_full_run_secs is not None else None,
        "skip_reason": skip_reason,
    }
    if estimate_details:
        payload["estimate_details"] = dict(estimate_details)
    return payload


def run_size(
    n_neurons: int,
    *,
    max_iter: int,
    popsize: int,
    n_samples: int,
    seed: int,
    max_estimated_runtime_secs: float,
    probe_reference_n: int | None = None,
    probe_reference_eval_secs: float | None = None,
) -> dict[str, Any]:
    if int(n_neurons) == 1000:
        history_estimate = historical_skip_estimate_for_1000()
        if float(history_estimate["estimated_total_secs"]) > float(max_estimated_runtime_secs):
            return build_skip_result(
                n_neurons=n_neurons,
                probe_realism=None,
                probe_objective=None,
                probe_elapsed_secs=0.0,
                estimated_full_run_secs=float(history_estimate["estimated_total_secs"]),
                skip_reason=str(history_estimate["skip_reason"]),
                estimate_details=history_estimate,
            )

    probe_source_n = None
    probe_eval_secs: float | None = None
    probe_elapsed_secs: float | None = None
    probe_realism: float | None = None
    probe_objective: float | None = None
    if int(n_neurons) in PROBE_SIZES:
        optimizer = build_optimizer(n_neurons)
        t0 = time.time()
        probe_objective = float(optimizer.evaluate_genome(optimizer._initial_genome_mean(), n_samples=1, current_generation=0))
        probe_elapsed_secs = time.time() - t0
        probe_state = dict(optimizer._last_evaluation or {})
        probe_realism = float(probe_state.get("mean_realism", 0.0))
        probe_eval_secs = float(probe_elapsed_secs)
        probe_source_n = int(n_neurons)
    elif probe_reference_n and probe_reference_eval_secs:
        probe_source_n = int(probe_reference_n)
        scale = max(float(n_neurons) / float(probe_source_n), 1.0)
        probe_eval_secs = float(probe_reference_eval_secs) * scale

    if probe_eval_secs is not None:
        estimated_full_run_secs = estimate_from_probe(
            probe_eval_secs=probe_eval_secs,
            max_iter=max_iter,
            popsize=popsize,
            n_samples=n_samples,
        )
        if float(estimated_full_run_secs) > float(max_estimated_runtime_secs):
            return {
                **build_skip_result(
                    n_neurons=n_neurons,
                    probe_realism=probe_realism,
                    probe_objective=probe_objective,
                    probe_elapsed_secs=probe_elapsed_secs,
                    estimated_full_run_secs=estimated_full_run_secs,
                    skip_reason=(
                        f"Projected runtime {estimated_full_run_secs / 3600.0:.2f}h exceeds "
                        f"configured ceiling {max_estimated_runtime_secs / 60.0:.0f}m"
                    ),
                    estimate_details={
                        "estimated_from": "single warm-start evaluate_genome probe",
                        "probe_source_n": probe_source_n,
                        "probe_eval_secs": probe_eval_secs,
                        "projection_formula": "probe_eval_secs * max_iter * popsize * n_samples",
                    },
                ),
                "probe_eval_secs": float(probe_eval_secs),
            }

    optimizer = build_optimizer(n_neurons)
    started = time.time()
    result = run_full_cma(optimizer, max_iter=max_iter, popsize=popsize, n_samples=n_samples, seed=seed)
    elapsed = time.time() - started
    history = result["history"]
    score_trajectory = [float(entry["best_realism_score"]) for entry in history]
    gate_pass_count = sum(1 for entry in history if bool(entry["passed_gate"]))
    return {
        "status": "completed",
        "n_neurons": int(n_neurons),
        "final_score": float(score_trajectory[-1]) if score_trajectory else None,
        "best_realism_score": float(result["best_realism_score"]),
        "best_objective_score": float(result["best_objective_score"]),
        "gate_pass_rate": float(gate_pass_count / len(history)) if history else 0.0,
        "wall_time_secs": float(elapsed),
        "score_trajectory": score_trajectory,
        "history": history,
        "stop_reason": result["stop_reason"],
        "probe_eval_secs": float(probe_eval_secs) if probe_eval_secs is not None else None,
    }


def analyze_results(results: Mapping[str, Any]) -> dict[str, Any]:
    completed = [(int(size), payload) for size, payload in results.items() if payload.get("status") == "completed"]
    comparable = []
    gate_rates = []
    for size_text, payload in results.items():
        size = int(size_text)
        final_score = payload.get("final_score")
        wall_time = payload.get("wall_time_secs")
        if final_score is not None and wall_time and float(wall_time) > 0.0:
            comparable.append((size, float(final_score), float(final_score) / float(wall_time)))
        if payload.get("gate_pass_rate") is not None:
            gate_rates.append((size, float(payload["gate_pass_rate"])))

    peak_n = max(completed, key=lambda item: float(item[1].get("final_score", float("-inf"))))[0] if completed else None
    knee_n = max(comparable, key=lambda item: item[2])[0] if comparable else None
    gate_drop = None
    if len(gate_rates) >= 2:
        gate_drop = all(gate_rates[idx][1] >= gate_rates[idx + 1][1] for idx in range(len(gate_rates) - 1))
    elif len(gate_rates) == 1:
        gate_drop = None

    return {
        "peak_final_score_n": peak_n,
        "knee_n": knee_n,
        "gate_pass_rate_monotone_nonincreasing": gate_drop,
        "completed_sizes": [size for size, _payload in completed],
        "skipped_sizes": [int(size) for size, payload in results.items() if payload.get("status") != "completed"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--popsize", type=int, default=POPSIZE)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-estimated-runtime-secs", type=float, default=DEFAULT_MAX_ESTIMATED_RUNTIME_SECS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_cma()

    results: dict[str, Any] = {}
    probe_reference_n: int | None = None
    probe_reference_eval_secs: float | None = None
    for n_neurons in args.sizes:
        print(f"\n=== n_neurons={n_neurons} ===", flush=True)
        result = run_size(
            n_neurons=int(n_neurons),
            max_iter=int(args.max_iter),
            popsize=int(args.popsize),
            n_samples=int(args.n_samples),
            seed=int(args.random_seed),
            max_estimated_runtime_secs=float(args.max_estimated_runtime_secs),
            probe_reference_n=probe_reference_n,
            probe_reference_eval_secs=probe_reference_eval_secs,
        )
        results[str(n_neurons)] = result
        print(json.dumps(result, indent=2), flush=True)
        if result.get("probe_eval_secs") and result.get("wall_time_secs", 0.0) > 0.0 and result.get("probe_realism_score") is not None:
            probe_reference_n = int(n_neurons)
            probe_reference_eval_secs = float(result["probe_eval_secs"])

        partial_payload = {
            "experiment": "grow-to-spec-size-fidelity-curves",
            "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "critic_path": str(CRITIC_PATH),
            "type_stats_path": str(TYPE_STATS_PATH),
            "gate_schedule": GATE_SCHEDULE,
            "gate_steps": GATE_STEPS,
            "popsize": int(args.popsize),
            "n_samples": int(args.n_samples),
            "max_iter": int(args.max_iter),
            "random_seed": int(args.random_seed),
            "max_estimated_runtime_secs": float(args.max_estimated_runtime_secs),
            "sizes_tested": [int(value) for value in args.sizes],
            "results": results,
            "analysis": analyze_results(results),
            "notes": [
                "Warm start is generated by aggregating the 17-type l1em construction rules into the 6 io_class realism-critic classes.",
                "Before each size run, the script uses a single warm-start evaluate_genome probe to estimate the requested 10-generation CMA-ES wall time.",
                "When the estimate exceeds --max-estimated-runtime-secs, the size is recorded as skipped_estimate instead of launching a multi-hour run.",
            ],
        }
        OUT_PATH.write_text(json.dumps(partial_payload, indent=2), encoding="utf-8")

    payload = {
        "experiment": "grow-to-spec-size-fidelity-curves",
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "critic_path": str(CRITIC_PATH),
        "type_stats_path": str(TYPE_STATS_PATH),
        "gate_schedule": GATE_SCHEDULE,
        "gate_steps": GATE_STEPS,
        "popsize": int(args.popsize),
        "n_samples": int(args.n_samples),
        "max_iter": int(args.max_iter),
        "random_seed": int(args.random_seed),
        "max_estimated_runtime_secs": float(args.max_estimated_runtime_secs),
        "sizes_tested": [int(value) for value in args.sizes],
        "results": results,
        "analysis": analyze_results(results),
        "notes": [
            "Warm start is generated by aggregating the 17-type l1em construction rules into the 6 io_class realism-critic classes.",
            "Before each size run, the script uses a single warm-start evaluate_genome probe to estimate the requested 10-generation CMA-ES wall time.",
            "When the estimate exceeds --max-estimated-runtime-secs, the size is recorded as skipped_estimate instead of launching a multi-hour run.",
        ],
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved results to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
