from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from flybrain_body_adapter import FlyBrainBodyAdapter as BodyAdapter
from flybrain_wiring_features import EdgeSource, WiringFeatureParams, compute_wiring_features

REALISM_GATE = 0.90
CRITIC_ASE_DIM = 8
DEFAULT_CACHE_ROOT = Path.home() / ".loci" / "flybrain-grower-cache"
DEFAULT_CELL_TYPES: tuple[str, ...] = (
    "ascending",
    "dn_sez",
    "dn_vnc",
    "interneuron",
    "rgn",
    "sensory",
)
DEFAULT_COMPARTMENT_BIAS: Mapping[str, Mapping[str, float]] = {
    "ascending": {"axon_out_frac": 0.74, "dendrite_in_frac": 0.72},
    "dn_sez": {"axon_out_frac": 0.83, "dendrite_in_frac": 0.64},
    "dn_vnc": {"axon_out_frac": 0.85, "dendrite_in_frac": 0.62},
    "interneuron": {"axon_out_frac": 0.61, "dendrite_in_frac": 0.61},
    "rgn": {"axon_out_frac": 0.80, "dendrite_in_frac": 0.66},
    "sensory": {"axon_out_frac": 0.88, "dendrite_in_frac": 0.86},
}


def load_warm_start_genome(type_stats_path: str | Path) -> dict[str, Any]:
    from flybrain_grower_v0 import genome_from_data

    return genome_from_data(type_stats_path)


class GrowerOptimizer:
    """CMA-ES scaffold for grow-to-spec optimization over a compact SBM genome.

    The current body-adapter integration expects the adapter to expose a
    ``reward(...)`` method. The optimizer passes the sampled connectomes,
    decoded SBM and realism metadata to that hook after the realism gate passes.
    """

    def __init__(
        self,
        body_adapter: BodyAdapter,
        realism_critic_path: str,
        n_neurons: int = 300,
        sigma0: float = 0.3,
        warm_start_genome: Mapping[str, Any] | None = None,
        gate_schedule: list[float] | None = None,
        gate_steps: list[int] | None = None,
    ) -> None:
        self.body_adapter = body_adapter
        self.realism_critic_path = str(realism_critic_path)
        self.n_neurons = int(n_neurons)
        self.sigma0 = float(sigma0)
        self.warm_start_genome = dict(warm_start_genome) if isinstance(warm_start_genome, Mapping) else warm_start_genome
        if self.n_neurons < 2:
            raise ValueError("n_neurons must be >= 2")
        if self.sigma0 <= 0.0:
            raise ValueError("sigma0 must be > 0")

        self._critic = joblib.load(self.realism_critic_path)
        self.cell_types = self._critic_cell_types()
        self.n_cell_types = len(self.cell_types)
        self.genome_dim = self.n_cell_types + (2 * self.n_cell_types * self.n_cell_types)
        self._critic_feature_columns = self._critic_feature_names()
        self._class_index = {str(name): idx for idx, name in enumerate(self._critic_classes())}
        self.gate_schedule, self.gate_steps = self._normalize_gate_schedule(gate_schedule, gate_steps)
        self._cache_root = DEFAULT_CACHE_ROOT
        self._last_evaluation: dict[str, Any] | None = None
        self._current_generation: int = 0

    def genome_to_sbm(self, genome: np.ndarray) -> dict[str, Any]:
        flat = self._coerce_genome(genome)
        count_slice, conn_slice, syn_slice = self._layout_slices()
        count_logits = flat[count_slice]
        conn_logits = flat[conn_slice].reshape(self.n_cell_types, self.n_cell_types)
        syn_logits = flat[syn_slice].reshape(self.n_cell_types, self.n_cell_types)

        type_counts = self._counts_from_logits(count_logits)
        conn_prob = self._sigmoid(conn_logits)
        syn_mean = self._softplus(syn_logits) + 0.5
        syn_std = np.maximum(np.sqrt(syn_mean) + 0.5, syn_mean * 0.75)

        connection_probs = self._matrix_to_nested_dict(conn_prob)
        syn_mean_dict = self._matrix_to_nested_dict(syn_mean)
        synapse_count_params = {
            pre: {
                post: {
                    "mean": float(syn_mean[row, col]),
                    "std": float(syn_std[row, col]),
                }
                for col, post in enumerate(self.cell_types)
            }
            for row, pre in enumerate(self.cell_types)
        }
        compartment_bias = {
            cell_type: {
                "axon_out_frac": float(DEFAULT_COMPARTMENT_BIAS.get(cell_type, {}).get("axon_out_frac", 0.5)),
                "dendrite_in_frac": float(DEFAULT_COMPARTMENT_BIAS.get(cell_type, {}).get("dendrite_in_frac", 0.5)),
            }
            for cell_type in self.cell_types
        }

        return {
            "cell_types": list(self.cell_types),
            "type_counts": {cell_type: int(count) for cell_type, count in zip(self.cell_types, type_counts)},
            "connection_probs": connection_probs,
            "syn_mean": syn_mean_dict,
            "synapse_count_params": synapse_count_params,
            "compartment_bias": compartment_bias,
            "metadata": {
                "schema_version": "flybrain-grower-genome/v0",
                "generator": "GrowerOptimizer.genome_to_sbm",
                "constraint_tier": "fly_constrained",
                "n_neurons": self.n_neurons,
                "realism_critic_path": self.realism_critic_path,
            },
            "genome_vector": flat.copy(),
            "genome_layout": {
                "type_count_logits": [count_slice.start, count_slice.stop],
                "connection_prob_logits": [conn_slice.start, conn_slice.stop],
                "syn_mean_logits": [syn_slice.start, syn_slice.stop],
            },
            "raw_parameters": {
                "type_count_logits": flat[count_slice].copy(),
                "connection_prob_logits": flat[conn_slice].copy(),
                "syn_mean_logits": flat[syn_slice].copy(),
            },
        }

    def sbm_to_genome(self, sbm: Mapping[str, Any]) -> np.ndarray:
        genome = sbm.get("genome_vector")
        if genome is not None:
            return self._coerce_genome(np.asarray(genome, dtype=np.float64))
        raw = sbm.get("raw_parameters")
        if isinstance(raw, Mapping):
            pieces = []
            for key in ("type_count_logits", "connection_prob_logits", "syn_mean_logits"):
                if key not in raw:
                    raise ValueError(f"missing raw genome slice {key!r}")
                pieces.append(np.asarray(raw[key], dtype=np.float64).reshape(-1))
            return self._coerce_genome(np.concatenate(pieces, axis=0))
        raise ValueError("sbm payload does not contain genome_vector or raw_parameters")

    def evaluate_genome(self, genome: np.ndarray, n_samples: int = 10, current_generation: int | None = None) -> float:
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
            self._last_evaluation = {
                "passed_realism_gate": False,
                "task_reward": 0.0,
                "genome": genome.copy(),
                "realism_gate": gate_threshold,
            }
            return 0.0

        task_reward = self._task_reward(genome=self.sbm_to_genome(sbm), sbm=sbm, samples=sampled)
        self._last_evaluation = {
            "passed_realism_gate": True,
            "task_reward": float(task_reward),
            "genome": genome.copy(),
            "realism_gate": gate_threshold,
        }
        return float(mean_realism * task_reward)

    def run(self, max_iter: int = 100) -> dict[str, Any]:
        if int(max_iter) < 1:
            raise ValueError("max_iter must be >= 1")
        try:
            import cma
        except ImportError as exc:  # pragma: no cover - exercised only when cma is missing
            raise ImportError("GrowerOptimizer.run() requires the cma package; install it with `pip install cma`") from exc

        es = cma.CMAEvolutionStrategy(
            self._initial_genome_mean(),
            self.sigma0,
            {
                "seed": int(self._stable_seed(np.asarray([self.n_neurons, self.sigma0], dtype=np.float64))),
                "verb_disp": 0,
                "verb_log": 0,
                "verbose": -9,
                "tolFun": 1e-11,
                "tolX": 1e-11,
            },
        )
        best_genome: np.ndarray | None = None
        best_score = float("-inf")
        history: list[dict[str, float]] = []
        gate_passing_genomes: list[np.ndarray] = []
        gate_passing_rewards: list[float] = []

        self._current_generation = 0
        for iteration in range(int(max_iter)):
            self._current_generation = iteration
            candidates = [self._coerce_genome(np.asarray(candidate, dtype=np.float64)) for candidate in es.ask()]
            scores: list[float] = []
            for candidate in candidates:
                score = float(self.evaluate_genome(candidate, current_generation=self._current_generation))
                scores.append(score)
                evaluation = self._last_evaluation if isinstance(self._last_evaluation, Mapping) else None
                if evaluation and evaluation.get("passed_realism_gate"):
                    gate_passing_rewards.append(float(evaluation.get("task_reward", 0.0)))
                    if len(gate_passing_genomes) < 50:
                        passed_genome = evaluation.get("genome")
                        gate_passing_genomes.append(
                            self._coerce_genome(np.asarray(passed_genome if passed_genome is not None else candidate, dtype=np.float64)).copy()
                        )
                elif evaluation is None and score > 0.0:
                    gate_passing_rewards.append(score)
                    if len(gate_passing_genomes) < 50:
                        gate_passing_genomes.append(candidate.copy())
            es.tell(candidates, [-score for score in scores])

            iteration_best = max(scores)
            iteration_mean = float(np.mean(scores))
            history.append(
                {
                    "iteration": float(iteration + 1),
                    "best_score": float(iteration_best),
                    "mean_score": iteration_mean,
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
        behavioral_diversity = float(np.std(np.asarray(gate_passing_rewards, dtype=np.float64))) if len(gate_passing_rewards) >= 2 else 0.0
        return {
            "best_genome": best_genome,
            "best_score": best_score,
            "best_sbm": self.genome_to_sbm(best_genome),
            "history": history,
            "degeneracy_count": int(len(gate_passing_rewards)),
            "behavioral_diversity": behavioral_diversity,
            "gate_passing_genomes": gate_passing_genomes,
            "stop_reason": dict(es.stop()),
        }

    def _critic_cell_types(self) -> tuple[str, ...]:
        classes = tuple(self._critic_classes())
        return classes if classes else DEFAULT_CELL_TYPES

    def _critic_classes(self) -> tuple[str, ...]:
        if hasattr(self._critic, "classes_"):
            return tuple(str(name) for name in getattr(self._critic, "classes_"))
        clf = getattr(getattr(self._critic, "named_steps", {}), "get", lambda *_: None)("clf")
        if clf is not None and hasattr(clf, "classes_"):
            return tuple(str(name) for name in clf.classes_)
        return DEFAULT_CELL_TYPES

    def _critic_feature_names(self) -> tuple[str, ...]:
        names = getattr(self._critic, "feature_names_in_", None)
        if names is not None:
            return tuple(str(name) for name in names)
        schema_path = Path(self.realism_critic_path).with_name("schema.json")
        if schema_path.is_file():
            import json

            payload = json.loads(schema_path.read_text(encoding="utf-8"))
            numeric = [str(name) for name in payload.get("numeric", [])]
            categorical = [str(name) for name in payload.get("categorical", [])]
            if numeric or categorical:
                return tuple(numeric + categorical)
        raise ValueError("critic model does not expose feature_names_in_ and schema.json is missing")

    def gate_at(self, generation: int) -> float:
        """Return the realism gate threshold for this generation."""

        gate = self.gate_schedule[0]
        for step, threshold in zip(self.gate_steps, self.gate_schedule):
            if int(generation) < step:
                break
            gate = threshold
        return gate

    def _layout_slices(self) -> tuple[slice, slice, slice]:
        counts = slice(0, self.n_cell_types)
        conn_start = counts.stop
        conn_stop = conn_start + (self.n_cell_types * self.n_cell_types)
        conn = slice(conn_start, conn_stop)
        syn = slice(conn.stop, conn.stop + (self.n_cell_types * self.n_cell_types))
        return counts, conn, syn

    def _coerce_genome(self, genome: np.ndarray) -> np.ndarray:
        flat = np.asarray(genome, dtype=np.float64).reshape(-1)
        if flat.shape != (self.genome_dim,):
            raise ValueError(f"expected genome shape {(self.genome_dim,)}, got {flat.shape}")
        if not np.all(np.isfinite(flat)):
            raise ValueError("genome must contain only finite values")
        return flat

    def _counts_from_logits(self, logits: np.ndarray) -> np.ndarray:
        shifted = logits - float(np.max(logits))
        probs = np.exp(shifted)
        probs /= probs.sum()
        raw = probs * self.n_neurons
        counts = np.floor(raw).astype(int)
        remainder = self.n_neurons - int(counts.sum())
        if remainder > 0:
            order = np.argsort(-(raw - counts), kind="stable")
            counts[order[:remainder]] += 1
        return counts

    def _matrix_to_nested_dict(self, matrix: np.ndarray) -> dict[str, dict[str, float]]:
        return {
            pre: {post: float(matrix[row, col]) for col, post in enumerate(self.cell_types)}
            for row, pre in enumerate(self.cell_types)
        }

    def _sample_connectome(self, sbm: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
        rng = np.random.default_rng(int(seed))
        type_counts = {str(key): int(value) for key, value in sbm["type_counts"].items()}
        type_to_ids: dict[str, list[int]] = {}
        node_types: list[str] = []
        cursor = 0
        for cell_type in self.cell_types:
            count = type_counts.get(cell_type, 0)
            ids = list(range(cursor, cursor + count))
            type_to_ids[cell_type] = ids
            node_types.extend([cell_type] * count)
            cursor += count
        adjacency: list[tuple[int, int, int]] = []

        for pre_type in self.cell_types:
            pre_ids = np.asarray(type_to_ids[pre_type], dtype=np.int64)
            if pre_ids.size == 0:
                continue
            for post_type in self.cell_types:
                post_ids = np.asarray(type_to_ids[post_type], dtype=np.int64)
                if post_ids.size == 0:
                    continue
                prob = float(sbm["connection_probs"][pre_type][post_type])
                if prob <= 0.0:
                    continue
                exists = rng.random((pre_ids.size, post_ids.size)) < prob
                if pre_type == post_type:
                    diag = np.arange(min(pre_ids.size, post_ids.size))
                    exists[diag, diag] = False
                edge_rows, edge_cols = np.where(exists)
                if edge_rows.size == 0:
                    continue
                params = sbm["synapse_count_params"][pre_type][post_type]
                counts = self._sample_synapse_counts(
                    mean=float(params["mean"]),
                    std=float(params["std"]),
                    size=int(edge_rows.size),
                    rng=rng,
                )
                keep = counts > 0
                if not np.any(keep):
                    continue
                for pre_id, post_id, weight in zip(pre_ids[edge_rows[keep]], post_ids[edge_cols[keep]], counts[keep]):
                    adjacency.append((int(pre_id), int(post_id), int(weight)))

        return {
            "adjacency": adjacency,
            "cell_type_map": {idx: node_types[idx] for idx in range(len(node_types))},
            "node_types": tuple(node_types),
            "n_neurons": len(node_types),
            "genome": self.sbm_to_genome(sbm),
            "sbm": sbm,
            "seed": int(seed),
        }

    def _sample_synapse_counts(self, *, mean: float, std: float, size: int, rng: np.random.Generator) -> np.ndarray:
        if size < 1:
            return np.zeros(0, dtype=np.int64)
        if not math.isfinite(mean) or mean <= 0.0:
            raise ValueError("synapse-count mean must be finite and > 0")
        if not math.isfinite(std) or std <= 0.0:
            raise ValueError("synapse-count std must be finite and > 0")
        variance = std * std
        if variance <= mean * (1.0 + 1e-9):
            return rng.poisson(mean, size=size).astype(np.int64)
        r = (mean * mean) / (variance - mean)
        p = r / (r + mean)
        if not (math.isfinite(r) and r > 0.0 and math.isfinite(p) and 0.0 < p < 1.0):
            raise ValueError("invalid negative-binomial parameters derived from mean/std")
        return rng.negative_binomial(r, p, size=size).astype(np.int64)

    def _extract_critic_frame(self, connectome: Mapping[str, Any], sbm: Mapping[str, Any]) -> pd.DataFrame:
        weights = self._adjacency_matrix(connectome)
        node_types = list(connectome["node_types"])
        wiring = self._wiring_feature_frame(connectome, node_types)
        structured = self._structured_feature_frame(weights, node_types, sbm)
        features = pd.concat([wiring.reset_index(drop=True), structured.reset_index(drop=True)], axis=1)
        return self._align_critic_features(features)

    def _wiring_feature_frame(self, connectome: Mapping[str, Any], node_types: Sequence[str]) -> pd.DataFrame:
        edge_path = self._edge_table_path(connectome)
        edge_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._write_edge_table(edge_path, connectome["adjacency"])
            frame, _stats = compute_wiring_features(
                edges=EdgeSource(
                    path=str(edge_path),
                    pre="pre",
                    post="post",
                    weight="weight",
                    unique_pairs=True,
                    format="parquet",
                ),
                node_ids=[str(i) for i in range(connectome["n_neurons"])],
                node_categories=list(node_types),
                params=WiringFeatureParams(top_k_neuropils=0, reciprocity=True, two_hop=True, category_tag=""),
            )
            return frame
        finally:
            try:
                edge_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _structured_feature_frame(
        self,
        weights: np.ndarray,
        node_types: Sequence[str],
        sbm: Mapping[str, Any],
    ) -> pd.DataFrame:
        node_count = len(node_types)
        biases = sbm.get("compartment_bias", {})
        cmpt = pd.DataFrame(
            {
                "cmpt__axon_output_share": [
                    float(biases.get(node_type, {}).get("axon_out_frac", 0.5)) for node_type in node_types
                ],
                "cmpt__axon_input_share": [
                    1.0 - float(biases.get(node_type, {}).get("dendrite_in_frac", 0.5)) for node_type in node_types
                ],
            }
        )

        out_total = weights.sum(axis=1)
        in_total = weights.sum(axis=0)
        etype_accum = {name: np.zeros((node_count,), dtype=np.float64) for name in ("aa", "ad", "da", "dd")}
        in_accum = {name: np.zeros((node_count,), dtype=np.float64) for name in ("aa", "ad", "da", "dd")}
        rows, cols = np.nonzero(weights)
        for pre_idx, post_idx in zip(rows, cols):
            edge_weight = float(weights[pre_idx, post_idx])
            pre_type = node_types[int(pre_idx)]
            post_type = node_types[int(post_idx)]
            axon_out = float(biases.get(pre_type, {}).get("axon_out_frac", 0.5))
            dendrite_in = float(biases.get(post_type, {}).get("dendrite_in_frac", 0.5))
            dendrite_out = 1.0 - axon_out
            axon_in = 1.0 - dendrite_in
            fractions = {
                "aa": axon_out * axon_in,
                "ad": axon_out * dendrite_in,
                "da": dendrite_out * axon_in,
                "dd": dendrite_out * dendrite_in,
            }
            for name, frac in fractions.items():
                contribution = edge_weight * frac
                etype_accum[name][pre_idx] += contribution
                in_accum[name][post_idx] += contribution

        etype = {}
        with np.errstate(divide="ignore", invalid="ignore"):
            for name in ("aa", "ad", "da", "dd"):
                etype[f"etype_out__{name}"] = np.where(out_total > 0.0, etype_accum[name] / out_total, 0.0)
                etype[f"etype_in__{name}"] = np.where(in_total > 0.0, in_accum[name] / in_total, 0.0)
        ase = self._spectral_embedding_features(weights, dim=CRITIC_ASE_DIM)
        return pd.concat([pd.DataFrame(etype), cmpt, ase], axis=1)

    def _spectral_embedding_features(self, weights: np.ndarray, *, dim: int) -> pd.DataFrame:
        node_count = weights.shape[0]
        out = np.zeros((node_count, dim), dtype=np.float64)
        inn = np.zeros((node_count, dim), dtype=np.float64)
        if node_count >= 2 and np.any(weights > 0.0):
            transformed = np.log1p(weights)
            k = min(int(dim), node_count - 1)
            if k > 0:
                try:
                    if node_count <= (dim + 1):
                        u, s, vt = np.linalg.svd(transformed, full_matrices=False)
                        u, s, vt = u[:, :k], s[:k], vt[:k, :]
                    else:
                        from scipy import sparse
                        from scipy.sparse.linalg import svds

                        u, s, vt = svds(
                            sparse.csr_matrix(transformed),
                            k=k,
                            v0=np.full(node_count, 1.0 / np.sqrt(node_count)),
                            solver="arpack",
                        )
                        order = np.argsort(-s, kind="stable")
                        u, s, vt = u[:, order], s[order], vt[order, :]
                    signs = np.sign(u[np.argmax(np.abs(u), axis=0), np.arange(k)])
                    signs[signs == 0.0] = 1.0
                    scale = np.sqrt(s) * signs
                    out[:, :k] = u * scale
                    inn[:, :k] = vt.T * scale
                except Exception:
                    pass
        data = {f"ase__out_{idx:02d}": out[:, idx] for idx in range(dim)}
        data.update({f"ase__in_{idx:02d}": inn[:, idx] for idx in range(dim)})
        return pd.DataFrame(data)

    def _align_critic_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        aligned = frame.copy()
        for column in self._critic_feature_columns:
            if column not in aligned.columns:
                aligned[column] = 0.0
        aligned = aligned.loc[:, list(self._critic_feature_columns)]
        aligned = aligned.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return aligned.astype(np.float64)

    def _realism_score(self, critic_frame: pd.DataFrame, node_types: Sequence[str]) -> float:
        probabilities = np.asarray(self._critic.predict_proba(critic_frame), dtype=np.float64)
        if probabilities.shape[0] != len(node_types):
            raise ValueError("critic returned a different number of rows than the sampled connectome")
        row_scores = []
        for row_index, node_type in enumerate(node_types):
            class_index = self._class_index.get(str(node_type))
            row_scores.append(0.0 if class_index is None else float(probabilities[row_index, class_index]))
        return float(np.mean(row_scores)) if row_scores else 0.0

    def _task_reward(self, *, genome: np.ndarray, sbm: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]) -> float:
        reward_fn = getattr(self.body_adapter, "reward", None)
        if reward_fn is None or not callable(reward_fn):
            raise TypeError("body_adapter must provide a callable reward(...) method for GrowerOptimizer")
        value = self._call_with_supported_kwargs(
            reward_fn,
            genome=genome.copy(),
            sbm=sbm,
            samples=list(samples),
            connectomes=[entry["connectome"] for entry in samples],
            critic_features=[entry["critic_features"] for entry in samples],
            realism_scores=[float(entry["realism_score"]) for entry in samples],
            n_neurons=self.n_neurons,
        )
        if isinstance(value, Mapping):
            if "reward" not in value:
                raise ValueError("adapter reward mapping must contain a 'reward' key")
            value = value["reward"]
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                raise ValueError("adapter reward sequence must not be empty")
            return float(np.mean(np.asarray(value, dtype=np.float64)))
        return float(value)

    def _call_with_supported_kwargs(self, fn: Any, **kwargs: Any) -> Any:
        signature = inspect.signature(fn)
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return fn(**kwargs)
        supported = {name: value for name, value in kwargs.items() if name in signature.parameters}
        return fn(**supported)

    def _adjacency_matrix(self, connectome: Mapping[str, Any]) -> np.ndarray:
        weights = np.zeros((connectome["n_neurons"], connectome["n_neurons"]), dtype=np.float64)
        for pre_id, post_id, weight in connectome["adjacency"]:
            weights[int(pre_id), int(post_id)] += float(weight)
        return weights

    def _write_edge_table(self, path: Path, adjacency: Sequence[tuple[int, int, int]]) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if adjacency:
            pre, post, weight = zip(*adjacency)
        else:
            pre, post, weight = (), (), ()
        table = pa.table(
            {
                "pre": pa.array(list(pre), type=pa.int64()),
                "post": pa.array(list(post), type=pa.int64()),
                "weight": pa.array(list(weight), type=pa.float64()),
            }
        )
        pq.write_table(table, path)

    def _edge_table_path(self, connectome: Mapping[str, Any]) -> Path:
        digest = hashlib.sha256()
        digest.update(str(connectome["seed"]).encode("utf-8"))
        digest.update(str(connectome["n_neurons"]).encode("utf-8"))
        for pre_id, post_id, weight in connectome["adjacency"]:
            digest.update(f"{pre_id}:{post_id}:{weight};".encode("utf-8"))
        return self._cache_root / "edge-tables" / f"grower-{digest.hexdigest()[:24]}.parquet"

    def _initial_genome_mean(self) -> np.ndarray:
        if self.warm_start_genome is not None:
            try:
                return self.sbm_to_genome(self.warm_start_genome).copy()
            except Exception:
                pass
        count_slice, conn_slice, syn_slice = self._layout_slices()
        mean = np.zeros((self.genome_dim,), dtype=np.float64)
        mean[conn_slice] = -2.0
        mean[syn_slice] = self._inverse_softplus(np.full((syn_slice.stop - syn_slice.start,), 3.0, dtype=np.float64))
        return mean

    @staticmethod
    def _normalize_gate_schedule(
        gate_schedule: Sequence[float] | None,
        gate_steps: Sequence[int] | None,
    ) -> tuple[list[float], list[int]]:
        schedule = list(gate_schedule) if gate_schedule is not None else [REALISM_GATE]
        steps = list(gate_steps) if gate_steps is not None else [0]
        if not schedule or not steps:
            raise ValueError("gate_schedule and gate_steps must be non-empty")
        if len(schedule) != len(steps):
            raise ValueError("gate_schedule and gate_steps must have the same length")
        normalized = sorted((int(step), float(threshold)) for step, threshold in zip(steps, schedule))
        resolved_steps = [step for step, _threshold in normalized]
        resolved_schedule = [threshold for _step, threshold in normalized]
        if any(step < 0 for step in resolved_steps):
            raise ValueError("gate_steps must be >= 0")
        if any(not math.isfinite(threshold) for threshold in resolved_schedule):
            raise ValueError("gate_schedule must contain only finite values")
        return resolved_schedule, resolved_steps

    @staticmethod
    def _stable_seed(values: np.ndarray) -> int:
        digest = hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).digest()
        return int.from_bytes(digest[:8], "little") % (2**32 - 1)

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-values))

    @staticmethod
    def _softplus(values: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0.0)

    @staticmethod
    def _inverse_softplus(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return np.log(np.expm1(values))


__all__ = ["GrowerOptimizer", "BodyAdapter", "REALISM_GATE", "load_warm_start_genome"]
