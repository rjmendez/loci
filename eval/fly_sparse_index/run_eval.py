"""Offline A/B evaluation of the fly sparse-expansion index (see PREREGISTRATION.md).

Reads cached embeddings (embed_corpus.py) and a measured PN->KC spec file
(pinned sha256). Writes only aggregate metrics to --out. Never touches Loci's
live stores. Run with an interpreter that has numpy (hnswlib optional):

    /mnt/f/loci-eval/venv-ann/bin/python eval/fly_sparse_index/run_eval.py \
        --spec /mnt/f/.flybrain/cache/mb-params/<ver>/mb_pn_kc_params.json \
        --spec-sha256 <sha> --out /mnt/f/loci-eval/fly-sparse-index/results/<run>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mcp"))
import sparse_expansion_index as sei  # noqa: E402

try:
    import hnswlib  # type: ignore
except ImportError:  # pragma: no cover - optional
    hnswlib = None

TOPK = 10
CAND = 100
SPLIT_SEED = 12345
N_LOCI_QUERIES = 300
N_BOOT = 2000
# pre-registered held-out topics (PREREGISTRATION.md); others stay in the stored pool only
NOVELTY_TOPICS = ("mcp-core", "mcp-tests", "mcp-flybrain", "scripts", "docs", "mlops", "nfcorpus")


# ---------------------------------------------------------------- data


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def load_data(cache: Path) -> dict:
    loci = _unit(np.load(cache / "private" / "loci_emb.npy", allow_pickle=False))
    topics = [json.loads(line)["topic"] for line in (cache / "private" / "loci_chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(topics) != len(loci):
        raise SystemExit("loci chunks/embeddings length mismatch")
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(loci))
    q_idx, x_idx = np.sort(perm[:N_LOCI_QUERIES]), np.sort(perm[N_LOCI_QUERIES:])
    nf_docs = _unit(np.load(cache / "nfcorpus" / "doc_emb.npy", allow_pickle=False))
    nf_q = _unit(np.load(cache / "nfcorpus" / "query_emb.npy", allow_pickle=False))
    return {
        "loci": {"x": loci[x_idx], "q": loci[q_idx]},
        "nfcorpus": {"x": nf_docs, "q": nf_q},
        "pool": {"x": np.vstack([loci, nf_docs]), "topic": np.array(topics + ["nfcorpus"] * len(nf_docs))},
    }


def exact_topk(x: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    s = q @ x.T
    return np.argsort(-s, axis=1, kind="stable")[:, :k]


# ---------------------------------------------------------------- methods


class SimHash:
    def __init__(self, n_bits: int, seed: int):
        self.n_bits, self.seed = n_bits, seed

    def fit(self, x, basis=None):
        rng = np.random.default_rng(self.seed + 7_777)
        self.mean = x.mean(axis=0)
        self.planes = rng.standard_normal((x.shape[1], self.n_bits)).astype(np.float32)
        return self

    def encode(self, x):
        return sei.pack_codes(((x - self.mean) @ self.planes) > 0)

    def add(self, x):
        self.codes = self.encode(x)

    def distances(self, q):
        return sei.hamming_distances(self.encode(q), self.codes)

    def novelty(self, q):
        return self.distances(q).min(axis=1) / float(self.n_bits)

    @property
    def memory_bytes(self):
        return int(self.codes.nbytes + self.planes.nbytes + self.mean.nbytes)

    @property
    def code_bytes_per_item(self):
        return int(self.codes.shape[1] * 8)


class Fly:
    def __init__(self, spec, n_kc, wta, projection, mode, seed):
        self.idx = sei.SparseExpansionIndex(spec, n_kc, wta_fraction=wta, projection=projection, mode=mode, seed=seed)

    def fit(self, x, basis=None):
        self.idx.fit(x, basis=basis if self.idx.projection == "pca" else None)
        return self

    def add(self, x):
        self.idx.add(x)

    def distances(self, q):
        return sei.hamming_distances(self.idx.encode(q), self.idx._codes)

    def novelty(self, q):
        return self.idx.novelty(q)

    @property
    def memory_bytes(self):
        return self.idx.memory_bytes

    @property
    def code_bytes_per_item(self):
        return self.idx.code_bytes_per_item


def make_method(name: str, cfg: dict, specs: dict, seed: int):
    g, k, wta, c = cfg["G"], cfg["K"], cfg["wta"], cfg["claws"]
    meas = specs[cfg.get("matrix", "fw_right")]
    if name == "simhash":
        return SimHash(k, seed)
    if name == "flyhash_random":
        return Fly(sei.idealised_spec(g, c), k, wta, "pca", "sample", seed)
    if name == "flyhash_random_fullD":
        return Fly(sei.idealised_spec(cfg["D"], c), k, wta, "identity", "sample", seed)
    if name == "fly_measured_matrix":
        return Fly(meas, k, wta, "pca", "matrix", seed)
    if name == "fly_measured_sample":
        return Fly(meas, k, wta, "pca", "sample", seed)
    if name == "ctrl_measured_claws_uniform_glom":
        return Fly(meas.with_uniform_glomeruli(), k, wta, "pca", "sample", seed)
    if name == "ctrl_fixed_claws_measured_glom":
        return Fly(meas.with_fixed_claws(c), k, wta, "pca", "sample", seed)
    if name == "measured_glom_fixed_claws":  # claws sweep on measured weights
        return Fly(meas.with_fixed_claws(c), k, wta, "pca", "sample", seed)
    raise KeyError(name)


def run_code_method(name, cfg, specs, seed, x, q, truth, truth1, basis=None):
    t0 = time.perf_counter()
    m = make_method(name, cfg, specs, seed).fit(x, basis)
    m.add(x)
    build = time.perf_counter() - t0
    t0 = time.perf_counter()
    d = m.distances(q)
    cand = np.argsort(d, axis=1, kind="stable")[:, :CAND]
    lat = (time.perf_counter() - t0) / len(q) * 1e3
    top = cand[:, :TOPK]
    recall = np.array([len(set(a) & set(b)) / TOPK for a, b in zip(top, truth)])
    rr = np.array([1.0 / (list(c).index(t) + 1) if t in c else 0.0 for c, t in zip(cand, truth1)])
    sims = np.einsum("qd,qcd->qc", q, x[cand])
    rer = np.take_along_axis(cand, np.argsort(-sims, axis=1, kind="stable")[:, :TOPK], axis=1)
    recall_rr = np.array([len(set(a) & set(b)) / TOPK for a, b in zip(rer, truth)])
    return {"recall": recall, "rr": rr, "recall_rerank": recall_rr, "latency_ms": lat, "build_s": build,
            "memory_bytes": m.memory_bytes, "code_bytes_per_item": m.code_bytes_per_item}


def run_exact(x, q, truth, truth1):
    t0 = time.perf_counter()
    top = exact_topk(x, q, CAND)
    lat = (time.perf_counter() - t0) / len(q) * 1e3
    ones = np.ones(len(q))
    return {"recall": ones, "rr": ones, "recall_rerank": ones, "latency_ms": lat, "build_s": 0.0,
            "memory_bytes": int(x.nbytes), "code_bytes_per_item": int(x.shape[1] * 4)}


def run_hnsw(x, q, truth, truth1, seed, tmpdir):
    if hnswlib is None:
        return None
    t0 = time.perf_counter()
    p = hnswlib.Index(space="cosine", dim=x.shape[1])
    p.init_index(max_elements=len(x), ef_construction=200, M=16, random_seed=seed)
    p.set_num_threads(1)
    p.add_items(x, np.arange(len(x)))
    p.set_ef(64)  # hnswlib searches with max(ef, k), i.e. 100 for the top-100 query
    build = time.perf_counter() - t0
    t0 = time.perf_counter()
    labels, _ = p.knn_query(q, k=CAND)
    lat = (time.perf_counter() - t0) / len(q) * 1e3
    top = labels[:, :TOPK]
    recall = np.array([len(set(a) & set(b)) / TOPK for a, b in zip(top, truth)])
    rr = np.array([1.0 / (list(c).index(t) + 1) if t in c else 0.0 for c, t in zip(labels, truth1)])
    path = Path(tmpdir) / f"hnsw_{seed}.bin"
    p.save_index(str(path))
    mem = path.stat().st_size
    path.unlink()
    return {"recall": recall, "rr": rr, "recall_rerank": recall, "latency_ms": lat, "build_s": build,
            "memory_bytes": int(mem), "code_bytes_per_item": int(mem // len(x))}


# ---------------------------------------------------------------- stats


def boot_ci(v: np.ndarray, seed: int = 0, n: int = N_BOOT) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    s = np.concatenate([pos, neg])
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---------------------------------------------------------------- experiment blocks


def retrieval_block(data, cfg, specs, methods, seeds, tmpdir, with_hnsw=True):
    x, q = data["x"], data["q"]
    truth_full = exact_topk(x, q, TOPK)
    truth1 = truth_full[:, 0]
    out = {}
    if "_basis" not in data:  # shared PCA fit, computed once per dataset (build_s excludes it)
        data["_basis"] = sei.pca_basis(x)
    basis = data["_basis"]
    for name in methods:
        per_seed = [run_code_method(name, cfg, specs, s, x, q, truth_full, truth1, basis) for s in seeds]
        out[name] = per_seed
    out["exact"] = [run_exact(x, q, truth_full, truth1)]
    if with_hnsw and hnswlib is not None:
        out["hnsw"] = [run_hnsw(x, q, truth_full, truth1, s, tmpdir) for s in seeds]
    return out


def summarise_block(block):
    rows = {}
    for name, runs in block.items():
        rec = np.mean([r["recall"] for r in runs], axis=0)
        rr = np.mean([r["rr"] for r in runs], axis=0)
        rer = np.mean([r["recall_rerank"] for r in runs], axis=0)
        rows[name] = {
            "recall@10": boot_ci(rec), "mrr@100": boot_ci(rr), "recall@10_rerank100": boot_ci(rer),
            "recall@10_seed_sd": float(np.std([r["recall"].mean() for r in runs], ddof=1)) if len(runs) > 1 else 0.0,
            "latency_ms_per_query": float(np.median([r["latency_ms"] for r in runs])),
            "build_s": float(np.median([r["build_s"] for r in runs])),
            "memory_bytes": int(np.median([r["memory_bytes"] for r in runs])),
            "code_bytes_per_item": int(np.median([r["code_bytes_per_item"] for r in runs])),
            "_per_query_recall": rec, "_per_query_rerank": rer,
        }
    return rows


def diff(rows, a, b, key="_per_query_recall"):
    return boot_ci(rows[a][key] - rows[b][key], seed=1)


def novelty_block(pool, cfg, specs, methods, seeds, tmpdir, min_topic=150, max_items=500):
    topics, counts = np.unique(pool["topic"], return_counts=True)
    folds = [t for t, c in zip(topics, counts) if c >= min_topic and t in NOVELTY_TOPICS]
    rng = np.random.default_rng(SPLIT_SEED + 1)
    results = {m: {} for m in methods + ["exact", "hnsw"]}
    fold_data = []
    for t in folds:
        in_t = np.flatnonzero(pool["topic"] == t)
        out_t = rng.permutation(np.flatnonzero(pool["topic"] != t))
        n_store = int(0.8 * len(out_t))
        store, neg = out_t[:n_store], out_t[n_store:][:max_items]
        pos = rng.permutation(in_t)[:max_items]
        fold_data.append((t, store, pos, neg))
    x = pool["x"]
    for t, store, pos, neg in fold_data:
        xs = x[store]
        qp, qn = x[pos], x[neg]
        basis = sei.pca_basis(xs)
        results["exact"][t] = [(1 - (qp @ xs.T).max(axis=1), 1 - (qn @ xs.T).max(axis=1))]
        if hnswlib is not None:
            runs = []
            for s in seeds:
                p = hnswlib.Index(space="cosine", dim=x.shape[1])
                p.init_index(max_elements=len(xs), ef_construction=200, M=16, random_seed=s)
                p.set_num_threads(1)
                p.add_items(xs)
                p.set_ef(64)
                runs.append((p.knn_query(qp, k=1)[1][:, 0], p.knn_query(qn, k=1)[1][:, 0]))
            results["hnsw"][t] = runs
        for name in methods:
            runs = []
            for s in seeds:
                m = make_method(name, cfg, specs, s).fit(xs, basis)
                m.add(xs)
                runs.append((m.novelty(qp), m.novelty(qn)))
            results[name][t] = runs
    summary = {}
    per_fold = {}
    boots = {}
    rngb = np.random.default_rng(3)
    boot_idx = {t: [(rngb.integers(0, len(p), (1000, len(p))), rngb.integers(0, len(n), (1000, len(n))))
                    for p, n in [(fold[2], fold[3])]][0] for fold in fold_data for t in [fold[0]]}
    for name, by_fold in results.items():
        if not by_fold:
            continue
        aucs, boot_mat = [], []
        per_fold[name] = {}
        for t, runs in by_fold.items():
            sp = np.mean([r[0] for r in runs], axis=0)
            sn = np.mean([r[1] for r in runs], axis=0)
            a = auroc(sp, sn)
            aucs.append(a)
            per_fold[name][t] = round(a, 4)
            ip, ineg = boot_idx[t]
            boot_mat.append([auroc(sp[ip[b]], sn[ineg[b]]) for b in range(200)])
        bm = np.mean(np.array(boot_mat), axis=0)
        boots[name] = bm
        summary[name] = {"auroc_mean_over_folds": float(np.mean(aucs)), "ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]}
    diffs = {}
    for a in methods:
        for b in ("exact", "hnsw", "simhash", "flyhash_random"):
            if a != b and a in boots and b in boots:
                d = boots[a] - boots[b]
                diffs[f"{a} - {b}"] = {"mean": summary[a]["auroc_mean_over_folds"] - summary[b]["auroc_mean_over_folds"],
                                       "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return {"folds": [f[0] for f in fold_data], "fold_sizes": {f[0]: [len(f[1]), len(f[2]), len(f[3])] for f in fold_data},
            "summary": summary, "per_fold": per_fold, "differences": diffs}


def strip(rows):
    return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in rows.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/mnt/f/loci-eval/fly-sparse-index")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--spec-sha256", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sweep-seeds", type=int, default=3)
    ap.add_argument("--skip-sweeps", action="store_true")
    ap.add_argument("--skip-novelty", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    t_start = time.time()

    specs = {key: sei.load_spec(args.spec, expected_sha256=args.spec_sha256, matrix_key=key) for key in ("fw_right", "fw_left", "hb")}
    data = load_data(Path(args.cache))
    G = specs["fw_right"].n_glomeruli
    claws = int(round(specs["fw_right"].mean_claws))
    base = {"G": G, "K": 20 * G, "wta": 0.05, "claws": claws, "D": data["loci"]["x"].shape[1], "matrix": "fw_right"}
    seeds = list(range(args.seeds))
    sweep_seeds = list(range(args.sweep_seeds))
    primary_methods = ["simhash", "flyhash_random", "flyhash_random_fullD", "fly_measured_matrix", "fly_measured_sample",
                       "ctrl_measured_claws_uniform_glom", "ctrl_fixed_claws_measured_glom"]
    report = {
        "preregistration": "eval/fly_sparse_index/PREREGISTRATION.md",
        "operating_point": base,
        "specs": {k: {"n_glomeruli": s.n_glomeruli, "mean_claws_nonzero": round(s.mean_claws, 3),
                      "n_kc_with_claws": int((s.matrix.sum(axis=0) > 0).sum()), "provenance": s.provenance} for k, s in specs.items()},
        "env": {"python": platform.python_version(), "numpy": np.__version__, "hnswlib": getattr(hnswlib, "__version__", None) if hnswlib else None},
        "datasets": {k: {"n_index": int(v["x"].shape[0]), "n_queries": int(v["q"].shape[0])} for k, v in data.items() if k != "pool"},
    }
    with tempfile.TemporaryDirectory(dir=str(Path(args.cache) / "private")) as tmpdir:
        primary = {}
        for ds in ("loci", "nfcorpus"):
            print(f"[primary] {ds}", flush=True)
            rows = summarise_block(retrieval_block(data[ds], base, specs, primary_methods, seeds, tmpdir))
            diffs = {
                "fly_measured_matrix - flyhash_random": diff(rows, "fly_measured_matrix", "flyhash_random"),
                "fly_measured_sample - flyhash_random": diff(rows, "fly_measured_sample", "flyhash_random"),
                "fly_measured_matrix - simhash": diff(rows, "fly_measured_matrix", "simhash"),
                "fly_measured_matrix - flyhash_random_fullD": diff(rows, "fly_measured_matrix", "flyhash_random_fullD"),
                "fly_measured_matrix - flyhash_random (rerank100)": diff(rows, "fly_measured_matrix", "flyhash_random", "_per_query_rerank"),
                "fly_measured_matrix - simhash (rerank100)": diff(rows, "fly_measured_matrix", "simhash", "_per_query_rerank"),
                "ctrl_measured_claws_uniform_glom - flyhash_random": diff(rows, "ctrl_measured_claws_uniform_glom", "flyhash_random"),
                "ctrl_fixed_claws_measured_glom - flyhash_random": diff(rows, "ctrl_fixed_claws_measured_glom", "flyhash_random"),
            }
            if "hnsw" in rows:
                diffs["fly_measured_matrix - hnsw"] = diff(rows, "fly_measured_matrix", "hnsw")
            baselines = ["simhash", "flyhash_random", "flyhash_random_fullD"]
            best = max(baselines, key=lambda b: rows[b]["recall@10"][0])
            diffs[f"fly_measured_matrix - best_baseline({best})"] = diff(rows, "fly_measured_matrix", best)
            primary[ds] = {"rows": strip(rows), "differences": diffs, "best_memory_matched_baseline": best}
            primary[ds]["replicates"] = {}
            primary[ds]["_raw"] = rows
            print("  " + ", ".join(f"{m}={rows[m]['recall@10'][0]:.3f}" for m in rows), flush=True)
        # decisions (pre-registered: loci corpus, pure code search)
        d = primary["loci"]["differences"]
        best = primary["loci"]["best_memory_matched_baseline"]
        d1 = d["fly_measured_matrix - flyhash_random"]
        d2 = d[f"fly_measured_matrix - best_baseline({best})"]
        report["decision"] = {
            "D1_measured_vs_idealised": {"diff": d1, "adopt": bool(d1[0] > 0 and d1[1] > 0)},
            "D2_adopt_for_loci": {"best_baseline": best, "diff": d2, "adopt": bool(d2[0] > 0 and d2[1] > 0)},
        }
        # replicate differences vs the same flyhash_random per-query vector
        for ds in primary:
            raw = primary[ds].pop("_raw")
            for key in ("fw_left", "hb"):
                cfg = dict(base, matrix=key)
                blk = retrieval_block(data[ds], cfg, specs, ["fly_measured_matrix"], seeds, tmpdir, with_hnsw=False)
                r = summarise_block(blk)["fly_measured_matrix"]
                primary[ds]["replicates"][key] = {
                    "recall@10": r["recall@10"],
                    "minus_flyhash_random": boot_ci(r["_per_query_recall"] - raw["flyhash_random"]["_per_query_recall"], seed=1),
                }
        report["primary"] = primary

        if not args.skip_sweeps:
            print("[sweeps]", flush=True)
            sweeps = {"expansion": {}, "claws": {}, "wta": {}}
            sweep_methods = ["simhash", "flyhash_random", "fly_measured_matrix", "fly_measured_sample"]
            for f in (2, 5, 10, 20, 50):
                cfg = dict(base, K=f * G)
                rows = summarise_block(retrieval_block(data["loci"], cfg, specs, sweep_methods, sweep_seeds, tmpdir, with_hnsw=False))
                sweeps["expansion"][str(f)] = {**{m: rows[m]["recall@10"] for m in sweep_methods},
                                               "fly_measured_matrix - flyhash_random": diff(rows, "fly_measured_matrix", "flyhash_random"),
                                               "fly_measured_matrix - simhash": diff(rows, "fly_measured_matrix", "simhash")}
                print(f"  expansion {f}: " + ", ".join(f"{m}={rows[m]['recall@10'][0]:.3f}" for m in sweep_methods), flush=True)
            for c in (2, 4, 6, 7, 10, 15):
                cfg = dict(base, claws=c)
                ms = ["flyhash_random", "measured_glom_fixed_claws"]
                rows = summarise_block(retrieval_block(data["loci"], cfg, specs, ms, sweep_seeds, tmpdir, with_hnsw=False))
                sweeps["claws"][str(c)] = {**{m: rows[m]["recall@10"] for m in ms},
                                           "measured_glom - uniform_glom": diff(rows, "measured_glom_fixed_claws", "flyhash_random")}
                print(f"  claws {c}: " + ", ".join(f"{m}={rows[m]['recall@10'][0]:.3f}" for m in ms), flush=True)
            for w in (0.01, 0.02, 0.05, 0.10, 0.20):
                cfg = dict(base, wta=w)
                ms = ["flyhash_random", "fly_measured_matrix", "fly_measured_sample"]
                rows = summarise_block(retrieval_block(data["loci"], cfg, specs, ms, sweep_seeds, tmpdir, with_hnsw=False))
                sweeps["wta"][str(w)] = {**{m: rows[m]["recall@10"] for m in ms},
                                         "fly_measured_matrix - flyhash_random": diff(rows, "fly_measured_matrix", "flyhash_random")}
                print(f"  wta {w}: " + ", ".join(f"{m}={rows[m]['recall@10'][0]:.3f}" for m in ms), flush=True)
            report["sweeps"] = sweeps

        if not args.skip_novelty:
            print("[novelty]", flush=True)
            report["novelty"] = novelty_block(data["pool"], base, specs,
                                              ["simhash", "flyhash_random", "fly_measured_matrix", "fly_measured_sample"],
                                              list(range(args.sweep_seeds)), tmpdir)
    report["wall_s"] = round(time.time() - t_start, 1)
    (out / "results.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps(report["decision"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
