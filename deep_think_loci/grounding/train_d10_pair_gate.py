#!/usr/bin/env python3
"""Train and export the D10 pair-interaction gate from the frozen grounding dataset.

Reproduces the ``mlp_pair`` model that won the offline D10 comparison
(docs/d10_shadow_gate.md) and exports it for mcp/d10_gate.py, which runs it in
SHADOW mode only.

  1. Read grounding_dataset.jsonl (claim, evidence, label, signal; texts cut at 2000).
  2. Embed each unique text once (nomic-embed-text, unit-normalised), either from a
     local /v1/embeddings endpoint or from a precomputed (texts.json, embeddings.npy)
     pair whose text list must equal this dataset's.
  3. Recover the run of each finding. The rows carry no run id, so the run is read
     from the file's own structure: lineage (derived_from) pairs link findings of one
     deep-think run; a topic component whose members all sit in one lineage component
     (an *anchored* topic: the builder never pairs a synthesis finding across runs)
     pulls every finding it is paired with into that run. Groups of fewer than 5
     findings would be a train-only pool.
  4. Fit StandardScaler + MLPClassifier(64, relu, alpha 1e-3, adam, 200 iters, seed 0)
     on the topical pairs of every run, leaving one run out at a time; pool the
     out-of-fold scores and pick tau_R95, the largest threshold whose pooled OOF
     grounded recall is >= 0.95. Report bleed rejection at recall >= 0.95 (M1,
     negative-weighted mean over runs) for the model and for the 0.59 cosine rule.
  5. Refit on all topical pairs and export ``weights.npz`` (no pickle) and
     ``manifest.json`` (hashes, recipe, metrics, threshold table).

After exporting, set ``PINNED_MANIFEST_SHA256`` in mcp/d10_gate.py to the printed
manifest sha256; the gate refuses any other manifest.

Usage:
  python3 train_d10_pair_gate.py --loci-commit <sha> \
      [--dataset grounding_dataset.jsonl] [--out ../../mcp/models/d10_pair_gate] \
      (--embed-url http://127.0.0.1:11434 | --texts-json T --embeddings-npy E)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import urllib.request
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "mcp"))
import d10_gate as G  # noqa: E402  - the one feature definition, shared with inference

VERSION = "d10-mlp-pair-v1"
RECALL_FLOOR = 0.95
COSINE_RULE = 0.59
MIN_UNIT_FINDINGS = 5
EMBED_MODEL = "nomic-embed-text"
RECIPE = {"model": "StandardScaler + sklearn.neural_network.MLPClassifier", "hidden": 64,
          "activation": "relu", "alpha": 1e-3, "solver": "adam", "max_iter": 200,
          "early_stopping": False, "random_state": 0,
          "features": "[u, v, |u-v|, u*v] of unit-norm embeddings (u=claim, v=evidence) + "
                      + ", ".join(G.TABULAR_COLUMNS),
          "train_rows": "topical pairs of all runs"}
THRESHOLD_FLOORS = (0.90, 0.95, 0.975, 0.99)
CHECK_SEED = 20260926
CHECK_ROWS = 16
# The pre-registered offline comparison this artifact reproduces. Quoted, not recomputed:
# the intervals come from a 2,000-draw run-level bootstrap run there.
OFFLINE_RESULTS = {
    "source": "FlyBrain D10 (rjmendez/flybrain PR #21, docs/tasks/loci_d10.md section 3)",
    "protocol": "leave-one-run-out over 4 recovered runs, 1,811 topical test pairs",
    "cosine_rule_0.59": {"m1_tnr_at_r95": 0.949, "m1_ci95_run": [0.49, 0.985]},
    "mlp_pair": {"m1_tnr_at_r95": 0.987, "m1_ci95_run": [0.84, 1.00], "f1": 0.943,
                 "params": 203541, "p95_us_per_pair": 582, "ece_debiased": 0.033,
                 "lineage_recall": 0.503},
    "paired_m1_gain_over_cosine": {"point": 0.038, "ci95_run": [0.012, 0.377]},
    "caveats": ["labels are structural proxies (same dt_target tag = grounded), not judgments "
                "that evidence grounds a claim",
                "only 4 recovered runs; run-level intervals are wide",
                "lineage recall of strong models is 0.42-0.50: they learned target-tag agreement",
                "trained on finding-finding pairs; the live gate scores question-finding pairs"],
}


# --------------------------------------------------------------------------- data


def read_pairs(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = path.read_bytes()
    rows = []
    for n, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        r = json.loads(line)
        claim, evidence, label, signal = r.get("claim"), r.get("evidence"), r.get("label"), r.get("signal")
        if not isinstance(claim, str) or not isinstance(evidence, str) or not claim or not evidence:
            raise SystemExit(f"row {n}: claim and evidence must be non-empty strings")
        if label not in (0, 1) or isinstance(label, bool):
            raise SystemExit(f"row {n}: label must be 0 or 1")
        rows.append({"claim": claim[:G.MAX_TEXT_CHARS], "evidence": evidence[:G.MAX_TEXT_CHARS],
                     "label": int(label), "signal": str(signal)})
    return rows, hashlib.sha256(raw).hexdigest()


def index_texts(rows: Sequence[dict[str, Any]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    texts = sorted({r["claim"] for r in rows} | {r["evidence"] for r in rows})
    pos = {t: i for i, t in enumerate(texts)}
    return (texts, np.array([pos[r["claim"]] for r in rows], dtype=np.int64),
            np.array([pos[r["evidence"]] for r in rows], dtype=np.int64))


def embed_endpoint(texts: Sequence[str], base_url: str, batch: int = 16) -> np.ndarray:
    url = base_url.rstrip("/") + "/v1/embeddings"
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        body = json.dumps({"model": EMBED_MODEL, "input": [t[:G.MAX_TEXT_CHARS] for t in texts[i:i + batch]]}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())["data"]
        out += [d["embedding"] for d in sorted(data, key=lambda d: d.get("index", 0))]
    return np.asarray(out, dtype=np.float32)


def embed_precomputed(texts: Sequence[str], texts_json: Path, emb_npy: Path) -> np.ndarray:
    cached = json.loads(Path(texts_json).read_text())
    if list(cached) != list(texts):
        raise SystemExit(f"{texts_json}: its text list is not this dataset's (sorted unique, cut at 2000)")
    emb = np.load(emb_npy, allow_pickle=False)
    if emb.shape[0] != len(texts):
        raise SystemExit(f"{emb_npy}: {emb.shape[0]} rows for {len(texts)} texts")
    return emb


# --------------------------------------------------------------------------- run structure


def components(n: int, edges) -> np.ndarray:
    """Connected-component label per node, numbered by size (largest 0), ties by min node."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    uniq, counts = np.unique(roots, return_counts=True)
    order = sorted(range(len(uniq)), key=lambda i: (-counts[i], uniq[i]))
    relabel = {int(uniq[i]): k for k, i in enumerate(order)}
    return np.array([relabel[int(r)] for r in roots], dtype=np.int64)


def recover_units(n_texts: int, a: np.ndarray, b: np.ndarray, label: np.ndarray,
                  signal: np.ndarray, min_unit: int = MIN_UNIT_FINDINGS) -> np.ndarray:
    """Run proxy per finding (0..K-1, largest first) or -1 for the train-only pool."""
    topic = components(n_texts, [(x, y) for x, y, l, s in zip(a, b, label, signal) if s == "topical" and l == 1])
    lineage = components(n_texts, [(x, y) for x, y, s in zip(a, b, signal) if s == "lineage"])
    major = set(np.flatnonzero(np.bincount(lineage) >= min_unit).tolist())
    anchored = {}
    for t in np.unique(topic):
        members = np.flatnonzero(topic == t)
        homes = set(lineage[members].tolist())
        if len(members) >= 2 and len(homes) == 1:
            anchored[int(t)] = homes.pop()
    edges = [(int(lineage[q]), anchored[int(topic[p])])
             for x, y in zip(a, b) for p, q in ((x, y), (y, x)) if int(topic[p]) in anchored]
    joined = components(int(lineage.max()) + 1, edges)
    groups: dict[int, set[int]] = {}
    for c in major:
        groups.setdefault(int(joined[c]), set()).add(c)
    if any(len(g) > 1 for g in groups.values()):
        raise SystemExit("anchored topics join two runs; the run proxy does not hold for this dataset")
    runs = joined[lineage]
    sizes = np.bincount(runs)
    keep = [c for c in range(len(sizes)) if sizes[c] >= min_unit]
    remap = {c: k for k, c in enumerate(keep)}
    return np.array([remap.get(int(c), -1) for c in runs], dtype=np.int64)


# --------------------------------------------------------------------------- metrics


def tnr_at_recall(y: np.ndarray, s: np.ndarray, floor: float = RECALL_FLOOR) -> float:
    """Bleed rejection at the best threshold whose grounded recall is >= floor (keep = s >= t)."""
    y, s = np.asarray(y), np.asarray(s, dtype=np.float64)
    pos, neg = float((y == 1).sum()), float((y == 0).sum())
    if not pos or not neg:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ss, yy = s[order], y[order]
    last = np.r_[ss[1:] != ss[:-1], True]
    recall = np.cumsum(yy == 1)[last] / pos
    tnr = 1.0 - np.cumsum(yy == 0)[last] / neg
    return float(tnr[recall >= floor - 1e-12].max())


def threshold_for_recall(y: np.ndarray, s: np.ndarray, floor: float = RECALL_FLOOR) -> float:
    """Largest threshold t with recall(s >= t) >= floor."""
    pos = np.sort(np.asarray(s, dtype=np.float64)[np.asarray(y) == 1])[::-1]
    k = int(math.ceil(floor * len(pos) - 1e-9))
    return float(pos[max(k, 1) - 1])


def rates(y: np.ndarray, keep: np.ndarray) -> dict[str, float]:
    y, keep = np.asarray(y), np.asarray(keep, dtype=bool)
    tp, fp = int(((y == 1) & keep).sum()), int(((y == 0) & keep).sum())
    fn = int(((y == 1) & ~keep).sum())
    return {"recall": tp / max(1, int((y == 1).sum())),
            "tnr": 1.0 - fp / max(1, int((y == 0).sum())),
            "f1": 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)}


def m1_by_run(y: np.ndarray, s: np.ndarray, unit: np.ndarray) -> tuple[float, dict[int, float]]:
    """Negative-weighted mean over runs of per-run M1 (the offline point estimate)."""
    per, num, den = {}, 0.0, 0.0
    for u in np.unique(unit):
        sel = unit == u
        per[int(u)] = tnr_at_recall(y[sel], s[sel])
        n_neg = float((y[sel] == 0).sum())
        if not math.isnan(per[int(u)]):
            num += n_neg * per[int(u)]
            den += n_neg
    return num / den, per


# --------------------------------------------------------------------------- model


def make_model():
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(RECIPE["hidden"],), activation="relu", alpha=RECIPE["alpha"], solver="adam",
        max_iter=RECIPE["max_iter"], early_stopping=False, random_state=RECIPE["random_state"]))


def fit(X: np.ndarray, y: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # 200 adam iterations do not converge; the recipe fixes 200
        return make_model().fit(X, y)


def weights_of(model) -> dict[str, np.ndarray]:
    scaler, mlp = model.steps[0][1], model.steps[1][1]
    if len(mlp.coefs_) != 2 or mlp.activation != "relu" or mlp.out_activation_ != "logistic":
        raise SystemExit("export supports one ReLU hidden layer with a logistic output only")
    return {"scaler_mean": np.asarray(scaler.mean_, dtype=np.float64),
            "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64),
            "W1": np.asarray(mlp.coefs_[0], dtype=np.float64), "b1": np.asarray(mlp.intercepts_[0], dtype=np.float64),
            "W2": np.asarray(mlp.coefs_[1], dtype=np.float64), "b2": np.asarray(mlp.intercepts_[1], dtype=np.float64)}


def check_batch(n_features: int, embed_dim: int) -> np.ndarray:
    """A fixed, data-free feature batch whose sklearn probabilities the manifest records."""
    rng = np.random.default_rng(CHECK_SEED)
    u = G.unit_rows(rng.normal(size=(CHECK_ROWS, embed_dim)))
    v = G.unit_rows(u + rng.normal(scale=0.8, size=(CHECK_ROWS, embed_dim)))
    ta = [f"check {i} not 42" for i in range(CHECK_ROWS)]
    tb = [f"check {i} value {i * 7}" for i in range(CHECK_ROWS)]
    X = G.pair_features(u, v, ta, tb)
    assert X.shape[1] == n_features
    return X


def export(model, out_dir: Path, meta: dict[str, Any]) -> tuple[str, str]:
    """Write weights.npz + manifest.json. Returns (manifest sha256, weights sha256)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    w = weights_of(model)
    buf = io.BytesIO()
    np.savez_compressed(buf, **w)
    data = buf.getvalue()
    (out_dir / G.WEIGHTS_NAME).write_bytes(data)
    n_in, hidden = w["W1"].shape
    Xc = check_batch(n_in, meta["embed_dim"])
    manifest = {
        "schema": G.ARTIFACT_SCHEMA,
        "version": VERSION,
        "feature_dim": int(n_in),
        "n_params": int(sum(v.size for v in w.values())),
        "weights": {"file": G.WEIGHTS_NAME, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)},
        "check": {"seed": CHECK_SEED, "rows": CHECK_ROWS,
                  "sklearn_proba": [float(p) for p in model.predict_proba(Xc)[:, 1]]},
        **meta,
    }
    manifest["recipe"] = {**manifest["recipe"], "hidden": int(hidden)}
    raw = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    (out_dir / G.MANIFEST_NAME).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), manifest["weights"]["sha256"]


# --------------------------------------------------------------------------- main


def train(rows, texts, a, b, emb, *, log=print) -> dict[str, Any]:
    label = np.array([r["label"] for r in rows], dtype=np.int64)
    signal = np.array([r["signal"] for r in rows])
    unit = recover_units(len(texts), a, b, label, signal)
    n_units = int(unit.max()) + 1
    log(f"runs recovered: {n_units}, findings per run {np.bincount(unit[unit >= 0]).tolist()}, "
        f"pool {int((unit < 0).sum())}")
    top = np.flatnonzero(signal == "topical")
    X = G.pair_features(emb[a[top]], emb[b[top]], [texts[i] for i in a[top]], [texts[i] for i in b[top]])
    y = label[top]
    cos = X[:, 4 * emb.shape[1]]
    ua, ub = unit[a[top]], unit[b[top]]
    oof_rows, oof_s, oof_unit = [], [], []
    for t in range(n_units):
        test = np.flatnonzero((ua == t) & (ub == t))
        train_rows = np.flatnonzero((ua != t) & (ub != t))
        m = fit(X[train_rows], y[train_rows])
        oof_rows.append(test)
        oof_s.append(m.predict_proba(X[test])[:, 1])
        oof_unit.append(np.full(len(test), t))
    r_ = np.concatenate(oof_rows)
    s_, u_ = np.concatenate(oof_s), np.concatenate(oof_unit)
    y_, c_ = y[r_], cos[r_]
    tau = threshold_for_recall(y_, s_, RECALL_FLOOR)
    m1_mlp, per_mlp = m1_by_run(y_, s_, u_)
    m1_cos, per_cos = m1_by_run(y_, c_, u_)
    table = []
    for floor in THRESHOLD_FLOORS:
        t = threshold_for_recall(y_, s_, floor)
        table.append({"recall_floor": floor, "tau": t, **{f"oof_{k}": v for k, v in rates(y_, s_ >= t).items()}})
    log(f"LORO OOF on {len(r_)} topical pairs: M1 mlp {m1_mlp:.3f} {per_mlp} | cosine {m1_cos:.3f} {per_cos}")
    log(f"tau_R95 = {tau:.6f}; OOF at tau {rates(y_, s_ >= tau)}; cosine 0.59 {rates(y_, c_ >= COSINE_RULE)}")
    final = fit(X, y)
    return {"model": final, "tau": tau, "n_units": n_units, "unit_sizes": np.bincount(unit[unit >= 0]).tolist(),
            "n_train_pairs": int(len(y)), "n_train_pos": int(y.sum()),
            "reproduced": {"n_oof_pairs": int(len(r_)),
                           "mlp_m1_tnr_at_r95": m1_mlp, "mlp_m1_per_run": per_mlp,
                           "cosine_m1_tnr_at_r95": m1_cos, "cosine_m1_per_run": per_cos,
                           "oof_at_tau": rates(y_, s_ >= tau),
                           "cosine_0.59_oof": rates(y_, c_ >= COSINE_RULE)},
            "threshold_table": table}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", type=Path, default=HERE / "grounding_dataset.jsonl")
    ap.add_argument("--out", type=Path, default=REPO / "mcp" / "models" / "d10_pair_gate")
    ap.add_argument("--loci-commit", required=True, help="commit the dataset was read at")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--embed-url", help="Ollama base URL serving nomic-embed-text")
    src.add_argument("--texts-json", type=Path, help="precomputed: sorted unique texts (JSON list)")
    ap.add_argument("--embeddings-npy", type=Path, help="precomputed: embeddings aligned with --texts-json")
    ap.add_argument("--embedding-digest", default=None, help="embedder digest to record (e.g. from /api/tags)")
    args = ap.parse_args(argv)

    rows, data_sha = read_pairs(args.dataset)
    texts, a, b = index_texts(rows)
    if args.embed_url:
        emb = embed_endpoint(texts, args.embed_url)
        emb_source = "endpoint"
    else:
        if not args.embeddings_npy:
            ap.error("--texts-json needs --embeddings-npy")
        emb = embed_precomputed(texts, args.texts_json, args.embeddings_npy)
        emb_source = "precomputed"
    emb = G.unit_rows(emb)
    emb_sha = hashlib.sha256(np.ascontiguousarray(emb.astype(np.float32)).tobytes()).hexdigest()
    print(f"dataset {args.dataset.name}: {len(rows)} rows, {len(texts)} findings, sha256 {data_sha}")
    res = train(rows, texts, a, b, emb)
    import sklearn
    meta = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "embed_dim": int(emb.shape[1]),
        "embedding": {"model": EMBED_MODEL, "normalised": True, "max_chars": G.MAX_TEXT_CHARS,
                      "source": emb_source, "digest": args.embedding_digest,
                      "float32_sha256": emb_sha},
        "training_data": {"repo": "rjmendez/loci", "path": "deep_think_loci/grounding/grounding_dataset.jsonl",
                          "loci_commit": args.loci_commit, "sha256": data_sha, "n_rows": len(rows),
                          "n_findings": len(texts), "n_runs": res["n_units"], "run_sizes": res["unit_sizes"],
                          "n_train_pairs": res["n_train_pairs"], "n_train_pos": res["n_train_pos"],
                          "label_note": "structural proxy: topical = same dt_target tag"},
        "recipe": {**RECIPE, "sklearn": sklearn.__version__, "numpy": np.__version__},
        "threshold": {"name": "tau_R95", "tau": res["tau"], "recall_floor": RECALL_FLOOR,
                      "chosen_on": "pooled leave-one-run-out out-of-fold scores (topical pairs)"},
        "threshold_table": res["threshold_table"],
        "loro_reproduced_here": res["reproduced"],
        "loro_offline_results": OFFLINE_RESULTS,
    }
    msha, wsha = export(res["model"], args.out, meta)
    print(f"exported {args.out}: weights sha256 {wsha}")
    print(f"manifest sha256 {msha}  <- set PINNED_MANIFEST_SHA256 in mcp/d10_gate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
