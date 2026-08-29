"""Retrain the RAG-bleed grounding classifier with richer features and OOS-aware model selection."""
import argparse
import glob
import hashlib
import json
import os
import pathlib
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np


def _resolve_backends() -> None:
    """Fill OLLAMA_BASE_URL/EMBED_MODEL from ~/.loci/backends.toml when unset.

    Run standalone or from cron this inherits no MCP environment, and nothing
    listens on the localhost default. backends.py already reads the config file
    that records the real endpoint; without this the run fails at the first
    embedding call with a bare connection-refused.

    Called from __main__ only. It writes to os.environ, which is right for a run
    and wrong on import — a test that imports this module would otherwise change
    what every other module in the process resolves."""
    try:
        import sys as _sys
        _repo = pathlib.Path(__file__).resolve().parents[2]
        _sys.path.insert(0, str(_repo / "mcp"))
        import backends
        backends.load_env(_repo)
    except Exception:
        pass  # a missing config is not a reason to refuse to run


# Folds are independent, so they run in parallel. Serial 10-fold
# GradientBoosting over 12,684 x 1,540 took over 45 minutes on a 28-core box —
# long enough that the loop's own 3600s step bound would cut the nightly before
# it finished. Measured 5.2x here (not 10x: GBC is single-threaded per fold, the
# folds do not balance evenly, and BLAS threads contend). Parallelism lives at
# the CV level only; giving the estimators n_jobs as well just oversubscribes.
CV_JOBS = int(os.environ.get("LOCI_TRAIN_CV_JOBS", "-1"))


def _emb_model():
    """Read at call time: _resolve_backends() runs after import."""
    return os.environ.get("EMBED_MODEL") or "nomic-embed-text"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLF_PATH = os.path.join(REPO_ROOT, "deep_think_loci", "grounding", "grounding_bleed_clf.joblib")
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".emb_cache.npz")


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        data = np.load(CACHE_PATH, allow_pickle=True)
        return {k: data[k] for k in data.files}
    return {}


# Cap the on-disk embedding cache. Every unseen text adds a 768-float entry and
# nothing ever removed one, so the file grew for the life of the checkout.
# Bounded rather than rotated: the cache is a pure speedup — a miss costs one
# re-embed, never a wrong answer — so dropping the oldest entries is free.
EMB_CACHE_MAX = int(os.environ.get("LOCI_EMB_CACHE_MAX", "50000"))


def _save_cache(cache: dict) -> None:
    if len(cache) > EMB_CACHE_MAX:
        # dict preserves insertion order, so this keeps the most recent entries
        keep = list(cache)[-EMB_CACHE_MAX:]
        cache = {k: cache[k] for k in keep}
    np.savez_compressed(CACHE_PATH, **cache)


def embed_texts(texts: list, ollama_base: str, cache: dict) -> np.ndarray:
    url = ollama_base.rstrip("/") + "/v1/embeddings"
    need = [t for t in texts if _sha(t) not in cache]
    for i in range(0, len(need), 16):
        batch = [t[:2000] for t in need[i:i + 16]]
        body = json.dumps({"model": _emb_model(), "input": batch}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
        for j, d in enumerate(resp["data"]):
            cache[_sha(need[i + j])] = np.array(d["embedding"], dtype=np.float32)
    out = np.array([cache[_sha(t)] for t in texts], dtype=np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    return out / norms


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _token_overlap(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _len_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    return min(la, lb) / (max(la, lb) + 1)


def make_features(claims: list, evidences: list, emb_claims: np.ndarray, emb_evidences: np.ndarray) -> np.ndarray:
    diff = np.abs(emb_claims - emb_evidences)
    prod = emb_claims * emb_evidences
    cos = (emb_claims * emb_evidences).sum(axis=1, keepdims=True)
    cos_sq = cos ** 2
    lr = np.array([[_len_ratio(c, e)] for c, e in zip(claims, evidences)], dtype=np.float32)
    jac = np.array([[_token_overlap(c, e)] for c, e in zip(claims, evidences)], dtype=np.float32)
    return np.concatenate([diff, prod, cos, cos_sq, lr, jac], axis=1)


# ---------------------------------------------------------------------------
# Cosine baseline F1 (best threshold on same fold split)
# ---------------------------------------------------------------------------

def cosine_f1_on_folds(cos_scores: np.ndarray, labels: np.ndarray, fold_indices: list) -> tuple:
    from sklearn.metrics import f1_score
    per_fold = []
    thresholds = np.linspace(0.2, 0.9, 71)
    for train_idx, val_idx in fold_indices:
        cos_val = cos_scores[val_idx]
        y_val = labels[val_idx]
        best_f1 = max(
            f1_score(y_val, (cos_val > t).astype(int), zero_division=0)
            for t in thresholds
        )
        per_fold.append(best_f1)
    return float(np.mean(per_fold)), float(np.std(per_fold))


# ---------------------------------------------------------------------------
# OOS evaluation from findings glob
# ---------------------------------------------------------------------------

def oos_from_findings(
    findings_glob: str,
    cache: dict,
    ollama_base: str,
    candidates: dict,
) -> dict:
    import itertools
    from sklearn.metrics import f1_score

    files = glob.glob(os.path.expanduser(findings_glob))
    if len(files) < 2:
        print(f"[oos] found {len(files)} runs — need >=2; skipping true OOS pass")
        return {}

    runs = {}
    for f in files:
        run_id = f.split("/")[-2]
        recs = []
        for line in open(f, errors="ignore"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            text = (r.get("text") or "")[:2000]
            topic = _extract_topic(r)
            if topic and text:
                recs.append({"text": text, "topic": topic})
        if recs:
            runs[run_id] = recs

    run_ids = sorted(runs)
    all_texts = sorted({x["text"] for recs in runs.values() for x in recs})
    print(f"[oos] embedding {len(all_texts)} unique texts across {len(run_ids)} runs...")
    embs_arr = embed_texts(all_texts, ollama_base, cache)
    emb_map = {t: embs_arr[i] for i, t in enumerate(all_texts)}

    def feat_pair(ta, tb):
        va, vb = emb_map[ta], emb_map[tb]
        cos = float(va @ vb)
        c_arr = np.array([[cos]], dtype=np.float32)
        diff = np.abs(va - vb).reshape(1, -1)
        prod = (va * vb).reshape(1, -1)
        lr = np.array([[_len_ratio(ta, tb)]], dtype=np.float32)
        jac = np.array([[_token_overlap(ta, tb)]], dtype=np.float32)
        return np.concatenate([diff, prod, c_arr, c_arr ** 2, lr, jac], axis=1)[0], cos

    model_f1s = {name: [] for name in candidates}
    cos_f1s = []

    for held in run_ids:
        train_recs = [(x["text"], x["topic"]) for rid in run_ids if rid != held for x in runs[rid]]
        test_recs = runs[held]

        X_tr, Y_tr = [], []
        for (ta, ga), (tb, gb) in itertools.combinations(train_recs, 2):
            f_vec, _ = feat_pair(ta, tb)
            X_tr.append(f_vec)
            Y_tr.append(1 if ga == gb else 0)
        if len(set(Y_tr)) < 2:
            continue

        X_tr_arr = np.array(X_tr, dtype=np.float32)
        Y_tr_arr = np.array(Y_tr, dtype=np.int32)

        fold_clfs = {}
        for name, clf_proto in candidates.items():
            import copy
            c = copy.deepcopy(clf_proto)
            print(f"  [oos] held={held} fitting {name} on {X_tr_arr.shape[0]} pairs...",
                  flush=True)
            c.fit(X_tr_arr, Y_tr_arr)
            fold_clfs[name] = c

        topics = sorted({x["topic"] for recs in runs.values() for x in recs})
        X_te, Y_te, cos_te = [], [], []
        for topic in topics:
            topic_emb = np.mean([emb_map[x["text"]] for x in train_recs if x[1] == topic] or
                                [np.zeros(embs_arr.shape[1])], axis=0)
            norm = np.linalg.norm(topic_emb) + 1e-9
            topic_emb /= norm
            for x in test_recs:
                fv, cos = feat_pair(topic_emb if False else x["text"], x["text"])
                # use cross-finding similarity directly
                pass
        # Simpler: all pairs in test findings
        test_pairs = list(itertools.combinations(test_recs, 2))
        if not test_pairs:
            continue
        for xa, xb in test_pairs:
            fv, cos = feat_pair(xa["text"], xb["text"])
            X_te.append(fv)
            Y_te.append(1 if xa["topic"] == xb["topic"] else 0)
            cos_te.append(cos)

        if not Y_te or len(set(Y_te)) < 2:
            continue
        X_te_arr = np.array(X_te, dtype=np.float32)

        best_cos_f1 = max(
            f1_score(Y_te, (np.array(cos_te) > t).astype(int), zero_division=0)
            for t in np.linspace(0.2, 0.9, 71)
        )
        cos_f1s.append(best_cos_f1)
        for name, c in fold_clfs.items():
            proba = c.predict_proba(X_te_arr)[:, 1]
            mf1 = f1_score(Y_te, (proba > 0.5).astype(int), zero_division=0)
            model_f1s[name].append(mf1)
            print(f"  [oos] held={held} {name} f1={mf1:.3f} | cosine f1={best_cos_f1:.3f}",
                  flush=True)

    if not cos_f1s:
        return {}

    return {
        name: (float(np.mean(v)), float(np.std(v))) if v else (0.0, 0.0)
        for name, v in model_f1s.items()
    } | {"__cosine__": (float(np.mean(cos_f1s)), float(np.std(cos_f1s)))}


def _extract_topic(rec: dict) -> str:
    import re
    tags = rec.get("tags") or []
    tg = {t.split(":", 1)[0]: t.split(":", 1)[1] for t in tags if ":" in t}
    if tg.get("dt_target"):
        return tg["dt_target"]
    if tg.get("dt_phase") == "bench":
        m = re.match(r"\s*([^:]{3,40}):", rec.get("text", ""))
        return ("bench:" + m.group(1).strip().lower()) if m else "bench:misc"
    if tg.get("dt_phase") in ("final", "adversarial"):
        return "synthesis:" + tg["dt_phase"]
    return tg.get("dt_phase", "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train grounding bleed classifier")
    ap.add_argument("--dataset", default=os.path.join(REPO_ROOT, "deep_think_loci", "grounding", "grounding_dataset.jsonl"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "mlops", "grounding", "train_metrics.json"))
    ap.add_argument("--findings-glob", default=None)
    ap.add_argument("--ollama", default=None,
                    help="Ollama base URL. Default: $OLLAMA_BASE_URL, else "
                         "~/.loci/backends.toml, else http://localhost:11434")
    ap.add_argument("--dry-run", action="store_true", help="Eval only; never write the joblib")
    ap.add_argument("--candidate-out", default=None,
                    help="Write winning candidate model here instead of live path (used by loop.py)")
    args = ap.parse_args()
    _resolve_backends()
    args.ollama = args.ollama or os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    import joblib

    rows = []
    with open(args.dataset) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    from collections import Counter
    sigs = Counter(r.get("signal", "?") for r in rows)
    labels_arr = [r["label"] for r in rows]
    print(f"Loaded {len(rows)} rows | pos={sum(labels_arr)} neg={len(labels_arr)-sum(labels_arr)} | signals={dict(sigs)}")

    if args.dry_run:
        print("DRY RUN — dataset stats only, no embedding or training.")
        return

    cache = _load_cache()
    claims = [r["claim"] for r in rows]
    evidences = [r["evidence"] for r in rows]
    labels = np.array([r["label"] for r in rows], dtype=np.int32)
    cos_scores = np.array([r.get("cos", 0.0) for r in rows], dtype=np.float32)

    all_texts = list(dict.fromkeys(claims + evidences))
    print(f"Embedding {len(all_texts)} unique texts (cache has {len(cache)} entries)...",
          flush=True)
    embs = embed_texts(all_texts, args.ollama, cache)
    _save_cache(cache)
    print(f"Cache saved to {CACHE_PATH}")

    text_to_emb = {t: embs[i] for i, t in enumerate(all_texts)}
    emb_claims = np.array([text_to_emb[c] for c in claims], dtype=np.float32)
    emb_evidences = np.array([text_to_emb[e] for e in evidences], dtype=np.float32)

    X = make_features(claims, evidences, emb_claims, emb_evidences)
    print(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} dims")

    candidates = {
        "LogisticRegression": LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300, max_depth=6, class_weight="balanced", random_state=42
        ),
    }

    cos_quartiles = np.digitize(cos_scores, np.percentile(cos_scores, [25, 50, 75]))
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_indices = list(skf.split(X, cos_quartiles))

    cos_cv_mean, cos_cv_std = cosine_f1_on_folds(cos_scores, labels, fold_indices)
    # These folds split PAIRS, so a finding lands on both sides of a split and
    # every number below is optimistic. Measured on this corpus: GBC scores 0.944
    # here and 0.908 when findings are split before pairs are built, against a
    # cosine baseline that barely moves (0.864 -> 0.878). Two thirds of the
    # apparent margin over cosine is the leak. Say so on every line, because this
    # is the figure that gets quoted later as though it were the model's score.
    print(f"\nPair-level 10-fold CV (optimistic — a finding appears in train and test):")
    print(f"  cosine baseline: F1 = {cos_cv_mean:.3f} ± {cos_cv_std:.3f}")

    cv_results = {}
    for name, clf in candidates.items():
        # Announce before fitting, not after. GradientBoosting over this feature
        # matrix is 2,000 trees x 10 folds and runs for tens of minutes; printing
        # only on completion leaves an unattended run silent and indistinguishable
        # from hung.
        print(f"  {name}: fitting 10 folds on {X.shape[0]}x{X.shape[1]} "
              f"(n_jobs={CV_JOBS})...", flush=True)
        t0 = time.monotonic()
        proba = cross_val_predict(
            clf, X, labels,
            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
            method="predict_proba",
            n_jobs=CV_JOBS,
        )[:, 1]
        elapsed = time.monotonic() - t0
        per_fold_f1 = []
        for _, val_idx in fold_indices:
            pv = proba[val_idx]
            yv = labels[val_idx]
            per_fold_f1.append(f1_score(yv, (pv > 0.5).astype(int), zero_division=0))
        mean_f1 = float(np.mean(per_fold_f1))
        std_f1 = float(np.std(per_fold_f1))
        cv_results[name] = {"mean": mean_f1, "std": std_f1, "fit_seconds": round(elapsed, 1)}
        print(f"  {name}: CV F1 = {mean_f1:.3f} ± {std_f1:.3f}  ({elapsed:.0f}s)", flush=True)

    best_name = max(cv_results, key=lambda n: cv_results[n]["mean"])
    best_f1 = cv_results[best_name]["mean"]
    print(f"\nBest pair-level CV model: {best_name} (F1={best_f1:.3f}, optimistic)")

    oos_results = {}
    if args.findings_glob:
        print(f"\nRunning true leave-one-run-out OOS with glob: {args.findings_glob}")
        oos_results = oos_from_findings(args.findings_glob, cache, args.ollama, candidates)
        _save_cache(cache)
        if oos_results:
            cos_oos_mean, cos_oos_std = oos_results.pop("__cosine__", (cos_cv_mean, cos_cv_std))
            print(f"\nOOS cosine baseline: F1 = {cos_oos_mean:.3f} ± {cos_oos_std:.3f}")
            for name, (mf, sf) in oos_results.items():
                print(f"  OOS {name}: F1 = {mf:.3f} ± {sf:.3f}")
            best_name = max(oos_results, key=lambda n: oos_results[n][0])
            best_f1 = oos_results[best_name][0]
            eval_baseline = cos_oos_mean
            print(f"\nBest OOS model: {best_name} (F1={best_f1:.3f})")
        else:
            eval_baseline = cos_cv_mean
    else:
        eval_baseline = cos_cv_mean

    beat_baseline = best_f1 > eval_baseline
    decision = "PROMOTE" if beat_baseline else "HOLD"
    basis = "leave-one-run-out OOS" if oos_results else "pair-level CV (OPTIMISTIC)"
    print(f"\n{'='*60}")
    print(f"Decision: {decision}   [on {basis}]")
    print(f"  Best model F1:    {best_f1:.3f}  ({best_name})")
    print(f"  Cosine baseline:  {eval_baseline:.3f}")
    print(f"  Beat baseline:    {beat_baseline}")
    if not oos_results:
        print("  WARNING: no --findings-glob, so this decision rests on the leaky")
        print("           pair-level split. Expect ~2/3 of the margin to be leak.")
    print(f"{'='*60}\n")

    best_clf = candidates[best_name]
    print(f"Refitting {best_name} on all {X.shape[0]} rows...", flush=True)
    t0 = time.monotonic()
    best_clf.fit(X, labels)
    print(f"  refit done in {time.monotonic() - t0:.0f}s", flush=True)

    all_proba = best_clf.predict_proba(X)[:, 1]
    thresholds = np.linspace(0.2, 0.9, 71)
    best_thresh = float(max(thresholds, key=lambda t: f1_score(labels, (all_proba > t).astype(int), zero_division=0)))

    write_path = args.candidate_out if args.candidate_out else CLF_PATH
    if not args.dry_run and decision == "PROMOTE":
        os.makedirs(os.path.dirname(os.path.abspath(write_path)), exist_ok=True)
        joblib.dump(best_clf, write_path)
        print(f"Model written to {write_path}")
    elif args.dry_run:
        print("--dry-run: skipping joblib write")
    else:
        print(f"HOLD: not overwriting {CLF_PATH}")

    metrics = {
        "model": best_name,
        "cv_f1_mean": round(cv_results[best_name]["mean"], 4),
        "cv_f1_std": round(cv_results[best_name]["std"], 4),
        "cosine_baseline_cv_f1": round(cos_cv_mean, 4),
        "beat_baseline": beat_baseline,
        "decision": decision,
        "decision_basis": "oos_leave_one_run_out" if oos_results else "pair_level_cv_optimistic",
        "cv_is_pair_level_and_optimistic": True,
        "n_train": len(rows),
        "feature_dim": X.shape[1],
        "threshold": round(best_thresh, 4),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "all_models": {
            name: {"cv_f1_mean": round(r["mean"], 4), "cv_f1_std": round(r["std"], 4)}
            for name, r in cv_results.items()
        },
    }
    if oos_results:
        metrics["oos_results"] = {
            name: {"oos_f1_mean": round(v[0], 4), "oos_f1_std": round(v[1], 4)}
            for name, v in oos_results.items()
        }

    with open(args.out, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Metrics written to {args.out}")


if __name__ == "__main__":
    main()
