#!/usr/bin/env python3
"""Build the grounding dataset + train the RAG-bleed specialist from a Loci corpus.

Reproducible builder for grounding_dataset.jsonl + grounding_bleed_clf.joblib +
metrics.json. Harvests labeled pairs from deep-think-loci investigation findings:

  topical      same dt_target = grounded (1) / cross-target = bleed (0)
  lineage      a finding paired with a derived_from parent it built on (1)
  hallucination an ungrounded synthesis vs the real corpus (0, hard negative)

The topical task is what the shipped classifier trains on (the others are too
sparse to train yet — accumulate via more runs). On the current corpus a tuned
cosine threshold (~0.59 per-target) is competitive with the LR; the model is the
drop-in upgrade for when the corpus grows (high CV AUC = headroom).

Usage:
  python3 build_grounding_dataset.py \
    --findings ~/.hermes/memory-sessions/dt-loci-*/findings.jsonl \
    --out . [--ollama http://ollama-host:11434/v1/embeddings]
"""
import argparse, glob, itertools, json, os, pathlib, random, re, urllib.request
from collections import Counter
import numpy as np

def _resolve_backends() -> None:
    """Fill OLLAMA_BASE_URL/EMBED_MODEL from ~/.loci/backends.toml when unset.

    Run standalone or from cron this inherits no MCP environment, and nothing
    listens on ollama.internal or the localhost default. backends.py already
    reads the config file that records the real endpoint; without this the run
    dies at the first embedding call with a bare name-resolution error.

    Called from main(), not on import: it writes to os.environ, which is right
    for a run and wrong on import — a test that imports this module would
    otherwise change what every other module in the process resolves."""
    try:
        import sys as _sys
        _repo = pathlib.Path(__file__).resolve().parents[2]
        _sys.path.insert(0, str(_repo / "mcp"))
        import backends
        backends.load_env(_repo)
    except Exception:
        pass  # a missing config is not a reason to refuse to run


def _emb_model():
    """Read at call time: _resolve_backends() runs after import."""
    return os.environ.get("EMBED_MODEL") or "nomic-embed-text"


def _default_ollama():
    base = os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434"
    return base.rstrip("/") + "/v1/embeddings"


def topic_of(rec):
    tags = rec.get("tags") or []
    tg = {t.split(":", 1)[0]: t.split(":", 1)[1] for t in tags if ":" in t}
    if tg.get("dt_target"):
        return tg["dt_target"]
    if tg.get("dt_phase") == "bench":
        m = re.match(r"\s*([^:]{3,40}):", rec.get("text", ""))
        return ("bench:" + m.group(1).strip().lower()) if m else "bench:misc"
    if tg.get("dt_phase") in ("final", "adversarial"):
        return "synthesis:" + tg["dt_phase"]
    return tg.get("dt_phase", "unknown")


def embed(texts, url):
    out = []
    for i in range(0, len(texts), 16):
        body = json.dumps({"model": _emb_model(),
                           "input": [t[:2000] for t in texts[i:i + 16]]}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        out += [d["embedding"] for d in json.loads(urllib.request.urlopen(req, timeout=60).read())["data"]]
    v = np.array(out, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--findings", nargs="+", required=True, help="findings.jsonl glob(s)")
    ap.add_argument("--out", default=".")
    ap.add_argument("--ollama", default=None,
                    help="embeddings endpoint (default: $OLLAMA_BASE_URL, else "
                         "~/.loci/backends.toml, else localhost:11434, + /v1/embeddings)")
    ap.add_argument("--neg-ratio", type=int, default=2)
    a = ap.parse_args()
    _resolve_backends()
    a.ollama = a.ollama or _default_ollama()
    random.seed(0); np.random.seed(0)

    # sorted: glob does not promise an order, and "first row per id wins" makes
    # the kept row depend on it. An unordered builder is not a reproducible one.
    files = sorted({f for pat in a.findings for f in glob.glob(pat)})
    # findings.jsonl is a mixed log: alongside real findings it carries text-less
    # `access` rows, and one id can appear many times. Keeping them produced a
    # dataset that was 97% duplicate rows, 69% of it a single claim=""/evidence=""
    # pair labelled positive, because two empty strings embed identically and
    # score cos=1.0. Take one row per id — the first with usable text — and drop
    # the rest before anything is embedded or paired.
    rows = seen = 0
    by_id = {}
    for f in files:
        inv = f.split("/")[-2]
        for line in open(f, errors="ignore"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            rows += 1
            fid, text = r.get("id"), (r.get("text") or "").strip()[:2000]
            if not fid or not text:
                continue
            seen += 1
            if fid in by_id:
                continue
            by_id[fid] = {"id": fid, "text": text, "topic": topic_of(r),
                          "derived_from": r.get("derived_from") or [], "inv": inv}
    findings = list(by_id.values())
    if not findings:
        raise SystemExit(f"no usable findings in {len(files)} file(s): "
                         f"{rows} rows, none with both an id and text")
    print(f"findings: {len(findings)} (from {rows} rows, {rows - seen} text-less or id-less, "
          f"{seen - len(findings)} duplicate ids) topics: {dict(Counter(x['topic'] for x in findings))}")

    E = embed([x["text"] for x in findings], a.ollama)
    emb = {findings[i]["id"]: E[i] for i in range(len(findings))}

    ids = [x["id"] for x in findings]
    pos, neg = [], []
    for x, y in itertools.combinations(ids, 2):
        tx, ty = by_id[x]["topic"], by_id[y]["topic"]
        if by_id[x]["inv"] != by_id[y]["inv"] and (tx.startswith("synthesis") or ty.startswith("synthesis")):
            continue
        cos = float(emb[x] @ emb[y])
        (pos if tx == ty else neg).append((x, y, cos))
    random.shuffle(neg); neg = neg[:len(pos) * a.neg_ratio]
    print(f"topical pairs: {len(pos)}(+) / {len(neg)}(-)  "
          f"cos same={np.mean([c for *_, c in pos]):.3f} cross={np.mean([c for *_, c in neg]):.3f}")

    def feat(x, y):
        va, vb = emb[x], emb[y]
        return np.concatenate([np.abs(va - vb), va * vb, [float(va @ vb)]])
    X = np.array([feat(x, y) for x, y, _ in pos + neg])
    Y = np.array([1] * len(pos) + [0] * len(neg))
    cosv = np.array([c for *_, c in pos + neg])

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score, accuracy_score
    import joblib
    clf = LogisticRegression(max_iter=2000, C=1.0)
    proba = cross_val_predict(clf, X, Y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              method="predict_proba")[:, 1]
    auc, acc = roc_auc_score(Y, proba), accuracy_score(Y, proba > 0.5)
    base = max(accuracy_score(Y, cosv > t) for t in np.linspace(0.2, 0.8, 61))
    print(f"LR 5-fold CV: AUC={auc:.3f} acc={acc:.3f} | cosine baseline acc={base:.3f}")
    clf.fit(X, Y); joblib.dump(clf, f"{a.out}/grounding_bleed_clf.joblib")

    ds = []
    for x, y, c in pos: ds.append({"claim": by_id[x]["text"], "evidence": by_id[y]["text"], "label": 1, "signal": "topical", "cos": round(c, 3)})
    for x, y, c in neg: ds.append({"claim": by_id[x]["text"], "evidence": by_id[y]["text"], "label": 0, "signal": "topical", "cos": round(c, 3)})
    lin = 0
    for x in findings:
        for p in x["derived_from"]:
            if p in by_id:
                ds.append({"claim": x["text"], "evidence": by_id[p]["text"], "label": 1, "signal": "lineage"}); lin += 1
    with open(f"{a.out}/grounding_dataset.jsonl", "w") as fo:
        for r in ds: fo.write(json.dumps(r) + "\n")
    json.dump({"trained_task": "rag_bleed / topical-grounding", "cv_auc": round(auc, 3), "cv_acc": round(acc, 3),
               "cosine_baseline_acc": round(base, 3), "n_topical": len(pos) + len(neg), "n_lineage": lin,
               "n_hallucination": 0, "total": len(ds)}, open(f"{a.out}/metrics.json", "w"), indent=1)
    print(f"wrote grounding_dataset.jsonl ({len(ds)}), grounding_bleed_clf.joblib, metrics.json -> {a.out}")


if __name__ == "__main__":
    main()
