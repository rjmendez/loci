#!/usr/bin/env python3
"""mlops/routing/critic.py — Tiny-Critic retrieval quality router.

Implements the Tiny-Critic RAG gate (arXiv:2603.00846, Mar 2026): a lightweight
routing layer that judges whether a retrieval result is good enough before it
reaches the main model. Starts as a heuristic, upgrades to a trained
LogisticRegression classifier once 50+ labelled examples accumulate.

Usage:
    from mlops.routing.critic import RetrievalCritic
    critic = RetrievalCritic()
    verdict = critic.route(query, chunks, scores)
    if not verdict["pass"]:
        # fallback: re-rank, broaden query, etc.
"""

import json
import os
from pathlib import Path

_LABELS_PATH = Path(__file__).parent / "critic_labels.jsonl"
_MODEL_PATH = Path(__file__).parent / "critic_model.pkl"

MIN_SAMPLES_TRAIN = 50
TOP_SCORE_THRESHOLD = 0.50
MEAN_SCORE_THRESHOLD = 0.45


class RetrievalCritic:
    def __init__(self):
        self._clf = None
        self._try_load_classifier()

    def _try_load_classifier(self):
        if not _MODEL_PATH.exists():
            return
        try:
            import pickle
            with open(_MODEL_PATH, "rb") as fh:
                self._clf = pickle.load(fh)
        except Exception:
            self._clf = None

    def _feature_vector(self, query: str, chunks: list, scores: list) -> list:
        top = max(scores) if scores else 0.0
        mean = sum(scores) / len(scores) if scores else 0.0
        min_s = min(scores) if scores else 0.0
        spread = top - min_s
        n = len(chunks)
        return [top, mean, min_s, spread, float(n), float(len(query.split()))]

    def route(self, query: str, chunks: list, scores: list) -> dict:
        feats = self._feature_vector(query, chunks, scores)
        top_score = feats[0]
        mean_score = feats[1]

        if self._clf is not None:
            try:
                proba = self._clf.predict_proba([feats])[0][1]
                passed = bool(proba >= 0.5)
                return {
                    "pass": passed,
                    "reason": f"classifier proba={proba:.3f}",
                    "confidence": float(proba),
                    "backend": "classifier",
                    "scores": {"top": top_score, "mean": mean_score},
                }
            except Exception:
                pass

        passed = top_score >= TOP_SCORE_THRESHOLD and mean_score >= MEAN_SCORE_THRESHOLD
        reasons = []
        if top_score < TOP_SCORE_THRESHOLD:
            reasons.append(f"top_score={top_score:.3f}<{TOP_SCORE_THRESHOLD}")
        if mean_score < MEAN_SCORE_THRESHOLD:
            reasons.append(f"mean_score={mean_score:.3f}<{MEAN_SCORE_THRESHOLD}")
        return {
            "pass": passed,
            "reason": "; ".join(reasons) if reasons else "heuristic ok",
            "confidence": top_score,
            "backend": "heuristic",
            "scores": {"top": top_score, "mean": mean_score},
        }

    def record_label(self, query: str, chunks: list, scores: list, label: int) -> None:
        record = {
            "query": query,
            "n_chunks": len(chunks),
            "scores": scores,
            "label": label,
        }
        with open(_LABELS_PATH, "a") as fh:
            fh.write(json.dumps(record) + "\n")

    def train(self, min_samples: int = MIN_SAMPLES_TRAIN) -> dict:
        if not _LABELS_PATH.exists():
            return {"trained": False, "reason": "no labels file"}
        rows = []
        with open(_LABELS_PATH) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        if len(rows) < min_samples:
            return {"trained": False, "reason": f"need {min_samples} samples, have {len(rows)}"}
        try:
            from sklearn.linear_model import LogisticRegression
            import pickle

            X, y = [], []
            for r in rows:
                scores = r.get("scores", [])
                query = r.get("query", "")
                n = r.get("n_chunks", len(scores))
                top = max(scores) if scores else 0.0
                mean = sum(scores) / len(scores) if scores else 0.0
                mn = min(scores) if scores else 0.0
                X.append([top, mean, mn, top - mn, float(n), float(len(query.split()))])
                y.append(int(r.get("label", 0)))

            clf = LogisticRegression(max_iter=500)
            clf.fit(X, y)
            with open(_MODEL_PATH, "wb") as fh:
                pickle.dump(clf, fh)
            self._clf = clf
            return {"trained": True, "n_samples": len(rows)}
        except ImportError:
            return {"trained": False, "reason": "scikit-learn not installed"}
        except Exception as exc:
            return {"trained": False, "reason": str(exc)}
