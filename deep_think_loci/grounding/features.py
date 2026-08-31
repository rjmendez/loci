"""The grounding feature contract, in one place.

Three sites built these vectors independently and two of them drifted apart:
build_grounding_dataset.py and ground_gate.py emit 1537 columns, while
mlops/grounding/train.py emits 1540 — it added cos**2, a length ratio and a token
overlap. Nothing noticed because the loop has never promoted a model. The first
promotion would have handed ground_gate a 1540-column classifier and fed it 1537,
raising at runtime inside the live grounding path.

So the shape is defined once here, and the version a model was trained against is
recoverable from the model itself via n_features_in_.
"""
from __future__ import annotations

import numpy as np

# Embedding width is fixed by the embedder (nomic-embed-text, 768).
EMBED_DIM = 768
# |a-b| + a*b + cos  — what build_grounding_dataset.py and ground_gate.py have
# always produced, and what the currently shipped joblib expects.
LEGACY_DIM = 2 * EMBED_DIM + 1
# ... + cos**2 + length ratio + token overlap.
CURRENT_DIM = LEGACY_DIM + 3


def token_overlap(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def len_ratio(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    return min(la, lb) / (max(la, lb) + 1)


def make_features(claims, evidences, emb_claims, emb_evidences, *, dim: int = CURRENT_DIM):
    """Feature matrix for (claim, evidence) pairs.

    `dim` selects the contract version. Pass a model's n_features_in_ to build
    exactly what that model was trained on, rather than assuming.
    """
    emb_claims = np.asarray(emb_claims, dtype=np.float32)
    emb_evidences = np.asarray(emb_evidences, dtype=np.float32)
    diff = np.abs(emb_claims - emb_evidences)
    prod = emb_claims * emb_evidences
    cos = prod.sum(axis=1, keepdims=True)
    if dim == LEGACY_DIM:
        return np.concatenate([diff, prod, cos], axis=1)
    if dim != CURRENT_DIM:
        raise ValueError(
            f"no grounding feature contract produces {dim} columns "
            f"(known: {LEGACY_DIM} legacy, {CURRENT_DIM} current)"
        )
    lr = np.array([[len_ratio(c, e)] for c, e in zip(claims, evidences)], dtype=np.float32)
    jac = np.array([[token_overlap(c, e)] for c, e in zip(claims, evidences)], dtype=np.float32)
    return np.concatenate([diff, prod, cos, cos ** 2, lr, jac], axis=1)


def supported_dims() -> tuple[int, ...]:
    return (LEGACY_DIM, CURRENT_DIM)
