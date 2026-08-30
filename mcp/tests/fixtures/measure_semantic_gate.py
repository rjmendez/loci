#!/usr/bin/env python3
"""Regenerate semantic_gate_probe.json (this directory) from the live corpus.

Needs a reachable Qdrant + embedder (QDRANT_URL / QDRANT_API_KEY / OLLAMA_BASE_URL,
LOCI_QDRANT_RETENTION_DAYS=0) and an explicit
LOCI_SEMANTIC_GATE_SOURCE_DIR pointing at the source session store. Run with
mcp/.venv/bin/python.

Two probe classes, 300 each, seed 1183:

  positive  a finding's own text, checked against its OWN investigation with that
            finding's own indexed point excluded from the hits and its own record
            excluded from the lexical pool. This is the "the claim really is
            backed by this investigation" case.
  negative  a finding's text, copied verbatim, checked against a DIFFERENT
            randomly chosen investigation. Nothing in that investigation supports
            it, so every reported support is false support.

Only scores, per-ref lexical overlap and pool size are written out -- no
embeddings and no finding text, so the fixture stays a few hundred KB and carries
nothing private.
"""
import collections
import glob
import json
import logging
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "mcp"))
os.environ.setdefault("LOCI_QDRANT_RETENTION_DAYS", "0")
logging.disable(logging.WARNING)

if not os.environ.get("LOCI_SEMANTIC_GATE_SOURCE_DIR"):
    raise SystemExit(
        "Refusing to default to a live memory store; set "
        "LOCI_SEMANTIC_GATE_SOURCE_DIR explicitly."
    )
MEM = os.path.expanduser(os.environ["LOCI_SEMANTIC_GATE_SOURCE_DIR"])
os.environ["LOCI_MEMORY_DIR"] = MEM
os.environ["HERMES_MEMORY_DIR"] = MEM

import qdrant_ops  # noqa: E402
import server as S  # noqa: E402
from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: E402

SEED = 1183
N = 300
OUT = Path(__file__).resolve().parent / "semantic_gate_probe.json"


def load_findings():
    texts, by_inv = {}, collections.defaultdict(list)
    for path in glob.glob(MEM + "/*/findings.jsonl"):
        for line in open(path):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("text") and rec.get("id"):
                texts[rec["id"]] = rec
                by_inv[rec["investigation_id"]].append(rec)
    return texts, by_inv


def indexed_ids(client, col):
    ids, offset = set(), None
    while True:
        points, offset = client.scroll(
            collection_name=col, limit=1000, offset=offset, with_payload=["id"]
        )
        for point in points:
            pid = (point.payload or {}).get("id")
            if pid:
                ids.add(str(pid))
        if offset is None:
            return ids


def main():
    random.seed(SEED)
    client, col = qdrant_ops._get_qdrant()
    if client is None:
        sys.exit("qdrant unavailable")
    texts, by_inv = load_findings()
    indexed = indexed_ids(client, col)

    def search(claim, inv, limit=S._SEMANTIC_NEIGHBOURHOOD_K):
        vector = qdrant_ops._embed(claim[:2000])
        if vector is None:
            return []
        res = client.query_points(
            collection_name=col, query=vector, using="dense",
            query_filter=Filter(must=[
                FieldCondition(key="investigation_id", match=MatchValue(value=inv))
            ]),
            limit=limit, with_payload=True,
        )
        return [{"id": str((p.payload or {}).get("id")), "score": float(p.score),
                 "text": str((p.payload or {}).get("text") or "")} for p in res.points]

    pools = {}

    def pool(inv):
        if inv not in pools:
            pools[inv] = S.build_validation_evidence(inv, min_confidence="medium")
        return pools[inv]

    def probe(claim, inv, label, exclude_id=None):
        hits = [h for h in search(claim, inv) if h["id"] != exclude_id]
        if not hits:
            return None
        claim_tokens = S.tokenize(claim)
        negated = bool(S._NEGATION_RE.search(claim))
        evidence = [e for e in pool(inv) if str(e.get("evidence_id")) != exclude_id]
        support, _contra = S._pre_answer_lexical_refs(claim_tokens, negated, evidence)
        surfaced = [h for h in hits[:S._SEMANTIC_REF_LIMIT]
                    if h["score"] >= S._QDRANT_SUPPORT_MIN_SCORE]
        return {
            "label": label,
            "investigation_id": inv,
            "scores": [round(h["score"], 6) for h in hits],
            "refs": [{
                "score": round(h["score"], 6),
                "lexical_overlap": round(
                    S._lexical_match_score(claim_tokens, S.tokenize(h["text"])), 3),
            } for h in surfaced],
            "lexical_support_refs": len(support),
        }

    # Positives: parent/child pairs inside one investigation, child text as claim.
    children = [(c, texts[par]) for c in texts.values()
                for par in (c.get("derived_from") or [])
                if par in texts and texts[par]["investigation_id"] == c["investigation_id"]]
    pos_pairs = [(c, p) for c, p in children if p["id"] in indexed]
    random.shuffle(pos_pairs)
    rows = []
    for child, _parent in pos_pairs[:N]:
        row = probe(child["text"], child["investigation_id"], "positive", exclude_id=child["id"])
        if row:
            rows.append(row)

    # Negatives: a real finding's text asked of a different investigation.
    invs = [i for i in by_inv if any(f["id"] in indexed for f in by_inv[i])]
    sources = [r for r in texts.values() if len(r["text"]) > 80]
    random.shuffle(sources)
    negatives = 0
    for rec in sources:
        if negatives >= N:
            break
        other = random.choice(invs)
        if other == rec["investigation_id"]:
            continue
        row = probe(rec["text"], other, "negative")
        if row:
            rows.append(row)
            negatives += 1

    OUT.write_text(json.dumps({
        "generated_by": "mcp/tests/fixtures/measure_semantic_gate.py",
        "corpus": f"{MEM}, dense vectors from the configured Qdrant collection",
        "embedder": "nomic-embed-text (cosine)",
        "seed": SEED,
        "notes": (
            "positive = a finding's text against its own investigation, its own indexed point "
            "and its own lexical record excluded; negative = a finding's text against a "
            "different investigation. 'scores' is the full retrieved neighbourhood (k=25, "
            "fewer when the investigation has fewer indexed points); 'refs' are the surfaced "
            "top-5 at or above the 0.55 pre-filter. No embeddings, no finding text."
        ),
        "rows": rows,
    }, indent=1))
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
