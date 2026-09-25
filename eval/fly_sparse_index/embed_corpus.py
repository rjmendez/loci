"""Embed the offline evaluation corpora for the fly sparse-index design study.

Reads ONLY: this git worktree (tracked *.md / *.py files) and a local BEIR
NFCorpus copy. Writes ONLY under the cache directory (default
/mnt/f/loci-eval/fly-sparse-index). Never touches ~/.loci, ~/.hermes, Qdrant
or Mnemosyne; talks only to the local Ollama embedding endpoint with
nomic-embed-text (no pulls, no model config changes).

Loci-derived text and vectors go to ``<cache>/private`` (chmod 600 where the
filesystem honours it; on the F: drvfs mount the directory is additionally
restricted with a Windows ACL to the operator account).

Loci calls /api/embed without nomic task prefixes (mcp/embed_ops.py), so this
script does the same.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

MODEL = "nomic-embed-text"
CHUNK_CHARS = 1200
MIN_CHARS = 200
MAX_DOC_CHARS = 2000


def topic_of(path: str) -> str:
    parts = path.split("/")
    if parts[0] == "mcp":
        if len(parts) > 2 and parts[1] == "tests":
            return "mcp-tests"
        if parts[-1].startswith("flybrain_"):
            return "mcp-flybrain"
        return "mcp-core"
    if parts[0] == "scripts":
        return "scripts"
    if parts[0] in ("docs", "mlops", "a2a_server"):
        return parts[0]
    return "other"


def chunk_text(text: str, size: int = CHUNK_CHARS, min_chars: int = MIN_CHARS) -> list[str]:
    """Greedy line-boundary chunks of at most ``size`` characters."""
    chunks, cur = [], []
    cur_len = 0
    for line in text.splitlines(keepends=True):
        while len(line) > size:  # very long line: hard split
            if cur:
                chunks.append("".join(cur))
                cur, cur_len = [], 0
            chunks.append(line[:size])
            line = line[size:]
        if cur_len + len(line) > size and cur:
            chunks.append("".join(cur))
            cur, cur_len = [], 0
        cur.append(line)
        cur_len += len(line)
    if cur:
        chunks.append("".join(cur))
    return [c for c in chunks if len(c.strip()) >= min_chars]


def loci_chunks(repo: Path) -> list[dict]:
    files = subprocess.run(["git", "-C", str(repo), "ls-files", "*.md", "*.py"], check=True,
                           capture_output=True, text=True).stdout.split()
    rows = []
    for rel in sorted(files):
        try:
            text = (repo / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, chunk in enumerate(chunk_text(text)):
            rows.append({"id": f"{rel}#{i}", "path": rel, "topic": topic_of(rel), "text": chunk,
                         "sha256": hashlib.sha256(chunk.encode("utf-8")).hexdigest()})
    return rows


def embed(texts: list[str], base: str, batch: int) -> np.ndarray:
    out = []
    for s in range(0, len(texts), batch):
        body = json.dumps({"model": MODEL, "input": texts[s:s + batch], "truncate": True}).encode()
        req = urllib.request.Request(f"{base}/api/embed", data=body, headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    embs = json.loads(resp.read())["embeddings"]
                break
            except Exception:  # noqa: BLE001 - retry then fail closed
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        if len(embs) != len(texts[s:s + batch]):
            raise RuntimeError("embedding count mismatch")
        out.extend(embs)
        if (s // batch) % 20 == 0:
            print(f"  embedded {min(s + batch, len(texts))}/{len(texts)}", flush=True)
    arr = np.asarray(out, dtype=np.float32)
    if arr.ndim != 2 or not np.isfinite(arr).all():
        raise RuntimeError("bad embeddings")
    return arr


def _private_write(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    os.chmod(path, 0o600)


def _save_npy(path: Path, arr: np.ndarray, private: bool) -> None:
    import io
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    if private:
        _private_write(path, buf.getvalue())
    else:
        path.write_bytes(buf.getvalue())


def model_digest(base: str) -> str:
    with urllib.request.urlopen(f"{base}/api/tags", timeout=10) as resp:
        for m in json.loads(resp.read())["models"]:
            if m["name"].split(":")[0] == MODEL:
                return m.get("digest", "")
    raise RuntimeError(f"{MODEL} not present in Ollama; refusing to pull")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--cache", default="/mnt/f/loci-eval/fly-sparse-index")
    ap.add_argument("--nfcorpus", default="/mnt/f/loci-eval/fly-sparse-index/beir/nfcorpus")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args(argv)
    cache = Path(args.cache)
    private = cache / "private"
    private.mkdir(parents=True, exist_ok=True)
    digest = model_digest(args.ollama)
    repo_head = subprocess.run(["git", "-C", args.repo, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()

    rows = loci_chunks(Path(args.repo))
    print(f"loci: {len(rows)} chunks", flush=True)
    emb = embed([r["text"] for r in rows], args.ollama, args.batch)
    _private_write(private / "loci_chunks.jsonl", "".join(json.dumps(r) + "\n" for r in rows).encode("utf-8"))
    _save_npy(private / "loci_emb.npy", emb, private=True)

    nf = Path(args.nfcorpus)
    docs = [json.loads(line) for line in (nf / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    queries = {q["_id"]: q["text"] for q in map(json.loads, (nf / "queries.jsonl").read_text(encoding="utf-8").splitlines())}
    test_qids = sorted({line.split("\t")[0] for line in (nf / "qrels" / "test.tsv").read_text().splitlines()[1:]})
    doc_texts = [((d.get("title") or "") + "\n" + (d.get("text") or ""))[:MAX_DOC_CHARS] for d in docs]
    print(f"nfcorpus: {len(docs)} docs, {len(test_qids)} test queries", flush=True)
    nf_emb = embed(doc_texts, args.ollama, args.batch)
    q_emb = embed([queries[q] for q in test_qids], args.ollama, args.batch)
    cache.joinpath("nfcorpus").mkdir(exist_ok=True)
    _save_npy(cache / "nfcorpus" / "doc_emb.npy", nf_emb, private=False)
    _save_npy(cache / "nfcorpus" / "query_emb.npy", q_emb, private=False)
    (cache / "nfcorpus" / "ids.json").write_text(json.dumps({"doc_ids": [d["_id"] for d in docs], "query_ids": test_qids}))

    meta = {
        "model": MODEL, "ollama_model_digest": digest, "endpoint": "/api/embed", "prefixes": "none (matches mcp/embed_ops.py)",
        "loci_repo_head": repo_head, "loci_chunks": len(rows), "chunk_chars": CHUNK_CHARS, "min_chars": MIN_CHARS,
        "loci_topics": {t: sum(r["topic"] == t for r in rows) for t in sorted({r["topic"] for r in rows})},
        "nfcorpus_docs": len(docs), "nfcorpus_test_queries": len(test_qids), "nfcorpus_max_doc_chars": MAX_DOC_CHARS,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sha256": {
            "private/loci_emb.npy": hashlib.sha256((private / "loci_emb.npy").read_bytes()).hexdigest(),
            "nfcorpus/doc_emb.npy": hashlib.sha256((cache / "nfcorpus" / "doc_emb.npy").read_bytes()).hexdigest(),
            "nfcorpus/query_emb.npy": hashlib.sha256((cache / "nfcorpus" / "query_emb.npy").read_bytes()).hexdigest(),
        },
    }
    (cache / "embeddings_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
