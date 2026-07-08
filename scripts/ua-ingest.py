#!/usr/bin/env python3
"""
ua-ingest.py — Ingest an understand-anything final-graph.json into Qdrant.

Usage:
    python3 ua-ingest.py <project_root>

Reads:  <project_root>/.understand-anything/intermediate/final-graph.json
        <project_root>/.understand-anything/fingerprints.json  (optional, for git hash)

Writes: Qdrant collection `understand_anything` (creates if missing, 768-dim nomic-embed-text)

Each node becomes one Qdrant point:
  - vector: nomic-embed-text embedding of "<type> <name>: <summary> [tags]"
  - payload: all node fields + project, repo_name, git_hash, ingest_at

Idempotent: points are upserted by deterministic UUID derived from (repo_name + node.id).
Edges are stored as payload on each node (outgoing_edges, incoming_edges lists).

Collection schema:
  Collection: understand_anything
  Vector:     768-dim, cosine
  Payload indexes:
    project   (keyword)
    repo_name (keyword)
    type      (keyword)
    layer     (keyword)
    domain    (keyword)
    git_hash  (keyword)
"""

import sys
import os
import json
import uuid
import hashlib
import subprocess
from datetime import datetime, timezone

# ── Config from env or defaults ────────────────────────────────────────────
QDRANT_URL  = os.environ.get("QDRANT_URL")
QDRANT_KEY  = os.environ.get("QDRANT_KEY",  "8324728be0accd776f7450ae9e5e4f8ebd155c48a4cfbc3d3d94e7490aaa60ab")
OLLAMA_URL  = os.environ.get("OLLAMA_URL")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
COLLECTION  = os.environ.get("UA_COLLECTION", "understand_anything")
BATCH_SIZE  = int(os.environ.get("UA_BATCH_SIZE", "32"))  # embed + upsert batch

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "requests"])
    import requests


# ── Helpers ─────────────────────────────────────────────────────────────────

def qdrant(method, path, **kwargs):
    url = f"{QDRANT_URL}{path}"
    headers = {"api-key": QDRANT_KEY, "Content-Type": "application/json"}
    r = getattr(requests, method)(url, headers=headers, **kwargs)
    r.raise_for_status()
    return r.json()


_fastembed_model = None

def embed_batch(texts):
    """Embed texts via Ollama /api/embed, or fastembed if OLLAMA_URL is unset."""
    global _fastembed_model
    if OLLAMA_URL:
        r = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": texts},
            timeout=120
        )
        r.raise_for_status()
        return r.json()["embeddings"]
    # Fallback: fastembed TextEmbedding (CPU, same nomic-embed-text model)
    if _fastembed_model is None:
        from fastembed import TextEmbedding
        _fastembed_model = TextEmbedding("nomic-ai/nomic-embed-text-v1.5")
    return [v.tolist() for v in _fastembed_model.embed(texts)]


def node_text(node):
    """Build the text string to embed for a node."""
    parts = [node.get("type", ""), node.get("name", "")]
    summary = node.get("summary", "")
    if summary:
        parts.append(": " + summary)
    tags = node.get("tags", [])
    if tags:
        parts.append("[" + ", ".join(tags) + "]")
    layer = node.get("layer", {})
    if isinstance(layer, dict) and layer.get("name"):
        parts.append("layer:" + layer["name"])
    domain = node.get("domain", {})
    if isinstance(domain, dict) and domain.get("name"):
        parts.append("domain:" + domain["name"])
    return " ".join(parts)


def stable_point_id(repo_name, node_id):
    """Derive a stable UUID from (repo_name, node_id) for idempotent upserts."""
    raw = f"{repo_name}::{node_id}"
    h = hashlib.sha256(raw.encode()).hexdigest()
    return str(uuid.UUID(h[:32]))


def ensure_collection():
    """Create the understand_anything collection if it doesn't exist."""
    try:
        qdrant("get", f"/collections/{COLLECTION}")
        print(f"  Collection {COLLECTION!r} already exists.")
        return
    except requests.HTTPError:
        pass

    print(f"  Creating collection {COLLECTION!r} (768-dim cosine)...")
    qdrant("put", f"/collections/{COLLECTION}", json={
        "vectors": {"size": 768, "distance": "Cosine"}
    })

    # Create payload indexes
    for field, schema in [
        ("project",   {"type": "keyword"}),
        ("repo_name", {"type": "keyword"}),
        ("type",      {"type": "keyword"}),
        ("layer_name",{"type": "keyword"}),
        ("domain_name",{"type":"keyword"}),
        ("git_hash",  {"type": "keyword"}),
        ("file_path", {"type": "keyword"}),
    ]:
        qdrant("put", f"/collections/{COLLECTION}/index", json={"field_name": field, "field_schema": schema})

    print("  Collection and indexes created.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: ua-ingest.py <project_root>")
        sys.exit(1)

    project_root = os.path.abspath(sys.argv[1])
    ua_dir       = os.path.join(project_root, ".understand-anything")
    graph_path   = os.path.join(ua_dir, "intermediate", "final-graph.json")
    fp_path      = os.path.join(ua_dir, "fingerprints.json")

    if not os.path.exists(graph_path):
        # Fallback to assembled-graph
        graph_path = os.path.join(ua_dir, "intermediate", "assembled-graph.json")
        if not os.path.exists(graph_path):
            print(f"ERROR: No final-graph.json or assembled-graph.json found under {ua_dir}")
            sys.exit(1)
        print(f"  Warning: using assembled-graph.json (no final-graph.json)")

    print(f"\nua-ingest: {project_root}")
    print(f"  Graph:  {graph_path}")

    # ── Load graph ──────────────────────────────────────────────────────────
    with open(graph_path) as f:
        graph = json.load(f)

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    print(f"  Nodes: {len(nodes)}  Edges: {len(edges)}")

    # ── Metadata ────────────────────────────────────────────────────────────
    repo_name = os.path.basename(project_root)

    git_hash = ""
    try:
        git_hash = subprocess.run(
            ["git", "-C", project_root, "rev-parse", "HEAD"],
            capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        pass

    if not git_hash and os.path.exists(fp_path):
        try:
            with open(fp_path) as f:
                fp = json.load(f)
            git_hash = fp.get("gitCommitHash", "")
        except Exception:
            pass

    ingest_at = datetime.now(timezone.utc).isoformat()
    print(f"  Repo:   {repo_name}  git: {git_hash[:12] or 'unknown'}  at: {ingest_at}")

    # ── Build edge index per node ───────────────────────────────────────────
    out_edges = {}   # node_id -> list of edge dicts
    in_edges  = {}

    for e in edges:
        src, tgt = e.get("source", ""), e.get("target", "")
        out_edges.setdefault(src, []).append({"target": tgt, "type": e.get("type"), "weight": e.get("weight")})
        in_edges.setdefault(tgt, []).append({"source": src, "type": e.get("type"), "weight": e.get("weight")})

    # ── Ensure collection ───────────────────────────────────────────────────
    ensure_collection()

    # ── Embed + Upsert in batches ───────────────────────────────────────────
    total_upserted = 0
    batch_texts, batch_nodes = [], []

    def flush_batch():
        nonlocal total_upserted
        if not batch_texts:
            return
        vectors = embed_batch(batch_texts)
        points = []
        for node, vec in zip(batch_nodes, vectors):
            nid = node["id"]
            layer  = node.get("layer",  {}) or {}
            domain = node.get("domain", {}) or {}
            payload = {
                "node_id":      nid,
                "type":         node.get("type", ""),
                "name":         node.get("name", ""),
                "summary":      node.get("summary", ""),
                "tags":         node.get("tags", []),
                "complexity":   node.get("complexity", ""),
                "file_path":    node.get("filePath", ""),
                "layer_id":     layer.get("id", ""),
                "layer_name":   layer.get("name", ""),
                "layer_color":  layer.get("color", ""),
                "domain_id":    domain.get("id", ""),
                "domain_name":  domain.get("name", ""),
                "domain_color": domain.get("color", ""),
                "repo_name":    repo_name,
                "project":      project_root,
                "git_hash":     git_hash,
                "ingest_at":    ingest_at,
                "outgoing_edges": out_edges.get(nid, [])[:20],
                "incoming_edges": in_edges.get(nid, [])[:20],
            }
            points.append({
                "id":      stable_point_id(repo_name, nid),
                "vector":  vec,
                "payload": payload
            })
        qdrant("put", f"/collections/{COLLECTION}/points", json={"points": points})
        total_upserted += len(points)
        print(f"  Upserted {total_upserted}/{len(nodes)}...", end="\r", flush=True)
        batch_texts.clear()
        batch_nodes.clear()

    for node in nodes:
        batch_texts.append(node_text(node))
        batch_nodes.append(node)
        if len(batch_texts) >= BATCH_SIZE:
            flush_batch()

    flush_batch()

    print(f"\n  Done. {total_upserted} points upserted into {COLLECTION!r}")

    # ── Ensure knowledge-graph.json symlink (for UA sub-skills + dashboard) ──
    kg_symlink = os.path.join(ua_dir, "knowledge-graph.json")
    rel_target = os.path.relpath(graph_path, ua_dir)
    if os.path.islink(kg_symlink):
        if os.readlink(kg_symlink) != rel_target:
            os.unlink(kg_symlink)
            os.symlink(rel_target, kg_symlink)
            print(f"  Symlink updated: knowledge-graph.json -> {rel_target}")
    elif not os.path.exists(kg_symlink):
        os.symlink(rel_target, kg_symlink)
        print(f"  Symlink created: knowledge-graph.json -> {rel_target}")

    # ── Verify ───────────────────────────────────────────────────────────────
    info = qdrant("get", f"/collections/{COLLECTION}")
    count = info.get("result", {}).get("points_count", "?")
    print(f"  Collection total points: {count}")

    # Print type breakdown
    print("\n  Breakdown by type:")
    type_counts = {}
    for n in nodes:
        t = n.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:15s} {c}")

    print(f"\nIngestion complete. Search with:")
    print(f"  QDRANT_URL={QDRANT_URL}")
    print(f"  Collection: {COLLECTION}")
    print(f"  Repo filter: {{\"must\": [{{\"key\": \"repo_name\", \"match\": {{\"value\": \"{repo_name}\"}}}}]}}")


if __name__ == "__main__":
    main()
