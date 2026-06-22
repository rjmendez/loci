"""
spreading_activation.py — SA-RAG-style BFS spreading activation with SYNAPSE fan-effect dampening.

References:
  SA-RAG: arxiv 2512.15922
  SYNAPSE fan-effect: arxiv 2601.02744

Usage (CLI):
  python spreading_activation.py --db ~/.hermes/mnemosyne/data/mnemosyne.db \
      --seeds id1,id2 --max-results 5

Environment variables (all optional):
  MNEMOSYNE_DB              Path to mnemosyne.db (default: ~/.hermes/mnemosyne/data/mnemosyne.db)
  SA_EDGE_FLOOR             Min edge weight to traverse (default: 0.4)
  SA_ACTIVATION_THRESHOLD   Min activation to include in results (default: 0.5)
  SA_MAX_HOPS               BFS depth limit (default: 2)
  SA_FAN_EFFECT             Divide activation by out-degree (default: true)
  SA_HYBRID_VECTOR_WEIGHT   Weight for initial Qdrant score in hybrid (default: 0.7)
  SA_HYBRID_ACTIVATION_WEIGHT Weight for SA score in hybrid (default: 0.3)
"""

import argparse
import json
import os
import sqlite3
import sys

# ---------------------------------------------------------------------------
# Config — all overridable via environment
# ---------------------------------------------------------------------------

DB_PATH = os.environ.get(
    "MNEMOSYNE_DB",
    os.path.expanduser("~/.hermes/mnemosyne/data/mnemosyne.db"),
)
SA_EDGE_FLOOR = float(os.environ.get("SA_EDGE_FLOOR", "0.4"))
SA_ACTIVATION_THRESHOLD = float(os.environ.get("SA_ACTIVATION_THRESHOLD", "0.5"))
SA_MAX_HOPS = int(os.environ.get("SA_MAX_HOPS", "2"))
SA_FAN_EFFECT = os.environ.get("SA_FAN_EFFECT", "true").lower() not in ("false", "0", "no")
SA_HYBRID_VECTOR_WEIGHT = float(os.environ.get("SA_HYBRID_VECTOR_WEIGHT", "0.7"))
SA_HYBRID_ACTIVATION_WEIGHT = float(os.environ.get("SA_HYBRID_ACTIVATION_WEIGHT", "0.3"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Print a status line to stderr with [sa] prefix."""
    print(f"[sa] {msg}", file=sys.stderr)


def _placeholders(n: int) -> str:
    """Return a comma-separated string of n SQL placeholders."""
    return ",".join("?" * n)


def _fetch_edges(conn: sqlite3.Connection, node_ids: list[str], edge_floor: float) -> list[tuple]:
    """
    Load graph_edges where source is in node_ids and weight >= edge_floor.
    Returns list of (source, target, weight).
    """
    if not node_ids:
        return []
    sql = (
        "SELECT source, target, weight FROM graph_edges "
        f"WHERE source IN ({_placeholders(len(node_ids))}) "
        "AND weight >= ?"
    )
    params = node_ids + [edge_floor]
    try:
        cursor = conn.execute(sql, params)
        return cursor.fetchall()
    except sqlite3.OperationalError as exc:
        _log(f"graph_edges query failed ({exc}); treating as empty")
        return []


def _fetch_content(conn: sqlite3.Connection, node_ids: list[str]) -> dict[str, dict]:
    """
    Retrieve content and importance for node_ids from working_memory,
    then fall back to episodic_memory for any IDs not found.
    Returns dict: id -> {"content": str, "importance": float}.
    """
    if not node_ids:
        return {}

    results: dict[str, dict] = {}

    # Try working_memory first
    try:
        sql = (
            f"SELECT id, content, importance FROM working_memory "
            f"WHERE id IN ({_placeholders(len(node_ids))})"
        )
        for row in conn.execute(sql, node_ids):
            results[row[0]] = {"content": row[1], "importance": float(row[2] or 0.0)}
    except sqlite3.OperationalError as exc:
        _log(f"working_memory query failed ({exc})")

    # Fetch any remaining IDs from episodic_memory
    missing = [nid for nid in node_ids if nid not in results]
    if missing:
        try:
            sql = (
                f"SELECT id, content, importance FROM episodic_memory "
                f"WHERE id IN ({_placeholders(len(missing))})"
            )
            for row in conn.execute(sql, missing):
                results[row[0]] = {"content": row[1], "importance": float(row[2] or 0.0)}
        except sqlite3.OperationalError as exc:
            _log(f"episodic_memory query failed ({exc})")

    return results


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def run_spreading_activation(
    db_path: str,
    seed_ids: list[str],
    seed_scores: dict[str, float],
    max_results: int = 5,
) -> list[dict]:
    """
    Perform BFS spreading activation over graph_edges starting from seed_ids.

    Parameters
    ----------
    db_path     : Path to mnemosyne SQLite database.
    seed_ids    : Initial memory IDs (strings) to activate from.
    seed_scores : Mapping of seed memory ID -> Qdrant cosine score (0–1).
    max_results : Maximum number of results to return.

    Returns
    -------
    List of dicts sorted by hybrid score descending:
        {"memory_id": str, "activation": float, "content": str, "importance": float}
    Only nodes NOT in the original seed set with activation >= SA_ACTIVATION_THRESHOLD
    are included.
    """
    if not seed_ids:
        _log("no seed IDs provided; returning empty")
        return []

    _log(f"starting SA: {len(seed_ids)} seeds, max_hops={SA_MAX_HOPS}, "
         f"edge_floor={SA_EDGE_FLOOR}, fan_effect={SA_FAN_EFFECT}")

    # Normalize seeds — ensure all seeds appear in seed_scores (default 1.0)
    activation: dict[str, float] = {}
    for sid in seed_ids:
        activation[sid] = seed_scores.get(sid, 1.0)

    seed_set = set(seed_ids)
    # Nodes activated at the current frontier (eligible to propagate)
    frontier: list[str] = list(seed_ids)

    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as exc:
        _log(f"cannot open database {db_path!r}: {exc}")
        return []

    try:
        for hop in range(SA_MAX_HOPS):
            if not frontier:
                _log(f"hop {hop + 1}: empty frontier; stopping early")
                break

            _log(f"hop {hop + 1}: frontier size={len(frontier)}")
            edges = _fetch_edges(conn, frontier, SA_EDGE_FLOOR)

            if not edges:
                _log(f"hop {hop + 1}: no qualifying edges found")
                break

            # Compute out-degree per source (number of qualifying edges per node)
            out_degree: dict[str, int] = {}
            for source, _target, _weight in edges:
                out_degree[source] = out_degree.get(source, 0) + 1

            newly_activated: list[str] = []

            for source, target, weight in edges:
                # Rescale weight relative to the floor
                w_prime = (weight - SA_EDGE_FLOOR) / (1.0 - SA_EDGE_FLOOR)

                src_activation = activation.get(source, 0.0)
                if SA_FAN_EFFECT:
                    degree = out_degree.get(source, 1)
                    contribution = src_activation * w_prime / max(degree, 1)
                else:
                    contribution = src_activation * w_prime

                prior = activation.get(target, 0.0)
                activation[target] = min(1.0, prior + contribution)

                if target not in activation or prior == 0.0:
                    newly_activated.append(target)

            # Next frontier: targets that were not already in the frontier set
            frontier = [
                t for (_, t, _) in edges
                if t not in seed_set
            ]
            # Deduplicate frontier
            seen: set[str] = set()
            unique_frontier: list[str] = []
            for nid in frontier:
                if nid not in seen:
                    seen.add(nid)
                    unique_frontier.append(nid)
            frontier = unique_frontier

        # Filter: exclude seeds, require activation >= threshold
        candidate_ids = [
            nid for nid, act in activation.items()
            if nid not in seed_set and act >= SA_ACTIVATION_THRESHOLD
        ]

        _log(f"candidates above threshold ({SA_ACTIVATION_THRESHOLD}): {len(candidate_ids)}")

        if not candidate_ids:
            return []

        # Fetch content for candidates
        content_map = _fetch_content(conn, candidate_ids)

        # Build result list with hybrid score
        results: list[dict] = []
        for nid in candidate_ids:
            act_score = activation[nid]
            meta = content_map.get(nid, {"content": "", "importance": 0.0})
            results.append({
                "memory_id": nid,
                "activation": act_score,
                "content": meta["content"],
                "importance": meta["importance"],
            })

        # Sort by activation descending, then trim
        results.sort(key=lambda r: r["activation"], reverse=True)
        return results[:max_results]

    finally:
        conn.close()


def run_spreading_activation_hybrid(
    db_path: str,
    seed_ids: list[str],
    seed_scores: dict[str, float],
    max_results: int = 5,
) -> list[dict]:
    """
    Wrapper that applies SA-RAG hybrid scoring:
        hybrid = SA_HYBRID_VECTOR_WEIGHT * vector_score
                 + SA_HYBRID_ACTIVATION_WEIGHT * activation_score

    For seeds not in the SA result set, vector_score comes from seed_scores.
    For activated non-seed nodes, vector_score defaults to 0.0.

    Returns same shape as run_spreading_activation but with an added
    "hybrid_score" key, sorted by hybrid_score descending.
    """
    sa_results = run_spreading_activation(db_path, seed_ids, seed_scores, max_results)

    for item in sa_results:
        vector_score = seed_scores.get(item["memory_id"], 0.0)
        item["hybrid_score"] = (
            SA_HYBRID_VECTOR_WEIGHT * vector_score
            + SA_HYBRID_ACTIVATION_WEIGHT * item["activation"]
        )

    sa_results.sort(key=lambda r: r["hybrid_score"], reverse=True)
    return sa_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SA-RAG spreading activation over Mnemosyne graph_edges",
    )
    parser.add_argument(
        "--db",
        default=DB_PATH,
        help="Path to mnemosyne SQLite DB (default: MNEMOSYNE_DB env or ~/.hermes/mnemosyne/data/mnemosyne.db)",
    )
    parser.add_argument(
        "--seeds",
        required=True,
        help="Comma-separated list of seed memory IDs",
    )
    parser.add_argument(
        "--seed-scores",
        default="",
        help="Optional comma-separated id:score pairs for seed vector scores, e.g. id1:0.9,id2:0.8",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum number of results to return (default: 5)",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Include hybrid_score in output (combines vector + activation scores)",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    seed_ids = [s.strip() for s in args.seeds.split(",") if s.strip()]
    if not seed_ids:
        _log("no valid seed IDs in --seeds; exiting")
        sys.exit(1)

    # Parse optional seed scores
    seed_scores: dict[str, float] = {}
    if args.seed_scores:
        for pair in args.seed_scores.split(","):
            pair = pair.strip()
            if ":" in pair:
                nid, score_str = pair.rsplit(":", 1)
                try:
                    seed_scores[nid.strip()] = float(score_str.strip())
                except ValueError:
                    _log(f"invalid score in pair {pair!r}; skipping")

    # Default missing seed scores to 1.0
    for sid in seed_ids:
        if sid not in seed_scores:
            seed_scores[sid] = 1.0

    _log(f"db={args.db}, seeds={seed_ids}, max_results={args.max_results}")

    if args.hybrid:
        results = run_spreading_activation_hybrid(
            db_path=args.db,
            seed_ids=seed_ids,
            seed_scores=seed_scores,
            max_results=args.max_results,
        )
    else:
        results = run_spreading_activation(
            db_path=args.db,
            seed_ids=seed_ids,
            seed_scores=seed_scores,
            max_results=args.max_results,
        )

    _log(f"returning {len(results)} result(s)")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
