"""Contagion check — find the lineage contaminated by a hallucinated seed.

When a self-generated hallucination (e.g. a fabricated endpoint
``http://localhost:8080/v1/foo``) is stored as a finding, downstream
infra/findings get built referencing it and many memories reinforce it. Once
testing reveals the seed never existed, the whole contaminated lineage must be
identified so it can be retracted (reversibly) — not just the seed.

``find_contamination`` computes that cluster from three OR'd signals:

- **Entity anchor** — findings sharing >= ``min_shared_entities`` *distinctive*
  entities (URLs / hosts / paths / identifiers) with a seed. Rare/structured
  entities are weighted over common words so a shared ``localhost:8080`` counts
  but a shared ``the`` does not.
- **Semantic** — a precomputed set of finding ids that qdrant found near a seed,
  passed in by the caller (the tool computes these; this pure function just
  unions them in).
- **Derivation** — any finding whose ``derived_from`` chain transitively reaches
  a seed (forward-derivation links recorded at store time).

The seeds themselves are always included. Pure + fail-safe: a malformed finding
is skipped, never raised; ``derived_from`` cycles are guarded.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Union

__all__ = ["find_contamination"]

# Entity buckets considered "distinctive" — structured identifiers an agent
# would not coincidentally share between unrelated findings. Common-word
# buckets (if any) are ignored so the anchor stays precise.
_DISTINCTIVE_BUCKETS = (
    "urls", "url", "hosts", "hostnames", "host", "paths", "path",
    "ips", "ip", "hashes", "cves", "emails", "identifiers", "endpoints",
)


def _finding_id(finding: dict, index: int) -> str:
    """Stable id for a finding: prefer an explicit id, else derive from index."""
    fid = finding.get("id") or finding.get("finding_id")
    if fid:
        return str(fid)
    inv = finding.get("investigation_id")
    return f"{inv}:{index}" if inv else f"finding:{index}"


def _distinctive_entities(raw: Union[dict, set, list, None]) -> set[str]:
    """Reduce ``entities_of(text)`` output to a set of distinctive entity tokens.

    Accepts the typed-bucket dict the server's ``_extract_entities`` returns
    (``{"ips": [...], "hostnames": [...], ...}``), or a flat set/list. From a
    typed dict only the distinctive buckets are kept; from a flat collection
    every element is kept (the caller already chose what to pass).
    """
    out: set[str] = set()
    if raw is None:
        return out
    if isinstance(raw, dict):
        for bucket, values in raw.items():
            if str(bucket).lower() not in _DISTINCTIVE_BUCKETS:
                continue
            if isinstance(values, (list, tuple, set)):
                for v in values:
                    token = str(v).strip().lower()
                    if token:
                        out.add(token)
            elif values:
                token = str(values).strip().lower()
                if token:
                    out.add(token)
        return out
    if isinstance(raw, (set, list, tuple)):
        for v in raw:
            token = str(v).strip().lower()
            if token:
                out.add(token)
    return out


def _normalize_seed_ids(seed_ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for sid in seed_ids or []:
        s = str(sid)
        if s and s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def find_contamination(
    seed_ids: Iterable[str],
    findings: list[dict],
    *,
    entities_of: Callable[[str], Union[dict, set]],
    semantic_neighbor_ids: Optional[Iterable[str]] = None,
    min_shared_entities: int = 1,
) -> dict:
    """Compute the contaminated cluster reachable from ``seed_ids``.

    Parameters
    ----------
    seed_ids:
        Finding ids known to be hallucinated (the contamination origin).
    findings:
        All findings in scope, plain dicts. Each may carry ``id``/``finding_id``,
        ``text``, and an optional ``derived_from`` (str or list of finding ids).
    entities_of:
        Injected entity extractor (the server's ``_extract_entities``). Given a
        finding's text it returns a typed-bucket dict or a flat set/list of
        entities. Distinctive (URL/host/path/identifier) entities are weighted.
    semantic_neighbor_ids:
        Optional precomputed ids qdrant found near a seed; unioned in verbatim.
    min_shared_entities:
        Minimum count of distinctive entities a finding must share with a seed
        to be flagged by the entity anchor.

    Returns
    -------
    ``{"contaminated_ids": [...], "reasons": {id: ["entity:...", "semantic:...",
    "derived_from:<seed>"]}}``. Seeds are always included (reason ``"seed"``).
    Pure + fail-safe.
    """
    seeds = _normalize_seed_ids(seed_ids)
    seed_set = set(seeds)
    reasons: dict[str, list[str]] = {}

    def _add_reason(fid: str, reason: str) -> None:
        bucket = reasons.setdefault(fid, [])
        if reason not in bucket:
            bucket.append(reason)

    # Index findings by id; capture text, distinctive entities, derived_from.
    by_id: dict[str, dict] = {}
    entities_by_id: dict[str, set[str]] = {}
    derived_edges: dict[str, set[str]] = {}  # child id -> set(parent ids)

    for index, finding in enumerate(findings or []):
        if not isinstance(finding, dict):
            continue
        try:
            fid = _finding_id(finding, index)
            by_id[fid] = finding

            text = str(finding.get("text", "") or "")
            try:
                ents = _distinctive_entities(entities_of(text)) if text else set()
            except Exception:
                ents = set()
            entities_by_id[fid] = ents

            raw_parents = finding.get("derived_from")
            parents: set[str] = set()
            if isinstance(raw_parents, str):
                if raw_parents.strip():
                    parents.add(raw_parents.strip())
            elif isinstance(raw_parents, (list, tuple, set)):
                for p in raw_parents:
                    ps = str(p).strip()
                    if ps:
                        parents.add(ps)
            if parents:
                derived_edges[fid] = parents
        except Exception:
            # Fail-safe: a malformed finding is skipped, never raised.
            continue

    contaminated: set[str] = set(seed_set)
    for sid in seeds:
        _add_reason(sid, "seed")

    # --- Semantic: union in the precomputed neighbor ids verbatim. ---
    for nid in (semantic_neighbor_ids or []):
        try:
            n = str(nid)
        except Exception:
            continue
        if not n:
            continue
        if n not in contaminated:
            contaminated.add(n)
        if n not in seed_set:
            _add_reason(n, "semantic")

    # --- Entity anchor: distinctive entities shared with ANY seed. ---
    seed_entities: dict[str, set[str]] = {
        sid: entities_by_id.get(sid, set()) for sid in seeds
    }
    threshold = max(1, int(min_shared_entities))
    for fid, ents in entities_by_id.items():
        if fid in seed_set or not ents:
            continue
        for sid, s_ents in seed_entities.items():
            shared = ents & s_ents
            if len(shared) >= threshold:
                contaminated.add(fid)
                _add_reason(fid, "entity:" + ",".join(sorted(shared)))
                break

    # --- Derivation: transitively include any finding whose derived_from chain
    # reaches the contaminated set (which always contains the seeds, and is
    # grown by the entity + semantic signals). Walking to the contaminated set
    # rather than only to a seed propagates contamination through intermediate
    # contaminated findings (e.g. seed -> entity-hit f2 -> derived f3). Computed
    # as a cycle-guarded fixpoint so order doesn't matter.
    def _reaches_contaminated(start: str, targets: set[str]) -> Optional[str]:
        """Return a target id reached via derived_from from ``start``, else None."""
        stack = list(derived_edges.get(start, ()))
        visited: set[str] = {start}
        while stack:
            parent = stack.pop()
            if parent in visited:
                continue
            visited.add(parent)
            if parent in targets:
                return parent
            stack.extend(derived_edges.get(parent, ()))
        return None

    changed = True
    while changed:
        changed = False
        for fid in by_id:
            if fid in contaminated:
                continue
            reached = _reaches_contaminated(fid, contaminated)
            if reached is not None:
                contaminated.add(fid)
                _add_reason(fid, f"derived_from:{reached}")
                changed = True

    # Deterministic order: seeds first (in input order), then the rest sorted.
    rest = sorted(c for c in contaminated if c not in seed_set)
    contaminated_ids = [s for s in seeds if s in contaminated] + rest

    return {"contaminated_ids": contaminated_ids, "reasons": reasons}
