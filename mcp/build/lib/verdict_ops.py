"""Claim-verdict recording (extracted from server.py).

The `from memcheck...` / `from qdrant_client...` imports stay FUNCTION-LOCAL:
hoisting them would turn an unimportable memcheck from a fail-open ``None``
into an import-time crash. ``asyncio`` and ``threading`` are module-level
because the running-loop detection in _record_claim_verdicts needs them.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Optional

from qdrant_ops import _embed

logger = logging.getLogger("loci-mcp")

_verdict_backend = None                # QdrantBackend for loci_verdicts (pre_answer_check)
_verdict_backend_failed = False        # permanent-failure sentinel — don't retry
_verdict_backend_lock = threading.Lock()  # guards _verdict_backend lazy-init (#106)


def _get_verdict_backend():
    """Lazy QdrantBackend for the loci_verdicts collection (pre_answer_check verdicts).

    Reuses the same Qdrant instance as investigations but in a separate collection
    so claim-check history never pollutes finding storage. Fail-open: returns None
    when Qdrant is unavailable or memcheck is not importable.
    """
    global _verdict_backend, _verdict_backend_failed
    if _verdict_backend_failed:
        return None
    if _verdict_backend is not None:
        return _verdict_backend
    with _verdict_backend_lock:
        if _verdict_backend_failed:
            return None
        if _verdict_backend is not None:
            return _verdict_backend
        qdrant_url = os.environ.get("QDRANT_URL", "")
        if not qdrant_url:
            return None
        try:
            from memcheck.qdrant import QdrantBackend
            from qdrant_client import QdrantClient
            _vb_api_key = os.environ.get("QDRANT_API_KEY", "") or None
            client = QdrantClient(url=qdrant_url, api_key=_vb_api_key, timeout=5)
            _verdict_backend = QdrantBackend(
                client,
                collection="loci_verdicts",
                embed=_embed,
                vector_name="dense",
            )
            return _verdict_backend
        except Exception as exc:
            logger.debug("Verdict backend unavailable: %s", exc)
            _verdict_backend_failed = True
            return None


def _record_claim_verdicts(
    investigation_id: str,
    claim_results: list[dict],
    *,
    record: bool,
) -> dict:
    """Record a verdict per claim to loci_verdicts and annotate claim_results in-place.

    Each claim result gains three fields: ``verdict_type`` (claim_supported /
    claim_contradicted / claim_unsupported), ``prior_occurrences`` (how many
    times this exact claim was checked before in this investigation), and
    ``verdict_conflict`` (True when the current verdict contradicts the most
    recent prior verdict — e.g. was supported before, now contradicted).
    All steps are fail-open. Returns a summary dict.
    """
    if not record:
        for cr in claim_results:
            cr.update({"verdict_type": None, "prior_occurrences": 0, "verdict_conflict": False})
        return {"recorded": 0, "qdrant": "disabled"}

    backend = _get_verdict_backend()
    if backend is None:
        for cr in claim_results:
            cr.update({"verdict_type": None, "prior_occurrences": 0, "verdict_conflict": False})
        return {"recorded": 0, "qdrant": "unavailable"}

    from memcheck.verdict import Verdict, make_signature, new_verdict, redact_excerpt

    _VERDICT_MAP = {
        "claim_ambiguous":    ("warn",  0.75, "claim has supporting evidence but cross-investigation benign baseline also exists — disambiguation required"),
        "claim_contradicted": ("flag",  0.90, "claim contradicted by negation-mismatch evidence"),
        "claim_supported":    ("allow", 0.85, "claim supported by investigation evidence"),
        "claim_unsupported":  ("warn",  0.70, "no supporting evidence found in investigation"),
    }

    # PE-gated reconsolidation (Nader 2000 / Sevenster 2013).
    # Verdict severity order: supported(1) < ambiguous(2) < unsupported(3) < contradicted(4).
    # Prediction error = |new_severity - prior_severity| / 3. High PE on an
    # established prior → verdict is provisional (recorded, not enforced) until
    # a second independent observation confirms the direction change.
    _VERDICT_SEVERITY = {
        "claim_supported": 1, "claim_ambiguous": 2,
        "claim_unsupported": 3, "claim_contradicted": 4,
    }
    _PE_HIGH_THRESH = float(os.environ.get("LOCI_PE_HIGH_THRESH", "0.5"))
    _PE_PROTECTION_MIN_OCC = int(os.environ.get("LOCI_PE_PROTECTION_MIN_OCC", "3"))

    recorded = 0
    qdrant_ok = True

    async def _process() -> None:
        nonlocal recorded, qdrant_ok
        for cr in claim_results:
            claim = str(cr.get("claim", ""))
            if cr.get("contradicted"):
                vtype = "claim_contradicted"
            elif cr.get("ambiguous"):
                # Supported but cross-investigation benign baseline also present.
                # Record as ambiguous so the signal survives in verdict history.
                vtype = "claim_ambiguous"
            elif cr.get("supported"):
                vtype = "claim_supported"
            else:
                vtype = "claim_unsupported"

            sig = make_signature("claim_check", f"{investigation_id}:{claim}")
            decision, confidence, rationale = _VERDICT_MAP[vtype]

            # Recall prior verdict by exact point-id to detect conflicts and count history.
            prior_vtype: Optional[str] = None
            prior_count: int = 0
            try:
                pid_fn = getattr(backend, "point_id", None)
                retrieve_fn = getattr(getattr(backend, "_client", None), "retrieve", None)
                if callable(pid_fn) and callable(retrieve_fn):
                    pid = pid_fn(sig)
                    hits = await asyncio.to_thread(
                        retrieve_fn,
                        collection_name="loci_verdicts",
                        ids=[pid],
                        with_payload=True,
                    )
                    if hits:
                        pl = getattr(hits[0], "payload", None)
                        if pl:
                            prior = Verdict.from_payload(dict(pl))
                            prior_vtype = prior.verdict_type
                            prior_count = prior.occurrences
            except Exception as exc:
                logger.debug("Verdict recall failed for claim %r: %s", claim[:60], exc)

            cr["verdict_type"] = vtype
            cr["prior_occurrences"] = prior_count
            # Flag a conflict whenever the verdict transitions into or out of a
            # "warning" state (contradicted or ambiguous).  Transitions between
            # supported↔ambiguous matter: a claim that was clean and now has a
            # benign baseline (or had one and now appears clean) deserves scrutiny.
            cr["verdict_conflict"] = bool(
                prior_vtype and prior_vtype != vtype and (
                    vtype in ("claim_contradicted", "claim_ambiguous") or
                    prior_vtype in ("claim_contradicted", "claim_ambiguous")
                )
            )

            refs = [
                str(r.get("evidence_id", ""))
                for r in cr.get("support_refs", [])
                if r.get("evidence_id")
            ][:5]

            # PE-gated reconsolidation: measure direction change against prior.
            provisional = False
            if prior_vtype and prior_vtype != vtype:
                prior_sev = _VERDICT_SEVERITY.get(prior_vtype, 2)
                new_sev   = _VERDICT_SEVERITY.get(vtype, 2)
                pe = abs(new_sev - prior_sev) / 3.0
                if pe >= _PE_HIGH_THRESH and prior_count >= _PE_PROTECTION_MIN_OCC:
                    provisional = True
                    rationale = (
                        rationale
                        + f" [PROVISIONAL: PE={pe:.2f}, prior={prior_vtype}×{prior_count},"
                        " requires second confirmation before enforcement]"
                    )
                    logger.debug(
                        "PE-provisional verdict for claim %r: PE=%.2f prior=%s×%d",
                        claim[:60], pe, prior_vtype, prior_count,
                    )

            v = new_verdict(
                subject_kind="memory",
                subject_signature=sig,
                subject_excerpt=redact_excerpt(f"{investigation_id}: {claim}"),
                verdict_type=vtype,
                decision=decision,
                confidence=confidence,
                rationale=rationale,
                source="rule",
                refs=refs,
                provisional=provisional,
            )
            try:
                await backend.record(v)
                recorded += 1
            except Exception as exc:
                logger.debug("Verdict record failed for claim %r: %s", claim[:60], exc)
                qdrant_ok = False

    # FastMCP dispatches sync @mcp.tool() functions inline on the running event
    # loop (fn(**kwargs), no executor).  asyncio.run() requires *no* running loop
    # and raises RuntimeError when one already exists.  Detect the situation and
    # delegate to a fresh thread that owns its own event loop instead.
    try:
        asyncio.get_running_loop()
        # A loop IS running — we are being called from a sync tool on the loop
        # thread (FastMCP inline dispatch).  Delegate to a daemon thread that
        # owns its own event loop; threading.Thread is lighter than a full
        # ThreadPoolExecutor for a one-shot fire-and-join.
        _exc: list[Exception] = []

        def _run() -> None:
            try:
                asyncio.run(_process())
            except Exception as e:
                _exc.append(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join()
        if _exc:
            raise _exc[0]
    except RuntimeError:
        # No running loop — safe to call asyncio.run() directly.
        try:
            asyncio.run(_process())
        except Exception as exc:
            logger.debug("_record_claim_verdicts failed: %s", exc)
            qdrant_ok = False
    except Exception as exc:
        logger.debug("_record_claim_verdicts failed: %s", exc)
        qdrant_ok = False

    return {"recorded": recorded, "qdrant": "ok" if qdrant_ok else "partial"}
