# REGRESSION FIXTURE -- DO NOT IMPORT, DO NOT RUN, DO NOT COPY BACK INTO THE TREE.
#
# BUG D (producer): MEMORY_DIR + _get_ladybug, which opens MEMORY_DIR / "graph.ladybug".
#
# Source : mcp/server.py @ c1c40a9165a3282df6581d981c9401812da1cddc  (slice L159-166, L218-287)
#          `git show c1c40a9165a3:mcp/server.py`
# Status : THIS BUG IS FIXED IN THE REAL TREE. This file preserves the
#          pre-fix code purely so the callgraph tool can be re-pointed at it
#          and proven to still surface the defect. It is parsed by ast only
#          and is excluded from the analyzed corpus (config.SELF_PACKAGE_REL).
# ---- VERBATIM BODY BELOW (do not edit; see test_regression_real_bugs.py) ----
# --- mcp/server.py lines 159-166 @ c1c40a9165a3 ---
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MEMORY_DIR = Path(os.environ.get(
    "HERMES_MEMORY_DIR",
    Path.home() / ".hermes" / "memory-sessions",
))

# --- mcp/server.py lines 218-287 @ c1c40a9165a3 ---
# ---------------------------------------------------------------------------
# LadybugDB graph store (primary relationship/graph backend) — fail-open like Qdrant.
# Mirrors findings/entities/derivation into an embedded graph and backs the
# entity-lookup / related-cases / contamination / code-symbol paths. If ladybug or
# the module is unavailable, every consumer degrades to the pre-existing path.
# ---------------------------------------------------------------------------
_ladybug_store = None                     # LadybugStore singleton once initialized
_ladybug_failed = False                   # PERMANENT-failure latch (ladybug unimportable) — don't retry
_ladybug_last_attempt = 0.0               # monotonic ts of last TRANSIENT init failure
_LADYBUG_RETRY_SECONDS = 30               # backoff before retrying after a transient failure
_ladybug_lock = threading.Lock()


def _get_ladybug():
    """Lazy, fail-open LadybugStore singleton. Returns None if unavailable.

    Distinguishes a genuinely UNRECOVERABLE failure (ladybug is not importable) — which
    latches permanently so we stop retrying — from a TRANSIENT one (another process
    holds Kuzu's single-writer lock, or a transient IO error at open time), which does
    NOT latch: a later call retries after a short backoff so the code graph self-heals
    once the other writer releases the lock. Never raises.
    """
    global _ladybug_store, _ladybug_failed, _ladybug_last_attempt
    if _ladybug_store is not None:
        return _ladybug_store
    if _ladybug_failed:
        return None
    with _ladybug_lock:
        if _ladybug_store is not None:
            return _ladybug_store
        if _ladybug_failed:
            return None
        # Back off between transient-failure retries so we don't hammer a held lock.
        if _ladybug_last_attempt and (time.monotonic() - _ladybug_last_attempt) < _LADYBUG_RETRY_SECONDS:
            return None
        try:
            from graph import ladybug_store as _kz
            if not getattr(_kz, "_HAS_LADYBUG", True):
                # ladybug itself isn't importable — unrecoverable, latch permanently.
                _ladybug_failed = True
                logger.warning("LadybugDB not importable — graph features disabled (permanent).")
                return None
            MEMORY_DIR.mkdir(parents=True, exist_ok=True)  # ladybug won't create parents
            ks = _kz.LadybugStore(str(MEMORY_DIR / "graph.ladybug"))
            if not ks.available():
                # Import worked but open failed — almost always another process holds
                # the single-writer lock. Treat as TRANSIENT: retry after the backoff.
                _ladybug_last_attempt = time.monotonic()
                logger.warning("LadybugDB store unavailable (lock contention or transient IO?) "
                               "— will retry after %ss.", _LADYBUG_RETRY_SECONDS)
                return None
            _ladybug_store = ks
            _ladybug_last_attempt = 0.0
        except ImportError as exc:
            # graph module / ladybug genuinely missing — unrecoverable, latch permanently.
            _ladybug_failed = True
            logger.warning("LadybugDB graph module missing (%r) — graph features disabled (permanent).", exc)
            return None
        except Exception as exc:  # fail-open — never break the server on graph init
            # Unknown/transient error (e.g. IO on mkdir/open) — do NOT latch; retry later.
            _ladybug_last_attempt = time.monotonic()
            logger.warning("LadybugDB graph init failed (%r) — will retry after %ss.", exc, _LADYBUG_RETRY_SECONDS)
            return None
    # One-time backfill of pre-existing findings, guarded by an empty-graph check.
    try:
        _ladybug_backfill_if_empty(_ladybug_store)
    except Exception as exc:
        logger.debug("LadybugDB backfill skipped (fail-open): %r", exc)
    return _ladybug_store


