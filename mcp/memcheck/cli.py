"""memcheck CLI — the PreToolUse hook entrypoint (audit-only).

Invoked as ``python -m memcheck.cli <subcommand>``. The headline subcommand is
``check-action`` (also the default), wired to a Claude Code ``PreToolUse`` hook
that fires on every action-bearing tool call. Because it sits in the hot path of
every tool call, it is built to three rules, in priority order:

1. **FAST** — lazy imports, a deterministic hash embedder (no model load), a
   single qdrant ``retrieve`` + one ``upsert``, and a short (~1s) qdrant
   timeout. Worst case is a quick exit 0.
2. **FAIL-OPEN** — *any* error (bad JSON, missing fields, import error, qdrant
   down, timeout) results in writing nothing to stdout and ``sys.exit(0)``.
   ``check-action`` is structurally incapable of a non-zero exit or non-empty
   stdout on either the normal or the error path.
3. **AUDIT-ONLY** — observe and record; never block, never auto-approve. The
   hook contract reads: exit 0 + empty stdout = "no opinion, proceed normally".
   We never emit ``{"decision":"approve"}`` (that would bypass the user's
   permission prompt) and never emit a block / exit 2 (that would block the
   tool). All diagnostics go to the audit log file or stderr, never to the
   structured stdout the hook parses.

``tail`` and ``stats`` are human-facing helpers (stdout is fine there — they are
not on the hook path).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ``memcheck`` itself is stdlib-only and cheap to import. Everything heavy
# (qdrant_client) is imported lazily inside check_action so the hot path pays
# for it only when qdrant is actually reachable.
from .backend import VerdictBackend
from .verdict import make_signature, new_verdict, redact_excerpt

__all__ = [
    "EMBED_DIM",
    "COLLECTION",
    "DEFAULT_AUDIT_LOG",
    "QDRANT_TIMEOUT_S",
    "PROMOTE_AFTER",
    "hash_embed",
    "redact_tool_input",
    "build_descriptor",
    "audit_log_path",
    "process_action",
    "process_code",
    "check_action",
    "main",
]

# Tools whose payloads carry a file path we should run code checks against.
_CODE_TOOLS = ("Write", "Edit", "MultiEdit")

EMBED_DIM = 384
COLLECTION = "hermes_verdicts"
VECTOR_NAME = "dense"
QDRANT_TIMEOUT_S = 1.0
PROMOTE_AFTER = 3  # mirrors EmlConfig.promote_after; a block-class verdict at
# this many occurrences is what enforce-mode WOULD act on.
DEFAULT_AUDIT_LOG = "~/.hermes/memcheck-audit.jsonl"

# Decisions that count as a "block-class" verdict for would_flag accounting.
_BLOCKING = ("flag", "warn", "quarantine")

# Keys whose values are secret-ish and must be dropped/blanked before the
# tool_input ever touches a signature, a verdict excerpt, or the audit log.
_SECRET_KEY_HINTS = (
    "token",
    "secret",
    "password",
    "passwd",
    "key",
    "authorization",
    "auth",
    "cookie",
    "credential",
)
_SECRET_REDACTION = "[REDACTED]"
# Bound on any single stringified value before it enters the descriptor/log.
_VALUE_MAX_CHARS = 256


# --------------------------------------------------------------------------- #
# Deterministic hash embedder (NO model load — must stay cheap)
# --------------------------------------------------------------------------- #
def hash_embed(text: str) -> list[float]:
    """Deterministic text -> ``EMBED_DIM``-float vector in [-1, 1].

    Seeds a byte stream from ``sha256(text)`` and tiles its bytes to
    ``EMBED_DIM`` floats. No model, no network — identical text always yields an
    identical vector, which is all the EXACT-match recall path needs (recall is
    keyed on the stable point id, not on vector proximity). Returning a real
    384-dim vector keeps the qdrant collection's named ``dense`` vector valid.
    """
    if text is None:
        text = ""
    # Expand the 32-byte digest deterministically to >= EMBED_DIM bytes by
    # hashing (digest || counter) repeatedly.
    out: list[float] = []
    counter = 0
    seed = text.encode("utf-8")
    while len(out) < EMBED_DIM:
        block = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        for b in block:
            if len(out) >= EMBED_DIM:
                break
            # byte 0..255 -> float in [-1, 1]
            out.append((b / 127.5) - 1.0)
        counter += 1
    return out


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #
def _looks_secret(key: str) -> bool:
    k = key.lower()
    return any(hint in k for hint in _SECRET_KEY_HINTS)


def redact_tool_input(obj: Any) -> Any:
    """Recursively redact secret-ish values and bound long strings.

    - Dict values under a secret-ish key are replaced with ``[REDACTED]``.
    - Any string value is length-bounded via ``redact_excerpt`` (truncation
      only bounds length; secret *content* is removed by the key match above).
    - Lists/tuples are walked element-wise; other scalars pass through.

    The result is JSON-safe so it can be compacted into the descriptor and the
    audit excerpt without leaking raw secrets.
    """
    if isinstance(obj, dict):
        redacted: dict[str, Any] = {}
        for k, v in obj.items():
            key = str(k)
            if _looks_secret(key):
                redacted[key] = _SECRET_REDACTION
            else:
                redacted[key] = redact_tool_input(v)
        return redacted
    if isinstance(obj, (list, tuple)):
        return [redact_tool_input(v) for v in obj]
    if isinstance(obj, str):
        return redact_excerpt(obj, max_chars=_VALUE_MAX_CHARS)
    return obj


def _compact_json(obj: Any) -> str:
    """Deterministic, compact JSON (sorted keys, no spaces) — safe on anything."""
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:  # noqa: BLE001 — never let serialization break the hook
        return "{}"


def build_descriptor(tool_name: str, tool_input: Any) -> str:
    """Normalized action descriptor: ``"<tool_name> <compact redacted input>"``.

    ``tool_input`` is redacted *before* it is serialized, so neither the
    signature nor the descriptor (which becomes the verdict excerpt and is what
    we log) can carry a raw secret.
    """
    redacted = redact_tool_input(tool_input if tool_input is not None else {})
    return f"{tool_name} {_compact_json(redacted)}"


# --------------------------------------------------------------------------- #
# Audit log
# --------------------------------------------------------------------------- #
def audit_log_path() -> Path:
    """Resolve the audit-log path from ``MEMCHECK_AUDIT_LOG`` (env) or default."""
    raw = os.environ.get("MEMCHECK_AUDIT_LOG") or DEFAULT_AUDIT_LOG
    return Path(raw).expanduser()


def _append_audit_line(record: dict) -> None:
    """Append one JSON line to the audit log. Best-effort — never raises."""
    try:
        path = audit_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception:  # noqa: BLE001 — logging must never break the hook
        pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# --------------------------------------------------------------------------- #
# qdrant wiring (lazy; short timeout; create collection if missing)
# --------------------------------------------------------------------------- #
def _build_qdrant_backend() -> Optional[VerdictBackend]:
    """Construct a QdrantBackend against the configured URL, or None on failure.

    Imports ``qdrant_client`` lazily, connects with a short timeout, ensures the
    ``hermes_verdicts`` collection exists (creating it with a 384-dim cosine
    ``dense`` vector if missing), and wires the deterministic ``hash_embed`` so
    no embedding model is ever loaded. Any failure (import error, connection
    refused, timeout) returns None so the caller logs ``qdrant_unavailable`` and
    exits 0 rather than stalling.
    """
    from .qdrant import QdrantBackend  # local import keeps package import cheap

    from qdrant_client import QdrantClient
    from qdrant_client import models as qmodels

    url = os.environ.get("QDRANT_URL")
    client = QdrantClient(url=url, timeout=QDRANT_TIMEOUT_S)

    # Ensure the collection exists. collection_exists is a single fast call; if
    # it raises (qdrant down), we propagate so the caller treats qdrant as
    # unavailable.
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                VECTOR_NAME: qmodels.VectorParams(
                    size=EMBED_DIM, distance=qmodels.Distance.COSINE
                )
            },
        )

    return QdrantBackend(
        client,
        collection=COLLECTION,
        embed=hash_embed,
        vector_name=VECTOR_NAME,
    )


# --------------------------------------------------------------------------- #
# process_action — the shared testable core (used by check-action AND daemon)
# --------------------------------------------------------------------------- #
def process_action(payload: dict, engine) -> dict:
    """Audit-only observe+record for one PreToolUse payload.

    Shared core for both ``cli check-action`` (in-process fallback) and the
    warm ``memcheck daemon``. Takes the parsed hook ``payload`` and a
    ``VerdictEngine`` (or ``None`` for the qdrant-unavailable case); the backend
    is read off ``engine.backend``. Returns the audit record it appended (also
    returned for tests); callers on the hook path discard it.

    Behavior:
      1. Extract ``tool_name`` / ``tool_input`` / ``session_id``; redact the
         input; build the descriptor + signature.
      2. EXACT-match recall: look up the existing point id for the signature to
         read prior ``occurrences`` and the prior decision.
      3. Record/observe: upsert this ``action`` verdict (occurrences read-
         modify-write happens in the backend), decision ``flag``, source
         ``rule``, verdict_type ``observed_action``.
      4. ``would_flag`` = a promoted block-class verdict exists, i.e. the
         decision is block-class AND ``occurrences + 1 >= PROMOTE_AFTER``.
      5. Append one audit line.

    Every step is wrapped fail-open: a raising backend degrades to
    ``qdrant: "unavailable"`` and still produces an audit line. This function
    never raises and never writes to stdout.
    """
    import asyncio

    from .verdict import Verdict

    backend: Optional[VerdictBackend] = (
        getattr(engine, "backend", None) if engine is not None else None
    )

    tool_name = str(payload.get("tool_name", "") or "")
    tool_input = payload.get("tool_input", {})
    session_id = payload.get("session_id")

    descriptor = build_descriptor(tool_name, tool_input)
    signature = make_signature("action", descriptor)
    excerpt = redact_excerpt(descriptor)

    qdrant_status = "ok" if backend is not None else "unavailable"
    prior_occurrences = 0
    prior_decision: Optional[str] = None

    async def _observe() -> None:
        nonlocal prior_occurrences, prior_decision
        # --- EXACT-match recall by stable point id (no embedding search) ---
        point_id_fn = getattr(backend, "point_id", None)
        retrieve_fn = getattr(getattr(backend, "_client", None), "retrieve", None)
        if callable(point_id_fn) and callable(retrieve_fn):
            pid = point_id_fn(signature)
            existing = await asyncio.to_thread(
                retrieve_fn,
                collection_name=COLLECTION,
                ids=[pid],
                with_payload=True,
            )
            if existing:
                pl = getattr(existing[0], "payload", None)
                if pl:
                    prior = Verdict.from_payload(dict(pl))
                    prior_occurrences = prior.occurrences
                    prior_decision = prior.decision
        else:
            # Injected (test) backend without a qdrant client: recall by kind
            # and match on signature so occurrences accumulate across calls.
            scored = await backend.recall(descriptor, hash_embed(descriptor), "action", 50)
            for s in scored:
                if s.verdict.subject_signature == signature:
                    prior_occurrences = s.verdict.occurrences
                    prior_decision = s.verdict.decision
                    break

        # --- Record / observe: upsert the action verdict ---
        verdict = new_verdict(
            subject_kind="action",
            subject_signature=signature,
            subject_excerpt=excerpt,
            verdict_type="observed_action",
            decision="flag",
            confidence=1.0,
            rationale="audit-only observed CLI/tool action",
            source="rule",
        )
        embedding = hash_embed(descriptor)
        rwe = getattr(backend, "record_with_embedding", None)
        if callable(rwe):
            await rwe(verdict, embedding)
        else:
            await backend.record(verdict)

    if backend is not None:
        try:
            asyncio.run(_observe())
        except Exception:  # noqa: BLE001 — fail-open: degrade to unavailable
            qdrant_status = "unavailable"

    # occurrences AFTER this observation (prior + this one).
    occurrences = prior_occurrences + 1
    # would_flag: what enforce-mode WOULD do — a promoted block-class verdict.
    decision_for_flag = prior_decision or "flag"
    would_flag = decision_for_flag in _BLOCKING and occurrences >= PROMOTE_AFTER

    record = {
        "ts": _now_iso(),
        "tool_name": tool_name,
        "signature": signature,
        "occurrences": occurrences,
        "would_flag": would_flag,
        "qdrant": qdrant_status,
        "session_id": session_id,
    }
    _append_audit_line(record)
    return record


def process_code(payload: dict, engine, *, repo_root: Optional[str] = None) -> dict:
    """Audit-only code-hallucination check for one PostToolUse payload.

    The PostToolUse counterpart of :func:`process_action`. Reads ``tool_name``
    (``Write`` / ``Edit`` / ``MultiEdit``) and the target ``file_path`` from
    ``tool_input``; if that resolves to an existing ``*.py`` file, runs the
    vendored static checker via ``run_code_checks`` and records each resulting
    ``code`` verdict to the engine's backend (fail-open). Appends ONE audit line.

    Audit line on a checked file::

        {ts, event:"code", tool_name, file, n_issues, codes:[...], qdrant}

    A non-``.py`` path, a missing file, or no resolvable path records nothing
    and writes a ``skip`` audit line instead. Like ``process_action`` this never
    raises and never writes to stdout; every recording step is wrapped fail-open
    so qdrant being down degrades to ``qdrant: "unavailable"`` but still audits.
    """
    import asyncio

    from .checks import run_code_checks

    backend: Optional[VerdictBackend] = (
        getattr(engine, "backend", None) if engine is not None else None
    )

    tool_name = str(payload.get("tool_name", "") or "")
    tool_input = payload.get("tool_input", {})
    if not isinstance(tool_input, dict):
        tool_input = {}

    raw_path = tool_input.get("file_path")
    file_str = str(raw_path) if raw_path else ""

    # Resolve relpath for the audit line the same way the verdicts do, so the
    # audit log never carries a dev-local absolute path either.
    def _audit_relpath() -> str:
        if not file_str:
            return ""
        p = Path(file_str)
        if repo_root:
            try:
                return p.resolve().relative_to(Path(repo_root).resolve()).as_posix()
            except (ValueError, OSError):
                pass
        return p.name

    # Skip: no path, not a .py file, or the file doesn't exist.
    is_py = file_str.endswith(".py")
    exists = bool(file_str) and Path(file_str).is_file()
    if not file_str or not is_py or not exists:
        record = {
            "ts": _now_iso(),
            "event": "code",
            "tool_name": tool_name,
            "file": _audit_relpath(),
            "skipped": True,
            "qdrant": "ok" if backend is not None else "unavailable",
        }
        _append_audit_line(record)
        return record

    relpath = _audit_relpath()
    verdicts = run_code_checks(file_str, repo_root=repo_root)
    qdrant_status = "ok" if backend is not None else "unavailable"

    if backend is not None and verdicts:
        async def _record_all() -> None:
            for verdict in verdicts:
                embedding = hash_embed(verdict.subject_excerpt)
                rwe = getattr(backend, "record_with_embedding", None)
                if callable(rwe):
                    await rwe(verdict, embedding)
                else:
                    await backend.record(verdict)

        try:
            asyncio.run(_record_all())
        except Exception:  # noqa: BLE001 — fail-open: degrade to unavailable
            qdrant_status = "unavailable"

    record = {
        "ts": _now_iso(),
        "event": "code",
        "tool_name": tool_name,
        "file": relpath,
        "n_issues": len(verdicts),
        "codes": [v.verdict_type for v in verdicts],
        "qdrant": qdrant_status,
    }
    _append_audit_line(record)
    return record


class _BackendEngine:
    """Minimal engine shim exposing ``.backend`` for ``process_action``.

    ``process_action`` only reads ``engine.backend``; the in-process
    ``check-action`` path constructs a backend directly (no full
    ``VerdictEngine`` needed), so this thin wrapper keeps the two callers on the
    identical code path without dragging the engine import into the hot path.
    """

    __slots__ = ("backend",)

    def __init__(self, backend: Optional[VerdictBackend]) -> None:
        self.backend = backend


def check_action(payload: dict, backend: Optional[VerdictBackend]) -> dict:
    """Backward-compatible wrapper: observe+record given an injected backend.

    Frozen API for the in-process fallback path and existing tests. Delegates
    to the shared :func:`process_action`, wrapping ``backend`` in a tiny shim
    so both ``check-action`` and the daemon run identical logic.
    """
    return process_action(payload, _BackendEngine(backend))


def _check_action_from_stdin() -> int:
    """Thin stdin/exit wrapper around ``check_action`` — the hook path.

    Reads+parses stdin JSON, builds the qdrant backend, and runs the observe.
    EVERYTHING is wrapped so the only outcomes are: write nothing to stdout and
    return 0. On any failure it optionally drops a debug line to the audit log
    and returns 0. This function is structurally incapable of returning non-zero
    or writing to stdout.
    """
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, dict):
            payload = {}

        backend: Optional[VerdictBackend] = None
        try:
            backend = _build_qdrant_backend()
        except Exception:  # noqa: BLE001 — qdrant unreachable/import error
            backend = None
            _append_audit_line(
                {
                    "ts": _now_iso(),
                    "event": "qdrant_unavailable",
                    "tool_name": str(payload.get("tool_name", "") or ""),
                    "qdrant": "unavailable",
                    "session_id": payload.get("session_id"),
                }
            )

        check_action(payload, backend)
    except Exception:  # noqa: BLE001 — absolute fail-open boundary
        # Never let anything escape to stdout or a non-zero exit.
        pass
    return 0


# --------------------------------------------------------------------------- #
# tail / stats — human-facing (stdout fine; NOT the hook path)
# --------------------------------------------------------------------------- #
def _cmd_tail(argv: list[str]) -> int:
    n = 20
    if "-n" in argv:
        try:
            n = int(argv[argv.index("-n") + 1])
        except (ValueError, IndexError):
            n = 20
    path = audit_log_path()
    if not path.exists():
        print(f"(no audit log at {path})")
        return 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"(could not read {path}: {exc})")
        return 0
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            print(json.dumps(rec, indent=2, sort_keys=True))
        except json.JSONDecodeError:
            print(line)
    return 0


def _cmd_stats(argv: list[str]) -> int:
    import asyncio

    from .engine import EmlConfig, VerdictEngine

    try:
        backend = _build_qdrant_backend()
    except Exception as exc:  # noqa: BLE001
        print(f"qdrant unavailable: {exc}")
        return 0
    if backend is None:
        print("qdrant unavailable")
        return 0
    engine = VerdictEngine(backend, EmlConfig())
    stats = asyncio.run(engine.stats())
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


# --------------------------------------------------------------------------- #
# entrypoint
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    sub = argv[0] if argv else "check-action"
    rest = argv[1:]

    if sub in ("check-action", "check", ""):
        return _check_action_from_stdin()
    if sub == "check-code":
        return _cmd_check_code(rest)
    if sub == "tail":
        return _cmd_tail(rest)
    if sub == "stats":
        return _cmd_stats(rest)
    if sub == "daemon":
        return _cmd_daemon(rest)

    # Unknown subcommand: be conservative. If invoked as the default hook with a
    # stray arg, still fail-open to 0; otherwise print usage to stderr.
    sys.stderr.write(
        "usage: python -m memcheck.cli "
        "[check-action|check-code FILE...|tail [-n N]|stats|daemon [--socket PATH]]\n"
    )
    return 0


def _cmd_check_code(argv: list[str]) -> int:
    """Manual/CI code-hallucination check over one or more files.

    ``python -m memcheck.cli check-code FILE [FILE...]``. Prints each file's
    issues to stdout (human-facing — not on the hook path, so stdout is fine)
    and, when a qdrant backend is reachable, records the verdicts via the shared
    :func:`process_code`. Standalone-friendly: with qdrant down it still prints
    the issues. Always returns 0 (audit-only — never a non-zero/blocking exit).
    """
    from .checks import run_code_checks

    files = [a for a in argv if not a.startswith("-")]
    if not files:
        sys.stderr.write("usage: python -m memcheck.cli check-code FILE [FILE...]\n")
        return 0

    # Build a backend once if qdrant is reachable; recording is best-effort.
    backend: Optional[VerdictBackend] = None
    try:
        backend = _build_qdrant_backend()
    except Exception:  # noqa: BLE001 — standalone-friendly: print without recording
        backend = None

    total_issues = 0
    for f in files:
        verdicts = run_code_checks(f)
        for v in verdicts:
            total_issues += 1
            # subject_excerpt is "relpath:line CODE"; rationale is the message.
            print(f"{v.subject_excerpt}: {v.rationale}")
        if backend is not None:
            try:
                process_code(
                    {"tool_name": "Write", "tool_input": {"file_path": f}},
                    _BackendEngine(backend),
                )
            except Exception:  # noqa: BLE001 — recording must never break the CLI
                pass

    if total_issues == 0:
        print("no code-hallucination issues found")
    return 0


def _cmd_daemon(argv: list[str]) -> int:
    """Run the warm memcheck daemon (long-running; serves the hook client).

    ``python -m memcheck.cli daemon [--socket PATH]``. Imported lazily so the
    hot ``check-action`` path never pays for the daemon module.
    """
    from .daemon import serve

    socket_path: Optional[str] = None
    if "--socket" in argv:
        try:
            socket_path = argv[argv.index("--socket") + 1]
        except IndexError:
            sys.stderr.write("memcheck daemon: --socket requires a PATH\n")
            return 2
    return serve(socket_path=socket_path)


if __name__ == "__main__":
    sys.exit(main())
