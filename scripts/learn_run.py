#!/usr/bin/env python3
"""Bounded policy evolver (Tripwire-Governor S18).  NOT SCHEDULED, NOT WIRED: nothing runs this yet.

Reads trajectory rows, evolves at most 5 tighten-only gene scales per policy (loci.policy.GENES) with a
log-space evolution strategy (pop 8, sigma 0.2), accepts a candidate only if its train gain is >= 0.02 AND its
held-out gain is >= 0.01, and (only with ``--apply``) writes it into the policy's ``config['learned_genes']``
with the previous value and an evidence hash recorded for a one-command revert.

DRY-RUN IS THE DEFAULT.  Without ``--apply`` nothing is written anywhere (no patch, no log, no lock, no ledger).

Refusals (nothing is written, exit code in brackets):

* [4] the vitals ladder is not L0 (missing or stale heartbeat counts as not L0);
* [3] the C11 prerequisite is not met: there are no trajectory rows (``rows_read == 0``).  Today the live
  store has 0 rows, so a real run refuses;
* [10] the 512 MiB Job Object / BELOW_NORMAL priority could not be applied; [8] another run holds the lock;
  [9] the 20 min wall clock passed.

Limits it enforces: 1 job, 1 thread, <= 64 policies, <= 512 MiB, <= 20 min, learn dir <= 256 MiB on C:, free
space >= 2 GB, at most 1 patch per cell (policy) per 24 h, <= 10% per gene, a widening patch is never applied
(review_required), the held-out rows of a policy are scored once (scripts/policy_prereg ledger).

    python scripts/learn_run.py --policy-dir D --trajectory-dir D --learn-dir D --vitals-state F [--apply]
    python scripts/learn_run.py ... --status           # promotion state of every applied patch (read-only)
    python scripts/learn_run.py ... --revert [--apply] # restore the last patch's exact previous values

``--revert`` (like every write) needs ``--apply``; it also registers ``learned_genes`` as OFF in the flag
registry (``--flags``) with ``review_by`` 30 days out.  Stdlib only (Python 3.11+).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from loci.drpg import (  # noqa: E402
    EVOLVER_WALL_S, MAX_POLICIES_PER_CALL, MIN_FREE_BYTES, Deadline, LearnerBusy, LearnWriter, LimitsNotApplied,
    LowDiskSpace, NothingToRevert, OsFs, OsLimits, PatchConflict, PatchLog, PatchLogCorrupt, PatchRateLimited,
    PatchTampered, PolicyLimitError, QuotaExceeded, ReviewRequired, RunLock, SandboxViolation, TimeLimitExceeded,
    TrajectoryStore, apply_os_limits, apply_patch, evaluate_promotion, evolve_genome, learner_blocked,
    outcomes_from_rows, plan_patch, replay_score, revert_last_patch, row_key, split_rows,
)
from loci.policy import GENES, GeneBoundsError, PolicyBackend, read_learned_genes  # noqa: E402
from loci.vitals import check_on_c  # noqa: E402
from policy_prereg.registry import PreregStore, ScoredOnceLedger, split_fingerprint  # noqa: E402
from policy_prereg.stats import evaluate_guards  # noqa: E402

EXIT_OK = 0
EXIT_PREREQ = 3
EXIT_NOT_L0 = 4
EXIT_BUSY = 8
EXIT_TIMEOUT = 9
EXIT_LIMITS = 10
EXIT_FAILED = 7
EXIT_NOTHING_TO_REVERT = 11
EXIT_REVERT_REFUSED = 12
EXIT_STORAGE = 13   # quota, free space or sandbox

MIN_TRAIN_ROWS = 24
MIN_HELDOUT_ROWS = 6
DEFAULT_FLAGS = _ROOT / "scripts" / "policy_flags.toml"
C11_REASON = "C11 prerequisite not met: rows_read == 0 (no trajectory rows exist), so there is nothing to learn from"

__all__ = ["run_evolver", "run_revert", "run_status", "stage0_passed", "main"]


def _seed_for(policy_id: str, fingerprint: str, seed: Optional[int]) -> int:
    if seed is not None:
        return seed
    return int(hashlib.sha256(f"{policy_id}:{fingerprint}".encode("utf-8")).hexdigest()[:8], 16)


def stage0_passed(learn_dir: Path, policy_id: str) -> tuple:
    """``(passed, reason)`` from ``<learn_dir>/stage0/<policy_id>.json``.

    The record names a registration (id + sha256) and the incumbent/arm metrics.  It counts only if the
    registration still hashes to that sha256 (policy_prereg.PreregStore.verify) AND every guard of the pass
    rule (policy_prereg.stats.evaluate_guards) passes on the metrics.  A record cannot declare itself passed.
    """
    path = Path(learn_dir) / "stage0" / f"{policy_id}.json"
    if not path.exists():
        return False, "no stage0 record"
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
        PreregStore(Path(learn_dir) / "prereg").verify(rec["registration_id"], rec["registration_sha256"])
        guards = evaluate_guards(rec["incumbent"], rec["arm"])
    except Exception as exc:  # any unreadable, tampered or incomplete record is NOT a pass
        return False, f"stage0 record unusable: {type(exc).__name__}: {exc}"
    failed = sorted(name for name, ok in guards.items() if not ok)
    return (not failed), ("guards failed: " + ", ".join(failed) if failed else "all guards passed")


def run_evolver(*, policy_dir: Path, trajectory_dir: Path, learn_dir: Path, vitals_state: Path, apply: bool = False,
                clock: Callable[[], float] = time.time, fs: Optional[OsFs] = None, limits: Optional[OsLimits] = None,
                apply_limits: bool = False, seed: Optional[int] = None, protected: tuple = (),
                min_free_bytes: int = MIN_FREE_BYTES) -> tuple:
    """``(exit_code, report)``; see the module docstring.  ``apply_limits`` is False here so a test cannot cap
    its own process by accident; ``main`` passes True."""
    fs = fs or OsFs()
    report: Dict[str, Any] = {"mode": "apply" if apply else "dry-run", "rows_read": 0, "policies": []}
    blocked = learner_blocked(vitals_state, clock)
    if blocked:
        report["refused"] = blocked
        return EXIT_NOT_L0, report
    now = clock()
    learn_dir, policy_dir, trajectory_dir = Path(learn_dir), Path(policy_dir), Path(trajectory_dir)
    writer = LearnWriter(learn_dir, fs=fs, protected=protected, also_allowed=[policy_dir], min_free_bytes=min_free_bytes)
    deadline = Deadline(EVOLVER_WALL_S, clock)
    try:
        if apply_limits:
            report["limits"] = apply_os_limits(limits)
        rows_by_policy: Dict[str, list] = {}
        if trajectory_dir.is_dir():
            store = TrajectoryStore(trajectory_dir)
            ids = sorted(store.list_policies())
            if len(ids) > MAX_POLICIES_PER_CALL:
                raise PolicyLimitError(f"{len(ids)} policies; at most {MAX_POLICIES_PER_CALL}")
            for pid in ids:
                deadline.check()
                rows_by_policy[pid] = store.read(pid)
        report["rows_read"] = sum(len(r) for r in rows_by_policy.values())
        if report["rows_read"] == 0:
            report["refused"] = C11_REASON
            return EXIT_PREREQ, report
        backend = PolicyBackend(policy_dir) if policy_dir.is_dir() else None
        lock = RunLock(writer, learn_dir / "learn.lock", clock) if apply else None
        if lock:
            lock.acquire()
        try:
            ledger = ScoredOnceLedger(writer.check_path(learn_dir / "heldout_ledger.jsonl"), clock)
            log = PatchLog(learn_dir / "patches.jsonl", writer)
            for pid, rows in rows_by_policy.items():
                deadline.check()
                report["policies"].append(_evolve_one(pid, rows, backend, ledger, log, writer, now, apply, seed))
        finally:
            if lock:
                lock.release()
        report["applied"] = sum(1 for p in report["policies"] if p["status"] == "applied")
        report["proposed"] = sum(1 for p in report["policies"] if p["status"] == "proposed")
        return EXIT_OK, report
    except LimitsNotApplied as exc:
        report["refused"] = str(exc)
        return EXIT_LIMITS, report
    except LearnerBusy as exc:
        report["refused"] = str(exc)
        return EXIT_BUSY, report
    except (QuotaExceeded, LowDiskSpace, SandboxViolation) as exc:
        report["refused"] = f"{type(exc).__name__}: {exc}"
        return EXIT_STORAGE, report
    except TimeLimitExceeded as exc:
        report["failed"] = str(exc)
        return EXIT_TIMEOUT, report
    except (PolicyLimitError, PatchLogCorrupt, OSError, ValueError) as exc:
        report["failed"] = f"{type(exc).__name__}: {exc}"
        return EXIT_FAILED, report


def _evolve_one(pid: str, rows: list, backend: Optional[PolicyBackend], ledger: ScoredOnceLedger, log: PatchLog,
                writer: LearnWriter, now: float, apply: bool, seed: Optional[int]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"policy_id": pid, "rows": len(rows)}
    policy = backend.load(pid) if backend is not None else None
    if policy is None:
        return {**out, "status": "skipped", "reason": "policy_not_found"}
    stored, problems = read_learned_genes(policy.config)
    if problems:
        out["gene_problems"] = problems  # reported; the invalid entries are treated as the human default
    train, held = split_rows(rows)
    if len(train) < MIN_TRAIN_ROWS or len(held) < MIN_HELDOUT_ROWS:
        return {**out, "status": "skipped", "reason": "insufficient_rows",
                "train_rows": len(train), "heldout_rows": len(held)}
    fingerprint = split_fingerprint(row_key(r) for r in held)
    if ledger.times_scored(pid, fingerprint):
        return {**out, "status": "skipped", "reason": "heldout_already_scored"}
    try:
        log.check_rate(pid, now)
    except PatchRateLimited as exc:
        return {**out, "status": "skipped", "reason": "rate_limited", "detail": str(exc)}
    current = {g.name: stored.get(g.name, g.default) for g in GENES}
    scored = [False]

    def train_fn(genome: Dict[str, float]) -> float:
        return replay_score(policy, genome, current, train)

    def held_fn(genome: Dict[str, float]) -> float:
        if apply and not scored[0]:
            ledger.record_scoring(pid, fingerprint)  # the held-out split is consumed the first time it is scored
            scored[0] = True
        return replay_score(policy, genome, current, held)

    rng = random.Random(_seed_for(pid, fingerprint, seed))
    result = evolve_genome(current, rng, train_fn, held_fn)
    out.update({"reason": result.reason, "train_gain": result.train_gain, "heldout_gain": result.heldout_gain,
                "discarded_nonfinite": result.discarded_nonfinite})
    if not result.accepted:
        return {**out, "status": "rejected"}
    details = {"train_gain": result.train_gain, "heldout_gain": result.heldout_gain, "train_rows": len(train),
               "heldout_rows": len(held), "heldout_fingerprint": fingerprint, "pop": 8, "sigma": 0.2}
    try:
        patch = plan_patch(pid, stored, result.genome, now, details)
    except GeneBoundsError as exc:
        return {**out, "status": "rejected", "reason": f"patch_out_of_bounds: {exc}"}
    if patch is None:
        return {**out, "status": "rejected", "reason": "no_change"}
    out["patch"] = {"id": patch.id, "evidence_hash": patch.evidence_hash, "previous": patch.previous,
                    "proposed": patch.proposed, "widening": patch.widening}
    if not apply:
        return {**out, "status": "proposed"}
    try:
        apply_patch(patch, backend=backend, writer=writer, log=log, now=now)
    except (PatchRateLimited, PatchConflict, ReviewRequired) as exc:
        return {**out, "status": "skipped", "reason": type(exc).__name__, "detail": str(exc)}
    return {**out, "status": "applied"}


def run_revert(*, policy_dir: Path, learn_dir: Path, flags_path: Path, apply: bool = False,
               clock: Callable[[], float] = time.time, fs: Optional[OsFs] = None,
               protected: tuple = ()) -> tuple:
    """Revert the last applied patch (needs ``apply``).  Works at any vitals level: undoing a patch is the
    safety action, so it skips the L0 gate and the 2 GB free-space floor (the write is a few KiB)."""
    fs = fs or OsFs()
    report: Dict[str, Any] = {"mode": "apply" if apply else "dry-run"}
    writer = LearnWriter(Path(learn_dir), fs=fs, protected=protected, also_allowed=[policy_dir, flags_path],
                         min_free_bytes=0)
    try:
        log = PatchLog(Path(learn_dir) / "patches.jsonl", writer)
        applied = log.applied_patches()
        if not applied:
            raise NothingToRevert("no applied patch to revert")
        patch = applied[-1]
        report["patch"] = {"id": patch.id, "cell": patch.cell, "previous": patch.previous, "proposed": patch.proposed}
        if not apply:
            return EXIT_OK, report
        report["reverted"] = revert_last_patch(backend=PolicyBackend(Path(policy_dir)), writer=writer, log=log,
                                               registry_path=Path(flags_path), now=clock())
        return EXIT_OK, report
    except NothingToRevert as exc:
        report["refused"] = str(exc)
        return EXIT_NOTHING_TO_REVERT, report
    except (PatchTampered, PatchConflict) as exc:
        report["refused"] = f"{type(exc).__name__}: {exc}"
        return EXIT_REVERT_REFUSED, report
    except (QuotaExceeded, SandboxViolation) as exc:
        report["refused"] = f"{type(exc).__name__}: {exc}"
        return EXIT_STORAGE, report
    except (PatchLogCorrupt, OSError, ValueError) as exc:
        report["failed"] = f"{type(exc).__name__}: {exc}"
        return EXIT_FAILED, report


def run_status(*, trajectory_dir: Path, learn_dir: Path, clock: Callable[[], float] = time.time) -> tuple:
    """Read-only: the promotion state of every applied patch, from the trajectory rows written since it."""
    report: Dict[str, Any] = {"mode": "status", "patches": []}
    writer = LearnWriter(Path(learn_dir))
    try:
        log = PatchLog(Path(learn_dir) / "patches.jsonl", writer)
        store = TrajectoryStore(Path(trajectory_dir)) if Path(trajectory_dir).is_dir() else None
        for patch in log.applied_patches():
            rows = store.read(patch.cell) if store is not None else []
            outcomes, skipped = outcomes_from_rows(rows, since=patch.ts)
            passed, why = stage0_passed(Path(learn_dir), patch.cell)
            status = evaluate_promotion(outcomes, widening=patch.widening, stage0_passed=passed)
            report["patches"].append({"patch_id": patch.id, "cell": patch.cell, "state": status.state,
                                      "reasons": list(status.reasons), "outcomes": status.n, "days": status.days,
                                      "rows_without_timestamp": skipped, "stage0": why})
    except (PatchLogCorrupt, OSError, ValueError) as exc:
        report["failed"] = f"{type(exc).__name__}: {exc}"
        return EXIT_FAILED, report
    return EXIT_OK, report


def main(argv: Optional[List[str]] = None, *, clock: Callable[[], float] = time.time, fs: Optional[OsFs] = None,
         limits: Optional[OsLimits] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--policy-dir", type=Path, required=True)
    ap.add_argument("--trajectory-dir", type=Path, required=True)
    ap.add_argument("--learn-dir", type=Path, required=True, help="on C:, <= 256 MiB")
    ap.add_argument("--vitals-state", type=Path, default=None, help="required unless --revert or --status")
    ap.add_argument("--flags", type=Path, default=DEFAULT_FLAGS, help="learned-flag registry (revert only)")
    ap.add_argument("--protect", type=Path, action="append", default=[], help="extra path the learner may never write")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--apply", action="store_true", help="WRITE (patches / revert); the default is a dry-run")
    ap.add_argument("--revert", action="store_true", help="revert the last applied patch")
    ap.add_argument("--status", action="store_true", help="print the promotion state of every applied patch")
    args = ap.parse_args(argv)
    try:
        check_on_c(args.learn_dir)
    except ValueError as exc:
        print(f"learn_run: {exc}", file=sys.stderr)
        return EXIT_FAILED
    protected = tuple(args.protect)
    if args.status:
        code, report = run_status(trajectory_dir=args.trajectory_dir, learn_dir=args.learn_dir, clock=clock)
    elif args.revert:
        code, report = run_revert(policy_dir=args.policy_dir, learn_dir=args.learn_dir, flags_path=args.flags,
                                  apply=args.apply, clock=clock, fs=fs, protected=protected)
    else:
        if args.vitals_state is None:
            ap.error("--vitals-state is required to evolve")
        code, report = run_evolver(policy_dir=args.policy_dir, trajectory_dir=args.trajectory_dir,
                                   learn_dir=args.learn_dir, vitals_state=args.vitals_state, apply=args.apply,
                                   clock=clock, fs=fs, limits=limits, apply_limits=True, seed=args.seed,
                                   protected=protected)
    print(json.dumps(report, sort_keys=True, default=str), file=sys.stderr if code else sys.stdout)
    return code


if __name__ == "__main__":
    sys.exit(main())
