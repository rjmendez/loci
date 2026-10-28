# Bounded evolver and nightly consolidation (Tripwire-Governor S18)

Status: code and tests only. Nothing is scheduled, wired or enabled. Dry-run is the default everywhere.

## What it may change

At most 8 genes (5 today), each a scale in `[floor, 1.0]` on one tier limit, 1.0 being the human default:

| gene | tier limit | floor |
|---|---|---|
| `retrieval_hops_scale` | `max_retrieval_hops` | 0.5 |
| `refinement_steps_scale` | `max_refinement_steps` | 0.5 |
| `perspectives_scale` | `max_perspectives` | 0.5 |
| `fanout_scale` | `max_fanout` | 0.4 |
| `context_chars_scale` | `max_context_chars` | 0.25 |

Tighten-only, at most 10% of the current value per gene per patch, at most 1 patch per cell (policy) per 24 h.
The owner envelope's caps are not genes: `loci.policy.check_cap_change` raises `CapViolation` on any widening.
Stored in `Policy.config["learned_genes"]`. They take effect only through `Policy.get_effective_constraints()`,
and only when **both** `LOCI_LEARNED_GENES=1` is set **and** `learned_genes` is not listed in
`scripts/policy_flags.toml` (it is listed today, `review_by = 2026-11-17`; removing the entry is the owner's act).
`Policy.get_constraints()` is unchanged.

## Commands (all dry-run unless `--apply`)

```
python scripts/learn_run.py --policy-dir D --trajectory-dir D --learn-dir D --vitals-state F [--apply]
python scripts/learn_run.py ... --status            # promotion state of each applied patch (read-only)
python scripts/learn_run.py ... --revert --apply    # restore the last patch's exact previous values, flag OFF
python scripts/consolidate.py --trajectory-dir D --learn-dir D --vitals-state F [--apply]
python scripts/consolidate.py --learn-dir D --check-heartbeat      # exit 2 if last_sleep.json is > 36 h old
```

No default paths: every directory is explicit. The learn dir must be on C:.

## Refusals

| exit | cause |
|---|---|
| 3 | C11 prerequisite: no trajectory rows (`rows_read == 0`). Today there are 0, so a real run refuses. |
| 4 | vitals ladder not at L0 (missing or stale heartbeat counts as not L0) |
| 5 / 6 | free space would fall below 2 GB / learn dir would exceed 256 MiB (nothing written) |
| 8 / 9 / 10 | another run holds the lock / wall clock (evolver 20 min, consolidation 10 min) / the 512 MiB Job Object or BELOW_NORMAL priority could not be applied |
| 12 | revert refused: patch evidence no longer hashes, or the policy changed since the patch |

`--revert` works at any vitals level and below the 2 GB floor: undoing a patch is the safety action.

## Promotion

`collecting_evidence` until at least 30 distinct outcomes over at least 3 UTC days, two independent windows
(different session, at least 1 h apart) and a Stage-0 pass; then `approved_for_promotion`. A widening patch is
always `review_required` (and is never applied by the learner); a DEGENERATE verdict from
`loci.degeneracy.judge_arm` is `rollback_triggered`. A Stage-0 record counts only if its preregistration still
hashes to the recorded sha256 (`scripts/policy_prereg`) and every pass-rule guard passes on its metrics.
The held-out rows of a policy are scored once (`ScoredOnceLedger`).

## Honest limits

The ES objective is a deterministic replay proxy (the row's own reward score, a hinge when the tightened limits
would have cut a row off, a saving credit), not a quality measurement; the quality oracle (S12) is what would
make a gain real. Rollout needs S16/S17 passed and the owner's approval.
