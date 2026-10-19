# agentd: bounded delegation of coding agents

Tripwire-Governor slices S10 (core) and S11 (containment). Hermes asks agentd to run a seat-billed coding agent
(Claude Code, Copilot CLI); agentd gives the run a lease, a git worktree and a Windows Job Object, supervises it and
writes one summary row. Code: `loci/agentd.py`, `loci/agentd_job_win.py`. Nothing is wired into Hermes or a scheduler.

## Flag

agentd is **OFF** unless `LOCI_AGENTD=1`. With the flag off, `Agentd.start` refuses with `flag_off` before it touches
git, the disk, the run store or the OS (a test proves it). Read-write delegation is a second, separate gate: it needs
`AgentdConfig.rw_enabled`, a per-run `rw_flag` and a containment object, and it stays off until the restricted-token test
passes (see Containment below).

## Lease (decision 9)

| limit | default | hard |
|---|---|---|
| wall clock | 20 min | 60 min (a request above it is refused `wall_over_hard`) |
| turns | 60 | 150 (`turns_over_hard`) |
| runs per local day | 12 | `run_count_limit`; survives a restart (counted from the run store) |
| concurrent runs | 2 at L0/L1, 1 at L2, 0 at L3/L4 | `concurrency_limit`, or `governor_level` at L2+ for a second run |

The wall limit stops a run when `elapsed >= wall`; the turn limit when `turns > limit`. Governor levels come from
`loci/vitals.py` (`~/.loci/vitals_state.json`); an unreadable, stale or nonsense level refuses with
`governor_unavailable` (fail closed).

## Refusal codes (first failing check wins)

`flag_off`, `paused` (the PAUSE file exists), `invalid_request`, `wall_over_hard`, `turns_over_hard`, `rw_disabled`,
`rw_requires_containment`, `governor_unavailable`, `governor_level`, `concurrency_limit`, `run_count_limit`,
`run_count_unknown`, `cwd_is_home`, `disk_free_low` (< 10 GB free), `repo_untracked_limit` (> 5,000 untracked files),
`worktree_failed`, `job_failed`, `containment_failed`.

## Stop codes of a run

`completed`, `exit_nonzero`, `wall_limit`, `turn_limit`, `deadman`. The deadman kills a run with no progress for 90 s.
Progress is **transcript growth in bytes, or child CPU time adding up to 0.5 s since the last progress**. File mtimes are
never read: a loop that touches files proves nothing.

## Worktree and Job Object

One `git worktree add -b agentd/<run_id>` per run under `AgentdConfig.worktrees_root` (required; agentd never picks a
place on disk). A killed run's worktree is kept, never removed. The agent starts suspended, is assigned to its job, then
resumed. The job has `KILL_ON_JOB_CLOSE`, `ActiveProcessLimit` 64, a 6 GiB job commit limit and a 50 % CPU hard cap, and
no breakaway. After the agent exits the job is counted: a nonzero census is a **LEAK** (logged with the pids, counted in
`stats()["leaks"]` and in the summary row) and the job is killed. A census that is still draining a few milliseconds
after the exit code shows is given 0.5 s to settle before it counts as a leak.

All OS calls go through `JobOsApi`; unit tests use a recording fake. The real-Windows tests spawn a harmless child
(`ping.exe`, `python -c`) inside their own job and touch nothing else.

## Run store

Each started run appends a `start` row and each finished run a `summary` row (`run_id`, `backend`, `mode`, `wall_s`,
`turns`, `stop_code`, `leak`, `survivors`, `day`) to a JSONL `RunStore` (cap 16 MiB; a write past the cap is dropped and
counted in `stats()["store_dropped"]`). The store path is injected; no default path is written.

## Rollout gate (not met)

Flag off by default; 10 supervised runs reviewed by the owner; 7 days with zero LEAK; first 10 delegated PRs
owner-reviewed. The Claude Code and Copilot CLI smoke test under a restricted job has **not** been run (it needs the
real CLIs and an account) and its result must be recorded in the PR before rollout; a failure blocks rollout.

## Rollback

Unset `LOCI_AGENTD` (Hermes returns to its previous launch path; rehearse once first). `paused` halts delegation
without a restart. Discard worktrees by hand with `git worktree remove` once their diffs are reviewed.
