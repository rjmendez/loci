# FlyBrain Phase-1 Operator Training Checklist

This checklist defines the concrete maintenance burden and operator-readiness gates for the phase-1 local FlyBrain harness (`hb` graph + `fw` metadata), aligned with the storage layout, backup/recovery, and rollout milestones docs.

## Scope and role expectations

- **In scope:** local harness operations under `LOCI_FLYBRAIN_STORAGE_ROOT` for `graph`, `snapshots`, `cache`, `backups`, and `logs`.
- **Phase-1 data scope:** `hb=neuprint_JRC_Hemibrain_1point2point1` and `fw=flywire783` only.
- **Operator baseline:** can run `dry-run`/`plan`/`execute`, interpret path-policy denials, verify manifest/checksum integrity, and stop promotion when gates fail.

## Maintenance burden (phase-1)

| Task | Cadence | Primary owner | Typical effort | Training requirement | Evidence artifact |
|---|---|---|---:|---|---|
| Preflight + policy gate review (`dry-run`) | Per execution run | On-call operator | 10-15 min | Path-policy and scope-pins training | `logs\pipelines\<run_id>\preflight.json` |
| Plan review and approval package (`plan`) | Per execution run | Operator + reviewer | 15-25 min | Idempotency and write-intent review | `logs\pipelines\<run_id>\plan.json` |
| Execute run monitoring (`execute`) | Per execution run | Operator | 30-90 min | Resume/retry and failure-boundary handling | `logs\pipelines\<run_id>\summary.json` + stage logs |
| Manifest/checksum integrity verification | Per dataset refresh | Operator | 10-20 min | `fbh-manifest/v1` and SHA-256 validation | `snapshots\<dataset>\<version>\manifest\manifest.json` + `.sha256` |
| Snapshot rotation and storage cleanup | Weekly (or when `>=150 GB` watch threshold) | Operator | 20-45 min | Storage budget and retention policy training | Rotation record in `logs\operations\snapshot-rotation\` |
| Backup checksum verification | Weekly | Operator | 15-30 min | Backup catalog + integrity checks | backup verification log |
| Restore drill | Quarterly | Operator + approver | 60-120 min | Restore procedure, replay smoke test | `logs\restore-drills\<date>\drill-report.json` |
| Budget posture review (`watch`/`soft`/`critical`) | Weekly and pre-execute | Operator | 10-15 min | Freeze trigger recognition | utilization report in `logs\operations\budget\` |

## Approval gates (must pass before go-forward)

### Gate A - Preflight approval (Milestone 0)

Required before `plan` or `execute`:

- [ ] All resolved paths are under `LOCI_FLYBRAIN_STORAGE_ROOT` and pass path-policy checks.
- [ ] Dataset scope is exactly `hb,fw`; version pins match approved phase-1 values.
- [ ] Storage state is below hard-stop and no unapproved override is active.
- [ ] Preflight evidence archived (`preflight.json`).

**Approver:** Operator (self-approval allowed) with audit trail in run logs.

### Gate B - Plan approval (Milestone 1)

Required before `execute`:

- [ ] Deterministic stage graph present for each pinned dataset.
- [ ] Idempotency keys recorded and replay-safe.
- [ ] Predicted writes remain within allowlisted subtrees (`snapshots`, `cache`, `graph`, `logs`).
- [ ] Storage budget impact reviewed against soft/critical thresholds.

**Approver:** Secondary reviewer (recommended) or designated operator lead.

### Gate C - Promotion approval (Milestone 3)

Required before hb candidate promotion:

- [ ] Candidate graph smoke checks pass.
- [ ] Manifest + checksum integrity pass for new artifacts.
- [ ] Promotion action uses atomic pointer metadata update (no in-place mutation of active store).
- [ ] Query-pack reproducibility check recorded.

**Approver:** Operator lead (cannot be bypassed if integrity warnings exist).

### Gate D - Recovery readiness approval

Required after restore drill or corruption incident before new publish/promote actions:

- [ ] Latest completed backup bundle validated by checksum.
- [ ] Restore drill/recovery replay smoke test passed within RTO target.
- [ ] Any divergence/corruption corrective actions are documented and closed.

**Approver:** Incident owner + operator lead.

## Snapshot rotation policy

Apply rotation to `snapshots\` content to cap durable growth while preserving reproducibility:

1. **Keep windows (default):**
   - Keep the latest **3 successful snapshot versions** per dataset (`hb`, `fw`).
   - Keep at least **1 known-good snapshot** older than the current active promotion point.
2. **Archive path:** move/prune only inside `backups\snapshot-archive\` or approved backup root.
3. **Prune order under pressure:**
   - First: `cache\downloads` and other rebuildable cache artifacts.
   - Then: non-active snapshots older than retention window.
   - Never prune active promoted graph or required run logs.
4. **Rotation triggers:**
   - Scheduled weekly maintenance, or
   - Immediate rotation review when usage reaches watch threshold (`>=150 GB`).
5. **Required records per rotation event:** timestamp, operator, deleted/archived targets, reclaimed size, validation that active snapshot/promotion pointers were untouched.

## Data freeze policy (phase-1)

Data freeze means blocking publish/promotion writes until risk is cleared.

### Freeze triggers (automatic stop)

- Checksum mismatch on required source/manifest artifacts.
- Manifest schema/provenance validation failure.
- Promotion pointer integrity risk or active-link inconsistency.
- Storage hard-stop breach (`>=200 GB`) or critical threshold without approved remediation.
- Failed restore drill where replay cannot be validated.

### Freeze levels

- **Level 1 - Promotion freeze:** block candidate promotion; allow read-only checks and diagnostics.
- **Level 2 - Publish freeze:** block new snapshot/manifests for affected dataset; allow incident triage.
- **Level 3 - Global write freeze:** block all nonessential harness writes until storage/integrity incident is resolved.

### Freeze exit criteria

- Root cause documented.
- Corrective action completed and verified.
- Relevant gate(s) (A-D) re-passed with fresh evidence.
- Explicit sign-off by operator lead.

## Operator training and sign-off checklist

An operator is considered phase-1 ready only after all items are complete.

### Training completion

- [ ] Reviewed: `FLYBRAIN_HARNESS_STORAGE_LAYOUT.md`.
- [ ] Reviewed: `FLYBRAIN_HARNESS_BACKUP_RECOVERY_STRATEGY.md`.
- [ ] Reviewed: `FLYBRAIN_HARNESS_ROLLOUT_MILESTONES.md`.
- [ ] Reviewed: `FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md`.
- [ ] Completed one supervised `dry-run` + `plan` cycle with valid evidence artifacts.
- [ ] Completed one supervised `execute` run with manifest/checksum validation.
- [ ] Completed one restore drill walkthrough (or observed latest recorded drill with Q&A).

### Practical competency checks

- [ ] Can explain stop/go boundaries between Milestones 0 -> 1 -> 2 -> 3.
- [ ] Can identify which failures are retryable vs hard-stop.
- [ ] Can run snapshot rotation without touching active promotion paths.
- [ ] Can invoke freeze policy and document incident evidence.
- [ ] Can produce an operator handoff package (preflight, plan, summary, integrity outputs).

### Sign-off record

- **Operator name:**
- **Date:**
- **Mentor/reviewer:**
- **Run IDs used for qualification:**
- **Restore drill evidence path:**
- **Sign-off decision:** Approved / Not approved
- **Notes and follow-ups:**

## Linked references

- [FLYBRAIN_HARNESS_STORAGE_LAYOUT.md](./FLYBRAIN_HARNESS_STORAGE_LAYOUT.md)
- [FLYBRAIN_HARNESS_BACKUP_RECOVERY_STRATEGY.md](./FLYBRAIN_HARNESS_BACKUP_RECOVERY_STRATEGY.md)
- [FLYBRAIN_HARNESS_ROLLOUT_MILESTONES.md](./FLYBRAIN_HARNESS_ROLLOUT_MILESTONES.md)
- [FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md](./FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md)
- [FLYBRAIN_LOCAL_QUERY_OPERATIONAL_GUIDE.md](./FLYBRAIN_LOCAL_QUERY_OPERATIONAL_GUIDE.md)
