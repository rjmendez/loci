# FlyBrain Harness Observability and Alerting

## Purpose

Define config-driven telemetry and alerting for FlyBrain harness storage pressure and failed sync jobs, using the repo's canonical storage root conventions instead of machine-specific paths or drive-letter assumptions.

This plan is aligned with:

- [FLYBRAIN_HARNESS_STORAGE_LAYOUT.md](./FLYBRAIN_HARNESS_STORAGE_LAYOUT.md)
- [FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md](./FLYBRAIN_WRITE_PATH_SAFETY_POLICY.md)
- [FLYBRAIN_FETCH_SYNC_PIPELINES_PLAN.md](./FLYBRAIN_FETCH_SYNC_PIPELINES_PLAN.md)
- [FLYBRAIN_STORAGE_BUDGET_GUARDRAILS.md](../artifacts/flybrain/fbh_storage_budget_guardrails_20260922.md) (artifact snapshot)

## Canonical root and config contract

The harness must treat the effective write root as a single configured value:

- `LOCI_FLYBRAIN_STORAGE_ROOT`
- optional overrides: `HARNESS_DATA_ROOT`, `HARNESS_STORAGE_ROOT`, or a stricter approved root in config

All alerting and telemetry must emit the configured root value as metadata, not a literal workstation path. Example references should always look like:

- `${LOCI_FLYBRAIN_STORAGE_ROOT}`
- `${LOCI_FLYBRAIN_STORAGE_ROOT}/graph`
- `${LOCI_FLYBRAIN_STORAGE_ROOT}/logs`

No example may hardcode `C:\`, `D:\`, `/mnt/...`, or any other machine-specific drive path.

Recommended config keys:

- `LOCI_FLYBRAIN_STORAGE_ROOT`
- `LOCI_FLYBRAIN_STORAGE_WATCH_GB`
- `LOCI_FLYBRAIN_STORAGE_SOFT_CAP_GB`
- `LOCI_FLYBRAIN_STORAGE_CRITICAL_GB`
- `LOCI_FLYBRAIN_STORAGE_HARD_CAP_GB`
- `LOCI_FLYBRAIN_SYNC_FAILURE_WINDOW_MINUTES`
- `LOCI_FLYBRAIN_SYNC_FAILURE_THRESHOLD`
- `LOCI_FLYBRAIN_SYNC_CONSECUTIVE_FAILURES`
- `LOCI_FLYBRAIN_JOB_HEALTH_LOG_DIR`

These values should be read from the same environment/config chain already used by the repo and should default to the current FlyBrain storage-budget guardrail policy values rather than a host-specific assumption.

## Storage-root telemetry

Collect filesystem metrics by resolving the configured root and evaluating each durable subtree under it:

- `graph`
- `snapshots`
- `cache`
- `backups`
- `logs`

Required metrics:

- `flybrain_storage_root_bytes_total`
- `flybrain_storage_root_bytes_used`
- `flybrain_storage_root_bytes_free`
- `flybrain_storage_root_utilization_pct`
- `flybrain_storage_subtree_bytes{path="graph|snapshots|cache|backups|logs"}`
- `flybrain_storage_pressure_level{level="watch|soft|critical|hard_stop"}`
- `flybrain_storage_last_checked_ts{root_env="LOCI_FLYBRAIN_STORAGE_ROOT"}`

Recommended derivations:

- Build usage percentage from the configured root and the configured budget thresholds.
- Emit per-subtree bytes so operators can tell whether pressure is concentrated in `cache`, `snapshots`, or active `graph` state.
- Include the storage root label in every event payload so alerts remain portable across machines and environments.

Telemetry should be produced by a single shared collector that resolves the root once, validates the root against the write-path guard rules, and then records usage against the canonical durables under that root. The collector should never accept raw drive paths from the local machine as defaults.

## Job health metrics

The harness should surface job-level health for all dataset sync and stage operations, including start, retry, completion, and failure states. These metrics must be emitted as structured signals in the job journal and in the runtime metrics stream.

Required metrics:

- `flybrain_job_runs_total{job_name, status}`
- `flybrain_job_duration_seconds{job_name, dataset, stage}`
- `flybrain_job_retries_total{job_name, dataset}`
- `flybrain_job_last_status{job_name, dataset, stage, status}`
- `flybrain_dataset_sync_attempts_total{dataset, version, stage}`
- `flybrain_dataset_sync_failures_total{dataset, version, stage, reason}`
- `flybrain_dataset_sync_success_total{dataset, version, stage}`
- `flybrain_dataset_sync_latency_seconds{dataset, version, stage}`
- `flybrain_dataset_sync_consecutive_failures{dataset, version}`
- `flybrain_dataset_sync_state{dataset, version, state="success|retry|failed|blocked"}`

Recommended labels:

- `dataset` (`hb`, `fw`, or future scoped identifiers)
- `version`
- `stage` (`preflight`, `plan`, `acquire`, `verify`, `promote`, `finalize`)
- `run_id`
- `root_env` (`LOCI_FLYBRAIN_STORAGE_ROOT`)
- `failure_class` (`network`, `integrity`, `policy`, `budget`, `transient`, `unknown`)

These metrics should be written to the standard harness logs under `${LOCI_FLYBRAIN_STORAGE_ROOT}/logs` and also to the job telemetry stream when one exists.

## Alert thresholds

Alert thresholds must remain config-driven and relative to the allowed root budget. The defaults below align with the repo's current storage budget guardrail policy and are intentionally conservative.

### Storage pressure alerts

Use these thresholds against the root usage percentage for `${LOCI_FLYBRAIN_STORAGE_ROOT}`:

- `watch`: configured threshold, default 150 GB used
  - action: pause new large downloads, archive stale cache artifacts, and review retention
- `soft_cap`: configured threshold, default 160 GB used
  - action: switch noncritical jobs to read-only mode and keep only active artifacts and current durable outputs
- `critical`: configured threshold, default 180 GB used
  - action: freeze nonessential writes, purge temp/staging/scratch, and require explicit operator override to continue
- `hard_stop`: configured threshold, default 200 GB used
  - action: block all nonessential FlyBrain writes; keep only the minimal active job set

The alert evaluator should compute the thresholds from configuration values rather than from a local machine path or static drive-letter assumption. It should also emit a single derived state, `flybrain_storage_pressure_level`, to simplify downstream dashboards and notifications.

### Failed sync alerts

Use both rolling-window and consecutive-failure logic so the harness does not silently hide repeated failure patterns.

Recommended defaults:

- `LOCI_FLYBRAIN_SYNC_FAILURE_WINDOW_MINUTES = 30`
- `LOCI_FLYBRAIN_SYNC_FAILURE_THRESHOLD = 3`
- `LOCI_FLYBRAIN_SYNC_CONSECUTIVE_FAILURES = 2`

Alert rules:

- `warning`: 2 or more failed syncs for a dataset in the rolling window, or 1 dataset with 2 consecutive failures
- `critical`: 3 or more failed syncs in the rolling window, or 2 consecutive failed attempts for the same dataset/version
- `page`: 5 or more failed syncs in the rolling window, or 3 consecutive failures for a single dataset/version, or any failure that blocks a critical stage while the storage root is at hard-stop or critical pressure

Also record the reason class so operators can distinguish:

- network/service interruption
- integrity checksum mismatch
- policy/root validation failure
- budget exhaustion
- retry exhaustion
- unknown or unclassified failure

## Operational handling

When an alert fires, the harness should:

1. log the alert in `${LOCI_FLYBRAIN_STORAGE_ROOT}/logs`
2. annotate the job run journal with the threshold + current usage state
3. stop new writes if the pressure level reaches `hard_stop`
4. stop or pause the affected dataset sync if the retry threshold is exceeded
5. keep the last valid manifests and prior successful snapshots until cleanup is explicitly approved

The alert path should follow the same config-driven semantics as the rest of the harness: compute the root from `LOCI_FLYBRAIN_STORAGE_ROOT`, emit generic labels, and avoid direct references to any machine-specific mount or drive letter.

## Expected output shape

A metrics payload should include enough metadata to support lookup and incident triage without requiring local machine knowledge:

```json
{
  "root": "${LOCI_FLYBRAIN_STORAGE_ROOT}",
  "pressure_level": "watch",
  "used_bytes": 160000000000,
  "total_bytes": 300000000000,
  "utilization_pct": 53,
  "dataset": "hb",
  "version": "neuprint_JRC_Hemibrain_1point2point1",
  "job": "fetch_sync",
  "stage": "acquire",
  "status": "failed",
  "failure_reason": "network",
  "retry_count": 3,
  "timestamp": "2026-09-22T00:00:00Z"
}
```

This stays generic across Windows, Linux, and mixed workstation/shared-drive layouts while remaining anchored to the repo's configured FlyBrain storage root.

## Decision summary

The FlyBrain harness should treat disk pressure and failed syncs as first-class operational signals, not ad hoc script logs. The alert model must stay aligned with the existing storage policy and layout, use the configured root variable as the source of truth, and remain portable across hosts without any hardcoded drive letters.
