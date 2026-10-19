#!/usr/bin/env python3
"""GPU load sidecar -- polls nvidia-smi and writes the shared Loci load signal.

Designed to run as a long-lived systemd user service:
  ~/.config/systemd/user/loci-gpu-load.service

Writes /run/user/<uid>/loci_gpu_load.json every POLL_INTERVAL_S seconds.
The file lives in tmpfs and vanishes on reboot -- no stale state survives across sessions.

Falls back gracefully: if nvidia-smi is unavailable or returns no rows the signal file
is simply not written (or not updated), and callers fall back to normal routing.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_LOG = logging.getLogger("loci.gpu_load_sidecar")
_FAILURE_LOG_EVERY = 60
_POLL_INTERVAL_S = float(os.environ.get("LOCI_GPU_POLL_INTERVAL_S", "5"))


def parse_nvidia_smi_csv(text: str) -> list:
    """GpuEntry list from ``nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total
    --format=csv,noheader,nounits`` output (LF or CRLF). Rows with too few columns or non-numeric
    fields (nvidia-smi prints ``[N/A]`` for what a driver mode does not expose) are skipped at WARNING."""
    from gpu_load import GpuEntry

    entries = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            if line.strip():
                _LOG.warning("skipping short nvidia-smi row %r", line)
            continue
        try:
            entries.append(
                GpuEntry(
                    index=int(parts[0]),
                    name=parts[1],
                    util_pct=float(parts[2]),
                    vram_used_mb=float(parts[3]),
                    vram_total_mb=float(parts[4]),
                )
            )
        except ValueError as exc:
            _LOG.warning("skipping unparseable nvidia-smi row %r: %s", line, exc)
    return entries


def _poll_nvidia_smi() -> list:
    """Return a list of GpuEntry objects from nvidia-smi, or [] on failure."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        _LOG.warning("nvidia-smi exited %d: %s", result.returncode, result.stderr.strip())
        return []
    return parse_nvidia_smi_csv(result.stdout)


def _note_failure(failures: int, why: str) -> int:
    """Count one failed poll; WARNING on the first and then every _FAILURE_LOG_EVERY-th in a row.
    While polls fail the signal file goes stale and readers report it, so this is the writer-side half."""
    failures += 1
    if failures == 1 or failures % _FAILURE_LOG_EVERY == 0:
        _LOG.warning("GPU load signal not updated (%d consecutive failed polls): %s", failures, why)
    return failures


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _LOG.info("loci-gpu-load sidecar starting (poll=%.0fs)", _POLL_INTERVAL_S)

    running = True

    def _stop(signum, _frame):
        nonlocal running
        _LOG.info("received signal %s, stopping", signum)
        running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    from gpu_load import write_gpu_load

    failures = 0
    while running:
        try:
            gpus = _poll_nvidia_smi()
            if gpus:
                write_gpu_load(gpus)
                _LOG.debug("wrote load signal: %d GPU(s)", len(gpus))
                failures = 0
            else:
                failures = _note_failure(failures, "no GPU rows from nvidia-smi")
        except Exception as exc:
            failures = _note_failure(failures, f"poll error: {exc!r}")

        deadline = time.monotonic() + _POLL_INTERVAL_S
        while running and time.monotonic() < deadline:
            time.sleep(0.2)

    _LOG.info("loci-gpu-load sidecar stopped")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
