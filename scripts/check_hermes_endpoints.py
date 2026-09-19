#!/usr/bin/env python3
"""Fail fast when Hermes local endpoints drift onto bridge/private IPs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "mcp") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "mcp"))

from legacy_env import validate_hermes_endpoint_policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    issues = validate_hermes_endpoint_policy()
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}")
        print("Use 127.0.0.1/localhost for Hermes local backends; bridge/private IPs are rejected.")
        return 1
    if not args.quiet:
        print("Hermes endpoint policy OK: all local backends are loopback-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
