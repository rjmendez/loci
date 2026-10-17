# Loci Copilot Instructions

## Policy Enforcement
- LOCI_POLICY_ENABLE=1 is set in the copilot environment
- LOCI_TRAJECTORY_DIR=~/.loci/trajectories for DRPG step logging
- Use PolicyTier.LIGHT for exploratory/read-only tasks, MEDIUM for code changes, HEAVY for architecture decisions

## Testing
- All tests: `cd mcp && python -m pytest tests/ -v`
- Policy tests only: `python -m pytest tests/test_policy_backend.py tests/test_drpg_evolution.py -v`

## Code Conventions
- Policy objects are dataclasses with `to_dict()`/`from_dict()` roundtrip
- All file I/O uses `tmp_path` in tests
- Hooks are async functions returning dicts

## Environment Variables
The following environment variables are set in the Copilot coding agent environment:

| Variable | Value | Purpose |
|---|---|---|
| `LOCI_POLICY_ENABLE` | `1` | Enables policy enforcement |
| `LOCI_TRAJECTORY_DIR` | `~/.loci/trajectories` | DRPG step logging directory |
| `LOCI_CACHE_CONFIDENCE` | `0.85` | Minimum confidence threshold for cache hits |
| `LOCI_CACHE_TTL_HOURS` | `24` | Cache entry time-to-live in hours |
