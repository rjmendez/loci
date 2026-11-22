# ops/dotloci

Deployed copies of the Loci operations scripts that the Windows scheduled tasks run from `%USERPROFILE%\.loci\`
(Loci-Watchdog, Loci-OpsLogs-*, Loci-Opnsense*, Loci-QdrantBackup, Loci-OllamaInventory) plus `start_loci.ps1`.
They were not in git; this directory is the tracked source, the `.loci` directory is where they run.

Not tracked on purpose: the runtime state and health files next to them (`*_state.json`, `*_health.json`, `ops_selftest.json`),
and four one-off remediation scripts from 2026-10-04/05 (`remediate_*`, `remove_vllm_*`, `apply_hooks_fix_*`, `a2a_xhost.py`)
that carry fixed tailnet addresses.
