# defense-stack

Tracked record of the Hermes -> LiteLLM -> Loci defense layers. See MANIFEST.md for what maps where.

Verify: `python defense-check/check_defense.py` (exit 0 = every live layer loaded and wired).
Test the Hermes self-echo filter: `python hermes/plugins/loci-rag/test_selfecho.py`.
Restore: `pwsh install.ps1` (add `-Tasks` to re-register the scheduled tasks). Config merge is manual.
