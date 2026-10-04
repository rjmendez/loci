# Global operating rules (all Copilot CLI sessions, all subagents)

## Loci RAG grounding is mandatory
Before substantive work - answering a non-trivial question, making a claim,
editing/creating files, running builds/deploys, or starting a new investigation
phase - ground in Loci first: call `loci-rag_context_search` (or `loci-ground`)
and/or `loci-investigation_search`, and rely on the returned cited evidence, not
memory. This applies to subagents too. A PreToolUse hook enforces it mechanically;
treat a hook reminder/block as a hard requirement.

## Honesty about simulation vs validated results
Respect the honesty machinery; never bypass it. Artifacts carry `exploratory`,
`backend: mock`, "not a FlyBrain result", or "relative only (F9)" labels - never
present exploratory/mock/sim output as validated. The Python `evaluate_teams("exact")`
path is the parity ORACLE; Rust goldens and `scripts/check_test_honesty.py` are the
gates. If something is fake/mock/unvalidated, say so plainly.

## Verify before claiming done
Produce evidence (tests/build/live/visual) before claiming success; prefer
sanctioned tooling over ad-hoc bypasses.
