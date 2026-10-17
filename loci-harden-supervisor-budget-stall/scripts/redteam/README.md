# Loci red-team harness

Sandboxed-only adversarial testing for Loci MCP tools.

## Scope

- Uses a local **heretic/abliterated Ollama model** as the attacker model.
- Uses **PyRIT** for attacker-target orchestration and prompt-injection scoring.
- Executes only against **synthetic sandbox state** created under `scripts/redteam/`.
- **Never** uses production investigations, real Qdrant state, or destructive actions.
- Focuses on **semantic/adversarial tool-call payloads** (OWASP LLM01 / LLM06), not random mutation.

## Current target tools

- `wiring_obligation_scan`
- `investigation_pre_answer_check`
- `conflict_resolve`

These were chosen because they either consume untrusted text directly or sit on sensitive validation/write paths.

## Install

PyRIT is intentionally not a repo-wide required dependency; install it in the MCP venv before running:

```bash
cd mcp
.venv/bin/python -m pip install pyrit
```

The harness also needs access to a local Ollama generation endpoint already configured for Loci (`mcp/backends.py`).

## Run

```bash
cd /home/rjmendez/development/loci
mcp/.venv/bin/python scripts/redteam/loci_adversarial_harness.py \
  --tools wiring_obligation_scan investigation_pre_answer_check conflict_resolve \
  --cases-per-tool 2 \
  --output scripts/redteam/reports/latest-sandbox-report.json
```

The report is JSON and separates:

- attacker metadata/model inventory
- generated red-team cases
- per-case findings
- defender triage guidance

## Safety boundary

The harness patches Loci into a synthetic local sandbox for every case:

- `server.MEMORY_DIR` redirected to sandbox storage
- `QDRANT_URL` and `QDRANT_API_KEY` cleared
- Qdrant upserts, Mnemosyne writes, graph mirroring, and event-log appends disabled
- only synthetic investigation content is seeded

If a finding looks real, treat it as a **human-review item** first. This harness is for detection and triage, not silent auto-remediation.
