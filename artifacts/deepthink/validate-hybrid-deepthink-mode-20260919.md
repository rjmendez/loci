# Hybrid deep-think routing validation (2026-09-19)

## Scope
- Repo: `C:\Users\rjmendez-admin\development\loci`
- Script under test: `scripts/local_deep_think.py`
- Todo: `validate-hybrid-deepthink-mode`

## Repro commands

```powershell
Set-Location C:\Users\rjmendez-admin\development\loci
python artifacts\deepthink\run_validate_hybrid_deepthink_mode.py
```

Output artifact:
- `artifacts\deepthink\validate-hybrid-deepthink-mode-run1.json`

## Active routing inputs observed
- `LOCI_CONFIG`: `C:\Users\rjmendez-admin\.loci\backends.toml` (exists)
- `OLLAMA_BASE_URL`: `http://localhost:11434`
- `LOCI_OLLAMA_GEN_URL`/`OLLAMA_GEN_URL`: unset
- `LOCI_VLLM_FALLBACK`: unset (disabled)
- `QDRANT_URL`: unset
- Resolved: `ollama_url=http://localhost:11434`, `ollama_gen_url=http://localhost:11434`, `vllm_url=http://localhost:18000`, `qdrant_url=""`

## Deep-think run configuration
- Topic: fleet queue saturation governance/root-cause reasoning prompt
- Flags: `--red-team --strict-grounding`
- Models:
  - ideate: `llama3.1-agent:latest`, `qwen3.8:latest`
  - verify/synthesize/self-reflect: `qwen3.8:latest`
  - red-team: `local-heretic-qwen38-27b:q4km`

## Operational results
- Generation calls observed: `5`
- Tier usage observed: `ollama=5`, `vllm=0`
- Success/degradation: `ok=4`, `not_ok=1`
  - degraded call: `qwen3.8:latest` ideation timed out at Ollama (`ReadTimeout`, 120s)
- Retrieval/grounding:
  - retrieval hits: `0` (`loci_memory`)
  - top-level gate: `mode=empty`, `kept=0`, `dropped=0`
  - verify context ids total: `0`
  - strict-grounding outcome: `synthesis.grounding_status=ungrounded`
  - synthesis emitted strict warning prefix (`⚠ UNGROUNDED (strict mode)...`)
- Red-team tier: enabled, skipped because no verified survivors (`survivor_count=0`)

## Context footprint indicators
- Configured evidence budget: `evidence_chars=4200`
- Actual evidence used:
  - synthesis supporting IDs: `[]`
  - verify `context_ids`: `[]` for all reports
- No truncation marker emitted in this run because retrieval returned no context chunks.

## Correctness/documentation mismatch found and patched
- Issue: native Windows direct execution of `scripts/local_deep_think.py` can fail when runtime imports `mcp/inv_store.py` (`fcntl` missing on Windows).
- Patch: added note to `docs/OPERATIONS.md` under environment/operations guidance:
  - use WSL/Linux for default persisted run, or inject deps (`run_chain(..., deps=...)`) to bypass MCP-store imports.
