# Quality-first model catalog benchmark, 2026-09-17

Methodology:
- 15 deterministic hard/adversarial cases per role: cheap fan-out extraction/classification, guardian safety classification, escalation/adversarial verification, and synthesis.
- Candidate calls used local Ollama only: `http://100.73.200.19:11434`.
- Scoring is deterministic and quality-only. A case is correct only when the expected answer/verdict and required rubric terms are present and forbidden terms are absent.
- Timeout, unavailable, and transport errors are recorded separately from wrong answers.
- Latency is informational-only for audit and timeout interpretation. It is not used for winner selection or tie-breaking; equal quality remains a tie.
- Live run command:
  `python3 scripts/bench_model_catalog_quality.py --base-url http://100.73.200.19:11434 --timeout-s 45 --max-tokens 96 --output artifacts/model_catalog/quality_first_20260917.json`

Focused validation:
- `python3 -m pytest scripts/tests/test_bench_model_catalog_quality.py scripts/tests/test_model_catalog.py -q`

## Results

| role | model | quality | status counts | p50 latency ms (informational) | rank |
| --- | --- | ---: | --- | ---: | --- |
| cheap_fanout | `heretic-llama31-8b-instruct:latest` | 13/15 (86.7%) | wrong 2, timeout 0, unavailable 0, transport_error 0 | 366 | winner |
| cheap_fanout | `llama3.1-agent:latest` | 13/15 (86.7%) | wrong 2, timeout 0, unavailable 0, transport_error 0 | 342 | winner |
| cheap_fanout | `qwen2.5:3b` | 7/15 (46.7%) | wrong 8, timeout 0, unavailable 0, transport_error 0 | 313 | lower_quality |
| cheap_fanout | `qwen3.8:latest` | 11/15 (73.3%) | wrong 4, timeout 0, unavailable 0, transport_error 0 | 2552 | lower_quality |
| escalation | `heretic-llama31-8b-instruct:latest` | 4/15 (26.7%) | wrong 11, timeout 0, unavailable 0, transport_error 0 | 409 | lower_quality |
| escalation | `llama3.1-agent:latest` | 4/15 (26.7%) | wrong 11, timeout 0, unavailable 0, transport_error 0 | 406 | lower_quality |
| escalation | `qwen2.5:3b` | 4/15 (26.7%) | wrong 11, timeout 0, unavailable 0, transport_error 0 | 325 | lower_quality |
| escalation | `qwen3.8:latest` | 9/15 (60.0%) | wrong 6, timeout 0, unavailable 0, transport_error 0 | 2045 | winner |
| guardian | `granite3-guardian:2b` | 11/15 (73.3%) | wrong 4, timeout 0, unavailable 0, transport_error 0 | 77 | lower_quality |
| guardian | `heretic-llama31-8b-instruct:latest` | 13/15 (86.7%) | wrong 2, timeout 0, unavailable 0, transport_error 0 | 405 | lower_quality |
| guardian | `llama-guard3:8b` | 15/15 (100.0%) | wrong 0, timeout 0, unavailable 0, transport_error 0 | 132 | winner |
| guardian | `llama3.1-agent:latest` | 13/15 (86.7%) | wrong 2, timeout 0, unavailable 0, transport_error 0 | 416 | lower_quality |
| guardian | `qwen2.5:3b` | 8/15 (53.3%) | wrong 7, timeout 0, unavailable 0, transport_error 0 | 318 | lower_quality |
| guardian | `qwen3.8:latest` | 15/15 (100.0%) | wrong 0, timeout 0, unavailable 0, transport_error 0 | 2833 | winner |
| synthesis | `heretic-llama31-8b-instruct:latest` | 5/15 (33.3%) | wrong 10, timeout 0, unavailable 0, transport_error 0 | 574 | lower_quality |
| synthesis | `llama3.1-agent:latest` | 5/15 (33.3%) | wrong 10, timeout 0, unavailable 0, transport_error 0 | 553 | lower_quality |
| synthesis | `qwen2.5:3b` | 4/15 (26.7%) | wrong 11, timeout 0, unavailable 0, transport_error 0 | 381 | lower_quality |
| synthesis | `qwen3.8:latest` | 8/15 (53.3%) | wrong 7, timeout 0, unavailable 0, transport_error 0 | 4020 | winner |

## Recommendation deltas versus the original PR

| role | original PR #354 recommendation | quality-first recommendation |
| --- | --- | --- |
| cheap fan-out | `qwen2.5:3b` | tie: `heretic-llama31-8b-instruct:latest`, `llama3.1-agent:latest` |
| guardian/safety | `qwen2.5:3b` | tie: `llama-guard3:8b`, `qwen3.8:latest` |
| escalation/adversarial verification | `heretic-llama31-8b-instruct:latest` | `qwen3.8:latest` |
| synthesis | `qwen2.5:3b` | `qwen3.8:latest` |

Timeout caveat: this run had no timeouts, unavailable models, or transport errors. Dedicated guardian models no longer timed out under this isolated run; `granite3-guardian:2b` returned fast but scored lower quality, while `llama-guard3:8b` tied the best guardian score.
