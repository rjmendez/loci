# Loci-native flagship workflow — measured run

**Run** `wf_a5bff9e3-d4d` · 6 agents · 0 errors · 285 s · 217,964 subagent tokens
**Task:** dogfood review of the Loci-native offload-tier modules across all tiers.

The `loci-flagship` workflow composes every tier of the offload hierarchy in one run and
self-instruments it. This is the canonical proof that the cost/quality-tiered design works —
and it earned its keep by finding (and we then fixed) a real bug in its own grounding code.

## Telemetry (auto-emitted by the workflow)

| Metric | Value |
|---|---|
| Tasks | 4 (`graph` 1 · `impl` 2 · `reason` 1) |
| **Graph tier resolved at 0 tokens** | **1** (no agent spawned — deterministic Kuzu fact) |
| Graph-fallback agents | 0 |
| Claude agents | 3 |
| **Verify pass-rate** | **100 %** (2 / 2 adversarial checks confirmed) |
| Grounding sources injected | 3 (1,187 chars, once → every agent) |
| Synthesis dedup | 4 kept / 0 dropped |
| `embed_gen_tools_ok` | **false** — Loci MCP was disconnected, so the synthesis agent fell back instead of calling the real `semantic_dedup`/`compress_text` tools (correctly flagged) |

## Cost accounting
- The **graph tier is genuinely 0 Claude tokens** — the workflow resolved the call-site
  inventory from a pre-computed fact with `Promise.resolve`, no `agent()` call. That's one
  task (~1/4 of the fan-out) done for free; at the run's ~36 K tokens/agent average, ≈ 36 K
  tokens saved on this small run, scaling linearly with the share of deterministic work.
- The **compute tiers are real**: a grep across every producer (`embed_ops`, `llm_local`,
  `batched_gen`, `query_expand`, `text_ops`, `grounding`, `graph_facts`) found **zero
  Anthropic/Claude calls** — only Claude *intent* in docstrings. Savings are real but hold
  only on the correctly-configured path (graph-fallback and MCP-down synthesis revert to Claude).

## Findings from the review
1. **🔴 Fail-open violation in `grounding.py` (FIXED).** `ground()` sections 1 (cases, `S.investigation_load`)
   and 3 (entities, `S.investigation_entity_lookup`) called server tools **outside** try/except
   while sections 4/5/6 were guarded — a raising source would abort grounding, violating the
   "fail-open everywhere" contract. Plus a `str(f.get(...))` that assumed dict findings.
   Adversarially **verify-confirmed** against live code. → Wrapped both loops + added an
   `isinstance` guard; 2 regression tests added (10/10 pass).
2. **🟡 Cost claims hold with nuance.** 0-token graph + local compute tiers are real, but three
   paths still spend Claude tokens (graph-fallback, an agent invoked only to call a local tool,
   MCP-down synthesis). Documented, not defects.
3. **🟢 `embed_ops.dedup` correct**, `llm_local.generate` fully fail-open — claims hold.

## The meta-point
A grounded, tiered, self-instrumented fan-out reviewed its own implementation, an adversarial
verifier confirmed the finding against live code, and the defect is now fixed and tested — for
the price of one deterministic tier being free and the compute tiers offloaded off Claude.
