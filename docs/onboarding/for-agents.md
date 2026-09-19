# Using Loci: A Guide for Agents

_Last reviewed: 2026-09-17 — rolling document. When tool signatures or the recommended patterns change, update this page so agents onboard against current behavior._

This page is for an **agent** that will use Loci as its memory and reasoning substrate. It covers the order of operations that works, the reasoning tools, how to store and verify memory, and the concrete patterns to reach for. See [tools.md](tools.md) for the full surface.

## The one rule: ground first

Before reasoning or answering, **assemble grounding**. Grounding pulls prior, provenance-tagged evidence into a char-budgeted block so you build on what is already known instead of re-deriving (or contradicting) it.

```
ground(title="<the question or task>",
       focus="<optional narrower angle>",
       entities=[...],       # IPs, hostnames, hashes, CVEs, etc.
       code_refs=[...],      # symbols/paths in play
       budget_chars=<fit your window>)
```

`ground` runs hybrid dense/sparse search with cross-encoder reranking and returns a block tagged with provenance. It is also the right tool at a **multi-agent handoff**: the assembling agent grounds, and the next agent receives the block as its starting context.

If you only need raw recall rather than an assembled block, use `rag_context_search` (across all investigations), `investigation_search` (within one), or `memory_surface` / `memory_hints` for proactive/recent context.

## The standard investigation pattern

Almost all durable work follows this shape:

1. **`investigation_start(id, title)`** — begin or resume a session. Returns a manifest with finding counts.
2. **`investigation_store(id, type, text, source)`** — record findings as you go. `type` is one of `observed` | `inferred` | `assumed` | `gap` | `procedure`. Writes to JSONL + Qdrant, returns a `finding_id`.
3. **`investigation_search(query, id)`** — retrieve prior work within the investigation.
4. **`investigation_pre_answer_check(id, claims)`** — validate your claims against stored evidence *before* you answer.
5. **`investigation_reason(id, question)`** + **`swarm_reason(topic)`** — synthesize when the question needs multi-perspective reasoning.
6. **`ground(title, entities, code_refs, budget)`** — assemble context to hand to the next agent.

After a restart, recover context with `investigation_load(id, last_n_findings=...)` rather than starting cold.

### Choosing a finding type

- `observed` — something you directly saw (command output, a file's contents, a measurement).
- `inferred` — a conclusion you drew from observed findings (it should have a `derived_from` lineage).
- `assumed` — a working assumption not yet checked.
- `gap` — a known unknown / open question worth recording.
- `procedure` — a repeatable how-to (retrievable later via `procedure_search`).

Be honest about the type. Provenance is what makes verification and conflict detection work: `investigation_finding_provenance` can trace an `inferred` finding back to the `observed` evidence it rests on.

## Reasoning: swarm_reason and llm_local

Offload reasoning to local models before spending cloud tokens.

- **`swarm_reason(topic, perspectives?, fanout?, ground_threshold?, synthesize_think?)`** — a bounded 4-stage swarm: fan-out across perspectives → triage → selective escalation → synthesis, with safety gates. Use it for multi-perspective analysis ("is this design sound?", "what could break here?").
- **`llm_local(prompt, model?, fmt?, max_tokens?)`** — a single local generation. Use `fmt='json'` to constrain output to JSON. Returns `{text, ok, model}`. Under the hood, `compress_text`, `classify_text`, `query_expand`, and `verify_finding` all route through local generation too.

All local tiers **fail open**: if a backend is down, you get degraded output, not a crash. Watch for `degraded=True` in results — an unpulled or misspelled model tag fails *silently* except for the `why` field.

## Verification: do not assert what you cannot support

Loci gives you trust gates. Use them.

- **`verify_finding(claim, context?)`** — adversarially verify a single claim with a local model. Returns `{ok, confidence, critique}`.
- **`investigation_pre_answer_check(id, claims)`** — validate a batch of claims against stored evidence before answering. It gates on lexical + semantic overlap (overlap ≥ 0.15, margin ≥ 0.05 over the pool median) and writes verdicts to the `loci_verdicts` collection.
- **`investigation_verify_all(id)`** — batch-verify open findings.
- **`memory_self_check(id)`** — provenance + contradiction check across the investigation.
- **`conflict_list` / `conflict_resolve`** — when `investigation_store` auto-detects a contradiction, resolve it explicitly (`a_wins` | `b_wins` | `both_valid` | `false_positive`).
- **`memory_confidence(query)`** — metamemory: ask how reliably memory even knows a topic (0.0–1.0) before you lean on it.

> Note: the adversarial verifier over-refutes (~22% false-refutation on small local models). Use it as a skeptic that flags things to re-check, not as a final arbiter, and never schedule the `verify` groom pass.

## Getting things wrong, on purpose

Memory is append-only and correctable, not immutable:

- Wrong finding → **`memory_retract(id, target, reason)`** (soft tombstone; `dry_run=True` by default, so pass `dry_run=False` to apply). It retracts derived lineage too.
- Retracted too much → **`memory_restore(id, finding_id=...)`**.
- Update the plan, not the log → **`investigation_note(id, field, value)`** for `hypothesis`, `next_step`, `questions`, `checked_sources`.

## Code-aware work

When the task touches code:

1. **`code_graph_ingest(path)`** — index the symbols (tree-sitter).
2. **`symbol_impact(symbol)`** / **`impact_report(symbol)`** — blast radius before you change something.
3. **`finding_code_context(finding_id)`** — the code a finding references, with callers/callees.
4. **`investigation_code_briefing(id)`** — the code story of the whole investigation.
5. **`related_investigations_via_code(id)`** — who else touched these symbols.

## Entities and prior cases

When an IP, hostname, hash, CVE, or domain shows up:

- `investigation_entity_lookup(entity)` — every finding that mentions it.
- `investigation_related_cases(entities)` — prior investigations dealing with the same entities.
- `entity_timeline(id, entity_id)` — its chronology within an investigation.

## Role-specialized shortcuts

- **Retrieval-only agents (read access):** `rag_context_search`, `memory_route`, `memory_hints`, `investigation_entity_lookup`, `investigation_related_cases`.
- **Verification agents (trust gates):** `verify_finding`, `investigation_verify_all`, `memory_self_check`, `conflict_resolve`.
- **Code-aware agents:** `code_graph_ingest`, `symbol_impact`, `impact_report`, `investigation_code_briefing`, `related_investigations_via_code`.
- **Reflection / self-improvement:** `reflection_loop_seed`, `reflection_loop_tick`, `memory_consolidate`, `memory_promote` / `memory_demote`.

## Working with other agents

Share investigations explicitly with `investigation_share(id, agent_ids)` (revoke with `investigation_unshare`). Across the mesh, `memory_route` searches all investigations filtered by ACL, and the A2A server can `context_broadcast` a grounding block to peers. For a portable handoff outside the mesh, `investigation_export` / `investigation_import` moves a full bundle.

## Habits that keep memory healthy

- Ground before you reason; verify before you assert.
- Store findings with an honest type and a real `source`.
- Prefer local models (`swarm_reason`, `llm_local`) before cloud reasoning.
- Record `gap` findings — known unknowns are memory too.
- Retract instead of overwriting; the log is append-only by design.
