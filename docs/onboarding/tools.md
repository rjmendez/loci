# Loci MCP Tool Surface

_Last reviewed: 2026-09-17 — rolling document. The tool count and signatures drift as the server evolves; when `mcp/server.py` gains or changes a tool, update the relevant group here._

The MCP server exposes roughly 75 tools (the live catalogue lists 76; `mcp/` internals count up to 77 — treat the exact number as approximate and the groups below as the map). Tools are organized into functional groups. Tools marked **CRITICAL** are the ones an agent reaches for most; see [for-agents.md](for-agents.md) for how they compose.

## Grounding & RAG — semantic search + evidence assembly

- **`ground(title, focus?, entities?, code_refs?, budget_chars?, ...)`** — **CRITICAL.** Assemble a char-budgeted, provenance-tagged grounding block. Hybrid dense/sparse search with cross-encoder reranking. Essential for multi-agent handoffs.
- `rag_context_search(query, investigation_id?, min_similarity?, ...)` — cross-collection search over findings + verdicts across all investigations.
- `investigation_search(query, investigation_id?, ...)` — hybrid semantic/keyword search within one investigation; filter by resolution/tier.
- `memory_route(query, agent_id?, top_k?, deduplicate?)` — agent-mesh search across all investigations; filters by ACL.
- `memory_surface(context, investigation_id?, top_k?)` — proactively surface prior findings for the current working context.
- `memory_hints(investigation_id, limit?, since_ts?, mode?)` — recent findings as lightweight, recency-scored hints; poll after `investigation_store`.

## Memory ops — persistence, validation, lifecycle

- **`investigation_store(investigation_id, finding_type, text, source, confidence?, ...)`** — **CRITICAL.** Record a finding (`observed` | `inferred` | `assumed` | `gap` | `procedure`). Auto-detects conflicts, writes to Qdrant + JSONL, returns a `finding_id`.
- `memory_retract(investigation_id, target, reason?, dry_run?, scope_semantic?)` — soft-delete findings and derived lineage. `dry_run=True` by default.
- `memory_restore(investigation_id, finding_id?, retraction_id?, reason?)` — undo a retraction; restore to the original tier.
- `investigation_note(investigation_id, field, value)` — update the manifest: `context` | `hypothesis` | `next_step` | `questions` | `checked_sources`.
- `investigation_reflect(investigation_id)` — synthesize investigation state + an entity-frequency histogram.
- `memory_consolidate(dry_run?)` — run Mnemosyne sleep consolidation over all sessions.

## Investigation session management

- `investigation_start(investigation_id, title, context?, author?, acl?)` — create or resume an investigation; returns a manifest with finding counts.
- `investigation_load(investigation_id, last_n_findings?, include_retracted?)` — load manifest + recent findings (context recovery after a restart).
- `investigation_list(limit?, offset?, summary?)` — paginated listing (default `limit=30`, `offset=0`); `summary=True` for a compact view.
- `investigation_as_of(investigation_id, as_of_timestamp)` — view findings as believed at a point in time.
- `investigation_share(investigation_id, agent_ids)` / `investigation_unshare(...)` — grant / revoke ACL access.
- `investigation_export(investigation_id, include_embeddings?)` — export a portable JSON bundle.
- `investigation_import(bundle_json, new_title?)` — import a bundle from `investigation_export`.
- (plus `investigation_finding_provenance`, `investigation_reflect`, `investigation_note` above)

## Claim validation & verification

- **`investigation_pre_answer_check(investigation_id, claims, ...)`** — **CRITICAL.** Validate claims against stored evidence before answering; detect hallucinations and contradictions.
- **`verify_finding(claim, context?, investigation_id?)`** — **CRITICAL.** Adversarially verify a claim with a local model. Returns `{ok, confidence, critique}`.
- `investigation_evidence_precheck(investigation_id, proposed_query, min_similarity?)` — lightweight duplicate/evidence check.
- `investigation_verify_all(investigation_id, limit?)` — batch adversarial-verify open findings.
- `memory_self_check(investigation_id?)` — provenance + contradiction self-check.
- `conflict_list(investigation_id)` — list detected contradictory findings.
- `conflict_resolve(investigation_id, conflict_id, verdict)` — record `a_wins` | `b_wins` | `both_valid` | `false_positive`.
- `memory_confidence(query, top_k?)` — metamemory: how reliably memory knows a topic (0.0–1.0).
- (plus `adversarial_review`, `finding_resolve`)

> Caveat: the `verify` groom pass has a ~22% false-refutation rate — do not schedule it. Interactive `verify_finding` is fine.

## Code graph & context — symbol knowledge + blast radius

- **`code_graph_ingest(path, max_files?, replace?)`** — **CRITICAL.** Parse with tree-sitter and ingest the symbol graph; enables code↔memory navigation.
- **`symbol_impact(symbol, hops?)`** — **CRITICAL.** Blast radius of a symbol across code and memory.
- `code_graph_query(cypher, params?)` — read-only Cypher over the code + findings graph.
- `code_memory_relink()` — idempotently MERGE `Finding → CodeSymbol` edges.
- `code_memory_map(anchor, anchor_type?, hops?)` — code↔memory neighbourhood around an anchor.
- `impact_report(symbol, hops?)` — change blast radius: transitive callers + co-referenced symbols/findings.
- `finding_code_context(finding_id)` — the code a finding references, with callers/callees.
- `investigation_code_briefing(investigation_id, top?)` — the code story of an investigation.
- `subsystem_report(anchor, limit?)` — full picture of code under a path or package prefix.
- `related_investigations_via_code(investigation_id, limit?)` — other investigations touching the same symbols.
- `dead_code_candidates(lang?, limit?)` — functions with no caller and no finding reference.

## Local model reasoning — LLM offload + swarm

- **`swarm_reason(topic, perspectives?, fanout?, ground_threshold?, synthesize_think?, ...)`** — **CRITICAL.** Bounded local-model swarm reasoning: fan-out → escalate → synthesize with safety gates. The core multi-perspective reasoning tool.
- **`llm_local(prompt, model?, fmt?, max_tokens?, temperature?, keep_alive?)`** — **CRITICAL.** Generate with a local Ollama model. `fmt='json'` constrains output. Returns `{text, ok, model}`.
- `generate_batch(prompts, model?, max_tokens?, fmt?)` — generate for many prompts at once (vLLM/TGI, or Ollama fallback).
- `query_expand(query, n_queries?, n_keywords?)` — HyDE-lite query expansion for search.
- `classify_text(text, labels)` — pick the best label for text.
- `compress_text(text, max_chars?)` — semantically condense text to a char budget.
- `semantic_dedup(items, threshold?, text_key?)` — cluster near-duplicate items by embedding similarity.
- `semantic_relevance(texts, topic)` — cosine relevance of each text to a topic.
- (plus `ground`, `adversarial_review`)

## Entity & relationship tracking

- `investigation_entity_lookup(entity, entity_type?, investigation_id?, limit?)` — find all findings mentioning an IP | email | hostname | hash | CVE | domain.
- `investigation_related_cases(entities, entity_type?, limit_per_entity?)` — prior investigations dealing with the same entities.
- `entity_list(investigation_id, entity_type?)` — list all entities extracted from findings.
- `entity_timeline(investigation_id, entity_id)` — chronological findings mentioning an entity.
- `investigation_finding_provenance(finding_id, investigation_id)` — trace lineage (the `derived_from` chain back to observed evidence).
- `causal_edges_list(investigation_id)` — causal edges inferred for an investigation.
- `causal_infer(investigation_id, limit?)` — infer causal edges from finding-derivation chains.

## Contracts, wiring, tiering, reflection, audit

- **Contracts & wiring:** `contract_declare`, `contract_query`, `contract_check`, `wiring_obligation_scan`, `wiring_obligation_declare`, `wiring_obligation_list`, `wiring_obligation_resolve`.
- **Memory tiering & correlation:** `memory_promote`, `memory_demote`, `code_memory_correlate`.
- **Reflection loop:** `reflection_loop_seed`, `reflection_loop_tick`, `reflection_loop_status`.
- **Procedure & audit:** `procedure_attempt`, `procedure_search`, `finding_resolve`, `audit_log`.
- **Health & diagnostics:** `loci_health`, `memory_health`, `retrieval_selftest`.
- **Advanced reasoning:** `investigation_reason` (grounded multi-perspective synthesis).

## Key files

- `mcp/server.py` — tool registration (FastMCP)
- `mcp/llm_tools.py`, `mcp/graph_tools.py`, `mcp/investigation_tools.py`
- `mcp/README.md`, `.mcp.json`
