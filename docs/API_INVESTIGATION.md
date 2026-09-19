# Investigation and Finding MCP Tool API Reference

An investigation in Loci is a durable case directory under `$LOCI_MEMORY_DIR/<investigation_id>/` with a `manifest.json` plus append-only logs such as `findings.jsonl`, `entities.jsonl`, `conflicts.jsonl`, and `retractions.jsonl`. The manifest holds case metadata; findings are the source-of-truth records; Qdrant is a search index over those records; the Ladybug graph links findings to code and entities. Loads and searches generally exclude soft-retracted findings by default, and lifecycle changes such as finding resolution are recorded as append-only overlay logs rather than in-place rewrites.

All tools below return `str` at the Python layer, containing JSON unless noted otherwise. Ground-truth check: the requested tool names all exist exactly, but they are split across `mcp/server.py`, `mcp/investigation_tools.py`, `mcp/graph_tools.py`, and `mcp/llm_tools.py`, then registered onto the shared FastMCP instance.

## Investigation lifecycle

### `investigation_start(investigation_id: str, title: str, context: Optional[str] = None) -> str`
Creates a new investigation manifest or resumes an existing one unchanged. The function is idempotent: if the ID already exists, it returns the current manifest and reports `status: "resumed"` instead of mutating context/title. The implementation also upserts the investigation node into the Ladybug investigation graph.

- **Key parameters**
  - `investigation_id`: Stable case slug or ticket-like identifier, e.g. `pww-actor-2026`.
  - `title`: One-line human description stored in the manifest.
  - `context`: Optional initial background; only used on first creation.
- **Returns**
  - `{"status":"created"|"resumed","manifest":{...}}`
  - `manifest` includes `id`, `title`, `context`, `status`, `hypothesis`, `open_questions`, `checked_sources`, `finding_counts`, ACL fields, and summary cache fields.
- **Example**
```python
investigation_start(
    investigation_id="inc-2026-09-17-auth-latency",
    title="Investigate auth latency spike",
    context="401s increased after the API gateway rollout at 18:10 UTC."
)
```

### `investigation_load(investigation_id: str, last_n_findings: int = 20, include_retracted: bool = False, requesting_agent_id: Optional[str] = None, fidelity: str = "full") -> str`
Loads an investigation for resume or handoff. In `full` mode it returns the manifest plus a recent-finding window; in `summary`/`brief` mode it prefers persisted summary fields and deterministic set-level invariants. The implementation explicitly filters non-finding access rows out of `findings.jsonl`, excludes soft-retracted findings by default, and can enforce ACL visibility when `requesting_agent_id` is supplied.

- **Key parameters**
  - `last_n_findings`: Recent window size. Older `gap` and `assumed` findings may be promoted into the window so open obligations do not age out silently.
  - `include_retracted`: Surfaces retracted findings instead of filtering them.
  - `requesting_agent_id`: Applies manifest ACL filtering when the case is shared.
  - `fidelity`: `full`, `summary`, or `brief`.
- **Returns**
  - `full`: `{"manifest":...,"fidelity":"full","total_findings":N,"recent_findings":[...],"field_invariants":...,"excluded_retracted":N,...}`
  - `summary`: manifest plus `summary_l1`, `summary_l2`, counts, and `field_invariants`.
  - `brief`: manifest plus `summary_l2` only.
  - `findings_omitted` appears only when the recent window is partial. `verifications` appears when `investigation_verify_all` has written advisory verdicts.
- **Example**
```python
investigation_load(
    investigation_id="inc-2026-09-17-auth-latency",
    last_n_findings=12,
    requesting_agent_id="agent-ops-01",
    fidelity="full"
)
```

### `investigation_as_of(investigation_id: str, as_of_timestamp: str) -> str`
Reconstructs what the investigation contained at a point in time. A finding is included only if it existed by `created_at_ts` and was still valid at the requested time according to `valid_until`; malformed or missing timestamps fail open rather than dropping the record. This is the tool for bi-temporal “what did we believe then?” reads.

- **Key parameters**
  - `as_of_timestamp`: ISO8601 timestamp; naive timestamps are treated as UTC.
- **Returns**
  - `{"investigation_id":...,"as_of":...,"findings":[...],"count":N}`
  - Each finding is the stored record with wrapped untrusted text.
- **Example**
```python
investigation_as_of(
    investigation_id="inc-2026-09-17-auth-latency",
    as_of_timestamp="2026-09-17T19:00:00+00:00"
)
```

### `investigation_note(investigation_id: str, field: str, value: str) -> str`
Updates investigation manifest state outside the findings log. Use it for case framing and workflow metadata: hypothesis, next step, open questions, checked sources, or final closeout summary. The tool validates empty values and updates the manifest in place, while logging a note event separately.

- **Key parameters**
  - `field`: One of `context`, `hypothesis`, `next_step`, `open_question_add`, `open_question_remove`, `checked_source`, `closed_summary`.
  - `value`: Meaning depends on `field`; `checked_source` must be `tool_name: summary`.
- **Returns**
  - `{"updated":"<field>","manifest":{...}}`
  - On `closed_summary`, the manifest is also transitioned to `status: "closed"` and stamped with `closed_at`.
- **Example**
```python
investigation_note(
    investigation_id="inc-2026-09-17-auth-latency",
    field="checked_source",
    value="query_logs: Confirmed p95 latency tripled only for /oauth/token after 18:10 UTC"
)
```

### `investigation_list(limit: int = 30, offset: int = 0, summary: bool = True) -> str`
Lists investigation manifests, newest-updated first. Summary mode is the default and returns compact rows sized for tool output; full mode scans each investigation’s findings to compute `tier_counts` and visibility metadata. Pagination is applied over manifest-bearing directories only.

- **Key parameters**
  - `limit`: Max investigations to return; `<= 0` means “everything starting at offset”.
  - `offset`: Number of investigations to skip.
  - `summary`: `True` for compact rows, `False` for expanded metadata.
- **Returns**
  - `{"investigations":[...],"total":N,"limit":...,"offset":...}`
  - Summary rows contain `id`, `title`, `status`, `updated_at`, `finding_counts`.
  - Full rows add `created_at`, `open_questions_count`, `hypothesis`, `visibility`, and `tier_counts`.
- **Example**
```python
investigation_list(limit=25, offset=0, summary=False)
```

### `investigation_share(investigation_id: str, agent_ids: list) -> str`
Adds agents to an investigation’s ACL so later `investigation_load(..., requesting_agent_id=...)` calls can see shared findings. The operation is additive and idempotent: already-present IDs stay in place and are not repeated in `shared_with`.

- **Key parameters**
  - `agent_ids`: List of agent IDs to grant access.
- **Returns**
  - `{"shared_with":[...],"total_acl":N}`
- **Example**
```python
investigation_share(
    investigation_id="inc-2026-09-17-auth-latency",
    agent_ids=["agent-ops-01", "agent-sre-02"]
)
```

### `investigation_unshare(investigation_id: str, agent_ids: list) -> str`
Removes agents from an investigation ACL. Missing IDs are ignored, so the tool is safe to use as an idempotent cleanup step after a handoff or review.

- **Key parameters**
  - `agent_ids`: List of agent IDs to revoke.
- **Returns**
  - `{"removed":[...],"total_acl":N}`
- **Example**
```python
investigation_unshare(
    investigation_id="inc-2026-09-17-auth-latency",
    agent_ids=["agent-sre-02"]
)
```

### `investigation_export(investigation_id: str, include_embeddings: bool = False) -> str`
Exports a portable JSON bundle for one investigation. The current implementation includes manifest, findings, conflicts, and entities, but `include_embeddings` is only a forward-compatibility flag: embeddings are not exported yet.

- **Key parameters**
  - `include_embeddings`: Accepted but currently ignored for payload content.
- **Returns**
  - `{"exported":true,"investigation_id":...,"bundle":{...},"finding_count":N,"size_bytes":N}`
  - `bundle` contains `schema_version`, `exported_at`, `manifest`, `findings`, `conflicts`, `entities`.
- **Example**
```python
investigation_export(
    investigation_id="inc-2026-09-17-auth-latency",
    include_embeddings=False
)
```

### `investigation_import(bundle_json: str, new_title: Optional[str] = None) -> str`
Imports an exported investigation bundle under a fresh investigation ID. The original ID is preserved as `imported_from`, findings are copied into the new case directory, and text-bearing findings are best-effort re-indexed into Qdrant. The tool accepts either the raw exported `bundle` object or the entire `investigation_export` response wrapper.

- **Key parameters**
  - `bundle_json`: JSON string containing either the raw bundle or the full export response.
  - `new_title`: Optional manifest title override.
- **Returns**
  - `{"imported":true,"new_investigation_id":"<uuid>","original_investigation_id":...,"findings_imported":N,"qdrant_indexed":N}`
  - Rejects bundles over 10 MB or with unsupported `schema_version`.
- **Example**
```python
investigation_import(
    bundle_json='{"bundle":{"schema_version":"1.0","manifest":{"id":"legacy-auth-case","title":"Legacy auth case"},"findings":[]}}',
    new_title="Imported auth latency case"
)
```

## Findings, lifecycle, and provenance

### `investigation_store(investigation_id: str, finding_type: str, text: str, source: str, confidence: str = "medium", tags: Optional[str] = None, derived_from: str | list[str] | None = None, numeric_confidence: float | None = None, procedure_preconditions: Optional[str] = None, procedure_steps: Optional[str] = None, procedure_postconditions: Optional[str] = None, valid_from: Optional[str] = None, valid_until: Optional[str] = None, authored_by: Optional[str] = None, tier: str = "warm", resolution: str = "open", code_refs: str | list[str] | None = None, metadata: Any | None = None, evidence_provenance_tier: Optional[str] = None) -> str`
Appends a finding to `findings.jsonl`, updates manifest counters, mirrors the record into Mnemosyne and optionally Qdrant, and best-effort links it into the code and investigation graphs. This is the central write path for evidence, hypotheses, gaps, and procedures. The implementation cross-checks lifecycle fields, stamps code-reference hashes when possible, records conflicts without blocking the write, and treats the second positional argument as `finding_type` rather than `record_type`.

- **Key parameters**
  - `finding_type`: `observed`, `inferred`, `assumed`, `gap`, or `procedure`.
  - `derived_from`: Upstream finding IDs or claims; used later for provenance tracing and retraction lineage.
  - `tier`: `hot`, `warm`, or `cold`; `cold` stays JSONL-only and is not indexed in Qdrant.
  - `resolution`: Initial lifecycle state: `open`, `fixed`, `intentional`, `wontfix`, `superseded`.
  - `code_refs`: Explicit authoritative file refs. Passing `[]`/`""` means “no refs”; only `None` triggers best-effort extraction from text.
  - `evidence_provenance_tier`: Authority tier such as `human_authored`, `tool_verified`, `deterministic_derived`, `model_asserted`.
- **Returns**
  - `{"stored":true,"finding_id":"<uuid>","type":"<finding_type>","mnemo_stored":true|false,"conflict_detected":bool,"tier":"hot|warm|cold"}`
  - When conflict detection fires, `conflicting_finding_id` and `conflict_id` are included.
- **Example**
```python
investigation_store(
    investigation_id="inc-2026-09-17-auth-latency",
    finding_type="observed",
    text="Gateway logs show p95 for /oauth/token rose from 180ms to 640ms immediately after deploy 9a4e7e1.",
    source="query_logs",
    confidence="high",
    tags="auth,latency,gateway",
    code_refs=["services/gateway/auth_proxy.py:120-176"],
    evidence_provenance_tier="tool_verified"
)
```

### `finding_resolve(investigation_id: str, finding_id: str, resolution: str, note: Optional[str] = None) -> str`
Changes a finding’s lifecycle state without rewriting the original finding. Instead of mutating `findings.jsonl`, it appends a resolution record to `finding_updates.jsonl`, and later reads fold those overlays in last-write-wins order. This is the intended path for closing out an item after review or remediation.

- **Key parameters**
  - `resolution`: One of `open`, `fixed`, `intentional`, `wontfix`, `superseded`.
  - `note`: Optional rationale for the lifecycle change.
- **Returns**
  - `{"resolved":true,"finding_id":...,"resolution":...}`
- **Example**
```python
finding_resolve(
    investigation_id="inc-2026-09-17-auth-latency",
    finding_id="c9ef1f7d-4fd0-49b3-aaf3-2ddab8b9da7f",
    resolution="fixed",
    note="Rollback restored latency; issue traced to sync token introspection call."
)
```

### `investigation_finding_provenance(finding_id: str, investigation_id: str) -> str`
Walks a finding’s `derived_from` chain back toward root evidence. The tool follows the first parent at each step, stops on cycles or missing nodes, caps depth at five, and computes an aggregate numeric confidence by multiplying each node’s numeric confidence. It is designed to answer “is this conclusion actually grounded in observed evidence, or only in other inferred/assumed records?”

- **Key parameters**
  - `finding_id`: Target finding to trace.
  - `investigation_id`: Owning investigation.
- **Returns**
  - `{"investigation_id":...,"chain_length":N,"grounded_in_observed":bool,"grounding_assessment":...,"aggregate_confidence":float|null,"chain":[...]}`
  - Each chain node includes `finding_id`, `ts`, `record_type`, `confidence`, `numeric_confidence`, `source`, wrapped `text`, and `derived_from`.
- **Example**
```python
investigation_finding_provenance(
    finding_id="c9ef1f7d-4fd0-49b3-aaf3-2ddab8b9da7f",
    investigation_id="inc-2026-09-17-auth-latency"
)
```

## Search, entity lookup, and code context

### `investigation_search(query: str, investigation_id: Optional[str] = None, limit: int = 10, include_retracted: bool = False, min_confidence: str = "low", resolution: Optional[str] = None) -> str`
Searches findings by a cascade: Mnemosyne recall first, Qdrant semantic/hybrid search second when needed, then local filtering and deduplication. The tool annotates lifecycle resolution on each row, excludes soft-retracted findings by default, wraps untrusted finding text, and reports which retrieval lanes were actually used.

- **Key parameters**
  - `investigation_id`: Restrict search to one case or omit for all investigations.
  - `min_confidence`: `low`, `medium`, or `high` floor applied after recall.
  - `resolution`: Optional lifecycle filter on the effective resolution state.
  - `include_retracted`: Include known-bad/retracted findings.
- **Returns**
  - `{"mode":...,"results":[...],"excluded_retracted":N,"include_retracted":bool,"resolution_filter":...,"mnemo_status":{...},"qdrant_status":{...}}`
  - Each result is a finding-shaped summary with investigation context and effective `resolution`.
  - Empty results return a `mode: "no_matches"` payload with backend status instead of an empty bare list.
- **Example**
```python
investigation_search(
    query="token introspection blocking event loop",
    investigation_id="inc-2026-09-17-auth-latency",
    limit=8,
    min_confidence="medium",
    resolution="open"
)
```

### `investigation_entity_lookup(entity: str, entity_type: str = "auto", investigation_id: Optional[str] = None, limit: int = 30) -> str`
Looks up a concrete observable such as an IP, email, hostname, hash, or CVE across one investigation or the entire memory store. It prefers indexed Qdrant payload lookup for exact entity search and falls back to JSONL scanning if necessary, then groups the findings by investigation.

- **Key parameters**
  - `entity`: Observable string to search for.
  - `entity_type`: `ip`, `email`, `hostname`, `hash`, `cve`, or `auto`.
  - `investigation_id`: Optional single-case scope.
- **Returns**
  - `{"entity":...,"entity_type":...,"scope":...,"total_findings":N,"investigations_count":N,"retrieval":"qdrant|jsonl_fallback","by_investigation":{...}}`
  - Grouped findings use compact summaries: `finding_id`, `investigation_id`, `ts`, `record_type`, `confidence`, `source`, `text`, and usually `tags`.
- **Example**
```python
investigation_entity_lookup(
    entity="api-gateway-01.corp",
    entity_type="hostname",
    limit=20
)
```

### `entity_list(investigation_id: str, entity_type: Optional[str] = None) -> str`
Returns entities already extracted into `entities.jsonl` for one investigation. This is the fastest “what named people/systems/concepts already appear in this case?” read and does not need vector search.

- **Key parameters**
  - `entity_type`: Optional filter such as `person`, `system`, `concept`, `location`, `other`.
- **Returns**
  - `{"entities":[{"entity_id", "name", "type", "finding_count"}, ...],"count":N}`
  - Results are sorted by `finding_count` descending.
- **Example**
```python
entity_list(
    investigation_id="inc-2026-09-17-auth-latency",
    entity_type="system"
)
```

### `entity_timeline(investigation_id: str, entity_id: str) -> str`
Builds a chronological story for one extracted entity by resolving its `finding_refs` back into the investigation’s findings log. Use it to see when a host, user, service, or concept first appeared and how later findings changed the picture.

- **Key parameters**
  - `entity_id`: Identifier returned by `entity_list`.
- **Returns**
  - `{"entity":{entity_id,name,type,aliases,first_seen,last_seen},"timeline":[{finding_id,ts,text,record_type}],"count":N}`
- **Example**
```python
entity_timeline(
    investigation_id="inc-2026-09-17-auth-latency",
    entity_id="system:api-gateway-01.corp"
)
```

### `investigation_related_cases(entities: str | list[str], entity_type: str = "auto", limit_per_entity: int = 5) -> str`
Finds prior investigations involving the same observables as a new alert or active case. Each entity is looked up independently, cross-case, then findings are grouped by related investigation with compact samples. This is the exact-entity counterpart to semantic case linkage.

- **Key parameters**
  - `entities`: One observable or a list; the implementation caps the list at 10.
  - `entity_type`: Exact type or `auto` per entity.
  - `limit_per_entity`: Controls how many findings to keep per entity lookup.
- **Returns**
  - `{"entities_queried":N,"results":[{"entity", "entity_type", "related_investigation_count", "retrieval", "related_investigations": {...}}, ...]}`
  - Each related-investigation entry contains `finding_count` and a `sample` of compact finding summaries.
- **Example**
```python
investigation_related_cases(
    entities=["api-gateway-01.corp", "CVE-2026-44102"],
    entity_type="auto",
    limit_per_entity=4
)
```

### `related_investigations_via_code(investigation_id: str, limit: int = 15) -> str`
Finds other investigations whose findings reference the same code symbols as the current case. Unlike entity lookup or semantic search, this is code-mediated case linkage: two cases can be related because they touch the same subsystem even if their wording and entities differ.

- **Key parameters**
  - `limit`: Maximum related investigations to return.
- **Returns**
  - `{"investigation_id":...,"related":[{"investigation", "shared_symbols", "sample_symbols"}, ...]}`
- **Example**
```python
related_investigations_via_code(
    investigation_id="inc-2026-09-17-auth-latency",
    limit=10
)
```

### `finding_code_context(finding_id: str) -> str`
Projects a finding into the code graph and returns the directly referenced symbols with their immediate caller/callee neighborhoods. Use it when a prose finding references code paths and you want the nearby actual code topology without writing a graph query yourself.

- **Key parameters**
  - `finding_id`: Stored finding UUID.
- **Returns**
  - `{"finding_id":...,"text":...,"symbols":[{"id","name","kind","file","line","callers":[...],"callees":[...]}, ...]}`
  - `symbols` is empty if the finding has no linked code refs or the graph is unavailable.
- **Example**
```python
finding_code_context("c9ef1f7d-4fd0-49b3-aaf3-2ddab8b9da7f")
```

### `investigation_code_briefing(investigation_id: str, top: int = 3) -> str`
Summarizes a case’s code footprint by composing multiple graph analytics: the investigation footprint, hotspot-symbol impact reports, and code-overlap case linkage. It is best read as an onboarding/triage briefing for “what code does this investigation actually touch?”

- **Key parameters**
  - `top`: Number of hotspot symbols to expand with impact details.
- **Returns**
  - `{"investigation":...,"finding_count":N,"symbols_touched":N,"files_touched":N,"top_symbols":[{"symbol","in_investigation_findings","transitive_callers","total_referencing_findings","other_investigations"}],"related_investigations":[...]}`
- **Example**
```python
investigation_code_briefing(
    investigation_id="inc-2026-09-17-auth-latency",
    top=5
)
```

## Verification, evidence checks, and skeptical review

### `investigation_pre_answer_check(investigation_id: str, claims: str | list[str], min_confidence: str = "medium", record: bool = True) -> str`
Checks one or more proposed claims against the investigation’s evidence before those claims are used in an answer. The implementation is stricter than “semantic similarity”: it combines lexical evidence, corroborated semantic candidates, contradiction detection, provenance-firewall gating, and optional local-model entailment checks. When `record=True`, claim verdicts are also written to the `loci_verdicts` collection so repeated claims and verdict conflicts are visible later.

- **Key parameters**
  - `claims`: One claim string or a list of claims.
  - `min_confidence`: Evidence floor: `low`, `medium`, `high`.
  - `record`: Whether to persist verdict history.
- **Returns**
  - `{"investigation_id":...,"checked_at":...,"claims_checked":N,"support_count":N,"contradiction_count":N,"unsupported_claims":[...],"matched_evidence_ids":[...],"matched_evidence":[...],"claim_results":[...],"verdict_recording":...,"min_chain_confidence":...,"confidence_summary":...,"qdrant_status":...,"evidence_lanes":...,"degraded_mode":...}`
  - Each `claim_results` row includes `supported`, `contradicted`, `ambiguous`, `support_basis`, `support_refs`, `semantic_candidates`, `contradiction_refs`, `benign_context_refs`, and `provenance_firewall`; `llm_entailment_check` appears only when the additive entailment lane runs.
- **Example**
```python
investigation_pre_answer_check(
    investigation_id="inc-2026-09-17-auth-latency",
    claims=[
        "The latency spike started immediately after deploy 9a4e7e1.",
        "The auth database was saturated during the incident."
    ],
    min_confidence="medium",
    record=True
)
```

### `investigation_evidence_precheck(investigation_id: str, proposed_query: str, min_similarity: float = 0.4) -> str`
Checks whether an intended probe likely duplicates evidence already stored in the investigation or its recent audit receipts. It is a lightweight planning guard, not a truth-verification tool: lexical matches and Qdrant matches are advisory “similar evidence” signals, then a provenance firewall can suppress model-only support.

- **Key parameters**
  - `proposed_query`: Planned question/probe text.
  - `min_similarity`: Numeric threshold clamped into `[0.1, 1.0]`.
- **Returns**
  - `{"investigation_id":...,"proposed_query":...,"has_similar_evidence":bool,"similar_evidence_count":N,"similar_evidence":[...],"provenance_firewall":...,"qdrant_status":...,"evidence_lanes":...,"degraded_mode":...}`
- **Example**
```python
investigation_evidence_precheck(
    investigation_id="inc-2026-09-17-auth-latency",
    proposed_query="Query gateway logs for p95 latency on /oauth/token after 18:10 UTC",
    min_similarity=0.45
)
```

### `verify_finding(claim: str, context: str = "", investigation_id: Optional[str] = None, finding_id: Optional[str] = None, auto_promote_procedures: bool = False) -> str`
Runs a local-model skeptic against a single claim and returns `confirmed`, `refuted`, or `uncertain`. If explicit `context` is absent, the tool will best-effort pull RAG grounding when `investigation_id` is provided. When both `investigation_id` and `finding_id` identify a stored finding, the wrapper also threads that finding’s provenance tier and sibling findings into the provenance firewall so model-asserted claims cannot self-confirm without independent support.

- **Key parameters**
  - `context`: Optional supporting evidence, code, or notes to challenge.
  - `investigation_id`: Enables best-effort automatic grounding.
  - `finding_id`: Allows stored-finding provenance context and optional procedure promotion.
  - `auto_promote_procedures`: On a non-degraded confirmed result, can promote action-shaped stored findings into procedure memory.
- **Returns**
  - `{"verdict":"confirmed|refuted|uncertain","refutation":str,"confidence":float,"degraded":bool}`
  - The underlying verifier also computes `reasoning`; callers should expect it to appear in the JSON payload even though the wrapper docstring emphasizes the core fields.
- **Example**
```python
verify_finding(
    claim="The sync token introspection call blocks the event loop in the gateway.",
    investigation_id="inc-2026-09-17-auth-latency",
    finding_id="c9ef1f7d-4fd0-49b3-aaf3-2ddab8b9da7f"
)
```

### `investigation_verify_all(investigation_id: str, limit: int = 20) -> str`
Batch-runs the skeptic over open findings in one investigation. This is explicitly advisory triage: the tool writes append-only verification notes but does not change finding lifecycle resolution. Internally it skips resolved and soft-retracted findings, passes code refs and provenance tiers into `verify_finding`, and marks degraded model-unavailable outcomes separately.

- **Key parameters**
  - `limit`: Maximum number of open findings to verify.
- **Returns**
  - `{"investigation_id":...,"verified":N,"results":[{"finding_id","verdict","confidence","degraded"}, ...]}`
  - Per-finding verdicts are also appended to `finding_verifications.jsonl` for later surfacing in `investigation_load`.
- **Example**
```python
investigation_verify_all(
    investigation_id="inc-2026-09-17-auth-latency",
    limit=10
)
```

## Reflection and reasoning

### `investigation_reflect(investigation_id: str) -> str`
Builds a structured “state of the case” reflection. It summarizes finding counts, open questions, checked sources, gaps, recent findings by type, extracted key entities, and self-check outputs such as unsupported observations and contradictions. When a local model is available, it also generates and persists `summary_l1`, `summary_l2`, and a short `self_critique`; otherwise it falls back to deterministic summaries.

- **Key parameters**
  - `investigation_id`: Target investigation.
- **Returns**
  - `{"investigation_id":...,"title":...,"status":...,"hypothesis":...,"next_step":...,"finding_counts":...,"open_questions":[...],"checked_sources":...,"gaps":[...],"recent_per_type":...,"key_entities":...,"excluded_retracted":N,"self_check":...,"summary_l1":[...],"summary_l2":str,"self_critique":str?}`
- **Example**
```python
investigation_reflect("inc-2026-09-17-auth-latency")
```

### `investigation_reason(investigation_id: str, question: str, perspectives: int = 3, ground_threshold: float = 0.59, persist: bool = False) -> str`
Runs multi-perspective grounded reasoning over one investigation. The tool first gates findings to the question by embedding similarity, then fans out to several adversarial analyst prompts and performs a synthesis pass to extract converged claims, contested areas, and a final answer. It is intentionally heavier than search or reflection because it makes `N+1` model calls inline.

- **Key parameters**
  - `question`: Problem to reason about.
  - `perspectives`: Number of analyst perspectives, clamped to 1-5.
  - `ground_threshold`: Similarity floor used before reasoning; lower-scoring findings are dropped.
  - `persist`: Whether converged claims should be stored back as `inferred` findings with source `investigation_reason`.
- **Returns**
  - `{"investigation_id":...,"question":...,"perspectives_used":[...],"grounded_findings":N,"gate_applied":bool,"confidence_score":...,"converged_claims":[...],"contested_areas":[...],"final_answer":str,"persisted_finding_ids":[...]}`
- **Example**
```python
investigation_reason(
    investigation_id="inc-2026-09-17-auth-latency",
    question="What is the most likely mechanism behind the post-deploy auth latency spike?",
    perspectives=3,
    ground_threshold=0.59,
    persist=False
)
```

## Closely related tools not in the requested list

These also sit in the same investigation/finding workflow and are worth knowing about, but are not expanded here:

- `procedure_attempt` / `procedure_search`: procedure-memory write and lookup tools adjacent to `verify_finding` and `investigation_store(... finding_type="procedure" ...)`.
- `audit_log`: append-only capture of tool calls and outputs into global and investigation-scoped audit logs.
- `memory_retract` / `memory_restore`: lifecycle operations for soft-retracting false findings and later restoring them.
