# Memory subsystem MCP tool API reference

This is a source-checked reference for Loci's memory-facing MCP tools. It complements
[docs/CONCEPTS.md](CONCEPTS.md) and the broader operational guide in
[docs/memory-and-code-review-tools.md](memory-and-code-review-tools.md): those docs explain
*why* the subsystem exists and how it is used in practice; this file focuses on the
actual tool signatures, defaults, return payloads, and example calls.

Source note: most memory tools below are defined directly in `mcp/server.py`. A smaller
set — `query_expand`, `classify_text`, `compress_text`, `semantic_dedup`,
`semantic_relevance`, and `ground` — are registered from `mcp/llm_tools.py` and then
re-exported by `mcp/server.py`, so they belong to the same MCP surface even though their
implementations live in the split-out helper module.

## Memory model and lifecycle

Loci uses two different memory concepts that are easy to conflate:

- **Storage tier on each finding:** `hot`, `warm`, or `cold`.
  - `hot` — searchable in Qdrant **and** its text snippet is appended to the investigation
    manifest notes for immediate in-context use.
  - `warm` — searchable in Qdrant; this is the normal vector-searchable tier.
  - `cold` — kept in `findings.jsonl` only; archived and **not** vector-searchable.
- **Mnemosyne substrate state:** `working_memory` and `episodic_memory` are the internal
  SQLite layers used by consolidation. They are separate from `hot`/`warm`/`cold`.

Key lifecycle tools:

- `memory_promote` / `memory_demote` change a finding's **storage tier**.
- `memory_retract` appends a reversible soft tombstone so a contaminated finding stops
  appearing in recall, search, and reflection.
- `memory_restore` appends `active:false` to reverse that tombstone.
- `memory_consolidate` runs Mnemosyne's sleep pass, merging `working_memory` into
  `episodic_memory`; it does **not** replace promote/demote.

Health vs trustworthiness:

- `memory_health` checks the **machinery**: Qdrant, embedders, Mnemosyne importability,
  dimension consistency, and retraction/store integrity.
- `retrieval_selftest` goes one level deeper on Qdrant and asks whether collections are
  actually retrievable, even when `memory_health` looks superficially green.
- `memory_self_check` checks the **stored findings** themselves for missing provenance and
  contradictions. It is advisory and never auto-retracts.
- `memory_confidence` estimates how reliable Loci's memory is for a specific claim/topic
  before you assert it.

Also note the distinction called out in
[docs/memory-and-code-review-tools.md](memory-and-code-review-tools.md): a finding's
**evidence provenance tier** (`human_authored`, `tool_verified`, `deterministic_derived`,
`model_asserted`) is separate from its `hot`/`warm`/`cold` storage tier.

## Tool reference

### `memory_self_check`

**Signature**

```python
memory_self_check(
    investigation_id: Optional[str] = None,
    checks: str = "provenance,contradiction",
    record: bool = True,
    llm_verify: bool = False,
) -> str
```

Runs Loci's advisory self-audit over stored findings. It checks whether `observed`
findings can be traced back to audit receipts and whether pairs of findings appear to
contradict each other. When both checks run, it also surfaces
`hallucination_candidates`: findings that are both unsupported and contradicted by a
receipted finding. The tool never mutates findings or auto-retracts anything.

**Key parameters**

- `investigation_id`: check one investigation; omit to scan all investigation directories.
- `checks`: comma-separated subset of `provenance,contradiction`.
- `record`: best-effort recording of verdicts into `loci_verdicts` when Qdrant is up.
- `llm_verify`: enables the deeper semantic contradiction path; otherwise the check stays
  pure/offline and lexical.

**Returns**

JSON string with top-level fields including `checked_at`, `advisory`, `checks`, `counts`,
`hallucination_candidates`, `recorded`, and `qdrant`. Single-investigation calls return
`investigation_id` plus a flat `verdicts` array; global calls return `investigation_ids`
and per-investigation entries under `investigations`.

**Example**

```json
{
  "tool": "memory_self_check",
  "arguments": {
    "investigation_id": "auth-review-2026-09-17",
    "checks": "provenance,contradiction",
    "record": true
  }
}
```

### `memory_health`

**Signature**

```python
memory_health(investigation_id: Optional[str] = None) -> str
```

Checks the memory substrate itself, read-only. Where `memory_self_check` judges what was
remembered, `memory_health` judges the server components that make memory work: Qdrant
reachability, collection layout, dense and sparse embedders, Mnemosyne availability in the
current venv, vector-width consistency, and investigation retraction/store integrity.

**Key parameters**

- `investigation_id`: scopes retraction/store-count checks to one investigation; the core
  substrate probes still run globally.

**Returns**

JSON string:

- `checked_at`
- `status`: `ok`, `degraded`, or `unhealthy`
- `scope`: investigation id or `all`
- `checks`: eight probe rows named `qdrant_reachable`, `qdrant_collections`,
  `embeddings_dense`, `embeddings_sparse`, `mnemo_mirror`, `dimension_consistency`,
  `retraction_integrity`, and `store_counts`
- `summary`

Each `checks[]` row contains `name`, `status`, `detail`, and sometimes `remediation`.

**Example**

```json
{
  "tool": "memory_health",
  "arguments": {
    "investigation_id": "incident-ssh-key-rotation"
  }
}
```

### `memory_retract`

**Signature**

```python
memory_retract(
    investigation_id: str,
    target: str,
    reason: str = "",
    dry_run: bool = True,
    scope_semantic: bool = True,
) -> str
```

Soft-tombstones a hallucinated or contaminated finding cluster without deleting source
records. The cluster is built from the seed finding(s), distinctive shared entities,
semantic neighbors in Qdrant, and forward `derived_from` lineage. The default is advisory:
`dry_run=True` previews what would be retracted so a caller can review it before applying.

**Key parameters**

- `investigation_id`: owning investigation.
- `target`: either a finding id or a claim/entity string to resolve into seed findings.
- `reason`: audit-trail explanation stored with the retraction.
- `dry_run`: preview only by default; set `false` to append active tombstones.
- `scope_semantic`: include Qdrant semantic neighbors when building the cluster.

**Returns**

- `dry_run=True`: `{seed_ids, would_retract, count, applied:false, scope_semantic, semantic_neighbors, advisory}`
- `dry_run=False`: `{seed_ids, retracted, count, verdicts_forgotten, applied:true, quarantine_verdict_recorded, reversible}`
- errors/busy cases return an error payload instead

Each `would_retract`/`retracted` row includes the affected finding id, excerpt, and reasons.

**Example**

```json
{
  "tool": "memory_retract",
  "arguments": {
    "investigation_id": "oauth-audit",
    "target": "finding-1d1e8c2f",
    "reason": "redirect endpoint never existed; confirmed by live test",
    "dry_run": true
  }
}
```

### `memory_restore`

**Signature**

```python
memory_restore(
    investigation_id: str,
    finding_id: Optional[str] = None,
    retraction_id: Optional[str] = None,
    reason: str = "",
) -> str
```

Reverses a retraction by appending an `active:false` entry to `retractions.jsonl`. That
makes the read path stop filtering the finding, so it returns to recall, search, and
reflection. No prior records are edited or removed.

**Key parameters**

- `finding_id`: direct restore target.
- `retraction_id`: alternate handle; resolved back to the underlying `finding_id`.
- `reason`: optional audit note.

Exactly one of `finding_id` or a resolvable `retraction_id` is required.

**Returns**

Usually `{finding_id, restored:true, reason}`. If the finding is not currently retracted,
the tool returns `{finding_id, restored:false, note}`. Invalid targets return an error
payload.

**Example**

```json
{
  "tool": "memory_restore",
  "arguments": {
    "investigation_id": "oauth-audit",
    "finding_id": "finding-1d1e8c2f",
    "reason": "test fixture was wrong; original finding is valid"
  }
}
```

### `memory_surface`

**Signature**

```python
memory_surface(
    context: str,
    investigation_id: Optional[str] = None,
    top_k: int = 5,
) -> str
```

Surfaces tangentially relevant prior findings from a free-form work-context paragraph. It
is intended for passive context injection at the start of a task, when the caller knows the
current work area but not yet the exact question to ask. Compared with `rag_context_search`,
it uses a looser score threshold and a simpler relevance note.

**Key parameters**

- `context`: paragraph describing the current work.
- `investigation_id`: optional Qdrant filter for one investigation.
- `top_k`: number of surfaced findings to return.

**Returns**

JSON string `{surfaced, context_used, count}` or an error payload. Each `surfaced[]` row is
`{finding_id, text, source, relevance_note, score, investigation_id}`. The text is trimmed
to at most 300 characters and wrapped as untrusted memory text.

**Example**

```json
{
  "tool": "memory_surface",
  "arguments": {
    "context": "Reviewing auth middleware and prior OAuth redirect handling before changing callback validation.",
    "investigation_id": "oauth-audit",
    "top_k": 4
  }
}
```

### `memory_consolidate`

**Signature**

```python
memory_consolidate(dry_run: bool = False) -> str
```

Runs Mnemosyne's sleep/consolidation cycle. This merges older `working_memory` entries into
`episodic_memory`, then — on non-dry runs — attempts a best-effort causal-inference pass on
the most recent sufficiently large investigation. It is maintenance for the memory
substrate, not a content-audit tool.

**Key parameters**

- `dry_run`: preview the consolidation without executing it.

**Returns**

On success: `{status:"ok", dry_run, result, causal_edges_inferred}` and, for real runs,
optionally `consolidation_quality_audit`. On internal failure: `{status:"error", error,
type, causal_edges_inferred}`.

**Example**

```json
{
  "tool": "memory_consolidate",
  "arguments": {
    "dry_run": true
  }
}
```

### `memory_confidence`

**Signature**

```python
memory_confidence(
    query: str,
    top_k: int = 8,
) -> str
```

Estimates how reliably Loci knows about a claim or topic before you state it as fact. The
score is computed from five cues drawn from retrieval results: fluency, accessibility,
source diversity, corroboration, and trust. It may also attach an advisory
`llm_entailment_note`, but that note does not change the numeric confidence verdict.

**Key parameters**

- `query`: the claim/topic to test.
- `top_k`: number of hits to use when computing cues.

**Returns**

Normal success payload:

- `confidence`
- `basis`: typically `recollection`, `corroboration`, `trust`, or `familiarity`
- `cues`: `fluency`, `accessibility`, `source_diversity`, `corroboration`, `trust`
- `top_hit_preview`
- `recommendation`
- optional `llm_entailment_note`

Hard-stop payloads use `confidence: 0.0` and a `basis` such as `empty_query`,
`qdrant_unavailable`, `embed_failed`, `search_failed`, or `no_trace`.

**Example**

```json
{
  "tool": "memory_confidence",
  "arguments": {
    "query": "We previously proved the OAuth callback must reject mismatched redirect_uri values.",
    "top_k": 6
  }
}
```

### `memory_promote`

**Signature**

```python
memory_promote(investigation_id: str, finding_id: str, tier: str) -> str
```

Moves a finding upward in Loci's storage-tier model. Promotion updates the finding's `tier`
in `findings.jsonl`, rewrites that file atomically, and performs the corresponding Qdrant or
manifest-note maintenance. Promoting to `hot` also appends a short text snippet to the
investigation manifest notes.

**Key parameters**

- `investigation_id`: owning investigation.
- `finding_id`: finding UUID/id.
- `tier`: target tier, one of `hot`, `warm`, or `cold`.

**Returns**

`{finding_id, old_tier, new_tier, ok:true}` on success, including no-op moves where the
requested tier already matches. Invalid tiers, missing investigations/findings, or lock
contention return error/busy payloads.

**Example**

```json
{
  "tool": "memory_promote",
  "arguments": {
    "investigation_id": "oauth-audit",
    "finding_id": "finding-1d1e8c2f",
    "tier": "hot"
  }
}
```

### `memory_demote`

**Signature**

```python
memory_demote(investigation_id: str, finding_id: str, tier: str) -> str
```

Moves a finding downward in the same `hot`/`warm`/`cold` model. The underlying implementation
is shared with `memory_promote`, but the operational effect matters: demoting to `cold`
removes the vector from Qdrant, so the finding remains on disk but disappears from semantic
search and other vector-backed recall paths.

**Key parameters**

- `investigation_id`: owning investigation.
- `finding_id`: finding UUID/id.
- `tier`: target tier, one of `hot`, `warm`, or `cold`.

**Returns**

`{finding_id, old_tier, new_tier, ok:true}` on success, otherwise an error/busy payload.

**Example**

```json
{
  "tool": "memory_demote",
  "arguments": {
    "investigation_id": "oauth-audit",
    "finding_id": "finding-1d1e8c2f",
    "tier": "cold"
  }
}
```

### `memory_hints`

**Signature**

```python
memory_hints(
    investigation_id: str,
    limit: int = 3,
    since_ts: Optional[str] = None,
    mode: Literal["normal", "compact"] = "normal",
) -> str
```

Returns recent findings from one investigation as lightweight polling hints. It prefers the
in-process session ring buffer for speed and falls back to the tail of `findings.jsonl`
after restarts. This is meant for "what changed recently?" refreshes without reloading a
full investigation or running a broader search.

**Key parameters**

- `investigation_id`: required investigation scope.
- `limit`: max hints, clamped to `1..20`.
- `since_ts`: ISO-8601 watermark; only findings with `ts > since_ts` are returned.
- `mode`: `compact` clips each hint's text but preserves the rest of the shape.

**Returns**

JSON string `{investigation_id, hints, count, as_of}`. Each `hints[]` row is
`{finding_id, text, source, record_type, recency_score, ts}` with most-recent-first output.

**Example**

```json
{
  "tool": "memory_hints",
  "arguments": {
    "investigation_id": "oauth-audit",
    "limit": 5,
    "since_ts": "2026-09-17T18:00:00+00:00",
    "mode": "compact"
  }
}
```

### `memory_route`

**Signature**

```python
memory_route(
    query: str,
    agent_id: Optional[str] = None,
    top_k: int = 10,
    deduplicate: bool = True,
) -> str
```

Searches across all investigations instead of a single investigation. This is the
memory-mesh routing tool for "who has seen something like this before?" workflows. It can
filter to findings authored by a specific agent or to investigations whose ACL includes that
agent, then optionally suppress near-duplicate hits by word overlap.

**Key parameters**

- `query`: required cross-investigation search query.
- `agent_id`: optional author/ACL filter.
- `top_k`: max returned rows after filtering/deduplication.
- `deduplicate`: suppress >80% word-overlap duplicates when true.

**Returns**

JSON string `{routed, query, agent_id, total_before_dedup, total_after_dedup, count}`.
Each `routed[]` row is `{finding_id, investigation_id, investigation_title, text, source,
authored_by, score, tier}`.

**Example**

```json
{
  "tool": "memory_route",
  "arguments": {
    "query": "redirect_uri mismatch handling",
    "agent_id": "copilot-reviewer",
    "top_k": 8
  }
}
```

### `procedure_attempt`

**Signature**

```python
procedure_attempt(
    investigation_id: str,
    finding_id: str,
    success: bool,
) -> str
```

Records an execution attempt against a procedure-type finding. It increments
`procedure_meta.attempt_count`, increments `success_count` when appropriate, rewrites
`findings.jsonl` atomically, and best-effort updates the finding payload in Qdrant. This is
how reusable runbooks accumulate real pass/fail history.

**Key parameters**

- `investigation_id`: owning investigation.
- `finding_id`: target procedure finding.
- `success`: whether this run succeeded.

**Returns**

`{finding_id, success_count, attempt_count, success_rate}` where `success_rate` is `null`
only when the procedure has never been attempted. Errors return `{error}`.

**Example**

```json
{
  "tool": "procedure_attempt",
  "arguments": {
    "investigation_id": "deploy-playbooks",
    "finding_id": "procedure-restart-worker-pool",
    "success": true
  }
}
```

### `procedure_search`

**Signature**

```python
procedure_search(
    query: str,
    investigation_id: Optional[str] = None,
    limit: int = 5,
) -> str
```

Finds procedure-type findings relevant to a query. The happy path uses Qdrant filtered to
`record_type == "procedure"`; if Qdrant is unavailable it falls back to a direct JSONL scan
over one investigation or all investigation directories.

**Key parameters**

- `query`: natural-language description of the procedure needed.
- `investigation_id`: optional scope restriction.
- `limit`: max results.

**Returns**

JSON string `{procedures, count}` or `{error, procedures: [], count: 0}` on failure. Each
`procedures[]` row includes `finding_id`, `text`, `source`, `investigation_id`,
`success_rate`, `procedure_meta`, and `score`.

**Example**

```json
{
  "tool": "procedure_search",
  "arguments": {
    "query": "restart background worker after queue schema change",
    "investigation_id": "deploy-playbooks",
    "limit": 3
  }
}
```

### `rag_context_search`

**Signature**

```python
rag_context_search(
    query: str,
    limit: int = 10,
    collections: Optional[list] = None,
    budget_chars: int = 6000,
    exclude_types: Optional[list] = None,
    decay: bool = True,
    expand_query: Optional[bool] = None,
    mode: Literal["normal", "compact"] = "normal",
) -> str
```

Runs Loci's Qdrant-backed hybrid RAG path and returns prompt-ready cited context. It can
search the main findings collection plus the configured code-chunks collection, widen the
candidate pool with `query_expand`, apply time decay to findings, re-rank across
collections, and finally assemble a prompt block with `[SOURCE N]` citations.

**Key parameters**

- `query`: required natural-language retrieval query.
- `collections`: explicit collection override. By default the tool searches
  `[QDRANT_COLLECTION_PREFIX] + [CODE_CHUNKS_COLLECTION if set]`.
- `budget_chars`: cap for the assembled context string.
- `exclude_types`: payload `type` values excluded from the code-chunks-style lane; default
  is `['gps_trajectory']`, and `[]` disables that filter.
- `decay`: apply Ebbinghaus-style score decay to findings results.
- `expand_query`: `None` follows `LOCI_RAG_EXPAND` (default on), otherwise forces query
  expansion on or off.
- `mode`: `normal` markdown block or `compact` terse cited lines.

**Returns**

Base success payload from `context_assemble`:

- `query`
- `context`
- `sources`: `{n, id, title, origin, score}`
- `total_chars`
- `truncated`
- `result_count`

Plus RAG-specific fields:

- `mode`: usually `rag_hybrid`, `rag_degraded`, `rag_failed`, or `rag_required`
- `collections_searched`
- `collections_failed`
- `qdrant_available`
- optional `query_expansion`
- optional `collection_errors`

If Qdrant is unavailable the tool returns `mode: "rag_required"` and an error payload.

**Example**

```json
{
  "tool": "rag_context_search",
  "arguments": {
    "query": "OAuth callback validation and redirect_uri mismatch findings",
    "limit": 6,
    "budget_chars": 3500,
    "mode": "compact"
  }
}
```

### `retrieval_selftest`

**Signature**

```python
retrieval_selftest(
    query: str = "system architecture",
    limit: int = 3,
    collections: Optional[list] = None,
    scope: str = "queried",
) -> str
```

Probes whether the Qdrant collections this server cares about are actually retrievable. It
bypasses the cross-encoder and relevance floor on purpose, because this tool is checking the
retrieval wiring itself rather than semantic quality. It is particularly useful for spotting
vector-width mismatches that look like harmless zero-hit searches from the outside.

**Key parameters**

- `query`: probe text; any in-domain phrase is acceptable.
- `limit`: rows requested per collection.
- `collections`: explicit collection list; overrides `scope`.
- `scope`: `queried` checks only the collections this server would normally search,
  while `all` inventories the whole store without counting out-of-scope breakage against
  health.

**Returns**

JSON string `{status, summary, scope, embedder_dim, collections, remediations}`.
Each `collections[]` row includes `collection`, `points`, `dense_dims`, `sparse`, `hits`,
`status`, `detail`, `queried_by_server`, and sometimes `remediation`. Collection status is
one of `ok`, `empty`, `no_results`, `width_mismatch`, or `error`.

**Example**

```json
{
  "tool": "retrieval_selftest",
  "arguments": {
    "scope": "all",
    "limit": 2
  }
}
```

### `query_expand`

**Signature**

```python
query_expand(query: str, n_queries: int = 3, n_keywords: int = 6) -> str
```

Expands a retrieval query into alternate phrasings plus domain keywords using the local
model path. Loci uses this as a cheap candidate-pool widener ahead of embedding retrieval
and reranking. The tool is fail-open: if the local generator is unavailable, callers still
get a runnable result anchored on the original query.

**Key parameters**

- `query`: base query to expand.
- `n_queries`: cap on total queries, including the original seed query.
- `n_keywords`: cap on extracted keywords.

**Returns**

`{queries, keywords, degraded}`. `queries` always begins with the original query when one is
provided. Empty input returns an empty `queries` list with `degraded:true`.

**Example**

```json
{
  "tool": "query_expand",
  "arguments": {
    "query": "redirect_uri mismatch",
    "n_queries": 4,
    "n_keywords": 8
  }
}
```

### `classify_text`

**Signature**

```python
classify_text(text: str, labels: list) -> str
```

Runs a strict single-label local-model classification step. It is intended as a cheap
router/gate in places where spawning a separate classifier agent would be overkill. The
returned label is validated against the allowed set, case-insensitively, before being
accepted.

**Key parameters**

- `text`: text to classify.
- `labels`: allowed labels; empty labels degrade immediately.

**Returns**

`{label, degraded}`. On any unavailable-model or out-of-set-label path, `label` becomes
`null` and `degraded` is `true`.

**Example**

```json
{
  "tool": "classify_text",
  "arguments": {
    "text": "Need cross-investigation recall before opening a new auth incident.",
    "labels": ["grounding", "memory-routing", "procedure", "unrelated"]
  }
}
```

### `compress_text`

**Signature**

```python
compress_text(text: str, max_chars: int = 600) -> str
```

Condenses text to a target character budget with a local model. It is meant for low-cost
pre-summarization before a Claude synthesis stage or before saving/passing large chunks of
context. If the text already fits, the tool returns it unchanged.

**Key parameters**

- `text`: source text.
- `max_chars`: hard output ceiling; over-budget model output is clamped and marked degraded.

**Returns**

`{text, degraded}`. On generator failure or overshoot, the fallback is a plain character
truncation to `max_chars` with `degraded:true`.

**Example**

```json
{
  "tool": "compress_text",
  "arguments": {
    "text": "<large retrieved context block>",
    "max_chars": 500
  }
}
```

### `semantic_dedup`

**Signature**

```python
semantic_dedup(
    items: list,
    threshold: float = 0.88,
    text_key: Optional[str] = None,
) -> str
```

Clusters near-duplicate items by local embedding cosine similarity so downstream synthesis
sees one representative per cluster. It is designed for cheap post-fanout cleanup, not for
canonical database deduplication. The algorithm is greedy and order-stable: first seen wins
inside each cluster.

**Key parameters**

- `items`: strings or dicts.
- `threshold`: cosine floor for duplicate clustering; higher values are stricter.
- `text_key`: when `items` contains dicts, choose the text field explicitly instead of
  relying on fallback keys such as `text`, `content`, `summary`, or `title`.

**Returns**

`{clusters, kept, dropped, degraded}` where each cluster is
`{rep_index, member_indices, text}`. If embeddings are unavailable, every item is kept,
`dropped` is `0`, and `degraded` is `true`.

**Example**

```json
{
  "tool": "semantic_dedup",
  "arguments": {
    "items": [
      {"summary": "OAuth callback rejects mismatched redirect_uri"},
      {"summary": "redirect_uri mismatch is denied at callback"},
      {"summary": "Refresh token rotation handled elsewhere"}
    ],
    "threshold": 0.9,
    "text_key": "summary"
  }
}
```

### `semantic_relevance`

**Signature**

```python
semantic_relevance(texts: list, topic: str) -> str
```

Scores each text's embedding cosine relevance to a topic. This is a cheap local scoring
primitive for routing, ranking, or thresholding before a more expensive Claude reasoning
step.

**Key parameters**

- `texts`: candidate texts to score.
- `topic`: required comparison topic.

**Returns**

Usually `{scores, degraded}` where `scores` aligns 1:1 with `texts` and each value is a
rounded float or `null`. Empty/missing topic returns `degraded:true` and an `error` field;
missing embeddings also degrade to `scores:[null, ...]`.

**Example**

```json
{
  "tool": "semantic_relevance",
  "arguments": {
    "texts": [
      "OAuth callback validates redirect_uri against the registered client.",
      "Queue worker restart procedure after schema migration.",
      "TLS certificate rotation playbook."
    ],
    "topic": "redirect_uri validation"
  }
}
```

### `ground`

**Signature**

```python
ground(
    title: str,
    focus: str = "",
    case_ids: Optional[list] = None,
    entities: Optional[list] = None,
    code_refs: Optional[list] = None,
    budget_chars: int = 4000,
    allow_keyword: bool = False,
    graph_available: bool = False,
    mode: Literal["normal", "compact"] = "normal",
) -> str
```

Builds a compact grounding block for a task so a caller can inject one curated memory bundle
into prompts instead of having every downstream agent perform its own memory lookups.
Retrieval is structured-first where possible: named investigations, exact entities,
code-graph references, semantic RAG, curated `MEMORY.md`, and finally optional keyword
fallback.

**Key parameters**

- `title`: required short task title; empty titles are rejected.
- `focus`: longer task description.
- `case_ids`: named investigations to load directly.
- `entities`: exact entity lookups.
- `code_refs`: code symbols for the optional code-graph lane.
- `budget_chars`: cap for the assembled block.
- `allow_keyword`: opt into the noisier keyword fallback lane.
- `graph_available`: enable the code-graph lane.
- `mode`: `normal` block or `compact` tagged lines.

**Returns**

`{block, sources, chars, degraded}`. `sources` is a list of short lane tags such as
`case:<id>`, `entity:<value>`, `rag`, `memory:<slug>`, or warning markers when a lane is
unconfigured or missing.

**Example**

```json
{
  "tool": "ground",
  "arguments": {
    "title": "Review OAuth callback validation",
    "focus": "Need prior findings, relevant entities, and any code-linked history before editing redirect checks.",
    "case_ids": ["oauth-audit"],
    "entities": ["redirect_uri"],
    "code_refs": ["validate_redirect_uri"],
    "budget_chars": 3000,
    "graph_available": true,
    "mode": "compact"
  }
}
```
