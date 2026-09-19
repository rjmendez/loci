# Code-graph, contract, causality, and audit MCP API reference

This page documents the MCP tool-call surface for Loci's code-graph-adjacent
APIs: graph ingest/query, code↔memory linking, impact analysis, contracts,
wiring obligations, conflicts, causal inference, and audit/health tools.

For graph architecture, runtime registration, storage boundaries, and deeper
internals, see [docs/CALLGRAPH.md](./CALLGRAPH.md). This page stays at the API
contract level: signatures, parameters, return shapes, and realistic examples.

All tools below return JSON-encoded strings. Success examples show the primary
shape; most tools fail open with a JSON object containing an `error` field.

> Source note: `code_graph_ingest`, `code_graph_query`, `code_memory_relink`,
> `code_memory_map`, `symbol_impact`, `impact_report`, `subsystem_report`, and
> `dead_code_candidates` are registered from `mcp/graph_tools.py` and re-exported
> by `mcp/server.py`. Adjacent related graph tools not expanded here:
> `finding_code_context`, `investigation_code_briefing`, and
> `related_investigations_via_code`.

## Code graph ingestion and query

### `code_graph_ingest`

**Signature**

```python
code_graph_ingest(path: str, max_files: int | None = None, replace: bool = False) -> str
```

**Purpose**

Parses one source file or a directory tree and ingests code structure into the
Ladybug graph so later graph queries, impact analysis, and code↔memory linking
have `CodeFile` / `CodeSymbol` / `CALLS` / `DEFINES` / `IMPORTS` data to work
with. Ingest is additive by default; `replace=True` first prunes existing code
nodes under the same path for idempotent re-ingest.

**Key parameters**

- `path`: File or directory to parse.
- `max_files`: Optional cap when walking a directory.
- `replace`: When `true`, deletes existing code nodes and inbound
  `REFERENCES` under `path` before re-ingesting.

**Return shape**

- Success: `{"path", "parsed_files", "ingested": {files, symbols, defines, calls, imports}}`
- With `replace=True`: adds `"pruned"`
- Failure: `{"error": ...}`

**Example**

```python
code_graph_ingest(path="mcp", max_files=200, replace=True)
```

### `code_graph_query`

**Signature**

```python
code_graph_query(cypher: str, params: dict | None = None) -> str
```

**Purpose**

Runs read-only Cypher against the live Ladybug graph. Use it for direct graph
inspection when you already know the traversal you want: callers of a symbol,
symbols defined in a file, findings that reference code, or ad hoc graph-backed
investigation queries.

**Key parameters**

- `cypher`: Read-only query string.
- `params`: Optional parameter map for `$name`-style placeholders.

**Return shape**

- Success: `{"row_count": int, "rows": [...]}`
- Rejected mutating query: `{"error": "rejected (read-only tool): ..."}`
- Other failures: `{"error": ...}`

**Example**

```python
code_graph_query(
    cypher="MATCH (c:CodeSymbol)-[:CALLS]->(t:CodeSymbol {name:$name}) RETURN c.id, c.file",
    params={"name": "_get_ladybug"},
)
```

## Code-memory correlation and mapping

### `code_memory_correlate`

**Signature**

```python
code_memory_correlate(
    investigation_id: str,
    target_file: str | None = None,
    entity: str | None = None,
) -> str
```

**Purpose**

Finds whether suspicious generated-code entities line up with memories already
stored in an investigation. It can anchor directly on an `entity` string or
scan a generated Python file via `target_file`, then report potentially
contaminated findings and a suggested `memory_retract(...)` follow-up.

**Key parameters**

- `investigation_id`: Investigation whose findings are scanned.
- `target_file`: Optional existing `.py` file to inspect for distinctive or
  hallucinated entities.
- `entity`: Optional direct anchor such as a fake endpoint, host, module, or
  symbol name.

**Return shape**

- Success: `{"investigation_id", "suspected_entities", "source", "code_findings"?, "contaminated_memories", "already_retracted", "count", "suggestion", "advisory": true}`
- May also include `"note"` when file scanning was advisory-only
- Failure: `{"error": ...}`

**Example**

```python
code_memory_correlate(
    investigation_id="pr-incident-2026-09-17",
    entity="http://localhost:8080/internal-api",
)
```

### `code_memory_relink`

**Signature**

```python
code_memory_relink() -> str
```

**Purpose**

Rebuilds `Finding -> CodeSymbol` `REFERENCES` edges for all stored findings.
Run it after a fresh `code_graph_ingest` so older findings gain links to newly
ingested code; new writes auto-link, but historical findings need this pass.

**Key parameters**

- None.

**Return shape**

- Success: `{"findings_scanned": int, "links_created": int}`
- Failure: `{"error": ...}`

**Example**

```python
code_memory_relink()
```

### `code_memory_map`

**Signature**

```python
code_memory_map(anchor: str, anchor_type: str = "auto", hops: int = 1) -> str
```

**Purpose**

Returns the mixed code/memory neighborhood around one anchor as a subgraph.
This is the quickest way to inspect how a symbol, finding, investigation, or
file connects across `CALLS`, `DEFINES`, `REFERENCES`, `IN_INVESTIGATION`, and
other graph edges.

**Key parameters**

- `anchor`: Code symbol id or name, finding id, investigation id, or file path.
- `anchor_type`: `"CodeSymbol"`, `"Finding"`, `"Investigation"`, `"CodeFile"`,
  or `"auto"` to resolve in that order.
- `hops`: Breadth-first radius.

**Return shape**

- Success: `{"matched": {"label", "key"} | null, "nodes": [...], "edges": [...]}`
- Failure: `{"error": ...}`

**Example**

```python
code_memory_map(anchor="mcp/server.py::_get_ladybug", anchor_type="CodeSymbol", hops=2)
```

## Impact analysis

### `symbol_impact`

**Signature**

```python
symbol_impact(symbol: str, hops: int = 3) -> str
```

**Purpose**

Computes a symbol's blast radius across both code and investigation memory. It
walks transitive callers of the target symbol and then gathers findings and
investigations that reference either the symbol itself or those callers.

**Key parameters**

- `symbol`: `CodeSymbol.id` (`file::Qualname`) or a bare symbol name.
- `hops`: Maximum transitive `CALLS` depth.

**Return shape**

- Success: `{"callers": [...], "findings": [...], "investigations": [...]}`
- Failure/unavailable graph: empty success shape or `{"error": ...}` depending on layer

**Example**

```python
symbol_impact(symbol="_get_ladybug", hops=3)
```

### `impact_report`

**Signature**

```python
impact_report(symbol: str, hops: int = 3) -> str
```

**Purpose**

Builds a higher-level change-impact report for a symbol or class. Compared with
`symbol_impact`, it summarizes direct callers, transitive caller count,
referencing-findings count, investigation samples, and the most co-referenced
symbols. For class targets it also folds in the class's methods.

**Key parameters**

- `symbol`: Symbol name or class name.
- `hops`: Caller depth, clamped to `1..6` in the analytics layer.

**Return shape**

- Success: `{"symbol", "resolved", "direct_callers", "transitive_caller_count", "referencing_finding_count", "investigations", "co_referenced", "partial", "queries_failed"}`
- Failure/unavailable graph: same shape with empty collections and zero counts

**Example**

```python
impact_report(symbol="contract_declare", hops=2)
```

### `dead_code_candidates`

**Signature**

```python
dead_code_candidates(lang: str | None = None, limit: int = 50) -> str
```

**Purpose**

Lists likely-dead functions or methods: no graph-detected caller and no
finding reference, while filtering out obvious framework entry points,
properties, private names, dunders, and tests. Treat results as candidates to
verify, especially for languages with sparser call-graph recall.

**Key parameters**

- `lang`: Optional language filter such as `"python"`, `"java"`, or `"go"`.
- `limit`: Maximum candidates to return.

**Return shape**

- Success: `{"candidates": [{"name", "kind", "file", "line", "decorators"}], "count": int, "note": str}`
- Failure/unavailable graph: same shape with empty candidates and explanatory note

**Example**

```python
dead_code_candidates(lang="python", limit=25)
```

## Contracts

### `contract_declare`

**Signature**

```python
contract_declare(
    investigation_id: str,
    entity: str,
    role: str,
    fields: str,
    protocol: str = "",
) -> str
```

**Purpose**

Stores a producer/consumer boundary contract as a gap finding so future agents
can ground against the declared schema before writing integration code. This is
for explicit serialization boundaries such as APIs, queues, topics, schemas,
or file formats.

**Key parameters**

- `investigation_id`: Investigation where the declaration is stored.
- `entity`: Boundary identifier, for example an endpoint or serializer name.
- `role`: Must be `"producer"` or `"consumer"`.
- `fields`: JSON object string, not a native object.
- `protocol`: Optional label such as `"JSON-HTTP"` or `"MQTT"`.

**Return shape**

- Success: `{"stored": true, "finding_id": "...", "entity": "...", "role": "..."}`
- Failure: `{"error": ...}`

**Example**

```python
contract_declare(
    investigation_id="api-drift-audit",
    entity="POST /api/users",
    role="producer",
    fields='{"user_id":"int","created_at":"ISO8601 string"}',
    protocol="JSON-HTTP",
)
```

### `contract_query`

**Signature**

```python
contract_query(investigation_id: str, entity: str, role: str = "") -> str
```

**Purpose**

Looks up stored contract declarations for one entity inside an investigation.
It first scans local findings by exact `entity:<name>` tag, then falls back to
Qdrant semantic search when the local file has no matching declarations.

**Key parameters**

- `investigation_id`: Investigation to search.
- `entity`: Exact entity tag value to retrieve.
- `role`: Optional `"producer"` / `"consumer"` filter.

**Return shape**

- Success: `{"contracts": [...], "count": int}`

**Example**

```python
contract_query(investigation_id="api-drift-audit", entity="POST /api/users")
```

### `contract_check`

**Signature**

```python
contract_check(investigation_id: str, field_name: str, entity: str = "") -> str
```

**Purpose**

Checks whether a new field name looks like rename drift against previously
stored contracts. It uses a near-miss prefix/suffix heuristic to flag cases
like `createdAt` vs `created_at` or shortened/extended variants on the same
entity.

**Key parameters**

- `investigation_id`: Investigation to inspect.
- `field_name`: Proposed field name from new code.
- `entity`: Optional entity scope.

**Return shape**

- Success: `{"conflicts": [{"entity", "your_field", "declared_field", "declared_type", "finding_id"}], "consistent": bool}`
- If no contracts exist: adds `"note": "no contracts stored"`

**Example**

```python
contract_check(
    investigation_id="api-drift-audit",
    entity="POST /api/users",
    field_name="createdAt",
)
```

## Wiring obligations

### `wiring_obligation_scan`

**Signature**

```python
wiring_obligation_scan(content: str, path: str = "", context: str = "") -> str
```

**Purpose**

Performs an advisory-only local-model scan for implicit follow-up obligations in
code, diff hunks, or comments. It never writes investigation state; it only
suggests candidates that may deserve an explicit `wiring_obligation_declare`.

**Key parameters**

- `content`: Code snippet, diff hunk, or file body.
- `path`: Optional label for the scanned source.
- `context`: Extra operator context, such as `"PR diff"`.

**Return shape**

- Success: `{"candidates": [{"description", "evidence_excerpt", "confidence", "suggested_declare_args": {"class_name", "method_name", "expected_effect"}}], "degraded": bool, "error": str | null}`
- Fail-open backend/model issues return an empty candidate list with `degraded=true`

**Example**

```python
wiring_obligation_scan(
    content="class Publisher:\n    def send(self, msg):\n        return acquire_lease(msg)",
    path="notifier.py",
    context="PR diff",
)
```

### `wiring_obligation_declare`

**Signature**

```python
wiring_obligation_declare(
    investigation_id: str,
    class_name: str,
    method_name: str,
    expected_effect: str,
) -> str
```

**Purpose**

Stores an unverified integration promise as a gap finding. Use it when a class
or method clearly implies some external effect that should exist, but you have
not yet verified the implementation.

**Key parameters**

- `investigation_id`: Investigation to store into.
- `class_name`: Responsible class.
- `method_name`: Responsible method.
- `expected_effect`: Human-readable integration promise.

**Return shape**

- Success: `{"stored": true, "finding_id": "...", "obligation": {"class", "method", "expected_effect"}}`
- Failure: `{"error": ...}`

**Example**

```python
wiring_obligation_declare(
    investigation_id="notification-audit",
    class_name="EmailNotifier",
    method_name="deliver",
    expected_effect="enqueue a provider send request and surface delivery errors",
)
```

### `wiring_obligation_list`

**Signature**

```python
wiring_obligation_list(investigation_id: str, resolved: bool = False) -> str
```

**Purpose**

Lists tracked wiring obligations for an investigation. By default it shows only
open obligations; with `resolved=True` it also includes obligations whose latest
record is already resolved.

**Key parameters**

- `investigation_id`: Investigation to read.
- `resolved`: Include resolved obligations too.

**Return shape**

- Success: `{"obligations": [...], "unresolved_count": int}`

**Example**

```python
wiring_obligation_list(investigation_id="notification-audit")
```

### `wiring_obligation_resolve`

**Signature**

```python
wiring_obligation_resolve(investigation_id: str, finding_id: str, evidence: str) -> str
```

**Purpose**

Marks a previously declared wiring obligation as fulfilled by appending a new
observed finding with resolution evidence and higher confidence. It does not
edit the original row in place; it appends a resolved version and updates
manifest counters.

**Key parameters**

- `investigation_id`: Investigation containing the obligation.
- `finding_id`: Open wiring-obligation finding id.
- `evidence`: Verification note, ideally file and line anchored.

**Return shape**

- Success: `{"resolved": true, "finding_id": "..."}`
- Failure: `{"error": ...}`

**Example**

```python
wiring_obligation_resolve(
    investigation_id="notification-audit",
    finding_id="7a4c...",
    evidence="notifier.py:88 calls provider.send(payload) inside deliver()",
)
```

## Conflicts

### `conflict_list`

**Signature**

```python
conflict_list(investigation_id: str) -> str
```

**Purpose**

Lists conflicts automatically detected between findings in an investigation.
These usually come from contradictory observations, opposing assumptions, or
new evidence that appears to negate earlier memory.

**Key parameters**

- `investigation_id`: Investigation identifier.

**Return shape**

- Success: `{"conflicts": [{"id", "finding_id_a", "finding_id_b", "detected_at", "status", "resolution"}], "count": int}`
- Failure: `{"error": ...}`

**Example**

```python
conflict_list(investigation_id="deploy-regression-2026-09-17")
```

### `conflict_resolve`

**Signature**

```python
conflict_resolve(investigation_id: str, conflict_id: str, verdict: str) -> str
```

**Purpose**

Resolves one detected conflict by recording a verdict. Verdicts are explicit:
`a_wins`, `b_wins`, `both_valid`, or `false_positive`.

**Key parameters**

- `investigation_id`: Investigation identifier.
- `conflict_id`: Conflict id from `conflict_list`.
- `verdict`: One of `a_wins`, `b_wins`, `both_valid`, `false_positive`.

**Return shape**

- Success: `{"resolved": true, "conflict_id": "...", "verdict": "..."}`
- Failure: `{"error": ...}`

**Example**

```python
conflict_resolve(
    investigation_id="deploy-regression-2026-09-17",
    conflict_id="c9d1...",
    verdict="both_valid",
)
```

## Causal inference

### `causal_infer`

**Signature**

```python
causal_infer(investigation_id: str, limit: int = 200) -> str
```

**Purpose**

Materializes inferred causal edges for an investigation into
`causal_edges.jsonl`. It combines declared lineage from `derived_from` with a
heuristic or model-backed inference lane so later tooling can distinguish
"never inferred" from "inferred and empty".

**Key parameters**

- `investigation_id`: Investigation to infer over.
- `limit`: Max recent findings to consider.

**Return shape**

- Success: `{"investigation_id", "findings_considered", "edges_written", "status"}`
- Too little data: same shape with `status="too_few_findings"`
- Failure: `{"error": ...}`

**Example**

```python
causal_infer(investigation_id="incident-2026-09-17", limit=100)
```

### `causal_edges_list`

**Signature**

```python
causal_edges_list(investigation_id: str) -> str
```

**Purpose**

Reads back the persisted causal edges for an investigation. Use it after
`causal_infer` or after a consolidation cycle that may have written edges via
its slow path.

**Key parameters**

- `investigation_id`: Investigation whose `causal_edges.jsonl` should be read.

**Return shape**

- Success: `{"edges": [{"id", "source_id", "target_id", "edge_type", "confidence", "inferred_at"}], "count": int}`
- Failure: `{"error": ... , "edges": [], "count": 0}`

**Example**

```python
causal_edges_list(investigation_id="incident-2026-09-17")
```

## Audit, subsystem, and health

### `audit_log`

**Signature**

```python
audit_log(
    tool_name: str,
    inputs_json: str,
    output: str,
    investigation_id: str | None = None,
    embedding_text: str | None = None,
) -> str
```

**Purpose**

Appends a complete tool-call record to the global audit stream and, when an
investigation is supplied, that investigation's local audit log too. It also
mirrors the event into Mnemosyne and Qdrant when available so prior tool output
can be reconstructed without rerunning the original query.

**Key parameters**

- `tool_name`: Tool that ran.
- `inputs_json`: JSON-encoded arguments string.
- `output`: Full tool output; intended to be untruncated.
- `investigation_id`: Optional investigation association.
- `embedding_text`: Optional semantic summary used instead of raw output text.

**Return shape**

- Success: `{"logged": true, "tool": str, "ts": str, "mnemo_stored": bool, "qdrant_indexed": bool, "investigation_logged": bool | null}`
- Busy/failure paths return `{"error": ...}` or a lock-busy payload

**Example**

```python
audit_log(
    tool_name="code_graph_query",
    inputs_json='{"cypher":"MATCH (c:CodeFile) RETURN c.path LIMIT 5"}',
    output='{"row_count":5,"rows":[...]}',
    investigation_id="graph-audit",
    embedding_text="Listed five ingested code files for graph-audit.",
)
```

### `subsystem_report`

**Signature**

```python
subsystem_report(anchor: str, limit: int = 15) -> str
```

**Purpose**

Summarizes a file, directory, or package-prefix subsystem from both the code
boundary and the memory layer. It resolves matching files, counts symbols by
kind, then reports inbound callers, outbound callees, hotspot symbols, and
linked investigations.

**Key parameters**

- `anchor`: Exact file path or path/package prefix.
- `limit`: Max rows in each ranked section.

**Return shape**

- Success: `{"anchor", "files", "symbol_count", "kinds", "inbound_callers", "outbound_callees", "hotspot_symbols", "investigations"}`
- Failure/unavailable graph: same shape with empty collections

**Example**

```python
subsystem_report(anchor="mcp/graph", limit=10)
```

### `loci_health`

**Signature**

```python
loci_health() -> str
```

**Purpose**

Returns a cheap, read-only health snapshot of the running Loci MCP server. It
probes code version, Ladybug state, backend reachability, embed warm state, and
current Qdrant retention/purge settings without taking write locks or failing
hard if one backend is down.

**Key parameters**

- None.

**Return shape**

- Success: `{"code_version", "ladybug", "ladybug_writer_pid"?, "ollama_reachable", "vllm_reachable", "qdrant_reachable", "embed_model", "rerank_model", "warm", "retention_days"?, "purge_active"?, "purge_warning"?}`

**Example**

```python
loci_health()
```
