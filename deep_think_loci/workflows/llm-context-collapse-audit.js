export const meta = {
  name: 'llm-context-collapse-audit',
  description: 'Hunt for context failures in codebases that use LLMs internally: RAG cross-user bleed, stale embedding caches, context budget overflow, ungrounded generation, and unvalidated LLM output parsing. Patterns CF from the LLM hallucination taxonomy. Only meaningful if the codebase calls LLM APIs or uses embedding/retrieval pipelines.',
  whenToUse: 'Run when the codebase contains LLM API calls, vector database queries, RAG pipelines, or any code that parses LLM output as structured data. Context failures are silent — the model generates plausible-looking wrong output with no exception.',
  phases: [
    { title: 'Hunt', detail: '5 context hunters in parallel — RAG cross-user bleed, stale cache, budget overflow, ungrounded generation, unvalidated output' },
    { title: 'Triage', detail: 'Adversarially verify each finding — is there an isolation filter or validation the hunter missed?' },
    { title: 'Prioritize', detail: 'Rank by: cross-user bleed (security) > unvalidated output (reliability) > stale context > budget overflow' },
  ],
}

// ── Parameters ────────────────────────────────────────────────────────────────
const A    = (typeof args === 'string') ? (() => { try { return JSON.parse(args) || {} } catch (e) { return {} } })() : (args || {})
const ROOT = A.root
if (!ROOT) { log('args.root is required.'); return { error: 'root_required' } }

const LANGS = A.language_stack || []
const INV   = A.loci_investigation || null

const LANG_NOTE = LANGS.length
  ? `The codebase uses: ${LANGS.join(', ')}. Adapt grep extensions and patterns accordingly.`
  : 'Detect languages from file extensions.'

const FINDING_SCHEMA = {
  type: 'object',
  required: ['findings', 'summary'],
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['severity', 'category', 'title', 'detail', 'evidence', 'fix_recipe', 'fix_effort'],
        properties: {
          severity:     { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          category:     { type: 'string' },
          title:        { type: 'string' },
          file:         { type: 'string' },
          line_hint:    { type: 'string' },
          detail:       { type: 'string' },
          evidence:     { type: 'string' },
          fix_recipe:   { type: 'string' },
          fix_effort:   { type: 'string', enum: ['trivial', 'small', 'medium', 'large'] },
        },
      },
    },
    summary: { type: 'string' },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  required: ['finding_id', 'is_real', 'confidence', 'reason'],
  properties: {
    finding_id: { type: 'string' },
    is_real:    { type: 'boolean' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    reason:     { type: 'string' },
    severity:   { type: 'string', enum: ['critical', 'high', 'medium', 'low', 'false-positive'] },
  },
}

// ── Preflight ─────────────────────────────────────────────────────────────────
phase('Hunt')

const llmUsage = await agent(
  `Check if ${ROOT} contains any LLM API calls or embedding/retrieval code.
  grep -rn "openai\|anthropic\|langchain\|llama\|cohere\|qdrant\|pinecone\|chroma\|weaviate\|sentence_transformers\|embed_text\|vector_store\|retrieval" ${ROOT} | head -20
  Return: has_llm_usage (bool), llm_libraries (list of detected libraries).`,
  { label: 'preflight:detect-llm', phase: 'Hunt', schema: {
    type: 'object',
    required: ['has_llm_usage'],
    properties: {
      has_llm_usage: { type: 'boolean' },
      llm_libraries: { type: 'array', items: { type: 'string' } },
    },
  }}
)

if (!llmUsage || !llmUsage.has_llm_usage) {
  log('No LLM/embedding usage detected in this codebase. Audit not applicable.')
  return { root: ROOT, skipped: true, reason: 'No LLM API usage detected' }
}

log(`LLM libraries detected: ${(llmUsage.llm_libraries || []).join(', ')}`)

const HUNTERS = [
  {
    key: 'rag_cross_target_bleed',
    prompt: `Hunt for RAG vector searches with no per-user, per-session, or per-namespace isolation in ${ROOT}.
${LANG_NOTE}

Mechanism (CF1): A vector store contains embeddings from multiple users or contexts.
A search against the store returns chunks from any user, not just the current one.
The model generates a response grounded in another user's private data. This is a
privacy/security vulnerability: one user's documents contaminate another user's results.

Steps:
1. Find vector search calls:
   grep -rn "\.search(\|\.query(\|similarity_search\|nearest_neighbors\|search_vectors" \
     ${ROOT} --include="*.py" --include="*.ts"
   For each search call: is there a filter on user_id, tenant_id, namespace, or session_id?

2. Find Qdrant usage:
   grep -rn "qdrant_client\|QdrantClient\|collection.*search\|points.*search" ${ROOT}
   For each: does the search include a filter clause? Qdrant without a filter searches
   ALL points in the collection — regardless of which user they belong to.

3. Find Pinecone usage:
   grep -rn "pinecone\|index\.query\|index\.fetch" ${ROOT}
   For each query: is a namespace provided (namespace=user_id or similar)?

4. Find Chroma usage:
   grep -rn "chromadb\|collection\.query\|collection\.get" ${ROOT}
   For each: is there a where clause filtering by user/session metadata?

5. Find embedding pipelines that store data without user metadata:
   grep -rn "add_texts\|upsert\|add_documents\|embed_and_store" ${ROOT}
   For each: is user_id or tenant_id included in the metadata payload?
   Documents stored without owner metadata cannot be filtered by owner at search time.

category="CF1-rag-cross-user-bleed"`,
  },
  {
    key: 'stale_context_cache',
    prompt: `Hunt for embedding caches and retrieval indexes that are not invalidated after data changes in ${ROOT}.
${LANG_NOTE}

Mechanism (CF2): Embeddings are computed and cached. The underlying data changes (a
document is updated, a function is renamed, a record is deleted). The cache is not
invalidated. The retrieval system returns chunks that describe the old state.
The model generates code or answers based on the pre-change state.

Steps:
1. Find embedding caches:
   grep -rn "embed.*cache\|cache.*embed\|embedding_cache\|vector_cache" ${ROOT}
   For each: what triggers cache invalidation? On every write? On TTL? Never?

2. Find ETL pipelines that build indexes:
   grep -rn "build_index\|create_index\|index_documents\|embed_and_store\|upsert_embeddings" ${ROOT}
   For each: is this called automatically on data change, or only manually?
   A manual index rebuild that must be remembered after each data change will be forgotten.

3. Find document modification paths without index update:
   grep -rn "\.save(\|\.update(\|PUT\|PATCH\|db\.commit()" ${ROOT} --include="*.py" | grep -v test
   For each: does the save/update trigger a vector store upsert for the changed document?

4. Find Redis or in-memory caches for embeddings:
   grep -rn "redis.*embed\|cache\.set.*embed\|lru_cache.*embed" ${ROOT}
   For each: what is the TTL? None = embeddings cached forever regardless of changes.

5. Find tests that verify cache invalidation:
   grep -rn "invalidat\|cache_clear\|evict\|refresh.*embed\|reindex" ${ROOT}/tests/ 2>/dev/null
   Absence of invalidation tests = unverified assumption that the cache is always fresh.

category="CF2-stale-context-cache"`,
  },
  {
    key: 'context_budget_overflow',
    prompt: `Hunt for LLM prompt assembly code that can exceed the model's context window without bounds in ${ROOT}.
${LANG_NOTE}

Mechanism (CF3): A RAG pipeline assembles a prompt by concatenating retrieved chunks
with no token counting and no size limit. CrossCodeEval shows EM accuracy drops from
21% (oracle context) to 8.82% (no context) — but excessively long context (50K+ tokens)
also degrades accuracy below the oracle level. Silent degradation: no exception, just
wrong answers.

Steps:
1. Find prompt assembly without token counting:
   grep -rn "f\".*context\|f\".*chunks\|f\".*documents\|prompt.*+=\|messages\.append" \
     ${ROOT} --include="*.py" --include="*.ts"
   For each: is there a token counting call before assembly?
   grep -rn "num_tokens\|count_tokens\|token_count\|tiktoken\|len.*tokens" ${ROOT}

2. Find retrieved chunks concatenated without truncation:
   grep -rn "docs\|chunks\|results\|hits" ${ROOT}
   Patterns: "\n\n".join(chunk.text for chunk in docs) without any slicing or token limit.
   If there is no max_chunks or max_tokens parameter: overflow is possible.

3. Find context window constants:
   grep -rn "MAX_TOKENS\|max_tokens\|context_window\|4096\|8192\|16384\|32768\|100000\|128000" ${ROOT}
   For each: is this constant used to bound the assembled context?

4. Find model API calls with no max_tokens parameter:
   grep -rn "client\.completions\.create\|client\.messages\.create\|openai\.ChatCompletion" ${ROOT}
   For each: is max_tokens set? Missing max_tokens = model decides, may use entire context.

5. Find prompt templates that include variable-length content without bounds:
   grep -rn "system_prompt\|user_message\|SYSTEM_TEMPLATE" ${ROOT}
   For each: can the content grow unboundedly based on retrieved data?

category="CF3-context-budget-overflow"`,
  },
  {
    key: 'ungrounded_generation',
    prompt: `Hunt for LLM calls that generate domain-specific facts without any retrieval or grounding step in ${ROOT}.
${LANG_NOTE}

Mechanism: An LLM is asked to produce content that requires current, specific, or
domain-specific knowledge (code references, database field names, API schemas, product
information, user-specific data). The model responds from training data, which is
outdated or wrong for this specific codebase. The hallucinated output is used directly.

Steps:
1. Find LLM calls with no preceding retrieval:
   grep -rn "anthropic\.\|openai\.\|client\.chat\|client\.complete\|model\.generate" ${ROOT} --include="*.py" --include="*.ts"
   For each: is there a vector search or database lookup in the same function before
   the LLM call? If not: the model is generating from training data only.

2. Find prompts that ask for code or schema information:
   grep -rn "\".*class\|function\|schema\|field\|column\|endpoint\|API\|route\"" ${ROOT} | grep -i "prompt\|message"
   If a prompt asks the model about your codebase's specific classes/fields/endpoints
   without providing them as context: hallucination is likely.

3. Find tool use that could be replaced by RAG:
   grep -rn "\.search(\|web_search\|search_results\|google\|bing" ${ROOT}
   Web search for facts about your own codebase is usually wrong — your codebase is
   not indexed. Should use local vector search instead.

4. Find generation without any source citation:
   grep -rn "model\.generate\|complete\|chat\|llm\.invoke" ${ROOT}
   For each: does the response include source citations? Generation without provenance
   cannot be verified and may be hallucinated.

category="CF-ungrounded-generation"`,
  },
  {
    key: 'llm_output_unchecked',
    prompt: `Hunt for LLM output that is parsed as structured data without schema validation in ${ROOT}.
${LANG_NOTE}

Mechanism: An LLM is asked to return JSON. The response is parsed with json.loads()
or JSON.parse(). If the model returns malformed JSON or a different schema than expected,
the parse fails with an exception — or worse, succeeds with the wrong schema and the
code uses the wrong field names silently.

Steps:
1. Find json.loads / JSON.parse on LLM output:
   grep -rn "json\.loads\|JSON\.parse" ${ROOT} --include="*.py" --include="*.ts"
   For each: is the result validated against a schema before use?
   grep -rn "jsonschema\.\|validate(\|pydantic.*model_validate\|TypeAdapter\|zod\.\|ajv\." ${ROOT}

2. Find LLM responses used as structured data without validation:
   grep -rn "response\.\(text\|content\|message\)\|completion\.\(text\|content\)" ${ROOT}
   For each: is the content parsed and used directly, or is it validated first?

3. Find prompt templates that request specific JSON schemas:
   grep -rn '"format.*json\|"output.*json\|respond.*json\|json_mode\|response_format.*json_object"' ${ROOT}
   For each: is the returned JSON validated against the expected schema?

4. Find field access on parsed LLM output without key existence check:
   grep -rn "parsed\[\"[^\"]\+\"\]\|result\[\"[^\"]\+\"\]\|output\.\w\+" ${ROOT}
   Direct key access on LLM output without checking key existence: KeyError when model
   omits a field.

5. Find retry logic for failed parses:
   grep -rn "json.*error\|parse.*error\|retry.*llm\|retry.*parse" ${ROOT}
   Good pattern: retry with a corrective prompt on parse failure.
   Absence of retry = first parse failure propagates as an unhandled exception.

category="CF4-llm-output-unchecked"`,
  },
]

const rawHunts = await parallel(HUNTERS.map(h => () =>
  agent(h.prompt, { label: `hunt:${h.key}`, phase: 'Hunt', schema: FINDING_SCHEMA })
    .then(r => r ? { ...r, hunter: h.key } : null)
))

const hunts       = rawHunts.filter(Boolean)
const allFindings = hunts.flatMap((h) =>
  (h.findings || []).map((f, j) => ({ ...f, id: `${h.hunter}-${j}` }))
)

log(`Hunt: ${allFindings.length} findings from ${hunts.length}/${HUNTERS.length} hunters`)

// ── Phase 2: Triage ───────────────────────────────────────────────────────────
phase('Triage')

const highPriority = allFindings.filter(f => f.severity === 'critical' || f.severity === 'high')

const verdicts = (await parallel(highPriority.map(f => () =>
  agent(
    `Adversarially verify this LLM context failure finding. Try to REFUTE it.
Default to is_real=false if you cannot confirm by reading the actual code.

Finding ID: ${f.id} | Category: ${f.category}
File: ${f.file || 'unknown'} (${f.line_hint || '?'})
Title: ${f.title}
Detail: ${f.detail}
Evidence: ${f.evidence}

Steps:
1. For RAG bleed: is there a filter applied to the vector search that the hunter missed?
2. For stale cache: does the write path trigger a cache invalidation that's in a different file?
3. For context overflow: is there token counting code in a utility module?
4. For ungrounded generation: is there a retrieval step in a parent function or middleware?
5. For output validation: is there schema validation in a response parser class?

Return is_real=true only if the context failure is confirmed and unmitigated.`,
    { label: `triage:${f.id}`, phase: 'Triage', schema: VERDICT_SCHEMA }
  ).then(v => v ? { finding: f, verdict: v } : null)
))).filter(Boolean)

const confirmed = verdicts.filter(v => v.verdict.is_real)
log(`Triage: ${confirmed.length}/${highPriority.length} confirmed context failures`)

// ── Phase 3: Prioritize ───────────────────────────────────────────────────────
phase('Prioritize')

const report = await agent(
  `Prioritize these LLM context failure findings.

Ranking: cross-user RAG bleed (privacy/security) > unvalidated output (crashes in production)
       > ungrounded generation (wrong answers) > stale cache (outdated answers) > context overflow (degraded answers)

CONFIRMED HIGH/CRITICAL (${confirmed.length}):
${JSON.stringify(confirmed.map(v => ({
  id: v.finding.id, category: v.finding.category, title: v.finding.title,
  detail: v.finding.detail, fix_recipe: v.finding.fix_recipe, fix_effort: v.finding.fix_effort,
  severity: v.verdict.severity || v.finding.severity,
})), null, 2)}

MEDIUM/LOW (${allFindings.filter(f => f.severity === 'medium' || f.severity === 'low').length}):
${JSON.stringify(allFindings.filter(f => f.severity === 'medium' || f.severity === 'low').map(
  f => ({ id: f.id, category: f.category, title: f.title })
), null, 2)}

Produce: executive_summary, pr_bundle (priority order), silent_failure_risk (findings
where the system generates wrong output with no visible error — users would not know
the system is failing).`,
  { label: 'prioritize', phase: 'Prioritize' }
)

if (INV && confirmed.length > 0) {
  await agent(
    `Store confirmed LLM context failure findings to Loci investigation "${INV}".
For each call mcp__loci__investigation_store(investigation_id="${INV}", finding_type="observed",
confidence="high", tags="llm-context-collapse-audit,category:<category>", text="<title>: <detail> | Fix: <fix_recipe>").
Findings: ${JSON.stringify(confirmed.map(v => v.finding), null, 2)}`,
    { label: 'store:loci', phase: 'Prioritize', model: 'haiku' }
  )
}

return { root: ROOT, findings_total: allFindings.length, confirmed_high_critical: confirmed.length, report }
