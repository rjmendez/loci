export const meta = {
  name: 'performance-audit',
  description: 'Hunt for performance anti-patterns: N+1 queries in ORM loops, missing database indexes, synchronous I/O blocking the event loop, unbounded result sets loaded into memory, and hot-path inefficiencies. Performance bugs are invisible at unit-test scale and only emerge under production load.',
  whenToUse: 'Run after adding any new ORM query, database access pattern, API endpoint, or data-processing loop. N+1 bugs and missing indexes are the two most common causes of production latency regressions — they look correct and pass all tests.',
  phases: [
    { title: 'Hunt', detail: '5 performance hunters in parallel — N+1 queries, missing indexes, sync I/O in async paths, unbounded result sets, hot-path inefficiencies' },
    { title: 'Triage', detail: 'Adversarially verify each critical/high finding — is there a batch call or index the hunter missed?' },
    { title: 'Prioritize', detail: 'Rank by: N+1 (multiplies with scale) > missing index (latency cliff) > unbounded result set (OOM risk) > sync I/O (blocks event loop) > hot-path inefficiency' },
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
  : 'Detect languages from file extensions. Adapt grep patterns to match.'

// ── Shared schemas ────────────────────────────────────────────────────────────
const FINDING_SCHEMA = {
  type: 'object',
  required: ['findings', 'summary'],
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'file', 'line', 'severity', 'title', 'detail', 'fix'],
        properties: {
          id:       { type: 'string' },
          file:     { type: 'string' },
          line:     { type: 'number' },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          title:    { type: 'string' },
          detail:   { type: 'string' },
          fix:      { type: 'string' },
          load_multiplier: { type: 'string' },
        },
      },
    },
    summary: { type: 'string' },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  required: ['finding_id', 'confirmed', 'reason'],
  properties: {
    finding_id: { type: 'string' },
    confirmed:  { type: 'boolean' },
    reason:     { type: 'string' },
    mitigation_missed: { type: 'string' },
  },
}

// ── Hunters ───────────────────────────────────────────────────────────────────
const HUNTERS = [
  {
    key: 'n_plus_one_query',
    prompt: `You are a performance-audit hunter specializing in N+1 query detection.

${LANG_NOTE}

Codebase root: ${ROOT}

Hunt for N+1 query patterns — places where a query is executed inside a loop, causing O(N) database round-trips where one batched query would suffice.

Search for:
1. ORM queries inside for/forEach/map loops:
   - Python/SQLAlchemy: \`for item in items:\\n    session.query\` or \`.filter()\` inside a loop
   - TypeScript/Prisma: \`for (const x of xs) { await prisma.model.find\`
   - TypeScript/TypeORM: \`for\` loop containing \`repository.find\` or \`getRepository\`
   - Django ORM: related-object access (\`obj.related_set.all()\`) inside a queryset loop
2. Lazy-loaded relations accessed in a loop without \`include\`/\`select_related\`/\`prefetch_related\`:
   \`grep -rn "\.find\\|\.query\\|\.get\\|repository\\." ${ROOT} --include="*.ts" --include="*.py" | head -30\`
3. \`Promise.all\` over an array where each element makes its own DB call — the correct pattern, but verify the call inside isn't itself an N+1
4. Missing \`include\`/\`eager_load\` on relations that are accessed after the main query

For each finding, estimate the load multiplier: if the loop runs over N items and the inner query is O(1), it's O(N) round-trips that become O(1) with batching. State the multiplier.

Return findings with: file, line, severity (high if in a hot path like API handler; medium if in a background job), title, detail (include the loop variable and the query being repeated), fix (the batch query or eager-load alternative), load_multiplier.`,
  },
  {
    key: 'missing_db_index',
    prompt: `You are a performance-audit hunter specializing in missing database index detection.

${LANG_NOTE}

Codebase root: ${ROOT}

Hunt for columns used in WHERE, ORDER BY, or JOIN clauses that have no corresponding index in the schema/migration files.

Search approach:
1. Find all WHERE filters in ORM queries:
   \`grep -rn "where:\\|filter(\\|WHERE " ${ROOT} --include="*.ts" --include="*.py" --include="*.sql" | grep -v test | head -40\`
2. Find all ORDER BY / orderBy clauses:
   \`grep -rn "orderBy\\|order_by\\|ORDER BY" ${ROOT} --include="*.ts" --include="*.py" --include="*.sql" | grep -v test | head -30\`
3. Find schema/migration files to check for index declarations:
   \`find ${ROOT} -name "schema.prisma" -o -name "*.migration.*" -o -name "*_migration.py" -o -name "models.py" | head -10\`
4. Cross-reference: for each filtered/sorted column, check if it appears in an \`@@index\`, \`CREATE INDEX\`, \`db_index=True\`, or \`index: true\` declaration
5. Pay special attention to:
   - Foreign key columns (often used in JOINs but missing index)
   - Status/type enum columns used as primary filters
   - created_at/updated_at columns used in ORDER BY
   - Compound filters where a composite index would help

Severity: high if the table is expected to grow unboundedly (audit logs, events, findings); medium if the table is small and bounded.

Return findings with: file, line, severity, title (include column and table name), detail (what query uses it, estimated table size if determinable), fix (the index declaration to add).`,
  },
  {
    key: 'sync_io_in_async_path',
    prompt: `You are a performance-audit hunter specializing in synchronous I/O blocking async event loops.

${LANG_NOTE}

Codebase root: ${ROOT}

Hunt for synchronous (blocking) I/O operations called inside async handlers, route handlers, or event-driven code paths where they block the event loop or thread pool.

Search for:
1. Node.js/TypeScript blocking APIs in async functions:
   \`grep -rn "readFileSync\\|writeFileSync\\|execSync\\|spawnSync\\|existsSync\\|readdirSync" ${ROOT} --include="*.ts" --include="*.js" | grep -v test | grep -v "node_modules" | head -30\`
2. Python blocking calls in async functions (asyncio):
   \`grep -rn "def async\\|async def" ${ROOT} --include="*.py" | head -20\`
   Then check if those async functions call \`subprocess.check_output\`, \`open()\` without \`aiofiles\`, \`time.sleep()\`, or \`requests.get()\` (not httpx/aiohttp)
3. \`child_process.execSync\` / \`child_process.spawnSync\` inside route handlers or middleware
4. Image/file processing (sharp, PIL, cv2) called synchronously in a request handler without offloading to a worker
5. \`JSON.parse\` on very large payloads inside the request path (blocks event loop on large inputs)

For each finding, identify:
- Is it on the hot request path (API handler, middleware) → high severity
- Is it in a background job / startup → medium/low severity
- What is the async alternative (fs.promises.readFile, asyncio subprocess, aiofiles, etc.)

Return findings with: file, line, severity, title, detail, fix.`,
  },
  {
    key: 'unbounded_result_set',
    prompt: `You are a performance-audit hunter specializing in unbounded query result sets.

${LANG_NOTE}

Codebase root: ${ROOT}

Hunt for database queries or collection operations that load all matching records into memory with no LIMIT, pagination, or streaming.

Search for:
1. ORM findMany/findAll with no \`take\`/\`limit\`/\`first\`/pagination:
   \`grep -rn "findMany\\|findAll\\|\.all()\\|\.objects\.filter" ${ROOT} --include="*.ts" --include="*.py" | grep -v test | grep -v "node_modules" | head -40\`
   Check each result: does the call have a \`take:\` or \`limit=\` argument? If not, flag it.
2. \`SELECT *\` or \`SELECT col\` with no \`LIMIT\` in raw SQL queries
3. In-memory aggregation over full table results:
   \`grep -rn "\.length\\|\.count\\|\.reduce\\|sum(\\|len(" ${ROOT} --include="*.ts" --include="*.py" | head -20\`
   Cross-check if these operate on the result of an unbounded query
4. Streaming/cursor patterns that are missing: large exports, report generation, or data migration jobs that should use \`cursor()\`/\`stream()\` but call \`findMany\`
5. API endpoints that return full lists without pagination metadata (\`total\`, \`page\`, \`pageSize\`) — implies no server-side limit is enforced

Severity: high if the table is an audit log, findings store, or event log expected to grow without bound; medium if table size is naturally bounded.

Return findings with: file, line, severity, title (include table/model name), detail (explain what happens at 10K/100K rows), fix (add \`take:\`/\`limit=\`, cursor pagination, or streaming).`,
  },
  {
    key: 'hot_path_inefficiency',
    prompt: `You are a performance-audit hunter specializing in algorithmic inefficiencies and redundant computation on hot paths.

${LANG_NOTE}

Codebase root: ${ROOT}

Hunt for O(N²) patterns, redundant recomputation, and expensive operations repeated unnecessarily in hot code paths.

Search for:
1. Nested loops over the same collection (O(N²)):
   Look for \`for x in items: for y in items\` or \`items.forEach(() => items.find/filter/map)\`
   \`grep -rn "\.find(\\|\.filter(\\|\.some(" ${ROOT} --include="*.ts" --include="*.py" | grep -v test | head -20\`
   Check if these are called inside another loop/map over the same or related array
2. String concatenation in a loop (should use join/StringBuilder):
   \`grep -rn "+= \`\\|+= \"\\|+= '" ${ROOT} --include="*.ts" --include="*.py" | grep -v test | head -20\`
3. Expensive repeated computation that should be memoized:
   - \`JSON.parse(JSON.stringify(\` (deep clone in a loop)
   - \`new RegExp(\` constructed inside a loop (should be outside)
   - \`Object.keys(\`/\`Object.entries(\` called repeatedly on the same object inside a loop
4. Array \`.indexOf\`/\`.includes\` inside a loop over a large array (should be a Set):
   \`grep -rn "\.includes(\\|\.indexOf(" ${ROOT} --include="*.ts" --include="*.py" | grep -v test | head -30\`
5. Missing caching on expensive cross-cutting computations called per-request:
   - Config parsing, env var reading with validation, or schema compilation repeated on every request
   - \`grep -rn "JSON\.parse(process\.env\\|dotenv\.config\\|zod.*parse" ${ROOT} --include="*.ts" | grep -v test | head -20\`

Return findings with: file, line, severity (high if in a route handler called at high frequency; medium if in a background job), title, detail (include the complexity and what triggers the hot path), fix.`,
  },
]

// ── Phase 1: Hunt ─────────────────────────────────────────────────────────────
phase('Hunt')
const huntResults = await parallel(HUNTERS.map(h => () =>
  agent(h.prompt, { label: `hunt:${h.key}`, phase: 'Hunt', schema: FINDING_SCHEMA })
))

const allFindings = huntResults
  .filter(Boolean)
  .flatMap((r, i) => (r.findings || []).map(f => ({
    ...f,
    id: `${HUNTERS[i].key}-${f.id || Math.random().toString(36).slice(2, 7)}`,
    hunter: HUNTERS[i].key,
  })))

log(`Hunt complete: ${allFindings.length} raw findings across ${HUNTERS.length} hunters`)

if (!allFindings.length) {
  log('No findings — codebase looks clean for this audit dimension.')
  return { root: ROOT, findings_total: 0, confirmed_high_critical: 0, report: null }
}

// ── Phase 2: Triage ───────────────────────────────────────────────────────────
phase('Triage')
const highCritical = allFindings.filter(f => f.severity === 'critical' || f.severity === 'high')

const triageResults = await parallel(highCritical.map(f => () =>
  agent(`You are an adversarial performance reviewer. Your job is to REFUTE the following finding if possible.

Finding: ${f.title}
File: ${f.file}:${f.line}
Detail: ${f.detail}
Fix suggested: ${f.fix}
${f.load_multiplier ? `Load multiplier: ${f.load_multiplier}` : ''}

Codebase root: ${ROOT}

Try to find evidence that this is NOT actually a performance problem:
- Is there a batch call, eager load, or index the hunter missed?
- Is this code path actually cold (only called once at startup, not per-request)?
- Is there a framework-level cache (Next.js ISR, Django cache middleware, CDN) that prevents the hot path from hitting the DB?
- Is the table provably small and bounded (e.g., a config table with 10 rows)?
- Is the result set already bounded by business logic (e.g., only active users returned, and there are always < 100)?

Read the file carefully before deciding. Default to confirmed=true if you cannot find a clear mitigation.`,
    { label: `triage:${f.id}`, phase: 'Triage', schema: VERDICT_SCHEMA }
  )
))

const confirmed = highCritical.filter((f, i) => {
  const v = triageResults[i]
  return !v || v.confirmed !== false
})
const mediumLow = allFindings.filter(f => f.severity !== 'critical' && f.severity !== 'high')

log(`Triage complete: ${confirmed.length}/${highCritical.length} high/critical confirmed, ${mediumLow.length} medium/low`)

// ── Phase 3: Prioritize ───────────────────────────────────────────────────────
phase('Prioritize')

const SEVERITY_RANK = { critical: 4, high: 3, medium: 2, low: 1 }
const HUNTER_RANK   = { n_plus_one_query: 5, missing_db_index: 4, unbounded_result_set: 3, sync_io_in_async_path: 2, hot_path_inefficiency: 1 }

const ranked = [...confirmed, ...mediumLow].sort((a, b) => {
  const sevDiff = (SEVERITY_RANK[b.severity] || 0) - (SEVERITY_RANK[a.severity] || 0)
  if (sevDiff !== 0) return sevDiff
  return (HUNTER_RANK[b.hunter] || 0) - (HUNTER_RANK[a.hunter] || 0)
})

const report = await agent(`You are a performance engineering lead. Synthesize these audit findings into an actionable report.

Confirmed high/critical findings (${confirmed.length}):
${JSON.stringify(confirmed, null, 2)}

Medium/low findings (${mediumLow.length}):
${JSON.stringify(mediumLow, null, 2)}

Produce:
1. executive_summary: 3-4 sentences. What is the overall performance risk profile? Which findings will hurt first as load scales?
2. pr_bundle: Group findings into logical PRs a developer can pick up independently. For each PR: title, priority (1=highest), addresses (finding IDs), severity, rationale (why this group), instructions (concrete fix steps). Order by: N+1 (highest leverage) > missing index > unbounded result set > sync I/O > hot path.
3. load_test_risk: Which findings will NOT surface in unit tests and require load testing to catch? For each: why it hides, what load test would reveal it.

Be specific: include file paths, line numbers, and exact fix code where possible.`,
    { label: 'prioritize', phase: 'Prioritize' }
  )

// ── Phase 4: Store to Loci ────────────────────────────────────────────────────
if (INV) {
  phase('Store')
  await parallel(ranked.slice(0, 20).map(f => () =>
    agent(`Store this performance finding to the Loci investigation using the mcp__loci__investigation_store tool.

Investigation ID: ${INV}
Finding:
- title: ${f.title}
- file: ${f.file}:${f.line}
- severity: ${f.severity}
- record_type: observed
- confidence: medium
- text: ${f.detail} FIX: ${f.fix}${f.load_multiplier ? ` LOAD MULTIPLIER: ${f.load_multiplier}` : ''}
- tags: performance-audit,${f.hunter},${f.severity}

Call mcp__loci__investigation_store with these values. Return the finding_id from the result.`,
      { label: `store:${f.id}`, phase: 'Store' }
    )
  ))
}

return {
  root: ROOT,
  findings_total: allFindings.length,
  confirmed_high_critical: confirmed.length,
  report,
}
