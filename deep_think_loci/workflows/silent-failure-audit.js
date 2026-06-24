export const meta = {
  name: 'silent-failure-audit',
  description: 'Hunt for code that accepts bad input or encounters failures and silently returns wrong results: swallowed exceptions, retry-on-permanent-error, observability gaps, metastable failure loops, and test stimuli absorbed by dedup gates. Patterns SS, OG, MF from the LLM hallucination taxonomy.',
  whenToUse: 'Run after adding error handlers, retry logic, dedup/rate-limit gates, or background workers. Silent failures are the hardest class of production bug: no exception, no alert, wrong behavior.',
  phases: [
    { title: 'Hunt', detail: '5 silence hunters in parallel — swallowed exceptions, retry logic, observability gaps, metastable loops, test stimulus absorption' },
    { title: 'Triage', detail: 'Adversarially verify each critical/high finding by reading actual handler bodies' },
    { title: 'Prioritize', detail: 'Rank by debuggability: no-evidence failure > wrong-metric > wrong-result' },
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
          severity:   { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          category:   { type: 'string' },
          title:      { type: 'string' },
          file:       { type: 'string' },
          line_hint:  { type: 'string' },
          detail:     { type: 'string' },
          evidence:   { type: 'string' },
          fix_recipe: { type: 'string' },
          fix_effort: { type: 'string', enum: ['trivial', 'small', 'medium', 'large'] },
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

// ── Phase 1: Hunt ─────────────────────────────────────────────────────────────
phase('Hunt')

const HUNTERS = [
  {
    key: 'swallowed_exceptions',
    prompt: `Hunt for swallowed exceptions in ${ROOT} — catch blocks that hide failures.
${LANG_NOTE}

Mechanism (SS1/OG1): An exception is caught. The handler only logs (or doesn't even log).
The function returns None, [], False, or a conservative default. The caller sees a
"successful" return and continues. The error vanishes.

Steps:
1. Find bare exception handlers:
   grep -rn "except Exception\|except:\|catch.*Error\|catch (e)\|recover()" ${ROOT}

2. For each handler, read the body:
   a) Does it re-raise after logging? → OK
   b) Does it return a sentinel that callers will check? → OK if documented
   c) Does it ONLY log and return None/[]/False? → PROBLEM — error evidence lost
   d) Is the body just "pass" or empty? → CRITICAL — completely silent

3. Find try/except wrapping broad scopes (entire function in try block):
   grep -rn "try:" ${ROOT} | head -50
   Read each: does the try cover too much? Does the handler distinguish error types?

4. Find error paths that don't log at all:
   Functions that return None or False without any log call in the failure branch.
   grep -rn "return None\|return False\|return \[\]\|return {}" ${ROOT}
   For each: is there a log call in the same code path before the return?

5. Find logger calls inside catch blocks that don't include the exception:
   "logger.error('something failed')" with no exc_info=True or e argument → no stack trace in logs

category="SS-swallowed-exception"`,
  },
  {
    key: 'retry_on_permanent',
    prompt: `Hunt for retry logic that retries permanent errors in ${ROOT}.
${LANG_NOTE}

Mechanism (SS/AC): A retry decorator or loop retries ALL exceptions. But some errors are
permanent (404 Not Found, 401 Unauthorized, 400 Bad Request). Retrying them wastes time,
exhausts retry budgets, and obscures the root cause. Worst case: infinite retry loop.

Steps:
1. Find retry logic:
   grep -rn "retry\|backoff\|@retry\|tenacity\|retries\|MAX_RETRY\|RETRY_" ${ROOT}
   Also: while True loops with sleep() that catch exceptions.

2. For each retry site, read the exception filter:
   - Does it retry ALL exceptions? → flag
   - Does it check HTTP status codes before retrying?
     404 / 400 / 401 / 403 should NOT be retried — they're permanent.
     429 / 503 / 500 MAY be retried with backoff.
   - Does it have a max retry count? Or could it loop forever?

3. Find background workers with infinite loops and exception handlers:
   grep -rn "while True:\|while running:\|loop.run_forever" ${ROOT}
   For each loop: what happens when an exception escapes to the outer catch?
   Does it sleep and retry, or does it exit?

4. Find celery/sidekiq/rq task definitions:
   grep -rn "@app.task\|@celery.task\|@shared_task" ${ROOT}
   Do they have max_retries set? Do they check exc type before self.retry()?

category="SS-retry-permanent"`,
  },
  {
    key: 'observability_gaps',
    prompt: `Hunt for observability gaps in ${ROOT} — errors and state changes that leave no evidence.
${LANG_NOTE}

Mechanism (OG patterns): A bug occurs. There is no log entry, no metric increment, no trace span.
Root cause can only be determined by inference. In production, this means hours of debugging.

Steps:
1. Find error paths with no logging:
   grep -rn "return None\|return False\|return {}\|return \[\]" ${ROOT}
   For each return of an empty/null sentinel in a non-trivial function:
   Is there a log call immediately before it? If not: OG gap.

2. Find background threads/async tasks with no error logging:
   grep -rn "threading.Thread\|asyncio.create_task\|executor.submit\|go func()" ${ROOT}
   For each: does the task body have a top-level try/except that logs errors?
   An unhandled exception in a background thread is silently swallowed in most frameworks.

3. Find state transitions with no audit trail:
   grep -rn "status = \|state = \|\.status =\|\.state =" ${ROOT}
   For important state fields: is there a log line that records the old and new value?
   "Changed status from X to Y" is invaluable in production debugging.

4. Find metric counters that should exist but don't:
   For each major code path that can fail (DB call, external HTTP call, message publish):
   Is there an error counter that increments on failure? Or only a success counter?
   "requests_total" without "errors_total" means 0 is ambiguous.

5. Find correlation ID propagation gaps:
   grep -rn "request_id\|trace_id\|correlation_id\|X-Request-ID" ${ROOT}
   Is the ID propagated into log messages? Into downstream calls? Into DB records?

category="OG-observability-gap"`,
  },
  {
    key: 'metastable_loops',
    prompt: `Hunt for metastable failure patterns in ${ROOT} — where a trigger resolves but a
sustaining feedback loop keeps the system stuck.
${LANG_NOTE}

Mechanism (MF patterns): A brief overload causes clients to retry aggressively. The retries
increase load. The system never recovers even after the original trigger resolves.

Steps:
1. Find retry without jitter:
   grep -rn "sleep.*RETRY\|time.sleep\|asyncio.sleep\|time.Sleep" ${ROOT}
   For each: is the sleep duration fixed (constant backoff)?
   Fixed-interval retry causes synchronized thundering herd. Should use exponential + jitter.

2. Find circuit breakers:
   grep -rn "circuit_breaker\|CircuitBreaker\|circuit.open\|CIRCUIT_" ${ROOT}
   Are there circuit breakers on external calls? If not: any external failure can cascade.

3. Find queue/buffer depth checks:
   grep -rn "queue.put\|channel.send\|\.put(\|\.push(" ${ROOT}
   For each: is there a depth check before adding? What happens when the queue is full?
   A caller that blocks on a full queue while holding a lock → deadlock.

4. Find connection pool exhaustion paths:
   grep -rn "pool\|connection.*pool\|max_connections\|pool_size" ${ROOT}
   Is there a timeout on pool.acquire()? What error is raised if all connections are busy?
   Is that error retried? (Retrying pool exhaustion increases pool demand → metastable)

5. Find cache invalidation thundering herds:
   A cache key expires. Multiple concurrent requests all miss and all go to DB simultaneously.
   grep -rn "cache.get\|cache.set\|\.get_or_set\|cache_key" ${ROOT}
   Is there a lock/semaphore on the cache-miss path? Or does every miss hit the DB?

category="MF-metastable"`,
  },
  {
    key: 'test_stimulus_absorption',
    prompt: `Hunt for dedup/rate-limit/idempotency gates in ${ROOT} that silently absorb test stimuli,
causing tests to pass without exercising the real behavior.
${LANG_NOTE}

Mechanism (SS2/TEC): A dedup filter is added to a pipeline. Tests that submit identical
payloads to fill a queue silently see only 1 item arrive (9 were deduped). Test assertions
count items and pass — because no assertion verified that dedup was actually intended.

Steps:
1. Find dedup/idempotency gates:
   grep -rn "seen\|dedup\|idempotent\|unique_id\|event_id\|message_id\|deduplicate" ${ROOT}
   For each: what is the scope of dedup? (in-memory dict, Redis, DB unique constraint)
   What TTL does it have? What is the key?

2. Find test files that submit repeated identical payloads:
   grep -rn "for.*range\|for.*in range\|for i in" ${ROOT}/tests/ ${ROOT}/test_* 2>/dev/null | head -20
   For each loop that submits to a pipeline: are the submitted items identical?
   If yes: dedup could silently drop all but one.

3. Find rate limiters with similar issues:
   grep -rn "rate_limit\|throttle\|RateLimit\|rate_limiter" ${ROOT}
   Tests that run fast (no real time passing) hit rate limits immediately.
   Does the test mock time? Or does it submit at a rate that bypasses the limit accidentally?

4. Find idempotency keys used in tests:
   grep -rn "idempotency_key\|request_id.*test\|\"test-id\"\|\"request-1\"" ${ROOT}/tests/ 2>/dev/null
   Hardcoded idempotency keys across test cases → second run of a test uses a "seen" key
   and the operation is silently no-opped.

category="TEC-stimulus-absorption"`,
  },
]

const rawHunts = await parallel(HUNTERS.map(h => () =>
  agent(h.prompt, { label: `hunt:${h.key}`, phase: 'Hunt', schema: FINDING_SCHEMA })
    .then(r => r ? { ...r, hunter: h.key } : null)
))

const hunts       = rawHunts.filter(Boolean)
const allFindings = hunts.flatMap((h, i) =>
  (h.findings || []).map((f, j) => ({ ...f, id: `${h.hunter}-${j}` }))
)

log(`Hunt: ${allFindings.length} findings from ${hunts.length}/${HUNTERS.length} hunters`)

// ── Phase 2: Triage ───────────────────────────────────────────────────────────
phase('Triage')

const highPriority = allFindings.filter(f => f.severity === 'critical' || f.severity === 'high')

const verdicts = (await parallel(highPriority.map(f => () =>
  agent(
    `Adversarially verify this silent-failure finding. Try to REFUTE it.

Finding ID: ${f.id} | Category: ${f.category}
File: ${f.file || 'unknown'} (${f.line_hint || '?'})
Title: ${f.title}
Detail: ${f.detail}
Evidence: ${f.evidence}

Steps:
1. Read the actual code at the location described.
2. Is there a log call with exc_info that the hunter missed?
3. Is there a framework-level error handler that catches and logs above this point?
4. Is the retry logic actually guarded by error type even if not visible in this snippet?
5. Is the metastable concern mitigated by infrastructure (e.g., circuit breaker at load balancer level)?

Return is_real=true only if the failure mode is confirmed unmitigated.`,
    { label: `triage:${f.id}`, phase: 'Triage', schema: VERDICT_SCHEMA }
  ).then(v => v ? { finding: f, verdict: v } : null)
))).filter(Boolean)

const confirmed = verdicts.filter(v => v.verdict.is_real)
log(`Triage: ${confirmed.length}/${highPriority.length} confirmed silent failures`)

// ── Phase 3: Prioritize ───────────────────────────────────────────────────────
phase('Prioritize')

const report = await agent(
  `Prioritize these silent-failure findings.

Ranking: no-evidence failure (impossible to debug) > wrong metric (misleads on-call) > wrong result

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

Produce: executive_summary, pr_bundle (priority order), on_call_nightmare (findings that would make
production incident investigation impossible — no logs, no metrics, no trace).`,
  { label: 'prioritize', phase: 'Prioritize' }
)

if (INV && confirmed.length > 0) {
  await agent(
    `Store confirmed silent-failure findings to Loci investigation "${INV}".
For each call mcp__loci__investigation_store(investigation_id="${INV}", finding_type="observed",
confidence="high", tags="silent-failure-audit,category:<category>", text="<title>: <detail> | Fix: <fix_recipe>").
Findings: ${JSON.stringify(confirmed.map(v => v.finding), null, 2)}`,
    { label: 'store:loci', phase: 'Prioritize', model: 'haiku' }
  )
}

return { root: ROOT, findings_total: allFindings.length, confirmed_high_critical: confirmed.length, report }
