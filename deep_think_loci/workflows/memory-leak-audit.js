export const meta = {
  name: 'memory-leak-audit',
  description: 'Hunt for resource and memory leaks: unclosed file handles and connections, goroutine/thread leaks, event listener accumulation, Python reference cycles with __del__, unbounded LRU caches, operational accumulators without drain, and two-phase operations missing saga compensation. Leaks are deterministic at small scale and only manifest as OOM, FD exhaustion, or orphaned paid storage under load.',
  whenToUse: 'Run after adding any network connection, file I/O, event listener, thread/goroutine, cache, external resource write, or operational tracking structure. Memory leaks rarely cause test failures — they surface as pod OOM kills, file descriptor exhaustion, orphaned blob storage costs, or DB connection pool timeouts in production.',
  phases: [
    { title: 'Hunt', detail: '7 leak hunters in parallel — unclosed resource, goroutine/thread leak, listener accumulation, reference cycles, unbounded cache, accumulator-without-drain (ML1), saga compensation (DC6)' },
    { title: 'Triage', detail: 'Adversarially verify each critical/high finding — is there a context manager, finally block, or compensation handler the hunter missed?' },
    { title: 'Prioritize', detail: 'Rank by: connection pool exhaustion > file descriptor exhaustion > orphaned paid storage > heap growth > GC pressure' },
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
          severity:       { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          category:       { type: 'string' },
          title:          { type: 'string' },
          file:           { type: 'string' },
          line_hint:      { type: 'string' },
          detail:         { type: 'string' },
          evidence:       { type: 'string' },
          fix_recipe:     { type: 'string' },
          fix_effort:     { type: 'string', enum: ['trivial', 'small', 'medium', 'large'] },
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
    key: 'unclosed_resource',
    prompt: `Hunt for file handles, database connections, and network sockets opened without guaranteed close in ${ROOT}.
${LANG_NOTE}

Mechanism: A resource is opened (file, socket, DB connection) but the close() is either
missing entirely, or only called in the happy path (not in finally/except). Any exception
between open and close leaks the resource. Over time: file descriptor exhaustion or
connection pool depletion.

Steps:
1. Find file opens without context manager:
   grep -rn "open(" ${ROOT} --include="*.py" | grep -v "with open\|test\|#"
   For each: is the result stored as a variable (f = open(...))? Does it have a
   corresponding f.close() in a finally block? If not: leak on exception.

2. Find database connections opened without context manager:
   grep -rn "connect(\|psycopg2\.\|pymysql\.\|sqlite3\.connect\|engine\.connect\|session\s*=" ${ROOT} --include="*.py"
   For each: is there a with statement, or a try/finally with .close()?

3. Find network socket opens:
   grep -rn "socket\.socket\|requests\.Session\|httpx\.Client\|aiohttp\.ClientSession\|urllib\.request\.urlopen" ${ROOT} --include="*.py"
   For each: is there a context manager (with client:) or explicit .close() in finally?

4. For TypeScript/Node.js:
   grep -rn "fs\.open\|createReadStream\|createWriteStream\|net\.createServer\|http\.createServer" ${ROOT} --include="*.ts" --include="*.js"
   For each: is there a .destroy() or .close() in a finally/catch block?

5. Find SQLAlchemy session leaks:
   grep -rn "Session()\|session = " ${ROOT} --include="*.py"
   SQLAlchemy sessions must be .close()'d — else connection returns to pool in an
   unknown state. Correct: session.close() in finally, or use session_scope() context manager.

category="ML-unclosed-resource"`,
  },
  {
    key: 'goroutine_leak',
    prompt: `Hunt for goroutine or thread leaks — background workers with no completion signal or cancellation path in ${ROOT}.
${LANG_NOTE}

Mechanism: A goroutine or thread is started. If the function returns normally, the
goroutine runs forever. If the function exits early (error return, panic), the goroutine
has no way to know and keeps running. Goroutine leaks exhaust the runtime scheduler;
thread leaks exhaust OS thread handles.

Steps:
1. Find goroutines without WaitGroups or error channels (Go):
   grep -rn "go func()\|go \w\+(" ${ROOT} --include="*.go"
   For each: is there a wg.Add(1)/wg.Wait() or error channel pattern?
   A goroutine that writes to a channel where no reader exists → goroutine blocked forever.

2. Find Python threads without join or daemon flag:
   grep -rn "threading\.Thread\|Thread(target=" ${ROOT} --include="*.py"
   For each: is .start() called? Is .join() called before the program could exit?
   Non-daemon threads prevent the process from exiting if they run forever.

3. Find asyncio tasks that run indefinitely without cancellation:
   grep -rn "asyncio\.create_task\|loop\.create_task" ${ROOT} --include="*.py"
   For each: is there a cancel() call path on application shutdown?
   Background tasks that run infinite loops (while True: await asyncio.sleep(N))
   must be cancelled on shutdown or they prevent event loop cleanup.

4. Find executor workers that accumulate:
   grep -rn "ThreadPoolExecutor\|ProcessPoolExecutor" ${ROOT} --include="*.py"
   For each: is the executor used as a context manager (with executor:)? Or is
   executor.shutdown() called on application exit?

5. Find leaked goroutines via HTTP handlers (Go):
   grep -rn "go.*Handle\|go.*Serve\|go.*handler" ${ROOT} --include="*.go"
   HTTP handler goroutines must complete. A handler that blocks on a channel or
   external call indefinitely holds the connection and the goroutine.

category="ML-goroutine-leak"`,
  },
  {
    key: 'listener_accumulation',
    prompt: `Hunt for event listeners that are added repeatedly without removal, causing listener accumulation in ${ROOT}.
${LANG_NOTE}

Mechanism: An event listener is added every time a component mounts, a class is
instantiated, or a request is processed — but never removed. The listener list grows
unboundedly. Node.js: MaxListenersExceededWarning. Python Django signals: the same
handler fires N times for the Nth connection.

Steps:
1. Find JavaScript/TypeScript addEventListener without removeEventListener:
   grep -rn "addEventListener(" ${ROOT} --include="*.ts" --include="*.js"
   For each: is there a corresponding removeEventListener in a cleanup function,
   componentWillUnmount, useEffect cleanup, or finally block?

2. Find Node.js EventEmitter .on() without .off() or .once():
   grep -rn "\.on(\|emitter\.on\|process\.on(" ${ROOT} --include="*.ts" --include="*.js"
   For each: if the .on() is inside a function or request handler, is there a
   corresponding .off() or .removeAllListeners() when the handler is done?

3. Find Django signal connects without disconnect:
   grep -rn "signal\.connect\|post_save\.connect\|pre_delete\.connect\|receiver(" ${ROOT} --include="*.py"
   For each: is there a .disconnect() in teardown? Django test runners may not
   disconnect signals between tests — causing interference.

4. Find Python event libraries without cleanup:
   grep -rn "\.connect(\|signal\.\|blinker\|pyee\|eventemitter" ${ROOT} --include="*.py"
   For each: is the listener attached to a long-lived emitter? Is there a
   corresponding cleanup?

5. Find React useEffect with no cleanup for subscriptions:
   grep -rn "useEffect" ${ROOT} --include="*.tsx" --include="*.jsx"
   For each useEffect that calls .subscribe(), .on(), or .addEventListener:
   does the return value clean up the subscription? Missing return () => unsub()
   causes accumulation on re-renders.

category="ML-listener-accumulation"`,
  },
  {
    key: 'reference_cycle_del',
    prompt: `Hunt for Python reference cycles involving __del__ that prevent garbage collection in ${ROOT}.
${LANG_NOTE}

Mechanism: Python's GC can collect reference cycles IF the objects in the cycle have no
__del__ method. A cycle involving an object with __del__ is put in gc.garbage (uncollectable
list) — it leaks permanently. The cycle detector cannot invoke __del__ in a defined order
without risk.

Steps:
1. Find classes with __del__:
   grep -rn "def __del__" ${ROOT} --include="*.py"
   For each: does this class hold a reference to another object that also has __del__?
   Or does it hold a reference back to its container?

2. Find common cycle patterns:
   - Parent holds list of children; each child has a parent= reference back.
   If either parent or child has __del__: uncollectable cycle.
   - Observer/callback patterns: Subject holds list of observers; Observer holds
   reference to Subject. If either has __del__: leak.

3. Find weakref usage (the fix):
   grep -rn "weakref\.\|import weakref\|WeakValueDictionary\|WeakSet" ${ROOT} --include="*.py"
   Files that use weakref are handling back-references correctly.
   Files that have __del__ but don't use weakref are candidates.

4. Find gc.garbage checks in tests:
   grep -rn "gc\.garbage\|gc\.collect()" ${ROOT} --include="*.py"
   Tests that check gc.garbage are explicitly testing for this leak type.
   If there are __del__ classes but no gc.garbage test: no verification.

category="ML-reference-cycle-del"`,
  },
  {
    key: 'unbounded_cache',
    prompt: `Hunt for unbounded caches that grow without limit in ${ROOT}.
${LANG_NOTE}

Mechanism: @lru_cache with maxsize=None caches every unique call forever. On a class
method decorated with @lru_cache, each instance of the class gets its own separate cache
(the instance is part of the cache key via self) — so N instances = N unbounded caches.
In high-throughput scenarios, the cache fills all available memory.

Steps:
1. Find @lru_cache with no maxsize or maxsize=None:
   grep -rn "@lru_cache\|@functools\.lru_cache" ${ROOT} --include="*.py"
   For each: what is the maxsize? None = unlimited. Missing = defaults to 128 (OK).
   But lru_cache on a method means self is in the key — unbounded instances = unbounded cache.

2. Find @cache (Python 3.9+ — always unbounded):
   grep -rn "@cache\b\|@functools\.cache\b" ${ROOT} --include="*.py"
   @cache is equivalent to @lru_cache(maxsize=None). Any use on a hot path is risky.

3. Find manual dict/list caches with no eviction:
   grep -rn "_cache\s*=\s*{}\|self\._cache\|self\.cache\s*=\s*{}" ${ROOT} --include="*.py"
   For each: is there any eviction logic? max size check? TTL?

4. Find Redis/Memcached usage without TTL:
   grep -rn "\.set(\|cache\.set\|redis\.set\b" ${ROOT} --include="*.py" --include="*.ts"
   For each: does the .set() call include an expiry/TTL argument?
   Redis keys without TTL accumulate until Redis runs out of memory.

5. Find in-memory TypeScript/JavaScript caches:
   grep -rn "new Map()\|new Set()\|Object.create(null)" ${ROOT} --include="*.ts" --include="*.js"
   For each: is there a size limit or eviction? Maps that accumulate request-scoped
   data (keyed by request ID, user ID, etc.) and never get cleared are leaks.

category="ML-unbounded-cache"`,
  },
  {
    key: 'accumulator_without_drain',
    prompt: `Hunt for operational tracking structures that accumulate entries without ever draining them in ${ROOT}.
${LANG_NOTE}

Mechanism (ML1): A plain Map, Set, dict, or list is used as a tracking structure — entries
are added on every event, request, or entity — but no code ever removes entries. Unlike
intentional caches, these structures were not designed to be bounded; the developer never
considered when each entry's purpose is finished. The structure grows monotonically with
the number of distinct keys ever seen since process start.

Sub-variant — drain-values-not-keys: The list/set contents of each entry are pruned
(e.g., removing timestamps older than a TTL window) but the key itself is never deleted
once empty. IP addresses, request IDs, or entity IDs accumulate as empty-list keys forever.

Steps:
1. Find module-level or class-level Maps, dicts, and lists (NOT local variables):
   grep -rn "^[a-z_]\+\s*:\s*dict\s*=\s*{}\|^[a-z_]\+\s*:\s*Dict\|defaultdict(\|^[A-Z_]\+\s*=\s*{}" ${ROOT} --include="*.py" | grep -v test
   grep -rn "^const [a-z_]\+\s*=\s*new Map(\|^const [a-z_]\+\s*=\s*new Set(\|^const [a-z_]\+:\s*Record<" ${ROOT} --include="*.ts" | grep -v test

2. For each structure found, look for the add-site (fires repeatedly in a handler):
   grep -rn "\[client_ip\]\|setdefault(\|\[request_id\]\|\[user_id\]\|\[entity_id\]\|\.set(" ${ROOT} --include="*.py" --include="*.ts" | grep -v test
   Is the add-site inside a function that fires on every request, event, or entity? If yes: accumulator candidate.

3. Check for eviction — is there a delete/pop/clear that removes the KEY (not just the value)?
   grep -rn "\.pop(\|del \w\+\[\\|\.delete(\|\.clear()" ${ROOT} --include="*.py" --include="*.ts" | grep -v test
   A prune like [t for t in attempts if now-t < window] removes VALUES but leaves the KEY.
   After the prune, check: is the key deleted when the result is empty? If not: drain-values-not-keys.

4. Distinguish accumulators from intentional caches:
   - Caches: key is a query/args, value is the computed result, bounded key-space OK
   - Accumulators: key is an entity ID, IP address, or request ID — key-space grows with production traffic

5. Check for size guards or maximum-age eviction:
   grep -rn "len(\w\+)\s*>\|\.size\s*>\|MAX_\|evict\|prune_cold\|oldest" ${ROOT} --include="*.py" --include="*.ts"
   Structures that appear in step 1-2 but NOT here are candidates.

For each finding: report the structure name, where it is declared, what the key represents,
where entries are added, whether keys are ever deleted (not just values), and what the key-space
is in practice (does it grow with production traffic?).

category="ML-accumulator-drain"`,
  },
  {
    key: 'saga_compensation',
    prompt: `Hunt for two-phase operations where an external resource write is not rolled back on DB failure in ${ROOT}.
${LANG_NOTE}

Mechanism (DC6): Step 1 mutates an external resource (blob storage, S3, file system,
external API, email). Step 2 writes a reference to the DB. If the DB write fails, step 1
is not undone — the external resource is permanently orphaned. LLMs generate both steps
reliably and generate the obvious error path (network retry) but not the compensation path
(undo step 1 if step 2 fails), because compensation requires reasoning about partial state.

Recognition signal: A comment like "orphaned blob is harmless" or "cleanup is best-effort"
next to an upload-then-write pattern is an acknowledged-but-dismissed DC6.

Steps:
1. Find external resource mutations (step 1 candidates):
   grep -rn "await put(\|await upload(\|\.upload(\|\.putObject(\|s3\.\|vercelBlob\|@vercel/blob\|\.write(" ${ROOT} --include="*.ts" | grep -v test
   grep -rn "requests\.post\|httpx\.post\|urllib\.request\.urlopen\|smtplib\|sendgrid\|stripe\." ${ROOT} --include="*.py" | grep -v test

2. For each external write, look for a DB write in the same function body:
   grep -rn "prisma\.\|await db\.\|\.create(\|\.update(\|\.upsert(\|INSERT INTO\|cursor\.execute" ${ROOT} --include="*.ts" --include="*.py" | grep -v test

3. For each function that does BOTH an external write AND a DB write, read the full function:
   - Is the DB write inside a try block?
   - If the DB write throws, is the external resource cleaned up in the catch?
   - Pattern to flag: external_write() followed by db.create/update() with no catch that deletes the external resource.

4. Look for the compensation function that already exists (used in the "replace" case):
   grep -rn "del(\|deleteBlob\|deleteObject\|\.delete(\|remove_file" ${ROOT} --include="*.ts" --include="*.py" | grep -v test
   If a delete function exists but is only called for "old resource on update" and not in the
   "new resource on DB failure" path: confirmed DC6.

5. Check for acknowledged gaps:
   grep -rn "orphan\|harmless\|best.effort\|cleanup.*fail\|if.*fail.*delete\|TODO.*clean" ${ROOT} --include="*.ts" --include="*.py"

For each finding: report step1 (what external resource is created), step2 (what DB write follows),
whether a catch block exists for step2, whether the compensation function exists elsewhere in the
codebase (and where), and the consequence of step2 failure.

category="ML-saga-compensation"`,
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
    `Adversarially verify this memory/resource leak finding. Try to REFUTE it.
Default to is_real=false if you cannot confirm by reading the actual code.

Finding ID: ${f.id} | Category: ${f.category}
File: ${f.file || 'unknown'} (${f.line_hint || '?'})
Title: ${f.title}
Detail: ${f.detail}
Evidence: ${f.evidence}

Steps:
1. Read the actual code at the file/line described.
2. Is there a context manager (with statement) or try/finally that the hunter missed?
3. Does the framework guarantee cleanup (Django ORM sessions, FastAPI dependency cleanup)?
4. For caches: is the maxsize present but on a separate decorator line the hunter didn't see?
5. For listeners: is there a cleanup function called from a teardown/shutdown hook?

Return is_real=true only if the resource leak is confirmed unmitigated.`,
    { label: `triage:${f.id}`, phase: 'Triage', schema: VERDICT_SCHEMA }
  ).then(v => v ? { finding: f, verdict: v } : null)
))).filter(Boolean)

const confirmed = verdicts.filter(v => v.verdict.is_real)
log(`Triage: ${confirmed.length}/${highPriority.length} confirmed leaks`)

// ── Phase 3: Prioritize ───────────────────────────────────────────────────────
phase('Prioritize')

const report = await agent(
  `Prioritize these memory/resource leak findings.

Ranking: connection pool exhaustion (blocks all DB requests) > file descriptor exhaustion
       (process can't open files) > heap growth (slow OOM) > GC pressure

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

Produce: executive_summary, pr_bundle (priority order), load_test_risk (which leaks
would only appear under load, not in unit tests — these require explicit load-test coverage).`,
  { label: 'prioritize', phase: 'Prioritize' }
)

if (INV && confirmed.length > 0) {
  await agent(
    `Store confirmed memory/resource leak findings to Loci investigation "${INV}".
For each call mcp__loci__investigation_store(investigation_id="${INV}", finding_type="observed",
confidence="high", tags="memory-leak-audit,category:<category>", text="<title>: <detail> | Fix: <fix_recipe>").
Findings: ${JSON.stringify(confirmed.map(v => v.finding), null, 2)}`,
    { label: 'store:loci', phase: 'Prioritize', model: 'haiku' }
  )
}

return { root: ROOT, findings_total: allFindings.length, confirmed_high_critical: confirmed.length, report }
