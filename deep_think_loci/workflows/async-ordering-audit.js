export const meta = {
  name: 'async-ordering-audit',
  description: 'Hunt for async/concurrency bugs: missing await on coroutines, callbacks registered before setup, shared mutable state without locks, race conditions in tests, and fire-and-forget tasks that swallow errors. Patterns AC, CP from the LLM hallucination taxonomy.',
  whenToUse: 'Run after adding any async function, background task, event emitter/listener, or shared mutable state. Async bugs are deterministic at small scale and non-deterministic under load — catch them before the load test.',
  phases: [
    { title: 'Hunt', detail: '5 async hunters in parallel — missing await, callback inversion, shared mutable state, test races, fire-and-forget loss' },
    { title: 'Triage', detail: 'Adversarially verify each critical/high finding — is there a lock or await the hunter missed?' },
    { title: 'Prioritize', detail: 'Rank by: data corruption > silent drop > ordering violation > performance degradation' },
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
          severity:    { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          category:    { type: 'string' },
          title:       { type: 'string' },
          file:        { type: 'string' },
          line_hint:   { type: 'string' },
          detail:      { type: 'string' },
          evidence:    { type: 'string' },
          fix_recipe:  { type: 'string' },
          fix_effort:  { type: 'string', enum: ['trivial', 'small', 'medium', 'large'] },
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
    key: 'missing_await',
    prompt: `Hunt for coroutine calls missing 'await' in ${ROOT}.
${LANG_NOTE}

Mechanism (AC5): An async function is called without await. Python creates a coroutine
object and discards it — no exception unless RuntimeWarning is enabled. The operation
silently does nothing. The caller proceeds as if it succeeded.

Steps:
1. Find all async function definitions:
   grep -rn "async def " ${ROOT} --include="*.py" | awk '{print $3}' | sed 's/(.*//' | sort -u
   Also for TypeScript/Go: grep -rn "async function\|async\s*(" ${ROOT} --include="*.ts"

2. For each async function name, find call sites WITHOUT 'await':
   grep -rn "self\.\(flush\|send\|publish\|commit\|save\|close\|write\|push\)(" ${ROOT} --include="*.py" | grep -v "await \|async def \|#"
   Also: grep -rn "\.\(flush\|send\|commit\|save\)();" ${ROOT} --include="*.ts" | grep -v "await "

3. Focus on action methods in class bodies (flush, send, publish, commit, save, close,
   write, push, sync). These are most dangerous when not awaited — the state mutation
   never happens.

4. Also check: asyncio.run() called inside async context, loop.run_until_complete()
   inside async code.

category="AC5-missing-await"`,
  },
  {
    key: 'callback_inversion',
    prompt: `Hunt for callbacks or listeners registered before setup is complete in ${ROOT}.
${LANG_NOTE}

Mechanism: A listener is registered on an event emitter before the emitter is configured
or connected. The first event fires during setup; the listener's context is incomplete.
Or: an event is emitted before any listener has registered for it (producer fires before
consumer is ready).

Steps:
1. Find event emitter patterns:
   grep -rn "EventEmitter\|event_emitter\|\.emit(\|\.trigger(\|signal\.send\|publish(" ${ROOT}
   Note the order: does the emit happen before or after subscribe/on/listen?

2. Find listener registrations:
   grep -rn "addEventListener\|\.on(\|add_listener\|subscribe(\|@receiver\|connect(" ${ROOT}
   For each: is the emitter already initialized when this runs?

3. Find async setup sequences with incorrect ordering:
   grep -rn "await.*connect\|await.*init\|await.*setup\|await.*start" ${ROOT}
   For each: does the code register callbacks BEFORE awaiting the connection?
   Pattern: client.on("message", handler); await client.connect()  ← usually OK
   Danger:  await client.connect(); ... emit() ... client.on("message", handler)  ← too late

4. Check test setup: does the test register event handlers before the system under
   test is started? A handler registered after start() may miss the first events.

category="CP-callback-inversion"`,
  },
  {
    key: 'shared_mutable_state',
    prompt: `Hunt for shared mutable state across async tasks or threads without synchronization in ${ROOT}.
${LANG_NOTE}

Mechanism (AC6): Two async tasks share a mutable object (dict, list, counter, cache).
Code that does 'await' between a read and a write creates a window for another task to
see or modify the shared state. LLMs generate each task independently and omit the lock
because each task looks safe in isolation.

Steps:
1. Find class-level mutable fields:
   grep -rn "self\.\w\+ = {}\|self\.\w\+ = \[\]\|self\.\w\+ = set()" ${ROOT} --include="*.py"
   Also: grep -rn "private.*Map\|private.*List\|private.*Set\|private.*\[\]" ${ROOT} --include="*.ts" --include="*.java"

2. For each mutable field, find the async methods that access it:
   Look for 'await' calls INSIDE the same method that reads/writes the field.
   A read then an await then a write on the same field is a race window.

3. Check for lock usage:
   grep -rn "asyncio\.Lock\|threading\.Lock\|asyncio\.Semaphore\|async with.*lock\|Lock()" ${ROOT}
   Classes with mutable fields but no lock in the same file are candidates.

4. Find module-level mutable globals used from async functions:
   grep -rn "^[A-Z_]\+\s*=\s*{}\|^[A-Z_]\+\s*=\s*\[\]" ${ROOT} --include="*.py"
   For each: is it accessed inside any async function? No lock = race.

category="AC6-shared-mutable-state"`,
  },
  {
    key: 'test_race_condition',
    prompt: `Hunt for race conditions in test setup that cause intermittent failures in ${ROOT}.
${LANG_NOTE}

Mechanism: Test setup code starts async operations but doesn't await their completion
before the test body runs. The test passes when the race is won, fails when it's lost.
Results: flaky tests that are hard to reproduce.

Steps:
1. Find asyncio.create_task inside test setup:
   grep -rn "create_task\|asyncio\.ensure_future\|loop\.create_task" ${ROOT}/tests/ ${ROOT}/test_* 2>/dev/null
   For each: is the task awaited before the test assertion runs?

2. Find test setup that calls async methods without await:
   grep -rn "def setUp\|def setup_method\|def setup\b" ${ROOT}/tests/ 2>/dev/null
   For each setUp: is it async? If not, how are async setup steps being called?
   (Often via loop.run_until_complete() which is correct — but sometimes omitted)

3. Find time.sleep() in async tests (should be asyncio.sleep()):
   grep -rn "time\.sleep\|import time" ${ROOT}/tests/ 2>/dev/null
   In async tests: time.sleep() blocks the event loop, causing false timeouts.

4. Find @pytest.mark.asyncio tests that use asyncio.run() inside the test body:
   asyncio.run() creates a new event loop; any state set up in the test's event loop
   is invisible to the new loop. This causes "fixture not found" or "task not running"
   errors that are hard to diagnose.

category="CP-test-race"`,
  },
  {
    key: 'fire_and_forget_loss',
    prompt: `Hunt for fire-and-forget async tasks whose errors are silently swallowed in ${ROOT}.
${LANG_NOTE}

Mechanism (AC4): asyncio.create_task() or equivalent creates a background task.
If the task raises an exception, the exception prints to stderr and disappears.
The calling code sees no error. The operation silently did not complete.

Steps:
1. Find fire-and-forget task creation without done callback:
   grep -rn "asyncio\.create_task\|loop\.create_task\|asyncio\.ensure_future" ${ROOT} --include="*.py"
   For each: is there a .add_done_callback() attached? If not: errors silently vanish.

2. Find threading.Thread starts without join or result check:
   grep -rn "threading\.Thread\|Thread(target=" ${ROOT} --include="*.py"
   For each: is the thread joined before the calling code proceeds? Is there a
   try/except or result queue in the thread's target function?

3. Find Go goroutines without error channels (if Go is in the stack):
   grep -rn "go func()\|go \w\+(" ${ROOT} --include="*.go"
   For each: does the goroutine have an error channel or WaitGroup?

4. Find executor.submit() without checking the future:
   grep -rn "executor\.submit\|ThreadPoolExecutor\|ProcessPoolExecutor" ${ROOT} --include="*.py"
   For each: is future.result() called? Or is the future stored and awaited later?

5. Also check celery/background tasks:
   grep -rn "@app\.task\|@celery\.task\|\.delay(\|\.apply_async(" ${ROOT}
   Does the caller check the AsyncResult? Or is it truly fire-and-forget with no
   error visibility?

category="AC4-fire-and-forget"`,
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
    `Adversarially verify this async/concurrency finding. Try to REFUTE it.
Default to is_real=false if you cannot confirm by reading the actual code.

Finding ID: ${f.id} | Category: ${f.category}
File: ${f.file || 'unknown'} (${f.line_hint || '?'})
Title: ${f.title}
Detail: ${f.detail}
Evidence: ${f.evidence}

Steps:
1. Read the actual code at the file/line described.
2. Is there an 'await' the hunter missed? (maybe on a different line)
3. Is there a lock protecting the shared state that's not in the same file?
4. Is the task result checked elsewhere (not in the immediate caller)?
5. Does a framework layer (Django, FastAPI, Celery) handle the async lifecycle transparently?

Return is_real=true only if the async ordering problem is confirmed unmitigated.`,
    { label: `triage:${f.id}`, phase: 'Triage', schema: VERDICT_SCHEMA }
  ).then(v => v ? { finding: f, verdict: v } : null)
))).filter(Boolean)

const confirmed = verdicts.filter(v => v.verdict.is_real)
log(`Triage: ${confirmed.length}/${highPriority.length} confirmed async bugs`)

// ── Phase 3: Prioritize ───────────────────────────────────────────────────────
phase('Prioritize')

const report = await agent(
  `Prioritize these async/concurrency findings.

Ranking: data corruption (race on shared state) > silent drop (missing await, fire-and-forget)
       > ordering violation (callback too early/late) > performance degradation

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

Produce: executive_summary, pr_bundle (priority order), flaky_test_risk (which findings
would cause intermittent test failures — hardest to debug).`,
  { label: 'prioritize', phase: 'Prioritize' }
)

if (INV && confirmed.length > 0) {
  await agent(
    `Store confirmed async/concurrency findings to Loci investigation "${INV}".
For each call mcp__loci__investigation_store(investigation_id="${INV}", finding_type="observed",
confidence="high", tags="async-ordering-audit,category:<category>", text="<title>: <detail> | Fix: <fix_recipe>").
Findings: ${JSON.stringify(confirmed.map(v => v.finding), null, 2)}`,
    { label: 'store:loci', phase: 'Prioritize', model: 'haiku' }
  )
}

return { root: ROOT, findings_total: allFindings.length, confirmed_high_critical: confirmed.length, report }
