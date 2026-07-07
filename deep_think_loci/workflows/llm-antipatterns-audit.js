export const meta = {
  name: 'llm-antipatterns-audit',
  description: 'Comprehensive audit across all major LLM code generation failure pattern categories: boundary blindness, wiring gaps, schema drift, and silent failures. Runs all category hunters in a single parallel sweep, deduplicates across categories, and produces a unified prioritized report.',
  whenToUse: 'Run before major releases, after large refactors, or whenever LLM-generated code has been merged across multiple PRs without systematic review. For targeted audits, use the individual category workflows instead.',
  phases: [
    { title: 'Hunt', detail: 'All 22 hunters across 4 categories in parallel' },
    { title: 'Triage', detail: 'Adversarially verify all critical/high findings' },
    { title: 'Deduplicate', detail: 'Merge findings that describe the same root cause across categories' },
    { title: 'Synthesize', detail: 'Unified prioritized report with cross-category patterns' },
  ],
}

// ── Parameters ────────────────────────────────────────────────────────────────
// Required:
//   root           — absolute path to the repository root
// Optional:
//   language_stack — e.g. ['python', 'typescript', 'go']
//   loci_investigation — investigation_id to store confirmed findings
//   severity_floor — 'critical'|'high'|'medium'|'low' (default: 'medium')
//   categories     — subset of ['boundary-blindness','wiring-gap','schema-drift','silent-failure']
//                    default: all four
// ─────────────────────────────────────────────────────────────────────────────
const A          = (typeof args === 'string') ? (() => { try { return JSON.parse(args) || {} } catch (e) { return {} } })() : (args || {})
const ROOT       = A.root
if (!ROOT) { log('args.root is required.'); return { error: 'root_required' } }

const LANGS      = A.language_stack || []
const INV        = A.loci_investigation || null
const FLOOR      = A.severity_floor || 'medium'
const CATS       = A.categories || ['boundary-blindness', 'wiring-gap', 'schema-drift', 'silent-failure']

const LANG_NOTE = LANGS.length
  ? `The codebase uses: ${LANGS.join(', ')}. Adapt grep extensions and patterns accordingly.`
  : 'Detect languages from file extensions. Adapt grep patterns to match.'

const SEVERITY_RANK = { critical: 4, high: 3, medium: 2, low: 1 }
const FLOOR_RANK    = SEVERITY_RANK[FLOOR] || 2

// ── Shared schemas ────────────────────────────────────────────────────────────
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
          severity:      { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          category:      { type: 'string' },
          title:         { type: 'string' },
          producer_file: { type: 'string' },
          consumer_file: { type: 'string' },
          file:          { type: 'string' },
          detail:        { type: 'string' },
          evidence:      { type: 'string' },
          fix_recipe:    { type: 'string' },
          fix_effort:    { type: 'string', enum: ['trivial', 'small', 'medium', 'large'] },
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

// ── All hunters, tagged by category ──────────────────────────────────────────
const ALL_HUNTERS = []

if (CATS.includes('boundary-blindness')) {
  ALL_HUNTERS.push(
    {
      key: 'bb:event_topics',
      cat: 'BB-event-topic',
      prompt: `Find every event/message topic published in ${ROOT} with no matching subscriber, or where publish path depth doesn't match subscribe wildcard depth. ${LANG_NOTE}
Check: grep -rn "publish|emit|dispatch|send_event" and "subscribe|listen|on_event". Cross-reference. Wildcards like "sensor/+" don't match "sensor/device/type/data". Report producer_file, consumer_file (or "none"), severity, fix_recipe.`,
    },
    {
      key: 'bb:class_callers',
      cat: 'BB-class-without-caller',
      prompt: `Find classes in ${ROOT} with action methods (persist, push, publish, emit, sync, train, save, record, export) that have zero production call sites. ${LANG_NOTE}
grep -rn "class.*Publisher|class.*Notifier|class.*Exporter|class.*Trainer|class.*Syncer". For each: count constructor call sites and action-method call sites excluding test files. Flag classes where action methods are never invoked in production code.`,
    },
    {
      key: 'bb:field_contracts',
      cat: 'BB-field-contract',
      prompt: `Find cross-layer field name divergence in ${ROOT}: field names defined at serialization (JSON put/assign) that are read under a different name at deserialization. ${LANG_NOTE}
Check JSON serialize and deserialize sites in different files/components. Flag names that appear only in producers or only in consumers across component boundaries. Specifically check cross-language boundaries.`,
    },
    {
      key: 'bb:plausible_stubs',
      cat: 'BB-plausible-stub',
      prompt: `Find methods named for an integration (send, publish, notify, push, export, upload) in ${ROOT} that contain ONLY log/print calls and no actual I/O. ${LANG_NOTE}
grep -rn "def send|def publish|def notify|async send|function send". For each: does the body contain HTTP, socket, DB, or queue I/O? If only logger.*() or pass → stub. Also: error handlers that only log and retry permanent errors (HTTP 404/401 being retried).`,
    },
    {
      key: 'bb:unbounded',
      cat: 'BB-unbounded-accumulation',
      prompt: `Find class-level collections in ${ROOT} that have append/add/put call sites but no corresponding eviction (remove/clear/evict/TTL). ${LANG_NOTE}
grep -rn "= {}|= []|= set()|HashMap<|ArrayList<" for class-level fields. For each: find add sites and remove/clear sites. Flag collections where appenders exist but evictor doesn't, or only runs on explicit flush.`,
    },
    {
      key: 'bb:data_flow',
      cat: 'BB-data-flow-gap',
      prompt: `Find data producers in ${ROOT} whose output path is a stub or has no downstream consumer. ${LANG_NOTE}
grep -rn "class.*Collector|class.*Monitor|class.*Sensor|class.*Reader". For each: trace what it does with data — publishes to broker (verify subscriber), writes to DB (verify schema), calls downstream (verify wired). Flag collectors where the data path ends in a no-op or missing component.`,
    }
  )
}

if (CATS.includes('wiring-gap')) {
  ALL_HUNTERS.push(
    {
      key: 'wg:named_stubs',
      cat: 'WG-named-stub',
      prompt: `Find classes named for an integration in ${ROOT} (Publisher, Notifier, Alerter, Exporter, Reporter, Sender) that contain no actual outbound I/O in their primary action method. ${LANG_NOTE}
For each: read the send/publish/notify/export method. Does it contain HTTP, socket, queue, or DB write? Or only logger.*() and return None? Also: methods that silently no-op when config key is absent.`,
    },
    {
      key: 'wg:metrics',
      cat: 'WG-metric-noop',
      prompt: `Find metric counters/gauges in ${ROOT} that are declared but have zero increment call sites in production code. ${LANG_NOTE}
grep -rn "Counter(|Gauge(|Histogram(|metrics.counter|statsd.incr" and then grep for each metric variable's .inc()/.observe()/.record() calls. Flag metrics where increment only exists in tests or doesn't exist at all. Also: /health endpoints that always return 200 regardless of internal state.`,
    },
    {
      key: 'wg:silent_observers',
      cat: 'WG-silent-observer',
      prompt: `Find event handlers/observers in ${ROOT} that are registered but whose handler bodies are stubs or do nothing useful. ${LANG_NOTE}
grep -rn "addEventListener|addObserver|add_listener|\.on(|\.subscribe(|register_handler". For each: read the handler body. Does it actually process the event, or only log it? Specifically: error event handlers that should trigger recovery but only log; data event handlers that collect but never persist.`,
    }
  )
}

if (CATS.includes('schema-drift')) {
  ALL_HUNTERS.push(
    {
      key: 'sd:serialization_types',
      cat: 'SB-serialization-type',
      prompt: `Find types in ${ROOT} that are valid in memory but fail at json.dumps/JSON.stringify/json.Marshal boundaries: numpy arrays, naive datetimes, Enum members, Decimal, bytes, NaN/Inf. ${LANG_NOTE}
grep -rn "json.dumps|json.encode|JSON.stringify|json.Marshal|serde_json::to". For each: does the serialized value include numpy/torch tensors, datetime without .isoformat(), Enum without .value, Decimal without float(), bytes without encoding?`,
    },
    {
      key: 'sd:field_rename',
      cat: 'SD-field-rename-drift',
      prompt: `Find field names in ${ROOT} that were renamed in definitions but not updated in string-based consumer accesses. ${LANG_NOTE}
Check recent git history: git -C ${ROOT} diff HEAD~5 HEAD -- '*.py' '*.ts' '*.java' '*.go' 2>/dev/null | grep '^[-+].*"\\w' | head -40. Also: build field name inventory from model/dataclass/struct definitions vs dict["field"] accesses. Flag names that appear in accesses but not in any current definition.`,
    },
    {
      key: 'sd:api_contracts',
      cat: 'SD-api-contract-drift',
      prompt: `Find API response field names in ${ROOT} emitted by the server that differ from what clients access. ${LANG_NOTE}
Find server response construction (return {}, jsonify(), c.JSON()) and client consumption (response.field, result["key"], .json()). Cross-reference: is there a field the client reads that the server no longer sends? Also check event-driven contracts: Kafka message schemas, MQTT payload schemas, WebSocket message formats.`,
    },
    {
      key: 'sd:cross_service_types',
      cat: 'SD-cross-service-type',
      prompt: `Find type mismatches at cross-service or cross-language boundaries in ${ROOT}: timestamps (seconds vs milliseconds), booleans (0/1 vs true/false), enums (int vs string), IDs (int vs string). ${LANG_NOTE}
Find HTTP call sites, shared queue producers/consumers, shared DB tables read by multiple services. For each: do both sides agree on epoch seconds vs milliseconds? Bool encoding? Enum int vs string? Null vs absent field?`,
    }
  )
}

if (CATS.includes('silent-failure')) {
  ALL_HUNTERS.push(
    {
      key: 'sf:swallowed',
      cat: 'SS-swallowed-exception',
      prompt: `Find catch/except blocks in ${ROOT} that swallow exceptions with no re-raise, no log with exc_info, or body that is just "pass". ${LANG_NOTE}
grep -rn "except Exception|except:|catch.*Error|catch (e)". For each: does the body re-raise? Does it log with full exception? Or does it just return None/False/[] silently? Flag "except: pass" as critical. Flag "except Exception: return None" without logging as high.`,
    },
    {
      key: 'sf:retry_permanent',
      cat: 'SS-retry-permanent',
      prompt: `Find retry logic in ${ROOT} that retries permanent errors (HTTP 404/401/403/400) or has no max retry count. ${LANG_NOTE}
grep -rn "@retry|@backoff|tenacity|MAX_RETRY|while True" and retry loops. For each: does the retry filter check HTTP status before retrying? Is there a max retry count? Does it use exponential backoff with jitter or fixed sleep (thundering herd)?`,
    },
    {
      key: 'sf:observability',
      cat: 'OG-observability-gap',
      prompt: `Find failure paths in ${ROOT} that leave no evidence: no log, no metric increment, no trace. ${LANG_NOTE}
For each major failure path (DB call, HTTP call, message publish), is there: a log call with the exception? An error metric that increments? A correlation ID in log messages? Background threads with no top-level error logging? State transitions with no audit trail?`,
    },
    {
      key: 'sf:metastable',
      cat: 'MF-metastable',
      prompt: `Find metastable failure patterns in ${ROOT}: retry without jitter, missing circuit breakers, queue-full deadlocks, cache thundering herds. ${LANG_NOTE}
grep -rn "sleep|time.Sleep|asyncio.sleep" in retry loops — is backoff fixed (thundering herd) or exponential+jitter? Are there circuit breakers on external calls? Connection pool timeout configured? Cache-miss paths without stampede protection (lock on miss)?`,
    }
  )
}

log(`Running ${ALL_HUNTERS.length} hunters across categories: ${CATS.join(', ')}`)

// ── Phase 1: Hunt ─────────────────────────────────────────────────────────────
phase('Hunt')

const rawHunts = await parallel(ALL_HUNTERS.map(h => () =>
  agent(h.prompt, { label: `hunt:${h.key}`, phase: 'Hunt', schema: FINDING_SCHEMA })
    .then(r => r ? { ...r, hunter: h.key, cat: h.cat } : null)
))

const hunts = rawHunts.filter(Boolean)
const allFindings = hunts.flatMap((h, i) =>
  (h.findings || [])
    .filter(f => (SEVERITY_RANK[f.severity] || 0) >= FLOOR_RANK)
    .map((f, j) => ({ ...f, category: f.category || h.cat, id: `${h.hunter}-${j}` }))
)

log(`Hunt: ${allFindings.length} findings at or above "${FLOOR}" from ${hunts.length}/${ALL_HUNTERS.length} hunters`)

// ── Phase 2: Triage ───────────────────────────────────────────────────────────
phase('Triage')

const highPriority = allFindings.filter(f => f.severity === 'critical' || f.severity === 'high')
log(`Verifying ${highPriority.length} critical/high findings`)

const verdicts = (await parallel(highPriority.map(f => () =>
  agent(
    `Adversarially verify this finding. Try to REFUTE it.
Default to is_real=false if you cannot confirm the evidence by reading actual files.

Finding ID: ${f.id} | Category: ${f.category}
Title: ${f.title}
Detail: ${f.detail}
Evidence: ${f.evidence}
Producer: ${f.producer_file || f.file || 'unknown'}
Consumer: ${f.consumer_file || 'unknown'}

Read the actual code. Is there a framework, middleware, or adapter that mitigates this?
Return is_real=true only if confirmed unmitigated.`,
    { label: `triage:${f.id}`, phase: 'Triage', schema: VERDICT_SCHEMA }
  ).then(v => v ? { finding: f, verdict: v } : null)
))).filter(Boolean)

const confirmed = verdicts.filter(v => v.verdict.is_real)
log(`Triage: ${confirmed.length}/${highPriority.length} critical/high confirmed real`)

// ── Phase 3: Deduplicate + Synthesize ────────────────────────────────────────
phase('Synthesize')

const catCounts = {}
allFindings.forEach(f => { catCounts[f.category] = (catCounts[f.category] || 0) + 1 })

const finalReport = await agent(
  `You are synthesizing a cross-category LLM antipatterns audit report for ${ROOT}.

CATEGORY BREAKDOWN:
${JSON.stringify(catCounts, null, 2)}

CONFIRMED HIGH/CRITICAL (${confirmed.length} findings):
${JSON.stringify(confirmed.map(v => ({
  id:          v.finding.id,
  category:    v.finding.category,
  title:       v.finding.title,
  detail:      v.finding.detail,
  fix_recipe:  v.finding.fix_recipe,
  fix_effort:  v.finding.fix_effort,
  severity:    v.verdict.severity || v.finding.severity,
  producer:    v.finding.producer_file || v.finding.file,
  consumer:    v.finding.consumer_file,
})), null, 2)}

MEDIUM/LOW (${allFindings.filter(f => f.severity === 'medium' || f.severity === 'low').length} unverified):
${JSON.stringify(allFindings.filter(f => f.severity === 'medium' || f.severity === 'low').map(
  f => ({ id: f.id, category: f.category, title: f.title })
), null, 2)}

Produce:
1. **executive_summary** (3 paragraphs):
   - What categories had the most findings and why
   - The highest-blast-radius finding and what production impact it would have
   - The cross-category meta-pattern (if any) — e.g., "this codebase has a systemic issue with
     not verifying that new publishers have subscribers"

2. **pr_bundle** — all confirmed findings ranked by:
   silent-data-loss > crash > wrong-result > observability-gap > test-fidelity
   Each entry: rank, category, title, exact_fix (file:line or command), effort

3. **deduplicated_root_causes** — if multiple findings share a root cause (e.g., 3 different
   fields all have the same "renamed in model but not in consumers" pattern), collapse them
   into one root cause with multiple instances

4. **systemic_issues** — patterns that appear across multiple categories, suggesting a process
   or architectural gap rather than individual bugs

5. **false_positive_rate** — of the ${highPriority.length} high/critical findings reviewed,
   ${highPriority.length - confirmed.length} were false positives. Note what caused them.`,
  { label: 'synthesize', phase: 'Synthesize' }
)

if (INV && confirmed.length > 0) {
  await agent(
    `Store ${confirmed.length} confirmed cross-category antipattern findings to Loci investigation "${INV}".
For each: mcp__loci__investigation_store(investigation_id="${INV}", finding_type="observed",
confidence="high", tags="llm-antipatterns-audit,category:<category>,severity:<severity>",
text="<title>: <detail> | Fix: <fix_recipe>").
Findings: ${JSON.stringify(confirmed.map(v => v.finding), null, 2)}`,
    { label: 'store:loci', phase: 'Synthesize', model: 'haiku' }
  )
}

return {
  root:                    ROOT,
  categories_run:          CATS,
  hunters_run:             ALL_HUNTERS.length,
  findings_total:          allFindings.length,
  confirmed_high_critical: confirmed.length,
  false_positives:         highPriority.length - confirmed.length,
  by_category:             catCounts,
  report:                  finalReport,
}
