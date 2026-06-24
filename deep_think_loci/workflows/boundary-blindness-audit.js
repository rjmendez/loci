export const meta = {
  name: 'boundary-blindness-audit',
  description: 'Hunt for cross-file integration bugs: producer-without-consumer, class-without-caller, field-name contract mismatches, plausible stubs, and data-flow coverage gaps. Catches the class of LLM-generated bug where each file looks correct but the system is broken at its boundaries.',
  whenToUse: 'Run after any PR that adds a new publisher, subscriber, cross-layer contract, or class with action methods. The unit of LLM generation is one file; the unit of failure is the boundary between files.',
  phases: [
    { title: 'Hunt', detail: '7 boundary hunters in parallel — events, data contracts, callers, field names, stubs, accumulation, coverage' },
    { title: 'Triage', detail: 'Adversarially verify each critical/high finding — is the other side really missing?' },
    { title: 'Prioritize', detail: 'Rank by blast radius: silent data loss > crash > correctness > performance' },
    { title: 'Store', detail: 'Persist confirmed new patterns to Loci (if loci_investigation provided)' },
  ],
}

// ── Parameters ────────────────────────────────────────────────────────────────
// Required:
//   root           — absolute path to the repository root
// Optional:
//   language_stack — array of languages in the codebase, e.g. ['python', 'typescript', 'rust']
//                    Agents adapt grep patterns to the listed languages.
//   loci_investigation — investigation_id to store confirmed new pattern instances
//   severity_floor — minimum severity to include in output ('critical'|'high'|'medium'|'low')
// ─────────────────────────────────────────────────────────────────────────────
const A      = (typeof args === 'string') ? (() => { try { return JSON.parse(args) || {} } catch (e) { return {} } })() : (args || {})
const ROOT = A.root
if (!ROOT) { log('args.root is required.'); return { error: 'root_required' } }

const LANGS  = A.language_stack || []
const INV    = A.loci_investigation || null
const FLOOR  = A.severity_floor || 'medium'

const LANG_NOTE = LANGS.length
  ? `The codebase uses: ${LANGS.join(', ')}. Adapt your grep file extensions and patterns accordingly.`
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
        required: ['severity', 'category', 'title', 'detail', 'evidence', 'fix_recipe', 'fix_effort'],
        properties: {
          severity:      { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          category:      { type: 'string' },
          title:         { type: 'string' },
          producer_file: { type: 'string' },
          consumer_file: { type: 'string' },
          contract:      { type: 'string' },
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

const SEVERITY_RANK = { critical: 4, high: 3, medium: 2, low: 1 }
const FLOOR_RANK = SEVERITY_RANK[FLOOR] || 2

// ── Phase 1: Hunt ─────────────────────────────────────────────────────────────
phase('Hunt')

const HUNTERS = [
  {
    key: 'event_topics',
    prompt: `Hunt for event/message topic contract mismatches in ${ROOT}.
${LANG_NOTE}

Goal: find every topic, channel, or event string that is PUBLISHED/EMITTED but has no
matching SUBSCRIBER/LISTENER, or where the publish path doesn't match the subscribe pattern.

Steps:
1. Find all publish call sites:
   grep -rn "publish\|emit\|dispatch\|send_event\|produce\|\.send(" ${ROOT} --include matching extensions
   Note exact topic/event name strings used.

2. Find all subscribe call sites:
   grep -rn "subscribe\|listen\|on_event\|consume\|\.on(" ${ROOT} --include matching extensions
   Note the topic patterns they register for.

3. Cross-reference:
   - Does every published topic have a subscriber that covers it?
   - Watch for wildcard depth mismatches (e.g., "sensor/+/data" won't match "sensor/device/type/data")
   - Watch for subscribers in test files only (no production subscriber)
   - Watch for topics added in recent code with no subscriber yet

For each mismatch: identify producer_file, consumer_file (or "none"), the exact contract string.
category="BB-event-topic"`,
  },
  {
    key: 'data_contracts',
    prompt: `Hunt for data pipeline field/type contract mismatches in ${ROOT}.
${LANG_NOTE}

Goal: find field names or output identifiers defined in one component that are read under
a DIFFERENT name or index in a downstream component. Silent wrong-value bugs, not crashes.

Steps:
1. Find data schema definitions — model output layer names, JSON field assignments, dict keys
   used in serialization (look for patterns like: field=, "key":, .put("key", ...), output_names=[...])

2. Find downstream readers — deserialization field lookups, model output index access, dict gets
   (look for patterns like: .get("key"), ["key"], outputNames[i], input_tensors["name"])

3. Cross-reference: for each field defined in a producer, verify a consumer reads it by the
   SAME name and at the SAME index/position. Flag any name that appears only in producers
   or only in consumers.

Common instances:
- ML model trainer names output "threat_score"; inference code reads outputNames[0] as "score"
- Android serializes JSON field "confidence"; Rust/Go deserializes field "score"
- Python exports CSV column "op_label"; training script reads column "label"

category="BB-data-contract"`,
  },
  {
    key: 'class_without_caller',
    prompt: `Hunt for classes with action methods that have zero production call sites in ${ROOT}.
${LANG_NOTE}

Goal: find classes whose primary purpose (persist, push, publish, emit, sync, train, save, record, export)
is never actually invoked in production code. The class exists; the feature is silently disabled.

Steps:
1. Find classes with action-method names:
   grep -rn "class.*Publisher\|class.*Notifier\|class.*Exporter\|class.*Recorder\|class.*Trainer\|class.*Syncer\|class.*Emitter" ${ROOT}
   Also: any class with methods named persist(), push(), publish(), sync(), train(), save(), record(), export()

2. For each candidate class, grep for its constructor call sites:
   grep -rn "ClassName(" or "new ClassName" or "ClassName.create" ${ROOT}
   Exclude: test files, __init__.py imports, type annotations

3. For each class with ≤1 constructor call site (or 0), check if its action methods
   are invoked anywhere:
   grep -rn ".actionMethod(" ${ROOT} -- exclude test files

4. Flag classes where: the class is instantiated but action methods are never called,
   OR the class is never instantiated at all outside tests.

category="BB-class-without-caller"`,
  },
  {
    key: 'field_name_contracts',
    prompt: `Hunt for cross-boundary field name divergence in ${ROOT}.
${LANG_NOTE}

Goal: find field names that travel across a serialization boundary (JSON, protobuf, CSV,
message queue payload) where producer and consumer use different names.

Steps:
1. Find serialization sites — where data is converted to a wire format:
   grep -rn "json.dumps\|json.encode\|JSONObject\|serde_json::to\|json.Marshal\|JSON.stringify" ${ROOT}
   Also: ORM model field definitions, protobuf field names, dataclass field names

2. Find deserialization sites — where wire format is converted back to in-memory:
   grep -rn "json.loads\|json.decode\|optString\|serde_json::from\|json.Unmarshal\|JSON.parse" ${ROOT}
   Also: ORM column mappings, proto field accesses, struct field names

3. Build a map of: field name at serialization → field name at deserialization
   Look for ANY name that appears in serialization but not in deserialization (or vice versa)
   across DIFFERENT files/components (same file is usually OK).

4. Specifically check cross-language or cross-service boundaries — these are highest risk:
   mobile app → backend API, Python trainer → mobile inference, microservice A → microservice B

category="BB-field-contract"`,
  },
  {
    key: 'plausible_stubs',
    prompt: `Hunt for plausible stub methods in ${ROOT} — methods that are wired into the call graph but implement nothing useful.
${LANG_NOTE}

Patterns to look for:

1. Methods named for an integration (send, publish, notify, push, export, upload, persist)
   that contain ONLY logging/print calls and no actual I/O:
   grep -rn "def send\|def publish\|def notify\|def push\|def export\|async send\|function send" ${ROOT}
   For each: check if the method body contains any HTTP call, socket write, DB write, file write.
   If only logger.*/log.*() or pass/return → plausible stub.

2. Error/failure handlers that only log (retry-on-permanent-error):
   grep -rn "except.*Exception\|catch.*Error\|\.catch(" ${ROOT}
   For each: does the handler check the error type before retrying? Or does it always retry?
   A 404 being retried is a silent infinite loop.

3. Methods where ALL branches return the same value regardless of input:
   Hard to grep directly — look for boolean methods that always return True or always return False.

4. Detectors or comparators that compare a value against itself:
   A snapshot taken and immediately compared (delta always 0, similarity always 1.0).

category="BB-plausible-stub"`,
  },
  {
    key: 'unbounded_accumulation',
    prompt: `Hunt for collections that grow without bound in ${ROOT}.
${LANG_NOTE}

Goal: find dicts, lists, sets, queues that have append/add/put calls but no corresponding
remove/clear/evict/expire/TTL. These cause OOM or unbounded slowdown in long-running processes.

Steps:
1. Find collection declarations (fields on classes, module-level dicts/sets/lists):
   grep -rn "= {}\|= \[\]\|= set()\|HashMap<\|ConcurrentHashMap<\|ArrayList<" ${ROOT}
   Focus on class-level fields (self.*, static fields), not local variables.

2. For each collection, find append sites (add, put, append, push, update):
   grep -rn "\.add(\|\.put(\|\.append(\|\.push(\|\.update(" ${ROOT}

3. Find eviction sites (remove, pop, clear, del, evict, expire, prune):
   grep -rn "\.remove(\|\.pop(\|\.clear(\|del \|\.evict(\|\.prune(" ${ROOT}

4. Flag collections where:
   - Appenders exist but no evictor exists in the same class
   - Evictor only runs on explicit flush (not on a periodic timer)
   - The collection is a dedup/seen-events set with no TTL

Also check: channel/queue capacity limits — what happens on overflow?

category="BB-unbounded-accumulation"`,
  },
  {
    key: 'data_flow_coverage',
    prompt: `Hunt for data producers with no downstream consumer visible in ${ROOT}.
${LANG_NOTE}

Goal: find components that collect, generate, or transform data but have no visible path
to a storage, analysis, training, or display component. The data is collected and silently dropped.

Steps:
1. List all data source components — collectors, sensors, scrapers, monitors:
   grep -rn "class.*Collector\|class.*Monitor\|class.*Sensor\|class.*Reader\|class.*Watcher" ${ROOT}
   Also look for background threads/tasks that gather data.

2. For each source, trace what it does with the data:
   - Does it publish to a message broker? → verify broker has subscriber
   - Does it write to a DB? → verify schema matches and writer succeeds
   - Does it write to a file/queue? → verify a reader exists
   - Does it call a downstream component directly? → verify that component exists and is wired

3. Find "orphaned" collectors — instantiated but their output path is a no-op:
   - Collector runs, calls publish(), publish() is a stub
   - Collector runs, writes to a table that nothing queries
   - Collector runs, appends to a list that is never consumed

Also check: training scripts that reference data sources which don't exist or were renamed.

category="BB-data-flow-gap"`,
  },
]

const rawHunts = await parallel(HUNTERS.map(h => () =>
  agent(h.prompt, { label: `hunt:${h.key}`, phase: 'Hunt', schema: FINDING_SCHEMA })
    .then(r => r ? { ...r, hunter: h.key } : null)
))

const hunts = rawHunts.filter(Boolean)
const allFindings = hunts.flatMap((h, i) =>
  (h.findings || [])
    .filter(f => (SEVERITY_RANK[f.severity] || 0) >= FLOOR_RANK)
    .map((f, j) => ({ ...f, id: `${h.hunter}-${j}` }))
)

log(`Hunt: ${allFindings.length} findings at or above "${FLOOR}" from ${hunts.length}/${HUNTERS.length} hunters`)

// ── Phase 2: Triage ───────────────────────────────────────────────────────────
phase('Triage')

const highPriority = allFindings.filter(f => f.severity === 'critical' || f.severity === 'high')
log(`Verifying ${highPriority.length} critical/high findings`)

const verdicts = (await parallel(highPriority.map(f => () =>
  agent(
    `Adversarially verify this boundary-blindness finding. Try to REFUTE it.
Default to is_real=false if you cannot confirm the evidence by reading the actual files.

Finding ID: ${f.id}
Category: ${f.category}
Title: ${f.title}
Detail: ${f.detail}
Evidence: ${f.evidence}
Producer file: ${f.producer_file || 'unknown'}
Consumer file: ${f.consumer_file || 'unknown'}
Contract: ${f.contract || 'unknown'}

Steps:
1. Read the producer file. Confirm the exact name/path/field being produced.
2. Read the consumer file (or confirm it doesn't exist). Confirm what it expects.
3. Do they actually mismatch? Is the other side genuinely missing?
   Consider: could an alias, interface, or framework adapter reconcile the mismatch?

Return is_real=true only if you can confirm the mismatch by reading both files.`,
    { label: `triage:${f.id}`, phase: 'Triage', schema: VERDICT_SCHEMA }
  ).then(v => v ? { finding: f, verdict: v } : null)
))).filter(Boolean)

const confirmed = verdicts.filter(v => v.verdict.is_real)
const allConfirmed = [
  ...confirmed.map(v => ({ ...v.finding, severity: v.verdict.severity || v.finding.severity })),
  ...allFindings.filter(f => f.severity === 'medium' || f.severity === 'low'),
]

log(`Triage: ${confirmed.length}/${highPriority.length} critical/high confirmed real`)

// ── Phase 3: Prioritize ───────────────────────────────────────────────────────
phase('Prioritize')

const report = await agent(
  `Produce a prioritized fix bundle from these boundary-blindness audit findings.

Ranking: silent data loss (no error, wrong result) > crash > correctness > performance

CONFIRMED HIGH/CRITICAL (${confirmed.length}):
${JSON.stringify(confirmed.map(v => ({
  id: v.finding.id, category: v.finding.category, title: v.finding.title,
  detail: v.finding.detail, fix_recipe: v.finding.fix_recipe, fix_effort: v.finding.fix_effort,
  evidence: v.finding.evidence, severity: v.verdict.severity || v.finding.severity,
})), null, 2)}

MEDIUM/LOW (not individually verified — ${allFindings.filter(f => f.severity === 'medium' || f.severity === 'low').length}):
${JSON.stringify(allFindings.filter(f => f.severity === 'medium' || f.severity === 'low').map(
  f => ({ id: f.id, category: f.category, title: f.title, detail: f.detail })
), null, 2)}

Produce:
1. executive_summary: 2-paragraph summary of boundary patterns found
2. pr_bundle: findings in priority order with file, fix instructions, and effort
3. silent_data_loss_risks: specifically call out any finding where the bug produces
   a wrong result with no error (most dangerous class)
4. new_pattern_instances: any findings that match a known pattern (BB, WG, SD, SS, SL, OG)
   but represent a NEW concrete instance worth adding to a pattern taxonomy`,
  { label: 'prioritize', phase: 'Prioritize' }
)

// ── Phase 4: Store to Loci ────────────────────────────────────────────────────
phase('Store')

if (INV && confirmed.length > 0) {
  await agent(
    `Store ${confirmed.length} confirmed boundary-blindness findings to Loci investigation "${INV}".

For each finding, call mcp__loci__investigation_store with:
  investigation_id="${INV}"
  finding_type="observed"
  confidence="high"
  tags="boundary-blindness-audit,category:<category>,severity:<severity>"
  text: "<title>: <detail> | Evidence: <evidence> | Fix: <fix_recipe>"

Findings:
${JSON.stringify(confirmed.map(v => v.finding), null, 2)}

Return the count of findings stored.`,
    { label: 'store:loci', phase: 'Store', model: 'haiku' }
  )
}

return {
  root: ROOT,
  findings_total:            allFindings.length,
  confirmed_high_critical:   confirmed.length,
  all_confirmed:             allConfirmed.length,
  report,
}
