export const meta = {
  name: 'wiring-gap-audit',
  description: 'Hunt for components that are named for an integration they never actually perform: stubs that look wired, metrics that never increment, agent tools that wrap no-ops, and observers that silently discard. Pattern WG/AT from the LLM hallucination taxonomy.',
  whenToUse: 'Run after adding any new Publisher, Notifier, Exporter, Reporter, metric counter, or agent tool. LLMs name classes for what they should do; they often implement less.',
  phases: [
    { title: 'Hunt', detail: '5 wiring hunters in parallel — named stubs, metric no-ops, agent tools, observers, health checks' },
    { title: 'Triage', detail: 'Adversarially verify each critical/high finding by reading the actual method bodies' },
    { title: 'Prioritize', detail: 'Rank by operational impact: silent failure > metric blindness > observability gap' },
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
          class_name:  { type: 'string' },
          method_name: { type: 'string' },
          file:        { type: 'string' },
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
    key: 'named_stubs',
    prompt: `Hunt for classes named for an integration that contains no actual integration call in ${ROOT}.
${LANG_NOTE}

Mechanism (WG1): The class name promises an outbound action (publish, alert, export, notify, upload).
The method body only logs, or silently returns when config is absent.
The system "looks wired" — logs say "sent", nothing arrives at the target.

Steps:
1. Find integration-named classes:
   grep -rn "class.*Publisher\|class.*Notifier\|class.*Alerter\|class.*Exporter\|class.*Reporter\|class.*Sender\|class.*Uploader\|class.*Forwarder\|class.*Dispatcher" ${ROOT}

2. For each class, read its primary action method (send, publish, notify, alert, export, upload):
   - Does the method body contain an actual outbound call?
     HTTP: requests.post, httpx.post, urllib, fetch, axios, reqwest::Client
     Socket: socket.send, WebSocket.send, mqtt.publish
     Queue: sqs.send_message, kafka.produce, rabbitmq.publish
     DB: session.add, db.insert, collection.insert
   - OR does it only: logger.*/log.*(), print(), return None, pass?

3. Also check: methods that only call if self.config is set — if the config key is None by default
   and no validation exists, the method is silently disabled in all deployments without that key.

Report each class/method with: what outbound call is missing, what the method actually does.
category="WG-named-stub"`,
  },
  {
    key: 'metric_noop',
    prompt: `Hunt for metrics that are declared but never incremented in the hot path in ${ROOT}.
${LANG_NOTE}

Mechanism (MC/OG): A metric counter or gauge is defined and appears in dashboards.
It shows 0. But it was never wired to the actual error/event path.
"0 errors" in production actually means "never measured."

Steps:
1. Find metric declarations:
   grep -rn "Counter(\|Gauge(\|Histogram(\|prometheus.Counter\|metrics.counter\|statsd.incr\|datadog.increment\|new Metric(" ${ROOT}
   Note each metric's name and variable.

2. For each metric, find its increment/observe/record call sites:
   grep -rn "<metric_var>\.inc(\|<metric_var>\.observe(\|<metric_var>\.record(\|<metric_var>\.labels(" ${ROOT}
   Count call sites, note which files they're in.

3. Flag metrics where:
   - Zero increment call sites exist anywhere
   - Increment only exists in test files
   - Increment is in an error handler that also catches and suppresses the error (so metric fires
     but error is swallowed — misleading signal)
   - Counter for "requests" exists but counter for "errors" was never added

Also check: health check endpoints that always return 200 regardless of internal state.
category="WG-metric-noop"`,
  },
  {
    key: 'agent_tools',
    prompt: `Hunt for agent tools (MCP tools, function-calling tools, LLM tool definitions) in ${ROOT}
that wrap an implementation which is a stub or no-op.
${LANG_NOTE}

Mechanism (AT patterns): An agent tool is defined with a clear name and description.
The underlying function it calls is a stub. The LLM sees a working tool; the tool does nothing.

Steps:
1. Find tool definitions — MCP tool decorators, function schemas, tool spec dicts:
   grep -rn "@mcp.tool\|@tool\|FunctionDefinition\|ToolSpec\|\"type\": \"function\"\|tool_name" ${ROOT}

2. For each tool, read the function it calls:
   - Does the function body contain the actual operation the tool description promises?
   - Or does it: return a hardcoded response, call a stub, only log, return empty/None?

3. Also check: tools that accept a "dry_run" or "preview" parameter but execute the real
   action regardless. And tools whose docstring says "creates X" but only validates input.

4. Check for tools that wrap filesystem or shell operations:
   - Does the tool use a safe path-validation step that silently truncates dangerous paths
     but succeeds rather than raising?

category="AT-tool-stub"`,
  },
  {
    key: 'silent_observers',
    prompt: `Hunt for observers, event handlers, and callbacks in ${ROOT} that are registered
but whose handler bodies are stubs or do less than expected.
${LANG_NOTE}

Mechanism: A listener is registered (on_event, add_listener, subscribe, addObserver).
The handler fires. But the handler body only logs, or calls a method that is a stub.
The event is "handled" — silently discarded.

Steps:
1. Find event handler registrations:
   grep -rn "addEventListener\|addObserver\|add_listener\|\.on(\|\.subscribe(\|register_handler\|@event_handler\|@receiver" ${ROOT}

2. For each registration, find the handler function/method it points to.
   Read the handler body. Does it:
   a) Actually process the event (update state, trigger action, persist data)?
   b) Only log the event?
   c) Call a method that is itself a stub?
   d) Return immediately for certain event types (silent filter)?

3. Specifically look for:
   - Error event handlers that only log (should trigger alerting, retry, or recovery)
   - Data event handlers that don't persist (collects data, discards it)
   - Disconnect/reconnect handlers that don't restore subscriptions

category="WG-silent-observer"`,
  },
  {
    key: 'health_checks',
    prompt: `Hunt for health check and readiness probe implementations in ${ROOT} that do not
actually check the thing they claim to check.
${LANG_NOTE}

Mechanism: A /health or /ready endpoint returns 200 OK. But it doesn't verify the database
connection, message queue, model, or external service it's supposed to check.
Kubernetes thinks the pod is healthy; the pod is broken.

Steps:
1. Find health check endpoints:
   grep -rn "/health\|/ready\|/live\|health_check\|readiness\|liveness" ${ROOT}
   Also: "ping", "status", "heartbeat" endpoints.

2. For each endpoint handler, read the implementation:
   - Does it actually connect to the DB and run a query?
   - Does it actually call the external service?
   - Does it check that the ML model is loaded and can run inference?
   - OR does it just return {"status": "ok"} unconditionally?

3. Also check: health checks that catch ALL exceptions and return 200 even on failure:
   try: check_db() except: return {"status": "ok"}  ← always healthy

4. Check circuit breakers: does the health check respect open circuit state?
   If DB circuit is open, /health should return 503.

category="WG-fake-health-check"`,
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
log(`Verifying ${highPriority.length} critical/high findings`)

const verdicts = (await parallel(highPriority.map(f => () =>
  agent(
    `Adversarially verify this wiring-gap finding. Try to REFUTE it.
Default to is_real=false if you cannot confirm the method is actually a stub.

Finding ID: ${f.id}
Category: ${f.category}
Class/Method: ${f.class_name || '?'} / ${f.method_name || '?'}
File: ${f.file || 'unknown'}
Detail: ${f.detail}
Evidence: ${f.evidence}

Steps:
1. Read the method body in the file.
2. Is there TRULY no outbound call, or did the hunter miss an indirect call through a helper?
3. Is there a framework/middleware that handles the actual I/O transparently?
4. Is the method a base class stub that's overridden in a subclass that does the real work?

Return is_real=true only if the method truly does less than its name/description promises.`,
    { label: `triage:${f.id}`, phase: 'Triage', schema: VERDICT_SCHEMA }
  ).then(v => v ? { finding: f, verdict: v } : null)
))).filter(Boolean)

const confirmed = verdicts.filter(v => v.verdict.is_real)
log(`Triage: ${confirmed.length}/${highPriority.length} confirmed real wiring gaps`)

// ── Phase 3: Prioritize ───────────────────────────────────────────────────────
phase('Prioritize')

const medLow = allFindings.filter(f => f.severity === 'medium' || f.severity === 'low')

const report = await agent(
  `Produce a prioritized fix bundle from these wiring-gap findings.

Ranking: silent failure (no error, operation looks done but isn't) > metric blindness > observability gap

CONFIRMED HIGH/CRITICAL (${confirmed.length}):
${JSON.stringify(confirmed.map(v => ({
  id: v.finding.id, category: v.finding.category, title: v.finding.title,
  class_name: v.finding.class_name, method_name: v.finding.method_name,
  detail: v.finding.detail, fix_recipe: v.finding.fix_recipe, fix_effort: v.finding.fix_effort,
  severity: v.verdict.severity || v.finding.severity,
})), null, 2)}

MEDIUM/LOW (${medLow.length}):
${JSON.stringify(medLow.map(f => ({ id: f.id, category: f.category, title: f.title, detail: f.detail })), null, 2)}

Produce:
1. executive_summary: key wiring gaps found and their operational risk
2. pr_bundle: findings in priority order with exact fix instructions
3. silent_failures: list any finding where the system reports success but the operation did not occur`,
  { label: 'prioritize', phase: 'Prioritize' }
)

if (INV && confirmed.length > 0) {
  await agent(
    `Store ${confirmed.length} confirmed wiring-gap findings to Loci investigation "${INV}".
For each call mcp__loci__investigation_store(investigation_id="${INV}", finding_type="observed",
confidence="high", tags="wiring-gap-audit,category:<category>", text="<title>: <detail> | Fix: <fix_recipe>").
Findings: ${JSON.stringify(confirmed.map(v => v.finding), null, 2)}`,
    { label: 'store:loci', phase: 'Prioritize', model: 'haiku' }
  )
}

return { root: ROOT, findings_total: allFindings.length, confirmed_high_critical: confirmed.length, report }
