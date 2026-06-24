export const meta = {
  name: 'test-coverage-gap-audit',
  description: 'Hunt for test coverage gaps that no code coverage tool will catch: integration classes never tested against real backends, golden-path entry points with no test, error paths with no exception test, cross-service contracts with no consumer test, and tests that mock so heavily they verify nothing.',
  whenToUse: 'Run after adding any new service integration, entry point, error handler, or cross-service interface. Code coverage metrics count lines executed — they say nothing about whether the integration actually works end-to-end.',
  phases: [
    { title: 'Hunt', detail: '5 coverage hunters in parallel — missing integration test, golden path absent, error path untested, contract test missing, mock overreach' },
    { title: 'Triage', detail: 'Adversarially verify each finding — is there a test in a different directory the hunter didn\'t search?' },
    { title: 'Prioritize', detail: 'Rank by: integration missing > golden path > cross-service contract > error path > mock overreach' },
  ],
}

// ── Parameters ────────────────────────────────────────────────────────────────
const ROOT = args && args.root
if (!ROOT) { log('args.root is required.'); return { error: 'root_required' } }

const LANGS = (args && args.language_stack) || []
const INV   = (args && args.loci_investigation) || null

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
          severity:          { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          category:          { type: 'string' },
          title:             { type: 'string' },
          class_or_function: { type: 'string' },
          file:              { type: 'string' },
          detail:            { type: 'string' },
          evidence:          { type: 'string' },
          fix_recipe:        { type: 'string' },
          fix_effort:        { type: 'string', enum: ['trivial', 'small', 'medium', 'large'] },
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
    key: 'integration_test_missing',
    prompt: `Hunt for integration classes that are never tested against a real or test-double backend in ${ROOT}.
${LANG_NOTE}

Mechanism (CA3): A class named for integration — Publisher, Connector, Bridge, Client,
Adapter, Gateway, Repository, Sink, Source — is tested only through unit tests where
every external call is mocked. The actual wire protocol, message format, connection
lifecycle, and error surface are never exercised. The class works in isolation and
breaks against a real backend.

Steps:
1. Find integration-class names:
   grep -rn "class.*\(Publisher\|Connector\|Bridge\|Client\|Adapter\|Gateway\|Repository\|Sink\|Source\|Handler\)" \
     ${ROOT} --include="*.py" --include="*.ts" --include="*.go" --include="*.java" | grep -v test
   Note each class name.

2. For each class: find its test file:
   find ${ROOT}/tests ${ROOT}/test_* 2>/dev/null -name "*.py" -o -name "*.ts" | \
     xargs grep -l "class_name" 2>/dev/null | head -5

3. If a test exists: does it patch/mock the underlying connection?
   grep -rn "patch\|mock\|MagicMock\|Mock()\|mocker\|jest\.mock\|vi\.mock" ${ROOT}/tests/ 2>/dev/null | head -20
   A test that mocks the DB connector for a DB class tests nothing real.

4. Find classes that are instantiated in tests WITH their real external dependency:
   grep -rn "real.*client\|actual.*connection\|testcontainers\|docker_services\|pytest_docker" \
     ${ROOT}/tests/ 2>/dev/null
   These are the good tests. Classes without equivalents are at risk.

5. Flag any integration class (non-test file) that has NO corresponding test file
   mentioning its name at all (test file doesn't import it, doesn't use its name in
   any form).

category="TC-integration-missing"`,
  },
  {
    key: 'golden_path_absent',
    prompt: `Hunt for entry points and golden-path functions with no test coverage in ${ROOT}.
${LANG_NOTE}

Mechanism: The most important path through the code — the one that is used 95% of the
time in production — has no test. LLMs generate business logic but omit tests for the
happy path because the happy path "obviously works." Tests exist for edge cases; the
nominal case that processes 10,000 requests/day is untested.

Steps:
1. Find entry-point function names:
   grep -rn "^def main\|^async def main\|^def run\|^def start\|^def handler\|^def execute\|^def process" \
     ${ROOT} --include="*.py" | grep -v test
   grep -rn "^async function main\|^function handler\|export function main\|exports\.handler" \
     ${ROOT} --include="*.ts" --include="*.js" | grep -v test

2. For each entry point: find its test:
   grep -rn "test_main\|test_run\|test_start\|test_handler\|test_execute\|test_process\|import.*main" \
     ${ROOT}/tests/ 2>/dev/null | head -20

3. Find API route handlers with no test:
   grep -rn "@app\.get\|@app\.post\|@router\.\|@app\.route" ${ROOT} --include="*.py" | grep -v test
   For each route: grep the test directory for the route path string.

4. Find the "default" or "index" response — the thing the service does when it gets
   a valid, normal request. Is there a test that sends a well-formed request and
   asserts a correct response?

5. Find CLI commands with no test:
   grep -rn "@click\.command\|@app\.command\|argparse\|sys\.argv" ${ROOT} --include="*.py" | grep -v test
   CLI commands are often tested manually only.

category="TC-golden-path-absent"`,
  },
  {
    key: 'error_path_untested',
    prompt: `Hunt for exception handlers and error paths that are never triggered in any test in ${ROOT}.
${LANG_NOTE}

Mechanism: A try/except block handles a specific error (ConnectionError, TimeoutError,
ValidationError). The happy-path test exercises the try block. No test triggers the
except block. The error handling code: may be wrong, may re-raise incorrectly, may
return the wrong HTTP status, or may not log the right fields.

Steps:
1. Find except blocks (Python):
   grep -rn "except \|except:" ${ROOT} --include="*.py" | grep -v "test\|#" | head -30
   For each: find the corresponding test that would trigger this exception.
   grep -rn "pytest\.raises\|with pytest\.raises\|assertRaises" ${ROOT}/tests/ 2>/dev/null | head -20

2. Find catch blocks (TypeScript/JavaScript):
   grep -rn "} catch (" ${ROOT} --include="*.ts" --include="*.js" | grep -v test | head -20
   For each: find a test using expect(...).toThrow() or try/catch in the test.

3. Find HTTP error responses with no test:
   grep -rn "raise HTTPException\|return Response.*status.*4\|return Response.*status.*5\|\.status(4\|\.status(5" \
     ${ROOT} --include="*.py" --include="*.ts" | grep -v test
   For each 4xx/5xx response: is there a test that triggers it?
   grep -rn "assert.*status.*4\|assert.*status.*5\|status_code == 4\|status_code == 5" \
     ${ROOT}/tests/ 2>/dev/null | head -10

4. Find error paths in connection handling:
   The reconnect logic, the timeout logic, the retry logic — these are often the most
   important code paths and the least tested.
   grep -rn "retry\|reconnect\|backoff\|timeout\|CircuitBreaker" ${ROOT} --include="*.py" | grep -v test
   For each: is there a test that actually times out or fails to connect?

category="TC-error-path-untested"`,
  },
  {
    key: 'cross_service_contract_untested',
    prompt: `Hunt for cross-service HTTP/gRPC/queue interfaces with no consumer-driven contract test in ${ROOT}.
${LANG_NOTE}

Mechanism: Service A sends a JSON payload to Service B. Service A is tested. Service B
is tested. But there is no test that verifies the SHAPE of what A sends matches what B
expects. Both evolve; a field is renamed on one side. The contract break only surfaces
in integration or production.

Steps:
1. Find outbound HTTP calls to other services:
   grep -rn "requests\.\(get\|post\|put\|delete\)\|httpx\.\(get\|post\)\|fetch(\|axios\." \
     ${ROOT} --include="*.py" --include="*.ts" | grep -v test | head -20
   For each: is there a test that validates the shape of the request body sent?

2. Find message queue producers:
   grep -rn "publish(\|send_message\|produce(\|\.send(\|channel\.basic_publish\|kafka.*send" \
     ${ROOT} --include="*.py" --include="*.ts" | grep -v test
   For each: is there a test that captures the message and asserts its schema?

3. Find gRPC clients:
   grep -rn "\.Stub(\|grpc\.\|stub\." ${ROOT} --include="*.py" --include="*.ts"
   For each: is there a contract test using a test double that validates the proto schema?

4. Find Pact or contract testing frameworks:
   grep -rn "pact\|consumer_driven\|contract_test\|schemathesis\|dredd" ${ROOT} --include="*.py" --include="*.ts"
   If no Pact/Schemathesis usage: there are no consumer-driven contract tests.

5. Find OpenAPI/JSON Schema validation in tests:
   grep -rn "validate_schema\|jsonschema\|openapi_validate\|assert_valid\|fastjsonschema" \
     ${ROOT}/tests/ 2>/dev/null
   Absence of schema validation in tests that exercise API responses = unverified contracts.

category="TC-contract-untested"`,
  },
  {
    key: 'mock_overreach',
    prompt: `Hunt for tests that mock so many dependencies that the test verifies nothing real in ${ROOT}.
${LANG_NOTE}

Mechanism: A test mocks the database, the HTTP client, the message queue, and the file
system. The unit under test now only interacts with mock objects. The test verifies that
mock.method() was called — not that the actual integration works. A bug in the real
call site (wrong argument, wrong field name, wrong protocol) cannot be caught.

Steps:
1. Find tests with many mocks:
   grep -rn "patch\|MagicMock\|Mock()" ${ROOT}/tests/ 2>/dev/null | \
     awk -F: '{print $1}' | sort | uniq -c | sort -rn | head -10
   Test files with 10+ mock usages are candidates.

2. Find tests that patch the ENTIRE external call (not just a test double):
   @patch("module.DatabaseClient")  ← patches the whole class
   vs
   @patch("module.DatabaseClient.execute")  ← patches just one method
   The first form can mask wrong instantiation, wrong __init__ args, etc.

3. Find tests that assert only on mock calls (not on output):
   grep -rn "assert_called\|assert_called_once\|assert_called_with\|called_once_with" \
     ${ROOT}/tests/ 2>/dev/null | head -20
   Tests that ONLY assert mock.method.called_with() verify call routing, not result
   correctness. The actual return path (what happens after the mock call) is unverified.

4. Find test classes where setUp mocks every external system:
   grep -rn "def setUp\|@pytest.fixture" ${ROOT}/tests/ 2>/dev/null | head -10
   For each setUp: count how many patches are applied. More than 5 external systems
   patched = likely overreach.

5. Look for any test that creates a real in-memory representation of the external system:
   grep -rn "sqlite3.*:memory:\|fakeredis\|moto\|localstack\|testcontainers" ${ROOT}/tests/ 2>/dev/null
   These are good — they use a real (or real-backed) implementation, not a mock.
   Files with many mocks but no real-backed test double are at risk.

category="TC-mock-overreach"`,
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
    `Adversarially verify this test coverage gap finding. Try to REFUTE it.
Default to is_real=false if you cannot confirm the test gap exists.

Finding ID: ${f.id} | Category: ${f.category}
Class/Function: ${f.class_or_function || 'unknown'}
File: ${f.file || 'unknown'}
Title: ${f.title}
Detail: ${f.detail}
Evidence: ${f.evidence}

Steps:
1. Search the test directory more thoroughly — is there a test in a subdir the hunter missed?
2. For integration classes: is there an end-to-end test suite (e2e/, integration/) that covers it?
3. For mocked tests: does the mock implement the same interface as the real object (good fidelity)?
4. For error paths: is the error path exercised by a higher-level integration test?
5. For contracts: does the OpenAPI spec enforce the contract at the framework level?

Return is_real=true only if the coverage gap is confirmed absent from ALL test suites.`,
    { label: `triage:${f.id}`, phase: 'Triage', schema: VERDICT_SCHEMA }
  ).then(v => v ? { finding: f, verdict: v } : null)
))).filter(Boolean)

const confirmed = verdicts.filter(v => v.verdict.is_real)
log(`Triage: ${confirmed.length}/${highPriority.length} confirmed coverage gaps`)

// ── Phase 3: Prioritize ───────────────────────────────────────────────────────
phase('Prioritize')

const report = await agent(
  `Prioritize these test coverage gap findings.

Ranking: integration missing (breaks in prod only) > golden path (core business logic untested)
       > cross-service contract (silent schema drift) > error path (wrong behavior under failure)
       > mock overreach (false confidence)

CONFIRMED HIGH/CRITICAL (${confirmed.length}):
${JSON.stringify(confirmed.map(v => ({
  id: v.finding.id, category: v.finding.category, title: v.finding.title,
  class_or_function: v.finding.class_or_function, detail: v.finding.detail,
  fix_recipe: v.finding.fix_recipe, fix_effort: v.finding.fix_effort,
  severity: v.verdict.severity || v.finding.severity,
})), null, 2)}

MEDIUM/LOW (${allFindings.filter(f => f.severity === 'medium' || f.severity === 'low').length}):
${JSON.stringify(allFindings.filter(f => f.severity === 'medium' || f.severity === 'low').map(
  f => ({ id: f.id, category: f.category, title: f.title })
), null, 2)}

Produce: executive_summary, pr_bundle (priority order), false_confidence_risk (tests
that pass but are actually testing nothing — the most dangerous category because they
suppress the reflex to add more tests).`,
  { label: 'prioritize', phase: 'Prioritize' }
)

if (INV && confirmed.length > 0) {
  await agent(
    `Store confirmed test coverage gap findings to Loci investigation "${INV}".
For each call mcp__loci__investigation_store(investigation_id="${INV}", finding_type="gap",
confidence="high", tags="test-coverage-gap-audit,category:<category>", text="<title>: <detail> | Fix: <fix_recipe>").
Findings: ${JSON.stringify(confirmed.map(v => v.finding), null, 2)}`,
    { label: 'store:loci', phase: 'Prioritize', model: 'haiku' }
  )
}

return { root: ROOT, findings_total: allFindings.length, confirmed_high_critical: confirmed.length, report }
