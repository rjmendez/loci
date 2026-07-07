export const meta = {
  name: 'ci-quality-gates',
  description: 'Run configured CI quality gates (mypy, pytest, ruff, vulture, schema) against a repo, store all findings as a Loci investigation, and produce a ranked action report. Optionally files GitHub issues for failures above a severity floor. Reusable for any Python project.',
  phases: [
    { title: 'Init',   detail: 'open Loci investigation for this gate run' },
    { title: 'Scan',   detail: 'run all configured quality gates in parallel' },
    { title: 'Store',  detail: 'classify and persist gate findings to Loci' },
    { title: 'Report', detail: 'ranked summary; optional GitHub issue filing' },
  ],
}

// ── args normalization ──
const A = (typeof args === 'string')
  ? (() => { try { return JSON.parse(args) || {} } catch (e) { return {} } })()
  : (args || {})

// ── parameters ──
const REPO_PATH   = A.repo          || '/mnt/c/Users/rjmendez-admin/development/loci'
const GITHUB_REPO = A.github_repo   || null   // 'owner/repo' — set to enable issue filing
const FILE_FLOOR  = A.file_floor    || 'high' // severity floor for auto-filing GitHub issues
const DRY_RUN     = A.dry_run       || false  // if true: scan + store only, skip issue filing
const INV_TITLE   = A.title         || `CI quality gate run — ${REPO_PATH.split('/').at(-1)}`
const VENV_PREFIX = A.venv          || ''     // e.g. '/path/to/.venv/bin/'

// ── gate definitions ──
// Each gate: { id, name, cmd, severity, record_type, tags }
// Callers can override via args.gates to add/remove/replace gates.
const DEFAULT_GATES = [
  {
    id:          'mypy',
    name:        'mypy type check',
    cmd:         `cd ${REPO_PATH}/mcp && ${VENV_PREFIX}mypy server.py --config-file ../mypy.ini --no-error-summary 2>&1 || true`,
    severity:    'medium',
    record_type: 'gap',
    tags:        ['mypy', 'type-safety'],
  },
  {
    id:          'mypy-scripts',
    name:        'mypy type check (scripts)',
    cmd:         `cd ${REPO_PATH} && ${VENV_PREFIX}mypy scripts/ --config-file mypy.ini --no-error-summary 2>&1 || true`,
    severity:    'medium',
    record_type: 'gap',
    tags:        ['mypy', 'type-safety', 'scripts'],
  },
  {
    id:          'ruff',
    name:        'ruff lint',
    cmd:         `cd ${REPO_PATH} && ${VENV_PREFIX}ruff check mcp/server.py mcp/memcheck/ a2a_server/server.py scripts/ --select E9,F401,F811,F821,F841 --ignore E402 2>&1`,
    severity:    'high',
    record_type: 'gap',
    tags:        ['ruff', 'lint'],
  },
  {
    id:          'schema-consistency',
    name:        'schema consistency tests',
    cmd:         `cd ${REPO_PATH} && ${VENV_PREFIX}python -m pytest tests/test_schema_consistency.py -v --tb=short 2>&1`,
    severity:    'critical',
    record_type: 'gap',
    tags:        ['pytest', 'schema', 'sqlite'],
  },
  {
    id:          'mcp-unit-tests',
    name:        'MCP server unit tests',
    cmd:         `cd ${REPO_PATH}/mcp && ${VENV_PREFIX}python -m pytest tests/ -v --tb=short --timeout=60 --ignore=tests/test_mcp_integration.py 2>&1`,
    severity:    'critical',
    record_type: 'gap',
    tags:        ['pytest', 'unit-tests'],
  },
  {
    id:          'mcp-integration-tests',
    name:        'MCP tool integration tests',
    cmd:         `cd ${REPO_PATH}/mcp && ${VENV_PREFIX}python -m pytest tests/test_mcp_integration.py -v --tb=short --timeout=60 2>&1`,
    severity:    'critical',
    record_type: 'gap',
    tags:        ['pytest', 'integration-tests'],
  },
  {
    id:          'vulture',
    name:        'vulture dead code',
    cmd:         `cd ${REPO_PATH} && ${VENV_PREFIX}vulture mcp/server.py mcp/memcheck/ scripts/ a2a_server/server.py .vulture_whitelist.py --min-confidence 80 2>&1 || true`,
    severity:    'low',
    record_type: 'observation',
    tags:        ['vulture', 'dead-code'],
  },
]

const GATES = A.gates || DEFAULT_GATES

const SEV_RANK = { critical: 4, high: 3, medium: 2, low: 1, info: 0 }
const floorRank = SEV_RANK[FILE_FLOOR] || 3

// ── schemas ──
const GATE_RESULT_SCHEMA = {
  type: 'object',
  required: ['gate_id', 'passed', 'error_count', 'findings', 'raw_summary'],
  properties: {
    gate_id:     { type: 'string' },
    passed:      { type: 'boolean' },
    error_count: { type: 'integer' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['text', 'file', 'line', 'severity', 'message'],
        properties: {
          text:     { type: 'string' },   // one-sentence description for Loci store
          file:     { type: 'string' },   // file path or "" if not file-specific
          line:     { type: 'integer' },  // line number or 0
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low', 'info'] },
          message:  { type: 'string' },   // raw tool output for this finding
        },
      },
    },
    raw_summary: { type: 'string' },   // last 30 lines of output for context
  },
}

const STORE_RESULT_SCHEMA = {
  type: 'object',
  required: ['gate_id', 'stored_count', 'finding_ids'],
  properties: {
    gate_id:     { type: 'string' },
    stored_count: { type: 'integer' },
    finding_ids:  { type: 'array', items: { type: 'string' } },
  },
}

const ISSUE_FILED_SCHEMA = {
  type: 'object',
  required: ['filed', 'issue_url', 'title', 'error'],
  properties: {
    filed:     { type: 'boolean' },
    issue_url: { type: 'string' },
    title:     { type: 'string' },
    error:     { type: 'string' },
  },
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 1: Init — open Loci investigation
// ─────────────────────────────────────────────────────────────────────────────
phase('Init')

const initResult = await agent(
  `Open a Loci investigation for a CI quality gate run.

Title: "${INV_TITLE}"
Context: "Automated quality gate sweep across mypy, ruff, pytest, vulture, and schema consistency checks for the repo at ${REPO_PATH}. Findings are stored here so they can be searched, correlated, and actioned in future sessions."
Hypothesis: "All gates pass. Any failures indicate actionable code quality issues."

Call mcp__loci__investigation_start with these exact values and return the investigation_id.`,
  {
    label: 'init:investigation',
    phase: 'Init',
    model: 'haiku',
    schema: {
      type: 'object',
      required: ['investigation_id'],
      properties: { investigation_id: { type: 'string' } },
    },
  }
)

const INV_ID = initResult?.investigation_id
if (!INV_ID) {
  log('ERROR: could not open Loci investigation — cannot continue')
  return { error: 'investigation_start failed', initResult }
}
log(`Init: investigation ${INV_ID} opened`)

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2: Scan — run all gates in parallel
// ─────────────────────────────────────────────────────────────────────────────
phase('Scan')

const scanResults = (await parallel(GATES.map(gate => () =>
  agent(
    `Run the "${gate.name}" quality gate for the repo at ${REPO_PATH}.

Command to run exactly:
  ${gate.cmd}

After running:
1. Count the number of distinct errors/issues (not lines — distinct problems).
2. Parse each finding into a structured record with: file path, line number, a one-sentence human-readable text description suitable for storing in a knowledge base, and the severity level (use "${gate.severity}" as the baseline unless the output clearly indicates otherwise — test failures are "critical", type errors "medium", lint "high", dead code "low").
3. If the output contains "passed", "no issues", "0 errors", or "no unused code" → passed=true, error_count=0, findings=[].
4. Summarize the last 30 lines of output as raw_summary.

IMPORTANT: Return gate_id="${gate.id}" exactly.`,
    {
      label: `scan:${gate.id}`,
      phase: 'Scan',
      model: 'haiku',
      schema: GATE_RESULT_SCHEMA,
    }
  )
))).filter(Boolean)

const failed = scanResults.filter(r => !r.passed)
const passed = scanResults.filter(r => r.passed)
const totalFindings = scanResults.reduce((n, r) => n + (r.findings?.length || 0), 0)

log(`Scan: ${passed.length}/${GATES.length} gates passed — ${totalFindings} total findings across ${failed.length} failing gates`)
if (failed.length) log(`Failing gates: ${failed.map(r => r.gate_id).join(', ')}`)

// ─────────────────────────────────────────────────────────────────────────────
// Phase 3: Store — persist all findings to the Loci investigation
// ─────────────────────────────────────────────────────────────────────────────
phase('Store')

const gateConfigById = Object.fromEntries(GATES.map(g => [g.id, g]))

// Store each gate's findings. Run sequentially to avoid JSONL write races.
const storeResults = []
for (const result of scanResults) {
  const gateConfig = gateConfigById[result.gate_id] || {}
  const findings = result.findings || []

  if (!findings.length && result.passed) {
    // Store a single "passed" observation so the investigation has a complete record.
    storeResults.push(await agent(
      `Store a single "passed" finding in the Loci investigation for the "${result.gate_id}" quality gate.

investigation_id: "${INV_ID}"
text: "${gateConfig.name || result.gate_id} passed — ${result.raw_summary?.split('\\n').filter(l => l.trim()).at(-1) || 'no issues found'}"
record_type: "observation"
confidence: "high"
source: "ci-quality-gates/${result.gate_id}"
tags: ${JSON.stringify(gateConfig.tags || [result.gate_id])}

Call mcp__loci__investigation_store with these values. Return gate_id and the stored finding_id.`,
      {
        label: `store:${result.gate_id}:pass`,
        phase: 'Store',
        model: 'haiku',
        schema: STORE_RESULT_SCHEMA,
      }
    ))
    continue
  }

  if (!findings.length) continue

  storeResults.push(await agent(
    `Store ${findings.length} quality gate findings from "${result.gate_id}" into the Loci investigation.

investigation_id: "${INV_ID}"
source prefix: "ci-quality-gates/${result.gate_id}"
tags to include on every finding: ${JSON.stringify(gateConfig.tags || [result.gate_id])}
record_type for all: "${gateConfig.record_type || 'gap'}"

Findings to store (call mcp__loci__investigation_store once per finding):
${JSON.stringify(findings.slice(0, 40), null, 2)}

For each finding:
- text: use the finding's "text" field
- record_type: "${gateConfig.record_type || 'gap'}"
- confidence: map severity → confidence (critical/high→"high", medium→"medium", low/info→"low")
- source: "ci-quality-gates/${result.gate_id}:${'{'}file{'}'}:${'{'}line{'}'}"
- tags: merge gate tags with ["${result.gate_id}"]

CRITICAL: call mcp__loci__investigation_store for EVERY finding. Return gate_id="${result.gate_id}", stored_count=<actual count stored>, finding_ids=[...actual IDs from store calls...].`,
    {
      label: `store:${result.gate_id}`,
      phase: 'Store',
      model: 'sonnet',
      schema: STORE_RESULT_SCHEMA,
    }
  ))
  log(`Store: ${result.gate_id} → ${findings.length} findings queued`)
}

const storedTotal = storeResults.filter(Boolean).reduce((n, r) => n + (r?.stored_count || 0), 0)
log(`Store: ${storedTotal} findings persisted to investigation ${INV_ID}`)

// Update the investigation's next_step.
await agent(
  `Update the Loci investigation next step.

investigation_id: "${INV_ID}"
next_step: "${failed.length ? `Address ${failed.length} failing gate(s): ${failed.map(r => r.gate_id).join(', ')}. Run 'ci-quality-gates' again to verify.` : 'All gates pass. No action required — close this investigation.'}"

Call mcp__loci__investigation_note with investigation_id and next_step. Return {"ok": true}.`,
  { label: 'store:note', phase: 'Store', model: 'haiku' }
)

// ─────────────────────────────────────────────────────────────────────────────
// Phase 4: Report — ranked summary + optional GitHub issue filing
// ─────────────────────────────────────────────────────────────────────────────
phase('Report')

// Collect all findings above the filing floor for issue creation.
const fileablefindings = scanResults.flatMap(r => {
  const gateConfig = gateConfigById[r.gate_id] || {}
  return (r.findings || [])
    .filter(f => (SEV_RANK[f.severity] || 0) >= floorRank)
    .map(f => ({ ...f, gate_id: r.gate_id, gate_name: gateConfig.name || r.gate_id }))
})

let issuedResults = []
if (GITHUB_REPO && !DRY_RUN && fileablefindings.length) {
  log(`Report: filing ${fileablefindings.length} issues ≥ ${FILE_FLOOR} to ${GITHUB_REPO}`)
  issuedResults = (await parallel(fileablefindings.map(f => () =>
    agent(
      `File a GitHub issue for a quality gate failure.

Repo: ${GITHUB_REPO}
Gate: ${f.gate_name}
Finding: ${f.text}
File: ${f.file || 'N/A'}${f.line ? `:${f.line}` : ''}
Severity: ${f.severity}
Raw message: ${f.message?.slice(0, 500) || ''}

Run:
  gh issue create \
    --repo "${GITHUB_REPO}" \
    --title "[${f.severity}] ${f.gate_name}: ${f.text.slice(0, 80)}" \
    --label "severity:${f.severity}" \
    --label "ci-gate" \
    --body "## Gate\\n${f.gate_name}\\n\\n## Location\\n${f.file || 'N/A'}${f.line ? `:${f.line}` : ''}\\n\\n## Finding\\n${f.text}\\n\\n## Raw output\\n\`\`\`\\n${(f.message || '').slice(0, 1000)}\\n\`\`\`\\n\\n*Filed by ci-quality-gates workflow — investigation: ${INV_ID}*" 2>&1

Return filed=true and the issue URL from the gh output, or filed=false and error=<reason>.`,
      {
        label: `report:issue:${f.gate_id}:${f.line || 0}`,
        phase: 'Report',
        model: 'haiku',
        schema: ISSUE_FILED_SCHEMA,
      }
    )
  ))).filter(Boolean)
} else if (GITHUB_REPO && DRY_RUN) {
  log(`Report: DRY RUN — would file ${fileablefindings.length} issues to ${GITHUB_REPO}`)
} else if (!GITHUB_REPO) {
  log(`Report: no github_repo set — skipping issue filing`)
}

const issuesFiled  = issuedResults.filter(r => r.filed)
const issuesFailed = issuedResults.filter(r => !r.filed)

// Final synthesis
const reportAgent = await agent(
  `Generate a quality gate run report for the repo at ${REPO_PATH}.

investigation_id: "${INV_ID}"
Gates run: ${GATES.length}
Gates passed: ${passed.length}
Gates failed: ${failed.length}
Total findings: ${totalFindings}
Stored in Loci: ${storedTotal}
${GITHUB_REPO ? `GitHub issues filed: ${issuesFiled.length}` : 'GitHub issue filing: not configured'}

Gate results summary:
${JSON.stringify(scanResults.map(r => ({
  gate: r.gate_id,
  passed: r.passed,
  error_count: r.error_count,
  top_findings: (r.findings || []).slice(0, 3).map(f => f.text),
})), null, 2)}

${issuesFiled.length ? `Issues filed:\n${issuesFiled.map(i => `  ${i.title} → ${i.issue_url}`).join('\\n')}` : ''}
${issuesFailed.length ? `Issue filing failures:\n${issuesFailed.map(i => `  ${i.title}: ${i.error}`).join('\\n')}` : ''}

Write a concise report with:
1. Overall health verdict (all clear / needs attention / critical failures)
2. Ranked list of action items by severity
3. One-sentence recommended next step for the developer

Return as plain text — this will be shown directly to the developer.`,
  { label: 'report:synthesis', phase: 'Report', model: 'haiku' }
)

return {
  investigation_id:    INV_ID,
  gates_run:           GATES.length,
  gates_passed:        passed.length,
  gates_failed:        failed.length,
  total_findings:      totalFindings,
  stored_in_loci:      storedTotal,
  issues_filed:        issuesFiled.length,
  failing_gates:       failed.map(r => ({ gate: r.gate_id, count: r.error_count })),
  critical_findings:   scanResults.flatMap(r => (r.findings || []).filter(f => f.severity === 'critical')).slice(0, 10),
  report:              typeof reportAgent === 'string' ? reportAgent : JSON.stringify(reportAgent),
}
