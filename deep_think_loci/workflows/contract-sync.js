export const meta = {
  name: 'contract-sync',
  description: 'Extract implicit producer/consumer contracts from recently changed files and store them to a Loci investigation. Detects field-name drift between producers and consumers by cross-checking all stored contracts. Designed to run as a post-commit hook on .py/.ts/.go files.',
  whenToUse: 'Run after commits that touch serialization, API endpoints, message producers/consumers, or data models. Stores contracts for query by other tools. Also run manually before generating a new consumer of an existing API.',
  phases: [
    { title: 'Extract', detail: 'Identify serialization points, API shapes, and message schemas in changed files' },
    { title: 'Store', detail: 'Declare each contract to Loci; cross-check against stored contracts for field drift' },
    { title: 'Report', detail: 'Summarize stored contracts and any detected conflicts' },
  ],
}

// ── Parameters ────────────────────────────────────────────────────────────────
const ROOT  = args && args.root
const INV   = args && args.loci_investigation
if (!ROOT)  { log('args.root is required.'); return { error: 'root_required' } }
if (!INV)   { log('args.loci_investigation is required.'); return { error: 'investigation_required' } }

const SINCE = (args && args.since_commit) || 'HEAD~1'
const LANGS = (args && args.language_stack) || []
const LANG_NOTE = LANGS.length
  ? `The codebase uses: ${LANGS.join(', ')}.`
  : 'Detect languages from file extensions.'

const CONTRACT_SCHEMA = {
  type: 'object',
  required: ['contracts'],
  properties: {
    contracts: {
      type: 'array',
      items: {
        type: 'object',
        required: ['entity', 'role', 'fields'],
        properties: {
          entity:   { type: 'string' },
          role:     { type: 'string', enum: ['producer', 'consumer'] },
          fields:   { type: 'object', additionalProperties: { type: 'string' } },
          protocol: { type: 'string' },
          file:     { type: 'string' },
          note:     { type: 'string' },
        },
      },
    },
    skipped_reason: { type: 'string' },
  },
}

// ── Phase 1: Extract ──────────────────────────────────────────────────────────
phase('Extract')

const changedFiles = await agent(
  `List files changed since ${SINCE} in ${ROOT} that are Python, TypeScript, Go, Rust, or Java.
Run: git -C ${ROOT} diff --name-only ${SINCE} HEAD 2>/dev/null || git -C ${ROOT} diff --name-only HEAD 2>/dev/null
Filter to .py, .ts, .go, .rs, .java extensions. Return as a JSON array of relative paths.
If no files changed or git is unavailable, return an empty array.`,
  { label: 'extract:changed-files', phase: 'Extract', schema: {
    type: 'object',
    required: ['files'],
    properties: { files: { type: 'array', items: { type: 'string' } } },
  }}
)

const files = (changedFiles && changedFiles.files) || []
log(`Changed files: ${files.length}`)

if (files.length === 0) {
  return { root: ROOT, investigation: INV, contracts_extracted: 0, conflicts: 0, skipped: true, reason: 'No relevant files changed' }
}

const MAX_FILES = 20
const targetFiles = files.slice(0, MAX_FILES)
if (files.length > MAX_FILES) {
  log(`Limiting to first ${MAX_FILES} of ${files.length} changed files`)
}

const extractResults = (await parallel(targetFiles.map(f => () =>
  agent(
    `Extract producer/consumer contracts from ${ROOT}/${f}.
${LANG_NOTE}

A contract is any point where this file defines or consumes a data structure that crosses
a boundary (HTTP response, message payload, database row, function argument).

Read the file completely. For each boundary, determine:
- entity: what is being serialized/deserialized (e.g., "POST /api/users response", "UserCreatedEvent", "users table row")
- role: "producer" if this file writes/sends/defines it; "consumer" if this file reads/receives/parses it
- fields: dict of {field_name: type_description} — use the EXACT field names from the code
- protocol: "JSON-HTTP", "MQTT", "gRPC", "Parquet", "SQLAlchemy", "Pydantic", "TypeScript-interface", etc.
- file: the relative file path

Look specifically for:
1. Pydantic models, dataclasses, TypeScript interfaces — these define producer contracts
2. json.loads(), JSON.parse(), from_dict(), model_validate() — these define consumer contracts
3. HTTP route return values with a schema (response_model=X in FastAPI)
4. Message publish calls with payload definitions
5. Database INSERT/SELECT with named columns

Return only concrete field names from the actual code (not guessed names).
If no contracts found, return contracts=[].`,
    { label: `extract:${f.split('/').pop()}`, phase: 'Extract', schema: CONTRACT_SCHEMA }
  ).then(r => r ? r.contracts || [] : [])
))).flat().filter(Boolean)

log(`Extracted ${extractResults.length} contracts from ${targetFiles.length} files`)

if (extractResults.length === 0) {
  return { root: ROOT, investigation: INV, contracts_extracted: 0, conflicts: 0, message: 'No contracts found in changed files' }
}

// ── Phase 2: Store + Cross-check ──────────────────────────────────────────────
phase('Store')

const storeResults = (await parallel(extractResults.map(contract => () =>
  agent(
    `Store this contract to Loci investigation "${INV}" and check for field drift.

Contract:
  entity:   ${contract.entity}
  role:     ${contract.role}
  fields:   ${JSON.stringify(contract.fields)}
  protocol: ${contract.protocol || ''}
  file:     ${contract.file || ''}

Steps:
1. Call mcp__loci__contract_declare with:
   investigation_id="${INV}"
   entity="${contract.entity}"
   role="${contract.role}"
   fields='${JSON.stringify(contract.fields)}'
   protocol="${contract.protocol || ''}"

2. For each field name in the contract, call mcp__loci__contract_check:
   investigation_id="${INV}"
   field_name="<field_name>"
   entity="${contract.entity}"

3. Return: finding_id (from declare), conflicts (from check calls), consistent (bool).`,
    { label: `store:${contract.entity.substring(0, 20)}`, phase: 'Store', schema: {
      type: 'object',
      properties: {
        finding_id: { type: 'string' },
        entity: { type: 'string' },
        role: { type: 'string' },
        conflicts: { type: 'array', items: { type: 'object' } },
        consistent: { type: 'boolean' },
        error: { type: 'string' },
      },
    }}
  ).then(r => r ? { ...r, contract } : null)
))).filter(Boolean)

const conflicts = storeResults.flatMap(r => r.conflicts || [])
  .filter(c => c && Object.keys(c).length > 0)

log(`Stored ${storeResults.filter(r => r.finding_id).length} contracts; detected ${conflicts.length} field conflicts`)

// ── Phase 3: Report ───────────────────────────────────────────────────────────
phase('Report')

if (conflicts.length > 0) {
  await agent(
    `Store ${conflicts.length} field drift conflict(s) to Loci investigation "${INV}" as high-severity gap findings.

For each conflict, call mcp__loci__investigation_store(
  investigation_id="${INV}",
  finding_type="gap",
  confidence="high",
  tags="contract-sync,field-drift,CA2-field-name-drift",
  text="Field drift detected: <stored_field> (stored contract for <entity>) conflicts with <new_field_name> in changed file — possible rename not propagated to consumer"
)

Conflicts: ${JSON.stringify(conflicts, null, 2)}`,
    { label: 'report:store-conflicts', phase: 'Report', model: 'haiku' }
  )
}

const summary = {
  root: ROOT,
  investigation: INV,
  since_commit: SINCE,
  files_examined: targetFiles.length,
  contracts_extracted: extractResults.length,
  contracts_stored: storeResults.filter(r => r.finding_id).length,
  conflicts_detected: conflicts.length,
  conflicts: conflicts.length > 0 ? conflicts : undefined,
  stored_entities: storeResults
    .filter(r => r.finding_id)
    .map(r => ({ entity: r.entity, role: r.role, finding_id: r.finding_id })),
}

if (conflicts.length > 0) {
  log(`CONFLICTS DETECTED: ${conflicts.length} field name drift(s) — stored as high-severity gap findings in investigation ${INV}`)
} else {
  log(`All ${summary.contracts_stored} contracts are consistent with stored declarations`)
}

return summary
