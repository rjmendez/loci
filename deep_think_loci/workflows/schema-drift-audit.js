export const meta = {
  name: 'schema-drift-audit',
  description: 'Hunt for structural contract drift across serialization boundaries: field names that change between layers, types that are valid in memory but fail on the wire, DB schemas out of sync with ORM models, and API responses that don\'t match client expectations. Patterns SD, SB, DC from the LLM hallucination taxonomy.',
  whenToUse: 'Run after any migration, API change, cross-service field rename, or addition of a new cross-language data path. Schema drift is invisible at unit-test level; it only shows in integration or production.',
  phases: [
    { title: 'Hunt', detail: '5 schema hunters in parallel — serialization types, field names, DB/ORM, API contracts, cross-service' },
    { title: 'Triage', detail: 'Adversarially verify each critical/high finding' },
    { title: 'Prioritize', detail: 'Rank by blast radius: silent wrong data > deserialization crash > query mismatch' },
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
          severity:        { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          category:        { type: 'string' },
          title:           { type: 'string' },
          producer_file:   { type: 'string' },
          consumer_file:   { type: 'string' },
          field_or_type:   { type: 'string' },
          detail:          { type: 'string' },
          evidence:        { type: 'string' },
          fix_recipe:      { type: 'string' },
          fix_effort:      { type: 'string', enum: ['trivial', 'small', 'medium', 'large'] },
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
    key: 'serialization_types',
    prompt: `Hunt for types that are valid in memory but fail at serialization boundaries in ${ROOT}.
${LANG_NOTE}

Mechanism (SB patterns): Data is constructed correctly in memory. When it crosses a wire
(JSON, DB write, HTTP response, message queue), it fails with a TypeError or produces
a wrong encoding — but only in production paths, not unit tests.

Common instances:
- numpy ndarray / torch.Tensor passed directly to json.dumps → TypeError
- datetime.utcnow() (naive) mixed with timezone-aware datetimes in DB → comparison fails
- Python Enum value passed as dict key to json.dumps → TypeError (Enum not serializable)
- decimal.Decimal in JSON → TypeError (not JSON serializable)
- bytes/bytearray in JSON → TypeError
- NaN or Infinity in JSON → JSON spec violation, parsers fail inconsistently

Steps:
1. Find json.dumps / json.encode / JSON.stringify / json.Marshal / serde_json::to_string call sites:
   grep -rn "json.dumps\|json.encode\|JSON.stringify\|json.Marshal\|serde_json::to" ${ROOT}

2. For each site, read what's being serialized. Does it contain:
   - Any numpy/torch tensor (look for imports of numpy, torch, tensorflow)
   - datetime objects not converted to .isoformat() or .timestamp()
   - Enum members not converted to .value
   - Decimal not converted to float/str
   - bytes not converted to base64 or hex string
   - NaN/Inf from float computations (division, log, exp on edge inputs)

3. Also check DB write paths: ORM model fields, SQLAlchemy column types, Mongoose schema types.
   Are there Python-side types that SQLAlchemy can't map?

category="SB-serialization-type"`,
  },
  {
    key: 'field_rename_drift',
    prompt: `Hunt for field names that were renamed in one layer but not updated in downstream consumers in ${ROOT}.
${LANG_NOTE}

Mechanism (SD1–SD3): A field is renamed during refactoring. The model/schema/ORM is updated.
But string-based accesses (dict lookups, JSON field names, CSV headers) in consumers still use
the old name. No error at import time — only wrong/missing values at runtime.

Steps:
1. Find recent field renames (if git is available):
   git -C ${ROOT} log --oneline --diff-filter=M -20 -- '*.py' '*.ts' '*.java' '*.go' 2>/dev/null | head -20
   git -C ${ROOT} diff HEAD~5 HEAD -- '*.py' '*.ts' '*.java' '*.go' 2>/dev/null | grep "^[-+].*\"\\w" | head -40

2. Build a field name inventory across the codebase:
   Find field definitions (ORM models, dataclasses, Pydantic models, TypeScript interfaces, Go structs)
   and string-based accesses (dict["field"], .get("field"), row.field_name, record["field"])

3. Look for field names that appear ONLY in accesses but not in any definition — these are
   candidates for "renamed on the definition side, not updated on the access side."

4. Check across language/service boundaries specifically — these are least likely to have IDE
   refactoring support catch the rename:
   Python model field → Go struct field → TypeScript response interface

category="SD-field-rename-drift"`,
  },
  {
    key: 'db_orm_mismatch',
    prompt: `Hunt for DB schema / ORM model mismatches in ${ROOT}.
${LANG_NOTE}

Mechanism (SD4/DC): The database has a column "user_id" with NOT NULL constraint.
The ORM model has the field as Optional[str]. Or a migration was added but not applied.
Or a column was renamed in the DB but not in the ORM. Results: wrong queries, silent nulls.

Steps:
1. Find ORM model definitions:
   grep -rn "class.*Model\|class.*Schema\|Column(\|db.Column\|@Column\|field(" ${ROOT}
   Also: SQLAlchemy declarative_base(), Django models.Model, TypeORM @Entity, Prisma schema

2. Find migration files:
   find ${ROOT} -name "*.sql" -o -name "migration*.py" -o -name "*_migration.*" | head -20
   grep -rn "ALTER TABLE\|ADD COLUMN\|DROP COLUMN\|RENAME COLUMN" in migration files

3. Cross-check: for each column in migrations, does the ORM model have a matching field
   with the same name, type, and nullable status?

4. Look for: columns that exist in migrations but not in ORM; ORM fields without corresponding
   columns; nullable=False in DB but Optional in ORM (or vice versa); type mismatches
   (DB: VARCHAR(255), ORM: Text — may cause truncation silently)

5. Also check query strings:
   grep -rn "SELECT.*FROM\|INSERT INTO\|UPDATE.*SET" ${ROOT}
   Do raw queries reference columns that exist in the ORM model?

category="SD-db-orm-mismatch"`,
  },
  {
    key: 'api_contract_drift',
    prompt: `Hunt for API response/request contract drift between server implementation and client expectations in ${ROOT}.
${LANG_NOTE}

Mechanism (SD/DC): The server adds a field to a response, removes a field, or changes a field's
type. Clients that access the old field name silently get None/undefined. No error.

Steps:
1. Find server-side response construction:
   grep -rn "return {.*}\|jsonify(\|json_response\|Response(json\|c.JSON(\|w.WriteHeader" ${ROOT}
   Note the field names in each response object.

2. Find client-side response consumption:
   grep -rn "response\.\|result\.\|data\.\|\.json()\|json.loads" ${ROOT}
   Note which field names clients access from responses.

3. Cross-reference server response fields vs client access fields:
   - Is there a field the client reads that the server no longer sends?
   - Is there a required field the server now sends that the client doesn't validate?
   - Are there optional fields the client assumes are always present?

4. If OpenAPI/Swagger specs exist: compare spec to server implementation.
   grep -rn "openapi.yaml\|swagger.json\|api_spec" ${ROOT}
   Are there endpoints where the spec doesn't match the actual implementation?

5. Also check event-driven contracts: Kafka message schemas, MQTT payload schemas,
   WebSocket message formats — wherever a message is emitted on one side and parsed on another.

category="SD-api-contract-drift"`,
  },
  {
    key: 'cross_service_types',
    prompt: `Hunt for type contract mismatches across service boundaries or language boundaries in ${ROOT}.
${LANG_NOTE}

Mechanism (DC/SB): Service A encodes a value as an integer. Service B decodes it as a string.
Or a timestamp is sent as epoch seconds (int) on one side, parsed as milliseconds on the other.
Silent wrong values — no deserialization error.

Steps:
1. Find cross-service data exchange points:
   - HTTP API call sites (where one service calls another): requests.get/post, fetch, axios, http.Get
   - Shared message formats (read from queue on one side, written on other)
   - Shared DB tables read by multiple services
   - Shared files (CSV, Parquet, JSON files written by one service, read by another)

2. For each exchange point, identify:
   - The type on the sending side (int, float, str, bool, enum, timestamp)
   - The type on the receiving side
   - Are they compatible? Could there be truncation, overflow, or semantic mismatch?

3. Specific patterns to check:
   - Unix timestamps: are they seconds or milliseconds? Both sides should agree.
   - Booleans: are they 0/1 or true/false? JSON vs query string encoding differs.
   - Enums: are they sent as int value or string name? Changed on one side?
   - Floats: NaN and Infinity serialize differently across JSON implementations.
   - IDs: string "123" vs integer 123 — some databases treat these differently.

category="SD-cross-service-type"`,
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
    `Adversarially verify this schema-drift finding. Try to REFUTE it.

Finding ID: ${f.id} | Category: ${f.category}
Title: ${f.title}
Field/Type: ${f.field_or_type || 'unknown'}
Producer: ${f.producer_file || 'unknown'} → Consumer: ${f.consumer_file || 'unknown'}
Detail: ${f.detail}
Evidence: ${f.evidence}

Steps:
1. Read the producer file. Confirm the exact field name/type it uses.
2. Read the consumer file. Confirm what it expects.
3. Is there a serializer/deserializer adapter between them that reconciles the difference?
4. Is there a migration or compatibility shim already in place?

Return is_real=true only if the mismatch is confirmed and unmitigated.`,
    { label: `triage:${f.id}`, phase: 'Triage', schema: VERDICT_SCHEMA }
  ).then(v => v ? { finding: f, verdict: v } : null)
))).filter(Boolean)

const confirmed = verdicts.filter(v => v.verdict.is_real)
log(`Triage: ${confirmed.length}/${highPriority.length} confirmed schema drift findings`)

// ── Phase 3: Prioritize ───────────────────────────────────────────────────────
phase('Prioritize')

const report = await agent(
  `Prioritize these schema-drift findings.

Ranking: silent wrong data (no error, wrong value) > deserialization crash > stale cache > query mismatch

CONFIRMED HIGH/CRITICAL (${confirmed.length}):
${JSON.stringify(confirmed.map(v => ({
  id: v.finding.id, category: v.finding.category, title: v.finding.title,
  field_or_type: v.finding.field_or_type, detail: v.finding.detail,
  fix_recipe: v.finding.fix_recipe, fix_effort: v.finding.fix_effort,
  severity: v.verdict.severity || v.finding.severity,
})), null, 2)}

MEDIUM/LOW (${allFindings.filter(f => f.severity === 'medium' || f.severity === 'low').length}):
${JSON.stringify(allFindings.filter(f => f.severity === 'medium' || f.severity === 'low').map(
  f => ({ id: f.id, category: f.category, title: f.title, detail: f.detail })
), null, 2)}

Produce: executive_summary, pr_bundle (priority order, fix instructions), silent_wrong_data (worst class).`,
  { label: 'prioritize', phase: 'Prioritize' }
)

if (INV && confirmed.length > 0) {
  await agent(
    `Store confirmed schema-drift findings to Loci investigation "${INV}".
For each call mcp__loci__investigation_store(investigation_id="${INV}", finding_type="observed",
confidence="high", tags="schema-drift-audit,category:<category>", text="<title>: <detail> | Fix: <fix_recipe>").
Findings: ${JSON.stringify(confirmed.map(v => v.finding), null, 2)}`,
    { label: 'store:loci', phase: 'Prioritize', model: 'haiku' }
  )
}

return { root: ROOT, findings_total: allFindings.length, confirmed_high_critical: confirmed.length, report }
