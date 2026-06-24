export const meta = {
  name: 'dependency-contract-audit',
  description: 'Hunt for breaking changes in imported dependencies: renamed APIs, removed exports, signature drift, type incompatibilities, and version pinning that has silently drifted. The class of bug where "it worked on my machine" because the local env has a different dep version than CI or production.',
  whenToUse: 'Run after any dependency update (pip install -U, npm update, cargo update), or when CI fails on a freshly provisioned machine with pinned versions that differ from the developer\'s. Dep contract bugs often only surface at runtime.',
  phases: [
    { title: 'Hunt', detail: '5 dependency hunters in parallel — renamed API, signature drift, removed export, type incompatibility, pinned version drift' },
    { title: 'Triage', detail: 'Adversarially verify each finding — is the API actually removed, or renamed with a compatibility shim?' },
    { title: 'Prioritize', detail: 'Rank by: removed export (import error) > renamed API (AttributeError at runtime) > signature drift > type mismatch' },
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
          package_name:    { type: 'string' },
          call_site_file:  { type: 'string' },
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
    key: 'renamed_api',
    prompt: `Hunt for call sites using method names that were renamed in a recent dependency update in ${ROOT}.
${LANG_NOTE}

Mechanism: A library releases a new version. A method is renamed (e.g., find() → get(),
connect() → open(), execute() → run()). The call site still uses the old name.
AttributeError at runtime; no error at import time.

Steps:
1. Find dependency definitions:
   cat ${ROOT}/requirements.txt ${ROOT}/pyproject.toml ${ROOT}/package.json 2>/dev/null | head -80
   Note version pins or ranges.

2. Find recently updated packages (if git is available):
   git -C ${ROOT} log --oneline --diff-filter=M -10 -- requirements*.txt pyproject.toml package.json 2>/dev/null

3. For key packages, check installed version vs pinned version:
   pip show <package> 2>/dev/null | grep Version
   npm list <package> 2>/dev/null | head -5

4. For packages with known breaking changes between versions, check call sites:
   Common patterns of renamed APIs:
   - SQLAlchemy 1.x → 2.x: session.execute() signature changed, session.Query removed
   - Django 3 → 4: ugettext → gettext, TEMPLATES DIRS format
   - FastAPI: response_model_exclude_unset param renamed
   - Click: standalone_mode vs result_callback
   grep -rn "session\.query\|ugettext\|standalone_mode" ${ROOT} --include="*.py"

5. Check CHANGELOG/BREAKING_CHANGES files in the dep's installed location:
   find $(pip show <key_package> 2>/dev/null | grep Location | awk '{print $2}') -name "CHANGELOG*" 2>/dev/null | head -3

category="DC-renamed-api"`,
  },
  {
    key: 'removed_export',
    prompt: `Hunt for imports of symbols that no longer exist in the installed version of a dependency in ${ROOT}.
${LANG_NOTE}

Mechanism: A symbol (class, function, constant) is imported from a package. In a newer
version of the package, the symbol was removed or moved to a different submodule.
ImportError at startup — hard to catch without actually running the import.

Steps:
1. Find all imports from third-party packages:
   grep -rn "^from \w\|^import \w" ${ROOT} --include="*.py" | grep -v "^from \." | head -60
   grep -rn "import.*from [\"']\w" ${ROOT} --include="*.ts" --include="*.js" | grep -v "from '\." | head -30

2. For each imported symbol from a key package, verify it exists in the installed version:
   python3 -c "from <package> import <symbol>" 2>&1
   If ImportError: the symbol was removed or moved.

3. Check for common moved symbols:
   - Python 3.10+: collections.Callable moved to collections.abc.Callable
   - Python 3.12: distutils removed (use setuptools)
   - SQLAlchemy 2.0: sqlalchemy.orm.session.Session.execute() signature changed
   - Pydantic v2: many v1 validators removed; BaseSettings moved to pydantic-settings

4. Check for deprecated import paths that were removed:
   grep -rn "from collections import.*Callable\|from distutils\|from typing import.*Union.*Optional" ${ROOT} --include="*.py"
   These are scheduled for removal in newer Python versions.

category="DC-removed-export"`,
  },
  {
    key: 'signature_drift',
    prompt: `Hunt for function calls with wrong number of positional arguments after a dependency update in ${ROOT}.
${LANG_NOTE}

Mechanism: A library function signature changes between versions (e.g., a positional
arg becomes keyword-only, or a new required arg is added). The call site passes args
in the old order. TypeError at runtime.

Steps:
1. Find function call patterns with multiple positional args to library functions:
   grep -rn "requests\.\(get\|post\|put\|patch\)(\|httpx\.\(get\|post\)(\|asyncio\.wait(" ${ROOT} --include="*.py"
   For each: check the installed version's signature.

2. Check asyncio API changes (Python 3.10+):
   asyncio.get_event_loop() deprecated → asyncio.get_running_loop()
   asyncio.coroutine() removed in 3.11
   loop parameter removed from many asyncio functions in 3.10

3. Find SQLAlchemy 2.0 signature breaks:
   grep -rn "session\.execute\|session\.query\|relationship(" ${ROOT} --include="*.py"
   SQLAlchemy 2.0 execute() requires text() wrapper; Query is legacy.

4. Find Pydantic v2 breaks:
   grep -rn "@validator\|\.dict()\|\.json()\|parse_obj\|parse_raw" ${ROOT} --include="*.py"
   Pydantic v2: .dict() → .model_dump(), .json() → .model_dump_json(),
   @validator → @field_validator, parse_obj → model_validate

5. For JavaScript: find common axios/fetch signature drift:
   grep -rn "axios\.\(get\|post\|put\)(\|fetch(" ${ROOT} --include="*.ts" --include="*.js"
   Check if the request config options match the installed version.

category="DC-signature-drift"`,
  },
  {
    key: 'version_pin_drift',
    prompt: `Hunt for packages where the pinned version in requirements/lock files differs from what is installed in ${ROOT}.
${LANG_NOTE}

Mechanism: requirements.txt pins package==1.2.3. A developer runs pip install without
--require-hashes or pip-sync. pip installs a different version (newer or older) that
satisfies a different constraint. The codebase runs with an untested version.

Steps:
1. Get pinned versions:
   cat ${ROOT}/requirements.txt ${ROOT}/requirements-prod.txt ${ROOT}/requirements-lock.txt 2>/dev/null | grep "==" | head -30
   cat ${ROOT}/package-lock.json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); [print(f'{k}: {v[\"version\"]}') for k,v in d.get('packages',{}).items() if k]" 2>/dev/null | head -30

2. Get installed versions:
   pip list --format=columns 2>/dev/null | head -40
   npm list --depth=0 2>/dev/null | head -30

3. Cross-reference pinned vs installed: flag any package where installed ≠ pinned.
   Pay special attention to packages with breaking changes between minor versions:
   - sqlalchemy, pydantic, django, fastapi, requests, boto3, pandas, numpy

4. Check if a Pipfile.lock / poetry.lock is present but not being used:
   ls ${ROOT}/Pipfile.lock ${ROOT}/poetry.lock ${ROOT}/uv.lock 2>/dev/null
   If present: is the CI/CD using the lock file (pip-sync, poetry install, uv sync)?
   Or is it using pip install -r requirements.txt (ignores the lock)?

5. Find dev-only packages that leaked into production deps:
   grep -rn "pytest\|coverage\|black\|mypy\|ruff\|pre-commit" ${ROOT}/requirements.txt 2>/dev/null
   These should be in requirements-dev.txt, not requirements.txt.

category="DC-version-pin-drift"`,
  },
  {
    key: 'type_incompatibility',
    prompt: `Hunt for type incompatibilities introduced by dependency version drift in ${ROOT}.
${LANG_NOTE}

Mechanism: A @types package for TypeScript or a type stub for Python is at a different
version than the runtime package. Generated code relies on the type definitions; at
runtime, the actual types are different. No type error at compile time; wrong behavior
at runtime.

Steps:
1. For TypeScript projects — check @types version vs runtime package:
   cat ${ROOT}/package.json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); deps={**d.get('dependencies',{}), **d.get('devDependencies',{})}; [print(k,v) for k,v in deps.items() if k.startswith('@types/') or k in ['typescript']]" 2>/dev/null
   For each @types/foo: is its major version compatible with foo?

2. For Python — check type stubs:
   pip list 2>/dev/null | grep types-  # e.g. types-requests, types-boto3
   For each stub: is its version pinned to match the runtime package?

3. Find Optional vs required type drift:
   grep -rn "Optional\[\|Union\[.*None\]" ${ROOT} --include="*.py"
   Did a recent update change a field from Optional to required? Check model constructors.

4. Find return type changes:
   A function used to return List[str]; in the new version it returns Generator[str].
   Code that calls list() on the result still works; code that calls len() breaks.
   Check CHANGELOG for "return type changed" entries.

5. Find Literal type drift:
   grep -rn "Literal\[" ${ROOT} --include="*.py"
   If a Literal enum set was updated in the library (e.g., Literal["v1", "v2"] → added "v3"),
   code that handles all Literal values exhaustively may now have an unhandled case.

category="DC-type-incompatibility"`,
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
    `Adversarially verify this dependency contract finding. Try to REFUTE it.
Default to is_real=false if you cannot confirm the API actually changed.

Finding ID: ${f.id} | Category: ${f.category}
Package: ${f.package_name || 'unknown'}
Call site: ${f.call_site_file || 'unknown'}
Title: ${f.title}
Detail: ${f.detail}
Evidence: ${f.evidence}

Steps:
1. Check the actual installed version of the package: pip show <pkg> or npm list <pkg>
2. Check the package changelog for the reported version range.
3. Does the installed version actually have the reported breaking change?
4. Is there a compatibility shim or deprecation warning that keeps the old API working?
5. Is there a wrapper in the codebase that abstracts the breaking change?

Return is_real=true only if the API contract break is confirmed in the installed version.`,
    { label: `triage:${f.id}`, phase: 'Triage', schema: VERDICT_SCHEMA }
  ).then(v => v ? { finding: f, verdict: v } : null)
))).filter(Boolean)

const confirmed = verdicts.filter(v => v.verdict.is_real)
log(`Triage: ${confirmed.length}/${highPriority.length} confirmed dependency breaks`)

// ── Phase 3: Prioritize ───────────────────────────────────────────────────────
phase('Prioritize')

const report = await agent(
  `Prioritize these dependency contract findings.

Ranking: removed export (fails at import = startup crash) > renamed API (AttributeError at first call)
       > signature drift (TypeError when called with args) > type incompatibility (wrong behavior)
       > version pin drift (latent risk)

CONFIRMED HIGH/CRITICAL (${confirmed.length}):
${JSON.stringify(confirmed.map(v => ({
  id: v.finding.id, category: v.finding.category, title: v.finding.title,
  package_name: v.finding.package_name, detail: v.finding.detail,
  fix_recipe: v.finding.fix_recipe, fix_effort: v.finding.fix_effort,
  severity: v.verdict.severity || v.finding.severity,
})), null, 2)}

MEDIUM/LOW (${allFindings.filter(f => f.severity === 'medium' || f.severity === 'low').length}):
${JSON.stringify(allFindings.filter(f => f.severity === 'medium' || f.severity === 'low').map(
  f => ({ id: f.id, category: f.category, title: f.title })
), null, 2)}

Produce: executive_summary, pr_bundle (fix each break), startup_crash_risk (findings
that would prevent the application from starting at all).`,
  { label: 'prioritize', phase: 'Prioritize' }
)

if (INV && confirmed.length > 0) {
  await agent(
    `Store confirmed dependency contract findings to Loci investigation "${INV}".
For each call mcp__loci__investigation_store(investigation_id="${INV}", finding_type="observed",
confidence="high", tags="dependency-contract-audit,category:<category>", text="<title>: <detail> | Fix: <fix_recipe>").
Findings: ${JSON.stringify(confirmed.map(v => v.finding), null, 2)}`,
    { label: 'store:loci', phase: 'Prioritize', model: 'haiku' }
  )
}

return { root: ROOT, findings_total: allFindings.length, confirmed_high_critical: confirmed.length, report }
