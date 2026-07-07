export const meta = {
  name: 'dead-code-audit',
  description: 'Hunt for dead and zombie code: exported symbols with no callers, feature flags that always evaluate the same branch, unreachable code after return/throw, packages imported nowhere, and stale TODO/FIXME comments that mark code intended for removal. Dead code enlarges the attack surface, confuses maintainers, and misleads LLM code generation.',
  whenToUse: 'Run before major refactors, when onboarding to a codebase, or after removing a feature. Dead code is invisible to linters and test suites — it accumulates silently and is never exercised, so its bugs are also never caught.',
  phases: [
    { title: 'Hunt', detail: '5 dead-code hunters in parallel — unreferenced exports, always-on/off flags, unreachable branches, unused dependencies, tombstoned comments' },
    { title: 'Triage', detail: 'Adversarially verify each high finding — is it actually called via dynamic import, reflection, or external consumer?' },
    { title: 'Prioritize', detail: 'Rank by: unused dependency (security surface) > unreferenced export (dead API, confusion risk) > always-on/off flag (complexity without benefit) > unreachable branch > tombstoned comment' },
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

// ── Shared schemas ────────────────────────────────────────────────────────────
const FINDING_SCHEMA = {
  type: 'object',
  required: ['findings', 'summary'],
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'file', 'line', 'severity', 'title', 'detail', 'fix'],
        properties: {
          id:       { type: 'string' },
          file:     { type: 'string' },
          line:     { type: 'number' },
          severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          title:    { type: 'string' },
          detail:   { type: 'string' },
          fix:      { type: 'string' },
          last_modified: { type: 'string' },
        },
      },
    },
    summary: { type: 'string' },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  required: ['finding_id', 'confirmed', 'reason'],
  properties: {
    finding_id: { type: 'string' },
    confirmed:  { type: 'boolean' },
    reason:     { type: 'string' },
    dynamic_usage: { type: 'string' },
  },
}

// ── Hunters ───────────────────────────────────────────────────────────────────
const HUNTERS = [
  {
    key: 'unreferenced_export',
    prompt: `You are a dead-code hunter specializing in exported symbols with no callers.

${LANG_NOTE}

Codebase root: ${ROOT}

Hunt for exported functions, classes, types, and constants that are never imported or called anywhere in the codebase.

Search approach:
1. List all exports:
   TypeScript: \`grep -rn "^export function\\|^export class\\|^export const\\|^export type\\|^export interface\\|^export enum" ${ROOT} --include="*.ts" --include="*.tsx" | grep -v "node_modules\\|dist\\|.next" | head -50\`
   Python: \`grep -rn "^def \\|^class \\|^[A-Z_]\\+ = " ${ROOT} --include="*.py" | grep -v "__pycache__\\|test_\\|_test.py" | head -50\`
2. For each exported symbol, search for any import or usage:
   \`grep -rn "SYMBOL_NAME" ${ROOT} --include="*.ts" --include="*.tsx" --include="*.py" | grep -v "the file that defines it" | grep -v "node_modules" | wc -l\`
   If the count is 0 (or 1 counting only the definition), the symbol is unreferenced.
3. Pay special attention to:
   - Functions exported from \`lib/\` or \`utils/\` files with no consumer
   - Re-exported types that are never used by any component
   - Python functions in \`scripts/\` or helper modules with no caller
   - Classes with a \`__init__\` but no instantiation site
4. False positive checks:
   - Entry points (\`main\`, \`handler\`, \`app\`, \`cli\`) are used by runners, not imports
   - Types/interfaces used only in type-position (TypeScript) still count as used
   - Public API symbols in a library package are expected to have external consumers

Severity: high if the dead export is in a security-sensitive or auth-related module (dead auth helper is confusing); medium otherwise.

Return findings with: file, line, severity, title (include symbol name), detail (how you verified it has no callers), fix (delete or unexport).`,
  },
  {
    key: 'always_on_off_flag',
    prompt: `You are a dead-code hunter specializing in feature flags that are permanently enabled or disabled.

${LANG_NOTE}

Codebase root: ${ROOT}

Hunt for feature flags, environment variable checks, and conditional branches that always evaluate to the same value — effectively dead code on one branch.

Search approach:
1. Find feature flag patterns:
   \`grep -rn "FEATURE_\\|FF_\\|ENABLE_\\|DISABLE_\\|FLAG_\\|isEnabled\\|featureFlag\\|getFeature" ${ROOT} --include="*.ts" --include="*.tsx" --include="*.py" | grep -v "node_modules\\|test\\|__pycache__" | head -40\`
2. Check if env var defaults make a branch always-true or always-false:
   - \`process.env.FEATURE_X === 'true'\` where \`FEATURE_X\` is always set to \`'true'\` in all env files
   - \`os.environ.get('ENABLE_X', 'false') == 'true'\` where 'false' default means the feature is off in all environments and the env var is never set
3. Look for hardcoded conditional bypasses:
   \`grep -rn "if (false\\|if (true\\|if (0\\|if (1\\||| true\\|&& false" ${ROOT} --include="*.ts" --include="*.py" | grep -v "node_modules\\|test" | head -20\`
4. Check all env.example / .env.* files for vars that are always empty or always the same value across all environments
5. Feature flags that have been 100% rolled out but never cleaned up — look for flags with no corresponding env var declaration in any .env file

Severity: high if the always-false branch contains security logic (bypassed auth check) or the always-true branch removes a safety guard; medium for product feature flags; low for debug flags.

Return findings with: file, line, severity, title (flag name and which branch is dead), detail (explain why the branch is always on/off), fix (delete the dead branch and clean up the flag).`,
  },
  {
    key: 'unreachable_branch',
    prompt: `You are a dead-code hunter specializing in unreachable code paths.

${LANG_NOTE}

Codebase root: ${ROOT}

Hunt for code that can never execute: statements after return/throw, conditions that are always true/false due to type constraints, and exhaustive switch cases with a dead default.

Search approach:
1. Code after return/throw/break in the same block:
   \`grep -rn "return\\|throw\\|break" ${ROOT} --include="*.ts" --include="*.tsx" --include="*.py" | grep -v "node_modules\\|test\\|__pycache__" | head -30\`
   For each, check if there are non-comment statements on the next line(s) before the closing brace
2. TypeScript: narrowed types that make an else branch impossible:
   - Union type where one variant is handled and the remaining \`else\` branch checks for a type that can't exist
   - \`if (x === 'a' || x === 'b') { ... } else { /* x can never be anything else */ }\`
3. Python: \`else\` clauses on for loops that always break, or \`except\` clauses on operations that can't raise that exception
4. Switch/match statements with a \`default\` that can never be reached because all enum values are covered
5. Async functions where a Promise is awaited inside a try block but the catch only handles a specific error type that the awaited function doesn't throw
6. Early-return guards where the condition is always true (making the rest of the function dead):
   \`grep -rn "if (!\\|if (!" ${ROOT} --include="*.ts" --include="*.py" | grep -v "node_modules\\|test" | head -20\`

Severity: high if the unreachable code contains error handling (the error path is silently disabled); medium for dead feature code; low for style issues.

Return findings with: file, line, severity, title, detail (why the code is unreachable), fix (delete the unreachable block).`,
  },
  {
    key: 'unused_dependency',
    prompt: `You are a dead-code hunter specializing in unused package dependencies.

${LANG_NOTE}

Codebase root: ${ROOT}

Hunt for packages declared in package.json / requirements.txt / pyproject.toml that are never imported or used in the codebase.

Search approach:
1. List all declared dependencies:
   \`cat ${ROOT}/package.json 2>/dev/null | grep -A 200 '"dependencies"' | grep '"@\\|"[a-z]' | head -40\`
   \`cat ${ROOT}/requirements.txt 2>/dev/null | grep -v "^#\\|^$" | head -40\`
   \`cat ${ROOT}/pyproject.toml 2>/dev/null | grep -A 50 '\\[tool.poetry.dependencies\\]\\|\\[project\\]' | head -40\`
2. For each declared package, search for any import:
   TypeScript: \`grep -rn "from '[package]'\\|require('[package]')" ${ROOT} --include="*.ts" --include="*.tsx" | grep -v "node_modules" | wc -l\`
   Python: \`grep -rn "^import [package]\\|^from [package]" ${ROOT} --include="*.py" | grep -v "__pycache__\\|test_" | wc -l\`
3. Flag packages with 0 imports. Common false positives to check:
   - CLI tools used in package.json scripts (not imported, but used via exec)
   - Type-only packages (\`@types/*\`) used by TypeScript compiler but not imported
   - Peer dependencies that are required by another dep but not directly imported
   - Build tools (postcss plugins, webpack loaders) configured in config files, not imported
4. Also look for devDependencies that are used in production code (wrong section):
   \`grep -rn "from 'vitest'\\|from 'jest'\\|import.*test" ${ROOT}/app ${ROOT}/lib ${ROOT}/src 2>/dev/null --include="*.ts" | head -10\`

Severity: high if the unused package has known CVEs or is a large transitive dependency chain; medium otherwise (unused deps still enlarge the attack surface and slow installs).

Return findings with: file (package.json or requirements.txt), line (approximate), severity, title (package name), detail (verified 0 imports, note any false-positive check performed), fix (remove from dependency list, run install to verify).`,
  },
  {
    key: 'tombstoned_comment',
    prompt: `You are a dead-code hunter specializing in stale TODO/FIXME/HACK/DEPRECATED comments that mark code intended for removal or replacement.

${LANG_NOTE}

Codebase root: ${ROOT}

Hunt for comments that indicate technical debt that was meant to be resolved but has been left in place, and code blocks that are commented-out rather than deleted.

Search approach:
1. Find all TODO/FIXME/HACK/DEPRECATED/XXX/TEMP comments:
   \`grep -rn "TODO\\|FIXME\\|HACK\\|DEPRECATED\\|XXX\\|TEMP:\\|REMOVE\\|DELETE ME\\|\\\\btemp\\b" ${ROOT} --include="*.ts" --include="*.tsx" --include="*.py" --include="*.js" | grep -v "node_modules\\|dist\\|\\.next\\|__pycache__" | head -50\`
2. For each, check the git log to estimate age:
   \`git -C ${ROOT} log --follow --format="%ar %s" -1 -- <file> 2>/dev/null | head -1\`
   Comments in files last modified more than 6 months ago and still marked TODO are tombstones.
3. Find large commented-out code blocks (3+ consecutive commented lines):
   \`grep -rn "^\\s*#\\s*[a-z]\\|^\\s*//\\s*[a-z]" ${ROOT} --include="*.ts" --include="*.py" | grep -v "node_modules\\|dist" | head -30\`
   Look for blocks where 3+ lines in a row are comments that contain code-like content (assignments, function calls, etc.)
4. Find \`@deprecated\` annotations with no corresponding removal plan:
   \`grep -rn "@deprecated\\|@Deprecated\\|DeprecationWarning" ${ROOT} --include="*.ts" --include="*.py" | grep -v "node_modules\\|test" | head -20\`
5. Flag \`pass\`/\`...\` stubs in Python or empty function bodies in TypeScript that have a TODO explaining what should be implemented

Severity: high if the TODO marks a security gap ("TODO: add auth check") or a data correctness issue ("FIXME: this gives wrong results for edge case X"); medium for missing features; low for cleanup items.

Return findings with: file, line, severity, title (include the comment text), detail (age if determinable, what it was blocking on), fix (either implement what the comment describes or delete the dead code/comment).`,
  },
]

// ── Phase 1: Hunt ─────────────────────────────────────────────────────────────
phase('Hunt')
const huntResults = await parallel(HUNTERS.map(h => () =>
  agent(h.prompt, { label: `hunt:${h.key}`, phase: 'Hunt', schema: FINDING_SCHEMA })
))

const allFindings = huntResults
  .filter(Boolean)
  .flatMap((r, i) => (r.findings || []).map(f => ({
    ...f,
    id: `${HUNTERS[i].key}-${f.id || Math.random().toString(36).slice(2, 7)}`,
    hunter: HUNTERS[i].key,
  })))

log(`Hunt complete: ${allFindings.length} raw findings across ${HUNTERS.length} hunters`)

if (!allFindings.length) {
  log('No findings — codebase looks clean for dead code.')
  return { root: ROOT, findings_total: 0, confirmed_high_critical: 0, report: null }
}

// ── Phase 2: Triage ───────────────────────────────────────────────────────────
phase('Triage')
const highCritical = allFindings.filter(f => f.severity === 'critical' || f.severity === 'high')

const triageResults = await parallel(highCritical.map(f => () =>
  agent(`You are an adversarial code reviewer. Your job is to REFUTE this dead-code finding if possible.

Finding: ${f.title}
File: ${f.file}:${f.line}
Detail: ${f.detail}

Codebase root: ${ROOT}

Try to find evidence this code is NOT actually dead:
- Is the exported symbol used via dynamic import() or require() with a variable module name?
- Is it referenced via reflection, eval, or string-based dispatch (e.g., obj[methodName]())?
- Is it an entry point called by an external runner (CLI, test harness, cloud function trigger)?
- Is it a type/interface used only in type-position that TypeScript still needs?
- Is the feature flag actually toggled on in a specific deployment environment not covered by .env.example?
- Is the "unreachable" branch actually reachable via a type assertion or runtime polymorphism?
- Is the commented-out code intentionally kept as documentation of a rejected approach?

Read the file and any related files carefully before deciding. Default to confirmed=true if you cannot find clear evidence the code is live.`,
    { label: `triage:${f.id}`, phase: 'Triage', schema: VERDICT_SCHEMA }
  )
))

const confirmed = highCritical.filter((f, i) => {
  const v = triageResults[i]
  return !v || v.confirmed !== false
})
const mediumLow = allFindings.filter(f => f.severity !== 'critical' && f.severity !== 'high')

log(`Triage complete: ${confirmed.length}/${highCritical.length} high confirmed, ${mediumLow.length} medium/low`)

// ── Phase 3: Prioritize ───────────────────────────────────────────────────────
phase('Prioritize')

const SEVERITY_RANK = { critical: 4, high: 3, medium: 2, low: 1 }
const HUNTER_RANK   = { unused_dependency: 5, unreferenced_export: 4, always_on_off_flag: 3, unreachable_branch: 2, tombstoned_comment: 1 }

const ranked = [...confirmed, ...mediumLow].sort((a, b) => {
  const sevDiff = (SEVERITY_RANK[b.severity] || 0) - (SEVERITY_RANK[a.severity] || 0)
  if (sevDiff !== 0) return sevDiff
  return (HUNTER_RANK[b.hunter] || 0) - (HUNTER_RANK[a.hunter] || 0)
})

const report = await agent(`You are a senior software engineer conducting a dead-code audit. Synthesize these findings into an actionable cleanup report.

Confirmed high findings (${confirmed.length}):
${JSON.stringify(confirmed, null, 2)}

Medium/low findings (${mediumLow.length}):
${JSON.stringify(mediumLow, null, 2)}

Produce:
1. executive_summary: 3-4 sentences. What is the overall dead-code burden? Which findings increase attack surface or will mislead future LLM code generation?
2. pr_bundle: Group findings into logical PRs. For each PR: title, priority (1=highest), addresses (finding IDs), rationale, instructions. Order by: unused dep (security/supply chain) > unreferenced export (confusion risk) > always-on/off flag > unreachable branch > tombstones.
3. llm_generation_risk: Which dead-code findings will confuse future LLM code generation? (Dead exports look like valid APIs; dead flags look like valid configuration knobs; commented-out code looks like an implementation pattern to follow.)

Be specific: include file paths, symbol names, and exact deletion commands where possible.`,
    { label: 'prioritize', phase: 'Prioritize' }
  )

// ── Phase 4: Store to Loci ────────────────────────────────────────────────────
if (INV) {
  phase('Store')
  await parallel(ranked.slice(0, 20).map(f => () =>
    agent(`Store this dead-code finding to the Loci investigation using the mcp__loci__investigation_store tool.

Investigation ID: ${INV}
Finding:
- title: ${f.title}
- file: ${f.file}:${f.line}
- severity: ${f.severity}
- record_type: observed
- confidence: medium
- text: ${f.detail} FIX: ${f.fix}
- tags: dead-code-audit,${f.hunter},${f.severity}

Call mcp__loci__investigation_store with these values. Return the finding_id from the result.`,
      { label: `store:${f.id}`, phase: 'Store' }
    )
  ))
}

return {
  root: ROOT,
  findings_total: allFindings.length,
  confirmed_high_critical: confirmed.length,
  report,
}
