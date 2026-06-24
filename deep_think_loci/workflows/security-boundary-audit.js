export const meta = {
  name: 'security-boundary-audit',
  description: 'Hunt for security boundary failures: auth checks in the wrong layer, secrets leaking across trust boundaries, missing authorization guards on privileged operations, injection surfaces, and CORS misconfigurations. Patterns PB, SB from the LLM hallucination taxonomy.',
  whenToUse: 'Run after adding any new route, auth middleware, privileged operation, external call, or config that handles secrets. LLMs frequently place auth checks in service code when they belong in middleware, and generate SQL/shell calls without parameterization.',
  phases: [
    { title: 'Hunt', detail: '5 security hunters in parallel — auth layer, secret scope, missing guard, injection surface, CORS' },
    { title: 'Triage', detail: 'Adversarially verify each critical/high finding — is there a guard at a higher layer the hunter missed?' },
    { title: 'Prioritize', detail: 'Rank by: injection > missing auth > secret leak > CORS misconfiguration' },
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
    key: 'auth_layer_mismatch',
    prompt: `Hunt for authentication and authorization logic placed in the wrong architectural layer in ${ROOT}.
${LANG_NOTE}

Mechanism: Auth/authz should live in middleware or gateway — one place, all routes protected.
When auth checks are placed in individual service functions, it's easy to add a new route
and forget the check. LLMs generate service code and include the auth check inline,
missing the architectural requirement that protection belongs at the boundary.

Steps:
1. Find route handlers / endpoint definitions:
   grep -rn "@app\.route\|@router\.\|@app\.get\|@app\.post\|@app\.put\|@app\.delete\|func.*Handler\|\.HandleFunc(" ${ROOT}
   Also: Flask, FastAPI, Express, Django URL patterns.

2. For each route, check if there is an auth decorator or middleware applied:
   Look for: @login_required, @require_auth, @jwt_required, Depends(get_current_user),
   middleware.use(auth), app.use(requireAuth), or an explicit token check in the handler.

3. Flag routes that perform sensitive operations (update, delete, create, admin actions)
   without any auth decorator or explicit token verification in the handler body.

4. Check middleware configuration: is the auth middleware applied to ALL routes, or
   only some? Are there any route groups that bypass the middleware?
   grep -rn "app\.use\|middleware\|Middleware\|exclude.*auth\|skip.*auth\|no_auth\|public_route" ${ROOT}

5. Find admin or privileged endpoints accessible without additional authorization:
   grep -rn '"/admin\|"/internal\|"/management\|"/debug\|"/metrics' ${ROOT}
   These should require elevated permissions (is_staff, is_admin, IP allowlist).

category="PB-auth-layer-mismatch"`,
  },
  {
    key: 'secret_scope_leak',
    prompt: `Hunt for secrets and credentials that escape their intended scope in ${ROOT}.
${LANG_NOTE}

Mechanism: A secret (API key, password, token) is passed to a log statement, included
in an error response, serialized in a model that's returned over HTTP, or stored in an
env var accessible to the wrong process. LLMs include secrets in helpful debug logs.

Steps:
1. Find secrets in log statements:
   grep -rn "logger\.\(info\|debug\|warning\|error\).*\(key\|password\|secret\|token\|api_key\|auth\)" ${ROOT} --include="*.py"
   grep -rn "console\.\(log\|error\|warn\).*\(key\|password\|secret\|token\)" ${ROOT} --include="*.ts" --include="*.js"
   For each: is the secret value being logged (not just its name)?

2. Find secrets in HTTP responses or serialized models:
   grep -rn "password\|secret\|api_key\|private_key\|token" ${ROOT}
   For each: is this field included in a Pydantic model / TypeScript interface / struct
   that is returned over HTTP? Does the serializer exclude it?

3. Find secrets in exception messages:
   grep -rn "raise.*\(password\|secret\|key\|token\)\|Exception.*f\".*{.*key" ${ROOT}
   Exception messages may propagate to clients in debug mode.

4. Find hardcoded secrets:
   grep -rn "password\s*=\s*[\"'][^\"']\|api_key\s*=\s*[\"']\|SECRET\s*=\s*[\"']" ${ROOT}
   Also check: git history for any recently removed hardcoded secrets (may still be in history).

5. Find secrets passed in URL parameters (logged by servers):
   grep -rn "requests\.get.*\(api_key\|token\|secret\)\|fetch.*\?.*token=\|?api_key=" ${ROOT}
   Credentials in URLs appear in access logs of every intermediate proxy.

category="SB-secret-scope-leak"`,
  },
  {
    key: 'missing_authorization_guard',
    prompt: `Hunt for privileged operations that execute without checking the caller's permissions in ${ROOT}.
${LANG_NOTE}

Mechanism: A delete, bulk-update, admin, or cross-tenant operation has no permission
check before it executes. LLMs add the happy-path logic and omit the authz check.

Steps:
1. Find dangerous operation endpoints:
   grep -rn "\.delete\(\|DELETE\|bulk_update\|force_delete\|purge\|truncate\|drop_table\|admin" ${ROOT}
   For each: is there a permission check BEFORE the destructive operation?

2. Find cross-tenant data access:
   grep -rn "user_id\|tenant_id\|org_id\|account_id" ${ROOT}
   For each query: does it filter by the current user's ID? Or could user A query user B's data?
   Pattern: db.query(Record).filter_by(id=record_id)  ← missing: AND user_id=current_user.id

3. Find operations that should require elevated roles:
   grep -rn "role.*admin\|is_admin\|is_staff\|permission.*required\|has_permission\|can_.*(" ${ROOT}
   For each role-protected operation: is the check applied before the action?
   Is there a test that verifies a non-admin user is rejected?

4. Find IDOR (insecure direct object reference) patterns:
   Endpoints that accept an ID parameter and look up a record without verifying ownership:
   grep -rn "get_object_or_404\|find_by_id\|get(id=\|findById\|getById" ${ROOT}
   For each: does the lookup also verify the current user owns/has access to that record?

category="PB-missing-auth-guard"`,
  },
  {
    key: 'injection_surface',
    prompt: `Hunt for injection vulnerabilities in ${ROOT} — SQL, shell, template, LDAP.
${LANG_NOTE}

Mechanism: User-controlled input is concatenated into a query string or shell command
without parameterization. LLMs frequently use f-strings in SQL because they look readable.

Steps:
1. Find SQL injection:
   grep -rn "f\"SELECT\|f'SELECT\|f\"INSERT\|f\"UPDATE\|f\"DELETE\|\"SELECT.*format(\|'SELECT.*%" ${ROOT}
   Also: string concatenation with +: "SELECT * FROM users WHERE id = " + user_id
   Correct pattern: cursor.execute("SELECT ... WHERE id = %s", (user_id,))

2. Find shell injection:
   grep -rn "subprocess\.\(Popen\|call\|run\|check_output\).*shell=True" ${ROOT}
   grep -rn "os\.system(\|os\.popen(\|exec(\|eval(" ${ROOT}
   For each: is user input involved in the command string?

3. Find template injection:
   grep -rn "Template(\|Jinja2\|render_template_string\|\.render(" ${ROOT}
   For each: is user input passed directly into the template string (not as a variable)?
   Pattern: Template(user_input).render()  ← injection; Template("{{ var }}").render(var=user_input)  ← safe

4. Find LDAP injection:
   grep -rn "ldap\|LDAP\|ldap3\|python-ldap" ${ROOT}
   For each LDAP filter: is user input escaped with ldap3.utils.escape_filter_chars()?

5. Find NoSQL injection (MongoDB):
   grep -rn "find(\|find_one(\|aggregate(" ${ROOT}
   For each: is a user-controlled dict used as a filter? MongoDB operators in user input
   ($where, $regex, $ne) can bypass intended filters.

category="SB-injection-surface"`,
  },
  {
    key: 'cors_misconfiguration',
    prompt: `Hunt for CORS misconfigurations in ${ROOT}.
${LANG_NOTE}

Mechanism: CORS headers allow cross-origin requests from origins that should not be
trusted. The worst case: Access-Control-Allow-Origin: * combined with
Access-Control-Allow-Credentials: true (forbidden by spec, but silently accepted by
some browsers). LLMs add CORS headers to "fix the browser error" without understanding
the security implications.

Steps:
1. Find CORS configuration:
   grep -rn "CORS\|cors\|Access-Control-Allow-Origin\|allow_origins\|origins=" ${ROOT}
   Note the configured origins.

2. Flag wildcard origins:
   grep -rn "allow_origins.*\*\|\*.*allow_origins\|Access-Control-Allow-Origin.*\*" ${ROOT}
   Wildcard origin + credentials is a critical vulnerability.

3. Find overly broad origin lists:
   Does the allowed origin list include localhost? localhost CORS in production
   allows any local attacker (running on the same machine) to make credentialed
   requests to the production API.

4. Find missing CORS configuration on sensitive endpoints:
   API endpoints that set session cookies or return auth tokens need strict CORS.
   If the framework's global CORS applies to public endpoints and auth endpoints
   the same way, auth endpoints may be more permissive than intended.

5. Find CORS applied only at the application layer when a reverse proxy also
   sets CORS headers — this can result in duplicate headers or the proxy's
   headers overriding the application's (or vice versa).
   grep -rn "nginx\|proxy_pass\|add_header.*Access-Control" ${ROOT}

category="SB-cors-misconfiguration"`,
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
    `Adversarially verify this security finding. Try to REFUTE it.
Default to is_real=false if you cannot confirm the vulnerability by reading the actual code.

Finding ID: ${f.id} | Category: ${f.category}
File: ${f.file || 'unknown'} (${f.line_hint || '?'})
Title: ${f.title}
Detail: ${f.detail}
Evidence: ${f.evidence}

Steps:
1. Read the actual code at the location described.
2. For auth: is there a higher-layer middleware or decorator that protects this route?
3. For injection: is the user input actually reaching the dangerous call, or is it
   sanitized/validated before? Is there a WAF or ORM that prevents the injection?
4. For secrets: is the value actually secret, or is it a public identifier?
5. For CORS: does the application actually use credentials with the wildcard origin?

Return is_real=true only if the security vulnerability is confirmed and unmitigated.`,
    { label: `triage:${f.id}`, phase: 'Triage', schema: VERDICT_SCHEMA }
  ).then(v => v ? { finding: f, verdict: v } : null)
))).filter(Boolean)

const confirmed = verdicts.filter(v => v.verdict.is_real)
log(`Triage: ${confirmed.length}/${highPriority.length} confirmed security issues`)

// ── Phase 3: Prioritize ───────────────────────────────────────────────────────
phase('Prioritize')

const report = await agent(
  `Prioritize these security findings.

Ranking: injection (RCE/data breach) > missing auth guard (unauthorized access)
       > secret leak (credentials exposed) > CORS misconfiguration (CSRF risk)

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

Produce: executive_summary, pr_bundle (priority order, with exact fix instructions),
critical_path (any finding that allows unauthenticated access to data or RCE).`,
  { label: 'prioritize', phase: 'Prioritize' }
)

if (INV && confirmed.length > 0) {
  await agent(
    `Store confirmed security findings to Loci investigation "${INV}".
For each call mcp__loci__investigation_store(investigation_id="${INV}", finding_type="observed",
confidence="high", tags="security-boundary-audit,category:<category>", text="<title>: <detail> | Fix: <fix_recipe>").
Findings: ${JSON.stringify(confirmed.map(v => v.finding), null, 2)}`,
    { label: 'store:loci', phase: 'Prioritize', model: 'haiku' }
  )
}

return { root: ROOT, findings_total: allFindings.length, confirmed_high_critical: confirmed.length, report }
