export const meta = {
  name: 'config-drift-audit',
  description: 'Hunt for configuration drift between what the code expects and what is deployed: env vars used in code but not declared in manifests, production-unsafe defaults, feature flag mismatches, hardcoded endpoints, and services missing from deployment config. Patterns CD from the LLM hallucination taxonomy.',
  whenToUse: 'Run after adding any new os.environ access, configuration variable, feature flag, or service. Configuration bugs are invisible at unit-test level and only surface in production or on a fresh deployment.',
  phases: [
    { title: 'Hunt', detail: '5 config hunters in parallel — undeclared env vars, prod-unsafe defaults, feature flag mismatch, hardcoded endpoints, missing manifest entry' },
    { title: 'Triage', detail: 'Adversarially verify each critical/high finding — is the env var declared somewhere the hunter didn\'t look?' },
    { title: 'Prioritize', detail: 'Rank by: prod-unsafe default > hardcoded secret/endpoint > undeclared env var > missing manifest' },
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
          severity:      { type: 'string', enum: ['critical', 'high', 'medium', 'low'] },
          category:      { type: 'string' },
          title:         { type: 'string' },
          file:          { type: 'string' },
          var_name:      { type: 'string' },
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

// ── Phase 1: Hunt ─────────────────────────────────────────────────────────────
phase('Hunt')

const HUNTERS = [
  {
    key: 'undeclared_env_var',
    prompt: `Hunt for environment variables accessed in code but not declared in deployment manifests in ${ROOT}.
${LANG_NOTE}

Mechanism (CD4): A developer adds os.environ["NEW_API_KEY"] to application code. The
env var is set in their local .env file. It is not added to the Dockerfile ENV,
Kubernetes ConfigMap/Secret, CI pipeline env, or .env.example. Deployment succeeds;
the application crashes on startup or silently uses None in production.

Steps:
1. Find all env var accesses in non-test application code:
   grep -rn 'os\.environ\["[^"]\+"\]\|os\.environ\.get("[^"]\+")\|process\.env\.[A-Z_]\+\|os\.getenv("[^"]\+")' \
     ${ROOT} --include="*.py" --include="*.ts" --include="*.js" | grep -v test | grep -v "#"
   Note each unique variable name.

2. Find declared env vars in deployment config:
   grep -rn "name:\s\|ENV \|environment:\|secretKeyRef:\|configMapKeyRef:\|key:\s" \
     ${ROOT} --include="*.yaml" --include="*.yml" --include="Dockerfile" --include=".env.example" --include="*.env"
   Also: cat ${ROOT}/.env.example 2>/dev/null
   Note each declared variable.

3. Cross-reference: for each variable accessed in code, is it declared in at least
   one deployment manifest or .env.example? Flag any that are only in code.

4. Also flag: variables with os.environ.get() and no default AND no later null check.
   These will silently fail on first use rather than at startup.

5. Check CI pipeline config:
   find ${ROOT} -name ".github" -o -name ".gitlab-ci.yml" -o -name "Jenkinsfile" -o -name ".circleci" 2>/dev/null | head -5
   For each: is the env var declared in the CI environment section?

category="CD4-undeclared-env-var"`,
  },
  {
    key: 'prod_unsafe_default',
    prompt: `Hunt for configuration defaults that are safe for development but dangerous in production in ${ROOT}.
${LANG_NOTE}

Mechanism (CD5): A config value has a default that is convenient locally but insecure
or incorrect in production. LLMs generate the default as convenient for local use;
the production deployment inherits it silently if the env var is not set.

Dangerous patterns:
- DEBUG = True (default)
- SECRET_KEY = "dev-key" (hardcoded, weak)
- CORS_ALLOW_ALL = True (default)
- SSL_VERIFY = False (default)
- ALLOWED_HOSTS = ["*"] (default)
- RATE_LIMIT = 0 (disabled by default)
- AUTH_REQUIRED = False (default)

Steps:
1. Find configuration defaults:
   grep -rn '\.get("DEBUG",\|\.get("SECRET_KEY",\|\.get("CORS_\|\.get("SSL_VERIFY"\|\.get("ALLOWED_HOSTS"' \
     ${ROOT} --include="*.py"
   grep -rn 'getenv("DEBUG",\|getenv("SECRET_KEY",\|process\.env\.\w\+ \?\?.*true\|process\.env\.\w\+ \|\| "' \
     ${ROOT} --include="*.ts" --include="*.js"

2. For each default: is the default value safe for production?
   - DEBUG: default should be False/false/"false"
   - SECRET_KEY: should have NO default — must be set explicitly
   - CORS: default should be empty list, not wildcard
   - SSL_VERIFY: default should be True
   - ALLOWED_HOSTS: default should be empty list, not "*"

3. Find settings files with insecure values:
   grep -rn "DEBUG\s*=\s*True\|SECRET_KEY\s*=\s*[\"'].\{1,20\}[\"']\|CORS.*=.*\*\|ssl_verify\s*=\s*False" \
     ${ROOT} --include="*.py" --include="*.env*" | grep -v test

4. Find feature flags defaulting to enabled:
   grep -rn "ENABLE_\w\+.*=.*True\|FF_\w\+.*=.*True\|FEATURE_\w\+.*=.*True" ${ROOT} --include="*.py"
   For each: should this feature default to ON in production, or should it default OFF
   and be explicitly enabled?

category="CD5-prod-unsafe-default"`,
  },
  {
    key: 'hardcoded_endpoint',
    prompt: `Hunt for hardcoded service addresses, ports, and credentials in production code paths in ${ROOT}.
${LANG_NOTE}

Mechanism (CD6): A developer hardcodes localhost, 127.0.0.1, or a specific IP/port
in a production code path. Works locally; fails in any multi-container or multi-host
deployment. Or: a hardcoded API key or database password committed to source.

Steps:
1. Find hardcoded localhost/IP in non-test files:
   grep -rn "localhost\|127\.0\.0\.1\|0\.0\.0\.0:\|192\.168\.\|10\.0\.\|172\.\(1[6-9]\|2[0-9]\|3[0-1]\)\." \
     ${ROOT} --include="*.py" --include="*.ts" --include="*.go" --include="*.java" \
     | grep -v "test\|spec\|__pycache__\|\.git\|#"

2. Find hardcoded ports in connection strings:
   grep -rn ":\(5432\|5433\|3306\|6379\|27017\|9200\|8080\|8000\)" ${ROOT} --include="*.py" --include="*.ts"
   For each: is this inside a string literal in a production code path?
   Configuration ports should come from environment variables.

3. Find hardcoded API keys and secrets:
   grep -rn "api_key\s*=\s*[\"'][A-Za-z0-9_-]\{10,\}[\"']\|password\s*=\s*[\"'][^\"']\{4,\}[\"']" \
     ${ROOT} --include="*.py" --include="*.ts" | grep -v "test\|example\|placeholder\|CHANGEME"

4. Find hardcoded connection strings:
   grep -rn "postgresql://\|mysql://\|mongodb://\|redis://\|amqp://" ${ROOT} --include="*.py" --include="*.ts"
   For each: does the URL contain credentials (user:password@)?

5. Find container-name assumptions (docker-compose service names hardcoded):
   grep -rn "redis:6379\|postgres:5432\|rabbitmq:5672\|elasticsearch:9200" ${ROOT}
   Container names work in docker-compose but not in k8s (where service names are different).

category="CD6-hardcoded-endpoint"`,
  },
  {
    key: 'feature_flag_mismatch',
    prompt: `Hunt for feature flags that are mismatched between code and deployment configuration in ${ROOT}.
${LANG_NOTE}

Mechanism: A feature flag is added to code and enabled in one environment but not
declared in all deployment configs. Or a flag is enabled in staging but the deployment
config for production was never updated. The feature silently runs (or doesn't run)
differently across environments.

Steps:
1. Find feature flag declarations in code:
   grep -rn "FEATURE_\|FF_\|ENABLE_\|flag\.\(is_enabled\|check\)\|feature_flag\|LaunchDarkly\|flagsmith\|unleash" \
     ${ROOT} --include="*.py" --include="*.ts" | grep -v test | head -30

2. Find feature flag values in deployment config:
   grep -rn "FEATURE_\|FF_\|ENABLE_" \
     ${ROOT} --include="*.yaml" --include="*.yml" --include="Dockerfile" --include=".env.example"

3. Cross-reference: for each flag in code, is it declared in all relevant deployment
   configs (dev, staging, prod)? Flags missing from prod config silently default
   to whatever the code's default is.

4. Find hardcoded flag values in tests that differ from production config:
   grep -rn "FEATURE_\|FF_\|ENABLE_" ${ROOT}/tests/ ${ROOT}/test_* 2>/dev/null | head -10
   Tests that hardcode flag values test behavior that may not match the deployment.

5. Find flags that are enabled in code (ENABLE_X = True) but disabled in config:
   A flag enabled in code but missing from the manifest means the behavior depends
   on the code default — which may be the wrong value for production.

category="CD-feature-flag-mismatch"`,
  },
  {
    key: 'missing_manifest_entry',
    prompt: `Hunt for new services or components that exist in code but are not represented in deployment manifests in ${ROOT}.
${LANG_NOTE}

Mechanism: A developer adds a new microservice directory or background worker. The service
runs locally (started manually or via docker-compose). It is not added to the k8s
deployment YAML, GitHub Actions workflow, or CI pipeline. The service never deploys;
its functionality is silently absent in production.

Steps:
1. Find service entry points (main files, Dockerfiles, uvicorn/gunicorn starts):
   find ${ROOT} -name "Dockerfile" | head -20
   find ${ROOT} -name "main.py" -o -name "app.py" -o -name "server.py" | head -20
   grep -rn "uvicorn\|gunicorn\|flask run\|node.*index" ${ROOT} --include="*.sh" --include="Makefile"

2. Find deployment config entries:
   find ${ROOT} -name "*.yaml" -o -name "*.yml" | xargs grep -l "kind: Deployment\|kind: Service\|services:" 2>/dev/null | head -10
   For each: what services are declared?

3. Cross-reference: for each Dockerfile or main.py, is there a corresponding k8s
   Deployment/Service, docker-compose service entry, or CI job?

4. Find docker-compose services vs k8s manifests:
   cat ${ROOT}/docker-compose*.yml 2>/dev/null | grep "^\s\+\w\+:" | head -20
   List services in docker-compose. Are all of them also in k8s manifests (if applicable)?

5. Find scheduled jobs/cron tasks in code with no corresponding k8s CronJob:
   grep -rn "schedule\|cron\|celery.*beat\|APScheduler" ${ROOT} --include="*.py"
   For each: is there a corresponding k8s CronJob YAML or systemd timer config?

category="CD-missing-manifest"`,
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
    `Adversarially verify this config-drift finding. Try to REFUTE it.
Default to is_real=false if you cannot confirm by checking the actual manifests/code.

Finding ID: ${f.id} | Category: ${f.category}
Variable/Setting: ${f.var_name || 'unknown'}
File: ${f.file || 'unknown'}
Title: ${f.title}
Detail: ${f.detail}
Evidence: ${f.evidence}

Steps:
1. For undeclared env vars: check .env.example, all YAML files, Dockerfile — is the
   var actually missing, or declared in a file the hunter didn't search?
2. For unsafe defaults: does the deployment set the env var to override the default?
3. For hardcoded endpoints: is this in a test file or a non-production path?
4. For missing manifest: is the service actually deployed via a different mechanism
   (Helm chart, Terraform, external script) not in the checked files?

Return is_real=true only if the config mismatch is confirmed and would affect production.`,
    { label: `triage:${f.id}`, phase: 'Triage', schema: VERDICT_SCHEMA }
  ).then(v => v ? { finding: f, verdict: v } : null)
))).filter(Boolean)

const confirmed = verdicts.filter(v => v.verdict.is_real)
log(`Triage: ${confirmed.length}/${highPriority.length} confirmed config drift issues`)

// ── Phase 3: Prioritize ───────────────────────────────────────────────────────
phase('Prioritize')

const report = await agent(
  `Prioritize these config-drift findings.

Ranking: prod-unsafe default (security risk day 1) > hardcoded secret/endpoint (breaks in multi-host)
       > undeclared env var (startup crash) > feature flag mismatch (wrong behavior) > missing manifest

CONFIRMED HIGH/CRITICAL (${confirmed.length}):
${JSON.stringify(confirmed.map(v => ({
  id: v.finding.id, category: v.finding.category, title: v.finding.title,
  var_name: v.finding.var_name, detail: v.finding.detail,
  fix_recipe: v.finding.fix_recipe, fix_effort: v.finding.fix_effort,
  severity: v.verdict.severity || v.finding.severity,
})), null, 2)}

MEDIUM/LOW (${allFindings.filter(f => f.severity === 'medium' || f.severity === 'low').length}):
${JSON.stringify(allFindings.filter(f => f.severity === 'medium' || f.severity === 'low').map(
  f => ({ id: f.id, category: f.category, title: f.title })
), null, 2)}

Produce: executive_summary, pr_bundle (priority order), pre_deploy_checklist (list of
env vars and config values that MUST be verified before next production deployment).`,
  { label: 'prioritize', phase: 'Prioritize' }
)

if (INV && confirmed.length > 0) {
  await agent(
    `Store confirmed config-drift findings to Loci investigation "${INV}".
For each call mcp__loci__investigation_store(investigation_id="${INV}", finding_type="observed",
confidence="high", tags="config-drift-audit,category:<category>", text="<title>: <detail> | Fix: <fix_recipe>").
Findings: ${JSON.stringify(confirmed.map(v => v.finding), null, 2)}`,
    { label: 'store:loci', phase: 'Prioritize', model: 'haiku' }
  )
}

return { root: ROOT, findings_total: allFindings.length, confirmed_high_critical: confirmed.length, report }
