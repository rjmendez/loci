export const meta = {
  name: 'loci-fix-sweep',
  description: 'Read open GitHub issues for a repo, triage for fixability, apply patches in worktree-isolated branches, and open PRs. Designed for the well-scoped single-file bug issues produced by deep-think-v4.',
  phases: [
    { title: 'Load',    detail: 'fetch and parse GitHub issues' },
    { title: 'Triage',  detail: 'one classifier per issue — fixable/deferred' },
    { title: 'Fix',     detail: 'worktree-isolated patch + test + commit per issue' },
    { title: 'Summary', detail: 'report PRs opened and deferred issues' },
  ],
}

// ── args normalization ──
const A = (typeof args === 'string')
  ? (() => { try { return JSON.parse(args) || {} } catch (e) { return {} } })()
  : (args || {})

// ── parameters ──
const REPO_PATH    = A.repo         || '/mnt/c/Users/rjmendez-admin/development/loci'
const GITHUB_REPO  = A.github_repo  || 'rjmendez/loci'
const ISSUES       = A.issues       || []         // [] → load from GitHub; or [8, 10, 17]
const MAX_ISSUES   = A.max_issues   || 8          // cap to prevent runaway
const BASE_BRANCH  = A.base_branch  || 'main'
const BRANCH_PFX   = A.branch_prefix || 'fix/'
const SEV_FLOOR    = A.severity_floor || 'medium' // only triage medium+ by default
const TEST_CMD     = A.test_cmd     || 'cd {repo} && python -m pytest mcp/tests/ -x -q --tb=short 2>&1 | tail -20'
const DRY_RUN      = A.dry_run      || false      // if true: plan only, no edits

// ── schemas ──
const ISSUE_SCHEMA = {
  type: 'object', required: ['number', 'title', 'severity', 'file_loc', 'body'],
  properties: {
    number:   { type: 'integer' },
    title:    { type: 'string' },
    severity: { type: 'string', enum: ['critical', 'high', 'medium', 'low', 'unknown'] },
    file_loc: { type: 'string' },  // e.g. "mcp/server.py:1488-1491" or ""
    body:     { type: 'string' },
  },
}
const TRIAGE_SCHEMA = {
  type: 'object', required: ['number', 'fixable', 'reason', 'file', 'line_start', 'line_end', 'fix_description'],
  properties: {
    number:          { type: 'integer' },
    fixable:         { type: 'boolean' },
    reason:          { type: 'string' },
    file:            { type: 'string' },
    line_start:      { type: 'integer' },
    line_end:        { type: 'integer' },
    fix_description: { type: 'string' },
  },
}
const FIX_SCHEMA = {
  type: 'object', required: ['number', 'success', 'branch', 'pr_url', 'error'],
  properties: {
    number:  { type: 'integer' },
    success: { type: 'boolean' },
    branch:  { type: 'string' },
    pr_url:  { type: 'string' },
    error:   { type: 'string' },
  },
}

const SEV_RANK = { critical: 4, high: 3, medium: 2, low: 1, unknown: 0 }
const floorRank = SEV_RANK[SEV_FLOOR] || 2

// ── Load ──
phase('Load')

const issuesFetcher = await agent(
  ISSUES.length > 0
    ? `Fetch specific GitHub issues from ${GITHUB_REPO}: ${ISSUES.join(', ')}.\n\nFor EACH issue number, run:\n  gh issue view <number> --repo ${GITHUB_REPO} --json number,title,labels,body\n\nParse each issue's body to extract:\n- severity: from label "severity:high" → "high" etc., or parse "[high]" / "[medium]" from title\n- file_loc: regex match for a path like "mcp/server.py:1488-1491" or "scripts/hooks/pre_tool_grounding.py:264-278" from the body\n\nReturn an array of parsed issues.`
    : `List open GitHub issues from ${GITHUB_REPO} with labels severity:critical, severity:high, or severity:medium.\n\nRun:\n  gh issue list --repo ${GITHUB_REPO} --label "severity:high" --state open --limit 20 --json number,title,labels\n  gh issue list --repo ${GITHUB_REPO} --label "severity:medium" --state open --limit 20 --json number,title,labels\n  gh issue list --repo ${GITHUB_REPO} --label "severity:critical" --state open --limit 20 --json number,title,labels\n\nFor each issue (up to ${MAX_ISSUES} total, prioritize critical > high > medium), run:\n  gh issue view <number> --repo ${GITHUB_REPO} --json number,title,labels,body\n\nParse each body for file_loc (e.g. "mcp/server.py:1488-1491"). Return array of parsed issues.`,
  {
    label: 'load:issues',
    phase: 'Load',
    model: 'haiku',
    schema: { type: 'object', required: ['issues'], properties: { issues: { type: 'array', items: ISSUE_SCHEMA } } },
  }
)

const loadedIssues = (issuesFetcher?.issues || [])
  .filter(iss => (SEV_RANK[iss.severity] || 0) >= floorRank)
  .slice(0, MAX_ISSUES)

log(`Load: ${loadedIssues.length} issues above floor=${SEV_FLOOR}: ${loadedIssues.map(i => `#${i.number}`).join(', ')}`)

if (!loadedIssues.length) {
  log('No issues to fix. Exiting.')
  return { opened: [], deferred: [], total: 0 }
}

// ── Triage ──
// Assess each issue: does it have a specific file:line? Is it a single-file change?
// Skip issues that need new files, schema migrations, or are too broad.
phase('Triage')

const triageResults = (await parallel(loadedIssues.map(iss => () =>
  agent(
    `Triage issue #${iss.number}: "${iss.title}"\n\nBody:\n${iss.body}\n\nYour task: assess whether this is fixable by a single automated patch.\n\nFIXABLE criteria:\n- Cites a specific file and line range\n- Change is ≤ ~50 lines in one file\n- No new file creation required\n- No database migration or schema change\n- No dependency bump\n\nDEFERRED if: too broad, needs architecture discussion, needs multiple files with complex interactions, unclear spec.\n\nParse from the body:\n- file: the file path (e.g. "mcp/server.py")\n- line_start/line_end: the line range from the location field\n- fix_description: one sentence describing exactly what to change\n\nReturn fixable=true/false, reason, and parsed file/lines/fix_description.`,
    { label: `triage:#${iss.number}`, phase: 'Triage', model: 'haiku', schema: TRIAGE_SCHEMA }
  )
))).filter(Boolean)

// ── Dedup check: skip issues that already have an open PR from someone else ──
const dedupResults = (await parallel(triageResults.filter(t => t.fixable).map(t => () =>
  agent(
    `Check if issue #${t.number} in ${GITHUB_REPO} already has an open PR from another contributor.\n\nRun: gh pr list --repo ${GITHUB_REPO} --search "#${t.number}" --state open --json number,title,author --limit 5\n\nAlso try: gh pr list --repo ${GITHUB_REPO} --state open --json number,title,body --limit 30 | grep -i "issue.${t.number}\\|closes.${t.number}\\|fixes.${t.number}"\n\nIf any open PR already references issue #${t.number} and was NOT authored by a bot or CI automation, return has_pr=true and pr_info="PR #<N> by <author>".\nOtherwise return has_pr=false, pr_info="".`,
    {
      label: `dedup:#${t.number}`, phase: 'Triage', model: 'haiku',
      schema: { type: 'object', required: ['number','has_pr','pr_info'], properties: {
        number: { type: 'integer' }, has_pr: { type: 'boolean' }, pr_info: { type: 'string' } } },
    }
  ).then(r => ({ ...t, has_pr: r?.has_pr || false, pr_info: r?.pr_info || '' }))
))).filter(Boolean)

const fixable = dedupResults.filter(t => !t.has_pr)
const existingPR = dedupResults.filter(t => t.has_pr)
const deferred = [...triageResults.filter(t => !t.fixable), ...existingPR.map(t => ({ ...t, reason: `existing PR: ${t.pr_info}` }))]

log(`Triage: ${fixable.length} fixable, ${existingPR.length} skipped (existing PR), ${deferred.length} deferred total`)
if (existingPR.length) log(`Skipped (existing PRs): ${existingPR.map(t => `#${t.number} → ${t.pr_info}`).join('; ')}`)
if (deferred.length) log(`Deferred: ${deferred.filter(d => !d.has_pr).map(d => `#${d.number} (${d.reason.slice(0,40)})`).join('; ')}`)

if (!fixable.length) {
  log('No fixable issues. Exiting.')
  return { opened: [], deferred: deferred.map(d => ({ number: d.number, reason: d.reason })), total: 0 }
}

if (DRY_RUN) {
  log('DRY RUN — stopping after triage. Set dry_run=false to apply fixes.')
  return {
    dry_run: true,
    fixable: fixable.map(f => ({ number: f.number, file: f.file, fix: f.fix_description })),
    deferred: deferred.map(d => ({ number: d.number, reason: d.reason })),
  }
}

// ── Fix ──
// Sequential: one agent per issue. Each creates its own branch from BASE_BRANCH,
// edits the file, runs tests, commits, pushes, and opens a PR.
// Sequential (not pipeline) avoids git-index races when multiple agents share one repo.
phase('Fix')

const issueMap = Object.fromEntries(loadedIssues.map(i => [i.number, i]))

const fixResults = []
for (const triage of fixable) {
  const iss = issueMap[triage.number] || {}
  const branch = `${BRANCH_PFX}issue-${triage.number}`
  const testCmd = TEST_CMD.replace('{repo}', REPO_PATH)
  const filePath = `${REPO_PATH}/${triage.file}`
  const readOffset = Math.max(0, triage.line_start - 15)
  const readLimit  = (triage.line_end - triage.line_start) + 50

  log(`Fix: starting #${triage.number} ${triage.file}:${triage.line_start}-${triage.line_end}`)
  const result = await agent(
    `Fix GitHub issue #${triage.number} in the loci repo at ${REPO_PATH}.\n\nIssue: "${iss.title}"\nSeverity: ${iss.severity}\nFile: ${filePath}\nTarget lines: ${triage.line_start}–${triage.line_end}\n\nFix description:\n${triage.fix_description}\n\nFull issue body:\n${iss.body}\n\nSteps — follow in order, stop and return success=false on any error:\n\n1. Read ${filePath} with offset=${readOffset} limit=${readLimit} to see the exact code\n2. Apply the minimal fix using Edit. Stay within the scope of the fix description.\n3. Run Bash: ${testCmd}\n   - If tests FAIL: restore the original (Edit it back), return success=false, error="tests failed: <last 10 lines of output>"\n   - If no tests exist for this file, note that and proceed\n4. Bash: git -C ${REPO_PATH} checkout -b ${branch} ${BASE_BRANCH} 2>&1\n   (If branch already exists, use git checkout ${branch} instead)\n5. Bash: git -C ${REPO_PATH} add ${filePath}\n6. Bash: git -C ${REPO_PATH} commit -m "fix: ${iss.title.replace(/"/g, '\\"').replace(/\[/g, '').replace(/\]/g, '')} (#${triage.number})"\n7. Bash: git -C ${REPO_PATH} push origin ${branch} 2>&1\n8. Bash: gh pr create --repo ${GITHUB_REPO} --base ${BASE_BRANCH} --head ${branch} --title "fix(#${triage.number}): ${iss.title.replace(/"/g, '\\"').replace(/\[/g, '').replace(/\]/g, '').slice(0, 60)}" --body "Closes #${triage.number}\\n\\n## Change\\n${triage.fix_description.replace(/"/g, '\\"')}\\n\\n*Generated by loci-fix-sweep*" 2>&1\n\nReturn success=true, branch="${branch}", pr_url=<the URL from gh pr create output>, error=""\nIf any step fails, return success=false, pr_url="", error=<description of what failed and why>.`,
    {
      label: `fix:#${triage.number}`,
      phase: 'Fix',
      model: 'sonnet',
      schema: FIX_SCHEMA,
    }
  )
  fixResults.push(result || { number: triage.number, success: false, branch, pr_url: '', error: 'agent returned null' })
  log(`Fix: #${triage.number} → ${result?.success ? `PR ${result.pr_url}` : `FAILED: ${result?.error?.slice(0,60)}`}`)
}

const opened  = fixResults.filter(r => r.success)
const failed  = fixResults.filter(r => !r.success)

log(`Fix: ${opened.length} PRs opened, ${failed.length} failed`)

// ── Summary ──
phase('Summary')
await agent(
  `Print a summary of the loci-fix-sweep run for GitHub repo ${GITHUB_REPO}.\n\nOpened PRs (${opened.length}):\n${JSON.stringify(opened, null, 2)}\n\nFailed (${failed.length}):\n${JSON.stringify(failed, null, 2)}\n\nDeferred (${deferred.length}):\n${JSON.stringify(deferred, null, 2)}\n\nReturn one paragraph summary and suggest next steps for any failed or deferred items.`,
  { label: 'summary', phase: 'Summary', model: 'haiku' }
)

return {
  opened:   opened.map(r => ({ number: r.number, pr_url: r.pr_url, branch: r.branch })),
  failed:   failed.map(r => ({ number: r.number, error: r.error })),
  deferred: deferred.map(d => ({ number: d.number, reason: d.reason })),
  total:    opened.length,
}
