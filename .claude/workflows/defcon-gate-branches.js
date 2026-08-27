export const meta = {
  name: 'defcon-gate-branches',
  description: 'Bring existing local fix branches to the repo done-bar: run typecheck (or Python gate) + a fresh-context reviewer per branch. Read-only, no push.',
  whenToUse: 'For local fix branches that only got a partial gate. Per branch: worktree at the branch, run tsc --noEmit + eslint (TS) or ruff + pytest (Python), then a fresh-context reviewer on the diff vs origin/main. args = {repo, branches:[{id,branch,lang}]}.',
  phases: [{ title: 'Gate', detail: 'per branch: typecheck/lint + fresh reviewer' }],
}

const A = typeof args === 'string' ? JSON.parse(args) : (args || {})
const REPO = A.repo
const BRANCHES = Array.isArray(A.branches) ? A.branches : []
if (!REPO || !BRANCHES.length) { log('need repo + branches[]'); return { error: 'missing args' } }

phase('Gate')
const results = await parallel(BRANCHES.map((b) => () =>
  agent(
    'Bring the EXISTING local branch ' + b.branch + ' of ' + REPO + ' to the repo done-bar. READ-ONLY: do not edit, '
    + 'commit, or push. This branch is language: ' + (b.lang || 'typescript') + '.\n\n'
    + '1. Check it out in a worktree:\n'
    + '   WT=/tmp/claude-1000/-home-rjmendez/2c90c7f0-8dc3-4c5b-a56d-b4cb6cdf34ce/scratchpad/gate-' + b.id + '\n'
    + '   rm -rf "$WT"; git -C ' + REPO + ' worktree add "$WT" ' + b.branch + '\n'
    + '   ln -s ' + REPO + '/node_modules "$WT/node_modules"; [ -d ' + REPO + '/app/generated ] && ln -s ' + REPO + '/app/generated "$WT/app/generated"\n'
    + '2. From "$WT", run the gate:\n'
    + (b.lang === 'python'
        ? '   - `ruff check <changed .py files>` (or `python -m py_compile`), and `python -m pytest <the branch\'s test file>`. '
          + 'tsc/eslint do NOT apply to Python.\n'
        : '   - `npx tsc --noEmit` (WHOLE-PROJECT typecheck — this is the bar the earlier pass skipped; give it a few minutes), '
          + '`npx eslint <changed files>`, and `npx vitest run <the branch\'s test file(s)>`.\n')
    + '   Report each result exactly (pass/fail + any errors). Do NOT try to fix failures — just report them.\n'
    + '3. Then REVIEW the diff as a fresh-context reviewer (you did not write it): '
    + '`git -C ' + REPO + ' diff origin/main..' + b.branch + '`. Check correctness, security, scope creep, test adequacy, '
    + 'and that NO AI-attribution trailer is present in the commit.\n'
    + '4. Clean up: `git -C ' + REPO + ' worktree remove --force "$WT"`.\n\n'
    + 'Return the gate results + your review verdict.',
    { label: 'gate:' + b.id, phase: 'Gate',
      schema: { type: 'object', required: ['id', 'typecheck', 'verdict'],
        properties: {
          id: { type: 'string' }, branch: { type: 'string' },
          typecheck: { type: 'string', description: 'tsc/ruff+pytest result: pass | fail: <detail> | n/a' },
          lint_test: { type: 'string', description: 'eslint + vitest (or py) result' },
          verdict: { type: 'string', description: 'approve | changes-requested' },
          notes: { type: 'string' } } } })
    .then((r) => r ? { ...r, id: r.id || b.id, branch: r.branch || b.branch } : { id: b.id, branch: b.branch, verdict: 'error' })
    .catch((e) => ({ id: b.id, branch: b.branch, verdict: 'error', notes: String(e) })),
))

const approved = results.filter((r) => r && r.verdict === 'approve')
log('gate: ' + approved.length + '/' + BRANCHES.length + ' approve')
return { results }
