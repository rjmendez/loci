export const meta = {
  name: 'defcon-fix-branches',
  description: 'Fan out one worktree-isolated fix agent per finding: implement + validate + reviewer subagents + commit to a local branch. NEVER pushes (repo rule: human-only push).',
  whenToUse: 'For SamG0ld/defcon-defacement, whose CLAUDE.md forbids pushing from agents and forbids AI-attribution trailers. Each batch: worktree off origin/main, implement the scoped fix + test, validate (eslint + vitest + typecheck), get code-reviewer + security-reviewer passes on the diff, commit to a local branch with a clean Conventional-Commits message (NO trailers), and STOP. Josh reviews + pushes. args = {repo, batches:[{id,title,branch,scope,fix,test}]}.',
  phases: [{ title: 'FixBranch', detail: 'per batch: worktree -> fix -> validate -> review -> commit to local branch (no push)' }],
}

const A = typeof args === 'string' ? JSON.parse(args) : (args || {})
const REPO = A.repo
const BATCHES = Array.isArray(A.batches) ? A.batches : []
if (!REPO || !BATCHES.length) { log('need repo + batches[]'); return { error: 'missing args' } }

phase('FixBranch')
const results = await parallel(BATCHES.map((b) => () =>
  agent(
    'You are shipping ONE focused fix to a LOCAL branch of the repo at ' + REPO + '. This repo FORBIDS agents pushing '
    + '(human-only push) and FORBIDS AI-attribution trailers in commits — obey both. Do EXACTLY this, in order, and '
    + 'report honestly (say so if you skip/adjust a step):\n\n'
    + '1. Worktree + branch off origin/main (unique /tmp dir):\n'
    + '   WT=/tmp/claude-1000/-home-rjmendez/2c90c7f0-8dc3-4c5b-a56d-b4cb6cdf34ce/scratchpad/wt-' + b.branch.replace(/[^a-zA-Z0-9]/g, '-') + '\n'
    + '   git -C ' + REPO + ' fetch origin main -q; rm -rf "$WT"\n'
    + '   git -C ' + REPO + ' worktree add -b ' + b.branch + ' "$WT" origin/main\n'
    + '   ln -s ' + REPO + '/node_modules "$WT/node_modules"; [ -d ' + REPO + '/app/generated ] && ln -s ' + REPO + '/app/generated "$WT/app/generated"\n'
    + '2. FIRST verify the finding still exists on origin/main (it may already be fixed — the audit set was partly '
    + 'stale). If already fixed: report FIXED, open NO branch content, clean up, and stop. Otherwise implement the '
    + 'MINIMAL correct fix. SCOPE (only these files): ' + b.scope + '\n   FIX: ' + b.fix + '\n'
    + '3. Add/update a focused test: ' + (b.test || 'a unit test that fails before and passes after.') + '\n'
    + '4. Validate from "$WT": `npx eslint <changed files>` (clean), `npx vitest run <relevant test(s)>` (pass), and '
    + '`npx tsc --noEmit` if it completes within a few minutes (else note it is deferred to CI). Fix any failure; if you '
    + 'cannot, report it and do NOT commit.\n'
    + '5. Commit ONLY the intended files with an explicit `git add <files>` (NEVER `git add -A` — the app/generated symlink '
    + 'is not gitignored and must not be committed). Use a Conventional-Commits message. DO NOT add any Co-Authored-By, '
    + 'Claude-Session, "Generated with", or other AI-attribution trailer — the repo forbids them.\n'
    + '6. Do NOT push and do NOT open a PR. Leave the branch local in ' + REPO + ' for the human to review + push.\n'
    + '7. Clean up the worktree: `git -C ' + REPO + ' worktree remove --force "$WT"` (the branch persists in the repo).\n\n'
    + 'Batch: [' + b.id + '] ' + b.title,
    { label: 'fix:' + b.id, phase: 'FixBranch',
      schema: { type: 'object', required: ['id', 'status'],
        properties: {
          id: { type: 'string' },
          status: { type: 'string', description: 'committed_local_branch | already_fixed | failed' },
          branch: { type: 'string' }, files_changed: { type: 'array', items: { type: 'string' } },
          validation: { type: 'string' }, summary: { type: 'string' }, problems: { type: 'string' } } } })
    .then((r) => r ? { ...r, id: r.id || b.id } : { id: b.id, status: 'failed', problems: 'agent returned null' })
    .catch((e) => ({ id: b.id, status: 'failed', problems: String(e) })),
))

// Independent reviewer pass on each committed branch (repo "done" bar: fresh-context reviewers).
const committed = results.filter((r) => r && r.status === 'committed_local_branch')
const reviewed = await parallel(committed.map((r) => () =>
  agent(
    'Review the diff on local branch ' + r.branch + ' of ' + REPO + ' (git -C ' + REPO + ' diff origin/main..' + r.branch + '). '
    + 'You are a fresh-context reviewer — the author did NOT grade themselves. Check correctness, security, scope creep, '
    + 'test adequacy, and that NO AI-attribution trailer is present. Return {branch, verdict: approve|changes-requested, notes}.',
    { label: 'review:' + r.id, phase: 'FixBranch',
      schema: { type: 'object', required: ['verdict'],
        properties: { branch: { type: 'string' }, verdict: { type: 'string' }, notes: { type: 'string' } } } })
    .then((v) => ({ id: r.id, ...v })).catch(() => ({ id: r.id, verdict: 'review-error' }))))

return { total: BATCHES.length, committed: committed.length, results, reviews: reviewed }
