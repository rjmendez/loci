export const meta = {
  name: 'defcon-fix-prs',
  description: 'Fan out one worktree-isolated fix agent per audit finding-batch: implement + validate + push + open PR + request Copilot review',
  whenToUse: 'After consolidating audit findings into PR-sized batches. Each batch runs in its OWN git worktree of the target repo off origin/main (parallel-safe), implements the scoped fix + a test, validates with eslint + targeted vitest, commits, pushes, opens a PR against main, and requests a Copilot review. args = {repo, batches:[{id,title,branch,scope,fix,test}]}.',
  phases: [{ title: 'FixPR', detail: 'per batch: worktree -> fix -> validate -> push -> PR -> request Copilot' }],
}

const A = typeof args === 'string' ? JSON.parse(args) : (args || {})
const REPO = A.repo
const BATCHES = Array.isArray(A.batches) ? A.batches : []
if (!REPO || !BATCHES.length) { log('need repo + batches[]'); return { error: 'missing args' } }

phase('FixPR')
const results = await parallel(BATCHES.map((b) => () =>
  agent(
    'You are a senior engineer shipping ONE focused fix PR to the repo at ' + REPO + ' (base branch: main). '
    + 'Work ENTIRELY inside a dedicated git worktree so parallel agents do not collide. Do EXACTLY this, in order, '
    + 'and stop + report if any step fails:\n\n'
    + '1. Create the worktree + branch off origin/main (use a unique dir under /tmp):\n'
    + '   WT=/tmp/claude-1000/-home-rjmendez/2c90c7f0-8dc3-4c5b-a56d-b4cb6cdf34ce/scratchpad/wt-' + b.branch.replace(/[^a-zA-Z0-9]/g, '-') + '\n'
    + '   git -C ' + REPO + ' fetch origin main -q; rm -rf "$WT"\n'
    + '   git -C ' + REPO + ' worktree add -b ' + b.branch + ' "$WT" origin/main\n'
    + '   ln -s ' + REPO + '/node_modules "$WT/node_modules"; [ -d ' + REPO + '/app/generated ] && ln -s ' + REPO + '/app/generated "$WT/app/generated"\n'
    + '2. cd "$WT". Implement the fix. SCOPE (only touch these files): ' + b.scope + '\n'
    + '   FIX: ' + b.fix + '\n'
    + '   Read the real current code first; make the MINIMAL correct change. Keep the existing code style.\n'
    + '3. Add or update a focused test proving the fix: ' + (b.test || 'add a unit test under tests/unit/ that fails before and passes after your change.') + '\n'
    + '4. Validate (from "$WT"): `npx eslint <the files you changed>` (must be clean) and '
    + '`npx vitest run --project unit <relevant test file(s)>` (must pass). Do NOT run the full tsc/build (slow; CI covers it). '
    + 'If validation fails, FIX it; if you cannot, report the failure and do NOT push.\n'
    + '5. Commit: `git -C "$WT" add -A && git -C "$WT" commit` with a Conventional-Commits message. '
    + 'End the commit body with these two trailer lines EXACTLY:\n'
    + '   Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\n'
    + '   Claude-Session: https://claude.ai/code/session_01AB14MYuygYSXLfVT1oco45\n'
    + '6. Push: `git -C "$WT" push -u origin ' + b.branch + '`\n'
    + '7. Open the PR: `gh pr create --repo SamG0ld/defcon-defacement --base main --head ' + b.branch + ' --title "..." --body "..."`. '
    + 'The body must: summarize the fix, cite the finding, note the test, and END with exactly these two lines:\n'
    + '   🤖 Generated with [Claude Code](https://claude.com/claude-code)\n'
    + '   https://claude.ai/code/session_01AB14MYuygYSXLfVT1oco45\n'
    + '8. Request a Copilot review (best-effort; capture whether it worked): try '
    + '`gh pr edit <PR#> --repo SamG0ld/defcon-defacement --add-reviewer @copilot` and, if that errors, '
    + '`gh api -X POST repos/SamG0ld/defcon-defacement/pulls/<PR#>/requested_reviewers -f "reviewers[]=copilot-pull-request-reviewer[bot]"`. '
    + 'Record the exact outcome (success or the error text).\n'
    + '9. Clean up: `git -C ' + REPO + ' worktree remove --force "$WT"` (the branch is safely on origin now).\n\n'
    + 'Report honestly. If you skipped or faked any step, say so. Title/body prefix for this batch: [' + b.id + '] ' + b.title,
    { label: 'fix:' + b.id, phase: 'FixPR',
      schema: { type: 'object', required: ['id', 'status'],
        properties: {
          id: { type: 'string' },
          status: { type: 'string', description: 'pushed_pr_opened | validated_not_pushed | failed' },
          branch: { type: 'string' }, pr_number: { type: 'string' }, pr_url: { type: 'string' },
          files_changed: { type: 'array', items: { type: 'string' } },
          validation: { type: 'string', description: 'eslint + vitest results' },
          copilot_review: { type: 'string', description: 'requested-ok | error text | not-attempted' },
          summary: { type: 'string' }, problems: { type: 'string' } } } })
    .then((r) => r ? { ...r, id: r.id || b.id } : { id: b.id, status: 'failed', problems: 'agent returned null' })
    .catch((e) => ({ id: b.id, status: 'failed', problems: String(e) })),
))

const ok = results.filter((r) => r && r.status === 'pushed_pr_opened')
log('opened ' + ok.length + '/' + BATCHES.length + ' PRs')
return { total: BATCHES.length, opened: ok.length, results }
