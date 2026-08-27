export const meta = {
  name: 'loci-improve',
  description: 'Fan out one worktree-isolated agent per Loci self-improvement: design-minimal implement + tests + pytest/ruff gate + fresh reviewer, commit, push, open PR (loci conventions: trailers ON).',
  whenToUse: 'Implement Loci reliability fixes + features discovered while dogfooding. Each batch: worktree off loci main, implement the MINIMAL backward-compatible change + tests, run the mcp pytest suite + ruff, commit WITH the standard loci trailers, push, open a PR, and get a fresh-context reviewer. args = {repo, batches:[{id,title,branch,scope,design,test}]}.',
  phases: [{ title: 'Improve', detail: 'per batch: worktree -> implement -> pytest/ruff -> commit -> push -> PR -> review' }],
}

const A = typeof args === 'string' ? JSON.parse(args) : (args || {})
const REPO = A.repo
const BATCHES = Array.isArray(A.batches) ? A.batches : []
if (!REPO || !BATCHES.length) { log('need repo + batches[]'); return { error: 'missing args' } }

const TRAILERS = 'Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01AB14MYuygYSXLfVT1oco45'
const PRFOOTER = '🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nhttps://claude.ai/code/session_01AB14MYuygYSXLfVT1oco45'

const results = await parallel(BATCHES.map((b) => () =>
  agent(
    'You are implementing ONE Loci self-improvement in the repo at ' + REPO + ' (a Python MCP server; base branch main). '
    + 'Work entirely in a dedicated git worktree. Be MINIMAL and BACKWARD-COMPATIBLE (this is a live-depended-on server): '
    + 'additive changes, fail-open, defaults that preserve current behavior. Steps:\n\n'
    + '1. Worktree + branch off origin/main; share the venv:\n'
    + '   WT=/tmp/claude-1000/-home-rjmendez/2c90c7f0-8dc3-4c5b-a56d-b4cb6cdf34ce/scratchpad/loci-' + b.branch.replace(/[^a-zA-Z0-9]/g, '-') + '\n'
    + '   git -C ' + REPO + ' fetch origin main -q; rm -rf "$WT"\n'
    + '   git -C ' + REPO + ' worktree add -b ' + b.branch + ' "$WT" origin/main\n'
    + '   ln -s ' + REPO + '/mcp/.venv "$WT/mcp/.venv"\n'
    + '2. Read the real current code FIRST. SCOPE (touch only what is needed here): ' + b.scope + '\n'
    + '   DESIGN / what to build: ' + b.design + '\n'
    + '   Keep it small and correct; match surrounding style; do NOT refactor unrelated code.\n'
    + '3. Add focused tests: ' + b.test + '\n'
    + '4. Validate from "$WT/mcp": `. .venv/bin/activate; python -m pytest -q` (the FULL mcp suite must stay green — this '
    + 'is a live server, no regressions) and `ruff check <changed files> --select E9,F401,F811,F821,F841 --ignore E402`. '
    + 'Fix any failure; if you cannot, report it and do NOT push.\n'
    + '5. Commit the intended files (explicit `git add`). Conventional-Commits message; END the body with EXACTLY these two trailer lines:\n'
    + TRAILERS + '\n'
    + '6. Push: `git -C "$WT" push -u origin ' + b.branch + '`\n'
    + '7. Open the PR: `gh pr create --repo rjmendez/loci --base main --head ' + b.branch + ' --title "..." --body "..."`. '
    + 'End the PR body with EXACTLY:\n' + PRFOOTER + '\n'
    + '8. Request a Copilot review (capture the exact outcome): try '
    + '`gh api --method POST repos/rjmendez/loci/pulls/<PR#>/requested_reviewers -f "reviewers[]=copilot-pull-request-reviewer[bot]"` '
    + '(this is the call that works; `gh pr edit --add-reviewer @copilot` errors on Projects-classic). Record whether the '
    + 'response shows Copilot in requested_reviewers.\n'
    + '9. Clean up: `git -C ' + REPO + ' worktree remove --force "$WT"`.\n\n'
    + 'Report honestly (disclose any skip/deviation). Batch: [' + b.id + '] ' + b.title,
    { label: 'impl:' + b.id, phase: 'Improve',
      schema: { type: 'object', required: ['id', 'status'],
        properties: {
          id: { type: 'string' }, status: { type: 'string', description: 'pushed_pr_opened | validated_not_pushed | failed' },
          branch: { type: 'string' }, pr_number: { type: 'string' }, pr_url: { type: 'string' },
          files_changed: { type: 'array', items: { type: 'string' } },
          copilot_review: { type: 'string', description: 'requested-ok | error text | not-attempted' },
          suite: { type: 'string', description: 'pytest result (N passed) + ruff' },
          summary: { type: 'string' }, problems: { type: 'string' } } } })
    .then((r) => r ? { ...r, id: r.id || b.id } : { id: b.id, status: 'failed', problems: 'null' })
    .catch((e) => ({ id: b.id, status: 'failed', problems: String(e) })),
))

const opened = results.filter((r) => r && r.status === 'pushed_pr_opened')
const reviewed = await parallel(opened.map((r) => () =>
  agent(
    'Fresh-context review of PR ' + (r.pr_url || r.branch) + ' in ' + REPO + ' (git -C ' + REPO + ' diff origin/main..' + r.branch + '). '
    + 'This is a change to a live-depended-on MCP server. Check: correctness, backward-compatibility (defaults preserve current '
    + 'behavior?), fail-open, test adequacy, no unrelated refactor, and that the full mcp pytest suite is green. '
    + 'Return {branch, verdict: approve|changes-requested, notes}.',
    { label: 'review:' + r.id, phase: 'Improve',
      schema: { type: 'object', required: ['verdict'], properties: { branch: { type: 'string' }, verdict: { type: 'string' }, notes: { type: 'string' } } } })
    .then((v) => ({ id: r.id, ...v })).catch(() => ({ id: r.id, verdict: 'review-error' }))))

return { total: BATCHES.length, opened: opened.length, results, reviews: reviewed }
