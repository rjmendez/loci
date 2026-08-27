export const meta = {
  name: 'loci-address-copilot',
  description: 'Address Copilot review comments on existing loci PR branches: fix each comment, re-gate (pytest+ruff), push, re-request Copilot. One agent per PR.',
  whenToUse: 'After Copilot reviews a batch of loci PRs. Each agent checks out the PR branch, addresses its listed comments minimally, re-runs the full mcp suite + ruff, commits (trailers on), pushes to update the PR, and re-requests a Copilot review. args = {repo, prs:[{number,branch,comments}]}.',
  phases: [{ title: 'Address', detail: 'per PR: fix Copilot comments -> gate -> push -> re-request Copilot' }],
}

const A = typeof args === 'string' ? JSON.parse(args) : (args || {})
const REPO = A.repo
const PRS = Array.isArray(A.prs) ? A.prs : []
if (!REPO || !PRS.length) { log('need repo + prs[]'); return { error: 'missing args' } }

const TRAILERS = 'Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01AB14MYuygYSXLfVT1oco45'

const results = await parallel(PRS.map((p) => () =>
  agent(
    'Address the Copilot review comments on PR #' + p.number + ' (branch ' + p.branch + ') of the loci repo at ' + REPO + '. '
    + 'Backward-compatible, minimal, fail-open — do not over-refactor. Steps:\n\n'
    + '1. Check out the existing branch in a worktree + share the venv:\n'
    + '   WT=/tmp/claude-1000/-home-rjmendez/2c90c7f0-8dc3-4c5b-a56d-b4cb6cdf34ce/scratchpad/cop-' + p.branch.replace(/[^a-zA-Z0-9]/g, '-') + '\n'
    + '   git -C ' + REPO + ' fetch origin ' + p.branch + ' -q; rm -rf "$WT"\n'
    + '   git -C ' + REPO + ' worktree add "$WT" ' + p.branch + '\n'
    + '   ln -s ' + REPO + '/mcp/.venv "$WT/mcp/.venv"\n'
    + '2. Read the current code, then address EACH comment below. Fix the substance, not just the symptom. For SECURITY '
    + 'comments (path traversal / arbitrary file read): restrict any file path derived from user-controlled text to paths '
    + 'UNDER the repo root, reject absolute paths and ".." traversal, and add a sane size cap; fail-open (skip a rejected ref, '
    + 'never raise). COMMENTS:\n' + p.comments + '\n'
    + '3. Validate from "$WT/mcp": `. .venv/bin/activate; python -m pytest -q` (FULL suite green) + '
    + '`ruff check <changed files> --select E9,F401,F811,F821,F841 --ignore E402`. Add/adjust tests where a comment asks for '
    + 'coverage. Fix any failure before committing.\n'
    + '4. Commit the intended files (explicit `git add`). Conventional-Commits message referencing the Copilot feedback; '
    + 'END the body with EXACTLY these two lines:\n' + TRAILERS + '\n'
    + '5. Push to update the PR: `git -C "$WT" push origin ' + p.branch + '`\n'
    + '6. Re-request Copilot: `gh api --method POST repos/rjmendez/loci/pulls/' + p.number + '/requested_reviewers -f "reviewers[]=copilot-pull-request-reviewer[bot]"` (capture outcome).\n'
    + '7. Clean up: `git -C ' + REPO + ' worktree remove --force "$WT"`.\n\n'
    + 'Report which comments you addressed and how, honestly (flag any you deliberately did NOT change + why).',
    { label: 'address:' + p.number, phase: 'Address',
      schema: { type: 'object', required: ['number', 'status'],
        properties: {
          number: { type: 'string' }, status: { type: 'string', description: 'pushed | failed' },
          addressed: { type: 'array', items: { type: 'string' } },
          not_changed: { type: 'string' }, suite: { type: 'string' },
          copilot_rerequested: { type: 'string' }, problems: { type: 'string' } } } })
    .then((r) => r ? { ...r, number: r.number || String(p.number) } : { number: String(p.number), status: 'failed', problems: 'null' })
    .catch((e) => ({ number: String(p.number), status: 'failed', problems: String(e) })),
))

const pushed = results.filter((r) => r && r.status === 'pushed')
log('addressed + re-requested Copilot on ' + pushed.length + '/' + PRS.length + ' PRs')
return { total: PRS.length, pushed: pushed.length, results }
