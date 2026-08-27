export const meta = {
  name: 'defcon-reverify',
  description: 'Re-verify each audit finding against a current-code worktree — pure read-only status check, no writes/PRs',
  whenToUse: 'When an audit finding set may be stale (generated against an out-of-date checkout). Fans out one agent per finding to read the CURRENT code and return FIXED / OPEN / CHANGED / INTENTIONAL with the corrected file:line. args = {tree, findings:[{id,claim,hint,kind}]}.',
  phases: [{ title: 'Reverify', detail: 'one read-only agent per finding against current code' }],
}

const A = typeof args === 'string' ? JSON.parse(args) : (args || {})
const TREE = A.tree
const FINDINGS = Array.isArray(A.findings) ? A.findings : []
if (!TREE || !FINDINGS.length) { log('need tree + findings[]'); return { error: 'missing args' } }

phase('Reverify')
const results = await parallel(FINDINGS.map((f) => () =>
  agent(
    'You are re-verifying ONE prior audit finding against the CURRENT code at ' + TREE + ' (this is the up-to-date '
    + 'origin/main; an earlier audit read a stale checkout, so some findings are already fixed). READ-ONLY: do not edit, '
    + 'commit, or push anything. Just read the real current code.\n\n'
    + 'FINDING [' + f.id + ']: ' + f.claim + '\n'
    + (f.hint ? 'Where to look (verify, do not trust): ' + f.hint + '\n' : '')
    + '\nDetermine the CURRENT status by reading the code:\n'
    + '- OPEN: the defect still exists on origin/main. Give the exact current file:line and the one line of code that proves it.\n'
    + '- FIXED: it has been resolved upstream. Cite the current code that fixes it (file:line).\n'
    + '- CHANGED: the code moved/differs enough that the finding must be re-stated. Give the new statement + location.\n'
    + '- INTENTIONAL: it is a documented, deliberate design choice (cite the doc/comment).\n'
    + 'Be strict and specific. If OPEN, also give a one-line severity (H/M/L) and a one-line fix direction.',
    { label: 'reverify:' + f.id, phase: 'Reverify',
      schema: { type: 'object', required: ['id', 'status'],
        properties: {
          id: { type: 'string' },
          status: { type: 'string', description: 'OPEN | FIXED | CHANGED | INTENTIONAL' },
          location: { type: 'string', description: 'current file:line' },
          evidence: { type: 'string', description: 'the deciding line(s) of current code' },
          severity: { type: 'string' },
          fix_direction: { type: 'string' },
          restated: { type: 'string', description: 'if CHANGED, the corrected finding' } } } })
    .then((r) => r ? { ...r, id: r.id || f.id, claim: f.claim } : { id: f.id, status: 'ERROR', claim: f.claim })
    .catch((e) => ({ id: f.id, status: 'ERROR', evidence: String(e), claim: f.claim })),
))

const by = (s) => results.filter((r) => r && r.status === s).map((r) => r.id)
log('OPEN=' + by('OPEN').length + ' FIXED=' + by('FIXED').length + ' CHANGED=' + by('CHANGED').length + ' INTENTIONAL=' + by('INTENTIONAL').length)
return { open: by('OPEN'), fixed: by('FIXED'), changed: by('CHANGED'), intentional: by('INTENTIONAL'), results }
