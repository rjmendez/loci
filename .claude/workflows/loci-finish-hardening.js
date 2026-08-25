export const meta = {
  name: 'loci-finish-hardening',
  description: 'Close out the three open Loci defects: grounding-gate calibration, audit receipts, guard-log wiring',
  whenToUse: 'After #218/#219 land. Each target is a defect already measured — agents design and implement, they do not re-derive.',
  phases: [
    { title: 'Investigate', detail: 'One agent per defect: verify the measurement against live code, then design the fix' },
    { title: 'Implement',   detail: 'Isolated worktree per defect: code + tests that fail without the fix' },
    { title: 'Refute',      detail: 'Adversarial pass — try to break each implementation before it is proposed' },
  ],
}

const GROUND = (typeof args === 'object' && args && args.ground) || ''
const REPO = (typeof args === 'object' && args && args.repo) || '.'

// House rules that cost the project real time when they were missed. Every agent
// gets these; they are not optional style notes.
const RULES = `
DISCIPLINE — these are the specific mistakes this codebase has already paid for:

1. MEASURE, do not assert. Lead with the number and how you got it. If you infer a
   mechanism, say it is inference and say what would falsify it. Consistency is not
   confirmation.
2. RUN the code. py_compile and ruff pass code that raises NameError at runtime.
   A recent fix called log() in a file that has no log() — inside an except block,
   so it would have crashed a hook on every session end.
3. A test must FAIL without the fix. Reintroduce the bug, watch it fail, restore.
   A test that passes both ways proves nothing. One recent test failed on the old
   code for the wrong reason (AttributeError on a new constant, not the defect).
4. Fix the CLASS at the shared site, not the one call site that showed the symptom.
5. Zero output is not automatically a defect. Check whether the producer ever RAN
   before calling it broken — an issue was filed on that error and had to be retracted.
6. Comment economy: rationale goes in the commit message, not in a wall of source
   comments. Explain WHY where it is not obvious; never narrate WHAT the code does.
`

const DESIGN_SCHEMA = {
  type: 'object',
  properties: {
    defect: { type: 'string' },
    measurement_confirmed: { type: 'boolean' },
    measurement_notes: { type: 'string' },
    root_cause: { type: 'string' },
    design: { type: 'string' },
    files_to_change: { type: 'array', items: { type: 'string' } },
    test_plan: { type: 'string' },
    risks: { type: 'string' },
    recommend_implement: { type: 'boolean' },
    why_not: { type: 'string' },
  },
  required: ['defect', 'measurement_confirmed', 'root_cause', 'design',
             'files_to_change', 'test_plan', 'recommend_implement'],
}

const IMPL_SCHEMA = {
  type: 'object',
  properties: {
    defect: { type: 'string' },
    branch: { type: 'string' },
    diff: { type: 'string' },
    tests_added: { type: 'array', items: { type: 'string' } },
    sabotage_result: { type: 'string' },
    suite_result: { type: 'string' },
    complete: { type: 'boolean' },
    blocked_reason: { type: 'string' },
  },
  required: ['defect', 'diff', 'tests_added', 'sabotage_result', 'complete'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    defect: { type: 'string' },
    refuted: { type: 'boolean' },
    strongest_objection: { type: 'string' },
    evidence: { type: 'string' },
    safe_to_propose: { type: 'boolean' },
  },
  required: ['defect', 'refuted', 'strongest_objection', 'safe_to_propose'],
}

const DEFECTS = [
  {
    id: 'grounding-gate-183',
    title: 'semantic support threshold admits false support',
    brief: `mcp/server.py: _QDRANT_SUPPORT_MIN_SCORE = 0.55 gates the SEMANTIC lane of
investigation_pre_answer_check. Measured previously on one investigation: negative
scores reached 0.5838 while positives started at 0.5816 — the distributions OVERLAP,
so no constant separates them and 0.55 admits false support.

The LEXICAL half of this issue is already fixed (stopwords added to tokenize(),
false support 68.8% -> 1.2%, true positives unchanged at 96.7%). Do not redo it.

Issue #183's own suggestion: treat the semantic lane as a CANDIDATE GENERATOR
requiring corroboration, rather than as an independent adjudicator. The issue also
notes the calibration set does not exist in this repo — so a design that needs one
is blocked, and you should say so rather than inventing labels.

First: reproduce the overlap on the live corpus (~/.hermes/memory-sessions, 141
investigations). If you cannot reproduce it, say so — that changes the fix.`,
  },
  {
    id: 'audit-receipts-193',
    title: 'nothing writes audit receipts',
    brief: `audit_log (mcp/server.py) is an @mcp.tool() documented as "a post-call hook
after any MCP tool invocation". Nothing hooks it. Measured: 1 of 141 investigations
has an audit.jsonl, 3 records total, newest 2026-06-20.

Consequence already fixed at the CONSUMER: run_provenance flags every observed
finding lacking a receipt, so with no receipts it flags 1,682 of them by
construction. #213 added _audit_lane_status so a report says "no receipts exist"
instead of "1,682 unsupported". Verdicts were deliberately NOT changed —
test_checks.py::test_observed_with_empty_audit_flagged pins that contract.

What remains is the producer. Constraints that make this non-trivial and that you
must address explicitly:
  - audit_log writes the FULL untruncated tool output. Hooking every tool call
    would write enormous volume. Say what you would record instead.
  - A PostToolUse hook is the obvious mechanism; check whether this host's hook
    config (~/.claude/settings.json) already has one and what it does.
  - Receipts only matter for findings whose provenance is checkable. Consider
    whether the right trigger is every tool call, or only investigation_store.
If you conclude this should NOT be built, say so with reasons — that is a valid
outcome and better than a hook nobody wants.`,
  },
  {
    id: 'guard-log-216',
    title: 'four consumers, zero producers',
    brief: `guard_bash_failures.log / guard_bash_successes.log / guard_tool_reflections.log
in STATE_DIR (~/.claude/hook-state) have FOUR production consumers:
  scripts/score_trace_collector.py   - SCoRe fine-tuning dataset
  scripts/skill_annotation_updater.py
  mlops/memory/live_evo.py           - confidence penalties from guard failures
  mlops/loop.py                      - passes hook_state_dir through
and ZERO producers. A repo-wide grep for a write to guard_*.log returns only two
TEST files, which create the fixture themselves.

Measured: ~/.claude/hook-state exists and is EMPTY; the SCoRe OUTPUT_DIR
(~/.hermes/mnemosyne/data/score_traces) is absent; positives/negatives/corrections
have 0 records; the qdrant collection score_traces does not exist; seven hooks are
installed and none writes a guard log.

Decide between: (a) write the producer — a PostToolUse hook appending Bash
success/failure records, or (b) delete the consumers. Do not split the difference.
Weigh: is the SCoRe fine-tuning path otherwise complete enough to be worth feeding?
Read mlops/finetune/ before deciding. Report the evidence, then the call.`,
  },
]

// ── Phase 1: investigate + design ────────────────────────────────────────────
phase('Investigate')
log(`Investigating ${DEFECTS.length} defects against live code`)

const designs = await parallel(DEFECTS.map(d => () =>
  agent(
    `${GROUND}

${RULES}

You are working in the Loci repo at ${REPO}. Defect: ${d.title}

${d.brief}

Verify every measurement above against the live code and data before you design
anything. If a measurement does not reproduce, that is the most important thing
you can report — say so and stop rather than designing on a false premise.

Then produce a design: root cause, the specific files to change, and a test plan
where each test FAILS without the fix. Do not write the implementation.

Set recommend_implement=false, with why_not, if the honest answer is that this
should not be built, or is blocked on something absent from the repo.`,
    { label: `design:${d.id}`, phase: 'Investigate', schema: DESIGN_SCHEMA,
      model: 'opus', effort: 'high' }
  ).then(r => r ? { ...r, _defect: d } : null)
))

const toBuild = designs.filter(Boolean).filter(x => x.recommend_implement)
const declined = designs.filter(Boolean).filter(x => !x.recommend_implement)
log(`${toBuild.length} to implement, ${declined.length} declined`)
for (const d of declined) log(`  declined ${d.defect}: ${(d.why_not || '').slice(0, 120)}`)

if (!toBuild.length) {
  return { implemented: [], declined: declined.map(d => ({ defect: d.defect, why_not: d.why_not })) }
}

// ── Phase 2+3: implement in isolation, then try to refute ────────────────────
const results = await pipeline(
  toBuild,
  d => agent(
    `${GROUND}

${RULES}

Implement this design in the Loci repo. You are in your OWN git worktree — the
other implementers are editing the same files in theirs, so do not worry about
their changes, and do not try to coordinate.

DEFECT: ${d.defect}
ROOT CAUSE: ${d.root_cause}
DESIGN: ${d.design}
FILES: ${(d.files_to_change || []).join(', ')}
TEST PLAN: ${d.test_plan}

Requirements:
  - Write the tests. Then REINTRODUCE the bug and confirm they fail, restore and
    confirm they pass. Report exactly what the failure said in sabotage_result.
    If a test passes with the bug present, it is not a test — fix it.
  - Run the relevant suite (mcp/ or scripts/) with ./mcp/.venv/bin/python -m pytest
    or python3 -m pytest. Report the counts in suite_result.
  - ruff check with --select E9,F401,F811,F821,F841 --ignore E402 must be clean.
  - If you add an @mcp.tool(), the callgraph tests pin the tool count and
    .vulture_whitelist.py needs the name — both will fail CI otherwise.
  - Return the full diff (git diff). Do NOT commit, push, or open a PR.

Set complete=false with blocked_reason if you cannot finish honestly.`,
    { label: `impl:${d._defect.id}`, phase: 'Implement', schema: IMPL_SCHEMA,
      model: 'opus', effort: 'high', isolation: 'worktree' }
  ),
  (impl, orig) => {
    if (!impl || !impl.complete) return { impl, verdict: null, defect: orig.defect }
    return agent(
      `${RULES}

Adversarially review this implementation. Your job is to REFUTE it, not to approve
it. Default to refuted=true when uncertain.

DEFECT: ${impl.defect}
SABOTAGE RESULT CLAIMED: ${impl.sabotage_result}
SUITE RESULT CLAIMED: ${impl.suite_result || 'not reported'}

DIFF:
${(impl.diff || '').slice(0, 30000)}

Attack it on these specifically:
  - Does the fix actually address the root cause, or only the symptom that was
    reported? Fixing one call site when a shared helper is the real site is the
    failure mode here.
  - Are the tests real? A test that passes with the bug reintroduced proves
    nothing. Does the claimed sabotage failure match the defect, or is it an
    incidental error like AttributeError on a newly-added name?
  - Is any new behaviour fail-open in a way that hides its own failure? This
    codebase has repeatedly shipped features that reported success while doing
    nothing.
  - Does it change a tested contract that was deliberate?
  - Would it write records, files or log volume that nobody consumes?

safe_to_propose=false unless you would defend it yourself.`,
      { label: `refute:${orig._defect.id}`, phase: 'Refute', schema: VERDICT_SCHEMA,
        model: 'opus', effort: 'xhigh' }
    ).then(v => ({ impl, verdict: v, defect: orig.defect }))
  }
)

const clean = results.filter(Boolean).filter(r => r.verdict && r.verdict.safe_to_propose && !r.verdict.refuted)
const rejected = results.filter(Boolean).filter(r => !r.verdict || r.verdict.refuted || !r.verdict.safe_to_propose)
log(`${clean.length} survived refutation, ${rejected.length} rejected`)

return {
  survived: clean.map(r => ({
    defect: r.defect,
    diff: r.impl.diff,
    tests_added: r.impl.tests_added,
    sabotage_result: r.impl.sabotage_result,
    suite_result: r.impl.suite_result,
  })),
  rejected: rejected.map(r => ({
    defect: r.defect,
    reason: r.verdict ? r.verdict.strongest_objection : (r.impl && r.impl.blocked_reason) || 'no verdict',
  })),
  declined: declined.map(d => ({ defect: d.defect, why_not: d.why_not })),
}
