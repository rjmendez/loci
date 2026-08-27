export const meta = {
  name: 'loci-finish-everything',
  description: 'Execute the remaining Loci work in parallel: multi-repo code grounding, the two specified-but-unexecuted declines, and a post-fix re-measurement',
  whenToUse: 'After #224. Each item is already measured and scoped; this executes them concurrently in isolated worktrees.',
  phases: [
    { title: 'Build',  detail: 'One agent per work item, isolated worktree, tests that fail without the change' },
    { title: 'Refute', detail: 'Adversarial pass per item before anything is proposed as a PR' },
  ],
}

const GROUND = (typeof args === 'object' && args && args.ground) || ''
const REPO = (typeof args === 'object' && args && args.repo) || '.'

const RULES = `
DISCIPLINE — every one of these cost this project real time today:

1. MEASURE, do not assert. Lead with the number and how you got it.
2. RUN it. py_compile and ruff pass code that raises NameError at runtime.
3. TEST ON THE RIGHT TREE. Twice today a fix was declared broken because it was
   tested on a branch that lacked its dependency. Confirm the code under test is
   actually present before concluding anything.
4. HERMETIC TESTS. Three tests today were non-hermetic: one reached the network,
   one read ~/.loci/backends.toml, one patched sys.modules where the code reads a
   package ATTRIBUTE (passed alone, failed in the suite). Patch the layer the
   code actually reads.
5. A LANE WITH NO READER IS NOT WORTH FEEDING. Establish the consumer first.
6. ZERO OUTPUT IS NOT AUTOMATICALLY A DEFECT — check whether the producer ran.
7. A NEGATIVE RESULT IS A RESULT. If the measurement says the change does not
   help, say so and stop. Two builds were correctly declined today.
8. Comment economy: rationale in the commit message, not a wall of source comments.
`

const IMPL_SCHEMA = {
  type: 'object',
  properties: {
    item: { type: 'string' },
    diff: { type: 'string' },
    tests_added: { type: 'array', items: { type: 'string' } },
    sabotage_result: { type: 'string' },
    suite_result: { type: 'string' },
    measured_effect: { type: 'string' },
    complete: { type: 'boolean' },
    abandoned: { type: 'boolean' },
    blocked_reason: { type: 'string' },
  },
  required: ['item', 'diff', 'tests_added', 'sabotage_result', 'complete'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    item: { type: 'string' },
    refuted: { type: 'boolean' },
    strongest_objection: { type: 'string' },
    safe_to_propose: { type: 'boolean' },
  },
  required: ['item', 'refuted', 'strongest_objection', 'safe_to_propose'],
}

const ITEMS = [
  {
    id: 'multi-repo-code-grounding',
    brief: `verify.py's code grounding has NEVER produced a line of source.

Measured over 100 grounded verifications: 32 findings carried file:line refs, 59
refs parsed, 59/59 rejected by _safe_resolve, 0/100 had any source fetched. And
31 of 91 non-degraded verifications explicitly ASKED for the operator's own
source — this is the largest measured grounding gap.

Cause: _safe_resolve sandboxes to _repo_root(), the LOCI checkout, but
loci_memory is a multi-repo corpus. Measured on the live corpus: 62 code_refs,
33 unique, 0 resolvable — yet ALL 33 basenames exist on this host, under
/home/rjmendez/hugbot5000, /home/rjmendez/hb-mountfix-v2 and similar. They are in
$HOME, NOT under ~/development.

THE HARD PART, and why this is not just a wider allow-list: the host carries MANY
WORKTREES of the same repo (hugbot5000-wt-audio-ant-decommission,
hugbot5000-wt-fix-347-respin, and more). The same relative path exists in all of
them. "First root that has the file" would ground the verifier on STALE CODE from
an arbitrary worktree, which is worse than grounding on nothing — a confident
answer from the wrong revision.

code_refs carry {path, hash} — the hash may let you pick the right checkout.
Investigate whether it does before designing around it.

Preserve the security property exactly: absolute paths and .. traversal must
still be refused, and a resolved path must still land under an allowed root.
Widening an allow-list is not removing it. There is a partial patch on branch
fix/multi-repo-code-grounding (a _search_roots() helper scanning ~/development
siblings) — it is INSUFFICIENT because it has the wrong parent and no worktree
disambiguation. Treat it as a starting point or discard it.

Report the hit rate BEFORE and AFTER over all 62 refs. If you cannot beat 0
without risking a stale-revision read, say so and abandon — that is a real result.`,
  },
  {
    id: 'guard-log-consumer-deletion',
    brief: `Execute the deletion that issue #216's investigation specified.

guard_bash_failures.log / guard_bash_successes.log / guard_tool_reflections.log
have FOUR production consumers and ZERO producers; the only writers in the repo
are two test files that create their own fixture. The investigation measured that
the SCoRe pipeline these feed cannot run here — a registered runner never started
across 8 nightly attempts, Ollama for it was unreachable, and the tuned model it
would produce has no caller. It recommended option (b), DELETE THE CONSUMERS,
with a fail-first test plan, and explicitly did not split the difference.

Consumers to remove or de-wire:
  scripts/score_trace_collector.py     - SCoRe dataset collection
  scripts/skill_annotation_updater.py  - reads guard_bash_failures optionally
  mlops/memory/live_evo.py             - _load_guard_failures, confidence penalties
  mlops/loop.py                        - passes hook_state_dir through to Live-Evo

Before deleting ANY of it, establish what else imports each module — mlops/loop.py
in particular may have callers unrelated to guard logs, and Live-Evo may do useful
work beyond the guard-failure path. Removing a module that something else imports
is worse than the dead lane. If a file has non-guard responsibilities, remove the
guard PATH and keep the rest.

Also check .github/workflows and cron/jobs.json for anything referencing these,
and mcp/tests + mlops/*/tests for tests that only exist to test the dead path.`,
  },
  {
    id: 'self-check-phase-0',
    brief: `Execute the Phase 0 that issue #193's investigation proposed.

The audit lane has 667 writes and 0 reads: over 5.2 days and 15,048 tool calls,
findings were stored 667 times and EVERY consumer of a receipt —
memory_self_check, investigation_pre_answer_check, investigation_evidence_precheck,
investigation_finding_provenance, investigation_verify_all, verify_finding — ran
ZERO times. The investigation declined to build the receipt producer and proposed
instead: "make the check RUN first (~30 lines into session_end_sync.py or
investigation_reflect) and see whether its output changes anyone's behaviour."

Build that. Constraints, all of which matter:
  - memory_self_check is ADVISORY. It must never block or slow a session end.
    Fail-open, bounded, and it must not add meaningful latency to a Stop hook.
  - #213 added _audit_lane_status so a report says "no receipts exist" rather
    than "1,682 unsupported findings". Whatever surfaces the output must carry
    that distinction, or it will read as 1,682 real defects.
  - It must be OFF by default or trivially disabled. Turning on an unattended
    check that nobody asked for is how this project got its dead lanes.
  - The output needs a READER. Do not write to a new file nothing opens — that
    is the exact failure this issue is about. Prefer surfacing where someone
    already looks; investigation_load is called 15x/5.2d, memory_self_check 0.

If you conclude the honest Phase 0 is smaller than 30 lines — a log line, or a
counter — build that instead and say why.`,
  },
  {
    id: 'verify-quality-remeasure',
    brief: `MEASUREMENT ONLY — produce no diff unless the measurement demands one.

A baseline over 100 grounded verifications found the verifier is not fit to
schedule: refutations judged sound 7/53 = 13.2%, verdict contradicts its own
reasoning in >=5/53, and 9/100 degraded. The degradations were traced to a 4096
context ceiling: vLLM max_model_len=4096, usable prompt 3712 tokens, HTTP 400
"prompt contains at least 3713 input tokens". All 9 degraded had claim+context
>= 11,587 chars.

That ceiling was dama-vllm's forwarder, reached through llm_local's automatic
fallback. PR #224 made that fallback OPT-IN (LOCI_VLLM_FALLBACK, default off), so
Ollama failures no longer reroute there.

Re-measure now, on the same shape: sample grounded verifications and report the
degraded rate and the verdict distribution. The question is narrow — did removing
the 4096 reroute eliminate the degraded cases, and did anything else move?

Do NOT re-litigate the soundness rate; 13.2% was hand-read and is not something to
re-derive cheaply. Report whether the DEGRADED count changed and whether the
verify groom pass is now safe to schedule. "Still not safe" is the expected and
acceptable answer.

The oxalis endpoint has been intermittently saturated — it returned nothing in
85s for a 4-token generation at one point. If you cannot get a clean measurement,
say so rather than reporting a number taken during contention.`,
  },
]

phase('Build')
log(`Executing ${ITEMS.length} work items in parallel`)

const results = await pipeline(
  ITEMS,
  it => agent(
    `${GROUND}\n${RULES}\nRepo: ${REPO}\n\nWORK ITEM: ${it.id}\n\n${it.brief}\n
You are in your OWN git worktree; the other agents are editing this repo in
theirs. Do not coordinate, do not commit, do not push, do not open a PR.

Requirements:
  - Tests must FAIL without the change. Reintroduce the absence, watch them fail,
    restore. Report the exact failure text.
  - Run the relevant suite (./mcp/.venv/bin/python -m pytest mcp/ -q for mcp,
    python3 -m pytest scripts/tests -q for scripts) and report counts. Note that
    test_graph_integration::test_code_graph_ingest_and_query fails on clean main
    for host-specific reasons — it is NOT yours.
  - ruff check --select E9,F401,F811,F821,F841 --ignore E402 must be clean.
  - Set abandoned=true with blocked_reason if the measurement says the change
    does not help. That is a respected outcome here, not a failure.
  - Return the full diff.`,
    { label: `build:${it.id}`, phase: 'Build', schema: IMPL_SCHEMA,
      model: 'opus', effort: 'high', isolation: 'worktree' }
  ),
  (impl, orig) => {
    if (!impl || !impl.complete || impl.abandoned) {
      return { impl, verdict: null, item: orig.id }
    }
    return agent(
      `${RULES}\n
Adversarially review this implementation. REFUTE it. Default refuted=true when unsure.

ITEM: ${orig.id}
SABOTAGE CLAIMED: ${impl.sabotage_result}
SUITE CLAIMED: ${impl.suite_result || 'not reported'}
EFFECT CLAIMED: ${impl.measured_effect || 'not reported'}

DIFF:
${(impl.diff || '').slice(0, 30000)}

Attack specifically:
  - Is the measured effect real, or measured on a harness? A grounding number
    taken without QDRANT_URL set is measuring nothing.
  - For code grounding: could it read a STALE worktree? That is a confident
    answer from the wrong revision — worse than no grounding. And does the
    sandbox still refuse absolute paths and .. traversal?
  - For deletions: does anything still import what was removed? Check, do not
    assume the author did.
  - For anything scheduled or hooked: does it add latency to a hot path, and
    does its output have a reader?
  - Is any test non-hermetic — network, operator config, or patching a layer the
    code does not read?
  - Does the sabotage failure match the defect, or is it incidental?

safe_to_propose=false unless you would defend it yourself.`,
      { label: `refute:${orig.id}`, phase: 'Refute', schema: VERDICT_SCHEMA,
        model: 'opus', effort: 'xhigh' }
    ).then(v => ({ impl, verdict: v, item: orig.id }))
  }
)

const all = (results || []).filter(Boolean)
const ready = all.filter(r => r.verdict && r.verdict.safe_to_propose && !r.verdict.refuted)
const abandoned = all.filter(r => r.impl && r.impl.abandoned)
const rejected = all.filter(r => r.verdict && (r.verdict.refuted || !r.verdict.safe_to_propose))

log(`${ready.length} ready for PR, ${rejected.length} rejected, ${abandoned.length} abandoned`)

return {
  ready: ready.map(r => ({ item: r.item, diff: r.impl.diff, tests: r.impl.tests_added,
                           sabotage: r.impl.sabotage_result, effect: r.impl.measured_effect })),
  rejected: rejected.map(r => ({ item: r.item, objection: r.verdict.strongest_objection })),
  abandoned: abandoned.map(r => ({ item: r.item, reason: r.impl.blocked_reason,
                                   effect: r.impl.measured_effect })),
}
