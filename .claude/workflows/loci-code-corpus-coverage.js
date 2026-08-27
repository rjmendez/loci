export const meta = {
  name: 'loci-code-corpus-coverage',
  description: 'Make Loci retrieval actually cover the repos its findings cite: audit the code corpus, index the missing repos, and stop payload filters failing open',
  whenToUse: 'The measured demand is for the operator OWN SOURCE (31/91), not documentation (2/91). agent_core_chunks indexes agent-mesh, not the repos the corpus talks about.',
  phases: [
    { title: 'Audit',  detail: 'What is really in the code corpus, who maintains it, and does anything filter on the missing index' },
    { title: 'Build',  detail: 'Close the coverage gap; isolated worktrees' },
    { title: 'Refute', detail: 'Adversarial pass before anything is proposed' },
  ],
}

const GROUND = (typeof args === 'object' && args && args.ground) || ''
const REPO = (typeof args === 'object' && args && args.repo) || '.'

const RULES = `
DISCIPLINE — each of these cost this project real time today:

1. ASSERT ON THE PAYLOAD, NOT THE COUNT. A Qdrant count with a filter on an
   UNINDEXED field fails OPEN and returns the whole collection. Four probes today
   returned an identical 3,030,551 and looked like four confirmations; they were
   one filter being ignored. Scroll and inspect actual values.
2. MEASURE, do not assert. Lead with the number and how you got it.
3. RUN it. py_compile and ruff pass code that raises NameError at runtime.
4. TEST ON THE RIGHT TREE. Twice today a fix was called broken because it was
   tested on a branch missing its dependency.
5. HERMETIC TESTS — no network, no ~/.loci/backends.toml, and patch the layer the
   code actually reads (a sys.modules patch does not affect 'from pkg import mod').
6. A LANE WITH NO READER IS NOT WORTH FEEDING. Establish the consumer first.
7. A NEGATIVE RESULT IS A RESULT. Two builds were correctly declined today.
8. Comment economy: rationale in the commit message, not the source.
`

const AUDIT_SCHEMA = {
  type: 'object',
  properties: {
    topic: { type: 'string' },
    findings: { type: 'string' },
    numbers: { type: 'string' },
    blockers: { type: 'array', items: { type: 'string' } },
    recommend: { type: 'string' },
  },
  required: ['topic', 'findings', 'numbers', 'blockers', 'recommend'],
}

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

phase('Audit')

const AUDITS = [
  {
    id: 'corpus-truth',
    prompt: `Establish what agent_core_chunks actually IS, and whether it is maintained.

Measured already, re-verify: 12,000 sampled points, only 298 carried a file_path
(2.5%), zero contained "hugbot", and the top path roots were a2a, deployment,
scouts, dispatch, mas-research — i.e. agent-mesh, not the repos Loci findings
cite. The collection holds ~6M points and is NOT quantized (hermes_memory is the
only quantized collection on the instance).

Answer, with evidence:
  - What are the 97.5% of points that have no file_path? Sample their payloads.
    If most of the collection is not code, "agent_core_chunks is Loci's code
    index" is false and the whole framing changes.
  - WHO WRITES IT? Find the producer. Is it in this repo, another repo, or a
    process nobody owns? When did it last write — check a timestamp field.
  - Does anything in Loci FILTER it by file_path or repo? If yes, those filters
    are failing open right now (no payload index) and that is a correctness bug,
    not a performance one. If nothing filters it, the missing index is latent.
  - What does rag_context_search actually retrieve from it in practice? Run a
    real query about a hugbot subsystem and report what comes back.`,
  },
  {
    id: 'ingest-path',
    prompt: `Find the cheapest correct way to get the cited repos into retrieval.

Loci findings cite /home/rjmendez/hugbot5000 and /home/rjmendez/hb-mountfix-v2.
Neither is in agent_core_chunks.

Two candidate mechanisms already exist — evaluate BOTH before recommending:
  a) .claude/workflows/loci-codebase-ingest.js — recovered today, takes args.repo,
     reads source and stores structured module docs into a Loci investigation via
     investigation_store. Verified working: 8 modules, idempotent (supersedes the
     prior doc per module). But it writes FINDINGS, which land in hermes_memory —
     consider whether flooding the findings corpus with generated module docs is
     acceptable, and what it does to retrieval for real findings.
  b) whatever populates agent_core_chunks (find it in the audit) — if it is a
     reusable indexer, pointing it at another repo may be a config change.

Also establish the SCALE: how many source files are in hugbot5000, and what would
indexing cost in embedding calls at 93ms each against 10.42.0.1. A number, not a
guess. If it is tens of thousands of files, a naive full index is not viable and
the recommendation must be scoped (changed files? cited files only? one subsystem?).

The host has MANY hugbot5000-wt-* worktrees. Indexing several revisions of the
same file would poison retrieval with near-duplicates. Say how to avoid that.`,
  },
]

log(`Auditing ${AUDITS.length} questions before building anything`)
const audit = (await parallel(AUDITS.map(a => () =>
  agent(`${GROUND}\n${RULES}\nRepo: ${REPO}\n\nTOPIC: ${a.id}\n\n${a.prompt}`,
    { label: `audit:${a.id}`, phase: 'Audit', schema: AUDIT_SCHEMA,
      model: 'opus', effort: 'high' })
))).filter(Boolean)

log(`${audit.length}/${AUDITS.length} audits returned`)
for (const a of audit) log(`  ${a.topic}: ${(a.blockers || []).length} blocker(s)`)

// ── Build ────────────────────────────────────────────────────────────────────
const ITEMS = [
  {
    id: 'payload-index-file-path',
    brief: `Stop payload filters on the code corpus failing open.

A Qdrant filter on an UNINDEXED payload field does not error — it returns the
whole collection. Measured today: four different MatchText probes on file_path
each returned an identical 3,030,551, which reads as four confirmations and is
one filter being ignored. Anything that filters agent_core_chunks by file_path or
repo is silently unfiltered right now.

Loci already has _create_payload_indexes in mcp/qdrant_ops.py and applies it to
its own collection. Establish from the audit whether Loci actually filters this
collection. If it does, add the index and a test that a filtered count differs
from an unfiltered one — that is the assertion that catches a filter failing
open, and it must FAIL without the index.

If nothing in Loci filters it, say so and set abandoned=true: adding an index to
a 6M-point collection Loci does not own, for a filter nobody issues, is not an
improvement. The audit answers this — do not guess.`,
  },
  {
    id: 'index-cited-repos',
    brief: `Close the coverage gap for the repos Loci findings actually cite, using
whichever mechanism the ingest-path audit recommends.

Constraints that decide the shape:
  - SCOPE IT. Do not full-index a repo of unknown size on a first pass. The audit
    reports the file count and the embedding cost; use it. Cited-files-first is a
    defensible scope: 33 unique paths are referenced by real findings, and those
    are the ones with measured demand.
  - NO NEAR-DUPLICATES. The host carries many hugbot5000-wt-* worktrees of the
    same files. Index one canonical revision, not several.
  - DO NOT POLLUTE hermes_memory with generated content unless the audit shows
    that is genuinely fine. Retrieval quality for real findings is the thing being
    protected here.
  - The result must have a READER. Establish which retrieval path will actually
    surface it before writing anything — rag_context_search's collections list is
    hardcoded to ["hermes_memory","agent_core_chunks"], so a new collection that
    nothing queries is another dead lane.

Report a BEFORE and AFTER: run a grounded query about a hugbot subsystem and show
what retrieval returns in each case. If you cannot demonstrate an improvement,
set abandoned=true — that is a real result and this project has accepted it twice
today.`,
  },
]

phase('Build')
const results = await pipeline(
  ITEMS,
  it => agent(
    `${GROUND}\n${RULES}\nRepo: ${REPO}

AUDIT FINDINGS:
${JSON.stringify(audit, null, 2).slice(0, 16000)}

WORK ITEM: ${it.id}

${it.brief}

You are in your OWN git worktree. Do not commit, push, or open a PR.
Tests must FAIL without the change — reintroduce the absence, watch them fail,
restore, and report the exact failure text. Run the relevant suite and report
counts; test_graph_integration::test_code_graph_ingest_and_query fails on clean
main for host-specific reasons and is NOT yours. ruff clean with
--select E9,F401,F811,F821,F841 --ignore E402. Return the full diff.`,
    { label: `build:${it.id}`, phase: 'Build', schema: IMPL_SCHEMA,
      model: 'opus', effort: 'high', isolation: 'worktree' }
  ),
  (impl, orig) => {
    if (!impl || !impl.complete || impl.abandoned) return { impl, verdict: null, item: orig.id }
    return agent(
      `${RULES}\n
Adversarially review this. REFUTE it. Default refuted=true when unsure.

ITEM: ${orig.id}
SABOTAGE CLAIMED: ${impl.sabotage_result}
SUITE CLAIMED: ${impl.suite_result || 'not reported'}
EFFECT CLAIMED: ${impl.measured_effect || 'not reported'}

DIFF:
${(impl.diff || '').slice(0, 30000)}

Attack specifically:
  - Is the measured effect real, or an artefact? A retrieval improvement measured
    without QDRANT_URL set is measuring nothing. A count taken with a filter on an
    unindexed field is measuring nothing.
  - Does it write generated content into hermes_memory, and what does that do to
    retrieval for REAL findings? Dilution is the risk nobody notices until later.
  - Could it index several worktree revisions of the same file?
  - Does the new content have a reader, or is rag_context_search's hardcoded
    collections list still excluding it?
  - Is any test non-hermetic?
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
log(`${ready.length} ready, ${rejected.length} rejected, ${abandoned.length} abandoned`)

return {
  audit,
  ready: ready.map(r => ({ item: r.item, diff: r.impl.diff, tests: r.impl.tests_added,
                           sabotage: r.impl.sabotage_result, effect: r.impl.measured_effect })),
  rejected: rejected.map(r => ({ item: r.item, objection: r.verdict.strongest_objection })),
  abandoned: abandoned.map(r => ({ item: r.item, reason: r.impl.blocked_reason })),
}
