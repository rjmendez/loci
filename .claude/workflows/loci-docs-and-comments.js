export const meta = {
  name: 'loci-docs-and-comments',
  description: 'Bring the docs back in line with 60 commits of drift, and cut narration comments without losing measured rationale',
  whenToUse: 'Docs last written 2026-08-10; HEAD is 2026-08-27 with 60 commits and +35,645 lines since. 94 files carry 1,522 lines of multi-line comment prose.',
  phases: [
    { title: 'Docs',    detail: 'One agent per document; correct what is now FALSE first' },
    { title: 'Comments', detail: 'One agent per file cluster; narration out, measured rationale kept' },
    { title: 'Refute',  detail: 'Adversarial pass — did anything load-bearing get deleted, is any doc claim wrong' },
  ],
}

const GROUND = (typeof args === 'object' && args && args.ground) || ''

const RULES = `
DISCIPLINE — each of these cost this project real time:

1. MEASURE, do not assert. Lead with the number and how you got it.
2. VERIFY AGAINST LIVE CODE. A doc sentence is a claim; check it before you keep
   it. grep the symbol, read the function, run the thing.
3. A NEGATIVE RESULT IS A RESULT. "This section is already correct" is a fine
   answer. Do not manufacture churn to look productive.
4. NEVER invent behaviour. If you cannot establish what something does, say so
   in blockers rather than describing what it probably does.
5. Do not commit, push, or open a PR. Return a diff.
`

const COMMENT_POLICY = `
THE RULE — this is the operator's standing directive of 2026-08-17, recovered
from Loci, and it is stricter than a general style preference:

  Comments must be short and sparse — ONE LINE MAX — and only for a genuinely
  non-obvious WHY: a hidden constraint, a subtle invariant, a workaround for a
  specific bug. NEVER multi-paragraph blocks. NEVER narrative or historical
  explanation ("this used to be", "the operator corrected", dated commentary).
  NEVER restating what the code already says. Rationale, history and design
  justification belong in commit messages and PR descriptions — which are
  already the durable record — not living permanently in source files.

So the default for a multi-line block is DELETE. There is no "keep the good
long ones" exception. A block survives only by being compressed to one line.

BUT NOTHING MEASURED MAY BE LOST. Some blocks carry real work: a table of
results across seeds, a swept constant, a note on what did NOT replicate. The
canonical case is the 28-line block above the reranker passage-chars constant in
mcp/qdrant_ops.py — three seeds, the value it replaces, and an explicit
non-replication. Under the directive that does NOT belong in source. Under this
project's discipline it must not be destroyed either.

So RELOCATE, do not discard:
  - Put the full original text of any block carrying measurements, tables, or a
    non-obvious design justification into 'relocated' — verbatim, with the file
    and the symbol it sat above. It goes into this change's PR description,
    which is the durable record the directive points at.
  - Then delete it from source, leaving at most ONE line: e.g.
      # 1024 chars: swept against identity recall; 512 scored worse than no CE.
  - If a block is pure narration, just delete it. Nothing to relocate.

ALWAYS DELETE:
  - Comments restating the next line, or the docstring above them.
  - Changelog in source: "added in #217", "previously X", any dated note.
  - Hedging: "note that", "it is worth mentioning", "this is a bit tricky".
  - Comments explaining Python itself or an obvious idiom.

KEEP AS ONE LINE (compress, do not expand):
  - A fail-open or silent-failure trap someone will otherwise reintroduce.
  - An ordering, concurrency or security property ("compare_digest on both
    branches: == leaks length").
  - Why the non-obvious option was chosen.
  - In tests: the defect the test guards, in one line. It is read at the moment
    the test fails. Compress the paragraph; do not delete the fact.

OUT OF SCOPE — do not touch:
  - Docstrings. Any of them, however long.
  - Section banners (# ---- Foo ----). Structural, not prose.
  - Any executable line. This change moves comments only.

WHEN IN DOUBT: compress to one line rather than delete outright, and list it in
kept_but_borderline. A deleted measurement not captured in 'relocated' is
unrecoverable; that is the only unforgivable outcome here.
`

const DOC_SCHEMA = {
  type: 'object',
  properties: {
    doc: { type: 'string' },
    diff: { type: 'string' },
    false_claims_found: { type: 'array', items: { type: 'string' } },
    gaps_filled: { type: 'array', items: { type: 'string' } },
    verified_against: { type: 'array', items: { type: 'string' } },
    blockers: { type: 'array', items: { type: 'string' } },
    complete: { type: 'boolean' },
  },
  required: ['doc', 'diff', 'false_claims_found', 'verified_against', 'complete'],
}

const COMMENT_SCHEMA = {
  type: 'object',
  properties: {
    cluster: { type: 'string' },
    diff: { type: 'string' },
    lines_removed: { type: 'number' },
    kept_but_borderline: { type: 'array', items: { type: 'string' } },
    relocated: { type: 'array', items: { type: 'string' } },
    suite_result: { type: 'string' },
    complete: { type: 'boolean' },
  },
  required: ['cluster', 'diff', 'lines_removed', 'relocated', 'suite_result', 'complete'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    item: { type: 'string' },
    refuted: { type: 'boolean' },
    strongest_objection: { type: 'string' },
    load_bearing_deletions: { type: 'array', items: { type: 'string' } },
    safe_to_propose: { type: 'boolean' },
  },
  required: ['item', 'refuted', 'strongest_objection', 'safe_to_propose'],
}

// ── what actually changed, so the doc agents are not guessing ────────────────
const RECENT = `
Landed since the docs were last written (2026-08-10), most recent first:
  #231 eval/verify_skeptic_eval.py — fixed benchmark for the adversarial skeptic.
       Measured 22% false-refutation on main; five prompt/guard variants all
       neutral or worse, so NO behaviour changed. Verify is NOT fit to schedule.
  #230 groom "summaries": empty / fully-retracted investigations are
       nothing_to_say, not errors.
  #229 findings.jsonl is a MIXED log — 3,681 of 6,610 records (55.7%) are
       text-less access rows. _only_findings() drops them. New pass_summaries
       drives the summary ladder; corpus went 6 -> 137 of 142 summaries.
  #228 the grounding hooks were inert: pre_llm_grounding read extra.user_message
       while Claude Code sends a top-level "prompt"; every named-vector Qdrant
       search sent {"dense": v} instead of {"name":..,"vector":..} and 400'd;
       session_end_sync looked Claude Code UUIDs up in a Hermes state.db and had
       never synced a session. scripts/hooks/install.sh --check reports drift.
  #226 rag: timeout, a shadowed expander, a docstring naming the wrong corpus.
  #222/#224/#225 generation resolves separately from embeddings; the vLLM
       fallback is opt-in (LOCI_VLLM_FALLBACK) and default OFF.
  #221 grounding gate corroborates the semantic lane instead of trusting 0.55.
  #220 one definition of the quantized-search params.
  #218/#212 causal_infer and derived_from lineage edges.
  #214 groom schedules investigation_verify_all and the reflection loop.
  #213 memcheck reports the audit lane's state instead of a bare count.
  #211/#206 bearer auth on the HTTP transports; loopback bind by default.
  #204 the retention default was deleting the index on every server start
       (30 days -> 0). This one changes what OPERATIONS should say about startup.

Live scheduled work (crontab), all verified through their real entrypoint:
  index every 6h (coverage 0.9476), knn_tags 03:20, codelink 03:40,
  summaries 04:50. verify is NOT scheduled and should not be described as if it is.
`

phase('Docs')

const DOCS = [
  { id: 'README',
    files: 'README.md',
    brief: `The front door. It is 300 lines and was last touched 2026-08-10.

Establish first what it CLAIMS and whether each claim is still true — a wrong
README is worse than a thin one. Check the install/run instructions actually
work as written, the tool list against the real @mcp.tool registrations in
mcp/server.py, and any configuration it names against what the code reads.` },

  { id: 'ARCHITECTURE+CONCEPTS',
    files: 'docs/ARCHITECTURE.md, docs/CONCEPTS.md',
    brief: `ARCHITECTURE (340 lines, 2026-08-09) and CONCEPTS (225 lines, 2026-08-07).

The storage story moved: the code graph is now graph.ladybug (LadybugStore, not
Kuzu naming), findings.jsonl is a mixed append log whose access rows are not
findings, and loci_memory/mnemosyne/loci_sessions all use a NAMED "dense"
vector while agent_core_chunks is unnamed. Check every diagram and every claim
about what is stored where against mcp/qdrant_ops.py, mcp/inv_store.py and
mcp/graph/ladybug_store.py.` },

  { id: 'COMPONENTS',
    files: 'docs/COMPONENTS.md',
    brief: `348 lines, last touched 2026-08-07 — the most stale document here.

It enumerates components, so the failure mode is components that no longer exist,
have been renamed, or have gained/lost responsibilities. Reconcile the list
against the actual modules under mcp/ and scripts/. Anything you cannot find in
the tree is either renamed (say what to) or gone (remove it).` },

  { id: 'OPERATIONS+DEPLOYMENT',
    files: 'docs/OPERATIONS.md, docs/DEPLOYMENT.md',
    brief: `OPERATIONS (193 lines) and DEPLOYMENT (246 lines), both 2026-08-10.

Highest risk of being actively WRONG, because they tell an operator what to run.
Reconcile against scripts/loci_groom.py PASSES and the real crontab shape
described above. Specifically: the retention default is now 0 (#204) and the
startup no longer purges; four groom passes are scheduled and "verify" is NOT
one of them; the hooks are installed via scripts/hooks/install.sh with a --check
mode that reports drift; bearer auth and loopback binding changed how the HTTP
transports come up (#206/#211).` },
]

log(`${DOCS.length} documents, ${DOCS.length + 5} agents planned`)

const docResults = await parallel(DOCS.map(d => () =>
  agent(`${GROUND}\n${RULES}\n${RECENT}

DOCUMENT: ${d.id}   (${d.files})

${d.brief}

Method, in this order:
  1. Read the document.
  2. For every factual claim it makes, verify it against the live tree. List the
     ones that are now FALSE in false_claims_found — that is the primary output.
  3. Correct those first. Then fill genuine gaps from RECENT above.
  4. Cite what you verified against in verified_against (file:line or command).

Write like the rest of this repo: direct, concrete, no marketing. Prefer a
measured number to an adjective. Do not pad, do not add a "Recent changes"
section — fold the truth into where it belongs. If a section is already correct,
leave it and say so.

Return the full unified diff.`,
    { label: `docs:${d.id}`, phase: 'Docs', schema: DOC_SCHEMA,
      model: 'opus', effort: 'high', isolation: 'worktree' })
)).then(r => r.filter(Boolean))

log(`${docResults.length}/${DOCS.length} documents returned`)
for (const d of docResults) {
  log(`  ${d.doc}: ${(d.false_claims_found || []).length} false claim(s) corrected`)
}

// ── Comments ────────────────────────────────────────────────────────────────
phase('Comments')

const CLUSTERS = [
  { id: 'server',
    files: 'mcp/server.py',
    note: '301 prose lines across 59 blocks — the single largest target. It also holds real security rationale (the compare_digest note) and fail-open warnings. Read every block; this is judgment, not a sweep.' },
  { id: 'graph',
    files: 'mcp/graph/ladybug_store.py, mcp/graph/code_parse.py, mcp/graph/linker.py',
    note: '131 prose lines. ladybug_store carries the leasing and checkpoint semantics — a mid-checkpoint death leaves a corrupt WAL that blocks BOTH read and write opens, and the lease PID is diagnostics only while fcntl.flock is the real lock. Those are keepers.' },
  { id: 'qdrant+inv',
    files: 'mcp/qdrant_ops.py, mcp/investigation_tools.py, mcp/llm_local.py',
    note: '177 prose lines. CONTAINS THE CANONICAL RELOCATE: the 28-line reranker passage-chars table in qdrant_ops.py — capture it verbatim in relocated, then cut it to one line. llm_local.py is 20.7% comments, the densest file in the repo, and much of it is narration.' },
  { id: 'scripts',
    files: 'scripts/loci_groom.py, scripts/hooks/pre_llm_grounding.py, scripts/hooks/session_end_sync.py, scripts/a2a_context_bridge.py, scripts/generate_memory_md.py, scripts/callgraph/resolve.py, scripts/amem_consolidation.py',
    note: '~180 prose lines. loci_groom carries why each pass refuses rather than writing junk — keep those, they are the difference between a pass that fails closed and one that fills a lane with noise.' },
  { id: 'tests',
    files: 'scripts/callgraph/tests/, scripts/hooks/tests/, mcp/tests/, eval/tests/',
    note: '327 prose lines, 21% of the total. A comment naming the defect a test guards is read at the moment it fails — COMPRESS those to one line, do not delete the fact. Delete narration outright ("# arrange", "# now call the function", restating the assert).' },
]

const commentResults = await pipeline(
  CLUSTERS,
  c => agent(`${GROUND}\n${RULES}\n${COMMENT_POLICY}

CLUSTER: ${c.id}
FILES: ${c.files}

${c.note}

Method:
  - Read each comment block and decide: narration (delete) or rationale (keep).
  - Delete whole blocks where they are narration. Where a block is half and half,
    keep the load-bearing sentences and drop the rest — but do not reword what
    survives.
  - Docstrings are NOT in scope. Leave every docstring alone.
  - Section banners are NOT in scope. Leave them.
  - After editing, run the relevant suite and report counts:
      ./mcp/.venv/bin/python -m pytest mcp/ -q          (for mcp/)
      python3 -m pytest scripts/tests scripts/hooks/tests -q
      python3 -m pytest scripts/callgraph/tests -q
    NOTE: mcp/tests/test_graph_integration.py::test_code_graph_ingest_and_query
    fails on clean main for host reasons — it is NOT yours.
  - ruff check --select E9,F401,F811,F821,F841 --ignore E402 must be clean.
  - Report lines_removed, and list anything you nearly deleted in
    kept_but_borderline.

Return the full unified diff.`,
    { label: `cut:${c.id}`, phase: 'Comments', schema: COMMENT_SCHEMA,
      model: 'opus', effort: 'high', isolation: 'worktree' }),

  (impl, orig) => {
    if (!impl || !impl.complete) return { impl, verdict: null, item: orig.id }
    return agent(`${RULES}\n${COMMENT_POLICY}

Adversarially review this comment purge. REFUTE it. Default refuted=true when unsure.

CLUSTER: ${orig.id}
LINES REMOVED: ${impl.lines_removed}
SUITE: ${impl.suite_result || 'not reported'}
BORDERLINE KEPT: ${JSON.stringify(impl.kept_but_borderline || [])}
RELOCATED (must cover every measured block the diff deletes):
${(impl.relocated || []).join('\n---\n').slice(0, 8000)}

DIFF:
${(impl.diff || '').slice(0, 30000)}

Attack specifically:
  - Did it delete a MEASURED rationale — numbers, a table, a seed count, a
    "what did not replicate" — WITHOUT capturing it verbatim in 'relocated'?
    That is the one unforgivable outcome: unrecoverable without redoing the
    work. Cross-check every such deletion in the diff against the relocated
    list and report misses in load_bearing_deletions.
  - Conversely: did it KEEP a multi-paragraph block? The directive is one line
    max. A surviving paragraph is a failure to do the job, not caution.
  - Did it delete a warning about a failure mode: fail-open behaviour, a silent
    failure, an ordering or concurrency requirement, a security property?
  - Did it delete a test comment naming the defect that test guards?
  - Did it touch a docstring or a section banner? Both are out of scope.
  - Did it REWORD rather than delete? The diff should read as deletion.
  - Did it change any executable line? Nothing but comments should move.

safe_to_propose=false unless you would defend every deletion in it.`,
      { label: `refute:${orig.id}`, phase: 'Refute', schema: VERDICT_SCHEMA,
        model: 'opus', effort: 'xhigh' }).then(v => ({ impl, verdict: v, item: orig.id }))
  }
)

const all = (commentResults || []).filter(Boolean)
const ready = all.filter(r => r.verdict && r.verdict.safe_to_propose && !r.verdict.refuted)
const rejected = all.filter(r => r.verdict && (r.verdict.refuted || !r.verdict.safe_to_propose))
const removed = ready.reduce((n, r) => n + (r.impl.lines_removed || 0), 0)
log(`comments: ${ready.length} clusters ready (${removed} lines), ${rejected.length} rejected`)

return {
  docs: docResults.map(d => ({
    doc: d.doc, diff: d.diff,
    false_claims: d.false_claims_found, gaps: d.gaps_filled,
    verified_against: d.verified_against, blockers: d.blockers,
  })),
  comments_ready: ready.map(r => ({
    cluster: r.item, diff: r.impl.diff, lines_removed: r.impl.lines_removed,
    borderline: r.impl.kept_but_borderline, relocated: r.impl.relocated,
    suite: r.impl.suite_result,
  })),
  comments_rejected: rejected.map(r => ({
    cluster: r.item, objection: r.verdict.strongest_objection,
    load_bearing_deletions: r.verdict.load_bearing_deletions,
  })),
}
