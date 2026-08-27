export const meta = {
  name: 'loci-hister-grounding',
  description: 'Design, gate and build a Hister-backed external-documentation grounding source for Loci retrieval',
  whenToUse: 'Loci grounds only against findings and the operator own repos. This evaluates whether an external doc corpus earns its keep, and builds it only if the measurement says yes.',
  phases: [
    { title: 'Recon',    detail: 'Stand Hister up, read its MCP surface, and measure the baseline it would have to beat' },
    { title: 'Design',   detail: 'Two independent integration designs, then a red-team pick' },
    { title: 'Gate',     detail: 'Does the measured baseline justify building at all' },
    { title: 'Build',    detail: 'Isolated worktrees, tests that fail without the change' },
    { title: 'Refute',   detail: 'Adversarial pass before anything is proposed as a PR' },
  ],
}

const GROUND = (typeof args === 'object' && args && args.ground) || ''
const REPO = (typeof args === 'object' && args && args.repo) || '.'

const RULES = `
DISCIPLINE — every one of these cost this project real time, most of them today:

1. MEASURE, do not assert. Lead with the number and how you got it. Mark inference
   as inference and say what would falsify it.
2. RUN it. py_compile and ruff pass code that raises NameError at runtime. And a
   probe returning nothing may be testing the wrong endpoint with stale code — a
   "27B produces empty output" result today was entirely that.
3. AVAILABILITY IS NOT CAPABILITY. llm_available() returned True for weeks while
   every call returned None: provider chosen on key-presence, key had a newline
   from an 80-column paste, and behind that the account had no credit. Three
   failures, one fail-open. Check a thing WORKS, not that it is configured.
4. ZERO OUTPUT IS NOT AUTOMATICALLY A DEFECT — check whether the producer ever ran.
5. A LANE WITH NO READER IS NOT WORTH FEEDING. finding_verifications.jsonl had 0
   readers; the audit lane had 667 writes and 0 reads. Establish the consumer first.
6. HERMETIC TESTS. A test that reads ~/.loci/backends.toml asserts against the
   operator machine and passes locally while failing in CI. Two tests did exactly
   that today.
7. Comment economy: rationale in the commit message, not a wall of source comments.
`

const RECON_SCHEMA = {
  type: 'object',
  properties: {
    topic: { type: 'string' },
    findings: { type: 'string' },
    numbers: { type: 'string' },
    blockers: { type: 'array', items: { type: 'string' } },
    confidence: { type: 'string' },
  },
  required: ['topic', 'findings', 'numbers', 'blockers'],
}

const DESIGN_SCHEMA = {
  type: 'object',
  properties: {
    approach: { type: 'string' },
    integration_point: { type: 'string' },
    files_to_change: { type: 'array', items: { type: 'string' } },
    failure_modes: { type: 'string' },
    test_plan: { type: 'string' },
    reversibility: { type: 'string' },
    cost: { type: 'string' },
  },
  required: ['approach', 'integration_point', 'files_to_change', 'failure_modes',
             'test_plan', 'reversibility'],
}

const GATE_SCHEMA = {
  type: 'object',
  properties: {
    build_it: { type: 'boolean' },
    reasoning: { type: 'string' },
    baseline_evidence: { type: 'string' },
    chosen_design: { type: 'string' },
    scope_cuts: { type: 'array', items: { type: 'string' } },
  },
  required: ['build_it', 'reasoning', 'baseline_evidence'],
}

const IMPL_SCHEMA = {
  type: 'object',
  properties: {
    diff: { type: 'string' },
    tests_added: { type: 'array', items: { type: 'string' } },
    sabotage_result: { type: 'string' },
    suite_result: { type: 'string' },
    measured_effect: { type: 'string' },
    complete: { type: 'boolean' },
    blocked_reason: { type: 'string' },
  },
  required: ['diff', 'tests_added', 'sabotage_result', 'complete'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    refuted: { type: 'boolean' },
    strongest_objection: { type: 'string' },
    safe_to_propose: { type: 'boolean' },
  },
  required: ['refuted', 'strongest_objection', 'safe_to_propose'],
}

// ── Phase 1: recon, in parallel ──────────────────────────────────────────────
phase('Recon')

const RECON = [
  {
    id: 'hister-surface',
    prompt: `Stand up Hister and document what it actually offers.

Download the prebuilt binary (v0.18.0, hister_0.18.0_linux_amd64, ~110MB) from
github.com/asciimoo/hister releases into a scratch dir under /tmp. Do NOT install
system-wide, do NOT add it to the Loci repo.

Establish, by RUNNING it:
  - what MCP tools it exposes (read server/mcp.go from the repo, and if you can
    start it, list the tools live). Names, parameters, return shapes.
  - how the embeddings endpoint is configured. Loci already runs nomic-embed-text
    at http://10.42.0.1:11434 (93ms). Can Hister use it, and at what dimension?
  - how documents are indexed (URL fetch? file watch? CLI?) and where state lives.
  - startup/auth requirements and resource footprint.

Blockers matter more than features here. If it cannot run headless, or needs a
browser extension to be useful, say so plainly.`,
  },
  {
    id: 'baseline',
    prompt: `Measure the baseline Hister would have to beat. This is the number the
whole decision rests on, so be careful with it.

In ${REPO}:
  - Sample real findings and run verify.verify_finding over them WITH grounding
    (investigation_id set). Record the distribution of verdict and degraded, and
    how often the model's own reasoning says the context was INSUFFICIENT rather
    than that the claim was wrong. That last number is the one an external doc
    corpus would move.
  - Separately, count how many stored findings make claims about THIRD-PARTY
    systems (qdrant, ollama, huggingface, a model card, a library) versus claims
    about this operator's own code. Only the former can be helped by external docs.

Use ./mcp/.venv/bin/python; scripts/loci_groom.py load_env() sets QDRANT_URL —
without it RAG silently returns nothing and you will measure your own harness.
Report exact counts and how you obtained them.`,
  },
  {
    id: 'retrieval-seam',
    prompt: `Find where an extra grounding source would have to plug in, and what
that costs.

In ${REPO}, read mcp/server.py's rag_context_search (collections default
["loci_memory","agent_core_chunks"]), the _rag_* helper chain, and
mcp/verify.py's _lazy_rag. Establish:
  - the exact seam where a THIRD source could be merged, and whether the
    cross-encoder re-pass would rank across heterogeneous sources sensibly.
  - whether an MCP-to-MCP call is even possible from inside Loci's server, or
    whether the merge has to happen in the agent instead. This is decisive for
    the design — check, do not assume.
  - what breaks if the extra source is unavailable. Loci's house style is
    fail-open; confirm the seam preserves that.
  - who reads the result: get real call counts from ~/.hermes/logs/tool-audit.log.`,
  },
]

log(`Recon: ${RECON.length} parallel probes`)
const recon = (await parallel(RECON.map(r => () =>
  agent(`${GROUND}\n${RULES}\nRepo: ${REPO}\n\nTOPIC: ${r.id}\n\n${r.prompt}`,
    { label: `recon:${r.id}`, phase: 'Recon', schema: RECON_SCHEMA,
      model: 'opus', effort: 'high' })
))).filter(Boolean)

log(`${recon.length}/${RECON.length} recon probes returned`)
for (const r of recon) log(`  ${r.topic}: ${(r.blockers || []).length} blocker(s)`)

// ── Phase 2: two independent designs ─────────────────────────────────────────
phase('Design')
const designs = (await parallel([
  {
    id: 'in-server',
    slant: `Design the integration INSIDE the Loci MCP server: Hister becomes a third
retrieval source that rag_context_search merges, so every existing consumer gets
it for free with no agent changes. Argue for it honestly, including what it costs
in coupling and what happens when Hister is down.`,
  },
  {
    id: 'sibling-mcp',
    slant: `Design the integration as a SIBLING MCP server: Hister runs alongside Loci,
agents call it directly, and Loci changes little or nothing. Argue for it honestly,
including what is lost when retrieval is not merged and ranked together.`,
  },
].map(d => () =>
  agent(`${GROUND}\n${RULES}\nRepo: ${REPO}

RECON FINDINGS:
${JSON.stringify(recon, null, 2).slice(0, 14000)}

${d.slant}

Ground the design in the recon above — if recon says an MCP-to-MCP call is
impossible from inside the server, a design that requires one is dead and you
should say so rather than writing it. Give a test plan where each test FAILS
without the change, and state how to REVERSE it.`,
    { label: `design:${d.id}`, phase: 'Design', schema: DESIGN_SCHEMA,
      model: 'opus', effort: 'high' })
))).filter(Boolean)

// ── Phase 3: the gate ────────────────────────────────────────────────────────
phase('Gate')
const gate = await agent(
  `${GROUND}\n${RULES}

RECON: ${JSON.stringify(recon, null, 2).slice(0, 14000)}

DESIGNS: ${JSON.stringify(designs, null, 2).slice(0, 14000)}

Decide whether this should be built AT ALL, then pick a design.

The bar: the baseline recon must show that a meaningful share of grounding
failures are "insufficient external context" rather than "the claim is wrong" or
"the corpus is thin". If most findings are about the operator's OWN code, an
external documentation corpus cannot help them and this is a second index to
back up for nothing.

build_it=false is a legitimate and useful answer. This project has twice declined
to build something for good measured reasons and been better for it. Do not build
to look productive.

If build_it=true, name the chosen design and list scope_cuts — what NOT to build
in the first PR.`,
  { label: 'gate', phase: 'Gate', schema: GATE_SCHEMA, model: 'opus', effort: 'xhigh' }
)

if (!gate || !gate.build_it) {
  log(`GATE: not building — ${(gate && gate.reasoning || 'no gate verdict').slice(0, 200)}`)
  return { recon, designs, gate, built: null }
}
log(`GATE: build — ${gate.chosen_design}`)

// ── Phase 4+5: build, then try to break it ───────────────────────────────────
const built = await pipeline(
  [gate],
  g => agent(
    `${GROUND}\n${RULES}\nRepo: ${REPO}

Implement the chosen design. You are in your OWN git worktree.

CHOSEN: ${g.chosen_design}
REASONING: ${g.reasoning}
DO NOT BUILD (deferred): ${(g.scope_cuts || []).join('; ') || 'nothing deferred'}

DESIGNS FOR REFERENCE:
${JSON.stringify(designs, null, 2).slice(0, 12000)}

Requirements:
  - Tests must FAIL without the change. Reintroduce the absence, watch them fail,
    restore. Report the exact failure text in sabotage_result.
  - HERMETIC: no test may read ~/.loci/backends.toml or reach the network. Two
    tests did exactly that today and passed locally while being meaningless.
  - Fail-open: if Hister is unreachable, retrieval must degrade to what it does
    today, not error. Test that explicitly.
  - Run the relevant suite (./mcp/.venv/bin/python -m pytest mcp/ -q) and report
    counts. ruff check with --select E9,F401,F811,F821,F841 --ignore E402 clean.
  - If you add an @mcp.tool(), the callgraph tests pin the tool count and
    .vulture_whitelist.py needs the name — both fail CI otherwise.
  - measured_effect: show the BEFORE and AFTER of the baseline number from recon.
    If you cannot measure an improvement, say so — that is a real result.
  - Return the full diff. Do NOT commit, push, or open a PR.`,
    { label: 'build', phase: 'Build', schema: IMPL_SCHEMA,
      model: 'opus', effort: 'high', isolation: 'worktree' }
  ),
  (impl) => {
    if (!impl || !impl.complete) return { impl, verdict: null }
    return agent(
      `${RULES}

Adversarially review this implementation. REFUTE it. Default refuted=true when unsure.

SABOTAGE CLAIMED: ${impl.sabotage_result}
SUITE CLAIMED: ${impl.suite_result || 'not reported'}
MEASURED EFFECT CLAIMED: ${impl.measured_effect || 'not reported'}

DIFF:
${(impl.diff || '').slice(0, 30000)}

Attack specifically:
  - Is the measured_effect real, or measured on a harness rather than the system?
    A grounding improvement measured without QDRANT_URL set is measuring nothing.
  - Does it fail OPEN when Hister is down, and is that actually tested?
  - Any test that reads operator config or touches the network is not hermetic.
  - Does the sabotage failure match the defect, or is it an incidental error like
    an AttributeError on a newly-added name?
  - Does this add a second index nobody will back up, for a benefit nobody reads?
  - Is the fix at the shared seam, or at one call site that happened to show it?

safe_to_propose=false unless you would defend it yourself.`,
      { label: 'refute', phase: 'Refute', schema: VERDICT_SCHEMA,
        model: 'opus', effort: 'xhigh' }
    ).then(v => ({ impl, verdict: v }))
  }
)

const r0 = (built || []).filter(Boolean)[0] || {}
return {
  recon, designs, gate,
  built: r0.impl || null,
  verdict: r0.verdict || null,
  ready_for_pr: !!(r0.verdict && r0.verdict.safe_to_propose && !r0.verdict.refuted),
}
