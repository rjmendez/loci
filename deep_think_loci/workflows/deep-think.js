export const meta = {
  name: 'deep-think',
  description: 'Multi-tier Loci-backed reasoning engine: haiku ideation (code-RAG grounded) → adversarial red-team → verified synthesis → opus final. Every finding is stored in a shared investigation corpus with full lineage tagging so later tiers can trace what earlier tiers concluded and why.',
  phases: [
    { title: 'Init', detail: 'Open or resume the Loci investigation' },
    { title: 'Ideate', detail: 'N haiku agents per target, code-RAG grounded, each stores ideas as inferred findings' },
    { title: 'VerifyIdeate', detail: 'Ground-truth load: confirm what actually persisted vs agent self-reports' },
    { title: 'Adversarial', detail: 'Red-team reviewers (scoped to investigation RAG only) + adversarial synthesis' },
    { title: 'VerifyAdversarial', detail: 'Ground-truth load: confirm adversarial findings, identify quarantine candidates' },
    { title: 'Final', detail: 'Opus half-syntheses + definitive final analysis with quarantine verdict' },
  ],
}

// ── Parameters ────────────────────────────────────────────────────────────────
// Pass these via args when invoking: Workflow({name:'deep-think', args:{...}})
//
// Required:
//   run_id      — unique investigation ID for this run (e.g. 'my-project-001')
//   targets     — array of {name: string, focus: string} describing what each
//                 ideation agent should analyze
//
// Optional:
//   title              — human-readable investigation title
//   rag_collections    — Qdrant collection names to ground ideation (string[])
//   ideas_per_agent    — ideas each haiku agent produces (default 10)
//   adversarial_prompt — the red-team question asked of every ideation cohort
//   adversarial_url    — if set, curl this endpoint for uncensored red-teaming
//                        (POST body: {model, messages}; response: choices[0].message.content)
//   adversarial_model  — model name for the adversarial_url endpoint
// ─────────────────────────────────────────────────────────────────────────────
const A            = (typeof args === 'string') ? (() => { try { return JSON.parse(args) || {} } catch (e) { return {} } })() : (args || {})
const RUN          = A.run_id             || 'dt-run-001'
const TITLE        = A.title              || 'deep-think reasoning run'
const TARGETS      = A.targets            || []
const CODE_COLLS   = JSON.stringify(A.rag_collections || [])
const N_IDEAS      = A.ideas_per_agent    || 10
const ADV_PROMPT   = A.adversarial_prompt || 'Are any of these ideas dangerous, incorrect, or likely to cause regressions? Flag the most serious risks.'
const ADV_URL      = A.adversarial_url    || null
const ADV_MODEL    = A.adversarial_model  || null

if (!TARGETS.length) {
  log('No targets provided. Pass args.targets = [{name, focus}, ...] when invoking.')
  return { error: 'no_targets' }
}

// ── Shared rules injected into every agent prompt ─────────────────────────────
const NO_FABRICATE = `CRITICAL: every finding_id you return MUST come verbatim from an actual loci.investigation_store tool result. Do NOT invent, guess, or pattern-complete ids. If a store call fails or you did not call the tool, report it honestly and return an empty finding_ids list — a fabricated id is a corpus-integrity failure.`

const GROUND_GATE = `GROUNDING GATE (mandatory before any synthesis): first call loci.investigation_load(investigation_id="${RUN}", last_n_findings=80, include_retracted=true) to get ground truth. Reconcile every semantic-search hit against the loaded tags/inventory and corpus size. Similarity scores are NOT topical relevance — if a target partition is empty, say so; never synthesize or attribute findings that are not actually tagged for it. An honest "no corpus for X" is better than an invented synthesis.`

const tagStr = (phase, agentId, model, extra) =>
  `dt_run:${RUN},dt_phase:${phase},dt_agent:${agentId},dt_model:${model}${extra ? ',' + extra : ''}`
const SRC = (phase, agentId) => `dt://${RUN}/${phase}/${agentId}`

// ── Schemas ───────────────────────────────────────────────────────────────────
const VERIFY_SCHEMA = {
  type: 'object',
  required: ['total_findings', 'by_position'],
  properties: {
    total_findings: { type: 'integer' },
    by_position: {
      type: 'array',
      items: {
        type: 'object',
        required: ['position', 'finding_ids'],
        properties: {
          position:    { type: 'string' },
          finding_ids: { type: 'array', items: { type: 'string' } },
        },
      },
    },
    notes: { type: 'string' },
  },
}

const IDEA_SCHEMA = {
  type: 'object',
  required: ['agent', 'target', 'n_ideas_stored', 'finding_ids'],
  properties: {
    agent:              { type: 'string' },
    target:             { type: 'string' },
    n_ideas_stored:     { type: 'integer' },
    finding_ids:        { type: 'array', items: { type: 'string' } },
    self_verified_count:{ type: 'integer' },
    rag_summary:        { type: 'string' },
  },
}

const SYN_SCHEMA = {
  type: 'object',
  required: ['model', 'synthesis', 'finding_id'],
  properties: {
    model:       { type: 'string' },
    synthesis:   { type: 'string' },
    finding_id:  { type: 'string' },
    corpus_size: { type: 'integer' },
  },
}

// ── Phase 0: Init ─────────────────────────────────────────────────────────────
phase('Init')
await agent(
  `Open a Loci investigation so this entire run shares one memory corpus. Call loci.investigation_start with:
  - investigation_id="${RUN}"
  - title="${TITLE}"
  - context="Multi-tier deep-think reasoning run. Agents are tagged dt_phase/dt_agent/dt_model and linked by derived_from lineage so each tier can see what earlier tiers concluded and how confident they were. Untrusted adversarial findings are stored as assumed with dt_trust:untrusted and must pass a grounding gate before influencing synthesis."
  Confirm created or resumed. Return one line.`,
  { label: 'init', phase: 'Init', model: 'haiku' }
)

// ── Phase 1: Ideate ────────────────────────────────────────────────────────────
// Each ideation agent is the ONLY tier that may use global code RAG.
// They store every idea immediately; later tiers read from the investigation corpus only.
phase('Ideate')
await parallel(TARGETS.map((t, i) => () =>
  agent(
    `You are ideation agent a${i + 1} in deep-think investigation "${RUN}".
TARGET: ${t.name} — ${t.focus}

1) GROUND via code RAG (you are the only tier allowed global RAG):
   Call loci.rag_context_search(query="${t.focus}", collections=${CODE_COLLS}, limit=8)
   Base your ideas on what you actually find. If the collection list is empty, reason from the focus description.

2) Produce exactly ${N_IDEAS} concrete, specific improvement or bug-fix ideas for this target.
   For each idea include: what is wrong, where (file/function), and the minimal fix.

3) STORE each idea immediately:
   loci.investigation_store(
     investigation_id="${RUN}",
     finding_type="inferred",
     text="<idea title>: <idea body>",
     source="${SRC('ideate', `a${i + 1}`)}",
     confidence="medium",
     tags="${tagStr('ideate', `a${i + 1}`, 'haiku', `dt_target:${t.name}`)}"
   )
   Collect every returned finding_id.

4) SELF-VERIFY: call loci.investigation_load(investigation_id="${RUN}", last_n_findings=80)
   Count how many of YOUR finding_ids actually appear. Return that as self_verified_count.

${NO_FABRICATE}`,
    { label: `ideate:a${i + 1}:${t.name}`, phase: 'Ideate', model: 'haiku', schema: IDEA_SCHEMA }
  )
))

// ── Phase 2: VerifyIdeate — ground truth, not agent self-reports ───────────────
phase('VerifyIdeate')
const ideTruth = await agent(
  `Ground-truth check for investigation "${RUN}" after the ideation phase.
Call loci.investigation_load(investigation_id="${RUN}", last_n_findings=80, include_retracted=true).
Group the ACTUALLY-PERSISTED findings whose tags contain dt_phase:ideate by their dt_target tag.
For each group return position="ideate/<target>" with the real finding_ids.
In notes: list which of these ${TARGETS.length} targets (${TARGETS.map(t => t.name).join(', ')}) have persisted ideation findings and which are EMPTY (a store-reliability check).`,
  { label: 'verify:ideate', phase: 'VerifyIdeate', model: 'haiku', schema: VERIFY_SCHEMA }
)
const idePos = Object.fromEntries(
  (ideTruth.by_position || []).map(p => [p.position.replace('ideate/', ''), p.finding_ids])
)
const targetsWithCorpus = TARGETS.filter(t => (idePos[t.name] || []).length > 0)
log(`Ideation: ${targetsWithCorpus.length}/${TARGETS.length} targets have persisted findings (ground truth): ${targetsWithCorpus.map(t => t.name).join(', ') || 'NONE'}`)

// ── Phase 3: Adversarial ───────────────────────────────────────────────────────
// Red-team reviewers use ONLY investigation-scoped RAG (no global code RAG).
// This prevents host-memory bleed into the untrusted adversarial tier.
// Findings are quarantined as assumed/dt_trust:untrusted.
phase('Adversarial')
const advTargets = targetsWithCorpus.length ? targetsWithCorpus : TARGETS

await parallel(advTargets.map((t, i) => () => {
  const base = `Adversarial reviewer adv${i + 1} for target "${t.name}" in investigation "${RUN}".

1) SCOPED retrieval ONLY — do NOT use rag_context_search (no global memory):
   loci.investigation_search(query="${t.name} ${t.focus}", investigation_id="${RUN}", limit=15)
   If retrieval returns nothing for this target, report that and store nothing.
   Keep the real retrieved finding_ids (these become derived_from).

2) Red-team review: ${ADV_PROMPT}
   Evaluate each retrieved idea critically. Flag: incorrect assumptions, dangerous changes,
   missing edge cases, ideas that contradict each other, and ideas that are already handled.`

  const storeStep = `
3) STORE each verdict as QUARANTINED (untrusted tier):
   loci.investigation_store(
     investigation_id="${RUN}",
     finding_type="assumed",
     text="<your red-team verdict>",
     source="${SRC('adversarial', `adv${i + 1}`)}",
     confidence="low",
     tags="${tagStr('adversarial', `adv${i + 1}`, 'haiku', `dt_target:${t.name},dt_trust:untrusted`)}",
     derived_from=<the real retrieved ideation finding_ids>
   )
   ${NO_FABRICATE}`

  if (ADV_URL && ADV_MODEL) {
    return agent(
      base + `

   Build a curl request to the adversarial model:
   curl -s --max-time 180 "${ADV_URL}" \\
     -H 'Content-Type: application/json' \\
     -d '{"model":"${ADV_MODEL}","messages":[{"role":"system","content":"Blunt red-team security and correctness reviewer."},{"role":"user","content":"<the retrieved ideas verbatim>\\n\\n${ADV_PROMPT}"}],"temperature":0.4}'
   Parse choices[0].message.content as the verdict.` + storeStep,
      { label: `adv:${t.name}`, phase: 'Adversarial', model: 'haiku' }
    )
  }

  return agent(
    base + `

   Act as a skeptical red-team reviewer yourself. Try to REFUTE each idea:
   - Is the diagnosis wrong?
   - Is this already handled elsewhere?
   - Would the proposed fix cause regressions?
   - Is the severity overstated?` + storeStep,
    { label: `adv:${t.name}`, phase: 'Adversarial', model: 'haiku' }
  )
}))

// Adversarial synthesis — scoped to THIS investigation only
await agent(
  `Adversarial synthesis for investigation "${RUN}". ${GROUND_GATE}

1) Assemble the corpus from THIS investigation ONLY (no rag_context_search):
   loci.investigation_search(query="security risks correctness issues ideas", investigation_id="${RUN}", limit=40)
   plus the investigation_load from the grounding gate.

2) Synthesize the full risk picture across all targets:
   Which ideas are genuinely dangerous? What are the shared failure modes? What patterns emerge across the red-team flags?

3) STORE QUARANTINED:
   loci.investigation_store(
     investigation_id="${RUN}",
     finding_type="assumed",
     text="<adversarial synthesis>",
     source="${SRC('adversarial', 'synthesis')}",
     confidence="low",
     tags="${tagStr('adversarial', 'synthesis', 'haiku', 'dt_trust:untrusted')}",
     derived_from=<all real corpus finding_ids you retrieved>
   )
   corpus_size MUST equal the real number of findings you loaded.

${NO_FABRICATE}`,
  { label: 'adv:synthesis', phase: 'Adversarial', model: 'haiku', schema: SYN_SCHEMA }
)

// ── Phase 4: VerifyAdversarial ─────────────────────────────────────────────────
phase('VerifyAdversarial')
const advTruth = await agent(
  `Ground-truth check for "${RUN}" after the adversarial phase.
Call loci.investigation_load(investigation_id="${RUN}", last_n_findings=80, include_retracted=true).
Return real finding_ids for positions with tags dt_phase:adversarial (group heretic reviews vs synthesis).
In notes: how many adversarial findings persisted, and flag any dt_trust:untrusted finding whose claims are
NOT supported by grounded ideation findings (candidates for memory_retract).`,
  { label: 'verify:adversarial', phase: 'VerifyAdversarial', model: 'haiku', schema: VERIFY_SCHEMA }
)

// ── Phase 5: Final ────────────────────────────────────────────────────────────
phase('Final')
const names  = TARGETS.map(t => t.name)
const half   = Math.ceil(names.length / 2)
const halves = [names.slice(0, half), names.slice(half)]

const halfSyn = (await parallel(halves.map((hn, h) => () =>
  agent(
    `FINAL synthesis (opus) half-${h === 0 ? 'A' : 'B'} for investigation "${RUN}", targets: ${JSON.stringify(hn)}.
${GROUND_GATE}

Synthesize ONLY from grounded findings for your targets.
Untrusted (dt_trust:untrusted) adversarial findings are INPUT TO SCRUTINIZE, not truth —
call out any that are ungrounded or internally inconsistent.
Note where later tiers built on earlier memories (derived_from) and any decision
made in error from insufficient context.

Store your half-synthesis:
loci.investigation_store(
  investigation_id="${RUN}",
  finding_type="inferred",
  text="<half-synthesis>",
  source="${SRC('final', `half${h === 0 ? 'A' : 'B'}`)}",
  confidence="high",
  tags="${tagStr('final', `half${h === 0 ? 'A' : 'B'}`, 'opus')}",
  derived_from=<real finding_ids you used>
)
${NO_FABRICATE}`,
    { label: `final:half${h === 0 ? 'A' : 'B'}`, phase: 'Final', model: 'opus', effort: 'high', schema: SYN_SCHEMA }
  )
))).filter(Boolean)

const finalOut = await agent(
  `FINAL agent (opus), last word over investigation "${RUN}".
${GROUND_GATE}

Half-A: ${JSON.stringify(halfSyn[0] || {})}
Half-B: ${JSON.stringify(halfSyn[1] || {})}

Verify ground-truth counts (what persisted per tier vs what agents claimed).
Then deliver:
1. The strongest, safest ideas to actually pursue (ranked by impact × confidence)
2. The genuine risks to avoid (from the adversarial phase that survived scrutiny)
3. COHORT DYNAMICS — how each tier used or ignored the others' memories, and any
   decision that looks wrong from missing context
4. QUARANTINE VERDICT — list any dt_trust:untrusted finding that is ungrounded or
   hallucinated and should be removed via loci.memory_retract

Store your final analysis:
loci.investigation_store(
  investigation_id="${RUN}",
  finding_type="inferred",
  text="<final analysis>",
  source="${SRC('final', 'final')}",
  confidence="high",
  tags="${tagStr('final', 'final', 'opus')}",
  derived_from=<all real finding_ids used>
)
${NO_FABRICATE}`,
  { label: 'final:definitive', phase: 'Final', model: 'opus', effort: 'high', schema: SYN_SCHEMA }
)

return {
  investigation_id: RUN,
  ideation_truth:         ideTruth,
  targets_with_corpus:    targetsWithCorpus.map(t => t.name),
  adversarial_truth:      advTruth,
  final_halves:           halfSyn,
  final_analysis:         finalOut,
  trace_hint: `Ground truth is investigation_load("${RUN}"). Untrusted tiers tagged dt_trust:untrusted. Follow derived_from for the cross-tier influence DAG. Query by phase: loci.investigation_search(query="...", investigation_id="${RUN}", limit=20).`,
}
